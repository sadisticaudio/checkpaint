#include "CheckpointProcessor.h"
#include "OldTransformer.h"
#include <ATen/ParallelOpenMP.h>
#include <typeinfo>
#include <numeric>

template <typename T> std::string getPythonClass(const T& x) { return std::string(x.attr("__class__").attr("__name__")); }
#include<unistd.h>
CheckpointProcessor::CheckpointProcessor(const py::list& model_checkpoints) { print("in list ctor"); load_checkpoints(model_checkpoints); }
CheckpointProcessor::CheckpointProcessor(const py::dict& full_run_data) { print("in dict ctor"); 
  torch::NoGradGuard noGradGuard;
  load_config(py::dict(full_run_data["config"].attr("to_dict")()));
  load_checkpoints(py::list(full_run_data["checkpoints"]));// list of OrderedDict of [str, Tensor]
  print("in CP Constructor");
}

void CheckpointProcessor::load_checkpoints(const py::list& state_dicts) {
  hi_res::time_point t1 { hi_res::now() };
  const size_t num_checkpoints { state_dicts.size() };
  auto emb0 = state_dicts[0]["embed.W_E"];
  const auto& emb0tensor = THPVariable_Unpack(emb0.ptr());
  // print("emb0tensorref size(-2), size(-1), pModulo+1", emb0tensor.sizes(), emb0tensor.size(-2), emb0tensor.size(-1), pModulo+1);
  bool oldModel { THPVariable_Unpack(state_dicts[0]["embed.W_E"].ptr()).size(-1) == pModulo + 1 };
  // print("oldModel", oldModel);
  // print("in load_checkpoints");
  models.clear();
  models.reserve(num_checkpoints);
  for (size_t i {}; i < num_checkpoints; ++i) models.push_back(Transformer(cfg, (int)i));
  auto model { models.begin() };

  for (auto sd { state_dicts.begin() }; model != models.end(); ++sd, ++model) {
    py::list names = (*sd).attr("keys")();
    py::list tensors = (*sd).attr("values")();
    auto nameItr = names.begin();
    auto tensorItr = tensors.begin();

    for (; nameItr != names.end(); ++nameItr, ++tensorItr) {
      auto name = std::string(py::str(*nameItr));
      const auto& tensor = THPVariable_Unpack((*tensorItr).ptr());
      if(++sd == state_dicts.end()) {
        if (++nameItr == names.end()) {
          // if (oldModel)  print("about to importOldParameter", name, tensor.sizes());
          // else print("about to importParameter", name, tensor.sizes());
          // if(name.find(".b_") == std::string::npos && name.find("IGNORE") == std::string::npos) 
          //   print("last model of " + std::to_string(state_dicts.size()), name, tensor.sizes(), get_first_elements(tensor, 4));
          auto W_E = (*model)->named_parameters()["embed.W_E"];
          auto spectrum = torch::fft::fft2(W_E.squeeze().index({ Slice(None,-1), Slice() })).abs().sum(-1);
          auto kv = spectrum.index({ Slice(None, spectrum.size(0)/2 + 1) }).topk(5, -1, true, true);
          auto [values, indices] = kv;
          if (values[3].item<float>() < values[2].item<float>()/2.f)
            indices = indices.index({ Slice(0, -2) });
          else if (values[4].item<float>() < values[2].item<float>()/2.f)
            indices = indices.index({ Slice(0, -1) });
          std::vector<int> idx;
          for (int i = 0; i < indices.size(0); ++i) idx.push_back(static_cast<int>(indices[i].item<long>()));
          trig.makeTrigData(idx);
          print("got freqz", idx);
        }
        nameItr--;
      }
      sd--;
      if (oldModel) {
        // print("about to importOldParameter", name, tensor.sizes());
        (*model)->importOldParameter(name, tensor);
      }
      else {
        auto* paramPtr = (*model)->named_parameters().find(name);
        if (paramPtr) {
          // print("about to importParameter", name, tensor.sizes(), "expected shape", (*model)->named_parameters()[name].sizes());
          (*model)->importParameter(name, tensor);
        }
      }
    }
  }
}

std::vector<int> CheckpointProcessor::get_key_freqs() { return trig.key_freqs; }

std::vector<std::vector<int>> CheckpointProcessor::get_shapes(const py::object& pyInput, const std::vector<std::string>& names) {
  const auto& input1 = THPVariable_Unpack(pyInput.ptr());
  const auto input = input1.to(device);

  Transformer model { cfg };
  model->to(input.device());
  auto hook_names { names };
  for (int i {}; i < hook_names.size();) {
    if (hook_names[i].find("hook_") == std::string::npos)
      hook_names.erase(hook_names.begin() + i);
    else ++i;
  }
  print("hook names", hook_names);
  model->set_hooks(hook_names);
  auto logits = model(input);
  auto add_shapes = [&](const std::string& name, const at::Tensor& t) { 
    std::vector<int> szs;
    for (const auto& s : t.sizes())
      szs.push_back(static_cast<int>(s));
    print("adding shape", name, t.sizes(), "szs.size()", szs.size());
    sizeMap[name] = szs;
  };
  for (const auto& x : model->named_parameters()) add_shapes(x.key(), x.value());
  // add_shapes("hook_logits", logits);
  add_shapes("cos_apb", trig.cos_apb.to(device));
  add_shapes("sin_apb", trig.sin_apb.to(device));
  add_shapes("LABELS", all_labels.to(device));
  std::vector<std::vector<int>> shapes;
  for (const auto& x : names) shapes.push_back(sizeMap[x]);// print("x in names", x, "sizeMap[x]", sizeMap[x]);
  return shapes; 
}

at::Tensor CheckpointProcessor::get_metrics() {
  std::vector<std::string> names { "hook_logits", "blocks.0.mlp.hook_post", "blocks.0.hook_resid_mid" };
  hi_res::time_point t1 { hi_res::now() };
  torch::NoGradGuard ngGuard;

  at::Tensor rv, t { all_data };
  t.set_requires_grad(false);

  size_t single_mem_usage { size_t(-1) }, n_ensemble { 1U }, N { 1U }, idx {}, num_checkpoints { models.size() };
  for (; idx < num_checkpoints; idx += N) {
    // try {
        N = std::min(n_ensemble, num_checkpoints - idx);
        Transformer model { cfg, 0, N };
        model->importEnsemble(&models[idx], N);
        model->set_hooks(names);
        model->to(device);
        auto logits = model(t);
        auto cache = model->getActivationCache();
        auto metrics = get_all_metrics(trig.cos_apb.to(device), trig.sin_apb.to(device), all_labels, logits, cache["blocks.0.mlp.W_out"], cache["unembed.W_U"], cache["blocks.0.mlp.hook_post"], cache["blocks.0.hook_resid_mid"]);
        rv = (idx ? torch::cat({ rv, metrics }) : metrics);
        if(!idx) {
          single_mem_usage = getCudaMemUsage();
          n_ensemble = std::min(1UL, std::max(1UL, size_t((getCudaMemTotal()/single_mem_usage) * 0.9)));
          print("increased n_ensemble", n_ensemble);
        }

    // } catch (const std::exception& e) {
    //     std::cerr << "Caught an exception: " << e.what() << std::endl;
    //     cudaError_t error = cudaGetLastError();
    //     if (error != cudaSuccess) {
    //         std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    //     }
    //     if(idx == 0 && N == 1) {
    //       print("not enough memory to get metrics for a single model");
    //       break;
    //     }
    //     N = 0;
    //     n_ensemble--;
    //     print("decreased n_ensemble", n_ensemble);
    // }
    
  }

  // auto tsr = [&](int n, int n1, int n2, int n3, int d) { return nDigits(rv.index({ Slice(), n1 }).transpose(0,1).transpose(0,2)[n2][n][n3].item<float>(), d); };
  // print("\n    train     test        restricted         excluded");
  // for (int i = 0; i < 250; ++i) {
  //   std::cout << nDigits(i,3) << "  " <<  tsr(i,0,0,0,5) << " " << tsr(i,0,0,1,3) << "  ";
  //   std::cout <<                          tsr(i,0,1,0,5) << " " << tsr(i,0,1,1,3) << "  ";
  //   std::cout <<                          tsr(i,1,0,0,5) << " " << tsr(i,1,0,1,3) << "  ";
  //   std::cout <<                          tsr(i,1,1,0,5) << " " << tsr(i,1,1,1,3) << "  ";
  //   std::cout <<                          tsr(i,2,0,0,5) << " " << tsr(i,2,0,1,3) << "  ";
  //   std::cout <<                          tsr(i,2,1,0,5) << " " << tsr(i,2,1,1,3) << "\n";
  // }
  std::cout << "total time = " << std::chrono::duration_cast<std::chrono::microseconds>(hi_res::now() - t1).count()/1000000.0 << "[s]" << std::endl;
  printCudaMemUsage("After all metrics are got");
  return rv;
}

torch::Tensor CheckpointProcessor::get_custom_metric(const std::string& jitPath, std::vector<torch::Tensor> globals, std::vector<std::string> names, torch::Tensor input) {
  const size_t G { globals.size() }, num_checkpoints { models.size() }, n_ensemble { 9 };
  torch::jit::script::Module scriptModule { torch::jit::load(jitPath) };

  auto hook_names { names };
  for (size_t i {}; i < hook_names.size();) {
    if (hook_names[i].find("hook_") == std::string::npos 
    && hook_names[i].find(".") == std::string::npos
    && hook_names[i].find("cos_apb") == std::string::npos
    && hook_names[i].find("sin_apb") == std::string::npos)
      hook_names.erase(hook_names.begin() + i);
    else ++i;
  }
  print("hook_names", hook_names);

  std::vector<c10::IValue> inputs;
  inputs.resize(names.size() + G);
  for (size_t i {}; i < G; ++i) inputs[i] = globals[i];
  for (size_t i { G }; i < inputs.size(); ++i) inputs[i] = at::Tensor();

  torch::NoGradGuard ngGuard;

  at::Tensor all_metrics { torch::empty({ 0 }).to(device) };

  for (int idx { 0 }; idx < num_checkpoints; idx += n_ensemble) {
    auto N { std::min(n_ensemble, num_checkpoints - idx) };
    std::vector<at::Tensor> metrics;
    metrics.reserve(N);
    Transformer model { cfg, 0, N };
    model->importEnsemble(&models[idx], N);
    model->set_hooks(names);
    model->to(device);
    model(input.to(device));
    auto cache = model->getActivationCache();
    if (idx == 0) for (const auto& x : cache.data) print(x.first, x.second.sizes());
    if(contains(names, "cos_apb")) cache.data.push_back({ "cos_apb", trig.cos_apb });
    if(contains(names, "sin_apb")) cache.data.push_back({ "sin_apb", trig.sin_apb });
    // print("num inputs, G, names", inputs.size(), G, names);
    // for (auto x : names) print(x + " ");
    for (int n {}; n < N; ++n) {
      auto is = inputs;
      // print("\nready g?", is.size());
      // print("names", names, "hook_names", hook_names);
      for (int i { 0 }; i < is.size(); ++i) {
        if(i >= G) {
          // print("PRINTING, i, names[i - G]", i, names[i - G], cache[names[i - G]].sizes());
          if(names[i - G] != "cos_apb" && names[i - G] != "sin_apb") {
            if(names[i - G].find("hook_") != std::string::npos) {
              // if(n == 0) print("G", G, "i", i, "found hook_", names[i - G], cache[names[i - G]].sizes());
              is[i] = cache[names[i - G]][n];//.index({ Slice(), Slice(n,n + 1) });
            }
            else {
              // if(n == 0) print("G", G, "i", i, "NOT a hook_", names[i - G], cache[names[i - G]].index({ Slice(n,n + 1) }).sizes());
              is[i] = cache[names[i - G]].index({ Slice(n,n + 1) });
            }
          }
          else {
            is[i] = cache[names[i - G]];
          }
        }
        else {
          print("not printing, i, names[i - G]", i, names[i - G], cache[names[i - G]].sizes());
        }
      }
      auto scriptReturn = scriptModule.forward(is);
      metrics.push_back(scriptReturn.toTensor());
    }
    print("about to cat all_metrics & metrics", all_metrics.sizes(), torch::stack(at::TensorList(metrics)).sizes());
    all_metrics = torch::cat({ all_metrics, torch::stack(at::TensorList(metrics)) });
  }
  return all_metrics;
}

std::map<std::string, torch::Tensor> CheckpointProcessor::get_all_activations(torch::Tensor input) {
  const size_t num_checkpoints { models.size() }, n_ensemble { 9 };
  torch::NoGradGuard ngGuard;
  std::map<std::string, torch::Tensor> fullCache;
  long totalMem = 0;

  for (int idx { 0 }; idx < num_checkpoints; idx += n_ensemble) {
    auto N { std::min(n_ensemble, num_checkpoints - idx) };
    Transformer model { cfg, 0, N };
    model->importEnsemble(&models[idx], N);
    model->set_hooks();
    model->to(device);
    model(input.to(device));
    auto cache = model->getActivationCache();
    for (const auto& [name, tensor] : cache) {
      fullCache[name] = idx == 0 ? tensor : torch::cat({ fullCache[name], tensor });
      totalMem += 4 * tensor.numel();
      if (idx == 0 || idx + n_ensemble >= num_checkpoints)
        print(name, tensor.sizes(), "size in cache", fullCache[name].sizes(), "totalMem", toHumanReadable(totalMem));
    }
    if (idx == n_ensemble) print("\n< - - - " + std::to_string(num_checkpoints - (N + n_ensemble)) + " more batches - - - >\n\n");
  }
  return fullCache;
}

std::map<std::string, torch::Tensor> 
CheckpointProcessor::get_last_activations(torch::Tensor input, std::vector<std::string> hook_names = {}) {
  Transformer model { cfg };
  model->importEnsemble(&models[models.size() - 1], 1);
  model->set_hooks(hook_names);
  model->to(device);
  model(input.to(device));
  auto cache = model->getActivationCache();
  return cache.to_map();
}

// std::map<std::string, torch::Tensor> 
// CheckpointProcessor::get_activations_slice(torch::Tensor input,
//                                             std::vector<std::string> hook_names,
//                                             int dim_idx,
//                                             std::vector<int> cp_indices) {
  
//   if (cp_indices == std::vector<int>{}) {
//     cp_indices = std::vector<int>(models.size());
//     std::iota(cp_indices.begin(), cp_indices.end(), 0);
//   }
//   std::map<std::string, torch::Tensor> slices;
//   for (const auto& name : hook_names) slices[name] = torch::empty({ 0 }).to(device);

//   for (size_t i { 0 }; i < cp_indices.size(); ++i) {
//     Transformer model { cfg };
//     model->importEnsemble(&models[(size_t(cp_indices[i]) + models.size()) % models.size()], 1);
//     model->set_hooks(hook_names);
//     model->to(device);
//     model(input.to(device));
//     auto cache = model->getActivationCache();
//     for (const auto& name : hook_names) {
//       int idx { (dim_idx + int(cache[name].size(-1))) % int(cache[name].size(-1)) };
//       slices[name] = torch::cat({ slices[name], cache[name].index({"...", idx}) });
//     }
//   }
  
//   return slices;
// }

std::map<std::string, torch::Tensor> 
CheckpointProcessor::get_activations_slice(torch::Tensor input,
                                            std::vector<std::string> hook_names,
                                            std::vector<py::object> dim_idx,
                                            std::vector<int> cp_indices) {
  
  std::vector<at::indexing::TensorIndex> t_idx;
  if (dim_idx == std::vector<py::object>{}) {
    t_idx.push_back(at::indexing::TensorIndex(Slice()));
  }
  else {
    t_idx.push_back(at::indexing::TensorIndex("..."));
    for (const auto& idx : dim_idx) {
      if (py::isinstance<py::int_>(idx)) {
        t_idx.push_back(at::indexing::TensorIndex(idx.cast<int>()));
      } else if (py::isinstance<torch::Tensor>(idx)) {
        t_idx.push_back(at::indexing::TensorIndex(idx.cast<torch::Tensor>()));
      } else if (py::isinstance<py::slice>(idx)) {
        t_idx.push_back(at::indexing::TensorIndex(idx.cast<Slice>()));
      }
    }
  }

  if (cp_indices == std::vector<int>{}) {
    cp_indices = std::vector<int>(models.size());
    std::iota(cp_indices.begin(), cp_indices.end(), 0);
  }
  std::map<std::string, torch::Tensor> slices;
  for (const auto& name : hook_names) slices[name] = torch::empty({ 0 }).to(device);

  for (size_t i { 0 }; i < cp_indices.size(); ++i) {
    Transformer model { cfg };
    model->importEnsemble(&models[(size_t(cp_indices[i]) + models.size()) % models.size()], 1);
    model->set_hooks(hook_names);
    model->to(device);
    model(input.to(device));
    auto cache = model->getActivationCache();
    for (const auto& name : hook_names) {
      // int idx { (dim_idx + int(cache[name].size(-1))) % int(cache[name].size(-1)) };
      slices[name] = torch::cat({ slices[name], cache[name].index(t_idx) });
    }
  }
  
  return slices;
}

std::map<std::string, torch::Tensor> 
CheckpointProcessor::get_modified_activations(torch::Tensor input,
                                              int checkpoint_idx,
                                              std::map<std::string, torch::Tensor> weight_dict,
                                              std::vector<std::string> hook_names) {
  Transformer model { cfg };
  int idx { (int(models.size()) + checkpoint_idx) % int(models.size()) };
  model->importEnsemble(&models[idx], 1);
  for (auto [name, tensor] : weight_dict) {
    if (name.find("hook") == std::string::npos) {
      model->importParameter(name, tensor);
    }
    else {
      for (auto& m : model->named_modules()) {
        if (m.key().find("hook") != std::string::npos) {
          auto hook_point = std::dynamic_pointer_cast<HookPointImpl>(m.value());
          if(hook_point->hook_name == name.substr(name.find("hook_") + 5)) {
            hook_point->copier.load(tensor);
            hook_point->hook_fn = hook_point->copier;
          }
        }
      }
    }
  }
  model->set_hooks(hook_names);
  model->to(device);
  model(input.to(device));
  auto cache = model->getActivationCache();
  return cache.to_map();
}

torch::Tensor CheckpointProcessor::get_modified_logits(torch::Tensor input,
                                                      int checkpoint_idx,
                                                      std::map<std::string, torch::Tensor> weight_dict) {
  Transformer model { cfg };
  std::vector<std::string> hook_names;
  int idx { (int(models.size()) + checkpoint_idx) % int(models.size()) };
  model->importEnsemble(&models[idx], 1);
  for (auto [name, tensor] : weight_dict) {
    if (name.find("hook") == std::string::npos) {
      model->importParameter(name, tensor);
    }
    else {
      hook_names.push_back(name);
      for (auto& m : model->named_modules()) {
        if (m.key().find("hook") != std::string::npos) {
          auto hook_point = std::dynamic_pointer_cast<HookPointImpl>(m.value());
          if(hook_point->hook_name == name.substr(name.find("hook_") + 5)) {
            hook_point->copier.load(tensor);
            hook_point->hook_fn = hook_point->copier;
          }
        }
      }
    }
  }
  if (hook_names.size()) model->set_hooks(hook_names);
  model->to(device);
  return model(input.to(device));
}

void CheckpointProcessor::compare_cache(const py::dict& pyCache) {
  auto input = torch::tensor({ { 60L, 90L, 113L } }).to(device), label = torch::tensor({ { 37L } }).to(device);
  print("testing input", input, "label", label);
  Transformer cpModel { cfg, 0 };
  Vec<at::Tensor> logits { { cpModel->forward(input) } };
  auto cache = cpModel->getActivationCache();
  auto names = pyCache.attr("keys")();
  auto ts = pyCache.attr("values")();
  auto namet = names.begin();
  for (auto tt { ts.begin() }; tt != ts.end(); ++tt, ++namet) {
    auto name = std::string(py::str(*namet));
    auto t = THPVariable_Unpack((*tt).ptr());
    if (cache.contains(name)) {
      if (cache[{name}].sizes() != t.sizes()) print("WHOAAAA!!!!, MISMATCH..", name, cache[{name}].sizes(), t.sizes());
      else {
        if(torch::allclose(t.to(torch::kCPU), cache[{name}].to(torch::kCPU), 1E-03F, 1E-05F)) print(name, " caches are the same");
        else print(name + ": caches are not equal, python", get_first_elements(t, 4), "c++", get_first_elements(cache[name], 4));
      }
    }
  }
}

void CheckpointProcessor::load_config(const py::dict& config) {
  cfg["lr"] = 1e-3f;
  cfg["frac_train"] = 0.3f;
  forEach([&](auto name){ cfg[name] = py::object(config[py::str(name)]).cast<int>(); }, 
    "d_model", "d_vocab", "d_vocab_out", "n_ctx", "d_head", "d_mlp", "n_heads", "n_layers");
}

std::map<std::string, torch::Tensor> CheckpointProcessor::get_state_dict(int n) {
  auto pairs { models[n]->named_parameters().pairs() };
  auto it = std::remove_if(pairs.begin(), pairs.end(), [](const auto& x) { return x.first.find("hook_") != std::string::npos; });
  pairs.erase(it, pairs.end());
  return { pairs.begin(), pairs.end() };
}
#include <unistd.h>
at::Tensor CheckpointProcessor::get_mag_symmetries(const at::Tensor& input) {
  print("made it here 0");
  
  std::cout << std::flush;
  auto mags = input;
  auto num_freqs = mags.size(-1);
  auto sz = mags.sizes();
  print("made it here");
  
  std::cout << std::flush;
  sleep(5);
  
  mags.index_put_({torch::logical_and(torch::greater(mags, -0.0000000000000001f), torch::less(mags, 0.0000000002f))}, 0.0000000002f);

  print("made it 2");
  std::cout << std::flush;
  sleep(5);

  std::terminate();
  auto maxandindex = mags.max(-1, true);
  auto [max_mags, max_idx] = maxandindex;
  auto rel_mags = mags / max_mags;

  auto t = rel_mags.flatten(0, -1);
  auto numel = t.numel()/num_freqs;
  std::vector<float> res { numel * numel };
  const auto* ptr = t.data_ptr<float>();
  auto mags_similar = [&](int a, int b) { 
    bool similar = true;
    for (int f = 1; f < num_freqs; ++f) {
      if (std::abs(ptr[a+f] - ptr[b+f]) > 0.1f) {
        similar = false;
        break;
      }
    }
    if (similar) res[a * numel + b] = 1.f;
  };
  for (int j = 0; j < numel; ++j) {
    for (int i = 0; i < numel; ++i)
      mags_similar(j, i);
  }

  return torch::from_blob(res.data(), {{int(numel)}, {int(numel)}}).to(mags.device());
}

void CheckpointProcessor::set_state_dict(int n, const std::map<std::string, torch::Tensor>& state_dict) {
  models[n]->set_state_dict(state_dict);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        checkpaint module
        -----------------------

        .. currentmodule:: checkpaint

        .. autosummary::
           :toctree: _generate

           CheckpointProcessor
    )pbdoc";

    //////////////// CheckpointProcessor class /////////////////
    py::class_<CheckpointProcessor>(m, "CheckpointProcessor")
        .def(py::init<py::list>())
        .def(py::init<py::dict>())
        .def("get_metrics", &CheckpointProcessor::get_metrics, R"pbdoc(
        Gets metrics from all checkpoints.
    )pbdoc")
        .def("get_key_freqs", &CheckpointProcessor::get_key_freqs)
        .def("get_custom_metric", &CheckpointProcessor::get_custom_metric)
        .def("get_last_activations", &CheckpointProcessor::get_last_activations)
        .def("get_modified_activations", &CheckpointProcessor::get_modified_activations)
        .def("get_modified_logits", &CheckpointProcessor::get_modified_logits)
        .def("get_activations_slice", &CheckpointProcessor::get_activations_slice)
        // .def("get_activations_slice", &CheckpointProcessor::get_activations_slice)
        // .def("add", py::overload_cast<int, int>(&add), "Add two integers");
        .def("get_all_activations", &CheckpointProcessor::get_all_activations)
        .def("load_checkpoints", &CheckpointProcessor::load_checkpoints)
        .def("get_state_dict", &CheckpointProcessor::get_state_dict)
        .def("set_state_dict", &CheckpointProcessor::set_state_dict)
        .def("get_mag_symmetries", &CheckpointProcessor::get_mag_symmetries)
        .def("get_shapes", &CheckpointProcessor::get_shapes, R"pbdoc(
        Gets the shapes of activations and other tensors required to provide
        inputs to trace a jit model. This model can be used to get metrics.
    )pbdoc")
        .def("compare_cache", &CheckpointProcessor::compare_cache, R"pbdoc(
        Compares an ActivationCache to verify against internal cache.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}