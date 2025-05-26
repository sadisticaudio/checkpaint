#pragma once
#include <torch/csrc/autograd/python_variable.h>
#include <boost/python.hpp>
#include <random>

#include "../neuralBase.h"

using namespace torch::indexing;


// template <int N = 0> at::Tensor cos_xpy(at::Tensor tensor, float freq, bool collapse_dim = false) {
//           auto cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1).flatten().to(tensor.device());
//           auto sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq).flatten().to(tensor.device());
//           auto cos_xpy_dir = (cosx_cosy_direction - sinx_siny_direction)/std::sqrt(2.f);
//           if (tensor.size(0) ==  numTrain) cos_xpy_dir = cos_xpy_dir.slice(0, 0, numTrain);
//           else if (tensor.size(0) ==  numTest) cos_xpy_dir = cos_xpy_dir.slice(0, numTrain, numTotal);
//           auto cos1 = cos_xpy_dir.index({ Slice(), None }), cos2 = cos_xpy_dir.index({ None, Slice() });
//           // print ("cos1, cos2, tensor", cos1.sizes(), cos2.sizes(), tensor.sizes());
//           if (collapse_dim) return torch::matmul(cos_xpy_dir, tensor);
//           else return torch::matmul(torch::matmul(cos1, cos2), tensor); 
//         }
        
//         template <int N = 0> at::Tensor sin_xpy(at::Tensor tensor, float freq, bool collapse_dim = false) {
//           auto sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1).flatten().to(tensor.device());
//           auto cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq).flatten().to(tensor.device());
//           auto sin_xpy_dir = (sinx_cosy_direction + cosx_siny_direction)/std::sqrt(2.f);
//           if (tensor.size(0) ==  numTrain) sin_xpy_dir = sin_xpy_dir.slice(0, 0, numTrain);
//           else if (tensor.size(0) ==  numTest) sin_xpy_dir = sin_xpy_dir.slice(0, numTrain, numTotal);
//           if (collapse_dim) return torch::matmul(sin_xpy_dir, tensor);
//           else return torch::matmul(torch::matmul(sin_xpy_dir.index({ Slice(), None }), sin_xpy_dir.index({ None, Slice() })), tensor); 
//         }

// template <int N = 0> at::Tensor calculate_restricted_loss(at::Tensor logits, const std::string& mode = "train") {
        //   at::Tensor trig_logits;
        //   for (const auto& freq : key_freqs) {
        //     auto x = cos_xpy(logits, freq, false) + sin_xpy(logits, freq, false);
        //     trig_logits = x + (trig_logits.numel() ? trig_logits : torch::zeros_like(x));
        //   }
        //   return test_logits(trig_logits, mode, true, &logits);
        // }

// std::map<int, at::Tensor> makeCoses() {
        //   constexpr long p { pModulo };
        //   auto a = torch::arange(p, torch::kFloat).index({ Slice(), None, None });
        //   auto b = torch::arange(p, torch::kFloat).index({ None, Slice(), None });
        //   auto c = torch::arange(p, torch::kFloat).index({ None, None, Slice() });
        //   static std::map<int, at::Tensor> cosMap;
        //   for (const auto& freq : key_freqs) {
        //     auto cube_predicted_logits { torch::cos(freq * 2.f * pi<float> / p * (a + b - c)) };
        //     cosMap[freq] = cube_predicted_logits/cube_predicted_logits.norm();
        //   }
        //   return cosMap;
        // }

        // at::Tensor makeCosCube() {
        //   constexpr long p { pModulo };
        //   auto a = torch::arange(p, torch::kFloat).index({ Slice(), None, None });
        //   auto b = torch::arange(p, torch::kFloat).index({ None, Slice(), None });
        //   auto c = torch::arange(p, torch::kFloat).index({ None, None, Slice() });
        //   static std::vector<at::Tensor> cube;
        //   for (long freq { 1 }; freq < p/2 + 1; ++freq) {
        //     auto cube_predicted_logits { torch::cos(freq * 2.f * pi<float> / p * (a + b - c)) };
        //     cube.push_back(cube_predicted_logits/cube_predicted_logits.norm());
        //   }
        //   return torch::stack({ cube }, 0L);
        // }

        // template <int N = 0> at::Tensor get_cos_sim(at::Tensor logits) { return get_cos_coeffs(logits)/logits.norm(); }

        // template <int N = 0> at::Tensor get_residual_cos_sim(at::Tensor logits) {
        //     auto vals = get_cos_coeffs(logits);
        //     auto residual = logits - (vals.index({ Slice(), None, None, None }) * cos_cube).sum(0);
        //     return residual.norm() / logits.norm();
        // }

        // template <int N = 0> at::Tensor embed_to_cos_sin(at::Tensor fourier_embed) {
        //   if (fourier_embed.dim() == 1)
        //     return torch::stack({fourier_embed.index({Slice(1, None, 2)}), fourier_embed.index({Slice(2, None, 2)})});
        //     else return torch::stack({ fourier_embed.index({ Slice(), Slice(1, None, 2)}), fourier_embed.index({ Slice(), Slice(2, None, 2)}) }, 1);
        // }

        // template<int N = 0> at::Tensor get_cos_coeffs(at::Tensor logits) {//, const std::string& mode) {
        //   logits = logits.view({ pModulo, pModulo, pModulo });
        //   auto vals = (cos_cube * logits.unsqueeze(0)).sum({ -3, -2, -1 });
        //   return vals;
        // }


  // void initWeights() {
  //   torch::NoGradGuard guard;
  //   for (auto& pp : this->named_parameters()) {
  //     auto& param { pp.value() };
  //     if (pp.key().contains("bias")) param.copy_(torch::zeros_like(param));
  //     else if (pp.key().contains("unembed")) param.copy_(torch::randn_like(param)/sqrtf(float(int(cfg["d_vocab"]))));
  //     else param.copy_(torch::randn_like(param)/sqrtf(float(int(cfg["d_model"]))));
  //   }
  // }


  // template<typename T> inline
// std::vector<T> to_vec(const bp::object& iterable) { using it = bp::stl_input_iterator<T>; return { it(iterable), it() }; }


// template <class T> inline bp::list to_list(std::vector<T> vector) {
//   typename std::vector<T>::iterator iter;
//   bp::list list;
//   for (iter = vector.begin(); iter != vector.end(); ++iter) {
//     list.append(*iter);
//   }
//   return list;
// }

// void CheckpointProcessor::test_submodule(bp::list input_tensors, bp::list result_tensors, bp::str fnName, int n) {
//   if (currentModel == nullptr) { print("testing subModule of current model which is a nullptr"); return; }
//   auto fn { fromStr(fnName) };
//   bool name_exists = [&]{ bool found { false }; for (auto x : tests) if (fn == x) return true; return false; }();
//   if (!name_exists) { print("testing subModule with fn \"" + fn + "\", which does not exist"); return; }
//   auto& model { *currentModel };
//   auto input_tensor = bp::stl_input_iterator<bp::api::object>(input_tensors);
//   at::Tensor py_x { THPVariable_Unpack((*input_tensor).ptr()).to(model.embed->W_E.device()) }, x { py_x };
//   auto result_tensor { bp::stl_input_iterator<bp::api::object>(result_tensors) };
//   at::Tensor py_y { THPVariable_Unpack((*result_tensor).ptr()).to(model.embed->W_E.device()) };
//   auto maybeCheck = [&](const std::string& s, const at::Tensor& y) {
//     // lambda to check if C++ RETURN TENSOR(S) EQUAL TO PYTHON RETURN TENSOR(S)???
//     if (s == fn && torch::equal(y, py_y)) print(fn + " test passed");
//     else if (s == fn) {
//       print(fn + " test failed");
//       print("x", x);
//       print("py_x", py_x);
//       print("py_y, y, sizes", py_y.sizes(), y.sizes());
//       print(fn + " test failed\npython result first 4 elements", get_first_elements(py_y, 4));
//       print(fn + " test failed\nc++ result first 4 elements", get_first_elements(y, 4));
//     } };

//   // TEST
//   Vec<at::Tensor> tv {{ x }};
//   maybeCheck("embed", tv[0] = model.embed(tv[0]));
//   maybeCheck("pos_embed", tv[0] = tv[0] + model.pos_embed(tv[0]));
//   tv.push_back(torch::zeros_like(tv[0]));
//   tv = model.blocks(tv);
//   maybeCheck("blocks", tv[0]);
//   maybeCheck("unembed", tv[0] = model.unembed(tv[0]));
// }

// template <int N = 0> at::Tensor cos_xpy(at::Tensor tensor, float freq, bool collapse_dim = false) {
        //   auto cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1).flatten().to(tensor.device());
        //   auto sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq).flatten().to(tensor.device());
        //   auto cos_xpy_dir = (cosx_cosy_direction - sinx_siny_direction)/std::sqrt(2.f);
        //   if (tensor.size(0) ==  numTrain) cos_xpy_dir = cos_xpy_dir.slice(0, 0, numTrain);
        //   else if (tensor.size(0) ==  numTest) cos_xpy_dir = cos_xpy_dir.slice(0, numTrain, numTotal);
        //   auto cos1 = cos_xpy_dir.index({ Slice(), None }), cos2 = cos_xpy_dir.index({ None, Slice() });
        //   // print ("cos1, cos2, tensor", cos1.sizes(), cos2.sizes(), tensor.sizes());
        //   if (collapse_dim) return torch::matmul(cos_xpy_dir, tensor);
        //   else return torch::matmul(torch::matmul(cos1, cos2), tensor); 
        // }
        
        // template <int N = 0> at::Tensor sin_xpy(at::Tensor tensor, float freq, bool collapse_dim = false) {
        //   auto sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1).flatten().to(tensor.device());
        //   auto cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq).flatten().to(tensor.device());
        //   auto sin_xpy_dir = (sinx_cosy_direction + cosx_siny_direction)/std::sqrt(2.f);
        //   if (tensor.size(0) ==  numTrain) sin_xpy_dir = sin_xpy_dir.slice(0, 0, numTrain);
        //   else if (tensor.size(0) ==  numTest) sin_xpy_dir = sin_xpy_dir.slice(0, numTrain, numTotal);
        //   auto sin1 = sin_xpy_dir.index({ Slice(), None }), sin2 = sin_xpy_dir.index({ None, Slice() });
        //   if (collapse_dim) return torch::matmul(sin_xpy_dir, tensor);
        //   // else return torch::matmul(torch::matmul(sin_xpy_dir.index({ Slice(), None }), sin_xpy_dir.index({ None, Slice() })), tensor); 
        //   else return torch::matmul(torch::matmul(sin1, sin2), tensor); 
        // }

        // at::Tensor calculate_excluded_loss_2D(at::Tensor logits, at::Tensor labels) {
        //   using Slice = torch::indexing::Slice;
        //   const auto None = torch::indexing::None;
        //   constexpr long p { pModulo };
        //   logits = getLastLogit(logits);
        //   std::vector<at::Tensor> row;
        //   for (long freq { 1 }; freq < pModulo/2 + 1; ++freq) {
        //     row.push_back(test_logits(logits 
        //                               - cos_xpy(logits, freq, false) 
        //                               - sin_xpy(logits, freq, false), labels, false, nullptr));
        //   }
        //   // return row;
        //   return torch::stack({ row }); // return a Tensor or TensorList (vector<Tensor>)???
        // }

        // template <int N = 0> at::Tensor calculate_excluded_loss(at::Tensor logits, at::Tensor labels, const std::string& mode = "train") {
        //   auto new_logits = logits.clone();
        //   for (const auto& freq : key_freqs) {
        //     new_logits -= cos_xpy(logits, freq, false);
        //     new_logits -= sin_xpy(logits, freq, false);
        //   }
        //   // print("new_logits at bottom of calculate", new_logits.sizes());
        //   return test_logits(new_logits, labels);
        // }

        

        // template <int N = 0> at::Tensor calculate_excluded_accuracy(at::Tensor logits, at::Tensor labels, const std::string& mode = "train") {
        //   auto new_logits = logits.clone();
        //   for (const auto& freq : key_freqs) {
        //     new_logits -= cos_xpy(logits, freq, false);
        //     new_logits -= sin_xpy(logits, freq, false);
        //   }
        //   // print("new_logits at bottom of calculate", new_logits.sizes(), new_logits.device(), LABELS[mode].device());
        //   return getAccuracy(new_logits, labels);
        // }

        

        // template <int N = 0> at::Tensor calculate_restricted_logits(at::Tensor logits, at::Tensor labels) {
        //   at::Tensor trig_logits;
        //   for (const auto& freq : key_freqs) {
        //     auto x = cos_xpy(logits, freq, false) + sin_xpy(logits, freq, false);
        //     trig_logits = x + (trig_logits.numel() ? trig_logits : torch::zeros_like(x));
        //   }
        //   return trig_logits;//test_logits(trig_logits, labels, true, &logits);
        // }

        // template <int N = 0> at::Tensor calculate_restricted_accuracy(at::Tensor logits, const std::string& mode = "train") {
        //   at::Tensor trig_logits;
        //   for (const auto& freq : key_freqs) {
        //     auto x = cos_xpy(logits, freq, false) + sin_xpy(logits, freq, false);
        //     trig_logits = x + (trig_logits.numel() ? trig_logits : torch::zeros_like(x));
        //   }
        //   trig_logits = trig_logits + logits.mean(0, true) - trig_logits.mean(0, true);
        //   return getAccuracy(trig_logits, LABELS[mode]);
        // }

// template<typename M> long getNumParams(const M& m) { long n {}; for (auto& q : m->parameters()) n += q.numel(); return n; }
// template<typename M> long getParamSize(const M& m) { long n {}; for (auto& q : m->parameters()) n += q.numel() * q.element_size(); return n; }
// template<typename M> long getBufferSize(const M& m) { long n {}; for (auto& q : m->buffers()) n += q.numel() * q.element_size(); return n; }
// template<typename M> long getGradSize(const M& m) { long n {}; for (auto& q : m->parameters()) n += q.grad().numel() * q.element_size(); for (auto& q : m->buffers()) n += q.grad().numel() * q.element_size(); return n; }

// template<typename M> long getEstCudaUse(const M& m) { return (getParamSize(m) + getBufferSize(m) + getGradSize(m)); }

// void print(nn::Module* model) {
//   std::cout << "\n" << model->name() << ": nParams: " << getNumParams(model) << ", " << 
//   toHumanReadable(getParamSize(model)) << ", " << toHumanReadable(getEstCudaUse(model)) << " (x4 ≈ est. GPU Usage)\n"; }

// template <typename ModuleType> void printWithLayersReduced(const ModuleType& model) {
//   std::map<std::string, long> paramMap;
//   long total {};
//   for (const auto& x : model->named_parameters()) {
//     const auto& s = x.key();
//     std::string mapKey { s };
//     if (s.contains(".layer")) {
//       size_t pos { s.find(".layer") + 1UL };
//       pos = s.find(".", pos) + 1UL;
//       mapKey = s.substr(0UL, 7UL) + s.substr(pos);
//     }
//     if (mapKey.contains(".weight")) mapKey = mapKey.substr(0UL, mapKey.length() - 7UL);
//     if (mapKey.contains(".bias")) mapKey = mapKey.substr(0UL, mapKey.length() - 5UL);
//     total += x.value().numel() * x.value().element_size();
//     paramMap[mapKey] += x.value().numel() * x.value().element_size();
//   }
//   for (const auto& x : paramMap) std::cout << x.first + ": " << toHumanReadable(x.second) + "\n";
//   std::cout << "\nTOTAL: " << toHumanReadable(total) + "\n\n";
// }

// template<typename T> void print(T label, Tensor values...) { std::cout << label << ":\n";
//         forEach ([&](auto value){ std::cout << value << "\n"; }, values); std::cout << "\n"; }

// template <typename T> Vec<T> toVec(Tensor t) { 
//   auto vec { Vec<T>::make(t.size(-1)) }; for (int i {}; i < t.size(-1); ++i) vec[i] = t[i].item<T>(); return vec; }

// void printTensor(const torch::Tensor& t, std::string name = std::string()) {
//   if (name != std::string()) print(name);
//   print("tensor sizes", t.sizes());
//   std::cout << std::fixed << std::setprecision(6);
//   // print("content below if rank is 3");
//   if (t.dim() == 3) {
//     for (int n3 { 0 }; n3 < std::min(2, int(t.size(-3))); ++n3) {
//       std::cout << "\n batch " << n3 << ":   ";
//       for (int n2 { 0 }; n2 < std::min(8, int(t.size(-2))); ++n2) {
//         std::cout << "\n";
//         for (int n1 { 0 }; n1 < std::min(6, int(t.size(-1))); ++n1) {
//           std::cout << t[n3][n2][n1].item().toFloat() << ", ";
//         }
//       }
//     }
//   }
//   else if (t.dim() == 2) {
//     for (int n2 { 0 }; n2 < std::min(3, int(t.size(-2))); ++n2) {
//       std::cout << "\n";
//       for (int n1 { 0 }; n1 < std::min(6, int(t.size(-1))); ++n1) {
//         std::cout << t[n2][n1].item().toFloat() << ", ";
//       }
//     }
//   }
//   std::cout << "\n";
// }

// void printVersion(const Tensor& t, std::string name = std::string()) { if (!name.empty()) std::cout << name << ": ";
//     std::cout << "version " << t._version() << "\n"; }
// void printTVVersions(const Vec<Tensor>& tv, std::string name = std::string()) { if (!name.empty()) print(name);
//     for (int i {}; i < tv.size(); ++i) std::cout << "tv[" << i << "/" << tv.size() << "] version " << tv[i]._version() << "\n"; }
// void checkTVSizes(const Vec<Tensor>& tv, std::string name = std::string()) { if (!name.empty()) print(name);
//     for (int i {}; i < tv.size(); ++i) std::cout << "tv[" << i << "/" << tv.size() << "]" << tv[i].sizes() << "\n"; }

// template<typename M1, typename M2> 
// Vec<int> equalizeMemoryUsage(GPTConfig& cfg1, GPTConfig& cfg2, string pName, SadisticRange range, string pOther, int N) {
//   Vec<int> retVals;
//   for (int n {}; n < N; ++n) {
//     cfg1[pName].value = range.map(float(n)/N);
//     M1 model { cfg1, torch::kCPU };
//     size_t estUse { getEstCudaUse(&model) };
//     if (estUse > size_t(4e+9)) { print("est use to equalize on is greater than 4Gb"); std::terminate(); }
//     model->to(torch::kCUDA);
//     // size_t usage { getCudaMemUsage() };
//     // M2 model2 { cfg2, torch::kCPU };
//     // size_t estUse2 { getEstCudaUse(&model2) };
//   }
// }

// struct TrainingResult {
//   static constexpr const char* names[] { 
//     "trainLoss", "valLoss", "fakeWords", "badCaps", "badPunc", "accuracy", "perplexity", "confidence", "uncertainty" };
//   static constexpr const size_t numMetrics { sizeof(names)/sizeof(const char*) };
//   float& operator[](const char* x) { for(int i { 0 }; i < numMetrics; ++i) if (names[i] == x) return metrics[i]; return metrics[0]; }
//   const float& operator[](const char* x) const { for(int i { 0 }; i < numMetrics; ++i) if (names[i] == x) return metrics[i]; return metrics[0]; }
//   float& operator[](int i) { return metrics[i]; }
//   const float& operator[](int i) const { return metrics[i]; }
//   SadisticMap<std::string, float> metrics { names };
// };

// struct TrainingResults {
//   using T = TrainingResult;
//   T& operator[](int i) { return vec[i]; }
//   const T& operator[](int i) const { return vec[i]; }
//   T& push_back(const T& x) { vec.push_back(x); return vec[vec.size() - 1]; }
//   T& push_back(T&& x) { vec.push_back(x); return vec[vec.size() - 1]; }
//   int size() const { return vec.size(); }
//   std::string title;
//   Vec<T> vec;
//   float bestLoss{ 999.f };
// };

// std::ostream& operator<<(std::ostream& os, const TrainingResults& r) {
//   os << r.title << ":\n";
//   for (const auto& x : r.vec) { for (const auto& y : x.metrics.data)
//   os << y << " "; os << "\n"; }
//   return os;
// }

// std::string operator/(const TrainingResults& l, const TrainingResults& r) {
//   std::string x { l.title + "/" + r.title + " (LOWER IS BETTER)\n" };
//   float tLosses{}, vLosses{}, fakeWords{}, badCaps{}, badPunc{};
//   // int num { std::min(l.vec.size(), r.vec[0]["trainLoss"].size()) };
//   int numEpochs { int(r.size()) };
//   constexpr int numMetrics { TrainingResult::numMetrics };
//   float ML[numMetrics][99]{}, MR[numMetrics][99]{};
//   for (int m {}; m < numMetrics; ++m) {
//     x += "\n" + l[0].metrics.data[m].first + ": ";
    
//     for (int e {}; e < numEpochs; ++e) {
//       float mL{ l[e][m] }, mR{ r[e][m] };
//       std::string mLS { std::to_string(mL) }, mRS { std::to_string(mR) };
//       x += std::to_string(e) + ": (" + mLS + "/" + mRS + ") -> " + std::to_string((mL + 0.01f)/(mR + 0.01f)) + " ";
//     }
    
//   }
//   return x;
// }

// template<template <int,int> class ModuleTemplate, typename ConfigType = GPTConfig, int N = defaultNumLayers, int n = 0> 
// struct ModuleStack : nn::Module {
//   using ModuleTypeImpl = ModuleTemplate<N,n>;
//   // torch::detail::return_type_of_forward_t<>
//   using SubModuleTypeImpl = ModuleStack<ModuleTemplate, ConfigType, N, n + 1>;
//   template<typename ...Ts> ModuleStack(const ConfigType& cfg, Ts ...ts) : c(cfg),
//   module(register_module(std::to_string(n), ModuleType(c, ts...))),
//   subModules(this, c, ts...) {}
//   template<typename ...Ts> ModuleStack(nn::Module* parent, const ConfigType& cfg, Ts ...ts) : c(cfg),
//   module(parent->register_module(std::to_string(n), ModuleType(c, ts...))),
//   subModules(parent, c, ts...) {}
//   Vec<Tensor> forward (Vec<Tensor> tv) {
//     tv = module(tv);
//     tv = subModules(tv);
//     return tv; }
//   TORCH_MODULE(ModuleType);
//   TORCH_MODULE(SubModuleType);
//   const ConfigType& c;
//   ModuleType module;
//   SubModuleType subModules;
// };

// template<template <int,int> class ModuleType, typename ConfigType, int N> struct ModuleStack<ModuleType, ConfigType, N, N> : nn::Module {
//   template<typename ...Ts> ModuleStack(Ts ...ts) {}
//   Vec<Tensor> forward (Vec<Tensor> input) { return input; }
// };

// // static int getActiveMemory() { return c10::cuda::CUDACachingAllocator::getDeviceStats(0).active_bytes[0].current/1000000; }
  
// bool checkTensor(Tensor x, std::vector<int> p, std::string name = std::string()) {
//   if (p.size() > x.dim()) { print("Tensor doesn't have enough dims " + name, p.size(), x.dim()); return false; }
//   for (int i {}; i < p.size(); ++i)
//     if (p[i] != -1 && p[i] != x.size(i)) { print("Tensor dim not correct size " + name + ": i, p[i], x.size(i)", i, p[i], x.size(i)); return false; }
//   return true;
// }

// bool checkTensors(Vec<Tensor> tv, std::vector<std::vector<int>> szs, std::string name = std::string()) {
//   bool allGood = true;
//   for (int i {}; i < tv.size(); ++i) {
//     if (!checkTensor(tv[i], szs[i], name + " - tv[" + std::to_string(i) + "]")) {
//       for (int i {}; i < szs.size(); ++i) print( "expected sizes() for tv[" + std::to_string(i) + "]", szs[i]);
//       for (int i {}; i < tv.size(); ++i) print("actual sizes() for tv[" + std::to_string(i) + "]", tv[i].sizes());
//       allGood = false;
//     }
//   }
//   return allGood;
// }

// Tensor padLeft(Tensor x, std::vector<int> sizes) {
//   Vec<int> intArr;
//   while (x.dim() > sizes.size()) { x = x.slice(0, x.size(0) - 1, x.size(0)); x.squeeze(0); }
//   while (x.dim() < sizes.size()) { x = x.unsqueeze(0); }
//   for (int i { 1 }, r { int(sizes.size()) - 1 }; r >= 0; ++i, --r) {
//     if (x.size(-i) > sizes[r]) x = x.slice(-i, x.size(-i) - sizes[r], x.size(-i));
//     intArr.push_back(sizes[r] - x.size(-i));
//     intArr.push_back(0);
//   }
//   x = pad(x, torch::nn::functional::PadFuncOptions({ intArr.begin(), intArr.end() }));
//   return x;
// }

// Tensor getTopkIncludingFirstColumn(Tensor x, int k) {
//   auto [val, idx] = x.topk(1, -2);
//   val.slice(-1, 0, 1) = 666666.f;
//   return std::get<1>(val.mean(1).topk(k, -1)).slice(-1, 0, k).squeeze();
// }

// Tensor sadGather(Tensor x, Tensor idx) {
//   long n { idx.dim() };
//   auto sz { x.sizes() };
//   std::vector<long> vec { sz.begin(), sz.end() };
//   vec[n - 1] = idx.size(n - 1);
//   // print("vec", vec);
//   auto newIdx = idx;
//   for (int i { 0 }; i < x.dim() - n; ++i) newIdx = newIdx.unsqueeze(-1);
//   // print("x.sizes(), idx.sizes(), newIdx.sizes(), vec", x.sizes(), idx.sizes(), newIdx.sizes(), vec);
//   return x.gather(n - 1, newIdx.expand({ vec.data(), vec.size() }));
// }

// at::Tensor getExcludedLoss(const std::string& mode, at::Tensor labels, at::Tensor W_U, at::Tensor W_out, ActivationCache& cache) {
//         // if (logits.size(0) != numTrain) { print("trying to getExcludedLoss of test (or all) logits", logits.sizes()); return at::Tensor(); }
//         print("cache, cache[\"blocks.0.mlp.hook_post\"], cos_apb_vec[f].sizes()", cache.data.size(), cache["blocks.0.mlp.hook_post"].sizes(), cos_apb_vec[0].sizes());
//         auto neuron_acts = getLastLogit(cache["blocks.0.mlp.hook_post"]);
//         print("neuron_acts.sizes()", neuron_acts.sizes());

//         // auto neuron_acts = cache[{ "post", 0, "mlp" }];
//         // neuron_acts = neuron_acts.slice(1, neuron_acts.size(1) - 1, neuron_acts.size(1));
//         auto approx_neuron_acts = torch::zeros_like(neuron_acts);

//         for (const auto& f : key_freqs) {
//         print("neuron_acts.sizes(), cos_apb_vec[f].sizes()", neuron_acts.sizes(), cos_apb_map[f].sizes());
//         approx_neuron_acts += (neuron_acts * cos_apb_map[f]).sum(0) * cos_apb_map[f];
//         approx_neuron_acts += (neuron_acts * sin_apb_map[f]).sum(0) * sin_apb_map[f];
//         }
//         auto excluded_neuron_acts = neuron_acts - approx_neuron_acts;
//         auto residual_stream_final = torch::matmul(excluded_neuron_acts, W_out) + getLastLogit(cache["blocks.0.hook_resid_pre"]);
//         auto excluded_logits = torch::matmul(residual_stream_final, W_U);
//         return cross_entropy_high_precision(excluded_logits, labels);
//         }

//         template <int N = 0> at::Tensor calculate_restricted_accuracy(at::Tensor logits, at::Tensor labels) {
//         at::Tensor trig_logits;
//         for (const auto& freq : key_freqs) {
//         auto x = cos_xpy(logits, freq, false) + sin_xpy(logits, freq, false);
//         trig_logits = (trig_logits.numel() ? trig_logits + x : x);
//         }
//         trig_logits = trig_logits + logits.mean(0, true) - trig_logits.mean(0, true);
//         return getAccuracy(trig_logits, labels);
//         }

//         template <int N = 0> at::Tensor calculate_restricted_logits(at::Tensor logits, at::Tensor labels) {
//         logits = logits.to(torch::kCPU); labels = labels.to(torch::kCPU);
//         at::Tensor trig_logits;
//         for (const auto& freq : key_freqs) {
//         auto x = cos_xpy(logits, freq, false) + sin_xpy(logits, freq, false);
//         trig_logits = (trig_logits.numel() ? trig_logits + x : x);
//         }
//         return trig_logits + logits.mean(0, true) - trig_logits.mean(0, true);
//         }

// struct GrokkingMetricsJitModel : torch::nn::Module {
//   using T = torch::Tensor;

//   T forward(T logits, T LABELS, T W_out, T W_U, T post_mlp_act, T resid_mid_act, T cos_apb, T sin_apb) {
//     logits = getLastLogit(logits);
//     auto neuron_acts { post_mlp_act.index({ Slice(), -1, Slice() }) };
//     auto resid_mid { resid_mid_act.index({ Slice(), -1, Slice() }) };
//     auto approx_neuron_acts = torch::zeros_like(neuron_acts);
//     approx_neuron_acts += neuron_acts.mean(0);

//     for (int freq {}; freq < num_key_freqs; ++freq) {
//       //freq is index here... actual freq is in key_freqs[freq]
//       approx_neuron_acts += (neuron_acts * cos_apb[freq]).sum(0) * cos_apb[freq];
//       approx_neuron_acts += (neuron_acts * sin_apb[freq]).sum(0) * sin_apb[freq];
//     }
//     auto restricted_logits = torch::matmul(torch::matmul(approx_neuron_acts, W_out), W_U);
//     restricted_logits += logits.mean(0, true) - restricted_logits.mean(0, true);
//     auto excluded_neuron_acts = neuron_acts - approx_neuron_acts;
//     auto excluded_logits = torch::matmul(torch::matmul(excluded_neuron_acts, W_out) + resid_mid, W_U);
    
//     auto ret { torch::zeros({ 3, 2, 2 }, logits.device()) };
//     auto getMetric = [&ret, &LABELS, idx = 0](const T& lgtz) mutable { 
//       ret[idx][0][0] = cross_entropy_high_precision(getMode(lgtz, "train"), getMode(LABELS, "train"));
//       ret[idx][0][1] = getAccuracy(getMode(lgtz, "train"), getMode(LABELS, "train"));
//       ret[idx][1][0] = cross_entropy_high_precision(getMode(lgtz, "test"), getMode(LABELS, "test"));
//       std::cout << "idx before: " << idx << ", idx after: ";
//       ret[idx++][1][1] = getAccuracy(getMode(lgtz, "test"), getMode(LABELS, "test"));
//       std::cout << idx << std::endl;
//     };
//     getMetric(logits);
//     getMetric(restricted_logits);
//     getMetric(excluded_logits);
//     return ret;
//   }
// };
// // TORCH_MODULE(GrokkingMetricsJitModel);