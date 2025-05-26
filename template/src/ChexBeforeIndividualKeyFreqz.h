#pragma once

#include "helperz.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include "math.h"
#include <torch/script.h>

//////////////////////////// EPIPHONY - (POSSIBLY...) GROKKING IS DUE TO EXCLUDED MEMORIZATION ACCURACY 
//////////////////////////// HITTING A LOWER BOUND AND RESTRICTED (GENERALIZATION) ACCURACY HITTING 100%
//////////////////////////// THUS, THE WEIGHT DECAY NO LONGER HAS TO REMOVE FROM MEMORIZATION AND ADD TO
//////////////////////////// GENERALIZATION AND CAN SOLELY DECREASE TOTAL WEIGHT NORM AND ELIMINATE NOISE
//////////////////////////// IMPROVING CONFIDENCE / DECREASING TOTAL LOSS AT HIGHEST RATE

using hi_res = std::chrono::high_resolution_clock;

template<typename T> T sadMatmul(T x1, T x2, std::string nm = "") { //std::cout << nm + " " << x1.sizes() << " @ " << x2.sizes() << std::endl; 
  return torch::matmul(x1,x2); }

template<typename T> T badMatmul(T x, T W, std::string nm = "") { //std::cout << nm + " " << x1.sizes() << " @ " << x2.sizes() << std::endl; 
  size_t N = size_t(W.size(0));
  std::vector<at::Tensor> vec { N, at::Tensor() };
  for (int i {}; i < N; ++i) {
    // print("x W sadMatmul x W", x.sizes(), W.sizes(), x.index({ Slice(), i }).sizes(), W[i].sizes());
    vec[i] = sadMatmul(x.index({ Slice(), i }), W[i], nm + " " + std::to_string(i));
  }
  x = torch::stack(at::TensorList(vec), 1);
return x; }

template<int N = 0> size_t getCudaMemUsage() { size_t memTotal{}, memFree{}; cudaMemGetInfo(&memFree, &memTotal); size_t rv = memTotal - memFree; return rv; }
template<int N = 0> size_t getCudaMemTotal() { size_t memTotal{}, memFree{}; cudaMemGetInfo(&memFree, &memTotal); size_t rv = memTotal; return rv; }
template<int N = 0> size_t getCudaMemFree() { size_t memTotal{}, memFree{}; cudaMemGetInfo(&memFree, &memTotal); size_t rv = memFree; return rv; }

template<int N = 0> void printCudaMemUsage(std::string msg = std::string()) {
  auto toH = [](auto x) { return toHumanReadable(x); };
  auto statToH = [&](auto x)->std::string { return "current: (" + toH(x[0].current) + ") peak: (" + toH(x[0].peak) + ") allocated: (" + toH(x[0].allocated) + ")\n"; };
  auto statsToH = [&](auto s)->std::string { return "reserved_bytes: " + statToH(s.reserved_bytes) + "active_bytes: " + statToH(s.active_bytes) + "\n"; };
  auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  print(msg + ": " + "CUDA Memory", toH(getCudaMemUsage()) + " / " + toH(getCudaMemTotal()));
  print(statsToH(stats));
}

struct AllData { TensorMap dataset, labels; };

inline AllData makeAllData() {
  print("in makeAllData()");
  constexpr long p { pModulo };

  AllData allData;
  auto& [dataset, labels] = allData;

  auto a = einsad::repeat(torch::arange(p), "i -> (i j)", { "j", p });
  auto b = einsad::repeat(torch::arange(p), "j -> (i j)", { "i", p });
  auto eq = einsad::repeat(torch::tensor(p), " -> (i j)", { "i", p }, { "j", p });
  dataset["all"] = torch::stack(at::TensorList({ a, b, eq }), 1);
  labels["all"] = (dataset["all"].index({ Slice(), 0 }) + dataset["all"].index({ Slice(), 1 })) % p;
  dataset["train"] = dataset["all"].index({ indices["train"] });
  dataset["test"] = dataset["all"].index({ indices["test"] });
  labels["train"] = labels["all"].index({ indices["train"] });
  labels["test"] = labels["all"].index({ indices["test"] });
  print("train_data", dataset["train"].index({ Slice(None, 5) }));
  print("test_data", dataset["test"].index({ Slice(None, 5) }));
  print("train_labels", labels["train"].index({ Slice(None, 5) }));
  print("test_labels", labels["test"].index({ Slice(None, 5) }));
  
  return allData;
}

template<class M, typename ConfigType = GPTConfig, int N = 1, int n = 0> 
struct ModuleStack : torch::nn::Module {
  using ModuleTypeImpl = M;
  using SubModuleTypeImpl = ModuleStack<M, ConfigType, N, n + 1>;
  TORCH_MODULE(ModuleType);
  TORCH_MODULE(SubModuleType);
  template<typename ...Ts> ModuleStack(const ConfigType cfg, Ts ...ts) : c(cfg),
  module(this->register_module(std::to_string(n), ModuleType(cfg, ts...))),
  subModules(this, c, ts...) {}
  template<typename ...Ts> ModuleStack(torch::nn::Module* parent, const ConfigType cfg, Ts ...ts) : c(cfg),
  module(parent->register_module(std::to_string(n), ModuleType(c, ts...))),
  subModules(parent, c, ts...) {}
  at::Tensor forward (const at::Tensor& input) {
    auto x = module(input);
    x = subModules(x);
    return x; }
  ConfigType c;
  ModuleType module;
  SubModuleType subModules;
};

template<class M, typename ConfigType, int N> struct ModuleStack<M, ConfigType, N, N> : torch::nn::Module {// torch::nn::Cloneable<ModuleStack<ModuleType, ConfigType, N, N>> {
  template<typename ...Ts> ModuleStack(Ts ...ts) {}
  at::Tensor forward (const at::Tensor& input) { return input; }
};

struct HookPointImpl : torch::nn::Module {
  HookPointImpl(torch::nn::Module* p, const std::string nm) : parent(p), hook_name(nm) {}
  torch::Tensor forward(torch::Tensor x) {
    checknan(x);
    if (hook_added) {
      if (!act.numel()) { time = hi_res::now(); act = x; parent->register_parameter("hook_" + hook_name, act, false); }
      else act = x;
    }
    return x;
  }
  torch::nn::Module* parent;
  std::string hook_name;
  torch::Tensor act;
  hi_res::time_point time {};
  bool hook_added { false };
};
TORCH_MODULE(HookPoint);

template<typename T> static std::shared_ptr<HookPointImpl> register_hook(T* module, std::string name) {
    return module->register_module("hook_module_" + name, HookPoint((torch::nn::Module*)module, name));
}

struct EmbedImpl : torch::nn::Module {
  EmbedImpl(GPTConfig cc, int m = 1) : cfg(cc),
  W_E(register_parameter("W_E", torch::randn({ m, d_vocab, C })/std::sqrt(float(C)), false)) {}

  at::Tensor forward (at::Tensor x) {
    auto selected = W_E.index_select(-2, x.flatten());
    x = selected.reshape({ W_E.size(0), x.size(0), x.size(1), C }).transpose(0,1);
    return x;
  }

  GPTConfig cfg;
  int C { cfg["d_model"] }, d_vocab { cfg["d_vocab"] };
  torch::Tensor W_E;
};
TORCH_MODULE(Embed);

struct UnembedImpl : torch::nn::Module {
  UnembedImpl(GPTConfig cc, int m = 1) : cfg(cc),
  W_U(register_parameter("W_U", torch::randn({ m, C, d_vocab - 1 })/std::sqrt(float(d_vocab)), false)) {}

  at::Tensor forward (at::Tensor x) {
    // printCudaMemUsage("unembed0");
    auto ret = badMatmul(x, W_U, "unembed");
    // printCudaMemUsage("unembed1");
    // print("x", x.sizes(), "W_U", W_U.sizes(), "ret", ret.sizes());
    return ret;//.view({ x.size(0), x.size(1), x.size(2), W_U.size(-1) }); 
  }

  GPTConfig cfg;
  int C { cfg["d_model"] }, d_vocab { cfg["d_vocab"] };
  torch::Tensor W_U;
};
TORCH_MODULE(Unembed);

struct PosEmbedImpl : torch::nn::Module {
  PosEmbedImpl(GPTConfig cc, int m = 1) : cfg(cc), 
  W_pos(register_parameter("W_pos", torch::randn({ m, T, C })/std::sqrt(float(C)), false)) {}

  at::Tensor forward (at::Tensor x) {
    return einsad::repeat(W_pos, "mod pos d_model -> batch mod pos d_model", { "batch", x.size(0) }); }

  GPTConfig cfg;
  int T { cfg["n_ctx"] }, C { cfg["d_model"] };
  torch::Tensor W_pos;
};
TORCH_MODULE(PosEmbed);

using TArray = std::array<std::array<at::Tensor, 3>,4>;

struct NandaAttentionImpl : torch::nn::Module {
  NandaAttentionImpl(GPTConfig cc, int m = 1) : cfg(cc),
  W_K(register_parameter("W_K", torch::randn({ m, nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_Q(register_parameter("W_Q", torch::randn({ m, nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_V(register_parameter("W_V", torch::randn({ m, nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_O(register_parameter("W_O", torch::randn({ m, nh, C/nh, C })/std::sqrt(float(C)), false)),
  hook_k(register_hook(this, "k")),
  hook_q(register_hook(this, "q")),
  hook_v(register_hook(this, "v")),
  hook_z(register_hook(this, "z")),
  hook_pattern(register_hook(this, "pattern")),
  hook_attn_scores(register_hook(this, "attn_scores")) {}

  torch::Tensor forward (torch::Tensor x) {
    auto B = x.size(0), M = x.size(1);
    auto q = hook_q(einsad::einsum("m i d h, b m p d -> b m p i h", W_Q, x));// [nh C hs] @ [B T C] = [B T nh hs]
    auto k = hook_k(einsad::einsum("m i d h, b m p d -> b m p i h", W_K, x));// [nh C hs] @ [B T C] = [B T nh hs]
    auto v = hook_v(einsad::einsum("m i d h, b m p d -> b m p i h", W_V, x));// [nh C hs] @ [B T C] = [B T nh hs]
    q = einsad::rearrange(q, "b m p i h -> b p (m i) h");
    k = einsad::rearrange(k, "b m p i h -> b p (m i) h");
    v = einsad::rearrange(v, "b m p i h -> b p (m i) h");
    q = q.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    k = k.transpose(1,2).transpose(-2, -1); // [B T nh hs] -> [B nh T hs] -> [B nh hs T]
    v = v.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    auto attn_scores_pre = sadMatmul(q, k, "q k fwd")/std::sqrt(float(hs));// [B nh T hs] @ [B nh hs T] = [B nh T T]
    auto attn_scores_masked = attn_scores_pre + torch::full({ 1, T, T }, nInf).triu(1).to(k.device());// [B nh T T]
    auto pattern = hook_pattern(torch::nn::functional::softmax(hook_attn_scores(attn_scores_masked), -1));// [B nh T T]
    auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    z = einsad::rearrange(z, "batch (model head) query_pos d_head -> batch query_pos model head d_head", { "model", x.size(1) }, { "head", nh });
    auto w = einsad::rearrange(W_O, "model head d_head d_model -> d_model model head d_head");
    auto out = einsad::einsum("batch pos model head d_head, d_model model head d_head -> batch model pos d_model", z, w);
    return out;
  }
  
  GPTConfig cfg;
  int nh { cfg["n_heads"] }, T { cfg["n_ctx"] }, C { cfg["d_model"] }, hs { C/nh };
  torch::Tensor W_K, W_Q, W_V, W_O;
  HookPoint hook_k, hook_q, hook_v, hook_z, hook_pattern, hook_attn_scores;
};
TORCH_MODULE(NandaAttention);

struct NandaMLPImpl : torch::nn::Module {
  NandaMLPImpl(GPTConfig cfg, int m = 1) : c(cfg),
  W_in(register_parameter("W_in", torch::randn({ m, C, d_mlp })/std::sqrt(float(C)), false)),
  reLU(register_module("reLU", torch::nn::ReLU(torch::nn::ReLUOptions()))),
  W_out(register_parameter("W_out", torch::randn({ m, d_mlp, C })/std::sqrt(float(C)), false)),
  hook_pre(register_hook(this, "pre")),
  hook_post(register_hook(this, "post")) {}

  torch::Tensor forward(torch::Tensor x) {
    // WAS... x = torch::matmul(x, W_in); ... but, intermediate state was a huge bottleneck I think, didn't double check, same also for out projection
    x = badMatmul(x, W_in);
    x = hook_post(reLU(hook_pre(x)));
    x = badMatmul(x, W_out);
    return x;
  }
  GPTConfig c;
  int C { c["d_model"] }, d_mlp { c["d_mlp"] };
  torch::Tensor W_in;
  torch::nn::ReLU reLU;
  torch::Tensor W_out;
  HookPoint hook_pre, hook_post;
};
TORCH_MODULE(NandaMLP);

struct NandaDecoderBlock : torch::nn::Module {
  NandaDecoderBlock(GPTConfig cfg, int m = 1) : c(cfg),
  attn(register_module("attn", NandaAttention(c, m))),
  mlp(register_module("mlp", NandaMLP(c, m))),
  hook_attn_out(register_hook(this, "attn_out")),
  hook_mlp_out(register_hook(this, "mlp_out")),
  hook_resid_pre(register_hook(this, "resid_pre")),
  hook_resid_mid(register_hook(this, "resid_mid")),
  hook_resid_post(register_hook(this, "resid_post")) {}

  at::Tensor forward (at::Tensor x) {
    x = hook_resid_mid(x + hook_attn_out(attn(hook_resid_pre(x))));
    x = hook_resid_post(x + hook_mlp_out(mlp(x)));
    return x;
  }
  GPTConfig c;
  NandaAttention attn;
  NandaMLP mlp;
  HookPoint hook_attn_out, hook_mlp_out, hook_resid_pre, hook_resid_mid, hook_resid_post;
};

using NandaDecoderStackImpl = ModuleStack<NandaDecoderBlock>;
TORCH_MODULE(NandaDecoderStack);

struct TransformerImpl : torch::nn::Module {
  std::string name() { return "Transformer" + std::to_string(n); }
  TransformerImpl(GPTConfig c, int nnn = 0, int m = 1) : cfg(c), n(nnn), 
  embed(register_module("embed", Embed(cfg, m))), 
  pos_embed(register_module("pos_embed", PosEmbed(cfg, m))), 
  blocks(register_module("blocks", NandaDecoderStack(cfg, m))), 
  unembed(register_module("unembed", Unembed(cfg, m))), 
  hook_tokens(register_hook(this, "tokens")), 
  hook_embed(register_hook(this, "embed")), 
  hook_pos_embed(register_hook(this, "pos_embed")), 
  hook_logits(register_hook(this, "logits")) {}

  template<typename T> void importEnsemble(T* models, int num) { for (int i {}; i < num; ++i) importModel(models[i], i); }
  template<typename T> void importEnsemble(std::vector<T> models, int num) { for (int i {}; i < num; ++i) importModel(models[i], i); }

  template<typename T> void importModel(T& other, int i = 0) {
    auto state_dict = other->named_parameters();
    for (auto p : named_parameters())
      importParameter(p.key(), state_dict[p.key()], i);
  }

  void importParameter(const std::string& name, const torch::Tensor& tensor, int m = 0) {
    for (auto& param : named_parameters()) {
      if (param.key() == name) {
        auto p = param.value()[m];
        auto t { tensor };
        if (t.size(0) == 1) t = t.squeeze(0);
        p.copy_(t.detach().clone());
        p.set_requires_grad(false);
        break;
      }
    }
  }

  at::Tensor forward (const at::Tensor& x) {
    auto embeddings { hook_embed(embed(hook_tokens(x))) }, positions { hook_pos_embed(pos_embed(x)) };
    auto residual { blocks(embeddings + positions) };
    auto logits { hook_logits(unembed(residual)) };
    return logits;
  }

  std::pair<at::Tensor, ActivationCache> run_with_cache(const at::Tensor& x, Vec<std::string> h = {}) { 
    set_hooks(h); auto logits = forward(x); 
    return { logits, getActivationCache() }; }

  at::Tensor W_E() { return embed->W_E.slice(0, 0, -1); }
  at::Tensor W_neur() {
    constexpr auto mm { torch::matmul };
    return mm(mm(mm(W_E(), blocks->module->attn->W_V), blocks->module->attn->W_O), blocks->module->mlp->W_in);
  }
  at::Tensor W_logit() {
    constexpr auto mm { torch::matmul };
    return mm(blocks->module->mlp->W_out, unembed->W_U);
  }

  ActivationCache getActivationCache() {
    ActivationCache cache;
    Vec<std::shared_ptr<HookPointImpl>> hooks;
    for (const auto& item : named_modules("", false))
      if (item.key().find("hook_module_") != std::string::npos)
        hooks.push_back(std::dynamic_pointer_cast<HookPointImpl>(item.value()));
    std::sort(hooks.begin(), hooks.end(), [](auto l, auto r) { return l->time < r->time; });
    for (auto& hook : hooks)
      for (auto& p : hook->parent->named_parameters(false))
        if(p.key().find("hook_") != std::string::npos)
          for (const auto& param : named_parameters())
            if (param.key().find(p.key()) != std::string::npos && !cache.contains(param.key()))
              cache.data.push_back({ param.key(), param.value() });
    size_t idx = 0UL;
    for (const auto& param : named_parameters())
      if (param.key().find("hook_") == std::string::npos)
        cache.data.insert(cache.data.begin() + idx++, { param.key(), param.value() });
    return cache;
  }
  
  template<typename Container> ActivationCache getActivationCache(const Container& c) {
    auto cache { getActivationCache() };
    for(const auto& [name, t] : c) cache.data.push_back({ name, t });
    return cache;
  }

  void set_hooks(const Vec<std::string>& hook_names = Vec<std::string>()) {
    auto named { named_modules() };
    if (hook_names.size()) {
      for (auto name : hook_names) {
        if (name.find("hook_") != std::string::npos) {
          std::string moduleName { name.replace(name.find("hook_"), 5UL, "hook_module_") };
          auto hook_point = std::dynamic_pointer_cast<HookPointImpl>(named[moduleName]);
          hook_point->hook_added = true;
        }
      }
    }
    else {
      for (auto mod : named)
        if (mod.key().find("hook_module_") != std::string::npos)
          std::dynamic_pointer_cast<HookPointImpl>(mod.value())->hook_added = true;
    }
  }

  GPTConfig cfg;
  int p { 113 }, T { cfg["n_ctx"] }, C { cfg["d_model"] }, n { 0 };
  Embed embed;
  PosEmbed pos_embed;
  NandaDecoderStack blocks;
  Unembed unembed;
  HookPoint hook_tokens, hook_embed, hook_pos_embed, hook_logits;
};
TORCH_MODULE(Transformer);

// template<typename T = at::Tensor>
// T get_all_metrics(const T& cos_apb, const T& sin_apb, const T& LABELS, const T& thelogits, const T& W_out, const T& W_U, const T& post_mlp_act, const T& resid_mid_act) {
//   constexpr auto p = 113;
//   constexpr auto cutoff = 3830; // torch::tensor(int(113 * 113 * 0.3))
//   auto logits = thelogits.index({ "...", -1, Slice() });// [B,E,T,V]
//   auto labels = LABELS;// [B]
//   print("logits", logits.sizes(), "labels", labels.sizes());
//   auto neuron_acts = post_mlp_act.index({ "...", -1, Slice() });// [B,E,T,M]
  
//   auto resid_mid = resid_mid_act.index({ "...", -1, Slice() });
//   auto approx_neuron_acts = torch::zeros_like(neuron_acts);
//   print("approx_neuron_acts", approx_neuron_acts.sizes(), "neuron_acts", neuron_acts.sizes());
//   approx_neuron_acts += neuron_acts.mean(0);
  
//   // key_freqs = [17, 25, 32, 47]

//   for (int freq {}; freq < num_key_freqs; ++freq) {
//       approx_neuron_acts += (neuron_acts * cos_apb[freq]).sum(0) * cos_apb[freq];
//       approx_neuron_acts += (neuron_acts * sin_apb[freq]).sum(0) * sin_apb[freq];
//   }
  
//   // print("matmul sizes approx_neuron_acts, W_out", approx_neuron_acts.sizes(), W_out.sizes());
//   auto restricted_logits = torch::matmul(torch::matmul(approx_neuron_acts, W_out), W_U);
//   // print("restricted_logits", restricted_logits.sizes(), "logits.mean(0, true)", logits.mean(0, true).sizes(), "restricted_logits.mean(0, true)", restricted_logits.mean(0, true).sizes());
//   restricted_logits += logits.mean(0, true) - restricted_logits.mean(0, true);
//   auto excluded_neuron_acts = neuron_acts - approx_neuron_acts;
//   // print("torch::matmul(excluded_neuron_acts, W_out)", torch::matmul(excluded_neuron_acts, W_out).sizes(), "resid_mid", resid_mid.sizes());
//   // print("matmul sizes excluded_neuron_acts, W_out", excluded_neuron_acts.sizes(), W_out.sizes());
//   auto excluded_logits = torch::matmul(torch::matmul(excluded_neuron_acts, W_out) + resid_mid, W_U);

//   auto ret = torch::zeros({ 3, 2, 2 });
//   // acc = lambda predictions, truth: torch::mean((torch::argmax(predictions, -1) == truth).to(torch::kFloat));
//   // cehp=lambda lgz, lbz:-((lgz.index({ Slice(),-1 }).to(torch::kDouble).log_softmax(-1)).gather(-1, index=lbz.index({ Slice(), -1 }).long()).index({ Slice(), 0 })).mean();
//   print("indexing logits", logits.sizes(), "with .index({ \"...\", indices[\"train\"], Slice() })", indices["train"].sizes());
//   auto lgz = logits.index({ "...", indices["train"], Slice() });//.slice(0, 0, cutoff);
//   auto lbz = labels.index({ indices["train"] });//.slice(0, 0, cutoff);
//   print("lgz", lgz.sizes(), "lbz", lbz.sizes());

//   ret[0][0][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);//, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
//   ret[0][0][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
//   print("indexing logits", logits.sizes(), "with indices[test].unsqueeze(-2)", indices["test"].unsqueeze(-2).sizes());
//   lgz = logits.index({ "...", indices["test"], Slice() });
//   lbz = labels.index({ indices["test"] });
//   ret[0][1][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
//   ret[0][1][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
//   print("indexing restricted logits", restricted_logits.sizes(), "with indices[train].unsqueeze(-2)", indices["test"].unsqueeze(-2).sizes());
//   lgz = restricted_logits.index({ "...", indices["train"], Slice() });
//   lbz = labels.index({ indices["train"] });//.slice(0, 0, cutoff);
//   ret[1][0][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
//   ret[1][0][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
//   lgz = restricted_logits.index({ "...", indices["test"], Slice() });
//   lbz = labels.index({ indices["test"] });//.slice(0, cutoff, -1);
//   ret[1][1][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
//   ret[1][1][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));

//   lgz = excluded_logits.index({ "...", indices["train"], Slice() });
//   lbz = labels.index({ indices["train"] });
//   ret[2][0][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
//   ret[2][0][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
//   lgz = excluded_logits.index({ "...", indices["test"], Slice() });
//   lbz = labels.index({ indices["test"] });
//   ret[2][1][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
//   ret[2][1][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
//   print("returning ret from get_all_metrics");
//   return ret;
// }

struct Chex {
  static constexpr int p { 113 };
  void load_models();

  // template<int XXX=0> at::Tensor get_restricted_test_loss() {
  //   auto& model { models.back() };
  //   Vec<std::string> hook_names { { "hook_logits", "blocks.0.mlp.W_out", "unembed.W_U", "blocks.0.mlp.hook_post", "blocks.0.hook_resid_mid" } };
  //   auto [logits, cache] = model->run_with_cache(all_data, hook_names);
  //   std::vector<at::Tensor> input { trig.cos_apb, trig.sin_apb, all_labels };
  //   for (auto name : hook_names) input.push_back(cache[name]);
  //   auto all_metrics = get_all_metrics(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);
  //   return all_metrics[1][1][0];
  // }

  // template<int XXX=0> at::Tensor get_test_loss() {
  //   auto& model { models.back() };
  //   Vec<std::string> hook_names { { "hook_logits", "blocks.0.mlp.W_out", "unembed.W_U", "blocks.0.mlp.hook_post", "blocks.0.hook_resid_mid" } };
  //   auto [logits, cache] = model->run_with_cache(all_data, hook_names);
  //   std::vector<at::Tensor> input { trig.cos_apb, trig.sin_apb, all_labels };
  //   for (auto name : hook_names) input.push_back(cache[name]);
  //   auto all_metrics = get_all_metrics(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);
  //   return all_metrics[1][1][0];
  // }
  
  template<typename T = Transformer> ActivationCache getActivations(Vec<int> indexes, c10::Device dev, std::string mode, std::string name = "") {
    // if (name != "") print("getting activations for: " + name + "\n");
    T model { cfg, 0, indexes.size() };
    for (int i {}; i < indexes.size(); ++i)
      model->importModel(models[indexes[i]], i);
    model->set_hooks();
    model->to(dev);
    auto logits = model(all_data.index({ indices[mode] }).to(dev));
    auto cache = model->getActivationCache().detach();
    return cache;
  }

  template<typename T = at::Tensor> T logitsToMetrics(const T& logits) {
    // auto l0 = cross_entropy_high_precision(logits, all_labels).to(torch::kCPU);
    auto l1 = cross_entropy_high_precision(logits.index({ train_indices }), train_labels);
    auto l2 = cross_entropy_high_precision(logits.index({ test_indices }), test_labels);
    auto loss = torch::cat({ l1.unsqueeze(1), l2.unsqueeze(1) }, 1);
    // auto a0 = getAccuracy<torch::kDouble>(logits.index({ Slice(), Slice(), -1 }), all_labels);
    auto a1 = getAccuracy<torch::kDouble>(logits.index({ train_indices, Slice(), -1 }), train_labels);
    auto a2 = getAccuracy<torch::kDouble>(logits.index({ test_indices, Slice(), -1 }), test_labels);
    auto acc = torch::cat({ a1.unsqueeze(1), a2.unsqueeze(1) }, 1);
    auto metrics = torch::cat({ loss.unsqueeze(1), acc.unsqueeze(1) }, 1).to(torch::kCPU);
    return metrics;
  }

  template<typename T = at::Tensor>
  T get_all_metrics(const T& cos_apb, const T& sin_apb, const T& labels, const T& all_logits, const T& W_out, const T& W_U, const T& post_mlp_act, const T& resid_mid_act) {
    // auto logits = all_logits.index({ "...", Slice(-1,None), Slice() });// [B,E,T,V]
    // auto B = logits.size(0), M = logits.size(1);
    // // print("logits", logits.sizes(), "labels", labels.sizes());
    // auto neuron_acts = post_mlp_act.index({ "...", Slice(-1,None), Slice() });// [B,E,T,M]
    // auto resid_mid = resid_mid_act.index({ "...", Slice(-1,None), Slice() });
    // auto approx_neuron_acts = torch::zeros_like(neuron_acts);
    // approx_neuron_acts += neuron_acts.mean(0);
    // // print("approx_neuron_acts", approx_neuron_acts.sizes(), "neuron_acts", neuron_acts.sizes(), "cos_apb", cos_apb.sizes(), "cos_apb[0].unsqueeze(-1)", cos_apb[0].unsqueeze(-1).sizes());
    // // key_freqs = [17, 25, 32, 47]

    // //////////////////////////////////////////////////////////////////////////////////////////
    // // auto cos_apb_unsqueezed = cos_apb.unsqueeze(-1).unsqueeze(-1), sin_apb_unsqueezed = sin_apb.unsqueeze(-1).unsqueeze(-1);
    // // auto cos_acts_pre = (neuron_acts.unsqueeze(0) * cos_apb_unsqueezed).sum(1,true);
    // // auto sin_acts_pre = (neuron_acts.unsqueeze(0) * sin_apb_unsqueezed).sum(1,true);
    // // auto approx_neuron_acts = (cos_acts_pre * cos_apb_unsqueezed).sum(0);
    // // approx_neuron_acts += (sin_acts_pre * sin_apb_unsqueezed).sum(0);
    // // approx_neuron_acts += neuron_acts.mean(0);
    // //////////////////////////////////////////////////////////////////////////////////////////

    // for (int freq {}; freq < num_key_freqs; ++freq) {
    //   auto c = cos_apb[freq].unsqueeze(-1).unsqueeze(-1), s = sin_apb[freq].unsqueeze(-1).unsqueeze(-1);
    //   approx_neuron_acts += (neuron_acts * c).sum(0) * c;//[B M T N] += ([B M T N] * [B 1 1 1])
    //   approx_neuron_acts += (neuron_acts * s).sum(0) * s;
    // }

    // T restricted_logits, excluded_logits;
    
    // // print("matmul sizes approx_neuron_acts, W_out", approx_neuron_acts.sizes(), W_out.sizes(), W_U.sizes());
    // {
    //   auto restricted = einsad::einsum("batch model pos neur, model neur d_model -> batch model pos d_model", approx_neuron_acts, W_out);
    //   restricted_logits = einsad::einsum("batch model pos d_model, model d_model d_vocab -> batch model pos d_vocab", restricted, W_U);
    //   restricted_logits += logits.mean(0, true) - restricted_logits.mean(0, true);
    // }
    // {
    //   auto excluded_neuron_acts = neuron_acts - approx_neuron_acts;
    //   auto excluded = einsad::einsum("batch model pos neur, model neur d_model -> batch model pos d_model", excluded_neuron_acts, W_out) + resid_mid;
    //   excluded_logits = einsad::einsum("batch model pos d_model, model d_model d_vocab -> batch model pos d_vocab", excluded, W_U);
    // }
    // auto ret = torch::zeros({ 3, M, 2, 2 });
    // ret[0] = logitsToMetrics(logits);
    // ret[1] = logitsToMetrics(restricted_logits);
    // ret[2] = logitsToMetrics(excluded_logits);
    // ret = ret.transpose(0,1);
    // print("returning ret from get_all_metrics");
    // return ret;
    
    T restricted_logits, excluded_logits, logits = all_logits.index({ "...", Slice(-1,None), Slice() });// [B,E,T,V]
    auto B = logits.size(0), M = logits.size(1);
    // print("logits", logits.sizes(), "labels", labels.sizes());
    auto neuron_acts = post_mlp_act.index({ "...", Slice(-1,None), Slice() });// [B,E,T,M]
    auto resid_mid = resid_mid_act.index({ "...", Slice(-1,None), Slice() });
    // auto approx_neuron_acts = neuron_acts.mean(0,true).expand(neuron_acts.sizes());
    // print("approx_neuron_acts", approx_neuron_acts.sizes(), "neuron_acts", neuron_acts.sizes(), "cos_apb", cos_apb.sizes(), "cos_apb[0].unsqueeze(-1)", cos_apb[0].unsqueeze(-1).sizes());
    // key_freqs = [17, 25, 32, 47]

    //////////////////////////////////////////////////////////////////////////////////////////
    auto cos_apb_unsqueezed = cos_apb.unsqueeze(-1).unsqueeze(-1), sin_apb_unsqueezed = sin_apb.unsqueeze(-1).unsqueeze(-1);
    auto cos_acts_pre = (neuron_acts.unsqueeze(0) * cos_apb_unsqueezed).sum(1,true);
    auto sin_acts_pre = (neuron_acts.unsqueeze(0) * sin_apb_unsqueezed).sum(1,true);
    auto approx_neuron_acts = (cos_acts_pre * cos_apb_unsqueezed).sum(0);
    approx_neuron_acts += (sin_acts_pre * sin_apb_unsqueezed).sum(0) + neuron_acts.mean(0,true);
    //////////////////////////////////////////////////////////////////////////////////////////

    // for (int freq {}; freq < num_key_freqs; ++freq) {
    //   auto c = cos_apb[freq].unsqueeze(-1).unsqueeze(-1), s = sin_apb[freq].unsqueeze(-1).unsqueeze(-1);
    //   approx_neuron_acts += (neuron_acts * c).sum(0) * c;//[B M T N] += ([B M T N] * [B 1 1 1])
    //   approx_neuron_acts += (neuron_acts * s).sum(0) * s;
    // }
    
    // print("matmul sizes approx_neuron_acts, W_out", approx_neuron_acts.sizes(), W_out.sizes(), W_U.sizes());
    {
      auto restricted = einsad::einsum("batch model pos neur, model neur d_model -> batch model pos d_model", approx_neuron_acts, W_out);
      restricted_logits = einsad::einsum("batch model pos d_model, model d_model d_vocab -> batch model pos d_vocab", restricted, W_U);
      restricted_logits += logits.mean(0, true) - restricted_logits.mean(0, true);
    }
    {
      auto excluded_neuron_acts = neuron_acts - approx_neuron_acts;
      auto excluded = einsad::einsum("batch model pos neur, model neur d_model -> batch model pos d_model", excluded_neuron_acts, W_out) + resid_mid;
      excluded_logits = einsad::einsum("batch model pos d_model, model d_model d_vocab -> batch model pos d_vocab", excluded, W_U);
    }
    auto ret = torch::zeros({ 3, M, 2, 2 });
    ret[0] = logitsToMetrics(logits);
    ret[1] = logitsToMetrics(restricted_logits);
    ret[2] = logitsToMetrics(excluded_logits);
    ret = ret.transpose(0,1);
    // print("returning ret from get_all_metrics");
    return ret;
  }

  struct TrigData {
    template<typename T> void makeTrigData(const std::vector<T>& kf) {
      for (const auto& freq : kf) key_freqs.push_back(static_cast<int>(freq));
      constexpr long p { pModulo };
      auto a = torch::arange(p).index({ Slice(), None });
      auto b = torch::arange(p).index({ None, Slice() });
      for (const auto& freq : key_freqs) {
        auto cos_vec = torch::cos((freq * 2.0 * pifloat / p) * (a + b));
        cos_vec = cos_vec/cos_vec.norm();
        cos_vec = einsad::rearrange(cos_vec, "a b -> (a b) 1");
        auto sin_vec = torch::sin((freq * 2.0 * pifloat / p) * (a + b));
        sin_vec = sin_vec/sin_vec.norm();
        sin_vec = einsad::rearrange(sin_vec, "a b -> (a b) 1");
        cos_apb = (freq == key_freqs[0] ? cos_vec.unsqueeze(0) : torch::cat({ cos_apb, cos_vec.unsqueeze(0) }));
        sin_apb = (freq == key_freqs[0] ? sin_vec.unsqueeze(0) : torch::cat({ sin_apb, sin_vec.unsqueeze(0) }));
      }         
    }
    std::vector<int> key_freqs;
    at::Tensor cos_apb, sin_apb;
  };

  GPTConfig cfg {};
  c10::Device device { torch::cuda::is_available() ? torch::kCUDA : torch::kCPU };
  AllData allData { makeAllData() };
  TrigData trig;
  at::Tensor all_data { allData.dataset["all"].to(device) }, all_labels { allData.labels["all"].to(device) };
  at::Tensor train_indices { indices["train"].to(device) }, test_indices { indices["test"].to(device) };
  at::Tensor train_data { all_data.index({ train_indices }).contiguous() };
  at::Tensor test_data { all_data.index({ test_indices }).contiguous() };
  at::Tensor train_labels { all_labels.index({ train_indices }).contiguous() };
  at::Tensor test_labels { all_labels.index({ test_indices }).contiguous() };
  std::vector<Transformer> models;
  std::map<std::string, std::vector<int>> sizeMap;
};