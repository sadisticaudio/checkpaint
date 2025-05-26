#pragma once

#include "helperz.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include "math.h"
#include <torch/script.h>

using hi_res = std::chrono::high_resolution_clock;

template<typename T> T sadMatmul(T x1, T x2, std::string nm = "") { std::cout << nm + " " << x1.sizes() << " @ " << x2.sizes() << std::endl; return torch::matmul(x1,x2); }

template<int N = 0> size_t getCudaMemUsage() { size_t memTotal{}, memFree{}; cudaMemGetInfo(&memFree, &memTotal); size_t rv = memTotal - memFree; return rv; }
template<int N = 0> size_t getCudaMemTotal() { size_t memTotal{}, memFree{}; cudaMemGetInfo(&memFree, &memTotal); size_t rv = memTotal; return rv; }
template<int N = 0> size_t getCudaMemFree() { size_t memTotal{}, memFree{}; cudaMemGetInfo(&memFree, &memTotal); size_t rv = memFree; return rv; }

template<int N = 0> void printCudaMemUsage(std::string msg = std::string()) {
  auto toH = [](auto x) { return toHumanReadable(x); };
  auto statToH = [&](auto x)->std::string { return "current: (" + toH(x[0].current) + ") peak: (" + toH(x[0].peak) + ") allocated: (" + toH(x[0].allocated) + ")\n"; };
  auto statsToH = [&](auto s)->std::string { return "allocated_bytes: " + statToH(s.allocated_bytes) + "reserved_bytes: " + statToH(s.reserved_bytes) + "active_bytes: " + statToH(s.active_bytes) + "\n"; };
  auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  if(!msg.empty()) print(msg + ": ");
  print("CUDA Memory", toH(getCudaMemUsage()) + " / " + toH(getCudaMemTotal()));
  print("CUDA Stats", statsToH(stats));
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

  // auto a { torch::arange(p).unsqueeze(1).repeat({ 1, p }).flatten() };
  // auto b { torch::arange(p).unsqueeze(0).repeat({ p, 1 }).flatten() };
  // auto eq { torch::full({ p * p }, p) };

  dataset["all"] = torch::stack(at::TensorList({ a, b, eq }), 1);
  labels["all"] = (dataset["all"].index({ Slice(), 0 }) + dataset["all"].index({ Slice(), 1 })) % p;
  // dataset["all"] = dataset["all"].index({ indices["all"] });
  // labels["all"] = labels["all"].index({ indices["all"] });


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
  EmbedImpl(GPTConfig cc, int bbb = 1) : cfg(cc),
  W_E(register_parameter("W_E", torch::randn({ bbb, d_vocab, C })/std::sqrt(float(C)), false)) {}

  at::Tensor forward (at::Tensor x) {
    if (roughly_same(x.slice(-1, 0, 1), x.slice(-1, 2, 3))) print("embed is inputting all positions the same");
    // else print("embed input is ok");
    // print("x", x.sizes(), "W_E", W_E.sizes());
    auto flatTokens = x.flatten();//.unsqueeze(1).expand({x.size(0), W_E.size(0), x.size(1)}).flatten();
    // print("flatTokens", flatTokens.sizes());
    // print("index_select(-2, flatTokens)", W_E.index_select(-2, flatTokens).sizes());
    auto selected = W_E.index_select(-2, flatTokens);
    // print("selected", selected.sizes());
    x = selected.reshape({ W_E.size(0), x.size(0), x.size(1), C }).transpose(0,1);
    // print("embed output x", x.sizes());
    if (roughly_same(W_E.slice(-2, 0, 1), x.slice(-2, 2, 3))) print("embedding matrix positions arr all the same");
    if (roughly_same(x.slice(-2, 0, 1), x.slice(-2, 2, 3))) print("embed is outputting all positions the same");
    return x;
  }

  GPTConfig cfg;
  int C { cfg["d_model"] }, d_vocab { cfg["d_vocab"] };
  torch::Tensor W_E;
};
TORCH_MODULE(Embed);

struct UnembedImpl : torch::nn::Module {
  UnembedImpl(GPTConfig cc, int bbb = 1) : cfg(cc),
  W_U(register_parameter("W_U", torch::randn({ bbb, C, d_vocab - 1 })/std::sqrt(float(d_vocab)), false)) {}

  // at::Tensor forward (at::Tensor x) { 
  //   // print("matmul sizes x, W_U", x.sizes(), W_U.sizes());
  //   auto w = //(x.dim() == 4) ? W_U.unsqueeze(-3) : 
  //   W_U;
  //   // print("matmul sizes x, W_U, w", x.sizes(), W_U.sizes(), w.sizes());
  //   return torch::matmul(x, w);
  // }

  at::Tensor forward (at::Tensor x) { 
    auto ret = sadMatmul(x, W_U, "unembed");
    // print("x", x.sizes(), "W_U", W_U.sizes(), "ret", ret.sizes());
    return ret;//.view({ x.size(0), x.size(1), x.size(2), W_U.size(-1) }); 
    
  }

  GPTConfig cfg;
  int C { cfg["d_model"] }, d_vocab { cfg["d_vocab"] };
  torch::Tensor W_U;
};
TORCH_MODULE(Unembed);

struct PosEmbedImpl : torch::nn::Module {
  PosEmbedImpl(GPTConfig cc, int bbb = 1) : cfg(cc), 
  W_pos(register_parameter("W_pos", torch::randn({ bbb, T, C })/std::sqrt(float(C)), false)) {}

  at::Tensor forward (at::Tensor x) {
    return einsad::repeat(W_pos, "mod pos d_model -> batch mod pos d_model", { "batch", x.size(0) }); }

  GPTConfig cfg;
  int T { cfg["n_ctx"] }, C { cfg["d_model"] };
  torch::Tensor W_pos;
};
TORCH_MODULE(PosEmbed);

using TArray = std::array<std::array<at::Tensor, 3>,4>;

struct NandaAttentionImpl : torch::nn::Module {
  NandaAttentionImpl(GPTConfig cc, int bbb = 1) : cfg(cc),
  W_K(register_parameter("W_K", torch::randn({ bbb, nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_Q(register_parameter("W_Q", torch::randn({ bbb, nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_V(register_parameter("W_V", torch::randn({ bbb, nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_O(register_parameter("W_O", torch::randn({ bbb, nh, C/nh, C })/std::sqrt(float(C)), false)),
  hook_k(register_hook(this, "k")),
  hook_q(register_hook(this, "q")),
  hook_v(register_hook(this, "v")),
  hook_z(register_hook(this, "z")),
  hook_pattern(register_hook(this, "pattern")),
  hook_attn_scores(register_hook(this, "attn_scores")) {}

  torch::Tensor forwardOLD (torch::Tensor x) {

    auto B = x.size(0), M = x.size(1);

    if (roughly_same(x.slice(-2, 0, 1), x.slice(-2, 2, 3))) print("attention is inputting all positions the same");
    auto q = hook_q(einsad::einsum("m i d h, b m p d -> b m p i h", W_Q, x));// [nh C hs] @ [B T C] = [B T nh hs]
    print("einsad::einsum(\"m i d h, b m p d -> b m p i h\", W_Q, x))", W_Q.sizes(), x.sizes(), "->", q.sizes());
    auto k = hook_k(einsad::einsum("m i d h, b m p d -> b m p i h", W_K, x));// [nh C hs] @ [B T C] = [B T nh hs]
    auto v = hook_v(einsad::einsum("m i d h, b m p d -> b m p i h", W_V, x));// [nh C hs] @ [B T C] = [B T nh hs]
    q = einsad::rearrange(q, "b m p i h -> b p (m i) h");
    k = einsad::rearrange(k, "b m p i h -> b p (m i) h");
    v = einsad::rearrange(v, "b m p i h -> b p (m i) h");
    q = q.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    k = k.transpose(1,2).transpose(-2, -1); // [B T nh hs] -> [B nh T hs] -> [B nh hs T]
    v = v.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    auto attn_scores_pre = sadMatmul(q, k, "q k fwdOld")/std::sqrt(float(hs));// [B nh T hs] @ [B nh hs T] = [B nh T T]
    print("torch::matmul(q, k)/std::sqrt(float(hs))", q.sizes(), k.sizes(), "->", attn_scores_pre.sizes());
    auto attn_scores_masked = attn_scores_pre + torch::full({ 1, T, T }, nInf).triu(1).to(k.device());// [B nh T T]
    auto pattern = hook_pattern(torch::nn::functional::softmax(hook_attn_scores(attn_scores_masked), -1));// [B nh T T]

    if (x.size(0) < 10000) print("\n\n\n\n\nLISTEN\n(new tformer pattern)", "this is where things diverge, this should be the same", pattern.sizes(), get_first_elements(pattern, 4));
    auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    z = einsad::rearrange(z, "batch (model head) query_pos d_head -> batch model query_pos (head d_head)", { "model", x.size(1) }, { "head", nh });
    if (x.size(0) < 10000) print("(new tformer z)", z.sizes(), get_first_elements(z, 4));
    auto w = einsad::rearrange(W_O, "model head_index d_head d_model -> model d_model (head_index d_head)");
    if (x.size(0) < 10000) print("(new tformer w)", w.sizes(), get_first_elements(w, 4));
    at::Tensor out;
    if(w.dim() == 3 && w.size(0) == 1)
      out = torch::nn::functional::linear(z.reshape({ B, M, T, C }), w.squeeze());
    else out = sadMatmul(z.reshape({ B, M, T, C }), w.unsqueeze(0), "z w fwdOld");
    out = out.reshape({ x.sizes() });
    if (roughly_same(out.slice(-2, 0, 1), out.slice(-2, 2, 3))) print("attention is outputting all positions the same");
    return out;
  }

  TArray forwardSingle (torch::Tensor x) {
    x = x.squeeze();
    print("in forwardSingle");
    TArray ret;
    auto q = hook_q(torch::einsum("idh,bpd->bpih", { W_Q.squeeze(), x }));// [nh C hs] @ [B T C] = [B T nh hs]
    ret[0] = { W_Q.squeeze(), x, q };
    auto k = hook_k(torch::einsum("idh,bpd->bpih", { W_K.squeeze(), x }));// [nh C hs] @ [B T C] = [B T nh hs]
    auto v = hook_v(torch::einsum("idh,bpd->bpih", { W_V.squeeze(), x }));// [nh C hs] @ [B T C] = [B T nh hs]
    q = q.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    k = k.transpose(1,2).transpose(-2, -1); // [B T nh hs] -> [B nh T hs] -> [B nh hs T]
    v = v.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    auto attn_scores_pre = sadMatmul(q, k, "q k fwdSingle")/std::sqrt(float(hs));// [B nh T hs] @ [B nh hs T] = [B nh T T]
    ret[1] = { q, k, attn_scores_pre };
    auto attn_scores_masked = attn_scores_pre + torch::full({ 1, T, T }, nInf).triu(1).to(k.device());// [B nh T T]
    auto pattern = hook_pattern(torch::nn::functional::softmax(hook_attn_scores(attn_scores_masked), -1));// [B nh T T]
    auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    

    z = hook_z(einsad::rearrange(z, "batch head_index query_pos d_head -> batch query_pos head_index d_head"));
    ret[2] = { v, pattern, z };
    auto w = einsad::rearrange(W_O.squeeze(), "head_index d_head d_model -> d_model (head_index d_head)");
    z = z.reshape({ z.size(0), z.size(1), C });
    auto out = torch::nn::functional::linear(z, w);

    ret[3] = { z, w, out };
    return ret;
  }

  TArray forwardDouble (torch::Tensor x) {
    print("in forwardDouble");
    TArray ret;
    auto B = x.size(0), M = x.size(1);
    auto q = hook_q(einsad::einsum("m i d h, b m p d -> b m p i h", W_Q, x));// [nh C hs] @ [B T C] = [B T nh hs]
    ////////////////////////////////////////////////
    ret[0] = { W_Q, x, q };
    auto k = hook_k(einsad::einsum("m i d h, b m p d -> b m p i h", W_K, x));// [nh C hs] @ [B T C] = [B T nh hs]
    auto v = hook_v(einsad::einsum("m i d h, b m p d -> b m p i h", W_V, x));// [nh C hs] @ [B T C] = [B T nh hs]
    q = einsad::rearrange(q, "b m p i h -> b p (m i) h");
    k = einsad::rearrange(k, "b m p i h -> b p (m i) h");
    v = einsad::rearrange(v, "b m p i h -> b p (m i) h");
    q = q.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    k = k.transpose(1,2).transpose(-2, -1); // [B T nh hs] -> [B nh T hs] -> [B nh hs T]
    v = v.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    auto attn_scores_pre = sadMatmul(q, k, "q k fwdDouble")/std::sqrt(float(hs));// [B nh T hs] @ [B nh hs T] = [B nh T T]
    //////////////////////////////////////////////////
    ret[1] = { q, k, attn_scores_pre.view({ B,M,nh,T,T }) };
    // ret.push_back(attn_scores_pre.view({ B, M, nh, T, T }));
    auto attn_scores_masked = attn_scores_pre + torch::full({ 1, T, T }, nInf).triu(1).to(k.device());// [B nh T T]
    auto pattern = hook_pattern(torch::nn::functional::softmax(hook_attn_scores(attn_scores_masked), -1));// [B nh T T]
    auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    
    ////////////////////////////////////////////////////
    
    z = einsad::rearrange(z, "batch modelhead query_pos d_head -> batch query_pos modelhead d_head");
    // z = ;//, { "model", x.size(1) }, { "head", nh }));
    ret[2] = { v, pattern, hook_z(einsad::rearrange(z, "batch pos (model head) d_head -> batch model pos head d_head", { "model", x.size(1) }, { "head", nh })) };
    // auto w = einsad::rearrange(W_O, "model head_index d_head d_model -> model d_model (head_index d_head)");
    auto w = einsad::rearrange(W_O, "model head d_head d_model -> d_model (model head) d_head");
    // at::Tensor out;
    // if(w.dim() == 3 && w.size(0) == 1)
    //   out = torch::nn::functional::linear(z = z.reshape({ B, M, T, C }), w = w.squeeze());
    // else 
    // z = z.reshape({ z.size(0), z.size(1), z.size(2), C });
    // w = w.unsqueeze(0);
    print("z,w", z.sizes(), w.sizes());
    // auto out = torch::matmul(z,w);// = z.reshape({ B, M, T, C }), w = w.unsqueeze(0));
    auto out = einsad::einsum("batch pos modelhead d_head, d_model modelhead d_head -> batch pos modelhead d_model", z, w);// [batch, pos, head, d_model]
    out = einsad::rearrange(out, "batch pos (model head) d_model -> batch model pos head d_model", { "model", x.size(1) }, { "head", nh }).sum(-2);
    
    // z = hook_z(einsad::rearrange(z, "batch head_index query_pos d_head -> batch query_pos head_index d_head"));
    // auto w = einsad::rearrange(W_O.squeeze(), "head_index d_head d_model -> d_model (head_index d_head)");
    // z = z.reshape({ z.size(0), z.size(1), C });
    // auto out = torch::nn::functional::linear(z, w);

    ///////////////////////////////////////////////////////
    ret[3] = { z, w, out };
    return ret;
  }

  torch::Tensor forward (torch::Tensor x) {
    // print("in forwardNEW");
    // TArray ret;
    auto B = x.size(0), M = x.size(1);
    auto q = hook_q(einsad::einsum("m i d h, b m p d -> b m p i h", W_Q, x));// [nh C hs] @ [B T C] = [B T nh hs]
    // ret[0] = { W_Q, x, q };
    auto k = hook_k(einsad::einsum("m i d h, b m p d -> b m p i h", W_K, x));// [nh C hs] @ [B T C] = [B T nh hs]
    auto v = hook_v(einsad::einsum("m i d h, b m p d -> b m p i h", W_V, x));// [nh C hs] @ [B T C] = [B T nh hs]
    q = einsad::rearrange(q, "b m p i h -> b p (m i) h");
    k = einsad::rearrange(k, "b m p i h -> b p (m i) h");
    v = einsad::rearrange(v, "b m p i h -> b p (m i) h");
    q = q.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    k = k.transpose(1,2).transpose(-2, -1); // [B T nh hs] -> [B nh T hs] -> [B nh hs T]
    v = v.transpose(1,2);// [B T nh hs] -> [B nh T hs]
    auto attn_scores_pre = sadMatmul(q, k, "q k fwd")/std::sqrt(float(hs));// [B nh T hs] @ [B nh hs T] = [B nh T T]
    // ret[1] = { q, k, attn_scores_pre.view({ B,M,nh,T,T }) };
    auto attn_scores_masked = attn_scores_pre + torch::full({ 1, T, T }, nInf).triu(1).to(k.device());// [B nh T T]
    auto pattern = hook_pattern(torch::nn::functional::softmax(hook_attn_scores(attn_scores_masked), -1));// [B nh T T]
    auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    
    z = einsad::rearrange(z, "batch modelhead query_pos d_head -> batch query_pos modelhead d_head");
    // ret[2] = { v, pattern, hook_z(einsad::rearrange(z, "batch pos (model head) d_head -> batch model pos head d_head", { "model", x.size(1) }, { "head", nh })) };
    auto w = einsad::rearrange(W_O, "model head d_head d_model -> d_model (model head) d_head");
    // print("z,w", z.sizes(), w.sizes());
    auto out = einsad::einsum("batch pos modelhead d_head, d_model modelhead d_head -> batch pos modelhead d_model", z, w);// [batch, pos, head, d_model]
    out = einsad::rearrange(out, "batch pos (model head) d_model -> batch model pos head d_model", { "model", x.size(1) }, { "head", nh }).sum(-2);
    // ret[3] = { z, w, out };
    return out;
  }
  
  GPTConfig cfg;
  int nh { cfg["n_heads"] }, T { cfg["n_ctx"] }, C { cfg["d_model"] }, hs { C/nh };
  torch::Tensor W_K, W_Q, W_V, W_O;
  HookPoint hook_k, hook_q, hook_v, hook_z, hook_pattern, hook_attn_scores;
};
TORCH_MODULE(NandaAttention);

struct NandaMLPImpl : torch::nn::Module {
  NandaMLPImpl(GPTConfig cfg, int bbb = 1) : c(cfg),
  W_in(register_parameter("W_in", torch::randn({ bbb, C, d_mlp })/std::sqrt(float(C)), false)),
  reLU(register_module("reLU", torch::nn::ReLU(torch::nn::ReLUOptions()))),
  W_out(register_parameter("W_out", torch::randn({ bbb, d_mlp, C })/std::sqrt(float(C)), false)),
  hook_pre(register_hook(this, "pre")),
  hook_post(register_hook(this, "post")) {}

  torch::Tensor forward(torch::Tensor x) {
    std::vector<at::Tensor> vec { W_in.size(0), at::Tensor() };
    // x = sadMatmul(x, W_in, "mlp x W_in");
    for (int i {}; i < W_in.size(0); ++i) vec[i] = sadMatmul(x.index({ Slice(), i }), W_in[i], "mlp x W_in");
    x = torch::stack(at::TensorList(vec), 1);
    x = hook_post(reLU(hook_pre(x)));
    // x = sadMatmul(x, W_out, "mlp x W_out");
    for (int i {}; i < W_out.size(0); ++i) vec[i] = sadMatmul(x.index({ Slice(), i }), W_out[i], "mlp x W_out");
    x = torch::stack(at::TensorList(vec), 1);
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
  NandaDecoderBlock(GPTConfig cfg, int bbb = 1) : c(cfg),
  attn(register_module("attn", NandaAttention(c, bbb))),
  mlp(register_module("mlp", NandaMLP(c, bbb))),
  hook_attn_out(register_hook(this, "attn_out")),
  hook_mlp_out(register_hook(this, "mlp_out")),
  hook_resid_pre(register_hook(this, "resid_pre")),
  hook_resid_mid(register_hook(this, "resid_mid")),
  hook_resid_post(register_hook(this, "resid_post")) {}

  at::Tensor forward (at::Tensor x) {
    if (roughly_same(x.slice(-2, 0, 1), x.slice(-2, 2, 3))) print("block is inputting all positions the same");
    // print("in block");
    x = hook_resid_mid(x + hook_attn_out(attn(hook_resid_pre(x))));
    print("x + mlp_out", x.sizes(), mlp(x).sizes());
    x = hook_resid_post(x + hook_mlp_out(mlp(x)));
    if (roughly_same(x.slice(-2, 0, 1), x.slice(-2, 2, 3))) print("block is outputting all positions the same");
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
  TransformerImpl(GPTConfig c, int nnn = 0, int bbb = 1) : cfg(c), n(nnn), 
  embed(register_module("embed", Embed(cfg, bbb))), 
  pos_embed(register_module("pos_embed", PosEmbed(cfg, bbb))), 
  blocks(register_module("blocks", NandaDecoderStack(cfg, bbb))), 
  unembed(register_module("unembed", Unembed(cfg, bbb))), 
  hook_tokens(register_hook(this, "tokens")), 
  hook_embed(register_hook(this, "embed")), 
  hook_pos_embed(register_hook(this, "pos_embed")), 
  hook_logits(register_hook(this, "logits")) { //if (n == 0) print("in Transformer ctr"); 
  }

  template<typename T> void importEnsemble(T* models, int num) { for (int i {}; i < num; ++i) importModel(models[i], i); }
  template<typename T>  void importEnsemble(std::vector<T> models, int num) { for (int i {}; i < num; ++i) importModel(models[i], i); }

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
    auto residual = blocks(embeddings + positions);
    auto logits = hook_logits(unembed(residual));
    return logits;
  }

  TArray forwardSingle (const at::Tensor& x) {
    auto embeddings { hook_embed(embed(x)) }, positions { hook_pos_embed(pos_embed(x)) };
    return blocks->module->attn->forwardSingle(embeddings + positions); }
  TArray forwardDouble (const at::Tensor& x) {
    auto embeddings { hook_embed(embed(x)) }, positions { hook_pos_embed(pos_embed(x)) };
    return blocks->module->attn->forwardDouble(embeddings + positions); }
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
  
  Vec<torch::Tensor> computeLoss(Vec<torch::Tensor> tv) {
    // print("tv[0] (loss) / tv.back() (trg)", tv[0].sizes(), tv.back().sizes());
    // tv[0] = getCrossEntropy(tv[0].slice(1, 2, 3).view({ -1, tv[0].size(-1) }), tv.back().view({ -1 }));//      .squeeze().to(torch::kDouble), tv.back());
    return tv;
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

template<typename T = at::Tensor>
  T get_all_metrics2(const T& cos_apb, const T& sin_apb, const T& LABELS, const T& thelogits, const T& W_out, const T& W_U, const T& post_mlp_act, const T& resid_mid_act) {
    auto logits = getLastLogit(thelogits);
    auto neuron_acts { post_mlp_act.index({ Slice(), -1, Slice() }) };
    auto resid_mid { resid_mid_act.index({ Slice(), -1, Slice() }) };
    auto approx_neuron_acts = torch::zeros_like(neuron_acts);
    approx_neuron_acts += neuron_acts.mean(0);

    for (int freq {}; freq < num_key_freqs; ++freq) {
      //freq is index here... actual freq is in key_freqs[freq]
      approx_neuron_acts += (neuron_acts * cos_apb[freq]).sum(0) * cos_apb[freq];
      approx_neuron_acts += (neuron_acts * sin_apb[freq]).sum(0) * sin_apb[freq];
    }
    // print("matmul sizes x, W_in", x.sizes(), W_in.sizes());
    auto restricted_logits = torch::matmul(torch::matmul(approx_neuron_acts, W_out), W_U);
    restricted_logits += logits.mean(0, true) - restricted_logits.mean(0, true);
    auto excluded_neuron_acts = neuron_acts - approx_neuron_acts;
    // print("matmul sizes x, W_in", x.sizes(), W_in.sizes());
    auto excluded_logits = torch::matmul(torch::matmul(excluded_neuron_acts, W_out) + resid_mid, W_U);
    
    auto ret { torch::zeros({ 3, 2, 2 }, logits.device()) };
    auto getMetric = [&ret, &LABELS, idx = 0](const T& lgtz) mutable { 
      ret[idx][0][0] = cross_entropy_high_precision(getMode(lgtz, "train"), getMode(LABELS, "train"));
      ret[idx][0][1] = getAccuracy(getMode(lgtz, "train"), getMode(LABELS, "train"));
      ret[idx][1][0] = cross_entropy_high_precision(getMode(lgtz, "test"), getMode(LABELS, "test"));
      // std::cout << "idx before: " << idx << ", idx after: ";
      ret[idx++][1][1] = getAccuracy(getMode(lgtz, "test"), getMode(LABELS, "test"));
      // std::cout << idx << std::endl;
    };
    getMetric(logits);
    getMetric(restricted_logits);
    getMetric(excluded_logits);
    return ret;
  }

  template<typename T = at::Tensor>
  T get_all_metrics(const T& cos_apb, const T& sin_apb, const T& LABELS, const T& thelogits, const T& W_out, const T& W_U, const T& post_mlp_act, const T& resid_mid_act) {
    constexpr auto p = 113;
    constexpr auto cutoff = 3830; // torch::tensor(int(113 * 113 * 0.3))
    constexpr auto p_squared = p*p;
    auto logits = thelogits.index({ "...", -1, Slice() });// [B,E,T,V]
    auto labels = LABELS;// [B]
    print("logits", logits.sizes(), "labels", labels.sizes());
    auto neuron_acts = post_mlp_act.index({ "...", -1, Slice() });// [B,E,T,M]
    
    auto resid_mid = resid_mid_act.index({ "...", -1, Slice() });
    auto approx_neuron_acts = torch::zeros_like(neuron_acts);
    print("approx_neuron_acts", approx_neuron_acts.sizes(), "neuron_acts", neuron_acts.sizes());
    approx_neuron_acts += neuron_acts.mean(0);
    
    // key_freqs = [17, 25, 32, 47]

    for (int freq {}; freq < num_key_freqs; ++freq) {
        approx_neuron_acts += (neuron_acts * cos_apb[freq]).sum(0) * cos_apb[freq];
        approx_neuron_acts += (neuron_acts * sin_apb[freq]).sum(0) * sin_apb[freq];
    }
    
    // print("matmul sizes approx_neuron_acts, W_out", approx_neuron_acts.sizes(), W_out.sizes());
    auto restricted_logits = torch::matmul(torch::matmul(approx_neuron_acts, W_out), W_U);
    // print("restricted_logits", restricted_logits.sizes(), "logits.mean(0, true)", logits.mean(0, true).sizes(), "restricted_logits.mean(0, true)", restricted_logits.mean(0, true).sizes());
    restricted_logits += logits.mean(0, true) - restricted_logits.mean(0, true);
    auto excluded_neuron_acts = neuron_acts - approx_neuron_acts;
    // print("torch::matmul(excluded_neuron_acts, W_out)", torch::matmul(excluded_neuron_acts, W_out).sizes(), "resid_mid", resid_mid.sizes());
    // print("matmul sizes excluded_neuron_acts, W_out", excluded_neuron_acts.sizes(), W_out.sizes());
    auto excluded_logits = torch::matmul(torch::matmul(excluded_neuron_acts, W_out) + resid_mid, W_U);

    auto ret = torch::zeros({ 3, 2, 2 });
    // acc = lambda predictions, truth: torch::mean((torch::argmax(predictions, -1) == truth).to(torch::kFloat));
    // cehp=lambda lgz, lbz:-((lgz.index({ Slice(),-1 }).to(torch::kDouble).log_softmax(-1)).gather(-1, index=lbz.index({ Slice(), -1 }).long()).index({ Slice(), 0 })).mean();
    print("indexing logits", logits.sizes(), "with .index({ \"...\", indices[\"train\"], Slice() })", indices["train"].sizes());
    auto lgz = logits.index({ "...", indices["train"], Slice() });//.slice(0, 0, cutoff);
    auto lbz = labels.index({ indices["train"] });//.slice(0, 0, cutoff);
    print("lgz", lgz.sizes(), "lbz", lbz.sizes());

    // if len(logits.shape)==3:
    //     logits = logits[:, -1]
    // auto xlogits = logits.to(torch::kDouble);
    // auto log_probs = xlogits.log_softmax(-1);
    // auto correct_log_probs = log_probs.gather(-1, labels.index({ Slice(), None })).index({ Slice(), 0 });
    // ret[0][0][0] = -correct_log_probs.mean();

    // ret[0][0][0] = -((lgz.to(torch::kDouble).log_softmax(-1)).gather(-1, lbz)).mean();
    //torch::nn::functional::cross_entropy(lgz.to(torch::kDouble).view({ -1, lgz.size(0) }), lbz);
    
    ret[0][0][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);//, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
    ret[0][0][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
    print("indexing logits", logits.sizes(), "with indices[test].unsqueeze(-2)", indices["test"].unsqueeze(-2).sizes());
    lgz = logits.index({ "...", indices["test"], Slice() });
    lbz = labels.index({ indices["test"] });
    ret[0][1][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
    ret[0][1][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
    print("indexing restricted logits", restricted_logits.sizes(), "with indices[train].unsqueeze(-2)", indices["test"].unsqueeze(-2).sizes());
    lgz = restricted_logits.index({ "...", indices["train"], Slice() });
    lbz = labels.index({ indices["train"] });//.slice(0, 0, cutoff);
    ret[1][0][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
    ret[1][0][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
    lgz = restricted_logits.index({ "...", indices["test"], Slice() });
    lbz = labels.index({ indices["test"] });//.slice(0, cutoff, -1);
    ret[1][1][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
    ret[1][1][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
    // print("neuron_acts", neuron_acts[0][0].item());
    // print("approx_neuron_acts", approx_neuron_acts[0][0].item());
    // print("model.blocks[0].mlp.W_out", W_out[0][0].item());
    // print("model.unembed.W_U", W_U[0][0].item());
    // print("cos_apb_vec", cos_apb[3][0][0].item());
    // print("restricted_logits", restricted_logits[0][0].item());
    // print("rv", ret[1][1][0].item());
    lgz = excluded_logits.index({ "...", indices["train"], Slice() });
    lbz = labels.index({ indices["train"] });
    ret[2][0][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
    ret[2][0][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
    lgz = excluded_logits.index({ "...", indices["test"], Slice() });
    lbz = labels.index({ indices["test"] });
    ret[2][1][0] = torch::nn::functional::cross_entropy(lgz.view({ -1, lgz.size(-1) }).to(torch::kDouble), lbz);
    ret[2][1][1] = torch::mean((torch::argmax(lgz, -1) == lbz).to(torch::kFloat));
    print("returning ret from get_all_metrics");
    return ret;
  }

inline std::vector<c10::IValue> cloneIValues(const std::vector<c10::IValue>& iValues) {
  std::vector<c10::IValue> ret;
  for (const auto& x : iValues) {
    auto& tensor = x.toTensor();
    if (tensor.numel())
      ret.push_back(tensor.detach().clone());
    else ret.push_back(at::Tensor());
  }
  return ret;
}

template<typename M>
std::vector<M> emplace_models(torch::nn::Module* model, M* mBegin, M* mEnd) {
  int i { 0 };
  for (M* m { mBegin }; m != mEnd; ++m) model->register_module(m->ptr()->name() + std::to_string(i), *m);
  return { mBegin, mEnd };
}

struct BatchThreadOldImpl : torch::nn::Module {
  BatchThreadOldImpl(Transformer* m, 
                  int num,
                  const torch::jit::script::Module& sm,
                  const std::vector<c10::IValue>& ins,
                  const std::vector<std::string>& nms) : 
                    models(emplace_models(this, m, m + num)),
                    scriptModule(sm.clone()),
                    inputs(cloneIValues(ins)),
                    hook_names(nms) {
    // print("hook_names", hook_names);
    for (auto& model : models) model->set_hooks(hook_names);
                    }

  at::Tensor forward(const at::Tensor& input) {
    size_t sz { models.size() };
    std::vector<at::Tensor> outputs;
    outputs.resize(sz);
    for (size_t m {}; m < sz; ++m) {
      // do we need this???
      auto newInput = input.clone();
      std::cout << "(" << m << "/" << sz << ") ";
      models[m]->to(newInput.device());


      auto spectrum = torch::fft::fft(models[m]->embed->W_E);
      print("spectrum.sizes()", spectrum.sizes());
      std::terminate();
      float amp[113]{};
      // for (int i{}; i < 113; ++i) amp[i] = 


      models[m](newInput);
      const auto named = models[m]->named_parameters();
      // for (const auto& x : named) print("named params", x.key(), x.value().sizes());
      for (size_t i { 0 }, start { inputs.size() - hook_names.size() }; i < hook_names.size(); ++i)
        inputs[start + i] = named[hook_names[i]];
      // auto& in3 = inputs[3].toTensor();
      // auto& in6 = inputs[6].toTensor();
      // auto& in7 = inputs[7].toTensor();
      // in3 = in3.slice(1, in3.size(1) - 1, in3.size(1));
      // in6 = in6.slice(1, in6.size(1) - 1, in6.size(1));
      // in7 = in7.slice(1, in7.size(1) - 1, in7.size(1));


      // print("inputs[0], inputs[3], inputs[7] sizes", inputs[0].toTensor().sizes(), inputs[3].toTensor().sizes(), inputs[7].toTensor().sizes());
      

      outputs[m] = get_all_metrics( inputs[0].toTensor(), 
                                    inputs[1].toTensor(), 
                                    inputs[2].toTensor(), 
                                    inputs[3].toTensor(), 
                                    inputs[4].toTensor(), 
                                    inputs[5].toTensor(), 
                                    inputs[6].toTensor(), 
                                    inputs[7].toTensor() 
                                    ).to(torch::kCPU, false);

      // scriptModule.eval();
      // auto iValue = scriptModule.forward(inputs);
      // outputs[m] = iValue.toTensor().clone().to(torch::kCPU, false);

      models[m]->to(torch::kCPU, false);
    }
    // print("outputs size [0].sizes()", outputs.size(), outputs[0].sizes());
    return torch::stack(at::TensorList(outputs)).clone().to(torch::kCPU);
  }
  std::vector<Transformer> models;
  torch::jit::script::Module scriptModule;
  std::vector<c10::IValue> inputs;
  const std::vector<std::string> hook_names;
};
TORCH_MODULE(BatchThreadOld);

struct BatchThreadImpl : torch::nn::Module {
  BatchThreadImpl(Transformer* m, 
                  int num,
                  const torch::jit::script::Module& sm,
                  const std::vector<c10::IValue>& ins,
                  const std::vector<std::string>& nms) : 
                    // scriptModule(sm.clone()),
                    inputs(cloneIValues(ins)),
                    hook_names(nms) {
    for (int i {}; i < num; ++i) {
      models.push_back(Transformer((*m)->cfg, i));
      // models[i]->importEnsemble(m, num);
      models[i]->set_hooks(hook_names);
      models[i]->to(inputs[0].toTensor().device());
    }
    for (auto& model : models) model->set_hooks(hook_names);
                    }

  at::Tensor forward(const at::Tensor& input) {
    size_t sz { models.size() };
    std::vector<at::Tensor> outputs;
    outputs.resize(sz);
    for (size_t m {}; m < sz; ++m) {
      // do we need this???
      auto newInput = input.clone();
      std::cout << "(" << m << "/" << sz << ") ";
      models[m]->to(newInput.device());


      auto spectrum = torch::fft::rfftn(models[m]->embed->W_E.transpose(-2,-1).slice(-1, 0, 113));
      print("spectrum.sizes()", spectrum.sizes());
      spectrum = torch::abs(spectrum);
      spectrum = spectrum.sum({-3, -2});
      // spectrum = spectrum.view(spectrum.size(0), -1);
      auto [min, other1] = spectrum.min(-1, true);
      spectrum -= min;
      auto [max, other2] = spectrum.max(-1, true);
      spectrum /= max;
      spectrum = spectrum * 19.f;
      // spectrum = spectrum.reshape({ spectrum.size(0), 128, 113 });
      print("spectrum.sizes()", spectrum.sizes());
      for (int j = 0; j < 113; ++j) {

      }
      std::cout << "\n";
      std::cout << "\n";
      for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 57; ++j) {
          if(spectrum[j].item<int>() == i) std::cout << "X";
          else std::cout << " ";

        }
        std::cout << "\n";
      }
      std::cout << std::endl;
      // std::terminate();
      float amp[113]{};
      // for (int i{}; i < 113; ++i) amp[i] = 


      models[m](newInput);
      const auto named = models[m]->named_parameters();
      // for (const auto& x : named) print("named params", x.key(), x.value().sizes());
      for (size_t i { 0 }, start { inputs.size() - hook_names.size() }; i < hook_names.size(); ++i)
        inputs[start + i] = named[hook_names[i]];
      // auto& in3 = inputs[3].toTensor();
      // auto& in6 = inputs[6].toTensor();
      // auto& in7 = inputs[7].toTensor();
      // in3 = in3.slice(1, in3.size(1) - 1, in3.size(1));
      // in6 = in6.slice(1, in6.size(1) - 1, in6.size(1));
      // in7 = in7.slice(1, in7.size(1) - 1, in7.size(1));


      // print("inputs[0], inputs[3], inputs[7] sizes", inputs[0].toTensor().sizes(), inputs[3].toTensor().sizes(), inputs[7].toTensor().sizes());
      

      outputs[m] = get_all_metrics( inputs[0].toTensor(), 
                                    inputs[1].toTensor(), 
                                    inputs[2].toTensor(), 
                                    inputs[3].toTensor(), 
                                    inputs[4].toTensor(), 
                                    inputs[5].toTensor(), 
                                    inputs[6].toTensor(), 
                                    inputs[7].toTensor() 
                                    ).to(torch::kCPU, false);

      // scriptModule.eval();
      // auto iValue = scriptModule.forward(inputs);
      // outputs[m] = iValue.toTensor().clone().to(torch::kCPU, false);

      // models[m]->to(torch::kCPU, false);
    }
    // print("outputs size [0].sizes()", outputs.size(), outputs[0].sizes());
    return torch::stack(at::TensorList(outputs)).clone().to(torch::kCPU);
  }
  std::vector<Transformer> models;
  torch::jit::script::Module scriptModule;
  std::vector<c10::IValue> inputs;
  const std::vector<std::string> hook_names;
};
TORCH_MODULE(BatchThread);

struct Chex {
  static constexpr int p { 113 };
  void load_models();

  template<int XXX=0> at::Tensor get_restricted_test_loss() {
    auto& model { models.back() };
    Vec<std::string> hook_names { { "hook_logits", "blocks.0.mlp.W_out", "unembed.W_U", "blocks.0.mlp.hook_post", "blocks.0.hook_resid_mid" } };
    auto [logits, cache] = model->run_with_cache(all_data, hook_names);
    std::vector<at::Tensor> input { tester.cos_apb, tester.sin_apb, all_labels };
    for (auto name : hook_names) input.push_back(cache[name]);
    auto all_metrics = get_all_metrics(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);
    return all_metrics[1][1][0];
  }

  template<int XXX=0> at::Tensor get_test_loss() {
    auto& model { models.back() };
    Vec<std::string> hook_names { { "hook_logits", "blocks.0.mlp.W_out", "unembed.W_U", "blocks.0.mlp.hook_post", "blocks.0.hook_resid_mid" } };
    auto [logits, cache] = model->run_with_cache(all_data, hook_names);
    std::vector<at::Tensor> input { tester.cos_apb, tester.sin_apb, all_labels };
    for (auto name : hook_names) input.push_back(cache[name]);
    auto all_metrics = get_all_metrics(input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]);
    return all_metrics[1][1][0];
  }

  template<int XXX=0> at::Tensor test_loss(const at::Tensor the_indices) {
    // auto& model { models.back() };
    // auto logits = model(all_data);
    // print("logits, logits.index({ the_indices }), LABELS[all].index({ the_indices })", logits.sizes(), logits.index({ the_indices }).sizes(), all_labels.index({ the_indices }).sizes());
    // auto loss = cross_entropy_high_precision(logits.index({ Slice(), -1, Slice() }).index({ the_indices }), all_labels.index({ the_indices }));
    // return loss;
    return at::Tensor();
  }
  
  template<typename T = Transformer> ActivationCache getActivations(Vec<int> indexes, c10::Device dev, std::string mode, std::string name = "") {
    // if (name != "") print("getting activations for: " + name + "\n");
    T model { cfg, 0, indexes.size() };
    for (int i {}; i < indexes.size(); ++i)
      model->importModel(models[indexes[i]], i);
    model->set_hooks();
    model->to(dev);
    auto logits = model(all_data.index({ indices[mode] }).to(dev));
    // print("train_labels", train_labels.slice(0, 0, 5));
    // print("test_labels", test_labels.slice(0, 0, 5));
    // auto trnlblz { train_labels }, tstlblz { test_labels };
    // if(logits.dim() == 4) {
      // print("logits", logits.sizes());
      // print("trnlblz", trnlblz.sizes());
      // trnlblz = trnlblz.unsqueeze(-1).repeat({ 1, logits.size(1) });
      // tstlblz = tstlblz.unsqueeze(-1).repeat({ 1, logits.size(1) }).flatten();
      // print("trnlblz", trnlblz.sizes());
      // trnlblz = trnlblz.flatten();
      // print("trnlblz", trnlblz.sizes());
    // }
    // print("best train loss", cross_entropy_high_precision(logits.index({ indices["train"] }), trnlblz));
    // print("best test loss", cross_entropy_high_precision(logits.index({ indices["test"] }), tstlblz));
    auto cache = model->getActivationCache().detach();
    return cache;
  }

  GPTConfig cfg {};
  c10::Device device { torch::cuda::is_available() ? torch::kCUDA : 
  torch::kCPU };
  AllData allData { makeAllData() };
  Tester tester {};
  at::Tensor all_data { allData.dataset["all"].to(device) };
  at::Tensor all_labels { allData.labels["all"].to(device) };
  at::Tensor train_indices { indices["train"].to(device) };
  at::Tensor test_indices { indices["test"].to(device) };
  at::Tensor train_data { all_data.index({ train_indices }).contiguous() };
  at::Tensor test_data { all_data.index({ test_indices }).contiguous() };
  at::Tensor train_labels { all_labels.index({ train_indices }).contiguous() };
  at::Tensor test_labels { all_labels.index({ test_indices }).contiguous() };
  std::vector<Transformer> models;
  std::map<std::string, std::vector<int>> sizeMap;
};