#pragma once
#include "Chex.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

struct EmbedOldImpl : torch::nn::Module {
  EmbedOldImpl(GPTConfig cc) : cfg(cc),
  W_E(register_parameter("W_E", torch::randn({ d_vocab, C })/std::sqrt(float(C)), false)) {}

  at::Tensor forward (at::Tensor x) { 
    return W_E.index_select(0, x.flatten()).view({ x.size(0), x.size(1), C }); }

  GPTConfig cfg;
  int C { cfg["d_model"] }, d_vocab { cfg["d_vocab"] };
  torch::Tensor W_E;
};
TORCH_MODULE(EmbedOld);

struct UnEmbedOldImpl : torch::nn::Module {
  UnEmbedOldImpl(GPTConfig cc) : cfg(cc),
  W_U(register_parameter("W_U", torch::randn({ C, d_vocab - 1 })/std::sqrt(float(d_vocab)), false)) {}

  at::Tensor forward (at::Tensor x) { 
    print("OLD", "x", x.sizes(), "W_U", W_U.sizes());
    auto ret = torch::matmul(x, W_U);
    return ret; 
    
  }

  GPTConfig cfg;
  int C { cfg["d_model"] }, d_vocab { cfg["d_vocab"] };
  torch::Tensor W_U;
};
TORCH_MODULE(UnEmbedOld);

struct PosEmbedOldImpl : torch::nn::Module {
  PosEmbedOldImpl(GPTConfig cc) : cfg(cc), 
  W_pos(register_parameter("W_pos", torch::randn({ T, C })/std::sqrt(float(C)), false)) {}

  at::Tensor forward (at::Tensor x) {
    return einsad::repeat(W_pos, "pos d_model -> batch pos d_model", { "batch", x.size(0) }); }

  GPTConfig cfg;
  int T { cfg["n_ctx"] }, C { cfg["d_model"] };
  torch::Tensor W_pos;
};
TORCH_MODULE(PosEmbedOld);

struct NandaAttentionOldImpl : torch::nn::Module {
  NandaAttentionOldImpl(GPTConfig cc) : cfg(cc),
  W_K(register_parameter("W_K", torch::randn({ nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_Q(register_parameter("W_Q", torch::randn({ nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_V(register_parameter("W_V", torch::randn({ nh, C, C/nh })/std::sqrt(float(C)), false)),
  W_O(register_parameter("W_O", torch::randn({ nh, C/nh, C })/std::sqrt(float(C)), false)),
  hook_k(register_hook(this, "k")),
  hook_q(register_hook(this, "q")),
  hook_v(register_hook(this, "v")),
  hook_z(register_hook(this, "z")),
  hook_pattern(register_hook(this, "pattern")),
  hook_attn_scores(register_hook(this, "attn_scores")) {}

  torch::Tensor forward (torch::Tensor x) {

    auto B = x.size(0);

    auto q = hook_q(torch::einsum("idh,bpd->bpih", { W_Q, x }));// [nh C hs] @ [B T C] = [B T nh hs]
    auto k = hook_k(torch::einsum("idh,bpd->bpih", { W_K, x }));// [nh C hs] @ [B T C] = [B T nh hs]
    auto v = hook_v(torch::einsum("idh,bpd->bpih", { W_V, x }));// [nh C hs] @ [B T C] = [B T nh hs]
    q = q.transpose(-3,-2);// [B T nh hs] -> [B nh T hs]
    k = k.transpose(-3,-2).transpose(-2, -1); // [B T nh hs] -> [B nh T hs] -> [B nh hs T]
    v = v.transpose(-3,-2);// [B T nh hs] -> [B nh T hs]

    auto attn_scores_pre = torch::matmul(q, k)/std::sqrt(float(hs));// [B nh T hs] @ [B nh hs T] = [B nh T T]
    auto attn_scores_masked = attn_scores_pre + torch::full({ T, T }, nInf).triu(1).to(k.device());// [B nh T T]
    auto pattern = hook_pattern(torch::nn::functional::softmax(hook_attn_scores(attn_scores_masked), -1));// [B nh T T]

    if (x.size(0) < 10000) print("\n\n\n\n\nLISTEN\n(old tformer pattern)", "this is where things diverge, this should be the same", pattern.sizes(), get_first_elements(pattern, 4));
    
    auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    
    z = hook_z(einsad::rearrange(z, "batch head_index query_pos d_head -> batch query_pos head_index d_head"));
    if (x.size(0) < 10000) print("(old tformer z)", z.sizes(), get_first_elements(z, 4));
    auto w = einsad::rearrange(W_O, "head_index d_head d_model -> d_model (head_index d_head)");
    if (x.size(0) < 10000) print("(old tformer w)", w.sizes(), get_first_elements(w, 4));
    print("linear args - z.reshape({ B, T, C }), w", z.reshape({ B, T, C }).sizes(), w.sizes());
    auto out = torch::nn::functional::linear(z.reshape({ B, T, C }), w);

    print("out", out.sizes());
    std::cout << "\n\n\n" << std::endl;

    // at::Tensor temp;
    // auto checkIfSame = [&](const auto& t1, const auto& t2, std::string nm1 = "tensor1", std::string nm2 = "tensor2") {
    //   if (torch::allclose(t1.to(torch::kCPU), t2.to(torch::kCPU), 1E-03F, 1E-05F))
    //     print(nm1 + " is the same as " + nm2, t1.sizes());
    //   else print("NOT THE SAME", nm1, t1.sizes(), nm2, t2.sizes());
    // };
    // auto q = hook_q(torch::einsum("idh,bpd->bpih", { W_Q, x }));// [nh C hs] @ [B T C] = [B T nh hs]
    // auto k = hook_k(torch::einsum("idh,bpd->bpih", { W_K, x }));// [nh C hs] @ [B T C] = [B T nh hs]
    // auto v = hook_v(torch::einsum("idh,bpd->bpih", { W_V, x }));// [nh C hs] @ [B T C] = [B T nh hs]

    // // test code, delete
    // {
    //   std::cout << "\n\nTESTING ENSEMBLING METHOD\nTRYING Q MAP\n";
    //   auto W_Q_ENS = torch::cat({ W_Q.unsqueeze(0), torch::rand_like(W_Q.unsqueeze(0)) });
    //   print("x", x.sizes(), "W_Q", W_Q.sizes(), "W_Q_ENS", W_Q_ENS.sizes());
    //   auto q_ENS = torch::einsum("...idh,bpd->...bpih", { W_Q_ENS, x });
    //   auto q_ENS2 = torch::einsum("bpd,midh->mbpih", { x, W_Q_ENS });
    //   checkIfSame(q_ENS[0], q, "q_ENS[0]", "q");
    //   checkIfSame(q_ENS2[0], q, "q_ENS2[0]", "q");
    //   // checkIfSame(q, torch::matmul(W_Q_ENS, x)[0], "q", "torch::matmul(W_Q_ENS, x)[0]");
    // }

    // print("q sizes before", q.sizes());
    // print("q.transpose(1, 2), q.transpose(-3, -2) after", q.transpose(1,2).sizes(), q.transpose(-3, -2).sizes());
    // q = q.transpose(-3,-2);// [B T nh hs] -> [B nh T hs]
    // print("q after transpose sizes", q.sizes());
    // // std::terminate();
    // k = k.transpose(-3,-2).transpose(-2, -1); // [B T nh hs] -> [B nh T hs] -> [B nh hs T]
    // v = v.transpose(-3,-2);// [B T nh hs] -> [B nh T hs]

    // auto attn_scores_pre = torch::matmul(q, k)/std::sqrt(float(hs));// [B nh T hs] @ [B nh hs T] = [B nh T T]
    // auto attn_scores_masked = attn_scores_pre + torch::full({ T, T }, nInf).triu(1).to(k.device());// [B nh T T]
    // auto pattern = hook_pattern(torch::nn::functional::softmax(hook_attn_scores(attn_scores_masked), -1));// [B nh T T]
    
    
    // auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    // z = hook_z(einsad::rearrange(z, "batch head_index query_pos d_head -> batch query_pos head_index d_head"));
    // auto w = einsad::rearrange(W_O, "head_index d_head d_model -> d_model (head_index d_head)");
    // auto out = torch::nn::functional::linear(z.reshape({ z.size(0), z.size(1), C }), w);
    
    
    // auto z = einsad::einsum("batch head k_pos d_head, batch head q_pos k_pos -> batch head q_pos d_head", v, pattern);
    // temp = torch::matmul(pattern,v);
    // print("z.sizes()", z.sizes(), v.sizes(), pattern.sizes(), temp.sizes());
    // checkIfSame(z, torch::matmul(pattern,v), "z", "torch::matmul(pattern,v)");
    // auto transposedNotRearranged = z.transpose(-3,-2);
    // z = hook_z(einsad::rearrange(z, "batch head_index query_pos d_head -> batch query_pos head_index d_head"));
    // checkIfSame(z, transposedNotRearranged, "z", "transposedNotRearranged");
    // // z = hook_z(z.transpose(-3,-2));
    // auto reshapedNotRearranged = W_O.reshape({ C, C }).transpose(-2,-1);
    // auto w = einsad::rearrange(W_O, "head_index d_head d_model -> d_model (head_index d_head)");
    // // auto w = W_O.reshape({ C, C });
    // checkIfSame(w, reshapedNotRearranged, "w", "reshapedNotRearranged");

    // auto neg1Reshape = torch::nn::functional::linear(z.reshape({ -1, T, C }), w);
    // auto out = torch::nn::functional::linear(z.reshape({ z.size(0), z.size(1), C }), w);
    // checkIfSame(out, neg1Reshape, "out", "neg1Reshape");
    
    // print("Well everything worked");
    return out;
  }
  
  GPTConfig cfg;
  int nh { cfg["n_heads"] }, T { cfg["n_ctx"] }, C { cfg["d_model"] }, hs { C/nh };
  torch::Tensor W_K, W_Q, W_V, W_O;
  HookPoint hook_k, hook_q, hook_v, hook_z, hook_pattern, hook_attn_scores;
};
TORCH_MODULE(NandaAttentionOld);

struct NandaMLPOldImpl : torch::nn::Module {
  NandaMLPOldImpl(GPTConfig cfg) : c(cfg),
  W_in(register_parameter("W_in", torch::randn({ C, d_mlp })/std::sqrt(float(C)), false)),
  reLU(register_module("reLU", torch::nn::ReLU(torch::nn::ReLUOptions()))),
  W_out(register_parameter("W_out", torch::randn({ d_mlp, C })/std::sqrt(float(C)), false)),
  hook_pre(register_hook(this, "pre")),
  hook_post(register_hook(this, "post")) {}
  void reset() {}//override {}
  torch::Tensor forward(torch::Tensor x) { 
    int cudaIdx = 1;
    // printCudaMemUsage("MLP Batched " + std::to_string(cudaIdx++));
    x = hook_pre(torch::matmul(x, W_in));
    x = reLU(x);
    x = hook_post(x);
    // printCudaMemUsage("MLP Batched " + std::to_string(cudaIdx++));
    x = torch::matmul(x, W_out);
    // printCudaMemUsage("MLP Batched " + std::to_string(cudaIdx++));
    return x;
  }
  GPTConfig c;
  int C { c["d_model"] }, d_mlp { c["d_mlp"] };
  torch::Tensor W_in;
  torch::nn::ReLU reLU;
  torch::Tensor W_out;
  HookPoint hook_pre, hook_post;
};
TORCH_MODULE(NandaMLPOld);

struct NandaDecoderBlockOld : torch::nn::Module {
  NandaDecoderBlockOld(GPTConfig cfg) : c(cfg),
  attn(register_module("attn", NandaAttentionOld(c))),
  mlp(register_module("mlp", NandaMLPOld(c))),
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
  NandaAttentionOld attn;
  NandaMLPOld mlp;
  HookPoint hook_attn_out, hook_mlp_out, hook_resid_pre, hook_resid_mid, hook_resid_post;
};

using NandaDecoderStackOldImpl = ModuleStack<NandaDecoderBlockOld>;
TORCH_MODULE(NandaDecoderStackOld);

struct TransformerOldImpl : torch::nn::Module {
  std::string name() { return "Transformer" + std::to_string(n); }
  TransformerOldImpl(GPTConfig c, c10::Device dev, int nnn = 0) : cfg(c), n(nnn), 
  embed(register_module("embed", EmbedOld(cfg))), 
  pos_embed(register_module("pos_embed", PosEmbedOld(cfg))), 
  blocks(register_module("blocks", NandaDecoderStackOld(cfg))), 
  unembed(register_module("unembed", UnEmbedOld(cfg))), 
  hook_tokens(register_hook(this, "tokens")), 
  hook_embed(register_hook(this, "embed")), 
  hook_pos_embed(register_hook(this, "pos_embed")), 
  hook_logits(register_hook(this, "logits")) { if (n == 0) print("in Transformer ctr"); }

  void importParameter(const std::string& name, const torch::Tensor& tensor) {
    // print("in importParameter");
    auto actualName { name };
    size_t pos { 0UL };
    actualName = replaceAll(actualName, {"sAd"}, {"."});
    while (((pos = actualName.find("sAd")) != std::string::npos)) actualName.replace(pos++, 3, ".");

    for (auto& p : named_parameters()) {
      if (actualName == p.key() || p.key() == name) {
        auto t { tensor };
        // if (isTesting) { for (int d {}; d < t.dim(); ++d) t = t.slice(d, 0, p.value().size(d)); }
        if (t.sizes() != p.value().sizes()) {
          auto szVec = t.sizes().vec();
          szVec.erase(szVec.begin());
          if (szVec == p.value().sizes().vec()) {
            print("squeezing first dim", name, actualName, t.sizes(), p.value().sizes(), p.key());
            t = t.squeeze(0);
          }
          else print("mismatch copying IValue", name, actualName, t.sizes(), p.value().sizes(), p.key(), szVec);
        }
        p.value().copy_(t.detach());
        p.value().set_requires_grad(false);
        break;
      }
    }
  }

  void swapParameter(const std::string& name, const torch::Tensor& tensor) {
    auto actualName { name };
    size_t pos { 0UL };
    actualName = replaceAll(actualName, {"sAd"}, {"."});
    while (((pos = actualName.find("sAd")) != std::string::npos)) actualName.replace(pos++, 3, ".");

    for (auto& p : named_parameters()) {
      if (actualName == p.key() || p.key() == name) {
        auto t { tensor.detach() };
        // if (isTesting) { for (int d {}; d < t.dim(); ++d) t = t.slice(d, 0, p.value().size(d)); }
        if (t.sizes() != p.value().sizes()) print("mismatch copying IValue", name, actualName, t.sizes(), p.value().sizes(), p.key());
        p.value() = t;
        p.value().set_requires_grad(false);
        break;
      }
    }
  }

  at::Tensor forward (const at::Tensor& x) {
    auto embeddings { hook_embed(embed(hook_tokens(x))) }, positions { hook_pos_embed(pos_embed(x)) };
    auto resid = blocks->module(embeddings + positions);
    auto logits = hook_logits(unembed(resid));
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
      for (auto& p : hook->parent->named_parameters())
        if(p.key().find("hook_") != std::string::npos) {
          for (const auto& param : named_parameters())
            if (param.key().find(p.key()) != std::string::npos && !cache.contains(param.key()))
              cache.data.push_back({ param.key(), param.value() });
        }
    size_t idx = 0UL;
    for (const auto& param : named_parameters())
      if (param.key().find("hook_") == std::string::npos)
        cache.data.insert(cache.data.begin() + idx++, { param.key(), param.value() });
    // for (auto p : named_parameters()) cache.data.push_back({ p.key(), p.value() });
    // for(const auto& p : cache.data) print("pringing in order", p.first);
    return cache;
  }

  // ActivationCache getActivationCache() {
  //   ActivationCache cache;
  //   // Vec<std::shared_ptr<HookPointImpl>> hooks;
  //   // for (const auto& item : named_modules("", false))
  //   //   if (item.key().find("hook_module_") != std::string::npos)
  //   //     hooks.push_back(std::dynamic_pointer_cast<HookPointImpl>(item.value()));
  //   // std::sort(hooks.begin(), hooks.end(), [](auto l, auto r) { return l->time < r->time; });
  //   // for (auto& hook : hooks)
  //   //   for (auto& param : hook->parent->named_parameters())
  //   //     if(param.key().find("hook_") != std::string::npos) cache.data.push_back({ hook->parent->name() + "." + param.key(), param.value() });
  //   for (auto p : named_parameters()) cache.data.push_back({ p.key(), p.value() });
  //   return cache;
  // }

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
  EmbedOld embed;
  PosEmbedOld pos_embed;
  NandaDecoderStackOld blocks;
  UnEmbedOld unembed;
  HookPoint hook_tokens, hook_embed, hook_pos_embed, hook_logits;
};
TORCH_MODULE(TransformerOld);


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////