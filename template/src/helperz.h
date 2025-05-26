#pragma once
#include <random>

#include "neuralBase.h"

static constexpr long pModulo { 113 }, DATA_SEED { 598 };
static constexpr float frac_train { 0.3f }, lr { 1e-3f }, wd { 1.f }, betas[] { 0.9f, 0.98f };
static constexpr long num_epochs { 25000 }, checkpoint_every { 100 }, numTotal { pModulo * pModulo };
static constexpr long numTrain { static_cast<long>(numTotal * frac_train) }, numTest { numTotal - numTrain };
static constexpr double pifloat = M_PI;//3.14159265f;
using namespace torch::indexing;
using TensorMap = std::map<std::string, at::Tensor>;

namespace einsad {
  using NamedInt = std::tuple<std::string, int64_t>;
  at::Tensor rearrange(at::Tensor x, std::string pattern);
  at::Tensor rearrange(at::Tensor x, std::string p, NamedInt a);
  at::Tensor rearrange(at::Tensor x, std::string p, NamedInt a1, NamedInt a2);
  at::Tensor repeat(at::Tensor x, std::string p, NamedInt a);
  at::Tensor repeat(at::Tensor x, std::string p, NamedInt a1, NamedInt a2);
  at::Tensor einsum(std::string pattern, at::Tensor, at::Tensor);
  at::Tensor einsumx(std::string pattern, at::Tensor, at::Tensor);
}

inline TensorMap makeINDICES() {
  TensorMap indices;
  torch::manual_seed(uint64_t(DATA_SEED));
  indices["all"] = torch::randperm(pModulo*pModulo);
  indices["train"] = indices["all"].slice(0, 0, numTrain);
  indices["test"] = indices["all"].slice(0, numTrain, numTotal);
  return indices;
}

inline TensorMap indices { makeINDICES() };

template<typename T> bool roughly_same(const T& t1, const T& t2, std::string nm1 = "tensor1", std::string nm2 = "tensor2") {
  return torch::allclose(t1.to(torch::kCPU), t2.to(torch::kCPU), 1E-01F, 1E-02F);
}

struct ActivationCache {
  struct ActKeys {
    template<typename T> std::string stringify(const T& x) { if constexpr (std::is_integral_v<T>) return std::to_string(int(x)); else return { x }; }
    template<typename ...Ts> ActKeys(Ts... ts) { forEach([&](auto x){ data.push_back(stringify(x)); }, ts...); } 
    std::vector<std::string> data;
  };
  auto begin() { return data.begin(); }
  const auto begin() const { return data.begin(); }
  auto end() { return data.end(); }
  const auto end() const { return data.end(); }
  at::Tensor operator[](ActKeys keys) {
    for(auto& [n, t] : data) { 
      bool found = true; 
      for(const auto& x : keys.data)
        if (n.find(x) == std::string::npos) 
          found = false;
      if (found) return t;
    }
    // data.push_back({ keys }, at::Tensor());
    // return data.back();
    for(const auto& [n, t] : data) print(n, t.sizes());
    print("ERROR - keys not in ActivationCache!!! Returning Tensor()", keys.data, "cache size", data.size()); 
    return torch::Tensor();
  }
  bool contains(const std::string& s) const { for(const auto& [name, t] : data) if (name == s) return true; return false; }
  
  long getNumBytes() {
    long tot { 0 };
    for (const auto& [name, t] : data) tot += t.numel() * t.element_size();
    return tot;
  }
  std::map<std::string, at::Tensor> to_map() { return { data.begin(), data.end() }; }
  std::vector<std::pair<std::string, at::Tensor>> data;
};

template<typename TSR> TSR cross_entropy_high_precision(TSR logits, TSR labels) {
  auto B { logits.size(0) }, M { logits.size(1) }, T { logits.size(-2) }, V { logits.size(-1) }, numel { logits.numel() };
  at::Tensor ret;
  
  for (int m {}; m < M; ++m) {
    auto r = getCrossEntropy<torch::kDouble>(logits.slice(1, m, m + 1), labels);
    ret = (m ? torch::cat({ ret, r.unsqueeze(0) }) : r.unsqueeze(0));
  }
  return ret;
}

template<int XXX=0> void checknan(torch::Tensor x, std::string name = "") {
  // print(x.isnan().any());
  if (x.isnan().any().item<bool>() == true) {
    print("found nan " + name);
    std::terminate();
  }
}

