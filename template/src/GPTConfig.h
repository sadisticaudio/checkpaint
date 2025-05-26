#pragma once

#include "/home/frye/CODE/frye/common.h"
#include <map>

static constexpr int defaultNumLayers { 8 }, maxTokens { 4096 };

template<typename T> 
std::string nDigits(T x, int n) {
  std::string space = "";
  if constexpr (std::is_floating_point_v<T>) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(n) << std::setw(n + 2);
    if (n > 4) ss << std::scientific;
    ss << x;
    return space + ss.str();
  }
  else {
    auto s { std::to_string(x) }; while (s.size() < size_t(n)) s.insert(0U, " "); return s;
  }
}

struct HyperParam {
  HyperParam() : value(0) { print("HyperParam Default Constructor, probaby means this param was not in Config Map"); std::terminate(); }
  HyperParam(int val) : value(val) {}
  HyperParam(float val) : value(val), isFloat(true) {}

  operator int() const { return to<int>(); }
  operator int&() { return value.i; }

  template<typename T> T to() const {
    if (std::is_integral_v<T> && isFloat) print("\n\nconverting float hyper param to integral!", value.f, static_cast<T>(isFloat ? value.f : value.i));
    if constexpr (std::is_integral_v<T>) { return static_cast<T>(isFloat ? roundToInt(value.f) : value.i); }
    else return static_cast<T>(isFloat ? value.f : value.i);
  }
  int& operator=(const int& x) { value = x; return value.i; }
  float& operator=(const float& x) { value = x; return value.f; }
  std::string toString() const { if (isFloat) return std::to_string(value.f); else return std::to_string(value.i); }

  union IorF {
    IorF(int x) : i(x) {}
    IorF(float x) : f(x) {}
    int i; float f;
  } value;
  bool isFloat { false };
};

struct GPTConfig {
  // GPTConfig() = default;
  HyperParam& operator[](std::string x) { if (!(params.find(x) != params.end())) print("trying to access element not in map", x); return params.at(x); }
  const HyperParam& operator[](const std::string& x) const { return params.at(x); }

  size_t estMemUsage() {
    int B { params["trainingBatchSize"] }, T { params["n_ctx"] }, N { params["n_heads"] }, C { params["d_model"] };
    int D { C/N }, L { params["n_layers"] }, Mmodel { 7 * L * C * C }, Mact { B * N * L * T * (T + 2 * D) };
    // float modelActRatio { float(Mmodel)/float(Mact) };
    return 80ULL * size_t(Mmodel) + 42ULL * size_t(Mact);
  }

  size_t estMemUsage(long paramSize) {
    int B { params["trainingBatchSize"] }, T { params["n_ctx"] }, N { params["n_heads"] }, C { params["d_model"] };
    int D { C/N }, L { params["n_layers"] };
    size_t Mmodel { (size_t)paramSize }, Mact { size_t(B * N * L * T * (T + 6 * D)) };
    // float modelActRatio { float(Mmodel)/float(Mact) };
    return 4ULL * Mmodel + 22ULL * Mact;
  }

  std::map<std::string, HyperParam> params {
    { "d_vocab", { 114 } },
    { "d_vocab_out", { 113 } },
    { "d_model", { 128 } },
    { "d_head", { 4 } },
    { "d_mlp", { 512 } },
    { "evalIters", { 15 } },
    { "batchLimit", { 75 } }, 
    { "n_ctx", { 3 } }, 
    { "n_heads", { 4 } }, 
    { "n_layers", { 1 } }, 
    { "numEpochs", { 25000 } }, 
    { "trainingBatchSize", { 256 } }, 
    { "validationBatchSize", { 256 } }, 
    { "frac_train", { 0.9f } }, 
    { "dropOut", { 0.1f } }, 
    { "lr", { 1E-3F } },//0.00031f } },
    { "weightDecay", { 1.f } },
    { "beta1", { 0.9f } },
    { "beta2", { 0.999f } }
  };
  // int& d_vocab { params["d_vocab"].value.i };
  // int& d_model { params["d_model"].value.i };
  // int& evalIters { params["evalIters"].value.i };
  // int& batchLimit { params["batchLimit"].value.i };
  // int& n_ctx { params["n_ctx"].value.i };
  // int& n_heads { params["n_heads"].value.i };
  // int& n_layers { params["n_layers"].value.i };
  // int& numEpochs { params["numEpochs"].value.i };
  // int& trainingBatchSize { params["trainingBatchSize"].value.i };
  // int& validationBatchSize { params["validationBatchSize"].value.i };
  // float& frac_train { params["frac_train"].value.f };
  // float& dropOut { params["dropOut"].value.f };
  // float& lr { params["lr"].value.f };
  // int& numPhases { params["numPhases"].value.i };
  // int& tPhase { params["tPhase"].value.i };
};

// std::ostream& operator<<(std::ostream& os, const HyperParam& dt) { os << "\nHYPERPARAMETER: " << dt.toString(); return os; }
// std::ostream& operator<<(std::ostream& os, HyperParam& dt) { os << "\nHYPERPARAMETER: " << dt.toString(); return os; }