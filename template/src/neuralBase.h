#pragma once
#include "GPTConfig.h"
#include <torch/torch.h>
// #include <c10/cuda/CUDACachingAllocator.h>
// #include <cuda_runtime_api.h>

template<typename T> std::ofstream& sadSerialize(std::ofstream& os, const T& x) {
  if constexpr (std::is_same_v<std::decay_t<T>, torch::Tensor>) {
    // print("serializing a Tensor on device", x.sizes(), x.device());
    auto st { x.scalar_type() };
    const int64_t nDim { std::max(1L, int64_t(x.dim())) }, nElem { int64_t(x.numel()) };
    const int64_t tSize { int64_t(x.element_size()) };
    const int64_t sType { st == c10::kFloat ? 6L : st == c10::kLong ? 4L : st == c10::kBool ? 11L : st == c10::kDouble ? 7L : 0L };
    forEach([&](const int64_t& x){ os.write(r2cChar(&x), sizeof(int64_t)); }, nDim, nElem, tSize, sType);

    os.write(r2cChar(x.sizes().data()), sizeof(int64_t) * nDim);
    os.write(r2cChar(x.strides().data()), sizeof(int64_t) * nDim);
    // print("b4 reportAndWrite, nDim, nElem, tSize, sType, totalSize, pos", nDim, nElem, tSize, sType, tSize * nElem, os.tellp());
    auto reportAndWrite = [&](const auto* data) { os.write(r2cChar(data), tSize * nElem); };
    if (st == c10::kFloat) reportAndWrite(x.template data_ptr<float>());
    else if (st == c10::kLong) reportAndWrite(x.template data_ptr<long>());
    else if (st == c10::kBool) reportAndWrite(x.template data_ptr<bool>());
    else if (st == c10::kDouble) reportAndWrite(x.template data_ptr<double>());
    else print("unknown dtype, not serializing!", x.dtype());
  }
  else if constexpr (isPair<T>) { sadSerialize(os,x.first); sadSerialize(os,x.second); }
  else if constexpr (std::is_same_v<std::decay_t<T>, Byte>) { sadSerialize(os, x.data); }
  // else if constexpr (std::is_same_v<std::decay_t<T>, typename torch::OrderedDict<std::basic_string<char>, at::Tensor>::Item>) { sadSerialize(os,x.key()); sadSerialize(os,x.value()); }
  else if constexpr (isContainer<T>) {
    size_t len { size_t(x.size()) }; 
    os.write(r2cChar(&len), sizeof(size_t));
    // print("ofs << " + getClassName(x) + "/(" + std::to_string(len) + ")", "isHeap<value_type> = ", isHeap<typename T::value_type>);
    if constexpr (isHeap<typename T::value_type>) {
      for (size_t i {}; i < len; ++i) 
        sadSerialize(os, x[i]);
    }
    else {
      os.write(r2cChar(&x[0]), len * sizeof(typename T::value_type));
    }
  }
  else if constexpr (!isHeap<T>) { os.write(r2cChar(&x), sizeof(T)); }
  else { 
    print("ofstream sadSerialize can't handle this type", getClassName(x));
    if constexpr (hasSize<T> && hasValueType<T> && !hasIterators<T>) print("doesn't have iterators");
  }
  return os;
}

template<typename T> std::ifstream& sadSerialize(std::ifstream& is, T& x){
  if constexpr (isPair<T>) { sadSerialize(is,x.first); sadSerialize(is,x.second); }
  else if constexpr (std::is_same_v<std::decay_t<T>, Byte>) { sadSerialize(is, x.data); }
  else if constexpr (std::is_same_v<std::decay_t<T>, at::Tensor>) {
    int64_t nDim {}, nElem {}, tSize{}, sType{};

    is.read(r2Char(&nDim), sizeof(int64_t));
    is.read(r2Char(&nElem), sizeof(int64_t));
    is.read(r2Char(&tSize), sizeof(int64_t));
    is.read(r2Char(&sType), sizeof(int64_t));
    c10::ScalarType dType { sType == 6L ? c10::kFloat : sType == 4L ? c10::kLong : sType == 11L ? c10::kBool : c10::kFloat };
    std::vector<int64_t> sizes, strides;
    // print("nDim, nElem, tSize, sType", nDim, nElem, tSize, sType);
    sizes.resize(size_t(nDim));
    strides.resize(size_t(nDim));
    is.read(r2Char(sizes.data()), sizeof(int64_t) * nDim);
    is.read(r2Char(strides.data()), sizeof(int64_t) * nDim);
    std::vector<char> scratch;
    scratch.resize(nElem * tSize);
    is.read(scratch.data(), nElem * tSize);
    x = torch::from_blob((void*)scratch.data(), { sizes.data(), size_t(nDim) }, { strides.data(), size_t(nDim) }, dType).clone().to(torch::kCPU);
  }
  else if constexpr (isContainer<T>) {
    size_t len {}; 
    is.read(r2Char(&len), sizeof(size_t));
    x.resize(len);
    if constexpr (isHeap<typename T::value_type>) for (size_t i {}; i < len; ++i) sadSerialize(is, x[i]);
    else is.read(r2Char(&x[0]), len * sizeof(typename T::value_type));
  }
  else if constexpr (!isHeap<T>) { is.read(r2Char(&x), sizeof(T)); }
  else { 
    print("ifstream sadSerialize can't handle this type", getClassName(x));
    if constexpr (hasSize<T> && hasValueType<T> && !hasIterators<T>) print("doesn't have iterators");
  }
  return is; 
}

template <typename T> bool loadContainer(T& vec, std::string path) {
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if ( !ifs || ifs.fail() ) return false;
  std::cout << "loading container from: " << path << " ... ";
  sadSerialize(ifs, vec);
  std::cout << vec.size() << " elements\n";
  return true;
}

template<typename T> void saveContainer(const T& vec, std::string path) {
    std::ofstream ofs(path, std::ofstream::trunc);
    std::cout << "\nSaving container of type " << getClassName(vec) << " to path: " << path << "\n";
    if (ofs) sadSerialize(ofs, vec);
}

template<typename T = float> std::string get_first_elements(at::Tensor x, int n) {
  while (x.dim() > 1) x = x[0];
  x=x.flatten();
  std::string out;
  for (int i {}; i < std::min(int(x.size(0)), n); ++i) out += std::to_string(x[i].item<T>()).substr(0, 8) + " ";
  return out;
}

template<typename T = float> std::string get_first_axis_elements(at::Tensor input, int n) {
  auto x = input.flatten();
  std::string out = "";
  auto div = input.numel();
  for (int d = 0; d < input.dim(); ++d) {
    div = div/input.size(d);
    out += "dim " + std::to_string(d) + "\n";
    for (int i = 0; i < std::min(n, int(x.numel()/div)); ++i)
      out += std::to_string(x[i * div].item<float>()).substr(0, 8) + " ";
  }
  out += "\n";
  return out;
}

template<typename T = float> std::string get_N_from_data_ptr(at::Tensor x, int n) {
  x = x.to(torch::kCPU);
  std::string s;
  const T* ptr { (const T*)x.storage().data() }; 
  for (int i {}; i < std::min(int(x.numel()), n); ++i, ++ptr) s += std::to_string(*ptr) + " ";
  return s;
}

template<typename T = float> T get_first_element(at::Tensor x) {
  while (x.dim()) x = x[0];
  return x.item<T>();
}

// static constexpr float nInf { -1E10F };//666666.f };//std::numeric_limits<float>::infinity() };//666666.f };
static constexpr float nInf { -std::numeric_limits<float>::infinity() };//666666.f };

template <torch::ScalarType t = torch::kFloat> torch::Tensor getTokenwiseCrossEntropy(torch::Tensor x, torch::Tensor label) { 
  print("getTokenwiseCrossEntropy(x, t) args ->", x.sizes(), label.sizes());
  return torch::nn::functional::cross_entropy(x.to(t), label, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)); }//.view({ -1, x.size(-1) }), t.view({ -1 })); }

template <torch::ScalarType t = torch::kFloat> torch::Tensor getCrossEntropy(torch::Tensor x, torch::Tensor label) { 
  // print("getCrossEntropy(x, t) args ->", x.sizes(), t.sizes());
  return torch::nn::functional::cross_entropy(x.to(t).view({ -1, x.size(-1) }), label, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean)); }
  
template <int N = 0> torch::Tensor getTokenwiseUncertainty(torch::Tensor logits) {
  auto probs = torch::softmax(logits, -1);
    // Calculate the entropy
  auto log_probs = torch::log(probs);
  auto entropy = -torch::sum(probs * log_probs, -1);
  return entropy;
}

template <int N = 0> torch::Tensor getUncertainty(torch::Tensor logits) { 
  auto tokWise { getTokenwiseUncertainty(logits) };
  auto mean { tokWise.mean() }, stddev { tokWise.std() };
  // print("Uncertainty mean, stddev", mean.item<float>(), stddev.item<float>());
  return mean; }

template <int N = 0> float getUncertaintyFloat(torch::Tensor logits) { return getUncertainty(logits).item<float>(); }

// template <int N = 0> torch::Tensor getTokenwisePerplexity(torch::Tensor x, torch::Tensor t) {
//   auto tokWise { getTokenwiseCrossEntropy(x, t) };
//   tokWise = torch::exp(tokWise);
//   auto mean { tokWise.mean() }, stddev { tokWise.std(-1, false) };\
//   return mean; }

// template <int N = 0> torch::Tensor getPerplexity(torch::Tensor x, torch::Tensor t) {
//     torch::Tensor tokenwisePerplexity { getTokenwisePerplexity(x, t) };
//     return tokenwisePerplexity.mean();
// }

// template <int N = 0> float getPerplexityFloat(torch::Tensor x, torch::Tensor t) { return getPerplexity(x,t).item<float>(); }

template <int N = 0> torch::Tensor getTokenwiseConfidence(torch::Tensor logits) {
  auto probs = torch::softmax(logits, -1);
  // Calculate confidence as the maximum probability for each token in the sequence
  torch::Tensor confidence = std::get<0>(torch::max(probs, -1));
  return confidence;
}

template <int N = 0> float getConfidence(torch::Tensor logits) { 
  auto tokWise { getTokenwiseConfidence(logits.to(torch::kCPU)) };
  auto mean { tokWise.mean() }, stddev { tokWise.std() };
  // print("Confidence mean, stddev", mean.item<float>(), stddev.item<float>());
  return mean.item<float>(); }

template <torch::ScalarType t = torch::kFloat> torch::Tensor getAccuracy(torch::Tensor logits, const torch::Tensor& labels) { // logits, labels
  // logits = logits.view({ -1, logits.size(-1) });
  auto classes = torch::argmax(logits.to(t), -1);
  auto truth = labels;
  // print("logits, labels, truth, classes (argmax)", logits.sizes(), labels.sizes(), truth.sizes(), classes.sizes());
  while (truth.dim() < logits.dim() - 1) truth = truth.unsqueeze(1);
  // print("logits, labels, truth 2, classes (argmax)", logits.sizes(), labels.sizes(), truth.sizes(), classes.sizes());
  auto rv = torch::mean((classes == truth.expand({ classes.sizes() })).to(torch::kFloat), 0);
  return rv.dim() ? rv : rv.unsqueeze(0);
}

template <int N = 0> float getAccuracyFloat(torch::Tensor predictions, torch::Tensor truth) { return getAccuracy(predictions, truth).item<float>(); }
// static Tensor unEmbedToken(Tensor y, bool argMax = false) {
//   return argMax ? y.argmax(-1) : y.softmax(-1, kFloat).view({ -1, y.size(-1) }).multinomial(1);
// }