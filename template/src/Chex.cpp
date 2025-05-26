#include "Chex.h"
#define EINOPS_TORCH_BACKEND
#include <einops.hpp>
// #include <torch/script.h>
// #include "/usr/local/cuda-11.8/targets/x86_64-linux/include/cuda_runtime_api.h"

namespace einsad {
  at::Tensor rearrange(at::Tensor x, std::string p) { return einops::rearrange(x, p); }
  at::Tensor rearrange(at::Tensor x, std::string p, NamedInt a) { return einops::rearrange(x, p, a); }
  at::Tensor rearrange(at::Tensor x, std::string p, NamedInt a1, NamedInt a2) { return einops::rearrange(x, p, a1, a2); }
  at::Tensor repeat(at::Tensor x, std::string p, NamedInt a) { return einops::repeat(x, p, a); }
  at::Tensor repeat(at::Tensor x, std::string p, NamedInt a1, NamedInt a2) { return einops::repeat(x, p, a1, a2); }
  at::Tensor einsum(std::string pattern, at::Tensor t1, at::Tensor t2) { return einops::einsum(pattern, t1, t2); }
  at::Tensor einsumx(std::string pattern, at::Tensor t1, at::Tensor t2) { return einops::einsum(pattern, t1, t2); }
}