#pragma once
// #include <torch/torch.h>
#include <torch/extension.h>
#include "Chex.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct CheckpointProcessor : Chex {
  CheckpointProcessor(const py::dict&);
  CheckpointProcessor(const py::list&);
  void load_checkpoints(const py::list& state_dicts);
  void load_config(const py::dict& config);
  void compare_cache(const py::dict&);
  std::vector<std::vector<int>> get_shapes(const py::object&, const std::vector<std::string>&);
  std::vector<int> get_key_freqs();
  torch::Tensor get_metrics();
  torch::Tensor get_custom_metric(const std::string&, std::vector<torch::Tensor>, std::vector<std::string>, torch::Tensor);
  std::map<std::string, torch::Tensor> get_last_activations(torch::Tensor, std::vector<std::string>);
  std::map<std::string, torch::Tensor> get_modified_activations(torch::Tensor input,
                                                            int checkpoint_idx,
                                                            std::map<std::string, torch::Tensor> weight_dict,
                                                            std::vector<std::string> hook_names = {});
  std::map<std::string, torch::Tensor> get_activations_slice(torch::Tensor input,
                                            std::vector<std::string> hook_names = {},
                                            std::vector<py::object> dim_idx = {},
                                            std::vector<int> cp_indices = {});
  torch::Tensor get_modified_logits(torch::Tensor input,
                                                      int checkpoint_idx,
                                                      std::map<std::string, torch::Tensor> weight_dict);
  std::map<std::string, torch::Tensor> get_all_activations(torch::Tensor);
  std::map<std::string, torch::Tensor> get_state_dict(int);
  void set_state_dict(int, const std::map<std::string, torch::Tensor>&);
  at::Tensor get_mag_symmetries(const at::Tensor& mags);
};