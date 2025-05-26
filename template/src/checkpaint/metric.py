import torch
import torch.nn as nn
import transformer_lens
from transformer_lens.loading_from_pretrained import fill_missing_keys
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import checkpaint
from collections import OrderedDict
from collections.abc import Callable
import einops
from pathlib import Path
import copy

# from fancy_einsum import einsum
p = 113
frac_train = 0.3
cutoff = int(p*p*frac_train)

def load_old_state_dict(model, state_dict):
    sd = copy.deepcopy(state_dict)
    sd["blocks.0.attn.W_O"] = sd["blocks.0.attn.W_O"].reshape(sd["blocks.0.attn.W_Q"].shape).transpose(-2,-1)
    for name, param in sd.items():
        if "W_" in name and not "pos_embed" in name and not "unembed" in name:
            sd[name] = sd[name].transpose(-2,-1)
    for name, param in model.named_parameters():
        if not name in sd:
            sd[name] = param
    for name, param in model.named_buffers():
        if not name in sd:
            sd[name] = param
    sd["unembed.W_U"] = sd["unembed.W_U"][...,:-1]
    model.load_state_dict(sd)

def getAccuracy(predictions, truth):
    classes = torch.argmax(predictions, -1)
    return torch.mean((classes == truth).to(torch.float))

def get_first_elements(input, n):
    x = input
    while x.dim() > 1:
        x = x[0]
    out = ''
    for i, itm in enumerate(x[:n]):
        out += str(round(x[i].item(), 5)) + ' '
    return out

def get_first_axis_elements(input, n):
    x = input.flatten()
    out = ''
    div = input.numel()
    for d in range(input.ndim):
        div = div//input.shape[d]
        out += "dim " + str(d) + " "
        for i in range(min(n, x.numel()//div)):
            out += str(round(x[i * div].item(), 5)) + ' '
        out += "\n"
    return out
                
class MetricModel(nn.Module):
    def __init__(self, fwd_fn: Callable):
        super().__init__()
        self.fwd_fn = fwd_fn
    def forward(self, *args):
        return self.fwd_fn(args)
    
def compare_restricted_test_loss(model, dataset, labels, key_freqs, device, the_indices):
    logits, cache = model.run_with_cache(dataset)
    logits = logits[:, -1, :]
    neuron_acts = cache["post", 0, "mlp"][:, -1, :]
    approx_neuron_acts = torch.zeros_like(neuron_acts)
    approx_neuron_acts += neuron_acts.mean(dim=0)
    
    a = torch.arange(p)[:, None]
    b = torch.arange(p)[None, :]
    for freq in key_freqs:
        cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
        cos_apb_vec /= cos_apb_vec.norm()
        cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
        sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
        sin_apb_vec /= sin_apb_vec.norm()
        sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
    
    restricted_logits = approx_neuron_acts @ model.blocks[0].mlp.W_out @ model.unembed.W_U
    # Add bias term
    restricted_logits += logits.mean(dim=0, keepdim=True) - restricted_logits.mean(dim=0, keepdim=True)
    rv = cross_entropy_high_precision(restricted_logits[the_indices], labels[the_indices].to(device))
    return rv

def get_restricted_loss(model, dataset, labels, key_freqs, device, the_indices):
    logits, cache = model.run_with_cache(dataset)
    logits = logits[:, -1, :]
    neuron_acts = cache["post", 0, "mlp"][:, -1, :]
    approx_neuron_acts = torch.zeros_like(neuron_acts)
    approx_neuron_acts += neuron_acts.mean(dim=0)
    
    a = torch.arange(p)[:, None]
    b = torch.arange(p)[None, :]
    for freq in key_freqs:
        cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
        cos_apb_vec /= cos_apb_vec.norm()
        cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
        sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
        sin_apb_vec /= sin_apb_vec.norm()
        sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
    
    restricted_logits = approx_neuron_acts @ model.blocks[0].mlp.W_out @ model.unembed.W_U
    # Add bias term
    restricted_logits += logits.mean(dim=0, keepdim=True) - restricted_logits.mean(dim=0, keepdim=True)
    rv = cross_entropy_high_precision(restricted_logits[the_indices], labels[the_indices].to(device))
    print("neuron_acts", neuron_acts[0][0].item())
    print("approx_neuron_acts", approx_neuron_acts[0][0].item())
    print("model.blocks[0].mlp.W_out", model.blocks[0].mlp.W_out[0][0].item())
    print("model.unembed.W_U", model.unembed.W_U[0][0].item())
    print("cos_apb_vec", cos_apb_vec[0][0].item())
    print("restricted_logits", restricted_logits[0][0].item())
    print("rv", rv.item())
    return rv

class Metrix:
    def __init__(self, cached_data_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("in Metrix.__init__(), about to load")
        self.cached_data = torch.load(cached_data_path, map_location=torch.device(self.device))
        # print("in Metrix.__init__(), loaded")
        self.model_checkpoints = self.cached_data["checkpoints"] if "checkpoints" in self.cached_data else self.cached_data["state_dicts"]
        self.cfg = self.cached_data["config"]
        self.state_dict = self.model_checkpoints[-1]# self.cached_data["model"]
        # print("in Metrix.__init__(), about to call cpp ctor")
        self.cpp = checkpaint.CheckpointProcessor(self.model_checkpoints)
    
    def get_hooked_model(self, idx):
        sd = self.cpp.get_state_dict(idx)
        for name, param in sd.items():
            sd[name] = param.squeeze()
        cfg = HookedTransformerConfig(
            n_layers = 1,
            n_heads = 4,
            d_model = 128,
            d_head = 32,
            d_mlp = 512,
            act_fn = "relu",
            normalization_type=None,
            d_vocab=p+1,
            d_vocab_out=p,
            n_ctx=3,
            init_weights=True,
            device=self.device,
            seed = 999,
        )
        model = HookedTransformer(cfg)
        sd = fill_missing_keys(model, sd)
        model.load_state_dict(sd)
        return model

    # takes a sample input Tensor and a list of names
    # returns a list of shapes of params, activations and other known named Tensors used to get metrics
    def get_shapes(self, input: torch.Tensor, names: list[str]):
        shapes = self.cpp.get_shapes(input, names)
        return shapes
    
    def get_state_dict(self, idx):
        return self.cpp.get_state_dict(idx)
    
    def get_activations_slice(self, input, hook_names, dim_idx = 0, cp_indices = []):
        return self.cpp.get_activations_slice(input, hook_names, dim_idx, cp_indices)
    
    def set_state_dict(self, idx, state_dict):
        self.cpp.set_state_dict(idx, state_dict)
    
    def get_key_freqs(self):
        return self.cpp.get_key_freqs()
    
    def get_mag_symmetries(self, mags):
        return self.cpp.get_mag_symmetries(mags)
    
    def get_metrics(self):
        return self.cpp.get_metrics()
    
    def get_modified_logits(self, input, checkpoint_idx, weight_dict):
        return self.cpp.get_modified_logits(input, checkpoint_idx, weight_dict)
    
    def get_modified_activations(self, input, checkpoint_idx, weight_dict, names = []):
        return self.cpp.get_modified_activations(input, checkpoint_idx, weight_dict, names)
        
    def get_last_activations(self, input, names = []):
        return self.cpp.get_last_activations(input, names)
    
    def get_neuron_history(self, indices):
        return self.cpp.get_neuron_history(indices)
    
    def get_all_activations(self, input):
        return self.cpp.get_all_activations(input)
    
    def load_checkpoints(self, checkpoints):
        self.cpp.load_checkpoints(checkpoints)
    
    def get_from_all_checkpoints(self, input: torch.Tensor, name: str, slices: list[list[int]] = []):
        
        shapes = self.cpp.get_shapes(input, [name])
        print("shapes[0]", shapes[0])

        def index_activation(tensor: torch.Tensor):
            for i in reversed(range(len(slices))):
                print("i", i, "tensor", tensor.shape)
                tensor = torch.index_select(tensor, slices[i][0], torch.arange(slices[i][1], slices[i][2] if len(slices[i]) == 3 else slices[i][1] + 1).to(tensor.device))
            for i in range(len(slices)):
                if len(slices[i]) == 2:
                    tensor = tensor.squeeze(slices[i][0])
            return tensor
        
        traced = torch.jit.trace(index_activation, torch.randn(shapes[0]).to(self.device))
        print("traced")
        # print(traced.code_with_constants)
        traced_path = str(Path.home()) + "/traced_model.pt"
        print("saving jitted model to", traced_path)
        traced.save(traced_path)
        return self.cpp.get_custom_metric(traced_path, [], [name], input)
                

    def get_custom_metric(self,
                    input: torch.Tensor,
                    # globals: list[torch.Tensor],
                    names: list[str], 
                    fwd_fn):
        input = input.to(self.device)
        print("names", names)
        shapes = self.cpp.get_shapes(input.to("cuda"), names)
        print("shapes", shapes)
        
        tensors = []

        for shape in shapes:
            tensors.append(torch.rand(shape))
        
        metric_model = MetricModel(fwd_fn)
        traced = torch.jit.trace(metric_model, tensors)
        # print("traced")
        # print(traced.code_with_constants)
        frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(traced.eval()))
        home = Path.home()
        traced_path = str(home) + "/traced_model.pt"
        print("saving jitted model to", traced_path)
        frozen_mod.save(traced_path)
        return self.cpp.get_custom_metric(traced_path, [], names, input)