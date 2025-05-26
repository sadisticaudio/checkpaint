# import faulthandler
import torch
# faulthandler.enable(file=open("/root/pythonfault.txt", "w"), all_threads=True)
# import transformer_lens
# import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
#     HookedRootModule,
#     HookPoint,
# )  # Hooking utilities
# from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import checkpaint
from checkpaint.utils import *


import einops
# from fancy_einsum import einsum

p = 113

p_squared = torch.tensor(113*113)
frac_train = 0.3
cutoff = torch.tensor(round(113 * 113 * 0.3) - 1)

def cross_entropy_high_precision(logits, labels):
    # return F.cross_entropy(logits, )
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()

def getAccuracy(predictions, truth):
    classes = torch.argmax(predictions, -1)
    return torch.mean((classes == truth).to(torch.float))

def getMode(all, mode):
    return all[:cutoff] if mode == 'train' else all[cutoff:p_squared] if mode == 'test' else all

def get_first_elements(x, n):
    while x.dim() > 1:
        x = x[0]
    out = ''
    for i, itm in enumerate(x[:n]):
        out += str(x[i].item()) + ' '
    return out


                    

device = "cuda" if torch.cuda.is_available() else "cpu"
# full_run_data = torch.load('/media/frye/sda5/progress-measures-paper/large_files/full_run_data.pth')
# cached_data = torch.load('/root/checkpaint/tests/grokking/workspace/_scratch/grokking_demo_cpu.pth', map_location=torch.device(device))
# torch.save(cached_data, '/root/grokking_demo2.pth')
    # cached_data = torch.load(PTH_LOCATION)
# model_checkpoints = cached_data["checkpoints"]
# for key, value in model_checkpoints[0].items():
#     print(key, value.device)

# torch.save(model_checkpoints, '/media/frye/sda5/checkpaint/all_dicts.pth')

# full_run_data = torch.jit.load('/media/frye/sda5/checkpaint/grokking/all_dicts.pt')
# print('cached_data')
# from torch.utils.cpp_extension import load


# print_object_info(cached_data)
from checkpaint.utils import *
from checkpaint.metric import Metrix

metrix = Metrix('/root/checkpaint/tests/grokking/workspace/_scratch/grokking_demo_new.pth')

# # @torch.jit.export
# def fwd_fn(tensors: list[torch.Tensor]):
#     # with torch.no_grad():
#     #     p = 113
#     #     cutoff = 3830 # torch.tensor(int(113 * 113 * 0.3))
#     #     p_squared = p*p
#     #     cos_apb, sin_apb, LABELS, logits, W_out, W_U, post_mlp_act, resid_mid_act = tensors
#     #     print(cos_apb.shape, sin_apb.shape, LABELS.shape, logits.shape, W_out.shape, W_U.shape, post_mlp_act.shape, resid_mid_act.shape)
#     #     logits = logits[...,-1,:]
#     #     labels = LABELS.long()
#     #     print("logits", logits.shape, "labels", labels.shape)
#     #     neuron_acts = post_mlp_act[...,-1,:]
#     #     resid_mid = resid_mid_act[...,-1,:]
#     #     approx_neuron_acts = torch.zeros_like(neuron_acts)
#     #     print("approx_neuron_acts", approx_neuron_acts.shape)
#     #     print("neuron_acts", neuron_acts.shape, "neuron_acts.mean(-2)", neuron_acts.mean(-2).shape, "neuron_acts.mean(-1)", neuron_acts.mean(-1).shape, "neuron_acts.mean(-3)", neuron_acts.mean(-3).shape)
#     #     approx_neuron_acts += neuron_acts.mean(-3)
        
#     #     key_freqs = [17, 25, 32, 47]

#     #     for freq in enumerate(key_freqs):
#     #         print("approx_neuron_acts", approx_neuron_acts.shape, "(neuron_acts * cos_apb[freq])", (neuron_acts * cos_apb[freq]).shape)
#     #         approx_neuron_acts += (neuron_acts * cos_apb[freq]).sum(-3) * cos_apb[freq]
#     #         approx_neuron_acts += (neuron_acts * sin_apb[freq]).sum(-3) * sin_apb[freq]
        
#     #     restricted_logits = torch.matmul(torch.matmul(approx_neuron_acts, W_out), W_U)
#     #     restricted_logits += logits.mean(-3, True) - restricted_logits.mean(-3, True)
#     #     excluded_neuron_acts = neuron_acts - approx_neuron_acts
#     #     excluded_logits = torch.matmul(torch.matmul(excluded_neuron_acts, W_out) + resid_mid, W_U)

#     #     dims = [3, 2, 2]
#     #     if len(logits.shape) == 3:
#     #         logits = logits.unsqueeze(0)
#     #     dims.append(logits.size(0))
#     #     ret = torch.zeros(dims)#, device=logits.device)
#     #     # acc = lambda predictions, truth: torch.mean((torch.argmax(predictions, -1) == truth).to(torch.float))
#     #     # cehp=lambda lgz, lbz:-((lgz[:,-1].to(torch.float64).log_softmax(dim=-1)).gather(dim=-1, index=lbz[:, -1].long())[:, 0]).mean()
#     #     lgz = logits[...,:cutoff,:]
#     #     lbz = labels[:cutoff]
#     #     print("lgz", lgz.shape, "lbz", lbz.shape)
#     #     # ret[0][0][0] = -((lgz.view(-1, lgz.size(-1)).to(torch.float64).log_softmax(dim=-1)).gather(dim=-1, index=lbz[None,:].long())).mean(-1, True).view(dims[-1], -1)
#     #     # ret[0][0][0] = F.cross_entropy(lgz.view(-1, lgz.size(-1)), lbz, reduction='none').view(lgz.size(0), lgz.size(1), -1).mean(-1, True)
#     #     # ret[0][0][1] = (torch.argmax(lgz, -1) == lbz).to(torch.float).mean(-1, True)
#     #     # lgz = logits[...,cutoff:p_squared,:]
#     #     # lbz = labels[cutoff:p_squared]
#     #     # ret[0][1][0] = -((lgz.view(-1, lgz.size(-1)).to(torch.float64).log_softmax(dim=-1)).gather(dim=-1, index=lbz[None,:].long())).mean(-1, True)
#     #     # ret[0][1][1] = (torch.argmax(lgz, -1) == lbz).to(torch.float).mean(-1, True)
#     #     # lgz = restricted_logits[...,:cutoff,:]
#     #     # lbz = labels[:cutoff]
#     #     # ret[1][0][0] = -((lgz.view(-1, lgz.size(-1)).to(torch.float64).log_softmax(dim=-1)).gather(dim=-1, index=lbz[None,:].long())).mean(-1, True)
#     #     # ret[1][0][1] = (torch.argmax(lgz, -1) == lbz).to(torch.float).mean(-1, True)
#     #     # lgz = restricted_logits[...,cutoff:p_squared,:]
#     #     # lbz = labels[cutoff:p_squared]
#     #     # ret[1][1][0] = -((lgz.view(-1, lgz.size(-1)).to(torch.float64).log_softmax(dim=-1)).gather(dim=-1, index=lbz[None,:].long())).mean(-1, True)
#     #     # ret[1][1][1] = (torch.argmax(lgz, -1) == lbz).to(torch.float).mean(-1, True)
#     #     # lgz = excluded_logits[...,:cutoff,:]
#     #     # lbz = labels[:cutoff]
#     #     # ret[2][0][0] = -((lgz.view(-1, lgz.size(-1)).to(torch.float64).log_softmax(dim=-1)).gather(dim=-1, index=lbz[None,:].long())).mean(-1, True)
#     #     # ret[2][0][1] = (torch.argmax(lgz, -1) == lbz).to(torch.float).mean(-1, True)
#     #     # lgz = excluded_logits[...,cutoff:p_squared,:]
#     #     # lbz = labels[cutoff:p_squared]
#     #     # ret[2][1][0] = -((lgz.view(-1, lgz.size(-1)).to(torch.float64).log_softmax(dim=-1)).gather(dim=-1, index=lbz[None,:].long())).mean(-1, True)
#     #     # ret[2][1][1] = (torch.argmax(lgz, -1) == lbz).to(torch.float).mean(-1, True)
#     #     return ret
#     return torch.rand(2,3)
    
# @torch.jit.export
def fwd_fk(tensors: list[torch.Tensor]):
    with torch.no_grad():
        cos_apb, sin_apb, W_E = tensors
        fourier_basis = []
        fourier_basis.append(torch.ones(p))
        for freq in range(1, p//2+1):
            fourier_basis.append(torch.cos(torch.arange(p)*2 * torch.pi * freq / p))
            fourier_basis.append(torch.sin(torch.arange(p)*2 * torch.pi * freq / p))
        fourier_basis = torch.stack(fourier_basis, dim=0)
        fourier_basis = fourier_basis/fourier_basis.norm(dim=-1, keepdim=True)
        # print("fourier_basis.shape, W_E.shape", fourier_basis.shape, W_E[...,:-1,:].shape)
        # return torch.matmul(fourier_basis.cuda(), W_E[...,:-1,:].cuda())
        return torch.fft.fft2(W_E[...,:-1,:]).abs()
        
frac_train = 0.3
a_vector = torch.arange(p).unsqueeze(1).repeat(1, p).flatten()
b_vector = torch.arange(p).unsqueeze(0).repeat(p, 1).flatten()
equals_vector = torch.tensor(p).repeat(p_squared)
print("dataset first 5s")
print(a_vector[:5])
print(b_vector[:5])
print(equals_vector[:5])
# a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
# b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
# equals_vector = a_vector # einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)
# print("a, b, eq vectors", a_vector.shape, b_vector.shape, equals_vector.shape)
DATA_SEED = 598
torch.manual_seed(DATA_SEED)
indices = torch.randperm(p*p).to(device)
cutoff = int(p*p*frac_train)
train_indices = indices[:cutoff]
test_indices = indices[cutoff:]
key_freqs = [9, 33, 36, 38] # [17, 25, 32, 47]
dataset = torch.stack((a_vector, b_vector, equals_vector), dim=1).to(device)
labels = (dataset[:, 0] + dataset[:, 1]) % p
input = dataset#.to(device) # torch.ones([7,3], dtype=torch.long)
print("input, dataset, train_indices, test_indices, labels", input.shape, dataset.shape, train_indices.shape, test_indices.shape, labels.shape)
print("input, dataset, train_indices, test_indices, labels", input.device, dataset.device, train_indices.device, test_indices.device, labels.device)
# names = ['cos_apb', 'sin_apb', 'LABELS', 'hook_logits', 'blocks.0.mlp.W_out', 'unembed.W_U', 'blocks.0.mlp.hook_post', 'blocks.0.hook_resid_mid']
names = ['cos_apb', 'sin_apb', 'embed.W_E']# 'blocks.0.mlp.hook_post']

# shapes = metrix.get_shapes(input, ["blocks.0.attn.hook_attn_scores", "embed.W_E"])
# print("get_shapes", shapes)
# shapes = [input.shape]
# print("shapes", shapes)
# metrics = metrix.get_metrics()
result = metrix.get_custom_metric(input, names, fwd_fk)
# print("result")
# print(torch.cat((torch.arange(p).unsqueeze(-1).to(result.device), result.sum(-1)[0].unsqueeze(-1), result.sum(-1)[80].unsqueeze(-1), result.sum(-1)[100].unsqueeze(-1), result.sum(-1)[120].unsqueeze(-1), result.sum(-1)[-1].unsqueeze(-1)), 1).long(), result.shape)


print("done with cpp")


######
######



# # Optimizer config
# lr = 1e-3
# wd = 1.
# betas = (0.9, 0.98)

# num_epochs = 25000
# checkpoint_every = 100



# cfg = HookedTransformerConfig(
#     n_layers = 1,
#     n_heads = 4,
#     d_model = 128,
#     d_head = 32,
#     d_mlp = 512,
#     act_fn = "relu",
#     normalization_type=None,
#     d_vocab=p+1,
#     d_vocab_out=p,
#     n_ctx=3,
#     init_weights=True,
#     # device=device,
#     seed = 999,
# )
# print("before HookedTransformer constructor")
# model = HookedTransformer(cfg)
# print("before model.setup()")
# model.setup()
print("end of script")
# faulthandler.dump_traceback(file=open("/root/pythonfault.txt", "w"), all_threads=True)
# model.load_state_dict(model_checkpoints[249])









# input = torch.arange(60, 150, 30)
# input[2] = 113
# input = input.unsqueeze(0)
# print('py input', input)
# # print('first element of W_E[0]', model.embed.W_E[0][0].item())
# # print('first element of W_E[1]', model.embed.W_E[1][0].item())
# # print('first element of W_E[60]', model.embed.W_E[60][0].item())
# # print('first element of W_pos[0]', model.pos_embed.W_pos[0][0].item())
# # print('first element of W_pos[1]', model.pos_embed.W_pos[1][0].item())
# logits, cache = model.run_with_cache(input)
# # print('cache keys', cache.keys())

# cpp.compare_cache(cache.cache_dict)

# from torch import nn
# from torch.nn import functional as F

# def loss_fn(logits, labels):
#     # return F.cross_entropy_loss(logits, )
#     if len(logits.shape)==3:
#         logits = logits[:, -1]
#     logits = logits.to(torch.float64)
#     log_probs = logits.log_softmax(dim=-1)
#     correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
#     return -correct_log_probs.mean()

# # def get_restricted_loss(model, logits, cache):
# #     # logits, cache = model.run_with_cache(dataset)
# #     logits = logits[:, -1, :]
# #     neuron_acts = cache["post", 0, "mlp"][:, -1, :]
# #     print("neuron_acts", get_first_elements(neuron_acts, 3))
# #     approx_neuron_acts = torch.zeros_like(neuron_acts)
# #     approx_neuron_acts += neuron_acts.mean(dim=0)
# #     print("approx_neuron_acts", get_first_elements(approx_neuron_acts, 3))
# #     a = torch.arange(p)[:, None]
# #     b = torch.arange(p)[None, :]
# #     for freq in key_freqs:
# #         cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
# #         cos_apb_vec /= cos_apb_vec.norm()
# #         cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
# #         print("neuron_acts, cos_apb_vec, (neuron_acts * cos_apb_vec).sum(dim=0)", neuron_acts.shape, cos_apb_vec.shape, (neuron_acts * cos_apb_vec).sum(dim=0).shape)
# #         approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
# #         print("freq", freq, "approx_neuron_acts after +cos", get_first_elements(approx_neuron_acts, 3))
# #         sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
# #         sin_apb_vec /= sin_apb_vec.norm()
# #         sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
# #         approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
# #         print("freq", freq, "approx_neuron_acts after +sin", get_first_elements(approx_neuron_acts, 3))
# #     print("cos_apb_vec", get_first_elements(cos_apb_vec, 3))
# #     print("sin_apb_vec", get_first_elements(sin_apb_vec, 3))
# #     restricted_logits = approx_neuron_acts @ model.blocks[0].mlp.W_out @ model.unembed.W_U
# #     print("restricted_logits", get_first_elements(restricted_logits, 3))
# #     # Add bias term
# #     restricted_logits += logits.mean(dim=0, keepdim=True) - restricted_logits.mean(dim=0, keepdim=True)
# #     print("restricted_logits", get_first_elements(restricted_logits, 3))
# #     return loss_fn(restricted_logits[test_indices], test_labels)
# # get_restricted_loss(model, logits, cache)




# logits = model(input)
# logits = logits[:, -1]
# print("logits", logits.shape)
# print("labels", torch.tensor([[37]]).to(logits.device).shape)

# loss = loss_fn(logits.view(-1, logits.shape[-1]), torch.tensor([37]).to(logits.device))
# print("loss", loss)
# max_logit = logits.argmax(-1)
# print("max_logit", max_logit)


# import os

# class SOModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(10, 16)
#         self.relu = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(16, 1)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x
    
# os.remove("/root/model.so")

# with torch.no_grad():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = SOModel().to(device=device)
#     example_inputs=(torch.randn(8, 10, device=device),)
#     batch_dim = torch.export.Dim("batch", min=1, max=1024)
#     so_path = torch._export.aot_compile(
#         model,
#         example_inputs,
#         # Specify the first dimension of the input x as dynamic
#         dynamic_shapes={"x": {0: batch_dim}},
#         # Specify the generated shared library path
#         options={"aot_inductor.output_path": "/root/model.so"},# os.path.join(os.getcwd(), "model.so")},
#     )
#     cpp.processCallable("/root/model.so", [""], [""])







# # full_run_data = torch.load('/media/frye/sda5/progress-measures-paper/large_files/full_run_data.pth')
# cached_data = torch.load('/root/checkpaint/tests/grokking/workspace/_scratch/grokking_demo.pth')
#     # cached_data = torch.load(PTH_LOCATION)
# model_checkpoints = cached_data["checkpoints"]
# # torch.save(model_checkpoints, '/media/frye/sda5/checkpaint/all_dicts.pth')

# # full_run_data = torch.jit.load('/media/frye/sda5/checkpaint/grokking/all_dicts.pt')
# print('cached_data')
# # from torch.utils.cpp_extension import load


# print_object_info(cached_data)

# # Chex = load(name='Chex', sources=['/media/frye/sda5/checkpaint/CheckpointProcessor.cpp'])
# # cpp = checkpaint.CheckpointProcessor(full_run_data.items())
# # cpp = checkpaint.CheckpointProcessor(str('/media/frye/sda5/checkpaint/all_dicts.pth'))
# # cpp = checkpaint.CheckpointProcessor(model_checkpoints)



# ######
# ######

# p = 113
# frac_train = 0.3

# # Optimizer config
# lr = 1e-3
# wd = 1.
# betas = (0.9, 0.98)

# num_epochs = 25000
# checkpoint_every = 100

# DATA_SEED = 598

# key_freqs = [17, 25, 32, 47]
# device = "cuda" if torch.cuda.is_available() else "cpu"

# a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
# b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
# equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)

# dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
# # print(dataset[:5])
# # print(dataset.shape)

# labels = (dataset[:, 0] + dataset[:, 1]) % p
# # print(labels.shape)
# # print(labels[:5])

# torch.manual_seed(DATA_SEED)
# indices = torch.randperm(p_squared)
# cutoff = torch.int64(p_squared*frac_train)
# train_indices = indices[:cutoff]
# test_indices = indices[cutoff:]

# train_data = dataset[train_indices]
# train_labels = labels[train_indices]
# test_data = dataset[test_indices]
# test_labels = labels[test_indices]
# # print(train_data[:5])
# # print(train_labels[:5])
# # print(train_data.shape)
# # print(test_data[:5])
# # print(test_labels[:5])
# # print(test_data.shape)

# cfg = HookedTransformerConfig(
#     n_layers = 1,
#     n_heads = 4,
#     d_model = 128,
#     d_head = 32,
#     d_mlp = 512,
#     act_fn = "relu",
#     normalization_type=None,
#     d_vocab=p+1,
#     d_vocab_out=p,
#     n_ctx=3,
#     init_weights=True,
#     # device=device,
#     seed = 999,
# )

# model = HookedTransformer(cfg)
# model.setup()

# model.load_state_dict(model_checkpoints[249])









# # input = torch.arange(60, 150, 30)
# # input[2] = 113
# # input = input.unsqueeze(0)
# # print('py input', input)
# # # print('first element of W_E[0]', model.embed.W_E[0][0].item())
# # # print('first element of W_E[1]', model.embed.W_E[1][0].item())
# # # print('first element of W_E[60]', model.embed.W_E[60][0].item())
# # # print('first element of W_pos[0]', model.pos_embed.W_pos[0][0].item())
# # # print('first element of W_pos[1]', model.pos_embed.W_pos[1][0].item())
# # logits, cache = model.run_with_cache(input)
# # # print('cache keys', cache.keys())

# # cpp.compare_cache(cache.cache_dict)

# # from torch import nn
# # from torch.nn import functional as F

# # def loss_fn(logits, labels):
# #     # return F.cross_entropy_loss(logits, )
# #     if len(logits.shape)==3:
# #         logits = logits[:, -1]
# #     logits = logits.to(torch.float64)
# #     log_probs = logits.log_softmax(dim=-1)
# #     correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
# #     return -correct_log_probs.mean()

# # # def get_restricted_loss(model, logits, cache):
# # #     # logits, cache = model.run_with_cache(dataset)
# # #     logits = logits[:, -1, :]
# # #     neuron_acts = cache["post", 0, "mlp"][:, -1, :]
# # #     print("neuron_acts", get_first_elements(neuron_acts, 3))
# # #     approx_neuron_acts = torch.zeros_like(neuron_acts)
# # #     approx_neuron_acts += neuron_acts.mean(dim=0)
# # #     print("approx_neuron_acts", get_first_elements(approx_neuron_acts, 3))
# # #     a = torch.arange(p)[:, None]
# # #     b = torch.arange(p)[None, :]
# # #     for freq in key_freqs:
# # #         cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
# # #         cos_apb_vec /= cos_apb_vec.norm()
# # #         cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
# # #         print("neuron_acts, cos_apb_vec, (neuron_acts * cos_apb_vec).sum(dim=0)", neuron_acts.shape, cos_apb_vec.shape, (neuron_acts * cos_apb_vec).sum(dim=0).shape)
# # #         approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
# # #         print("freq", freq, "approx_neuron_acts after +cos", get_first_elements(approx_neuron_acts, 3))
# # #         sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
# # #         sin_apb_vec /= sin_apb_vec.norm()
# # #         sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
# # #         approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
# # #         print("freq", freq, "approx_neuron_acts after +sin", get_first_elements(approx_neuron_acts, 3))
# # #     print("cos_apb_vec", get_first_elements(cos_apb_vec, 3))
# # #     print("sin_apb_vec", get_first_elements(sin_apb_vec, 3))
# # #     restricted_logits = approx_neuron_acts @ model.blocks[0].mlp.W_out @ model.unembed.W_U
# # #     print("restricted_logits", get_first_elements(restricted_logits, 3))
# # #     # Add bias term
# # #     restricted_logits += logits.mean(dim=0, keepdim=True) - restricted_logits.mean(dim=0, keepdim=True)
# # #     print("restricted_logits", get_first_elements(restricted_logits, 3))
# # #     return loss_fn(restricted_logits[test_indices], test_labels)
# # # get_restricted_loss(model, logits, cache)




# # logits = model(input)
# # logits = logits[:, -1]
# # print("logits", logits.shape)
# # print("labels", torch.tensor([[37]]).to(logits.device).shape)

# # loss = loss_fn(logits.view(-1, logits.shape[-1]), torch.tensor([37]).to(logits.device))
# # print("loss", loss)
# # max_logit = logits.argmax(-1)
# # print("max_logit", max_logit)



# # import os

# # class SOModel(torch.nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.fc1 = torch.nn.Linear(10, 16)
# #         self.relu = torch.nn.ReLU()
# #         self.fc2 = torch.nn.Linear(16, 1)
# #         self.sigmoid = torch.nn.Sigmoid()

# #     def forward(self, x):
# #         x = self.fc1(x)
# #         x = self.relu(x)
# #         x = self.fc2(x)
# #         x = self.sigmoid(x)
# #         return x
    
# # os.remove("/root/model.so")

# # with torch.no_grad():
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     model = SOModel().to(device=device)
# #     example_inputs=(torch.randn(8, 10, device=device),)
# #     batch_dim = torch.export.Dim("batch", min=1, max=1024)
# #     so_path = torch._export.aot_compile(
# #         model,
# #         example_inputs,
# #         # Specify the first dimension of the input x as dynamic
# #         dynamic_shapes={"x": {0: batch_dim}},
# #         # Specify the generated shared library path
# #         options={"aot_inductor.output_path": "/root/model.so"},# os.path.join(os.getcwd(), "model.so")},
# #     )
# #     cpp.processCallable("/root/model.so", [""], [""])
