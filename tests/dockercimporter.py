import torch
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import checkpaint

import collections
import einops
from fancy_einsum import einsum

def get_first_elements(x, n):
    while x.dim() > 1:
        x = x[0]
    out = ''
    for i, itm in enumerate(x[:n]):
        out += str(x[i].item()) + ' '
    return out

def print_object_info(obj, indent=0):
    def add_indent(text, level):
        return '  ' * level + text
    
    def get_unique_types(container):
        unique_types = set()
        if isinstance(container, (dict, collections.OrderedDict)):
            for value in container.values():
                unique_types.add(type(value))
                # if isinstance(value, (dict, list, set, tuple)):
                #     unique_types.update(get_unique_types(value))
        elif isinstance(container, (list, set, tuple)):
            for item in container:
                unique_types.add(type(item))
                # if isinstance(item, (dict, list, set, tuple)):
                #     unique_types.update(get_unique_types(item))
        return unique_types
    
    obj_type = type(obj)
    print(add_indent(f"Type: {obj_type}", indent))
    
    if isinstance(obj, dict):
        print(add_indent(f"Size: {len(obj)}", indent))
        unique_types = get_unique_types(obj)
        for typ in unique_types:
            print(add_indent(f"Contains type: {typ}", indent + 1))
        typeset = set()
        for key, value in obj.items():
            # if type(value) not in typeset:
            #     typeset.add(type(value))
            if isinstance(value, (dict, list, set, tuple)):
                print(add_indent(f"Key: {key} ->", indent + 1))
                print_object_info(value, indent + 2)
    elif isinstance(obj, (list, set, tuple)):
        print(add_indent(f"Size: {len(obj)}", indent))
        unique_types = get_unique_types(obj)
        for typ in unique_types:
            print(add_indent(f"Contains type: {typ}", indent + 1))
        typeset = set()
        for index, item in enumerate(obj):
            if type(item) not in typeset:
                typeset.add(type(item))
                if isinstance(item, (dict, list, set, tuple)):
                    print(add_indent(f"Element {index} ->", indent + 1))
                    print_object_info(item, indent + 2)
                    


# full_run_data = torch.load('/media/frye/sda5/progress-measures-paper/large_files/full_run_data.pth')
cached_data = torch.load('/app/grokking/workspace/_scratch/grokking_demo.pth')
    # cached_data = torch.load(PTH_LOCATION)
model_checkpoints = cached_data["checkpoints"]
# torch.save(model_checkpoints, '/media/frye/sda5/checkpaint/all_dicts.pth')

# full_run_data = torch.jit.load('/media/frye/sda5/checkpaint/grokking/all_dicts.pt')
print('cached_data')
# from torch.utils.cpp_extension import load


print_object_info(cached_data)

# Chex = load(name='Chex', sources=['/media/frye/sda5/checkpaint/CheckpointProcessor.cpp'])
# cpp = checkpaint.CheckpointProcessor(full_run_data.items())
# cpp = checkpaint.CheckpointProcessor(str('/media/frye/sda5/checkpaint/all_dicts.pth'))
cpp = checkpaint.CheckpointProcessor(model_checkpoints)



######
######

p = 113
frac_train = 0.3

# Optimizer config
lr = 1e-3
wd = 1.
betas = (0.9, 0.98)

num_epochs = 25000
checkpoint_every = 100

DATA_SEED = 598

key_freqs = [17, 25, 32, 47]
device = "cuda" if torch.cuda.is_available() else "cpu"

a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)

dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
# print(dataset[:5])
# print(dataset.shape)

labels = (dataset[:, 0] + dataset[:, 1]) % p
# print(labels.shape)
# print(labels[:5])

torch.manual_seed(DATA_SEED)
indices = torch.randperm(p*p)
cutoff = int(p*p*frac_train)
train_indices = indices[:cutoff]
test_indices = indices[cutoff:]

train_data = dataset[train_indices]
train_labels = labels[train_indices]
test_data = dataset[test_indices]
test_labels = labels[test_indices]
# print(train_data[:5])
# print(train_labels[:5])
# print(train_data.shape)
# print(test_data[:5])
# print(test_labels[:5])
# print(test_data.shape)

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
    # device=device,
    seed = 999,
)

model = HookedTransformer(cfg)
model.setup()

model.load_state_dict(model_checkpoints[249])

# train_losses = []
# test_losses = []
# model_checkpoints = []
# checkpoint_epochs = []

# TRAIN_MODEL=False

# if TRAIN_MODEL:
#     for epoch in range(num_epochs):

#         if ((epoch+1)%checkpoint_every)==0:
#             checkpoint_epochs.append(epoch)
#             model_checkpoints.append(copy.deepcopy(model.state_dict()))
#             print(f"Epoch {epoch}")











input = torch.arange(60, 150, 30)
input[2] = 113
input = input.unsqueeze(0)
print('py input', input)
# print('first element of W_E[0]', model.embed.W_E[0][0].item())
# print('first element of W_E[1]', model.embed.W_E[1][0].item())
# print('first element of W_E[60]', model.embed.W_E[60][0].item())
# print('first element of W_pos[0]', model.pos_embed.W_pos[0][0].item())
# print('first element of W_pos[1]', model.pos_embed.W_pos[1][0].item())
logits, cache = model.run_with_cache(input)
# print('cache keys', cache.keys())

cpp.compare_cache(cache.cache_dict)

from torch import nn
from torch.nn import functional as F

def loss_fn(logits, labels):
    # return F.cross_entropy_loss(logits, )
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()

# def get_restricted_loss(model, logits, cache):
#     # logits, cache = model.run_with_cache(dataset)
#     logits = logits[:, -1, :]
#     neuron_acts = cache["post", 0, "mlp"][:, -1, :]
#     print("neuron_acts", get_first_elements(neuron_acts, 3))
#     approx_neuron_acts = torch.zeros_like(neuron_acts)
#     approx_neuron_acts += neuron_acts.mean(dim=0)
#     print("approx_neuron_acts", get_first_elements(approx_neuron_acts, 3))
#     a = torch.arange(p)[:, None]
#     b = torch.arange(p)[None, :]
#     for freq in key_freqs:
#         cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
#         cos_apb_vec /= cos_apb_vec.norm()
#         cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
#         print("neuron_acts, cos_apb_vec, (neuron_acts * cos_apb_vec).sum(dim=0)", neuron_acts.shape, cos_apb_vec.shape, (neuron_acts * cos_apb_vec).sum(dim=0).shape)
#         approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
#         print("freq", freq, "approx_neuron_acts after +cos", get_first_elements(approx_neuron_acts, 3))
#         sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
#         sin_apb_vec /= sin_apb_vec.norm()
#         sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
#         approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
#         print("freq", freq, "approx_neuron_acts after +sin", get_first_elements(approx_neuron_acts, 3))
#     print("cos_apb_vec", get_first_elements(cos_apb_vec, 3))
#     print("sin_apb_vec", get_first_elements(sin_apb_vec, 3))
#     restricted_logits = approx_neuron_acts @ model.blocks[0].mlp.W_out @ model.unembed.W_U
#     print("restricted_logits", get_first_elements(restricted_logits, 3))
#     # Add bias term
#     restricted_logits += logits.mean(dim=0, keepdim=True) - restricted_logits.mean(dim=0, keepdim=True)
#     print("restricted_logits", get_first_elements(restricted_logits, 3))
#     return loss_fn(restricted_logits[test_indices], test_labels)
# get_restricted_loss(model, logits, cache)




logits = model(input)
logits = logits[:, -1]
print("logits", logits.shape)
print("labels", torch.tensor([[37]]).to(logits.device).shape)

loss = loss_fn(logits.view(-1, logits.shape[-1]), torch.tensor([37]).to(logits.device))
print("loss", loss)
max_logit = logits.argmax(-1)
print("max_logit", max_logit)






print("cpp.nam= ", cpp.nam)
