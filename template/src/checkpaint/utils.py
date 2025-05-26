import torch
import collections
import io
# from IPython.display import display, Javascript
from math import exp
import random
import sys
import numpy as np

def printvar(var, globals, file = sys.stderr):
    """Prints the variable name and its value."""
    for name, value in globals.items():
        if value is var:
            if isinstance(var, float):
                value = round(value, 4)
            if isinstance(var, torch.Tensor):
                shape = list(var.shape)
                print(f"{name} = {shape}", end=" ", file=file)
                if var.numel() < 10:
                    print("[", end=" ", file=file)
                    var = var.flatten()
                    for i in range(var.numel()):
                        print(round(var[i].item(), 4), end=" ", file=file)
                    print("]", end=" ", file=file)
            else:
                print(f"{name} = {value}", end=" ", file=file)
            return

def printvars(*kwargs, file = sys.stderr):
    for value in kwargs: printvar(value, globals(), file)
    print(" ", file=file)
    
def getvars(*kwargs):
    output = io.StringIO()
    printvars(kwargs, globals(), file=output)
    # Redirect stdout to the StringIO object
    sys.stdout = output
    # Get the captured output as a string
    captured_output = output.getvalue()
    # Reset stdout to its original value
    sys.stdout = sys.__stdout__
    # Reset the standard output
    output.close()
    return captured_output

def tstr(x):
    def round_nested_list(nested_list, decimal_places=3):
        if isinstance(nested_list, (float, int)): return round(nested_list, decimal_places)
        rounded_list = []
        for item in nested_list:
            if isinstance(item, torch.Tensor):
                rounded_list.append(tstr(item.tolist()))
            elif isinstance(item, list) and len(item) == 1 and isinstance(item[0], (int, float)):
                rounded_list.append(round(item[0], decimal_places))
            elif isinstance(item, list):
                rounded_list.append(round_nested_list(item, decimal_places))  # Recursive call for sublists
            elif isinstance(item, (int, float)):
                rounded_list.append(round(item, decimal_places))
            else:
                rounded_list.append(item)  # Keep non-numeric elements as they are
        return rounded_list
    def find_max_length_key(input_dict):
        if not input_dict:
            return None
        if not isinstance(list(input_dict.keys())[0], str):
            new_list = [str(x) for x in input_dict.keys()]
            return len(str(max(new_list, key=len)))
        else: return len(str(max(input_dict, key=len)))
    if isinstance(x, list):
        return "[" + str([tstr(t) + " " for t in x]) + "]"
    elif isinstance(x, dict):
        ret = ""
        maxlen = find_max_length_key(x)
        for name in x:
            # if isinstance(x[name], torch.Tensor): print("dict key", str(name) + ":", tstr(x[name]))
            ret += str(name) + str(": " + " " * (maxlen - len(str(name)))) + tstr(x[name]) + '\n'
        return ret
    elif isinstance(x, tuple): return "(" + str(round_nested_list(list(x)))[1:-1] + ")"
    elif isinstance(x, torch.Size): return str(list(x))
    elif isinstance(x, torch.Tensor): return str(round_nested_list((torch.view_as_real(x) if x.dtype in (torch.cfloat, torch.cdouble) else x).tolist()))
    elif isinstance(x, np.generic): return str(np.round(x, 5))
    elif isinstance(x, slice): return "slice(None)" if x == slice(None) else str(x)
    else: return str(x)
   
def tprint(*args, **kwargs):
    args = [tstr(x) for x in args]
    kwargs = {tstr(name): tstr(x) for name, x in kwargs.items()}
    print(*args, **kwargs)

def to_human_readable(size, decimal_places=2):
    """Convert bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def get_first_axis_elements(input, n):
    x = input.flatten()
    total = input.numel() if isinstance(input, torch.Tensor) else input.size
    div = total
    # idx =
    out = "total size = " + str(total)
    for d in range(input.ndim):
        div = div//input.shape[d]
        out += "\ndim " + str(d) + "\n"
        # out += "i (" + str(i) + "), i * div (" + str(i * div) + 
        for i in range(min(n, int(total - n * div))):
            if total//div + i < total:
                out += str(total//div + i) + ": " + str(round(x[total//div + i].item(), 4)) + " "
    out += "\n"
    print(out)
    return out

def get_random_elements(input, title = "", indices = [666,777,888,999]):
    x = input.flatten()
    totallen = x.shape[0]
    
    out = title + ": random elements for tensor of size " + str(totallen)
    
    for i in range(len(indices)):
        idx = indices[i] % totallen
        elem = round(x[idx].item(),4)
        out += str(idx) + ": (" + str(elem) + "), "
    out += "\n"
    return out
        

def printCudaMemUsage(title=""):
    mem = torch.cuda.memory_allocated(torch.device("cuda")) if torch.cuda.is_available() else 0
    mem = to_human_readable(mem)
    print(f"{title}: CUDA Allocated memory: {mem}")

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
                    
def generalized_logistic(t, *, A=0, B=1, C=1, K=1, Q=1, v=1, shift=0):
    if isinstance(t, torch.Tensor):
        return A + ((K - A)/torch.pow(C + Q * torch.exp(-B * (t - shift)), 1/v))
    else:
        return A + ((K - A)/pow(C + Q * exp(-B * (t - shift)), 1/v))
    
def get_index_of_closest(tensor, target):
    distances = torch.abs(tensor - target)
    return torch.argmin(distances).item()
    
def tune_diverse_logistic_paramaters(x):
    N = x.numel()
    x, _ = x.flatten().sort()
    def get_distance_and_shift(skewed, divs):
        indices = [get_index_of_closest(skewed, skewed[0] + (skewed[-1] - skewed[0]) * div/divs) for div in range(divs + 1)]
        idx_spacing = N/divs
        distance, shift = int(0), int(0)
        for i in range(1,divs + 1):
            shift = indices[i] - i * idx_spacing
            distance += abs(shift)
        return distance
    bB, bShift = 25, 0.5

    for iter in range(1,5):
        printCudaMemUsage("in tune_diverse_logistic_paramaters iter " + str(iter))
        divs = pow(2,iter)
        shiftdelta = 1/pow(2,iter+1)
        leftskewed = generalized_logistic(x, B=bB, shift=(bShift-shiftdelta))
        leftdistance = get_distance_and_shift(leftskewed, divs)
        rightskewed = generalized_logistic(x, B=bB, shift=(bShift+shiftdelta))
        rightdistance = get_distance_and_shift(rightskewed, divs)
        bShift = bShift-shiftdelta if leftdistance < rightdistance else bShift+shiftdelta
        upskewed = generalized_logistic(x, B=bB*2, shift=bShift)
        updistance = get_distance_and_shift(upskewed, divs)
        downskewed = generalized_logistic(x, B=bB/2, shift=bShift)
        downdistance = get_distance_and_shift(downskewed, divs)
        bB = bB*2 if updistance < downdistance else bB/2
    return bB, bShift
    
def interpDFT(input, size=0, dim=-1):
    arr = input
    D = arr.ndim
    dim = (dim + D) % D
    sz = arr.shape[dim]
    if dim != D-1: arr = arr.transpose(dim, -1)
    if size == 0: size = sz * 3
    spectrum = torch.fft.fft(arr)
    if sz % 2 == 1 and size != 0 and size % 2 == 0:
        spectrum[...,sz//2] = (spectrum[...,sz//2] + spectrum[...,(sz+1)//2])*2
        for i in range((sz+1)//2, sz-1):
            spectrum[...,i] = spectrum[...,i+1]
        sz = sz - 1
        spectrum = spectrum[...,:sz]
    
    big = torch.roll(spectrum, sz//2, -1)
    leftPad = (size-sz)//2
    # padding = [(size-sz)//2 if i == dim * 2 else (size-sz) - (size-sz)//2 if i == dim * 2 + 1 else 0 for i in range(D*2)]
    BIG = torch.nn.functional.pad(big, [leftPad, size-(leftPad+sz)])
    BIG = torch.roll(BIG, (size+1)//2, -1)
    ARR = torch.real(torch.fft.ifft(BIG))*size/(sz+1)
    if dim != D-1: ARR = ARR.transpose(dim,-1)
    return ARR

def interp2D(arr, size):
    print("arr", arr.shape, "sizes", size)
    ARR = interpDFT(arr, size[0], -2)
    ARR = interpDFT(ARR, size[1], -1)
    return ARR

def running_sum(x, dim, *, weights=None, create_normalized_list=False):
    if dim != -1: x = x.transpose(dim, -1)
    if weights is None: weights = torch.ones([x.size(-1)], device=x.device).expand(x.shape)
    r_sum = torch.zeros_like(x)
    for d in range(x.size(-1)):
        r_sum[...,d:] += x[...,d:d+1] * weights[...,d:d+1]
    if dim != -1: 
        r_sum = r_sum.transpose(dim, -1)
        x = x.transpose(dim, -1)
    if create_normalized_list: return [r_sum, x * (r_sum.abs().max() / x.abs().max())]
    else: return r_sum

def get_top_k_freqs(tensor, k, dim=-1, *, sumlist=[], freqs_allowed=[], squeeze=False):
    if sumlist != []: sumlist = [(x + tensor.ndim) % tensor.ndim for x in sorted(sumlist)]
    dim = (tensor.ndim + dim) % tensor.ndim
    dft = torch.fft.fft(tensor, dim=dim)
    if freqs_allowed == []: freqs_allowed = [x for x in range(tensor.shape[dim]//2+1)]
    if not isinstance(freqs_allowed, torch.Tensor): freqs_allowed = torch.tensor(freqs_allowed, device=tensor.device, dtype=torch.long)
    mags = torch.index_select(dft, dim, freqs_allowed).abs()
    # print("mags", mags.shape, "dft", dft.shape, "tensor.shape", tensor.shape)
    for dimsum in sumlist: dim = dim - 1 if dimsum < dim else dim
    if sumlist != []: mags = mags.sum(sumlist)
    mags, freqs = mags.topk(k, dim=dim)
    freqs = freqs_allowed[freqs]
    if squeeze: freqs, mags = freqs.squeeze(), mags.squeeze()
    # print("mags", mags.shape, "freqs", freqs.shape)
    return freqs, mags

def print_top_k_freqs(tensor, n, dim, sumlist = []):
    top4, mags = get_top_k_freqs(tensor, n, dim, sumlist)
    print("top", n, "freqs of", tensor.shape[dim], end=" [")
    for i in len(min(top4.numel())):
        print(round(top4[i].item(),3), end=" ")
    print("}")

def get_dft_coeffs(x, dim=-1, *, freqs=None, include_negative=False):
    coeffs = torch.fft.fft(x, dim=dim)
    if freqs is not None:
        freq_tensor = (freqs if isinstance(freqs, torch.Tensor) else torch.tensor(freqs)).to(coeffs.device)
        if include_negative: freq_tensor = torch.cat((freq_tensor, x.shape[dim] - freq_tensor), 0)
        coeffs = torch.index_select(coeffs, dim, freq_tensor)
    return coeffs

def get_mags_and_phases(x, dim=-1, *, freqs=None, include_negative=False):
    coeffs = get_dft_coeffs(x, dim, freqs=freqs, include_negative=include_negative)
    return torch.abs(coeffs), torch.angle(coeffs)

def get_phase(x, dim=-1, *, freqs=None, include_negative=False):
    coeffs = get_dft_coeffs(x, dim, freqs=freqs, include_negative=include_negative)
    return torch.angle(coeffs)

def get_central_phase(x, freq, dim=-1):
    coeffs = get_dft_coeffs(x, dim, freqs=freq)
    return torch.angle(coeffs.sum())

def normalize_tensor(tensor):
    min_val, max_val = tensor.min(), tensor.max() + 0.0000000001
    if max_val - min_val < 0.00000001: print("no data to normalize")
    return (tensor - min_val) / (max_val - min_val)

class CacheDict(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        for k in self.keys():
            if key == k or "hook_" + key in k:
                return dict.__getitem__(self, k)
        ks = [k for k in self.keys() if key in k]
        if len(ks) == 0: print(key, "not a substring of any keys")
        elif key in ks: return dict.__getitem__(self, key)
        elif len(ks) > 1: print("more than one key contains substring", key)
        else: 
            # print("found substring", key, "in", ks[0])
            return dict.__getitem__(self, ks[0])

    def __setitem__(self, key, val): dict.__setitem__(self, key, val)
    def __repr__(self): return '%s(%s)' % (type(self).__name__, dict.__repr__(self))
        
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
    
def intersecting(a,b): return a[(a.view(1, -1) == b.view(-1, 1)).any(dim=0)]

def expand_x_left_to_y(x, y):
    while x.ndim < y.ndim: x = x.unsqueeze(0)
    return x.expand_as(y)
def expand_x_right_to_y(x, y):
    while x.ndim < y.ndim: x = x.unsqueeze(-1)
    return x.expand_as(y)
def expand_all_left(target, *args): return tuple([expand_x_left_to_y(x, target) for x in args])
def expand_all_right(target, *args): return tuple([expand_x_right_to_y(x, target) for x in args])

def squeeze_cache(cachein, last_pos_only=False):
    cache = CacheDict(cachein)
    for name, t in cachein.items():
        for i in reversed(range(t.ndim)):
            # if t.size(i) == 114:
            #     t = t.narrow(i, 0, 113)
            if last_pos_only and t.size(i) == 3 and "attn.hook" not in name and "resid_pre" not in name and "embed" not in name:
                t = t.index_select(i, torch.full((1,), 2, dtype=torch.long).to(t.device))
        cache[name] = t.squeeze()
    return cache

def get_data(*, device=None, p=113):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = torch.tensor([(i, j, p) for i in range(p) for j in range(p)])
    labels = (dataset[:, 0] + dataset[:, 1]) % p
    return dataset.to(device), labels.to(device)

def get_old_indices(seed=0, frac_train=0.3, p=113, *, device=None):
    def gen_train_test(frac_train, num, seed):
        # Generate train and test split
        pairs = [(i, j, num) for i in range(num) for j in range(num)]
        random.seed(seed)
        random.shuffle(pairs)
        div = int(frac_train*len(pairs))
        return pairs[:div], pairs[div:]

    train, _ = gen_train_test(frac_train, p, seed)

    # Creates an array of Boolean indices according to whether each data point is in 
    # train or test
    # Used to index into the big batch of all possible data
    is_train = []
    is_test = []
    for x in range(p):
        for y in range(p):
            if (x, y, 113) in train:
                is_train.append(True)
                is_test.append(False)
            else:
                is_train.append(False)
                is_test.append(True)
    is_train = torch.tensor(is_train)
    is_test = torch.tensor(is_test)
    if device is not None:
        is_train = is_train.to(device)
        is_test = is_test.to(device)
    return is_train, is_test

def get_new_indices(seed=598, frac_train=0.3, p=113, *, device='cpu'):
    torch.manual_seed(seed)
    indices = torch.randperm(p*p)
    cutoff = int(p*p*frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]
    return train_indices.to(device), test_indices.to(device)

def get_fourier_basis(N, *, device=None):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    basis = []
    basis_names = []
    basis.append(torch.ones(N))
    basis_names.append("Constant")
    for freq in range(1, N//2+1):
        basis.append(torch.sin(torch.arange(N)*2 * torch.pi * freq / N))
        basis_names.append(f"Sin {freq}")
        basis.append(torch.cos(torch.arange(N)*2 * torch.pi * freq / N))
        basis_names.append(f"Cos {freq}")
    basis = torch.stack(basis, dim=0).to(device)
    basis = basis/basis.norm(dim=-1, keepdim=True).to(device)
    return basis, basis_names

def get_2D_fourier_basis(N, M, *, device=None):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    bases = []
    bases_names = []
    xbasis, xbasis_names = get_fourier_basis(N, device=device)
    bases.append(xbasis)
    bases_names.append(xbasis_names)
    ybasis, ybasis_names = get_fourier_basis(M, device=device)
    bases.append(ybasis.T)
    bases_names.append(ybasis_names)
    return bases, bases_names

def pull_out_freqs(x, freqs, *, dims=[-1], keep_dc=True):
    if not isinstance(dims, list) and not isinstance(dims, tuple): dims = [dims]
    if not isinstance(freqs, torch.Tensor): freqs = torch.tensor(freqs, dtype=torch.long, device=x.device)
    freqs = freqs[freqs > 0]
    x = x.repeat([len(freqs) + 2] + [1 for i in range(x.ndim)])
    for dim in dims:
        if dim >= 0: dim = dim - x.ndim
        X = torch.fft.fft(x, dim=dim)
        fspec = torch.zeros_like(X)
        fspec[-1] = X[-1]
        for f in range(freqs.shape[0]):
            freq = freqs[f].unsqueeze(-1)
            if keep_dc: freq = torch.cat((torch.tensor([0], dtype=torch.long, device=x.device), freq))
            fspec[f+1].index_copy_(dim, freq, torch.index_select(X[f+1], dim, freq))
            fspec[f+1].index_copy_(dim, X.size(dim) - freq[freq > 0], torch.index_select(X[f+1], dim, X.size(dim) - freq[freq > 0]))
        if not keep_dc: freqs = torch.cat((torch.tensor([0], dtype=torch.long, device=x.device), freqs))
        fspec[-1].index_fill_(dim, freqs, 0.0)
        fspec[-1].index_fill_(dim, fspec.size(dim) - freqs[freqs > 0], 0.0)

        x[1:] = torch.fft.ifft(fspec[1:], dim=dim).real
    return x

def zero_out_freqs(x, freqs, dim=-1):
    if not isinstance(freqs, torch.Tensor): freqs = torch.tensor(freqs, device=x.device, dtype=torch.long)
    if not isinstance(dim, list): dim = [dim]
    for d in dim:
        if d != -1: x = torch.swapaxes(x, d, -1)
        X = torch.fft.fft(x)
        X[...,freqs] = 0
        X[...,X.size(-1) - freqs[freqs > 0]] = 0
        x = torch.fft.ifft(X).real
        if d != -1: x = torch.swapaxes(x, -1, d)
    return x
