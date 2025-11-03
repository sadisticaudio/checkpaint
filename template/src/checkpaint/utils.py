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
        ret = "["
        for t in x: ret += tstr(t) + " "
        if ret[-1:] == " ": ret = ret[:-1]
        return ret + "]"
        # return "[" + str([tstr(t) + " " for t in x]) + "]"
    elif isinstance(x, dict):
        ret = ""
        maxlen = find_max_length_key(x)
        for name in x:
            # if isinstance(x[name], torch.Tensor): print("dict key", str(name) + ":", tstr(x[name]))
            ret += str(name) + str(": " + " " * (maxlen - len(str(name)))) + tstr(x[name]) + '\n'
        return ret
    elif isinstance(x, tuple): 
        ret = "(" + tstr(round_nested_list(x))[1:-1]
        if ret[-1:] == " ": ret = ret[:-1]
        return ret + ")"
        # return "(" + tstr(round_nested_list(x))[1:-1] + ")"
    elif isinstance(x, torch.Size): return tstr(list(x))
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

def clamp_inf_to_bounds(tensor):
    finite_vals = tensor[torch.isfinite(tensor)]
    if finite_vals.numel() == 0 and tensor.numel() > 0:
        tensor[:] = 0
    elif tensor.numel() > 0:
        min_val = finite_vals.min()
        max_val = finite_vals.max()
        # tensor = tensor.clone()
        tensor[tensor == float('inf')] = max_val
        tensor[tensor == float('-inf')] = min_val
    return tensor

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

# def get_top_k_freqs(tensor, k, dim=-1, *, sumlist=[], freqs_allowed=[], squeeze=False, return_phases=False):
#     if sumlist != []: sumlist = [(x + tensor.ndim) % tensor.ndim for x in sorted(sumlist)]
#     dim = (tensor.ndim + dim) % tensor.ndim
#     dft = torch.fft.fft(tensor, dim=dim)
#     if freqs_allowed == []: freqs_allowed = [x for x in range(tensor.shape[dim]//2+1)]
#     if not isinstance(freqs_allowed, torch.Tensor): freqs_allowed = torch.tensor(freqs_allowed, device=tensor.device, dtype=torch.long)
#     mags = torch.index_select(dft, dim, freqs_allowed).abs()
#     phases = torch.index_select(dft, dim, freqs_allowed).angle()
#     # print("mags", mags.shape, "dft", dft.shape, "tensor.shape", tensor.shape)
#     for dimsum in sumlist: dim = dim - 1 if dimsum < dim else dim
#     if sumlist != []: mags = mags.sum(sumlist)
#     mags, freqs = mags.topk(k, dim=dim)
#     freq_idx = freqs
#     while freq_idx.ndim < phases.ndim: freq_idx = freq_idx[None]
#     tprint("doing torch.gather(phases, dim, freq_idx)", "phases", phases.shape, "freq_idx", freq_idx.shape)
#     phases = torch.gather(phases, dim, freq_idx)
#     freqs = freqs_allowed[freqs]
#     if squeeze: freqs, mags, phases = freqs.squeeze(), mags.squeeze(), phases.squeeze()
#     # print("mags", mags.shape, "freqs", freqs.shape)
#     if return_phases: return freqs, mags, phases
#     else: return freqs, mags

def phase_shift_rows(X: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """
    Phase–shift each row of X by row_idx * factor * 2π / M.

    Args:
        X: (M, N) real or complex tensor.
        factor: scaling factor for phase increment (1.0 → 2π/M, 2.0 → 4π/M, etc.)

    Returns:
        Y: (M, N) tensor with shifted rows.
    """
    M, N = X.shape
    # FFT along columns (dim=-1), row by row
    Xf = torch.fft.rfft(X, dim=-1)  # (M, N//2+1)

    # frequency bin indices
    k = torch.arange(Xf.size(-1), device=X.device)  # (N//2+1,)

    # phase shift per row: row_idx * factor * 2π / M
    row_idx = torch.arange(M, device=X.device)[:, None]  # (M,1)
    phase = torch.exp(-1j * (row_idx * factor * N * 2 * torch.pi / M) * k[None, :] / N)

    # apply shift in frequency domain
    Yf = Xf * phase

    # back to time domain
    Y = torch.fft.irfft(Yf, n=N, dim=-1)
    return Y

def get_top_k_freqs(tensor, k, dim=-1, *, sumlist=[], freqs_allowed=[], squeeze=False, return_phases=False):
    if sumlist != []: sumlist = [(x + tensor.ndim) % tensor.ndim for x in sorted(sumlist)]
    dim = (tensor.ndim + dim) % tensor.ndim
    dft = torch.fft.fft(tensor, dim=dim)
    if freqs_allowed == []: freqs_allowed = [x for x in range(tensor.shape[dim]//2+1)]
    if not isinstance(freqs_allowed, torch.Tensor): freqs_allowed = torch.tensor(freqs_allowed, device=tensor.device, dtype=torch.long)
    mags = torch.index_select(dft, dim, freqs_allowed).abs()
    phases = torch.index_select(dft, dim, freqs_allowed).angle()
    # print("mags", mags.shape, "dft", dft.shape, "tensor.shape", tensor.shape)
    for dimsum in sumlist: dim = dim - 1 if dimsum < dim else dim
    if sumlist != []: mags = mags.sum(sumlist)
    mags, freqs = mags.topk(k, dim=dim)
    # tprint("mags", mags.shape, mags)
    # tprint("freqs", freqs.shape, freqs)

    freq_idx = freqs
    while freq_idx.ndim < phases.ndim: freq_idx = freq_idx[None]
    
    # this doesn't work that's why i've zeroed it out below, fix it
    # tprint("doing torch.gather(phases, dim, freq_idx)", "phases", phases.shape, "freq_idx", freq_idx.shape)
    phases = torch.gather(phases, dim, freq_idx)
    phases = torch.zeros_like(phases)
    
    # phases = torch.gather(phases, dim, freqs)#phases[...,freqs]
    # tprint("phases", phases.shape, phases)
    
    freqs = freqs_allowed[freqs]
    if squeeze: freqs, mags, phases = freqs.squeeze(), mags.squeeze(), phases.squeeze()
    # print("mags", mags.shape, "freqs", freqs.shape)
    if return_phases: return freqs, mags, phases
    else: return freqs, mags

def print_top_k_freqs(tensor, n, dim, sumlist = []):
    top4, mags = get_top_k_freqs(tensor, n, dim, sumlist)
    print("top", n, "freqs of", tensor.shape[dim], end=" [")
    for i in len(min(top4.numel())):
        print(round(top4[i].item(),3), end=" ")
    print("}")

def get_dft_coeffs(x, dim=-1, *, freqs=None, include_negative=False):
    if freqs is None: freqs = [i for i in range(x.size(dim)//2+1)]
    coeffs = torch.fft.fft(x, dim=dim)
    if freqs is not None:
        freq_tensor = (freqs if isinstance(freqs, torch.Tensor) else torch.tensor(freqs)).to(coeffs.device)
        if include_negative: freq_tensor = torch.cat((freq_tensor, x.shape[dim] - freq_tensor), 0)
        coeffs = torch.index_select(coeffs, dim, freq_tensor)
    return coeffs

def get_mags_and_phases(x, dim=-1, *, freqs=None, include_negative=False):
    coeffs = get_dft_coeffs(x, dim, freqs=freqs, include_negative=include_negative)
    return torch.abs(coeffs)/(x.size(dim)/2), torch.angle(coeffs)

def from_mags_and_phases(freqs, mags, phases, length, sample_rate, composite=True):
    if not isinstance(freqs, torch.Tensor): freqs = torch.tensor(freqs)
    while freqs.ndim < mags.ndim: freqs = freqs[None]
    # assert len(freqs) == mags.size(-1) and len(freqs) == phases.size(-1), "freqs " + tstr(len(freqs)) + " not same as mags/phases shape" + tstr(mags.shape)
    # while len(freqs) < mags.ndim: freqs = freqs[None]
    tprint("mags", mags.shape, "length", length, "freqs", freqs.shape, "phases", phases.shape)
    waves = mags * torch.cos(torch.arange(length)[...,None] * freqs[...,None] * 2 * np.pi / sample_rate + phases[None])
    if composite: waves = waves.sum(-1)
    return waves

def get_phase(x, dim=-1, *, freqs=None, include_negative=False):
    coeffs = get_dft_coeffs(x, dim, freqs=freqs, include_negative=include_negative)
    return torch.angle(coeffs)

def get_central_phase(x, freq, dim=-1):
    coeffs = get_dft_coeffs(x, dim, freqs=freq)
    return torch.angle(coeffs.sum())

def get_2D_freq_idx(freqs, lengths_to_include_negative=None):
    '''
        freqs will be converted to (row_idx, col_idx)
        if freqs is a single list or tensor, a len(freqs)^^2 grid with each pair will be covered
        i.e. if freqs == [5,6,7], this will yield [[5,5,5,6,6,6,7,7,7], [5,6,7,5,6,7,5,6,7]]
        if freqs is already a container with 2 tensors or lists, it will be used directly
    '''
    if freqs is None:
        tprint("freqs should not be none typicall for 2D DFT stuff...")
        return (slice(None), slice(None))
    if isinstance(freqs, torch.Tensor) or isinstance(freqs[0], int):
        row_idx = torch.as_tensor(freqs).clone().repeat_interleave(len(freqs))
        col_idx = torch.as_tensor(freqs).clone().repeat(len(freqs))
    else:
        row_idx = freqs[0]
        col_idx = freqs[1]
    freq_idx = [row_idx,col_idx]
    if not isinstance(freq_idx[0], torch.Tensor): freq_idx = [torch.tensor(list(x)) for x in freq_idx]
    if lengths_to_include_negative: 
        freq_idx[0] = torch.cat((freq_idx[0], (lengths_to_include_negative[0] - freq_idx[0]) % lengths_to_include_negative[0]), 0)
        freq_idx[1] = torch.cat((freq_idx[1], (lengths_to_include_negative[1] - freq_idx[1]) % lengths_to_include_negative[1]), 0)
        # current_length = len(freq_idx[0])
        # for i in range(current_length):
        #     if torch.nonzero(freq_idx[0][i] != 0) + torch.nonzero(freq_idx[1][i] != 0) == 2:
        #         freq_idx[0] = torch.cat((freq_idx[0], lengths_to_include_negative[0] - freq_idx[0][i:i+1]), 0)
        #         freq_idx[1] = torch.cat((freq_idx[1], lengths_to_include_negative[1] - freq_idx[1][i:i+1]), 0)
    return tuple(freq_idx)

def get_2D_dft_coeffs(x, dim=(-2,-1), *, freqs=None, include_negative=False):
    s = (x.size(dim[0]), x.size(dim[1]))
    if freqs is None: freqs = [i for i in range(min(x.size(dim[0])//2+1, x.size(dim[1])//2+1))]
    freq_idx = get_2D_freq_idx(freqs, s if include_negative else None)
    # tprint("freqs type", type(freqs), "freq_idx type", type(freq_idx), "row_idx", row_idx.shape, "col_idx", col_idx.shape)

    if include_negative: 
        freq_idx[0] = torch.cat((freq_idx[0], x.shape[dim] - freq_idx[0]), 0)
        freq_idx[1] = torch.cat((freq_idx[1], x.shape[dim] - freq_idx[1]), 0)

    coeffs = torch.fft.fft2(x, s=s, dim=dim, norm="forward") * 2
    coeffs = coeffs[...,*freq_idx]
    
    return coeffs

def get_2D_mags_and_phases(x, dim=(-2,-1), *, freqs=None, include_negative=False):
    coeffs = get_2D_dft_coeffs(x, dim, freqs=freqs, include_negative=include_negative)
    return torch.abs(coeffs), torch.angle(coeffs)

def synth_2D_from_coeffs(coeffs, freqs, W, H, *, dim=(-2,-1), include_negative=False):
    mags, phases = torch.abs(coeffs), torch.angle(coeffs)
    shape = mags.shape[:-2] + ((W,H) if include_negative else (W//2+1,H//2+1))
    X = torch.zeros(shape, dtype=torch.complex64)
    s = (W, H)
    if freqs is None: freqs = [i for i in range(min(W//2+1, H//2+1))]
    freq_idx = get_2D_freq_idx(freqs, s if include_negative else None)

    X[...,*freq_idx] = torch.polar(mags * W * H / 2, phases)
    if include_negative == True: return torch.fft.ifft2(X, s=(W,H), dim=dim).real
    else: return torch.fft.irfft2(X, (W,H), dim).real

def synth_2D_from_mags_and_phases(mags, phases, freqs, W, H, *, dim=(-2,-1), include_negative=False):
    return synth_2D_from_coeffs(torch.complex(mags,phases), freqs, W, H, dim=dim, include_negative=include_negative)

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
# def pad_axes_left(N, *args):
def tallclose(x,y): return torch.allclose(x,y,rtol=1e-4,atol=1e-3)
    
def prepare_for_optimizer(*xs, device='cuda'):
    ''' returns a big tuple of prepared tensors plus a list of them to actually hand to the optimizer '''
    to_optimize = []
    for x in xs:
        p_x = x.detach().clone()
        p_x = p_x.to(device)
        p_x.requires_grad_(True)
        to_optimize.append(p_x)
    return tuple(to_optimize) + (to_optimize,)

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

def pull_out_freqs(x, freqs, *, dims=[-1], keep_dc=True, seperate=True):
    if not isinstance(dims, list) and not isinstance(dims, tuple): dims = [dims]
    if not isinstance(freqs, torch.Tensor): freqs = torch.tensor(list(freqs), dtype=torch.long, device=x.device)
    freqs = freqs[freqs > 0]
    x = x.repeat([len(freqs) + 1 if seperate else 2] + [1 for i in range(x.ndim)])
    for dim in dims:
        if dim >= 0: dim = dim - x.ndim
        X = torch.fft.fft(x, dim=dim)
        fspec = torch.zeros_like(X)
        fspec[-1] = X[-1]
        if seperate:
            for f in range(freqs.shape[0]):
                freq = freqs[f].unsqueeze(-1)
                if keep_dc: freq = torch.cat((torch.tensor([0], dtype=torch.long, device=x.device), freq))
                fspec[f+1].index_copy_(dim, freq, torch.index_select(X[f+1], dim, freq))
                fspec[f+1].index_copy_(dim, X.size(dim) - freq[freq > 0], torch.index_select(X[f+1], dim, X.size(dim) - freq[freq > 0]))
        else:
            for f in range(freqs.shape[0]):
                freq = freqs[f].unsqueeze(-1)
                if keep_dc: freq = torch.cat((torch.tensor([0], dtype=torch.long, device=x.device), freq))
                fspec[1].index_add_(dim, freq, torch.index_select(X[1], dim, freq))
                fspec[1].index_add_(dim, X.size(dim) - freq[freq > 0], torch.index_select(X[1], dim, X.size(dim) - freq[freq > 0]))
        if not keep_dc: freqs = torch.cat((torch.tensor([0], dtype=torch.long, device=x.device), freqs))
        fspec[-1].index_fill_(dim, freqs, 0.0)
        fspec[-1].index_fill_(dim, fspec.size(dim) - freqs[freqs > 0], 0.0)

        x[1:] = torch.fft.ifft(fspec[1:], dim=dim).real
    return x

def zero_out_freqs(x, freqs, dim=-1):
    if not isinstance(freqs, torch.Tensor): freqs = torch.tensor(list(freqs), device=x.device, dtype=torch.long)
    if not isinstance(dim, (list,tuple)): dim = [dim]
    for d in dim:
        if d >= 0: d -= x.ndim
        if d != -1: x = torch.swapaxes(x, d, -1)
        X = torch.fft.fft(x)
        dim_freqs = freqs[freqs < (x.size(-1)//2 + 1 if x.size(-1) % 2 == 1 else 0)]
        X[...,dim_freqs] = 0
        X[...,X.size(-1) - dim_freqs[dim_freqs > 0]] = 0
        x = torch.fft.ifft(X).real
        if d != -1: x = torch.swapaxes(x, -1, d)
    return x

def zero_out_freqs_2D(x, freqs, dim=(-2,-1)):
    s = (x.size(dim[0]), x.size(dim[1]))
    freq_idx = get_2D_freq_idx(freqs, s)
    # tprint("s", s, "freq_idx", freq_idx)
    X = torch.fft.fft2(x, s=s, dim=dim)
    X[...,*freq_idx] = 0
    return torch.fft.ifft2(X, s=s, dim=dim).real

def zero_all_but_freqs_2D(x, freqs, dim=(-2,-1)):
    s = (x.size(dim[0]), x.size(dim[1]))
    freq_idx = get_2D_freq_idx(freqs, s)
    # tprint("s", s, "freq_idx", freq_idx)
    X = torch.fft.fft2(x, s=s, dim=dim)
    NEW_X = torch.zeros_like(X)
    NEW_X[...,*freq_idx] = X[...,*freq_idx]
    return torch.fft.ifft2(NEW_X, s=s, dim=dim).real

# def zero_all_but_freqs(x, freqs, dim=-1, *, keep_dc=True):
#     if not isinstance(freqs, torch.Tensor): freqs = torch.tensor(list(freqs), device=x.device, dtype=torch.long)
#     if keep_dc: freqs = torch.cat((torch.arange(1,device=x.device), freqs),0)
#     if not isinstance(dim, list): dim = [dim]
#     top_len = x.size(dim[0])
#     for d in dim: top_len = max(top_len, x.size(d))
#     max_freq = (top_len//2 + 1 if top_len % 2 == 1 else 0)
#     all_but_freqs = torch.arange(max_freq, device=x.device, dtype=torch.long)
#     mask = ~torch.isin(all_but_freqs, freqs)
#     all_but_freqs = all_but_freqs[mask]
#     return zero_out_freqs(x, all_but_freqs, dim)

def zero_all_but_freqs(x, freqs, dim=-1, *, keep_dc=True):
    if not isinstance(freqs, torch.Tensor): freqs = torch.tensor(list(freqs), device=x.device, dtype=torch.long)
    if keep_dc and 0 not in freqs.tolist(): freqs = torch.cat((torch.zeros_like(freqs[0:1]), freqs),0)
    if not isinstance(dim, (list, tuple)): dim = [dim]
    for d in dim:
        X = torch.fft.rfft(x, x.size(d), d)
        NEWX = torch.zeros_like(X)
        NEWX.index_copy_(d, freqs, torch.index_select(X, d, freqs))
        x = torch.fft.irfft(NEWX, x.size(d), d)
    return x

def zero_all_but_freq_2D(x, freqx, freqy):
    X = torch.fft.rfft2(x)
    X[...,:freqx,:] = 0
    X[...,freqx+1:,:] = 0
    X[...,:freqy] = 0
    X[...,freqy+1:] = 0
    return torch.fft.irfft2(X, s=(x.size(-2), x.size(-1)))

def x_correlate(x, y, dim, *, keepdim=False):
    if y.ndim == 1:
        while y.ndim <= dim: y = y[None]
        while y.ndim < x.ndim: y = y[...,None]
    x = x.transpose(dim, -1)
    y = y.transpose(dim, -1)
    corr = torch.einsum(" ...i, ...i -> ... ", x, y)
    if keepdim: 
        corr = corr[...,None]
        corr = corr.transpose(-1, dim)
    return corr

def wrap_phase(phase_tensor): return (phase_tensor + torch.pi) % (2 * torch.pi) - torch.pi

def unwrap_phase(x: torch.Tensor, dim: int = -1, *, period: float = 2*torch.pi, discont: float | None = None) -> torch.Tensor:
    """
    Unwrap a phase-like signal along `dim` so jumps larger than `discont` are corrected by ±period.
    - x: real phases in radians, or complex tensor (angle will be used)
    - dim: dimension along which to unwrap (default last)
    - period: wrap period (default 2π)
    - discont: jump threshold; defaults to period/2 (i.e., π for 2π period)
    """
    if discont is None:
        discont = period / 2

    # Convert complex to angle (radians)
    phi = x.angle() if torch.is_complex(x) else x

    # If length <= 1 along dim, nothing to do
    if phi.size(dim) <= 1:
        return phi.clone()

    # Differences along the unwrap dimension
    dphi = phi.diff(dim=dim)

    # Map differences to (-discont, discont]
    # Equivalent of: dmod = (dphi + discont) % period - discont
    dmod = (dphi + discont)
    remainder = (dphi + discont).remainder(period) - discont
    # PyTorch doesn't guarantee exact equality around boundaries due to floating point.
    # Handle the special case like numpy.unwrap: when dmod == -discont and original dphi > 0,
    # prefer +discont (i.e., add one period)
    eps = torch.finfo(phi.dtype).eps if phi.dtype.is_floating_point else 1e-12
    mask = (remainder <= (-discont + eps)) & (dphi > 0)
    dmod = torch.where(mask, remainder + period, remainder)

    # Correction to apply cumulatively
    corr = dmod - dphi
    corr_cumsum = torch.cumsum(corr, dim=dim)

    # Build the unwrapped result: first slice unchanged, rest plus cumulative correction
    # Take first element along dim
    idx_first = [slice(None)] * phi.ndim
    idx_first[dim] = slice(0, 1)
    first = phi[tuple(idx_first)]

    # Take remaining elements
    idx_rest = [slice(None)] * phi.ndim
    idx_rest[dim] = slice(1, None)
    rest = phi[tuple(idx_rest)] + corr_cumsum

    return torch.cat([first, rest], dim=dim)
    
def normalize_phase(phases: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Shift phases by ±2π so that the mean along `dim` is close to 0.
    Keeps data continuous after unwrap.
    """
    phases = unwrap_phase(phases)
    two_pi = 2 * torch.pi
    # Mean along chosen dimension (preserve dims for broadcasting)
    mean = phases.mean(dim=dim, keepdim=True)
    # Compute integer multiple of 2π closest to mean
    shift = torch.round(mean / two_pi) * two_pi
    return phases - shift
    
def get_relu_magnitude(B=0, *, device='cpu'):
    """
    This function was first written by ChatGPT but painstakingly corrected... haha
    Computes the fundamental Fourier coefficient for a ReLU applied to a biased cosine wave.
    
    Args:
        B (float): DC bias, should be between -1 and 1.
        
    Returns:
        # tuple:
        #     -coeff: The Fourier coefficient.
            -magnitude
            # -phase
    """

    if not isinstance(B, torch.Tensor): B = torch.tensor([B], device=device)
    ret = torch.zeros_like(B)

    B = torch.clamp(B, -1.0 + 1e-18, 1.0 - 1e-18)

    # FIX THIS!!!  ...HAD TO FLIP B DUE TO AN ERROR BY CHATGPT
    B = -B
    theta = torch.arccos(-B)
    t1 = theta / (2 * torch.pi)
    t2 = 1 - t1

    j = 1j
    term1 = (B / (-j * 2 * torch.pi)) * (torch.exp(-j * 2 * torch.pi * t2) - torch.exp(-j * 2 * torch.pi * t1))
    term2 = 0.5 * (t2 - t1)
    term3 = (1 / (2 * (-j * 4 * torch.pi))) * (torch.exp(-j * 4 * torch.pi * t2) - torch.exp(-j * 4 * torch.pi * t1))
    
    coeff = term1 + term2 + term3
    magnitude = torch.abs(coeff) * 2
    return magnitude

def sinusoidal_norm(x, dim=-1, *, reduce=True): return x.square().mean(dim).sqrt().mean() if reduce else x.square().mean(dim).sqrt()

def normalize_like(x, y, * , dims=None, normalize_mean=False):
    if dims is None: dims = tuple([i for i in range(x.ndim)])
    xmean, ymean = x.mean(dims, True), y.mean(dims, True)
    x = x + ymean - xmean if normalize_mean else x
    for d in dims: x = x * torch.linalg.norm(y, None, d, True)/torch.linalg.norm(x, None, d, True)
    return x

def remove_mean(x, * , dims=None):
    if dims is None: dims = tuple([i for i in range(x.ndim)])
    for d in dims: x = x - x.mean(d, True)
    return x

def torch_interp1d(x, y, dim, x_new):
    """
    PyTorch linear interpolation, similar to scipy.interpolate.interp1d.

    Args:
        x (Tensor): 1D tensor of original x positions, shape (N,)
        y (Tensor): Tensor of y values corresponding to x, shape (..., N, ...)
                    where interpolation happens along the 'dim' axis.
         dim (int): Axis of y along which to interpolate.
         x_new (Tensor): 1D tensor of new x positions to interpolate, shape (M,)

    Returns:
        Tensor: Interpolated y values at x_new, same shape as y but with size M along 'dim'.
    """
    # Ensure x is sorted
    assert torch.all(x[1:] > x[:-1]), "x must be strictly increasing"

    y = y.transpose(dim, -1)
    orig_shape = y.shape
    N = orig_shape[-1]

    if x.ndim == 1:
        while x.ndim < y.ndim:
            x = x[None]
            x_new = x_new[None]
    else:
        x = x.transpose(dim, -1)
        x_new = x_new.transpose(dim, -1)

    idx = torch.searchsorted(x.contiguous(), x_new.contiguous(), right=False)
    idx = torch.clamp(idx, 1, N - 1)

    def expand_to_x(a,n):
        while a.ndim < n: a = a[None]
        while a.ndim < x.ndim: a = a[...,None]
        return a.expand_as(x)

    all_idx = tuple([expand_to_x(torch.arange(x.size(a)), a) for a in range(x.ndim - 1)] + [idx])
    all_idx_minus_one = tuple([expand_to_x(torch.arange(x.size(a)), a) for a in range(x.ndim - 1)] + [idx - 1])

    x0 = x[all_idx_minus_one]
    x1 = x[all_idx]

    y0 = y[all_idx_minus_one]
    y1 = y[all_idx]
    t = (x_new - x0) / (x1 - x0)
    y_interp = y0 + t * (y1 - y0)
    y_interp = y_interp.transpose(-1, dim)

    return y_interp

def torch_interp_like(x_ref, y, dim, x_new_ref, *, assume_sorted=True, circular=False, period=2*np.pi):
    """
    Interpolate values `y` along axis `dim` using the interpolation weights induced
    by mapping from reference positions `x_ref` -> `x_new_ref`.

    Typical use: x_ref are uneven (sorted) phases, y are magnitudes aligned to those phases,
    x_new_ref are evenly spaced phases; this resamples magnitudes onto the even phase grid.

    Args:
        x_ref (Tensor): 1D tensor of original reference positions (e.g., phases), shape (N,).
        y (Tensor):     Tensor of values aligned with x_ref along `dim`, shape (..., N, ...).
        dim (int):      Axis of y along which to interpolate.
        x_new_ref (Tensor): 1D tensor of target reference positions, shape (M,).
        assume_sorted (bool): If False, will sort x_ref (and y along `dim`) first.
        circular (bool): If True, treat x_ref as circular with given `period` (e.g., 2π).
                         We cut at the largest gap to unwrap, then roll x_ref and y consistently.
        period (float): Period for circular domain.

    Returns:
        Tensor: y resampled at x_new_ref along `dim`, same shape as y but with size M on `dim`.
    """
    # Normalize dim
    dim = dim % y.ndim

    # Move interpolation axis to the end for simpler indexing
    yT = y.transpose(dim, -1).contiguous()
    N = yT.shape[-1]
    assert x_ref.ndim == 1 and x_new_ref.ndim == 1, "x_ref and x_new_ref must be 1D"
    assert x_ref.shape[0] == N, "x_ref length must match y.size(dim)"

    # Optionally sort x_ref (and y) if not guaranteed sorted
    if not assume_sorted:
        x_sorted, order = torch.sort(x_ref)
        yT = yT[..., order]
    else:
        x_sorted = x_ref

    # Optional circular handling: find biggest gap, cut there, and roll
    if circular:
        # Bring x into a continuous increasing ramp by allowing a single cut
        xs = x_sorted
        # Differences including last->first (wrapped by +period)
        diffs = torch.empty_like(xs)
        diffs[:-1] = xs[1:] - xs[:-1]
        diffs[-1]   = (xs[0] + period) - xs[-1]
        cut = torch.argmax(diffs).item()  # index of largest gap ending at `cut`
        # Roll so that new start is after the largest gap
        roll = -(cut + 1) % N
        x_sorted = torch.remainder(xs.roll(roll), period)
        yT = yT.roll(roll, dims=-1)

        # Also wrap x_new_ref into the same [start, start+period) frame as x_sorted
        # We choose the frame where x_sorted.min() is near 0
        # Map x_new_ref to [0, period)
        x_new = torch.remainder(x_new_ref - x_sorted.min(), period)
    else:
        x_new = x_new_ref

    # Now x_sorted must be strictly increasing
    if not torch.all(x_sorted[1:] > x_sorted[:-1]):
        raise ValueError("x_ref must be strictly increasing (after optional sorting/unwrap).")

    # Prepare broadcastable views of x_sorted and x_new for searchsorted
    xs = x_sorted
    xn = x_new

    # searchsorted expects 1D for the reference and query; we’ll use it directly
    idx = torch.searchsorted(xs.contiguous(), xn.contiguous(), right=False)
    idx = torch.clamp(idx, 1, N - 1)  # in [1, N-1]

    # Gather neighbors
    x0 = xs[idx - 1]
    x1 = xs[idx]
    # Compute t in [0,1]
    t = (xn - x0) / (x1 - x0)
    # Shape them to be broadcastable with yT[..., N]
    # yT shape: (..., N)
    # idx shape: (M,)
    # We'll index last axis with idx/idx-1, then blend with t expanded
    y0 = yT[..., idx - 1]
    y1 = yT[..., idx]
    # Expand t to match leading dims of y0/y1
    while t.ndim < y0.ndim:
        t = t.unsqueeze(0)
    y_interp = y0 + t * (y1 - y0)

    # Put interpolation axis back to `dim`
    return y_interp.transpose(-1, dim)

def lerp_phase_data(data, phases, dim=-1):
    std_phases = torch.arange(phases.size(0)) * 2 * np.pi / phases.size(0) - np.pi
    return torch.clamp(torch_interp1d(phases, data, dim, std_phases), -np.pi, np.pi)

def lerp_like(data, phases, dim=-1):
    std_phases = torch.arange(phases.size(0)) * 2 * np.pi / phases.size(0) - np.pi
    return torch_interp_like(phases, data, dim, std_phases)

# def get_circulant_error(x, dims=[-2,-1], *, increment=-1):
#     '''
#         args:
#             -x          input
#             -dims       dim[0] is the dim over which the shift increments
#                         dim[1] is the dim that shifts, i.e. circulates/rolls (look at torch.roll)
#             -increment  the number of elements that the vectors shift incrementally

#         returns:
#             -error      difference between shifted mean and the sum of shifted data
#     '''
#     dM, dN, M, N = dims[0], dims[1], x.size(dims[0]), x.size(dims[1])
#     mean_of_dim = torch.zeros_like(x.narrow(dM, 0, 1))
#     for i in range(M): 
#         # tprint("x.narrow(dM, i, 1)", x.narrow(dM, i, 1))
#         mean_of_dim += torch.roll(x.narrow(dM, i, 1), (-increment * i) % N, dN) / M
#     error = 0.0
#     # tprint("dM, dN, M, N", dM, dN, M, N)
#     # tprint("mean_of_dim", mean_of_dim)
#     for i in range(M):
#         rolled = torch.roll(x.narrow(dM, i, 1), (-increment * i) % N, dN)
#         # tprint("rolled", rolled)
#         error += (rolled - mean_of_dim).abs().sum().item()
#     return error/(1e-10 + x.square().sqrt().sum().item())

def get_circulant_error(x, dims=[-2, -1], *, increment=-1):
    '''
    Args:
        x          : input tensor with at least 2 dimensions
        dims       : [dM, dN]
                     - dM is the dim over which the shift increments (circulant axis)
                     - dN is the dim that rolls (shifting elements within each vector)
        increment  : shift increment per step along dM

    Returns:
        Tensor of circulant error values with shape equal to batch dims
    '''
    dM, dN = tuple([d if d > -1 else d + x.ndim for d in dims])
    M, N = x.size(dM), x.size(dN)

    # Move dM and dN to known positions for easier indexing (e.g., last 2)
    permute_dims = [i for i in range(x.ndim) if i not in (dM, dN)] + [dM, dN]
    x_perm = x.permute(permute_dims)  # [..., M, N]
    
    # Compute mean over shifted dimension
    mean = 0
    for i in range(M):
        shift_amount = (-increment * i) % N
        mean += torch.roll(x_perm[..., i, :], shifts=shift_amount, dims=-1)
    mean /= M

    # Compute sum of absolute differences from mean
    error = 0
    for i in range(M):
        shift_amount = (-increment * i) % N
        rolled = torch.roll(x_perm[..., i, :], shifts=shift_amount, dims=-1)
        error += (rolled - mean).abs().sum(dim=-1)

    # Normalize by L1 norm of x (or you could use L2)
    norm = x_perm.abs().sum(dim=(-2, -1))
    return error / (norm + 1e-10)

# def make_square_coeffs(fracs, n_harms, base=3):
#     n_waves = fracs.size(0)
#     X = torch.zeros([n_waves, n_harms], dtype=torch.complex64)
#     freqs = base * torch.arange(n_harms) * 2 + base
#     for w in range(n_waves):
#         for i in range(n_harms):
#             h = i*2+1
#             X[w,i] = torch.exp(1j * base * h * -(p*fracs[w]) * 2 * np.pi / p) * (1j) / h
#     return X, freqs

# def make_triangle_coeffs(fracs, n_harms, base=3):
#     n_waves = fracs.size(0)
#     X = torch.zeros([n_waves, n_harms], dtype=torch.complex64)
#     freqs = base * torch.arange(n_harms) * 2 + base
#     for w in range(n_waves):
#         for i in range(n_harms):
#             h = i*2+1
#             A_m = (4.0 / np.pi) / h   # sine-series amplitude
#             X[w,i] += (1j) * (p / 2.0) * A_m
#             X[w,i] *= torch.exp(1j * base * h * -(p*fracs[w]) * 2 * np.pi / p)
#     return X, freqs