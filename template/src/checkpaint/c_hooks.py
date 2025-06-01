import torch
import numpy as np
from checkpaint.utils import *

p = 113
n_heads = 4

def cross_entropy_high_precision(logits, labels):
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()

def obey_nyquist(x): return p - x if x > 56.5 else -x if x < 0 else x

def get_harmonics(freqs):
    harmonics = torch.zeros_like(freqs) if isinstance(freqs, torch.Tensor) else list(freqs)
    for i, freq in enumerate(freqs): harmonics[i] = p - 2 * freq.item() if 2 * freq.item() > p//2 else 2 * freq
    return harmonics

def get_subharmonics(freqs):
    harmonics = torch.zeros_like(freqs) if isinstance(freqs, torch.Tensor) else list(freqs)
    for i, freq in enumerate(freqs): harmonics[i] = round(p/2 - freq.item()/2) if freq.item() % 2 == 1 else freq//2
    return harmonics

def get_sum_and_difference_frequencies(freqs, include_dc=False):
    sum_diffs = []
    for f1 in freqs:
        for f2 in freqs:
            big_key, small_key = max(f1,f2), min(f1,f2)
            sum, diff = obey_nyquist(big_key + small_key), obey_nyquist(big_key - small_key)
            if sum != 0 and sum not in sum_diffs: sum_diffs.append(sum)
            if diff != 0 and diff not in sum_diffs: sum_diffs.append(diff)
    sum_diffs = sorted(sum_diffs)
    if 0 in sum_diffs and not include_dc: sum_diffs = sum_diffs[1:]
    return torch.tensor(sum_diffs, device=freqs.device) if isinstance(freqs, torch.Tensor) else sum_diffs

def inputs_last(x):
    if x.size(0) == p*p:
        x = x.unflatten(0,(p,p))
    if x.ndim > 2 and x.size(0) == p and x.size(1) == p:
        x = torch.moveaxis(x, (0, 1), (-2, -1))
    return x
def inputs_first(x):
    x = x.flatten(-2,-1)
    x = torch.moveaxis(x,-1,0)
    return x
def all_inputs_last(cache):
    for name, tensor in cache.items():
        cache[name] = inputs_last(tensor)
    return cache

class BaseCorrelation:
    # takes all forms of the data and returns on tensor with data needed to determine if an example vector is correlated with all others
    def prepare_data(x, X, mags, phases, *, dim=-1): return x
    # takes one example vector from the prepared_data and returns a float tensor with ones where the prepared data is correlated with the example
    def get_correlated_vectors(example, prepared_data): 
        is_correlated = torch.zeros_like(prepared_data, dtype=torch.float)
        tprint("BaseCorr    BETTER NOT SEE THIS", "prepared_data", prepared_data.shape, "is_correlated", is_correlated.shape)
        return prepared_data[is_correlated.bool()], is_correlated
    # returns a unique hashable key of an example (that is selected from the prepared_data) used to identify the correlation
    def get_key(example, correlated): return example.data_ptr()

class PowerSpectrumCorrelation(BaseCorrelation):
    # spectra are each normalized to give the top frequency a mag of 1, this does not include the DC (bias) bin
    def prepare_data(x, X, mags, phases, *, dim=-1): return mags[...,1:] / torch.max(mags[...,1:], dim, keepdim=True)[0]
    # get the MSE between all spectra and the example spectrum
    # 0.2 is somewhat arbitrary..  a test with mags in hook_k found that ~0.0841 seemed to be
    # the threshold where the numbers changed from being the same spectrum to not the same
    def get_correlated_vectors(example, prepared_data): 
        is_correlated = torch.where((example - prepared_data).abs().sum(-1) < 0.2, 1.0, 0.0)
        correlated = prepared_data[is_correlated == 1.0]
        # tprint("PowerSpectrumCorrelation", "prepared_data", prepared_data.shape, "is_correlated", is_correlated.shape, "is_correlated.bool()", is_correlated.bool().shape, "correlated", correlated.shape)
        return correlated, is_correlated
    def get_key(example, correlated): return tuple(torch.mean(correlated, dim=list(range(correlated.ndim))[:-1]).tolist())

class DCCorrelation(BaseCorrelation):
    def prepare_data(x, X, mags, phases, *, dim=-1): return mags[...,0:1]
    def get_correlated_vectors(example, prepared_data):
        is_correlated = torch.where((example - prepared_data).abs() < 0.2, 1.0, 0.0).squeeze(-1)
        # tprint("DCCorrelation", "prepared_data", prepared_data.shape, "is_correlated", is_correlated.shape)
        return prepared_data[is_correlated.bool()], is_correlated
    def get_key(example, correlated): return tuple(torch.mean(correlated, dim=list(range(correlated.ndim))[:-1]).tolist())

class NegationCorrelation(BaseCorrelation):
    def prepare_data(x, X, mags, phases, *, dim=-1): 
        centered = x - x.mean(-1, keepdim=True)
        prepared = centered / torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        return prepared
    def get_correlated_vectors(example, prepared_data):
        is_correlated = torch.zeros_like(prepared_data[...,0], dtype=torch.float)
        nexample = -example
        is_neg = (prepared_data + nexample).sum(-1)/prepared_data.size(-1) < 0.2
        is_correlated[is_neg] = 1.0
        if torch.count_nonzero(is_correlated):
            is_pos = (prepared_data - example).sum(-1)/prepared_data.size(-1) < 0.2
            is_correlated[is_pos] = 1.0
        # tprint("NegationCorrelation", "prepared_data", prepared_data.shape, "is_correlated", is_correlated.shape)
        return prepared_data[is_correlated.bool()], is_correlated

class ReversalCorrelation(BaseCorrelation):
    def prepare_data(x, X, mags, phases, *, dim=-1): 
        centered = x - x.mean(-1, keepdim=True)
        prepared = centered / torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        return prepared
    def get_correlated_vectors(example, prepared_data):
        is_correlated = torch.zeros_like(prepared_data[...,0], dtype=torch.float)
        rexample = torch.flip(example, [-1])
        is_bwd = (prepared_data - rexample).sum(-1)/prepared_data.size(-1) < 0.2
        is_correlated[is_bwd] = 1.0
        if torch.count_nonzero(is_correlated):
            is_fwd = (prepared_data - example).sum(-1)/prepared_data.size(-1) < 0.2
            is_correlated[is_fwd] = 1.0
        # tprint("Reversal", "prepared_data", prepared_data.shape, "is_correlated", is_correlated.shape)
        return prepared_data[is_correlated.bool()], is_correlated

# class TranslationCorrelation(BaseCorrelation):
#     def get_correlated_vectors(example, prepared_data):
#         is_correlated = torch.zeros_like(prepared_data[...,0], dtype=torch.float)
#         return is_correlated

statistical_correlations = {
    "spectra":    PowerSpectrumCorrelation,
    "dc":         DCCorrelation,
    "negation":   NegationCorrelation,
    "reversal":   ReversalCorrelation,
}

def get_hook_correlations(hook_x, freqs, *, pos=slice(0,3), cpos=0, name=""):
    if isinstance(freqs, torch.Tensor): freqs = freqs.tolist()
    HOOK_X = torch.fft.fft(hook_x)
    mags, phases = torch.abs(HOOK_X)[...,[0] + freqs]/(p/2), torch.angle(HOOK_X)[...,[0] + freqs]
    mags[...,0] /= 2

    # return value: statistics: dict of info and correlations (see below)
    # statistics["correlations"]: a dict of statistical correlations keyed by name { name: correlation }
    # correlations are also dicts { corr_key: [indices] }:
    #   keys: can be different types depending on the correlation
    #   values: the correlated indices are a len(ndim) list containing either tensors of indices or slice objects
    statistics = { "name": name, "shape": hook_x[...,0].shape, "mags": mags, "phases": phases, "correlations": {} }

    for corr_name, Corr_Type in statistical_correlations.items():

        prepared_data = Corr_Type.prepare_data(hook_x, HOOK_X, mags, phases)
        remaining = torch.ones_like(prepared_data[...,0])
        # tprint("\n" + corr_name, "orig remaining", remaining.shape, "prepared_data", prepared_data.shape, "hook_x", hook_x.shape)
        num_dims = remaining.numel()
    
        timeout = 5
        while torch.count_nonzero(remaining) and timeout > 0:

            # pluck out the first example from the remaining (uncorrelated) vectors
            example = prepared_data[remaining.bool()].flatten(0,-2)[0]

            # perform the particular correlation determination
            correlated, is_correlated = Corr_Type.get_correlated_vectors(example, prepared_data)
            # print("correlated", list(correlated.shape))
            
            for a in range(is_correlated.ndim):
                # tprint("a", a, "correlated", correlated.shape, "is_correlated", is_correlated.shape, "prepared_data", prepared_data.shape)
                # sum up all axes other than the current axis (a) 
                dimsum = is_correlated.sum(tuple([i for i in range(is_correlated.ndim) if i != a]))
                
                # we are only pulling out full dims here, meaning every other axis is represented in full
                full_axes = dimsum == num_dims//prepared_data.size(a)

                # if a whole dimension is represented by this single correlation, add it to the dict
                if torch.count_nonzero(full_axes):
                    corr_key = Corr_Type.get_key(example, correlated)
                    
                    # indices = [torch.arange(correlated.size(i), device=hook_x.device) if i != a else torch.nonzero(full_axes).squeeze() for i in range(correlated.ndim)]
                    indices = [slice(None) if i != a else torch.nonzero(full_axes).squeeze() for i in range(is_correlated.ndim)]
                    if corr_name not in statistics["correlations"]: statistics["correlations"][corr_name] = {}
                    statistics["correlations"][corr_name][corr_key] = indices
                    
                    # tprint(corr_name, "corr_key", corr_key, "indiced", indices)

                    # this should? replace the line below "remaining[correlated == 1.0] = 0.0"
                    # remaining[indices] = 0.0#, torch.tensor(0.0, dtype=remaining.dtype, device=remaining.device))

            remaining[is_correlated == 1.0] = 0.0
            timeout = timeout - 1
            # tprint("remaining", torch.count_nonzero(remaining))

        # tprint(corr_name, torch.count_nonzero(remaining), "/", num_dims, "remaining", num_dims - torch.count_nonzero(remaining), "/", num_dims, "correlated")
    # print("printing statistics", statistics)
    return statistics

def print_correlation_summary(statistics):
    name, shape, correlations = statistics["name"], statistics["shape"], statistics["correlations"]
    print(name, "statistical correlations")
    
    ndim = len(shape)
    # print("ndim", ndim, "shape", shape)
    for corr_name in correlations:
        # print("corr_name", corr_name)
        if corr_name == "shape": continue
        corr = correlations[corr_name]
        axis_sums = [0] * (ndim - 1)
        for corr_key, indices in corr.items():
            # print("len(axis_sums)", len(axis_sums), "type(indices)", type(indices), "len(indices)", len(indices))
            for i in range(ndim - 1):
                axis_sums[i] += shape[i] if indices[i] == slice(None) else indices[i].numel()
        found_full_axes = []
        for i in range(ndim - 1):
            if axis_sums[i] >= shape[i]:
                found_full_axes.append(i)
        if len(found_full_axes): print(corr_name + ":" + " " * (12 - len(corr_name)), "axes:", found_full_axes, "are correlated with", len(corr), "variation" + ("s" if len(corr) > 1 else ""))
        
        

def synthesize_hook(freqs, magnitude, phase, dc, shape, statistics):
    wk = freqs * 2 * torch.pi / p

    # wk = wk.expand_as(phase)
    # polarity = torch.where(torch.sign(second_order_stats["q"][...,0:1,0:1,1]) == torch.sign(second_order_stats["q"][0:1,0:1,0:1,0:1,0:1,1]), 1.0, -1.0)
    # printvars(magnitude, phase, polarity)
    # amplitude = polarity * magnitude
    # p_indices = torch.arange(p).to(device)[None,None,None,...,None]
    # printvars(dc, magnitude, phase, amplitude, polarity, p_indices, wk)
    # def d_value(x): return tstr(x[0,1,0,0])
    # # tprint("mag", d_value(magnitude))
    # # tprint("amp", d_value(amplitude))
    # # tprint("polarity", d_value(polarity))
    # # tprint("wk", d_value(wk))
    # # tprint("phase", d_value(phase))
    # # tprint("dc", d_value(dc))
    # # def flip_polarity(x): return torch.fft.ifft(torch.conj(torch.fft.fft(x))).real
    # def flip_polarity_keep_dc(x):
    #     X = torch.fft.fft(x)
    #     X[...,1:] = -X[...,1:]
    #     return torch.fft.ifft(X).real

    # base_phase = phase[:,0:1,0:1,0:1,:]
    # r_wave = (amplitude * torch.cos(p_indices * wk + base_phase)).sum(-1)
    # tprint("reconstructed_wave", r_wave.shape, "dc", dc.shape)
    # r_wave[1,...] = -r_wave[1,...]
    # r_wave = dc[...,0] + r_wave

def get_mse(x,y): return ((x - y).square().sum() / y.numel()).sqrt().item()
def get_mse_and_worst_vector(x, hook):
    worst_idx = None
    worst_sum = 0.0
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            for k in range(x.size(2)):
                new_sum = get_mse(x[i][j][k], hook[i][j][k])
                if new_sum > worst_sum:
                    worst_sum = new_sum
                    worst_idx = [i,j,k]
    return get_mse(x, hook), worst_sum, worst_idx

def divide_hook_symmetrically(hook_x, *, normalize=True):
    if hook_x.size(0) == n_heads and hook_x.size(1) == 3 and hook_x.size(2) == 3: 
        hook_x = torch.moveaxis(hook_x[:,-1], 0, 1)
        tprint("seems like you are looking at the attention scores/pattern, i probably fucked this view up, don't trust it", hook_x.shape)
    hook_a, hook_b = hook_x[0], hook_x[1]
    if normalize: hook_a, hook_b = hook_a - hook_a.mean(-1, True), hook_b - hook_b.mean(-1, True)
    hook_a_greater_than_b, hook_b_greater_than_a = torch.zeros_like(hook_a), torch.zeros_like(hook_a)
    for c in range(p):
        syms = [c//2+1, c//2+p//2+1]
        hook_a_greater_than_b[...,c,:syms[0]+1] = hook_b[...,c,:syms[0]+1]
        hook_a_greater_than_b[...,c,syms[0]:syms[1]] = hook_a[...,c,syms[0]:syms[1]]
        hook_a_greater_than_b[...,c,syms[1]:] = hook_b[...,c,syms[1]:]
        hook_b_greater_than_a[...,c,:syms[0]+1] = hook_a[...,c,:syms[0]+1]
        hook_b_greater_than_a[...,c,syms[0]:syms[1]] = hook_b[...,c,syms[0]:syms[1]]
        hook_b_greater_than_a[...,c,syms[1]:] = hook_a[...,c,syms[1]:]
    return torch.stack((hook_a, hook_b, hook_a_greater_than_b, hook_b_greater_than_a))

def get_freq_indices(x, k):
    x = inputs_last(x)
    all_mags = torch.fft.fft(x).abs()[...,:p//2+1]
    all_mags[...,0] = 0
    while all_mags.ndim > 2: all_mags = all_mags.sum(-2)
    _, freqs = all_mags.sort(descending=True)
    key_set, freq_dict = {}, {}
    best_freqs = []
    
    for d in range(x.shape[0]):
        for i in range(p//2+1):
            freq = freqs[d][i].item()
            if freq in key_freqs:
                best_freqs.append(freq)
                if freq not in key_set.keys():
                    # print(d, "i", i, "creating dict item for freq", freq)
                    key_set[freq] = all_mags[d].detach().clone()
                else:
                    # print(d, "i", i, "adding to", freq)
                    key_set[freq] += all_mags[d]
                break
    for freq, mags in key_set.items():
        topmags, topfreqs = mags.topk(k)
        for i, kf in enumerate(key_freqs):
            # print("kf", kf, "topfreqs", topfreqs.tolist())
            the_freq = topfreqs[i].item()
            if the_freq != kf and kf in topfreqs:
                topfreqs[i] = kf
                for j in range(1,len(key_freqs)):
                    if topfreqs[(i + j) % len(key_freqs)].item() == kf:
                        topfreqs[(i + j) % len(key_freqs)] = the_freq
                        
            for j in range(len(key_freqs)):
                other_freq = topfreqs[j].item()
                if other_freq != kf:
                    topfreqs[j]
        print("freq", freq, "set", topfreqs.tolist(), "mags", topmags.tolist())
        freq_dict[freq] = topfreqs
    idx_size = list(x.shape[:-2]) + [1,k]
    idx_size = [int(i) for i in idx_size]
    idx_size = tuple(idx_size)
    freq_idx = torch.zeros(idx_size, dtype=torch.long).to(x.device)
    for d in range(x.shape[0]):
        freq_idx[d] = freq_dict[best_freqs[d]]
    
    return freq_idx, best_freqs, freq_dict

def get_phasors(activation, freqs, *, dims=None):
    """Gets phasors for each frequency in freqs for each dim in dims.

    Given a + b % p = c:
    These phasors are to be multiplied with a cosine wave centered at c/2

    args:
    activation Important: expected to be in row c, col a (c_idx) order with shape [...,p,p])

    keyword args:
    freqs: collection of frequencies for which to make phasors, if None, the top sets of frequencies of len(key_freqs) are found
    dims: collection of dimensions for which to make phasors
    """
    act = activation

    if act.shape[-2] != p or act.shape[-1] != p:
        print("get_phasors activation argument is expected to be in row c, col a (c_idx) order with shape [...,p,p]")
        return None
    act = act[dims] if dims is not None else act
    mean = act.mean(-1, True)
    act = act - mean

    freqs = freqs if isinstance(freqs, torch.Tensor) else torch.tensor(freqs, dtype=torch.long, device=act.device)

    act2 = interpDFT(act, p*2, -1)
    signal = torch.zeros_like(act2[...,0,:])
    for i in range(p*2):
        signal[...,i] = act2[...,i%p,i]

    spectrum = torch.fft.fft(signal)
    phasors = torch.empty(0).to(act.device)
    
    # this can probably go, it was from a remnant from a previous scheme
    if freqs.ndim > 1:
        mag = 2 * torch.abs(spectrum.gather(-1, freqs.squeeze()))/(p*2)
        phase = torch.angle(spectrum.gather(-1, freqs.squeeze()))
    else:
        mag = 2 * torch.abs(torch.index_select(spectrum, -1, freqs.squeeze()))/(p*2)
        phase = torch.angle(spectrum.index_select(-1, freqs.squeeze()))
    polar = torch.polar(mag, phase)
    phasors = polar.unsqueeze(-2)#torch.cat((phasors, polar.unsqueeze(-1)), -1)

    return mean, phasors

def reconstruct_with_phasors(act, freqs):
    freqs = freqs if isinstance(freqs, torch.Tensor) else torch.tensor(freqs, dtype=torch.long, device=act.device)
    inputs_are_first = act.shape[0] == p*p
    act = inputs_last(act)
    mean, phasors = get_phasors(act, freqs)
    mags, phases = torch.abs(phasors), torch.angle(phasors)
    output = (torch.zeros_like(act) + mean).unsqueeze(0)

    for f in range(freqs.shape[-1]):
        output = torch.cat((output, torch.zeros_like(output[0:1])), 0)
        freq = freqs[...,f:f+1]
        cos_waves = torch.cos((-torch.arange(p)[...,None]/2 + torch.arange(p)[None,...]).to(act.device) * freq * 2 * torch.pi / p).to(act.device)
        phasor = mags[...,f:f+1] * torch.cos(phases[...,f:f+1] + torch.arange(p)[...,None].to(act.device) * (freq/2) * 2 * torch.pi / p)
        output[-1] = cos_waves * phasor + mean
        output[0] += cos_waves * phasor
    if inputs_are_first: output = inputs_first(output)
    return output

def make_custom_wave(device):
    rv = torch.cat((
            torch.tensor([0.6,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3]),torch.tensor([0.5] * 23),-torch.rand([10]),
            torch.tensor([1.0,0.9,0.8, 0.7,0.6,0.5,0.4,0.3,0.2,0.1]),
            torch.rand([20]), torch.tensor([0.5] * 25), -torch.rand([10]), torch.tensor([-0.5] * 5)), 0).to(device)
    # tprint("custom wave", rv.shape)
    return rv
    
def get_reversed_shifted_waves(wave):
    wave_length, wave_range = wave.size(0), wave.max() - wave.min()
    waves = wave.expand(3, wave_length, wave_length).clone()
    waves[1,...,0] = waves[0,...,0].clone()
    waves[1,...,1:] = torch.flip(waves[0,...,1:], [-1])
    for c in range(wave_length): waves[1][c] = torch.roll(waves[1][c], c, -1)
    waves[2] = waves[0] + waves[1]
    # tprint("reversing", "input wave", wave.shape, "waves before splitting into list of 3", waves.shape)
    return [waves[2] - wave_range * 1.5, waves[0] + wave_range, waves[1]]