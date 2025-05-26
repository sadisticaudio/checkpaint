import torch
import numpy as np

p=113

############ Helper Functions ###############

def unflatten_first(tensor):
    if tensor.shape[0]==p*p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else: 
        return tensor
def inputs_last(x):
    if x.size(0) == p*p:
        x = x.unflatten(0,(p,p))
    if x.size(0) == p and x.size(1) == p:
        x = torch.moveaxis(x, -1, 0)
        x = torch.moveaxis(x, (1, 2), (-2, -1))
    return x
def inputs_first(x):
    x = x.flatten(-2,-1)
    x = torch.moveaxis(x,-1,0)
    return x
def all_inputs_last(cache):
    for name, tensor in cache.items():
        cache[name] = inputs_last(tensor)
    return cache
def cos(x, y):
    return (x.dot(y))/x.norm()/y.norm()
def mod_div(a, b):
    return (a*pow(b, p-2, p))%p
def normalize(tensor, axes=0):
    if not isinstance(axes, (tuple, list)):
        axes = [axes]
    for axis in axes:
        tensor = tensor/(tensor).pow(2).sum(dim=axis, keepdim=True).sqrt()
    return tensor
def center(tensor, axes=-1):
    if not isinstance(axes, tuple):
        axes = (axes,)
    return tensor - (tensor).mean(dim=axes, keepdim=True)
def get_freq_idx(freq):
    index_1d = [0, 2*freq-1, 2*freq]
    return [[i]*3 for i in index_1d], [index_1d]*3
def extract_freq_2d(tensor, freq):
    # Takes in a pxpx... or batch x ... tensor, returns a 3x3x... tensor of the 
    # Linear and quadratic terms of frequency freq
    tensor = unflatten_first(tensor)
    # Extracts the linear and quadratic terms corresponding to frequency freq
    index_1d = [0, 2*freq-1, 2*freq]
    # Some dumb manipulation to use fancy array indexing rules
    # Gets the rows and columns in index_1d
    print("index_1d", [[i]*3 for i in index_1d], [index_1d]*3)
    rv = tensor[[[i]*3 for i in index_1d], [index_1d]*3]
    # if freq == 1: 
    #     print("extract_freq_2d index_1d", index_1d, "tensor", tensor.shape, "index", [[i]*3 for i in index_1d], [index_1d]*3)
    #     print("rv", rv.shape)
    return rv
def get_cov(tensor, norm=True):
    # Calculate covariance matrix
    if norm:
        tensor = normalize(tensor, axis=1)
    return tensor @ tensor.T
  
def get_fourier_basis(N, device='cpu'):
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
  # print("N", N, "basis norm", basis.norm(dim=-1)[:5].tolist())
  basis = basis/basis.norm(dim=-1, keepdim=True).to(device)
  
  return basis.to(device), basis_names

fourier_basis, fourier_basis_names = get_fourier_basis(p)
inv_fourier_basis = fourier_basis.clone()
inv_fourier_basis[1::2,1:] *= -1

############ Fourier Functions ###############

def from_torch_spectrum(x, dim=-1):
    N = x.size(dim)
    x = x/np.sqrt(N/2)
    # flipped = torch.complex(-x.imag, x.real)
    # # print("flipped", flipped.shape, flipped)
    # flipped_real = torch.view_as_real(flipped)
    # # print("flipped_real", flipped_real.shape, flipped_real)
    # flat = flipped_real.flatten(-2,-1)
    # # print("flat", flat.shape, flat)
    # trimmed = flat[1:N+1]
    # # print("trimmed", trimmed.shape, trimmed)
    
    tx = torch.view_as_real(torch.complex(-x.imag, x.real)).flatten(-2,-1)[1:N+1]
    tx[...,0] *= (np.sqrt(2)/2)
    # print("final", tx.shape, tx)
    return tx

def from_torch_2d_spectrum(X):
    M, N = X.shape[-2], X.shape[-1]
    TX = torch.view_as_real(torch.complex(-X.imag, X.real)).flatten(-2,-1)[...,1:M+1,1:N+1]
    X = X/(np.sqrt(M/2) * np.sqrt(N/2))
    TX[...,0,:] *= (np.sqrt(2)/2) 
    TX[...,:,0] *= (np.sqrt(2)/2) 
    return TX

def fft1d(x, inverse=False):
    # basis = inv_fourier_basis if inverse else fourier_basis
    # if x.size(-1) != p: basis, _ = get_fourier_basis(x.size(-1))
    # shape = x.shape
    # for i in range(x.ndim):
    #     if x.size(i) == p*p:
    #         x = x.unflatten(i, (p,p))
    # return (x @ basis.T).reshape(shape)
    return from_torch_spectrum(torch.fft.fft(x))

def real_fourier_2d(mat):
    shape = mat.shape
    M, N = mat.shape[-2], mat.shape[-1]
    MAT = torch.zeros_like(mat, dtype=torch.cfloat)
    for u in range(M):
        for v in range(N):
            MAT[...,u,v] += torch.exp(2j * np.pi * (u * np.arange(M) / M + v * np.arange(N) / N))

def fft2d2(x, inverse=False):
    return from_torch_2d_spectrum(torch.fft.fft2(x))
    

def fourier_2d_basis_term(x_index, y_index):
    # Returns the 2D Fourier basis term corresponding to the outer product of 
    # the x_index th component in the x direction and y_index th component in the 
    # y direction
    # Returns a 1D vector of length p^2
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()

def fft2d(mat, inverse=False):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    basis = inv_fourier_basis if inverse and mat.shape[-1] == p else fourier_basis if mat.shape[-1] == p else get_fourier_basis(mat.shape[-1])#fourier_basis
    shape = mat.shape
    for i in reversed(range(mat.ndim)):
        if mat.size(i) == p*p:
            mat = mat.unflatten(i, (p,p))
    # print("basis.shape", basis[0].shape, "mat.shape", mat.shape)
    mat = (basis @ mat @ basis.T).reshape(shape)
    # mat = torch.flip(mat, dims=(-2,-1))
    # mat = mat * (p*p)
    return mat

def analyse_fourier_2d(tensor, top_k=10):
    # Processes a (p,p) or (p*p) tensor in the 2D Fourier Basis, showing the 
    # top_k terms and how large a fraction of the variance they explain
    values, indices = tensor.flatten().pow(2).sort(descending=True)
    rows = []
    total = values.sum().item()
    for i in range(top_k):
        rows.append([tensor.flatten()[indices[i]].item(),
                     values[i].item()/total, 
                     values[:i+1].sum().item()/total, 
                     fourier_basis_names[indices[i].item()//p], 
                     fourier_basis_names[indices[i]%p]])
    # display(pd.DataFrame(rows, columns=['Coefficient', 'Frac explained', 'Cumulative frac explained', 'x', 'y']))

def get_2d_fourier_component(tensor, x, y):
    # Takes in a batch x ... tensor and projects it onto the 2D Fourier Component 
    # (x, y)
    vec = fourier_2d_basis_term(x, y).flatten()
    ret = vec[:, None] @ (vec[None, :] @ tensor)
    print("2D foufier comp for", tensor.shape, "xy", x, y, "vec", vec.shape, "ret", ret.shape)
    return ret

def get_freq_spectral_sum(spec, x, y):
    # Takes input channels last
    if spec.ndim < 3: spec = spec.unsqueeze(0)
    a = spec[...,2*x - 1:2 * x + 1,:].abs().sum()
    b = spec[...,:,2*y - 1:2 * y + 1].abs().sum()
    ret = (a + b).item()
    return ret

def get_component_cos_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to cos(freq*(x+y)) in the 2D Fourier basis
    # This is equivalent to the matrix cos((x+y)*freq*2pi/p)
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq).flatten()
    # Divide by sqrt(2) to ensure it remains normalised
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction)/np.sqrt(2)
    # Collapse_dim says whether to project back into R^(p*p) space or not
    if collapse_dim:
        return (cos_xpy_direction @ tensor)
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)

def get_component_sin_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to sin((x+y)*freq*2pi/p) in the 2D Fourier basis
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (sin_xpy_direction @ tensor)
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)