import torch
import numpy as np
from pylinalg import vec_transform, vec_unproject
from checkpaint.utils import *
import numbers
from pythreejs import *

def get_fourier_basis(N):
    basis = []
    basis.append(torch.ones(N))
    for freq in range(1, N//2+1):
        basis.append(torch.sin(torch.arange(N)*2 * torch.pi * freq / N))
        basis.append(torch.cos(torch.arange(N)*2 * torch.pi * freq / N))
    basis = torch.stack(basis, dim=0)#.to(device)
    basis = basis/basis.norm(dim=-1, keepdim=True)#.to(device)
    return basis

get_tick = lambda length : pow(10, round(np.log10(min(1e+10, max(-1e+10, length))/10)))
def get_total_ticks_and_length(length, padding=0):
  tick = get_tick(length)
  amt_of_ticks = length / tick
  n_ticks = np.ceil(amt_of_ticks) + 1
  return  n_ticks, tick * n_ticks

identity_map = lambda x,start,end : str(x) if isinstance(x, numbers.Integral) else str(f"{x:.1e}")
fourier_map = lambda x,start,end : str(f"cos{int(x/2)}")

class ScaledMap:
  def __init__(self, map, scale_factor):
    self.map = map
    self.sf = scale_factor
  def __call__(self, x, start, end): return self.map(x * self.sf,start,end)
  
def convert_to_geometry(y, num, orig_num, repeat_idx):
  # tprint("y", y.shape, "num (data_points)", num, "repeat_idx", repeat_idx)
  y = y[...,None]
  # tprint("y", y.shape)
  y = y.repeat(tuple(repeat_idx))
  # tprint("y", y.shape)
  new_y_shape = y.shape[:-2] + (num * 2,)
  # tprint("resizing y", y.shape, "to new_y_shape", new_y_shape)
  y = y.reshape(new_y_shape)
  # tprint("y", y.shape)
  y = y[...,1:-1]
  # tprint("y", y.shape)
  new_y_shape = y.shape[:-1] + (y.shape[-1]//2,2)
  y = y.reshape(new_y_shape).contiguous()
  # tprint("y", y.shape)
  x = torch.arange(num, dtype=torch.float, device=y.device)[...,None].repeat(1,2).flatten()[1:-1].reshape(-1,2).expand_as(y)
  x = x[...,None].cpu().numpy()# * (orig_num/num)
  y = y[...,None].cpu().numpy()
  # print("x", x.shape, "y", y.shape)
  # tprint("x[...,:5]", x.reshape(-1,x.shape[-2],x.shape[-1])[:5])
  # tprint("y[...,:5]", y.reshape(-1,y.shape[-2],y.shape[-1])[:5])
  return np.concatenate((x, y, np.full_like(x, 0.01)), axis=-1).copy()

class GuiStyle():
  name = "gui_name"
  n_axes = 2
  n_gui_axes = 2
  orientation = "xy"

  @classmethod
  def get_gui_ranges(cls, state):
    ranges = [[0, state.shape[a]] for a in state.axes]
    # print("ranges", ranges, "axes", state.axes, "shape", state.shape)
    for i in range(cls.n_gui_axes - cls.n_axes): ranges.append([state.get_min(), state.get_max()])
    return ranges
  @classmethod
  def get_gui_lengths(cls, state): return [r[1] - r[0] for r in cls.get_gui_ranges(state)]
  
class Lines(GuiStyle):
  name = "lines"
  n_axes = 1
  n_gui_axes = 2
  @classmethod

  def create_rendereables(self, state):
    num_tensors, shape, colors = state.num_tensors, state.shape, state.colors
    data_points = shape[state.axes[0]]
    rendereables = Group()
    x = np.repeat(np.linspace(0, data_points, data_points, endpoint=False, dtype=np.float32), 2)[1:-1].reshape(-1,2)
    for v in range(num_tensors):
      positions = np.concatenate((x[...,None], np.zeros_like(x)[...,None], np.full_like(x,0.01)[...,None]), axis=-1)
      rendereables.add(LineSegments2(LineSegmentsGeometry(positions=positions), LineMaterial(linewidth=3, color=colors[v])))
    return rendereables
  
  def split_and_prepare_data(state, all_data):
    time_data = state.get_permuted_data(all_data)
    freq_data = state.fft1d(time_data).contiguous().clone()
    # tprint("all_data", all_data.shape, "time_data", time_data.shape, "freq_data", freq_data.shape)
    state.min["spacetime"] = time_data.min().item()
    state.max["spacetime"] = time_data.max().item()
    state.min["fourier"] = freq_data.min().item()
    state.max["fourier"] = freq_data.max().item()
    data_points = time_data.shape[-1]
    repeat_idx = [1] * (time_data.ndim) + [2]
    prepared = {}
    prepared["spacetime"] = convert_to_geometry(time_data, data_points, state.shape[-1], repeat_idx)
    prepared["fourier"] = convert_to_geometry(freq_data, data_points, state.shape[-1], repeat_idx)
    # tprint("prepared spacetime", prepared["spacetime"].shape, "fourier", prepared["fourier"].shape)
    return prepared
      
  def update_renderables_fast(state, gui_data):
    rendereables = state.rendereables

    for v, child in enumerate(rendereables.children):
      child.geometry.positions = gui_data[v]
      child.geometry.needsUpdate = True