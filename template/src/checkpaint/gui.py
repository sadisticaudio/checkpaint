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
  
def convert_to_geometry(x, num, repeat_idx):
    y = x[...,None]
    # tprint("y", y.shape)
    y = y.repeat(tuple(repeat_idx))
    # tprint("y", y.shape)
    new_y_shape = y.shape[:-2] + (num * 2,)
    y = y.reshape(new_y_shape)
    # tprint("y", y.shape)
    y = y[...,1:-1]
    # tprint("y", y.shape)
    new_y_shape = y.shape[:-1] + (y.shape[-1]//2,2)
    y = y.reshape(new_y_shape)
    # tprint("y", y.shape)
    x = torch.arange(num, dtype=torch.float, device=x.device)[...,None].repeat(1,2).flatten()[1:-1].reshape(-1,2).expand_as(y)
    x = x[...,None].cpu().numpy()
    y = y[...,None].cpu().numpy()
    # print("x", x.shape, "y", y.shape)
    return np.concatenate((x, y, np.full_like(x, 0.01)), axis=-1)

class GuiStyle():
  name = "gui_name"
  n_axes = 2
  n_gui_axes = 2
  orientation = "xy"

  def get_min(tensors):
    the_min = tensors[0][torch.isfinite(tensors[0])].min()
    for i in range(1,len(tensors)): the_min = torch.minimum(the_min, tensors[i][torch.isfinite(tensors[i])].min())
    return the_min.cpu().item()

  def get_max(tensors):
    the_max = tensors[0][torch.isfinite(tensors[0])].max()
    for i in range(1,len(tensors)): the_max = torch.maximum(the_max, tensors[i][torch.isfinite(tensors[i])].max())
    return the_max.cpu().item()

  @classmethod
  def get_gui_ranges(cls, state):
    ranges = [[0, state.shape[a]] for a in state.axes]
    # print("ranges", ranges, "axes", state.axes, "shape", state.shape)
    for i in range(cls.n_gui_axes - cls.n_axes): ranges.append([state.min, state.max])
    return ranges
  @classmethod
  def get_gui_lengths(cls, state): return [r[1] - r[0] for r in cls.get_gui_ranges(state)]
  
class Lines(GuiStyle):
  name = "lines"
  n_axes = 1
  n_gui_axes = 2
  @classmethod
  def setup_camera(cls, state, camera, canvas):
    ranges, lengths = cls.get_gui_ranges(state), cls.get_gui_lengths(state)
    return None
  
  def create_rendereables(state):
    num_tensors, shape, colors = state.num_tensors, state.shape, state.colors
    data_points = shape[state.axes[0]]
    rendereables = Group()
    x = np.repeat(np.linspace(0, data_points, data_points, endpoint=False, dtype=np.float32), 2)[1:-1].reshape(-1,2)
    for v in range(num_tensors):
      positions = np.concatenate((x[...,None], np.zeros_like(x)[...,None], np.full_like(x,0.01)[...,None]), axis=-1)
      rendereables.add(LineSegments2(LineSegmentsGeometry(positions=positions), LineMaterial(linewidth=3, color=colors[v])))
    return rendereables
  
  def transform_data(state, all_data):
    x = state.get_permuted_data(all_data)
    if state["gui_type"] == "fourier": x = state.fft1d(x)
    elif state["gui_type"] == "radial": x = torch.angle(torch.fft.fft(x))
    return x.cpu().numpy()
  
  def split_and_prepare_data(state, all_data):
    all_gui_data = state.get_permuted_data(all_data)
    all_gui_data_spec = state.fft1d(all_gui_data)
    # else: all_gui_data = all_gui_data.cpu().numpy()
    data_points = all_gui_data.shape[-1]
    repeat_idx = [1] * (all_gui_data.ndim) + [2]
    prepared = {}
    prepared["spacetime"] = convert_to_geometry(all_gui_data, data_points, repeat_idx)
    prepared["fourier"] = convert_to_geometry(all_gui_data, data_points, repeat_idx)
    return prepared
  def update_renderables(state, gui_data, sphere, *, get_points=None, get_message=None):
    data_points = gui_data.shape[-1]
    x = np.linspace(0, data_points, data_points, endpoint=False, dtype=gui_data.dtype)
    rendereables = state.rendereables

    for v, child in enumerate(rendereables.children):
      new_x, new_y = np.repeat(x, 2)[1:-1].reshape(-1,2), np.repeat(gui_data[v], 2)[1:-1].reshape(-1,2)
      new_positions = np.concatenate((new_x[...,None], new_y[...,None], np.full_like(new_x, 0.01)[...,None]), axis=-1)
      child.geometry.positions = new_positions
      child.geometry.needsUpdate = True
      
  def update_renderables_fast(state, gui_data):
    rendereables = state.rendereables

    for v, child in enumerate(rendereables.children):
      child.geometry.positions = gui_data[v]
      child.geometry.needsUpdate = True

