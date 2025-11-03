import torch
import numpy as np
from pylinalg import vec_transform, vec_unproject
from checkpaint.utils import *
import numbers
from pythreejs import *
from matplotlib import cm

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
  # tprint("y", y.shape, "num (data_points)", num, "repeat_idx", repeat_idx, "orig_num", orig_num)
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
  
  def convert_to_geometry(y: torch.Tensor, num, orig_num, repeat_idx):
    y = y[...,None]
    y = y.repeat(tuple(repeat_idx))
    new_y_shape = y.shape[:-2] + (num * 2,)
    y = y.reshape(new_y_shape)
    y = y[...,1:-1]
    new_y_shape = y.shape[:-1] + (y.shape[-1]//2,2)
    y = y.reshape(new_y_shape).contiguous()
    x = torch.arange(num, dtype=torch.float, device=y.device)[...,None].repeat(1,2).flatten()[1:-1].reshape(-1,2).expand_as(y)
    x = x[...,None].cpu().numpy()# * (orig_num/num)
    y = y[...,None].cpu().numpy()
    return np.concatenate((x, y, np.full_like(x, 0.01)), axis=-1).copy()
  
  def split_and_prepare_data(state, all_data: torch.Tensor):
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
      
  def update_renderables_fast(state, gui_data: np.array):
    rendereables = state.rendereables

    for v, child in enumerate(rendereables.children):
      child.geometry.positions = gui_data[v]
      child.geometry.needsUpdate = True

from pythreejs import (Group, PlaneGeometry, Mesh, MeshBasicMaterial, DataTexture)
  
class Heatmap(GuiStyle):
  name = "heatmap"
  n_axes = 2
  n_gui_axes = 2  # same idea as before: two GUI axes
  
  @classmethod
  def get_gui_ranges(cls, state):
    ranges = [[0, state.shape[a]] for a in state.axes]
    # print("ranges", ranges, "axes", state.axes, "shape", state.shape)
    # for i in range(cls.n_gui_axes - cls.n_axes): ranges.append([state.get_min(), state.get_max()])
    return ranges
  
  @classmethod
  def get_gui_lengths(cls, state): return [r[1] - r[0] for r in cls.get_gui_ranges(state)]

  # ---------- helpers ----------
  @staticmethod
  def _to_rgba_uint8(x2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
      """
      Map a 2D float array to RGBA uint8 (grayscale by default).
      x2d: (H, W) float32/float64
      Returns (H, W, 4) uint8 with alpha=255.
      """
      if not np.isfinite(vmin): vmin = np.nanmin(x2d)
      if not np.isfinite(vmax): vmax = np.nanmax(x2d)
      if vmax <= vmin:
          vmax = vmin + (1e-12 if vmin == 0 else abs(vmin) * 1e-12)

      # normalize to [0,1]
      y = (x2d - vmin) / (vmax - vmin)
      y = np.clip(y + 1e-12, 0.0, 1.0)
      
      rgba = (cm.viridis(y / (y.max())) * 255).astype(np.uint8)

      # # grayscale -> RGBA
      # g = (y * 255.0).astype(np.uint8)
      # rgba = np.empty((*g.shape, 4), dtype=np.uint8)
      # rgba[..., 0] = g
      # rgba[..., 1] = np.zeros_like(g)
      # rgba[..., 2] = g
      # rgba[..., 3] = 255
      return rgba

  @staticmethod
  def _make_texture(rgba: np.ndarray) -> DataTexture:
      """
      rgba: (H, W, 4) uint8
      Returns a pythreejs DataTexture with sensible sampling for crisp heatmaps.
      """
      H, W, _ = rgba.shape
      # pythreejs expects a flat array
      
      data = rgba.astype(np.uint8)#.ravel(order="C")
      # tprint("rgba", rgba.shape, "data", data.shape)
      tex = DataTexture(
          data=data,
          width=W,
          height=H,
          # format=RGBAFormat,
          # type=UnsignedByteType,
          # magFilter=NearestFilter,
          # minFilter=NearestFilter,
          # wrapS=ClampToEdgeWrapping,
          # wrapT=ClampToEdgeWrapping,
          flipY=True,  # put (0,0) at bottom-left in screen coords
      )
      tex_w = getattr(tex, "_width", None)
      tex_h = getattr(tex, "_height", None)
      # tprint("tex_w", tex_w, "tex_h", tex_h, "w/h", W,H)
      tex.needsUpdate = True
      return tex

  # ---------- API mirroring your Lines class ----------
  @classmethod
  
  def create_rendereables(self, state):
      """
      Build one heatmap per leading slice (num_tensors),
      using the last two axes as (H, W).
      """
      num_tensors, shape = state.num_tensors, state.shape
      h_axis = state.axes[-2]
      w_axis = state.axes[-1]
      H = shape[h_axis]
      W = shape[w_axis]

      # geometry sized to data grid; you can scale to your world units here
      geom = PlaneGeometry(W, H)

      rendereables = Group()
      # placeholder single-pixel textures until first update; keeps structure consistent
      placeholder = self._make_texture(np.zeros((W, H, 4), dtype=np.uint8))
      for v in range(num_tensors):
          mat = MeshBasicMaterial(map=placeholder)
          mesh = Mesh(geometry=geom, material=mat)
          # center it like your grid: (0..W-1, 0..H-1) centered at origin
          mesh.position = [W * 0.5 - 0.5, H * 0.5 - 0.5, 0.01 * v]  # slight z offset per slice
          rendereables.add(mesh)
      return rendereables

  @staticmethod
  def convert_to_geometry(y: torch.Tensor, vmin: float, vmax: float):
      """
      Convert a tensor shaped (..., H, W) into a list/stack of RGBA uint8 arrays,
      one per leading slice, for DataTexture consumption.
      Returns: np.ndarray of shape (num_tensors, H, W, 4), dtype=uint8.
      """
      assert y.ndim >= 2, "Heatmap expects at least 2D input on the last two axes."
      H, W = y.shape[-2], y.shape[-1]
      leading = int(np.prod(y.shape[:-2]))  # flatten all leading dims to num_tensors

      # Move to CPU and float64/32 for stable normalization
      y_np = y.detach().to("cpu", torch.float32).reshape(leading, H, W).numpy()

      # per-CLASS (global) normalization using provided vmin/vmax
      out = np.empty((leading, H, W, 4), dtype=np.uint8)
      for i in range(leading):
          out[i] = Heatmap._to_rgba_uint8(y_np[i], vmin, vmax)
      out = out.reshape(y.shape + (4,))
      return out

  @staticmethod
  def split_and_prepare_data(state, all_data: torch.Tensor):
      """
      Similar contract to your Lines.split_and_prepare_data, but for 2D heatmaps:
        - "spacetime": raw data on last 2 axes
        - "fourier"  : |FFT2| (rfft2 magnitude) of same
      Stores min/max for each view in state.min/max as before.
      Returns a dict of {key: np.uint8 RGBA arrays} ready for textures.
      """
      # Reorder so the last two axes are the GUI axes (state should already do this)
      time_data = state.get_permuted_data(all_data)  # (..., H, W)
      time_data = torch.flip(time_data, [-2])
      # tprint("1 time_data", time_data.shape, "min", time_data.amin(), "max", time_data.amax())
      time_data -= torch.amin(time_data, (-2,-1), True) - 1e-8
      # tprint("1.5 time_data", time_data.shape, "min", time_data.amin(), "max", time_data.amax())
      time_data /= (torch.amax(time_data, (-2,-1), True) + 1e-8)
      # tprint("1.6 time_data", time_data.shape, "min", time_data.amin(), "max", time_data.amax())
      time_data *= time_data.size(-1)
      # time_data += 1e-1
      # tprint("2 time_data", time_data.shape, "min", time_data.amin(), "max", time_data.amax())

      # FFT2 magnitude for a nicer display (real-input assumed; rfft2)
      # If you want full symmetry, use fft2 instead; here we use log1p(abs(...)) for contrast.
      # Note: keep things real for a heatmap.
      freq_c = torch.fft.rfft2(time_data)  # (..., H, W//2+1)
      freq_data = torch.log1p(freq_c.abs())
      # tprint("1 freq_data", freq_data.shape)
      reals = torch.view_as_real(freq_c).flatten(-2,-1)
      first_idx = (0 if time_data.shape[-1] % 2 == 0 else 1)
      last_idx = (-1 if time_data.shape[-1] % 2 == 0 else 0)
      freq_data = torch.log1p(reals[...,first_idx:].abs())
      # tprint("2 freq_data", freq_data.shape, "reals", reals.shape, "time_data.shape[-1] % 2 == 0", time_data.shape[-1] % 2 == 0, "first_idx", first_idx, "last_idx", last_idx)

      # mins/maxes for consistent color scaling across slices
      state.min["spacetime"] = time_data.min().item()
      state.max["spacetime"] = time_data.max().item()
      state.min["fourier"] = freq_data.min().item()
      state.max["fourier"] = freq_data.max().item()

      # Prepare RGBA arrays
      prepared = {}
      prepared["spacetime"] = Heatmap.convert_to_geometry(
          time_data, state.min["spacetime"], state.max["spacetime"]
      )
      prepared["fourier"] = Heatmap.convert_to_geometry(
          freq_data, state.min["fourier"], state.max["fourier"]
      )
      return prepared

  @staticmethod
  def update_renderables_fast(state, gui_data: np.ndarray):
      """
      gui_data: np.ndarray of shape (num_tensors, H, W, 4), dtype=uint8
      Updates textures in-place for speed.
      """
      rendereables = state.rendereables
      # Update each mesh's texture data
      # tprint("rendereables info")
      # print_object_info(rendereables)
      # tprint("rendereables.children info")
      # print_object_info(rendereables.children)
      # tprint("rendereables.children[0] info")
      # print_object_info(rendereables.children[0])
      for v, child in enumerate(rendereables.children):
          rgba = gui_data[v]
          H, W, _ = rgba.shape
          # (Re)build texture if size changed, else update data buffer
          mat = child.material
          tex = mat.map
          size_changed = False#(tex.width != W) or (tex.height != H)
          

          if size_changed:
              # Replace the entire texture
              new_tex = Heatmap._make_texture(rgba)
              mat.map = new_tex
              mat.needsUpdate = True
          else:
              # Update the in-place data buffer
              flat = rgba#.ravel(order="C")
              # tprint("rgba", rgba.shape, "flat", flat.shape)
              # pythreejs DataTexture stores a typed array; assign and flag updates
              tex.data = flat
              tex.needsUpdate = True
          tex_w = getattr(tex, "_width", None)
          tex_h = getattr(tex, "_height", None)
          # tprint("tex_w", tex_w, "tex_h", tex_h, "w/h", W,H)
          # For a BufferGeometry, the size is pre-computed, but it's good practice to ensure it's up to date
          if hasattr(child, 'geometry') and hasattr(child.geometry, 'computeBoundingBox'):
              child.geometry.computeBoundingBox()
              
              # Check if boundingBox is not None
              if child.geometry.boundingBox:
                  size = child.geometry.boundingBox.size
                  print(f"Child {i} (type: {child.__class__.__name__}) has size: {size}")