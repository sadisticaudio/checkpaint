# import os
import torch
import numpy as np
# from wgpu.gui.jupyter import WgpuCanvas, run
import pygfx as gfx
from pylinalg import vec_transform, vec_transform_quat, vec_unproject, quat_from_euler, vec_normalize
# import ipywidgets as widgets
# from ipywidgets import Widget
# from IPython.display import display, Javascript
from checkpaint.utils import *
import numbers
# from dataclasses import dataclass
from pygfx.geometries._plane import generate_plane
from pygfx.utils import Color
# from pygfx.utils.text._fontmanager import font_manager

def get_fourier_basis(N):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    basis = []
    basis.append(torch.ones(N))
    for freq in range(1, N//2+1):
        basis.append(torch.sin(torch.arange(N)*2 * torch.pi * freq / N))
        basis.append(torch.cos(torch.arange(N)*2 * torch.pi * freq / N))
    basis = torch.stack(basis, dim=0)#.to(device)
    basis = basis/basis.norm(dim=-1, keepdim=True)#.to(device)
    return basis

get_tick = lambda length : pow(10, round(np.log10(min(1e+10, max(-1e+10, length))/10)))
identity_map = lambda x,start,end : str(x) if isinstance(x, numbers.Integral) else str(f"{x:.1e}")
fourier_map = lambda x,start,end : str(f"cos{int(x/2)}")

class ScaledMap:
  def __init__(self, map, scale_factor):
    self.map = map
    self.sf = scale_factor
  def __call__(self, x, start, end): return self.map(x * self.sf,start,end)
  

#### NICKED FROM generate_plane() IN pygfx/geometries/_plane.py ####
  
def generate_contour(data, style="triangles"):
  width, height = data.shape[-2], data.shape[-1]
  width_segments, height_segments = width - 1, height - 1
  nx, ny = width_segments + 1, height_segments + 1

  x = np.linspace(0, width, nx, dtype=np.float32)
  y = np.linspace(0, height, ny, dtype=np.float32)
  xx, yy = np.meshgrid(x, y)
  xx, yy = xx.flatten(), yy.flatten()
  positions = np.column_stack([xx, yy, data.transpose(-2,-1).flatten()])

  # tmin, tmax = data.min(), data.max()
  # texcoords = (positions[..., 2:3] - tmin) / (0000000.1 + tmax - tmin)
  # texcoords = np.concatenate((texcoords, texcoords), axis=-1)
  
  dim = np.array([width, height], dtype=np.float32)
  texcoords = (positions[..., :2] + dim / 2) / dim
  texcoords[..., 1] = 1 - texcoords[..., 1]

  # the amount of vertices
  indices = np.arange(ny * nx, dtype=np.uint32).reshape((ny, nx))
  # for every panel (height_segments, width_segments) there is a quad (2, 3)
  index = np.empty((height_segments, width_segments, 2, 3), dtype=np.uint32)
  
  if style == "triangles":
    index[:, :, 0, 2] = indices[np.arange(height_segments)[:, None], np.arange(width_segments)[None, :]]
    index[:, :, 0, 0] = index[:, :, 0, 2] + 1
    index[:, :, 0, 1] = index[:, :, 0, 2] + nx + 1
    index[:, :, 1, 0] = index[:, :, 0, 2]
    index[:, :, 1, 1] = index[:, :, 0, 2]
    index[:, :, 1, 2] = index[:, :, 0, 2]
  # elif style == 
  else:
    # create a grid of initial indices for the panels
    index[:, :, 0, 0] = indices[np.arange(height_segments)[:, None], np.arange(width_segments)[None, :]]
    # the remainder of the indices for every panel are relative
    index[:, :, 0, 1] = index[:, :, 0, 0] + 1
    index[:, :, 0, 2] = index[:, :, 0, 0] + nx
    index[:, :, 1, 0] = index[:, :, 0, 0] + nx + 1
    index[:, :, 1, 1] = index[:, :, 1, 0] - 1
    index[:, :, 1, 2] = index[:, :, 1, 0] - nx

  normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (ny * nx, 1))

  return positions, normals, texcoords, index.reshape((-1, 3))


def contour_geometry(data, triangles=False): 
  positions, normals, texcoords, indices = generate_contour(data, triangles)
  return gfx.Geometry(indices=indices,positions=positions,normals=normals,texcoords=texcoords,)

def colored_plane_geometry(width=1, height=1, color = (0,0,0,1)):
    pos, norm, tex, idx = generate_plane(width, height,1,1)
    geo = gfx.Geometry(indices=idx.reshape((-1, 3)),positions=pos,normals=norm,texcoords=tex)
    plane = gfx.Mesh(geo, gfx.MeshBasicMaterial(color=color))
    plane.local.x += width/2
    plane.local.y += height/2
    return plane

def map_screen_to_world(camera, pos, viewport_size):
    # first convert position to NDC
    if viewport_size[0] == 0: print("viewport_size[0] == 0, dividing by zero")
    if viewport_size[1] == 0: print("viewport_size[1] == 0, dividing by zero")
    x = pos[0] / viewport_size[0] * 2 - 1
    y = -(pos[1] / viewport_size[1] * 2 - 1)
    pos_ndc = (x, y, 0)
    pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)
    # unproject to world space
    pos_world = vec_unproject(pos_ndc[:2], camera.camera_matrix)
    
    return pos_world
  
class GuiStyle():
  name = "gui_name"
  n_axes = 2
  n_gui_axes = 2
  orientation = "xy"
  # def get_min(tensor): return tensor.min().cpu().item()
  # def get_max(tensor): return tensor.max().cpu().item()
  def get_min(tensors):
    the_min = tensors[0][torch.isfinite(tensors[0])].min()
    for i in range(1,len(tensors)): the_min = torch.minimum(the_min, tensors[i][torch.isfinite(tensors[i])].min())
    return the_min.cpu().item()
    # the_min = 1e+10
    # for tensor in tensors:
    #   finiteonly = torch.isfinite(tensor)
    #   the_min = min(the_min, tensor[finiteonly].min().cpu().item())
    # return the_min
  def get_max(tensors):
    the_max = tensors[0][torch.isfinite(tensors[0])].max()
    for i in range(1,len(tensors)): the_max = torch.maximum(the_max, tensors[i][torch.isfinite(tensors[i])].max())
    return the_max.cpu().item()
    the_max = 1e-10
    for tensor in tensors:
      finiteonly = torch.isfinite(tensor)
      the_max = min(the_max, tensor[finiteonly].min().cpu().item())
    return the_max
  @classmethod
  def get_gui_ranges(cls, state):
    ranges = [[0, state.shape[a]] for a in state.axes]
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
    # canvas.set_logical_size(lengths[0] + lengths[0] * 3 / 20, (lengths[0] + lengths[0] * 3 / 20)/2)
    # camera.set_view_size(lengths[0] + lengths[0] * 3 / 20, lengths[1])
    # camera.update_projection_matrix()
    # camera.show_rect(ranges[0][0] - lengths[0]/10, ranges[0][1] + lengths[0]/20, state.min, state.max)
    return None
  def transform_data(state, all_data):
    x = state.get_permuted_data(all_data)
    if state["gui_type"] == "fourier": x = state.fft1d(x)
    elif state["gui_type"] == "radial": x = torch.angle(torch.fft.fft(x))
    return x.cpu().numpy()
  def get_renderables(state, gui_data, sphere, *, get_points=None, get_message=None):
    the_key_freqs = [14]
    data_points = gui_data.shape[-1]
    x = np.linspace(0, data_points, data_points, endpoint=False, dtype=gui_data.dtype)
    colors = ["#faa", "#6a6", "#99e", "#774", "#747", "#477"]
    lines = []
    for v in range(gui_data.shape[0]):
      mag = (gui_data[v].max() - gui_data[v].min())/2
      dc = gui_data[v].min() + mag
      if state["gui_type"] == "fourier":
        cos_0 = torch.tensor(gui_data[v][...,:1])
        spec_0 = torch.complex(cos_0, torch.zeros_like(cos_0))
        spec = torch.complex(torch.tensor(gui_data[v][...,2:gui_data.shape[-1]:2]), -torch.tensor(gui_data[v][...,1:gui_data.shape[-1]:2]))
        spec = torch.cat((spec_0, spec), -1)
        _, freqs = torch.topk(torch.abs(torch.complex(torch.tensor(gui_data[v][...,2:gui_data.shape[-1]:2]), -torch.tensor(gui_data[v][...,1:gui_data.shape[-1]:2]))), 5)
        freqs += 1
        freq = freqs[0]
        sfreq = 2 * freq
        if sfreq > 113//2: sfreq = 113 - sfreq
        def phz_str(f): return str(f) + ": (" + tstr(torch.abs(torch.complex(torch.tensor(gui_data[v][...,f*2]), -torch.tensor(gui_data[v][...,f*2-1])))) + ") "  + " (" + tstr(torch.angle(torch.complex(torch.tensor(gui_data[v][...,f*2]), -torch.tensor(gui_data[v][...,f*2-1])))) + ") "
        # tprint([phz_str(ddd) for ddd in [14,35,41,42,52]])
        mag35, mag43 = torch.abs(spec[...,35]), torch.abs(spec[...,43])
        phase35, phase43 = torch.angle(spec[...,35]), torch.angle(spec[...,43])
        tprint("35", mag35, phase35, "43", mag43, phase43, "35/43 ratio", mag35/mag43)
      # else: print("mag", mag, "DC", dc)
      if get_message is not None and v == 0:
        msg_text = get_message(state.indices)
        # for ff in font_manager.get_fonts(): print(ff.family, "-", ff.variant)
        msg = gfx.Text(
          geometry=gfx.TextGeometry(
            text=msg_text,
            anchor="top-right",
            font_size=25,
            screen_space=True,
            # family=("Z003", )
          ),
          material=gfx.TextMaterial(color="#FFF"),
        )
        msg.local.x = data_points - 1
        msg.local.y = state.max
        lines.append(msg)
      if get_points is not None and v == 0:
        # must be an ndarray of size [num_points, 3]
        x_coords, labels = get_points(state.indices)
        pt_x = np.array(x_coords).astype(np.float32)
        # pt_y = np.zeros_like(pt_x)
        all_left = np.floor(pt_x).astype(int)
        all_frac = pt_x - all_left
        all_right = np.clip(np.ceil(pt_x), 0, data_points - 1).astype(int)
        pt_y = all_frac * gui_data[v][all_left] + (1 - all_frac) * gui_data[v][all_right]
        # for i, pt in enumerate(pt_x):
        #   left = int(pt)
        #   frac = pt - left
        #   if frac != 0: pt_y[i] = frac * gui_data[v][left] + (1-frac) * gui_data[v][min(left + 1, gui_data[v].shape[0] - 1)]
        #   else: pt_y[i] = gui_data[v][left]
        pt_positions = np.column_stack([pt_x, pt_y, np.zeros_like(pt_x)])
        sizes = np.full(pt_positions.shape[0], 8.0).astype(np.float32)
        pt_colors = np.tile(Color("white").rgba,(pt_positions.shape[0],1)).astype(np.float32)
        geometry = gfx.Geometry(positions=pt_positions.astype(np.float32), sizes=sizes, colors=pt_colors)
        
        # print("pt_positions", pt_positions, "left", all_left, "frac", all_frac, "right", all_right)

        material = gfx.PointsMaterial(color_mode="vertex", size_mode="vertex")
        points = gfx.Points(geometry, material)
        lines.append(points)

        for i, label in enumerate(labels):
          obj = gfx.Text(
            geometry=gfx.TextGeometry(
              text=label,
              anchor="bottom-left",
              font_size=20,
              screen_space=True,
            ),
            material=gfx.TextMaterial(color="#FFF"),
          )
          obj.local.x = pt_positions[i][0].item()
          obj.local.y = pt_positions[i][1].item()
          lines.append(obj)
      if state["gui_type"] == "radial":
        length = len(x)
        minx, maxx = 0.0, len(x)
        diameterx, midx = length, length/2
        miny, maxy = gui_data[v].min(), gui_data[v].max()
        diametery, midy = maxy - miny, miny + (maxy - miny)/2
        xlist, ylist = [], []
        for i in range(len(the_key_freqs)):
          xlist.append(midx)
          ylist.append(midy)
          xlist.append(midx + np.cos(gui_data[v][...,the_key_freqs[i]]) * diameterx/2)
          ylist.append(midy + np.sin(gui_data[v][...,the_key_freqs[i]]) * diametery/2)
          xlist.append(midx)
          ylist.append(midy)
        positions = np.column_stack([np.array(xlist, dtype=np.float32), np.array(ylist, dtype=np.float32), np.zeros_like(np.array(xlist, dtype=np.float32))])
        print("positions", positions)
      else:
        positions = np.column_stack([x, gui_data[v], np.zeros_like(x)])
      lines.append(gfx.Line(gfx.Geometry(positions=positions), gfx.LineMaterial(thickness=3.0, color=colors[v])))
      # this chunk of code below adds doubly symmetric lines that scroll with the play dim, should be removed
      # l1x, l1y = np.array([state.get_play_dim()/2,state.get_play_dim()/2]), np.array([gui_data[v].min(), gui_data[v].max()])
      # l2x, l2y = np.array([state.get_play_dim()/2 + 56.5,state.get_play_dim()/2 + 56.5]), np.array([gui_data[v].min(), gui_data[v].max()])
      # l1positions = np.column_stack([l1x.astype(np.float32), l1y.astype(np.float32), np.zeros_like(l1x).astype(np.float32)])
      # l2positions = np.column_stack([l2x.astype(np.float32), l2y.astype(np.float32), np.zeros_like(l2x).astype(np.float32)])
      # lines.append(gfx.Line(gfx.Geometry(positions=l1positions), gfx.LineMaterial(thickness=3.0, color="white")))
      # lines.append(gfx.Line(gfx.Geometry(positions=l2positions), gfx.LineMaterial(thickness=3.0, color="white")))
    return lines
  
  
  
def sigmoid(x, bB, bShift): return generalized_logistic(x, B=bB, shift=bShift)# def sigmoid(x, bB, bShift): return 1/(1 + torch.exp(-x))
def mix(x,y,ratio=1): return ratio * x + (1-ratio) * y
  
class Hilbert(GuiStyle):
  name = "hilbert"
  n_axes = 1
  n_gui_axes = 2
  @classmethod
  def setup_camera(cls, state, camera, canvas):
    diameter = max(abs(state.min), abs(state.max))
    camera.show_rect(state.min-diameter/8, state.max+diameter * 1.05 + diameter/8, state.min-diameter/8, state.max+diameter * 1.05 + diameter/8)
    return None
  def transform_data(state, all_data):
    x = state.get_permuted_data(all_data)
    x = torch.fft.fft(x)
    x[...,0] = 0
    x = torch.view_as_real(x)
    return x.cpu().numpy()
  def get_renderables(state, gui_data, sphere):
    colors = ["#faa", "#474", "#447", "#774", "#747", "#477"]
    all_points = []
    for v in range(gui_data.shape[0]):      
      positions = np.column_stack([gui_data[v][...,0], gui_data[v][...,1], np.zeros_like(gui_data[v][...,1])])
      print("gui_data[v]", gui_data[v].shape, "positions", positions.shape, "positions[:5]", positions[:5])
      points = gfx.Points(
        gfx.Geometry(positions=positions),
        gfx.PointsMaterial(size=4, color=colors[v]),
      )
      all_points.append(points) 
    return all_points
  
class KeyPhases(GuiStyle):
  name = "keyphases"
  n_axes = 2
  n_gui_axes = 2
  key_freqs = [14,35,41,42,52]
  @classmethod
  def get_min(cls, tensor): return -max(cls.key_freqs)
  @classmethod
  def get_max(cls, tensor): return max(cls.key_freqs)
  @classmethod
  def get_gui_ranges(cls, state):
    return [[-max(cls.key_freqs), max(cls.key_freqs)], [-max(cls.key_freqs), max(cls.key_freqs)]]
  @classmethod
  def setup_camera(cls, state, camera, canvas):
    diameter = max(abs(state.min), abs(state.max))
    camera.show_rect(state.min-diameter/8, state.max + diameter/8, state.min-diameter/8, state.max + diameter/8)
    return None
  @classmethod
  def transform_data(cls, state, all_data):
    x = state.get_permuted_data(all_data)
    print("orig x", x.shape)
    X = torch.fft.fft(x)[...,cls.key_freqs]
    print("X -> torch.fft.fft(x)[...,cls.key_freqs]", X.shape)
    angles = torch.angle(X)
    print("angles", angles.shape, "angles[...,0,0] min", angles[...,0,0].min().item(), "angles[...,0,0] max", angles[...,0,0].max().item())
    X = torch.exp(1j * angles) * torch.tensor(cls.key_freqs, device=x.device)[None,None,...]
    print("torch.exp(1j * angles)", torch.exp(1j * angles).shape)
    print("torch.tensor(cls.key_freqs, device=x.device)[None,None,...]", torch.tensor(cls.key_freqs, device=x.device)[None,None,...].shape)
    print("X", X.shape)
    X = torch.view_as_real(X)
    print("X -> torch.view_as_real(X)", X.shape)
    return X.cpu().numpy()
  @classmethod
  def get_renderables(cls, state, gui_data, sphere):
    n_points = gui_data.shape[1]
    colors = [(i / len(cls.key_freqs), 1 - (i/len(cls.key_freqs)), 1.0) for i in range(len(cls.key_freqs))]
    all_points = []
    for f in range(len(cls.key_freqs)):
      for i in range(n_points):
        positions = np.column_stack([gui_data[0][...,i,f,0], gui_data[0][...,i,f,1], np.zeros_like(gui_data[0][...,i,f,1])])
        # print("gui_data[0]", gui_data[0].shape, "positions", positions.shape, "positions[:5,:]", positions[:5,:])
        new_color = colors[f][:2] + (i/n_points,)
        
        points = gfx.Points(
          gfx.Geometry(positions=positions),
          gfx.PointsMaterial(size=4, color=new_color),
        )
        all_points.append(points) 
    return all_points
  
class KeyPhase(GuiStyle):
  name = "keyphase"
  n_axes = 1
  n_gui_axes = 2
  key_freqs = [14,35,41,42,52]
  @classmethod
  def get_min(cls, tensor): return -max(cls.key_freqs)
  @classmethod
  def get_max(cls, tensor): return max(cls.key_freqs)
  @classmethod
  def get_gui_ranges(cls, state):
    return [[-max(cls.key_freqs), max(cls.key_freqs)], [-max(cls.key_freqs), max(cls.key_freqs)]]
  @classmethod
  def setup_camera(cls, state, camera, canvas):
    diameter = max(abs(state.min), abs(state.max))
    camera.show_rect(state.min-diameter/8, state.max + diameter/8, state.min-diameter/8, state.max + diameter/8)
    return None
  @classmethod
  def transform_data(cls, state, all_data):
    x = state.get_permuted_data(all_data)
    print("orig x", x.shape)
    X = torch.fft.fft(x)[...,cls.key_freqs]
    print("X -> torch.fft.fft(x)[...,cls.key_freqs]", X.shape)
    angles = torch.angle(X)
    print("angles", angles.shape, "angles[...,0,0] min", angles[...,0,0].min().item(), "angles[...,0,0] max", angles[...,0,0].max().item())
    X = torch.exp(1j * angles) * torch.tensor(cls.key_freqs, device=x.device)[None,None,...]
    print("torch.exp(1j * angles)", torch.exp(1j * angles).shape)
    print("torch.tensor(cls.key_freqs, device=x.device)[None,None,...]", torch.tensor(cls.key_freqs, device=x.device)[None,None,...].shape)
    print("X", X.shape)
    X = torch.view_as_real(X)
    print("X -> torch.view_as_real(X)", X.shape)
    return X.cpu().numpy()
  @classmethod
  def get_renderables(cls, state, gui_data, sphere):
    n_points = gui_data.shape[1]
    colors = [(i / len(cls.key_freqs), 1 - (i/len(cls.key_freqs)), 1.0) for i in range(len(cls.key_freqs))]
    all_points = []
    for f in range(len(cls.key_freqs)):
      positions = np.column_stack([gui_data[0][...,f,0], gui_data[0][...,f,1], np.zeros_like(gui_data[0][...,f,1])])
      # print("gui_data[0]", gui_data[0].shape, "positions", positions.shape, "positions[:5,:]", positions[:5,:])
      new_color = colors[f]
      
      points = gfx.Points(
        gfx.Geometry(positions=positions),
        gfx.PointsMaterial(size=4, color=new_color),
      )
      all_points.append(points) 
    return all_points
  
class Heatmap(GuiStyle):
  name = "heatmap"
  n_axes = 2
  n_gui_axes = 2
  @classmethod
  def setup_camera(cls, state, camera, canvas):
    sizes = [state.get_gui_dim_sizes()[i] for i in range(len(state.get_gui_dim_sizes()))]
    camera.show_rect(-round(sizes[0]/8), round(sizes[0] * 1.05) + round(sizes[0]/8), -round(sizes[1]/8), round(sizes[1] * 1.05) + round(sizes[1]/8))
    return None
  def transform_data(state, all_data):
    x = state.get_permuted_data(all_data)
    pmin, pmax = x.min().item(), x.max().item()
    if state["gui_type"] == "fourier": 
      x = state.fft2d(x)
      pmin, pmax = x[...,1:,1:].min().item(), x[...,1:,1:].max().item()
    
    state["bB"], state["bShift"] = 1,1#tune_diverse_logistic_paramaters(x)
    x = mix(x, sigmoid(x, state["bB"], state["bShift"]), pow(2, 3 * (state["contrast"] - (0.5 if state["gui_type"] == "spacetime" else -0.8))))
    x = (x - pmin) / (0.00000001 + pmax - pmin)
    x = x * pow(2, 4 * (state["brightness"] - (0.5 if state["gui_type"] == "spacetime" else 0.0)))
    return x.cpu().numpy()
  def get_renderables(state, gui_data, sphere):
    xlen, ylen = state.get_gui_dim_sizes()
    image_256 = np.clip(np.rint((np.flip(gui_data[0], axis=(-2)) * 256)).astype(int), a_min=0, a_max=255)
    # cmap = gfx.utils.cm.viridis.data
    image_texture = gfx.Texture(cmap[image_256], dim=2)#, format="1xf4", force_contiguous=True)
    image_mesh = gfx.Mesh(gfx.plane_geometry(width=xlen, height=ylen), gfx.MeshBasicMaterial(map=image_texture),)
    image_mesh.local.x += xlen / 2#.local.x + 50 + 512 + 256 / 2
    image_mesh.local.y += ylen / 2
    return [image_mesh]

def rotate_axis(input, angle, axis):
  vert = np.array(input)
  cosa, sina = np.cos(angle), np.sin(angle)
  if axis == "x": return (vert @ np.array([[1,0,0], [0,cosa,-sina], [0,sina,cosa]]))
  elif axis == "y": return (vert @ np.array([[cosa,0,sina], [0,1,0], [-sina,0,cosa]]))
  elif axis == "z": return (vert @ np.array([[cosa,-sina,0], [sina,cosa,0], [0,0,1]]))
  print("trying to rotate unknown axis:", axis)
  return input

def rotate_axes(input, angles, center = (0,0,0)):
  vert = np.array(input) - np.array(center)
  for i in range(len(angles)): vert = rotate_axis(vert, angles[i], "x" if i == 0 else "y" if i == 1 else "z")
  return vert + np.array(center)

def fill_colormap(x, smin=None, smax=None, cmap=None):# = gfx.utils.cm.viridis.data):
  xmin, xmax = x.min(), x.max()
  if not (smin is None and smax is None): x = (x - xmin) * 255 * ((xmax-xmin)/(smax-smin)) / (0.00000001 + xmax - xmin)
  else: x = (x - xmin) * 255 / (0.00000001 + xmax - xmin)
  x = np.clip(x,0,255).astype(np.uint8)
  return cmap[x]

def fill_colormap_idx(x, xmin=None, xmax=None, cmap=None):# = gfx.utils.cm.plasma.data):
  if xmin is None and xmax is None: xmin, xmax = x.min(), x.max()
  x = (x - xmin) * 255 / (0.00000001 + xmax - xmin)
  x = np.clip(x,0,255).astype(np.uint8)
  return x

class Contour(GuiStyle):
  name = "contour"
  orientation = "xy"
  n_gui_axes = 3
  @classmethod
  def setup_camera(cls, state, camera, canvas):
    width, height = state.get_gui_dim_sizes()
    if state["gui_type"] == "fourier":
      width = width//2+1
      height = height//2+1
    radius = max(width, height)/2
    angles = (0,0,np.pi - np.pi/6)
    sphere = (radius*0.95,radius*0.75,radius*1.1, radius*1.45)
    camera_pos = (-radius*1, -radius*1.5, -radius*8)
    camera.show_object(sphere, rotate_axes(camera_pos, angles, sphere[:3]), up=(0,0,1))
    return sphere

  def transform_data(state, all_data):
    x = state.get_permuted_data(all_data)
    if state["gui_type"] == "fourier":
      x = state.fft2d(x)
      state.min, state.max = x[...,1:,1:].min().item(), x[...,1:,1:].max().item()
    else:
      state.min, state.max = x.min().item(), x.max().item()
    # x = (x - state.min) / (0.00000001 + state.max - state.min)
    # x = torch.clamp(x, state.min, state.max)
    return x.cpu().numpy()
  def get_renderables(state, gui_data, sphere):
    xlen, ylen = state.get_gui_dim_sizes()
    image_256 = np.clip(np.rint((np.flip(gui_data[0], axis=(-2)) * 256)).astype(int), a_min=0, a_max=255)
    # cmap = gfx.utils.cm.viridis.data
    image_texture = gfx.Texture(cmap[image_256], dim=2)#, format="1xf4", force_contiguous=True)
    image_mesh = gfx.Mesh(gfx.plane_geometry(width=xlen, height=ylen), gfx.MeshBasicMaterial(map=image_texture),)
    
    geometry = contour_geometry(gui_data[0])
    colors = geometry.positions.data[...,2:3]
    # need to fix this because colab complained that 
    # colors = fill_colormap(colors) if state["gui_type"] == "spacetime" else fill_colormap(colors, state.min, state.max)
    setattr(geometry, "colors", gfx.Buffer(colors))
    material = gfx.MeshPhongMaterial(color_mode="vertex", side="both")
    contour = gfx.Mesh(geometry, gfx.MeshBasicMaterial(map=image_texture),)#material)
    light = gfx.DirectionalLight("#fff", 4, target=contour)
    light.local.position = (sphere[0], sphere[1], sphere[2] + sphere[3] * (state["brightness"]-0.5),)
    

    data_scaler = np.eye(4)
    data_scaler[2][2] = max(xlen, ylen) / (0.0000001 + state.max - state.min)
    contour.local.matrix = data_scaler
    axesHelper = gfx.helpers.AxesHelper(30 if state["gui_type"] == "fourier" else 60, 10)
    return [contour, axesHelper, light]
