import os
import torch
import numpy as np
from wgpu.gui.jupyter import WgpuCanvas, run
import pygfx as gfx
from pylinalg import vec_transform, vec_unproject
import ipywidgets as widgets
from ipywidgets import Widget
from IPython.display import display, Javascript
from checkpaint.utils import *
import numbers
from dataclasses import dataclass

def fft2d_new2(x):
  shape = x.shape
  x = torch.fft.fft(x)
  xr, xi = x.real, x.imag
  XR = torch.fft.fft(xr, dim=-2)
  XI = torch.fft.fft(xi, dim=-2)
  printvars(x, xr, xi, XR, XI)
  # for dim in range(-2,0,1):
  #   N = x.size(dim)
  #   if dim != -1: x = x.swapaxes(dim, -1)
  #   new_shape = x.shape
  #   x = x.view(-1, x.shape[-1])
  #   x = torch.fft.fft(x) / np.sqrt(N/2)
  #   x = torch.view_as_real(torch.complex(-x.imag, x.real)).flatten(-2,-1)[...,1:N+1]
  #   x[...,0] *= (np.sqrt(2)/2)
  #   x = x.reshape(new_shape)
  #   if dim != -1: x = x.swapaxes(dim, -1)
  return x

# def fft2d_new(x):
#     shape = x.shape
#     M,N = shape[-2], shape[-1]
#     x = torch.fft.fft(x)
#     xr, xi = x.real, x.imag
#     XR = torch.view_as_real(torch.fft.fft(xr, dim=-2))
#     XI = torch.view_as_real(torch.fft.fft(xi, dim=-2))
#     print(x.shape, xr.shape, xi.shape, XR.shape, XI.shape)
#     x = torch.stack((XR[...,:], XI[...,:]), -3)
#     print(x.shape)
#     x = x.flatten(-3,-2)[...,1:M+1,:].flatten(-2,-1)[...,1:N+1]
#     print(x.shape)
#     return x

def fft2d_new(x):
    shape = x.shape
    M,N = shape[-2], shape[-1]
    x = torch.fft.fft(x, dim=-2)
    x = torch.fft.fft(x, dim=-2)
    print("double fft shape", x.shape)
    xr, xi = x.real, x.imag
    XR = torch.view_as_real(torch.fft.fft(xr, dim=-2))
    XI = torch.view_as_real(torch.fft.fft(xi, dim=-2))
    print(x.shape, xr.shape, xi.shape, XR.shape, XI.shape)
    x = torch.stack((XR[...,:], XI[...,:]), -3)
    print(x.shape)
    x = x.flatten(-3,-2)[...,1:M+1,:].flatten(-2,-1)[...,1:N+1]
    print(x.shape)
    return x
  
def fft2d0(x):
  shape = x.shape
  X = torch.fft.fft(x, dim=-2)
  X = torch.view_as_real(X).transpose(-2,-1).flatten(-3,-2)
  X = torch.fft.fft(X, dim=-1)
  X = torch.view_as_real(X).flatten(-2,-1)
  X = torch.cat((X[...,0:1,:], X[...,2:shape[-2]+1,:]), -2)
  X = torch.cat((X[...,0:1], X[...,2:shape[-1]+1]), -1)
  return X

def fft2d1(x): return torch.fft.fft2(x).abs()

def from_torch_spectrum(x, dim=-1):
    N = x.size(dim)
    x = x/np.sqrt(N/2)
    tx = torch.view_as_real(torch.complex(-x.imag, x.real)).flatten(-2,-1)[1:N+1]
    tx[...,0] *= (np.sqrt(2)/2)
    return tx

def from_torch_2d_spectrum(X):
    M, N = X.shape[-2], X.shape[-1]
    X = X/(np.sqrt(M/2) * np.sqrt(N/2))
    TX = torch.view_as_real(torch.complex(-X.imag, X.real)).flatten(-2,-1)[...,1:M+1,1:N+1]
    TX[...,0,:] *= (np.sqrt(2)/2) 
    TX[...,:,0] *= (np.sqrt(2)/2) 
    return TX

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

identity_map = lambda x,start,end : str(x) if isinstance(x, numbers.Integral) else str(f"{x:.1e}")
fourier_map = lambda x,start,end : str(f"cos{int(x/2)}")
class ScaledMap:
  def __init__(self, map, scale_factor):
    self.map = map
    self.sf = scale_factor
  def __call__(self, x, start, end): return self.map(x * self.sf,start,end)
  

#### NICKED FROM generate_plane() IN pygfx/geometries/_plane.py ####
  
def generate_contour(data):
  width, height = data.shape[-2], data.shape[-1]
  width_segments, height_segments = width - 1, height - 1
  nx, ny = width_segments + 1, height_segments + 1
  
  # width/=4
  # height/=4

  x = np.linspace(0, width, nx, dtype=np.float32)
  y = np.linspace(0, height, ny, dtype=np.float32)
  xx, yy = np.meshgrid(x, y)
  xx, yy = xx.flatten(), yy.flatten()
  positions = np.column_stack([xx, yy, data.flatten()])

  dim = np.array([width, height], dtype=np.float32)
  texcoords = (positions[..., :2] + dim / 2) / dim
  texcoords[..., 1] = 1 - texcoords[..., 1]

  # the amount of vertices
  indices = np.arange(ny * nx, dtype=np.uint32).reshape((ny, nx))
  # for every panel (height_segments, width_segments) there are two triangles (4, 3)
  index = np.empty((height_segments, width_segments, 4, 3), dtype=np.uint32)
  # create a grid of initial indices for the panels
  index[:, :, 0, 0] = indices[
      np.arange(height_segments)[:, None], np.arange(width_segments)[None, :]
  ]
  # the remainder of the indices for every panel are relative
  index[:, :, 0, 1] = index[:, :, 0, 0] + 1
  index[:, :, 0, 2] = index[:, :, 0, 0] + nx
  index[:, :, 1, 0] = index[:, :, 0, 0] + nx + 1
  index[:, :, 1, 1] = index[:, :, 1, 0] - 1
  index[:, :, 1, 2] = index[:, :, 1, 0] - nx

  normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (ny * nx, 1))

  return positions, normals, texcoords, index.reshape((-1, 3))


def contour_geometry(data):
  """Generate a plane.

  Creates a flat (2D) rectangle in the local xy-plane that has its center at
  local origin. The plane may be subdivided into segments along the x- or
  y-axis respectively.

  Parameters
  ----------
  width : float
      The plane's width measured along the x-axis.
  height : float
      The plane's width measured along the y-axis.
  width_segments : int
      The number of evenly spaced segments along the x-axis into which the
      plane should be divided.
  height_segments : int
      The number of evenly spaced segments along the y-axis into which the
      plane should be divided.

  Returns
  -------
  plane : Geometry
      A geometry object representing the requested plane.
      Mathematically, it is an open orientable manifold.

  """

  positions, normals, texcoords, indices = generate_contour(
      data#width, height, width_segments, height_segments
  )

  return gfx.Geometry(
      indices=indices.reshape((-1, 3)),
      positions=positions,
      normals=normals,
      texcoords=texcoords,
  )

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
  orientation = "xy"
  
class Lines(GuiStyle):
  name = "lines"
  n_axes = 1
  def setup_camera(state, camera):
    size = state.get_gui_sizes()[0]
    y_range = (max(state.max - state.min, 0.00001))
    camera.set_view_size(round(size * 1.05) + round(size/8), round(y_range * 1.05) + round(y_range/8))
    camera.show_rect(-round(size/10), round(size * 1.05), state.min, state.max)
  def transform_data(state, all_data):
    permuted_data = state.get_permuted_data(all_data)
    if state["gui_type"] == "fourier":
      # print("gui_data", gui_data.shape, "basis shape", state.bases[gui_data.shape[-1]].shape)
      permuted_data = state.fft1d(permuted_data)
    # gui_tensor = (gui_tensor - (state.max - state.min)/2) / (0.00000001 + state.max - state.min)
    return permuted_data.cpu().numpy()
  def get_renderables(state, gui_data):
    # print("painting lines frame gui_data", gui_data.shape)
    data_points = state.get_gui_sizes()[0]
    x = np.linspace(0, data_points, data_points, endpoint=False, dtype=gui_data.dtype)
    colors = ["#faa", "#474", "#447", "#774", "#747", "#477"]
    lines = []
    for v in range(gui_data.shape[0]):
      line_data = gui_data[v]# * state.scale()[1]
      positions = np.column_stack([x, line_data, np.zeros_like(x)])
      lines.append(gfx.Line(gfx.Geometry(positions=positions), gfx.LineMaterial(thickness=3.0, color=colors[v]))) 
    return lines
  
def sigmoid(x, bB, bShift): return generalized_logistic(x, B=bB, shift=bShift)# def sigmoid(x, bB, bShift): return 1/(1 + torch.exp(-x))
def mix(x,y,ratio=1): return ratio * x + (1-ratio) * y
def dont_transform_data(all_data, state): return state.get_permuted_data(all_data).cpu().numpy()
def print_colormap_values(x, N, name=""):
  x = x.flatten()
  length = x.shape[0]
  x, _ = torch.sort(x)
  print(name, torch.round(x[::length//N], decimals=4).tolist(), "last 4", torch.round(x[-4:], decimals=4).tolist())
  
class Heatmap(GuiStyle):
  name = "heatmap"
  n_axes = 2
  def setup_camera(state, camera):
    sizes = [state.get_gui_sizes()[i] for i in range(len(state.get_gui_sizes()))]
    camera.set_view_size(round(sizes[0] * 1.05) + round(sizes[0]/8), round(sizes[1] * 1.05) + round(sizes[1]/8))
    ############ THIS IS THE LINE THAT SKEWS THE GRAPH. COMMENTING IT OUT TO MAKE THE DATA FULL SIZE #######
    # camera.show_rect(-round(sizes[0]/8), round(sizes[0] * 1.05), -round(sizes[1]/8), round(sizes[1] * 1.05))
  def transform_data(state, all_data):
    # if state["gui_type"] == "fourier":
      # print("first axis elements all_data (transform_data_heatmap)", get_random_elements(all_data))
    permuted_data = state.get_permuted_data(all_data)
    
    if state["gui_type"] == "fourier":
      print_colormap_values(permuted_data, 10, "permuted_data 0")
      # print("first axis elements permuted_data", get_random_elements(permuted_data))
      # size_before = [permuted_data.shape[-2], permuted_data.shape[-1]]
      permuted_data = state.fft2d(permuted_data)
      print_colormap_values(permuted_data, 10, "permuted_data after fft")
      # size_after = [permuted_data.shape[-2], permuted_data.shape[-1]]
      # print("size before rfft (previously):", size_before, "size after", size_after)
      # permuted_data = torch.log(permuted_data/(permuted_data.shape[-2] * permuted_data.shape[-1]))
      # print_colormap_values(permuted_data, 10, "permuted_data after taking log")
      # print("first axis elements fft permuted", get_random_elements(permuted_data))
    pmin, pmax = permuted_data.min().item(), permuted_data.max().item()
    # if state["gui_type"] == "fourier": permuted_data = permuted_data * 2
    bB = state["bB"]
    bShift = state["bShift"]#tune_diverse_logistic_paramaters(gui_tensor)
    bB, bShift = tune_diverse_logistic_paramaters(permuted_data)
    permuted_data = mix(permuted_data, sigmoid(permuted_data, bB, bShift), pow(2, 3 * (state["contrast"] - (0.5 if state["gui_type"] == "spacetime" else -0.3))))
    if state["gui_type"] == "fourier": print_colormap_values(permuted_data, 10, "permuted_data after contrast")
    
    permuted_data = (permuted_data - pmin) / (0.00000001 + pmax - pmin)
    if state["gui_type"] == "fourier": print_colormap_values(permuted_data, 10, "permuted_data normalized")
    permuted_data = permuted_data * pow(2, 4 * (state["brightness"] - (0.5 if state["gui_type"] == "spacetime" else 0.3)))
    if state["gui_type"] == "fourier": print_colormap_values(permuted_data, 10, "permuted_data after brightness (LAST)")
    return permuted_data.cpu().numpy()
  def get_renderables(state, gui_data):
    # print("painting heatmap frame gui_data", gui_data.shape)
    orig_xlen, orig_ylen = state.shape[-2], state.shape[-1]
    new_gui_data = gui_data[0]
    # print("new_gui_data", new_gui_data.shape)
    image_data = np.flip(new_gui_data, axis=(-2))
    # image_data = gui_data[0][state.get_play_dim()]
    image_idx = np.clip(np.rint((image_data * 256)).astype(int), a_min=0, a_max=255)
    # print("image_data.shape", image_data.shape, "image_idx.shape", image_idx.shape)
    # image_idx = np.repeat(image_idx, 3, axis=-1)
    # print("image_idx.shape 2", image_idx.shape)
    cmap = gfx.utils.cm.viridis.data
    # print("cmap", cmap.shape)
    image_texture = gfx.Texture(cmap[image_idx], dim=2)#, format="1xf4", force_contiguous=True)
    image_mesh = gfx.Mesh(gfx.plane_geometry(width=orig_xlen, height=orig_ylen), gfx.MeshBasicMaterial(map=image_texture),)
    image_mesh.local.x += orig_xlen / 2#.local.x + 50 + 512 + 256 / 2
    image_mesh.local.y += orig_ylen / 2
    # image_mesh.local.scale_y = -1

    return [image_mesh]

# def setup_camera_contour(state, camera):
#   setup_camera_heatmap(state, camera)
#   sizes = [state.get_gui_sizes()[i] for i in range(len(state.get_gui_sizes()))]
#   camera.set_view_size(round(sizes[0] * 1.05) + round(sizes[0]/8), round(sizes[1] * 1.05) + round(sizes[1]/8))
  

class Contour(GuiStyle):
  name = "contour"
  def setup_camera(state, camera):
    # camera = gfx.PerspectiveCamera(70, 1)
    # camera.local.z = 4
    width, height = state.get_gui_sizes()
    radius = max(width, height)/2
    camera.show_object((width/2, height/2, radius/2, radius*0.77), (radius*3, radius*3, radius*2))#, up=(0,0,1))
    # camera.set_view_size(round(width * 1.05) + round(width/8), round(height * 1.05) + round(height/8))
    ############ THIS IS THE LINE THAT SKEWS THE GRAPH. COMMENTING IT OUT TO MAKE THE DATA FULL SIZE #######
    # camera.show_rect(-round(sizes[0]/8), round(sizes[0] * 1.05), -round(sizes[1]/8), round(sizes[1] * 1.05))
  def transform_data(state, all_data):
    width, height = state.get_gui_sizes()
    permuted_data = state.get_permuted_data(all_data)
    permuted_data = (permuted_data - state.min) * 0.05 * max(width, height) / (0.00000001 + state.max - state.min)
    return permuted_data.cpu().numpy()
  def get_renderables(state, gui_data):
    # orig_xlen, orig_ylen = state.shape[-2], state.shape[-1]
    new_gui_data = gui_data[0]
    image_data = np.flip(new_gui_data, axis=(-2))
    # image_idx = np.clip(np.rint((image_data * 256)).astype(int), a_min=0, a_max=255)
    # cmap = gfx.utils.cm.viridis.data
    # image_texture = gfx.Texture(cmap[image_idx], dim=2)#, format="1xf4", force_contiguous=True)
    
    width, height = state.get_gui_sizes()
    # x = np.linspace(0, width, width, endpoint=False, dtype=gui_data.dtype)
    # y = np.linspace(0, height, height, endpoint=False, dtype=gui_data.dtype)
    # xx, yy = np.meshgrid(x, y)
    # positions = np.hstack([xx, yy, gui_data[0]]).reshape(-1,3)
    floor_material = gfx.MeshBasicMaterial(color=(0.2, 0.2, 0.2, 1.0), wireframe=True, wireframe_thickness=2, side="front")#, map=image_texture)
    floor_geometry = contour_geometry(np.ones_like(image_data))#gfx.plane_geometry(width, height, width - 1, height - 1)
    plane = gfx.Mesh(floor_geometry, floor_material)
    # plane.local.x += width / 2
    # plane.local.y += height / 2
    material1 = gfx.MeshBasicMaterial(color=(0.3, 0.8, 0.8, 1))
    material = gfx.MeshPhongMaterial(color=(0.2, 0.6, 0.6, 1), wireframe=True, wireframe_thickness=1.5)#, map=image_texture)
    geometry = contour_geometry(image_data)
    obj1 = gfx.Mesh(geometry, material1)
    contour = gfx.Mesh(geometry, material)
      
    return [obj1, contour, gfx.helpers.AxesHelper(30, 10), gfx.AmbientLight(0.2)]
