import torch
import numpy as np
from wgpu.gui.jupyter import WgpuCanvas# JupyterWgpuCanvas
from wgpu.gui.jupyter import run
import pygfx as gfx
from pylinalg import vec_transform, vec_unproject
import ipywidgets as widgets
from IPython.display import display, HTML
from checkpaint.utils import *
import numbers
from pygfx.renderers import jupyter_rfb
# from dataclasses import dataclass
from .gui import *
from pygfx.utils.text._fontfinder import get_all_fonts, get_builtin_fonts

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
    
class AxisState:
  def __init__(self, tensors, gui, start_idx=[0], scale_factor=1):
    self.gui = gui
    self.scale_factor = scale_factor
    self.n_axes = gui.n_axes
    self.style = gui.name
    self.shape = tensors[0].shape
    tensor = tensors[0]
    self.bases = {d: get_fourier_basis(d).to(tensor.device) for d in self.shape}
    N = len(self.shape)
    self.axes = [N - self.n_axes + i for i in range(self.n_axes)]
    self.play_axis = 0
    self.indices = [i - N if i > N - self.n_axes else start_idx[i] if i < len(start_idx) else 0 for i in range(N)]
    # finiteonly = torch.isfinite(tensor)
    # self.min = gui.get_min(tensor[finiteonly])
    # self.max = gui.get_max(tensor[finiteonly])
    self.min = gui.get_min(tensors)
    self.max = gui.get_max(tensors)
    self.data = { "brightness": 0.5, "contrast": 0.5, "gui_type": "spacetime" }
        
  def __getitem__(self, index): return self.data[index]
  def __setitem__(self, index, value): self.data[index] = value
  def set_play_dim(self, d: int): self.indices[self.play_axis] = d
  def get_play_dim(self): return self.indices[self.play_axis]
  def set_play_axis(self, a: int):
    self.play_axis = a
    self.indices[a] = max(0, self.indices[a])
  def get_gui_dim_sizes(self): return [self.shape[a] for a in self.axes]
    
  def with_axes(self, axes):
    for a in range(len(axes)):
      for i in range(len(self.indices)):
        if self.indices[i] == -self.n_axes + a:
          self.indices[i] = 0
      self.indices[axes[a]] = -self.n_axes + a
    self.axes = axes
    return self
  
  def get_permuted_data(self, original_data):
    permuted_data = original_data.detach().clone()
    for a in self.axes:
      permuted_data = permuted_data.transpose(a+1, self.indices[a])
    return permuted_data
  
  def get_gui_frame(self, permuted_data):
    idx = [x for x in self.indices if x >= 0]
    frame = permuted_data
    # print("frame = permuted_data", frame.shape, "self.axes", self.axes, "self.indices", self.indices, "idx", idx)
    for i in reversed(range(len(idx))):
      if idx[i] >= 0: frame = np.take(frame, idx[i], axis=i+1)
    return frame
  
  def fft1d(self, x):
    X = ((x/np.sqrt(x.shape[-1]))/(np.sqrt(2)/2)) @ self.bases[x.shape[-1]].transpose(-2,-1)
    X[...,0] /= 2
    return X
  def fft2d(self, x): return torch.fft.rfft2(x)[...,:x.shape[-2]//2+1,:].abs()
  
class WorldGrid:
  def __init__(self, state, gui, axis_map = []):
    self.style = state.style
    self.ruler = tuple(gfx.Ruler(tick_side="right" if i == 0 else "left") for i in range(gui.n_gui_axes))
    dims = self.get_dimensionality()
    self.axis_map = tuple(axis_map[state.axes[i]] if i < min(len(axis_map), len(state.axes)) else identity_map for i in range(dims))
    while len(self.axis_map) < dims: self.axis_map.insert(0, identity_map)
    ranges = gui.get_gui_ranges(state)[:2]
    # axes_range = tuple(state.shape[a] for a in state.axes)
    # data_range = (state.max + 0.000001) - state.min
    # maj = tuple(get_tick(data_range if (state.style in ("lines")) and i == dims - 1 else axes_range[i]) for i in range(min(2,dims)))
    # if len(maj) == 1:
    #   maj = maj + maj
    maj = tuple(get_tick(ranges[i][1] - ranges[i][0]) for i in range(dims))
    self.gui_type = state["gui_type"]
    # print("state.axes", state.axes, "maj", maj, "dims", dims, "self.axis_map", self.axis_map)

    self.grid = gfx.Grid(
      None,
      gfx.GridMaterial(
        major_step=maj,
        minor_step=tuple(maj[i]/5 for i in range(len(maj))),
        thickness_space="screen",
        major_thickness=2,
        minor_thickness=0.5,
        infinite=True,
      ),
      orientation=gui.orientation
    )
    self.grid.local.z = -0.5 if self.style == "contour" else -1
    
  def get_dimensionality(self): return min(2,len(self.ruler))
    
  def update(self, canvas, camera, renderer):
    ssize = renderer.logical_size
    dims = self.get_dimensionality()
    world_min = map_screen_to_world(camera, (0, ssize[1], 0), renderer.logical_size)
    world_max = map_screen_to_world(camera, (ssize[0], 0, 0), renderer.logical_size)
    
    for i in range(dims):
      self.ruler[i].tick_format = fourier_map if self.gui_type == "fourier" and not (i > 0 and self.style == "lines") else self.axis_map[i] if i < len(self.axis_map) else identity_map
      # set start and end positions of rulers
      self.ruler[i].start_pos = self.ruler[i].end_pos = (0,0,0)
      self.ruler[i].start_value = self.ruler[i].start_pos[i] = world_min[i]
      self.ruler[i].end_pos[i] = world_max[i]
      # print("i", i, "self.ruler", self.ruler[i].start_pos, self.ruler[i].end_pos, self.ruler[i].start_value, self.ruler[i].end_value)
    stats = tuple(self.ruler[i].update(camera, canvas.get_logical_size()) for i in range(dims))
    maj = tuple(stats[i]["tick_step"] for i in range(dims))
    if len(maj) == 1:
      maj = maj + maj
    self.grid.material.major_step = maj
    self.grid.material.minor_step = tuple(maj[i]/5 for i in range(len(maj)))
    
  # def get_grid_chunk(state, orient, pos, width, height)
    
##########################################################################################################################
###########################################      DRAW DATA       ######################################################
##########################################################################################################################

#### label_maps are functions that take x,start,end and map to strings for major axis tick marks
#### they are a list making up the last len(label_maps) axes. the remainder of axes will be supplied the identity_map
def draw_data(tensors, gui, *,
              time_in_seconds=10, 
              label_maps=None,
              descriptor=None,
              get_points=None,
              get_message=None,
              axis_names=[], 
              scale_factor=1,
              start_play_axis=0,
              start_indices=[0],
              start_gui_type="spacetime",
              add_more_renderables=None):
  # print("builtin_fonts", get_builtin_fonts())
  # print("all_fonts", get_all_fonts())
  # printCudaMemUsage("beginning of draw_data for " + gui.name)
  if isinstance(tensors, torch.Tensor): tensors = [tensors]
  for t in tensors:
    while t.ndim <= gui.n_axes:
      t = t.unsqueeze(0)
  state = AxisState(tensors, gui, start_indices, scale_factor)
  state["gui_type"] = start_gui_type
  shape = state.shape
  label_maps = [] if label_maps is None else label_maps if isinstance(label_maps, list) else [label_maps]
  while len(label_maps) < len(shape): label_maps = [identity_map] + label_maps
  all_data = torch.stack(tensors)
  # print("first axis elements all_data", get_random_elements(all_data))

  hueSliders = [widgets.FloatSlider(value=0.5, max=1, step=0.01, description="brightness"), widgets.FloatSlider(value=0.5, max=1, step=0.01, description="contrast")]
  play, slider = widgets.Play(min=0, max=shape[0]-1, value=0), widgets.IntSlider(min=0, max=shape[0]-1, value=0, step=1, readout=False)
  play_link = widgets.jslink((play, 'value'), (slider, 'value'))

  selected_layout = widgets.Layout(width='max-content', height='30px', border='2px solid blue')
  regular_layout = widgets.Layout(width='max-content', height='30px')
  playAxisButtons, guiAxisMenus = [], []
  for d in range(len(shape)):
    playAxisButtons.append(widgets.Button(decription='0', 
                                          font_weight='bold' if d == 0 else 'normal', 
                                          layout=selected_layout if d == 0 else regular_layout, 
                                          button_color='darkgrey' if d == 0 else 'black'))
  for a in range(gui.n_axes):
    guiAxisMenus.append(widgets.Dropdown(options=[((str(i) + "(" + str(shape[i]) + ")"), i) for i in range(len(shape))],
                                         description=str("x" if a == 0 else "y" if a == 1 else "z") + " axis:",
                                         value=len(shape) - gui.n_axes + a,
                                         layout=regular_layout))
  
  label = widgets.Label(value = "")
  shape_label = widgets.Label(value = "shape: " + str(list(all_data[0].shape)))
  box_layout = widgets.Layout(display='flex', align_items='center', justify_content='flex-start', align_content='space-around')
  play_box = widgets.HBox([play] + hueSliders)
  graph_button = widgets.Button(decription='lines', font_weight='normal', layout=regular_layout, button_color='black')
  dim_widgets = [slider, widgets.HBox(playAxisButtons), shape_label, widgets.HBox(guiAxisMenus), graph_button]
  dim_box = widgets.HBox(dim_widgets, layout=box_layout)
  
  scene = gfx.Scene()
  grid = WorldGrid(state, gui, label_maps)
  camera = gfx.OrthographicCamera(maintain_aspect=False)
  aspect_ratio = 2.0 if (gui.name == "lines" or gui.name == "contour") else max(0.5, min(2.0, shape[-2]/shape[-1]))
  canvas = WgpuCanvas(size=(500 * aspect_ratio, 500 / aspect_ratio))
  renderer = gfx.WgpuRenderer(canvas)
  controller = gfx.PanZoomController(camera, register_events=renderer)
    
  def set_scene():
    nonlocal grid
    scene.clear()
    grid = WorldGrid(state, gui, label_maps)
    scene.add(gfx.Background(None, gfx.BackgroundMaterial("#000"), name="background"), grid.grid, *grid.ruler)
    # if gui.name != "contour": scene.add(grid.grid)
    
  camera_data = None
  
  def set_axes(axes, indices=None):
    assert len(axes) == gui.n_axes
    nonlocal state, renderer, camera, controller, camera_data
    state = state.with_axes(axes)
    camera_data = gui.setup_camera(state, camera, canvas)
    # ranges, lengths = [[0,state.shape[-1]], [state.min, state.max]], [state.shape[-1], state.max - state.min]
    ranges, lengths = gui.get_gui_ranges(state), gui.get_gui_lengths(state)
    # state = state.with_axes(axes)
    for d in range(len(shape)):
      if d in axes: playAxisButtons[d].description = "x" if axes[0] == d else "y" if axes[1] == d else "z"
      else: playAxisButtons[d].description = str(state.indices[d])
      playAxisButtons[d].style = widgets.ButtonStyle(button_color='grey') if d in axes else widgets.ButtonStyle()
      playAxisButtons[d].layout = selected_layout if d == state.play_axis else regular_layout
      playAxisButtons[d].disabled = True if d in axes else False
    # camera_data = gui.setup_camera(state, camera)

    
    
    renderer = gfx.WgpuRenderer(canvas)
    controller = gfx.PanZoomController(camera, register_events=renderer)
    # print("showing rect", ranges[0][0] - lengths[0]/8, ranges[0][1] + lengths[0]/20, ranges[1][0] - lengths[1]/8, ranges[1][1] + lengths[1]/20)
    
    camera.show_rect(ranges[0][0] - lengths[0]/8, ranges[0][1] + lengths[0]/20, ranges[1][0] - lengths[1]/8, ranges[1][1] + lengths[1]/8)
    set_scene()
    
  set_axes(state.axes)
  
  all_gui_data = None

  def animate():
    grid.update(canvas, camera, renderer)
    gui_data = state.get_gui_frame(all_gui_data)
    to_add = gui.get_renderables(state, gui_data, camera_data, get_points=get_points, get_message=get_message)
    if add_more_renderables is not None: to_add = add_more_renderables(state, gui_data, camera_data, to_add)
    scene.add(*to_add)
    renderer.render(scene, camera)
    scene.remove(*to_add)
  
  def handle_play_change(change):
    state.set_play_dim(change.new)
    playAxisButtons[state.play_axis].description = str(change.new)
    if descriptor is not None: label.value = descriptor(state.indices)
    canvas.request_draw(animate)
    
  def update_animation():
    nonlocal all_gui_data
    state["brightness"] = hueSliders[0].value
    state["contrast"] = hueSliders[1].value
    all_gui_data = gui.transform_data(state, all_data)
    canvas.request_draw(animate)
    
  def graph_button_clicked(change):
    nonlocal camera_data
    # state["gui_type"] = "fourier" if state["gui_type"] == "spacetime" else "radial" if state["gui_type"] == "fourier" else "spacetime"
    state["gui_type"] = "fourier" if state["gui_type"] == "spacetime" else "spacetime"
    graph_button.description = state["gui_type"]
    set_scene()
    update_animation()
    if gui.name == "contour": camera_data = gui.setup_camera(state, camera, canvas)
    
  play.observe(handle_play_change, 'value')
  
  def set_play_axis(axis):
    play.unobserve(handle_play_change, 'value')
    playAxisButtons[state.play_axis].font_weight = "normal"
    playAxisButtons[state.play_axis].layout = regular_layout
    state.set_play_axis(axis)
    playAxisButtons[axis].font_weight = "bold"
    playAxisButtons[axis].layout = selected_layout
    play.max = slider.max = shape[axis] - 1
    play.value = slider.value = max(0, state.indices[axis])
    play.interval = time_in_seconds*1000/shape[axis]
    play.observe(handle_play_change, 'value')
    playAxisButtons[axis].blur()
    slider.focus()
    update_animation()
    
  set_play_axis(start_play_axis)
      
  def set_gui_axis(new_gui_axis, new_axis):
    # nonlocal state, guiAxisMenus
    axes = list(state.axes)
    #### NO OP - AXES REMAIN THE SAME ####
    if axes[new_gui_axis] == new_axis: return
    #### SWAPPING THE PLAY AXIS WITH A GUI AXIS ####
    if new_axis == state.play_axis:
      for i in range(len(shape)):
        if not i in axes and i != new_axis:
          set_play_axis(i)
          break
    #### SWAPPING TWO OF THE GUI AXES ####
    elif new_axis in axes:
      for old_gui_axis in range(len(axes)):
        if axes[old_gui_axis] == new_axis:
          guiAxisMenus[old_gui_axis].value = axes[new_gui_axis]
          axes[old_gui_axis] = axes[new_gui_axis]
          break
    
    axes[new_gui_axis] = new_axis
    set_axes(axes)
  
  class PlayAxisChanger:
    def __init__(self, idx): self.idx = idx
    def __call__(self, change): 
      set_play_axis(self.idx)
  button_callbacks = [PlayAxisChanger(i) for i in range(len(playAxisButtons))]
  for i in range(len(playAxisButtons)): playAxisButtons[i].on_click(button_callbacks[i])
    
  class GuiAxisChanger:
    def __init__(self, idx): self.idx = idx
    def __call__(self, change):
      set_gui_axis(self.idx, change.new)
      update_animation()
    
  gui_axis_callbacks = [GuiAxisChanger(i) for i in range(len(guiAxisMenus))]
  for i in range(len(guiAxisMenus)):
    guiAxisMenus[i].observe(gui_axis_callbacks[i], names='value')
    
  graph_button.on_click(graph_button_clicked)
  graph_button.description = state["gui_type"]
    
  for obj in hueSliders: obj.observe(lambda _ : update_animation(), 'value')
  display(HTML('''<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"> '''))
  display(play_box)
  display(dim_box)
  display(label)
  
  run()

def draw_vector(vectors: torch.Tensor, **kwargs): draw_data(vectors, Lines, **kwargs)
def draw_matrix(mats: torch.Tensor, **kwargs): draw_data(mats, Heatmap, **kwargs)
def draw_hilbert(mats: torch.Tensor, **kwargs): draw_data(mats, Hilbert, **kwargs)
def draw_phases(mats: torch.Tensor, **kwargs): draw_data(mats, KeyPhases, **kwargs)
def draw_phase(mats: torch.Tensor, **kwargs): draw_data(mats, KeyPhase, **kwargs)
def draw_contour(mats: torch.Tensor, **kwargs): draw_data(mats, Contour, **kwargs)