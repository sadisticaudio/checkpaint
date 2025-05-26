import torch
import numpy as np
from pylinalg import vec_transform, vec_unproject
import ipywidgets as widgets
from IPython.display import display, HTML
from checkpaint.utils import *
import numbers
from .gui import *
from pythreejs import *
import time
import inspect

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
    
class AxisState:
  def __init__(self, tensors, gui, start_idx=[0], scale_factor=1):
    self.num_tensors = len(tensors)
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
    self.colors = ["cyan", "magenta", "red", "orange", "#faa", "#6a6", "#99e", "#774", "#747", "#477"]
    self.rendereables = self.gui.create_rendereables(self)
    # self.array_idx = tuple([self.indices[i] if i > -1 else slice(None) for i in range(N + 1)])
        
  def __getitem__(self, index): return self.data[index]
  def __setitem__(self, index, value): self.data[index] = value
  def set_play_dim(self, d: int): self.indices[self.play_axis] = d
  def get_play_dim(self): return self.indices[self.play_axis]
  def set_play_axis(self, a: int):
    self.play_axis = a
    self.indices[a] = max(0, self.indices[a])
  def get_gui_dim_sizes(self): return [self.shape[a] for a in self.axes]
    
  def change_axes(self, axes):
    for a in range(len(axes)):
      for i in range(len(self.indices)):
        if self.indices[i] == -self.n_axes + a:
          self.indices[i] = 0
      self.indices[axes[a]] = -self.n_axes + a
    self.axes = axes
    self.rendereables = self.gui.create_rendereables(self)
  
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
  
  def get_fast_gui_frame(self, permuted_data):
    idx = [slice(None)] + [x if x >= 0 else slice(None) for x in self.indices]
    print("fast frame idx", idx)
    return permuted_data[tuple(idx)]
  
  def fft1d(self, x):
    X = ((x/np.sqrt(x.shape[-1]))/(np.sqrt(2)/2)) @ self.bases[x.shape[-1]].transpose(-2,-1)
    X[...,0] /= 2
    return X
  def fft2d(self, x): return torch.fft.rfft2(x)[...,:x.shape[-2]//2+1,:].abs()
  
def get_major_minor_ticks(x_min, x_max, *, fourier=False):
  if fourier: 
    x_max /= 2
  log_len = np.log10(x_max - x_min)
  log_mod = log_len % 1
  major_tick = np.power(10.0,round(log_len))/(10 if fourier else 5 if log_mod > 0.2 else 2)
  minor_tick = major_tick / (5)
  major_start, minor_start = np.ceil(x_min / major_tick) * major_tick, np.ceil(x_min / minor_tick) * minor_tick
  major_stop, minor_stop = np.ceil(x_max / major_tick) * major_tick, np.ceil(x_max / minor_tick) * minor_tick
  num_major, num_minor = round((major_stop - major_start)/major_tick), round((minor_stop - minor_start)/minor_tick)
  major_ticks = [major_start + i * major_tick * (2 if fourier else 1) for i in range(num_major)]
  minor_ticks = [minor_start + i * minor_tick * (2 if fourier else 1) for i in range(num_minor)]
  # print("min", x_min, "max", x_max, "log_len", log_len, "fourier", fourier, "major_tick", major_tick, "log %", log_len % 1)
  return major_ticks, minor_ticks

def create_custom_grid(x_min, x_max, y_min, y_max, screen_width, screen_height, *, fourier=False):
  data_range_x, data_range_y = x_max - x_min, y_max - y_min
  pixel_size_x, pixel_size_y = data_range_x/screen_width, data_range_y/screen_height
  # if data_range_y < 1:
  #   pixel_size_x /= data_range_y
  #   pixel_size_y /= data_range_y

  major_ticks_x, minor_ticks_x = get_major_minor_ticks(x_min, x_max, fourier=fourier)
  major_ticks_y, minor_ticks_y = get_major_minor_ticks(y_min, y_max)
  
  major_lines_x = np.array([[[x, y_min, 0], [x, y_max, 0]] for x in major_ticks_x], dtype=np.float32)
  minor_lines_x = np.array([[[x, y_min, 0], [x, y_max, 0]] for x in minor_ticks_x], dtype=np.float32)
  major_lines_y = np.array([[[x_min, y, 0], [x_max, y, 0]] for y in major_ticks_y], dtype=np.float32)
  minor_lines_y = np.array([[[x_min, y, 0], [x_max, y, 0]] for y in minor_ticks_y], dtype=np.float32)
  line_x_0 = np.array([[[0, y_min, 0], [0, y_max, 0]]], dtype=np.float32)
  line_y_0 = np.array([[[x_min, 0, 0], [x_max, 0, 0]]], dtype=np.float32)
  
  lines = Group()
  lines.add(LineSegments2(LineSegmentsGeometry(positions=major_lines_x), LineMaterial(linewidth=2, color='#999')))
  lines.add(LineSegments2(LineSegmentsGeometry(positions=minor_lines_x), LineMaterial(linewidth=1, color='#555')))
  lines.add(LineSegments2(LineSegmentsGeometry(positions=major_lines_y), LineMaterial(linewidth=2, color='#999')))
  lines.add(LineSegments2(LineSegmentsGeometry(positions=minor_lines_y), LineMaterial(linewidth=1, color='#555')))
  lines.add(LineSegments2(LineSegmentsGeometry(positions=line_x_0), LineMaterial(linewidth=3, color='#bbb')))
  lines.add(LineSegments2(LineSegmentsGeometry(positions=line_y_0), LineMaterial(linewidth=3, color='#bbb')))
  def get_label_string(x, regime):
    if regime == "integer": return str(round(x))
    elif regime == "scientific": 
      sci = "{:.1e}".format(x)
      if sci[-2] == '0': sci = sci[:-2] + sci[-1:]
      # print("sci", sci)
      return "bb"#sci.replace(".0", "")
    else: return str(round(x, 2))
  def get_tick_labels(ticks, axis, *, fourier=False):
    tick_low, tick_high = ticks[0], ticks[-1]
    regime = "decimal"
    if axis == "x" and tick_low % 1 == 0 and tick_high % 1 == 0: regime = "integer"
    # elif tick_low == 0
    else:
      log_low, log_high = np.log10(np.abs(tick_low)) if tick_low != 0 else 0, np.log10(np.abs(tick_high)) if tick_high != 0 else 0
      # print("tick_low", tick_low, "tick_high", tick_high, "log_low", log_low, "log_high", log_high)
      if log_low < -3 or log_low > 3 or log_high < -3 or log_high > 3: regime = "scientific"
    labels = []
    for val in ticks:
      text = get_label_string(val//2 if fourier else val, regime)
      text_len = len(text)
      text_width, text_height = min(4,text_len) * 10 * pixel_size_x, 15 * pixel_size_y
      # if regime == "scientific": 
      # print("val", val, "text", text, "text_w", text_width, "text_h", text_height, "pxl_sz_x", pixel_size_x, "pxl_sz_y", pixel_size_y, "text_len", text_len)
      # if regime == "scientific": text_width /= 1.5
      # print("data_ranges", data_range_x, data_range_y)
      # print("text_width", text_width, "text_height", text_height, "pixel_size_x", pixel_size_x, "pixel_size_y", pixel_size_y)
      t = TextTexture(string=text,size=64, fontsize=64, fontFace="noto", width=text_len * 128, height=128, squareTexture=False)
      m = MeshBasicMaterial(map=t, transparent=True)
      g = PlaneGeometry(text_width, text_height)
      # g.position.needsUpdate
      digit = Mesh(geometry=g, material=m)
      digit.position = [val+text_width/2, y_min - text_height/2, 0] if axis == "x" else [x_min-(text_width/2 + text_height/10), val, 0]
      labels.append(digit)
    return labels
  lines.add(get_tick_labels(minor_ticks_x if len(minor_ticks_x) < 14 else major_ticks_x, "x", fourier=fourier))
  lines.add(get_tick_labels(minor_ticks_y if len(minor_ticks_y) < 14 else major_ticks_y, "y"))

  return lines
  
class WorldGrid:
  def __init__(self, state, gui, width, height, axis_map = []):
    self.style = state.style
    self.gui = gui
    self.gui_type = state["gui_type"]
    ranges, lengths = gui.get_gui_ranges(state)[:2], gui.get_gui_lengths(state)
    self.grid = create_custom_grid(ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1], width, height, fourier=self.gui_type=="fourier")
    dims = self.get_dimensionality()
    self.axis_map = tuple(axis_map[state.axes[i]] if i < min(len(axis_map), len(state.axes)) else identity_map for i in range(dims))
    while len(self.axis_map) < dims: self.axis_map.insert(0, identity_map)
    
    
  def get_dimensionality(self): return min(2,self.gui.n_gui_axes)
    
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
              full_mode=False):
  # print("builtin_fonts", get_builtin_fonts())
  # print("all_fonts", get_all_fonts())
  # printCudaMemUsage("beginning of draw_data for " + gui.name)
  if isinstance(tensors, torch.Tensor): tensors = [tensors]
  for t in tensors:
    # tprint("tensor in list min", t.min(), "max", t.max())
    while t.ndim <= gui.n_axes:
      t = t.unsqueeze(0)
  state = AxisState(tensors, gui, start_indices, scale_factor)
  state["gui_type"] = start_gui_type
  shape = state.shape
  label_maps = [] if label_maps is None else label_maps if isinstance(label_maps, list) else [label_maps]
  while len(label_maps) < len(shape): label_maps = [identity_map] + label_maps
  all_data = torch.stack(tensors)

  hueSliders = [widgets.FloatSlider(value=0.5, max=1, step=0.01, description="brightness"), widgets.FloatSlider(value=0.5, max=1, step=0.01, description="contrast")]
  play, slider = widgets.Play(min=0, max=shape[0]-1, value=0), widgets.IntSlider(min=0, max=shape[0]-1, value=0, step=1, readout=False if full_mode else True)
  # play_link = widgets.jslink((play, 'value'), (slider, 'value'))

  selected_layout = widgets.Layout(width='max-content', height='30px', border='2px solid blue')
  regular_layout = widgets.Layout(width='max-content', height='30px')
  playAxisButtons, guiAxisMenus = [], []
  for d in range(len(shape)):
    playAxisButtons.append(widgets.Button(decription='0', 
                                          font_weight='bold' if d == 0 else 'normal', 
                                          layout=selected_layout if d == 0 else regular_layout, 
                                          button_color='darkgray' if d == 0 else 'black'))
  for a in range(gui.n_axes):
    guiAxisMenus.append(widgets.Dropdown(options=[((str(i) + "(" + str(shape[i]) + ")"), i) for i in range(len(shape))],
                                         description=str("x" if a == 0 else "y" if a == 1 else "z") + " axis:",
                                         value=len(shape) - gui.n_axes + a,
                                         layout=regular_layout))
  
  shape_label = widgets.Label(value = "shape: " + str(list(all_data[0].shape)))
  box_layout = widgets.Layout(display='flex', align_items='center', justify_content='flex-start', align_content='space-around')
  graph_button = widgets.Button(decription='lines', font_weight='normal', layout=regular_layout, button_color='black')
  if full_mode:
    play_box = widgets.HBox([play] + hueSliders)
    dim_widgets = widgets.HBox([slider, widgets.HBox(playAxisButtons), shape_label, widgets.HBox(guiAxisMenus), graph_button])
    top_box = widgets.VBox([play_box, dim_widgets])
  else: 
    top_box = widgets.HBox([play, slider, shape_label, graph_button])

  # dim_box = widgets.HBox(dim_widgets, layout=box_layout)
  
  scene = Scene(background=None)
  grid, camera, renderer, camera_data, all_gui_data = None, None, None, None, None, None

  def set_scene():
    nonlocal grid, camera, renderer
    ranges, lengths = gui.get_gui_ranges(state), gui.get_gui_lengths(state)
    left, right, bottom, top = ranges[0][0], ranges[0][1], state.min, state.max
    center = left + lengths[0]/2, bottom + (state.max - state.min)/2
    # print("left", left, "right", right, "bottom", bottom, "top", top, "center", center, "scene scale", scene.scale)
    
    new_ratio = shape[-1] / shape[state.axes[0]]
    scene.scale = (new_ratio, 1.0, 1.0)
    
    camera = OrthographicCamera(left=-(lengths[0] * 1.1)/2, right=(lengths[0] * 1.1)/2, top=(lengths[1] * 1.1)/2, bottom=-(lengths[1] * 1.2)/2, near=0.1, far=100)
    camera.position = [center[0], center[1], 10]
    camera.lookAt([center[0], center[1], 0])
    grid = WorldGrid(state, gui, 800, 300, label_maps)
    scene.children = grid.grid, state.rendereables
    renderer = Renderer(camera=camera, scene=scene, width=800, height=300)
  
  def set_axes(axes):
    assert len(axes) == gui.n_axes
    nonlocal state, renderer, camera, camera_data
    state.change_axes(axes)
    # camera_data = gui.setup_camera(state, camera, canvas)

    for d in range(len(shape)):
      if d in axes: playAxisButtons[d].description = "x" if axes[0] == d else "y" if axes[1] == d else "z"
      else: playAxisButtons[d].description = str(state.indices[d])
      playAxisButtons[d].style = widgets.ButtonStyle(button_color='gray') if d in axes else widgets.ButtonStyle()
      playAxisButtons[d].layout = selected_layout if d == state.play_axis else regular_layout
      playAxisButtons[d].disabled = True if d in axes else False

    set_scene()
    
  set_axes(state.axes)

  def animate():
    if full_mode:
      gui_data = state.get_gui_frame(all_gui_data)
      gui.update_renderables(state, gui_data, camera_data, get_points=get_points, get_message=get_message)
    else:
      gui_data = state.get_fast_gui_frame(all_gui_data)
      gui.update_renderables_fast(state, gui_data)
  
  def handle_play_change(change):
    if state.get_play_dim() != change.new:
      # print("current play dim", state.get_play_dim(), "change.new", change.new)
      state.set_play_dim(change.new)
      slider.value = change.new
      playAxisButtons[state.play_axis].description = str(change.new)
      # if descriptor is not None: label.value = descriptor(state.indices)
      animate()

  play.observe(handle_play_change, 'value')
  slider.observe(lambda change: handle_play_change(change), 'value')
    
  def update_animation():
    nonlocal all_gui_data
    if full_mode:
      state["brightness"] = hueSliders[0].value
      state["contrast"] = hueSliders[1].value
      print("in_transform_data")
      all_gui_data = gui.transform_data(state, all_data)
    else: all_gui_data = gui.split_and_prepare_data(state, all_data)
    animate()
    
  def graph_button_clicked(change):
    state["gui_type"] = "fourier" if state["gui_type"] == "spacetime" else "spacetime"
    graph_button.description = state["gui_type"]
    set_scene()
    if full_mode:
      update_animation()
  
  def set_play_axis(axis):
    # play.unobserve(handle_play_change, 'value')
    playAxisButtons[state.play_axis].font_weight = "normal"
    playAxisButtons[state.play_axis].layout = regular_layout
    state.set_play_axis(axis)
    playAxisButtons[axis].font_weight = "bold"
    playAxisButtons[axis].layout = selected_layout
    play.max = slider.max = shape[axis] - 1
    play.value = slider.value = max(0, state.indices[axis])
    play.interval = time_in_seconds*1000/shape[axis]
    # play.observe(handle_play_change, 'value')
    # playAxisButtons[axis].blur()
    # slider.focus()
    update_animation()
    
  set_play_axis(start_play_axis)
      
  def set_gui_axis(new_gui_axis, new_axis):
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
  
  if full_mode:
    class PlayAxisChanger:
      def __init__(self, idx): self.idx = idx
      def __call__(self, change): 
        set_play_axis(self.idx)
    button_callbacks = [PlayAxisChanger(i) for i in range(len(playAxisButtons))]
    for i in range(len(playAxisButtons)):
      playAxisButtons[i].on_click(button_callbacks[i])
    
    class GuiAxisChanger:
      def __init__(self, idx): self.idx = idx
      def __call__(self, change):
        set_gui_axis(self.idx, change.new)
        update_animation()
      
    gui_axis_callbacks = [GuiAxisChanger(i) for i in range(len(guiAxisMenus))]
    for i in range(len(guiAxisMenus)):
      guiAxisMenus[i].observe(gui_axis_callbacks[i], names='value')
    for obj in hueSliders: obj.observe(lambda _ : update_animation(), 'value')
    
  graph_button.on_click(graph_button_clicked)
  graph_button.description = state["gui_type"]
    
  ui = widgets.VBox([top_box, renderer])
  display(ui)

def draw_vector(vectors: torch.Tensor, **kwargs): draw_data(vectors, Lines, **kwargs)