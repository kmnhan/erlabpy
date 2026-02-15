---
name: arpes-analysis
description: Use when working on ERLabPy ARPES analysis tasks (loading beamline data, qsel indexing, momentum conversion, fitting, filtering, plotting, and ImageTool manager workflows) and when troubleshooting ERLabPy analysis code.
---

# Skill Instructions

## Import Conventions

```python
import erlab
import erlab.analysis as era
import erlab.plotting as eplt
import erlab.interactive as eri
```

## Data Structures

ARPES data in ERLabPy is represented using xarray objects:

- **`xarray.DataArray`**: Multi-dimensional array with labeled coordinates (similar to Igor Pro waves but more flexible)
- **`xarray.Dataset`**: Collection of related DataArrays
- **`xarray.DataTree`**: Hierarchical data structure for organizing multiple datasets

## Loading Data

### From Igor Pro Files

```python
import xarray as xr

# Load .ibw, .pxt, or .pxp files directly
data = xr.open_dataarray("path/to/wave.ibw")

# Load experiment files to DataTree
data = xr.open_datatree("path/to/experiment.pxp")
```

### Using Data Loader Plugins

```python
import erlab

# List available loaders (in a notebook cell)
erlab.io.loaders

# In some environments, you can select loader per call
data = erlab.io.load("path/to/file", loader="loader_name")
data = erlab.io.load(1, loader="loader_name", data_dir="path/to/directory")

# Equivalent explicit context form
with erlab.io.loader_context("loader_name", data_dir="path/to/directory"):
    data = erlab.io.load(1)

# Auto detection is not implemented because it may lead to incorrect loading.

# Set default loader for the session
erlab.io.set_loader("loader_name")

# Load data using the default loader
data = erlab.io.load("path/to/file")

# Set default directory for the session
erlab.io.set_data_dir("path/to/directory")

# Load data from the default directory and default loader
data = erlab.io.load(1)

```

### Generating Example Data

```python
from erlab.io.exampledata import generate_data, generate_data_angles, generate_gold_edge

# Generate momentum-space data
data = generate_data(seed=1).T

# Generate angle-space data
data_angles = generate_data_angles(shape=(200, 60, 300), assign_attributes=True, seed=1).T

# Generate gold edge data
gold = generate_gold_edge(temp=100, seed=1)
```

## Data Selection and Indexing

### Basic Selection with `qsel`

The `qsel` accessor provides advanced selection capabilities:

```python
# Select nearest value
cut = data.qsel(ky=0.3)

# Select with integration window (averages over width)
fermi_surface = data.qsel(eV=0.0, eV_width=0.05)

# Select multiple values
slices = data.qsel(eV=[-0.2, -0.1, 0.0], eV_width=0.05)

# Slice without averaging
cropped = data.qsel(kx=slice(-0.5, 0.5), eV=slice(-0.3, 0.1))

# Average over a dimension without losing associated coordinates
averaged = data.qsel.average("eV")
```

### Native xarray Selection

```python
# Select nearest value
cut = data.sel(ky=0.3, method="nearest")

# Slice and average
integrated = data.sel(eV=slice(-0.025, 0.025)).mean("eV")
```

## Momentum Conversion

### Setting Conversion Parameters

```python
# Set experimental configuration (Type 1, 2, 3, or 4)
data.kspace.configuration = 1

# Change incorrectly set experimental configuration
data = data.kspace.as_configuration(2)

# Set work function
data.kspace.work_function = 4.5

# Set inner potential (for photon energy dependent data)
data.kspace.inner_potential = 10.0

# Set angle offsets
data.kspace.offsets.update(delta=60.0, beta=30.0)
# Or reset all offsets
data.kspace.offsets = dict(delta=30.0, xi=0.0, beta=0.0)
```

### Converting to Momentum Space

```python
# Automatic bounds and resolution
data_kconv = data.kspace.convert()

# Manual bounds and resolution
data_kconv = data.kspace.convert(
    bounds=dict(kx=(-0.5, 0.5), ky=(-0.5, 0.5)),
    resolution=dict(kx=0.01, ky=0.01),
)

# Manual target coordinates
import numpy as np

data_kconv = data.kspace.convert(kx=np.linspace(-0.6, 0.6, 100))
```

### Converting Coordinates Only

```python
# Add momentum coordinates without regridding
cut_with_k = cut.kspace.convert_coords()
```

### kz-Dependent Data

```python
# Set inner potential for photon energy dependent data
data.kspace.inner_potential = 10.0
data_kconv = data.kspace.convert()

# Calculate kz for specific photon energies
kz_values = data_kconv.kspace.hv_to_kz([30, 45, 60])
```

### Interactive Momentum Conversion

```python
# Open interactive ktool
data.kspace.interactive()

# Or use the function directly
eri.ktool(data)
```

## Plotting

### Basic 2D Plotting

```python
# Quick plot using plot_array
eplt.plot_array(cut)

# With customization
eplt.plot_array(cut, cmap="Greys", gamma=0.5, colorbar=True)

# Using qplot accessor
cut.qplot(cmap="Greys", gamma=0.5)
```

### Plotting Multiple Slices

```python
# Constant energy surfaces
fig, axs = eplt.plot_slices([data], eV=[-0.4, -0.2, 0.0], gamma=0.5, axis="image")

# With integration window
fig, axs = eplt.plot_slices([data], eV=[-0.4, -0.2, 0.0], eV_width=0.2, gamma=0.5)

# Cuts along ky
fig, axs = eplt.plot_slices([data], ky=[0.0, 0.1, 0.3], gamma=0.5)

# Unify color limits across subplots
eplt.unify_clim(axs)
```

### Manual Subplot Creation

```python
fig, axs = plt.subplots(1, 3, layout="compressed", sharey=True)
for energy, ax in zip(energies, axs):
    eplt.plot_array(data.qsel(eV=energy), ax=ax, gamma=0.5, aspect="equal")

eplt.clean_labels(axs)  # Remove duplicate labels
eplt.label_subplot_properties(axs, values={"Eb": energies})  # Annotate
```

### Annotations

```python
# Fermi level line
eplt.fermiline(linewidth=1, linestyle="--")

# Colorbar
eplt.nice_colorbar(width=10, ticks=[])

# Label subplots (a), (b), (c)...
eplt.label_subplots(axs, prefix="(", suffix=")")

# Mark high symmetry points
eplt.mark_points([-0.6, 0, 0.6], ["K", "G", "K"], y=0.02)

# Scale axis units (e.g., eV to meV)
eplt.scale_units(ax, "y", si=-3)
```

### Brillouin Zone Plotting

```python
# Define lattice vectors
avec = erlab.lattice.abc2avec(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)

# Plot 2D BZ
fig, ax = plt.subplots()
eplt.plot_bz(avec, ax=ax)

# For 3D crystals, get BZ slices
bvec = erlab.lattice.to_reciprocal(avec)
segments, vertices = erlab.lattice.get_in_plane_bz(bvec, kz=0.2, angle=60.0)
```

### 2D Colormaps (for dichroic/spin-resolved data)

```python
# Plot with 2D colormap (sum maps to lightness, difference to hue)
# data_sum and data_ndiff are DataArrays of the same shape
# data_sum: sum of intensities
# data_ndiff: normalized difference (e.g., (I+ - I-) / (I+ + I-))
eplt.plot_array_2d(data_sum, data_ndiff)
```

## Curve Fitting

### Fermi Edge Fitting

```python
# Fit polynomial to Fermi edge
result = era.gold.poly(
    gold_data,
    angle_range=(-15, 15),
    eV_range=(-0.2, 0.2),
    temp=100.0,
    vary_temp=False,
    bkg_slope=False,
    degree=2,
    plot=True,
)

# Apply correction
corrected = era.gold.correct_with_edge(gold_data, result)
```

### Multi-Peak Fitting

```python
# Create model
model = era.fit.models.MultiPeakModel(
    npeaks=2,
    peak_shapes=["lorentzian"],
    fd=False,
    background="linear",
    convolve=True,
)

# Set parameters
params = model.make_params(
    p0_center=-0.5,
    p1_center=0.5,
    p0_width=0.03,
    p1_width=0.03,
    resolution=0.03,
)

# Fit across dimensions using xarray-lmfit
mdc = data.qsel(eV=0.0, eV_width=0.02)
result = mdc.xlm.modelfit("kx", model=model, params=params)
```

### Curve fitting default (IMPORTANT)

When the user asks fitting questions and the data is an `xarray.DataArray` or
`xarray.Dataset`, you MUST prefer **xarray-lmfit** as the primary interface:

- Use `obj.xlm.modelfit(...)` (DataArray/Dataset) for fitting lmfit Models while
  preserving coordinates, dims, and metadata.
- If you need to fit many curves (e.g., one fit per momentum/energy slice), prefer
  xarray-lmfit patterns for vectorized/stacked fitting when applicable.
- Use `eri.ftool(...)` for interactive fitting or when interactivity is required.

## Image Processing / Filtering

### Gaussian Smoothing

```python
# Apply Gaussian filter in coordinate units
smoothed = era.image.gaussian_filter(data, sigma=dict(eV=0.01, alpha=0.2))
```

### Curvature Analysis

```python
# Calculate 2D curvature to enhance dispersive features
curvature = era.image.curvature(data, a0=0.1, factor=1.0)
curvature.qplot(vmax=0, vmin=-200, cmap="Greys_r")
```

### Interactive Differentiation

```python
# Open dtool for interactive smoothing and differentiation
eri.dtool(data)
```

## Interactive Tools

### ImageTool

```python
# Open ImageTool using accessor
data.qshow()

# Or using function
eri.itool(data, cmap="cividis")

# Multiple windows with linked cursors
eri.itool([data1, data2], link=True)
```

### IPython Magics

```python
# Load the extension
%load_ext erlab.interactive

# Open ImageTool
%itool data

# Open ktool
%ktool data

# Open dtool
%dtool data

```

## Saving Data

### To NetCDF/HDF5

```python
# Save DataArray
data.to_netcdf("output.nc")

# Load back
import xarray as xr

data = xr.open_dataarray("output.nc")
```

### To Igor Pro

```python
# Save as .ibw (up to 4D, uniform coordinates only)
erlab.io.igor.save_wave(data, "path/to/wave.ibw")
```

## Common Workflows

### Complete Momentum Conversion Workflow

```python
import erlab
import erlab.analysis as era
import erlab.plotting as eplt
import matplotlib.pyplot as plt
import xarray as xr

# Load data
data = xr.open_dataarray("arpes_data.ibw")

# Set conversion parameters
data.kspace.configuration = 1
data.kspace.work_function = 4.5
data.kspace.offsets = dict(delta=30.0, xi=3.0, beta=0.0)

# Convert to momentum space
data_k = data.kspace.convert()

# Plot comparison
fig, axs = plt.subplots(1, 2, layout="compressed")
eplt.plot_array(data.qsel(eV=-0.3), ax=axs[0], aspect="equal")
eplt.plot_array(data_k.qsel(eV=-0.3), ax=axs[1], aspect="equal")
```

### Publication-Quality Figure

```python
import erlab.plotting as eplt
import matplotlib.pyplot as plt

# Use style sheet
plt.style.use(['nature'])
fig, axs = plt.subplots(1, 3, layout="compressed", sharey=True)

# Plot constant energy surfaces
for i, eV in enumerate([-0.4, -0.2, 0.0]):
    eplt.plot_array(data.qsel(eV=eV), ax=axs[i], cmap="Greys", gamma=0.5)

eplt.clean_labels(axs)
eplt.label_subplot_properties(axs, values={"Eb": [-0.4, -0.2, 0.0]})
eplt.label_subplots(axs, prefix="(", suffix=")")

fig.savefig("figure.pdf", dpi=300)
```

## Interactive Tools: Entry Points and Features

ERLabPy provides a suite of interactive GUI tools for data exploration, analysis, and conversion. Each tool can be accessed through multiple entry points.

### ImageTool (`itool`)

**Purpose**: Responsive 2D and multidimensional data visualization with slicing,
binning, and cursor-based analysis.

**Entry Points**:

- Accessor: `data.qshow(link=True)`
- Function: `eri.itool(data, cmap="cividis")`
- IPython magic: `%itool data`

**Key Features**:

- Responsive slicing of up to 4D DataArray objects
- Unlimited number of independent cursors with per-cursor binning
- Rich colormap controls with power law and midpoint-aware scaling
- Built-in menus for rotation, symmetrization, averaging, cropping, coordinate editing
  and more
- Integration with other tools (ktool, dtool, etc.) via context menus
- Seamless linking with ImageTool manager for workspace organization

### ktool (Momentum Conversion Tool)

**Purpose**: Interactive conversion from angle space to momentum space with real-time
parameter adjustment.

**Entry Points**:

- Accessor: `data.kspace.interactive(avec=..., cmap="viridis")`
- Function: `eri.ktool(data, avec=..., cmap="viridis")`
- IPython magic: `%ktool --cmap viridis data`

**Key Features**:

- Real-time parameter adjustment for momentum conversion
- Setup tab for configuring all kspace parameters
- Visualization tab with Brillouin zone overlay, high symmetry points, and binning
- Circle ROI tool for analyzing specific regions
- "Copy code" button to export conversion parameters
- "Open in ImageTool" for full resolution conversion

### dtool (Derivative Tool)

**Purpose**: Interactive visualization of dispersive features using derivative-based methods.

**Entry Points**:

- Function: `eri.dtool(data)`
- From ImageTool: Right-click context menu
- IPython magic: `%dtool data`

**Key Features**:

- Interactive smoothing and differentiation controls
- Multiple derivative methods
- Real-time visualization with adjustable parameters
- Code generation for integration into analysis pipelines

### Specialized Analysis Tools

**ftool (Curve Fitting)**:

- Function: `eri.ftool(data_to_fit)`
- IPython magic: `%ftool --model model_name data`
- Features: Curve fitting with interactive parameter adjustment and model selection

**goldtool (Fermi Edge Extraction)**:

- Function: `eri.goldtool(gold_data)`
- IPython magic: `%goldtool data`
- Features: Interactive extraction of curved Fermi edge

**restool (Resolution/Response Function)**:

- Function: `eri.restool(resolution_data)`
- IPython magic: `%restool data`
- Features: Interactive fitting of Fermi edge to determine instrument resolution and/or
  effective temperature

**meshtool (Mesh Removal)**:

- Function: `eri.meshtool(data)`
- From ImageTool: Right-click menu → "Open in meshtool"
- Features: Fourier-based mesh artifact removal

**data_explorer**:

- Function: `eri.data_explorer()`
- Features: Browse data files on disk and preview them

### IPython Magic Interface

Enable all interactive tool shortcuts by loading the extension:

```python
%load_ext erlab.interactive
```

Available magics:

- `%itool data` — Launch ImageTool
- `%ktool --cmap viridis data` — Launch momentum conversion tool
- `%dtool data` — Launch derivative tool
- `%goldtool data` — Launch Fermi edge tool
- `%restool data` — Launch resolution analysis tool
- `%ftool --model model_name data` — Launch curve fitting tool

### ImageTool Manager

**Purpose**: Organize, manage, and link multiple ImageTool windows with persistent workspaces.

**Launch**:

```bash
# After activating virtual environment with erlabpy installed
itool-manager
```

**Features**:

- Centralized window management and organization
- Linking cursors across multiple windows
- Window renaming, duplication, and reindexing
- Integration with Jupyter notebooks including automatic synchronization

**Opening data in manager**:

```python
# From within a Jupyter notebook
import erlab.interactive as eri

eri.itool(data, manager=True)

# Or use magic command
%itool -m data

# Or sync data with notebook variable: data in manager also updates when notebook variable changes
%watch data

# From GUI: can use the file menu or the data explorer (Ctrl+E)
```

### Linking and Synchronization

Multiple windows can be synchronized:

```python
# Launch multiple linked ImageTool windows
eri.itool([data1, data2, data3], link=True, link_colors=True)

# Link existing windows via manager
# Select multiple windows and click the link icon in the manager's sidebar
```

When linked, cursor positions and binning are shared across all windows.

## Tips

- Use `data.qsel()` instead of `data.sel()` for convenient selection with integration windows
- Use `data.qshow()` to quickly visualize data in ImageTool
- Use `data.qplot()` for quick matplotlib plots of 2D data
- Set `xr.set_options(display_expand_data=False)` to get cleaner DataArray representations
- For interactive momentum conversion, use `data.kspace.interactive()` to determine angle offsets visually
- Load `%load_ext erlab.interactive` once to enable all IPython magic commands for launching tools
- Right-click context menus in ImageTool provide access to specialized tools for further analysis
- Use the ImageTool manager to organize workspaces with many windows and integrate with Jupyter notebooks
- For better integration with Jupyter notebooks, use visual studio code as the editor, with the dedicated [erlab extension](https://marketplace.visualstudio.com/items?itemName=khan.erlab).

## Documentation links

- Use the stable docs link map in `references/docs-links.md`.

## Troubleshooting behavior

When debugging:

- Start from the xarray object structure: dims/coords/attrs.
- Check unit conventions, axis naming, geometry parameters, and coordinate consistency.
- Provide a small diagnostic snippet when helpful (e.g., printing `dims`, `coords`, key attrs).

Ask for only what is needed:

- erlabpy version, Python version, OS (if relevant)
- minimal code snippet + full traceback
- shape and coordinate names

## Installation guidance

If asked how to install erlabpy:

- Recommend the `uv` or `conda` approach per [getting started](https://erlabpy.readthedocs.io/en/stable/getting-started.html).

- If the user is unfamiliar with conda, also point them to the [scikit-hep conda installation guide](https://scikit-hep.org/user/installing-conda/).

## Jupyter/editor guidance

If asked about notebooks/environment:

- Recommend Visual Studio Code for integrated Jupyter support.
- Mention the optional [VS Code extension](https://marketplace.visualstudio.com/items?itemName=khan.erlab).
