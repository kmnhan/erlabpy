(how-to-plotting-polygon-masking)=

# Polygon masking

Use a polygon mask to retain measured intensity inside a selected momentum-space
region. This example starts with a converted three-dimensional `data` array and uses
the first Brillouin zone as the boundary.

## Python

Select the constant energy surface. Then calculate the ordered zone vertices from the
real-space lattice and apply the polygon mask:

```python
import numpy as np

import erlab
import erlab.analysis as era

binding_energy = -0.2
energy_width = 0.02
lattice_constant = 6.97

constant_energy_map = data.qsel(
    eV=binding_energy,
    eV_width=energy_width,
)
real_space_basis = lattice_constant * np.array(
    [
        [1.0, 0.0],
        [-0.5, np.sqrt(3) / 2],
    ]
)
first_bz_vertices = erlab.lattice.get_2d_vertices(
    real_space_basis,
    reciprocal=False,
    rotate=30.0,
)
masked_map = era.mask.mask_with_polygon(
    constant_energy_map,
    first_bz_vertices,
    dims=("kx", "ky"),
)
```

Plot the source map and masked result with common color limits. Close the vertex list
only for drawing the boundary. {func}`erlab.analysis.mask.mask_with_polygon` closes the
mask polygon automatically:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

closed_vertices = np.vstack([first_bz_vertices, first_bz_vertices[0]])
_, axes = plt.subplots(
    1,
    2,
    figsize=(6.4, 3.0),
    layout="compressed",
    sharex=True,
    sharey=True,
)
for ax, map_data in zip(
    axes,
    (constant_energy_map, masked_map),
    strict=True,
):
    eplt.plot_array(
        map_data,
        ax=ax,
        cmap="Greys",
        gamma=0.5,
        aspect="equal",
    )
axes[0].plot(
    closed_vertices[:, 0],
    closed_vertices[:, 1],
    color="tab:red",
)
eplt.unify_clim(axes)
eplt.clean_labels(axes)
eplt.set_titles(axes, ["First Brillouin zone", "Masked data"])
```

```{eval-rst}
.. plot:: how_to/inspection_and_selection.py mask_momentum_region
   :include-source: false
   :alt: Repeated-zone constant energy surface with the first Brillouin-zone boundary beside the intensity retained inside that zone
```

The 30° rotation aligns the Γ–M direction with $k_x$ for this model. Points outside the
first Brillouin zone become missing values. Use `invert=True` to mask the zone instead.
Use `drop=True` to remove coordinate labels for rows and columns that contain no
retained values. See {ref}`how-to-plotting-brillouin-zone-overlay` for other zone
overlays.

## Figure Composer

Start with the same three-dimensional `data` array, binding energy, energy width, and
lattice constant used in the Python procedure. Figure Composer does not calculate a
polygon mask. Create the derived array in ImageTool before you assemble the figure.

### ImageTool calculation

1. Open `data` in a managed ImageTool.
2. Choose {menuselection}`Edit --> Select Data…`. For `eV`, select `qsel`, enter
   `-0.2`, enable {guilabel}`Width`, and enter `0.02`. Set
   {guilabel}`Result Placement` to {guilabel}`Open Child Window`.
3. Calculate `first_bz_vertices` with the lattice calculation in the Python section.
4. In the constant-energy-map child, right-click the image and choose
   {guilabel}`Add Polygon ROI`.
5. Right-click the ROI and choose {guilabel}`Edit ROI…`. Enter the calculated vertices
   in the `kx` and `ky` columns in their listed order. Turn on {guilabel}`Closed`.
6. Right-click the ROI and choose {guilabel}`Mask Data with ROI`.
7. Leave {guilabel}`Invert Mask` and {guilabel}`Drop Masked Values` off. Set
   {guilabel}`Result Placement` to {guilabel}`Open Child Window`, and create the masked
   map.

For other polygon boundaries, move the ROI handles or enter different coordinates.
See {ref}`how-to-gui-mask-polygon` for the mask controls and result behavior.

### Figure assembly

1. In the constant-energy-map ImageTool, right-click the image and choose
   {guilabel}`New Figure`.
2. Set {guilabel}`Layout` to a $1 \times 2$ grid. Target the existing
   {guilabel}`Image Plot` step to the left axes. Set {guilabel}`Gamma` to `0.5` and
   {guilabel}`Aspect` to `equal`.
3. In the masked-map child, right-click the image and choose
   {guilabel}`Append to Figure`. Select the same figure and the right axes. Set the new
   {guilabel}`Image Plot` step to the same gamma and aspect.

The single polygon boundary does not have an editable Figure Composer step.

```{include} ../../_includes/figure-composer-planned-step.md
```

4. Add a {guilabel}`Python` step after the left image. Review this code, then enter it
   in {guilabel}`Code`:

```python
import numpy as np

import erlab

lattice_constant = 6.97
real_space_basis = lattice_constant * np.array(
    [
        [1.0, 0.0],
        [-0.5, np.sqrt(3) / 2],
    ]
)
first_bz_vertices = erlab.lattice.get_2d_vertices(
    real_space_basis,
    reciprocal=False,
    rotate=30.0,
)
closed_vertices = np.vstack([first_bz_vertices, first_bz_vertices[0]])
axs[0, 0].plot(
    closed_vertices[:, 0],
    closed_vertices[:, 1],
    color="tab:red",
)
```

5. Add an {guilabel}`ERLab Method` step for
   {func}`unify_clim <erlab.plotting.unify_clim>` and target both axes.
6. Add an {guilabel}`ERLab Method` step for
   {func}`clean_labels <erlab.plotting.clean_labels>` and target both axes.
7. Add an {guilabel}`ERLab Method` step for
   {func}`set_titles <erlab.plotting.set_titles>` and target both axes. Enter
   `First Brillouin zone` and `Masked data` on separate lines in {guilabel}`Text`.
