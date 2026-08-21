(how-to-plotting-two-dimensional-bz)=

(how-to-python-draw-two-dimensional-bz)=

# Brillouin zones

(how-to-plotting-first-brillouin-zone)=

## First Brillouin zone

### Python

Construct the real-space lattice vectors for the crystal. Pass them to
{func}`erlab.plotting.plot_bz`:

```python
import matplotlib.pyplot as plt

import erlab
import erlab.plotting as eplt

avec = erlab.lattice.abc2avec(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)

fig, ax = plt.subplots(figsize=(2.5, 2.5), layout="compressed")
eplt.plot_bz(avec, ax=ax)
ax.set(
    xlabel=r"$k_x$ (Å$^{-1}$)",
    ylabel=r"$k_y$ (Å$^{-1}$)",
    xlim=(-1.5, 1.5),
    ylim=(-1.5, 1.5),
    aspect="equal",
)
```

```{eval-rst}
.. plot:: how_to/plotting.py draw_two_dimensional_brillouin_zone
   :include-source: false
   :alt: Hexagonal first Brillouin-zone polygon on equal reciprocal-momentum axes
```

### Figure Composer

The current {guilabel}`BZ Overlay` step draws an in-plane section. It does not draw the
single first-zone polygon in this guide.

```{include} ../../_includes/figure-composer-planned-step.md
```

1. In ImageTool Manager, choose
   {menuselection}`File --> New Empty Figure`.
2. Set {guilabel}`Layout` to a $1 \times 1$ grid.
3. Add a {guilabel}`Python` step.
4. Review this code, enter it in {guilabel}`Code`, and enable {guilabel}`Trusted`:

```python
import erlab

avec = erlab.lattice.abc2avec(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)
eplt.plot_bz(avec, ax=ax)
ax.set(
    xlabel=r"$k_x$ (Å$^{-1}$)",
    ylabel=r"$k_y$ (Å$^{-1}$)",
    aspect="equal",
)
```

Replace the lattice parameters with those of the measured material. See
{func}`erlab.plotting.plot_bz` for reciprocal input, rotation, and offset arguments.

(how-to-plotting-in-plane-brillouin-zone-sections)=

(how-to-python-draw-in-plane-bz)=

## In-plane sections

### Python

Construct the conventional real-space lattice vectors. Convert the centered
conventional cell to primitive lattice vectors. Then calculate the reciprocal-lattice
vectors:

```python
import matplotlib.pyplot as plt

import erlab
import erlab.plotting as eplt

avec = erlab.lattice.abc2avec(6.0, 10.0, 25.0, 90.0, 90.0, 90.0)
avec_primitive = erlab.lattice.to_primitive(avec, centering_type="F")
bvec = erlab.lattice.to_reciprocal(avec_primitive)

fig, ax = plt.subplots(figsize=(3.0, 3.0), layout="compressed")
eplt.plot_in_plane_bz(
    bvec,
    kz=0.2,
    angle=60.0,
    bounds=(-1.5, 1.5, -1.5, 1.5),
    ax=ax,
    vertices=True,
    color="tab:purple",
    linewidth=1.5,
)
ax.set(
    xlabel=r"$k_x$ (Å$^{-1}$)",
    ylabel=r"$k_y$ (Å$^{-1}$)",
    aspect="equal",
)
```

```{eval-rst}
.. plot:: how_to/plotting.py draw_in_plane_brillouin_zone
   :include-source: false
   :alt: Constant-kz section through the Brillouin zones of a face-centered orthorhombic crystal
```

Set `kz` to the out-of-plane momentum of the measured section. Use `angle` for the
rotation about the $k_z$ axis. Set `bounds` to the required in-plane momentum window.

{func}`erlab.plotting.plot_in_plane_bz` obtains the boundary segments and vertices from
{func}`erlab.lattice.get_bz_slice`. Use {func}`~erlab.lattice.get_bz_slice` directly
for an arbitrary plane. Supply a point on the plane, its normal vector, and the bounds
in the local plane coordinates.

### Figure Composer

1. In ImageTool Manager, choose
   {menuselection}`File --> New Empty Figure`.
2. Add a {guilabel}`BZ Overlay` step.
3. Under {guilabel}`Slice`, set {guilabel}`Mode` to `In-plane`. Enter the
   {guilabel}`kz`, {guilabel}`Angle`, and {guilabel}`Bounds` for the required section.
4. Under {guilabel}`Lattice`, enter the lattice parameters and
   {guilabel}`Centering` for the sample.
5. Under {guilabel}`Style`, enable {guilabel}`Vertices` if corner markers are useful.
6. Add {guilabel}`Axes Method` steps for
   {meth}`set_xlabel <matplotlib.axes.Axes.set_xlabel>` and
   {meth}`set_ylabel <matplotlib.axes.Axes.set_ylabel>`. Enter the in-plane momentum
   labels and units.
7. Add an {guilabel}`Axes Method` step for
   {meth}`set_aspect <matplotlib.axes.Axes.set_aspect>`, and set
   {guilabel}`Aspect` to `equal`.

The {guilabel}`kz` control accepts both multiples of $\pi/c$ and Å$^{-1}$. See
{func}`erlab.plotting.plot_in_plane_bz` for the corresponding Python arguments.

(how-to-plotting-out-of-plane-brillouin-zones)=

(how-to-python-draw-out-of-plane-bz)=

## Out-of-plane sections

### Python

Construct the real-space lattice vectors and apply the crystal centering. Convert the
primitive vectors to reciprocal lattice vectors before you calculate the section:

```python
import matplotlib.pyplot as plt

import erlab
import erlab.plotting as eplt

avec = erlab.lattice.abc2avec(6.0, 10.0, 25.0, 90.0, 90.0, 90.0)
avec_primitive = erlab.lattice.to_primitive(avec, centering_type="F")
bvec = erlab.lattice.to_reciprocal(avec_primitive)

fig, ax = plt.subplots(figsize=(3.0, 3.0), layout="compressed")
eplt.plot_out_of_plane_bz(
    bvec,
    k_parallel=0.0,
    angle=90.0,
    bounds=(-1.5, 1.5, -1.5, 1.5),
    ax=ax,
    vertices=True,
    color="tab:purple",
    linewidth=1.5,
)
ax.set(
    xlabel=r"$k_x$ (Å$^{-1}$)",
    ylabel=r"$k_z$ (Å$^{-1}$)",
    aspect="equal",
)
```

```{eval-rst}
.. plot:: how_to/plotting.py draw_out_of_plane_brillouin_zone
   :include-source: false
   :alt: Out-of-plane Brillouin-zone section with vertices marked on equal momentum axes
```

### Figure Composer

1. In ImageTool Manager, choose
   {menuselection}`File --> New Empty Figure`.
2. Set {guilabel}`Layout` to a $1 \times 1$ grid.
3. Add a {guilabel}`BZ Overlay` step.
4. Under {guilabel}`Slice`, set {guilabel}`Mode` to `Out-of-plane`. Enter the
   {guilabel}`Angle`, fixed {guilabel}`k parallel`, and {guilabel}`Bounds` for the
   required section.
5. Under {guilabel}`Lattice`, enter the lattice parameters and
   {guilabel}`Centering` for the sample.
6. Under {guilabel}`Points`, enable {guilabel}`Vertices` if corner markers are useful.
7. Add an {guilabel}`Axes Method` step for
   {meth}`set_xlabel <matplotlib.axes.Axes.set_xlabel>`, and set {guilabel}`Label` to
   $k_x$ (Å$^{-1}$).
8. Add an {guilabel}`Axes Method` step for
   {meth}`set_ylabel <matplotlib.axes.Axes.set_ylabel>`, and set {guilabel}`Label` to
   $k_z$ (Å$^{-1}$).
9. Add an {guilabel}`Axes Method` step for
   {meth}`set_aspect <matplotlib.axes.Axes.set_aspect>`, and set {guilabel}`Aspect` to
   `equal`.

To compare the section with measured intensity, plot the corresponding momentum-space
intensity on the same axes before the boundary. Confirm the fixed momentum, azimuthal
direction, and coordinate orientation first.

See {func}`erlab.plotting.plot_out_of_plane_bz` for the supported slice parameters.

(how-to-plotting-brillouin-zone-overlays)=

(how-to-python-overlay-brillouin-zone)=

## Brillouin-zone overlays

### Python

For a hexagonal material, use its measured in-plane lattice constant to draw the
boundary over a constant energy surface:

```python
import matplotlib.pyplot as plt

import erlab.plotting as eplt

lattice_constant = 6.97

fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
eplt.plot_array(
    constant_energy_map,
    ax=ax,
    cmap="Greys",
    gamma=0.5,
    aspect="equal",
)
eplt.plot_hex_bz(
    a=lattice_constant,
    ax=ax,
    fill=False,
    edgecolor="tab:purple",
    linestyle="--",
    linewidth=1.2,
)
```

```{eval-rst}
.. plot:: how_to/plotting.py overlay_brillouin_zone
   :include-source: false
   :alt: Constant energy surface with an in-plane Brillouin zone boundary
```

### Figure Composer

1. Set {guilabel}`Layout` to a $1 \times 1$ grid.
2. Add `constant_energy_map` in {guilabel}`Sources`.
3. Add an {guilabel}`Image Plot` step and set {guilabel}`Aspect` to `equal`.

The current {guilabel}`BZ Overlay` step draws an in-plane section. It does not draw the
single hexagonal boundary used here.

```{include} ../../_includes/figure-composer-planned-step.md
```

4. Add a {guilabel}`Python` step after the image.
5. Review this code, enter it in {guilabel}`Code`, and enable {guilabel}`Trusted`:

```python
eplt.plot_hex_bz(
    a=6.97,
    ax=ax,
    fill=False,
)
```

Replace `a` with the in-plane lattice constant of the measured material. The boundary
must use the correct lattice parameters and orientation. Do not use the appearance of
the intensity to select or resize the boundary. See
{func}`erlab.plotting.plot_hex_bz` for rotation and offset arguments.
