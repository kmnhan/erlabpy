(how-to-plotting-out-of-plane-brillouin-zones)=

(how-to-python-draw-out-of-plane-bz)=

# Out-of-plane Brillouin-zone sections

## Python

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

## Figure Composer

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
7. Add an {guilabel}`Axes Method` step for `set_xlabel`, and set
   {guilabel}`Label` to $k_x$ (Å$^{-1}$).
8. Add an {guilabel}`Axes Method` step for `set_ylabel`, and set
   {guilabel}`Label` to $k_z$ (Å$^{-1}$).
9. Add an {guilabel}`Axes Method` step for `set_aspect`, and set
   {guilabel}`Aspect` to `equal`.

To compare the section with measured intensity, plot the corresponding momentum map on
the same axes before the boundary. Confirm the fixed momentum, azimuthal direction, and
coordinate orientation first.

See {func}`erlab.plotting.plot_out_of_plane_bz` for the supported slice parameters.
