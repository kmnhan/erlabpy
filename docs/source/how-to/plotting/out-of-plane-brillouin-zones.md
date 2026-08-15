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
2. Set {guilabel}`Layout` to a $1 \times 1$ grid. Set the size to $3.0 \times 3.0$
   inches and select the compressed layout engine.
3. Add a {guilabel}`BZ Overlay` step.
4. Under {guilabel}`Slice`, set {guilabel}`Mode` to `Out-of-plane`,
   {guilabel}`Angle` to `90`, {guilabel}`k parallel` to `0`, and
   {guilabel}`Bounds` to `-1.5, 1.5, -1.5, 1.5`.
5. Under {guilabel}`Lattice`, set $a$, $b$, and $c$ to `6`, `10`, and `25`. Set
   $\alpha$, $\beta$, and $\gamma$ to `90`. Set {guilabel}`Centering` to `F`.
6. Under {guilabel}`Style`, set {guilabel}`Line color` to `tab:purple` and
   {guilabel}`Width` to `1.5`. Under {guilabel}`Points`, enable
   {guilabel}`Vertices`.
7. Add an {guilabel}`Axes Method` step for `set_xlabel`, and set
   {guilabel}`Label` to $k_x$ (Å$^{-1}$).
8. Add an {guilabel}`Axes Method` step for `set_ylabel`, and set
   {guilabel}`Label` to $k_z$ (Å$^{-1}$).
9. Add an {guilabel}`Axes Method` step for `set_aspect`, and set
   {guilabel}`Aspect` to `equal`.

Here, `angle=90.0` fixes the in-plane momentum along $k_y$, so the horizontal axis of
the section is $k_x$. To compare the section with measured intensity, plot the
$k_x$-$k_z$ map on `ax` before the boundary. Confirm the fixed momentum, azimuthal
direction, and coordinate orientation first.

See {func}`erlab.plotting.plot_out_of_plane_bz` for the supported slice parameters.
