(how-to-plotting-high-symmetry-cuts)=

# High-symmetry cuts

Use this guide to show a selected reciprocal-space path beside its energy–momentum cut.
Start with converted `momentum_data`, `high_symmetry_vertices`, and
`high_symmetry_cut` prepared with {ref}`how-to-python-extract-path`.

## Python

Calculate the path positions of the vertices. Use them for the x-axis ticks and the
internal guide lines:

```python
import numpy as np

path_vertices = np.column_stack(
    [high_symmetry_vertices["kx"], high_symmetry_vertices["ky"]]
)
segment_lengths = np.linalg.norm(np.diff(path_vertices, axis=0), axis=1)
path_vertex_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
```

Plot the selected path and the interpolated cut:

```python
import matplotlib.pyplot as plt

import erlab.plotting as eplt

path_energy_map = momentum_data.qsel(eV=-0.2, eV_width=0.02)
fig, axes = plt.subplots(
    1,
    2,
    figsize=(6.4, 3.2),
    layout="compressed",
    gridspec_kw={"width_ratios": (1.0, 1.35)},
)

eplt.plot_array(
    path_energy_map,
    ax=axes[0],
    cmap="Greys",
    gamma=0.7,
    aspect="equal",
)
eplt.plot_hex_bz(
    a=lattice_constant,
    ax=axes[0],
    fill=False,
    edgecolor="0.35",
    linewidth=0.8,
)
axes[0].plot(
    high_symmetry_vertices["kx"],
    high_symmetry_vertices["ky"],
    color="tab:red",
    marker="o",
    markersize=3,
    linewidth=1.2,
)
axes[0].set_title(r"$E = E_F - 0.2$ eV")

eplt.plot_array(high_symmetry_cut, ax=axes[1], cmap="Greys", gamma=0.7)
eplt.fermiline(ax=axes[1], linestyle="--", linewidth=0.8)
for position in path_vertex_positions[1:-1]:
    axes[1].axvline(position, color="0.5", linestyle="--", linewidth=0.8)
axes[1].set_xticks(path_vertex_positions, labels=["Γ", "M", "K", "Γ"])
axes[1].set(
    xlabel="",
    xlim=(path_vertex_positions[0] - 0.03, path_vertex_positions[-1] + 0.03),
)
```

```{eval-rst}
.. plot:: how_to/inspection_and_selection.py plot_high_symmetry_cut
   :include-source: false
   :alt: Γ–M–K–Γ path on a constant energy surface beside the interpolated energy–momentum cut
```

The interpolation follows an ideal line. It does not average intensity over a finite
width perpendicular to the path.

## Figure Composer

Prepare `path_energy_map` and `high_symmetry_cut` with the Python code above. Add both
arrays in {guilabel}`Sources`.

1. Use a $1 \times 2$ layout.
2. Add an {guilabel}`Image Plot` step for `path_energy_map` on the left axes. Set
   {guilabel}`Aspect` to `equal`.
3. Add an {guilabel}`Image Plot` step for `high_symmetry_cut` on the right axes.

The single first-zone boundary drawn by {func}`erlab.plotting.plot_hex_bz` does not have
an editable Figure Composer step.

```{include} ../../_includes/figure-composer-planned-step.md
```

4. Add a {guilabel}`Python` step after the left image.
5. Review this code, enter it in {guilabel}`Code`, and enable {guilabel}`Trusted`:

```python
lattice_constant = 6.97
eplt.plot_hex_bz(
    a=lattice_constant,
    ax=axs[0, 0],
    fill=False,
    edgecolor="0.35",
    linewidth=0.8,
)
```

6. Add an {guilabel}`Axes Method` step after the boundary. Select
   {meth}`plot <matplotlib.axes.Axes.plot>` and target the left axes. Set
   {guilabel}`Plot data` to {guilabel}`Pick from data`. For X, select
   `high_symmetry_cut` and `kx`. For Y, select `high_symmetry_cut` and `ky`.
7. Add an {guilabel}`ERLab Method` step for
   {func}`fermiline <erlab.plotting.fermiline>` on the right axes.
8. Add one {guilabel}`Axes Method` step for each internal guide. Select
   {meth}`Vertical line <matplotlib.axes.Axes.axvline>`, target the right axes, and
   enter the corresponding value from `path_vertex_positions[1:-1]`.
9. Add an {guilabel}`Axes Method` step for
   {meth}`Set x ticks <matplotlib.axes.Axes.set_xticks>` on the right axes. Enter the
   values from `path_vertex_positions` and the labels `Γ, M, K, Γ`.
10. Add an {guilabel}`Axes Method` step for
    {meth}`set_xlabel <matplotlib.axes.Axes.set_xlabel>` on the right axes. Leave
    {guilabel}`Label` empty so that the cumulative path distance is not shown.

Replace the lattice constant and path positions with values for the measured material.
