(how-to-plotting-cut-trajectories)=

# Cut trajectories

Use a cut with calculated momentum coordinates to show its trajectory on a converted
constant energy surface. Prepare `cut_path` and `constant_energy_map` with
{ref}`how-to-python-convert-coordinates-only`. Both objects must use the same
experimental configuration, work function, and normal-emission position.

## Python

Plot the calculated `kx` and `ky` coordinates over the constant energy surface:

```python
import matplotlib.pyplot as plt

import erlab.plotting as eplt

fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
eplt.plot_array(
    constant_energy_map,
    ax=ax,
    cmap="Greys",
    gamma=0.5,
    aspect="equal",
)
ax.plot(cut_path.kx, cut_path.ky, color="tab:red")
```

```{eval-rst}
.. plot:: how_to/momentum_conversion.py overlay_cut_path
   :include-source: false
   :alt: Angular cut trajectory overlaid on a converted constant energy surface
```

The line must remain inside the measured momentum coverage. A mismatch usually means
that the cut and surface use different momentum-conversion parameters.

## Figure Composer

1. Add `constant_energy_map` and `cut_path` in {guilabel}`Sources`.
2. Use one axes in {guilabel}`Layout`.
3. Add an {guilabel}`Image Plot` step for `constant_energy_map`. Set
   {guilabel}`Aspect` to `equal`.
4. Add an {guilabel}`Axes Method` step after the image. Select
   {meth}`plot <matplotlib.axes.Axes.plot>` and target the same axes.
5. Set {guilabel}`Plot data` to {guilabel}`Pick from data`.
6. For X, select `cut_path` as the {guilabel}`DataArray` and `kx` as the coordinate.
   For Y, select `cut_path` and `ky`.
7. Set the line color and style as required.
