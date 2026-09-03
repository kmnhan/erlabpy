(how-to-plotting-cut-trajectories)=

# Cut trajectories

Use a cut with calculated momentum coordinates to show its trajectory on a converted
constant energy surface. Start with angle-resolved `data` that has the required
experimental configuration, work function, and normal-emission position.

## Python

Select the cut in angle space. Calculate its momentum coordinates without interpolating
its intensity. Convert the complete dataset for the constant energy surface:

```python
binding_energy = -0.3

cut = data.qsel(beta=-10)
cut_with_momentum = cut.kspace.convert_coords()
converted = data.kspace.convert()

cut_path = cut_with_momentum.qsel(eV=binding_energy)
constant_energy_map = converted.qsel(eV=binding_energy)
```

Plot the calculated `kx` and `ky` coordinates over the converted intensity:

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
that the cut and surface use different momentum-conversion parameters. See
{ref}`how-to-python-convert-coordinates-only` for the difference between calculating
coordinates and interpolating intensity.

## Figure Composer

Start with the same angle-resolved `data` used in the Python procedure:

1. Open `data` in ImageTool Manager.
2. Choose {menuselection}`Edit --> Select Data…`. Select `beta=-10` with `qsel`, and
   set {guilabel}`Result Placement` to {guilabel}`Open Child Window`.
3. In the original ImageTool, choose {menuselection}`Edit --> Convert to kspace…`.
   Configure the conversion and open the result as a child window.
4. Convert the angular-cut child. Re-enter the same configuration, work function,
   normal-emission offsets, bounds, and resolution. Open the result as another child
   window.
5. Add the converted volume and converted cut in {guilabel}`Sources`. Give them the
   aliases `converted` and `converted_cut`.
6. Select each source in {guilabel}`Sources`. Enable the `eV` selection row, select
   `qsel`, and enter the same binding energy for both sources.
7. Use one axes in {guilabel}`Layout`.
8. Add an {guilabel}`Image Plot` step for `converted`. Set {guilabel}`Aspect` to
   `equal`.
9. Add an {guilabel}`Axes Method` step after the image. Select
   {meth}`plot <matplotlib.axes.Axes.plot>` and target the same axes.
10. Set {guilabel}`Plot data` to {guilabel}`Pick from data`. For X, select
    `converted_cut` and `kx`. For Y, select `converted_cut` and `ky`.
11. Set the line color and style as required.

ImageTool interpolates the angular cut during momentum conversion. The Python procedure
uses {meth}`convert_coords <xarray.DataArray.kspace.convert_coords>` because only the
trajectory coordinates are required.
