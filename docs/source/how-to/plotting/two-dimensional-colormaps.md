(how-to-plotting-intensity-and-asymmetry)=

(how-to-python-plot-intensity-and-asymmetry)=

# 2D colormaps

## Python

Use a two-dimensional colormap when one array supplies the lightness and a second array
supplies the hue. For two measurements with matching coordinates:

```python
import xarray as xr

import erlab.plotting as eplt

data_a, data_b = xr.align(data_a, data_b, join="exact")
intensity = data_a + data_b
asymmetry = ((data_a - data_b) / intensity).where(intensity > 0)
```

Plot the two quantities separately first. Check where low total intensity makes the
normalized difference unstable:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 3.0),
    layout="compressed",
    sharex=True,
    sharey=True,
)
intensity_image = eplt.plot_array(
    intensity,
    ax=axes[0],
    cmap="viridis",
    aspect="equal",
)
asymmetry_image = eplt.plot_array(
    asymmetry,
    ax=axes[1],
    cmap="bwr",
    norm=eplt.CenteredPowerNorm(1.0, vcenter=0.0, halfrange=1.0),
    aspect="equal",
)
eplt.nice_colorbar(ax=axes[0], mappable=intensity_image, width=7)
eplt.nice_colorbar(ax=axes[1], mappable=asymmetry_image, width=7)
eplt.set_titles(axes, ["Total intensity", "Normalized difference"])
eplt.clean_labels(axes)
```

```{eval-rst}
.. plot:: how_to/plotting.py compare_intensity_and_asymmetry
   :include-source: false
   :alt: Total intensity and normalized difference plotted separately for the same constant energy surface
```

Map total intensity to lightness and normalized difference to hue:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="compressed")

_, colorbar = eplt.plot_array_2d(
    intensity,
    asymmetry,
    ax=ax,
    lnorm=eplt.InversePowerNorm(0.5),
    cnorm=eplt.CenteredInversePowerNorm(0.7, vcenter=0.0, halfrange=1.0),
)
colorbar.ax.set_xticks(colorbar.ax.get_xlim(), labels=["Min", "Max"])
colorbar.ax.set(xlabel="Intensity", ylabel="Asymmetry")
```

```{eval-rst}
.. plot:: how_to/plotting.py plot_intensity_and_asymmetry
   :include-source: false
   :alt: Two-dimensional colormap of measured intensity and asymmetry
```

## Figure Composer

First calculate `intensity` and `asymmetry` with the Python code above. Open both arrays
in ImageTool Manager and add them as Figure Composer sources. The separate diagnostic
plots then use editable recipe steps:

1. Set {guilabel}`Layout` to a $1 \times 2$ grid.
2. Add `intensity` and `asymmetry` in {guilabel}`Sources`.
3. Add one {guilabel}`Image Plot` step for each source and target one axes per step.
4. Set the intensity colormap to `viridis`.
5. Set the asymmetry colormap to `bwr`. Select
   {class}`CenteredPowerNorm <erlab.plotting.CenteredPowerNorm>`, and set
   {guilabel}`Gamma` to `1`. Under {guilabel}`Center/range`, set `vcenter` to `0`
   and `halfrange` to `1`.
6. Add one {guilabel}`ERLab Method` step with
   {func}`nice_colorbar <erlab.plotting.nice_colorbar>` after each image.
7. Add {func}`set_titles <erlab.plotting.set_titles>` and
   {func}`clean_labels <erlab.plotting.clean_labels>` {guilabel}`ERLab Method` steps.

The combined lightness-and-hue plot does not have an editable recipe step.

```{include} ../../_includes/figure-composer-planned-step.md
```

1. Add `data_a` and `data_b` in {guilabel}`Sources`.
2. Add a {guilabel}`Python` step to a $1 \times 1$ figure.
3. Review this code, enter it in {guilabel}`Code`, and enable {guilabel}`Trusted`:

```python
data_a, data_b = xr.align(data_a, data_b, join="exact")
intensity = data_a + data_b
asymmetry = ((data_a - data_b) / intensity).where(intensity > 0)

_, colorbar = eplt.plot_array_2d(
    intensity,
    asymmetry,
    ax=ax,
    lnorm=eplt.InversePowerNorm(0.5),
    cnorm=eplt.CenteredInversePowerNorm(0.7, vcenter=0.0, halfrange=1.0),
)
colorbar.ax.set_xticks(colorbar.ax.get_xlim(), labels=["Min", "Max"])
colorbar.ax.set(xlabel="Intensity", ylabel="Asymmetry")
```

Mask or otherwise handle points where the summed intensity is too small for a stable
normalized difference. Confirm that `data_a` and `data_b` have aligned coordinates
before calculating the asymmetry. The two-dimensional colorbar shows the independent
lightness and hue mappings.

See {func}`erlab.plotting.plot_array_2d` for color normalization and colorbar arguments.
