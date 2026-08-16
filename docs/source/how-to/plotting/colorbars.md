(how-to-plotting-add-colorbar)=
(how-to-python-add-colorbar)=

# Colorbars

## Python

Keep the image artist returned by {func}`erlab.plotting.plot_array`. Pass that artist
to {func}`erlab.plotting.nice_colorbar` so the colorbar uses the same normalization:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

fig, ax = plt.subplots(figsize=(3.4, 2.1), layout="compressed")
image = eplt.plot_array(data, ax=ax, cmap="Greys", gamma=0.5)
colorbar = eplt.nice_colorbar(
    ax=ax,
    mappable=image,
    width=10,
    minmax=True,
)
```

```{eval-rst}
.. plot:: how_to/plotting.py add_intensity_colorbar
   :include-source: false
   :alt: Energy-momentum intensity plot with a narrow colorbar on its right side
```

Use `minmax=True` when the colorbar only needs the displayed range endpoints. Supply
explicit ticks when readers must compare intermediate intensity values. Confirm that
the colorbar limits match the plotted image before export.

See {func}`erlab.plotting.nice_colorbar` for colorbar orientation, ticks, and placement.

## Figure Composer

1. **Layout:** In {guilabel}`Layout`, use one axes.
2. **Sources:** In {guilabel}`Sources`, choose {guilabel}`Add…` and add the
   two-dimensional `data` array.
3. **Recipe:** Add an {guilabel}`Image Plot` step that targets the axes. Set
   {guilabel}`Image data` to `data`.
4. Add an {guilabel}`ERLab Method` step after the image. Select `nice_colorbar`, target
   the same axes, and set {guilabel}`Min/max ticks` to {guilabel}`True`.
