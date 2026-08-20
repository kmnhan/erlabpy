(how-to-plotting-combine-maps-and-cuts)=
(how-to-python-combine-maps-and-cuts)=

# Maps and cuts

## Python

Create the complete axes layout first. Pass each row to
{func}`erlab.plotting.plot_slices` through its `axes` argument:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

energies = [-0.4, -0.2, 0.0]
ky_values = [0.0, 0.1, 0.3]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(6.4, 4.0),
    layout="compressed",
    sharex="col",
    sharey="row",
)
eplt.plot_slices(
    [data],
    eV=energies,
    axes=axes[0],
    axis="image",
    gamma=0.5,
    same_limits=True,
    annotate=False,
)
eplt.plot_slices(
    [data],
    ky=ky_values,
    axes=axes[1],
    gamma=0.5,
    same_limits=True,
    annotate=False,
)
eplt.label_subplot_properties(axes[0], values={"eV": energies})
eplt.label_subplot_properties(axes[1], values={"ky": ky_values})
eplt.clean_labels(axes)
```

```{eval-rst}
.. plot:: how_to/plotting.py combine_maps_and_cuts
   :include-source: false
   :alt: Two-row ARPES figure with three constant energy maps above three energy–momentum cuts
```

The number of axes in each row must match the number of requested slices. Use shared
limits within a row when intensity differences between its panels must remain visible.
Use independent limits when the task is only to compare feature positions.

### Reference color limits

Use one panel as the intensity reference when its color limits are suitable for all
slices:

```python
figure, axes = eplt.plot_slices(
    [data],
    ky=[0.0, 0.1, 0.3],
    gamma=0.5,
    annotate=False,
)
eplt.unify_clim(axes, target=axes.flat[1])
```

```{eval-rst}
.. plot:: how_to/plotting.py share_reference_color_limits
   :include-source: false
   :alt: Three energy-momentum cuts with color limits taken from the middle cut
```

The `target` axes supplies the color limits. Without `target`,
{func}`unify_clim <erlab.plotting.unify_clim>` uses the lowest and highest color limits
among the plotted mappables.

## Figure Composer

1. **Layout:** In {guilabel}`Layout`, set {guilabel}`Rows` to `2` and
   {guilabel}`Columns` to `3`.
2. **Sources:** In {guilabel}`Sources`, choose {guilabel}`Add…` and add the
   three-dimensional `data` array.
3. **Constant energy maps:** Add a {guilabel}`Slice Plot` step and target the three axes
   in the top row. Select `data` under {guilabel}`Inputs`. Set
   {guilabel}`Dimension` to `eV`, {guilabel}`Values` to
   {guilabel}`Manual values`, and {guilabel}`Manual` to `-0.4, -0.2, 0.0`. Set
   {guilabel}`Axis` to {guilabel}`image`, {guilabel}`Gamma` to `0.5`, and
   {guilabel}`Match limits` to {guilabel}`True`. Clear {guilabel}`Annotate`.
4. **Energy–momentum cuts:** Add a second {guilabel}`Slice Plot` step and target the
   bottom row. Select `data` under {guilabel}`Inputs`. Set {guilabel}`Dimension` to
   `ky`, {guilabel}`Values` to {guilabel}`Manual values`, and {guilabel}`Manual` to
   `0.0, 0.1, 0.3`. Set {guilabel}`Gamma` to `0.5` and
   {guilabel}`Match limits` to {guilabel}`True`. Clear {guilabel}`Annotate`.
5. **Slice labels:** Add an {guilabel}`ERLab Method` step for
   {func}`label_subplot_properties <erlab.plotting.label_subplot_properties>` on the
   top row. Set {guilabel}`Values` to
   `{"eV": [-0.4, -0.2, 0.0]}`. Add another step for the bottom row with
   `{"ky": [0.0, 0.1, 0.3]}`.
6. Add an {guilabel}`ERLab Method` step for
   {func}`clean_labels <erlab.plotting.clean_labels>` and target all six axes.
