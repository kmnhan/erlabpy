(how-to-plotting-display-energy-in-mev)=
(how-to-python-display-energy-in-mev)=

# Energy in meV

## Python

Plot the data in eV. Then scale the displayed tick labels and unit prefix:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

fig, ax = plt.subplots(figsize=(3.4, 2.1), layout="compressed")
cut.qplot(ax=ax)

eplt.scale_units(ax, "y", si=-3)
```

```{eval-rst}
.. plot:: how_to/plotting.py display_energy_in_mev
   :include-source: false
   :alt: Energy–momentum cut with its vertical axis displayed in millielectronvolts
```

Call {func}`erlab.plotting.scale_units` after plot commands that can replace the tick
formatter. Confirm that the axis label contains meV. The energy coordinates in `cut`
remain in eV. A custom tick formatter is not supported. Restore the default Matplotlib
scalar formatter before you apply this procedure.

## Figure Composer

1. **Layout:** In {guilabel}`Layout`, use one axes.
2. **Sources:** In {guilabel}`Sources`, choose {guilabel}`Add…` and add the
   two-dimensional `cut` array.
3. **Recipe:** Add an {guilabel}`Image Plot` step that targets the axes. Set
   {guilabel}`Image data` to `cut`.
4. Add an {guilabel}`ERLab Method` step after the image. Select
   {func}`scale_units <erlab.plotting.scale_units>`, target the same axes, set
   {guilabel}`Axis` to `y`, and set {guilabel}`SI exponent` to `-3`.
