(how-to-plotting-annotate-arpes-figure)=
(how-to-python-annotate-arpes-figure)=

# ARPES cut annotations

## Python

Add the Fermi level, high-symmetry labels, and panel labels after plotting the data:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

fig, ax = plt.subplots()
eplt.plot_array(data, ax=ax, cmap="Greys", gamma=0.5)
eplt.fermiline(ax=ax, linestyle="--")
eplt.mark_points(
    [-0.6, 0.0, 0.6],
    ["K", "G", "K"],
    y=0.02,
    ax=ax,
)
eplt.label_subplots(ax, prefix="(", suffix=")")
```

```{eval-rst}
.. plot:: how_to/plotting.py annotate_arpes_figure
   :include-source: false
   :alt: ARPES cut with a Fermi-level line, momentum labels, and a panel label
```

Replace the positions and labels with the measured path in the plotted coordinate
system. See {func}`erlab.plotting.mark_points`, {func}`erlab.plotting.fermiline`, and
{func}`erlab.plotting.label_subplots` for placement and formatting arguments. See the
{doc}`Python workflow tutorial <../../tutorials/python/index>` for the basic plotting
sequence.

## Figure Composer

1. **Layout:** In {guilabel}`Layout`, use one axes.
2. **Sources:** In {guilabel}`Sources`, choose {guilabel}`Add…` and add the
   two-dimensional `data` array.
3. **Recipe:** Add an {guilabel}`Image Plot` step that targets the axes. Set
   {guilabel}`Image data` to `data`, {guilabel}`Colormap` to `Greys`, and
   {guilabel}`Gamma` to `0.5`.
4. Add an {guilabel}`ERLab Method` step for `fermiline`. Target the axes and set
   {guilabel}`Line style` to `--`.
5. Add an {guilabel}`ERLab Method` step for `mark_points`. Target the axes, set
   {guilabel}`Points` to `-0.6, 0.0, 0.6`, set {guilabel}`Labels` to `K, G, K`, and
   set {guilabel}`Y` to `0.02`.
6. Add an {guilabel}`ERLab Method` step for `label_subplots`. Target the axes, set
   {guilabel}`Prefix` to `(`, and set {guilabel}`Suffix` to `)`.
