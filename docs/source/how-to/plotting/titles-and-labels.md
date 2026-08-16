(how-to-plotting-titles-and-labels)=
(how-to-python-set-panel-labels)=

# Titles and axis labels

## Python

Create the axes in their final layout. Plot each dataset. Then supply one title and one
axis label for each panel:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="compressed")

eplt.plot_array(
    constant_energy_map,
    ax=axes[0],
    cmap="Greys",
    gamma=0.5,
    aspect="equal",
)
axes[1].plot(edc.eV, edc, color="0.2")
eplt.fermiline(ax=axes[1], orientation="v", linestyle="--")

eplt.set_titles(axes, ["Constant energy map", "Energy distribution curve"])
eplt.set_xlabels(axes, [r"$k_x$ (Å$^{-1}$)", r"$E-E_F$ (eV)"])
eplt.set_ylabels(axes, [r"$k_y$ (Å$^{-1}$)", "Intensity (arb. units)"])
```

```{eval-rst}
.. plot:: how_to/plotting.py set_panel_titles_and_labels
   :include-source: false
   :alt: Constant energy map and energy distribution curve with distinct scientific axis labels
```

The default sequence is row-major. Use `order="F"` when the labels must follow the
columns first. The number of labels must match the number of axes. Pass one string when
all panels need the same label.

See {func}`erlab.plotting.set_titles`, {func}`erlab.plotting.set_xlabels`, and
{func}`erlab.plotting.set_ylabels` for text-formatting arguments.

## Figure Composer

Add `constant_energy_map` and `edc` to the figure as sources. Then create this ordered
recipe:

1. In {guilabel}`Layout`, select {guilabel}`subplots`. Set {guilabel}`Rows` to `1`
   and {guilabel}`Columns` to `2`.
2. Add an {guilabel}`Image Plot` step for `constant_energy_map` on the left axes. Set
   {guilabel}`Aspect` to `equal`.
3. Add a {guilabel}`Line/Profile` step for `edc` on the right axes.
4. Add an {guilabel}`ERLab Method` step on the right axes. Select `fermiline`, set
   {guilabel}`Orientation` to `v`.
5. Add an {guilabel}`ERLab Method` step on both axes. Select `set_titles`, keep
   {guilabel}`Order` set to {guilabel}`C (Row)`, and enter `Constant energy map` and
   `Energy distribution curve` on separate lines in {guilabel}`Text`.
6. Add another {guilabel}`ERLab Method` step on both axes. Select `set_xlabels` and
   enter `$k_x$ (Å$^{-1}$)` and `$E-E_F$ (eV)` on separate lines in
   {guilabel}`Text`.
7. Add another {guilabel}`ERLab Method` step on both axes. Select `set_ylabels` and
   enter `$k_y$ (Å$^{-1}$)` and `Intensity (arb. units)` on separate lines in
   {guilabel}`Text`.

Keep the three label steps after the plotting steps. This order replaces labels that a
plotting step supplies from the source coordinates.
