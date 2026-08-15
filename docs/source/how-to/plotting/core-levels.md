(how-to-plotting-core-levels)=
(how-to-python-mark-core-levels)=

# Annotating core levels

## Python

Plot the measured spectrum first. Then mark the reference core-level binding energies
for the elements in the sample:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

fig, ax = plt.subplots()
core_spectrum.plot.line(ax=ax)
eplt.plot_core_levels(
    ["Bi", "Se"],
    ax=ax,
    energy="binding",
    binding_energy_sign="negative",
    linestyle="--",
)
```

```{eval-rst}
.. plot:: how_to/plotting.py mark_core_levels
   :include-source: false
   :alt: Synthetic Bi₂Se₃ core-level spectrum with bismuth and selenium reference lines
```

The synthetic spectrum uses published Bi₂Se₃ peak positions
{cite:t}`polyakov2019bi2se3`. Each d doublet has the 3:2 statistical area ratio.
The Bi-to-Se intensity is illustrative because cross sections and instrument response
affect it.
The compound peak positions do not coincide with the tabulated atomic reference lines.

Use `binding_energy_sign="positive"` for a conventional positive XPS binding-energy
axis. For a kinetic-energy axis, use `energy="kinetic"` and supply the measured photon
energy with `hv`. Also supply `work_function` when it is needed for the conversion.

The marked energies are tabulated reference values. Confirm each assignment from the
measured peak shape and the known sample composition. Chemical shifts, charging, and
an incorrect energy zero can move measured peaks away from the reference lines.

See {func}`erlab.plotting.plot_core_levels` for line orientation, labels, colors, and
explicit energy limits.

(how-to-gui-mark-core-levels)=

## Figure Composer

Add the measured spectrum to Figure Composer as a source. Then create this ordered
recipe:

1. Add a {guilabel}`Line/Profile` step for the spectrum on the target axes.
2. Add an {guilabel}`ERLab Method` step after the spectrum step. Select
   `plot_core_levels` and target the same axes.
3. Enter `["Bi", "Se"]` in {guilabel}`Elements`.
4. Set {guilabel}`Energy` to {guilabel}`binding` and {guilabel}`Binding sign` to
   {guilabel}`negative`.
5. Set {guilabel}`Line style` to `--`.
6. Check every marked line against the measured peaks and the known sample
   composition.

For a positive binding-energy axis, set {guilabel}`Binding sign` to
{guilabel}`positive`. For a kinetic-energy axis, set {guilabel}`Energy` to
{guilabel}`kinetic` and enter {guilabel}`Photon energy`. Enter the required
{guilabel}`Work function` when the energy conversion includes it.

Use {guilabel}`Text labels`, {guilabel}`Legend labels`, and {guilabel}`Text options` to
keep labels readable without hiding the data.
