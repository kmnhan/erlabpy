(fermi edge fitting)=

(how-to-python-fermi-edge-correction)=

# Fermi edge correction

Use a measured reference spectrum with known temperature and energy units. The
reference and the data to correct must use the same analyzer settings and the same
detector-coordinate definition. Restrict the correction to the detector-coordinate
range covered by the reference fit.

For the corresponding Manager workflow, see
{doc}`../gui/fermi-edge-correction`.

(how-to-python-correct-fermi-edge)=

## Curved Fermi edge correction

Fit the edge over verified angle and energy ranges:

```python
import erlab.analysis as era

edge_fit = era.gold.poly(
    gold_reference,
    angle_range=(-15, 15),
    eV_range=(-0.2, 0.2),
    temp=sample_temperature,
    vary_temp=False,
    bkg_slope=False,
    degree=2,
    plot=True,
)
```

Inspect the fitted edge centers, their uncertainties, and the polynomial before you
apply the correction. Correct the reference or compatible sample data without
overwriting the original array:

```python
corrected = era.gold.correct_with_edge(data, edge_fit)
```

```{eval-rst}
.. plot:: how_to/fermi_edge_correction.py correct_curved_fermi_edge
   :include-source: false
   :alt: Fermi edge fit diagnostic and corrected reference map
```

By default, {func}`erlab.analysis.gold.correct_with_edge` can change the `eV`
coordinate and its length to retain the shifted spectra. The other dimensions remain
in their original order. Set `shift_coords=False` only when the output must keep the
original energy grid and shape. This setting can discard intensity at the grid
boundaries.

Use `use_step_edge=True` and a unit-checked resolution estimate when the reference
temperature is missing or unreliable. Do not derive an angle-dependent correction from
sample-band positions.

See {func}`erlab.analysis.gold.poly` and
{func}`erlab.analysis.gold.correct_with_edge` for accepted inputs and fit output.

(how-to-python-fit-fermi-edge-separate-ranges)=

## Separate EDC fit ranges

Use a separate fit range for each EDC when the edge position changes substantially
across a measured reference or when other spectral features make one fixed range
unreliable. Set one outer energy range that contains the edge and usable background for
every EDC:

```python
import erlab.analysis as era

edge_energy_range = (-0.35, 0.20)
edge_fit = era.gold.poly(
    gold_reference,
    angle_range=(-15, 15),
    eV_range=edge_energy_range,
    adaptive=True,
    temp=sample_temperature,
    resolution=energy_resolution,
    vary_temp=False,
    degree=2,
    plot=True,
)
```

Inspect a difficult EDC and its estimated range when you need to check the selection:

```python
selected_angle = 0.0
selected_edc = gold_reference.sel(alpha=selected_angle, method="nearest")
selected_edc = selected_edc.where(
    (selected_edc.eV >= min(edge_energy_range))
    & (selected_edc.eV <= max(edge_energy_range)),
    drop=True,
)
estimated_range = era.gold.guess_edge_fit_range(
    selected_edc,
    temp=sample_temperature,
    resolution=energy_resolution,
)
estimated_edc = selected_edc.where(
    (selected_edc.eV >= estimated_range[0])
    & (selected_edc.eV <= estimated_range[1]),
    drop=True,
)
estimated_edc.plot()
```

```{eval-rst}
.. plot:: how_to/fermi_edge_correction.py fit_fermi_edge_with_separate_ranges
   :include-source: false
   :alt: Three Fermi edge EDCs with independently estimated fit ranges
```

Supply the measured temperature and an energy-resolution estimate in electronvolts.
Use `use_step_edge=True` when the temperature is missing or unreliable. Adaptive range
selection detects a falling edge. Do not use it to derive a reference correction from
sample-band positions.

The outer `eV_range` limits the data available to every fit. With `adaptive=True`,
ERLabPy estimates a separate interval inside that limit for each EDC. The estimate uses
the expected thermal and instrumental edge width. It changes the fit region, not the
edge model.

See {func}`erlab.analysis.gold.guess_edge_fit_range` for the estimator requirements.
