# Fermi-level calibration

Use this workflow before momentum conversion, EDC or MDC analysis, and final plotting.
Keep the original energy coordinate in the raw-data variable.

## Contents

- [Choose the calibration branch](#choose-the-calibration-branch)
- [Inspect the edge](#inspect-the-edge)
- [Correct a curved reference edge](#correct-a-curved-reference-edge)
- [Fit a flat edge](#fit-a-flat-edge)
- [Handle residual curvature without a reference](#handle-residual-curvature-without-a-reference)
- [Handle photon-energy scans](#handle-photon-energy-scans)
- [Validate and record the calibration](#validate-and-record-the-calibration)

## Choose the calibration branch

Use this order:

1. Use a compatible Au or Ag reference when one is supplied.
2. Decide from the raw angle-energy image whether the edge is flat, curved, or slanted.
3. Use a polynomial edge correction only with a reference spectrum.
4. Use a scalar edge center for a flat edge.
5. Treat photon-energy-dependent centers independently.

Do not assume that an energy coordinate named `eV` already has its Fermi level at zero.

## Inspect the edge

Plot a narrow region around the expected Fermi level as a function of detector angle.
Use raw intensity for the decision. A derivative image can support the inspection, but
it must not replace the raw image.

Check:

- whether a visible edge exists across the selected angular range;
- whether the edge is flat, curved, or slanted;
- whether narrow bands or gaps dominate the selected region;
- whether the temperature and instrumental resolution are known;
- whether `eV` is monotonic and expressed in electronvolts;
- whether a reference and sample used compatible analyzer geometry, slit or lens mode,
  pass energy, detector region, and acquisition correction.

Compare the embedded photon energy, temperature, and fixed-angle metadata with the
authoritative experimental log before fitting. Use the source that the user specifies.
Record every overwritten value. Photon energy affects both the expected kinetic Fermi
level and momentum conversion, so do not postpone this check.

Choose the first fit window carefully. A poorly conditioned first fit can contaminate
the polynomial fit even when later angle bins contain a clear edge.

- Restrict the angular range to bins with usable intensity and a visible edge.
- Exclude detector edges, gaps, mesh artifacts, and isolated bad channels.
- Use a local energy window that contains enough baseline on both sides of the edge.
- Exclude unrelated bands and avoid a window that is much wider than needed.
- Confirm that the window contains enough energy samples for the edge width.
- Coarsen weak data with a documented `bin_size` when averaging improves the fit.

Record the reference path, selected angular range, energy window, metadata temperature,
initial resolution, binning, and acquisition conditions in the notebook.

## Correct a curved reference edge

Use `erlab.analysis.gold.poly` for a curved Au or Ag reference. Start with degree 4.
Use a lower degree when degree 4 follows noise or oscillates between fitted centers.
Do not increase the degree only to reduce residuals.

Read the temperature from `sample_temp` metadata. Use the full Fermi-Dirac model only
when this value is present, finite, physically credible, and consistent with the
experimental record. If it is missing or unreliable, use `use_step_edge=True` so the
edge fits use the Gaussian-broadened step model.

Supply a realistic initial energy resolution in electronvolts. This can improve
convergence. Confirm the units before passing a metadata value; do not assume that an
instrument metadata field is already in electronvolts.

Keep one fit range when it contains a clean edge and suitable background for every EDC.
Set `adaptive=True` when a verified outer range contains every edge but a large edge
shift or nearby spectral features make one fixed range unreliable. The adaptive option
estimates a separate range for each EDC. If one estimate fails, that EDC uses the
complete outer range.

```python
edge_angle_range = (-15.0, 15.0)
edge_energy_range = (-0.15, 0.10)
initial_energy_resolution = 0.015  # eV
use_separate_fit_ranges = False

metadata_temperature = reference_data.attrs.get("sample_temp")
try:
    reference_temperature = float(metadata_temperature)
except (TypeError, ValueError):
    reference_temperature = None
temperature_is_reliable = (
    reference_temperature is not None
    and np.isfinite(reference_temperature)
    and reference_temperature >= 0.0
)

edge_fit_kwargs = {
    "use_step_edge": not temperature_is_reliable,
    "resolution": initial_energy_resolution,
}
if temperature_is_reliable:
    edge_fit_kwargs["temp"] = reference_temperature

reference_edge_fit = era.gold.poly(
    reference_data,
    along="alpha",
    angle_range=edge_angle_range,
    eV_range=edge_energy_range,
    adaptive=use_separate_fit_ranges,
    vary_temp=False,
    degree=4,
    plot=True,
    **edge_fit_kwargs,
)
```

The finite-value check is necessary but not sufficient. Set
`temperature_is_reliable = False` when the metadata conflicts with the experiment or
when the full edge model gives an implausible temperature-dependent line shape.

Inspect the edge-center points, their error bars, the polynomial, and the residuals in
the generated figure. Inspect the underlying edge fits when the polynomial has outliers
or unsupported structure.

Apply the accepted model only to data with compatible detector-angle coordinates and
acquisition conditions:

```python
reference_corrected = era.gold.correct_with_edge(
    reference_data,
    reference_edge_fit,
    along="alpha",
    plot=True,
)

sample_corrected = era.gold.correct_with_edge(
    sample_data,
    reference_edge_fit,
    along="alpha",
)
```

Do not use polynomial extrapolation outside the calibrated angular interval without a
clear warning. Crop to the overlap or obtain a wider reference when possible.

For reuse, keep only the public coefficient data and attach plain calibration metadata.
This form can be saved with xarray and remains accepted by `correct_with_edge`:

```python
edge_calibration = reference_edge_fit[["modelfit_coefficients"]].copy()
edge_calibration.attrs.update(
    {
        "reference_file": str(reference_path),
        "angle_range": edge_angle_range,
        "energy_range": edge_energy_range,
        "temperature_K": (
            reference_temperature if temperature_is_reliable else "unreliable"
        ),
        "initial_resolution_eV": initial_energy_resolution,
        "edge_model": "fermi" if temperature_is_reliable else "step",
        "polynomial_degree": 4,
    }
)
edge_calibration.to_netcdf("fermi_edge_calibration.nc")
```

Do not serialize `modelfit_results` with pickle as the primary calibration artifact.
Keep the notebook code that reproduces the fit.

## Fit a flat edge

For a flat edge, integrate over a detector-angle region with bulk continuum. If no such
region exists, average the full detector-angle range and state that the result relies on
density-of-states averaging.

Use an explicit region so the notebook shows what contributed to the EDC:

```python
continuum_center = 0.0
continuum_width = 4.0
edge_energy_range = (-0.12, 0.08)
sample_temperature = 20.0  # K
initial_resolution = 0.015  # eV

edge_edc = sample_data.qsel(
    alpha=continuum_center,
    alpha_width=continuum_width,
)
edge_edc = edge_edc.mean([dim for dim in edge_edc.dims if dim != "eV"])

flat_edge_fit = era.gold.quick_fit(
    edge_edc,
    eV_range=edge_energy_range,
    temp=sample_temperature,
    resolution=initial_resolution,
    fix_temp=True,
    plot=True,
)
```

Extract the fitted center and uncertainty from the public xarray-lmfit outputs:

```python
fermi_level = flat_edge_fit.modelfit_coefficients.sel(param="center").item()
fermi_level_stderr = flat_edge_fit.modelfit_stderr.sel(param="center").item()

sample_corrected = sample_data.assign_coords(eV=sample_data.eV - fermi_level)
```

Reassign the coordinate for one scalar energy offset. This avoids unnecessary
interpolation. Use `correct_with_edge` when the required shift varies over another
dimension.

Accept the scalar fit only when:

- the local window contains both occupied and unoccupied sides of a visible edge;
- the best fit follows the edge and the residuals have no clear structure;
- the center uncertainty is finite and small relative to the requested analysis scale;
- temperature and resolution are fixed or constrained well enough to avoid a strongly
  degenerate fit;
- the center is not held at a bound.

Use `FermiEdgeModel` through `quick_fit` by default. A broadened step model can provide
a rough diagnostic when thermal modeling is not possible, but do not interpret its
width as a separate physical temperature and resolution.

## Handle residual curvature without a reference

If a nominally flat edge is curved or slanted, do not derive an angular correction from
sample bands. Matrix elements, gaps, dispersing states, and the density of states can
produce a false edge trajectory.

Exclude detector and analyzer-acceptance boundaries from edge diagnostics. A circular
deflector-map mask is an analyzer boundary. Its shape and center do not measure sample
normal emission or Fermi edge curvature.

Retain the angle-energy diagnostic and request a compatible reference spectrum. You may
report a scalar fit to an integrated EDC as an approximate global energy offset, but:

- do not call the data edge-corrected;
- do not remove the observed angle dependence;
- state that the remaining curvature can bias near-Fermi analysis;
- do not use the approximate result for a publication claim without approval.

If the user explicitly requests a feasibility test without a reference, keep it as a
separate sample-derived diagnostic. Require the same smooth response in multiple
independent maps, exclude visible bands and acceptance boundaries, and validate the
model on a held-out map. Show raw and scalar-aligned data beside the diagnostic result.
Label it `sample-derived`, not `reference-calibrated`. Do not apply it to quantitative
publication figures without user approval.

## Handle photon-energy scans

Do not use one photon-energy slice as the Fermi calibration for an entire scan.
Monochromator miscalibration or drift can make the fitted center depend on photon
energy.

For a flat edge, create one integrated EDC per photon energy and fit each EDC. Preserve
the `hv` coordinate when collecting centers and standard errors. Plot both against
photon energy before applying a correction.

Plot the fitted center uncertainty and the residual from the expected kinetic-energy
trend. Treat an isolated point as an outlier candidate even when the optimizer reports
success. Inspect its raw EDC, fit, residual, acquisition window, and neighboring photon
energies. Do not connect or smooth through the point without marking it. Decide whether
the point is a failed edge fit, an analyzer-energy offset, or a photon-energy anomaly.
Keep it missing when the edge cannot be identified. When a clear local edge is present,
distinguish the per-slice energy correction from the separate trend anomaly.

Apply a photon-energy-dependent shift only where fits are successful, uncertainties are
finite, and the edge is visible. `correct_with_edge` accepts a center `DataArray` that
broadcasts over `hv`:

```python
photon_energy_corrected = era.gold.correct_with_edge(
    photon_energy_data,
    fermi_center_by_hv,
)
```

When a photon-energy-dependent reference contains a curved edge, `gold.poly` can retain
the additional dimensions in its fit output. Inspect the angular correction and Fermi
center at each photon energy before applying the multidimensional result.

If only one reference energy exists, state that the full scan has no measured
monochromator-drift correction.

## Validate and record the calibration

Before continuing:

1. Plot raw and corrected data with the same limits and normalization.
2. Confirm that a corrected reference edge is flat at zero energy.
3. Confirm that a scalar correction places the fitted center at zero.
4. Report the fitted center, standard error, polynomial degree, and selected ranges.
5. Record whether the calibration came from a reference, sample continuum, or user
   input.
6. Record unresolved curvature, extrapolation, failed fits, and photon-energy coverage.

Do not modify the raw variable. Use the accepted corrected variable for all later
momentum conversion, EDC/MDC fitting, and figures.
