# EDC, MDC, and dispersion analysis

Use energy- and momentum-calibrated data when the requested result depends on those
axes. Keep every selection width and fit choice in a visible configuration cell.

## Contents

- [Inspect and select data](#inspect-and-select-data)
- [Fit an MDC](#fit-an-mdc)
- [Fit an EDC](#fit-an-edc)
- [Fit a sequence and trace dispersion](#fit-a-sequence-and-trace-dispersion)
- [Validate fits](#validate-fits)
- [Use interactive fitting](#use-interactive-fitting)

## Inspect and select data

Plot the parent cut before extracting curves. Confirm axis names, units, energy zero,
momentum zero, coordinate order, and finite sampling.

Use `qsel` widths to average a physical interval rather than selecting one noisy pixel.
Record the center and full width:

```python
edc = momentum_data.qsel(
    kx=edc_kx,
    kx_width=edc_kx_width,
    ky=edc_ky,
    ky_width=edc_ky_width,
)

mdc = momentum_cut.qsel(
    eV=mdc_energy,
    eV_width=mdc_energy_width,
)
```

Plot each extracted curve and its source region. Do not fit until the requested feature
is visible and the curve has enough points across the peak and background.

## Fit an MDC

Use Voigt peaks by default. The analytic Gaussian-Lorentzian convolution is numerically
stable, and it separates a Lorentzian lifetime width (`gamma`) from a Gaussian
instrumental width (`sigma`). Disable the Fermi cutoff for an MDC.

Use one named Gaussian-width parameter and make every `pN_sigma` an expression of that
parameter. Do not let the optimizer assign a different instrumental Gaussian width to
each peak. A Voigt `sigma` is a standard deviation, not a full width at half maximum.

```python
n_peaks = 2
instrument_sigma = momentum_resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))

mdc_model = era.fit.models.MultiPeakModel(
    npeaks=n_peaks,
    peak_shapes="voigt",
    fd=False,
    background="linear",
    convolve=False,
)

mdc_params = mdc_model.make_params(
    p0_center=-0.20,
    p0_gamma=0.03,
    p0_amplitude=500.0,
    p1_center=0.20,
    p1_gamma=0.03,
    p1_amplitude=500.0,
)
mdc_params.add("instrument_sigma", value=instrument_sigma, min=0.0, vary=False)
for peak_index in range(n_peaks):
    mdc_params[f"p{peak_index}_sigma"].set(expr="instrument_sigma")

mdc_fit = mdc.xlm.modelfit(
    "kx",
    model=mdc_model,
    params=mdc_params,
    guess=True,
)
```

Set `instrument_sigma` to vary only when the data can constrain one common Gaussian
width and no trusted instrumental value exists. Adapt centers, bounds, amplitudes,
Lorentzian widths, peak count, and fitted coordinate to the data. Do not copy numeric
guesses from this example without inspecting the curve.

`MultiPeakModel(convolve=True)` adds a separate Gaussian `resolution` convolution. Do
not vary `resolution` and a Voigt `instrument_sigma` as two representations of the same
instrumental broadening. Use the shared Voigt sigma with `convolve=False` for the
default peak fit. Use the global convolution only when the complete model, such as a
Fermi cutoff, must be resolution-broadened; then fix the Voigt sigma to zero or to a
separately justified nonduplicate contribution.

Inspect the scalar result and the residual:

```python
mdc_result = mdc_fit.modelfit_results.item()
mdc_result.plot()

mdc_residual = mdc - mdc_fit.modelfit_best_fit
mdc_residual.qplot()
```

## Fit an EDC

Choose the EDC model from the selected energy window:

- Use Voigt peaks by default and tie every peak sigma to one named Gaussian-width
  parameter.
- Use `MultiPeakModel(fd=True, convolve=True)` when the spectral peaks cross a calibrated
  Fermi cutoff and the complete model must be broadened by a known resolution. In this
  case, do not duplicate that resolution in the Voigt sigma.
- Use `fd=False` when the fitted window is fully below the cutoff or when the model must
  not impose one.
- Use `era.gold.quick_fit` for Fermi-edge calibration. Do not use a spectral peak fit as
  a substitute for energy calibration.
- Fix the calibrated Fermi center to zero when the EDC model includes a cutoff, unless
  the analysis explicitly tests a remaining offset.

Example for peaks whose shared Voigt sigma represents the instrumental contribution:

```python
n_peaks = 1
instrument_sigma = energy_resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))

edc_model = era.fit.models.MultiPeakModel(
    npeaks=n_peaks,
    peak_shapes="voigt",
    fd=True,
    background="linear",
    convolve=False,
)

edc_params = edc_model.make_params(
    p0_center=initial_peak_energy,
    p0_gamma=initial_lorentzian_width,
    p0_amplitude=initial_peak_amplitude,
    efermi={"value": 0.0, "vary": False},
    temp={"value": sample_temperature, "vary": False},
)
edc_params.add("instrument_sigma", value=instrument_sigma, min=0.0, vary=False)
for peak_index in range(n_peaks):
    edc_params[f"p{peak_index}_sigma"].set(expr="instrument_sigma")

edc_fit = edc.xlm.modelfit(
    "eV",
    model=edc_model,
    params=edc_params,
    guess=True,
)
```

Fit only a physically justified energy range. Check whether the chosen background can
be distinguished from broad peaks and the cutoff. When the Fermi cutoff itself needs
resolution convolution, enable `convolve=True`, pass a fixed or constrained
`resolution`, and do not count the same Gaussian width again in `instrument_sigma`.

## Fit a sequence and trace dispersion

For an energy-momentum cut, fit the momentum coordinate to obtain one MDC fit per
energy. Limit the energy range to slices with identifiable peaks:

```python
dispersion_region = momentum_cut.sel(eV=slice(*dispersion_energy_range))

dispersion_fit = dispersion_region.xlm.modelfit(
    "kx",
    model=mdc_model,
    params=mdc_params,
    guess=True,
)

peak_centers = dispersion_fit.modelfit_coefficients.sel(
    param=["p0_center", "p1_center"]
)
peak_stderr = dispersion_fit.modelfit_stderr.sel(param=["p0_center", "p1_center"])
```

Reverse the slice endpoints when `eV` is descending, or sort the coordinate explicitly
and record that change.

Construct a success mask from the actual `ModelResult` objects. Combine it with finite
uncertainty and parameter-bound checks:

```python
fit_success = xr.apply_ufunc(
    lambda result: result is not None and bool(result.success),
    dispersion_fit.modelfit_results,
    vectorize=True,
    output_dtypes=[bool],
)

valid_centers = peak_centers.where(fit_success & np.isfinite(peak_stderr))
```

Plot the centers on the original intensity cut with error bars. Inspect representative
fits at the start, middle, end, crossings, weak-intensity regions, and every apparent
discontinuity.

Independent fits do not preserve band identity. When peaks cross or exchange intensity:

- use physically justified bounds and continuity constraints;
- compare fitted curves and components, not parameter names alone;
- relabel branches only with documented physical evidence;
- stop a trace when the feature disappears;
- show gaps in the overlay instead of connecting rejected points.

Photon-energy sequences need an additional check. Bulk bands can disperse with
out-of-plane momentum, and matrix elements can replace one visible peak with another.
Do not treat all peaks or derivative ridges at different photon energies as one band.
Track only a branch that remains identifiable from its dispersion and neighboring raw
cuts. Show the selected points and rejected alternatives. Stop the trace when identity
is not supported.

Do not smooth fitted centers before validating the individual fits. If smoothing is a
separate requested analysis, plot unsmoothed centers and residuals as well.

## Validate fits

Accept a fit only after checking:

- `ModelResult.success` and its message;
- finite standard errors for reported parameters;
- parameter values and errors in physical units;
- contact with bounds;
- correlations or degeneracy between peaks, background, temperature, and resolution;
- residual structure and missed peaks;
- agreement between the best-fit curve and the plotted data;
- stability under a reasonable change of fit range or initial value when the result is
  scientifically important.

Optimizer success is not sufficient. Reject a result when the model is visibly wrong,
the peak lies outside the selected data, the width collapses to the sampling interval,
or uncertainty is too large for the claimed conclusion.

Report integration widths, fit ranges, model components, fixed parameters, rejected
slices, and uncertainty handling in the notebook.

## Use interactive fitting

Use `ftool` to choose regions, models, bounds, and parameter propagation when sequence
fits need visual tuning. Use `goldtool` for Fermi-edge fits. Copy the accepted public
API code into the notebook, rerun it, and validate the resulting dataset. Do not make
the final notebook depend on the interactive window.
