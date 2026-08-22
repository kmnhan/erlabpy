# Curve fitting

Use these guides to correct a curved Fermi edge, fit overlapping peaks and EDC stacks,
inspect and save fit results, and calculate parameter uncertainties.

(fermi edge fitting)=

(how-to-python-correct-fermi-edge)=

## Correcting a curved Fermi edge

Use a measured reference spectrum whose temperature and energy units are known. Fit the
edge over verified angle and energy ranges:

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

Inspect the fit before applying it. Correct the reference or compatible sample data
without overwriting the original array:

```python
corrected = era.gold.correct_with_edge(data, edge_fit)
```

```{eval-rst}
.. plot:: how_to/curve_fitting.py correct_curved_fermi_edge
   :include-source: false
   :alt: Fermi edge fit diagnostic and corrected reference map
```

Use `use_step_edge=True` and a unit-checked resolution estimate when the reference
temperature is missing or unreliable. Do not derive an angle-dependent correction from
sample-band positions.

See {func}`erlab.analysis.gold.poly` and
{func}`erlab.analysis.gold.correct_with_edge` for accepted inputs and fit output.

(how-to-python-fit-fermi-edge-separate-ranges)=

(explanation-fitting-per-edc-ranges)=

## Choosing a fit range for each EDC

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
.. plot:: how_to/curve_fitting.py fit_fermi_edge_with_separate_ranges
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

(pre-defined-models)=

(how-to-python-fit-multiple-peaks)=

## Fitting multiple peaks to a spectrum

Select a one-dimensional spectrum and construct a model that matches the expected peak
shapes and background:

```python
import erlab.analysis as era

model = era.fit.models.MultiPeakModel(
    npeaks=2,
    peak_shapes=["lorentzian"],
    fd=False,
    background="linear",
    convolve=True,
)

params = model.make_params(
    p0_height=800.0,
    p0_center=-0.5,
    p0_width=0.03,
    p1_height=800.0,
    p1_center=0.5,
    p1_width=0.03,
    lin_bkg={"value": 0.0, "vary": False},
    const_bkg=0.0,
    resolution=0.03,
)

fit_result = spectrum.xlm.modelfit("kx", model=model, params=params, guess=False)
```

Inspect the fitted curve and residual:

```python
lmfit_result = fit_result.modelfit_results.item()
lmfit_result.plot()
```

Replace the coordinate name and initial parameter values with values appropriate for
the selected spectrum. Inspect the data, best fit, individual components, and residual
before interpreting peak parameters. Do not accept optimizer success alone as evidence
for a physically valid fit.

See {class}`erlab.analysis.fit.models.MultiPeakModel` for supported peak and background
models. See {doc}`curve fitting <../../explanation/fitting>` for fit validation and the
distinction between model components and measured intensity.

(how-to-python-fit-spectra-across-coordinate)=

## Independent fitting across coordinates

Use an xarray-lmfit model when `spectra` contains one curve along `eV` at each value of
another coordinate. Select the fit range, create the model, and run one independent fit
per coordinate:

```python
import erlab.analysis as era

fit_data = spectra.sel(eV=slice(-0.2, 0.2))
model = era.fit.models.FermiEdgeModel()
params = {
    "temp": {"value": sample_temperature, "vary": False},
    "back1": {"value": 0.0, "vary": False},
}

fit_result = fit_data.xlm.modelfit("eV", model=model, params=params, guess=True)
```

Inspect failed fits, parameter uncertainties, and the fitted curves before using the
fitted parameter values:

```python
centers = fit_result.modelfit_coefficients.sel(param="center")
center_errors = fit_result.modelfit_stderr.sel(param="center")
```

```{eval-rst}
.. plot:: how_to/curve_fitting.py fit_spectra_across_coordinate
   :include-source: false
   :alt: Fermi edge centers and uncertainties fitted across angle
```

See {ref}`explanation-fitting-independent` for the distinction between independent and
global fits. Use {ref}`how-to-python-inspect-fit-results` when many fitted spectra must
be checked. See {meth}`xarray.DataArray.xlm.modelfit` for the returned variables.

(how-to-python-inspect-fit-results)=

## Inspecting fitted spectra and parameters

Use the interactive result view when a fit result contains many spectra. This task
requires the `viz` optional dependencies listed in {ref}`optional-dependencies`.
Here, `fit_result` is the completed xarray fit
{class}`Dataset <xarray.Dataset>`.

Open the fitted curves, model components, parameter values, and fit statistics:

```python
fit_result.qshow(plot_components=True)
```

Move the coordinate slider to each suspicious fit. Compare the measured data with the
best-fit curve and each model component. Check the parameter uncertainties and fit
statistics before you use the fitted values.

Open a separate view of each fitted parameter and its standard error:

```python
fit_result.qshow.params()
```

This parameter view requires one coordinate that indexes the fitted spectra. Select or
stack other indexing dimensions first when the result has more than one.

### Static component and residual plot

For the one-dimensional `spectrum` from {ref}`how-to-python-fit-multiple-peaks`, use a
fixed inspection figure. Here, `spectrum` is one selected source spectrum.
`lmfit_result` is its corresponding lmfit {class}`lmfit.model.ModelResult`:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

coordinate = spectrum.kx.values
measured = spectrum.values
components = lmfit_result.eval_components(x=coordinate)
residual = measured - lmfit_result.best_fit

fig, axes = plt.subplots(
    2,
    1,
    figsize=(6.4, 3.0),
    layout="compressed",
    sharex=True,
    height_ratios=(3, 1),
)
axes[0].plot(coordinate, measured, "o", markersize=3, label="Measured data")
axes[0].plot(coordinate, lmfit_result.best_fit, label="Best fit")
for name, component in components.items():
    label = name.rstrip("_").replace("_", " ").title()
    axes[0].plot(coordinate, component, "--", label=label)
axes[0].set_ylabel("Intensity")
axes[0].legend(ncols=2)

axes[1].axhline(0.0, color="0.5", linewidth=1)
axes[1].plot(coordinate, residual, ".-", color="tab:red", markersize=3)
axes[1].set(xlabel=r"$k_x$ (Å$^{-1}$)", ylabel="Residual")
eplt.clean_labels(axes)
```

```{eval-rst}
.. plot:: how_to/curve_fitting.py inspect_fit_components_and_residuals
   :include-source: false
   :alt: Measured spectrum with a best fit and separate peak and background components above its residual
```

The generated example uses two Lorentzian peaks and a linear background for an MDC.
Use the components from the model that is justified for the measured spectrum.
Structured residuals show behavior that the model does not describe.

See {ref}`explanation-fitting-independent` for the meaning of independent fit results.
See {meth}`xarray.Dataset.qshow.fit` and {meth}`xarray.Dataset.qshow.params` for the
accepted result layouts.

(how-to-python-fit-edcs-in-parallel)=

## Parallel fitting of large EDC stacks

Use this guide when independent fits across a large EDC stack take too long. The
`spectra` array must contain one EDC along `eV` at each point of one or more other
dimensions.

Select the fit range and prepare the model:

```python
import erlab.analysis as era

fit_data = spectra.sel(eV=slice(-0.2, 0.2))
model = era.fit.models.FermiEdgeModel()
params = {
    "temp": {"value": sample_temperature, "vary": False},
    "back1": {"value": 0.0, "vary": False},
}
```

Chunk the coordinate across which the independent fits are distributed. Do not chunk
the fitted `eV` dimension:

```python
spectra_chunked = fit_data.chunk({"alpha": 20})
```

Create the lazy fit result:

```python
lazy_fit_result = spectra_chunked.xlm.modelfit(
    "eV",
    model=model,
    params=params,
    guess=True,
)
```

Start a local process-based cluster and compute the result inside context managers:

```python
from dask.distributed import Client, LocalCluster

with LocalCluster(
    n_workers=4,
    threads_per_worker=1,
    processes=True,
) as cluster:
    with Client(cluster):
        fit_result = lazy_fit_result.compute()
```

The context managers close the local client, scheduler, and workers after a successful
computation or an error. Select the worker count for the available CPU cores and
memory.

If an external Dask scheduler is available, connect to it instead:

```python
from dask.distributed import Client

with Client("tcp://scheduler-address:8786"):
    fit_result = lazy_fit_result.compute()
```

This context closes only the local client connection. It does not stop the external
scheduler or its workers. Stop those services through the system that started them.

Adjust the `alpha` chunk size for the available memory and the cost of one fit.
Use the actual independent dimension name when it is not `alpha`. Keep `eV` in one
chunk so each fit receives a complete EDC.

See {ref}`explanation-fitting-dask-execution` for the distinction between Dask chunks
and execution. See {meth}`xarray.DataArray.xlm.modelfit` for Dask requirements and output
variables.

(how-to-python-save-fit-results)=

## Saving and reopening fit results

Use the xarray-lmfit fit format when you must continue work with the fitted parameters,
curves, and lmfit result objects in another Python session:

```python
from xarray_lmfit import load_fit, save_fit

fit_path = "fit-result.h5"
save_fit(fit_result, fit_path, engine="h5netcdf")
restored_fit = load_fit(fit_path, engine="h5netcdf")
```

Access the parameter arrays and serialized lmfit results after loading:

```python
restored_fit.modelfit_coefficients
restored_fit.modelfit_stderr
restored_fit.modelfit_results
```

The file contains the data stored in the fit result. It does not restore source data
outside the fitted range. Save the complete source data separately when you need it.

The serialized lmfit objects do not guarantee compatibility across Python, lmfit, or
xarray-lmfit versions. Reopen long-lived results in a compatible environment. Supply
the `funcdefs` argument to {func}`xarray_lmfit.load_fit` when a saved custom model uses
functions that the loader cannot import.

See {func}`xarray_lmfit.save_fit` and {func}`xarray_lmfit.load_fit` for file options and
serialization limits. Use {ref}`how-to-gui-reopen-saved-fit` when you must inspect the
saved result in ftool.

(how-to-python-uncertainty-after-derivative-free-fit)=

## Standard errors after derivative-free fitting

Use this procedure when you have justified a derivative-free lmfit method and need a
local covariance estimate. Install `numdifftools` from the `misc` optional dependencies
listed in {ref}`optional-dependencies`.

Request covariance calculation when you fit the prepared spectrum:

```python
fit_result = spectrum.xlm.modelfit(
    "kx",
    model=model,
    params=params,
    guess=False,
    method="nelder",
    calc_covar=True,
)
```

Inspect the standard errors and the underlying lmfit result:

```python
standard_errors = fit_result.modelfit_stderr
lmfit_result = fit_result.modelfit_results.item()
lmfit_result.errorbars
```

Do not change the minimizer only to obtain finite errors. Missing or unstable errors can
indicate active bounds, strong parameter correlations, or a model that does not identify
the parameters. Check those conditions before you report the covariance estimate.

See {meth}`lmfit.model.Model.fit` for `method` and `calc_covar`. See
{doc}`curve fitting <../../explanation/fitting>` for the limits of local covariance and
standard errors.

(how-to-python-fit-with-iminuit)=

## Profile likelihood intervals

Use ERLabPy's Minuit adapter when you need Minuit-specific covariance and profile
likelihood intervals for a prepared lmfit model. Install the optional `iminuit`
dependency before you run this procedure:

```python
import erlab.analysis as era

model = era.fit.models.MultiPeakModel(
    npeaks=2,
    peak_shapes=["lorentzian"],
    fd=False,
    background="linear",
    convolve=True,
)

minuit_fit = era.fit.minuit.Minuit.from_lmfit(
    model,
    spectrum,
    spectrum.kx,
    p0_center=-0.5,
    p1_center=0.5,
    p0_width=0.03,
    p1_width=0.03,
    p0_height=1000,
    p1_height=1000,
    lin_bkg={"value": 0.0, "vary": False},
    const_bkg=0.0,
    resolution=0.03,
)

minuit_fit.migrad()
minuit_fit.minos()
minuit_fit.hesse()
```

Replace the coordinate and parameter values with those for the prepared spectrum and
model. Inspect parameter limits, correlations, uncertainty estimates, and the fitted
curve before interpreting the result.

{meth}`hesse <iminuit.Minuit.hesse>` estimates local covariance near the minimum.
{meth}`minos <iminuit.Minuit.minos>` calculates parameter-dependent profile intervals
that can be asymmetric. Active bounds and strong correlations can affect both results.

See {class}`erlab.analysis.fit.minuit.Minuit` for supported minimization and uncertainty
methods. Compare the estimates only after you confirm that both fits use the same data,
model, parameters, bounds, and weights.

(how-to-python-adjust-minuit-start-values)=

## Interactive adjustment of Minuit starting values

Use the Minuit interactive view when a fit is sensitive to its initial parameter values.
Prepare `minuit_fit` with {meth}`erlab.analysis.fit.minuit.Minuit.from_lmfit`, as in
{ref}`how-to-python-fit-with-iminuit`. In a Jupyter notebook with `ipywidgets` installed,
open the interactive view:

```python
minuit_widget = minuit_fit.interactive()
minuit_widget
```

Adjust the parameters until the model follows the expected peaks and background. Then
run the minimization from those values:

```python
minuit_fit.migrad()
```

Inspect the fitted curve, parameter limits, and covariance after minimization. Return to
the interactive view when the optimizer reaches an unsuitable minimum. Do not use the
widget to select a model only because it can follow the measured noise.

See {meth}`iminuit.Minuit.interactive` for widget options. See
{doc}`curve fitting <../../explanation/fitting>` for the role of initial values and fit
validation.
