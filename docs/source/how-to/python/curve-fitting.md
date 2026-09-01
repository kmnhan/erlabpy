# Curve fitting

Use these guides to fit overlapping peaks and EDC stacks, inspect and save fit results,
and calculate parameter uncertainties.

ERLabPy uses [lmfit](https://lmfit.github.io/lmfit-py/) for models, parameters, and
optimization. It uses [xarray-lmfit](https://xarray-lmfit.readthedocs.io/stable/) to
fit xarray objects. Use the lmfit documentation for general curve-fitting concepts.

For measured-reference fitting and Fermi edge correction, see
{doc}`fermi-edge-correction`.

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
models.

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

Use {ref}`how-to-python-inspect-fit-results` when many fitted spectra must be checked.
See {meth}`xarray.DataArray.xlm.modelfit` for the returned variables.

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

Use a two-panel plot when you must include the fit components and residual in a static
report. This example fits a Bi 5d spin-orbit doublet. Select a range with background on
both sides of the doublet. Here, `core_spectrum` is a one-dimensional spectrum with an
`eV` coordinate.

```python
import erlab.analysis as era
import matplotlib.pyplot as plt
import erlab.plotting as eplt

energy = core_spectrum.eV.values
intensity = core_spectrum.values

model = era.fit.models.MultiPeakModel(
    npeaks=2,
    peak_shapes="voigt",
    fd=False,
    background="shirley",
    convolve=False,
)
params = model.guess(intensity, x=energy)

# Bi 5d5/2 position and initial Voigt widths, in eV
params["p0_center"].set(value=-25.0, min=-25.5, max=-24.5)
params["p0_sigma"].set(value=0.20, min=0.0, max=0.5)
params["p0_gamma"].set(value=0.10, min=0.0, max=0.5)

# Bi 5d3/2 splitting, shared Gaussian width, and 3:2 area ratio
params["p1_center"].set(expr="p0_center - 3.0")
params["p1_sigma"].set(expr="p0_sigma")
params["p1_gamma"].set(value=0.10, min=0.0, max=0.5)
params["p1_amplitude"].set(expr="2 * p0_amplitude / 3")

fit_result = core_spectrum.xlm.modelfit(
    "eV",
    model=model,
    params=params,
    guess=False,
)
lmfit_result = fit_result.modelfit_results.item()
fit_data = fit_result.modelfit_data
best_fit = fit_result.modelfit_best_fit
components = lmfit_result.eval_components(x=fit_data.eV.values)
residual = fit_data - best_fit

fig, axes = plt.subplots(
    2,
    1,
    figsize=(4.8, 3.6),
    layout="compressed",
    sharex=True,
    height_ratios=(3, 1),
)
axes[0].plot(fit_data.eV, fit_data, "o", markersize=2, label="Measured data")
axes[0].plot(best_fit.eV, best_fit, color="black", label="Best fit")
axes[0].plot(fit_data.eV, components["2Peak_p0"], label=r"Bi 5d$_{5/2}$")
axes[0].plot(fit_data.eV, components["2Peak_p1"], label=r"Bi 5d$_{3/2}$")
axes[0].plot(
    fit_data.eV,
    components["2Peak_baseline"] + components["2Peak_shirley"],
    "--",
    color="0.45",
    label="Shirley background",
)
axes[0].set_ylabel("Intensity (arb. units)")
axes[0].legend(ncols=2)

axes[1].axhline(0.0, color="0.5", linewidth=1)
axes[1].plot(residual.eV, residual, ".", color="0.25", markersize=2)
axes[1].set(xlabel=r"$E-E_F$ (eV)", ylabel="Residual")
eplt.clean_labels(axes)
```

```{eval-rst}
.. plot:: how_to/curve_fitting.py inspect_fit_components_and_residuals
   :include-source: false
   :alt: Bi 5d spectrum with a Voigt doublet, Shirley background, best fit, and residual
```

{meth}`xarray.DataArray.xlm.modelfit` retains the energy coordinate in
`modelfit_data` and `modelfit_best_fit`. The `modelfit_results` entry contains the
underlying lmfit result. This example uses it only to evaluate the named model
components.

{meth}`~erlab.analysis.fit.models.MultiPeakModel.guess` initializes the constant
background from the low-binding-energy end of the fit range. It initializes the
Shirley step from the endpoint difference and the integrated signal. Do not set an
arbitrary Shirley step before you inspect this estimate.

The 3:2 area constraint is the statistical ratio for a d doublet. The shared Gaussian
width represents the common instrumental broadening. The Lorentzian widths remain
independent in this example. Replace the center, splitting, and width bounds with values
that apply to the selected core level and instrument. Structured residuals show
behavior that the model does not describe.

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

See {meth}`xarray.DataArray.xlm.modelfit` for Dask requirements and output variables.

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

See {meth}`lmfit.model.Model.fit` for `method` and `calc_covar`.

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

See {meth}`iminuit.Minuit.interactive` for widget options.
