# Curve fitting

ERLabPy uses [lmfit](https://lmfit.github.io/lmfit-py/) for models and parameters.
It uses [xarray-lmfit](https://xarray-lmfit.readthedocs.io/stable/) to apply the
models to labeled arrays. The fit results retain the coordinates of each source
spectrum.

Fitted parameter values depend on the model, fit range, constraints, and weights.
Optimizer convergence alone does not validate the line shape or its physical
interpretation.

## Models and fit results

| Object | Purpose |
| --- | --- |
| lmfit model | Line shape, independent variables, and model parameters |
| lmfit parameters | Initial values, bounds, fixed values, and expressions |
| ERLabPy predefined model | Predefined line shapes and parameter names for common ARPES fits |
| xarray-lmfit accessor | Repeated fits along named dimensions |
| Result Dataset | Coordinate-aware parameters, errors, fitted curves, residuals, and fit-result objects |

Predefined models use consistent components and parameter names. They do not select the
correct physical interpretation. A component can compensate for an unsuitable line
shape or missing background.

The result Dataset keeps numerical arrays and lmfit result objects together. Parameter
maps can therefore be aligned with other xarray data. The fit-result objects retain
diagnostic information that is not represented by one parameter array.

(explanation-fitting-independent)=

## Independent and global fits

{meth}`xarray.DataArray.xlm.modelfit` can fit one named dimension at every position
along the other dimensions. These fits share a model definition. They do not share
fitted parameter values.

| Fit structure | Main question | Main risk |
| --- | --- | --- |
| Independent fits | How do parameters vary across coordinates? | Noise and failed fits can produce irregular parameter maps |
| Global fit | Can one parameterization describe the complete region? | An incorrect shared relation can bias all fitted points |

A smooth trend is not imposed by independent fitting. A global fit imposes only the
relationships in its model.

The `coords` argument defines the dimensions fitted by one model evaluation. A single
coordinate such as `"eV"` fits each EDC independently across the remaining dimensions.
Multiple coordinates such as `["eV", "alpha"]` fit one multidimensional model across
the complete selected region. {class}`erlab.analysis.fit.models.FermiEdge2dModel` is
one model for the latter structure. It illustrates multidimensional fitting; it is not
a standard replacement for independent EDC analysis.

## Weights and fit validation

Weights of $1/\sigma$ express residuals in units of the expected standard uncertainty
$\sigma$. Weights change the fitted objective and the parameter estimates. Invalid or
zero uncertainties can invalidate the fit.

| Check | Information |
| --- | --- |
| Data, best fit, and residual | Structure that the model does not describe |
| Bounds and correlations | Restricted or weakly identifiable parameters |
| Standard errors | Covariance-based parameter uncertainty |
| Failed-fit locations | Regions where parameter maps are not valid |
| Fit-range and initial-value changes | Sensitivity to the selected fit range and starting values |

Strong correlations, active bounds, or a non-quadratic $\chi^2$ surface near the
minimum can make symmetric standard errors incomplete. Physical interpretation also
requires the measurement geometry and model assumptions.

(explanation-fitting-dask-execution)=

## Parallel fitting with Dask

Dask can represent independent fits as a lazy task graph. Dimensions that index the
fits can be split into chunks. The complete fitted dimension must remain available to
each task.

Chunking defines the graph. It does not select the scheduler. Without a distributed
client, `compute()` normally uses a local scheduler. A process-based client can run
CPU-bound fits in separate Python processes.

The result remains lazy until computation starts. Scheduler selection, worker count,
and chunk layout affect cost. They do not change the mathematical independence of the
fits.

The fitting procedures are in {doc}`../how-to/python/curve-fitting` and
{doc}`../how-to/gui/curve-fitting`. Available models and parameters are in
{mod}`erlab.analysis.fit.models`.
