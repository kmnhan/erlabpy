"""Fermi edge fitting."""

__all__ = [
    "correct_with_edge",
    "edge",
    "poly",
    "poly_from_edge",
    "quick_fit",
    "quick_resolution",
    "resolution",
    "resolution_roi",
    "spline_from_edge",
]
import typing
import warnings
from collections.abc import Callable

import lmfit
import lmfit.model
import matplotlib
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import xarray as xr

import erlab

if typing.TYPE_CHECKING:
    import joblib
    import tqdm.auto as tqdm
else:
    import lazy_loader as _lazy

    joblib = _lazy.load("joblib")
    tqdm = erlab.utils.misc.LazyImport("tqdm.auto")


def correct_with_edge(
    darr: xr.DataArray,
    modelresult: lmfit.model.ModelResult
    | xr.Dataset
    | npt.NDArray[np.floating]
    | Callable,
    shift_coords: bool = True,
    plot: bool = False,
    plot_kw: dict | None = None,
    **shift_kwargs,
):
    """
    Corrects the given data array `darr` with the given values or fit result.

    Parameters
    ----------
    darr
        The input data array to be corrected.
    modelresult
        The model result that contains the fermi edge information. It can be an instance
        of `lmfit.model.ModelResult`, a numpy array containing the edge position at each
        angle, a fit result dataset that contains polynomial coefficients, or a callable
        function that takes an array of angles and returns the corresponding energy
        value.
    shift_coords
        If `True`, the coordinates of the output data will be changed so that the output
        contains all the values of the original data. If `False`, the coordinates and
        shape of the original data will be retained, and only the data will be shifted.
        Defaults to `False`.
    plot
        Whether to plot the original and corrected data arrays. Defaults to `False`.
    plot_kw
        Additional keyword arguments for the plot. Defaults to `None`.
    **shift_kwargs
        Additional keyword arguments to :func:`erlab.analysis.transform.shift`.

    Returns
    -------
    corrected : xarray.DataArray
        The edge corrected data.
    """
    if plot_kw is None:
        plot_kw = {}

    if isinstance(modelresult, xr.Dataset) and "modelfit_results" in modelresult:
        modelresult = modelresult.modelfit_results.values.item()

    if isinstance(modelresult, lmfit.model.ModelResult):
        if isinstance(modelresult.model, erlab.analysis.fit.models.FermiEdge2dModel):
            edge_quad = np.polynomial.polynomial.polyval(
                darr.alpha,
                np.array(
                    [
                        modelresult.best_values[f"c{i}"]
                        for i in range(modelresult.model.func.poly.degree + 1)
                    ]
                ),
            )
        else:
            edge_quad = modelresult.eval(x=darr.alpha)

    elif callable(modelresult):
        edge_quad = modelresult(darr.alpha.values)

    elif isinstance(modelresult, np.ndarray | xr.DataArray):
        if len(darr.alpha) != len(modelresult):
            raise ValueError(
                "Length of modelresult must be equal to the length of alpha in data"
            )
        edge_quad = modelresult

    else:
        raise TypeError(
            "modelresult must be one of "
            "lmfit.model.ModelResult, "
            "and np.ndarray or a callable"
        )

    if isinstance(edge_quad, np.ndarray):
        edge_quad = xr.DataArray(
            edge_quad, coords={"alpha": darr.alpha}, dims=["alpha"]
        )

    corrected = erlab.analysis.transform.shift(
        darr, -edge_quad, "eV", shift_coords=shift_coords, **shift_kwargs
    )

    if plot is True:
        axes = typing.cast(
            "npt.NDArray", plt.subplots(1, 2, layout="constrained", figsize=(10, 5))[1]
        )

        plot_kw.setdefault("cmap", "copper")
        plot_kw.setdefault("gamma", 0.5)

        if darr.ndim > 2:
            avg_dims = list(darr.dims)[:]
            avg_dims.remove("alpha")
            avg_dims.remove("eV")
            erlab.plotting.plot_array(darr.mean(avg_dims), ax=axes[0], **plot_kw)
            erlab.plotting.plot_array(corrected.mean(avg_dims), ax=axes[1], **plot_kw)
        else:
            erlab.plotting.plot_array(darr, ax=axes[0], **plot_kw)
            erlab.plotting.plot_array(corrected, ax=axes[1], **plot_kw)
        edge_quad.plot(ax=axes[0], ls="--", color="0.35")

        erlab.plotting.proportional_colorbar(ax=axes[0])
        erlab.plotting.proportional_colorbar(ax=axes[1])
        axes[0].set_title("Data")
        axes[1].set_title("Edge Corrected")

    return corrected


def edge(
    gold: xr.DataArray,
    *,
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    bkg_slope: bool = True,
    resolution: float = 0.02,
    fast: bool = False,
    method: str = "least_squares",
    scale_covar: bool = True,
    normalize: bool = True,
    fixed_center: float | None = None,
    progress: bool = True,
    parallel_kw: dict | None = None,
    parallel_obj: joblib.Parallel | None = None,
    return_full: bool = False,
    **kwargs,
) -> tuple[xr.DataArray, xr.DataArray] | xr.Dataset:
    """
    Fit a Fermi edge to the given gold data.

    Only successful fits with valid error estimates are returned.

    Parameters
    ----------
    gold
        The gold data to fit the edge model to.
    angle_range
        The range of alpha values to consider.
    eV_range
        The range of eV values to consider.
    bin_size
        The bin size for coarsening the gold data, by default (1, 1).
    temp
        The temperature in Kelvins. If `None`, the temperature is inferred from the
        attributes, by default `None`
    vary_temp
        Whether to fit the temperature value during fitting, by default `False`.
    bkg_slope
        Whether to include a linear background above the Fermi level. If `False`, the
        background above the Fermi level is fit with a constant. Defaults to `True`.
    resolution
        The initial resolution value to use for fitting, by default `0.02`.
    fast
        Whether to use the Gaussian-broadeded step function to fit the edge, by default
        `False`.
    method
        The fitting method to use, by default ``"least_squares"``.
    scale_covar
        Whether to scale the covariance matrix, by default `True`.
    fixed_center
        The fixed center value. If provided, the Fermi level will be fixed at the given
        value, by default `None`.
    normalize
        Whether to normalize the energy coordinates, by default `True`.
    progress
        Whether to display the fitting progress, by default `True`.
    parallel_kw
        Additional keyword arguments for parallel fitting, by default `None`.
    parallel_obj
        The `joblib.Parallel` object to use for fitting, by default `None`. If provided,
        `parallel_kw` will be ignored.
    return_full
        Whether to return the full fit results, by default `False`.
    **kwargs
        Additional keyword arguments to fitting.

    Returns
    -------
    center_arr, center_stderr
        The fitted center values and their standard errors, returned when `return_full`
        is `False`.
    fit_result
        A dataset containing the full fit results, returned when `return_full` is
        `True`.

    """
    if any(b != 1 for b in bin_size):
        gold_binned = gold.coarsen(alpha=bin_size[0], eV=bin_size[1], boundary="trim")
        gold = gold_binned.mean()  # type: ignore[attr-defined]

    gold_sel = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))

    if normalize:
        # Normalize energy coordinates
        avgx, stdx = gold_sel.eV.values.mean(), gold_sel.eV.values.std()
        gold_sel = gold_sel.assign_coords(eV=(gold_sel.eV - avgx) / stdx)

    if temp is None:
        temp = gold.qinfo.get_value("sample_temp")
        if temp is None:
            if fast:
                temp = 10.0
            else:
                raise ValueError(
                    "Temperature not found in data attributes, please provide manually"
                )

    if normalize and temp is not None:
        temp = float(temp / stdx)

    if fast:
        params = lmfit.create_params(
            sigma=(resolution + 3.53 * erlab.constants.kb_eV * temp)
            / np.sqrt(8 * np.log(2))
        )
        model_cls: lmfit.Model = erlab.analysis.fit.models.StepEdgeModel
    else:
        params = lmfit.create_params(
            temp={"value": float(temp), "vary": vary_temp}, resolution=resolution
        )
        model_cls = erlab.analysis.fit.models.FermiEdgeModel

    model = model_cls()

    if parallel_kw is None:
        parallel_kw = {}

    if not bkg_slope:
        params["back1"] = lmfit.Parameter("back1", value=0, vary=False)

    if fixed_center is not None:
        params["back1"] = lmfit.Parameter("center", value=fixed_center, vary=False)

    # Assuming Poisson noise, the weights are the square root of the counts.
    weights = 1 / np.sqrt(np.asarray(gold_sel.sum("eV").values))

    n_fits = len(gold_sel.alpha)

    if parallel_obj is None:
        if n_fits > 20:
            parallel_kw.setdefault("n_jobs", -1)
        else:
            parallel_kw.setdefault("n_jobs", 1)

        parallel_kw.setdefault("max_nbytes", None)
        parallel_kw.setdefault("return_as", "generator")
        parallel_kw.setdefault("pre_dispatch", "n_jobs")

        parallel_obj = joblib.Parallel(**parallel_kw)

    def _fit(data, w):
        pars = model.guess(data, x=data["eV"]).update(params)

        return data.xlm.modelfit(
            "eV",
            model=model,
            params=pars,
            method=method,
            scale_covar=scale_covar,
            weights=w,
            **kwargs,
        )

    tqdm_kw = {"desc": "Fitting", "total": n_fits, "disable": not progress}

    if parallel_obj.return_generator:
        fit_result = tqdm.tqdm(
            parallel_obj(
                joblib.delayed(_fit)(gold_sel.isel(alpha=i), weights[i])
                for i in range(n_fits)
            ),
            **tqdm_kw,
        )
    elif progress:
        with erlab.utils.parallel.joblib_progress(**tqdm_kw) as _:
            fit_result = parallel_obj(
                joblib.delayed(_fit)(gold_sel.isel(alpha=i), weights[i])
                for i in range(n_fits)
            )
    else:
        fit_result = parallel_obj(
            joblib.delayed(_fit)(gold_sel.isel(alpha=i), weights[i])
            for i in range(n_fits)
        )

    fit_result = xr.concat(fit_result, "alpha")

    if return_full:
        return fit_result

    xval: list[npt.NDArray] = []
    res_vals = []

    for i, r in enumerate(fit_result.modelfit_results.values):
        if hasattr(r, "uvars"):
            center_ufloat = r.uvars["center"]

            if normalize:
                center_ufloat = center_ufloat * stdx + avgx

            if not np.isnan(center_ufloat.std_dev):
                xval.append(gold_sel.alpha.values[i])
                res_vals.append([center_ufloat.nominal_value, center_ufloat.std_dev])

    if len(res_vals) == 0:
        erlab.utils.misc.emit_user_level_warning(
            "No valid fits found, returning empty arrays"
        )
        return xr.DataArray([], dims=["alpha"]), xr.DataArray([], dims=["alpha"])

    coords = {"alpha": np.asarray(xval)}
    yval, yerr = np.asarray(res_vals).T

    return xr.DataArray(yval, coords=coords), xr.DataArray(yerr, coords=coords)


def poly_from_edge(
    center, weights=None, degree=4, method="least_squares", scale_covar=True
) -> xr.Dataset:
    model = erlab.analysis.fit.models.PolynomialModel(degree=degree)
    return center.xlm.modelfit(
        "alpha",
        model=model,
        params=model.guess(center.values, x=center["alpha"].values),
        weights=np.asarray(weights),
        method=method,
        scale_covar=scale_covar,
        output_result=True,
    )


def spline_from_edge(
    center, weights: npt.ArrayLike | None = None, lam: float | None = None
) -> scipy.interpolate.BSpline:
    return scipy.interpolate.make_smoothing_spline(
        center.alpha.values, center.values, w=np.asarray(weights), lam=lam
    )


def _plot_gold_fit(
    fig, gold, angle_range, eV_range, center_arr, center_stderr, res
) -> None:
    if isinstance(res, xr.Dataset) and "modelfit_results" in res:
        is_callable = False
        res = res.modelfit_results.values.item()
    elif isinstance(res, lmfit.model.ModelResult):
        is_callable = False
    elif callable(res):
        is_callable = True
    else:
        raise TypeError(
            "res must be one of callable, lmfit.model.ModelResult, "
            "and fit result dataset"
        )

    if not isinstance(fig, plt.Figure):
        fig = plt.figure(figsize=erlab.plotting.figwh(0.75, wscale=1.75))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    plt.tick_params("x", labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)

    if gold.dims[0] == "eV":
        gold.qplot(ax=ax0, cmap="copper", gamma=0.5)
    else:
        gold.T.qplot(ax=ax0, cmap="copper", gamma=0.5)

    rect = matplotlib.patches.Rectangle(
        (angle_range[0], eV_range[0]),
        np.diff(angle_range)[0],
        np.diff(eV_range)[0],
        ec="w",
        alpha=0.5,
        lw=0.75,
        fc="none",
    )
    ax0.add_patch(rect)
    ax0.errorbar(
        center_arr.alpha,
        center_arr,
        center_stderr,
        fmt="o",
        lw=0.5,
        mfc="w",
        zorder=0,
        ms=2,
    )

    if is_callable:
        ax0.plot(gold.alpha, res(gold.alpha), "r-", lw=0.75)
    else:
        ax0.plot(gold.alpha, res.eval(res.params, x=gold.alpha), "r-", lw=0.75)
    ax0.set_ylim(gold.eV[[0, -1]])

    data_kws = {"lw": 0.5, "ms": 2, "mfc": "w", "zorder": 0, "c": "0.4", "capsize": 0}
    fit_kws = {"c": "r", "lw": 0.75}

    if is_callable:
        residuals = res(center_arr.alpha.values) - center_arr.values
        x_eval = np.linspace(
            min(center_arr.alpha.values),
            max(center_arr.alpha.values),
            3 * len(center_arr.alpha),
        )
        ax1.axhline(0, **fit_kws)
        ax1.errorbar(
            center_arr.alpha,
            residuals,
            yerr=lmfit.model.propagate_err(
                center_arr.values, center_stderr.values, "abs"
            ),
            fmt="o",
            **data_kws,
        )
        ax1.set_ylabel("residuals")

        ax2.errorbar(
            center_arr.alpha,
            center_arr.values,
            yerr=lmfit.model.propagate_err(
                center_arr.values, center_stderr.values, "abs"
            ),
            fmt="o",
            label="data",
            **data_kws,
        )
        ax2.plot(x_eval, res(x_eval), "-", label="best fit", **fit_kws)
        ax2.legend()
        ax1.set_ylim(erlab.plotting.autoscale_to(residuals))
        ax2.set_ylim(erlab.plotting.autoscale_to(center_arr.values))
    else:
        res.plot_residuals(ax=ax1, data_kws=data_kws, fit_kws=fit_kws)
        res.plot_fit(
            ax=ax2,
            data_kws=data_kws,
            fit_kws=fit_kws,
            numpoints=3 * len(center_arr.alpha),
        )
        ax1.set_ylim(erlab.plotting.autoscale_to(res.eval() - res.data))
        ax2.set_ylim(erlab.plotting.autoscale_to(res.data))

    ax1.set_title("")
    ax2.set_title("")


def poly(
    gold: xr.DataArray,
    *,
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    bkg_slope: bool = True,
    resolution: float = 0.02,
    fast: bool = False,
    method: str = "least_squares",
    normalize: bool = True,
    degree: int = 4,
    correct: bool = False,
    crop_correct: bool = False,
    parallel_kw: dict | None = None,
    plot: bool = True,
    fig: matplotlib.figure.Figure | None = None,
    scale_covar: bool = True,
    scale_covar_edge: bool = True,
) -> xr.Dataset | tuple[xr.Dataset, xr.DataArray]:
    center_arr, center_stderr = typing.cast(
        "tuple[xr.DataArray, xr.DataArray]",
        edge(
            gold,
            angle_range=angle_range,
            eV_range=eV_range,
            bin_size=bin_size,
            temp=temp,
            vary_temp=vary_temp,
            bkg_slope=bkg_slope,
            resolution=resolution,
            fast=fast,
            method=method,
            normalize=normalize,
            parallel_kw=parallel_kw,
            scale_covar=scale_covar_edge,
        ),
    )

    results = poly_from_edge(
        center_arr,
        weights=1.0 / center_stderr,
        degree=degree,
        method=method,
        scale_covar=scale_covar,
    )
    if plot:
        _plot_gold_fit(
            fig, gold, angle_range, eV_range, center_arr, center_stderr, results
        )
    if correct:
        if crop_correct:
            gold = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))
        corr = correct_with_edge(gold, results, plot=False)
        return results, corr
    return results


def spline(
    gold: xr.DataArray,
    *,
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    bkg_slope: bool = True,
    resolution: float = 0.02,
    fast: bool = False,
    method: str = "least_squares",
    lam: float | None = None,
    correct: bool = False,
    crop_correct: bool = False,
    parallel_kw: dict | None = None,
    plot: bool = True,
    fig: matplotlib.figure.Figure | None = None,
    scale_covar_edge: bool = True,
) -> scipy.interpolate.BSpline | tuple[scipy.interpolate.BSpline, xr.DataArray]:
    center_arr, center_stderr = typing.cast(
        "tuple[xr.DataArray, xr.DataArray]",
        edge(
            gold,
            angle_range=angle_range,
            eV_range=eV_range,
            bin_size=bin_size,
            temp=temp,
            vary_temp=vary_temp,
            bkg_slope=bkg_slope,
            resolution=resolution,
            fast=fast,
            method=method,
            parallel_kw=parallel_kw,
            scale_covar=scale_covar_edge,
        ),
    )

    spl = spline_from_edge(center_arr, weights=1 / center_stderr, lam=lam)
    if plot:
        _plot_gold_fit(fig, gold, angle_range, eV_range, center_arr, center_stderr, spl)
    if correct:
        if crop_correct:
            gold = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))
        corr = correct_with_edge(gold, spl, plot=False)
        return spl, corr
    return spl


def quick_fit(
    darr: xr.DataArray,
    *,
    eV_range: tuple[float, float] | None = None,
    method: str = "leastsq",
    temp: float | None = None,
    resolution: float | None = None,
    center: float | None = None,
    fix_temp: bool = True,
    fix_center: bool = False,
    fix_resolution: bool = False,
    bkg_slope: bool = True,
    plot: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    plot_fit_kwargs: dict[str, typing.Any] | None = None,
    plot_data_kwargs: dict[str, typing.Any] | None = None,
    plot_line_kwargs: dict[str, typing.Any] | None = None,
    plot_span_kwargs: dict[str, typing.Any] | None = None,
    **kwargs,
) -> xr.Dataset:
    """Perform a Fermi edge fit on an EDC.

    This function is a convenient wrapper around :meth:`DataArray.xlm.modelfit` that
    fits a Fermi edge to the given data.

    If data with 2 or more dimensions is provided, the data is averaged over all
    dimensions except the energy prior to fitting.

    Parameters
    ----------
    darr
        The input data to be fitted.
    eV_range
        The energy range to consider for fitting. If `None`, the entire energy range is
        used. Defaults to `None`.
    method
        The fitting method to use that is compatible with `lmfit`. Defaults to
        "leastsq".
    temp
        The temperature value to use for fitting. If `None`, the temperature is inferred
        from the data attributes.
    resolution
        The initial resolution value to use for fitting. If `None`, the resolution is
        set to 0.02, or to the ``'TotalResolution'`` attribute if present.
    center
        The initial center value to use for fitting. If `None`, the center is
        automatically guessed if `fix_center` is `False`. Otherwise, the center is fixed
        to 0.
    fix_temp
        Whether to fix the temperature value during fitting. Defaults to `True`.
    fix_center
        Whether to fix the Fermi level during fitting. If `True`, the Fermi level is
        fixed to 0. Defaults to `False`.
    fix_resolution
        Whether to fix the resolution value during fitting. Defaults to `False`.
    bkg_slope
        Whether to include a linear background above the Fermi level. If `False`, the
        background above the Fermi level is fit with a constant. Defaults to `True`.
    plot
        Whether to plot the result of the fit. Defaults to `False`.
    ax
        The axes to plot the result on if ``plot`` is `True`. If `None`, the current
        axes are used.
    plot_fit_kwargs
        Additional keyword arguments for the fit plot, passed to
        :meth:`matplotlib.axes.Axes.plot`. Defaults to `None`.
    plot_data_kwargs
        Additional keyword arguments for the data plot, passed to
        :meth:`matplotlib.axes.Axes.plot`. Defaults to `None`.
    plot_line_kwargs
        Additional keyword arguments for the plot line that indicates the fitted center,
        passed to :meth:`matplotlib.axes.Axes.axvline`. Defaults to `None`.
    plot_span_kwargs
        Additional keyword arguments for the plot span that indicates the fitted FWHM,
        passed to :meth:`matplotlib.axes.Axes.axvspan`. Defaults to `None`.
    **kwargs
        Additional keyword arguments to :meth:`DataArray.xlm.modelfit`.

    Returns
    -------
    result : xarray.Dataset
        The result of the fit.

    """
    with xr.set_options(keep_attrs=True):
        data = darr.mean([d for d in darr.dims if d != "eV"])
        data_fit = data.sel(eV=slice(*eV_range)) if eV_range is not None else data

    if temp is None:
        temp = data.qinfo.get_value("sample_temp")
        if temp is None:
            raise ValueError(
                "Temperature not found in data attributes, please provide manually"
            )

    if resolution is None:
        if "TotalResolution" in data.attrs:
            resolution = float(data.attrs["TotalResolution"]) * 1e-3
        else:
            resolution = 0.02

    params = {
        "temp": {"value": temp, "vary": not fix_temp, "min": 0},
        "resolution": {"value": resolution, "vary": not fix_resolution, "min": 0},
    }

    if not bkg_slope:
        params["back1"] = {"value": 0, "vary": False}

    if center is not None:
        params["center"] = {"value": center}

    if fix_center:
        if center is None:
            params["center"] = {"value": 0.0}
        params["center"]["vary"] = False

    kwargs.setdefault("guess", True)
    fit_result = data_fit.xlm.modelfit(
        "eV",
        model=erlab.analysis.fit.models.FermiEdgeModel(),
        method=method,
        params=params,
        **kwargs,
    )
    if plot:
        if ax is None:
            ax = plt.gca()

        _plot_resolution_fit(
            ax=ax,
            data=data_fit,
            result=fit_result,
            fix_center=fix_center,
            plot_fit_kwargs=plot_fit_kwargs,
            plot_data_kwargs=plot_data_kwargs,
            plot_line_kwargs=plot_line_kwargs,
            plot_span_kwargs=plot_span_kwargs,
        )

    return fit_result


def _plot_resolution_fit(
    ax: matplotlib.axes.Axes,
    data: xr.DataArray,
    result: xr.Dataset,
    fix_center: bool,
    plot_fit_kwargs: dict[str, typing.Any] | None = None,
    plot_data_kwargs: dict[str, typing.Any] | None = None,
    plot_line_kwargs: dict[str, typing.Any] | None = None,
    plot_span_kwargs: dict[str, typing.Any] | None = None,
) -> None:
    """Plot the results of a single Fermi edge fit."""
    plot_data_kwargs = {} if plot_data_kwargs is None else plot_data_kwargs
    plot_fit_kwargs = {} if plot_fit_kwargs is None else plot_fit_kwargs
    plot_line_kwargs = {} if plot_line_kwargs is None else plot_line_kwargs
    plot_span_kwargs = {} if plot_span_kwargs is None else plot_span_kwargs

    plot_data_kwargs["ls"] = plot_data_kwargs.pop(
        "ls", plot_data_kwargs.pop("linestyle", "none")
    )
    plot_data_kwargs["ms"] = plot_data_kwargs.pop(
        "ms", plot_data_kwargs.pop("markersize", 5)
    )
    plot_data_kwargs["mew"] = plot_data_kwargs.pop(
        "mew", plot_data_kwargs.pop("markeredgewidth", 0.4)
    )
    plot_data_kwargs["mec"] = plot_data_kwargs.pop(
        "mec", plot_data_kwargs.pop("markeredgecolor", "0.5")
    )
    plot_data_kwargs["mfc"] = plot_data_kwargs.pop(
        "mfc", plot_data_kwargs.pop("markerfacecolor", "none")
    )
    plot_data_kwargs.setdefault("marker", ".")
    plot_data_kwargs.setdefault("label", "Data")
    ax.plot(data.eV, data, **plot_data_kwargs)

    plot_fit_kwargs["c"] = plot_fit_kwargs.pop(
        "c", plot_fit_kwargs.pop("color", "tab:red")
    )
    plot_fit_kwargs.setdefault("label", "Fit")
    ax.plot(result.modelfit_best_fit.eV, result.modelfit_best_fit, **plot_fit_kwargs)

    ax.set_ylabel("Intensity (arb. units)")
    if (data.eV[0] * data.eV[-1]) < 0:
        ax.set_xlabel("$E - E_F$ (eV)")
    else:
        ax.set_xlabel(r"$E_{kin}$ (eV)")

    coeffs = result.modelfit_coefficients
    modelresult: lmfit.model.ModelResult = result.modelfit_results.item()

    if hasattr(modelresult, "uvars"):
        center = modelresult.uvars["center"]
        resolution = modelresult.uvars["resolution"]
        center_bounds = ((center - resolution).n, (center + resolution).n)

        center_repr = (
            f"$E_F = {center * 1e3:L}$ meV"
            if center < 0.1
            else f"$E_F = {center:L}$ eV"
        )
        resolution_repr = f"$\\Delta E = {resolution * 1e3:L}$ meV"

    else:
        center = coeffs.sel(param="center")
        resolution = coeffs.sel(param="resolution")
        center_bounds = (center - resolution, center + resolution)

        center_repr = (
            f"$E_F = {center * 1e3:.3f}$ meV"
            if center < 0.1
            else f"$E_F = {center:.6f}$ eV"
        )
        resolution_repr = f"$\\Delta E = {resolution * 1e3:.3f}$ meV"

    info_list: list[str] = [
        f"$T ={coeffs.sel(param='temp'):.3f}$ K",
        center_repr,
        resolution_repr,
    ]

    if fix_center:
        info_list.pop(1)

    fig = ax.figure
    if fig is not None:
        ax.text(
            0,
            0,
            "\n".join(info_list),
            ha="left",
            va="baseline",
            transform=ax.transAxes
            + matplotlib.transforms.ScaledTranslation(
                6 / 72, 6 / 72, fig.dpi_scale_trans
            ),
        )
    ax.set_xlim(data.eV[[0, -1]])
    ax.set_title("")

    plot_line_kwargs["c"] = plot_line_kwargs.pop(
        "c", plot_line_kwargs.pop("color", "k")
    )
    plot_line_kwargs["ls"] = plot_line_kwargs.pop(
        "ls", plot_line_kwargs.pop("linestyle", "--")
    )
    plot_line_kwargs["lw"] = plot_line_kwargs.pop(
        "lw", plot_line_kwargs.pop("linewidth", 0.4)
    )
    plot_line_kwargs.setdefault("alpha", 0.5)
    ax.axvline(coeffs.sel(param="center"), **plot_line_kwargs)

    plot_span_kwargs["fc"] = plot_span_kwargs.pop(
        "fc", plot_span_kwargs.pop("facecolor", "tab:red")
    )
    plot_span_kwargs["ec"] = plot_span_kwargs.pop(
        "ec", plot_span_kwargs.pop("edgecolor", "none")
    )
    plot_span_kwargs.setdefault("label", "FWHM")
    plot_span_kwargs.setdefault("alpha", 0.2)
    plot_span_kwargs.setdefault("ymin", -0.01)
    plot_span_kwargs.setdefault("ymax", 1.01)
    ax.axvspan(*center_bounds, **plot_span_kwargs)


def quick_resolution(
    darr: xr.DataArray, ax: matplotlib.axes.Axes | None = None, **kwargs
) -> xr.Dataset:
    """Fit a Fermi edge to the given data and plot the results.

    .. deprecated:: 3.5.1

        Use :func:`quick_fit` with ``plot=True`` instead.

    """
    warnings.warn(
        "erlab.analysis.gold.quick_resolution is deprecated, "
        "use erlab.analysis.gold.quick_fit with plot=True instead",
        FutureWarning,
        stacklevel=1,
    )

    kwargs["plot"] = True
    kwargs["ax"] = ax
    return quick_fit(darr, **kwargs)


def resolution(
    gold: xr.DataArray,
    angle_range: tuple[float, float],
    eV_range_edge: tuple[float, float],
    eV_range_fit: tuple[float, float] | None = None,
    bin_size: tuple[int, int] = (1, 1),
    degree: int = 4,
    fast: bool = False,
    method: str = "leastsq",
    plot: bool = True,
    parallel_kw: dict | None = None,
    scale_covar: bool = True,
) -> lmfit.model.ModelResult:  # pragma: no cover
    """Fit a Fermi edge and obtain the resolution from the corrected data.

    .. deprecated:: 3.5.1

        Use :func:`poly` and :func:`quick_fit` instead.

    """
    warnings.warn(
        "erlab.analysis.gold.resolution is deprecated, "
        "use erlab.analysis.gold.quick_fit instead",
        FutureWarning,
        stacklevel=1,
    )

    pol, gold_corr = typing.cast(
        "tuple[xr.Dataset, xr.DataArray]",
        poly(
            gold,
            angle_range=angle_range,
            eV_range=eV_range_edge,
            bin_size=bin_size,
            degree=degree,
            correct=True,
            fast=fast,
            method=method,
            plot=plot,
            parallel_kw=parallel_kw,
        ),
    )

    if eV_range_fit is None:
        eV_range_fit = tuple(r - np.mean(pol.best_fit) for r in eV_range_edge)
    del pol
    gold_roi = gold_corr.sel(alpha=slice(*angle_range))
    edc_avg = gold_roi.mean("alpha").sel(eV=slice(*eV_range_fit))

    params = lmfit.create_params(
        temp={"value": gold_roi.attrs["sample_temp"], "vary": False},
        resolution={"value": 0.1, "vary": True, "min": 0},
    )
    model = erlab.analysis.fit.models.FermiEdgeModel()
    params = model.guess(edc_avg, x=edc_avg["eV"]).update(params)
    fit = erlab.analysis.fit.models.FermiEdgeModel().fit(
        edc_avg, x=edc_avg["eV"], params=params, method=method, scale_covar=scale_covar
    )
    if plot:
        plt.show()
        ax = plt.gca()
        gold_corr.qplot(ax=ax, cmap="copper", gamma=0.5)
        rect = matplotlib.patches.Rectangle(
            (angle_range[0], eV_range_fit[0]),
            np.diff(angle_range)[0],
            np.diff(eV_range_fit)[0],
            ec="w",
            alpha=0.5,
            lw=0.75,
            fc="none",
        )
        ax.add_patch(rect)
        ax.set_ylim(gold_corr.eV[[0, -1]])

        fit.plot(
            data_kws={"lw": 0.75, "ms": 4, "mfc": "w", "zorder": 0, "c": "0.4"},
            fit_kws={"c": "r", "lw": 1.5},
        )

    if hasattr(fit, "uvars"):
        center_uf = fit.uvars["center"]
        res_uf = fit.uvars["resolution"]
        print(f"center = {center_uf:S} eV\nresolution = {res_uf:.4S} eV")
    else:
        print(
            f"center = {fit.params['center'].value} eV\n"
            f"resolution = {fit.params['resolution'].value} eV"
        )

    return fit


def resolution_roi(
    gold_roi: xr.DataArray,
    eV_range: tuple[float, float],
    fix_temperature: bool = True,
    method: str = "leastsq",
    plot: bool = True,
    scale_covar: bool = True,
) -> lmfit.model.ModelResult:  # pragma: no cover
    """Fit a Fermi edge to the data and obtain the resolution.

    .. deprecated:: 3.5.1

        Use :func:`quick_fit` instead.

    """
    warnings.warn(
        "erlab.analysis.gold.resolution is deprecated, "
        "use erlab.analysis.gold.quick_fit instead",
        FutureWarning,
        stacklevel=1,
    )

    edc_avg = gold_roi.mean("alpha").sel(eV=slice(*eV_range))

    params = lmfit.create_params(
        temp={"value": gold_roi.attrs["sample_temp"], "vary": not fix_temperature},
        resolution={"value": 0.1, "vary": True, "min": 0},
    )
    model = erlab.analysis.fit.models.FermiEdgeModel()
    params = model.guess(edc_avg, x=edc_avg["eV"]).update(params)
    fit = model.fit(
        edc_avg,
        x=edc_avg["eV"],
        params=params,
        method=method,
        scale_covar=scale_covar,  # weights=1 / edc_stderr
    )
    if plot:
        ax = plt.gca()
        gold_roi.qplot(ax=ax, cmap="copper", gamma=0.5)
        ax.fill_between(
            gold_roi.alpha,
            eV_range[0],
            eV_range[1],
            ec="w",
            fc="none",
            alpha=0.4,
            lw=0.75,
        )
        ax.set_ylim(gold_roi.eV[[0, -1]])

        fit.plot(
            data_kws={"lw": 0.75, "ms": 4, "mfc": "w", "zorder": 0, "c": "0.4"},
            fit_kws={"c": "r", "lw": 1.5},
        )

    if hasattr(fit, "uvars"):
        center_uf = fit.uvars["center"]
        res_uf = fit.uvars["resolution"]
        print(f"center = {center_uf:S} eV\nresolution = {res_uf:.4S} eV")
    else:
        print(
            f"center = {fit.params['center'].value} eV\n"
            f"resolution = {fit.params['resolution'].value} eV"
        )

    return fit
