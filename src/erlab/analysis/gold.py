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

from collections.abc import Callable

import joblib
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
import tqdm.auto
import uncertainties
import xarray as xr

from erlab.analysis.fit.models import (
    FermiEdge2dModel,
    FermiEdgeModel,
    PolynomialModel,
    StepEdgeModel,
)
from erlab.analysis.utils import shift
from erlab.plotting.colors import proportional_colorbar
from erlab.plotting.general import autoscale_to, figwh, plot_array
from erlab.utils.parallel import joblib_progress


def correct_with_edge(
    darr: xr.DataArray,
    modelresult: lmfit.model.ModelResult | npt.NDArray[np.floating] | Callable,
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
        angle, or a callable function that takes an array of angles and returns the
        corresponding energy value.
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
        Additional keyword arguments to `erlab.analysis.utils.shift`.

    Returns
    -------
    corrected : xarray.DataArray
        The edge corrected data.
    """
    if plot_kw is None:
        plot_kw = {}

    if isinstance(modelresult, lmfit.model.ModelResult):
        if isinstance(modelresult.model, FermiEdge2dModel):
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
        else:
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

    corrected = shift(darr, -edge_quad, "eV", shift_coords=shift_coords, **shift_kwargs)

    if plot is True:
        _, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))

        plot_kw.setdefault("cmap", "copper")
        plot_kw.setdefault("gamma", 0.5)

        if darr.ndim > 2:
            avg_dims = list(darr.dims)[:]
            avg_dims.remove("alpha")
            avg_dims.remove("eV")
            plot_array(darr.mean(avg_dims), ax=axes[0], **plot_kw)
            plot_array(corrected.mean(avg_dims), ax=axes[1], **plot_kw)
        else:
            plot_array(darr, ax=axes[0], **plot_kw)
            plot_array(corrected, ax=axes[1], **plot_kw)
        edge_quad.plot(ax=axes[0], ls="--", color="0.35")

        proportional_colorbar(ax=axes[0])
        proportional_colorbar(ax=axes[1])
        axes[0].set_title("Data")
        axes[1].set_title("Edge Corrected")

    return corrected


def edge(
    gold: xr.DataArray,
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    fast: bool = False,
    method: str = "leastsq",
    scale_covar: bool = True,
    normalize: bool = True,
    fixed_center: float | None = None,
    progress: bool = True,
    parallel_kw: dict | None = None,
    parallel_obj: joblib.Parallel | None = None,
    return_full: bool = False,
    **kwargs,
) -> tuple[xr.DataArray, xr.DataArray] | list[lmfit.model.ModelResult]:
    """
    Fit a Fermi edge to the given gold data.

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
    fast
        Whether to use the Gaussian-broadeded step function to fit the edge, by default
        `False`.
    method
        The fitting method to use, by default ``"leastsq"``.
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
    fitresults
        A dataset containing the full fit results, returned when `return_full` is
        `True`.

    """
    if fast:
        params = lmfit.create_params()
        model_cls: lmfit.Model = StepEdgeModel
    else:
        if temp is None:
            temp = gold.attrs["temp_sample"]
        params = lmfit.create_params(temp={"value": temp, "vary": vary_temp})
        model_cls = FermiEdgeModel

    model = model_cls()

    if parallel_kw is None:
        parallel_kw = {}

    if fixed_center is not None:
        params["center"].set(value=fixed_center, vary=False)

    if any(b != 1 for b in bin_size):
        gold_binned = gold.coarsen(alpha=bin_size[0], eV=bin_size[1], boundary="trim")
        gold = gold_binned.mean()  # type: ignore[attr-defined]

    gold_sel = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))

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

    if normalize:
        # Normalize energy coordinates
        avgx, stdx = gold_sel.eV.values.mean(), gold_sel.eV.values.std()
        gold_sel = gold_sel.assign_coords(eV=(gold_sel.eV - avgx) / stdx)

    def _fit(data, w):
        pars = model.guess(data, x=data["eV"]).update(params)

        res = model.fit(
            data,
            x=data["eV"],
            params=pars,
            method=method,
            scale_covar=scale_covar,
            weights=w,
            **kwargs,
        )
        return res

    tqdm_kw = {"desc": "Fitting", "total": n_fits, "disable": not progress}

    if parallel_obj.return_generator:
        fitresults = tqdm.auto.tqdm(  # type: ignore[call-overload]
            parallel_obj(
                joblib.delayed(_fit)(gold_sel.isel(alpha=i), weights[i])
                for i in range(n_fits)
            ),
            **tqdm_kw,
        )
    else:
        if progress:
            with joblib_progress(**tqdm_kw) as _:
                fitresults = parallel_obj(
                    joblib.delayed(_fit)(gold_sel.isel(alpha=i), weights[i])
                    for i in range(n_fits)
                )
        else:
            fitresults = parallel_obj(
                joblib.delayed(_fit)(gold_sel.isel(alpha=i), weights[i])
                for i in range(n_fits)
            )

    if return_full:
        return list(fitresults)

    xval: list[npt.NDArray] = []
    res_vals = []

    for i, r in enumerate(fitresults):
        if hasattr(r, "uvars"):
            center_ufloat = r.uvars["center"]

            if normalize:
                center_ufloat = center_ufloat * stdx + avgx

            if not np.isnan(center_ufloat.std_dev):
                xval.append(gold_sel.alpha.values[i])
                res_vals.append([center_ufloat.nominal_value, center_ufloat.std_dev])

    coords = {"alpha": np.asarray(xval)}
    yval, yerr = np.asarray(res_vals).T

    return xr.DataArray(yval, coords=coords), xr.DataArray(yerr, coords=coords)


def poly_from_edge(
    center, weights=None, degree=4, method="leastsq", scale_covar=True
) -> lmfit.model.ModelResult:
    model = PolynomialModel(degree=degree)
    pars = model.guess(center.values, x=center[center.dims[0]].values)
    modelresult = model.fit(
        center,
        x=center[center.dims[0]].values,
        params=pars,
        weights=weights,
        method=method,
        scale_covar=scale_covar,
    )
    return modelresult


def spline_from_edge(
    center, weights: npt.ArrayLike | None = None, lam: float | None = None
) -> scipy.interpolate.BSpline:
    spl = scipy.interpolate.make_smoothing_spline(
        center.alpha.values,
        center.values,
        w=np.asarray(weights),
        lam=lam,
    )
    return spl


def _plot_gold_fit(fig, gold, angle_range, eV_range, center_arr, center_stderr, res):
    if isinstance(res, lmfit.model.ModelResult):
        is_callable = False
    elif callable(res):
        is_callable = True
    else:
        raise TypeError("res must be a callable or a lmfit.model.ModelResult")

    if not isinstance(fig, plt.Figure):
        fig = plt.figure(figsize=figwh(0.75, wscale=1.75))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    plt.tick_params("x", labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)

    gold.qplot(ax=ax0, cmap="copper", gamma=0.5)
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
        ax1.set_ylim(autoscale_to(residuals))
        ax2.set_ylim(autoscale_to(center_arr.values))
    else:
        res.plot_residuals(
            ax=ax1,
            data_kws=data_kws,
            fit_kws=fit_kws,
        )
        res.plot_fit(
            ax=ax2,
            data_kws=data_kws,
            fit_kws=fit_kws,
            numpoints=3 * len(center_arr.alpha),
        )
        ax1.set_ylim(autoscale_to(res.eval() - res.data))
        ax2.set_ylim(autoscale_to(res.data))

    ax1.set_title("")
    ax2.set_title("")


def poly(
    gold: xr.DataArray,
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    fast: bool = False,
    method: str = "leastsq",
    normalize: bool = True,
    degree: int = 4,
    correct: bool = False,
    crop_correct: bool = False,
    parallel_kw: dict | None = None,
    plot: bool = True,
    fig: matplotlib.figure.Figure | None = None,
    scale_covar: bool = True,
    scale_covar_edge: bool = True,
) -> lmfit.model.ModelResult | tuple[lmfit.model.ModelResult, xr.DataArray]:
    center_arr, center_stderr = edge(
        gold,
        angle_range,
        eV_range,
        bin_size=bin_size,
        temp=temp,
        vary_temp=vary_temp,
        fast=fast,
        method=method,
        normalize=normalize,
        parallel_kw=parallel_kw,
        scale_covar=scale_covar_edge,
    )

    modelresult = poly_from_edge(
        center_arr,
        weights=1.0 / center_stderr,
        degree=degree,
        method=method,
        scale_covar=scale_covar,
    )
    if plot:
        _plot_gold_fit(
            fig, gold, angle_range, eV_range, center_arr, center_stderr, modelresult
        )
    if correct:
        if crop_correct:
            gold = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))
        corr = correct_with_edge(gold, modelresult, plot=False)
        return modelresult, corr
    else:
        return modelresult


def spline(
    gold: xr.DataArray,
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    fast: bool = False,
    method: str = "leastsq",
    lam: float | None = None,
    correct: bool = False,
    crop_correct: bool = False,
    parallel_kw: dict | None = None,
    plot: bool = True,
    fig: matplotlib.figure.Figure | None = None,
    scale_covar_edge: bool = True,
) -> scipy.interpolate.BSpline | tuple[scipy.interpolate.BSpline, xr.DataArray]:
    center_arr, center_stderr = edge(
        gold,
        angle_range,
        eV_range,
        bin_size=bin_size,
        temp=temp,
        vary_temp=vary_temp,
        fast=fast,
        method=method,
        parallel_kw=parallel_kw,
        scale_covar=scale_covar_edge,
    )

    spl = spline_from_edge(center_arr, weights=1 / center_stderr, lam=lam)
    if plot:
        _plot_gold_fit(fig, gold, angle_range, eV_range, center_arr, center_stderr, spl)
    if correct:
        if crop_correct:
            gold = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))
        corr = correct_with_edge(gold, spl, plot=False)
        return spl, corr
    else:
        return spl


def quick_fit(
    darr: xr.DataArray,
    eV_range: tuple[float, float] | None = None,
    method: str = "leastsq",
    temp: float | None = None,
    resolution: float | None = None,
    fix_temp: bool = True,
    fix_center: bool = False,
    fix_resolution: bool = False,
    bkg_slope: bool = True,
) -> xr.Dataset:
    """Perform a quick Fermi edge fit on the given data.

    The data is averaged over all dimensions except the energy prior to fitting.

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

    Returns
    -------
    result : xarray.Dataset
        The result of the fit.

    """
    data = darr.mean([d for d in darr.dims if d != "eV"])

    if eV_range is not None:
        data_fit = data.sel(eV=slice(*eV_range))
    else:
        data_fit = data

    if temp is None:
        if "temp_sample" in data.attrs:
            temp = float(data.attrs["temp_sample"])
        else:
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

    if fix_center:
        params["center"] = {"value": 0, "vary": False}

    return data_fit.modelfit(
        "eV", model=FermiEdgeModel(), method=method, params=params, guess=True
    )


def quick_resolution(
    darr: xr.DataArray,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> xr.Dataset:
    """Fit a Fermi edge to the given data and plot the results.

    This function is a wrapper around `quick_fit` that plots the data and the obtained
    resolution. The data is averaged over all dimensions except the energy prior to
    fitting.

    Parameters
    ----------
    darr
        The input data to be fitted.
    ax
        The axis to plot the data and fit on. If `None`, the current axis is used.
        Defaults to `None`.
    **kwargs
        Additional keyword arguments to `quick_fit`.

    Returns
    -------
    result : xarray.Dataset
        The result of the fit.

    """
    if ax is None:
        ax = plt.gca()

    darr = darr.mean([d for d in darr.dims if d != "eV"])
    result = quick_fit(darr, **kwargs)
    ax.plot(
        darr.eV, darr, ".", mec="0.6", alpha=1, mfc="none", ms=5, mew=0.3, label="Data"
    )

    result.modelfit_best_fit.qplot(ax=ax, c="r", label="Fit")

    ax.set_ylabel("Intensity (arb. units)")
    if (darr.eV[0] * darr.eV[-1]) < 0:
        ax.set_xlabel("$E - E_F$ (eV)")
    else:
        ax.set_xlabel(r"$E_{kin}$ (eV)")

    coeffs = result.modelfit_coefficients
    center = result.modelfit_results.item().uvars["center"]
    resolution = result.modelfit_results.item().uvars["resolution"]

    ax.text(
        0,
        0,
        "\n".join(
            [
                f"$T ={coeffs.sel(param='temp'):.3f}$ K",
                f"$E_F = {center * 1e3:L}$ meV"
                if center < 0.1
                else f"$E_F = {center:L}$ eV",
                f"$\\Delta E = {resolution * 1e3:L}$ meV",
            ]
        ),
        ha="left",
        va="baseline",
        transform=ax.transAxes
        + matplotlib.transforms.ScaledTranslation(
            6 / 72, 6 / 72, ax.figure.dpi_scale_trans
        ),
    )
    ax.set_xlim(darr.eV[[0, -1]])
    ax.set_title("")
    ax.axvline(coeffs.sel(param="center"), ls="--", c="k", lw=0.4, alpha=0.5)
    ax.axvspan(
        (center - resolution).n,
        (center + resolution).n,
        color="r",
        alpha=0.2,
        label="FWHM",
    )
    return result


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
) -> lmfit.model.ModelResult:
    pol, gold_corr = poly(
        gold,
        angle_range,
        eV_range_edge,
        bin_size=bin_size,
        degree=degree,
        correct=True,
        fast=fast,
        method=method,
        plot=plot,
        parallel_kw=parallel_kw,
    )

    if eV_range_fit is None:
        eV_range_fit = tuple(r - np.mean(pol.best_fit) for r in eV_range_edge)
    del pol
    gold_roi = gold_corr.sel(alpha=slice(*angle_range))
    edc_avg = gold_roi.mean("alpha").sel(eV=slice(*eV_range_fit))

    params = lmfit.create_params(
        temp={"value": gold_roi.attrs["temp_sample"], "vary": False},
        resolution={"value": 0.1, "vary": True, "min": 0},
    )
    model = FermiEdgeModel()
    params = model.guess(edc_avg, x=edc_avg["eV"]).update(params)
    fit = FermiEdgeModel().fit(
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

    center_uf = uncertainties.ufloat(fit.params["center"], fit.params["center"].stderr)
    res_uf = uncertainties.ufloat(
        fit.params["resolution"], fit.params["resolution"].stderr
    )
    print(f"center = {center_uf:S} eV\nresolution = {res_uf:.4S} eV")
    return fit


def resolution_roi(
    gold_roi: xr.DataArray,
    eV_range: tuple[float, float],
    fix_temperature: bool = True,
    method: str = "leastsq",
    plot: bool = True,
    scale_covar: bool = True,
) -> lmfit.model.ModelResult:
    edc_avg = gold_roi.mean("alpha").sel(eV=slice(*eV_range))

    params = lmfit.create_params(
        temp={"value": gold_roi.attrs["temp_sample"], "vary": not fix_temperature},
        resolution={"value": 0.1, "vary": True, "min": 0},
    )
    model = FermiEdgeModel()
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

    center_uf = uncertainties.ufloat(fit.params["center"], fit.params["center"].stderr)
    res_uf = uncertainties.ufloat(
        fit.params["resolution"], fit.params["resolution"].stderr
    )
    print(f"center = {center_uf:S} eV\nresolution = {res_uf:.4S} eV")
    return fit
