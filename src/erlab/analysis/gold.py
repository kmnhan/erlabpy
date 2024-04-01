"""Fermi edge fitting."""

__all__ = [
    "edge",
    "poly",
    "poly_from_edge",
    "spline_from_edge",
    "resolution",
    "resolution_roi",
]

from collections.abc import Sequence

import arpes
import arpes.fits
import lmfit.model
import matplotlib
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import uncertainties
import xarray as xr

from erlab.analysis.fit.models import (
    ExtendedAffineBroadenedFD,
    PolynomialModel,
    StepEdgeModel,
)
from erlab.analysis.utilities import correct_with_edge
from erlab.plotting.general import autoscale_to, figwh


def edge(
    gold: xr.DataArray | xr.Dataset,
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    fast: bool = False,
    method: str = "leastsq",
    progress: bool = True,
    parallel_kw: dict | None = None,
    return_full: bool = False,
    fixed_center: float | None = None,
    scale_covar: bool = True,
    **kwargs: dict,
) -> tuple[npt.NDArray, npt.NDArray] | xr.Dataset:
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
    progress
        Whether to display the fitting progress, by default `True`.
    parallel_kw
        Additional keyword arguments for parallel fitting, by default `None`.
    return_full
        Whether to return the full fit results, by default `False`.
    fixed_center
        The fixed center value. If provided, the Fermi level will be fixed at the given
        value, by default `None`.
    scale_covar
        Whether to scale the covariance matrix, by default `True`.
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
        params = {}
        model_cls = StepEdgeModel
    else:
        if temp is None:
            temp = gold.attrs["temp_sample"]
        params = {
            "temp": {"value": temp, "vary": vary_temp},
        }
        model_cls = ExtendedAffineBroadenedFD

    if parallel_kw is None:
        parallel_kw = {}

    if fixed_center is not None:
        params["center"] = {"value": fixed_center, "vary": False}

    if any(b != 1 for b in bin_size):
        gold_binned = gold.coarsen(alpha=bin_size[0], eV=bin_size[1], boundary="trim")
        gold = gold_binned.mean()

    gold_sel = gold.sel(alpha=slice(*angle_range), eV=slice(*eV_range))

    # Assuming Poisson noise, the weights are the square root of the counts.
    weights = np.sqrt(gold.sum("eV"))

    fitresults = arpes.fits.broadcast_model(
        model_cls,
        gold_sel,
        "alpha",
        weights=weights,
        params=params,
        method=method,
        progress=progress,
        parallel_kw=parallel_kw,
        scale_covar=scale_covar,
        **kwargs,
    )
    if return_full:
        return fitresults

    center_arr, center_stderr = fitresults.F.p("center"), fitresults.F.s("center")
    center_arr = center_arr.where(~center_stderr.isnull(), drop=True)
    center_stderr = center_stderr.where(~center_stderr.isnull(), drop=True)
    return center_arr, center_stderr


def poly_from_edge(
    center, weights=None, degree=4, method="leastsq", scale_covar=True
) -> lmfit.model.ModelResult:
    modelresult = PolynomialModel(degree=degree).guess_fit(
        center, weights=weights, method=method, scale_covar=scale_covar
    )
    return modelresult


def spline_from_edge(
    center, weights: Sequence[float] | None = None, lam: float | None = None
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
        raise ValueError

    if not isinstance(fig, plt.Figure):
        fig = plt.figure(figsize=figwh(0.75, wscale=1.75))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    plt.tick_params("x", labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)

    gold.qplot(ax=ax0, cmap="copper", gamma=0.5)
    rect = mpatches.Rectangle(
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
    gold: xr.DataArray | xr.Dataset,
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    fast: bool = False,
    method: str = "leastsq",
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
        parallel_kw=parallel_kw,
        scale_covar=scale_covar_edge,
    )

    modelresult = poly_from_edge(
        center_arr,
        weights=1 / center_stderr,
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
    gold: xr.DataArray | xr.Dataset,
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


def resolution(
    gold: xr.DataArray | xr.Dataset,
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

    params = {
        "temp": {"value": gold.attrs["temp_sample"], "vary": False},
        "resolution": {"value": 0.1, "vary": True, "min": 0},
    }
    fit = ExtendedAffineBroadenedFD().guess_fit(
        edc_avg, params=params, method=method, scale_covar=scale_covar
    )
    if plot:
        plt.show()
        ax = plt.gca()
        gold_corr.qplot(ax=ax, cmap="copper", gamma=0.5)
        rect = mpatches.Rectangle(
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

    params = {
        "temp": {"value": gold_roi.attrs["temp_sample"], "vary": not fix_temperature},
        "resolution": {"value": 0.1, "vary": True, "min": 0},
    }
    fit = ExtendedAffineBroadenedFD().guess_fit(
        edc_avg,
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
