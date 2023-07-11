"""Fermi edge fitting.

"""

__all__ = [
    "edge",
    "poly",
    "poly_from_edge",
    "spline_from_edge",
    "resolution",
    "resolution_roi",
]

import arpes
import arpes.fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from matplotlib.patches import Rectangle
from uncertainties import ufloat
import lmfit.model

from erlab.analysis.fit.models import (
    ExtendedAffineBroadenedFD,
    PolynomialModel,
    StepEdgeModel,
)
from erlab.analysis.utilities import correct_with_edge
from erlab.plotting.general import autoscale_to, figwh


def edge(
    gold,
    phi_range,
    eV_range,
    bin_size=(1, 1),
    temp=None,
    vary_temp=False,
    fast=False,
    method="leastsq",
    progress=True,
    parallel_kw=dict(),
    return_full=False,
    fixed_center=None,
    scale_covar=True,
):
    if fast:
        params = {}
        model = StepEdgeModel
    else:
        if temp is None:
            temp = gold.S.temp
        params = {
            "temp": dict(value=temp, vary=vary_temp),
        }
        model = ExtendedAffineBroadenedFD

    if fixed_center is not None:
        params["center"] = dict(value=fixed_center, vary=False)
    weights = None

    if any([b != 1 for b in bin_size]):
        gold_binned = gold.coarsen(phi=bin_size[0], eV=bin_size[1], boundary="trim")
        gold = gold_binned.mean()
        # gold_stderr = (gold_binned.std() / np.sqrt(np.prod(bin_size))).sel(
        #     phi=slice(*phi_range), eV=slice(*eV_range)
        # )
        # if (gold_stderr > 0).all():
        #     weights = 1 / gold_stderr

    gold_sel = gold.sel(phi=slice(*phi_range), eV=slice(*eV_range))

    fitresults = arpes.fits.broadcast_model(
        model,
        gold_sel,
        "phi",
        weights=weights,
        params=params,
        method=method,
        progress=progress,
        parallel_kw=parallel_kw,
        scale_covar=scale_covar,
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


def spline_from_edge(center, weights=None, lam=None):
    spl = scipy.interpolate.make_smoothing_spline(
        center.phi.values,
        center.values,
        w=np.asarray(weights),
        lam=lam,
    )
    return spl


def _plot_gold_fit(fig, gold, phi_range, eV_range, center_arr, center_stderr, res):
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

    gold.S.plot(ax=ax0, cmap="copper", gamma=0.5)
    rect = Rectangle(
        (phi_range[0], eV_range[0]),
        np.diff(phi_range)[0],
        np.diff(eV_range)[0],
        ec="w",
        alpha=0.5,
        lw=0.75,
        fc="none",
    )
    ax0.add_patch(rect)
    ax0.errorbar(
        center_arr.phi,
        center_arr,
        center_stderr,
        fmt="o",
        lw=0.5,
        mfc="w",
        zorder=0,
        ms=2,
    )

    if is_callable:
        ax0.plot(gold.phi, res(gold.phi), "r-", lw=0.75)
    else:
        ax0.plot(gold.phi, res.eval(res.params, x=gold.phi), "r-", lw=0.75)
    ax0.set_ylim(gold.eV[[0, -1]])

    data_kws = dict(lw=0.5, ms=2, mfc="w", zorder=0, c="0.4", capsize=0)
    fit_kws = dict(c="r", lw=0.75)

    if is_callable:
        residuals = res(center_arr.phi.values) - center_arr.values
        x_eval = np.linspace(
            min(center_arr.phi.values),
            max(center_arr.phi.values),
            3 * len(center_arr.phi),
        )
        ax1.axhline(0, **fit_kws)
        ax1.errorbar(
            center_arr.phi,
            residuals,
            yerr=lmfit.model.propagate_err(
                center_arr.values, center_stderr.values, "abs"
            ),
            fmt="o",
            **data_kws,
        )
        ax1.set_ylabel("residuals")

        ax2.errorbar(
            center_arr.phi,
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
            numpoints=3 * len(center_arr.phi),
        )
        ax1.set_ylim(autoscale_to(res.eval() - res.data))
        ax2.set_ylim(autoscale_to(res.data))

    ax1.set_title("")
    ax2.set_title("")


def poly(
    gold,
    phi_range,
    eV_range,
    bin_size=(1, 1),
    temp=None,
    vary_temp=False,
    fast=False,
    method="leastsq",
    degree=4,
    correct=False,
    crop_correct=False,
    parallel_kw=dict(),
    plot=True,
    fig=None,
    scale_covar=True,
    scale_covar_edge=True,
):
    center_arr, center_stderr = edge(
        gold,
        phi_range,
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
            fig, gold, phi_range, eV_range, center_arr, center_stderr, modelresult
        )
    if correct:
        if crop_correct:
            gold = gold.sel(phi=slice(*phi_range), eV=slice(*eV_range))
        corr = correct_with_edge(gold, modelresult, plot=False)
        return modelresult, corr
    else:
        return modelresult


def spline(
    gold,
    phi_range,
    eV_range,
    bin_size=(1, 1),
    temp=None,
    vary_temp=False,
    fast=False,
    method="leastsq",
    lam=None,
    correct=False,
    crop_correct=False,
    parallel_kw=dict(),
    plot=True,
    fig=None,
    scale_covar_edge=True,
):
    center_arr, center_stderr = edge(
        gold,
        phi_range,
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
        _plot_gold_fit(fig, gold, phi_range, eV_range, center_arr, center_stderr, spl)
    if correct:
        if crop_correct:
            gold = gold.sel(phi=slice(*phi_range), eV=slice(*eV_range))
        corr = correct_with_edge(gold, spl, plot=False)
        return spl, corr
    else:
        return spl


def resolution(
    gold,
    phi_range,
    eV_range_edge,
    eV_range_fit=None,
    bin_size=(1, 1),
    degree=4,
    fast=False,
    method="leastsq",
    plot=True,
    parallel_kw=dict(),
    scale_covar=True,
):
    pol, gold_corr = poly(
        gold,
        phi_range,
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
    gold_roi = gold_corr.sel(phi=slice(*phi_range))
    edc_avg = gold_roi.mean("phi").sel(eV=slice(*eV_range_fit))

    params = dict(
        temp=dict(value=gold.S.temp, vary=False),
        resolution=dict(value=0.1, vary=True, min=0),
    )
    fit = ExtendedAffineBroadenedFD().guess_fit(
        edc_avg, params=params, method=method, scale_covar=scale_covar
    )
    if plot:
        plt.show()
        ax = plt.gca()
        gold_corr.S.plot(ax=ax, cmap="copper", gamma=0.5)
        rect = Rectangle(
            (phi_range[0], eV_range_fit[0]),
            np.diff(phi_range)[0],
            np.diff(eV_range_fit)[0],
            ec="w",
            alpha=0.5,
            lw=0.75,
            fc="none",
        )
        ax.add_patch(rect)
        ax.set_ylim(gold_corr.eV[[0, -1]])

        fit.plot(
            data_kws=dict(lw=0.75, ms=4, mfc="w", zorder=0, c="0.4"),
            fit_kws=dict(c="r", lw=1.5),
        )

    center_uf = ufloat(fit.params["center"], fit.params["center"].stderr)
    res_uf = ufloat(fit.params["resolution"], fit.params["resolution"].stderr)
    print(f"center = {center_uf:S} eV\n" f"resolution = {res_uf:.4S} eV")
    return fit


def resolution_roi(
    gold_roi,
    eV_range,
    fix_temperature=True,
    method="leastsq",
    plot=True,
    scale_covar=True,
):
    edc_avg = gold_roi.mean("phi").sel(eV=slice(*eV_range))

    params = dict(
        temp=dict(value=gold_roi.S.temp, vary=not fix_temperature),
        resolution=dict(value=0.1, vary=True, min=0),
    )
    fit = ExtendedAffineBroadenedFD().guess_fit(
        edc_avg,
        params=params,
        method=method,
        scale_covar=scale_covar,  # weights=1 / edc_stderr
    )
    if plot:
        ax = plt.gca()
        gold_roi.S.plot(ax=ax, cmap="copper", gamma=0.5)
        ax.fill_between(
            gold_roi.phi,
            eV_range[0],
            eV_range[1],
            ec="w",
            fc="none",
            alpha=0.4,
            lw=0.75,
        )
        ax.set_ylim(gold_roi.eV[[0, -1]])

        fit.plot(
            data_kws=dict(lw=0.75, ms=4, mfc="w", zorder=0, c="0.4"),
            fit_kws=dict(c="r", lw=1.5),
        )

    center_uf = ufloat(fit.params["center"], fit.params["center"].stderr)
    res_uf = ufloat(fit.params["resolution"], fit.params["resolution"].stderr)
    print(f"center = {center_uf:S} eV\n" f"resolution = {res_uf:.4S} eV")
    return fit
