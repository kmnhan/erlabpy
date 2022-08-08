import arpes
import arpes.fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from uncertainties import ufloat

from .utilities import correct_with_edge

__all__ = [
    "gold_edge",
    "gold_poly",
    "gold_poly_from_edge",
    "gold_resolution",
    "gold_resolution_roi",
]


def gold_edge(
    gold,
    phi_range,
    eV_range,
    bin_size=(1, 1),
    fast=False,
    method="leastsq",
    progress=True,
    parallel_kw=dict(),
):
    if fast:
        params = {}
        model = arpes.fits.GStepBModel
    else:
        params = {
            "temp": dict(value=gold.S.temp, vary=False),
        }
        model = arpes.fits.ExtendedAffineBroadenedFD

    weights = None

    if any([b != 1 for b in bin_size]):
        gold_binned = gold.coarsen(phi=bin_size[0], eV=bin_size[1], boundary="trim")
        gold = gold_binned.mean()
        gold_stderr = (gold_binned.std() / np.sqrt(np.prod(bin_size))).sel(
            phi=slice(*phi_range), eV=slice(*eV_range)
        )
        if (gold_stderr > 0).all():
            weights = 1 / gold_stderr

    fitresults = arpes.fits.broadcast_model(
        model,
        gold.sel(phi=slice(*phi_range), eV=slice(*eV_range)),
        "phi",
        weights=weights,
        params=params,
        method=method,
        progress=progress,
        parallel_kw=parallel_kw,
    )

    center_arr, center_stderr = fitresults.F.p("center"), fitresults.F.s("center")
    center_arr = center_arr.where(~center_stderr.isnull(), drop=True)
    center_stderr = center_stderr.where(~center_stderr.isnull(), drop=True)
    return center_arr, center_stderr


def gold_poly_from_edge(center, weights=None, degree=4, method="leastsq"):
    modelresult = arpes.fits.PolynomialModel(degree=degree).guess_fit(
        center, weights=weights, method=method
    )
    return modelresult


def gold_poly(
    gold,
    phi_range,
    eV_range,
    bin_size=(1, 1),
    degree=4,
    correct=False,
    fast=False,
    method="leastsq",
    parallel_kw=dict(),
    plot=True,
    ax=None,
):
    center_arr, center_stderr = gold_edge(
        gold,
        phi_range,
        eV_range,
        bin_size=bin_size,
        fast=fast,
        method=method,
        parallel_kw=parallel_kw,
    )

    modelresult = arpes.fits.PolynomialModel(degree=degree).guess_fit(
        center_arr, weights=1 / center_stderr, method=method
    )
    if plot:
        if ax is None:
            ax = plt.gca()
        gold.S.plot(ax=ax, cmap="copper", gamma=0.5)
        rect = Rectangle(
            (phi_range[0], eV_range[0]),
            np.diff(phi_range)[0],
            np.diff(eV_range)[0],
            ec="w",
            alpha=0.5,
            lw=0.75,
            fc="none",
        )
        ax.add_patch(rect)
        ax.errorbar(center_arr.phi, center_arr, center_stderr, fmt="o", lw=0.75, ms=2)
        ax.plot(
            gold.phi, modelresult.eval(modelresult.params, x=gold.phi), "r-", lw=0.5
        )
        ax.set_ylim(gold.eV[[0, -1]])

        modelresult.plot(
            data_kws=dict(lw=0.75, ms=4, mfc="w", zorder=0, c="0.4"),
            fit_kws=dict(c="r", lw=1.5),
        )
    if correct:
        return modelresult, correct_with_edge(gold, modelresult, plot=False)
    else:
        return modelresult


def gold_resolution(
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
):
    poly, gold = gold_poly(
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
        eV_range_fit = tuple(r - np.mean(poly.best_fit) for r in eV_range_edge)
    del poly
    gold_roi = gold.sel(phi=slice(*phi_range))
    edc_avg = gold_roi.mean("phi").sel(eV=slice(*eV_range_fit))

    params = dict(
        temp=dict(value=gold.S.temp, vary=False),
        resolution=dict(value=0.1, vary=True, min=0),
    )
    fit = arpes.fits.ExtendedAffineBroadenedFD().guess_fit(
        edc_avg, params=params, method=method
    )
    if plot:
        plt.show()
        ax = plt.gca()
        gold.S.plot(ax=ax, cmap="copper", gamma=0.5)
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
        ax.set_ylim(gold.eV[[0, -1]])

        fit.plot(
            data_kws=dict(lw=0.75, ms=4, mfc="w", zorder=0, c="0.4"),
            fit_kws=dict(c="r", lw=1.5),
        )
    print(
        f"center = {ufloat(fit.params['center'], fit.params['center'].stderr):S} eV\n"
        f"resolution = {ufloat(fit.params['resolution'], fit.params['resolution'].stderr):.4S} eV"
    )
    return fit


def gold_resolution_roi(
    gold_roi,
    eV_range,
    fix_temperature=True,
    method="leastsq",
    plot=True,
):
    edc_avg = gold_roi.mean("phi").sel(eV=slice(*eV_range))

    params = dict(
        temp=dict(value=gold_roi.S.temp, vary=not fix_temperature),
        resolution=dict(value=0.1, vary=True, min=0),
    )
    fit = arpes.fits.ExtendedAffineBroadenedFD().guess_fit(
        edc_avg,
        params=params,
        method=method,  # weights=1 / edc_stderr
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
    print(
        f"center = {ufloat(fit.params['center'], fit.params['center'].stderr):S} eV\n"
        f"resolution = {ufloat(fit.params['resolution'], fit.params['resolution'].stderr):.4S} eV"
    )
    return fit
