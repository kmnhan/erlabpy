import arpes as arp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from uncertainties import ufloat

from .utilities import correct_with_edge

__all__ = ["gold_edge", "gold_poly", "gold_resolution"]


def gold_edge(gold, phi_range, eV_range, method="least_square", parallel_kw=dict()):
    params = {
        "temp": dict(value=gold.S.temp, vary=False),
    }

    fitresults = arp.fits.broadcast_model(
        arp.fits.ExtendedAffineBroadenedFD,
        gold.sel(phi=slice(*phi_range), eV=slice(*eV_range)),
        "phi",
        params=params,
        method=method,
        parallel_kw=parallel_kw,
    )

    center_arr, center_stderr = fitresults.F.p("center"), fitresults.F.s("center")
    center_arr = center_arr.where(~center_stderr.isnull(), drop=True)
    center_stderr = center_stderr.where(~center_stderr.isnull(), drop=True)
    return center_arr, center_stderr


def gold_poly(
    gold,
    phi_range,
    eV_range,
    degree=4,
    correct=False,
    method="least_square",
    parallel_kw=dict(),
    plot=True,
    ax=None,
):
    center_arr, center_stderr = gold_edge(
        gold, phi_range, eV_range, method=method, parallel_kw=parallel_kw
    )

    modelresult = arp.fits.PolynomialModel(degree=degree).guess_fit(
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
        ax.errorbar(center_arr.phi, center_arr, center_stderr, fmt=".")
        ax.plot(center_arr.phi, modelresult.best_fit, "-")

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
    degree=4,
    method="least_square",
    plot=True,
    parallel_kw=dict(),
):
    poly, gold = gold_poly(
        gold,
        phi_range,
        eV_range_edge,
        degree=degree,
        correct=True,
        method=method,
        plot=plot,
        parallel_kw=parallel_kw,
    )
    if eV_range_fit is None:
        eV_range_fit = tuple(r - np.mean(poly.best_fit) for r in eV_range_edge)
    del poly

    edc_avg = gold.mean("phi").sel(eV=slice(*eV_range_fit))
    edc_stderr = gold.std("phi").sel(eV=slice(*eV_range_fit)) / np.sqrt(len(gold.phi))

    params = dict(temp=dict(value=gold.S.temp, vary=False))
    fit = arp.fits.ExtendedAffineBroadenedFD().guess_fit(
        edc_avg, params=params, weights=1 / edc_stderr, method=method
    )
    if plot:
        fit.plot(
            data_kws=dict(lw=0.75, ms=4, mfc="w", zorder=0, c="0.4"),
            fit_kws=dict(c="r", lw=1.5),
        )
    print(
        f"center = {ufloat(fit.params['center'], fit.params['center'].stderr):S}\n"
        f"resolution = {ufloat(fit.params['broadening'], fit.params['broadening'].stderr):S}"
    )
    return fit
