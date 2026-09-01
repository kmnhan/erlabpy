import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import erlab.analysis as era
import erlab.plotting as eplt
from erlab.io.exampledata import generate_gold_edge


def correct_curved_fermi_edge() -> None:
    gold_reference = generate_gold_edge(temp=100.0, seed=1)
    edge_fit = era.gold.poly(
        gold_reference,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        temp=100.0,
        vary_temp=False,
        bkg_slope=False,
        degree=2,
        plot=True,
    )
    corrected = era.gold.correct_with_edge(gold_reference, edge_fit)

    _, ax = plt.subplots(figsize=(3.4, 2.1), layout="compressed")
    eplt.plot_array(corrected, ax=ax, cmap="Greys", gamma=0.5)
    eplt.fermiline(ax=ax, linestyle="--")


def fit_fermi_edge_with_separate_ranges() -> None:
    gold_reference = generate_gold_edge(
        temp=100.0,
        Eres=0.02,
        edge_coeffs=(0.0, 0.008, 0.0),
        seed=1,
    )
    _, axes = plt.subplots(1, 3, figsize=(6.4, 2.3), layout="compressed", sharey=True)
    for angle, ax in zip((-12.0, 0.0, 12.0), axes, strict=True):
        edc = gold_reference.sel(alpha=angle, method="nearest").sel(
            eV=slice(-0.35, 0.20)
        )
        lower, upper = era.gold.guess_edge_fit_range(
            edc,
            temp=100.0,
            resolution=0.02,
        )
        edc.plot.line(ax=ax, color="0.2")
        ax.axvspan(lower, upper, color="tab:blue", alpha=0.2)
        ax.set_title(rf"$\alpha = {float(edc.alpha):.1f}^\circ$")
    eplt.clean_labels(axes)


def fit_spectra_across_coordinate() -> None:
    spectra = generate_gold_edge(temp=100.0, Eres=0.02, seed=1)
    fit_data = spectra.sel(eV=slice(-0.2, 0.2))
    model = era.fit.models.FermiEdgeModel()
    params = {
        "temp": {"value": 100.0, "vary": False},
        "back1": {"value": 0.0, "vary": False},
    }
    fit_result = fit_data.xlm.modelfit("eV", model=model, params=params, guess=True)
    centers = fit_result.modelfit_coefficients.sel(param="center")
    center_errors = fit_result.modelfit_stderr.sel(param="center")

    _, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="compressed")
    eplt.plot_array(spectra, ax=axes[0], cmap="Greys", gamma=0.5)
    axes[0].errorbar(spectra.alpha, centers, yerr=center_errors, fmt=".")
    axes[1].errorbar(spectra.alpha, centers, yerr=center_errors, fmt=".")
    axes[1].set(
        xlabel=r"$\alpha$ (deg)",
        ylabel=r"Fitted $E_F$ (eV)",
    )
    eplt.set_titles(axes, ["Fits across the reference", "Centers and uncertainties"])


def inspect_fit_components_and_residuals() -> None:
    energy = np.linspace(-31.5, -22.0, 800)
    peaks = [
        era.fit.functions.voigt(
            energy,
            center=-25.0,
            sigma=0.22,
            gamma=0.12,
            amplitude=0.60,
        ),
        era.fit.functions.voigt(
            energy,
            center=-28.0,
            sigma=0.22,
            gamma=0.14,
            amplitude=0.40,
        ),
    ]
    background = era.fit.functions.active_shirley(
        energy,
        peaks,
        k_steps=[0.09, 0.09],
        const_bkg=0.04,
    )
    expected = sum(peaks, start=np.zeros_like(energy)) + sum(
        background.values(),
        start=np.zeros_like(energy),
    )
    count_scale = 5000.0 / expected.max()
    counts = np.random.default_rng(1).poisson(count_scale * expected)
    intensity = counts / count_scale
    core_spectrum = xr.DataArray(
        intensity,
        coords={"eV": energy},
        dims="eV",
        name="photoelectron_intensity",
    )

    model = era.fit.models.MultiPeakModel(
        npeaks=2,
        peak_shapes="voigt",
        fd=False,
        background="shirley",
        convolve=False,
    )
    params = model.guess(intensity, x=energy)
    params["p0_center"].set(value=-25.0, min=-25.5, max=-24.5)
    params["p0_sigma"].set(value=0.20, min=0.0, max=0.5)
    params["p0_gamma"].set(value=0.10, min=0.0, max=0.5)
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
    _, axes = plt.subplots(
        2,
        1,
        figsize=(4.8, 3.6),
        layout="compressed",
        sharex=True,
        height_ratios=(3, 1),
    )
    axes[0].plot(
        fit_data.eV,
        fit_data,
        "o",
        markersize=2,
        markerfacecolor="none",
        markeredgecolor="0.25",
        markeredgewidth=0.5,
        label="Measured data",
    )
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
    axes[1].plot(
        residual.eV,
        residual,
        ".",
        color="0.25",
        markersize=2,
    )
    axes[1].set(xlabel=r"$E-E_F$ (eV)", ylabel="Residual")
    eplt.clean_labels(axes)
