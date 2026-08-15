import lmfit
import matplotlib.pyplot as plt

import erlab.analysis as era
import erlab.plotting as eplt
from erlab.io.exampledata import generate_data, generate_gold_edge


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

    _fig, ax = plt.subplots(figsize=(3.4, 2.1), layout="compressed")
    eplt.plot_array(corrected, ax=ax, cmap="Greys", gamma=0.5)
    eplt.fermiline(ax=ax, linestyle="--")
    plt.show()


def fit_fermi_edge_with_separate_ranges() -> None:
    gold_reference = generate_gold_edge(
        temp=100.0,
        Eres=0.02,
        edge_coeffs=(0.0, 0.008, 0.0),
        seed=1,
    )
    selected_angles = (-12.0, 0.0, 12.0)

    _fig, axes = plt.subplots(
        1, 3, figsize=(6.4, 2.3), layout="compressed", sharey=True
    )
    for angle, ax in zip(selected_angles, axes, strict=True):
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
    plt.show()


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

    _fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="compressed")
    eplt.plot_array(spectra, ax=axes[0], cmap="Greys", gamma=0.5)
    axes[0].errorbar(spectra.alpha, centers, yerr=center_errors, fmt=".")
    axes[1].errorbar(spectra.alpha, centers, yerr=center_errors, fmt=".")
    axes[1].set(
        xlabel=r"$\alpha$ (deg)",
        ylabel=r"Fitted $E_F$ (eV)",
    )
    eplt.set_titles(axes, ["Fits across the reference", "Centers and uncertainties"])
    plt.show()


def inspect_fit_components_and_residuals() -> None:
    spectrum = generate_data(seed=1).T.qsel(ky=0.3, eV=0.0)
    spectrum = spectrum.sel(kx=slice(-0.75, 0.75))
    coordinate = spectrum.kx.values
    measured = spectrum.values / float(spectrum.max())

    model = lmfit.models.LorentzianModel(prefix="left_peak_")
    model += lmfit.models.LorentzianModel(prefix="right_peak_")
    model += lmfit.models.LinearModel(prefix="background_")
    parameters = model.make_params(
        left_peak_amplitude=0.04,
        left_peak_center=-0.52,
        left_peak_sigma=0.03,
        right_peak_amplitude=0.04,
        right_peak_center=0.52,
        right_peak_sigma=0.03,
        background_slope=0.0,
        background_intercept=0.03,
    )
    lmfit_result = model.fit(
        measured,
        x=coordinate,
        params=parameters,
    )
    components = lmfit_result.eval_components(x=coordinate)
    residual = measured - lmfit_result.best_fit

    _figure, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 3.0),
        layout="compressed",
        sharex=True,
        height_ratios=(3, 1),
    )
    axes[0].plot(
        coordinate,
        measured,
        ".",
        markersize=2,
        color="0.25",
        label="Measured data",
    )
    axes[0].plot(coordinate, lmfit_result.best_fit, label="Best fit")
    for name, component in components.items():
        label = name.rstrip("_").replace("_", " ").title()
        axes[0].plot(
            coordinate,
            component,
            "--",
            label=label,
        )
    axes[0].set_ylabel("Normalized intensity")
    axes[0].legend(ncols=2)

    axes[1].axhline(0.0, color="0.5", linewidth=1)
    axes[1].plot(coordinate, residual, ".-", color="tab:red", markersize=3)
    axes[1].set(xlabel=r"$k_x$ (Å$^{-1}$)", ylabel="Residual")
    eplt.clean_labels(axes)
    plt.show()
