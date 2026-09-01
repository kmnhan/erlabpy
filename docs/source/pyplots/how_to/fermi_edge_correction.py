import matplotlib.pyplot as plt

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
        parallel_kw={"n_jobs": 1},
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
