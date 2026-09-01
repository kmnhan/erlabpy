import matplotlib.pyplot as plt

import erlab.analysis as era
import erlab.plotting as eplt
from erlab.io.exampledata import (
    generate_data,
    generate_data_angles,
    generate_hvdep_cuts,
)


def rotate_map() -> None:
    volume = generate_data(
        shape=(160, 160, 180),
        bandshift=-0.2,
        seed=1,
    ).transpose("ky", "kx", "eV")
    rotated = era.transform.rotate(
        volume,
        angle=25.0,
        axes=("ky", "kx"),
        center={"ky": 0.0, "kx": 0.0},
        reshape=True,
    )
    energies = [0.0, -0.2, -0.4]
    _, axes = eplt.plot_slices(
        [volume, rotated],
        eV=energies,
        figsize=(7.2, 4.8),
        order="C",
        cmap="Greys",
        gamma=0.5,
        axis="image",
        same_limits=True,
        annotate=False,
        subplot_kw={"layout": "compressed", "sharex": True, "sharey": True},
    )
    eplt.set_titles(axes[0], [rf"Input, $E={energy:.1f}$ eV" for energy in energies])
    eplt.set_titles(axes[1], [rf"Rotated, $E={energy:.1f}$ eV" for energy in energies])
    eplt.clean_labels(axes)


def align_spectra_with_offsets() -> None:
    spectra = generate_hvdep_cuts(
        shape=(9, 250, 300),
        Erange=(-0.25, 0.08),
        hvrange=(60.0, 100.0),
        bandshift=-0.2,
        hv_shift=(-0.012, 0.008),
        noise=False,
    )
    energy_offsets = (
        spectra.qsel(alpha=0.0)
        .xlm.modelfit("eV", model=era.fit.models.StepEdgeModel(), guess=True)
        .modelfit_coefficients.sel(param="center")
    )
    aligned = era.transform.shift(spectra, shift=-energy_offsets, along="eV")
    photon_energies = spectra.hv.values[[0, -1]]

    _, axes = plt.subplots(
        2,
        3,
        figsize=(9.6, 5.4),
        layout="compressed",
        sharex="col",
        sharey=True,
    )
    for row, data in enumerate((spectra, aligned)):
        for ax, view in zip(
            axes[row],
            (data.isel(hv=0).T, data.isel(hv=-1).T, data.qsel(alpha=0.0)),
            strict=True,
        ):
            eplt.plot_array(view, ax=ax, cmap="Greys", gamma=0.5)

    axes[0, 2].plot(
        energy_offsets.hv,
        energy_offsets,
        "o-",
        color="tab:red",
        linewidth=0.9,
        markersize=2.5,
    )
    eplt.fermiline(ax=axes, linestyle="--", linewidth=0.75)
    for column in axes.T:
        eplt.unify_clim(column)
    for row, stage in enumerate(("Before", "After")):
        eplt.set_titles(
            axes[row],
            [
                rf"{stage}, $h\nu={photon_energies[0]:.1f}$ eV",
                rf"{stage}, $h\nu={photon_energies[1]:.1f}$ eV",
                rf"{stage}, $\alpha=0.0^\circ$",
            ],
        )
    eplt.clean_labels(axes)


def preserve_shifted_coordinate_range() -> None:
    spectra = generate_hvdep_cuts(
        shape=(9, 250, 300),
        Erange=(-0.25, 0.08),
        hvrange=(60.0, 100.0),
        bandshift=-0.2,
        hv_shift=(-0.012, 0.008),
        noise=False,
    )
    energy_offsets = (
        spectra.qsel(alpha=0.0)
        .xlm.modelfit("eV", model=era.fit.models.StepEdgeModel(), guess=True)
        .modelfit_coefficients.sel(param="center")
    )
    aligned = era.transform.shift(spectra, shift=-energy_offsets, along="eV")
    aligned_with_full_range = era.transform.shift(
        spectra,
        shift=-energy_offsets,
        along="eV",
        shift_coords=True,
    )

    _, axes = plt.subplots(
        1,
        2,
        figsize=(6.4, 3.0),
        layout="compressed",
        sharex=True,
    )
    for ax, data in zip(
        axes,
        (aligned, aligned_with_full_range),
        strict=True,
    ):
        eplt.plot_array(data.qsel(alpha=0.0), ax=ax, cmap="Greys", gamma=0.5)
    eplt.unify_clim(axes)
    eplt.fermiline(ax=axes, linestyle="--", linewidth=0.75)
    eplt.set_titles(axes, ["Fixed coordinate range", "Expanded coordinate range"])
    eplt.clean_labels(axes)


def apply_symmetry() -> None:
    cut = generate_data(seed=3, bandshift=-0.2).qsel(ky=0.3).T
    kx_weight = 0.1 + (cut.kx - cut.kx.min()) / (cut.kx.max() - cut.kx.min())
    cut = cut * kx_weight
    symmetrized = era.transform.symmetrize(
        cut,
        dim="kx",
        center=0.0,
    )
    antisymmetrized = era.transform.symmetrize(
        cut,
        dim="kx",
        center=0.0,
        subtract=True,
    )

    _, axes = eplt.plot_slices(
        [cut, symmetrized, antisymmetrized],
        figsize=(9.0, 3.0),
        order="F",
        axis="auto",
        cmap=["Greys", "Greys", "bwr"],
        norm=[None, None, eplt.CenteredInversePowerNorm(1.0)],
        gamma=0.5,
    )
    eplt.set_titles(axes, ["Original", "Symmetrized", "Antisymmetrized"])
    eplt.clean_labels(axes)


def apply_rotational_symmetry() -> None:
    volume = generate_data(
        shape=(160, 160, 180),
        bandshift=-0.2,
        seed=1,
    ).transpose("ky", "kx", "eV")
    partial_volume = volume.where(
        (volume.kx < -0.25) * (volume.ky < 0.15),
        drop=True,
    )
    symmetrized = era.transform.symmetrize_nfold(
        partial_volume,
        6,
        axes=("kx", "ky"),
        center={"kx": 0.0, "ky": 0.0},
        reshape=True,
    )

    energies = [0.0, -0.2, -0.4]
    _, axes = eplt.plot_slices(
        [partial_volume, symmetrized],
        eV=energies,
        figsize=(7.2, 4.8),
        order="C",
        axis="image",
        cmap="Greys",
        gamma=0.5,
        same_limits=True,
        annotate=False,
        subplot_kw={"layout": "compressed", "sharex": True, "sharey": True},
    )
    eplt.set_titles(axes[0], [rf"Input, $E={energy:.1f}$ eV" for energy in energies])
    eplt.set_titles(axes[1], [rf"Six-fold, $E={energy:.1f}$ eV" for energy in energies])
    eplt.clean_labels(axes)


def gaussian_convolution() -> None:
    simulated_data = generate_data_angles(
        shape=(500, 1, 500),
        angrange={"alpha": (-15, 15), "beta": (-5, 5)},
        seed=1,
        bandshift=-0.2,
    ).T
    broadened = era.image.gaussian_filter(
        simulated_data,
        sigma={"eV": 0.01, "alpha": 0.2},
    )
    _, axes = plt.subplots(
        1,
        2,
        figsize=(6.4, 3.0),
        layout="compressed",
        sharex=True,
        sharey=True,
    )
    eplt.plot_array(simulated_data, ax=axes[0], cmap="Greys", gamma=0.5)
    eplt.plot_array(broadened, ax=axes[1], cmap="Greys", gamma=0.5)
    eplt.unify_clim(axes)
    eplt.set_titles(axes, ["Simulation", "Gaussian convolution"])
    eplt.clean_labels(axes)
