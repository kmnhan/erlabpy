import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import erlab
import erlab.plotting as eplt
from erlab.io.exampledata import generate_data


def add_intensity_colorbar() -> None:
    data = generate_data(bandshift=-0.2, seed=1).T
    cut = data.qsel(ky=0.3)
    _, ax = plt.subplots(figsize=(3.4, 2.1), layout="compressed")
    image = eplt.plot_array(cut, ax=ax, cmap="Greys", gamma=0.5)
    eplt.nice_colorbar(ax=ax, mappable=image, width=10, minmax=True)


def combine_maps_and_cuts() -> None:
    data = generate_data(bandshift=-0.2, seed=1).T
    energies = [-0.4, -0.2, 0.0]
    ky_values = [0.0, 0.1, 0.3]

    _, axes = plt.subplots(
        2,
        3,
        figsize=(6.4, 4.0),
        layout="compressed",
        sharex="col",
        sharey="row",
    )
    eplt.plot_slices(
        [data],
        eV=energies,
        axes=axes[0],
        axis="image",
        gamma=0.5,
        same_limits=True,
        annotate=False,
    )
    eplt.plot_slices(
        [data],
        ky=ky_values,
        axes=axes[1],
        gamma=0.5,
        same_limits=True,
        annotate=False,
    )
    eplt.label_subplot_properties(axes[0], values={"eV": energies})
    eplt.label_subplot_properties(axes[1], values={"ky": ky_values})
    eplt.clean_labels(axes)


def annotate_arpes_figure() -> None:
    data = generate_data(bandshift=-0.2, seed=1).T
    cut = data.qsel(ky=0.3)
    _, ax = plt.subplots(figsize=(3.4, 2.1), layout="compressed")
    eplt.plot_array(cut, ax=ax, cmap="Greys", gamma=0.5)
    eplt.fermiline(ax=ax, linestyle="--")
    eplt.mark_points([-0.6, 0.0, 0.6], ["K", "G", "K"], y=0.02, ax=ax)
    eplt.label_subplots(ax, prefix="(", suffix=")")


def set_panel_titles_and_labels() -> None:
    data = generate_data(bandshift=-0.2, seed=1).T
    constant_energy_map = data.qsel(eV=-0.2)
    edc = data.qsel.around(0.06, kx=0.52, ky=0.3)

    _, axes = plt.subplots(
        1,
        2,
        figsize=(6.4, 3.0),
        layout="compressed",
    )
    eplt.plot_array(
        constant_energy_map,
        ax=axes[0],
        cmap="Greys",
        gamma=0.5,
        aspect="equal",
    )
    axes[1].plot(edc.eV, edc, color="0.2")
    eplt.fermiline(ax=axes[1], orientation="v", linestyle="--")
    eplt.set_titles(axes, ["Constant energy map", "Energy distribution curve"])
    eplt.set_xlabels(axes, [r"$k_x$ (Å$^{-1}$)", r"$E-E_F$ (eV)"])
    eplt.set_ylabels(axes, [r"$k_y$ (Å$^{-1}$)", "Intensity (arb. units)"])


def display_energy_in_mev() -> None:
    data = generate_data(bandshift=-0.2, seed=1).T
    cut = data.qsel(ky=0.3)
    _, ax = plt.subplots(figsize=(3.4, 2.1), layout="compressed")
    cut.qplot(ax=ax)
    eplt.scale_units(ax, "y", si=-3)


def mark_core_levels() -> None:
    energy = np.linspace(-65.0, -15.0, 1600)

    # The positions reproduce the pristine Bi2Se3 spectrum in Polyakov et al.
    # Each d5/2:d3/2 pair has a 3:2 integrated-area ratio. The total Bi:Se
    # intensity is illustrative because photoionization cross sections and the
    # instrument response determine it.
    peak_components = []
    for center_5_2, splitting, total_area, sigma, gamma in (
        (-25.0, 3.0, 1.00, 0.23, 0.12),  # Bi 5d
        (-53.5, 0.8, 0.75, 0.18, 0.08),  # Se 3d
    ):
        peak_components.extend(
            (
                erlab.analysis.fit.functions.voigt(
                    energy,
                    center=center_5_2,
                    sigma=sigma,
                    gamma=gamma,
                    amplitude=3.0 * total_area / 5.0,
                ),
                erlab.analysis.fit.functions.voigt(
                    energy,
                    center=center_5_2 - splitting,
                    sigma=sigma,
                    gamma=gamma,
                    amplitude=2.0 * total_area / 5.0,
                ),
            )
        )

    # Add an inelastic background that rises on the high-binding-energy side.
    background_components = erlab.analysis.fit.functions.active_shirley(
        energy,
        peak_components,
        k_steps=[0.08] * len(peak_components),
        const_bkg=0.04,
    )
    expected_intensity = sum(peak_components, start=np.zeros_like(energy)) + sum(
        background_components.values(), start=np.zeros_like(energy)
    )
    expected_counts = 3000.0 * expected_intensity / expected_intensity.max()
    intensity = np.random.default_rng(1).poisson(expected_counts) / 3000.0

    core_spectrum = xr.DataArray(
        intensity,
        coords={"eV": energy},
        dims="eV",
        name="photoelectron_intensity",
    )

    _, ax = plt.subplots(figsize=(6.4, 3.0), layout="compressed")
    core_spectrum.plot.line(ax=ax, color="0.15", linewidth=0.8)
    eplt.plot_core_levels(
        ["Bi", "Se"],
        ax=ax,
        energy="binding",
        binding_energy_sign="negative",
        linestyle="--",
    )
    ax.set(
        xlabel=r"$E - E_\mathrm{F}$ (eV)",
        ylabel="Photoelectron intensity (arb. units)",
        ylim=(0.0, None),
    )


def overlay_brillouin_zone() -> None:
    lattice_constant = 6.97
    data = generate_data(a=lattice_constant, seed=1).T
    constant_energy_surface = data.qsel(eV=-0.2)

    _, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
    eplt.plot_array(
        constant_energy_surface,
        ax=ax,
        cmap="Greys",
        gamma=0.5,
        aspect="equal",
    )
    eplt.plot_hex_bz(
        a=lattice_constant,
        ax=ax,
        fill=False,
        edgecolor="tab:purple",
        linestyle="--",
        linewidth=1.2,
    )


def draw_out_of_plane_brillouin_zone() -> None:
    avec = erlab.lattice.abc2avec(6.0, 10.0, 25.0, 90.0, 90.0, 90.0)
    avec_primitive = erlab.lattice.to_primitive(avec, centering_type="F")
    bvec = erlab.lattice.to_reciprocal(avec_primitive)

    _, ax = plt.subplots(figsize=(3.0, 3.0), layout="compressed")
    eplt.plot_out_of_plane_bz(
        bvec,
        k_parallel=0.0,
        angle=90.0,
        bounds=(-1.5, 1.5, -1.5, 1.5),
        ax=ax,
        vertices=True,
        color="tab:purple",
        linewidth=1.5,
    )
    ax.set(
        xlabel=r"$k_x$ (Å$^{-1}$)",
        ylabel=r"$k_z$ (Å$^{-1}$)",
        aspect="equal",
    )


def draw_two_dimensional_brillouin_zone() -> None:
    avec = erlab.lattice.abc2avec(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)

    _, ax = plt.subplots(figsize=(2.5, 2.5), layout="compressed")
    eplt.plot_bz(avec, ax=ax)
    ax.set(
        xlabel=r"$k_x$ (Å$^{-1}$)",
        ylabel=r"$k_y$ (Å$^{-1}$)",
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        aspect="equal",
    )


def _intensity_and_asymmetry() -> tuple[xr.DataArray, xr.DataArray]:
    data_a, data_b = generate_data(
        shape=(250, 250, 2),
        Erange=(-0.3, 0.3),
        temp=0.0,
        seed=1,
        count=1_000_000,
    ).T
    intensity = data_a + data_b
    asymmetry = ((data_a - data_b) / intensity).where(intensity > 0)
    return intensity, asymmetry


def compare_intensity_and_asymmetry() -> None:
    intensity, asymmetry = _intensity_and_asymmetry()

    _, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        layout="compressed",
        sharex=True,
        sharey=True,
    )
    intensity_image = eplt.plot_array(
        intensity,
        ax=axes[0],
        cmap="viridis",
        aspect="equal",
    )
    asymmetry_image = eplt.plot_array(
        asymmetry,
        ax=axes[1],
        cmap="bwr",
        norm=eplt.CenteredPowerNorm(1.0, vcenter=0.0, halfrange=1.0),
        aspect="equal",
    )
    eplt.nice_colorbar(ax=axes[0], mappable=intensity_image, width=7)
    eplt.nice_colorbar(ax=axes[1], mappable=asymmetry_image, width=7)
    eplt.set_titles(axes, ["Total intensity", "Normalized difference"])
    eplt.clean_labels(axes)


def plot_intensity_and_asymmetry() -> None:
    intensity, asymmetry = _intensity_and_asymmetry()

    _, ax = plt.subplots(figsize=(4.8, 3.4), layout="compressed")
    _, colorbar = eplt.plot_array_2d(
        intensity,
        asymmetry,
        ax=ax,
        lnorm=eplt.InversePowerNorm(0.5),
        cnorm=eplt.CenteredInversePowerNorm(0.7, vcenter=0.0, halfrange=1.0),
    )
    colorbar.ax.set_xticks(colorbar.ax.get_xlim(), labels=["Min", "Max"])
    colorbar.ax.set(xlabel="Intensity", ylabel="Asymmetry")
