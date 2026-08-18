import matplotlib.pyplot as plt

import erlab.plotting as eplt
from erlab.io.exampledata import generate_data_angles, generate_hvdep_cuts


def convert_angle_resolved_data() -> None:
    data = generate_data_angles(shape=(200, 60, 300), assign_attributes=True, seed=1).T
    data.kspace.set_normal(alpha=0.0, beta=0.0, delta=30.0)
    data.kspace.work_function = 4.5
    angle_map = data.qsel(eV=-0.3)
    automatic_grid_map = data.kspace.convert().transpose("eV", "ky", "kx").qsel(eV=-0.3)
    specified_grid_map = (
        data.kspace.convert(
            bounds={"kx": (-0.5, 0.5), "ky": (-0.5, 0.5)},
            resolution={"kx": 0.01, "ky": 0.01},
        )
        .transpose("eV", "ky", "kx")
        .qsel(eV=-0.3)
    )

    _, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), layout="compressed")
    eplt.plot_array(angle_map, ax=axes[0], cmap="viridis", aspect="equal")
    eplt.plot_array(automatic_grid_map, ax=axes[1], cmap="viridis", aspect="equal")
    eplt.plot_array(specified_grid_map, ax=axes[2], cmap="viridis", aspect="equal")
    eplt.set_titles(
        axes,
        ["Angle coordinates", "Automatic momentum grid", "Specified momentum grid"],
    )


def convert_hv_dependent_scan() -> None:
    data = generate_hvdep_cuts(seed=1)
    data.kspace.inner_potential = 10.0
    angle_map = data.qsel(eV=-0.3).T
    converted_map = data.kspace.convert().qsel(eV=-0.3).T

    _, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="compressed")
    eplt.plot_array(angle_map, ax=axes[0], cmap="viridis")
    eplt.plot_array(converted_map, ax=axes[1], cmap="viridis")
    eplt.set_titles(
        axes, [r"$h\nu$ and angle coordinates", r"Converted $k_z$ coverage"]
    )


def overlay_cut_path() -> None:
    data = (
        generate_data_angles(
            shape=(200, 60, 300),
            assign_attributes=True,
            seed=1,
        )
        .assign_coords(xi=3.0)
        .T
    )
    data.kspace.offsets = {"xi": 3.0}

    cut_path = data.qsel(beta=-10).kspace.convert_coords().qsel(eV=-0.3)
    constant_energy_surface = (
        data.kspace.convert().transpose("eV", "ky", "kx").qsel(eV=-0.3)
    )

    _, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
    eplt.plot_array(
        constant_energy_surface,
        ax=ax,
        cmap="Greys",
        gamma=0.5,
        aspect="equal",
    )
    ax.plot(cut_path.kx, cut_path.ky, color="tab:red")


def annotate_photon_energies() -> None:
    data = generate_hvdep_cuts(seed=1)
    data.kspace.inner_potential = 10.0
    converted = data.kspace.convert()
    photon_energies = [30, 45, 60]
    binding_energy = -0.3
    kz_values = converted.kspace.hv_to_kz(photon_energies).qsel(eV=binding_energy)

    _, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
    eplt.plot_array(
        converted.qsel(eV=binding_energy).T,
        ax=ax,
        cmap="viridis",
        aspect="equal",
    )
    for index in range(kz_values.sizes["hv"]):
        kz = kz_values.isel(hv=index)
        ax.plot(kz.kx, kz, label=rf"$h\nu={float(kz.hv):g}$ eV")
    ax.legend()
