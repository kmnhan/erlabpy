import matplotlib.pyplot as plt

import erlab.plotting as eplt
from erlab.io.exampledata import generate_data_angles, generate_hvdep_cuts


def convert_angle_resolved_data() -> None:
    data = generate_data_angles(shape=(200, 60, 300), assign_attributes=True, seed=1).T
    data.kspace.set_normal(alpha=0.0, beta=0.0, delta=30.0)
    data.kspace.work_function = 4.5
    converted = data.kspace.convert().transpose("eV", "ky", "kx")
    converted_on_target_grid = data.kspace.convert(
        bounds={"kx": (-0.5, 0.5), "ky": (-0.5, 0.5)},
        resolution={"kx": 0.01, "ky": 0.01},
    ).transpose("eV", "ky", "kx")
    angle_map = data.qsel(eV=-0.3)
    converted_map = converted.qsel(eV=-0.3)
    target_grid_map = converted_on_target_grid.qsel(eV=-0.3)

    _fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), layout="compressed")
    eplt.plot_array(angle_map, ax=axes[0], cmap="viridis", aspect="equal")
    eplt.plot_array(converted_map, ax=axes[1], cmap="viridis", aspect="equal")
    eplt.plot_array(target_grid_map, ax=axes[2], cmap="viridis", aspect="equal")
    eplt.set_titles(
        axes,
        ["Angle coordinates", "Automatic momentum grid", "Specified momentum grid"],
    )
    plt.show()


def convert_hv_dependent_scan() -> None:
    data = generate_hvdep_cuts(seed=1)
    data.kspace.inner_potential = 10.0
    converted = data.kspace.convert()
    angle_map = data.qsel(eV=-0.3).T
    converted_map = converted.qsel(eV=-0.3).T

    _fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="compressed")
    eplt.plot_array(angle_map, ax=axes[0], cmap="viridis")
    eplt.plot_array(converted_map, ax=axes[1], cmap="viridis")
    eplt.set_titles(
        axes, [r"$h\nu$ and angle coordinates", r"Converted $k_z$ coverage"]
    )
    plt.show()


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

    cut = data.qsel(beta=-10).kspace.convert_coords()
    converted_map = data.kspace.convert().transpose("eV", "ky", "kx").qsel(eV=-0.3)
    cut_path = cut.qsel(eV=-0.3)

    _fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
    eplt.plot_array(converted_map, ax=ax, cmap="Greys", gamma=0.5, aspect="equal")
    ax.plot(cut_path.kx, cut_path.ky, color="tab:red")
    plt.show()


def annotate_photon_energies() -> None:
    data = generate_hvdep_cuts(seed=1)
    data.kspace.inner_potential = 10.0
    converted = data.kspace.convert()
    photon_energies = [30, 45, 60]
    binding_energy = -0.3
    kz_values = converted.kspace.hv_to_kz(photon_energies).qsel(eV=binding_energy)

    _fig, ax = plt.subplots(figsize=(3.4, 3.0), layout="compressed")
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
    plt.show()
