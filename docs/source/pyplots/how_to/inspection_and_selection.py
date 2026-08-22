import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import erlab
import erlab.analysis as era
import erlab.plotting as eplt
from erlab.io.exampledata import generate_data


def mask_momentum_region() -> None:
    data = generate_data(seed=1).T
    constant_energy_map = data.qsel(eV=-0.2, eV_width=0.02)
    lattice_constant = 6.97
    real_space_basis = lattice_constant * np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3) / 2],
        ]
    )
    first_bz_vertices = erlab.lattice.get_2d_vertices(
        real_space_basis,
        reciprocal=False,
        rotate=30.0,
    )
    masked_map = era.mask.mask_with_polygon(
        constant_energy_map,
        first_bz_vertices,
        dims=("kx", "ky"),
    )
    closed_vertices = np.vstack([first_bz_vertices, first_bz_vertices[0]])

    _, axes = plt.subplots(
        1,
        2,
        figsize=(6.4, 3.0),
        layout="compressed",
        sharex=True,
        sharey=True,
    )
    for ax, map_data in zip(
        axes,
        (constant_energy_map, masked_map),
        strict=True,
    ):
        eplt.plot_array(
            map_data,
            ax=ax,
            cmap="Greys",
            gamma=0.5,
            aspect="equal",
        )

    axes[0].plot(
        closed_vertices[:, 0],
        closed_vertices[:, 1],
        color="tab:red",
        linewidth=1.2,
    )
    eplt.unify_clim(axes)
    eplt.clean_labels(axes)
    eplt.set_titles(axes, ["First Brillouin zone", "Masked data"])


def compare_radial_neighborhoods() -> None:
    data = generate_data(seed=1).T
    kx_center = 0.52
    ky_center = 0.30
    radii = (0.03, 0.06, 0.09, 0.12)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(radii)]

    _, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="compressed")
    eplt.plot_array(
        data.qsel(eV=-0.2),
        ax=axes[0],
        cmap="Greys",
        gamma=0.5,
        aspect="equal",
    )

    for radius, color in zip(radii, colors, strict=True):
        axes[0].add_patch(
            Circle(
                (kx_center, ky_center),
                radius,
                fill=False,
                color=color,
                linewidth=1.2,
            )
        )
        edc = data.qsel.around(radius, kx=kx_center, ky=ky_center)
        axes[1].plot(edc.eV, edc, color=color, label=f"{radius:.2f} Å⁻¹")

    axes[0].plot(kx_center, ky_center, ".", color="black")
    eplt.fermiline(ax=axes[1], orientation="v", linestyle="--")
    axes[1].set(xlabel=r"$E-E_F$ (eV)", ylabel="Intensity")
    axes[1].legend(title="Radius")
    eplt.set_titles(axes, ["Momentum neighborhoods", "Averaged EDCs"])


def compare_multidimensional_slices() -> None:
    data = generate_data(seed=1).T
    energies = [-0.4, -0.2, 0.0]
    _, axes = plt.subplots(
        2,
        3,
        figsize=(6.4, 4.0),
        layout="compressed",
        sharex=True,
        sharey=True,
    )

    for row in axes:
        eplt.plot_slices(
            [data],
            eV=energies,
            eV_width=0.05,
            axes=row,
            axis="image",
            gamma=0.5,
            annotate=False,
            cmap="Greys",
        )
        eplt.label_subplot_properties(row, values={"Eb": energies})

    eplt.unify_clim(axes[1])
    eplt.clean_labels(axes)
    axes[0, 0].set_title("Independent intensity limits", loc="left")
    axes[1, 0].set_title("Shared intensity limits", loc="left")


def plot_high_symmetry_cut() -> None:
    lattice_constant = 6.97
    high_symmetry_vertices = {
        "kx": [
            0.0,
            2 * np.pi / (np.sqrt(3) * lattice_constant),
            2 * np.pi / (np.sqrt(3) * lattice_constant),
            0.0,
        ],
        "ky": [0.0, 0.0, 2 * np.pi / (3 * lattice_constant), 0.0],
    }
    momentum_data = generate_data(a=lattice_constant, seed=1).T
    high_symmetry_cut = era.interpolate.slice_along_path(
        momentum_data,
        vertices=high_symmetry_vertices,
        step_size=0.005,
    )
    path_vertices = np.column_stack(
        [high_symmetry_vertices["kx"], high_symmetry_vertices["ky"]]
    )
    segment_lengths = np.linalg.norm(np.diff(path_vertices, axis=0), axis=1)
    path_vertex_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    path_energy_map = momentum_data.qsel(eV=-0.2, eV_width=0.02)

    _, axes = plt.subplots(
        1,
        2,
        figsize=(6.4, 3.2),
        layout="compressed",
        gridspec_kw={"width_ratios": (1.0, 1.35)},
    )
    eplt.plot_array(
        path_energy_map,
        ax=axes[0],
        cmap="Greys",
        gamma=0.7,
        aspect="equal",
    )
    eplt.plot_hex_bz(
        a=lattice_constant,
        ax=axes[0],
        fill=False,
        edgecolor="0.35",
        linewidth=0.8,
    )
    axes[0].plot(
        high_symmetry_vertices["kx"],
        high_symmetry_vertices["ky"],
        color="tab:red",
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    axes[0].set_title(r"$E = E_F - 0.2$ eV")

    eplt.plot_array(high_symmetry_cut, ax=axes[1], cmap="Greys", gamma=0.7)
    eplt.fermiline(ax=axes[1], linestyle="--", linewidth=0.8)
    for position in path_vertex_positions[1:-1]:
        axes[1].axvline(
            position,
            color="0.5",
            linestyle="--",
            linewidth=0.8,
        )
    axes[1].set_xticks(path_vertex_positions, labels=["Γ", "M", "K", "Γ"])
    axes[1].set(
        xlabel="",
        xlim=(path_vertex_positions[0] - 0.03, path_vertex_positions[-1] + 0.03),
    )
