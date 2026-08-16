import matplotlib.pyplot as plt
import numpy as np

import erlab.analysis as era
import erlab.plotting as eplt
from erlab.io.exampledata import generate_data


def plot_high_symmetry_cut() -> None:
    lattice_constant = 6.97
    vertices = {
        "kx": [
            0.0,
            2 * np.pi / (np.sqrt(3) * lattice_constant),
            2 * np.pi / (np.sqrt(3) * lattice_constant),
            0.0,
        ],
        "ky": [0.0, 0.0, 2 * np.pi / (3 * lattice_constant), 0.0],
    }
    data = generate_data(a=lattice_constant, seed=1).T
    path_data = era.interpolate.slice_along_path(
        data,
        vertices=vertices,
        step_size=0.005,
    )
    path_vertices = np.column_stack([vertices["kx"], vertices["ky"]])
    segment_lengths = np.linalg.norm(np.diff(path_vertices, axis=0), axis=1)
    vertex_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    energy_map = data.qsel(eV=-0.2, eV_width=0.02)

    _fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.4, 3.2),
        layout="compressed",
        gridspec_kw={"width_ratios": (1.0, 1.35)},
    )
    eplt.plot_array(
        energy_map,
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
        vertices["kx"],
        vertices["ky"],
        color="tab:red",
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    axes[0].set_title(r"$E = E_F - 0.2$ eV")

    eplt.plot_array(path_data, ax=axes[1], cmap="Greys", gamma=0.7)
    eplt.fermiline(ax=axes[1], linestyle="--", linewidth=0.8)
    for position in vertex_positions[1:-1]:
        axes[1].axvline(
            position,
            color="0.5",
            linestyle="--",
            linewidth=0.8,
        )
    axes[1].set_xticks(vertex_positions, labels=["Γ", "M", "K", "Γ"])
    axes[1].set(
        xlabel="",
        xlim=(vertex_positions[0] - 0.03, vertex_positions[-1] + 0.03),
    )
    plt.show()
