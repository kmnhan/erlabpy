import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import erlab.plotting as eplt
from erlab.io.exampledata import generate_data_angles


def compare_angular_offsets() -> None:
    data = (
        generate_data_angles(
            shape=(200, 60, 300),
            assign_attributes=True,
            seed=1,
        )
        .assign_coords(xi=3.0)
        .T
    )

    data.kspace.offsets = {}
    uncorrected = data.kspace.convert().transpose("eV", "ky", "kx")

    data.kspace.offsets = {"xi": 3.0}
    corrected = data.kspace.convert().transpose("eV", "ky", "kx")

    maps = [
        data.qsel(eV=-0.3),
        uncorrected.qsel(eV=-0.3),
        corrected.qsel(eV=-0.3),
    ]
    norm = mcolors.Normalize(
        vmin=min(float(data_map.quantile(0.01)) for data_map in maps),
        vmax=max(float(data_map.quantile(0.99)) for data_map in maps),
    )

    _figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), layout="compressed")
    for data_map, ax in zip(maps, axes, strict=True):
        eplt.plot_array(
            data_map,
            ax=ax,
            cmap="viridis",
            norm=norm,
            aspect="equal",
        )
    kx_limits = (
        min(float(data_map.kx.min()) for data_map in maps[1:]),
        max(float(data_map.kx.max()) for data_map in maps[1:]),
    )
    ky_limits = (
        min(float(data_map.ky.min()) for data_map in maps[1:]),
        max(float(data_map.ky.max()) for data_map in maps[1:]),
    )
    for ax in axes[1:]:
        ax.axvline(0.0, color="w", linestyle="--", linewidth=0.7)
        ax.axhline(0.0, color="w", linestyle="--", linewidth=0.7)
        ax.set(xlim=kx_limits, ylim=ky_limits)
    eplt.set_titles(
        axes,
        ["Angle coordinates", r"Omitted $\xi$ offset", r"$\xi$ offset = $3^\circ$"],
    )
    plt.show()
