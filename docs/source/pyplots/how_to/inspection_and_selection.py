import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import erlab.plotting as eplt
from erlab.io.exampledata import generate_data


def compare_radial_neighborhoods() -> None:
    data = generate_data(seed=1).T
    center = {"kx": 0.52, "ky": 0.30}
    radii = (0.03, 0.06, 0.09, 0.12)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(radii)]

    _fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="compressed")
    eplt.plot_array(
        data.qsel(eV=-0.2),
        ax=axes[0],
        cmap="Greys",
        gamma=0.5,
        aspect="equal",
    )

    for radius, color in zip(radii, colors, strict=True):
        axes[0].add_patch(
            mpatches.Circle(
                (center["kx"], center["ky"]),
                radius,
                fill=False,
                color=color,
                linewidth=1.2,
            )
        )
        edc = data.qsel.around(radius, **center)
        axes[1].plot(edc.eV, edc, color=color, label=f"{radius:.2f} Å⁻¹")

    axes[0].plot(center["kx"], center["ky"], ".", color="black")
    eplt.fermiline(ax=axes[1], orientation="v", linestyle="--")
    axes[1].set(xlabel=r"$E-E_F$ (eV)", ylabel="Intensity")
    axes[1].legend(title="Radius")
    eplt.set_titles(axes, ["Momentum neighborhoods", "Averaged EDCs"])
    plt.show()


def compare_multidimensional_slices() -> None:
    data = generate_data(seed=1).T
    energies = [-0.4, -0.2, 0.0]
    _fig, axes = plt.subplots(
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
    plt.show()
