import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pyqtgraph.colormap import modulatedBarData

import erlab.plotting.erplot as eplt

plt.style.use("khan")

sns.set_palette("Set2")

gamma = 0.3


def example_1():
    cmap = "Greys"
    sample_plot(
        [mcolors.Normalize, mcolors.PowerNorm, eplt.InversePowerNorm],
        [
            "Linear",
            f"matplotlib.colors.PowerNorm ($\\gamma={gamma}$)",
            f"InversePowerNorm ($\\gamma={gamma}$)",
        ],
        dict(vmin=0, vmax=1),
        [dict(), dict(gamma=gamma), dict(gamma=gamma)],
        cmap,
    )


def example_2():
    cmap = "RdYlBu"
    sample_plot(
        [mcolors.CenteredNorm, eplt.CenteredPowerNorm, eplt.CenteredInversePowerNorm],
        [
            "matplotlib.colors.CenteredNorm",
            f"CenteredPowerNorm ($\\gamma={gamma}$)",
            f"CenteredInversePowerNorm ($\\gamma={gamma}$)",
        ],
        [dict(halfrange=0.5), dict(halfrange=0.5), dict(halfrange=0.5)],
        [
            dict(vcenter=0.5),
            dict(gamma=gamma, vcenter=0.5),
            dict(gamma=gamma, vcenter=0.5),
        ],
        cmap,
    )


def sample_plot(norms, labels, kw0, kw1, cmap):
    if isinstance(kw0, dict):
        kw0 = (kw0, kw0, kw0)
    if isinstance(kw1, dict):
        kw1 = (kw1, kw1, kw1)
    x = np.linspace(0, 1, 2048)
    num = len(norms)

    _, axs = plt.subplots(
        1,
        num + 1,
        width_ratios=[9 - num] + [1] * num,
        layout="constrained",
        figsize=eplt.figwh(),
    )

    for norm, label, k0, k1 in zip(norms, labels, kw0, kw1):
        axs[0].plot(x, norm(**k0, **k1)(x), label=label)

    bar_data = modulatedBarData(384, 256)
    for i, (ax, norm, k1) in enumerate(zip(axs[1:], norms, kw1)):
        ax.plot(
            0.5,
            1,
            "o",
            c="k",
            mew=0.5,
            ms=5,
            mfc=f"C{i}",
            transform=ax.transAxes,
            zorder=10,
            clip_on=False,
        )
        ax.imshow(
            bar_data,
            extent=(0, 1, 0, 1),
            aspect="auto",
            interpolation="none",
            rasterized=True,
            cmap=cmap,
            norm=norm(**k1),
        )
        ax.yaxis.tick_right()
        ax.set_xticks([])

    for ax in axs[1:-1]:
        ax.set_yticklabels([])

    axs[0].set_xlim([0, 1])
    axs[0].sharey(axs[-1])
    axs[0].legend(
        fontsize="x-small",
        framealpha=0.8,
        edgecolor="0.8",
        fancybox=True,
        loc="lower right",
    )
    plt.show()
