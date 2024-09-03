import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from pyqtgraph.colormap import modulatedBarData

import erlab.plotting.erplot as eplt


def sample_plot(norms, kw0, kw1, cmap):
    if isinstance(kw0, dict):
        kw0 = (kw0,) * len(norms)
    if isinstance(kw1, dict):
        kw1 = (kw1,) * len(norms)
    num = len(norms)

    _, axs = plt.subplots(
        num,
        1,
        layout="constrained",
        figsize=eplt.figwh(),
        squeeze=False,
    )

    bar_data = modulatedBarData(384, 256)
    for ax, norm, k0, k1 in zip(axs.flat, norms, kw0, kw1, strict=True):
        ax.imshow(
            bar_data,
            extent=(0, 1, 0, 1),
            aspect="auto",
            interpolation="none",
            rasterized=True,
            cmap=cmap,
            norm=norm(**k0, **k1),
        )
        eplt.proportional_colorbar(ax=ax)

    eplt.unify_clim(axs)
    plt.close()


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1, 2.0, 10.0])
def test_InversePowerNorm(gamma):
    cmap = "Greys"
    sample_plot([eplt.InversePowerNorm], {"vmin": 0, "vmax": 1}, {"gamma": gamma}, cmap)
    plt.close()


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1, 2.0, 10.0])
def test_norms_diverging(gamma):
    cmap = "RdYlBu"
    sample_plot(
        [
            eplt.TwoSlopePowerNorm,
            eplt.TwoSlopeInversePowerNorm,
            eplt.CenteredPowerNorm,
            eplt.CenteredInversePowerNorm,
        ],
        [{}, {}, {"halfrange": 0.5}, {"halfrange": 0.5}],
        {"gamma": gamma, "vcenter": 0.5},
        cmap,
    )
    plt.close()


@pytest.mark.parametrize("background", [None, "black"])
@pytest.mark.parametrize(
    "cnorm", [None, eplt.CenteredInversePowerNorm(0.7, vcenter=0.0, halfrange=16.0)]
)
@pytest.mark.parametrize("lnorm", [None, eplt.InversePowerNorm(0.5)])
@pytest.mark.parametrize("cmap", [None, "bwr"])
def test_2d_cmap(cmap, lnorm, cnorm, background):
    test = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]).astype(
        np.float64
    )

    test_t = test.copy(data=test.values.T)

    _, cb = eplt.plot_array_2d(
        test + test_t,
        test - test_t,
        cmap=cmap,
        lnorm=lnorm,
        cnorm=cnorm,
        background=background,
    )
    assert cb.ax.get_ylim() == (-16.0, 16.0)
    assert cb.ax.get_xlim() == (0.0, 48.0)

    plt.close()
