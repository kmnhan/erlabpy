import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from pyqtgraph.colormap import modulatedBarData

import erlab.plotting as eplt


def sample_plot(norms, kw0, kw1, cmap):
    if isinstance(kw0, dict):
        kw0 = (kw0,) * len(norms)
    if isinstance(kw1, dict):
        kw1 = (kw1,) * len(norms)
    num = len(norms)

    _, axs = plt.subplots(
        num, 1, layout="constrained", figsize=eplt.figwh(), squeeze=False
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
    return axs


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1, 2.0, 10.0])
def test_InversePowerNorm(gamma) -> None:
    cmap = "Greys"
    sample_plot([eplt.InversePowerNorm], {"vmin": 0, "vmax": 1}, {"gamma": gamma}, cmap)
    plt.close()


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1, 2.0, 10.0])
def test_norms_diverging(gamma) -> None:
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
def test_2d_cmap(cmap, lnorm, cnorm, background) -> None:
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


@pytest.mark.parametrize("autoscale", [True, False])
def test_unify_clim(autoscale) -> None:
    fig, axs = plt.subplots(2, 2)

    rng = np.random.default_rng(1)

    data1 = rng.random((10, 10))
    data2 = rng.random((10, 10)) * 2
    data3 = rng.random((10, 10)) * 3
    data4 = rng.random((10, 10)) * 4

    im1 = axs[0, 0].imshow(data1, cmap="viridis")
    im2 = axs[0, 1].imshow(data2, cmap="viridis")
    im3 = axs[1, 0].imshow(data3, cmap="viridis")
    im4 = axs[1, 1].imshow(data4, cmap="viridis")

    eplt.unify_clim(axs, autoscale=autoscale)

    vmin = min(im1.norm.vmin, im2.norm.vmin, im3.norm.vmin, im4.norm.vmin)
    vmax = max(im1.norm.vmax, im2.norm.vmax, im3.norm.vmax, im4.norm.vmax)

    assert im1.norm.vmin == vmin
    assert im1.norm.vmax == vmax
    assert im2.norm.vmin == vmin
    assert im2.norm.vmax == vmax
    assert im3.norm.vmin == vmin
    assert im3.norm.vmax == vmax
    assert im4.norm.vmin == vmin
    assert im4.norm.vmax == vmax

    plt.close(fig)


def test_unify_clim_with_target() -> None:
    fig, axs = plt.subplots(2, 2)

    rng = np.random.default_rng(1)

    data1 = rng.random((10, 10))
    data2 = rng.random((10, 10)) * 2
    data3 = rng.random((10, 10)) * 3
    data4 = rng.random((10, 10)) * 4

    im1 = axs[0, 0].imshow(data1, cmap="viridis")
    im2 = axs[0, 1].imshow(data2, cmap="viridis")
    im3 = axs[1, 0].imshow(data3, cmap="viridis")
    im4 = axs[1, 1].imshow(data4, cmap="viridis")

    vmin = im3.norm.vmin
    vmax = im3.norm.vmax

    eplt.unify_clim(axs, target=axs[1, 0])

    assert im1.norm.vmin == vmin
    assert im1.norm.vmax == vmax
    assert im2.norm.vmin == vmin
    assert im2.norm.vmax == vmax
    assert im3.norm.vmin == vmin
    assert im3.norm.vmax == vmax
    assert im4.norm.vmin == vmin
    assert im4.norm.vmax == vmax

    eplt.unify_clim(axs, target=im3)

    assert im1.norm.vmin == vmin
    assert im1.norm.vmax == vmax
    assert im2.norm.vmin == vmin
    assert im2.norm.vmax == vmax
    assert im3.norm.vmin == vmin
    assert im3.norm.vmax == vmax
    assert im4.norm.vmin == vmin
    assert im4.norm.vmax == vmax

    plt.close(fig)
