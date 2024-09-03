import holoviews.element.chart
import lmfit
import matplotlib.image
import matplotlib.lines
import numpy as np
import panel.layout.base
import pytest
import xarray as xr

import erlab.accessors  # noqa: F401


def test_da_qplot():
    dat = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))

    assert isinstance(dat.qplot(), matplotlib.image.AxesImage)
    assert isinstance(dat[0].qplot()[0], matplotlib.lines.Line2D)


def test_da_qshow_hvplot():
    # Imagetool from 2D data is tested in interactive/test_imagetool.py
    # Hvplot curve from 1D data is tested here
    dat = xr.DataArray(
        np.arange(25) ** 2.0, coords={"x": np.arange(25) * 1.0}, name="1D"
    )
    assert isinstance(dat.qshow(), holoviews.element.chart.Curve)


@pytest.mark.parametrize("plot_components", [True, False])
def test_ds_qshow_fit(plot_components: bool):
    # Define angle coordinates for 2D data
    alpha = np.linspace(-5.0, 5.0, 100)
    beta = np.linspace(-1.0, 1.0, 3)

    # Center of the peaks along beta
    center = np.array([-2.0, 0.0, 2.0])[:, np.newaxis]

    # Gaussian peak on a linear background
    y = -0.1 * alpha + 2 + 3 * np.exp(-((alpha - center) ** 2) / (2 * 1**2))

    # Add some noise with fixed seed for reproducibility
    rng = np.random.default_rng(5)
    yerr = np.full_like(y, 0.05)
    y = rng.normal(y, yerr)

    # Construct DataArray
    darr = xr.DataArray(
        y, dims=["beta", "alpha"], coords={"beta": beta, "alpha": alpha}
    )

    result_ds = darr.modelfit(
        coords="alpha",
        model=lmfit.models.GaussianModel() + lmfit.models.LinearModel(),
        params={"center": xr.DataArray([-2, 0, 2], coords=[darr.beta]), "slope": -0.1},
    )
    assert isinstance(
        result_ds.qshow(plot_components=plot_components), panel.layout.base.Column
    )
