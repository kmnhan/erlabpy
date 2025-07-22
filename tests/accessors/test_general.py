import holoviews.element.chart
import lmfit
import matplotlib.image
import matplotlib.lines
import numpy as np
import panel.layout.base
import pytest
import xarray as xr

import erlab.accessors  # noqa: F401


def test_da_qplot() -> None:
    dat = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))

    assert isinstance(dat.qplot(), matplotlib.image.AxesImage)
    assert isinstance(dat[0].qplot()[0], matplotlib.lines.Line2D)


def test_da_qshow_hvplot() -> None:
    # Imagetool from 2D data is tested in interactive/test_imagetool.py
    # Hvplot curve from 1D data is tested here
    dat = xr.DataArray(
        np.arange(25) ** 2.0, coords={"x": np.arange(25) * 1.0}, name="1D"
    )
    assert isinstance(dat.qshow(), holoviews.element.chart.Curve)


@pytest.mark.parametrize("plot_components", [True, False])
def test_ds_qshow_fit(plot_components: bool) -> None:
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

    # Construct DataArray to fit
    darr = xr.DataArray(
        y, dims=["beta", "alpha"], coords={"beta": beta, "alpha": alpha}
    )
    result_ds = darr.xlm.modelfit(
        coords="alpha",
        model=lmfit.models.GaussianModel() + lmfit.models.LinearModel(),
        params={"center": xr.DataArray([-2, 0, 2], coords=[darr.beta]), "slope": -0.1},
    )
    assert isinstance(
        result_ds.qshow(plot_components=plot_components), panel.layout.base.Column
    )

    result_ds.qshow.params()

    # Test with a Dataset
    ds = darr.rename("testvar").to_dataset()
    result_ds = ds.xlm.modelfit(
        coords="alpha",
        model=lmfit.models.GaussianModel() + lmfit.models.LinearModel(),
        params={"center": xr.DataArray([-2, 0, 2], coords=[ds.beta]), "slope": -0.1},
    )
    assert "testvar_modelfit_results" in result_ds
    assert "testvar_modelfit_stats" in result_ds
    assert "testvar_modelfit_data" in result_ds

    result_ds.qshow(plot_components=plot_components)
    result_ds.qshow.params()

    with pytest.raises(
        ValueError,
        match="Fit results for data variable `some_nonexistent_var` "
        "were not found in the Dataset.",
    ):
        result_ds.qshow.params("some_nonexistent_var")


@pytest.mark.parametrize(
    ("indexers", "expected_shape"),
    [
        ({"x": 2.0}, (5,)),
        ({"x": slice(1.0, 3.0)}, (3, 5)),
        ({"x": 2.0, "x_width": 1.0}, (5,)),
        ({"x": slice(1.0, 3.0), "y": 2.0}, (3,)),
    ],
)
def test_qsel_shape(indexers, expected_shape) -> None:
    dat = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
    result = dat.qsel(indexers)
    assert result.shape == expected_shape


def test_qsel_invalid_dimension() -> None:
    dat = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
    with pytest.raises(ValueError, match="Dimension `z` not found in data"):
        dat.qsel({"z": 2.0})


def test_qsel_collection() -> None:
    dat = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))

    # List of values & width
    xr.testing.assert_identical(
        dat.qsel(x=[1, 3], x_width=2),
        xr.DataArray(
            np.array([[5, 6, 7, 8, 9], [15, 16, 17, 18, 19]], dtype=float),
            dims=("x", "y"),
        ),
    )

    # List of values & width for both dimensions
    xr.testing.assert_equal(
        dat.qsel(x=[1, 3], x_width=2, y=[1, 3], y_width=2),
        xr.DataArray(
            np.array([[6, 8], [16, 18]], dtype=float),
            dims=("x", "y"),
        ),
    )

    # Check dim order consistency
    xr.testing.assert_equal(
        dat.qsel(y=[1, 3], y_width=2, x=[1, 3], x_width=2),
        xr.DataArray(
            np.array([[6, 8], [16, 18]], dtype=float),
            dims=("x", "y"),
        ),
    )


def test_qsel_slice_with_width() -> None:
    dat = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
    with pytest.raises(
        ValueError,
        match="Slice not allowed for value of dimension `x` with width specified",
    ):
        dat.qsel({"x": slice(1.0, 3.0), "x_width": 1.0})


def test_qsel_associated_dim() -> None:
    # 1D associated coordinate
    dat = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("x", "y"),
        coords={"x": np.arange(5), "y": np.arange(5), "z": ("x", np.arange(5))},
    )
    xr.testing.assert_identical(
        dat.qsel(x=2, x_width=3),
        xr.DataArray(
            np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
            dims=("y",),
            coords={"y": np.arange(5), "x": 2.0, "z": 2.0},
        ),
    )

    # 2D associated coordinate
    dat = xr.DataArray(
        np.arange(5 * 4 * 3).reshape(5, 4, 3).astype(float),
        dims=("x", "y", "z"),
        coords={
            "x": np.arange(5),
            "y": np.arange(4),
            "z": np.arange(3),
            "w": (["x", "y"], np.arange(5 * 4).reshape(5, 4)),
        },
    )

    expected = xr.DataArray(
        np.array(
            [
                [24.0, 25.0, 26.0],
                [27.0, 28.0, 29.0],
                [30.0, 31.0, 32.0],
                [33.0, 34.0, 35.0],
            ]
        ),
        dims=("y", "z"),
        coords={
            "y": np.arange(4),
            "z": np.arange(3),
            "x": 2.0,
            "w": (["y"], np.array([8.0, 9.0, 10.0, 11.0])),
        },
    )

    xr.testing.assert_identical(dat.qsel(x=2, x_width=3), expected)


def test_qsel_value_outside_bounds() -> None:
    dat = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("x", "y"),
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    with pytest.warns(
        UserWarning, match="Selected value 10.0 for `x` is outside coordinate bounds"
    ):
        dat.qsel({"x": 10.0})


def test_qsel_drop_unindexed_dims() -> None:
    dat = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
    result = dat.qsel({"x": 2.0})

    xr.testing.assert_equal(result, dat.isel(x=2))


def test_qsel_around() -> None:
    dat = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
    result = dat.qsel.around(radius=2.0, x=2.0, y=2.0, average=False)
    xr.testing.assert_identical(
        result,
        xr.DataArray(
            np.array(
                [
                    [np.nan, np.nan, 2.0, np.nan, np.nan],
                    [np.nan, 6.0, 7.0, 8.0, np.nan],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [np.nan, 16.0, 17.0, 18.0, np.nan],
                    [np.nan, np.nan, 22.0, np.nan, np.nan],
                ]
            ),
            dims=("x", "y"),
        ),
    )

    xr.testing.assert_identical(
        dat.qsel.around(radius=2.0, x=2.0, y=2.0, average=True), xr.DataArray(12.0)
    )


def test_qsel_average_single_dim() -> None:
    # Create a simple 2D DataArray with dims 'x' and 'y'

    x = np.array([10, 20])
    y = np.array([30, 40, 50])
    data = np.array([[1, 2, 3], [4, 5, 6]])
    da = xr.DataArray(data, dims=("x", "y"), coords={"x": x, "y": y})

    # Average over the 'x' dimension using qsel.average.
    # The expected result is the mean along 'x' and retaining the averaged coordinate.
    # Expected mean data: [[(1+4)/2, (2+5)/2, (3+6)/2]] = [[2.5, 3.5, 4.5]]
    expected = data.mean(axis=0)
    result = da.qsel.average("x")

    # After averaging, 'x' should not be a dimension; it is stored as a coordinate.
    assert "x" not in result.dims
    # Compare the resulting data with the expected average.
    np.testing.assert_allclose(result.data, expected)
    # Check that the coordinate 'x' is the mean of the original x-values.
    np.testing.assert_allclose(result.coords["x"].data, x.mean())


def test_qsel_average_multiple_dim() -> None:
    # Create a simple 2D DataArray with dims 'x' and 'y'

    x = np.array([0, 10, 20])
    y = np.array([100, 200])
    data = np.array([[1, 2], [4, 5], [7, 8]])
    da = xr.DataArray(data, dims=("x", "y"), coords={"x": x, "y": y})

    # Average over both 'x' and 'y'
    # Expected result is a scalar value: mean of all data.
    expected = data.mean()
    result = da.qsel.average(["x", "y"])

    # The resulting DataArray should have no dimensions.
    assert not result.dims
    # Data should be equal to the overall mean.
    np.testing.assert_allclose(result.data, expected)
    # Coordinates are retained as scalars equal to the mean of the original coords.
    np.testing.assert_allclose(result.coords["x"].data, x.mean())
    np.testing.assert_allclose(result.coords["y"].data, y.mean())


def test_qsel_average_invalid_dim() -> None:
    # Create a simple 1D DataArray

    x = np.array([0, 1, 2])
    data = np.array([10, 20, 30])
    da = xr.DataArray(data, dims=("x",), coords={"x": x})

    # Averaging over an invalid dimension should raise a ValueError
    with pytest.raises(ValueError, match="Dimension `z` not found in data"):
        _ = da.qsel.average("z")
