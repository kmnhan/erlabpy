import joblib
import numpy as np
import pytest
import scipy.signal
import xarray as xr

from erlab.analysis import correlation


def test_acf2_handles_nans_and_renames_axes() -> None:
    data = xr.DataArray(
        [[1.0, np.nan], [2.0, 3.0]],
        dims=("kx", "ky"),
        coords={"kx": [0.0, 0.5], "ky": [-1.0, 1.0]},
        attrs={"source": "test"},
    )

    result = correlation.acf2(data)

    assert result.dims == ("qx", "qy")
    assert result.attrs["source"] == "test"
    np.testing.assert_allclose(result.qx.values, [-0.5, 0.0, 0.5])
    np.testing.assert_allclose(result.qy.values, [-2.0, 0.0, 2.0])
    assert np.isclose(result.sel(qx=0.0, qy=0.0), 1.0)


def test_acf2stack_invalid_dim_count(monkeypatch) -> None:
    monkeypatch.setattr(joblib.parallel, "DEFAULT_BACKEND", "threading")
    arr = xr.DataArray(np.zeros((2, 2, 2, 2)), dims=("a", "b", "c", "d"))

    with pytest.raises(ValueError, match="number of dimensions"):
        correlation.acf2stack(arr, stack_dims=("a",))


def test_xcorr1d_aligns_coordinate_zero() -> None:
    in1 = xr.DataArray(
        [1.0, 0.0, -1.0],
        dims="x",
        coords={"x": [0.0, 1.0, 2.0]},
    )
    in2 = xr.DataArray(
        [0.0, 2.0, 0.0],
        dims="x",
        coords={"x": [0.0, 2.0, 4.0]},
    )

    result = correlation.xcorr1d(in1, in2, method="direct")

    expected = scipy.signal.correlate(
        in1.fillna(0).values,
        in2.interp_like(in1).fillna(0).values,
        mode="same",
        method="direct",
    )

    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_allclose(result["x"].values, [-1.0, 0.0, 1.0])
