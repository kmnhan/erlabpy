import numpy as np
import pytest
import xarray as xr
import xarray.testing

from erlab.utils.array import (
    broadcast_args,
    check_arg_has_no_nans,
    is_dims_uniform,
    is_monotonic,
    is_uniform_spaced,
    trim_na,
    uniform_dims,
)


def test_broadcast_args() -> None:
    def testfunc(x, y):
        return x * y

    testfunc_ = broadcast_args(testfunc)

    x_val = np.linspace(0, 1, 5)
    y_val = np.linspace(2, 3, 10)

    expected_vals = testfunc(x_val[:, np.newaxis], y_val[np.newaxis, :])

    np.testing.assert_array_equal(
        testfunc_(x_val[:, np.newaxis], y_val[np.newaxis, :]), expected_vals
    )

    expected = xr.DataArray(expected_vals, coords={"x": x_val, "y": y_val})

    xr.testing.assert_identical(
        testfunc_(
            xr.DataArray(x_val, coords={"x": x_val}),
            xr.DataArray(y_val, coords={"y": y_val}),
        ),
        expected,
    )


def test_is_uniform_spaced() -> None:
    assert is_uniform_spaced([1])
    assert is_uniform_spaced([1, 2, 3, 4])
    assert is_uniform_spaced([1, 2, 3, 4], atol=1e-6)
    assert is_uniform_spaced([1, 2, 3, 4], rtol=1e-6)
    assert not is_uniform_spaced([1, 2, 3, 5])
    assert not is_uniform_spaced([1, 2, 3, 5], atol=1e-6)
    assert not is_uniform_spaced([1, 2, 3, 5], rtol=1e-6)


def test_is_monotonic() -> None:
    assert is_monotonic([])
    assert is_monotonic([1])
    assert is_monotonic([1, 1, 1, 1])
    assert is_monotonic([1, 2, 2, 3])
    assert is_monotonic([1, 2, 3, 3])
    assert is_monotonic([1, 2, 3, 4])
    assert is_monotonic([3, 2, 1, 1])
    assert is_monotonic([3, 2, 2, 1])
    assert is_monotonic([4, 3, 2, 1])
    assert not is_monotonic([1, 2, 3, 2])
    assert not is_monotonic([3, 2, 1, 2])


def test_uniform() -> None:
    # Test case 1: Uniformly spaced dimensions
    darr = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        dims=("x", "y"),
        coords={"x": [1, 2, 3], "y": [1, 2, 3]},
    )
    assert uniform_dims(darr) == {"x", "y"}
    assert is_dims_uniform(darr)

    # Test case 2: Single non-uniformly spaced dimensions
    darr = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        dims=("x", "y"),
        coords={"x": [1, 2, 4], "y": [1, 2, 3]},
    )
    assert uniform_dims(darr) == {"y"}
    assert not is_dims_uniform(darr)

    # Test case 3: Single dimension
    darr = xr.DataArray(np.array([1, 2, 3]), dims=("x",), coords={"x": [1, 2, 3]})
    assert uniform_dims(darr) == {"x"}
    assert is_dims_uniform(darr)

    # Test case 4: No dimensions
    darr = xr.DataArray(np.array(1), dims=(), coords={})
    assert uniform_dims(darr) == set()
    assert is_dims_uniform(darr)


def test_check_arg_has_no_nans() -> None:
    @check_arg_has_no_nans
    def decorated_func(arr):
        pass

    # Test case 1: No NaN values
    arr = np.array([1, 2, 3])
    decorated_func(arr)

    # Test case 2: NaN values present
    arr = np.array([1, np.nan, 3])
    with pytest.raises(ValueError, match="Input must not contain any NaN values"):
        decorated_func(arr)

    # Test case 3: NaN values present, DataArray
    arr = xr.DataArray([1, np.nan, 3])
    with pytest.raises(ValueError, match="Input must not contain any NaN values"):
        decorated_func(arr)

    # Test case 4: Empty array
    arr = np.array([])
    assert decorated_func(arr) is None


def test_trim_na() -> None:
    # Test case 1: Trim along all dimensions
    darr = xr.DataArray(
        np.array([[np.nan, 2, 3], [np.nan, 5, 6], [np.nan, np.nan, np.nan]]),
        dims=("x", "y"),
        coords={"x": [1, 2, 3], "y": [1, 2, 3]},
    )
    expected_result = xr.DataArray(
        np.array([[2, 3], [5, 6]]), dims=("x", "y"), coords={"x": [1, 2], "y": [2, 3]}
    )
    xarray.testing.assert_identical(trim_na(darr), expected_result)

    darr = xr.DataArray(
        np.array([[np.nan, 2, 3], [np.nan, 5, 6], [np.nan, 8, 9]]),
        dims=("x", "y"),
        coords={"x": [1, 2, 3], "y": [1, 2, 3]},
    )
    expected_result = xr.DataArray(
        np.array([[2, 3], [5, 6], [8, 9]]),
        dims=("x", "y"),
        coords={"x": [1, 2, 3], "y": [2, 3]},
    )
    xarray.testing.assert_identical(trim_na(darr), expected_result)

    # Test case 2: Trim along specific dimension
    xarray.testing.assert_identical(trim_na(darr, dims=("x",)), darr)

    # Test case 3: No NaN values
    xarray.testing.assert_identical(trim_na(darr.fillna(0)), darr.fillna(0))

    # Test case 4: Size 1 and size 0 arrays
    darr1 = xr.DataArray(np.array([np.nan]), dims=("x",), coords={"x": [1]})
    darr0 = xr.DataArray(np.array([]), dims=("x",), coords={"x": []})
    xarray.testing.assert_identical(trim_na(darr1), darr0)
    xarray.testing.assert_identical(trim_na(darr0), darr0)
