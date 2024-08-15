import numpy as np
import pytest
import xarray as xr
from erlab.utils.array import (
    check_arg_has_no_nans,
    is_dims_uniform,
    is_monotonic,
    is_uniform_spaced,
    uniform_dims,
)


def test_is_uniform_spaced():
    assert is_uniform_spaced([1])
    assert is_uniform_spaced([1, 2, 3, 4])
    assert is_uniform_spaced([1, 2, 3, 4], atol=1e-6)
    assert is_uniform_spaced([1, 2, 3, 4], rtol=1e-6)
    assert not is_uniform_spaced([1, 2, 3, 5])
    assert not is_uniform_spaced([1, 2, 3, 5], atol=1e-6)
    assert not is_uniform_spaced([1, 2, 3, 5], rtol=1e-6)


def test_is_monotonic():
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


def test_uniform():
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


def test_check_arg_has_no_nans():
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
