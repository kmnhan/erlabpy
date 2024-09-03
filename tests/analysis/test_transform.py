import numpy as np
import pytest
import xarray as xr
import xarray.testing

from erlab.analysis.transform import rotate, shift


def test_rotate():
    input_arr = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0]},
    )
    expected_output = xr.DataArray(
        np.array([[3, 7, 11], [2, 6, 10], [1, 5, 9], [0, 4, 8]], dtype=float),
        dims=("y", "x"),
        coords={"y": [-3.0, -2.0, -1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )

    xarray.testing.assert_allclose(
        rotate(input_arr, 90, reshape=True, order=1), expected_output
    )
    xarray.testing.assert_allclose(
        rotate(
            input_arr,
            90,
            axes=("y", "x"),
            center={"x": 0, "y": 0},
            reshape=True,
            order=1,
        ),
        expected_output,
    )
    xarray.testing.assert_allclose(
        rotate(input_arr, 90, center={"x": 3, "y": 2}, reshape=True, order=1),
        xr.DataArray(
            np.array([[3, 7, 11], [2, 6, 10], [1, 5, 9], [0, 4, 8]], dtype=float),
            dims=("y", "x"),
            coords={"y": [2.0, 3.0, 4.0, 5.0], "x": [1.0, 2.0, 3.0]},
        ),
    )

    xarray.testing.assert_allclose(
        rotate(input_arr, 90, reshape=False, order=1),
        xr.DataArray(
            np.array(
                [
                    [0, 4, 8, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ],
                dtype=float,
            ),
            dims=("y", "x"),
            coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0]},
        ),
    )

    # Higher dimensional array
    input_arr = xr.DataArray(
        np.arange(24).reshape((3, 4, 2)).astype(float),
        dims=("y", "x", "z"),
        coords={"y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0, 3.0], "z": [0.0, 1.0]},
    )
    xarray.testing.assert_allclose(
        rotate(input_arr, 90, reshape=True, order=1),
        xr.DataArray(
            np.array(
                [
                    [[6, 7], [14, 15], [22, 23]],
                    [[4, 5], [12, 13], [20, 21]],
                    [[2, 3], [10, 11], [18, 19]],
                    [[0, 1], [8, 9], [16, 17]],
                ],
                dtype=float,
            ),
            dims=("y", "x", "z"),
            coords={
                "y": [-3.0, -2.0, -1.0, 0.0],
                "x": [0.0, 1.0, 2.0],
                "z": [0.0, 1.0],
            },
        ),
    )

    # Test with associated coordinates
    input_arr = xr.DataArray(
        np.arange(12).reshape((3, 4)).astype(float),
        dims=("y", "x"),
        coords={
            "y": [0.0, 1.0, 2.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "yy": ("y", [0.0, 1.0, 2.0]),
        },
    )
    expected_output = xr.DataArray(
        np.array([[3, 7, 11], [2, 6, 10], [1, 5, 9], [0, 4, 8]], dtype=float),
        dims=("y", "x"),
        coords={"y": [-3.0, -2.0, -1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )

    # Catch exceptions
    with pytest.raises(ValueError, match="input array should be at least 2D"):
        rotate(xr.DataArray(np.arange(5)), 90)

    with pytest.raises(ValueError, match="center must have keys that match axes"):
        rotate(input_arr, 90, center={"x": 0, "z": 0})

    with pytest.raises(
        ValueError, match="all coordinates along axes must be evenly spaced"
    ):
        rotate(
            xr.DataArray(
                np.arange(12).reshape((3, 4)).astype(float),
                dims=("y", "x"),
                coords={"y": [0.0, 1.0, 3.0], "x": [0.0, 1.0, 2.0, 3.0]},
            ),
            90,
        )


def test_shift():
    # Create a test input DataArray
    darr = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(float), dims=["x", "y"]
    )

    # Create a test shift DataArray
    shift_arr = xr.DataArray([1, 0, 2], dims=["x"])

    # Perform the shift operation
    shifted = shift(darr, shift_arr, along="y")

    # Define the expected result
    expected = xr.DataArray(
        np.array([[np.nan, 1.0, 2.0], [4.0, 5.0, 6.0], [np.nan, np.nan, 7.0]]),
        dims=["x", "y"],
    )

    # Check if the shifted array matches the expected result
    assert np.allclose(shifted, expected, equal_nan=True)
