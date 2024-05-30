import erlab.analysis as era
import numpy as np
import pytest
import xarray as xr


def test_gaussian_filter():
    # Create a test input DataArray
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"])

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [3, 5, 7, 8, 10],
                [10, 12, 14, 15, 17],
                [20, 22, 24, 25, 27],
                [29, 31, 33, 34, 36],
                [36, 38, 40, 41, 43],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the gaussian_filter function
    result = era.image.gaussian_filter(darr, sigma={"x": 1.0, "y": 1.0})

    # Check if the result matches the expected output
    assert np.allclose(result, expected_output)


def test_gaussian_laplace():
    # Create a test input DataArray
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"])

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [4, 4, 4, 4, 4],
                [2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0],
                [-2, -2, -2, -2, -2],
                [-4, -4, -4, -4, -4],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the gaussian_laplace function
    result = era.image.gaussian_laplace(darr, sigma={"x": 1.0, "y": 1.0})

    # Check if the result matches the expected output
    assert np.allclose(result, expected_output)

    # Additional test case
    # Define the expected output
    expected_output2 = xr.DataArray(
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the gaussian_laplace function
    result2 = era.image.gaussian_laplace(darr, sigma=2.0)

    # Check if the result matches the expected output
    assert np.allclose(result2, expected_output2)


def test_laplace():
    # Create a test input DataArray
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"])

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [12, 10, 10, 10, 8],
                [2, 0, 0, 0, -2],
                [2, 0, 0, 0, -2],
                [2, 0, 0, 0, -2],
                [-8, -10, -10, -10, -12],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the laplace function
    result = era.image.laplace(darr)

    # Check if the result matches the expected output
    assert np.allclose(result, expected_output)


def test_minimum_gradient():
    # Create a test input DataArray
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"])

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [0.0, 0.00283506, 0.00567012, 0.00850517, 0.01215542],
                [0.01031404, 0.01225726, 0.01430013, 0.01634301, 0.01856527],
                [0.02062807, 0.02247164, 0.02451452, 0.02655739, 0.0288793],
                [0.03094211, 0.03268602, 0.0347289, 0.03677177, 0.03919334],
                [0.06077708, 0.05953621, 0.06237127, 0.06520633, 0.06622662],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the minimum_gradient function
    result = era.image.minimum_gradient(darr).astype(np.float32)

    # Check if the result matches the expected output
    assert np.allclose(result, expected_output)

    # Test decorators that check for input validity
    darr = xr.DataArray(np.arange(5), dims=["x"])
    with pytest.raises(
        ValueError, match="Input must be a 2-dimensional xarray.DataArray"
    ):
        era.image.minimum_gradient(darr)

    darr = xr.DataArray(
        np.arange(50, step=2).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": [0, 1, 2, 4, 5]},
    )
    with pytest.raises(
        ValueError, match="Coordinates for all dimensions must be uniformly spaced"
    ):
        era.image.minimum_gradient(darr)


def test_scaled_laplace():
    # Create a test input DataArray
    darr = xr.DataArray(
        np.arange(50, step=2).reshape((5, 5)).astype(float), dims=["x", "y"]
    )

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [12.0, 10.0, 10.0, 10.0, 8.0],
                [2.0, 0.0, 0.0, 0.0, -2.0],
                [2.0, 0.0, 0.0, 0.0, -2.0],
                [2.0, 0.0, 0.0, 0.0, -2.0],
                [-8.0, -10.0, -10.0, -10.0, -12.0],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the scaled_laplace function
    result = era.image.scaled_laplace(darr)

    # Check if the result matches the expected output
    assert np.allclose(result, expected_output)


def test_curvature():
    # Create a test input DataArray
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"]) ** 2

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [0.11852942, 0.11855069, 0.11778077, 0.11184719, 0.10558571],
                [0.16448492, 0.16107288, 0.15683689, 0.14772279, 0.1385244],
                [0.17403091, 0.16649966, 0.15876051, 0.14719689, 0.1361662],
                [0.09486956, 0.09038468, 0.08598027, 0.0783563, 0.07130584],
                [0.05139264, 0.04942051, 0.04746361, 0.04241876, 0.03781145],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the curvature function
    result = era.image.curvature(darr).astype(np.float32)

    # Check if the result matches the expected output
    assert np.allclose(result, expected_output)
