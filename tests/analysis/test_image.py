import numpy as np
import pytest
import xarray as xr

import erlab.analysis as era


def test_gaussian_filter() -> None:
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


def test_gaussian_laplace() -> None:
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


def test_laplace() -> None:
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


def test_minimum_gradient() -> None:
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


def test_scaled_laplace() -> None:
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


def test_curvature() -> None:
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


def test_curvature1d() -> None:
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"]) ** 2

    expected_x = xr.DataArray(
        np.array(
            [
                [98.00562, 96.1526, 93.77119, 90.929756, 87.70485],
                [138.605, 134.04039, 128.96048, 123.48621, 117.73539],
                [149.09091, 141.11177, 133.15814, 125.32661, 117.69529],
                [82.74362, 77.44496, 72.39856, 67.618515, 63.111576],
                [46.649776, 43.553825, 40.640636, 37.909084, 35.35534],
            ]
        ),
        dims=["x", "y"],
    )
    expected_y = xr.DataArray(
        np.array(
            [
                [3.9972854, 5.98374, 7.913863, 5.8562593, 3.8705053],
                [3.692493, 5.4577284, 7.042242, 5.090019, 3.3263268],
                [3.044281, 4.457489, 5.651266, 4.0200005, 2.6078095],
                [2.3268301, 3.3887246, 4.2550626, 3.0020146, 1.940408],
                [1.7117059, 2.4875224, 3.1124096, 2.1901085, 1.4142135],
            ]
        ),
        dims=["x", "y"],
    )

    result_x = era.image.curvature1d(darr, "x").astype(np.float32)
    assert np.allclose(result_x, expected_x)

    result_y = era.image.curvature1d(darr, "y").astype(np.float32)
    assert np.allclose(result_y, expected_y)
