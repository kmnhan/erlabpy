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
    xr.testing.assert_identical(result, expected_output)


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
    xr.testing.assert_identical(result2, expected_output2)


def test_boxcar_filter() -> None:
    # Create a test input DataArray
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"])

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [4, 5, 7, 9, 10],
                [10, 12, 14, 16, 17],
                [20, 22, 24, 26, 27],
                [30, 32, 34, 36, 37],
                [37, 38, 40, 42, 44],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the gaussian_filter function
    result = era.image.boxcar_filter(darr, size={"x": 3, "y": 3})

    xr.testing.assert_identical(result, expected_output)


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
    xr.testing.assert_identical(result, expected_output)


def test_diffn() -> None:
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"]) ** 3

    expected_dx1 = xr.DataArray(
        np.array(
            [
                [-2000.0, -1880.0, -1520.0, -920.0, -80.0],
                [4000.0, 5320.0, 6880.0, 8680.0, 10720.0],
                [13000.0, 15520.0, 18280.0, 21280.0, 24520.0],
                [28000.0, 31720.0, 35680.0, 39880.0, 44320.0],
                [46000.0, 50920.0, 56080.0, 61480.0, 67120.0],
            ]
        ),
        dims=["x", "y"],
    )
    expected_dx2 = xr.DataArray(
        np.array(
            [
                [0.0, 1200.0, 2400.0, 3600.0, 4800.0],
                [6000.0, 7200.0, 8400.0, 9600.0, 10800.0],
                [12000.0, 13200.0, 14400.0, 15600.0, 16800.0],
                [18000.0, 19200.0, 20400.0, 21600.0, 22800.0],
                [24000.0, 25200.0, 26400.0, 27600.0, 28800.0],
            ]
        ),
        dims=["x", "y"],
    )

    dx1, dx2 = era.image.diffn(darr, "x", order=(1, 2))
    xr.testing.assert_allclose(dx1, expected_dx1)
    xr.testing.assert_allclose(dx2, expected_dx2)

    xr.testing.assert_allclose(
        era.image.diffn(darr, "y", order=2),
        xr.DataArray(
            np.array(
                [
                    [0.0, 48.0, 96.0, 144.0, 192.0],
                    [240.0, 288.0, 336.0, 384.0, 432.0],
                    [480.0, 528.0, 576.0, 624.0, 672.0],
                    [720.0, 768.0, 816.0, 864.0, 912.0],
                    [960.0, 1008.0, 1056.0, 1104.0, 1152.0],
                ]
            ),
            dims=["x", "y"],
        ),
    )


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
            ],
            dtype=np.float32,
        ),
        dims=["x", "y"],
    )

    # Apply the minimum_gradient function
    result = era.image.minimum_gradient(darr).astype(np.float32)

    # Check if the result matches the expected output
    xr.testing.assert_allclose(result, expected_output)

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
        np.arange(50, step=2).reshape((5, 5)).astype(float) ** 3, dims=["x", "y"]
    )

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [0.0, 1248.0, 2496.0, 3744.0, 4992.0],
                [6240.0, 7488.0, 8736.0, 9984.0, 11232.0],
                [12480.0, 13728.0, 14976.0, 16224.0, 17472.0],
                [18720.0, 19968.0, 21216.0, 22464.0, 23712.0],
                [24960.0, 26208.0, 27456.0, 28704.0, 29952.0],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the scaled_laplace function
    result = era.image.scaled_laplace(darr)

    # Check if the result matches the expected output
    xr.testing.assert_allclose(result, expected_output)


def test_curvature() -> None:
    # Create a test input DataArray
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"]) ** 3

    # Define the expected output
    expected_output = xr.DataArray(
        np.array(
            [
                [0.0, 0.01857236, 0.03715973, 0.05576602, 0.07437308],
                [0.0924652, 0.11049135, 0.12807213, 0.1449815, 0.16094468],
                [0.17562444, 0.18865352, 0.1996744, 0.20832807, 0.21433369],
                [0.21746108, 0.21762464, 0.21491814, 0.20952505, 0.2017888],
                [0.20445243, 0.19285835, 0.18007514, 0.16659571, 0.15289789],
            ]
        ),
        dims=["x", "y"],
    )

    # Apply the curvature function
    result = era.image.curvature(darr)

    # Check if the result matches the expected output
    xr.testing.assert_allclose(result, expected_output)


def test_curvature1d() -> None:
    darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"]) ** 2

    expected_x = xr.DataArray(
        np.array(
            [
                [200.0, 199.48029466, 197.93460578, 195.40241265, 191.94692572],
                [187.65148811, 182.61505885, 176.9472, 170.76299365, 164.17826675],
                [157.30541648, 150.25002525, 143.10835056, 135.96568395, 128.89550438],
                [121.95930664, 115.20696383, 108.67748003, 102.4, 96.39496156],
                [90.67529858, 85.2476256, 80.11335561, 75.26972057, 70.71067812],
            ]
        ),
        dims=["x", "y"],
    )
    expected_y = xr.DataArray(
        np.array(
            [
                [8.0, 7.97921179, 7.91738423, 7.81609651, 7.67787703],
                [7.50605952, 7.30460235, 7.077888, 6.83051975, 6.56713067],
                [6.29221666, 6.01000101, 5.72433402, 5.43862736, 5.15582018],
                [4.87837227, 4.60827855, 4.3470992, 4.096, 3.85579846],
                [3.62701194, 3.40990502, 3.20453422, 3.01078882, 2.82842712],
            ]
        ),
        dims=["x", "y"],
    )

    result_x = era.image.curvature1d(darr, "x")
    xr.testing.assert_allclose(result_x, expected_x)

    result_y = era.image.curvature1d(darr, "y")
    xr.testing.assert_allclose(result_y, expected_y)
