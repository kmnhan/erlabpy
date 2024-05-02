import numpy as np
import scipy.interpolate
import xarray as xr
from erlab.analysis.interpolate import interpn, slice_along_path


def value_func_1d(x):
    return 2 * x**2 + 1


def value_func_2d(x, y):
    return 2 * x + 3 * y


def value_func_3d(x, y, z):
    return 2 * x + 3 * y - z


def test_interpn_1d():
    x = np.linspace(0, 4, 5)
    points = (x,)
    values = value_func_1d(*np.meshgrid(*points, indexing="ij"))
    point = np.array([[2.21], [2.67]])

    assert np.allclose(
        interpn(points, values, point),
        scipy.interpolate.interpn(
            points, values, point, method="linear", bounds_error=False
        ),
    )


def test_interpn_2d():
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 5, 6)
    points = (x, y)
    values = value_func_2d(*np.meshgrid(*points, indexing="ij"))
    point = np.array([[2.21, 3.12], [2.67, 3.54]])

    assert np.allclose(
        interpn(points, values, point),
        scipy.interpolate.interpn(
            points, values, point, method="linear", bounds_error=False
        ),
    )


def test_interpn_3d():
    x = np.linspace(0, 4, 5)
    y = np.linspace(0, 5, 6)
    z = np.linspace(0, 6, 7)
    points = (x, y, z)
    values = value_func_3d(*np.meshgrid(*points, indexing="ij"))
    point = np.array([[2.21, 3.12, 1.15], [2.67, 3.54, 1.03]])

    assert np.allclose(
        interpn(points, values, point),
        scipy.interpolate.interpn(
            points, values, point, method="linear", bounds_error=False
        ),
    )


def test_slice_along_path():
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    z = np.linspace(0, 10, 11)
    data = np.random.default_rng(1).random((11, 11, 11))
    darr = xr.DataArray(data, coords={"x": x, "y": y, "z": z}, dims=["x", "y", "z"])
    vertices = {"x": [0, 5, 10], "y": [0, 5, 10], "z": [0, 5, 10]}
    interp = slice_along_path(darr, vertices, step_size=1.0)

    np.testing.assert_allclose(
        interp.values,
        np.array(
            [
                0.51182162,
                0.71780756,
                0.59044693,
                0.2016458,
                0.40237743,
                0.44487846,
                0.65585751,
                0.52695829,
                0.77584909,
                0.73380143,
                0.65321432,
                0.80464538,
                0.68497213,
                0.583326,
                0.58202596,
                0.60825019,
                0.21265952,
            ]
        ),
    )

    np.testing.assert_allclose(interp.x.values, interp.y.values)
    np.testing.assert_allclose(interp.x.values, interp.z.values)

    np.testing.assert_allclose(
        interp.x.values,
        np.array(
            [
                0.0,
                0.625,
                1.25,
                1.875,
                2.5,
                3.125,
                3.75,
                4.375,
                5.0,
                5.625,
                6.25,
                6.875,
                7.5,
                8.125,
                8.75,
                9.375,
                10.0,
            ]
        ),
    )
