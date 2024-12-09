import numpy as np
import pytest
import scipy.interpolate
import xarray as xr

from erlab.analysis.interpolate import interpn, slice_along_path, slice_along_vector


def value_func_1d(x):
    return 2 * x**2 + 1


def value_func_2d(x, y):
    return 2 * x + 3 * y


def value_func_3d(x, y, z):
    return 2 * x + 3 * y - z


def value_func_4d(x, y, z, w):
    return 2 * x + 3 * y - z + w


x = np.linspace(0, 4, 5)
y = np.linspace(0, 5, 6)
z = np.linspace(0, 6, 7)
w = np.linspace(0, 7, 8)

values_1d = value_func_1d(*np.meshgrid(x, indexing="ij"))
values_2d = value_func_2d(*np.meshgrid(x, y, indexing="ij"))
values_3d = value_func_3d(*np.meshgrid(x, y, z, indexing="ij"))
values_4d = value_func_4d(*np.meshgrid(x, y, z, w, indexing="ij"))


@pytest.mark.parametrize("values", [values_1d, values_2d, values_3d, values_4d])
@pytest.mark.parametrize("point", [np.array([2.21, 2.67]), np.array([[2.21], [2.67]])])
def test_interpn_1d(values, point) -> None:
    points = (x,)
    assert np.allclose(
        interpn(points, values, point),
        scipy.interpolate.interpn(
            points, values, point, method="linear", bounds_error=False
        ),
    )


@pytest.mark.parametrize("values", [values_2d, values_3d, values_4d])
@pytest.mark.parametrize(
    "point", [np.array([2.21, 3.12]), np.array([[2.21, 3.12], [2.67, 3.54]])]
)
def test_interpn_2d(values, point) -> None:
    points = (x, y)
    assert np.allclose(
        interpn(points, values, point),
        scipy.interpolate.interpn(
            points, values, point, method="linear", bounds_error=False
        ),
    )


@pytest.mark.parametrize("values", [values_3d, values_4d])
@pytest.mark.parametrize("point", [np.array([[2.21, 3.12, 1.15], [2.67, 3.54, 1.03]])])
def test_interpn_3d(values, point) -> None:
    points = (x, y, z)
    assert np.allclose(
        interpn(points, values, point),
        scipy.interpolate.interpn(
            points, values, point, method="linear", bounds_error=False
        ),
    )


def test_slice_along_path() -> None:
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


def test_slice_along_vector() -> None:
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    z = np.linspace(0, 1, 5)

    darr = xr.DataArray(
        np.random.default_rng(1).random((5, 5, 5)),
        coords={"x": x, "y": y, "z": z},
        dims=["x", "y", "z"],
    )

    # Test with tuple stretch
    interp = slice_along_vector(
        darr,
        center={"x": 0.5, "y": 0.3},
        direction={"x": np.sqrt(3), "y": 1.0},
        stretch=(0.1, 0.3),
    )

    np.testing.assert_allclose(
        interp.values,
        np.array(
            [
                [0.23207336, 0.59910469, 0.77890933, 0.84194666, 0.52078597],
                [0.18583922, 0.68822936, 0.28374755, 0.36078411, 0.42699386],
            ]
        ),
    )
    np.testing.assert_allclose(interp.x.values, np.array([0.41339746, 0.75980762]))
    np.testing.assert_allclose(interp.y.values, np.array([0.25, 0.45]))
    np.testing.assert_allclose(interp.z.values, z)
    np.testing.assert_allclose(interp.path.values, np.array([0.0, 0.4]))

    # Test with scalar stretch

    interp = slice_along_vector(
        darr,
        center={"x": 0.5, "y": 0.3},
        direction={"x": np.sqrt(3), "y": 1.0},
        stretch=0.2,
    )
    np.testing.assert_allclose(
        interp.values,
        np.array(
            [
                [0.4484835, 0.39773655, 0.60736742, 0.71743957, 0.60188075],
                [0.26960418, 0.5175887, 0.50620098, 0.45499338, 0.57402158],
            ]
        ),
    )
    np.testing.assert_allclose(interp.x.values, np.array([0.32679492, 0.67320508]))
    np.testing.assert_allclose(interp.y.values, np.array([0.2, 0.4]))
    np.testing.assert_allclose(interp.z.values, z)
    np.testing.assert_allclose(interp.path.values, np.array([0.0, 0.4]))

    # Test with 3D input
    interp = slice_along_vector(
        darr,
        center={"x": 0.5, "y": 0.3, "z": 0.5},
        direction={"x": np.sqrt(3), "y": 1.0, "z": 1.0},
        stretch=0.2,
    )
    np.testing.assert_allclose(interp.values, np.array([0.56640928, 0.52730286]))
    np.testing.assert_allclose(interp.x.values, np.array([0.34508067, 0.65491933]))
    np.testing.assert_allclose(interp.y.values, np.array([0.21055728, 0.38944272]))
    np.testing.assert_allclose(interp.path.values, np.array([0.0, 0.4]))
