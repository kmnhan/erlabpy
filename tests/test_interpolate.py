import numpy as np
import scipy.interpolate
from erlab.analysis.interpolate import interpn


def value_func_3d(x, y, z):
    return 2 * x + 3 * y - z


def test_interpn():
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
