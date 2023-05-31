"""Fast linear interpolation"""

__all__ = ["interp3"]

import math
from collections.abc import Iterable, Sequence

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(nogil=True, cache=True)
def _do_interp1(x, v0, v1):
    return v0 * (1 - x) + v1 * x


@numba.njit(nogil=True, cache=True)
def _do_interp2(x, y, v0, v1, v2, v3):
    return _do_interp1(y, _do_interp1(x, v0, v1), _do_interp1(x, v2, v3))


@numba.njit(nogil=True, cache=True)
def _do_interp3(x, y, z, v0, v1, v2, v3, v4, v5, v6, v7):
    return _do_interp1(
        z, _do_interp2(x, y, v0, v1, v2, v3), _do_interp2(x, y, v4, v5, v6, v7)
    )


@numba.njit(nogil=True, cache=True)
def _calc_interp3(values, v0, v1, v2):
    i0, i1, i2 = math.floor(v0), math.floor(v1), math.floor(v2)
    return _do_interp3(
        v0 - i0,
        v1 - i1,
        v2 - i2,
        values[i0, i1, i2],
        values[i0 + 1, i1, i2],
        values[i0, i1 + 1, i2],
        values[i0 + 1, i1 + 1, i2],
        values[i0, i1, i2 + 1],
        values[i0 + 1, i1, i2 + 1],
        values[i0, i1 + 1, i2 + 1],
        values[i0 + 1, i1 + 1, i2 + 1],
    )


@numba.njit(nogil=True, cache=True)
def _val2ind(val, coord):
    if val > coord[-1] or val < coord[0]:
        return np.nan
    return (val - coord[0]) / (coord[1] - coord[0])


@numba.njit(nogil=True, parallel=True, cache=True)
def _interp3(x, y, z, values, xc, yc, zc, fill_value=np.nan):
    n = len(xc)

    arr_new = np.empty(n, values.dtype)

    for m in numba.prange(n):
        v0 = _val2ind(xc[m], x)
        v1 = _val2ind(yc[m], y)
        v2 = _val2ind(zc[m], z)

        if np.isnan(v0) or np.isnan(v1) or np.isnan(v2):
            arr_new[m] = fill_value
            continue
        else:
            arr_new[m] = _calc_interp3(values, v0, v1, v2)
    return arr_new


def interp3(
    points: Sequence[npt.NDArray],
    values: npt.NDArray,
    xi: Sequence[npt.NDArray] | npt.NDArray,
    fill_value: np.number = np.nan,
):
    """Multidimensional interpolation on evenly spaced coordinates.

    For evenly spaced, strictly ascending, three-dimensional coordinates, this can be
    used as a drop-in replacement for `scipy.interpolate.interpn` in case of linear
    interpolation.

    Parameters
    ----------
    points
        The points defining the regular grid in n dimensions. The points in
        each dimension (i.e. every element of the points tuple) must be strictly
        ascending.
    values
        The data on the regular grid in n dimensions.
    xi
        The coordinates to sample the gridded data at. In addition to the
        scipy-compatible syntax, a tuple of coordinates is also acceptable.
    fill_value
        The value to use for points outside of the interpolation domain.

    Returns
    -------
    values_x : numpy.ndarray
        Interpolated values at input coordinates.

    """
    if isinstance(xi, np.ndarray):
        xi = [xi.T[i].T for i in range(3)]
    return _interp3(
        *points, values, *[c.ravel() for c in xi], fill_value=fill_value
    ).reshape(xi[0].shape)
