"""Fast linear interpolation."""

__all__ = ["FastInterpolator", "interpn"]

import math
import warnings
from collections.abc import Sequence

import numba
import numpy as np
import scipy.interpolate
import xarray as xr
import xarray.core.missing


@numba.njit(nogil=True, inline="always")
def _do_interp1(x, v0, v1):
    return v0 * (1 - x) + v1 * x


@numba.njit(nogil=True, inline="always")
def _do_interp2(x, y, v0, v1, v2, v3):
    return _do_interp1(y, _do_interp1(x, v0, v1), _do_interp1(x, v2, v3))


@numba.njit(nogil=True, inline="always")
def _do_interp3(x, y, z, v0, v1, v2, v3, v4, v5, v6, v7):
    return _do_interp1(
        z, _do_interp2(x, y, v0, v1, v2, v3), _do_interp2(x, y, v4, v5, v6, v7)
    )


# @numba.njit
# def _do_interp2(x, y, v0, v1, v2, v3):
#     return _do_interp1(y, v0 * (1 - x) + v1 * x, v2 * (1 - x) + v3 * x)


# @numba.njit
# def _do_interp3(x, y, z, v0, v1, v2, v3, v4, v5, v6, v7):
#     return (
#         _do_interp2(x, y, v0, v1, v2, v3) * (1 - z)
#         + _do_interp2(x, y, v4, v5, v6, v7) * z
#     )


@numba.njit(nogil=True, inline="always")
def _calc_interp2(values, v0, v1):
    i0, i1 = math.floor(v0), math.floor(v1)
    n0, n1 = values.shape
    j0, j1 = min(i0 + 1, n0 - 1), min(i1 + 1, n1 - 1)
    return _do_interp2(
        v0 - i0,
        v1 - i1,
        values[i0, i1],
        values[j0, i1],
        values[i0, j1],
        values[j0, j1],
    )


@numba.njit(nogil=True, inline="always")
def _calc_interp3(values, v0, v1, v2):
    i0, i1, i2 = math.floor(v0), math.floor(v1), math.floor(v2)
    n0, n1, n2 = values.shape
    j0, j1, j2 = min(i0 + 1, n0 - 1), min(i1 + 1, n1 - 1), min(i2 + 1, n2 - 1)
    return _do_interp3(
        v0 - i0,
        v1 - i1,
        v2 - i2,
        values[i0, i1, i2],
        values[j0, i1, i2],
        values[i0, j1, i2],
        values[j0, j1, i2],
        values[i0, i1, j2],
        values[j0, i1, j2],
        values[i0, j1, j2],
        values[j0, j1, j2],
    )


@numba.njit(nogil=True, inline="always")
def _val2ind(val, coord):
    if val > coord[-1] or val < coord[0]:
        return np.nan
    else:
        return np.divide(val - coord[0], coord[1] - coord[0])


@numba.njit(nogil=True, parallel=True)
def _interp2(x, y, values, xc, yc, fill_value=np.nan):
    n = len(xc)

    arr_new = np.empty(n, values.dtype)

    for m in numba.prange(n):
        v0, v1 = _val2ind(xc[m], x), _val2ind(yc[m], y)

        if np.isnan(v0) or np.isnan(v1):
            arr_new[m] = fill_value
        else:
            arr_new[m] = _calc_interp2(values, v0, v1)
    return arr_new


@numba.njit(nogil=True, parallel=True)
def _interp3(x, y, z, values, xc, yc, zc, fill_value=np.nan):
    n = len(xc)

    arr_new = np.empty(n, values.dtype)

    for m in numba.prange(n):
        v0, v1, v2 = _val2ind(xc[m], x), _val2ind(yc[m], y), _val2ind(zc[m], z)

        if np.isnan(v0) or np.isnan(v1) or np.isnan(v2):
            arr_new[m] = fill_value
        else:
            arr_new[m] = _calc_interp3(values, v0, v1, v2)
    return arr_new


def _get_interp_func(ndim):
    if ndim == 3:
        return _interp3
    elif ndim == 2:
        return _interp2
    else:
        raise ValueError("Fast interpolation only supported for 2D or 3D")


def _check_even(arr):
    dif = np.diff(arr)
    return np.allclose(dif, dif[0])


class FastInterpolator(scipy.interpolate.RegularGridInterpolator):
    """Fast linear multidimensional interpolation on evenly spaced coordinates.

    This is an extension from `scipy.interpolate.RegularGridInterpolator` with vast
    performance improvements and integration with `xarray`.

    The input arguments are identical to `scipy.interpolate.RegularGridInterpolator`
    except `bounds_error`, which is set to `False` by default.

    Performance improvements are enabled for 2D and 3D linear interpolation on uniformly
    spaced coordinates with extrapolation disabled. Otherwise,
    `scipy.interpolate.RegularGridInterpolator` is called. See below for more
    information.

    Note
    ----
    Parallel acceleration is only applied when all of the following are true.

    * `method` is ``"linear"``.
    * Coordinates along all dimensions are evenly spaced.
    * Values are 2D or 3D.
    * Extrapolation is disabled, i.e., `fill_value` is not `None`.
    * The dimension of coordinates `xi` matches the number of dimensions of the values.
      Also, each coordinate array in `xi` must have the same shape.

    See Also
    --------
    interpn : a convenience function which wraps `FastInterpolator`

    """

    def __init__(
        self,
        points,
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    ):
        super().__init__(
            points,
            values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

        self.uneven_dims = tuple(
            i for i, g in enumerate(self.grid) if not _check_even(g)
        )

    @classmethod
    def from_xarray(
        cls,
        data: xr.DataArray,
        method="linear",
        bounds_error: bool = False,
        fill_value: float | None = np.nan,
    ):
        """Construct an interpolator from a `xarray.DataArray`.

        Parameters
        ----------
        data
            The source `xarray.DataArray`.
        method
            The method of interpolation to perform.
        fill_value
            The value to use for points outside of the interpolation domain.

        """
        return cls(
            tuple(data[dim].values for dim in data.dims),
            data.values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

    def __call__(self, xi, method=None):
        is_linear: bool = method == "linear" or self.method == "linear"
        nd_supported: bool = self.values.ndim in (2, 3)
        no_extrap: bool = self.fill_value is not None

        if (len(self.uneven_dims) == 0) and is_linear and nd_supported and no_extrap:
            if isinstance(xi, np.ndarray):
                xi = tuple(xi.take(i, axis=-1) for i in range(xi.shape[-1]))

            xi_shapes = [x.shape for x in xi]
            if not all(s == xi_shapes[0] for s in xi_shapes):
                warnings.warn(
                    "Not all coordinate arrays have the same shape, "
                    "falling back to scipy.",
                    RuntimeWarning,
                    stacklevel=1,
                )
            elif len(xi) != self.values.ndim:
                warnings.warn(
                    f"Number of input dimensions ({len(xi)}) does not match "
                    "the input data dimensions, "
                    "falling back to scipy.",
                    RuntimeWarning,
                    stacklevel=1,
                )
            else:
                result = _get_interp_func(self.values.ndim)(
                    *self.grid,
                    self.values,
                    *(c.ravel() for c in xi),
                    fill_value=self.fill_value,
                ).reshape(xi[0].shape + self.values.shape[self.values.ndim :])
                return result

        if (len(self.uneven_dims) != 0) and is_linear:
            warnings.warn(
                f"Dimension(s) {self.uneven_dims} are not uniform, "
                "falling back to scipy.",
                RuntimeWarning,
                stacklevel=1,
            )
        return super().__call__(xi, method)


def interpn(
    points: Sequence[np.ndarray],
    values: np.ndarray,
    xi: Sequence[np.ndarray] | np.ndarray,
    method: str = "linear",
    bounds_error: bool = False,
    fill_value: float | None = np.nan,
):
    """Multidimensional interpolation on evenly spaced coordinates.

    This can be used as a drop-in replacement for `scipy.interpolate.interpn`.
    Performance optimization is applied in some special cases, documented in
    `FastInterpolator`.

    Parameters
    ----------
    points
        The points defining the regular grid in `n` dimensions. The points in
        each dimension (i.e. every element of the points tuple) must be strictly
        ascending.
    values
        The data on the regular grid in `n` dimensions.
    xi
        The coordinates to sample the gridded data at. In addition to the
        scipy-compatible syntax, a tuple of coordinates is also acceptable.
    method
        The method of interpolation to perform.
    fill_value
        The value to use for points outside of the interpolation domain.

    Returns
    -------
    values_x : numpy.ndarray
        Interpolated values at input coordinates.


    Note
    ----
    This optimized version of linear interpolation can be used with the `xarray`
    interpolation methods `xarray.Dataset.interp` and `xarray.DataArray.interp` by
    supplying ``method="linearfast"``. Note that the fallback to `scipy` will be silent
    except when a non-uniform dimension is found, when a warning will be issued.

    """
    interp = FastInterpolator(
        points, values, method=method, bounds_error=bounds_error, fill_value=fill_value
    )
    return interp(xi)


_get_interpolator_nd_original = xarray.core.missing._get_interpolator_nd
_get_interpolator_original = xarray.core.missing._get_interpolator


def _get_interpolator_fast(method, **kwargs):
    if method == "linearfast":
        method = "linear"
    return _get_interpolator_original(method, **kwargs)


def _get_interpolator_nd_fast(method, **kwargs):
    if method == "linearfast":
        return interpn, kwargs
    else:
        return _get_interpolator_nd_original(method, **kwargs)


xarray.core.missing._get_interpolator = _get_interpolator_fast
xarray.core.missing._get_interpolator_nd = _get_interpolator_nd_fast
