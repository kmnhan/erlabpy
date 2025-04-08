"""Utilities for interpolation."""

__all__ = ["FastInterpolator", "interpn", "slice_along_path", "slice_along_vector"]

import itertools
import math
import typing
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import numba
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import xarray as xr
import xarray.core.missing

import erlab
from erlab.accessors.utils import either_dict_or_kwargs


class FastInterpolator(scipy.interpolate.RegularGridInterpolator):
    """Fast linear multidimensional interpolation on evenly spaced coordinates.

    This is an extension from `scipy.interpolate.RegularGridInterpolator` with vast
    performance improvements and integration with `xarray`.

    The input arguments are identical to `scipy.interpolate.RegularGridInterpolator`
    except `bounds_error`, which is set to `False` by default.

    Performance improvements are enabled for 1D, 2D and 3D linear interpolation on
    uniformly spaced coordinates with extrapolation disabled. Otherwise,
    `scipy.interpolate.RegularGridInterpolator` is called. See below for more
    information.

    Note
    ----
    Parallel acceleration is only applied when all of the following are true.

    * `method` is ``"linear"``.
    * Coordinates along all dimensions are evenly spaced.
    * Points are 1D, 2D or 3D.
    * Extrapolation is disabled, i.e., `fill_value` is not `None`.

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
        *,
        solver=None,
        solver_args=None,
    ) -> None:
        super().__init__(
            points,
            values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
            solver=solver,
            solver_args=solver_args,
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

    def __call__(self, xi, method: str | None = None):
        ndim: int = len(self.grid)
        is_linear: bool = method == "linear" or self.method == "linear"
        nd_supported: bool = ndim in (1, 2, 3)
        no_extrap: bool = self.fill_value is not None

        if (len(self.uneven_dims) == 0) and is_linear and nd_supported and no_extrap:
            if isinstance(xi, np.ndarray):
                if xi.ndim == 1:
                    xi_tuple = (xi,)
                else:
                    xi_tuple = tuple(xi.take(i, axis=-1) for i in range(xi.shape[-1]))
            else:
                xi_tuple = tuple(xi)

            xi_shapes = [x.shape for x in xi_tuple]
            if not all(s == xi_shapes[0] for s in xi_shapes):
                erlab.utils.misc.emit_user_level_warning(
                    "Not all coordinate arrays have the same shape, "
                    "falling back to scipy.",
                    RuntimeWarning,
                )

            elif ndim == len(xi_tuple):
                interp_func = _get_interp_func(ndim)
                return interp_func(
                    *self.grid, self.values, *xi_tuple, fill_value=self.fill_value
                )

        if (len(self.uneven_dims) != 0) and is_linear:
            erlab.utils.misc.emit_user_level_warning(
                f"Dimension(s) {self.uneven_dims} are not uniform, "
                "falling back to scipy.",
                RuntimeWarning,
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
def _calc_interp1(values, v0):
    i0 = math.floor(v0)
    n0 = values.shape[0]
    j0 = min(i0 + 1, n0 - 1)
    return _do_interp1(v0 - i0, values[i0], values[j0])


@numba.njit(nogil=True, cache=True)
def _calc_interp2(values, v0, v1):
    i0, i1 = math.floor(v0), math.floor(v1)
    n0, n1 = values.shape[:2]
    j0, j1 = min(i0 + 1, n0 - 1), min(i1 + 1, n1 - 1)
    return _do_interp2(
        v0 - i0, v1 - i1, values[i0, i1], values[j0, i1], values[i0, j1], values[j0, j1]
    )


@numba.njit(nogil=True, cache=True)
def _calc_interp3(values, v0, v1, v2):
    i0, i1, i2 = math.floor(v0), math.floor(v1), math.floor(v2)
    n0, n1, n2 = values.shape[:3]
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


@numba.njit(nogil=True, cache=True)
def _val2ind(val, coord):
    if val > coord[-1] or val < coord[0]:
        return np.nan
    return np.divide(val - coord[0], coord[1] - coord[0])


@numba.njit(nogil=True, parallel=True, cache=True)
def _interp1(x, values, xc, fill_value=np.nan):
    out_shape = xc.shape + values.shape[1:]
    xc_flat = xc.ravel()
    n = len(xc_flat)

    arr_new = np.empty((n,) + values.shape[1:], values.dtype)

    for m in numba.prange(n):
        v0 = _val2ind(xc_flat[m], x)

        if np.isnan(v0):
            arr_new[m] = fill_value
        else:
            arr_new[m] = _calc_interp1(values, v0)

    return arr_new.reshape(out_shape)


@numba.njit(nogil=True, parallel=True, cache=True)
def _interp2(x, y, values, xc, yc, fill_value=np.nan):
    out_shape = xc.shape + values.shape[2:]
    xc_flat, yc_flat = xc.ravel(), yc.ravel()
    n = len(xc_flat)

    arr_new = np.empty((n,) + values.shape[2:], values.dtype)

    for m in numba.prange(n):
        v0, v1 = _val2ind(xc_flat[m], x), _val2ind(yc_flat[m], y)

        if np.isnan(v0) or np.isnan(v1):
            arr_new[m] = fill_value
        else:
            arr_new[m] = _calc_interp2(values, v0, v1)

    return arr_new.reshape(out_shape)


@numba.njit(nogil=True, parallel=True, cache=True)
def _interp3(x, y, z, values, xc, yc, zc, fill_value=np.nan):
    out_shape = xc.shape + values.shape[3:]
    xc_flat, yc_flat, zc_flat = xc.ravel(), yc.ravel(), zc.ravel()
    n = len(xc_flat)

    arr_new = np.empty((n,) + values.shape[3:], values.dtype)

    for m in numba.prange(n):
        v0, v1, v2 = (
            _val2ind(xc_flat[m], x),
            _val2ind(yc_flat[m], y),
            _val2ind(zc_flat[m], z),
        )

        if np.isnan(v0) or np.isnan(v1) or np.isnan(v2):
            arr_new[m] = fill_value
        else:
            arr_new[m] = _calc_interp3(values, v0, v1, v2)

    return arr_new.reshape(out_shape)


def _get_interp_func(ndim: int) -> Callable:
    match ndim:
        case 3:
            return _interp3
        case 2:
            return _interp2
        case 1:
            return _interp1
        case _:
            raise ValueError("Fast interpolation only supported for 2D or 3D")


def _check_even(arr) -> bool:
    dif = np.diff(arr)
    if dif.size == 0:
        return False
    return np.allclose(dif, dif[0])


def slice_along_path(
    darr: xr.DataArray,
    vertices: Mapping[Hashable, Sequence[float]],
    step_size: float | None = None,
    dim_name: str = "path",
    interp_kwargs: dict | None = None,
    **vertices_kwargs,
) -> xr.DataArray:
    """Interpolate a DataArray along a path defined by a sequence of vertices.

    Parameters
    ----------
    darr
        The data array to interpolate.
    vertices
        Dictionary specifying the vertices of the path along which to interpolate the
        DataArray. The keys of the dictionary should correspond to the dimensions of the
        DataArray along which to interpolate.
    step_size
        The step size to use for the interpolation. This determines the number of points
        along the path at which the data array will be interpolated. If None, the step
        size is determined automatically as the smallest step size of the coordinates
        along the dimensions of the vertices if all coordinates are evenly spaced. If
        there exists a dimension where the coordinates are not evenly spaced,
        `step_size` must be specified.
    dim_name
        The name of the new dimension that corresponds to the distance along the
        interpolated path. Default is "path".
    interp_kwargs
        Additional keyword arguments passed to `xarray.DataArray.interp`.
    **vertices_kwargs
        The keyword arguments form of `vertices`. One of `vertices` or `vertices_kwargs`
        must be provided.

    Returns
    -------
    interpolated : DataArray
        The interpolated data along the path.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from erlab.analysis.interpolate import slice_along_path
    >>> x = np.linspace(0, 10, 11)
    >>> y = np.linspace(0, 10, 11)
    >>> z = np.linspace(0, 10, 11)
    >>> data = np.random.rand(11, 11, 11)
    >>> darr = xr.DataArray(data, coords={"x": x, "y": y, "z": z}, dims=["x", "y", "z"])
    >>> vertices = {"x": [0, 5, 10], "y": [0, 5, 10], "z": [0, 5, 10]}
    >>> interp = slice_along_path(darr, vertices)

    See Also
    --------
    xarray.DataArray.interp
        The method used to perform the interpolation.
    :func:`erlab.analysis.interpolate.slice_along_vector`
        Slice along a vector defined by a center point, direction vector, and lengths.

    """
    interp_kwargs = interp_kwargs or {}
    vertices = dict(
        either_dict_or_kwargs(vertices, vertices_kwargs, "slice_along_path")
    )

    for dim, vert in vertices.items():
        if dim not in darr.dims:
            raise ValueError(f"Dimension {dim} not found in data array")

        # Convert to numpy array
        vert_arr = np.asarray(vert)
        if vert_arr.ndim != 1:
            raise ValueError("Each vertex must be 1-dimensional")
        vertices[dim] = vert_arr

    if not all(len(p) == len(next(iter(vertices.values()))) for p in vertices.values()):
        raise ValueError("Vertices must have the same length along all dimensions")

    if step_size is None:
        step_size = np.inf
        for dim in vertices:
            dif = np.diff(darr[dim].values)
            if np.allclose(dif, dif[0], equal_nan=True):
                step_size = min(step_size, np.abs(dif[0]))

        if not np.isfinite(typing.cast("float", step_size)):
            raise ValueError("Could not determine step size automatically")
    else:
        if not np.isfinite(step_size):
            raise ValueError("Step size must be finite")
        if step_size <= 0:
            raise ValueError("Step size must be positive")

    points: npt.NDArray[np.floating] = np.array(list(vertices.values())).T

    # Calculate number of points for each segment
    num_points = [
        round(np.linalg.norm(p2 - p1) / step_size) - 1  # type: ignore[call-overload,operator]
        for p1, p2 in itertools.pairwise(points)
    ]
    # Generate points for each segment
    segments = [
        np.linspace(points[i], points[i + 1], num, axis=-1, endpoint=False)
        for i, num in enumerate(num_points[:-1])
    ]
    segments.append(
        np.linspace(points[-2], points[-1], num_points[-1] + 1, axis=-1, endpoint=True)
    )

    # Concatenate the points
    points_arr = np.concatenate(segments, axis=-1)
    points_arr = np.atleast_2d(points_arr)  # Ensure 2D for 1D paths

    # Distance between each pair of consecutive points
    distances = np.linalg.norm(np.diff(points_arr, axis=-1), axis=0)

    # Cumulative sum of the distances
    path_coord = np.concatenate(([0], np.cumsum(distances)))

    interp_coords = [
        xr.DataArray(p, dims=dim_name, coords={dim_name: path_coord})
        for p in points_arr
    ]
    interp_kwargs.setdefault("method", "linearfast")

    return darr.interp(
        dict(zip(vertices.keys(), interp_coords, strict=False)), **interp_kwargs
    )


def slice_along_vector(
    data: xr.DataArray,
    center: dict[str, float],
    direction: dict[str, float],
    stretch: float | tuple[float, float],
    **kwargs,
):
    """Slice along a vector defined by the center point, direction vector, and lengths.

    Parameters
    ----------
    data
        The data to be sliced.
    center
        The center point of the vector given as a dictionary. The keys should correspond
        to dimensions in `data`.
    direction
        The direction vector given as a dictionary with the same keys as `center`. The
        direction is normalized before slicing.
    stretch
        The length of the vector in both directions in data units. If a single value is
        given, the vector is stretched in both directions by the same amount. If a tuple
        is given, the vector is stretched by different amounts in the negative and
        positive directions.
    **kwargs
        Additional keyword arguments to be passed to `slice_along_path`.

    Returns
    -------
    interpolated : DataArray
        The interpolated data along the vector.

    Examples
    --------
    ::

        import erlab.analysis as era
        import numpy as np
        from erlab.io.exampledata import generate_data

        # Generate some example data: a 3D kx-ky-eV ARPES map.
        data = generate_data()

        # Slice along a vector centered at (kx, ky) = (-0.5, -0.3).
        # The direction of the vector is defined by (kx, ky) = (sqrt(3), 1.0).
        # The slice starts from -0.1 inverse Å and ends at 0.2 inverse Å.

        era.interpolate.slice_along_vector(
            data,
            center=dict(kx=-0.5, ky=-0.3),
            direction=dict(kx=np.sqrt(3), ky=1.0),
            stretch=(0.1, 0.3),
        )

    See Also
    --------
    :func:`erlab.analysis.interpolate.slice_along_path`

    """
    direction: dict[str, float] = {
        k: float(v / np.linalg.norm(list(direction.values())))
        for k, v in direction.items()
    }
    if isinstance(stretch, Iterable):
        lm, lp = stretch
    else:
        lm = lp = stretch

    vert: Mapping[Hashable, Sequence[float]] = {
        k: (center[k] - lm * direction[k], center[k] + lp * direction[k])
        for k in center
    }
    return slice_along_path(data, vertices=vert, **kwargs)


# Monkey patch xarray to make fast interpolator available

_get_interpolator_nd_original = xarray.core.missing._get_interpolator_nd
_get_interpolator_original = xarray.core.missing._get_interpolator


def _get_interpolator_fast(method, **kwargs):
    if method == "linearfast":
        method = "linear"
        # For 1D interpolation, always fall back to scipy
    return _get_interpolator_original(method, **kwargs)


def _get_interpolator_nd_fast(method, **kwargs):
    if method == "linearfast":
        # For ND interpolation, return the fast interpolator
        return interpn, kwargs
    return _get_interpolator_nd_original(method, **kwargs)


xarray.core.missing._get_interpolator = _get_interpolator_fast  # type: ignore[assignment]
xarray.core.missing._get_interpolator_nd = _get_interpolator_nd_fast
