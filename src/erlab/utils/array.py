"""Utility functions for working with numpy and xarray."""

import functools
from collections.abc import Callable, Hashable, Iterable

import numpy as np
import numpy.typing as npt
import xarray as xr


def is_uniform_spaced(arr: npt.NDArray, **kwargs) -> bool:
    """Check if the given array is uniformly spaced.

    Parameters
    ----------
    arr : array-like
        The input array.
    **kwargs
        Additional keyword arguments passed to `numpy.isclose`.

    Returns
    -------
    bool
        `True` if the array is uniformly spaced, `False` otherwise.

    Examples
    --------
    >>> is_uniform_spaced([1, 2, 3, 4])
    True
    >>> is_uniform_spaced([1, 2, 3, 5])
    False
    """
    arr = np.atleast_1d(np.array(arr, dtype=float))
    dif = np.diff(arr)
    if dif.size == 0:
        return True
    return np.allclose(dif, dif[0], **kwargs)


def is_monotonic(arr: npt.NDArray) -> np.bool_:
    """Check if an array is monotonic.

    Parameters
    ----------
    arr : array-like
        The input array.

    Returns
    -------
    bool
        `True` if the array is monotonic (either non-decreasing or non-increasing),
        `False` otherwise.
    """
    arr = np.atleast_1d(np.array(arr, dtype=float))
    dif = np.diff(arr)
    return np.all(dif >= 0) or np.all(dif <= 0)


def uniform_dims(darr: xr.DataArray, **kwargs) -> set[Hashable]:
    """Return the set of dimensions that are uniformly spaced.

    Parameters
    ----------
    darr
        The input xarray DataArray.
    **kwargs
        Additional keyword arguments passed to `is_uniform_spaced`.

    Returns
    -------
    dims
        A set of dimensions that are uniformly spaced.
    """
    return {d for d in darr.dims if is_uniform_spaced(darr[d].values, **kwargs)}


def is_dims_uniform(
    darr: xr.DataArray, dims: Iterable[Hashable] | None = None, **kwargs
) -> bool:
    """
    Check if the given dimensions of a DataArray have uniform spacing.

    Parameters
    ----------
    darr
        The DataArray to check.
    dims
        The dimensions to check. If `None`, all dimensions of the DataArray will be
        checked.
    **kwargs
        Additional keyword arguments to be passed to `is_uniform_spaced`.

    Returns
    -------
    bool
        `True` if all dimensions have uniform spacing, `False` otherwise.
    """
    if dims is None:
        dims = darr.dims

    return all(is_uniform_spaced(darr[dim].values, **kwargs) for dim in dims)


def check_arg_2d_darr(func: Callable | None = None):
    """Decorate a function to check if the first argument is a 2D DataArray."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if not isinstance(args[0], xr.DataArray) or args[0].ndim != 2:
                raise ValueError("Input must be a 2-dimensional xarray.DataArray")
            return func(*args, **kwargs)

        return _wrapper

    if func is not None:
        return _decorator(func)
    return _decorator


def check_arg_uniform_dims(func: Callable | None = None):
    """Decorate a function to check if all dims in the first argument are uniform."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if not isinstance(args[0], xr.DataArray) or not is_dims_uniform(args[0]):
                raise ValueError(
                    "Coordinates for all dimensions must be uniformly spaced"
                )
            return func(*args, **kwargs)

        return _wrapper

    if func is not None:
        return _decorator(func)
    return _decorator


def check_arg_has_no_nans(func: Callable | None = None):
    """Decorate a function to check if the first argument has no NaNs."""

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if np.isnan(args[0]).any():
                raise ValueError("Input must not contain any NaN values")
            return func(*args, **kwargs)

        return _wrapper

    if func is not None:
        return _decorator(func)
    return _decorator


def trim_na(darr: xr.DataArray, dims: Iterable[Hashable] | None = None) -> xr.DataArray:
    """Drop all-NaN coordinates from the edges of a DataArray.

    Parameters
    ----------
    darr
        The DataArray to trim.
    dims
        The dimensions along which to trim. If not provided, the data will be trimmed
        along all dimensions.

    Returns
    -------
    darr
        The trimmed DataArray.
    """
    if dims is None:
        dims = darr.dims
    for dim in dims:
        darr = _trim_na_edges(darr, dim)
    return darr


def _trim_na_edges(darr: xr.DataArray, dim: Hashable) -> xr.DataArray:
    return _trim_na_trailing_edge(_trim_na_leading_edge(darr, dim), dim)


def _trim_na_leading_edge(darr: xr.DataArray, dim: Hashable) -> xr.DataArray:
    if darr[dim].size > 0 and np.isnan(darr.isel({dim: 0}).values).all():
        darr = darr.isel({dim: slice(1, None)})
        return _trim_na_leading_edge(darr, dim)
    return darr


def _trim_na_trailing_edge(darr: xr.DataArray, dim: Hashable) -> xr.DataArray:
    if darr[dim].size > 0 and np.isnan(darr.isel({dim: -1}).values).all():
        darr = darr.isel({dim: slice(None, -1)})
        return _trim_na_trailing_edge(darr, dim)
    return darr
