"""Utility functions for working with numpy and xarray."""

__all__ = ["is_dims_uniform", "is_monotonic", "is_uniform_spaced", "uniform_dims"]

from collections.abc import Hashable, Iterable

import numpy as np
import numpy.typing as npt
import xarray as xr


def is_uniform_spaced(arr: npt.NDArray, **kwargs) -> bool:
    """Check if the given array is uniformly spaced.

    Parameters
    ----------
    arr
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
    arr
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
    dims : set of Hashable
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
        checked. Defaults to `None`.
    **kwargs
        Additional keyword arguments to be passed to `is_uniform_spaced`.

    Returns
    -------
    bool
        `True` if all dimensions have uniform spacing, `False` otherwise.
    """
    if dims is None:
        dims = darr.dims

    for dim in dims:
        if not is_uniform_spaced(darr[dim].values, **kwargs):
            return False
    return True
