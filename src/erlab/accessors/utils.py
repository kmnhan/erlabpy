"""Utility functions for creating accessors."""

import typing
from collections.abc import Hashable, Mapping

import xarray as xr


class ERLabDataArrayAccessor:
    """Base class for DataArray accessors."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj


class ERLabDatasetAccessor:
    """Base class for Dataset accessors."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj


def is_dict_like(
    value: typing.Any,
) -> typing.TypeGuard[Mapping[typing.Any, typing.Any]]:
    """Check if the given value is dict-like."""
    # From xarray.namedarray.utils.is_dict_like
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


_T = typing.TypeVar("_T")


def either_dict_or_kwargs(
    pos_kwargs: Mapping[typing.Any, _T] | None,
    kw_kwargs: Mapping[str, _T],
    func_name: str,
) -> Mapping[Hashable, _T]:
    """Return the positional or keyword arguments as a dictionary."""
    # From xarray.namedarray.utils.either_dict_or_kwargs
    if pos_kwargs is None or pos_kwargs == {}:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return typing.cast(Mapping[Hashable, _T], kw_kwargs)

    if not is_dict_like(pos_kwargs):
        raise ValueError(f"the first argument to .{func_name} must be a dictionary")
    if kw_kwargs:
        raise ValueError(
            f"cannot specify both keyword and positional arguments to .{func_name}"
        )
    return pos_kwargs
