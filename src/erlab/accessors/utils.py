"""Utility functions for creating accessors."""

from collections.abc import Hashable, Mapping
from typing import Any, TypeGuard, TypeVar, cast

import xarray as xr

T = TypeVar("T")

# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets, from xarray.core.dataarray
_THIS_ARRAY: str = "<this-array>"


class ERLabDataArrayAccessor:
    """Base class for accessors."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj


class ERLabDatasetAccessor:
    """Base class for accessors."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj


def is_dict_like(value: Any) -> TypeGuard[Mapping[Any, Any]]:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def either_dict_or_kwargs(
    pos_kwargs: Mapping[Any, T] | None,
    kw_kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    if pos_kwargs is None or pos_kwargs == {}:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return cast(Mapping[Hashable, T], kw_kwargs)

    if not is_dict_like(pos_kwargs):
        raise ValueError(f"the first argument to .{func_name} must be a dictionary")
    if kw_kwargs:
        raise ValueError(
            f"cannot specify both keyword and positional arguments to .{func_name}"
        )
    return pos_kwargs
