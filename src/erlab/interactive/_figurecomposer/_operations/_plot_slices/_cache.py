"""Small bounded cache for prepared Plot Slices selections."""

from __future__ import annotations

import collections
import typing
from collections.abc import Hashable, Iterator, MutableMapping

import xarray as xr

_MAX_ENTRIES = 8
_MAX_BYTES = 256 * 1024**2


class _PlotSlicesSelectionCache(MutableMapping[Hashable, tuple[xr.DataArray, ...]]):
    """Least-recently-used Plot Slices selections with a fixed memory budget."""

    def __init__(self) -> None:
        self._values: collections.OrderedDict[
            Hashable, tuple[tuple[xr.DataArray, ...], int]
        ] = collections.OrderedDict()
        self._nbytes = 0

    def __getitem__(self, key: Hashable) -> tuple[xr.DataArray, ...]:
        value, nbytes = self._values.pop(key)
        self._values[key] = value, nbytes
        return value

    def __setitem__(self, key: Hashable, value: tuple[xr.DataArray, ...]) -> None:
        nbytes = sum(int(data.nbytes) for data in value)
        previous = self._values.pop(key, None)
        if previous is not None:
            self._nbytes -= previous[1]
        if nbytes > _MAX_BYTES:
            return

        import dask

        if any(dask.base.is_dask_collection(data) for data in value):
            value = typing.cast(
                "tuple[xr.DataArray, ...]",
                dask.persist(*value),
            )
        self._values[key] = value, nbytes
        self._nbytes += nbytes
        while len(self._values) > _MAX_ENTRIES or self._nbytes > _MAX_BYTES:
            _old_key, (_old_value, old_nbytes) = self._values.popitem(last=False)
            self._nbytes -= old_nbytes

    def __delitem__(self, key: Hashable) -> None:
        _value, nbytes = self._values.pop(key)
        self._nbytes -= nbytes

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def clear(self) -> None:
        self._values.clear()
        self._nbytes = 0
