"""Preserve xarray's chunk calculation when guarding an automatic Dask read."""

from __future__ import annotations

import typing

import dask.array as da
import numpy as np
from dask.base import tokenize
from xarray.namedarray.daskmanager import DaskManager

from erlab._provenance._checkpoint import FileVersion, _FileVariable, _guarded_graphs


class _VersionedDaskManager(DaskManager):
    def __init__(self, backing: FileVersion, token: str) -> None:
        super().__init__()
        self.backing = backing
        self.token = token
        self.occurrences: dict[str, int] = {}

    def from_array(
        self, data: typing.Any, chunks: typing.Any, **kwargs: typing.Any
    ) -> da.Array:
        # xarray visits the variables of this one file in its native order.
        # Older DataTree readers reuse a name for equal variables across groups.
        name = kwargs.get("name", "")
        index = self.occurrences.get(name, 0)
        self.occurrences[name] = index + 1
        kwargs["name"] = "read-file-version-" + tokenize(
            self.token, name, index, chunks
        )
        kwargs["meta"] = np.empty((0,) * data.ndim, dtype=data.dtype)
        array = da.from_array(
            _FileVariable(data, self.backing), chunks=chunks, **kwargs
        )
        _guarded_graphs[id(array.dask), array.name, self.backing] = array.dask
        return array
