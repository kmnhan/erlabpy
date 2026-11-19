"""Own incoming data and verify the backing file of a lazy checkpoint."""

from __future__ import annotations

import copy
import mmap
import os
import stat
import typing
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.indexing import PandasIndexingAdapter
from xarray.indexes import PandasIndex

if typing.TYPE_CHECKING:
    from collections.abc import Hashable


@dataclass(frozen=True)
class FileVersion:
    """Identity of the regular file that backs all deferred reads of an input."""

    path: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def capture(cls, path: str | os.PathLike[str]) -> FileVersion:
        """Capture the version before opening the array or scheduling any reads."""
        path = os.path.abspath(path)
        return cls.from_stat(path, os.stat(path))

    @classmethod
    def from_stat(cls, path: str, info: os.stat_result) -> FileVersion:
        """Capture the version at an absolute path from its open file handle."""
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"A versioned provenance input needs a regular file: {path}"
            )
        return cls(
            path,
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def verify(self) -> None:
        try:
            observed = FileVersion.capture(self.path)
        except (OSError, ValueError) as exc:
            raise OSError(
                f"Recorded input {self.path!r} is unavailable. Restore the file or "
                "use a saved checkpoint of the original data."
            ) from exc
        if self != observed:
            raise OSError(
                f"Recorded input {self.path!r} has changed. Use a saved checkpoint of "
                "the original data, or reload its source to use the current file."
            )


def _buffer_owner(array: object) -> object:
    while isinstance(array, np.ndarray | memoryview):
        if isinstance(array, memoryview):
            array = array.obj
        elif array.base is not None:
            array = array.base
        else:
            break
    return array


def _checked_chunk(block: np.ndarray, *, backing: FileVersion) -> np.ndarray:
    # Detach a mapped view before validating the bytes we will return.
    if isinstance(_buffer_owner(block), mmap.mmap):
        block = block.copy()
    backing.verify()
    return block


class _FileVariable:
    """Expose an unchunked xarray backend to Dask without loading the variable."""

    def __init__(self, variable: xr.Variable, backing: FileVersion) -> None:
        self.variable = variable.copy(deep=False)
        self.backing = backing
        self.shape = variable.shape
        self.dtype = variable.dtype
        self.ndim = variable.ndim

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        self.backing.verify()
        return _checked_chunk(self.variable[key].values, backing=self.backing)


def _owned_categories(values: pd.Categorical) -> pd.Categorical:
    return pd.Categorical.from_codes(
        values.codes.copy(), _owned_index(values.categories), ordered=values.ordered
    )


def _owned_index(index: pd.Index) -> pd.Index:
    # A deep pandas Index copy still shares Python objects in the index.
    if isinstance(index, pd.MultiIndex):
        return index.set_levels(
            [_owned_index(level) for level in index.levels], verify_integrity=False
        )
    if isinstance(index, pd.CategoricalIndex):
        return pd.CategoricalIndex(_owned_categories(index.array), name=index.name)
    if index.dtype != np.dtype(object):
        return index
    values = copy.deepcopy(index.to_numpy())
    if type(index) is pd.Index:
        return pd.Index(values, dtype=index.dtype, name=index.name, tupleize_cols=False)
    return type(index)(values, name=index.name)


def _own_index_data(
    data: xr.DataArray, *, copy_object_chunks: bool = False
) -> xr.DataArray:
    """Detach pandas index members and xarray's linked coordinate metadata."""
    variable = data.variable
    coordinates = data.coords.variables
    # Dask blocks with structured dtypes may also contain Python objects.
    has_objects = variable.dtype.kind in {"O", "V"}
    for coord in coordinates.values():
        has_objects |= coord.dtype.kind in {"O", "V"}
        if isinstance(coord, xr.IndexVariable):
            # Xarray recreates coordinate variables from their index and reuses
            # nested metadata even when the DataArray copy requests deep=True.
            if coord.attrs:
                coord.attrs = copy.deepcopy(coord.attrs)
            if coord.encoding:
                coord.encoding = copy.deepcopy(coord.encoding)
    if not has_objects:
        return data
    for item in (variable, *coordinates.values()):
        if copy_object_chunks and item.chunks is not None and item.dtype.hasobject:
            # Dask's final NumPy copy of a computed object array is shallow.
            array = item.data
            item.data = array.map_blocks(
                copy.deepcopy, dtype=array.dtype, meta=array._meta
            )
        if item.dtype.kind != "O" or isinstance(item, xr.IndexVariable):
            continue
        if isinstance(item.dtype, pd.CategoricalDtype):
            item.data = _owned_categories(item.data)
        elif isinstance(item._data, PandasIndexingAdapter):
            # drop_indexes can leave the pandas backing on an ordinary Variable.
            item.data = copy.deepcopy(item.data)
    if isinstance(variable, xr.IndexVariable) and variable.dtype.kind == "O":
        variable = variable.copy(deep=False, data=_owned_index(variable.to_index()))
    if not any(
        isinstance(coord, xr.IndexVariable) and coord.dtype.kind == "O"
        for coord in coordinates.values()
    ):
        return data if variable is data.variable else data._replace(variable)
    coords = dict(coordinates)
    indexes = dict(data.xindexes)
    for name, coord in coordinates.items():
        if (
            name not in indexes
            and isinstance(coord, xr.IndexVariable)
            and coord.dtype.kind == "O"
        ):
            coords[name] = coord.copy(deep=False, data=_owned_index(coord.to_index()))
    for index, variables in data.xindexes.group_by_index():
        if isinstance(index, PandasIndex) and any(
            coord.dtype.kind == "O" for coord in variables.values()
        ):
            owned = index._replace(_owned_index(index.to_pandas_index()))
            coords.update(owned.create_variables(variables))
            indexes.update(dict.fromkeys(variables, owned))
    return data._replace(variable, coords, indexes=indexes)


def editable_data(data: xr.DataArray) -> xr.DataArray:
    """Give the caller independent values, coordinates, and nested metadata."""
    return _own_index_data(data.copy(deep=True), copy_object_chunks=True)


def _guard_file_variables(
    data: xr.DataArray, backing: FileVersion, *, storage_key: str | None = None
) -> xr.DataArray:
    result = data.copy(deep=False)
    variables: list[tuple[tuple[Hashable, ...], xr.Variable]] = [((), result.variable)]
    variables.extend(((key,), coord) for key, coord in result.coords.variables.items())
    for identity, variable in variables:
        if variable._in_memory and not isinstance(
            _buffer_owner(variable.data), mmap.mmap
        ):
            continue
        if variable.chunks is None:
            import dask.array as da

            preferred = variable.encoding.get("preferred_chunks") or {}
            name: str | bool = False
            if storage_key is not None:
                from dask.base import tokenize

                # xarray chunks={} uses the file layout, or the full extent for
                # a contiguous variable. Dask's getter can fuse a narrower slice.
                chunks = tuple(
                    preferred.get(dim, size)
                    for dim, size in zip(variable.dims, variable.shape, strict=True)
                )
                name = "read-file-version-" + tokenize(storage_key, identity)
            else:
                chunks = (
                    tuple(preferred.get(d, "auto") for d in variable.dims)
                    if preferred
                    else "auto"
                )
            variable.data = da.from_array(
                _FileVariable(variable, backing),
                chunks=chunks,
                name=name,
                meta=np.empty((0,) * variable.ndim, dtype=variable.dtype),
            )
        else:
            array = variable.data
            variable.data = array.map_blocks(
                _checked_chunk, backing=backing, dtype=array.dtype, meta=array._meta
            )
    return result


def checkpoint_data(
    data: xr.DataArray, *, backing: FileVersion | None = None
) -> xr.DataArray:
    """Detach external buffers. Keep a declared, versioned file lazy."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("Revision data must be a DataArray")
    if backing is not None:
        backing.verify()
        data = _guard_file_variables(data, backing)
    elif not data.variable._in_memory or any(
        not coord.variable._in_memory for coord in data.coords.values()
    ):
        # A delayed task can close over a mutable buffer. Evaluate all variables
        # together so data and coordinates that share a task only read it once.
        data = data.compute()
    # The computation can still return mutable objects. Copy them and all
    # metadata. Only a numeric buffer backed by immutable bytes can be shared.
    values = data.data
    shared = values if isinstance(_buffer_owner(values), bytes) else None
    captured = _own_index_data(data.copy(deep=True, data=shared))
    if backing is not None:
        backing.verify()
    return captured
