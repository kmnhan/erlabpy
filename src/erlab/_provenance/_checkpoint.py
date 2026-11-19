"""Own incoming data and verify the backing file of a lazy checkpoint."""

from __future__ import annotations

import copy
import mmap
import os
import stat
from dataclasses import dataclass

import numpy as np
import xarray as xr


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
        info = os.stat(path)
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
                f"Recorded input {self.path!r} is unavailable. Restore the file."
            ) from exc
        if self != observed:
            raise OSError(
                f"Recorded input {self.path!r} has changed. Reload its source."
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


def editable_data(data: xr.DataArray) -> xr.DataArray:
    """Give the caller independent values, coordinates, and nested metadata."""
    result = data.copy(deep=True)
    for variable in (result.variable, *(c.variable for c in result.coords.values())):
        if variable.dtype.hasobject and variable.chunks is not None:
            # Dask's final NumPy copy of a computed object array is shallow.
            array = variable.data
            variable.data = array.map_blocks(
                copy.deepcopy, dtype=array.dtype, meta=array._meta
            )
    return result


def _guard_file_variables(data: xr.DataArray, backing: FileVersion) -> xr.DataArray:
    import dask.array as da

    result = data.copy(deep=False)
    for variable in (result.variable, *(c.variable for c in result.coords.values())):
        if variable._in_memory and not isinstance(
            _buffer_owner(variable.data), mmap.mmap
        ):
            continue
        if variable.chunks is None:
            preferred = variable.encoding.get("preferred_chunks")
            chunks = (
                tuple(preferred.get(d, "auto") for d in variable.dims)
                if preferred
                else "auto"
            )
            variable.data = da.from_array(
                _FileVariable(variable, backing),
                chunks=chunks,
                name=False,
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
    captured = data.copy(deep=True, data=shared)
    if backing is not None:
        backing.verify()
    return captured
