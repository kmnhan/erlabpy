"""xarray file-handle helpers for manager workspace IO."""

from __future__ import annotations

import os
import pathlib
import threading
import typing

import h5netcdf
import hdf5plugin
import numpy as np
import xarray as xr
from xarray.backends import CachingFileManager, H5NetCDFStore

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

    from erlab.interactive._options.schema import WorkspaceCompressionMode

_WORKSPACE_FILE_LOCKS: dict[str, threading.RLock] = {}
_WORKSPACE_COMPRESSION_MIN_BYTES = 1 << 20  # 1 MiB


def _normalized_file_path(path: object) -> str | None:
    """Return an absolute normalized path string for path-like values."""
    if not isinstance(path, (str, bytes, os.PathLike)):
        return None
    try:
        path_str = os.fsdecode(path)
    except TypeError:
        return None
    if not path_str:
        return None
    try:
        return str(pathlib.Path(path_str).resolve())
    except OSError:
        return str(pathlib.Path(path_str).absolute())


def _workspace_file_lock(path: str | os.PathLike[str]) -> threading.RLock:
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    lock = _WORKSPACE_FILE_LOCKS.get(target)
    if lock is None:
        lock = threading.RLock()
        _WORKSPACE_FILE_LOCKS[target] = lock
    return lock


def _workspace_file_identity(path: str | os.PathLike[str]) -> tuple[str, int, int, int]:
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    try:
        stat_result = os.stat(target)
    except OSError:
        return target, 0, 0, 0
    return target, stat_result.st_dev, stat_result.st_ino, stat_result.st_mtime_ns


def _xarray_source_path(value: object) -> str | None:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    return _normalized_file_path(value)


def dataarray_source_paths(data_array: xr.DataArray) -> tuple[str, ...]:
    """Return normalized file sources referenced by a DataArray and its coords."""
    paths: list[str] = []

    def _append_source(value: object) -> None:
        source = _xarray_source_path(value)
        if source is not None and source not in paths:
            paths.append(source)

    _append_source(data_array.encoding.get("source"))
    for coord in data_array.coords.values():
        _append_source(coord.encoding.get("source"))
    return tuple(paths)


def dataarray_is_numpy_backed(data_array: xr.DataArray) -> bool:
    """Return True when a DataArray is already backed by an in-memory ndarray."""
    return isinstance(data_array.variable._data, (np.ndarray, np.generic))


def dataarray_is_file_backed(data_array: xr.DataArray) -> bool:
    """Return True for non-dask xarray arrays still backed by a file manager."""
    return (
        data_array.chunks is None
        and not dataarray_is_numpy_backed(data_array)
        and bool(dataarray_source_paths(data_array))
    )


def ensure_workspace_hdf5_filters_registered() -> None:
    """Register HDF5 filters needed by compressed workspace files."""
    hdf5plugin.register(force=False)


def workspace_compression_enabled() -> bool:
    return workspace_compression_mode() != "none"


def workspace_compression_mode() -> WorkspaceCompressionMode:
    import erlab

    return erlab.interactive.options.model.io.workspace.compression


def _workspace_blosc2_encoding(
    compression_mode: WorkspaceCompressionMode,
) -> dict[str, typing.Any]:
    if compression_mode == "none":
        return {}

    ensure_workspace_hdf5_filters_registered()
    match compression_mode:
        case "blosclz3":
            cname = "blosclz"
            clevel = 3
        case "zstd1":
            cname = "zstd"
            clevel = 1
        case _:
            raise ValueError(f"Unknown workspace compression mode: {compression_mode}")

    return dict(
        hdf5plugin.Blosc2(
            cname=cname,
            clevel=clevel,
            filters=hdf5plugin.Blosc2.SHUFFLE,
        )
    )


def _resolve_workspace_compression_mode(
    *,
    compression_mode: WorkspaceCompressionMode | None,
    compress: bool | None,
) -> WorkspaceCompressionMode:
    if compression_mode is not None:
        return compression_mode
    if compress is False:
        return "none"
    if compress is True:
        return "zstd1"
    return workspace_compression_mode()


def _should_compress_workspace_variable(
    variable: xr.Variable, *, min_bytes: int
) -> bool:
    if variable.dtype.kind not in "iufc":
        return False
    return int(variable.nbytes) >= min_bytes


def _workspace_chunksizes_for_dataarray(
    data_array: xr.DataArray,
) -> tuple[int, ...] | None:
    """Return a valid fixed HDF5 chunk shape for dask-backed workspace data."""
    chunks = data_array.chunks
    if chunks is None or data_array.ndim == 0:
        return None

    chunksizes: list[int] = []
    for size, dim_chunks in zip(data_array.shape, chunks, strict=True):
        if size <= 0 or len(dim_chunks) == 0:
            return None
        first = int(dim_chunks[0])
        if first <= 0:
            return None
        chunksizes.append(min(first, int(size)))
    return tuple(chunksizes)


def workspace_dataset_encoding(
    ds: xr.Dataset,
    *,
    min_bytes: int = _WORKSPACE_COMPRESSION_MIN_BYTES,
    compress: bool | None = None,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> dict[Hashable, dict[str, typing.Any]]:
    """Return h5netcdf encodings for workspace data variables."""
    compression_mode = _resolve_workspace_compression_mode(
        compression_mode=compression_mode, compress=compress
    )
    compression_encoding = _workspace_blosc2_encoding(compression_mode)

    encoding: dict[Hashable, dict[str, typing.Any]] = {}
    for name, data_array in ds.data_vars.items():
        var_encoding: dict[str, typing.Any] = {}
        chunksizes = _workspace_chunksizes_for_dataarray(data_array)
        if chunksizes is not None:
            var_encoding["chunksizes"] = chunksizes
        if compression_encoding and _should_compress_workspace_variable(
            data_array.variable, min_bytes=min_bytes
        ):
            var_encoding.update(compression_encoding)
        if var_encoding:
            encoding[name] = var_encoding
    return encoding


def workspace_datatree_encoding(
    tree: xr.DataTree,
    *,
    min_bytes: int = _WORKSPACE_COMPRESSION_MIN_BYTES,
    compress: bool | None = None,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> dict[str, dict[Hashable, dict[str, typing.Any]]]:
    """Return nested h5netcdf encodings for workspace payloads."""
    compression_mode = _resolve_workspace_compression_mode(
        compression_mode=compression_mode, compress=compress
    )

    encoding: dict[str, dict[Hashable, dict[str, typing.Any]]] = {}
    for node in tree.subtree:
        node_encoding = workspace_dataset_encoding(
            node.to_dataset(inherit=False),
            min_bytes=min_bytes,
            compression_mode=compression_mode,
        )
        if node_encoding:
            encoding[node.path] = node_encoding
    return encoding


class WorkspaceFileManager(CachingFileManager):
    """xarray file manager for manager-owned workspace readers."""

    def __init__(self, path: str | os.PathLike[str]) -> None:

        ensure_workspace_hdf5_filters_registered()
        target = _normalized_file_path(path)
        if target is None:
            target = os.fsdecode(path)
        identity = _workspace_file_identity(target)
        super().__init__(
            h5netcdf.File,
            target,
            mode="r+",
            kwargs={
                "invalid_netcdf": None,
                "phony_dims": "sort",
                "decode_vlen_strings": True,
            },
            lock=_workspace_file_lock(target),
            # xarray documents manager_id as a test/dependency-injection hook,
            # but the manager needs repeated workspace readers for the same
            # file identity to share one cached h5netcdf handle.
            manager_id=("erlab-workspace", *identity, "r+"),
        )
        self.workspace_path = target


def _iter_h5netcdf_group_paths(group: object, path: str = "/") -> Iterator[str]:
    yield path
    groups = getattr(group, "groups", {})
    for name, child in groups.items():
        child_path = f"/{name}" if path == "/" else f"{path}/{name}"
        yield from _iter_h5netcdf_group_paths(child, child_path)


def _open_workspace_dataset_from_manager(
    file_manager: WorkspaceFileManager,
    group: str,
    *,
    chunks: typing.Any,
) -> xr.Dataset:
    store = H5NetCDFStore(
        file_manager,
        group=group,
        mode="r+",
        lock=_workspace_file_lock(file_manager.workspace_path),
        autoclose=False,
    )
    if chunks is None:
        return xr.open_dataset(store)
    return xr.open_dataset(store, chunks=chunks)


def open_workspace_dataset(
    path: str | os.PathLike[str],
    group: str,
    *,
    chunks: typing.Any,
) -> xr.Dataset:
    """Open a workspace group through the manager-owned file manager."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    return _open_workspace_dataset_from_manager(
        WorkspaceFileManager(target), group, chunks=chunks
    )


def open_workspace_datatree(
    path: str | os.PathLike[str], *, chunks: typing.Any
) -> xr.DataTree:
    """Open a workspace tree through the manager-owned file manager."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    file_manager = WorkspaceFileManager(target)
    with file_manager.acquire_context() as h5_file:
        group_paths = tuple(_iter_h5netcdf_group_paths(h5_file))

    groups: dict[str, xr.Dataset] = {}
    try:
        for group_path in group_paths:
            groups[group_path] = _open_workspace_dataset_from_manager(
                file_manager, group_path, chunks=chunks
            )
        tree = xr.DataTree.from_dict(groups)
    except Exception:
        for ds in groups.values():
            ds.close()
        raise

    for group_path, ds in groups.items():
        tree[group_path].set_close(ds.close)
    return tree
