"""xarray file-handle helpers for manager workspace IO."""

from __future__ import annotations

import os
import pathlib
import threading
import typing

import h5netcdf
import hdf5plugin
import xarray as xr
from xarray.backends import CachingFileManager, H5NetCDFStore

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

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


def ensure_workspace_hdf5_filters_registered() -> None:
    """Register HDF5 filters needed by compressed workspace files."""
    hdf5plugin.register(force=False)


def workspace_compression_enabled() -> bool:
    import erlab

    return bool(erlab.interactive.options.model.io.workspace.compress)


def _workspace_blosc2_encoding() -> dict[str, typing.Any]:
    ensure_workspace_hdf5_filters_registered()

    return dict(
        hdf5plugin.Blosc2(
            cname="blosclz",
            clevel=3,
            filters=hdf5plugin.Blosc2.SHUFFLE,
        )
    )


def _should_compress_workspace_variable(
    variable: xr.Variable, *, min_bytes: int
) -> bool:
    if variable.dtype.kind not in "iufc":
        return False
    return int(variable.nbytes) >= min_bytes


def workspace_dataset_encoding(
    ds: xr.Dataset,
    *,
    min_bytes: int = _WORKSPACE_COMPRESSION_MIN_BYTES,
    compress: bool | None = None,
) -> dict[Hashable, dict[str, typing.Any]]:
    """Return h5netcdf encodings for compressible workspace data variables."""
    if compress is None:
        compress = workspace_compression_enabled()
    if not compress:
        return {}

    encoding: dict[Hashable, dict[str, typing.Any]] = {}
    for name, variable in ds.data_vars.items():
        if _should_compress_workspace_variable(variable.variable, min_bytes=min_bytes):
            encoding[name] = _workspace_blosc2_encoding()
    return encoding


def workspace_datatree_encoding(
    tree: xr.DataTree,
    *,
    min_bytes: int = _WORKSPACE_COMPRESSION_MIN_BYTES,
    compress: bool | None = None,
) -> dict[str, dict[Hashable, dict[str, typing.Any]]]:
    """Return nested h5netcdf encodings for compressible workspace payloads."""
    if compress is None:
        compress = workspace_compression_enabled()
    if not compress:
        return {}

    encoding: dict[str, dict[Hashable, dict[str, typing.Any]]] = {}
    for node in tree.subtree:
        node_encoding = workspace_dataset_encoding(
            node.to_dataset(inherit=False), min_bytes=min_bytes, compress=compress
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
    chunks: str | None,
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
    chunks: str | None,
) -> xr.Dataset:
    """Open a workspace group through the manager-owned file manager."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    return _open_workspace_dataset_from_manager(
        WorkspaceFileManager(target), group, chunks=chunks
    )


def open_workspace_datatree(
    path: str | os.PathLike[str], *, chunks: str | None
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
