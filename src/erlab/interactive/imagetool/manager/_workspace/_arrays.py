"""Array encoding and HDF5 payload storage for manager workspaces."""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import os
import pathlib
import queue
import shutil
import stat
import sys
import threading
import typing
import uuid
import weakref
from dataclasses import dataclass

import h5netcdf
import hdf5plugin
import numpy as np
import xarray as xr
from xarray.backends import CachingFileManager, FileManager, H5NetCDFStore

import erlab
from erlab.interactive.imagetool import _serialization

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping

    import h5py

    from erlab.interactive._options.schema import WorkspaceCompressionMode
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

from erlab.interactive.imagetool.manager._workspace._format import (
    _is_workspace_internal_group_name,
    _restore_workspace_serialized_attrs,
    _sanitize_workspace_attr_names,
    _workspace_file_is_workspace,
    _workspace_serializable_attrs,
)

_WORKSPACE_FILE_LOCKS: weakref.WeakValueDictionary[str, threading.RLock] = (
    weakref.WeakValueDictionary()
)
_WORKSPACE_FILE_LOCKS_LOCK = threading.Lock()
# Current generations stay alive until their final reader lease is cleaned up.
# This lets a save find and close a handle even when final lease release races it.
_WORKSPACE_FILE_GENERATIONS: dict[str, _WorkspaceFileGeneration] = {}
_WORKSPACE_FILE_GENERATIONS_LOCK = threading.Lock()
# CPython's SimpleQueue.put is reentrant, so a weakref callback can enqueue without
# entering workspace locks or cleanup code.
_WORKSPACE_FILE_CLEANUP_QUEUE: queue.SimpleQueue[_WorkspaceFileGeneration] = (
    queue.SimpleQueue()
)
_WORKSPACE_FILE_CLEANUP_WORKER: threading.Thread | None = None
_WORKSPACE_FILE_CLEANUP_WORKER_LOCK = threading.Lock()
_WORKSPACE_READER_DIRECTORY_SUFFIX = ".readers"
_WORKSPACE_COMPRESSION_MIN_BYTES = 1 << 20  # 1 MiB
_WORKSPACE_MATERIALIZED_CHUNKSIZES = "_erlab_workspace_materialized_chunksizes"
_TOOL_DATA_BLOB_NAME_ATTR = _serialization.TOOL_DATA_BLOB_NAME_ATTR
_SAVED_TOOL_DATA_REFERENCE_DIM = _serialization.SAVED_TOOL_DATA_REFERENCE_DIM
_SAVED_TOOL_DATA_BLOB_DIM_PREFIX = _serialization.SAVED_TOOL_DATA_BLOB_DIM_PREFIX
_WORKSPACE_H5PY_DIMENSION_SCALE_ATTRS = frozenset(
    {"CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST"}
)


@dataclass(frozen=True)
class _WorkspaceReaderFile:
    """One private file that pins a workspace reader generation."""

    path: str
    cleanup: Callable[[], None]


@dataclass(frozen=True)
class _WorkspaceReaderHandoff:
    """One serialized-reader lease for a logical workspace generation."""

    reader_file: _WorkspaceReaderFile
    generation_key: tuple[str, str]


_WORKSPACE_READER_HANDOFFS: dict[str, _WorkspaceReaderHandoff] = {}
_WORKSPACE_READER_HANDOFFS_LOCK = threading.Lock()
_WORKSPACE_READER_HANDOFFS_OWNER_PID = os.getpid()
_WORKSPACE_MAX_PENDING_READER_GENERATIONS = 8
_WORKSPACE_MAX_PENDING_READER_HANDOFFS = 64


def _normalized_workspace_group(group: str) -> str:
    """Return one absolute HDF5 group path."""
    stripped = group.strip("/")
    return "/" if not stripped else f"/{stripped}"


def _hide_workspace_internal_path(path: str | os.PathLike[str]) -> None:
    """Hide an internal workspace file or directory when the platform supports it."""
    path_str = os.fsdecode(path)
    if sys.platform == "darwin":
        with contextlib.suppress(AttributeError, OSError):
            path_stat = os.lstat(path_str)
            if stat.S_ISLNK(path_stat.st_mode):
                return
            os.chflags(path_str, path_stat.st_flags | stat.UF_HIDDEN)
        return
    if os.name != "nt":
        return

    with contextlib.suppress(Exception):
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        attributes = windll.kernel32.GetFileAttributesW(path_str)
        if attributes in (-1, 0xFFFFFFFF):
            return
        windll.kernel32.SetFileAttributesW(
            path_str,
            attributes | 0x2,  # FILE_ATTRIBUTE_HIDDEN
        )


def _workspace_reader_directory(
    workspace_path: str | os.PathLike[str],
) -> pathlib.Path:
    """Return the private reader-generation directory for a workspace."""
    normalized = _normalized_file_path(workspace_path)
    path = pathlib.Path(
        os.fsdecode(workspace_path) if normalized is None else normalized
    )
    return path.with_name(f".{path.name}{_WORKSPACE_READER_DIRECTORY_SUFFIX}")


def _ensure_workspace_reader_directory(
    workspace_path: str | os.PathLike[str],
) -> pathlib.Path:
    directory = _workspace_reader_directory(workspace_path)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        directory_stat = directory.lstat()
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
            directory_stat.st_mode
        ):
            raise OSError(
                f"Workspace reader path is not a private directory: {directory}"
            ) from None
    _hide_workspace_internal_path(directory)
    return directory


def _cleanup_workspace_reader_path(path: pathlib.Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
    with contextlib.suppress(OSError):
        path.parent.rmdir()


def _create_workspace_reader_file(
    source_path: str | os.PathLike[str],
    workspace_path: str | os.PathLike[str],
    *,
    kind: typing.Literal["reader", "export", "handoff"] = "reader",
    copy_only: bool = False,
) -> _WorkspaceReaderFile:
    """Create a private hard link or copy for one reader generation."""
    source = pathlib.Path(source_path).resolve()
    directory = _ensure_workspace_reader_directory(workspace_path)
    destination = directory / f"{kind}-{os.getpid()}-{uuid.uuid4().hex}.itws"
    try:
        if copy_only:
            shutil.copyfile(source, destination)
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
    except BaseException:
        _cleanup_workspace_reader_path(destination)
        raise
    return _WorkspaceReaderFile(
        path=str(destination),
        cleanup=lambda: _cleanup_workspace_reader_path(destination),
    )


def _create_workspace_group_reader_file(
    source_path: str | os.PathLike[str],
    workspace_path: str | os.PathLike[str],
    group: str,
    *,
    kind: typing.Literal["reader", "export"] = "reader",
) -> _WorkspaceReaderFile:
    """Copy one workspace group to a private reader file."""
    source = pathlib.Path(source_path).resolve()
    workspace_group = _normalized_workspace_group(group)
    directory = _ensure_workspace_reader_directory(workspace_path)
    destination = directory / f"{kind}-{os.getpid()}-{uuid.uuid4().hex}.itws"
    copied = True
    try:
        if workspace_group == "/":
            shutil.copyfile(source, destination)
        else:
            ensure_workspace_hdf5_filters_registered()
            with (
                h5py.File(source, "r") as source_file,
                h5py.File(destination, "w") as target_file,
            ):
                copied = _copy_workspace_h5_group_to_open_file(
                    source_file,
                    target_file,
                    workspace_group,
                    workspace_group,
                    None,
                )
    except BaseException:
        _cleanup_workspace_reader_path(destination)
        raise
    if not copied:
        _cleanup_workspace_reader_path(destination)
        raise KeyError(f"Workspace reader group is missing: {workspace_group}")
    return _WorkspaceReaderFile(
        path=str(destination),
        cleanup=lambda: _cleanup_workspace_reader_path(destination),
    )


def _workspace_reader_file_owner_pid(path: pathlib.Path) -> int | None:
    for kind in ("reader", "export", "handoff"):
        prefix = f"{kind}-"
        if not path.name.startswith(prefix) or not path.name.endswith(".itws"):
            continue
        pid_text, separator, token = path.name[len(prefix) :].partition("-")
        if not separator:
            return None
        try:
            pid = int(pid_text)
            uuid.UUID(hex=token.removesuffix(".itws"))
        except (ValueError, TypeError):
            return None
        return pid if pid > 0 else None
    return None


def _cleanup_stale_workspace_reader_files(
    workspace_path: str | os.PathLike[str],
) -> None:
    """Remove private reader files whose owner process no longer exists."""
    directory = _workspace_reader_directory(workspace_path)
    try:
        directory_stat = directory.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        return

    import psutil

    current_pid = os.getpid()
    for path in directory.iterdir():
        owner_pid = _workspace_reader_file_owner_pid(path)
        if (
            owner_pid is None
            or owner_pid == current_pid
            or psutil.pid_exists(owner_pid)
        ):
            continue
        with contextlib.suppress(FileNotFoundError, OSError):
            if stat.S_ISREG(path.lstat().st_mode):
                path.unlink()
    with contextlib.suppress(OSError):
        directory.rmdir()


def _create_workspace_reader_handoff(
    export_file: _WorkspaceReaderFile,
    workspace_path: str | os.PathLike[str],
    revision: str,
) -> _WorkspaceReaderFile:
    """Create one independently owned handoff for a serialized reader."""
    with _WORKSPACE_READER_HANDOFFS_LOCK:
        for path in tuple(_WORKSPACE_READER_HANDOFFS):
            if not pathlib.Path(path).exists():
                _WORKSPACE_READER_HANDOFFS.pop(path, None)
        generation_key = (os.fsdecode(workspace_path), revision)
        pending_generations = {
            handoff.generation_key for handoff in _WORKSPACE_READER_HANDOFFS.values()
        }
        if len(
            _WORKSPACE_READER_HANDOFFS
        ) >= _WORKSPACE_MAX_PENDING_READER_HANDOFFS or (
            generation_key not in pending_generations
            and len(pending_generations) >= _WORKSPACE_MAX_PENDING_READER_GENERATIONS
        ):
            raise RuntimeError(
                "Too many workspace readers are waiting for another process. Wait "
                "for background work to finish, then try again. If no work is "
                "running, restart the ImageTool Manager."
            )
        handoff = _create_workspace_reader_file(
            export_file.path,
            workspace_path,
            kind="handoff",
        )
        _WORKSPACE_READER_HANDOFFS[handoff.path] = _WorkspaceReaderHandoff(
            handoff,
            generation_key,
        )
    return handoff


def _consume_workspace_reader_handoff(
    path: str | os.PathLike[str],
    workspace_path: str | os.PathLike[str],
) -> None:
    """Remove a handoff after its receiver owns a private reader file."""
    handoff_path = pathlib.Path(path)
    try:
        expected_parent = _workspace_reader_directory(workspace_path).resolve()
        is_internal = handoff_path.resolve().parent == expected_parent
    except OSError:
        is_internal = False
    if not is_internal or not handoff_path.name.startswith("handoff-"):
        return
    with _WORKSPACE_READER_HANDOFFS_LOCK:
        handoff = _WORKSPACE_READER_HANDOFFS.pop(str(handoff_path), None)
    if handoff is not None:
        handoff.reader_file.cleanup()
    else:
        _cleanup_workspace_reader_path(handoff_path)


def _cleanup_workspace_reader_handoffs() -> None:
    """Remove unconsumed reader handoffs owned by this Python process."""
    if os.getpid() != _WORKSPACE_READER_HANDOFFS_OWNER_PID:
        return
    with _WORKSPACE_READER_HANDOFFS_LOCK:
        handoffs = tuple(_WORKSPACE_READER_HANDOFFS.values())
        _WORKSPACE_READER_HANDOFFS.clear()
    for handoff in handoffs:
        with contextlib.suppress(Exception):
            handoff.reader_file.cleanup()


def _reset_workspace_reader_handoffs_after_fork() -> None:
    """Forget parent-owned handoffs without deleting them in a forked child."""
    global _WORKSPACE_READER_HANDOFFS_LOCK
    global _WORKSPACE_READER_HANDOFFS_OWNER_PID

    _WORKSPACE_READER_HANDOFFS.clear()
    _WORKSPACE_READER_HANDOFFS_LOCK = threading.Lock()
    _WORKSPACE_READER_HANDOFFS_OWNER_PID = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_workspace_reader_handoffs_after_fork)


atexit.register(_cleanup_workspace_reader_handoffs)


class _WorkspaceFileGenerationResources:
    """Evictable handle and preserved file owned by one generation."""

    def __init__(
        self,
        path: str,
        lock: threading.RLock,
        file_identity: tuple[str, int, int, int],
    ) -> None:
        self.path = path
        self.lock = lock
        self.file_identity = file_identity
        self.file_manager: CachingFileManager[typing.Any] | None = (
            self._new_file_manager(path)
        )
        self.cleanup: Callable[[], None] | None = None
        self.exports: dict[tuple[str, str], _WorkspaceReaderFile] = {}

    def _new_file_manager(self, path: str) -> CachingFileManager[typing.Any]:
        return CachingFileManager(
            h5netcdf.File,
            path,
            mode="r",
            kwargs={
                "invalid_netcdf": None,
                "phony_dims": "sort",
                "decode_vlen_strings": True,
            },
            lock=self.lock,
        )

    def _assert_file_identity(self) -> None:
        identity = _workspace_file_identity(self.path)
        if identity[1:] != self.file_identity[1:]:
            raise RuntimeError(
                "Workspace reader generation changed or is no longer available: "
                f"{self.path}"
            )

    def _acquire_locked(self) -> typing.Any:
        file_manager = self.file_manager
        if file_manager is None:
            raise RuntimeError("Workspace file generation is no longer available")
        self._assert_file_identity()
        file = file_manager.acquire(needs_lock=False)
        try:
            self._assert_file_identity()
        except BaseException:
            file_manager.close(needs_lock=False)
            raise
        return file

    def acquire(self, *, needs_lock: bool) -> typing.Any:
        if needs_lock:
            with self.lock:
                return self._acquire_locked()
        return self._acquire_locked()

    @contextlib.contextmanager
    def acquire_context(self, *, needs_lock: bool) -> Iterator[typing.Any]:
        file_manager = self.file_manager
        if file_manager is None:
            raise RuntimeError("Workspace file generation is no longer available")
        maybe_lock = self.lock if needs_lock else contextlib.nullcontext()
        with maybe_lock:
            self._assert_file_identity()
            with file_manager.acquire_context(needs_lock=False) as file:
                self._assert_file_identity()
                yield file

    def close(self, *, needs_lock: bool) -> None:
        file_manager = self.file_manager
        if file_manager is not None:
            file_manager.close(needs_lock=needs_lock)

    def retire(
        self,
        path: str,
        file_identity: tuple[str, int, int, int],
        cleanup: Callable[[], None],
    ) -> None:
        self.close(needs_lock=False)
        self.path = path
        self.file_identity = file_identity
        self.file_manager = self._new_file_manager(path)
        self.cleanup = cleanup

    def activate(self, path: str, file_identity: tuple[str, int, int, int]) -> None:
        self.close(needs_lock=False)
        self.clear_exports()
        self.path = path
        self.file_identity = file_identity
        self.file_manager = self._new_file_manager(path)

    def export(
        self,
        workspace_path: str,
        revision: str,
        group: str,
    ) -> _WorkspaceReaderFile:
        """Create or reuse the immutable export for one logical generation."""
        workspace_group = _normalized_workspace_group(group)
        key = (revision, workspace_group)
        export_file = self.exports.get(key)
        if export_file is not None:
            _export_identity, export_exists = _workspace_file_state(export_file.path)
            if export_exists:
                return export_file
            self.exports.pop(key, None)

        self.close(needs_lock=False)
        source_identity = _workspace_file_identity(self.path)
        if source_identity[1:] != self.file_identity[1:]:
            raise RuntimeError(
                "Workspace reader generation changed before it could be exported: "
                f"{self.path}"
            )
        export_file = _create_workspace_group_reader_file(
            self.path,
            workspace_path,
            workspace_group,
            kind="export",
        )
        try:
            source_identity_after = _workspace_file_identity(self.path)
        except BaseException:
            export_file.cleanup()
            raise
        if source_identity_after[1:] != source_identity[1:]:
            export_file.cleanup()
            raise RuntimeError(
                "Workspace reader generation changed while it was being exported: "
                f"{self.path}"
            )
        self.exports[key] = export_file
        return export_file

    def clear_exports(self) -> None:
        """Release exports after their handoffs have independent file links."""
        exports = tuple(self.exports.values())
        self.exports.clear()
        for export_file in exports:
            with contextlib.suppress(Exception):
                export_file.cleanup()

    def release(self) -> None:
        file_manager = self.file_manager
        self.file_manager = None
        cleanup = self.cleanup
        self.cleanup = None
        exports = tuple(self.exports.values())
        self.exports.clear()
        with contextlib.suppress(Exception):
            if file_manager is not None:
                file_manager.close(needs_lock=False)
        if cleanup is not None:
            with contextlib.suppress(Exception):
                cleanup()
        for export_file in exports:
            with contextlib.suppress(Exception):
                export_file.cleanup()


class _WorkspaceFileGeneration:
    """Own one workspace file generation and its shared reader handle."""

    def __init__(
        self,
        workspace_path: str,
        file_identity: tuple[str, int, int, int],
        *,
        created_without_file: bool = False,
        reader_file: _WorkspaceReaderFile | None = None,
    ) -> None:
        self.workspace_path = workspace_path
        self.file_identity = file_identity
        resource_path = workspace_path if reader_file is None else reader_file.path
        resource_identity = (
            file_identity
            if reader_file is None
            else _workspace_file_identity(reader_file.path)
        )
        self._resources = _WorkspaceFileGenerationResources(
            resource_path,
            _workspace_file_lock(workspace_path),
            resource_identity,
        )
        if reader_file is not None:
            self._resources.cleanup = reader_file.cleanup
        self._manager_count = 0
        self._managers: weakref.WeakSet[WorkspaceFileManager] = weakref.WeakSet()
        self._state_lock = threading.Lock()
        self._retired = reader_file is not None
        self._created_without_file = created_without_file
        self._revision = uuid.uuid4().hex
        self._invalid_reason: str | None = None
        self._cleanup_scheduled = False
        self._disposed = False
        self._finalizer = weakref.finalize(self, self._resources.release)

    @property
    def path(self) -> str:
        return self._resources.path

    @property
    def retired(self) -> bool:
        with self._state_lock:
            return self._retired

    @property
    def created_without_file(self) -> bool:
        with self._state_lock:
            return self._created_without_file

    @property
    def revision(self) -> str:
        """Return the logical identity of the current generation contents."""
        with self._state_lock:
            return self._revision

    def add_manager(self, manager: WorkspaceFileManager | None = None) -> None:
        with self._state_lock:
            if self._disposed or self._invalid_reason is not None:
                raise RuntimeError(
                    self._invalid_reason
                    or "Workspace file generation is no longer available"
                )
            self._manager_count += 1
            if manager is not None:
                self._managers.add(manager)

    def release_manager(self, manager: WorkspaceFileManager | None = None) -> bool:
        """Release one reader and claim cleanup after the final reader."""
        with self._state_lock:
            if manager is not None:
                self._managers.discard(manager)
            self._manager_count -= 1
            if self._manager_count != 0 or self._disposed or self._cleanup_scheduled:
                return False
            self._cleanup_scheduled = True
            return True

    def has_managers(self) -> bool:
        with self._state_lock:
            return self._manager_count != 0

    def managers(self) -> tuple[WorkspaceFileManager, ...]:
        """Return the live managers that use this generation."""
        with self._state_lock:
            return tuple(self._managers)

    def acquire(self, *, needs_lock: bool) -> typing.Any:
        self._raise_if_unavailable()
        return self._resources.acquire(needs_lock=needs_lock)

    def acquire_context(
        self, *, needs_lock: bool
    ) -> contextlib.AbstractContextManager[typing.Any]:
        self._raise_if_unavailable()
        return self._resources.acquire_context(needs_lock=needs_lock)

    def close(self, *, needs_lock: bool) -> None:
        self._resources.close(needs_lock=needs_lock)

    def retire(
        self,
        path: str,
        file_identity: tuple[str, int, int, int],
        cleanup: Callable[[], None],
    ) -> None:
        self._resources.retire(path, file_identity, cleanup)
        with self._state_lock:
            self._retired = True

    def activate_new_file(self) -> None:
        with self._state_lock:
            if not self._created_without_file:
                raise RuntimeError(
                    "Cannot bind an existing workspace reader generation to a new file"
                )
        identity = _workspace_file_identity(self.workspace_path)
        self._resources.activate(self.workspace_path, identity)
        self.file_identity = identity
        with self._state_lock:
            self._created_without_file = False
            self._revision = uuid.uuid4().hex

    def refresh_in_place(self, file_identity: tuple[str, int, int, int]) -> None:
        """Reopen this generation after a managed in-place transaction."""
        self._raise_if_unavailable()
        self._resources.activate(self.workspace_path, file_identity)
        self.file_identity = file_identity
        with self._state_lock:
            self._revision = uuid.uuid4().hex

    def invalidate(self, reason: str) -> None:
        self._resources.close(needs_lock=False)
        with self._state_lock:
            self._retired = True
            self._invalid_reason = reason

    def export_reader_state(
        self, group: str
    ) -> tuple[str, str, tuple[int, int, int], str]:
        """Create an owned handoff for one cross-process reader."""
        self._raise_if_unavailable()
        export_file = self._resources.export(
            self.workspace_path,
            self.revision,
            group,
        )
        handoff = _create_workspace_reader_handoff(
            export_file,
            self.workspace_path,
            self.revision,
        )
        identity = _workspace_file_identity(handoff.path)[1:]
        return self.workspace_path, handoff.path, identity, group

    def _raise_if_unavailable(self) -> None:
        with self._state_lock:
            if self._disposed or self._invalid_reason is not None:
                raise RuntimeError(
                    self._invalid_reason
                    or "Workspace file generation is no longer available"
                )

    def keep_if_in_use(self) -> bool:
        """Cancel pending cleanup when this generation has a new reader."""
        with self._state_lock:
            if self._manager_count == 0:
                return False
            self._cleanup_scheduled = False
            return True

    def dispose(self) -> None:
        """Release this generation once it has no reader leases."""
        with self._state_lock:
            if self._disposed:
                return
            self._disposed = True
            self._cleanup_scheduled = False
        self._finalizer()


def _replace_h5_attrs(target_attrs, attrs: Mapping[typing.Any, typing.Any]) -> None:
    for key in list(target_attrs):
        del target_attrs[key]
    for key, value in _workspace_serializable_attrs(attrs).items():
        target_attrs[key] = value


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
    with _WORKSPACE_FILE_LOCKS_LOCK:
        lock = _WORKSPACE_FILE_LOCKS.get(target)
        if lock is None:
            lock = threading.RLock()
            _WORKSPACE_FILE_LOCKS[target] = lock
        return lock


def _workspace_file_state(
    path: str | os.PathLike[str],
) -> tuple[tuple[str, int, int, int], bool]:
    """Read a workspace identity and existence state with one stat call."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    try:
        stat_result = os.stat(target)
    except FileNotFoundError:
        return (target, 0, 0, 0), False
    return (
        (target, stat_result.st_dev, stat_result.st_ino, stat_result.st_mtime_ns),
        True,
    )


def _workspace_file_identity(
    path: str | os.PathLike[str],
) -> tuple[str, int, int, int]:
    return _workspace_file_state(path)[0]


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


def ensure_workspace_hdf5_filters_registered() -> None:
    """Register HDF5 filters needed by compressed workspace files."""
    hdf5plugin.register(force=False)


def workspace_compression_mode() -> WorkspaceCompressionMode:

    return erlab.interactive.options.model.io.workspace.compression


def _workspace_blosc2_encoding(
    compression_mode: WorkspaceCompressionMode,
) -> dict[str, typing.Any]:
    if compression_mode == "none":
        return {}

    ensure_workspace_hdf5_filters_registered()
    cname: typing.Literal["blosclz", "zstd"]
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
    """Return a valid fixed HDF5 chunk shape for workspace data."""
    chunks = data_array.chunks
    if chunks is None:
        materialized_chunks = data_array.encoding.get(
            _WORKSPACE_MATERIALIZED_CHUNKSIZES
        )
        if (
            not isinstance(materialized_chunks, tuple)
            or len(materialized_chunks) != data_array.ndim
        ):
            return None
        chunks = tuple((chunk,) for chunk in materialized_chunks)
    if data_array.ndim == 0:
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


def _workspace_file_generation(
    path: str,
    manager: WorkspaceFileManager | None = None,
) -> _WorkspaceFileGeneration:
    """Return a current generation and optionally register its reader lease."""
    _ensure_workspace_file_cleanup_worker()
    with _workspace_file_lock(path):
        identity, file_exists = _workspace_file_state(path)
        with _WORKSPACE_FILE_GENERATIONS_LOCK:
            generation = _WORKSPACE_FILE_GENERATIONS.get(path)
        # Published workspace files are immutable. A replacement or an external
        # in-place modification therefore always starts a new generation.
        if generation is None or generation.file_identity[1:] != identity[1:]:
            if generation is not None:
                generation.invalidate(
                    "Workspace file was replaced outside its managed save operation: "
                    f"{path}"
                )
            generation = _WorkspaceFileGeneration(
                path,
                identity,
                created_without_file=not file_exists,
            )
            with _WORKSPACE_FILE_GENERATIONS_LOCK:
                _WORKSPACE_FILE_GENERATIONS[path] = generation
        if manager is not None:
            generation.add_manager(manager)
        return generation


def _workspace_file_generation_from_reader(
    workspace_path: str,
    reader_path: str,
    expected_identity: tuple[int, int, int],
) -> _WorkspaceFileGeneration:
    """Create a detached generation from an immutable serialized reader file."""
    _ensure_workspace_file_cleanup_worker()
    with _workspace_file_lock(workspace_path):
        if _workspace_file_identity(reader_path)[1:] != expected_identity:
            _consume_workspace_reader_handoff(reader_path, workspace_path)
            raise RuntimeError(
                "Serialized workspace reader generation is no longer available: "
                f"{reader_path}"
            )
        reader_file = _create_workspace_reader_file(reader_path, workspace_path)
        try:
            reader_identity_after = _workspace_file_identity(reader_path)[1:]
        except BaseException:
            reader_file.cleanup()
            raise
        if reader_identity_after != expected_identity:
            reader_file.cleanup()
            _consume_workspace_reader_handoff(reader_path, workspace_path)
            raise RuntimeError(
                "Serialized workspace reader generation changed while it was "
                f"being opened: {reader_path}"
            )
        try:
            generation = _WorkspaceFileGeneration(
                workspace_path,
                _workspace_file_identity(workspace_path),
                reader_file=reader_file,
            )
        except BaseException:
            reader_file.cleanup()
            raise
        _consume_workspace_reader_handoff(reader_path, workspace_path)
        return generation


def _current_workspace_file_generation(
    path: str | os.PathLike[str],
) -> _WorkspaceFileGeneration | None:
    """Return the current generation for a path while its file lock is held."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    with _WORKSPACE_FILE_GENERATIONS_LOCK:
        return _WORKSPACE_FILE_GENERATIONS.get(target)


def _retire_workspace_file_generation(
    path: str | os.PathLike[str],
    generation: _WorkspaceFileGeneration,
    preserved_path: str,
    preserved_identity: tuple[str, int, int, int],
    cleanup: Callable[[], None],
) -> None:
    """Move a published generation to its preserved backing file."""
    _discard_workspace_file_generation(path, generation)
    generation.retire(preserved_path, preserved_identity, cleanup)


def _discard_workspace_file_generation(
    path: str | os.PathLike[str],
    generation: _WorkspaceFileGeneration,
) -> None:
    """Remove a generation from the current path registry."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    with _WORKSPACE_FILE_GENERATIONS_LOCK:
        if _WORKSPACE_FILE_GENERATIONS.get(target) is generation:
            del _WORKSPACE_FILE_GENERATIONS[target]


def _release_workspace_file_generation(
    generation: _WorkspaceFileGeneration,
    manager: WorkspaceFileManager | None = None,
) -> None:
    """Release one reader lease without doing cleanup in the finalizer."""
    if generation.release_manager(manager):
        _WORKSPACE_FILE_CLEANUP_QUEUE.put(generation)


def _ensure_workspace_file_cleanup_worker() -> None:
    """Start the shared workspace generation cleanup worker once."""
    global _WORKSPACE_FILE_CLEANUP_WORKER

    with _WORKSPACE_FILE_CLEANUP_WORKER_LOCK:
        worker = _WORKSPACE_FILE_CLEANUP_WORKER
        if worker is not None and worker.is_alive():
            return
        worker = threading.Thread(
            target=_workspace_file_cleanup_worker,
            name="erlab-workspace-reader-cleanup",
            daemon=True,
        )
        _WORKSPACE_FILE_CLEANUP_WORKER = worker
        worker.start()


def _workspace_file_cleanup_worker() -> None:
    """Clean unused generations outside object finalizer call stacks."""
    while True:
        generation = _WORKSPACE_FILE_CLEANUP_QUEUE.get()
        try:
            _cleanup_workspace_file_generation(generation)
        finally:
            # Do not retain the last disposed generation while the queue is idle.
            del generation


def _cleanup_workspace_file_generation(
    generation: _WorkspaceFileGeneration,
) -> None:
    """Wait for the path lock and finish a pending generation cleanup."""
    with _workspace_file_lock(generation.workspace_path):
        _cleanup_workspace_file_generation_locked(generation)


def _cleanup_workspace_file_generation_locked(
    generation: _WorkspaceFileGeneration,
) -> None:
    """Dispose an unused generation while its workspace path is locked."""
    if generation.keep_if_in_use():
        return
    _discard_workspace_file_generation(generation.workspace_path, generation)
    generation.dispose()


def _workspace_groups_intersect(first: str, second: str) -> bool:
    first = first.strip("/")
    second = second.strip("/")
    if not first or not second:
        return True
    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


def _detach_workspace_file_generation_readers(
    path: str | os.PathLike[str],
    rewrite_groups: Iterable[str],
) -> None:
    """Move readers of groups that will change to private snapshots."""
    generation = _current_workspace_file_generation(path)
    if generation is None:
        return
    normalized_rewrites = tuple(
        _normalized_workspace_group(group) for group in rewrite_groups
    )
    managers = tuple(
        manager
        for manager in generation.managers()
        if manager._generation is generation
        and manager._finalizer.alive
        and any(
            _workspace_groups_intersect(manager.workspace_group, rewrite_group)
            for rewrite_group in normalized_rewrites
        )
    )
    if not managers:
        return

    generation.close(needs_lock=False)
    source_identity = _workspace_file_identity(generation.path)
    if source_identity[1:] != generation._resources.file_identity[1:]:
        raise RuntimeError(
            "Workspace reader generation changed before active readers could be "
            f"preserved: {generation.path}"
        )

    detached_by_group: dict[str, _WorkspaceFileGeneration] = {}
    try:
        for group in {manager.workspace_group for manager in managers}:
            reader_file = _create_workspace_group_reader_file(
                generation.path,
                generation.workspace_path,
                group,
            )
            detached_by_group[group] = _WorkspaceFileGeneration(
                generation.workspace_path,
                generation.file_identity,
                reader_file=reader_file,
            )
    except BaseException:
        for detached in detached_by_group.values():
            if not detached.has_managers():
                detached.dispose()
        raise
    source_identity_after = _workspace_file_identity(generation.path)
    if source_identity_after[1:] != source_identity[1:]:
        for detached in detached_by_group.values():
            detached.dispose()
        raise RuntimeError(
            "Workspace reader generation changed while active readers were being "
            f"preserved: {generation.path}"
        )
    try:
        for manager in managers:
            manager._switch_generation(detached_by_group[manager.workspace_group])
    except BaseException:
        for detached in detached_by_group.values():
            if not detached.has_managers():
                detached.dispose()
        raise


def _workspace_file_manager_pickle_state(
    manager: WorkspaceFileManager,
) -> tuple[str, str, tuple[int, int, int], str]:
    """Return an immutable serialization reference for one generation."""
    with _workspace_file_lock(manager.workspace_path):
        return manager._generation.export_reader_state(manager.workspace_group)


class WorkspaceFileManager(FileManager[typing.Any]):
    """xarray file manager backed by an explicit workspace generation."""

    def __init__(self, path: str | os.PathLike[str], group: str = "/") -> None:
        self._initialize(path, group)

    def _initialize(self, path: str | os.PathLike[str], group: str) -> None:
        ensure_workspace_hdf5_filters_registered()
        target = _normalized_file_path(path)
        if target is None:
            target = os.fsdecode(path)
        self.workspace_path = target
        self.workspace_group = _normalized_workspace_group(group)
        self._generation = _workspace_file_generation(target, self)
        self._finalizer = weakref.finalize(
            self, _release_workspace_file_generation, self._generation
        )

    def _switch_generation(self, generation: _WorkspaceFileGeneration) -> None:
        """Move this manager to an already prepared reader generation."""
        with _workspace_file_lock(self.workspace_path):
            if generation is self._generation:
                return
            generation.add_manager(self)
            previous = self._generation
            if self._finalizer.detach() is None:
                _release_workspace_file_generation(generation, self)
                raise RuntimeError("Workspace file manager is already released")
            self._generation = generation
            self._finalizer = weakref.finalize(
                self, _release_workspace_file_generation, generation
            )
            _release_workspace_file_generation(previous, self)

    def __getstate__(self) -> tuple[str, str, tuple[int, int, int], str]:
        """Serialize an immutable reference to this workspace generation."""
        return _workspace_file_manager_pickle_state(self)

    def __dask_tokenize__(self) -> tuple[str, str, str, str]:
        """Return a stable token without creating a cross-process handoff."""
        return (
            type(self).__name__,
            self.workspace_path,
            self.workspace_group,
            self._generation.revision,
        )

    def __setstate__(self, state: tuple[str, str, tuple[int, int, int], str]) -> None:
        workspace_path, reader_path, expected_identity, group = state
        ensure_workspace_hdf5_filters_registered()
        target = _normalized_file_path(workspace_path)
        if target is None:
            target = os.fsdecode(workspace_path)
        self.workspace_path = target
        self.workspace_group = _normalized_workspace_group(group)
        with _workspace_file_lock(target):
            generation = _workspace_file_generation_from_reader(
                target,
                reader_path,
                expected_identity,
            )
            try:
                generation.add_manager(self)
            except BaseException:
                generation.dispose()
                raise
            self._generation = generation
        self._finalizer = weakref.finalize(
            self, _release_workspace_file_generation, self._generation
        )

    def acquire(self, needs_lock: bool = True) -> typing.Any:
        if needs_lock:
            with _workspace_file_lock(self.workspace_path):
                return self._generation.acquire(needs_lock=False)
        return self._generation.acquire(needs_lock=needs_lock)

    def acquire_context(
        self, needs_lock: bool = True
    ) -> contextlib.AbstractContextManager[typing.Any]:
        if needs_lock:
            return self._acquire_context_locked()
        return self._generation.acquire_context(needs_lock=needs_lock)

    @contextlib.contextmanager
    def _acquire_context_locked(self) -> Iterator[typing.Any]:
        with (
            _workspace_file_lock(self.workspace_path),
            self._generation.acquire_context(needs_lock=False) as file,
        ):
            yield file

    def close(self, needs_lock: bool = True) -> None:
        if needs_lock:
            with _workspace_file_lock(self.workspace_path):
                self._generation.close(needs_lock=False)
            return
        self._generation.close(needs_lock=needs_lock)

    def _release(self) -> None:
        """Close this manager and release its generation lease."""
        self.close()
        self._finalizer()


class _WorkspaceH5NetCDFStore(H5NetCDFStore):
    """HDF5 store with side-effect-free Dask generation identity."""

    def __dask_tokenize__(self) -> tuple[object, ...]:
        """Identify lazy arrays without serializing their file manager."""
        manager = typing.cast("WorkspaceFileManager", self._manager)
        return (
            type(self).__name__,
            manager.__dask_tokenize__(),
            self._group,
            self._mode,
        )


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
    store = _WorkspaceH5NetCDFStore(
        file_manager,
        group=group,
        mode="r",
        lock=_workspace_file_lock(file_manager.workspace_path),
        autoclose=False,
    )
    if chunks is None:
        dataset = xr.open_dataset(store)
    else:
        dataset = xr.open_dataset(store, chunks=chunks)
    for variable in dataset.variables.values():
        if "source" in variable.encoding:
            variable.encoding["source"] = file_manager.workspace_path
    return dataset


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
        WorkspaceFileManager(target, group), group, chunks=chunks
    )


def open_workspace_datatree(
    path: str | os.PathLike[str], *, chunks: typing.Any
) -> xr.DataTree:
    """Open a workspace tree through the manager-owned file manager."""
    target = _normalized_file_path(path)
    if target is None:
        target = os.fsdecode(path)
    enumerator = WorkspaceFileManager(target)
    groups: dict[str, xr.Dataset] = {}
    try:
        with enumerator.acquire_context() as h5_file:
            group_paths = tuple(_iter_h5netcdf_group_paths(h5_file))
        for group_path in group_paths:
            groups[group_path] = _open_workspace_dataset_from_manager(
                WorkspaceFileManager(target, group_path),
                group_path,
                chunks=chunks,
            )
        tree = xr.DataTree.from_dict(groups)
    except Exception:
        for ds in groups.values():
            ds.close()
        raise
    finally:
        enumerator._release()

    for group_path, ds in groups.items():
        tree[group_path].set_close(ds.close)
    return tree


def _h5_path_exists(h5_file, path: str) -> bool:
    stripped = path.strip("/")
    return stripped == "" or stripped in h5_file


def _delete_h5_path(h5_file, path: str) -> None:
    stripped = path.strip("/")
    if stripped and stripped in h5_file:
        del h5_file[stripped]


def _ensure_h5_parent_group(h5_file, path: str):
    parent = h5_file
    parts = [part for part in path.strip("/").split("/") if part]
    for part in parts[:-1]:
        parent = parent.require_group(part)
    return parent


def _h5py_attrs_to_dict(
    attrs: typing.Any, *, exclude: Iterable[typing.Hashable] = ()
) -> dict[typing.Hashable, typing.Any]:
    excluded = set(exclude)
    out: dict[typing.Hashable, typing.Any] = {}
    for key, value in attrs.items():
        if key in excluded:
            continue
        if isinstance(value, bytes):
            value = value.decode()
        out[key] = value
    return _restore_workspace_serialized_attrs(out)


def _read_workspace_root_attrs_h5py(
    fname: str | os.PathLike[str],
) -> dict[typing.Hashable, typing.Any]:
    ensure_workspace_hdf5_filters_registered()
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        return _h5py_attrs_to_dict(h5_file.attrs)


def _workspace_live_root_group_copy_groups(
    fname: str | os.PathLike[str],
) -> tuple[tuple[str, str, dict[str, typing.Any] | None], ...]:
    ensure_workspace_hdf5_filters_registered()
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        return tuple(
            (name, name, None)
            for name, item in h5_file.items()
            if isinstance(item, h5py.Group)
            and not _is_workspace_internal_group_name(name)
        )


def _workspace_h5_object_storage_size(obj: typing.Any) -> int:
    if isinstance(obj, h5py.Dataset):
        return max(0, int(obj.id.get_storage_size()))
    if isinstance(obj, h5py.Group):
        return sum(_workspace_h5_object_storage_size(child) for child in obj.values())
    return 0


def _workspace_h5_paths_storage_size(
    fname: str | os.PathLike[str],
    paths: Iterable[str],
) -> tuple[int, int]:
    total = 0
    existing_count = 0
    ensure_workspace_hdf5_filters_registered()
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        for path in paths:
            path = path.strip("/")
            if path not in h5_file:
                continue
            existing_count += 1
            total += _workspace_h5_object_storage_size(h5_file[path])
    return total, existing_count


def _workspace_live_h5_storage_size(fname: str | os.PathLike[str]) -> int:
    total = 0
    ensure_workspace_hdf5_filters_registered()
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError("Not a valid workspace file")
        for name, item in h5_file.items():
            if isinstance(item, h5py.Group) and not _is_workspace_internal_group_name(
                name
            ):
                total += _workspace_h5_object_storage_size(item)
    return total


def _workspace_h5py_dataset_storage_supported(dataset: typing.Any) -> bool:
    return dataset.dtype.kind in "biufcS" or (
        dataset.dtype.kind == "O" and h5py.check_string_dtype(dataset.dtype) is not None
    )


def _workspace_h5py_coord_dims_fit(
    coord: xr.DataArray, data_array: xr.DataArray
) -> bool:
    return all(
        dim in data_array.sizes and coord.sizes[dim] == data_array.sizes[dim]
        for dim in coord.dims
    )


def _workspace_h5py_variable_payload(
    variable: xr.Variable, name: typing.Hashable
) -> tuple[typing.Any, dict[typing.Hashable, typing.Any], typing.Any] | None:
    try:
        if variable.dtype.kind == "M":
            variable = xr.coders.CFDatetimeCoder().encode(variable, name=str(name))
        elif variable.dtype.kind == "m":
            variable = xr.coders.CFTimedeltaCoder().encode(variable, name=str(name))
    except Exception:
        return None

    data = np.asarray(variable.data)
    dtype = None
    if data.dtype.kind == "U":
        data = data.astype(object)
        dtype = h5py.string_dtype(encoding="utf-8")
    elif data.dtype.kind not in "biufcS":
        return None
    return data, dict(variable.attrs), dtype


def _workspace_h5py_dataarray_can_write(data_array: xr.DataArray) -> bool:
    if data_array.chunks is not None:
        return False
    return (
        _workspace_h5py_variable_payload(data_array.variable, data_array.name)
        is not None
    )


def _workspace_h5py_read_values(dataset: typing.Any) -> typing.Any:
    string_info = h5py.check_string_dtype(dataset.dtype)
    if dataset.dtype.kind == "O" and string_info is not None:
        values = np.asarray(dataset.asstr()[()])
        if values.dtype.kind == "O":
            values = values.astype(str)
        return values
    return np.asarray(dataset[()])


def _workspace_h5py_decode_coord_variable(
    variable: xr.Variable, name: str
) -> xr.Variable | None:
    attrs = variable.attrs
    dtype_attr = attrs.get("dtype")
    units_attr = attrs.get("units")
    calendar_attr = attrs.get("calendar")
    try:
        if isinstance(dtype_attr, str) and dtype_attr.startswith("timedelta64["):
            return xr.coders.CFTimedeltaCoder().decode(variable, name=name)
        if (
            isinstance(units_attr, str)
            and isinstance(calendar_attr, str)
            and " since " in units_attr
        ):
            return xr.coders.CFDatetimeCoder().decode(variable, name=name)
    except Exception:
        return None
    return variable


def _workspace_h5py_dataset_variable(
    dataset: typing.Any,
    dims: tuple[str, ...],
    *,
    name: str,
    exclude_attrs: Iterable[typing.Hashable],
) -> xr.Variable | None:
    if not _workspace_h5py_dataset_storage_supported(dataset):
        return None
    variable = xr.Variable(
        dims,
        _workspace_h5py_read_values(dataset),
        _h5py_attrs_to_dict(dataset.attrs, exclude=exclude_attrs),
    )
    return _workspace_h5py_decode_coord_variable(variable, name)


def _workspace_h5py_create_kwargs(
    encoding: Mapping[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    if encoding is None:
        return {}
    kwargs: dict[str, typing.Any] = {}
    if "chunksizes" in encoding:
        kwargs["chunks"] = encoding["chunksizes"]
    for key in ("compression", "compression_opts", "shuffle", "fletcher32"):
        if key in encoding:
            kwargs[key] = encoding[key]
    return kwargs


def _workspace_h5py_filter_options(dataset: typing.Any) -> dict[int, tuple[int, ...]]:
    create_plist = dataset.id.get_create_plist()
    return {
        create_plist.get_filter(index)[0]: tuple(create_plist.get_filter(index)[2])
        for index in range(create_plist.get_nfilters())
    }


def _workspace_h5py_blosc2_options_match(
    actual_options: tuple[int, ...], expected_options: tuple[int, ...]
) -> bool:
    if actual_options == expected_options:
        return True
    if len(actual_options) < 7 or len(expected_options) < 7:
        return False
    return actual_options[4:7] == expected_options[4:7]


def _workspace_h5py_dataset_matches_encoding(
    dataset: typing.Any,
    encoding: Mapping[typing.Any, typing.Any],
) -> bool:
    filters = _workspace_h5py_filter_options(dataset)
    expected_filter = encoding.get("compression")
    if expected_filter is None:
        return not filters
    actual_options = filters.get(int(expected_filter))
    if actual_options is None:
        return False
    expected_options = encoding.get("compression_opts")
    if expected_options is None:
        return True
    expected_options = tuple(expected_options)
    if int(expected_filter) == hdf5plugin.Blosc2.filter_id:
        return _workspace_h5py_blosc2_options_match(actual_options, expected_options)
    return actual_options == expected_options


def _workspace_h5_group_matches_compression_mode(
    h5_file: typing.Any,
    group_path: str,
    ds: xr.Dataset,
    compression_mode: WorkspaceCompressionMode,
) -> bool:
    group_path = group_path.strip("/")
    if group_path not in h5_file:
        return False
    group = h5_file[group_path]
    encoding = workspace_dataset_encoding(ds, compression_mode=compression_mode)
    for name in ds.data_vars:
        dataset_name = str(name)
        if dataset_name not in group:
            return False
        dataset = group[dataset_name]
        if not _workspace_h5py_dataset_matches_encoding(
            dataset, encoding.get(name, {})
        ):
            return False
    return True


def _workspace_h5py_attr_text(value: typing.Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode()
    if hasattr(value, "decode"):
        decoded = value.decode()
        return str(decoded)
    if isinstance(value, str):
        return value
    return None


def _workspace_h5py_dataset_is_dimension_scale(dataset: typing.Any) -> bool:
    return _workspace_h5py_attr_text(dataset.attrs.get("CLASS")) == "DIMENSION_SCALE"


def _h5_group_matches_compression(
    h5_file: typing.Any,
    group_path: str,
    compression_mode: WorkspaceCompressionMode,
) -> bool:
    group_path = group_path.strip("/")
    if group_path not in h5_file:
        return False
    group = h5_file[group_path]
    if not isinstance(group, h5py.Group):
        return False
    compression_encoding = _workspace_blosc2_encoding(compression_mode)
    for item in group.values():
        if not isinstance(item, h5py.Dataset):
            continue
        if _workspace_h5py_dataset_is_dimension_scale(item):
            continue
        encoding = {}
        if (
            compression_encoding
            and item.dtype.kind in "iufc"
            and int(item.size) * int(item.dtype.itemsize)
            >= _WORKSPACE_COMPRESSION_MIN_BYTES
        ):
            encoding = compression_encoding
        if not _workspace_h5py_dataset_matches_encoding(item, encoding):
            return False
    return True


def _workspace_h5py_type_contains_reference(type_id: typing.Any) -> bool:
    type_class = type_id.get_class()
    if type_class == h5py.h5t.REFERENCE:
        return True
    if type_class in {h5py.h5t.ARRAY, h5py.h5t.VLEN}:
        super_type = type_id.get_super()
        try:
            return _workspace_h5py_type_contains_reference(super_type)
        finally:
            super_type.close()
    if type_class == h5py.h5t.COMPOUND:
        for index in range(type_id.get_nmembers()):
            member_type = type_id.get_member_type(index)
            try:
                if _workspace_h5py_type_contains_reference(member_type):
                    return True
            finally:
                member_type.close()
    return False


def _workspace_h5py_attr_contains_reference(
    source_attrs: typing.Any, key: typing.Hashable
) -> bool:
    attr_id = source_attrs.get_id(key)
    type_id = attr_id.get_type()
    try:
        return _workspace_h5py_type_contains_reference(type_id)
    finally:
        type_id.close()
        attr_id.close()


def _workspace_h5py_copy_regular_attrs(
    source_attrs: typing.Any,
    target_attrs: typing.Any,
    *,
    skip_dimension_scale_attrs: bool,
) -> None:
    for key, value in source_attrs.items():
        if skip_dimension_scale_attrs and key in _WORKSPACE_H5PY_DIMENSION_SCALE_ATTRS:
            continue
        if _workspace_h5py_attr_contains_reference(source_attrs, key):
            continue
        target_attrs[key] = value


def _workspace_h5py_rebuild_dimension_scales(
    source_group: typing.Any,
    target_group: typing.Any,
) -> None:
    _workspace_h5py_copy_regular_attrs(
        source_group.attrs,
        target_group.attrs,
        skip_dimension_scale_attrs=False,
    )
    scales_by_id: dict[int, typing.Any] = {}
    for name, source_obj in source_group.items():
        target_obj = target_group[name]
        if isinstance(source_obj, h5py.Group):
            _workspace_h5py_rebuild_dimension_scales(source_obj, target_obj)
            continue
        if not isinstance(source_obj, h5py.Dataset):
            continue
        _workspace_h5py_copy_regular_attrs(
            source_obj.attrs,
            target_obj.attrs,
            skip_dimension_scale_attrs=True,
        )
        if not _workspace_h5py_dataset_is_dimension_scale(source_obj):
            continue
        dim_id = source_obj.attrs.get("_Netcdf4Dimid")
        if dim_id is None:
            continue
        scale_name = _workspace_h5py_attr_text(source_obj.attrs.get("NAME")) or name
        target_obj.make_scale(scale_name)
        target_obj.attrs["_Netcdf4Dimid"] = np.int32(dim_id)
        if "_Netcdf4Coordinates" in source_obj.attrs:
            target_obj.attrs["_Netcdf4Coordinates"] = source_obj.attrs[
                "_Netcdf4Coordinates"
            ]
        scales_by_id[int(dim_id)] = target_obj

    for name, source_obj in source_group.items():
        if not isinstance(source_obj, h5py.Dataset):
            continue
        if _workspace_h5py_dataset_is_dimension_scale(source_obj):
            continue
        if "_Netcdf4Coordinates" not in source_obj.attrs:
            continue
        target_obj = target_group[name]
        coordinate_ids = np.asarray(source_obj.attrs["_Netcdf4Coordinates"]).reshape(-1)
        if len(coordinate_ids) != target_obj.ndim:
            continue
        for axis, dim_id in enumerate(coordinate_ids):
            scale = scales_by_id.get(int(dim_id))
            if scale is not None:
                target_obj.dims[axis].attach_scale(scale)


def _copy_workspace_h5_group_to_open_file(
    source_file: typing.Any,
    target_file: typing.Any,
    source_path: str,
    target_path: str,
    attrs: Mapping[str, typing.Any] | None,
) -> bool:
    source_path = source_path.strip("/")
    target_path = target_path.strip("/")
    if source_path not in source_file:
        return False
    source_group = source_file[source_path]
    if not isinstance(source_group, h5py.Group):
        return False
    parent = _ensure_h5_parent_group(target_file, target_path)
    target_name = target_path.rsplit("/", maxsplit=1)[-1]
    if target_name in parent:
        del parent[target_name]
    source_file.copy(source_path, parent, name=target_name, without_attrs=True)
    target_group = parent[target_name]
    _workspace_h5py_rebuild_dimension_scales(source_group, target_group)
    if attrs is not None:
        _replace_h5_attrs(target_group.attrs, attrs)
    return True


def _workspace_h5py_create_dataset(
    group: typing.Any,
    name: str,
    variable: xr.Variable,
    *,
    encoding: Mapping[str, typing.Any] | None = None,
) -> typing.Any | None:
    payload = _workspace_h5py_variable_payload(variable, name)
    if payload is None:
        return None
    data, attrs, dtype = payload
    kwargs = _workspace_h5py_create_kwargs(encoding)
    if dtype is not None:
        kwargs["dtype"] = dtype
    dataset = group.create_dataset(name, data=data, **kwargs)
    for key, value in attrs.items():
        dataset.attrs[key] = value
    return dataset


def _workspace_h5py_tool_data_blob_dim(variable_name: str) -> str:
    return f"{_SAVED_TOOL_DATA_BLOB_DIM_PREFIX}{variable_name.encode().hex()}>"


def _workspace_h5py_dataarray_is_tool_reference(data_array: xr.DataArray) -> bool:
    return (
        data_array.ndim == 1
        and data_array.dims == (_SAVED_TOOL_DATA_REFERENCE_DIM,)
        and data_array.size == 0
    )


def _workspace_h5py_dataarray_is_tool_blob(data_array: xr.DataArray) -> bool:
    return _TOOL_DATA_BLOB_NAME_ATTR in data_array.attrs


def _workspace_h5py_dataarray_is_independent_tool_item(
    data_array: xr.DataArray,
) -> bool:
    return _workspace_h5py_dataarray_can_write(data_array) and (
        _workspace_h5py_dataarray_is_tool_reference(data_array)
        or _workspace_h5py_dataarray_is_tool_blob(data_array)
    )


def _workspace_h5py_dataset_independent_tool_variable(
    dataset: typing.Any,
    variable_name: str,
    *,
    exclude_attrs: Iterable[typing.Hashable],
) -> xr.Variable | None:
    if not _workspace_h5py_dataset_storage_supported(dataset):
        return None
    attrs = _h5py_attrs_to_dict(dataset.attrs, exclude=exclude_attrs)
    if _TOOL_DATA_BLOB_NAME_ATTR in attrs:
        dims = (_workspace_h5py_tool_data_blob_dim(variable_name),)
    elif dataset.ndim == 1 and dataset.size == 0:
        dims = (_SAVED_TOOL_DATA_REFERENCE_DIM,)
    else:
        return None
    return xr.Variable(dims, _workspace_h5py_read_values(dataset), attrs)


def _workspace_h5py_extra_tool_data_names(
    ds: xr.Dataset, data_name: typing.Hashable
) -> frozenset[typing.Hashable]:
    if data_name not in {
        _serialization.ITOOL_DATA_NAME,
        _serialization.SAVED_TOOL_DATA_NAME,
    }:
        return frozenset()
    return frozenset(
        name
        for name, data_array in ds.data_vars.items()
        if name != data_name
        and _workspace_h5py_dataarray_is_independent_tool_item(data_array)
    )


def _workspace_h5py_dataset_has_only_independent_tool_items(
    ds: xr.Dataset, data_name: typing.Hashable
) -> bool:
    return (
        data_name == _serialization.SAVED_TOOL_DATA_NAME
        and not ds.coords
        and all(
            _workspace_h5py_dataarray_is_independent_tool_item(data_array)
            for data_array in ds.data_vars.values()
        )
    )


def _read_workspace_dataset_group_h5py(
    fname: str | os.PathLike[str],
    group_path: str,
    *,
    preferred_data_name: str | None = None,
) -> xr.Dataset | None:
    ensure_workspace_hdf5_filters_registered()
    group_path = group_path.strip("/")
    internal_attrs = (
        "CLASS",
        "DIMENSION_LIST",
        "NAME",
        "REFERENCE_LIST",
        "_Netcdf4Coordinates",
        "_Netcdf4Dimid",
        "coordinates",
    )
    with _workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        if group_path not in h5_file or not isinstance(h5_file[group_path], h5py.Group):
            return None
        group = h5_file[group_path]
        datasets: dict[str, h5py.Dataset] = {}
        for name, obj in group.items():
            if not isinstance(
                obj, h5py.Dataset
            ) or not _workspace_h5py_dataset_storage_supported(obj):
                continue
            marker = obj.attrs.get("CLASS")
            if isinstance(marker, bytes):
                marker = marker.decode()
            if marker == "DIMENSION_SCALE":
                continue
            datasets[name] = obj
        if preferred_data_name is not None and preferred_data_name in datasets:
            data_name = preferred_data_name
        elif len(datasets) == 1:
            data_name = next(iter(datasets))
        else:
            return None

        if preferred_data_name == data_name == _serialization.SAVED_TOOL_DATA_NAME:
            independent_data_vars: dict[typing.Hashable, xr.Variable] = {}
            for variable_name, dataset in datasets.items():
                variable = _workspace_h5py_dataset_independent_tool_variable(
                    dataset,
                    variable_name,
                    exclude_attrs=internal_attrs,
                )
                if variable is None:
                    break
                independent_data_vars[variable_name] = variable
            else:
                if _serialization.SAVED_TOOL_DATA_NAME in independent_data_vars:
                    return xr.Dataset(
                        independent_data_vars,
                        attrs=_h5py_attrs_to_dict(group.attrs),
                    )

        def _dataset_dims(dataset: h5py.Dataset) -> tuple[str, ...] | None:
            dims: list[str] = []
            for axis, dim in enumerate(dataset.dims):
                dim_keys = list(dim.keys())
                if len(dim_keys) != 1:
                    return None
                dim_name = str(dim_keys[0])
                scale = dim[dim_name]
                if (
                    not isinstance(scale, h5py.Dataset)
                    or scale.ndim != 1
                    or scale.shape[0] != dataset.shape[axis]
                    or not _workspace_h5py_dataset_storage_supported(scale)
                ):
                    return None
                dims.append(dim_name)
            return tuple(dims)

        data_dataset = datasets[data_name]
        if data_dataset.dtype.kind not in "biufc":
            return None
        dims: list[str] = []
        coords: dict[str, typing.Any] = {}
        for axis, dim in enumerate(data_dataset.dims):
            dim_keys = list(dim.keys())
            if len(dim_keys) != 1:
                return None
            dim_name = str(dim_keys[0])
            scale = dim[dim_name]
            if (
                not isinstance(scale, h5py.Dataset)
                or scale.ndim != 1
                or scale.shape[0] != data_dataset.shape[axis]
                or not _workspace_h5py_dataset_storage_supported(scale)
            ):
                return None
            dims.append(dim_name)
            coord_variable = _workspace_h5py_dataset_variable(
                scale,
                (dim_name,),
                name=dim_name,
                exclude_attrs=internal_attrs,
            )
            if coord_variable is None:
                return None
            coords[dim_name] = coord_variable

        scalar_coord_names = data_dataset.attrs.get("coordinates", "")
        if isinstance(scalar_coord_names, bytes):
            scalar_coord_names = scalar_coord_names.decode()
        legacy_spaced_coord_names = tuple(
            name
            for name, dataset in datasets.items()
            if name != data_name
            and _serialization.coord_name_needs_private_storage(name)
            and _workspace_h5py_dataset_storage_supported(dataset)
        )
        if isinstance(scalar_coord_names, str):
            for coord_name in scalar_coord_names.split():
                if coord_name not in group or not isinstance(
                    group[coord_name], h5py.Dataset
                ):
                    if legacy_spaced_coord_names:
                        continue
                    return None
                coord_dataset = group[coord_name]
                if coord_dataset.ndim == 0:
                    coord_dims = ()
                else:
                    coord_dims = _dataset_dims(coord_dataset)
                    if coord_dims is None or not all(dim in dims for dim in coord_dims):
                        return None
                coord_variable = _workspace_h5py_dataset_variable(
                    coord_dataset,
                    coord_dims,
                    name=coord_name,
                    exclude_attrs=internal_attrs,
                )
                if coord_variable is None:
                    return None
                coords[coord_name] = coord_variable

        data_attrs = _h5py_attrs_to_dict(data_dataset.attrs, exclude=internal_attrs)
        data_values = np.asarray(data_dataset[()])
        data_vars: dict[typing.Hashable, typing.Any] = {
            data_name: (
                tuple(dims),
                data_values,
                data_attrs,
            )
        }

        private_records = _serialization.private_coord_records_from_attrs(data_attrs)
        for record in private_records or ():
            variable_name = record["variable_name"]
            if variable_name not in group or not isinstance(
                group[variable_name], h5py.Dataset
            ):
                return None
            coord_dataset = group[variable_name]
            coord_dims = tuple(record["dims"])
            if (
                coord_dataset.ndim != len(coord_dims)
                or not _workspace_h5py_dataset_storage_supported(coord_dataset)
                or not all(dim in dims for dim in coord_dims)
            ):
                return None
            coord_variable = _workspace_h5py_dataset_variable(
                coord_dataset,
                coord_dims,
                name=variable_name,
                exclude_attrs=internal_attrs,
            )
            if coord_variable is None:
                return None
            data_vars[variable_name] = coord_variable

        for variable_name in legacy_spaced_coord_names:
            if variable_name in data_vars:
                continue
            coord_dataset = datasets[variable_name]
            coord_dims = _dataset_dims(coord_dataset)
            if coord_dims is None or not all(dim in dims for dim in coord_dims):
                continue
            coord_variable = _workspace_h5py_dataset_variable(
                coord_dataset,
                coord_dims,
                name=variable_name,
                exclude_attrs=internal_attrs,
            )
            if coord_variable is None:
                continue
            data_vars[variable_name] = coord_variable

        if data_name in {
            _serialization.ITOOL_DATA_NAME,
            _serialization.SAVED_TOOL_DATA_NAME,
        }:
            for variable_name, dataset in datasets.items():
                if (
                    variable_name == data_name
                    or variable_name in data_vars
                    or variable_name in coords
                ):
                    continue
                variable = _workspace_h5py_dataset_independent_tool_variable(
                    dataset,
                    variable_name,
                    exclude_attrs=internal_attrs,
                )
                if variable is None:
                    if data_name == _serialization.SAVED_TOOL_DATA_NAME:
                        return None
                    continue
                data_vars[variable_name] = variable

        return _serialization.restore_private_coords(
            xr.Dataset(
                data_vars,
                coords=coords,
                attrs=_h5py_attrs_to_dict(group.attrs),
            ),
            data_name,
        )


def _workspace_h5py_data_name(ds: xr.Dataset) -> typing.Hashable | None:
    if _serialization.ITOOL_DATA_NAME in ds.data_vars:
        return _serialization.ITOOL_DATA_NAME
    if _serialization.SAVED_TOOL_DATA_NAME in ds.data_vars:
        return _serialization.SAVED_TOOL_DATA_NAME
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    return None


def _workspace_dataset_can_write_h5py(ds: xr.Dataset) -> bool:
    data_name = _workspace_h5py_data_name(ds)
    if data_name is None:
        return False
    if _workspace_h5py_dataset_has_only_independent_tool_items(ds, data_name):
        return True
    private_data_names = set(_serialization.private_coord_variable_names(ds, data_name))
    extra_data_names = _workspace_h5py_extra_tool_data_names(ds, data_name)
    if any(
        name != data_name
        and name not in private_data_names
        and name not in extra_data_names
        for name in ds.data_vars
    ):
        return False
    data_array = ds[data_name]
    if data_array.chunks is not None or data_array.dtype.kind not in "biufc":
        return False
    for private_name in private_data_names:
        if private_name not in ds.data_vars:
            return False
        private_data = ds[private_name]
        if (
            private_data.chunks is not None
            or not _workspace_h5py_coord_dims_fit(private_data, data_array)
            or not _workspace_h5py_dataarray_can_write(private_data)
        ):
            return False
    for dim in data_array.dims:
        coord = ds.coords.get(dim)
        if (
            coord is None
            or coord.dims != (dim,)
            or not _workspace_h5py_dataarray_can_write(coord)
        ):
            return False
    for name, coord in ds.coords.items():
        if name in data_array.dims:
            continue
        if (
            _serialization.coord_name_needs_private_storage(name)
            or not _workspace_h5py_coord_dims_fit(coord, data_array)
            or not _workspace_h5py_dataarray_can_write(coord)
        ):
            return False
    return True


def _write_workspace_independent_tool_items_h5py(
    group: typing.Any,
    ds: xr.Dataset,
    *,
    encoding: Mapping[typing.Hashable, Mapping[str, typing.Any]] | None = None,
) -> bool:
    for variable_name, data_array in ds.data_vars.items():
        dataset = _workspace_h5py_create_dataset(
            group,
            str(variable_name),
            data_array.variable,
            encoding=None if encoding is None else encoding.get(variable_name),
        )
        if dataset is None:
            return False
    return True


def _write_workspace_dataset_group_h5py(
    fname: str | os.PathLike[str],
    group_path: str,
    ds: xr.Dataset,
    *,
    encoding: Mapping[typing.Hashable, Mapping[str, typing.Any]] | None = None,
) -> bool:
    if not _workspace_dataset_can_write_h5py(ds):
        return False
    ds = _sanitize_workspace_attr_names(ds)
    data_name = _workspace_h5py_data_name(ds)
    if data_name is None:
        return False
    private_data_names = _serialization.private_coord_variable_names(ds, data_name)

    ensure_workspace_hdf5_filters_registered()
    group_path = group_path.strip("/")
    with h5py.File(fname, "a") as h5_file:
        if group_path in h5_file:
            del h5_file[group_path]
        parent = _ensure_h5_parent_group(h5_file, group_path)
        group_name = group_path.rsplit("/", maxsplit=1)[-1]
        group = parent.create_group(group_name)
        try:
            for key, value in ds.attrs.items():
                group.attrs[key] = value
            if encoding is None:
                encoding = workspace_dataset_encoding(ds)
            if _workspace_h5py_dataset_has_only_independent_tool_items(ds, data_name):
                if not _write_workspace_independent_tool_items_h5py(
                    group, ds, encoding=encoding
                ):
                    del parent[group_name]
                    return False
                return True
            data_array = ds[data_name]
            dim_scales = []
            dim_scales_by_name = {}
            for dim_id, dim in enumerate(data_array.dims):
                coord = ds.coords[dim]
                coord_dataset = _workspace_h5py_create_dataset(
                    group, str(dim), coord.variable
                )
                if coord_dataset is None:
                    del parent[group_name]
                    return False
                coord_dataset.make_scale(str(dim))
                coord_dataset.attrs["_Netcdf4Dimid"] = np.int32(dim_id)
                coord_dataset.attrs["_Netcdf4Coordinates"] = np.asarray(
                    [dim_id], dtype=np.int32
                )
                dim_scales.append(coord_dataset)
                dim_scales_by_name[dim] = coord_dataset

            data_encoding = encoding.get(data_name, {})
            data_dataset = group.create_dataset(
                str(data_name),
                data=np.asarray(data_array.data),
                **_workspace_h5py_create_kwargs(data_encoding),
            )
            for dim_index, scale in enumerate(dim_scales):
                data_dataset.dims[dim_index].attach_scale(scale)
            data_dataset.attrs["_Netcdf4Coordinates"] = np.arange(
                len(dim_scales), dtype=np.int32
            )
            for key, value in data_array.attrs.items():
                data_dataset.attrs[key] = value

            scalar_coord_names: list[str] = []
            for name, coord in ds.coords.items():
                if name in data_array.dims:
                    continue
                coord_dataset = _workspace_h5py_create_dataset(
                    group, str(name), coord.variable
                )
                if coord_dataset is None:
                    del parent[group_name]
                    return False
                coordinate_ids = []
                for coord_dim_index, coord_dim in enumerate(coord.dims):
                    scale = dim_scales_by_name[coord_dim]
                    coord_dataset.dims[coord_dim_index].attach_scale(scale)
                    coordinate_ids.append(data_array.dims.index(coord_dim))
                if coordinate_ids:
                    coord_dataset.attrs["_Netcdf4Coordinates"] = np.asarray(
                        coordinate_ids, dtype=np.int32
                    )
                    coord_dataset.attrs["_Netcdf4Dimid"] = np.int32(coordinate_ids[0])
                scalar_coord_names.append(str(name))
            if scalar_coord_names:
                existing_coordinates = data_dataset.attrs.get("coordinates")
                if isinstance(existing_coordinates, bytes):
                    existing_coordinates = existing_coordinates.decode()
                coordinates = " ".join(scalar_coord_names)
                if isinstance(existing_coordinates, str) and existing_coordinates:
                    coordinates = f"{existing_coordinates} {coordinates}"
                data_dataset.attrs["coordinates"] = coordinates

            for private_name in private_data_names:
                private_data = ds[private_name]
                private_dataset = _workspace_h5py_create_dataset(
                    group, str(private_name), private_data.variable
                )
                if private_dataset is None:
                    del parent[group_name]
                    return False
                private_coordinate_ids: list[int] = []
                for private_dim_index, private_dim in enumerate(private_data.dims):
                    scale = dim_scales_by_name[private_dim]
                    private_dataset.dims[private_dim_index].attach_scale(scale)
                    private_coordinate_ids.append(data_array.dims.index(private_dim))
                if private_coordinate_ids:
                    private_dataset.attrs["_Netcdf4Coordinates"] = np.asarray(
                        private_coordinate_ids, dtype=np.int32
                    )
                    private_dataset.attrs["_Netcdf4Dimid"] = np.int32(
                        private_coordinate_ids[0]
                    )
            for extra_name in _workspace_h5py_extra_tool_data_names(ds, data_name):
                extra_data = ds[extra_name]
                extra_dataset = _workspace_h5py_create_dataset(
                    group,
                    str(extra_name),
                    extra_data.variable,
                    encoding=encoding.get(extra_name),
                )
                if extra_dataset is None:
                    del parent[group_name]
                    return False
        except Exception:
            del parent[group_name]
            return False
    return True


def _write_workspace_dataset_group_to_file(
    fname: str | os.PathLike[str],
    group_path: str,
    ds: xr.Dataset,
    *,
    lock_path: str | os.PathLike[str] | None = None,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    encoding = workspace_dataset_encoding(ds, compression_mode=compression_mode)
    if lock_path is not None:
        normalized_lock_path = _normalized_file_path(lock_path)
        if normalized_lock_path is not None and any(
            normalized_lock_path in dataarray_source_paths(data_array)
            for data_array in (*ds.data_vars.values(), *ds.coords.values())
        ):
            ds = ds.load()

    data_name = _workspace_h5py_data_name(ds)
    if data_name is not None:
        ds = _serialization.encode_private_coords(ds, data_name)
    ds = _sanitize_workspace_attr_names(ds)
    stale_encoding_keys = {
        "chunksizes",
        "compression",
        "compression_opts",
        "contiguous",
        "fletcher32",
        "original_shape",
        "preferred_chunks",
        "shuffle",
        "source",
        _WORKSPACE_MATERIALIZED_CHUNKSIZES,
    }
    for variable in ds.variables.values():
        for key in stale_encoding_keys:
            variable.encoding.pop(key, None)

    maybe_lock = (
        _workspace_file_lock(lock_path)
        if lock_path is not None
        else contextlib.nullcontext()
    )
    with maybe_lock:
        if _write_workspace_dataset_group_h5py(
            fname, group_path, ds, encoding=encoding
        ):
            return
        ds.to_netcdf(
            fname,
            mode="a",
            engine="h5netcdf",
            group=f"/{group_path.strip('/')}",
            invalid_netcdf=True,
            encoding=encoding,
        )
