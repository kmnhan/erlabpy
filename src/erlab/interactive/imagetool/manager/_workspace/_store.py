"""Document lifetime and HDF5 access for ImageTool Manager workspaces."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import pathlib
import shutil
import threading
import time
import typing
import uuid
import weakref
from dataclasses import dataclass
from typing import Self

import erlab
from erlab.interactive.imagetool.manager._workspace._format import (
    _current_workspace_schema_version,
    _workspace_schema_uses_immutable_generations,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import h5py
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")


_WORKSPACE_OBJECTS_GROUP = "__itws_objects"
_WORKSPACE_STAGING_GROUP = "__itws_staging"
_WORKSPACE_GENERATIONS_GROUP = "__itws_generations"
_WORKSPACE_MANIFEST_DATASET = "manifest"
_WORKSPACE_GENERATION_WIDTH = 20
_WORKSPACE_ID_ATTR = "imagetool_workspace_id"
_FILE_ACCESS_RETRY_DELAYS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)
_HDF5_CONTENTION_RETRY_DELAYS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)
_RETRYABLE_FILE_ACCESS_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EAGAIN,
        errno.EBUSY,
        errno.EPERM,
        getattr(errno, "ETXTBSY", errno.EBUSY),
    }
)
_HDF5_CONTENTION_ERRNOS = frozenset(
    {
        errno.EAGAIN,
        errno.EBUSY,
        getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
        getattr(errno, "ETXTBSY", errno.EBUSY),
    }
)
_HDF5_LOCK_UNAVAILABLE_ERRNOS = frozenset(
    {
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.ENOSYS),
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }
)


def _is_retryable_file_access_error(exc: OSError) -> bool:
    """Return whether a file access failure can be temporary."""
    return exc.errno in _RETRYABLE_FILE_ACCESS_ERRNOS


def _retry_file_access(
    operation: Callable[[], typing.Any],
    *,
    on_wait: Callable[[], None] | None = None,
) -> typing.Any:
    """Run one file operation with bounded retries for temporary access errors."""
    retry_delays = iter(_FILE_ACCESS_RETRY_DELAYS)
    waiting = False
    while True:
        try:
            return operation()
        except OSError as exc:
            if not _is_retryable_file_access_error(exc):
                raise
            if not waiting:
                waiting = True
                if on_wait is not None:
                    on_wait()
            try:
                delay = next(retry_delays)
            except StopIteration:
                raise exc from None
            time.sleep(delay)


def _is_hdf5_file_contention_error(exc: OSError) -> bool:
    """Return whether another HDF5 reader or writer blocked an open."""
    if exc.errno in _HDF5_CONTENTION_ERRNOS:
        return True
    if getattr(exc, "winerror", None) in {32, 33}:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "file is already open for read-only",
            "file is already open for write",
            "unable to lock file",
            "resource temporarily unavailable",
            "sharing violation",
            "lock violation",
        )
    )


def _is_hdf5_file_lock_unavailable_error(exc: OSError) -> bool:
    """Return whether the filesystem rejected required HDF5 locking."""
    if exc.errno in _HDF5_LOCK_UNAVAILABLE_ERRNOS:
        return True
    message = str(exc).lower()
    return "lock" in message and any(
        marker in message
        for marker in (
            "function not implemented",
            "not supported",
            "locking is disabled",
        )
    )


def _wait_for_hdf5_access(
    operation: Callable[[], typing.Any],
    *,
    on_wait: Callable[[], None] | None = None,
    before_attempt: Callable[[], None] | None = None,
    after_failed_attempt: Callable[[], None] | None = None,
) -> typing.Any:
    """Wait until a conflicting HDF5 reader or writer releases the file."""
    delay_index = 0
    while True:
        if before_attempt is not None:
            before_attempt()
        try:
            return operation()
        except OSError as exc:
            if after_failed_attempt is not None:
                after_failed_attempt()
            if _is_hdf5_file_lock_unavailable_error(exc):
                raise
            if not _is_hdf5_file_contention_error(exc):
                raise
            if on_wait is not None:
                on_wait()
            delay = _HDF5_CONTENTION_RETRY_DELAYS[
                min(delay_index, len(_HDF5_CONTENTION_RETRY_DELAYS) - 1)
            ]
            delay_index += 1
            time.sleep(delay)


@dataclass(frozen=True)
class _WorkspaceGeneration:
    """One validated committed workspace generation."""

    sequence: int
    manifest: dict[str, typing.Any]


@dataclass(frozen=True)
class _SerializedReaderPinSnapshot:
    """Versioned reader pins captured at one instant."""

    object_versions: dict[str, int]
    legacy_group_versions: dict[str, int]

    @property
    def object_ids(self) -> frozenset[str]:
        return frozenset(self.object_versions)

    @property
    def legacy_group_paths(self) -> frozenset[str]:
        return frozenset(self.legacy_group_versions)

    @property
    def empty(self) -> bool:
        return not self.object_versions and not self.legacy_group_versions


class WorkspaceStoreConflictError(RuntimeError):
    """The path no longer identifies the file opened by the store."""


class WorkspaceStoreReopenError(RuntimeError):
    """A replaced workspace could not be reopened."""


class WorkspaceReaderUnavailableError(RuntimeError):
    """A workspace reader no longer has a payload that it can export."""


class WorkspaceStore:
    """Own the lifetime and access policy for one workspace document.

    Payload objects are immutable. A generation becomes visible only after its
    completed staging group moves into the generations group. The store keeps a
    read-only handle while the document is idle. A bounded write session closes
    that handle, opens the document for writing, and closes the writable handle
    before it returns.
    """

    _active: weakref.WeakValueDictionary[str, WorkspaceStore] = (
        weakref.WeakValueDictionary()
    )
    _pending_readers: typing.ClassVar[dict[str, weakref.WeakSet[typing.Any]]] = {}
    _serialized_object_pins: typing.ClassVar[dict[str, dict[str, int]]] = {}
    _serialized_legacy_group_pins: typing.ClassVar[dict[str, dict[str, int]]] = {}
    _serialized_pin_version = 0
    _active_lock = threading.RLock()
    _fork_stores: tuple[WorkspaceStore, ...] = ()

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        create: bool = False,
        workspace_id: str | None = None,
    ) -> None:
        if not create and workspace_id is not None:
            raise ValueError("workspace_id is valid only when create is true")
        self._path = pathlib.Path(path).resolve()
        self._lock = threading.RLock()
        self._access_condition = threading.Condition(self._lock)
        self._write_lock = threading.RLock()
        self._object_leases: dict[str, int] = {}
        self._readers: weakref.WeakSet[typing.Any] = weakref.WeakSet()
        self._state: typing.Literal["open", "conflicted", "closed"] = "closed"
        self._handle_generation = 0
        self._h5_file: typing.Any = None
        self._write_depth = 0
        self._write_pending = False
        self._write_opening = False
        self._write_target_path: pathlib.Path | None = None
        self._locking_supported = True
        self._workspace_id: str | None = None
        self._path_identity: tuple[int, int] | None = None
        self._recovery_path: pathlib.Path | None = None
        try:
            self._close_pending_reader_caches()
            self._open(create=create, workspace_id=workspace_id)
            self._register()
        except Exception:
            self._close_handle()
            raise

    @staticmethod
    def _key(path: str | os.PathLike[str]) -> str:
        return os.path.normcase(str(pathlib.Path(path).resolve()))

    @classmethod
    def _serialization_key(
        cls,
        workspace_id: str | None,
        path: str | os.PathLike[str],
    ) -> str:
        return f"{cls._key(path)}\0{workspace_id or ''}"

    @classmethod
    def pin_serialized_reader(
        cls,
        *,
        workspace_id: str | None,
        path: str | os.PathLike[str],
        object_id: str | None,
        legacy_group_path: str | None,
    ) -> None:
        """Keep data reachable after a reader is sent to another process.

        Dask does not provide a reliable release callback for every client and
        scheduler. These small pins therefore remain until the user confirms
        workspace compaction. They do not block saves. They only defer reclamation
        of the referenced data.
        """
        key = cls._serialization_key(workspace_id, path)
        with cls._active_lock:
            cls._serialized_pin_version += 1
            version = cls._serialized_pin_version
            if object_id is not None:
                cls._serialized_object_pins.setdefault(key, {})[object_id] = version
            if legacy_group_path is not None:
                cls._serialized_legacy_group_pins.setdefault(key, {})[
                    legacy_group_path
                ] = version

    @classmethod
    def active(cls, path: str | os.PathLike[str]) -> WorkspaceStore | None:
        """Return the store that owns *path*, if this process has one."""
        with cls._active_lock:
            store = cls._active.get(cls._key(path))
        if store is None:
            return None
        with store.lock:
            return None if store.closed else store

    def _close_pending_reader_caches(self) -> None:
        """Release local caches before this store opens their shared path."""
        with self._active_lock:
            readers = tuple(self._pending_readers.get(self._key(self._path), ()))
        for reader in readers:
            reader.close()

    @classmethod
    def register_path_reader(
        cls,
        path: str | os.PathLike[str],
        reader: typing.Any,
    ) -> WorkspaceStore | None:
        """Register a reader and return the active store for its path."""
        key = cls._key(path)
        with cls._active_lock:
            store = cls._active.get(key)
            if store is None or store.closed:
                cls._pending_readers.setdefault(key, weakref.WeakSet()).add(reader)
                return None
        store.register_reader(reader)
        return store

    @classmethod
    def unregister_path_reader(
        cls,
        path: str | os.PathLike[str],
        reader: typing.Any,
    ) -> None:
        """Remove a reader that is not attached to an active store."""
        key = cls._key(path)
        with cls._active_lock:
            readers = cls._pending_readers.get(key)
            if readers is None:
                return
            readers.discard(reader)
            if not readers:
                cls._pending_readers.pop(key, None)

    @classmethod
    def _before_fork(cls) -> None:
        """Close inherited HDF5 handles before a process forks."""
        with cls._active_lock:
            stores = tuple(
                sorted(
                    (store for store in cls._active.values() if not store.closed),
                    key=lambda store: str(store.path),
                )
            )
        acquired: list[WorkspaceStore] = []
        try:
            for store in stores:
                store._write_lock.acquire()
                store._lock.acquire()
                acquired.append(store)
            for store in stores:
                store._release_handle()
        except Exception:
            for store in reversed(acquired):
                store._lock.release()
                store._write_lock.release()
            raise
        cls._fork_stores = stores

    @classmethod
    def _after_fork_parent(cls) -> None:
        for store in reversed(cls._fork_stores):
            store._lock.release()
            store._write_lock.release()
        cls._fork_stores = ()

    @classmethod
    def _after_fork_child(cls) -> None:
        for store in cls._fork_stores:
            store._h5_file = None
            store._state = "closed"
            store._lock = threading.RLock()
            store._access_condition = threading.Condition(store._lock)
            store._write_lock = threading.RLock()
            store._write_pending = False
            store._write_opening = False
        cls._active = weakref.WeakValueDictionary()
        cls._pending_readers = {}
        cls._active_lock = threading.RLock()
        cls._fork_stores = ()

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def write_lock(self) -> threading.RLock:
        """Serialize complete saves and maintenance operations."""
        return self._write_lock

    @property
    def locking_supported(self) -> bool:
        """Return whether required HDF5 locks work for this workspace path."""
        return self._locking_supported

    @property
    def write_in_progress(self) -> bool:
        """Return whether this store is waiting to write or is writing."""
        return self._write_pending or self._write_depth > 0

    @property
    def closed(self) -> bool:
        """Return whether this store released ownership of its path."""
        return self._state == "closed"

    @property
    def conflicted(self) -> bool:
        """Return whether the workspace path changed outside this store."""
        return self._state == "conflicted"

    @property
    def h5_file(self) -> typing.Any:
        """Return the current document handle.

        The handle is read-only outside :meth:`write_session`.
        """
        if self.conflicted:
            raise WorkspaceStoreConflictError(
                f"Workspace store no longer owns its path: {self._path}"
            )
        if self.closed:
            raise RuntimeError("Workspace store is closed")
        return self._ensure_read_handle()

    @property
    def read_h5_file(self) -> typing.Any:
        """Return the readable session handle, including during recovery."""
        if self.closed:
            raise RuntimeError("Workspace store has no readable handle")
        return self._ensure_read_handle()

    @property
    def workspace_id(self) -> str | None:
        """Return the stored identity, or ``None`` before first publication."""
        return self._workspace_id

    @property
    def handle_generation(self) -> int:
        """Return a value that changes each time the HDF5 handle reopens."""
        return self._handle_generation

    @property
    def recovery_path(self) -> pathlib.Path | None:
        """Return the temporary document retained for read-only recovery."""
        with self._lock:
            return self._recovery_path

    def _register(self) -> None:
        key = self._key(self._path)
        with self._active_lock:
            existing = self._active.get(key)
            if existing is not None and existing is not self:
                self._close_handle()
                raise RuntimeError(
                    f"Workspace already has an active store: {self._path}"
                )
            self._active[key] = self
            pending_readers = tuple(self._pending_readers.pop(key, ()))
        for reader in pending_readers:
            reader._attach_store(self)

    def _unregister(self, path: pathlib.Path | None = None) -> None:
        key = self._key(self._path if path is None else path)
        with self._active_lock:
            if self._active.get(key) is self:
                self._active.pop(key, None)

    @staticmethod
    def _ensure_workspace_id(h5_file: typing.Any, preferred: str | None = None) -> str:
        workspace_id = preferred or h5_file.attrs.get(_WORKSPACE_ID_ATTR)
        if isinstance(workspace_id, bytes):
            workspace_id = workspace_id.decode()
        if not isinstance(workspace_id, str) or not workspace_id:
            workspace_id = uuid.uuid4().hex
        if h5_file.attrs.get(_WORKSPACE_ID_ATTR) != workspace_id:
            h5_file.attrs[_WORKSPACE_ID_ATTR] = workspace_id
            h5_file.flush()
        return workspace_id

    @staticmethod
    def _workspace_id_from_file(h5_file: typing.Any) -> str | None:
        workspace_id = h5_file.attrs.get(_WORKSPACE_ID_ATTR)
        if isinstance(workspace_id, bytes):
            workspace_id = workspace_id.decode()
        if isinstance(workspace_id, str) and workspace_id:
            return workspace_id
        return None

    def _open_with_lock_detection(
        self,
        mode: str,
        **kwargs: typing.Any,
    ) -> typing.Any:
        try:
            return _wait_for_hdf5_access(
                lambda: h5py.File(self._path, mode, locking=True, **kwargs)
            )
        except OSError as exc:
            if not _is_hdf5_file_lock_unavailable_error(exc):
                raise
        self._locking_supported = False
        return h5py.File(self._path, mode, locking=False, **kwargs)

    def _open(self, *, create: bool, workspace_id: str | None = None) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._locking_supported = True
        if create:
            h5_file = self._open_with_lock_detection(
                "w",
                libver="latest",
                fs_strategy="fsm",
                fs_persist=True,
            )
            self._workspace_id = self._ensure_workspace_id(h5_file, workspace_id)
            h5_file.attrs["imagetool_workspace_schema_version"] = (
                _current_workspace_schema_version()
            )
            h5_file.attrs["erlab_version"] = str(erlab.__version__)
            h5_file.require_group(_WORKSPACE_OBJECTS_GROUP)
            h5_file.require_group(_WORKSPACE_STAGING_GROUP)
            h5_file.require_group(_WORKSPACE_GENERATIONS_GROUP)
            h5_file.flush()
            h5_file.close()
        else:
            with self._open_with_lock_detection("r") as h5_file:
                self._workspace_id = self._workspace_id_from_file(h5_file)
        self._h5_file = None
        path_stat = self._path.stat()
        self._path_identity = (path_stat.st_dev, path_stat.st_ino)
        self._recovery_path = None
        self._state = "open"

    def _ensure_read_handle(self) -> typing.Any:
        if self._h5_file is None:
            source = self._recovery_path or self._path
            self._h5_file = _wait_for_hdf5_access(
                lambda: h5py.File(
                    source,
                    "r",
                    locking="best-effort",
                )
            )
            self._handle_generation += 1
        return self._h5_file

    def _release_handle(self) -> None:
        """Release the current HDF5 handle without closing the store."""
        for reader in tuple(self._readers):
            with contextlib.suppress(Exception):
                reader._close_store_wrapper()
        h5_file = self._h5_file
        self._h5_file = None
        if h5_file is not None:
            with contextlib.suppress(Exception):
                h5_file.close()
        self._handle_generation += 1

    def _prepare_copy_on_write(self) -> pathlib.Path:
        target = self._path.with_name(f".{self._path.name}.write-{uuid.uuid4().hex}")
        shutil.copy2(self._path, target)
        return target

    def _publish_copy_on_write(
        self,
        source: pathlib.Path,
        expected_state: tuple[int, int, int, int],
        *,
        on_contention: Callable[[], None] | None,
    ) -> None:
        def _replace() -> None:
            self._require_path_state(expected_state)
            os.replace(source, self._path)

        _retry_file_access(_replace, on_wait=on_contention)
        path_stat = self._path.stat()
        self._path_identity = (path_stat.st_dev, path_stat.st_ino)
        if os.name == "posix":
            with contextlib.suppress(OSError):
                file_descriptor = os.open(self._path.parent, os.O_RDONLY)
                try:
                    os.fsync(file_descriptor)
                finally:
                    os.close(file_descriptor)

    @contextlib.contextmanager
    def read_session(self) -> typing.Iterator[typing.Any]:
        """Yield the shared handle when no writer is changing its open mode."""
        with self._access_condition:
            while self._write_opening:
                self._access_condition.wait()
            if self.closed:
                raise RuntimeError("Workspace store is closed")
            yield self.read_h5_file

    @contextlib.contextmanager
    def write_session(
        self,
        *,
        on_contention: Callable[[], None] | None = None,
    ) -> typing.Iterator[typing.Any]:
        """Yield the only writable handle for one bounded document operation."""
        with self._write_lock:
            copy_on_write_path: pathlib.Path | None = None
            expected_path_state: tuple[int, int, int, int] | None = None
            succeeded = False
            with self._access_condition:
                self.require_current_path()
                outermost = self._write_depth == 0
                if outermost:
                    self._write_pending = True
                    path_identity = self._path_identity
                    if not self._locking_supported and path_identity is None:
                        self._write_pending = False
                        self._access_condition.notify_all()
                        raise RuntimeError("Workspace store has no path identity")
                else:
                    self._write_depth += 1

            if outermost:

                def _begin_open_attempt() -> None:
                    with self._access_condition:
                        self._release_handle()
                        self._write_opening = True

                def _end_failed_open_attempt() -> None:
                    with self._access_condition:
                        self._write_opening = False
                        self._access_condition.notify_all()

                try:
                    if self._locking_supported:
                        h5_file = _wait_for_hdf5_access(
                            lambda: h5py.File(
                                self._path,
                                "r+",
                                locking="best-effort",
                            ),
                            on_wait=on_contention,
                            before_attempt=_begin_open_attempt,
                            after_failed_attempt=_end_failed_open_attempt,
                        )
                    else:
                        with self._access_condition:
                            self._release_handle()
                            self._write_opening = True
                        path_stat = self._path.stat()
                        expected_path_state = (
                            path_stat.st_dev,
                            path_stat.st_ino,
                            path_stat.st_size,
                            path_stat.st_mtime_ns,
                        )
                        copy_on_write_path = self._prepare_copy_on_write()
                        h5_file = h5py.File(
                            copy_on_write_path,
                            "r+",
                            locking=False,
                        )
                except BaseException:
                    with self._access_condition:
                        self._write_target_path = None
                        self._write_pending = False
                        self._write_opening = False
                        self._access_condition.notify_all()
                    if copy_on_write_path is not None:
                        with contextlib.suppress(OSError):
                            copy_on_write_path.unlink()
                    raise
                with self._access_condition:
                    self._h5_file = h5_file
                    self._write_target_path = copy_on_write_path
                    self._handle_generation += 1
                    self._write_depth = 1
                    self._write_opening = False
                    self._access_condition.notify_all()
            try:
                yield self._h5_file
                succeeded = True
            finally:
                if not outermost:
                    with self._lock:
                        self._write_depth -= 1
                else:
                    try:
                        with self._lock:
                            self._write_depth = 0
                            try:
                                if succeeded and copy_on_write_path is not None:
                                    self.flush(durable=True)
                            finally:
                                self._release_handle()
                                self._write_target_path = None
                        if (
                            succeeded
                            and copy_on_write_path is not None
                            and expected_path_state is not None
                        ):
                            try:
                                self._publish_copy_on_write(
                                    copy_on_write_path,
                                    expected_path_state,
                                    on_contention=on_contention,
                                )
                            except WorkspaceStoreConflictError:
                                try:
                                    self._use_recovery_source(copy_on_write_path)
                                except Exception:
                                    self._mark_conflicted()
                                else:
                                    copy_on_write_path = None
                                raise
                    finally:
                        with self._access_condition:
                            self._write_pending = False
                            self._write_opening = False
                            self._access_condition.notify_all()
                        if copy_on_write_path is not None:
                            with contextlib.suppress(OSError):
                                copy_on_write_path.unlink()

    def _close_handle(self) -> None:
        self._release_handle()
        recovery_path = self._recovery_path
        self._recovery_path = None
        if recovery_path is not None:
            with contextlib.suppress(OSError):
                recovery_path.unlink()
        self._state = "closed"

    def _mark_conflicted(self) -> None:
        """Quarantine writes while retaining readable session data."""
        self._state = "conflicted"

    def _use_recovery_source(self, path: pathlib.Path) -> None:
        """Use a prepared document as the read-only source after a conflict."""
        self._close_handle()
        try:
            self._h5_file = h5py.File(
                path,
                "r",
                locking="best-effort",
            )
        except Exception:
            self._h5_file = None
            self._state = "conflicted"
            raise
        self._handle_generation += 1
        self._path_identity = None
        self._recovery_path = path
        self._state = "conflicted"

    def _require_path_identity(self, expected: tuple[int, int]) -> None:
        try:
            path_stat = self._path.stat()
        except OSError as exc:
            raise WorkspaceStoreConflictError(
                f"Open workspace path is no longer available: {self._path}"
            ) from exc
        if (path_stat.st_dev, path_stat.st_ino) != expected:
            raise WorkspaceStoreConflictError(
                f"Open workspace path now identifies another file: {self._path}"
            )

    def _require_path_state(self, expected: tuple[int, int, int, int]) -> None:
        """Fail if an unlocked copy-on-write source changed before publication."""
        try:
            path_stat = self._path.stat()
        except OSError as exc:
            raise WorkspaceStoreConflictError(
                f"Open workspace path is no longer available: {self._path}"
            ) from exc
        current = (
            path_stat.st_dev,
            path_stat.st_ino,
            path_stat.st_size,
            path_stat.st_mtime_ns,
        )
        if current != expected:
            raise WorkspaceStoreConflictError(
                f"Open workspace changed during save: {self._path}"
            )

    def require_current_path(self) -> None:
        """Fail if the open handle is no longer the file at ``path``."""
        if self.conflicted:
            raise WorkspaceStoreConflictError(
                f"Workspace store no longer owns its path: {self._path}"
            )
        if self.closed:
            raise RuntimeError("Workspace store is closed")
        path_identity = self._path_identity
        if path_identity is None:
            raise RuntimeError("Workspace store has no path identity")
        try:
            self._require_path_identity(path_identity)
        except WorkspaceStoreConflictError:
            self._mark_conflicted()
            raise

    def close(self) -> None:
        """Close the document handle and remove this store from the registry."""
        with self._write_lock, self._lock:
            self._unregister()
            self._close_handle()

    def switch_path(self, path: str | os.PathLike[str]) -> None:
        """Open *path* through this store without invalidating store references."""
        new_path = pathlib.Path(path).resolve()
        locking_supported = True
        try:
            new_h5_file = _wait_for_hdf5_access(
                lambda: h5py.File(new_path, "r", locking=True)
            )
        except OSError as exc:
            if not _is_hdf5_file_lock_unavailable_error(exc):
                raise
            locking_supported = False
            new_h5_file = h5py.File(new_path, "r", locking=False)
        with self._write_lock, self._lock, new_h5_file:
            new_stat = new_path.stat()
            new_identity = (new_stat.st_dev, new_stat.st_ino)
            workspace_id = self._workspace_id_from_file(new_h5_file)
            if workspace_id is None:
                raise ValueError("Workspace file has no stable identity")
            with self._active_lock:
                existing = self._active.get(self._key(new_path))
                if existing is not None and existing is not self:
                    raise RuntimeError(
                        f"Workspace already has an active store: {new_path}"
                    )
                old_path = self._path
                self._unregister(old_path)
                self._close_handle()
                self._path = new_path
                self._workspace_id = workspace_id
                self._locking_supported = locking_supported
                self._path_identity = new_identity
                self._recovery_path = None
                self._state = "open"
                self._active[self._key(new_path)] = self

    def reopen(self) -> None:
        """Close and reopen the current file through the same store object."""
        with self._write_lock, self._lock:
            if self.conflicted:
                raise WorkspaceStoreConflictError(
                    f"Workspace store no longer owns its path: {self._path}"
                )
            self._close_handle()
            self._open(create=False)

    def _reopen_after_file_operation(self) -> None:
        """Reopen the document after a file operation released its handle."""

        def _reopen() -> None:
            try:
                self._open(create=False)
            except Exception:
                self._close_handle()
                raise

        _retry_file_access(_reopen)

    def replace_from(
        self,
        prepared_path: str | os.PathLike[str],
        replace: Callable[[pathlib.Path, pathlib.Path], None],
        *,
        before_close: Callable[[], None] | None = None,
    ) -> None:
        """Replace this document while its HDF5 handle is closed.

        The store object stays stable, so existing lazy arrays use the reopened
        file after the replacement.
        """
        source = pathlib.Path(prepared_path).resolve()
        with self._write_lock, self._lock:
            if self.conflicted:
                raise WorkspaceStoreConflictError(
                    f"Workspace store no longer owns its path: {self._path}"
                )
            if self.closed:
                raise RuntimeError("Workspace store is closed")
            if before_close is not None:
                try:
                    before_close()
                except WorkspaceStoreConflictError:
                    self._mark_conflicted()
                    raise
            path_identity = self._path_identity
            if path_identity is None:
                raise RuntimeError("Workspace store has no path identity")
            self._release_handle()
            try:
                self._require_path_identity(path_identity)
                replace(source, self._path)
            except WorkspaceStoreConflictError:
                try:
                    self._use_recovery_source(source)
                except Exception:
                    self._mark_conflicted()
                raise
            except Exception as exc:
                try:
                    self._require_path_identity(path_identity)
                except WorkspaceStoreConflictError as conflict:
                    self._mark_conflicted()
                    raise conflict from exc
                try:
                    self._reopen_after_file_operation()
                except Exception:
                    try:
                        self._use_recovery_source(source)
                    except Exception:
                        self._mark_conflicted()
                raise
            else:
                try:
                    self._reopen_after_file_operation()
                except Exception as exc:
                    raise WorkspaceStoreReopenError(
                        "Workspace was replaced but could not be reopened: "
                        f"{self._path}"
                    ) from exc

    def acquire_object(self, object_id: str) -> None:
        """Keep a payload object reachable while a lazy array uses it."""
        with self._lock:
            self._object_leases[object_id] = self._object_leases.get(object_id, 0) + 1

    def register_reader(self, reader: object) -> None:
        """Register a lazy reader that must refresh when this store reopens."""
        with self._lock:
            self._readers.add(reader)

    def pin_serialized_reader_reference(
        self,
        *,
        object_id: str | None,
        legacy_group_path: str | None,
    ) -> None:
        """Pin one exported reader without racing workspace compaction."""
        with self._lock:
            if self.closed or self.conflicted:
                raise WorkspaceReaderUnavailableError(
                    "The workspace changed before its Dask reader was exported"
                )
            with self.read_session() as h5_file:
                if (
                    object_id is not None
                    and self.object_path(object_id).strip("/") not in h5_file
                ):
                    raise WorkspaceReaderUnavailableError(
                        "The workspace payload was compacted before its Dask reader "
                        "was exported"
                    )
                if (
                    legacy_group_path is not None
                    and legacy_group_path.strip("/") not in h5_file
                ):
                    raise WorkspaceReaderUnavailableError(
                        "The workspace payload was compacted before its Dask reader "
                        "was exported"
                    )
            self.pin_serialized_reader(
                workspace_id=self.workspace_id,
                path=self.path,
                object_id=object_id,
                legacy_group_path=legacy_group_path,
            )

    def unregister_reader(self, reader: object) -> None:
        """Remove a reader that no longer uses this store."""
        with self._lock:
            self._readers.discard(reader)

    def release_object(self, object_id: str) -> None:
        """Release one lazy-array reference to a payload object."""
        with self._lock:
            count = self._object_leases.get(object_id, 0)
            if count <= 1:
                self._object_leases.pop(object_id, None)
            else:
                self._object_leases[object_id] = count - 1

    @property
    def leased_object_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._object_leases)

    def _serialization_keys(self) -> tuple[str, ...]:
        path_key = self._serialization_key(None, self._path)
        if self._workspace_id is None:
            return (path_key,)
        return (self._serialization_key(self._workspace_id, self._path), path_key)

    @property
    def serialized_object_ids(self) -> frozenset[str]:
        """Return objects pinned by readers serialized in this process."""
        return self.serialized_reader_pin_snapshot().object_ids

    @property
    def serialized_legacy_group_paths(self) -> frozenset[str]:
        """Return old-format groups pinned by serialized readers."""
        return self.serialized_reader_pin_snapshot().legacy_group_paths

    def serialized_reader_pin_snapshot(self) -> _SerializedReaderPinSnapshot:
        """Return the newest pin version for each exported reader target."""
        with self._active_lock:
            object_versions: dict[str, int] = {}
            legacy_group_versions: dict[str, int] = {}
            for key in self._serialization_keys():
                for object_id, version in self._serialized_object_pins.get(
                    key, {}
                ).items():
                    object_versions[object_id] = max(
                        version, object_versions.get(object_id, -1)
                    )
                for group_path, version in self._serialized_legacy_group_pins.get(
                    key, {}
                ).items():
                    legacy_group_versions[group_path] = max(
                        version, legacy_group_versions.get(group_path, -1)
                    )
            return _SerializedReaderPinSnapshot(
                object_versions=object_versions,
                legacy_group_versions=legacy_group_versions,
            )

    @property
    def has_serialized_readers(self) -> bool:
        """Return whether this workspace has data pinned after serialization."""
        with self._active_lock:
            return any(
                self._serialized_object_pins.get(key)
                or self._serialized_legacy_group_pins.get(key)
                for key in self._serialization_keys()
            )

    def clear_serialized_reader_pins(self) -> None:
        """Release data pinned when this workspace's readers were serialized."""
        with self._active_lock:
            for key in self._serialization_keys():
                self._serialized_object_pins.pop(key, None)
                self._serialized_legacy_group_pins.pop(key, None)

    def release_serialized_reader_pins(
        self,
        snapshot: _SerializedReaderPinSnapshot,
    ) -> None:
        """Release a confirmed snapshot of serialized reader pins."""
        with self._active_lock:
            for key in self._serialization_keys():
                pinned_objects = self._serialized_object_pins.get(key)
                if pinned_objects is not None:
                    for (
                        object_id,
                        confirmed_version,
                    ) in snapshot.object_versions.items():
                        current_version = pinned_objects.get(object_id)
                        if (
                            current_version is not None
                            and current_version <= confirmed_version
                        ):
                            pinned_objects.pop(object_id, None)
                    if not pinned_objects:
                        self._serialized_object_pins.pop(key, None)
                pinned_groups = self._serialized_legacy_group_pins.get(key)
                if pinned_groups is not None:
                    for (
                        group_path,
                        confirmed_version,
                    ) in snapshot.legacy_group_versions.items():
                        current_version = pinned_groups.get(group_path)
                        if (
                            current_version is not None
                            and current_version <= confirmed_version
                        ):
                            pinned_groups.pop(group_path, None)
                    if not pinned_groups:
                        self._serialized_legacy_group_pins.pop(key, None)

    @property
    def leased_legacy_group_paths(self) -> frozenset[str]:
        """Return old-format payload groups that still have lazy readers."""
        with self.read_session() as h5_file:
            paths = {
                path
                for reader in self._readers
                if (path := reader.legacy_group_path) not in {None, "/"}
            }
            return frozenset(
                path
                for path in paths
                if path.strip("/") in h5_file
                and isinstance(h5_file[path], h5py.Group)
                and any(
                    isinstance(item, h5py.Dataset) for item in h5_file[path].values()
                )
            )

    @staticmethod
    def object_path(object_id: str) -> str:
        if not object_id or "/" in object_id:
            raise ValueError("Workspace object ID must be one path component")
        return f"/{_WORKSPACE_OBJECTS_GROUP}/{object_id}"

    @staticmethod
    def _manifest_text(manifest: Mapping[str, typing.Any]) -> str:
        return json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _write_manifest(
        cls,
        group: typing.Any,
        manifest: Mapping[str, typing.Any],
    ) -> None:
        text = cls._manifest_text(manifest)
        dataset = group.create_dataset(
            _WORKSPACE_MANIFEST_DATASET,
            data=text,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        dataset.attrs["sha256"] = hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def _read_manifest(cls, group: typing.Any) -> dict[str, typing.Any]:
        if _WORKSPACE_MANIFEST_DATASET not in group:
            raise ValueError("Workspace generation has no manifest")
        dataset = group[_WORKSPACE_MANIFEST_DATASET]
        raw = dataset.asstr()[()]
        if not isinstance(raw, str):
            raise TypeError("Workspace generation manifest is not text")
        expected = dataset.attrs.get("sha256")
        if isinstance(expected, bytes):
            expected = expected.decode()
        actual = hashlib.sha256(raw.encode()).hexdigest()
        if expected != actual:
            raise ValueError("Workspace generation manifest checksum does not match")
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise TypeError("Workspace generation manifest is not an object")
        if not _workspace_schema_uses_immutable_generations(
            int(manifest.get("schema_version", 0))
        ):
            raise ValueError("Workspace generation has an unsupported schema")
        nodes = manifest.get("nodes")
        if not isinstance(nodes, list):
            raise TypeError("Workspace generation nodes are not a list")
        for entry in nodes:
            if not isinstance(entry, dict):
                raise TypeError("Workspace generation node is not an object")
            object_id = entry.get("payload_object_id")
            if not isinstance(object_id, str):
                raise TypeError("Workspace generation node has no payload object")
            object_path = cls.object_path(object_id)
            if entry.get("payload_path") != object_path:
                raise ValueError("Workspace generation payload path is not canonical")
            if object_path.strip("/") not in group.file:
                raise ValueError("Workspace generation payload object is missing")
        return manifest

    def generations(self) -> tuple[_WorkspaceGeneration, ...]:
        """Return all valid committed generations in ascending order."""
        with self.read_session() as h5_file:
            root = h5_file.get(_WORKSPACE_GENERATIONS_GROUP)
            if root is None:
                return ()
            generations: list[_WorkspaceGeneration] = []
            for name in sorted(root):
                if len(name) != _WORKSPACE_GENERATION_WIDTH or not name.isdigit():
                    continue
                with contextlib.suppress(Exception):
                    generations.append(
                        _WorkspaceGeneration(int(name), self._read_manifest(root[name]))
                    )
            return tuple(generations)

    def current_generation(self) -> _WorkspaceGeneration:
        """Return the newest valid committed generation."""
        generations = self.generations()
        if not generations:
            raise ValueError("Workspace has no committed generation")
        return generations[-1]

    def publish(self, manifest: Mapping[str, typing.Any]) -> _WorkspaceGeneration:
        """Publish one completed manifest as the newest generation."""
        with self.write_session():
            if self._workspace_id is None:
                self._workspace_id = self._ensure_workspace_id(self.h5_file)
            generation_root = self.h5_file.require_group(_WORKSPACE_GENERATIONS_GROUP)
            existing_sequences = [
                int(name)
                for name in generation_root
                if len(name) == _WORKSPACE_GENERATION_WIDTH and name.isdigit()
            ]
            sequence = 1 if not existing_sequences else max(existing_sequences) + 1
            staging_root = self.h5_file.require_group(_WORKSPACE_STAGING_GROUP)
            staging_name = uuid.uuid4().hex
            staging = staging_root.create_group(staging_name)
            generation_manifest = dict(manifest)
            generation_manifest["schema_version"] = _current_workspace_schema_version()
            generation_manifest.pop("generation", None)
            try:
                self._write_manifest(staging, generation_manifest)
                generation_manifest = self._read_manifest(staging)
                self.flush()
            except Exception:
                with contextlib.suppress(Exception):
                    del staging_root[staging_name]
                    self.flush()
                raise

            generation_name = f"{sequence:0{_WORKSPACE_GENERATION_WIDTH}d}"
            self.h5_file.move(
                f"/{_WORKSPACE_STAGING_GROUP}/{staging_name}",
                f"/{_WORKSPACE_GENERATIONS_GROUP}/{generation_name}",
            )
            self.h5_file.attrs["imagetool_workspace_schema_version"] = (
                _current_workspace_schema_version()
            )
            self.h5_file.attrs["erlab_version"] = str(erlab.__version__)
            self.flush(durable=True)
            return _WorkspaceGeneration(sequence, generation_manifest)

    def flush(self, *, durable: bool = False) -> None:
        """Flush HDF5 buffers and optionally ask the operating system to sync."""
        self.h5_file.flush()
        if not durable:
            return
        with contextlib.suppress(Exception):
            handle = self.h5_file.id.get_vfd_handle()
            if isinstance(handle, tuple):
                handle = handle[0]
            os.fsync(int(handle))

    @staticmethod
    def manifest_node_object_ids(manifest: Mapping[str, typing.Any]) -> frozenset[str]:
        """Return data object IDs referenced by workspace nodes."""
        object_ids: set[str] = set()
        nodes = manifest.get("nodes", ())
        if not isinstance(nodes, list):
            return frozenset()
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            object_id = entry.get("payload_object_id")
            if isinstance(object_id, str) and object_id:
                object_ids.add(object_id)
        return frozenset(object_ids)

    @staticmethod
    def manifest_extension_object_ids(
        manifest: Mapping[str, typing.Any],
    ) -> frozenset[str]:
        """Return embedded extension object IDs referenced by a manifest."""
        object_ids: set[str] = set()
        requirements = manifest.get("extension_requirements", ())
        if isinstance(requirements, list):
            for requirement in requirements:
                if not isinstance(requirement, dict):
                    continue
                object_id = requirement.get("embedded_object_id")
                if isinstance(object_id, str) and object_id:
                    object_ids.add(object_id)
        return frozenset(object_ids)

    @classmethod
    def manifest_object_ids(cls, manifest: Mapping[str, typing.Any]) -> frozenset[str]:
        """Return all immutable object IDs referenced by a manifest."""
        return cls.manifest_node_object_ids(
            manifest
        ) | cls.manifest_extension_object_ids(manifest)

    def collect_garbage(
        self,
        *,
        max_objects: int = 1,
        on_contention: Callable[[], None] | None = None,
    ) -> bool:
        """Retire old generations and unlink a bounded number of dead objects.

        Returns ``True`` when more obsolete objects remain.
        """
        if max_objects < 1:
            raise ValueError("max_objects must be positive")
        with self.write_session(on_contention=on_contention):
            generations = self.generations()
            retained = generations[-2:]
            retained_names = {
                f"{generation.sequence:0{_WORKSPACE_GENERATION_WIDTH}d}"
                for generation in retained
            }
            generation_root = self.h5_file.require_group(_WORKSPACE_GENERATIONS_GROUP)
            for name in list(generation_root):
                if name in retained_names:
                    continue
                del generation_root[name]

            reachable = set(self._object_leases)
            reachable.update(self.serialized_object_ids)
            for generation in retained:
                reachable.update(self.manifest_object_ids(generation.manifest))

            object_root = self.h5_file.require_group(_WORKSPACE_OBJECTS_GROUP)
            obsolete = [name for name in object_root if name not in reachable]
            remove_count = len(obsolete) if not self._locking_supported else max_objects
            for name in obsolete[:remove_count]:
                del object_root[name]
            self.h5_file.flush()
            return len(obsolete) > remove_count

    def clear_staging(self) -> None:
        """Remove unpublished staging groups left by interrupted saves."""
        with self.read_session() as h5_file:
            staging_root = h5_file.get(_WORKSPACE_STAGING_GROUP)
            if staging_root is None or not staging_root:
                return
        with self.write_session():
            staging_root = self.h5_file.require_group(_WORKSPACE_STAGING_GROUP)
            for name in list(staging_root):
                del staging_root[name]
            self.h5_file.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=WorkspaceStore._before_fork,
        after_in_parent=WorkspaceStore._after_fork_parent,
        after_in_child=WorkspaceStore._after_fork_child,
    )
