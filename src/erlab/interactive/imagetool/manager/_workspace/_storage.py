"""Immutable workspace generations and legacy transaction recovery."""

from __future__ import annotations

import contextlib
import ctypes
import errno
import json
import os
import pathlib
import stat
import sys
import typing
import uuid
from dataclasses import dataclass

from qtpy import QtCore

import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
from erlab.interactive.imagetool.manager._workspace._format import (
    _WORKSPACE_BACKUP_GROUP_PREFIX,
    _WORKSPACE_PENDING_GROUP_PREFIX,
    _WORKSPACE_TRANSACTION_GROUP_PREFIX,
    _workspace_file_is_workspace,
    _workspace_path_is_itws,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import h5py
    import xarray as xr

    from erlab.interactive._options.schema import WorkspaceCompressionMode
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")


@dataclass(frozen=True)
class _WorkspaceObjectWrite:
    """One immutable payload object required by a generation."""

    object_id: str
    dataset: xr.Dataset | None = None
    source_file: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class _WorkspaceGroupCopy:
    """One live old-format group that a new document must preserve."""

    source_file: str
    source_path: str
    target_path: str


@dataclass(frozen=True)
class _WorkspaceGenerationPlan:
    """A complete manifest and the new objects that it references."""

    manifest: dict[str, typing.Any]
    objects: tuple[_WorkspaceObjectWrite, ...]
    preserved_groups: tuple[_WorkspaceGroupCopy, ...] = ()


def _workspace_object_copy_source(item: _WorkspaceObjectWrite) -> tuple[str, str]:
    if item.source_file is None or item.source_path is None:
        raise ValueError(f"Workspace object {item.object_id!r} has no source")
    return item.source_file, item.source_path


@contextlib.contextmanager
def _workspace_generation_copy_source(
    path: str | os.PathLike[str],
) -> Iterator[typing.Any]:
    active_store = workspace_store.WorkspaceStore.active(path)
    if active_store is not None:
        with active_store.read_session() as h5_file:
            yield h5_file
        return
    workspace_arrays.ensure_workspace_hdf5_filters_registered()
    with workspace_arrays._workspace_file_lock(path), h5py.File(path, "r") as h5_file:
        yield h5_file


def _copy_workspace_group(
    target_store: workspace_store.WorkspaceStore,
    *,
    source_file: str,
    source_path: str,
    target_path: str,
) -> None:
    try:
        with (
            _workspace_generation_copy_source(source_file) as source_h5_file,
            target_store.lock,
        ):
            copied = workspace_arrays._copy_workspace_h5_group_to_open_file(
                source_h5_file,
                target_store.h5_file,
                source_path,
                target_path,
                None,
            )
    except Exception as exc:
        with target_store.lock:
            workspace_arrays._delete_h5_path(target_store.h5_file, target_path)
            target_store.h5_file.flush()
        if isinstance(exc, FileNotFoundError):
            raise _WorkspaceBackingFileNotFoundError(source_file) from exc
        raise
    if not copied:
        with target_store.lock:
            workspace_arrays._delete_h5_path(target_store.h5_file, target_path)
            target_store.h5_file.flush()
        raise KeyError(
            f"Workspace payload group {source_path!r} is missing from {source_file!r}"
        )


def _write_workspace_generation(
    target_store: workspace_store.WorkspaceStore,
    plan: _WorkspaceGenerationPlan,
    *,
    compression_mode: WorkspaceCompressionMode,
    on_contention: Callable[[], None] | None = None,
) -> workspace_store._WorkspaceGeneration:
    """Write new immutable objects and publish one generation."""
    with target_store.write_session(on_contention=on_contention):
        created_object_ids: list[str] = []
        try:
            with target_store.lock:
                target_store.require_current_path()
            for item in plan.preserved_groups:
                with target_store.lock:
                    if item.target_path.strip("/") in target_store.h5_file:
                        continue
                _copy_workspace_group(
                    target_store,
                    source_file=item.source_file,
                    source_path=item.source_path,
                    target_path=item.target_path,
                )
            for item in plan.objects:
                target_path = target_store.object_path(item.object_id)
                with target_store.lock:
                    if target_path.strip("/") in target_store.h5_file:
                        continue
                created_object_ids.append(item.object_id)
                if item.dataset is not None:
                    if workspace_arrays._workspace_dataset_can_write_h5py(item.dataset):
                        with target_store.lock:
                            workspace_arrays._write_workspace_dataset_group_to_file(
                                target_store.h5_file,
                                target_path,
                                item.dataset,
                                compression_mode=compression_mode,
                            )
                    else:
                        workspace_arrays._write_workspace_dataset_group_to_file(
                            target_store.h5_file,
                            target_path,
                            item.dataset,
                            compression_mode=compression_mode,
                        )
                    continue
                source_file, source_path = _workspace_object_copy_source(item)
                _copy_workspace_group(
                    target_store,
                    source_file=source_file,
                    source_path=source_path,
                    target_path=target_path,
                )
            return target_store.publish(plan.manifest)
        except Exception:
            with contextlib.suppress(Exception), target_store.lock:
                referenced_object_ids: set[str] = set()
                for generation in target_store.generations():
                    referenced_object_ids.update(
                        target_store.manifest_object_ids(generation.manifest)
                    )
                for object_id in created_object_ids:
                    if object_id not in referenced_object_ids:
                        workspace_arrays._delete_h5_path(
                            target_store.h5_file,
                            target_store.object_path(object_id),
                        )
                target_store.h5_file.flush()
            raise


def _compact_workspace_store(
    store: workspace_store.WorkspaceStore,
    *,
    discard_serialized_reader_pins: (
        workspace_store._SerializedReaderPinSnapshot | None
    ) = None,
) -> None:
    """Rewrite a workspace while omitting confirmed obsolete reader pins."""
    workspace_path = store.path
    prepared_path = workspace_path.with_name(
        f".{workspace_path.name}.compact-{uuid.uuid4().hex}"
    )
    try:
        with store.write_lock, store.lock:
            current = store.current_generation()
            baseline_manifest = dict(current.manifest)
            baseline_manifest.pop("delta_save_count", None)
            baseline_manifest.pop("estimated_obsolete_bytes", None)
            baseline_manifest.pop("replacement_delta_count", None)
            baseline_manifest.pop("repack_estimate_known", None)
            object_ids = set(store.manifest_object_ids(baseline_manifest))
            object_ids.update(store.leased_object_ids)
            serialized_pins = store.serialized_reader_pin_snapshot()
            discarded_pins = (
                discard_serialized_reader_pins
                or workspace_store._SerializedReaderPinSnapshot({}, {})
            )
            object_ids.update(
                object_id
                for object_id, version in serialized_pins.object_versions.items()
                if version > discarded_pins.object_versions.get(object_id, -1)
            )
            legacy_group_paths = store.leased_legacy_group_paths | {
                group_path
                for group_path, version in serialized_pins.legacy_group_versions.items()
                if version > discarded_pins.legacy_group_versions.get(group_path, -1)
            }
            expected_state = _workspace_publication_state(workspace_path)

            with (
                workspace_store.WorkspaceStore(
                    prepared_path,
                    create=True,
                    workspace_id=store.workspace_id,
                ) as compacted,
                compacted.write_session(),
            ):
                with compacted.lock:
                    for group_path in sorted(legacy_group_paths):
                        copied = workspace_arrays._copy_workspace_h5_group_to_open_file(
                            store.h5_file,
                            compacted.h5_file,
                            group_path,
                            group_path,
                            None,
                        )
                        if not copied:
                            raise KeyError(
                                f"Workspace payload group {group_path!r} is missing"
                            )
                    for object_id in sorted(object_ids):
                        object_path = store.object_path(object_id)
                        copied = workspace_arrays._copy_workspace_h5_group_to_open_file(
                            store.h5_file,
                            compacted.h5_file,
                            object_path,
                            object_path,
                            None,
                        )
                        if not copied:
                            raise KeyError(f"Workspace object {object_id!r} is missing")
                compacted.publish(baseline_manifest)
                compacted.publish(baseline_manifest)
                generations = compacted.generations()
                if (
                    len(generations) != 2
                    or generations[0].manifest != generations[1].manifest
                ):
                    raise RuntimeError(
                        "Compacted workspace baseline generations do not match"
                    )
                if any(
                    compacted.object_path(object_id).strip("/") not in compacted.h5_file
                    for object_id in object_ids
                ):
                    raise RuntimeError(
                        "Compacted workspace is missing a payload object"
                    )
                compacted.flush(durable=True)

            expected_identity = expected_state[1:3]

            def _publish_compacted_workspace(
                source: pathlib.Path, destination: pathlib.Path
            ) -> None:
                closed_state = _workspace_publication_state(destination)
                if not closed_state[0] or closed_state[1:3] != expected_identity:
                    raise _WorkspacePublicationConflictError(destination)
                _replace_workspace_file(
                    source,
                    destination,
                    expected_state=closed_state,
                )

            store.replace_from(
                prepared_path,
                _publish_compacted_workspace,
                before_close=lambda: _require_workspace_publication_state(
                    workspace_path, expected_state
                ),
            )
    finally:
        if store.recovery_path != prepared_path:
            with contextlib.suppress(OSError):
                prepared_path.unlink()


_WorkspacePublicationState: typing.TypeAlias = tuple[bool, int, int, int, int, int]


class _WorkspaceBackingFileNotFoundError(FileNotFoundError):
    """A workspace payload source disappeared before it could be copied."""

    def __init__(self, source_path: str | os.PathLike[str]) -> None:
        self.source_path = os.fsdecode(source_path)
        super().__init__(
            errno.ENOENT,
            "Workspace backing file is missing",
            self.source_path,
        )


class _WorkspacePublicationConflictError(workspace_store.WorkspaceStoreConflictError):
    """A published workspace changed while a save was in progress."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fsdecode(path)
        super().__init__(
            "Workspace changed outside ImageTool Manager while it was being saved: "
            f"{self.path}"
        )


def _workspace_publication_state(
    path: str | os.PathLike[str],
) -> _WorkspacePublicationState:
    """Return the state used to detect an external document replacement."""
    try:
        stat_result = os.stat(path)
    except FileNotFoundError:
        return False, 0, 0, 0, 0, 0
    return (
        True,
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _replace_workspace_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    expected_state: _WorkspacePublicationState | None,
) -> None:
    """Publish a prepared workspace with bounded file-access retries."""

    def _replace() -> None:
        if (
            expected_state is not None
            and _workspace_publication_state(destination) != expected_state
        ):
            raise _WorkspacePublicationConflictError(destination)
        os.replace(source, destination)

    workspace_store._retry_file_access(_replace)
    _fsync_parent_directory(destination)


def _fsync_parent_directory(path: str | os.PathLike[str]) -> None:
    """Ask POSIX to persist a completed directory entry replacement."""
    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        file_descriptor = os.open(pathlib.Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)


def _require_workspace_publication_state(
    path: str | os.PathLike[str], expected_state: _WorkspacePublicationState
) -> None:
    if _workspace_publication_state(path) != expected_state:
        raise _WorkspacePublicationConflictError(path)


@dataclass(frozen=True)
class _WorkspaceDocumentLockInfo:
    path: str
    owner: str
    hostname: str
    appname: str
    pid: int | None


@contextlib.contextmanager
def _open_workspace_h5_file_for_update(
    fname: str | os.PathLike[str],
) -> Iterator[typing.Any]:
    with workspace_arrays._workspace_file_lock(fname), h5py.File(fname, "a") as h5_file:
        yield h5_file


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_workspace_file_lock_error(exc: BaseException) -> bool:
    lock_errnos = {errno.EACCES, errno.EAGAIN}
    if hasattr(errno, "EWOULDBLOCK"):
        lock_errnos.add(errno.EWOULDBLOCK)

    for err in _iter_exception_chain(exc):
        if isinstance(err, BlockingIOError):
            return True
        message = str(err).lower()
        if not any(
            marker in message
            for marker in (
                "unable to lock file",
                "resource temporarily unavailable",
                "file is already open",
            )
        ):
            continue
        err_no = getattr(err, "errno", None)
        if err_no in lock_errnos or "unable to lock file" in message:
            return True
        if "file is already open" in message:
            return True
    return False


def _workspace_lock_path(fname: str | os.PathLike[str]) -> str:
    workspace_path = pathlib.Path(fname).resolve()
    return str(workspace_path.with_name(f".{workspace_path.name}.lock"))


def _hide_workspace_lock_file(lock_path: str) -> None:
    if sys.platform == "darwin":
        with contextlib.suppress(AttributeError, OSError):
            if not stat.S_ISREG(os.lstat(lock_path).st_mode):
                return
            os.chflags(lock_path, stat.UF_HIDDEN)
        return
    if os.name != "nt":
        return

    with contextlib.suppress(Exception):
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return
        windll.kernel32.SetFileAttributesW(
            str(lock_path),
            0x2,  # FILE_ATTRIBUTE_HIDDEN
        )


def _workspace_document_lock_info(
    fname: str | os.PathLike[str],
) -> _WorkspaceDocumentLockInfo:
    lock_path = _workspace_lock_path(fname)
    lock = QtCore.QLockFile(lock_path)
    locked = False
    pid = 0
    hostname = ""
    appname = ""
    with contextlib.suppress(Exception):
        locked, pid, hostname, appname = lock.getLockInfo()
    if not locked:
        pid = 0
        hostname = ""
        appname = ""
    owner = QtCore.QFileInfo(lock_path).owner()
    return _WorkspaceDocumentLockInfo(
        path=lock_path,
        owner=owner,
        hostname=hostname,
        appname=appname,
        pid=pid if pid > 0 else None,
    )


def _acquire_workspace_document_lock(
    fname: str | os.PathLike[str],
) -> QtCore.QLockFile:
    lock_path = _workspace_lock_path(fname)
    lock = QtCore.QLockFile(lock_path)
    # Document locks are long-lived; Qt uses 0 to disable age-based stale detection.
    lock.setStaleLockTime(0)
    # Do not block the UI thread when another manager already owns the workspace.
    if not lock.tryLock(0):
        raise BlockingIOError(
            errno.EAGAIN,
            f"Workspace file is already open or locked: {fname}",
        )
    _hide_workspace_lock_file(lock_path)
    return lock


def _workspace_txn_attr_target(h5_file, target_path: str):
    if target_path == "/":
        return h5_file.attrs
    group_path = target_path.strip("/")
    if group_path not in h5_file:
        return None
    return h5_file[group_path].attrs


def _write_workspace_attr_backup(
    txn_group, index: int, target_path: str, attrs
) -> None:
    backup_group = txn_group.require_group("attr_backups").create_group(str(index))
    backup_group.attrs["target_path"] = target_path
    attrs_group = backup_group.create_group("attrs")
    for key, value in attrs.items():
        attrs_group.attrs[key] = value


def _restore_workspace_attr_backups(h5_file, txn_group) -> None:
    if "attr_backups" not in txn_group:
        return
    backups = txn_group["attr_backups"]
    for key in sorted(backups, key=lambda value: int(value) if value.isdigit() else 0):
        backup_group = backups[key]
        target_path = backup_group.attrs.get("target_path")
        if isinstance(target_path, bytes):
            target_path = target_path.decode()
        if not isinstance(target_path, str) or "attrs" not in backup_group:
            continue
        target_attrs = _workspace_txn_attr_target(h5_file, target_path)
        if target_attrs is None:
            continue
        workspace_arrays._replace_h5_attrs(target_attrs, backup_group["attrs"].attrs)


def _workspace_transaction_operations(txn_group) -> dict[str, typing.Any]:
    raw_operations = txn_group.attrs.get("operations")
    if isinstance(raw_operations, bytes):
        raw_operations = raw_operations.decode()
    if isinstance(raw_operations, str):
        with contextlib.suppress(json.JSONDecodeError):
            operations = json.loads(raw_operations)
            if isinstance(operations, dict):
                return operations
    return {}


def _workspace_transaction_roots(txn_group) -> tuple[str | None, str | None]:
    pending_root = txn_group.attrs.get("pending_root")
    backup_root = txn_group.attrs.get("backup_root")
    if isinstance(pending_root, bytes):
        pending_root = pending_root.decode()
    if isinstance(backup_root, bytes):
        backup_root = backup_root.decode()
    return (
        pending_root if isinstance(pending_root, str) else None,
        backup_root if isinstance(backup_root, str) else None,
    )


def _cleanup_workspace_transaction_roots(
    h5_file,
    txn_path: str,
    *,
    pending_root: str | None,
    backup_root: str | None,
) -> None:
    for path in (pending_root, backup_root, txn_path):
        if path is not None:
            workspace_arrays._delete_h5_path(h5_file, path)


def _rollback_workspace_group_operations(
    h5_file,
    operations: Mapping[str, typing.Any],
) -> None:
    group_replacements = operations.get("group_replacements", ())
    if not isinstance(group_replacements, list):
        return
    for operation in reversed(group_replacements):
        if not isinstance(operation, dict):
            continue
        group_path = operation.get("group_path")
        backup_path = operation.get("backup_path")
        old_exists = bool(operation.get("old_exists", False))
        if not isinstance(group_path, str) or not isinstance(backup_path, str):
            continue
        backup_exists = workspace_arrays._h5_path_exists(h5_file, backup_path)
        if backup_exists:
            workspace_arrays._delete_h5_path(h5_file, group_path)
            workspace_arrays._ensure_h5_parent_group(h5_file, group_path)
            h5_file.move(backup_path.strip("/"), group_path.strip("/"))
        elif not old_exists:
            workspace_arrays._delete_h5_path(h5_file, group_path)


def _recover_open_workspace_transaction(h5_file, txn_path: str) -> None:
    txn_group = h5_file[txn_path]
    status = txn_group.attrs.get("status")
    if isinstance(status, bytes):
        status = status.decode()
    operations = _workspace_transaction_operations(txn_group)
    pending_root, backup_root = _workspace_transaction_roots(txn_group)

    if status == "committing":
        _rollback_workspace_group_operations(h5_file, operations)
        _restore_workspace_attr_backups(h5_file, txn_group)

    _cleanup_workspace_transaction_roots(
        h5_file,
        txn_path,
        pending_root=pending_root,
        backup_root=backup_root,
    )


def _cleanup_orphan_workspace_internal_groups(h5_file) -> None:
    transaction_roots: set[str] = set()
    for name in list(h5_file):
        if not name.startswith(_WORKSPACE_TRANSACTION_GROUP_PREFIX):
            continue
        pending_root, backup_root = _workspace_transaction_roots(h5_file[name])
        transaction_roots.update(
            root for root in (pending_root, backup_root) if root is not None
        )
    internal_prefixes = (
        _WORKSPACE_PENDING_GROUP_PREFIX,
        _WORKSPACE_BACKUP_GROUP_PREFIX,
    )
    for name in list(h5_file):
        if name.startswith(internal_prefixes) and name not in transaction_roots:
            del h5_file[name]


def _recover_workspace_transactions(fname: str | os.PathLike[str]) -> None:
    with workspace_arrays._workspace_save_lock(fname):
        if not pathlib.Path(fname).exists():
            return
        if not _workspace_path_is_itws(fname):
            return
        active_store = workspace_store.WorkspaceStore.active(fname)
        file_context = (
            active_store.write_session()
            if active_store is not None
            else h5py.File(fname, "a")
        )
        with file_context as h5_file:
            if not _workspace_file_is_workspace(h5_file):
                return
            for name in list(h5_file):
                if name.startswith(_WORKSPACE_TRANSACTION_GROUP_PREFIX):
                    _recover_open_workspace_transaction(h5_file, name)
            _cleanup_orphan_workspace_internal_groups(h5_file)
            h5_file.flush()
