"""Transactional persistence and save workers for manager workspaces."""

from __future__ import annotations

import collections
import contextlib
import ctypes
import errno
import json
import logging
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import time
import traceback
import typing
import uuid
from dataclasses import dataclass

import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
from erlab.interactive import _qt_state
from erlab.interactive.imagetool.manager._dialogs import _ChooseFromDataTreeDialog
from erlab.interactive.imagetool.manager._widgets import (
    _strip_workspace_modified_placeholder,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping

    import h5py

    from erlab.interactive._options.schema import WorkspaceCompressionMode
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._widgets import _WorkspaceDocumentAccess
    from erlab.interactive.imagetool.manager._workspace._controller import (
        _WorkspaceController,
    )
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

from erlab.interactive.imagetool.manager._workspace._format import (
    _WORKSPACE_BACKUP_GROUP_PREFIX,
    _WORKSPACE_PENDING_GROUP_PREFIX,
    _WORKSPACE_SCHEMA_VERSION,
    _WORKSPACE_TRANSACTION_GROUP_PREFIX,
    _WORKSPACE_TRANSACTION_PROTOCOL,
    _compacted_workspace_root_attrs,
    _require_itws_workspace_path,
    _workspace_file_is_workspace,
    _workspace_path_is_itws,
)

logger = logging.getLogger(__name__)
_WORKSPACE_SAVE_SUFFIX_ERROR = "ImageTool workspace documents must be saved as .itws"
_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES = 128 * 1024 * 1024
_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO = 0.10

_WorkspaceCopyGroup: typing.TypeAlias = tuple[str, str, dict[str, typing.Any] | None]
_WorkspaceCopyGroupWithSource: typing.TypeAlias = tuple[
    str, str, str, dict[str, typing.Any] | None
]


@dataclass
class _WorkspaceSaveSnapshot:
    generation: int
    root_attrs: dict[str, typing.Any]
    delta_save_count: int
    estimated_obsolete_bytes: int = 0
    replacement_delta_count: int = 0
    repack_estimate_known: bool = True
    compression_mode: WorkspaceCompressionMode = "zstd1"
    file_repack: bool = False
    full_tree: xr.DataTree | None = None
    copy_source: str | None = None
    copy_groups: tuple[_WorkspaceCopyGroup, ...] = ()
    copy_group_sources: tuple[_WorkspaceCopyGroupWithSource, ...] = ()
    rewrite_groups: tuple[tuple[str, dict[str, xr.Dataset]], ...] = ()
    attr_updates: tuple[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]], ...
    ] = ()

    def close(self) -> None:
        if self.full_tree is not None:
            self.full_tree.close()


@dataclass(frozen=True)
class _WorkspaceDocumentLockInfo:
    path: str
    owner: str
    hostname: str
    appname: str
    pid: int | None


class _WorkspaceSaveWorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(bool, float, str)


class _WorkspaceSaveResultReceiver(QtCore.QObject):
    def __init__(
        self,
        *,
        callback: Callable[[bool, float, str], None] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._callback = callback

    @QtCore.Slot(bool, float, str)
    def finish(self, ok: bool, elapsed: float, error_text: str) -> None:
        if self._callback is not None:
            self._callback(ok, elapsed, error_text)


class _WorkspaceSaveWorker(QtCore.QRunnable):
    def __init__(
        self,
        fname: str | os.PathLike[str],
        snapshot: _WorkspaceSaveSnapshot,
    ) -> None:
        super().__init__()
        self.signals = _WorkspaceSaveWorkerSignals()
        self._fname = fname
        self._snapshot = snapshot

    def run(self) -> None:
        start_time = time.perf_counter()
        error_text = ""
        ok = False
        try:
            if self._snapshot.file_repack:
                _write_full_workspace_tree_file(
                    self._fname,
                    None,
                    self._snapshot.root_attrs,
                    copy_source=self._snapshot.copy_source,
                    copy_groups=self._snapshot.copy_groups,
                    copy_group_sources=self._snapshot.copy_group_sources,
                    compression_mode=self._snapshot.compression_mode,
                )
            elif self._snapshot.full_tree is None:
                _write_workspace_transaction_file(
                    self._fname,
                    self._snapshot.rewrite_groups,
                    self._snapshot.attr_updates,
                    self._snapshot.root_attrs,
                    compression_mode=self._snapshot.compression_mode,
                )
            else:
                _write_full_workspace_tree_file(
                    self._fname,
                    self._snapshot.full_tree,
                    self._snapshot.root_attrs,
                    copy_source=self._snapshot.copy_source,
                    copy_groups=self._snapshot.copy_groups,
                    copy_group_sources=self._snapshot.copy_group_sources,
                    compression_mode=self._snapshot.compression_mode,
                )
            ok = True
        except Exception:
            error_text = traceback.format_exc()
        finally:
            with contextlib.suppress(Exception):
                self._snapshot.close()
        self.signals.finished.emit(ok, time.perf_counter() - start_time, error_text)


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


def _workspace_path_is_unc(path: str | os.PathLike[str]) -> bool:
    path_str = os.fsdecode(path)
    return path_str.startswith(("\\\\", "//"))


def _workspace_path_is_likely_cloud_path(path: str | os.PathLike[str]) -> bool:
    cloud_markers = {
        "com~apple~clouddocs",
        "dropbox",
        "google drive",
        "icloud drive",
        "icloud~com~apple~clouddocs",
        "mobile documents",
        "onedrive",
        "one drive",
    }
    try:
        parts = pathlib.Path(path).resolve().parts
    except OSError:
        parts = pathlib.Path(path).absolute().parts
    normalized_parts = {part.casefold().replace("-", " ") for part in parts}
    return any(marker in part for part in normalized_parts for marker in cloud_markers)


def _workspace_path_is_likely_network_path(path: str | os.PathLike[str]) -> bool:
    if _workspace_path_is_unc(path):
        return True
    try:
        resolved = pathlib.Path(path).resolve()
    except OSError:
        resolved = pathlib.Path(path).absolute()
    parts = resolved.parts
    if sys.platform == "darwin" and len(parts) > 2 and parts[1] == "Volumes":
        return True
    return len(parts) > 1 and parts[1] in {"net", "nfs", "smb"}


def _workspace_path_is_high_risk(path: str | os.PathLike[str]) -> bool:
    return _workspace_path_is_likely_network_path(
        path
    ) or _workspace_path_is_likely_cloud_path(path)


def _workspace_use_incremental_enabled() -> bool:
    return bool(erlab.interactive.options.model.io.workspace.use_incremental)


def _workspace_incremental_save_on_remote_enabled() -> bool:
    return bool(erlab.interactive.options.model.io.workspace.incremental_save_on_remote)


def _workspace_requires_full_save(
    fname: str | os.PathLike[str],
    *,
    needs_full_save: bool,
    schema_version: int,
    structure_modified: bool,
    has_dirty_added: bool,
    has_dirty_removed: bool,
) -> bool:
    if not _workspace_use_incremental_enabled():
        return True
    if (
        not _workspace_incremental_save_on_remote_enabled()
        and _workspace_path_is_high_risk(fname)
    ):
        return True
    return (
        needs_full_save
        or not pathlib.Path(fname).exists()
        or schema_version != _WORKSPACE_SCHEMA_VERSION
        or structure_modified
        or has_dirty_added
        or has_dirty_removed
    )


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
    if not pathlib.Path(fname).exists():
        return
    if not _workspace_path_is_itws(fname):
        return
    with workspace_arrays._workspace_file_lock(fname), h5py.File(fname, "a") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            return
        for name in list(h5_file):
            if name.startswith(_WORKSPACE_TRANSACTION_GROUP_PREFIX):
                _recover_open_workspace_transaction(h5_file, name)
        _cleanup_orphan_workspace_internal_groups(h5_file)
        h5_file.flush()


def _write_root_attrs_to_open_workspace_file(
    h5_file,
    attrs: Mapping[str, typing.Any],
    *,
    replace: bool = False,
) -> None:
    if replace:
        for key in list(h5_file.attrs):
            del h5_file.attrs[key]
    for key, value in attrs.items():
        h5_file.attrs[key] = value


def _write_workspace_root_attrs_to_file(
    fname: str | os.PathLike[str],
    attrs: Mapping[str, typing.Any],
    *,
    replace: bool = False,
) -> None:
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        _write_root_attrs_to_open_workspace_file(h5_file, attrs, replace=replace)


def _workspace_obsolete_estimate(
    fname: str | os.PathLike[str],
) -> int:
    try:
        file_size = pathlib.Path(fname).stat().st_size
    except OSError:
        return 0
    live_storage_size = workspace_arrays._workspace_live_h5_storage_size(fname)
    return max(0, int(file_size) - live_storage_size)


def _workspace_file_repack_payload(
    fname: str | os.PathLike[str],
) -> tuple[
    dict[str, typing.Any], tuple[tuple[str, str, dict[str, typing.Any] | None], ...]
]:
    _recover_workspace_transactions(fname)
    root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(fname)
    return (
        _compacted_workspace_root_attrs(root_attrs),
        workspace_arrays._workspace_live_root_group_copy_groups(fname),
    )


def _write_workspace_constructor_groups_to_pending(
    fname: str | os.PathLike[str],
    constructor: Mapping[str, xr.Dataset],
    group_path: str,
    pending_path: str,
    *,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    target_group_path = group_path.strip("/")
    pending_path = pending_path.strip("/")
    for constructor_group_path, ds in sorted(
        constructor.items(), key=lambda item: item[0].count("/")
    ):
        source_group_path = constructor_group_path.strip("/")
        if source_group_path != target_group_path and not source_group_path.startswith(
            f"{target_group_path}/"
        ):
            continue
        relative_path = source_group_path.removeprefix(target_group_path).strip("/")
        pending_group_path = (
            pending_path if not relative_path else f"{pending_path}/{relative_path}"
        )
        workspace_arrays._write_workspace_dataset_group_to_file(
            fname,
            pending_group_path,
            ds,
            lock_path=fname,
            compression_mode=compression_mode,
        )


def _path_is_at_or_under(path: str, root_path: str) -> bool:
    path = path.strip("/")
    root_path = root_path.strip("/")
    return path == root_path or path.startswith(f"{root_path}/")


def _move_h5_path(h5_file, source_path: str, destination_path: str) -> None:
    workspace_arrays._ensure_h5_parent_group(h5_file, destination_path)
    h5_file.move(source_path.strip("/"), destination_path.strip("/"))


def _set_workspace_transaction_status(h5_file, txn_path: str, status: str) -> None:
    h5_file[txn_path].attrs["status"] = status
    h5_file.flush()


def _prepare_workspace_transaction(
    fname: str | os.PathLike[str],
    txn_path: str,
    pending_root: str,
    backup_root: str,
    rewrite_map: dict[str, tuple[str, dict[str, xr.Dataset]]],
    attr_updates: tuple[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]], ...
    ],
    root_attrs: Mapping[str, typing.Any],
) -> tuple[
    list[dict[str, typing.Any]],
    list[tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]],
]:
    attr_updates_to_write: list[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
    ] = []
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        txn_group = h5_file.create_group(txn_path)
        txn_group.attrs["protocol"] = _WORKSPACE_TRANSACTION_PROTOCOL
        txn_group.attrs["status"] = "preparing"
        txn_group.attrs["pending_root"] = pending_root
        txn_group.attrs["backup_root"] = backup_root

        attr_backup_index = 0
        for payload_path, attrs, fallback in attr_updates:
            attr_path = payload_path.strip("/")
            if any(
                _path_is_at_or_under(attr_path, rewrite_path)
                for rewrite_path in rewrite_map
            ):
                continue
            if attr_path in h5_file:
                _write_workspace_attr_backup(
                    txn_group,
                    attr_backup_index,
                    payload_path,
                    h5_file[attr_path].attrs,
                )
                attr_backup_index += 1
                attr_updates_to_write.append((payload_path, attrs, fallback))
            else:
                fallback_path = fallback[0].strip("/")
                if not any(
                    _path_is_at_or_under(fallback_path, rewrite_path)
                    for rewrite_path in rewrite_map
                ):
                    rewrite_map[fallback_path] = fallback

        _write_workspace_attr_backup(txn_group, attr_backup_index, "/", h5_file.attrs)

        group_operations: list[dict[str, typing.Any]] = []
        for group_path in sorted(rewrite_map):
            pending_path = f"{pending_root}/{group_path}"
            backup_path = f"{backup_root}/{group_path}"
            group_operations.append(
                {
                    "group_path": group_path,
                    "pending_path": pending_path,
                    "backup_path": backup_path,
                    "old_exists": workspace_arrays._h5_path_exists(h5_file, group_path),
                }
            )
        txn_group.attrs["operations"] = json.dumps(
            {"group_replacements": group_operations}
        )
        h5_file.flush()
    return group_operations, attr_updates_to_write


def _write_workspace_transaction_pending_groups(
    fname: str | os.PathLike[str],
    rewrite_map: Mapping[str, tuple[str, dict[str, xr.Dataset]]],
    pending_root: str,
    *,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    try:
        for group_path, (rewrite_group_path, constructor) in sorted(
            rewrite_map.items()
        ):
            pending_group_path = f"{pending_root}/{group_path}"
            _write_workspace_constructor_groups_to_pending(
                fname,
                constructor,
                rewrite_group_path,
                pending_group_path,
                compression_mode=compression_mode,
            )
    except Exception:
        with _open_workspace_h5_file_for_update(fname) as h5_file:
            workspace_arrays._delete_h5_path(h5_file, pending_root)
            h5_file.flush()
        raise


def _commit_workspace_transaction(
    fname: str | os.PathLike[str],
    txn_path: str,
    group_operations: Iterable[Mapping[str, typing.Any]],
    attr_updates: Iterable[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
    ],
    root_attrs: Mapping[str, typing.Any],
) -> None:
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        _set_workspace_transaction_status(h5_file, txn_path, "committing")
        for operation in group_operations:
            group_path = typing.cast("str", operation["group_path"])
            pending_path = typing.cast("str", operation["pending_path"])
            backup_path = typing.cast("str", operation["backup_path"])
            if not workspace_arrays._h5_path_exists(h5_file, pending_path):
                raise KeyError(
                    f"Workspace pending group {pending_path!r} was not written"
                )
            if workspace_arrays._h5_path_exists(h5_file, group_path):
                _move_h5_path(h5_file, group_path, backup_path)
            _move_h5_path(h5_file, pending_path, group_path)

        for payload_path, attrs, _fallback in attr_updates:
            target_attrs = _workspace_txn_attr_target(h5_file, payload_path)
            if target_attrs is not None:
                workspace_arrays._replace_h5_attrs(target_attrs, attrs)

        _write_root_attrs_to_open_workspace_file(h5_file, root_attrs)
        _set_workspace_transaction_status(h5_file, txn_path, "committed")


def _write_workspace_transaction_file(
    fname: str | os.PathLike[str],
    rewrite_groups: Iterable[tuple[str, dict[str, xr.Dataset]]],
    attr_updates: Iterable[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
    ],
    root_attrs: Mapping[str, typing.Any],
    *,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    _recover_workspace_transactions(fname)
    rewrite_map = {
        group_path.strip("/"): (group_path, constructor)
        for group_path, constructor in rewrite_groups
    }
    attr_updates_tuple = tuple(attr_updates)
    txn_id = uuid.uuid4().hex
    txn_path = f"{_WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{_WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{_WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, attr_updates_to_write = _prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        attr_updates_tuple,
        root_attrs,
    )
    try:
        _write_workspace_transaction_pending_groups(
            fname,
            rewrite_map,
            pending_root,
            compression_mode=compression_mode,
        )
        _commit_workspace_transaction(
            fname,
            txn_path,
            group_operations,
            attr_updates_to_write,
            root_attrs,
        )
    except Exception:
        _recover_workspace_transactions(fname)
        raise
    _recover_workspace_transactions(fname)


def _write_full_workspace_tree_file(
    fname: str | os.PathLike[str],
    tree: xr.DataTree | None,
    root_attrs: Mapping[str, typing.Any],
    *,
    copy_source: str | os.PathLike[str] | None = None,
    copy_groups: Iterable[_WorkspaceCopyGroup] = (),
    copy_group_sources: Iterable[_WorkspaceCopyGroupWithSource] = (),
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    fname = os.fsdecode(fname)
    use_scratch = _workspace_path_is_high_risk(fname)
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    destination_tmp: str | None = None
    if use_scratch:
        tmp_dir = tempfile.TemporaryDirectory(prefix="erlab-itws-")
        tmp_fname = str(pathlib.Path(tmp_dir.name) / pathlib.Path(fname).name)
    else:
        tmp_fname = f"{fname}.tmp-{uuid.uuid4().hex}"
    try:
        copied_paths: set[str] = set()
        workspace_arrays.ensure_workspace_hdf5_filters_registered()
        copy_groups_tuple = tuple(copy_groups)
        copy_group_sources_tuple = tuple(
            (os.fsdecode(source_file), source_path, destination_path, attrs)
            for source_file, source_path, destination_path, attrs in copy_group_sources
        )
        if use_scratch and _workspace_path_is_likely_network_path(fname):
            if tree is None and copy_source is not None and copy_groups_tuple:
                raise ValueError(
                    "File-level workspace repack cannot run when HDF5 group "
                    "copy reuse is disabled"
                )
            copy_source = None
            copy_groups_tuple = ()

        copy_jobs_by_source: dict[
            str, list[tuple[str, str, dict[str, typing.Any] | None, bool]]
        ] = {}
        if copy_source is not None and copy_groups_tuple:
            copy_jobs_by_source[os.fsdecode(copy_source)] = [
                (source_path, destination_path, attrs, False)
                for source_path, destination_path, attrs in copy_groups_tuple
            ]
        for (
            source_file,
            source_path,
            destination_path,
            attrs,
        ) in copy_group_sources_tuple:
            copy_jobs_by_source.setdefault(source_file, []).append(
                (source_path, destination_path, attrs, True)
            )

        with h5py.File(tmp_fname, "w") as tmp_file:
            _write_root_attrs_to_open_workspace_file(tmp_file, root_attrs, replace=True)

        for source_fname, copy_jobs in copy_jobs_by_source.items():
            with (
                workspace_arrays._workspace_file_lock(source_fname),
                h5py.File(source_fname, "r") as source_file,
                h5py.File(tmp_fname, "a") as tmp_file,
            ):
                for source_path, destination_path, attrs, required in copy_jobs:
                    destination_path = destination_path.strip("/")
                    if workspace_arrays._copy_workspace_h5_group_to_open_file(
                        source_file,
                        tmp_file,
                        source_path,
                        destination_path,
                        attrs,
                    ):
                        copied_paths.add(destination_path)
                    elif required:
                        raise ValueError(
                            "Required workspace payload group "
                            f"{source_path!r} was not found in {source_fname!r}"
                        )

        if tree is not None:
            for node in sorted(tree.subtree, key=lambda value: value.path.count("/")):
                group_path = node.path.strip("/")
                if not group_path or group_path in copied_paths:
                    continue
                ds = node.to_dataset(inherit=False)
                if ds.variables or ds.attrs:
                    workspace_arrays._write_workspace_dataset_group_to_file(
                        tmp_fname,
                        group_path,
                        ds,
                        compression_mode=compression_mode,
                    )

        _validate_workspace_h5_file(tmp_fname)
        _fsync_file(tmp_fname)
        if use_scratch:
            try:
                os.replace(tmp_fname, fname)
            except OSError as err:
                if err.errno != errno.EXDEV:
                    raise
                destination_tmp = f"{fname}.tmp-{uuid.uuid4().hex}"
                shutil.copyfile(tmp_fname, destination_tmp)
                _fsync_file(destination_tmp)
                os.replace(destination_tmp, fname)
            _fsync_parent_directory(fname)
        else:
            os.replace(tmp_fname, fname)
            _fsync_parent_directory(fname)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_fname)
        if destination_tmp is not None:
            with contextlib.suppress(FileNotFoundError):
                os.remove(destination_tmp)
        if tmp_dir is not None:
            tmp_dir.cleanup()


def _validate_workspace_h5_file(fname: str | os.PathLike[str]) -> None:
    with h5py.File(fname, "r") as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            raise ValueError(f"Temporary workspace file is not valid: {fname}")


def _fsync_file(fname: str | os.PathLike[str]) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(fname, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_parent_directory(fname: str | os.PathLike[str]) -> None:
    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        fd = os.open(pathlib.Path(fname).parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class _WorkspaceSaver:
    """Serialize and snapshot workspace state for one manager."""

    def __init__(
        self, manager: ImageToolManager, controller: _WorkspaceController
    ) -> None:
        self._manager = manager
        self._controller = controller

    def _annotate_workspace_dataset(
        self,
        ds: xr.Dataset,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        kind: typing.Literal["imagetool", "tool"],
    ) -> xr.Dataset:
        ds.attrs["manager_node_uid"] = node.uid
        ds.attrs["manager_node_kind"] = kind
        ds.attrs["manager_node_snapshot_token"] = node.snapshot_token
        ds.attrs["manager_node_source_snapshot_token"] = node.source_snapshot_token
        ds.attrs["manager_node_added_at"] = node.added_time_iso
        if node.note:
            ds.attrs["manager_node_note"] = node.note
        else:
            ds.attrs.pop("manager_node_note", None)
        persistence = node.persistence_view()
        provenance_spec = persistence.provenance_spec
        if provenance_spec is not None:
            ds.attrs["manager_node_provenance_spec"] = json.dumps(
                provenance_spec.model_dump(mode="json")
            )
        if isinstance(node, _ImageToolWrapper) and node.source_input_ndim is not None:
            ds.attrs["manager_node_source_input_ndim"] = int(node.source_input_ndim)
        if isinstance(node, _ImageToolWrapper) and node.watched:
            watched_metadata = node.watched_metadata()
            ds.attrs["manager_node_watched_varname"] = typing.cast(
                "str", watched_metadata["varname"]
            )
            ds.attrs["manager_node_watched_uid"] = typing.cast(
                "str", watched_metadata["uid"]
            )
            workspace_link_id = watched_metadata.get("workspace_link_id")
            if workspace_link_id is not None:
                ds.attrs["manager_node_watched_workspace_link_id"] = str(
                    workspace_link_id
                )
            source_label = watched_metadata.get("source_label")
            if source_label is not None:
                ds.attrs["manager_node_watched_source_label"] = str(source_label)
            source_uid = watched_metadata.get("source_uid")
            if source_uid is not None:
                ds.attrs["manager_node_watched_source_uid"] = str(source_uid)
            ds.attrs["manager_node_watched_connected"] = bool(
                watched_metadata.get("connected", False)
            )
        output_id = persistence.output_id
        if kind == "imagetool" and output_id is not None:
            ds.attrs["manager_node_output_id"] = output_id
        source_spec = persistence.source_spec
        if kind == "imagetool" and source_spec is not None:
            ds.attrs["manager_node_live_source_spec"] = json.dumps(
                source_spec.model_dump(mode="json")
            )
        if kind == "imagetool" and (source_spec is not None or output_id is not None):
            ds.attrs["manager_node_source_state"] = persistence.source_state
            ds.attrs["manager_node_source_auto_update"] = bool(
                persistence.source_auto_update
            )
        return ds

    def _serialize_workspace_node(
        self,
        constructor: dict[str, xr.Dataset],
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: str,
        *,
        include_children: bool,
    ) -> None:
        if node.is_imagetool:
            target: int | str = (
                node.index if isinstance(node, _ImageToolWrapper) else node.uid
            )
            tool = self._manager.get_imagetool(target)
            ds = tool.to_dataset()
            ds.attrs["itool_title"] = node.name
            constructor[f"{path}/imagetool"] = self._annotate_workspace_dataset(
                ds,
                node,
                kind="imagetool",
            )
        else:
            if (
                node.pending_workspace_payload is not None
                and not node.materialize_pending_workspace_payload()
            ):
                raise ValueError("Could not read this saved tool from the workspace.")
            tool = typing.cast("erlab.interactive.utils.ToolWindow", node.tool_window)
            if not tool.can_save_and_load():
                return
            with tool._save_tool_data_reference_context(
                self._manager._tool_graph.nodes
            ):
                ds = tool.to_dataset()
            ds.attrs["tool_title"] = _strip_workspace_modified_placeholder(
                ds.attrs.get("tool_title", "")
            )
            constructor[f"{path}/tool"] = self._annotate_workspace_dataset(
                ds, node, kind="tool"
            )

        if not include_children:
            return
        for child_uid in node._childtool_indices:
            child = self._manager._child_node(child_uid)
            self._serialize_workspace_node(
                constructor,
                child,
                f"{path}/childtools/{child_uid}",
                include_children=include_children,
            )

    def _to_datatree(
        self, close: bool = False, include_children: bool = True
    ) -> xr.DataTree:
        """Convert the current state of the manager to a DataTree object."""
        constructor: dict[str, xr.Dataset] = {}
        for index in self._workspace_root_indices():
            self._serialize_workspace_node(
                constructor,
                self._manager._tool_graph.root_wrappers[index],
                str(index),
                include_children=include_children,
            )
            if close:
                self._manager.remove_imagetool(index)
        for uid in list(self._manager._tool_graph.figure_uids):
            node = self._manager._tool_graph.nodes.get(uid)
            if not isinstance(node, _ManagedWindowNode):
                continue
            self._serialize_workspace_node(
                constructor,
                node,
                f"figures/{uid}",
                include_children=False,
            )
            if close:
                self._manager._remove_childtool(uid)
        tree = xr.DataTree.from_dict(constructor)
        workspace_format._set_legacy_workspace_schema(tree.attrs)
        return tree

    def _workspace_node_path(self, uid: str) -> str:
        node = self._manager._tool_graph.nodes[uid]
        if isinstance(node, _ImageToolWrapper):
            return str(node.index)
        if self._manager._is_figure_node(node):
            return f"figures/{uid}"
        if node.parent_uid is None:
            raise KeyError(f"Node {uid!r} has no parent")
        return f"{self._workspace_node_path(node.parent_uid)}/childtools/{uid}"

    def _workspace_payload_path(self, uid: str) -> str:
        node = self._manager._tool_graph.nodes[uid]
        payload_name = "imagetool" if node.is_imagetool else "tool"
        return f"{self._workspace_node_path(uid)}/{payload_name}"

    def _workspace_root_indices(self) -> tuple[int, ...]:
        return self._manager._tool_graph.root_indices_for_workspace()

    def _workspace_link_metadata_by_uid(self) -> dict[str, tuple[int, bool]]:
        metadata: dict[str, tuple[int, bool]] = {}
        group_index = 0
        structural_groups: dict[str, tuple[list[str], bool]] = {}
        for uid, node in self._manager._tool_graph.nodes.items():
            link_key = node.workspace_link_key
            if link_key is None:
                continue
            group_nodes, _link_colors = structural_groups.setdefault(
                link_key, ([], node.workspace_link_colors)
            )
            group_nodes.append(uid)
        for group_nodes, link_colors in structural_groups.values():
            if len(group_nodes) <= 1:
                continue
            for uid in group_nodes:
                metadata[uid] = (group_index, link_colors)
            group_index += 1
        for linker in self._manager._link_registry.linkers:
            linked_nodes: list[_ImageToolWrapper | _ManagedWindowNode] = []
            for slicer_area in linker.children:
                node = self._manager.node_from_slicer_area(slicer_area)
                if (
                    node is None
                    or not node.is_imagetool
                    or node.imagetool is None
                    or node.slicer_area._linking_proxy is not linker
                ):
                    continue
                linked_nodes.append(node)
            if len(linked_nodes) <= 1:
                continue
            for node in linked_nodes:
                if node.uid in metadata:
                    continue
                metadata[node.uid] = (group_index, bool(linker.link_colors))
            group_index += 1
        return metadata

    def _workspace_node_manifest_entries(self) -> list[dict[str, typing.Any]]:
        entries: list[dict[str, typing.Any]] = []
        link_metadata = self._workspace_link_metadata_by_uid()

        def _append(uid: str) -> None:
            node = self._manager._tool_graph.nodes[uid]
            entry: dict[str, typing.Any] = {
                "uid": uid,
                # Payload group path relative to the workspace root HDF5 group.
                "path": self._workspace_node_path(uid),
                # Restores graph node type without probing payload attrs first.
                "kind": "imagetool" if node.is_imagetool else "tool",
                "parent_uid": node.parent_uid,
                "display_name": node.display_text,
            }
            if node.is_imagetool:
                # Distinguishes embedded data from lazy file-backed/dask payloads.
                entry["data_backing"] = node.persistence_data_backing()[0]
                link_info = link_metadata.get(uid)
                if link_info is not None:
                    # link_group is an ordinal within this manifest, not a stable id.
                    entry["link_group"], entry["link_colors"] = link_info
            entries.append(entry)
            for child_uid in node._childtool_indices:
                if child_uid in self._manager._tool_graph.nodes:
                    _append(child_uid)

        for index in self._workspace_root_indices():
            _append(self._manager._tool_graph.root_wrappers[index].uid)
        for uid in self._manager._tool_graph.figure_uids:
            if uid in self._manager._tool_graph.nodes:
                _append(uid)
        return entries

    def _workspace_root_attrs_payload(
        self,
        *,
        delta_save_count: int | None = None,
        estimated_obsolete_bytes: int | None = None,
        replacement_delta_count: int | None = None,
        repack_estimate_known: bool | None = None,
    ) -> dict[str, typing.Any]:
        state = self._manager._workspace_state
        if delta_save_count is None:
            delta_save_count = state.delta_save_count
        if estimated_obsolete_bytes is None:
            estimated_obsolete_bytes = state.estimated_obsolete_bytes
        if replacement_delta_count is None:
            replacement_delta_count = state.replacement_delta_count
        if repack_estimate_known is None:
            repack_estimate_known = state.repack_estimate_known
        return workspace_format._workspace_root_attrs_payload(
            root_order=self._workspace_root_indices(),
            nodes=self._workspace_node_manifest_entries(),
            delta_save_count=delta_save_count,
            erlab_version=str(erlab.__version__),
            workspace_link_id=self._manager._workspace_state.link_id,
            manager_layout=self._workspace_layout_snapshot(),
            loader_state=self._workspace_loader_state_snapshot(),
            standalone_apps=self._workspace_standalone_apps_snapshot(),
            option_overrides=self._workspace_option_overrides_snapshot(),
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            repack_estimate_known=repack_estimate_known,
        )

    def _workspace_compression_mode(self) -> WorkspaceCompressionMode:
        return self._manager.effective_interactive_options.io.workspace.compression

    def _workspace_layout_snapshot(self) -> dict[str, typing.Any]:
        return {
            "window_state": _qt_state.qt_window_state_payload(self._manager),
            # QSplitter state preserves pane sizes and collapsed/expanded handles.
            "main_splitter": erlab.interactive.utils._qt_bytearray_to_base64(
                self._manager.main_splitter.saveState()
            ),
            "right_splitter": erlab.interactive.utils._qt_bytearray_to_base64(
                self._manager.right_splitter.saveState()
            ),
        }

    def _workspace_option_overrides_snapshot(self) -> dict[str, typing.Any]:
        return workspace_format.WorkspaceOptionOverridesState(
            overrides=erlab.interactive._options.core.normalize_workspace_option_overrides(
                self._manager._workspace_state.option_overrides
            )
        ).model_dump(mode="json")

    def _workspace_loader_state_snapshot(self) -> dict[str, typing.Any]:
        manager_loader_kwargs = self._manager._recent_loader_kwargs_by_filter
        manager_loader_extensions = self._manager._recent_loader_extensions_by_filter
        explorer_kwargs = self._controller._loader_state.explorer_loader_kwargs_by_name
        explorer_extensions = (
            self._controller._loader_state.explorer_loader_extensions_by_name
        )
        explorer = self._manager._standalone_app_windows.get("explorer")
        if explorer is not None and erlab.interactive.utils.qt_is_valid(explorer):
            kwargs_getter = getattr(explorer, "loader_kwargs_by_name", None)
            if callable(kwargs_getter):
                explorer_kwargs = kwargs_getter()
            extensions_getter = getattr(explorer, "loader_extensions_by_name", None)
            if callable(extensions_getter):
                explorer_extensions = extensions_getter()
        state = workspace_format.WorkspaceLoaderState(
            recent_directory=self._manager._recent_directory,
            recent_name_filter=self._manager._recent_name_filter,
            manager_loader_kwargs_by_filter={
                str(name): dict(kwargs)
                for name, kwargs in manager_loader_kwargs.items()
            },
            manager_loader_extensions_by_filter={
                str(name): dict(extensions)
                for name, extensions in manager_loader_extensions.items()
            },
            explorer_loader_kwargs_by_name={
                str(name): dict(kwargs) for name, kwargs in explorer_kwargs.items()
            },
            explorer_loader_extensions_by_name={
                str(name): dict(extensions)
                for name, extensions in explorer_extensions.items()
            },
        )
        self._controller._loader_state = state
        return state.model_dump(mode="json", exclude_none=True)

    def _workspace_standalone_apps_snapshot(self) -> dict[str, typing.Any]:
        app_states: dict[str, dict[str, typing.Any]] = {}
        for key in self._manager._standalone_app_specs:
            widget = self._manager._standalone_app_windows.get(key)
            state: dict[str, typing.Any] | None = None
            if widget is not None and erlab.interactive.utils.qt_is_valid(widget):
                state_getter = getattr(widget, "workspace_state_payload", None)
                if callable(state_getter):
                    state = typing.cast("dict[str, typing.Any]", state_getter())
            elif key in self._manager._standalone_app_pending_states:
                state = self._manager._standalone_app_pending_states[key]
            if state is None:
                continue
            validated = self._controller._validated_standalone_app_state(key, state)
            if validated is not None:
                app_states[key] = validated
        return workspace_format.StandaloneAppsState(apps=app_states).model_dump(
            mode="json", exclude_none=True
        )

    def _workspace_datatree_for_payload_uids(self, uids: Iterable[str]) -> xr.DataTree:
        constructor: dict[str, xr.Dataset] = {}
        for uid in sorted(set(uids), key=self._workspace_node_path):
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None:
                continue
            self._serialize_workspace_node(
                constructor,
                node,
                self._workspace_node_path(uid),
                include_children=False,
            )
        tree = xr.DataTree.from_dict(constructor)
        workspace_format._set_legacy_workspace_schema(tree.attrs)
        return tree

    def _serialize_workspace_node_for_full_save_fallback(
        self,
        constructor: dict[str, xr.Dataset],
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: str,
        copy_group_sources: list[tuple[str, str, str, dict[str, typing.Any] | None]],
        *,
        include_children: bool,
        pending_copy_allowed: Callable[[tuple[str | os.PathLike[str], str]], bool],
    ) -> None:
        pending_payload = node.pending_workspace_payload
        pending_kind = node.pending_workspace_payload_kind
        if (
            pending_payload is not None
            and pending_kind is not None
            and node.uid not in self._manager._workspace_state.dirty_data
            and (
                pending_kind != "tool"
                or self._pending_workspace_tool_references_available(node)
            )
            and pending_copy_allowed(pending_payload)
        ):
            source_file, source_path = pending_payload
            copy_group_sources.append(
                (
                    os.fsdecode(source_file),
                    source_path,
                    f"{path}/{pending_kind}",
                    self._pending_workspace_payload_attrs_for_save(node),
                )
            )
        else:
            self._serialize_workspace_node(
                constructor,
                node,
                path,
                include_children=False,
            )

        if not include_children:
            return
        for child_uid in node._childtool_indices:
            child = self._manager._child_node(child_uid)
            self._serialize_workspace_node_for_full_save_fallback(
                constructor,
                child,
                f"{path}/childtools/{child_uid}",
                copy_group_sources,
                include_children=True,
                pending_copy_allowed=pending_copy_allowed,
            )

    def _workspace_full_save_fallback_tree(
        self,
        *,
        require_matching_compression: bool,
        compression_mode: WorkspaceCompressionMode,
    ) -> tuple[
        xr.DataTree,
        tuple[tuple[str, str, str, dict[str, typing.Any] | None], ...],
    ]:
        constructor: dict[str, xr.Dataset] = {}
        copy_group_sources: list[
            tuple[str, str, str, dict[str, typing.Any] | None]
        ] = []
        with contextlib.ExitStack() as stack:
            pending_compression_cache: dict[tuple[str, str], bool] = {}
            pending_compression_files: dict[str, typing.Any] = {}

            def _pending_copy_allowed(
                pending_payload: tuple[str | os.PathLike[str], str],
            ) -> bool:
                if not require_matching_compression:
                    return True
                source_file, source_path = pending_payload
                source_key = os.fsdecode(source_file)
                source_path = source_path.strip("/")
                cache_key = (source_key, source_path)
                if cache_key in pending_compression_cache:
                    return pending_compression_cache[cache_key]
                try:
                    h5_file = pending_compression_files.get(source_key)
                    if h5_file is None:
                        workspace_arrays.ensure_workspace_hdf5_filters_registered()
                        stack.enter_context(
                            workspace_arrays._workspace_file_lock(source_key)
                        )
                        h5_file = stack.enter_context(h5py.File(source_key, "r"))
                        pending_compression_files[source_key] = h5_file
                    matches = workspace_arrays._h5_group_matches_compression(
                        h5_file,
                        source_path,
                        compression_mode,
                    )
                except Exception:
                    logger.debug(
                        "Cannot verify pending workspace payload compression",
                        exc_info=True,
                    )
                    matches = False
                pending_compression_cache[cache_key] = matches
                return matches

            for index in self._workspace_root_indices():
                self._serialize_workspace_node_for_full_save_fallback(
                    constructor,
                    self._manager._tool_graph.root_wrappers[index],
                    str(index),
                    copy_group_sources,
                    include_children=True,
                    pending_copy_allowed=_pending_copy_allowed,
                )
            for uid in list(self._manager._tool_graph.figure_uids):
                node = self._manager._tool_graph.nodes.get(uid)
                if not isinstance(node, _ManagedWindowNode):
                    continue
                self._serialize_workspace_node_for_full_save_fallback(
                    constructor,
                    node,
                    f"figures/{uid}",
                    copy_group_sources,
                    include_children=False,
                    pending_copy_allowed=_pending_copy_allowed,
                )
        tree = xr.DataTree.from_dict(constructor)
        workspace_format._set_legacy_workspace_schema(tree.attrs)
        return tree, tuple(copy_group_sources)

    def _workspace_full_save_source_identities(
        self,
    ) -> tuple[pathlib.Path, dict[tuple[str, str], str]] | None:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return None
        workspace_path = pathlib.Path(workspace_path)
        if (
            self._manager._workspace_state.schema_version
            != workspace_format._current_workspace_schema_version()
            or not workspace_path.exists()
        ):
            return None

        try:
            root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(
                workspace_path
            )
        except Exception:
            return None
        schema_version, _delta_save_count, manifest = (
            workspace_format._workspace_file_metadata_from_attrs(root_attrs)
        )
        if (
            schema_version != workspace_format._current_workspace_schema_version()
            or manifest is None
        ):
            return None

        manifest_entries = workspace_format._workspace_manifest_payload_entries(
            manifest
        )
        identities = {
            (uid, kind): payload_path for uid, kind, payload_path in manifest_entries
        }
        if not identities:
            return None
        return workspace_path, identities

    def _workspace_full_save_manifest_entries(
        self, root_attrs: Mapping[str, typing.Any]
    ) -> list[tuple[str, str, str]]:
        manifest = workspace_format._workspace_manifest_from_attrs(
            typing.cast("Mapping[Hashable, typing.Any]", root_attrs)
        )
        return workspace_format._workspace_manifest_payload_entries(manifest)

    def _workspace_full_save_dirty_payload_uids(self) -> set[str]:
        state = self._manager._workspace_state
        serialize_uids: set[str] = set()
        for uid in state.dirty_added | state.dirty_data:
            if uid not in self._manager._tool_graph.nodes:
                continue
            serialize_uids.add(uid)
            serialize_uids.update(self._manager._iter_descendant_uids(uid))
        serialize_uids.update(
            uid for uid in state.dirty_state if uid in self._manager._tool_graph.nodes
        )
        return serialize_uids

    def _workspace_full_save_manifest_first_snapshot(
        self,
        generation: int,
        fname: str | os.PathLike[str],
        root_attrs: dict[str, typing.Any],
        *,
        compression_mode: WorkspaceCompressionMode,
        require_matching_compression: bool,
    ) -> _WorkspaceSaveSnapshot | None:
        if _workspace_path_is_likely_network_path(fname):
            return None
        source = self._workspace_full_save_source_identities()
        if source is None:
            return None
        workspace_path, identities = source

        copy_groups: list[tuple[str, str, dict[str, typing.Any] | None]] = []
        copy_group_sources: list[
            tuple[str, str, str, dict[str, typing.Any] | None]
        ] = []
        serialize_uids = self._workspace_full_save_dirty_payload_uids()
        reason_counts: collections.Counter[str] = collections.Counter()
        copied_bytes = 0
        serialized_existing_bytes = 0
        plan_started_at = time.perf_counter()
        try:
            workspace_arrays.ensure_workspace_hdf5_filters_registered()
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    workspace_arrays._workspace_file_lock(workspace_path)
                )
                h5_file = stack.enter_context(h5py.File(workspace_path, "r"))
                pending_compression_cache: dict[tuple[str, str], bool] = {}
                pending_compression_files: dict[str, typing.Any] = {}

                def _pending_copy_allowed(
                    pending_payload: tuple[str | os.PathLike[str], str],
                ) -> bool:
                    if not require_matching_compression:
                        return True
                    source_file, source_path = pending_payload
                    source_key = os.fsdecode(source_file)
                    source_path = source_path.strip("/")
                    cache_key = (source_key, source_path)
                    if cache_key in pending_compression_cache:
                        return pending_compression_cache[cache_key]
                    try:
                        pending_h5_file = pending_compression_files.get(source_key)
                        if pending_h5_file is None:
                            stack.enter_context(
                                workspace_arrays._workspace_file_lock(source_key)
                            )
                            pending_h5_file = stack.enter_context(
                                h5py.File(source_key, "r")
                            )
                            pending_compression_files[source_key] = pending_h5_file
                        matches = workspace_arrays._h5_group_matches_compression(
                            pending_h5_file,
                            source_path,
                            compression_mode,
                        )
                    except Exception:
                        logger.debug(
                            "Cannot verify pending workspace payload compression",
                            exc_info=True,
                        )
                        matches = False
                    pending_compression_cache[cache_key] = matches
                    return matches

                for (
                    uid,
                    kind,
                    payload_path,
                ) in self._workspace_full_save_manifest_entries(root_attrs):
                    node = self._manager._tool_graph.nodes.get(uid)
                    if node is None:
                        reason_counts["missing_node"] += 1
                        continue
                    if not node.is_imagetool:
                        if node.pending_workspace_tool_payload is not None:
                            if not self._pending_workspace_tool_references_available(
                                node
                            ):
                                serialize_uids.add(uid)
                                reason_counts["stale_tool_reference"] += 1
                        else:
                            tool = node.tool_window
                            if tool is None or not tool.can_save_and_load():
                                reason_counts["unsupported_tool"] += 1
                                continue
                    pending_payload = node.pending_workspace_payload
                    if (
                        pending_payload is not None
                        and node.pending_workspace_payload_kind == kind
                        and uid not in self._manager._workspace_state.dirty_data
                        and (
                            kind != "tool"
                            or self._pending_workspace_tool_references_available(node)
                        )
                    ):
                        source_path_for_pending = identities.get((uid, kind))
                        pending_workspace_path, pending_payload_path = pending_payload
                        pending_source_matches = (
                            source_path_for_pending is not None
                            and workspace_arrays._normalized_file_path(
                                pending_workspace_path
                            )
                            == workspace_arrays._normalized_file_path(workspace_path)
                            and pending_payload_path
                            == source_path_for_pending.strip("/")
                        )
                        pending_external_copy = (
                            uid in self._manager._workspace_state.dirty_added
                            or not pending_source_matches
                        )
                        if pending_external_copy:
                            if _pending_copy_allowed(pending_payload):
                                pending_source_file, pending_source_path = (
                                    pending_payload
                                )
                                copy_group_sources.append(
                                    (
                                        os.fsdecode(pending_source_file),
                                        pending_source_path,
                                        payload_path,
                                        self._pending_workspace_payload_attrs_for_save(
                                            node
                                        ),
                                    )
                                )
                                serialize_uids.discard(uid)
                                reason_counts["pending_external"] += 1
                                continue
                            serialize_uids.add(uid)
                            reason_counts["pending_compression_mismatch"] += 1
                            continue
                    source_path = identities.get((uid, kind))
                    if source_path is None or source_path != payload_path:
                        serialize_uids.add(uid)
                        reason_counts[
                            "missing_source" if source_path is None else "moved"
                        ] += 1
                        continue
                    source_group = h5_file.get(source_path)
                    if not isinstance(source_group, h5py.Group):
                        serialize_uids.add(uid)
                        reason_counts["missing_source_group"] += 1
                        continue
                    source_storage_size = (
                        workspace_arrays._workspace_h5_object_storage_size(source_group)
                    )
                    compression_mismatch = (
                        require_matching_compression
                        and not workspace_arrays._h5_group_matches_compression(
                            h5_file, source_path, compression_mode
                        )
                    )
                    pending_payload = node.pending_workspace_payload
                    pending_source_matches = False
                    if pending_payload is not None:
                        pending_workspace_path, pending_payload_path = pending_payload
                        pending_source_matches = workspace_arrays._normalized_file_path(
                            pending_workspace_path
                        ) == workspace_arrays._normalized_file_path(
                            workspace_path
                        ) and pending_payload_path == source_path.strip("/")
                    if compression_mismatch:
                        serialize_uids.add(uid)
                        reason_counts["compression_mismatch"] += 1
                        serialized_existing_bytes += source_storage_size
                        continue
                    if (
                        pending_source_matches
                        and uid in self._manager._workspace_state.dirty_state
                        and uid not in self._manager._workspace_state.dirty_data
                        and uid not in self._manager._workspace_state.dirty_added
                    ):
                        serialize_uids.discard(uid)
                        update = self._workspace_attr_update_snapshot(uid)
                        copy_groups.append(
                            (
                                source_path,
                                payload_path,
                                None if update is None else update[1],
                            )
                        )
                        reason_counts["pending_dirty_state"] += 1
                        copied_bytes += source_storage_size
                        continue
                    if uid in serialize_uids:
                        state = self._manager._workspace_state
                        if uid in state.dirty_added:
                            reason_counts["dirty_added"] += 1
                        elif uid in state.dirty_data:
                            reason_counts["dirty_data"] += 1
                        elif uid in state.dirty_state:
                            reason_counts["dirty_state"] += 1
                        else:
                            reason_counts["dirty_descendant"] += 1
                        serialized_existing_bytes += source_storage_size
                        continue
                    copy_groups.append((source_path, payload_path, None))
                    copied_bytes += source_storage_size
        except Exception:
            logger.debug(
                "Falling back to DataTree full-save snapshot",
                exc_info=True,
            )
            return None

        tree = self._workspace_datatree_for_payload_uids(serialize_uids)
        logger.debug(
            "Workspace manifest-first full-save plan: copied %d groups "
            "(%.1f MiB), serialized %d existing groups (%.1f MiB), "
            "reasons=%s, planning %.3f s",
            len(copy_groups),
            copied_bytes / 1024**2,
            len(serialize_uids),
            serialized_existing_bytes / 1024**2,
            dict(reason_counts),
            time.perf_counter() - plan_started_at,
        )
        return _WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=0,
            compression_mode=compression_mode,
            full_tree=tree,
            copy_source=str(workspace_path),
            copy_groups=tuple(copy_groups),
            copy_group_sources=tuple(copy_group_sources),
        )

    def _write_full_workspace_file(
        self,
        fname: str | os.PathLike[str],
        *,
        reuse_unchanged_groups: bool = True,
        require_matching_compression: bool = False,
    ) -> None:
        snapshot = self._workspace_full_save_snapshot(
            self._manager._workspace_state.dirty_generation,
            fname=fname,
            reuse_unchanged_groups=reuse_unchanged_groups,
            require_matching_compression=require_matching_compression,
        )
        try:
            _write_full_workspace_tree_file(
                fname,
                snapshot.full_tree,
                snapshot.root_attrs,
                copy_source=snapshot.copy_source,
                copy_groups=snapshot.copy_groups,
                copy_group_sources=snapshot.copy_group_sources,
                compression_mode=snapshot.compression_mode,
            )
        finally:
            snapshot.close()

    def _workspace_highest_dirty_data_roots(self) -> list[str]:
        dirty_existing = [
            uid
            for uid in self._manager._workspace_state.dirty_data
            if uid in self._manager._tool_graph.nodes
        ]
        dirty_set = set(dirty_existing)
        roots: list[str] = []
        for uid in sorted(
            dirty_existing, key=lambda value: self._workspace_node_path(value)
        ):
            node = self._manager._tool_graph.nodes[uid]
            parent_uid = node.parent_uid
            has_dirty_ancestor = False
            while parent_uid is not None:
                if parent_uid in dirty_set:
                    has_dirty_ancestor = True
                    break
                parent_uid = self._manager._tool_graph.nodes[parent_uid].parent_uid
            if not has_dirty_ancestor:
                roots.append(uid)
        return roots

    @classmethod
    def _workspace_manifest_node_uids(
        cls, root_attrs: Mapping[str, typing.Any]
    ) -> frozenset[str]:
        manifest = workspace_format._workspace_manifest_from_attrs(
            typing.cast("Mapping[Hashable, typing.Any]", root_attrs)
        )
        uids: set[str] = set()
        for node in workspace_format._iter_workspace_manifest_node_entries(manifest):
            uid = node.get("uid")
            if uid is not None:
                uids.add(str(uid))
        return frozenset(uids)

    def _workspace_stale_reference_rewrite_uids(
        self, available_uids: frozenset[str]
    ) -> list[str]:
        rewrite_uids: list[str] = []
        for uid, node in self._manager._tool_graph.nodes.items():
            if node.is_imagetool:
                continue
            if node.pending_workspace_tool_payload is not None:
                if not self._pending_workspace_tool_references_available(node):
                    rewrite_uids.append(uid)
                continue
            tool = typing.cast("erlab.interactive.utils.ToolWindow", node.tool_window)
            if tool is None:
                continue
            if not tool.can_save_and_load():
                continue
            if tool._persistence_reference_node_uids() - available_uids:
                rewrite_uids.append(uid)
        return sorted(rewrite_uids, key=self._workspace_node_path)

    def _save_workspace_delta(self, fname: str | os.PathLike[str]) -> None:
        delta_save_count = self._manager._workspace_state.delta_save_count + 1
        snapshot = self._workspace_delta_save_snapshot(
            self._manager._workspace_state.dirty_generation,
            self._workspace_root_attrs_payload(delta_save_count=delta_save_count),
            delta_save_count,
        )
        try:
            _write_workspace_transaction_file(
                fname,
                snapshot.rewrite_groups,
                snapshot.attr_updates,
                snapshot.root_attrs,
                compression_mode=snapshot.compression_mode,
            )
            self._manager._workspace_state.delta_save_count = snapshot.delta_save_count
            self._manager._workspace_state.set_repack_estimate(
                estimated_obsolete_bytes=snapshot.estimated_obsolete_bytes,
                replacement_delta_count=snapshot.replacement_delta_count,
                known=snapshot.repack_estimate_known,
            )
        finally:
            snapshot.close()

    def _save_workspace_document(
        self,
        fname: str | os.PathLike[str],
        *,
        force_full: bool = False,
        document_access: _WorkspaceDocumentAccess | None = None,
        reuse_unchanged_groups: bool = True,
        require_matching_compression: bool = False,
        mark_clean: bool = True,
    ) -> None:
        if document_access is None:
            _require_itws_workspace_path(fname, _WORKSPACE_SAVE_SUFFIX_ERROR)
            with self._controller._workspace_document_access_context(fname) as access:
                self._save_workspace_document(
                    access.path,
                    force_full=force_full,
                    document_access=access,
                    reuse_unchanged_groups=reuse_unchanged_groups,
                    require_matching_compression=require_matching_compression,
                    mark_clean=mark_clean,
                )
            return

        fname = document_access.path
        _require_itws_workspace_path(fname, _WORKSPACE_SAVE_SUFFIX_ERROR)
        self._manager._workspace_state.saving_depth += 1
        try:
            _recover_workspace_transactions(fname)
            requires_full_save = force_full or self._workspace_requires_full_save(fname)
            if requires_full_save:
                self._write_full_workspace_file(
                    fname,
                    reuse_unchanged_groups=reuse_unchanged_groups,
                    require_matching_compression=require_matching_compression,
                )
                self._manager._workspace_state.delta_save_count = 0
                self._manager._workspace_state.reset_repack_estimate()
                self._manager._workspace_state.schema_version = (
                    workspace_format._current_workspace_schema_version()
                )
            else:
                self._save_workspace_delta(fname)
        finally:
            self._manager._workspace_state.saving_depth -= 1
        if mark_clean:
            self._manager._workspace_state.needs_full_save = False
            self._controller._mark_workspace_clean()

    def _workspace_requires_full_save(self, fname: str | os.PathLike[str]) -> bool:
        return _workspace_requires_full_save(
            fname,
            needs_full_save=self._manager._workspace_state.needs_full_save,
            schema_version=self._manager._workspace_state.schema_version,
            structure_modified=self._manager._workspace_state.structure_modified,
            has_dirty_added=bool(self._manager._workspace_state.dirty_added),
            has_dirty_removed=bool(self._manager._workspace_state.dirty_removed),
        )

    def _workspace_has_non_layout_modifications(self) -> bool:
        state = self._manager._workspace_state
        return (
            state.structure_modified
            or bool(state.dirty_added)
            or bool(state.dirty_data)
            or bool(state.dirty_state)
            or bool(state.dirty_removed)
        )

    def _workspace_layout_only_modified(self) -> bool:
        return (
            self._manager._workspace_state.layout_modified
            or self._manager._workspace_state.options_modified
        ) and not self._workspace_has_non_layout_modifications()

    def _workspace_rewrite_group_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, xr.Dataset]]:
        constructor: dict[str, xr.Dataset] = {}
        node = self._manager._tool_graph.nodes[uid]
        node_path = self._workspace_node_path(uid)
        self._serialize_workspace_node(
            constructor, node, node_path, include_children=True
        )
        return node_path, constructor

    @staticmethod
    def _pending_workspace_node_attrs(
        node: _ImageToolWrapper | _ManagedWindowNode,
        attrs: Mapping[str, typing.Any] | None,
        *,
        kind: typing.Literal["imagetool", "tool"],
    ) -> dict[str, typing.Any]:
        if attrs is None:
            attrs = {}
        attrs = dict(attrs)
        attrs["manager_node_uid"] = node.uid
        attrs["manager_node_kind"] = kind
        attrs["manager_node_snapshot_token"] = node.snapshot_token
        attrs["manager_node_source_snapshot_token"] = node.source_snapshot_token
        attrs["manager_node_added_at"] = node.added_time_iso
        if node.note:
            attrs["manager_node_note"] = node.note
        else:
            attrs.pop("manager_node_note", None)
        return attrs

    def _pending_workspace_imagetool_attrs(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> dict[str, typing.Any]:
        attrs = self._pending_workspace_node_attrs(
            node, node.pending_workspace_payload_attrs, kind="imagetool"
        )
        attrs["itool_name"] = node.name
        attrs["itool_title"] = node.name

        provenance_spec = node.provenance_spec
        if provenance_spec is None:
            attrs.pop("manager_node_provenance_spec", None)
        else:
            attrs["manager_node_provenance_spec"] = json.dumps(
                provenance_spec.model_dump(mode="json")
            )

        if isinstance(node, _ImageToolWrapper):
            if node.source_input_ndim is None:
                attrs.pop("manager_node_source_input_ndim", None)
            else:
                attrs["manager_node_source_input_ndim"] = int(node.source_input_ndim)
            if node.watched:
                watched_metadata = node.watched_metadata()
                attrs["manager_node_watched_varname"] = watched_metadata["varname"]
                attrs["manager_node_watched_uid"] = watched_metadata["uid"]
                workspace_link_id = watched_metadata.get("workspace_link_id")
                if workspace_link_id is None:
                    attrs.pop("manager_node_watched_workspace_link_id", None)
                else:
                    attrs["manager_node_watched_workspace_link_id"] = str(
                        workspace_link_id
                    )
                source_label = watched_metadata.get("source_label")
                if source_label is None:
                    attrs.pop("manager_node_watched_source_label", None)
                else:
                    attrs["manager_node_watched_source_label"] = str(source_label)
                source_uid = watched_metadata.get("source_uid")
                if source_uid is None:
                    attrs.pop("manager_node_watched_source_uid", None)
                else:
                    attrs["manager_node_watched_source_uid"] = str(source_uid)
                attrs["manager_node_watched_connected"] = bool(
                    watched_metadata.get("connected", False)
                )
            else:
                for key in (
                    "manager_node_watched_varname",
                    "manager_node_watched_uid",
                    "manager_node_watched_workspace_link_id",
                    "manager_node_watched_source_label",
                    "manager_node_watched_source_uid",
                    "manager_node_watched_connected",
                ):
                    attrs.pop(key, None)

        output_id = node.output_id
        if output_id is None:
            attrs.pop("manager_node_output_id", None)
        else:
            attrs["manager_node_output_id"] = output_id

        source_spec = node.source_spec
        if source_spec is None:
            attrs.pop("manager_node_live_source_spec", None)
        else:
            attrs["manager_node_live_source_spec"] = json.dumps(
                source_spec.model_dump(mode="json")
            )
        if source_spec is None and output_id is None:
            attrs.pop("manager_node_source_state", None)
            attrs.pop("manager_node_source_auto_update", None)
        else:
            attrs["manager_node_source_state"] = node.source_state
            attrs["manager_node_source_auto_update"] = bool(node.source_auto_update)
        return attrs

    def _pending_workspace_tool_attrs(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> dict[str, typing.Any]:
        attrs = self._pending_workspace_node_attrs(
            node, node.pending_workspace_payload_attrs, kind="tool"
        )
        old_display_name = workspace_format._decode_workspace_attr_text(
            attrs.get("tool_display_name")
        )
        old_title = workspace_format._decode_workspace_attr_text(
            attrs.get("tool_title")
        )
        attrs["tool_display_name"] = node.name
        if old_title and old_display_name and old_title.endswith(old_display_name):
            attrs["tool_title"] = old_title[: -len(old_display_name)] + node.name
        else:
            attrs["tool_title"] = node.name

        source_spec = node.source_spec
        if source_spec is None:
            attrs.pop(erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR, None)
        else:
            attrs[erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR] = json.dumps(
                source_spec.model_dump(mode="json")
            )
        source_binding = node.source_binding
        if source_spec is not None or source_binding is None:
            attrs.pop(erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR, None)
        else:
            attrs[erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR] = json.dumps(
                source_binding.model_dump(mode="json")
            )
        if node.has_source_binding:
            attrs[erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR] = node.source_state
            attrs[erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR] = bool(
                node.source_auto_update
            )
        else:
            attrs.pop(erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR, None)
            attrs.pop(erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR, None)
        return attrs

    def _pending_workspace_payload_attrs_for_save(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> dict[str, typing.Any] | None:
        match node.pending_workspace_payload_kind:
            case "imagetool":
                return self._pending_workspace_imagetool_attrs(node)
            case "tool":
                return self._pending_workspace_tool_attrs(node)
            case _:
                return None

    def _pending_workspace_tool_references_available(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> bool:
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            return True
        payload = attrs.get(erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR)
        if payload is None:
            return True
        if isinstance(payload, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                payload = payload.decode()
        if not isinstance(payload, str):
            return False
        try:
            references = json.loads(payload)
        except Exception:
            return False
        if not isinstance(references, dict):
            return False
        for reference in references.values():
            if not isinstance(reference, dict):
                return False
            kind = reference.get("kind")
            if kind == "parent_source":
                if (
                    node.parent_uid is None
                    or node.parent_uid not in self._manager._tool_graph.nodes
                ):
                    return False
                continue
            if kind != "manager_node":
                return False
            referenced_uid = reference.get("node_uid")
            if (
                not isinstance(referenced_uid, str)
                or not referenced_uid
                or referenced_uid not in self._manager._tool_graph.nodes
            ):
                return False
        return True

    def _workspace_attr_update_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]] | None:
        constructor: dict[str, xr.Dataset] = {}
        node = self._manager._tool_graph.nodes[uid]
        node_path = self._workspace_node_path(uid)
        payload_path = self._workspace_payload_path(uid)
        pending_attrs = self._pending_workspace_payload_attrs_for_save(node)
        if pending_attrs is not None:
            return (
                payload_path,
                pending_attrs,
                (node_path, constructor),
            )
        self._serialize_workspace_node(
            constructor,
            node,
            node_path,
            include_children=False,
        )
        ds = constructor.get(payload_path)
        if ds is None:
            return None
        return payload_path, dict(ds.attrs), (node_path, constructor)

    def _workspace_delta_save_snapshot(
        self,
        generation: int,
        root_attrs: dict[str, typing.Any],
        delta_save_count: int,
    ) -> _WorkspaceSaveSnapshot:
        state = self._manager._workspace_state
        rewrite_groups: list[tuple[str, dict[str, xr.Dataset]]] = []
        rewritten_uids: set[str] = set()
        for uid in self._workspace_highest_dirty_data_roots():
            rewrite_groups.append(self._workspace_rewrite_group_snapshot(uid))
            rewritten_uids.add(uid)
            rewritten_uids.update(self._manager._iter_descendant_uids(uid))

        manifest_uids = self._workspace_manifest_node_uids(root_attrs)
        for uid in self._workspace_stale_reference_rewrite_uids(manifest_uids):
            if uid in rewritten_uids:
                continue
            rewrite_groups.append(self._workspace_rewrite_group_snapshot(uid))
            rewritten_uids.add(uid)
            rewritten_uids.update(self._manager._iter_descendant_uids(uid))

        attr_updates: list[
            tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
        ] = []
        for uid in sorted(self._manager._workspace_state.dirty_state - rewritten_uids):
            if uid not in self._manager._tool_graph.nodes:
                continue
            update = self._workspace_attr_update_snapshot(uid)
            if update is not None:
                attr_updates.append(update)

        estimated_obsolete_bytes = state.estimated_obsolete_bytes
        replacement_delta_count = state.replacement_delta_count
        repack_estimate_known = state.repack_estimate_known
        if repack_estimate_known and rewrite_groups and state.path is not None:
            old_bytes, replaced_group_count = (
                workspace_arrays._workspace_h5_paths_storage_size(
                    state.path,
                    (group_path for group_path, _constructor in rewrite_groups),
                )
            )
            estimated_obsolete_bytes += old_bytes
            if replaced_group_count > 0:
                replacement_delta_count += 1
        root_attrs = workspace_format._workspace_root_attrs_with_repack_estimate(
            root_attrs,
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            repack_estimate_known=repack_estimate_known,
        )

        return _WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=delta_save_count,
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            repack_estimate_known=repack_estimate_known,
            compression_mode=self._workspace_compression_mode(),
            rewrite_groups=tuple(rewrite_groups),
            attr_updates=tuple(attr_updates),
        )

    def _workspace_save_snapshot(
        self, fname: str | os.PathLike[str]
    ) -> _WorkspaceSaveSnapshot:
        self._controller._drain_workspace_deferred_events()
        generation = self._manager._workspace_state.dirty_generation
        self._manager._workspace_state.saving_depth += 1
        try:
            if self._workspace_requires_full_save(fname):
                return self._workspace_full_save_snapshot(generation)
            if self._workspace_layout_only_modified():
                delta_save_count = self._manager._workspace_state.delta_save_count
                root_attrs = self._workspace_root_attrs_payload(
                    delta_save_count=delta_save_count
                )
                return self._workspace_delta_save_snapshot(
                    generation, root_attrs, delta_save_count
                )
            delta_save_count = self._manager._workspace_state.delta_save_count + 1
            root_attrs = self._workspace_root_attrs_payload(
                delta_save_count=delta_save_count
            )
            return self._workspace_delta_save_snapshot(
                generation, root_attrs, delta_save_count
            )
        finally:
            self._manager._workspace_state.saving_depth -= 1

    def _workspace_full_save_snapshot(
        self,
        generation: int,
        *,
        fname: str | os.PathLike[str] | None = None,
        reuse_unchanged_groups: bool = True,
        require_matching_compression: bool = False,
    ) -> _WorkspaceSaveSnapshot:
        compression_mode = self._workspace_compression_mode()
        root_attrs = self._workspace_root_attrs_payload(delta_save_count=0)
        if fname is None:
            fname = self._manager._workspace_state.path
        target_drops_copy_groups = (
            fname is not None and _workspace_path_is_likely_network_path(fname)
        )
        if (
            reuse_unchanged_groups
            and fname is not None
            and not target_drops_copy_groups
        ):
            snapshot = self._workspace_full_save_manifest_first_snapshot(
                generation,
                fname,
                root_attrs,
                compression_mode=compression_mode,
                require_matching_compression=require_matching_compression,
            )
            if snapshot is not None:
                return snapshot

        tree, pending_copy_groups = self._workspace_full_save_fallback_tree(
            require_matching_compression=require_matching_compression,
            compression_mode=compression_mode,
        )
        if reuse_unchanged_groups and not target_drops_copy_groups:
            copy_source, copy_groups = self._workspace_full_save_copy_groups(
                tree,
                compression_mode=compression_mode,
                require_matching_compression=require_matching_compression,
            )
        else:
            copy_source, copy_groups = None, ()
        return _WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=0,
            compression_mode=compression_mode,
            full_tree=tree,
            copy_source=copy_source,
            copy_groups=copy_groups,
            copy_group_sources=pending_copy_groups,
        )

    def _workspace_file_repack_snapshot(
        self, generation: int
    ) -> _WorkspaceSaveSnapshot | None:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return None
        if _workspace_path_is_likely_network_path(workspace_path):
            return None
        try:
            root_attrs, copy_groups = _workspace_file_repack_payload(workspace_path)
        except Exception:
            logger.debug(
                "Skipping shutdown compaction; file-level repack snapshot failed",
                exc_info=True,
            )
            return None
        return _WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=0,
            compression_mode=self._workspace_compression_mode(),
            file_repack=True,
            copy_source=str(workspace_path),
            copy_groups=copy_groups,
        )

    def _workspace_should_repack_before_shutdown(self) -> bool:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return False
        state = self._manager._workspace_state
        if state.repack_estimate_known:
            if state.replacement_delta_count <= 0:
                return False
            estimated_obsolete_bytes = state.estimated_obsolete_bytes
        else:
            try:
                estimated_obsolete_bytes = _workspace_obsolete_estimate(workspace_path)
            except Exception:
                logger.debug(
                    "Failed to estimate workspace repack benefit before shutdown",
                    exc_info=True,
                )
                return False
        try:
            file_size = pathlib.Path(workspace_path).stat().st_size
        except OSError:
            return False
        if file_size <= 0:
            return False
        return (
            estimated_obsolete_bytes >= _WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES
            and estimated_obsolete_bytes / file_size
            >= _WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO
        )

    def _workspace_full_save_copy_groups(
        self,
        tree: xr.DataTree,
        *,
        compression_mode: WorkspaceCompressionMode | None = None,
        require_matching_compression: bool = True,
    ) -> tuple[str | None, tuple[tuple[str, str, dict[str, typing.Any] | None], ...]]:
        if self._manager._workspace_state.path is None:
            return None, ()
        workspace_path = pathlib.Path(self._manager._workspace_state.path)
        if (
            self._manager._workspace_state.schema_version
            != workspace_format._current_workspace_schema_version()
            or not workspace_path.exists()
        ):
            return None, ()

        try:
            root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(
                workspace_path
            )
        except Exception:
            return None, ()
        schema_version, _delta_save_count, manifest = (
            workspace_format._workspace_file_metadata_from_attrs(root_attrs)
        )
        if (
            schema_version != workspace_format._current_workspace_schema_version()
            or manifest is None
        ):
            return None, ()

        manifest_entries = workspace_format._workspace_manifest_payload_entries(
            manifest
        )
        identities = {
            (uid, kind): payload_path for uid, kind, payload_path in manifest_entries
        }
        copy_groups: list[tuple[str, str, dict[str, typing.Any] | None]] = []
        context = contextlib.nullcontext(None)
        if require_matching_compression and compression_mode is not None:
            context = h5py.File(workspace_path, "r")
        with context as h5_file:
            for uid, node in self._manager._tool_graph.nodes.items():
                if (
                    uid in self._manager._workspace_state.dirty_data
                    or uid in self._manager._workspace_state.dirty_added
                ):
                    continue
                if not node.is_imagetool:
                    if node.pending_workspace_tool_payload is not None:
                        if not self._pending_workspace_tool_references_available(node):
                            continue
                    else:
                        tool = node.tool_window
                        if tool is None or not tool.can_save_and_load():
                            continue
                kind = "imagetool" if node.is_imagetool else "tool"
                source_path = identities.get((uid, kind))
                if source_path is None:
                    continue
                payload_path = self._workspace_payload_path(uid)
                try:
                    payload_tree = typing.cast("xr.DataTree", tree[payload_path])
                except KeyError:
                    continue
                payload_ds = payload_tree.to_dataset(inherit=False)
                compression_matches = (
                    h5_file is None
                    or compression_mode is None
                    or workspace_arrays._workspace_h5_group_matches_compression_mode(
                        h5_file,
                        source_path,
                        payload_ds,
                        compression_mode,
                    )
                )
                if (
                    require_matching_compression
                    and compression_mode is not None
                    and not compression_matches
                ):
                    continue
                attrs = None
                if (
                    uid in self._manager._workspace_state.dirty_state
                    or source_path != payload_path
                ):
                    attrs = dict(payload_ds.attrs)
                copy_groups.append((source_path, payload_path, attrs))
        return str(workspace_path), tuple(copy_groups)

    def _save_to_file(self, fname: str):
        """Export a selected subset of the workspace to ``fname``.

        This helper preserves the older selection-dialog behavior used by tests and
        private callers. Document-style Save and Save As use
        :meth:`_save_workspace_document` instead.
        """
        _require_itws_workspace_path(fname, _WORKSPACE_SAVE_SUFFIX_ERROR)
        tree: xr.DataTree = self._to_datatree()
        try:
            dialog = _ChooseFromDataTreeDialog(self._manager, tree, mode="save")
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return

            def _prune(node: xr.DataTree, item: QtWidgets.QTreeWidgetItem) -> None:
                if "childtools" not in node:
                    return
                child_tree = typing.cast("xr.DataTree", node["childtools"])
                for i in reversed(range(item.childCount())):
                    child_item = typing.cast("QtWidgets.QTreeWidgetItem", item.child(i))
                    child_key = str(child_item.data(0, QtCore.Qt.ItemDataRole.UserRole))
                    if child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked:
                        del child_tree[child_key]
                        continue
                    _prune(
                        typing.cast("xr.DataTree", child_tree[child_key]), child_item
                    )
                if len(child_tree) == 0:
                    del node["childtools"]

            root_item = dialog._tree_widget.invisibleRootItem()
            if root_item is None:
                return
            for i in reversed(range(root_item.childCount())):
                item = typing.cast("QtWidgets.QTreeWidgetItem", root_item.child(i))
                key = str(item.data(0, QtCore.Qt.ItemDataRole.UserRole))
                if item.checkState(0) == QtCore.Qt.CheckState.Unchecked:
                    del tree[key]
                    continue
                _prune(typing.cast("xr.DataTree", tree[key]), item)
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Saving workspace..."
            ):
                for node in tree.subtree:
                    ds = node.to_dataset(inherit=False)
                    if ds.variables or ds.attrs:
                        node.dataset = workspace_format._sanitize_workspace_attr_names(
                            ds
                        )
                tree.to_netcdf(
                    fname,
                    engine="h5netcdf",
                    invalid_netcdf=True,
                    encoding=workspace_arrays.workspace_datatree_encoding(
                        tree,
                        compression_mode=self._workspace_compression_mode(),
                    ),
                )
        finally:
            tree.close()
