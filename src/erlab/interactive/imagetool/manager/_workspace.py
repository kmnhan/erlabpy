from __future__ import annotations

import contextlib
import errno
import json
import os
import pathlib
import sys
import time
import traceback
import typing
import uuid
from dataclasses import dataclass

from qtpy import QtCore

import erlab
from erlab.interactive.imagetool.manager import _xarray as _manager_xarray

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, MutableMapping

    import xarray as xr

_WORKSPACE_SCHEMA_VERSION = 4
_WORKSPACE_LEGACY_SCHEMA_VERSION = 3
_WORKSPACE_MANIFEST_ATTR = "imagetool_workspace_manifest"
_WORKSPACE_TRANSACTION_PROTOCOL = "recoverable-delta-v1"
_WORKSPACE_PENDING_GROUP_PREFIX = "__itws_pending_"
_WORKSPACE_BACKUP_GROUP_PREFIX = "__itws_backup_"
_WORKSPACE_TRANSACTION_GROUP_PREFIX = "__itws_txn_"
_WORKSPACE_INTERNAL_GROUP_PREFIXES = (
    _WORKSPACE_PENDING_GROUP_PREFIX,
    _WORKSPACE_BACKUP_GROUP_PREFIX,
    _WORKSPACE_TRANSACTION_GROUP_PREFIX,
)


@dataclass
class _WorkspaceRewriteGroup:
    group_path: str
    constructor: dict[str, xr.Dataset]


@dataclass
class _WorkspaceAttrUpdate:
    payload_path: str
    attrs: dict[str, typing.Any]
    fallback: _WorkspaceRewriteGroup


@dataclass
class _WorkspaceSaveSnapshot:
    generation: int
    root_attrs: dict[str, typing.Any]
    delta_save_count: int
    full_tree: xr.DataTree | None = None
    rewrite_groups: tuple[_WorkspaceRewriteGroup, ...] = ()
    attr_updates: tuple[_WorkspaceAttrUpdate, ...] = ()

    def close(self) -> None:
        if self.full_tree is not None:
            self.full_tree.close()


@dataclass(frozen=True)
class _WorkspaceFileMetadata:
    schema_version: int
    delta_save_count: int
    manifest: dict[str, typing.Any] | None


@dataclass(frozen=True)
class _WorkspaceDirtyEvent:
    generation: int
    uid: str | None = None
    data: bool = False
    state: bool = False
    added: bool = False
    removed: str | None = None
    structure: str | None = None


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
        loop: QtCore.QEventLoop,
        result: dict[str, typing.Any],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._loop = loop
        self._result = result

    @QtCore.Slot(bool, float, str)
    def finish(self, ok: bool, elapsed: float, error_text: str) -> None:
        self._result["ok"] = ok
        self._result["elapsed"] = elapsed
        self._result["error"] = error_text
        self._loop.quit()


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
            if self._snapshot.full_tree is None:
                _write_workspace_transaction_file(
                    self._fname,
                    self._snapshot.rewrite_groups,
                    self._snapshot.attr_updates,
                    self._snapshot.root_attrs,
                )
            else:
                _write_full_workspace_tree_file(
                    self._fname,
                    self._snapshot.full_tree,
                    self._snapshot.root_attrs,
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
    import h5py

    with _manager_xarray._workspace_file_lock(fname), h5py.File(fname, "a") as h5_file:
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
            import stat

            if not stat.S_ISREG(os.lstat(lock_path).st_mode):
                return
            os.chflags(lock_path, stat.UF_HIDDEN)
        return
    if os.name != "nt":
        return

    with contextlib.suppress(Exception):
        import ctypes

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


def _workspace_manifest_from_attrs(
    attrs: Mapping[typing.Hashable, typing.Any],
) -> dict[str, typing.Any]:
    raw_manifest = attrs.get(_WORKSPACE_MANIFEST_ATTR)
    if isinstance(raw_manifest, bytes):
        raw_manifest = raw_manifest.decode()
    if isinstance(raw_manifest, str):
        with contextlib.suppress(json.JSONDecodeError):
            manifest = json.loads(raw_manifest)
            if isinstance(manifest, dict):
                return manifest
    return {}


def _workspace_delta_save_count_from_attrs(
    attrs: Mapping[typing.Hashable, typing.Any],
) -> int:
    value = _workspace_manifest_from_attrs(attrs).get("delta_save_count", 0)
    with contextlib.suppress(TypeError, ValueError):
        return max(0, int(value))
    return 0


def _workspace_file_metadata_from_attrs(
    attrs: Mapping[typing.Hashable, typing.Any],
) -> _WorkspaceFileMetadata:
    schema_version = int(attrs.get("imagetool_workspace_schema_version", 1))
    manifest = None
    if schema_version >= _WORKSPACE_SCHEMA_VERSION:
        manifest = _workspace_manifest_from_attrs(attrs) or None
    return _WorkspaceFileMetadata(
        schema_version=schema_version,
        delta_save_count=_workspace_delta_save_count_from_attrs(attrs),
        manifest=manifest,
    )


def _current_workspace_schema_version() -> int:
    return _WORKSPACE_SCHEMA_VERSION


def _set_legacy_workspace_schema(
    attrs: MutableMapping[typing.Hashable, typing.Any],
) -> None:
    attrs["imagetool_workspace_schema_version"] = _WORKSPACE_LEGACY_SCHEMA_VERSION


def _workspace_schema_requires_conversion(schema_version: int) -> bool:
    return schema_version < _WORKSPACE_LEGACY_SCHEMA_VERSION


def _workspace_schema_requires_full_save(schema_version: int) -> bool:
    return (
        _WORKSPACE_LEGACY_SCHEMA_VERSION <= schema_version < _WORKSPACE_SCHEMA_VERSION
    )


def _workspace_root_attrs_payload(
    *,
    root_order: Iterable[typing.Any],
    nodes: Iterable[Mapping[str, typing.Any]],
    delta_save_count: int,
    erlab_version: str,
) -> dict[str, typing.Any]:
    manifest: dict[str, typing.Any] = {
        "schema_version": _WORKSPACE_SCHEMA_VERSION,
        "erlab_version": erlab_version,
        "root_order": list(root_order),
        "nodes": list(nodes),
    }
    if delta_save_count > 0:
        manifest["transaction_protocol"] = _WORKSPACE_TRANSACTION_PROTOCOL
        manifest["delta_save_count"] = int(delta_save_count)
    return {
        "imagetool_workspace_schema_version": _WORKSPACE_SCHEMA_VERSION,
        _WORKSPACE_MANIFEST_ATTR: json.dumps(manifest),
        "erlab_version": erlab_version,
    }


def _is_workspace_internal_group_name(name: typing.Any) -> bool:
    return str(name).startswith(_WORKSPACE_INTERNAL_GROUP_PREFIXES)


def _workspace_root_keys(
    tree: typing.Any, manifest: Mapping[str, typing.Any] | None
) -> list[str]:
    root_keys: list[str] = []
    if manifest is not None:
        raw_root_order = manifest.get("root_order", ())
        if isinstance(raw_root_order, list):
            root_keys.extend(
                str(item)
                for item in raw_root_order
                if str(item) not in root_keys
                and not _is_workspace_internal_group_name(item)
            )
    root_keys.extend(
        str(key)
        for key in tree
        if str(key) not in root_keys and not _is_workspace_internal_group_name(key)
    )
    return root_keys


def _workspace_file_is_workspace(h5_file) -> bool:
    if "imagetool_workspace_schema_version" in h5_file.attrs:
        return True
    return h5_file.attrs.get("is_itool_workspace", 0) == 1


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


def _replace_h5_attrs(target_attrs, attrs: Mapping[str, typing.Any]) -> None:
    for key in list(target_attrs):
        del target_attrs[key]
    for key, value in attrs.items():
        target_attrs[key] = value


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
        _replace_h5_attrs(target_attrs, backup_group["attrs"].attrs)


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
            _delete_h5_path(h5_file, path)


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
        backup_exists = _h5_path_exists(h5_file, backup_path)
        if backup_exists:
            _delete_h5_path(h5_file, group_path)
            _ensure_h5_parent_group(h5_file, group_path)
            h5_file.move(backup_path.strip("/"), group_path.strip("/"))
        elif not old_exists:
            _delete_h5_path(h5_file, group_path)


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
    import h5py

    if not pathlib.Path(fname).exists():
        return
    with _manager_xarray._workspace_file_lock(fname), h5py.File(fname, "a") as h5_file:
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


def _write_workspace_constructor_groups_to_pending(
    fname: str | os.PathLike[str],
    constructor: Mapping[str, xr.Dataset],
    group_path: str,
    pending_path: str,
) -> None:
    target_group_path = group_path.strip("/")
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
        with _manager_xarray._workspace_file_lock(fname):
            ds.to_netcdf(
                fname,
                mode="a",
                engine="h5netcdf",
                group=f"/{pending_group_path}",
                invalid_netcdf=True,
                encoding=_manager_xarray.workspace_dataset_encoding(ds),
            )


def _path_is_at_or_under(path: str, root_path: str) -> bool:
    path = path.strip("/")
    root_path = root_path.strip("/")
    return path == root_path or path.startswith(f"{root_path}/")


def _move_h5_path(h5_file, source_path: str, destination_path: str) -> None:
    _ensure_h5_parent_group(h5_file, destination_path)
    h5_file.move(source_path.strip("/"), destination_path.strip("/"))


def _set_workspace_transaction_status(h5_file, txn_path: str, status: str) -> None:
    h5_file[txn_path].attrs["status"] = status
    h5_file.flush()


def _prepare_workspace_transaction(
    fname: str | os.PathLike[str],
    txn_path: str,
    pending_root: str,
    backup_root: str,
    rewrite_map: dict[str, _WorkspaceRewriteGroup],
    attr_updates: tuple[_WorkspaceAttrUpdate, ...],
    root_attrs: Mapping[str, typing.Any],
) -> tuple[list[dict[str, typing.Any]], list[_WorkspaceAttrUpdate]]:
    attr_updates_to_write: list[_WorkspaceAttrUpdate] = []
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        txn_group = h5_file.create_group(txn_path)
        txn_group.attrs["protocol"] = _WORKSPACE_TRANSACTION_PROTOCOL
        txn_group.attrs["status"] = "preparing"
        txn_group.attrs["pending_root"] = pending_root
        txn_group.attrs["backup_root"] = backup_root

        attr_backup_index = 0
        for attr_update in attr_updates:
            attr_path = attr_update.payload_path.strip("/")
            if any(
                _path_is_at_or_under(attr_path, rewrite_path)
                for rewrite_path in rewrite_map
            ):
                continue
            if attr_path in h5_file:
                _write_workspace_attr_backup(
                    txn_group,
                    attr_backup_index,
                    attr_update.payload_path,
                    h5_file[attr_path].attrs,
                )
                attr_backup_index += 1
                attr_updates_to_write.append(attr_update)
            else:
                fallback_path = attr_update.fallback.group_path.strip("/")
                if not any(
                    _path_is_at_or_under(fallback_path, rewrite_path)
                    for rewrite_path in rewrite_map
                ):
                    rewrite_map[fallback_path] = attr_update.fallback

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
                    "old_exists": _h5_path_exists(h5_file, group_path),
                }
            )
        txn_group.attrs["operations"] = json.dumps(
            {"group_replacements": group_operations}
        )
        h5_file.flush()
    return group_operations, attr_updates_to_write


def _write_workspace_transaction_pending_groups(
    fname: str | os.PathLike[str],
    rewrite_map: Mapping[str, _WorkspaceRewriteGroup],
    pending_root: str,
) -> None:
    try:
        for group_path, rewrite_group in sorted(rewrite_map.items()):
            pending_group_path = f"{pending_root}/{group_path}"
            _write_workspace_constructor_groups_to_pending(
                fname,
                rewrite_group.constructor,
                rewrite_group.group_path,
                pending_group_path,
            )
    except Exception:
        with _open_workspace_h5_file_for_update(fname) as h5_file:
            _delete_h5_path(h5_file, pending_root)
            h5_file.flush()
        raise


def _commit_workspace_transaction(
    fname: str | os.PathLike[str],
    txn_path: str,
    group_operations: Iterable[Mapping[str, typing.Any]],
    attr_updates: Iterable[_WorkspaceAttrUpdate],
    root_attrs: Mapping[str, typing.Any],
) -> None:
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        _set_workspace_transaction_status(h5_file, txn_path, "committing")
        for operation in group_operations:
            group_path = typing.cast("str", operation["group_path"])
            pending_path = typing.cast("str", operation["pending_path"])
            backup_path = typing.cast("str", operation["backup_path"])
            if not _h5_path_exists(h5_file, pending_path):
                raise KeyError(
                    f"Workspace pending group {pending_path!r} was not written"
                )
            if _h5_path_exists(h5_file, group_path):
                _move_h5_path(h5_file, group_path, backup_path)
            _move_h5_path(h5_file, pending_path, group_path)

        for attr_update in attr_updates:
            target_attrs = _workspace_txn_attr_target(h5_file, attr_update.payload_path)
            if target_attrs is not None:
                _replace_h5_attrs(target_attrs, attr_update.attrs)

        _write_root_attrs_to_open_workspace_file(h5_file, root_attrs)
        _set_workspace_transaction_status(h5_file, txn_path, "committed")


def _write_workspace_transaction_file(
    fname: str | os.PathLike[str],
    rewrite_groups: Iterable[_WorkspaceRewriteGroup],
    attr_updates: Iterable[_WorkspaceAttrUpdate],
    root_attrs: Mapping[str, typing.Any],
) -> None:
    _recover_workspace_transactions(fname)
    rewrite_map = {
        rewrite_group.group_path.strip("/"): rewrite_group
        for rewrite_group in rewrite_groups
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
        _write_workspace_transaction_pending_groups(fname, rewrite_map, pending_root)
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
    tree: xr.DataTree,
    root_attrs: Mapping[str, typing.Any],
) -> None:
    fname = str(fname)
    tmp_fname = f"{fname}.tmp-{uuid.uuid4().hex}"
    try:
        tree.to_netcdf(
            tmp_fname,
            engine="h5netcdf",
            invalid_netcdf=True,
            encoding=_manager_xarray.workspace_datatree_encoding(tree),
        )
        _write_workspace_root_attrs_to_file(tmp_fname, root_attrs, replace=True)
        _validate_workspace_h5_file(tmp_fname)
        _fsync_file(tmp_fname)
        os.replace(tmp_fname, fname)
        _fsync_parent_directory(fname)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_fname)


def _validate_workspace_h5_file(fname: str | os.PathLike[str]) -> None:
    import h5py

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
