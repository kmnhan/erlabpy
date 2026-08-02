"""Low-level storage, locking, and transaction mechanics for workspaces."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import typing
import uuid
from dataclasses import dataclass

from qtpy import QtCore

import erlab
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
from erlab.interactive.imagetool.manager._workspace._format import (
    _WORKSPACE_BACKUP_GROUP_PREFIX,
    _WORKSPACE_PENDING_GROUP_PREFIX,
    _WORKSPACE_SCHEMA_VERSION,
    _WORKSPACE_TRANSACTION_GROUP_PREFIX,
    _WORKSPACE_TRANSACTION_PROTOCOL,
    _compacted_workspace_root_attrs,
    _workspace_file_is_workspace,
    _workspace_path_is_itws,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    import h5py
    import xarray as xr

    from erlab.interactive._options.schema import WorkspaceCompressionMode
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")


_WorkspaceCopyGroup: typing.TypeAlias = tuple[str, str, dict[str, typing.Any] | None]
_WorkspaceCopyGroupWithSource: typing.TypeAlias = tuple[
    str, str, str, dict[str, typing.Any] | None
]
_WINDOWS_WORKSPACE_REPLACE_RETRY_DELAYS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)


class _WorkspaceBackingFileNotFoundError(FileNotFoundError):
    """A workspace payload source disappeared before it could be copied."""

    def __init__(self, source_path: str | os.PathLike[str]) -> None:
        self.source_path = os.fsdecode(source_path)
        super().__init__(
            errno.ENOENT,
            "Workspace backing file is missing",
            self.source_path,
        )


class _WorkspacePublicationConflictError(RuntimeError):
    """A workspace changed before its prepared successor could be published."""

    def __init__(self, path: str | os.PathLike[str], message: str) -> None:
        self.path = os.fsdecode(path)
        super().__init__(f"{message}: {self.path}")


@dataclass(frozen=True)
class _WorkspaceDocumentLockInfo:
    path: str
    owner: str
    hostname: str
    appname: str
    pid: int | None


@dataclass(frozen=True)
class _PreservedWorkspaceFile:
    """A private file that keeps one retired workspace generation available."""

    path: str
    file_identity: tuple[str, int, int, int]
    cleanup: Callable[[], None]


@contextlib.contextmanager
def _open_workspace_h5_file_for_update(
    fname: str | os.PathLike[str],
) -> Iterator[typing.Any]:
    """Open a workspace file for mutation while its process lock is held."""
    with workspace_arrays._workspace_file_lock(fname), h5py.File(fname, "a") as h5_file:
        yield h5_file


@contextlib.contextmanager
def _workspace_in_place_update(
    fname: str | os.PathLike[str],
    *,
    rewrite_groups: Iterable[str] = (),
    expected_state: tuple[tuple[str, int, int, int], bool] | None = None,
) -> Iterator[None]:
    """Coordinate one recoverable in-place workspace update."""
    with workspace_arrays._workspace_file_lock(fname):
        generation = workspace_arrays._current_workspace_file_generation(fname)
        initial_state = workspace_arrays._workspace_file_state(fname)
        if expected_state is not None and initial_state != expected_state:
            raise _WorkspacePublicationConflictError(
                fname, "Workspace changed while its incremental save was prepared"
            )
        if generation is not None:
            generation.close(needs_lock=False)
            workspace_arrays._detach_workspace_file_generation_readers(
                fname, rewrite_groups
            )

        body_error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            current_generation = workspace_arrays._current_workspace_file_generation(
                fname
            )
            current_identity, current_exists = workspace_arrays._workspace_file_state(
                fname
            )
            initial_identity, initial_exists = initial_state
            same_file = (
                initial_exists
                and current_exists
                and current_identity[1:3] == initial_identity[1:3]
            )
            if generation is not None and current_generation is generation:
                if same_file:
                    generation.refresh_in_place(current_identity)
                else:
                    generation.invalidate(
                        "Workspace file changed during an in-place save: "
                        f"{os.fsdecode(fname)}"
                    )
                    workspace_arrays._discard_workspace_file_generation(
                        fname, generation
                    )
                    if body_error is None:
                        raise _WorkspacePublicationConflictError(
                            fname, "Workspace changed during its in-place save"
                        )
            elif not same_file and body_error is None:
                raise _WorkspacePublicationConflictError(
                    fname, "Workspace changed during its in-place save"
                )


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


def _hide_workspace_internal_file(path: str) -> None:
    workspace_arrays._hide_workspace_internal_path(path)


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
    with contextlib.suppress(Exception):
        workspace_arrays._cleanup_stale_workspace_reader_files(fname)
    _hide_workspace_internal_file(lock_path)
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


def _workspace_replace_retry_delays(exc: PermissionError) -> tuple[float, ...]:
    if os.name != "nt":
        return ()
    if getattr(exc, "winerror", None) not in (None, 5, 32):
        return ()
    if exc.errno not in (None, errno.EACCES, errno.EPERM):
        return ()
    return _WINDOWS_WORKSPACE_REPLACE_RETRY_DELAYS


def _preserve_workspace_file_generation(
    path: str | os.PathLike[str],
    expected_identity: tuple[str, int, int, int],
) -> _PreservedWorkspaceFile:
    """Preserve one workspace generation for existing lazy readers."""
    source_identity, source_exists = workspace_arrays._workspace_file_state(path)
    if not source_exists:
        raise FileNotFoundError(errno.ENOENT, "Workspace file is missing", path)
    if source_identity[1:] != expected_identity[1:]:
        raise _WorkspacePublicationConflictError(
            path, "Workspace changed before it could be preserved"
        )

    reader_file = workspace_arrays._create_workspace_reader_file(path, path)
    cleanup = reader_file.cleanup
    try:
        source_identity_after = workspace_arrays._workspace_file_identity(path)
        preserved_identity = workspace_arrays._workspace_file_identity(reader_file.path)
    except BaseException:
        cleanup()
        raise
    if source_identity_after[1:] != source_identity[1:]:
        cleanup()
        raise _WorkspacePublicationConflictError(
            path, "Workspace changed while it was being preserved"
        )
    return _PreservedWorkspaceFile(
        path=reader_file.path,
        file_identity=preserved_identity,
        cleanup=cleanup,
    )


def _validate_workspace_publication_state(
    path: str | os.PathLike[str],
    expected_state: tuple[tuple[str, int, int, int], bool],
    message: str,
) -> None:
    if workspace_arrays._workspace_file_state(path) != expected_state:
        raise _WorkspacePublicationConflictError(path, message)


def _replace_workspace_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    expected_state: tuple[tuple[str, int, int, int], bool] | None = None,
) -> None:
    """Atomically replace a workspace and preserve active reader generations."""
    with workspace_arrays._workspace_file_lock(destination):
        current_state = workspace_arrays._workspace_file_state(destination)
        if expected_state is not None and current_state != expected_state:
            raise _WorkspacePublicationConflictError(
                destination, "Workspace changed while its successor was prepared"
            )
        generation = workspace_arrays._current_workspace_file_generation(destination)
        generation_has_managers = generation is not None and generation.has_managers()
        generation_started_without_file = (
            generation is not None and generation.created_without_file
        )
        preserved: _PreservedWorkspaceFile | None = None
        try:
            if generation is not None:
                generation.close(needs_lock=False)
            if (
                generation is not None
                and generation_has_managers
                and not generation_started_without_file
            ):
                preserved = _preserve_workspace_file_generation(
                    destination, generation.file_identity
                )

            retry_delays: tuple[float, ...] | None = None
            while True:
                _validate_workspace_publication_state(
                    destination,
                    current_state,
                    "Workspace changed while replacement was waiting for access",
                )
                try:
                    os.replace(source, destination)
                except PermissionError as exc:
                    if retry_delays is None:
                        retry_delays = _workspace_replace_retry_delays(exc)
                    if not retry_delays:
                        raise
                    delay, *remaining_delays = retry_delays
                    retry_delays = tuple(remaining_delays)
                    time.sleep(delay)
                else:
                    break
        except BaseException:
            if preserved is not None:
                if (
                    generation is not None
                    and workspace_arrays._workspace_file_state(destination)
                    != current_state
                ):
                    workspace_arrays._retire_workspace_file_generation(
                        destination,
                        generation,
                        preserved.path,
                        preserved.file_identity,
                        preserved.cleanup,
                    )
                else:
                    with contextlib.suppress(Exception):
                        preserved.cleanup()
            raise

        if generation is not None:
            if preserved is not None:
                workspace_arrays._retire_workspace_file_generation(
                    destination,
                    generation,
                    preserved.path,
                    preserved.file_identity,
                    preserved.cleanup,
                )
            elif generation_has_managers:
                generation.activate_new_file()
            else:
                workspace_arrays._discard_workspace_file_generation(
                    destination, generation
                )
                generation.dispose()


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


def _workspace_transaction_recovery_needed(
    fname: str | os.PathLike[str],
) -> bool:
    """Return True when a workspace contains unfinished transaction state."""
    if not pathlib.Path(fname).exists() or not _workspace_path_is_itws(fname):
        return False
    with workspace_arrays._workspace_file_lock(fname):
        generation = workspace_arrays._current_workspace_file_generation(fname)
        if generation is not None:
            generation.close(needs_lock=False)
        with h5py.File(fname, "r") as h5_file:
            if not _workspace_file_is_workspace(h5_file):
                return False
            return any(
                name.startswith(
                    (
                        _WORKSPACE_TRANSACTION_GROUP_PREFIX,
                        _WORKSPACE_PENDING_GROUP_PREFIX,
                        _WORKSPACE_BACKUP_GROUP_PREFIX,
                    )
                )
                for name in h5_file
            )


def _recover_workspace_transactions_in_place(
    fname: str | os.PathLike[str],
) -> None:
    """Recover transaction state while the workspace update lock is held."""
    with _open_workspace_h5_file_for_update(fname) as h5_file:
        if not _workspace_file_is_workspace(h5_file):
            return
        for name in list(h5_file):
            if name.startswith(_WORKSPACE_TRANSACTION_GROUP_PREFIX):
                _recover_open_workspace_transaction(h5_file, name)
        _cleanup_orphan_workspace_internal_groups(h5_file)
        h5_file.flush()


def _recover_workspace_transactions(fname: str | os.PathLike[str]) -> None:
    with workspace_arrays._workspace_save_lock(fname):
        _recover_workspace_transactions_coordinated(fname)


def _recover_workspace_transactions_coordinated(
    fname: str | os.PathLike[str],
) -> None:
    """Recover transactions while this process owns the workspace save lock."""
    if not _workspace_transaction_recovery_needed(fname):
        return
    with _workspace_in_place_update(fname):
        _recover_workspace_transactions_in_place(fname)
        _fsync_file(fname)


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
    _write_workspace_root_attrs_transaction_file(fname, attrs, replace=replace)


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
    with workspace_arrays._workspace_save_lock(fname):
        _recover_workspace_transactions_coordinated(fname)
        root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(fname)
        return (
            _compacted_workspace_root_attrs(root_attrs),
            workspace_arrays._workspace_live_root_group_copy_groups(fname),
        )


def _write_workspace_constructor_groups_to_path(
    fname: str | os.PathLike[str],
    constructor: Mapping[str, xr.Dataset],
    group_path: str,
    destination_path: str,
    *,
    compression_mode: WorkspaceCompressionMode | None = None,
) -> None:
    target_group_path = group_path.strip("/")
    destination_path = destination_path.strip("/")
    for constructor_group_path, ds in sorted(
        constructor.items(), key=lambda item: item[0].count("/")
    ):
        source_group_path = constructor_group_path.strip("/")
        if source_group_path != target_group_path and not source_group_path.startswith(
            f"{target_group_path}/"
        ):
            continue
        relative_path = source_group_path.removeprefix(target_group_path).strip("/")
        output_group_path = (
            destination_path
            if not relative_path
            else f"{destination_path}/{relative_path}"
        )
        workspace_arrays._write_workspace_dataset_group_to_file(
            fname,
            output_group_path,
            ds,
            lock_path=fname,
            compression_mode=compression_mode,
        )


def _materialize_workspace_constructor_sources(
    fname: str | os.PathLike[str],
    rewrite_map: Mapping[str, tuple[str, dict[str, xr.Dataset]]],
) -> None:
    """Load constructor data that reads from the file about to be mutated."""
    normalized_path = workspace_arrays._normalized_file_path(fname)
    materialized: dict[int, xr.Dataset] = {}
    for _group_path, constructor in rewrite_map.values():
        for constructor_group_path, ds in tuple(constructor.items()):
            loaded_ds = materialized.get(id(ds))
            if loaded_ds is not None:
                constructor[constructor_group_path] = loaded_ds
                continue
            if normalized_path is not None and any(
                normalized_path in workspace_arrays.dataarray_source_paths(data_array)
                for data_array in (*ds.data_vars.values(), *ds.coords.values())
            ):
                variable_chunks: dict[typing.Hashable, tuple[int, ...]] = {}
                for name, data_array in ds.data_vars.items():
                    chunksizes = workspace_arrays._workspace_chunksizes_for_dataarray(
                        data_array
                    )
                    if chunksizes is not None:
                        variable_chunks[name] = chunksizes
                loaded_ds = ds.compute()
                for name, chunksizes in variable_chunks.items():
                    loaded_ds[name].encoding[
                        workspace_arrays._WORKSPACE_MATERIALIZED_CHUNKSIZES
                    ] = chunksizes
                for variable in loaded_ds.variables.values():
                    variable.encoding.pop("source", None)
                materialized[id(ds)] = loaded_ds
                constructor[constructor_group_path] = loaded_ds


def _path_is_at_or_under(path: str, root_path: str) -> bool:
    path = path.strip("/")
    root_path = root_path.strip("/")
    return path == root_path or path.startswith(f"{root_path}/")


def _workspace_missing_attr_fallback_groups(
    fname: str | os.PathLike[str],
    rewrite_paths: Iterable[str],
    attr_updates: Iterable[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
    ],
) -> tuple[str, ...]:
    """Return fallback groups needed for missing attribute targets."""
    initial_paths = tuple(path.strip("/") for path in rewrite_paths)
    required_paths = set(initial_paths)
    with h5py.File(fname, "r") as h5_file:
        for payload_path, _attrs, fallback in attr_updates:
            attr_path = payload_path.strip("/")
            if any(
                _path_is_at_or_under(attr_path, rewrite_path)
                for rewrite_path in required_paths
            ):
                continue
            fallback_path = fallback[0].strip("/")
            if attr_path not in h5_file and not any(
                _path_is_at_or_under(fallback_path, rewrite_path)
                for rewrite_path in required_paths
            ):
                required_paths.add(fallback_path)
    return tuple(sorted(required_paths.difference(initial_paths)))


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
            _write_workspace_constructor_groups_to_path(
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
    *,
    replace_root_attrs: bool = False,
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

        _write_root_attrs_to_open_workspace_file(
            h5_file, root_attrs, replace=replace_root_attrs
        )
        _set_workspace_transaction_status(h5_file, txn_path, "committed")


def _write_workspace_root_attrs_transaction_file(
    fname: str | os.PathLike[str],
    attrs: Mapping[str, typing.Any],
    *,
    replace: bool,
) -> None:
    """Write root attributes through the recoverable transaction protocol."""
    with workspace_arrays._workspace_save_lock(fname):
        _write_workspace_root_attrs_transaction_file_coordinated(
            fname, attrs, replace=replace
        )


def _write_workspace_root_attrs_transaction_file_coordinated(
    fname: str | os.PathLike[str],
    attrs: Mapping[str, typing.Any],
    *,
    replace: bool,
) -> None:
    """Write root attributes while this process owns the workspace save lock."""
    _recover_workspace_transactions_coordinated(fname)
    txn_id = uuid.uuid4().hex
    txn_path = f"{_WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{_WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{_WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    with _workspace_in_place_update(fname):
        try:
            group_operations, attr_updates = _prepare_workspace_transaction(
                fname,
                txn_path,
                pending_root,
                backup_root,
                {},
                (),
                attrs,
            )
            _commit_workspace_transaction(
                fname,
                txn_path,
                group_operations,
                attr_updates,
                attrs,
                replace_root_attrs=replace,
            )
        except BaseException:
            with contextlib.suppress(Exception):
                _recover_workspace_transactions_in_place(fname)
                _fsync_file(fname)
            raise
        _recover_workspace_transactions_in_place(fname)
        _fsync_file(fname)


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
    """Write one recoverable incremental workspace transaction."""
    with workspace_arrays._workspace_save_lock(fname):
        _write_workspace_transaction_file_coordinated(
            fname,
            rewrite_groups,
            attr_updates,
            root_attrs,
            compression_mode=compression_mode,
        )


def _write_workspace_transaction_file_coordinated(
    fname: str | os.PathLike[str],
    rewrite_groups: Iterable[tuple[str, dict[str, xr.Dataset]]],
    attr_updates: Iterable[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
    ],
    root_attrs: Mapping[str, typing.Any],
    *,
    compression_mode: WorkspaceCompressionMode | None,
) -> None:
    """Prepare and write a delta while this process owns the save lock."""
    _recover_workspace_transactions_coordinated(fname)
    rewrite_map = {
        group_path.strip("/"): (group_path, constructor)
        for group_path, constructor in rewrite_groups
    }
    attr_updates_tuple = tuple(attr_updates)
    with workspace_arrays._workspace_file_lock(fname):
        expected_state = workspace_arrays._workspace_file_state(fname)
        missing_fallback_groups = _workspace_missing_attr_fallback_groups(
            fname, rewrite_map, attr_updates_tuple
        )
        missing_fallback_group_set = set(missing_fallback_groups)
        for _payload_path, _attrs, fallback in attr_updates_tuple:
            fallback_path = fallback[0].strip("/")
            if fallback_path in missing_fallback_group_set:
                rewrite_map[fallback_path] = fallback

    # Dask can schedule reads on worker threads. Materialize before the file lock so
    # those reads can acquire their generation without waiting on the save thread.
    _materialize_workspace_constructor_sources(fname, rewrite_map)

    txn_id = uuid.uuid4().hex
    txn_path = f"{_WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{_WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{_WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    with _workspace_in_place_update(
        fname,
        rewrite_groups=rewrite_map,
        expected_state=expected_state,
    ):
        try:
            group_operations, attr_updates_to_write = _prepare_workspace_transaction(
                fname,
                txn_path,
                pending_root,
                backup_root,
                rewrite_map,
                attr_updates_tuple,
                root_attrs,
            )
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
        except BaseException:
            with contextlib.suppress(Exception):
                _recover_workspace_transactions_in_place(fname)
                _fsync_file(fname)
            raise
        _recover_workspace_transactions_in_place(fname)
        _fsync_file(fname)


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
    """Write and publish one complete workspace successor."""
    with workspace_arrays._workspace_save_lock(fname):
        _write_full_workspace_tree_file_coordinated(
            fname,
            tree,
            root_attrs,
            copy_source=copy_source,
            copy_groups=copy_groups,
            copy_group_sources=copy_group_sources,
            compression_mode=compression_mode,
        )


def _write_full_workspace_tree_file_coordinated(
    fname: str | os.PathLike[str],
    tree: xr.DataTree | None,
    root_attrs: Mapping[str, typing.Any],
    *,
    copy_source: str | os.PathLike[str] | None,
    copy_groups: Iterable[_WorkspaceCopyGroup],
    copy_group_sources: Iterable[_WorkspaceCopyGroupWithSource],
    compression_mode: WorkspaceCompressionMode | None,
) -> None:
    """Build a successor while this process owns the workspace save lock."""
    fname = os.fsdecode(fname)
    with workspace_arrays._workspace_file_lock(fname):
        generation = workspace_arrays._current_workspace_file_generation(fname)
        if generation is not None:
            generation.close(needs_lock=False)
        expected_state = workspace_arrays._workspace_file_state(fname)
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
            with workspace_arrays._workspace_file_lock(source_fname):
                try:
                    source_file = h5py.File(source_fname, "r")
                except FileNotFoundError as exc:
                    raise _WorkspaceBackingFileNotFoundError(source_fname) from exc
                with source_file, h5py.File(tmp_fname, "a") as tmp_file:
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
                _replace_workspace_file(tmp_fname, fname, expected_state=expected_state)
            except OSError as err:
                if err.errno != errno.EXDEV:
                    raise
                destination_tmp = f"{fname}.tmp-{uuid.uuid4().hex}"
                shutil.copyfile(tmp_fname, destination_tmp)
                _fsync_file(destination_tmp)
                _replace_workspace_file(
                    destination_tmp, fname, expected_state=expected_state
                )
            _fsync_parent_directory(fname)
        else:
            _replace_workspace_file(tmp_fname, fname, expected_state=expected_state)
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
