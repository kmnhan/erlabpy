import contextlib
import errno
import json
import os
import pathlib
import shutil
import types
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import h5py
import numpy as np
import pytest
import xarray as xr
from qtpy import QtWidgets

import erlab
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
from erlab.interactive import _persistence_constants
from tests.interactive.imagetool.manager.workspace._support import (
    _assert_no_workspace_internal_groups,
    _transaction_test_dataset,
    _transaction_test_root_attrs,
    _write_transaction_test_workspace,
)


def _read_transaction_test_value(fname: pathlib.Path) -> float:
    opened = workspace_arrays.open_workspace_datatree(fname, chunks=None)
    try:
        ds = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        return float(ds["data"].item())
    finally:
        opened.close()


def _create_legacy_transaction(
    h5_file: h5py.File,
    txn_id: str,
    *,
    status: str,
    group_replacements: list[dict[str, object]] | None = None,
) -> tuple[h5py.Group, str, str]:
    txn_path = f"{_persistence_constants.WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"__itws_pending_{txn_id}"
    backup_root = f"__itws_backup_{txn_id}"
    txn_group = h5_file.create_group(txn_path)
    txn_group.attrs.update(
        {
            "status": status,
            "pending_root": pending_root,
            "backup_root": backup_root,
            "operations": json.dumps({"group_replacements": group_replacements or []}),
        }
    )
    h5_file.create_group(pending_root)
    h5_file.create_group(backup_root)
    return txn_group, pending_root, backup_root


def test_replace_workspace_file_retries_transient_access_denied(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "prepared.itws"
    destination = tmp_path / "workspace.itws"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    expected_state = workspace_storage._workspace_publication_state(destination)
    original_replace = workspace_storage.os.replace
    replace_attempts = 0
    delays: list[float] = []
    synced_parents: list[pathlib.Path] = []

    def _replace_with_transient_denial(src, dst) -> None:
        nonlocal replace_attempts
        replace_attempts += 1
        if replace_attempts < 3:
            raise PermissionError(errno.EACCES, "file is in use", dst)
        original_replace(src, dst)

    monkeypatch.setattr(workspace_storage.os, "replace", _replace_with_transient_denial)
    monkeypatch.setattr(workspace_store.time, "sleep", delays.append)
    monkeypatch.setattr(
        workspace_storage, "_fsync_parent_directory", synced_parents.append
    )

    workspace_storage._replace_workspace_file(
        source, destination, expected_state=expected_state
    )

    assert replace_attempts == 3
    assert delays == [0.02, 0.05]
    assert synced_parents == [destination]
    assert destination.read_bytes() == b"new"


def test_replace_workspace_file_stops_if_destination_changes_during_retry(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "prepared.itws"
    destination = tmp_path / "workspace.itws"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    expected_state = workspace_storage._workspace_publication_state(destination)

    def _replace_after_external_change(_src, dst) -> None:
        pathlib.Path(dst).write_bytes(b"changed outside the manager")
        raise PermissionError(errno.EACCES, "file is in use", dst)

    monkeypatch.setattr(workspace_storage.os, "replace", _replace_after_external_change)
    monkeypatch.setattr(workspace_store.time, "sleep", lambda _delay: None)

    with pytest.raises(workspace_storage._WorkspacePublicationConflictError):
        workspace_storage._replace_workspace_file(
            source, destination, expected_state=expected_state
        )

    assert source.read_bytes() == b"new"
    assert destination.read_bytes() == b"changed outside the manager"


def test_replace_workspace_file_preserves_permission_error_after_retries(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "prepared.itws"
    destination = tmp_path / "workspace.itws"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    expected_state = workspace_storage._workspace_publication_state(destination)
    attempts = 0
    synced_parents: list[pathlib.Path] = []

    def _deny_replace(_src, dst) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError(errno.EACCES, "file is in use", dst)

    monkeypatch.setattr(
        workspace_store,
        "_FILE_ACCESS_RETRY_DELAYS",
        (0.0, 0.0),
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _deny_replace)
    monkeypatch.setattr(workspace_store.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(
        workspace_storage, "_fsync_parent_directory", synced_parents.append
    )

    with pytest.raises(PermissionError, match="file is in use"):
        workspace_storage._replace_workspace_file(
            source, destination, expected_state=expected_state
        )

    assert attempts == 3
    assert synced_parents == []
    assert source.read_bytes() == b"new"
    assert destination.read_bytes() == b"old"


def test_replace_workspace_file_ignores_metadata_only_ctime_changes(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "prepared.itws"
    destination = tmp_path / "workspace.itws"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    expected_state = workspace_storage._workspace_publication_state(destination)
    real_state = workspace_storage.os.stat(destination)
    ctime_ns = real_state.st_ctime_ns

    def _stat_with_new_ctime(_path):
        nonlocal ctime_ns
        ctime_ns += 1
        return types.SimpleNamespace(
            st_dev=real_state.st_dev,
            st_ino=real_state.st_ino,
            st_size=real_state.st_size,
            st_mtime_ns=real_state.st_mtime_ns,
            st_ctime_ns=ctime_ns,
        )

    monkeypatch.setattr(workspace_storage.os, "stat", _stat_with_new_ctime)

    workspace_storage._replace_workspace_file(
        source, destination, expected_state=expected_state
    )

    assert destination.read_bytes() == b"new"


def test_workspace_publication_state_detects_same_size_content_changes(
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    path.write_bytes(b"before")
    expected = workspace_storage._workspace_publication_state(path)
    previous_mtime_ns = path.stat().st_mtime_ns
    path.write_bytes(b"after!")
    os.utime(
        path,
        ns=(previous_mtime_ns + 1_000_000_000, previous_mtime_ns + 1_000_000_000),
    )

    with pytest.raises(workspace_storage._WorkspacePublicationConflictError):
        workspace_storage._require_workspace_publication_state(path, expected)


def test_workspace_file_access_retry_filter_is_specific() -> None:
    access_denied = PermissionError(errno.EACCES, "access denied")
    busy = OSError(errno.EBUSY, "busy")
    wrong_error = PermissionError(errno.ENOENT, "missing")

    assert workspace_store._is_retryable_file_access_error(access_denied)
    assert workspace_store._is_retryable_file_access_error(busy)
    assert not workspace_store._is_retryable_file_access_error(wrong_error)


def test_copy_workspace_group_rejects_missing_source_group(tmp_path) -> None:
    source = tmp_path / "source.itws"
    target = tmp_path / "target.itws"
    with h5py.File(source, "w"):
        pass
    with workspace_store.WorkspaceStore(target, create=True) as target_store:
        with (
            target_store.write_session(),
            pytest.raises(KeyError, match="payload group"),
        ):
            workspace_storage._copy_workspace_group(
                target_store,
                source_file=str(source),
                source_path="/missing",
                target_path=target_store.object_path("missing"),
            )
        assert target_store.object_path("missing").strip("/") not in (
            target_store.h5_file
        )


def test_workspace_generation_keeps_existing_preserved_group(tmp_path) -> None:
    path = tmp_path / "workspace.itws"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest={"schema_version": 5, "nodes": [], "root_order": []},
        objects=(),
        preserved_groups=(
            workspace_storage._WorkspaceGroupCopy(
                source_file=str(tmp_path / "unused.itws"),
                source_path="/unused",
                target_path="/preserved",
            ),
        ),
    )
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file.create_group("preserved")
        generation = workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )

        assert generation.sequence == 1
        assert "preserved" in store.h5_file


def test_workspace_storage_rejects_changed_publication_and_nonworkspace_recovery(
    tmp_path,
) -> None:
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["plain"] = True
    expected = workspace_storage._workspace_publication_state(path)
    path.write_bytes(b"changed")

    with pytest.raises(workspace_storage._WorkspacePublicationConflictError):
        workspace_storage._require_workspace_publication_state(path, expected)

    workspace_storage._recover_workspace_transactions(tmp_path / "missing.itws")
    workspace_storage._recover_workspace_transactions(tmp_path / "not-workspace.txt")
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["plain"] = True
    workspace_storage._recover_workspace_transactions(path)


def test_open_workspace_h5_file_for_update_uses_legacy_path(tmp_path) -> None:
    path = tmp_path / "legacy.h5"
    with workspace_storage._open_workspace_h5_file_for_update(path) as h5_file:
        h5_file.attrs["updated"] = True
    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs["updated"]


def test_workspace_recovery_discards_prepared_legacy_transaction(tmp_path) -> None:
    fname = tmp_path / "prepared.itws"
    _write_transaction_test_workspace(fname)
    with h5py.File(fname, "a") as h5_file:
        _create_legacy_transaction(h5_file, "prepared", status="preparing")

    workspace_storage._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


@pytest.mark.parametrize("replacement_moved", [False, True])
def test_workspace_recovery_rolls_back_legacy_group_replacement(
    tmp_path, *, replacement_moved: bool
) -> None:
    fname = tmp_path / "committing.itws"
    _write_transaction_test_workspace(fname)
    backup_path = "__itws_backup_committing/imagetool"
    operation = {
        "group_path": "0/imagetool",
        "pending_path": "__itws_pending_committing/imagetool",
        "backup_path": backup_path,
        "old_exists": True,
    }
    with h5py.File(fname, "a") as h5_file:
        _create_legacy_transaction(
            h5_file,
            "committing",
            status="committing",
            group_replacements=[operation],
        )
        h5_file.move("0/imagetool", backup_path)
        if replacement_moved:
            workspace_arrays._write_workspace_dataset_group_to_file(
                h5_file,
                "0/imagetool",
                _transaction_test_dataset(2.0, title="new"),
            )

    workspace_storage._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_keeps_committed_legacy_replacement(tmp_path) -> None:
    fname = tmp_path / "committed.itws"
    _write_transaction_test_workspace(fname)
    backup_path = "__itws_backup_committed/imagetool"
    operation = {
        "group_path": "0/imagetool",
        "pending_path": "__itws_pending_committed/imagetool",
        "backup_path": backup_path,
        "old_exists": True,
    }
    with h5py.File(fname, "a") as h5_file:
        _create_legacy_transaction(
            h5_file,
            "committed",
            status="committed",
            group_replacements=[operation],
        )
        h5_file.move("0/imagetool", backup_path)
        workspace_arrays._write_workspace_dataset_group_to_file(
            h5_file,
            "0/imagetool",
            _transaction_test_dataset(2.0, title="new"),
        )

    workspace_storage._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 2.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_restores_legacy_attribute_backups(tmp_path) -> None:
    fname = tmp_path / "attrs.itws"
    _write_transaction_test_workspace(fname)
    with h5py.File(fname, "a") as h5_file:
        txn_group, _pending_root, _backup_root = _create_legacy_transaction(
            h5_file, "attrs", status="committing"
        )
        workspace_storage._write_workspace_attr_backup(txn_group, 0, "/", h5_file.attrs)
        workspace_storage._write_workspace_attr_backup(
            txn_group, 1, "0/imagetool", h5_file["0/imagetool"].attrs
        )
        workspace_arrays._replace_h5_attrs(
            h5_file.attrs, _transaction_test_root_attrs(delta_save_count=1)
        )
        h5_file["0/imagetool"].attrs["itool_title"] = "new"

    workspace_storage._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert h5_file["0/imagetool"].attrs["itool_title"] == "old"
        manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert "delta_save_count" not in manifest
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_cleans_orphan_internal_groups(tmp_path) -> None:

    fname = tmp_path / "orphan-internal.itws"
    _write_transaction_test_workspace(fname)
    with h5py.File(fname, "a") as h5_file:
        h5_file.create_group("__itws_pending_orphan")
        h5_file.create_group("__itws_backup_orphan")

    workspace_storage._recover_workspace_transactions(fname)

    _assert_no_workspace_internal_groups(fname)


def test_workspace_lock_path_uses_hidden_sidecar(tmp_path) -> None:
    fname = tmp_path / "example.itws"

    assert workspace_storage._workspace_lock_path(fname) == str(
        (tmp_path / ".example.itws.lock").resolve()
    )


def test_workspace_lock_conflict_is_reported(tmp_path) -> None:
    fname = tmp_path / "locked.itws"
    _write_transaction_test_workspace(fname)
    hidden_lock_path = pathlib.Path(workspace_storage._workspace_lock_path(fname))
    visible_lock_path = pathlib.Path(f"{fname.resolve()}.lock")
    lock = workspace_storage._acquire_workspace_document_lock(fname)
    try:
        assert lock.staleLockTime() == 0
        assert hidden_lock_path.exists()
        assert not visible_lock_path.exists()
        with pytest.raises(BlockingIOError):
            workspace_storage._acquire_workspace_document_lock(fname)
    finally:
        lock.unlock()


def test_hide_workspace_lock_file_sets_macos_hidden_flag(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    lock_path = "/workspace/.workspace.itws.lock"
    regular_stat = types.SimpleNamespace(st_mode=0o100600)

    monkeypatch.setattr(workspace_storage.sys, "platform", "darwin")
    monkeypatch.setattr(workspace_storage.os, "lstat", lambda _path: regular_stat)
    monkeypatch.setattr(
        workspace_storage.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    workspace_storage._hide_workspace_lock_file(lock_path)

    assert calls == [(lock_path, 0x8000)]


def test_hide_workspace_lock_file_skips_macos_symlink(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    symlink_stat = types.SimpleNamespace(st_mode=0o120777)

    monkeypatch.setattr(workspace_storage.sys, "platform", "darwin")
    monkeypatch.setattr(workspace_storage.os, "lstat", lambda _path: symlink_stat)
    monkeypatch.setattr(
        workspace_storage.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    workspace_storage._hide_workspace_lock_file("/workspace/.workspace.itws.lock")

    assert calls == []


def test_workspace_lock_error_message_names_owner(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "busy-message.itws"
    _write_transaction_test_workspace(fname)
    lock = workspace_storage._acquire_workspace_document_lock(fname)
    lock_info = workspace_storage._workspace_document_lock_info(fname)
    calls: list[dict[str, object]] = []

    def _critical(*args, **kwargs) -> int:
        calls.append({"args": args, "kwargs": kwargs})
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)
    try:
        manager_widgets._show_workspace_file_lock_error(None, fname)
    finally:
        lock.unlock()

    assert len(calls) == 1
    args = calls[0]["args"]
    assert isinstance(args, tuple)
    assert args[1] == "Workspace Already Open"
    assert args[2] == "This workspace is already open somewhere else."
    informative_text = args[3]
    assert isinstance(informative_text, str)
    assert fname.name in informative_text
    assert "lock" not in informative_text.casefold()
    if lock_info.owner:
        assert lock_info.owner in informative_text
    if lock_info.hostname:
        assert lock_info.hostname in informative_text
    detailed_text = calls[0]["kwargs"]["detailed_text"]
    assert isinstance(detailed_text, str)
    assert "Temporary workspace ownership marker:" in detailed_text
    assert lock_info.path in detailed_text


def test_workspace_lock_text_variants(tmp_path) -> None:
    app_only = workspace_storage._WorkspaceDocumentLockInfo(
        path="marker",
        owner="user",
        hostname="",
        appname="ImageTool",
        pid=None,
    )
    pid_only = workspace_storage._WorkspaceDocumentLockInfo(
        path="marker",
        owner="",
        hostname="",
        appname="",
        pid=123,
    )
    full_info = workspace_storage._WorkspaceDocumentLockInfo(
        path="marker",
        owner="user",
        hostname="workstation",
        appname="ImageTool",
        pid=123,
    )

    assert manager_widgets._workspace_lock_owner_text(app_only) == (
        "user using ImageTool"
    )
    assert manager_widgets._workspace_lock_owner_text(pid_only) == ("using process 123")
    assert manager_widgets._workspace_lock_owner_text(full_info) == (
        "user on workstation using ImageTool (process 123)"
    )

    def _raise_owner_details_failed() -> None:
        raise RuntimeError("owner details failed")

    def _details_from_active_exception() -> str:
        try:
            _raise_owner_details_failed()
        except RuntimeError:
            return manager_widgets._workspace_lock_details_text(
                tmp_path / "workspace.itws", full_info
            )

    details = _details_from_active_exception()

    assert "owner details failed" in details
    assert "Temporary workspace ownership marker: marker" in details


def test_workspace_lock_error_message_without_owner(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "busy-message.itws"
    calls: list[dict[str, object]] = []
    lock_info = workspace_storage._WorkspaceDocumentLockInfo(
        path=str(tmp_path / ".busy-message.itws.lock"),
        owner="",
        hostname="",
        appname="",
        pid=None,
    )

    def _critical(*args, **kwargs) -> int:
        calls.append({"args": args, "kwargs": kwargs})
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        workspace_storage, "_workspace_document_lock_info", lambda _fname: lock_info
    )
    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)

    manager_widgets._show_workspace_file_lock_error(None, fname)

    args = calls[0]["args"]
    assert isinstance(args, tuple)
    informative_text = args[3]
    assert isinstance(informative_text, str)
    assert informative_text == (
        "Close the other ImageTool Manager that has busy-message.itws open, "
        "then try again."
    )


def test_workspace_document_access_releases_lock(tmp_path) -> None:
    class _FakeLock:
        def __init__(self) -> None:
            self.unlock_count = 0

        def unlock(self) -> None:
            self.unlock_count += 1

    lock = _FakeLock()
    access = manager_widgets._WorkspaceDocumentAccess(
        tmp_path / "workspace.itws", manager_widgets._WorkspaceDocumentLock(lock)
    )
    retained = access.retain()
    access.release()
    access.release()
    assert lock.unlock_count == 0
    retained.release()
    retained.release()
    assert lock.unlock_count == 1
    with pytest.raises(RuntimeError, match="released"):
        access.retain()


def test_workspace_lock_error_detection_message_variants() -> None:
    transient = OSError(errno.EACCES, "resource temporarily unavailable")
    assert workspace_storage._is_workspace_file_lock_error(transient)
    assert workspace_storage._is_workspace_file_lock_error(
        RuntimeError("file is already open by another process")
    )
    assert workspace_storage._is_workspace_file_lock_error(
        RuntimeError("unable to lock file")
    )
    assert not workspace_storage._is_workspace_file_lock_error(
        OSError(errno.EINVAL, "resource temporarily unavailable")
    )


def test_hide_workspace_lock_file_windows_paths(monkeypatch) -> None:
    import ctypes

    calls: list[tuple[str, int]] = []

    class _Kernel32:
        @staticmethod
        def SetFileAttributesW(path: str, attrs: int) -> None:
            calls.append((path, attrs))

    monkeypatch.setattr(workspace_storage.sys, "platform", "win32")
    monkeypatch.setattr(workspace_storage.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    workspace_storage._hide_workspace_lock_file("missing-windll.itws.lock")
    assert calls == []

    monkeypatch.setattr(
        ctypes, "windll", types.SimpleNamespace(kernel32=_Kernel32()), raising=False
    )
    workspace_storage._hide_workspace_lock_file("hidden.itws.lock")
    assert calls == [("hidden.itws.lock", 0x2)]


def test_workspace_document_lock_info_without_lock(tmp_path) -> None:
    info = workspace_storage._workspace_document_lock_info(tmp_path / "free.itws")

    assert info.pid is None
    assert info.hostname == ""
    assert info.appname == ""


def test_workspace_h5_transaction_helper_edge_cases(tmp_path) -> None:

    fname = tmp_path / "transaction-helpers.itws"
    with h5py.File(fname, "w") as h5_file:
        h5_file.attrs["imagetool_workspace_schema_version"] = (
            workspace_format._current_workspace_schema_version()
        )
        assert workspace_storage._workspace_txn_attr_target(h5_file, "/missing") is None

        txn = h5_file.create_group(
            f"{_persistence_constants.WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
        )
        txn_name = txn.name.strip("/")
        workspace_storage._restore_workspace_attr_backups(h5_file, txn)

        txn.attrs["operations"] = b'{"group_replacements": []}'
        assert workspace_storage._workspace_transaction_operations(txn) == {
            "group_replacements": []
        }
        txn.attrs["operations"] = "{not-json"
        assert workspace_storage._workspace_transaction_operations(txn) == {}

        txn.attrs["pending_root"] = b"__itws_pending_x"
        txn.attrs["backup_root"] = b"__itws_backup_x"
        assert workspace_storage._workspace_transaction_roots(txn) == (
            "__itws_pending_x",
            "__itws_backup_x",
        )

        workspace_storage._rollback_workspace_group_operations(
            h5_file, {"group_replacements": "not-a-list"}
        )
        workspace_storage._rollback_workspace_group_operations(
            h5_file,
            {"group_replacements": [None, {"group_path": 1, "backup_path": "x"}]},
        )

        target = h5_file.create_group("target")
        target.attrs["value"] = "old"
        txn.attrs["status"] = b"committing"
        txn.attrs["operations"] = json.dumps(
            {
                "group_replacements": [
                    {
                        "group_path": "target",
                        "backup_path": "missing-backup",
                        "old_exists": False,
                    }
                ]
            }
        )
        pending = h5_file.create_group("__itws_pending_x")
        pending.attrs["unused"] = True
        backup = h5_file.create_group("__itws_backup_x")
        backup.attrs["unused"] = True

        workspace_storage._recover_open_workspace_transaction(h5_file, txn.name)

        assert "target" not in h5_file
        assert "__itws_pending_x" not in h5_file
        assert "__itws_backup_x" not in h5_file
        assert txn_name not in h5_file


def test_recover_workspace_transactions_ignores_non_workspace_file(tmp_path) -> None:
    fname = tmp_path / "plain.itws"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_group(
            f"{_persistence_constants.WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
        )

    workspace_storage._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert (
            f"{_persistence_constants.WORKSPACE_TRANSACTION_GROUP_PREFIX}x" in h5_file
        )


def test_workspace_storage_links_and_compares_immutable_payload_groups(
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            legacy = h5_file.create_group("legacy")
            legacy.create_dataset("data", data=np.array([1.0]))
            legacy.create_group("nested")
            equal = h5_file.create_group("equal")
            equal.create_dataset("data", data=np.array([1.0]))
            equal.create_group("nested")
            h5_file.create_group("different").create_dataset(
                "data", data=np.array([2.0])
            )
            h5_file.create_group("different_names").create_dataset(
                "other", data=np.array([1.0])
            )

            assert workspace_storage._link_workspace_group(
                store, source_path="/legacy", target_path="/objects/payload"
            )
            assert not workspace_storage._link_workspace_group(
                store, source_path="/legacy", target_path="/objects/payload"
            )
            assert workspace_storage._workspace_groups_have_equal_datasets(
                h5_file["legacy"], h5_file["objects/payload"]
            )
            assert workspace_storage._workspace_groups_have_equal_datasets(
                h5_file["legacy"], h5_file["equal"]
            )
            assert not workspace_storage._workspace_groups_have_equal_datasets(
                h5_file["legacy"], h5_file["different"]
            )
            assert not workspace_storage._workspace_groups_have_equal_datasets(
                h5_file["legacy"], h5_file["different_names"]
            )
        with pytest.raises(KeyError, match="missing"):
            workspace_storage._link_workspace_group(
                store, source_path="/missing", target_path="/objects/missing"
            )


def test_workspace_blob_and_dataset_comparison_edge_cases(tmp_path) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest={"schema_version": 6, "nodes": []},
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        "blob", blob=b"source", blob_kind="kind"
                    ),
                ),
            ),
            compression_mode="none",
        )
    with h5py.File(path, "r+") as h5_file:
        h5_file[f"{_persistence_constants.WORKSPACE_OBJECTS_GROUP}/blob"].attrs[
            "erlab_object_kind"
        ] = np.bytes_(b"kind")

    assert workspace_storage._read_workspace_blob(path, "blob") == (
        b"source",
        "kind",
    )
    assert not workspace_storage._workspace_dataset_values_equal(
        np.arange(2), np.arange(3)
    )
    assert workspace_storage._workspace_dataset_values_equal(
        np.array(1.0), np.array(1.0)
    )
    assert workspace_storage._workspace_dataset_values_equal(
        np.empty((0, 2)), np.empty((0, 2))
    )
    assert workspace_storage._workspace_dataset_values_equal(
        np.array(["value"], dtype=object),
        np.array(["value"], dtype=object),
    )


@pytest.mark.parametrize(
    ("left", "right", "expected_index"),
    [
        (np.array(1.0), np.array(2.0), ()),
        (np.array(np.nan), np.array(2.0), ()),
        (np.array([False, True]), np.array([False, False]), (1,)),
        (np.array([1, 2], dtype="u8"), np.array([1, 3], dtype="u8"), (1,)),
        (np.array(1, dtype=">i4"), np.array(2, dtype=">i4"), ()),
        (np.array(["a", "b"]), np.array(["a", "c"]), (1,)),
        (np.array(b"a", dtype="S5"), np.array(b"b", dtype="S5"), ()),
        (
            np.array([complex(np.nan, 1), complex(1, np.nan), 2 + 3j]),
            np.array([complex(2, np.nan), complex(np.nan, 9), 2 + 4j]),
            (2,),
        ),
        (np.array(complex(np.nan, 1)), np.array(2 + 1j), ()),
    ],
)
def test_workspace_dataset_comparison_reports_owned_scalar(
    left, right, expected_index
) -> None:
    mismatches: list[tuple[tuple[int, ...], np.ndarray]] = []

    assert not workspace_storage._workspace_dataset_values_equal(
        left,
        right,
        on_mismatch=lambda index, value: mismatches.append((index, value)),
    )

    assert len(mismatches) == 1
    index, value = mismatches[0]
    assert index == expected_index
    assert isinstance(value, np.ndarray)
    assert value.shape == ()
    assert value.dtype == left.dtype
    assert value.tobytes() == np.asarray(left[index], dtype=left.dtype).tobytes()
    assert not np.shares_memory(value, left)
    assert not workspace_storage._workspace_dataset_values_equal(
        value, np.asarray(right[index])
    )


def test_workspace_dataset_comparison_reports_absolute_index_after_first_block() -> (
    None
):
    # uint64 limits each block to fewer than two complete outer-axis slices.
    left = np.zeros((2, 1025, 1024), dtype=np.uint64)
    right = left.copy()
    expected_index = (1, 5, 11)
    left[expected_index] = 7
    right[expected_index] = 8
    right[1, 6, 1] = 9
    mismatches: list[tuple[tuple[int, ...], np.ndarray]] = []

    assert not workspace_storage._workspace_dataset_values_equal(
        left,
        right,
        on_mismatch=lambda index, value: mismatches.append((index, value)),
    )

    assert len(mismatches) == 1
    index, value = mismatches[0]
    assert index == expected_index
    assert value.shape == ()
    assert value.dtype == left.dtype
    left[index] = 10
    assert value.item() == 7


@pytest.mark.parametrize(
    ("left", "right", "expected_equal"),
    [
        (np.array(1.0), np.array(1.0), True),
        (np.array(np.nan), np.array(np.nan), True),
        (
            np.array([complex(np.nan, 1), complex(1, np.nan)]),
            np.array([complex(2, np.nan), complex(np.nan, 9)]),
            True,
        ),
        (np.empty((0, 2)), np.empty((0, 2)), True),
        (np.arange(2), np.arange(3), False),
        (np.array([1], dtype="u1"), np.array([1], dtype="u2"), False),
        (np.array(["a"], dtype=object), np.array(["b"], dtype=object), False),
        (
            np.array([(1,)], dtype=[("value", "i4")]),
            np.array([(2,)], dtype=[("value", "i4")]),
            False,
        ),
    ],
)
def test_workspace_dataset_comparison_without_scalar_witness(
    left, right, expected_equal
) -> None:
    mismatches: list[tuple[tuple[int, ...], np.ndarray]] = []

    assert (
        workspace_storage._workspace_dataset_values_equal(
            left,
            right,
            on_mismatch=lambda index, value: mismatches.append((index, value)),
        )
        is expected_equal
    )
    assert mismatches == []
    assert (
        workspace_storage._workspace_dataset_values_equal(left, right) is expected_equal
    )


def test_rebind_equivalent_legacy_readers_ignores_unmapped_reader(tmp_path) -> None:
    path = tmp_path / "groups.h5"
    with h5py.File(path, "w") as h5_file:
        h5_file.create_group("legacy")
        rebound: list[dict[str, str]] = []

        class _Store:
            write_lock = contextlib.nullcontext()
            lock = contextlib.nullcontext()
            leased_legacy_group_paths = frozenset({"/legacy"})

            @staticmethod
            def read_session():
                return contextlib.nullcontext(h5_file)

            @staticmethod
            def object_path(object_id: str) -> str:
                return f"/objects/{object_id}"

            @staticmethod
            def rebind_legacy_readers(mappings) -> None:
                rebound.append(mappings)

        store = typing.cast("workspace_store.WorkspaceStore", _Store())
        workspace_storage._rebind_equivalent_legacy_readers(
            store, {"schema_version": 6, "nodes": []}
        )
        workspace_storage._rebind_equivalent_legacy_readers(
            store,
            {
                "schema_version": 6,
                "nodes": [
                    {
                        "path": "0",
                        "kind": "imagetool",
                        "payload_object_id": "payload",
                    }
                ],
            },
        )
        assert rebound == [{}]


def _save_as_seed_manifest(*object_ids: str) -> dict[str, typing.Any]:
    return {
        "schema_version": _persistence_constants.WORKSPACE_SCHEMA_VERSION,
        "nodes": [
            {
                "uid": f"n{index}",
                "path": str(index),
                "kind": "imagetool",
                "payload_object_id": object_id,
                "payload_path": workspace_store.WorkspaceStore.object_path(object_id),
            }
            for index, object_id in enumerate(object_ids)
        ],
        "root_order": [f"n{index}" for index in range(len(object_ids))],
    }


def _save_as_seed_dataset() -> xr.Dataset:
    bits = np.array(
        [
            0x7FF8000000000042,
            0x8000000000000000,
            0x3FF0000000000000,
            0xBFF0000000000000,
            0x7FF0000000000000,
            0x0000000000000001,
        ],
        dtype=np.uint64,
    )
    return xr.Dataset(
        {
            "data": (
                ("x", "y"),
                bits.view(np.float64).reshape(2, 3),
                {"units": "counts"},
            )
        },
        coords={"x": [1.0, 2.0], "y": [3.0, 4.0, 5.0], "hv": 72.0},
        attrs={"sample": "source"},
    )


@pytest.fixture
def save_as_seed_source(tmp_path) -> pathlib.Path:
    path = tmp_path / "source.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            assert workspace_arrays._write_workspace_dataset_group_h5py(
                h5_file,
                store.object_path("kept"),
                _save_as_seed_dataset(),
                encoding={
                    "data": {
                        "compression": "gzip",
                        "compression_opts": 1,
                        "chunksizes": (1, 3),
                    }
                },
            )
            obsolete = h5_file.require_group(store.object_path("obsolete"))
            obsolete.create_dataset("data", data=[9.0])
        store.publish(_save_as_seed_manifest("kept", "obsolete"))
        store.publish(_save_as_seed_manifest("kept"))
        with store.write_session() as h5_file:
            h5_file[_persistence_constants.WORKSPACE_STAGING_GROUP].create_group(
                "unfinished"
            )
    return path


def _copy_workspace_file(src, dst, *, allow_byte_copy=True) -> bool:
    """Stand in for a successful clone when testing document preparation."""
    shutil.copyfile(src, dst)
    return True


def _save_as_seed_plan(
    source: pathlib.Path,
) -> workspace_storage._WorkspaceGenerationPlan:
    return workspace_storage._WorkspaceGenerationPlan(
        manifest=_save_as_seed_manifest("kept"),
        objects=(
            workspace_storage._WorkspaceObjectWrite(
                "kept",
                source_file=str(source),
                source_path=workspace_store.WorkspaceStore.object_path("kept"),
            ),
        ),
    )


@pytest.mark.parametrize("fixed_version", [False, True])
def test_workspace_unchanged_generation_never_opens_a_writer(
    tmp_path, monkeypatch, fixed_version
) -> None:
    path = tmp_path / "unchanged.itws"
    manifest = _save_as_seed_manifest("kept")
    manifest["metadata"] = {"nan": float("nan"), "values": (1, 2, 3)}
    with workspace_store.WorkspaceStore(path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(
                manifest=manifest,
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        "kept", dataset=_save_as_seed_dataset()
                    ),
                ),
            ),
            compression_mode="none",
        )
        manifest.pop("schema_version")
        manifest["generation"] = 100
        plan = replace(_save_as_seed_plan(path), manifest=manifest)
        if fixed_version:
            with store.write_session() as h5_file:
                del h5_file.attrs["erlab_version"]
                h5_file.attrs.create(
                    "erlab_version", str(erlab.__version__).encode(), dtype="S64"
                )
        before = path.read_bytes()
        current = store.current_generation()

        @contextlib.contextmanager
        def _unexpected_write(**_kwargs):
            raise AssertionError("Unchanged saves must stay read-only")
            yield  # pragma: no cover

        monkeypatch.setattr(store, "write_session", _unexpected_write)
        for _ in range(25):
            saved = workspace_storage._write_workspace_generation(
                store, plan, compression_mode="none"
            )
            assert saved.sequence == current.sequence
            assert path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        "metadata",
        "version",
        "schema",
        "missing_object",
        "checksum",
        "manifest_text",
        "manifest_type",
        "dangling_generation",
        "missing_preserved_group",
    ],
)
def test_workspace_unchanged_generation_does_not_skip_required_writes(
    tmp_path, change
) -> None:
    path = tmp_path / "changed.itws"
    manifest = _save_as_seed_manifest("kept")
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file.require_group(store.object_path("kept"))
        store.publish(manifest)
        store.publish(manifest)
        plan = _save_as_seed_plan(path)
        newest = f"/{_persistence_constants.WORKSPACE_GENERATIONS_GROUP}/{2:020d}"
        with store.write_session() as h5_file:
            if change == "metadata":
                plan.manifest["note"] = "Changed metadata"
            elif change == "version":
                h5_file.attrs.modify("erlab_version", "older")
            elif change == "schema":
                h5_file.attrs.modify("imagetool_workspace_schema_version", 0)
            elif change == "missing_object":
                del h5_file[store.object_path("kept")]
                plan = replace(
                    plan,
                    objects=(
                        workspace_storage._WorkspaceObjectWrite(
                            "kept", dataset=_save_as_seed_dataset()
                        ),
                    ),
                )
            elif change == "checksum":
                h5_file[newest]["manifest"].attrs.modify("sha256", b"invalid")
            elif change == "manifest_text":
                h5_file[newest]["manifest"][()] = b"{}"
            elif change == "manifest_type":
                checksum = h5_file[newest]["manifest"].attrs["sha256"]
                del h5_file[newest]["manifest"]
                h5_file[newest].create_group("manifest").attrs["sha256"] = checksum
            elif change == "dangling_generation":
                del h5_file[newest]
                h5_file[newest] = h5py.SoftLink("/missing")
            elif change == "missing_preserved_group":
                h5_file.require_group("legacy-source")
                plan = replace(
                    plan,
                    preserved_groups=(
                        workspace_storage._WorkspaceGroupCopy(
                            str(path), "/legacy-source", "/preserved"
                        ),
                    ),
                )
        saved = workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        assert saved.sequence == 3
        assert store.current_generation().sequence == 3
        with store.read_session() as h5_file:
            assert h5_file.attrs["erlab_version"] == str(erlab.__version__)
            assert (
                h5_file.attrs["imagetool_workspace_schema_version"]
                == _persistence_constants.WORKSPACE_SCHEMA_VERSION
            )
            assert store.object_path("kept") in h5_file
            if change == "missing_preserved_group":
                assert "preserved" in h5_file


def test_workspace_unchanged_generation_keeps_existing_legacy_links(tmp_path) -> None:
    path = tmp_path / "legacy-links.itws"
    manifest = _save_as_seed_manifest("kept")
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file.require_group(store.object_path("kept"))
            h5_file["legacy"] = h5_file[store.object_path("kept")]
        first = store.publish(manifest)
        plan = replace(
            _save_as_seed_plan(path),
            preserved_groups=(
                workspace_storage._WorkspaceGroupCopy(str(path), "/legacy", "/legacy"),
            ),
            legacy_object_links=(("/legacy", "kept"),),
            legacy_reader_rebindings=(("/legacy", "kept"),),
        )
        before = path.read_bytes()
        saved = workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        assert saved.sequence == first.sequence
        assert path.read_bytes() == before


def test_workspace_unchanged_generation_assigns_missing_identity(tmp_path) -> None:
    path = tmp_path / "missing-identity.itws"
    manifest = _save_as_seed_manifest()
    with workspace_store.WorkspaceStore(path, create=True) as store:
        store.publish(manifest)
    with h5py.File(path, "r+") as h5_file:
        del h5_file.attrs[_persistence_constants.WORKSPACE_ID_ATTR]
    with workspace_store.WorkspaceStore(path) as store:
        assert store.workspace_id is None
        saved = workspace_storage._write_workspace_generation(
            store,
            workspace_storage._WorkspaceGenerationPlan(manifest=manifest, objects=()),
            compression_mode="none",
        )
        assert saved.sequence == 2
        assert store.workspace_id is not None


@pytest.mark.parametrize("active_source", [False, True])
def test_workspace_save_as_seed_preserves_payload_and_resets_document(
    monkeypatch, tmp_path, save_as_seed_source, active_source
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    original_bytes = source.read_bytes()
    copy_calls = []

    def _copy_file(src, dst, *, allow_byte_copy=True) -> bool:
        copy_calls.append((pathlib.Path(src), pathlib.Path(dst)))
        shutil.copyfile(src, dst)
        return True

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "copy_workspace_file", _copy_file
    )
    source_context = (
        workspace_store.WorkspaceStore(source)
        if active_source
        else contextlib.nullcontext()
    )
    with source_context:
        assert workspace_storage._seed_workspace_save_as(
            target, _save_as_seed_plan(source)
        )

    assert copy_calls == [(source, target)]
    assert source.read_bytes() == original_bytes
    with h5py.File(source, "r") as old, h5py.File(target, "r") as new:
        assert (
            new.attrs[_persistence_constants.WORKSPACE_ID_ATTR]
            != old.attrs[_persistence_constants.WORKSPACE_ID_ATTR]
        )
        assert set(new[_persistence_constants.WORKSPACE_OBJECTS_GROUP]) == {"kept"}
        assert len(new[_persistence_constants.WORKSPACE_GENERATIONS_GROUP]) == 0
        assert len(new[_persistence_constants.WORKSPACE_STAGING_GROUP]) == 0
        data = new[workspace_store.WorkspaceStore.object_path("kept")]["data"]
        assert data.chunks == (1, 3)
        assert data.compression == "gzip"
        assert data[()].tobytes() == _save_as_seed_dataset()["data"].values.tobytes()
    opened = workspace_arrays.open_workspace_dataset(
        target,
        workspace_store.WorkspaceStore.object_path("kept"),
        chunks=None,
        cache=False,
    )
    try:
        xr.testing.assert_identical(opened, _save_as_seed_dataset())
    finally:
        opened.close()


def test_workspace_save_as_seed_publishes_new_arrays_and_blobs(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    plan = _save_as_seed_plan(source)
    changed = _save_as_seed_dataset().copy(deep=True)
    changed["data"].values[1, 0] = 17.0
    plan = replace(
        plan,
        manifest=_save_as_seed_manifest("kept", "changed", "source-code"),
        objects=(
            *plan.objects,
            workspace_storage._WorkspaceObjectWrite("changed", dataset=changed),
            workspace_storage._WorkspaceObjectWrite(
                "source-code", blob=b"stored source", blob_kind="extension"
            ),
        ),
    )
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        _copy_workspace_file,
    )
    assert workspace_storage._seed_workspace_save_as(target, plan)
    monkeypatch.setattr(
        workspace_storage,
        "_copy_workspace_group",
        lambda *_args, **_kwargs: pytest.fail("Seeded payload was copied again"),
    )
    with workspace_store.WorkspaceStore(target) as store:
        assert store.generations() == ()
        generation = workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        assert generation.sequence == 1
        assert generation.manifest["nodes"] == plan.manifest["nodes"]
        assert set(store.h5_file[_persistence_constants.WORKSPACE_OBJECTS_GROUP]) == {
            "kept",
            "changed",
            "source-code",
        }
        assert store.h5_file[store.object_path("changed")]["data"][1, 0] == 17.0
    assert workspace_storage._read_workspace_blob(target, "source-code") == (
        b"stored source",
        "extension",
    )
    with h5py.File(source, "r") as original:
        assert (
            original[workspace_store.WorkspaceStore.object_path("kept")]["data"][1, 0]
            == -1.0
        )


@pytest.mark.parametrize(
    "unsafe_plan",
    [
        "no-reused-object",
        "mixed-source",
        "noncanonical-copy",
        "duplicate-object-id",
        "legacy-group",
        "legacy-link",
        "legacy-reader",
        "missing-object",
    ],
)
def test_workspace_save_as_seed_uses_group_copy_for_unsafe_plans(
    monkeypatch, tmp_path, save_as_seed_source, unsafe_plan
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    plan = _save_as_seed_plan(source)
    if unsafe_plan == "no-reused-object":
        plan = replace(plan, objects=())
    elif unsafe_plan == "mixed-source":
        plan = replace(
            plan,
            objects=(
                *plan.objects,
                workspace_storage._WorkspaceObjectWrite(
                    "other",
                    source_file=str(tmp_path / "other.itws"),
                    source_path=workspace_store.WorkspaceStore.object_path("other"),
                ),
            ),
        )
    elif unsafe_plan == "noncanonical-copy":
        plan = replace(plan, objects=(replace(plan.objects[0], source_path="/legacy"),))
    elif unsafe_plan == "duplicate-object-id":
        plan = replace(plan, objects=(*plan.objects, plan.objects[0]))
    elif unsafe_plan == "legacy-group":
        plan = replace(
            plan,
            preserved_groups=(
                workspace_storage._WorkspaceGroupCopy(
                    source_file=str(source),
                    source_path="/legacy",
                    target_path="/legacy",
                ),
            ),
        )
    elif unsafe_plan == "legacy-link":
        plan = replace(plan, legacy_object_links=(("/legacy", "kept"),))
    elif unsafe_plan == "legacy-reader":
        plan = replace(plan, legacy_reader_rebindings=(("/legacy", "kept"),))
    else:
        plan = replace(
            plan,
            objects=(
                replace(
                    plan.objects[0],
                    object_id="missing",
                    source_path=workspace_store.WorkspaceStore.object_path("missing"),
                ),
            ),
        )
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        lambda *_args, **_kwargs: pytest.fail("Unsafe plan used whole-file copying"),
    )
    assert not workspace_storage._seed_workspace_save_as(target, plan)
    assert not target.exists()


@pytest.mark.parametrize(
    "source_change", ["legacy-schema", "extra-root", "no-generation"]
)
def test_workspace_save_as_seed_uses_group_copy_for_unsupported_documents(
    monkeypatch, tmp_path, save_as_seed_source, source_change
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    with h5py.File(source, "a") as h5_file:
        if source_change == "legacy-schema":
            h5_file.attrs["imagetool_workspace_schema_version"] = 3
        elif source_change == "extra-root":
            h5_file.create_group("legacy")
        else:
            for key in list(
                h5_file[_persistence_constants.WORKSPACE_GENERATIONS_GROUP]
            ):
                del h5_file[_persistence_constants.WORKSPACE_GENERATIONS_GROUP][key]
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        lambda *_args, **_kwargs: pytest.fail(
            "Unsupported document used whole-file copying"
        ),
    )
    assert not workspace_storage._seed_workspace_save_as(
        target, _save_as_seed_plan(source)
    )
    assert not target.exists()


def test_workspace_save_as_seed_preserves_existing_target(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    target = tmp_path / "prepared.itws"
    target.write_bytes(b"existing destination")
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        lambda *_args, **_kwargs: pytest.fail("Existing destination was overwritten"),
    )
    assert not workspace_storage._seed_workspace_save_as(
        target, _save_as_seed_plan(save_as_seed_source)
    )
    assert target.read_bytes() == b"existing destination"


def test_workspace_save_as_seed_removes_partial_copy_on_io_error(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    original = source.read_bytes()

    def _fail_copy(_src, dst, *, allow_byte_copy=True) -> None:
        pathlib.Path(dst).write_bytes(b"partial")
        raise OSError(errno.EIO, "copy failed")

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "copy_workspace_file", _fail_copy
    )
    assert not workspace_storage._seed_workspace_save_as(
        target, _save_as_seed_plan(source)
    )
    assert not target.exists()
    assert source.read_bytes() == original


def test_workspace_save_as_seed_rejects_source_changes_during_copy(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    original = source.read_bytes()
    previous_stat = source.stat()

    def _copy_then_change_source(src, dst, *, allow_byte_copy=True) -> bool:
        shutil.copyfile(src, dst)
        os.utime(
            src,
            ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000),
        )
        return True

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        _copy_then_change_source,
    )
    with pytest.raises(workspace_storage._WorkspacePublicationConflictError):
        workspace_storage._seed_workspace_save_as(target, _save_as_seed_plan(source))
    assert not target.exists()
    assert source.read_bytes() == original


def test_workspace_save_as_seed_removes_copy_that_cannot_be_prepared(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    target = tmp_path / "prepared.itws"

    def _bad_copy(_src, dst, *, allow_byte_copy=True) -> bool:
        pathlib.Path(dst).write_bytes(b"not an HDF5 file")
        return True

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "copy_workspace_file", _bad_copy
    )
    with pytest.raises(OSError, match="file signature"):
        workspace_storage._seed_workspace_save_as(
            target, _save_as_seed_plan(save_as_seed_source)
        )
    assert not target.exists()


@pytest.mark.parametrize("new_object_kind", ["dataset", "blob"])
def test_workspace_save_as_seed_replaces_old_objects_with_new_payloads(
    monkeypatch, tmp_path, save_as_seed_source, new_object_kind
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    plan = _save_as_seed_plan(source)
    changed = _save_as_seed_dataset()
    changed["data"].values[1, 0] = 17.0
    replacement = (
        workspace_storage._WorkspaceObjectWrite("obsolete", dataset=changed)
        if new_object_kind == "dataset"
        else workspace_storage._WorkspaceObjectWrite(
            "obsolete", blob=b"new source", blob_kind="extension"
        )
    )
    plan = replace(
        plan,
        manifest=_save_as_seed_manifest("kept", "obsolete"),
        objects=(*plan.objects, replacement),
    )
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        _copy_workspace_file,
    )
    assert workspace_storage._seed_workspace_save_as(target, plan)
    with h5py.File(target, "r") as copied:
        assert set(copied[_persistence_constants.WORKSPACE_OBJECTS_GROUP]) == {"kept"}
    with workspace_store.WorkspaceStore(target) as store:
        workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        if new_object_kind == "dataset":
            assert store.h5_file[store.object_path("obsolete")]["data"][1, 0] == 17.0
    if new_object_kind == "blob":
        assert workspace_storage._read_workspace_blob(target, "obsolete") == (
            b"new source",
            "extension",
        )
    with h5py.File(source, "r") as original:
        assert (
            original[workspace_store.WorkspaceStore.object_path("obsolete")]["data"][0]
            == 9.0
        )


@pytest.mark.parametrize(
    "source_state", ["writing", "conflicted", "recovery", "unlocked"]
)
def test_workspace_save_as_seed_uses_group_copy_for_unstable_stores(
    monkeypatch, tmp_path, save_as_seed_source, source_state
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        lambda *_args, **_kwargs: pytest.fail(
            "Unstable source used whole-file copying"
        ),
    )
    with workspace_store.WorkspaceStore(source) as store:
        if source_state == "conflicted":
            store._mark_conflicted()
        elif source_state == "recovery":
            recovery = tmp_path / "recovery.itws"
            shutil.copyfile(source, recovery)
            store._use_recovery_source(recovery)
        elif source_state == "unlocked":
            store._locking_supported = False
        source_context = (
            store.write_session()
            if source_state == "writing"
            else contextlib.nullcontext()
        )
        with source_context:
            assert not workspace_storage._seed_workspace_save_as(
                target, _save_as_seed_plan(source)
            )
    assert not target.exists()


def test_workspace_save_as_seed_uses_group_copy_after_source_replacement(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        lambda *_args, **_kwargs: pytest.fail(
            "Replaced source used whole-file copying"
        ),
    )
    with workspace_store.WorkspaceStore(source):
        replacement = tmp_path / "replacement.itws"
        shutil.copyfile(source, replacement)
        replacement.replace(source)
        assert not workspace_storage._seed_workspace_save_as(
            target, _save_as_seed_plan(source)
        )
    assert not target.exists()


def test_workspace_save_as_seed_allows_source_reads_during_file_copy(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    with (
        workspace_store.WorkspaceStore(source) as store,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):

        def _read_source() -> float:
            with store.read_session() as h5_file:
                data = h5_file[store.object_path("kept")]["data"]
                return float(data[0, 2])

        def _copy_while_reading(src, dst, *, allow_byte_copy=True) -> bool:
            assert executor.submit(_read_source).result(timeout=5.0) == 1.0
            shutil.copyfile(src, dst)
            return True

        monkeypatch.setattr(
            workspace_storage.workspace_file_copy,
            "copy_workspace_file",
            _copy_while_reading,
        )
        assert workspace_storage._seed_workspace_save_as(
            target, _save_as_seed_plan(source)
        )


def test_workspace_save_as_seed_uses_group_copy_after_removing_current_rows(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    target = tmp_path / "prepared.itws"
    plan = replace(
        _save_as_seed_plan(save_as_seed_source), manifest=_save_as_seed_manifest()
    )
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        lambda *_args, **_kwargs: pytest.fail("Row removal used whole-file copying"),
    )
    assert not workspace_storage._seed_workspace_save_as(target, plan)
    assert not target.exists()


def test_workspace_save_as_seed_preserves_target_created_during_copy(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    target = tmp_path / "prepared.itws"

    def _target_created_elsewhere(_src, dst, *, allow_byte_copy=True) -> None:
        pathlib.Path(dst).write_bytes(b"created elsewhere")
        raise FileExistsError(errno.EEXIST, "target exists", dst)

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        _target_created_elsewhere,
    )
    with pytest.raises(FileExistsError, match="target exists"):
        workspace_storage._seed_workspace_save_as(
            target, _save_as_seed_plan(save_as_seed_source)
        )
    assert target.read_bytes() == b"created elsewhere"


def test_workspace_save_as_seed_identifies_missing_backing_file(tmp_path) -> None:
    source = tmp_path / "missing.itws"
    target = tmp_path / "prepared.itws"
    with pytest.raises(
        workspace_storage._WorkspaceBackingFileNotFoundError
    ) as exc_info:
        workspace_storage._seed_workspace_save_as(target, _save_as_seed_plan(source))
    assert exc_info.value.source_path == str(source)
    assert not target.exists()


@pytest.mark.parametrize("lookup_state", ["closed", "switched"])
def test_workspace_save_as_seed_declines_stale_store_lookup(
    monkeypatch, tmp_path, save_as_seed_source, lookup_state
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    store = workspace_store.WorkspaceStore(source)
    if lookup_state == "closed":
        store.close()
    else:
        switched = tmp_path / "switched.itws"
        shutil.copyfile(source, switched)
        store.switch_path(switched)
    monkeypatch.setattr(
        workspace_store.WorkspaceStore,
        "active",
        classmethod(lambda _cls, _path: store),
    )
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        lambda *_args, **_kwargs: pytest.fail("Stale store used whole-file copying"),
    )
    try:
        assert not workspace_storage._seed_workspace_save_as(
            target, _save_as_seed_plan(source)
        )
    finally:
        store.close()
    assert not target.exists()


@pytest.mark.parametrize("manifest_owner", ["source", "target"])
@pytest.mark.parametrize("invalid_uid", ["missing", "nonstring", "duplicate"])
def test_workspace_save_as_seed_declines_invalid_node_identities(
    monkeypatch, tmp_path, save_as_seed_source, manifest_owner, invalid_uid
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    manifest = _save_as_seed_manifest("kept")
    if invalid_uid == "missing":
        del manifest["nodes"][0]["uid"]
    elif invalid_uid == "nonstring":
        manifest["nodes"][0]["uid"] = 0
    else:
        manifest["nodes"].append({**manifest["nodes"][0], "path": "1"})
    plan = _save_as_seed_plan(source)
    if manifest_owner == "source":
        with workspace_store.WorkspaceStore(source) as store:
            store.publish(manifest)
    else:
        plan = replace(plan, manifest=manifest)
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        lambda *_args, **_kwargs: pytest.fail(
            "Invalid node identities used whole-file copying"
        ),
    )
    assert not workspace_storage._seed_workspace_save_as(target, plan)
    assert not target.exists()


@pytest.mark.parametrize("active_source", [False, True])
@pytest.mark.parametrize("source_change", ["mtime", "replacement"])
def test_workspace_save_as_seed_rejects_changes_during_source_inspection(
    monkeypatch, tmp_path, save_as_seed_source, active_source, source_change
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    previous_stat = source.stat()
    read_attrs = workspace_arrays._read_workspace_root_attrs_h5py

    def _read_then_change_source(path):
        attrs = read_attrs(path)
        if source_change == "mtime":
            os.utime(
                source,
                ns=(
                    previous_stat.st_atime_ns,
                    previous_stat.st_mtime_ns + 1_000_000_000,
                ),
            )
        else:
            replacement = tmp_path / "replacement.itws"
            shutil.copyfile(source, replacement)
            try:
                replacement.replace(source)
            except PermissionError:
                pytest.skip("The filesystem does not allow replacing the open source")
        return attrs

    monkeypatch.setattr(
        workspace_arrays, "_read_workspace_root_attrs_h5py", _read_then_change_source
    )
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        _copy_workspace_file,
    )
    source_context = (
        workspace_store.WorkspaceStore(source)
        if active_source
        else contextlib.nullcontext()
    )
    with (
        source_context,
        pytest.raises(workspace_storage._WorkspacePublicationConflictError),
    ):
        workspace_storage._seed_workspace_save_as(target, _save_as_seed_plan(source))
    assert not target.exists()


def test_workspace_save_as_seed_retries_busy_target_open(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    original_bytes = source.read_bytes()
    real_file = h5py.File
    source_modes: list[str] = []
    target_modes: list[str] = []
    delays: list[float] = []
    contention_calls: list[bool] = []

    def _open_file(path, mode="r", **kwargs):
        if pathlib.Path(path) == target and mode == "r+":
            target_modes.append(mode)
            if len(target_modes) == 1:
                raise OSError(errno.EAGAIN, "busy")
        elif pathlib.Path(path) == source:
            source_modes.append(mode)
        return real_file(path, mode, **kwargs)

    monkeypatch.setattr(workspace_storage.h5py, "File", _open_file)
    monkeypatch.setattr(workspace_store.time, "sleep", delays.append)
    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        _copy_workspace_file,
    )
    assert workspace_storage._seed_workspace_save_as(
        target,
        _save_as_seed_plan(source),
        on_contention=lambda: contention_calls.append(True),
    )

    assert target_modes == ["r+", "r+"]
    assert source_modes
    assert set(source_modes) == {"r"}
    assert delays == [workspace_store._HDF5_CONTENTION_RETRY_DELAYS[0]]
    assert contention_calls == [True]
    assert source.read_bytes() == original_bytes
    with real_file(target, "r") as copied:
        assert len(copied[_persistence_constants.WORKSPACE_GENERATIONS_GROUP]) == 0
        assert set(copied[_persistence_constants.WORKSPACE_OBJECTS_GROUP]) == {"kept"}


def test_workspace_save_as_seed_rechecks_ownership_after_source_state_capture(
    monkeypatch, tmp_path, save_as_seed_source
) -> None:
    source = save_as_seed_source
    target = tmp_path / "prepared.itws"
    publication_state = workspace_storage._workspace_publication_state
    replaced = False
    copy_calls: list[tuple[pathlib.Path, pathlib.Path]] = []

    def _replace_before_state_capture(path):
        nonlocal replaced
        if pathlib.Path(path) == source and not replaced:
            replaced = True
            replacement = tmp_path / "replacement.itws"
            shutil.copyfile(source, replacement)
            with h5py.File(replacement, "r+") as h5_file:
                data = h5_file[workspace_store.WorkspaceStore.object_path("kept")][
                    "data"
                ]
                data[0, 2] = 999.0
            try:
                replacement.replace(source)
            except PermissionError:
                pytest.skip("The filesystem does not allow replacing the open source")
        return publication_state(path)

    def _copy_file(src, dst, *, allow_byte_copy=True) -> bool:
        copy_calls.append((pathlib.Path(src), pathlib.Path(dst)))
        shutil.copyfile(src, dst)
        return True

    with workspace_store.WorkspaceStore(source) as store:
        # Retain the old HDF5 handle while another process replaces the path.
        store.current_generation()
        monkeypatch.setattr(
            workspace_storage,
            "_workspace_publication_state",
            _replace_before_state_capture,
        )
        monkeypatch.setattr(
            workspace_storage.workspace_file_copy, "copy_workspace_file", _copy_file
        )
        with pytest.raises(
            workspace_storage._WorkspacePublicationConflictError
        ) as exc_info:
            workspace_storage._seed_workspace_save_as(
                target, _save_as_seed_plan(source)
            )

    assert exc_info.value.path == str(source)
    assert replaced
    assert copy_calls == []
    assert not target.exists()
    with h5py.File(source, "r") as external_source:
        assert (
            external_source[workspace_store.WorkspaceStore.object_path("kept")]["data"][
                0, 2
            ]
            == 999.0
        )


def _save_as_tiny_reuse_source(tmp_path):
    source = tmp_path / "large-rewritten-source.itws"
    with workspace_store.WorkspaceStore(source, create=True) as store:
        with store.write_session() as h5_file:
            old = h5_file.require_group(store.object_path("old-large"))
            old.create_dataset("data", data=np.arange(1 << 18, dtype=np.float64))
            tiny = h5_file.require_group(store.object_path("tiny"))
            tiny.create_dataset("data", data=np.array([11.0, 12.0]))
        store.publish(_save_as_seed_manifest("old-large", "tiny"))
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_save_as_seed_manifest("new-small", "tiny"),
        objects=(
            workspace_storage._WorkspaceObjectWrite(
                "new-small", dataset=xr.Dataset({"data": ("x", [31.0, 32.0])})
            ),
            workspace_storage._WorkspaceObjectWrite(
                "tiny",
                source_file=str(source),
                source_path=workspace_store.WorkspaceStore.object_path("tiny"),
            ),
        ),
    )
    return source, plan


def test_workspace_save_as_tiny_reuse_declines_full_copy_and_writes_generation(
    monkeypatch, tmp_path
):
    source, plan = _save_as_tiny_reuse_source(tmp_path)
    target = tmp_path / "tiny-reuse.itws"
    original_bytes = source.read_bytes()
    allow_byte_copy_calls = []

    def decline_clone(_source, _target, *, allow_byte_copy):
        allow_byte_copy_calls.append(allow_byte_copy)
        assert not allow_byte_copy
        return False

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "copy_workspace_file", decline_clone
    )
    assert not workspace_storage._seed_workspace_save_as(target, plan)
    assert allow_byte_copy_calls
    assert not any(allow_byte_copy_calls)
    assert not target.exists()
    with workspace_store.WorkspaceStore(target, create=True) as store:
        generation = workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        assert generation.manifest["nodes"] == plan.manifest["nodes"]
        assert set(store.h5_file[_persistence_constants.WORKSPACE_OBJECTS_GROUP]) == {
            "new-small",
            "tiny",
        }
        np.testing.assert_array_equal(
            store.h5_file[store.object_path("tiny")]["data"], [11.0, 12.0]
        )
        np.testing.assert_array_equal(
            store.h5_file[store.object_path("new-small")]["data"], [31.0, 32.0]
        )
    assert target.stat().st_size < source.stat().st_size / 10
    assert source.read_bytes() == original_bytes


def test_workspace_save_as_tiny_reuse_still_tries_native_clone(monkeypatch, tmp_path):
    source, plan = _save_as_tiny_reuse_source(tmp_path)
    target = tmp_path / "cloned-tiny-reuse.itws"
    original_bytes = source.read_bytes()
    allow_byte_copy_calls = []

    def clone_file(src, dst, *, allow_byte_copy):
        allow_byte_copy_calls.append(allow_byte_copy)
        shutil.copyfile(src, dst)
        return True

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "copy_workspace_file", clone_file
    )
    assert workspace_storage._seed_workspace_save_as(target, plan)
    assert allow_byte_copy_calls
    assert not any(allow_byte_copy_calls)
    with workspace_store.WorkspaceStore(target) as store:
        assert set(store.h5_file[_persistence_constants.WORKSPACE_OBJECTS_GROUP]) == {
            "tiny"
        }
        workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        np.testing.assert_array_equal(
            store.h5_file[store.object_path("tiny")]["data"], [11.0, 12.0]
        )
        np.testing.assert_array_equal(
            store.h5_file[store.object_path("new-small")]["data"], [31.0, 32.0]
        )
    assert source.read_bytes() == original_bytes


def test_workspace_save_as_reuse_cost_uses_compressed_physical_storage(
    monkeypatch, tmp_path
):
    source, plan = _save_as_tiny_reuse_source(tmp_path)
    target = tmp_path / "compressed-reuse.itws"
    with (
        workspace_store.WorkspaceStore(source) as store,
        store.write_session() as h5_file,
    ):
        group = h5_file[store.object_path("tiny")]
        del group["data"]
        data = group.create_dataset(
            "data",
            data=np.zeros(1 << 20, dtype=np.float64),
            compression="gzip",
            compression_opts=9,
            chunks=(8192,),
        )
        assert data.nbytes > source.stat().st_size
        assert data.id.get_storage_size() < source.stat().st_size / 10
    allow_byte_copy_calls = []

    def decline_clone(_source, _target, *, allow_byte_copy):
        allow_byte_copy_calls.append(allow_byte_copy)
        return False

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "copy_workspace_file", decline_clone
    )
    assert not workspace_storage._seed_workspace_save_as(target, plan)
    assert allow_byte_copy_calls
    assert not any(allow_byte_copy_calls)
    assert not target.exists()


@pytest.mark.parametrize("link_kind", ["group", "dataset", "cycle"])
def test_workspace_save_as_reuse_cost_counts_hardlinked_storage_once(
    monkeypatch, tmp_path, link_kind
):
    source, plan = _save_as_tiny_reuse_source(tmp_path)
    target = tmp_path / "hardlinked-reuse.itws"
    linked_ids = []
    with workspace_store.WorkspaceStore(source) as store:
        with store.write_session() as h5_file:
            group = h5_file[store.object_path("tiny")]
            del group["data"]
            group.create_dataset("data", data=np.arange(1 << 17, dtype=np.float64))
            for index in range(8):
                if link_kind == "group":
                    object_id = f"tiny-link-{index}"
                    h5_file[store.object_path(object_id)] = group
                    linked_ids.append(object_id)
                elif link_kind == "dataset":
                    group[f"data-link-{index}"] = group["data"]
                else:
                    group[f"self-link-{index}"] = group
        store.publish(_save_as_seed_manifest("old-large", "tiny", *linked_ids))
    plan = replace(
        plan,
        manifest=_save_as_seed_manifest("new-small", "tiny", *linked_ids),
        objects=(
            *plan.objects,
            *(
                workspace_storage._WorkspaceObjectWrite(
                    object_id,
                    source_file=str(source),
                    source_path=workspace_store.WorkspaceStore.object_path(object_id),
                )
                for object_id in linked_ids
            ),
        ),
    )
    allow_byte_copy_calls = []

    def decline_clone(_source, _target, *, allow_byte_copy):
        allow_byte_copy_calls.append(allow_byte_copy)
        return False

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "copy_workspace_file", decline_clone
    )
    assert not workspace_storage._seed_workspace_save_as(target, plan)
    assert allow_byte_copy_calls
    assert not any(allow_byte_copy_calls)
    assert not target.exists()


@pytest.mark.parametrize(
    "copy_error",
    [errno.EACCES, errno.EPERM, errno.EIO, errno.ENOSPC, getattr(errno, "EDQUOT", 122)],
)
@pytest.mark.parametrize("failed_stage", ["clone", "byte_copy"])
def test_workspace_save_as_copy_error_cleans_partial_copy_and_falls_back(
    monkeypatch, tmp_path, save_as_seed_source, caplog, copy_error, failed_stage
):
    source = save_as_seed_source
    target = tmp_path / "copy-error-fallback.itws"
    original_bytes = source.read_bytes()
    attempts = []

    def fail_after_partial_copy(_source, destination):
        attempts.append(pathlib.Path(destination))
        pathlib.Path(destination).write_bytes(b"partial workspace")
        raise OSError(copy_error, "Copy operation failed")

    def native_copy(src, dst, *, allow_byte_copy):
        if failed_stage == "clone":
            return fail_after_partial_copy(src, dst)
        return False

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "_copy_native", native_copy
    )
    monkeypatch.setattr(workspace_store.time, "sleep", lambda _delay: None)
    if failed_stage == "byte_copy":
        monkeypatch.setattr(
            workspace_storage.workspace_file_copy.shutil,
            "copyfile",
            fail_after_partial_copy,
        )
    plan = _save_as_seed_plan(source)
    with caplog.at_level("DEBUG", logger=workspace_storage.__name__):
        assert not workspace_storage._seed_workspace_save_as(target, plan)
    assert attempts
    assert any(
        record.exc_info
        and isinstance(record.exc_info[1], OSError)
        and record.exc_info[1].errno == copy_error
        for record in caplog.records
    )
    assert not target.exists()
    with workspace_store.WorkspaceStore(target, create=True) as store:
        generation = workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        assert generation.sequence == 1
        assert set(store.h5_file[_persistence_constants.WORKSPACE_OBJECTS_GROUP]) == {
            "kept"
        }
        assert (
            store.h5_file[store.object_path("kept")]["data"][()].tobytes()
            == _save_as_seed_dataset()["data"].values.tobytes()
        )
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize("active_source", [False, True])
@pytest.mark.parametrize("copy_result", ["decline", "capacity"])
def test_workspace_save_as_fallback_still_rechecks_source_ownership(
    monkeypatch, tmp_path, save_as_seed_source, active_source, copy_result
):
    source = save_as_seed_source
    target = tmp_path / "changed-source-fallback.itws"
    before = source.stat()
    original_bytes = source.read_bytes()

    def change_source_then_decline(_source, _target, *, allow_byte_copy):
        os.utime(
            source,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
        )
        if copy_result == "capacity":
            raise OSError(errno.ENOSPC, "Insufficient destination capacity")
        return False

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy,
        "copy_workspace_file",
        change_source_then_decline,
    )
    source_context = (
        workspace_store.WorkspaceStore(source)
        if active_source
        else contextlib.nullcontext()
    )
    with (
        source_context,
        pytest.raises(workspace_storage._WorkspacePublicationConflictError),
    ):
        workspace_storage._seed_workspace_save_as(target, _save_as_seed_plan(source))
    assert not target.exists()
    assert source.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "unsupported", ["external_link", "soft_link", "virtual", "external_storage"]
)
def test_workspace_reuse_estimate_rejects_indirect_payload_storage(
    tmp_path, unsupported
):
    source = tmp_path / "indirect-storage.h5"
    external = tmp_path / "external-data.h5"
    with h5py.File(external, "w") as external_h5:
        external_h5.create_dataset("data", data=[1.0, 2.0])
    with h5py.File(source, "w") as h5_file:
        group = h5_file.require_group(
            workspace_store.WorkspaceStore.object_path("reused")
        )
        h5_file.create_dataset("backing", data=[1.0, 2.0])
        if unsupported == "external_link":
            group["data"] = h5py.ExternalLink(str(external), "/data")
        elif unsupported == "soft_link":
            group["data"] = h5py.SoftLink("/backing")
        elif unsupported == "virtual":
            layout = h5py.VirtualLayout(shape=(2,), dtype=np.float64)
            layout[:] = h5py.VirtualSource(str(external), "/data", shape=(2,))
            group.create_virtual_dataset("data", layout)
        else:
            group.create_dataset(
                "data",
                data=[1.0, 2.0],
                external=[(str(tmp_path / "raw-data.bin"), 0, h5py.h5f.UNLIMITED)],
            )
        # Even a tiny file must use the conservative path for indirect storage.
        assert not workspace_storage._workspace_reused_payload_allows_byte_copy(
            h5_file, {"reused"}, 1
        )


def test_workspace_reuse_estimate_caps_hardlink_traversal(tmp_path):
    with h5py.File(tmp_path / "many-links.h5", "w") as h5_file:
        group = h5_file.require_group(
            workspace_store.WorkspaceStore.object_path("reused")
        )
        data = group.create_dataset("data", data=[1.0])
        for index in range(4096):
            group[f"link-{index}"] = data
        assert not workspace_storage._workspace_reused_payload_allows_byte_copy(
            h5_file, {"reused"}, 1
        )


def test_workspace_reuse_estimate_reads_only_storage_metadata(monkeypatch, tmp_path):
    with h5py.File(tmp_path / "metadata-only.h5", "w") as h5_file:
        group = h5_file.require_group(
            workspace_store.WorkspaceStore.object_path("reused")
        )
        data = group.create_dataset("data", data=np.arange(1024.0))
        stored_size = data.id.get_storage_size()

        def reject_data_read(*_args, **_kwargs):
            pytest.fail("The reuse estimate read dataset values")

        monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_data_read)
        assert workspace_storage._workspace_reused_payload_allows_byte_copy(
            h5_file, {"reused"}, stored_size
        )


def test_workspace_save_as_cleanup_failure_remains_fatal(
    monkeypatch, tmp_path, save_as_seed_source
):
    source = save_as_seed_source
    target = tmp_path / "cannot-remove-partial.itws"
    original_unlink = pathlib.Path.unlink
    original_file = h5py.File
    target_opens = []

    def fail_copy(_source, destination, *, allow_byte_copy):
        pathlib.Path(destination).write_bytes(b"partial workspace")
        raise OSError(errno.EIO, "Copy operation failed")

    def fail_target_unlink(path, *args, **kwargs):
        if path == target:
            raise PermissionError(errno.EPERM, "Partial target cleanup blocked")
        return original_unlink(path, *args, **kwargs)

    def open_file(path, *args, **kwargs):
        if pathlib.Path(path) == target:
            target_opens.append(path)
            pytest.fail("An incomplete target was opened after cleanup failed")
        return original_file(path, *args, **kwargs)

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "copy_workspace_file", fail_copy
    )
    monkeypatch.setattr(pathlib.Path, "unlink", fail_target_unlink)
    monkeypatch.setattr(h5py, "File", open_file)
    with pytest.raises(PermissionError, match="cleanup blocked") as exc_info:
        workspace_storage._seed_workspace_save_as(target, _save_as_seed_plan(source))
    assert exc_info.value.errno == errno.EPERM
    assert target.read_bytes() == b"partial workspace"
    assert not target_opens


@pytest.mark.parametrize("copy_error", [ValueError, KeyboardInterrupt])
def test_workspace_save_as_non_oserror_copy_failures_remain_fatal(
    monkeypatch, tmp_path, save_as_seed_source, copy_error
):
    source = save_as_seed_source
    target = tmp_path / "cancelled-copy.itws"

    def fail_native_copy(_source, destination, *, allow_byte_copy):
        pathlib.Path(destination).write_bytes(b"partial workspace")
        raise copy_error("Copy stopped")

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "_copy_native", fail_native_copy
    )
    with pytest.raises(copy_error, match="Copy stopped"):
        workspace_storage._seed_workspace_save_as(target, _save_as_seed_plan(source))
    assert not target.exists()


def test_workspace_save_as_unreadable_path_uses_retained_source_handle(
    monkeypatch, tmp_path, save_as_seed_source
):
    if os.name == "nt":
        pytest.skip("POSIX read permission bits are required")
    source = save_as_seed_source
    target = tmp_path / "retained-source.itws"
    mode = source.stat().st_mode
    native_attempts = []

    def open_source_path(src, _destination, *, allow_byte_copy):
        native_attempts.append(src)
        with pathlib.Path(src).open("rb"):
            return False

    monkeypatch.setattr(
        workspace_storage.workspace_file_copy, "_copy_native", open_source_path
    )
    monkeypatch.setattr(workspace_store.time, "sleep", lambda _delay: None)
    with workspace_store.WorkspaceStore(source) as source_store:
        source_store.current_generation()
        source.chmod(0)
        try:
            try:
                with source.open("rb"):
                    pytest.skip("The current account bypasses POSIX read permissions")
            except PermissionError:
                pass
            plan = _save_as_seed_plan(source)
            assert not workspace_storage._seed_workspace_save_as(target, plan)
            assert native_attempts
            assert not target.exists()
            with workspace_store.WorkspaceStore(target, create=True) as target_store:
                workspace_storage._write_workspace_generation(
                    target_store, plan, compression_mode="none"
                )
                assert (
                    target_store.h5_file[target_store.object_path("kept")]["data"][
                        ()
                    ].tobytes()
                    == _save_as_seed_dataset()["data"].values.tobytes()
                )
        finally:
            source.chmod(mode)
