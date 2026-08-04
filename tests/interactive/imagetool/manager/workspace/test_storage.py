import errno
import json
import pathlib
import types
import typing

import h5py
import pytest
from qtpy import QtWidgets

import erlab
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
from tests.interactive.imagetool.manager.workspace._support import (
    _assert_no_workspace_internal_groups,
    _transaction_test_dataset,
    _transaction_test_root_attrs,
    _write_transaction_test_workspace,
)

if typing.TYPE_CHECKING:
    import xarray as xr


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
    txn_path = f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
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

    monkeypatch.setattr(
        workspace_storage,
        "_is_retryable_windows_workspace_replace_error",
        lambda _exc: True,
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _replace_with_transient_denial)
    monkeypatch.setattr(workspace_storage.time, "sleep", delays.append)
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

    monkeypatch.setattr(
        workspace_storage,
        "_is_retryable_windows_workspace_replace_error",
        lambda _exc: True,
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _replace_after_external_change)
    monkeypatch.setattr(workspace_storage.time, "sleep", lambda _delay: None)

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
        workspace_storage,
        "_WINDOWS_WORKSPACE_REPLACE_RETRY_DELAYS",
        (0.0, 0.0),
    )
    monkeypatch.setattr(
        workspace_storage,
        "_is_retryable_windows_workspace_replace_error",
        lambda _exc: True,
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _deny_replace)
    monkeypatch.setattr(workspace_storage.time, "sleep", lambda _delay: None)
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


def test_windows_workspace_replace_retry_filter_is_specific(monkeypatch) -> None:
    access_denied = PermissionError(errno.EACCES, "access denied")
    wrong_error = PermissionError(errno.ENOENT, "missing")

    assert not workspace_storage._is_retryable_windows_workspace_replace_error(
        access_denied
    )
    with monkeypatch.context() as patch:
        patch.setattr(workspace_storage.os, "name", "nt")
        assert workspace_storage._is_retryable_windows_workspace_replace_error(
            access_denied
        )
        assert not workspace_storage._is_retryable_windows_workspace_replace_error(
            wrong_error
        )


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
        h5_file.create_group(
            f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}orphan"
        )
        h5_file.create_group(f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}orphan")

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
    access = manager_widgets._WorkspaceDocumentAccess(tmp_path / "workspace.itws", lock)

    assert access.take_lock() is lock
    access.release()
    assert lock.unlock_count == 0

    access = manager_widgets._WorkspaceDocumentAccess(tmp_path / "workspace.itws", lock)
    access.release()
    access.release()
    assert lock.unlock_count == 1


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
            f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
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

    fname = tmp_path / "plain.h5"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_group(f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}x")

    workspace_storage._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}x" in h5_file
