from __future__ import annotations

import errno
import gc
import hashlib
import json
import shutil
import threading
import typing

import h5py
import numpy as np
import pytest
import xarray as xr

from erlab.interactive.imagetool.manager._workspace import _arrays as workspace_arrays
from erlab.interactive.imagetool.manager._workspace import _storage as workspace_storage
from erlab.interactive.imagetool.manager._workspace import _store as workspace_store

if typing.TYPE_CHECKING:
    import pathlib


def _manifest(*object_ids: str) -> dict[str, object]:
    return {
        "schema_version": 6,
        "nodes": [
            {
                "uid": f"node-{index}",
                "kind": "imagetool",
                "path": str(index),
                "payload_object_id": object_id,
                "payload_path": workspace_store.WorkspaceStore.object_path(object_id),
            }
            for index, object_id in enumerate(object_ids)
        ],
        "root_order": list(range(len(object_ids))),
    }


def test_workspace_file_access_retry_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    waits = 0
    failed_attempts = 0

    def _temporarily_blocked() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise BlockingIOError(errno.EAGAIN, "busy")
        return "opened"

    def _waited() -> None:
        nonlocal waits
        waits += 1

    def _failed() -> None:
        nonlocal failed_attempts
        failed_attempts += 1

    monkeypatch.setattr(workspace_store.time, "sleep", lambda _delay: None)
    assert (
        workspace_store._wait_for_hdf5_access(
            _temporarily_blocked,
            on_wait=_waited,
            before_attempt=_waited,
            after_failed_attempt=_failed,
        )
        == "opened"
    )
    assert attempts == 2
    assert waits == 3
    assert failed_attempts == 1

    attempts = 0
    waits = 0
    assert (
        workspace_store._retry_file_access(_temporarily_blocked, on_wait=_waited)
        == "opened"
    )
    assert attempts == 2
    assert waits == 1

    permission_error = PermissionError(errno.EACCES, "denied")
    monkeypatch.setattr(workspace_store, "_FILE_ACCESS_RETRY_DELAYS", ())
    with pytest.raises(PermissionError, match="denied"):
        workspace_store._retry_file_access(
            lambda: (_ for _ in ()).throw(permission_error)
        )

    with pytest.raises(OSError, match="invalid"):
        workspace_store._wait_for_hdf5_access(
            lambda: (_ for _ in ()).throw(OSError(errno.EINVAL, "invalid"))
        )


def test_workspace_store_manifest_validation_contract() -> None:
    class _Dataset:
        def __init__(self, raw: object, checksum: object) -> None:
            self._raw = raw
            self.attrs = {"sha256": checksum}

        def asstr(self):
            return self

        def __getitem__(self, _key):
            return self._raw

    class _Group(dict):
        def __init__(
            self, dataset: _Dataset | None, objects: set[str] | None = None
        ) -> None:
            super().__init__()
            if dataset is not None:
                self[workspace_store._WORKSPACE_MANIFEST_DATASET] = dataset
            self.file = set() if objects is None else objects

    def _group(manifest: object, *, objects: set[str] | None = None) -> _Group:
        raw = json.dumps(manifest)
        checksum = hashlib.sha256(raw.encode()).hexdigest().encode()
        return _Group(_Dataset(raw, checksum), objects)

    with pytest.raises(ValueError, match="no manifest"):
        workspace_store.WorkspaceStore._read_manifest(_Group(None))
    with pytest.raises(TypeError, match="not text"):
        workspace_store.WorkspaceStore._read_manifest(_Group(_Dataset(1, "unused")))
    with pytest.raises(ValueError, match="checksum"):
        workspace_store.WorkspaceStore._read_manifest(
            _Group(_Dataset(json.dumps(_manifest()), "wrong"))
        )
    with pytest.raises(TypeError, match="not an object"):
        workspace_store.WorkspaceStore._read_manifest(_group([]))
    with pytest.raises(ValueError, match="unsupported schema"):
        workspace_store.WorkspaceStore._read_manifest(
            _group({"schema_version": 4, "nodes": []})
        )
    with pytest.raises(TypeError, match="not a list"):
        workspace_store.WorkspaceStore._read_manifest(
            _group({"schema_version": 5, "nodes": {}})
        )
    with pytest.raises(TypeError, match="not an object"):
        workspace_store.WorkspaceStore._read_manifest(
            _group({"schema_version": 5, "nodes": [None]})
        )
    with pytest.raises(TypeError, match="no payload object"):
        workspace_store.WorkspaceStore._read_manifest(
            _group({"schema_version": 5, "nodes": [{}]})
        )
    with pytest.raises(ValueError, match="not canonical"):
        workspace_store.WorkspaceStore._read_manifest(
            _group(
                {
                    "schema_version": 5,
                    "nodes": [{"payload_object_id": "data", "payload_path": "/bad"}],
                }
            )
        )
    with pytest.raises(ValueError, match="object is missing"):
        workspace_store.WorkspaceStore._read_manifest(_group(_manifest("data")))


def test_workspace_store_value_validation_and_closed_state(
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    with pytest.raises(ValueError, match="only when create is true"):
        workspace_store.WorkspaceStore(path, workspace_id="invalid")
    for object_id in ("", ".", "..", "nested/object", "\x00"):
        with pytest.raises(ValueError, match="one path component"):
            workspace_store.WorkspaceStore.object_path(object_id)

    assert workspace_store.WorkspaceStore.manifest_object_ids({"nodes": {}}) == set()
    assert workspace_store.WorkspaceStore.manifest_object_ids(
        {
            "nodes": [
                None,
                {},
                {"payload_object_id": ""},
                {"payload_object_id": "\x00"},
                {"payload_object_id": "ok"},
            ]
        }
    ) == {"ok"}
    assert workspace_store.WorkspaceStore.manifest_extension_object_ids(
        {
            "embedded_extension_sources": [
                {"object_id": ""},
                {"object_id": "."},
                {"object_id": ".."},
                {"object_id": "nested/object"},
                {"object_id": "\x00"},
                {
                    "script_name": "valid.py",
                    "source_hash": "a" * 64,
                    "object_id": f"extension-source-{'a' * 64}",
                },
                {"future": {"sources": [{"object_id": "extension-nested"}]}},
            ]
        }
    ) == {f"extension-source-{'a' * 64}"}

    store = workspace_store.WorkspaceStore(path, create=True)
    store.close()
    assert workspace_store.WorkspaceStore.active(path) is None
    with pytest.raises(RuntimeError, match="closed"):
        _ = store.h5_file
    with pytest.raises(RuntimeError, match="no readable handle"):
        _ = store.read_h5_file
    with pytest.raises(RuntimeError, match="closed"), store.read_session():
        pass
    with pytest.raises(RuntimeError, match="closed"):
        store.require_current_path()
    with pytest.raises(RuntimeError, match="closed"):
        store.replace_from(path, lambda _source, _target: None)


def test_workspace_store_rebinds_only_existing_legacy_payloads(tmp_path) -> None:
    path = tmp_path / "workspace.itws"

    class _Reader:
        def __init__(self, legacy_group_path: str | None) -> None:
            self.legacy_group_path = legacy_group_path
            self.object_ids: list[str] = []

        def _rebind_legacy_group_to_object(self, object_id: str) -> None:
            self.object_ids.append(object_id)

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("payload")
        matching = _Reader("/legacy/tool")
        unrelated = _Reader("/other/tool")
        store.register_reader(matching)
        store.register_reader(unrelated)

        store.rebind_legacy_readers({"/legacy/tool": "payload"})
        assert matching.object_ids == ["payload"]
        assert unrelated.object_ids == []

        with pytest.raises(KeyError, match="missing"):
            store.rebind_legacy_readers({"/legacy/tool": "missing"})
        assert matching.object_ids == ["payload"]


def test_workspace_compaction_allows_missing_optional_extension_source(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    source_hash = "a" * 64
    manifest = {
        "schema_version": 6,
        "nodes": [],
        "embedded_extension_sources": [
            {
                "script_name": "missing.py",
                "source_hash": source_hash,
                "object_id": f"extension-source-{source_hash}",
            }
        ],
    }
    with workspace_store.WorkspaceStore(path, create=True) as store:
        store.publish(manifest)

        workspace_storage._compact_workspace_store(store)

        assert store.current_generation().manifest == manifest


def test_workspace_store_identity_helpers_decode_bytes() -> None:
    class _File:
        def __init__(self, value: object) -> None:
            self.attrs = {workspace_store._WORKSPACE_ID_ATTR: value}
            self.flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1

    existing = _File(b"workspace-id")
    assert (
        workspace_store.WorkspaceStore._ensure_workspace_id(existing) == "workspace-id"
    )
    assert workspace_store.WorkspaceStore._workspace_id_from_file(existing) == (
        "workspace-id"
    )

    missing = _File(None)
    generated = workspace_store.WorkspaceStore._ensure_workspace_id(missing)
    assert generated
    assert missing.attrs[workspace_store._WORKSPACE_ID_ATTR] == generated
    assert missing.flush_count == 1
    assert workspace_store.WorkspaceStore._workspace_id_from_file(_File("")) is None
    assert (
        workspace_store.WorkspaceStore._workspace_id_from_file(_File(b"bytes-id"))
        == "bytes-id"
    )


def test_workspace_store_defers_missing_identity_until_publish(tmp_path) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True):
        pass
    with h5py.File(path, "a") as h5_file:
        del h5_file.attrs[workspace_store._WORKSPACE_ID_ATTR]
        h5_file.attrs["imagetool_workspace_schema_version"] = 4

    unchanged = path.read_bytes()
    with workspace_store.WorkspaceStore(path) as store:
        assert store.workspace_id is None
        workspace_store.WorkspaceStore.pin_serialized_reader(
            workspace_id=None,
            path=path,
            object_id="old-object",
            legacy_group_path="/legacy",
        )
        old_pins = store.serialized_reader_pin_snapshot()
        assert store.serialized_object_ids == {"old-object"}
        assert store.serialized_legacy_group_paths == {"/legacy"}
    assert path.read_bytes() == unchanged
    with h5py.File(path, "r") as h5_file:
        assert workspace_store._WORKSPACE_ID_ATTR not in h5_file.attrs

    with workspace_store.WorkspaceStore(path) as store:
        store.publish(_manifest())
        workspace_id = store.workspace_id
        assert store.serialized_object_ids == {"old-object"}
        assert store.serialized_legacy_group_paths == {"/legacy"}
        assert store.has_serialized_readers
        workspace_store.WorkspaceStore.pin_serialized_reader(
            workspace_id=workspace_id,
            path=path,
            object_id="kept-object",
            legacy_group_path="/kept-legacy",
        )
        store.release_serialized_reader_pins(old_pins)
        assert store.serialized_object_ids == {"kept-object"}
        assert store.serialized_legacy_group_paths == {"/kept-legacy"}
        store.clear_serialized_reader_pins()
        assert not store.has_serialized_readers
        assert store.serialized_object_ids == set()
        assert store.serialized_legacy_group_paths == set()

    assert workspace_id
    with h5py.File(path, "r") as h5_file:
        assert h5_file.attrs[workspace_store._WORKSPACE_ID_ATTR] == workspace_id


def test_hdf5_error_filters_cover_platform_independent_messages() -> None:
    sharing_error = OSError("sharing violation")
    sharing_error.winerror = 32
    assert workspace_store._is_hdf5_file_contention_error(sharing_error)
    assert workspace_store._is_hdf5_file_lock_unavailable_error(
        OSError("HDF5 locking is disabled")
    )
    assert workspace_store._is_hdf5_file_lock_unavailable_error(
        OSError(errno.ENOSYS, "unsupported")
    )


def test_hdf5_access_does_not_retry_when_locking_is_unavailable() -> None:
    attempts = 0
    waits = 0

    def _open() -> None:
        nonlocal attempts
        attempts += 1
        raise OSError(errno.ENOSYS, "unable to lock file: function not implemented")

    def _wait() -> None:
        nonlocal waits
        waits += 1

    with pytest.raises(OSError, match="unable to lock file"):
        workspace_store._wait_for_hdf5_access(_open, on_wait=_wait)

    assert attempts == 1
    assert waits == 0


def test_workspace_store_rejects_non_lock_open_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        monkeypatch.setattr(
            workspace_store.h5py,
            "File",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(errno.EINVAL, "invalid HDF5 open")
            ),
        )
        with pytest.raises(OSError, match="invalid HDF5 open"):
            store._open_with_lock_detection("r")


def test_workspace_store_switches_when_required_locking_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_path = tmp_path / "source.itws"
    target_path = tmp_path / "target.itws"
    with workspace_store.WorkspaceStore(target_path, create=True):
        pass
    with workspace_store.WorkspaceStore(source_path, create=True) as source:
        monkeypatch.setattr(
            workspace_store,
            "_wait_for_hdf5_access",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError(errno.ENOSYS, "locking is not supported")
            ),
        )

        source.switch_path(target_path)

        assert source.path == target_path.resolve()
        assert not source.locking_supported


def test_workspace_store_conflicted_guards(tmp_path) -> None:
    path = tmp_path / "workspace.itws"
    prepared = tmp_path / "prepared.itws"
    with workspace_store.WorkspaceStore(prepared, create=True):
        pass
    with workspace_store.WorkspaceStore(path, create=True) as store:
        store._mark_conflicted()
        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            store.require_current_path()
        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            store.reopen()
        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            store.replace_from(prepared, lambda _source, _target: None)


def test_workspace_store_missing_path_and_recovery_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        identity = typing.cast("tuple[int, int]", store._path_identity)
        path.unlink()
        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            store._require_path_identity(identity)
        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            store._require_path_state((*identity, 0, 0))

        monkeypatch.setattr(
            workspace_store.h5py,
            "File",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")),
        )
        with pytest.raises(OSError, match="unreadable"):
            store._use_recovery_source(tmp_path / "missing-recovery.itws")
        assert store.conflicted


def test_workspace_store_flush_accepts_tuple_vfd_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _Id:
        @staticmethod
        def get_vfd_handle():
            return (123, object())

    class _File:
        id = _Id()

        @staticmethod
        def flush() -> None:
            pass

    synced: list[int] = []
    store = workspace_store.WorkspaceStore(tmp_path / "workspace.itws", create=True)
    store._release_handle()
    store._h5_file = _File()
    monkeypatch.setattr(workspace_store.os, "fsync", synced.append)
    try:
        store.flush(durable=True)
    finally:
        store._h5_file = None
        store.close()
    assert synced == [123]


def test_workspace_store_rejects_duplicate_owner_and_invalid_switch(
    tmp_path,
) -> None:
    import h5py

    source_path = tmp_path / "source.itws"
    target_path = tmp_path / "target.itws"
    invalid_path = tmp_path / "invalid.itws"
    with workspace_store.WorkspaceStore(source_path, create=True) as source:
        with pytest.raises(RuntimeError, match="active store"):
            workspace_store.WorkspaceStore(source_path)

        with h5py.File(invalid_path, "w"):
            pass
        with pytest.raises(ValueError, match="no stable identity"):
            source.switch_path(invalid_path)

        with (
            workspace_store.WorkspaceStore(target_path, create=True),
            pytest.raises(RuntimeError, match="active store"),
        ):
            source.switch_path(target_path)


def test_workspace_store_write_open_failure_restores_idle_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    prepared = tmp_path / ".workspace.itws.write-test"

    with workspace_store.WorkspaceStore(path, create=True) as store:
        store._locking_supported = False

        def _prepare() -> pathlib.Path:
            prepared.write_bytes(b"not-hdf5")
            return prepared

        monkeypatch.setattr(store, "_prepare_copy_on_write", _prepare)
        with pytest.raises(OSError, match="file signature"), store.write_session():
            pass

        assert not store.write_in_progress
        assert store._write_target_path is None
        assert not prepared.exists()

        store._path_identity = None
        with (
            pytest.raises(RuntimeError, match="no path identity"),
            store.write_session(),
        ):
            pass
        assert not store.write_in_progress


def test_workspace_store_replace_conflict_callbacks_quarantine_writes(
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    prepared = tmp_path / "prepared.itws"
    with workspace_store.WorkspaceStore(prepared, create=True):
        pass
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            store.replace_from(
                prepared,
                lambda _source, _target: None,
                before_close=lambda: (_ for _ in ()).throw(
                    workspace_store.WorkspaceStoreConflictError("changed")
                ),
            )
        assert store.conflicted


def test_workspace_store_replace_detects_path_change_after_failure(
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    prepared = tmp_path / "prepared.itws"
    with workspace_store.WorkspaceStore(prepared, create=True):
        pass
    with workspace_store.WorkspaceStore(path, create=True) as store:

        def _remove_then_fail(_source, target) -> None:
            target.unlink()
            raise RuntimeError("replace failed")

        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            store.replace_from(prepared, _remove_then_fail)
        assert store.conflicted


def test_workspace_store_replace_failure_keeps_recovery_error_quarantined(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    prepared = tmp_path / "prepared.itws"
    with workspace_store.WorkspaceStore(prepared, create=True):
        pass
    with workspace_store.WorkspaceStore(path, create=True) as store:
        monkeypatch.setattr(
            store,
            "_reopen_after_file_operation",
            lambda: (_ for _ in ()).throw(OSError("cannot reopen")),
        )
        monkeypatch.setattr(
            store,
            "_use_recovery_source",
            lambda _path: (_ for _ in ()).throw(OSError("cannot recover")),
        )

        with pytest.raises(RuntimeError, match="replace failed"):
            store.replace_from(
                prepared,
                lambda _source, _target: (_ for _ in ()).throw(
                    RuntimeError("replace failed")
                ),
            )
        assert store.conflicted


def test_workspace_store_reports_reopen_failure_after_successful_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    prepared = tmp_path / "prepared.itws"
    with workspace_store.WorkspaceStore(prepared, create=True):
        pass
    with workspace_store.WorkspaceStore(path, create=True) as store:
        monkeypatch.setattr(
            store,
            "_reopen_after_file_operation",
            lambda: (_ for _ in ()).throw(OSError("cannot reopen")),
        )

        with pytest.raises(workspace_store.WorkspaceStoreReopenError):
            store.replace_from(prepared, lambda source, target: source.replace(target))


def test_workspace_store_rejects_invalid_gc_limit_and_clears_staging(
    tmp_path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with pytest.raises(ValueError, match="must be positive"):
            store.collect_garbage(max_objects=0)

        store.clear_staging()
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_STAGING_GROUP].create_group("orphan")
        store.clear_staging()
        assert len(store.h5_file[workspace_store._WORKSPACE_STAGING_GROUP]) == 0


def test_workspace_store_publishes_valid_generations(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            objects = h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]
            objects.create_group("first")
            objects.create_group("second")
        first = store.publish(_manifest("first"))
        second = store.publish(_manifest("second"))

        assert first.sequence == 1
        assert second.sequence == 2
        assert store.current_generation() == second

        with store.write_session() as h5_file:
            generation_group = h5_file[
                f"{workspace_store._WORKSPACE_GENERATIONS_GROUP}/{second.sequence:020d}"
            ]
            generation_group[workspace_store._WORKSPACE_MANIFEST_DATASET].attrs[
                "sha256"
            ] = "invalid"

        assert store.current_generation() == first


def test_workspace_store_rejects_generation_with_missing_object(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with pytest.raises(ValueError, match="payload object is missing"):
            store.publish(_manifest("missing"))

        assert store.generations() == ()
        assert len(store.h5_file[workspace_store._WORKSPACE_STAGING_GROUP]) == 0


def test_workspace_store_limits_writable_handle_to_write_session(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        assert store.h5_file.mode == "r"
        with store.write_session() as h5_file:
            assert h5_file.mode == "r+"
        assert store._h5_file is None
        assert store.h5_file.mode == "r"


def test_workspace_store_uses_copy_on_write_without_filesystem_locks(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"

    with workspace_store.WorkspaceStore(path, create=True) as store:
        store._locking_supported = False
        with store.write_session() as h5_file:
            assert h5_file.filename != str(path)
            h5_file.attrs["copy_on_write"] = True

        assert store.h5_file.attrs["copy_on_write"]
        assert set(tmp_path.iterdir()) == {path}


def test_workspace_store_discards_failed_copy_on_write(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"

    with workspace_store.WorkspaceStore(path, create=True) as store:
        store._locking_supported = False

        def _fail_write() -> None:
            with store.write_session() as h5_file:
                h5_file.attrs["must_not_publish"] = True
                raise RuntimeError("injected failure")

        with pytest.raises(RuntimeError, match="injected failure"):
            _fail_write()

        assert "must_not_publish" not in store.h5_file.attrs
        assert set(tmp_path.iterdir()) == {path}


def test_workspace_store_rejects_changed_copy_on_write_source(
    tmp_path: pathlib.Path,
) -> None:
    import h5py

    path = tmp_path / "workspace.itws"

    with workspace_store.WorkspaceStore(path, create=True) as store:
        store._locking_supported = False

        def _write_while_source_changes() -> None:
            with store.write_session() as staged:
                staged.attrs["staged"] = True
                with h5py.File(path, "r+", locking=False) as external:
                    external.attrs["external"] = True
                    external.create_dataset("external_payload", data=np.arange(1000))
                    external.flush()

        with pytest.raises(
            workspace_store.WorkspaceStoreConflictError,
            match="changed during save",
        ):
            _write_while_source_changes()

        assert store.conflicted
        assert store.recovery_path is not None
        assert store.read_h5_file.attrs["staged"]
        with h5py.File(path, "r", locking=False) as external:
            assert "staged" not in external.attrs
            assert external.attrs["external"]
        with (
            pytest.raises(
                workspace_store.WorkspaceStoreConflictError,
                match="no longer owns its path",
            ),
            store.write_session(),
        ):
            pass


def test_workspace_store_quarantines_failed_copy_on_write_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    import h5py

    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        store._locking_supported = False
        monkeypatch.setattr(
            store,
            "_use_recovery_source",
            lambda _path: (_ for _ in ()).throw(RuntimeError("cannot reopen")),
        )

        def _write_while_source_changes() -> None:
            with (
                store.write_session(),
                h5py.File(path, "r+", locking=False) as external,
            ):
                external.attrs["external"] = True
                external.flush()

        with pytest.raises(
            workspace_store.WorkspaceStoreConflictError,
            match="changed during save",
        ):
            _write_while_source_changes()

        assert store.conflicted
        assert store.recovery_path is None
        assert set(tmp_path.iterdir()) == {path}


def test_workspace_store_fast_path_does_not_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"

    with workspace_store.WorkspaceStore(path, create=True) as store:

        def _unexpected_copy() -> pathlib.Path:
            raise AssertionError("locking-supported writes must not copy the workspace")

        monkeypatch.setattr(store, "_prepare_copy_on_write", _unexpected_copy)
        with store.write_session() as h5_file:
            h5_file.attrs["direct"] = True

        assert store.h5_file.attrs["direct"]


def test_workspace_store_detects_unavailable_required_locking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    original_file = workspace_store.h5py.File

    def _open_file(*args, **kwargs):
        if kwargs.get("locking") is True:
            raise OSError(errno.ENOSYS, "unable to lock file: not supported")
        return original_file(*args, **kwargs)

    monkeypatch.setattr(workspace_store.h5py, "File", _open_file)

    with workspace_store.WorkspaceStore(path, create=True) as store:
        assert not store.locking_supported
        with store.write_session() as h5_file:
            assert h5_file.filename != str(path)
            h5_file.attrs["safe_fallback"] = True

        assert store.h5_file.attrs["safe_fallback"]


def test_hdf5_contention_filter_does_not_treat_permissions_as_overlap() -> None:
    permission_error = PermissionError(errno.EACCES, "access denied")
    lock_error = BlockingIOError(errno.EAGAIN, "resource temporarily unavailable")

    assert not workspace_store._is_hdf5_file_contention_error(permission_error)
    assert workspace_store._is_hdf5_file_contention_error(lock_error)


def test_workspace_store_gc_retains_two_generations_and_leases(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            objects = h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]
            for object_id in ("first", "second", "third"):
                objects.create_group(object_id)
        store.publish(_manifest("first"))
        store.publish(_manifest("second"))
        store.acquire_object("first")
        store.publish(_manifest("third"))

        assert not store.collect_garbage(max_objects=10)
        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "first",
            "second",
            "third",
        }
        assert len(store.generations()) == 2

        store.release_object("first")
        assert not store.collect_garbage(max_objects=10)
        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "second",
            "third",
        }


def test_workspace_store_gc_preserves_serialized_reader_objects(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            objects = h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]
            for object_id in ("first", "second", "third"):
                objects.create_group(object_id)
        first_generation = store.publish(_manifest("first"))
        workspace_store.WorkspaceStore.pin_serialized_reader(
            workspace_id=None,
            path=path,
            object_id="first",
            legacy_group_path=None,
        )
        store.publish(_manifest("second"))
        store.publish(_manifest("third"))

        assert not store.collect_garbage(max_objects=1)
        assert len(store.generations()) == 2
        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "first",
            "second",
            "third",
        }
        assert store.serialized_object_ids == {"first"}

    with workspace_store.WorkspaceStore(path) as reopened:
        assert reopened.serialized_object_ids == {"first"}
        assert reopened.has_serialized_readers
        assert reopened.current_generation().sequence > first_generation.sequence
        workspace_id = reopened.workspace_id
        reopened.clear_serialized_reader_pins()
        assert not reopened.has_serialized_readers
        assert not reopened.collect_garbage(max_objects=10)
        assert set(reopened.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "second",
            "third",
        }

    copied_path = tmp_path / "copied-workspace.itws"
    shutil.copy2(path, copied_path)
    with workspace_store.WorkspaceStore(copied_path) as copied:
        assert copied.workspace_id == workspace_id
        assert copied.serialized_object_ids == set()


def test_workspace_reader_export_waits_for_compaction_boundary(tmp_path) -> None:
    path = tmp_path / "workspace.itws"
    export_started = threading.Event()
    export_finished = threading.Event()
    export_errors: list[Exception] = []

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("obsolete")

        def _export_reader() -> None:
            export_started.set()
            try:
                store.pin_serialized_reader_reference(
                    object_id="obsolete",
                    legacy_group_path=None,
                )
            except Exception as exc:
                export_errors.append(exc)
            finally:
                export_finished.set()

        with store.lock:
            export_thread = threading.Thread(target=_export_reader)
            export_thread.start()
            assert export_started.wait(timeout=2)
            assert not export_finished.wait(timeout=0.05)
            with store.write_session() as h5_file:
                del h5_file[f"{workspace_store._WORKSPACE_OBJECTS_GROUP}/obsolete"]

        export_thread.join(timeout=2)
        assert not export_thread.is_alive()
        assert len(export_errors) == 1
        assert isinstance(
            export_errors[0], workspace_store.WorkspaceReaderUnavailableError
        )
        assert not store.has_serialized_readers


def test_workspace_reader_export_rejects_unavailable_targets(tmp_path) -> None:
    path = tmp_path / "workspace.itws"
    store = workspace_store.WorkspaceStore(path, create=True)

    with pytest.raises(
        workspace_store.WorkspaceReaderUnavailableError,
        match="payload was compacted",
    ):
        store.pin_serialized_reader_reference(
            object_id=None,
            legacy_group_path="/missing-legacy",
        )

    store.close()
    with pytest.raises(
        workspace_store.WorkspaceReaderUnavailableError,
        match="workspace changed",
    ):
        store.pin_serialized_reader_reference(
            object_id=None,
            legacy_group_path=None,
        )


def test_workspace_store_gc_removes_malformed_generation_names(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("current")
        store.publish(_manifest("current"))
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_GENERATIONS_GROUP].create_group("1")

        assert not store.collect_garbage(max_objects=1)

        assert set(store.h5_file[workspace_store._WORKSPACE_GENERATIONS_GROUP]) == {
            "00000000000000000001"
        }


def test_workspace_generation_write_blocks_concurrent_gc(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    object_written = threading.Event()
    release_write = threading.Event()
    gc_finished = threading.Event()
    errors: list[BaseException] = []

    with workspace_store.WorkspaceStore(path, create=True) as store:
        plan = workspace_storage._WorkspaceGenerationPlan(
            manifest=_manifest("pending"),
            objects=(
                workspace_storage._WorkspaceObjectWrite(
                    "pending", dataset=xr.Dataset()
                ),
            ),
        )

        monkeypatch.setattr(
            workspace_arrays, "_workspace_dataset_can_write_h5py", lambda _ds: False
        )

        def _delayed_write(h5_file, group_path, _ds, **_kwargs) -> None:
            h5_file.require_group(group_path)
            object_written.set()
            if not release_write.wait(2):
                raise TimeoutError("generation write was not released")

        monkeypatch.setattr(
            workspace_arrays,
            "_write_workspace_dataset_group_to_file",
            _delayed_write,
        )

        def _save() -> None:
            try:
                workspace_storage._write_workspace_generation(
                    store, plan, compression_mode="none"
                )
            except BaseException as exc:
                errors.append(exc)

        def _collect() -> None:
            try:
                store.collect_garbage(max_objects=1)
            except BaseException as exc:
                errors.append(exc)
            finally:
                gc_finished.set()

        save_thread = threading.Thread(target=_save)
        gc_thread = threading.Thread(target=_collect)
        save_thread.start()
        assert object_written.wait(2)
        gc_thread.start()
        try:
            assert not gc_finished.wait(0.05)
        finally:
            release_write.set()
        save_thread.join(2)
        gc_thread.join(2)

        assert not save_thread.is_alive()
        assert not gc_thread.is_alive()
        assert errors == []
        assert store.current_generation().manifest == _manifest("pending")
        assert "pending" in store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]


def test_workspace_generation_removes_partial_object_after_write_error(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest("partial"),
        objects=(
            workspace_storage._WorkspaceObjectWrite(
                "partial", dataset=xr.Dataset({"data": ("x", np.arange(3))})
            ),
        ),
    )

    def _write_partial_then_fail(h5_file, group_path, _dataset, **_kwargs) -> None:
        h5_file.require_group(group_path)
        raise RuntimeError("write failed")

    monkeypatch.setattr(
        workspace_arrays,
        "_write_workspace_dataset_group_to_file",
        _write_partial_then_fail,
    )

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with pytest.raises(RuntimeError, match="write failed"):
            workspace_storage._write_workspace_generation(
                store, plan, compression_mode="none"
            )

        assert store.object_path("partial").strip("/") not in store.h5_file
        assert store.generations() == ()


def test_workspace_generation_removes_partial_object_after_copy_error(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    source_path = tmp_path / "source.itws"
    target_path = tmp_path / "target.itws"
    with workspace_store.WorkspaceStore(source_path, create=True) as source_store:
        with source_store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("source")
        source_store.publish(_manifest("source"))

    def _copy_partial_then_fail(
        _source_file,
        target_file,
        _source_path,
        target_group_path,
        _attrs,
    ) -> bool:
        target_file.require_group(target_group_path)
        raise RuntimeError("copy failed")

    monkeypatch.setattr(
        workspace_arrays,
        "_copy_workspace_h5_group_to_open_file",
        _copy_partial_then_fail,
    )
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest("copied"),
        objects=(
            workspace_storage._WorkspaceObjectWrite(
                "copied",
                source_file=str(source_path),
                source_path=workspace_store.WorkspaceStore.object_path("source"),
            ),
        ),
    )

    with workspace_store.WorkspaceStore(target_path, create=True) as target_store:
        with pytest.raises(RuntimeError, match="copy failed"):
            workspace_storage._write_workspace_generation(
                target_store, plan, compression_mode="none"
            )

        assert target_store.object_path("copied").strip("/") not in target_store.h5_file
        assert target_store.generations() == ()


def test_workspace_generation_removes_all_new_objects_after_plan_error(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest("written"),
        objects=(
            workspace_storage._WorkspaceObjectWrite(
                "written", dataset=xr.Dataset({"data": ("x", np.arange(3))})
            ),
            workspace_storage._WorkspaceObjectWrite("missing-source"),
        ),
    )

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with pytest.raises(ValueError, match="has no source"):
            workspace_storage._write_workspace_generation(
                store, plan, compression_mode="none"
            )

        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == set()
        assert store.generations() == ()


def test_workspace_generation_removes_legacy_link_after_publish_error(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest("written"),
        objects=(
            workspace_storage._WorkspaceObjectWrite(
                "written", dataset=xr.Dataset({"data": ("x", np.arange(3))})
            ),
        ),
        preserved_groups=(
            workspace_storage._WorkspaceGroupCopy(
                source_file=str(path),
                source_path="/legacy",
                target_path="/legacy",
            ),
        ),
        legacy_reader_rebindings=(("/legacy", "written"),),
    )

    with workspace_store.WorkspaceStore(path, create=True) as store:

        def _fail_publish(_manifest) -> None:
            raise RuntimeError("publish failed")

        monkeypatch.setattr(store, "publish", _fail_publish)
        with pytest.raises(RuntimeError, match="publish failed"):
            workspace_storage._write_workspace_generation(
                store, plan, compression_mode="none"
            )

        assert "legacy" not in store.h5_file
        assert store.object_path("written").strip("/") not in store.h5_file
        assert store.generations() == ()


def test_workspace_generation_removes_in_place_object_link_after_publish_error(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest("written"),
        objects=(
            workspace_storage._WorkspaceObjectWrite(
                "written",
                source_file=str(path),
                source_path="/legacy",
            ),
        ),
        legacy_object_links=(("/legacy", "written"),),
    )

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file.create_group("legacy").create_dataset("data", data=np.arange(3))
        legacy_address = h5py.h5o.get_info(store.h5_file["/legacy"].id).addr

        def _fail_publish(_manifest) -> None:
            raise RuntimeError("publish failed")

        monkeypatch.setattr(store, "publish", _fail_publish)
        with pytest.raises(RuntimeError, match="publish failed"):
            workspace_storage._write_workspace_generation(
                store, plan, compression_mode="none"
            )

        assert "legacy" in store.h5_file
        assert h5py.h5o.get_info(store.h5_file["/legacy"].id).addr == legacy_address
        assert store.object_path("written").strip("/") not in store.h5_file
        assert store.generations() == ()


def test_workspace_generation_keeps_objects_referenced_by_committed_generation(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest("committed"),
        objects=(
            workspace_storage._WorkspaceObjectWrite(
                "committed", dataset=xr.Dataset({"data": ("x", np.arange(3))})
            ),
        ),
    )

    with workspace_store.WorkspaceStore(path, create=True) as store:
        original_publish = store.publish

        def _publish_then_fail(manifest) -> workspace_store._WorkspaceGeneration:
            original_publish(manifest)
            raise RuntimeError("post-publication failure")

        monkeypatch.setattr(store, "publish", _publish_then_fail)
        with pytest.raises(RuntimeError, match="post-publication failure"):
            workspace_storage._write_workspace_generation(
                store, plan, compression_mode="none"
            )

        assert store.current_generation().manifest == _manifest("committed")
        assert "committed" in store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]


def test_workspace_store_shares_lazy_reader_and_generation_writer(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    dataset = xr.Dataset(
        {"data": (("x", "y"), np.arange(30, dtype=np.float64).reshape(5, 6))}
    )
    object_id = "payload"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest(object_id),
        objects=(workspace_storage._WorkspaceObjectWrite(object_id, dataset=dataset),),
    )

    with workspace_store.WorkspaceStore(path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store,
            plan,
            compression_mode="none",
        )
        opened = workspace_arrays.open_workspace_dataset(
            path,
            store.object_path(object_id),
            chunks={},
        )
        try:
            assert opened["data"].chunks is not None
            assert float(opened["data"].sum().compute()) == 435.0
            derived_id = "derived"
            derived = opened + 1
            import dask

            with dask.config.set(scheduler="processes"):
                workspace_storage._write_workspace_generation(
                    store,
                    workspace_storage._WorkspaceGenerationPlan(
                        manifest=_manifest(derived_id),
                        objects=(
                            workspace_storage._WorkspaceObjectWrite(
                                derived_id,
                                dataset=derived,
                            ),
                        ),
                    ),
                    compression_mode="none",
                )
            saved = workspace_arrays.open_workspace_dataset(
                path,
                store.object_path(derived_id),
                chunks={},
            )
            try:
                assert float(saved["data"].sum().compute()) == 465.0
                workspace_storage._compact_workspace_store(store)
                assert float(opened["data"].sum().compute()) == 435.0
                assert float(saved["data"].sum().compute()) == 465.0
            finally:
                saved.close()
            assert store.h5_file.id.valid
        finally:
            opened.close()


def test_standalone_lazy_reader_attaches_when_workspace_store_opens(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    object_id = "payload"
    dataset = xr.Dataset({"data": ("x", np.arange(5, dtype=np.float64))})
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest(object_id),
        objects=(workspace_storage._WorkspaceObjectWrite(object_id, dataset=dataset),),
    )
    with workspace_store.WorkspaceStore(path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )

    opened = workspace_arrays.open_workspace_dataset(
        path,
        workspace_store.WorkspaceStore.object_path(object_id),
        chunks={},
    )
    try:
        assert float(opened["data"].sum().compute()) == 10.0
        with workspace_store.WorkspaceStore(path) as store:
            assert object_id in store.leased_object_ids
            with store.write_session() as h5_file:
                h5_file.attrs["attached_reader"] = True
            assert float(opened["data"].sum().compute()) == 10.0
    finally:
        opened.close()


def test_workspace_lazy_array_uses_bounded_process_reads(
    tmp_path: pathlib.Path,
) -> None:
    import dask

    path = tmp_path / "workspace.itws"
    dataset = xr.Dataset(
        {"data": (("x", "y"), np.arange(30, dtype=np.float64).reshape(5, 6))}
    )
    object_id = "payload"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest(object_id),
        objects=(workspace_storage._WorkspaceObjectWrite(object_id, dataset=dataset),),
    )

    with workspace_store.WorkspaceStore(path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        opened = workspace_arrays.open_workspace_dataset(
            path, store.object_path(object_id), chunks={"x": 2}
        )
        try:
            with dask.config.set(scheduler="processes"):
                result = opened["data"].sum().compute()
            assert float(result) == 435.0
            assert store.h5_file.mode == "r"
        finally:
            opened.close()


def test_workspace_lazy_array_uses_process_localcluster_without_copy(
    tmp_path: pathlib.Path,
) -> None:
    distributed = pytest.importorskip("distributed")

    path = tmp_path / "workspace.itws"
    dataset = xr.Dataset(
        {"data": (("x", "y"), np.arange(30, dtype=np.float64).reshape(5, 6))}
    )
    object_id = "payload"
    plan = workspace_storage._WorkspaceGenerationPlan(
        manifest=_manifest(object_id),
        objects=(workspace_storage._WorkspaceObjectWrite(object_id, dataset=dataset),),
    )

    with workspace_store.WorkspaceStore(path, create=True) as store:
        workspace_storage._write_workspace_generation(
            store, plan, compression_mode="none"
        )
        opened = workspace_arrays.open_workspace_dataset(
            path, store.object_path(object_id), chunks={"x": 2}
        )
        try:
            with (
                distributed.LocalCluster(
                    n_workers=2,
                    threads_per_worker=1,
                    processes=True,
                    dashboard_address=None,
                ) as cluster,
                distributed.Client(cluster, set_as_default=False) as client,
            ):
                result = client.compute(opened["data"].sum()).result()
            assert float(result) == 435.0
            assert store.serialized_object_ids == {object_id}
            assert set(tmp_path.iterdir()) == {path}
        finally:
            opened.close()


def test_workspace_writer_waits_only_for_active_hdf5_reader(
    tmp_path: pathlib.Path,
) -> None:
    import h5py

    path = tmp_path / "workspace.itws"
    write_started = threading.Event()
    write_waiting = threading.Event()
    write_finished = threading.Event()
    errors: list[BaseException] = []

    with workspace_store.WorkspaceStore(path, create=True) as store:

        def _write() -> None:
            write_started.set()
            try:
                with store.write_session(on_contention=write_waiting.set) as h5_file:
                    h5_file.attrs["overlap_test"] = True
            except BaseException as exc:
                errors.append(exc)
            finally:
                write_finished.set()

        with h5py.File(path, "r", locking="best-effort"):
            writer = threading.Thread(target=_write)
            writer.start()
            assert write_started.wait(2)
            assert write_waiting.wait(2)
            assert not write_finished.is_set()
            assert store.generations() == ()

        writer.join(2)
        assert not writer.is_alive()
        assert write_finished.is_set()
        assert errors == []
        assert store.h5_file.attrs["overlap_test"]


def test_workspace_writer_does_not_report_wait_without_overlap(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    waiting = threading.Event()

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session(on_contention=waiting.set) as h5_file:
            h5_file.attrs["direct_write"] = True

        assert not waiting.is_set()


def test_workspace_worker_reports_inaccessible_shared_path(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    group_path = workspace_store.WorkspaceStore.object_path("payload")
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file.require_group(group_path)
        manager = workspace_arrays.WorkspaceFileManager(
            path, object_id="payload", group_path=group_path
        )
        state = manager.__getstate__()
    path.unlink()
    worker_manager = workspace_arrays.WorkspaceFileManager.__new__(
        workspace_arrays.WorkspaceFileManager
    )
    worker_manager.__setstate__(state)

    with pytest.raises(
        workspace_arrays.WorkspaceWorkerAccessError,
        match="same readable file on every worker",
    ):
        worker_manager._read_bounded_variable("data", slice(None))


def test_workspace_worker_waits_for_overlapping_writer(
    tmp_path: pathlib.Path,
) -> None:
    import subprocess
    import sys

    path = tmp_path / "workspace.itws"
    group_path = workspace_store.WorkspaceStore.object_path("payload")
    dataset = xr.Dataset({"data": ("x", np.arange(4, dtype=np.float64))})

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            workspace_arrays._write_workspace_dataset_group_to_file(
                h5_file,
                group_path,
                dataset,
                compression_mode="none",
            )
        manager = workspace_arrays.WorkspaceFileManager(
            path,
            object_id="payload",
            group_path=group_path,
        )
        state = manager.__getstate__()
        code = """
import sys
from erlab.interactive.imagetool.manager._workspace import _arrays
m = _arrays.WorkspaceFileManager.__new__(_arrays.WorkspaceFileManager)
m.__setstate__((
    "erlab-workspace-bounded-reader-v1",
    sys.argv[1], sys.argv[1], sys.argv[2], "payload", sys.argv[3]
))
print("ready", flush=True)
print(float(m._read_bounded_variable("data", slice(None)).sum()), flush=True)
"""
        with store.write_session():
            reader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(path),
                    typing.cast("str", state[3]),
                    group_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if reader.stdout is None:
                raise RuntimeError("Worker stdout is unavailable")
            assert reader.stdout.readline().strip() == "ready"
            assert reader.poll() is None

        stdout, stderr = reader.communicate(timeout=5)
        assert reader.returncode == 0, stderr
        assert float(stdout.strip()) == 6.0


def test_workspace_store_closes_cached_handles_before_fork(
    tmp_path: pathlib.Path,
) -> None:
    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable")

    path = tmp_path / "workspace.itws"
    context = multiprocessing.get_context("fork")
    child_ready = context.Event()
    child_release = context.Event()

    def _wait_in_child() -> None:
        child_ready.set()
        child_release.wait(5)

    with workspace_store.WorkspaceStore(path, create=True) as store:
        _ = store.h5_file
        process = context.Process(target=_wait_in_child)
        process.start()
        try:
            assert child_ready.wait(2)
            with store.write_session() as h5_file:
                h5_file.attrs["written_after_fork"] = True
        finally:
            child_release.set()
            process.join(5)
        assert process.exitcode == 0
        assert store.h5_file.attrs["written_after_fork"]


def test_workspace_worker_rejects_different_file_at_shared_path(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    replacement = tmp_path / "replacement.itws"
    group_path = workspace_store.WorkspaceStore.object_path("payload")
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file.require_group(group_path)
        manager = workspace_arrays.WorkspaceFileManager(
            path, object_id="payload", group_path=group_path
        )
        state = manager.__getstate__()
    with workspace_store.WorkspaceStore(replacement, create=True):
        pass
    replacement.replace(path)
    worker_manager = workspace_arrays.WorkspaceFileManager.__new__(
        workspace_arrays.WorkspaceFileManager
    )
    worker_manager.__setstate__(state)

    with pytest.raises(
        workspace_store.WorkspaceStoreConflictError,
        match="identity changed",
    ):
        worker_manager._read_bounded_variable("data", slice(None))


def test_workspace_raw_reads_reuse_active_store_handle(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with (
        workspace_store.WorkspaceStore(path, create=True) as store,
        workspace_arrays._open_workspace_h5_file_for_read(path) as h5_file,
    ):
        assert h5_file is store.h5_file


def test_workspace_compaction_keeps_current_state_and_reopens_store(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            objects = h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]
            objects.create_group("obsolete").create_dataset(
                "data", data=np.ones(2_000_000, dtype=np.float64)
            )
            objects.create_group("current").create_dataset(
                "data", data=np.arange(10, dtype=np.float64)
            )
        store.publish(_manifest("obsolete"))
        store.publish(_manifest("current"))
        size_before = path.stat().st_size
        workspace_id = store.workspace_id

        workspace_storage._compact_workspace_store(store)

        assert workspace_store.WorkspaceStore.active(path) is store
        assert store.workspace_id == workspace_id
        assert store.h5_file.id.valid
        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "current"
        }
        generations = store.generations()
        assert len(generations) == 2
        assert generations[0].manifest == generations[1].manifest
        assert store.current_generation().manifest == _manifest("current")
        assert path.stat().st_size < size_before


def test_workspace_store_active_waits_for_file_replacement(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    prepared_path = tmp_path / "prepared.itws"
    with workspace_store.WorkspaceStore(prepared_path, create=True) as prepared_store:
        with prepared_store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("prepared")
        prepared_store.publish(_manifest("prepared"))

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("current")
        store.publish(_manifest("current"))
        replacement_started = threading.Event()
        allow_replacement = threading.Event()
        lookup_finished = threading.Event()
        lookup_result: list[workspace_store.WorkspaceStore | None] = []
        errors: list[BaseException] = []

        def _replace(source, destination) -> None:
            replacement_started.set()
            if not allow_replacement.wait(2):
                raise TimeoutError("replacement was not released")
            source.replace(destination)

        def _run_replacement() -> None:
            try:
                store.replace_from(prepared_path, _replace)
            except BaseException as exc:
                errors.append(exc)

        def _lookup_active_store() -> None:
            lookup_result.append(workspace_store.WorkspaceStore.active(path))
            lookup_finished.set()

        replacement_thread = threading.Thread(target=_run_replacement)
        lookup_thread = threading.Thread(target=_lookup_active_store)
        replacement_thread.start()
        assert replacement_started.wait(2)
        lookup_thread.start()
        try:
            assert not lookup_finished.wait(0.05)
        finally:
            allow_replacement.set()
        replacement_thread.join(2)
        lookup_thread.join(2)

        assert not replacement_thread.is_alive()
        assert not lookup_thread.is_alive()
        assert errors == []
        assert lookup_result == [store]
        assert store.current_generation().manifest == _manifest("prepared")


def test_workspace_store_reopens_after_failed_file_replacement(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    prepared_path = tmp_path / "prepared.itws"
    prepared_path.write_bytes(b"prepared")
    with workspace_store.WorkspaceStore(path, create=True) as store:
        store.publish(_manifest())

        def _fail_replace(_source, destination) -> None:
            raise PermissionError(f"Cannot replace {destination}")

        with pytest.raises(PermissionError, match="Cannot replace"):
            store.replace_from(prepared_path, _fail_replace)

        assert not store.conflicted
        assert workspace_store.WorkspaceStore.active(path) is store
        assert store.current_generation().manifest == _manifest()


def test_workspace_store_uses_prepared_recovery_if_original_cannot_reopen(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    prepared_path = tmp_path / "prepared.itws"
    with workspace_store.WorkspaceStore(prepared_path, create=True) as prepared_store:
        with prepared_store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("prepared")
        prepared_store.publish(_manifest("prepared"))

    with workspace_store.WorkspaceStore(path, create=True) as store:
        store.publish(_manifest())

        def _deny_reopen(*, create: bool, workspace_id: str | None = None) -> None:
            del workspace_id
            if create:
                raise AssertionError("Replacement recovery must not create a file")
            raise PermissionError(errno.EACCES, "reopen denied", path)

        monkeypatch.setattr(store, "_open", _deny_reopen)
        monkeypatch.setattr(workspace_store, "_FILE_ACCESS_RETRY_DELAYS", (0.0, 0.0))
        monkeypatch.setattr(workspace_store.time, "sleep", lambda _delay: None)

        def _fail_replace(_source, destination) -> None:
            raise PermissionError(f"Cannot replace {destination}")

        with pytest.raises(PermissionError, match="Cannot replace"):
            store.replace_from(prepared_path, _fail_replace)

        assert store.conflicted
        assert workspace_store.WorkspaceStore.active(path) is store
        assert store.current_generation().manifest == _manifest("prepared")


def test_workspace_store_retries_reopen_after_successful_replacement(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    prepared_path = tmp_path / "prepared.itws"
    with workspace_store.WorkspaceStore(prepared_path, create=True) as prepared_store:
        with prepared_store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("prepared")
        prepared_store.publish(_manifest("prepared"))

    with workspace_store.WorkspaceStore(path, create=True) as store:
        store.publish(_manifest())
        original_open = store._open
        reopen_attempts = 0
        delays: list[float] = []

        def _open_with_transient_denial(
            *, create: bool, workspace_id: str | None = None
        ) -> None:
            nonlocal reopen_attempts
            del workspace_id
            reopen_attempts += 1
            if reopen_attempts < 3:
                raise PermissionError(errno.EACCES, "reopen denied", path)
            original_open(create=create)

        monkeypatch.setattr(store, "_open", _open_with_transient_denial)
        monkeypatch.setattr(workspace_store, "_FILE_ACCESS_RETRY_DELAYS", (0.0, 0.0))
        monkeypatch.setattr(workspace_store.time, "sleep", delays.append)

        store.replace_from(
            prepared_path,
            lambda source, destination: source.replace(destination),
        )

        assert reopen_attempts == 3
        assert delays == [0.0, 0.0]
        assert workspace_store.WorkspaceStore.active(path) is store
        assert store.current_generation().manifest == _manifest("prepared")


def test_workspace_compaction_retains_prepared_recovery_file(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    recovery_path: pathlib.Path | None = None
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("current")
        store.publish(_manifest("current"))

        def _deny_reopen(*, create: bool, workspace_id: str | None = None) -> None:
            del workspace_id
            if create:
                raise AssertionError("Replacement recovery must not create a file")
            raise PermissionError(errno.EACCES, "reopen denied", path)

        def _fail_replace(*_args, **_kwargs) -> None:
            raise PermissionError(errno.EACCES, "replace denied", path)

        monkeypatch.setattr(store, "_open", _deny_reopen)
        monkeypatch.setattr(workspace_storage, "_replace_workspace_file", _fail_replace)
        monkeypatch.setattr(workspace_store, "_FILE_ACCESS_RETRY_DELAYS", (0.0,))
        monkeypatch.setattr(workspace_store.time, "sleep", lambda _delay: None)

        with pytest.raises(PermissionError, match="replace denied"):
            workspace_storage._compact_workspace_store(store)

        recovery_path = store.recovery_path
        assert recovery_path is not None
        assert recovery_path.exists()
        assert store.conflicted
        assert store.current_generation().manifest == _manifest("current")

    assert recovery_path is not None
    assert not recovery_path.exists()


def test_workspace_compaction_does_not_overwrite_external_replacement(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    external_path = tmp_path / "external.itws"
    with workspace_store.WorkspaceStore(external_path, create=True) as external_store:
        with external_store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("external")
        external_store.publish(_manifest("external"))

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("current")
        store.publish(_manifest("current"))
        original_release = store._release_handle
        replacement_installed = False

        def _release_and_replace_document() -> None:
            nonlocal replacement_installed
            original_release()
            if not replacement_installed:
                replacement_installed = True
                external_path.replace(path)

        monkeypatch.setattr(store, "_release_handle", _release_and_replace_document)

        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            workspace_storage._compact_workspace_store(store)

        assert store.conflicted
        assert workspace_store.WorkspaceStore.active(path) is store
        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            _ = store.h5_file
        store.close()
        with workspace_store.WorkspaceStore(path) as replacement_store:
            assert replacement_store.current_generation().manifest == _manifest(
                "external"
            )


def test_workspace_compaction_rejects_replacement_after_identity_check(
    monkeypatch, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "workspace.itws"
    external_path = tmp_path / "external.itws"
    with workspace_store.WorkspaceStore(external_path, create=True) as external_store:
        with external_store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("external")
        external_store.publish(_manifest("external"))

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("current")
        store.publish(_manifest("current"))
        original_require_identity = store._require_path_identity

        def _replace_after_identity_check(expected: tuple[int, int]) -> None:
            original_require_identity(expected)
            external_path.replace(path)

        monkeypatch.setattr(
            store, "_require_path_identity", _replace_after_identity_check
        )

        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            workspace_storage._compact_workspace_store(store)

        assert store.conflicted
        assert store.current_generation().manifest == _manifest("current")
        store.close()
        with workspace_store.WorkspaceStore(path) as replacement_store:
            assert replacement_store.current_generation().manifest == _manifest(
                "external"
            )


def test_workspace_compaction_preserves_leased_and_serialized_payloads(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            objects = h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]
            objects.create_group("leased")
            objects.create_group("serialized")
            objects.create_group("current")
        store.publish(_manifest("leased"))
        store.publish(_manifest("current"))
        store.acquire_object("leased")
        workspace_store.WorkspaceStore.pin_serialized_reader(
            workspace_id=store.workspace_id,
            path=path,
            object_id="serialized",
            legacy_group_path=None,
        )

        workspace_storage._compact_workspace_store(store)

        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "current",
            "leased",
            "serialized",
        }

        confirmed_pins = store.serialized_reader_pin_snapshot()
        workspace_store.WorkspaceStore.pin_serialized_reader(
            workspace_id=store.workspace_id,
            path=path,
            object_id="serialized",
            legacy_group_path=None,
        )
        workspace_storage._compact_workspace_store(
            store,
            discard_serialized_reader_pins=confirmed_pins,
        )

        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "current",
            "leased",
            "serialized",
        }

        discarded_pins = store.serialized_reader_pin_snapshot()
        workspace_storage._compact_workspace_store(
            store,
            discard_serialized_reader_pins=discarded_pins,
        )

        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "current",
            "leased",
        }
        assert store.serialized_object_ids == {"serialized"}
        store.release_serialized_reader_pins(discarded_pins)
        assert not store.has_serialized_readers
        store.release_object("leased")
        assert not store.collect_garbage(max_objects=1)
        remaining = set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP])
        assert remaining == {"current"}


def test_workspace_compaction_preserves_serialized_legacy_group(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    dataset = xr.Dataset({"data": ("x", np.arange(5, dtype=np.float64))})
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            workspace_arrays._write_workspace_dataset_group_to_file(
                h5_file, "legacy/imagetool", dataset, compression_mode="none"
            )
            h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group("current")
        store.publish(_manifest("current"))
        manager = workspace_arrays.WorkspaceFileManager(
            path,
            group_path="/legacy/imagetool",
        )
        manager.__getstate__()
        del manager
        gc.collect()
        assert store.leased_legacy_group_paths == set()

        workspace_storage._compact_workspace_store(store)

        assert "legacy/imagetool" in store.h5_file
        opened = workspace_arrays.open_workspace_dataset(
            path, "legacy/imagetool", chunks={}
        )
        try:
            assert float(opened["data"].sum().compute()) == 10.0
        finally:
            opened.close()


def test_workspace_store_switch_keeps_preserved_lazy_readers(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.itws"
    target_path = tmp_path / "target.itws"
    dataset = xr.Dataset({"data": ("x", np.arange(5, dtype=np.float64))})
    with workspace_store.WorkspaceStore(source_path, create=True) as source_store:
        with source_store.write_session() as h5_file:
            for group_path in (
                source_store.object_path("old"),
                source_store.object_path("current"),
                "/legacy/imagetool",
            ):
                workspace_arrays._write_workspace_dataset_group_to_file(
                    h5_file,
                    group_path,
                    dataset,
                    compression_mode="none",
                )
        source_store.publish(_manifest("current"))
        old_reader = workspace_arrays.open_workspace_dataset(
            source_path, source_store.object_path("old"), chunks={}
        )
        legacy_reader = workspace_arrays.open_workspace_dataset(
            source_path, "/legacy/imagetool", chunks={}
        )
        try:
            plan = workspace_storage._WorkspaceGenerationPlan(
                manifest=_manifest("current"),
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        "current",
                        source_file=str(source_path),
                        source_path=source_store.object_path("current"),
                    ),
                    workspace_storage._WorkspaceObjectWrite(
                        "old",
                        source_file=str(source_path),
                        source_path=source_store.object_path("old"),
                    ),
                ),
                preserved_groups=(
                    workspace_storage._WorkspaceGroupCopy(
                        source_file=str(source_path),
                        source_path="/legacy/imagetool",
                        target_path="/legacy/imagetool",
                    ),
                ),
            )
            with workspace_store.WorkspaceStore(
                target_path, create=True
            ) as target_store:
                workspace_storage._write_workspace_generation(
                    target_store, plan, compression_mode="none"
                )

            source_store.switch_path(target_path)
            source_path.unlink()

            import dask

            with dask.config.set(scheduler="processes"):
                assert float(old_reader["data"].sum().compute()) == 10.0
                assert float(legacy_reader["data"].sum().compute()) == 10.0
        finally:
            old_reader.close()
            legacy_reader.close()


def test_workspace_store_rejects_replaced_document_path(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    replacement = tmp_path / "replacement.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with workspace_store.WorkspaceStore(replacement, create=True):
            pass
        path.unlink()
        replacement.rename(path)

        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            store.publish(_manifest())
        assert store.conflicted
        assert workspace_store.WorkspaceStore.active(path) is store
        with pytest.raises(workspace_store.WorkspaceStoreConflictError):
            _ = store.h5_file


def test_workspace_store_conflict_keeps_lazy_data_for_save_as(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    replacement = tmp_path / "replacement.itws"
    recovered = tmp_path / "recovered.itws"
    dataset = xr.Dataset({"data": ("x", np.arange(5, dtype=np.float64))})

    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            workspace_arrays._write_workspace_dataset_group_to_file(
                h5_file,
                store.object_path("current"),
                dataset,
                compression_mode="none",
            )
        store.publish(_manifest("current"))
        opened = workspace_arrays.open_workspace_dataset(
            path, store.object_path("current"), chunks={}
        )
        try:
            with workspace_store.WorkspaceStore(replacement, create=True) as other:
                with other.write_session() as h5_file:
                    h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP].create_group(
                        "external"
                    )
                other.publish(_manifest("external"))
            replacement.replace(path)

            with pytest.raises(workspace_store.WorkspaceStoreConflictError):
                store.publish(_manifest("current"))

            assert store.conflicted
            assert float(opened["data"].sum().compute()) == 10.0
            plan = workspace_storage._WorkspaceGenerationPlan(
                manifest=_manifest("current"),
                objects=(
                    workspace_storage._WorkspaceObjectWrite(
                        "current",
                        source_file=str(path),
                        source_path=store.object_path("current"),
                    ),
                ),
            )
            with workspace_store.WorkspaceStore(recovered, create=True) as target:
                workspace_storage._write_workspace_generation(
                    target, plan, compression_mode="none"
                )

            store.switch_path(recovered)
            assert not store.conflicted
            assert float(opened["data"].sum().compute()) == 10.0
        finally:
            opened.close()

    with workspace_store.WorkspaceStore(path) as replacement_store:
        assert replacement_store.current_generation().manifest == _manifest("external")
