from __future__ import annotations

import errno
import threading
import typing

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
        "schema_version": 5,
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
            with (
                store.computation_session(),
                dask.config.set(scheduler="processes"),
            ):
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
                distributed.Client(cluster),
                store.computation_session(),
            ):
                result = opened["data"].sum().compute()
            assert float(result) == 435.0
            assert set(tmp_path.iterdir()) == {path}
        finally:
            opened.close()


def test_workspace_computation_session_blocks_background_write(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    write_started = threading.Event()
    write_finished = threading.Event()
    errors: list[BaseException] = []

    with workspace_store.WorkspaceStore(path, create=True) as store:
        value = xr.DataArray([1.0])
        value.encoding["source"] = str(path)

        def _write() -> None:
            write_started.set()
            try:
                store.clear_staging()
            except BaseException as exc:
                errors.append(exc)
            finally:
                write_finished.set()

        with workspace_arrays.workspace_computation_session(value):
            writer = threading.Thread(target=_write)
            writer.start()
            assert write_started.wait(2)
            assert not write_finished.wait(0.05)

        writer.join(2)
        assert not writer.is_alive()
        assert write_finished.is_set()
        assert errors == []


def test_workspace_worker_reports_inaccessible_shared_path(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    group_path = workspace_store.WorkspaceStore.object_path("payload")
    with workspace_store.WorkspaceStore(path, create=True):
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


def test_workspace_worker_rejects_different_file_at_shared_path(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    replacement = tmp_path / "replacement.itws"
    group_path = workspace_store.WorkspaceStore.object_path("payload")
    with workspace_store.WorkspaceStore(path, create=True):
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


def test_workspace_compaction_preserves_leased_payloads(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "workspace.itws"
    with workspace_store.WorkspaceStore(path, create=True) as store:
        with store.write_session() as h5_file:
            objects = h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]
            objects.create_group("leased")
            objects.create_group("current")
        store.publish(_manifest("leased"))
        store.publish(_manifest("current"))
        store.acquire_object("leased")

        workspace_storage._compact_workspace_store(store)

        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "current",
            "leased",
        }
        store.release_object("leased")
        assert not store.collect_garbage(max_objects=1)
        assert set(store.h5_file[workspace_store._WORKSPACE_OBJECTS_GROUP]) == {
            "current"
        }


def test_workspace_compaction_preserves_live_legacy_group(
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
        opened = workspace_arrays.open_workspace_dataset(
            path, "legacy/imagetool", chunks={}
        )
        try:
            workspace_storage._compact_workspace_store(store)

            assert "legacy/imagetool" in store.h5_file
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

            with (
                source_store.computation_session(),
                dask.config.set(scheduler="processes"),
            ):
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
