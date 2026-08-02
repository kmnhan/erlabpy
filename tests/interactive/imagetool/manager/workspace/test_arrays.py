import contextlib
import gc
import json
import logging
import os
import pathlib
import pickle
import subprocess
import sys
import threading
import time
import types
import typing
import warnings

import dask.base
import h5py
import hdf5plugin
import numpy as np
import pytest
import xarray
import xarray as xr

import erlab
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from tests.interactive.imagetool.manager.workspace._support import (
    _assert_rich_workspace_attr,
    _hdf5_blosc2_level_codec,
    _hdf5_filter_ids,
    _rich_workspace_attr_value,
    _transaction_test_dataset,
    _transaction_test_root_attrs,
    _write_transaction_test_workspace,
)


@pytest.fixture(autouse=True)
def _cleanup_workspace_reader_handoffs() -> typing.Iterator[None]:
    yield
    workspace_arrays._cleanup_workspace_reader_handoffs()


def test_workspace_h5py_helpers_reject_non_workspace_files(tmp_path) -> None:

    fname = tmp_path / "not-workspace.itws"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_group("0")

    with pytest.raises(ValueError, match="Not a valid workspace file"):
        workspace_arrays._workspace_live_root_group_copy_groups(fname)
    with pytest.raises(ValueError, match="Not a valid workspace file"):
        workspace_arrays._workspace_h5_paths_storage_size(fname, ("0",))
    with pytest.raises(ValueError, match="Not a valid workspace file"):
        workspace_arrays._workspace_live_h5_storage_size(fname)

    assert (
        workspace_storage._workspace_obsolete_estimate(tmp_path / "missing.itws") == 0
    )
    assert workspace_arrays._workspace_h5_object_storage_size(object()) == 0


def test_workspace_file_managers_share_one_generation_per_path(tmp_path) -> None:
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(first, engine="h5netcdf")
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(second, engine="h5netcdf")
    first_manager = workspace_arrays.WorkspaceFileManager(first)
    second_first_manager = workspace_arrays.WorkspaceFileManager(first)
    other_manager = workspace_arrays.WorkspaceFileManager(second)
    first_file = first_manager.acquire()
    second_first_file = second_first_manager.acquire()
    other_file = other_manager.acquire()
    try:
        assert first_manager._generation is second_first_manager._generation
        assert first_manager._generation is not other_manager._generation
        assert first_file is second_first_file
        first_manager.close()
        assert first_file._closed
        assert not other_file._closed
        assert second_first_manager.acquire() is not first_file
    finally:
        first_manager.close()
        other_manager.close()


def test_workspace_file_generation_registration_uses_short_registry_lock(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "registry-lock.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    generation_type = workspace_arrays._WorkspaceFileGeneration
    original_add_manager = generation_type.add_manager
    construction_lock_states: list[bool] = []
    lease_lock_states: list[bool] = []

    def _new_generation(*args, **kwargs):
        construction_lock_states.append(
            workspace_arrays._WORKSPACE_FILE_GENERATIONS_LOCK.locked()
        )
        return generation_type(*args, **kwargs)

    def _add_manager(generation, manager=None) -> None:
        lease_lock_states.append(
            workspace_arrays._WORKSPACE_FILE_GENERATIONS_LOCK.locked()
        )
        original_add_manager(generation, manager)

    monkeypatch.setattr(workspace_arrays, "_WorkspaceFileGeneration", _new_generation)
    monkeypatch.setattr(generation_type, "add_manager", _add_manager)

    manager = workspace_arrays.WorkspaceFileManager(fname)
    try:
        assert construction_lock_states == [False]
        assert lease_lock_states == [False]
    finally:
        manager.close()


def test_workspace_file_generation_binds_one_observed_file_state(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "single-state.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    original_file_state = workspace_arrays._workspace_file_state
    observed_state = original_file_state(fname)
    state_calls = 0

    def _file_state(path):
        nonlocal state_calls
        state_calls += 1
        assert pathlib.Path(path).resolve() == fname.resolve()
        return observed_state

    monkeypatch.setattr(workspace_arrays, "_workspace_file_state", _file_state)

    manager = workspace_arrays.WorkspaceFileManager(fname)
    try:
        assert state_calls == 1
        assert manager._generation.file_identity == observed_state[0]
        assert manager._generation._resources.file_identity == observed_state[0]
    finally:
        manager.close()


def test_workspace_file_generation_registers_lease_under_path_lock(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "atomic-lease.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    generation_type = workspace_arrays._WorkspaceFileGeneration
    original_add_manager = generation_type.add_manager
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    cleanup_thread: threading.Thread | None = None

    def _add_manager(generation, manager=None) -> None:
        nonlocal cleanup_thread

        def _cleanup() -> None:
            cleanup_started.set()
            workspace_arrays._cleanup_workspace_file_generation(generation)
            cleanup_finished.set()

        cleanup_thread = threading.Thread(target=_cleanup)
        cleanup_thread.start()
        assert cleanup_started.wait(2)
        assert not cleanup_finished.wait(0.05)
        original_add_manager(generation, manager)

    monkeypatch.setattr(generation_type, "add_manager", _add_manager)

    manager = workspace_arrays.WorkspaceFileManager(fname)
    if cleanup_thread is None:
        raise AssertionError("Generation cleanup thread did not start")
    cleanup_thread.join(2)
    try:
        assert not cleanup_thread.is_alive()
        assert cleanup_finished.is_set()
        assert manager._generation.has_managers()
        manager.acquire()
    finally:
        manager.close()


def test_workspace_reader_helpers_reject_unsafe_directory_and_names(tmp_path) -> None:
    workspace_path = tmp_path / "workspace.itws"
    reader_directory = workspace_arrays._workspace_reader_directory(workspace_path)
    reader_directory.write_text("not a directory")

    workspace_arrays._cleanup_stale_workspace_reader_files(workspace_path)
    with pytest.raises(OSError, match="not a private directory"):
        workspace_arrays._ensure_workspace_reader_directory(workspace_path)

    valid_token = "1" * 32
    assert (
        workspace_arrays._workspace_reader_file_owner_pid(
            pathlib.Path(f"reader-123-{valid_token}.itws")
        )
        == 123
    )
    assert (
        workspace_arrays._workspace_reader_file_owner_pid(
            pathlib.Path(f"handoff-456-{valid_token}.itws")
        )
        == 456
    )
    for name in (
        "reader.itws",
        "reader-bad-token.itws",
        f"export-0-{valid_token}.itws",
        "notes.txt",
    ):
        assert (
            workspace_arrays._workspace_reader_file_owner_pid(pathlib.Path(name))
            is None
        )


def test_workspace_reader_handoffs_do_not_delete_parent_files_after_fork(
    tmp_path,
) -> None:
    handoff_path = tmp_path / "parent-handoff.itws"
    handoff_path.write_bytes(b"parent")
    reader_file = workspace_arrays._WorkspaceReaderFile(
        str(handoff_path), handoff_path.unlink
    )
    workspace_arrays._WORKSPACE_READER_HANDOFFS[str(handoff_path)] = (
        workspace_arrays._WorkspaceReaderHandoff(
            reader_file,
            (str(tmp_path / "workspace.itws"), "parent-revision", "/"),
        )
    )
    old_lock = workspace_arrays._WORKSPACE_READER_HANDOFFS_LOCK
    old_save_lock = workspace_arrays._workspace_save_lock(tmp_path / "workspace.itws")

    workspace_arrays._reset_workspace_reader_handoffs_after_fork()

    assert workspace_arrays._WORKSPACE_READER_HANDOFFS == {}
    assert workspace_arrays._WORKSPACE_READER_HANDOFFS_LOCK is not old_lock
    assert (
        workspace_arrays._workspace_save_lock(tmp_path / "workspace.itws")
        is not old_save_lock
    )
    assert handoff_path.exists()
    reader_file.cleanup()


def test_workspace_file_manager_finalizer_defers_cleanup(tmp_path) -> None:
    fname = tmp_path / "deferred-cleanup.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    generation = manager._generation

    with workspace_arrays._workspace_file_lock(fname):
        manager._finalizer()
        assert generation._finalizer.alive
        assert workspace_arrays._current_workspace_file_generation(fname) is generation

    for _ in range(100):
        if not generation._finalizer.alive:
            break
        threading.Event().wait(0.01)
    assert not generation._finalizer.alive


def test_workspace_file_manager_is_pickleable(tmp_path) -> None:
    fname = tmp_path / "pickleable.itws"
    xr.Dataset({"data": ("x", np.arange(6))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    manager.acquire()
    serialized = pickle.dumps(manager)
    manager.close()

    restored = pickle.loads(serialized)
    try:
        data = restored.acquire().variables["data"][:]
        np.testing.assert_array_equal(data, np.arange(6))
    finally:
        restored.close()


def test_workspace_file_manager_dask_token_does_not_create_handoff(tmp_path) -> None:
    fname = tmp_path / "dask-token.itws"
    _write_transaction_test_workspace(fname)
    manager = workspace_arrays.WorkspaceFileManager(fname, "0/imagetool")
    reader_directory = workspace_arrays._workspace_reader_directory(fname)

    try:
        token_before = dask.base.tokenize(manager)
        assert dask.base.tokenize(manager) == token_before
        dataset_before = workspace_arrays._open_workspace_dataset_from_manager(
            manager,
            "0/imagetool",
            chunks="auto",
        )
        dask_name_before = dataset_before["data"].data.name
        dataset_before.close()
        assert not list(reader_directory.glob("handoff-*.itws"))

        workspace_storage._write_workspace_root_attrs_to_file(
            fname,
            {
                **_transaction_test_root_attrs(),
                "generation_marker": "updated",
            },
        )
        assert dask.base.tokenize(manager) != token_before
        dataset_after = workspace_arrays._open_workspace_dataset_from_manager(
            manager,
            "0/imagetool",
            chunks="auto",
        )
        try:
            assert dataset_after["data"].data.name != dask_name_before
        finally:
            dataset_after.close()
        assert not list(reader_directory.glob("handoff-*.itws"))
    finally:
        manager.close()


def test_workspace_file_manager_opens_published_generation_read_only(tmp_path) -> None:
    fname = tmp_path / "read-only.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)

    try:
        h5_file = manager.acquire()
        assert h5_file.mode == "r"
        with pytest.raises(OSError, match="no write intent"):
            h5_file.attrs["unexpected_write"] = 1
    finally:
        manager.close()


def test_workspace_file_manager_copies_process_export(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "process-export.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    exported_groups: list[str] = []
    original_create = workspace_arrays._create_workspace_group_reader_file

    def _create_reader_file(source, workspace, group, **kwargs):
        exported_groups.append(group)
        return original_create(source, workspace, group, **kwargs)

    monkeypatch.setattr(
        workspace_arrays, "_create_workspace_group_reader_file", _create_reader_file
    )
    try:
        _workspace_path, export_path, _identity, group = manager.__getstate__()
        assert exported_groups == ["/"]
        assert group == "/"
        assert not pathlib.Path(export_path).samefile(fname)
    finally:
        manager.close()


def test_workspace_file_manager_exports_only_requested_group(tmp_path) -> None:
    fname = tmp_path / "group-export.itws"
    tree = xr.DataTree.from_dict(
        {
            "0/imagetool": xr.Dataset({"data": ("x", np.arange(3))}),
            "1/imagetool": xr.Dataset({"data": ("x", np.arange(3) + 10)}),
        }
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()
    manager = workspace_arrays.WorkspaceFileManager(fname, "0/imagetool")

    try:
        _workspace_path, export_path, _identity, group = manager.__getstate__()
        assert group == "/0/imagetool"
        with h5py.File(export_path, "r") as export_file:
            assert set(export_file) == {"0"}
            assert set(export_file["0"]) == {"imagetool"}
            np.testing.assert_array_equal(export_file["0/imagetool/data"], np.arange(3))
    finally:
        manager.close()


def test_workspace_file_manager_handoff_outlives_sender_generation(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "reader-handoff.itws"
    xr.Dataset({"data": ("x", np.arange(6))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    generation = manager._generation
    handoff_paths: list[pathlib.Path] = []
    original_handoff = workspace_arrays._create_workspace_reader_handoff

    def _record_handoff(export_file, workspace_path, revision, group):
        handoff = original_handoff(export_file, workspace_path, revision, group)
        handoff_paths.append(pathlib.Path(handoff.path))
        return handoff

    monkeypatch.setattr(
        workspace_arrays, "_create_workspace_reader_handoff", _record_handoff
    )
    serialized = pickle.dumps(manager)
    assert len(handoff_paths) == 1
    handoff_path = handoff_paths[0]
    export_paths = tuple(
        workspace_arrays._workspace_reader_directory(fname).glob("export-*.itws")
    )
    assert len(export_paths) == 1

    manager._finalizer()
    with workspace_arrays._workspace_file_lock(fname):
        workspace_arrays._cleanup_workspace_file_generation_locked(generation)

    assert not export_paths[0].exists()
    assert handoff_path.exists()
    restored = pickle.loads(serialized)
    try:
        assert handoff_path.exists()
        np.testing.assert_array_equal(
            restored.acquire().variables["data"][:], np.arange(6)
        )
    finally:
        restored.close()


def test_workspace_file_manager_cleans_retired_serialization_handoff(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "retired-handoff.itws"
    xr.Dataset({"data": ("x", np.arange(6))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    generation = manager._generation
    pickle.dumps(manager)
    handoff_path = next(
        workspace_arrays._workspace_reader_directory(fname).glob("handoff-*.itws")
    )
    monkeypatch.setattr(
        workspace_arrays, "_WORKSPACE_READER_HANDOFF_RETENTION_SECONDS", 0.0
    )

    manager._finalizer()
    with workspace_arrays._workspace_file_lock(fname):
        workspace_arrays._cleanup_workspace_file_generation_locked(generation)
    workspace_arrays._cleanup_expired_workspace_reader_handoffs()

    assert not handoff_path.exists()
    assert not workspace_arrays._WORKSPACE_READER_HANDOFFS


def test_workspace_file_manager_reuses_serialization_handoff(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "reusable-handoff.itws"
    xr.Dataset({"data": ("x", np.arange(6))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    handoff_paths: list[pathlib.Path] = []
    original_handoff = workspace_arrays._create_workspace_reader_handoff

    def _record_handoff(export_file, workspace_path, revision, group):
        handoff = original_handoff(export_file, workspace_path, revision, group)
        handoff_paths.append(pathlib.Path(handoff.path))
        return handoff

    monkeypatch.setattr(
        workspace_arrays, "_create_workspace_reader_handoff", _record_handoff
    )
    first_payload = pickle.dumps(manager)
    second_payload = pickle.dumps(manager)

    assert len(handoff_paths) == 2
    assert handoff_paths[0] == handoff_paths[1]
    assert all(path.exists() for path in handoff_paths)
    first = pickle.loads(first_payload)
    repeated = pickle.loads(first_payload)
    second = pickle.loads(second_payload)
    try:
        assert all(path.exists() for path in handoff_paths)
        for restored in (first, repeated, second):
            np.testing.assert_array_equal(
                restored.acquire().variables["data"][:], np.arange(6)
            )
    finally:
        manager.close()
        first.close()
        repeated.close()
        second.close()


def test_workspace_file_manager_failed_pickle_reuses_handoff(tmp_path) -> None:
    fname = tmp_path / "failed-pickle.itws"
    xr.Dataset({"data": ("x", np.arange(6))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)

    try:
        for _ in range(3):
            with pytest.raises(pickle.PicklingError):
                pickle.dumps({"manager": manager, "unpickleable": lambda: None})

        handoff_paths = tuple(
            workspace_arrays._workspace_reader_directory(fname).glob("handoff-*.itws")
        )
        assert len(handoff_paths) == 1
        assert tuple(workspace_arrays._WORKSPACE_READER_HANDOFFS) == (
            str(handoff_paths[0]),
        )
    finally:
        manager.close()


def test_workspace_file_manager_bounds_unconsumed_reader_generations(
    monkeypatch, tmp_path
) -> None:
    first_path = tmp_path / "first-pending.itws"
    second_path = tmp_path / "second-pending.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(first_path, engine="h5netcdf")
    xr.Dataset({"data": ("x", np.arange(3) + 10)}).to_netcdf(
        second_path, engine="h5netcdf"
    )
    first = workspace_arrays.WorkspaceFileManager(first_path)
    second = workspace_arrays.WorkspaceFileManager(second_path)
    monkeypatch.setattr(
        workspace_arrays, "_WORKSPACE_MAX_PENDING_READER_GENERATIONS", 1
    )
    first_payload = pickle.dumps(first)

    try:
        with pytest.raises(RuntimeError, match="Wait for background work"):
            pickle.dumps(second)
        restored = pickle.loads(first_payload)
        try:
            np.testing.assert_array_equal(
                restored.acquire().variables["data"][:], np.arange(3)
            )
        finally:
            restored.close()
    finally:
        first.close()
        second.close()


def test_workspace_file_manager_bounds_unconsumed_reader_handoffs(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "pending-readers.itws"
    tree = xr.DataTree.from_dict(
        {
            "0": xr.Dataset({"data": ("x", np.arange(3))}),
            "1": xr.Dataset({"data": ("x", np.arange(3) + 10)}),
        }
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()
    first_manager = workspace_arrays.WorkspaceFileManager(fname, "0")
    second_manager = workspace_arrays.WorkspaceFileManager(fname, "1")
    monkeypatch.setattr(workspace_arrays, "_WORKSPACE_MAX_PENDING_READER_HANDOFFS", 1)
    missing_handoff = tmp_path / "missing-handoff.itws"
    workspace_arrays._WORKSPACE_READER_HANDOFFS[str(missing_handoff)] = (
        workspace_arrays._WorkspaceReaderHandoff(
            workspace_arrays._WorkspaceReaderFile(str(missing_handoff), lambda: None),
            (str(fname), "stale-revision", "/"),
        )
    )
    first_payload = pickle.dumps(first_manager)
    assert str(missing_handoff) not in workspace_arrays._WORKSPACE_READER_HANDOFFS
    repeated_payload = pickle.dumps(first_manager)

    try:
        with pytest.raises(RuntimeError, match="Wait for background work"):
            pickle.dumps(second_manager)
        first = pickle.loads(first_payload)
        repeated = pickle.loads(repeated_payload)
        try:
            for restored in (first, repeated):
                np.testing.assert_array_equal(
                    restored.acquire().groups["0"].variables["data"][:], np.arange(3)
                )
        finally:
            first.close()
            repeated.close()
    finally:
        first_manager.close()
        second_manager.close()


def test_workspace_file_manager_rejects_missing_serialized_export(tmp_path) -> None:
    fname = tmp_path / "missing-export.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    state = manager.__getstate__()
    pathlib.Path(state[1]).unlink()
    restored = workspace_arrays.WorkspaceFileManager.__new__(
        workspace_arrays.WorkspaceFileManager
    )

    try:
        with pytest.raises(RuntimeError, match="no longer available"):
            restored.__setstate__(state)
    finally:
        manager.close()


def test_workspace_file_manager_root_attr_update_refreshes_current_generation(
    tmp_path,
) -> None:
    fname = tmp_path / "updated-export.itws"
    _write_transaction_test_workspace(fname)
    manager = workspace_arrays.WorkspaceFileManager(fname)
    generation = manager._generation
    serialized_before = pickle.dumps(manager)

    workspace_storage._write_workspace_root_attrs_to_file(
        fname,
        {
            **_transaction_test_root_attrs(),
            "generation_marker": "updated",
        },
    )
    current = workspace_arrays.WorkspaceFileManager(fname)

    before = pickle.loads(serialized_before)
    try:
        assert manager._generation is generation
        assert current._generation is generation
        assert not generation.retired
        assert manager.acquire().attrs["generation_marker"] == "updated"
        assert "generation_marker" not in before.acquire().attrs
        assert current.acquire().attrs["generation_marker"] == "updated"
    finally:
        manager.close()
        before.close()
        current.close()


def test_workspace_file_manager_export_uses_logical_generation_revision(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "coarse-timestamp.itws"
    _write_transaction_test_workspace(fname, value=1.0)
    original_file_state = workspace_arrays._workspace_file_state
    canonical_path = fname.resolve()
    fixed_mtime = fname.stat().st_mtime_ns

    def _coarse_file_state(path):
        identity, exists = original_file_state(path)
        if pathlib.Path(identity[0]).resolve() == canonical_path:
            identity = (*identity[:3], fixed_mtime)
        return identity, exists

    monkeypatch.setattr(workspace_arrays, "_workspace_file_state", _coarse_file_state)
    before = workspace_arrays.WorkspaceFileManager(fname, "0/imagetool")
    generation = before._generation
    revision_before = generation.revision
    state_before = before.__getstate__()

    workspace_storage._write_workspace_transaction_file(
        fname,
        (("0", {"0/imagetool": _transaction_test_dataset(9.0, title="new")}),),
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    after = workspace_arrays.WorkspaceFileManager(fname, "0/imagetool")
    state_after = after.__getstate__()
    try:
        assert generation.revision != revision_before
        assert state_after[1] != state_before[1]
        with h5py.File(state_before[1], "r") as old_export:
            assert np.asarray(old_export["0/imagetool/data"]).item() == 1.0
        with h5py.File(state_after[1], "r") as new_export:
            assert np.asarray(new_export["0/imagetool/data"]).item() == 9.0
    finally:
        before.close()
        after.close()


def test_workspace_file_manager_export_rejects_source_change(
    monkeypatch, tmp_path
) -> None:
    destination = tmp_path / "current.itws"
    replacement = tmp_path / "replacement.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(destination, engine="h5netcdf")
    xr.Dataset({"data": ("x", np.arange(3) + 1)}).to_netcdf(
        replacement, engine="h5netcdf"
    )
    manager = workspace_arrays.WorkspaceFileManager(destination)
    original_create = workspace_arrays._create_workspace_group_reader_file

    def _create_then_replace(source, workspace_path, group, **kwargs):
        result = original_create(source, workspace_path, group, **kwargs)
        os.replace(replacement, destination)
        return result

    monkeypatch.setattr(
        workspace_arrays,
        "_create_workspace_group_reader_file",
        _create_then_replace,
    )
    try:
        with pytest.raises(RuntimeError, match="changed while it was being exported"):
            pickle.dumps(manager)
        assert not list(
            workspace_arrays._workspace_reader_directory(destination).glob(
                "export-*.itws"
            )
        )
    finally:
        manager.close()


def test_serialized_workspace_reader_rejects_changed_export(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "serialized.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    state = manager.__getstate__()
    original_create = workspace_arrays._create_workspace_reader_file

    def _create_then_touch(source, workspace_path, **kwargs):
        result = original_create(source, workspace_path, **kwargs)
        source_stat = pathlib.Path(source).stat()
        os.utime(
            source,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
        )
        return result

    monkeypatch.setattr(
        workspace_arrays, "_create_workspace_reader_file", _create_then_touch
    )
    restored = workspace_arrays.WorkspaceFileManager.__new__(
        workspace_arrays.WorkspaceFileManager
    )
    try:
        with pytest.raises(RuntimeError, match="changed while it was being opened"):
            restored.__setstate__(state)
        assert not list(
            workspace_arrays._workspace_reader_directory(fname).glob("reader-*.itws")
        )
    finally:
        manager.close()


def test_pickled_workspace_reader_does_not_hold_canonical_file(tmp_path) -> None:
    destination = tmp_path / "current.itws"
    source = tmp_path / "replacement.itws"
    xr.Dataset({"data": ("x", np.arange(6))}).to_netcdf(destination, engine="h5netcdf")
    xr.Dataset({"data": ("x", np.arange(6) + 10)}).to_netcdf(source, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(destination)
    serialized = pickle.dumps(manager)
    serialized_path = tmp_path / "manager.pickle"
    ready_path = tmp_path / "reader.ready"
    release_path = tmp_path / "reader.release"
    serialized_path.write_bytes(serialized)
    child_code = """
import json
import pathlib
import pickle
import sys
import time

manager = pickle.loads(pathlib.Path(sys.argv[1]).read_bytes())
try:
    h5_file = manager.acquire()
    pathlib.Path(sys.argv[2]).touch()
    deadline = time.monotonic() + 10
    while not pathlib.Path(sys.argv[3]).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Workspace reader process did not receive release")
        time.sleep(0.01)
    print(json.dumps(h5_file.variables["data"][:].tolist()))
finally:
    manager.close()
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_code,
            str(serialized_path),
            str(ready_path),
            str(release_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    try:
        while not ready_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert ready_path.exists()
        workspace_storage._replace_workspace_file(source, destination)
    finally:
        release_path.touch()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
            raise
        finally:
            manager.close()

    assert process.returncode == 0, stderr
    assert json.loads(stdout) == np.arange(6).tolist()
    with xr.open_dataset(destination, engine="h5netcdf") as current:
        np.testing.assert_array_equal(current["data"], np.arange(6) + 10)


def test_workspace_file_manager_cache_evicts_old_handles(tmp_path) -> None:
    paths = [tmp_path / f"cache-{index}.itws" for index in range(3)]
    for path in paths:
        xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(path, engine="h5netcdf")

    managers = [workspace_arrays.WorkspaceFileManager(path) for path in paths]
    try:
        with xr.set_options(file_cache_maxsize=1):
            first_file = managers[0].acquire()
            second_file = managers[1].acquire()
            assert first_file._closed

            third_file = managers[2].acquire()
            assert second_file._closed
            assert not third_file._closed
    finally:
        for manager in managers:
            manager.close()


def test_workspace_file_manager_serialization_pins_replaced_generation(
    tmp_path,
) -> None:
    destination = tmp_path / "current.itws"
    source = tmp_path / "replacement.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(destination, engine="h5netcdf")
    xr.Dataset({"data": ("x", np.arange(3) + 1)}).to_netcdf(source, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(destination)
    _workspace_path, export_path, _identity, _group = manager.__getstate__()
    assert pathlib.Path(export_path) != destination.resolve()
    serialized = pickle.dumps(manager)

    workspace_storage._replace_workspace_file(source, destination)

    restored = pickle.loads(serialized)
    serialized_after_replace = pickle.dumps(manager)
    restored_after_replace = pickle.loads(serialized_after_replace)
    current = workspace_arrays.WorkspaceFileManager(destination)
    try:
        for old_manager in (manager, restored, restored_after_replace):
            np.testing.assert_array_equal(
                old_manager.acquire().variables["data"][:], np.arange(3)
            )
        np.testing.assert_array_equal(
            current.acquire().variables["data"][:], np.arange(3) + 1
        )
    finally:
        for file_manager in (manager, restored, restored_after_replace, current):
            file_manager.close()
        del manager, restored, restored_after_replace, current
        gc.collect()


def test_workspace_file_manager_rejects_external_generation_replacement(
    tmp_path,
) -> None:
    destination = tmp_path / "current.itws"
    source = tmp_path / "replacement.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(destination, engine="h5netcdf")
    xr.Dataset({"data": ("x", np.arange(3) + 1)}).to_netcdf(source, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(destination)
    manager.acquire()
    manager.close()

    os.replace(source, destination)

    try:
        with pytest.raises(RuntimeError, match="reader generation changed"):
            manager.acquire()
        with pytest.raises(RuntimeError, match="changed before it could be exported"):
            pickle.dumps(manager)
        current = workspace_arrays.WorkspaceFileManager(destination)
        try:
            np.testing.assert_array_equal(
                current.acquire().variables["data"][:], np.arange(3) + 1
            )
        finally:
            current.close()
    finally:
        manager.close()


def test_workspace_file_manager_rejects_external_in_place_mutation(tmp_path) -> None:
    fname = tmp_path / "mutated.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    original_stat = fname.stat()
    manager.acquire()
    manager.close()

    with h5py.File(fname, "r+") as h5_file:
        h5_file.attrs["external_update"] = True
    os.utime(
        fname,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
    )

    try:
        with pytest.raises(RuntimeError, match="reader generation changed"):
            manager.acquire()
        current = workspace_arrays.WorkspaceFileManager(fname)
        try:
            assert current.acquire().attrs["external_update"]
        finally:
            current.close()
    finally:
        manager.close()


def test_disposed_workspace_generation_rejects_reuse(tmp_path) -> None:
    fname = tmp_path / "disposed.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    generation = manager._generation
    manager._finalizer.detach()
    assert generation.release_manager()

    with workspace_arrays._workspace_file_lock(fname):
        workspace_arrays._cleanup_workspace_file_generation_locked(generation)

    with pytest.raises(RuntimeError, match="no longer available"):
        generation.acquire(needs_lock=True)
    with pytest.raises(RuntimeError, match="no longer available"):
        generation.acquire_context(needs_lock=True)
    with pytest.raises(RuntimeError, match="no longer available"):
        generation.add_manager()
    generation.dispose()


@pytest.mark.parametrize("already_open", [False, True])
def test_workspace_file_manager_acquire_context_error_closes_only_new_handle(
    tmp_path, *, already_open: bool
) -> None:
    fname = tmp_path / "acquire-context.itws"
    xr.Dataset({"data": ("x", np.arange(3))}).to_netcdf(fname, engine="h5netcdf")
    manager = workspace_arrays.WorkspaceFileManager(fname)
    if already_open:
        manager.acquire()

    try:
        with (
            pytest.raises(RuntimeError, match="read failed"),
            manager.acquire_context() as acquired,
        ):
            raise RuntimeError("read failed")
        assert acquired._closed is not already_open
    finally:
        manager.close()


def test_workspace_h5py_filter_matching_edge_cases(tmp_path) -> None:

    fname = tmp_path / "filters.h5"
    with h5py.File(fname, "w") as h5_file:
        plain = h5_file.create_dataset("plain", data=np.arange(3))
        compressed = h5_file.create_dataset(
            "compressed",
            data=np.arange(3),
            **hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            ),
        )
        group = h5_file.create_group("payload")
        gzip_data = group.create_dataset("data", data=np.arange(3), compression="gzip")
        metadata_group = h5_file.create_group("metadata")
        metadata_group.create_group("nested")

        assert workspace_arrays._workspace_h5py_blosc2_options_match((1, 2), (1, 2))
        assert not workspace_arrays._workspace_h5py_blosc2_options_match((1,), (2,))
        assert workspace_arrays._workspace_h5py_dataset_matches_encoding(plain, {})
        assert not workspace_arrays._workspace_h5py_dataset_matches_encoding(
            plain, {"compression": hdf5plugin.Blosc2.filter_id}
        )
        assert workspace_arrays._workspace_h5py_dataset_matches_encoding(
            compressed, {"compression": hdf5plugin.Blosc2.filter_id}
        )
        assert workspace_arrays._workspace_h5py_dataset_matches_encoding(
            compressed, workspace_arrays._workspace_blosc2_encoding("zstd1")
        )
        assert not workspace_arrays._workspace_h5py_dataset_matches_encoding(
            compressed, workspace_arrays._workspace_blosc2_encoding("blosclz3")
        )
        gzip_filter = workspace_arrays._workspace_h5py_filter_options(gzip_data)
        assert workspace_arrays._workspace_h5py_dataset_matches_encoding(
            gzip_data,
            {"compression": 1, "compression_opts": gzip_filter[1]},
        )
        assert not workspace_arrays._h5_group_matches_compression(
            h5_file, "missing", "none"
        )
        assert not workspace_arrays._h5_group_matches_compression(
            h5_file, "plain", "none"
        )
        assert workspace_arrays._h5_group_matches_compression(
            h5_file, "metadata", "none"
        )
        assert not workspace_arrays._workspace_h5_group_matches_compression_mode(
            h5_file,
            "missing",
            xr.Dataset({"data": ("x", np.arange(3))}),
            "none",
        )
        assert not workspace_arrays._workspace_h5_group_matches_compression_mode(
            h5_file,
            "payload",
            xr.Dataset({"missing": ("x", np.arange(3))}),
            "none",
        )
        assert not workspace_arrays._workspace_h5_group_matches_compression_mode(
            h5_file,
            "payload",
            xr.Dataset({"data": ("x", np.arange(3))}),
            "none",
        )


def test_workspace_h5py_copy_rebuilds_attrs_and_dimension_scales(tmp_path) -> None:

    class _FakeH5Type:
        def __init__(
            self,
            type_class: object,
            *,
            super_type: "_FakeH5Type | None" = None,
            member_types: tuple["_FakeH5Type", ...] = (),
        ) -> None:
            self._type_class = type_class
            self._super_type = super_type
            self._member_types = member_types
            self.closed = False

        def get_class(self) -> object:
            return self._type_class

        def get_super(self) -> "_FakeH5Type":
            if self._super_type is None:
                raise RuntimeError("missing super type")
            return self._super_type

        def get_nmembers(self) -> int:
            return len(self._member_types)

        def get_member_type(self, index: int) -> "_FakeH5Type":
            return self._member_types[index]

        def close(self) -> None:
            self.closed = True

    array_member = _FakeH5Type(h5py.h5t.REFERENCE)
    array_type = _FakeH5Type(h5py.h5t.ARRAY, super_type=array_member)
    assert workspace_arrays._workspace_h5py_type_contains_reference(array_type)
    assert array_member.closed

    plain_member = _FakeH5Type(h5py.h5t.INTEGER)
    compound_type = _FakeH5Type(h5py.h5t.COMPOUND, member_types=(plain_member,))
    assert not workspace_arrays._workspace_h5py_type_contains_reference(compound_type)
    assert plain_member.closed

    fname = tmp_path / "dimension-scales.h5"
    with h5py.File(fname, "w") as h5_file:
        source = h5_file.create_group("source")
        source.create_group("nested")
        source.create_dataset("plain", data=np.arange(2))
        source["plain"].attrs["_Netcdf4Coordinates"] = np.array([0, 1])
        source.create_dataset("scale_without_dimid", data=np.arange(2))
        source["scale_without_dimid"].attrs["CLASS"] = b"DIMENSION_SCALE"
        source["named_type"] = np.dtype("int32")

        scale = source.create_dataset("x", data=np.arange(2))
        scale.attrs["CLASS"] = b"DIMENSION_SCALE"
        scale.attrs["NAME"] = b"x"
        scale.attrs["_Netcdf4Dimid"] = np.int32(0)

        values = source.create_dataset("values", data=np.arange(2))
        values.attrs["_Netcdf4Coordinates"] = np.array([0])
        values_missing_scale = source.create_dataset(
            "values_missing_scale", data=np.arange(2)
        )
        values_missing_scale.attrs["_Netcdf4Coordinates"] = np.array([99])
        source.attrs["reference"] = values.ref
        source.attrs["reference_array"] = np.array([values.ref], dtype=h5py.ref_dtype)
        source.attrs["reference_compound"] = np.array(
            [(values.ref, 1)],
            dtype=np.dtype([("reference", h5py.ref_dtype), ("value", np.int32)]),
        )[0]

        target = h5_file.create_group("target")
        target.create_group("nested")
        target.create_dataset("plain", data=np.arange(2))
        target.create_dataset("scale_without_dimid", data=np.arange(2))
        target["named_type"] = np.dtype("int32")
        target.create_dataset("x", data=np.arange(2))
        target.create_dataset("values", data=np.arange(2))
        target.create_dataset("values_missing_scale", data=np.arange(2))

        assert workspace_arrays._workspace_h5py_attr_text(np.bytes_(b"x")) == "x"
        assert (
            workspace_arrays._workspace_h5py_attr_text(
                types.SimpleNamespace(decode=lambda: "decoded")
            )
            == "decoded"
        )
        assert workspace_arrays._workspace_h5py_attr_text("x") == "x"
        assert workspace_arrays._workspace_h5py_attr_text(object()) is None

        workspace_arrays._workspace_h5py_rebuild_dimension_scales(source, target)

        assert "reference" not in target.attrs
        assert "reference_array" not in target.attrs
        assert "reference_compound" not in target.attrs
        assert target["x"].attrs["_Netcdf4Dimid"] == 0
        assert len(target["values"].dims[0]) == 1


def test_copy_workspace_h5_group_to_open_file_edge_cases(tmp_path) -> None:

    fname = tmp_path / "copy.h5"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_dataset("dataset", data=np.arange(2))
        h5_file.create_group("source").create_dataset("data", data=np.arange(2))
        h5_file.create_group("target").create_group("source")

        assert not workspace_arrays._copy_workspace_h5_group_to_open_file(
            h5_file, h5_file, "missing", "target/missing", None
        )
        assert not workspace_arrays._copy_workspace_h5_group_to_open_file(
            h5_file, h5_file, "dataset", "target/dataset", None
        )
        assert workspace_arrays._copy_workspace_h5_group_to_open_file(
            h5_file,
            h5_file,
            "source",
            "target/source",
            {"title": "copied"},
        )
        assert h5_file["target/source"].attrs["title"] == "copied"


def test_write_workspace_dataset_group_h5py_cleans_failed_independent_items(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "independent-items.itws"
    saved_tool_data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    ds = xr.Dataset(
        {
            saved_tool_data_name: (
                (workspace_arrays._SAVED_TOOL_DATA_REFERENCE_DIM,),
                np.empty(0, dtype=np.float64),
            )
        }
    )
    monkeypatch.setattr(
        workspace_arrays,
        "_workspace_h5py_create_dataset",
        lambda *_args, **_kwargs: None,
    )

    assert not workspace_arrays._write_workspace_dataset_group_h5py(fname, "0/tool", ds)

    with h5py.File(fname, "r") as h5_file:
        assert "0/tool" not in h5_file


def test_workspace_dataset_encoding_compresses_only_large_numeric_payloads() -> None:
    import hdf5plugin

    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            ),
            "small": ("x", np.arange(512, dtype=np.float64)),
            "metadata": ("label", np.array(["a", "b"], dtype=object)),
        },
        coords={
            "x": np.linspace(-1.0, 1.0, 512),
            "y": np.linspace(-2.0, 2.0, 512),
            "label": ["a", "b"],
        },
    )

    encoding = workspace_arrays.workspace_dataset_encoding(ds)

    assert set(encoding) == {"large"}
    assert encoding["large"] == dict(
        hdf5plugin.Blosc2(
            cname="zstd",
            clevel=1,
            filters=hdf5plugin.Blosc2.SHUFFLE,
        )
    )


def test_workspace_dataset_encoding_supports_compression_modes() -> None:
    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        }
    )

    assert (
        workspace_arrays.workspace_dataset_encoding(ds, compression_mode="none") == {}
    )
    assert workspace_arrays.workspace_dataset_encoding(
        ds, compression_mode="blosclz3"
    ) == {
        "large": dict(
            hdf5plugin.Blosc2(
                cname="blosclz",
                clevel=3,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    }
    assert workspace_arrays.workspace_dataset_encoding(
        ds, compression_mode="zstd1"
    ) == {
        "large": dict(
            hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    }
    assert workspace_arrays.workspace_dataset_encoding(ds, compress=True) == {
        "large": dict(
            hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    }
    with pytest.raises(ValueError, match="Unknown workspace compression mode"):
        workspace_arrays.workspace_dataset_encoding(
            ds,
            compression_mode=typing.cast(
                "workspace_arrays.WorkspaceCompressionMode", "missing"
            ),
        )


def test_workspace_dataset_encoding_respects_compression_preference() -> None:
    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        }
    )
    old_value = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        assert workspace_arrays.workspace_dataset_encoding(ds) == {}

        erlab.interactive.options["io/workspace/compression"] = "blosclz3"
        assert workspace_arrays.workspace_dataset_encoding(ds)["large"] == dict(
            hdf5plugin.Blosc2(
                cname="blosclz",
                clevel=3,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )

        erlab.interactive.options["io/workspace/compression"] = "zstd1"
        assert workspace_arrays.workspace_dataset_encoding(ds)["large"] == dict(
            hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_value


def test_workspace_dataset_encoding_persists_dask_chunksizes() -> None:
    data = xr.DataArray(
        np.arange(25, dtype=np.float64).reshape(5, 5),
        dims=("x", "y"),
    ).chunk({"x": (2, 3), "y": (4, 1)})
    ds = xr.Dataset({"data": data})

    assert workspace_arrays.workspace_dataset_encoding(ds, compress=False) == {
        "data": {"chunksizes": (2, 4)}
    }


def test_workspace_chunksizes_rejects_invalid_chunk_shapes() -> None:
    assert (
        workspace_arrays._workspace_chunksizes_for_dataarray(
            types.SimpleNamespace(chunks=((1,),), ndim=1, shape=(0,))
        )
        is None
    )
    assert (
        workspace_arrays._workspace_chunksizes_for_dataarray(
            types.SimpleNamespace(chunks=((0,),), ndim=1, shape=(5,))
        )
        is None
    )


def test_workspace_xarray_path_helpers_cover_fallbacks(monkeypatch, tmp_path) -> None:
    class _BadPath(os.PathLike):
        def __fspath__(self) -> str:
            raise TypeError

    assert workspace_arrays._normalized_file_path(object()) is None
    assert workspace_arrays._normalized_file_path(_BadPath()) is None
    assert workspace_arrays._normalized_file_path("") is None

    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert workspace_arrays._normalized_file_path(tmp_path / "workspace.itws") == str(
        tmp_path / "workspace.itws"
    )

    monkeypatch.setattr(workspace_arrays, "_normalized_file_path", lambda _path: None)
    lock = workspace_arrays._workspace_file_lock("fallback.itws")
    assert lock is workspace_arrays._workspace_file_lock("fallback.itws")

    def _raise_stat_file_not_found(_path: str):
        raise FileNotFoundError

    monkeypatch.setattr(workspace_arrays.os, "stat", _raise_stat_file_not_found)
    assert workspace_arrays._workspace_file_identity("missing.itws") == (
        "missing.itws",
        0,
        0,
        0,
    )

    def _raise_stat_permission_error(_path: str):
        raise PermissionError

    monkeypatch.setattr(workspace_arrays.os, "stat", _raise_stat_permission_error)
    with pytest.raises(PermissionError):
        workspace_arrays._workspace_file_identity("denied.itws")


def test_workspace_file_manager_uses_fsdecode_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_arrays, "ensure_workspace_hdf5_filters_registered", lambda: None
    )
    monkeypatch.setattr(workspace_arrays, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(
        workspace_arrays,
        "_workspace_file_identity",
        lambda path: (path, 0, 0, 0),
    )

    file_manager = workspace_arrays.WorkspaceFileManager("fallback.itws")

    assert file_manager.workspace_path == "fallback.itws"
    assert file_manager._generation.path == "fallback.itws"


def test_open_workspace_dataset_uses_fsdecode_fallback(monkeypatch) -> None:
    calls: list[tuple[object, str, str | None]] = []

    class _FakeFileManager:
        def __init__(self, path: str, _group: str = "/") -> None:
            self.workspace_path = path

    def _fake_open(file_manager, group: str, *, chunks: str | None):
        calls.append((file_manager, group, chunks))
        return "dataset"

    monkeypatch.setattr(workspace_arrays, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(workspace_arrays, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        workspace_arrays, "_open_workspace_dataset_from_manager", _fake_open
    )

    assert (
        workspace_arrays.open_workspace_dataset("fallback.itws", "/0", chunks=None)
        == "dataset"
    )
    file_manager, group, chunks = calls[0]
    assert isinstance(file_manager, _FakeFileManager)
    assert file_manager.workspace_path == "fallback.itws"
    assert group == "/0"
    assert chunks is None


def test_open_workspace_datatree_closes_partial_groups_on_error(monkeypatch) -> None:
    closed: list[str] = []

    class _FakeDataset:
        def __init__(self, group_path: str) -> None:
            self.group_path = group_path

        def close(self) -> None:
            closed.append(self.group_path)

    class _FakeFileManager:
        workspace_path = "fallback.itws"

        def __init__(self, _path: str, _group: str = "/") -> None:
            pass

        def acquire_context(self):
            return contextlib.nullcontext(object())

        def _release(self) -> None:
            pass

    def _fake_open(_file_manager, group_path: str, *, chunks: str | None):
        if group_path == "/broken":
            raise RuntimeError("broken group")
        return _FakeDataset(group_path)

    monkeypatch.setattr(workspace_arrays, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(workspace_arrays, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        workspace_arrays,
        "_iter_h5netcdf_group_paths",
        lambda _h5_file: ("/", "/broken"),
    )
    monkeypatch.setattr(
        workspace_arrays, "_open_workspace_dataset_from_manager", _fake_open
    )

    with pytest.raises(RuntimeError, match="broken group"):
        workspace_arrays.open_workspace_datatree("fallback.itws", chunks="auto")

    assert closed == ["/"]


def test_open_workspace_datatree_reads_uncompressed_workspace(tmp_path) -> None:
    ds = xr.Dataset(
        {"data": (("x", "y"), np.arange(12, dtype=np.float64).reshape(3, 4))},
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    fname = tmp_path / "uncompressed.itws"
    try:
        tree.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)
    finally:
        tree.close()

    opened = workspace_arrays.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
    finally:
        opened.close()


def test_workspace_h5py_attrs_and_root_validation(tmp_path) -> None:

    assert workspace_arrays._h5py_attrs_to_dict({"name": b"value"}) == {"name": "value"}

    fname = tmp_path / "plain.h5"
    with h5py.File(fname, "w"):
        pass

    with pytest.raises(ValueError, match="Not a valid workspace file"):
        workspace_arrays._read_workspace_root_attrs_h5py(fname)


def test_replace_h5_attrs_drops_invalid_attr_names(tmp_path) -> None:

    fname = tmp_path / "replace-invalid-attrs.itws"
    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("0/imagetool")
        group.attrs["old"] = "removed"

        workspace_arrays._replace_h5_attrs(
            group.attrs,
            {"": "dropped", None: "dropped", "note": "", "valid": "kept"},
        )

        assert "old" not in group.attrs
        assert "" not in list(group.attrs)
        assert group.attrs["note"] == ""
        assert group.attrs["valid"] == "kept"


def test_replace_h5_attrs_encodes_non_native_attr_values(tmp_path) -> None:

    fname = tmp_path / "replace-rich-attrs.itws"
    rich_attr = _rich_workspace_attr_value()
    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("0/imagetool")

        workspace_arrays._replace_h5_attrs(
            group.attrs,
            {"Single Motor Scan": rich_attr, "valid": "kept"},
        )

        assert "Single Motor Scan" not in group.attrs
        assert workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR in group.attrs
        decoded = workspace_arrays._h5py_attrs_to_dict(group.attrs)
        assert decoded["valid"] == "kept"
        _assert_rich_workspace_attr(decoded["Single Motor Scan"])


def _assert_workspace_h5py_roundtrip(
    tmp_path: pathlib.Path, label: str, data: xr.DataArray
) -> tuple[xr.Dataset, xr.Dataset, pathlib.Path]:
    data_name = _ITOOL_DATA_NAME
    fname = tmp_path / f"{label}.itws"
    ds = data.rename(data_name).to_dataset()

    assert workspace_arrays._workspace_dataset_can_write_h5py(ds)
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=data_name,
    )
    assert loaded is not None

    opened = workspace_arrays.open_workspace_dataset(fname, "0/imagetool", chunks=None)
    try:
        opened_loaded = opened.load()
    finally:
        opened.close()
    xr.testing.assert_equal(loaded, opened_loaded)
    return loaded, opened_loaded, fname


def test_workspace_h5py_fast_path_roundtrips_scalar_coords(tmp_path) -> None:

    fname = tmp_path / "scalar-fast-path.itws"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(3.0), "temperature": 20.0},
        attrs={"coordinates": b""},
        name=_ITOOL_DATA_NAME,
    )
    ds = data.to_dataset()

    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=_ITOOL_DATA_NAME,
    )

    assert loaded is not None
    expected = data.copy()
    expected.attrs.pop("coordinates")
    xr.testing.assert_equal(
        loaded[_ITOOL_DATA_NAME],
        expected,
    )
    assert loaded.coords["temperature"].item() == 20.0
    with h5py.File(fname, "r") as h5_file:
        saved_data = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
        coordinates = saved_data.attrs["coordinates"]
    if isinstance(coordinates, bytes):
        coordinates = coordinates.decode()
    assert coordinates == "temperature"


def test_workspace_writer_encodes_saved_tool_spaced_associated_coord(
    tmp_path,
) -> None:
    data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={
            "x": np.arange(2.0),
            "y": np.arange(3.0),
            "Fake Motor": ("x", np.linspace(10.0, 20.0, 2)),
        },
        name=data_name,
    )
    fname = tmp_path / "saved-tool-spaced-coord.itws"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        workspace_arrays._write_workspace_dataset_group_to_file(
            fname, "0/tool", data.to_dataset()
        )

    assert not any("space in its name" in str(item.message) for item in caught)
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "0/tool", preferred_data_name=data_name
    )

    assert loaded is not None
    xr.testing.assert_equal(
        loaded[data_name].coords["Fake Motor"], data.coords["Fake Motor"]
    )


def test_workspace_h5py_fast_path_roundtrips_saved_tool_extra_blob(
    tmp_path,
) -> None:
    import hdf5plugin

    data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    primary = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={
            "x": np.arange(2.0),
            "y": np.arange(3.0),
            "temperature": ("x", np.linspace(100.0, 200.0, 2)),
            "Fake Motor": ("x", np.linspace(10.0, 20.0, 2)),
        },
        name="primary",
    )
    secondary = xr.DataArray(
        np.arange(200_000.0),
        dims=("z",),
        coords={"z": np.linspace(0.0, 1.0, 200_000)},
        name=None,
    )
    ds = primary.to_dataset(name=data_name)
    ds["data_1"] = erlab.interactive.utils._tool_data_to_blob(secondary, "data_1")
    fname = tmp_path / "saved-tool-extra-blob.itws"

    workspace_arrays._write_workspace_dataset_group_to_file(fname, "0/tool", ds)
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/tool",
        preferred_data_name=data_name,
    )

    assert loaded is not None
    assert workspace_arrays._workspace_dataset_can_write_h5py(
        imagetool_serialization.encode_private_coords(ds, data_name)
    )
    with h5py.File(fname, "r") as h5_file:
        assert hdf5plugin.Blosc2.filter_id in _hdf5_filter_ids(h5_file["0/tool/data_1"])
        assert _hdf5_blosc2_level_codec(h5_file["0/tool/data_1"]) == (1, 5)
    xr.testing.assert_identical(loaded[data_name], primary.rename(data_name))
    xr.testing.assert_equal(
        loaded[data_name].coords["Fake Motor"], primary.coords["Fake Motor"]
    )
    restored_secondary = erlab.interactive.utils._tool_data_from_blob(loaded["data_1"])
    xr.testing.assert_equal(restored_secondary, secondary)

    old_value = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "blosclz3"
        blosclz_fname = tmp_path / "blosclz-saved-tool-extra-blob.itws"
        workspace_arrays._write_workspace_dataset_group_to_file(
            blosclz_fname, "0/tool", ds
        )

        erlab.interactive.options["io/workspace/compression"] = "none"
        uncompressed_fname = tmp_path / "uncompressed-saved-tool-extra-blob.itws"
        workspace_arrays._write_workspace_dataset_group_to_file(
            uncompressed_fname, "0/tool", ds
        )
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_value
    with h5py.File(blosclz_fname, "r") as h5_file:
        assert _hdf5_blosc2_level_codec(h5_file["0/tool/data_1"]) == (3, 0)
    with h5py.File(uncompressed_fname, "r") as h5_file:
        assert hdf5plugin.Blosc2.filter_id not in _hdf5_filter_ids(
            h5_file["0/tool/data_1"]
        )


def test_workspace_h5py_fast_path_roundtrips_saved_tool_references(tmp_path) -> None:
    data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    ds = xr.Dataset(
        {
            data_name: erlab.interactive.utils._tool_data_placeholder(),
            "data_1": erlab.interactive.utils._tool_data_placeholder(),
        },
        attrs={
            erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                {
                    data_name: {"kind": "manager_node", "node_uid": "uid-0"},
                    "data_1": {"kind": "manager_node", "node_uid": "uid-1"},
                }
            )
        },
    )
    fname = tmp_path / "saved-tool-references.itws"

    assert workspace_arrays._write_workspace_dataset_group_h5py(fname, "0/tool", ds)
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/tool",
        preferred_data_name=data_name,
    )

    assert loaded is not None
    assert set(loaded.data_vars) == {data_name, "data_1"}
    reference_dim = erlab.interactive.utils._SAVED_TOOL_DATA_REFERENCE_DIM
    assert loaded[data_name].dims == (reference_dim,)
    assert loaded["data_1"].dims == (reference_dim,)
    assert json.loads(
        loaded.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR]
    ) == json.loads(ds.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR])


def test_workspace_h5py_fast_path_roundtrips_associated_coords_and_xarray(
    tmp_path,
) -> None:

    data_name = _ITOOL_DATA_NAME
    base = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(3.0)},
    )

    divided = base.assign_coords(mesh_current=("x", [1.0, 2.0]))
    divided = divided / divided.coords["mesh_current"]
    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path, "divide-by-coord", divided
    )
    assert loaded.coords["mesh_current"].dims == ("x",)
    np.testing.assert_allclose(loaded.coords["mesh_current"], [1.0, 2.0])

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "two-dimensional-associated-coord",
        base.assign_coords(
            detector_norm=(("x", "y"), np.arange(6.0).reshape(2, 3) + 1.0)
        ),
    )
    assert loaded.coords["detector_norm"].dims == ("x", "y")

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "unicode-scalar-coord",
        base.assign_coords(label="sample"),
    )
    assert loaded.coords["label"].item() == "sample"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "unicode-associated-coord",
        base.assign_coords(label=("x", np.array(["left", "right"]))),
    )
    assert loaded.coords["label"].dtype.kind == "U"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "bytes-associated-coord",
        base.assign_coords(raw=("x", np.array([b"a", b"bb"], dtype="S2"))),
    )
    assert loaded.coords["raw"].dtype.kind == "S"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "datetime-associated-coord",
        xr.DataArray(
            np.arange(2.0),
            dims=("time",),
            coords={
                "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
                "event_time": (
                    "time",
                    np.array(["2024-02-01", "2024-02-02"], dtype="datetime64[D]"),
                ),
            },
        ),
    )
    assert loaded.coords["time"].dtype.kind == "M"
    assert loaded.coords["event_time"].dtype.kind == "M"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "timedelta-associated-coord",
        xr.DataArray(
            np.arange(2.0),
            dims=("delay",),
            coords={
                "delay": np.array([0, 5], dtype="timedelta64[ms]"),
                "exposure": (
                    "delay",
                    np.array([1, 2], dtype="timedelta64[s]"),
                ),
            },
        ),
    )
    assert loaded.coords["delay"].dtype == np.dtype("timedelta64[ms]")
    assert loaded.coords["exposure"].dtype == np.dtype("timedelta64[s]")

    with h5py.File(_fname, "r") as h5_file:
        coordinates = h5_file["0/imagetool"][data_name].attrs["coordinates"]
    if isinstance(coordinates, bytes):
        coordinates = coordinates.decode()
    assert coordinates == "exposure"


def test_workspace_h5py_fast_path_keeps_numeric_since_units(tmp_path) -> None:
    data_name = _ITOOL_DATA_NAME
    fname = tmp_path / "numeric-since-units.itws"
    data = xr.DataArray(
        [1.0, 2.0],
        dims=("x",),
        coords={
            "x": [0.0, 1.0],
            "elapsed": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={"units": "seconds since start"},
            ),
        },
        name=data_name,
    )
    ds = data.to_dataset()

    assert workspace_arrays._workspace_dataset_can_write_h5py(ds)
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=data_name,
    )

    assert loaded is not None
    xr.testing.assert_equal(loaded[data_name], data)
    assert loaded.coords["elapsed"].attrs["units"] == "seconds since start"


def test_workspace_writer_roundtrips_non_native_attr_values_from_fast_path(
    tmp_path,
) -> None:

    data_name = _ITOOL_DATA_NAME
    fname = tmp_path / "rich-attrs-fast-path.itws"
    rich_attr = _rich_workspace_attr_value()
    data = xr.DataArray(
        np.arange(2.0),
        dims=("x",),
        coords={
            "x": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={"axis_config": rich_attr},
            ),
            "temperature": xr.DataArray(
                [20.0, 21.0],
                dims=("x",),
                attrs={"scan_config": rich_attr},
            ),
        },
        attrs={"Single Motor Scan": rich_attr},
        name=data_name,
    )
    ds = data.to_dataset()
    ds.attrs["dataset_config"] = rich_attr

    workspace_arrays._write_workspace_dataset_group_to_file(fname, "0/imagetool", ds)

    assert ds.attrs["dataset_config"] is rich_attr
    assert ds[data_name].attrs["Single Motor Scan"] is rich_attr
    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        saved_data = group[data_name]
        assert "dataset_config" not in group.attrs
        assert "Single Motor Scan" not in saved_data.attrs
        assert workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR in group.attrs
        assert workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR in saved_data.attrs

    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "0/imagetool", preferred_data_name=data_name
    )
    assert loaded is not None
    _assert_rich_workspace_attr(loaded.attrs["dataset_config"])
    _assert_rich_workspace_attr(loaded[data_name].attrs["Single Motor Scan"])
    _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])
    _assert_rich_workspace_attr(loaded.coords["temperature"].attrs["scan_config"])

    opened = workspace_arrays.open_workspace_dataset(fname, "0/imagetool", chunks=None)
    try:
        restored = workspace_format._restore_workspace_dataset_attrs(opened.load())
    finally:
        opened.close()
    _assert_rich_workspace_attr(restored.attrs["dataset_config"])
    _assert_rich_workspace_attr(restored[data_name].attrs["Single Motor Scan"])
    _assert_rich_workspace_attr(restored.coords["x"].attrs["axis_config"])


def test_workspace_writer_drops_invalid_attr_names_from_fast_path(tmp_path) -> None:

    data_name = _ITOOL_DATA_NAME
    fname = tmp_path / "invalid-attrs-fast-path.itws"
    data = xr.DataArray(
        np.arange(2.0),
        dims=("x",),
        coords={
            "x": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={"": "dropped", "axis_note": ""},
            ),
            "temperature": xr.DataArray(
                [20.0, 21.0],
                dims=("x",),
                attrs={None: "dropped", "units": "K"},
            ),
        },
        attrs={"": "dropped", 1: "dropped", "note": ""},
        name=data_name,
    )
    ds = data.to_dataset()
    ds.attrs[""] = "dropped"
    ds.attrs["dataset_note"] = ""

    workspace_arrays._write_workspace_dataset_group_to_file(fname, "0/imagetool", ds)

    assert "" in ds.attrs
    assert "" in ds[data_name].attrs
    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        saved_data = group[data_name]

        assert "" not in list(group.attrs)
        assert group.attrs["dataset_note"] == ""
        assert "" not in list(saved_data.attrs)
        assert saved_data.attrs["note"] == ""
        assert "" not in list(group["x"].attrs)
        assert group["x"].attrs["axis_note"] == ""
        assert "" not in list(group["temperature"].attrs)
        assert group["temperature"].attrs["units"] == "K"

    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "0/imagetool", preferred_data_name=data_name
    )
    assert loaded is not None
    assert "" not in loaded.attrs
    assert loaded.attrs["dataset_note"] == ""
    assert "" not in loaded[data_name].attrs
    assert loaded[data_name].attrs["note"] == ""
    assert loaded.coords["temperature"].attrs["units"] == "K"


def test_workspace_writer_drops_invalid_attr_names_from_fallback(tmp_path) -> None:
    fname = tmp_path / "invalid-attrs-fallback.itws"
    rich_attr = _rich_workspace_attr_value()
    ds = xr.Dataset(
        {
            "left": xr.DataArray(
                [1.0, 2.0],
                dims=("x",),
                attrs={
                    "": "dropped",
                    "left_note": "",
                    "Single Motor Scan": rich_attr,
                },
            ),
            "right": ("x", [3.0, 4.0]),
        },
        coords={
            "x": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={None: "dropped", "axis_note": "", "axis_config": rich_attr},
            )
        },
        attrs={"": "dropped", "dataset_note": "", "dataset_config": rich_attr},
    )

    workspace_arrays._write_workspace_dataset_group_to_file(fname, "0/tool", ds)

    opened = xr.open_dataset(fname, group="/0/tool", engine="h5netcdf")
    try:
        loaded = workspace_format._restore_workspace_dataset_attrs(opened.load())
    finally:
        opened.close()

    assert "" in ds.attrs
    assert "" in ds["left"].attrs
    assert "" not in loaded.attrs
    assert loaded.attrs["dataset_note"] == ""
    assert "" not in loaded["left"].attrs
    assert loaded["left"].attrs["left_note"] == ""
    _assert_rich_workspace_attr(loaded["left"].attrs["Single Motor Scan"])
    assert "" not in loaded.coords["x"].attrs
    assert loaded.coords["x"].attrs["axis_note"] == ""
    _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])
    _assert_rich_workspace_attr(loaded.attrs["dataset_config"])


def test_workspace_h5py_fast_path_rejects_invalid_payloads(
    caplog, monkeypatch, tmp_path
) -> None:
    data_name = _ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR

    assert not workspace_arrays._workspace_dataset_can_write_h5py(
        xr.Dataset(
            {
                data_name: ("x", [1.0]),
                "extra": ("x", [2.0]),
            },
            coords={"x": [0.0]},
        )
    )

    missing_private = xr.Dataset({data_name: ("x", [1.0])}, coords={"x": [0.0]})
    missing_private[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "missing", "dims": ["x"]}]
    )
    assert not workspace_arrays._workspace_dataset_can_write_h5py(missing_private)

    bad_private_dims = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "private": ("z", [2.0]),
        },
        coords={"x": [0.0], "z": [0.0]},
    )
    bad_private_dims[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["z"]}]
    )
    assert not workspace_arrays._workspace_dataset_can_write_h5py(bad_private_dims)

    assert not workspace_arrays._workspace_dataset_can_write_h5py(
        xr.Dataset(
            {data_name: ("x", [1.0])},
            coords={"x": np.array([object()], dtype=object)},
        )
    )

    bad_associated_dims = xr.Dataset(
        {data_name: ("x", [1.0])},
        coords={"x": [0.0], "z": [0.0], "bad": ("z", [1.0])},
    )
    assert not workspace_arrays._workspace_dataset_can_write_h5py(bad_associated_dims)

    import dask.array as da

    chunked_coord = xr.Dataset(
        {data_name: ("x", [1.0, 2.0])},
        coords={
            "x": [0.0, 1.0],
            "chunked": ("x", da.from_array(np.array([1.0, 2.0]), chunks=(1,))),
        },
    )
    assert not workspace_arrays._workspace_dataset_can_write_h5py(chunked_coord)

    monkeypatch.setattr(
        workspace_arrays, "_workspace_dataset_can_write_h5py", lambda _ds: True
    )
    assert not workspace_arrays._write_workspace_dataset_group_h5py(
        tmp_path / "no-data-name.itws", "0/imagetool", xr.Dataset()
    )

    bad_attrs = xr.Dataset({data_name: ("x", [1.0])}, coords={"x": [0.0]})
    bad_attrs.attrs["bad"] = object()
    fname = tmp_path / "bad-attrs.itws"
    with caplog.at_level(logging.WARNING, logger=workspace_format.logger.name):
        assert workspace_arrays._write_workspace_dataset_group_h5py(
            fname, "0/imagetool", bad_attrs
        )
    assert "unsupported value type object" in caplog.text

    with h5py.File(fname, "r") as h5_file:
        assert "0/imagetool" in h5_file
        assert "bad" not in h5_file["0/imagetool"].attrs


def test_workspace_h5py_reader_rejects_malformed_groups(tmp_path) -> None:

    data_name = _ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "malformed-reader.itws"

    with h5py.File(fname, "w") as h5_file:
        h5_file.create_dataset("not-a-group", data=np.arange(2.0))
        multi = h5_file.create_group("multi")
        multi.create_dataset("a", data=np.arange(2.0))
        multi.create_dataset("b", data=np.arange(2.0))
        no_dims = h5_file.create_group("no-dims")
        no_dims.create_dataset(data_name, data=np.arange(2.0))
        bad_scale = h5_file.create_group("bad-scale")
        scale = bad_scale.create_dataset("x", data=np.arange(4.0).reshape(2, 2))
        scale.make_scale("x")
        bad_data = bad_scale.create_dataset(data_name, data=np.arange(2.0))
        bad_data.dims[0].attach_scale(scale)
        missing_scalar = h5_file.create_group("missing-scalar")
        x = missing_scalar.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        missing_data = missing_scalar.create_dataset(data_name, data=np.arange(2.0))
        missing_data.dims[0].attach_scale(x)
        missing_data.attrs["coordinates"] = np.bytes_("missing")
        missing_private = h5_file.create_group("missing-private")
        x = missing_private.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        private_data = missing_private.create_dataset(data_name, data=np.arange(2.0))
        private_data.dims[0].attach_scale(x)
        private_data.attrs[private_attr] = json.dumps(
            [{"coord_name": "Fake Motor", "variable_name": "missing", "dims": ["x"]}]
        )
        bad_private = h5_file.create_group("bad-private")
        x = bad_private.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        private_data = bad_private.create_dataset(data_name, data=np.arange(2.0))
        private_data.dims[0].attach_scale(x)
        bad_coord = bad_private.create_dataset("private", data=np.arange(2.0))
        bad_coord.dims[0].attach_scale(x)
        private_data.attrs[private_attr] = json.dumps(
            [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["z"]}]
        )
        bad_associated_no_scale = h5_file.create_group("bad-associated-no-scale")
        x = bad_associated_no_scale.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_associated_no_scale.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        bad_associated_no_scale.create_dataset("associated", data=np.arange(2.0))
        bad_associated_length = h5_file.create_group("bad-associated-length")
        x = bad_associated_length.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_associated_length.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        associated = bad_associated_length.create_dataset(
            "associated", data=np.arange(3.0)
        )
        associated.dims[0].attach_scale(x)
        bad_associated_foreign_dim = h5_file.create_group("bad-associated-foreign-dim")
        x = bad_associated_foreign_dim.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        z = bad_associated_foreign_dim.create_dataset("z", data=np.arange(2.0))
        z.make_scale("z")
        data = bad_associated_foreign_dim.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        associated = bad_associated_foreign_dim.create_dataset(
            "associated", data=np.arange(2.0)
        )
        associated.dims[0].attach_scale(z)
        bad_time = h5_file.create_group("bad-time-metadata")
        x = bad_time.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_time.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "time"
        time = bad_time.create_dataset("time", data=np.arange(2, dtype=np.int64))
        time.dims[0].attach_scale(x)
        time.attrs["units"] = "days since not-a-date"
        time.attrs["calendar"] = "proleptic_gregorian"

    assert workspace_arrays._read_workspace_dataset_group_h5py(fname, "missing") is None
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(fname, "not-a-group")
        is None
    )
    assert workspace_arrays._read_workspace_dataset_group_h5py(fname, "multi") is None
    assert workspace_arrays._read_workspace_dataset_group_h5py(fname, "no-dims") is None
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(fname, "bad-scale") is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(fname, "missing-scalar")
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "missing-private", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-private", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-associated-no-scale", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-associated-length", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-associated-foreign-dim", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-time-metadata", preferred_data_name=data_name
        )
        is None
    )


def test_workspace_h5py_reader_restores_legacy_spaced_coords(tmp_path) -> None:

    data_name = _ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "legacy-spaced-coord.itws"

    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("valid")
        x = group.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = group.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "missing"
        fake = group.create_dataset("Fake Motor", data=np.arange(2.0) + 10.0)
        fake.dims[0].attach_scale(x)
        duplicate = h5_file.create_group("duplicate")
        x = duplicate.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = duplicate.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs[private_attr] = json.dumps(
            [
                {
                    "coord_name": "Fake Motor",
                    "variable_name": "Fake Motor",
                    "dims": ["x"],
                }
            ]
        )
        fake = duplicate.create_dataset("Fake Motor", data=np.arange(2.0) + 20.0)
        fake.dims[0].attach_scale(x)
        invalid = h5_file.create_group("invalid")
        x = invalid.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = invalid.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        invalid.create_dataset("Fake Motor", data=np.arange(2.0) + 30.0)

    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "valid", preferred_data_name=data_name
    )
    assert loaded is not None
    np.testing.assert_allclose(loaded.coords["Fake Motor"].values, [10.0, 11.0])
    duplicate_loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "duplicate", preferred_data_name=data_name
    )
    assert duplicate_loaded is not None
    np.testing.assert_allclose(
        duplicate_loaded.coords["Fake Motor"].values, [20.0, 21.0]
    )
    invalid_loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "invalid", preferred_data_name=data_name
    )
    assert invalid_loaded is not None
    assert "Fake Motor" not in invalid_loaded


def test_workspace_h5py_writer_replaces_groups_and_preserves_attrs(tmp_path) -> None:

    data_name = _ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "writer-attrs.itws"
    ds = xr.Dataset(
        {
            data_name: (
                ("x", "y"),
                np.arange(4.0).reshape(2, 2),
                {"coordinates": "legacy"},
            ),
            "private": (("x",), np.arange(2.0), {"private_attr": "kept"}),
        },
        coords={
            "x": ("x", np.arange(2.0), {"axis_attr": "x"}),
            "y": ("y", np.arange(2.0), {"axis_attr": "y"}),
            "temperature": ((), 20.0, {"units": "K"}),
        },
    )
    ds[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["x"]}]
    )

    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )

    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        assert group["x"].attrs["axis_attr"] == "x"
        assert group["temperature"].attrs["units"] == "K"
        assert group["private"].attrs["private_attr"] == "kept"
        coordinates = group[data_name].attrs["coordinates"]
        if isinstance(coordinates, bytes):
            coordinates = coordinates.decode()
        assert coordinates == "legacy temperature"
