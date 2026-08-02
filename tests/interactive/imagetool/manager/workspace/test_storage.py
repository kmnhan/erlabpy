import errno
import gc
import json
import pathlib
import threading
import types
import typing
import weakref
from collections.abc import Callable

import h5py
import hdf5plugin
import numpy as np
import psutil
import pytest
import xarray
import xarray as xr
from qtpy import QtWidgets

import erlab
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from tests.interactive.imagetool.manager.workspace._support import (
    _assert_no_workspace_internal_groups,
    _assert_rich_workspace_attr,
    _hdf5_filter_ids,
    _rich_workspace_attr_value,
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


def _wait_for_workspace_cleanup(condition: Callable[[], bool]) -> bool:
    for _ in range(100):
        gc.collect()
        if condition():
            return True
        threading.Event().wait(0.01)
    return condition()


def test_workspace_metadata_failure_leaves_published_generation_unchanged(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "partial-update.itws"
    _write_transaction_test_workspace(fname)
    manager = workspace_arrays.WorkspaceFileManager(fname)
    export_before = pathlib.Path(manager.__getstate__()[1])
    original_write = workspace_storage._write_root_attrs_to_open_workspace_file

    def _write_then_fail(h5_file, attrs, *, replace=False) -> None:
        original_write(h5_file, attrs, replace=replace)
        raise RuntimeError("update failed")

    monkeypatch.setattr(
        workspace_storage,
        "_write_root_attrs_to_open_workspace_file",
        _write_then_fail,
    )

    try:
        with pytest.raises(RuntimeError, match="update failed"):
            workspace_storage._write_workspace_root_attrs_to_file(
                fname, {"partial_update": True}
            )

        export_after = pathlib.Path(manager.__getstate__()[1])
        with h5py.File(export_before, "r") as h5_file:
            assert "partial_update" not in h5_file.attrs
        with h5py.File(export_after, "r") as h5_file:
            assert "partial_update" not in h5_file.attrs
        with h5py.File(fname, "r") as h5_file:
            assert "partial_update" not in h5_file.attrs
    finally:
        manager.close()


def test_full_workspace_save_does_not_overwrite_newer_generation(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "workspace.itws"
    external = tmp_path / "external.itws"
    _write_transaction_test_workspace(fname, value=1.0)
    _write_transaction_test_workspace(external, value=7.0)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="save snapshot")}
    )
    original_validate = workspace_storage._validate_workspace_h5_file
    replaced = False

    def _validate_then_replace(path) -> None:
        nonlocal replaced
        original_validate(path)
        if not replaced:
            replaced = True
            external.replace(fname)

    monkeypatch.setattr(
        workspace_storage, "_validate_workspace_h5_file", _validate_then_replace
    )
    try:
        with pytest.raises(RuntimeError, match="changed while its successor"):
            workspace_storage._write_full_workspace_tree_file(
                fname, tree, _transaction_test_root_attrs()
            )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 7.0
    assert not list(tmp_path.glob("workspace.itws.tmp-*"))


def test_workspace_file_repack_payload_strips_delta_and_skips_internal_groups(
    tmp_path,
) -> None:

    fname = tmp_path / "file-repack.itws"
    _write_transaction_test_workspace(fname)
    workspace_storage._write_workspace_root_attrs_to_file(
        fname,
        _transaction_test_root_attrs(delta_save_count=3),
        replace=True,
    )
    with h5py.File(fname, "a") as h5_file:
        h5_file.create_group("__itws_pending_orphan")
        h5_file.create_dataset("root_dataset", data=np.arange(3))

    assert workspace_arrays._workspace_live_root_group_copy_groups(fname) == (
        ("0", "0", None),
    )
    storage_size, existing_count = workspace_arrays._workspace_h5_paths_storage_size(
        fname, ("0", "missing")
    )
    assert storage_size >= np.dtype(np.float64).itemsize
    assert existing_count == 1
    assert workspace_arrays._workspace_live_h5_storage_size(fname) == storage_size
    assert workspace_storage._workspace_obsolete_estimate(fname) >= 0
    root_attrs, copy_groups = workspace_storage._workspace_file_repack_payload(fname)

    manifest = workspace_format._workspace_manifest_from_attrs(root_attrs)
    assert "delta_save_count" not in manifest
    assert "transaction_protocol" not in manifest
    assert (
        manifest["schema_version"]
        == workspace_format._current_workspace_schema_version()
    )
    assert manifest["erlab_version"] == erlab.__version__
    assert copy_groups == (("0", "0", None),)

    workspace_storage._write_full_workspace_tree_file(
        fname,
        None,
        root_attrs,
        copy_source=fname,
        copy_groups=copy_groups,
    )

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)
    with h5py.File(fname, "r") as h5_file:
        assert set(h5_file) == {"0"}
        manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert "delta_save_count" not in manifest


def test_write_full_workspace_tree_file_compresses_payload_not_coords(
    tmp_path,
) -> None:

    ds = xr.Dataset(
        {
            "data": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            ),
            "small": ("x", np.arange(512, dtype=np.int64)),
        },
        coords={
            "x": np.linspace(-1.0, 1.0, 512),
            "y": np.linspace(-2.0, 2.0, 512),
        },
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    fname = tmp_path / "compressed.itws"
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, {"imagetool_workspace_schema_version": 4}
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        assert hdf5plugin.Blosc2.filter_id in _hdf5_filter_ids(
            h5_file["0/imagetool/data"]
        )
        assert _hdf5_filter_ids(h5_file["0/imagetool/x"]) == []
        assert _hdf5_filter_ids(h5_file["0/imagetool/y"]) == []
        assert _hdf5_filter_ids(h5_file["0/imagetool/small"]) == []

    opened = workspace_arrays.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
        xarray.testing.assert_equal(loaded["x"], ds["x"])
        xarray.testing.assert_equal(loaded["y"], ds["y"])
    finally:
        opened.close()


def test_prepare_workspace_transaction_promotes_missing_attr_fallback(
    tmp_path,
) -> None:
    fname = tmp_path / "fallback.itws"
    _write_transaction_test_workspace(fname)
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )
    rewrite_map: dict[str, tuple[str, dict[str, xr.Dataset]]] = {}

    group_operations, attr_updates = workspace_storage._prepare_workspace_transaction(
        fname,
        f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}fallback",
        f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}fallback",
        f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}fallback",
        rewrite_map,
        (("0/missing", {"itool_title": "new"}, fallback),),
        _transaction_test_root_attrs(delta_save_count=1),
    )

    assert rewrite_map == {"0": fallback}
    assert attr_updates == []
    assert group_operations[0]["group_path"] == "0"
    workspace_storage._recover_workspace_transactions(fname)
    _assert_no_workspace_internal_groups(fname)


def test_write_full_workspace_tree_file_skips_missing_copy_source_group(
    tmp_path,
) -> None:
    fname = tmp_path / "missing-copy-group.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(3.0, title="rewritten")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("missing/source", "0/imagetool", None),),
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 3.0


def test_write_full_workspace_tree_file_reports_missing_backing_source(
    tmp_path,
) -> None:
    fname = tmp_path / "missing-backing-source-target.itws"
    missing_source = tmp_path / "deleted-source.itws"
    _write_transaction_test_workspace(fname)
    original_contents = fname.read_bytes()

    with pytest.raises(
        workspace_storage._WorkspaceBackingFileNotFoundError
    ) as exc_info:
        workspace_storage._write_full_workspace_tree_file(
            fname,
            None,
            _transaction_test_root_attrs(),
            copy_group_sources=(
                (
                    str(missing_source),
                    "0/imagetool",
                    "0/imagetool",
                    None,
                ),
            ),
        )

    assert exc_info.value.source_path == str(missing_source)
    assert fname.read_bytes() == original_contents


def test_write_full_workspace_tree_file_replaces_stale_root_attrs(tmp_path) -> None:

    fname = tmp_path / "root-attrs.itws"
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(1.0, title="old")}
    )
    tree.attrs["stale_workspace_attr"] = "remove me"
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        assert "stale_workspace_attr" not in h5_file.attrs
        manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest == {"schema_version": 4, "root_order": [0], "nodes": []}


def test_write_full_workspace_tree_file_preserves_cached_workspace_reader(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "cached-reader.itws"
    _write_transaction_test_workspace(fname)
    manager = workspace_arrays.WorkspaceFileManager(fname)
    generation = manager._generation
    cached_file = manager.acquire()
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    original_link = workspace_storage.os.link
    original_replace = workspace_storage.os.replace
    preservation_checked = False
    replacement_checked = False

    def _link_after_reader_close(source, destination):
        nonlocal preservation_checked
        preservation_checked = True
        assert cached_file._closed
        original_link(source, destination)

    def _replace_after_reader_close(source, destination):
        nonlocal replacement_checked
        if pathlib.Path(destination) == fname:
            replacement_checked = True
            assert cached_file._closed
        original_replace(source, destination)

    monkeypatch.setattr(workspace_storage.os, "link", _link_after_reader_close)
    monkeypatch.setattr(workspace_storage.os, "replace", _replace_after_reader_close)
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
        assert preservation_checked
        assert replacement_checked
        assert generation.retired
        assert pathlib.Path(generation.path).is_file()
        assert pathlib.Path(generation.path) != fname
        assert pathlib.Path(
            generation.path
        ).parent == workspace_arrays._workspace_reader_directory(fname)
        preserved_file = manager.acquire()
        preserved_value = (
            preserved_file.groups["0"].groups["imagetool"].variables["data"][()]
        )
        assert np.asarray(preserved_value).item() == 1.0
        reopened_manager = workspace_arrays.WorkspaceFileManager(fname)
        assert reopened_manager._generation is not generation
        reopened_file = reopened_manager.acquire()
        value = reopened_file.groups["0"].groups["imagetool"].variables["data"][()]
        assert np.asarray(value).item() == 2.0
    finally:
        tree.close()
        manager.close()
        if "reopened_manager" in locals():
            reopened_manager.close()


def test_replace_workspace_file_defers_finalizer_cleanup_until_after_publish(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            source, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    managers = [workspace_arrays.WorkspaceFileManager(destination)]
    generation = managers[0]._generation
    managers[0].acquire()
    original_link = workspace_storage.os.link

    def _release_last_reader_during_preservation(link_source, link_destination):
        managers.clear()
        gc.collect()
        assert not generation.has_managers()
        assert generation._finalizer.alive
        original_link(link_source, link_destination)

    monkeypatch.setattr(
        workspace_storage.os, "link", _release_last_reader_during_preservation
    )

    workspace_storage._replace_workspace_file(source, destination)

    generation_path = pathlib.Path(generation.path)
    assert generation.retired
    assert _wait_for_workspace_cleanup(lambda: not generation_path.exists())
    assert not generation._finalizer.alive
    assert _read_transaction_test_value(destination) == 2.0


def test_preserve_workspace_generation_hard_links_on_workspace_filesystem(
    tmp_path,
) -> None:
    source = tmp_path / "workspace.itws"
    source.write_bytes(b"workspace generation")

    preserved = workspace_storage._preserve_workspace_file_generation(
        source, workspace_arrays._workspace_file_identity(source)
    )
    preserved_path = pathlib.Path(preserved.path)
    try:
        assert preserved_path.parent == workspace_arrays._workspace_reader_directory(
            source
        )
        assert preserved_path.samefile(source)
    finally:
        preserved.cleanup()

    assert not preserved_path.exists()


def test_preserve_workspace_generation_rejects_source_replacement(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "workspace.itws"
    replacement = tmp_path / "replacement.itws"
    source.write_bytes(b"old generation")
    replacement.write_bytes(b"new generation")
    expected_identity = workspace_arrays._workspace_file_identity(source)
    wrong_identity = (
        expected_identity[0],
        expected_identity[1],
        expected_identity[2] + 1,
        expected_identity[3],
    )
    with pytest.raises(RuntimeError, match="changed before it could be preserved"):
        workspace_storage._preserve_workspace_file_generation(source, wrong_identity)

    original_link = workspace_arrays.os.link
    original_replace = workspace_arrays.os.replace

    def _replace_before_link(link_source, link_destination) -> None:
        original_replace(replacement, source)
        original_link(link_source, link_destination)

    monkeypatch.setattr(workspace_arrays.os, "link", _replace_before_link)

    with pytest.raises(RuntimeError, match="changed while it was being preserved"):
        workspace_storage._preserve_workspace_file_generation(source, expected_identity)

    assert source.read_bytes() == b"new generation"
    assert not list(
        workspace_arrays._workspace_reader_directory(source).glob("reader-*.itws")
    )


def test_preserve_workspace_generation_cleans_reader_after_identity_error(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "workspace.itws"
    source.write_bytes(b"workspace generation")
    expected_identity = workspace_arrays._workspace_file_identity(source)
    original_identity = workspace_arrays._workspace_file_identity

    def _fail_source_identity(path):
        if pathlib.Path(path).resolve() == source.resolve():
            raise PermissionError("identity denied")
        return original_identity(path)

    monkeypatch.setattr(
        workspace_arrays, "_workspace_file_identity", _fail_source_identity
    )

    with pytest.raises(PermissionError, match="identity denied"):
        workspace_storage._preserve_workspace_file_generation(source, expected_identity)

    assert not list(
        workspace_arrays._workspace_reader_directory(source).glob("reader-*.itws")
    )


def test_write_full_workspace_tree_file_preserves_stale_lazy_dataarray(
    tmp_path,
) -> None:
    fname = tmp_path / "lazy-reader.itws"
    _write_transaction_test_workspace(fname)
    opened = workspace_arrays.open_workspace_dataset(fname, "0/imagetool", chunks={})
    lazy_data = opened["data"].copy(deep=False)
    opened.close()
    assert lazy_data.compute().item() == 1.0
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
        assert lazy_data.compute().item() == 1.0
        reopened = workspace_arrays.open_workspace_dataset(
            fname, "0/imagetool", chunks={}
        )
        try:
            assert reopened["data"].compute().item() == 2.0
        finally:
            reopened.close()
    finally:
        tree.close()
        del lazy_data
        gc.collect()


def test_replace_workspace_file_keeps_reader_generations_isolated(tmp_path) -> None:
    destination = tmp_path / "current.itws"

    def _write_generation(path: pathlib.Path, value: float) -> None:
        tree = xr.DataTree.from_dict(
            {"0/imagetool": _transaction_test_dataset(value, title=str(value))}
        )
        try:
            workspace_storage._write_full_workspace_tree_file(
                path, tree, _transaction_test_root_attrs()
            )
        finally:
            tree.close()

    def _read_value(manager: workspace_arrays.WorkspaceFileManager) -> float:
        h5_file = manager.acquire()
        value = h5_file.groups["0"].groups["imagetool"].variables["data"][()]
        return float(np.asarray(value).item())

    _write_generation(destination, 1.0)
    first_manager = workspace_arrays.WorkspaceFileManager(destination)
    assert _read_value(first_manager) == 1.0

    second_source = tmp_path / "second.itws"
    _write_generation(second_source, 2.0)
    workspace_storage._replace_workspace_file(second_source, destination)
    second_manager = workspace_arrays.WorkspaceFileManager(destination)
    assert _read_value(first_manager) == 1.0
    assert _read_value(second_manager) == 2.0

    third_source = tmp_path / "third.itws"
    _write_generation(third_source, 3.0)
    workspace_storage._replace_workspace_file(third_source, destination)
    current_manager = workspace_arrays.WorkspaceFileManager(destination)
    try:
        assert _read_value(first_manager) == 1.0
        assert _read_value(second_manager) == 2.0
        assert _read_value(current_manager) == 3.0
    finally:
        first_manager.close()
        second_manager.close()
        current_manager.close()


def test_replace_workspace_file_activates_manager_for_new_destination(tmp_path) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(source)
    manager = workspace_arrays.WorkspaceFileManager(destination)
    generation = manager._generation

    try:
        workspace_storage._replace_workspace_file(source, destination)

        assert not generation.retired
        assert generation.path == str(destination.resolve())
        with workspace_arrays._workspace_file_lock(destination):
            assert (
                workspace_arrays._current_workspace_file_generation(destination)
                is generation
            )
        value = manager.acquire().groups["0"].groups["imagetool"].variables["data"][()]
        assert np.asarray(value).item() == 1.0
    finally:
        manager.close()


def test_replace_workspace_file_copy_fallback_cleans_preserved_generation(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            source, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()
    manager = workspace_arrays.WorkspaceFileManager(destination)
    manager.acquire()

    def _fail_link(_source, _destination):
        raise OSError(errno.EPERM, "hard links unavailable")

    monkeypatch.setattr(workspace_storage.os, "link", _fail_link)
    workspace_storage._replace_workspace_file(source, destination)

    generation = manager._generation
    assert generation.retired
    generation_path = pathlib.Path(generation.path)
    generation_directory = generation_path.parent
    assert generation_path.exists()
    assert generation_directory != tmp_path
    preserved_file = manager.acquire()
    value = preserved_file.groups["0"].groups["imagetool"].variables["data"][()]
    assert np.asarray(value).item() == 1.0

    manager.close()
    del preserved_file, generation, manager
    assert _wait_for_workspace_cleanup(lambda: not generation_path.exists())
    assert not generation_directory.exists()


def test_retired_workspace_generation_lives_until_last_manager_released(
    tmp_path,
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            source, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    manager = workspace_arrays.WorkspaceFileManager(destination)
    manager.acquire()
    workspace_storage._replace_workspace_file(source, destination)
    cached_file = manager.acquire()
    generation = manager._generation
    assert generation.retired
    generation_path = pathlib.Path(generation.path)
    generation_ref = weakref.ref(generation)
    assert generation_path.exists()

    del manager, generation
    assert _wait_for_workspace_cleanup(lambda: generation_ref() is None)
    assert cached_file._closed
    assert not generation_path.exists()


def test_workspace_generation_cleanup_finishes_after_path_lock_released(
    tmp_path,
) -> None:
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)

    first_manager = workspace_arrays.WorkspaceFileManager(destination)
    second_manager = workspace_arrays.WorkspaceFileManager(destination)
    generation = first_manager._generation
    cached_file = second_manager.acquire()
    first_manager_ref = weakref.ref(first_manager)
    second_manager_ref = weakref.ref(second_manager)
    generation_ref = weakref.ref(generation)
    assert second_manager._generation is generation
    lock_held = threading.Event()
    release_lock = threading.Event()
    lock = workspace_arrays._workspace_file_lock(destination)

    def _hold_workspace_lock() -> None:
        with lock:
            lock_held.set()
            release_lock.wait(2)

    lock_thread = threading.Thread(target=_hold_workspace_lock)
    lock_thread.start()
    assert lock_held.wait(2)
    try:
        del first_manager, second_manager, generation
        gc.collect()
        assert first_manager_ref() is None
        assert second_manager_ref() is None
        assert generation_ref() is not None
        assert not cached_file._closed
    finally:
        release_lock.set()
        lock_thread.join(2)
    assert not lock_thread.is_alive()

    assert _wait_for_workspace_cleanup(lambda: generation_ref() is None)
    assert cached_file._closed
    assert _read_transaction_test_value(destination) == 1.0


def test_workspace_generation_cleanup_keeps_a_new_reader(tmp_path) -> None:
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)
    manager = workspace_arrays.WorkspaceFileManager(destination)
    generation = manager._generation
    manager._finalizer.detach()
    assert generation.release_manager()

    reopened_manager = workspace_arrays.WorkspaceFileManager(destination)
    try:
        with workspace_arrays._workspace_file_lock(destination):
            workspace_arrays._cleanup_workspace_file_generation_locked(generation)

        assert reopened_manager._generation is generation
        value = (
            reopened_manager.acquire()
            .groups["0"]
            .groups["imagetool"]
            .variables["data"][()]
        )
        assert np.asarray(value).item() == 1.0
    finally:
        reopened_manager.close()
        del reopened_manager, manager, generation
        gc.collect()


def test_replace_workspace_file_discards_unused_current_generation(tmp_path) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            source, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    manager = workspace_arrays.WorkspaceFileManager(destination)
    generation = manager._generation
    cached_file = manager.acquire()
    manager._finalizer.detach()
    assert generation.release_manager()

    workspace_storage._replace_workspace_file(source, destination)

    assert cached_file._closed
    assert not generation._finalizer.alive
    with workspace_arrays._workspace_file_lock(destination):
        assert workspace_arrays._current_workspace_file_generation(destination) is None
    assert _read_transaction_test_value(destination) == 2.0


def test_replace_workspace_file_blocks_new_readers_during_commit(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    replace_started = threading.Event()
    allow_replace = threading.Event()
    reader_acquired = threading.Event()
    errors: list[BaseException] = []
    original_replace = workspace_storage.os.replace

    def _paused_replace(src, dst):
        replace_started.set()
        if not allow_replace.wait(2):
            raise TimeoutError("Replacement test did not resume")
        original_replace(src, dst)

    def _replace() -> None:
        try:
            workspace_storage._replace_workspace_file(source, destination)
        except BaseException as exc:  # pragma: no cover - reported in main thread
            errors.append(exc)

    def _read() -> None:
        try:
            with workspace_arrays._workspace_file_lock(destination):
                reader_acquired.set()
        except BaseException as exc:  # pragma: no cover - reported in main thread
            errors.append(exc)

    monkeypatch.setattr(workspace_storage.os, "replace", _paused_replace)
    replace_thread = threading.Thread(target=_replace)
    reader_thread = threading.Thread(target=_read)
    replace_thread.start()
    assert replace_started.wait(2)
    reader_thread.start()
    try:
        assert not reader_acquired.wait(0.05)
    finally:
        allow_replace.set()
        replace_thread.join(2)
        reader_thread.join(2)

    assert not replace_thread.is_alive()
    assert not reader_thread.is_alive()
    assert reader_acquired.is_set()
    assert errors == []
    assert destination.read_bytes() == b"new"


def test_replace_workspace_file_preserves_in_flight_multichunk_reader(
    monkeypatch, tmp_path
) -> None:
    from xarray.backends.h5netcdf_ import H5NetCDFArrayWrapper

    fname = tmp_path / "multichunk-reader.itws"

    def _write_values(values: np.ndarray) -> None:
        tree = xr.DataTree.from_dict(
            {"0/imagetool": xr.Dataset({"data": ("x", values)})}
        )
        try:
            workspace_storage._write_full_workspace_tree_file(
                fname, tree, _transaction_test_root_attrs()
            )
        finally:
            tree.close()

    _write_values(np.ones(8, dtype=np.float64))
    opened = workspace_arrays.open_workspace_dataset(
        fname, "0/imagetool", chunks={"x": 1}
    )
    lazy_data = opened["data"]
    first_read = threading.Event()
    continue_read = threading.Event()
    original_getitem = H5NetCDFArrayWrapper._getitem
    read_count = 0

    def _pause_after_first_read(self, key):
        nonlocal read_count
        result = original_getitem(self, key)
        if self.variable_name == "data":
            read_count += 1
            if read_count == 1:
                first_read.set()
                if not continue_read.wait(2):
                    raise TimeoutError("Workspace replacement did not finish")
        return result

    monkeypatch.setattr(H5NetCDFArrayWrapper, "_getitem", _pause_after_first_read)
    results: list[xr.DataArray] = []
    errors: list[BaseException] = []

    def _compute() -> None:
        try:
            results.append(lazy_data.compute(scheduler="single-threaded"))
        except BaseException as exc:  # pragma: no cover - reported in main thread
            errors.append(exc)

    compute_thread = threading.Thread(target=_compute)
    compute_thread.start()
    try:
        assert first_read.wait(2)
        _write_values(np.full(8, 2.0, dtype=np.float64))
    finally:
        continue_read.set()
        compute_thread.join(2)
        opened.close()

    assert not compute_thread.is_alive()
    assert errors == []
    assert len(results) == 1
    np.testing.assert_array_equal(results[0], np.ones(8, dtype=np.float64))

    reopened = workspace_arrays.open_workspace_dataset(fname, "0/imagetool", chunks={})
    try:
        np.testing.assert_array_equal(
            reopened["data"].compute(), np.full(8, 2.0, dtype=np.float64)
        )
    finally:
        reopened.close()


def test_replace_workspace_file_retries_windows_sharing_violation(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    attempts = 0
    delays: list[float] = []
    original_replace = workspace_storage.os.replace

    def _replace_after_transient_failures(src, dst):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(errno.EACCES, "sharing violation")
        original_replace(src, dst)

    monkeypatch.setattr(
        workspace_storage, "_workspace_replace_retry_delays", lambda _exc: (0.0, 0.0)
    )
    monkeypatch.setattr(
        workspace_storage.os, "replace", _replace_after_transient_failures
    )
    monkeypatch.setattr(workspace_storage.time, "sleep", delays.append)

    workspace_storage._replace_workspace_file(source, destination)

    assert attempts == 3
    assert delays == [0.0, 0.0]
    assert destination.read_bytes() == b"new"


def test_replace_workspace_file_rejects_change_during_retry(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    external = tmp_path / "external.itws"
    xr.Dataset({"data": 2}).to_netcdf(source, engine="h5netcdf")
    xr.Dataset({"data": 1}).to_netcdf(destination, engine="h5netcdf")
    xr.Dataset({"data": 3}).to_netcdf(external, engine="h5netcdf")
    reader = workspace_arrays.WorkspaceFileManager(destination)
    assert np.asarray(reader.acquire().variables["data"]).item() == 1
    expected_state = workspace_arrays._workspace_file_state(destination)
    original_replace = workspace_storage.os.replace
    attempts = 0

    def _fail_first_replace(src, dst):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(errno.EACCES, "sharing violation")
        original_replace(src, dst)

    def _replace_during_retry(_delay: float) -> None:
        original_replace(external, destination)

    monkeypatch.setattr(
        workspace_storage, "_workspace_replace_retry_delays", lambda _exc: (0.0,)
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _fail_first_replace)
    monkeypatch.setattr(workspace_storage.time, "sleep", _replace_during_retry)

    try:
        with pytest.raises(RuntimeError, match="changed while replacement was waiting"):
            workspace_storage._replace_workspace_file(
                source,
                destination,
                expected_state=expected_state,
            )

        assert attempts == 1
        with h5py.File(destination, "r") as h5_file:
            assert np.asarray(h5_file["data"]).item() == 3
        with h5py.File(source, "r") as h5_file:
            assert np.asarray(h5_file["data"]).item() == 2
        assert np.asarray(reader.acquire().variables["data"]).item() == 1
    finally:
        reader.close()


def test_workspace_replace_retry_delays_only_allow_windows_sharing_errors(
    monkeypatch,
) -> None:
    error = PermissionError(errno.EACCES, "sharing violation")
    error.winerror = 5

    assert workspace_storage._workspace_replace_retry_delays(error) == ()

    monkeypatch.setattr(workspace_storage.os, "name", "nt")
    assert workspace_storage._workspace_replace_retry_delays(error) == (
        workspace_storage._WINDOWS_WORKSPACE_REPLACE_RETRY_DELAYS
    )

    error.winerror = 123
    assert workspace_storage._workspace_replace_retry_delays(error) == ()
    error.winerror = 32
    error.errno = errno.ENOENT
    assert workspace_storage._workspace_replace_retry_delays(error) == ()


def test_replace_workspace_file_preservation_failure_keeps_cached_reader(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            source, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()
    manager = workspace_arrays.WorkspaceFileManager(destination)
    cached_file = manager.acquire()

    def _fail_link(_source, _destination):
        raise OSError(errno.EPERM, "hard links unavailable")

    def _fail_copy(_source, _destination):
        raise OSError(errno.ENOSPC, "copy failed")

    monkeypatch.setattr(workspace_storage.os, "link", _fail_link)
    monkeypatch.setattr(workspace_storage.shutil, "copyfile", _fail_copy)
    try:
        with pytest.raises(OSError, match="copy failed"):
            workspace_storage._replace_workspace_file(source, destination)
        assert cached_file._closed
        reopened_file = manager.acquire()
        assert reopened_file is not cached_file
        value = reopened_file.groups["0"].groups["imagetool"].variables["data"][()]
        assert np.asarray(value).item() == 1.0
        assert source.exists()
    finally:
        manager.close()


def test_replace_workspace_file_failure_keeps_old_file_reader_usable(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            source, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()
    manager = workspace_arrays.WorkspaceFileManager(destination)
    cached_file = manager.acquire()

    def _fail_replace(_source, _destination):
        raise PermissionError(errno.EACCES, "replacement denied")

    preserved_paths: list[pathlib.Path] = []
    original_preserve = workspace_storage._preserve_workspace_file_generation

    def _record_preserved_path(
        path: str | pathlib.Path,
        expected_identity: tuple[str, int, int, int],
    ):
        preserved = original_preserve(path, expected_identity)
        preserved_paths.append(pathlib.Path(preserved.path))

        def _cleanup() -> None:
            preserved.cleanup()
            raise RuntimeError("temporary cleanup failed")

        return workspace_storage._PreservedWorkspaceFile(
            path=preserved.path,
            file_identity=preserved.file_identity,
            cleanup=_cleanup,
        )

    monkeypatch.setattr(workspace_storage.os, "replace", _fail_replace)
    hidden_paths: list[str] = []
    monkeypatch.setattr(
        workspace_storage, "_hide_workspace_internal_file", hidden_paths.append
    )
    monkeypatch.setattr(
        workspace_storage,
        "_preserve_workspace_file_generation",
        _record_preserved_path,
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_replace_retry_delays", lambda _exc: ()
    )
    try:
        with pytest.raises(PermissionError, match="replacement denied"):
            workspace_storage._replace_workspace_file(source, destination)
        assert cached_file._closed
        reopened_file = manager.acquire()
        value = reopened_file.groups["0"].groups["imagetool"].variables["data"][()]
        assert np.asarray(value).item() == 1.0
        assert not manager._generation.retired
        assert pathlib.Path(manager._generation.path) == destination.resolve()
        assert preserved_paths
        assert all(not path.exists() for path in preserved_paths)
        assert hidden_paths == []
        assert source.exists()
    finally:
        manager.close()


def test_replace_workspace_file_does_not_adopt_disappeared_destination(
    monkeypatch, tmp_path
) -> None:
    source = tmp_path / "new.itws"
    destination = tmp_path / "current.itws"
    _write_transaction_test_workspace(destination)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="replacement")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            source, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()
    manager = workspace_arrays.WorkspaceFileManager(destination)
    manager.acquire()
    original_preserve = workspace_storage._preserve_workspace_file_generation

    def _remove_before_preservation(
        path: str | pathlib.Path,
        expected_identity: tuple[str, int, int, int],
    ):
        pathlib.Path(path).unlink()
        return original_preserve(path, expected_identity)

    monkeypatch.setattr(
        workspace_storage,
        "_preserve_workspace_file_generation",
        _remove_before_preservation,
    )
    try:
        with pytest.raises(FileNotFoundError):
            workspace_storage._replace_workspace_file(source, destination)
        assert source.exists()
        assert not destination.exists()
        with pytest.raises(RuntimeError, match="reader generation changed"):
            manager.acquire()
    finally:
        manager.close()


def test_cleanup_stale_workspace_reader_files_keeps_live_owners(
    monkeypatch, tmp_path
) -> None:
    workspace_path = tmp_path / "workspace.itws"
    workspace_path.write_bytes(b"workspace")
    directory = workspace_arrays._ensure_workspace_reader_directory(workspace_path)
    current = directory / f"reader-{workspace_arrays.os.getpid()}-{'1' * 32}.itws"
    stale = directory / f"export-424242-{'2' * 32}.itws"
    stale_handoff = directory / f"handoff-424242-{'3' * 32}.itws"
    unrelated = directory / "notes.txt"
    current.write_bytes(b"current")
    stale.write_bytes(b"stale")
    stale_handoff.write_bytes(b"stale handoff")
    unrelated.write_bytes(b"keep")
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid != 424242)

    workspace_arrays._cleanup_stale_workspace_reader_files(workspace_path)

    assert current.exists()
    assert not stale.exists()
    assert not stale_handoff.exists()
    assert unrelated.exists()


def test_write_full_workspace_tree_file_local_path_uses_destination_temp(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "local.itws"
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(1.0, title="local")}
    )
    write_targets: list[pathlib.Path] = []
    original_write = workspace_arrays._write_workspace_dataset_group_to_file

    def _record_write(target, *args, **kwargs):
        write_targets.append(pathlib.Path(target))
        return original_write(target, *args, **kwargs)

    monkeypatch.setattr(
        workspace_arrays, "_write_workspace_dataset_group_to_file", _record_write
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_cloud_path", lambda _path: False
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert write_targets
    assert all(target.parent == fname.parent for target in write_targets)
    assert all(target.name.startswith(f"{fname.name}.tmp-") for target in write_targets)


def test_write_full_workspace_tree_file_cloud_path_uses_scratch_and_replace_first(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "Dropbox" / "cloud.itws"
    fname.parent.mkdir()
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="cloud")}
    )
    write_targets: list[pathlib.Path] = []
    replace_calls: list[tuple[pathlib.Path, pathlib.Path]] = []
    original_write = workspace_arrays._write_workspace_dataset_group_to_file

    def _record_write(target, *args, **kwargs):
        write_targets.append(pathlib.Path(target))
        return original_write(target, *args, **kwargs)

    def _replace_by_copy(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        replace_calls.append((src_path, dst_path))
        dst_path.write_bytes(src_path.read_bytes())
        src_path.unlink()

    monkeypatch.setattr(
        workspace_arrays, "_write_workspace_dataset_group_to_file", _record_write
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _replace_by_copy)
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert write_targets
    assert all(target.parent != fname.parent for target in write_targets)
    assert replace_calls == [(write_targets[0], fname)]
    assert _read_transaction_test_value(fname) == 2.0


def test_write_full_workspace_tree_file_copies_unchanged_payload_groups(
    monkeypatch,
    tmp_path,
) -> None:

    fname = tmp_path / "copy.itws"
    ds = xr.Dataset(
        {
            _ITOOL_DATA_NAME: (
                ("x", "y"),
                np.arange(12, dtype=np.float64).reshape(3, 4),
            )
        },
        coords={
            "x": np.arange(3, dtype=np.float64),
            "y": np.arange(4, dtype=np.float64),
        },
        attrs={
            "itool_title": "old",
            "manager_node_uid": "n0",
            "manager_node_kind": "imagetool",
        },
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    rewritten = ds.assign_attrs(
        {
            "itool_title": "new",
            "manager_node_uid": "n0",
            "manager_node_kind": "imagetool",
            "Single Motor Scan": _rich_workspace_attr_value(),
        }
    )
    tree = xr.DataTree.from_dict({"0/imagetool": rewritten})

    def _fail_to_netcdf(*_args, **_kwargs):
        raise AssertionError("unchanged payload should be copied with h5py")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _fail_to_netcdf)
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0/imagetool", "0/imagetool", dict(rewritten.attrs)),),
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        assert group.attrs["itool_title"] == "new"
        decoded_attrs = workspace_arrays._h5py_attrs_to_dict(group.attrs)
        _assert_rich_workspace_attr(decoded_attrs["Single Motor Scan"])
        np.testing.assert_array_equal(
            group[_ITOOL_DATA_NAME][...],
            np.arange(12, dtype=np.float64).reshape(3, 4),
        )
    opened = workspace_arrays.open_workspace_datatree(fname, chunks=None)
    try:
        xr.testing.assert_identical(
            opened["0/imagetool"].to_dataset()[_ITOOL_DATA_NAME],
            rewritten[_ITOOL_DATA_NAME],
        )
    finally:
        opened.close()


def test_write_full_workspace_tree_file_network_scratch_skips_copy_reuse(
    monkeypatch, tmp_path
) -> None:
    import shutil

    fname = tmp_path / "network-copy-reuse.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(3.0, title="rewritten")}
    )

    def _fail_copyfile(*_args, **_kwargs):
        raise AssertionError("network scratch save should not copy old workspace")

    def _replace_by_copy(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        dst_path.write_bytes(src_path.read_bytes())
        src_path.unlink()

    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_network_path", lambda _path: True
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_cloud_path", lambda _path: False
    )
    monkeypatch.setattr(shutil, "copyfile", _fail_copyfile)
    monkeypatch.setattr(workspace_storage.os, "replace", _replace_by_copy)
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0/imagetool", "0/imagetool", None),),
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 3.0


def test_write_full_workspace_tree_file_scratch_exdev_fallback(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "fallback.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(4.0, title="fallback")}
    )
    original_replace = workspace_storage.os.replace
    replace_calls: list[tuple[pathlib.Path, pathlib.Path]] = []
    scratch_path: pathlib.Path | None = None

    def _replace_with_exdev(src, dst):
        nonlocal scratch_path
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        replace_calls.append((src_path, dst_path))
        if dst_path == fname and src_path.parent != fname.parent:
            scratch_path = src_path
            raise OSError(errno.EXDEV, "cross-device link")
        return original_replace(src, dst)

    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _replace_with_exdev)
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 4.0
    assert scratch_path is not None
    assert not scratch_path.exists()
    assert len(replace_calls) == 2
    assert replace_calls[0] == (scratch_path, fname)
    assert replace_calls[1][0].parent == fname.parent
    assert replace_calls[1][1] == fname
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_write_full_workspace_tree_file_rejects_file_repack_on_network_path(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "network.itws"
    _write_transaction_test_workspace(fname)

    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_high_risk", lambda *_: True
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_network_path", lambda *_: True
    )

    with pytest.raises(ValueError, match="File-level workspace repack cannot run"):
        workspace_storage._write_full_workspace_tree_file(
            fname,
            None,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0", "0", None),),
        )
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(5.0, title="network")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0", "0", None),),
        )
    finally:
        tree.close()
    assert _read_transaction_test_value(fname) == 5.0


def test_write_full_workspace_tree_file_scratch_replace_failure_preserves_old(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "replace-failure.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(5.0, title="failure")}
    )
    scratch_paths: list[pathlib.Path] = []

    def _fail_replace(src, dst):
        src_path = pathlib.Path(src)
        if pathlib.Path(dst) == fname:
            scratch_paths.append(src_path)
        raise OSError(errno.EPERM, "replace failed")

    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _fail_replace)
    try:
        with pytest.raises(OSError, match="replace failed"):
            workspace_storage._write_full_workspace_tree_file(
                fname, tree, _transaction_test_root_attrs()
            )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 1.0
    assert scratch_paths
    assert all(not scratch_path.exists() for scratch_path in scratch_paths)
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_write_full_workspace_tree_file_scratch_copy_failure_cleans_destination_tmp(
    monkeypatch, tmp_path
) -> None:
    import shutil

    fname = tmp_path / "copy-failure.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(6.0, title="failure")}
    )
    original_replace = workspace_storage.os.replace
    scratch_paths: list[pathlib.Path] = []

    def _replace_with_exdev(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        if dst_path == fname and src_path.parent != fname.parent:
            scratch_paths.append(src_path)
            raise OSError(errno.EXDEV, "cross-device link")
        return original_replace(src, dst)

    def _fail_copyfile(src, dst):
        pathlib.Path(dst).write_bytes(b"partial")
        raise OSError(errno.EIO, "copy failed")

    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(workspace_storage.os, "replace", _replace_with_exdev)
    monkeypatch.setattr(shutil, "copyfile", _fail_copyfile)
    try:
        with pytest.raises(OSError, match="copy failed"):
            workspace_storage._write_full_workspace_tree_file(
                fname, tree, _transaction_test_root_attrs()
            )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 1.0
    assert scratch_paths
    assert all(not scratch_path.exists() for scratch_path in scratch_paths)
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_workspace_recovery_discards_pending_only_transaction(tmp_path) -> None:
    fname = tmp_path / "pending-only.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "pendingonly"
    txn_path = f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"

    workspace_storage._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    workspace_storage._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    workspace_storage._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_restores_backup_before_pending_move(tmp_path) -> None:

    fname = tmp_path / "backup-before-pending.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "backuponly"
    txn_path = f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, _ = workspace_storage._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    workspace_storage._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    with h5py.File(fname, "a") as h5_file:
        workspace_storage._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        operation = group_operations[0]
        workspace_storage._move_h5_path(
            h5_file,
            typing.cast("str", operation["group_path"]),
            typing.cast("str", operation["backup_path"]),
        )

    workspace_storage._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_rolls_back_active_moved_before_commit(tmp_path) -> None:

    fname = tmp_path / "active-before-commit.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "activemoved"
    txn_path = f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, _ = workspace_storage._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    workspace_storage._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    with h5py.File(fname, "a") as h5_file:
        workspace_storage._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        operation = group_operations[0]
        workspace_storage._move_h5_path(
            h5_file,
            typing.cast("str", operation["group_path"]),
            typing.cast("str", operation["backup_path"]),
        )
        workspace_storage._move_h5_path(
            h5_file,
            typing.cast("str", operation["pending_path"]),
            typing.cast("str", operation["group_path"]),
        )

    workspace_storage._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_accepts_committed_before_cleanup(tmp_path) -> None:
    fname = tmp_path / "committed-before-cleanup.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "committed"
    txn_path = f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    root_attrs = _transaction_test_root_attrs(delta_save_count=1)
    group_operations, attr_updates = workspace_storage._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        root_attrs,
    )
    workspace_storage._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )
    workspace_storage._commit_workspace_transaction(
        fname, txn_path, group_operations, attr_updates, root_attrs
    )

    workspace_storage._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 2.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_rolls_back_attr_only_transaction(tmp_path) -> None:

    fname = tmp_path / "attrs-before-commit.itws"
    _write_transaction_test_workspace(fname)
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )
    attr_update = ("0/imagetool", {"itool_title": "new"}, fallback)
    txn_id = "attrrollback"
    txn_path = f"{workspace_format._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    root_attrs = _transaction_test_root_attrs(delta_save_count=1)
    _, attr_updates = workspace_storage._prepare_workspace_transaction(
        fname, txn_path, pending_root, backup_root, {}, (attr_update,), root_attrs
    )

    with h5py.File(fname, "a") as h5_file:
        workspace_storage._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        workspace_arrays._replace_h5_attrs(
            h5_file["0/imagetool"].attrs, attr_updates[0][1]
        )
        workspace_storage._write_root_attrs_to_open_workspace_file(h5_file, root_attrs)
        h5_file.flush()

    workspace_storage._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert h5_file["0/imagetool"].attrs["itool_title"] == "old"
        assert (
            workspace_format._workspace_delta_save_count_from_attrs(h5_file.attrs) == 0
        )
    _assert_no_workspace_internal_groups(fname)


def test_workspace_transaction_attr_update_encodes_non_native_values(tmp_path) -> None:

    fname = tmp_path / "rich-attrs-transaction.itws"
    _write_transaction_test_workspace(fname)
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )
    rich_attr = _rich_workspace_attr_value()
    workspace_storage._write_workspace_transaction_file(
        fname,
        (),
        (
            (
                "0/imagetool",
                {"itool_title": "new", "Single Motor Scan": rich_attr},
                fallback,
            ),
        ),
        _transaction_test_root_attrs(delta_save_count=1),
    )

    with h5py.File(fname, "r") as h5_file:
        decoded_attrs = workspace_arrays._h5py_attrs_to_dict(
            h5_file["0/imagetool"].attrs
        )
        assert decoded_attrs["itool_title"] == "new"
        _assert_rich_workspace_attr(decoded_attrs["Single Motor Scan"])
        assert (
            workspace_format._workspace_delta_save_count_from_attrs(h5_file.attrs) == 1
        )
    _assert_no_workspace_internal_groups(fname)


def test_workspace_transaction_does_not_copy_or_replace_full_file(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "incremental.itws"
    _write_transaction_test_workspace(fname)

    def _fail_full_file_operation(*_args, **_kwargs):
        raise AssertionError("incremental save used a full-file operation")

    monkeypatch.setattr(workspace_storage.shutil, "copyfile", _fail_full_file_operation)
    monkeypatch.setattr(workspace_storage.os, "replace", _fail_full_file_operation)

    workspace_storage._write_workspace_transaction_file(
        fname,
        (("0", {"0/imagetool": _transaction_test_dataset(4.0, title="new")}),),
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )

    assert _read_transaction_test_value(fname) == 4.0


def test_workspace_transaction_materializes_lazy_source_before_mutation(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "lazy-constructor-source.itws"
    _write_transaction_test_workspace(fname, value=1.0)
    opened = workspace_arrays.open_workspace_dataset(
        fname, "0/imagetool", chunks={"x": 1}
    )
    snapshot = opened[["data"]].copy(deep=False)
    original_materialize = workspace_storage._materialize_workspace_constructor_sources

    def _materialize_without_reader_lock(*args, **kwargs) -> None:
        lock_results: list[bool] = []

        def _probe_reader_lock() -> None:
            lock = workspace_arrays._workspace_file_lock(fname)
            acquired = lock.acquire(timeout=1)
            lock_results.append(acquired)
            if acquired:
                lock.release()

        probe = threading.Thread(target=_probe_reader_lock)
        probe.start()
        probe.join(2)
        assert lock_results == [True]
        original_materialize(*args, **kwargs)

    monkeypatch.setattr(
        workspace_storage,
        "_materialize_workspace_constructor_sources",
        _materialize_without_reader_lock,
    )

    try:
        workspace_storage._write_workspace_transaction_file(
            fname,
            (("1", {"1/figure": snapshot}),),
            (),
            _transaction_test_root_attrs(delta_save_count=1),
        )
        assert snapshot["data"].compute().item() == 1.0
        with h5py.File(fname, "r") as h5_file:
            assert np.asarray(h5_file["1/figure/data"]).item() == 1.0
    finally:
        opened.close()


def test_workspace_transaction_serializes_save_preparation(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "concurrent-saves.itws"
    _write_transaction_test_workspace(fname, value=1.0)
    original_materialize = workspace_storage._materialize_workspace_constructor_sources
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()
    errors: list[BaseException] = []
    call_count = 0

    def _coordinate_materialization(*args, **kwargs) -> None:
        nonlocal call_count
        with call_lock:
            call_index = call_count
            call_count += 1
        if call_index == 0:
            first_entered.set()
            if not release_first.wait(5):
                raise TimeoutError("First save was not released")
        else:
            second_entered.set()
        original_materialize(*args, **kwargs)

    def _save(value: float) -> None:
        try:
            workspace_storage._write_workspace_transaction_file(
                fname,
                (
                    (
                        "0",
                        {
                            "0/imagetool": _transaction_test_dataset(
                                value, title=f"save {value}"
                            )
                        },
                    ),
                ),
                (),
                _transaction_test_root_attrs(delta_save_count=int(value)),
            )
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(
        workspace_storage,
        "_materialize_workspace_constructor_sources",
        _coordinate_materialization,
    )
    first = threading.Thread(target=_save, args=(2.0,))
    second = threading.Thread(target=_save, args=(3.0,))

    first.start()
    assert first_entered.wait(5)
    second.start()
    try:
        assert not second_entered.wait(0.1)
    finally:
        release_first.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert errors == []
    assert _read_transaction_test_value(fname) == 3.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_transaction_rejects_change_during_materialization(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "incremental-conflict.itws"
    external = tmp_path / "external.itws"
    _write_transaction_test_workspace(fname, value=1.0)
    _write_transaction_test_workspace(external, value=7.0)
    original_materialize = workspace_storage._materialize_workspace_constructor_sources

    def _materialize_then_replace(*args, **kwargs) -> None:
        original_materialize(*args, **kwargs)
        external.replace(fname)

    monkeypatch.setattr(
        workspace_storage,
        "_materialize_workspace_constructor_sources",
        _materialize_then_replace,
    )

    with pytest.raises(
        workspace_storage._WorkspacePublicationConflictError,
        match="incremental save was prepared",
    ):
        workspace_storage._write_workspace_transaction_file(
            fname,
            (
                (
                    "0",
                    {"0/imagetool": _transaction_test_dataset(2.0, title="stale save")},
                ),
            ),
            (),
            _transaction_test_root_attrs(delta_save_count=1),
        )

    assert _read_transaction_test_value(fname) == 7.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_constructor_materialization_reuses_data_and_chunks(tmp_path) -> None:
    fname = tmp_path / "shared-lazy-source.itws"
    xr.Dataset({"data": ("x", np.arange(6))}).to_netcdf(fname, engine="h5netcdf")
    opened = workspace_arrays.open_workspace_dataset(fname, "/", chunks={"x": 2})
    snapshot = opened[["data"]].copy(deep=False)
    first_constructor = {"first": snapshot}
    second_constructor = {"second": snapshot}
    rewrite_map = {
        "first": ("first", first_constructor),
        "second": ("second", second_constructor),
    }

    try:
        workspace_storage._materialize_workspace_constructor_sources(fname, rewrite_map)
        first = first_constructor["first"]
        second = second_constructor["second"]
        assert first is second
        assert first["data"].chunks is None
        assert "source" not in first["data"].encoding
        assert workspace_arrays.workspace_dataset_encoding(first)["data"][
            "chunksizes"
        ] == (2,)
    finally:
        opened.close()


def test_workspace_attr_transaction_does_not_copy_unchanged_payload(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "attr-only.itws"
    _write_transaction_test_workspace(fname)
    manager = workspace_arrays.WorkspaceFileManager(fname, "0/imagetool")
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )

    def _fail_payload_copy(*_args, **_kwargs):
        raise AssertionError("attribute-only save copied an unchanged payload")

    monkeypatch.setattr(
        workspace_arrays,
        "_create_workspace_group_reader_file",
        _fail_payload_copy,
    )
    try:
        workspace_storage._write_workspace_transaction_file(
            fname,
            (),
            (("0/imagetool", {"itool_title": "new"}, fallback),),
            _transaction_test_root_attrs(delta_save_count=1),
        )
        assert (
            manager.acquire().groups["0"].groups["imagetool"].attrs["itool_title"]
            == "new"
        )
    finally:
        manager.close()


def test_workspace_transaction_publishes_immutable_reader_generation(tmp_path) -> None:
    fname = tmp_path / "immutable-delta.itws"
    _write_transaction_test_workspace(fname, value=1.0)
    before = workspace_arrays.open_workspace_dataset(fname, "0/imagetool", chunks={})
    old_data = before["data"].copy(deep=False)

    workspace_storage._write_workspace_transaction_file(
        fname,
        (("0", {"0/imagetool": _transaction_test_dataset(9.0, title="new")}),),
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    after = workspace_arrays.open_workspace_dataset(fname, "0/imagetool", chunks={})
    try:
        assert old_data.compute().item() == 1.0
        assert after["data"].compute().item() == 9.0
    finally:
        before.close()
        after.close()


def test_workspace_transaction_switches_waiting_reader_before_commit(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "waiting-reader.itws"
    _write_transaction_test_workspace(fname, value=1.0)
    manager = workspace_arrays.WorkspaceFileManager(fname, "0/imagetool")
    snapshot_started = threading.Event()
    allow_snapshot = threading.Event()
    reader_finished = threading.Event()
    errors: list[BaseException] = []
    values: list[float] = []
    original_snapshot = workspace_arrays._create_workspace_group_reader_file

    def _pause_snapshot(*args, **kwargs):
        snapshot_started.set()
        if not allow_snapshot.wait(2):
            raise TimeoutError("Reader snapshot test did not resume")
        return original_snapshot(*args, **kwargs)

    def _write() -> None:
        try:
            workspace_storage._write_workspace_transaction_file(
                fname,
                (
                    (
                        "0",
                        {"0/imagetool": _transaction_test_dataset(2.0, title="new")},
                    ),
                ),
                (),
                _transaction_test_root_attrs(delta_save_count=1),
            )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def _read() -> None:
        try:
            h5_file = manager.acquire()
            value = h5_file.groups["0"].groups["imagetool"].variables["data"][()]
            values.append(float(np.asarray(value).item()))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)
        finally:
            reader_finished.set()

    monkeypatch.setattr(
        workspace_arrays, "_create_workspace_group_reader_file", _pause_snapshot
    )
    writer = threading.Thread(target=_write)
    reader = threading.Thread(target=_read)
    writer.start()
    assert snapshot_started.wait(2)
    reader.start()
    try:
        assert not reader_finished.wait(0.05)
    finally:
        allow_snapshot.set()
        writer.join(2)
        reader.join(2)
        manager.close()

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert values == [1.0]
    assert _read_transaction_test_value(fname) == 2.0


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


def test_hide_workspace_internal_file_sets_macos_hidden_flag(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    lock_path = "/workspace/.workspace.itws.lock"
    regular_stat = types.SimpleNamespace(st_mode=0o100600, st_flags=0)

    monkeypatch.setattr(workspace_storage.sys, "platform", "darwin")
    monkeypatch.setattr(workspace_storage.os, "lstat", lambda _path: regular_stat)
    monkeypatch.setattr(
        workspace_storage.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    workspace_storage._hide_workspace_internal_file(lock_path)

    assert calls == [(lock_path, 0x8000)]


def test_hide_workspace_internal_file_skips_macos_symlink(monkeypatch) -> None:
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

    workspace_storage._hide_workspace_internal_file("/workspace/.workspace.itws.lock")

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


def test_workspace_high_risk_path_detection() -> None:
    assert workspace_storage._workspace_path_is_high_risk(
        pathlib.Path.home() / "OneDrive" / "workspace.itws"
    )
    assert workspace_storage._workspace_path_is_high_risk(
        pathlib.Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "workspace.itws"
    )
    assert workspace_storage._workspace_path_is_high_risk(
        pathlib.Path("//server/share/workspace.itws")
    )


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


def test_hide_workspace_internal_file_windows_paths(monkeypatch) -> None:
    import ctypes

    calls: list[tuple[str, int]] = []

    class _Kernel32:
        @staticmethod
        def GetFileAttributesW(_path: str) -> int:
            return 0x20

        @staticmethod
        def SetFileAttributesW(path: str, attrs: int) -> None:
            calls.append((path, attrs))

    monkeypatch.setattr(workspace_storage.sys, "platform", "win32")
    monkeypatch.setattr(workspace_storage.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    workspace_storage._hide_workspace_internal_file("missing-windll.itws.lock")
    assert calls == []

    monkeypatch.setattr(
        ctypes, "windll", types.SimpleNamespace(kernel32=_Kernel32()), raising=False
    )
    workspace_storage._hide_workspace_internal_file("hidden.itws.lock")
    assert calls == [("hidden.itws.lock", 0x22)]


def test_workspace_document_lock_info_without_lock(tmp_path) -> None:
    info = workspace_storage._workspace_document_lock_info(tmp_path / "free.itws")

    assert info.pid is None
    assert info.hostname == ""
    assert info.appname == ""


def test_workspace_path_risk_detection_fallbacks(monkeypatch, tmp_path) -> None:
    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert workspace_storage._workspace_path_is_likely_cloud_path(
        tmp_path / "Dropbox" / "workspace.itws"
    )
    assert workspace_storage._workspace_path_is_likely_network_path(
        pathlib.Path("/net/server/workspace.itws")
    )

    monkeypatch.setattr(workspace_storage.sys, "platform", "darwin")
    assert workspace_storage._workspace_path_is_likely_network_path(
        pathlib.Path("/Volumes/share/workspace.itws")
    )


def test_workspace_requires_full_save_reasons(tmp_path) -> None:
    options = erlab.interactive.options
    old_incremental = options["io/workspace/use_incremental"]
    old_remote = options["io/workspace/incremental_save_on_remote"]
    existing = tmp_path / "existing.itws"
    existing.touch()
    try:
        options["io/workspace/use_incremental"] = False
        assert workspace_storage._workspace_requires_full_save(
            existing,
            needs_full_save=False,
            schema_version=workspace_format._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )

        options["io/workspace/use_incremental"] = True
        options["io/workspace/incremental_save_on_remote"] = True
        assert workspace_storage._workspace_requires_full_save(
            tmp_path / "missing.itws",
            needs_full_save=False,
            schema_version=workspace_format._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )
        for kwargs in (
            {"needs_full_save": True},
            {
                "schema_version": (
                    workspace_format._current_workspace_schema_version() - 1
                )
            },
            {"structure_modified": True},
            {"has_dirty_added": True},
            {"has_dirty_removed": True},
        ):
            call_kwargs = {
                "needs_full_save": False,
                "schema_version": workspace_format._current_workspace_schema_version(),
                "structure_modified": False,
                "has_dirty_added": False,
                "has_dirty_removed": False,
            }
            call_kwargs.update(kwargs)
            assert workspace_storage._workspace_requires_full_save(
                existing, **call_kwargs
            )
    finally:
        options["io/workspace/use_incremental"] = old_incremental
        options["io/workspace/incremental_save_on_remote"] = old_remote


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


def test_validate_workspace_h5_file_rejects_non_workspace(tmp_path) -> None:

    fname = tmp_path / "invalid.h5"
    with h5py.File(fname, "w"):
        pass

    with pytest.raises(ValueError, match="not valid"):
        workspace_storage._validate_workspace_h5_file(fname)


def test_fsync_parent_directory_skips_non_posix(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(workspace_storage.os, "name", "nt")
    monkeypatch.setattr(
        workspace_storage.os,
        "open",
        lambda *args, **kwargs: pytest.fail("non-posix platforms should not fsync"),
    )

    workspace_storage._fsync_parent_directory(tmp_path / "workspace.itws")
