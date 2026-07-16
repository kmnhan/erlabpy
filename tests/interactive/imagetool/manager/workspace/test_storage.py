import errno
import json
import pathlib
import types
import typing

import h5py
import hdf5plugin
import numpy as np
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
