import contextlib
import datetime
import json
import logging
import os
import pathlib
import sys
import time
import types
import typing
import warnings
from collections.abc import Callable, Iterable, Mapping

import h5py
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._saving as workspace_saving
import erlab.interactive.imagetool.manager._workspace._state as workspace_state
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.viewer as imagetool_viewer
from erlab.interactive._options.schema import AppOptions
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._model import full_data
from erlab.interactive.imagetool._provenance._operations import (
    ImageToolSelectionSourceBinding,
)
from erlab.interactive.imagetool.manager import ImageToolManager
from erlab.interactive.imagetool.manager._workspace import (
    _controller as workspace_controller,
)
from tests.interactive.imagetool.manager.helpers import (
    action_map_by_object_name,
    adopt_workspace_path,
    select_child_tool,
    select_tools,
)
from tests.interactive.imagetool.manager.workspace._support import (
    _AddedTimeChildTool,
    _assert_no_workspace_internal_groups,
    _assert_rich_workspace_attr,
    _compute_first_value,
    _hdf5_blosc2_level_codec,
    _open_external_file_backed_hdf5_imagetool_data,
    _open_external_lazy_hdf5_imagetool_data,
    _request_workspace_save_and_wait,
    _request_workspace_save_as_and_wait,
    _rich_workspace_attr_value,
    _transaction_test_root_attrs,
    _write_transaction_test_workspace,
)


def test_manager_workspace_saves_added_time_for_all_node_kinds(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    root_added = datetime.datetime(
        2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=9))
    )
    child_added = datetime.datetime(
        2024, 1, 3, 4, 5, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))
    )
    tool_added = datetime.datetime(2024, 1, 4, 5, 6, 7, tzinfo=datetime.UTC)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root_index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            show=False,
            created_time=root_added,
        )
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(test_data + 1, _in_manager=True),
            root_index,
            show=False,
            created_time=child_added,
        )
        tool_uid = manager.add_childtool(
            _AddedTimeChildTool(test_data),
            root_index,
            show=False,
            created_time=tool_added,
        )

        fname = tmp_path / "added-time.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

    with h5py.File(fname, "r") as h5_file:
        assert h5_file["0/imagetool"].attrs[
            "manager_node_added_at"
        ] == root_added.isoformat(timespec="seconds")
        assert h5_file[f"0/childtools/{child_uid}/imagetool"].attrs[
            "manager_node_added_at"
        ] == child_added.isoformat(timespec="seconds")
        assert h5_file[f"0/childtools/{tool_uid}/tool"].attrs[
            "manager_node_added_at"
        ] == tool_added.isoformat(timespec="seconds")


def test_manager_workspace_layout_only_save_updates_root_manifest_only(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "layout-only.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        delta_save_count = manager._workspace_state.delta_save_count

        manager.resize(manager.width() + 40, manager.height() + 30)
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        assert manager._workspace_state.layout_modified
        assert manager._workspace_controller._dirty_details_text()
        manager._mark_workspace_layout_dirty()

        def _forbid_node_serialization(*_args, **_kwargs):
            raise AssertionError("layout-only save serialized a node")

        def _forbid_full_save(*_args, **_kwargs):
            raise AssertionError("layout-only save requested a full workspace write")

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            _forbid_node_serialization,
        )
        monkeypatch.setattr(
            workspace_storage, "_write_full_workspace_tree_file", _forbid_full_save
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == delta_save_count
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert "delta_save_count" not in manifest
        assert (
            manifest["manager_layout"]
            == manager._workspace_controller.saving._workspace_layout_snapshot()
        )


def test_manager_workspace_standalone_app_only_save_updates_root_manifest_only(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "standalone-only.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        delta_save_count = manager._workspace_state.delta_save_count

        manager.show_ptable()
        ptable = manager.ptable_window
        ptable.hv_edit.setText("150")
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        assert manager._workspace_state.layout_modified

        def _forbid_node_serialization(*_args, **_kwargs):
            raise AssertionError("standalone-only save serialized a node")

        def _forbid_full_save(*_args, **_kwargs):
            raise AssertionError(
                "standalone-only save requested a full workspace write"
            )

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            _forbid_node_serialization,
        )
        monkeypatch.setattr(
            workspace_storage, "_write_full_workspace_tree_file", _forbid_full_save
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == delta_save_count
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert "delta_save_count" not in manifest
        assert manifest["standalone_apps"]["apps"]["ptable"]["photon_energy"] == "150"


def test_manager_workspace_unlink_removes_saved_link_group(
    qtbot,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        fname = tmp_path / "unlinked.itws"
        manager.link_imagetools(0, 1)
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        select_tools(manager, [0, 1])
        manager.unlink_selected()
        assert manager.is_workspace_modified
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        manifest = workspace_format._workspace_manifest_from_attrs(
            workspace_arrays._read_workspace_root_attrs_h5py(fname)
        )
        assert all("link_group" not in entry for entry in manifest["nodes"])

        assert manager._workspace_controller.loading._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert not manager.get_imagetool(0).slicer_area.is_linked
        assert not manager.get_imagetool(1).slicer_area.is_linked
        assert not manager.is_workspace_modified


def test_manager_load_workspace_dataset_ignores_invalid_saved_metadata(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(2.0)},
    )
    saved = itool(data, manager=False, execute=False)
    qtbot.addWidget(saved)
    assert isinstance(saved, erlab.interactive.imagetool.ImageTool)
    ds = saved.to_dataset()
    ds.attrs["manager_node_uid"] = "loaded"
    ds.attrs["manager_node_provenance_spec"] = "{not-json"
    ds.attrs["manager_node_live_source_spec"] = "{not-json"
    ds.attrs["manager_node_live_source_binding"] = "{not-json"

    with manager_context() as manager:
        target = (
            manager._workspace_controller.loading._load_workspace_imagetool_dataset(
                ds, parent_target=None, node_path="-1"
            )
        )

        assert target in manager._tool_graph.root_wrappers
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        binding = ImageToolSelectionSourceBinding(
            selection_mode="isel",
            selection_indexers={"x": 0},
        )
        bound_ds = saved.to_dataset()
        bound_ds.attrs["manager_node_uid"] = "bound"
        bound_ds.attrs.pop("manager_node_live_source_spec", None)
        bound_ds.attrs["manager_node_live_source_binding"] = json.dumps(
            binding.model_dump(mode="json")
        )
        bound_ds.attrs.pop("itool_name", None)

        bound_target = (
            manager._workspace_controller.loading._load_workspace_imagetool_dataset(
                bound_ds, parent_target=None, node_path="-2"
            )
        )

        assert manager._node_for_target(bound_target).source_binding == binding
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)


def test_manager_workspace_full_save_copy_group_edge_cases(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace_path = tmp_path / "copy-groups.itws"
    workspace_path.touch()

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._workspace_state.path = workspace_path
        monkeypatch.setattr(
            workspace_arrays,
            "_read_workspace_root_attrs_h5py",
            lambda _path: (_ for _ in ()).throw(RuntimeError("metadata failed")),
        )
        assert manager._workspace_controller.saving._workspace_full_save_copy_groups(
            xr.DataTree()
        ) == (None, ())

        monkeypatch.setattr(
            workspace_arrays,
            "_read_workspace_root_attrs_h5py",
            lambda _path: {"imagetool_workspace_schema_version": 1},
        )
        assert manager._workspace_controller.saving._workspace_full_save_copy_groups(
            xr.DataTree()
        ) == (None, ())

        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid
        tool = DerivativeTool(data)
        monkeypatch.setattr(tool, "can_save_and_load", lambda: False)
        manager.add_childtool(tool, 0, show=False)
        manager.add_childtool(_AddedTimeChildTool(data), 0, show=False)
        manager._workspace_controller._mark_workspace_clean()
        tree = manager._workspace_controller.saving._to_datatree()
        monkeypatch.setitem(
            manager._tool_graph.nodes,
            "missing-tool",
            types.SimpleNamespace(
                is_imagetool=False,
                tool_window=None,
                pending_workspace_tool_payload=None,
            ),
        )
        try:
            manifest_without_identities = {
                "schema_version": workspace_format._current_workspace_schema_version(),
                "nodes": [[]],
                "root_order": [0],
            }
            monkeypatch.setattr(
                workspace_arrays,
                "_read_workspace_root_attrs_h5py",
                lambda _path: {
                    "imagetool_workspace_schema_version": (
                        workspace_format._current_workspace_schema_version()
                    ),
                    workspace_format._WORKSPACE_MANIFEST_ATTR: json.dumps(
                        manifest_without_identities
                    ),
                },
            )
            assert (
                manager._workspace_controller.saving._workspace_full_save_copy_groups(
                    tree
                )
                == (
                    str(workspace_path),
                    (),
                )
            )

            manifest_with_missing_tree_payload = {
                "schema_version": workspace_format._current_workspace_schema_version(),
                "nodes": [
                    [],
                    {"uid": uid, "kind": "imagetool", "path": "0"},
                ],
                "root_order": [0],
            }
            monkeypatch.setattr(
                workspace_arrays,
                "_read_workspace_root_attrs_h5py",
                lambda _path: {
                    "imagetool_workspace_schema_version": (
                        workspace_format._current_workspace_schema_version()
                    ),
                    workspace_format._WORKSPACE_MANIFEST_ATTR: json.dumps(
                        manifest_with_missing_tree_payload
                    ),
                },
            )
            assert (
                manager._workspace_controller.saving._workspace_full_save_copy_groups(
                    xr.DataTree()
                )
                == (
                    str(workspace_path),
                    (),
                )
            )
        finally:
            tree.close()
            manager._tool_graph.nodes.pop("missing-tool", None)


def test_workspace_state_repeated_options_dirty_during_save() -> None:
    state = workspace_state._ManagerWorkspaceState()

    assert state.mark_options_dirty()
    assert state.dirty_generation == 1
    assert not state.mark_options_dirty()
    assert state.dirty_generation == 1

    state.save_in_progress = True

    assert state.mark_options_dirty()
    assert state.dirty_generation == 2


def test_manager_close_save_path_updates_file_path(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = tmp_path / "close-save.itws"
        manager._workspace_state.structure_modified = True
        save_closing_states: list[bool] = []
        save_callbacks: list[Callable[[bool], None]] = []
        file_path_calls: list[str] = []
        close_calls: list[str] = []

        def _save(
            *, native: bool = True, on_finished: Callable[[bool], None] | None = None
        ) -> bool:
            save_closing_states.append(manager._workspace_state.closing_document)
            if on_finished is not None:
                save_callbacks.append(on_finished)
            return True

        with monkeypatch.context() as patch:
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                lambda _manager, path: file_path_calls.append(path),
            )
            patch.setattr(
                QtWidgets.QMessageBox,
                "exec",
                lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
            )
            patch.setattr(manager._workspace_controller, "save", _save)
            patch.setattr(manager, "close", lambda: close_calls.append("close") or True)
            event = QtGui.QCloseEvent()
            manager.closeEvent(event)
            assert not event.isAccepted()
            manager._workspace_controller._mark_workspace_clean()
            save_callbacks[0](True)

        assert save_closing_states == [True]
        assert file_path_calls == [str(manager._workspace_state.path)]
        assert close_calls == ["close"]
        assert not manager._workspace_state.closing_document


def test_manager_close_ignores_active_save_and_async_compaction(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        saving_event = QtGui.QCloseEvent()
        manager.closeEvent(saving_event)

        assert not saving_event.isAccepted()

        manager._workspace_state.save_in_progress = False
        manager._workspace_state.path = tmp_path / "close-compact.itws"
        close_calls: list[str] = []
        compaction_callbacks: list[Callable[[], None]] = []

        def _compact_workspace_before_shutdown(
            *, on_finished: Callable[[], None]
        ) -> bool:
            compaction_callbacks.append(on_finished)
            return True

        with monkeypatch.context() as patch:
            patch.setattr(
                manager._workspace_controller,
                "_dirty_workspace_save_choice",
                lambda _message: "discard",
            )
            patch.setattr(
                manager._workspace_controller,
                "_compact_workspace_before_shutdown",
                _compact_workspace_before_shutdown,
            )
            patch.setattr(manager, "close", lambda: close_calls.append("close") or True)
            compacting_event = QtGui.QCloseEvent()
            manager.closeEvent(compacting_event)
            assert not compacting_event.isAccepted()
            assert len(compaction_callbacks) == 1
            compaction_callbacks[0]()

        assert close_calls == ["close"]


def test_manager_close_compacts_clean_delta_workspace(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = tmp_path / "close-no-compact.itws"
        manager._workspace_state.delta_save_count = 1
        compact_calls: list[str] = []

        monkeypatch.setattr(
            manager._workspace_controller,
            "_compact_workspace_before_shutdown",
            lambda **_kwargs: compact_calls.append("compact") or False,
        )

        assert manager.close()

    assert compact_calls == ["compact"]


def test_manager_workspace_save_as_locked_target_does_not_write(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-save-as.itws"
    _write_transaction_test_workspace(fname)
    lock = workspace_storage._acquire_workspace_document_lock(fname)
    operation_errors: list[tuple[typing.Any, ...]] = []
    try:
        with manager_context() as manager:
            monkeypatch.setattr(
                manager._workspace_controller,
                "_workspace_save_dialog",
                lambda *args, **kwargs: str(fname),
            )
            monkeypatch.setattr(
                manager._workspace_controller.saving,
                "_save_workspace_document",
                lambda *args, **kwargs: pytest.fail(
                    "Save As should lock the target before writing"
                ),
            )
            monkeypatch.setattr(
                manager,
                "_show_operation_error",
                lambda *args, **kwargs: operation_errors.append(args),
            )

            assert not manager._workspace_controller.save_as(native=False)
    finally:
        lock.unlock()

    assert operation_errors


def test_manager_workspace_save_as_reports_snapshot_error(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "write-error.itws"
    operation_errors: list[tuple[typing.Any, ...]] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda *args, **kwargs: str(fname),
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_full_save_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args, **kwargs: operation_errors.append(args),
        )

        assert not manager._workspace_controller.save_as(native=False)

    assert operation_errors == [
        (
            "Error while saving workspace",
            "An error occurred while saving the workspace file.",
        )
    ]


def test_manager_workspace_save_as_rejects_h5_target(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "workspace.h5"
    warnings: list[tuple[typing.Any, ...]] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda *args, **kwargs: str(fname),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args)
        )

        assert not manager._workspace_controller.save_as(native=False)

    assert len(warnings) == 1
    assert warnings[0][1:] == (
        "Workspace Not Saved",
        "ImageTool Manager saves workspaces as .itws files.",
    )
    assert not fname.exists()


def test_manager_workspace_load_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-load.itws"
    _write_transaction_test_workspace(fname)
    lock = workspace_storage._acquire_workspace_document_lock(fname)
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            workspace_storage,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Load should lock the workspace before recovery")
            ),
        )
        with manager_context() as manager, pytest.raises(BlockingIOError):
            manager._workspace_controller.loading._load_workspace_file(
                fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )
    finally:
        lock.unlock()

    assert recovery_calls == []


def test_manager_workspace_path_lock_contract(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _FakeLock:
        def __init__(self) -> None:
            self.unlock_count = 0

        def unlock(self) -> None:
            self.unlock_count += 1

    with manager_context() as manager:
        current = (tmp_path / "current.itws").resolve()
        manager._workspace_state.path = current
        lock = _FakeLock()

        manager._workspace_controller._set_workspace_path(current, workspace_lock=lock)

        assert lock.unlock_count == 1
        with pytest.raises(RuntimeError, match="pre-acquired document lock"):
            manager._workspace_controller._set_workspace_path(tmp_path / "other.itws")


def test_manager_open_recent_menu_stays_disabled_while_save_in_progress(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "recent.itws"
    workspace.touch()

    with manager_context() as manager:
        manager._workspace_controller._record_recent_workspace(workspace)
        assert manager.open_recent_menu.isEnabled()

        manager._workspace_state.save_in_progress = True
        manager._workspace_controller._refresh_open_recent_menu_action()
        assert not manager.open_recent_menu.isEnabled()
        manager._workspace_controller._populate_open_recent_menu()
        assert not manager.open_recent_menu.isEnabled()
        assert "manager_recent_workspace_action_0" in action_map_by_object_name(
            manager.open_recent_menu
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: pytest.fail(
                "Open Recent should not show dialogs during workspace save"
            ),
        )
        assert not manager.open_recent_workspace(tmp_path / "missing.itws")

        manager._workspace_state.save_in_progress = False
        manager._workspace_controller._refresh_open_recent_menu_action()
        assert manager.open_recent_menu.isEnabled()


def test_manager_compact_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(manager._workspace_controller, "save_as", lambda: True)
        assert manager.compact_workspace()

        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.save_in_progress = True
        assert not manager.compact_workspace()
        manager._workspace_state.save_in_progress = False

        operation_errors: list[tuple[typing.Any, ...]] = []
        focus_restores: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_save_workspace_document",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("compact failed")
            ),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args: operation_errors.append(args),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restores.append(origin),
        )

        assert not manager.compact_workspace()
        assert operation_errors == [
            (
                "Error while compacting workspace",
                "An error occurred while compacting the workspace file.",
            )
        ]
        assert focus_restores == [None]


def test_manager_compact_workspace_copies_matching_groups(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "manual-repack.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        tree = manager._workspace_controller.saving._to_datatree()
        try:
            copy_source, copy_groups = (
                manager._workspace_controller.saving._workspace_full_save_copy_groups(
                    tree
                )
            )
        finally:
            tree.close()
        assert copy_source == str(fname)
        assert copy_groups

        original_write = workspace_storage._write_full_workspace_tree_file
        full_write_calls: list[
            tuple[
                str | os.PathLike[str] | None,
                tuple[tuple[str, str, dict[str, typing.Any] | None], ...],
            ]
        ] = []

        def _record_full_write(
            write_fname: str | os.PathLike[str],
            write_tree: xr.DataTree,
            root_attrs: Mapping[str, typing.Any],
            **kwargs: typing.Any,
        ) -> None:
            full_write_calls.append(
                (kwargs.get("copy_source"), tuple(kwargs.get("copy_groups", ())))
            )
            original_write(write_fname, write_tree, root_attrs, **kwargs)

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_full_workspace_tree_file",
            _record_full_write,
        )

        assert manager.compact_workspace()

        copy_source, copy_groups = full_write_calls[-1]
        assert copy_source == str(fname)
        assert copy_groups


def test_manager_compact_workspace_reduces_internal_holes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_compression = erlab.interactive.options["io/workspace/compression"]
    erlab.interactive.options["io/workspace/compression"] = "none"
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            rng = np.random.default_rng(1234)
            data = xr.DataArray(
                rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8),
                dims=("x", "y"),
            )
            updated = xr.DataArray(
                rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8),
                dims=("x", "y"),
            )

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            uid = manager._tool_graph.root_wrappers[0].uid

            fname = tmp_path / "hole-repack.itws"
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
            adopt_workspace_path(manager, fname)
            manager._workspace_controller._mark_workspace_clean()
            size_full = fname.stat().st_size

            manager.get_imagetool(0).slicer_area.replace_source_data(
                updated,
                auto_compute=False,
            )
            manager._workspace_controller._mark_node_data_dirty(uid)
            assert _request_workspace_save_and_wait(qtbot, manager)
            size_incremental = fname.stat().st_size

            monkeypatch.setattr(
                erlab.interactive.utils,
                "wait_dialog",
                lambda *args, **kwargs: contextlib.nullcontext(),
            )
            assert manager.compact_workspace()
            size_compact = fname.stat().st_size

            assert size_incremental > size_full + data.nbytes // 2
            assert size_compact < size_incremental - data.nbytes // 2
            _assert_no_workspace_internal_groups(fname)
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_compact_workspace_reapplies_compression_mode(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    old_compression = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                dims=("x", "y"),
            )

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            fname = tmp_path / "compression-repack.itws"
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
            adopt_workspace_path(manager, fname)
            manager._workspace_controller._mark_workspace_clean()
            with h5py.File(fname, "r") as h5_file:
                assert (
                    _hdf5_blosc2_level_codec(h5_file["0/imagetool"][_ITOOL_DATA_NAME])
                    is None
                )

            original_write = workspace_storage._write_full_workspace_tree_file
            full_write_calls: list[
                tuple[str | os.PathLike[str] | None, tuple[object, ...]]
            ] = []

            def _record_full_write(
                write_fname: str | os.PathLike[str],
                write_tree: xr.DataTree,
                root_attrs: Mapping[str, typing.Any],
                **kwargs: typing.Any,
            ) -> None:
                full_write_calls.append(
                    (kwargs.get("copy_source"), tuple(kwargs.get("copy_groups", ())))
                )
                original_write(write_fname, write_tree, root_attrs, **kwargs)

            erlab.interactive.options["io/workspace/compression"] = "zstd1"
            monkeypatch.setattr(
                erlab.interactive.utils,
                "wait_dialog",
                lambda *args, **kwargs: contextlib.nullcontext(),
            )
            monkeypatch.setattr(
                workspace_storage,
                "_write_full_workspace_tree_file",
                _record_full_write,
            )
            assert manager.compact_workspace()

            assert full_write_calls[-1] == (str(fname), ())
            with h5py.File(fname, "r") as h5_file:
                assert _hdf5_blosc2_level_codec(
                    h5_file["0/imagetool"][_ITOOL_DATA_NAME]
                ) == (1, 5)
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_shutdown_compact_preserves_existing_dataset_filters(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    old_compression = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                dims=("x", "y"),
            )

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            uid = manager._tool_graph.root_wrappers[0].uid

            fname = tmp_path / "shutdown-preserve-filters.itws"
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
            adopt_workspace_path(manager, fname)
            manager._workspace_controller._mark_workspace_clean()
            manager._workspace_controller._mark_node_data_dirty(uid)
            assert _request_workspace_save_and_wait(qtbot, manager)

            with h5py.File(fname, "r") as h5_file:
                assert (
                    _hdf5_blosc2_level_codec(h5_file["0/imagetool"][_ITOOL_DATA_NAME])
                    is None
                )

            erlab.interactive.options["io/workspace/compression"] = "zstd1"
            monkeypatch.setattr(
                workspace_saving,
                "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
                1,
            )
            monkeypatch.setattr(
                workspace_saving,
                "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
                0.0,
            )

            assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)

            with h5py.File(fname, "r") as h5_file:
                assert (
                    _hdf5_blosc2_level_codec(h5_file["0/imagetool"][_ITOOL_DATA_NAME])
                    is None
                )
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_shutdown_compaction_logs_failure(
    caplog,
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.delta_save_count = 1
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=100,
            replacement_delta_count=1,
        )

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_should_repack_before_shutdown",
            lambda: True,
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_file_repack_snapshot",
            lambda generation: workspace_saving._WorkspaceSaveSnapshot(
                generation=generation,
                root_attrs=_transaction_test_root_attrs(delta_save_count=1),
                delta_save_count=0,
                file_repack=True,
            ),
        )

        with caplog.at_level(logging.ERROR):
            assert manager._workspace_controller._compact_workspace_before_shutdown()
            assert len(pool.workers) == 1
            pool.workers[0].finish(
                error=workspace_saving._WorkspaceSaveError("compact failed"),
            )
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

    assert "Failed to compact workspace before shutdown" in caplog.text
    assert "compact failed" in caplog.text


def test_manager_workspace_save_dialog_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    calls: list[tuple[str, object]] = []
    original_file_dialog = QtWidgets.QFileDialog

    class _FakeFileDialog:
        AcceptMode = original_file_dialog.AcceptMode
        FileMode = original_file_dialog.FileMode
        Option = original_file_dialog.Option
        exec_result = 0

        def __init__(self, _parent, caption: str) -> None:
            calls.append(("caption", caption))

        def setAcceptMode(self, mode) -> None:
            calls.append(("accept", mode))

        def setFileMode(self, mode) -> None:
            calls.append(("file_mode", mode))

        def setNameFilter(self, name_filter: str) -> None:
            calls.append(("filter", name_filter))

        def setDefaultSuffix(self, suffix: str) -> None:
            calls.append(("suffix", suffix))

        def selectFile(self, fname: str) -> None:
            calls.append(("select", fname))

        def setDirectory(self, directory: str) -> None:
            calls.append(("directory", directory))

        def setOption(self, option) -> None:
            calls.append(("option", option))

        def exec(self) -> int:
            return self.exec_result

        def selectedFiles(self) -> list[str]:
            return [str(tmp_path / "selected.itws")]

    monkeypatch.setattr(QtWidgets, "QFileDialog", _FakeFileDialog)
    with manager_context() as manager:
        assert (
            manager._workspace_controller._workspace_save_dialog(
                native=False, selected_file=tmp_path / "explicit.itws"
            )
            is None
        )
        assert ("select", str(tmp_path / "explicit.itws")) in calls

        _FakeFileDialog.exec_result = 1
        manager._workspace_state.path = tmp_path / "bound.itws"
        assert manager._workspace_controller._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("select", str(tmp_path / "bound.itws")) in calls

        manager._workspace_state.path = None
        manager._recent_directory = None
        default_dir = tmp_path / "default"
        default_dir.mkdir()
        erlab.interactive.options.model = AppOptions().model_copy(
            update={
                "io": AppOptions().io.model_copy(
                    update={"default_directory": str(default_dir)}
                )
            }
        )
        assert manager._workspace_controller._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("directory", str(default_dir)) in calls

        manager._recent_directory = str(tmp_path)
        assert manager._workspace_controller._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("directory", str(tmp_path)) in calls


def test_manager_dirty_workspace_save_choice_save_branch(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = pathlib.Path("dirty.itws")
        manager._workspace_state.structure_modified = True
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
        )

        assert (
            manager._workspace_controller._dirty_workspace_save_choice(
                "Save before continuing."
            )
            == "save"
        )


def test_manager_legacy_itws_schema_save_helpers(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Ok,
        )
        manager._workspace_controller._show_legacy_workspace_upgrade_message(
            tmp_path / "legacy-schema.itws"
        )

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: None,
        )
        assert (
            manager._workspace_controller._save_legacy_workspace_as_v4(
                tmp_path / "legacy-schema.itws"
            )
            is None
        )

        dirty_reasons: list[str] = []
        monkeypatch.setattr(
            manager._workspace_controller,
            "_save_legacy_workspace_as_v4",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_mark_workspace_structure_dirty",
            dirty_reasons.append,
        )
        manager._workspace_controller._associate_loaded_workspace_file(
            tmp_path / "legacy-schema.itws",
            workspace_format._WORKSPACE_LEGACY_SCHEMA_VERSION - 1,
        )

        assert manager._workspace_state.path is None
        assert manager._workspace_state.needs_full_save
        assert dirty_reasons == ["Legacy workspace needs conversion"]


class _DeferredWorkspaceSaveWorker:
    def __init__(
        self,
        _fname: str | os.PathLike[str],
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
    ) -> None:
        self.signals = workspace_saving._WorkspaceSaveWorkerSignals()
        self._snapshot = snapshot

    def finish(
        self,
        *,
        elapsed: float = 0.0,
        error: workspace_saving._WorkspaceSaveError | None = None,
    ) -> None:
        self._snapshot.close()
        self.signals.finished.emit(elapsed, error)


class _DeferredWorkspaceSaveThreadPool:
    def __init__(self) -> None:
        self.workers: list[_DeferredWorkspaceSaveWorker] = []

    def start(self, worker: _DeferredWorkspaceSaveWorker) -> None:
        self.workers.append(worker)


def _install_deferred_workspace_save_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> _DeferredWorkspaceSaveThreadPool:
    pool = _DeferredWorkspaceSaveThreadPool()
    monkeypatch.setattr(
        QtCore.QThreadPool, "globalInstance", staticmethod(lambda: pool)
    )
    monkeypatch.setattr(
        workspace_saving, "_WorkspaceSaveWorker", _DeferredWorkspaceSaveWorker
    )
    return pool


def _bind_dirty_workspace_for_save_test(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    tmp_path: pathlib.Path,
) -> pathlib.Path:
    fname = tmp_path / "background-save.itws"
    fname.touch()
    manager._workspace_state.path = fname.resolve()
    manager._workspace_controller._mark_workspace_clean()
    manager._workspace_state.mark_layout_dirty()
    return fname


def _workspace_save_test_snapshot(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
) -> workspace_saving._WorkspaceSaveSnapshot:
    return workspace_saving._WorkspaceSaveSnapshot(
        generation=manager._workspace_state.dirty_generation,
        root_attrs={},
        delta_save_count=manager._workspace_state.delta_save_count + 1,
    )


def _compact_workspace_before_shutdown_and_wait(
    qtbot,
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
) -> bool:
    completed = False

    def _mark_completed() -> None:
        nonlocal completed
        completed = True

    requested = manager._workspace_controller._compact_workspace_before_shutdown(
        on_finished=_mark_completed
    )
    if requested:
        qtbot.wait_until(lambda: completed, timeout=10000)
    return requested


def test_workspace_save_worker_reports_missing_backing_source(tmp_path) -> None:
    missing_source = tmp_path / "deleted-source.itws"
    target = tmp_path / "target.itws"
    snapshot = workspace_saving._WorkspaceSaveSnapshot(
        generation=0,
        root_attrs=_transaction_test_root_attrs(),
        delta_save_count=0,
        full_tree=xr.DataTree(),
        copy_group_sources=(
            (
                str(missing_source),
                "0/imagetool",
                "0/imagetool",
                None,
            ),
        ),
    )
    worker = workspace_saving._WorkspaceSaveWorker(target, snapshot)
    results: list[tuple[float, workspace_saving._WorkspaceSaveError | None]] = []
    receiver = workspace_saving._WorkspaceSaveResultReceiver(
        callback=lambda elapsed, error: results.append((elapsed, error)),
        parent=worker.signals,
    )
    worker.signals.finished.connect(receiver.finish)

    worker.run()

    assert len(results) == 1
    _elapsed, error = results[0]
    assert isinstance(error, workspace_saving._WorkspaceSaveError)
    assert error.missing_source_path == str(missing_source)
    assert error.traceback_text


def test_manager_async_save_request_error_paths(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _critical(*args, **kwargs) -> int:
        critical_calls.append(args)
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)
    with manager_context() as manager:
        manager._show_workspace_save_worker_error(
            workspace_saving._WorkspaceSaveError("Traceback text")
        )
        assert critical_calls[-1][2] == (
            "An error occurred while saving the workspace file."
        )

        missing_error = workspace_saving._WorkspaceSaveError(
            traceback_text="Missing source traceback",
            missing_source_path=str(tmp_path / "deleted-source.itws"),
        )
        manager._show_workspace_save_worker_error(missing_error)
        assert len(critical_calls) == 2

        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.save_in_progress = True
        assert not manager._workspace_controller.save()

        manager._workspace_state.save_in_progress = False
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            lambda _origin: None,
        )
        callback_results: list[bool] = []
        assert not manager._workspace_controller.save(
            on_finished=callback_results.append
        )
        assert callback_results == [False]
        assert operation_errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: None,
        )
        assert not manager._workspace_controller.save_as()


def test_manager_save_action_runs_workspace_save_in_background(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()
        fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        assert manager.workspace_path == str(fname.resolve())
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "save_as",
            lambda **_kwargs: pytest.fail("Save action unexpectedly used Save As"),
        )

        manager.save_action.trigger()

        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        assert not manager.save_action.isEnabled()
        assert not manager.save_as_action.isEnabled()
        assert not manager.compact_workspace_action.isEnabled()
        assert not manager.import_workspace_action.isEnabled()
        assert manager.is_workspace_modified
        manager._update_actions()
        assert not manager.offload_action.isEnabled()
        manager.tree_view.deselect_all()
        manager._update_actions()
        assert not manager.offload_action.isEnabled()

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert manager.save_action.isEnabled()
        assert manager.save_as_action.isEnabled()
        assert manager.compact_workspace_action.isEnabled()
        assert manager.import_workspace_action.isEnabled()
        assert not manager.offload_action.isEnabled()
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_workspace_save_keeps_new_changes_and_queues_followup(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager._workspace_controller.save()
        manager._workspace_controller._mark_workspace_options_dirty()
        assert not manager._workspace_controller.save()

        assert len(pool.workers) == 1
        pool.workers[0].finish()
        qtbot.wait_until(lambda: len(pool.workers) == 2)
        assert manager._workspace_state.save_in_progress
        assert manager.is_workspace_modified

        pool.workers[1].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_workspace_save_keeps_duplicate_layout_change_dirty(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager._workspace_controller.save()
        assert len(pool.workers) == 1
        snapshot_generation = pool.workers[0]._snapshot.generation
        assert manager._workspace_state.mark_layout_dirty()
        assert manager._workspace_state.dirty_generation > snapshot_generation

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert manager.is_workspace_modified
        assert manager._workspace_state.layout_modified
        manager._workspace_state.path = None


def test_manager_background_full_save_preserves_post_snapshot_data_edit(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=("x", "y"),
            coords={"x": np.arange(5.0), "y": np.arange(5.0)},
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "post-snapshot-edit.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            fname, targets=[0], chunks={}
        )
        slicer_area = root.slicer_area
        assert slicer_area.data_chunked
        manager._workspace_controller._mark_workspace_clean()

        manager._workspace_controller._mark_node_data_dirty(uid)
        manager._workspace_state.needs_full_save = True
        assert manager._workspace_controller.save()
        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        snapshot_generation = pool.workers[0]._snapshot.generation

        replacement = xr.DataArray(
            np.full((5, 5), 42.0),
            dims=("x", "y"),
            coords={"x": np.arange(5.0), "y": np.arange(5.0)},
            name="source",
        )
        slicer_area.replace_source_data(
            replacement,
            auto_compute=False,
            emit_edited=True,
        )
        assert any(
            event.uid == uid and event.data and event.generation > snapshot_generation
            for event in manager._workspace_state.dirty_events
        )

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        np.testing.assert_array_equal(slicer_area._data.values, replacement.values)
        assert not slicer_area.data_chunked
        assert manager.is_workspace_modified
        assert uid in manager._workspace_state.dirty_data
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_workspace_save_failure_restores_state(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        errors: list[workspace_saving._WorkspaceSaveError] = []
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        monkeypatch.setattr(manager, "_show_workspace_save_worker_error", errors.append)

        assert manager._workspace_controller.save()
        assert manager._workspace_state.save_in_progress

        pool.workers[0].finish(
            error=workspace_saving._WorkspaceSaveError("worker boom"),
        )
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert errors
        assert "worker boom" in errors[-1].traceback_text
        assert manager.save_action.isEnabled()
        assert manager.save_as_action.isEnabled()
        assert manager.compact_workspace_action.isEnabled()
        assert manager.import_workspace_action.isEnabled()
        assert manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_state.path = None


def test_manager_save_slot_requests_async_save(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager.save() is None
        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        assert manager.is_workspace_modified

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert not manager._workspace_state.save_in_progress
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_close_ignored_while_workspace_save_in_progress(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        event = QtGui.QCloseEvent()
        try:
            manager.closeEvent(event)
            assert not event.isAccepted()
        finally:
            manager._workspace_state.save_in_progress = False


def test_open_multiple_files_workspace_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-dropped.itws"
    _write_transaction_test_workspace(fname)
    lock = workspace_storage._acquire_workspace_document_lock(fname)
    lock_calls: list[pathlib.Path] = []
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            workspace_storage,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Dropped workspace should lock before recovery")
            ),
        )
        monkeypatch.setattr(
            workspace_controller,
            "_show_workspace_file_lock_error",
            lambda _parent, locked_fname: lock_calls.append(pathlib.Path(locked_fname)),
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "file_loaders",
            lambda *args, **kwargs: pytest.fail(
                "locked workspace should not fall through to loaders"
            ),
        )

        with manager_context() as manager:
            manager._data_ingress.open_multiple_files([fname], try_workspace=True)
    finally:
        lock.unlock()

    assert recovery_calls == []
    assert lock_calls == [fname]


def test_manager_workspace_v4_save_open_replaces_and_binds_path(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        fname = tmp_path / "bound.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            assert h5_file.attrs["imagetool_workspace_schema_version"] == 4
            manifest = json.loads(h5_file.attrs["imagetool_workspace_manifest"])
        assert manifest["schema_version"] == 4
        assert {node["uid"] for node in manifest["nodes"]} >= {
            manager._tool_graph.root_wrappers[0].uid,
            child_uid,
        }

        extra = itool(data + 2, manager=False, execute=False)
        assert isinstance(extra, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(extra, show=False)
        assert manager.ntools == 2

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == [child_uid]
        assert manager.get_imagetool(0).slicer_area._data.chunks is None
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0


def test_manager_workspace_import_ignored_while_save_in_progress(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        monkeypatch.setattr(
            QtWidgets,
            "QFileDialog",
            lambda *_args, **_kwargs: pytest.fail(
                "Import should not open a file dialog during workspace save"
            ),
        )

        assert not manager.import_workspace(native=False)

        manager._workspace_state.save_in_progress = False


def test_manager_workspace_load_ignored_while_save_in_progress(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        monkeypatch.setattr(
            QtWidgets,
            "QFileDialog",
            lambda *_args, **_kwargs: pytest.fail(
                "Open should not show a file dialog during workspace save"
            ),
        )

        assert not manager.load(native=False)

        manager._workspace_state.save_in_progress = False


def test_manager_workspace_save_as_preserves_live_in_memory_windows(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            child = itool(data + 1.0, manager=False, execute=False)
            assert isinstance(child, erlab.interactive.imagetool.ImageTool)
            child_uid = manager.add_imagetool_child(child, 0, show=False)

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager._workspace_controller.loading,
                "_load_workspace_file",
                _load_workspace_file_should_not_run,
            )

            new_fname = tmp_path / "new.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

            assert manager.workspace_path == str(new_fname.resolve())
            assert not manager.is_workspace_modified
            assert manager.get_imagetool(0) is root
            assert manager._child_node(child_uid).imagetool is child
            assert manager._tool_graph.root_wrappers[0]._childtool_indices == [
                child_uid
            ]
            assert root.slicer_area._data.chunks is None
            assert child.slicer_area._data.chunks is None
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_offload_to_workspace_save_as_rebinds_root_as_dask(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)), dims=["x", "y"], name="source"
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()
        assert root.slicer_area._data.chunks is None

        fname = tmp_path / "offload.itws"

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(fname.name)

        results: list[bool] = []
        accept_dialog(
            lambda: results.append(manager.offload_to_workspace([0], native=False)),
            pre_call=_go_to_file,
        )

        assert results == [True]
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert workspace_arrays._normalized_file_path(
            rebound.encoding.get("source")
        ) == (str(fname.resolve()))
        assert _compute_first_value(rebound) == 0.0

        manager._update_actions()
        assert not manager.offload_action.isEnabled()


def test_manager_workspace_load_reopens_offloaded_data_as_dask(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "offload-reopen.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is not None
        assert workspace_arrays._normalized_file_path(
            loaded.encoding.get("source")
        ) == (str(fname.resolve()))
        assert _compute_first_value(loaded) == 0.0


def test_manager_workspace_import_reopens_offloaded_data_as_dask(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "offload-import.itws"
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        loaded: list[bool] = []
        accept_dialog(
            lambda: loaded.append(
                manager._workspace_controller.loading._load_workspace_file(
                    fname,
                    replace=False,
                    associate=False,
                    mark_dirty=True,
                    select=True,
                )
            )
        )

        assert loaded == [True]
        loaded_data = manager.get_imagetool(0).slicer_area._data
        assert loaded_data.chunks is not None
        assert workspace_arrays._normalized_file_path(
            loaded_data.encoding.get("source")
        ) == str(fname.resolve())
        assert _compute_first_value(loaded_data) == 0.0


def test_manager_workspace_load_reopens_offloaded_spaced_coord_data_as_dask(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": np.arange(5.0),
                "y": np.arange(5.0),
                "Fake Motor": ("x", np.linspace(10.0, 20.0, 5)),
            },
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "offload-spaced-coord.itws"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
        assert not any("space in its name" in str(item.message) for item in caught)
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is not None
        assert "Fake Motor" in loaded.coords
        np.testing.assert_allclose(
            np.asarray(loaded.coords["Fake Motor"]),
            np.asarray(data.coords["Fake Motor"]),
        )


def test_manager_offload_to_workspace_saves_dirty_workspace_before_rebind(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "dirty-offload.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        updated = data + 10.0
        root.slicer_area.replace_source_data(
            updated, auto_compute=False, emit_edited=True
        )
        uid = manager._tool_graph.root_wrappers[0].uid
        assert ("snapshot-token-refresh", uid) in manager._interaction_gate.pending_keys
        assert manager.is_workspace_modified

        original_save = manager._workspace_controller.save
        save_calls: list[bool] = []

        def _save(
            *,
            native: bool = True,
            on_finished: Callable[[bool], None] | None = None,
        ) -> bool:
            save_calls.append(native)
            return original_save(native=native, on_finished=on_finished)

        monkeypatch.setattr(manager._workspace_controller, "save", _save)

        assert manager.offload_to_workspace([0], native=False)
        assert save_calls == [False]
        qtbot.wait_until(
            lambda: root.slicer_area._data.chunks is not None,
            timeout=30000,
        )

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert _compute_first_value(rebound) == 10.0
        assert (
            "snapshot-token-refresh",
            uid,
        ) not in manager._interaction_gate.pending_keys
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 10.0


def test_manager_compute_offloaded_workspace_data_marks_backing_dirty(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "compute-offloaded.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert not manager.is_workspace_modified

        root.slicer_area._compute_chunked()

        assert root.slicer_area._data.chunks is None
        assert uid in manager._workspace_state.dirty_data
        assert manager.is_workspace_modified

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert saved.chunks is None


def test_manager_offload_to_workspace_save_cancel_or_failure_noop(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: None,
        )
        assert not manager.offload_to_workspace([0], native=False)
        assert manager.workspace_path is None
        assert root.slicer_area._data.chunks is None

        fname = tmp_path / "failure-offload.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(
            manager._tool_graph.root_wrappers[0].uid
        )

        monkeypatch.setattr(
            manager._workspace_controller, "save", lambda **_kwargs: False
        )
        assert not manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is None


def test_manager_offload_to_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[list[int]] = []
        monkeypatch.setattr(manager, "_selected_imagetool_targets", lambda: [0])
        monkeypatch.setattr(
            manager,
            "offload_to_workspace",
            lambda targets: calls.append(list(targets)) or True,
        )
        manager.offload_selected_to_workspace()
        assert calls == [[0]]

    with manager_context() as manager:
        assert not manager.offload_to_workspace([])

    fake_node = types.SimpleNamespace(
        is_imagetool=True,
        imagetool=object(),
        slicer_area=types.SimpleNamespace(data_chunked=False),
        pending_workspace_memory_payload=None,
    )

    with manager_context() as manager:
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: fake_node)
        monkeypatch.setattr(
            manager._workspace_controller, "save_as", lambda **_kwargs: False
        )
        assert not manager.offload_to_workspace([0], native=False)

    with manager_context() as manager:
        workspace = tmp_path / "offload-error.itws"
        manager._workspace_state.path = workspace
        manager._workspace_state.needs_full_save = False
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: fake_node)
        monkeypatch.setattr(
            manager._workspace_controller,
            "_active_managed_window",
            lambda: typing.cast("typing.Any", None),
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *_args, **_kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager._workspace_controller.loading,
            "_rebind_workspace_backed_imagetools",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        errors: list[tuple[str, str]] = []
        restored: list[object | None] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, message: errors.append((title, message)),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            restored.append,
        )

        assert not manager.offload_to_workspace([0], native=False)
        assert errors == [
            (
                "Error while offloading to workspace",
                "An error occurred while reconnecting data from the workspace file.",
            )
        ]
        assert restored == [None]


def test_manager_offload_to_workspace_preserves_child_source_state(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False, provenance_spec=full_data())

        child = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)

        fname = tmp_path / "child-offload.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert manager.get_imagetool(0).slicer_area._data.chunks is not None
        assert child_node.source_state == "fresh"
        assert child.slicer_area._data.chunks is None

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.offload_action.isEnabled()

        assert manager.offload_to_workspace([child_uid], native=False)
        assert child.slicer_area._data.chunks is not None
        assert child_node.source_state == "fresh"
        assert _compute_first_value(child.slicer_area._data) == 0.0

        manager._update_actions()
        assert not manager.offload_action.isEnabled()


def test_manager_manual_chunk_edits_persist_on_next_workspace_save(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "manual-chunks.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        root.slicer_area._set_chunks({"x": 2, "y": 3})

        assert root.slicer_area._data.chunks == ((2, 2, 1), (3, 2))
        assert uid in manager._workspace_state.dirty_data
        assert manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert saved.chunks is None

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert saved.chunks == (2, 3)

        opened = workspace_arrays.open_workspace_dataset(
            fname,
            manager._workspace_controller.saving._workspace_payload_path(uid),
            chunks={},
        )
        try:
            rebound = opened[_ITOOL_DATA_NAME]
            assert rebound.chunks == ((2, 2, 1), (3, 2))
        finally:
            opened.close()


def test_manager_workspace_full_save_preserves_non_dask_data(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            assert root.slicer_area._data.chunks is None

            fname = tmp_path / "full-save.itws"
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
            adopt_workspace_path(manager, fname)
            manager._workspace_controller._mark_workspace_clean()
            manager._workspace_state.needs_full_save = True

            assert _request_workspace_save_and_wait(qtbot, manager)
            assert root.slicer_area._data.chunks is None
            assert _compute_first_value(root.slicer_area._data) == 0.0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_save_as_preserves_external_non_dask_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        old_fname = tmp_path / "old.h5"
        new_fname = tmp_path / "new.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            old_fname, engine="h5netcdf", invalid_netcdf=True
        )
        source = _open_external_file_backed_hdf5_imagetool_data(old_fname)
        assert source.chunks is None

        root = itool(source, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        live_data = manager.get_imagetool(0).slicer_area._data
        old_source = str(old_fname.resolve())
        assert (
            workspace_arrays._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        rebound_source = workspace_arrays._normalized_file_path(
            rebound.encoding.get("source")
        )
        assert rebound_source == old_source
        assert rebound_source != new_source
        assert rebound.chunks is None

        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_rebinds_workspace_non_dask_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        old_fname = tmp_path / "old.itws"
        new_fname = tmp_path / "new.itws"
        manager._workspace_controller.saving._save_workspace_document(
            old_fname, force_full=True
        )
        adopt_workspace_path(manager, old_fname)
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            old_fname, targets=[0], chunks=None
        )

        live_data = manager.get_imagetool(0).slicer_area._data
        old_source = str(old_fname.resolve())
        assert (
            workspace_arrays._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )
        assert live_data.chunks is None

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        assert (
            workspace_arrays._normalized_file_path(rebound.encoding.get("source"))
            == new_source
        )
        assert rebound.chunks is None

        old_fname.unlink()
        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_preserves_manually_chunked_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        old_fname = tmp_path / "old.h5"
        new_fname = tmp_path / "new.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            old_fname, engine="h5netcdf", invalid_netcdf=True
        )
        source = _open_external_file_backed_hdf5_imagetool_data(old_fname)
        assert source.chunks is None

        root = itool(source, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.slicer_area.replace_source_data(
            root.slicer_area._data.chunk({"x": 2, "y": 2}),
            auto_compute=False,
        )
        assert root.slicer_area._data.chunks is not None

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert workspace_arrays._normalized_file_path(
            rebound.encoding.get("source")
        ) == str(new_fname.resolve())

        old_fname.unlink()
        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_rebinds_lazy_data_to_new_document(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            old_fname = tmp_path / "old.h5"
            new_fname = tmp_path / "new.itws"
            xr.DataTree.from_dict(
                {"0/imagetool": data.to_dataset(name="data")}
            ).to_netcdf(old_fname, engine="h5netcdf", invalid_netcdf=True)
            old_lazy = _open_external_lazy_hdf5_imagetool_data(old_fname)
            root.slicer_area.replace_source_data(old_lazy + 0, auto_compute=False)
            assert _compute_first_value(old_lazy) == 0

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager._workspace_controller.loading,
                "_load_workspace_file",
                _load_workspace_file_should_not_run,
            )

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

            assert manager.workspace_path == str(new_fname.resolve())
            rebound = manager.get_imagetool(0).slicer_area._data
            assert rebound.chunks is not None
            assert workspace_arrays._normalized_file_path(
                rebound.encoding.get("source")
            ) == str(new_fname.resolve())
            old_fname.unlink()
            assert _compute_first_value(rebound) == 0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_save_clears_deferred_dirty_events(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "deferred-dirty.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        QtCore.QTimer.singleShot(
            0, lambda: manager._workspace_controller._mark_node_state_dirty(uid)
        )
        manager._workspace_controller._mark_node_state_dirty(uid)
        assert manager.is_workspace_modified
        assert not root.isWindowModified()

        manager._flush_idle_work(force=True)

        assert root.isWindowModified()

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            manager._workspace_controller, "_active_managed_window", lambda: root
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        manager._workspace_controller._drain_workspace_deferred_events()
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()
        assert focus_restored == [root]


def test_manager_workspace_save_during_active_interaction_uses_dirty_state(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "active-save.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        manager._note_interaction_activity()
        manager._workspace_controller._mark_node_state_dirty(uid)

        assert manager.is_workspace_modified
        assert not root.isWindowModified()

        assert _request_workspace_save_and_wait(qtbot, manager)

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()

        manager._flush_idle_work(force=True)

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_workspace_save_drain_does_not_force_deferred_delete(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager, monkeypatch.context() as save_drain_patch:
        event_types: list[int] = []
        idle_flushes: list[dict[str, object]] = []
        save_drain_patch.setattr(
            QtWidgets.QApplication,
            "sendPostedEvents",
            lambda _receiver, event_type: event_types.append(event_type),
        )
        save_drain_patch.setattr(
            QtWidgets.QApplication, "processEvents", lambda *_args, **_kwargs: None
        )
        save_drain_patch.setattr(
            manager, "_flush_idle_work", lambda **kwargs: idle_flushes.append(kwargs)
        )

        manager._workspace_controller._drain_workspace_deferred_events()

        assert event_types == [int(QtCore.QEvent.Type.MetaCall.value)] * 6
        assert idle_flushes == [{"force": True}]


def test_manager_workspace_state_save_updates_attrs_without_full_rewrite(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "state-delta.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(uid)

        original_transaction_write = workspace_storage._write_workspace_transaction_file
        attr_write_calls: list[str] = []

        def _record_transaction_write(
            _fname: str | os.PathLike[str],
            rewrite_groups: Iterable[tuple[str, dict[str, xr.Dataset]]],
            attr_updates: Iterable[
                tuple[
                    str,
                    dict[str, typing.Any],
                    tuple[str, dict[str, xr.Dataset]],
                ]
            ],
            root_attrs: Mapping[str, typing.Any],
            **kwargs: typing.Any,
        ) -> None:
            rewrite_groups = tuple(rewrite_groups)
            updates = tuple(attr_updates)
            assert rewrite_groups == ()
            attr_write_calls.extend(update[0] for update in updates)
            original_transaction_write(
                _fname, rewrite_groups, updates, root_attrs, **kwargs
            )

        monkeypatch.setattr(
            workspace_storage,
            "_write_full_workspace_tree_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only Save should not rewrite the full workspace"
            ),
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_root_attrs_to_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only Save should batch root attrs with node attrs"
            ),
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_transaction_file",
            _record_transaction_write,
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert attr_write_calls == ["0/imagetool"]
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_workspace_save_does_not_close_live_workspace_handles(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "live-handles.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(uid)

        assert _request_workspace_save_and_wait(qtbot, manager)

        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)


def test_manager_workspace_save_preserves_live_lazy_readers_during_write(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "live-lazy.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        root = manager.get_imagetool(0)
        uid = manager._tool_graph.root_wrappers[0].uid
        root.slicer_area.replace_source_data(
            manager._workspace_controller.loading._workspace_rebind_data_for_uid(
                fname, uid, chunks="auto"
            ),
            auto_compute=False,
        )
        live_data = root.slicer_area._data
        assert live_data.chunks is not None
        assert _compute_first_value(live_data) == 0.0

        manager._workspace_controller._mark_node_state_dirty(uid)
        original_write = workspace_storage._write_workspace_transaction_file
        computed_values: list[object] = []

        def _slow_write_workspace_transaction_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        def _compute_live_data() -> None:
            computed_values.append(live_data.isel({"x": 1, "y": 1}).compute().item())

        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_transaction_file",
            _slow_write_workspace_transaction_file,
        )
        QtCore.QTimer.singleShot(10, _compute_live_data)

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert computed_values == [6.0]


def test_manager_workspace_slow_save_reports_status_after_background_write(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "slow-save.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(
            manager._tool_graph.root_wrappers[0].uid
        )

        original_write = workspace_storage._write_workspace_transaction_file

        def _slow_write_workspace_transaction_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._workspace._controller,
            "_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_transaction_file",
            _slow_write_workspace_transaction_file,
        )
        monkeypatch.setattr(
            manager._workspace_controller, "_active_managed_window", lambda: root
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._status_bar.currentMessage().startswith("Workspace saved")
        assert focus_restored == [root]


def test_manager_workspace_save_keeps_post_command_changes_dirty(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "post-command-dirty.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)

        assert manager._workspace_controller.save()
        manager._workspace_controller._mark_node_state_dirty(uid)
        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert manager.is_workspace_modified
        assert root.isWindowModified()
        details = manager._workspace_controller._dirty_details_text()
        assert "State modified:" in details
        assert "Data modified:" not in details


def test_manager_workspace_compact_resets_delta_count_and_cleans_internal_groups(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "compact.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest["estimated_obsolete_bytes"] == 0
        assert manifest["replacement_delta_count"] == 0

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )

        assert manager.compact_workspace()
        assert manager._workspace_state.delta_save_count == 0
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        _assert_no_workspace_internal_groups(fname)
        with h5py.File(fname, "r") as h5_file:
            assert (
                workspace_format._workspace_delta_save_count_from_attrs(h5_file.attrs)
                == 0
            )
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_delta_save_tracks_repack_benefit(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "delta-benefit.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)

        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.estimated_obsolete_bytes > 0
        assert manager._workspace_state.replacement_delta_count == 1
        assert manager._workspace_state.repack_estimate_known
        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert (
            manifest["estimated_obsolete_bytes"]
            == manager._workspace_state.estimated_obsolete_bytes
        )
        assert manifest["replacement_delta_count"] == 1

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        assert manager._workspace_state.delta_save_count == 0
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert "estimated_obsolete_bytes" not in manifest
        assert "replacement_delta_count" not in manifest


def test_manager_workspace_shutdown_compacts_clean_delta_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-compact.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.estimated_obsolete_bytes > 0
        assert manager._workspace_state.replacement_delta_count == 1

        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )

        assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)

        assert manager._workspace_state.delta_save_count == 0
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        _assert_no_workspace_internal_groups(fname)
        with h5py.File(fname, "r") as h5_file:
            assert (
                workspace_format._workspace_delta_save_count_from_attrs(h5_file.attrs)
                == 0
            )
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_shutdown_compact_uses_file_level_repack(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-file-repack.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.replacement_delta_count == 1

        def _fail_to_datatree() -> xr.DataTree:
            pytest.fail("Shutdown file-level repack should not serialize tools")

        monkeypatch.setattr(
            manager._workspace_controller.saving, "_to_datatree", _fail_to_datatree
        )
        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )

        assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)

        assert manager._workspace_state.delta_save_count == 0
        _assert_no_workspace_internal_groups(fname)
        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_shutdown_compact_skips_state_only_delta(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-state-skip.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.replacement_delta_count == 0

        monkeypatch.setattr(
            manager._workspace_controller,
            "_start_workspace_save_worker",
            lambda *args, **kwargs: pytest.fail(
                "State-only shutdown compaction should skip"
            ),
        )

        assert not manager._workspace_controller._compact_workspace_before_shutdown()

        assert manager._workspace_state.delta_save_count == 1


def test_manager_workspace_shutdown_compact_skips_below_threshold_data_delta(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-threshold-skip.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.estimated_obsolete_bytes > 0
        assert manager._workspace_state.replacement_delta_count == 1

        monkeypatch.setattr(
            manager._workspace_controller,
            "_start_workspace_save_worker",
            lambda *args, **kwargs: pytest.fail(
                "Below-threshold shutdown compaction should skip"
            ),
        )

        assert not manager._workspace_controller._compact_workspace_before_shutdown()

        assert manager._workspace_state.delta_save_count == 1


def test_manager_workspace_shutdown_compact_scans_when_estimate_missing(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-missing-estimate.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        with h5py.File(fname, "a") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
            manifest.pop("estimated_obsolete_bytes", None)
            manifest.pop("replacement_delta_count", None)
            h5_file.attrs[workspace_format._WORKSPACE_MANIFEST_ATTR] = json.dumps(
                manifest
            )
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=0,
            replacement_delta_count=0,
            known=False,
        )

        scan_calls: list[pathlib.Path] = []

        def _fake_estimate(path: str | os.PathLike[str]) -> int:
            scan_calls.append(pathlib.Path(path))
            return 100

        monkeypatch.setattr(
            workspace_storage,
            "_workspace_obsolete_estimate",
            _fake_estimate,
        )
        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )

        assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)

        assert scan_calls == [fname.resolve()]
        assert manager._workspace_state.delta_save_count == 0


def test_manager_workspace_shutdown_compact_skips_if_file_repack_unavailable(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-fallback-repack.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1

        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_file_repack_snapshot",
            lambda _generation: None,
        )

        def _fail_full_save_snapshot(
            _generation: int,
        ) -> workspace_saving._WorkspaceSaveSnapshot:
            pytest.fail("Shutdown compaction should not serialize as a fallback")

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_full_save_snapshot",
            _fail_full_save_snapshot,
        )

        assert not manager._workspace_controller._compact_workspace_before_shutdown()

        assert manager._workspace_state.delta_save_count == 1


def test_manager_workspace_shutdown_compact_runs_in_background(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "slow-shutdown-compact.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1

        original_write = workspace_storage._write_full_workspace_tree_file

        def _slow_write_full_workspace_tree_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._workspace._controller,
            "_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            workspace_saving,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_full_workspace_tree_file",
            _slow_write_full_workspace_tree_file,
        )
        assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 0


def test_manager_workspace_shutdown_compact_skips_dirty_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "dirty-shutdown-compact.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_state.delta_save_count = 1
        manager._workspace_controller._mark_node_state_dirty(uid)

        monkeypatch.setattr(
            manager._workspace_controller,
            "_start_workspace_save_worker",
            lambda *args, **kwargs: pytest.fail(
                "Dirty shutdown compaction should not write discarded changes"
            ),
        )

        assert not manager._workspace_controller._compact_workspace_before_shutdown()


def test_manager_workspace_high_risk_path_forces_full_save_snapshot(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "high-risk.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(uid)
        monkeypatch.setattr(
            workspace_storage, "_workspace_path_is_high_risk", lambda *_args: True
        )
        monkeypatch.setattr(
            workspace_storage,
            "_workspace_path_is_likely_network_path",
            lambda *_args: True,
        )

        snapshot = manager._workspace_controller.saving._workspace_save_snapshot(fname)
        try:
            assert snapshot.full_tree is not None
            assert snapshot.delta_save_count == 0
            assert snapshot.copy_groups == ()
        finally:
            snapshot.close()


def test_manager_workspace_save_snapshot_uses_compression_override(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        manager._set_workspace_option_overrides(
            {"io/workspace/compression": "blosclz3"}
        )
        assert (
            manager._workspace_controller.saving._workspace_compression_mode()
            == "blosclz3"
        )
        manager._workspace_state.delta_save_count = 2
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=128,
            replacement_delta_count=3,
            known=False,
        )
        root_attrs = (
            manager._workspace_controller.saving._workspace_root_attrs_payload()
        )
        manifest = workspace_format._workspace_manifest_from_attrs(root_attrs)
        assert manifest["delta_save_count"] == 2
        assert "estimated_obsolete_bytes" not in manifest
        root_attrs = manager._workspace_controller.saving._workspace_root_attrs_payload(
            delta_save_count=1,
            estimated_obsolete_bytes=1024,
            replacement_delta_count=5,
            repack_estimate_known=True,
        )
        manifest = workspace_format._workspace_manifest_from_attrs(root_attrs)
        assert manifest["estimated_obsolete_bytes"] == 1024
        assert manifest["replacement_delta_count"] == 5

        snapshot = manager._workspace_controller.saving._workspace_save_snapshot(
            tmp_path / "snapshot.itws"
        )
        try:
            assert snapshot.compression_mode == "blosclz3"
        finally:
            snapshot.close()


def test_manager_workspace_delta_save_updates_repack_state(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        writes: list[tuple[object, ...]] = []

        def _snapshot(
            generation: int,
            root_attrs: dict[str, typing.Any],
            delta_save_count: int,
        ) -> workspace_saving._WorkspaceSaveSnapshot:
            return workspace_saving._WorkspaceSaveSnapshot(
                generation=generation,
                root_attrs=root_attrs,
                delta_save_count=delta_save_count,
                estimated_obsolete_bytes=256,
                replacement_delta_count=4,
                rewrite_groups=(),
                attr_updates=(),
            )

        def _record_write(*args, **kwargs) -> None:
            writes.append((*args, kwargs))

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_delta_save_snapshot",
            _snapshot,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_transaction_file",
            _record_write,
        )

        manager._workspace_state.dirty_generation = 5
        manager._workspace_state.delta_save_count = 6
        manager._workspace_controller.saving._save_workspace_delta(
            tmp_path / "delta.itws"
        )

        assert writes
        assert manager._workspace_state.delta_save_count == 7
        assert manager._workspace_state.estimated_obsolete_bytes == 256
        assert manager._workspace_state.replacement_delta_count == 4


def test_manager_workspace_delta_snapshot_keeps_replacement_count_for_missing_groups(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        saver = controller.saving

        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=100,
            replacement_delta_count=2,
        )

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_highest_dirty_data_roots",
            lambda: ("uid",),
        )
        monkeypatch.setattr(
            controller.saving,
            "_workspace_rewrite_group_snapshot",
            lambda _uid: ("0", {}),
        )
        monkeypatch.setattr(manager, "_iter_descendant_uids", lambda _uid: ())
        monkeypatch.setattr(
            controller.saving,
            "_workspace_manifest_node_uids",
            lambda _root_attrs: frozenset(),
        )
        monkeypatch.setattr(
            controller.saving,
            "_workspace_stale_reference_rewrite_uids",
            lambda _manifest_uids: (),
        )
        monkeypatch.setattr(
            workspace_arrays,
            "_workspace_h5_paths_storage_size",
            lambda *_args, **_kwargs: (256, 0),
        )

        snapshot = saver._workspace_delta_save_snapshot(1, {}, 3)

        assert snapshot.estimated_obsolete_bytes == 356
        assert snapshot.replacement_delta_count == 2


def test_manager_write_full_workspace_file_can_disable_group_reuse(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        saver = controller.saving
        tree = xr.DataTree()
        writes: list[dict[str, typing.Any]] = []

        monkeypatch.setattr(
            manager._workspace_controller.saving, "_to_datatree", lambda: tree
        )
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_full_save_copy_groups",
            lambda *_args, **_kwargs: pytest.fail("copy groups should not be reused"),
        )

        def _record_write(*_args, **kwargs) -> None:
            writes.append(kwargs)

        monkeypatch.setattr(
            workspace_storage,
            "_write_full_workspace_tree_file",
            _record_write,
        )

        saver._write_full_workspace_file(
            tmp_path / "full.itws", reuse_unchanged_groups=False
        )

        assert writes
        assert writes[0]["copy_source"] is None
        assert writes[0]["copy_groups"] == ()


def test_workspace_manifest_first_helper_edge_cases(
    qtbot,
    tmp_path,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        saver = controller.saving
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        with monkeypatch.context() as mp:
            mp.setattr(saver, "_workspace_node_path", lambda uid: uid)
            assert not saver._workspace_datatree_for_payload_uids(("missing",)).children
        assert (
            saver._workspace_full_save_manifest_entries(
                {
                    workspace_format._WORKSPACE_MANIFEST_ATTR: json.dumps(
                        {"nodes": {"not": "a-list"}}
                    )
                }
            )
            == []
        )
        root_attrs = {
            workspace_format._WORKSPACE_MANIFEST_ATTR: json.dumps(
                {
                    "nodes": [
                        None,
                        {"uid": uid, "kind": "unknown", "path": "0"},
                        {"uid": uid, "kind": "imagetool", "path": "0"},
                    ]
                }
            )
        }
        assert saver._workspace_full_save_manifest_entries(root_attrs) == [
            (uid, "imagetool", "0/imagetool")
        ]

        manager._workspace_state.dirty_added.add("missing")
        manager._workspace_state.dirty_data.add(uid)
        assert saver._workspace_full_save_dirty_payload_uids() == {uid}
        manager._workspace_state.dirty_added.clear()
        manager._workspace_state.dirty_data.clear()

        assert saver._workspace_full_save_source_identities() is None
        missing_workspace = tmp_path / "missing.itws"
        manager._workspace_state.path = missing_workspace
        assert saver._workspace_full_save_source_identities() is None

        workspace = tmp_path / "identity.itws"
        manifest = {
            "nodes": [
                "bad-entry",
                {"uid": uid, "kind": "tool", "path": 1},
                {"uid": uid, "kind": "imagetool", "path": "0"},
            ]
        }
        with h5py.File(workspace, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = (
                workspace_format._current_workspace_schema_version()
            )
            h5_file.attrs[workspace_format._WORKSPACE_MANIFEST_ATTR] = json.dumps(
                manifest
            )
        manager._workspace_state.path = workspace
        source = saver._workspace_full_save_source_identities()
        assert source is not None
        assert source[0] == workspace
        assert source[1] == {(uid, "imagetool"): "0/imagetool"}

        monkeypatch.setattr(
            workspace_arrays,
            "_read_workspace_root_attrs_h5py",
            lambda _path: (_ for _ in ()).throw(OSError("boom")),
        )
        assert saver._workspace_full_save_source_identities() is None


def test_workspace_save_worker_start_and_finish_error_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class Snapshot:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller

        start_errors: list[str] = []
        snapshot = Snapshot()
        monkeypatch.setattr(
            workspace_controller.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: None),
        )
        assert not controller._start_workspace_save_worker(
            tmp_path / "none.itws",
            typing.cast("workspace_saving._WorkspaceSaveSnapshot", snapshot),
            on_finished=lambda *_args: None,
            on_start_error=lambda: start_errors.append("none"),
        )
        assert snapshot.closed
        assert start_errors == ["none"]

        class RaisingPool:
            def start(self, _worker) -> None:
                raise RuntimeError("cannot start")

        snapshot = Snapshot()
        monkeypatch.setattr(
            workspace_controller.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: RaisingPool()),
        )
        assert not controller._start_workspace_save_worker(
            tmp_path / "raise.itws",
            typing.cast("workspace_saving._WorkspaceSaveSnapshot", snapshot),
            on_finished=lambda *_args: None,
            on_start_error=lambda: start_errors.append("raise"),
        )
        assert snapshot.closed
        assert start_errors == ["none", "raise"]
        assert not manager._workspace_state.save_in_progress
        assert controller._background_save_worker is None
        assert controller._background_save_receiver is None

        class RecordingPool:
            def __init__(self) -> None:
                self.worker = None

            def start(self, worker) -> None:
                self.worker = worker

        errors: list[tuple[str, str]] = []
        pool = RecordingPool()
        monkeypatch.setattr(
            workspace_controller.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: pool),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        assert controller._start_workspace_save_worker(
            tmp_path / "finish.itws",
            typing.cast("workspace_saving._WorkspaceSaveSnapshot", Snapshot()),
            on_finished=lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert pool.worker is not None
        pool.worker.signals.finished.emit(0.1, None)
        assert errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]
        assert not manager._workspace_state.save_in_progress


def test_background_workspace_save_finish_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        workspace_path = tmp_path / "queued.itws"
        manager._workspace_state.path = workspace_path
        manager._workspace_state.dirty_state.add("uid")
        controller._background_save_requested = True

        queued_callbacks: list[Callable[[], None]] = []
        finished: list[bool] = []
        monkeypatch.setattr(
            controller,
            "_finish_workspace_save_result",
            lambda **_kwargs: True,
        )
        monkeypatch.setattr(
            controller,
            "_current_workspace_document_path",
            lambda: workspace_path,
        )
        monkeypatch.setattr(
            workspace_controller.QtCore.QTimer,
            "singleShot",
            staticmethod(lambda _delay, callback: queued_callbacks.append(callback)),
        )
        controller._finish_background_workspace_save(
            document_id=manager._workspace_state.document_id,
            workspace_path=workspace_path,
            old_workspace_path=None,
            backing_snapshot={},
            snapshot=typing.cast(
                "workspace_saving._WorkspaceSaveSnapshot", types.SimpleNamespace()
            ),
            worker_elapsed=0.1,
            error=None,
            origin=None,
            snapshot_elapsed=0.0,
            started_at=time.perf_counter(),
            restore_focus=False,
            on_finished=finished.append,
        )
        assert len(queued_callbacks) == 1
        assert finished == [True]
        assert not controller._background_save_requested

        errors: list[tuple[str, str]] = []
        finished.clear()
        monkeypatch.setattr(
            controller,
            "_finish_workspace_save_result",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        controller._finish_background_workspace_save(
            document_id=manager._workspace_state.document_id,
            workspace_path=workspace_path,
            old_workspace_path=None,
            backing_snapshot={},
            snapshot=typing.cast(
                "workspace_saving._WorkspaceSaveSnapshot", types.SimpleNamespace()
            ),
            worker_elapsed=0.1,
            error=None,
            origin=None,
            snapshot_elapsed=0.0,
            started_at=time.perf_counter(),
            restore_focus=False,
            on_finished=finished.append,
        )
        assert finished == [False]
        assert errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]


def test_workspace_save_and_save_as_error_continuation_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    def snapshot(generation: int = 0) -> workspace_saving._WorkspaceSaveSnapshot:
        return workspace_saving._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs={},
            delta_save_count=0,
            estimated_obsolete_bytes=0,
            replacement_delta_count=0,
            full_tree=xr.DataTree(),
        )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid
        controller = manager._workspace_controller

        finished: list[bool] = []
        manager._workspace_state.save_in_progress = True
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]
        manager._workspace_state.save_in_progress = False

        warnings: list[bool] = []
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: tmp_path / "bad.txt",
        )
        monkeypatch.setattr(
            workspace_controller,
            "_show_itws_workspace_warning",
            lambda _parent: warnings.append(True),
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert warnings == [True]
        assert finished == [False]

        errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: tmp_path / "save.itws",
        )
        monkeypatch.setattr(
            controller.saving,
            "_workspace_full_save_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]
        assert errors[-1] == (
            "Error while saving workspace",
            "An error occurred while saving the workspace file.",
        )

        monkeypatch.setattr(
            controller.saving,
            "_workspace_full_save_snapshot",
            lambda generation, **_kwargs: snapshot(generation),
        )
        worker_errors: list[workspace_saving._WorkspaceSaveError] = []
        monkeypatch.setattr(
            manager, "_show_workspace_save_worker_error", worker_errors.append
        )

        def _fail_worker(_fname, _snapshot, *, on_finished, on_start_error=None):
            del on_start_error
            on_finished(
                0.1,
                workspace_saving._WorkspaceSaveError("write failed"),
            )
            return True

        monkeypatch.setattr(controller, "_start_workspace_save_worker", _fail_worker)
        finished.clear()
        assert controller.save_as(native=False, on_finished=finished.append)
        assert [error.traceback_text for error in worker_errors] == ["write failed"]
        assert finished == [False]

        manager._workspace_controller._mark_workspace_clean()

        def _dirty_during_worker(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ):
            del on_start_error
            manager._workspace_controller._mark_node_state_dirty(uid)
            on_finished(0.1, None)
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _dirty_during_worker
        )
        finished.clear()
        assert controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]

        def _start_error_worker(_fname, _snapshot, *, on_finished, on_start_error=None):
            del on_finished
            if on_start_error is not None:
                on_start_error()
            return False

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _start_error_worker
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]

        manager._workspace_state.path = tmp_path / "current.itws"
        monkeypatch.setattr(
            controller.saving, "_workspace_save_snapshot", lambda _path: snapshot()
        )
        finished.clear()
        assert not controller.save(on_finished=finished.append, restore_focus=False)
        assert finished == [False]


def test_workspace_save_completion_ignores_inactive_document(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        workspace_path = tmp_path / "stale-save.itws"
        manager._workspace_state.path = workspace_path.resolve()
        manager._workspace_state.delta_save_count = 7
        manager._workspace_state.mark_layout_dirty()

        snapshot = workspace_saving._WorkspaceSaveSnapshot(
            generation=manager._workspace_state.dirty_generation,
            root_attrs={},
            delta_save_count=99,
        )
        monkeypatch.setattr(
            controller.saving, "_workspace_save_snapshot", lambda _path: snapshot
        )
        recorded_recent: list[pathlib.Path] = []
        monkeypatch.setattr(
            controller, "_record_recent_workspace", recorded_recent.append
        )

        def _finish_after_document_change(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ) -> bool:
            del on_start_error
            manager._workspace_state.advance_document_identity()
            manager._workspace_state.path = tmp_path / "replacement.itws"
            on_finished(0.1, None)
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _finish_after_document_change
        )

        finished: list[bool] = []
        assert controller.save(on_finished=finished.append, restore_focus=False)

        assert finished == [False]
        assert manager._workspace_state.delta_save_count == 7
        assert manager.is_workspace_modified
        assert recorded_recent == []


def test_workspace_save_as_completion_ignores_inactive_document(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        target = tmp_path / "stale-save-as.itws"
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        manager._workspace_state.mark_layout_dirty()

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target,
        )
        monkeypatch.setattr(
            controller.saving,
            "_workspace_full_save_snapshot",
            lambda generation, **_kwargs: workspace_saving._WorkspaceSaveSnapshot(
                generation=generation,
                root_attrs={},
                delta_save_count=0,
            ),
        )

        def _finish_after_document_change(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ) -> bool:
            del on_start_error
            manager._workspace_state.advance_document_identity()
            on_finished(0.1, None)
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _finish_after_document_change
        )

        finished: list[bool] = []
        assert controller.save_as(native=False, on_finished=finished.append)

        assert finished == [False]
        assert manager.workspace_path is None
        assert manager.is_workspace_modified


def test_manager_manifest_first_full_save_copies_clean_payloads(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(25, dtype=float).reshape(5, 5) + offset,
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-clean.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            lambda *_args, **_kwargs: pytest.fail(
                "Clean full save should copy payload groups"
            ),
        )

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        with h5py.File(fname, "r") as h5_file:
            assert "0/imagetool" in h5_file
            assert "1/imagetool" in h5_file


def test_manager_manifest_first_full_save_serializes_missing_source_group(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(25, dtype=float).reshape(5, 5) + offset,
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-missing-source.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        with h5py.File(fname, "a") as h5_file:
            del h5_file["0/imagetool"]

        original_serialize = (
            manager._workspace_controller.saving._serialize_workspace_node
        )
        serialized_paths: list[str] = []

        def _record_serialize(*args, **kwargs) -> None:
            serialized_paths.append(args[2])
            original_serialize(*args, **kwargs)

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            _record_serialize,
        )

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        assert serialized_paths == ["0"]
        with h5py.File(fname, "r") as h5_file:
            assert "0/imagetool" in h5_file
            assert "1/imagetool" in h5_file


def test_manager_manifest_first_full_save_preserves_clean_mismatched_compression(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_compression = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            fname = tmp_path / "manifest-copy-mismatched-compression.itws"
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
            adopt_workspace_path(manager, fname)
            manager._workspace_controller._mark_workspace_clean()
            with h5py.File(fname, "r") as h5_file:
                data_ds = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
                assert _hdf5_blosc2_level_codec(data_ds) is None

            writes: list[
                tuple[
                    str | os.PathLike[str] | None,
                    tuple[tuple[str, str, dict[str, typing.Any] | None], ...],
                ]
            ] = []
            original_write = workspace_storage._write_full_workspace_tree_file

            def _record_write(*args, **kwargs) -> None:
                writes.append(
                    (kwargs.get("copy_source"), tuple(kwargs.get("copy_groups", ())))
                )
                original_write(*args, **kwargs)

            erlab.interactive.options["io/workspace/compression"] = "zstd1"
            monkeypatch.setattr(
                manager._workspace_controller.saving,
                "_serialize_workspace_node",
                lambda *_args, **_kwargs: pytest.fail(
                    "Clean mismatched payload should be copied"
                ),
            )
            monkeypatch.setattr(
                workspace_storage,
                "_write_full_workspace_tree_file",
                _record_write,
            )

            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )

            assert writes[-1] == (
                str(fname),
                (("0/imagetool", "0/imagetool", None),),
            )
            with h5py.File(fname, "r") as h5_file:
                data_ds = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
                assert _hdf5_blosc2_level_codec(data_ds) is None
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_manifest_first_full_save_serializes_only_dirty_data(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(25, dtype=float).reshape(5, 5) + offset,
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-dirty.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        dirty_uid = manager._tool_graph.root_wrappers[0].uid
        updated = xr.DataArray(np.full((5, 5), 42.0), dims=("x", "y"))
        manager.get_imagetool(0).slicer_area.replace_source_data(
            updated, auto_compute=False
        )
        manager._workspace_controller._mark_node_data_dirty(dirty_uid)

        original_serialize = (
            manager._workspace_controller.saving._serialize_workspace_node
        )
        serialized_paths: list[str] = []

        def _record_serialize(*args, **kwargs) -> None:
            serialized_paths.append(args[2])
            original_serialize(*args, **kwargs)

        original_write = workspace_storage._write_full_workspace_tree_file
        writes: list[tuple[tuple[str, str, dict[str, typing.Any] | None], ...]] = []

        def _record_write(*args, **kwargs) -> None:
            writes.append(tuple(kwargs.get("copy_groups", ())))
            original_write(*args, **kwargs)

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            _record_serialize,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_full_workspace_tree_file",
            _record_write,
        )

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        assert serialized_paths == ["0"]
        assert ("1/imagetool", "1/imagetool", None) in writes[-1]
        assert all(group[1] != "0/imagetool" for group in writes[-1])


def test_manager_manifest_first_full_save_rewrites_dirty_state_attrs(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25, dtype=float).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-state.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._tool_graph.root_wrappers[0].name = "renamed"

        original_serialize = (
            manager._workspace_controller.saving._serialize_workspace_node
        )
        serialized_paths: list[str] = []

        def _record_serialize(*args, **kwargs) -> None:
            serialized_paths.append(args[2])
            original_serialize(*args, **kwargs)

        original_write = workspace_storage._write_full_workspace_tree_file
        writes: list[tuple[tuple[str, str, dict[str, typing.Any] | None], ...]] = []

        def _record_write(*args, **kwargs) -> None:
            writes.append(tuple(kwargs.get("copy_groups", ())))
            original_write(*args, **kwargs)

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            _record_serialize,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_full_workspace_tree_file",
            _record_write,
        )

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        assert serialized_paths == ["0"]
        assert all(group[1] != "0/imagetool" for group in writes[-1])
        with h5py.File(fname, "r") as h5_file:
            assert h5_file["0/imagetool"].attrs["itool_title"] == "renamed"


def test_manager_manifest_first_save_as_copies_clean_payloads(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25, dtype=float).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        source = tmp_path / "manifest-copy-source.itws"
        target = tmp_path / "manifest-copy-target.itws"
        manager._workspace_controller.saving._save_workspace_document(
            source, force_full=True
        )
        adopt_workspace_path(manager, source)
        manager._workspace_controller._mark_workspace_clean()

        writes: list[
            tuple[
                str | os.PathLike[str] | None,
                tuple[tuple[str, str, dict[str, typing.Any] | None], ...],
            ]
        ] = []
        original_write = workspace_storage._write_full_workspace_tree_file

        def _record_write(*args, **kwargs) -> None:
            writes.append(
                (kwargs.get("copy_source"), tuple(kwargs.get("copy_groups", ())))
            )
            original_write(*args, **kwargs)

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_serialize_workspace_node",
            lambda *_args, **_kwargs: pytest.fail(
                "Clean Save As should copy payload groups"
            ),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_write_full_workspace_tree_file",
            _record_write,
        )

        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)

        assert manager.workspace_path == str(target.resolve())
        assert writes[-1] == (str(source), (("0/imagetool", "0/imagetool", None),))
        with h5py.File(target, "r") as h5_file:
            assert "0/imagetool" in h5_file


def test_manager_manifest_first_full_save_falls_back_without_usable_source(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25, dtype=float).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-fallback.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        monkeypatch.setattr(
            workspace_arrays,
            "_read_workspace_root_attrs_h5py",
            lambda _path: {"imagetool_workspace_schema_version": 1},
        )

        def _fail_materializing_fallback() -> typing.NoReturn:
            pytest.fail("fallback full save should not call _to_datatree()")

        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_to_datatree",
            _fail_materializing_fallback,
        )

        snapshot = manager._workspace_controller.saving._workspace_full_save_snapshot(
            1, fname=fname
        )
        try:
            assert snapshot.full_tree is not None
            assert snapshot.copy_source is None
            assert snapshot.copy_groups == ()
            assert snapshot.copy_group_sources == ()
            ds = typing.cast(
                "xr.DataTree", snapshot.full_tree["0/imagetool"]
            ).to_dataset(inherit=False)
            assert _ITOOL_DATA_NAME in ds
        finally:
            snapshot.close()


def test_manager_associate_loaded_legacy_workspace_resets_repack_state(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        converted = tmp_path / "converted.itws"
        manager._workspace_state.path = converted.resolve()

        monkeypatch.setattr(
            workspace_format,
            "_workspace_schema_requires_conversion",
            lambda _schema_version: True,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_save_legacy_workspace_as_v4",
            lambda *_, **__: (str(converted), None),
        )

        manager._workspace_controller._associate_loaded_workspace_file(
            tmp_path / "legacy.itws",
            1,
            delta_save_count=5,
            estimated_obsolete_bytes=100,
            replacement_delta_count=2,
            repack_estimate_known=False,
            rebind_data=False,
        )

        assert manager._workspace_state.path == converted
        assert manager._workspace_state.delta_save_count == 0
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        assert manager._workspace_state.repack_estimate_known


def test_manager_workspace_file_repack_snapshot_guards(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        saver = controller.saving

        manager._workspace_state.path = None
        assert saver._workspace_file_repack_snapshot(1) is None

        manager._workspace_state.path = tmp_path / "network.itws"
        monkeypatch.setattr(
            workspace_storage,
            "_workspace_path_is_likely_network_path",
            lambda _path: True,
        )
        assert saver._workspace_file_repack_snapshot(1) is None

        monkeypatch.setattr(
            workspace_storage,
            "_workspace_path_is_likely_network_path",
            lambda _path: False,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_workspace_file_repack_payload",
            lambda _path: (_ for _ in ()).throw(RuntimeError("repack failed")),
        )
        assert saver._workspace_file_repack_snapshot(1) is None


def test_manager_workspace_shutdown_repack_gate_edges(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        saver = controller.saving

        manager._workspace_state.path = None
        assert not saver._workspace_should_repack_before_shutdown()

        workspace_path = tmp_path / "workspace.itws"
        manager._workspace_state.path = workspace_path
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=0,
            replacement_delta_count=0,
            known=False,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_workspace_obsolete_estimate",
            lambda _path: (_ for _ in ()).throw(RuntimeError("estimate failed")),
        )
        assert not saver._workspace_should_repack_before_shutdown()

        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=100,
            replacement_delta_count=1,
        )
        assert not saver._workspace_should_repack_before_shutdown()

        workspace_path.touch()
        assert not saver._workspace_should_repack_before_shutdown()


def test_workspace_remote_incremental_option_allows_delta_save(
    monkeypatch,
    tmp_path,
) -> None:
    options = erlab.interactive.options
    incremental_name = "io/workspace/incremental_save_on_remote"
    use_incremental_name = "io/workspace/use_incremental"
    old_remote_value = options[incremental_name]
    old_incremental_value = options[use_incremental_name]
    fname = tmp_path / "remote-incremental.itws"
    fname.touch()
    monkeypatch.setattr(
        workspace_storage, "_workspace_path_is_high_risk", lambda *_args: True
    )
    options[incremental_name] = True
    options[use_incremental_name] = True
    try:
        assert not workspace_storage._workspace_requires_full_save(
            fname,
            needs_full_save=False,
            schema_version=workspace_format._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )
    finally:
        options[incremental_name] = old_remote_value
        options[use_incremental_name] = old_incremental_value


def test_manager_workspace_dirty_marker_not_saved_in_titles(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)), dims=["x", "y"], name="source"
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root_uid = manager._tool_graph.root_wrappers[0].uid
        tool = DerivativeTool(data)
        tool_uid = manager.add_childtool(tool, 0, show=False)

        root.setWindowTitle("stale root title[*]")
        manager._tool_graph.root_wrappers[0].update_title()
        assert "stale root title" not in manager._tool_graph.root_wrappers[0].label_text
        assert "[*]" not in manager._tool_graph.root_wrappers[0].label_text

        root.setWindowTitle("stale root title[*]")
        tool.setWindowTitle("stale tool title[*]")
        manager._workspace_controller._set_node_window_modified(root_uid, True)
        manager._workspace_controller._set_node_window_modified(tool_uid, True)

        expect_title_placeholder = sys.platform != "darwin"
        assert ("[*]" in root.windowTitle()) is expect_title_placeholder
        assert ("[*]" in tool.windowTitle()) is expect_title_placeholder
        assert (
            root.windowTitle()
            == manager_widgets._window_title_with_modified_placeholder(
                manager._tool_graph.root_wrappers[0].label_text
            )
        )
        assert (
            tool.windowTitle()
            == manager_widgets._window_title_with_modified_placeholder(
                f"{tool.tool_name}: {tool._tool_display_name}"
            )
        )

        fname = tmp_path / "titles.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        import h5py

        with h5py.File(fname, "r") as h5_file:
            root_title = h5_file["0/imagetool"].attrs["itool_title"]
            tool_title = h5_file[f"0/childtools/{tool_uid}/tool"].attrs["tool_title"]

        assert "[*]" not in root_title
        assert "[*]" not in tool_title
        assert root_title == "source"


def test_manager_workspace_full_save_drops_empty_attr_name(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            attrs={"": "dropped", "note": ""},
            name="data",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "empty-attr-name.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        assert "" in root.slicer_area._data.attrs
        with h5py.File(fname, "r") as h5_file:
            saved_attrs = h5_file["0/imagetool"][_ITOOL_DATA_NAME].attrs
            assert "" not in list(saved_attrs)
            assert saved_attrs["note"] == ""


def test_manager_workspace_full_save_roundtrips_non_native_data_attrs(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    rich_attr = _rich_workspace_attr_value()
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": xr.DataArray(
                    np.arange(5),
                    dims=("x",),
                    attrs={"axis_config": rich_attr},
                ),
                "y": np.arange(5),
            },
            attrs={"Single Motor Scan": rich_attr},
            name="data",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        live_rich_attr = root.slicer_area._data.attrs["Single Motor Scan"]
        live_axis_attr = root.slicer_area._data.coords["x"].attrs["axis_config"]

        fname = tmp_path / "rich-data-attrs.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        assert root.slicer_area._data.attrs["Single Motor Scan"] is live_rich_attr
        assert root.slicer_area._data.coords["x"].attrs["axis_config"] is live_axis_attr
        assert (
            workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR
            not in root.slicer_area._data.attrs
        )
        with h5py.File(fname, "r") as h5_file:
            saved_data = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert "Single Motor Scan" not in saved_data.attrs
            assert workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR in saved_data.attrs

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        assert manager._workspace_controller.loading._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded = manager.get_imagetool(0).slicer_area._data
        _assert_rich_workspace_attr(loaded.attrs["Single Motor Scan"])
        _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        tree = workspace_arrays.open_workspace_datatree(fname, chunks=None)
        try:
            assert manager._workspace_controller.loading._from_datatree(
                tree,
                replace=True,
                mark_dirty=False,
                select=False,
                workspace_file_path=fname,
            )
        finally:
            tree.close()
        loaded = manager.get_imagetool(0).slicer_area._data
        _assert_rich_workspace_attr(loaded.attrs["Single Motor Scan"])
        _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])


def test_manager_workspace_save_preserves_reordered_roots(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in range(3):
            data = xr.DataArray(
                np.full((5, 5), value), dims=["x", "y"], name=f"data_{value}"
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "ordered.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        model = manager.tree_view._model
        assert model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            model.rowCount(),
            0,
            QtCore.QModelIndex(),
        )
        assert manager._tool_graph.displayed_indices == [1, 2, 0]
        assert manager.is_workspace_modified
        assert _request_workspace_save_and_wait(qtbot, manager)

        with h5py.File(fname, "r") as h5_file:
            manifest = json.loads(h5_file.attrs["imagetool_workspace_manifest"])
        assert manifest["root_order"] == [1, 2, 0]

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        loaded_order = [
            int(manager.get_imagetool(index).slicer_area._data.values[0, 0])
            for index in manager._tool_graph.displayed_indices
        ]
        assert loaded_order == [1, 2, 0]


def test_manager_workspace_child_save_shortcuts_use_background_save(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        calls: list[bool] = []

        def _fake_save(*, native: bool = True) -> bool:
            calls.append(native)
            return True

        monkeypatch.setattr(manager._workspace_controller, "save", _fake_save)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        root_shortcuts = root.findChildren(QtWidgets.QShortcut)
        root_save = [
            shortcut
            for shortcut in root_shortcuts
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(root_save) == 1
        root_save[0].activated.emit()

        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool_child(child, 0, show=False)
        child_save = [
            shortcut
            for shortcut in child.findChildren(QtWidgets.QShortcut)
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(child_save) == 1
        child_save[0].activated.emit()

        tool = DerivativeTool(data)
        manager.add_childtool(tool, 0, show=False)
        tool_save = [
            shortcut
            for shortcut in tool.findChildren(QtWidgets.QShortcut)
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(tool_save) == 1
        tool_save[0].activated.emit()

        assert calls == [True, True, True]


def test_manager_workspace_delta_save_splits_state_and_data_writes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "delta.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)

        dataset_writes: list[str | None] = []
        original_to_netcdf = xr.Dataset.to_netcdf

        def _to_netcdf_spy(self, *args, **kwargs):
            dataset_writes.append(kwargs.get("group"))
            return original_to_netcdf(self, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", _to_netcdf_spy)

        manager.rename_imagetool(0, "state only")
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert dataset_writes == []

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement)
        assert _request_workspace_save_and_wait(qtbot, manager)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_full_save_keeps_full_persistence_for_serialized_nodes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "full-persistence.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 1
        root.slicer_area.replace_source_data(replacement)

        original = imagetool_viewer.ImageSlicerArea.persistence_data_and_state
        calls = 0

        def _persistence_data_and_state_spy(self):
            nonlocal calls
            calls += 1
            return original(self)

        monkeypatch.setattr(
            imagetool_viewer.ImageSlicerArea,
            "persistence_data_and_state",
            _persistence_data_and_state_spy,
        )

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        assert calls >= 1


def test_manager_workspace_full_save_preserves_in_memory_backing_after_rebind(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "memory.itws"
        backing_snapshot = (
            manager._workspace_controller.loading._workspace_data_backing_snapshot()
        )
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            fname,
            backing_snapshot=backing_snapshot,
            old_workspace_path=None,
        )

        saved_data = manager.get_imagetool(0).slicer_area._data
        assert workspace_arrays.dataarray_is_numpy_backed(saved_data)


def test_manager_workspace_load_keeps_visible_saved_data_in_memory(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
            dims=["x", "y"],
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)

        fname = tmp_path / "load-visible-memory.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area
        assert not loaded.data_chunked
        assert not loaded.data_file_backed
        assert loaded.data_loadable is False
        assert workspace_arrays.dataarray_is_numpy_backed(loaded._data)
        assert loaded._data.values.flags.writeable
        np.testing.assert_array_equal(loaded._data.values, data.values)


def test_manager_workspace_lazy_data_delta_save_uses_pending_group_before_replacing(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-data.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        manager.get_imagetool(0).slicer_area.replace_source_data(
            replacement, auto_compute=False
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert list(tmp_path.glob("lazy-data.itws.delta-*")) == []

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_same_file_lazy_data_delta_save_does_not_deadlock(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
            dims=["x", "y"],
            coords={"x": np.arange(512), "y": np.arange(512)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "same-file-lazy.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            fname, targets=[0], chunks={}
        )
        assert manager.get_imagetool(0).slicer_area.data_chunked
        manager.get_imagetool(0).slicer_area._set_chunks({"x": 128, "y": 64})

        uid = manager._tool_graph.root_wrappers[0].uid
        manager._workspace_controller._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert saved.chunks == (128, 64)


def test_manager_workspace_lazy_data_delta_pending_failure_preserves_old_group(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-failure.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement, auto_compute=False)

        def _write_partial_pending_then_raise(
            fname: str | os.PathLike[str],
            _constructor: Mapping[str, xr.Dataset],
            _group_path: str,
            pending_path: str,
        ) -> None:
            with workspace_storage._open_workspace_h5_file_for_update(fname) as h5_file:
                h5_file.create_group(pending_path)
            raise RuntimeError("pending write failed")

        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_constructor_groups_to_pending",
            _write_partial_pending_then_raise,
        )
        monkeypatch.setattr(
            manager, "_show_workspace_save_worker_error", lambda *args: None
        )

        assert not _request_workspace_save_and_wait(qtbot, manager)
        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert not any(
                workspace_format._is_workspace_internal_group_name(name)
                for name in h5_file
            )


def test_manager_workspace_stale_pending_groups_do_not_poison_open_or_save(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "stale-pending.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.rename_imagetool(0, "cleaned")
        assert _request_workspace_save_and_wait(qtbot, manager)
        with h5py.File(fname, "r") as h5_file:
            assert not any(
                workspace_format._is_workspace_internal_group_name(name)
                for name in h5_file
            )


def test_manager_workspace_delta_save_persists_geometry_changes(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "geometry.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        root.setGeometry(12, 34, 321, 234)
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        expected_rect = tuple(root.geometry().getRect())

        assert _request_workspace_save_and_wait(qtbot, manager)
        with h5py.File(fname, "r") as h5_file:
            saved_state = json.loads(h5_file["0/imagetool"].attrs["itool_window_state"])
            saved_rect = tuple(int(value) for value in saved_state["rect"])
        assert saved_rect == expected_rect
