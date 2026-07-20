import json
import pathlib
import types
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._widgets as manager_widgets
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._execution import (
    replay_file_provenance,
    replay_script_provenance,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    _ProvenanceDisplayRow,
    _ProvenanceReorderBlock,
    _ProvenanceReorderBlockRef,
    _ProvenanceReorderSection,
    _ProvenanceReorderSectionRef,
    _ProvenanceStepRef,
    compose_full_provenance,
    full_data,
    script,
    stamp_operation_group,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    IselOperation,
    NormalizeOperation,
    QSelAggregationOperation,
    RenameOperation,
    ScriptCodeOperation,
)
from erlab.interactive.imagetool.manager._provenance_edit import (
    _controller as manager_provenance_controller,
)
from erlab.interactive.imagetool.manager._provenance_edit._controller import (
    _ProvenanceEditController,
    _ProvenanceReorderSession,
)
from erlab.interactive.imagetool.manager._provenance_edit._reorder import (
    _REORDER_BLOCK_ROLE,
    _ProvenanceReorderDialog,
    _ProvenanceReorderListModel,
    _ProvenanceReorderListView,
)
from erlab.interactive.imagetool.manager._widgets import _TrustedScriptReplayCancelled
from tests.interactive.imagetool.manager.helpers import (
    select_metadata_rows,
    select_tools,
)

from ._common import (
    _add_file_replay_tool,
    _manager_replay_file_spec,
    _provenance_paste_test_data,
    _set_aggregate,
    _set_provenance_steps_clipboard,
)


def _composed_reorder_file_spec(file_path: pathlib.Path) -> ToolProvenanceSpec:
    spec = _manager_replay_file_spec(file_path)
    for source in (
        full_data(
            IselOperation(kwargs={"x": slice(0, 2)}),
            NormalizeOperation(dims=("x",), mode="area"),
        ),
        full_data(AssignAttrsOperation(attrs={"recorded_stage": "second"})),
    ):
        composed = compose_full_provenance(spec, source)
        assert composed is not None
        spec = composed
    return spec


def test_manager_provenance_file_load_edit_accept_and_cancel(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    first = test_data.copy(deep=True)
    second = (test_data * 3).rename(test_data.name)
    first.to_netcdf(first_path, engine="h5netcdf")
    second.to_netcdf(second_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            first,
            manager=True,
            file_path=first_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._tool_graph.root_wrappers[0]
        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [0])
        row = manager._selected_derivation_row()
        assert row is not None
        editable, _reason = manager._provenance_edit_controller.can_edit_row(row)
        assert editable

        before_spec = root.provenance_spec
        before_data = root.slicer_area._data.copy(deep=True)

        accept_dialog(
            manager._edit_selected_derivation_step,
            accept_call=lambda dialog: dialog.reject(),
        )
        xr.testing.assert_identical(root.slicer_area._data, before_data)
        assert root.provenance_spec == before_spec

        def _edit_file(dialog: QtWidgets.QDialog) -> None:
            dialog.path_edit.setText(str(second_path))  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_file)

        assert root.provenance_spec is not None
        assert root.provenance_spec.file_load_source is not None
        assert pathlib.Path(root.provenance_spec.file_load_source.path) == second_path
        xr.testing.assert_identical(
            root.slicer_area._data.rename(None),
            second.astype(np.float64).rename(None),
        )


def test_manager_provenance_file_load_batch_edit_updates_matching_peer(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    paths = {
        "old_a": old_dir / "a.h5",
        "old_b": old_dir / "b.h5",
        "new_a": new_dir / "a.h5",
        "new_b": new_dir / "b.h5",
    }
    old_a = test_data.copy(deep=True)
    old_b = (test_data + 10).rename(test_data.name)
    new_a = (test_data + 100).rename(test_data.name)
    new_b = (test_data + 200).rename(test_data.name)
    for data, path in (
        (old_a, paths["old_a"]),
        (old_b, paths["old_b"]),
        (new_a, paths["new_a"]),
        (new_b, paths["new_b"]),
    ):
        data.to_netcdf(path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        first_tool = _add_file_replay_tool(
            manager,
            old_a,
            _manager_replay_file_spec(paths["old_a"]),
        )
        second_tool = _add_file_replay_tool(
            manager,
            old_b,
            _manager_replay_file_spec(paths["old_b"]),
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [0])

        def _edit_batch(dialog: QtWidgets.QDialog) -> None:
            dialog.path_edit.setText(str(paths["new_a"]))  # type: ignore[attr-defined]
            dialog.batch_apply_check.setChecked(True)  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_batch)

        xr.testing.assert_identical(
            first_tool.slicer_area._data.rename(None),
            new_a.astype(np.float64).rename(None),
        )
        xr.testing.assert_identical(
            second_tool.slicer_area._data.rename(None),
            new_b.astype(np.float64).rename(None),
        )
        first_spec = manager._tool_graph.root_wrappers[0].provenance_spec
        second_spec = manager._tool_graph.root_wrappers[1].provenance_spec
        assert first_spec is not None
        assert first_spec.file_load_source is not None
        assert second_spec is not None
        assert second_spec.file_load_source is not None
        assert pathlib.Path(first_spec.file_load_source.path) == paths["new_a"]
        assert pathlib.Path(second_spec.file_load_source.path) == paths["new_b"]


def test_manager_provenance_nested_file_load_batch_relinks_deleted_parents(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_a_path = old_dir / "a.h5"
    old_b_path = old_dir / "b.h5"
    new_a_path = new_dir / "a.h5"
    new_b_path = new_dir / "b.h5"
    old_a = test_data.copy(deep=True)
    old_b = (test_data + 10).rename(test_data.name)
    new_a = (test_data + 100).rename(test_data.name)
    new_b = (test_data + 200).rename(test_data.name)
    new_a.to_netcdf(new_a_path, engine="h5netcdf")
    new_b.to_netcdf(new_b_path, engine="h5netcdf")

    first_spec = _manager_replay_file_spec(old_a_path)
    second_spec = _manager_replay_file_spec(old_b_path)
    root_spec = script(
        ScriptCodeOperation(
            label="Combine deleted parent tools",
            code="derived = data_0 + data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="ImageTool 0: a",
                node_uid="deleted-a",
                node_snapshot_token=str(object()),
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="ImageTool 1: b",
                node_uid="deleted-b",
                node_snapshot_token=str(object()),
                provenance_spec=second_spec,
            ),
        ),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        tool = _add_file_replay_tool(
            manager,
            old_a + old_b,
            root_spec,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root = manager._tool_graph.root_wrappers[0]

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [2])
        row = manager._selected_derivation_row()
        assert row is not None
        assert row.script_input_path == (0,)
        assert manager._provenance_edit_controller.can_edit_row(row)[0]

        def _edit_batch(dialog: QtWidgets.QDialog) -> None:
            dialog.path_edit.setText(str(new_a_path))  # type: ignore[attr-defined]
            dialog.batch_apply_check.setChecked(True)  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_batch)

        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            (new_a + new_b).astype(np.float64).rename(None),
        )
        assert root.provenance_spec is not None
        first_input, second_input = root.provenance_spec.script_inputs
        assert first_input.node_uid is None
        assert second_input.node_uid is None
        first_relinked = first_input.parsed_provenance_spec()
        second_relinked = second_input.parsed_provenance_spec()
        assert first_relinked is not None
        assert second_relinked is not None
        assert first_relinked.file_load_source is not None
        assert second_relinked.file_load_source is not None
        assert pathlib.Path(first_relinked.file_load_source.path) == new_a_path
        assert pathlib.Path(second_relinked.file_load_source.path) == new_b_path


def test_manager_provenance_rows_dim_when_not_activatable(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    script_data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        name="scripted",
    )
    script_spec = script(
        ScriptCodeOperation(label="Copy source", code="derived = data"),
        QSelAggregationOperation(dims=("x",), func="mean"),
        start_label="Run script",
        active_name="derived",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        _add_file_replay_tool(manager, test_data, _manager_replay_file_spec(file_path))
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        select_tools(manager, [0])
        manager._update_info()

        load_item = manager.metadata_derivation_list.item(0)
        assert load_item is not None
        assert (
            load_item.data(manager_widgets._METADATA_DERIVATION_ACTIVATABLE_ROLE)
            is True
        )
        assert (
            load_item.data(manager_widgets._METADATA_DERIVATION_COPYABLE_ROLE) is False
        )
        assert load_item.foreground().style() == QtCore.Qt.BrushStyle.NoBrush
        select_metadata_rows(manager, [0])
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert menu.defaultAction() is manager._metadata_edit_step_action

        tool = itool(script_data.qsel.mean("x"), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=script_spec)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        manager.tree_view.clearSelection()
        select_tools(manager, [1])
        manager._update_info()

        script_operation_item = None
        for row in range(manager.metadata_derivation_list.count()):
            item = manager.metadata_derivation_list.item(row)
            if item is None:
                continue
            if item.data(
                manager_widgets._METADATA_DERIVATION_COPYABLE_ROLE
            ) and not item.data(manager_widgets._METADATA_DERIVATION_ACTIVATABLE_ROLE):
                script_operation_item = item
                break
        assert script_operation_item is not None
        assert script_operation_item.toolTip()
        assert script_operation_item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled
        assert script_operation_item.foreground().color() == (
            manager.metadata_derivation_list.palette().color(
                QtGui.QPalette.ColorGroup.Disabled,
                QtGui.QPalette.ColorRole.Text,
            )
        )
        select_metadata_rows(
            manager,
            [manager.metadata_derivation_list.row(script_operation_item)],
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert menu.defaultAction() is None


def test_manager_provenance_row_activation_uses_edit_default_action(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        QSelAggregationOperation(dims=("alpha",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, test_data.qsel.mean("alpha"), spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        activated_rows: list[_ProvenanceDisplayRow | None] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "edit_row",
            activated_rows.append,
        )

        select_metadata_rows(manager, [0])
        manager.metadata_derivation_list.setFocus()
        qtbot.keyClick(
            manager.metadata_derivation_list,
            QtCore.Qt.Key.Key_Return,
        )
        assert activated_rows == [manager._selected_derivation_row()]

        activated_rows.clear()
        item = manager.metadata_derivation_list.item(0)
        assert item is not None
        manager.metadata_derivation_list.itemActivated.emit(item, 0)
        assert activated_rows == [manager._selected_derivation_row()]

        activated_rows.clear()
        manager.metadata_derivation_list.clearSelection()
        selection_model = manager.metadata_derivation_list.selectionModel()
        assert selection_model is not None
        for row in (0, 1):
            selection_model.select(
                manager.metadata_derivation_list.model().index(row, 0),
                QtCore.QItemSelectionModel.SelectionFlag.Select,
            )
        selection_model.setCurrentIndex(
            manager.metadata_derivation_list.model().index(0, 0),
            QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
        )
        manager.metadata_derivation_list.itemActivated.emit(item, 0)
        assert activated_rows == []


def test_manager_provenance_row_activation_ignores_noneditable_row(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        name="scan",
    )
    spec = script(
        ScriptCodeOperation(label="Copy source", code="derived = data"),
        QSelAggregationOperation(dims=("x",), func="mean"),
        start_label="Run script",
        active_name="derived",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = itool(data.qsel.mean("x"), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        select_tools(manager, [0])
        manager._update_info()

        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "edit_row",
            lambda _row: pytest.fail("non-editable activation should be ignored"),
        )
        item = None
        for row in range(manager.metadata_derivation_list.count()):
            candidate = manager.metadata_derivation_list.item(row)
            if candidate is None:
                continue
            if candidate.data(
                manager_widgets._METADATA_DERIVATION_COPYABLE_ROLE
            ) and not candidate.data(
                manager_widgets._METADATA_DERIVATION_ACTIVATABLE_ROLE
            ):
                item = candidate
                break
        assert item is not None
        select_metadata_rows(manager, [manager.metadata_derivation_list.row(item)])
        manager.metadata_derivation_list.itemActivated.emit(item, 0)


def test_manager_provenance_context_menu_preserves_extended_selection(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    operation = AssignAttrsOperation(attrs={"note": "selected"})
    spec = _manager_replay_file_spec(
        file_path,
        QSelAggregationOperation(dims=("alpha",), func="mean"),
        operation,
    )
    displayed = operation.apply(test_data.qsel.mean("alpha"))

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, displayed, spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        assert manager.metadata_derivation_list.count() >= 3
        select_metadata_rows(manager, [0, 1])
        target_item = manager.metadata_derivation_list.item(2)
        assert target_item is not None
        assert not target_item.isSelected()

        build_metadata_derivation_menu = manager._build_metadata_derivation_menu
        menu = build_metadata_derivation_menu()
        assert menu is not None
        assert not manager._metadata_edit_step_action.isEnabled()
        assert not manager._metadata_revert_step_action.isEnabled()
        assert not manager._metadata_delete_step_action.isEnabled()

        captured_selection: list[list[int]] = []

        def _capture_menu(*, include_row_actions: bool = True) -> None:
            assert include_row_actions
            captured_selection.append(
                [
                    manager.metadata_derivation_list.row(item)
                    for item in manager.metadata_derivation_list.selectedItems()
                ]
            )
            return

        monkeypatch.setattr(manager, "_build_metadata_derivation_menu", _capture_menu)
        pos = manager.metadata_derivation_list.visualItemRect(target_item).center()
        manager._show_metadata_derivation_menu(pos)

        assert captured_selection == [[0, 1]]
        assert [
            manager.metadata_derivation_list.row(item)
            for item in manager.metadata_derivation_list.selectedItems()
        ] == [0, 1]
        assert not target_item.isSelected()

        manager.metadata_derivation_list.setCurrentItem(target_item)
        manager.metadata_derivation_list.clearSelection()
        assert manager.metadata_derivation_list.currentItem() is target_item
        assert manager.metadata_derivation_list.selectedItems() == []
        menu = build_metadata_derivation_menu()
        assert menu is not None
        assert not manager._metadata_edit_step_action.isEnabled()
        assert not manager._metadata_revert_step_action.isEnabled()
        assert not manager._metadata_copy_selected_action.isEnabled()
        assert not manager._metadata_delete_step_action.isEnabled()


def test_manager_provenance_context_menu_on_empty_space_keeps_paste(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        QSelAggregationOperation(dims=("alpha",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, test_data.qsel.mean("alpha"), spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        _set_provenance_steps_clipboard(
            (AssignAttrsOperation(attrs={"copied": "yes"}),)
        )
        menu = manager._build_metadata_derivation_menu(include_row_actions=False)
        assert menu is not None
        assert manager._metadata_paste_steps_action in menu.actions()
        assert manager._metadata_paste_steps_action.isEnabled()
        assert not manager._metadata_edit_step_action.isEnabled()
        assert not manager._metadata_revert_step_action.isEnabled()
        assert not manager._metadata_copy_selected_action.isEnabled()
        assert not manager._metadata_delete_step_action.isEnabled()


def test_manager_provenance_paste_filter_respects_focus_guards(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = _provenance_paste_test_data()

    with manager_context() as manager:
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=full_data())
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._provenance_paste_filter is not None
        paste_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_V,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        paste_calls = 0

        def _record_paste() -> None:
            nonlocal paste_calls
            paste_calls += 1

        monkeypatch.setattr(
            manager,
            "_paste_provenance_steps_from_clipboard",
            _record_paste,
        )
        monkeypatch.setattr(
            manager._provenance_paste_filter,
            "_should_handle_paste",
            lambda: True,
        )

        assert manager._provenance_paste_filter.eventFilter(
            manager.tree_view, paste_event
        )
        assert paste_calls == 1

        paste_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_V,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        monkeypatch.setattr(
            manager._provenance_paste_filter,
            "_should_handle_paste",
            lambda: False,
        )
        assert not manager._provenance_paste_filter.eventFilter(
            manager.tree_view, paste_event
        )
        assert paste_calls == 1

        assert manager_mainwindow._widget_accepts_text_paste(
            manager.text_box, stop_at=manager
        )
        assert not manager_mainwindow._widget_accepts_text_paste(
            manager.tree_view, stop_at=manager
        )


def test_manager_provenance_context_menu_groups_row_commands(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        QSelAggregationOperation(dims=("alpha",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, test_data.qsel.mean("alpha"), spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None

        assert manager._metadata_delete_step_action.isEnabled()

        actions = menu.actions()
        assert actions[0] is manager._metadata_edit_step_action
        assert actions[1] is manager._metadata_reorder_steps_action
        assert actions[2] is manager._metadata_revert_step_action
        assert actions[3].isSeparator()
        assert actions[4] is manager._metadata_copy_selected_action
        assert actions[5] is manager._metadata_paste_steps_action
        if manager._metadata_copy_full_action in actions:
            assert actions[6] is manager._metadata_copy_full_action
            separator_index = 7
        else:
            separator_index = 6
        assert actions[separator_index].isSeparator()
        assert actions[separator_index + 1] is manager._metadata_delete_step_action


def test_manager_provenance_reorder_requires_available_replay(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    recorded_input = _manager_replay_file_spec(file_path)
    steps = (
        AssignAttrsOperation(attrs={"first": True}),
        AssignAttrsOperation(attrs={"second": True}),
    )
    opaque_script = script(
        ScriptCodeOperation(
            label="Opaque operation",
            code=None,
            copyable=False,
        ),
        *steps,
        start_label="Run opaque code",
        active_name="derived",
    )
    missing_input_script = script(
        *steps,
        start_label="Run script",
        seed_code="derived = missing_input",
        active_name="derived",
        script_inputs=(ScriptInput(name="missing_input", label="Unavailable input"),),
    )
    missing_file = _composed_reorder_file_spec(tmp_path / "missing.h5")
    detached_live = full_data(*steps)
    opaque_live = full_data(
        ScriptCodeOperation(
            label="Opaque live operation",
            code=None,
            copyable=False,
        ),
        *steps,
    )
    trusted_script = script(
        ScriptCodeOperation(
            label="Run trusted code",
            code="import math\nderived = recorded_input + math.pi",
        ),
        AssignAttrsOperation(attrs={"trusted": True}),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="recorded_input",
                label="Recorded input",
                provenance_spec=recorded_input.model_dump(mode="json"),
            ),
        ),
    )
    unavailable_specs = (
        opaque_script,
        missing_input_script,
        missing_file,
        detached_live,
    )
    assert all(spec._reorder_sections() for spec in unavailable_specs)
    assert trusted_script._reorder_sections()

    with manager_context() as manager:
        tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(data, manager=False, execute=False),
        )
        manager.add_imagetool(tool, show=False, provenance_spec=opaque_script)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root = manager._tool_graph.root_wrappers[0]
        select_tools(manager, [0])

        for unavailable_spec in unavailable_specs:
            root.set_detached_provenance(unavailable_spec)
            manager._update_info()
            assert not manager._provenance_edit_controller.can_reorder_steps()[0]

        root.set_detached_provenance(opaque_live, live_parent_data=data)
        manager._update_info()
        assert not manager._provenance_edit_controller.can_reorder_steps()[0]

        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert not manager._metadata_reorder_steps_action.isEnabled()

        root.set_detached_provenance(detached_live, live_parent_data=data)
        manager._update_info()
        assert manager._provenance_edit_controller.can_reorder_steps()[0]

        root.set_detached_provenance(trusted_script)
        manager._update_info()
        assert manager._provenance_edit_controller.can_reorder_steps()[0]
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert manager._metadata_reorder_steps_action.isEnabled()


def test_manager_provenance_reorder_controller_tracks_dependencies_and_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Dependency:
        def snapshot_token_for_role(self, role: str) -> str:
            return f"token-{role}"

    manager = types.SimpleNamespace(
        _tool_graph=types.SimpleNamespace(nodes={"available": _Dependency()}),
    )
    controller = _ProvenanceEditController(typing.cast("typing.Any", manager))
    spec = script(
        AssignAttrsOperation(attrs={"first": True}),
        AssignAttrsOperation(attrs={"second": True}),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="first_input",
                node_uid="available",
                data_role="source",
            ),
            ScriptInput(
                name="duplicate_input",
                node_uid="available",
                data_role="source",
            ),
            ScriptInput(name="missing_input", node_uid="missing"),
        ),
    )
    assert controller._dependency_snapshot_tokens(spec) == (
        ("available", "source", "token-source"),
        ("missing", "displayed", None),
    )

    display_node = types.SimpleNamespace(
        parent_uid=None,
        source_spec=None,
        displayed_source_spec=None,
        displayed_provenance_spec=spec,
    )
    assert controller._reorder_target(display_node) == ("display", spec)
    source_node = types.SimpleNamespace(
        parent_uid="parent",
        source_spec=spec,
        displayed_source_spec=spec,
        displayed_provenance_spec=None,
    )
    assert controller._reorder_target(source_node) == ("source", spec)

    no_source_file = _manager_replay_file_spec(pathlib.Path("missing.h5")).model_copy(
        update={"file_load_source": None}
    )
    reason = controller._reorder_replay_unavailable_reason(
        display_node,
        "display",
        no_source_file,
    )
    assert reason is not None

    live_spec = full_data(
        AssignAttrsOperation(attrs={"first": True}),
        AssignAttrsOperation(attrs={"second": True}),
    )
    orphan = types.SimpleNamespace(uid="orphan", parent_uid="deleted")
    assert (
        controller._reorder_replay_unavailable_reason(orphan, "source", live_spec)
        is not None
    )
    manager._tool_graph.nodes["parent"] = object()
    orphan.parent_uid = "parent"
    assert (
        controller._reorder_replay_unavailable_reason(orphan, "source", live_spec)
        is None
    )

    monkeypatch.setattr(controller, "_metadata_node", lambda: None)
    unavailable: list[str] = []
    monkeypatch.setattr(controller, "_show_unavailable", unavailable.append)
    assert not controller.can_reorder_steps()[0]
    controller.open_reorder_dialog()
    assert len(unavailable) == 1

    no_spec_node = types.SimpleNamespace(
        is_imagetool=True,
        imagetool=None,
        pending_workspace_memory_payload=object(),
        parent_uid=None,
        source_spec=None,
        displayed_provenance_spec=None,
    )
    monkeypatch.setattr(controller, "_metadata_node", lambda: no_spec_node)
    assert not controller.can_reorder_steps()[0]

    parent_revision = "parent-revision"
    parent = types.SimpleNamespace(snapshot_token=parent_revision)
    source_revision = "source-revision"
    source_node = types.SimpleNamespace(
        uid="source-node",
        is_imagetool=True,
        imagetool=None,
        pending_workspace_memory_payload=object(),
        parent_uid="parent",
        source_spec=live_spec,
        displayed_source_spec=live_spec,
        displayed_provenance_spec=None,
        snapshot_token=source_revision,
    )
    manager._tool_graph.nodes.update({"parent": parent, source_node.uid: source_node})
    manager._parent_node = lambda _node: parent
    monkeypatch.setattr(controller, "_metadata_node", lambda: source_node)
    opened: list[bool] = []

    class _Signal:
        def connect(self, _slot) -> None:
            pass

    class _Dialog:
        def __init__(self, **_kwargs) -> None:
            self.apply_requested = _Signal()

        def exec(self) -> None:
            opened.append(True)

    monkeypatch.setattr(
        manager_provenance_controller,
        "_ProvenanceReorderDialog",
        _Dialog,
    )
    controller.open_reorder_dialog()
    assert opened == [True]

    data = xr.DataArray([1.0, 2.0], dims="x")
    live_script = full_data(
        ScriptCodeOperation(
            label="Offset data",
            code="derived = derived + 1.0",
        )
    )
    replay_node = types.SimpleNamespace(
        parent_uid=None,
        detached_live_parent_data=data,
    )
    xr.testing.assert_identical(
        controller._replay_live_script_candidate(replay_node, "display", live_script),
        data + 1.0,
    )
    replay_node.uid = "replay-node"
    two_script_steps = full_data(
        *live_script.operations,
        ScriptCodeOperation(
            label="Offset data again",
            code="derived = derived + 1.0",
        ),
    )
    assert (
        controller._reorder_replay_unavailable_reason(
            replay_node,
            "display",
            two_script_steps,
        )
        is None
    )


def test_manager_provenance_reorder_session_rejects_each_stale_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_revision = "node-snapshot"
    parent_revision = "parent-snapshot"
    spec = full_data(
        AssignAttrsOperation(attrs={"first": True}),
        AssignAttrsOperation(attrs={"second": True}),
    )
    node = types.SimpleNamespace(
        uid="node",
        is_imagetool=True,
        imagetool=None,
        pending_workspace_memory_payload=object(),
        parent_uid=None,
        source_spec=None,
        displayed_source_spec=None,
        displayed_provenance_spec=spec,
        snapshot_token=node_revision,
    )
    parent = types.SimpleNamespace(snapshot_token=parent_revision)
    manager = types.SimpleNamespace(
        _tool_graph=types.SimpleNamespace(nodes={"node": node, "parent": parent}),
        _parent_node=lambda _node: parent,
    )
    controller = _ProvenanceEditController(typing.cast("typing.Any", manager))
    sections = spec._reorder_sections()
    session = _ProvenanceReorderSession(
        node_uid=node.uid,
        scope="display",
        spec=spec,
        sections=sections,
        node_snapshot_token=node.snapshot_token,
        parent_snapshot_token=None,
        dependency_snapshot_tokens=(),
    )

    current, reason = controller._reorder_session_current(session)
    assert current is node
    assert not reason

    manager._tool_graph.nodes.pop("node")
    assert controller._reorder_session_current(session)[0] is None
    manager._tool_graph.nodes["node"] = node

    node.displayed_provenance_spec = full_data(
        *spec.operations, RenameOperation(name="x")
    )
    assert controller._reorder_session_current(session)[0] is None
    node.displayed_provenance_spec = spec

    node.snapshot_token = "changed"  # noqa: S105
    assert controller._reorder_session_current(session)[0] is None
    node.snapshot_token = session.node_snapshot_token

    source_session = _ProvenanceReorderSession(
        node_uid=node.uid,
        scope="source",
        spec=spec,
        sections=sections,
        node_snapshot_token=node.snapshot_token,
        parent_snapshot_token=parent.snapshot_token,
        dependency_snapshot_tokens=(),
    )
    node.parent_uid = "parent"
    node.source_spec = spec
    node.displayed_source_spec = spec
    parent.snapshot_token = "changed-parent"  # noqa: S105
    assert controller._reorder_session_current(source_session)[0] is None
    parent.snapshot_token = source_session.parent_snapshot_token

    monkeypatch.setattr(
        controller,
        "_dependency_snapshot_tokens",
        lambda _spec: (("dependency", "displayed", "changed"),),
    )
    assert controller._reorder_session_current(source_session)[0] is None
    monkeypatch.setattr(controller, "_dependency_snapshot_tokens", lambda _spec: ())
    monkeypatch.setattr(controller, "_reorder_sections", lambda _node, _spec: ())
    assert controller._reorder_session_current(source_session)[0] is None


def test_manager_provenance_reorder_cancelled_replay_restores_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_revision = "snapshot"
    spec = full_data(
        AssignAttrsOperation(attrs={"first": True}),
        AssignAttrsOperation(attrs={"second": True}),
    )
    sections = spec._reorder_sections()
    section = sections[0]
    session = _ProvenanceReorderSession(
        node_uid="node",
        scope="display",
        spec=spec,
        sections=sections,
        node_snapshot_token=node_revision,
        parent_snapshot_token=None,
        dependency_snapshot_tokens=(),
    )
    node = object()
    controller = _ProvenanceEditController(typing.cast("typing.Any", object()))
    monkeypatch.setattr(
        controller,
        "_reorder_session_current",
        lambda _session: (node, ""),
    )

    def cancel_replay(*_args, **_kwargs) -> None:
        raise _TrustedScriptReplayCancelled

    monkeypatch.setattr(controller, "_validate_and_replace", cancel_replay)

    class _Dialog:
        def __init__(self) -> None:
            self.busy_states: list[bool] = []

        def reorder_plan(self):
            return {section.ref: tuple(block.ref for block in section.blocks)}

        def set_busy(self, busy: bool) -> None:
            self.busy_states.append(busy)

        def finish_success(self) -> None:
            raise AssertionError("cancelled replay must leave the dialog open")

    dialog = _Dialog()
    controller._apply_reorder_dialog(
        typing.cast("_ProvenanceReorderDialog", dialog),
        session,
    )
    assert dialog.busy_states == [True, False]


def test_manager_provenance_reorder_dialog_controls_and_drop_boundaries(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    staged = _manager_replay_file_spec(
        tmp_path / "scan.h5",
        IselOperation(kwargs={"x": slice(0, 2)}),
        NormalizeOperation(dims=("x",), mode="area"),
    ).append_replay_stage(
        full_data(
            AssignAttrsOperation(attrs={"first": True}),
            AssignAttrsOperation(attrs={"second": True}),
        )
    )
    hidden_boundary = ScriptCodeOperation(
        label="Internal boundary",
        code="derived = derived.copy(deep=False)",
        visible=False,
    )
    spec = script(
        start_label="Run script",
        seed_code="derived = data",
        active_name="derived",
        steps=(
            *staged.steps[:2],
            ReplayStep(operation=hidden_boundary),
            *staged.steps[2:],
            ReplayStep(
                operation=hidden_boundary.model_copy(
                    update={"label": "Second internal boundary"}
                )
            ),
            ReplayStep(operation=AssignAttrsOperation(attrs={"third": True})),
            ReplayStep(operation=AssignAttrsOperation(attrs={"fourth": True})),
        ),
    )
    sections = spec._reorder_sections()
    assert len(sections) == 3

    dialog = _ProvenanceReorderDialog(
        sections=sections,
    )
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.list_stack.count() == len(sections)
    assert dialog.scope_combo.isVisible()
    first_view = dialog.current_view
    second_view = dialog.view_for_section(sections[1].ref)
    assert isinstance(first_view, _ProvenanceReorderListView)
    assert first_view.reorder_model.rowCount() == len(sections[0].blocks)
    assert second_view.reorder_model.rowCount() == len(sections[1].blocks)

    original_first_order = first_view.reorder_model.order()
    original_second_order = second_view.reorder_model.order()
    assert not first_view.move_current(0)
    assert not first_view.move_current(-1)
    first_view.clearSelection()
    first_view.setCurrentIndex(QtCore.QModelIndex())
    assert not first_view.move_current(1)
    first_view.setCurrentIndex(first_view.reorder_model.index(0, 0))

    class _InternalDropEvent(QtGui.QDropEvent):
        def __init__(self, source: QtCore.QObject) -> None:
            super().__init__(
                QtCore.QPointF(),
                QtCore.Qt.DropAction.MoveAction,
                QtCore.QMimeData(),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            self._source = source

        def source(self) -> QtCore.QObject:
            return self._source

    def drop_event(source: QtCore.QObject = first_view) -> _InternalDropEvent:
        return _InternalDropEvent(source)

    first_view.dropEvent(None)
    external_drop = drop_event(dialog)
    first_view.dropEvent(external_drop)
    assert not external_drop.isAccepted()

    first_view.setCurrentIndex(QtCore.QModelIndex())
    no_source_drop = drop_event()
    first_view.dropEvent(no_source_drop)
    assert not no_source_drop.isAccepted()
    first_view.setCurrentIndex(first_view.reorder_model.index(0, 0))

    monkeypatch.setattr(first_view, "indexAt", lambda _position: QtCore.QModelIndex())
    monkeypatch.setattr(
        first_view,
        "dropIndicatorPosition",
        lambda: QtWidgets.QAbstractItemView.DropIndicatorPosition.AboveItem,
    )
    invalid_target_drop = drop_event()
    first_view.dropEvent(invalid_target_drop)
    assert not invalid_target_drop.isAccepted()

    target_index = first_view.reorder_model.index(1, 0)
    monkeypatch.setattr(first_view, "indexAt", lambda _position: target_index)
    monkeypatch.setattr(
        first_view,
        "dropIndicatorPosition",
        lambda: QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem,
    )
    moved_drop = drop_event()
    first_view.dropEvent(moved_drop)
    assert moved_drop.isAccepted()
    assert first_view.reorder_model.rowCount() == len(original_first_order)
    assert set(first_view.reorder_model.order()) == set(original_first_order)
    assert first_view.reorder_model.order() == tuple(reversed(original_first_order))
    assert dialog.order_changed()

    monkeypatch.setattr(first_view, "indexAt", lambda _position: QtCore.QModelIndex())
    monkeypatch.setattr(
        first_view,
        "dropIndicatorPosition",
        lambda: QtWidgets.QAbstractItemView.DropIndicatorPosition.OnViewport,
    )
    viewport_drop = drop_event()
    first_view.dropEvent(viewport_drop)
    assert viewport_drop.isAccepted()
    assert first_view.reorder_model.order() == tuple(reversed(original_first_order))

    monkeypatch.setattr(
        first_view,
        "indexAt",
        lambda _position: first_view.reorder_model.index(0, 0),
    )
    monkeypatch.setattr(
        first_view,
        "dropIndicatorPosition",
        lambda: QtWidgets.QAbstractItemView.DropIndicatorPosition.AboveItem,
    )
    above_drop = drop_event()
    first_view.dropEvent(above_drop)
    assert above_drop.isAccepted()
    assert first_view.reorder_model.order() == original_first_order

    monkeypatch.setattr(
        first_view,
        "dropIndicatorPosition",
        lambda: QtWidgets.QAbstractItemView.DropIndicatorPosition.OnItem,
    )
    on_item_drop = drop_event()
    first_view.dropEvent(on_item_drop)
    assert not on_item_drop.isAccepted()
    assert first_view.move_current(1)
    assert first_view.reorder_model.order() == tuple(reversed(original_first_order))

    second_view.setCurrentIndex(second_view.reorder_model.index(0, 0))
    monkeypatch.setattr(
        second_view,
        "indexAt",
        lambda _position: second_view.reorder_model.index(1, 0),
    )
    monkeypatch.setattr(
        second_view,
        "dropIndicatorPosition",
        lambda: QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem,
    )
    cross_scope_drop = drop_event(first_view)
    second_view.dropEvent(cross_scope_drop)
    assert not cross_scope_drop.isAccepted()
    assert second_view.reorder_model.order() == original_second_order

    first_view.setCurrentIndex(first_view.reorder_model.index(0, 0))
    dialog.set_busy(False)
    dialog.set_busy(True)
    assert not dialog.list_stack.isEnabled()
    assert not dialog.scope_combo.isEnabled()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.cancel_button.isEnabled()
    dialog.reject()
    assert dialog.isVisible()
    close_event = QtGui.QCloseEvent()
    dialog.closeEvent(close_event)
    assert not close_event.isAccepted()
    dialog.set_busy(False)
    assert dialog.apply_button.isEnabled()
    assert dialog.cancel_button.isEnabled()

    dialog.scope_combo.setCurrentIndex(1)
    assert dialog.current_view is second_view
    assert second_view.move_current(1)
    assert dialog.order_changed()
    dialog.reset_order()
    assert first_view.reorder_model.order() == original_first_order
    assert second_view.reorder_model.order() == original_second_order
    assert not dialog.order_changed()
    assert not dialog.apply_button.isEnabled()


def test_manager_provenance_reorder_list_models_keep_atomic_rows(
    qtbot,
) -> None:
    grouped = stamp_operation_group(
        (
            AssignAttrsOperation(attrs={"first": True}),
            AssignAttrsOperation(attrs={"second": True}),
        ),
        kind="test",
        group_id="atomic-reorder-test",
    )
    operation_spec = full_data(
        *grouped,
        AssignAttrsOperation(attrs={"third": True}),
    )
    section = operation_spec._reorder_sections()[0]
    model = _ProvenanceReorderListModel(section)
    assert model.rowCount() == 2
    assert model.rowCount(model.index(0, 0)) == 0
    assert model.data(QtCore.QModelIndex()) is None
    assert model.data(model.index(0, 0), _REORDER_BLOCK_ROLE) == section.blocks[0].ref
    assert model.flags(QtCore.QModelIndex()) & QtCore.Qt.ItemFlag.ItemIsDropEnabled
    assert model.supportedDropActions() == QtCore.Qt.DropAction.MoveAction
    mime_data = model.mimeData([model.index(0, 0)])
    assert mime_data.hasFormat(model.mimeTypes()[0])
    assert model.canDropMimeData(
        mime_data,
        QtCore.Qt.DropAction.MoveAction,
        1,
        -1,
        QtCore.QModelIndex(),
    )
    assert not model.canDropMimeData(
        QtCore.QMimeData(),
        QtCore.Qt.DropAction.MoveAction,
        1,
        -1,
        QtCore.QModelIndex(),
    )

    original = model.order()
    assert model.move_row(0, 1)
    assert model.order() == tuple(reversed(original))
    assert not model.move_row(1, 1)
    assert not model.move_row(-1, 0)
    assert model.reset_order(original)
    assert not model.reset_order(original)
    with pytest.raises(ValueError, match="every provenance block once"):
        model.reset_order(original[:1])

    dialog = _ProvenanceReorderDialog(
        sections=(section,),
    )
    qtbot.addWidget(dialog)
    dialog.show()
    assert not dialog.scope_combo.isVisible()
    assert dialog.list_stack.count() == 1
    assert dialog.current_view.reorder_model.rowCount() == len(section.blocks)
    assert dialog.current_view.move_current(1)
    dialog.reset_order()
    assert dialog.reorder_plan()[section.ref] == original


def test_manager_provenance_reorder_flat_presentation_edge_cases(qtbot) -> None:
    entries = tuple(
        AssignAttrsOperation(attrs={f"value_{index}": index}).derivation_entry()
        for index in range(4)
    )
    blocks = (
        _ProvenanceReorderBlock(
            _ProvenanceReorderBlockRef(0, 1),
            entries,
            tooltip="Stage source details",
        ),
        _ProvenanceReorderBlock(
            _ProvenanceReorderBlockRef(1, 2),
            (),
        ),
        _ProvenanceReorderBlock(
            _ProvenanceReorderBlockRef(2, 5),
            entries[:1],
        ),
    )
    section = _ProvenanceReorderSection(
        _ProvenanceReorderSectionRef(0, 3),
        "Presentation test",
        blocks,
    )
    model = _ProvenanceReorderListModel(section)
    for row in range(model.rowCount()):
        index = model.index(row, 0)
        assert isinstance(
            model.data(index, int(QtCore.Qt.ItemDataRole.DisplayRole)),
            str,
        )
        assert isinstance(
            model.data(index, int(QtCore.Qt.ItemDataRole.ToolTipRole)),
            str,
        )
    assert model.data(model.index(0, 0), 9999) is None
    assert not model.mimeData([QtCore.QModelIndex()]).formats()

    empty_section = _ProvenanceReorderSection(
        _ProvenanceReorderSectionRef(0, 0),
        "Empty",
        (),
    )
    empty_view = _ProvenanceReorderListView(empty_section)
    qtbot.addWidget(empty_view)
    assert not empty_view.currentIndex().isValid()
    empty_view.select_block(None)
    empty_view.select_block(blocks[0].ref)

    single_operation_section = _ProvenanceReorderSection(
        _ProvenanceReorderSectionRef(0, 1),
        "Single operation",
        blocks[:1],
    )
    dialog = _ProvenanceReorderDialog(
        sections=(single_operation_section,),
    )
    qtbot.addWidget(dialog)
    dialog._set_current_scope(-1)

    empty_dialog = _ProvenanceReorderDialog(
        sections=(),
    )
    qtbot.addWidget(empty_dialog)
    assert not empty_dialog.move_up_button.isEnabled()
    assert not empty_dialog.move_down_button.isEnabled()
    with pytest.raises(TypeError, match="current step list"):
        _ = empty_dialog.current_view


def test_manager_provenance_reorder_dialog_is_transactional(
    qtbot,
    accept_dialog,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    source = xr.DataArray(
        [[1.0, 2.0], [3.0, 6.0], [10.0, 20.0]],
        dims=("x", "y"),
        name="scan",
    )
    source.to_netcdf(file_path, engine="h5netcdf")
    spec = _composed_reorder_file_spec(file_path)
    displayed = replay_file_provenance(spec)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        _add_file_replay_tool(manager, displayed, spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root = manager._tool_graph.root_wrappers[0]
        select_tools(manager, [0])
        manager._update_info()

        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        assert manager._metadata_reorder_steps_action.isEnabled()

        controller = manager._provenance_edit_controller
        replay_calls = 0
        original_replay = controller._replay_candidate_result

        def _count_replay(*args, **kwargs):
            nonlocal replay_calls
            replay_calls += 1
            return original_replay(*args, **kwargs)

        monkeypatch.setattr(controller, "_replay_candidate_result", _count_replay)
        before_spec = root.provenance_spec
        before_data = root.slicer_area._data.copy(deep=True)

        def _cancel_reordered(dialog: QtWidgets.QDialog) -> None:
            assert isinstance(dialog, _ProvenanceReorderDialog)
            assert dialog.current_view.move_current(1)
            assert dialog.current_view.move_current(1)
            assert replay_calls == 0
            assert root.provenance_spec == before_spec
            xr.testing.assert_identical(root.slicer_area._data, before_data)
            dialog.reject()

        accept_dialog(
            controller.open_reorder_dialog,
            accept_call=_cancel_reordered,
        )
        assert replay_calls == 0
        assert root.provenance_spec == before_spec
        xr.testing.assert_identical(root.slicer_area._data, before_data)

        def _apply_reordered(dialog: QtWidgets.QDialog) -> None:
            assert isinstance(dialog, _ProvenanceReorderDialog)
            assert dialog.current_view.move_current(1)
            assert dialog.current_view.move_current(1)
            assert replay_calls == 0
            dialog.apply_button.click()

        accept_dialog(
            controller.open_reorder_dialog,
            accept_call=_apply_reordered,
        )
        assert replay_calls == 1
        assert root.provenance_spec is not None
        assert [operation.op for operation in root.provenance_spec.operations] == [
            "normalize",
            "assign_attrs",
            "isel",
        ]
        xr.testing.assert_identical(
            root.slicer_area._data,
            replay_file_provenance(root.provenance_spec),
        )

        tree = manager._workspace_controller.saving._to_datatree()
        saved_spec = json.loads(
            tree["0/imagetool"].attrs["manager_node_provenance_spec"]
        )
        assert [step["operation"]["op"] for step in saved_spec["steps"]] == [
            "normalize",
            "assign_attrs",
            "isel",
        ]

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        for saved_node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", saved_node)
            )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        restored = manager._tool_graph.root_wrappers[0]
        assert restored.provenance_spec is not None
        assert [operation.op for operation in restored.provenance_spec.operations] == [
            "normalize",
            "assign_attrs",
            "isel",
        ]
        xr.testing.assert_identical(
            restored.slicer_area._data,
            replay_file_provenance(restored.provenance_spec),
        )


def test_manager_provenance_reorder_failure_keeps_dialog_and_data(
    qtbot,
    accept_dialog,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    source = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        name="scan",
    )
    source.to_netcdf(file_path, engine="h5netcdf")
    spec = _composed_reorder_file_spec(file_path)
    displayed = replay_file_provenance(spec)

    with manager_context() as manager:
        _add_file_replay_tool(manager, displayed, spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root = manager._tool_graph.root_wrappers[0]
        select_tools(manager, [0])
        manager._update_info()
        before_spec = root.provenance_spec
        before_data = root.slicer_area._data.copy(deep=True)
        failures: list[Exception] = []

        def _fail_validation(*_args, **_kwargs) -> None:
            raise ValueError("invalid reordered recipe")

        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_validate_and_replace",
            _fail_validation,
        )
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_failed",
            lambda _title, exc, **_kwargs: failures.append(exc),
        )

        def _apply_invalid(dialog: QtWidgets.QDialog) -> None:
            assert isinstance(dialog, _ProvenanceReorderDialog)
            assert dialog.current_view.move_current(1)
            dialog.apply_button.click()
            assert dialog.isVisible()
            assert dialog.apply_button.isEnabled()
            dialog.reject()

        accept_dialog(
            manager._provenance_edit_controller.open_reorder_dialog,
            accept_call=_apply_invalid,
        )

        assert len(failures) == 1
        assert root.provenance_spec == before_spec
        xr.testing.assert_identical(root.slicer_area._data, before_data)


def test_manager_provenance_reorder_rejects_stale_dialog(
    qtbot,
    accept_dialog,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    source = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        name="scan",
    )
    source.to_netcdf(file_path, engine="h5netcdf")
    spec = _composed_reorder_file_spec(file_path)
    displayed = replay_file_provenance(spec)

    with manager_context() as manager:
        _add_file_replay_tool(manager, displayed, spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root = manager._tool_graph.root_wrappers[0]
        select_tools(manager, [0])
        manager._update_info()
        apply_calls = 0
        messages: list[str] = []

        def _record_apply(*_args, **_kwargs) -> None:
            nonlocal apply_calls
            apply_calls += 1

        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_validate_and_replace",
            _record_apply,
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            lambda _parent, _title, text: messages.append(text),
        )

        replacement = displayed + 10.0

        def _apply_after_data_change(dialog: QtWidgets.QDialog) -> None:
            assert isinstance(dialog, _ProvenanceReorderDialog)
            assert dialog.current_view.move_current(1)
            root.replace_with_detached_data(replacement, spec)
            dialog.apply_button.click()
            assert dialog.isVisible()
            dialog.reject()

        accept_dialog(
            manager._provenance_edit_controller.open_reorder_dialog,
            accept_call=_apply_after_data_change,
        )

        assert apply_calls == 0
        assert len(messages) == 1
        assert root.provenance_spec == spec
        xr.testing.assert_identical(root.slicer_area._data, replacement)


def test_manager_provenance_unresolved_script_prefix_blocks_structured_edit(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape((2, 3)),
        dims=("x", "y"),
        name="scan",
    )
    spec = script(
        ScriptCodeOperation(label="Copy source", code="derived = data"),
        QSelAggregationOperation(dims=("x",), func="mean"),
        start_label="Run script",
        active_name="derived",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = itool(data.qsel.mean("x"), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [2])
        row = manager._selected_derivation_row()
        assert row is not None
        assert row.edit_ref is not None

        editable, _reason = manager._provenance_edit_controller.can_edit_row(row)
        assert not editable


def test_manager_provenance_script_structured_row_can_revert(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(3 * 4 * 2 * 2, dtype=float).reshape((3, 4, 2, 2)),
        dims=("x", "y", "z", "w"),
        coords={
            "x": [0.0, 1.0, 2.0],
            "y": np.arange(4),
            "z": [0.0, 1.0],
            "w": [0.0, 1.0],
        },
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    file_spec = _manager_replay_file_spec(file_path)
    spec = script(
        ScriptCodeOperation(
            label="Use source",
            code="derived = source.copy()",
        ),
        QSelAggregationOperation(dims=("x",), func="mean"),
        IselOperation(kwargs={"y": 0}),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="source",
                label="Recorded source",
                provenance_spec=file_spec,
            ),
        ),
    )
    initial = replay_script_provenance(spec, {})

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = itool(initial, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False, provenance_spec=spec)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root = manager._tool_graph.root_wrappers[0]
        select_tools(manager, [0])
        manager._update_info()
        rows = root.derivation_display_rows
        aggregate_row = next(
            row
            for row in rows
            if row.replay_ref == _ProvenanceStepRef("operation", operation_index=1)
        )
        aggregate_item = None
        for row_index in range(manager.metadata_derivation_list.count()):
            item = manager.metadata_derivation_list.item(row_index)
            if (
                item is not None
                and item.data(manager_details_panel._METADATA_DERIVATION_ROW_ROLE)
                == aggregate_row
            ):
                aggregate_item = item
                break
        assert aggregate_item is not None
        select_metadata_rows(
            manager,
            [manager.metadata_derivation_list.row(aggregate_item)],
        )
        row = manager._selected_derivation_row()
        assert row is not None

        revertible, _reason = manager._provenance_edit_controller.can_revert_row(row)
        assert revertible

        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_confirm_revert",
            lambda: True,
        )
        manager._revert_selected_derivation_step()

        assert root.provenance_spec is not None
        assert [operation.op for operation in root.provenance_spec.operations] == [
            "script_code",
            "qsel_aggregate",
        ]
        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )


def test_manager_provenance_structured_operation_edit_accept_and_cancel(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(24, dtype=float).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": np.arange(4), "z": [0.0, 1.0]},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        QSelAggregationOperation(dims=("y",), func="mean"),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = _add_file_replay_tool(
            manager,
            replay_file_provenance(spec),
            spec,
        )
        root = manager._tool_graph.root_wrappers[0]

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        row = manager._selected_derivation_row()
        assert row is not None
        assert manager._provenance_edit_controller.can_edit_row(row)[0]

        before_spec = root.provenance_spec
        before_data = tool.slicer_area._data.copy(deep=True)
        accept_dialog(
            manager._edit_selected_derivation_step,
            accept_call=lambda dialog: dialog.reject(),
        )
        assert root.provenance_spec == before_spec
        xr.testing.assert_identical(tool.slicer_area._data, before_data)

        def _edit_aggregate(dialog: QtWidgets.QDialog) -> None:
            _set_aggregate(dialog, dims=("x",), func="sum")

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_aggregate)

        assert root.provenance_spec is not None
        assert root.provenance_spec.operations == (
            QSelAggregationOperation(dims=("x",), func="sum"),
        )
        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            data.qsel.sum("x").rename(None),
        )


def test_manager_provenance_script_derived_structured_step_is_editable(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(24, dtype=float).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": np.arange(4), "z": [0.0, 1.0]},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        input_tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(data, manager=False, execute=False),
        )
        manager.add_imagetool(input_tool, show=False)
        input_node = manager._tool_graph.root_wrappers[0]
        derived_data = (data + 1.0).qsel.mean("y")
        spec = script(
            ScriptCodeOperation(
                label="Evaluate console expression",
                code="derived = data_0 + 1.0",
            ),
            QSelAggregationOperation(dims=("y",), func="mean"),
            start_label="Run ImageTool manager console code",
            active_name="derived",
            script_inputs=(
                ScriptInput(
                    name="data_0",
                    label="ImageTool 0: scan",
                    node_uid=input_node.uid,
                    node_snapshot_token=input_node.snapshot_token,
                ),
            ),
        )
        derived_tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(derived_data, manager=False, execute=False),
        )
        manager.add_imagetool(derived_tool, show=False, provenance_spec=spec)
        derived_node = manager._tool_graph.root_wrappers[1]

        select_tools(manager, [1])
        manager._update_info()
        script_code_row, structured_row = spec.display_rows()[2:]
        assert manager._provenance_edit_controller.can_edit_row(script_code_row) == (
            True,
            "",
        )
        assert manager._provenance_edit_controller.can_edit_row(structured_row) == (
            True,
            "",
        )
        select_metadata_rows(manager, [3])

        def _edit_aggregate(dialog: QtWidgets.QDialog) -> None:
            _set_aggregate(dialog, dims=("x",), func="sum")

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_aggregate)

        assert derived_node.provenance_spec is not None
        assert derived_node.provenance_spec.operations == (
            ScriptCodeOperation(
                label="Evaluate console expression",
                code="derived = data_0 + 1.0",
            ),
            QSelAggregationOperation(dims=("x",), func="sum"),
        )
        xr.testing.assert_identical(
            derived_tool.slicer_area._data.rename(None),
            (data + 1.0).qsel.sum("x").rename(None),
        )


def test_manager_provenance_active_filter_edit_accept_and_cancel(
    qtbot,
    accept_dialog,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(24, dtype=float).reshape((3, 4, 2)) + 1.0,
        dims=("x", "y", "z"),
        coords={"x": [0.0, 1.0, 2.0], "y": np.arange(4), "z": [0.0, 1.0]},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(file_path)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        tool = _add_file_replay_tool(manager, data, spec)
        operation = NormalizeOperation(dims=("x",), mode="area")
        tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])
        row = manager._selected_derivation_row()
        assert row is not None
        assert manager._provenance_edit_controller.can_edit_row(row)[0]

        before_source = tool.slicer_area._data.copy(deep=True)
        before_filter = tool.slicer_area._accepted_filter_provenance_operation
        accept_dialog(
            manager._edit_selected_derivation_step,
            accept_call=lambda dialog: dialog.reject(),
        )
        xr.testing.assert_identical(tool.slicer_area._data, before_source)
        assert tool.slicer_area._accepted_filter_provenance_operation == before_filter

        def _edit_filter(dialog: QtWidgets.QDialog) -> None:
            for check in dialog.dim_checks.values():  # type: ignore[attr-defined]
                check.setChecked(False)
            dialog.dim_checks["y"].setChecked(True)  # type: ignore[attr-defined]
            dialog.opts[2].setChecked(True)  # type: ignore[attr-defined]

        accept_dialog(manager._edit_selected_derivation_step, pre_call=_edit_filter)

        assert tool.slicer_area._accepted_filter_provenance_operation == (
            NormalizeOperation(dims=("y",), mode="min")
        )
        xr.testing.assert_identical(tool.slicer_area._data, before_source)


def test_manager_provenance_edit_rejects_incompatible_downstream_and_reverts(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(48, dtype=float).reshape((3, 4, 2, 2)) + 1.0,
        dims=("x", "y", "z", "w"),
        coords={
            "x": [1.0, 2.0, 4.0],
            "y": np.arange(4),
            "z": [0.0, 1.0],
            "w": [0.0, 1.0],
        },
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    spec = _manager_replay_file_spec(
        file_path,
        QSelAggregationOperation(dims=("x",), func="mean"),
        IselOperation(kwargs={"y": 0}),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        initial = replay_file_provenance(spec)
        tool = _add_file_replay_tool(manager, initial, spec)
        root = manager._tool_graph.root_wrappers[0]

        select_tools(manager, [0])
        manager._update_info()
        select_metadata_rows(manager, [1])

        before_spec = root.provenance_spec
        before_data = tool.slicer_area._data.copy(deep=True)
        failures: list[tuple[str, Exception]] = []
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_show_failed",
            lambda title, exc: failures.append((title, exc)),
        )
        monkeypatch.setattr(
            manager._provenance_edit_controller,
            "_edited_native_operations",
            lambda *_args, **_kwargs: [
                QSelAggregationOperation(dims=("y",), func="mean")
            ],
        )

        manager._edit_selected_derivation_step()
        assert failures
        assert root.provenance_spec == before_spec
        xr.testing.assert_identical(tool.slicer_area._data, before_data)

        accept_dialog(manager._revert_selected_derivation_step)
        assert root.provenance_spec == before_spec
        xr.testing.assert_identical(tool.slicer_area._data, before_data)

        def _confirm_revert(dialog: QtWidgets.QDialog) -> None:
            button = typing.cast(
                "QtWidgets.QMessageBox",
                dialog,
            ).button(QtWidgets.QMessageBox.StandardButton.Yes)
            assert button is not None
            button.click()

        accept_dialog(
            manager._revert_selected_derivation_step,
            accept_call=_confirm_revert,
        )

        assert root.provenance_spec is not None
        assert root.provenance_spec.operations == (
            QSelAggregationOperation(dims=("x",), func="mean"),
        )
        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )
