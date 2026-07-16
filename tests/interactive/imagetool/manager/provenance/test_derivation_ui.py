import pathlib
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
    ScriptInput,
    _ProvenanceDisplayRow,
    _ProvenanceStepRef,
    full_data,
    script,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    IselOperation,
    NormalizeOperation,
    QSelAggregationOperation,
    ScriptCodeOperation,
)
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
    displayed = operation.apply(
        test_data.qsel.mean("alpha"),
        parent_data=test_data.qsel.mean("alpha"),
    )

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
        assert actions[1] is manager._metadata_revert_step_action
        assert actions[2].isSeparator()
        assert actions[3] is manager._metadata_copy_selected_action
        assert actions[4] is manager._metadata_paste_steps_action
        if manager._metadata_copy_full_action in actions:
            assert actions[5] is manager._metadata_copy_full_action
            separator_index = 6
        else:
            separator_index = 5
        assert actions[separator_index].isSeparator()
        assert actions[separator_index + 1] is manager._metadata_delete_step_action


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
        stage = root.provenance_spec.replay_stages[0]
        assert stage.operations == (QSelAggregationOperation(dims=("x",), func="sum"),)
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
        assert root.provenance_spec.replay_stages[0].operations == (
            QSelAggregationOperation(dims=("x",), func="mean"),
        )
        xr.testing.assert_identical(
            tool.slicer_area._data.rename(None),
            data.qsel.mean("x").rename(None),
        )
