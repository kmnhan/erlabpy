import typing
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._stylesheets
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureOperationState,
    FigureSourceState,
)
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool.manager._figurecomposer import _dialogs
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    select_child_tool,
    select_tools,
)

from ._common import (
    _materialized_figure_tool,
    _render_figure_composer_rgba,
    _SourcePickerDummyTool,
)

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool.manager._modelview import (
        _ImageToolWrapperItemModel,
    )


def _source_picker_item(
    dialog: QtWidgets.QDialog, uid: str
) -> QtWidgets.QTreeWidgetItem:
    tree = dialog.findChild(QtWidgets.QTreeWidget, "managerFigureSourcePickerTree")
    assert tree is not None
    root = tree.invisibleRootItem()
    stack = [root.child(index) for index in range(root.childCount())]
    while stack:
        item = stack.pop()
        assert item is not None
        if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == uid:
            return item
        stack.extend(item.child(index) for index in range(item.childCount()))
    raise AssertionError(f"missing picker row for {uid!r}")


def test_manager_figure_action_replace_source_keeps_recipe_steps(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operation_count = len(figure_tool.tool_status.operations)

        select_tools(manager, [1])

        class FakeReplaceDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_REPLACE_SOURCE

            def figure_uid(self) -> str:
                return figure_uid

            def selected_source_alias(self) -> str:
                return "first"

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeReplaceDialog)

        manager.create_figure_action.trigger()

        assert len(figure_tool.tool_status.operations) == operation_count
        xr.testing.assert_identical(figure_tool.source_data()["first"], second)
        [source] = figure_tool.source_states()
        assert source.name == "first"
        assert source.node_uid == manager._node_for_target(1).uid
        assert "second" not in figure_tool.source_data()


def test_manager_figure_sources_reveal_associated_imagetool_rows(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2), dims=("x", "y"), name="first"
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2), dims=("x", "y"), name="second"
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        assert not manager._figure_workflows._reveal_figure_sources(
            "missing", ("first",)
        )

        def raise_reveal_error(_source_name: str) -> bool:
            raise RuntimeError("source is unavailable")

        original_reveal_available = figure_tool._source_reveal_available_callback
        figure_tool._source_reveal_available_callback = raise_reveal_error
        assert not figure_tool._source_reveal_available("first")
        figure_tool._source_reveal_available_callback = original_reveal_available

        figure_tool.source_panel.set_selected_names(("first",), current_name="first")
        figure_tool.source_panel.duplicate_requested.emit(("first",))
        source_names = {source.name for source in figure_tool.source_states()}
        figure_tool.source_panel.set_selected_names(
            tuple(source_names), current_name="first"
        )
        figure_tool._refresh_source_controls()
        assert figure_tool.source_panel.reveal_sources_button.isEnabled()
        assert not figure_tool.source_panel.reveal_sources_button.autoRaise()
        assert (
            figure_tool.source_panel.reveal_sources_button.toolButtonStyle()
            == QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )

        figure_tool.source_panel.reveal_sources_button.click()

        assert manager.tree_view.selected_imagetool_indices == [0, 1]
        assert manager.left_tabs.currentWidget() is manager.tree_view

        manager.tree_view.clearSelection()
        figure_tool.source_panel.source_list.context_menu_requested.emit(
            QtCore.QPoint(0, 0)
        )
        reveal_action = figure_tool.source_panel.source_list.findChild(
            QtGui.QAction, "figureComposerContextRevealSourceAction"
        )
        assert reveal_action is not None
        source_menu = reveal_action.parent()
        assert isinstance(source_menu, QtWidgets.QMenu)
        assert reveal_action.isEnabled()
        reveal_action.trigger()
        assert manager.tree_view.selected_imagetool_indices == [0, 1]
        source_menu.close()

        manager.remove_imagetool(1)
        figure_tool.source_panel.set_selected_names(("second",), current_name="second")
        figure_tool._refresh_source_controls()
        assert not figure_tool.source_panel.reveal_sources_button.isEnabled()
        figure_tool.source_panel.reveal_requested.emit(("second",))


def test_manager_figure_source_row_refresh_updates_only_selected_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        original_second = figure_tool.source_data()["second"]
        operations = figure_tool.tool_status.operations
        manager._workspace_controller._mark_workspace_clean()

        updated_first = first.copy(data=np.asarray(first.data) + 100.0, deep=True)
        updated_first.name = "first updated"
        updated_second = second.copy(data=np.asarray(second.data) + 200.0, deep=True)
        updated_second.name = "second updated"
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_first, manager=True, replace=0)
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_second, manager=True, replace=1)

        figure_tool.source_panel.set_selected_names(("first",), current_name="first")
        figure_tool._refresh_source_controls()
        assert figure_tool.source_panel.refresh_sources_button.isEnabled()
        with qtbot.wait_signal(figure_tool.sigDataChanged, timeout=5000):
            figure_tool.source_panel.refresh_sources_button.click()

        source_data = figure_tool.source_data()
        xr.testing.assert_identical(source_data["first"], updated_first)
        xr.testing.assert_identical(source_data["second"], original_second)
        [source_0, source_1] = figure_tool.source_states()
        assert source_0.name == "first"
        assert source_0.node_uid == manager._node_for_target(0).uid
        assert source_1.name == "second"
        assert figure_tool.tool_status.operations == operations
        assert "first" in figure_tool.generated_code()

        _render_figure_composer_rgba(figure_tool)
        namespace = _exec_generated_code(
            figure_tool.generated_code(),
            {"first": updated_first, "second": original_second},
        )
        assert isinstance(namespace["fig"], Figure)
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        manager.remove_imagetool(0)
        figure_tool._refresh_source_controls()
        assert not figure_tool.source_panel.refresh_sources_button.isEnabled()
        first_item = figure_tool.source_panel.source_list.topLevelItem(0)
        assert first_item is not None
        assert first_item.icon(0).isNull()


def test_manager_figure_refresh_sources_updates_live_sources_and_skips_detached(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        detached = xr.DataArray(
            np.array([1.0, 2.0]),
            dims=("x",),
            coords={"x": [0.0, 1.0]},
            name="detached",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        figure_tool.add_sources(
            (FigureSourceState(name="detached", label="Detached"),),
            {"detached": detached},
        )
        operations = figure_tool.tool_status.operations
        manager._workspace_controller._mark_workspace_clean()

        updated_first = first.copy(data=np.asarray(first.data) + 100.0, deep=True)
        updated_first.name = "first updated"
        updated_second = second.copy(data=np.asarray(second.data) + 200.0, deep=True)
        updated_second.name = "second updated"
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_first, manager=True, replace=0)
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_second, manager=True, replace=1)

        figure_tool.source_panel.set_selected_names(
            ("first", "second", "detached"), current_name="first"
        )
        figure_tool._refresh_source_controls()
        assert figure_tool.source_panel.refresh_sources_button.isEnabled()
        figure_tool.source_panel.refresh_sources_button.click()

        source_data = figure_tool.source_data()
        xr.testing.assert_identical(source_data["first"], updated_first)
        xr.testing.assert_identical(source_data["second"], updated_second)
        xr.testing.assert_identical(source_data["detached"], detached)
        assert figure_tool.tool_status.operations == operations
        assert not figure_tool.source_panel.source_status_label.isHidden()
        assert figure_tool._operation_render_errors == {}
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]


def test_manager_figure_sources_use_readable_unique_aliases(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    arrays = (
        xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"), name="sample_map"),
        xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            name="reference map",
        ),
        xr.DataArray(
            np.arange(8.0, 12.0).reshape(2, 2),
            dims=("x", "y"),
            name="sample_map",
        ),
        xr.DataArray(
            np.arange(12.0, 16.0).reshape(2, 2),
            dims=("x", "y"),
            name=" !!! ",
        ),
    )

    with manager_context() as manager:
        for data in arrays:
            itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == len(arrays), timeout=5000)

        figure_uid = manager.create_figure_from_targets(range(len(arrays)), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)

        source_names = tuple(source.name for source in figure_tool.source_states())
        assert source_names == (
            "sample_map",
            "reference_map",
            "sample_map_2",
            "data_3",
        )
        assert tuple(figure_tool.source_data()) == source_names
        reference_item = figure_tool.source_panel.source_list.topLevelItem(1)
        assert reference_item is not None
        assert (
            reference_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "reference_map"
        )
        assert "Original name: reference map" in reference_item.toolTip(0)


def test_manager_figure_source_only_append_uses_readable_conflict_suffix(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"), name="map")
    second = xr.DataArray(
        np.arange(4.0, 8.0).reshape(2, 2), dims=("x", "y"), name="map"
    )

    with manager_context() as manager:
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)

        _resolved_targets, sources, source_data = (
            manager._figure_workflows._figure_sources_from_targets((1,))
        )
        manager._figure_workflows._add_sources_to_figure(
            figure_uid, sources, source_data, show=False
        )

        assert tuple(source.name for source in figure_tool.source_states()) == (
            "map",
            "map_2",
        )
        xr.testing.assert_identical(figure_tool.source_data()["map_2"], second)


def test_manager_figure_source_helper_edge_contracts(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="data",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        image_uid = manager._node_for_target(0).uid
        child = itool(data.sum("y"), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        select_child_tool(manager, figure_uid)
        assert (
            manager._figure_workflows._selected_figure_uid_for_figure_dialog()
            == figure_uid
        )

        source_states = figure_tool.source_states()
        source_data = figure_tool.source_data()
        source_alias = source_states[0].name
        replacement_source = FigureSourceState(
            name="replacement",
            label="replacement",
            node_uid=image_uid,
        )
        assert not manager._figure_workflows._add_sources_to_figure(
            "missing",
            source_states,
            source_data,
            show=False,
        )
        assert (
            manager._figure_workflows._figure_source_state(figure_tool, "missing")
            is None
        )
        assert (
            manager._figure_workflows._figure_source_live_node("missing", source_alias)
            is None
        )
        assert not manager._figure_workflows._refresh_figure_source(
            figure_uid, "missing"
        )
        assert not manager._figure_workflows._replace_figure_source(
            figure_uid,
            source_alias,
            (),
            {},
            show=False,
        )
        assert not manager._figure_workflows._replace_figure_source(
            figure_uid,
            source_alias,
            (replacement_source,),
            {},
            show=False,
        )

        with monkeypatch.context() as context:
            context.setattr(
                manager,
                "_is_figure_uid",
                lambda uid: uid in {figure_uid, child_uid},
            )
            assert not manager._figure_workflows._add_sources_to_figure(
                child_uid,
                source_states,
                source_data,
                show=False,
            )
            assert not manager._figure_workflows._replace_figure_source(
                child_uid,
                source_alias,
                (replacement_source,),
                {"replacement": data},
                show=False,
            )
            assert (
                manager._figure_workflows._figure_source_live_node(
                    child_uid, source_alias
                )
                is None
            )
            assert not manager._figure_workflows._refresh_figure_source(
                child_uid, source_alias
            )

        with monkeypatch.context() as context:
            context.setattr(figure_tool, "replace_source", lambda *_args: False)
            assert not manager._figure_workflows._replace_figure_source(
                figure_uid,
                source_alias,
                (replacement_source,),
                {"replacement": data},
                show=False,
            )
            assert not manager._figure_workflows._refresh_figure_source(
                figure_uid, source_alias
            )

        with monkeypatch.context() as context:
            context.setattr(
                manager._figure_workflows,
                "_figure_source_live_node",
                lambda *_args: manager._node_for_target(0),
            )
            context.setattr(manager, "_is_figure_uid", lambda uid: uid == child_uid)
            assert not manager._figure_workflows._refresh_figure_source(
                child_uid, source_alias
            )

        with monkeypatch.context() as context:
            context.setattr(manager, "_figure_uids", lambda: ("missing",))
            manager._figure_workflows._refresh_figure_source_controls()

        with monkeypatch.context() as context:
            context.setattr(manager, "_selected_figure_source_targets", lambda: (0,))
            context.setattr(
                manager._figure_workflows,
                "_figure_sources_from_targets",
                lambda _targets: ((), (), {}),
            )
            manager.create_figure_from_selection()

        dialog_events: list[str] = []

        class AliasNoneDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                _figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid == figure_uid
                dialog_events.append("init")

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_REPLACE_SOURCE

            def selected_source_alias(self) -> str | None:
                dialog_events.append("alias")
                return None

        with monkeypatch.context() as context:
            context.setattr(manager, "_selected_figure_source_targets", lambda: (0,))
            context.setattr(manager, "_figure_uids", lambda: (figure_uid,))
            context.setattr(
                manager._figure_workflows,
                "_figure_sources_from_targets",
                lambda _targets: ((0,), source_states, source_data),
            )
            context.setattr(
                manager._figure_workflows,
                "_selected_figure_uid_for_figure_dialog",
                lambda: figure_uid,
            )
            context.setattr(
                _dialogs,
                "_AppendFigureTargetDialog",
                AliasNoneDialog,
            )
            manager.create_figure_from_selection()
        assert dialog_events == ["init", "alias"]


def test_manager_figure_action_add_source_only_keeps_recipe_steps(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operation_count = len(figure_tool.tool_status.operations)
        workspace_path = tmp_path / "add-source-only-delta.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )

        select_tools(manager, [1])

        class FakeSourceOnlyDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_ADD_SOURCE

            def figure_uid(self) -> str:
                return figure_uid

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeSourceOnlyDialog)

        manager.create_figure_action.trigger()

        assert len(figure_tool.tool_status.operations) == operation_count
        xr.testing.assert_identical(figure_tool.source_data()["second"], second)
        assert {source.name for source in figure_tool.source_states()} == {
            "first",
            "second",
        }
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        manager._workspace_controller.saving._save_workspace_document(workspace_path)
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded_node = manager._child_node(figure_uid)
        loaded_tool = loaded_node.tool_window
        if loaded_tool is None:
            assert loaded_node.pending_workspace_tool_payload is not None
            loaded_node.show()
            loaded_tool = loaded_node.tool_window
        assert isinstance(loaded_tool, FigureComposerTool)
        xr.testing.assert_identical(loaded_tool.source_data()["second"], second)


def test_manager_figure_source_picker_selects_imagetool_rows_only(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    root_data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="root",
    )
    child_data = root_data + 10.0
    with manager_context() as manager:
        itool(root_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root_uid = manager._tool_graph.root_wrappers[0].uid
        dummy_uid = manager.add_childtool(
            _SourcePickerDummyTool(root_data.rename("dummy")),
            0,
            show=False,
        )
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(
                child_data.rename("child"), _in_manager=True
            ),
            dummy_uid,
            show=False,
        )

        dialog = _dialogs._FigureSourcePickerDialog(
            manager, prechecked_uids=(root_uid, child_uid)
        )
        qtbot.addWidget(dialog)
        root_item = _source_picker_item(dialog, root_uid)
        dummy_item = _source_picker_item(dialog, dummy_uid)
        child_item = _source_picker_item(dialog, child_uid)

        checkable = QtCore.Qt.ItemFlag.ItemIsUserCheckable
        assert root_item.flags() & checkable
        assert child_item.flags() & checkable
        assert not dummy_item.flags() & checkable
        assert root_item.checkState(0) == QtCore.Qt.CheckState.Checked
        assert child_item.checkState(0) == QtCore.Qt.CheckState.Checked
        assert dialog.selected_targets() == (root_uid, child_uid)

        assert dummy_item.isExpanded()
        dummy_item.setExpanded(False)
        dialog.search_edit.setText("child")
        assert not root_item.isHidden()
        assert not dummy_item.isHidden()
        assert not child_item.isHidden()
        assert dummy_item.isExpanded()
        assert dialog.selected_targets() == (root_uid, child_uid)
        dialog.search_edit.clear()
        assert not dummy_item.isExpanded()
        assert root_item.checkState(0) == QtCore.Qt.CheckState.Checked
        assert child_item.checkState(0) == QtCore.Qt.CheckState.Checked

        root_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        child_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        ok_button = dialog.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        assert ok_button is not None
        assert not ok_button.isEnabled()

        assert manager.create_figure_from_targets((dummy_uid,), show=False) is None
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        assert not manager.append_figure_from_targets(
            (dummy_uid,), figure_uid=figure_uid, show=False
        )
        resolved_targets, sources, source_data = (
            manager._figure_workflows._figure_sources_from_targets(
                (dummy_uid, child_uid)
            )
        )
        assert resolved_targets == (child_uid,)
        assert len(sources) == len(source_data) == 1


def test_figure_sources_add_button_adds_imagetool_sources(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="first",
    )
    second = first + 10.0
    with manager_context() as manager:
        itool(first, manager=True)
        itool(second.rename("second"), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operation_count = len(figure_tool.tool_status.operations)
        second_uid = manager._tool_graph.root_wrappers[1].uid
        select_tools(manager, [1])
        manager._workspace_controller._mark_workspace_clean()

        class RejectingPicker:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                *,
                prechecked_uids: Iterable[str] = (),
            ) -> None:
                assert tuple(prechecked_uids) == (second_uid,)

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Rejected

        monkeypatch.setattr(_dialogs, "_FigureSourcePickerDialog", RejectingPicker)
        figure_tool.source_panel.add_source_button.click()

        assert len(figure_tool.tool_status.operations) == operation_count
        assert {source.name for source in figure_tool.source_states()} == {"first"}
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid not in snapshot["dirty_data"]
        assert figure_uid not in snapshot["dirty_state"]

        class AcceptingPicker(RejectingPicker):
            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_targets(self) -> tuple[str, ...]:
                return (second_uid,)

        monkeypatch.setattr(_dialogs, "_FigureSourcePickerDialog", AcceptingPicker)
        with qtbot.wait_signal(figure_tool.sigDataChanged, timeout=5000):
            figure_tool.source_panel.add_source_button.click()

        assert len(figure_tool.tool_status.operations) == operation_count
        xr.testing.assert_identical(
            figure_tool.source_data()["second"], second.rename("second")
        )
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]


def test_manager_figure_source_add_reports_partial_rejection(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="first",
    )
    second = first.copy(data=np.arange(4.0, 8.0).reshape(2, 2)).rename("second")
    incompatible = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("z", "y"),
        coords={"z": [0.0, 1.0], "y": [0.0, 1.0]},
        name="replacement",
    )
    with manager_context() as manager:
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        figure_tool.source_panel.set_selected_names(("first",), current_name="first")
        figure_tool.source_panel.selection_dimension_requested.emit(
            ("first",), "x", "qsel", "0.0", ""
        )
        original_first = figure_tool.source_data()["first"]
        original_operations = figure_tool.tool_status.operations

        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(incompatible, manager=True, replace=0)
        manager._workspace_controller._mark_workspace_clean()

        assert manager._figure_workflows._add_imagetool_sources_to_figure(
            figure_uid, (0, 1), show=False
        )

        xr.testing.assert_identical(figure_tool.source_data()["first"], original_first)
        xr.testing.assert_identical(figure_tool.source_data()["second"], second)
        assert figure_tool.tool_status.operations == original_operations
        assert not figure_tool.source_panel.source_status_label.isHidden()
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        manager._workspace_controller._mark_workspace_clean()
        assert manager.append_figure_from_targets(
            (0, 1),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )
        assert figure_tool.tool_status.operations == original_operations
        assert not figure_tool.source_panel.source_status_label.isHidden()
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]


def test_manager_rejected_source_update_does_not_mark_dirty_or_add_step(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    incompatible = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("z", "y"),
        coords={"z": [0.0, 1.0], "y": [0.0, 1.0]},
        name="replacement",
    )
    with manager_context() as manager:
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        figure_tool.source_panel.set_selected_names(("data",), current_name="data")
        figure_tool.source_panel.selection_dimension_requested.emit(
            ("data",), "x", "qsel", "0.0", ""
        )
        original_data = figure_tool.source_data()["data"]
        original_operations = figure_tool.tool_status.operations

        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(incompatible, manager=True, replace=0)
        manager._workspace_controller._mark_workspace_clean()

        assert not manager._figure_workflows._add_imagetool_sources_to_figure(
            figure_uid, (0,), show=False
        )
        xr.testing.assert_identical(figure_tool.source_data()["data"], original_data)
        assert not figure_tool.source_panel.source_status_label.isHidden()
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid not in snapshot["dirty_data"]
        assert figure_uid not in snapshot["dirty_state"]

        assert not manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )

        xr.testing.assert_identical(figure_tool.source_data()["data"], original_data)
        assert figure_tool.tool_status.operations == original_operations
        assert not figure_tool.source_panel.source_status_label.isHidden()
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid not in snapshot["dirty_data"]
        assert figure_uid not in snapshot["dirty_state"]


def test_figure_sources_drag_mime_adds_root_and_child_imagetools(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="first",
    )
    second = first + 10.0
    child_data = first + 20.0
    with manager_context() as manager:
        itool(first, manager=True)
        itool(second.rename("second"), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(
                child_data.rename("child"), _in_manager=True
            ),
            0,
            show=False,
        )
        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        second_uid = manager._tool_graph.root_wrappers[1].uid
        second_mime = model.mimeData([model.index(1, 0)])

        assert manager.tree_view.figure_source_uids_from_mime(second_mime) == (
            second_uid,
        )
        assert figure_tool._source_drop_available(second_mime)
        assert figure_tool._add_sources_from_mime(second_mime)
        xr.testing.assert_identical(
            figure_tool.source_data()["second"], second.rename("second")
        )

        child_mime = model.mimeData([model._row_index(child_uid)])
        assert manager.tree_view.figure_source_uids_from_mime(child_mime) == (
            child_uid,
        )
        window = figure_tool.figure_window
        assert window._handle_source_drag_event(None) is False
        assert figure_tool._source_drop_available(child_mime)
        assert figure_tool._add_sources_from_mime(child_mime)
        child_source_name = next(
            source.name
            for source in figure_tool.source_states()
            if source.node_uid == child_uid
        )
        xr.testing.assert_identical(
            figure_tool.source_data()[child_source_name], child_data.rename("child")
        )

        assert not manager._figure_workflows._add_imagetool_sources_to_figure(
            figure_uid, (figure_uid,), show=False
        )
        assert not manager._figure_workflows._request_add_sources_to_figure(
            "missing-figure"
        )
        assert not manager._figure_workflows._add_figure_sources_from_mime(
            figure_uid, QtCore.QMimeData()
        )

        original_add_sources = manager._figure_workflows._add_sources_to_figure
        monkeypatch.setattr(
            manager._figure_workflows,
            "_add_sources_to_figure",
            lambda *_args, **_kwargs: False,
        )
        assert not manager._figure_workflows._add_imagetool_sources_to_figure(
            figure_uid, (second_uid,), show=False
        )
        monkeypatch.setattr(
            manager._figure_workflows, "_add_sources_to_figure", original_add_sources
        )

        source_names = tuple(figure_tool.source_data())
        assert manager._figure_workflows._add_imagetool_sources_to_figure(
            figure_uid, (second_uid, figure_uid), show=False
        )
        assert tuple(figure_tool.source_data()) == source_names
        xr.testing.assert_identical(
            figure_tool.source_data()["second"], second.rename("second")
        )
        assert not figure_tool.source_panel.source_status_label.isHidden()


def test_manager_figure_remove_unused_source_persists_workspace(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        _resolved_targets, sources, source_data = (
            manager._figure_workflows._figure_sources_from_targets((1,))
        )
        [source] = sources
        manager._figure_workflows._add_sources_to_figure(
            figure_uid, sources, source_data, show=False
        )
        assert source.name in figure_tool.source_data()
        manager._workspace_controller._mark_workspace_clean()

        figure_tool.source_panel.set_selected_names(
            (source.name,), current_name=source.name
        )
        figure_tool._refresh_source_controls()
        assert figure_tool.source_panel.remove_selected_source_button.isEnabled()
        with qtbot.wait_signal(figure_tool.sigDataChanged, timeout=5000):
            figure_tool.source_panel.remove_selected_source_button.click()

        assert source.name not in figure_tool.source_data()
        assert source.name not in {
            source.name for source in figure_tool.source_states()
        }
        snapshot = manager._workspace_controller._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        workspace_path = tmp_path / "remove-unused-figure-source.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded_node = manager._child_node(figure_uid)
        assert loaded_node.pending_workspace_tool_payload is not None
        loaded_tool = _materialized_figure_tool(manager, figure_uid)
        assert source.name not in loaded_tool.source_data()
        assert source.name not in {
            source.name for source in loaded_tool.source_states()
        }
