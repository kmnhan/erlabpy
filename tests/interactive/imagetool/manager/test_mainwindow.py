import concurrent.futures
import json
import logging
import pathlib
import types
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.dialogs as imagetool_dialogs
import erlab.interactive.imagetool.manager._acquisition_context as acquisition_context
import erlab.interactive.imagetool.manager._details_panel as manager_details_panel
import erlab.interactive.imagetool.manager._dialogs as manager_dialogs
import erlab.interactive.imagetool.manager._io as manager_io
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._metadata_editor as metadata_editor
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
import erlab.interactive.utils
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureComposerTool,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
    _seeding,
)
from erlab.interactive._figurecomposer._exceptions import (
    FigureComposerPlotSlicesSelectionError,
)
from erlab.interactive._figurecomposer._model._document import FigureSourceAddResult
from erlab.interactive._widgets import _CenteredIconToolButton
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool
from erlab.interactive.imagetool import _kspace_conversion, itool
from erlab.interactive.imagetool._load_source import _LoadSourceDetails
from erlab.interactive.imagetool._provenance._execution import replay_file_provenance
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    full_data,
    iter_operation_refs,
    public_data,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    AssignScalarCoordOperation,
    ImageDerivativeOperation,
    ImageToolSelectionSourceBinding,
    IselOperation,
    NormalizeOperation,
    RenameOperation,
    RestoreNonuniformDimsOperation,
    ScriptCodeOperation,
    TransposeOperation,
)
from erlab.interactive.imagetool.manager import fetch, replace_data
from erlab.interactive.imagetool.manager._acquisition_context import (
    AcquisitionContextDialog,
    AcquisitionContextField,
    AcquisitionContextState,
    ContextIngressSummary,
    _ContextFieldDialog,
    _ContextSourcePickerDialog,
)
from erlab.interactive.imagetool.manager._details_panel import _DetailsPanelController
from erlab.interactive.imagetool.manager._dialogs import (
    _batch_operation_dialog_classes,
    _BatchOperationDialog,
    _ConcatDialog,
    _RenameDialog,
)
from erlab.interactive.imagetool.manager._figurecomposer import _dialogs
from erlab.interactive.imagetool.manager._metadata_editor import (
    MetadataCellEdit,
    MetadataEditorDialog,
    MetadataField,
    _MetadataTableModel,
)
from erlab.interactive.imagetool.manager._modelview import (
    _TOOL_TYPE_ROLE,
    _ImageToolWrapperItemDelegate,
)
from erlab.interactive.imagetool.manager._widgets import (
    _LoadSourceDetailsDialog,
    _WorkspacePropertiesDialog,
    _WorkspacePropertiesState,
)

from .helpers import (
    _exec_generated_code,
    activate_widget_shortcut,
    assert_nonempty_tooltip,
    bring_manager_to_top,
    child_status_badge,
    click_child_status_badge,
    click_tree_view_pos,
    configure_goldtool_child,
    copy_full_code_for_uid,
    metadata_derivation_texts,
    select_child_tool,
    select_tools,
    trigger_menu_action,
)

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool.manager._modelview import (
        _ImageToolWrapperItemModel,
    )

logger = logging.getLogger(__name__)


def test_register_linked_nodes_invalidates_workspace_link_color_cache() -> None:
    registered: list[tuple[str, object]] = []
    graph = types.SimpleNamespace(
        register_root=lambda node: registered.append(("root", node)),
        register_child=lambda node: registered.append(("child", node)),
        register_figure=lambda node: registered.append(("figure", node)),
    )
    manager = types.SimpleNamespace(
        _tool_graph=graph,
        _workspace_link_color_cache_dirty=False,
    )
    manager._invalidate_workspace_link_color_cache = types.MethodType(
        manager_mainwindow.ImageToolManager._invalidate_workspace_link_color_cache,
        manager,
    )
    nodes = {
        "root": types.SimpleNamespace(workspace_link_key="root-link"),
        "child": types.SimpleNamespace(
            workspace_link_key="child-link", tool_window=None
        ),
        "figure": types.SimpleNamespace(
            workspace_link_key="figure-link", tool_window=None
        ),
    }

    for kind, method_name in (
        ("root", "_register_root_wrapper"),
        ("child", "_register_child_node"),
        ("figure", "_register_figure_node"),
    ):
        manager._workspace_link_color_cache_dirty = False
        getattr(manager_mainwindow.ImageToolManager, method_name)(manager, nodes[kind])
        assert manager._workspace_link_color_cache_dirty

    assert registered == [(kind, nodes[kind]) for kind in nodes]


def test_color_for_linker_falls_back_without_structural_link_key() -> None:
    child = object()
    linker = types.SimpleNamespace(children=(child,))
    manager = types.SimpleNamespace(
        node_from_slicer_area=lambda slicer_area: types.SimpleNamespace(
            workspace_link_key=None
        ),
        _link_registry=types.SimpleNamespace(index=lambda candidate: 1),
    )

    assert (
        manager_mainwindow.ImageToolManager.color_for_linker(manager, linker).getRgb()[
            :3
        ]
        == manager_mainwindow._LINKER_COLORS[1]
    )


def _record_reload_unavailable_dialog(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reasons: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "_show_reload_unavailable_dialog",
        lambda _parent, reason: reasons.append(reason),
    )
    return reasons


def _batch_data(
    name: str,
    *,
    dims: tuple[str, str] = ("x", "y"),
    offset: float = 0.0,
    first_coord: list[float] | None = None,
) -> xr.DataArray:
    coords = {
        dims[0]: np.arange(3, dtype=float) if first_coord is None else first_coord,
        dims[1]: np.arange(4, dtype=float),
    }
    return xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4) + offset,
        dims=dims,
        coords=coords,
        name=name,
    )


def _batch_volume(
    name: str,
    *,
    dims: tuple[str, str, str] = ("x", "y", "z"),
    offset: float = 0.0,
    first_coord: list[float] | None = None,
) -> xr.DataArray:
    coords = {
        dims[0]: np.arange(3, dtype=float) if first_coord is None else first_coord,
        dims[1]: np.arange(3, dtype=float),
        dims[2]: np.arange(4, dtype=float),
    }
    return xr.DataArray(
        np.arange(36, dtype=float).reshape(3, 3, 4) + offset,
        dims=dims,
        coords=coords,
        name=name,
    )


def _add_batch_tools(
    qtbot,
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    *data_arrays: xr.DataArray,
) -> None:
    initial_count = manager.ntools
    for data in data_arrays:
        itool(data, manager=True)
    qtbot.wait_until(
        lambda: manager.ntools == initial_count + len(data_arrays),
        timeout=5000,
    )


def _selection_shortcut_sequences(
    widget: QtWidgets.QWidget,
) -> set[str]:
    return {
        shortcut.key().toString(QtGui.QKeySequence.SequenceFormat.PortableText)
        for shortcut in widget.findChildren(QtWidgets.QShortcut)
        if shortcut.parent() is widget
    }


def test_managed_window_actions_reveal_tree_and_figure_rows(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    standalone_imagetool = typing.cast(
        "erlab.interactive.imagetool.ImageTool",
        itool(test_data, manager=False, execute=False),
    )
    qtbot.addWidget(standalone_imagetool)
    standalone_imagetool_menu = typing.cast(
        "erlab.interactive.imagetool._mainwindow.ItoolMenuBar",
        standalone_imagetool.menuBar(),
    )
    assert (
        not standalone_imagetool_menu.menu_dict["windowMenu"].menuAction().isVisible()
    )
    standalone_imagetool._reveal_in_manager()

    standalone_tool = erlab.interactive.utils.ToolWindow()
    qtbot.addWidget(standalone_tool)
    assert not standalone_tool._tool_window_menu.menuAction().isVisible()
    standalone_tool._reveal_in_manager()

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, test_data, test_data + 1)
        root_tool = manager.get_imagetool(0)
        second_uid = manager._tool_graph.root_wrappers[1].uid

        child_tool = erlab.interactive.utils.ToolWindow()
        child_uid = manager.add_childtool(child_tool, 0, show=False)
        figure_tool = FigureComposerTool(test_data)
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        root_menu = typing.cast(
            "erlab.interactive.imagetool._mainwindow.ItoolMenuBar",
            root_tool.menuBar(),
        ).menu_dict["windowMenu"]
        assert root_menu.menuAction().isVisible()
        assert root_tool.reveal_in_manager_act in root_menu.actions()
        assert child_tool._tool_window_menu.menuAction().isVisible()
        assert figure_tool._tool_window_menu.menuAction().isVisible()

        manager.tree_view.clearSelection()
        root_tool.reveal_in_manager_act.trigger()
        assert manager.tree_view.selected_imagetool_indices == [0]
        assert manager.left_tabs.currentWidget() is manager.tree_view

        child_index = manager.tree_view._model._row_index(child_uid)
        manager.tree_view.collapse(child_index.parent())
        child_tool.reveal_in_manager_action.trigger()
        assert manager.tree_view.selected_childtool_uids == [child_uid]
        assert manager.tree_view.isExpanded(child_index.parent())

        assert manager.reveal_nodes((child_uid, second_uid, child_uid, "missing"))
        assert manager.tree_view.selected_imagetool_indices == [1]
        assert manager.tree_view.selected_childtool_uids == [child_uid]
        assert not manager.reveal_nodes(("missing",))

        figure_tool.reveal_in_manager_action.trigger()
        figure_pane = manager._figure_collection.pane
        assert figure_pane is not None
        assert manager.left_tabs.currentWidget() is figure_pane
        selected_figure_uids = {
            manager._figure_collection.uid_from_item(item)
            for item in figure_pane.list_widget.selectedItems()
        }
        assert selected_figure_uids == {figure_uid}
        assert not manager.tree_view.selectedIndexes()

        child_node = manager._child_node(child_uid)
        manager_ref = child_node._manager
        child_node._manager = lambda: None
        assert not child_node.reveal_in_manager()
        child_node._manager = manager_ref
        manager._remove_childtool(child_uid)
        assert not child_node.reveal_in_manager()
        assert not child_tool.reveal_in_manager_action.isVisible()


def test_reveal_nodes_restores_minimized_manager(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, test_data)
        calls: list[str] = []
        monkeypatch.setattr(manager, "isMinimized", lambda: True)
        monkeypatch.setattr(manager, "showNormal", lambda: calls.append("normal"))
        monkeypatch.setattr(manager, "raise_", lambda: calls.append("raise"))
        monkeypatch.setattr(manager, "activateWindow", lambda: calls.append("activate"))

        assert manager.reveal_nodes((manager._tool_graph.root_wrappers[0].uid,))

        assert calls == ["normal", "raise", "activate"]

        calls.clear()
        monkeypatch.setattr(manager, "isMinimized", lambda: False)
        monkeypatch.setattr(manager, "isVisible", lambda: False)
        monkeypatch.setattr(manager, "show", lambda: calls.append("show"))

        assert manager.reveal_nodes((manager._tool_graph.root_wrappers[0].uid,))

        assert calls == ["show", "raise", "activate"]


def test_manager_open_settings_reuses_live_dialog(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        manager.open_settings()

        dialog = manager._additional_windows.get("settings")
        if dialog is None:
            raise AssertionError("Settings dialog was not registered")
        assert isinstance(dialog, erlab.interactive._options.OptionDialog)
        assert dialog.isVisible()

        manager.open_settings()

        assert manager._additional_windows.get("settings") is dialog

        dialog.deleteLater()
        qtbot.wait_until(lambda: "settings" not in manager._additional_windows)


def test_manager_layout_events_wait_for_tracking_enabled(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        workspace_controller = manager._workspace_controller
        manager._manager_layout_tracking_enabled = False
        del manager._workspace_controller
        try:
            manager.event(QtCore.QEvent(QtCore.QEvent.Type.WindowStateChange))
        finally:
            manager._workspace_controller = workspace_controller
            manager._manager_layout_tracking_enabled = True

        dirty_calls = 0

        def _record_workspace_layout_dirty() -> None:
            nonlocal dirty_calls
            dirty_calls += 1

        monkeypatch.setattr(
            manager,
            "_mark_workspace_layout_dirty",
            _record_workspace_layout_dirty,
        )
        manager._manager_layout_tracking_enabled = False
        manager.event(QtCore.QEvent(QtCore.QEvent.Type.WindowStateChange))
        assert dirty_calls == 0

        manager._manager_layout_tracking_enabled = True
        manager.event(QtCore.QEvent(QtCore.QEvent.Type.WindowStateChange))
        assert dirty_calls == 1


@pytest.mark.parametrize(
    ("platform", "rename_shortcut", "show_shortcut", "expected_shortcuts"),
    [
        (
            "darwin",
            "Return",
            "Ctrl+Down",
            {"Return", "Enter", "Ctrl+Down"},
        ),
        (
            "linux",
            "F2",
            "Return",
            {"F2", "Return", "Enter"},
        ),
    ],
)
def test_manager_tree_view_selection_shortcuts_are_platform_native(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    test_data: xr.DataArray,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    platform: str,
    rename_shortcut: str,
    show_shortcut: str,
    expected_shortcuts: set[str],
) -> None:
    monkeypatch.setattr(manager_mainwindow.sys, "platform", platform)

    with manager_context() as manager:
        manager.show()
        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        select_tools(manager, [0])
        manager._update_actions()

        assert manager.show_action.shortcut().isEmpty()
        assert _selection_shortcut_sequences(manager.tree_view) == expected_shortcuts

        tool = manager.get_imagetool(0)
        tool.hide()

        activate_widget_shortcut(manager.tree_view, rename_shortcut)
        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText(f"{platform}_shortcut_name")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: (
                manager._tool_graph.root_wrappers[0].name == f"{platform}_shortcut_name"
            ),
            timeout=5000,
        )
        assert not tool.isVisible()

        activate_widget_shortcut(manager.tree_view, show_shortcut)
        qtbot.wait_until(tool.isVisible, timeout=5000)


def _block_message_dialog(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    calls: list[tuple[tuple, dict]] = []

    def _critical(*args, **kwargs) -> int:
        calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_critical),
    )
    return calls


def _select_batch_operation(
    dialog: _BatchOperationDialog,
    dialog_cls: type[imagetool_dialogs._DataManipulationDialog],
) -> None:
    for top_index in range(dialog._operation_tree.topLevelItemCount()):
        top_item = dialog._operation_tree.topLevelItem(top_index)
        for child_index in range(top_item.childCount()):
            child_item = top_item.child(child_index)
            if child_item.data(0, QtCore.Qt.ItemDataRole.UserRole) is dialog_cls:
                dialog._operation_tree.setCurrentItem(child_item)
                return
    raise AssertionError(f"{dialog_cls.__name__} was not rendered")


class _BatchTransformStub:
    operation_types = (AssignAttrsOperation,)

    def __init__(
        self,
        operations: list[ToolProvenanceOperation] | None = None,
        *,
        source_error: Exception | None = None,
        public_source: bool = False,
    ) -> None:
        self._operations = operations
        self._source_error = source_error
        self._public_source = public_source
        self.keep_colors = False

    def source_operations(self) -> list[ToolProvenanceOperation]:
        if self._source_error is not None:
            raise self._source_error
        if self._operations is not None:
            return list(self._operations)
        return [AssignAttrsOperation(attrs={"batch": True})]

    def source_spec_for_data(
        self,
        data: xr.DataArray,
        new_name: str | None = None,
    ) -> ToolProvenanceSpec:
        del data, new_name
        builder = public_data if self._public_source else full_data
        return builder(*self.source_operations())

    def _detached_provenance_spec(
        self,
        parent_provenance: ToolProvenanceSpec | None,
        source_spec: ToolProvenanceSpec,
        new_name: str,
    ) -> ToolProvenanceSpec:
        del parent_provenance, new_name
        return source_spec

    def _compose_transform_provenance(
        self,
        base_spec: ToolProvenanceSpec | None,
        source_spec: ToolProvenanceSpec,
        new_name: str,
    ) -> ToolProvenanceSpec:
        del base_spec, new_name
        return source_spec

    def _itool_kwargs(
        self,
        processed: xr.DataArray,
        slicer_area=None,
    ) -> dict[str, typing.Any]:
        del slicer_area
        return {"data": processed, "execute": False}


class _BatchFilterStub:
    operation_types = (NormalizeOperation,)

    def __init__(
        self,
        operation: ToolProvenanceOperation | None,
    ) -> None:
        self._operation = operation

    def filter_operation(self) -> ToolProvenanceOperation | None:
        return self._operation


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 2.0],
        (1.0, 2.0),
        {"start": 1.0},
        {1.0, 2.0},
        frozenset({1.0, 2.0}),
        np.array([1.0, 2.0]),
    ],
)
def test_acquisition_context_coordinates_are_scalar_only(value) -> None:
    with pytest.raises(ValueError, match="Coordinates require a scalar value"):
        AcquisitionContextField.from_value(kind="coordinate", name="angle", value=value)
    scalar_field = AcquisitionContextField.from_value(
        kind="coordinate", name="angle", value=1.0
    )
    with pytest.raises(ValueError, match="Coordinates require a scalar value"):
        scalar_field.with_value(value)


def test_acquisition_context_values_are_serialization_stable() -> None:
    with pytest.raises(ValueError, match="stable serializable representation"):
        AcquisitionContextField.from_value(
            kind="attribute", name="labels", value={"sample", "reference"}
        )
    field = AcquisitionContextField.from_value(
        kind="attribute", name="range", value=(1.0, 2.0)
    )
    payload = AcquisitionContextState(fields=(field,)).model_dump(mode="json")
    restored = AcquisitionContextState.model_validate(payload)
    assert restored.fields[0].decoded_value == (1.0, 2.0)
    assert not AcquisitionContextState(enabled=True).enabled
    with pytest.raises(ValueError, match="unique by kind"):
        AcquisitionContextState(fields=(field, field))


def test_acquisition_context_field_dialog_validates_and_restores(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _ContextFieldDialog(parent)

    with pytest.raises(RuntimeError, match="was not accepted"):
        _ = dialog.field

    messages = _block_message_dialog(monkeypatch)
    dialog.accept()
    assert messages
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted

    dialog.kind_combo.setCurrentIndex(
        dialog.kind_combo.findData("attribute", QtCore.Qt.ItemDataRole.UserRole)
    )
    dialog.name_edit.setText("run")
    dialog.type_combo.setCurrentText("Int")
    dialog.value_edit.setText("4")
    dialog.existing_combo.setCurrentIndex(
        dialog.existing_combo.findData(True, QtCore.Qt.ItemDataRole.UserRole)
    )
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert dialog.field == AcquisitionContextField.from_value(
        kind="attribute", name="run", value=4, replace_existing=True
    )

    restored = _ContextFieldDialog(parent, field=dialog.field)
    assert restored.kind_combo.currentData() == "attribute"
    assert restored.name_edit.text() == "run"
    assert restored.type_combo.currentText() == "Int"
    assert restored.value_edit.text() == "4"
    assert restored.existing_combo.currentData() is True


def test_acquisition_context_dialog_commands_are_atomic(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    sample = AcquisitionContextField.from_value(
        kind="attribute", name="sample", value="A"
    )
    run = AcquisitionContextField.from_value(kind="attribute", name="run", value=1)
    changed_sample = sample.with_value("B")

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, _batch_data("scan"))
        manager._acquisition_context.set_state(
            AcquisitionContextState(fields=(sample, run)), mark_dirty=False
        )
        dialog = AcquisitionContextDialog(manager, manager._acquisition_context)

        select_tools(manager, [])
        assert dialog._selected_data() is None
        dialog._select_row(-1)
        assert dialog._current_row() == -1
        dialog._remove_field()
        dialog._edit_field()

        dialog._merge_fields((changed_sample,))
        assert dialog._fields == [changed_sample, run]
        dialog._rows_reordered(["invalid"], None, None)
        assert dialog._fields == [changed_sample, run]
        dialog._rows_reordered((dialog._field_row_id(run),), None, None)
        assert dialog.table.topLevelItemCount() == 2
        dialog._rows_reordered(
            (dialog._field_row_id(run), dialog._field_row_id(changed_sample)),
            None,
            None,
        )
        assert dialog._fields == [run, changed_sample]

        dialog._select_row(0)
        dialog._remove_field()
        assert dialog._fields == [changed_sample]

        select_tools(manager, [0])

        class _AcceptedSourcePicker:
            selected_fields = (run,)

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

        monkeypatch.setattr(
            acquisition_context, "_ContextSourcePickerDialog", _AcceptedSourcePicker
        )
        dialog._from_selected()
        assert dialog._fields == [changed_sample, run]
        monkeypatch.setattr(
            _AcceptedSourcePicker,
            "exec",
            lambda _self: QtWidgets.QDialog.DialogCode.Rejected,
        )
        dialog._from_selected()
        assert dialog._fields == [changed_sample, run]

        temperature = AcquisitionContextField.from_value(
            kind="coordinate", name="temperature", value=20.0
        )

        class _AcceptedFieldDialog:
            next_field = temperature
            result = QtWidgets.QDialog.DialogCode.Accepted

            def __init__(self, *_args, **_kwargs) -> None:
                self.field = self.next_field

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return self.result

        monkeypatch.setattr(
            acquisition_context, "_ContextFieldDialog", _AcceptedFieldDialog
        )
        dialog._add_field()
        assert dialog._fields[-1] == temperature
        _AcceptedFieldDialog.result = QtWidgets.QDialog.DialogCode.Rejected
        dialog._add_field()
        assert dialog._fields[-1] == temperature
        dialog._select_row(0)
        dialog._edit_field()
        assert dialog._fields[0] == changed_sample
        _AcceptedFieldDialog.result = QtWidgets.QDialog.DialogCode.Accepted

        warnings: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            staticmethod(lambda *args, **_kwargs: warnings.append(args)),
        )
        dialog._select_row(0)
        _AcceptedFieldDialog.next_field = run
        dialog._edit_field()
        assert warnings
        assert dialog._fields[0] == changed_sample

        replacement = changed_sample.with_value("C")
        _AcceptedFieldDialog.next_field = replacement
        dialog._edit_field()
        assert dialog._fields[0] == replacement

        messages = _block_message_dialog(monkeypatch)

        def _fail_set_state(*_args, **_kwargs):
            raise ValueError("invalid state")

        monkeypatch.setattr(dialog._controller, "set_state", _fail_set_state)
        dialog._save()
        assert messages
        assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            staticmethod(
                lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Cancel
            ),
        )
        dialog._clear()
        assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted

        saved: list[AcquisitionContextState] = []
        monkeypatch.setattr(
            dialog._controller,
            "set_state",
            lambda state, **_kwargs: saved.append(state),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            staticmethod(
                lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes
            ),
        )
        dialog._clear()
        assert saved == [AcquisitionContextState()]
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted


def test_acquisition_context_source_picker_rejects_empty_selection(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _batch_data("scan").assign_coords(
        temperature=20.0,
        unsupported=xr.Variable((), object()),
    )
    data.attrs["unsupported"] = {1, 2}
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _ContextSourcePickerDialog(parent, data)
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *args, **_kwargs: warnings.append(args)),
    )

    dialog.accept()

    assert warnings
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    root = dialog.tree.invisibleRootItem()
    assert root is not None
    assert all(
        child.text(0) != "unsupported"
        for group_index in range(root.childCount())
        if (group := root.child(group_index)) is not None
        for row in range(group.childCount())
        if (child := group.child(row)) is not None
    )


def test_acquisition_context_controller_recovers_from_invalid_state(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        controller = manager._acquisition_context
        warnings: list[str] = []
        monkeypatch.setattr(
            erlab.utils.misc,
            "emit_user_level_warning",
            lambda message, *_args, **_kwargs: warnings.append(message),
        )
        controller.restore_state_payload({"fields": [{"kind": "invalid"}]})
        assert warnings
        assert controller.state == AcquisitionContextState()

        manager._workspace_state.acquisition_context = {"fields": [{"kind": "invalid"}]}
        assert controller.state == AcquisitionContextState()
        manager._workspace_state.acquisition_context = "invalid"  # type: ignore[assignment]
        replacement_controller = type(controller)(manager)
        assert manager._workspace_state.acquisition_context == {}
        replacement_controller.deleteLater()

        button = controller._status_button
        controller._status_button = None
        controller.refresh_ui()
        controller._status_button = button

        shown: list[object] = []

        class _Editor:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def exec(self) -> None:
                shown.append(self)

        monkeypatch.setattr(acquisition_context, "AcquisitionContextDialog", _Editor)
        controller.show_editor()
        assert shown

        class _InvalidField:
            name = "invalid"
            kind = "attribute"
            replace_existing = False
            decoded_value = "value"

            def operation(self):
                raise RuntimeError("cannot create operation")

        resolution = controller.resolve(
            _batch_data("scan"), typing.cast("typing.Any", (_InvalidField(),))
        )
        assert resolution.operations == ()
        assert resolution.errors == ("invalid: cannot create operation",)


def test_acquisition_context_field_actions_use_toolbar_and_add_menu(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    field = AcquisitionContextField.from_value(
        kind="attribute", name="sample", value="reference"
    )

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, _batch_data("scan"))
        select_tools(manager, [0])
        manager._acquisition_context.set_state(
            AcquisitionContextState(fields=(field,)), mark_dirty=False
        )
        dialog = AcquisitionContextDialog(manager, manager._acquisition_context)
        qtbot.addWidget(dialog)

        assert isinstance(dialog.add_button, QtWidgets.QToolButton)
        assert isinstance(dialog.edit_button, QtWidgets.QToolButton)
        assert isinstance(dialog.remove_button, QtWidgets.QToolButton)
        for button in (
            dialog.add_button,
            dialog.edit_button,
            dialog.remove_button,
        ):
            assert button.toolButtonStyle() == (
                QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
            )
            assert button.icon().isNull()
        assert dialog.add_button.property("uses_inline_menu_arrow") is True
        assert dialog.add_menu.actions() == [
            dialog.add_field_action,
            dialog.from_selected_action,
        ]
        assert dialog.from_selected_action.isEnabled()
        assert not dialog.edit_button.isEnabled()
        assert not dialog.remove_button.isEnabled()

        root_layout = dialog.layout()
        assert root_layout is not None
        assert root_layout.indexOf(dialog.table) == 2
        toolbar_layout = root_layout.itemAt(1).layout()
        assert toolbar_layout is not None
        assert toolbar_layout.indexOf(dialog.add_button) == 0
        assert toolbar_layout.indexOf(dialog.edit_button) == 1
        assert toolbar_layout.indexOf(dialog.remove_button) == 2

        dialog.show()
        qtbot.mouseClick(dialog.add_button, QtCore.Qt.MouseButton.LeftButton)
        qtbot.wait_until(dialog.add_menu.isVisible)
        dialog.add_menu.hide()

        dialog.table.setCurrentItem(dialog.table.topLevelItem(0))
        assert dialog.edit_button.isEnabled()
        assert dialog.remove_button.isEnabled()


def test_acquisition_context_drag_order_controls_provenance_order(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    sample = AcquisitionContextField.from_value(
        kind="attribute", name="sample", value="reference"
    )
    photon_energy = AcquisitionContextField.from_value(
        kind="coordinate", name="photon_energy", value=21.2
    )
    run = AcquisitionContextField.from_value(kind="attribute", name="run", value=4)
    initial_state = AcquisitionContextState(
        enabled=True,
        fields=(sample, photon_energy, run),
    )
    source = _batch_data("scan")

    with manager_context() as manager:
        manager._acquisition_context.set_state(initial_state, mark_dirty=False)
        initial = manager._acquisition_context.resolve(source)
        assert [type(operation) for operation in initial.operations] == [
            AssignAttrsOperation,
            AssignScalarCoordOperation,
            AssignAttrsOperation,
        ]

        dialog = AcquisitionContextDialog(manager, manager._acquisition_context)
        qtbot.addWidget(dialog)
        assert (
            dialog.table.dragDropMode()
            == QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        assert dialog.table.defaultDropAction() == QtCore.Qt.DropAction.MoveAction
        assert dialog.table.showDropIndicator()
        assert not dialog.table.dragDropOverwriteMode()
        assert all(
            bool(
                typing.cast(
                    "QtWidgets.QTreeWidgetItem", dialog.table.topLevelItem(row)
                ).flags()
                & QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            and not bool(
                typing.cast(
                    "QtWidgets.QTreeWidgetItem", dialog.table.topLevelItem(row)
                ).flags()
                & QtCore.Qt.ItemFlag.ItemIsDropEnabled
            )
            for row in range(dialog.table.topLevelItemCount())
        )

        moved = dialog.table.takeTopLevelItem(1)
        assert moved is not None
        dialog.table.insertTopLevelItem(0, moved)
        dialog.table.setCurrentItem(moved)
        dialog.table._queue_rows_reordered()
        qtbot.waitUntil(
            lambda: dialog._fields == [photon_energy, sample, run],
        )

        dialog._save()

        assert manager._acquisition_context.state.fields == (
            photon_energy,
            sample,
            run,
        )
        assert manager._workspace_state.context_modified
        processed, operations, resolution = (
            manager._acquisition_context.apply_to_file_data(source)
        )
        assert not resolution.errors
        assert [type(operation) for operation in operations] == [
            AssignScalarCoordOperation,
            AssignAttrsOperation,
        ]
        attrs_operation = typing.cast("AssignAttrsOperation", operations[1])
        assert tuple(attrs_operation.attrs) == ("sample", "run")
        assert processed.coords["photon_energy"].item() == 21.2
        assert processed.attrs == {"sample": "reference", "run": 4}


def test_saving_unchanged_default_acquisition_context_keeps_workspace_clean(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_controller._mark_workspace_clean()
        generation = manager._workspace_state.dirty_generation
        dialog = AcquisitionContextDialog(manager, manager._acquisition_context)
        qtbot.addWidget(dialog)

        dialog._save()

        assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
        assert not manager._workspace_state.context_modified
        assert manager._workspace_state.dirty_generation == generation
        assert not manager._workspace_state.is_modified(has_nodes=False)


def test_acquisition_context_is_hidden_until_active_and_enriches_file_data(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("scan").assign_coords(temperature=12.0)
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")
    fields = (
        AcquisitionContextField.from_value(
            kind="coordinate", name="temperature", value=20.0
        ),
        AcquisitionContextField.from_value(
            kind="coordinate",
            name="photon_energy",
            value=21.2,
        ),
        AcquisitionContextField.from_value(
            kind="attribute", name="sample", value="reference"
        ),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        assert manager.acquisition_context_status_button.isHidden()
        file_actions = manager.file_menu.actions()
        assert manager.acquisition_context_action in file_actions
        assert manager.acquisition_context_action not in manager.edit_menu.actions()
        assert manager.metadata_editor_action in manager.edit_menu.actions()
        assert manager.metadata_editor_action not in file_actions
        assert not manager.metadata_editor_action.isEnabled()
        assert file_actions.index(manager.acquisition_context_action) == (
            file_actions.index(manager.explorer_action) + 1
        )

        manager._workspace_controller._mark_workspace_clean()
        manager._acquisition_context.set_state(
            AcquisitionContextState(enabled=True, fields=fields)
        )
        assert manager._workspace_state.context_modified
        qtbot.wait_until(manager.acquisition_context_status_button.isVisible)

        assert manager._data_ingress.receive_data(
            [source],
            {
                "file_path": file_path,
                "load_func": (
                    xr.load_dataarray,
                    {"engine": "h5netcdf"},
                    FileDataSelection(kind="dataarray"),
                ),
            },
            show=False,
        ) == [True]
        actual = manager.get_imagetool(0).slicer_area._data
        assert actual.coords["temperature"].item() == 12.0
        assert actual.coords["photon_energy"].item() == 21.2
        assert actual.attrs["sample"] == "reference"
        assert "photon_energy" not in source.coords
        assert "sample" not in source.attrs
        select_tools(manager, [0])
        assert manager.metadata_editor_action.isEnabled()

        provenance = manager._tool_graph.root_wrappers[0].displayed_provenance_spec
        assert provenance is not None
        xr.testing.assert_identical(replay_file_provenance(provenance), actual)
        code = provenance.display_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"data": source.copy(deep=True)})
        xr.testing.assert_identical(namespace["derived"], actual)

        manager._acquisition_context.set_state(AcquisitionContextState())
        assert manager.acquisition_context_status_button.isHidden()

        replacement = _batch_data("replacement", offset=100.0)
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            replace_data(0, replacement)
        replaced = manager.get_imagetool(0).slicer_area.data
        assert replaced.coords["photon_energy"].item() == 21.2
        assert replaced.attrs["sample"] == "reference"


def test_failed_file_ingress_does_not_report_acquisition_context(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("scan")
    field = AcquisitionContextField.from_value(
        kind="attribute", name="sample", value="reference"
    )
    load_options = {
        "load_func": (
            xr.load_dataarray,
            {},
            FileDataSelection(kind="dataarray"),
        )
    }
    creation_errors: list[None] = []

    def fail_registration(*_args: object, **_kwargs: object) -> typing.NoReturn:
        raise RuntimeError("registration failed")

    with manager_context() as manager:
        manager._acquisition_context.set_state(
            AcquisitionContextState(enabled=True, fields=(field,)),
            mark_dirty=False,
        )
        monkeypatch.setattr(manager_io, "ImageTool", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(manager, "add_imagetool", fail_registration)
        monkeypatch.setattr(
            manager._data_ingress,
            "_error_creating_imagetool",
            lambda: creation_errors.append(None),
        )

        manager._status_bar.showMessage("unchanged")
        assert manager._data_ingress.receive_data(
            [source], load_options.copy(), show=False
        ) == [False]
        assert manager._status_bar.currentMessage() == "unchanged"

        summary = ContextIngressSummary()
        assert manager._data_ingress.receive_data(
            [source],
            load_options.copy(),
            show=False,
            _context_summary=summary,
        ) == [False]
        assert summary == ContextIngressSummary()
        assert len(creation_errors) == 2


def test_acquisition_context_collision_policy_and_incompatible_input_are_atomic(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("scan").assign_coords(temperature=12.0)
    keep = AcquisitionContextField.from_value(
        kind="coordinate", name="temperature", value=20.0
    )
    replace = keep.model_copy(update={"replace_existing": True})
    identical = keep.with_value(12.0)
    incompatible_fields = (
        AcquisitionContextField.from_value(
            kind="attribute", name="sample", value="reference"
        ),
        AcquisitionContextField.from_value(
            kind="coordinate",
            name="x",
            value=1.0,
            replace_existing=True,
        ),
    )

    with manager_context() as manager:
        identical_resolution = manager._acquisition_context.resolve(
            source, (identical,)
        )
        assert identical_resolution.identical == 1
        assert identical_resolution.operations == ()

        keep_resolution = manager._acquisition_context.resolve(source, (keep,))
        assert keep_resolution.kept == 1
        assert keep_resolution.operations == ()

        replace_resolution = manager._acquisition_context.resolve(source, (replace,))
        assert replace_resolution.replaced == 1
        replaced = full_data(*replace_resolution.operations).apply(source)
        assert replaced.coords["temperature"].item() == 20.0

        manager._acquisition_context.set_state(
            AcquisitionContextState(enabled=True, fields=incompatible_fields),
            mark_dirty=False,
        )
        processed, operations, resolution = (
            manager._acquisition_context.apply_to_file_data(source)
        )
        assert operations == ()
        assert resolution.errors
        assert resolution.added == 1
        summary = ContextIngressSummary()
        summary.add_resolution(resolution)
        assert summary.added == 0
        assert summary.replaced == 0
        assert summary.failed == 1
        xr.testing.assert_identical(processed, source)
        assert "sample" not in processed.attrs


def test_metadata_assignments_survive_live_replacement_and_update_provenance(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("scan").assign_coords(temperature=12.0)
    edits = (
        MetadataCellEdit(
            MetadataField(kind="coordinate", name="temperature"), value=20.0
        ),
        MetadataCellEdit(MetadataField(kind="coordinate", name="angle"), value=1.5),
        MetadataCellEdit(
            MetadataField(kind="attribute", name="sample"), value="reference"
        ),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        assert manager._data_ingress.receive_data([source], {}, show=False) == [True]
        assert manager._metadata_editor.apply_edits({0: edits})

        replacement = _batch_data("replacement", offset=100.0)
        replacement_payloads: list[xr.DataArray] = []
        manager.get_imagetool(0).slicer_area.sigSourceDataReplaced.connect(
            replacement_payloads.append
        )
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            replace_data(0, replacement)

        actual = manager.get_imagetool(0).slicer_area.data
        expected = replacement.assign_coords(temperature=20.0, angle=1.5)
        expected = expected.assign_attrs(sample="reference")
        xr.testing.assert_identical(actual, expected)
        assert replacement_payloads
        xr.testing.assert_identical(replacement_payloads[-1], expected)

        node = manager._tool_graph.root_wrappers[0]
        assert node.detached_live_parent_data is not None
        xr.testing.assert_identical(node.detached_live_parent_data, replacement)
        provenance = node.displayed_provenance_spec
        assert provenance is not None
        xr.testing.assert_identical(
            provenance.apply(node.detached_live_parent_data), actual
        )

        incompatible = xr.DataArray(
            np.arange(16.0).reshape(4, 4),
            dims=("angle", "y"),
            name="incompatible",
        )
        before = actual.copy(deep=True)
        with pytest.raises(ValueError, match="angle"):
            replace_data(0, incompatible)
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, before)
        xr.testing.assert_identical(node.detached_live_parent_data, replacement)


def test_metadata_edits_and_replacement_preserve_nonuniform_public_data(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data(
        "scan",
        dims=("sample_temp", "eV"),
        first_coord=[20.0, 22.0, 25.0],
    )
    field = MetadataField(kind="attribute", name="sample")

    with manager_context() as manager:
        tool = itool(source.copy(deep=True), manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        index = manager.add_imagetool(
            tool,
            show=False,
            provenance_spec=full_data(),
        )
        assert manager._metadata_editor.apply_edits(
            {index: (MetadataCellEdit(field, value="reference"),)}
        )

        expected = source.assign_attrs(sample="reference")
        xr.testing.assert_identical(tool.slicer_area._data, expected)
        assert tuple(tool.slicer_area.data.dims) == ("sample_temp_idx", "eV")
        xr.testing.assert_identical(tool.slicer_area.displayed_data, expected)

        replacement = _batch_data(
            "replacement",
            dims=("sample_temp", "eV"),
            offset=100.0,
            first_coord=[30.0, 32.0, 35.0],
        )
        replace_data(index, replacement)

        replaced_expected = replacement.assign_attrs(sample="reference")
        xr.testing.assert_identical(tool.slicer_area._data, replaced_expected)
        assert tuple(tool.slicer_area.data.dims) == ("sample_temp_idx", "eV")
        xr.testing.assert_identical(tool.slicer_area.displayed_data, replaced_expected)
        node = manager._tool_graph.root_wrappers[index]
        assert node.detached_live_parent_data is not None
        xr.testing.assert_identical(node.detached_live_parent_data, replacement)
        assert node.displayed_provenance_spec is not None
        xr.testing.assert_identical(
            node.displayed_provenance_spec.apply(node.detached_live_parent_data),
            replaced_expected,
        )


def test_multi_target_replacement_preflights_preserved_metadata_atomically(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source0 = _batch_data("scan0")
    source1 = _batch_data("scan1", offset=100.0)
    angle = MetadataField(kind="coordinate", name="angle")

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, source0, source1)
        assert manager._metadata_editor.apply_edits(
            {1: (MetadataCellEdit(angle, value=1.5),)}
        )
        before0 = manager.get_imagetool(0).slicer_area._data.copy(deep=True)
        before1 = manager.get_imagetool(1).slicer_area._data.copy(deep=True)
        replacements = [
            _batch_data("replacement0", offset=200.0),
            _batch_data(
                "replacement1",
                dims=("angle", "y"),
                offset=300.0,
            ),
        ]
        replacement_signals: list[None] = []
        manager._sigDataReplaced.connect(lambda: replacement_signals.append(None))

        with pytest.raises(ValueError, match="angle"):
            manager._data_replace(replacements, [0, 1])

        assert not replacement_signals
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area._data, before0)
        xr.testing.assert_identical(manager.get_imagetool(1).slicer_area._data, before1)

        valid_replacements = [
            replacements[0],
            _batch_data("replacement1", offset=300.0),
        ]
        manager._data_replace(valid_replacements, [0, 1])

        assert replacement_signals == [None]
        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area._data,
            valid_replacements[0],
        )
        xr.testing.assert_identical(
            manager.get_imagetool(1).slicer_area._data,
            valid_replacements[1].assign_coords(angle=1.5),
        )


def test_multi_target_replacement_adds_consecutive_new_indices(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    replacements = [
        _batch_data("replacement0"),
        _batch_data("replacement1", offset=100.0),
    ]

    with manager_context() as manager:
        first_new_idx = manager.next_idx
        replacement_signals: list[None] = []
        manager._sigDataReplaced.connect(lambda: replacement_signals.append(None))

        manager._data_replace(
            replacements,
            [first_new_idx, first_new_idx + 1],
        )

        assert replacement_signals == [None]
        assert manager.next_idx == first_new_idx + 2
        for offset, replacement in enumerate(replacements):
            xr.testing.assert_identical(
                manager.get_imagetool(first_new_idx + offset).slicer_area._data,
                replacement,
            )


@pytest.mark.parametrize(("fail_on", "expected_tools"), [(1, 0), (2, 1)])
def test_multi_target_replacement_stops_when_new_tool_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    fail_on: int,
    expected_tools: int,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    replacements = [
        _batch_data(f"replacement{index}", offset=100.0 * index) for index in range(3)
    ]

    with manager_context() as manager:
        original_receive_data = manager._data_ingress.receive_data
        receive_calls = 0

        def receive_data(data, kwargs):
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls == fail_on:
                return [False]
            return original_receive_data(data, kwargs, show=False)

        monkeypatch.setattr(manager._data_ingress, "receive_data", receive_data)
        replacement_signals: list[None] = []
        manager._sigDataReplaced.connect(lambda: replacement_signals.append(None))

        manager._data_replace(replacements, [0, 1, 2])

        assert receive_calls == fail_on
        assert manager.ntools == expected_tools
        assert manager.next_idx == expected_tools
        assert replacement_signals == ([None] if expected_tools else [])
        if expected_tools:
            xr.testing.assert_identical(
                manager.get_imagetool(0).slicer_area._data, replacements[0]
            )


def test_metadata_assignments_survive_watched_variable_updates(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("scan")
    edits = (
        MetadataCellEdit(
            MetadataField(kind="attribute", name="sample"), value="reference"
        ),
        MetadataCellEdit(MetadataField(kind="coordinate", name="angle"), value=1.5),
    )
    messages = _block_message_dialog(monkeypatch)

    with manager_context() as manager:
        manager._data_watched_update("scan", "watched-uid", source)
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, source)
        assert manager._metadata_editor.apply_edits({0: edits})

        replacement = _batch_data("scan", offset=100.0)
        manager._data_watched_update("scan", "watched-uid", replacement)

        actual = manager.get_imagetool(0).slicer_area.data
        expected = replacement.assign_coords(angle=1.5)
        expected = expected.assign_attrs(sample="reference")
        xr.testing.assert_identical(actual, expected)
        node = manager._tool_graph.root_wrappers[0]
        provenance = node.displayed_provenance_spec
        assert provenance is not None
        code = provenance.display_code()
        assert code is not None
        namespace = _exec_generated_code(code, {"scan": replacement})
        xr.testing.assert_identical(namespace["derived"], actual)

        incompatible = xr.DataArray(
            np.arange(16.0).reshape(4, 4), dims=("angle", "y"), name="scan"
        )
        manager._data_watched_update("scan", "watched-uid", incompatible)
        assert messages
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, expected)


def test_acquisition_context_does_not_apply_when_child_is_promoted(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    field = AcquisitionContextField.from_value(
        kind="attribute", name="sample", value="reference"
    )
    child_data = _batch_data("child", offset=100.0)

    with manager_context() as manager:
        manager._acquisition_context.set_state(
            AcquisitionContextState(enabled=True, fields=(field,)), mark_dirty=False
        )
        _add_batch_tools(qtbot, manager, _batch_data("root"))
        child_tool = itool(child_data, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child_tool, 0, show=False)

        promoted_index = manager.promote_child_imagetool(child_uid)

        xr.testing.assert_identical(
            manager.get_imagetool(promoted_index).slicer_area.data, child_data
        )


def test_live_replacement_preserves_explicit_assignment_when_source_matches(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    field = MetadataField(kind="coordinate", name="temperature")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        assert manager._data_ingress.receive_data(
            [_batch_data("scan")], {}, show=False
        ) == [True]
        assert manager._metadata_editor.apply_edits(
            {0: (MetadataCellEdit(field, value=20.0),)}
        )

        replacement = _batch_data("replacement", offset=100.0).assign_coords(
            temperature=12.0
        )
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            replace_data(0, replacement)

        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area.data,
            replacement.assign_coords(temperature=20.0),
        )
        node = manager._tool_graph.root_wrappers[0]
        assert node.displayed_provenance_spec is not None
        assert node.detached_live_parent_data is not None


def test_acquisition_context_can_copy_selected_coordinates_and_attributes(
    qtbot,
) -> None:
    data = _batch_data("scan").assign_coords(temperature=20.0)
    data = data.assign_coords(grid=(("x", "y"), np.zeros(data.shape)))
    data.attrs.update({"sample": "reference", "run": 4})
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    dialog = _ContextSourcePickerDialog(parent, data)

    selected_keys = {("coordinate", "temperature"), ("attribute", "sample")}
    root = dialog.tree.invisibleRootItem()
    assert root is not None
    available: dict[tuple[str, str], QtWidgets.QTreeWidgetItem] = {}
    for group_index in range(root.childCount()):
        group = root.child(group_index)
        assert group is not None
        for row in range(group.childCount()):
            item = group.child(row)
            assert item is not None
            field = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            assert isinstance(field, AcquisitionContextField)
            available[field.key] = item
    assert (
        not {("coordinate", "x"), ("coordinate", "y"), ("coordinate", "grid")}
        & available.keys()
    )
    for key in selected_keys:
        available[key].setCheckState(0, QtCore.Qt.CheckState.Checked)

    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert {field.key for field in dialog.selected_fields} == selected_keys


def test_metadata_editor_scalar_value_and_operation_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadArray:
        def __array__(self, *_args, **_kwargs):
            raise RuntimeError("not array-like")

    assert metadata_editor._values_equal(np.nan, np.nan)
    assert metadata_editor._values_equal(np.int64(1), 1)
    assert metadata_editor._values_equal(np.float64(1.0), 1.0)
    assert not metadata_editor._values_equal(1, 1.0)
    assert not metadata_editor._values_equal(1, True)
    assert not metadata_editor._values_equal([1], [1.0])
    assert not metadata_editor._values_equal({"value": 1}, {"other": 1})
    assert not metadata_editor._values_equal({"value": 1}, {"value": True})
    assert not metadata_editor._values_equal([1, 2], [1])
    assert not metadata_editor._values_equal(np.ones((1, 2)), np.ones((2, 1)))
    assert not metadata_editor._values_equal(np.array([1]), np.array(["1"]))
    assert not metadata_editor._values_equal(_BadArray(), 1)

    original_array_equal = np.array_equal
    calls = 0

    def _array_equal(left, right, *, equal_nan=False):
        nonlocal calls
        calls += 1
        if equal_nan:
            raise TypeError
        return original_array_equal(left, right)

    monkeypatch.setattr(metadata_editor.np, "array_equal", _array_equal)
    assert metadata_editor._values_equal(1, 1)
    assert calls == 2

    for value, expected in (
        (True, "Bool"),
        (np.int64(1), "Int"),
        (np.float64(1.0), "Float"),
        ("value", "String"),
        ([1, 2], "Python literal"),
    ):
        assert metadata_editor._value_type_name(value) == expected
    assert metadata_editor._parse_typed_value("String", " value ") == " value "
    assert metadata_editor._parse_typed_value("Int", " 4 ") == 4
    assert metadata_editor._parse_typed_value("Float", " 4.5 ") == 4.5
    assert metadata_editor._parse_typed_value("Bool", "yes") is True
    assert metadata_editor._parse_typed_value("Bool", "0") is False
    assert metadata_editor._parse_typed_value("Python literal", "[1, 2]") == [1, 2]
    assert metadata_editor._parse_editor_value(20, "20.5") == 20.5
    assert metadata_editor._parse_editor_value(20.0, "20") == 20.0
    assert metadata_editor._parse_editor_value(False, "0") is False
    assert metadata_editor._parse_editor_value(True, "1") is True
    assert metadata_editor._parse_editor_value("20", "20") == "20"
    assert metadata_editor._parse_editor_value("20", "21") == "21"
    assert metadata_editor._parse_editor_value("sample", "reference") == "reference"
    assert metadata_editor._parse_editor_value(
        metadata_editor._UNKNOWN_REFERENCE, "[1, 2]"
    ) == [1, 2]
    assert (
        metadata_editor._parse_editor_value(
            metadata_editor._UNKNOWN_REFERENCE, "unquoted text"
        )
        == "unquoted text"
    )
    with pytest.raises(ValueError, match="Boolean"):
        metadata_editor._parse_typed_value("Bool", "sometimes")
    with pytest.raises(ValueError, match="Unknown value type"):
        metadata_editor._parse_typed_value("unknown", "1")
    with pytest.raises(ValueError, match="Metadata field names cannot be empty"):
        MetadataField(kind="attribute", name=" ")

    coord = MetadataField(kind="coordinate", name="temperature")
    attr = MetadataField(kind="attribute", name="sample")
    assert isinstance(coord.operation(20), AssignScalarCoordOperation)
    assert isinstance(attr.operation("reference"), AssignAttrsOperation)
    data = _batch_data("scan").assign_coords(temperature=np.int64(20))
    assert metadata_editor._field_value(data, coord) == 20
    assert metadata_editor._field_value(data, attr) is metadata_editor._MISSING
    assert metadata_editor._operation_value(coord.operation(20), coord) == 20
    assert (
        metadata_editor._operation_value(attr.operation("reference"), coord)
        is metadata_editor._MISSING
    )
    assert metadata_editor._editable_attribute([1, 2])
    assert metadata_editor._editable_attribute({"range": (1, 2)})
    assert not metadata_editor._editable_attribute({1, 2})
    assert not metadata_editor._editable_attribute(frozenset({1, 2}))
    assert not metadata_editor._editable_attribute(np.array([1, 2]))
    assert not metadata_editor._editable_attribute(object())
    assert metadata_editor._primitive_value(np.float64(1.0))
    assert not metadata_editor._primitive_value([1.0])


def test_metadata_editor_table_model_tracks_origin_and_pending_state(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    temperature = MetadataField(kind="coordinate", name="temperature")
    angle = MetadataField(kind="coordinate", name="angle")
    sample = MetadataField(kind="attribute", name="sample")
    enabled = MetadataField(kind="attribute", name="enabled")
    acquired = MetadataField(kind="coordinate", name="acquired")
    source = _batch_data("scan").assign_coords(
        temperature=np.int64(20), acquired=np.datetime64("2024-01-01")
    )
    source.attrs["sample"] = "source"

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, source)
        assert manager._metadata_editor.apply_edits(
            {0: (MetadataCellEdit(angle, value=1.5),)}
        )
        model = _MetadataTableModel(
            manager._metadata_editor,
            (0,),
            (temperature, angle, sample, enabled, acquired),
            {
                temperature: 0,
                angle: 0.0,
                sample: "",
                enabled: False,
                acquired: np.datetime64("2024-01-01"),
            },
        )

        assert model.rowCount() == 1
        assert model.columnCount() == 6
        assert model.rowCount(model.index(0, 0)) == 0
        assert model.columnCount(model.index(0, 0)) == 0
        assert model.cell_at(model.index(0, 0)) is None
        assert model.field_at(-1) is None
        assert model.field_at(model.columnCount()) is None
        assert model.data(QtCore.QModelIndex()) is None
        foreign_model = QtGui.QStandardItemModel(1, 1)
        foreign_index = foreign_model.index(0, 0)
        assert model.cell_at(foreign_index) is None
        assert model.data(foreign_index) is None
        assert model.flags(foreign_index) == QtCore.Qt.ItemFlag.NoItemFlags
        assert not model.setData(foreign_index, "ignored")
        assert (
            model.headerData(
                -1,
                QtCore.Qt.Orientation.Horizontal,
                QtCore.Qt.ItemDataRole.DisplayRole,
            )
            is None
        )
        assert (
            model.headerData(
                model.columnCount(),
                QtCore.Qt.Orientation.Horizontal,
                QtCore.Qt.ItemDataRole.DisplayRole,
            )
            is None
        )
        assert isinstance(model.index(0, 0).data(), str)
        assert model.index(0, 0).data(_MetadataTableModel.TargetRole) == 0
        assert model.index(0, 1).data(_MetadataTableModel.OriginRole) == "source"
        assert model.index(0, 2).data(_MetadataTableModel.OriginRole) == "assigned"
        assert model.index(0, 3).data(_MetadataTableModel.OriginRole) == "source"
        assert model.index(0, 4).data(_MetadataTableModel.OriginRole) == "missing"
        assert model.index(0, 1).data(_MetadataTableModel.FieldRole) == temperature
        assert model.index(0, 1).data(_MetadataTableModel.TargetRole) == 0
        assert not model.index(0, 1).data(_MetadataTableModel.DirtyRole)
        assert not model.index(0, 1).data(_MetadataTableModel.RevertRole)
        assert isinstance(
            model.index(0, 2).data(QtCore.Qt.ItemDataRole.ToolTipRole), str
        )
        assert isinstance(
            model.index(0, 4).data(QtCore.Qt.ItemDataRole.ToolTipRole), str
        )
        assert isinstance(
            model.index(0, 1).data(QtCore.Qt.ItemDataRole.TextAlignmentRole), int
        )
        assert (
            model.headerData(
                1, QtCore.Qt.Orientation.Horizontal, _MetadataTableModel.FieldRole
            )
            == temperature
        )
        assert not (model.flags(model.index(0, 0)) & QtCore.Qt.ItemFlag.ItemIsEditable)
        assert model.flags(model.index(0, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable
        assert not model.setData(model.index(0, 0), "ignored")
        assert not model.setData(
            model.index(0, 1), "21", QtCore.Qt.ItemDataRole.DisplayRole
        )
        assert not model.setData(model.index(0, 1), "not-an-int")
        assert model.setData(model.index(0, 1), "20")
        assert not model.has_changes
        assert model.setData(model.index(0, 1), "20.0")
        assert model.edited_values[(0, temperature)] == 20.0
        assert isinstance(model.edited_values[(0, temperature)], float)
        model.undo_stack.undo()
        assert not model.has_changes
        assert model.setData(model.index(0, 1), "20.5")
        assert model.edited_values[(0, temperature)] == 20.5
        assert isinstance(model.edited_values[(0, temperature)], float)
        model.undo_stack.undo()
        assert not model.has_changes
        assert model.setData(model.index(0, 1), "21")
        assert model.has_changes

        acquired_index = model.index(0, 5)
        rejected: list[str] = []
        model.editRejected.connect(rejected.append)
        assert not (model.flags(acquired_index) & QtCore.Qt.ItemFlag.ItemIsEditable)
        assert not model.setData(acquired_index, "2024-01-02")
        assert rejected

        model.revert_indexes((model.index(0, 0), model.index(0, 3)))
        assert not model.index(0, 3).data(_MetadataTableModel.RevertRole)
        model.revert_indexes((model.index(0, 2),))
        assert model.index(0, 2).data(_MetadataTableModel.RevertRole)
        assert model.index(0, 2).data(_MetadataTableModel.DirtyRole)
        assert isinstance(
            model.index(0, 2).data(QtCore.Qt.ItemDataRole.ToolTipRole), str
        )
        model.undo_stack.undo()
        assert not model.index(0, 2).data(_MetadataTableModel.RevertRole)
        assert model.index(0, 1).data(_MetadataTableModel.DirtyRole)
        model.undo_stack.redo()
        assert model.index(0, 2).data(_MetadataTableModel.RevertRole)

        model.add_field(enabled, False)
        enabled_index = model.index(0, model.fields.index(enabled) + 1)
        assert model.setData(enabled_index, "yes")
        edits = model.edits_by_target()[0]
        assert {edit.field for edit in edits} == {temperature, angle, enabled}

        delegate = metadata_editor._MetadataCellDelegate(manager)
        pixmap = QtGui.QPixmap(180, 36)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 180, 36)
        painter = QtGui.QPainter(pixmap)
        delegate.paint(painter, option, model.index(0, 0))
        delegate.paint(painter, option, model.index(0, 1))
        delegate.paint(painter, option, model.index(0, 2))
        painter.end()

        model.set_fields((sample,), model.field_defaults)
        assert model.has_changes
        assert {edit.field for edit in model.edits_by_target()[0]} == {
            temperature,
            angle,
            enabled,
        }


def test_metadata_editor_supports_paste_and_per_row_values(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = _batch_data("scan0")
    data1 = _batch_data("scan1", offset=100.0)
    temperature = MetadataField(kind="coordinate", name="temperature")
    sample = MetadataField(kind="attribute", name="sample")

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, data0, data1)
        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0, 1))
        qtbot.addWidget(dialog)
        dialog.model.add_field(temperature, 20.0)
        dialog.model.add_field(sample, "unknown")
        dialog.table.setCurrentIndex(dialog.model.index(0, 1))
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText("15.0\tA\n25.0\tB")
        dialog.table._paste_clipboard()

        temperature_column = dialog.model.fields.index(temperature) + 1
        temperature_selection = QtCore.QItemSelection(
            dialog.model.index(0, temperature_column),
            dialog.model.index(1, temperature_column),
        )
        dialog.table.setCurrentIndex(dialog.model.index(0, temperature_column))
        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", dialog.table.selectionModel()
        )
        selection_model.select(
            temperature_selection,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )
        series_values = iter(((30.0, True), (5.0, True)))
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getDouble",
            staticmethod(lambda *_args, **_kwargs: next(series_values)),
        )
        dialog._fill_series()

        sample_column = dialog.model.fields.index(sample) + 1
        sample_selection = QtCore.QItemSelection(
            dialog.model.index(0, sample_column),
            dialog.model.index(1, sample_column),
        )
        dialog.table.setCurrentIndex(dialog.model.index(0, sample_column))
        selection_model.select(
            sample_selection,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )
        dialog._fill_down()

        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            dialog._apply()
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted

        for index, temperature, sample, source in (
            (0, 30.0, "A", data0),
            (1, 35.0, "A", data1),
        ):
            actual = manager.get_imagetool(index).slicer_area._data
            assert actual.coords["temperature"].item() == temperature
            assert actual.attrs["sample"] == sample
            provenance = manager._tool_graph.root_wrappers[
                index
            ].displayed_provenance_spec
            assert provenance is not None
            xr.testing.assert_identical(provenance.apply(source), actual)


def test_metadata_editor_fill_series_rejects_boolean_columns(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    enabled = MetadataField(kind="attribute", name="enabled")
    sources = tuple(
        _batch_data(f"scan{index}", offset=100.0 * index).assign_attrs(
            enabled=bool(index % 2)
        )
        for index in range(3)
    )

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, *sources)
        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0, 1, 2))
        qtbot.addWidget(dialog)
        column = dialog.model.fields.index(enabled) + 1
        selection = QtCore.QItemSelection(
            dialog.model.index(0, column), dialog.model.index(2, column)
        )
        dialog.table.setCurrentIndex(dialog.model.index(0, column))
        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", dialog.table.selectionModel()
        )
        selection_model.select(
            selection, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
        dialog._refresh_summary()
        assert not dialog.fill_series_button.isEnabled()
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getDouble",
            staticmethod(
                lambda *_args, **_kwargs: pytest.fail(
                    "Boolean columns must not open the numeric series dialog"
                )
            ),
        )

        dialog._fill_series()

        assert not dialog.model.has_changes


def test_metadata_editor_copies_selection_and_undoes_bulk_edits(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    temperature = MetadataField(kind="coordinate", name="temperature")
    sample = MetadataField(kind="attribute", name="sample")
    sources = (
        _batch_data("scan0").assign_coords(temperature=10.0).assign_attrs(sample="A"),
        _batch_data("scan1", offset=100.0)
        .assign_coords(temperature=20.0)
        .assign_attrs(sample="B"),
    )

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, *sources)
        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0, 1))
        qtbot.addWidget(dialog)
        temperature_column = dialog.model.fields.index(temperature) + 1
        sample_column = dialog.model.fields.index(sample) + 1
        header = dialog.table.horizontalHeader()
        header.moveSection(header.visualIndex(sample_column), 1)

        selection = QtCore.QItemSelection(
            dialog.model.index(0, temperature_column),
            dialog.model.index(1, sample_column),
        )
        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", dialog.table.selectionModel()
        )
        selection_model.select(
            selection, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
        dialog.table._copy_selection()
        clipboard = QtWidgets.QApplication.clipboard()
        assert clipboard.text() == "A\t10.0\nB\t20.0"

        selection_model.clearSelection()
        for index in (
            dialog.model.index(0, sample_column),
            dialog.model.index(1, temperature_column),
        ):
            selection_model.select(
                index, QtCore.QItemSelectionModel.SelectionFlag.Select
            )
        dialog.table._copy_selection()
        assert clipboard.text() == "A\t\n\t20.0"

        copy_combination = QtGui.QKeySequence.keyBindings(
            QtGui.QKeySequence.StandardKey.Copy
        )[0][0]
        copy_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            copy_combination.key(),
            copy_combination.keyboardModifiers(),
        )
        dialog.table.keyPressEvent(copy_event)
        assert copy_event.isAccepted()

        dialog.table.setCurrentIndex(dialog.model.index(0, sample_column))
        dialog.table.moveCursor(
            QtWidgets.QAbstractItemView.CursorAction.MoveRight,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        undo_shortcuts = {
            shortcut.toString(QtGui.QKeySequence.SequenceFormat.PortableText)
            for shortcut in dialog.undo_action.shortcuts()
        }
        redo_shortcuts = {
            shortcut.toString(QtGui.QKeySequence.SequenceFormat.PortableText)
            for shortcut in dialog.redo_action.shortcuts()
        }
        assert undo_shortcuts == {
            shortcut.toString(QtGui.QKeySequence.SequenceFormat.PortableText)
            for shortcut in QtGui.QKeySequence.keyBindings(
                QtGui.QKeySequence.StandardKey.Undo
            )
        }
        assert redo_shortcuts == {
            shortcut.toString(QtGui.QKeySequence.SequenceFormat.PortableText)
            for shortcut in QtGui.QKeySequence.keyBindings(
                QtGui.QKeySequence.StandardKey.Redo
            )
        }

        dialog.table.setCurrentIndex(dialog.model.index(0, sample_column))
        clipboard.setText("C\t30\nD\t40")
        paste_combination = QtGui.QKeySequence.keyBindings(
            QtGui.QKeySequence.StandardKey.Paste
        )[0][0]
        paste_event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            paste_combination.key(),
            paste_combination.keyboardModifiers(),
        )
        dialog.table.keyPressEvent(paste_event)
        assert paste_event.isAccepted()
        assert dialog.model.has_changes
        assert dialog.model.cell_at(dialog.model.index(0, sample_column)).value == "A"
        assert dialog.model.index(0, sample_column).data() == "C"
        assert dialog.model.index(1, temperature_column).data() == "40.0"

        dialog.undo_action.trigger()
        assert not dialog.model.has_changes
        assert dialog.model.index(0, sample_column).data() == "A"
        assert dialog.model.index(1, temperature_column).data() == "20.0"

        dialog.redo_action.trigger()
        assert dialog.model.has_changes
        assert dialog.model.index(0, sample_column).data() == "C"
        assert dialog.model.index(1, temperature_column).data() == "40.0"


def test_metadata_editor_bulk_edits_are_atomic_for_mixed_cell_types(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    mixed = MetadataField(kind="attribute", name="mixed")
    sources = (
        _batch_data("scan0").assign_attrs(mixed=2),
        _batch_data("scan1", offset=100.0).assign_attrs(mixed=1.0 + 2.0j),
    )

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, *sources)
        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0, 1))
        qtbot.addWidget(dialog)
        column = dialog.model.fields.index(mixed) + 1
        current = dialog.model.index(0, column)
        selection = QtCore.QItemSelection(current, dialog.model.index(1, column))
        dialog.table.setCurrentIndex(current)
        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", dialog.table.selectionModel()
        )
        selection_model.select(
            selection, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
        messages = _block_message_dialog(monkeypatch)

        dialog._fill_down()
        assert messages
        assert not dialog.model.has_changes

        messages.clear()
        series_values = iter(((2.0, True), (1.0, True)))
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getDouble",
            staticmethod(lambda *_args, **_kwargs: next(series_values)),
        )
        dialog._fill_series()
        assert messages
        assert not dialog.model.has_changes

        messages.clear()
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText("3\n2")
        dialog.table._paste_clipboard()
        assert messages
        assert not dialog.model.has_changes


def test_metadata_editor_field_chooser_and_frozen_data_column(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = _batch_data("scan0").assign_coords(temperature=20.0)
    data0.attrs.update({"sample": "A", "temperature": "measured"})
    data1 = _batch_data("scan1", offset=100.0).assign_attrs(sample="B")

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, data0, data1)
        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0, 1))
        dialog.show()
        qtbot.mouseClick(dialog.fields_button, QtCore.Qt.MouseButton.LeftButton)
        qtbot.wait_until(dialog.fields_menu.isVisible)
        dialog.fields_menu.hide()

        coord = MetadataField(kind="coordinate", name="temperature")
        attr = MetadataField(kind="attribute", name="temperature")
        sample = MetadataField(kind="attribute", name="sample")
        assert {coord, attr, sample} <= set(dialog.available_fields)
        assert dialog.fields_widget._coverage[coord] == 1
        assert dialog.fields_widget._coverage[attr] == 1
        assert dialog.fields_widget._coverage[sample] == 2
        assert dialog.table.frozen_view.model() is dialog.model
        assert not dialog.table.frozen_view.isColumnHidden(0)
        assert all(
            dialog.table.frozen_view.isColumnHidden(column)
            for column in range(1, dialog.model.columnCount())
        )
        assert dialog.table.alternatingRowColors()
        assert dialog.table.frozen_view.alternatingRowColors()
        qtbot.wait_until(
            lambda: (
                dialog.table.horizontalHeader().height()
                == dialog.table.frozen_view.horizontalHeader().height()
            )
        )
        assert (
            dialog.table.viewport().mapToGlobal(QtCore.QPoint()).y()
            == dialog.table.frozen_view.viewport().mapToGlobal(QtCore.QPoint()).y()
        )
        metadata_widths = {
            dialog.table.columnWidth(column)
            for column in range(1, dialog.model.columnCount())
        }
        assert len(metadata_widths) > 1

        chooser = dialog.fields_widget
        chooser.search_edit.setText("sample")
        root = chooser.tree.invisibleRootItem()
        assert root is not None
        chooser_items: dict[MetadataField, QtWidgets.QTreeWidgetItem] = {}
        for group_index in range(root.childCount()):
            group = root.child(group_index)
            assert group is not None
            for row in range(group.childCount()):
                item = group.child(row)
                assert item is not None
                field = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(field, MetadataField):
                    chooser_items[field] = item
        assert not chooser_items[sample].isHidden()
        assert chooser_items[coord].isHidden()
        chooser.search_edit.clear()

        visible_field_sets: list[tuple[MetadataField, ...]] = []
        chooser.visible_fields_changed.connect(visible_field_sets.append)
        chooser._preset("common")
        assert visible_field_sets[-1] == (sample,)
        assert dialog.model.fields == [sample]
        chooser._preset("incomplete")
        assert set(visible_field_sets[-1]) == {coord, attr}
        assert set(dialog.model.fields) == {coord, attr}
        clear_button = chooser.findChild(
            QtWidgets.QToolButton, "manager_metadata_editor_fields_clear_button"
        )
        assert clear_button is not None
        chooser._preset("clear")
        assert visible_field_sets[-1] == ()
        assert dialog.model.columnCount() == 1
        chooser._preset("all")
        assert set(dialog.model.fields) == {coord, attr, sample}

        messages = _block_message_dialog(monkeypatch)
        chooser.name_edit.clear()
        chooser._add_field()
        assert messages
        added: list[tuple[MetadataField, object]] = []
        chooser.field_added.connect(lambda field, value: added.append((field, value)))
        chooser.kind_combo.setCurrentIndex(1)
        chooser.name_edit.setText("run")
        chooser.type_combo.setCurrentText("Int")
        chooser.value_edit.setText("4")
        chooser._add_field()
        run = MetadataField(kind="attribute", name="run")
        assert added[-1] == (run, 4)
        assert run in dialog.model.fields
        chooser.add_field(run)
        chooser.set_field_visible(
            MetadataField(kind="attribute", name="not-listed"), True
        )
        sample_item = chooser._items[sample]
        chooser._item_changed(sample_item, 1)
        sample_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        assert sample not in dialog.model.fields
        sample_item.setCheckState(0, QtCore.Qt.CheckState.Checked)
        assert sample in dialog.model.fields

        dialog._set_field_visible(object(), True)
        dialog._set_visible_fields([coord])
        dialog._add_field(object(), 0)

        messages.clear()
        dialog._show_cell_edit_error("invalid value")
        assert messages

        dialog._set_field_visible(coord, False)
        assert coord not in dialog.model.fields
        assert coord not in manager._metadata_editor.layout_state.fields
        dialog._set_field_visible(coord, True)
        assert coord in dialog.model.fields
        assert manager._metadata_editor.layout_state.fields[-1] == coord
        dialog._set_field_visible(sample, True)

        index = dialog.model.index(0, dialog.model.fields.index(sample) + 1)
        assert dialog.model.setData(index, "changed")

        dialog.table.setColumnWidth(0, 210)
        dialog.table.setRowHeight(0, 38)
        assert dialog.table.frozen_view.columnWidth(0) == 210
        assert dialog.table.frozen_view.rowHeight(0) == 38
        manager._metadata_editor.set_column_width(None, 210)

        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", dialog.table.selectionModel()
        )
        selection_model.clearSelection()
        dialog.table.setCurrentIndex(index)
        dialog._fill_down()
        dialog._revert_assignment()

        header = typing.cast("QtWidgets.QHeaderView", dialog.table.horizontalHeader())
        sample_column = dialog.model.fields.index(sample) + 1
        sample_position = QtCore.QPoint(
            header.sectionViewportPosition(sample_column) + 1, 1
        )

        def _choose_hide_field() -> None:
            menu = QtWidgets.QApplication.activePopupWidget()
            if not isinstance(menu, QtWidgets.QMenu):
                pytest.fail("The metadata header menu did not open.")
            action = menu.actions()[0]
            qtbot.mouseClick(
                menu,
                QtCore.Qt.MouseButton.LeftButton,
                pos=menu.actionGeometry(action).center(),
            )

        QtCore.QTimer.singleShot(0, _choose_hide_field)
        dialog._show_header_menu(sample_position)
        assert sample not in dialog.model.fields
        dialog._set_field_visible(run, False)
        assert run not in dialog.model.fields

        dialog.table.setCurrentIndex(QtCore.QModelIndex())
        dialog._refresh_summary()
        assert not dialog.fill_down_button.isEnabled()
        assert not dialog.fill_series_button.isEnabled()
        dialog._fill_down()
        dialog._fill_series()
        dialog.table._paste_clipboard()
        dialog.reject()
        assert manager.get_imagetool(0).slicer_area.data.attrs["sample"] == "A"


def test_metadata_editor_shows_file_source_and_cell_origin(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")
    first_field = MetadataField(kind="attribute", name="sample")
    next_field = MetadataField(kind="coordinate", name="temperature")

    with manager_context() as manager:
        assert manager._data_ingress.receive_data(
            [source],
            {
                "file_path": file_path,
                "load_func": (
                    xr.load_dataarray,
                    {"engine": "h5netcdf"},
                    FileDataSelection(kind="dataarray"),
                ),
            },
            show=False,
        ) == [True]
        assert manager._metadata_editor.apply_edits(
            {0: (MetadataCellEdit(first_field, value="reference"),)}
        )
        assert manager.get_imagetool(0).slicer_area._file_path is None

        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0,))
        qtbot.addWidget(dialog)
        dialog.model.set_fields(
            (first_field, next_field),
            {first_field: "reference", next_field: 20.0},
        )
        assert dialog.model.index(0, 0).data(QtCore.Qt.ItemDataRole.ToolTipRole) == str(
            file_path
        )
        assert dialog.model.index(0, 1).data(_MetadataTableModel.OriginRole) == (
            "assigned"
        )
        assert dialog.model.index(0, 2).data(_MetadataTableModel.OriginRole) == (
            "missing"
        )


def test_metadata_editor_layout_discovery_and_action_dispatch(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    context_field = AcquisitionContextField.from_value(
        kind="coordinate", name="photon_energy", value=21.2
    )
    saved_field = MetadataField(kind="attribute", name="operator")
    complex_field = MetadataField(kind="attribute", name="history")
    invalid_field = MetadataField(kind="attribute", name="unserializable")
    source = _batch_data("scan").assign_coords(temperature=20.0)
    source.attrs.update(
        {
            "sample": "reference",
            complex_field.name: [1, 2],
            invalid_field.name: object(),
        }
    )

    with manager_context() as manager:
        controller = manager._metadata_editor
        controller.restore_layout_payload(None)
        manager._acquisition_context.set_state(
            AcquisitionContextState(fields=(context_field,)), mark_dirty=False
        )
        controller.set_layout_fields((saved_field,))

        _add_batch_tools(qtbot, manager, source)
        fields, coverage, defaults = controller.discover_fields((0,))
        context_key = MetadataField(kind="coordinate", name="photon_energy")
        sample = MetadataField(kind="attribute", name="sample")
        temperature = MetadataField(kind="coordinate", name="temperature")
        assert {context_key, saved_field, complex_field, sample, temperature} <= set(
            fields
        )
        assert invalid_field not in fields
        assert coverage[context_key] == 0
        assert defaults[context_key] == 21.2
        assert controller.visible_fields(fields, defaults) == (saved_field,)
        assert controller.layout_payload()["initialized"] is True

        dirty_calls = 0

        def _mark_dirty() -> None:
            nonlocal dirty_calls
            dirty_calls += 1

        monkeypatch.setattr(
            manager._workspace_controller, "_mark_workspace_layout_dirty", _mark_dirty
        )
        controller.set_layout_fields((saved_field,))
        assert dirty_calls == 0
        controller.set_layout_fields((sample,))
        assert dirty_calls == 1
        controller.set_column_width(None, 237)
        controller.set_column_width(sample, 119)
        assert dirty_calls == 3
        controller.set_column_width(sample, 119)
        assert dirty_calls == 3
        controller.set_layout_fields((saved_field,))
        assert dirty_calls == 4
        assert controller.saved_column_width(None) == 237
        assert controller.saved_column_width(sample) == 119

        with pytest.warns(UserWarning, match="metadata editor layout"):
            controller.restore_layout_payload({"fields": [{"kind": "invalid"}]})
        assert not controller.layout_state.initialized
        manager._workspace_state.metadata_editor_layout = "invalid"  # type: ignore[assignment]
        assert not controller.layout_state.initialized
        controller.restore_layout_payload(None)

        calls: list[tuple[int | str, ...]] = []
        monkeypatch.setattr(
            metadata_editor.MetadataEditorDialog,
            "exec",
            lambda dialog: (
                calls.append(dialog.targets)
                or int(QtWidgets.QDialog.DialogCode.Rejected)
            ),
        )
        manager.tree_view.clearSelection()
        controller.show_editor()
        assert not calls
        select_tools(manager, [0])
        controller.show_editor()
        assert calls == [(0,)]


@pytest.mark.parametrize("remember_type", [False, True])
def test_metadata_editor_parses_remembered_missing_numeric_fields(
    qtbot,
    remember_type: bool,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    field = MetadataField(kind="coordinate", name="temperature")
    payload: dict[str, object] = {
        "schema_version": 3 if remember_type else 2,
        "initialized": True,
        "fields": [field.model_dump(mode="json")],
    }
    if remember_type:
        payload["field_types"] = [
            {"field": field.model_dump(mode="json"), "value_type": "Float"}
        ]

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, _batch_data("scan"))
        manager._metadata_editor.restore_layout_payload(payload)
        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0,))
        qtbot.addWidget(dialog)
        column = dialog.model.fields.index(field) + 1

        assert dialog.model.setData(dialog.model.index(0, column), "20.5")
        value = dialog.model.edited_values[(0, field)]
        assert value == 20.5
        assert isinstance(value, float)


def test_metadata_editor_updates_child_source_assignments(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    parent = _batch_data("parent")
    child = parent.assign_attrs(seed=1)
    field = MetadataField(kind="attribute", name="sample")

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, parent)
        child_tool = itool(child, manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(AssignAttrsOperation(attrs={"seed": 1})),
        )
        child_node = manager._node_for_target(child_uid)
        assert child_node.displayed_source_spec is not None

        assert manager._metadata_editor.apply_edits(
            {child_uid: (MetadataCellEdit(field, value="reference"),)}
        )

        assert manager.get_imagetool(child_uid).slicer_area.data.attrs["sample"] == (
            "reference"
        )
        assert "sample" not in manager.get_imagetool(0).slicer_area.data.attrs


def test_metadata_preserving_replacement_keeps_child_source_binding(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    parent = _batch_data("parent")
    source_spec = full_data(
        AssignAttrsOperation(attrs={"sample": "reference", "seed": 1})
    )

    with manager_context() as manager:
        _add_batch_tools(qtbot, manager, parent)
        child_tool = itool(source_spec.apply(parent), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=source_spec,
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)

        manual = _batch_data("manual", offset=100.0)
        replace_data(child_uid, manual)

        assert child_node.has_source_binding
        assert child_node.source_auto_update
        assert child_node.source_spec == source_spec
        xr.testing.assert_identical(
            child_tool.slicer_area._data,
            manual.assign_attrs(sample="reference", seed=1),
        )

        updated_parent = _batch_data("updated", offset=200.0)
        replace_data(0, updated_parent)
        qtbot.wait_until(
            lambda: (
                child_tool.slicer_area._data.attrs.get("sample") == "reference"
                and np.array_equal(
                    child_tool.slicer_area._data.values,
                    updated_parent.values,
                )
            )
        )
        xr.testing.assert_identical(
            child_tool.slicer_area._data,
            updated_parent.assign_attrs(sample="reference", seed=1),
        )


def test_metadata_editor_replaces_assignments_in_place_and_reverts(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")
    context_fields = (
        AcquisitionContextField.from_value(
            kind="coordinate", name="temperature", value=20.0
        ),
        AcquisitionContextField.from_value(
            kind="attribute", name="sample", value="reference"
        ),
        AcquisitionContextField.from_value(kind="attribute", name="run", value=1),
    )
    fields = (
        MetadataField(kind="coordinate", name="temperature"),
        MetadataField(kind="attribute", name="sample"),
        MetadataField(kind="attribute", name="run"),
    )
    edits = tuple(
        MetadataCellEdit(field, value=value)
        for field, value in zip(fields, (25.0, "edited", 2), strict=True)
    )

    with manager_context() as manager:
        manager._acquisition_context.set_state(
            AcquisitionContextState(enabled=True, fields=context_fields),
            mark_dirty=False,
        )
        assert manager._data_ingress.receive_data(
            [source],
            {
                "file_path": file_path,
                "load_func": (
                    xr.load_dataarray,
                    {"engine": "h5netcdf"},
                    FileDataSelection(kind="dataarray"),
                ),
            },
            show=False,
        ) == [True]
        node = manager._tool_graph.root_wrappers[0]
        before_spec = node.displayed_provenance_spec
        assert before_spec is not None
        before_assignments = tuple(
            operation
            for _ref, operation in iter_operation_refs(before_spec)
            if isinstance(operation, AssignScalarCoordOperation | AssignAttrsOperation)
        )
        assert len(before_assignments) == 2

        file_path.unlink()
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            assert manager._metadata_editor.apply_edits({0: edits})

        actual = manager.get_imagetool(0).slicer_area.data
        assert actual.coords["temperature"].item() == 25.0
        assert actual.attrs["sample"] == "edited"
        assert actual.attrs["run"] == 2
        edited_spec = node.displayed_provenance_spec
        assert edited_spec is not None
        edited_assignments = tuple(
            operation
            for _ref, operation in iter_operation_refs(edited_spec)
            if isinstance(operation, AssignScalarCoordOperation | AssignAttrsOperation)
        )
        assert len(edited_assignments) == len(before_assignments)
        scalar_operation = next(
            operation
            for operation in edited_assignments
            if isinstance(operation, AssignScalarCoordOperation)
        )
        attrs_operation = next(
            operation
            for operation in edited_assignments
            if isinstance(operation, AssignAttrsOperation)
        )
        assert scalar_operation.decoded_value == 25.0
        assert attrs_operation.attrs == {"sample": "edited", "run": 2}

        source.to_netcdf(file_path, engine="h5netcdf")
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            assert manager._metadata_editor.apply_edits(
                {0: tuple(MetadataCellEdit(field, revert=True) for field in fields)}
            )
        xr.testing.assert_identical(manager.get_imagetool(0).slicer_area.data, source)
        deactivated_spec = node.displayed_provenance_spec
        assert deactivated_spec is not None
        assert not any(
            isinstance(operation, AssignScalarCoordOperation | AssignAttrsOperation)
            for _ref, operation in iter_operation_refs(deactivated_spec)
        )


def test_metadata_editor_preflights_every_target(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    compatible = _batch_data("compatible")
    incompatible = _batch_data("incompatible", dims=("angle", "y"))
    field = MetadataField(kind="coordinate", name="angle")
    edit = MetadataCellEdit(field, value=1.5)

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, compatible, incompatible)
        assert not manager._metadata_editor.apply_edits({0: (edit,), 1: (edit,)})
        assert messages
        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area.data, compatible
        )
        xr.testing.assert_identical(
            manager.get_imagetool(1).slicer_area.data, incompatible
        )


def test_batch_operation_metadata_matches_launcher() -> None:
    dialog_classes = _batch_operation_dialog_classes()
    assert dialog_classes
    assert imagetool_dialogs.EdgeCorrectionDialog not in dialog_classes
    assert imagetool_dialogs.ROIPathDialog not in dialog_classes
    assert imagetool_dialogs.ROIMaskDialog not in dialog_classes
    assert [
        dialog_cls for dialog_cls in dialog_classes if not dialog_cls.operation_types
    ] == []
    assert [
        operation_type
        for dialog_cls in dialog_classes
        for operation_type in dialog_cls.operation_types
        if not operation_type.batch_available
    ] == []
    assert RestoreNonuniformDimsOperation.batch_available


def test_batch_dialog_defensive_paths_and_launch(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(
            qtbot,
            manager,
            _batch_data("scan0"),
            _batch_data("scan1", offset=100.0),
        )
        select_tools(manager, [0, 1])
        dialog = _BatchOperationDialog(manager)

        with monkeypatch.context() as mp:
            mp.setattr(
                manager_dialogs,
                "_BATCH_OPERATION_CATEGORIES",
                (("Hidden", (imagetool_dialogs.EdgeCorrectionDialog,)),),
            )
            dialog._populate_operations()
            assert dialog._operation_tree.topLevelItemCount() == 0
        dialog._populate_operations()

        dialog.show()
        qtbot.wait_until(lambda: dialog.isVisible(), timeout=1000)
        dialog.close()

        _select_batch_operation(dialog, imagetool_dialogs.NormalizeDialog)
        manager.tree_view.deselect_all()
        select_tools(manager, [0])
        dialog._launch_selected_operation()
        assert messages

        launched_dialogs = []

        def _capture_tool_window(widget, **kwargs) -> None:
            launched_dialogs.append((widget, kwargs))

        select_tools(manager, [0, 1])
        monkeypatch.setattr(
            manager.get_imagetool(0).slicer_area,
            "add_tool_window",
            _capture_tool_window,
        )
        dialog._launch_selected_operation()
        assert isinstance(
            launched_dialogs[0][0],
            imagetool_dialogs.NormalizeDialog,
        )
        assert launched_dialogs[0][1] == {
            "update_title": False,
            "transfer_to_manager": False,
        }

        invalid_item = QtWidgets.QTreeWidgetItem(["invalid"])
        invalid_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, object())
        dialog._target_item_changed(invalid_item, 0)

        missing_item = QtWidgets.QTreeWidgetItem(["missing"])
        missing_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, "missing")
        dialog._target_item_changed(missing_item, 0)

        valid_item = dialog._target_items[0]
        monkeypatch.setattr(manager.tree_view, "selectionModel", lambda: None)
        dialog._target_item_changed(valid_item, 0)

        invalid_operation_item = QtWidgets.QTreeWidgetItem(["invalid"])
        invalid_operation_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, object())
        dialog._operation_tree.addTopLevelItem(invalid_operation_item)
        dialog._operation_tree.setCurrentItem(invalid_operation_item)
        assert dialog._selected_dialog_cls() is None

        old_launch_button = dialog._launch_button
        dialog._launch_button = None
        dialog._update_launch_enabled()
        dialog._launch_button = old_launch_button

        dialog._manager = lambda: None
        dialog._populate_targets()
        dialog._add_target_item(0)
        dialog._sync_target_checks()
        dialog._target_item_changed(valid_item, 0)
        dialog._launch_selected_operation()


def test_update_info_handles_legacy_imagetool_preview_attribute(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        node = manager._tool_graph.root_wrappers[0]

        def _missing_preview_attribute(_node) -> tuple[float, QtGui.QPixmap]:
            raise AttributeError("_preview_image")

        monkeypatch.setattr(
            type(node), "_preview_image", property(_missing_preview_attribute)
        )
        select_tools(manager, [0])

        manager._update_info()

        assert manager.preview_widget.isVisible()


def test_manager_summary_sorts_coordinate_order_at_presentation_boundary(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={
            "x": [0.0, 1.0],
            "y": [0.0, 1.0, 2.0],
            "aux": ("x", [10.0, 20.0]),
        },
    )
    unsorted = erlab.utils.array.sort_coord_order(
        data,
        ("aux", "y", "x"),
        dims_first=False,
    )
    assert tuple(unsorted.coords) == ("aux", "y", "x")
    formatted_coord_orders: list[tuple[typing.Any, ...]] = []

    def format_darr_html(value: xr.DataArray, **_kwargs: typing.Any) -> str:
        formatted_coord_orders.append(tuple(value.coords))
        return "<p>summary</p>"

    with manager_context() as manager:
        tool = itool(data, manager=False, execute=False)
        manager.add_imagetool(tool, show=False)
        node = manager._tool_graph.root_wrappers[0]
        monkeypatch.setattr(type(node), "_metadata_data", lambda _self: unsorted)
        monkeypatch.setattr(
            erlab.utils.formatting,
            "format_darr_html",
            format_darr_html,
        )

        assert "summary" in node.info_text

    assert formatted_coord_orders == [("x", "y", "aux")]


def test_details_panel_update_info_hides_missing_child_preview_pixmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakePreviewWidget:
        def __init__(self) -> None:
            self.visible = True
            self.pixmaps: list[QtGui.QPixmap] = []

        def setPixmap(self, pixmap: QtGui.QPixmap) -> None:
            self.pixmaps.append(pixmap)

        def setVisible(self, visible: bool) -> None:
            self.visible = visible

    class _FakeImageItem:
        def getPixmap(self) -> None:
            return None

    metadata_nodes: list[object] = []
    preview_widget = _FakePreviewWidget()
    node = types.SimpleNamespace(
        uid="child-1",
        is_imagetool=False,
        tool_window=types.SimpleNamespace(
            preview_pixmap=None,
            preview_pixmap_stale=False,
            preview_imageitem=_FakeImageItem(),
        ),
    )
    manager = types.SimpleNamespace(
        text_box=types.SimpleNamespace(setHtml=lambda _html: None),
        preview_widget=preview_widget,
        _selected_imagetool_targets=list,
        _selected_tool_uids=lambda: ["child-1"],
        _node_for_target=lambda _target: node,
        _node_info_html=lambda _node: "<p>child</p>",
        _set_metadata_node=lambda metadata_node: metadata_nodes.append(metadata_node),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "qt_is_valid",
        lambda *objects: all(obj is not None for obj in objects),
    )
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )

    controller._update_info(uid="child-1")

    assert metadata_nodes == [node]
    assert preview_widget.visible is False
    assert preview_widget.pixmaps == []


def test_details_panel_update_info_uses_default_child_preview_aspect_ratio() -> None:
    class _FakePreviewWidget:
        def __init__(self) -> None:
            self.visible = False
            self.pixmap_calls: list[tuple[QtGui.QPixmap, dict[str, object]]] = []

        def setPixmap(self, pixmap: QtGui.QPixmap, **kwargs: object) -> None:
            self.pixmap_calls.append((pixmap, kwargs))

        def setVisible(self, visible: bool) -> None:
            self.visible = visible

    metadata_nodes: list[object] = []
    preview_widget = _FakePreviewWidget()
    pixmap = QtGui.QPixmap(16, 8)
    pixmap.fill(QtGui.QColor("red"))
    node = types.SimpleNamespace(
        uid="child-1",
        is_imagetool=False,
        tool_window=types.SimpleNamespace(
            preview_pixmap=pixmap,
            preview_pixmap_stale=False,
            preview_imageitem=None,
        ),
    )
    manager = types.SimpleNamespace(
        text_box=types.SimpleNamespace(setHtml=lambda _html: None),
        preview_widget=preview_widget,
        _selected_imagetool_targets=list,
        _selected_tool_uids=lambda: ["child-1"],
        _node_for_target=lambda _target: node,
        _node_info_html=lambda _node: "<p>child</p>",
        _set_metadata_node=lambda metadata_node: metadata_nodes.append(metadata_node),
    )
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )

    controller._update_info(uid="child-1")

    assert metadata_nodes == [node]
    assert preview_widget.pixmap_calls == [(pixmap, {})]
    assert preview_widget.visible is True


def test_details_panel_script_input_labels_omit_redundant_history() -> None:
    manager = types.SimpleNamespace(
        _tool_graph=types.SimpleNamespace(nodes={}),
    )
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )

    assert (
        controller._script_input_row_label(
            ScriptInput(name="source"), include_history=False
        )
        == "Use source"
    )
    assert (
        controller._script_input_row_label(
            ScriptInput(name="source", node_uid="missing"),
            include_history=True,
        )
        == "Missing source for source"
    )


def test_single_image_preview_does_not_show_null_pixmap(qtbot) -> None:
    preview = manager_widgets._SingleImagePreview()
    qtbot.addWidget(preview)

    preview.setPixmap(QtGui.QPixmap())
    preview._fit_pixmap_in_view()
    preview.setVisible(True)
    preview.show()

    assert not preview.isVisible()
    assert preview.sizeHint() == QtCore.QSize(0, 0)
    assert preview.minimumSizeHint() == QtCore.QSize(0, 0)

    pixmap = QtGui.QPixmap(16, 8)
    pixmap.fill(QtGui.QColor("red"))
    preview.setPixmap(pixmap)
    preview.setVisible(True)

    assert preview.isVisible()


def test_single_image_preview_keeps_legacy_stretch_on_resize(qtbot) -> None:
    preview = manager_widgets._SingleImagePreview()
    qtbot.addWidget(preview)
    pixmap = QtGui.QPixmap(160, 80)
    pixmap.fill(QtGui.QColor("red"))

    preview.resize(320, 80)
    preview.setPixmap(pixmap)
    preview.setVisible(True)
    qtbot.waitExposed(preview)
    first_transform = preview.transform()

    preview.resize(120, 300)
    qtbot.wait(0)
    second_transform = preview.transform()

    assert first_transform.m11() != pytest.approx(first_transform.m22())
    assert second_transform.m11() != pytest.approx(second_transform.m22())


def test_single_image_preview_can_keep_aspect_ratio_on_resize(qtbot) -> None:
    preview = manager_widgets._SingleImagePreview()
    qtbot.addWidget(preview)
    pixmap = QtGui.QPixmap(160, 80)
    pixmap.fill(QtGui.QColor("red"))

    preview.resize(320, 80)
    preview.setPixmap(
        pixmap,
        aspect_ratio_mode=QtCore.Qt.AspectRatioMode.KeepAspectRatio,
    )
    preview.setVisible(True)
    qtbot.waitExposed(preview)
    first_transform = preview.transform()

    preview.resize(120, 300)
    qtbot.wait(0)
    second_transform = preview.transform()

    assert first_transform.m11() == pytest.approx(first_transform.m22())
    assert second_transform.m11() == pytest.approx(second_transform.m22())


def test_batch_action_transform_error_paths(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, _batch_data("scan0"))
        manager.show_batch_operations()
        assert manager._actions_controller._ActionsController__batch_dialog is None

        _add_batch_tools(qtbot, manager, _batch_data("scan1", offset=100.0))
        select_tools(manager, [0, 1])

        with monkeypatch.context() as mp:
            mp.setattr(manager, "_selected_imagetool_targets", lambda: [0])
            assert not manager.apply_batch_transform_dialog(
                _BatchTransformStub(),
                "replace",
            )
            mp.setattr(manager, "_selected_imagetool_targets", lambda: [0, "tool"])
            mp.setattr(
                manager,
                "_is_imagetool_target",
                lambda target: target == 0,
            )
            assert not manager.apply_batch_transform_dialog(
                _BatchTransformStub(),
                "replace",
            )
        select_tools(manager, [0, 1])

        assert not manager.apply_batch_transform_dialog(
            _BatchTransformStub(source_error=RuntimeError("source failed")),
            "replace",
        )
        assert not manager.apply_batch_transform_dialog(
            _BatchTransformStub([]),
            "replace",
        )
        assert not manager.apply_batch_transform_dialog(
            _BatchTransformStub([NormalizeOperation(dims=("x",))]),
            "replace",
        )

        class _ScriptTransformStub(_BatchTransformStub):
            operation_types = (ScriptCodeOperation,)

        assert not manager.apply_batch_transform_dialog(
            _ScriptTransformStub([ScriptCodeOperation(label="Custom", code=None)]),
            "replace",
        )

        manager._node_for_target(0).set_source_binding(full_data())
        manager._node_for_target(1).set_detached_provenance(full_data())
        assert manager.apply_batch_transform_dialog(
            _BatchTransformStub(public_source=True),
            "replace",
        )

        monkeypatch.setattr(erlab.interactive, "itool", lambda **kwargs: None)
        assert not manager.apply_batch_transform_dialog(
            _BatchTransformStub(),
            "detach",
        )

        def _raise_replace(*args, **kwargs) -> None:
            raise RuntimeError("replace failed")

        monkeypatch.setattr(
            manager.get_imagetool(0).slicer_area,
            "replace_source_data",
            _raise_replace,
        )
        assert not manager.apply_batch_transform_dialog(
            _BatchTransformStub(),
            "replace",
        )
        assert len(messages) >= 6


def test_batch_transform_memory_preflight_runs_before_processing(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    memory = _kspace_conversion.KspaceMemoryBudget(
        total_bytes=4,
        available_bytes=2,
        reserve_bytes=1,
        safe_budget_bytes=1,
    )
    estimate = _kspace_conversion.KspaceConversionEstimate(
        input_dims=(),
        output_dims=("kx",),
        axis_sizes={"kx": 2},
        output_sizes={"kx": 2},
        bounds={"kx": (-1.0, 1.0)},
        resolution={"kx": 1.0},
        total_points=2,
        final_bytes=16,
        peak_bytes=32,
        memory=memory,
    )

    class _PreflightStub(_BatchTransformStub):
        def __init__(self) -> None:
            super().__init__(public_source=True)
            self.source_spec_calls = 0
            self.preflight_calls = 0

        def source_spec_for_data(
            self,
            data: xr.DataArray,
            new_name: str | None = None,
        ) -> ToolProvenanceSpec:
            self.source_spec_calls += 1
            return super().source_spec_for_data(data, new_name)

        def preflight_data(self, data: xr.DataArray) -> None:
            del data
            self.preflight_calls += 1
            if self.preflight_calls == 2:
                raise _kspace_conversion.KspaceConversionMemoryError(estimate)

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(
            qtbot,
            manager,
            _batch_data("scan0"),
            _batch_data("scan1", offset=100.0),
        )
        select_tools(manager, [0, 1])
        dialog = _PreflightStub()

        assert not manager.apply_batch_transform_dialog(dialog, "replace")

    assert dialog.preflight_calls == 2
    assert dialog.source_spec_calls == 2
    assert messages
    assert messages[-1][1]["buttons"] == QtWidgets.QDialogButtonBox.StandardButton.Ok


def test_batch_filter_error_paths(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(
            qtbot,
            manager,
            _batch_data("scan0"),
            _batch_data("scan1", offset=100.0),
        )
        select_tools(manager, [0, 1])

        class _ScriptFilterStub(_BatchFilterStub):
            operation_types = (ScriptCodeOperation,)

        assert not manager.apply_batch_filter_dialog(
            _ScriptFilterStub(ScriptCodeOperation(label="Custom", code=None))
        )
        assert not manager.apply_batch_filter_dialog(
            _BatchFilterStub(NormalizeOperation(dims=("missing",)))
        )

        real_get_imagetool = manager.get_imagetool
        get_imagetool_calls = 0

        def _raise_during_commit(target):
            nonlocal get_imagetool_calls
            get_imagetool_calls += 1
            if get_imagetool_calls > 2:
                raise RuntimeError("commit failed")
            return real_get_imagetool(target)

        monkeypatch.setattr(manager, "get_imagetool", _raise_during_commit)
        assert not manager.apply_batch_filter_dialog(
            _BatchFilterStub(NormalizeOperation(dims=("x",)))
        )
        assert len(messages) >= 3


def test_batch_dialog_accept_wrappers(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(
            qtbot,
            manager,
            _batch_volume("scan0"),
            _batch_volume("scan1", offset=100.0),
        )
        area = manager.get_imagetool(0).slicer_area

        single_dialog = imagetool_dialogs.RotationDialog(area)
        qtbot.addWidget(single_dialog)
        assert single_dialog.batch_manager is None
        assert single_dialog.source_spec().kind == "full_data"
        assert single_dialog._itool_kwargs(area.data)["data"] is area.data

        transform_dialog = imagetool_dialogs.RotationDialog(
            area,
            batch_manager=manager,
        )
        monkeypatch.setattr(
            manager,
            "apply_batch_transform_dialog",
            lambda dialog, launch_mode: False,
        )
        transform_dialog.accept()
        assert transform_dialog.result() == int(QtWidgets.QDialog.DialogCode.Rejected)

        transform_dialog = imagetool_dialogs.RotationDialog(
            area,
            batch_manager=manager,
        )
        monkeypatch.setattr(
            manager,
            "apply_batch_transform_dialog",
            lambda dialog, launch_mode: True,
        )
        transform_dialog.accept()
        assert transform_dialog.result() == int(QtWidgets.QDialog.DialogCode.Accepted)

        transform_dialog = imagetool_dialogs.RotationDialog(
            area,
            batch_manager=manager,
        )

        def _raise_transform(dialog, launch_mode) -> bool:
            raise RuntimeError("batch failed")

        monkeypatch.setattr(
            manager,
            "apply_batch_transform_dialog",
            _raise_transform,
        )
        transform_dialog.accept()

        filter_dialog = imagetool_dialogs.NormalizeDialog(area, batch_manager=manager)
        monkeypatch.setattr(
            manager,
            "apply_batch_filter_dialog",
            lambda dialog: False,
        )
        filter_dialog.accept()
        assert filter_dialog.result() == int(QtWidgets.QDialog.DialogCode.Rejected)

        filter_dialog = imagetool_dialogs.NormalizeDialog(area, batch_manager=manager)
        monkeypatch.setattr(
            manager,
            "apply_batch_filter_dialog",
            lambda dialog: True,
        )
        filter_dialog.accept()
        assert filter_dialog.result() == int(QtWidgets.QDialog.DialogCode.Accepted)

        filter_dialog = imagetool_dialogs.NormalizeDialog(area, batch_manager=manager)

        def _raise_filter(dialog) -> bool:
            raise RuntimeError("batch failed")

        monkeypatch.setattr(manager, "apply_batch_filter_dialog", _raise_filter)
        filter_dialog.accept()
        assert messages


def test_batch_dialog_categories_and_target_checks_update_manager_selection(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(
            qtbot,
            manager,
            _batch_data("scan0"),
            _batch_data("scan1", offset=100.0),
        )
        manager.tree_view.deselect_all()
        manager._update_actions()
        assert manager.batch_action.isEnabled()

        select_tools(manager, [0, 1])
        dialog = _BatchOperationDialog(manager)

        assert dialog._operation_tree.topLevelItemCount() == 4
        rendered_dialogs = []
        for top_index in range(dialog._operation_tree.topLevelItemCount()):
            top_item = dialog._operation_tree.topLevelItem(top_index)
            assert top_item.childCount() > 0
            rendered_dialogs.extend(
                top_item.child(child_index).data(0, QtCore.Qt.ItemDataRole.UserRole)
                for child_index in range(top_item.childCount())
            )
        assert set(_batch_operation_dialog_classes()) == set(rendered_dialogs)

        assert dialog._launch_button is not None
        assert not dialog._launch_button.isEnabled()
        _select_batch_operation(dialog, imagetool_dialogs.NormalizeDialog)
        assert dialog._launch_button.isEnabled()

        dialog._target_items[1].setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        qtbot.wait_until(
            lambda: manager._selected_imagetool_targets() == [0],
            timeout=1000,
        )
        assert not dialog._launch_button.isEnabled()

        dialog._target_items[1].setCheckState(0, QtCore.Qt.CheckState.Checked)
        qtbot.wait_until(
            lambda: manager._selected_imagetool_targets() == [0, 1],
            timeout=1000,
        )
        assert dialog._launch_button.isEnabled()


def test_batch_dialog_open_refreshes_cached_targets(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(
            qtbot,
            manager,
            _batch_data("scan0"),
            _batch_data("scan1", offset=100.0),
        )

        manager.show_batch_operations()
        dialog = manager._actions_controller._batch_dialog
        qtbot.wait_until(lambda: dialog.isVisible(), timeout=1000)
        assert set(dialog._target_items) == {0, 1}

        dialog.close()
        _add_batch_tools(qtbot, manager, _batch_data("scan2", offset=200.0))

        manager.show_batch_operations()
        qtbot.wait_until(lambda: dialog.isVisible(), timeout=1000)
        assert set(dialog._target_items) == {0, 1, 2}


def test_batch_action_updates_when_child_imagetool_count_changes(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, _batch_data("scan0"))
        assert manager.batch_target_count() == 1
        assert not manager.batch_action.isEnabled()

        child_tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(_batch_data("child"), manager=False, execute=False),
        )
        child_uid = manager.add_imagetool_child(child_tool, 0, show=False)
        assert manager.batch_target_count() == 2
        assert manager.batch_action.isEnabled()

        manager._remove_childtool(child_uid)
        assert manager.batch_target_count() == 1
        assert not manager.batch_action.isEnabled()


def test_batch_action_counts_pending_memory_imagetools_without_materializing(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        for offset in (0.0, 100.0):
            tool = typing.cast(
                "erlab.interactive.imagetool.ImageTool",
                itool(
                    xr.DataArray(
                        np.arange(25, dtype=np.float64).reshape((5, 5)) + offset,
                        dims=["x", "y"],
                        name=f"pending_batch_{int(offset)}",
                    ),
                    manager=False,
                    execute=False,
                ),
            )
            manager.add_imagetool(tool, show=False)
            tool.hide()

        workspace_path = tmp_path / "pending-batch-targets.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("opening the batch dialog should not materialize pending data")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        select_tools(manager, [0, 1])
        manager._update_actions()
        assert manager.batch_target_count() == 2
        assert manager.batch_action.isEnabled()

        manager.show_batch_operations()
        dialog = manager._actions_controller._batch_dialog
        qtbot.wait_until(lambda: dialog.isVisible(), timeout=1000)
        assert set(dialog._target_items) == {0, 1}
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )


def test_batch_dialog_expands_child_targets(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, _batch_data("scan0"))

        child_tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(_batch_data("child"), manager=False, execute=False),
        )
        child_uid = manager.add_imagetool_child(child_tool, 0, show=False)

        manager.show_batch_operations()
        dialog = manager._actions_controller._batch_dialog
        qtbot.wait_until(lambda: dialog.isVisible(), timeout=1000)

        parent_item = dialog._target_tree.topLevelItem(0)
        assert parent_item is dialog._target_items[0]
        assert dialog._target_items[child_uid].parent() is parent_item
        assert parent_item.isExpanded()


def test_batch_action_updates_when_tool_child_subtree_removes_imagetool(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, _batch_data("scan0"))

        tool_uid = manager.add_childtool(
            erlab.interactive.utils.ToolWindow(),
            0,
            show=False,
        )
        child_tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(_batch_data("child"), manager=False, execute=False),
        )
        manager.add_imagetool_child(child_tool, tool_uid, show=False)
        assert manager.batch_target_count() == 2
        assert manager.batch_action.isEnabled()

        manager._remove_childtool(tool_uid)
        assert manager.batch_target_count() == 1
        assert not manager.batch_action.isEnabled()


@pytest.mark.parametrize("launch_mode", ["replace", "nest", "detach"])
def test_batch_transform_placements(
    qtbot,
    launch_mode: str,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = _batch_volume("scan0")
    data1 = _batch_volume("scan1", offset=100.0)

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, data0, data1)
        select_tools(manager, [0, 1])

        dialog = imagetool_dialogs.AggregateDialog(
            manager.get_imagetool(0).slicer_area,
            batch_manager=manager,
        )
        dialog.dim_checks["y"].setChecked(True)

        assert manager.apply_batch_transform_dialog(
            dialog,
            typing.cast("typing.Literal['replace', 'detach', 'nest']", launch_mode),
        )

        expected = [data0.qsel.mean("y"), data1.qsel.mean("y")]
        if launch_mode == "replace":
            assert manager.ntools == 2
            for index, expected_data in enumerate(expected):
                xarray.testing.assert_identical(
                    manager.get_imagetool(index).slicer_area._data.rename(None),
                    expected_data.rename(None),
                )
        elif launch_mode == "nest":
            assert manager.ntools == 2
            for index, expected_data in enumerate(expected):
                child_uids = manager._tool_graph.root_wrappers[index]._childtool_indices
                assert len(child_uids) == 1
                child_tool = manager.get_imagetool(child_uids[0])
                xarray.testing.assert_identical(
                    child_tool.slicer_area._data.rename(None),
                    expected_data.rename(None),
                )
        else:
            assert manager.ntools == 4
            for index, expected_data in zip((2, 3), expected, strict=True):
                xarray.testing.assert_identical(
                    manager.get_imagetool(index).slicer_area._data.rename(None),
                    expected_data.rename(None),
                )


def test_batch_sortby_transform_replace(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data0 = _batch_data("scan0").assign_coords(temperature=("x", [20.0, 10.0, 30.0]))
    data1 = _batch_data("scan1", offset=100.0).assign_coords(
        temperature=("x", [25.0, 5.0, 15.0])
    )

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, data0, data1)
        select_tools(manager, [0, 1])

        dialog = imagetool_dialogs.SortByDialog(
            manager.get_imagetool(0).slicer_area,
            batch_manager=manager,
        )
        for row in range(dialog.key_table.rowCount()):
            item = dialog.key_table.item(row, 0)
            if (
                item is not None
                and item.data(QtCore.Qt.ItemDataRole.UserRole) == "temperature"
            ):
                item.setCheckState(QtCore.Qt.CheckState.Checked)
        dialog.ascending_combo.setCurrentIndex(
            dialog.ascending_combo.findData(False, QtCore.Qt.ItemDataRole.UserRole)
        )

        assert manager.apply_batch_transform_dialog(dialog, "replace")
        assert manager.ntools == 2
        for index, expected_data in enumerate(
            (
                data0.sortby("temperature", ascending=False),
                data1.sortby("temperature", ascending=False),
            )
        ):
            xarray.testing.assert_identical(
                manager.get_imagetool(index).slicer_area._data.rename(None),
                expected_data.rename(None),
            )


def test_batch_filter_applies_in_place_only(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        _add_batch_tools(
            qtbot,
            manager,
            _batch_data("scan0"),
            _batch_data("scan1", offset=100.0),
        )
        select_tools(manager, [0, 1])

        dialog = imagetool_dialogs.NormalizeDialog(
            manager.get_imagetool(0).slicer_area,
            batch_manager=manager,
        )
        dialog.dim_checks["y"].setChecked(True)

        assert manager.apply_batch_filter_dialog(dialog)
        assert manager.ntools == 2
        for index in (0, 1):
            operation = manager.get_imagetool(
                index
            ).slicer_area._accepted_filter_provenance_operation
            assert isinstance(operation, NormalizeOperation)


def test_batch_transform_preflight_failure_leaves_all_targets_unchanged(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    data0 = _batch_volume("scan0")
    data1 = _batch_volume("scan1", dims=("x", "other", "z"), offset=100.0)

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, data0, data1)
        select_tools(manager, [0, 1])

        dialog = imagetool_dialogs.AggregateDialog(
            manager.get_imagetool(0).slicer_area,
            batch_manager=manager,
        )
        dialog.dim_checks["y"].setChecked(True)

        assert not manager.apply_batch_transform_dialog(dialog, "replace")
        assert messages
        xarray.testing.assert_identical(
            manager.get_imagetool(0).slicer_area._data.rename(None),
            data0.rename(None),
        )
        xarray.testing.assert_identical(
            manager.get_imagetool(1).slicer_area._data.rename(None),
            data1.rename(None),
        )


def test_batch_sortby_preflight_failure_leaves_all_targets_unchanged(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    data0 = _batch_data("scan0").assign_coords(temperature=("x", [20.0, 10.0, 30.0]))
    data1 = _batch_data("scan1", offset=100.0)

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, data0, data1)
        select_tools(manager, [0, 1])

        dialog = imagetool_dialogs.SortByDialog(
            manager.get_imagetool(0).slicer_area,
            batch_manager=manager,
        )
        for row in range(dialog.key_table.rowCount()):
            item = dialog.key_table.item(row, 0)
            if (
                item is not None
                and item.data(QtCore.Qt.ItemDataRole.UserRole) == "temperature"
            ):
                item.setCheckState(QtCore.Qt.CheckState.Checked)

        assert not manager.apply_batch_transform_dialog(dialog, "replace")
        assert messages
        xarray.testing.assert_identical(
            manager.get_imagetool(0).slicer_area._data.rename(None),
            data0.rename(None),
        )
        xarray.testing.assert_identical(
            manager.get_imagetool(1).slicer_area._data.rename(None),
            data1.rename(None),
        )


def test_batch_add_coord_duplicate_target_aborts_all_unchanged(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    messages = _block_message_dialog(monkeypatch)
    data0 = _batch_data("scan0")
    data1 = _batch_data("scan1", offset=100.0).assign_coords(foo=5.0)

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, data0, data1)
        select_tools(manager, [0, 1])

        dialog = imagetool_dialogs.AssignCoordsDialog(
            manager.get_imagetool(0).slicer_area,
            batch_manager=manager,
        )
        dialog._mode_tabs.setCurrentIndex(1)
        dialog._add_name_edit.setText("foo")

        assert not manager.apply_batch_transform_dialog(dialog, "replace")
        assert messages
        xarray.testing.assert_identical(
            manager.get_imagetool(0).slicer_area._data.rename(None),
            data0.rename(None),
        )
        xarray.testing.assert_identical(
            manager.get_imagetool(1).slicer_area._data.rename(None),
            data1.rename(None),
        )


def test_batch_nonuniform_restore_is_target_specific(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    uniform = _batch_volume("uniform")
    nonuniform = _batch_volume(
        "nonuniform",
        offset=100.0,
        first_coord=[0.0, 0.4, 1.1],
    )

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, uniform, nonuniform)
        select_tools(manager, [0, 1])

        dialog = imagetool_dialogs.AggregateDialog(
            manager.get_imagetool(0).slicer_area,
            batch_manager=manager,
        )
        dialog.dim_checks["y"].setChecked(True)

        assert manager.apply_batch_transform_dialog(dialog, "nest")

        uniform_child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        nonuniform_child_uid = manager._tool_graph.root_wrappers[1]._childtool_indices[
            0
        ]
        uniform_source_spec = manager._child_node(uniform_child_uid).source_spec
        nonuniform_source_spec = manager._child_node(nonuniform_child_uid).source_spec
        assert uniform_source_spec is not None
        assert nonuniform_source_spec is not None
        assert [op.op for op in uniform_source_spec.operations] == ["qsel_aggregate"]
        assert [op.op for op in nonuniform_source_spec.operations] == [
            "qsel_aggregate",
            "restore_nonuniform_dims",
        ]


def test_manager_metadata_full_code_generated_only_when_copied(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    calls: list[str] = []
    copied: list[str] = []

    def fake_derivation_code(wrapper):
        calls.append(wrapper.uid)
        return "derived = xr.DataArray([1.0])"

    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda text: copied.append(text) or text,
    )

    with manager_context() as manager:
        manager.show()
        itool(xr.DataArray([1.0], dims=("x",)), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        wrapper = manager._tool_graph.root_wrappers[0]
        wrapper.set_detached_provenance(
            full_data(RenameOperation(name="renamed")).to_replay_spec()
        )

        monkeypatch.setattr(
            type(wrapper),
            "derivation_code",
            property(fake_derivation_code),
        )
        manager._set_metadata_node(wrapper)

        assert not calls
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert calls == [wrapper.uid]
        assert copied == ["derived = xr.DataArray([1.0])"]


def test_load_source_details_dialog_uses_native_readonly_details(
    qtbot, monkeypatch, tmp_path
) -> None:
    kwargs_text = (
        'engine="h5netcdf", '
        "very_long_keyword_argument_name=123, "
        'another_long_keyword_argument_name="abcdef"'
    )
    revealed: list[pathlib.Path] = []
    opened_in_explorer: list[pathlib.Path] = []
    source_path = tmp_path / "scan.nc"
    source_path.write_bytes(b"")
    monkeypatch.setattr(
        erlab.utils.misc,
        "open_in_file_manager",
        lambda path: revealed.append(pathlib.Path(path)),
    )
    dialog = _LoadSourceDetailsDialog(
        _LoadSourceDetails(
            path=source_path,
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text=kwargs_text,
            load_code=None,
        ),
        show_in_data_explorer=lambda path: opened_in_explorer.append(
            pathlib.Path(path)
        ),
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog.findChildren(QtWidgets.QLineEdit)
    assert not dialog.findChildren(QtWidgets.QPlainTextEdit)
    assert dialog.minimumSize() == dialog.maximumSize()
    assert dialog.height() == dialog.minimumSizeHint().height()
    assert dialog.height() <= dialog.sizeHint().height()
    assert dialog.reveal_button is not None
    assert dialog.data_explorer_button is not None
    assert dialog.reveal_button.isEnabled()
    assert dialog.data_explorer_button.isEnabled()
    assert (
        dialog.findChild(QtWidgets.QAbstractButton, "manager_edit_load_source_button")
        is None
    )

    path_label = dialog.findChild(
        QtWidgets.QLabel, "manager_load_source_path_value_label"
    )
    loader_label = dialog.findChild(
        QtWidgets.QLabel, "manager_load_source_loader_value_label"
    )
    arguments_label = dialog.findChild(
        QtWidgets.QLabel, "manager_load_source_arguments_value_label"
    )
    assert path_label is not None
    assert loader_label is not None
    assert arguments_label is not None
    assert isinstance(path_label, manager_widgets._ElidedValueLabel)
    assert isinstance(loader_label, manager_widgets._ElidedValueLabel)
    assert not isinstance(arguments_label, manager_widgets._ElidedValueLabel)
    assert path_label.text() == str(source_path)
    assert path_label.full_text == str(source_path)
    assert path_label.toolTip() == str(source_path)
    assert path_label.textFormat() == QtCore.Qt.TextFormat.PlainText
    assert path_label.textInteractionFlags() == (
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    assert not path_label.wordWrap()
    assert path_label.sizePolicy().horizontalPolicy() == (
        QtWidgets.QSizePolicy.Policy.Ignored
    )
    assert loader_label.text() == "xarray.load_dataarray"
    assert not loader_label.wordWrap()
    assert loader_label.toolTip() != loader_label.text()
    assert "xarray.load_dataarray" in loader_label.toolTip()
    assert kwargs_text not in loader_label.toolTip()
    assert arguments_label.text() == kwargs_text
    assert arguments_label.toolTip() == kwargs_text
    assert arguments_label.textInteractionFlags() == (
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    assert arguments_label.wordWrap()

    dialog.reveal_button.click()
    assert dialog.result() == int(QtWidgets.QDialog.DialogCode.Accepted)
    assert revealed == [source_path]

    explorer_dialog = _LoadSourceDetailsDialog(
        _LoadSourceDetails(
            path=source_path,
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text=kwargs_text,
            load_code=None,
        ),
        show_in_data_explorer=lambda path: opened_in_explorer.append(
            pathlib.Path(path)
        ),
    )
    qtbot.addWidget(explorer_dialog)
    assert explorer_dialog.data_explorer_button is not None
    explorer_dialog.data_explorer_button.click()
    assert explorer_dialog.result() == int(QtWidgets.QDialog.DialogCode.Accepted)
    assert opened_in_explorer == [source_path]


def test_load_source_details_dialog_disables_data_explorer_without_callback(
    qtbot, tmp_path
) -> None:
    source_path = tmp_path / "scan.nc"
    source_path.write_bytes(b"")
    dialog = _LoadSourceDetailsDialog(
        _LoadSourceDetails(
            path=source_path,
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            load_code=None,
        )
    )
    qtbot.addWidget(dialog)

    data_explorer_button = dialog.findChild(
        QtWidgets.QAbstractButton, "manager_show_load_source_in_explorer_button"
    )
    assert data_explorer_button is not None
    assert not data_explorer_button.isEnabled()


def test_load_source_details_dialog_loader_tooltip_includes_plugin_description(
    qtbot, tmp_path, example_loader
) -> None:
    source_path = tmp_path / "scan.h5"
    source_path.write_bytes(b"")
    dialog = _LoadSourceDetailsDialog(
        _LoadSourceDetails(
            path=source_path,
            loader_label="Loader",
            loader_text="example",
            kwargs_text="single=True",
            load_code=None,
        )
    )
    qtbot.addWidget(dialog)

    loader_label = dialog.findChild(
        QtWidgets.QLabel, "manager_load_source_loader_value_label"
    )
    assert loader_label is not None
    assert "example" in loader_label.toolTip()
    assert erlab.io.loaders["example"].description in loader_label.toolTip()
    assert "single=True" not in loader_label.toolTip()


def test_load_source_details_dialog_marks_missing_source_file(qtbot, tmp_path) -> None:
    source_path = (
        tmp_path
        / ("very_long_directory_name_" * 3)
        / "missing_scan_with_a_long_name.nc"
    )
    dialog = _LoadSourceDetailsDialog(
        _LoadSourceDetails(
            path=source_path,
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            load_code=None,
        ),
        show_in_data_explorer=lambda _path: None,
    )
    qtbot.addWidget(dialog)

    status_label = dialog.findChild(
        QtWidgets.QLabel, "manager_load_source_status_label"
    )
    path_label = dialog.findChild(
        QtWidgets.QLabel, "manager_load_source_path_value_label"
    )
    assert status_label is not None
    assert path_label is not None
    assert isinstance(path_label, manager_widgets._ElidedValueLabel)
    assert status_label.property("manager_source_missing") is True
    assert path_label.property("manager_source_missing") is True
    assert path_label.text() == str(source_path)
    assert path_label.full_text == str(source_path)
    assert not path_label.wordWrap()
    assert path_label.sizePolicy().horizontalPolicy() == (
        QtWidgets.QSizePolicy.Policy.Ignored
    )
    assert dialog.reveal_button is None
    assert dialog.data_explorer_button is None
    assert (
        dialog.findChild(
            QtWidgets.QAbstractButton, "manager_reveal_load_source_path_button"
        )
        is None
    )
    assert (
        dialog.findChild(
            QtWidgets.QAbstractButton, "manager_show_load_source_in_explorer_button"
        )
        is None
    )
    assert path_label.textFormat() == QtCore.Qt.TextFormat.PlainText
    assert path_label.textInteractionFlags() == (
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    assert path_label.cursor().shape() != QtCore.Qt.CursorShape.PointingHandCursor
    edit_button = dialog.findChild(
        QtWidgets.QAbstractButton, "manager_edit_load_source_button"
    )
    assert edit_button is not None
    assert not edit_button.isEnabled()
    assert edit_button.toolTip()
    assert edit_button not in dialog.button_box.buttons()
    details_layout_item = typing.cast("QtWidgets.QVBoxLayout", dialog.layout()).itemAt(
        1
    )
    details_layout = details_layout_item.layout()
    assert isinstance(details_layout, QtWidgets.QGridLayout)
    for index in range(details_layout.count()):
        if details_layout.itemAt(index).widget() is edit_button:
            assert details_layout.getItemPosition(index) == (0, 2, 1, 1)
            break
    else:
        pytest.fail("edit button was not placed in the details grid")


def test_load_source_details_dialog_missing_source_edit_button_requests_edit(
    qtbot,
    tmp_path,
) -> None:
    source_path = tmp_path / "missing.nc"
    dialog = _LoadSourceDetailsDialog(
        _LoadSourceDetails(
            path=source_path,
            loader_label="Loader",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            load_code=None,
        ),
        can_edit_file_load=True,
        edit_file_load_tooltip="Select the current source file.",
    )
    qtbot.addWidget(dialog)

    edit_button = dialog.findChild(
        QtWidgets.QAbstractButton, "manager_edit_load_source_button"
    )
    assert edit_button is not None
    assert edit_button.isEnabled()
    assert edit_button.toolTip()
    assert not dialog.edit_file_load_requested

    edit_button.click()

    assert dialog.edit_file_load_requested
    assert dialog.result() == int(QtWidgets.QDialog.DialogCode.Accepted)


def test_load_source_details_controller_opens_source_in_data_explorer(tmp_path) -> None:
    source_path = tmp_path / "scan.nc"
    shown_paths: list[pathlib.Path] = []
    requested_apps: list[str] = []

    class _FakeExplorer:
        def show_path(self, path: pathlib.Path) -> None:
            shown_paths.append(pathlib.Path(path))

    def _show_standalone_app(key: str) -> _FakeExplorer:
        requested_apps.append(key)
        return _FakeExplorer()

    manager = types.SimpleNamespace(_show_standalone_app=_show_standalone_app)
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )

    controller._show_load_source_in_data_explorer(source_path)

    assert requested_apps == ["explorer"]
    assert shown_paths == [source_path]


def test_details_panel_file_field_info_button_passes_metadata_node_uid(
    qtbot,
    tmp_path,
) -> None:
    source_path = tmp_path / "scan.nc"
    details = _LoadSourceDetails(
        path=source_path,
        loader_label="Loader",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        load_code=None,
    )
    metadata_widget = QtWidgets.QWidget()
    qtbot.addWidget(metadata_widget)
    shown: list[tuple[_LoadSourceDetails, str | None]] = []
    manager = types.SimpleNamespace(
        metadata_details_widget=metadata_widget,
        metadata_details_layout=QtWidgets.QGridLayout(metadata_widget),
        _metadata_detail_labels={},
        _metadata_monospace_font=QtGui.QFont(),
        _metadata_node_uid="node-1",
        _show_load_source_details=lambda details, *, node_uid=None: shown.append(
            (details, node_uid)
        ),
    )
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )

    controller._set_metadata_fields(
        [
            manager_wrapper._MetadataField(
                "File",
                str(source_path),
                details=details,
            )
        ]
    )

    value_label = manager._metadata_detail_labels["File"]
    assert isinstance(value_label, manager_widgets._ElidedValueLabel)
    assert value_label.foregroundRole() != QtGui.QPalette.ColorRole.Link
    assert value_label.cursor().shape() != QtCore.Qt.CursorShape.PointingHandCursor
    assert value_label.textInteractionFlags() == (
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    qtbot.mouseClick(value_label, QtCore.Qt.MouseButton.LeftButton)
    assert not shown

    details_button = metadata_widget.findChild(
        QtWidgets.QToolButton,
        "manager_metadata_file_details_button",
    )
    assert details_button is not None
    assert_nonempty_tooltip(details_button.toolTip())
    assert details_button.accessibleName()
    assert not details_button.icon().isNull()
    assert isinstance(
        details_button,
        _CenteredIconToolButton,
    )
    key_label = typing.cast(
        "QtWidgets.QLabel",
        manager.metadata_details_layout.itemAtPosition(0, 0).widget(),
    )
    assert key_label.alignment() == (
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
    )
    assert value_label.alignment() == (
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
    )
    assert manager.metadata_details_layout.itemAtPosition(0, 2) is None
    file_value_item = manager.metadata_details_layout.itemAtPosition(0, 1)
    assert file_value_item is not None
    file_value_widget = file_value_item.widget()
    assert file_value_widget is not None
    assert details_button.parentWidget() is file_value_widget
    file_value_layout = typing.cast("QtWidgets.QHBoxLayout", file_value_widget.layout())
    assert file_value_layout.itemAt(0).widget() is value_label
    details_button_item = file_value_layout.itemAt(1)
    assert details_button_item.widget() is details_button
    assert details_button_item.alignment() & QtCore.Qt.AlignmentFlag.AlignVCenter
    details_button.click()

    assert shown == [(details, "node-1")]


def test_details_panel_single_line_rows_stay_compact_and_elide(
    qtbot,
    tmp_path,
) -> None:
    source_path = tmp_path / "scan.nc"
    details = _LoadSourceDetails(
        path=source_path,
        loader_label="Loader",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        load_code=None,
    )
    metadata_widget = QtWidgets.QWidget()
    qtbot.addWidget(metadata_widget)
    manager = types.SimpleNamespace(
        metadata_details_widget=metadata_widget,
        metadata_details_layout=QtWidgets.QGridLayout(metadata_widget),
        _metadata_detail_labels={},
        _metadata_monospace_font=QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        ),
        _metadata_node_uid="node-1",
        _show_load_source_details=lambda _details, *, node_uid=None: None,
    )
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )
    long_time = "2024-01-02 03:04:05 Pacific Daylight Time (-0700)"

    controller._set_metadata_fields(
        [
            manager_wrapper._MetadataField("Kind", "ImageTool"),
            manager_wrapper._MetadataField("Added", long_time, monospace=True),
            manager_wrapper._MetadataField(
                "File",
                str(source_path),
                monospace=True,
                details=details,
            ),
            manager_wrapper._MetadataField("Inputs", "first\nsecond", wrap=True),
        ]
    )

    kind_label = manager._metadata_detail_labels["Kind"]
    added_label = manager._metadata_detail_labels["Added"]
    file_label = manager._metadata_detail_labels["File"]
    inputs_label = manager._metadata_detail_labels["Inputs"]
    assert isinstance(kind_label, manager_widgets._ElidedValueLabel)
    assert isinstance(added_label, manager_widgets._ElidedValueLabel)
    assert isinstance(file_label, manager_widgets._ElidedValueLabel)
    assert not isinstance(inputs_label, manager_widgets._ElidedValueLabel)
    assert added_label.full_text == long_time
    assert added_label.sizePolicy().horizontalPolicy() == (
        QtWidgets.QSizePolicy.Policy.Ignored
    )
    assert added_label.sizeHint().width() < (
        added_label.fontMetrics().horizontalAdvance(long_time)
    )

    layout = manager.metadata_details_layout
    assert layout.rowMinimumHeight(0) == 0
    assert layout.rowMinimumHeight(1) == 0
    assert layout.rowMinimumHeight(2) == 0
    assert layout.rowMinimumHeight(3) == 0
    details_button = metadata_widget.findChild(
        QtWidgets.QToolButton,
        "manager_metadata_file_details_button",
    )
    assert details_button is not None
    file_key_label = typing.cast(
        "QtWidgets.QLabel",
        layout.itemAtPosition(2, 0).widget(),
    )
    button_style = details_button.style() or QtWidgets.QApplication.style()
    small_icon_size = (
        button_style.pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_SmallIconSize,
            None,
            details_button,
        )
        if button_style is not None
        else file_label.fontMetrics().height()
    )
    compact_row_height = max(
        file_key_label.sizeHint().height(),
        file_label.sizeHint().height(),
        small_icon_size,
    )
    assert details_button.minimumHeight() == compact_row_height
    assert details_button.maximumHeight() == compact_row_height
    assert details_button.iconSize() == QtCore.QSize(small_icon_size, small_icon_size)
    assert layout.itemAtPosition(2, 2) is None
    file_value_widget = typing.cast(
        "QtWidgets.QWidget",
        layout.itemAtPosition(2, 1).widget(),
    )
    assert details_button.parentWidget() is file_value_widget
    metadata_widget.resize(360, metadata_widget.sizeHint().height())
    metadata_widget.show()
    QtWidgets.QApplication.processEvents()
    button_rect = details_button.geometry()
    assert layout.cellRect(1, 1).right() == layout.contentsRect().right()
    assert button_rect.right() < file_value_widget.rect().right()
    inputs_key_label = typing.cast(
        "QtWidgets.QLabel",
        layout.itemAtPosition(3, 0).widget(),
    )
    assert inputs_key_label.alignment() == (
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
    )
    assert inputs_label.alignment() == (
        QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
    )


def test_centered_icon_tool_button_centers_visible_icon_with_stylesheet(
    qtbot,
) -> None:
    pixmap = QtGui.QPixmap(10, 6)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.fillRect(QtCore.QRect(0, 0, 4, 6), QtGui.QColor(255, 0, 0))
    painter.end()

    button = _CenteredIconToolButton()
    qtbot.addWidget(button)
    button.setIcon(QtGui.QIcon(pixmap))
    button.setIconSize(QtCore.QSize(10, 6))
    button.setStyleSheet(
        "QToolButton { padding-left: 9px; padding-top: 4px; "
        "padding-right: 0px; padding-bottom: 0px; }"
    )
    button.resize(28, 22)
    button.show()

    image = button.grab().toImage()
    red_pixels: list[tuple[int, int]] = []
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.red() > 220 and color.green() < 40 and color.blue() < 40:
                red_pixels.append((x, y))

    assert red_pixels
    min_x = min(x for x, _y in red_pixels)
    max_x = max(x for x, _y in red_pixels)
    min_y = min(y for _x, y in red_pixels)
    max_y = max(y for _x, y in red_pixels)
    assert abs(((min_x + max_x) / 2) - ((image.width() - 1) / 2)) <= 0.5
    assert abs(((min_y + max_y) / 2) - ((image.height() - 1) / 2)) <= 0.5


def test_centered_icon_tool_button_visible_rect_fallbacks() -> None:
    transparent = QtGui.QPixmap(8, 6)
    transparent.fill(QtCore.Qt.GlobalColor.transparent)
    transparent.setDevicePixelRatio(0.0)

    rect = _CenteredIconToolButton._visible_pixmap_rect(transparent)

    assert rect == QtCore.QRectF(0.0, 0.0, 8.0, 6.0)


def test_centered_icon_tool_button_paint_skips_missing_icon(qtbot) -> None:
    button = _CenteredIconToolButton()
    qtbot.addWidget(button)
    button.resize(24, 24)
    button.show()

    button.paintEvent(None)


def test_load_source_details_controller_opens_file_load_edit_after_dialog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_path = tmp_path / "missing.nc"
    details = _LoadSourceDetails(
        path=source_path,
        loader_label="Loader",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        load_code=None,
    )
    node = types.SimpleNamespace(uid="node-1")
    events: list[tuple[str, object, object | None]] = []

    class _FakeDialog:
        edit_file_load_requested = True

        def __init__(
            self,
            details: _LoadSourceDetails,
            parent: QtWidgets.QWidget,
            *,
            show_in_data_explorer: Callable[[pathlib.Path], None],
            can_edit_file_load: bool,
            edit_file_load_tooltip: str,
        ) -> None:
            del parent, show_in_data_explorer
            events.append(("dialog", details, can_edit_file_load))
            events.append(("tooltip", edit_file_load_tooltip, None))

        def exec(self) -> int:
            events.append(("exec", source_path, None))
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    class _FakeProvenanceEditController:
        def can_edit_file_load_source(
            self,
            edit_node: object,
            path: pathlib.Path,
        ) -> tuple[bool, str]:
            events.append(("can", edit_node, path))
            return True, "Select the current source file."

        def edit_file_load_source(
            self,
            edit_node: object,
            path: pathlib.Path,
        ) -> None:
            events.append(("edit", edit_node, path))

    monkeypatch.setattr(
        manager_details_panel,
        "_LoadSourceDetailsDialog",
        _FakeDialog,
    )
    manager = types.SimpleNamespace(
        _metadata_node_uid="node-1",
        _tool_graph=types.SimpleNamespace(nodes={"node-1": node}),
        _provenance_edit_controller=_FakeProvenanceEditController(),
        _show_standalone_app=lambda _key: None,
    )
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )

    controller._show_load_source_details(details)

    assert events == [
        ("can", node, source_path),
        ("dialog", details, True),
        ("tooltip", "Select the current source file.", None),
        ("exec", source_path, None),
        ("edit", node, source_path),
    ]


def test_load_source_details_controller_skips_edit_when_node_disappears(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_path = tmp_path / "missing.nc"
    details = _LoadSourceDetails(
        path=source_path,
        loader_label="Loader",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        load_code=None,
    )
    node = types.SimpleNamespace(uid="node-1")
    events: list[str] = []

    class _FakeDialog:
        edit_file_load_requested = True

        def __init__(
            self,
            _details: _LoadSourceDetails,
            _parent: QtWidgets.QWidget,
            *,
            show_in_data_explorer: Callable[[pathlib.Path], None],
            can_edit_file_load: bool,
            edit_file_load_tooltip: str,
        ) -> None:
            del show_in_data_explorer, can_edit_file_load, edit_file_load_tooltip

        def exec(self) -> int:
            manager._tool_graph.nodes.clear()
            events.append("exec")
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    class _FakeProvenanceEditController:
        def can_edit_file_load_source(
            self,
            edit_node: object,
            path: pathlib.Path,
        ) -> tuple[bool, str]:
            assert edit_node is node
            assert path == source_path
            return True, "Select the current source file."

        def edit_file_load_source(
            self,
            _edit_node: object,
            _path: pathlib.Path,
        ) -> None:
            pytest.fail("edit should not run for a missing node")

    monkeypatch.setattr(
        manager_details_panel,
        "_LoadSourceDetailsDialog",
        _FakeDialog,
    )
    manager = types.SimpleNamespace(
        _metadata_node_uid="node-1",
        _tool_graph=types.SimpleNamespace(nodes={"node-1": node}),
        _provenance_edit_controller=_FakeProvenanceEditController(),
        _show_standalone_app=lambda _key: None,
    )
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )

    controller._show_load_source_details(details)

    assert events == ["exec"]


def test_details_panel_selected_derivation_row_rejects_unexpected_item_data(
    qtbot,
) -> None:
    derivation_list = QtWidgets.QListWidget()
    qtbot.addWidget(derivation_list)
    item = QtWidgets.QListWidgetItem("row")
    item.setData(manager_details_panel._METADATA_DERIVATION_ROW_ROLE, object())
    derivation_list.addItem(item)
    item.setSelected(True)

    manager = types.SimpleNamespace(metadata_derivation_list=derivation_list)
    controller = _DetailsPanelController(
        typing.cast("manager_mainwindow.ImageToolManager", manager)
    )
    manager._selected_derivation_items = controller._selected_derivation_items

    assert controller._selected_derivation_row() is None


def test_workspace_properties_dialog_actions(qtbot, monkeypatch, tmp_path) -> None:
    workspace_path = (tmp_path / "workspace.itws").resolve()
    workspace_path.write_bytes(b"itws")
    copied: list[str] = []
    revealed: list[pathlib.Path] = []

    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda content: copied.append(str(content)) or str(content),
    )
    monkeypatch.setattr(
        erlab.utils.misc,
        "open_in_file_manager",
        lambda path: revealed.append(pathlib.Path(path)),
    )

    dialog = _WorkspacePropertiesDialog(
        workspace_path,
        state=_WorkspacePropertiesState(
            is_modified=True,
            top_level_window_count=3,
        ),
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog.findChildren(QtWidgets.QLineEdit)
    path_label = dialog.findChild(
        QtWidgets.QLabel, "manager_workspace_path_value_label"
    )
    assert path_label is not None
    assert path_label.text() == str(workspace_path)
    assert path_label.toolTip() == str(workspace_path)
    assert path_label.wordWrap()
    assert path_label.textInteractionFlags() == (
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )
    assert dialog.minimumSize() == dialog.maximumSize()
    assert dialog.height() == dialog.minimumSizeHint().height()
    assert dialog.height() <= dialog.sizeHint().height()
    assert dialog.value_labels["open_windows"].text() == "3"
    assert dialog.value_labels["size"].text()
    assert dialog.value_labels["modified"].text()
    assert (
        dialog.findChild(
            QtWidgets.QPlainTextEdit, "manager_workspace_dirty_details_edit"
        )
        is None
    )

    assert dialog.copy_path_button is not None
    assert dialog.reveal_button is not None

    dialog.copy_path_button.click()
    dialog.reveal_button.click()

    assert copied == [str(workspace_path)]
    assert revealed == [workspace_path]


def test_workspace_properties_dialog_without_associated_file(qtbot) -> None:
    dialog = _WorkspacePropertiesDialog(
        None,
        state=_WorkspacePropertiesState(is_modified=False, top_level_window_count=0),
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog.findChildren(QtWidgets.QLineEdit)
    assert (
        dialog.findChild(QtWidgets.QLabel, "manager_workspace_path_value_label") is None
    )
    assert dialog.value_labels["open_windows"].text() == "0"
    assert dialog.copy_path_button is None
    assert dialog.reveal_button is None
    assert (
        dialog.findChild(
            QtWidgets.QAbstractButton, "manager_copy_workspace_path_button"
        )
        is None
    )
    assert (
        dialog.findChild(
            QtWidgets.QAbstractButton, "manager_reveal_workspace_path_button"
        )
        is None
    )
    dialog._copy_path()
    dialog._reveal_path()


def test_workspace_properties_dialog_file_detail_branches(
    qtbot, monkeypatch, tmp_path
) -> None:
    missing_workspace = (tmp_path / "missing.itws").resolve()
    state = _WorkspacePropertiesState(is_modified=False, top_level_window_count=1)
    dialog = _WorkspacePropertiesDialog(
        missing_workspace,
        state=state,
    )
    qtbot.addWidget(dialog)

    assert dialog.value_labels["size"].text() == "File not found"
    assert dialog.value_labels["modified"].text() == "File not found"
    assert dialog.value_labels["open_windows"].text() == "1"
    assert dialog._status_text(missing_workspace, state) == "Associated file"

    assert manager_widgets._workspace_file_type_text(None) == (
        "Unsaved ImageTool workspace"
    )
    assert manager_widgets._workspace_file_type_text(tmp_path / "workspace.itws") == (
        "ImageTool Workspace (.itws)"
    )
    assert manager_widgets._workspace_file_type_text(tmp_path / "workspace.h5") == (
        "HDF5 file (.h5)"
    )
    assert manager_widgets._workspace_file_type_text(tmp_path / "notes.txt") == (
        "TXT file"
    )
    assert manager_widgets._workspace_file_type_text(tmp_path / "README") == "File"

    assert manager_widgets._format_workspace_file_size(1) == "1 byte"
    assert manager_widgets._format_workspace_file_size(999) == "999 bytes"
    assert manager_widgets._format_workspace_file_size(1_000).startswith("1.00 KB")
    assert manager_widgets._format_workspace_file_size(10**16).startswith("10000.00 PB")

    monkeypatch.setattr(manager_mainwindow.sys, "platform", "darwin")
    assert manager_widgets._workspace_file_manager_action_text() == ("Reveal in Finder")
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "win32")
    assert manager_widgets._workspace_file_manager_action_text() == (
        "Reveal in File Explorer"
    )
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "linux")
    assert manager_widgets._workspace_file_manager_action_text() == (
        "Open Containing Folder"
    )


def test_manager_notes_editor_actions(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    copied: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda content: copied.append(str(content)) or str(content),
    )

    with manager_context() as manager:
        manager.show()
        manager.resize(640, 700)
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        wrapper = manager._tool_graph.root_wrappers[0]
        manager._workspace_controller._mark_workspace_clean()

        select_tools(manager, [0])
        manager._update_info()
        assert manager.inspector_tabs.count() == 3
        assert manager.metadata_group.parentWidget() is manager.right_splitter
        assert (
            manager.metadata_details_widget.parentWidget()
            is manager.metadata_details_page
        )
        assert (
            manager.metadata_derivation_list.parentWidget()
            is manager.metadata_provenance_page
        )
        assert manager.metadata_details_page.layout().contentsMargins().left() > 0

        manager.edit_note_action.trigger()
        assert manager.inspector_tabs.currentWidget() is manager.notes_page
        assert isinstance(manager.notes_title_label, manager_widgets._ElidedValueLabel)
        assert manager.notes_title_label.minimumSizeHint().width() == 0
        assert manager.notes_title_label.full_text == wrapper.display_text
        manager.notes_editor.setPlainText("root intent\nsecond line")

        qtbot.wait_until(
            lambda: wrapper.note == "root intent\nsecond line",
            timeout=1500,
        )
        assert manager._workspace_state.dirty_state == {wrapper.uid}
        assert manager._workspace_state.dirty_data == set()

        manager.copy_note_action.trigger()
        assert copied == ["root intent\nsecond line"]

        manager.clear_note_action.trigger()
        assert wrapper.note == ""

        compact_notes: list[str] = []
        monkeypatch.setattr(
            manager._workspace_controller,
            "compact_workspace",
            lambda: compact_notes.append(wrapper.note) or True,
        )
        manager.notes_editor.setPlainText("pending compact note")
        manager._note_commit_timer.stop()
        assert manager.compact_workspace()
        assert compact_notes == ["pending compact note"]

        load_notes: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            manager._workspace_controller,
            "load",
            lambda *, native=True: load_notes.append((wrapper.note, native)) or True,
        )
        manager.notes_editor.setPlainText("pending load note")
        manager._note_commit_timer.stop()
        assert manager.load(native=False)
        assert load_notes == [("pending load note", False)]

        manager.tree_view.clearSelection()
        manager._update_info()
        assert manager.notes_title_label.full_text == ""
        assert manager.notes_title_label.toolTip() == ""
        assert manager.notes_title_label.accessibleName() == ""


def test_manager_notes_controller_guard_branches(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    copied: list[str] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda content: copied.append(str(content)) or str(content),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        wrapper = manager._tool_graph.root_wrappers[0]

        select_tools(manager, [0])
        manager.edit_note_action.trigger()
        manager.notes_editor.setPlainText(wrapper.note)
        manager._updating_note_editor = True
        try:
            manager._commit_note_editor()
        finally:
            manager._updating_note_editor = False

        manager.tree_view.clearSelection()
        manager._update_info()
        manager.edit_selected_note()
        manager.copy_selected_note()
        manager.clear_selected_note()
        assert copied == []

        monkeypatch.setattr(
            manager._details_panel,
            "_notes_ui_available",
            lambda: False,
        )
        manager._details_panel._update_note_actions()
        manager._schedule_note_commit()
        manager._commit_note_editor()
        manager.edit_selected_note()
        manager.copy_selected_note()
        manager.clear_selected_note()


def test_manager_inspector_tabs_fallback_between_pages(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        def enable_inspector_tabs() -> None:
            for page in (
                manager.metadata_details_page,
                manager.metadata_provenance_page,
                manager.notes_page,
            ):
                manager.inspector_tabs.setTabEnabled(
                    manager.inspector_tabs.indexOf(page), True
                )

        manager._set_metadata_fields([])
        manager.metadata_derivation_list.clear()
        manager.metadata_derivation_list.addTopLevelItem(QtWidgets.QTreeWidgetItem())
        manager._notes_node_uid = None
        enable_inspector_tabs()
        manager.inspector_tabs.setCurrentWidget(manager.metadata_details_page)
        manager._update_metadata_pane()
        assert (
            manager.inspector_tabs.currentWidget() is manager.metadata_provenance_page
        )

        manager.metadata_derivation_list.clear()
        manager._notes_node_uid = None
        enable_inspector_tabs()
        manager.inspector_tabs.setCurrentWidget(manager.notes_page)
        manager._update_metadata_pane()
        assert manager.inspector_tabs.currentWidget() is manager.metadata_details_page

        manager.metadata_derivation_list.clear()
        manager._notes_node_uid = "note-target"
        enable_inspector_tabs()
        manager.inspector_tabs.setCurrentWidget(manager.metadata_details_page)
        manager._update_metadata_pane()
        assert manager.inspector_tabs.currentWidget() is manager.notes_page

        manager._set_metadata_fields(
            [manager_wrapper._MetadataField("Kind", "ImageTool")]
        )
        manager._notes_node_uid = None
        enable_inspector_tabs()
        manager.inspector_tabs.setCurrentWidget(manager.metadata_provenance_page)
        manager._update_metadata_pane()
        assert manager.inspector_tabs.currentWidget() is manager.metadata_details_page

        manager._set_metadata_fields([])
        manager._notes_node_uid = "note-target"
        enable_inspector_tabs()
        manager.inspector_tabs.setCurrentWidget(manager.metadata_provenance_page)
        manager._update_metadata_pane()
        assert manager.inspector_tabs.currentWidget() is manager.notes_page

        manager._set_metadata_fields(
            [manager_wrapper._MetadataField("Kind", "ImageTool")]
        )
        manager._notes_node_uid = None
        enable_inspector_tabs()
        manager.inspector_tabs.setCurrentWidget(manager.notes_page)
        manager._update_metadata_pane()
        assert manager.inspector_tabs.currentWidget() is manager.metadata_details_page

        manager._set_metadata_fields([])
        manager.metadata_derivation_list.clear()
        manager.metadata_derivation_list.addTopLevelItem(QtWidgets.QTreeWidgetItem())
        manager._notes_node_uid = None
        enable_inspector_tabs()
        manager.inspector_tabs.setCurrentWidget(manager.notes_page)
        manager._update_metadata_pane()
        assert (
            manager.inspector_tabs.currentWidget() is manager.metadata_provenance_page
        )


@pytest.mark.parametrize("use_socket", [False, True], ids=["no_socket", "socket"])
def test_manager(
    qtbot,
    accept_dialog,
    test_data,
    use_socket,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context(use_socket=use_socket) as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        logger.info("Manager is running, adding test data")
        test_data.qshow(manager=True)

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        logger.info("Confirmed tool is added, checking data")
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        logger.info("Checking data retrieval via fetch")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fetch, 0)

            # Let the GUI thread keep processing events while waiting
            qtbot.waitUntil(lambda: fut.done(), timeout=10000)

            result = fut.result()

        xr.testing.assert_identical(result, test_data)

        logger.info("Confirmed fetch works, adding more tools")
        # Add two tools
        for tool in itool(
            [test_data, test_data], link=False, execute=False, manager=False
        ):
            tool.move_to_manager()

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Linking
        select_tools(manager, [1, 2])
        manager.link_selected()
        manager.tree_view.refresh(None)

        # Unlinking one unlinks both
        select_tools(manager, [1])
        manager.unlink_selected()
        manager.tree_view.refresh(None)
        assert not manager.get_imagetool(1).slicer_area.is_linked
        assert not manager.get_imagetool(2).slicer_area.is_linked

        # Linking again
        select_tools(manager, [1, 2])
        manager.link_selected()
        manager.tree_view.refresh(None)
        assert manager.get_imagetool(1).slicer_area.is_linked
        assert manager.get_imagetool(2).slicer_area.is_linked

        # Toggle visibility
        geometry = manager.get_imagetool(1).geometry()
        manager._tool_graph.root_wrappers[1].hide()
        assert not manager.get_imagetool(1).isVisible()
        manager._tool_graph.root_wrappers[1].show()
        assert manager.get_imagetool(1).geometry() == geometry

        # Removing tool
        manager.remove_imagetool(0)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Batch renaming
        select_tools(manager, [1, 2])

        def _handle_renaming(dialog: _RenameDialog):
            dialog._new_name_lines[1].setText("new_name_1")
            dialog._new_name_lines[2].setText("new_name_2")

        accept_dialog(manager.rename_action.trigger, pre_call=_handle_renaming)
        assert manager._tool_graph.root_wrappers[1].name == "new_name_1"
        assert manager._tool_graph.root_wrappers[2].name == "new_name_2"

        # Rename single
        select_tools(manager, [2], deselect=True)
        select_tools(manager, [1])
        manager.rename_action.trigger()

        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("new_name_1_single")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: manager._tool_graph.root_wrappers[1].name == "new_name_1_single",
            timeout=5000,
        )

        # Select single tool
        select_tools(manager, [1])

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager._update_info()

        # Batch show/hide
        select_tools(manager, [1, 2])
        manager.hide_action.trigger()
        manager.show_action.trigger()

        assert manager.get_imagetool(1).isVisible()
        assert manager.get_imagetool(2).isVisible()

        # Add third tool
        xr.concat(
            [
                manager.get_imagetool(1).slicer_area._data,
                manager.get_imagetool(2).slicer_area._data,
            ],
            "concat_dim",
        ).qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_tools(manager, [1, 2, 3])
        manager._update_info()

        # Show goldtool
        logger.info("Opening goldtool")
        manager.get_imagetool(3).slicer_area.images[2].open_in_goldtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[3]._childtools) == 1,
            timeout=5000,
        )
        assert isinstance(
            next(iter(manager._tool_graph.root_wrappers[3]._childtools.values())),
            GoldTool,
        )
        logger.info("Confirmed goldtool is added")

        # Trigger paint event
        manager.tree_view.expandAll()

        # Test rename goldtool
        goldtool_uid: str = manager._tool_graph.root_wrappers[3]._childtool_indices[0]

        # Bring manager to top
        manager.tree_view.clearSelection()
        bring_manager_to_top(qtbot, manager)
        select_child_tool(manager, goldtool_uid)

        manager._update_actions()
        assert manager.rename_action.isEnabled()
        manager.rename_action.trigger()
        qtbot.wait_until(
            lambda: (
                manager.tree_view.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("new_goldtool_name")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: (
                next(
                    iter(manager._tool_graph.root_wrappers[3]._childtools.values())
                )._tool_display_name
                == "new_goldtool_name"
            ),
            timeout=5000,
        )

        # Close goldtool
        logger.info("Closing goldtool")
        manager._remove_childtool(
            next(iter(manager._tool_graph.root_wrappers[3]._childtools.keys()))
        )

        # Show dtool
        logger.info("Opening dtool")
        manager.get_imagetool(3).slicer_area.images[2].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[3]._childtools) == 1,
            timeout=5000,
        )
        assert isinstance(
            next(iter(manager._tool_graph.root_wrappers[3]._childtools.values())),
            DerivativeTool,
        )
        logger.info("Confirmed dtool is added")
        manager.tree_view.expandAll()
        tool_uid: str = manager._tool_graph.root_wrappers[3]._childtool_indices[0]

        # Show dtool
        manager.show_childtool(tool_uid)

        # Tool and parent
        logger.info("Checking parent and childtool retrieval")
        tool, idx = manager._get_childtool_and_parent(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert idx == 3

        # Check dtool info printing
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_child_tool(manager, tool_uid)
        manager._update_info(uid=tool_uid)

        # Duplicate dtool
        logger.info("Duplicating dtool")
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_child_tool(manager, tool_uid)
        manager.duplicate_selected()
        manager.tree_view.refresh(None)

        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[3]._childtools) == 2,
            timeout=5000,
        )

        # Check calling invalid indices
        logger.info("Checking invalid index handling")
        parent_qindex = manager.tree_view._model._row_index(3)
        assert manager.tree_view._model.index(1, 0, parent_qindex).isValid()
        assert not manager.tree_view._model.index(4, 0, parent_qindex).isValid()

        valid_but_wrong_pointer_type = manager.tree_view._model.createIndex(
            parent_qindex.row(), parent_qindex.column(), "invalid data"
        )
        assert not manager.tree_view._model.index(
            1, 0, valid_but_wrong_pointer_type
        ).isValid()

        # Close dtools
        logger.info("Closing dtools")
        for uid in list(manager._tool_graph.root_wrappers[3]._childtools.keys()):
            manager._remove_childtool(uid)

        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[3]._childtools) == 0,
            timeout=5000,
        )
        logger.info("Confirmed dtools are removed")

        # Bring manager to top
        logger.info("Testing mouse events")
        with qtbot.waitExposed(manager):
            manager.hide_all()  # Prevent windows from obstructing the manager
            manager.activateWindow()
            manager.raise_()
            manager.preview_action.setChecked(True)

        # Test hover-preview delegate painting without moving the shared display cursor.
        first_index = manager.tree_view.model().index(0, 0)
        option = delegate._option_for_index(manager.tree_view, first_index)
        option.state |= QtWidgets.QStyle.StateFlag.State_MouseOver
        pixmap = QtGui.QPixmap(manager.tree_view.viewport().size())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        try:
            delegate.paint(painter, option, first_index)
        finally:
            painter.end()
        delegate.eventFilter(
            manager.tree_view.viewport(),
            QtCore.QEvent(QtCore.QEvent.Type.Leave),
        )

        # Remove third tool
        select_tools(manager, [3])
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Test concatenate
        concat_data = xr.concat(
            [
                manager.get_imagetool(1).slicer_area._data,
                manager.get_imagetool(2).slicer_area._data,
            ],
            "concat_dim",
        )
        select_tools(manager, [1, 2])
        accept_dialog(manager.concat_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(3).slicer_area._data, concat_data
        )

        # Test concatenate (remove originals)
        select_tools(manager, [1, 2])

        def _handle_concat(dialog: _ConcatDialog):
            dialog._sources_combo.setCurrentIndex(
                dialog._sources_combo.findData(_ConcatDialog._SOURCES_REMOVE)
            )

        accept_dialog(manager.concat_action.trigger, pre_call=_handle_concat)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(4).slicer_area._data, concat_data
        )

        # Remove all selected
        select_tools(manager, [3, 4])
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        # Show about dialog
        accept_dialog(manager.about)


def test_remove_from_window_shortcut(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)

        with qtbot.waitExposed(tool):
            tool.activateWindow()
            tool.raise_()
            tool.setFocus()

        assert tool.remove_act.isVisible()

        delete_key = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete)
        assert (
            tool.remove_act.shortcut().matches(delete_key)
            == QtGui.QKeySequence.SequenceMatch.ExactMatch
        )
        assert (
            tool.remove_act.shortcutContext()
            == QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )

        accept_dialog(tool.remove_act.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_remove_childtool_direct_removal(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()

        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        uid, _child = next(iter(wrapper._childtools.items()))

        manager._remove_childtool(uid)
        qtbot.wait_until(lambda: uid not in wrapper._childtools, timeout=5000)


def test_shutdown_bulk_remove_skips_final_ui_refresh(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[str] = []
        previous_closing = manager._workspace_state.closing_document
        monkeypatch.setattr(
            manager, "_cleanup_linkers", lambda: calls.append("cleanup")
        )
        monkeypatch.setattr(manager, "_update_actions", lambda: calls.append("actions"))
        monkeypatch.setattr(manager, "_update_info", lambda: calls.append("info"))

        try:
            manager._workspace_state.closing_document = True
            with manager._bulk_remove_context():
                assert not manager.updatesEnabled()
                assert not manager.tree_view.updatesEnabled()
        finally:
            manager._workspace_state.closing_document = previous_closing

        assert manager.updatesEnabled()
        assert manager.tree_view.updatesEnabled()
        assert calls == []


def test_shutdown_remove_all_tools_skips_teardown_ui_refresh(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="data",
    )
    with manager_context() as manager:
        root_tool = itool(data, manager=False, execute=False)
        root_index = manager.add_imagetool(root_tool, show=False)
        figure_uid = manager.add_figuretool(FigureComposerTool(data), show=False)
        calls: list[str] = []

        monkeypatch.setattr(
            manager.tree_view,
            "childtool_removed",
            lambda _uid: calls.append("childtool_removed"),
        )
        monkeypatch.setattr(
            manager._figure_collection,
            "sync",
            lambda **_kwargs: calls.append("figures"),
        )
        monkeypatch.setattr(manager, "_update_actions", lambda: calls.append("actions"))
        monkeypatch.setattr(
            manager,
            "_refresh_dependency_dependents",
            lambda _uid: calls.append("dependents"),
        )
        monkeypatch.setattr(
            manager._figure_workflows,
            "_refresh_figure_source_controls",
            lambda: calls.append("source_controls"),
        )

        previous_closing = manager._workspace_state.closing_document
        try:
            manager._workspace_state.closing_document = True
            manager.remove_all_tools()
        finally:
            manager._workspace_state.closing_document = previous_closing

        assert root_index not in manager._tool_graph.root_wrappers
        assert figure_uid not in manager._tool_graph.nodes
        assert calls == []


def test_manager_append_reports_post_alias_plot_slices_error(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("source")
    existing = _batch_data("existing")
    with manager_context() as manager:
        source_tool = itool(source, manager=False, execute=False)
        manager.add_imagetool(source_tool, show=False)
        figure_tool = FigureComposerTool(
            existing,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(),
                sources=(FigureSourceState(name="existing"),),
                operations=(),
                primary_source="existing",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        calls: list[tuple[str, ...]] = []

        def image_operations(
            _targets: tuple[int | str, ...], source_names: tuple[str, ...]
        ) -> tuple[FigureOperationState, ...]:
            calls.append(source_names)
            if len(calls) == 2:
                raise FigureComposerPlotSlicesSelectionError("source")
            return (
                FigureOperationState.plot_array(label="source", source=source_names[0]),
            )

        errors: list[FigureComposerPlotSlicesSelectionError] = []
        monkeypatch.setattr(
            manager._figure_workflows,
            "_figure_operations_from_image_targets",
            image_operations,
        )
        monkeypatch.setattr(
            manager._figure_workflows,
            "_show_figure_plot_slices_selection_error",
            errors.append,
        )

        assert not manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )
        assert len(calls) == 2
        assert len(errors) == 1
        assert figure_tool.tool_status.operations == ()


def test_manager_append_partial_source_result_shows_figure_without_new_step(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(np.arange(4.0), dims=("x",), name="source")
    existing = xr.DataArray(np.arange(4.0), dims=("x",), name="existing")
    with manager_context() as manager:
        source_tool = itool(source, manager=False, execute=False)
        manager.add_imagetool(source_tool, show=False)
        figure_tool = FigureComposerTool(
            existing,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(),
                sources=(FigureSourceState(name="existing"),),
                operations=(),
                primary_source="existing",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        node = manager._child_node(figure_uid)
        source_name = manager._script_input_name_for_node(manager._node_for_target(0))
        shown: list[str] = []
        monkeypatch.setattr(
            figure_tool,
            "add_sources",
            lambda _sources, _source_data: FigureSourceAddResult(
                added=((source_name, source_name),),
                skipped=(("rejected", "rejected"),),
            ),
        )
        monkeypatch.setattr(node, "show", lambda: shown.append(figure_uid))

        assert manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            operation=FigureOperationState.line(label="line", source=source_name),
            show=True,
        )
        assert shown == [figure_uid]
        assert figure_tool.tool_status.operations == ()


def test_manager_append_reuses_short_axes_id_selection_for_all_operations() -> None:
    operations = (
        FigureOperationState.plot_array(label="first", source="first"),
        FigureOperationState.plot_array(label="second", source="second"),
    )
    selection = FigureAxesSelectionState(axes_ids=("only",))

    mapped = _seeding._operations_with_append_axes(operations, selection)

    assert [operation.axes for operation in mapped] == [selection, selection]


def test_manager_figure_source_picker_skips_stale_rows_and_deduplicates_targets(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(np.arange(4.0), dims=("x",), name="source")
    with manager_context() as manager:
        source_tool = itool(source, manager=False, execute=False)
        manager.add_imagetool(source_tool, show=False)
        root = manager._tool_graph.root_wrappers[0]
        figure_uid = manager.add_figuretool(FigureComposerTool(source), show=False)
        original_children = list(root._childtool_indices)
        root._childtool_indices.extend(("missing-child", figure_uid))
        try:
            with monkeypatch.context() as context:
                context.setattr(
                    manager._tool_graph,
                    "root_indices_for_workspace",
                    lambda: (999, 0),
                )
                dialog = _dialogs._FigureSourcePickerDialog(manager)
                qtbot.addWidget(dialog)
                assert dialog.tree.topLevelItemCount() == 1
                root_item = dialog.tree.topLevelItem(0)
                assert root_item is not None
                assert root_item.childCount() == 0
        finally:
            root._childtool_indices[:] = original_children

        assert (
            manager._figure_workflows._figure_source_uid_for_target("missing") is None
        )
        assert manager._figure_workflows._figure_imagetool_targets(
            (0, root.uid, "missing", figure_uid)
        ) == (0,)

        with monkeypatch.context() as context:
            context.setattr(
                manager,
                "_selected_imagetool_targets",
                lambda: (0, root.uid, "missing", figure_uid),
            )
            assert manager._figure_workflows._selected_figure_source_uids() == (
                root.uid,
            )


def test_remove_child_imagetool_remove_action(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = typing.cast(
            "erlab.interactive.imagetool.ImageTool",
            itool(_batch_data("child"), manager=False, execute=False),
        )
        child_uid = manager.add_imagetool_child(child_tool, 0, show=True, activate=True)
        child = manager.get_imagetool(child_uid)

        with qtbot.waitExposed(child):
            child.activateWindow()
            child.raise_()
            child.setFocus()

        assert child.remove_act.isVisible()

        accept_dialog(child.remove_act.trigger)
        qtbot.wait_until(
            lambda: child_uid not in manager._tool_graph.nodes, timeout=5000
        )


def test_manager_childtool_type_badge_only_for_tool_windows(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        named_data = test_data.rename("source")
        itool(named_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        delegate = typing.cast(
            "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
        )
        parent = manager._tool_graph.root_wrappers[0]
        parent_tool = manager.get_imagetool(0)
        root_index = model._row_index(0)
        assert root_index.data(_TOOL_TYPE_ROLE) is None

        parent_tool.slicer_area.images[0].open_in_new_window()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 1, timeout=5000)
        image_uid = parent._childtool_indices[0]
        image_node = manager._child_node(image_uid)
        image_index = model._row_index(image_uid)
        assert image_node.is_imagetool
        assert image_index.data(_TOOL_TYPE_ROLE) is None

        option = QtWidgets.QStyleOptionViewItem()
        option.rect = QtCore.QRect(0, 0, 360, 25)
        option.font = manager.tree_view.font()
        option.palette = manager.tree_view.palette()
        assert delegate._compute_tool_type_info(option, image_node) == (
            None,
            None,
            None,
        )

        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(parent._childtool_indices) == 2, timeout=5000)
        tool_uid = next(uid for uid in parent._childtool_indices if uid != image_uid)
        tool_node = manager._child_node(tool_uid)
        tool = manager.get_childtool(tool_uid)
        tool_index = model._row_index(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert tool_index.data(_TOOL_TYPE_ROLE) == tool.tool_name
        assert tool_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == (
            tool._tool_display_name
        )
        assert tool_index.data(QtCore.Qt.ItemDataRole.EditRole) == (
            tool._tool_display_name
        )

        type_rect, type_text, _ = delegate._compute_tool_type_info(option, tool_node)
        assert type_rect is not None
        assert type_text == tool.tool_name

        tooltip_text = None

        def _show_tooltip(*args, **kwargs) -> None:
            nonlocal tooltip_text
            tooltip_text = args[1]

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", _show_tooltip)
        help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            type_rect.center(),
            manager.tree_view.viewport().mapToGlobal(type_rect.center()),
        )
        assert delegate.helpEvent(help_event, manager.tree_view, option, tool_index)
        assert_nonempty_tooltip(tooltip_text)

        manager.tree_view.expand(root_index)
        actual_option = QtWidgets.QStyleOptionViewItem()
        delegate.initStyleOption(actual_option, tool_index)
        actual_option.rect = manager.tree_view.visualRect(tool_index)
        actual_type_rect, _, _ = delegate._compute_tool_type_info(
            actual_option, tool_node
        )
        assert actual_type_rect is not None
        show_calls: list[str] = []
        monkeypatch.setattr(
            manager, "show_childtool", lambda uid: show_calls.append(uid)
        )
        click_tree_view_pos(manager.tree_view, actual_type_rect.center())
        assert show_calls == [tool_uid]

        editor = QtWidgets.QLineEdit(manager.tree_view.viewport())
        delegate.updateEditorGeometry(editor, option, tool_index)
        assert editor.geometry().left() > type_rect.right()
        editor.deleteLater()

        pixmap = QtGui.QPixmap(option.rect.size())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        delegate.paint(painter, option, tool_index)
        painter.end()

        assert model.setData(
            tool_index,
            "renamed_dtool",
            QtCore.Qt.ItemDataRole.EditRole,
        )
        assert tool._tool_display_name == "renamed_dtool"
        assert tool_index.data(_TOOL_TYPE_ROLE) == tool.tool_name
        assert tool_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == "renamed_dtool"
        assert tool_index.data(QtCore.Qt.ItemDataRole.EditRole) == "renamed_dtool"


def test_manager_multi_data_not_shown(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert not manager.get_imagetool(0).isVisible()
        assert not manager.get_imagetool(1).isVisible()


def test_manager_replace(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        # Replace data in the tool
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(test_data**2, manager=True, replace=0)

        assert manager.get_imagetool(0).array_slicer.point_value(0) == 144.0

        # Replacing 1 should create a new tool
        itool(test_data**2, manager=True, replace=1)
        qtbot.wait_until(lambda: manager.ntools == 2)
        assert manager.get_imagetool(1).array_slicer.point_value(0) == 144.0

        # Negative indexing
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(test_data, manager=True, replace=-1)

        assert manager.get_imagetool(1).array_slicer.point_value(0) == 12.0


def test_manager_childtool_source_updates(
    qtbot,
    accept_dialog,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        uid, child = next(iter(wrapper._childtools.items()))
        assert isinstance(child, DerivativeTool)
        assert child.source_spec is not None

        initial = test_data.transpose("eV", "alpha")
        xr.testing.assert_identical(child.tool_data, initial)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 3

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(child.tool_data, initial)

        badge_rect, badge_text, index = child_status_badge(manager, uid)
        assert badge_text == "Stale"
        delegate = typing.cast(
            "_ImageToolWrapperItemDelegate", manager.tree_view.itemDelegate()
        )
        option = QtWidgets.QStyleOptionViewItem()
        option.rect = manager.tree_view.visualRect(index)
        option.font = manager.tree_view.font()
        tooltip_text = None

        def _show_tooltip(*args, **kwargs) -> None:
            nonlocal tooltip_text
            tooltip_text = args[1]

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", _show_tooltip)
        help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(badge_rect.center()),
        )
        assert delegate.helpEvent(help_event, manager.tree_view, option, index)
        assert_nonempty_tooltip(tooltip_text)
        assert index == manager.tree_view.indexAt(badge_rect.center())

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        refresh_calls: list[str] = []
        original_refresh_chain = manager._refresh_source_chain_to_uid

        def _track_refresh_chain(refresh_uid: str) -> bool:
            refresh_calls.append(refresh_uid)
            return original_refresh_chain(refresh_uid)

        monkeypatch.setattr(
            manager, "_refresh_source_chain_to_uid", _track_refresh_chain
        )

        click_child_status_badge(
            manager,
            uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        assert refresh_calls == [uid]
        assert child.source_state == "fresh"
        assert child.source_auto_update is True
        xr.testing.assert_identical(child.tool_data, replaced.transpose("eV", "alpha"))
        auto_badge_rect, auto_badge_text, _ = child_status_badge(manager, uid)
        assert auto_badge_text == "Auto"
        assert not child._source_status_bar.isHidden()
        assert child.source_status_text == "Automatic Updates Enabled"

        tooltip_text = None
        auto_help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            auto_badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(auto_badge_rect.center()),
        )
        assert delegate.helpEvent(auto_help_event, manager.tree_view, option, index)
        assert_nonempty_tooltip(tooltip_text)

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )

        assert child.source_state == "fresh"
        assert child.source_auto_update is False
        assert child._source_status_bar.isHidden()

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(child.tool_data, replaced.transpose("eV", "alpha"))

        accept_dialog(
            lambda: child._source_status_button.click(), accept_call=_update_now
        )

        assert refresh_calls == [uid, uid]
        assert child.source_state == "fresh"
        xr.testing.assert_identical(child.tool_data, replaced2.transpose("eV", "alpha"))

        child._set_source_state("unavailable")
        qtbot.wait_until(lambda: child.source_state == "unavailable", timeout=5000)
        unavailable_badge_rect, unavailable_badge_text, _ = child_status_badge(
            manager, uid
        )
        assert isinstance(unavailable_badge_text, str)
        assert unavailable_badge_text.strip()

        tooltip_text = None
        unavailable_help_event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            unavailable_badge_rect.center(),
            manager.tree_view.viewport().mapToGlobal(unavailable_badge_rect.center()),
        )
        assert delegate.helpEvent(
            unavailable_help_event, manager.tree_view, option, index
        )
        assert_nonempty_tooltip(tooltip_text)


def test_manager_reload_selected_preserves_manual_root_name(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.astype(float).rename("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        root_index = model._row_index(0)
        assert model.setData(
            root_index,
            "manual root name",
            QtCore.Qt.ItemDataRole.EditRole,
        )
        root_tool = manager.get_imagetool(0)
        assert manager.name_of_imagetool(0) == "manual root name"
        assert root_tool.slicer_area._data.name == "manual root name"
        assert root_index.data(QtCore.Qt.ItemDataRole.EditRole) == "manual root name"
        assert (
            root_index.data(QtCore.Qt.ItemDataRole.DisplayRole)
            == "0: manual root name (scan)"
        )
        assert (
            manager_widgets._strip_workspace_modified_placeholder(
                root_tool.windowTitle()
            )
            == "0: manual root name (scan)"
        )

        updated = (source + 100.0).rename("reloaded_scan")
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        manager._update_actions()
        assert manager.reload_action.isVisible()

        with qtbot.wait_signal(root_tool.slicer_area.sigDataChanged, timeout=5000):
            manager.reload_selected()

        assert manager.name_of_imagetool(0) == "manual root name"
        assert (
            manager_widgets._strip_workspace_modified_placeholder(
                root_tool.windowTitle()
            )
            == "0: manual root name (scan)"
        )
        xr.testing.assert_identical(fetch(0), updated.rename("manual root name"))


def test_manager_file_suffix_does_not_seed_unnamed_root_name(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.astype(float).rename(None)
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        root_index = model._row_index(0)

        assert manager.name_of_imagetool(0) == ""
        assert root_index.data(QtCore.Qt.ItemDataRole.EditRole) == ""
        assert root_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == "0 (scan)"
        assert (
            manager_widgets._strip_workspace_modified_placeholder(
                manager.get_imagetool(0).windowTitle()
            )
            == "0 (scan)"
        )


def test_manager_rename_updates_accepted_filter_data(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    operation = NormalizeOperation(
        dims=("alpha",),
        mode="min",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data.astype(float).rename("scan"), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root_tool = manager.get_imagetool(0)
        root_tool.slicer_area.apply_filter_operation(operation)

        manager.rename_imagetool(0, "filtered scan")

        assert root_tool.slicer_area._data.name == "filtered scan"
        assert root_tool.slicer_area._accepted_filter_data is not None
        assert root_tool.slicer_area._accepted_filter_data.name == "filtered scan"
        assert root_tool.slicer_area.array_slicer._obj.name == "filtered scan"


def test_manager_reload_selected_preserves_manual_child_imagetool_name(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = itool(source.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=False,
        )

        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        child_index = model._row_index(child_uid)
        assert model.setData(
            child_index,
            "manual child name",
            QtCore.Qt.ItemDataRole.EditRole,
        )
        child_node = manager._child_node(child_uid)
        assert child_node.name == "manual child name"
        assert child_tool.slicer_area._data.name == "manual child name"
        assert child_index.data(QtCore.Qt.ItemDataRole.EditRole) == "manual child name"
        assert (
            child_index.data(QtCore.Qt.ItemDataRole.DisplayRole) == "manual child name"
        )
        assert (
            manager_widgets._strip_workspace_modified_placeholder(
                child_tool.windowTitle()
            )
            == "manual child name"
        )

        updated = (source + 50.0).rename("reloaded_scan")
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        refresh_key = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Refresh)
        assert (
            manager.reload_action.shortcut().matches(refresh_key)
            == QtGui.QKeySequence.SequenceMatch.ExactMatch
        )
        assert (
            manager.reload_action.shortcutContext()
            == QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        assert manager.reload_action in manager.file_menu.actions()
        assert manager.reload_action.isVisible()

        with qtbot.wait_signal(child_tool.slicer_area.sigDataChanged, timeout=5000):
            manager.reload_selected()

        assert child_node.source_state == "fresh"
        assert child_node.name == "manual child name"
        assert (
            manager_widgets._strip_workspace_modified_placeholder(
                child_tool.windowTitle()
            )
            == "manual child name"
        )
        xr.testing.assert_identical(
            fetch(child_uid), updated.rename("manual child name")
        )


def test_managed_child_imagetool_file_menu_reload_refreshes_file_parent(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = itool(source.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=False,
        )

        child_menu_bar = typing.cast(
            "erlab.interactive.imagetool._mainwindow.ItoolMenuBar",
            child_tool.menuBar(),
        )
        child_file_menu = child_menu_bar.menu_dict["fileMenu"]
        child_file_menu.aboutToShow.emit()
        assert child_tool.slicer_area.reload_act.isVisible()

        updated = (source + 75.0).rename("reloaded_scan")
        updated.to_netcdf(file_path, engine="h5netcdf")

        with qtbot.wait_signal(child_tool.slicer_area.sigDataChanged, timeout=5000):
            child_tool.slicer_area.reload_act.trigger()

        assert manager._child_node(child_uid).source_state == "fresh"
        xr.testing.assert_identical(fetch(0), updated)
        xr.testing.assert_identical(fetch(child_uid), updated)


def test_acquisition_context_is_replayed_when_file_data_reloads(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = _batch_data("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")
    state = AcquisitionContextState(
        enabled=True,
        fields=(
            AcquisitionContextField.from_value(
                kind="coordinate", name="temperature", value=20.0
            ),
            AcquisitionContextField.from_value(
                kind="attribute", name="sample", value="reference"
            ),
        ),
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._acquisition_context.set_state(state, mark_dirty=False)
        assert manager._data_ingress.receive_data(
            [source],
            {
                "file_path": file_path,
                "load_func": (
                    xr.load_dataarray,
                    {"engine": "h5netcdf"},
                    FileDataSelection(kind="dataarray"),
                ),
            },
            show=False,
        ) == [True]
        select_tools(manager, [0])

        updated = (source + 100.0).rename("updated")
        updated.to_netcdf(file_path, engine="h5netcdf")
        with qtbot.wait_signal(
            manager.get_imagetool(0).slicer_area.sigDataChanged, timeout=5000
        ):
            manager.reload_selected()

        actual = manager.get_imagetool(0).slicer_area.data
        assert actual.coords["temperature"].item() == 20.0
        assert actual.attrs["sample"] == "reference"
        xr.testing.assert_identical(
            actual,
            updated.assign_coords(temperature=20.0).assign_attrs(sample="reference"),
        )


def test_imagetool_source_chain_reload_target_falls_back_without_managed_source(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    standalone_tool = itool(test_data, manager=False, execute=False)
    assert isinstance(standalone_tool, erlab.interactive.imagetool.ImageTool)
    assert standalone_tool.slicer_area._managed_source_chain_reload_target() is None
    assert not standalone_tool.slicer_area._reload()

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        root_tool = manager.get_imagetool(0)
        assert root_tool.slicer_area._managed_source_chain_reload_target() is None

        child_tool = itool(test_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool_child(child_tool, 0, show=False)
        assert child_tool.slicer_area._managed_source_chain_reload_target() is None


def test_managed_child_imagetool_file_menu_reload_uses_own_file_source(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.rename("scan")
    file_path = tmp_path / "child_scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")
    unavailable_reasons = _record_reload_unavailable_dialog(monkeypatch)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(source, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = itool(
            source.copy(deep=True),
            manager=False,
            execute=False,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child_tool, 0, show=False)
        assert manager._reload_target_for_child(child_uid) is None
        assert child_tool.slicer_area._reload_unavailable_reason() is None

        child_menu_bar = typing.cast(
            "erlab.interactive.imagetool._mainwindow.ItoolMenuBar",
            child_tool.menuBar(),
        )
        child_file_menu = child_menu_bar.menu_dict["fileMenu"]
        child_file_menu.aboutToShow.emit()
        assert child_tool.slicer_area.reload_act.isVisible()
        assert child_tool.slicer_area.reload_act.isEnabled()

        updated = source.copy(deep=True)
        updated.data = np.asarray(updated.data) + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        with qtbot.wait_signal(child_tool.slicer_area.sigDataChanged, timeout=5000):
            child_tool.slicer_area.reload_act.trigger()

        assert unavailable_reasons == []
        xr.testing.assert_identical(fetch(child_uid), updated)
        xr.testing.assert_identical(child_tool.slicer_area.data, updated)


def test_manager_workspace_reload_preserves_manual_root_name(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.astype(float).rename("scan")
    source_path = tmp_path / "scan.h5"
    source.to_netcdf(source_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=source_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.rename_imagetool(0, "saved manual root")
        workspace_path = tmp_path / "manual-root-name.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.name_of_imagetool(0) == "saved manual root"
        assert manager.get_imagetool(0).slicer_area._data.name == "saved manual root"

        updated = (source + 200.0).rename("updated_scan")
        updated.to_netcdf(source_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_tools(manager, [0])
        with qtbot.wait_signal(
            manager.get_imagetool(0).slicer_area.sigDataChanged, timeout=5000
        ):
            manager.reload_selected()

        assert manager.name_of_imagetool(0) == "saved manual root"
        xr.testing.assert_identical(fetch(0), updated.rename("saved manual root"))


def test_manager_workspace_reload_preserves_manual_child_imagetool_name(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    source_path = tmp_path / "scan.h5"
    source.to_netcdf(source_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=source_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = itool(source.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=False,
        )
        manager._child_node(child_uid).name = "saved manual child"

        workspace_path = tmp_path / "manual-child-name.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        loaded_child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        loaded_child_node = manager._child_node(loaded_child_uid)
        loaded_child_tool = manager.get_imagetool(loaded_child_uid)
        assert loaded_child_node.name == "saved manual child"
        assert loaded_child_tool.slicer_area._data.name == "saved manual child"

        updated = (source + 125.0).rename("updated_scan")
        updated.to_netcdf(source_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, loaded_child_uid)
        with qtbot.wait_signal(
            loaded_child_tool.slicer_area.sigDataChanged, timeout=5000
        ):
            manager.reload_selected()

        assert loaded_child_node.source_state == "fresh"
        assert loaded_child_node.name == "saved manual child"
        xr.testing.assert_identical(
            fetch(loaded_child_uid), updated.rename("saved manual child")
        )


def test_manager_notes_persist_workspace_roundtrip(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(source, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root = manager._tool_graph.root_wrappers[0]
        root.note = "root note"

        child_tool = itool(source.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            note="child note",
        )
        figure_uid = manager.add_figuretool(
            FigureComposerTool(source),
            show=False,
            note="figure note",
        )

        workspace_path = tmp_path / "notes.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        loaded_root = manager._tool_graph.root_wrappers[0]
        loaded_child_uid = loaded_root._childtool_indices[0]
        assert loaded_root.note == "root note"
        assert manager._child_node(loaded_child_uid).note == "child note"
        assert loaded_child_uid == child_uid
        assert manager._tool_graph.figure_uids == [figure_uid]
        assert manager._child_node(figure_uid).note == "figure note"


def test_acquisition_context_persists_on_open_but_not_workspace_import(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    saved_state = AcquisitionContextState(
        enabled=True,
        fields=(
            AcquisitionContextField.from_value(
                kind="coordinate", name="temperature", value=20.0
            ),
            AcquisitionContextField.from_value(
                kind="attribute", name="sample", value="reference"
            ),
        ),
    )
    delta_saved_state = AcquisitionContextState(
        enabled=True,
        fields=(
            AcquisitionContextField.from_value(
                kind="attribute", name="operator", value="current user"
            ),
            AcquisitionContextField.from_value(
                kind="coordinate", name="photon_energy", value=21.2
            ),
            AcquisitionContextField.from_value(kind="attribute", name="run", value=4),
        ),
    )
    transient_state = AcquisitionContextState(
        enabled=True,
        fields=(
            AcquisitionContextField.from_value(kind="attribute", name="run", value=4),
        ),
    )

    with manager_context() as manager:
        manager.show()
        _add_batch_tools(qtbot, manager, _batch_data("scan"))
        manager._acquisition_context.set_state(saved_state, mark_dirty=False)
        workspace_path = tmp_path / "acquisition-context.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )

        manager._acquisition_context.set_state(delta_saved_state)
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=False
        )
        assert manager._workspace_state.delta_save_count == 1

        manager._acquisition_context.set_state(transient_state, mark_dirty=False)
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=False,
            associate=False,
            mark_dirty=True,
            select=False,
        )
        assert manager._acquisition_context.state == transient_state

        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert manager._acquisition_context.state == delta_saved_state
        assert manager.acquisition_context_status_button.isVisible()


def test_metadata_assignment_provenance_persists_for_live_replacement(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    field = MetadataField(kind="attribute", name="sample")
    source = _batch_data("scan")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        assert manager._data_ingress.receive_data([source], {}, show=False) == [True]
        assert manager._metadata_editor.apply_edits(
            {0: (MetadataCellEdit(field, value="reference"),)}
        )

        workspace_path = tmp_path / "metadata-assignment.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert manager._acquisition_context.state == AcquisitionContextState()
        spec = manager._tool_graph.root_wrappers[0].displayed_provenance_spec
        assert spec is not None
        assert any(
            isinstance(operation, AssignAttrsOperation)
            for _ref, operation in iter_operation_refs(spec)
        )

        replacement = _batch_data("replacement", offset=100.0)
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            replace_data(0, replacement)
        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area.data,
            replacement.assign_attrs(sample="reference"),
        )


def test_metadata_editor_layout_persists_with_workspace(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    sample = MetadataField(kind="attribute", name="sample")
    operator = MetadataField(kind="attribute", name="operator")
    transient = MetadataField(kind="coordinate", name="temperature")

    with manager_context() as manager:
        _add_batch_tools(
            qtbot,
            manager,
            _batch_data("scan").assign_attrs(sample="reference", operator="user"),
        )
        manager._metadata_editor.set_layout_fields(
            (sample, operator), {sample: "reference", operator: "user"}
        )
        assert manager._metadata_editor.saved_field_type(sample) == "String"
        assert manager._metadata_editor.saved_field_type(operator) == "String"
        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0,))
        qtbot.addWidget(dialog)
        dialog.show()
        dialog.table.frozen_view.setColumnWidth(0, 237)
        assert dialog.table.columnWidth(0) == 237
        dialog.table.setColumnWidth(dialog.model.fields.index(sample) + 1, 119)
        dialog.table.setColumnWidth(dialog.model.fields.index(operator) + 1, 173)
        assert manager._metadata_editor.saved_column_width(None) == 237
        assert manager._metadata_editor.saved_column_width(sample) == 119
        assert manager._metadata_editor.saved_column_width(operator) == 173

        dialog._set_visible_fields((operator, sample))
        assert dialog.table.columnWidth(dialog.model.fields.index(sample) + 1) == 119
        assert dialog.table.columnWidth(dialog.model.fields.index(operator) + 1) == 173
        dialog.reject()

        reopened = MetadataEditorDialog(manager, manager._metadata_editor, (0,))
        qtbot.addWidget(reopened)
        assert reopened.table.columnWidth(0) == 237
        assert (
            reopened.table.columnWidth(reopened.model.fields.index(sample) + 1) == 119
        )
        assert (
            reopened.table.columnWidth(reopened.model.fields.index(operator) + 1) == 173
        )
        reopened.reject()

        workspace_path = tmp_path / "metadata-layout.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        manager._metadata_editor.set_layout_fields((transient,))
        manager._metadata_editor.set_column_width(None, 301)
        manager._metadata_editor.set_column_width(transient, 181)
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert manager._metadata_editor.layout_state.fields == (operator, sample)
        assert manager._metadata_editor.saved_column_width(None) == 237
        assert manager._metadata_editor.saved_column_width(sample) == 119
        assert manager._metadata_editor.saved_column_width(operator) == 173
        assert manager._metadata_editor.saved_column_width(transient) is None
        assert manager._metadata_editor.saved_field_type(sample) == "String"
        assert manager._metadata_editor.saved_field_type(operator) == "String"

        restored = MetadataEditorDialog(manager, manager._metadata_editor, (0,))
        qtbot.addWidget(restored)
        assert restored.table.columnWidth(0) == 237
        assert (
            restored.table.columnWidth(restored.model.fields.index(sample) + 1) == 119
        )
        assert (
            restored.table.columnWidth(restored.model.fields.index(operator) + 1) == 173
        )


def test_metadata_editor_reads_deferred_workspace_metadata_without_materializing(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    temperature = MetadataField(kind="coordinate", name="temperature")
    sample = MetadataField(kind="attribute", name="sample")

    with manager_context() as manager:
        manager.show()
        for index in range(2):
            data = _batch_data(f"deferred{index}").assign_coords(
                temperature=20.0 + index
            )
            data.attrs["sample"] = f"sample-{index}"
            tool = typing.cast(
                "erlab.interactive.imagetool.ImageTool",
                itool(data, manager=False, execute=False),
            )
            manager.add_imagetool(tool, show=False)
            tool.hide()

        workspace_path = tmp_path / "deferred-metadata-editor.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        wrappers = tuple(manager._tool_graph.root_wrappers.values())
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("opening the metadata editor materialized deferred data")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )
        dialog = MetadataEditorDialog(manager, manager._metadata_editor, (0, 1))
        qtbot.addWidget(dialog)

        temperature_column = dialog.model.fields.index(temperature) + 1
        sample_column = dialog.model.fields.index(sample) + 1
        assert (
            dialog.model.cell_at(dialog.model.index(0, temperature_column)).value
            == 20.0
        )
        assert (
            dialog.model.cell_at(dialog.model.index(1, temperature_column)).value
            == 21.0
        )
        assert dialog.model.cell_at(dialog.model.index(0, sample_column)).value == (
            "sample-0"
        )
        assert dialog.model.cell_at(dialog.model.index(1, sample_column)).value == (
            "sample-1"
        )
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )

        select_tools(manager, [0])
        context_dialog = AcquisitionContextDialog(manager, manager._acquisition_context)
        qtbot.addWidget(context_dialog)
        selected_data = context_dialog._selected_data()
        assert selected_data is not None
        assert selected_data.coords["temperature"].item() == 20.0
        assert selected_data.attrs["sample"] == "sample-0"
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )


def test_manager_notes_preserved_by_duplicate_and_promote(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        manager._tool_graph.root_wrappers[0].note = "root note"

        child_tool = itool(test_data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            note="child note",
        )
        select_tools(manager, [0])
        manager.edit_note_action.trigger()
        manager.notes_editor.setPlainText("pending duplicate root note")
        manager._note_commit_timer.stop()
        duplicated_target = manager.duplicate_imagetool(0)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        duplicated = manager._node_for_target(duplicated_target)
        assert duplicated.note == "pending duplicate root note"
        duplicated_child_uid = duplicated._childtool_indices[0]
        assert manager._child_node(duplicated_child_uid).note == "child note"

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager.edit_note_action.trigger()
        manager.notes_editor.setPlainText("pending promoted child note")
        manager._note_commit_timer.stop()
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        manager.promote_selected()
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        assert (
            manager._tool_graph.nodes[child_uid].note == "pending promoted child note"
        )
        figure_uid = manager.add_figuretool(
            FigureComposerTool(test_data),
            show=False,
            note="figure note",
        )
        duplicated_figure_uid = manager.duplicate_childtool(figure_uid)
        assert manager._child_node(duplicated_figure_uid).note == "figure note"


@pytest.mark.parametrize("auto_update", [False, True], ids=["manual", "auto"])
def test_manager_reload_selected_child_tool_refreshes_from_file_parent(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    auto_update: bool,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.rename("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)
        child.set_source_binding(child.source_spec, auto_update=auto_update)

        updated = source.copy(deep=True)
        updated.data = np.asarray(updated.data) + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.reload_action.isVisible()

        with qtbot.wait_signal(child.sigDataChanged, timeout=5000):
            manager.reload_selected()

        assert child.source_state == "fresh"
        xr.testing.assert_identical(fetch(0), updated)
        xr.testing.assert_identical(child.tool_data, updated.transpose("eV", "alpha"))


def test_managed_child_tool_file_menu_reload_refreshes_file_parent(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.rename("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)
        assert child.source_state == "fresh"

        reload_action = child.findChild(QtGui.QAction, "tool_reload_data_action")
        file_menu = child._tool_file_menu
        assert reload_action is not None
        assert child._source_status_bar.isHidden()
        file_menu.aboutToShow.emit()
        assert file_menu.menuAction().isVisible()
        assert reload_action.isVisible()
        assert reload_action.isEnabled()
        refresh_key = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Refresh)
        assert (
            reload_action.shortcut().matches(refresh_key)
            == QtGui.QKeySequence.SequenceMatch.ExactMatch
        )

        updated = source.copy(deep=True)
        updated.data = np.asarray(updated.data) + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        with qtbot.wait_signal(child.sigDataChanged, timeout=5000):
            reload_action.trigger()

        assert child.source_state == "fresh"
        xr.testing.assert_identical(fetch(0), updated)
        xr.testing.assert_identical(child.tool_data, updated.transpose("eV", "alpha"))


def test_managed_nested_child_tool_file_menu_reload_refreshes_file_ancestor(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = test_data.rename("scan")
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_source_spec = selection(IselOperation(kwargs={"alpha": slice(0, 4)}))
        child_tool = itool(
            child_source_spec.apply(source), manager=False, execute=False
        )
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=child_source_spec,
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtools) == 1, timeout=5000)

        tool_uid = child_node._childtool_indices[0]
        tool = manager.get_childtool(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert child_node.source_state == "fresh"
        assert tool.source_state == "fresh"

        reload_action = tool.findChild(QtGui.QAction, "tool_reload_data_action")
        assert reload_action is not None
        assert reload_action.isEnabled()

        updated = source.copy(deep=True)
        updated.data = np.asarray(updated.data) + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        with qtbot.wait_signal(tool.sigDataChanged, timeout=5000):
            reload_action.trigger()

        expected_child = updated.isel(alpha=slice(0, 4))
        assert child_node.source_state == "fresh"
        assert tool.source_state == "fresh"
        xr.testing.assert_identical(fetch(0), updated)
        xr.testing.assert_identical(fetch(child_uid), expected_child)
        xr.testing.assert_identical(
            tool.tool_data, expected_child.transpose("eV", "alpha")
        )


def test_managed_child_tool_shows_reload_reason_without_reloadable_ancestor(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    unavailable_reasons = _record_reload_unavailable_dialog(monkeypatch)
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)
        child_node = manager._child_node(child_uid)
        with monkeypatch.context() as patch:
            patch.setattr(child_node, "_tool_window", None)
            assert child_node.reload_unavailable_reason()

        reload_action = child.findChild(QtGui.QAction, "tool_reload_data_action")
        assert reload_action is not None
        child._tool_file_menu.aboutToShow.emit()
        assert child._tool_file_menu.menuAction().isVisible()
        assert reload_action.isVisible()
        assert reload_action.isEnabled()
        assert not child.reload_source_data()
        assert unavailable_reasons
        assert not manager._reload_source_chain_for_child(child_uid)

        updated = test_data.copy(deep=True)
        updated.data = np.asarray(updated.data) + 100.0
        file_path = tmp_path / "scan.h5"
        updated.to_netcdf(file_path, engine="h5netcdf")
        parent_tool.slicer_area._file_path = file_path
        parent_tool.slicer_area._load_func = (
            xr.load_dataarray,
            {"engine": "h5netcdf"},
            FileDataSelection(kind="dataarray"),
        )

        unavailable_reasons.clear()
        child._tool_file_menu.aboutToShow.emit()
        assert reload_action.isVisible()
        assert reload_action.isEnabled()
        with qtbot.wait_signal(child.sigDataChanged, timeout=5000):
            reload_action.trigger()

        assert not unavailable_reasons
        xr.testing.assert_identical(fetch(0), updated)
        xr.testing.assert_identical(child.tool_data, updated.transpose("eV", "alpha"))


def test_manager_reload_selected_nested_child_refreshes_from_file_ancestor(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_tool = itool(source.copy(deep=False), manager=False, execute=False)
        assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=False,
            source_spec=full_data(),
            source_auto_update=False,
        )

        grandchild_tool = itool(
            source.isel(y=slice(0, 2)), manager=False, execute=False
        )
        assert isinstance(grandchild_tool, erlab.interactive.imagetool.ImageTool)
        grandchild_uid = manager.add_imagetool_child(
            grandchild_tool,
            child_uid,
            show=False,
            source_spec=selection(IselOperation(kwargs={"y": slice(0, 2)})),
            source_auto_update=False,
        )

        child_node = manager._child_node(child_uid)
        grandchild_node = manager._child_node(grandchild_uid)
        updated = source + 50.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, grandchild_uid)
        manager._update_actions()
        assert manager.reload_action.isVisible()

        with qtbot.wait_signal(
            grandchild_tool.slicer_area.sigDataChanged, timeout=5000
        ):
            manager.reload_selected()

        assert child_node.source_state == "fresh"
        assert grandchild_node.source_state == "fresh"
        xr.testing.assert_identical(fetch(child_uid), updated)
        xr.testing.assert_identical(fetch(grandchild_uid), updated.isel(y=slice(0, 2)))


def test_manager_reload_multi_selected_children_dedupes_file_ancestor(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source = xr.DataArray(
        np.arange(24, dtype=float).reshape((6, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(6), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        first_tool = itool(source.isel(x=slice(0, 2)), manager=False, execute=False)
        second_tool = itool(source.isel(x=slice(2, 4)), manager=False, execute=False)
        assert isinstance(first_tool, erlab.interactive.imagetool.ImageTool)
        assert isinstance(second_tool, erlab.interactive.imagetool.ImageTool)
        first_uid = manager.add_imagetool_child(
            first_tool,
            0,
            show=False,
            source_spec=selection(IselOperation(kwargs={"x": slice(0, 2)})),
            source_auto_update=False,
        )
        second_uid = manager.add_imagetool_child(
            second_tool,
            0,
            show=False,
            source_spec=selection(IselOperation(kwargs={"x": slice(2, 4)})),
            source_auto_update=False,
        )

        root_node = manager._tool_graph.root_wrappers[0]
        root_area = root_node.slicer_area
        original_reload = root_area._reload
        reload_calls: list[int] = []

        def _track_reload() -> bool:
            reload_calls.append(root_node.index)
            return original_reload()

        monkeypatch.setattr(root_area, "_reload", _track_reload)

        updated = source + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, first_uid)
        select_child_tool(manager, second_uid)
        manager._update_actions()
        assert manager.reload_action.isVisible()

        manager.reload_selected()

        assert reload_calls == [0]
        qtbot.wait_until(
            lambda: (
                manager._child_node(first_uid).source_state == "fresh"
                and manager._child_node(second_uid).source_state == "fresh"
            ),
            timeout=5000,
        )
        xr.testing.assert_identical(fetch(first_uid), updated.isel(x=slice(0, 2)))
        xr.testing.assert_identical(fetch(second_uid), updated.isel(x=slice(2, 4)))


def test_manager_reload_mixed_child_selection_requires_all_children_eligible(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    unavailable_reasons = _record_reload_unavailable_dialog(monkeypatch)
    source = xr.DataArray(
        np.arange(12, dtype=float).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )
    file_path = tmp_path / "scan.h5"
    source.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(
            source,
            manager=True,
            file_path=file_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        eligible_tool = itool(source.isel(x=slice(0, 2)), manager=False, execute=False)
        unbound_tool = itool(source.isel(x=slice(1, 3)), manager=False, execute=False)
        assert isinstance(eligible_tool, erlab.interactive.imagetool.ImageTool)
        assert isinstance(unbound_tool, erlab.interactive.imagetool.ImageTool)
        eligible_uid = manager.add_imagetool_child(
            eligible_tool,
            0,
            show=False,
            source_spec=selection(IselOperation(kwargs={"x": slice(0, 2)})),
            source_auto_update=False,
        )
        unbound_uid = manager.add_imagetool_child(unbound_tool, 0, show=False)

        root_node = manager._tool_graph.root_wrappers[0]
        reload_calls: list[int] = []
        monkeypatch.setattr(
            root_node.slicer_area,
            "_reload",
            lambda: reload_calls.append(root_node.index) or True,
        )

        updated = source + 100.0
        updated.to_netcdf(file_path, engine="h5netcdf")

        manager.tree_view.clearSelection()
        select_child_tool(manager, eligible_uid)
        select_child_tool(manager, unbound_uid)
        manager._update_actions()
        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()

        manager.reload_selected()

        assert unavailable_reasons
        assert reload_calls == []
        xr.testing.assert_identical(fetch(0), source)
        xr.testing.assert_identical(fetch(eligible_uid), source.isel(x=slice(0, 2)))


def test_manager_selected_reload_targets_handles_stale_selection(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager, monkeypatch.context() as stale_selection:
        stale_selection.setattr(
            type(manager.tree_view),
            "selected_imagetool_indices",
            property(lambda _view: []),
        )
        stale_selection.setattr(
            type(manager.tree_view),
            "selected_childtool_uids",
            property(lambda _view: ["stale-child"]),
        )
        stale_selection.setattr(
            manager,
            "_child_node",
            lambda _uid: (_ for _ in ()).throw(KeyError("missing")),
        )

        assert manager._selected_reload_targets() is None

        child_node = types.SimpleNamespace(has_source_binding=True)
        stale_selection.setattr(manager, "_child_node", lambda _uid: child_node)
        stale_selection.setattr(
            manager,
            "_parent_node",
            lambda _node: (_ for _ in ()).throw(KeyError("missing-parent")),
        )

        assert manager._selected_reload_targets() is None


def test_manager_reload_selected_skips_child_refresh_when_parent_reload_fails(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager, monkeypatch.context() as reload_failure:
        node = types.SimpleNamespace(
            imagetool=object(),
            slicer_area=types.SimpleNamespace(_reload=lambda: False),
        )
        refreshed: list[str] = []
        reload_failure.setattr(
            manager, "_selected_reload_targets", lambda: ([0], {0: ["child"]})
        )
        reload_failure.setattr(manager, "_node_for_target", lambda _target: node)
        reload_failure.setattr(
            manager, "_refresh_source_chain_to_uid", refreshed.append
        )
        manager.reload_selected()

        assert refreshed == []


def test_manager_full_data_childtool_updates_follow_transposed_view(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.transpose_main_image()
        assert parent_tool.slicer_area.data.dims == ("eV", "alpha")

        parent_tool.slicer_area.open_in_meshtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child = next(iter(manager._tool_graph.root_wrappers[0]._childtools.values()))
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        assert child._update_from_parent_source() is True
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5

        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        xarray.testing.assert_identical(child.tool_data, parent_tool.slicer_area.data)


def test_manager_selection_child_replays_stable_source_spec_after_coordinate_shift(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4 * 5 * 3, dtype=float).reshape((4, 5, 3)),
        dims=("x", "y", "z"),
        coords={"x": np.arange(4), "y": np.arange(5), "z": np.arange(3)},
        name="scan",
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.set_value(axis=2, value=1.0, cursor=0)
        parent_tool.array_slicer.set_bin(0, axis=2, value=3, update=True)
        parent_tool.slicer_area.images[0].open_in_new_window()

        root = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(lambda: len(root._childtool_indices) == 1, timeout=5000)
        child_uid = root._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        assert child_node.source_binding is None
        assert child_node.source_spec is not None
        original_source_spec = child_node.source_spec

        tree = manager._workspace_controller.saving._to_datatree()
        child_attrs = tree[f"0/childtools/{child_uid}/imagetool"].attrs
        assert "manager_node_live_source_binding" not in child_attrs
        assert "manager_node_live_source_spec" in child_attrs
        child_attrs["manager_node_live_source_binding"] = json.dumps(
            ImageToolSelectionSourceBinding(
                selection_mode="isel",
                selection_indexers={"z": 0},
            ).model_dump(mode="json")
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        loaded_child = manager._child_node(child_uid)
        assert loaded_child.source_binding is None
        assert loaded_child.source_spec == original_source_spec

        shifted = data.assign_coords(z=[10.0, 11.0, 12.0])
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, shifted)

        qtbot.wait_until(lambda: loaded_child.source_state == "stale", timeout=5000)
        assert loaded_child._update_from_parent_source() is True
        child_data = manager.get_imagetool(child_uid).slicer_area._data.rename(None)
        xarray.testing.assert_identical(
            child_data,
            original_source_spec.apply(shifted).rename(None),
        )


def test_manager_add_imagetool_child_materializes_source_binding_without_spec(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": np.arange(3.0), "y": np.arange(4.0)},
        name="scan",
    )
    source_binding = ImageToolSelectionSourceBinding(
        selection_mode="isel",
        selection_indexers={"y": slice(1, 3)},
    )
    source_spec = source_binding.materialize(data)

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = itool(source_spec.apply(data), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        qtbot.addWidget(child)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_binding=source_binding,
        )
        child_node = manager._child_node(child_uid)

        assert child_node.source_binding is None
        assert child_node.source_spec == source_spec


def test_manager_goldtool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid = manager.add_childtool(
            GoldTool(gold.copy(deep=True), data_name="gold_input"),
            0,
            show=False,
        )
        child = manager.get_childtool(child_uid)
        assert isinstance(child, GoldTool)
        configure_goldtool_child(child, fitted=True, spline=False)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.is_imagetool
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "goldtool.corrected"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child.corrected)
        manager.tree_view.clearSelection()
        select_child_tool(manager, output_uid)
        manager._update_info(uid=output_uid)
        assert metadata_derivation_texts(manager) == [
            "Start from current goldtool input data",
            "Fit and correct current data with the polynomial edge model",
        ]
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        assert "era.gold.poly(" in copied
        assert "corrected = era.gold.correct_with_edge(" in copied


def test_manager_dtool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.is_imagetool
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "dtool.result"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child.result.T)
        manager.tree_view.clearSelection()
        select_child_tool(manager, output_uid)
        manager._update_info(uid=output_uid)
        assert isinstance(
            output_node.provenance_spec.operations[-2],
            ImageDerivativeOperation,
        )
        assert isinstance(
            output_node.provenance_spec.operations[-1],
            TransposeOperation,
        )
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        namespace = _exec_generated_code(
            copied,
            {"data": parent_tool.slicer_area.data.copy(deep=True)},
        )
        result = namespace["result"]
        assert isinstance(result, xr.DataArray)
        xr.testing.assert_identical(result, child.result.T)


def test_manager_ktool_output_itool_nests_under_tool(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert manager.ntools == 1
        assert output_node.parent_uid == child_uid
        assert output_node.output_id == "ktool.converted_output"
        assert output_node.source_spec is None
        assert output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(output_uid), child._itool.slicer_area.data)
        copied = copy_full_code_for_uid(monkeypatch, manager, output_uid)
        assert ".kspace.set_normal(" in copied
        assert "_kconv = " in copied

        replaced = anglemap.copy(deep=True)
        replaced.data = np.asarray(replaced.data) + 1.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=True,
            state="fresh",
        )

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 2.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child._converted_output())


def test_manager_ktool_output_itool_marks_stale_without_recomputing(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)
        assert output_node.source_auto_update is False

        select_child_tool(manager, child_uid)
        qtbot.wait(child._MANAGER_NOTIFY_DELAY_MS + 50)
        manager._flush_idle_work(force=True)
        metadata_updates: list[str] = []
        original_set_metadata_node = manager._set_metadata_node

        def _record_metadata_rebuild(node) -> None:
            metadata_updates.append(node.uid)
            original_set_metadata_node(node)

        monkeypatch.setattr(manager, "_set_metadata_node", _record_metadata_rebuild)
        manager._workspace_controller._mark_workspace_clean()

        before = fetch(output_uid).copy(deep=True)
        wait_ms = int(1000 / child._UPDATE_LIMIT_HZ) + 50
        qtbot.wait(wait_ms)

        call_count = 0
        original_converted_output = child._converted_output

        def _counting_converted_output():
            nonlocal call_count
            call_count += 1
            return original_converted_output()

        monkeypatch.setattr(child, "_converted_output", _counting_converted_output)
        notification_count = 0
        original_notify_data_changed = child._notify_data_changed

        def _counting_notify_data_changed():
            nonlocal notification_count
            notification_count += 1
            original_notify_data_changed()

        monkeypatch.setattr(
            child, "_notify_data_changed", _counting_notify_data_changed
        )

        delta_spin = child._offset_spins["delta"]
        with QtCore.QSignalBlocker(delta_spin):
            delta_spin.setValue(delta_spin.value() + 0.01)
        child.update()
        child.update()
        child.update()

        assert child_uid in manager._workspace_state.dirty_state
        assert metadata_updates == []
        assert notification_count == 0
        qtbot.wait(child._MANAGER_NOTIFY_DELAY_MS + 50)
        manager._flush_idle_work(force=True)
        assert notification_count == 1
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        qtbot.wait(wait_ms)

        assert call_count == 0
        xr.testing.assert_identical(fetch(output_uid), before)


def test_manager_reused_output_child_keeps_stale_state(
    qtbot,
    monkeypatch,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = typing.cast("typing.Any", manager.get_childtool(child_uid))
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        child.set_source_binding(child.source_spec, auto_update=False, state="stale")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=False,
            state="stale",
        )

        monkeypatch.setattr(
            child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        child.show_converted()

        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child._converted_output())


def test_manager_dtool_output_itool_refreshes_with_parent_updates(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "stale", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")
        output_node.set_output_binding(
            typing.cast("str", output_node.output_id),
            provenance_spec=output_node.provenance_spec,
            auto_update=True,
            state="fresh",
        )

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), child.result.T)


def test_manager_output_itool_auto_update_can_be_disabled_from_auto_badge(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]
        output_node = manager._child_node(output_uid)

        child.set_source_binding(child.source_spec, auto_update=True, state="fresh")

        replaced = test_data.copy(deep=True)
        replaced.data = np.asarray(replaced.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)

        def _enable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(True)  # type: ignore[attr-defined]

        def _update_now(dialog: QtWidgets.QDialog) -> None:
            dialog.update_button.click()  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            output_uid,
            accept_dialog,
            pre_call=_enable_auto_update,
            accept_call=_update_now,
        )

        qtbot.wait_until(lambda: output_node.source_state == "fresh", timeout=5000)
        assert output_node.source_auto_update is True
        xr.testing.assert_identical(fetch(output_uid), child.result.T)
        refreshed_output = fetch(output_uid).copy(deep=True)
        _, badge_text, _ = child_status_badge(manager, output_uid)
        assert badge_text == "Auto"

        def _disable_auto_update(dialog: QtWidgets.QDialog) -> None:
            dialog.auto_update_check.setChecked(False)  # type: ignore[attr-defined]

        click_child_status_badge(
            manager,
            output_uid,
            accept_dialog,
            pre_call=_disable_auto_update,
        )
        assert output_node.source_auto_update is False

        replaced2 = replaced.copy(deep=True)
        replaced2.data = np.asarray(replaced2.data) + 5.0
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(replaced2, link=False, manager=True, replace=0)

        qtbot.wait_until(lambda: child.source_state == "fresh", timeout=5000)
        qtbot.wait_until(lambda: output_node.source_state == "stale", timeout=5000)
        xr.testing.assert_identical(fetch(output_uid), refreshed_output)
