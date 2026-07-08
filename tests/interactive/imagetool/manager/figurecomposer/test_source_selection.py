from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

import pytest
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._figurecomposer._widgets as figurecomposer_widgets
from erlab.interactive._figurecomposer import (
    FigureDataSelectionState,
    FigureOperationState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._editor_controls import MIXED_VALUE
from erlab.interactive._figurecomposer._operations import (
    _source_selection as source_selection,
)
from erlab.interactive._figurecomposer._text import FigureComposerInputError


def _operation_factory(
    operation: FigureOperationState,
    sources: tuple[str, ...],
    selections: tuple[FigureDataSelectionState, ...],
) -> FigureOperationState:
    return operation.model_copy(
        update={"sources": sources, "map_selections": selections}
    )


class _FakeSelectionTool:
    def __init__(self, operation: FigureOperationState) -> None:
        self.operation = operation
        self.editor_updates = 0
        self.update_flags: tuple[bool, bool] | None = None

    def _update_operation_editor(self) -> None:
        self.editor_updates += 1

    def _update_operations(
        self,
        callback: Callable[[int, FigureOperationState], FigureOperationState],
        *,
        rebuild_editor: bool,
        defer_editor_rebuild: bool,
    ) -> None:
        self.operation = callback(0, self.operation)
        self.update_flags = (rebuild_editor, defer_editor_rebuild)


class _FakeEditorTool:
    def __init__(self) -> None:
        self.marked: list[QtWidgets.QWidget] = []
        self.cleared: list[QtWidgets.QWidget] = []
        self.form_rows: list[tuple[str, QtWidgets.QWidget, str]] = []

    def _mark_editor_control(self, widget: QtWidgets.QWidget) -> None:
        self.marked.append(widget)

    def _connect_editor_signal(
        self,
        _widget: QtWidgets.QWidget,
        signal: Any,
        callback: Callable[..., None],
    ) -> None:
        signal.connect(callback)

    def _connect_line_edit_finished(
        self,
        edit: QtWidgets.QLineEdit,
        callback: Callable[[str], None],
    ) -> None:
        edit.editingFinished.connect(lambda edit=edit: callback(edit.text()))

    def _clear_editor_input_error(self, widget: QtWidgets.QWidget) -> None:
        self.cleared.append(widget)

    def _batch_is_mixed(
        self,
        _operation: FigureOperationState,
        getter: Callable[[FigureOperationState], str],
    ) -> bool:
        return getter(self._alternate_operation) == "mean"

    def _batch_text(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], str],
        _converter: type[str],
    ) -> tuple[str, bool]:
        return getter(operation), getter(self._alternate_operation) != getter(operation)

    @staticmethod
    def _line_edit(text: str, *, parent: QtWidgets.QWidget) -> QtWidgets.QLineEdit:
        return QtWidgets.QLineEdit(text, parent)

    @staticmethod
    def _apply_mixed_line_edit(edit: QtWidgets.QLineEdit, mixed: bool) -> None:
        edit.setProperty("batch_mixed", mixed)

    def _add_form_row(
        self,
        layout: QtWidgets.QFormLayout,
        label: str,
        widget: QtWidgets.QWidget,
        tooltip: str,
    ) -> None:
        self.form_rows.append((label, widget, tooltip))
        layout.addRow(label, widget)


def test_source_selection_state_helpers_cover_shared_and_per_source_paths() -> None:
    empty = FigureDataSelectionState(source="data")
    first = FigureDataSelectionState(source="a", qsel={"eV": 0.0})
    first_for_b = first.model_copy(update={"source": "b"})
    second = FigureDataSelectionState(source="b", isel={"eV": 1})

    assert not source_selection.selection_has_effect(empty)
    assert source_selection.selection_has_effect(first)
    assert source_selection.selection_content_equal(first, first_for_b)
    assert source_selection.shared_selection(()) == FigureDataSelectionState(source="")
    assert source_selection.shared_selection((first, first_for_b)) == first.model_copy(
        update={"source": ""}
    )
    assert source_selection.shared_selection((first, second)) is None

    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("a", "b"),
        map_selections=(first, second),
    )
    fallback = FigureDataSelectionState(source="", mean_dims=("kx",))
    assert source_selection.selection_for_source(operation, "a", fallback) == first
    assert source_selection.selection_for_source(
        operation, "missing", fallback
    ) == fallback.model_copy(update={"source": "missing"})

    assert source_selection.source_selection_per_source_enabled(
        _FakeSelectionTool(operation), operation, attr_name="_enabled_ids"
    )
    shared_operation = operation.model_copy(
        update={"map_selections": (first, first_for_b)}
    )
    tool = _FakeSelectionTool(shared_operation)
    assert not source_selection.source_selection_per_source_enabled(
        tool, shared_operation, attr_name="_enabled_ids"
    )

    source_selection.set_source_selection_per_source_enabled(
        tool,
        shared_operation,
        True,
        attr_name="_enabled_ids",
        source_getter=lambda target: target.sources,
        operation_factory=_operation_factory,
        keep_source_only=False,
    )
    assert shared_operation.operation_id in tool._enabled_ids
    assert tool.editor_updates == 1

    source_selection.set_source_selection_per_source_enabled(
        tool,
        operation,
        False,
        attr_name="_enabled_ids",
        source_getter=lambda target: target.sources,
        operation_factory=_operation_factory,
        keep_source_only=False,
    )
    assert shared_operation.operation_id not in tool._enabled_ids
    assert tool.update_flags == (True, True)
    assert tool.operation.map_selections == (first, first_for_b)

    no_selection_operation = operation.model_copy(update={"map_selections": ()})
    empty_tool = _FakeSelectionTool(no_selection_operation)
    source_selection.set_source_selection_per_source_enabled(
        empty_tool,
        no_selection_operation,
        False,
        attr_name="_enabled_ids",
        source_getter=lambda target: target.sources,
        operation_factory=_operation_factory,
        keep_source_only=False,
    )
    assert empty_tool.operation.map_selections == ()


def test_source_selection_operation_helpers_cover_empty_and_effective_paths() -> None:
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("a", "b"),
    )
    empty = FigureDataSelectionState(source="")
    qsel = FigureDataSelectionState(source="", qsel={"eV": 0.0})

    shared_empty = source_selection.operation_with_shared_source_selection(
        operation,
        ("a", "b"),
        empty,
        operation_factory=_operation_factory,
        keep_source_only=False,
    )
    assert shared_empty.map_selections == ()

    shared_source_only = source_selection.operation_with_shared_source_selection(
        operation,
        ("a", "b"),
        empty,
        operation_factory=_operation_factory,
        keep_source_only=True,
    )
    assert shared_source_only.map_selections == (
        FigureDataSelectionState(source="a"),
        FigureDataSelectionState(source="b"),
    )

    shared_qsel = source_selection.operation_with_shared_source_selection(
        operation,
        ("a", "b"),
        qsel,
        operation_factory=_operation_factory,
        keep_source_only=False,
    )
    assert shared_qsel.map_selections == (
        qsel.model_copy(update={"source": "a"}),
        qsel.model_copy(update={"source": "b"}),
    )

    assert (
        source_selection.operation_with_per_source_selection(
            operation,
            "missing",
            qsel,
            source_getter=lambda target: target.sources,
            operation_factory=_operation_factory,
            keep_source_only=False,
        )
        is operation
    )

    per_source = source_selection.operation_with_per_source_selection(
        operation.model_copy(
            update={
                "map_selections": (
                    FigureDataSelectionState(source="a", isel={"kx": 1}),
                    FigureDataSelectionState(source="b", qsel={"eV": 0.0}),
                )
            }
        ),
        "b",
        FigureDataSelectionState(source="b", mean_dims=("kx",)),
        source_getter=lambda target: target.sources,
        operation_factory=_operation_factory,
        keep_source_only=False,
    )
    assert per_source.map_selections == (
        FigureDataSelectionState(source="a", isel={"kx": 1}),
        FigureDataSelectionState(source="b", mean_dims=("kx",)),
    )

    cleared = source_selection.operation_with_per_source_selection(
        operation,
        "a",
        FigureDataSelectionState(source="a"),
        source_getter=lambda target: target.sources,
        operation_factory=_operation_factory,
        keep_source_only=False,
    )
    assert cleared.map_selections == ()


def test_source_selection_dimension_parsing_and_updates() -> None:
    selection = FigureDataSelectionState(
        source="data",
        isel={"kx": 1},
        qsel={"eV": 0.0, "eV_width": 0.2},
        mean_dims=("ky",),
    )

    assert source_selection.selection_dim_mode(selection, "kx") == "isel"
    assert source_selection.selection_dim_mode(selection, "eV") == "qsel"
    assert source_selection.selection_dim_mode(selection, "eV_width") == "qsel"
    assert source_selection.selection_dim_mode(selection, "ky") == "mean"
    assert source_selection.selection_dim_mode(selection, "missing") == "keep"
    assert source_selection.selection_dim_value_text(selection, "kx") == "1"
    assert source_selection.selection_dim_value_text(selection, "eV") == "0.0"
    assert source_selection.selection_dim_value_text(selection, "missing") == ""
    assert source_selection.selection_width_key("eV") == "eV_width"
    assert source_selection.selection_dim_width_text(selection, "eV") == "0.2"
    assert source_selection.selection_dim_width_text(selection, "kx") == ""

    assert source_selection.selection_value_from_text(" slice(0, 2) ") == slice(0, 2)
    assert source_selection.selection_width_from_text("") is None
    assert source_selection.selection_width_from_text("0.5") == 0.5
    with pytest.raises(FigureComposerInputError):
        source_selection.selection_value_from_text("")
    with pytest.raises(FigureComposerInputError):
        source_selection.selection_value_from_text("slice(")
    with pytest.raises(FigureComposerInputError):
        source_selection.selection_width_from_text("slice(0, 2)")
    with pytest.raises(FigureComposerInputError):
        source_selection.selection_width_from_text("bad(")

    assert source_selection.selection_with_dimension(
        selection, "kx", "keep"
    ) == FigureDataSelectionState(
        source="data", qsel={"eV": 0.0, "eV_width": 0.2}, mean_dims=("ky",)
    )
    assert source_selection.selection_with_dimension(
        selection, "eV", "isel", value=2
    ) == FigureDataSelectionState(
        source="data", isel={"kx": 1, "eV": 2}, mean_dims=("ky",)
    )
    assert source_selection.selection_with_dimension(
        selection, "kx", "qsel", value=0.0, width=0.1
    ) == FigureDataSelectionState(
        source="data",
        qsel={"eV": 0.0, "eV_width": 0.2, "kx": 0.0, "kx_width": 0.1},
        mean_dims=("ky",),
    )
    assert source_selection.selection_with_dimension(
        selection, "kx", "mean"
    ) == FigureDataSelectionState(
        source="data",
        qsel={"eV": 0.0, "eV_width": 0.2},
        mean_dims=("ky", "kx"),
    )
    with pytest.raises(FigureComposerInputError):
        source_selection.selection_with_dimension(
            selection, "eV", "qsel", value=slice(0, 2), width=0.1
        )


def test_source_selection_combo_and_dimension_controls(qtbot) -> None:
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    tool = _FakeEditorTool()

    combo = source_selection.selection_mode_combo(
        cast("Any", tool),
        current="qsel",
        mixed=False,
        parent=parent,
    )
    assert combo.currentData() == "qsel"
    assert tool.marked == [combo]

    default_combo = source_selection.selection_mode_combo(
        cast("Any", tool),
        current=None,
        mixed=False,
        parent=parent,
    )
    assert default_combo.currentData() == "keep"

    unknown_combo = source_selection.selection_mode_combo(
        cast("Any", tool),
        current="unknown",
        mixed=False,
        parent=parent,
    )
    assert unknown_combo.currentData() == "keep"

    mixed_combo = source_selection.selection_mode_combo(
        cast("Any", tool),
        current=None,
        mixed=True,
        parent=parent,
    )
    assert mixed_combo.currentData() is MIXED_VALUE
    assert not mixed_combo.model().item(0).isEnabled()

    value_edit = QtWidgets.QLineEdit(parent)
    width_edit = QtWidgets.QLineEdit(parent)
    updates: list[tuple[str, str, str, str]] = []
    source_selection.connect_selection_dimension_controls(
        cast("Any", tool),
        dim="eV",
        mode_combo=combo,
        value_edit=value_edit,
        width_edit=width_edit,
        update_dimension=lambda dim, mode, value, width: updates.append(
            (dim, mode, value, width)
        ),
    )

    value_edit.setText("0.0")
    width_edit.setText("0.1")
    combo.activated.emit(combo.currentIndex())
    assert updates[-1] == ("eV", "qsel", "0.0", "0.1")
    assert not value_edit.isHidden()
    assert not width_edit.isHidden()

    update_count = len(updates)
    value_edit.setText("")
    combo.activated.emit(combo.currentIndex())
    assert len(updates) == update_count

    value_edit.setText("1.0")
    value_edit.editingFinished.emit()
    assert updates[-1] == ("eV", "qsel", "1.0", "0.1")

    width_edit.setText("0.2")
    width_edit.editingFinished.emit()
    assert updates[-1] == ("eV", "qsel", "1.0", "0.2")

    keep_index = combo.findData("keep")
    combo.setCurrentIndex(keep_index)
    combo.activated.emit(keep_index)
    assert updates[-1] == ("eV", "keep", "", "")
    assert value_edit.text() == ""
    assert width_edit.text() == ""
    assert not value_edit.isVisible()
    assert not width_edit.isVisible()
    assert tool.cleared[-2:] == [value_edit, width_edit]

    update_count = len(updates)
    value_edit.editingFinished.emit()
    width_edit.editingFinished.emit()
    assert len(updates) == update_count


def test_add_selection_dimension_rows_builds_empty_and_dimensional_rows(qtbot) -> None:
    page = QtWidgets.QWidget()
    qtbot.addWidget(page)
    layout = QtWidgets.QFormLayout(page)
    tool = _FakeEditorTool()
    operation = FigureOperationState.plot_array(label="plot_array", source="data")
    alternate = FigureDataSelectionState(source="data", mean_dims=("eV",))
    tool._alternate_operation = operation.model_copy(
        update={"map_selections": (alternate,)}
    )
    selection = FigureDataSelectionState(
        source="data", qsel={"eV": 0.0, "eV_width": 0.1}
    )

    source_selection.add_selection_dimension_rows(
        cast("Any", tool),
        operation=operation,
        layout=layout,
        page=page,
        dimensions=(),
        selection_getter=lambda _operation: selection,
        update_dimension=lambda *_args: None,
        object_prefix="figureComposerTest",
        property_prefix="figure_composer_test",
    )
    assert page.findChild(QtWidgets.QLabel, "figureComposerTestDimensionsMessage")

    source_selection.add_selection_dimension_rows(
        cast("Any", tool),
        operation=operation,
        layout=layout,
        page=page,
        dimensions=("eV", "kx"),
        selection_getter=lambda target: (
            target.map_selections[0] if target.map_selections else selection
        ),
        update_dimension=lambda *_args: None,
        object_prefix="figureComposerTest",
        property_prefix="figure_composer_test",
    )

    rows = page.findChildren(QtWidgets.QWidget, "figureComposerTestDimRow0")
    assert rows
    eV_value = page.findChild(QtWidgets.QLineEdit, "figureComposerTestValueEdit0")
    eV_width = page.findChild(QtWidgets.QLineEdit, "figureComposerTestWidthEdit0")
    kx_value = page.findChild(QtWidgets.QLineEdit, "figureComposerTestValueEdit1")
    assert eV_value is not None
    assert eV_value.text() == "0.0"
    assert eV_value.property("figure_composer_test_dim") == "eV"
    assert eV_value.property("figure_composer_test_selection_field") == "value"
    assert eV_width is not None
    assert eV_width.text() == "0.1"
    assert kx_value is not None
    assert not kx_value.isVisible()
    assert [label for label, _widget, _tooltip in tool.form_rows] == ["eV", "kx"]


def test_figure_display_window_source_drop_event_branches(qtbot) -> None:
    window = figurecomposer_widgets._FigureComposerDisplayWindow(FigureSubplotsState())
    qtbot.addWidget(window)
    mime = QtCore.QMimeData()
    assert figurecomposer_widgets._false_mime_state(mime) is False
    assert window._handle_source_drag_event(None) is False
    assert (
        window._handle_source_drag_event(QtCore.QEvent(QtCore.QEvent.Type.DragEnter))
        is False
    )

    def drag_event() -> QtGui.QDragEnterEvent:
        return QtGui.QDragEnterEvent(
            QtCore.QPoint(0, 0),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    def drop_event() -> QtGui.QDropEvent:
        return QtGui.QDropEvent(
            QtCore.QPointF(0.0, 0.0),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    assert window._handle_source_drag_event(drag_event()) is False

    window.set_source_drop_callbacks(can_drop=lambda data: data is mime)
    accepted_drag = drag_event()
    assert window._handle_source_drag_event(accepted_drag)
    assert accepted_drag.isAccepted()

    window.set_source_drop_callbacks(
        can_drop=lambda _data: True,
        drop=lambda _data: False,
    )
    assert window._handle_source_drag_event(drop_event()) is False

    window.set_source_drop_callbacks(
        can_drop=lambda _data: True,
        drop=lambda _data: True,
    )
    accepted_drop = drop_event()
    assert window._handle_source_drag_event(accepted_drop)
    assert accepted_drop.isAccepted()
