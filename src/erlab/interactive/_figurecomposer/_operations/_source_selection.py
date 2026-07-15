"""Shared Figure Composer source-selection editor helpers."""

from __future__ import annotations

import typing
from collections.abc import Callable

from qtpy import QtWidgets

import erlab
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._model._sources import (
    selection_has_effect,
    selection_width_key,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureDataSelectionState,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._text import _dict_from_text
from erlab.interactive._figurecomposer._ui._editor_controls import (
    MIXED_VALUE,
    MIXED_VALUES_TEXT,
    ComboBoxDataControlAdapter,
)

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


SelectionSourceGetter = Callable[[FigureOperationState], tuple[str, ...]]
SelectionOperationFactory = Callable[
    [FigureOperationState, tuple[str, ...], tuple[FigureDataSelectionState, ...]],
    FigureOperationState,
]

_SELECTION_MODE_LABELS = {
    "keep": "None",
    "isel": "isel",
    "qsel": "qsel",
    "mean": "Mean",
}
_SELECTION_VALUE_MODES = {"isel", "qsel"}
_SELECTION_VALUE_MESSAGE = "Enter a selection value, such as 0 or slice(0, 2)."
_SELECTION_WIDTH_MESSAGE = "Enter a qsel width, such as 0.1."
_SELECTION_MODE_TOOLTIP = (
    "Choose whether this dimension stays, is selected, or is averaged."
)
_SELECTION_VALUE_TOOLTIP = (
    "Selection value. Use integer positions or slices for isel; use coordinate "
    "values or slices for qsel."
)
_SELECTION_WIDTH_TOOLTIP = (
    "Optional qsel width centered on the value. Leave blank for nearest "
    "coordinate selection."
)


def selection_content_equal(
    first: FigureDataSelectionState, second: FigureDataSelectionState
) -> bool:
    return (
        first.isel == second.isel
        and first.qsel == second.qsel
        and first.mean_dims == second.mean_dims
    )


def shared_selection(
    selections: Sequence[FigureDataSelectionState],
) -> FigureDataSelectionState | None:
    if not selections:
        return FigureDataSelectionState(source="")
    first = selections[0].model_copy(update={"source": ""})
    if all(selection_content_equal(first, selection) for selection in selections[1:]):
        return first
    return None


def selection_for_source(
    operation: FigureOperationState,
    source: str,
    fallback: FigureDataSelectionState,
) -> FigureDataSelectionState:
    for selection in operation.map_selections:
        if selection.source == source:
            return selection
    return fallback.model_copy(update={"source": source})


def source_selection_per_source_enabled(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    *,
    attr_name: str,
) -> bool:
    if shared_selection(operation.map_selections) is None:
        return True
    return operation.operation_id in typing.cast(
        "set[str]", getattr(tool, attr_name, set())
    )


def set_source_selection_per_source_enabled(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    enabled: bool,
    *,
    attr_name: str,
    source_getter: SelectionSourceGetter,
    operation_factory: SelectionOperationFactory,
    keep_source_only: bool,
) -> None:
    enabled_ids = set(getattr(tool, attr_name, set()))
    if enabled:
        enabled_ids.add(operation.operation_id)
        setattr(tool, attr_name, enabled_ids)
        tool._update_operation_editor()
        return

    enabled_ids.discard(operation.operation_id)
    setattr(tool, attr_name, enabled_ids)
    shared = shared_selection(operation.map_selections)
    if shared is None and operation.map_selections:
        shared = operation.map_selections[0].model_copy(update={"source": ""})
    if shared is None:
        shared = FigureDataSelectionState(source="")

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        sources = source_getter(target)
        return operation_with_shared_source_selection(
            target,
            sources,
            shared,
            operation_factory=operation_factory,
            keep_source_only=keep_source_only,
        )

    tool._update_operations(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def operation_with_shared_source_selection(
    operation: FigureOperationState,
    sources: tuple[str, ...],
    selection: FigureDataSelectionState,
    *,
    operation_factory: SelectionOperationFactory,
    keep_source_only: bool,
) -> FigureOperationState:
    map_selections = _selection_tuple_for_sources(
        sources,
        selection,
        keep_source_only=keep_source_only,
    )
    return operation_factory(operation, sources, map_selections)


def operation_with_per_source_selection(
    operation: FigureOperationState,
    source: str,
    selection: FigureDataSelectionState,
    *,
    source_getter: SelectionSourceGetter,
    operation_factory: SelectionOperationFactory,
    keep_source_only: bool,
) -> FigureOperationState:
    sources = source_getter(operation)
    if source not in sources:
        return operation
    shared = shared_selection(operation.map_selections)
    if shared is None:
        shared = FigureDataSelectionState(source="")
    selections = tuple(
        (
            selection.model_copy(update={"source": source_name})
            if source_name == source
            else selection_for_source(operation, source_name, shared)
        )
        for source_name in sources
    )
    if not keep_source_only and not any(
        selection_has_effect(source_selection) for source_selection in selections
    ):
        selections = ()
    return operation_factory(operation, sources, tuple(selections))


def _selection_tuple_for_sources(
    sources: tuple[str, ...],
    selection: FigureDataSelectionState,
    *,
    keep_source_only: bool,
) -> tuple[FigureDataSelectionState, ...]:
    if not keep_source_only and not selection_has_effect(selection):
        return ()
    return tuple(selection.model_copy(update={"source": source}) for source in sources)


def selection_dim_mode(selection: FigureDataSelectionState, dim: str) -> str:
    if dim in selection.isel:
        return "isel"
    if dim in selection.qsel or selection_width_key(dim) in selection.qsel:
        return "qsel"
    if dim in selection.mean_dims:
        return "mean"
    return "keep"


def selection_dim_value_text(selection: FigureDataSelectionState, dim: str) -> str:
    if dim in selection.isel:
        return erlab.interactive.utils._parse_single_arg(selection.isel[dim])
    if dim in selection.qsel:
        return erlab.interactive.utils._parse_single_arg(selection.qsel[dim])
    return ""


def selection_dim_width_text(selection: FigureDataSelectionState, dim: str) -> str:
    width_key = selection_width_key(dim)
    if width_key in selection.qsel:
        return erlab.interactive.utils._parse_single_arg(selection.qsel[width_key])
    return ""


def selection_value_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        raise FigureComposerInputError(_SELECTION_VALUE_MESSAGE)
    try:
        return _dict_from_text(f"value={stripped}", allow_slice=True)["value"]
    except FigureComposerInputError as exc:
        raise FigureComposerInputError(_SELECTION_VALUE_MESSAGE) from exc


def selection_width_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = _dict_from_text(f"value={stripped}", allow_slice=True)["value"]
    except FigureComposerInputError as exc:
        raise FigureComposerInputError(_SELECTION_WIDTH_MESSAGE) from exc
    if isinstance(value, slice):
        raise FigureComposerInputError(_SELECTION_WIDTH_MESSAGE)
    return value


def selection_mode_combo(
    tool: FigureComposerTool,
    *,
    current: str | None,
    mixed: bool,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox(parent)
    tool._mark_editor_control(combo)
    if mixed:
        combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
    for mode, label in _SELECTION_MODE_LABELS.items():
        combo.addItem(label, mode)
    if mixed:
        item = typing.cast("typing.Any", combo.model()).item(0)
        if item is not None:
            item.setEnabled(False)
        combo.setCurrentIndex(0)
    elif current is not None:
        for index in range(combo.count()):
            if combo.itemData(index) == current:
                combo.setCurrentIndex(index)
                break
    return combo


def connect_selection_dimension_controls(
    tool: FigureComposerTool,
    *,
    dim: str,
    mode_combo: QtWidgets.QComboBox,
    value_edit: QtWidgets.QLineEdit,
    width_edit: QtWidgets.QLineEdit,
    update_dimension: Callable[[str, str, str, str], None],
) -> None:
    def set_control_visibility(mode: str) -> None:
        value_edit.setVisible(mode in _SELECTION_VALUE_MODES)
        width_edit.setVisible(mode == "qsel")

    def current_value_text() -> str:
        return value_edit.text()

    def current_width_text() -> str:
        return width_edit.text()

    def clear_hidden_controls(mode: str) -> None:
        if mode not in _SELECTION_VALUE_MODES:
            value_edit.clear()
            value_edit.setModified(False)
            tool._clear_editor_input_error(value_edit)
        if mode != "qsel":
            width_edit.clear()
            width_edit.setModified(False)
            tool._clear_editor_input_error(width_edit)

    def mode_changed(value: typing.Any) -> None:
        mode = value if isinstance(value, str) else ""
        set_control_visibility(mode)
        clear_hidden_controls(mode)
        value_mode = mode in _SELECTION_VALUE_MODES
        if value_mode:
            if current_value_text().strip():
                update_dimension(dim, mode, current_value_text(), current_width_text())
            return
        update_dimension(dim, mode, "", "")

    def value_changed(text: str) -> None:
        mode = mode_combo.currentData()
        if isinstance(mode, str) and mode in _SELECTION_VALUE_MODES:
            update_dimension(dim, mode, text, current_width_text())

    def width_changed(text: str) -> None:
        mode = mode_combo.currentData()
        if isinstance(mode, str) and mode == "qsel":
            update_dimension(dim, mode, current_value_text(), text)

    ComboBoxDataControlAdapter(mode_combo).connect_commit(
        tool._connect_editor_signal,
        mode_changed,
    )
    tool._connect_line_edit_finished(value_edit, value_changed)
    tool._connect_line_edit_finished(width_edit, width_changed)


def add_selection_dimension_rows(
    tool: FigureComposerTool,
    *,
    operation: FigureOperationState,
    layout: QtWidgets.QFormLayout,
    page: QtWidgets.QWidget,
    dimensions: Sequence[str],
    selection_getter: Callable[[FigureOperationState], FigureDataSelectionState],
    update_dimension: Callable[[str, str, str, str], None],
    object_prefix: str,
    property_prefix: str,
) -> None:
    if not dimensions:
        dimensions_message = QtWidgets.QLabel("No dimensions.", page)
        dimensions_message.setObjectName(f"{object_prefix}DimensionsMessage")
        layout.addRow("Dimensions", dimensions_message)
        return

    selection = selection_getter(operation)
    for dim_index, dim_name in enumerate(dimensions):

        def mode_getter(target: FigureOperationState, dim_name: str = dim_name) -> str:
            return selection_dim_mode(selection_getter(target), dim_name)

        def value_getter(target: FigureOperationState, dim_name: str = dim_name) -> str:
            return selection_dim_value_text(selection_getter(target), dim_name)

        def width_getter(target: FigureOperationState, dim_name: str = dim_name) -> str:
            return selection_dim_width_text(selection_getter(target), dim_name)

        mode_mixed = tool._batch_is_mixed(operation, mode_getter)
        value_text, value_mixed = tool._batch_text(operation, value_getter, str)
        width_text, width_mixed = tool._batch_text(operation, width_getter, str)
        current_mode = None if mode_mixed else selection_dim_mode(selection, dim_name)
        row = QtWidgets.QWidget(page)
        row.setObjectName(f"{object_prefix}DimRow{dim_index}")
        row.setProperty(f"{property_prefix}_dim", dim_name)
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        mode_combo = selection_mode_combo(
            tool,
            current=current_mode,
            mixed=mode_mixed,
            parent=row,
        )
        mode_combo.setObjectName(f"{object_prefix}ModeCombo{dim_index}")
        mode_combo.setProperty(f"{property_prefix}_dim", dim_name)
        mode_combo.setToolTip(_SELECTION_MODE_TOOLTIP)
        value_edit = tool._line_edit(value_text, parent=row)
        value_edit.setObjectName(f"{object_prefix}ValueEdit{dim_index}")
        value_edit.setProperty(f"{property_prefix}_dim", dim_name)
        value_edit.setProperty(f"{property_prefix}_selection_field", "value")
        value_edit.setPlaceholderText("value")
        value_edit.setToolTip(_SELECTION_VALUE_TOOLTIP)
        value_edit.setVisible(current_mode in _SELECTION_VALUE_MODES)
        tool._apply_mixed_line_edit(value_edit, value_mixed)
        width_edit = tool._line_edit(width_text, parent=row)
        width_edit.setObjectName(f"{object_prefix}WidthEdit{dim_index}")
        width_edit.setProperty(f"{property_prefix}_dim", dim_name)
        width_edit.setProperty(f"{property_prefix}_selection_field", "width")
        width_edit.setPlaceholderText("width")
        width_edit.setToolTip(_SELECTION_WIDTH_TOOLTIP)
        width_edit.setVisible(current_mode == "qsel")
        tool._apply_mixed_line_edit(width_edit, width_mixed)
        connect_selection_dimension_controls(
            tool,
            dim=dim_name,
            mode_combo=mode_combo,
            value_edit=value_edit,
            width_edit=width_edit,
            update_dimension=update_dimension,
        )

        row_layout.addWidget(mode_combo)
        row_layout.addWidget(value_edit, 1)
        row_layout.addWidget(width_edit, 1)
        tool._add_form_row(
            layout,
            dim_name,
            row,
            "Choose how this dimension is prepared before plotting.",
        )
