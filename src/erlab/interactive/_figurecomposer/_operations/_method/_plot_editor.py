"""Qt editor for DataArray-backed method plot values."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtWidgets

from erlab.interactive._figurecomposer._model._state import (
    FigureMethodPlotValueState,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._method._catalog import (
    MethodSpec,
    _method_spec,
)
from erlab.interactive._figurecomposer._operations._method._plot_data import (
    _PLOT_DATA_AXES,
    _default_plot_value_state,
    _plot_axis_field,
    _plot_axis_label,
    _plot_axis_required,
    _plot_axis_source,
    _plot_axis_value_state,
    _plot_method_call_name,
    _plot_value_options,
)
from erlab.interactive._figurecomposer._operations._method._state import (
    _format_literal_value,
    _is_axes_errorbar_method,
    _method_args,
)
from erlab.interactive._figurecomposer._text import (
    _format_literal_sequence,
    _literal_from_text,
    _literal_sequence_from_text,
)
from erlab.interactive._figurecomposer._ui._editor_controls import (
    MIXED_VALUE,
    MIXED_VALUES_TEXT,
    ComboBoxDataControlAdapter,
)
from erlab.interactive._figurecomposer._ui._source_inspector import source_value_tooltip

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import xarray as xr

    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


def _plot_x_arg_value(
    operation: FigureOperationState, spec: MethodSpec
) -> tuple[typing.Any, ...] | None:
    args = _method_args(operation, spec)
    if len(args) < 2:
        return None
    return typing.cast("tuple[typing.Any, ...]", args[0])


def _plot_y_arg_value(
    operation: FigureOperationState, spec: MethodSpec
) -> tuple[typing.Any, ...]:
    args = _method_args(operation, spec)
    if len(args) >= 2:
        return typing.cast("tuple[typing.Any, ...]", args[1])
    if args:
        return typing.cast("tuple[typing.Any, ...]", args[0])
    if spec.default_args:
        return typing.cast("tuple[typing.Any, ...]", spec.default_args[0])
    return ()


def _format_plot_sequence(value: typing.Any) -> str:
    if isinstance(value, (list, tuple)):
        return _format_literal_sequence(value)
    return _format_literal_value(value)


def _format_optional_plot_sequence(value: typing.Any) -> str:
    return "" if value is None else _format_plot_sequence(value)


_PLOT_DATA_MODE_LABELS = {
    "entered": "Enter values",
    "from_data": "Pick from data",
}


def _plot_method_object_prefix(spec: MethodSpec) -> str:
    return "Errorbar" if _is_axes_errorbar_method(spec) else "Plot"


def _plot_axis_object_part(axis: _PLOT_DATA_AXES) -> str:
    return {
        "x": "X",
        "y": "Y",
        "xerr": "XError",
        "yerr": "YError",
    }[axis]


def _plot_data_mode_text(mode: str) -> str:
    return _PLOT_DATA_MODE_LABELS.get(mode, _PLOT_DATA_MODE_LABELS["entered"])


def _plot_data_mode_combo(
    editor: FigureOperationEditor,
    current: str | None,
    changed: Callable[[str], None],
    *,
    parent: QtWidgets.QWidget | None,
    mixed: bool = False,
) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox(parent or editor)
    editor.mark_control(combo)
    if mixed:
        combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
    for mode, text in _PLOT_DATA_MODE_LABELS.items():
        combo.addItem(text, mode)
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
    ComboBoxDataControlAdapter(combo).connect_commit(
        editor.connect_signal,
        lambda value: changed(str(value)),
    )
    return combo


def _build_plot_data_args_editor(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    spec: MethodSpec,
    layout: QtWidgets.QFormLayout,
) -> None:
    mode_mixed = editor.batch_is_mixed(
        operation, lambda target: target.method_plot_data_mode
    )
    mode_combo = _plot_data_mode_combo(
        editor,
        None if mode_mixed else operation.method_plot_data_mode,
        lambda mode: _update_current_plot_data_mode(editor, mode),
        parent=layout.parentWidget(),
        mixed=mode_mixed,
    )
    mode_combo.setObjectName("figureComposerAxesMethodPlotDataModeCombo")
    editor.add_form_row(
        layout,
        "Plot data",
        mode_combo,
        f"Choose whether {_plot_method_call_name(spec)} receives entered values "
        "or values picked from available DataArrays.",
    )
    if mode_mixed:
        return
    if operation.method_plot_data_mode == "from_data":
        _build_picked_plot_data_args_editor(editor, operation, spec, layout)
        return
    _build_entered_plot_data_args_editor(editor, operation, spec, layout)


def _build_entered_plot_data_args_editor(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    spec: MethodSpec,
    layout: QtWidgets.QFormLayout,
) -> None:
    x_required = _plot_axis_required(spec, "x")
    x_text, x_mixed = editor.batch_text(
        operation,
        lambda target: _plot_x_arg_value(target, spec),
        _format_plot_sequence if x_required else _format_optional_plot_sequence,
    )
    x_edit = editor.line_edit(x_text, parent=layout.parentWidget())
    editor.apply_mixed_line_edit(x_edit, x_mixed)
    prefix = _plot_method_object_prefix(spec)
    call_name = _plot_method_call_name(spec)
    x_edit.setObjectName(f"figureComposerAxesMethod{prefix}XEdit")
    editor.connect_line_edit_finished(
        x_edit,
        lambda text: _update_current_plot_data_arg(editor, "x", text),
    )
    x_tooltip = (
        f"Required entered x sequence passed to {call_name}."
        if x_required
        else "Optional entered x sequence.\nLeave blank to use default x positions."
    )
    editor.add_form_row(layout, "X values", x_edit, x_tooltip)
    y_text, y_mixed = editor.batch_text(
        operation,
        lambda target: _plot_y_arg_value(target, spec),
        _format_plot_sequence,
    )
    y_edit = editor.line_edit(y_text, parent=layout.parentWidget())
    editor.apply_mixed_line_edit(y_edit, y_mixed)
    y_edit.setObjectName(f"figureComposerAxesMethod{prefix}YEdit")
    editor.connect_line_edit_finished(
        y_edit,
        lambda text: _update_current_plot_data_arg(editor, "y", text),
    )
    editor.add_form_row(
        layout,
        "Y values",
        y_edit,
        f"Required entered y sequence passed to {call_name}.",
    )
    if _is_axes_errorbar_method(spec):
        _build_entered_plot_error_kwarg_editor(
            editor,
            operation,
            layout,
            key="xerr",
            label="X error",
            object_name=f"figureComposerAxesMethod{prefix}XErrorEdit",
            tooltip=f"Optional entered x error values passed as {call_name} xerr.",
        )
        _build_entered_plot_error_kwarg_editor(
            editor,
            operation,
            layout,
            key="yerr",
            label="Y error",
            object_name=f"figureComposerAxesMethod{prefix}YErrorEdit",
            tooltip=f"Optional entered y error values passed as {call_name} yerr.",
        )


def _build_entered_plot_error_kwarg_editor(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    layout: QtWidgets.QFormLayout,
    *,
    key: typing.Literal["xerr", "yerr"],
    label: str,
    object_name: str,
    tooltip: str,
) -> None:
    text, mixed = editor.batch_text(
        operation,
        lambda target: target.method_kwargs.get(key),
        _format_literal_value,
    )
    edit = editor.line_edit(text, parent=layout.parentWidget())
    editor.apply_mixed_line_edit(edit, mixed)
    edit.setObjectName(object_name)
    editor.connect_line_edit_finished(
        edit,
        lambda value: _update_current_plot_error_kwarg(editor, key, value),
    )
    editor.add_form_row(layout, label, edit, tooltip)


def _build_picked_plot_data_args_editor(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    spec: MethodSpec,
    layout: QtWidgets.QFormLayout,
) -> None:
    _build_picked_plot_data_row(editor, operation, spec, layout, axis="x")
    _build_picked_plot_data_row(editor, operation, spec, layout, axis="y")
    if _is_axes_errorbar_method(spec):
        _build_picked_plot_data_row(editor, operation, spec, layout, axis="xerr")
        _build_picked_plot_data_row(editor, operation, spec, layout, axis="yerr")


def _plot_value_combo_data(
    state: FigureMethodPlotValueState | None,
) -> tuple[str, str | None] | None:
    if state is None:
        return None
    return (state.kind, state.name if state.kind == "coord" else None)


def _plot_value_combo_data_parts(value: typing.Any) -> tuple[str, str | None]:
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and value[0] in {"data", "coord"}
    ):
        return str(value[0]), None if value[1] is None else str(value[1])
    raise ValueError(f"Unknown plot value selection: {value!r}")


def _plot_value_display(
    state: FigureMethodPlotValueState | None,
) -> str:
    if state is None:
        return "Choose values"
    if state.kind == "data":
        return "Data values"
    return state.name or "Missing coordinate"


def _plot_source_combo(
    editor: FigureOperationEditor,
    current: str | None,
    changed: Callable[[str | None], None],
    *,
    axis: _PLOT_DATA_AXES,
    parent: QtWidgets.QWidget | None,
    allow_none: bool,
    mixed: bool = False,
) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox(parent or editor)
    editor.mark_control(combo)
    if mixed:
        combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
    if allow_none:
        combo.addItem(f"No {_plot_axis_label(axis)} DataArray", None)
    elif current is None:
        combo.addItem(f"Choose {_plot_axis_label(axis)} DataArray", None)
    source_names = editor.context.source_names()
    for source in source_names:
        combo.addItem(editor.source_display_name(source), source)
        combo.setItemData(
            combo.count() - 1,
            editor.source_tooltip(source),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
    if current is not None and current not in source_names and not mixed:
        combo.addItem(editor.source_display_name(current), current)
        combo.setItemData(
            combo.count() - 1,
            editor.source_tooltip(current),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
    if mixed:
        item = typing.cast("typing.Any", combo.model()).item(0)
        if item is not None:
            item.setEnabled(False)
        combo.setCurrentIndex(0)
    else:
        for index in range(combo.count()):
            if combo.itemData(index) == current:
                combo.setCurrentIndex(index)
                break
    combo.setEnabled(bool(source_names) or current is not None or allow_none)
    combo.setAccessibleName(f"{_plot_axis_label(axis)} DataArray")
    combo.setProperty("figureComposerPlotDataRole", f"{axis}_source")
    combo.setToolTip(
        _plot_source_combo_tooltip(
            axis,
            has_sources=bool(source_names),
            required=not allow_none,
        )
    )
    ComboBoxDataControlAdapter(combo).connect_commit(
        editor.connect_signal,
        lambda value: changed(typing.cast("str | None", value)),
    )
    return combo


def _plot_values_combo(
    editor: FigureOperationEditor,
    source: str | None,
    current: FigureMethodPlotValueState | None,
    changed: Callable[[typing.Any], None],
    *,
    axis: _PLOT_DATA_AXES,
    parent: QtWidgets.QWidget | None,
    allow_none: bool,
    mixed: bool = False,
    enabled: bool = True,
) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox(parent or editor)
    editor.mark_control(combo)
    if mixed:
        combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
    if allow_none:
        combo.addItem(f"No {_plot_axis_label(axis)} values", None)
    elif current is None:
        combo.addItem(f"Choose {_plot_axis_label(axis)} values", None)
    options = _plot_value_options(editor.context.source_data, source)
    for text, value in options:
        combo.addItem(text, value)
        combo.setItemData(
            combo.count() - 1,
            _plot_values_item_tooltip(editor.context.source_data, source, axis, value),
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
    current_data = _plot_value_combo_data(current)
    if (
        current_data is not None
        and current_data not in {value for _text, value in options}
        and not mixed
    ):
        combo.addItem(_plot_value_display(current), current_data)
    if mixed:
        item = typing.cast("typing.Any", combo.model()).item(0)
        if item is not None:
            item.setEnabled(False)
        combo.setCurrentIndex(0)
    else:
        for index in range(combo.count()):
            if combo.itemData(index) == current_data:
                combo.setCurrentIndex(index)
                break
    combo.setEnabled(enabled and (source is not None or allow_none))
    combo.setAccessibleName(f"{_plot_axis_label(axis)} values")
    combo.setProperty("figureComposerPlotDataRole", f"{axis}_values")
    combo.setToolTip(
        _plot_values_combo_tooltip(
            axis,
            source=source,
            value_options_match=enabled,
            required=not allow_none,
        )
    )
    ComboBoxDataControlAdapter(combo).connect_commit(
        editor.connect_signal,
        changed,
    )
    return combo


def _plot_source_combo_tooltip(
    axis: _PLOT_DATA_AXES, *, has_sources: bool, required: bool = False
) -> str:
    if not has_sources:
        return "No Figure Composer DataArrays are available."
    if axis == "x" and not required:
        return (
            "Choose the DataArray for optional x values. "
            "No X DataArray uses default x positions."
        )
    if not required:
        return (
            f"Choose the DataArray that supplies optional "
            f"{_plot_axis_label(axis).lower()} values."
        )
    return (
        f"Choose the DataArray that supplies required "
        f"{_plot_axis_label(axis).lower()} values."
    )


def _plot_values_combo_tooltip(
    axis: _PLOT_DATA_AXES,
    *,
    source: str | None,
    value_options_match: bool,
    required: bool = False,
) -> str:
    if not value_options_match:
        return (
            f"{_plot_axis_label(axis)} values are disabled because selected plot "
            "steps have different available choices."
        )
    if source is None:
        if axis == "x" and not required:
            return (
                "Use default x positions, or choose an X DataArray to pick "
                "data values or a coordinate."
            )
        if not required:
            return f"Leave {_plot_axis_label(axis)} blank or choose a DataArray."
        return (
            f"Choose a {_plot_axis_label(axis)} DataArray before choosing "
            f"{_plot_axis_label(axis).lower()} values."
        )
    if axis == "x" and not required:
        return (
            "Choose optional x values from the selected DataArray: data values "
            "or a 1D coordinate."
        )
    if not required:
        return (
            f"Choose optional {_plot_axis_label(axis).lower()} values from the "
            "selected DataArray: data values or a 1D coordinate."
        )
    return (
        f"Choose required {_plot_axis_label(axis).lower()} values from the "
        "selected DataArray: data values or a 1D coordinate."
    )


def _plot_values_item_tooltip(
    source_data: Mapping[str, xr.DataArray],
    source: str | None,
    axis: _PLOT_DATA_AXES,
    value: tuple[str, str | None],
) -> str:
    return source_value_tooltip(
        None if source is None else source_data.get(source),
        value,
        axis=axis,
    )


def _plot_value_options_for_target(
    source_data: Mapping[str, xr.DataArray],
    target: FigureOperationState,
    axis: _PLOT_DATA_AXES,
) -> tuple[tuple[str, tuple[str, str | None]], ...]:
    return _plot_value_options(source_data, _plot_axis_source(target, axis))


def _build_picked_plot_data_row(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    spec: MethodSpec,
    layout: QtWidgets.QFormLayout,
    *,
    axis: _PLOT_DATA_AXES,
) -> None:
    current = _plot_axis_value_state(operation, axis)
    current_source = None if current is None else current.source
    required = _plot_axis_required(spec, axis)
    prefix = _plot_method_object_prefix(spec)
    object_part = _plot_axis_object_part(axis)
    source_mixed = editor.batch_is_mixed(
        operation, lambda target: _plot_axis_source(target, axis)
    )
    value_mixed = editor.batch_is_mixed(
        operation,
        lambda target: _plot_value_combo_data(_plot_axis_value_state(target, axis)),
    )
    value_options_match = editor.batch_options_match(
        operation,
        lambda target: _plot_value_options_for_target(
            editor.context.source_data, target, axis
        ),
    )
    container = QtWidgets.QWidget(layout.parentWidget())
    row_layout = QtWidgets.QHBoxLayout(container)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)
    source_combo = _plot_source_combo(
        editor,
        None if source_mixed else current_source,
        lambda source: _update_current_plot_value_source(editor, axis, source),
        axis=axis,
        parent=container,
        allow_none=not required,
        mixed=source_mixed,
    )
    source_combo.setObjectName(
        f"figureComposerAxesMethod{prefix}{object_part}SourceCombo"
    )
    values_combo = _plot_values_combo(
        editor,
        None if source_mixed else current_source,
        None if value_mixed else current,
        lambda value: _update_current_plot_value_selection(editor, axis, value),
        axis=axis,
        parent=container,
        allow_none=not required,
        mixed=value_mixed,
        enabled=not source_mixed and value_options_match,
    )
    values_combo.setObjectName(
        f"figureComposerAxesMethod{prefix}{object_part}ValuesCombo"
    )
    if not value_options_match:
        values_combo.setToolTip(
            _plot_values_combo_tooltip(
                axis,
                source=None if source_mixed else current_source,
                value_options_match=False,
                required=required,
            )
        )
    row_layout.addWidget(source_combo, 1)
    row_layout.addWidget(values_combo, 1)
    tooltip = (
        f"Required {_plot_axis_label(axis).lower()} values picked from a DataArray."
        if required
        else (
            f"Optional {_plot_axis_label(axis).lower()} values picked from a DataArray."
        )
    )
    editor.add_form_row(layout, f"{_plot_axis_label(axis)} data", container, tooltip)


def _plot_sequence_from_text(text: str) -> tuple[typing.Any, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    return tuple(_literal_sequence_from_text(stripped))


def _plot_error_value_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if "," in stripped and stripped[0] not in "[({":
        return _literal_sequence_from_text(stripped)
    return _literal_from_text(stripped)


def _update_current_plot_data_arg(
    editor: FigureOperationEditor,
    axis: typing.Literal["x", "y"],
    text: str,
) -> None:
    stripped = text.strip()
    parsed_value = _plot_sequence_from_text(text) if stripped else ()

    def update_args(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        spec = _method_spec(operation)
        x_value = _plot_x_arg_value(operation, spec)
        y_value = _plot_y_arg_value(operation, spec)
        if axis == "x":
            x_value = (
                None
                if not stripped and not _plot_axis_required(spec, "x")
                else parsed_value
            )
        else:
            y_value = parsed_value
        args = (y_value,) if x_value is None else (x_value, y_value)
        return operation.model_copy(update={"method_args": args})

    editor.request_transform(update_args)


def _update_current_plot_error_kwarg(
    editor: FigureOperationEditor,
    key: typing.Literal["xerr", "yerr"],
    text: str,
) -> None:
    value = None if not text.strip() else _plot_error_value_from_text(text)

    def update_kwarg(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        kwargs = dict(operation.method_kwargs)
        if value is None:
            kwargs.pop(key, None)
        else:
            kwargs[key] = value
        return operation.model_copy(update={"method_kwargs": kwargs})

    editor.request_transform(update_kwarg)


def _update_current_plot_data_mode(editor: FigureOperationEditor, mode: str) -> None:
    if mode not in _PLOT_DATA_MODE_LABELS:
        return

    def update_mode(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {"method_plot_data_mode": mode}
        if mode == "from_data" and operation.method_plot_y is None:
            source_names = editor.context.source_names()
            if source_names:
                updates["method_plot_y"] = _default_plot_value_state(
                    editor.context.source_data, source_names[0]
                )
        return operation.model_copy(update=updates)

    editor.request_transform(update_mode, rebuild_editor=True)


def _update_current_plot_value_source(
    editor: FigureOperationEditor,
    axis: _PLOT_DATA_AXES,
    source: str | None,
) -> None:
    def update_source(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        state = (
            None
            if source is None
            else _default_plot_value_state(editor.context.source_data, source)
        )
        return operation.model_copy(
            update={
                _plot_axis_field(axis): state,
            }
        )

    editor.request_transform(update_source, rebuild_editor=True)


def _update_current_plot_value_selection(
    editor: FigureOperationEditor,
    axis: _PLOT_DATA_AXES,
    value: typing.Any,
) -> None:
    def update_value(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        current = _plot_axis_value_state(operation, axis)
        if value is None or current is None:
            state = None
        else:
            kind, name = _plot_value_combo_data_parts(value)
            state = FigureMethodPlotValueState(
                source=current.source,
                kind=typing.cast("typing.Literal['data', 'coord']", kind),
                name=name if kind == "coord" else None,
            )
        return operation.model_copy(
            update={
                _plot_axis_field(axis): state,
            }
        )

    editor.request_transform(update_value)
