"""Line/Profile operation editor, renderer, and code generation."""

from __future__ import annotations

import typing

from qtpy import QtWidgets

from erlab.interactive._figurecomposer._code import _axes_sequence_code, _selection_code
from erlab.interactive._figurecomposer._line_style import (
    LINE_MARKER_OPTIONS,
    LINE_STYLE_DEFAULT_LABEL,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    line_kw_float,
    line_kw_style_value,
    line_kw_text,
    optional_positive_spinbox,
    optional_positive_spinbox_value,
    update_current_line_kw,
)
from erlab.interactive._figurecomposer._line_transform import (
    add_line_transform_controls,
    profile_transform_code_lines,
    transform_profiles,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
    _render_preview,
)
from erlab.interactive._figurecomposer._sources import (
    _available_source_dims,
    _public_source_data,
    _selected_data,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._state import (
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._text import (
    _code_kwargs,
    _dict_from_text,
    _format_dict,
    _format_string_tuple,
    _string_tuple_from_text,
)
from erlab.interactive._figurecomposer._widgets import (
    _ColorLineEditWidget,
    _ColorListEditorWidget,
)
from erlab.interactive.imagetool import provenance

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib.axes
    import xarray as xr

    from erlab.interactive._figurecomposer._state import FigureLimit
    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_LINE_PROFILE_STYLE_KEY_ALIASES = {
    "c": "color",
    "ls": "linestyle",
    "lw": "linewidth",
    "ms": "markersize",
    "mfc": "markerfacecolor",
    "mec": "markeredgecolor",
}

_LINE_REDUCE_TEXT = {
    "disabled": "Disabled",
    "coarsen": "Coarsen",
    "thin": "Thin",
    "both": "Both",
}


def _line_placement_text(placement: str) -> str:
    if placement == "one_per_axis":
        return "One profile per axis"
    return "All profiles on each axis"


def _line_placement_from_text(
    text: str,
) -> typing.Literal["all_axes", "one_per_axis"]:
    if text == "One profile per axis":
        return "one_per_axis"
    return "all_axes"


def _line_reduce_text(reduce: str) -> str:
    return _LINE_REDUCE_TEXT.get(reduce, _LINE_REDUCE_TEXT["disabled"])


def _line_reduce_from_text(
    text: str,
) -> typing.Literal["disabled", "coarsen", "thin", "both"]:
    for reduce, label in _LINE_REDUCE_TEXT.items():
        if text == label:
            return typing.cast(
                "typing.Literal['disabled', 'coarsen', 'thin', 'both']", reduce
            )
    return "disabled"


def _line_reduce_active(operation: FigureOperationState) -> bool:
    return operation.line_iter_dim is not None and operation.line_reduce != "disabled"


def _line_choice_data(
    tool: FigureComposerTool, operation: FigureOperationState, *, values: bool
) -> xr.DataArray | None:
    if operation.line_source is None:
        return None
    data = tool._source_data.get(operation.line_source)
    if data is None:
        return None
    data = _public_source_data(data)
    if operation.line_selection:
        data = data.qsel(operation.line_selection)
    data = data.squeeze(drop=True)
    if values and operation.line_y is not None and operation.line_y in data.coords:
        return data[operation.line_y].squeeze(drop=True)
    return data


def _available_line_value_names(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    data = _line_choice_data(tool, operation, values=False)
    if data is None:
        return []
    return [str(name) for name in data.coords]


def _available_line_coordinate_names(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    data = _line_choice_data(tool, operation, values=True)
    if data is None:
        return []
    profile_dims = {str(dim) for dim in data.dims}
    if operation.line_iter_dim is not None:
        profile_dims.discard(operation.line_iter_dim)
    names = [str(dim) for dim in data.dims if str(dim) in profile_dims]
    for name, coord in data.coords.items():
        name_str = str(name)
        if (
            name_str not in names
            and coord.ndim > 0
            and {str(dim) for dim in coord.dims}.issubset(profile_dims)
        ):
            names.append(name_str)
    return names


def _available_line_offset_coords(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    if operation.line_source is None or operation.line_iter_dim is None:
        return []
    data = tool._source_data.get(operation.line_source)
    if data is None:
        return []
    data = _public_source_data(data)
    if operation.line_selection:
        data = data.qsel(operation.line_selection)
    line_data = data.squeeze(drop=True)
    if operation.line_y:
        line_data = line_data[operation.line_y]
    coords: list[str] = []
    for name, coord in line_data.coords.items():
        if name == operation.line_iter_dim:
            continue
        if coord.dims == (operation.line_iter_dim,):
            coords.append(str(name))
    return coords


def _build_line_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[tuple[str, str, QtWidgets.QWidget]]:
    page, layout = tool._new_step_form_page("figureComposerLinePage")
    tool.operation_editor = page
    tool.operation_editor_layout = layout
    coordinate_options = _available_line_coordinate_names(tool, operation)
    coordinate_options_match = tool._batch_options_match(
        operation, lambda target: _available_line_coordinate_names(tool, target)
    )
    coordinate_mixed = tool._batch_is_mixed(operation, lambda target: target.line_x)
    coordinate_combo = tool._optional_name_combo(
        coordinate_options,
        None if coordinate_mixed else operation.line_x,
        "Automatic profile coordinate",
        lambda value: tool._update_current_operation(line_x=value),
        parent=page,
        mixed=coordinate_mixed,
        enabled=coordinate_options_match,
    )
    coordinate_combo.setObjectName("figureComposerProfileCoordinateCombo")
    tool._add_form_row(
        tool.operation_editor_layout,
        "Profile coordinate",
        coordinate_combo,
        "Coordinate along each profile.\n"
        "Automatic: use the only remaining dimension.\n"
        "Data values axis decides where this coordinate is drawn."
        + (
            "\nDisabled while selected steps have different valid choices."
            if not coordinate_options_match
            else ""
        ),
    )

    value_options = _available_line_value_names(tool, operation)
    value_options_match = tool._batch_options_match(
        operation, lambda target: _available_line_value_names(tool, target)
    )
    value_mixed = tool._batch_is_mixed(operation, lambda target: target.line_y)
    values_combo = tool._optional_name_combo(
        value_options,
        None if value_mixed else operation.line_y,
        "Data array values",
        lambda value: tool._update_current_operation_rebuild(line_y=value),
        parent=page,
        mixed=value_mixed,
        enabled=value_options_match,
    )
    values_combo.setObjectName("figureComposerProfileValuesCombo")
    tool._add_form_row(
        tool.operation_editor_layout,
        "Profile values",
        values_combo,
        "Values plotted from the selected profile.\n"
        "Data array values: use the selected data itself.\n"
        "Choose a coordinate to plot that coordinate instead."
        + (
            "\nDisabled while selected steps have different valid choices."
            if not value_options_match
            else ""
        ),
    )

    labels_text, labels_mixed = tool._batch_text(
        operation,
        lambda target: target.line_labels,
        lambda value: _format_string_tuple(typing.cast("tuple[str, ...]", value)),
    )
    labels_edit = tool._line_edit(labels_text)
    tool._apply_mixed_line_edit(labels_edit, labels_mixed)
    labels_edit.setObjectName("figureComposerLineLabelsEdit")
    tool._connect_line_edit_finished(
        labels_edit,
        lambda text: _update_current_line_labels(tool, text),
    )
    tool._add_form_row(
        tool.operation_editor_layout,
        "Legend labels",
        labels_edit,
        "Optional legend labels.\n"
        "Use one value for every profile, or one value per profile.",
    )

    colors_text, colors_mixed = tool._batch_text(
        operation,
        lambda target: target.line_colors,
        lambda value: _format_string_tuple(typing.cast("tuple[str, ...]", value)),
    )
    colors_widget = _ColorListEditorWidget(
        _string_tuple_from_text(colors_text), parent=page
    )
    colors_widget.setMainEditObjectName("figureComposerLineColorsEdit")
    if colors_mixed:
        colors_widget.setMixedPlaceholder("(multiple values)")
    tool._connect_value_signal(
        colors_widget,
        colors_widget.colorsChanged,
        lambda colors: tuple(colors),
        lambda colors: tool._update_current_operation(line_colors=colors),
        unchanged_mixed=colors_widget.batchUnchanged,
    )
    tool._add_form_row(
        tool.operation_editor_layout,
        "Colors",
        colors_widget,
        "Optional Matplotlib colors.\n"
        "Use one value for every profile, or one value per profile.",
    )

    _add_line_style_controls(tool, operation, page)

    selection_text, selection_mixed = tool._batch_text(
        operation, lambda target: target.line_selection, _format_dict
    )
    selection_edit = tool._line_edit(selection_text)
    tool._apply_mixed_line_edit(selection_edit, selection_mixed)
    selection_edit.setObjectName("figureComposerLineSelectionEdit")
    tool._connect_line_edit_finished(
        selection_edit,
        lambda text: tool._update_current_operation(
            line_selection=_dict_from_text(text, allow_slice=True)
        ),
    )
    tool._add_form_row(
        tool.operation_editor_layout,
        "Data selection",
        selection_edit,
        "Selection kwargs passed to qsel.\n"
        "Use dimension keys such as kx=slice(-1, 1) or beta=0.",
    )

    iter_dim_options = [
        "",
        *_available_source_dims(tool._source_data, (operation.line_source or "",)),
    ]
    iter_dim_options_match = tool._batch_options_match(
        operation,
        lambda target: [
            "",
            *_available_source_dims(tool._source_data, (target.line_source or "",)),
        ],
    )
    iter_dim_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_iter_dim
    )
    iter_dim_combo = tool._combo(
        iter_dim_options,
        None if iter_dim_mixed else operation.line_iter_dim or "",
        lambda text: _update_current_line_iter_dim(tool, text),
        parent=page,
        mixed=iter_dim_mixed,
        enabled=iter_dim_options_match,
    )
    tool._add_form_row(
        tool.operation_editor_layout,
        "One profile per",
        iter_dim_combo,
        "Optional dimension used to split selected data into one profile per axis."
        + (
            "\nDisabled while selected steps have different valid choices."
            if not iter_dim_options_match
            else ""
        ),
    )
    if _line_reduce_controls_visible(tool, operation):
        _add_line_reduce_controls(tool, operation, page)

    placement_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_placement
    )
    placement_combo = tool._combo(
        ["All profiles on each axis", "One profile per axis"],
        None if placement_mixed else _line_placement_text(operation.line_placement),
        lambda text: tool._update_current_operation(
            line_placement=_line_placement_from_text(text)
        ),
        parent=page,
        mixed=placement_mixed,
    )
    placement_combo.setObjectName("figureComposerProfilePlacementCombo")
    tool._add_form_row(
        tool.operation_editor_layout,
        "Profile placement",
        placement_combo,
        "Choose how extracted profiles map onto selected axes.\n"
        "All profiles: draw every profile on every target axis.\n"
        "One per axis: pair profiles with axes in order.",
    )

    values_axis_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_values_axis
    )
    values_axis_combo = tool._combo(
        ["y", "x"],
        None if values_axis_mixed else operation.line_values_axis,
        lambda text: tool._update_current_operation(line_values_axis=text),
        parent=page,
        mixed=values_axis_mixed,
    )
    values_axis_combo.setObjectName("figureComposerDataValuesAxisCombo")
    tool._add_form_row(
        tool.operation_editor_layout,
        "Data values axis",
        values_axis_combo,
        "Axis that receives profile data values.\n"
        "y: coordinate on x, values on y.\n"
        "x: values on x, coordinate on y.",
    )

    add_line_transform_controls(
        tool,
        operation,
        page,
        tool.operation_editor_layout,
        object_prefix="figureComposerLine",
        offset_coord_options=lambda target: _available_line_offset_coords(tool, target),
    )
    return [("line", "Line", page)]


def _update_current_line_iter_dim(tool: FigureComposerTool, text: str) -> None:
    updates: dict[str, typing.Any] = {"line_iter_dim": text or None}
    if not text:
        updates["line_reduce"] = "disabled"
    tool._update_current_operation_rebuild(**updates)


def _line_reduce_controls_visible(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    editable = tool._editable_operations()
    if len(editable) > 1:
        return all(target.line_iter_dim is not None for _index, target in editable)
    return operation.line_iter_dim is not None


def _add_line_reduce_controls(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
) -> None:
    reduce_mixed = tool._batch_is_mixed(operation, lambda target: target.line_reduce)
    reduce_combo = tool._combo(
        list(_LINE_REDUCE_TEXT.values()),
        None if reduce_mixed else _line_reduce_text(operation.line_reduce),
        lambda text: tool._update_current_operation_rebuild(
            line_reduce=_line_reduce_from_text(text)
        ),
        parent=page,
        mixed=reduce_mixed,
    )
    reduce_combo.setObjectName("figureComposerProfileReduceCombo")
    reduce_combo.setToolTip(
        "Reduce the One profile per axis before extracting profiles.\n"
        "Coarsen averages neighboring coordinates.\n"
        "Thin keeps every Nth coordinate."
    )

    row = QtWidgets.QWidget(page)
    row_layout = QtWidgets.QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)
    row_layout.addWidget(reduce_combo)
    if not reduce_mixed and operation.line_reduce in {"coarsen", "both"}:
        _add_line_reduce_factor_control(
            tool,
            operation,
            page,
            row_layout,
            label="Coarsen",
            field="line_reduce_coarsen",
            object_name="figureComposerProfileReduceCoarsenSpin",
            tooltip=(
                "Number of coordinate points averaged into one profile.\n"
                "Uses boundary='trim' and mean() on the coarsened data."
            ),
        )
    if not reduce_mixed and operation.line_reduce in {"thin", "both"}:
        _add_line_reduce_factor_control(
            tool,
            operation,
            page,
            row_layout,
            label="Thin",
            field="line_reduce_thin",
            object_name="figureComposerProfileReduceThinSpin",
            tooltip="Keep every Nth coordinate after any coarsening step.",
        )
    row_layout.addStretch(1)
    tool._add_form_row(
        tool.operation_editor_layout,
        "Reduce",
        row,
        "Reduce the One profile per axis before profile extraction.",
    )


def _add_line_reduce_factor_control(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QHBoxLayout,
    *,
    label: str,
    field: typing.Literal["line_reduce_coarsen", "line_reduce_thin"],
    object_name: str,
    tooltip: str,
) -> None:
    factor_mixed = tool._batch_is_mixed(
        operation, lambda target: getattr(target, field)
    )
    label_widget = QtWidgets.QLabel(label, page)
    label_widget.setToolTip(tooltip)
    spin = QtWidgets.QSpinBox(page)
    spin.setRange(2, 1_000_000)
    spin.setKeyboardTracking(False)
    spin.setValue(int(getattr(operation, field)))
    spin.setObjectName(object_name)
    spin.setToolTip(tooltip)
    tool._connect_value_signal(
        spin,
        spin.valueChanged,
        int,
        lambda value: tool._update_current_operation(**{field: value}),
    )
    layout.addWidget(label_widget)
    layout.addWidget(tool._mixed_value_widget(spin, mixed=factor_mixed, parent=page))


def _add_line_style_controls(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
) -> None:
    line_style_mixed = tool._batch_is_mixed(
        operation, lambda target: line_kw_style_value(target, "linestyle", "ls")
    )
    line_style_combo = tool._optional_name_combo(
        LINE_STYLE_OPTIONS,
        None if line_style_mixed else line_kw_style_value(operation, "linestyle", "ls"),
        LINE_STYLE_DEFAULT_LABEL,
        lambda text: update_current_line_kw(tool, "linestyle", text, aliases=("ls",)),
        parent=page,
        mixed=line_style_mixed,
    )
    line_style_combo.setObjectName("figureComposerLineStyleCombo")

    line_width_mixed = tool._batch_is_mixed(
        operation, lambda target: line_kw_text(target, "linewidth", "lw")
    )
    line_width_spin = optional_positive_spinbox(
        None if line_width_mixed else line_kw_float(operation, "linewidth", "lw"),
        parent=page,
    )
    tool._connect_editor_signal(
        line_width_spin,
        line_width_spin.valueChanged,
        lambda value: update_current_line_kw(
            tool,
            "linewidth",
            optional_positive_spinbox_value(value),
            aliases=("lw",),
        ),
    )
    line_width_spin.setObjectName("figureComposerLineWidthSpin")
    line_width_row_widget = tool._mixed_value_widget(
        line_width_spin, mixed=line_width_mixed, parent=page
    )
    tool._add_compound_form_row(
        tool.operation_editor_layout,
        "Line",
        (
            (
                "Style",
                line_style_combo,
                "Matplotlib linestyle for the extracted profiles.",
            ),
            (
                "Width",
                line_width_row_widget,
                "Matplotlib linewidth for the extracted profiles.",
            ),
        ),
        "Line style controls for the extracted profiles.",
    )

    marker_mixed = tool._batch_is_mixed(
        operation, lambda target: line_kw_style_value(target, "marker")
    )
    marker_combo = tool._optional_name_combo(
        LINE_MARKER_OPTIONS,
        None if marker_mixed else line_kw_style_value(operation, "marker"),
        LINE_STYLE_DEFAULT_LABEL,
        lambda text: update_current_line_kw(tool, "marker", text),
        parent=page,
        mixed=marker_mixed,
    )
    marker_combo.setObjectName("figureComposerLineMarkerCombo")

    marker_size_mixed = tool._batch_is_mixed(
        operation, lambda target: line_kw_text(target, "markersize", "ms")
    )
    marker_size_spin = optional_positive_spinbox(
        None if marker_size_mixed else line_kw_float(operation, "markersize", "ms"),
        parent=page,
    )
    tool._connect_editor_signal(
        marker_size_spin,
        marker_size_spin.valueChanged,
        lambda value: update_current_line_kw(
            tool,
            "markersize",
            optional_positive_spinbox_value(value),
            aliases=("ms",),
        ),
    )
    marker_size_spin.setObjectName("figureComposerLineMarkerSizeSpin")
    marker_size_row_widget = tool._mixed_value_widget(
        marker_size_spin, mixed=marker_size_mixed, parent=page
    )
    tool._add_compound_form_row(
        tool.operation_editor_layout,
        "Marker",
        (
            (
                "Style",
                marker_combo,
                "Matplotlib marker style for the extracted profiles.",
            ),
            (
                "Size",
                marker_size_row_widget,
                "Matplotlib marker size for the extracted profiles.",
            ),
        ),
        "Marker style controls for the extracted profiles.",
    )

    marker_face_text, marker_face_mixed = tool._batch_text(
        operation,
        lambda target: line_kw_text(target, "markerfacecolor", "mfc"),
        str,
    )
    marker_inherited_color = (
        operation.line_colors[0]
        if operation.line_colors
        else line_kw_text(operation, "color", "c") or None
    )
    marker_face_edit = _ColorLineEditWidget(
        marker_face_text,
        parent=page,
        inherited_color=marker_inherited_color,
    )
    marker_face_edit.setLineEditObjectName("figureComposerLineMarkerFaceColorEdit")
    marker_face_edit.setColorButtonObjectName("figureComposerLineMarkerFaceColorButton")
    tool._apply_mixed_line_edit(marker_face_edit.line_edit, marker_face_mixed)
    tool._connect_value_signal(
        marker_face_edit,
        marker_face_edit.editingFinished,
        marker_face_edit.text,
        lambda text: update_current_line_kw(
            tool,
            "markerfacecolor",
            color_kw_value_from_text(text),
            aliases=("mfc",),
        ),
        unchanged_mixed=lambda: tool._line_edit_batch_unchanged(
            marker_face_edit.line_edit
        ),
    )
    marker_edge_text, marker_edge_mixed = tool._batch_text(
        operation,
        lambda target: line_kw_text(target, "markeredgecolor", "mec"),
        str,
    )
    marker_edge_edit = _ColorLineEditWidget(
        marker_edge_text,
        parent=page,
        inherited_color=marker_inherited_color,
    )
    marker_edge_edit.setLineEditObjectName("figureComposerLineMarkerEdgeColorEdit")
    marker_edge_edit.setColorButtonObjectName("figureComposerLineMarkerEdgeColorButton")
    tool._apply_mixed_line_edit(marker_edge_edit.line_edit, marker_edge_mixed)
    tool._connect_value_signal(
        marker_edge_edit,
        marker_edge_edit.editingFinished,
        marker_edge_edit.text,
        lambda text: update_current_line_kw(
            tool,
            "markeredgecolor",
            color_kw_value_from_text(text),
            aliases=("mec",),
        ),
        unchanged_mixed=lambda: tool._line_edit_batch_unchanged(
            marker_edge_edit.line_edit
        ),
    )
    tool._add_compound_form_row(
        tool.operation_editor_layout,
        "Marker colors",
        (
            (
                "Face",
                marker_face_edit,
                "Matplotlib marker face color for the extracted profiles.",
            ),
            (
                "Edge",
                marker_edge_edit,
                "Matplotlib marker edge color for the extracted profiles.",
            ),
        ),
        "Marker face and edge colors for the extracted profiles.",
    )


def _update_current_line_labels(tool: FigureComposerTool, text: str) -> None:
    editable = tool._editable_operations()
    if not editable:
        return
    labels = _string_tuple_from_text(text)
    selected_ids = {operation.operation_id for _index, operation in editable}
    original_operations = {
        operation.operation_id: operation for _index, operation in editable
    }
    operations = list(tool._recipe.operations)
    newly_labeled_groups: dict[
        tuple[tuple[tuple[int, int], ...], tuple[str, ...], str],
        tuple[int, FigureOperationState],
    ] = {}
    for index, operation in enumerate(tuple(operations)):
        if operation.operation_id not in selected_ids:
            continue
        updated = operation.model_copy(update={"line_labels": labels})
        operations[index] = updated
        original_operation = original_operations.get(operation.operation_id)
        if original_operation is None or not labels or original_operation.line_labels:
            continue
        key = _line_axes_key(updated)
        previous = newly_labeled_groups.get(key)
        if previous is None or index > previous[0]:
            newly_labeled_groups[key] = (index, updated)

    for index, operation in sorted(
        newly_labeled_groups.values(), key=lambda item: item[0], reverse=True
    ):
        legend_operation = FigureOperationState.method(
            family=FigureMethodFamily.AXES,
            name="legend",
            label="Legend",
            axes=operation.axes.model_copy(deep=True),
        )
        if not _has_later_legend_step(operations, index, legend_operation):
            operations.insert(index + 1, legend_operation)
    tool._recipe = tool._recipe.model_copy(update={"operations": tuple(operations)})
    tool._refresh_operation_list()
    tool._sync_axes_selector()
    tool._update_step_action_buttons()
    tool._refresh_step_section_button_texts()
    current = tool._current_operation()
    tool._update_source_status(current[1] if current is not None else None)
    _render_preview(tool)
    tool.sigInfoChanged.emit()


def _line_axes_key(
    operation: FigureOperationState,
) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...], str]:
    return operation.axes.axes, operation.axes.axes_ids, operation.axes.expression


def _has_later_legend_step(
    operations: list[FigureOperationState],
    index: int,
    operation: FigureOperationState,
) -> bool:
    axes = operation.axes.model_dump()
    return any(
        later.kind == FigureOperationKind.METHOD
        and later.method_family == FigureMethodFamily.AXES
        and later.method_name == "legend"
        and later.axes.model_dump() == axes
        for later in operations[index + 1 :]
    )


def _render_line(
    tool: FigureComposerTool, operation: FigureOperationState, axs: typing.Any
) -> None:
    line_items = _line_data_items(tool, operation)
    if not line_items:
        return
    axes = _iter_axes(
        _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)
    )
    if operation.line_placement == "one_per_axis":
        _render_one_profile_per_axis(tool, operation, axes, line_items)
        return
    line_items = transform_profiles(operation, line_items)
    styles = _line_styles_for_profiles(operation, len(line_items))
    for axis in axes:
        for line_data, kwargs in zip(line_items, styles, strict=True):
            if operation.line_values_axis == "x":
                coordinate = _line_coordinate(line_data, operation.line_x)
                axis.plot(line_data.values, coordinate.values, **kwargs)
            else:
                line_data.plot(ax=axis, x=operation.line_x, **kwargs)
        _apply_line_axes_limits(axis, operation)


def _render_one_profile_per_axis(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    axes: tuple[matplotlib.axes.Axes, ...],
    profiles: list[xr.DataArray],
) -> None:
    if len(axes) == 1 and len(profiles) > 1:
        axes = axes * len(profiles)
    elif len(profiles) == 1 and len(axes) > 1:
        profiles = profiles * len(axes)
    profiles = transform_profiles(operation, profiles)
    styles = _line_styles_for_profiles(operation, len(profiles))
    for axis, profile, kwargs in zip(axes, profiles, styles, strict=True):
        coordinate = _line_coordinate(profile, operation.line_x)
        if operation.line_values_axis == "x":
            axis.plot(profile.values, coordinate.values, **kwargs)
        else:
            axis.plot(coordinate.values, profile.values, **kwargs)
        _apply_line_axes_limits(axis, operation)


def _apply_line_axes_limits(
    axis: matplotlib.axes.Axes, operation: FigureOperationState
) -> None:
    if operation.xlim is not None:
        _set_axis_xlim(axis, operation.xlim)
    if operation.ylim is not None:
        _set_axis_ylim(axis, operation.ylim)


def _set_axis_xlim(axis: matplotlib.axes.Axes, limit: FigureLimit) -> None:
    if isinstance(limit, tuple):
        axis.set_xlim(*limit)
    else:
        axis.set_xlim(limit)


def _set_axis_ylim(axis: matplotlib.axes.Axes, limit: FigureLimit) -> None:
    if isinstance(limit, tuple):
        axis.set_ylim(*limit)
    else:
        axis.set_ylim(limit)


def _line_data_items(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[xr.DataArray]:
    if operation.map_selections:
        line_items = [
            selected.squeeze(drop=True)
            for selection in operation.map_selections
            if (selected := _selected_data(tool._source_data, selection)) is not None
        ]
    elif operation.line_source is not None:
        data = tool._source_data.get(operation.line_source)
        if data is None:
            return []
        data = _public_source_data(data)
        if operation.line_selection:
            data = data.qsel(operation.line_selection)
        line_data = data.squeeze(drop=True)
        if operation.line_y:
            line_data = line_data[operation.line_y]
        line_data = _reduced_line_iter_data(line_data, operation)
        if operation.line_iter_dim and operation.line_iter_dim in line_data.dims:
            line_items = [
                item.squeeze(drop=True)
                for item in line_data.transpose(operation.line_iter_dim, ...)
            ]
        else:
            line_items = [line_data]
    else:
        return []

    profiles: list[xr.DataArray] = []
    for line_data in line_items:
        if line_data.ndim != 1:
            continue
        profiles.append(line_data)
    return profiles


def _reduced_line_iter_data(
    line_data: xr.DataArray, operation: FigureOperationState
) -> xr.DataArray:
    if (
        not _line_reduce_active(operation)
        or operation.line_iter_dim not in line_data.dims
    ):
        return line_data
    data = line_data
    for reduce_operation in _line_reduce_operations(operation):
        data = reduce_operation.apply(data, parent_data=data)
    return data


def _line_reduce_operations(
    operation: FigureOperationState,
) -> tuple[provenance.CoarsenOperation | provenance.ThinOperation, ...]:
    if operation.line_iter_dim is None:
        return ()
    operations: list[provenance.CoarsenOperation | provenance.ThinOperation] = []
    if operation.line_reduce in {"coarsen", "both"}:
        operations.append(
            provenance.CoarsenOperation(
                dim={operation.line_iter_dim: operation.line_reduce_coarsen},
                boundary="trim",
                side="left",
                coord_func="mean",
                reducer="mean",
            )
        )
    if operation.line_reduce in {"thin", "both"}:
        operations.append(
            provenance.ThinOperation(
                mode="per_dim",
                factors={operation.line_iter_dim: operation.line_reduce_thin},
            )
        )
    return tuple(operations)


def _line_coordinate(line_data: xr.DataArray, dim: str | None) -> xr.DataArray:
    if dim:
        return line_data[dim]
    if line_data.ndim != 1:
        raise ValueError("Profile overlays require one-dimensional profiles")
    return line_data[line_data.dims[0]]


def _line_styles_for_profiles(
    operation: FigureOperationState, count: int
) -> tuple[dict[str, typing.Any], ...]:
    labels = _line_text_values(operation.line_labels, count, default=None)
    colors = _line_text_values(operation.line_colors, count, default=None)
    style_kwargs = _line_profile_style_kwargs(operation)
    styles: list[dict[str, typing.Any]] = []
    for label, color in zip(labels, colors, strict=True):
        kwargs = dict(style_kwargs)
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        styles.append(kwargs)
    return tuple(styles)


def _line_profile_style_kwargs(
    operation: FigureOperationState,
) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {}
    for key, value in operation.line_kw.items():
        canonical = _LINE_PROFILE_STYLE_KEY_ALIASES.get(key, key)
        if canonical == "color":
            continue
        kwargs[canonical] = value
    return kwargs


def _line_text_values(
    values: Sequence[str], count: int, *, default: str | None
) -> tuple[str | None, ...]:
    if count < 1:
        return ()
    if not values:
        return (default,) * count
    if len(values) == 1:
        return (values[0],) * count
    if len(values) != count:
        raise ValueError(
            "Profile labels and colors must be one value or one per profile"
        )
    return tuple(values)


def _line_code(tool: FigureComposerTool, operation: FigureOperationState) -> list[str]:
    if operation.map_selections:
        return _line_selection_code(tool, operation)
    if operation.line_source is None:
        return []
    source = _valid_source_variable(operation.line_source)
    if operation.line_selection:
        selection = _code_kwargs(operation.line_selection)
        lines = [f"profile_data = {source}.qsel({selection}).squeeze(drop=True)"]
    else:
        lines = [f"profile_data = {source}.squeeze(drop=True)"]
    if operation.line_y:
        lines.append(f"profile_data = profile_data[{operation.line_y!r}]")
    lines.extend(_line_reduce_code(operation, "profile_data"))
    if operation.line_iter_dim:
        transpose_code = f"profile_data.transpose({operation.line_iter_dim!r}, ...)"
        lines.append(
            f"profiles = [profile.squeeze(drop=True) for profile in {transpose_code}]"
        )
    else:
        lines.append("profiles = [profile_data]")
    if operation.line_placement == "one_per_axis":
        lines.extend(_one_profile_per_axis_code(tool, operation))
        return lines
    lines.extend(_regular_line_code(tool, operation))
    return lines


def _line_reduce_code(operation: FigureOperationState, input_name: str) -> list[str]:
    if not _line_reduce_active(operation):
        return []
    return [
        f"{input_name} = {reduce_operation.expression_code(input_name)}"
        for reduce_operation in _line_reduce_operations(operation)
    ]


def _line_selection_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    lines = ["profiles = ["]
    lines.extend(
        f"    {_selection_code(selection)}.squeeze(drop=True),"
        for selection in operation.map_selections
    )
    lines.append("]")
    if operation.line_placement == "one_per_axis":
        lines.extend(_one_profile_per_axis_code(tool, operation))
        return lines
    lines.extend(_regular_line_code(tool, operation))
    return lines


def _regular_line_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    lines: list[str] = []
    profiles = _line_data_items(tool, operation)
    lines.extend(profile_transform_code_lines(operation, profiles=profiles))
    loop_names = ["profile"]
    loop_values = ["profiles"]
    style_lines, kwargs_text = _line_style_code(
        operation,
        loop_names=loop_names,
        loop_values=loop_values,
    )
    lines.extend(style_lines)
    if len(loop_names) == 1:
        loop = "for profile in profiles:"
    else:
        loop = (
            "for "
            + ", ".join(loop_names)
            + " in zip("
            + ", ".join(loop_values)
            + ", strict=True):"
        )
    lines.append(f"for ax in {_axes_sequence_code(tool, operation.axes)}:")
    lines.append(f"    {loop}")
    coordinate = _line_coordinate_code(operation)
    if operation.line_values_axis == "x":
        call_args = f"profile, {coordinate}"
        if kwargs_text:
            call_args += f", {kwargs_text}"
        lines.append(f"        ax.plot({call_args})")
    else:
        call_args = "ax=ax"
        if operation.line_x:
            call_args += f", x={operation.line_x!r}"
        if kwargs_text:
            call_args += f", {kwargs_text}"
        lines.append(f"        profile.plot({call_args})")
    if axes_limits_code := _line_axes_limits_code(operation):
        lines.append(f"    {axes_limits_code}")
    return lines


def _line_style_code(
    operation: FigureOperationState,
    *,
    loop_names: list[str],
    loop_values: list[str],
) -> tuple[list[str], str]:
    lines: list[str] = []
    kwargs = [_code_kwargs(_line_profile_style_kwargs(operation))]
    if operation.line_labels:
        if len(operation.line_labels) == 1:
            lines.append(f"profile_label = {operation.line_labels[0]!r}")
            kwargs.append("label=profile_label")
        else:
            lines.append(f"profile_labels = {list(operation.line_labels)!r}")
            loop_names.append("label")
            loop_values.append("profile_labels")
            kwargs.append("label=label")

    line_colors = operation.line_colors
    if line_colors:
        if len(line_colors) == 1:
            lines.append(f"profile_color = {line_colors[0]!r}")
            kwargs.append("color=profile_color")
        else:
            lines.append(f"profile_colors = {list(line_colors)!r}")
            loop_names.append("color")
            loop_values.append("profile_colors")
            kwargs.append("color=color")
    return lines, ", ".join(item for item in kwargs if item)


def _one_profile_per_axis_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    lines: list[str] = []
    lines.append(f"target_axes = list({_axes_sequence_code(tool, operation.axes)})")
    lines.append("if len(target_axes) == 1 and len(profiles) > 1:")
    lines.append("    target_axes = target_axes * len(profiles)")
    lines.append("elif len(profiles) == 1 and len(target_axes) > 1:")
    lines.append("    profiles = profiles * len(target_axes)")
    profiles = _line_data_items(tool, operation)
    lines.extend(profile_transform_code_lines(operation, profiles=profiles))
    loop_names = ["ax", "profile"]
    loop_values = ["target_axes", "profiles"]
    style_lines, kwargs_text = _line_style_code(
        operation,
        loop_names=loop_names,
        loop_values=loop_values,
    )
    lines.extend(style_lines)
    coordinate = _line_coordinate_code(operation)
    lines.append(
        "for "
        + ", ".join(loop_names)
        + " in zip("
        + ", ".join(loop_values)
        + ", strict=True):"
    )
    if operation.line_values_axis == "x":
        call_args = f"profile, {coordinate}"
    else:
        call_args = f"{coordinate}, profile"
    if kwargs_text:
        call_args += f", {kwargs_text}"
    lines.append(f"    ax.plot({call_args})")
    if axes_limits_code := _line_axes_limits_code(operation):
        lines.append(f"    {axes_limits_code}")
    return lines


def _line_axes_limits_code(operation: FigureOperationState) -> str:
    kwargs: list[str] = []
    if operation.xlim is not None:
        kwargs.append(f"xlim={operation.xlim!r}")
    if operation.ylim is not None:
        kwargs.append(f"ylim={operation.ylim!r}")
    if not kwargs:
        return ""
    return "ax.set(" + ", ".join(kwargs) + ")"


def _line_coordinate_code(operation: FigureOperationState) -> str:
    if operation.line_x:
        return f"profile[{operation.line_x!r}]"
    return "profile[profile.dims[0]]"


def _create_line_operation(tool: FigureComposerTool) -> FigureOperationState:
    source_names = tool._source_names()
    first_source = source_names[0] if source_names else tool._recipe.primary_source
    return FigureOperationState.line(
        label="line",
        source=first_source,
        axes=tool._selected_axes_state(),
    ).model_copy(update=_seeded_line_operation_defaults(tool, first_source))


def _seeded_line_operation_defaults(
    tool: FigureComposerTool, source_name: str
) -> dict[str, typing.Any]:
    updates: dict[str, typing.Any] = {}
    current = tool._current_operation()
    if current is None:
        return updates
    _index, current_operation = current
    if current_operation.kind != FigureOperationKind.PLOT_SLICES:
        return updates

    updates.update(
        {
            "line_placement": "one_per_axis",
            "line_values_axis": "y",
            "line_normalize": "max",
            "line_colors": ("black",),
        }
    )
    if current_operation.slice_dim and current_operation.slice_values:
        updates["line_selection"] = {
            current_operation.slice_dim: list(current_operation.slice_values)
        }
        if current_operation.slice_width is not None:
            updates["line_selection"][f"{current_operation.slice_dim}_width"] = (
                current_operation.slice_width
            )
        updates["line_iter_dim"] = current_operation.slice_dim
    if not updates.get("line_x"):
        updates["line_x"] = _default_profile_x_dim(tool, source_name, current_operation)
    return updates


def _default_profile_x_dim(
    tool: FigureComposerTool, source_name: str, operation: FigureOperationState
) -> str | None:
    data = tool._source_data.get(source_name)
    if data is None:
        return None
    data = _public_source_data(data)
    candidates = [str(dim) for dim in data.dims if str(dim) != operation.slice_dim]
    if candidates:
        return candidates[-1]
    return None


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    prefix = "Needs axes: " if _has_invalid_target(tool, operation) else ""
    source = (
        tool._source_display_name(operation.line_source)
        if operation.line_source is not None
        else "missing source"
    )
    return f"{prefix}Line/profile: {source}"


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    if operation.line_placement == "one_per_axis":
        action = "Draws one extracted profile per target axis."
    else:
        action = "Overlays one or more 1D line/profile traces."
    return f"{action}\nTargets: {tool._axes_target_text(operation.axes)}"


def _has_invalid_target(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    return tool._axes_selection_has_invalid_target(operation.axes)


def _source_names(operation: FigureOperationState) -> tuple[str, ...]:
    return (operation.line_source,) if operation.line_source is not None else ()


def _build_source_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> None:
    source_mixed = tool._batch_is_mixed(operation, lambda target: target.line_source)
    source_combo = tool._source_combo(
        tool._source_names(),
        None if source_mixed else operation.line_source,
        lambda source: tool._update_current_operation(line_source=source),
        parent=tool.step_source_controls,
        mixed=source_mixed,
    )
    source_combo.setObjectName("figureComposerLineSourceCombo")
    tool._add_form_row(
        tool.step_source_controls_layout,
        "Line data",
        source_combo,
        "Data array used for this line/profile overlay.",
    )


def _editor_sections(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    return tuple(
        StepSection(
            key,
            title,
            page,
            "Configure the line/profile overlay for this step.",
        )
        for key, title, page in _build_line_editor(tool, operation)
    )


def _section_summary(
    tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    match key:
        case "sources":
            if operation.line_source is None:
                return "none"
            return tool._source_display_name(operation.line_source)
        case "axes":
            return tool._axes_target_text(operation.axes)
        case "line":
            if operation.line_placement == "one_per_axis":
                return "one per axis"
            return operation.line_x or "profile"
    return ""


SPEC = OperationSpec(
    kind=FigureOperationKind.LINE,
    add_actions=(
        AddStepActionSpec(
            action_id=FigureOperationKind.LINE.value,
            text="Line/Profile",
            tooltip="Overlay a 1D line or extracted profile on the selected axes.",
            create_operation=_create_line_operation,
        ),
    ),
    display_text=_display_text,
    tooltip=_tooltip,
    target_text=lambda tool, operation: tool._axes_target_text(operation.axes),
    has_invalid_target=_has_invalid_target,
    uses_axes=lambda _operation: True,
    uses_source_section=lambda _operation: True,
    source_names=_source_names,
    build_source_editor=_build_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=lambda tool, operation, _figure, axs: _render_line(tool, operation, axs),
    code_lines=_line_code,
)
