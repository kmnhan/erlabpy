"""Line/Profile operation editor, renderer, and code generation."""

from __future__ import annotations

import math
import typing

import numpy as np

from erlab.interactive._figurecomposer import _rendering
from erlab.interactive._figurecomposer._code import _axes_sequence_code, _selection_code
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
)
from erlab.interactive._figurecomposer._sources import (
    _available_source_dims,
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
    _float_tuple_from_text,
    _format_dict,
    _format_string_tuple,
    _format_tuple,
    _string_tuple_from_text,
)

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib.axes
    import xarray as xr
    from qtpy import QtWidgets

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_PROFILE_NORMALIZE_CODE = {
    "max": "profiles = [profile / profile.max(skipna=True) for profile in profiles]",
    "mean": "profiles = [profile / profile.mean(skipna=True) for profile in profiles]",
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


def _line_normalize_text(mode: str) -> str:
    match mode:
        case "max":
            return "Each profile by maximum"
        case "mean":
            return "Each profile by mean"
    return "None"


def _line_normalize_from_text(
    text: str,
) -> typing.Literal["none", "max", "mean"]:
    match text:
        case "Each profile by maximum":
            return "max"
        case "Each profile by mean":
            return "mean"
    return "none"


def _line_choice_data(
    tool: FigureComposerTool, operation: FigureOperationState, *, values: bool
) -> xr.DataArray | None:
    if operation.line_source is None:
        return None
    data = tool._source_data.get(operation.line_source)
    if data is None:
        return None
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

    for label, attr, getter, formatter, tooltip, object_name in (
        (
            "Legend labels",
            "line_labels",
            lambda target: target.line_labels,
            lambda value: _format_string_tuple(typing.cast("tuple[str, ...]", value)),
            "Optional legend labels.\n"
            "Use one value for every profile, or one value per profile.",
            "figureComposerLineLabelsEdit",
        ),
        (
            "Color",
            "line_color",
            lambda target: target.line_color,
            lambda value: "" if value is None else str(value),
            "Shared Matplotlib color for every profile.",
            "figureComposerLineColorEdit",
        ),
        (
            "Profile colors",
            "line_colors",
            lambda target: target.line_colors,
            lambda value: _format_string_tuple(typing.cast("tuple[str, ...]", value)),
            "Optional per-profile Matplotlib colors.\n"
            "Use comma-separated values or a Python list literal.",
            "figureComposerLineColorsEdit",
        ),
    ):
        text, mixed = tool._batch_text(operation, getter, formatter)
        edit = tool._line_edit(text)
        tool._apply_mixed_line_edit(edit, mixed)
        edit.setObjectName(object_name)
        if attr == "line_labels":
            edit.editingFinished.connect(
                lambda edit=edit: (
                    None
                    if tool._line_edit_batch_unchanged(edit)
                    else _update_current_line_labels(tool, edit.text())
                )
            )
        elif attr == "line_colors":
            edit.editingFinished.connect(
                lambda edit=edit: (
                    None
                    if tool._line_edit_batch_unchanged(edit)
                    else tool._update_current_operation(
                        line_colors=_string_tuple_from_text(edit.text())
                    )
                )
            )
        else:
            edit.editingFinished.connect(
                lambda edit=edit: (
                    None
                    if tool._line_edit_batch_unchanged(edit)
                    else tool._update_current_operation(
                        line_color=edit.text().strip() or None
                    )
                )
            )
        tool._add_form_row(tool.operation_editor_layout, label, edit, tooltip)

    selection_text, selection_mixed = tool._batch_text(
        operation, lambda target: target.line_selection, _format_dict
    )
    selection_edit = tool._line_edit(selection_text)
    tool._apply_mixed_line_edit(selection_edit, selection_mixed)
    selection_edit.setObjectName("figureComposerLineSelectionEdit")
    selection_edit.editingFinished.connect(
        lambda edit=selection_edit: (
            None
            if tool._line_edit_batch_unchanged(edit)
            else tool._update_current_operation(
                line_selection=_dict_from_text(edit.text())
            )
        )
    )
    tool._add_form_row(
        tool.operation_editor_layout,
        "Data selection",
        selection_edit,
        "Dict literal or keyword arguments used to select data.",
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
        lambda text: tool._update_current_operation_rebuild(line_iter_dim=text or None),
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

    normalize_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_normalize
    )
    normalize_combo = tool._combo(
        [
            _line_normalize_text("none"),
            _line_normalize_text("max"),
            _line_normalize_text("mean"),
        ],
        None if normalize_mixed else _line_normalize_text(operation.line_normalize),
        lambda text: tool._update_current_operation(
            line_normalize=_line_normalize_from_text(text)
        ),
        parent=page,
        mixed=normalize_mixed,
    )
    normalize_combo.setObjectName("figureComposerLineNormalizeCombo")
    tool._add_form_row(
        tool.operation_editor_layout,
        "Normalize",
        normalize_combo,
        "Normalize each extracted 1D profile independently before "
        "scale/offset. This does not normalize the source image or all "
        "profiles together.",
    )

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

    scales_text, scales_mixed = tool._batch_text(
        operation, lambda target: target.line_scales, _format_tuple
    )
    scales_edit = tool._line_edit(scales_text)
    tool._apply_mixed_line_edit(scales_edit, scales_mixed)
    scales_edit.editingFinished.connect(
        lambda edit=scales_edit: (
            None
            if tool._line_edit_batch_unchanged(edit)
            else tool._update_current_operation(
                line_scales=_float_tuple_from_text(edit.text())
            )
        )
    )
    tool._add_form_row(
        tool.operation_editor_layout,
        "Scales",
        scales_edit,
        "Scale applied to profile data values.\n"
        "Use one value or comma-separated per-profile values.",
    )

    offset_source_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_offset_source
    )
    offset_source_combo = tool._combo(
        ["manual", "index", "coordinate", "associated"],
        None if offset_source_mixed else operation.line_offset_source,
        lambda source: _update_current_line_offset_source(tool, source),
        parent=page,
        mixed=offset_source_mixed,
    )
    offset_source_combo.setObjectName("figureComposerLineOffsetSourceCombo")
    tool._add_form_row(
        tool.operation_editor_layout,
        "Offset source",
        offset_source_combo,
        "Where offsets come from.\n"
        "manual: use Offsets.\n"
        "index/coordinate/associated: derive from One profile per.",
    )

    if operation.line_offset_source == "associated":
        offset_coord_values = ["", *_available_line_offset_coords(tool, operation)]
        offset_coord_options_match = tool._batch_options_match(
            operation,
            lambda target: ["", *_available_line_offset_coords(tool, target)],
        )
        offset_coord_mixed = tool._batch_is_mixed(
            operation, lambda target: target.line_offset_coord
        )
        if (
            operation.line_offset_coord is not None
            and operation.line_offset_coord not in offset_coord_values
        ):
            offset_coord_values.append(operation.line_offset_coord)
        offset_coord_combo = tool._combo(
            offset_coord_values,
            None if offset_coord_mixed else operation.line_offset_coord or "",
            lambda text: tool._update_current_operation(line_offset_coord=text or None),
            parent=page,
            mixed=offset_coord_mixed,
            enabled=offset_coord_options_match,
        )
        offset_coord_combo.setObjectName("figureComposerLineOffsetCoordinateCombo")
        tool._add_form_row(
            tool.operation_editor_layout,
            "Offset coordinate",
            offset_coord_combo,
            "Associated coordinate used when Offset source is associated."
            + (
                "\nDisabled while selected steps have different valid choices."
                if not offset_coord_options_match
                else ""
            ),
        )

    if operation.line_offset_source != "manual":
        offset_scale_text, offset_scale_mixed = tool._batch_text(
            operation,
            lambda target: target.line_offset_scale,
            lambda value: f"{float(value):g}",
        )
        offset_scale_edit = tool._line_edit(offset_scale_text)
        tool._apply_mixed_line_edit(offset_scale_edit, offset_scale_mixed)
        offset_scale_edit.setObjectName("figureComposerLineOffsetScaleEdit")
        offset_scale_edit.editingFinished.connect(
            lambda edit=offset_scale_edit: (
                None
                if tool._line_edit_batch_unchanged(edit)
                else tool._update_current_operation(
                    line_offset_scale=float(edit.text()) if edit.text().strip() else 1.0
                )
            )
        )
        tool._add_form_row(
            tool.operation_editor_layout,
            "Offset scale",
            offset_scale_edit,
            "Multiplier applied to offsets from the selected source.",
        )

    if operation.line_offset_source == "manual":
        offsets_text, offsets_mixed = tool._batch_text(
            operation, lambda target: target.line_offsets, _format_tuple
        )
        offsets_edit = tool._line_edit(offsets_text)
        tool._apply_mixed_line_edit(offsets_edit, offsets_mixed)
        offsets_edit.setObjectName("figureComposerLineOffsetsEdit")
        offsets_edit.editingFinished.connect(
            lambda edit=offsets_edit: (
                None
                if tool._line_edit_batch_unchanged(edit)
                else tool._update_current_operation(
                    line_offsets=_float_tuple_from_text(edit.text())
                )
            )
        )
        tool._add_form_row(
            tool.operation_editor_layout,
            "Offsets",
            offsets_edit,
            "Offset applied to profile data values.\n"
            "Use one value or comma-separated per-profile values.",
        )
    return [("line", "Line", page)]


def _update_current_line_offset_source(tool: FigureComposerTool, source: str) -> None:
    updates: dict[str, typing.Any] = {"line_offset_source": source}
    if source == "manual":
        updates["line_offset_scale"] = 1.0
    tool._update_current_operation_rebuild(**updates)


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
        tuple[tuple[tuple[int, int], ...], str], tuple[int, FigureOperationState]
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
    _rendering._render_preview(tool)
    tool.sigInfoChanged.emit()


def _line_axes_key(
    operation: FigureOperationState,
) -> tuple[tuple[tuple[int, int], ...], str]:
    return operation.axes.axes, operation.axes.expression


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
    tool: FigureComposerTool, operation: FigureOperationState, axs: np.ndarray
) -> None:
    line_items = _line_data_items(tool, operation)
    if not line_items:
        return
    axes = _rendering._iter_axes(
        _rendering._axes_from_selection(
            tool, operation.axes, axs, for_plot_slices=False
        )
    )
    if operation.line_placement == "one_per_axis":
        _render_one_profile_per_axis(tool, operation, axes, line_items)
        return
    scales = _line_transform_values(operation.line_scales, len(line_items), default=1.0)
    offsets = _line_offsets_for_profiles(tool, operation, line_items)
    styles = _line_styles_for_profiles(operation, len(line_items))
    for axis in axes:
        for line_data, scale, offset, kwargs in zip(
            line_items, scales, offsets, styles, strict=True
        ):
            line_values = offset + scale * line_data
            if operation.line_values_axis == "x":
                coordinate = _line_coordinate(line_data, operation.line_x)
                axis.plot(line_values.values, coordinate.values, **kwargs)
            else:
                line_values.plot(ax=axis, x=operation.line_x, **kwargs)


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
    scales = _line_transform_values(operation.line_scales, len(profiles), default=1.0)
    offsets = _line_offsets_for_profiles(tool, operation, profiles)
    styles = _line_styles_for_profiles(operation, len(profiles))
    for axis, profile, scale, offset, kwargs in zip(
        axes, profiles, scales, offsets, styles, strict=True
    ):
        coordinate = _line_coordinate(profile, operation.line_x)
        values = offset + scale * profile
        if operation.line_values_axis == "x":
            axis.plot(values.values, coordinate.values, **kwargs)
        else:
            axis.plot(coordinate.values, values.values, **kwargs)


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
        if operation.line_selection:
            data = data.qsel(operation.line_selection)
        line_data = data.squeeze(drop=True)
        if operation.line_y:
            line_data = line_data[operation.line_y]
        if operation.line_iter_dim and operation.line_iter_dim in line_data.dims:
            line_items = [
                item.squeeze(drop=True)
                for item in line_data.transpose(operation.line_iter_dim, ...)
            ]
        else:
            line_items = [line_data]
    else:
        return []

    normalized: list[xr.DataArray] = []
    for line_data in line_items:
        if line_data.ndim != 1:
            continue
        line_data = _normalize_line_data(line_data, operation.line_normalize)
        normalized.append(line_data)
    return normalized


def _normalize_line_data(
    line_data: xr.DataArray, mode: typing.Literal["none", "max", "mean"]
) -> xr.DataArray:
    if mode == "none":
        return line_data
    if mode == "max":
        scale = float(line_data.max(skipna=True))
    else:
        scale = float(line_data.mean(skipna=True))
    if math.isfinite(scale) and scale != 0.0:
        return line_data / scale
    return line_data


def _line_coordinate(line_data: xr.DataArray, dim: str | None) -> xr.DataArray:
    if dim:
        return line_data[dim]
    if line_data.ndim != 1:
        raise ValueError("Profile overlays require one-dimensional profiles")
    return line_data[line_data.dims[0]]


def _line_offsets_for_profiles(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    profiles: Sequence[xr.DataArray],
) -> tuple[float, ...]:
    count = len(profiles)
    if operation.line_offset_source == "manual":
        return _line_transform_values(operation.line_offsets, count, default=0.0)
    if operation.line_offset_source == "index":
        offsets = tuple(float(index) for index in range(count))
    else:
        coord_name = _line_offset_coordinate_name(operation)
        offsets = tuple(
            _profile_scalar_coord_value(profile, coord_name) for profile in profiles
        )
    if operation.line_offset_scale == 1.0:
        return offsets
    return tuple(operation.line_offset_scale * offset for offset in offsets)


def _line_offset_coordinate_name(operation: FigureOperationState) -> str:
    if operation.line_offset_source == "coordinate":
        if operation.line_iter_dim is None:
            raise ValueError("Coordinate offsets require One profile per")
        return operation.line_iter_dim
    if operation.line_offset_coord is None:
        raise ValueError("Associated-coordinate offsets require a coordinate")
    return operation.line_offset_coord


def _profile_scalar_coord_value(profile: xr.DataArray, coord_name: str) -> float:
    coord = profile.coords.get(coord_name)
    if coord is None:
        raise ValueError(f"Profile has no coordinate named {coord_name!r}")
    values = np.asarray(coord.values).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Profile coordinate {coord_name!r} is not scalar")
    return float(values[0])


def _line_styles_for_profiles(
    operation: FigureOperationState, count: int
) -> tuple[dict[str, typing.Any], ...]:
    labels = _line_text_values(operation.line_labels, count, default=None)
    colors = _line_text_values(
        operation.line_colors, count, default=operation.line_color
    )
    styles: list[dict[str, typing.Any]] = []
    for label, color in zip(labels, colors, strict=True):
        kwargs: dict[str, typing.Any] = {}
        if label:
            kwargs["label"] = label
        if color:
            kwargs["color"] = color
        styles.append(kwargs)
    return tuple(styles)


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


def _line_transform_values(
    values: Sequence[float], count: int, *, default: float
) -> tuple[float, ...]:
    if count < 1:
        return ()
    if not values:
        return (default,) * count
    if len(values) == 1:
        return (float(values[0]),) * count
    if len(values) != count:
        raise ValueError(
            "Profile scales and offsets must be one value or one per profile"
        )
    return tuple(float(value) for value in values)


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
    if operation.line_normalize != "none":
        lines.append(_PROFILE_NORMALIZE_CODE[operation.line_normalize])
    loop_names = ["profile"]
    loop_values = ["profiles"]
    lines.extend(
        _line_value_code(
            operation,
            loop_names=loop_names,
            loop_values=loop_values,
        )
    )
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
    value_expr = _line_profile_value_expression(operation)
    if operation.line_values_axis == "x":
        call_args = f"{value_expr}, {coordinate}"
        if kwargs_text:
            call_args += f", {kwargs_text}"
        lines.append(f"        ax.plot({call_args})")
    else:
        call_args = "ax=ax"
        if operation.line_x:
            call_args += f", x={operation.line_x!r}"
        if kwargs_text:
            call_args += f", {kwargs_text}"
        plot_target = "profile" if value_expr == "profile" else f"({value_expr})"
        lines.append(f"        {plot_target}.plot({call_args})")
    return lines


def _line_value_code(
    operation: FigureOperationState,
    *,
    loop_names: list[str],
    loop_values: list[str],
) -> list[str]:
    lines: list[str] = []
    if operation.line_scales:
        if len(operation.line_scales) == 1:
            lines.append(f"profile_scale = {operation.line_scales[0]!r}")
        else:
            lines.append(f"profile_scales = {list(operation.line_scales)!r}")
            loop_names.append("scale")
            loop_values.append("profile_scales")

    if not _line_uses_offsets(operation):
        return lines

    if operation.line_offset_source == "manual":
        if len(operation.line_offsets) == 1:
            lines.append(f"profile_offset = {operation.line_offsets[0]!r}")
            return lines
        lines.append(f"profile_offsets = {list(operation.line_offsets)!r}")
    elif operation.line_offset_source == "index":
        lines.append(
            "profile_offsets = [float(index) for index in range(len(profiles))]"
        )
    else:
        coord_name = _line_offset_coordinate_name(operation)
        lines.append(
            "profile_offsets = "
            f"[float(profile[{coord_name!r}]) for profile in profiles]"
        )

    if operation.line_offset_scale != 1.0:
        lines.append(f"profile_offset_scale = {operation.line_offset_scale!r}")
        lines.append(
            "profile_offsets = "
            "[profile_offset_scale * offset for offset in profile_offsets]"
        )
    loop_names.append("offset")
    loop_values.append("profile_offsets")
    return lines


def _line_uses_offsets(operation: FigureOperationState) -> bool:
    return operation.line_offset_source != "manual" or bool(operation.line_offsets)


def _line_profile_value_expression(operation: FigureOperationState) -> str:
    if not operation.line_scales:
        value_expr = "profile"
    elif len(operation.line_scales) == 1:
        value_expr = "profile_scale * profile"
    else:
        value_expr = "scale * profile"
    if not _line_uses_offsets(operation):
        return value_expr
    if operation.line_offset_source == "manual" and len(operation.line_offsets) == 1:
        return f"profile_offset + {value_expr}"
    return f"offset + {value_expr}"


def _line_style_code(
    operation: FigureOperationState,
    *,
    loop_names: list[str],
    loop_values: list[str],
) -> tuple[list[str], str]:
    lines: list[str] = []
    kwargs: list[str] = []
    if operation.line_labels:
        if len(operation.line_labels) == 1:
            lines.append(f"profile_label = {operation.line_labels[0]!r}")
            kwargs.append("label=profile_label")
        else:
            lines.append(f"profile_labels = {list(operation.line_labels)!r}")
            loop_names.append("label")
            loop_values.append("profile_labels")
            kwargs.append("label=label")

    if operation.line_colors:
        if len(operation.line_colors) == 1:
            lines.append(f"profile_color = {operation.line_colors[0]!r}")
            kwargs.append("color=profile_color")
        else:
            lines.append(f"profile_colors = {list(operation.line_colors)!r}")
            loop_names.append("color")
            loop_values.append("profile_colors")
            kwargs.append("color=color")
    elif operation.line_color:
        kwargs.append(f"color={operation.line_color!r}")
    return lines, ", ".join(kwargs)


def _one_profile_per_axis_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    lines: list[str] = []
    axes_code = _axes_sequence_code(tool, operation.axes)
    if operation.line_normalize != "none":
        lines.append(_PROFILE_NORMALIZE_CODE[operation.line_normalize])
    loop_names = ["ax", "profile"]
    loop_values = [axes_code, "profiles"]
    lines.extend(
        _line_value_code(
            operation,
            loop_names=loop_names,
            loop_values=loop_values,
        )
    )
    style_lines, kwargs_text = _line_style_code(
        operation,
        loop_names=loop_names,
        loop_values=loop_values,
    )
    lines.extend(style_lines)
    value_expr = _line_profile_value_expression(operation)
    coordinate = _line_coordinate_code(operation)
    lines.append(
        "for "
        + ", ".join(loop_names)
        + " in zip("
        + ", ".join(loop_values)
        + ", strict=True):"
    )
    if operation.line_values_axis == "x":
        call_args = f"{value_expr}, {coordinate}"
    else:
        call_args = f"{coordinate}, {value_expr}"
    if kwargs_text:
        call_args += f", {kwargs_text}"
    lines.append(f"    ax.plot({call_args})")
    return lines


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
            "line_color": "black",
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
    candidates = [str(dim) for dim in data.dims if str(dim) != operation.slice_dim]
    if candidates:
        return candidates[-1]
    return None


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    prefix = "Needs axes: " if _has_invalid_target(tool, operation) else ""
    source = operation.line_source or "missing source"
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
    return (
        bool(operation.axes.invalid_axes(tool._recipe.setup))
        and not operation.axes.expression
    )


def _source_names(operation: FigureOperationState) -> tuple[str, ...]:
    return (operation.line_source,) if operation.line_source is not None else ()


def _build_source_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> None:
    source_combo = tool._combo(
        tool._source_names(),
        operation.line_source,
        lambda text: tool._update_current_operation(line_source=text or None),
        parent=tool.step_source_controls,
    )
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
            return operation.line_source or "none"
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
