"""Plot Slices operation editor, renderer, and code generation."""

from __future__ import annotations

import math
import typing

import numpy as np
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.plotting as eplt
from erlab.interactive._figurecomposer._code import (
    _axes_code,
    _maybe_squeeze_drop_code,
    _selection_code,
)
from erlab.interactive._figurecomposer._defaults import _current_options
from erlab.interactive._figurecomposer._label_help import legend_label_input_widget
from erlab.interactive._figurecomposer._labels import (
    label_context,
    label_coord_placeholder_name,
    label_editor_text,
    label_field_names,
    label_fstring_code,
    label_text_tooltip,
    label_text_uses_placeholders,
    labels_from_text,
    string_literal_expression,
    update_current_line_label_text,
)
from erlab.interactive._figurecomposer._line_colormap import (
    LINE_COLOR_CMAP_TRIM_MAX,
    colormap_code_lines,
    colors_from_values,
    effective_line_color_cmap,
    effective_line_color_cmap_trim,
    effective_line_color_coord,
    line_color_cmap_trim_control_values,
    line_colormap_active,
    numeric_context_field_names,
    values_from_contexts,
)
from erlab.interactive._figurecomposer._line_style import (
    CONTROLLED_LINE_KW_KEYS,
    LINE_MARKER_OPTIONS,
    LINE_STYLE_DEFAULT_LABEL,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    configure_style_combo,
    extra_line_kw,
    line_kw_float,
    line_kw_style_value,
    line_kw_text,
    optional_positive_spinbox,
    optional_positive_spinbox_value,
    set_style_combo_value,
    style_combo_value,
    update_current_extra_line_kw,
    update_current_line_kw,
)
from erlab.interactive._figurecomposer._line_transform import (
    add_line_transform_controls,
    line_transform_active,
    profile_transform_code_lines,
    transform_profiles,
)
from erlab.interactive._figurecomposer._norms import (
    _MATPLOTLIB_NORM_NAMES,
    _NORM_CHOICES,
    _ZERO_VCENTER_NORMS,
    _cmap_base_and_reverse,
    _cmap_with_reverse,
    _effective_norm_name,
    _norm_code,
    _norm_combo_choices,
    _norm_combo_text,
    _norm_kwarg_fields,
    _norm_name_from_combo_text,
    _norm_object,
    _norm_updates_from_kwargs,
    _use_powernorm_plot_kwargs,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
    _live_layout_axes,
)
from erlab.interactive._figurecomposer._sources import (
    _available_source_dims,
    _public_source_data,
    _selected_data,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._state import (
    _POWER_NORM_NAME,
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    _PlotSlicesShape,
)
from erlab.interactive._figurecomposer._text import (
    _code_kwargs,
    _dict_from_text,
    _float_tuple_from_text,
    _format_dict,
    _format_pair,
    _format_plot_limit,
    _format_tuple,
    _plot_limit_from_text,
    _RawCode,
    _selection_value_count,
)
from erlab.interactive._figurecomposer._widgets import _ColorLineEditWidget

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import matplotlib.axes

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


_PLOT_SLICES_EXPLICIT_KWARGS = frozenset(
    (
        "figsize",
        "transpose",
        "xlim",
        "ylim",
        "crop",
        "same_limits",
        "axis",
        "show_all_labels",
        "colorbar",
        "hide_colorbar_ticks",
        "annotate",
        "cmap",
        "norm",
        "gamma",
        "vmin",
        "vmax",
        "line_kw",
        "line_order",
        "order",
        "cmap_order",
        "norm_order",
        "gradient",
        "gradient_kw",
        "subplot_kw",
        "annotate_kw",
        "colorbar_kw",
        "axes",
    )
)
_MISSING = object()
_PLOT_SLICES_PANEL_LINE = "line"
_PLOT_SLICES_PANEL_IMAGE = "image"
_PLOT_SLICES_PANEL_MIXED = "mixed"
_PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR = "_figure_composer_operation_id"
_PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR = "_figure_composer_panel_key"
_SLICE_VALUES_MODE_LABELS = {
    "manual": "Manual values",
    "all": "All coordinate values",
}
_SLICE_VALUES_LABEL_MODES = {
    label: mode for mode, label in _SLICE_VALUES_MODE_LABELS.items()
}
_LINE_COLOR_MODE_TEXT = {
    "manual": "Manual",
    "coordinate": "By coordinate",
}


class _PlotSlicesPanelKey(typing.NamedTuple):
    map_index: int
    slice_index: int
    label: str


def _operation_dim_names(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    maps = _operation_maps(tool, operation)
    dims: list[str] = []
    for data in maps:
        for dim in data.dims:
            dim_text = str(dim)
            if dim_text not in dims:
                dims.append(dim_text)
    if dims:
        return tuple(dims)
    return tuple(_available_source_dims(tool._source_data, operation.sources))


def _slice_values_mode_text(mode: str) -> str:
    return _SLICE_VALUES_MODE_LABELS.get(mode, _SLICE_VALUES_MODE_LABELS["manual"])


def _slice_values_mode_from_text(text: str) -> typing.Literal["manual", "all"]:
    return typing.cast(
        'typing.Literal["manual", "all"]',
        _SLICE_VALUES_LABEL_MODES.get(text, "manual"),
    )


def _line_color_mode_text(mode: str) -> str:
    return _LINE_COLOR_MODE_TEXT.get(mode, _LINE_COLOR_MODE_TEXT["manual"])


def _line_color_mode_from_text(
    text: str,
) -> typing.Literal["manual", "coordinate"]:
    if text == _LINE_COLOR_MODE_TEXT["coordinate"]:
        return "coordinate"
    return "manual"


def _use_all_coordinate_slice_values(operation: FigureOperationState) -> bool:
    return operation.slice_values_mode == "all"


def _all_coordinate_slice_values(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[float, ...]:
    if not operation.slice_dim:
        return ()
    maps = _operation_maps(tool, operation)
    if not maps:
        return ()
    data = maps[0]
    if operation.slice_dim not in data.dims:
        return ()
    try:
        values = np.asarray(
            data.thin({operation.slice_dim: operation.slice_values_thin})
            .coords[operation.slice_dim]
            .values
        ).reshape(-1)
        return tuple(float(value) for value in values)
    except (KeyError, TypeError, ValueError):
        return ()


def _effective_slice_values(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[float, ...]:
    if _use_all_coordinate_slice_values(operation):
        return _all_coordinate_slice_values(tool, operation)
    return operation.slice_values


def _all_coordinate_slice_values_error(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    dims: Sequence[str],
) -> str:
    if not _use_all_coordinate_slice_values(operation):
        return ""
    if not operation.slice_dim:
        return "Choose a dimension before using all coordinate values."
    if operation.slice_dim not in dims:
        return f"{operation.slice_dim!r} is not an input dimension."
    if _effective_slice_values(tool, operation):
        return ""
    return f"{operation.slice_dim!r} coordinate values must be numeric and non-empty."


def _all_coordinate_slice_values_summary(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str:
    if not operation.slice_dim:
        return "Choose a dimension."
    maps = _operation_maps(tool, operation)
    if not maps:
        return "Select at least one valid source."
    data = maps[0]
    if operation.slice_dim not in data.dims:
        return f"{operation.slice_dim!r} is not an input dimension."
    slice_values = _effective_slice_values(tool, operation)
    if not slice_values:
        return (
            f"{operation.slice_dim!r} coordinate values must be numeric and non-empty."
        )
    total_count = int(data.sizes[operation.slice_dim])
    plotted_count = len(slice_values)
    if operation.slice_values_thin == 1 or plotted_count == total_count:
        return f"{operation.slice_dim}: {plotted_count} values"
    return f"{operation.slice_dim}: {total_count} values, {plotted_count} plotted"


def _first_plot_slices_source_code(operation: FigureOperationState) -> str | None:
    if operation.map_selections:
        return _selection_code(operation.map_selections[0])
    if operation.sources:
        return _valid_source_variable(operation.sources[0])
    return None


def _all_coordinate_slice_values_code(
    operation: FigureOperationState,
) -> str | None:
    if not _use_all_coordinate_slice_values(operation) or not operation.slice_dim:
        return None
    source_code = _first_plot_slices_source_code(operation)
    if source_code is None:
        return None
    dim_code = erlab.interactive.utils._parse_single_arg(operation.slice_dim)
    if operation.slice_values_thin == 1:
        return f"{source_code}.coords[{dim_code}].values"
    return (
        f"{source_code}.thin({{{dim_code}: {operation.slice_values_thin}}})"
        f".coords[{dim_code}].values"
    )


def _plot_slices_panel_kind(shape: _PlotSlicesShape) -> str:
    if shape.plot_ndim == 1:
        return _PLOT_SLICES_PANEL_LINE
    return _PLOT_SLICES_PANEL_IMAGE


def _plot_slices_batch_panel_kind(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str:
    operations = tuple(
        target
        for _index, target in tool._editable_operations()
        if target.kind == FigureOperationKind.PLOT_SLICES
    )
    if not operations:
        operations = (operation,)
    kinds = {
        _plot_slices_panel_kind(_plot_slices_shape(tool, target))
        for target in operations
    }
    if len(kinds) == 1:
        return kinds.pop()
    return _PLOT_SLICES_PANEL_MIXED


def _plot_slices_slice_count(
    tool: FigureComposerTool, operation: FigureOperationState
) -> int:
    operation = _normalized_selection_operation(tool, operation)
    dims = _operation_dim_names(tool, operation)
    selected_dims: set[str] = set()
    slice_count = 1
    slice_values = _effective_slice_values(tool, operation)
    if operation.slice_dim and slice_values:
        selected_dims.add(operation.slice_dim)
        slice_count = len(slice_values)
    for key, value in operation.slice_kwargs.items():
        if key.endswith("_width") or key not in dims:
            continue
        count = _selection_value_count(value)
        if count is None:
            selected_dims.add(key)
        else:
            selected_dims.add(key)
            slice_count = max(slice_count, count)
    return max(slice_count, 1)


def _plot_slices_panel_keys(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[_PlotSlicesPanelKey, ...]:
    operation = _normalized_selection_operation(tool, operation)
    maps = _operation_maps(tool, operation)
    map_count = len(maps) or max(len(operation.sources), 1)
    slice_count = _plot_slices_slice_count(tool, operation)
    source_names = (
        tuple(selection.source for selection in operation.map_selections)
        if operation.map_selections
        else operation.sources
    )
    map_labels = tuple(
        tool._source_display_name(source_names[index])
        if index < len(source_names)
        else f"map {index + 1}"
        for index in range(map_count)
    )
    slice_labels = _plot_slices_slice_labels(
        operation,
        slice_count,
        _effective_slice_values(tool, operation),
    )

    keys: list[_PlotSlicesPanelKey] = []
    if operation.order == "F":
        indices = (
            (map_index, slice_index)
            for slice_index in range(slice_count)
            for map_index in range(map_count)
        )
    else:
        indices = (
            (map_index, slice_index)
            for map_index in range(map_count)
            for slice_index in range(slice_count)
        )
    for map_index, slice_index in indices:
        label = f"{map_labels[map_index]}"
        if slice_count > 1 or operation.slice_dim or operation.slice_kwargs:
            label += f", {slice_labels[slice_index]}"
        keys.append(_PlotSlicesPanelKey(map_index, slice_index, label))
    return tuple(keys)


def _plot_slices_slice_labels(
    operation: FigureOperationState,
    slice_count: int,
    slice_values: Sequence[float] | None = None,
) -> tuple[str, ...]:
    if slice_values is None:
        slice_values = operation.slice_values
    if operation.slice_dim and slice_values:
        return tuple(
            f"{operation.slice_dim}={value:g}"
            for value in tuple(slice_values)[:slice_count]
        )
    for key, value in operation.slice_kwargs.items():
        if key.endswith("_width"):
            continue
        count = _selection_value_count(value)
        if count is None:
            continue
        return tuple(f"{key}[{index}]" for index in range(slice_count))
    return tuple(f"slice {index + 1}" for index in range(slice_count))


def _panel_style_key(style: FigurePlotSlicesPanelStyleState) -> tuple[int, int]:
    return (style.map_index, style.slice_index)


def _panel_style_map(
    operation: FigureOperationState,
) -> dict[tuple[int, int], FigurePlotSlicesPanelStyleState]:
    return {_panel_style_key(style): style for style in operation.panel_styles}


def _panel_style_map_for_keys(
    operation: FigureOperationState,
    keys: tuple[_PlotSlicesPanelKey, ...],
) -> dict[tuple[int, int], FigurePlotSlicesPanelStyleState]:
    valid_keys = {(key.map_index, key.slice_index) for key in keys}
    return {
        key: style
        for key, style in _panel_style_map(operation).items()
        if key in valid_keys
    }


def _panel_style_from_map(
    styles: Mapping[tuple[int, int], FigurePlotSlicesPanelStyleState],
    key: _PlotSlicesPanelKey,
) -> FigurePlotSlicesPanelStyleState:
    return styles.get(
        (key.map_index, key.slice_index),
        FigurePlotSlicesPanelStyleState(
            map_index=key.map_index,
            slice_index=key.slice_index,
        ),
    )


def _panel_style_has_cmap_override(style: FigurePlotSlicesPanelStyleState) -> bool:
    return style.cmap is not None


def _panel_style_has_norm_override(style: FigurePlotSlicesPanelStyleState) -> bool:
    return (
        style.norm_name is not None
        or style.norm_gamma is not None
        or style.norm_clip is not None
        or bool(style.norm_kwargs)
        or style.vmin is not None
        or style.vmax is not None
        or style.vcenter is not None
        or style.halfrange is not None
    )


def _panel_style_has_line_override(style: FigurePlotSlicesPanelStyleState) -> bool:
    return bool(style.line_kw)


def _panel_style_has_overrides(style: FigurePlotSlicesPanelStyleState) -> bool:
    return (
        _panel_style_has_cmap_override(style)
        or _panel_style_has_norm_override(style)
        or _panel_style_has_line_override(style)
    )


def _effective_panel_cmap(
    operation: FigureOperationState, style: FigurePlotSlicesPanelStyleState
) -> str:
    if style.cmap is not None:
        return style.cmap
    return operation.cmap or _current_options().colors.cmap.name


def _operation_with_panel_norm_style(
    operation: FigureOperationState,
    style: FigurePlotSlicesPanelStyleState,
) -> FigureOperationState:
    if not _panel_style_has_norm_override(style):
        return operation
    updates: dict[str, typing.Any] = {}
    for attr in (
        "norm_name",
        "norm_gamma",
        "norm_clip",
        "norm_kwargs",
        "vmin",
        "vmax",
        "vcenter",
        "halfrange",
    ):
        value = getattr(style, attr)
        if value is not None and value != {}:
            updates[attr] = value
    if "norm_name" not in updates:
        updates["norm_name"] = operation.norm_name or _POWER_NORM_NAME
    return operation.model_copy(update=updates)


def _style_sequence_shape(
    keys: tuple[_PlotSlicesPanelKey, ...],
) -> tuple[int, int]:
    map_count = max((key.map_index for key in keys), default=0) + 1
    slice_count = max((key.slice_index for key in keys), default=0) + 1
    return map_count, slice_count


def _nested_panel_values(
    keys: tuple[_PlotSlicesPanelKey, ...],
    order: str,
    value_getter: Callable[[_PlotSlicesPanelKey], typing.Any],
) -> list[list[typing.Any]]:
    map_count, slice_count = _style_sequence_shape(keys)
    if order == "F":
        return [
            [
                value_getter(_PlotSlicesPanelKey(map_index, slice_index, ""))
                for map_index in range(map_count)
            ]
            for slice_index in range(slice_count)
        ]
    return [
        [
            value_getter(_PlotSlicesPanelKey(map_index, slice_index, ""))
            for slice_index in range(slice_count)
        ]
        for map_index in range(map_count)
    ]


def _panel_cmap_argument(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | list[list[str]] | None:
    if not operation.panel_styles_enabled or not operation.panel_styles:
        return operation.cmap
    keys = _plot_slices_panel_keys(tool, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    if not any(_panel_style_has_cmap_override(style) for style in styles.values()):
        return operation.cmap

    def value_getter(key: _PlotSlicesPanelKey) -> str:
        style = styles.get(
            (key.map_index, key.slice_index),
            FigurePlotSlicesPanelStyleState(
                map_index=key.map_index,
                slice_index=key.slice_index,
            ),
        )
        return _effective_panel_cmap(operation, style)

    values = [value_getter(key) for key in keys]
    first = values[0] if values else operation.cmap
    if first is not None and all(value == first for value in values[1:]):
        return first
    return _nested_panel_values(keys, operation.cmap_order, value_getter)


def _panel_norm_argument(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[list[object]] | object | None:
    if not operation.panel_styles_enabled or not operation.panel_styles:
        return None
    keys = _plot_slices_panel_keys(tool, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    if not any(_panel_style_has_norm_override(style) for style in styles.values()):
        return None
    order = operation.norm_order or operation.cmap_order

    def value_getter(key: _PlotSlicesPanelKey) -> object:
        style = styles.get(
            (key.map_index, key.slice_index),
            FigurePlotSlicesPanelStyleState(
                map_index=key.map_index,
                slice_index=key.slice_index,
            ),
        )
        return _norm_object(_operation_with_panel_norm_style(operation, style))

    if len(keys) == 1:
        return value_getter(keys[0])
    return _nested_panel_values(keys, order, value_getter)


def _panel_line_kw_argument(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any] | list[list[dict[str, typing.Any]]]:
    labels = _plot_slices_line_labels(tool, operation)
    has_labels = any(labels)
    color_active = _plot_slices_line_colormap_active(tool, operation)
    line_colors = (
        _plot_slices_line_colormap_colors(tool, operation) if color_active else ()
    )
    if (
        not has_labels
        and (not operation.panel_styles_enabled or not operation.panel_styles)
        and not color_active
    ):
        return dict(operation.line_kw)
    keys = _plot_slices_panel_keys(tool, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    has_line_overrides = any(
        _panel_style_has_line_override(style) for style in styles.values()
    )
    if not has_labels and not has_line_overrides and not color_active:
        return dict(operation.line_kw)
    labels_by_key = (
        {
            (key.map_index, key.slice_index): label
            for key, label in zip(keys, labels, strict=True)
        }
        if has_labels
        else {}
    )
    colors_by_key = (
        {
            (key.map_index, key.slice_index): color
            for key, color in zip(keys, line_colors, strict=True)
        }
        if color_active
        else {}
    )

    def value_getter(key: _PlotSlicesPanelKey) -> dict[str, typing.Any]:
        style = styles.get(
            (key.map_index, key.slice_index),
            FigurePlotSlicesPanelStyleState(
                map_index=key.map_index,
                slice_index=key.slice_index,
            ),
        )
        line_kw = {**operation.line_kw, **style.line_kw}
        if has_labels:
            label = labels_by_key.get((key.map_index, key.slice_index))
            if label:
                line_kw["label"] = label
        if color_active:
            line_kw.pop("c", None)
            line_kw["color"] = colors_by_key[(key.map_index, key.slice_index)]
        return line_kw

    values = [value_getter(key) for key in keys]
    first = values[0] if values else dict(operation.line_kw)
    if all(value == first for value in values[1:]):
        return first
    return _nested_panel_values(keys, operation.order, value_getter)


def _has_panel_line_kw_overrides(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    if not operation.panel_styles_enabled:
        return False
    keys = _plot_slices_panel_keys(tool, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    return any(_panel_style_has_line_override(style) for style in styles.values())


def _plot_slices_line_label_contexts(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[dict[str, typing.Any], ...]:
    keys = _plot_slices_panel_keys(tool, operation)
    source_labels = _plot_slices_source_labels(tool, operation)
    slice_values = _effective_slice_values(tool, operation)
    contexts: list[dict[str, typing.Any]] = []
    for index, key in enumerate(keys):
        value = None
        if operation.slice_dim and key.slice_index < len(slice_values):
            value = slice_values[key.slice_index]
        source = (
            source_labels[key.map_index] if key.map_index < len(source_labels) else None
        )
        context = label_context(
            index=index,
            source=source,
            dim=operation.slice_dim,
            value=value,
        )
        contexts.append(context)
    return tuple(contexts)


def _plot_slices_line_labels(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str | None, ...]:
    contexts = _plot_slices_line_label_contexts(tool, operation)
    return labels_from_text(
        operation.line_label_text,
        contexts,
        literal_values=operation.line_labels,
        default=None,
        item_name="slice",
    )


def _plot_slices_line_colormap_active(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    return (
        line_colormap_active(operation)
        and _plot_slices_shape(tool, operation).plot_ndim == 1
    )


def _available_plot_slices_line_color_coords(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    names = list(
        numeric_context_field_names(_plot_slices_line_label_contexts(tool, operation))
    )
    for name in (operation.slice_dim, operation.line_color_coord):
        if name and name not in names:
            names.append(name)
    return names


def _plot_slices_line_color_values(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[float, ...]:
    return values_from_contexts(
        _plot_slices_line_label_contexts(tool, operation),
        effective_line_color_coord(operation, operation.slice_dim),
        item_name="slice",
    )


def _plot_slices_line_colormap_colors(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[tuple[float, float, float, float], ...]:
    return colors_from_values(
        _plot_slices_line_color_values(tool, operation),
        effective_line_color_cmap(operation),
        trim=effective_line_color_cmap_trim(operation),
    )


def _plot_slices_source_labels(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    source_names = (
        tuple(selection.source for selection in operation.map_selections)
        if operation.map_selections
        else operation.sources
    )
    map_count = len(_operation_maps(tool, operation)) or max(len(source_names), 1)
    return tuple(
        tool._source_display_name(source_names[index])
        if index < len(source_names)
        else f"map {index + 1}"
        for index in range(map_count)
    )


def _plot_slices_label_line_kw_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    contexts = _plot_slices_line_label_contexts(tool, operation)
    if not label_text_uses_placeholders(operation.line_label_text, contexts):
        return None
    keys = _plot_slices_panel_keys(tool, operation)
    if not keys:
        return None
    fields = label_field_names(operation.line_label_text)
    source_labels = _plot_slices_source_labels(tool, operation)
    slice_values_code = _plot_slices_slice_values_code(tool, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    has_style_overrides = operation.panel_styles_enabled and any(
        _panel_style_has_line_override(style) for style in styles.values()
    )
    if has_style_overrides:
        return _plot_slices_styled_label_line_kw_code(
            operation,
            keys,
            source_labels,
            slice_values_code,
            styles,
            fields,
        )
    return _plot_slices_label_line_kw_comprehension_code(
        operation,
        keys,
        source_labels,
        slice_values_code,
        fields,
    )


def _plot_slices_line_kw_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    if not _plot_slices_line_colormap_active(tool, operation):
        return _plot_slices_label_line_kw_code(tool, operation)
    keys = _plot_slices_panel_keys(tool, operation)
    if not keys:
        return None
    contexts = _plot_slices_line_label_contexts(tool, operation)
    use_placeholder_labels = label_text_uses_placeholders(
        operation.line_label_text, contexts
    )
    labels = _plot_slices_line_labels(tool, operation)
    fields = label_field_names(operation.line_label_text)
    source_labels = _plot_slices_source_labels(tool, operation)
    slice_values_code = _plot_slices_slice_values_code(tool, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    panel_index = {
        (key.map_index, key.slice_index): index for index, key in enumerate(keys)
    }

    def value_getter(key: _PlotSlicesPanelKey) -> dict[str, typing.Any]:
        style = styles.get(
            (key.map_index, key.slice_index),
            FigurePlotSlicesPanelStyleState(
                map_index=key.map_index,
                slice_index=key.slice_index,
            ),
        )
        line_kw = {**operation.line_kw, **style.line_kw}
        if use_placeholder_labels:
            index = panel_index[(key.map_index, key.slice_index)]
            line_kw["label"] = _RawCode(
                _plot_slices_label_fstring_code(
                    operation,
                    fields,
                    source_expr=string_literal_expression(source_labels[key.map_index])
                    if key.map_index < len(source_labels)
                    else "None",
                    value_expr=_plot_slices_indexed_slice_value_code(
                        slice_values_code, key.slice_index
                    ),
                    index_expr=str(index),
                )
            )
        elif labels:
            label = labels[panel_index[(key.map_index, key.slice_index)]]
            if label:
                line_kw["label"] = label
        line_kw.pop("c", None)
        line_kw["color"] = _RawCode(
            f"line_colors[{panel_index[(key.map_index, key.slice_index)]}]"
        )
        return line_kw

    if len(keys) == 1:
        return repr(value_getter(keys[0]))
    return repr(_nested_panel_values(keys, operation.order, value_getter))


def _plot_slices_slice_values_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str:
    if operation.slice_dim:
        all_values_code = _all_coordinate_slice_values_code(operation)
        if all_values_code is not None:
            return all_values_code
        slice_values = _effective_slice_values(tool, operation)
        if slice_values:
            return repr(list(slice_values))
    return "[None]"


def _plot_slices_line_color_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    if not _plot_slices_line_colormap_active(tool, operation):
        return []
    coord = effective_line_color_coord(operation, operation.slice_dim)
    if coord is None:
        raise ValueError("Choose a coordinate to color slices")
    if coord != operation.slice_dim:
        raise ValueError(f"Cannot color slices by {coord!r}")
    _plot_slices_line_color_values(tool, operation)
    keys = _plot_slices_panel_keys(tool, operation)
    if not keys:
        return []
    slice_values_code = _plot_slices_slice_values_code(tool, operation)
    map_count, slice_count = _style_sequence_shape(keys)
    if map_count == 1:
        values_code = f"[float(slice_value) for slice_value in {slice_values_code}]"
    elif slice_count == 1:
        slice_value_code = _plot_slices_indexed_slice_value_code(slice_values_code, 0)
        values_code = f"[float({slice_value_code}) for _ in range({map_count})]"
    elif operation.order == "F":
        values_code = (
            f"[float(slice_value) for slice_value in {slice_values_code} "
            f"for _ in range({map_count})]"
        )
    else:
        values_code = (
            f"[float(slice_value) for _ in range({map_count}) "
            f"for slice_value in {slice_values_code}]"
        )
    return colormap_code_lines(
        values_code,
        effective_line_color_cmap(operation),
        trim=effective_line_color_cmap_trim(operation),
    )


def _plot_slices_label_line_kw_comprehension_code(
    operation: FigureOperationState,
    keys: tuple[_PlotSlicesPanelKey, ...],
    source_labels: tuple[str, ...],
    slice_values_code: str,
    fields: set[str],
) -> str:
    source_labels_code = repr(list(source_labels))
    map_count, slice_count = _style_sequence_shape(keys)
    if map_count == 1 and slice_count == 1:
        return _plot_slices_label_line_kw_item_code(
            operation,
            fields,
            source_expr=string_literal_expression(source_labels[0])
            if source_labels
            else "None",
            value_expr=_plot_slices_indexed_slice_value_code(slice_values_code, 0),
            index_expr="0",
        )
    if map_count == 1:
        item_code = _plot_slices_label_line_kw_item_code(
            operation,
            fields,
            source_expr=string_literal_expression(source_labels[0])
            if source_labels
            else "None",
            value_expr="slice_value",
            index_expr="slice_index",
        )
        if {"index", "number"} & fields:
            return (
                f"[{item_code} "
                f"for slice_index, slice_value in enumerate({slice_values_code})]"
            )
        return f"[{item_code} for slice_value in {slice_values_code}]"
    if slice_count == 1:
        item_code = _plot_slices_label_line_kw_item_code(
            operation,
            fields,
            source_expr="source",
            value_expr=_plot_slices_indexed_slice_value_code(slice_values_code, 0),
            index_expr="map_index",
        )
        if {"index", "number"} & fields:
            return (
                f"[{item_code} "
                f"for map_index, source in enumerate({source_labels_code})]"
            )
        return f"[{item_code} for source in {source_labels_code}]"
    item_code = _plot_slices_label_line_kw_item_code(
        operation,
        fields,
        source_expr="source",
        value_expr="slice_value",
        index_expr=_plot_slices_panel_index_expr(
            operation.order,
            map_count=map_count,
            slice_count=slice_count,
            map_index_expr="map_index",
            slice_index_expr="slice_index",
        ),
    )
    if operation.order == "F":
        return (
            "["
            "["
            f"{item_code} "
            f"for map_index, source in enumerate({source_labels_code})"
            "] "
            f"for slice_index, slice_value in enumerate({slice_values_code})"
            "]"
        )
    return (
        "["
        "["
        f"{item_code} "
        f"for slice_index, slice_value in enumerate({slice_values_code})"
        "] "
        f"for map_index, source in enumerate({source_labels_code})"
        "]"
    )


def _plot_slices_label_line_kw_item_code(
    operation: FigureOperationState,
    fields: set[str],
    *,
    source_expr: str,
    value_expr: str,
    index_expr: str,
) -> str:
    label_code = _plot_slices_label_fstring_code(
        operation,
        fields,
        source_expr=source_expr,
        value_expr=value_expr,
        index_expr=index_expr,
    )
    return _line_kw_dict_code(operation.line_kw, label_code)


def _plot_slices_styled_label_line_kw_code(
    operation: FigureOperationState,
    keys: tuple[_PlotSlicesPanelKey, ...],
    source_labels: tuple[str, ...],
    slice_values_code: str,
    styles: Mapping[tuple[int, int], FigurePlotSlicesPanelStyleState],
    fields: set[str],
) -> str:
    panel_index = {key: index for index, key in enumerate(keys)}

    def value_getter(key: _PlotSlicesPanelKey) -> dict[str, typing.Any]:
        style = styles.get(
            (key.map_index, key.slice_index),
            FigurePlotSlicesPanelStyleState(
                map_index=key.map_index,
                slice_index=key.slice_index,
            ),
        )
        label_code = _plot_slices_label_fstring_code(
            operation,
            fields,
            source_expr=string_literal_expression(source_labels[key.map_index])
            if key.map_index < len(source_labels)
            else "None",
            value_expr=_plot_slices_indexed_slice_value_code(
                slice_values_code, key.slice_index
            ),
            index_expr=str(panel_index[key]),
        )
        return {**operation.line_kw, **style.line_kw, "label": _RawCode(label_code)}

    if len(keys) == 1:
        return repr(value_getter(keys[0]))
    return repr(_nested_panel_values(keys, operation.order, value_getter))


def _plot_slices_label_fstring_code(
    operation: FigureOperationState,
    fields: set[str],
    *,
    source_expr: str,
    value_expr: str,
    index_expr: str,
) -> str:
    field_expressions: dict[str, str] = {}
    if "index" in fields:
        field_expressions["index"] = index_expr
    if "number" in fields:
        field_expressions["number"] = f"{index_expr} + 1"
    if "source" in fields:
        field_expressions["source"] = source_expr
    if "dim" in fields and operation.slice_dim:
        field_expressions["dim"] = string_literal_expression(operation.slice_dim)
    if "value" in fields and operation.slice_dim:
        field_expressions["value"] = value_expr
    if operation.slice_dim:
        coord_field = label_coord_placeholder_name(operation.slice_dim)
        if coord_field in fields:
            field_expressions[coord_field] = value_expr
    return label_fstring_code(operation.line_label_text, field_expressions)


def _plot_slices_panel_index_expr(
    order: str,
    *,
    map_count: int,
    slice_count: int,
    map_index_expr: str,
    slice_index_expr: str,
) -> str:
    if order == "F":
        return f"{slice_index_expr} * {map_count} + {map_index_expr}"
    return f"{map_index_expr} * {slice_count} + {slice_index_expr}"


def _plot_slices_indexed_slice_value_code(
    slice_values_code: str, slice_index: int
) -> str:
    return f"({slice_values_code})[{slice_index}]"


def _line_kw_dict_code(base_kwargs: Mapping[str, typing.Any], label_code: str) -> str:
    if base_kwargs:
        return f"{{**{dict(base_kwargs)!r}, 'label': {label_code}}}"
    return f"{{'label': {label_code}}}"


def _plot_slices_uses_transformed_line_maps(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    return _plot_slices_shape(tool, operation).plot_ndim == 1 and line_transform_active(
        operation
    )


def _plot_slices_panel_qsel_kwargs(
    operation: FigureOperationState,
    key: _PlotSlicesPanelKey,
    slice_values: Sequence[float] | None = None,
) -> dict[str, typing.Any]:
    kwargs = dict(operation.slice_kwargs)
    if slice_values is None:
        slice_values = operation.slice_values
    if operation.slice_dim and slice_values:
        kwargs[operation.slice_dim] = slice_values[key.slice_index]
        if operation.slice_width is not None:
            kwargs[f"{operation.slice_dim}_width"] = operation.slice_width
    return kwargs


def _plot_slices_panel_profile_data(
    data: xr.DataArray,
    operation: FigureOperationState,
    key: _PlotSlicesPanelKey,
    slice_values: Sequence[float] | None = None,
) -> xr.DataArray:
    kwargs = _plot_slices_panel_qsel_kwargs(operation, key, slice_values)
    if kwargs:
        return data.qsel(**kwargs)
    return data


def _plot_slices_line_profiles(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    maps: Sequence[xr.DataArray] | None = None,
) -> tuple[list[xr.DataArray], tuple[_PlotSlicesPanelKey, ...]]:
    operation = _normalized_selection_operation(tool, operation)
    if maps is None:
        maps = _operation_maps(tool, operation)
    keys = _plot_slices_panel_keys(tool, operation)
    slice_values = _effective_slice_values(tool, operation)
    profiles: list[xr.DataArray] = []
    profile_keys: list[_PlotSlicesPanelKey] = []
    for key in keys:
        if key.map_index >= len(maps):
            continue
        profile = _plot_slices_panel_profile_data(
            maps[key.map_index], operation, key, slice_values
        ).squeeze(drop=True)
        if profile.ndim != 1:
            continue
        profiles.append(profile)
        profile_keys.append(key)
    return profiles, tuple(profile_keys)


def _plot_slices_transformed_maps(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    maps: Sequence[xr.DataArray],
) -> list[xr.DataArray]:
    profiles, keys = _plot_slices_line_profiles(tool, operation, maps)
    transformed = transform_profiles(operation, profiles)
    slice_values = list(_effective_slice_values(tool, operation))
    if not operation.slice_dim or not slice_values:
        return transformed

    map_profiles: list[list[tuple[int, xr.DataArray]]] = [[] for _map in maps]
    for key, profile in zip(keys, transformed, strict=True):
        map_profiles[key.map_index].append((key.slice_index, profile))

    transformed_maps: list[xr.DataArray] = []
    for profiles_for_map in map_profiles:
        if len(profiles_for_map) != len(slice_values):
            continue
        ordered_profiles = [
            profile
            for _index, profile in sorted(profiles_for_map, key=lambda item: item[0])
        ]
        transformed_maps.append(
            xr.concat(
                ordered_profiles,
                dim=operation.slice_dim,
                coords="different",
                compat="equals",
            ).assign_coords({operation.slice_dim: slice_values})
        )
    return transformed_maps


def _available_plot_slices_offset_coords(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    profiles, _keys = _plot_slices_line_profiles(tool, operation)
    names: list[str] = []
    for profile in profiles:
        for name, coord in profile.coords.items():
            name_str = str(name)
            if name_str in profile.dims or name_str in names:
                continue
            if np.asarray(coord.values).reshape(-1).size == 1:
                names.append(name_str)
    return names


def _panel_norm_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    if not operation.panel_styles_enabled or not operation.panel_styles:
        return None
    keys = _plot_slices_panel_keys(tool, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    if not any(_panel_style_has_norm_override(style) for style in styles.values()):
        return None
    order = operation.norm_order or operation.cmap_order

    def value_getter(key: _PlotSlicesPanelKey) -> _RawCode:
        style = styles.get(
            (key.map_index, key.slice_index),
            FigurePlotSlicesPanelStyleState(
                map_index=key.map_index,
                slice_index=key.slice_index,
            ),
        )
        return _RawCode(_norm_code(_operation_with_panel_norm_style(operation, style)))

    if len(keys) == 1:
        return str(value_getter(keys[0]))
    return repr(_nested_panel_values(keys, order, value_getter))


def _panel_norm_uses_matplotlib_colors(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    if not operation.panel_styles_enabled or not operation.panel_styles:
        return False
    keys = _plot_slices_panel_keys(tool, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    if not any(_panel_style_has_norm_override(style) for style in styles.values()):
        return False
    for key in keys:
        style = styles.get(
            (key.map_index, key.slice_index),
            FigurePlotSlicesPanelStyleState(
                map_index=key.map_index,
                slice_index=key.slice_index,
            ),
        )
        effective_operation = _operation_with_panel_norm_style(operation, style)
        if (
            _effective_norm_name(effective_operation.norm_name)
            in _MATPLOTLIB_NORM_NAMES
        ):
            return True
    return False


class _PanelStyleEditorWidget(QtWidgets.QWidget):
    """Editor for optional per-panel image style overrides."""

    sigPanelStylesChanged = QtCore.Signal(object)

    def __init__(
        self,
        operation: FigureOperationState,
        panel_keys: tuple[_PlotSlicesPanelKey, ...],
        connect_signal: Callable[
            [QtWidgets.QWidget, typing.Any, Callable[..., None]], None
        ],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation
        self._panel_keys = panel_keys
        self._styles = _panel_style_map(operation)
        self._updating = False

        self.panel_list = QtWidgets.QListWidget(self)
        self.panel_list.setObjectName("figureComposerPlotSlicesPanelStyleList")
        self.panel_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.panel_list.setMaximumHeight(96)
        self.panel_list.setToolTip("Select one or more panels to override.")
        for key in panel_keys:
            item = QtWidgets.QListWidgetItem(self._panel_row_text(key))
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                (key.map_index, key.slice_index),
            )
            self.panel_list.addItem(item)
        if self.panel_list.count():
            self.panel_list.setCurrentRow(0)

        self.cmap_override_check = QtWidgets.QCheckBox("Override colormap", self)
        self.cmap_override_check.setObjectName("figureComposerPanelCmapOverrideCheck")
        self.cmap_override_check.setToolTip(
            "Store a colormap override for the selected panels."
        )
        self.cmap_combo = erlab.interactive.colors.ColorMapComboBox(self)
        self.cmap_combo.setObjectName("figureComposerPanelCmapCombo")
        self.cmap_combo.setToolTip("Per-panel colormap override.")
        self.cmap_combo.ensure_populated()
        self.cmap_reverse_check = QtWidgets.QCheckBox("Reverse", self)
        self.cmap_reverse_check.setObjectName("figureComposerPanelCmapReverseCheck")
        self.cmap_reverse_check.setToolTip("Append _r to the per-panel colormap.")

        self.norm_override_check = QtWidgets.QCheckBox("Override norm", self)
        self.norm_override_check.setObjectName("figureComposerPanelNormOverrideCheck")
        self.norm_override_check.setToolTip(
            "Store normalization overrides for the selected panels."
        )
        self.norm_combo = QtWidgets.QComboBox(self)
        self.norm_combo.setObjectName("figureComposerPanelNormCombo")
        self.norm_combo.addItems(list(_NORM_CHOICES))
        self.norm_combo.setToolTip("Per-panel normalization class.")

        self.gamma_edit = self._number_edit("figureComposerPanelGammaEdit")
        self.vmin_edit = self._number_edit("figureComposerPanelVminEdit")
        self.vmax_edit = self._number_edit("figureComposerPanelVmaxEdit")
        self.vcenter_edit = self._number_edit("figureComposerPanelVcenterEdit")
        self.halfrange_edit = self._number_edit("figureComposerPanelHalfrangeEdit")
        self.clip_combo = QtWidgets.QComboBox(self)
        self.clip_combo.setObjectName("figureComposerPanelClipCombo")
        self.clip_combo.addItems(["inherit", "False", "True"])
        self.clip_combo.setToolTip("Per-panel norm clip argument.")
        self.norm_kwargs_edit = QtWidgets.QLineEdit(self)
        self.norm_kwargs_edit.setObjectName("figureComposerPanelNormKwargsEdit")
        self.norm_kwargs_edit.setToolTip(
            "Extra keyword arguments for selected panel norm constructors."
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.panel_list)

        cmap_row = QtWidgets.QHBoxLayout()
        cmap_row.setContentsMargins(0, 0, 0, 0)
        cmap_row.addWidget(self.cmap_override_check)
        cmap_row.addWidget(self.cmap_combo, 1)
        cmap_row.addWidget(self.cmap_reverse_check)
        layout.addLayout(cmap_row)

        norm_row = QtWidgets.QHBoxLayout()
        norm_row.setContentsMargins(0, 0, 0, 0)
        norm_row.addWidget(self.norm_override_check)
        norm_row.addWidget(self.norm_combo, 1)
        layout.addLayout(norm_row)

        numbers = QtWidgets.QGridLayout()
        numbers.setContentsMargins(0, 0, 0, 0)
        numbers.setHorizontalSpacing(6)
        numbers.setVerticalSpacing(4)
        for row, (label, widget) in enumerate(
            (
                ("Gamma", self.gamma_edit),
                ("vmin", self.vmin_edit),
                ("vmax", self.vmax_edit),
                ("vcenter", self.vcenter_edit),
                ("halfrange", self.halfrange_edit),
                ("Clip", self.clip_combo),
                ("Norm kwargs", self.norm_kwargs_edit),
            )
        ):
            numbers.addWidget(QtWidgets.QLabel(label, self), row, 0)
            numbers.addWidget(widget, row, 1)
        layout.addLayout(numbers)

        connect_signal(self, self.panel_list.itemSelectionChanged, self._sync_controls)
        connect_signal(
            self, self.cmap_override_check.stateChanged, self._cmap_override_changed
        )
        connect_signal(self, self.cmap_combo.activated, self._cmap_changed)
        connect_signal(
            self, self.cmap_reverse_check.stateChanged, self._cmap_reverse_changed
        )
        connect_signal(
            self, self.norm_override_check.stateChanged, self._norm_override_changed
        )
        connect_signal(self, self.norm_combo.activated, self._norm_changed)
        for attr, edit in (
            ("norm_gamma", self.gamma_edit),
            ("vmin", self.vmin_edit),
            ("vmax", self.vmax_edit),
            ("vcenter", self.vcenter_edit),
            ("halfrange", self.halfrange_edit),
        ):
            connect_signal(
                self,
                edit.editingFinished,
                lambda attr=attr, edit=edit: self._number_changed(attr, edit),
            )
        connect_signal(self, self.clip_combo.activated, self._clip_changed)
        connect_signal(
            self, self.norm_kwargs_edit.editingFinished, self._norm_kwargs_changed
        )
        self._sync_controls()

    def styles(self) -> tuple[FigurePlotSlicesPanelStyleState, ...]:
        valid_keys = {(key.map_index, key.slice_index) for key in self._panel_keys}
        return tuple(
            self._styles[key]
            for key in sorted(self._styles)
            if key in valid_keys and _panel_style_has_overrides(self._styles[key])
        )

    @staticmethod
    def _number_edit(object_name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setObjectName(object_name)
        edit.setToolTip("Leave blank to inherit the global value.")
        return edit

    def _panel_row_text(self, key: _PlotSlicesPanelKey) -> str:
        style = _panel_style_from_map(self._styles, key)
        parts = [key.label]
        if _panel_style_has_cmap_override(style):
            parts.append(_effective_panel_cmap(self._operation, style))
        if _panel_style_has_norm_override(style):
            norm_operation = _operation_with_panel_norm_style(self._operation, style)
            parts.append(_effective_norm_name(norm_operation.norm_name))
        return " | ".join(parts)

    def _selected_keys(self) -> tuple[_PlotSlicesPanelKey, ...]:
        by_index = {(key.map_index, key.slice_index): key for key in self._panel_keys}
        keys: list[_PlotSlicesPanelKey] = []
        for item in self.panel_list.selectedItems():
            raw_key = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if raw_key in by_index:
                keys.append(by_index[raw_key])
        if keys:
            return tuple(keys)
        if self._panel_keys:
            return (self._panel_keys[0],)
        return ()

    def _selected_styles(self) -> tuple[FigurePlotSlicesPanelStyleState, ...]:
        return tuple(
            _panel_style_from_map(self._styles, key) for key in self._selected_keys()
        )

    @staticmethod
    def _common_value(values: tuple[typing.Any, ...]) -> typing.Any:
        if not values:
            return None
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        return _MISSING

    def _sync_controls(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            styles = self._selected_styles()
            cmap_override = self._common_value(
                tuple(_panel_style_has_cmap_override(style) for style in styles)
            )
            self._set_check_state(self.cmap_override_check, cmap_override)
            cmap_enabled = cmap_override is True
            self.cmap_combo.setEnabled(cmap_enabled)
            self.cmap_reverse_check.setEnabled(cmap_enabled)
            self._set_cmap_value(styles)

            norm_override = self._common_value(
                tuple(_panel_style_has_norm_override(style) for style in styles)
            )
            self._set_check_state(self.norm_override_check, norm_override)
            norm_enabled = norm_override is True
            self.norm_combo.setEnabled(norm_enabled)
            norm_name = self._set_norm_value(styles)
            self._sync_norm_fields(styles, norm_enabled, norm_name)
        finally:
            self._updating = False

    @staticmethod
    def _set_check_state(check: QtWidgets.QCheckBox, value: object) -> None:
        with QtCore.QSignalBlocker(check):
            check.setTristate(value is _MISSING)
            if value is _MISSING:
                check.setCheckState(QtCore.Qt.CheckState.PartiallyChecked)
            else:
                check.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if value is True
                    else QtCore.Qt.CheckState.Unchecked
                )

    def _set_cmap_value(
        self, styles: tuple[FigurePlotSlicesPanelStyleState, ...]
    ) -> None:
        values = tuple(
            _cmap_base_and_reverse(style.cmap)[0]
            if style.cmap is not None
            else _effective_panel_cmap(self._operation, style)
            for style in styles
        )
        reversed_values = tuple(
            _cmap_base_and_reverse(style.cmap)[1] if style.cmap is not None else False
            for style in styles
        )
        value = self._common_value(values)
        reversed_value = self._common_value(reversed_values)
        with QtCore.QSignalBlocker(self.cmap_combo):
            if value is _MISSING:
                self._set_combo_mixed(self.cmap_combo)
            else:
                self._remove_combo_mixed(self.cmap_combo)
                self.cmap_combo.setCurrentText(str(value))
        self._set_check_state(self.cmap_reverse_check, reversed_value)

    def _set_norm_value(
        self, styles: tuple[FigurePlotSlicesPanelStyleState, ...]
    ) -> str | None:
        values = tuple(
            _effective_norm_name(
                _operation_with_panel_norm_style(self._operation, style).norm_name
            )
            for style in styles
        )
        value = self._common_value(values)
        with QtCore.QSignalBlocker(self.norm_combo):
            if value is _MISSING:
                self._set_combo_mixed(self.norm_combo)
                return None
            self._remove_combo_mixed(self.norm_combo)
            self.norm_combo.setCurrentText(str(value))
            return str(value)

    def _sync_norm_fields(
        self,
        styles: tuple[FigurePlotSlicesPanelStyleState, ...],
        enabled: bool,
        norm_name: str | None,
    ) -> None:
        norm_fields = set(_norm_kwarg_fields(norm_name)) if norm_name else set()
        for attr, edit in (
            ("norm_gamma", self.gamma_edit),
            ("vmin", self.vmin_edit),
            ("vmax", self.vmax_edit),
            ("vcenter", self.vcenter_edit),
            ("halfrange", self.halfrange_edit),
        ):
            field_name = "gamma" if attr == "norm_gamma" else attr
            field_enabled = enabled and field_name in norm_fields
            edit.setEnabled(field_enabled)
            values = tuple(getattr(style, attr) for style in styles)
            value = self._common_value(values)
            with QtCore.QSignalBlocker(edit):
                edit.setText("" if value in (None, _MISSING) else f"{value:g}")
                edit.setPlaceholderText(
                    "(multiple values)" if value is _MISSING else "inherit"
                )
                edit.setModified(False)
        self.clip_combo.setEnabled(enabled and "clip" in norm_fields)
        clip_values = tuple(style.norm_clip for style in styles)
        clip_value = self._common_value(clip_values)
        with QtCore.QSignalBlocker(self.clip_combo):
            if clip_value is _MISSING:
                self._set_combo_mixed(self.clip_combo)
            else:
                self._remove_combo_mixed(self.clip_combo)
                self.clip_combo.setCurrentText(
                    "inherit" if clip_value is None else str(clip_value)
                )
        self.norm_kwargs_edit.setEnabled(enabled)
        kwargs_values = tuple(style.norm_kwargs for style in styles)
        kwargs_value = self._common_value(kwargs_values)
        with QtCore.QSignalBlocker(self.norm_kwargs_edit):
            self.norm_kwargs_edit.setText(
                "" if kwargs_value in (None, _MISSING) else _format_dict(kwargs_value)
            )
            self.norm_kwargs_edit.setPlaceholderText(
                "(multiple values)" if kwargs_value is _MISSING else "optional"
            )
            self.norm_kwargs_edit.setModified(False)

    @staticmethod
    def _set_combo_mixed(combo: QtWidgets.QComboBox) -> None:
        if combo.findData(_MISSING) < 0:
            combo.insertItem(0, "(multiple values)", _MISSING)
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
        combo.setCurrentIndex(0)

    @staticmethod
    def _remove_combo_mixed(combo: QtWidgets.QComboBox) -> None:
        index = combo.findData(_MISSING)
        if index >= 0:
            combo.removeItem(index)

    def _cmap_override_changed(self, state: int) -> None:
        check_state = QtCore.Qt.CheckState(state)
        if self._updating or check_state == QtCore.Qt.CheckState.PartiallyChecked:
            return
        if check_state == QtCore.Qt.CheckState.Checked:
            cmap = self._operation.cmap or _current_options().colors.cmap.name
            self._update_selected_styles({"cmap": cmap})
        else:
            self._update_selected_styles({"cmap": None})

    def _cmap_override_active(self) -> bool:
        return self.cmap_override_check.checkState() == QtCore.Qt.CheckState.Checked

    def _cmap_changed(self, _index: int) -> None:
        if (
            self._updating
            or not self._cmap_override_active()
            or self.cmap_combo.currentData() is _MISSING
        ):
            return
        base = self.cmap_combo.currentText()
        reverse = self.cmap_reverse_check.checkState() == QtCore.Qt.CheckState.Checked
        self._update_selected_styles({"cmap": _cmap_with_reverse(base, reverse)})

    def _cmap_reverse_changed(self, state: int) -> None:
        check_state = QtCore.Qt.CheckState(state)
        if (
            self._updating
            or not self._cmap_override_active()
            or check_state == QtCore.Qt.CheckState.PartiallyChecked
        ):
            return
        base = self.cmap_combo.currentText()
        if self.cmap_combo.currentData() is _MISSING:
            base = self._operation.cmap or _current_options().colors.cmap.name
        reverse = check_state == QtCore.Qt.CheckState.Checked
        self._update_selected_styles({"cmap": _cmap_with_reverse(base, reverse)})

    def _norm_override_changed(self, state: int) -> None:
        check_state = QtCore.Qt.CheckState(state)
        if self._updating or check_state == QtCore.Qt.CheckState.PartiallyChecked:
            return
        if check_state == QtCore.Qt.CheckState.Checked:
            self._update_selected_styles(
                {"norm_name": self._operation.norm_name or _POWER_NORM_NAME}
            )
        else:
            self._update_selected_styles(
                {
                    "norm_name": None,
                    "norm_gamma": None,
                    "norm_clip": None,
                    "norm_kwargs": {},
                    "vmin": None,
                    "vmax": None,
                    "vcenter": None,
                    "halfrange": None,
                }
            )

    def _norm_override_active(self) -> bool:
        return self.norm_override_check.checkState() == QtCore.Qt.CheckState.Checked

    def _norm_changed(self, _index: int) -> None:
        if (
            self._updating
            or not self._norm_override_active()
            or self.norm_combo.currentData() is _MISSING
        ):
            return
        self._update_selected_styles({"norm_name": self.norm_combo.currentText()})

    def _number_changed(self, attr: str, edit: QtWidgets.QLineEdit) -> None:
        if (
            self._updating
            or not self._norm_override_active()
            or (edit.placeholderText() == "(multiple values)" and not edit.isModified())
        ):
            return
        text = edit.text().strip()
        self._update_selected_styles({attr: float(text) if text else None})

    def _clip_changed(self, _index: int) -> None:
        if (
            self._updating
            or not self._norm_override_active()
            or self.clip_combo.currentData() is _MISSING
        ):
            return
        text = self.clip_combo.currentText()
        self._update_selected_styles({"norm_clip": _norm_clip_from_text(text)})

    def _norm_kwargs_changed(self) -> None:
        if (
            self._updating
            or not self._norm_override_active()
            or (
                self.norm_kwargs_edit.placeholderText() == "(multiple values)"
                and not self.norm_kwargs_edit.isModified()
            )
        ):
            return
        self._update_selected_styles(
            {"norm_kwargs": _dict_from_text(self.norm_kwargs_edit.text())}
        )

    def _update_selected_styles(self, updates: dict[str, typing.Any]) -> None:
        for key in self._selected_keys():
            style_key = (key.map_index, key.slice_index)
            style = self._styles.get(
                style_key,
                FigurePlotSlicesPanelStyleState(
                    map_index=key.map_index,
                    slice_index=key.slice_index,
                ),
            )
            next_style = style.model_copy(update=updates)
            if _panel_style_has_overrides(next_style):
                self._styles[style_key] = next_style
            else:
                self._styles.pop(style_key, None)
        styles = self.styles()
        self._operation = self._operation.model_copy(
            update={
                "panel_styles_enabled": bool(styles),
                "panel_styles": styles,
            }
        )
        self.sigPanelStylesChanged.emit(styles)
        self._sync_rows()
        self._sync_controls()

    def _sync_rows(self) -> None:
        for row, key in enumerate(self._panel_keys):
            item = self.panel_list.item(row)
            if item is not None:
                item.setText(self._panel_row_text(key))


class _PanelLineStyleEditorWidget(QtWidgets.QWidget):
    """Editor for optional per-panel 1D line style overrides."""

    sigPanelStylesChanged = QtCore.Signal(object)

    def __init__(
        self,
        operation: FigureOperationState,
        panel_keys: tuple[_PlotSlicesPanelKey, ...],
        connect_signal: Callable[
            [QtWidgets.QWidget, typing.Any, Callable[..., None]], None
        ],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._operation = operation
        self._panel_keys = panel_keys
        self._styles = _panel_style_map(operation)
        self._updating = False

        self.panel_list = QtWidgets.QListWidget(self)
        self.panel_list.setObjectName("figureComposerPlotSlicesPanelLineStyleList")
        self.panel_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.panel_list.setMaximumHeight(96)
        for key in panel_keys:
            item = QtWidgets.QListWidgetItem(self._panel_row_text(key))
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                (key.map_index, key.slice_index),
            )
            self.panel_list.addItem(item)
        if self.panel_list.count():
            self.panel_list.setCurrentRow(0)

        self.color_edit = _ColorLineEditWidget(parent=self)
        self.color_edit.setLineEditObjectName("figureComposerPanelLineColorEdit")
        self.color_edit.setColorButtonObjectName("figureComposerPanelLineColorButton")
        self.style_combo = QtWidgets.QComboBox(self)
        self.style_combo.setObjectName("figureComposerPanelLineStyleCombo")
        configure_style_combo(self.style_combo, LINE_STYLE_OPTIONS, None)
        self.width_edit = self._line_edit("figureComposerPanelLineWidthEdit")
        self.marker_combo = QtWidgets.QComboBox(self)
        self.marker_combo.setObjectName("figureComposerPanelLineMarkerCombo")
        configure_style_combo(self.marker_combo, LINE_MARKER_OPTIONS, None)
        self.marker_size_edit = self._line_edit("figureComposerPanelLineMarkerSizeEdit")
        self.marker_face_edit = _ColorLineEditWidget(parent=self)
        self.marker_face_edit.setLineEditObjectName(
            "figureComposerPanelLineMarkerFaceEdit"
        )
        self.marker_face_edit.setColorButtonObjectName(
            "figureComposerPanelLineMarkerFaceButton"
        )
        self.marker_edge_edit = _ColorLineEditWidget(parent=self)
        self.marker_edge_edit.setLineEditObjectName(
            "figureComposerPanelLineMarkerEdgeEdit"
        )
        self.marker_edge_edit.setColorButtonObjectName(
            "figureComposerPanelLineMarkerEdgeButton"
        )
        self.line_kwargs_edit = self._line_edit("figureComposerPanelLineKwEdit")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.panel_list)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        form.addRow("Color", self.color_edit)
        form.addRow("Line style", self.style_combo)
        form.addRow("Line width", self.width_edit)
        form.addRow("Marker", self.marker_combo)
        form.addRow("Marker size", self.marker_size_edit)
        form.addRow("Marker face", self.marker_face_edit)
        form.addRow("Marker edge", self.marker_edge_edit)
        form.addRow("Line kwargs", self.line_kwargs_edit)
        layout.addLayout(form)

        connect_signal(self, self.panel_list.itemSelectionChanged, self._sync_controls)
        connect_signal(
            self,
            self.color_edit.editingFinished,
            lambda: self._line_kw_changed(
                "color",
                color_kw_value_from_text(self.color_edit.text()),
                aliases=("c",),
            ),
        )
        connect_signal(
            self,
            self.style_combo.activated,
            lambda _index: self._line_kw_changed(
                "linestyle",
                style_combo_value(self.style_combo),
                aliases=("ls",),
            ),
        )
        connect_signal(
            self,
            self.width_edit.editingFinished,
            lambda: self._line_kw_changed(
                "linewidth",
                self._optional_float(self.width_edit.text()),
                aliases=("lw",),
            ),
        )
        connect_signal(
            self,
            self.marker_combo.activated,
            lambda _index: self._line_kw_changed(
                "marker", style_combo_value(self.marker_combo)
            ),
        )
        connect_signal(
            self,
            self.marker_size_edit.editingFinished,
            lambda: self._line_kw_changed(
                "markersize",
                self._optional_float(self.marker_size_edit.text()),
                aliases=("ms",),
            ),
        )
        connect_signal(
            self,
            self.marker_face_edit.editingFinished,
            lambda: self._line_kw_changed(
                "markerfacecolor",
                color_kw_value_from_text(self.marker_face_edit.text()),
                aliases=("mfc",),
            ),
        )
        connect_signal(
            self,
            self.marker_edge_edit.editingFinished,
            lambda: self._line_kw_changed(
                "markeredgecolor",
                color_kw_value_from_text(self.marker_edge_edit.text()),
                aliases=("mec",),
            ),
        )
        connect_signal(
            self, self.line_kwargs_edit.editingFinished, self._extra_line_kw_changed
        )
        self._sync_controls()

    def styles(self) -> tuple[FigurePlotSlicesPanelStyleState, ...]:
        valid_keys = {(key.map_index, key.slice_index) for key in self._panel_keys}
        return tuple(
            self._styles[key]
            for key in sorted(self._styles)
            if key in valid_keys and _panel_style_has_overrides(self._styles[key])
        )

    @staticmethod
    def _line_edit(object_name: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit()
        edit.setObjectName(object_name)
        edit.setPlaceholderText("inherit")
        return edit

    @staticmethod
    def _optional_float(text: str) -> float | None:
        stripped = text.strip()
        return None if not stripped else float(stripped)

    def _panel_row_text(self, key: _PlotSlicesPanelKey) -> str:
        style = _panel_style_from_map(self._styles, key)
        color = line_kw_text(
            self._operation.model_copy(update={"line_kw": style.line_kw}),
            "color",
            "c",
        )
        return key.label if not color else f"{key.label} | {color}"

    def _selected_keys(self) -> tuple[_PlotSlicesPanelKey, ...]:
        by_index = {(key.map_index, key.slice_index): key for key in self._panel_keys}
        keys: list[_PlotSlicesPanelKey] = []
        for item in self.panel_list.selectedItems():
            raw_key = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if raw_key in by_index:
                keys.append(by_index[raw_key])
        if keys:
            return tuple(keys)
        if self._panel_keys:
            return (self._panel_keys[0],)
        return ()

    def _selected_styles(self) -> tuple[FigurePlotSlicesPanelStyleState, ...]:
        return tuple(
            _panel_style_from_map(self._styles, key) for key in self._selected_keys()
        )

    @staticmethod
    def _common_value(values: tuple[typing.Any, ...]) -> typing.Any:
        if not values:
            return None
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        return _MISSING

    def _line_value(self, style: FigurePlotSlicesPanelStyleState, *keys: str) -> str:
        operation = self._operation.model_copy(update={"line_kw": style.line_kw})
        return line_kw_text(operation, *keys)

    def _line_style_value(
        self, style: FigurePlotSlicesPanelStyleState, *keys: str
    ) -> str | None:
        operation = self._operation.model_copy(update={"line_kw": style.line_kw})
        return line_kw_style_value(operation, *keys)

    def _sync_controls(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            styles = self._selected_styles()
            self._set_color_widget(
                self.color_edit,
                self._common_value(
                    tuple(self._line_value(style, "color", "c") for style in styles)
                ),
            )
            self._set_combo(
                self.style_combo,
                self._common_value(
                    tuple(
                        self._line_style_value(style, "linestyle", "ls")
                        for style in styles
                    )
                ),
            )
            self._set_line_edit(
                self.width_edit,
                self._common_value(
                    tuple(
                        self._line_value(style, "linewidth", "lw") for style in styles
                    )
                ),
            )
            self._set_combo(
                self.marker_combo,
                self._common_value(
                    tuple(self._line_style_value(style, "marker") for style in styles)
                ),
            )
            self._set_line_edit(
                self.marker_size_edit,
                self._common_value(
                    tuple(
                        self._line_value(style, "markersize", "ms") for style in styles
                    )
                ),
            )
            self._set_color_widget(
                self.marker_face_edit,
                self._common_value(
                    tuple(
                        self._line_value(style, "markerfacecolor", "mfc")
                        for style in styles
                    )
                ),
            )
            self._set_color_widget(
                self.marker_edge_edit,
                self._common_value(
                    tuple(
                        self._line_value(style, "markeredgecolor", "mec")
                        for style in styles
                    )
                ),
            )
            self._set_line_edit(
                self.line_kwargs_edit,
                self._common_value(
                    tuple(
                        _format_dict(self._extra_line_kw(style.line_kw))
                        for style in styles
                    )
                ),
            )
        finally:
            self._updating = False

    @staticmethod
    def _set_line_edit(edit: QtWidgets.QLineEdit, value: typing.Any) -> None:
        with QtCore.QSignalBlocker(edit):
            edit.setText("" if value in (None, _MISSING) else str(value))
            edit.setPlaceholderText(
                "(multiple values)" if value is _MISSING else "inherit"
            )
            edit.setModified(False)

    def _set_color_widget(
        self, widget: _ColorLineEditWidget, value: typing.Any
    ) -> None:
        self._set_line_edit(widget.line_edit, value)
        widget.setText("" if value in (None, _MISSING) else str(value))

    @staticmethod
    def _set_combo(combo: QtWidgets.QComboBox, value: typing.Any) -> None:
        with QtCore.QSignalBlocker(combo):
            if value is _MISSING:
                _PanelStyleEditorWidget._set_combo_mixed(combo)
            else:
                _PanelStyleEditorWidget._remove_combo_mixed(combo)
                set_style_combo_value(combo, None if value is None else str(value))

    @classmethod
    def _extra_line_kw(cls, line_kw: dict[str, typing.Any]) -> dict[str, typing.Any]:
        return {
            key: value
            for key, value in line_kw.items()
            if key not in CONTROLLED_LINE_KW_KEYS
        }

    def _line_kw_changed(
        self,
        key: str,
        value: typing.Any,
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        if self._updating:
            return
        self._update_selected_line_kw({key: value}, clear_keys=(key, *aliases))

    def _extra_line_kw_changed(self) -> None:
        if self._updating or (
            self.line_kwargs_edit.placeholderText() == "(multiple values)"
            and not self.line_kwargs_edit.isModified()
        ):
            return
        extra_kwargs = _dict_from_text(self.line_kwargs_edit.text())
        for key in CONTROLLED_LINE_KW_KEYS:
            extra_kwargs.pop(key, None)
        self._update_selected_extra_line_kw(extra_kwargs)

    def _update_selected_line_kw(
        self,
        updates: dict[str, typing.Any],
        *,
        clear_keys: tuple[str, ...],
    ) -> None:
        for panel_key in self._selected_keys():
            style_key = (panel_key.map_index, panel_key.slice_index)
            style = self._styles.get(
                style_key,
                FigurePlotSlicesPanelStyleState(
                    map_index=panel_key.map_index,
                    slice_index=panel_key.slice_index,
                ),
            )
            line_kw = dict(style.line_kw)
            for clear_key in clear_keys:
                line_kw.pop(clear_key, None)
            line_kw.update(
                {
                    key: value
                    for key, value in updates.items()
                    if value not in (None, "")
                }
            )
            self._replace_style(
                style_key,
                style.model_copy(update={"line_kw": line_kw}),
            )
        self._emit_styles_changed()

    def _update_selected_extra_line_kw(
        self, extra_kwargs: dict[str, typing.Any]
    ) -> None:
        for panel_key in self._selected_keys():
            style_key = (panel_key.map_index, panel_key.slice_index)
            style = self._styles.get(
                style_key,
                FigurePlotSlicesPanelStyleState(
                    map_index=panel_key.map_index,
                    slice_index=panel_key.slice_index,
                ),
            )
            line_kw = {
                key: value
                for key, value in style.line_kw.items()
                if key in CONTROLLED_LINE_KW_KEYS
            }
            line_kw.update(extra_kwargs)
            self._replace_style(
                style_key,
                style.model_copy(update={"line_kw": line_kw}),
            )
        self._emit_styles_changed()

    def _replace_style(
        self,
        style_key: tuple[int, int],
        style: FigurePlotSlicesPanelStyleState,
    ) -> None:
        if _panel_style_has_overrides(style):
            self._styles[style_key] = style
        else:
            self._styles.pop(style_key, None)

    def _emit_styles_changed(self) -> None:
        styles = self.styles()
        self._operation = self._operation.model_copy(
            update={
                "panel_styles_enabled": bool(styles),
                "panel_styles": styles,
            }
        )
        self.sigPanelStylesChanged.emit(styles)
        self._sync_rows()
        self._sync_controls()

    def _sync_rows(self) -> None:
        for row, key in enumerate(self._panel_keys):
            item = self.panel_list.item(row)
            if item is not None:
                item.setText(self._panel_row_text(key))


def _is_slice_kwarg_key(key: typing.Any, dims: Iterable[str]) -> bool:
    if not isinstance(key, str):
        return False
    dim_names = set(dims)
    if key in dim_names and key not in _PLOT_SLICES_EXPLICIT_KWARGS:
        return True
    return key.endswith("_width") and key[: -len("_width")] in dim_names


def _split_slice_kwargs(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    kwargs: dict[typing.Any, typing.Any],
) -> tuple[dict[str, typing.Any], dict[typing.Any, typing.Any]]:
    dims = _operation_dim_names(tool, operation)
    slice_kwargs: dict[str, typing.Any] = {}
    extra_kwargs: dict[typing.Any, typing.Any] = {}
    for key, value in kwargs.items():
        if _is_slice_kwarg_key(key, dims):
            slice_kwargs[str(key)] = value
        else:
            extra_kwargs[key] = value
    return slice_kwargs, extra_kwargs


def _selection_values(value: typing.Any) -> tuple[float, ...] | None:
    if isinstance(value, str | bytes | slice):
        return None
    if isinstance(value, np.ndarray):
        raw_values = value.ravel().tolist()
    elif isinstance(value, list | tuple):
        raw_values = value
    else:
        raw_values = (value,)
    try:
        return tuple(float(item) for item in raw_values)
    except (TypeError, ValueError):
        return None


def _selection_width(value: typing.Any) -> float | None:
    values = _selection_values(value)
    if not values:
        return None
    unique_values = set(values)
    if len(unique_values) != 1:
        return None
    return unique_values.pop()


def _pop_promotable_width(
    slice_kwargs: dict[str, typing.Any], dim: str
) -> float | None:
    width_key = f"{dim}_width"
    value = slice_kwargs.get(width_key, _MISSING)
    if value is _MISSING:
        return None
    width = _selection_width(value)
    if width is None:
        return None
    slice_kwargs.pop(width_key)
    return width


def _selection_updates_from_kwargs(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    slice_kwargs: dict[typing.Any, typing.Any],
    extra_kwargs: dict[typing.Any, typing.Any],
) -> dict[str, typing.Any]:
    parsed_slice_kwargs, slice_extra_kwargs = _split_slice_kwargs(
        tool, operation, slice_kwargs
    )
    extra_slice_kwargs, parsed_extra_kwargs = _split_slice_kwargs(
        tool, operation, extra_kwargs
    )
    next_slice_kwargs = {**parsed_slice_kwargs, **extra_slice_kwargs}
    next_extra_kwargs = {**slice_extra_kwargs, **parsed_extra_kwargs}
    next_slice_dim = operation.slice_dim
    next_slice_values = operation.slice_values
    next_slice_width = operation.slice_width

    if next_slice_dim is not None:
        value = next_slice_kwargs.get(next_slice_dim, _MISSING)
        values = None if value is _MISSING else _selection_values(value)
        if values:
            next_slice_values = values
            next_slice_kwargs.pop(next_slice_dim)
        width = _pop_promotable_width(next_slice_kwargs, next_slice_dim)
        if width is not None:
            next_slice_width = width
    else:
        dims = set(_operation_dim_names(tool, operation))
        candidates = [
            (key, values)
            for key, value in next_slice_kwargs.items()
            if key in dims and (values := _selection_values(value))
        ]
        if len(candidates) == 1:
            next_slice_dim, next_slice_values = candidates[0]
            next_slice_kwargs.pop(next_slice_dim)
            next_slice_width = _pop_promotable_width(next_slice_kwargs, next_slice_dim)

    return {
        "slice_dim": next_slice_dim,
        "slice_values": next_slice_values,
        "slice_width": next_slice_width,
        "slice_kwargs": next_slice_kwargs,
        "extra_kwargs": next_extra_kwargs,
    }


def _effective_slice_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    slice_kwargs, _extra_kwargs = _split_slice_kwargs(
        tool, operation, operation.extra_kwargs
    )
    return {**slice_kwargs, **operation.slice_kwargs}


def _effective_extra_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[typing.Any, typing.Any]:
    _slice_kwargs, extra_kwargs = _split_slice_kwargs(
        tool, operation, operation.extra_kwargs
    )
    return extra_kwargs


def _normalized_selection_operation(
    tool: FigureComposerTool, operation: FigureOperationState
) -> FigureOperationState:
    updates = _selection_updates_from_kwargs(
        tool,
        operation,
        _effective_slice_kwargs(tool, operation),
        _effective_extra_kwargs(tool, operation),
    )
    return operation.model_copy(update=updates)


def _add_plot_slices_line_color_controls(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    row = QtWidgets.QWidget(page)
    row_layout = QtWidgets.QVBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(4)

    mode_mixed = tool._batch_is_mixed(operation, lambda target: target.line_color_mode)
    mode_combo = tool._combo(
        list(_LINE_COLOR_MODE_TEXT.values()),
        None if mode_mixed else _line_color_mode_text(operation.line_color_mode),
        lambda text: tool._update_current_operation_rebuild(
            line_color_mode=_line_color_mode_from_text(text)
        ),
        parent=page,
        mixed=mode_mixed,
    )
    mode_combo.setObjectName("figureComposerPlotSlicesLineColorModeCombo")
    mode_combo.setToolTip(
        "Manual: use one Matplotlib line color.\n"
        "By coordinate: map the slice coordinate values through a colormap."
    )
    row_layout.addWidget(mode_combo)

    if not mode_mixed and line_colormap_active(operation):
        _add_plot_slices_coordinate_color_controls(tool, operation, page, row_layout)
    else:
        _add_plot_slices_manual_color_controls(tool, operation, page, row_layout)

    tool._add_form_row(
        layout,
        "Color",
        row,
        "Choose a manual line color or color 1D panels from coordinate values.",
    )


def _add_plot_slices_manual_color_controls(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QVBoxLayout,
) -> None:
    line_color_text, line_color_mixed = tool._batch_text(
        operation,
        lambda target: line_kw_text(target, "color", "c") or "",
        str,
    )
    line_color_edit = _ColorLineEditWidget(
        line_color_text,
        parent=page,
    )
    line_color_edit.setLineEditObjectName("figureComposerPlotSlicesLineColorEdit")
    line_color_edit.setColorButtonObjectName("figureComposerPlotSlicesLineColorButton")
    line_color_edit.setToolTip(
        "Matplotlib color stored as line_kw color for 1D panels."
    )
    tool._apply_mixed_line_edit(line_color_edit.line_edit, line_color_mixed)
    tool._connect_value_signal(
        line_color_edit,
        line_color_edit.editingFinished,
        line_color_edit.text,
        lambda text: update_current_line_kw(
            tool,
            "color",
            color_kw_value_from_text(text),
            aliases=("c",),
            clear_stale_cmap=True,
            clear_stale_line_colormap=True,
        ),
        unchanged_mixed=lambda: tool._line_edit_batch_unchanged(
            line_color_edit.line_edit
        ),
    )
    layout.addWidget(line_color_edit)


def _add_plot_slices_coordinate_color_controls(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QVBoxLayout,
) -> None:
    coord_options = _available_plot_slices_line_color_coords(tool, operation)
    coord_options_match = tool._batch_options_match(
        operation, lambda target: _available_plot_slices_line_color_coords(tool, target)
    )
    coord_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_color_coord
    )
    coord_combo = tool._optional_name_combo(
        coord_options,
        None
        if coord_mixed
        else effective_line_color_coord(operation, operation.slice_dim),
        "Choose coordinate",
        lambda value: tool._update_current_operation(line_color_coord=value),
        parent=page,
        mixed=coord_mixed,
        enabled=coord_options_match,
    )
    coord_combo.setObjectName("figureComposerPlotSlicesLineColorCoordCombo")
    coord_combo.setToolTip(
        "Numeric scalar coordinate used to color each 1D panel.\n"
        "The slice dimension is selected by default."
    )

    cmap_row = QtWidgets.QWidget(page)
    cmap_layout = QtWidgets.QHBoxLayout(cmap_row)
    cmap_layout.setContentsMargins(0, 0, 0, 0)
    cmap_layout.setSpacing(6)

    cmap_combo = erlab.interactive.colors.ColorMapComboBox(cmap_row)
    cmap_combo.setObjectName("figureComposerPlotSlicesLineColorCmapCombo")
    cmap_combo.setToolTip("Colormap used for coordinate-colored 1D panels.")
    cmap_combo.ensure_populated()
    cmap_base, cmap_reverse = _cmap_base_and_reverse(operation.line_color_cmap)
    cmap_combo.setCurrentText(cmap_base)

    reverse_check = QtWidgets.QCheckBox("Reverse", cmap_row)
    reverse_check.setObjectName("figureComposerPlotSlicesLineColorCmapReverseCheck")
    reverse_check.setToolTip("Reverse the selected line colormap.")
    reverse_check.setChecked(operation.line_color_cmap_reverse or cmap_reverse)

    tool._connect_editor_signal(
        cmap_combo,
        cmap_combo.activated,
        lambda _index: _update_current_plot_slices_line_color_cmap(
            tool, cmap_combo.currentText(), reverse_check.isChecked()
        ),
    )
    tool._connect_editor_signal(
        reverse_check,
        reverse_check.stateChanged,
        lambda state: _update_current_plot_slices_line_color_cmap(
            tool,
            cmap_combo.currentText(),
            QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked,
        ),
    )

    cmap_layout.addWidget(cmap_combo, 1)
    cmap_layout.addWidget(reverse_check)

    trim_lower_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_color_cmap_trim_lower
    )
    trim_upper_mixed = tool._batch_is_mixed(
        operation, lambda target: target.line_color_cmap_trim_upper
    )
    trim_lower, trim_upper = line_color_cmap_trim_control_values(operation)
    trim_row = QtWidgets.QWidget(page)
    trim_layout = QtWidgets.QHBoxLayout(trim_row)
    trim_layout.setContentsMargins(0, 0, 0, 0)
    trim_layout.setSpacing(6)
    trim_tooltip = "Skip fractions from the low and high ends of the colormap."
    trim_label = QtWidgets.QLabel("Trim", trim_row)
    trim_label.setToolTip(trim_tooltip)
    lower_spin = _line_color_trim_spin(
        "figureComposerPlotSlicesLineColorCmapTrimLowerSpin",
        0.0 if trim_lower_mixed else trim_lower,
        trim_tooltip,
        trim_row,
    )
    upper_spin = _line_color_trim_spin(
        "figureComposerPlotSlicesLineColorCmapTrimUpperSpin",
        0.0 if trim_upper_mixed else trim_upper,
        trim_tooltip,
        trim_row,
    )
    tool._connect_value_signal(
        lower_spin,
        lower_spin.valueChanged,
        float,
        lambda value: tool._update_current_operation(line_color_cmap_trim_lower=value),
    )
    tool._connect_value_signal(
        upper_spin,
        upper_spin.valueChanged,
        float,
        lambda value: tool._update_current_operation(line_color_cmap_trim_upper=value),
    )
    trim_layout.addWidget(trim_label)
    trim_layout.addWidget(QtWidgets.QLabel("Low", trim_row))
    trim_layout.addWidget(
        tool._mixed_value_widget(lower_spin, mixed=trim_lower_mixed, parent=page)
    )
    trim_layout.addWidget(QtWidgets.QLabel("High", trim_row))
    trim_layout.addWidget(
        tool._mixed_value_widget(upper_spin, mixed=trim_upper_mixed, parent=page)
    )
    trim_layout.addStretch(1)

    layout.addWidget(coord_combo)
    layout.addWidget(cmap_row)
    layout.addWidget(trim_row)


def _line_color_trim_spin(
    object_name: str,
    value: float,
    tooltip: str,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox(parent)
    spin.setObjectName(object_name)
    spin.setRange(0.0, LINE_COLOR_CMAP_TRIM_MAX)
    spin.setDecimals(2)
    spin.setSingleStep(0.05)
    spin.setKeyboardTracking(False)
    spin.setValue(value)
    spin.setToolTip(tooltip)
    line_edit = spin.lineEdit()
    if line_edit is not None:
        line_edit.setToolTip(tooltip)
    return spin


def _update_current_plot_slices_line_color_cmap(
    tool: FigureComposerTool, base: str, reverse: bool
) -> None:
    if tool._updating_controls:
        return
    tool._update_current_operation(
        line_color_cmap=_cmap_with_reverse(base, False),
        line_color_cmap_reverse=reverse,
    )


def _build_plot_slices_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[tuple[str, str, QtWidgets.QWidget]]:
    operation = _normalized_selection_operation(tool, operation)
    shape = _plot_slices_shape(tool, operation)
    panel_kind = _plot_slices_batch_panel_kind(tool, operation)
    is_line_plot = panel_kind == _PLOT_SLICES_PANEL_LINE
    is_image_plot = panel_kind == _PLOT_SLICES_PANEL_IMAGE
    is_mixed_panel_kind = panel_kind == _PLOT_SLICES_PANEL_MIXED
    selection_page, selection_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesSelectionPage"
    )
    view_page, view_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesViewPage"
    )
    colors_page, colors_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesColorsPage"
    )
    transform_page, transform_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesTransformPage"
    )
    advanced_page, advanced_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesAdvancedPage"
    )
    tool.operation_editor = selection_page
    tool.operation_editor_layout = selection_layout

    tool._add_form_section(
        selection_layout,
        "Dimensions",
        object_name="figureComposerPlotSlicesSelectionDimensionsSection",
    )
    shape_summary = QtWidgets.QLabel(
        "\n".join(
            (
                f"Input dims: {shape.source_text}",
                f"Plotted dims: {shape.panel_text}",
                *(("Status: " + shape.selection_text,) if not shape.valid else ()),
            )
        ),
        selection_page,
    )
    shape_summary.setObjectName("figureComposerPlotSlicesShapeSummary")
    shape_summary.setWordWrap(True)
    if not shape.valid:
        shape_summary.setForegroundRole(QtGui.QPalette.ColorRole.Link)
    tool._add_form_row(
        selection_layout,
        "Summary",
        shape_summary,
        "Shows the input dimensions and the dimensions plotted by this step.",
    )

    tool._add_form_section(
        selection_layout,
        "Slice values",
        object_name="figureComposerPlotSlicesSelectionValuesSection",
    )
    dims = _available_source_dims(tool._source_data, operation.sources)
    dim_mixed = tool._batch_is_mixed(operation, lambda target: target.slice_dim)
    dim_combo = tool._combo(
        ["", *dims],
        None if dim_mixed else operation.slice_dim or "",
        lambda text: tool._update_current_operation_rebuild(slice_dim=text or None),
        mixed=dim_mixed,
    )
    dim_combo.setObjectName("figureComposerPlotSlicesDimensionCombo")
    tool._add_form_row(
        selection_layout,
        "Dimension",
        dim_combo,
        "Data dimension passed as the slice keyword to plot_slices.",
    )

    values_mode_mixed = tool._batch_is_mixed(
        operation, lambda target: target.slice_values_mode
    )
    values_mode_combo = tool._combo(
        tuple(_SLICE_VALUES_MODE_LABELS.values()),
        None
        if values_mode_mixed
        else _slice_values_mode_text(operation.slice_values_mode),
        lambda text: tool._update_current_operation_rebuild(
            slice_values_mode=_slice_values_mode_from_text(text)
        ),
        parent=selection_page,
        mixed=values_mode_mixed,
    )
    values_mode_combo.setObjectName("figureComposerPlotSlicesValuesModeCombo")
    values_mode_combo.setToolTip(
        "Choose manual values or all values from the dimension coordinate."
    )
    tool._add_form_row(
        selection_layout,
        "Values",
        values_mode_combo,
        values_mode_combo.toolTip(),
    )
    if not values_mode_mixed and _use_all_coordinate_slice_values(operation):
        coordinate_summary = QtWidgets.QLabel(
            _all_coordinate_slice_values_summary(tool, operation),
            selection_page,
        )
        coordinate_summary.setObjectName("figureComposerPlotSlicesCoordinateSummary")
        coordinate_summary.setWordWrap(True)
        tool._add_form_row(
            selection_layout,
            "Coordinate",
            coordinate_summary,
            "Shows the coordinate values that will be passed to plot_slices.",
        )

        thin_mixed = tool._batch_is_mixed(
            operation, lambda target: target.slice_values_thin
        )
        thin_spin = erlab.interactive.utils.BetterSpinBox(
            selection_page,
            integer=True,
            minimum=1,
            value=operation.slice_values_thin,
        )
        thin_spin.setObjectName("figureComposerPlotSlicesValuesThinSpin")
        thin_spin.setToolTip("Keep every Nth coordinate value.")
        tool._connect_value_signal(
            thin_spin,
            thin_spin.valueChanged,
            int,
            lambda value: tool._update_current_operation_rebuild(
                slice_values_thin=value
            ),
        )
        tool._add_form_row(
            selection_layout,
            "Thin",
            tool._mixed_value_widget(
                thin_spin, mixed=thin_mixed, parent=selection_page
            ),
            thin_spin.toolTip(),
        )
    elif not values_mode_mixed:
        values_text, values_mixed = tool._batch_text(
            operation, lambda target: target.slice_values, _format_tuple
        )
        values_edit = tool._line_edit(values_text, parent=selection_page)
        tool._apply_mixed_line_edit(values_edit, values_mixed)
        values_edit.setObjectName("figureComposerPlotSlicesValuesEdit")
        tool._connect_line_edit_finished(
            values_edit,
            lambda text: tool._update_current_operation_rebuild(
                slice_values=_float_tuple_from_text(text)
            ),
        )
        tool._add_form_row(
            selection_layout,
            "Manual",
            values_edit,
            "Comma-separated coordinate values to select along the dimension.",
        )

    width_text, width_mixed = tool._batch_text(
        operation,
        lambda target: target.slice_width,
        lambda value: "" if value is None else f"{value:g}",
    )
    width_edit = tool._line_edit(width_text)
    tool._apply_mixed_line_edit(width_edit, width_mixed)
    width_edit.setObjectName("figureComposerPlotSlicesWidthEdit")
    tool._connect_line_edit_finished(
        width_edit,
        lambda text: tool._update_current_operation_rebuild(
            slice_width=float(text) if text.strip() else None
        ),
    )
    tool._add_form_row(
        selection_layout,
        "Width",
        width_edit,
        "Optional qsel width around each selected value before plotting.",
    )

    slice_kwargs_text, slice_kwargs_mixed = tool._batch_text(
        operation, lambda target: target.slice_kwargs, _format_dict
    )
    slice_kwargs_edit = tool._line_edit(slice_kwargs_text)
    tool._apply_mixed_line_edit(slice_kwargs_edit, slice_kwargs_mixed)
    slice_kwargs_edit.setObjectName("figureComposerPlotSlicesSliceKwargsEdit")
    tool._connect_line_edit_finished(
        slice_kwargs_edit,
        lambda text: _update_current_slice_kwargs(tool, text),
    )
    tool._add_form_row(
        selection_layout,
        "Extra kwargs",
        slice_kwargs_edit,
        "Additional plot_slices selection kwargs passed to qsel.\n"
        "Use dimension keys such as kx=slice(-1, 1) or beta=0.",
    )

    tool._add_form_section(
        view_layout,
        "Panels",
        object_name="figureComposerPlotSlicesViewPanelsSection",
    )
    order_mixed = tool._batch_is_mixed(operation, lambda target: target.order)
    order_combo = tool._combo(
        ["C", "F"],
        None if order_mixed else operation.order,
        lambda text: tool._update_current_operation_rebuild(order=text),
        parent=view_page,
        mixed=order_mixed,
    )
    order_combo.setObjectName("figureComposerOrderCombo")
    tool._add_form_row(
        view_layout,
        "Order",
        order_combo,
        "C places sources by row and selected values by column. F places selected "
        "values by row and sources by column.",
    )

    transpose_mixed = tool._batch_is_mixed(operation, lambda target: target.transpose)
    transpose_check = tool._check_box(
        operation.transpose,
        lambda checked: tool._update_current_operation_rebuild(transpose=checked),
        mixed=transpose_mixed,
    )
    transpose_check.setObjectName("figureComposerTransposeCheck")
    transpose_check.setText("")
    transpose_check.setToolTip("Swap the plotted x/y orientation.")
    tool._add_form_row(
        view_layout,
        "Transpose",
        transpose_check,
        "Swap the plotted x/y orientation.",
    )

    limit_controls: list[tuple[str, QtWidgets.QWidget, str]] = []
    for label, attr in (("x", "xlim"), ("y", "ylim")):
        text, mixed = tool._batch_text(
            operation,
            _operation_field_getter(attr),
            _format_plot_limit,
        )
        edit = tool._line_edit(text)
        edit.setObjectName(f"figureComposerPlotSlices{label.upper()}LimEdit")
        tool._apply_mixed_line_edit(edit, mixed)
        placeholder = (
            "" if mixed else _plot_slices_limit_placeholder(tool, operation, attr)
        )
        if placeholder:
            edit.setPlaceholderText(placeholder)
        tool._connect_line_edit_finished(
            edit,
            _plot_limit_update_callback(tool, attr),
        )
        limit_controls.append(
            (
                label,
                edit,
                f"Optional {attr}: one number for symmetric limits, "
                "or two comma-separated numbers for lower and upper limits.",
            )
        )
    tool._add_compound_form_row(
        view_layout,
        "Limits",
        limit_controls,
        "Optional x/y plot limits for this step.",
    )

    tool._add_form_section(
        view_layout,
        "Axes",
        object_name="figureComposerPlotSlicesViewAxesSection",
    )
    crop_mixed = tool._batch_is_mixed(operation, lambda target: target.crop)
    crop_check = tool._check_box(
        operation.crop,
        lambda checked: tool._update_current_operation(crop=checked),
        parent=view_page,
        mixed=crop_mixed,
    )
    crop_check.setObjectName("figureComposerPlotSlicesCropCheck")
    crop_check.setText("")
    crop_check.setToolTip("Crop each slice to explicit x/y limits before plotting.")
    tool._add_form_row(
        view_layout,
        "Crop",
        crop_check,
        "Crop each slice to explicit x/y limits before plotting.",
    )

    axis_mixed = tool._batch_is_mixed(operation, lambda target: target.axis)
    axis_combo = tool._combo(
        ["auto", "on", "off", "equal", "scaled", "tight", "image", "square"],
        None if axis_mixed else operation.axis,
        lambda text: tool._update_current_operation(axis=text),
        parent=view_page,
        mixed=axis_mixed,
    )
    axis_combo.setObjectName("figureComposerAxisCombo")
    tool._add_form_row(
        view_layout,
        "Axis",
        axis_combo,
        "Matplotlib axis mode passed through plot_slices.",
    )

    label_options_widget = QtWidgets.QWidget(view_page)
    label_options_layout = QtWidgets.QHBoxLayout(label_options_widget)
    label_options_layout.setContentsMargins(0, 0, 0, 0)
    show_labels_mixed = tool._batch_is_mixed(
        operation, lambda target: target.show_all_labels
    )
    show_labels_check = tool._check_box(
        operation.show_all_labels,
        lambda checked: tool._update_current_operation(show_all_labels=checked),
        mixed=show_labels_mixed,
    )
    show_labels_check.setText("All labels")
    show_labels_check.setToolTip("Ask plot_slices to show labels on every axis.")
    annotate_mixed = tool._batch_is_mixed(operation, lambda target: target.annotate)
    annotate_check = tool._check_box(
        operation.annotate,
        lambda checked: tool._update_current_operation(annotate=checked),
        mixed=annotate_mixed,
    )
    annotate_check.setText("Annotate")
    annotate_check.setToolTip("Show the slice-value annotation text.")
    label_options_layout.addWidget(show_labels_check)
    label_options_layout.addWidget(annotate_check)
    label_options_layout.addStretch(1)
    tool._add_form_row(
        view_layout,
        "Labels",
        label_options_widget,
        "Label and annotation visibility options for plot_slices.",
    )

    annotate_kwargs_text, annotate_kwargs_mixed = tool._batch_text(
        operation, lambda target: target.annotate_kw, _format_dict
    )
    annotate_kwargs_edit = tool._line_edit(
        annotate_kwargs_text,
        parent=view_page,
    )
    tool._apply_mixed_line_edit(annotate_kwargs_edit, annotate_kwargs_mixed)
    annotate_kwargs_edit.setObjectName("figureComposerAnnotateKwEdit")
    tool._connect_line_edit_finished(
        annotate_kwargs_edit,
        lambda text: tool._update_current_operation(annotate_kw=_dict_from_text(text)),
    )
    tool._add_form_row(
        view_layout,
        "Annotation",
        annotate_kwargs_edit,
        "Dict literal or keyword arguments forwarded as annotate_kw.",
    )

    if is_line_plot:
        tool._add_form_section(
            colors_layout,
            "Legend",
            object_name="figureComposerPlotSlicesStyleLegendSection",
        )
        labels_text, labels_mixed = tool._batch_text(
            operation,
            label_editor_text,
            str,
        )
        labels_edit = tool._line_edit(labels_text, parent=colors_page)
        tool._apply_mixed_line_edit(labels_edit, labels_mixed)
        labels_edit.setObjectName("figureComposerPlotSlicesLineLabelsEdit")
        tool._connect_line_edit_finished(
            labels_edit,
            lambda text: update_current_line_label_text(tool, text),
        )
        label_contexts = _plot_slices_line_label_contexts(tool, operation)
        labels_widget = legend_label_input_widget(
            labels_edit,
            label_contexts,
            item_name="slice",
            button_object_name="figureComposerPlotSlicesLineLabelsHelpButton",
            parent=colors_page,
        )
        tool._add_form_row(
            colors_layout,
            "Labels",
            labels_widget,
            label_text_tooltip(label_contexts, item_name="slice"),
        )

        tool._add_form_section(
            colors_layout,
            "Line",
            object_name="figureComposerPlotSlicesStyleLineSection",
        )
        _add_plot_slices_line_color_controls(
            tool, operation, colors_page, colors_layout
        )

        line_style_mixed = tool._batch_is_mixed(
            operation, lambda target: line_kw_style_value(target, "linestyle", "ls")
        )
        line_style_combo = tool._optional_name_combo(
            LINE_STYLE_OPTIONS,
            None
            if line_style_mixed
            else line_kw_style_value(operation, "linestyle", "ls"),
            LINE_STYLE_DEFAULT_LABEL,
            lambda text: update_current_line_kw(
                tool, "linestyle", text, aliases=("ls",)
            ),
            parent=colors_page,
            mixed=line_style_mixed,
        )
        line_style_combo.setObjectName("figureComposerPlotSlicesLineStyleCombo")
        line_width_mixed = tool._batch_is_mixed(
            operation, lambda target: line_kw_text(target, "linewidth", "lw")
        )
        line_width_spin = optional_positive_spinbox(
            None if line_width_mixed else line_kw_float(operation, "linewidth", "lw"),
            parent=colors_page,
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
        line_width_spin.setObjectName("figureComposerPlotSlicesLineWidthSpin")
        line_width_row_widget = tool._mixed_value_widget(
            line_width_spin,
            mixed=line_width_mixed,
            parent=colors_page,
        )
        tool._add_compound_form_row(
            colors_layout,
            "Stroke",
            (
                (
                    "Style",
                    line_style_combo,
                    "Matplotlib linestyle for 1D plot_slices panels.",
                ),
                (
                    "Width",
                    line_width_row_widget,
                    "Matplotlib linewidth for 1D plot_slices panels.",
                ),
            ),
            "Line style controls for 1D plot_slices panels.",
        )

        marker_mixed = tool._batch_is_mixed(
            operation, lambda target: line_kw_style_value(target, "marker")
        )
        marker_combo = tool._optional_name_combo(
            LINE_MARKER_OPTIONS,
            None if marker_mixed else line_kw_style_value(operation, "marker"),
            LINE_STYLE_DEFAULT_LABEL,
            lambda text: update_current_line_kw(tool, "marker", text),
            parent=colors_page,
            mixed=marker_mixed,
        )
        marker_combo.setObjectName("figureComposerPlotSlicesMarkerCombo")
        marker_size_mixed = tool._batch_is_mixed(
            operation, lambda target: line_kw_text(target, "markersize", "ms")
        )
        marker_size_spin = optional_positive_spinbox(
            None if marker_size_mixed else line_kw_float(operation, "markersize", "ms"),
            parent=colors_page,
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
        marker_size_spin.setObjectName("figureComposerPlotSlicesMarkerSizeSpin")
        marker_size_row_widget = tool._mixed_value_widget(
            marker_size_spin,
            mixed=marker_size_mixed,
            parent=colors_page,
        )
        tool._add_compound_form_row(
            colors_layout,
            "Marker",
            (
                (
                    "Style",
                    marker_combo,
                    "Matplotlib marker style for 1D plot_slices panels.",
                ),
                (
                    "Size",
                    marker_size_row_widget,
                    "Matplotlib marker size for 1D plot_slices panels.",
                ),
            ),
            "Marker style controls for 1D plot_slices panels.",
        )

        marker_face_text, marker_face_mixed = tool._batch_text(
            operation,
            lambda target: line_kw_text(target, "markerfacecolor", "mfc"),
            str,
        )
        marker_inherited_color = (
            None
            if line_colormap_active(operation)
            else line_kw_text(operation, "color", "c") or None
        )
        marker_face_edit = _ColorLineEditWidget(
            marker_face_text,
            parent=colors_page,
            inherited_color=marker_inherited_color,
        )
        marker_face_edit.setLineEditObjectName(
            "figureComposerPlotSlicesMarkerFaceColorEdit"
        )
        marker_face_edit.setColorButtonObjectName(
            "figureComposerPlotSlicesMarkerFaceColorButton"
        )
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
            parent=colors_page,
            inherited_color=marker_inherited_color,
        )
        marker_edge_edit.setLineEditObjectName(
            "figureComposerPlotSlicesMarkerEdgeColorEdit"
        )
        marker_edge_edit.setColorButtonObjectName(
            "figureComposerPlotSlicesMarkerEdgeColorButton"
        )
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
            colors_layout,
            "Colors",
            (
                (
                    "Face",
                    marker_face_edit,
                    "Matplotlib marker face color for 1D plot_slices panels.",
                ),
                (
                    "Edge",
                    marker_edge_edit,
                    "Matplotlib marker edge color for 1D plot_slices panels.",
                ),
            ),
            "Marker face and edge colors for 1D plot_slices panels.",
        )

        line_kwargs_text, line_kwargs_mixed = tool._batch_text(
            operation, extra_line_kw, _format_dict
        )
        line_kwargs_edit = tool._line_edit(line_kwargs_text, parent=colors_page)
        tool._apply_mixed_line_edit(line_kwargs_edit, line_kwargs_mixed)
        line_kwargs_edit.setObjectName("figureComposerPlotSlicesLineKwEdit")
        tool._connect_line_edit_finished(
            line_kwargs_edit,
            lambda text: update_current_extra_line_kw(tool, _dict_from_text(text)),
        )
        tool._add_form_row(
            colors_layout,
            "Kwargs",
            line_kwargs_edit,
            "Additional Matplotlib Line2D kwargs not covered by the controls above.",
        )

        add_line_transform_controls(
            tool,
            operation,
            transform_page,
            transform_layout,
            object_prefix="figureComposerPlotSlicesLine",
            offset_coord_options=lambda target: _available_plot_slices_offset_coords(
                tool, target
            ),
        )

        tool._add_form_section(
            colors_layout,
            "Fill",
            object_name="figureComposerPlotSlicesStyleFillSection",
        )

        gradient_mixed = tool._batch_is_mixed(operation, lambda target: target.gradient)
        gradient_check = tool._check_box(
            operation.gradient,
            lambda checked: tool._update_current_operation(gradient=checked),
            parent=colors_page,
            mixed=gradient_mixed,
        )
        gradient_check.setObjectName("figureComposerGradientCheck")
        gradient_check.setText("Gradient Fill")
        tool._add_form_row(
            colors_layout,
            "Gradient",
            gradient_check,
            "Fill the area under each 1D line with a gradient.",
        )

        gradient_kwargs_text, gradient_kwargs_mixed = tool._batch_text(
            operation, lambda target: target.gradient_kw, _format_dict
        )
        gradient_kwargs_edit = tool._line_edit(gradient_kwargs_text, parent=colors_page)
        tool._apply_mixed_line_edit(gradient_kwargs_edit, gradient_kwargs_mixed)
        gradient_kwargs_edit.setObjectName("figureComposerGradientKwEdit")
        tool._connect_line_edit_finished(
            gradient_kwargs_edit,
            lambda text: tool._update_current_operation(
                gradient_kw=_dict_from_text(text)
            ),
        )
        tool._add_form_row(
            colors_layout,
            "Kwargs",
            gradient_kwargs_edit,
            "Dict literal or keyword arguments forwarded as gradient_kw.",
        )

        tool._add_form_section(
            colors_layout,
            "Panel overrides",
            object_name="figureComposerPlotSlicesStylePanelOverridesSection",
        )
        panel_styles_mixed = tool._batch_is_mixed(
            operation, lambda target: target.panel_styles_enabled
        )
        panel_styles_check = tool._check_box(
            operation.panel_styles_enabled,
            lambda checked: _update_current_panel_styles_enabled(tool, checked),
            parent=colors_page,
            mixed=panel_styles_mixed,
        )
        panel_styles_check.setObjectName("figureComposerPlotSlicesPanelStylesCheck")
        panel_styles_check.setText("Use panel-specific line styles")
        tool._add_form_row(
            colors_layout,
            "Per-panel",
            panel_styles_check,
            "Override line color, style, marker, or kwargs for individual panels.",
        )
        if operation.panel_styles_enabled:
            panel_editor = _PanelLineStyleEditorWidget(
                operation,
                _plot_slices_panel_keys(tool, operation),
                tool._connect_editor_signal,
                colors_page,
            )
            panel_editor.setObjectName("figureComposerPlotSlicesPanelLineStyleEditor")
            tool._mark_editor_control(panel_editor)
            tool._connect_value_signal(
                panel_editor,
                panel_editor.sigPanelStylesChanged,
                lambda styles: styles,
                lambda styles: _update_current_panel_styles(tool, styles),
            )
            tool._add_form_row(
                colors_layout,
                "Styles",
                panel_editor,
                "Select panels and set optional line-style overrides.",
            )
    elif is_image_plot:
        tool._add_form_section(
            colors_layout,
            "Image color",
            object_name="figureComposerPlotSlicesColorsImageColorSection",
        )
        cmap_base, cmap_reversed = _cmap_base_and_reverse(operation.cmap)
        cmap_widget = QtWidgets.QWidget(colors_page)
        cmap_layout = QtWidgets.QHBoxLayout(cmap_widget)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.setSpacing(4)
        cmap_mixed = tool._batch_is_mixed(
            operation, lambda target: _cmap_base_and_reverse(target.cmap)[0]
        )
        reverse_mixed = tool._batch_is_mixed(
            operation, lambda target: _cmap_base_and_reverse(target.cmap)[1]
        )
        cmap_combo = erlab.interactive.colors.ColorMapComboBox(cmap_widget)
        tool._mark_editor_control(cmap_combo)
        cmap_combo.setObjectName("figureComposerCmapCombo")
        cmap_combo.setToolTip("Colormap passed to plot_slices.")
        cmap_combo.default_cmap = None if cmap_mixed else cmap_base
        with QtCore.QSignalBlocker(cmap_combo):
            cmap_combo.ensure_populated()
            if cmap_mixed:
                tool._set_combo_mixed_placeholder(cmap_combo)
            else:
                cmap_combo.setCurrentText(cmap_base)
        cmap_reverse_check = tool._check_box(
            cmap_reversed,
            lambda checked: _update_current_cmap(tool, reverse=checked),
            parent=cmap_widget,
            mixed=reverse_mixed,
        )
        cmap_reverse_check.setText("Reverse")
        cmap_reverse_check.setObjectName("figureComposerCmapReverseCheck")
        cmap_reverse_check.setToolTip("Append _r to the selected Matplotlib colormap.")

        tool._connect_editor_signal(
            cmap_combo,
            cmap_combo.activated,
            lambda _index, combo=cmap_combo: (
                None
                if tool._mixed_combo_text(combo.currentText())
                else _update_current_cmap(tool, base=combo.currentText())
            ),
        )
        cmap_combo.blockSignals(False)
        cmap_layout.addWidget(cmap_combo, 1)
        cmap_layout.addWidget(cmap_reverse_check)
        tool._add_form_row(
            colors_layout,
            "Colormap",
            cmap_widget,
            "Colormap and reverse-colormap controls for image panels.",
        )

        norm_combo = tool._combo(
            _norm_combo_choices(operation.norm_name),
            tool._batch_combo_text(
                operation,
                lambda target: target.norm_name,
                _norm_combo_text,
            ),
            lambda text: _update_current_norm_name(
                tool, _norm_name_from_combo_text(text)
            ),
            parent=colors_page,
            mixed=tool._batch_is_mixed(
                operation, lambda target: _norm_combo_text(target.norm_name)
            ),
        )
        norm_combo.setObjectName("figureComposerNormCombo")
        norm_combo.setToolTip("Color normalization used for image plot_slices panels.")
        tool._add_form_row(colors_layout, "Norm", norm_combo, norm_combo.toolTip())

        norm_fields = _norm_kwarg_fields(operation.norm_name)
        if "gamma" in norm_fields:
            gamma_mixed = tool._batch_is_mixed(
                operation, lambda target: _norm_gamma_value(target)
            )
            gamma_widget = erlab.interactive.colors.ColorMapGammaWidget(
                colors_page,
                value=_norm_gamma_value(operation),
                spin_cls=erlab.interactive.utils.BetterSpinBox,
            )
            gamma_widget.setObjectName("figureComposerGammaWidget")
            gamma_widget.setToolTip("Gamma value for the selected normalization.")
            tool._connect_editor_signal(
                gamma_widget,
                gamma_widget.valueChanged,
                lambda value: _update_current_norm_gamma(tool, value),
            )
            gamma_row_widget = tool._mixed_value_widget(
                gamma_widget,
                mixed=gamma_mixed,
                parent=colors_page,
            )
            tool._add_form_row(
                colors_layout,
                "Gamma",
                gamma_row_widget,
                gamma_widget.toolTip(),
            )

        norm_number_fields = {
            "vmin": ("vmin", operation.vmin, "Lower color-normalization bound."),
            "vmax": ("vmax", operation.vmax, "Upper color-normalization bound."),
            "vcenter": (
                "vcenter",
                operation.vcenter,
                "Center value for diverging normalization classes.",
            ),
            "halfrange": (
                "halfrange",
                operation.halfrange,
                "Symmetric half-range for centered ERLab normalization classes.",
            ),
        }
        color_limit_placeholders = (
            _plot_slices_color_limit_placeholders(tool, operation)
            if "vmin" in norm_fields or "vmax" in norm_fields
            else {}
        )
        norm_number_widgets: dict[str, tuple[str, QtWidgets.QWidget, str]] = {}
        for attr in ("vmin", "vmax", "vcenter", "halfrange"):
            if attr not in norm_fields:
                continue
            label, _value, tooltip = norm_number_fields[attr]
            text, mixed = tool._batch_text(
                operation,
                _operation_field_getter(attr),
                lambda value: "" if value is None else str(value),
            )
            edit = tool._line_edit(text)
            tool._apply_mixed_line_edit(edit, mixed)
            edit.setObjectName(f"figureComposer{attr[0].upper()}{attr[1:]}NormEdit")
            placeholder = (
                ""
                if mixed
                else color_limit_placeholders.get(
                    attr, _norm_field_placeholder(operation, attr)
                )
            )
            if placeholder:
                edit.setPlaceholderText(placeholder)
            tool._connect_line_edit_finished(
                edit,
                _norm_number_update_callback(tool, attr),
            )
            norm_number_widgets[attr] = (label, edit, tooltip)

        if "vmin" in norm_number_widgets and "vmax" in norm_number_widgets:
            tool._add_compound_form_row(
                colors_layout,
                "Color limits",
                (
                    norm_number_widgets.pop("vmin"),
                    norm_number_widgets.pop("vmax"),
                ),
                "Lower and upper color-normalization bounds.",
            )
        if "vcenter" in norm_number_widgets and "halfrange" in norm_number_widgets:
            tool._add_compound_form_row(
                colors_layout,
                "Center/range",
                (
                    norm_number_widgets.pop("vcenter"),
                    norm_number_widgets.pop("halfrange"),
                ),
                "Center and half-range for centered color normalization.",
            )
        for label, edit, tooltip in norm_number_widgets.values():
            tool._add_form_row(colors_layout, label, edit, tooltip)

        if "clip" in norm_fields:
            clip_mixed = tool._batch_is_mixed(
                operation, lambda target: target.norm_clip
            )
            clip_combo = tool._combo(
                ["default", "False", "True"],
                None if clip_mixed else _norm_clip_text(operation.norm_clip),
                lambda text: tool._update_current_operation(
                    norm_clip=_norm_clip_from_text(text)
                ),
                parent=colors_page,
                mixed=clip_mixed,
            )
            clip_combo.setObjectName("figureComposerNormClipCombo")
            tool._add_form_row(
                colors_layout,
                "Clip",
                clip_combo,
                "clip argument for the selected normalization object.",
            )

        norm_kwargs_text, norm_kwargs_mixed = tool._batch_text(
            operation, lambda target: target.norm_kwargs, _format_dict
        )
        norm_kwargs_edit = tool._line_edit(norm_kwargs_text)
        tool._apply_mixed_line_edit(norm_kwargs_edit, norm_kwargs_mixed)
        norm_kwargs_edit.setObjectName("figureComposerNormKwargsEdit")
        tool._connect_line_edit_finished(
            norm_kwargs_edit,
            lambda text: _update_current_norm_kwargs(tool, text),
        )
        tool._add_form_row(
            colors_layout,
            "Norm kwargs",
            norm_kwargs_edit,
            "Extra dict literal or keyword arguments for the norm constructor.",
        )

        same_limits_mixed = tool._batch_is_mixed(
            operation, lambda target: target.same_limits
        )
        same_limits_combo = tool._combo(
            ["False", "True", "row", "col", "all"],
            None if same_limits_mixed else str(operation.same_limits),
            lambda text: tool._update_current_operation(
                same_limits=_bool_or_text(text)
            ),
            parent=colors_page,
            mixed=same_limits_mixed,
        )
        same_limits_combo.setObjectName("figureComposerSameLimitsCombo")
        tool._add_form_row(
            colors_layout,
            "Match limits",
            same_limits_combo,
            "Control plot_slices same_limits for image color scaling.",
        )

        tool._add_form_section(
            colors_layout,
            "Colorbar",
            object_name="figureComposerPlotSlicesColorsColorbarSection",
        )
        colorbar_mixed = tool._batch_is_mixed(operation, lambda target: target.colorbar)
        colorbar_combo = tool._combo(
            ["none", "right", "rightspan", "all"],
            None if colorbar_mixed else operation.colorbar,
            lambda text: tool._update_current_operation(colorbar=text),
            parent=colors_page,
            mixed=colorbar_mixed,
        )
        tool._add_form_row(
            colors_layout,
            "Placement",
            colorbar_combo,
            "Where plot_slices should place colorbars for image panels.",
        )
        colorbar_kwargs_text, colorbar_kwargs_mixed = tool._batch_text(
            operation, lambda target: target.colorbar_kw, _format_dict
        )
        colorbar_kwargs_edit = tool._line_edit(
            colorbar_kwargs_text,
            parent=colors_page,
        )
        tool._apply_mixed_line_edit(colorbar_kwargs_edit, colorbar_kwargs_mixed)
        colorbar_kwargs_edit.setObjectName("figureComposerColorbarKwEdit")
        tool._connect_line_edit_finished(
            colorbar_kwargs_edit,
            lambda text: tool._update_current_operation(
                colorbar_kw=_dict_from_text(text)
            ),
        )
        tool._add_form_row(
            colors_layout,
            "Kwargs",
            colorbar_kwargs_edit,
            "Dict literal or keyword arguments forwarded as colorbar_kw.",
        )

        tool._add_form_section(
            colors_layout,
            "Panel overrides",
            object_name="figureComposerPlotSlicesColorsPanelOverridesSection",
        )
        panel_styles_mixed = tool._batch_is_mixed(
            operation, lambda target: target.panel_styles_enabled
        )
        panel_styles_check = tool._check_box(
            operation.panel_styles_enabled,
            lambda checked: _update_current_panel_styles_enabled(tool, checked),
            parent=colors_page,
            mixed=panel_styles_mixed,
        )
        panel_styles_check.setObjectName("figureComposerPlotSlicesPanelStylesCheck")
        panel_styles_check.setText("Use panel-specific styles")
        tool._add_form_row(
            colors_layout,
            "Per-panel",
            panel_styles_check,
            "Override colormaps and normalization for individual image panels.",
        )
        if operation.panel_styles_enabled:
            panel_editor = _PanelStyleEditorWidget(
                operation,
                _plot_slices_panel_keys(tool, operation),
                tool._connect_editor_signal,
                colors_page,
            )
            panel_editor.setObjectName("figureComposerPlotSlicesPanelStyleEditor")
            tool._mark_editor_control(panel_editor)
            tool._connect_value_signal(
                panel_editor,
                panel_editor.sigPanelStylesChanged,
                lambda styles: styles,
                lambda styles: _update_current_panel_styles(tool, styles),
            )
            tool._add_form_row(
                colors_layout,
                "Styles",
                panel_editor,
                "Select panels and set optional colormap or norm overrides.",
            )
    elif is_mixed_panel_kind:
        mixed_label = QtWidgets.QLabel(
            "Selected plot_slices steps produce both image and line panels. "
            "Select only image steps or only line steps to edit color controls.",
            colors_page,
        )
        mixed_label.setObjectName("figureComposerPlotSlicesMixedColorsLabel")
        mixed_label.setWordWrap(True)
        mixed_label.setEnabled(False)
        tool._add_form_row(
            colors_layout,
            "Colors",
            mixed_label,
            "Color controls are hidden for mixed image/line plot_slices selection.",
        )

    extra_text, extra_mixed = tool._batch_text(
        operation,
        lambda target: _effective_extra_kwargs(tool, target),
        _format_dict,
    )
    extra_edit = tool._line_edit(extra_text)
    tool._apply_mixed_line_edit(extra_edit, extra_mixed)
    extra_edit.setObjectName("figureComposerExtraKwEdit")
    tool._connect_line_edit_finished(
        extra_edit,
        lambda text: _update_current_extra_kwargs(tool, text),
    )
    tool._add_form_row(
        advanced_layout,
        "Extra kwargs",
        extra_edit,
        "Dict literal or keyword arguments merged into the plot_slices call.",
    )
    sections = [
        ("selection", "Selection", selection_page),
        ("view", "View", view_page),
    ]
    if is_line_plot:
        sections.extend(
            (
                ("colors", "Style", colors_page),
                ("transform", "Transform", transform_page),
            )
        )
    else:
        sections.append(
            (
                "colors",
                "Colors" if is_image_plot else "Style",
                colors_page,
            )
        )
    sections.append(("advanced", "Other", advanced_page))
    return sections


def _bool_or_text(text: str) -> bool | str:
    if text == "True":
        return True
    if text == "False":
        return False
    return text


def _optional_number_or_text(attr: str, text: str) -> float | str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if attr in {"cmap", "norm_name"}:
        return stripped
    return float(stripped)


def _plot_limit_update_callback(
    tool: FigureComposerTool, attr: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        tool._update_current_operation(**{attr: _plot_limit_from_text(text)})

    return update


def _norm_number_update_callback(
    tool: FigureComposerTool, attr: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        tool._update_current_operation(**{attr: _optional_number_or_text(attr, text)})

    return update


def _operation_field_value(operation: FigureOperationState, attr: str) -> typing.Any:
    return getattr(operation, attr)


def _operation_field_getter(
    attr: str,
) -> Callable[[FigureOperationState], typing.Any]:
    def getter(operation: FigureOperationState) -> typing.Any:
        return _operation_field_value(operation, attr)

    return getter


def _plot_slices_rendered_value(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    reader: Callable[[Sequence[matplotlib.axes.Axes]], typing.Any],
) -> typing.Any:
    if tool._preview_render_update_pending or tool._operation_has_invalid_axes(
        operation
    ):
        return None
    axs = _live_layout_axes(tool)
    if axs is None:
        return None
    try:
        axes = _iter_axes(
            _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)
        )
        if not axes:
            return None
        return reader(axes)
    except Exception:
        return None


def _plot_slices_limit_placeholder(
    tool: FigureComposerTool, operation: FigureOperationState, attr: str
) -> str:
    if attr not in {"xlim", "ylim"} or getattr(operation, attr) is not None:
        return ""
    limits = _plot_slices_rendered_value(
        tool,
        operation,
        lambda axes: axes[0].get_xlim() if attr == "xlim" else axes[0].get_ylim(),
    )
    if limits is None:
        return ""
    limit_pair = _rendered_float_pair(limits)
    return "" if limit_pair is None else _format_pair(limit_pair)


def _plot_slices_color_limit_placeholders(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, str]:
    if operation.vmin is not None and operation.vmax is not None:
        return {}
    clim = _plot_slices_rendered_value(tool, operation, _first_mappable_clim)
    clim_pair = _rendered_float_pair(clim)
    if clim_pair is None:
        return {}
    placeholders: dict[str, str] = {}
    if operation.vmin is None and (vmin := _format_placeholder_number(clim_pair[0])):
        placeholders["vmin"] = vmin
    if operation.vmax is None and (vmax := _format_placeholder_number(clim_pair[1])):
        placeholders["vmax"] = vmax
    return placeholders


def _rendered_float_pair(value: object) -> tuple[float, float] | None:
    if isinstance(value, str | bytes) or not isinstance(value, tuple | list):
        return None
    if len(value) != 2:
        return None
    try:
        first = float(value[0])
        second = float(value[1])
    except (TypeError, ValueError):
        return None
    return first, second


def _first_mappable_clim(
    axes: Sequence[matplotlib.axes.Axes],
) -> tuple[float, float] | None:
    for axis in axes:
        for mappable in (*axis.images, *axis.collections):
            get_clim = getattr(mappable, "get_clim", None)
            if get_clim is None:
                continue
            vmin, vmax = get_clim()
            if vmin is None or vmax is None:
                continue
            try:
                return float(vmin), float(vmax)
            except (TypeError, ValueError):
                continue
    return None


def _format_placeholder_number(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:g}"


def _norm_field_placeholder(operation: FigureOperationState, attr: str) -> str:
    if getattr(operation, attr) is not None:
        return ""
    if (
        attr == "vcenter"
        and _effective_norm_name(operation.norm_name) in _ZERO_VCENTER_NORMS
    ):
        return "0"
    return ""


def _norm_gamma_value(operation: FigureOperationState) -> float:
    value = operation.norm_gamma
    if value is None:
        value = operation.gamma
    if value is None:
        return 1.0
    return value


def _norm_clip_text(value: bool | None) -> str:
    if value is None:
        return "default"
    return str(value)


def _norm_clip_from_text(text: str) -> bool | None:
    if text == "True":
        return True
    if text == "False":
        return False
    return None


def _update_current_norm_name(tool: FigureComposerTool, name: str) -> None:
    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {"norm_name": name}
        if operation.norm_gamma is None and operation.gamma is not None:
            updates["norm_gamma"] = operation.gamma
            updates["gamma"] = None
        return operation.model_copy(update=updates)

    tool._update_operations(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_norm_gamma(tool: FigureComposerTool, value: float) -> None:
    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        return operation.model_copy(
            update={
                "norm_name": operation.norm_name or _POWER_NORM_NAME,
                "norm_gamma": value,
                "gamma": None,
            }
        )

    tool._update_operations(update_operation, defer_render=True)


def _update_current_norm_kwargs(tool: FigureComposerTool, text: str) -> None:
    updates = _norm_updates_from_kwargs(_dict_from_text(text))

    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        return operation.model_copy(update=updates)

    tool._update_operations(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_slice_kwargs(tool: FigureComposerTool, text: str) -> None:
    slice_kwargs = _dict_from_text(text, allow_slice=True)

    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates = _selection_updates_from_kwargs(
            tool,
            operation,
            slice_kwargs,
            _effective_extra_kwargs(tool, operation),
        )
        return operation.model_copy(update=updates)

    tool._update_operations(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_extra_kwargs(tool: FigureComposerTool, text: str) -> None:
    extra_kwargs = _dict_from_text(text, allow_slice=True)

    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates = _selection_updates_from_kwargs(
            tool,
            operation,
            _effective_slice_kwargs(tool, operation),
            extra_kwargs,
        )
        return operation.model_copy(update=updates)

    tool._update_operations(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_cmap(
    tool: FigureComposerTool,
    *,
    base: str | None = None,
    reverse: bool | None = None,
) -> None:
    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        operation_base, operation_reverse = _cmap_base_and_reverse(operation.cmap)
        next_base = operation_base if base is None else base
        next_reverse = operation_reverse if reverse is None else reverse
        return operation.model_copy(
            update={"cmap": _cmap_with_reverse(next_base, next_reverse)}
        )

    tool._update_operations(update_operation, defer_render=True)
    tool._update_step_action_buttons()
    tool._refresh_step_section_button_texts()


def _update_current_panel_styles_enabled(
    tool: FigureComposerTool, enabled: bool
) -> None:
    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {"panel_styles_enabled": enabled}
        if not enabled:
            updates["panel_styles"] = ()
        return operation.model_copy(update=updates)

    tool._update_operations(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_panel_styles(
    tool: FigureComposerTool,
    styles: tuple[FigurePlotSlicesPanelStyleState, ...],
) -> None:
    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        return operation.model_copy(
            update={
                "panel_styles_enabled": bool(styles),
                "panel_styles": tuple(styles),
            }
        )

    tool._update_operations(update_operation)


def _plot_slices_shape(
    tool: FigureComposerTool, operation: FigureOperationState
) -> _PlotSlicesShape:
    operation = _normalized_selection_operation(tool, operation)
    maps = _operation_maps(tool, operation)
    if not maps:
        return _PlotSlicesShape(
            source_text="unavailable",
            selection_text="Select at least one valid source.",
            panel_text="unavailable",
            axes_text="",
            plot_dims=(),
            plot_ndim=None,
            panel_count=0,
            valid=False,
        )

    dims = tuple(str(dim) for dim in maps[0].dims)
    if any(tuple(str(dim) for dim in data.dims) != dims for data in maps[1:]):
        return _PlotSlicesShape(
            source_text="mixed",
            selection_text="All inputs must have matching dimensions.",
            panel_text="unavailable",
            axes_text="",
            plot_dims=(),
            plot_ndim=None,
            panel_count=0,
            valid=False,
        )

    selected_dims: set[str] = set()
    selection_error = ""
    slice_count = 1
    slice_values = _effective_slice_values(tool, operation)
    if _use_all_coordinate_slice_values(operation) and not operation.slice_dim:
        selection_error = _all_coordinate_slice_values_error(tool, operation, dims)
    if operation.slice_dim and not selection_error:
        all_values_error = _all_coordinate_slice_values_error(tool, operation, dims)
        if all_values_error:
            selection_error = all_values_error
        elif operation.slice_dim not in dims and slice_values:
            selection_error = f"{operation.slice_dim!r} is not an input dimension."
        elif slice_values:
            selected_dims.add(operation.slice_dim)
            slice_count = len(slice_values)

    for key, value in operation.slice_kwargs.items():
        if key.endswith("_width") or key not in dims:
            continue
        count = _selection_value_count(value)
        if isinstance(value, slice):
            continue
        if count is None:
            selected_dims.add(key)
        else:
            selected_dims.add(key)
            slice_count = max(slice_count, count)

    plot_dims = tuple(dim for dim in dims if dim not in selected_dims)
    panel_count = len(maps) * slice_count
    if selection_error:
        panel_text = "unavailable"
        valid = False
    elif len(plot_dims) == 1:
        panel_text = f"{plot_dims[0]} (1D line)"
        valid = True
    elif len(plot_dims) == 2:
        panel_text = f"{_format_dim_names(plot_dims)} (2D image)"
        valid = True
    else:
        panel_text = f"{_format_dim_names(plot_dims)} ({len(plot_dims)}D, unsupported)"
        valid = False

    return _PlotSlicesShape(
        source_text=_format_dim_names(dims),
        selection_text=(
            selection_error
            or (
                "Choose selections until the plotted dimensions are 1D or 2D."
                if not valid
                else ""
            )
        ),
        panel_text=panel_text,
        axes_text="",
        plot_dims=plot_dims,
        plot_ndim=len(plot_dims),
        panel_count=panel_count,
        valid=valid,
    )


def _format_dim_names(dims: Sequence[str]) -> str:
    return ", ".join(dims) if dims else "none"


def _plot_slices_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    operation = _normalized_selection_operation(tool, operation)
    kwargs: dict[str, typing.Any] = {}
    shape = _plot_slices_shape(tool, operation)
    is_line_plot = shape.plot_ndim == 1
    kwargs.update(dict(operation.slice_kwargs))
    slice_values = _effective_slice_values(tool, operation)
    if operation.slice_dim and slice_values:
        kwargs[operation.slice_dim] = list(slice_values)
        if operation.slice_width is not None:
            kwargs[f"{operation.slice_dim}_width"] = operation.slice_width
    if operation.transpose:
        kwargs["transpose"] = True
    if operation.xlim is not None:
        kwargs["xlim"] = operation.xlim
    if operation.ylim is not None:
        kwargs["ylim"] = operation.ylim
    if not operation.crop:
        kwargs["crop"] = False
    if not is_line_plot and operation.same_limits is not False:
        kwargs["same_limits"] = operation.same_limits
    if operation.axis != "auto":
        kwargs["axis"] = operation.axis
    if operation.show_all_labels:
        kwargs["show_all_labels"] = True
    if not is_line_plot and operation.colorbar != "none":
        kwargs["colorbar"] = operation.colorbar
    if not is_line_plot and not operation.hide_colorbar_ticks:
        kwargs["hide_colorbar_ticks"] = False
    if not operation.annotate:
        kwargs["annotate"] = False
    if not is_line_plot:
        cmap = _panel_cmap_argument(tool, operation)
        if cmap is not None:
            kwargs["cmap"] = cmap
    if is_line_plot:
        line_kw = _panel_line_kw_argument(tool, operation)
        if line_kw:
            kwargs["line_kw"] = line_kw
        if isinstance(line_kw, list) and operation.order != operation.cmap_order:
            kwargs["line_order"] = operation.order
    if not is_line_plot:
        panel_norm = _panel_norm_argument(tool, operation)
        if panel_norm is not None:
            kwargs["norm"] = panel_norm
        elif _use_powernorm_plot_kwargs(operation):
            gamma = operation.norm_gamma
            if gamma is None:
                gamma = operation.gamma
            if gamma is not None:
                kwargs["gamma"] = gamma
            for name in ("gamma", "vmin", "vmax"):
                if name == "gamma":
                    continue
                value = getattr(operation, name)
                if value is not None:
                    kwargs[name] = value
        else:
            kwargs["norm"] = _norm_object(operation)
    if operation.order != "C":
        kwargs["order"] = operation.order
    if operation.cmap_order != "C":
        kwargs["cmap_order"] = operation.cmap_order
    if operation.norm_order is not None:
        kwargs["norm_order"] = operation.norm_order
    if is_line_plot and operation.gradient:
        kwargs["gradient"] = True
    if is_line_plot and operation.gradient_kw:
        kwargs["gradient_kw"] = dict(operation.gradient_kw)
    if operation.subplot_kw:
        kwargs["subplot_kw"] = dict(operation.subplot_kw)
    if operation.annotate_kw:
        kwargs["annotate_kw"] = dict(operation.annotate_kw)
    if not is_line_plot and operation.colorbar_kw:
        kwargs["colorbar_kw"] = dict(operation.colorbar_kw)
    kwargs.update(dict(_effective_extra_kwargs(tool, operation)))
    return kwargs


def _plot_slices_transformed_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    kwargs = _plot_slices_kwargs(tool, operation)
    for key in operation.slice_kwargs:
        kwargs.pop(key, None)
    if operation.slice_dim:
        kwargs.pop(f"{operation.slice_dim}_width", None)
        slice_values = _effective_slice_values(tool, operation)
        if slice_values:
            kwargs[operation.slice_dim] = list(slice_values)
        else:
            kwargs.pop(operation.slice_dim, None)
    return kwargs


def _render_plot_slices(
    tool: FigureComposerTool, operation: FigureOperationState, axs: typing.Any
) -> None:
    operation = _normalized_selection_operation(tool, operation)
    maps = _operation_maps(tool, operation)
    if not maps:
        return
    kwargs = _plot_slices_kwargs(tool, operation)
    if _plot_slices_uses_transformed_line_maps(tool, operation):
        maps = _plot_slices_transformed_maps(tool, operation, maps)
        kwargs = _plot_slices_transformed_kwargs(tool, operation)
    selection_cache = getattr(tool, "_plot_slices_selection_cache", None)
    if selection_cache is not None:
        kwargs["_selection_cache"] = selection_cache
        kwargs["_selection_cache_key"] = _plot_slices_selection_cache_key(
            operation, maps
        )
    axes = _plot_slices_axes(
        operation,
        maps,
        _axes_from_selection(tool, operation.axes, axs, for_plot_slices=True),
        slice_count=_plot_slices_slice_count(tool, operation),
    )
    axes_tuple = _iter_axes(axes)
    panel_keys = _plot_slices_panel_keys(tool, operation)
    mappable_ids_before = _axis_mappable_ids(axes_tuple)
    eplt.plot_slices(
        maps,
        axes=typing.cast("Iterable[matplotlib.axes.Axes]", axes),
        **kwargs,
    )
    _tag_plot_slices_mappables(
        operation,
        axes_tuple,
        panel_keys,
        mappable_ids_before,
    )


def _axis_mappables(axis: object) -> tuple[object, ...]:
    images = tuple(getattr(axis, "images", ()))
    collections = tuple(getattr(axis, "collections", ()))
    return (*images, *collections)


def _axis_mappable_ids(axes: Sequence[object]) -> dict[object, set[int]]:
    return {axis: {id(mappable) for mappable in _axis_mappables(axis)} for axis in axes}


def _tag_plot_slices_mappables(
    operation: FigureOperationState,
    axes: Sequence[object],
    panel_keys: Sequence[_PlotSlicesPanelKey],
    mappable_ids_before: Mapping[object, set[int]],
) -> None:
    if len(axes) != len(panel_keys):
        return
    for axis, panel_key in zip(axes, panel_keys, strict=True):
        previous_ids = mappable_ids_before.get(axis, set())
        for mappable in _axis_mappables(axis):
            if id(mappable) in previous_ids:
                continue
            setattr(
                mappable,
                _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
                operation.operation_id,
            )
            setattr(
                mappable,
                _PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
                (panel_key.map_index, panel_key.slice_index),
            )


def _plot_slices_axes(
    operation: FigureOperationState,
    maps: Sequence[xr.DataArray],
    axes: object,
    *,
    slice_count: int | None = None,
) -> object:
    if not isinstance(axes, np.ndarray):
        return axes
    slice_count = max(
        slice_count if slice_count is not None else len(operation.slice_values), 1
    )
    if operation.order == "F":
        shape = (slice_count, len(maps))
    else:
        shape = (len(maps), slice_count)
    if axes.size != math.prod(shape):
        return axes
    return axes.reshape(shape)


def _plot_slices_selection_cache_key(
    operation: FigureOperationState, maps: Sequence[xr.DataArray]
) -> tuple[object, ...]:
    source_key = tuple(
        (
            selection.source,
            repr(selection.isel),
            repr(selection.qsel),
            tuple(selection.mean_dims),
        )
        for selection in operation.map_selections
    )
    if not source_key:
        source_key = tuple((source,) for source in operation.sources)
    map_key = tuple(
        (id(data.data), tuple(data.dims), tuple(data.shape)) for data in maps
    )
    return (source_key, map_key)


def _operation_maps(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[xr.DataArray]:
    if operation.map_selections:
        return [
            selected
            for selection in operation.map_selections
            if (selected := _selected_data(tool._source_data, selection)) is not None
        ]

    return [
        _public_source_data(tool._source_data[name])
        for name in operation.sources
        if name in tool._source_data
    ]


def _plot_slices_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    operation = _normalized_selection_operation(tool, operation)
    if operation.map_selections:
        maps_code = (
            _selection_code(operation.map_selections[0])
            if len(operation.map_selections) == 1
            else "selected_maps"
        )
    else:
        sources = [_valid_source_variable(source) for source in operation.sources]
        if not sources:
            return None
        maps_code = sources[0] if len(sources) == 1 else f"[{', '.join(sources)}]"

    kwargs = _plot_slices_code_kwargs(tool, operation)
    kwargs["axes"] = _RawCode(_axes_code(tool, operation.axes, for_plot_slices=True))
    kwargs_text = _code_kwargs(kwargs)
    return f"eplt.plot_slices({maps_code}, {kwargs_text})"


def _plot_slices_profile_source_codes(
    operation: FigureOperationState,
) -> tuple[str, ...]:
    if operation.map_selections:
        return tuple(
            _selection_code(selection) for selection in operation.map_selections
        )
    return tuple(_valid_source_variable(source) for source in operation.sources)


def _plot_slices_profile_code(
    source_code: str,
    operation: FigureOperationState,
    key: _PlotSlicesPanelKey,
    profile_data: xr.DataArray,
    slice_values: Sequence[float] | None = None,
) -> str:
    kwargs = _plot_slices_panel_qsel_kwargs(operation, key, slice_values)
    if not kwargs:
        return _maybe_squeeze_drop_code(source_code, profile_data)
    code = f"{source_code}.qsel({_code_kwargs(kwargs)})"
    return _maybe_squeeze_drop_code(code, profile_data)


def _plot_slices_transformed_maps_code(
    operation: FigureOperationState,
    keys: tuple[_PlotSlicesPanelKey, ...],
    slice_values: Sequence[float] | None = None,
    slice_values_code: str | None = None,
) -> tuple[list[str], str]:
    if slice_values is None:
        slice_values = operation.slice_values
    if not operation.slice_dim or not slice_values:
        maps_code = "profiles[0]" if len(keys) == 1 else "profiles"
        return [], maps_code

    map_count = max((key.map_index for key in keys), default=-1) + 1
    slice_values = list(slice_values)
    dim_code = erlab.interactive.utils._parse_single_arg(operation.slice_dim)
    if slice_values_code is None:
        coords_code = erlab.interactive.utils._parse_single_arg(
            {operation.slice_dim: slice_values}
        )
    else:
        coords_code = f"{{{dim_code}: {slice_values_code}}}"
    map_lines: list[list[str]] = []
    for map_index in range(map_count):
        profile_indices = [
            index for index, key in enumerate(keys) if key.map_index == map_index
        ]
        if len(profile_indices) != len(slice_values):
            continue
        profile_items = ", ".join(f"profiles[{index}]" for index in profile_indices)
        map_lines.append(
            [
                "xr.concat(",
                f"    [{profile_items}],",
                f"    dim={dim_code},",
                '    coords="different",',
                '    compat="equals",',
                f").assign_coords({coords_code})",
            ]
        )
    if len(map_lines) == 1:
        return [], "\n".join(map_lines[0])
    lines = ["plot_maps = ["]
    for map_code in map_lines:
        lines.append("    " + map_code[0])
        lines.extend("    " + line for line in map_code[1:-1])
        lines.append("    " + map_code[-1] + ",")
    lines.append("]")
    return lines, "plot_maps"


def _plot_slices_call_lines(maps_code: str, kwargs_text: str) -> list[str]:
    if "\n" not in maps_code:
        return [f"eplt.plot_slices({maps_code}, {kwargs_text})"]
    lines = ["eplt.plot_slices("]
    maps_lines = maps_code.splitlines()
    lines.extend(f"    {line}" for line in maps_lines[:-1])
    lines.append(f"    {maps_lines[-1]},")
    lines.append(f"    {kwargs_text},")
    lines.append(")")
    return lines


def _plot_slices_transformed_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    operation = _normalized_selection_operation(tool, operation)
    maps = _operation_maps(tool, operation)
    profiles, _ = _plot_slices_line_profiles(tool, operation, maps)
    source_codes = _plot_slices_profile_source_codes(operation)
    if not source_codes:
        return []
    keys = tuple(
        key
        for key in _plot_slices_panel_keys(tool, operation)
        if key.map_index < len(source_codes) and key.map_index < len(maps)
    )
    if not keys:
        return []

    slice_values = _effective_slice_values(tool, operation)
    lines = ["profiles = ["]
    lines.extend(
        "    "
        + _plot_slices_profile_code(
            source_codes[key.map_index],
            operation,
            key,
            _plot_slices_panel_profile_data(
                maps[key.map_index], operation, key, slice_values
            ),
            slice_values,
        )
        + ","
        for key in keys
    )
    lines.append("]")
    lines.extend(profile_transform_code_lines(operation, profiles=profiles))
    map_lines, maps_code = _plot_slices_transformed_maps_code(
        operation,
        keys,
        slice_values,
        _all_coordinate_slice_values_code(operation),
    )
    lines.extend(map_lines)
    lines.extend(_plot_slices_line_color_code_lines(tool, operation))

    kwargs = _plot_slices_transformed_code_kwargs(tool, operation)
    kwargs["axes"] = _RawCode(_axes_code(tool, operation.axes, for_plot_slices=True))
    kwargs_text = _code_kwargs(kwargs)
    lines.extend(_plot_slices_call_lines(maps_code, kwargs_text))
    return lines


def _plot_slices_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    if _plot_slices_uses_transformed_line_maps(tool, operation):
        return _plot_slices_transformed_code_lines(tool, operation)
    operation = _normalized_selection_operation(tool, operation)
    code = _plot_slices_code(tool, operation)
    if code is None:
        return []
    color_lines = _plot_slices_line_color_code_lines(tool, operation)
    if not operation.map_selections:
        return [*color_lines, code]
    if len(operation.map_selections) == 1:
        return [*color_lines, code]
    lines = ["selected_maps = ["]
    lines.extend(
        f"    {_selection_code(selection)}," for selection in operation.map_selections
    )
    lines.append("]")
    lines.extend(color_lines)
    lines.append(code)
    return lines


def _plot_slices_code_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    kwargs = _plot_slices_kwargs(tool, operation)
    is_line_plot = _plot_slices_shape(tool, operation).plot_ndim == 1
    slice_values_code = _all_coordinate_slice_values_code(operation)
    if operation.slice_dim and slice_values_code is not None:
        kwargs[operation.slice_dim] = _RawCode(slice_values_code)
    panel_norm_code = None if is_line_plot else _panel_norm_code(tool, operation)
    if panel_norm_code is not None:
        kwargs["norm"] = _RawCode(panel_norm_code)
    elif not is_line_plot and not _use_powernorm_plot_kwargs(operation):
        kwargs["norm"] = _RawCode(_norm_code(operation))
    if is_line_plot:
        line_kw_code = _plot_slices_line_kw_code(tool, operation)
        if line_kw_code is not None:
            kwargs["line_kw"] = _RawCode(line_kw_code)
    return kwargs


def _plot_slices_transformed_code_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    kwargs = _plot_slices_transformed_kwargs(tool, operation)
    line_kw_code = _plot_slices_line_kw_code(tool, operation)
    if line_kw_code is not None:
        kwargs["line_kw"] = _RawCode(line_kw_code)
    return kwargs


_SECTION_TOOLTIPS = {
    "selection": "Choose dimension, values, and extraction options.",
    "view": "Set orientation, axis limits, labels, and annotation behavior.",
    "colors": "Set image color scaling or line styling for this plot_slices step.",
    "transform": "Normalize, scale, and offset 1D line slices before plotting.",
    "advanced": "Pass advanced keyword arguments to plot_slices.",
}


def _create_plot_slices_operation(tool: FigureComposerTool) -> FigureOperationState:
    source_names = tool._source_names()
    first_source = source_names[0] if source_names else tool._recipe.primary_source
    return FigureOperationState.plot_slices(
        label="plot_slices",
        sources=(first_source,),
        axes=tool._selected_axes_state(),
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    operation = _normalized_selection_operation(tool, operation)
    prefix = "Needs axes: " if _has_invalid_target(tool, operation) else ""
    source_text = ", ".join(tool._source_display_names(operation.sources))
    if not source_text:
        source_text = "missing source"
    shape = _plot_slices_shape(tool, operation)
    plot_kind = "Line slices" if shape.plot_ndim == 1 else "Image slices"
    slice_values = _effective_slice_values(tool, operation)
    if operation.slice_dim and slice_values:
        selection_text = f"{operation.slice_dim} = {len(slice_values)} values"
    else:
        selection_text = "current selection"
    return f"{prefix}{plot_kind}: {source_text}, {selection_text}"


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return (
        "Runs erlab.plotting.plot_slices.\n"
        f"Targets: {tool._axes_target_text(operation.axes)}"
    )


def _has_invalid_target(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    return tool._axes_selection_has_invalid_target(operation.axes)


def _source_names(operation: FigureOperationState) -> tuple[str, ...]:
    names: list[str] = []
    for source_name in operation.sources:
        if source_name not in names:
            names.append(source_name)
    for selection in operation.map_selections:
        if selection.source not in names:
            names.append(selection.source)
    return tuple(names)


def _plot_source_check_state(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    source_name: str,
) -> QtCore.Qt.CheckState:
    editable = tool._editable_operations()
    if len(editable) <= 1:
        return (
            QtCore.Qt.CheckState.Checked
            if source_name in operation.sources
            else QtCore.Qt.CheckState.Unchecked
        )
    selected_count = sum(source_name in target.sources for _index, target in editable)
    if selected_count == 0:
        return QtCore.Qt.CheckState.Unchecked
    if selected_count == len(editable):
        return QtCore.Qt.CheckState.Checked
    return QtCore.Qt.CheckState.PartiallyChecked


def _plot_source_check_changed(
    tool: FigureComposerTool,
    source_name: str,
    check: QtWidgets.QCheckBox,
    row_order: tuple[str, ...],
) -> None:
    if tool._updating_controls:
        return
    state = check.checkState()
    if state == QtCore.Qt.CheckState.PartiallyChecked:
        return
    checked = state == QtCore.Qt.CheckState.Checked

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        if checked:
            if source_name in target.sources:
                return target
            source_set = {*target.sources, source_name}
            ordered_sources = tuple(
                source for source in row_order if source in source_set
            )
            missing_sources = tuple(
                source for source in target.sources if source not in row_order
            )
            return target.model_copy(
                update={"sources": (*ordered_sources, *missing_sources)}
            )
        next_sources = tuple(
            source for source in target.sources if source != source_name
        )
        return target.model_copy(update={"sources": next_sources})

    tool._update_operations(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _plot_source_move(
    tool: FigureComposerTool,
    source_name: str,
    offset: typing.Literal[-1, 1],
) -> None:
    if tool._updating_controls:
        return
    available_sources = set(tool._source_names())

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        ordered_sources = [
            source for source in target.sources if source in available_sources
        ]
        if source_name not in ordered_sources:
            return target
        source_index = ordered_sources.index(source_name)
        target_index = source_index + offset
        if target_index < 0 or target_index >= len(ordered_sources):
            return target
        ordered_sources[source_index], ordered_sources[target_index] = (
            ordered_sources[target_index],
            ordered_sources[source_index],
        )
        missing_sources = tuple(
            source for source in target.sources if source not in available_sources
        )
        return target.model_copy(
            update={"sources": (*ordered_sources, *missing_sources)}
        )

    tool._update_operations(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _plot_source_order_matches(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    editable = tool._editable_operations()
    if len(editable) <= 1:
        return True
    available_sources = set(tool._source_names())
    expected = tuple(
        source for source in operation.sources if source in available_sources
    )
    return all(
        tuple(source for source in target.sources if source in available_sources)
        == expected
        for _index, target in editable
    )


def _plot_source_row_names(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    source_names = tool._source_names()
    selected_sources = tuple(
        source for source in operation.sources if source in source_names
    )
    unselected_sources = tuple(
        source for source in source_names if source not in selected_sources
    )
    return selected_sources + unselected_sources


class _PlotSourceMoveButton(QtWidgets.QToolButton):
    def __init__(
        self,
        direction: typing.Literal["up", "down"],
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self._direction = direction
        self._refresh_icon()

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        if event is not None and event.type() in {
            QtCore.QEvent.Type.ActivationChange,
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.EnabledChange,
            QtCore.QEvent.Type.PaletteChange,
        }:
            self._refresh_icon()
        super().changeEvent(event)

    def _refresh_icon(self) -> None:
        icon_name = "mdi6.arrow-up" if self._direction == "up" else "mdi6.arrow-down"
        palette = self.palette()
        window = self.window()
        color_group = (
            QtGui.QPalette.ColorGroup.Active
            if window is not None and window.isActiveWindow()
            else QtGui.QPalette.ColorGroup.Inactive
        )
        self.setIcon(
            erlab.interactive.utils.qtawesome.icon(
                icon_name,
                color=palette.color(color_group, QtGui.QPalette.ColorRole.ButtonText),
                color_disabled=palette.color(
                    QtGui.QPalette.ColorGroup.Disabled,
                    QtGui.QPalette.ColorRole.ButtonText,
                ),
            )
        )


def _build_plot_source_row(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    selector: QtWidgets.QWidget,
    source_name: str,
    index: int,
    selected_sources: tuple[str, ...],
    row_order: tuple[str, ...],
    order_controls_enabled: bool,
) -> tuple[QtWidgets.QCheckBox, QtWidgets.QToolButton, QtWidgets.QToolButton]:
    check = QtWidgets.QCheckBox(tool._source_display_name(source_name), selector)
    check.setObjectName(f"figureComposerPlotSlicesSourceCheck_{index}")
    check.setProperty("figure_source_name", source_name)
    check.setToolTip(
        "Include this DataArray in the maps passed to plot_slices.\n"
        + tool._source_tooltip(source_name)
    )
    state = _plot_source_check_state(tool, operation, source_name)
    check.setTristate(state == QtCore.Qt.CheckState.PartiallyChecked)
    check.setCheckState(state)
    tool._connect_editor_signal(
        check,
        check.stateChanged,
        lambda _state, source_name=source_name, check=check, row_order=row_order: (
            _plot_source_check_changed(tool, source_name, check, row_order)
        ),
    )

    source_selected = source_name in selected_sources
    selected_index = selected_sources.index(source_name) if source_selected else -1
    up_button = _plot_source_move_button(
        tool,
        selector,
        source_name,
        "up",
        order_controls_enabled and selected_index > 0,
        "Move this input earlier in the maps argument.",
        lambda: _plot_source_move(tool, source_name, -1),
    )
    down_button = _plot_source_move_button(
        tool,
        selector,
        source_name,
        "down",
        order_controls_enabled
        and source_selected
        and selected_index < len(selected_sources) - 1,
        "Move this input later in the maps argument.",
        lambda: _plot_source_move(tool, source_name, 1),
    )
    if source_selected:
        return check, up_button, down_button
    up_button.setVisible(False)
    down_button.setVisible(False)
    return check, up_button, down_button


def _plot_source_move_button(
    tool: FigureComposerTool,
    parent: QtWidgets.QWidget,
    source_name: str,
    direction: typing.Literal["up", "down"],
    enabled: bool,
    tooltip: str,
    clicked: Callable[[], None],
) -> QtWidgets.QToolButton:
    button = _PlotSourceMoveButton(direction, parent)
    button.setObjectName(
        f"figureComposerPlotSlicesSourceMove_{direction}_{source_name}"
    )
    button.setProperty("figure_source_name", source_name)
    button.setProperty("figure_source_move", direction)
    button.setEnabled(enabled)
    button.setToolTip(tooltip)
    tool._connect_editor_signal(
        button,
        button.clicked,
        lambda _checked=False: clicked(),
    )
    return button


def _build_source_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> None:
    selector = QtWidgets.QWidget(tool.step_source_controls)
    selector.setObjectName("figureComposerPlotSlicesSourceSelector")
    layout = QtWidgets.QGridLayout(selector)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(4)
    layout.setVerticalSpacing(2)
    row_names = _plot_source_row_names(tool, operation)
    selected_sources = tuple(
        source for source in operation.sources if source in tool._source_names()
    )
    order_controls_enabled = _plot_source_order_matches(tool, operation)
    if row_names:
        for index, source_name in enumerate(row_names):
            check, up_button, down_button = _build_plot_source_row(
                tool,
                operation,
                selector,
                source_name,
                index,
                selected_sources,
                row_names,
                order_controls_enabled,
            )
            layout.addWidget(check, index, 0)
            layout.addWidget(up_button, index, 1)
            layout.addWidget(down_button, index, 2)
    else:
        label = QtWidgets.QLabel("No source arrays are available.", selector)
        label.setEnabled(False)
        layout.addWidget(label, 0, 0)
    layout.setColumnStretch(0, 1)
    layout.setColumnStretch(1, 0)
    layout.setColumnStretch(2, 0)
    selector.setToolTip("Select one or more DataArrays to pass as maps to plot_slices.")
    tool._add_form_row(
        tool.step_source_controls_layout,
        "Inputs",
        selector,
        selector.toolTip(),
    )


def _editor_sections(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    return tuple(
        StepSection(key, title, page, _SECTION_TOOLTIPS[key])
        for key, title, page in _build_plot_slices_editor(tool, operation)
    )


def _section_summary(
    tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    operation = _normalized_selection_operation(tool, operation)
    match key:
        case "sources":
            return ", ".join(tool._source_display_names(operation.sources)) or "none"
        case "axes":
            return tool._axes_target_text(operation.axes)
        case "selection":
            slice_values = _effective_slice_values(tool, operation)
            if operation.slice_dim and slice_values:
                return f"{operation.slice_dim}, {len(slice_values)}"
            if operation.slice_kwargs:
                return "additional"
            return "none"
        case "view":
            labels = [
                label
                for label, value in (("x", operation.xlim), ("y", operation.ylim))
                if value is not None
            ]
            return ", ".join(labels) if labels else "auto"
        case "colors":
            panel_kind = _plot_slices_batch_panel_kind(tool, operation)
            if panel_kind == _PLOT_SLICES_PANEL_MIXED:
                return "mixed"
            if panel_kind == _PLOT_SLICES_PANEL_LINE:
                if line_colormap_active(operation):
                    coord = effective_line_color_coord(operation, operation.slice_dim)
                    return f"by {coord}" if coord else "by coordinate"
                return line_kw_text(operation, "color", "c") or "line"
            if operation.panel_styles_enabled and operation.panel_styles:
                return "per-panel"
            return operation.cmap or "default"
        case "transform":
            return "set" if line_transform_active(operation) else ""
        case "advanced":
            return "set" if _effective_extra_kwargs(tool, operation) else ""
    return ""


def _required_imports(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    imports = ["import erlab.plotting as eplt"]
    if (
        operation.enabled
        and _plot_slices_uses_transformed_line_maps(tool, operation)
        and operation.slice_dim
        and _effective_slice_values(tool, operation)
    ):
        imports.append("import xarray as xr")
    if (
        operation.enabled
        and _plot_slices_shape(tool, operation).plot_ndim != 1
        and (
            _panel_norm_uses_matplotlib_colors(tool, operation)
            or (
                not _use_powernorm_plot_kwargs(operation)
                and _effective_norm_name(operation.norm_name) in _MATPLOTLIB_NORM_NAMES
            )
        )
    ):
        imports.append("import matplotlib.colors as mcolors")
    if operation.enabled and _plot_slices_line_colormap_active(tool, operation):
        imports.append("import matplotlib.colors as mcolors")
    return tuple(imports)


SPEC = OperationSpec(
    kind=FigureOperationKind.PLOT_SLICES,
    add_actions=(
        AddStepActionSpec(
            action_id=FigureOperationKind.PLOT_SLICES.value,
            text="Slice Plot",
            tooltip="Add an editable erlab.plotting.plot_slices step.",
            create_operation=_create_plot_slices_operation,
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
    render=lambda tool, operation, _figure, axs: _render_plot_slices(
        tool, operation, axs
    ),
    code_lines=_plot_slices_code_lines,
    required_imports=_required_imports,
)
