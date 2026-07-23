"""Plot-slices state semantics, panel modeling, and data preparation."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import xarray as xr

from erlab.interactive._figurecomposer._defaults import (
    _current_options,
    _styled_rcparams_value,
    _tool_figure_options_context,
)
from erlab.interactive._figurecomposer._labels import label_context, labels_from_text
from erlab.interactive._figurecomposer._line_colormap import (
    colors_from_values,
    effective_line_color_cmap,
    effective_line_color_cmap_trim,
    effective_line_color_coord,
    line_colormap_active,
    numeric_context_field_names,
    values_from_contexts,
)
from erlab.interactive._figurecomposer._line_transform import (
    LineTransformPlan,
    line_transform_active,
    transform_profiles_from_plan,
)
from erlab.interactive._figurecomposer._model._sources import (
    _available_source_dims,
    _public_source_data,
)
from erlab.interactive._figurecomposer._model._state import (
    _POWER_NORM_NAME,
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    _PlotSlicesShape,
)
from erlab.interactive._figurecomposer._norms import (
    _matplotlib_cmap_name,
    _norm_object,
    _use_powernorm_plot_kwargs,
)
from erlab.interactive._figurecomposer._text import _selection_value_count

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
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
    context: FigureRecipeContext, operation: FigureOperationState
) -> tuple[str, ...]:
    maps = _operation_maps(context, operation)
    dims: list[str] = []
    for data in maps:
        for dim in data.dims:
            dim_text = str(dim)
            if dim_text not in dims:
                dims.append(dim_text)
    if dims:
        return tuple(dims)
    return tuple(
        _available_source_dims(
            context.source_data, _plot_slices_selection_sources(operation)
        )
    )


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
    context: FigureRecipeContext, operation: FigureOperationState
) -> tuple[float, ...]:
    if not operation.slice_dim:
        return ()
    maps = _operation_maps(context, operation)
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
    context: FigureRecipeContext, operation: FigureOperationState
) -> tuple[float, ...]:
    if _use_all_coordinate_slice_values(operation):
        return _all_coordinate_slice_values(context, operation)
    return operation.slice_values


def _all_coordinate_slice_values_error(
    context: FigureRecipeContext,
    operation: FigureOperationState,
    dims: Sequence[str],
) -> str:
    if not _use_all_coordinate_slice_values(operation):
        return ""
    if not operation.slice_dim:
        return "Choose a dimension before using all coordinate values."
    if operation.slice_dim not in dims:
        return f"{operation.slice_dim!r} is not an input dimension."
    if _effective_slice_values(context, operation):
        return ""
    return f"{operation.slice_dim!r} coordinate values must be numeric and non-empty."


def _all_coordinate_slice_values_summary(
    context: FigureRecipeContext, operation: FigureOperationState
) -> str:
    if not operation.slice_dim:
        return "Choose a dimension."
    maps = _operation_maps(context, operation)
    if not maps:
        return "Select at least one valid source."
    data = maps[0]
    if operation.slice_dim not in data.dims:
        return f"{operation.slice_dim!r} is not an input dimension."
    slice_values = _effective_slice_values(context, operation)
    if not slice_values:
        return (
            f"{operation.slice_dim!r} coordinate values must be numeric and non-empty."
        )
    total_count = int(data.sizes[operation.slice_dim])
    plotted_count = len(slice_values)
    if operation.slice_values_thin == 1 or plotted_count == total_count:
        return f"{operation.slice_dim}: {plotted_count} values"
    return f"{operation.slice_dim}: {total_count} values, {plotted_count} plotted"


def _plot_slices_panel_kind(shape: _PlotSlicesShape) -> str:
    if shape.plot_ndim == 1:
        return _PLOT_SLICES_PANEL_LINE
    return _PLOT_SLICES_PANEL_IMAGE


def _plot_slices_batch_panel_kind(
    context: FigureRecipeContext,
    editable_operations: Sequence[tuple[int, FigureOperationState]],
    operation: FigureOperationState,
) -> str:
    operations = tuple(
        target
        for _index, target in editable_operations
        if target.kind == FigureOperationKind.PLOT_SLICES
    )
    if not operations:
        operations = (operation,)
    kinds = {
        _plot_slices_panel_kind(_plot_slices_shape(context, target))
        for target in operations
    }
    if len(kinds) == 1:
        return kinds.pop()
    return _PLOT_SLICES_PANEL_MIXED


def _plot_slices_slice_count(
    context: FigureRecipeContext, operation: FigureOperationState
) -> int:
    operation = _normalized_selection_operation(context, operation)
    dims = _operation_dim_names(context, operation)
    selected_dims: set[str] = set()
    slice_count = 1
    slice_values = _effective_slice_values(context, operation)
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
    context: FigureRecipeContext,
    source_display_name: Callable[[str], str],
    operation: FigureOperationState,
) -> tuple[_PlotSlicesPanelKey, ...]:
    operation = _normalized_selection_operation(context, operation)
    maps = _operation_maps(context, operation)
    source_names = _plot_slices_selection_sources(operation)
    map_count = len(maps) or max(len(source_names), 1)
    slice_count = _plot_slices_slice_count(context, operation)
    map_labels = tuple(
        source_display_name(source_names[index])
        if index < len(source_names)
        else f"map {index + 1}"
        for index in range(map_count)
    )
    slice_labels = _plot_slices_slice_labels(
        operation,
        slice_count,
        _effective_slice_values(context, operation),
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
    operation: FigureOperationState,
    style: FigurePlotSlicesPanelStyleState,
    *,
    default_cmap: str | None = None,
) -> str:
    if style.cmap is not None:
        return _matplotlib_cmap_name(style.cmap)
    if operation.cmap is not None:
        return _matplotlib_cmap_name(operation.cmap)
    if default_cmap is None:
        default_cmap = _current_options().colors.cmap.name
    return _matplotlib_cmap_name(default_cmap)


def _plot_slices_default_cmap(tool: FigureComposerTool) -> str:
    with _tool_figure_options_context(tool):
        return str(_styled_rcparams_value("image.cmap"))


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
        return (
            _matplotlib_cmap_name(operation.cmap)
            if operation.cmap is not None
            else None
        )
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    if not any(_panel_style_has_cmap_override(style) for style in styles.values()):
        return (
            _matplotlib_cmap_name(operation.cmap)
            if operation.cmap is not None
            else None
        )

    default_cmap = _plot_slices_default_cmap(tool)

    def value_getter(key: _PlotSlicesPanelKey) -> str:
        style = styles.get(
            (key.map_index, key.slice_index),
            FigurePlotSlicesPanelStyleState(
                map_index=key.map_index,
                slice_index=key.slice_index,
            ),
        )
        return _effective_panel_cmap(operation, style, default_cmap=default_cmap)

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
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
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
    color_active = _plot_slices_line_colormap_active(tool._document, operation)
    line_colors = (
        _plot_slices_line_colormap_colors(tool, operation) if color_active else ()
    )
    if (
        not has_labels
        and (not operation.panel_styles_enabled or not operation.panel_styles)
        and not color_active
    ):
        return dict(operation.line_kw)
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
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
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
    styles = _panel_style_map_for_keys(operation, keys)
    return any(_panel_style_has_line_override(style) for style in styles.values())


def _plot_slices_line_label_contexts(
    context: FigureRecipeContext,
    source_display_name: Callable[[str], str],
    operation: FigureOperationState,
) -> tuple[dict[str, typing.Any], ...]:
    keys = _plot_slices_panel_keys(context, source_display_name, operation)
    source_labels = _plot_slices_source_labels(context, source_display_name, operation)
    slice_values = _effective_slice_values(context, operation)
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
    contexts = _plot_slices_line_label_contexts(
        tool._document, tool._source_display_name, operation
    )
    return labels_from_text(
        operation.line_label_text,
        contexts,
        literal_values=operation.line_labels,
        default=None,
        item_name="slice",
    )


def _plot_slices_line_colormap_active(
    context: FigureRecipeContext, operation: FigureOperationState
) -> bool:
    return (
        line_colormap_active(operation)
        and _plot_slices_shape(context, operation).plot_ndim == 1
    )


def _available_plot_slices_line_color_coords(
    context: FigureRecipeContext,
    source_display_name: Callable[[str], str],
    operation: FigureOperationState,
) -> list[str]:
    names = list(
        numeric_context_field_names(
            _plot_slices_line_label_contexts(context, source_display_name, operation)
        )
    )
    for name in (operation.slice_dim, operation.line_color_coord):
        if name and name not in names:
            names.append(name)
    return names


def _plot_slices_line_color_values(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[float, ...]:
    return values_from_contexts(
        _plot_slices_line_label_contexts(
            tool._document, tool._source_display_name, operation
        ),
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
    context: FigureRecipeContext,
    source_display_name: Callable[[str], str],
    operation: FigureOperationState,
) -> tuple[str, ...]:
    source_names = _plot_slices_selection_sources(operation)
    map_count = len(_operation_maps(context, operation)) or max(len(source_names), 1)
    return tuple(
        source_display_name(source_names[index])
        if index < len(source_names)
        else f"map {index + 1}"
        for index in range(map_count)
    )


def _plot_slices_uses_transformed_line_maps(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    return _plot_slices_shape(
        tool._document, operation
    ).plot_ndim == 1 and line_transform_active(operation)


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


@dataclasses.dataclass(frozen=True)
class _PlotSlicesTransformPlan:
    """Semantic inputs used to select and transform line slice maps."""

    sources: tuple[str, ...]
    slice_dim: str | None
    slice_values: tuple[float, ...]
    slice_width: float | None
    slice_kwargs: dict[str, typing.Any]
    slice_count: int
    order: typing.Literal["C", "F"]
    transform: LineTransformPlan

    @classmethod
    def from_operation(
        cls,
        context: FigureRecipeContext,
        operation: FigureOperationState,
    ) -> _PlotSlicesTransformPlan:
        operation = _normalized_selection_operation(context, operation)
        return cls(
            sources=operation.sources,
            slice_dim=operation.slice_dim,
            slice_values=_effective_slice_values(context, operation),
            slice_width=operation.slice_width,
            slice_kwargs=dict(operation.slice_kwargs),
            slice_count=_plot_slices_slice_count(context, operation),
            order=typing.cast("typing.Literal['C', 'F']", operation.order),
            transform=LineTransformPlan.from_operation(operation),
        )


def _plot_slices_plan_panel_keys(
    plan: _PlotSlicesTransformPlan,
    map_count: int,
) -> tuple[_PlotSlicesPanelKey, ...]:
    if plan.order == "F":
        indices = (
            (map_index, slice_index)
            for slice_index in range(plan.slice_count)
            for map_index in range(map_count)
        )
    else:
        indices = (
            (map_index, slice_index)
            for map_index in range(map_count)
            for slice_index in range(plan.slice_count)
        )
    return tuple(
        _PlotSlicesPanelKey(map_index, slice_index, "")
        for map_index, slice_index in indices
    )


def _plot_slices_panel_profile_data_from_plan(
    data: xr.DataArray,
    plan: _PlotSlicesTransformPlan,
    key: _PlotSlicesPanelKey,
) -> xr.DataArray:
    kwargs = dict(plan.slice_kwargs)
    if plan.slice_dim and plan.slice_values:
        kwargs[plan.slice_dim] = plan.slice_values[key.slice_index]
        if plan.slice_width is not None:
            kwargs[f"{plan.slice_dim}_width"] = plan.slice_width
    if kwargs:
        data = data.qsel(**kwargs)
    return data.squeeze(drop=True)


def _plot_slices_transformed_maps_from_plan(
    plan: _PlotSlicesTransformPlan,
    maps: Sequence[xr.DataArray],
) -> list[xr.DataArray]:
    profiles: list[xr.DataArray] = []
    profile_keys: list[_PlotSlicesPanelKey] = []
    for key in _plot_slices_plan_panel_keys(plan, len(maps)):
        profile = _plot_slices_panel_profile_data_from_plan(
            maps[key.map_index],
            plan,
            key,
        )
        if profile.ndim != 1:
            continue
        profiles.append(profile)
        profile_keys.append(key)

    transformed = transform_profiles_from_plan(plan.transform, profiles)
    if not plan.slice_dim or not plan.slice_values:
        return transformed

    map_profiles: list[list[tuple[int, xr.DataArray]]] = [[] for _map in maps]
    for key, profile in zip(profile_keys, transformed, strict=True):
        map_profiles[key.map_index].append((key.slice_index, profile))

    transformed_maps: list[xr.DataArray] = []
    for profiles_for_map in map_profiles:
        if len(profiles_for_map) != len(plan.slice_values):
            continue
        ordered_profiles = [
            profile
            for _index, profile in sorted(profiles_for_map, key=lambda item: item[0])
        ]
        transformed_maps.append(
            xr.concat(
                ordered_profiles,
                dim=plan.slice_dim,
                coords="different",
                compat="equals",
            ).assign_coords({plan.slice_dim: list(plan.slice_values)})
        )
    return transformed_maps


def _plot_slices_line_profiles(
    context: FigureRecipeContext,
    source_display_name: Callable[[str], str],
    operation: FigureOperationState,
    maps: Sequence[xr.DataArray] | None = None,
) -> tuple[list[xr.DataArray], tuple[_PlotSlicesPanelKey, ...]]:
    operation = _normalized_selection_operation(context, operation)
    if maps is None:
        maps = _operation_maps(context, operation)
    keys = _plot_slices_panel_keys(context, source_display_name, operation)
    slice_values = _effective_slice_values(context, operation)
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
    plan = _PlotSlicesTransformPlan.from_operation(tool._document, operation)
    return _plot_slices_transformed_maps_from_plan(plan, maps)


def _is_slice_kwarg_key(key: typing.Any, dims: Iterable[str]) -> bool:
    if not isinstance(key, str):
        return False
    dim_names = set(dims)
    if key in dim_names and key not in _PLOT_SLICES_EXPLICIT_KWARGS:
        return True
    return key.endswith("_width") and key[: -len("_width")] in dim_names


def _split_slice_kwargs(
    context: FigureRecipeContext,
    operation: FigureOperationState,
    kwargs: dict[typing.Any, typing.Any],
) -> tuple[dict[str, typing.Any], dict[typing.Any, typing.Any]]:
    dims = _operation_dim_names(context, operation)
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
    context: FigureRecipeContext,
    operation: FigureOperationState,
    slice_kwargs: dict[typing.Any, typing.Any],
    extra_kwargs: dict[typing.Any, typing.Any],
) -> dict[str, typing.Any]:
    parsed_slice_kwargs, slice_extra_kwargs = _split_slice_kwargs(
        context, operation, slice_kwargs
    )
    extra_slice_kwargs, parsed_extra_kwargs = _split_slice_kwargs(
        context, operation, extra_kwargs
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
        dims = set(_operation_dim_names(context, operation))
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
    context: FigureRecipeContext, operation: FigureOperationState
) -> dict[str, typing.Any]:
    slice_kwargs, _extra_kwargs = _split_slice_kwargs(
        context, operation, operation.extra_kwargs
    )
    return {**slice_kwargs, **operation.slice_kwargs}


def _effective_extra_kwargs(
    context: FigureRecipeContext, operation: FigureOperationState
) -> dict[typing.Any, typing.Any]:
    _slice_kwargs, extra_kwargs = _split_slice_kwargs(
        context, operation, operation.extra_kwargs
    )
    return extra_kwargs


def _normalized_selection_operation(
    context: FigureRecipeContext, operation: FigureOperationState
) -> FigureOperationState:
    updates = _selection_updates_from_kwargs(
        context,
        operation,
        _effective_slice_kwargs(context, operation),
        _effective_extra_kwargs(context, operation),
    )
    return operation.model_copy(update=updates)


def _plot_slices_selection_sources(
    operation: FigureOperationState,
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(operation.sources))


def _plot_slices_operation_with_sources(
    operation: FigureOperationState,
    sources: tuple[str, ...],
) -> FigureOperationState:
    return operation.model_copy(update={"sources": sources, "map_selections": ()})


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


def _plot_slices_shape(
    context: FigureRecipeContext, operation: FigureOperationState
) -> _PlotSlicesShape:
    operation = _normalized_selection_operation(context, operation)
    maps = _operation_maps(context, operation)
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
    slice_values = _effective_slice_values(context, operation)
    if _use_all_coordinate_slice_values(operation) and not operation.slice_dim:
        selection_error = _all_coordinate_slice_values_error(context, operation, dims)
    if operation.slice_dim and not selection_error:
        all_values_error = _all_coordinate_slice_values_error(context, operation, dims)
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
    operation = _normalized_selection_operation(tool._document, operation)
    kwargs: dict[str, typing.Any] = {}
    shape = _plot_slices_shape(tool._document, operation)
    is_line_plot = shape.plot_ndim == 1
    kwargs.update(dict(operation.slice_kwargs))
    slice_values = _effective_slice_values(tool._document, operation)
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
    kwargs.update(dict(_effective_extra_kwargs(tool._document, operation)))
    return kwargs


def _plot_slices_transformed_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    kwargs = _plot_slices_kwargs(tool, operation)
    for key in operation.slice_kwargs:
        kwargs.pop(key, None)
    if operation.slice_dim:
        kwargs.pop(f"{operation.slice_dim}_width", None)
        slice_values = _effective_slice_values(tool._document, operation)
        if slice_values:
            kwargs[operation.slice_dim] = list(slice_values)
        else:
            kwargs.pop(operation.slice_dim, None)
    return kwargs


def _operation_maps(
    context: FigureRecipeContext, operation: FigureOperationState
) -> list[xr.DataArray]:
    return [
        _public_source_data(context.source_data[name])
        for name in operation.sources
        if name in context.source_data
    ]
