"""Generate executable Python for plot-slices operations."""

from __future__ import annotations

import typing

import erlab
from erlab.interactive._figurecomposer._code import _axes_code, _maybe_squeeze_drop_code
from erlab.interactive._figurecomposer._labels import (
    label_coord_placeholder_name,
    label_field_names,
    label_fstring_code,
    label_text_uses_placeholders,
    string_literal_expression,
)
from erlab.interactive._figurecomposer._line_colormap import (
    colormap_code_lines,
    effective_line_color_cmap,
    effective_line_color_cmap_trim,
    effective_line_color_coord,
)
from erlab.interactive._figurecomposer._line_transform import (
    profile_transform_code_lines,
)
from erlab.interactive._figurecomposer._model._sources import _valid_source_variable
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
)
from erlab.interactive._figurecomposer._norms import (
    _MATPLOTLIB_NORM_NAMES,
    _effective_norm_name,
    _norm_code,
    _use_powernorm_plot_kwargs,
)
from erlab.interactive._figurecomposer._operations._plot_slices._model import (
    _effective_slice_values,
    _nested_panel_values,
    _normalized_selection_operation,
    _operation_maps,
    _operation_with_panel_norm_style,
    _panel_style_has_line_override,
    _panel_style_has_norm_override,
    _panel_style_map_for_keys,
    _plot_slices_kwargs,
    _plot_slices_line_color_values,
    _plot_slices_line_colormap_active,
    _plot_slices_line_label_contexts,
    _plot_slices_line_labels,
    _plot_slices_line_profiles,
    _plot_slices_panel_keys,
    _plot_slices_panel_profile_data,
    _plot_slices_panel_qsel_kwargs,
    _plot_slices_shape,
    _plot_slices_source_labels,
    _plot_slices_transformed_kwargs,
    _plot_slices_uses_transformed_line_maps,
    _PlotSlicesPanelKey,
    _style_sequence_shape,
    _use_all_coordinate_slice_values,
)
from erlab.interactive._figurecomposer._text import _code_kwargs, _RawCode

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import xarray as xr

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _first_plot_slices_source_code(operation: FigureOperationState) -> str | None:
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


def _plot_slices_label_line_kw_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    contexts = _plot_slices_line_label_contexts(
        tool._document, tool._source_display_name, operation
    )
    if not label_text_uses_placeholders(operation.line_label_text, contexts):
        return None
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
    if not keys:
        return None
    fields = label_field_names(operation.line_label_text)
    source_labels = _plot_slices_source_labels(
        tool._document, tool._source_display_name, operation
    )
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
    if not _plot_slices_line_colormap_active(tool._document, operation):
        return _plot_slices_label_line_kw_code(tool, operation)
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
    if not keys:
        return None
    contexts = _plot_slices_line_label_contexts(
        tool._document, tool._source_display_name, operation
    )
    use_placeholder_labels = label_text_uses_placeholders(
        operation.line_label_text, contexts
    )
    labels = _plot_slices_line_labels(tool, operation)
    fields = label_field_names(operation.line_label_text)
    source_labels = _plot_slices_source_labels(
        tool._document, tool._source_display_name, operation
    )
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
        slice_values = _effective_slice_values(tool._document, operation)
        if slice_values:
            return repr(list(slice_values))
    return "[None]"


def _plot_slices_line_color_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    if not _plot_slices_line_colormap_active(tool._document, operation):
        return []
    coord = effective_line_color_coord(operation, operation.slice_dim)
    if coord is None:
        raise ValueError("Choose a coordinate to color slices")
    if coord != operation.slice_dim:
        raise ValueError(f"Cannot color slices by {coord!r}")
    _plot_slices_line_color_values(tool, operation)
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
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
        label_code = _plot_slices_label_fstring_code(
            operation,
            fields,
            source_expr=string_literal_expression(source_labels[key.map_index])
            if key.map_index < len(source_labels)
            else "None",
            value_expr=_plot_slices_indexed_slice_value_code(
                slice_values_code, key.slice_index
            ),
            index_expr=str(panel_index[(key.map_index, key.slice_index)]),
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


def _panel_norm_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    if not operation.panel_styles_enabled or not operation.panel_styles:
        return None
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
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
    keys = _plot_slices_panel_keys(tool._document, tool._source_display_name, operation)
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


def _plot_slices_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    operation = _normalized_selection_operation(tool._document, operation)
    sources = [_valid_source_variable(source) for source in operation.sources]
    if not sources:
        return None
    maps_code = sources[0] if len(sources) == 1 else f"[{', '.join(sources)}]"

    kwargs = _plot_slices_code_kwargs(tool, operation)
    kwargs["axes"] = _RawCode(
        _axes_code(tool._document, operation.axes, for_plot_slices=True)
    )
    kwargs_text = _code_kwargs(kwargs)
    return f"eplt.plot_slices({maps_code}, {kwargs_text})"


def _plot_slices_profile_source_codes(
    operation: FigureOperationState,
) -> tuple[str, ...]:
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
    operation = _normalized_selection_operation(tool._document, operation)
    maps = _operation_maps(tool._document, operation)
    profiles, _ = _plot_slices_line_profiles(
        tool._document, tool._source_display_name, operation, maps
    )
    source_codes = _plot_slices_profile_source_codes(operation)
    if not source_codes:
        return []
    keys = tuple(
        key
        for key in _plot_slices_panel_keys(
            tool._document, tool._source_display_name, operation
        )
        if key.map_index < len(source_codes) and key.map_index < len(maps)
    )
    if not keys:
        return []

    slice_values = _effective_slice_values(tool._document, operation)
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
    kwargs["axes"] = _RawCode(
        _axes_code(tool._document, operation.axes, for_plot_slices=True)
    )
    kwargs_text = _code_kwargs(kwargs)
    lines.extend(_plot_slices_call_lines(maps_code, kwargs_text))
    return lines


def _plot_slices_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    if _plot_slices_uses_transformed_line_maps(tool, operation):
        return _plot_slices_transformed_code_lines(tool, operation)
    operation = _normalized_selection_operation(tool._document, operation)
    code = _plot_slices_code(tool, operation)
    if code is None:
        return []
    color_lines = _plot_slices_line_color_code_lines(tool, operation)
    return [*color_lines, code]


def _plot_slices_code_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    kwargs = _plot_slices_kwargs(tool, operation)
    is_line_plot = _plot_slices_shape(tool._document, operation).plot_ndim == 1
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
