"""Plot Slices operation editor, renderer, and code generation."""

from __future__ import annotations

import math
import typing

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.plotting as eplt
from erlab.interactive._figurecomposer import _rendering
from erlab.interactive._figurecomposer._code import _axes_code, _selection_code
from erlab.interactive._figurecomposer._defaults import (
    _figure_draw_context,
    _figure_style_context,
)
from erlab.interactive._figurecomposer._line_style import (
    LINE_MARKER_OPTIONS,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    extra_line_kw,
    line_kw_float,
    line_kw_text,
    optional_positive_spinbox,
    optional_positive_spinbox_value,
    update_current_extra_line_kw,
    update_current_line_kw,
)
from erlab.interactive._figurecomposer._norms import (
    _MATPLOTLIB_NORM_NAMES,
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
from erlab.interactive._figurecomposer._sources import (
    _available_source_dims,
    _selected_data,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._state import (
    _POWER_NORM_NAME,
    FigureOperationKind,
    FigureOperationState,
    _PlotSlicesShape,
)
from erlab.interactive._figurecomposer._text import (
    _code_kwargs,
    _dict_from_text,
    _float_tuple_from_text,
    _format_dict,
    _format_dim_sizes,
    _format_pair,
    _format_plot_limit,
    _format_tuple,
    _plot_limit_from_text,
    _RawCode,
    _selection_value_count,
)
from erlab.interactive._figurecomposer._widgets import _ColorLineEditWidget

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import matplotlib.axes
    import xarray as xr

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


def _build_plot_slices_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[tuple[str, str, QtWidgets.QWidget]]:
    operation = _normalized_selection_operation(tool, operation)
    shape = _plot_slices_shape(tool, operation)
    panel_kind = _plot_slices_batch_panel_kind(tool, operation)
    is_line_plot = panel_kind == _PLOT_SLICES_PANEL_LINE
    is_image_plot = panel_kind == _PLOT_SLICES_PANEL_IMAGE
    is_mixed_panel_kind = panel_kind == _PLOT_SLICES_PANEL_MIXED
    cuts_page, basic_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesCutsPage"
    )
    limits_page, limits_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesLimitsPage"
    )
    colors_page, colors_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesColorsPage"
    )
    style_page, style_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesStylePage"
    )
    advanced_page, advanced_layout = tool._new_step_form_page(
        "figureComposerPlotSlicesAdvancedPage"
    )
    tool.operation_editor = cuts_page
    tool.operation_editor_layout = basic_layout

    shape_summary = QtWidgets.QLabel(
        "\n".join(
            (
                f"Sources: {shape.source_text}",
                f"Cuts: {shape.selection_text}",
                f"Result: {shape.panel_text}",
                shape.axes_text,
            )
        ),
        cuts_page,
    )
    shape_summary.setObjectName("figureComposerPlotSlicesShapeSummary")
    shape_summary.setWordWrap(True)
    if not shape.valid:
        shape_summary.setForegroundRole(QtGui.QPalette.ColorRole.Link)
    tool._add_form_row(
        basic_layout,
        "Data shape",
        shape_summary,
        "Shows the source dimensions, cuts, result per subplot, and targets.",
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
        basic_layout,
        "Cut dimension",
        dim_combo,
        "Data dimension passed as the slice keyword to plot_slices.",
    )

    values_text, values_mixed = tool._batch_text(
        operation, lambda target: target.slice_values, _format_tuple
    )
    values_edit = tool._line_edit(values_text)
    tool._apply_mixed_line_edit(values_edit, values_mixed)
    values_edit.setObjectName("figureComposerPlotSlicesValuesEdit")
    tool._connect_line_edit_finished(
        values_edit,
        lambda text: tool._update_current_operation_rebuild(
            slice_values=_float_tuple_from_text(text)
        ),
    )
    tool._add_form_row(
        basic_layout,
        "Cut values",
        values_edit,
        "Comma-separated coordinate values to slice along the cut dimension.",
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
        basic_layout,
        "Integration width",
        width_edit,
        "Optional qsel width around each cut value before plotting.",
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
        basic_layout,
        "Additional slice kwargs",
        slice_kwargs_edit,
        "Additional plot_slices selection kwargs passed to qsel.\n"
        "Use dimension keys such as kx=slice(-1, 1) or beta=0.",
    )

    order_mixed = tool._batch_is_mixed(operation, lambda target: target.order)
    order_combo = tool._combo(
        ["C", "F"],
        None if order_mixed else operation.order,
        lambda text: tool._update_current_operation_rebuild(order=text),
        parent=cuts_page,
        mixed=order_mixed,
    )
    order_combo.setObjectName("figureComposerOrderCombo")
    tool._add_form_row(
        basic_layout,
        "Panel order",
        order_combo,
        "C places sources by row and cuts by column. F places cuts by row "
        "and sources by column.",
    )

    options_widget = QtWidgets.QWidget(cuts_page)
    options_layout = QtWidgets.QHBoxLayout(options_widget)
    options_layout.setContentsMargins(0, 0, 0, 0)
    transpose_mixed = tool._batch_is_mixed(operation, lambda target: target.transpose)
    transpose_check = tool._check_box(
        operation.transpose,
        lambda checked: tool._update_current_operation_rebuild(transpose=checked),
        mixed=transpose_mixed,
    )
    transpose_check.setText("Transpose")
    transpose_check.setToolTip("Swap the plotted x/y orientation.")
    options_layout.addWidget(transpose_check)
    options_layout.addStretch(1)
    tool._add_form_row(
        basic_layout,
        "Options",
        options_widget,
        "Common plot_slices boolean options for this step.",
    )

    for label, attr in (("xlim", "xlim"), ("ylim", "ylim")):
        text, mixed = tool._batch_text(
            operation,
            _operation_field_getter(attr),
            _format_plot_limit,
        )
        edit = tool._line_edit(text)
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
        tool._add_form_row(
            limits_layout,
            label,
            edit,
            f"Optional {label}: one number for symmetric limits, "
            "or two comma-separated numbers for lower and upper limits.",
        )

    limits_options_widget = QtWidgets.QWidget(limits_page)
    limits_options_layout = QtWidgets.QHBoxLayout(limits_options_widget)
    limits_options_layout.setContentsMargins(0, 0, 0, 0)
    crop_mixed = tool._batch_is_mixed(operation, lambda target: target.crop)
    crop_check = tool._check_box(
        operation.crop,
        lambda checked: tool._update_current_operation(crop=checked),
        parent=limits_page,
        mixed=crop_mixed,
    )
    crop_check.setObjectName("figureComposerPlotSlicesCropCheck")
    crop_check.setText("Crop")
    crop_check.setToolTip("Crop each slice to explicit x/y limits before plotting.")
    limits_options_layout.addWidget(crop_check)
    limits_options_layout.addStretch(1)
    tool._add_form_row(
        limits_layout,
        "Options",
        limits_options_widget,
        "Limit-related plot_slices options for this step.",
    )

    if is_image_plot:
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
            "Colorbar",
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
            "Colorbar kwargs",
            colorbar_kwargs_edit,
            "Dict literal or keyword arguments forwarded as colorbar_kw.",
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
            parent=style_page,
            mixed=same_limits_mixed,
        )
        same_limits_combo.setObjectName("figureComposerSameLimitsCombo")
        tool._add_form_row(
            style_layout,
            "Match color limits",
            same_limits_combo,
            "Control plot_slices same_limits for image color scaling.",
        )

    axis_mixed = tool._batch_is_mixed(operation, lambda target: target.axis)
    axis_combo = tool._combo(
        ["auto", "on", "off", "equal", "scaled", "tight", "image", "square"],
        None if axis_mixed else operation.axis,
        lambda text: tool._update_current_operation(axis=text),
        parent=style_page,
        mixed=axis_mixed,
    )
    axis_combo.setObjectName("figureComposerAxisCombo")
    tool._add_form_row(
        style_layout,
        "Axis",
        axis_combo,
        "Matplotlib axis mode passed through plot_slices.",
    )

    label_options_widget = QtWidgets.QWidget(style_page)
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
        style_layout,
        "Labels",
        label_options_widget,
        "Label and annotation visibility options for plot_slices.",
    )

    annotate_kwargs_text, annotate_kwargs_mixed = tool._batch_text(
        operation, lambda target: target.annotate_kw, _format_dict
    )
    annotate_kwargs_edit = tool._line_edit(
        annotate_kwargs_text,
        parent=style_page,
    )
    tool._apply_mixed_line_edit(annotate_kwargs_edit, annotate_kwargs_mixed)
    annotate_kwargs_edit.setObjectName("figureComposerAnnotateKwEdit")
    tool._connect_line_edit_finished(
        annotate_kwargs_edit,
        lambda text: tool._update_current_operation(annotate_kw=_dict_from_text(text)),
    )
    tool._add_form_row(
        style_layout,
        "Annotation kwargs",
        annotate_kwargs_edit,
        "Dict literal or keyword arguments forwarded as annotate_kw.",
    )

    if is_line_plot:
        line_color_text, line_color_mixed = tool._batch_text(
            operation,
            lambda target: line_kw_text(target, "color", "c") or "",
            str,
        )
        line_color_edit = _ColorLineEditWidget(
            line_color_text,
            parent=colors_page,
        )
        line_color_edit.setLineEditObjectName("figureComposerPlotSlicesLineColorEdit")
        line_color_edit.setColorButtonObjectName(
            "figureComposerPlotSlicesLineColorButton"
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
                clear_legacy_cmap=True,
            ),
            unchanged_mixed=lambda: tool._line_edit_batch_unchanged(
                line_color_edit.line_edit
            ),
        )
        tool._add_form_row(
            colors_layout,
            "Line color",
            line_color_edit,
            "Matplotlib color stored as line_kw color for 1D panels.",
        )

        line_style_mixed = tool._batch_is_mixed(
            operation, lambda target: line_kw_text(target, "linestyle", "ls")
        )
        line_style_combo = tool._combo(
            LINE_STYLE_OPTIONS,
            None if line_style_mixed else line_kw_text(operation, "linestyle", "ls"),
            lambda text: update_current_line_kw(
                tool, "linestyle", text or None, aliases=("ls",)
            ),
            parent=colors_page,
            mixed=line_style_mixed,
        )
        line_style_combo.setObjectName("figureComposerPlotSlicesLineStyleCombo")
        tool._add_form_row(
            colors_layout,
            "Line style",
            line_style_combo,
            "Matplotlib linestyle for 1D plot_slices panels.",
        )

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
        tool._add_form_row(
            colors_layout,
            "Line width",
            line_width_row_widget,
            "Matplotlib linewidth for 1D plot_slices panels.",
        )

        marker_mixed = tool._batch_is_mixed(
            operation, lambda target: line_kw_text(target, "marker")
        )
        marker_combo = tool._combo(
            LINE_MARKER_OPTIONS,
            None if marker_mixed else line_kw_text(operation, "marker"),
            lambda text: update_current_line_kw(tool, "marker", text or None),
            parent=colors_page,
            mixed=marker_mixed,
        )
        marker_combo.setObjectName("figureComposerPlotSlicesMarkerCombo")
        tool._add_form_row(
            colors_layout,
            "Marker",
            marker_combo,
            "Matplotlib marker style for 1D plot_slices panels.",
        )

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
        tool._add_form_row(
            colors_layout,
            "Marker size",
            marker_size_row_widget,
            "Matplotlib marker size for 1D plot_slices panels.",
        )

        marker_face_text, marker_face_mixed = tool._batch_text(
            operation,
            lambda target: line_kw_text(target, "markerfacecolor", "mfc"),
            str,
        )
        marker_face_edit = _ColorLineEditWidget(
            marker_face_text,
            parent=colors_page,
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
        tool._add_form_row(
            colors_layout,
            "Marker face",
            marker_face_edit,
            "Matplotlib marker face color for 1D plot_slices panels.",
        )

        marker_edge_text, marker_edge_mixed = tool._batch_text(
            operation,
            lambda target: line_kw_text(target, "markeredgecolor", "mec"),
            str,
        )
        marker_edge_edit = _ColorLineEditWidget(
            marker_edge_text,
            parent=colors_page,
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
        tool._add_form_row(
            colors_layout,
            "Marker edge",
            marker_edge_edit,
            "Matplotlib marker edge color for 1D plot_slices panels.",
        )

        line_order_mixed = tool._batch_is_mixed(
            operation, lambda target: target.line_order
        )
        line_order_combo = tool._combo(
            ["default", "C", "F"],
            None if line_order_mixed else operation.line_order or "default",
            lambda text: tool._update_current_operation(
                line_order=None if text == "default" else text
            ),
            parent=colors_page,
            mixed=line_order_mixed,
        )
        line_order_combo.setObjectName("figureComposerPlotSlicesLineOrderCombo")
        tool._add_form_row(
            colors_layout,
            "Line order",
            line_order_combo,
            "Order used when line style values are provided per panel.",
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
            "Line kwargs",
            line_kwargs_edit,
            "Additional Matplotlib Line2D kwargs not covered by the controls above.",
        )

        gradient_mixed = tool._batch_is_mixed(operation, lambda target: target.gradient)
        gradient_check = tool._check_box(
            operation.gradient,
            lambda checked: tool._update_current_operation(gradient=checked),
            parent=colors_page,
            mixed=gradient_mixed,
        )
        gradient_check.setObjectName("figureComposerGradientCheck")
        gradient_check.setText("Fill under line")
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
            "Gradient kwargs",
            gradient_kwargs_edit,
            "Dict literal or keyword arguments forwarded as gradient_kw.",
        )
    elif is_image_plot:
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
            placeholder = "" if mixed else _norm_field_placeholder(operation, attr)
            if placeholder:
                edit.setPlaceholderText(placeholder)
            tool._connect_line_edit_finished(
                edit,
                _norm_number_update_callback(tool, attr),
            )
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
    return [
        ("cuts", "Cuts", cuts_page),
        ("limits", "Limits", limits_page),
        ("colors", "Colors", colors_page),
        ("style", "Style", style_page),
        ("advanced", "Advanced", advanced_page),
    ]


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


def _plot_slices_limit_placeholder(
    tool: FigureComposerTool, operation: FigureOperationState, attr: str
) -> str:
    if attr not in {"xlim", "ylim"} or getattr(operation, attr) is not None:
        return ""
    with _figure_style_context():
        figure = Figure(
            figsize=tool._recipe.setup.figsize,
            dpi=tool._recipe.setup.dpi,
            layout=tool._recipe.setup.layout,
        )
        canvas = FigureCanvasAgg(figure)
        try:
            axs = _rendering._make_axes(tool, figure, sync_visible=False)
            if tool._operation_has_invalid_axes(operation):
                return ""
            _render_plot_slices(tool, operation, axs)
            with _figure_draw_context():
                canvas.draw()
            axes = _rendering._iter_axes(
                _rendering._axes_from_selection(
                    tool, operation.axes, axs, for_plot_slices=False
                )
            )
            if not axes:
                return ""
            limits = axes[0].get_xlim() if attr == "xlim" else axes[0].get_ylim()
        except Exception:
            return ""
        finally:
            figure.clear()
    return _format_pair((float(limits[0]), float(limits[1])))


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

    tool._update_operations(update_operation)


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

    tool._update_operations(update_operation, render=False)
    tool._update_step_action_buttons()
    tool._refresh_step_section_button_texts()
    erlab.interactive.utils.single_shot(
        tool, 0, lambda: _rendering._render_preview(tool)
    )
    tool.sigInfoChanged.emit()


def _plot_slices_shape(
    tool: FigureComposerTool, operation: FigureOperationState
) -> _PlotSlicesShape:
    operation = _normalized_selection_operation(tool, operation)
    maps = _operation_maps(tool, operation)
    if not maps:
        return _PlotSlicesShape(
            source_text="No source data is available for this step.",
            selection_text="Select at least one valid source.",
            panel_text="Cannot determine the plotted data shape.",
            axes_text="Targets: " + tool._axes_target_text(operation.axes),
            plot_dims=(),
            plot_ndim=None,
            panel_count=0,
            valid=False,
        )

    source_names = (
        tuple(selection.source for selection in operation.map_selections)
        if operation.map_selections
        else operation.sources
    )
    source_labels: list[str] = []
    for index, data in enumerate(maps):
        source_name = (
            source_names[index] if index < len(source_names) else f"map {index + 1}"
        )
        source_labels.append(f"{source_name}({_format_dim_sizes(data)})")

    plot_maps = [data.T if operation.transpose else data for data in maps]
    dims = tuple(str(dim) for dim in plot_maps[0].dims)
    if any(tuple(str(dim) for dim in data.dims) != dims for data in plot_maps[1:]):
        return _PlotSlicesShape(
            source_text="; ".join(source_labels),
            selection_text="All plot_slices sources must have matching dimensions.",
            panel_text="Cannot plot mixed source dimensions in one step.",
            axes_text="Targets: " + tool._axes_target_text(operation.axes),
            plot_dims=(),
            plot_ndim=None,
            panel_count=0,
            valid=False,
        )

    selected_dims: set[str] = set()
    selection_parts: list[str] = []
    slice_count = 1
    if operation.slice_dim:
        if operation.slice_dim not in dims:
            selection_parts.append(f"{operation.slice_dim}: not in source dims")
        elif operation.slice_values:
            selected_dims.add(operation.slice_dim)
            slice_count = len(operation.slice_values)
            width_text = (
                ""
                if operation.slice_width is None
                else f", width {operation.slice_width:g}"
            )
            selection_parts.append(
                f"{operation.slice_dim}: {slice_count} cut"
                f"{'s' if slice_count != 1 else ''}{width_text}"
            )
        else:
            selection_parts.append(f"{operation.slice_dim}: choose cut values")
    else:
        selection_parts.append("No cut dimension selected")

    advanced_selections: list[str] = []
    for key, value in operation.slice_kwargs.items():
        if key.endswith("_width") or key not in dims:
            continue
        count = _selection_value_count(value)
        if isinstance(value, slice):
            advanced_selections.append(f"{key}: range")
        elif count is None:
            selected_dims.add(key)
            advanced_selections.append(f"{key}: single value")
        else:
            selected_dims.add(key)
            slice_count = max(slice_count, count)
            advanced_selections.append(f"{key}: {count} cut{'s' if count != 1 else ''}")
    if advanced_selections:
        selection_parts.append("Advanced: " + ", ".join(advanced_selections))

    plot_dims = tuple(dim for dim in dims if dim not in selected_dims)
    panel_count = len(maps) * slice_count
    if operation.axes.expression:
        axes_text = f"Targets: {operation.axes.expression}"
    else:
        if tool._recipe.setup.layout_mode == "gridspec":
            target_count = len(operation.axes.axes_ids)
        else:
            target_count = len(operation.axes.valid_axes(tool._recipe.setup))
        axes_text = (
            f"Targets: {target_count} selected for {panel_count} panel"
            f"{'s' if panel_count != 1 else ''}"
        )

    if len(plot_dims) == 1:
        panel_text = f"Each target panel: 1D line over {plot_dims[0]}"
        valid = True
    elif len(plot_dims) == 2:
        panel_text = f"Each target panel: 2D image, y={plot_dims[0]}, x={plot_dims[1]}"
        valid = True
    else:
        dims_text = "none" if not plot_dims else " × ".join(plot_dims)
        panel_text = (
            f"Each target panel would be {len(plot_dims)}D ({dims_text}); "
            "choose cuts until each panel is 1D or 2D."
        )
        valid = False

    source_text = "; ".join(source_labels)
    if operation.transpose:
        source_text += " (plot order transposed)"
    return _PlotSlicesShape(
        source_text=source_text,
        selection_text="; ".join(selection_parts),
        panel_text=panel_text,
        axes_text=axes_text,
        plot_dims=plot_dims,
        plot_ndim=len(plot_dims),
        panel_count=panel_count,
        valid=valid,
    )


def _plot_slices_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    operation = _normalized_selection_operation(tool, operation)
    kwargs: dict[str, typing.Any] = {}
    shape = _plot_slices_shape(tool, operation)
    is_line_plot = shape.plot_ndim == 1
    kwargs.update(dict(operation.slice_kwargs))
    if operation.slice_dim and operation.slice_values:
        kwargs[operation.slice_dim] = list(operation.slice_values)
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
    if operation.cmap and not is_line_plot:
        kwargs["cmap"] = operation.cmap
    if is_line_plot:
        line_kw = dict(operation.line_kw)
        if line_kw:
            kwargs["line_kw"] = line_kw
        if operation.line_order is not None:
            kwargs["line_order"] = operation.line_order
    if not is_line_plot:
        if _use_powernorm_plot_kwargs(operation):
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


def _render_plot_slices(
    tool: FigureComposerTool, operation: FigureOperationState, axs: typing.Any
) -> None:
    operation = _normalized_selection_operation(tool, operation)
    maps = _operation_maps(tool, operation)
    if not maps:
        return
    axes = _plot_slices_axes(
        operation,
        maps,
        _rendering._axes_from_selection(
            tool, operation.axes, axs, for_plot_slices=True
        ),
    )
    eplt.plot_slices(
        maps,
        axes=typing.cast("Iterable[matplotlib.axes.Axes]", axes),
        **_plot_slices_kwargs(tool, operation),
    )


def _plot_slices_axes(
    operation: FigureOperationState, maps: Sequence[xr.DataArray], axes: object
) -> object:
    if not isinstance(axes, np.ndarray):
        return axes
    slice_count = max(len(operation.slice_values), 1)
    if operation.order == "F":
        shape = (slice_count, len(maps))
    else:
        shape = (len(maps), slice_count)
    if axes.size != math.prod(shape):
        return axes
    return axes.reshape(shape)


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
        tool._source_data[name]
        for name in operation.sources
        if name in tool._source_data
    ]


def _plot_slices_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    operation = _normalized_selection_operation(tool, operation)
    if operation.map_selections:
        maps_code = "selected_maps"
    else:
        sources = [_valid_source_variable(source) for source in operation.sources]
        if not sources:
            return None
        maps_code = sources[0] if len(sources) == 1 else f"[{', '.join(sources)}]"

    kwargs = _plot_slices_code_kwargs(tool, operation)
    kwargs["axes"] = _RawCode(_axes_code(tool, operation.axes, for_plot_slices=True))
    kwargs_text = _code_kwargs(kwargs)
    return f"eplt.plot_slices({maps_code}, {kwargs_text})"


def _plot_slices_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    code = _plot_slices_code(tool, operation)
    if code is None:
        return []
    if not operation.map_selections:
        return [code]
    lines = ["selected_maps = ["]
    lines.extend(
        f"    {_selection_code(selection)}," for selection in operation.map_selections
    )
    lines.append("]")
    lines.append(code)
    return lines


def _plot_slices_code_kwargs(
    tool: FigureComposerTool, operation: FigureOperationState
) -> dict[str, typing.Any]:
    kwargs = _plot_slices_kwargs(tool, operation)
    is_line_plot = _plot_slices_shape(tool, operation).plot_ndim == 1
    if not is_line_plot and not _use_powernorm_plot_kwargs(operation):
        kwargs["norm"] = _RawCode(_norm_code(operation))
    return kwargs


_SECTION_TOOLTIPS = {
    "cuts": "Choose slice dimension, cut values, and extraction options.",
    "limits": "Set explicit x/y axis limits for this slice plot.",
    "colors": "Set image color scaling or 1D line styling for this plot_slices step.",
    "style": "Set labels, annotation, aspect, and shared color-limit options.",
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
    source_text = ", ".join(operation.sources) or "missing source"
    shape = _plot_slices_shape(tool, operation)
    plot_kind = "Line slices" if shape.plot_ndim == 1 else "Image slices"
    if operation.slice_dim and operation.slice_values:
        cut_text = f"{operation.slice_dim} = {len(operation.slice_values)} cuts"
    else:
        cut_text = "current selection"
    return f"{prefix}{plot_kind}: {source_text}, {cut_text}"


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
    return operation.sources


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
    check = QtWidgets.QCheckBox(source_name, selector)
    check.setObjectName(f"figureComposerPlotSlicesSourceCheck_{index}")
    check.setProperty("figure_source_name", source_name)
    check.setToolTip("Include this DataArray in the maps passed to plot_slices.")
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
        QtCore.Qt.ArrowType.UpArrow,
        order_controls_enabled and selected_index > 0,
        "Move this input earlier in the maps argument.",
        lambda: _plot_source_move(tool, source_name, -1),
    )
    down_button = _plot_source_move_button(
        tool,
        selector,
        source_name,
        "down",
        QtCore.Qt.ArrowType.DownArrow,
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
    arrow: QtCore.Qt.ArrowType,
    enabled: bool,
    tooltip: str,
    clicked: typing.Callable[[], None],
) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton(parent)
    button.setObjectName(
        f"figureComposerPlotSlicesSourceMove_{direction}_{source_name}"
    )
    button.setProperty("figure_source_name", source_name)
    button.setProperty("figure_source_move", direction)
    button.setArrowType(arrow)
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
            return ", ".join(operation.sources) or "none"
        case "axes":
            return tool._axes_target_text(operation.axes)
        case "cuts":
            if operation.slice_dim and operation.slice_values:
                return f"{operation.slice_dim}, {len(operation.slice_values)}"
            if operation.slice_kwargs:
                return "additional"
            return "none"
        case "limits":
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
                return line_kw_text(operation, "color", "c") or "line"
            return operation.cmap or "default"
        case "style":
            return operation.axis
        case "advanced":
            return "set" if _effective_extra_kwargs(tool, operation) else "optional"
    return ""


def _required_imports(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    imports = ["import erlab.plotting as eplt"]
    if (
        operation.enabled
        and _plot_slices_shape(tool, operation).plot_ndim != 1
        and not _use_powernorm_plot_kwargs(operation)
        and _effective_norm_name(operation.norm_name) in _MATPLOTLIB_NORM_NAMES
    ):
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
