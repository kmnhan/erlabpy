"""Build and update the plot-slices operation editor."""

from __future__ import annotations

import math
import typing

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._figurecomposer._labels import (
    label_editor_text,
    label_text_tooltip,
    operations_with_line_label_text,
)
from erlab.interactive._figurecomposer._line_colormap import (
    LINE_COLOR_CMAP_TRIM_MAX,
    effective_line_color_coord,
    line_color_cmap_trim_control_values,
    line_colormap_active,
)
from erlab.interactive._figurecomposer._line_style import (
    LINE_MARKER_OPTIONS,
    LINE_STYLE_DEFAULT_LABEL,
    LINE_STYLE_OPTIONS,
    color_kw_value_from_text,
    extra_line_kw,
    line_kw_float,
    line_kw_style_value,
    line_kw_text,
)
from erlab.interactive._figurecomposer._model._sources import _available_source_dims
from erlab.interactive._figurecomposer._model._state import (
    _POWER_NORM_NAME,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
)
from erlab.interactive._figurecomposer._norms import (
    _ZERO_VCENTER_NORMS,
    _cmap_base_and_reverse,
    _cmap_with_reverse,
    _effective_norm_name,
    _norm_combo_choices,
    _norm_combo_text,
    _norm_kwarg_fields,
    _norm_name_from_combo_text,
    _norm_updates_from_kwargs,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _panel_style_editor,
)
from erlab.interactive._figurecomposer._operations._plot_slices._model import (
    _LINE_COLOR_MODE_TEXT,
    _PLOT_SLICES_PANEL_IMAGE,
    _PLOT_SLICES_PANEL_LINE,
    _PLOT_SLICES_PANEL_MIXED,
    _SLICE_VALUES_MODE_LABELS,
    _all_coordinate_slice_values_summary,
    _available_plot_slices_line_color_coords,
    _effective_extra_kwargs,
    _effective_slice_kwargs,
    _line_color_mode_from_text,
    _line_color_mode_text,
    _norm_clip_from_text,
    _norm_clip_text,
    _normalized_selection_operation,
    _plot_slices_batch_panel_kind,
    _plot_slices_line_label_contexts,
    _plot_slices_line_profiles,
    _plot_slices_operation_with_sources,
    _plot_slices_panel_keys,
    _plot_slices_selection_sources,
    _plot_slices_shape,
    _selection_updates_from_kwargs,
    _slice_values_mode_from_text,
    _slice_values_mode_text,
    _use_all_coordinate_slice_values,
)
from erlab.interactive._figurecomposer._text import (
    _dict_from_text,
    _float_tuple_from_text,
    _format_dict,
    _format_pair,
    _format_plot_limit,
    _format_tuple,
    _plot_limit_from_text,
)
from erlab.interactive._figurecomposer._ui._color_widgets import _ColorLineEditWidget
from erlab.interactive._figurecomposer._ui._label_help import legend_label_input_widget
from erlab.interactive._figurecomposer._ui._line_style import (
    optional_positive_spinbox,
    optional_positive_spinbox_value,
    update_current_extra_line_kw,
    update_current_line_kw,
)
from erlab.interactive._figurecomposer._ui._line_transform import (
    add_line_transform_controls,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import matplotlib.axes

    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


def _plot_slices_cmap_base_and_reverse(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[str, bool]:
    if operation.cmap is None:
        return _cmap_base_and_reverse(str(editor.styled_rcparams_value("image.cmap")))
    return _cmap_base_and_reverse(operation.cmap)


def _available_plot_slices_offset_coords(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> list[str]:
    profiles, _keys = _plot_slices_line_profiles(
        editor.context, editor.source_display_name, operation
    )
    names: list[str] = []
    for profile in profiles:
        for name, coord in profile.coords.items():
            name_str = str(name)
            if name_str in profile.dims or name_str in names:
                continue
            if np.asarray(coord.values).reshape(-1).size == 1:
                names.append(name_str)
    return names


def _add_plot_slices_line_color_controls(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
) -> None:
    row = QtWidgets.QWidget(page)
    row_layout = QtWidgets.QVBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(4)

    mode_mixed = editor.batch_is_mixed(operation, lambda target: target.line_color_mode)
    mode_combo = editor.combo(
        list(_LINE_COLOR_MODE_TEXT.values()),
        None if mode_mixed else _line_color_mode_text(operation.line_color_mode),
        lambda text: editor.request_update_rebuild(
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
        _add_plot_slices_coordinate_color_controls(editor, operation, page, row_layout)
    else:
        _add_plot_slices_manual_color_controls(editor, operation, page, row_layout)

    editor.add_form_row(
        layout,
        "Color",
        row,
        "Choose a manual line color or color 1D panels from coordinate values.",
    )


def _add_plot_slices_manual_color_controls(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QVBoxLayout,
) -> None:
    line_color_text, line_color_mixed = editor.batch_text(
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
    editor.apply_mixed_line_edit(line_color_edit.line_edit, line_color_mixed)
    editor.connect_value_signal(
        line_color_edit,
        line_color_edit.editingFinished,
        line_color_edit.text,
        lambda text: update_current_line_kw(
            editor,
            "color",
            color_kw_value_from_text(text),
            aliases=("c",),
            clear_stale_cmap=True,
            clear_stale_line_colormap=True,
        ),
        unchanged_mixed=lambda: editor.line_edit_batch_unchanged(
            line_color_edit.line_edit
        ),
    )
    layout.addWidget(line_color_edit)


def _add_plot_slices_coordinate_color_controls(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    page: QtWidgets.QWidget,
    layout: QtWidgets.QVBoxLayout,
) -> None:
    coord_options = _available_plot_slices_line_color_coords(
        editor.context, editor.source_display_name, operation
    )
    coord_options_match = editor.batch_options_match(
        operation,
        lambda target: _available_plot_slices_line_color_coords(
            editor.context, editor.source_display_name, target
        ),
    )
    coord_mixed = editor.batch_is_mixed(
        operation, lambda target: target.line_color_coord
    )
    coord_combo = editor.optional_name_combo(
        coord_options,
        None
        if coord_mixed
        else effective_line_color_coord(operation, operation.slice_dim),
        "Choose coordinate",
        lambda value: editor.request_update(line_color_coord=value),
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

    editor.connect_signal(
        cmap_combo,
        cmap_combo.activated,
        lambda _index: _update_current_plot_slices_line_color_cmap(
            editor, cmap_combo.current_matplotlib_name(), reverse_check.isChecked()
        ),
    )
    editor.connect_signal(
        reverse_check,
        reverse_check.stateChanged,
        lambda state: _update_current_plot_slices_line_color_cmap(
            editor,
            cmap_combo.current_matplotlib_name(),
            QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked,
        ),
    )

    cmap_layout.addWidget(cmap_combo, 1)
    cmap_layout.addWidget(reverse_check)

    trim_lower_mixed = editor.batch_is_mixed(
        operation, lambda target: target.line_color_cmap_trim_lower
    )
    trim_upper_mixed = editor.batch_is_mixed(
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
    editor.connect_value_signal(
        lower_spin,
        lower_spin.valueChanged,
        float,
        lambda value: editor.request_update(line_color_cmap_trim_lower=value),
    )
    editor.connect_value_signal(
        upper_spin,
        upper_spin.valueChanged,
        float,
        lambda value: editor.request_update(line_color_cmap_trim_upper=value),
    )
    trim_layout.addWidget(trim_label)
    trim_layout.addWidget(QtWidgets.QLabel("Low", trim_row))
    trim_layout.addWidget(
        editor.mixed_value_widget(lower_spin, mixed=trim_lower_mixed, parent=page)
    )
    trim_layout.addWidget(QtWidgets.QLabel("High", trim_row))
    trim_layout.addWidget(
        editor.mixed_value_widget(upper_spin, mixed=trim_upper_mixed, parent=page)
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
    editor: FigureOperationEditor, base: str, reverse: bool
) -> None:
    editor.request_update(
        line_color_cmap=_cmap_with_reverse(base, False),
        line_color_cmap_reverse=reverse,
    )


def _build_plot_slices_editor(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> list[tuple[str, str, QtWidgets.QWidget]]:
    operation = _normalized_selection_operation(editor.context, operation)
    shape = _plot_slices_shape(editor.context, operation)
    panel_kind = _plot_slices_batch_panel_kind(
        editor.context, editor.editable_operations(), operation
    )
    is_line_plot = panel_kind == _PLOT_SLICES_PANEL_LINE
    is_image_plot = panel_kind == _PLOT_SLICES_PANEL_IMAGE
    is_mixed_panel_kind = panel_kind == _PLOT_SLICES_PANEL_MIXED
    selection_page, selection_layout = editor.new_form_page(
        "figureComposerPlotSlicesSelectionPage"
    )
    view_page, view_layout = editor.new_form_page("figureComposerPlotSlicesViewPage")
    colors_page, colors_layout = editor.new_form_page(
        "figureComposerPlotSlicesColorsPage"
    )
    transform_page, transform_layout = editor.new_form_page(
        "figureComposerPlotSlicesTransformPage"
    )
    advanced_page, advanced_layout = editor.new_form_page(
        "figureComposerPlotSlicesAdvancedPage"
    )
    editor.add_form_section(
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
    editor.add_form_row(
        selection_layout,
        "Summary",
        shape_summary,
        "Shows the input dimensions and the dimensions plotted by this step.",
    )

    editor.add_form_section(
        selection_layout,
        "Slice values",
        object_name="figureComposerPlotSlicesSelectionValuesSection",
    )
    dims = _available_source_dims(
        editor.context.source_data, _plot_slices_selection_sources(operation)
    )
    dim_mixed = editor.batch_is_mixed(operation, lambda target: target.slice_dim)
    dim_combo = editor.combo(
        ["", *dims],
        None if dim_mixed else operation.slice_dim or "",
        lambda text: editor.request_update_rebuild(slice_dim=text or None),
        parent=selection_page,
        mixed=dim_mixed,
    )
    dim_combo.setObjectName("figureComposerPlotSlicesDimensionCombo")
    editor.add_form_row(
        selection_layout,
        "Dimension",
        dim_combo,
        "Data dimension passed as the slice keyword to plot_slices.",
    )

    values_mode_mixed = editor.batch_is_mixed(
        operation, lambda target: target.slice_values_mode
    )
    values_mode_combo = editor.combo(
        tuple(_SLICE_VALUES_MODE_LABELS.values()),
        None
        if values_mode_mixed
        else _slice_values_mode_text(operation.slice_values_mode),
        lambda text: editor.request_update_rebuild(
            slice_values_mode=_slice_values_mode_from_text(text)
        ),
        parent=selection_page,
        mixed=values_mode_mixed,
    )
    values_mode_combo.setObjectName("figureComposerPlotSlicesValuesModeCombo")
    values_mode_combo.setToolTip(
        "Choose manual values or all values from the dimension coordinate."
    )
    editor.add_form_row(
        selection_layout,
        "Values",
        values_mode_combo,
        values_mode_combo.toolTip(),
    )
    if not values_mode_mixed and _use_all_coordinate_slice_values(operation):
        coordinate_summary = QtWidgets.QLabel(
            _all_coordinate_slice_values_summary(editor.context, operation),
            selection_page,
        )
        coordinate_summary.setObjectName("figureComposerPlotSlicesCoordinateSummary")
        coordinate_summary.setWordWrap(True)
        editor.add_form_row(
            selection_layout,
            "Coordinate",
            coordinate_summary,
            "Shows the coordinate values that will be passed to plot_slices.",
        )

        thin_mixed = editor.batch_is_mixed(
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
        editor.connect_value_signal(
            thin_spin,
            thin_spin.valueChanged,
            int,
            lambda value: editor.request_update_rebuild(slice_values_thin=value),
        )
        editor.add_form_row(
            selection_layout,
            "Thin",
            editor.mixed_value_widget(
                thin_spin, mixed=thin_mixed, parent=selection_page
            ),
            thin_spin.toolTip(),
        )
    elif not values_mode_mixed:
        values_text, values_mixed = editor.batch_text(
            operation, lambda target: target.slice_values, _format_tuple
        )
        values_edit = editor.line_edit(values_text, parent=selection_page)
        editor.apply_mixed_line_edit(values_edit, values_mixed)
        values_edit.setObjectName("figureComposerPlotSlicesValuesEdit")
        editor.connect_line_edit_finished(
            values_edit,
            lambda text: editor.request_update_rebuild(
                slice_values=_float_tuple_from_text(text)
            ),
        )
        editor.add_form_row(
            selection_layout,
            "Manual",
            values_edit,
            "Comma-separated coordinate values to select along the dimension.",
        )

    width_text, width_mixed = editor.batch_text(
        operation,
        lambda target: target.slice_width,
        lambda value: "" if value is None else f"{value:g}",
    )
    width_edit = editor.line_edit(width_text, parent=selection_page)
    editor.apply_mixed_line_edit(width_edit, width_mixed)
    width_edit.setObjectName("figureComposerPlotSlicesWidthEdit")
    editor.connect_line_edit_finished(
        width_edit,
        lambda text: editor.request_update_rebuild(
            slice_width=float(text) if text.strip() else None
        ),
    )
    editor.add_form_row(
        selection_layout,
        "Width",
        width_edit,
        "Optional qsel width around each selected value before plotting.",
    )

    slice_kwargs_text, slice_kwargs_mixed = editor.batch_text(
        operation, lambda target: target.slice_kwargs, _format_dict
    )
    slice_kwargs_edit = editor.line_edit(slice_kwargs_text, parent=selection_page)
    editor.apply_mixed_line_edit(slice_kwargs_edit, slice_kwargs_mixed)
    slice_kwargs_edit.setObjectName("figureComposerPlotSlicesSliceKwargsEdit")
    editor.connect_line_edit_finished(
        slice_kwargs_edit,
        lambda text: _update_current_slice_kwargs(editor, text),
    )
    editor.add_form_row(
        selection_layout,
        "Extra kwargs",
        slice_kwargs_edit,
        "Additional plot_slices selection kwargs passed to qsel.\n"
        "Use dimension keys such as kx=slice(-1, 1) or beta=0.",
    )

    editor.add_form_section(
        view_layout,
        "Panels",
        object_name="figureComposerPlotSlicesViewPanelsSection",
    )
    order_mixed = editor.batch_is_mixed(operation, lambda target: target.order)
    order_combo = editor.combo(
        ["C", "F"],
        None if order_mixed else operation.order,
        lambda text: editor.request_update_rebuild(order=text),
        parent=view_page,
        mixed=order_mixed,
    )
    order_combo.setObjectName("figureComposerOrderCombo")
    editor.add_form_row(
        view_layout,
        "Order",
        order_combo,
        "C places sources by row and selected values by column. F places selected "
        "values by row and sources by column.",
    )

    transpose_mixed = editor.batch_is_mixed(operation, lambda target: target.transpose)
    transpose_check = editor.check_box(
        operation.transpose,
        lambda checked: editor.request_update_rebuild(transpose=checked),
        parent=view_page,
        mixed=transpose_mixed,
    )
    transpose_check.setObjectName("figureComposerTransposeCheck")
    transpose_check.setText("")
    transpose_check.setToolTip("Swap the plotted x/y orientation.")
    editor.add_form_row(
        view_layout,
        "Transpose",
        transpose_check,
        "Swap the plotted x/y orientation.",
    )

    limit_controls: list[tuple[str, QtWidgets.QWidget, str]] = []
    for label, attr in (("x", "xlim"), ("y", "ylim")):
        text, mixed = editor.batch_text(
            operation,
            _operation_field_getter(attr),
            _format_plot_limit,
        )
        edit = editor.line_edit(text, parent=view_page)
        edit.setObjectName(f"figureComposerPlotSlices{label.upper()}LimEdit")
        editor.apply_mixed_line_edit(edit, mixed)
        placeholder = (
            "" if mixed else _plot_slices_limit_placeholder(editor, operation, attr)
        )
        if placeholder:
            edit.setPlaceholderText(placeholder)
        editor.connect_line_edit_finished(
            edit,
            _plot_limit_update_callback(editor, attr),
        )
        limit_controls.append(
            (
                label,
                edit,
                f"Optional {attr}: one number for symmetric limits, "
                "or two comma-separated numbers for lower and upper limits.",
            )
        )
    editor.add_compound_form_row(
        view_layout,
        "Limits",
        limit_controls,
        "Optional x/y plot limits for this step.",
    )

    editor.add_form_section(
        view_layout,
        "Axes",
        object_name="figureComposerPlotSlicesViewAxesSection",
    )
    crop_mixed = editor.batch_is_mixed(operation, lambda target: target.crop)
    crop_check = editor.check_box(
        operation.crop,
        lambda checked: editor.request_update(crop=checked),
        parent=view_page,
        mixed=crop_mixed,
    )
    crop_check.setObjectName("figureComposerPlotSlicesCropCheck")
    crop_check.setText("")
    crop_check.setToolTip("Crop each slice to explicit x/y limits before plotting.")
    editor.add_form_row(
        view_layout,
        "Crop",
        crop_check,
        "Crop each slice to explicit x/y limits before plotting.",
    )

    axis_mixed = editor.batch_is_mixed(operation, lambda target: target.axis)
    axis_combo = editor.combo(
        ["auto", "on", "off", "equal", "scaled", "tight", "image", "square"],
        None if axis_mixed else operation.axis,
        lambda text: editor.request_update(axis=text),
        parent=view_page,
        mixed=axis_mixed,
    )
    axis_combo.setObjectName("figureComposerAxisCombo")
    editor.add_form_row(
        view_layout,
        "Axis",
        axis_combo,
        "Matplotlib axis mode passed through plot_slices.",
    )

    label_options_widget = QtWidgets.QWidget(view_page)
    label_options_layout = QtWidgets.QHBoxLayout(label_options_widget)
    label_options_layout.setContentsMargins(0, 0, 0, 0)
    show_labels_mixed = editor.batch_is_mixed(
        operation, lambda target: target.show_all_labels
    )
    show_labels_check = editor.check_box(
        operation.show_all_labels,
        lambda checked: editor.request_update(show_all_labels=checked),
        parent=label_options_widget,
        mixed=show_labels_mixed,
    )
    show_labels_check.setText("All labels")
    show_labels_check.setToolTip("Ask plot_slices to show labels on every axis.")
    annotate_mixed = editor.batch_is_mixed(operation, lambda target: target.annotate)
    annotate_check = editor.check_box(
        operation.annotate,
        lambda checked: editor.request_update(annotate=checked),
        parent=label_options_widget,
        mixed=annotate_mixed,
    )
    annotate_check.setText("Annotate")
    annotate_check.setToolTip("Show the slice-value annotation text.")
    label_options_layout.addWidget(show_labels_check)
    label_options_layout.addWidget(annotate_check)
    label_options_layout.addStretch(1)
    editor.add_form_row(
        view_layout,
        "Labels",
        label_options_widget,
        "Label and annotation visibility options for plot_slices.",
    )

    annotate_kwargs_text, annotate_kwargs_mixed = editor.batch_text(
        operation, lambda target: target.annotate_kw, _format_dict
    )
    annotate_kwargs_edit = editor.line_edit(
        annotate_kwargs_text,
        parent=view_page,
    )
    editor.apply_mixed_line_edit(annotate_kwargs_edit, annotate_kwargs_mixed)
    annotate_kwargs_edit.setObjectName("figureComposerAnnotateKwEdit")
    editor.connect_line_edit_finished(
        annotate_kwargs_edit,
        lambda text: editor.request_update(annotate_kw=_dict_from_text(text)),
    )
    editor.add_form_row(
        view_layout,
        "Annotation",
        annotate_kwargs_edit,
        "Dict literal or keyword arguments forwarded as annotate_kw.",
    )

    if is_line_plot:
        editor.add_form_section(
            colors_layout,
            "Legend",
            object_name="figureComposerPlotSlicesStyleLegendSection",
        )
        labels_text, labels_mixed = editor.batch_text(
            operation,
            label_editor_text,
            str,
        )
        labels_edit = editor.line_edit(labels_text, parent=colors_page)
        editor.apply_mixed_line_edit(labels_edit, labels_mixed)
        labels_edit.setObjectName("figureComposerPlotSlicesLineLabelsEdit")
        editor.connect_line_edit_finished(
            labels_edit,
            lambda text: editor.request_recipe_transform(
                lambda operations, operation_ids: operations_with_line_label_text(
                    operations, operation_ids, text
                )
            ),
        )
        label_contexts = _plot_slices_line_label_contexts(
            editor.context, editor.source_display_name, operation
        )
        labels_widget = legend_label_input_widget(
            labels_edit,
            label_contexts,
            item_name="slice",
            button_object_name="figureComposerPlotSlicesLineLabelsHelpButton",
            parent=colors_page,
        )
        editor.add_form_row(
            colors_layout,
            "Labels",
            labels_widget,
            label_text_tooltip(label_contexts, item_name="slice"),
        )

        editor.add_form_section(
            colors_layout,
            "Line",
            object_name="figureComposerPlotSlicesStyleLineSection",
        )
        _add_plot_slices_line_color_controls(
            editor, operation, colors_page, colors_layout
        )

        line_style_mixed = editor.batch_is_mixed(
            operation, lambda target: line_kw_style_value(target, "linestyle", "ls")
        )
        line_style_combo = editor.optional_name_combo(
            LINE_STYLE_OPTIONS,
            None
            if line_style_mixed
            else line_kw_style_value(operation, "linestyle", "ls"),
            LINE_STYLE_DEFAULT_LABEL,
            lambda text: update_current_line_kw(
                editor, "linestyle", text, aliases=("ls",)
            ),
            parent=colors_page,
            mixed=line_style_mixed,
        )
        line_style_combo.setObjectName("figureComposerPlotSlicesLineStyleCombo")
        line_width_mixed = editor.batch_is_mixed(
            operation, lambda target: line_kw_text(target, "linewidth", "lw")
        )
        line_width_spin = optional_positive_spinbox(
            None if line_width_mixed else line_kw_float(operation, "linewidth", "lw"),
            parent=colors_page,
        )
        editor.connect_signal(
            line_width_spin,
            line_width_spin.valueChanged,
            lambda value: update_current_line_kw(
                editor,
                "linewidth",
                optional_positive_spinbox_value(value),
                aliases=("lw",),
            ),
        )
        line_width_spin.setObjectName("figureComposerPlotSlicesLineWidthSpin")
        line_width_row_widget = editor.mixed_value_widget(
            line_width_spin,
            mixed=line_width_mixed,
            parent=colors_page,
        )
        editor.add_compound_form_row(
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

        marker_mixed = editor.batch_is_mixed(
            operation, lambda target: line_kw_style_value(target, "marker")
        )
        marker_combo = editor.optional_name_combo(
            LINE_MARKER_OPTIONS,
            None if marker_mixed else line_kw_style_value(operation, "marker"),
            LINE_STYLE_DEFAULT_LABEL,
            lambda text: update_current_line_kw(editor, "marker", text),
            parent=colors_page,
            mixed=marker_mixed,
        )
        marker_combo.setObjectName("figureComposerPlotSlicesMarkerCombo")
        marker_size_mixed = editor.batch_is_mixed(
            operation, lambda target: line_kw_text(target, "markersize", "ms")
        )
        marker_size_spin = optional_positive_spinbox(
            None if marker_size_mixed else line_kw_float(operation, "markersize", "ms"),
            parent=colors_page,
        )
        editor.connect_signal(
            marker_size_spin,
            marker_size_spin.valueChanged,
            lambda value: update_current_line_kw(
                editor,
                "markersize",
                optional_positive_spinbox_value(value),
                aliases=("ms",),
            ),
        )
        marker_size_spin.setObjectName("figureComposerPlotSlicesMarkerSizeSpin")
        marker_size_row_widget = editor.mixed_value_widget(
            marker_size_spin,
            mixed=marker_size_mixed,
            parent=colors_page,
        )
        editor.add_compound_form_row(
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

        marker_face_text, marker_face_mixed = editor.batch_text(
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
        editor.apply_mixed_line_edit(marker_face_edit.line_edit, marker_face_mixed)
        editor.connect_value_signal(
            marker_face_edit,
            marker_face_edit.editingFinished,
            marker_face_edit.text,
            lambda text: update_current_line_kw(
                editor,
                "markerfacecolor",
                color_kw_value_from_text(text),
                aliases=("mfc",),
            ),
            unchanged_mixed=lambda: editor.line_edit_batch_unchanged(
                marker_face_edit.line_edit
            ),
        )

        marker_edge_text, marker_edge_mixed = editor.batch_text(
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
        editor.apply_mixed_line_edit(marker_edge_edit.line_edit, marker_edge_mixed)
        editor.connect_value_signal(
            marker_edge_edit,
            marker_edge_edit.editingFinished,
            marker_edge_edit.text,
            lambda text: update_current_line_kw(
                editor,
                "markeredgecolor",
                color_kw_value_from_text(text),
                aliases=("mec",),
            ),
            unchanged_mixed=lambda: editor.line_edit_batch_unchanged(
                marker_edge_edit.line_edit
            ),
        )
        editor.add_compound_form_row(
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

        line_kwargs_text, line_kwargs_mixed = editor.batch_text(
            operation, extra_line_kw, _format_dict
        )
        line_kwargs_edit = editor.line_edit(line_kwargs_text, parent=colors_page)
        editor.apply_mixed_line_edit(line_kwargs_edit, line_kwargs_mixed)
        line_kwargs_edit.setObjectName("figureComposerPlotSlicesLineKwEdit")
        editor.connect_line_edit_finished(
            line_kwargs_edit,
            lambda text: update_current_extra_line_kw(editor, _dict_from_text(text)),
        )
        editor.add_form_row(
            colors_layout,
            "Kwargs",
            line_kwargs_edit,
            "Additional Matplotlib Line2D kwargs not covered by the controls above.",
        )

        add_line_transform_controls(
            editor,
            operation,
            transform_page,
            transform_layout,
            object_prefix="figureComposerPlotSlicesLine",
            offset_coord_options=lambda target: _available_plot_slices_offset_coords(
                editor, target
            ),
        )

        editor.add_form_section(
            colors_layout,
            "Fill",
            object_name="figureComposerPlotSlicesStyleFillSection",
        )

        gradient_mixed = editor.batch_is_mixed(
            operation, lambda target: target.gradient
        )
        gradient_check = editor.check_box(
            operation.gradient,
            lambda checked: editor.request_update(gradient=checked),
            parent=colors_page,
            mixed=gradient_mixed,
        )
        gradient_check.setObjectName("figureComposerGradientCheck")
        gradient_check.setText("Gradient Fill")
        editor.add_form_row(
            colors_layout,
            "Gradient",
            gradient_check,
            "Fill the area under each 1D line with a gradient.",
        )

        gradient_kwargs_text, gradient_kwargs_mixed = editor.batch_text(
            operation, lambda target: target.gradient_kw, _format_dict
        )
        gradient_kwargs_edit = editor.line_edit(
            gradient_kwargs_text, parent=colors_page
        )
        editor.apply_mixed_line_edit(gradient_kwargs_edit, gradient_kwargs_mixed)
        gradient_kwargs_edit.setObjectName("figureComposerGradientKwEdit")
        editor.connect_line_edit_finished(
            gradient_kwargs_edit,
            lambda text: editor.request_update(gradient_kw=_dict_from_text(text)),
        )
        editor.add_form_row(
            colors_layout,
            "Kwargs",
            gradient_kwargs_edit,
            "Dict literal or keyword arguments forwarded as gradient_kw.",
        )

        editor.add_form_section(
            colors_layout,
            "Panel overrides",
            object_name="figureComposerPlotSlicesStylePanelOverridesSection",
        )
        panel_styles_mixed = editor.batch_is_mixed(
            operation, lambda target: target.panel_styles_enabled
        )
        panel_styles_check = editor.check_box(
            operation.panel_styles_enabled,
            lambda checked: _update_current_panel_styles_enabled(editor, checked),
            parent=colors_page,
            mixed=panel_styles_mixed,
        )
        panel_styles_check.setObjectName("figureComposerPlotSlicesPanelStylesCheck")
        panel_styles_check.setText("Use panel-specific line styles")
        editor.add_form_row(
            colors_layout,
            "Per-panel",
            panel_styles_check,
            "Override line color, style, marker, or kwargs for individual panels.",
        )
        if operation.panel_styles_enabled:
            panel_editor = _panel_style_editor._PanelLineStyleEditorWidget(
                operation,
                _plot_slices_panel_keys(
                    editor.context, editor.source_display_name, operation
                ),
                editor.connect_signal,
                colors_page,
            )
            panel_editor.setObjectName("figureComposerPlotSlicesPanelLineStyleEditor")
            editor.mark_control(panel_editor)
            editor.connect_value_signal(
                panel_editor,
                panel_editor.sigPanelStylesChanged,
                lambda styles: styles,
                lambda styles: _update_current_panel_styles(editor, styles),
            )
            editor.add_form_row(
                colors_layout,
                "Styles",
                panel_editor,
                "Select panels and set optional line-style overrides.",
            )
    elif is_image_plot:
        editor.add_form_section(
            colors_layout,
            "Image color",
            object_name="figureComposerPlotSlicesColorsImageColorSection",
        )
        cmap_base, cmap_reversed = _plot_slices_cmap_base_and_reverse(editor, operation)
        cmap_widget = QtWidgets.QWidget(colors_page)
        cmap_layout = QtWidgets.QHBoxLayout(cmap_widget)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.setSpacing(4)
        cmap_mixed = editor.batch_is_mixed(
            operation, lambda target: _cmap_base_and_reverse(target.cmap)[0]
        )
        reverse_mixed = editor.batch_is_mixed(
            operation, lambda target: _cmap_base_and_reverse(target.cmap)[1]
        )
        cmap_combo = erlab.interactive.colors.ColorMapComboBox(cmap_widget)
        editor.mark_control(cmap_combo)
        cmap_combo.setObjectName("figureComposerCmapCombo")
        cmap_combo.setToolTip("Colormap passed to plot_slices.")
        cmap_combo.default_cmap = None if cmap_mixed else cmap_base
        cmap_combo.ensure_populated()
        with QtCore.QSignalBlocker(cmap_combo):
            if cmap_mixed:
                editor.set_combo_mixed_placeholder(cmap_combo)
            else:
                cmap_combo.setCurrentText(cmap_base)
        cmap_reverse_check = editor.check_box(
            cmap_reversed,
            lambda checked: _update_current_cmap(editor, reverse=checked),
            parent=cmap_widget,
            mixed=reverse_mixed,
        )
        cmap_reverse_check.setText("Reverse")
        cmap_reverse_check.setObjectName("figureComposerCmapReverseCheck")
        cmap_reverse_check.setToolTip("Append _r to the selected Matplotlib colormap.")

        editor.connect_signal(
            cmap_combo,
            cmap_combo.activated,
            lambda _index, combo=cmap_combo: (
                None
                if editor.mixed_combo_text(combo.currentText())
                else _update_current_cmap(editor, base=combo.current_matplotlib_name())
            ),
        )
        cmap_combo.blockSignals(False)
        cmap_layout.addWidget(cmap_combo, 1)
        cmap_layout.addWidget(cmap_reverse_check)
        editor.add_form_row(
            colors_layout,
            "Colormap",
            cmap_widget,
            "Colormap and reverse-colormap controls for image panels.",
        )

        norm_combo = editor.combo(
            _norm_combo_choices(operation.norm_name),
            editor.batch_combo_text(
                operation,
                lambda target: target.norm_name,
                _norm_combo_text,
            ),
            lambda text: _update_current_norm_name(
                editor, _norm_name_from_combo_text(text)
            ),
            parent=colors_page,
            mixed=editor.batch_is_mixed(
                operation, lambda target: _norm_combo_text(target.norm_name)
            ),
        )
        norm_combo.setObjectName("figureComposerNormCombo")
        norm_combo.setToolTip("Color normalization used for image plot_slices panels.")
        editor.add_form_row(colors_layout, "Norm", norm_combo, norm_combo.toolTip())

        norm_fields = _norm_kwarg_fields(operation.norm_name)
        if "gamma" in norm_fields:
            gamma_mixed = editor.batch_is_mixed(
                operation, lambda target: _norm_gamma_value(target)
            )
            gamma_widget = erlab.interactive.colors.ColorMapGammaWidget(
                colors_page,
                value=_norm_gamma_value(operation),
                spin_cls=erlab.interactive.utils.BetterSpinBox,
            )
            gamma_widget.setObjectName("figureComposerGammaWidget")
            gamma_widget.setToolTip("Gamma value for the selected normalization.")
            editor.connect_signal(
                gamma_widget,
                gamma_widget.valueChanged,
                lambda value: _update_current_norm_gamma(editor, value),
            )
            gamma_row_widget = editor.mixed_value_widget(
                gamma_widget,
                mixed=gamma_mixed,
                parent=colors_page,
            )
            editor.add_form_row(
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
            _plot_slices_color_limit_placeholders(editor, operation)
            if "vmin" in norm_fields or "vmax" in norm_fields
            else {}
        )
        norm_number_widgets: dict[str, tuple[str, QtWidgets.QWidget, str]] = {}
        for attr in ("vmin", "vmax", "vcenter", "halfrange"):
            if attr not in norm_fields:
                continue
            label, _value, tooltip = norm_number_fields[attr]
            text, mixed = editor.batch_text(
                operation,
                _operation_field_getter(attr),
                lambda value: "" if value is None else str(value),
            )
            edit = editor.line_edit(text, parent=colors_page)
            editor.apply_mixed_line_edit(edit, mixed)
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
            editor.connect_line_edit_finished(
                edit,
                _norm_number_update_callback(editor, attr),
            )
            norm_number_widgets[attr] = (label, edit, tooltip)

        if "vmin" in norm_number_widgets and "vmax" in norm_number_widgets:
            editor.add_compound_form_row(
                colors_layout,
                "Color limits",
                (
                    norm_number_widgets.pop("vmin"),
                    norm_number_widgets.pop("vmax"),
                ),
                "Lower and upper color-normalization bounds.",
            )
        if "vcenter" in norm_number_widgets and "halfrange" in norm_number_widgets:
            editor.add_compound_form_row(
                colors_layout,
                "Center/range",
                (
                    norm_number_widgets.pop("vcenter"),
                    norm_number_widgets.pop("halfrange"),
                ),
                "Center and half-range for centered color normalization.",
            )
        for label, edit, tooltip in norm_number_widgets.values():
            editor.add_form_row(colors_layout, label, edit, tooltip)

        if "clip" in norm_fields:
            clip_mixed = editor.batch_is_mixed(
                operation, lambda target: target.norm_clip
            )
            clip_combo = editor.combo(
                ["default", "False", "True"],
                None if clip_mixed else _norm_clip_text(operation.norm_clip),
                lambda text: editor.request_update(
                    norm_clip=_norm_clip_from_text(text)
                ),
                parent=colors_page,
                mixed=clip_mixed,
            )
            clip_combo.setObjectName("figureComposerNormClipCombo")
            editor.add_form_row(
                colors_layout,
                "Clip",
                clip_combo,
                "clip argument for the selected normalization object.",
            )

        norm_kwargs_text, norm_kwargs_mixed = editor.batch_text(
            operation, lambda target: target.norm_kwargs, _format_dict
        )
        norm_kwargs_edit = editor.line_edit(norm_kwargs_text, parent=colors_page)
        editor.apply_mixed_line_edit(norm_kwargs_edit, norm_kwargs_mixed)
        norm_kwargs_edit.setObjectName("figureComposerNormKwargsEdit")
        editor.connect_line_edit_finished(
            norm_kwargs_edit,
            lambda text: _update_current_norm_kwargs(editor, text),
        )
        editor.add_form_row(
            colors_layout,
            "Norm kwargs",
            norm_kwargs_edit,
            "Extra dict literal or keyword arguments for the norm constructor.",
        )

        same_limits_mixed = editor.batch_is_mixed(
            operation, lambda target: target.same_limits
        )
        same_limits_combo = editor.combo(
            ["False", "True", "row", "col", "all"],
            None if same_limits_mixed else str(operation.same_limits),
            lambda text: editor.request_update(same_limits=_bool_or_text(text)),
            parent=colors_page,
            mixed=same_limits_mixed,
        )
        same_limits_combo.setObjectName("figureComposerSameLimitsCombo")
        editor.add_form_row(
            colors_layout,
            "Match limits",
            same_limits_combo,
            "Control plot_slices same_limits for image color scaling.",
        )

        editor.add_form_section(
            colors_layout,
            "Colorbar",
            object_name="figureComposerPlotSlicesColorsColorbarSection",
        )
        colorbar_mixed = editor.batch_is_mixed(
            operation, lambda target: target.colorbar
        )
        colorbar_combo = editor.combo(
            ["none", "right", "rightspan", "all"],
            None if colorbar_mixed else operation.colorbar,
            lambda text: editor.request_update(colorbar=text),
            parent=colors_page,
            mixed=colorbar_mixed,
        )
        editor.add_form_row(
            colors_layout,
            "Placement",
            colorbar_combo,
            "Where plot_slices should place colorbars for image panels.",
        )
        colorbar_kwargs_text, colorbar_kwargs_mixed = editor.batch_text(
            operation, lambda target: target.colorbar_kw, _format_dict
        )
        colorbar_kwargs_edit = editor.line_edit(
            colorbar_kwargs_text,
            parent=colors_page,
        )
        editor.apply_mixed_line_edit(colorbar_kwargs_edit, colorbar_kwargs_mixed)
        colorbar_kwargs_edit.setObjectName("figureComposerColorbarKwEdit")
        editor.connect_line_edit_finished(
            colorbar_kwargs_edit,
            lambda text: editor.request_update(colorbar_kw=_dict_from_text(text)),
        )
        editor.add_form_row(
            colors_layout,
            "Kwargs",
            colorbar_kwargs_edit,
            "Dict literal or keyword arguments forwarded as colorbar_kw.",
        )

        editor.add_form_section(
            colors_layout,
            "Panel overrides",
            object_name="figureComposerPlotSlicesColorsPanelOverridesSection",
        )
        panel_styles_mixed = editor.batch_is_mixed(
            operation, lambda target: target.panel_styles_enabled
        )
        panel_styles_check = editor.check_box(
            operation.panel_styles_enabled,
            lambda checked: _update_current_panel_styles_enabled(editor, checked),
            parent=colors_page,
            mixed=panel_styles_mixed,
        )
        panel_styles_check.setObjectName("figureComposerPlotSlicesPanelStylesCheck")
        panel_styles_check.setText("Use panel-specific styles")
        editor.add_form_row(
            colors_layout,
            "Per-panel",
            panel_styles_check,
            "Override colormaps and normalization for individual image panels.",
        )
        if operation.panel_styles_enabled:
            panel_editor = _panel_style_editor._PanelStyleEditorWidget(
                operation,
                _plot_slices_panel_keys(
                    editor.context, editor.source_display_name, operation
                ),
                editor.connect_signal,
                str(editor.styled_rcparams_value("image.cmap")),
                colors_page,
            )
            panel_editor.setObjectName("figureComposerPlotSlicesPanelStyleEditor")
            editor.mark_control(panel_editor)
            editor.connect_value_signal(
                panel_editor,
                panel_editor.sigPanelStylesChanged,
                lambda styles: styles,
                lambda styles: _update_current_panel_styles(editor, styles),
            )
            editor.add_form_row(
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
        editor.add_form_row(
            colors_layout,
            "Colors",
            mixed_label,
            "Color controls are hidden for mixed image/line plot_slices selection.",
        )

    extra_text, extra_mixed = editor.batch_text(
        operation,
        lambda target: _effective_extra_kwargs(editor.context, target),
        _format_dict,
    )
    extra_edit = editor.line_edit(extra_text, parent=advanced_page)
    editor.apply_mixed_line_edit(extra_edit, extra_mixed)
    extra_edit.setObjectName("figureComposerExtraKwEdit")
    editor.connect_line_edit_finished(
        extra_edit,
        lambda text: _update_current_extra_kwargs(editor, text),
    )
    editor.add_form_row(
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
    editor: FigureOperationEditor, attr: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        editor.request_update(**{attr: _plot_limit_from_text(text)})

    return update


def _norm_number_update_callback(
    editor: FigureOperationEditor, attr: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        editor.request_update(**{attr: _optional_number_or_text(attr, text)})

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
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    reader: Callable[[Sequence[matplotlib.axes.Axes]], typing.Any],
) -> typing.Any:
    return editor.rendered_value(operation, reader)


def _plot_slices_limit_placeholder(
    editor: FigureOperationEditor, operation: FigureOperationState, attr: str
) -> str:
    if attr not in {"xlim", "ylim"} or getattr(operation, attr) is not None:
        return ""
    limits = _plot_slices_rendered_value(
        editor,
        operation,
        lambda axes: axes[0].get_xlim() if attr == "xlim" else axes[0].get_ylim(),
    )
    if limits is None:
        return ""
    limit_pair = _rendered_float_pair(limits)
    return "" if limit_pair is None else _format_pair(limit_pair)


def _plot_slices_color_limit_placeholders(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> dict[str, str]:
    if operation.vmin is not None and operation.vmax is not None:
        return {}
    clim = _plot_slices_rendered_value(editor, operation, _first_mappable_clim)
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


def _update_current_norm_name(editor: FigureOperationEditor, name: str) -> None:
    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {"norm_name": name}
        if operation.norm_gamma is None and operation.gamma is not None:
            updates["norm_gamma"] = operation.gamma
            updates["gamma"] = None
        return operation.model_copy(update=updates)

    editor.request_transform(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_norm_gamma(editor: FigureOperationEditor, value: float) -> None:
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

    editor.request_transform(update_operation, defer_render=True)


def _update_current_norm_kwargs(editor: FigureOperationEditor, text: str) -> None:
    updates = _norm_updates_from_kwargs(_dict_from_text(text))

    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        return operation.model_copy(update=updates)

    editor.request_transform(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_slice_kwargs(editor: FigureOperationEditor, text: str) -> None:
    slice_kwargs = _dict_from_text(text, allow_slice=True)

    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates = _selection_updates_from_kwargs(
            editor.context,
            operation,
            slice_kwargs,
            _effective_extra_kwargs(editor.context, operation),
        )
        return operation.model_copy(update=updates)

    editor.request_transform(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_extra_kwargs(editor: FigureOperationEditor, text: str) -> None:
    extra_kwargs = _dict_from_text(text, allow_slice=True)

    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates = _selection_updates_from_kwargs(
            editor.context,
            operation,
            _effective_slice_kwargs(editor.context, operation),
            extra_kwargs,
        )
        return operation.model_copy(update=updates)

    editor.request_transform(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_cmap(
    editor: FigureOperationEditor,
    *,
    base: str | None = None,
    reverse: bool | None = None,
) -> None:
    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        operation_base, operation_reverse = _plot_slices_cmap_base_and_reverse(
            editor, operation
        )
        next_base = operation_base if base is None else base
        next_reverse = operation_reverse if reverse is None else reverse
        return operation.model_copy(
            update={"cmap": _cmap_with_reverse(next_base, next_reverse)}
        )

    editor.request_transform(update_operation, defer_render=True)


def _update_current_panel_styles_enabled(
    editor: FigureOperationEditor, enabled: bool
) -> None:
    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {"panel_styles_enabled": enabled}
        if not enabled:
            updates["panel_styles"] = ()
        return operation.model_copy(update=updates)

    editor.request_transform(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _update_current_panel_styles(
    editor: FigureOperationEditor,
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

    editor.request_transform(update_operation)


def _plot_source_check_state(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    source_name: str,
) -> QtCore.Qt.CheckState:
    editable = editor.editable_operations()
    if len(editable) <= 1:
        return (
            QtCore.Qt.CheckState.Checked
            if source_name in _plot_slices_selection_sources(operation)
            else QtCore.Qt.CheckState.Unchecked
        )
    selected_count = sum(
        source_name in _plot_slices_selection_sources(target)
        for _index, target in editable
    )
    if selected_count == 0:
        return QtCore.Qt.CheckState.Unchecked
    if selected_count == len(editable):
        return QtCore.Qt.CheckState.Checked
    return QtCore.Qt.CheckState.PartiallyChecked


def _plot_source_check_changed(
    editor: FigureOperationEditor,
    source_name: str,
    check: QtWidgets.QCheckBox,
    row_order: tuple[str, ...],
) -> None:
    state = check.checkState()
    if state == QtCore.Qt.CheckState.PartiallyChecked:
        return
    checked = state == QtCore.Qt.CheckState.Checked

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        if checked:
            target_sources = _plot_slices_selection_sources(target)
            if source_name in target_sources:
                return target
            source_set = {*target_sources, source_name}
            ordered_sources = tuple(
                source for source in row_order if source in source_set
            )
            missing_sources = tuple(
                source for source in target_sources if source not in row_order
            )
            return _plot_slices_operation_with_sources(
                target, (*ordered_sources, *missing_sources)
            )
        next_sources = tuple(
            source
            for source in _plot_slices_selection_sources(target)
            if source != source_name
        )
        return _plot_slices_operation_with_sources(target, next_sources)

    editor.request_transform(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _plot_source_move(
    editor: FigureOperationEditor,
    source_name: str,
    offset: typing.Literal[-1, 1],
) -> None:
    available_sources = set(editor.context.source_names())

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        ordered_sources = [
            source
            for source in _plot_slices_selection_sources(target)
            if source in available_sources
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
            source
            for source in _plot_slices_selection_sources(target)
            if source not in available_sources
        )
        return _plot_slices_operation_with_sources(
            target, (*ordered_sources, *missing_sources)
        )

    editor.request_transform(
        update_operation,
        rebuild_editor=True,
        defer_editor_rebuild=True,
    )


def _plot_source_order_matches(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> bool:
    editable = editor.editable_operations()
    if len(editable) <= 1:
        return True
    available_sources = set(editor.context.source_names())
    expected = tuple(
        source
        for source in _plot_slices_selection_sources(operation)
        if source in available_sources
    )
    return all(
        tuple(
            source
            for source in _plot_slices_selection_sources(target)
            if source in available_sources
        )
        == expected
        for _index, target in editable
    )


def _plot_source_row_names(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[str, ...]:
    source_names = editor.context.source_names()
    selected_sources = tuple(
        source
        for source in _plot_slices_selection_sources(operation)
        if source in source_names
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
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    selector: QtWidgets.QWidget,
    source_name: str,
    index: int,
    selected_sources: tuple[str, ...],
    row_order: tuple[str, ...],
    order_controls_enabled: bool,
) -> tuple[QtWidgets.QCheckBox, QtWidgets.QToolButton, QtWidgets.QToolButton]:
    check = QtWidgets.QCheckBox(editor.source_display_name(source_name), selector)
    check.setObjectName(f"figureComposerPlotSlicesSourceCheck_{index}")
    check.setProperty("figure_source_name", source_name)
    check.setToolTip(
        "Include this DataArray in the maps passed to plot_slices.\n"
        + editor.source_tooltip(source_name)
    )
    state = _plot_source_check_state(editor, operation, source_name)
    check.setTristate(state == QtCore.Qt.CheckState.PartiallyChecked)
    check.setCheckState(state)
    editor.connect_signal(
        check,
        check.stateChanged,
        lambda _state, source_name=source_name, check=check, row_order=row_order: (
            _plot_source_check_changed(editor, source_name, check, row_order)
        ),
    )

    source_selected = source_name in selected_sources
    selected_index = selected_sources.index(source_name) if source_selected else -1
    up_button = _plot_source_move_button(
        editor,
        selector,
        source_name,
        "up",
        order_controls_enabled and selected_index > 0,
        "Move this input earlier in the maps argument.",
        lambda: _plot_source_move(editor, source_name, -1),
    )
    down_button = _plot_source_move_button(
        editor,
        selector,
        source_name,
        "down",
        order_controls_enabled
        and source_selected
        and selected_index < len(selected_sources) - 1,
        "Move this input later in the maps argument.",
        lambda: _plot_source_move(editor, source_name, 1),
    )
    if source_selected:
        return check, up_button, down_button
    up_button.setVisible(False)
    down_button.setVisible(False)
    return check, up_button, down_button


def _plot_source_move_button(
    editor: FigureOperationEditor,
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
    editor.connect_signal(
        button,
        button.clicked,
        lambda _checked=False: clicked(),
    )
    return button


def _build_source_editor(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> None:
    selector = QtWidgets.QWidget(editor.source_controls)
    selector.setObjectName("figureComposerPlotSlicesSourceSelector")
    layout = QtWidgets.QGridLayout(selector)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(4)
    layout.setVerticalSpacing(2)
    row_names = _plot_source_row_names(editor, operation)
    selected_sources = tuple(
        source
        for source in _plot_slices_selection_sources(operation)
        if source in editor.context.source_names()
    )
    order_controls_enabled = _plot_source_order_matches(editor, operation)
    if row_names:
        for index, source_name in enumerate(row_names):
            check, up_button, down_button = _build_plot_source_row(
                editor,
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
    editor.add_source_row(
        "Inputs",
        selector,
        selector.toolTip(),
    )
