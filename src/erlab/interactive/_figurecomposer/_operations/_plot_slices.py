"""Plot Slices operation editor, renderer, and code generation."""

from __future__ import annotations

import math
import typing

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from qtpy import QtGui, QtWidgets

import erlab
import erlab.plotting as eplt
from erlab.interactive._figurecomposer import _rendering
from erlab.interactive._figurecomposer._code import _axes_code, _selection_code
from erlab.interactive._figurecomposer._defaults import (
    _figure_draw_context,
    _figure_style_context,
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

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import matplotlib.axes
    import xarray as xr

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _build_plot_slices_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[tuple[str, str, QtWidgets.QWidget]]:
    shape = _plot_slices_shape(tool, operation)
    is_line_plot = shape.plot_ndim == 1
    is_image_plot = shape.plot_ndim != 1
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
    dim_combo = tool._combo(
        ["", *dims],
        operation.slice_dim or "",
        lambda text: tool._update_current_operation_rebuild(slice_dim=text or None),
    )
    tool._add_form_row(
        basic_layout,
        "Cut dimension",
        dim_combo,
        "Data dimension passed as the slice keyword to plot_slices.",
    )

    values_edit = tool._line_edit(_format_tuple(operation.slice_values))
    values_edit.editingFinished.connect(
        lambda edit=values_edit: tool._update_current_operation_rebuild(
            slice_values=_float_tuple_from_text(edit.text())
        )
    )
    tool._add_form_row(
        basic_layout,
        "Cut values",
        values_edit,
        "Comma-separated coordinate values to slice along the cut dimension.",
    )

    width_edit = tool._line_edit(
        "" if operation.slice_width is None else f"{operation.slice_width:g}"
    )
    width_edit.editingFinished.connect(
        lambda edit=width_edit: tool._update_current_operation_rebuild(
            slice_width=float(edit.text()) if edit.text().strip() else None
        )
    )
    tool._add_form_row(
        basic_layout,
        "Integration width",
        width_edit,
        "Optional qsel width around each cut value before plotting.",
    )

    order_combo = tool._combo(
        ["C", "F"],
        operation.order,
        lambda text: tool._update_current_operation_rebuild(order=text),
        parent=cuts_page,
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
    transpose_check = tool._check_box(
        operation.transpose,
        lambda checked: tool._update_current_operation_rebuild(transpose=checked),
    )
    transpose_check.setText("Transpose")
    transpose_check.setToolTip("Swap the plotted x/y orientation.")
    crop_check = tool._check_box(
        operation.crop,
        lambda checked: tool._update_current_operation(crop=checked),
    )
    crop_check.setText("Crop")
    crop_check.setToolTip("Crop each slice to explicit x/y limits before plotting.")
    options_layout.addWidget(transpose_check)
    options_layout.addWidget(crop_check)
    options_layout.addStretch(1)
    tool._add_form_row(
        basic_layout,
        "Options",
        options_widget,
        "Common plot_slices boolean options for this step.",
    )

    for label, attr, text in (
        ("xlim", "xlim", _format_plot_limit(operation.xlim)),
        ("ylim", "ylim", _format_plot_limit(operation.ylim)),
    ):
        edit = tool._line_edit(text)
        placeholder = _plot_slices_limit_placeholder(tool, operation, attr)
        if placeholder:
            edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(
            lambda attr=attr, edit=edit: tool._update_current_operation(
                **{attr: _plot_limit_from_text(edit.text())}
            )
        )
        tool._add_form_row(
            limits_layout,
            label,
            edit,
            f"Optional {label}: one number for symmetric limits, "
            "or two comma-separated numbers for lower and upper limits.",
        )

    if is_image_plot:
        colorbar_combo = tool._combo(
            ["none", "right", "rightspan", "all"],
            operation.colorbar,
            lambda text: tool._update_current_operation(colorbar=text),
            parent=colors_page,
        )
        tool._add_form_row(
            colors_layout,
            "Colorbar",
            colorbar_combo,
            "Where plot_slices should place colorbars for image panels.",
        )
        colorbar_kwargs_edit = tool._line_edit(
            _format_dict(operation.colorbar_kw),
            parent=colors_page,
        )
        colorbar_kwargs_edit.setObjectName("figureComposerColorbarKwEdit")
        colorbar_kwargs_edit.editingFinished.connect(
            lambda edit=colorbar_kwargs_edit: tool._update_current_operation(
                colorbar_kw=_dict_from_text(edit.text())
            )
        )
        tool._add_form_row(
            colors_layout,
            "Colorbar kwargs",
            colorbar_kwargs_edit,
            "Dict literal or keyword arguments forwarded as colorbar_kw.",
        )

        same_limits_combo = tool._combo(
            ["False", "True", "row", "col", "all"],
            str(operation.same_limits),
            lambda text: tool._update_current_operation(
                same_limits=_bool_or_text(text)
            ),
            parent=style_page,
        )
        same_limits_combo.setObjectName("figureComposerSameLimitsCombo")
        tool._add_form_row(
            style_layout,
            "Match color limits",
            same_limits_combo,
            "Control plot_slices same_limits for image color scaling.",
        )

    axis_combo = tool._combo(
        ["auto", "on", "off", "equal", "scaled", "tight", "image", "square"],
        operation.axis,
        lambda text: tool._update_current_operation(axis=text),
        parent=style_page,
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
    show_labels_check = tool._check_box(
        operation.show_all_labels,
        lambda checked: tool._update_current_operation(show_all_labels=checked),
    )
    show_labels_check.setText("All labels")
    show_labels_check.setToolTip("Ask plot_slices to show labels on every axis.")
    annotate_check = tool._check_box(
        operation.annotate,
        lambda checked: tool._update_current_operation(annotate=checked),
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

    annotate_kwargs_edit = tool._line_edit(
        _format_dict(operation.annotate_kw),
        parent=style_page,
    )
    annotate_kwargs_edit.setObjectName("figureComposerAnnotateKwEdit")
    annotate_kwargs_edit.editingFinished.connect(
        lambda edit=annotate_kwargs_edit: tool._update_current_operation(
            annotate_kw=_dict_from_text(edit.text())
        )
    )
    tool._add_form_row(
        style_layout,
        "Annotation kwargs",
        annotate_kwargs_edit,
        "Dict literal or keyword arguments forwarded as annotate_kw.",
    )

    if is_line_plot:
        line_color_edit = tool._line_edit(operation.cmap or "", parent=colors_page)
        line_color_edit.setObjectName("figureComposerLineColorEdit")
        line_color_edit.editingFinished.connect(
            lambda edit=line_color_edit: tool._update_current_operation_in_place(
                cmap=edit.text().strip() or None
            )
        )
        tool._add_form_row(
            colors_layout,
            "Line color",
            line_color_edit,
            "Matplotlib color passed through plot_slices cmap for 1D panels.",
        )

        gradient_check = tool._check_box(
            operation.gradient,
            lambda checked: tool._update_current_operation(gradient=checked),
            parent=colors_page,
        )
        gradient_check.setObjectName("figureComposerGradientCheck")
        gradient_check.setText("Fill under line")
        tool._add_form_row(
            colors_layout,
            "Gradient",
            gradient_check,
            "Fill the area under each 1D line with a gradient.",
        )

        gradient_kwargs_edit = tool._line_edit(
            _format_dict(operation.gradient_kw), parent=colors_page
        )
        gradient_kwargs_edit.setObjectName("figureComposerGradientKwEdit")
        gradient_kwargs_edit.editingFinished.connect(
            lambda edit=gradient_kwargs_edit: tool._update_current_operation(
                gradient_kw=_dict_from_text(edit.text())
            )
        )
        tool._add_form_row(
            colors_layout,
            "Gradient kwargs",
            gradient_kwargs_edit,
            "Dict literal or keyword arguments forwarded as gradient_kw.",
        )
    else:
        cmap_base, cmap_reversed = _cmap_base_and_reverse(operation.cmap)
        cmap_widget = QtWidgets.QWidget(colors_page)
        cmap_layout = QtWidgets.QHBoxLayout(cmap_widget)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.setSpacing(4)
        cmap_combo = erlab.interactive.colors.ColorMapComboBox(cmap_widget)
        cmap_combo.setObjectName("figureComposerCmapCombo")
        cmap_combo.setToolTip("Colormap passed to plot_slices.")
        cmap_combo.setCurrentText(cmap_base)
        cmap_reverse_check = QtWidgets.QCheckBox("Reverse", cmap_widget)
        cmap_reverse_check.setObjectName("figureComposerCmapReverseCheck")
        cmap_reverse_check.setChecked(cmap_reversed)
        cmap_reverse_check.setToolTip("Append _r to the selected Matplotlib colormap.")

        def update_cmap_from_controls(*_args: object) -> None:
            _update_current_cmap(
                tool, cmap_combo.currentText(), cmap_reverse_check.isChecked()
            )

        cmap_combo.currentTextChanged.connect(update_cmap_from_controls)
        cmap_reverse_check.toggled.connect(update_cmap_from_controls)
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
            _norm_combo_text(operation.norm_name),
            lambda text: _update_current_norm_name(
                tool, _norm_name_from_combo_text(text)
            ),
            parent=colors_page,
        )
        norm_combo.setObjectName("figureComposerNormCombo")
        norm_combo.setToolTip("Color normalization used for image plot_slices panels.")
        tool._add_form_row(colors_layout, "Norm", norm_combo, norm_combo.toolTip())

        norm_fields = _norm_kwarg_fields(operation.norm_name)
        if "gamma" in norm_fields:
            gamma_widget = erlab.interactive.colors.ColorMapGammaWidget(
                colors_page,
                value=_norm_gamma_value(operation),
                spin_cls=erlab.interactive.utils.BetterSpinBox,
            )
            gamma_widget.setObjectName("figureComposerGammaWidget")
            gamma_widget.setToolTip("Gamma value for the selected normalization.")
            gamma_widget.valueChanged.connect(
                lambda value: _update_current_norm_gamma(tool, value)
            )
            tool._add_form_row(
                colors_layout, "Gamma", gamma_widget, gamma_widget.toolTip()
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
            label, value, tooltip = norm_number_fields[attr]
            edit = tool._line_edit("" if value is None else str(value))
            edit.setObjectName(f"figureComposer{attr[0].upper()}{attr[1:]}NormEdit")
            placeholder = _norm_field_placeholder(operation, attr)
            if placeholder:
                edit.setPlaceholderText(placeholder)
            edit.editingFinished.connect(
                lambda attr=attr, edit=edit: tool._update_current_operation(
                    **{attr: _optional_number_or_text(attr, edit.text())}
                )
            )
            tool._add_form_row(colors_layout, label, edit, tooltip)

        if "clip" in norm_fields:
            clip_combo = tool._combo(
                ["default", "False", "True"],
                _norm_clip_text(operation.norm_clip),
                lambda text: tool._update_current_operation(
                    norm_clip=_norm_clip_from_text(text)
                ),
                parent=colors_page,
            )
            clip_combo.setObjectName("figureComposerNormClipCombo")
            tool._add_form_row(
                colors_layout,
                "Clip",
                clip_combo,
                "clip argument for the selected normalization object.",
            )

        norm_kwargs_edit = tool._line_edit(_format_dict(operation.norm_kwargs))
        norm_kwargs_edit.setObjectName("figureComposerNormKwargsEdit")
        norm_kwargs_edit.editingFinished.connect(
            lambda edit=norm_kwargs_edit: _update_current_norm_kwargs(tool, edit.text())
        )
        tool._add_form_row(
            colors_layout,
            "Norm kwargs",
            norm_kwargs_edit,
            "Extra dict literal or keyword arguments for the norm constructor.",
        )

    extra_edit = tool._line_edit(_format_dict(operation.extra_kwargs))
    extra_edit.setObjectName("figureComposerExtraKwEdit")
    extra_edit.editingFinished.connect(
        lambda edit=extra_edit: tool._update_current_operation_rebuild(
            extra_kwargs=_dict_from_text(edit.text())
        )
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
    current = tool._current_operation()
    if current is None:
        return
    index, operation = current
    updates: dict[str, typing.Any] = {"norm_name": name}
    if operation.norm_gamma is None and operation.gamma is not None:
        updates["norm_gamma"] = operation.gamma
        updates["gamma"] = None
    tool._replace_operation(
        index,
        operation.model_copy(update=updates),
        rebuild_editor=True,
    )


def _update_current_norm_gamma(tool: FigureComposerTool, value: float) -> None:
    current = tool._current_operation()
    if current is None:
        return
    tool._update_current_operation_in_place(
        norm_name=current[1].norm_name or _POWER_NORM_NAME,
        norm_gamma=value,
        gamma=None,
    )


def _update_current_norm_kwargs(tool: FigureComposerTool, text: str) -> None:
    current = tool._current_operation()
    if current is None:
        return
    index, operation = current
    updates = _norm_updates_from_kwargs(_dict_from_text(text))
    tool._replace_operation(
        index,
        operation.model_copy(update=updates),
        rebuild_editor=True,
    )


def _update_current_cmap(tool: FigureComposerTool, base: str, reverse: bool) -> None:
    tool._update_current_operation_in_place(cmap=_cmap_with_reverse(base, reverse))


def _plot_slices_shape(
    tool: FigureComposerTool, operation: FigureOperationState
) -> _PlotSlicesShape:
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
    for key, value in operation.extra_kwargs.items():
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
    selected_axes = operation.axes.valid_axes(tool._recipe.setup)
    if operation.axes.expression:
        axes_text = f"Targets: {operation.axes.expression}"
    else:
        axes_text = (
            f"Targets: {len(selected_axes)} selected for {panel_count} panel"
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
    kwargs: dict[str, typing.Any] = {}
    shape = _plot_slices_shape(tool, operation)
    is_line_plot = shape.plot_ndim == 1
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
    if operation.cmap:
        kwargs["cmap"] = operation.cmap
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
    kwargs.update(dict(operation.extra_kwargs))
    return kwargs


def _render_plot_slices(
    tool: FigureComposerTool, operation: FigureOperationState, axs: np.ndarray
) -> None:
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
    "colors": (
        "Set image color scaling or line color options for this plot_slices step."
    ),
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
    return (
        bool(operation.axes.invalid_axes(tool._recipe.setup))
        and not operation.axes.expression
    )


def _source_names(operation: FigureOperationState) -> tuple[str, ...]:
    return operation.sources


def _build_source_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> None:
    sources_edit = tool._line_edit(
        ", ".join(operation.sources), parent=tool.step_source_controls
    )
    sources_edit.setToolTip(
        "Comma-separated source names passed as maps to plot_slices."
    )
    sources_edit.editingFinished.connect(
        lambda edit=sources_edit: tool._update_current_operation_rebuild(
            sources=tuple(
                name.strip() for name in edit.text().split(",") if name.strip()
            )
        )
    )
    tool._add_form_row(
        tool.step_source_controls_layout,
        "Plot data",
        sources_edit,
        sources_edit.toolTip(),
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
    match key:
        case "sources":
            return ", ".join(operation.sources) or "none"
        case "axes":
            return tool._axes_target_text(operation.axes)
        case "cuts":
            if operation.slice_dim and operation.slice_values:
                return f"{operation.slice_dim}, {len(operation.slice_values)}"
            return "none"
        case "limits":
            labels = [
                label
                for label, value in (("x", operation.xlim), ("y", operation.ylim))
                if value is not None
            ]
            return ", ".join(labels) if labels else "auto"
        case "colors":
            shape = _plot_slices_shape(tool, operation)
            if shape.plot_ndim == 1:
                return operation.cmap or "line"
            return operation.cmap or "default"
        case "style":
            return operation.axis
        case "advanced":
            return "set" if operation.extra_kwargs else "optional"
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
