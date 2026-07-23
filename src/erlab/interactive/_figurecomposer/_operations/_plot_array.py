"""Plot Array operation editor, renderer, and code generation."""

from __future__ import annotations

import dataclasses
import typing

from qtpy import QtCore, QtWidgets

import erlab
import erlab.plotting as eplt
from erlab.interactive._figurecomposer._code import _axes_code, _maybe_squeeze_drop_code
from erlab.interactive._figurecomposer._model._gridspec import _gridspec_valid_axes_ids
from erlab.interactive._figurecomposer._model._sources import (
    _public_source_data,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._norms import (
    _MATPLOTLIB_NORM_NAMES,
    _cmap_base_and_reverse,
    _cmap_with_reverse,
    _effective_norm_name,
    _matplotlib_cmap_name,
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
    _always_render_cache_safe,
)
from erlab.interactive._figurecomposer._rendering import _axes_from_selection
from erlab.interactive._figurecomposer._text import (
    _code_kwargs,
    _dict_from_text,
    _format_dict,
    _format_plot_limit,
    _plot_limit_from_text,
    _RawCode,
)
from erlab.interactive._figurecomposer._ui._editor_controls import (
    MIXED_VALUE,
    MIXED_VALUES_TEXT,
    ComboBoxDataControlAdapter,
)
from erlab.interactive._figurecomposer._ui._operation_editor import StepSection
from erlab.plotting.general import _prepare_plot_array_data

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import matplotlib.axes
    import xarray as xr

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._render_context import FigureRenderContext
    from erlab.interactive._figurecomposer._tool import FigureComposerTool
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


def _create_plot_array_operation(tool: FigureComposerTool) -> FigureOperationState:
    source_names = tool._document.source_names()
    first_source = (
        source_names[0] if source_names else tool._document.recipe.primary_source
    )
    return FigureOperationState.plot_array(
        label="plot_array",
        source=first_source,
        axes=tool._selected_axes_state(),
    )


def _primary_source(operation: FigureOperationState) -> str | None:
    if operation.sources:
        return operation.sources[0]
    return None


def _selected_plot_array_data(
    context: FigureRecipeContext,
    operation: FigureOperationState,
    *,
    squeeze: bool = True,
) -> xr.DataArray | None:
    return _selected_plot_array_data_from_source(
        context,
        _primary_source(operation),
        squeeze=squeeze,
    )


def _selected_plot_array_data_from_source(
    context: FigureRecipeContext,
    source_name: str | None,
    *,
    squeeze: bool = True,
) -> xr.DataArray | None:
    if source_name is None or source_name not in context.source_data:
        return None
    data = _public_source_data(context.source_data[source_name])
    return data.squeeze(drop=True) if squeeze else data


@dataclasses.dataclass(frozen=True)
class _PlotArrayPreparePlan:
    """Semantic inputs used to prepare an image array for plotting."""

    source: str | None
    transpose: bool
    xlim: float | tuple[float | None, float | None] | None
    ylim: float | tuple[float | None, float | None] | None
    crop: bool
    rad2deg: bool | tuple[str, ...]

    @classmethod
    def from_operation_and_kwargs(
        cls,
        operation: FigureOperationState,
        kwargs: dict[str, typing.Any],
    ) -> _PlotArrayPreparePlan:
        rad2deg = kwargs.get("rad2deg", False)
        if not isinstance(rad2deg, bool):
            rad2deg = tuple(rad2deg)
        return cls(
            source=_primary_source(operation),
            transpose=operation.transpose,
            xlim=typing.cast(
                "float | tuple[float | None, float | None] | None",
                kwargs.get("xlim"),
            ),
            ylim=typing.cast(
                "float | tuple[float | None, float | None] | None",
                kwargs.get("ylim"),
            ),
            crop=bool(kwargs.get("crop", False)),
            rad2deg=rad2deg,
        )

    def prepare(self, context: FigureRecipeContext) -> xr.DataArray | None:
        data = _selected_plot_array_data_from_source(context, self.source)
        if data is None:
            return None
        if self.transpose:
            data = data.T
        prepared, _xlim, _ylim = _prepare_plot_array_data(
            data,
            xlim=self.xlim,
            ylim=self.ylim,
            crop=self.crop,
            rad2deg=self.rad2deg,
        )
        return prepared


def _safe_selected_plot_array_data(
    context: FigureRecipeContext,
    operation: FigureOperationState,
    *,
    squeeze: bool = True,
) -> xr.DataArray | None:
    try:
        return _selected_plot_array_data(context, operation, squeeze=squeeze)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _plot_array_selection_error(
    context: FigureRecipeContext, operation: FigureOperationState
) -> str | None:
    try:
        _selected_plot_array_data(context, operation)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return str(exc) or exc.__class__.__name__
    return None


def _plot_array_source_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    data = _selected_plot_array_data(tool._document, operation, squeeze=False)
    if data is None:
        return None
    source_name = _primary_source(operation)
    if source_name is None:
        return None
    code = _valid_source_variable(source_name)
    code = _maybe_squeeze_drop_code(code, data)
    if operation.transpose:
        code = f"{code}.T"
    return code


def _axes_count(context: FigureRecipeContext, operation: FigureOperationState) -> int:
    selection = operation.axes
    if selection.expression:
        return 1
    setup = context.recipe.setup
    if setup.layout_mode == "gridspec":
        return len(_gridspec_valid_axes_ids(setup, selection.axes_ids))
    return len(selection.valid_axes(setup))


def _has_invalid_target(
    context: FigureRecipeContext, operation: FigureOperationState
) -> bool:
    return context.axes_selection_has_invalid_target(operation.axes) or (
        _axes_count(context, operation) != 1
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    source_name = _primary_source(operation)
    source_text = (
        tool._source_display_name(source_name)
        if source_name is not None
        else "missing source"
    )
    if _plot_array_selection_error(tool._document, operation) is not None:
        shape = "invalid selection"
    else:
        data = _safe_selected_plot_array_data(tool._document, operation)
        if data is None:
            shape = "missing"
        elif data.ndim == 2:
            shape = "2D"
        else:
            shape = f"{data.ndim}D"
    return f"Image plot: {source_text}, {shape}"


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return (
        "Runs erlab.plotting.plot_array.\n"
        f"Targets: {tool._axes_target_text(operation.axes)}"
    )


def _update_current_source(editor: FigureOperationEditor, source: str | None) -> None:
    if source is None:
        return

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        return target.model_copy(update={"sources": (source,), "map_selections": ()})

    editor.request_transform(
        update_operation, rebuild_editor=True, defer_editor_rebuild=True
    )


def _build_source_editor(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> None:
    source_mixed = editor.batch_is_mixed(operation, _primary_source)
    source_combo = editor.source_combo(
        editor.context.source_names(),
        None if source_mixed else _primary_source(operation),
        lambda source: _update_current_source(editor, source),
        parent=editor.source_controls,
        mixed=source_mixed,
    )
    source_combo.setObjectName("figureComposerPlotArraySourceCombo")
    editor.add_source_row(
        "Image data",
        source_combo,
        "Data array plotted by this Image Plot step.",
    )


def _plot_array_kwargs(operation: FigureOperationState) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {}
    if operation.xlim is not None:
        kwargs["xlim"] = operation.xlim
    if operation.ylim is not None:
        kwargs["ylim"] = operation.ylim
    if operation.crop:
        kwargs["crop"] = True
    if operation.aspect is not None:
        kwargs["aspect"] = operation.aspect
    if operation.colorbar != "none":
        kwargs["colorbar"] = True
    if operation.colorbar_kw:
        kwargs["colorbar_kw"] = dict(operation.colorbar_kw)
    if operation.cmap is not None:
        kwargs["cmap"] = _matplotlib_cmap_name(operation.cmap)
    if _use_powernorm_plot_kwargs(operation):
        gamma = operation.norm_gamma
        if gamma is None:
            gamma = operation.gamma
        if gamma is not None:
            kwargs["gamma"] = gamma
        for name in ("vmin", "vmax"):
            value = getattr(operation, name)
            if value is not None:
                kwargs[name] = value
    else:
        kwargs["norm"] = _norm_object(operation)
    kwargs.update(dict(operation.extra_kwargs))
    return kwargs


def _render_plot_array(
    context: FigureRenderContext,
    operation: FigureOperationState,
    axs: typing.Any,
) -> None:
    kwargs = _plot_array_kwargs(operation)
    plan = _PlotArrayPreparePlan.from_operation_and_kwargs(operation, kwargs)
    data = context.cached_data(
        "plot-array",
        plan,
        lambda: plan.prepare(context.document),
    )
    if data is None:
        return
    if data.ndim != 2:
        raise ValueError("Image Plot requires a 2D DataArray")
    axis = _axes_from_selection(
        context.document,
        operation.axes,
        axs,
        for_plot_slices=False,
    )
    if isinstance(axis, (list, tuple)) or hasattr(axis, "flat"):
        raise ValueError("Image Plot requires exactly one target axis")
    kwargs["crop"] = False
    kwargs["rad2deg"] = False
    image = eplt.plot_array(
        data,
        ax=typing.cast("matplotlib.axes.Axes", axis),
        **kwargs,
    )
    if image is not None:
        tagged_image = typing.cast("typing.Any", image)
        tagged_image._figure_composer_operation_id = operation.operation_id
        tagged_image._figure_composer_panel_key = 0, 0


def _plot_array_code_kwargs(operation: FigureOperationState) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {}
    if operation.xlim is not None:
        kwargs["xlim"] = operation.xlim
    if operation.ylim is not None:
        kwargs["ylim"] = operation.ylim
    if operation.crop:
        kwargs["crop"] = True
    if operation.aspect is not None:
        kwargs["aspect"] = operation.aspect
    if operation.colorbar != "none":
        kwargs["colorbar"] = True
    if operation.colorbar_kw:
        kwargs["colorbar_kw"] = dict(operation.colorbar_kw)
    if operation.cmap is not None:
        kwargs["cmap"] = _matplotlib_cmap_name(operation.cmap)
    if _use_powernorm_plot_kwargs(operation):
        gamma = operation.norm_gamma
        if gamma is None:
            gamma = operation.gamma
        if gamma is not None:
            kwargs["gamma"] = gamma
        for name in ("vmin", "vmax"):
            value = getattr(operation, name)
            if value is not None:
                kwargs[name] = value
    else:
        kwargs["norm"] = _RawCode(_norm_code(operation))
    kwargs.update(dict(operation.extra_kwargs))
    return kwargs


def _plot_array_code_lines(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[str]:
    data_code = _plot_array_source_code(tool, operation)
    if data_code is None:
        return []
    kwargs = _plot_array_code_kwargs(operation)
    kwargs["ax"] = _RawCode(
        _axes_code(tool._document, operation.axes, for_plot_slices=False)
    )
    kwargs_text = _code_kwargs(kwargs)
    if kwargs_text:
        return [f"eplt.plot_array({data_code}, {kwargs_text})"]
    return [f"eplt.plot_array({data_code})"]


def _required_imports(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    imports = ["import erlab.plotting as eplt"]
    if (
        not _use_powernorm_plot_kwargs(operation)
        and _effective_norm_name(operation.norm_name) in _MATPLOTLIB_NORM_NAMES
    ):
        imports.append("import matplotlib.colors as mcolors")
    return tuple(imports)


def _operation_field_getter(
    attr: str,
) -> Callable[[FigureOperationState], object]:
    return lambda operation: getattr(operation, attr)


def _plot_limit_update_callback(
    editor: FigureOperationEditor, attr: str
) -> Callable[[str], None]:
    return lambda text: editor.request_update(**{attr: _plot_limit_from_text(text)})


def _format_aspect_value(value: typing.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return f"{float(value):g}"
    return str(value)


def _plot_array_aspect_combo(
    editor: FigureOperationEditor,
    operation: FigureOperationState,
    *,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QComboBox:
    aspect_options: tuple[tuple[str, str | None], ...] = (
        ("default", None),
        ("auto", "auto"),
        ("equal", "equal"),
    )
    aspect_mixed = editor.batch_is_mixed(operation, lambda target: target.aspect)
    combo = QtWidgets.QComboBox(parent)
    editor.mark_control(combo)
    if aspect_mixed:
        combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
    elif operation.aspect not in {None, "auto", "equal"}:
        combo.addItem(_format_aspect_value(operation.aspect), MIXED_VALUE)
    for label, value in aspect_options:
        combo.addItem(label, value)
    if aspect_mixed:
        typing.cast("typing.Any", combo.model()).item(0).setEnabled(False)
        combo.setCurrentIndex(0)
    elif operation.aspect in {None, "auto", "equal"}:
        combo.setCurrentIndex(combo.findData(operation.aspect))
    else:
        typing.cast("typing.Any", combo.model()).item(0).setEnabled(False)

    ComboBoxDataControlAdapter(combo).connect_commit(
        editor.connect_signal,
        lambda value: editor.request_update(aspect=value),
    )
    return combo


def _norm_gamma_value(operation: FigureOperationState) -> float:
    if operation.norm_gamma is not None:
        return operation.norm_gamma
    if operation.gamma is not None:
        return operation.gamma
    return 1.0


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


def _norm_number_update_callback(
    editor: FigureOperationEditor, attr: str
) -> Callable[[str], None]:
    def update(text: str) -> None:
        stripped = text.strip()
        editor.request_update(**{attr: None if not stripped else float(stripped)})

    return update


def _update_current_norm_name(editor: FigureOperationEditor, name: str) -> None:
    updates = {"norm_name": name, "gamma": None}
    if name != "PowerNorm":
        updates["norm_gamma"] = None
    editor.request_update_rebuild(**updates)


def _update_current_norm_gamma(editor: FigureOperationEditor, value: float) -> None:
    editor.request_update(norm_gamma=float(value), gamma=None)


def _update_current_norm_kwargs(editor: FigureOperationEditor, text: str) -> None:
    kwargs = _dict_from_text(text)
    updates = _norm_updates_from_kwargs(kwargs)
    editor.request_update_rebuild(**updates)


def _plot_array_default_cmap(editor: FigureOperationEditor) -> str:
    return str(editor.styled_rcparams_value("image.cmap"))


def _plot_array_cmap_base_and_reverse(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[str, bool]:
    if operation.cmap is None:
        return _cmap_base_and_reverse(_plot_array_default_cmap(editor))
    return _cmap_base_and_reverse(operation.cmap)


def _update_current_cmap(
    editor: FigureOperationEditor,
    *,
    base: str | None = None,
    reverse: bool | None = None,
) -> None:
    current = editor.current_operation()
    if current is None:
        return
    _index, operation = current
    old_base, old_reverse = _plot_array_cmap_base_and_reverse(editor, operation)
    if base is None:
        base = old_base
    if reverse is None:
        reverse = old_reverse
    editor.request_update(cmap=_cmap_with_reverse(base, reverse))


def _build_plot_array_view_page(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> QtWidgets.QWidget:
    page, layout = editor.new_form_page("figureComposerPlotArrayViewPage")

    editor.add_form_section(
        layout,
        "Image",
        object_name="figureComposerPlotArrayViewImageSection",
    )
    data = _safe_selected_plot_array_data(editor.context, operation)
    summary = QtWidgets.QLabel(
        "No source data is available."
        if data is None
        else f"Input dims: {', '.join(str(dim) for dim in data.dims) or 'none'}",
        page,
    )
    summary.setObjectName("figureComposerPlotArrayShapeSummary")
    summary.setWordWrap(True)
    editor.add_form_row(
        layout,
        "Summary",
        summary,
        "Shows the dimensions plotted by this Image Plot step.",
    )

    transpose_mixed = editor.batch_is_mixed(operation, lambda target: target.transpose)
    transpose_check = editor.check_box(
        operation.transpose,
        lambda checked: editor.request_update(transpose=checked),
        parent=page,
        mixed=transpose_mixed,
    )
    transpose_check.setObjectName("figureComposerPlotArrayTransposeCheck")
    transpose_check.setText("")
    editor.add_form_row(
        layout,
        "Transpose",
        transpose_check,
        "Swap the x/y orientation before calling plot_array.",
    )

    aspect_combo = _plot_array_aspect_combo(editor, operation, parent=page)
    aspect_combo.setObjectName("figureComposerPlotArrayAspectCombo")
    editor.add_form_row(
        layout,
        "Aspect",
        aspect_combo,
        "Aspect argument passed through to imshow.",
    )

    editor.add_form_section(
        layout,
        "Axes",
        object_name="figureComposerPlotArrayViewAxesSection",
    )
    limit_controls: list[tuple[str, QtWidgets.QWidget, str]] = []
    for label, attr in (("x", "xlim"), ("y", "ylim")):
        text, mixed = editor.batch_text(
            operation,
            _operation_field_getter(attr),
            _format_plot_limit,
        )
        edit = editor.line_edit(text, parent=page)
        edit.setObjectName(f"figureComposerPlotArray{label.upper()}LimEdit")
        editor.apply_mixed_line_edit(edit, mixed)
        editor.connect_line_edit_finished(
            edit, _plot_limit_update_callback(editor, attr)
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
        layout,
        "Limits",
        limit_controls,
        "Optional x/y plot limits for this image.",
    )

    crop_mixed = editor.batch_is_mixed(operation, lambda target: target.crop)
    crop_check = editor.check_box(
        operation.crop,
        lambda checked: editor.request_update(crop=checked),
        parent=page,
        mixed=crop_mixed,
    )
    crop_check.setObjectName("figureComposerPlotArrayCropCheck")
    crop_check.setText("")
    editor.add_form_row(
        layout,
        "Crop",
        crop_check,
        "Crop data to explicit x/y limits before plotting.",
    )
    return page


def _build_plot_array_colors_page(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> QtWidgets.QWidget:
    page, layout = editor.new_form_page("figureComposerPlotArrayColorsPage")

    editor.add_form_section(
        layout,
        "Image color",
        object_name="figureComposerPlotArrayColorsImageColorSection",
    )
    cmap_widget = QtWidgets.QWidget(page)
    cmap_layout = QtWidgets.QHBoxLayout(cmap_widget)
    cmap_layout.setContentsMargins(0, 0, 0, 0)
    cmap_layout.setSpacing(4)
    cmap_base, cmap_reversed = _plot_array_cmap_base_and_reverse(editor, operation)
    cmap_mixed = editor.batch_is_mixed(
        operation, lambda target: _plot_array_cmap_base_and_reverse(editor, target)[0]
    )
    reverse_mixed = editor.batch_is_mixed(
        operation, lambda target: _plot_array_cmap_base_and_reverse(editor, target)[1]
    )
    cmap_combo = erlab.interactive.colors.ColorMapComboBox(cmap_widget)
    editor.mark_control(cmap_combo)
    cmap_combo.setObjectName("figureComposerPlotArrayCmapCombo")
    cmap_combo.default_cmap = None if cmap_mixed else cmap_base
    cmap_combo.ensure_populated()
    if cmap_mixed:
        with QtCore.QSignalBlocker(cmap_combo):
            editor.set_combo_mixed_placeholder(cmap_combo)
    elif cmap_combo.currentText() != cmap_base:
        with QtCore.QSignalBlocker(cmap_combo):
            cmap_combo.setCurrentText(cmap_base)
    cmap_reverse_check = editor.check_box(
        cmap_reversed,
        lambda checked: _update_current_cmap(editor, reverse=checked),
        parent=cmap_widget,
        mixed=reverse_mixed,
    )
    cmap_reverse_check.setText("Reverse")
    cmap_reverse_check.setObjectName("figureComposerPlotArrayCmapReverseCheck")
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
        layout,
        "Colormap",
        cmap_widget,
        "Colormap and reverse-colormap controls.",
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
        parent=page,
        mixed=editor.batch_is_mixed(
            operation, lambda target: _norm_combo_text(target.norm_name)
        ),
    )
    norm_combo.setObjectName("figureComposerPlotArrayNormCombo")
    editor.add_form_row(layout, "Norm", norm_combo, "Color normalization.")

    norm_fields = _norm_kwarg_fields(operation.norm_name)
    if "gamma" in norm_fields:
        gamma_mixed = editor.batch_is_mixed(
            operation, lambda target: _norm_gamma_value(target)
        )
        gamma_widget = erlab.interactive.colors.ColorMapGammaWidget(
            page,
            value=_norm_gamma_value(operation),
            spin_cls=erlab.interactive.utils.BetterSpinBox,
        )
        gamma_widget.setObjectName("figureComposerPlotArrayGammaWidget")
        editor.connect_signal(
            gamma_widget,
            gamma_widget.valueChanged,
            lambda value: _update_current_norm_gamma(editor, value),
        )
        editor.add_form_row(
            layout,
            "Gamma",
            editor.mixed_value_widget(gamma_widget, mixed=gamma_mixed, parent=page),
            "Gamma value for the selected normalization.",
        )

    norm_number_fields = {
        "vmin": ("vmin", "Lower color-normalization bound."),
        "vmax": ("vmax", "Upper color-normalization bound."),
        "vcenter": ("vcenter", "Center value for diverging normalization classes."),
        "halfrange": (
            "halfrange",
            "Symmetric half-range for centered ERLab normalization classes.",
        ),
    }
    norm_number_widgets: dict[str, tuple[str, QtWidgets.QWidget, str]] = {}
    for attr, (label, tooltip) in norm_number_fields.items():
        if attr not in norm_fields:
            continue
        text, mixed = editor.batch_text(
            operation,
            _operation_field_getter(attr),
            lambda value: "" if value is None else str(value),
        )
        edit = editor.line_edit(text, parent=page)
        editor.apply_mixed_line_edit(edit, mixed)
        edit.setObjectName(f"figureComposerPlotArray{attr[0].upper()}{attr[1:]}Edit")
        editor.connect_line_edit_finished(
            edit,
            _norm_number_update_callback(editor, attr),
        )
        norm_number_widgets[attr] = (label, edit, tooltip)

    if "vmin" in norm_number_widgets and "vmax" in norm_number_widgets:
        editor.add_compound_form_row(
            layout,
            "Color limits",
            (
                norm_number_widgets.pop("vmin"),
                norm_number_widgets.pop("vmax"),
            ),
            "Lower and upper color-normalization bounds.",
        )
    if "vcenter" in norm_number_widgets and "halfrange" in norm_number_widgets:
        editor.add_compound_form_row(
            layout,
            "Center/range",
            (
                norm_number_widgets.pop("vcenter"),
                norm_number_widgets.pop("halfrange"),
            ),
            "Center and half-range for centered color normalization.",
        )
    for label, edit, tooltip in norm_number_widgets.values():
        editor.add_form_row(layout, label, edit, tooltip)

    if "clip" in norm_fields:
        clip_mixed = editor.batch_is_mixed(operation, lambda target: target.norm_clip)
        clip_combo = editor.combo(
            ["default", "False", "True"],
            None if clip_mixed else _norm_clip_text(operation.norm_clip),
            lambda text: editor.request_update(norm_clip=_norm_clip_from_text(text)),
            parent=page,
            mixed=clip_mixed,
        )
        clip_combo.setObjectName("figureComposerPlotArrayNormClipCombo")
        editor.add_form_row(
            layout,
            "Clip",
            clip_combo,
            "clip argument for the selected normalization object.",
        )

    norm_kwargs_text, norm_kwargs_mixed = editor.batch_text(
        operation, lambda target: target.norm_kwargs, _format_dict
    )
    norm_kwargs_edit = editor.line_edit(norm_kwargs_text, parent=page)
    editor.apply_mixed_line_edit(norm_kwargs_edit, norm_kwargs_mixed)
    norm_kwargs_edit.setObjectName("figureComposerPlotArrayNormKwargsEdit")
    editor.connect_line_edit_finished(
        norm_kwargs_edit,
        lambda text: _update_current_norm_kwargs(editor, text),
    )
    editor.add_form_row(
        layout,
        "Norm kwargs",
        norm_kwargs_edit,
        "Extra dict literal or keyword arguments for the norm constructor.",
    )

    editor.add_form_section(
        layout,
        "Colorbar",
        object_name="figureComposerPlotArrayColorsColorbarSection",
    )
    colorbar_mixed = editor.batch_is_mixed(operation, lambda target: target.colorbar)
    colorbar_combo = editor.combo(
        ["none", "right"],
        None if colorbar_mixed else operation.colorbar,
        lambda text: editor.request_update(colorbar=text),
        parent=page,
        mixed=colorbar_mixed,
    )
    colorbar_combo.setObjectName("figureComposerPlotArrayColorbarCombo")
    editor.add_form_row(
        layout,
        "Colorbar",
        colorbar_combo,
        "Whether plot_array should draw a colorbar.",
    )

    colorbar_kwargs_text, colorbar_kwargs_mixed = editor.batch_text(
        operation, lambda target: target.colorbar_kw, _format_dict
    )
    colorbar_kwargs_edit = editor.line_edit(colorbar_kwargs_text, parent=page)
    editor.apply_mixed_line_edit(colorbar_kwargs_edit, colorbar_kwargs_mixed)
    colorbar_kwargs_edit.setObjectName("figureComposerPlotArrayColorbarKwEdit")
    editor.connect_line_edit_finished(
        colorbar_kwargs_edit,
        lambda text: editor.request_update(colorbar_kw=_dict_from_text(text)),
    )
    editor.add_form_row(
        layout,
        "Colorbar kwargs",
        colorbar_kwargs_edit,
        "Dict literal or keyword arguments forwarded as colorbar_kw.",
    )
    return page


def _build_plot_array_advanced_page(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> QtWidgets.QWidget:
    page, layout = editor.new_form_page("figureComposerPlotArrayAdvancedPage")
    extra_text, extra_mixed = editor.batch_text(
        operation, lambda target: target.extra_kwargs, _format_dict
    )
    extra_edit = editor.line_edit(extra_text, parent=page)
    editor.apply_mixed_line_edit(extra_edit, extra_mixed)
    extra_edit.setObjectName("figureComposerPlotArrayExtraKwEdit")
    editor.connect_line_edit_finished(
        extra_edit,
        lambda text: editor.request_update(extra_kwargs=_dict_from_text(text)),
    )
    editor.add_form_row(
        layout,
        "Extra kwargs",
        extra_edit,
        "Dict literal or keyword arguments forwarded to plot_array.",
    )
    return page


def _editor_sections(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    return (
        StepSection(
            "view",
            "View",
            _build_plot_array_view_page(editor, operation),
            "Choose image orientation, limits, and cropping.",
        ),
        StepSection(
            "colors",
            "Colors",
            _build_plot_array_colors_page(editor, operation),
            "Set colormap, normalization, and colorbar options.",
        ),
        StepSection(
            "advanced",
            "Other",
            _build_plot_array_advanced_page(editor, operation),
            "Pass advanced keyword arguments to plot_array.",
        ),
    )


def _section_summary(
    tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    match key:
        case "sources":
            source_name = _primary_source(operation)
            return tool._source_display_name(source_name) if source_name else "none"
        case "axes":
            return tool._axes_target_text(operation.axes)
        case "view":
            labels = [
                label
                for label, value in (
                    ("x", operation.xlim),
                    ("y", operation.ylim),
                    ("T", operation.transpose),
                    ("aspect", operation.aspect),
                )
                if value
            ]
            return ", ".join(labels) if labels else "auto"
        case "colors":
            return operation.cmap or "default"
        case "advanced":
            return "set" if operation.extra_kwargs else ""
    return ""


SPEC = OperationSpec(
    kind=FigureOperationKind.PLOT_ARRAY,
    add_actions=(
        AddStepActionSpec(
            action_id=FigureOperationKind.PLOT_ARRAY.value,
            text="Image Plot",
            tooltip="Add an editable erlab.plotting.plot_array step.",
            create_operation=_create_plot_array_operation,
        ),
    ),
    display_text=_display_text,
    tooltip=_tooltip,
    target_text=lambda tool, operation: tool._axes_target_text(operation.axes),
    has_invalid_target=_has_invalid_target,
    uses_source_section=lambda _operation: True,
    build_source_editor=_build_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=lambda context, operation, _figure, axs: _render_plot_array(
        context, operation, axs
    ),
    code_lines=_plot_array_code_lines,
    render_cache_safe=_always_render_cache_safe,
    required_imports=_required_imports,
)
