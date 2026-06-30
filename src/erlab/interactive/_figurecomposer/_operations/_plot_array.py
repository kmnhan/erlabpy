"""Plot Array operation editor, renderer, and code generation."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtWidgets

import erlab
import erlab.plotting as eplt
from erlab.interactive._figurecomposer._code import (
    _axes_code,
    _maybe_squeeze_drop_code,
    _selection_code,
)
from erlab.interactive._figurecomposer._defaults import _styled_rcparams_value
from erlab.interactive._figurecomposer._editor_controls import (
    MIXED_VALUE,
    MIXED_VALUES_TEXT,
    ComboBoxDataControlAdapter,
)
from erlab.interactive._figurecomposer._gridspec import _gridspec_valid_axes_ids
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
    StepSection,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _tool_figure_options_context,
)
from erlab.interactive._figurecomposer._sources import (
    _public_source_data,
    _selected_data,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._state import (
    FigureDataSelectionState,
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._text import (
    FigureComposerInputError,
    _code_kwargs,
    _dict_from_text,
    _format_dict,
    _format_plot_limit,
    _plot_limit_from_text,
    _RawCode,
)

if typing.TYPE_CHECKING:
    import matplotlib.axes
    import xarray as xr

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _create_plot_array_operation(tool: FigureComposerTool) -> FigureOperationState:
    source_names = tool._source_names()
    first_source = source_names[0] if source_names else tool._recipe.primary_source
    return FigureOperationState.plot_array(
        label="plot_array",
        source=first_source,
        axes=tool._selected_axes_state(),
    )


def _source_names(operation: FigureOperationState) -> tuple[str, ...]:
    names: list[str] = []
    for source_name in operation.sources:
        if source_name not in names:
            names.append(source_name)
    for selection in operation.map_selections:
        if selection.source not in names:
            names.append(selection.source)
    return tuple(names)


def _primary_source(operation: FigureOperationState) -> str | None:
    if operation.map_selections:
        return operation.map_selections[0].source
    if operation.sources:
        return operation.sources[0]
    return None


def _selected_plot_array_data(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    *,
    squeeze: bool = True,
) -> xr.DataArray | None:
    if operation.map_selections:
        data = _selected_data(tool._source_data, operation.map_selections[0])
        if data is None:
            return None
        return data.squeeze(drop=True) if squeeze else data
    source_name = _primary_source(operation)
    if source_name is None or source_name not in tool._source_data:
        return None
    data = _public_source_data(tool._source_data[source_name])
    return data.squeeze(drop=True) if squeeze else data


def _safe_selected_plot_array_data(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    *,
    squeeze: bool = True,
) -> xr.DataArray | None:
    try:
        return _selected_plot_array_data(tool, operation, squeeze=squeeze)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _primary_selection(operation: FigureOperationState) -> FigureDataSelectionState:
    if operation.map_selections:
        return operation.map_selections[0]
    return FigureDataSelectionState(source=_primary_source(operation) or "")


def _plot_array_source_data(
    tool: FigureComposerTool, operation: FigureOperationState
) -> xr.DataArray | None:
    source_name = _primary_source(operation)
    if source_name is None or source_name not in tool._source_data:
        return None
    return _public_source_data(tool._source_data[source_name])


def _plot_array_selection_error(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    try:
        _selected_plot_array_data(tool, operation)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return str(exc) or exc.__class__.__name__
    return None


def _selection_summary(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str:
    source_data = _plot_array_source_data(tool, operation)
    selection_error = _plot_array_selection_error(tool, operation)
    selected = (
        None
        if selection_error is not None
        else (_safe_selected_plot_array_data(tool, operation))
    )
    input_dims = (
        "missing"
        if source_data is None
        else ", ".join(str(dim) for dim in source_data.dims) or "none"
    )
    plotted_dims = (
        "missing"
        if selected is None
        else ", ".join(str(dim) for dim in selected.dims) or "scalar"
    )
    text = f"Input dims: {input_dims}\nPlotted dims: {plotted_dims}"
    if selection_error is not None:
        text += f"\nSelection error: {selection_error}"
    return text


def _plot_array_operation_with_selection(
    operation: FigureOperationState,
    selection: FigureDataSelectionState,
) -> FigureOperationState:
    map_selections = (
        (selection,) if selection.isel or selection.qsel or selection.mean_dims else ()
    )
    return operation.model_copy(
        update={
            "sources": (selection.source,),
            "map_selections": map_selections,
        }
    )


def _update_current_selection_source(
    tool: FigureComposerTool, source: str | None
) -> None:
    if source is None:
        return

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        selection = _primary_selection(target).model_copy(update={"source": source})
        return _plot_array_operation_with_selection(target, selection)

    tool._update_operations(update_operation, rebuild_editor=True)


_PLOT_ARRAY_SELECTION_MODE_LABELS = {
    "keep": "None",
    "isel": "isel",
    "qsel": "qsel",
    "mean": "Mean",
}
_PLOT_ARRAY_SELECTION_VALUE_MODES = {"isel", "qsel"}


def _plot_array_selection_dim_mode(
    selection: FigureDataSelectionState, dim: str
) -> str:
    if dim in selection.isel:
        return "isel"
    if dim in selection.qsel:
        return "qsel"
    if dim in selection.mean_dims:
        return "mean"
    return "keep"


def _plot_array_selection_dim_value_text(
    selection: FigureDataSelectionState, dim: str
) -> str:
    if dim in selection.isel:
        return erlab.interactive.utils._parse_single_arg(selection.isel[dim])
    if dim in selection.qsel:
        return erlab.interactive.utils._parse_single_arg(selection.qsel[dim])
    return ""


def _plot_array_selection_value_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        raise FigureComposerInputError(
            "Enter a selection value, such as 0 or slice(0, 2)."
        )
    try:
        return _dict_from_text(f"value={stripped}", allow_slice=True)["value"]
    except FigureComposerInputError as exc:
        raise FigureComposerInputError(
            "Enter a selection value, such as 0 or slice(0, 2)."
        ) from exc


def _plot_array_selection_with_dimension(
    selection: FigureDataSelectionState,
    dim: str,
    mode: str,
    value: typing.Any = None,
) -> FigureDataSelectionState:
    isel = dict(selection.isel)
    qsel = dict(selection.qsel)
    mean_dims = [target for target in selection.mean_dims if target != dim]
    isel.pop(dim, None)
    qsel.pop(dim, None)
    if mode == "isel":
        isel[dim] = value
    elif mode == "qsel":
        qsel[dim] = value
    elif mode == "mean":
        mean_dims.append(dim)
    return selection.model_copy(
        update={
            "isel": isel,
            "qsel": qsel,
            "mean_dims": tuple(mean_dims),
        }
    )


def _update_current_selection_dimension(
    tool: FigureComposerTool,
    dim: str,
    mode: str,
    value_text: str = "",
) -> None:
    if mode not in _PLOT_ARRAY_SELECTION_MODE_LABELS:
        return
    value = (
        _plot_array_selection_value_from_text(value_text)
        if mode in _PLOT_ARRAY_SELECTION_VALUE_MODES
        else None
    )

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        selection = _plot_array_selection_with_dimension(
            _primary_selection(target),
            dim,
            mode,
            value,
        )
        return _plot_array_operation_with_selection(target, selection)

    tool._update_operations(update_operation, rebuild_editor=True)


def _plot_array_selection_mode_combo(
    tool: FigureComposerTool,
    *,
    current: str | None,
    mixed: bool,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox(parent)
    tool._mark_editor_control(combo)
    if mixed:
        combo.addItem(MIXED_VALUES_TEXT, MIXED_VALUE)
    for mode, label in _PLOT_ARRAY_SELECTION_MODE_LABELS.items():
        combo.addItem(label, mode)
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
    return combo


def _connect_plot_array_selection_dimension_controls(
    tool: FigureComposerTool,
    dim: str,
    mode_combo: QtWidgets.QComboBox,
    value_edit: QtWidgets.QLineEdit,
) -> None:
    def mode_changed(value: typing.Any) -> None:
        mode = value if isinstance(value, str) else ""
        value_mode = mode in _PLOT_ARRAY_SELECTION_VALUE_MODES
        value_edit.setEnabled(value_mode)
        if value_mode:
            if value_edit.text().strip():
                _update_current_selection_dimension(tool, dim, mode, value_edit.text())
            return
        value_edit.clear()
        value_edit.setModified(False)
        _update_current_selection_dimension(tool, dim, mode)

    def value_changed(text: str) -> None:
        mode = mode_combo.currentData()
        if isinstance(mode, str) and mode in _PLOT_ARRAY_SELECTION_VALUE_MODES:
            _update_current_selection_dimension(tool, dim, mode, text)

    ComboBoxDataControlAdapter(mode_combo).connect_commit(
        tool._connect_editor_signal,
        mode_changed,
    )
    tool._connect_line_edit_finished(value_edit, value_changed)


def _plot_array_source_code(
    tool: FigureComposerTool, operation: FigureOperationState
) -> str | None:
    data = _selected_plot_array_data(tool, operation, squeeze=False)
    if data is None:
        return None
    if operation.map_selections:
        code = _selection_code(operation.map_selections[0])
    else:
        source_name = _primary_source(operation)
        if source_name is None:
            return None
        code = _valid_source_variable(source_name)
    code = _maybe_squeeze_drop_code(code, data)
    if operation.transpose:
        code = f"{code}.T"
    return code


def _axes_count(tool: FigureComposerTool, operation: FigureOperationState) -> int:
    selection = operation.axes
    if selection.expression:
        return 1
    setup = tool._recipe.setup
    if setup.layout_mode == "gridspec":
        return len(_gridspec_valid_axes_ids(setup, selection.axes_ids))
    return len(selection.valid_axes(setup))


def _has_invalid_target(
    tool: FigureComposerTool, operation: FigureOperationState
) -> bool:
    return tool._axes_selection_has_invalid_target(operation.axes) or (
        _axes_count(tool, operation) != 1
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    prefix = "Needs one axis: " if _has_invalid_target(tool, operation) else ""
    source_name = _primary_source(operation)
    source_text = (
        tool._source_display_name(source_name)
        if source_name is not None
        else "missing source"
    )
    if _plot_array_selection_error(tool, operation) is not None:
        shape = "invalid selection"
    else:
        data = _safe_selected_plot_array_data(tool, operation)
        if data is None:
            shape = "missing"
        elif data.ndim == 2:
            shape = "2D"
        else:
            shape = f"{data.ndim}D"
    return f"{prefix}Image plot: {source_text}, {shape}"


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return (
        "Runs erlab.plotting.plot_array.\n"
        f"Targets: {tool._axes_target_text(operation.axes)}"
    )


def _update_current_source(tool: FigureComposerTool, source: str | None) -> None:
    if source is None:
        return

    def update_operation(
        _index: int, target: FigureOperationState
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {"sources": (source,)}
        if target.map_selections:
            updates["map_selections"] = tuple(
                selection.model_copy(update={"source": source})
                for selection in target.map_selections
            )
        return target.model_copy(update=updates)

    tool._update_operations(update_operation, rebuild_editor=True)


def _build_source_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> None:
    source_mixed = tool._batch_is_mixed(operation, _primary_source)
    source_combo = tool._source_combo(
        tool._source_names(),
        None if source_mixed else _primary_source(operation),
        lambda source: _update_current_source(tool, source),
        parent=tool.step_source_controls,
        mixed=source_mixed,
    )
    source_combo.setObjectName("figureComposerPlotArraySourceCombo")
    tool._add_form_row(
        tool.step_source_controls_layout,
        "Image data",
        source_combo,
        "Data array plotted by this Image Plot step.",
    )


def _build_plot_array_selection_page(
    tool: FigureComposerTool, operation: FigureOperationState
) -> QtWidgets.QWidget:
    page, layout = tool._new_step_form_page("figureComposerPlotArraySelectionPage")
    selection = _primary_selection(operation)

    tool._add_form_section(
        layout,
        "Data",
        object_name="figureComposerPlotArraySelectionDataSection",
    )
    summary = QtWidgets.QLabel(_selection_summary(tool, operation), page)
    summary.setObjectName("figureComposerPlotArraySelectionSummary")
    summary.setWordWrap(True)
    tool._add_form_row(
        layout,
        "Summary",
        summary,
        "Shows the original dimensions and the dimensions plotted by this image step.",
    )

    source_mixed = tool._batch_is_mixed(operation, _primary_source)
    source_combo = tool._source_combo(
        tool._source_names(),
        None if source_mixed else selection.source or _primary_source(operation),
        lambda source: _update_current_selection_source(tool, source),
        parent=page,
        mixed=source_mixed,
    )
    source_combo.setObjectName("figureComposerPlotArraySelectionSourceCombo")
    tool._add_form_row(
        layout,
        "Image data",
        source_combo,
        "Data array selected before plot_array draws this image.",
    )

    tool._add_form_section(
        layout,
        "Dimensions",
        object_name="figureComposerPlotArraySelectionDimensionsSection",
    )
    source_data = None if source_mixed else _plot_array_source_data(tool, operation)
    if source_mixed or source_data is None:
        dimensions_message = QtWidgets.QLabel(
            "Choose an image data source to edit dimension selections."
            if source_mixed
            else "No source dimensions are available.",
            page,
        )
        dimensions_message.setObjectName(
            "figureComposerPlotArraySelectionDimensionsMessage"
        )
        dimensions_message.setWordWrap(True)
        tool._add_form_row(
            layout,
            "Dimensions",
            dimensions_message,
            "Dimension controls are available after the source data is known.",
        )
        return page

    if not source_data.dims:
        dimensions_message = QtWidgets.QLabel("No dimensions.", page)
        dimensions_message.setObjectName(
            "figureComposerPlotArraySelectionDimensionsMessage"
        )
        tool._add_form_row(
            layout,
            "Dimensions",
            dimensions_message,
            "This source is scalar, so no dimension selections are available.",
        )
        return page

    for dim_index, dim in enumerate(source_data.dims):
        dim_name = str(dim)

        def mode_getter(target: FigureOperationState, dim_name: str = dim_name) -> str:
            return _plot_array_selection_dim_mode(_primary_selection(target), dim_name)

        def value_getter(target: FigureOperationState, dim_name: str = dim_name) -> str:
            return _plot_array_selection_dim_value_text(
                _primary_selection(target), dim_name
            )

        mode_mixed = tool._batch_is_mixed(operation, mode_getter)
        value_text, value_mixed = tool._batch_text(
            operation,
            value_getter,
            str,
        )
        current_mode = (
            None if mode_mixed else _plot_array_selection_dim_mode(selection, dim_name)
        )
        row = QtWidgets.QWidget(page)
        row.setObjectName(f"figureComposerPlotArraySelectionDimRow{dim_index}")
        row.setProperty("figure_composer_plot_array_dim", dim_name)
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        mode_combo = _plot_array_selection_mode_combo(
            tool,
            current=current_mode,
            mixed=mode_mixed,
            parent=row,
        )
        mode_combo.setObjectName(
            f"figureComposerPlotArraySelectionModeCombo{dim_index}"
        )
        mode_combo.setProperty("figure_composer_plot_array_dim", dim_name)
        value_edit = tool._line_edit(value_text, parent=row)
        value_edit.setObjectName(
            f"figureComposerPlotArraySelectionValueEdit{dim_index}"
        )
        value_edit.setProperty("figure_composer_plot_array_dim", dim_name)
        value_edit.setPlaceholderText("value")
        value_edit.setEnabled(
            current_mode in _PLOT_ARRAY_SELECTION_VALUE_MODES
            if current_mode is not None
            else False
        )
        tool._apply_mixed_line_edit(value_edit, value_mixed)
        _connect_plot_array_selection_dimension_controls(
            tool,
            dim_name,
            mode_combo,
            value_edit,
        )

        row_layout.addWidget(mode_combo)
        row_layout.addWidget(value_edit, 1)
        tool._add_form_row(
            layout,
            dim_name,
            row,
            "Keep this dimension, select by integer index, select by coordinate, "
            "or average it before plotting.",
        )
    return page


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
    tool: FigureComposerTool, operation: FigureOperationState, axs: typing.Any
) -> None:
    data = _selected_plot_array_data(tool, operation)
    if data is None:
        return
    if operation.transpose:
        data = data.T
    if data.ndim != 2:
        raise ValueError("Image Plot requires a 2D DataArray")
    axis = _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)
    if isinstance(axis, (list, tuple)) or hasattr(axis, "flat"):
        raise ValueError("Image Plot requires exactly one target axis")
    image = eplt.plot_array(
        data,
        ax=typing.cast("matplotlib.axes.Axes", axis),
        **_plot_array_kwargs(operation),
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
    kwargs["ax"] = _RawCode(_axes_code(tool, operation.axes, for_plot_slices=False))
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
) -> typing.Callable[[FigureOperationState], object]:
    return lambda operation: getattr(operation, attr)


def _plot_limit_update_callback(
    tool: FigureComposerTool, attr: str
) -> typing.Callable[[str], None]:
    return lambda text: tool._update_current_operation(
        **{attr: _plot_limit_from_text(text)}
    )


def _format_aspect_value(value: typing.Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return f"{float(value):g}"
    return str(value)


def _plot_array_aspect_combo(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    *,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QComboBox:
    aspect_options: tuple[tuple[str, str | None], ...] = (
        ("default", None),
        ("auto", "auto"),
        ("equal", "equal"),
    )
    aspect_mixed = tool._batch_is_mixed(operation, lambda target: target.aspect)
    combo = QtWidgets.QComboBox(parent)
    tool._mark_editor_control(combo)
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
        tool._connect_editor_signal,
        lambda value: tool._update_current_operation(aspect=value),
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
    tool: FigureComposerTool, attr: str
) -> typing.Callable[[str], None]:
    def update(text: str) -> None:
        stripped = text.strip()
        tool._update_current_operation(
            **{attr: None if not stripped else float(stripped)}
        )

    return update


def _update_current_norm_name(tool: FigureComposerTool, name: str) -> None:
    updates = {"norm_name": name, "gamma": None}
    if name != "PowerNorm":
        updates["norm_gamma"] = None
    tool._update_current_operation_rebuild(**updates)


def _update_current_norm_gamma(tool: FigureComposerTool, value: float) -> None:
    tool._update_current_operation(norm_gamma=float(value), gamma=None)


def _update_current_norm_kwargs(tool: FigureComposerTool, text: str) -> None:
    kwargs = _dict_from_text(text)
    updates = _norm_updates_from_kwargs(kwargs)
    tool._update_current_operation_rebuild(**updates)


def _plot_array_default_cmap(tool: FigureComposerTool) -> str:
    with _tool_figure_options_context(tool):
        return str(_styled_rcparams_value("image.cmap"))


def _plot_array_cmap_base_and_reverse(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, bool]:
    if operation.cmap is None:
        return _cmap_base_and_reverse(_plot_array_default_cmap(tool))
    return _cmap_base_and_reverse(operation.cmap)


def _update_current_cmap(
    tool: FigureComposerTool,
    *,
    base: str | None = None,
    reverse: bool | None = None,
) -> None:
    current = tool._current_operation()
    if current is None:
        return
    _index, operation = current
    old_base, old_reverse = _plot_array_cmap_base_and_reverse(tool, operation)
    if base is None:
        base = old_base
    if reverse is None:
        reverse = old_reverse
    tool._update_current_operation(cmap=_cmap_with_reverse(base, reverse))


def _build_plot_array_view_page(
    tool: FigureComposerTool, operation: FigureOperationState
) -> QtWidgets.QWidget:
    page, layout = tool._new_step_form_page("figureComposerPlotArrayViewPage")

    tool._add_form_section(
        layout,
        "Image",
        object_name="figureComposerPlotArrayViewImageSection",
    )
    data = _safe_selected_plot_array_data(tool, operation)
    summary = QtWidgets.QLabel(
        "No source data is available."
        if data is None
        else f"Input dims: {', '.join(str(dim) for dim in data.dims) or 'none'}",
        page,
    )
    summary.setObjectName("figureComposerPlotArrayShapeSummary")
    summary.setWordWrap(True)
    tool._add_form_row(
        layout,
        "Summary",
        summary,
        "Shows the dimensions plotted by this Image Plot step.",
    )

    transpose_mixed = tool._batch_is_mixed(operation, lambda target: target.transpose)
    transpose_check = tool._check_box(
        operation.transpose,
        lambda checked: tool._update_current_operation(transpose=checked),
        parent=page,
        mixed=transpose_mixed,
    )
    transpose_check.setObjectName("figureComposerPlotArrayTransposeCheck")
    transpose_check.setText("")
    tool._add_form_row(
        layout,
        "Transpose",
        transpose_check,
        "Swap the x/y orientation before calling plot_array.",
    )

    aspect_combo = _plot_array_aspect_combo(tool, operation, parent=page)
    aspect_combo.setObjectName("figureComposerPlotArrayAspectCombo")
    tool._add_form_row(
        layout,
        "Aspect",
        aspect_combo,
        "Aspect argument passed through to imshow.",
    )

    tool._add_form_section(
        layout,
        "Axes",
        object_name="figureComposerPlotArrayViewAxesSection",
    )
    limit_controls: list[tuple[str, QtWidgets.QWidget, str]] = []
    for label, attr in (("x", "xlim"), ("y", "ylim")):
        text, mixed = tool._batch_text(
            operation,
            _operation_field_getter(attr),
            _format_plot_limit,
        )
        edit = tool._line_edit(text, parent=page)
        edit.setObjectName(f"figureComposerPlotArray{label.upper()}LimEdit")
        tool._apply_mixed_line_edit(edit, mixed)
        tool._connect_line_edit_finished(edit, _plot_limit_update_callback(tool, attr))
        limit_controls.append(
            (
                label,
                edit,
                f"Optional {attr}: one number for symmetric limits, "
                "or two comma-separated numbers for lower and upper limits.",
            )
        )
    tool._add_compound_form_row(
        layout,
        "Limits",
        limit_controls,
        "Optional x/y plot limits for this image.",
    )

    crop_mixed = tool._batch_is_mixed(operation, lambda target: target.crop)
    crop_check = tool._check_box(
        operation.crop,
        lambda checked: tool._update_current_operation(crop=checked),
        parent=page,
        mixed=crop_mixed,
    )
    crop_check.setObjectName("figureComposerPlotArrayCropCheck")
    crop_check.setText("")
    tool._add_form_row(
        layout,
        "Crop",
        crop_check,
        "Crop data to explicit x/y limits before plotting.",
    )
    return page


def _build_plot_array_colors_page(
    tool: FigureComposerTool, operation: FigureOperationState
) -> QtWidgets.QWidget:
    page, layout = tool._new_step_form_page("figureComposerPlotArrayColorsPage")

    tool._add_form_section(
        layout,
        "Image color",
        object_name="figureComposerPlotArrayColorsImageColorSection",
    )
    cmap_widget = QtWidgets.QWidget(page)
    cmap_layout = QtWidgets.QHBoxLayout(cmap_widget)
    cmap_layout.setContentsMargins(0, 0, 0, 0)
    cmap_layout.setSpacing(4)
    cmap_base, cmap_reversed = _plot_array_cmap_base_and_reverse(tool, operation)
    cmap_mixed = tool._batch_is_mixed(
        operation, lambda target: _plot_array_cmap_base_and_reverse(tool, target)[0]
    )
    reverse_mixed = tool._batch_is_mixed(
        operation, lambda target: _plot_array_cmap_base_and_reverse(tool, target)[1]
    )
    cmap_combo = erlab.interactive.colors.ColorMapComboBox(cmap_widget)
    tool._mark_editor_control(cmap_combo)
    cmap_combo.setObjectName("figureComposerPlotArrayCmapCombo")
    cmap_combo.default_cmap = None if cmap_mixed else cmap_base
    cmap_combo.ensure_populated()
    if cmap_mixed:
        with QtCore.QSignalBlocker(cmap_combo):
            tool._set_combo_mixed_placeholder(cmap_combo)
    elif cmap_combo.currentText() != cmap_base:
        with QtCore.QSignalBlocker(cmap_combo):
            cmap_combo.setCurrentText(cmap_base)
    cmap_reverse_check = tool._check_box(
        cmap_reversed,
        lambda checked: _update_current_cmap(tool, reverse=checked),
        parent=cmap_widget,
        mixed=reverse_mixed,
    )
    cmap_reverse_check.setText("Reverse")
    cmap_reverse_check.setObjectName("figureComposerPlotArrayCmapReverseCheck")
    tool._connect_editor_signal(
        cmap_combo,
        cmap_combo.activated,
        lambda _index, combo=cmap_combo: (
            None
            if tool._mixed_combo_text(combo.currentText())
            else _update_current_cmap(tool, base=combo.current_matplotlib_name())
        ),
    )
    cmap_combo.blockSignals(False)
    cmap_layout.addWidget(cmap_combo, 1)
    cmap_layout.addWidget(cmap_reverse_check)
    tool._add_form_row(
        layout,
        "Colormap",
        cmap_widget,
        "Colormap and reverse-colormap controls.",
    )

    norm_combo = tool._combo(
        _norm_combo_choices(operation.norm_name),
        tool._batch_combo_text(
            operation,
            lambda target: target.norm_name,
            _norm_combo_text,
        ),
        lambda text: _update_current_norm_name(tool, _norm_name_from_combo_text(text)),
        parent=page,
        mixed=tool._batch_is_mixed(
            operation, lambda target: _norm_combo_text(target.norm_name)
        ),
    )
    norm_combo.setObjectName("figureComposerPlotArrayNormCombo")
    tool._add_form_row(layout, "Norm", norm_combo, "Color normalization.")

    norm_fields = _norm_kwarg_fields(operation.norm_name)
    if "gamma" in norm_fields:
        gamma_mixed = tool._batch_is_mixed(
            operation, lambda target: _norm_gamma_value(target)
        )
        gamma_widget = erlab.interactive.colors.ColorMapGammaWidget(
            page,
            value=_norm_gamma_value(operation),
            spin_cls=erlab.interactive.utils.BetterSpinBox,
        )
        gamma_widget.setObjectName("figureComposerPlotArrayGammaWidget")
        tool._connect_editor_signal(
            gamma_widget,
            gamma_widget.valueChanged,
            lambda value: _update_current_norm_gamma(tool, value),
        )
        tool._add_form_row(
            layout,
            "Gamma",
            tool._mixed_value_widget(gamma_widget, mixed=gamma_mixed, parent=page),
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
        text, mixed = tool._batch_text(
            operation,
            _operation_field_getter(attr),
            lambda value: "" if value is None else str(value),
        )
        edit = tool._line_edit(text, parent=page)
        tool._apply_mixed_line_edit(edit, mixed)
        edit.setObjectName(f"figureComposerPlotArray{attr[0].upper()}{attr[1:]}Edit")
        tool._connect_line_edit_finished(
            edit,
            _norm_number_update_callback(tool, attr),
        )
        norm_number_widgets[attr] = (label, edit, tooltip)

    if "vmin" in norm_number_widgets and "vmax" in norm_number_widgets:
        tool._add_compound_form_row(
            layout,
            "Color limits",
            (
                norm_number_widgets.pop("vmin"),
                norm_number_widgets.pop("vmax"),
            ),
            "Lower and upper color-normalization bounds.",
        )
    if "vcenter" in norm_number_widgets and "halfrange" in norm_number_widgets:
        tool._add_compound_form_row(
            layout,
            "Center/range",
            (
                norm_number_widgets.pop("vcenter"),
                norm_number_widgets.pop("halfrange"),
            ),
            "Center and half-range for centered color normalization.",
        )
    for label, edit, tooltip in norm_number_widgets.values():
        tool._add_form_row(layout, label, edit, tooltip)

    if "clip" in norm_fields:
        clip_mixed = tool._batch_is_mixed(operation, lambda target: target.norm_clip)
        clip_combo = tool._combo(
            ["default", "False", "True"],
            None if clip_mixed else _norm_clip_text(operation.norm_clip),
            lambda text: tool._update_current_operation(
                norm_clip=_norm_clip_from_text(text)
            ),
            parent=page,
            mixed=clip_mixed,
        )
        clip_combo.setObjectName("figureComposerPlotArrayNormClipCombo")
        tool._add_form_row(
            layout,
            "Clip",
            clip_combo,
            "clip argument for the selected normalization object.",
        )

    norm_kwargs_text, norm_kwargs_mixed = tool._batch_text(
        operation, lambda target: target.norm_kwargs, _format_dict
    )
    norm_kwargs_edit = tool._line_edit(norm_kwargs_text, parent=page)
    tool._apply_mixed_line_edit(norm_kwargs_edit, norm_kwargs_mixed)
    norm_kwargs_edit.setObjectName("figureComposerPlotArrayNormKwargsEdit")
    tool._connect_line_edit_finished(
        norm_kwargs_edit,
        lambda text: _update_current_norm_kwargs(tool, text),
    )
    tool._add_form_row(
        layout,
        "Norm kwargs",
        norm_kwargs_edit,
        "Extra dict literal or keyword arguments for the norm constructor.",
    )

    tool._add_form_section(
        layout,
        "Colorbar",
        object_name="figureComposerPlotArrayColorsColorbarSection",
    )
    colorbar_mixed = tool._batch_is_mixed(operation, lambda target: target.colorbar)
    colorbar_combo = tool._combo(
        ["none", "right"],
        None if colorbar_mixed else operation.colorbar,
        lambda text: tool._update_current_operation(colorbar=text),
        parent=page,
        mixed=colorbar_mixed,
    )
    colorbar_combo.setObjectName("figureComposerPlotArrayColorbarCombo")
    tool._add_form_row(
        layout,
        "Colorbar",
        colorbar_combo,
        "Whether plot_array should draw a colorbar.",
    )

    colorbar_kwargs_text, colorbar_kwargs_mixed = tool._batch_text(
        operation, lambda target: target.colorbar_kw, _format_dict
    )
    colorbar_kwargs_edit = tool._line_edit(colorbar_kwargs_text, parent=page)
    tool._apply_mixed_line_edit(colorbar_kwargs_edit, colorbar_kwargs_mixed)
    colorbar_kwargs_edit.setObjectName("figureComposerPlotArrayColorbarKwEdit")
    tool._connect_line_edit_finished(
        colorbar_kwargs_edit,
        lambda text: tool._update_current_operation(colorbar_kw=_dict_from_text(text)),
    )
    tool._add_form_row(
        layout,
        "Colorbar kwargs",
        colorbar_kwargs_edit,
        "Dict literal or keyword arguments forwarded as colorbar_kw.",
    )
    return page


def _build_plot_array_advanced_page(
    tool: FigureComposerTool, operation: FigureOperationState
) -> QtWidgets.QWidget:
    page, layout = tool._new_step_form_page("figureComposerPlotArrayAdvancedPage")
    extra_text, extra_mixed = tool._batch_text(
        operation, lambda target: target.extra_kwargs, _format_dict
    )
    extra_edit = tool._line_edit(extra_text, parent=page)
    tool._apply_mixed_line_edit(extra_edit, extra_mixed)
    extra_edit.setObjectName("figureComposerPlotArrayExtraKwEdit")
    tool._connect_line_edit_finished(
        extra_edit,
        lambda text: tool._update_current_operation(extra_kwargs=_dict_from_text(text)),
    )
    tool._add_form_row(
        layout,
        "Extra kwargs",
        extra_edit,
        "Dict literal or keyword arguments forwarded to plot_array.",
    )
    return page


def _editor_sections(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    return (
        StepSection(
            "selection",
            "Selection",
            _build_plot_array_selection_page(tool, operation),
            "Choose the source data and selection reduced to one 2D image.",
        ),
        StepSection(
            "view",
            "View",
            _build_plot_array_view_page(tool, operation),
            "Choose image orientation, limits, and cropping.",
        ),
        StepSection(
            "colors",
            "Colors",
            _build_plot_array_colors_page(tool, operation),
            "Set colormap, normalization, and colorbar options.",
        ),
        StepSection(
            "advanced",
            "Other",
            _build_plot_array_advanced_page(tool, operation),
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
        case "selection":
            if _plot_array_selection_error(tool, operation) is not None:
                return "invalid"
            selection = _primary_selection(operation)
            labels = [
                label
                for label, value in (
                    ("isel", selection.isel),
                    ("qsel", selection.qsel),
                    ("mean", selection.mean_dims),
                )
                if value
            ]
            return ", ".join(labels) if labels else "none"
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
    uses_axes=lambda _operation: True,
    uses_source_section=lambda _operation: True,
    source_names=_source_names,
    build_source_editor=_build_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=lambda tool, operation, _figure, axs: _render_plot_array(
        tool, operation, axs
    ),
    code_lines=_plot_array_code_lines,
    required_imports=_required_imports,
)
