"""Fit parameter selection, error bars, and public plotting code."""

from __future__ import annotations

import typing

import numpy as np
from qtpy import QtWidgets

from erlab.interactive._figurecomposer._code import _axes_code
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._fit_sources import (
    FigureFitParameterSourceSnapshot,
    FigureFitSourceSnapshot,
)
from erlab.interactive._figurecomposer._model._sources import _valid_source_variable
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
)
from erlab.interactive._figurecomposer._operations._fit_plot import _has_invalid_target
from erlab.interactive._figurecomposer._rendering import _axes_from_selection
from erlab.interactive._figurecomposer._text import _code_kwargs
from erlab.interactive._figurecomposer._ui._fit_style import (
    add_fit_error_controls,
    add_fit_style_controls,
)
from erlab.interactive._figurecomposer._ui._operation_editor import StepSection

if typing.TYPE_CHECKING:
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._tool import FigureComposerTool
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )

_KINDS = ("fit_parameters", "fit")


def _source(operation: FigureOperationState) -> str | None:
    return operation.sources[0] if operation.sources else None


def _dataset(
    context: FigureRecipeContext, operation: FigureOperationState
) -> xr.Dataset | None:
    alias = _source(operation) or ""
    for cls in (FigureFitParameterSourceSnapshot, FigureFitSourceSnapshot):
        snapshot = context.source_payload(alias, cls)
        if isinstance(
            snapshot, FigureFitParameterSourceSnapshot | FigureFitSourceSnapshot
        ):
            return snapshot.native_dataset
    return None


def _parameters(dataset: xr.Dataset | None) -> tuple[str, ...]:
    if dataset is None or "param" not in dataset.coords:
        return ()
    return tuple(str(value) for value in dataset.param.values)


def _selected_parameters(
    dataset: xr.Dataset, operation: FigureOperationState
) -> tuple[str, ...]:
    names = _parameters(dataset)
    if operation.fit_parameter is None:
        if names:
            return names
        raise FigureComposerInputError("The source has no fitted parameters.")
    if operation.fit_parameter not in names:
        raise FigureComposerInputError("Choose an available fit parameter.")
    return (operation.fit_parameter,)


def _dimension(dataset: xr.Dataset) -> str | None:
    dimensions = tuple(
        dim for dim in dataset.modelfit_coefficients.dims if dim != "param"
    )
    if len(dimensions) > 1:
        raise FigureComposerInputError("Choose parameters from a one-dimensional scan.")
    return str(dimensions[0]) if dimensions else None


def _source_names(context: FigureRecipeContext) -> tuple[str, ...]:
    return tuple(name for kind in _KINDS for name in context.source_kind_names(kind))


def _create_operation(tool: FigureComposerTool) -> FigureOperationState:
    names = _source_names(tool._document)
    return FigureOperationState.fit_parameters(
        source=names[0] if names else "", axes=tool._selected_axes_state()
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    label = tool._source_display_name(_source(operation) or "")
    return f"Fit parameters: {operation.fit_parameter or label}"


def _build_source_editor(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> None:
    mixed = editor.batch_is_mixed(operation, _source)
    container = QtWidgets.QWidget(editor.source_controls)
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    def source_changed(source: str | None) -> None:
        def update(_index: int, target: FigureOperationState) -> FigureOperationState:
            changed = target.model_copy(update={"sources": (source,) if source else ()})
            names = _parameters(_dataset(editor.context, changed))
            if changed.fit_parameter is not None and changed.fit_parameter not in names:
                changed = changed.model_copy(update={"fit_parameter": None})
            return changed

        editor.request_transform(update, rebuild_editor=True, defer_editor_rebuild=True)

    combo = editor.source_combo(
        _source_names(editor.context),
        None if mixed else _source(operation),
        source_changed,
        parent=container,
        mixed=mixed,
    )
    combo.setObjectName("figureComposerFitParametersSourceCombo")
    layout.addWidget(combo, 1)
    add = QtWidgets.QPushButton("Add…", container)
    add.setObjectName("figureComposerFitParametersAddSourceButton")
    add.setEnabled(editor.binding.can_add_source(_KINDS))
    editor.connect_signal(
        add,
        add.clicked,
        lambda _checked=False: editor.add_source_requested.emit(_KINDS),
    )
    layout.addWidget(add)
    editor.add_source_row("Fit parameters", container, "Stored parameters from ftool.")


def _build_editor_sections(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    page, layout = editor.new_form_page("figureComposerFitParametersPage")
    names = _parameters(_dataset(editor.context, operation))
    mixed = editor.batch_is_mixed(operation, lambda target: target.fit_parameter)
    values = ("", *names)
    labels = ("All parameters", *names)
    unavailable = (
        not mixed
        and operation.fit_parameter is not None
        and operation.fit_parameter not in names
    )
    if unavailable:
        values = (*values, typing.cast("str", operation.fit_parameter))
        labels = (*labels, f"{operation.fit_parameter} (unavailable)")
    parameter = editor.labeled_combo(
        values,
        labels,
        None if mixed else operation.fit_parameter or "",
        lambda value: editor.request_update(fit_parameter=value or None),
        parent=page,
        mixed=mixed,
    )
    if unavailable:
        item = typing.cast("typing.Any", parameter.model()).item(
            parameter.currentIndex()
        )
        if item is not None:
            item.setEnabled(False)
    parameter.setObjectName("figureComposerFitParametersParameterCombo")
    editor.add_form_row(
        layout, "Parameter", parameter, "Choose fitted parameter values."
    )
    errors = editor.check_box(
        operation.fit_show_yerr,
        lambda value: editor.request_update(fit_show_yerr=value),
        parent=page,
        mixed=editor.batch_is_mixed(operation, lambda target: target.fit_show_yerr),
    )
    errors.setObjectName("figureComposerFitParametersShowYerr")
    editor.add_form_row(
        layout, "Error bars", errors, "Show fitted parameter standard errors."
    )
    add_fit_style_controls(
        editor,
        operation,
        page,
        layout,
        "fit_data_kw",
        prefix="figureComposerFitParametersStyle",
        line=True,
        errorbars=True,
    )
    add_fit_error_controls(editor, operation, page, layout)
    return (
        StepSection(
            "fit_parameters",
            "Parameters",
            page,
            "Parameter values and standard errors.",
        ),
    )


def _plot_kwargs(operation: FigureOperationState, label: str) -> dict[str, typing.Any]:
    return {"label": label, **operation.fit_data_kw}


def _render(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    _figure: Figure,
    axs: typing.Any,
) -> None:
    dataset = _dataset(tool._document, operation)
    if dataset is None:
        raise FigureComposerInputError("Choose an available fit parameter source.")
    names = _selected_parameters(dataset, operation)
    dimension = _dimension(dataset)
    axis = typing.cast(
        "Axes", _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)
    )
    groups = (names,) if dimension is None else tuple((name,) for name in names)
    for group in groups:
        selected: str | list[str] = list(group) if dimension is None else group[0]
        values = dataset.modelfit_coefficients.sel(param=selected)
        errors = dataset.get("modelfit_stderr")
        yerr = None
        if operation.fit_show_yerr and errors is not None:
            errors = errors.sel(param=selected)
            yerr = errors.where(np.isfinite(errors) & (errors >= 0))
        axis.errorbar(
            list(group) if dimension is None else dataset[dimension].values,
            values.values,
            yerr=None if yerr is None else yerr.values,
            **_plot_kwargs(operation, group[0] if len(group) == 1 else "Parameters"),
        )


def _code_lines(tool: FigureComposerTool, operation: FigureOperationState) -> list[str]:
    dataset = _dataset(tool._document, operation)
    if dataset is None:
        raise FigureComposerInputError("Choose an available fit parameter source.")
    names = _selected_parameters(dataset, operation)
    dimension = _dimension(dataset)
    source = _valid_source_variable(_source(operation) or "")
    axis = _axes_code(tool._document, operation.axes, for_plot_slices=False)
    groups = (names,) if dimension is None else tuple((name,) for name in names)
    lines = []
    for group in groups:
        selected = repr(list(group) if dimension is None else group[0])
        x = (
            repr(list(group))
            if dimension is None
            else f"{source}[{dimension!r}].values"
        )
        y = f"{source}.modelfit_coefficients.sel(param={selected}).values"
        kwargs = _plot_kwargs(operation, group[0] if len(group) == 1 else "Parameters")
        arguments = f"{x}, {y}"
        if operation.fit_show_yerr and "modelfit_stderr" in dataset:
            errors = f"{source}.modelfit_stderr.sel(param={selected})"
            arguments += (
                f", yerr={errors}.where(np.isfinite({errors}) & ({errors} >= 0)).values"
            )
        lines.append(f"{axis}.errorbar({arguments}, {_code_kwargs(kwargs)})")
    return lines


SPEC = OperationSpec(
    kind=FigureOperationKind.FIT_PARAMETERS,
    add_actions=(
        AddStepActionSpec(
            "fit_parameters",
            "Fit parameters",
            "Plot fitted parameter values and standard errors.",
            _create_operation,
        ),
    ),
    display_text=_display_text,
    tooltip=lambda _tool, _operation: (
        "Plot fitted parameters with Matplotlib Axes.errorbar."
    ),
    target_text=lambda tool, operation: tool._axes_target_text(operation.axes),
    has_invalid_target=_has_invalid_target,
    uses_source_section=lambda _operation: True,
    build_source_editor=_build_source_editor,
    build_editor_sections=_build_editor_sections,
    section_summary=lambda _tool, _key, _operation: "",
    render=_render,
    code_lines=_code_lines,
    accepted_source_kinds=_KINDS,
    required_imports=lambda _tool, _operation: ("import numpy as np",),
)
