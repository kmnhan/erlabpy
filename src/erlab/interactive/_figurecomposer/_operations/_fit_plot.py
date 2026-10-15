"""Fit plot controls, rendering, and public code generation."""

from __future__ import annotations

import typing

from qtpy import QtWidgets

import erlab.plotting as eplt
from erlab.interactive._figurecomposer._code import _axes_code
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._fit_sources import FigureFitSourceSnapshot
from erlab.interactive._figurecomposer._model._gridspec import _gridspec_valid_axes_ids
from erlab.interactive._figurecomposer._model._sources import _valid_source_variable
from erlab.interactive._figurecomposer._model._state import (
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    SourceRequest,
)
from erlab.interactive._figurecomposer._rendering import _axes_from_selection
from erlab.interactive._figurecomposer._text import _code_kwargs, _RawCode
from erlab.interactive._figurecomposer._ui._fit_style import (
    add_fit_error_controls,
    add_fit_style_controls,
)
from erlab.interactive._figurecomposer._ui._operation_editor import StepSection

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._tool import FigureComposerTool
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


def _source(operation: FigureOperationState) -> str | None:
    return operation.sources[0] if operation.sources else None


def _has_weights(snapshot: FigureFitSourceSnapshot | None) -> bool:
    return snapshot is not None and any(
        str(name).endswith("modelfit_weights") for name in snapshot.native_dataset
    )


def _input_error_applies(
    context: FigureRecipeContext, operation: FigureOperationState, key: str
) -> bool:
    snapshot = context.source_payload(_source(operation) or "", FigureFitSourceSnapshot)
    if key.startswith("figureComposerFitPlotError"):
        return _has_weights(snapshot)
    for prefix in (
        "figureComposerFitPlotComponentLine:",
        "figureComposerFitPlotComponentBand:",
    ):
        if key.startswith(prefix):
            component = key.removeprefix(prefix).rsplit(":", 1)[0]
            return (
                operation.show_components
                and snapshot is not None
                and (
                    not snapshot.component_names
                    or component in snapshot.component_names
                )
            )
    return True


def _reconcile_sources(
    context: FigureRecipeContext, operation: FigureOperationState
) -> FigureOperationState:
    snapshot = context.source_payload(_source(operation) or "", FigureFitSourceSnapshot)
    if snapshot is None or not snapshot.component_names:
        return operation
    names = set(snapshot.component_names)
    updates = {
        field: {
            name: style
            for name, style in getattr(operation, field).items()
            if name in names
        }
        for field in ("fit_component_kw", "fit_component_band_kw")
        if not getattr(operation, field).keys() <= names
    }
    return operation.model_copy(update=updates) if updates else operation


def _create_operation(tool: FigureComposerTool) -> FigureOperationState:
    sources = tool.source_kind_names("fit")
    return FigureOperationState.fit_plot(
        source=sources[0] if sources else "", axes=tool._selected_axes_state()
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    source = _source(operation)
    label = tool._source_display_name(source) if source else "missing source"
    return f"Fit plot: {label}"


def _has_invalid_target(
    context: FigureRecipeContext, operation: FigureOperationState
) -> bool:
    if context.axes_selection_has_invalid_target(operation.axes):
        return True
    if operation.axes.expression:
        return False
    setup = context.recipe.setup
    if setup.layout_mode == "gridspec":
        return len(_gridspec_valid_axes_ids(setup, operation.axes.axes_ids)) != 1
    return len(operation.axes.valid_axes(setup)) != 1


def _build_source_editor(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> None:
    mixed = editor.batch_is_mixed(operation, _source)
    container = QtWidgets.QWidget(editor.source_controls)
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    def source_changed(source: str | None) -> None:
        if all(
            _source(target) == source for _index, target in editor.editable_operations()
        ):
            return
        editor.request_transform(
            lambda _index, target: _reconcile_sources(
                editor.context,
                target.model_copy(update={"sources": (source,) if source else ()}),
            ),
            rebuild_editor=True,
            defer_editor_rebuild=True,
        )

    combo = editor.source_combo(
        editor.context.source_kind_names("fit"),
        None if mixed else _source(operation),
        source_changed,
        parent=container,
        mixed=mixed,
    )
    combo.setObjectName("figureComposerFitPlotSourceCombo")
    layout.addWidget(combo, 1)
    add = QtWidgets.QPushButton("Add…", container)
    add.setObjectName("figureComposerFitPlotAddSourceButton")
    add.setEnabled(editor.binding.can_add_source(("fit",)))
    editor.connect_signal(
        add,
        add.clicked,
        lambda _checked=False: editor.add_source_requested.emit(("fit",)),
    )
    layout.addWidget(add)
    editor.add_source_row(
        "Fit result", container, "Stored result used by this fit plot."
    )


def _build_editor_sections(
    editor: FigureOperationEditor, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    snapshot = editor.context.source_payload(
        _source(operation) or "", FigureFitSourceSnapshot
    )
    data_page, data_layout = editor.new_form_page("figureComposerFitPlotDataPage")
    show = editor.check_box(
        operation.show_data,
        lambda value: editor.request_update(show_data=value),
        parent=data_page,
        mixed=editor.batch_is_mixed(operation, lambda target: target.show_data),
    )
    show.setObjectName("figureComposerFitPlotShowData")
    editor.add_form_row(
        data_layout, "Show data", show, "Plot the observations used in the fit."
    )
    has_weights = _has_weights(snapshot)
    if has_weights:
        editor.add_form_section(
            data_layout, "Error bars", object_name="figureComposerFitPlotErrorSection"
        )
        errors = editor.check_box(
            operation.fit_show_yerr,
            lambda value: editor.request_update(fit_show_yerr=value),
            parent=data_page,
            mixed=editor.batch_is_mixed(operation, lambda target: target.fit_show_yerr),
        )
        errors.setObjectName("figureComposerFitPlotShowYerr")
        editor.add_form_row(
            data_layout, "Error bars", errors, "Plot inverse fit weights as error bars."
        )
        add_fit_error_controls(editor, operation, data_page, data_layout)
        editor.add_form_section(
            data_layout,
            "Data style",
            object_name="figureComposerFitPlotDataStyleSection",
        )
    add_fit_style_controls(
        editor,
        operation,
        data_page,
        data_layout,
        "fit_data_kw",
        prefix="figureComposerFitPlotData",
        line=True,
        errorbars=has_weights,
    )

    fit_page, fit_layout = editor.new_form_page("figureComposerFitPlotBestFitPage")
    editor.add_form_row(
        fit_layout,
        "Show best fit",
        editor.check_box(
            operation.show_fit,
            lambda value: editor.request_update(show_fit=value),
            parent=fit_page,
            mixed=editor.batch_is_mixed(operation, lambda target: target.show_fit),
        ),
        "Plot the best-fit curve.",
    )
    sampling = editor.combo(
        ("Original sampling", "Smooth curve"),
        editor.batch_combo_text(
            operation,
            lambda target: target.fit_numpoints is not None,
            lambda smooth: "Smooth curve" if smooth else "Original sampling",
        ),
        lambda text: editor.request_update_rebuild(
            fit_numpoints=500 if text == "Smooth curve" else None
        ),
        parent=fit_page,
        mixed=editor.batch_is_mixed(
            operation, lambda target: target.fit_numpoints is not None
        ),
        enabled=snapshot is not None and snapshot.supports_resampling,
    )
    sampling.setObjectName("figureComposerFitPlotSampling")
    editor.add_form_row(
        fit_layout,
        "Sampling",
        sampling,
        "Evaluate the fitted model while keeping measured points "
        "at their original coordinates.",
    )
    count = QtWidgets.QSpinBox(fit_page)
    count.setRange(2, 1_000_000)
    count.setValue(operation.fit_numpoints or 500)
    count.setKeyboardTracking(False)
    count.setEnabled(operation.fit_numpoints is not None)
    count.setObjectName("figureComposerFitPlotPointCount")
    editor.mark_control(count)
    editor.connect_signal(
        count,
        count.valueChanged,
        lambda value: editor.request_update(fit_numpoints=value),
    )
    editor.add_form_row(
        fit_layout,
        "Points",
        editor.mixed_value_widget(
            count,
            mixed=editor.batch_is_mixed(operation, lambda target: target.fit_numpoints),
        ),
        "Number of evenly spaced model-evaluation positions.",
    )
    add_fit_style_controls(
        editor,
        operation,
        fit_page,
        fit_layout,
        "fit_line_kw",
        prefix="figureComposerFitPlotBestFit",
        line=True,
    )

    components_page, components_layout = editor.new_form_page(
        "figureComposerFitPlotComponentsPage"
    )
    component_settings = QtWidgets.QWidget(components_page)
    component_settings.setObjectName("figureComposerFitPlotComponentSettings")
    component_settings_layout = QtWidgets.QFormLayout(component_settings)
    component_settings_layout.setContentsMargins(0, 0, 0, 0)
    components_mixed = editor.batch_is_mixed(
        operation, lambda target: target.show_components
    )
    component_settings.setVisible(operation.show_components and not components_mixed)

    def show_components_changed(value: bool) -> None:
        component_settings.setVisible(value)
        if value:
            editor.request_update(show_components=value)
        else:
            # Discard unfinished edits so hidden inputs cannot block the fit plot.
            for control in component_settings.findChildren(QtWidgets.QWidget):
                editor.clear_input_error(control)
            editor.request_update_rebuild(show_components=value)

    components = editor.check_box(
        operation.show_components,
        show_components_changed,
        parent=components_page,
        mixed=components_mixed,
    )
    components.setObjectName("figureComposerFitPlotShowComponents")
    components.setEnabled(snapshot is not None)
    editor.add_form_row(
        components_layout,
        "Show components",
        components,
        "Plot the fitted model's individual components.",
    )
    component_options = []
    for _index, target in editor.editable_operations():
        target_snapshot = editor.context.source_payload(
            _source(target) or "", FigureFitSourceSnapshot
        )
        names = tuple(
            dict.fromkeys(
                (
                    *(
                        ()
                        if target_snapshot is None
                        else target_snapshot.component_names
                    ),
                    *target.fit_component_kw,
                    *target.fit_component_band_kw,
                )
            )
        )
        component_options.append(names)
    if not component_options:
        component_options = [() if snapshot is None else snapshot.component_names]
    component_names = tuple(
        name
        for name in component_options[0]
        if all(name in options for options in component_options[1:])
    )
    if component_names:
        component_stack = QtWidgets.QStackedWidget(component_settings)
        component_stack.setObjectName("figureComposerFitPlotComponentStyles")
        selector = editor.combo(
            component_names,
            component_names[0],
            lambda text: component_stack.setCurrentIndex(component_names.index(text)),
            parent=component_settings,
        )
        selector.setObjectName("figureComposerFitPlotComponent")
        editor.add_form_row(
            component_settings_layout,
            "Component",
            selector,
            "Choose the component to style.",
        )
        for index, name in enumerate(component_names):
            component_page = QtWidgets.QTabWidget(component_stack)
            component_page.setObjectName(f"figureComposerFitPlotComponentStyle{index}")
            component_page.setProperty("componentName", name)
            for title, field, prefix, line in (
                (
                    "Line",
                    "fit_component_kw",
                    "figureComposerFitPlotComponentLine",
                    True,
                ),
                (
                    "Confidence band",
                    "fit_component_band_kw",
                    "figureComposerFitPlotComponentBand",
                    False,
                ),
            ):
                style_page = QtWidgets.QWidget(component_page)
                style_layout = QtWidgets.QFormLayout(style_page)
                add_fit_style_controls(
                    editor,
                    operation,
                    style_page,
                    style_layout,
                    field,
                    prefix=prefix,
                    line=line,
                    component=name,
                )
                component_page.addTab(style_page, title)
            component_stack.addWidget(component_page)
        component_settings_layout.addRow(component_stack)
    components_layout.addRow(component_settings)

    band_page, band_layout = editor.new_form_page("figureComposerFitPlotConfidencePage")
    band = editor.check_box(
        operation.fit_sigma is not None,
        lambda value: editor.request_update_rebuild(fit_sigma=1.0 if value else None),
        parent=band_page,
        mixed=editor.batch_is_mixed(
            operation, lambda target: target.fit_sigma is not None
        ),
    )
    band.setObjectName("figureComposerFitPlotShowConfidence")
    band.setEnabled(snapshot is not None and snapshot.supports_confidence)
    editor.add_form_row(
        band_layout,
        "Show band",
        band,
        "Approximate pointwise mean uncertainty from fitted parameter covariance.",
    )
    level = QtWidgets.QDoubleSpinBox(band_page)
    level.setObjectName("figureComposerFitPlotSigmaSpin")
    level.setDecimals(6)
    level.setRange(0.000001, 1000.0)
    level.setValue(operation.fit_sigma or 1.0)
    level.setKeyboardTracking(False)
    level.setEnabled(operation.fit_sigma is not None)
    editor.mark_control(level)
    editor.connect_signal(
        level,
        level.valueChanged,
        lambda value: editor.request_update(fit_sigma=float(value)),
    )
    editor.add_form_row(
        band_layout,
        "Sigma",
        editor.mixed_value_widget(
            level,
            mixed=editor.batch_is_mixed(operation, lambda target: target.fit_sigma),
        ),
        "Values of 1 or more specify sigma multiples; values between 0 and 1 "
        "specify the pointwise probability.",
    )
    add_fit_style_controls(
        editor,
        operation,
        band_page,
        band_layout,
        "fit_band_kw",
        prefix="figureComposerFitPlotBand",
        line=False,
    )
    return (
        StepSection(
            "fit_data",
            "Data",
            data_page,
            "Measured data and measurement uncertainties.",
        ),
        StepSection(
            "fit_line", "Best fit", fit_page, "Best fit sampling and line style."
        ),
        StepSection(
            "fit_components",
            "Components",
            components_page,
            "Individual model components and their line styles.",
        ),
        StepSection(
            "fit_band", "Confidence band", band_page, "Pointwise model confidence band."
        ),
    )


def _plot_kwargs(operation: FigureOperationState) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {}
    if operation.fit_numpoints is not None:
        kwargs["numpoints"] = operation.fit_numpoints
    if operation.fit_sigma is not None:
        kwargs["sigma"] = operation.fit_sigma
    if not operation.show_data:
        kwargs["show_data"] = False
    if not operation.fit_show_yerr:
        kwargs["show_yerr"] = False
    if not operation.show_fit:
        kwargs["show_fit"] = False
    if operation.show_components:
        kwargs["show_components"] = True
    for field, key in (
        ("fit_data_kw", "data_kws"),
        ("fit_line_kw", "fit_kws"),
        ("fit_band_kw", "band_kws"),
        ("fit_component_kw", "component_kws"),
        ("fit_component_band_kw", "component_band_kws"),
    ):
        style = getattr(operation, field)
        if style:
            kwargs[key] = dict(style)
    return kwargs


def _render(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    _figure: Figure,
    axs: typing.Any,
) -> None:
    source = _source(operation)
    snapshot = tool.source_payload(source or "", FigureFitSourceSnapshot)
    if source is None or snapshot is None:
        raise FigureComposerInputError("Choose an available fit result source.")
    axis = _axes_from_selection(tool, operation.axes, axs, for_plot_slices=False)
    dataset = typing.cast(
        "xr.Dataset", tool._source_value(source, evaluate=_requires_model(operation))
    )
    kwargs = _plot_kwargs(operation)
    eplt.plot_fit(dataset, ax=typing.cast("Axes", axis), **kwargs)


def _code_lines(tool: FigureComposerTool, operation: FigureOperationState) -> list[str]:
    source = _source(operation)
    snapshot = tool.source_payload(source or "", FigureFitSourceSnapshot)
    if source is None or snapshot is None:
        raise FigureComposerInputError("Choose an available fit result source.")
    source_code = _valid_source_variable(source)
    kwargs = {
        "ax": _RawCode(
            _axes_code(tool._document, operation.axes, for_plot_slices=False)
        ),
        **_plot_kwargs(operation),
    }
    return [f"eplt.plot_fit({source_code}, {_code_kwargs(kwargs)})"]


def _requires_model(operation: FigureOperationState) -> bool:
    return (
        operation.fit_numpoints is not None
        or operation.fit_sigma is not None
        or operation.show_components
    )


def _source_requests(
    operation: FigureOperationState, _available_names: Iterable[str]
) -> tuple[SourceRequest, ...]:
    return tuple(
        SourceRequest(name, _requires_model(operation)) for name in operation.sources
    )


SPEC = OperationSpec(
    kind=FigureOperationKind.FIT_PLOT,
    add_actions=(
        AddStepActionSpec(
            "fit_plot",
            "Fit plot",
            "Plot data, a best fit, and a confidence band.",
            _create_operation,
        ),
    ),
    display_text=_display_text,
    tooltip=lambda _tool, _operation: "Runs erlab.plotting.plot_fit.",
    target_text=lambda tool, operation: tool._axes_target_text(operation.axes),
    has_invalid_target=_has_invalid_target,
    uses_source_section=lambda _operation: True,
    build_source_editor=_build_source_editor,
    build_editor_sections=_build_editor_sections,
    section_summary=lambda _tool, _key, _operation: "",
    render=_render,
    code_lines=_code_lines,
    accepted_source_kinds=("fit",),
    source_requests=_source_requests,
    reconcile_sources=_reconcile_sources,
    input_error_applies=_input_error_applies,
    required_imports=lambda _tool, _operation: ("import erlab.plotting as eplt",),
)
