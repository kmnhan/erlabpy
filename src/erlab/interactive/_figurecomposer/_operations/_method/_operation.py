"""Operation registry integration for curated method calls."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._model._operation_metadata import (
    operation_uses_axes,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureAxesSelectionState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    _empty_source_editor,
    _uses_no_source_section,
)
from erlab.interactive._figurecomposer._operations._method._catalog import (
    _FAMILY_LABELS,
    _FAMILY_TOOLTIPS,
    MethodTargetDomain,
    _method_display,
    _method_spec,
    _method_specs,
)
from erlab.interactive._figurecomposer._operations._method._editor import (
    _build_method_editor,
)
from erlab.interactive._figurecomposer._operations._method._execution import (
    _first_live_axis,
    _method_code,
    _render_method,
    _required_imports,
)
from erlab.interactive._figurecomposer._operations._method._state import (
    _default_method_args,
    _loaded_operation,
)

if typing.TYPE_CHECKING:
    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext
    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _method_operation(
    tool: FigureComposerTool, family: FigureMethodFamily
) -> FigureOperationState:
    spec = next(iter(_method_specs(family).values()))
    axes = (
        tool._selected_axes_state()
        if spec.target_domain == MethodTargetDomain.AXES
        else FigureAxesSelectionState(axes=())
    )
    return FigureOperationState.method(
        family=family,
        name=spec.name,
        label=spec.label,
        axes=axes,
        args=_default_method_args(spec, _first_live_axis(tool, axes)),
    )


def _method_add_action(family: FigureMethodFamily) -> AddStepActionSpec:
    def create_operation(tool: FigureComposerTool) -> FigureOperationState:
        return _method_operation(tool, family)

    return AddStepActionSpec(
        action_id=f"method:{family.value}",
        text=_FAMILY_LABELS[family],
        tooltip=_FAMILY_TOOLTIPS[family],
        create_operation=create_operation,
    )


def _target_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    spec = _method_spec(operation)
    if spec.target_domain == MethodTargetDomain.FIGURE:
        return "Figure"
    if spec.target_domain == MethodTargetDomain.NONE:
        return "none"
    return tool._axes_target_text(operation.axes)


def _has_invalid_target(
    context: FigureRecipeContext, operation: FigureOperationState
) -> bool:
    return operation_uses_axes(operation) and context.axes_selection_has_invalid_target(
        operation.axes
    )


def _display_text(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return _method_display(operation)


def _tooltip(tool: FigureComposerTool, operation: FigureOperationState) -> str:
    spec = _method_spec(operation)
    return f"{spec.tooltip}\nTargets: {_target_text(tool, operation)}"


def _section_summary(
    tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    if key == "axes":
        return _target_text(tool, operation)
    if key == "method":
        return ""
    return ""


SPEC = OperationSpec(
    kind=FigureOperationKind.METHOD,
    add_actions=(
        _method_add_action(FigureMethodFamily.ERLAB),
        _method_add_action(FigureMethodFamily.AXES),
        _method_add_action(FigureMethodFamily.FIGURE),
    ),
    display_text=_display_text,
    tooltip=_tooltip,
    target_text=_target_text,
    has_invalid_target=_has_invalid_target,
    uses_source_section=_uses_no_source_section,
    build_source_editor=_empty_source_editor,
    build_editor_sections=_build_method_editor,
    section_summary=_section_summary,
    render=_render_method,
    code_lines=_method_code,
    required_imports=_required_imports,
    loaded_operation=_loaded_operation,
)
