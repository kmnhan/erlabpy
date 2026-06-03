"""Trusted custom-code operation editor and renderer."""

from __future__ import annotations

import ast
import typing

from qtpy import QtCore, QtWidgets

from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_axis_code_tuple,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._operations._base import (
    AddStepActionSpec,
    OperationSpec,
    StepSection,
    _empty_source_editor,
    _empty_source_names,
    _no_invalid_target,
    _uses_no_axes,
    _uses_no_source_section,
)
from erlab.interactive._figurecomposer._rendering import _source_namespace
from erlab.interactive._figurecomposer._state import (
    FigureOperationKind,
    FigureOperationState,
)

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _build_custom_code_editor(
    tool: FigureComposerTool, operation: FigureOperationState
) -> list[tuple[str, str, QtWidgets.QWidget]]:
    page, layout = tool._new_step_form_page("figureComposerCodePage")
    tool.operation_editor = page
    tool.operation_editor_layout = layout
    trust = tool._check_box(
        operation.trusted,
        lambda checked: tool._update_current_operation(trusted=checked),
    )
    tool._add_form_row(
        tool.operation_editor_layout,
        "Trusted",
        trust,
        "Allow this custom Python step to execute during rendering.",
    )

    code_edit = QtWidgets.QPlainTextEdit(tool.operation_editor)
    code_edit.setPlainText(operation.code)
    code_edit.setMinimumHeight(160)
    code_edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    code_edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    tool._connect_plain_text_changed(
        code_edit,
        lambda text: tool._update_current_operation(code=text),
    )
    tool._add_form_row(
        tool.operation_editor_layout,
        "Code",
        code_edit,
        "Python code executed with fig, axs, ax, plt, eplt, np, xr, and source "
        "variables.",
    )
    return [("code", "Code", page)]


def _render_custom(
    tool: FigureComposerTool,
    operation: FigureOperationState,
    fig: Figure,
    axs: typing.Any,
) -> None:
    if not operation.trusted or not operation.code.strip():
        return
    namespace = _source_namespace(tool, fig, axs)
    # Custom code is the explicit trusted escape hatch in the recipe pipeline.
    exec(operation.code, namespace)  # noqa: S102


def _create_custom_operation(_tool: FigureComposerTool) -> FigureOperationState:
    return FigureOperationState.custom(label="custom code", code="", trusted=True)


def _display_text(_tool: FigureComposerTool, operation: FigureOperationState) -> str:
    return f"Python: {operation.label}"


def _tooltip(_tool: FigureComposerTool, _operation: FigureOperationState) -> str:
    return "Runs trusted custom Python code.\nTargets: none"


def _editor_sections(
    tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[StepSection, ...]:
    return tuple(
        StepSection(
            key,
            title,
            page,
            "Edit the trusted custom Python code for this step.",
        )
        for key, title, page in _build_custom_code_editor(tool, operation)
    )


def _section_summary(
    _tool: FigureComposerTool, key: str, operation: FigureOperationState
) -> str:
    if key == "code":
        return "trusted" if operation.trusted else "not trusted"
    return ""


def _code_lines(tool: FigureComposerTool, operation: FigureOperationState) -> list[str]:
    code = operation.code.strip()
    if not operation.trusted or not code:
        return []
    names = _custom_code_names(code)
    lines: list[str] = []
    if "axs" in names:
        lines.extend(_custom_axes_alias_lines(tool))
    if "ax" in names:
        lines.append(f"ax = {_custom_first_axis_code(tool)}")
    lines.extend(code.splitlines())
    return lines


def _required_imports(
    _tool: FigureComposerTool, operation: FigureOperationState
) -> tuple[str, ...]:
    code = operation.code.strip()
    if not operation.trusted or not code:
        return ()
    names = _custom_code_names(code)
    imports: list[str] = []
    if "np" in names:
        imports.append("import numpy as np")
    if "xr" in names:
        imports.append("import xarray as xr")
    if "eplt" in names:
        imports.append("import erlab.plotting as eplt")
    return tuple(imports)


def _custom_code_names(code: str) -> frozenset[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset()
    return frozenset(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def _custom_axes_alias_lines(tool: FigureComposerTool) -> list[str]:
    setup = tool._recipe.setup
    if setup.layout_mode != "gridspec":
        return []
    axes_ids = _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup))
    axes_code = _gridspec_axis_code_tuple(setup, axes_ids)
    lines = ["axs = {"]
    lines.extend(
        f"    {axis_id!r}: {axis_code},"
        for axis_id, axis_code in zip(axes_ids, axes_code, strict=True)
    )
    lines.append("}")
    return lines


def _custom_first_axis_code(tool: FigureComposerTool) -> str:
    setup = tool._recipe.setup
    if setup.layout_mode != "gridspec":
        return "axs[0, 0]"
    axes_ids = _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup))
    axes_code = _gridspec_axis_code_tuple(setup, axes_ids[:1])
    return axes_code[0] if axes_code else "None"


def _loaded_operation(operation: FigureOperationState) -> FigureOperationState:
    return operation.model_copy(update={"trusted": False})


SPEC = OperationSpec(
    kind=FigureOperationKind.CUSTOM,
    add_actions=(
        AddStepActionSpec(
            action_id=FigureOperationKind.CUSTOM.value,
            text="Python",
            tooltip="Add a trusted custom Python step.",
            create_operation=_create_custom_operation,
        ),
    ),
    display_text=_display_text,
    tooltip=_tooltip,
    target_text=lambda _tool, _operation: "none",
    has_invalid_target=_no_invalid_target,
    uses_axes=_uses_no_axes,
    uses_source_section=_uses_no_source_section,
    source_names=_empty_source_names,
    build_source_editor=_empty_source_editor,
    build_editor_sections=_editor_sections,
    section_summary=_section_summary,
    render=_render_custom,
    code_lines=_code_lines,
    required_imports=_required_imports,
    loaded_operation=_loaded_operation,
)
