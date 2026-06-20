"""Trusted custom-code operation editor and renderer."""

from __future__ import annotations

import ast
import typing

from qtpy import QtCore, QtWidgets

import erlab.interactive.utils
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

_CUSTOM_CODE_SETTLE_DELAY_MS = 900


def _connect_custom_code_editor(
    tool: FigureComposerTool, code_edit: erlab.interactive.utils.PythonCodeEditor
) -> None:
    code_edit.set_text_editing_settle_delay(_CUSTOM_CODE_SETTLE_DELAY_MS)
    code_edit.reset_text_editing_activity()
    code_edit_any = typing.cast("typing.Any", code_edit)
    pending_code = [code_edit.toPlainText()]
    pending_operation_ids: list[tuple[str, ...]] = [()]
    pending_dirty = [False]

    def capture_pending_code(*, refresh_operation_ids: bool) -> None:
        pending_code[0] = code_edit.toPlainText()
        if refresh_operation_ids:
            pending_operation_ids[0] = tuple(
                operation.operation_id
                for _index, operation in tool._editable_operations()
            )
        pending_dirty[0] = True

    def queue_commit() -> None:
        if tool._editor_control_signal_allowed(code_edit):
            capture_pending_code(refresh_operation_ids=True)

    def queue_contents_change(
        _position: int, chars_removed: int, chars_added: int
    ) -> None:
        if chars_removed or chars_added:
            queue_commit()

    def commit_code(*, render: bool) -> None:
        if not pending_dirty[0]:
            return
        if tool._closing or not erlab.interactive.utils.qt_is_valid(tool):
            return
        operation_ids = pending_operation_ids[0]
        if not operation_ids:
            return
        pending_dirty[0] = False
        code = pending_code[0]
        render_valid_code = render and code_edit.has_valid_python_syntax(code)
        changed = tool._update_operations_by_ids(
            operation_ids,
            lambda _index, target: target.model_copy(update={"code": code}),
            render=render_valid_code,
        )
        if changed and not render_valid_code:
            tool.sigInfoChanged.emit()

    def commit_settled_code(_code: str) -> None:
        commit_code(render=True)

    def flush_pending_commit(*, render: bool = False) -> None:
        if code_edit.text_editing_active():
            code_edit.reset_text_editing_activity()
            if pending_dirty[0]:
                pending_code[0] = code_edit.toPlainText()
        commit_code(render=render)

    tool._mark_editor_control(code_edit)
    document = code_edit.document()
    document.contentsChange.connect(queue_contents_change)
    code_edit.sigTextEditingSettled.connect(commit_settled_code)
    code_edit_any._figure_composer_custom_code_commit_handlers = (
        queue_contents_change,
        commit_settled_code,
        flush_pending_commit,
    )
    code_edit_any._figure_composer_custom_code_flush_pending_commit = (
        flush_pending_commit
    )


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

    code_edit = erlab.interactive.utils.PythonCodeEditor(tool.operation_editor)
    code_edit.setObjectName("figureComposerCustomCodeEdit")
    code_edit.setPlainText(operation.code)
    code_edit.setPlaceholderText("# Write Python code here")
    code_edit.setMinimumHeight(220)
    code_edit.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
    code_edit.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding,
        QtWidgets.QSizePolicy.Policy.Expanding,
    )
    code_edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    code_edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    _connect_custom_code_editor(tool, code_edit)
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
    axes_code = _gridspec_axis_code_tuple(
        setup, axes_ids, reserved_names=tool._source_names()
    )
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
    axes_code = _gridspec_axis_code_tuple(
        setup, axes_ids[:1], reserved_names=tool._source_names()
    )
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
