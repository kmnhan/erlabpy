"""Standalone Python code generation for Figure Composer recipes."""

from __future__ import annotations

import ast
import typing

from erlab.interactive._figurecomposer._code import _selection_code, _setup_code
from erlab.interactive._figurecomposer._defaults import (
    _style_code_lines,
    _style_required_imports,
)
from erlab.interactive._figurecomposer._model._sources import (
    _source_has_selection,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureDataSelectionState,
    FigureOperationKind,
)
from erlab.interactive._figurecomposer._operations import _registry

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def generated_code(
    tool: FigureComposerTool,
    *,
    skip_source_selection_names: frozenset[str] = frozenset(),
    source_name_map: Mapping[str, str] | None = None,
) -> str:
    invalid_indices = tool._invalid_operation_indices()
    if invalid_indices:
        if any(
            tool.operation_editor.has_input_error(
                tool._document.recipe.operations[index]
            )
            for index in invalid_indices
        ):
            raise ValueError(
                "Cannot generate code until invalid step inputs are fixed."
            )
        raise ValueError(
            "Cannot generate code until all enabled steps target axes in the "
            "current figure layout"
        )
    lines = ["import matplotlib.pyplot as plt"]
    required_imports: list[str] = []
    for operation in tool._document.recipe.operations:
        if not operation.enabled:
            continue
        for import_line in _registry.spec_for(operation.kind).required_imports(
            tool, operation
        ):
            if import_line not in required_imports:
                required_imports.append(import_line)
    for import_line in _style_required_imports():
        if (
            import_line.startswith("import erlab.plotting")
            and "import erlab.plotting as eplt" in required_imports
        ):
            continue
        if import_line not in required_imports:
            required_imports.append(import_line)
    lines.extend(required_imports)
    style_lines = _style_code_lines()
    if style_lines:
        lines.extend(["", *style_lines])
    source_name_replacements = (
        _normalized_source_load_replacements(source_name_map) if source_name_map else {}
    )
    operation_line_groups: list[list[str]] = []
    has_custom_code_lines = False
    for operation in tool._document.recipe.operations:
        if not operation.enabled:
            continue
        operation_lines = _registry.spec_for(operation.kind).code_lines(tool, operation)
        if source_name_replacements and operation.kind != FigureOperationKind.CUSTOM:
            operation_lines = _replace_source_load_names_in_lines(
                operation_lines, source_name_replacements
            )
        elif operation.kind == FigureOperationKind.CUSTOM and operation_lines:
            has_custom_code_lines = True
        operation_line_groups.append(operation_lines)

    if source_name_replacements and has_custom_code_lines:
        lines.extend(["", *_source_alias_assignment_lines(source_name_replacements)])

    source_lines = _source_selection_code_lines(
        tool,
        skip_source_names=skip_source_selection_names,
    )
    if source_name_replacements:
        source_lines = _replace_source_load_names_in_lines(
            source_lines, source_name_replacements
        )
    if source_lines:
        lines.extend(["", *source_lines])
    lines.extend(["", _setup_code(tool._document)])

    for operation_lines in operation_line_groups:
        lines.extend(operation_lines)

    return "\n".join(lines)


def _source_selection_code_lines(
    tool: FigureComposerTool,
    *,
    skip_source_names: frozenset[str] = frozenset(),
) -> list[str]:
    dependency_names = tool._document.source_dependency_names(
        tool._document.direct_sources_used_by_recipe(
            enabled_only=True, executable_only=True
        ),
        stop_at=skip_source_names,
        reject_cycles=True,
    )
    source_by_name = tool._document.source_by_name()
    selected_sources = tuple(
        source_by_name[name]
        for name in dependency_names
        if name in source_by_name
        and name not in skip_source_names
        and _source_has_selection(source_by_name[name])
    )

    lines: list[str] = []
    for source in selected_sources:
        target = _valid_source_variable(source.name)
        selection_source = source.selection_source or source.name
        selection = FigureDataSelectionState(
            source=selection_source,
            isel=dict(source.isel),
            qsel=dict(source.qsel),
            mean_dims=tuple(source.mean_dims),
        )
        lines.append(f"{target} = {_selection_code(selection)}")
    return lines


def _normalized_source_load_replacements(
    replacements: Mapping[str, str],
) -> dict[str, str]:
    return {
        source: target
        for source, target in replacements.items()
        if source != target and source.isidentifier() and target.isidentifier()
    }


def _replace_source_load_names_in_lines(
    lines: list[str], replacements: Mapping[str, str]
) -> list[str]:
    if not lines or not replacements:
        return lines
    return _replace_source_load_names("\n".join(lines), replacements).splitlines()


def _source_alias_assignment_lines(replacements: Mapping[str, str]) -> list[str]:
    if not replacements:
        return []
    sources = tuple(replacements)
    if len(sources) == 1:
        source = sources[0]
        return [f"{source} = {replacements[source]}"]
    return [
        f"{', '.join(sources)} = "
        f"{', '.join(replacements[source] for source in sources)}"
    ]


def _replace_source_load_names(code: str, replacements: Mapping[str, str]) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code

    normalized = _normalized_source_load_replacements(replacements)
    if not normalized:
        return code

    class SourceLoadNameReplacer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if not isinstance(node.ctx, ast.Load):
                return node
            replacement = normalized.get(node.id)
            if replacement is None:
                return node
            return ast.copy_location(ast.Name(replacement, ctx=node.ctx), node)

    updated = typing.cast("ast.Module", SourceLoadNameReplacer().visit(module))
    return ast.unparse(ast.fix_missing_locations(updated))
