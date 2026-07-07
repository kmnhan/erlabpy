"""Standalone Python code generation for Figure Composer recipes."""

from __future__ import annotations

import ast
import typing

from erlab.interactive._figurecomposer._code import _selection_code, _setup_code
from erlab.interactive._figurecomposer._defaults import (
    _style_code_lines,
    _style_required_imports,
)
from erlab.interactive._figurecomposer._operations import _registry
from erlab.interactive._figurecomposer._sources import (
    _source_has_selection,
    _valid_source_variable,
)
from erlab.interactive._figurecomposer._state import FigureDataSelectionState

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
            tool._operation_has_invalid_input(tool._recipe.operations[index])
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
    for operation in tool._recipe.operations:
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
    source_lines = _source_selection_code_lines(
        tool,
        skip_source_names=skip_source_selection_names,
    )
    if source_lines:
        lines.extend(["", *source_lines])
    lines.extend(["", _setup_code(tool)])

    for operation in tool._recipe.operations:
        if not operation.enabled:
            continue
        lines.extend(_registry.spec_for(operation.kind).code_lines(tool, operation))

    code = "\n".join(lines)
    if source_name_map:
        return _replace_source_load_names(code, source_name_map)
    return code


def _source_selection_code_lines(
    tool: FigureComposerTool,
    *,
    skip_source_names: frozenset[str] = frozenset(),
) -> list[str]:
    lines: list[str] = []
    for source in tool._recipe.sources:
        if not _source_has_selection(source) or source.name in skip_source_names:
            continue
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


def _replace_source_load_names(code: str, replacements: Mapping[str, str]) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code

    normalized = {
        source: target
        for source, target in replacements.items()
        if source != target and source.isidentifier() and target.isidentifier()
    }
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
