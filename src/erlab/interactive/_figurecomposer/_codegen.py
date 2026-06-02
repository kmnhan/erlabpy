"""Standalone Python code generation for Figure Composer recipes."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._code import _setup_code
from erlab.interactive._figurecomposer._defaults import (
    _style_code_lines,
    _style_required_imports,
)
from erlab.interactive._figurecomposer._operations import _registry

if typing.TYPE_CHECKING:
    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def generated_code(tool: FigureComposerTool) -> str:
    invalid_indices = tool._invalid_operation_indices()
    if invalid_indices:
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
        if import_line not in required_imports:
            required_imports.append(import_line)
    lines.extend(required_imports)
    style_lines = _style_code_lines()
    if style_lines:
        lines.extend(["", *style_lines])
    lines.extend(["", _setup_code(tool)])

    for operation in tool._recipe.operations:
        if not operation.enabled:
            continue
        lines.extend(_registry.spec_for(operation.kind).code_lines(tool, operation))

    return "\n".join(lines)
