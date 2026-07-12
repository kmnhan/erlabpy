"""Manager provenance helpers for Figure Composer recipes."""

from __future__ import annotations

import ast
import typing

import erlab.interactive._figurecomposer._codegen
from erlab.interactive.imagetool import provenance

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from erlab.interactive._figurecomposer._tool import FigureComposerTool


class _FigureBuildCodeOperation(provenance.ScriptCodeOperation):
    hoist_imports: typing.ClassVar[bool] = True


def _code_assigns_name(code: str, name: str) -> bool:
    module = ast.parse(code, mode="exec")

    def target_assigns_name(target: ast.AST) -> bool:
        if (
            isinstance(target, ast.Name)
            and target.id == name
            and isinstance(target.ctx, ast.Store)
        ):
            return True
        if isinstance(target, ast.Tuple | ast.List):
            return any(target_assigns_name(item) for item in target.elts)
        return False

    for statement in module.body:
        if isinstance(statement, ast.Assign) and any(
            target_assigns_name(target) for target in statement.targets
        ):
            return True
        if isinstance(statement, ast.AnnAssign) and target_assigns_name(
            statement.target
        ):
            return True
        if isinstance(statement, ast.AugAssign) and target_assigns_name(
            statement.target
        ):
            return True
    return False


def _figure_build_code(
    tool: FigureComposerTool,
    *,
    skip_source_selection_names: frozenset[str] = frozenset(),
    source_name_map: Mapping[str, str] | None = None,
) -> str | None:
    try:
        code = erlab.interactive._figurecomposer._codegen.generated_code(
            tool,
            skip_source_selection_names=skip_source_selection_names,
            source_name_map=source_name_map,
        )
    except Exception:
        # Details-panel refreshes happen while the user is editing. Temporarily
        # invalid recipes should make replay code unavailable, not crash metadata UI.
        return None
    try:
        if not _code_assigns_name(code, "fig"):
            return None
    except SyntaxError:
        return None
    return code


def _figure_build_operation(
    tool: FigureComposerTool,
    *,
    skip_source_selection_names: frozenset[str] = frozenset(),
    source_name_map: Mapping[str, str] | None = None,
) -> provenance.ScriptCodeOperation:
    code = _figure_build_code(
        tool,
        skip_source_selection_names=skip_source_selection_names,
        source_name_map=source_name_map,
    )
    return _FigureBuildCodeOperation(
        label="Build figure",
        code=code,
        copyable=code is not None,
    )
