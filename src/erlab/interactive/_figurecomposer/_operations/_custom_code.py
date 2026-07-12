"""Trusted custom-code operation editor and renderer."""

from __future__ import annotations

import ast
import collections
import io
import symtable
import tokenize
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
        tool._update_operations_by_ids(
            operation_ids,
            lambda _index, target: target.model_copy(update={"code": code}),
            render=render_valid_code,
        )

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
    if document is None:
        raise RuntimeError("Custom code editor has no text document")
    tool._connect_editor_signal(
        code_edit, document.contentsChange, queue_contents_change
    )
    tool._connect_editor_signal(
        code_edit, code_edit.sigTextEditingSettled, commit_settled_code
    )
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
    trust.setObjectName("figureComposerCustomCodeTrustedCheck")
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
    code = operation.code.strip()
    if not code:
        return
    if not operation.trusted:
        raise ValueError("Custom code is not trusted. Enable Trusted to render it.")
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
        tree = ast.parse(code, mode="exec")
        root = symtable.symtable(code, "<figure-composer-python-step>", "exec")
    except SyntaxError:
        return frozenset()

    analyzer = _TopLevelExternalNameAnalyzer()
    analyzer.visit(tree)
    scope_module_bindings = _scope_module_bindings(tree, analyzer.scope_bindings)
    external = set(analyzer.external)
    external.update(_class_local_external_names(tree, root, scope_module_bindings))
    external.update(_child_global_reference_names(tree, root, scope_module_bindings))
    external.update(_child_global_mutation_names(tree, scope_module_bindings))
    return frozenset(external)


def _scope_module_bindings(
    tree: ast.Module,
    direct_bindings: dict[int, frozenset[str]],
) -> dict[int, frozenset[str]]:
    """Propagate module bindings at top-level scope creation into nested scopes."""
    output: dict[int, frozenset[str]] = {}
    scope_types = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    def collect(node: ast.AST, inherited: frozenset[str]) -> None:
        current = direct_bindings.get(id(node), inherited)
        if isinstance(node, scope_types):
            output[id(node)] = current
        for child in ast.iter_child_nodes(node):
            collect(child, current)

    collect(tree, frozenset())
    return output


def _scope_symbol_key(node: ast.AST) -> tuple[str, str, int] | None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return ("function", node.name, node.lineno)
    if isinstance(node, ast.ClassDef):
        return ("class", node.name, node.lineno)
    if isinstance(node, ast.Lambda):
        return ("function", "lambda", node.lineno)
    if isinstance(node, ast.ListComp):
        return ("function", "listcomp", node.lineno)
    if isinstance(node, ast.SetComp):
        return ("function", "setcomp", node.lineno)
    if isinstance(node, ast.DictComp):
        return ("function", "dictcomp", node.lineno)
    if isinstance(node, ast.GeneratorExp):
        return ("function", "genexpr", node.lineno)
    return None


def _scope_symbol_tables(
    tree: ast.Module,
    root: symtable.SymbolTable,
) -> dict[int, symtable.SymbolTable]:
    tables_by_key: collections.defaultdict[
        tuple[str, str, int],
        collections.deque[symtable.SymbolTable],
    ] = collections.defaultdict(collections.deque)

    def collect_tables(table: symtable.SymbolTable) -> None:
        for child in table.get_children():
            lineno = child.get_lineno()
            if lineno is not None:
                tables_by_key[(child.get_type(), child.get_name(), lineno)].append(
                    child
                )
            collect_tables(child)

    collect_tables(root)

    output: dict[int, symtable.SymbolTable] = {}

    def pair_scopes(node: ast.AST) -> None:
        key = _scope_symbol_key(node)
        if key is not None and tables_by_key[key]:
            output[id(node)] = tables_by_key[key].popleft()
        for child in ast.iter_child_nodes(node):
            pair_scopes(child)

    pair_scopes(tree)
    return output


def _class_local_external_names(
    tree: ast.Module,
    root: symtable.SymbolTable,
    scope_module_bindings: dict[int, frozenset[str]],
) -> set[str]:
    """Return class-local names whose value is read before its class binding."""
    scope_tables = _scope_symbol_tables(tree, root)

    external: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        table = scope_tables.get(id(node))
        if table is None:
            continue
        analyzer = _TopLevelExternalNameAnalyzer()
        analyzer._analyze_block(node.body, set())
        class_local_names = {
            symbol.get_name() for symbol in table.get_symbols() if symbol.is_local()
        }
        external.update(
            (analyzer.external & class_local_names)
            - scope_module_bindings.get(id(node), frozenset())
        )
    return external


def _child_global_reference_names(
    tree: ast.Module,
    root: symtable.SymbolTable,
    scope_module_bindings: dict[int, frozenset[str]],
) -> set[str]:
    """Return child-scope global reads without a definite module binding."""
    scope_tables = _scope_symbol_tables(tree, root)
    external: set[str] = set()
    for node in ast.walk(tree):
        table = scope_tables.get(id(node))
        if table is None:
            continue
        referenced_globals = {
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_global()
            and not symbol.is_imported()
            and symbol.is_referenced()
        }
        external.update(
            referenced_globals - scope_module_bindings.get(id(node), frozenset())
        )
    return external


def _child_global_mutation_names(
    tree: ast.Module,
    scope_module_bindings: dict[int, frozenset[str]],
) -> set[str]:
    """Return child-scope globals read implicitly by mutation or deletion."""
    external: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        analyzer = _CurrentScopeGlobalMutationAnalyzer()
        for statement in node.body:
            analyzer.visit(statement)
        external.update(
            (analyzer.global_names & analyzer.mutated_names)
            - scope_module_bindings.get(id(node), frozenset())
        )
    return external


class _CurrentScopeGlobalMutationAnalyzer(ast.NodeVisitor):
    """Collect explicit globals mutated in one function or class scope."""

    def __init__(self) -> None:
        self.global_names: set[str] = set()
        self.mutated_names: set[str] = set()

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.mutated_names.add(node.target.id)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self.mutated_names.update(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
        self.generic_visit(node)

    def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, _node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, _node: ast.Lambda) -> None:
        return


def _match_pattern_bound_names(pattern: ast.pattern) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(pattern):
        if isinstance(node, ast.MatchAs | ast.MatchStar) and node.name is not None:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            bound.add(node.rest)
    return bound


def _custom_code_bound_names(code: str) -> frozenset[str]:
    """Return names that custom code can bind in the generated module."""
    try:
        root = symtable.symtable(code, "<figure-composer-python-step>", "exec")
    except SyntaxError:
        return frozenset()
    return _custom_code_bound_names_from_table(root)


def _custom_code_bound_names_from_table(
    root: symtable.SymbolTable,
) -> frozenset[str]:
    bound = {
        symbol.get_name()
        for symbol in root.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    }
    tables = list(root.get_children())
    while tables:
        table = tables.pop()
        tables.extend(table.get_children())
        bound.update(
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_global() and (symbol.is_assigned() or symbol.is_imported())
        )
    return frozenset(bound)


class _TopLevelExternalNameAnalyzer(ast.NodeVisitor):
    """Collect module-level names read before a definite local binding."""

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.external: set[str] = set()
        self.scope_bindings: dict[int, frozenset[str]] = {}

    def _analyze_block(self, statements: list[ast.stmt], initial: set[str]) -> set[str]:
        previous = self.bound
        self.bound = set(initial)
        for statement in statements:
            self.visit(statement)
        result = set(self.bound)
        self.bound = previous
        return result

    def _analyze_node(self, node: ast.AST, initial: set[str]) -> set[str]:
        previous = self.bound
        self.bound = set(initial)
        self.visit(node)
        result = set(self.bound)
        self.bound = previous
        return result

    def _read_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            if target.id not in self.bound:
                self.external.add(target.id)
            return
        self.visit(target)

    def _bind_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.bound.add(target.id)
            return
        if isinstance(target, ast.Starred):
            self._bind_target(target.value)
            return
        if isinstance(target, ast.Tuple | ast.List):
            for item in target.elts:
                self._bind_target(item)
            return
        self.visit(target)

    def _visit_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        for type_param in getattr(node, "type_params", ()):
            self.visit(type_param)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self.bound:
            self.external.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._bind_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is None:
            if not isinstance(node.target, ast.Name):
                self.visit(node.target)
            return
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._read_target(node.target)
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._read_target(target)
            if isinstance(target, ast.Name):
                self.bound.discard(target.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name.partition(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.bound.add(alias.asname or alias.name)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        initial = set(self.bound)
        body_bound = self._analyze_block(node.body, initial)
        else_bound = (
            self._analyze_block(node.orelse, initial) if node.orelse else initial
        )
        self.bound = body_bound & else_bound

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        initial = set(self.bound)
        body_bound = self._analyze_node(node.body, initial)
        else_bound = self._analyze_node(node.orelse, initial)
        self.bound = body_bound & else_bound

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not node.values:
            return
        self.visit(node.values[0])
        definite_bound = set(self.bound)
        conditional_bound = set(definite_bound)
        for value in node.values[1:]:
            conditional_bound = self._analyze_node(value, conditional_bound)
        self.bound = definite_bound

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        initial = set(self.bound)
        for case in node.cases:
            previous = self.bound
            self.bound = set(initial)
            self.visit(case.pattern)
            self.bound.update(_match_pattern_bound_names(case.pattern))
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)
            self.bound = previous
        self.bound = initial

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        initial = set(self.bound)
        previous = self.bound
        self.bound = set(initial)
        self._bind_target(node.target)
        for statement in node.body:
            self.visit(statement)
        self.bound = previous
        self._analyze_block(node.orelse, initial)
        self.bound = initial

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        initial = set(self.bound)
        self._analyze_block(node.body, initial)
        self._analyze_block(node.orelse, initial)
        self.bound = initial

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_Try(self, node: ast.Try | ast.TryStar) -> None:
        initial = set(self.bound)
        body_bound = self._analyze_block(node.body, initial)
        success_bound = self._analyze_block(node.orelse, body_bound)
        path_bounds = [success_bound]
        for handler in node.handlers:
            if handler.type is not None:
                previous = self.bound
                self.bound = set(initial)
                self.visit(handler.type)
                self.bound = previous
            handler_initial = set(initial)
            if handler.name is not None:
                handler_initial.add(handler.name)
            handler_bound = self._analyze_block(handler.body, handler_initial)
            if handler.name is not None:
                handler_bound.discard(handler.name)
            path_bounds.append(handler_bound)
        merged = set.intersection(*path_bounds)
        self.bound = self._analyze_block(node.finalbody, merged)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self.visit_Try(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_signature(node)
        self.scope_bindings[id(node)] = frozenset((*self.bound, node.name))
        self.bound.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_signature(node)
        self.scope_bindings[id(node)] = frozenset((*self.bound, node.name))
        self.bound.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        self.scope_bindings[id(node)] = frozenset(self.bound)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for type_param in getattr(node, "type_params", ()):
            self.visit(type_param)
        self.scope_bindings[id(node)] = frozenset(self.bound)
        class_analyzer = _TopLevelExternalNameAnalyzer()
        class_analyzer._analyze_block(node.body, set())
        self.external.update(class_analyzer.external - self.bound)
        self.bound.add(node.name)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        generators: list[ast.comprehension],
        *value_nodes: ast.expr,
    ) -> None:
        if not generators:
            for value_node in value_nodes:
                self.visit(value_node)
            self.scope_bindings[id(node)] = frozenset(self.bound)
            return

        first, *remaining = generators
        self.visit(first.iter)
        module_bound = set(self.bound)
        self.scope_bindings[id(node)] = frozenset(self.bound)

        self.bound = set(module_bound)
        self._bind_target(first.target)
        for condition in first.ifs:
            self.visit(condition)
        for generator in remaining:
            self.visit(generator.iter)
            self._bind_target(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for value_node in value_nodes:
            self.visit(value_node)
        self.bound = module_bound

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, node.generators, node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, node.generators, node.elt)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, node.generators, node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, node.generators, node.key, node.value)


def _renamed_source_loads(code: str, replacements: dict[str, str]) -> str:
    """Rename unambiguous source-variable reads without reformatting user code."""
    replacements = {
        old: new
        for old, new in replacements.items()
        if old != new and old.isidentifier()
    }
    if not replacements or not any(old in code for old in replacements):
        return code
    try:
        has_source_identifier = any(
            token.type == tokenize.NAME and token.string in replacements
            for token in tokenize.generate_tokens(io.StringIO(code).readline)
        )
    except (IndentationError, tokenize.TokenError):
        pass
    else:
        if not has_source_identifier:
            return code
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(
            "the Python step does not currently contain valid code"
        ) from exc

    load_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in replacements
    ]
    destructive_uses = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Del)
            and node.id in replacements
        )
        or (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in replacements
        )
    ]
    if not load_nodes and not destructive_uses:
        return code

    tables = [symtable.symtable(code, "<figure-composer-python-step>", "exec")]
    ambiguous_names: set[str] = set()
    while tables:
        table = tables.pop()
        tables.extend(table.get_children())
        for name in replacements:
            try:
                symbol = table.lookup(name)
            except KeyError:
                continue
            if (
                symbol.is_assigned()
                or symbol.is_imported()
                or symbol.is_parameter()
                or symbol.is_nonlocal()
                or symbol.is_declared_global()
            ):
                ambiguous_names.add(name)
    ambiguous = sorted(ambiguous_names)
    if ambiguous:
        names = ", ".join(repr(name) for name in ambiguous)
        raise ValueError(
            f"the Python step also binds {names}; rename that local binding first"
        )

    encoded = code.encode("utf-8")
    line_offsets: list[int] = []
    offset = 0
    for line in code.splitlines(keepends=True):
        line_offsets.append(offset)
        offset += len(line.encode("utf-8"))

    edits: list[tuple[int, int, bytes]] = []
    for node in load_nodes:
        if node.end_lineno is None or node.end_col_offset is None:  # pragma: no cover
            raise RuntimeError("parsed source name is missing location information")
        start = line_offsets[node.lineno - 1] + node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        edits.append((start, end, replacements[node.id].encode("utf-8")))
    for start, end, replacement in sorted(edits, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode("utf-8")


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
