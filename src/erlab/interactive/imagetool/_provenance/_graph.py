"""Internal replay graph for ImageTool provenance.

The saved provenance schema stays in
:mod:`erlab.interactive.imagetool.provenance`.
This module compiles those specs into an execution/code-generation graph at runtime so
shared file loads and shared structured operations are emitted or replayed once.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import keyword
import pathlib
import re
import symtable
import tokenize
import typing
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence

import xarray as xr

from erlab.interactive.imagetool._provenance._code import (
    _NONUNIFORM_RESTORE_FUNCTION_NAME,
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    _code_stores_name,
    _code_uses_name,
    _expression_starts_with_name,
    _nonuniform_restore_support_code,
    _replace_code_identifiers,
    _script_codes_output_name,
    _simplify_display_code,
    _statement_load_count,
    _statement_store_count,
    _validate_script_replay_code,
)
from erlab.interactive.imagetool._provenance._model import (
    _assignment_code,
    _script_input_reference_text,
    parse_tool_provenance_spec,
)

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool._provenance._operations import ScriptCodeOperation


class ReplayGraphError(Exception):
    """Raised when provenance cannot be compiled, emitted, or replayed."""


class ReplayNode:
    __slots__ = (
        "cacheable",
        "key",
        "kind",
        "parents",
        "payload",
    )

    def __init__(
        self,
        key: str,
        kind: str,
        *,
        parents: Sequence[str] = (),
        cacheable: bool = True,
        payload: Mapping[str, typing.Any] | None = None,
    ) -> None:
        self.key = key
        self.kind = kind
        self.parents = tuple(parents)
        self.cacheable = cacheable
        self.payload = dict(payload or {})


class ReplayGraph:
    __slots__ = (
        "_cacheable_keys",
        "_external_names",
        "_reserved_names",
        "aliases",
        "display",
        "nodes",
        "output_key",
        "trusted_user_code",
    )

    def __init__(
        self,
        *,
        reserved_names: set[str] | None = None,
        display: bool = False,
        trusted_user_code: bool = False,
    ) -> None:
        self.nodes: list[ReplayNode] = []
        self._cacheable_keys: dict[str, str] = {}
        self._external_names: set[str] = set()
        self._reserved_names = set(reserved_names or ())
        self.aliases: list[tuple[str, str]] = []
        self.display = bool(display)
        self.trusted_user_code = bool(trusted_user_code)
        self.output_key: str | None = None

    @property
    def reserved_names(self) -> set[str]:
        return set(self._reserved_names)

    @property
    def external_names(self) -> set[str]:
        """Return names that emitted script code receives from its caller."""
        return set(self._external_names)

    def add_alias(self, public_name: str, key: str) -> None:
        if _is_semantic_replay_name(public_name):
            self.aliases.append((public_name, key))

    def add_node(
        self,
        key: str,
        kind: str,
        *,
        parents: Sequence[str] = (),
        cacheable: bool = True,
        payload: Mapping[str, typing.Any] | None = None,
    ) -> str:
        if cacheable and key in self._cacheable_keys:
            return self._cacheable_keys[key]

        node_key = key if cacheable else f"{key}#{len(self.nodes)}"
        node = ReplayNode(
            node_key,
            kind,
            parents=parents,
            cacheable=cacheable,
            payload=payload,
        )
        self.nodes.append(node)
        if cacheable:
            self._cacheable_keys[key] = node_key
        return node_key


LiveInputResolver = Callable[[typing.Any], tuple[xr.DataArray, typing.Any] | None]
_REPLAY_ALIASES = {
    "era": "erlab.analysis",
    "eri": "erlab.interactive",
    "eplt": "erlab.plotting",
}
_REPLAY_FRAMEWORK_IMPORTS = {
    "erlab": "import erlab",
    "np": "import numpy as np",
    "numpy": "import numpy",
    "pathlib": "import pathlib",
    "xr": "import xarray as xr",
    "xarray": "import xarray",
    **{
        alias: f"import {target} as {alias}"
        for alias, target in _REPLAY_ALIASES.items()
    },
}
_REPLAY_RESERVED_PUBLIC_NAMES = {"data", "derived", "tools"}
_REPLAY_TEMP_PREFIX = "_itool_replay_"
_FILE_LOAD_OUTPUT_SENTINEL = "_itool_file_load_output"


def _canonical_key(kind: str, payload: Mapping[str, typing.Any]) -> str:
    return json.dumps(
        {"kind": kind, **dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_semantic_replay_name(name: str) -> bool:
    return (
        name.isidentifier()
        and not keyword.iskeyword(name)
        and name not in _REPLAY_RESERVED_PUBLIC_NAMES
        and not name.startswith("data_")
        and not name.startswith(_REPLAY_TEMP_PREFIX)
        and not name.startswith("__")
    )


def _reserved_names_from_spec(spec: typing.Any) -> set[str]:
    names: set[str] = set()
    active_name = getattr(spec, "active_name", None)
    if isinstance(active_name, str):
        names.add(active_name)
    for script_input in getattr(spec, "script_inputs", ()):
        names.add(script_input.name)
        nested = script_input.parsed_provenance_spec()
        if nested is not None:
            names.update(_reserved_names_from_spec(nested))
    for binding in getattr(spec, "script_context_bindings", ()):
        names.update(binding.names)
    return names


class _CurrentScopeNames(ast.NodeVisitor):
    def __init__(self, local_names: set[str] | None = None) -> None:
        self.loads: set[str] = set()
        self.stores: set[str] = set()
        self._local_names = set(local_names or ())

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self._local_names:
            self.loads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)) and node.id not in (
            self._local_names
        ):
            self.stores.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.stores.add(
                alias.asname
                if alias.asname is not None
                else alias.name.partition(".")[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self.stores.add(alias.asname if alias.asname is not None else alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stores.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_argument_expressions(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(typing.cast("ast.FunctionDef", node))

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_argument_expressions(node.args)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators, node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators, node.elt)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators, node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators, node.key, node.value)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stores.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword_arg in node.keywords:
            self.visit(keyword_arg)

    def _visit_argument_expressions(self, args: ast.arguments) -> None:
        for default in args.defaults:
            self.visit(default)
        for default in args.kw_defaults:
            if default is not None:
                self.visit(default)
        for arg in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *(arg for arg in (args.vararg, args.kwarg) if arg is not None),
        ):
            if arg.annotation is not None:
                self.visit(arg.annotation)

    def _visit_comprehension(
        self, generators: Sequence[ast.comprehension], *value_nodes: ast.AST
    ) -> None:
        local_names = set(self._local_names)
        for generator in generators:
            self._visit_with_local_names(generator.iter, local_names)
            target_names = _CurrentScopeNames(local_names)
            target_names.visit(generator.target)
            self.loads.update(target_names.loads)
            local_names.update(target_names.stores)
            for condition in generator.ifs:
                self._visit_with_local_names(condition, local_names)
        for node in value_nodes:
            self._visit_with_local_names(node, local_names)

    def _visit_with_local_names(self, node: ast.AST, local_names: set[str]) -> None:
        names = _CurrentScopeNames(local_names)
        names.visit(node)
        self.loads.update(names.loads)
        self.stores.update(names.stores)


def _statement_scope_names(stmt: ast.stmt) -> _CurrentScopeNames:
    names = _CurrentScopeNames()
    names.visit(stmt)
    return names


def _symbol_table_dependencies(table: symtable.SymbolTable) -> set[str]:
    """Return names that a nested scope resolves outside itself."""
    dependencies = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_referenced()
        and (symbol.is_global() or symbol.is_free())
        and symbol.get_name() != table.get_name()
    }
    for child in table.get_children():
        dependencies.update(_symbol_table_dependencies(child))
    return dependencies


def _script_function_dependencies(code: str) -> dict[tuple[str, int], set[str]]:
    table = symtable.symtable(code, "<ImageTool script provenance>", "exec")

    output: dict[tuple[str, int], set[str]] = {}
    for child in table.get_children():
        if child.get_type() != "function":
            continue
        lineno = child.get_lineno()
        if lineno is None:
            continue
        output[(child.get_name(), lineno)] = _symbol_table_dependencies(child)
    return output


def _validate_script_code_names(
    code: str,
    available_names: set[str],
    function_dependencies: dict[str, set[str]],
    *,
    external_name_candidates: Collection[str] | None = (),
) -> set[str]:
    """Validate reads and return permitted names supplied by the caller.

    ``None`` permits every unresolved name. A collection permits only its members.
    Deferred function dependencies are checked when the function is first read.
    """
    module = ast.parse(code, mode="exec")
    new_function_dependencies = _script_function_dependencies(code)
    external_names: set[str] = set()

    def require(name: str, visiting: set[str] | None = None) -> str | None:
        if name not in available_names:
            if external_name_candidates is None or name in external_name_candidates:
                available_names.add(name)
                external_names.add(name)
                return None
            return name
        deps = function_dependencies.get(name)
        if not deps:
            return None
        if visiting is None:
            visiting = set()
        if name in visiting:
            return None
        visiting.add(name)
        for dependency in deps:
            missing = require(dependency, visiting)
            if missing is not None:
                return missing
        visiting.remove(name)
        return None

    def require_loads(names: _CurrentScopeNames) -> None:
        for name in sorted(names.loads):
            missing = require(name)
            if missing is not None:
                raise ReplayGraphError(
                    f"Script provenance references unresolved name {missing!r}"
                )

    def validate_stmt(stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.For):
            iter_names = _CurrentScopeNames()
            iter_names.visit(stmt.iter)
            require_loads(iter_names)

            target_names = _CurrentScopeNames()
            target_names.visit(stmt.target)
            require_loads(target_names)
            available_names.update(target_names.stores)
            for body_stmt in stmt.body:
                validate_stmt(body_stmt)
            for orelse_stmt in stmt.orelse:
                validate_stmt(orelse_stmt)
            return

        if isinstance(stmt, ast.If):
            test_names = _CurrentScopeNames()
            test_names.visit(stmt.test)
            require_loads(test_names)

            available_before = set(available_names)
            dependencies_before = dict(function_dependencies)
            branch_stores: set[str] = set()
            branch_dependencies: dict[str, set[str]] = {}
            for branch in (stmt.body, stmt.orelse):
                available_names.clear()
                available_names.update(available_before)
                function_dependencies.clear()
                function_dependencies.update(dependencies_before)
                for branch_stmt in branch:
                    validate_stmt(branch_stmt)
                branch_stores.update(available_names - available_before)
                for name, dependencies in function_dependencies.items():
                    if dependencies_before.get(name) == dependencies:
                        continue
                    branch_dependencies.setdefault(name, set()).update(dependencies)

            available_names.clear()
            available_names.update(available_before)
            available_names.update(branch_stores)
            function_dependencies.clear()
            function_dependencies.update(dependencies_before)
            function_dependencies.update(branch_dependencies)
            return

        if isinstance(stmt, ast.Try):
            available_before = set(available_names)
            dependencies_before = dict(function_dependencies)

            for body_stmt in stmt.body:
                validate_stmt(body_stmt)
            available_after_body = set(available_names)
            dependencies_after_body = dict(function_dependencies)

            for handler in stmt.handlers:
                available_names.clear()
                available_names.update(available_before)
                function_dependencies.clear()
                function_dependencies.update(dependencies_before)
                if handler.type is not None:
                    handler_names = _CurrentScopeNames()
                    handler_names.visit(handler.type)
                    require_loads(handler_names)
                if handler.name is not None:
                    available_names.add(handler.name)
                for handler_stmt in handler.body:
                    validate_stmt(handler_stmt)

            available_names.clear()
            available_names.update(available_after_body)
            function_dependencies.clear()
            function_dependencies.update(dependencies_after_body)
            for orelse_stmt in stmt.orelse:
                validate_stmt(orelse_stmt)

            available_names.clear()
            available_names.update(available_before)
            function_dependencies.clear()
            function_dependencies.update(dependencies_before)
            return

        names = _statement_scope_names(stmt)
        require_loads(names)
        if isinstance(stmt, ast.FunctionDef):
            function_dependencies[stmt.name] = new_function_dependencies.get(
                (stmt.name, stmt.lineno),
                set(),
            )
        available_names.update(names.stores)

    for stmt in module.body:
        validate_stmt(stmt)
    return external_names


def _simple_name_assignment(code: str) -> tuple[str, str] | None:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return None
    if len(module.body) != 1:
        return None
    stmt = module.body[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Name):
        return target.id, stmt.value.id
    return None


def _name_value_is_live(codes: Sequence[str], name: str) -> bool:
    """Return whether later code reads the current value before replacing it."""
    for code in codes:
        for statement in ast.parse(code, mode="exec").body:
            if _statement_load_count(statement, name):
                return True
            if _statement_store_count(statement, name):
                return False
    return False


def _validate_script_provenance(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    allow_external_names: bool = False,
    strict_replay_code: bool = True,
) -> tuple[set[str], set[str]]:
    """Validate a script recipe and return all and seed-only caller names.

    Display seeds can start from caller variables. Later replay steps keep their
    normal strict name checks, except for ``load_script`` because generated extension
    setup owns that import name.
    """
    if spec.kind != "script":
        raise ReplayGraphError("Expected script provenance")
    if spec.active_name is None:
        raise ReplayGraphError(
            "Script provenance cannot be replayed without active_name"
        )

    builtin_names = set(_SCRIPT_REPLAY_ALLOWED_BUILTINS)
    if not strict_replay_code:
        builtin_names.update(vars(builtins))

    implicit_framework_names = {
        "erlab",
        "np",
        "numpy",
        "xr",
        "xarray",
        *_REPLAY_ALIASES,
    }
    available_names = set(builtin_names)
    if not allow_external_names:
        available_names.update(implicit_framework_names)
    if spec.script_inputs:
        available_names.update(script_input.name for script_input in spec.script_inputs)
    elif external_input_names is not None:
        available_names.update(external_input_names)
    function_dependencies: dict[str, set[str]] = {}
    caller_names: set[str] = set()
    seed_caller_names: set[str] = set()
    has_replay_step = False
    active_available = spec.active_name in available_names
    current_name: str | None = spec.active_name if active_available else None

    if spec.seed_code:
        has_replay_step = True
        if strict_replay_code:
            try:
                _validate_script_replay_code(spec.seed_code)
            except (TypeError, ValueError) as exc:
                raise ReplayGraphError(str(exc)) from exc
        else:
            try:
                ast.parse(spec.seed_code, mode="exec")
            except SyntaxError as exc:
                raise ReplayGraphError(
                    "Script replay code is not valid Python"
                ) from exc
        seed_caller_names = _validate_script_code_names(
            spec.seed_code,
            available_names,
            function_dependencies,
            external_name_candidates=None if allow_external_names else (),
        )
        caller_names.update(seed_caller_names)
        if allow_external_names:
            available_names.update(implicit_framework_names)
        active_available = active_available or _code_stores_name(
            spec.seed_code, spec.active_name
        )
        current_name = _script_codes_output_name(
            (spec.seed_code,),
            active_name=spec.active_name,
            current_name=current_name,
        )
    context_bindings_by_index: dict[int, list[str]] = {}
    for binding in spec.script_context_bindings:
        context_bindings_by_index.setdefault(binding.operation_index, []).extend(
            binding.names
        )
    for index, operation in enumerate(spec.operations):
        if context_names := context_bindings_by_index.get(index):
            if current_name is None:
                raise ReplayGraphError("Script provenance has no replay code")
            available_names.update(context_names)
        if getattr(operation, "op", None) == "script_code":
            script_operation = typing.cast("ScriptCodeOperation", operation)
            if not script_operation.copyable or script_operation.code is None:
                raise ReplayGraphError("Script provenance contains non-replayable code")
            has_replay_step = True
            framework_owned = script_operation.framework_owned
            operation_available_names = available_names
            implicit_names_added: set[str] = set()
            if framework_owned:
                operation_available_names = set(available_names)
                implicit_names_added = implicit_framework_names - available_names
                operation_available_names.update(implicit_framework_names)
            external_name_candidates: Collection[str] | None = ()
            if allow_external_names:
                external_name_candidates = None
            if strict_replay_code:
                try:
                    _validate_script_replay_code(script_operation.code)
                except (TypeError, ValueError) as exc:
                    raise ReplayGraphError(str(exc)) from exc
            else:
                try:
                    ast.parse(script_operation.code, mode="exec")
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
            caller_names.update(
                _validate_script_code_names(
                    script_operation.code,
                    operation_available_names,
                    function_dependencies,
                    external_name_candidates=external_name_candidates,
                )
            )
            if framework_owned:
                available_names.update(
                    name
                    for name in operation_available_names
                    if name not in implicit_names_added
                    or _code_stores_name(script_operation.code, name)
                )
            active_available = active_available or _code_stores_name(
                script_operation.code, spec.active_name
            )
            current_name = _script_codes_output_name(
                (script_operation.code,),
                active_name=spec.active_name,
                current_name=current_name,
            )
            continue
        if current_name is None:
            raise ReplayGraphError("Script provenance has no replay code")
        if not operation.live_applicable:
            raise ReplayGraphError(
                "Script provenance contains non-replayable operation"
            )
        has_replay_step = True
        available_names.add(current_name)
        preferred_name = operation.preferred_replay_output_name()
        if preferred_name is not None:
            current_name = preferred_name
            available_names.add(preferred_name)
            active_available = preferred_name == spec.active_name
        elif index == len(spec.operations) - 1:
            current_name = spec.active_name
            active_available = True
        else:
            active_available = active_available or current_name == spec.active_name
    if not has_replay_step:
        raise ReplayGraphError("Script provenance has no replay code")
    if not active_available and current_name != spec.active_name:
        raise ReplayGraphError("Script provenance has no replay code")
    return caller_names, seed_caller_names


def _file_seed_code_parts(seed_code: str, active_name: str) -> tuple[str | None, str]:

    def module_code(
        body: Sequence[ast.stmt],
        *,
        strip_standard_imports: bool = False,
    ) -> str | None:
        if not body:
            return None
        if strip_standard_imports:
            filtered_body: list[ast.stmt] = []
            for stmt in body:
                if isinstance(stmt, ast.Import):
                    names = [
                        alias
                        for alias in stmt.names
                        if alias.name != "erlab" or alias.asname not in {None, "erlab"}
                    ]
                    if names:
                        stmt = ast.Import(names=names)
                        filtered_body.append(stmt)
                    continue
                filtered_body.append(stmt)
            body = filtered_body
            if not body:
                return None
        module = ast.Module(body=list(body), type_ignores=[])
        return ast.unparse(ast.fix_missing_locations(module))

    try:
        module = ast.parse(seed_code, mode="exec")
    except SyntaxError as exc:
        raise ReplayGraphError("File replay code is not valid Python") from exc

    output_stmt_idx = next(
        (
            idx
            for idx, stmt in enumerate(module.body)
            if _statement_store_count(stmt, active_name) > 0
        ),
        None,
    )
    if output_stmt_idx is None:
        raise ReplayGraphError("File replay code does not assign its output")

    setup_code = module_code(
        module.body[:output_stmt_idx],
        strip_standard_imports=True,
    )
    load_code = module_code(module.body[output_stmt_idx:])
    if load_code is None:
        raise ReplayGraphError("File replay code does not assign its output")
    return setup_code, load_code


def _single_assignment_output_name(code: str) -> str | None:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return None
    output_name: str | None = None
    for stmt in module.body:
        target: ast.expr | None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
        else:
            continue
        if not isinstance(target, ast.Name):
            return None
        if output_name is not None:
            return None
        output_name = target.id
    return output_name


def _canonical_file_load_code(code: str, output_name: str) -> str:
    module = ast.parse(code, mode="exec")

    class FileLoadOutputCanonicalizer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == output_name:
                return ast.copy_location(
                    ast.Name(_FILE_LOAD_OUTPUT_SENTINEL, ctx=node.ctx),
                    node,
                )
            return node

    canonical = typing.cast("ast.Module", FileLoadOutputCanonicalizer().visit(module))
    return ast.unparse(ast.fix_missing_locations(canonical))


def _file_load_key_payload(
    load_source: typing.Any,
    *,
    setup_code: str | None,
    load_code: str | None,
    active_name: str,
) -> dict[str, typing.Any]:
    payload = typing.cast(
        "dict[str, typing.Any]",
        load_source.model_dump(mode="json"),
    )
    if _is_extension_loader_source(load_source):
        payload["load_code"] = None
        payload["setup_code"] = None
    else:
        payload["setup_code"] = setup_code
        payload["load_code"] = (
            None
            if load_code is None
            else _canonical_file_load_code(load_code, active_name)
        )
    return payload


def _is_extension_loader_source(load_source: typing.Any) -> bool:
    replay_call = getattr(load_source, "replay_call", None)
    return replay_call is not None and replay_call.kind == "extension_loader"


def _is_extension_loader_node(node: ReplayNode) -> bool:
    return node.kind == "file_load" and _is_extension_loader_source(
        node.payload.get("load_source")
    )


def _add_file_load_node(
    graph: ReplayGraph,
    load_source: typing.Any,
    *,
    setup_code: str | None,
    load_code: str | None,
    active_name: str,
) -> str:
    if _is_extension_loader_source(load_source):
        setup_code = None
        load_code = None
    setup_key = None
    if setup_code:
        setup_key = graph.add_node(
            _canonical_key("setup", {"code": setup_code}),
            "setup",
            payload={"code": setup_code},
        )
    return graph.add_node(
        _canonical_key(
            "file_load",
            _file_load_key_payload(
                load_source,
                setup_code=setup_code,
                load_code=load_code,
                active_name=active_name,
            ),
        ),
        "file_load",
        parents=() if setup_key is None else (setup_key,),
        payload={
            "active_name": active_name,
            "load_source": load_source,
            "load_code": load_code,
        },
    )


def _compile_replay_steps(
    graph: ReplayGraph,
    current_key: str,
    replay_steps: Sequence[typing.Any],
    *,
    display: bool,
) -> str:
    legacy_context_keys: dict[int, str] = {}
    previous_input_policy: str | None = None
    for step in replay_steps:
        operation = step.operation
        context_key = current_key
        input_policy = step.input_policy
        if step.legacy_context is not None:
            legacy_index = step.legacy_context.index
            if legacy_index not in legacy_context_keys:
                legacy_context_keys[legacy_index] = current_key
                input_policy = step.legacy_context.input_policy
            else:
                input_policy = None
            context_key = legacy_context_keys[legacy_index]
        if input_policy in {"current", "restored"} and (
            step.legacy_context is not None or previous_input_policy != input_policy
        ):
            source_kind = "full_data" if input_policy == "current" else "public_data"
            current_key = graph.add_node(
                _canonical_key(
                    "source_view",
                    {"parent": current_key, "source_kind": source_kind},
                ),
                "source_view",
                parents=(current_key,),
                payload={"source_kind": source_kind},
            )
            if step.legacy_context is None:
                context_key = current_key
        if display:
            entry = operation.derivation_entry()
            if entry.code in {
                "derived = derived.isel()",
                "derived = derived.qsel()",
                "derived = derived.sel()",
            }:
                previous_input_policy = step.input_policy
                continue
        current_key = graph.add_node(
            _canonical_key(
                "operation",
                {
                    "context": context_key,
                    "legacy_parent_context": step.legacy_context is not None,
                    "operation": operation.model_dump(mode="json"),
                    "parent": current_key,
                },
            ),
            "operation",
            parents=(current_key, context_key),
            payload={
                "operation": operation,
                "legacy_parent_context": step.legacy_context is not None,
            },
        )
        previous_input_policy = step.input_policy
    return current_key


def _script_seed_file_load_parts(
    seed_code: str,
    *,
    active_name: str,
    load_source: typing.Any,
) -> tuple[str | None, str, str] | None:
    if getattr(load_source, "replay_call", None) is None:
        return None
    seed_output_name = _script_codes_output_name(
        (seed_code,),
        active_name=active_name,
        current_name=None,
    )
    if seed_output_name is None:
        return None
    try:
        seed_setup_code, seed_load_code = _file_seed_code_parts(
            seed_code,
            seed_output_name,
        )
    except ReplayGraphError:
        return None

    recorded_load_code = getattr(load_source, "load_code", None)
    if recorded_load_code is None:
        return None
    recorded_output_name = _single_assignment_output_name(recorded_load_code)
    if recorded_output_name is None:
        return None
    try:
        recorded_setup_code, recorded_load_part = _file_seed_code_parts(
            recorded_load_code,
            recorded_output_name,
        )
    except ReplayGraphError:
        return None

    if (recorded_setup_code or None) != (seed_setup_code or None):
        return None
    if _canonical_file_load_code(
        recorded_load_part,
        recorded_output_name,
    ) != _canonical_file_load_code(seed_load_code, seed_output_name):
        return None
    return seed_setup_code, seed_load_code, seed_output_name


def _operation_replay_code(
    operation: typing.Any,
    *,
    active_name: str,
    context_name: str,
    parent_name: str | None = None,
    reserved_names: Collection[str] = (),
) -> str:
    input_name = active_name if parent_name is None else parent_name
    try:
        code = operation.replay_code(
            input_name,
            output_name=active_name,
            source_name=context_name,
            reserved_names=reserved_names,
        )
    except (AttributeError, NotImplementedError) as exc:
        raise ReplayGraphError("Operation does not provide replay code") from exc
    if code is None:
        raise ReplayGraphError("Operation does not provide replay code")
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ReplayGraphError("Operation replay code is not valid Python") from exc
    if not _code_stores_name(code, active_name) and not (
        input_name == active_name
        and getattr(operation, "statement_mutates_input", False)
    ):
        raise ReplayGraphError("Operation replay code does not assign its output")
    return code


def _compile_spec(
    graph: ReplayGraph,
    spec: typing.Any,
    *,
    display: bool,
    trusted_user_code: bool,
    structured_file_replay: bool,
    external_inputs: Mapping[str, xr.DataArray] | None,
    live_input_resolver: LiveInputResolver | None,
) -> str:
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None:
        raise ReplayGraphError("Expected provenance spec")
    if parsed.kind == "file":
        if parsed.file_load_source is None:
            raise ReplayGraphError("File provenance does not define a load source")
        if parsed.active_name is None:
            raise ReplayGraphError("File provenance does not define an active name")
        structured_replay = parsed.file_load_source.replay_call is not None
        extension_loader = _is_extension_loader_source(parsed.file_load_source)
        setup_code, load_code = (
            (None, None)
            if extension_loader
            or (structured_replay and structured_file_replay)
            or parsed.seed_code is None
            else _file_seed_code_parts(parsed.seed_code, parsed.active_name)
        )
        current_key = _add_file_load_node(
            graph,
            parsed.file_load_source,
            setup_code=setup_code,
            load_code=load_code,
            active_name=parsed.active_name,
        )
        return _compile_replay_steps(
            graph,
            current_key,
            parsed.steps,
            display=display,
        )

    if parsed.kind == "script":
        caller_names, seed_caller_names = _validate_script_provenance(
            parsed,
            external_input_names=set(external_inputs or ()),
            allow_external_names=display
            and not parsed.script_inputs
            and not external_inputs
            and parsed.file_load_source is None,
            strict_replay_code=not display and not trusted_user_code,
        )
        graph._external_names.update(caller_names)
        bindings: list[tuple[str, str]] = []
        if parsed.script_inputs:
            for script_input in parsed.script_inputs:
                live_data: xr.DataArray | None = None
                if external_inputs is not None and script_input.name in external_inputs:
                    live_data = external_inputs[script_input.name]
                elif (
                    live_input_resolver is not None
                    and (resolved := live_input_resolver(script_input)) is not None
                ):
                    live_data = resolved[0]

                if live_data is not None:
                    input_key = graph.add_node(
                        _canonical_key(
                            "live_input",
                            {
                                "name": script_input.name,
                                "node_snapshot_token": script_input.node_snapshot_token,
                                "node_uid": script_input.node_uid,
                            },
                        ),
                        "live_input",
                        cacheable=False,
                        payload={"data": live_data},
                    )
                else:
                    input_spec = script_input.parsed_provenance_spec()
                    if input_spec is None:
                        input_reference = _script_input_reference_text(script_input)
                        raise ReplayGraphError(
                            f"{input_reference} "
                            "does not contain recorded source provenance"
                        )
                    input_key = _compile_spec(
                        graph,
                        input_spec,
                        display=display,
                        trusted_user_code=trusted_user_code,
                        structured_file_replay=structured_file_replay,
                        external_inputs=external_inputs,
                        live_input_resolver=live_input_resolver,
                    )
                    if display:
                        graph.add_alias(script_input.name, input_key)
                bindings.append((script_input.name, input_key))
        elif external_inputs:
            for name, data in external_inputs.items():
                input_key = graph.add_node(
                    _canonical_key("external_input", {"name": name}),
                    "live_input",
                    cacheable=False,
                    payload={"data": data},
                )
                bindings.append((name, input_key))
        else:
            for name in sorted(
                caller_names - _REPLAY_FRAMEWORK_IMPORTS.keys() - {"load_script"}
            ):
                input_key = graph.add_node(
                    _canonical_key("caller_input", {"name": name}),
                    "caller_input",
                    cacheable=False,
                    payload={"name": name},
                )
                bindings.append((name, input_key))

        active_name = typing.cast("str", parsed.active_name)
        script_current_key: str | None = None
        current_name: str | None = None
        current_bindings = tuple(bindings)
        pending_codes: list[str] = []
        pending_code_hoist_imports: list[bool] = []
        pending_code_external_names: list[tuple[str, ...]] = []
        pending_code_framework_owned: list[bool] = []

        def relay_key(source_key: str) -> str:
            return graph.add_node(
                _canonical_key("relay", {"parent": source_key}),
                "relay",
                parents=(source_key,),
            )

        def binding_key(name: str) -> str | None:
            for binding_name, key in current_bindings:
                if binding_name == name:
                    return key
            return None

        def bind_name(name: str, key: str) -> tuple[tuple[str, str], ...]:
            output: list[tuple[str, str]] = []
            replaced = False
            for binding_name, binding_key_value in current_bindings:
                if binding_name == name:
                    if not replaced:
                        output.append((name, key))
                        replaced = True
                    continue
                output.append((binding_name, binding_key_value))
            if not replaced:
                output.append((name, key))
            return tuple(output)

        def ensure_script_current_key() -> None:
            nonlocal current_name, script_current_key
            flush_script()
            if script_current_key is not None:
                return
            current_names = tuple(
                name for name in (current_name, active_name, "derived") if name
            )
            matching_inputs = list(
                dict.fromkeys(
                    key for name, key in current_bindings if name in current_names
                )
            )
            if len(matching_inputs) != 1:
                raise ReplayGraphError("Script provenance has no replay code")
            script_current_key = relay_key(matching_inputs[0])
            current_name = current_names[0]

        def apply_simple_alias(code: str) -> bool:
            nonlocal current_bindings, current_name, script_current_key
            if pending_codes:
                return False
            assignment = _simple_name_assignment(code)
            if assignment is None:
                return False
            target_name, source_name = assignment
            if target_name not in {active_name, "derived"}:
                return False
            source_key = binding_key(source_name)
            if source_key is None:
                return False
            script_current_key = relay_key(source_key)
            current_name = target_name
            current_bindings = bind_name(target_name, script_current_key)
            return True

        def flush_script() -> None:
            nonlocal current_bindings, current_name, pending_code_hoist_imports
            nonlocal pending_code_external_names, pending_code_framework_owned
            nonlocal pending_codes, script_current_key
            if not pending_codes:
                return
            if len(pending_codes) > 1:
                seed_assignment = _simple_name_assignment(pending_codes[0])
                if seed_assignment is not None:
                    seed_name, _source_name = seed_assignment
                    if not _name_value_is_live(pending_codes[1:], seed_name):
                        del pending_codes[0]
                        del pending_code_external_names[0]
                        del pending_code_framework_owned[0]
                        del pending_code_hoist_imports[0]
            output_name = _script_codes_output_name(
                pending_codes,
                active_name=active_name,
                current_name=current_name,
            )
            if output_name is None:  # pragma: no cover - rejected by validation.
                raise ReplayGraphError("Script provenance has no replay code")
            script_current_key = graph.add_node(
                _canonical_key(
                    "script",
                    {
                        "active_name": output_name,
                        "bindings": current_bindings,
                        "codes": tuple(pending_codes),
                        "external_names": tuple(pending_code_external_names),
                        "framework_owned": tuple(pending_code_framework_owned),
                        "hoist_imports": tuple(pending_code_hoist_imports),
                    },
                ),
                "script",
                parents=tuple(key for _name, key in current_bindings),
                cacheable=False,
                payload={
                    "active_name": output_name,
                    "bindings": current_bindings,
                    "codes": tuple(pending_codes),
                    "external_names": tuple(pending_code_external_names),
                    "framework_owned": tuple(pending_code_framework_owned),
                    "hoist_imports": tuple(pending_code_hoist_imports),
                },
            )
            current_name = output_name
            current_bindings = bind_name(output_name, script_current_key)
            if display and _is_semantic_replay_name(output_name):
                graph.add_alias(output_name, script_current_key)
            pending_codes = []
            pending_code_external_names = []
            pending_code_framework_owned = []
            pending_code_hoist_imports = []

        def apply_context_binding(names: Sequence[str]) -> None:
            nonlocal current_bindings, current_name, script_current_key
            ensure_script_current_key()
            current_key = typing.cast("str", script_current_key)
            if display:
                for name in (current_name, active_name):
                    if name is not None:
                        graph.add_alias(name, current_key)
            for name in names:
                current_bindings = bind_name(name, current_key)

        if parsed.file_load_source is not None and _is_extension_loader_source(
            parsed.file_load_source
        ):
            script_current_key = _add_file_load_node(
                graph,
                parsed.file_load_source,
                setup_code=None,
                load_code=None,
                active_name=active_name,
            )
            current_name = active_name
            current_bindings = bind_name(active_name, script_current_key)
        elif parsed.seed_code:
            seed_code = parsed.seed_code
            seed_file_load_parts = None
            if parsed.file_load_source is not None:
                seed_file_load_parts = _script_seed_file_load_parts(
                    seed_code,
                    active_name=active_name,
                    load_source=parsed.file_load_source,
                )
            if seed_file_load_parts is None:
                if not apply_simple_alias(seed_code):
                    pending_codes.append(seed_code)
                    pending_code_external_names.append(tuple(sorted(seed_caller_names)))
                    pending_code_framework_owned.append(False)
                    pending_code_hoist_imports.append(False)
            else:
                seed_setup_code, seed_load_code, seed_output_name = seed_file_load_parts
                script_current_key = _add_file_load_node(
                    graph,
                    parsed.file_load_source,
                    setup_code=seed_setup_code,
                    load_code=seed_load_code,
                    active_name=seed_output_name,
                )
                current_name = seed_output_name
                current_bindings = bind_name(seed_output_name, script_current_key)
        operations = tuple(parsed.operations)
        legacy_context_keys: dict[int, str] = {}
        previous_input_policy: str | None = None
        for index, (step, operation) in enumerate(
            zip(parsed.steps, operations, strict=True)
        ):
            if step.context_names:
                apply_context_binding(step.context_names)
            if getattr(operation, "op", None) == "script_code":
                script_operation = typing.cast("ScriptCodeOperation", operation)
                operation_code = script_operation.code
                if not script_operation.copyable or operation_code is None:
                    raise ReplayGraphError(
                        "Script provenance contains non-replayable code"
                    )
                if display and operation_code in {
                    "derived = derived.isel()",
                    "derived = derived.qsel()",
                    "derived = derived.sel()",
                }:
                    previous_input_policy = step.input_policy
                    continue
                if not apply_simple_alias(operation_code):
                    pending_codes.append(operation_code)
                    accesses, _rebindings = _code_name_accesses(operation_code)
                    external_names = caller_names & accesses
                    if script_operation.framework_owned:
                        external_names -= _REPLAY_FRAMEWORK_IMPORTS.keys()
                    pending_code_external_names.append(tuple(sorted(external_names)))
                    pending_code_framework_owned.append(
                        script_operation.framework_owned
                    )
                    pending_code_hoist_imports.append(
                        bool(getattr(script_operation, "hoist_imports", False))
                    )
                previous_input_policy = None
                continue

            if step.input_policy is not None or step.legacy_context is not None:
                flush_script()
                ensure_script_current_key()
                current_key = typing.cast("str", script_current_key)
                context_key = current_key
                input_policy = step.input_policy
                if step.legacy_context is not None:
                    legacy_index = step.legacy_context.index
                    if legacy_index not in legacy_context_keys:
                        legacy_context_keys[legacy_index] = current_key
                        input_policy = step.legacy_context.input_policy
                    else:
                        input_policy = None
                    context_key = legacy_context_keys[legacy_index]
                if input_policy in {"current", "restored"} and (
                    step.legacy_context is not None
                    or previous_input_policy != input_policy
                ):
                    source_kind = (
                        "full_data" if input_policy == "current" else "public_data"
                    )
                    current_key = graph.add_node(
                        _canonical_key(
                            "source_view",
                            {"parent": current_key, "source_kind": source_kind},
                        ),
                        "source_view",
                        parents=(current_key,),
                        payload={"source_kind": source_kind},
                    )
                    if step.legacy_context is None:
                        context_key = current_key
                script_current_key = current_key
                current_bindings = bind_name(
                    typing.cast("str", current_name), script_current_key
                )
            else:
                context_key = script_current_key or ""

            if display:
                entry = operation.derivation_entry()
                if entry.code in {
                    "derived = derived.isel()",
                    "derived = derived.qsel()",
                    "derived = derived.sel()",
                }:
                    previous_input_policy = step.input_policy
                    continue
                if getattr(operation, "op", None) == "source_view" or (
                    getattr(operation, "op", None) == "restore_nonuniform_dims"
                    and getattr(operation, "dimension_mapping", None) is None
                ):
                    previous_input_policy = step.input_policy
                    continue

            preferred_name = operation.preferred_replay_output_name()
            if pending_codes and (
                preferred_name is not None
                or getattr(operation, "op", None) == "extension_routine"
            ):
                flush_script()

            if pending_codes:
                pending_output_name = _script_codes_output_name(
                    pending_codes,
                    active_name=active_name,
                    current_name=current_name,
                )
                if pending_output_name is None:  # pragma: no cover - validation guard.
                    raise ReplayGraphError("Script provenance has no replay code")
                operation_name = (
                    active_name if index == len(operations) - 1 else pending_output_name
                )
                if operation_name in _REPLAY_FRAMEWORK_IMPORTS:
                    flush_script()
                    ensure_script_current_key()
                    current_key = typing.cast("str", script_current_key)
                    if step.input_policy is None and step.legacy_context is None:
                        context_key = current_key
                    operation_name = active_name
                else:
                    pending_codes.append(
                        _operation_replay_code(
                            operation,
                            active_name=operation_name,
                            context_name=pending_output_name,
                            parent_name=pending_output_name,
                            reserved_names=graph.reserved_names,
                        )
                    )
                    pending_code_external_names.append(())
                    pending_code_framework_owned.append(True)
                    pending_code_hoist_imports.append(False)
                    continue

            ensure_script_current_key()
            current_key = typing.cast("str", script_current_key)
            if step.input_policy is None and step.legacy_context is None:
                context_key = current_key
            if preferred_name is not None:
                operation_name = preferred_name
            elif index == len(operations) - 1:
                operation_name = active_name
            else:
                operation_name = current_name or active_name
            script_current_key = graph.add_node(
                _canonical_key(
                    "operation",
                    {
                        "context": context_key,
                        "legacy_parent_context": step.legacy_context is not None,
                        "operation": operation.model_dump(mode="json"),
                        "parent": current_key,
                    },
                ),
                "operation",
                parents=(current_key, context_key),
                payload={
                    "operation": operation,
                    "legacy_parent_context": step.legacy_context is not None,
                },
            )
            previous_input_policy = step.input_policy
            current_name = operation_name
            current_bindings = bind_name(operation_name, script_current_key)

        flush_script()
        if script_current_key is None:
            matching_inputs = [
                key for name, key in current_bindings if name == active_name
            ]
            if len(matching_inputs) != 1:
                raise ReplayGraphError("Script provenance has no replay code")
            script_current_key = relay_key(matching_inputs[0])
            current_name = active_name
        if current_name != active_name:
            active_key = binding_key(active_name)
            if active_key is None:  # pragma: no cover - validation guard.
                raise ReplayGraphError("Script provenance has no replay code")
            script_current_key = relay_key(active_key)
            current_name = active_name
        if display:
            graph.add_alias(active_name, script_current_key)
        return script_current_key

    raise ReplayGraphError(f"{parsed.kind!r} provenance is not self-contained")


def compile_replay_graph(
    spec: typing.Any,
    *,
    display: bool = False,
    trusted_user_code: bool = False,
    structured_file_replay: bool = False,
    external_inputs: Mapping[str, xr.DataArray] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> ReplayGraph:
    """Compile provenance for code output or structured runtime execution.

    ``structured_file_replay`` makes a recorded replay call authoritative and omits
    copied file-load Python from runtime nodes. Code-output callers keep it false.
    """
    parsed = parse_tool_provenance_spec(spec)
    if parsed is None:
        raise ReplayGraphError("Expected provenance spec")
    reserved_names = _reserved_names_from_spec(parsed)
    if external_inputs:
        reserved_names.update(external_inputs)
    graph = ReplayGraph(
        reserved_names=reserved_names,
        display=display,
        trusted_user_code=trusted_user_code,
    )
    graph.output_key = _compile_spec(
        graph,
        parsed,
        display=display,
        trusted_user_code=trusted_user_code,
        structured_file_replay=structured_file_replay,
        external_inputs=external_inputs,
        live_input_resolver=live_input_resolver,
    )
    return graph


def _node_may_contain_image_tool_dimensions(
    graph: ReplayGraph,
    key: str,
) -> bool:
    """Return whether exact code replay may hold ImageTool rendering dimensions."""
    node_by_key = {node.key: node for node in graph.nodes}

    def visit(node_key: str) -> bool:
        node = node_by_key[node_key]
        if node.kind in {"script", "live_input", "caller_input"}:
            return True
        if node.kind in {"file_load", "setup"}:
            return False
        if node.kind == "source_view" and node.payload["source_kind"] != "full_data":
            return False
        if node.kind == "operation":
            operation = node.payload["operation"]
            operation_name = getattr(operation, "op", None)
            if operation_name == "source_view":
                return False
            # A recorded mapping may restore only some internal dimensions. Only
            # the dynamic operation inspects every dimension that reaches it.
            if (
                operation_name == "restore_nonuniform_dims"
                and getattr(operation, "dimension_mapping", None) is None
            ):
                return False
        return bool(node.parents) and visit(node.parents[0])

    return visit(key)


def _source_view_emits_code(graph: ReplayGraph, node: ReplayNode) -> bool:
    """Return whether this semantic view requires standalone conversion code."""
    if graph.display or node.payload["source_kind"] == "full_data":
        return False
    return _node_may_contain_image_tool_dimensions(graph, node.parents[0])


def _dotted_import_root_bindings(code: str) -> set[str]:
    """Return module-scope roots bound by unaliased dotted imports."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return set()

    roots: set[str] = set()

    class ImportVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            roots.update(
                alias.name.partition(".")[0]
                for alias in node.names
                if alias.asname is None and "." in alias.name
            )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    ImportVisitor().visit(module)
    return roots


def _node_names(
    graph: ReplayGraph,
    *,
    output_name: str | None = None,
) -> dict[str, str]:
    node_by_key = {node.key: node for node in graph.nodes}

    def emitted_key(key: str) -> str:
        node = node_by_key[key]
        while node.kind == "relay" or (
            node.kind == "source_view" and not _source_view_emits_code(graph, node)
        ):
            key = node.parents[0]
            node = node_by_key[key]
        return key

    copied_script_bindings = _copied_script_bindings(graph)
    copied_names_by_key: dict[str, set[str]] = {}
    for _node_key, input_name, input_key in copied_script_bindings:
        copied_names_by_key.setdefault(emitted_key(input_key), set()).add(input_name)
    preferred_names: dict[str, str] = {}
    for node in graph.nodes:
        if node.kind == "caller_input":
            preferred_names[node.key] = typing.cast("str", node.payload["name"])
    for public_name, key in graph.aliases:
        emitted_alias_key = emitted_key(key)
        if public_name not in copied_names_by_key.get(emitted_alias_key, set()):
            preferred_names.setdefault(emitted_alias_key, public_name)
    for node in graph.nodes:
        if node.kind != "operation":
            continue
        preferred_output_name = node.payload["operation"].preferred_replay_output_name()
        if preferred_output_name is not None:
            preferred_names[node.key] = preferred_output_name
    if output_name is not None:
        if graph.output_key is None:
            raise ReplayGraphError("Replay graph has no output")
        preferred_names[emitted_key(graph.output_key)] = output_name

    reserved_names = graph.reserved_names
    uses_extension_scripts = any(
        _is_extension_loader_node(node)
        or (
            node.kind == "operation"
            and getattr(node.payload.get("operation"), "op", None)
            == "extension_routine"
        )
        for node in graph.nodes
    )
    if uses_extension_scripts:
        reserved_names.add("load_script")
    dotted_import_roots: set[str] = set()
    if graph.display:
        for node in graph.nodes:
            code_fragments = [
                value
                for field in ("code", "load_code")
                if isinstance((value := node.payload.get(field)), str)
                and not (_is_extension_loader_node(node) and field == "load_code")
            ]
            code_fragments.extend(
                code for code in node.payload.get("codes", ()) if isinstance(code, str)
            )
            operation_code = getattr(node.payload.get("operation"), "code", None)
            if isinstance(operation_code, str):
                code_fragments.append(operation_code)
            for code in code_fragments:
                accesses, rebindings = _code_name_accesses(code)
                reserved_names.update(accesses)
                reserved_names.update(rebindings)
                dotted_import_roots.update(_dotted_import_root_bindings(code))
    for node in graph.nodes:
        if node.kind != "operation" or not node.parents:
            continue
        preferred_input_name = node.payload["operation"].preferred_replay_input_name()
        parent_key = emitted_key(node.parents[0])
        if (
            preferred_input_name is not None
            and preferred_input_name not in reserved_names
            and preferred_input_name not in preferred_names.values()
        ):
            preferred_names.setdefault(parent_key, preferred_input_name)

    names: dict[str, str] = {}
    used = reserved_names
    counter = 0

    def next_temp() -> str:
        nonlocal counter
        while True:
            name = f"{_REPLAY_TEMP_PREFIX}{counter}"
            counter += 1
            if name not in used:
                used.add(name)
                return name

    for node in graph.nodes:
        if (
            node.kind == "setup"
            or node.kind == "relay"
            or (node.kind == "source_view" and not _source_view_emits_code(graph, node))
        ):
            continue
        preferred_name = preferred_names.get(node.key)
        if (
            preferred_name is not None
            and preferred_name not in names.values()
            and preferred_name not in dotted_import_roots
            and preferred_name not in _REPLAY_FRAMEWORK_IMPORTS
            and not (uses_extension_scripts and preferred_name == "load_script")
        ):
            names[node.key] = preferred_name
            used.add(preferred_name)
        else:
            names[node.key] = next_temp()

    for node in graph.nodes:
        if node.kind == "relay" or (
            node.kind == "source_view" and not _source_view_emits_code(graph, node)
        ):
            names[node.key] = names[emitted_key(node.key)]

    data_consumer_counts = Counter(
        emitted_key(node.parents[0])
        for node in graph.nodes
        if node.parents
        and node.kind not in {"setup", "relay"}
        and not (
            node.kind == "source_view" and not _source_view_emits_code(graph, node)
        )
    )
    for node in graph.nodes:
        if node.kind != "operation" or not node.parents:
            continue
        if not getattr(node.payload["operation"], "statement_mutates_input", False):
            continue
        parent_key = emitted_key(node.parents[0])
        if data_consumer_counts[parent_key] == 1:
            names[node.key] = names[parent_key]

    if not graph.display and output_name is not None and graph.output_key is not None:
        current_key = emitted_key(graph.output_key)
        while True:
            current_node = node_by_key[current_key]
            if current_node.kind != "operation" or not current_node.parents:
                break
            parent_key = emitted_key(current_node.parents[0])
            parent_node = node_by_key[parent_key]
            current_mutates = bool(
                getattr(
                    current_node.payload["operation"],
                    "statement_mutates_input",
                    False,
                )
            )
            parent_mutates = parent_node.kind == "operation" and bool(
                getattr(
                    parent_node.payload["operation"],
                    "statement_mutates_input",
                    False,
                )
            )
            if data_consumer_counts[parent_key] != 1 or not (
                current_mutates or parent_mutates
            ):
                break
            names[current_key] = output_name
            names[parent_key] = output_name
            current_key = parent_key
    for node in graph.nodes:
        if node.kind == "relay" or (
            node.kind == "source_view" and not _source_view_emits_code(graph, node)
        ):
            names[node.key] = names[emitted_key(node.key)]
    return names


def _semantic_emitted_step_count(graph: ReplayGraph) -> int:
    """Count user-visible producers before display-code simplification."""
    count = 0
    for node in graph.nodes:
        if node.kind == "file_load":
            count += 1
        elif node.kind == "operation":
            if _operation_uses_dynamic_nonuniform_restore(graph, node):
                continue
            count += 1
        elif node.kind == "script":
            count += len(typing.cast("tuple[str, ...]", node.payload["codes"]))
    return count


def _copied_script_bindings(graph: ReplayGraph) -> set[tuple[str, str, str]]:
    """Return shared bindings that need independent script-owned arrays.

    Runtime replay gives every script binding its own deep copy. Emitted code preserves
    that ownership wherever two bindings share the same value or a value-producing
    graph ancestor. Setup nodes are excluded because sharing import/setup code does not
    imply sharing an array.
    """
    node_by_key = {node.key: node for node in graph.nodes}
    bindings = [
        (node.key, input_name, input_key)
        for node in graph.nodes
        if node.kind == "script"
        for input_name, input_key in typing.cast(
            "tuple[tuple[str, str], ...]", node.payload["bindings"]
        )
        if node_by_key[input_key].kind != "caller_input"
    ]
    ancestors_by_key: dict[str, frozenset[str]] = {}

    def value_ancestors(key: str) -> frozenset[str]:
        if key not in ancestors_by_key:
            node = node_by_key[key]
            ancestors = set() if node.kind == "setup" else {key}
            for parent_key in node.parents:
                ancestors.update(value_ancestors(parent_key))
            ancestors_by_key[key] = frozenset(ancestors)
        return ancestors_by_key[key]

    binding_counts = Counter(input_key for _node_key, _name, input_key in bindings)
    copied: set[tuple[str, str, str]] = set()
    for binding in bindings:
        _node_key, _input_name, input_key = binding
        if binding_counts[input_key] > 1 or any(
            binding != other_binding
            and not value_ancestors(input_key).isdisjoint(
                value_ancestors(other_binding[2])
            )
            for other_binding in bindings
        ):
            copied.add(binding)
    return copied


def _inline_adjacent_replay_assignments(
    code: str, *, protected_names: set[str] | None = None
) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    changed = False
    protected_names = set() if protected_names is None else protected_names

    def clone_expr(expr: ast.expr) -> ast.expr:
        return ast.parse(ast.unparse(expr), mode="eval").body

    while True:
        for idx, stmt in enumerate(module.body[:-1]):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id in protected_names:
                continue
            next_stmt = module.body[idx + 1]
            if not isinstance(
                next_stmt, ast.Assign
            ) or not _expression_starts_with_name(next_stmt.value, target.id):
                continue
            if _statement_load_count(next_stmt, target.id) != 1:
                continue
            if any(
                _statement_load_count(later_stmt, target.id) > 0
                for later_stmt in module.body[idx + 2 :]
            ):
                continue

            inline_target = target.id
            inline_value = clone_expr(stmt.value)

            class ReplayNameInliner(ast.NodeTransformer):
                def __init__(self, target_name: str, value: ast.expr) -> None:
                    self.target_name = target_name
                    self.value = value

                def visit_Name(self, node: ast.Name) -> ast.expr:
                    if node.id == self.target_name and isinstance(node.ctx, ast.Load):
                        return ast.copy_location(clone_expr(self.value), node)
                    return node

            module.body[idx + 1] = ast.fix_missing_locations(
                typing.cast(
                    "ast.stmt",
                    ReplayNameInliner(inline_target, inline_value).visit(next_stmt),
                )
            )
            del module.body[idx]
            changed = True
            break
        else:
            break

    if not changed:
        return code
    return ast.unparse(ast.fix_missing_locations(module))


def _inline_single_use_replay_names(code: str) -> str:
    """Inline one-use replay temporaries bound to side-effect-free names."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    changed = False

    class ReplayNameInliner(ast.NodeTransformer):
        def __init__(self, target_name: str, source_name: str) -> None:
            self.target_name = target_name
            self.source_name = source_name

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == self.target_name and isinstance(node.ctx, ast.Load):
                return ast.copy_location(
                    ast.Name(self.source_name, ctx=ast.Load()),
                    node,
                )
            return node

    while True:
        for idx, statement in enumerate(module.body[:-1]):
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or not statement.targets[0].id.startswith(_REPLAY_TEMP_PREFIX)
                or not isinstance(statement.value, ast.Name)
            ):
                continue
            target_name = statement.targets[0].id
            source_name = statement.value.id
            later_loads = [
                later_idx
                for later_idx, later in enumerate(module.body[idx + 1 :], start=idx + 1)
                if _statement_load_count(later, target_name)
            ]
            if len(later_loads) != 1:
                continue
            use_idx = later_loads[0]
            use_statement = module.body[use_idx]
            intervening = module.body[idx + 1 : use_idx]
            if (
                not isinstance(use_statement, ast.Assign)
                or _statement_load_count(use_statement, target_name) != 1
                or any(
                    not isinstance(item, ast.Assign)
                    or len(item.targets) != 1
                    or not isinstance(item.targets[0], ast.Name)
                    or not isinstance(item.value, ast.Name)
                    for item in intervening
                )
                or any(
                    _statement_store_count(item, target_name)
                    or _statement_store_count(item, source_name)
                    for item in intervening
                )
            ):
                continue
            module.body[use_idx] = ast.fix_missing_locations(
                typing.cast(
                    "ast.stmt",
                    ReplayNameInliner(target_name, source_name).visit(use_statement),
                )
            )
            del module.body[idx]
            changed = True
            break
        else:
            break

    if not changed:
        return code
    return ast.unparse(ast.fix_missing_locations(module))


def _remove_noop_assignments(code: str) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    original_count = len(module.body)
    module.body = [
        stmt
        for stmt in module.body
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
            and stmt.targets[0].id == stmt.value.id
        )
    ]
    if len(module.body) == original_count:
        return code
    return ast.unparse(ast.fix_missing_locations(module))


def _standalone_statement_span(
    code: str, statement: ast.stmt
) -> tuple[int, int] | None:
    """Return the byte span of a statement that occupies complete source lines."""
    if statement.end_lineno is None or statement.end_col_offset is None:
        return None
    encoded = code.encode()
    line_starts = [0]
    for line in encoded.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    start = line_starts[statement.lineno - 1] + statement.col_offset
    end = line_starts[statement.end_lineno - 1] + statement.end_col_offset
    line_start = line_starts[statement.lineno - 1]
    line_end = line_starts[statement.end_lineno]
    if encoded[line_start:start].strip() or encoded[end:line_end].strip():
        return None
    return line_start, line_end


def _replace_statement_load_name(
    code: str,
    statement: ast.stmt,
    *,
    target_name: str,
    source_name: str,
) -> str:
    """Replace identifier loads in one statement without reformatting its source."""
    encoded = code.encode()
    line_starts = [0]
    for line in encoded.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    spans = sorted(
        (
            line_starts[node.lineno - 1] + node.col_offset,
            line_starts[node.end_lineno - 1] + node.end_col_offset,
        )
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == target_name
        and node.end_lineno is not None
        and node.end_col_offset is not None
    )
    replacement = source_name.encode()
    for start, end in reversed(spans):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode()


def _replace_ast_names(
    code: str,
    module: ast.Module,
    replacements: Mapping[str, str],
) -> str:
    """Replace parsed name nodes without changing unrelated source formatting."""
    encoded = code.encode()
    source_lines = code.splitlines(keepends=True)
    line_starts = [0]
    for line in encoded.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    spans = [
        (
            line_starts[node.lineno - 1] + node.col_offset,
            line_starts[node.end_lineno - 1] + node.end_col_offset,
            replacements[node.id].encode(),
        )
        for node in ast.walk(module)
        if isinstance(node, ast.Name)
        and node.id in replacements
        and node.end_lineno is not None
        and node.end_col_offset is not None
    ]
    tokens = iter(tokenize.generate_tokens(io.StringIO(code).readline))
    for token in tokens:
        if token.type != tokenize.NAME or token.string not in {"class", "def"}:
            continue
        for name_token in tokens:
            if name_token.type != tokenize.NAME:
                continue
            replacement = replacements.get(name_token.string)
            if replacement is not None:
                line = source_lines[name_token.start[0] - 1]
                name_start = line_starts[name_token.start[0] - 1] + len(
                    line[: name_token.start[1]].encode()
                )
                spans.append(
                    (
                        name_start,
                        name_start + len(name_token.string.encode()),
                        replacement.encode(),
                    )
                )
            break
    for statement in ast.walk(module):
        if not isinstance(statement, ast.Import | ast.ImportFrom):
            continue
        for alias in statement.names:
            if (
                alias.name == "*"
                or alias.end_lineno is None
                or alias.end_col_offset is None
            ):
                continue
            bound_name = alias.asname or (
                alias.name.partition(".")[0]
                if isinstance(statement, ast.Import)
                else alias.name
            )
            replacement = replacements.get(bound_name)
            if replacement is None:
                continue
            alias_end = line_starts[alias.end_lineno - 1] + alias.end_col_offset
            if alias.asname is not None:
                spans.append(
                    (
                        alias_end - len(alias.asname.encode()),
                        alias_end,
                        replacement.encode(),
                    )
                )
            elif not isinstance(statement, ast.Import) or "." not in alias.name:
                spans.append((alias_end, alias_end, f" as {replacement}".encode()))
    for start, end, replacement in sorted(spans, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode()


def _replace_script_output_identifiers(
    codes: Sequence[str],
    *,
    previous_name: str,
    output_name: str,
) -> list[str]:
    """Rename one script output without changing earlier uses of the same name."""
    if previous_name == output_name:
        return list(codes)

    renamed: list[str] = []
    output_bound = False
    for code in codes:
        if output_bound:
            renamed.append(
                _replace_code_identifiers(code, {previous_name: output_name})
            )
            continue

        module = ast.parse(code, mode="exec")

        def is_simple_output_binding(statement: ast.stmt) -> bool:
            return (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == previous_name
            ) or (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == previous_name
            )

        def binds_only_unaliased_dotted_import(statement: ast.stmt) -> bool:
            if not isinstance(statement, ast.Import):
                return False
            matching = [
                alias
                for alias in statement.names
                if alias.asname is None
                and "." in alias.name
                and alias.name.partition(".")[0] == previous_name
            ]
            return (
                bool(matching)
                and previous_name in _statement_scope_names(statement).stores
            )

        simple_binding_index = next(
            (
                index
                for index, statement in enumerate(module.body)
                if is_simple_output_binding(statement)
            ),
            None,
        )
        binding_statement = (
            module.body[simple_binding_index]
            if simple_binding_index is not None
            else None
        )
        if simple_binding_index is not None:
            earlier_binding = next(
                (
                    statement
                    for statement in module.body[:simple_binding_index]
                    if previous_name in _statement_scope_names(statement).stores
                    and not binds_only_unaliased_dotted_import(statement)
                ),
                None,
            )
            if earlier_binding is not None:
                binding_statement = earlier_binding
        if binding_statement is None:
            binding_statement = next(
                (
                    statement
                    for statement in module.body
                    if previous_name in _statement_scope_names(statement).stores
                ),
                None,
            )
        if binding_statement is None:
            renamed.append(code)
            continue

        output_bound = True
        target: ast.Name | None = None
        if (
            isinstance(binding_statement, ast.Assign)
            and len(binding_statement.targets) == 1
            and isinstance(binding_statement.targets[0], ast.Name)
            and binding_statement.targets[0].id == previous_name
        ):
            target = binding_statement.targets[0]
        elif (
            isinstance(binding_statement, ast.AnnAssign)
            and isinstance(binding_statement.target, ast.Name)
            and binding_statement.target.id == previous_name
        ):
            target = binding_statement.target

        statement_span = _standalone_statement_span(code, binding_statement)
        if (
            target is None
            or target.end_lineno is None
            or target.end_col_offset is None
            or statement_span is None
        ):
            statement_start = statement_span[0] if statement_span is not None else 0
            encoded = code.encode()
            renamed.append(
                encoded[:statement_start].decode()
                + _replace_code_identifiers(
                    encoded[statement_start:].decode(),
                    {previous_name: output_name},
                )
            )
            continue

        encoded = code.encode()
        line_starts = [0]
        for line in encoded.splitlines(keepends=True):
            line_starts.append(line_starts[-1] + len(line))
        target_start = line_starts[target.lineno - 1] + target.col_offset
        target_end = line_starts[target.end_lineno - 1] + target.end_col_offset
        statement_start, statement_end = statement_span
        statement = encoded[statement_start:statement_end]
        relative_start = target_start - statement_start
        relative_end = target_end - statement_start
        renamed_statement = (
            statement[:relative_start] + output_name.encode() + statement[relative_end:]
        ).decode()
        renamed_tail = _replace_code_identifiers(
            encoded[statement_end:].decode(), {previous_name: output_name}
        )
        renamed.append(
            encoded[:statement_start].decode() + renamed_statement + renamed_tail
        )
    return renamed


def _format_long_call_assignments(code: str, *, line_length: int = 88) -> str:
    """Wrap top-level call assignments made too long by semantic renaming."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    encoded = code.encode()
    replacements: list[tuple[int, int, bytes]] = []
    for statement in module.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or not isinstance(statement.value, ast.Call)
        ):
            continue
        statement_code = ast.get_source_segment(code, statement)
        expression_code = ast.get_source_segment(code, statement.value)
        span = _standalone_statement_span(code, statement)
        if (
            statement_code is None
            or expression_code is None
            or span is None
            or max(map(len, statement_code.splitlines())) <= line_length
        ):
            continue
        formatted = _assignment_code(
            statement.targets[0].id,
            expression_code,
            line_length=line_length,
        )
        start, end = span
        suffix = b"\n" if encoded[start:end].endswith(b"\n") else b""
        replacements.append((start, end, formatted.encode() + suffix))
    for start, end, replacement in reversed(replacements):
        encoded = encoded[:start] + replacement + encoded[end:]
    return encoded.decode()


def _inline_standalone_alias(
    code: str,
    alias_statement: ast.stmt,
    use_statement: ast.stmt,
    *,
    target_name: str,
    source_name: str,
) -> str | None:
    """Inline one standalone alias without changing surrounding formatting."""
    alias_span = _standalone_statement_span(code, alias_statement)
    if alias_span is None:
        return None
    code = _replace_statement_load_name(
        code,
        use_statement,
        target_name=target_name,
        source_name=source_name,
    )
    alias_start, alias_end = alias_span
    encoded = code.encode()
    return (encoded[:alias_start] + encoded[alias_end:]).decode().strip("\n")


def _inline_simple_name_aliases(code: str) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    if _code_has_scoped_definition(code):
        return code

    changed = False

    class SimpleAliasInliner(ast.NodeTransformer):
        def __init__(self, target_name: str, source_name: str) -> None:
            self.target_name = target_name
            self.source_name = source_name

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == self.target_name and isinstance(node.ctx, ast.Load):
                return ast.copy_location(
                    ast.Name(self.source_name, ctx=ast.Load()),
                    node,
                )
            return node

    def receiver_root_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return receiver_root_name(node.func)
        if isinstance(node, ast.Attribute | ast.Subscript):
            return receiver_root_name(node.value)
        return None

    def receiver_load_count(stmt: ast.stmt, target_name: str) -> int:
        if not isinstance(stmt, ast.Assign):
            return 0
        load_count = _statement_load_count(stmt, target_name)
        if receiver_root_name(stmt.value) == target_name:
            return load_count
        return 0

    def statement_rebinds_name(stmt: ast.stmt, name: str) -> bool:
        return bool(_statement_store_count(stmt, name)) or (
            name in _statement_scope_names(stmt).stores
        )

    while True:
        for idx, stmt in enumerate(module.body[:-1]):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or not isinstance(stmt.value, ast.Name):
                continue
            target_name = target.id
            source_name = stmt.value.id
            if target_name == source_name:
                continue

            later_statements = module.body[idx + 1 :]
            load_indices = [
                later_idx
                for later_idx, later_stmt in enumerate(
                    later_statements,
                    start=idx + 1,
                )
                if _statement_load_count(later_stmt, target_name)
            ]
            if len(load_indices) == 1:
                use_idx = load_indices[0]
                preceding = module.body[idx + 1 : use_idx]
                use_statement = module.body[use_idx]
                use_has_prior_call = (
                    isinstance(use_statement, ast.Assign)
                    and receiver_root_name(use_statement.value) != target_name
                    and any(
                        isinstance(node, ast.Call)
                        for node in ast.walk(use_statement.value)
                    )
                )
                if (
                    not use_has_prior_call
                    and not statement_rebinds_name(use_statement, target_name)
                    and not any(
                        statement_rebinds_name(item, target_name)
                        or statement_rebinds_name(item, source_name)
                        for item in preceding
                    )
                ):
                    inlined_code = _inline_standalone_alias(
                        code,
                        stmt,
                        use_statement,
                        target_name=target_name,
                        source_name=source_name,
                    )
                    if inlined_code is not None:
                        code = inlined_code
                        module = ast.parse(code, mode="exec")
                    else:
                        inliner = SimpleAliasInliner(target_name, source_name)
                        cloned = ast.parse(
                            ast.unparse(module.body[use_idx]), mode="exec"
                        ).body[0]
                        module.body[use_idx] = ast.fix_missing_locations(
                            typing.cast("ast.stmt", inliner.visit(cloned))
                        )
                        del module.body[idx]
                        code = ast.unparse(ast.fix_missing_locations(module))
                        module = ast.parse(code, mode="exec")
                    changed = True
                    break

            rewrite_end: int | None = None
            can_rewrite = True
            for later_idx, later_stmt in enumerate(
                module.body[idx + 1 :],
                start=idx + 1,
            ):
                if statement_rebinds_name(later_stmt, source_name):
                    can_rewrite = False
                    break
                if not statement_rebinds_name(later_stmt, target_name):
                    continue
                if (
                    isinstance(later_stmt, ast.Assign)
                    and len(later_stmt.targets) == 1
                    and isinstance(later_stmt.targets[0], ast.Name)
                    and later_stmt.targets[0].id == target_name
                    and _statement_load_count(
                        later_stmt,
                        target_name,
                    )
                ):
                    rewrite_end = later_idx
                else:
                    can_rewrite = False
                break
            if rewrite_end != idx + 1 or not can_rewrite:
                continue
            if any(
                statement_rebinds_name(intervening, target_name)
                for intervening in module.body[idx + 1 : rewrite_end]
            ):
                continue
            rewrite_statements = module.body[idx + 1 : rewrite_end + 1]
            if any(
                _statement_load_count(item, target_name)
                != receiver_load_count(item, target_name)
                for item in rewrite_statements
            ):
                continue

            inlined_code = _inline_standalone_alias(
                code,
                stmt,
                rewrite_statements[0],
                target_name=target_name,
                source_name=source_name,
            )
            if inlined_code is not None:
                code = inlined_code
                module = ast.parse(code, mode="exec")
                changed = True
                break

            inliner = SimpleAliasInliner(target_name, source_name)
            rewritten: list[ast.stmt] = []
            for item in rewrite_statements:
                cloned = ast.parse(ast.unparse(item), mode="exec").body[0]
                rewritten.append(
                    ast.fix_missing_locations(
                        typing.cast("ast.stmt", inliner.visit(cloned))
                    )
                )
            module.body[idx + 1 : rewrite_end + 1] = rewritten
            del module.body[idx]
            code = ast.unparse(ast.fix_missing_locations(module))
            module = ast.parse(code, mode="exec")
            changed = True
            break
        else:
            break

    if not changed:
        return code
    return code


def _compact_replay_temp_names(code: str) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code

    replacements: dict[str, str] = {}
    used_names = {
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and not node.id.startswith(_REPLAY_TEMP_PREFIX)
    }

    def compact_name(name: str) -> str:
        if not name.startswith(_REPLAY_TEMP_PREFIX):
            return name
        if name in replacements:
            return replacements[name]
        index = len(replacements)
        while True:
            candidate = f"{_REPLAY_TEMP_PREFIX}{index}"
            index += 1
            if candidate not in used_names and candidate not in replacements.values():
                replacements[name] = candidate
                return candidate

    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            compact_name(node.id)
    if not replacements or all(key == value for key, value in replacements.items()):
        return code

    renamed = _replace_ast_names(code, module, replacements)
    return _format_long_call_assignments(renamed)


def _code_has_scoped_definition(code: str) -> bool:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return True
    return any(
        isinstance(
            node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Lambda)
        )
        for node in ast.walk(module)
    )


def _is_receiver_assignment_chain(codes: Sequence[str], active_name: str) -> bool:
    """Return whether code chunks are a side-effect-free receiver transform chain."""

    def receiver_root_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return receiver_root_name(node.func)
        if isinstance(node, ast.Attribute | ast.Subscript):
            return receiver_root_name(node.value)
        return None

    if len(codes) < 2:
        return False
    for index, code in enumerate(codes):
        try:
            module = ast.parse(code, mode="exec")
        except SyntaxError:
            return False
        if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
            return False
        statement = module.body[0]
        if (
            len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id != active_name
        ):
            return False
        if index and (
            receiver_root_name(statement.value) != active_name
            or _statement_load_count(statement, active_name) != 1
        ):
            return False
    return True


def _remove_unused_generated_copies(code: str, names: set[str]) -> str:
    if not names:
        return code
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    retained: list[ast.stmt] = []
    changed = False
    for index, statement in enumerate(module.body):
        target_name = (
            statement.targets[0].id
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            else None
        )
        if target_name not in names:
            retained.append(statement)
            continue
        later_code = ast.unparse(
            ast.Module(body=module.body[index + 1 :], type_ignores=[])
        )
        if _code_uses_name(later_code, target_name):
            retained.append(statement)
            continue
        changed = True
    if not changed:
        return code
    module.body = retained
    return ast.unparse(ast.fix_missing_locations(module))


def _cleanup_emitted_replay_code(
    code: str,
    *,
    generated_copy_names: set[str],
    protected_names: set[str] | None = None,
    compact_temporaries: bool = True,
) -> str:
    """Simplify replay code while optionally preserving graph-owned temp names.

    Display emission preserves the original temporary identifiers until semantic
    renaming can associate each surviving binding with the graph node that owns it.
    """
    code = _remove_noop_assignments(code)
    code = _inline_simple_name_aliases(code)
    code = _inline_adjacent_replay_assignments(code, protected_names=protected_names)
    code = _inline_single_use_replay_names(code)
    code = _inline_adjacent_replay_assignments(code, protected_names=protected_names)
    code = _remove_noop_assignments(code)
    code = _remove_unused_generated_copies(code, generated_copy_names)
    if compact_temporaries:
        return _compact_replay_temp_names(code)
    return code


def _leading_top_level_imports(code: str) -> tuple[list[tuple[str, str]], str]:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return [], code

    lines = code.splitlines()
    imports: list[tuple[str, str]] = []
    removed_lines: set[int] = set()
    for statement in module.body:
        if not isinstance(statement, ast.Import | ast.ImportFrom):
            break
        if statement.end_lineno is None or statement.end_col_offset is None:
            break
        start_index = statement.lineno - 1
        end_index = statement.end_lineno - 1
        if (
            lines[start_index][: statement.col_offset].strip()
            or lines[end_index][statement.end_col_offset :].strip()
        ):
            break
        source = "\n".join(lines[start_index : end_index + 1]).strip()
        imports.append((ast.unparse(statement), source))
        removed_lines.update(range(start_index, end_index + 1))

    if not imports:
        return [], code
    body = "\n".join(
        line for index, line in enumerate(lines) if index not in removed_lines
    ).strip("\n")
    return imports, body


def _split_module_prologue(code: str) -> tuple[str, str]:
    """Split off a leading module docstring and its future imports.

    Generated support definitions must follow both constructs: moving a definition
    ahead of the docstring changes ``__doc__``, while moving it ahead of a future
    import makes otherwise valid replay code fail to compile.
    """
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return "", code
    if not module.body:
        return "", code

    statement_index = 0
    prologue_end: ast.stmt | None = None
    first_statement = module.body[0]
    if (
        isinstance(first_statement, ast.Expr)
        and isinstance(first_statement.value, ast.Constant)
        and isinstance(first_statement.value.value, str)
    ):
        prologue_end = first_statement
        statement_index = 1

    while statement_index < len(module.body):
        statement = module.body[statement_index]
        if not (
            isinstance(statement, ast.ImportFrom) and statement.module == "__future__"
        ):
            break
        prologue_end = statement
        statement_index += 1

    if (
        prologue_end is None
        or prologue_end.end_lineno is None
        or prologue_end.end_col_offset is None
    ):
        return "", code

    encoded = code.encode()
    line_starts = [0]
    for line in encoded.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    split_at = line_starts[prologue_end.end_lineno - 1] + prologue_end.end_col_offset

    line_end = line_starts[prologue_end.end_lineno]
    line_suffix = encoded[split_at:line_end]
    if not line_suffix.strip() or line_suffix.lstrip().startswith(b"#"):
        split_at = line_end

    prologue = encoded[:split_at].decode().strip("\n")
    body = encoded[split_at:].decode().lstrip(" \t")
    if body.startswith(";"):
        body = body[1:].lstrip(" \t")
    return prologue, body.strip("\n")


def _partition_module_prologue(code: str) -> tuple[str | None, str, str]:
    """Separate one chunk's docstring, future imports, and executable body.

    Each replay chunk is valid as an independent module. The combined replay module
    can keep only one module docstring, but every future import must precede generated
    extension setup and support definitions.
    """
    prologue, body = _split_module_prologue(code)
    if not prologue:
        return None, "", code

    module = ast.parse(prologue, mode="exec")
    first_statement = module.body[0]
    has_docstring = (
        isinstance(first_statement, ast.Expr)
        and isinstance(first_statement.value, ast.Constant)
        and isinstance(first_statement.value.value, str)
    )
    first_future = next(
        (
            statement
            for statement in module.body
            if isinstance(statement, ast.ImportFrom)
            and statement.module == "__future__"
        ),
        None,
    )
    if first_future is None:
        return (prologue if has_docstring else None), "", body
    if not has_docstring:
        return None, prologue, body

    encoded = prologue.encode()
    line_starts = [0]
    for line in encoded.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    future_start = line_starts[first_future.lineno - 1] + first_future.col_offset
    prefix = encoded[:future_start].decode().rstrip("\n")
    future_imports = encoded[future_start:].decode().strip("\n")
    return prefix, future_imports, body


def _import_binding_targets(code: str) -> dict[str, str] | None:
    """Return names bound by one import and their canonical targets.

    Star imports cannot be reasoned about locally, so callers must leave them in place.
    """
    statement = ast.parse(code, mode="exec").body[0]
    if isinstance(statement, ast.Import):
        targets: dict[str, str] = {}
        for alias in statement.names:
            if alias.asname is None:
                name = alias.name.partition(".")[0]
                target = f"module:{name}"
            else:
                name = alias.asname
                target = f"module:{alias.name}"
            if name in targets and targets[name] != target:
                return None
            targets[name] = target
        return targets
    if not isinstance(statement, ast.ImportFrom):  # pragma: no cover - caller contract.
        return None
    if statement.module == "__future__":
        return {}
    targets = {}
    module = "." * statement.level + (statement.module or "")
    for alias in statement.names:
        if alias.name == "*":
            return None
        name = alias.asname or alias.name
        target = f"from:{module}:{alias.name}"
        if name in targets and targets[name] != target:
            return None
        targets[name] = target
    return targets


def _code_name_accesses(code: str) -> tuple[set[str], set[str]]:
    """Return conservatively accessed and rebound names in a code chunk."""
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        # Invalid generated code is diagnosed by the normal replay-code path. Treat it
        # as opaque so import grouping cannot make its behavior less predictable.
        return {"*"}, {"*"}

    accessed: set[str] = set()
    rebound: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            accessed.add(node.id)
            if isinstance(node.ctx, ast.Store | ast.Del):
                rebound.add(node.id)
        elif isinstance(node, ast.arg):
            accessed.add(node.arg)
            rebound.add(node.arg)
        elif isinstance(node, ast.AsyncFunctionDef | ast.ClassDef | ast.FunctionDef):
            accessed.add(node.name)
            rebound.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            bindings = _import_binding_targets(ast.unparse(node))
            if bindings is None:
                accessed.add("*")
                rebound.add("*")
            else:
                accessed.update(bindings)
                rebound.update(bindings)
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar):
            if node.name is not None:
                accessed.add(node.name)
                rebound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            accessed.add(node.rest)
            rebound.add(node.rest)
    return accessed, rebound


def _is_future_import(code: str) -> bool:
    statement = ast.parse(code, mode="exec").body[0]
    return isinstance(statement, ast.ImportFrom) and statement.module == "__future__"


def _group_framework_imports(chunks: Sequence[tuple[str, bool]]) -> str:
    """Group canonical framework imports without changing Python name resolution.

    Framework code producers should use one canonical import target for each alias. If
    independently generated chunks violate that invariant, keep the imports beside the
    code that uses them instead of silently changing the program by hoisting them.
    """
    parsed_chunks: list[tuple[str, bool, list[tuple[str, str]], str]] = []
    targets_by_name: dict[str, set[str]] = {}
    for code, group_imports in chunks:
        observed_imports, observed_body = _leading_top_level_imports(code)
        leading_imports = observed_imports if group_imports else []
        import_body = observed_body if group_imports else code
        parsed_chunks.append((code, group_imports, leading_imports, import_body))
        for canonical, _source in observed_imports:
            bindings = _import_binding_targets(canonical)
            if bindings is None:
                continue
            for name, target in bindings.items():
                targets_by_name.setdefault(name, set()).add(target)
    conflicting_names = {
        name for name, targets in targets_by_name.items() if len(targets) > 1
    }

    future_imports: list[str] = []
    imports: list[str] = []
    import_codes: set[str] = set()
    hoisted_bindings: dict[str, str] = {}
    seen_accesses: set[str] = set()
    seen_rebindings: set[str] = set()
    body: list[str] = []
    for original_code, group_imports, leading_imports, import_body in parsed_chunks:
        code = original_code
        if group_imports and leading_imports:
            regular_imports = [
                item for item in leading_imports if not _is_future_import(item[0])
            ]
            future_items = [
                item for item in leading_imports if _is_future_import(item[0])
            ]
            for canonical, source in future_items:
                if canonical not in import_codes:
                    import_codes.add(canonical)
                    future_imports.append(source)

            regular_bindings: dict[str, str] = {}
            safe_to_hoist = True
            for canonical, _source in regular_imports:
                bindings = _import_binding_targets(canonical)
                if bindings is None:
                    safe_to_hoist = False
                    break
                for name, target in bindings.items():
                    existing_target = hoisted_bindings.get(name)
                    if (
                        name in conflicting_names
                        or "*" in seen_accesses
                        or (existing_target is None and name in seen_accesses)
                        or (
                            existing_target is not None
                            and (existing_target != target or name in seen_rebindings)
                        )
                    ):
                        safe_to_hoist = False
                        break
                    regular_bindings[name] = target
                if not safe_to_hoist:
                    break

            if safe_to_hoist:
                code = import_body
                hoisted_bindings.update(regular_bindings)
                for canonical, source in regular_imports:
                    if canonical in import_codes:
                        continue
                    import_codes.add(canonical)
                    imports.append(source)
            elif future_items:
                # Future imports must remain at the beginning of the combined module;
                # retain every ordinary import next to its original body.
                code = "\n".join(
                    (*[source for _canonical, source in regular_imports], import_body)
                ).strip("\n")
        if code.strip():
            body.append(code)
        accesses, rebindings = _code_name_accesses(code)
        seen_accesses.update(accesses)
        seen_rebindings.update(rebindings)
    return "\n".join((*future_imports, *imports, *body))


def _operation_uses_dynamic_nonuniform_restore(
    graph: ReplayGraph,
    node: ReplayNode,
) -> bool:
    operation = node.payload["operation"]
    operation_name = getattr(operation, "op", None)
    if operation_name == "source_view":
        # An explicit source-view operation is the complete durable representation
        # of an operation-free public/selection source. Unlike an input-policy view
        # before another operation, it cannot be omitted from copied code without
        # dropping the source transformation itself.
        return getattr(operation, "source_kind", None) != "full_data"
    return (
        operation_name == "restore_nonuniform_dims"
        and getattr(operation, "dimension_mapping", None) is None
    )


def _dynamic_nonuniform_restore_name(
    graph: ReplayGraph,
    names: Mapping[str, str],
) -> str:
    """Return a support-function name that cannot shadow emitted replay code."""
    unavailable = graph.reserved_names | set(names.values())
    for node in graph.nodes:
        code_fragments: list[str] = []
        for field in ("code", "load_code"):
            value = node.payload.get(field)
            if isinstance(value, str):
                code_fragments.append(value)
        code_fragments.extend(
            code for code in node.payload.get("codes", ()) if isinstance(code, str)
        )
        operation_code = getattr(node.payload.get("operation"), "code", None)
        if isinstance(operation_code, str):
            code_fragments.append(operation_code)
        for code in code_fragments:
            accesses, rebindings = _code_name_accesses(code)
            unavailable.update(accesses)
            unavailable.update(rebindings)

    base_name = _NONUNIFORM_RESTORE_FUNCTION_NAME
    function_name = base_name
    suffix = 2
    while function_name in unavailable:
        function_name = f"{base_name}_{suffix}"
        suffix += 1
    return function_name


def _rename_display_replay_temps(
    graph: ReplayGraph,
    names: Mapping[str, str],
    code: str,
) -> str:
    """Replace surviving graph temporaries with collision-safe semantic names."""
    if not graph.display or _REPLAY_TEMP_PREFIX not in code:
        return code

    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:  # pragma: no cover - emitted code is validated upstream.
        return code
    surviving_temps = {
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and node.id.startswith(_REPLAY_TEMP_PREFIX)
    }
    if not surviving_temps:
        return code

    accessed, rebound = _code_name_accesses(code)
    used_names = {
        name
        for name in graph.reserved_names | accessed | rebound
        if not name.startswith(_REPLAY_TEMP_PREFIX)
    }
    replacements: dict[str, str] = {}
    for node in graph.nodes:
        temporary_name = names.get(node.key)
        if temporary_name not in surviving_temps or temporary_name in replacements:
            continue
        if node.kind == "file_load":
            base_name = "loaded_data"
        elif node.kind == "source_view":
            base_name = "source_data"
        elif node.kind == "operation":
            preferred_output_method = getattr(
                node.payload["operation"],
                "preferred_replay_output_name",
                None,
            )
            preferred_output = (
                preferred_output_method() if callable(preferred_output_method) else None
            )
            base_name = preferred_output or "processed_data"
        else:
            base_name = "script_result"

        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        replacements[temporary_name] = candidate
        used_names.add(candidate)

    renamed = _replace_ast_names(code, module, replacements)
    return _format_long_call_assignments(renamed)


def _extension_script_references(
    graph: ReplayGraph,
) -> dict[str, tuple[pathlib.Path, str]]:
    """Resolve each graph-owned extension call to one current local script."""
    from erlab.extensions import (
        ExtensionNotFoundError,
        LoaderDescriptor,
        RoutineDescriptor,
    )
    from erlab.extensions._api import _resolve_registered_script_capability
    from erlab.interactive.imagetool._provenance._operations._extension import (
        ExtensionRoutineOperation,
    )

    references: dict[str, tuple[pathlib.Path, str]] = {}
    for node in graph.nodes:
        capability_kind: typing.Literal["routine", "loader"]
        if _is_extension_loader_node(node):
            replay_call = node.payload["load_source"].replay_call
            script_name = replay_call.target
            capability_kind = "loader"
            capability_id = replay_call.capability_id
        elif node.kind == "operation" and isinstance(
            node.payload.get("operation"), ExtensionRoutineOperation
        ):
            operation = typing.cast(
                "ExtensionRoutineOperation", node.payload["operation"]
            )
            script_name = operation.script_name
            capability_kind = "routine"
            capability_id = operation.routine_id
        else:
            continue
        if not isinstance(capability_id, str):
            raise ReplayGraphError("Extension replay metadata is incomplete")
        try:
            reference = _resolve_registered_script_capability(
                script_name,
                capability_kind,
                capability_id,
            )
        except ExtensionNotFoundError as exc:
            raise ReplayGraphError(
                "Extension call does not have a registered local script"
            ) from exc
        descriptor = reference.descriptor
        expected_type = (
            RoutineDescriptor if capability_kind == "routine" else LoaderDescriptor
        )
        if not isinstance(descriptor, expected_type):
            raise ReplayGraphError("Registered extension capability has the wrong type")
        references[node.key] = (
            pathlib.Path(reference.registered_path).expanduser().resolve(),
            descriptor.function_name,
        )
    return references


def emit_replay_code(
    graph: ReplayGraph,
    *,
    output_name: str | None = None,
    include_all_aliases: bool = False,
) -> str:
    from erlab.interactive.imagetool._provenance._operations._extension import (
        ExtensionRoutineOperation,
    )

    names = _node_names(graph, output_name=output_name)
    node_by_key = {node.key: node for node in graph.nodes}
    extension_references = _extension_script_references(graph)
    copied_script_bindings = _copied_script_bindings(graph)
    chunks: list[tuple[str, bool]] = []
    chunk_external_names: list[tuple[str, ...]] = []
    chunk_framework_owned: list[bool] = []
    extension_setup_chunks: list[tuple[str, bool]] = []
    materialized_aliases: dict[str, str] = {}
    generated_copy_names: set[str] = set()
    materialized_binding_names = set(names.values())
    reserved_binding_names: set[str] = set()
    for replay_node in graph.nodes:
        for field in ("code", "load_code"):
            code = replay_node.payload.get(field)
            if isinstance(code, str) and not (
                _is_extension_loader_node(replay_node) and field == "load_code"
            ):
                accesses, rebindings = _code_name_accesses(code)
                reserved_binding_names.update(accesses)
                reserved_binding_names.update(rebindings)
        for code in replay_node.payload.get("codes", ()):
            if isinstance(code, str):
                accesses, rebindings = _code_name_accesses(code)
                reserved_binding_names.update(accesses)
                reserved_binding_names.update(rebindings)
    dynamic_restore_needed = any(
        (node.kind == "source_view" and _source_view_emits_code(graph, node))
        or (
            node.kind == "operation"
            and _operation_uses_dynamic_nonuniform_restore(graph, node)
        )
        for node in graph.nodes
    )
    dynamic_restore_name = (
        _dynamic_nonuniform_restore_name(graph, names)
        if dynamic_restore_needed
        else None
    )
    dynamic_restore_support = (
        _nonuniform_restore_support_code(dynamic_restore_name)
        if dynamic_restore_name is not None
        else None
    )
    extension_binding_names = {
        *graph.reserved_names,
        *reserved_binding_names,
        *names.values(),
        *_REPLAY_ALIASES,
        "erlab",
        "load_script",
        "np",
        "numpy",
        "pathlib",
        "xr",
        "xarray",
    }
    if dynamic_restore_name is not None:
        extension_binding_names.add(dynamic_restore_name)
    external_load_script_name: str | None = None
    if extension_references and "load_script" in graph.external_names:
        external_load_script_name = "input_data"
        suffix = 2
        while external_load_script_name in extension_binding_names:
            external_load_script_name = f"input_data_{suffix}"
            suffix += 1
        extension_binding_names.add(external_load_script_name)
    extension_script_bindings: dict[pathlib.Path, str] = {}
    for node in graph.nodes:
        reference = extension_references.get(node.key)
        if reference is None:
            continue
        source_path, _function_name = reference
        if source_path in extension_script_bindings:
            continue
        module_base = re.sub(r"\W", "_", source_path.stem)
        if not module_base.isidentifier() or keyword.iskeyword(module_base):
            module_base = "extension_script"
        module_name = module_base
        suffix = 2
        while module_name in extension_binding_names:
            module_name = f"{module_base}_{suffix}"
            suffix += 1
        extension_binding_names.add(module_name)
        extension_script_bindings[source_path] = module_name
    active_setup_key: str | None = None

    def append_code(
        code: str,
        *,
        group_imports: bool = False,
        external_names: tuple[str, ...] = (),
        framework_owned: bool = True,
    ) -> None:
        chunks.append((code, graph.display and group_imports))
        chunk_external_names.append(external_names)
        chunk_framework_owned.append(framework_owned)

    def extension_binding(node_key: str) -> tuple[str, str]:
        source_path, function_name = extension_references[node_key]
        module_name = extension_script_bindings[source_path]
        return module_name, function_name

    if extension_script_bindings:
        if external_load_script_name is not None:
            extension_setup_chunks.append(
                (f"{external_load_script_name} = load_script", False)
            )
        extension_setup_chunks.append(
            ("from erlab.extensions import load_script", graph.display)
        )
        for source_path, module_name in extension_script_bindings.items():
            extension_setup_chunks.append(
                (f"{module_name} = load_script({str(source_path)!r})", False)
            )

    def copied_binding_name(input_name: str) -> str:
        if input_name not in materialized_binding_names:
            materialized_binding_names.add(input_name)
            return input_name
        suffix = 2
        while True:
            candidate = f"{input_name}_{suffix}"
            if (
                candidate not in materialized_binding_names
                and candidate not in reserved_binding_names
            ):
                materialized_binding_names.add(candidate)
                return candidate
            suffix += 1

    for node in graph.nodes:
        if node.kind == "setup":
            continue
        if node.kind == "relay":
            continue
        name = names[node.key]
        if node.kind == "file_load":
            if _is_extension_loader_node(node):
                from erlab.interactive.imagetool._load_source import (
                    _extension_loader_load_code,
                )

                module_name, function_name = extension_binding(node.key)
                code = _extension_loader_load_code(
                    node.payload["load_source"],
                    assign=name,
                    loader_expression=f"{module_name}.{function_name}",
                )
                if code is None:
                    raise ReplayGraphError(
                        "Extension loader does not provide copied code"
                    )
                append_code(code, group_imports=True)
                continue
            active_name = typing.cast("str", node.payload["active_name"])
            load_code = node.payload["load_code"]
            if not isinstance(load_code, str):
                raise ReplayGraphError("File source does not provide copied code")
            setup_key = node.parents[0] if node.parents else None
            if setup_key is not None and active_setup_key != setup_key:
                setup_node = node_by_key[setup_key]
                append_code(
                    typing.cast("str", setup_node.payload["code"]),
                    group_imports=True,
                )
                active_setup_key = setup_key
            try:
                code = _replace_code_identifiers(load_code, {active_name: name})
            except SyntaxError as exc:
                raise ReplayGraphError("File replay code is not valid Python") from exc
            if not _code_stores_name(code, name):
                raise ReplayGraphError("File replay code does not assign its output")
            append_code(code, group_imports=True)
        elif node.kind == "caller_input":
            continue
        elif node.kind == "live_input":
            raise ReplayGraphError("Live inputs cannot be emitted as replay code")
        elif node.kind == "source_view":
            if not _source_view_emits_code(graph, node):
                continue
            if dynamic_restore_name is None:
                raise ReplayGraphError("Public source view has no restore support")
            parent_name = names[node.parents[0]]
            append_code(
                f"{name} = {dynamic_restore_name}({parent_name}.copy(deep=False))"
            )
        elif node.kind == "operation":
            parent_name = names[node.parents[0]]
            context_name = names[node.parents[1]]
            operation = node.payload["operation"]
            if _operation_uses_dynamic_nonuniform_restore(graph, node):
                if dynamic_restore_name is None:
                    raise ReplayGraphError(
                        "Nonuniform restore operation has no restore support"
                    )
                input_expression = parent_name
                if getattr(operation, "op", None) == "source_view":
                    input_expression = f"{parent_name}.copy(deep=False)"
                append_code(f"{name} = {dynamic_restore_name}({input_expression})")
            elif isinstance(operation, ExtensionRoutineOperation):
                module_name, function_name = extension_binding(node.key)
                append_code(
                    operation._bound_script_statement_code(
                        parent_name,
                        output_name=name,
                        module_name=module_name,
                        function_name=function_name,
                    )
                )
            else:
                append_code(
                    _operation_replay_code(
                        operation,
                        active_name=name,
                        context_name=context_name,
                        parent_name=parent_name,
                        reserved_names=(
                            graph.reserved_names
                            | reserved_binding_names
                            | set(names.values())
                        ),
                    )
                )
        elif node.kind == "script":
            codes = list(typing.cast("tuple[str, ...]", node.payload["codes"]))
            external_names_by_code = list(
                typing.cast(
                    "tuple[tuple[str, ...], ...]",
                    node.payload.get("external_names", ((),) * len(codes)),
                )
            )
            framework_owned_by_code = list(
                typing.cast(
                    "tuple[bool, ...]",
                    node.payload["framework_owned"],
                )
            )
            hoist_imports = list(
                typing.cast(
                    "tuple[bool, ...]",
                    node.payload.get("hoist_imports", (False,) * len(codes)),
                )
            )
            active_name = typing.cast("str", node.payload["active_name"])
            input_replacements: dict[str, str] = {}
            input_names: set[str] = set()
            if external_load_script_name is not None and not any(
                input_name == "load_script"
                for input_name, _input_key in typing.cast(
                    "tuple[tuple[str, str], ...]", node.payload["bindings"]
                )
            ):
                try:
                    codes = [
                        _replace_code_identifiers(
                            code, {"load_script": external_load_script_name}
                        )
                        for code in codes
                    ]
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
                if active_name == "load_script":
                    active_name = external_load_script_name
            for input_name, input_key in typing.cast(
                "tuple[tuple[str, str], ...]", node.payload["bindings"]
            ):
                input_names.add(input_name)
                input_value_name = names[input_key]
                if (node.key, input_name, input_key) in copied_script_bindings:
                    generated_input_name = copied_binding_name(input_name)
                    append_code(
                        f"{generated_input_name} = {input_value_name}.copy(deep=True)"
                    )
                    generated_copy_names.add(generated_input_name)
                    if generated_input_name != input_name:
                        input_replacements[input_name] = generated_input_name
                    materialized_aliases[input_name] = input_value_name
                    continue
                if input_name == input_value_name:
                    continue
                if (
                    graph.display
                    and not any(_code_has_scoped_definition(code) for code in codes)
                    and not _name_value_is_live(codes, input_name)
                ):
                    continue
                if (
                    graph.display
                    and not any(_code_has_scoped_definition(code) for code in codes)
                    and not any(_code_stores_name(code, input_name) for code in codes)
                    and not any(
                        input_value_name in _code_name_accesses(code)[1]
                        for code in codes
                    )
                ):
                    input_replacements[input_name] = input_value_name
                else:
                    append_code(f"{input_name} = {input_value_name}")
                    materialized_aliases[input_name] = input_value_name
            if input_replacements:
                try:
                    codes = [
                        _replace_code_identifiers(code, input_replacements)
                        for code in codes
                    ]
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
                active_name = input_replacements.get(active_name, active_name)
            if (
                graph.display
                and active_name != name
                and active_name not in input_names
                and active_name not in input_replacements.values()
            ):
                try:
                    codes = _replace_script_output_identifiers(
                        codes,
                        previous_name=active_name,
                        output_name=name,
                    )
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
                active_name = name
            if (
                graph.display
                and not any(hoist_imports)
                and len(set(framework_owned_by_code)) == 1
                and _is_receiver_assignment_chain(codes, active_name)
            ):
                codes = [
                    _simplify_display_code(
                        "\n".join(codes),
                        inline_targets={active_name},
                    )
                ]
                external_names_by_code = [
                    tuple(
                        sorted(
                            {
                                name
                                for names_for_code in external_names_by_code
                                for name in names_for_code
                            }
                        )
                    )
                ]
                framework_owned_by_code = [framework_owned_by_code[0]]
                hoist_imports = [False]
            for code, group_imports, external_names, framework_owned in zip(
                codes,
                hoist_imports,
                external_names_by_code,
                framework_owned_by_code,
                strict=True,
            ):
                append_code(
                    code,
                    group_imports=group_imports,
                    external_names=external_names,
                    framework_owned=framework_owned,
                )
            if active_name != name:
                append_code(f"{name} = {active_name}")
            active_setup_key = None
        else:
            raise ReplayGraphError(f"Unknown replay graph node kind {node.kind!r}")

    aliases = list(graph.aliases) if include_all_aliases else []
    if output_name is not None:
        if graph.output_key is None:
            raise ReplayGraphError("Replay graph has no output")
        output_alias = (output_name, graph.output_key)
        output_node = node_by_key[graph.output_key]
        omit_mutating_display_alias = (
            graph.display
            and output_node.kind == "operation"
            and bool(
                getattr(
                    output_node.payload["operation"],
                    "statement_mutates_input",
                    False,
                )
            )
        )
        if output_alias not in aliases and not omit_mutating_display_alias:
            aliases = [*aliases, output_alias]
    for public_name, key in aliases:
        planned_name = names[key]
        if (
            public_name != planned_name
            and materialized_aliases.get(public_name) != planned_name
        ):
            append_code(f"{public_name} = {planned_name}")
    module_docstring: str | None = None
    future_prologues: list[str] = []
    for index, (chunk, group_imports) in enumerate(chunks):
        if not chunk.strip():
            continue
        docstring, chunk_future_imports, body = _partition_module_prologue(chunk)
        if docstring is not None:
            if module_docstring is None:
                module_docstring = docstring
            else:
                body = "\n".join(part for part in (docstring, body) if part)
        if chunk_future_imports:
            future_prologues.append(chunk_future_imports)
        chunks[index] = (body, group_imports)

    framework_owned_names: set[str] = set()
    framework_import_targets: dict[str, str] = {}
    for name, import_code in _REPLAY_FRAMEWORK_IMPORTS.items():
        bindings = _import_binding_targets(import_code)
        if bindings is None or name not in bindings:
            raise ReplayGraphError("Framework import does not bind its public name")
        framework_import_targets[name] = bindings[name]

    known_framework_bindings: dict[str, str] = {}
    synthesize_framework_imports = (
        not graph.display or _semantic_emitted_step_count(graph) > 1
    )
    for index, ((chunk, group_imports), framework_owned) in enumerate(
        zip(chunks, chunk_framework_owned, strict=True)
    ):
        module = ast.parse(chunk, mode="exec")
        loaded_names = {
            node.id
            for node in ast.walk(module)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        imported_bindings: dict[str, str] = {}
        leading_imports, body = _leading_top_level_imports(chunk)
        for canonical, _source in leading_imports:
            bindings = _import_binding_targets(canonical)
            if bindings is not None:
                imported_bindings.update(bindings)

        bindings_at_body = known_framework_bindings | imported_bindings
        if framework_owned:
            framework_names = loaded_names & _REPLAY_FRAMEWORK_IMPORTS.keys()
            framework_owned_names.update(
                framework_names
                | (imported_bindings.keys() & _REPLAY_FRAMEWORK_IMPORTS.keys())
            )
            missing_imports = [
                name
                for name in _REPLAY_FRAMEWORK_IMPORTS
                if name in framework_names
                and (synthesize_framework_imports or name in graph.external_names)
                and bindings_at_body.get(name) != framework_import_targets[name]
            ]
            if missing_imports:
                missing_sources = [
                    _REPLAY_FRAMEWORK_IMPORTS[name] for name in missing_imports
                ]
                chunk = "\n".join(
                    (
                        *(source for _canonical, source in leading_imports),
                        *missing_sources,
                        body,
                    )
                ).strip("\n")
                chunks[index] = (chunk, graph.display or group_imports)
                for name in missing_imports:
                    imported_bindings[name] = framework_import_targets[name]

        known_framework_bindings.update(imported_bindings)
        _accesses, rebound_names = _code_name_accesses(body)
        for name in rebound_names & _REPLAY_FRAMEWORK_IMPORTS.keys():
            known_framework_bindings.pop(name, None)

    external_capture_chunks: list[tuple[str, bool]] = []
    external_name_replacements: dict[str, str] = {}
    for external_name in sorted(
        (graph.external_names & framework_owned_names) - {"load_script"}
    ):
        replacement = "input_data"
        suffix = 2
        while replacement in extension_binding_names:
            replacement = f"input_data_{suffix}"
            suffix += 1
        extension_binding_names.add(replacement)
        external_name_replacements[external_name] = replacement
        external_capture_chunks.append((f"{replacement} = {external_name}", False))

    for index, ((chunk, group_imports), external_names) in enumerate(
        zip(chunks, chunk_external_names, strict=True)
    ):
        replacements = {
            name: external_name_replacements[name]
            for name in external_names
            if name in external_name_replacements
        }
        if replacements:
            chunk = _replace_code_identifiers(chunk, replacements)
        chunks[index] = (chunk, group_imports)

    code = _group_framework_imports(
        (
            *external_capture_chunks,
            *extension_setup_chunks,
            *chunks,
        )
    )
    future_import_code = _group_framework_imports(
        tuple((future_prologue, True) for future_prologue in future_prologues)
    )
    module_prologue = "\n".join(
        part for part in (module_docstring, future_import_code) if part
    )
    if dynamic_restore_support is None:
        cleaned_code = _rename_display_replay_temps(
            graph,
            names,
            _cleanup_emitted_replay_code(
                code,
                generated_copy_names=generated_copy_names,
                protected_names=set(extension_script_bindings.values()),
                compact_temporaries=not graph.display,
            ),
        )
        return "\n\n".join(part for part in (module_prologue, cleaned_code) if part)

    leading_imports, body = _leading_top_level_imports(code)
    cleaned_body = _rename_display_replay_temps(
        graph,
        names,
        _cleanup_emitted_replay_code(
            body,
            generated_copy_names=generated_copy_names,
            protected_names=set(extension_script_bindings.values()),
            compact_temporaries=not graph.display,
        ),
    )
    import_prefix = "\n".join(source for _canonical, source in leading_imports)
    return "\n\n\n".join(
        part
        for part in (
            module_prologue,
            import_prefix,
            dynamic_restore_support,
            cleaned_body,
        )
        if part
    )


def script_inputs_code(script_inputs: Sequence[typing.Any], *, display: bool) -> str:
    reserved_names: set[str] = set()
    for script_input in script_inputs:
        reserved_names.add(script_input.name)
        reserved_names.update(
            _reserved_names_from_spec(script_input.parsed_provenance_spec())
        )
    graph = ReplayGraph(reserved_names=reserved_names, display=display)
    for script_input in script_inputs:
        input_spec = script_input.parsed_provenance_spec()
        if input_spec is None:
            raise ReplayGraphError(
                f"{_script_input_reference_text(script_input)} "
                "does not contain recorded source provenance"
            )
        input_key = _compile_spec(
            graph,
            input_spec,
            display=display,
            trusted_user_code=False,
            structured_file_replay=False,
            external_inputs=None,
            live_input_resolver=None,
        )
        graph.add_alias(script_input.name, input_key)
    return emit_replay_code(graph, include_all_aliases=True)
