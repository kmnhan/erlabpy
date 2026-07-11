"""Internal replay graph for ImageTool provenance.

The saved provenance schema stays in
:mod:`erlab.interactive.imagetool.provenance`.
This module compiles those specs into an execution/code-generation graph at runtime so
shared file loads and shared structured operations are emitted or replayed once.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import keyword
import symtable
import typing
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import xarray as xr

import erlab
from erlab.interactive.imagetool import _provenance_framework


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
        self._reserved_names = set(reserved_names or ())
        self.aliases: list[tuple[str, str]] = []
        self.display = bool(display)
        self.trusted_user_code = bool(trusted_user_code)
        self.output_key: str | None = None

    @property
    def reserved_names(self) -> set[str]:
        return set(self._reserved_names)

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


def _code_uses_name(code: str, name: str) -> bool:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(module)
    )


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


def _script_function_dependencies(code: str) -> dict[tuple[str, int], set[str]]:
    table = symtable.symtable(code, "<ImageTool script provenance>", "exec")

    def child_dependencies(child: symtable.SymbolTable) -> set[str]:
        deps = {
            symbol.get_name()
            for symbol in child.get_symbols()
            if symbol.is_referenced()
            and (symbol.is_global() or symbol.is_free())
            and symbol.get_name() != child.get_name()
        }
        for nested in child.get_children():
            deps.update(child_dependencies(nested))
        return deps

    output: dict[tuple[str, int], set[str]] = {}
    for child in table.get_children():
        if child.get_type() != "function":
            continue
        lineno = child.get_lineno()
        if lineno is None:
            continue
        output[(child.get_name(), lineno)] = child_dependencies(child)
    return output


def _validate_script_code_names(
    code: str,
    available_names: set[str],
    function_dependencies: dict[str, set[str]],
) -> None:
    module = ast.parse(code, mode="exec")
    new_function_dependencies = _script_function_dependencies(code)

    def require(name: str, visiting: set[str] | None = None) -> str | None:
        if name not in available_names:
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


def _simple_assignment_source_name(code: str, target_name: str) -> str | None:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return None
    if len(module.body) != 1:
        return None
    stmt = module.body[0]
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.simple and stmt.value is not None:
        target = stmt.target
        value = stmt.value
    else:
        return None
    if (
        isinstance(target, ast.Name)
        and target.id == target_name
        and isinstance(value, ast.Name)
    ):
        return value.id
    return None


def _validate_script_provenance(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    allow_free_seed_names: bool = False,
    strict_replay_code: bool = True,
) -> None:
    if spec.kind != "script":
        raise ReplayGraphError("Expected script provenance")
    if spec.active_name is None:
        raise ReplayGraphError(
            "Script provenance cannot be replayed without active_name"
        )

    builtin_names = set(_provenance_framework._SCRIPT_REPLAY_ALLOWED_BUILTINS)
    if not strict_replay_code:
        builtin_names.update(vars(builtins))

    available_names = {
        "erlab",
        "np",
        "numpy",
        "xr",
        "xarray",
        *_REPLAY_ALIASES,
        *builtin_names,
    }
    if spec.script_inputs:
        available_names.update(script_input.name for script_input in spec.script_inputs)
    elif external_input_names is not None:
        available_names.update(external_input_names)
    function_dependencies: dict[str, set[str]] = {}
    has_replay_step = False
    active_available = spec.active_name in available_names
    current_name: str | None = spec.active_name if active_available else None
    if spec.seed_code:
        has_replay_step = True
        if strict_replay_code:
            try:
                _provenance_framework._validate_script_replay_code(spec.seed_code)
            except (TypeError, ValueError) as exc:
                raise ReplayGraphError(str(exc)) from exc
        else:
            try:
                ast.parse(spec.seed_code, mode="exec")
            except SyntaxError as exc:
                raise ReplayGraphError(
                    "Script replay code is not valid Python"
                ) from exc
        if allow_free_seed_names:
            module = ast.parse(spec.seed_code, mode="exec")
            for stmt in module.body:
                names = _statement_scope_names(stmt)
                if isinstance(stmt, ast.FunctionDef):
                    function_dependencies[stmt.name] = set()
                available_names.update(names.stores)
        else:
            _validate_script_code_names(
                spec.seed_code,
                available_names,
                function_dependencies,
            )
        active_available = active_available or _provenance_framework._code_stores_name(
            spec.seed_code, spec.active_name
        )
        current_name = _provenance_framework._script_codes_output_name(
            (spec.seed_code,),
            active_name=spec.active_name,
            current_name=current_name,
        )
    context_bindings_by_index: dict[int, list[str]] = {}
    for binding in spec.script_context_bindings:
        context_bindings_by_index.setdefault(binding.operation_index, []).extend(
            binding.names
        )
    for stage in spec.replay_stages:
        if current_name is None:
            raise ReplayGraphError("Script provenance has no replay code")
        if any(not operation.live_applicable for operation in stage.operations):
            raise ReplayGraphError(
                "Script provenance contains non-replayable operation"
            )
        has_replay_step = True
        available_names.add(current_name)
        active_available = active_available or current_name == spec.active_name
    for index, operation in enumerate(spec.operations):
        if context_names := context_bindings_by_index.get(index):
            if current_name is None:
                raise ReplayGraphError("Script provenance has no replay code")
            available_names.update(context_names)
        if getattr(operation, "op", None) == "script_code":
            if not operation.copyable or operation.code is None:
                raise ReplayGraphError("Script provenance contains non-replayable code")
            has_replay_step = True
            if strict_replay_code:
                try:
                    _provenance_framework._validate_script_replay_code(operation.code)
                except (TypeError, ValueError) as exc:
                    raise ReplayGraphError(str(exc)) from exc
            else:
                try:
                    ast.parse(operation.code, mode="exec")
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
            _validate_script_code_names(
                operation.code,
                available_names,
                function_dependencies,
            )
            active_available = (
                active_available
                or _provenance_framework._code_stores_name(
                    operation.code, spec.active_name
                )
            )
            current_name = _provenance_framework._script_codes_output_name(
                (operation.code,),
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
        active_available = active_available or current_name == spec.active_name
    if not has_replay_step:
        raise ReplayGraphError("Script provenance has no replay code")
    if not active_available and current_name != spec.active_name:
        raise ReplayGraphError("Script provenance has no replay code")


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
            if _provenance_framework._statement_store_count(stmt, active_name) > 0
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
    load_code: str,
    active_name: str,
) -> dict[str, typing.Any]:
    payload = typing.cast(
        "dict[str, typing.Any]",
        load_source.model_dump(mode="json"),
    )
    payload["setup_code"] = setup_code
    payload["load_code"] = _canonical_file_load_code(load_code, active_name)
    return payload


def _add_file_load_node(
    graph: ReplayGraph,
    load_source: typing.Any,
    *,
    setup_code: str | None,
    load_code: str,
    active_name: str,
) -> str:
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


def _compile_replay_stages(
    graph: ReplayGraph,
    current_key: str,
    replay_stages: Sequence[typing.Any],
    *,
    display: bool,
) -> str:
    for stage in replay_stages:
        source_parent_key = current_key
        current_key = graph.add_node(
            _canonical_key(
                "source_view",
                {"parent": source_parent_key, "source_kind": stage.source_kind},
            ),
            "source_view",
            parents=(source_parent_key,),
            payload={"source_kind": stage.source_kind},
        )
        operations = stage.operations
        if display:
            operations = (
                _provenance_framework.ToolProvenanceSpec._streamlined_operations(
                    stage.source_kind,
                    operations,
                )
            )
        for operation in operations:
            current_key = graph.add_node(
                _canonical_key(
                    "operation",
                    {
                        "context": source_parent_key,
                        "operation": operation.model_dump(mode="json"),
                        "parent": current_key,
                    },
                ),
                "operation",
                parents=(current_key, source_parent_key),
                payload={"operation": operation},
            )
    return current_key


def _script_seed_file_load_parts(
    seed_code: str,
    *,
    active_name: str,
    load_source: typing.Any,
) -> tuple[str | None, str, str] | None:
    if getattr(load_source, "replay_call", None) is None:
        return None
    seed_output_name = _provenance_framework._script_codes_output_name(
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
) -> str:
    input_name = active_name if parent_name is None else parent_name
    try:
        code = operation.replay_code(
            input_name,
            output_name=active_name,
            source_name=context_name,
        )
    except (AttributeError, NotImplementedError) as exc:
        raise ReplayGraphError("Operation does not provide replay code") from exc
    if code is None:
        raise ReplayGraphError("Operation does not provide replay code")
    try:
        ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ReplayGraphError("Operation replay code is not valid Python") from exc
    if not _provenance_framework._code_stores_name(code, active_name) and not (
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
    external_inputs: Mapping[str, xr.DataArray] | None,
    live_input_resolver: LiveInputResolver | None,
) -> str:
    parsed = _provenance_framework.parse_tool_provenance_spec(spec)
    if parsed is None:
        raise ReplayGraphError("Expected provenance spec")

    if parsed.kind == "file":
        if parsed.file_load_source is None:
            raise ReplayGraphError("File provenance does not define a load source")
        if parsed.active_name is None or parsed.seed_code is None:
            raise ReplayGraphError("File provenance does not define replay code")

        setup_code, load_code = _file_seed_code_parts(
            parsed.seed_code,
            parsed.active_name,
        )
        current_key = _add_file_load_node(
            graph,
            parsed.file_load_source,
            setup_code=setup_code,
            load_code=load_code,
            active_name=parsed.active_name,
        )
        return _compile_replay_stages(
            graph,
            current_key,
            parsed.replay_stages,
            display=display,
        )

    if parsed.kind == "script":
        _validate_script_provenance(
            parsed,
            external_input_names=set(external_inputs or ()),
            allow_free_seed_names=display
            and not parsed.script_inputs
            and not external_inputs,
            strict_replay_code=not display and not trusted_user_code,
        )
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
                        raise ReplayGraphError(
                            f"{script_input.name} does not contain recorded source "
                            "provenance"
                        )
                    input_key = _compile_spec(
                        graph,
                        input_spec,
                        display=display,
                        trusted_user_code=trusted_user_code,
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

        active_name = typing.cast("str", parsed.active_name)
        script_current_key: str | None = None
        current_name: str | None = None
        current_bindings = tuple(bindings)
        pending_codes: list[str] = []
        pending_code_hoist_imports: list[bool] = []

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
            for target_name in dict.fromkeys((active_name, "derived")):
                source_name = _simple_assignment_source_name(code, target_name)
                if source_name is None:
                    continue
                source_key = binding_key(source_name)
                if source_key is None:
                    continue
                script_current_key = relay_key(source_key)
                current_name = target_name
                current_bindings = bind_name(target_name, script_current_key)
                return True
            return False

        def flush_script() -> None:
            nonlocal current_bindings, current_name, pending_code_hoist_imports
            nonlocal pending_codes, script_current_key
            if not pending_codes:
                return
            output_name = _provenance_framework._script_codes_output_name(
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
                    "hoist_imports": tuple(pending_code_hoist_imports),
                },
            )
            current_name = output_name
            current_bindings = bind_name(output_name, script_current_key)
            pending_codes = []
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

        if parsed.seed_code:
            seed_file_load_parts = None
            if parsed.file_load_source is not None:
                seed_file_load_parts = _script_seed_file_load_parts(
                    parsed.seed_code,
                    active_name=active_name,
                    load_source=parsed.file_load_source,
                )
            if seed_file_load_parts is None:
                if not apply_simple_alias(parsed.seed_code):
                    pending_codes.append(parsed.seed_code)
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
        if parsed.replay_stages:
            ensure_script_current_key()
            script_current_key = _compile_replay_stages(
                graph,
                typing.cast("str", script_current_key),
                parsed.replay_stages,
                display=display,
            )
            current_bindings = bind_name(
                typing.cast("str", current_name),
                script_current_key,
            )
        operations = tuple(parsed.operations)
        context_bindings_by_index: dict[int, list[str]] = {}
        for binding in parsed.script_context_bindings:
            context_bindings_by_index.setdefault(binding.operation_index, []).extend(
                binding.names
            )
        for index, operation in enumerate(operations):
            if context_names := context_bindings_by_index.get(index):
                apply_context_binding(context_names)
            if display:
                entry = operation.derivation_entry()
                if entry.code in {
                    "derived = derived.isel()",
                    "derived = derived.qsel()",
                    "derived = derived.sel()",
                } or _provenance_framework._is_internal_sort_coord_order_entry(entry):
                    continue
            if getattr(operation, "op", None) == "script_code":
                operation_code = typing.cast(
                    "str | None", getattr(operation, "code", None)
                )
                if not getattr(operation, "copyable", False) or operation_code is None:
                    raise ReplayGraphError(
                        "Script provenance contains non-replayable code"
                    )
                if not apply_simple_alias(operation_code):
                    pending_codes.append(operation_code)
                    pending_code_hoist_imports.append(
                        bool(getattr(operation, "hoist_imports", False))
                    )
                continue

            if pending_codes:
                pending_output_name = _provenance_framework._script_codes_output_name(
                    pending_codes,
                    active_name=active_name,
                    current_name=current_name,
                )
                if pending_output_name is None:  # pragma: no cover - validation guard.
                    raise ReplayGraphError("Script provenance has no replay code")
                pending_codes.append(
                    _operation_replay_code(
                        operation,
                        active_name=pending_output_name,
                        context_name=pending_output_name,
                    )
                )
                pending_code_hoist_imports.append(False)
                continue

            ensure_script_current_key()
            current_key = typing.cast("str", script_current_key)
            operation_name = current_name or active_name
            script_current_key = graph.add_node(
                _canonical_key(
                    "operation",
                    {
                        "context": current_key,
                        "operation": operation.model_dump(mode="json"),
                        "parent": current_key,
                    },
                ),
                "operation",
                parents=(current_key, current_key),
                payload={"operation": operation},
            )
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
    external_inputs: Mapping[str, xr.DataArray] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> ReplayGraph:
    parsed = _provenance_framework.parse_tool_provenance_spec(spec)
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
        external_inputs=external_inputs,
        live_input_resolver=live_input_resolver,
    )
    return graph


def _source_view_emits_code(graph: ReplayGraph, node: ReplayNode) -> bool:
    if node.payload["source_kind"] == "full_data":
        return False
    return not graph.display


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

    preferred_names: dict[str, str] = {}
    for public_name, key in graph.aliases:
        preferred_names[emitted_key(key)] = public_name
    if output_name is not None:
        if graph.output_key is None:
            raise ReplayGraphError("Replay graph has no output")
        preferred_names[emitted_key(graph.output_key)] = output_name

    names: dict[str, str] = {}
    used = graph.reserved_names
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
        if preferred_name is not None and preferred_name not in names.values():
            names[node.key] = preferred_name
            used.add(preferred_name)
        else:
            names[node.key] = next_temp()

    for node in graph.nodes:
        if node.kind == "relay" or (
            node.kind == "source_view" and not _source_view_emits_code(graph, node)
        ):
            names[node.key] = names[emitted_key(node.key)]
    return names


def _inline_adjacent_replay_assignments(code: str) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    changed = False

    def clone_expr(expr: ast.expr) -> ast.expr:
        return ast.parse(ast.unparse(expr), mode="eval").body

    while True:
        for idx, stmt in enumerate(module.body[:-1]):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or not target.id.startswith(
                _REPLAY_TEMP_PREFIX
            ):
                continue
            next_stmt = module.body[idx + 1]
            if not isinstance(next_stmt, ast.Assign):
                continue
            if _provenance_framework._statement_load_count(next_stmt, target.id) != 1:
                continue
            if any(
                _provenance_framework._statement_load_count(later_stmt, target.id) > 0
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


def _inline_single_use_replay_expressions(code: str) -> str:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError:
        return code
    changed = False

    def clone_expr(expr: ast.expr) -> ast.expr:
        return ast.parse(ast.unparse(expr), mode="eval").body

    def has_call(expr: ast.expr) -> bool:
        return any(isinstance(node, ast.Call) for node in ast.walk(expr))

    while True:
        for idx, stmt in enumerate(module.body[:-1]):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or not target.id.startswith(
                _REPLAY_TEMP_PREFIX
            ):
                continue
            if has_call(stmt.value):
                continue

            later_loads = [
                later_idx
                for later_idx, later in enumerate(module.body[idx + 1 :], start=idx + 1)
                if _provenance_framework._statement_load_count(later, target.id) > 0
            ]
            if len(later_loads) != 1:
                continue
            use_idx = later_loads[0]
            use_stmt = module.body[use_idx]
            if not isinstance(use_stmt, ast.Assign):
                continue
            if _provenance_framework._statement_load_count(use_stmt, target.id) != 1:
                continue

            replacement_load_names = {
                node.id
                for node in ast.walk(stmt.value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            intervening = module.body[idx + 1 : use_idx]
            if any(
                _provenance_framework._statement_store_count(item, target.id)
                for item in intervening
            ):
                continue
            if any(
                _provenance_framework._statement_store_count(item, name)
                for item in intervening
                for name in replacement_load_names
            ):
                continue

            inline_target = target.id
            inline_value = clone_expr(stmt.value)

            class ReplayExpressionInliner(ast.NodeTransformer):
                def __init__(self, target_name: str, value: ast.expr) -> None:
                    self.target_name = target_name
                    self.value = value

                def visit_Name(self, node: ast.Name) -> ast.expr:
                    if node.id == self.target_name and isinstance(node.ctx, ast.Load):
                        return ast.copy_location(clone_expr(self.value), node)
                    return node

            module.body[use_idx] = ast.fix_missing_locations(
                typing.cast(
                    "ast.stmt",
                    ReplayExpressionInliner(inline_target, inline_value).visit(
                        use_stmt
                    ),
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

    def kspace_receiver_load_count(stmt: ast.stmt, target_name: str) -> int:
        return sum(
            1
            for node in ast.walk(stmt)
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "kspace"
                and isinstance(node.value, ast.Name)
                and node.value.id == target_name
                and isinstance(node.value.ctx, ast.Load)
            )
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

            rewrite_end: int | None = None
            can_rewrite = True
            for later_idx, later_stmt in enumerate(
                module.body[idx + 1 :],
                start=idx + 1,
            ):
                if _provenance_framework._statement_store_count(
                    later_stmt,
                    source_name,
                ):
                    can_rewrite = False
                    break
                if not _provenance_framework._statement_store_count(
                    later_stmt,
                    target_name,
                ):
                    continue
                if (
                    isinstance(later_stmt, ast.Assign)
                    and len(later_stmt.targets) == 1
                    and isinstance(later_stmt.targets[0], ast.Name)
                    and later_stmt.targets[0].id == target_name
                    and _provenance_framework._statement_load_count(
                        later_stmt,
                        target_name,
                    )
                ):
                    rewrite_end = later_idx
                else:
                    can_rewrite = False
                break
            if rewrite_end is None or not can_rewrite:
                continue
            if any(
                _provenance_framework._statement_store_count(
                    intervening,
                    target_name,
                )
                for intervening in module.body[idx + 1 : rewrite_end]
            ):
                continue
            rewrite_statements = module.body[idx + 1 : rewrite_end + 1]
            if any(
                _provenance_framework._statement_load_count(item, target_name)
                != kspace_receiver_load_count(item, target_name)
                for item in rewrite_statements
            ):
                continue

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
            changed = True
            break
        else:
            break

    if not changed:
        return code
    return ast.unparse(ast.fix_missing_locations(module))


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

    class ReplayTempCompactor(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            replacement = replacements.get(node.id)
            if replacement is None:
                return node
            return ast.copy_location(ast.Name(replacement, ctx=node.ctx), node)

    compacted = typing.cast("ast.Module", ReplayTempCompactor().visit(module))
    return ast.unparse(ast.fix_missing_locations(compacted))


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


def _cleanup_emitted_replay_code(code: str) -> str:
    code = _remove_noop_assignments(code)
    code = _inline_simple_name_aliases(code)
    code = _inline_adjacent_replay_assignments(code)
    code = _inline_single_use_replay_expressions(code)
    code = _remove_noop_assignments(code)
    return _compact_replay_temp_names(code)


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


def _group_framework_imports(chunks: Sequence[tuple[str, bool]]) -> str:
    imports: list[str] = []
    import_codes: set[str] = set()
    body: list[str] = []
    for code, group_imports in chunks:
        if group_imports:
            leading_imports, code = _leading_top_level_imports(code)
            for canonical, source in leading_imports:
                if canonical in import_codes:
                    continue
                import_codes.add(canonical)
                imports.append(source)
        if code.strip():
            body.append(code)
    return "\n".join((*imports, *body))


def emit_replay_code(graph: ReplayGraph, *, output_name: str | None = None) -> str:
    names = _node_names(graph, output_name=output_name)
    node_by_key = {node.key: node for node in graph.nodes}
    chunks: list[tuple[str, bool]] = []
    active_setup_key: str | None = None

    def append_code(code: str, *, group_imports: bool = False) -> None:
        chunks.append((code, graph.display and group_imports))

    for node in graph.nodes:
        if node.kind == "setup":
            continue
        if node.kind == "relay":
            continue
        name = names[node.key]
        if node.kind == "file_load":
            active_name = typing.cast("str", node.payload["active_name"])
            load_code = typing.cast("str", node.payload["load_code"])
            setup_key = node.parents[0] if node.parents else None
            if setup_key is not None and active_setup_key != setup_key:
                setup_node = node_by_key[setup_key]
                append_code(
                    typing.cast("str", setup_node.payload["code"]),
                    group_imports=True,
                )
                active_setup_key = setup_key
            try:
                code = _provenance_framework._replace_code_identifiers(
                    load_code, {active_name: name}
                )
            except SyntaxError as exc:
                raise ReplayGraphError("File replay code is not valid Python") from exc
            if not _provenance_framework._code_stores_name(code, name):
                raise ReplayGraphError("File replay code does not assign its output")
            append_code(code, group_imports=True)
        elif node.kind == "live_input":
            raise ReplayGraphError("Live inputs cannot be emitted as replay code")
        elif node.kind == "source_view":
            parent_name = names[node.parents[0]]
            if not _source_view_emits_code(graph, node):
                continue
            append_code(
                f"{name} = erlab.interactive.imagetool.slicer."
                f"restore_nonuniform_dims({parent_name})"
            )
        elif node.kind == "operation":
            parent_name = names[node.parents[0]]
            context_name = names[node.parents[1]]
            operation = node.payload["operation"]
            append_code(
                _operation_replay_code(
                    operation,
                    active_name=name,
                    context_name=context_name,
                    parent_name=parent_name,
                )
            )
        elif node.kind == "script":
            codes = list(typing.cast("tuple[str, ...]", node.payload["codes"]))
            hoist_imports = list(
                typing.cast(
                    "tuple[bool, ...]",
                    node.payload.get("hoist_imports", (False,) * len(codes)),
                )
            )
            active_name = typing.cast("str", node.payload["active_name"])
            input_replacements: dict[str, str] = {}
            input_names: set[str] = set()
            for input_name, input_key in typing.cast(
                "tuple[tuple[str, str], ...]", node.payload["bindings"]
            ):
                input_names.add(input_name)
                input_value_name = names[input_key]
                if input_name == input_value_name:
                    continue
                if (
                    graph.display
                    and not _is_semantic_replay_name(input_name)
                    and not any(_code_has_scoped_definition(code) for code in codes)
                    and not any(
                        _provenance_framework._code_stores_name(code, input_name)
                        for code in codes
                    )
                ):
                    input_replacements[input_name] = input_value_name
                else:
                    append_code(f"{input_name} = {input_value_name}")
            if input_replacements:
                try:
                    codes = [
                        _provenance_framework._replace_code_identifiers(
                            code, input_replacements
                        )
                        for code in codes
                    ]
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
            if (
                graph.display
                and active_name != name
                and active_name not in input_names
                and active_name not in input_replacements.values()
            ):
                try:
                    codes = [
                        _provenance_framework._replace_code_identifiers(
                            code, {active_name: name}
                        )
                        for code in codes
                    ]
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
                active_name = name
            for code, group_imports in zip(codes, hoist_imports, strict=True):
                append_code(code, group_imports=group_imports)
            if active_name != name:
                append_code(f"{name} = {active_name}")
            active_setup_key = None
        else:
            raise ReplayGraphError(f"Unknown replay graph node kind {node.kind!r}")

    aliases = graph.aliases
    if output_name is not None:
        if graph.output_key is None:
            raise ReplayGraphError("Replay graph has no output")
        output_alias = (output_name, graph.output_key)
        if output_alias not in aliases:
            aliases = [*aliases, output_alias]
    for public_name, key in aliases:
        planned_name = names[key]
        if public_name != planned_name:
            append_code(f"{public_name} = {planned_name}")
    return _cleanup_emitted_replay_code(_group_framework_imports(chunks))


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
                f"{script_input.name} does not contain recorded source provenance"
            )
        input_key = _compile_spec(
            graph,
            input_spec,
            display=display,
            trusted_user_code=False,
            external_inputs=None,
            live_input_resolver=None,
        )
        graph.add_alias(script_input.name, input_key)
    return emit_replay_code(graph)


def _shares_array_memory(first: xr.DataArray, second: xr.DataArray) -> bool:
    try:
        return bool(np.shares_memory(first.data, second.data))
    except (TypeError, ValueError):
        return False


def execute_replay_graph(
    graph: ReplayGraph,
    *,
    cache: dict[str, xr.DataArray] | None = None,
) -> xr.DataArray:
    # Replay runs from manager actions; avoid optional native reduction accelerators
    # that can crash PySide6/Python 3.14 while Qt threads are alive.
    with xr.set_options(use_numbagg=False):
        return _execute_replay_graph(graph, cache=cache)


def _execute_replay_graph(
    graph: ReplayGraph,
    *,
    cache: dict[str, xr.DataArray] | None = None,
) -> xr.DataArray:
    replay_cache = {} if cache is None else cache
    values: dict[str, xr.DataArray] = {}

    for node in graph.nodes:
        if node.cacheable and node.key in replay_cache:
            values[node.key] = replay_cache[node.key].copy(deep=False)
            continue

        if node.kind == "file_load":
            data = _provenance_framework._load_file_source_data(
                node.payload["load_source"]
            )
        elif node.kind == "setup":
            continue
        elif node.kind == "live_input":
            data = typing.cast("xr.DataArray", node.payload["data"]).copy(deep=False)
        elif node.kind == "relay":
            data = values[node.parents[0]].copy(deep=False)
        elif node.kind == "source_view":
            parent_data = values[node.parents[0]]
            data = _provenance_framework.ToolProvenanceSpec._starting_data_for_kind(
                node.payload["source_kind"],
                parent_data,
            )
        elif node.kind == "operation":
            data = node.payload["operation"].apply(
                values[node.parents[0]],
                parent_data=values[node.parents[1]],
            )
        elif node.kind == "script":
            codes = typing.cast("tuple[str, ...]", node.payload["codes"])
            replay_builtins = (
                vars(builtins)
                if graph.trusted_user_code
                else _provenance_framework._SCRIPT_REPLAY_ALLOWED_BUILTINS
            )
            namespace: dict[str, typing.Any] = {
                "__builtins__": replay_builtins,
                "erlab": erlab,
                "np": np,
                "numpy": np,
                "xr": xr,
                "xarray": xr,
            }
            for alias, target in _REPLAY_ALIASES.items():
                if not any(_code_uses_name(code, alias) for code in codes):
                    continue
                value: typing.Any = erlab
                for attr in target.split(".")[1:]:
                    value = getattr(value, attr)
                namespace[alias] = value
            for input_name, input_key in typing.cast(
                "tuple[tuple[str, str], ...]", node.payload["bindings"]
            ):
                namespace[input_name] = values[input_key].copy(deep=True)
            for code in codes:
                compiled = compile(code, "<ImageTool script provenance>", "exec")
                exec(compiled, namespace, namespace)  # noqa: S102
            active_name = typing.cast("str", node.payload["active_name"])
            if active_name not in namespace:
                raise ReplayGraphError(
                    f"Script provenance did not create active variable {active_name!r}"
                )
            result = namespace[active_name]
            if not isinstance(result, xr.DataArray):
                raise ReplayGraphError(
                    "Script provenance did not produce an xarray.DataArray for "
                    f"{active_name!r}"
                )
            data = result
        else:
            raise ReplayGraphError(f"Unknown replay graph node kind {node.kind!r}")

        if node.cacheable:
            replay_cache[node.key] = data.copy(deep=False)
        values[node.key] = data

    if graph.output_key is None:
        raise ReplayGraphError("Replay graph has no output")
    output = values[graph.output_key]
    if any(
        node.kind == "live_input" and _shares_array_memory(output, values[node.key])
        for node in graph.nodes
    ):
        return output.copy(deep=True)
    return output


def replay_file_provenance(
    spec: typing.Any,
    *,
    cache: dict[str, xr.DataArray] | None = None,
) -> xr.DataArray:
    graph = compile_replay_graph(spec)
    return execute_replay_graph(graph, cache=cache)


def script_provenance_replayable(spec: typing.Any) -> bool:
    parsed = _provenance_framework.parse_tool_provenance_spec(spec)
    if parsed is None:
        return False
    try:
        _validate_script_provenance(parsed)
    except (ReplayGraphError, TypeError, ValueError):
        return False
    return True


def _script_provenance_validates(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
    strict_replay_code: bool,
) -> bool:
    parsed = _provenance_framework.parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        return False
    try:
        _validate_script_provenance(
            parsed,
            external_input_names=external_input_names,
            strict_replay_code=strict_replay_code,
        )
    except (ReplayGraphError, TypeError, ValueError):
        return False
    return True


def script_provenance_requires_trust(
    spec: typing.Any,
    *,
    external_input_names: set[str] | None = None,
) -> bool:
    parsed = _provenance_framework.parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        return False
    strict_replayable = _script_provenance_validates(
        parsed,
        external_input_names=external_input_names,
        strict_replay_code=True,
    )
    trusted_replayable = _script_provenance_validates(
        parsed,
        external_input_names=external_input_names,
        strict_replay_code=False,
    )
    current_requires_trust = not strict_replayable and trusted_replayable
    if current_requires_trust:
        return True
    if not strict_replayable:
        return False
    for script_input in parsed.script_inputs:
        input_spec = script_input.parsed_provenance_spec()
        if script_provenance_requires_trust(input_spec):
            return True
    return False


def _script_trust_payload(spec: typing.Any) -> dict[str, typing.Any] | None:
    parsed = _provenance_framework.parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        return None
    operations = []
    for operation in parsed.operations:
        if getattr(operation, "op", None) != "script_code":
            continue
        operations.append(
            {
                "code": getattr(operation, "code", None),
                "copyable": bool(getattr(operation, "copyable", False)),
            }
        )
    inputs = []
    for script_input in parsed.script_inputs:
        input_payload = _script_trust_payload(script_input.parsed_provenance_spec())
        if input_payload is None:
            continue
        inputs.append({"name": script_input.name, "payload": input_payload})
    return {
        "active_name": parsed.active_name,
        "inputs": inputs,
        "operations": operations,
        "seed_code": parsed.seed_code,
    }


def script_provenance_trust_key(spec: typing.Any) -> str | None:
    payload = _script_trust_payload(spec)
    if payload is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def replay_script_provenance(
    spec: typing.Any,
    inputs: Mapping[str, xr.DataArray],
    *,
    trusted_user_code: bool = False,
) -> xr.DataArray:
    graph = compile_replay_graph(
        spec,
        external_inputs=inputs,
        trusted_user_code=trusted_user_code,
    )
    return execute_replay_graph(graph)


def rebuild_script_provenance(
    spec: typing.Any,
    *,
    live_input_resolver: LiveInputResolver | None = None,
    cache: dict[str, xr.DataArray] | None = None,
    depth: int = 0,
    trusted_user_code: bool = False,
) -> tuple[xr.DataArray, typing.Any]:
    parsed = _provenance_framework.parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        raise ReplayGraphError("Selected provenance is not script-derived")
    if depth > 20:
        raise ReplayGraphError(
            "Nested script provenance exceeded the maximum reload depth"
        )
    if trusted_user_code:
        replayable = _script_provenance_validates(
            parsed,
            strict_replay_code=False,
        )
    else:
        replayable = script_provenance_replayable(parsed)
    if not replayable:
        raise ReplayGraphError(
            "The recorded operation cannot be replayed automatically"
        )

    live_results: dict[tuple[str, str | None], tuple[xr.DataArray, typing.Any]] = {}
    live_misses: set[tuple[str, str | None]] = set()

    def resolve_live(
        script_input: typing.Any,
    ) -> tuple[xr.DataArray, typing.Any] | None:
        if live_input_resolver is None:
            return None
        key = (script_input.name, script_input.node_uid)
        if key in live_results:
            return live_results[key]
        if key in live_misses:
            return None
        resolved = live_input_resolver(script_input)
        if resolved is None:
            live_misses.add(key)
            return None
        live_results[key] = resolved
        return resolved

    def resolve_inputs(current: typing.Any, current_depth: int) -> typing.Any:
        if current_depth > 20:
            raise ReplayGraphError(
                "Nested script provenance exceeded the maximum reload depth"
            )
        resolved_inputs = []
        for script_input in current.script_inputs:
            resolved = resolve_live(script_input)
            if resolved is not None:
                resolved_inputs.append(resolved[1])
                continue

            input_spec = script_input.parsed_provenance_spec()
            if input_spec is None:
                raise ReplayGraphError(
                    f"{script_input.name} is not open and "
                    "does not contain recorded source provenance."
                )
            if input_spec.kind == "file":
                resolved_inputs.append(
                    script_input.model_copy(
                        update={
                            "node_uid": None,
                            "node_snapshot_token": None,
                            "provenance_spec": input_spec.model_dump(mode="json"),
                        }
                    )
                )
                continue
            if input_spec.kind == "script":
                if trusted_user_code:
                    input_replayable = _script_provenance_validates(
                        input_spec,
                        strict_replay_code=False,
                    )
                else:
                    input_replayable = script_provenance_replayable(input_spec)
                if not input_replayable:
                    raise ReplayGraphError(
                        "The recorded operation cannot be replayed automatically"
                    )
                rebuilt_input = resolve_inputs(input_spec, current_depth + 1)
                resolved_inputs.append(
                    script_input.model_copy(
                        update={
                            "node_uid": None,
                            "node_snapshot_token": None,
                            "provenance_spec": rebuilt_input.model_dump(mode="json"),
                        }
                    )
                )
                continue
            raise ReplayGraphError(
                f"{script_input.name} is not open and "
                "does not contain reloadable script or file provenance."
            )
        return current.model_copy(update={"script_inputs": tuple(resolved_inputs)})

    rebuilt_spec = resolve_inputs(parsed, depth)
    graph = compile_replay_graph(
        rebuilt_spec,
        live_input_resolver=resolve_live,
        trusted_user_code=trusted_user_code,
    )
    return execute_replay_graph(graph, cache=cache), rebuilt_spec
