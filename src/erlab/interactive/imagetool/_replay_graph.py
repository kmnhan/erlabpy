"""Internal replay graph for ImageTool provenance.

The saved provenance schema stays in
:mod:`erlab.interactive.imagetool.provenance_framework`.
This module compiles those specs into an execution/code-generation graph at runtime so
shared file loads and shared structured operations are emitted or replayed once.
"""

from __future__ import annotations

import ast
import json
import keyword
import symtable
import typing
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import xarray as xr

import erlab
import erlab.interactive.imagetool.provenance_framework as prov


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
    )

    def __init__(
        self, *, reserved_names: set[str] | None = None, display: bool = False
    ) -> None:
        self.nodes: list[ReplayNode] = []
        self._cacheable_keys: dict[str, str] = {}
        self._reserved_names = set(reserved_names or ())
        self.aliases: list[tuple[str, str]] = []
        self.display = bool(display)
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
    def __init__(self) -> None:
        self.loads: set[str] = set()
        self.stores: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

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

    for stmt in module.body:
        names = _statement_scope_names(stmt)
        for name in sorted(names.loads):
            missing = require(name)
            if missing is not None:
                raise ReplayGraphError(
                    f"Script provenance references unresolved name {missing!r}"
                )
        if isinstance(stmt, ast.FunctionDef):
            function_dependencies[stmt.name] = new_function_dependencies.get(
                (stmt.name, stmt.lineno),
                set(),
            )
        available_names.update(names.stores)


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
) -> None:
    if spec.kind != "script":
        raise ReplayGraphError("Expected script provenance")
    if spec.active_name is None:
        raise ReplayGraphError(
            "Script provenance cannot be replayed without active_name"
        )

    available_names = {
        "erlab",
        "np",
        "numpy",
        "xr",
        "xarray",
        *_REPLAY_ALIASES,
        *prov._SCRIPT_REPLAY_ALLOWED_BUILTINS,
    }
    if spec.script_inputs:
        available_names.update(script_input.name for script_input in spec.script_inputs)
    elif external_input_names is not None:
        available_names.update(external_input_names)
    function_dependencies: dict[str, set[str]] = {}
    has_replay_step = False
    active_available = spec.active_name in available_names
    if spec.seed_code:
        has_replay_step = True
        try:
            prov._validate_script_replay_code(spec.seed_code)
        except (TypeError, ValueError) as exc:
            raise ReplayGraphError(str(exc)) from exc
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
        active_available = active_available or prov._code_stores_name(
            spec.seed_code, spec.active_name
        )
    for operation in spec.operations:
        if getattr(operation, "op", None) == "script_code":
            if not operation.copyable or operation.code is None:
                raise ReplayGraphError("Script provenance contains non-replayable code")
            has_replay_step = True
            try:
                prov._validate_script_replay_code(operation.code)
            except (TypeError, ValueError) as exc:
                raise ReplayGraphError(str(exc)) from exc
            _validate_script_code_names(
                operation.code,
                available_names,
                function_dependencies,
            )
            active_available = active_available or prov._code_stores_name(
                operation.code, spec.active_name
            )
            continue
        if not active_available:
            raise ReplayGraphError("Script provenance has no replay code")
        if not operation.live_applicable:
            raise ReplayGraphError(
                "Script provenance contains non-replayable operation"
            )
        has_replay_step = True
        available_names.add(spec.active_name)
        active_available = True
    if not has_replay_step:
        raise ReplayGraphError("Script provenance has no replay code")
    if not active_available:
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
            if prov._statement_store_count(stmt, active_name) > 0
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
    if not prov._code_stores_name(code, active_name):
        raise ReplayGraphError("Operation replay code does not assign its output")
    return code


def _compile_spec(
    graph: ReplayGraph,
    spec: typing.Any,
    *,
    display: bool,
    external_inputs: Mapping[str, xr.DataArray] | None,
    live_input_resolver: LiveInputResolver | None,
) -> str:
    parsed = prov.parse_tool_provenance_spec(spec)
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
        setup_key = None
        if setup_code:
            setup_key = graph.add_node(
                _canonical_key("setup", {"code": setup_code}),
                "setup",
                payload={"code": setup_code},
            )
        current_key = graph.add_node(
            _canonical_key(
                "file_load",
                parsed.file_load_source.model_dump(mode="json"),
            ),
            "file_load",
            parents=() if setup_key is None else (setup_key,),
            payload={
                "active_name": parsed.active_name,
                "load_source": parsed.file_load_source,
                "load_code": load_code,
            },
        )
        for stage in parsed.replay_stages:
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
                operations = prov.ToolProvenanceSpec._streamlined_operations(
                    stage.source_kind,
                    operations,
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

    if parsed.kind == "script":
        _validate_script_provenance(
            parsed,
            external_input_names=set(external_inputs or ()),
            allow_free_seed_names=display
            and not parsed.script_inputs
            and not external_inputs,
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
                            f"{script_input.name} from {script_input.label} "
                            "does not contain recorded source provenance"
                        )
                    input_key = _compile_spec(
                        graph,
                        input_spec,
                        display=display,
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
        current_bindings = tuple(bindings)
        pending_codes: list[str] = []

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

        def apply_simple_alias(code: str) -> bool:
            nonlocal current_bindings, script_current_key
            if pending_codes:
                return False
            source_name = _simple_assignment_source_name(code, active_name)
            if source_name is None:
                return False
            source_key = binding_key(source_name)
            if source_key is None:
                return False
            script_current_key = relay_key(source_key)
            current_bindings = bind_name(active_name, script_current_key)
            return True

        def flush_script() -> None:
            nonlocal current_bindings, pending_codes, script_current_key
            if not pending_codes:
                return
            script_current_key = graph.add_node(
                _canonical_key(
                    "script",
                    {
                        "active_name": active_name,
                        "bindings": current_bindings,
                        "codes": tuple(pending_codes),
                    },
                ),
                "script",
                parents=tuple(key for _name, key in current_bindings),
                cacheable=False,
                payload={
                    "active_name": active_name,
                    "bindings": current_bindings,
                    "codes": tuple(pending_codes),
                },
            )
            current_bindings = bind_name(active_name, script_current_key)
            pending_codes = []

        if parsed.seed_code and not apply_simple_alias(parsed.seed_code):
            pending_codes.append(parsed.seed_code)
        operations = tuple(parsed.operations)
        for index, operation in enumerate(operations):
            if display:
                if getattr(operation, "op", None) == "rename" and not any(
                    getattr(later_operation, "op", None) == "script_code"
                    for later_operation in operations[index + 1 :]
                ):
                    continue
                entry = operation.derivation_entry()
                if entry.code in {
                    "derived = derived.isel()",
                    "derived = derived.qsel()",
                    "derived = derived.sel()",
                } or prov._is_internal_sort_coord_order_entry(entry):
                    continue
                if prov._is_whole_array_rename_entry(entry) and not any(
                    getattr(later_operation, "op", None) == "script_code"
                    for later_operation in operations[index + 1 :]
                ):
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
                continue

            if pending_codes:
                pending_codes.append(
                    _operation_replay_code(
                        operation,
                        active_name=active_name,
                        context_name=active_name,
                    )
                )
                continue

            flush_script()
            if script_current_key is None:
                matching_inputs = [key for name, key in bindings if name == active_name]
                if len(matching_inputs) != 1:
                    raise ReplayGraphError("Script provenance has no replay code")
                script_current_key = relay_key(matching_inputs[0])
            script_current_key = graph.add_node(
                _canonical_key(
                    "operation",
                    {
                        "context": script_current_key,
                        "operation": operation.model_dump(mode="json"),
                        "parent": script_current_key,
                    },
                ),
                "operation",
                parents=(script_current_key, script_current_key),
                payload={"operation": operation},
            )
            current_bindings = bind_name(active_name, script_current_key)

        flush_script()
        if script_current_key is None:
            matching_inputs = [key for name, key in bindings if name == active_name]
            if len(matching_inputs) != 1:
                raise ReplayGraphError("Script provenance has no replay code")
            script_current_key = relay_key(matching_inputs[0])
        if display:
            graph.add_alias(active_name, script_current_key)
        return script_current_key

    raise ReplayGraphError(f"{parsed.kind!r} provenance is not self-contained")


def compile_replay_graph(
    spec: typing.Any,
    *,
    display: bool = False,
    external_inputs: Mapping[str, xr.DataArray] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> ReplayGraph:
    parsed = prov.parse_tool_provenance_spec(spec)
    if parsed is None:
        raise ReplayGraphError("Expected provenance spec")
    reserved_names = _reserved_names_from_spec(parsed)
    if external_inputs:
        reserved_names.update(external_inputs)
    graph = ReplayGraph(reserved_names=reserved_names, display=display)
    graph.output_key = _compile_spec(
        graph,
        parsed,
        display=display,
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
            if not isinstance(next_stmt, (ast.Assign, ast.Expr)):
                continue
            if prov._statement_load_count(next_stmt, target.id) != 1:
                continue
            if any(
                prov._statement_load_count(later_stmt, target.id) > 0
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
                if prov._statement_load_count(later, target.id) > 0
            ]
            if len(later_loads) != 1:
                continue
            use_idx = later_loads[0]
            use_stmt = module.body[use_idx]
            if prov._statement_load_count(use_stmt, target.id) != 1:
                continue

            replacement_load_names = {
                node.id
                for node in ast.walk(stmt.value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            intervening = module.body[idx + 1 : use_idx]
            if any(
                prov._statement_store_count(item, target.id) for item in intervening
            ):
                continue
            if any(
                prov._statement_store_count(item, name)
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
    code = _inline_adjacent_replay_assignments(code)
    code = _inline_single_use_replay_expressions(code)
    return _compact_replay_temp_names(code)


def emit_replay_code(graph: ReplayGraph, *, output_name: str | None = None) -> str:
    names = _node_names(graph, output_name=output_name)
    node_by_key = {node.key: node for node in graph.nodes}
    lines: list[str] = []
    active_setup_key: str | None = None

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
                lines.append(typing.cast("str", setup_node.payload["code"]))
                active_setup_key = setup_key
            try:
                code = prov._replace_code_identifiers(load_code, {active_name: name})
            except SyntaxError as exc:
                raise ReplayGraphError("File replay code is not valid Python") from exc
            if not prov._code_stores_name(code, name):
                raise ReplayGraphError("File replay code does not assign its output")
            lines.append(code)
        elif node.kind == "live_input":
            raise ReplayGraphError("Live inputs cannot be emitted as replay code")
        elif node.kind == "source_view":
            parent_name = names[node.parents[0]]
            if not _source_view_emits_code(graph, node):
                continue
            lines.append(
                f"{name} = erlab.interactive.imagetool.slicer."
                f"restore_nonuniform_dims({parent_name})"
            )
        elif node.kind == "operation":
            parent_name = names[node.parents[0]]
            context_name = names[node.parents[1]]
            operation = node.payload["operation"]
            lines.append(
                _operation_replay_code(
                    operation,
                    active_name=name,
                    context_name=context_name,
                    parent_name=parent_name,
                )
            )
        elif node.kind == "script":
            codes = list(typing.cast("tuple[str, ...]", node.payload["codes"]))
            active_name = typing.cast("str", node.payload["active_name"])
            input_replacements: dict[str, str] = {}
            for input_name, input_key in typing.cast(
                "tuple[tuple[str, str], ...]", node.payload["bindings"]
            ):
                input_value_name = names[input_key]
                if input_name == input_value_name:
                    continue
                if (
                    graph.display
                    and not _is_semantic_replay_name(input_name)
                    and not any(_code_has_scoped_definition(code) for code in codes)
                    and not any(
                        prov._code_stores_name(code, input_name) for code in codes
                    )
                ):
                    input_replacements[input_name] = input_value_name
                else:
                    lines.append(f"{input_name} = {input_value_name}")
            if input_replacements:
                try:
                    codes = [
                        prov._replace_code_identifiers(code, input_replacements)
                        for code in codes
                    ]
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
            if graph.display and active_name != name:
                try:
                    codes = [
                        prov._replace_code_identifiers(code, {active_name: name})
                        for code in codes
                    ]
                except SyntaxError as exc:
                    raise ReplayGraphError(
                        "Script replay code is not valid Python"
                    ) from exc
                active_name = name
            lines.extend(codes)
            if active_name != name:
                lines.append(f"{name} = {active_name}")
            active_setup_key = None
        else:
            raise ReplayGraphError(f"Unknown replay graph node kind {node.kind!r}")

    aliases = graph.aliases
    if output_name is not None:
        if graph.output_key is None:
            raise ReplayGraphError("Replay graph has no output")
        aliases = [*aliases, (output_name, graph.output_key)]
    for public_name, key in aliases:
        planned_name = names[key]
        if public_name != planned_name:
            lines.append(f"{public_name} = {planned_name}")
    return _cleanup_emitted_replay_code("\n".join(lines))


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
                f"{script_input.name} from {script_input.label} "
                "does not contain recorded source provenance"
            )
        input_key = _compile_spec(
            graph,
            input_spec,
            display=display,
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
    replay_cache = {} if cache is None else cache
    values: dict[str, xr.DataArray] = {}

    for node in graph.nodes:
        if node.cacheable and node.key in replay_cache:
            values[node.key] = replay_cache[node.key].copy(deep=False)
            continue

        if node.kind == "file_load":
            data = prov._load_file_source_data(node.payload["load_source"])
        elif node.kind == "setup":
            continue
        elif node.kind == "live_input":
            data = typing.cast("xr.DataArray", node.payload["data"]).copy(deep=False)
        elif node.kind == "relay":
            data = values[node.parents[0]].copy(deep=False)
        elif node.kind == "source_view":
            parent_data = values[node.parents[0]]
            data = prov.ToolProvenanceSpec._starting_data_for_kind(
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
            namespace: dict[str, typing.Any] = {
                "__builtins__": prov._SCRIPT_REPLAY_ALLOWED_BUILTINS,
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
    parsed = prov.parse_tool_provenance_spec(spec)
    if parsed is None:
        return False
    try:
        _validate_script_provenance(parsed)
    except (ReplayGraphError, TypeError, ValueError):
        return False
    return True


def replay_script_provenance(
    spec: typing.Any,
    inputs: Mapping[str, xr.DataArray],
) -> xr.DataArray:
    graph = compile_replay_graph(spec, external_inputs=inputs)
    return execute_replay_graph(graph)


def rebuild_script_provenance(
    spec: typing.Any,
    *,
    live_input_resolver: LiveInputResolver | None = None,
    cache: dict[str, xr.DataArray] | None = None,
    depth: int = 0,
) -> tuple[xr.DataArray, typing.Any]:
    parsed = prov.parse_tool_provenance_spec(spec)
    if parsed is None or parsed.kind != "script":
        raise ReplayGraphError("Selected provenance is not script-derived")
    if depth > 20:
        raise ReplayGraphError(
            "Nested script provenance exceeded the maximum reload depth"
        )
    if not script_provenance_replayable(parsed):
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
                    f"{script_input.name} from {script_input.label} is not open and "
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
                if not script_provenance_replayable(input_spec):
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
                f"{script_input.name} from {script_input.label} is not open and "
                "does not contain reloadable script or file provenance."
            )
        return current.model_copy(update={"script_inputs": tuple(resolved_inputs)})

    rebuilt_spec = resolve_inputs(parsed, depth)
    graph = compile_replay_graph(rebuilt_spec, live_input_resolver=resolve_live)
    return execute_replay_graph(graph, cache=cache), rebuilt_spec
