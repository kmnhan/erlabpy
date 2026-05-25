"""Internal replay graph for ImageTool provenance.

The saved provenance schema stays in :mod:`erlab.interactive.imagetool.provenance`.
This module compiles those specs into an execution/code-generation graph at runtime so
shared file loads and shared structured operations are emitted or replayed once.
"""

from __future__ import annotations

import ast
import json
import typing
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import xarray as xr

import erlab


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
    __slots__ = ("_cacheable_keys", "_reserved_names", "aliases", "nodes", "output_key")

    def __init__(self, *, reserved_names: set[str] | None = None) -> None:
        self.nodes: list[ReplayNode] = []
        self._cacheable_keys: dict[str, str] = {}
        self._reserved_names = set(reserved_names or ())
        self.aliases: list[tuple[str, str]] = []
        self.output_key: str | None = None

    @property
    def reserved_names(self) -> set[str]:
        return set(self._reserved_names)

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


def _prov():
    from erlab.interactive.imagetool import provenance

    return provenance


def _canonical_key(kind: str, payload: Mapping[str, typing.Any]) -> str:
    return json.dumps(
        {"kind": kind, **dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
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


def _script_replay_codes(spec: typing.Any) -> tuple[str, ...]:
    prov = _prov()
    if spec.kind != "script":
        raise ReplayGraphError("Expected script provenance")
    if spec.active_name is None:
        raise ReplayGraphError(
            "Script provenance cannot be replayed without active_name"
        )

    codes: list[str] = []
    if spec.seed_code:
        codes.append(spec.seed_code)
    for operation in spec.operations:
        if not isinstance(operation, prov.ScriptCodeOperation):
            raise ReplayGraphError(
                "Only script_code operations can be replayed as script provenance"
            )
        if not operation.copyable or operation.code is None:
            raise ReplayGraphError("Script provenance contains non-replayable code")
        codes.append(operation.code)
    if not codes:
        raise ReplayGraphError("Script provenance has no replay code")
    for code in codes:
        try:
            prov._validate_script_replay_code(code)
        except (TypeError, ValueError) as exc:
            raise ReplayGraphError(str(exc)) from exc
    return tuple(codes)


def _file_seed_code_parts(seed_code: str, active_name: str) -> tuple[str | None, str]:
    prov = _prov()

    def module_code(body: Sequence[ast.stmt]) -> str | None:
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

    setup_code = module_code(module.body[:output_stmt_idx])
    load_code = module_code(module.body[output_stmt_idx:])
    if load_code is None:
        raise ReplayGraphError("File replay code does not assign its output")
    return setup_code, load_code


def _compile_spec(
    graph: ReplayGraph,
    spec: typing.Any,
    *,
    display: bool,
    external_inputs: Mapping[str, xr.DataArray] | None,
    live_input_resolver: LiveInputResolver | None,
) -> str:
    prov = _prov()
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
        codes = _script_replay_codes(parsed)
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

        return graph.add_node(
            _canonical_key(
                "script",
                {
                    "active_name": parsed.active_name,
                    "bindings": tuple(bindings),
                    "codes": codes,
                },
            ),
            "script",
            parents=tuple(key for _name, key in bindings),
            cacheable=False,
            payload={
                "active_name": parsed.active_name,
                "bindings": tuple(bindings),
                "codes": codes,
            },
        )

    raise ReplayGraphError(f"{parsed.kind!r} provenance is not self-contained")


def compile_replay_graph(
    spec: typing.Any,
    *,
    display: bool = False,
    external_inputs: Mapping[str, xr.DataArray] | None = None,
    live_input_resolver: LiveInputResolver | None = None,
) -> ReplayGraph:
    parsed = _prov().parse_tool_provenance_spec(spec)
    if parsed is None:
        raise ReplayGraphError("Expected provenance spec")
    reserved_names = _reserved_names_from_spec(parsed)
    if external_inputs:
        reserved_names.update(external_inputs)
    graph = ReplayGraph(reserved_names=reserved_names)
    graph.output_key = _compile_spec(
        graph,
        parsed,
        display=display,
        external_inputs=external_inputs,
        live_input_resolver=live_input_resolver,
    )
    return graph


def _node_names(graph: ReplayGraph) -> dict[str, str]:
    names: dict[str, str] = {}
    used = graph.reserved_names
    counter = 0

    def next_temp() -> str:
        nonlocal counter
        while True:
            name = f"_itool_replay_{counter}"
            counter += 1
            if name not in used:
                used.add(name)
                return name

    for node in graph.nodes:
        names[node.key] = next_temp()
    return names


def emit_replay_code(graph: ReplayGraph, *, output_name: str | None = None) -> str:
    prov = _prov()
    names = _node_names(graph)
    node_by_key = {node.key: node for node in graph.nodes}
    lines: list[str] = []
    active_setup_key: str | None = None

    for node in graph.nodes:
        name = names[node.key]
        if node.kind == "setup":
            continue
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
            if node.payload["source_kind"] == "full_data":
                lines.append(f"{name} = {parent_name}")
            else:
                lines.append(
                    f"{name} = erlab.interactive.imagetool.slicer."
                    f"restore_nonuniform_dims({parent_name})"
                )
        elif node.kind == "operation":
            parent_name = names[node.parents[0]]
            context_name = names[node.parents[1]]
            operation = node.payload["operation"]
            entry = operation.derivation_entry()
            if entry is None:
                raise ReplayGraphError("Operation does not provide replay code")
            lines.append(f"{name} = {parent_name}")
            if entry.code is None:
                raise ReplayGraphError("Operation does not provide replay code")
            try:
                lines.append(
                    prov._replace_code_identifiers(
                        entry.code,
                        {"data": context_name, "derived": name},
                    )
                )
            except SyntaxError as exc:
                raise ReplayGraphError(
                    "Operation replay code is not valid Python"
                ) from exc
        elif node.kind == "script":
            for input_name, input_key in typing.cast(
                "tuple[tuple[str, str], ...]", node.payload["bindings"]
            ):
                input_value_name = names[input_key]
                if input_name != input_value_name:
                    lines.append(f"{input_name} = {input_value_name}")
            lines.extend(typing.cast("tuple[str, ...]", node.payload["codes"]))
            active_name = typing.cast("str", node.payload["active_name"])
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
    return "\n".join(lines)


def script_inputs_code(script_inputs: Sequence[typing.Any], *, display: bool) -> str:
    reserved_names: set[str] = set()
    for script_input in script_inputs:
        reserved_names.add(script_input.name)
        reserved_names.update(
            _reserved_names_from_spec(script_input.parsed_provenance_spec())
        )
    graph = ReplayGraph(reserved_names=reserved_names)
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
        graph.aliases.append((script_input.name, input_key))
    return emit_replay_code(graph)


def execute_replay_graph(
    graph: ReplayGraph,
    *,
    cache: dict[str, xr.DataArray] | None = None,
) -> xr.DataArray:
    prov = _prov()
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
            namespace: dict[str, typing.Any] = {
                "__builtins__": prov._SCRIPT_REPLAY_ALLOWED_BUILTINS,
                "erlab": erlab,
                "np": np,
                "numpy": np,
                "xr": xr,
                "xarray": xr,
            }
            for input_name, input_key in typing.cast(
                "tuple[tuple[str, str], ...]", node.payload["bindings"]
            ):
                namespace[input_name] = values[input_key].copy(deep=True)
            for code in typing.cast("tuple[str, ...]", node.payload["codes"]):
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
    return values[graph.output_key]


def replay_file_provenance(
    spec: typing.Any,
    *,
    cache: dict[str, xr.DataArray] | None = None,
) -> xr.DataArray:
    graph = compile_replay_graph(spec)
    return execute_replay_graph(graph, cache=cache)


def script_provenance_replayable(spec: typing.Any) -> bool:
    parsed = _prov().parse_tool_provenance_spec(spec)
    if parsed is None:
        return False
    try:
        _script_replay_codes(parsed)
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
    prov = _prov()
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

    shared_cache = {} if cache is None else cache
    resolved_inputs: list[tuple[xr.DataArray, typing.Any]] = []
    for script_input in parsed.script_inputs:
        resolved = live_input_resolver(script_input) if live_input_resolver else None
        if resolved is not None:
            resolved_inputs.append(resolved)
            continue

        input_spec = script_input.parsed_provenance_spec()
        if input_spec is None:
            raise ReplayGraphError(
                f"{script_input.name} from {script_input.label} is not open and "
                "does not contain recorded source provenance."
            )

        if input_spec.kind == "file":
            data = replay_file_provenance(input_spec, cache=shared_cache)
            resolved_inputs.append(
                (
                    data,
                    script_input.model_copy(
                        update={
                            "node_uid": None,
                            "node_snapshot_token": None,
                            "provenance_spec": input_spec.model_dump(mode="json"),
                        }
                    ),
                )
            )
            continue

        if input_spec.kind == "script":
            data, rebuilt_spec = rebuild_script_provenance(
                input_spec,
                live_input_resolver=live_input_resolver,
                cache=shared_cache,
                depth=depth + 1,
            )
            resolved_inputs.append(
                (
                    data,
                    script_input.model_copy(
                        update={
                            "node_uid": None,
                            "node_snapshot_token": None,
                            "provenance_spec": rebuilt_spec.model_dump(mode="json"),
                        }
                    ),
                )
            )
            continue

        raise ReplayGraphError(
            f"{script_input.name} from {script_input.label} is not open and "
            "does not contain reloadable script or file provenance."
        )

    rebuilt_spec = parsed.model_copy(
        update={
            "script_inputs": tuple(
                script_input for _data, script_input in resolved_inputs
            )
        }
    )
    input_data = {script_input.name: data for data, script_input in resolved_inputs}
    return replay_script_provenance(rebuilt_spec, input_data), rebuilt_spec
