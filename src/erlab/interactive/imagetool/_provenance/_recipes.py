"""Construct explicit source-to-result graphs without Manager or Python bindings."""

from __future__ import annotations

import hashlib
import json
import typing
import uuid

from erlab.interactive.imagetool._provenance._model import (
    CodeNode,
    FileLoadSource,
    FileSourceNode,
    InputNode,
    MemorySourceNode,
    OperationNode,
    ProvenanceGraphNode,
    ReplayStep,
    ScriptInput,
    SnapshotSourceNode,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    ValueRef,
    _normalize_legacy_script_input,
    _SourceViewOperation,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _identity(node: ProvenanceGraphNode) -> str:
    payload = node.model_dump(mode="json", exclude={"id", "label"})
    if isinstance(node, FileSourceNode):
        payload["source"] = {
            "path": node.source.path,
            "replay_call": node.source.replay_call.model_dump(mode="json")
            if node.source.replay_call is not None
            else None,
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _identified(node: ProvenanceGraphNode) -> ProvenanceGraphNode:
    encoded = _identity(node)
    return node.model_copy(
        update={"id": "value-" + hashlib.sha256(encoded.encode()).hexdigest()}
    )


def _graph(
    nodes: tuple[ProvenanceGraphNode, ...],
    output: ValueRef,
    *,
    input_ref: ValueRef | None = None,
) -> ToolProvenanceSpec:
    return ToolProvenanceSpec(
        schema_version=4, kind="graph", nodes=nodes, output=output, input_ref=input_ref
    ).model_copy(deep=True)


def _require_graph(spec: ToolProvenanceSpec) -> None:
    if spec.kind != "graph":
        raise TypeError("Import a legacy recipe before combining graph values")
    spec.topological_nodes()


def _merge_nodes(*graphs: ToolProvenanceSpec) -> dict[str, ProvenanceGraphNode]:
    nodes: dict[str, ProvenanceGraphNode] = {}
    for graph in graphs:
        _require_graph(graph)
        for node in graph.nodes:
            previous = nodes.get(node.id)
            if previous is not None and _identity(previous) != _identity(node):
                raise ValueError("A node ID cannot identify different graph values")
            nodes.setdefault(node.id, node)
    return nodes


def file_recipe(
    source: FileLoadSource, *, label: str | None = None
) -> ToolProvenanceSpec:
    """Start a graph from a typed file load, independent of copied load code."""
    node = _identified(FileSourceNode(id="source", source=source, label=label))
    return _graph((node,), ValueRef(node_id=node.id))


def memory_recipe(source_id: str, *, label: str | None = None) -> ToolProvenanceSpec:
    """Start a graph from a durable memory source identity, without a window ID."""
    node = _identified(MemorySourceNode(id="source", source_id=source_id, label=label))
    return _graph((node,), ValueRef(node_id=node.id))


def snapshot_recipe(
    revision_id: str, *, label: str | None = None
) -> ToolProvenanceSpec:
    """Start a graph at one fixed revision, without a producer fallback."""
    node = _identified(
        SnapshotSourceNode(id="source", revision_id=revision_id, label=label)
    )
    return _graph((node,), ValueRef(node_id=node.id))


def input_recipe(*, label: str | None = None) -> ToolProvenanceSpec:
    """Start a fragment at an explicit input value that grafting must replace."""
    node = InputNode(id="input-" + uuid.uuid4().hex, label=label)
    ref = ValueRef(node_id=node.id)
    return _graph((node,), ref, input_ref=ref)


def append_recipe(
    source: ToolProvenanceSpec,
    operation: ToolProvenanceOperation,
    *,
    inputs: Mapping[str, ToolProvenanceSpec] | None = None,
    label: str | None = None,
) -> ToolProvenanceSpec:
    """Connect an operation directly to its primary value and semantic input slots."""
    _require_graph(source)
    inputs = dict(inputs or {})
    if any(value.input_ref is not None for value in inputs.values()):
        raise ValueError("Auxiliary graph inputs must be complete recipes")
    nodes = _merge_nodes(source, *inputs.values())
    node = _identified(
        OperationNode(
            id="operation",
            operation=operation,
            data=typing.cast("ValueRef", source.output),
            inputs={
                slot: typing.cast("ValueRef", value.output)
                for slot, value in inputs.items()
            },
            label=label,
        )
    )
    nodes[node.id] = node
    return _graph(
        tuple(nodes.values()), ValueRef(node_id=node.id), input_ref=source.input_ref
    )


def code_recipe(
    code: str,
    *,
    inputs: Mapping[str, ToolProvenanceSpec] | None = None,
    output_name: str,
    uses_implicit_framework_imports: bool = False,
    label: str | None = None,
) -> ToolProvenanceSpec:
    """Run a Python block in its own scope with explicit graph inputs and output."""
    inputs = dict(inputs or {})
    if any(value.input_ref is not None for value in inputs.values()):
        raise ValueError("Python graph inputs must be complete recipes")
    nodes = _merge_nodes(*inputs.values())
    node = CodeNode(
        id="code-" + uuid.uuid4().hex,
        code=code,
        inputs={
            name: typing.cast("ValueRef", value.output)
            for name, value in inputs.items()
        },
        output_name=output_name,
        uses_implicit_framework_imports=uses_implicit_framework_imports,
        label=label,
    )
    nodes[node.id] = node
    return _graph(tuple(nodes.values()), ValueRef(node_id=node.id))


def graft_recipe(
    fragment: ToolProvenanceSpec, source: ToolProvenanceSpec
) -> ToolProvenanceSpec:
    """Replace a fragment's input reference while preserving its other graph edges."""
    _require_graph(fragment)
    _require_graph(source)
    if fragment.input_ref is None:
        raise ValueError("Only a fragment with an explicit input can be attached")
    nodes = _merge_nodes(source)
    refs = {fragment.input_ref.node_id: typing.cast("ValueRef", source.output)}
    for node in fragment.topological_nodes():
        if isinstance(node, InputNode):
            continue
        if isinstance(node, OperationNode):
            attached = _identified(
                node.model_copy(
                    update={
                        "data": refs[node.data.node_id],
                        "inputs": {
                            slot: refs[ref.node_id] for slot, ref in node.inputs.items()
                        },
                    }
                )
            )
        elif isinstance(node, CodeNode):
            inputs = {name: refs[ref.node_id] for name, ref in node.inputs.items()}
            attached = node.model_copy(update={"inputs": inputs})
            if inputs != node.inputs:
                payload = {
                    "origin": node.id,
                    "inputs": {name: ref.node_id for name, ref in inputs.items()},
                }
                encoded = json.dumps(payload, sort_keys=True).encode()
                attached = attached.model_copy(
                    update={"id": "code-" + hashlib.sha256(encoded).hexdigest()}
                )
        else:
            attached = node
        previous = nodes.get(attached.id)
        if previous is not None and _identity(previous) != _identity(attached):
            raise ValueError("A node ID cannot identify different graph values")
        nodes.setdefault(attached.id, attached)
        refs[node.id] = ValueRef(node_id=attached.id)
    return _graph(
        tuple(nodes.values()),
        refs[typing.cast("ValueRef", fragment.output).node_id],
        input_ref=source.input_ref,
    )


def _append_legacy_steps(
    current: ToolProvenanceSpec,
    steps: tuple[ReplayStep, ...],
    inputs: Mapping[str, ToolProvenanceSpec],
) -> ToolProvenanceSpec:
    previous_policy: str | None = None
    for step in steps:
        if step.legacy_context is not None or step.context_names:
            raise ValueError("Legacy replay contexts require explicit graph conversion")
        if step.input_policy is not None and step.input_policy != previous_policy:
            current = append_recipe(
                current,
                _SourceViewOperation(
                    source_kind="full_data"
                    if step.input_policy == "current"
                    else "public_data"
                ),
            )
        current = append_recipe(
            current,
            step.operation,
            inputs={slot: inputs[name] for slot, name in step.input_bindings.items()},
        )
        previous_policy = step.input_policy
    return current


def from_legacy_recipe(
    spec: ToolProvenanceSpec,
    *,
    parent: ToolProvenanceSpec | None = None,
    input_resolver: Callable[[ScriptInput], ToolProvenanceSpec | None] | None = None,
) -> ToolProvenanceSpec:
    """Explicitly import a legacy recipe without changing its saved representation.

    A caller must supply owned memory-source recipes for inputs without a durable
    producer. Existing fixed captures stay fixed. Arbitrary mixed Python and typed
    replay contexts need explicit conversion and are rejected rather than guessed.
    """
    if spec.kind == "graph":
        return spec.model_copy(deep=True)

    def resolve(item: ScriptInput) -> ToolProvenanceSpec:
        item = _normalize_legacy_script_input(item)
        resolved = False
        if not item.follow_source:
            revision_id = item.resolved_revision_id or item.revision_id
            if revision_id is None:
                raise ValueError("A fixed legacy input has no recorded revision")
            value = snapshot_recipe(revision_id, label=item.label)
            resolved = item.resolved_revision_id is not None
        else:
            value = None if input_resolver is None else input_resolver(item)
            if value is None:
                producer = item.parsed_provenance_spec()
                if producer is None:
                    raise ValueError("A legacy input needs an explicit source recipe")
                value = from_legacy_recipe(producer, input_resolver=input_resolver)
                resolved = item.fallback_role == "resolved"
            _require_graph(value)
        source = item.parsed_source_spec()
        if source is not None and not resolved:
            value = from_legacy_recipe(
                source, parent=value, input_resolver=input_resolver
            )
        return value

    if spec.is_live_source:
        if parent is None:
            raise ValueError("A relative source requires an explicit parent recipe")
        return _append_legacy_steps(parent, ReplayStep.from_source_spec(spec), {})
    if spec.kind == "file":
        if spec.file_load_source is None:
            raise ValueError("A file recipe must record its source")
        return _append_legacy_steps(file_recipe(spec.file_load_source), spec.steps, {})
    if spec.kind != "script":
        raise ValueError("Unsupported legacy provenance source")
    inputs = {item.name: resolve(item) for item in spec.replay_inputs}
    if spec.seed_code is not None or any(
        getattr(step.operation, "op", None) == "script_code" for step in spec.steps
    ):
        raise ValueError("Python recipes require explicit CodeNode inputs and output")
    primary = spec.replay_primary_input
    if primary is None:
        raise ValueError("A legacy recipe must identify its primary input")
    return _append_legacy_steps(inputs[primary], spec.steps, inputs)
