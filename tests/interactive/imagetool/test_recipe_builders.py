import json
import pathlib
from collections.abc import Mapping

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive._code_trust import issue_execution_capability, new_document_trust
from erlab.interactive.imagetool._load_source import _register_local_callable_loader
from erlab.interactive.imagetool._provenance._execution import execute_replay_graph
from erlab.interactive.imagetool._provenance._graph import (
    ReplayGraphError,
    compile_replay_graph,
    emit_replay_code,
)
from erlab.interactive.imagetool._provenance._model import (
    CodeNode,
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    FileSourceNode,
    InputNode,
    MemorySourceNode,
    OperationNode,
    ReplayStep,
    ScriptInput,
    SnapshotSourceNode,
    ToolProvenanceSpec,
    ValueRef,
    file_load,
    full_data,
    parse_tool_provenance_spec,
    public_data,
    recipe,
    script,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    AverageOperation,
    ConcatOperation,
    ScriptCodeOperation,
    ShiftFromInputOperation,
    UniformInterpolationOperation,
)
from erlab.interactive.imagetool._provenance._recipes import (
    append_recipe,
    code_recipe,
    file_recipe,
    from_legacy_recipe,
    graft_recipe,
    input_recipe,
    memory_recipe,
    snapshot_recipe,
)
from erlab.interactive.imagetool._provenance._trust import provenance_code_trust_entries


def _scan(frames: int = 2) -> xr.DataArray:
    return xr.DataArray(
        np.arange(frames * 15.0).reshape(frames, 3, 5),
        dims=("time", "alpha", "eV"),
        coords={
            "time": np.arange(frames),
            "alpha": [-1.0, 0.0, 1.0],
            "eV": np.linspace(-0.2, 0.2, 5),
        },
        name="scan",
    )


def _file_source(path: pathlib.Path) -> FileLoadSource:
    _register_local_callable_loader(xr.load_dataarray)
    return FileLoadSource(
        path=str(path),
        loader_label="xarray.load_dataarray",
        loader_text="xarray.load_dataarray",
        kwargs_text="engine='h5netcdf'",
        replay_call=FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selection=FileDataSelection(kind="dataarray"),
        ),
    )


def _legacy_file(path: pathlib.Path) -> ToolProvenanceSpec:
    return file_load(
        start_label="Load scan",
        seed_code=(
            "import xarray as xr\n\n"
            f"derived = xr.load_dataarray({str(path)!r}, engine='h5netcdf')"
        ),
        file_load_source=_file_source(path),
    )


def _resolve(values: Mapping[str, xr.DataArray]):
    def resolve(node: MemorySourceNode | SnapshotSourceNode) -> xr.DataArray:
        identity = (
            node.source_id if isinstance(node, MemorySourceNode) else node.revision_id
        )
        return values[identity]

    return resolve


def _execute(
    spec: ToolProvenanceSpec, values: Mapping[str, xr.DataArray] | None = None
):
    graph = compile_replay_graph(
        spec,
        structured_file_replay=True,
        trusted_user_code=True,
        source_resolver=None if values is None else _resolve(values),
    )
    return execute_replay_graph(
        graph,
        authorize=lambda entries: issue_execution_capability(
            new_document_trust(), entries
        )[1],
    )


def _copied(
    spec: ToolProvenanceSpec,
    values: Mapping[str, xr.DataArray] | None = None,
    *,
    input_names: Mapping[str, str] | None = None,
):
    graph = compile_replay_graph(spec, display=True)
    namespace = {}
    overrides = {}
    for node in graph.nodes:
        if node.kind != "external_input":
            continue
        identity = node.payload.get("source_id", node.payload.get("revision_id"))
        if identity is None:
            continue
        name = (input_names or {}).get(identity, f"source_{len(overrides)}")
        overrides[node.key] = name
        namespace[name] = (values or {})[identity]
    code = emit_replay_code(
        graph,
        output_name="result",
        input_name_overrides=overrides,
        include_framework_imports=True,
    )
    exec(code, namespace, namespace)  # noqa: S102
    return namespace["result"]


def _semantic_graph(spec: ToolProvenanceSpec) -> str:
    nodes = {node.id: node for node in spec.nodes}

    def value(ref: ValueRef):
        node = nodes[ref.node_id]
        if isinstance(node, FileSourceNode):
            return (
                "file",
                node.source.path,
                node.source.replay_call.model_dump(mode="json"),
            )
        if isinstance(node, MemorySourceNode):
            return ("memory", node.source_id)
        if isinstance(node, SnapshotSourceNode):
            return ("snapshot", node.revision_id)
        if isinstance(node, InputNode):
            return ("input",)
        if isinstance(node, OperationNode):
            return (
                "operation",
                node.operation.model_dump(mode="json"),
                value(node.data),
                {slot: value(parent) for slot, parent in sorted(node.inputs.items())},
            )
        return (
            "code",
            node.code,
            node.output_name,
            {name: value(parent) for name, parent in sorted(node.inputs.items())},
        )

    return json.dumps(value(spec.output), sort_keys=True)


@pytest.mark.parametrize("bound_shift", [False, True])
def test_append_and_graft_have_the_same_typed_derivation(bound_shift: bool) -> None:
    source = memory_recipe("scan", label="data")
    operation = AverageOperation(dims=("time",))
    prepared = append_recipe(source, operation)
    fragment = append_recipe(input_recipe(label="data"), operation)
    values = {"scan": _scan(), "edge": _scan().qsel.mean(("time", "eV")) / 1000.0}
    expected = values["scan"].qsel.mean("time")
    if bound_shift:
        shift = ShiftFromInputOperation(along="eV", negate=True)
        edge = memory_recipe("edge", label="data")
        prepared = append_recipe(prepared, shift, inputs={"shift": edge})
        fragment = append_recipe(fragment, shift, inputs={"shift": edge})
        expected = erlab.analysis.transform.shift(expected, -values["edge"], along="eV")
    grafted = graft_recipe(fragment, source)
    assert _semantic_graph(grafted) == _semantic_graph(prepared)
    assert grafted.input_ref is None
    assert all(
        isinstance(node.data, ValueRef)
        for node in grafted.nodes
        if isinstance(node, OperationNode)
    )
    xr.testing.assert_identical(_execute(grafted, values), expected)
    xr.testing.assert_identical(_copied(grafted, values), expected)


def test_append_and_graft_merge_a_roundtripped_shift_with_default_nan_fill() -> None:
    shifted = append_recipe(
        memory_recipe("scan"),
        ShiftFromInputOperation(along="eV"),
        inputs={"shift": memory_recipe("edge")},
    )
    roundtripped = parse_tool_provenance_spec(json.loads(shifted.model_dump_json()))
    assert roundtripped is not None
    operation = ConcatOperation(input_count=2, dim="repeat")
    appended = append_recipe(shifted, operation, inputs={"other_0": roundtripped})
    fragment = append_recipe(
        input_recipe(), operation, inputs={"other_0": roundtripped}
    )
    grafted = graft_recipe(fragment, shifted)
    assert _semantic_graph(appended) == _semantic_graph(grafted)
    values = {"scan": _scan(), "edge": _scan().qsel.mean(("time", "eV")) / 1000.0}
    expected_shift = erlab.analysis.transform.shift(
        values["scan"], values["edge"], along="eV"
    )
    expected = xr.concat([expected_shift, expected_shift], dim="repeat")
    xr.testing.assert_identical(_execute(appended, values), expected)
    xr.testing.assert_identical(_execute(grafted, values), expected)


def test_file_source_reloads_a_growing_scan(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "scan.nc"
    original, updated = _scan(), _scan(5) + 100.0
    original.to_netcdf(path, engine="h5netcdf")
    result = append_recipe(
        file_recipe(_file_source(path)), AverageOperation(dims=("time",))
    )
    xr.testing.assert_identical(_execute(result), original.qsel.mean("time"))
    updated.to_netcdf(path, engine="h5netcdf", mode="w")
    xr.testing.assert_identical(_execute(result), updated.qsel.mean("time"))
    xr.testing.assert_identical(_copied(result), updated.qsel.mean("time"))


def test_structured_file_replay_accepts_a_parsed_index_without_load_code(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    original = _scan()
    original.to_netcdf(path, engine="h5netcdf")
    source = _file_source(path)
    assert source.replay_call is not None
    source = source.model_copy(
        update={
            "replay_call": source.replay_call.model_copy(
                update={"selection": FileDataSelection(kind="parsed_index", value=0)}
            )
        }
    )
    assert source.load_code is None
    graph = append_recipe(file_recipe(source), AverageOperation(dims=("time",)))
    xr.testing.assert_identical(_execute(graph), original.qsel.mean("time"))


@pytest.mark.parametrize("factory", [memory_recipe, snapshot_recipe])
def test_owned_sources_resolve_by_identity_and_emit_caller_inputs(factory) -> None:
    data = _scan()
    source = factory("owned-value", label="arbitrary window title")
    result = append_recipe(source, AverageOperation(dims=("time",)))
    xr.testing.assert_identical(
        _execute(result, {"owned-value": data}), data.qsel.mean("time")
    )
    xr.testing.assert_identical(
        _copied(result, {"owned-value": data}), data.qsel.mean("time")
    )


def test_fixed_legacy_input_does_not_evaluate_its_historical_file(
    tmp_path: pathlib.Path,
) -> None:
    historical = _legacy_file(tmp_path / "missing.nc")
    legacy = recipe(
        ReplayStep(operation=AverageOperation(dims=("time",))),
        start_label="Average saved scan",
        active_name="result",
        primary_input="data",
        script_inputs=(
            ScriptInput(
                name="data",
                revision_id="fixed",
                follow_source=False,
                provenance_spec=historical,
            ),
        ),
    )
    adapted = from_legacy_recipe(legacy)
    assert any(isinstance(node, SnapshotSourceNode) for node in adapted.nodes)
    assert not any(isinstance(node, FileSourceNode) for node in adapted.nodes)
    xr.testing.assert_identical(
        _execute(adapted, {"fixed": _scan()}), _scan().qsel.mean("time")
    )
    xr.testing.assert_identical(
        _copied(adapted, {"fixed": _scan()}), _scan().qsel.mean("time")
    )


def test_shared_source_is_resolved_once() -> None:
    source = memory_recipe("shared")
    result = append_recipe(
        source, ConcatOperation(input_count=2, dim="repeat"), inputs={"other_0": source}
    )
    calls = []

    def resolve(node):
        calls.append(node.source_id)
        return _scan()

    actual = execute_replay_graph(compile_replay_graph(result, source_resolver=resolve))
    xr.testing.assert_identical(actual, xr.concat([_scan(), _scan()], dim="repeat"))
    assert calls == ["shared"]


def test_display_labels_do_not_select_or_change_dependencies() -> None:
    data, edge = _scan(), _scan().qsel.mean(("time", "eV")) / 1000.0
    graph = ToolProvenanceSpec(
        schema_version=4,
        kind="graph",
        nodes=(
            MemorySourceNode(id="scan", source_id="scan-value", label="same"),
            MemorySourceNode(id="edge", source_id="edge-value", label="same"),
            OperationNode(
                id="shift",
                operation=ShiftFromInputOperation(along="eV"),
                data=ValueRef(node_id="scan"),
                inputs={"shift": ValueRef(node_id="edge")},
                label="same",
            ),
        ),
        output=ValueRef(node_id="shift"),
    )
    renamed = graph.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"label": f"Renamed {index}"})
                for index, node in enumerate(graph.nodes)
            )
        }
    )
    values = {"scan-value": data, "edge-value": edge}
    assert _semantic_graph(graph) == _semantic_graph(renamed)
    assert (
        compile_replay_graph(graph, display=True).output_key
        == compile_replay_graph(renamed, display=True).output_key
    )
    expected = erlab.analysis.transform.shift(data, edge, along="eV")
    xr.testing.assert_identical(_execute(renamed, values), expected)
    xr.testing.assert_identical(_copied(renamed, values), expected)


@pytest.mark.parametrize("label", ["processed_data", "result", "np"])
def test_uniform_operation_names_do_not_change_results(label: str) -> None:
    source = memory_recipe("scan", label=label)
    operation = UniformInterpolationOperation(sizes={"eV": 9})
    graph = append_recipe(source, operation, label=label)
    expected = operation.apply(_scan())
    xr.testing.assert_identical(_execute(graph, {"scan": _scan()}), expected)
    xr.testing.assert_identical(
        _copied(graph, {"scan": _scan()}, input_names={"scan": "scan"}), expected
    )


@pytest.mark.parametrize("label", ["processed_data", "result", "data"])
def test_concat_auxiliary_is_not_overwritten_by_a_preferred_output_name(
    label: str,
) -> None:
    original, auxiliary = _scan(), _scan() + 1000.0
    interpolation = UniformInterpolationOperation(sizes={"eV": 9})
    prepared = append_recipe(memory_recipe("scan", label=label), interpolation)
    graph = append_recipe(
        prepared,
        ConcatOperation(input_count=2, dim="scan_index"),
        inputs={"other_0": memory_recipe("auxiliary", label=label)},
    )
    expected = xr.concat(
        [interpolation.apply(original), auxiliary],
        dim="scan_index",
        coords="minimal",
        compat="override",
        join="outer",
    )
    values = {"scan": original, "auxiliary": auxiliary}
    xr.testing.assert_identical(_execute(graph, values), expected)
    xr.testing.assert_identical(
        _copied(
            graph, values, input_names={"scan": "scan", "auxiliary": "processed_data"}
        ),
        expected,
    )


def test_code_nodes_have_separate_local_names_and_explicit_inputs() -> None:
    nodes = (
        MemorySourceNode(id="a", source_id="first", label="data"),
        MemorySourceNode(id="b", source_id="second", label="data"),
        CodeNode(
            id="first-code",
            code="derived = data + 1",
            inputs={"data": ValueRef(node_id="a")},
            output_name="derived",
            label="result",
        ),
        CodeNode(
            id="second-code",
            code="derived = data * 2",
            inputs={"data": ValueRef(node_id="b")},
            output_name="derived",
            label="result",
        ),
        OperationNode(
            id="concat",
            operation=ConcatOperation(input_count=2, dim="scan_index"),
            data=ValueRef(node_id="first-code"),
            inputs={"other_0": ValueRef(node_id="second-code")},
        ),
    )
    graph = ToolProvenanceSpec(
        schema_version=4, kind="graph", nodes=nodes, output=ValueRef(node_id="concat")
    )
    values = {"first": _scan(), "second": _scan() + 100.0}
    expected = xr.concat([values["first"] + 1, values["second"] * 2], dim="scan_index")
    xr.testing.assert_identical(_execute(graph, values), expected)
    xr.testing.assert_identical(_copied(graph, values), expected)


def test_independently_authored_code_runs_separately_while_shared_code_runs_once() -> (
    None
):
    source = memory_recipe("scan")
    first = code_recipe(
        "result = data + 1", inputs={"data": source}, output_name="result"
    )
    second = code_recipe(
        "result = data + 1", inputs={"data": source}, output_name="result"
    )
    expected = xr.concat([_scan() + 1, _scan() + 1], dim="repeat")
    for auxiliary, execution_count in ((second, 2), (first, 1)):
        combined = append_recipe(
            first,
            ConcatOperation(input_count=2, dim="repeat"),
            inputs={"other_0": auxiliary},
        )
        assert (
            sum(isinstance(node, CodeNode) for node in combined.nodes)
            == execution_count
        )
        compiled = compile_replay_graph(
            combined,
            trusted_user_code=True,
            source_resolver=_resolve({"scan": _scan()}),
        )
        code_keys = {node.key for node in compiled.nodes if node.kind == "script"}
        assert len(code_keys) == execution_count
        visited = []
        actual = execute_replay_graph(
            compiled,
            authorize=lambda entries: issue_execution_capability(
                new_document_trust(), entries
            )[1],
            _on_value=lambda key, _value, visited=visited: visited.append(key),
        )
        assert sum(key in code_keys for key in visited) == execution_count
        xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("code_as_auxiliary", [False, True])
def test_repeated_code_evaluation_invalidates_typed_descendant_cache(
    monkeypatch, code_as_auxiliary: bool
) -> None:
    draw_values = iter((1.0, 2.0))
    calls = []

    def draw(size):
        calls.append(size)
        return np.full(size, next(draw_values))

    monkeypatch.setattr(np.random, "random", draw)
    producer = code_recipe(
        "import numpy as np\nimport xarray as xr\n"
        "result = xr.DataArray(np.random.random(3), dims='x')",
        output_name="result",
    )
    zeros = xr.DataArray(np.zeros(3), dims="x")
    if code_as_auxiliary:
        producer = append_recipe(
            memory_recipe("fixed"),
            ConcatOperation(input_count=2, dim="branch"),
            inputs={"other_0": producer},
        )
    graph = append_recipe(producer, AssignAttrsOperation(attrs={"processed": True}))
    compiled = compile_replay_graph(
        graph, trusted_user_code=True, source_resolver=_resolve({"fixed": zeros})
    )
    cache = {}
    for value in (1.0, 2.0):
        actual = execute_replay_graph(
            compiled,
            cache=cache,
            authorize=lambda entries: issue_execution_capability(
                new_document_trust(), entries
            )[1],
        )
        expected = xr.DataArray(np.full(3, value), dims="x")
        if code_as_auxiliary:
            expected = xr.concat([zeros, expected], dim="branch")
        xr.testing.assert_identical(actual, expected.assign_attrs(processed=True))
    assert calls == [3, 3]


def test_code_nodes_require_runtime_trust() -> None:
    graph = ToolProvenanceSpec(
        schema_version=4,
        kind="graph",
        nodes=(
            CodeNode(
                id="code",
                code="import xarray as xr\nresult = xr.DataArray([1., 2.], dims='x')",
                inputs={},
                output_name="result",
            ),
        ),
        output=ValueRef(node_id="code"),
    )
    with pytest.raises((ReplayGraphError, ValueError, RuntimeError)):
        execute_replay_graph(compile_replay_graph(graph))
    xr.testing.assert_identical(_execute(graph), xr.DataArray([1.0, 2.0], dims="x"))


@pytest.mark.parametrize("with_input", [False, True])
def test_invalid_code_closure_keeps_its_stored_trust_inventory(
    with_input: bool,
) -> None:
    code = "result = unknown + 1"
    graph = code_recipe(
        code,
        output_name="result",
        inputs={"data": memory_recipe("source")} if with_input else None,
    )
    with pytest.raises((ReplayGraphError, ValueError)):
        compile_replay_graph(
            graph, trusted_user_code=True, source_resolver=_resolve({"source": _scan()})
        )
    entries = provenance_code_trust_entries(graph, location_prefix="graph")
    assert entries
    original_entry = next(entry for entry in entries if entry.code == code)
    assert original_entry.context == {
        "active_name": "result",
        "binding_names": ["data"] if with_input else [],
    }


def test_code_mutation_does_not_modify_a_shared_source() -> None:
    source = MemorySourceNode(id="source", source_id="shared")
    code = CodeNode(
        id="code",
        code="data += 100\nresult = data",
        inputs={"data": ValueRef(node_id="source")},
        output_name="result",
    )
    graph = ToolProvenanceSpec(
        schema_version=4,
        kind="graph",
        nodes=(
            source,
            code,
            OperationNode(
                id="concat",
                operation=ConcatOperation(input_count=2, dim="scan_index"),
                data=ValueRef(node_id="code"),
                inputs={"other_0": ValueRef(node_id="source")},
            ),
        ),
        output=ValueRef(node_id="concat"),
    )
    original = _scan()
    values = {"shared": original.copy(deep=True)}
    expected = xr.concat([original + 100, original], dim="scan_index")
    xr.testing.assert_identical(_execute(graph, values), expected)
    xr.testing.assert_identical(values["shared"], original)
    xr.testing.assert_identical(_copied(graph, values), expected)
    xr.testing.assert_identical(values["shared"], original)


def test_schema_four_roundtrip_has_typed_edges_and_no_legacy_or_window_fields() -> None:
    graph = append_recipe(
        memory_recipe("scan"),
        ShiftFromInputOperation(along="eV"),
        inputs={"shift": snapshot_recipe("edge-revision")},
    )
    payload = json.loads(graph.model_dump_json())
    assert payload["schema_version"] == 4
    assert payload["kind"] == "graph"
    assert (
        not {
            "script_inputs",
            "steps",
            "seed_code",
            "primary_input",
            "root_input",
            "file_load_source",
        }
        & payload.keys()
    )
    assert not any(
        {"node_uid", "node_snapshot_token", "follow_source", "provenance_spec"}
        & node.keys()
        for node in payload["nodes"]
    )
    reparsed = parse_tool_provenance_spec(payload)
    assert reparsed is not None
    assert reparsed.model_dump_json() == graph.model_dump_json()
    assert isinstance(reparsed.output, ValueRef)
    assert any(
        isinstance(node, OperationNode) and isinstance(node.inputs["shift"], ValueRef)
        for node in reparsed.nodes
    )


@pytest.mark.parametrize("kind", ["file", "script", "full_data"])
def test_legacy_schema_three_stays_unchanged_until_explicit_adaptation(
    tmp_path: pathlib.Path, kind: str
) -> None:
    if kind == "file":
        legacy = _legacy_file(tmp_path / "scan.nc")
    elif kind == "script":
        legacy = script(
            AssignAttrsOperation(attrs={"legacy": True}),
            start_label="Old scan",
            seed_code="result = data",
            active_name="result",
        )
    else:
        legacy = full_data(AverageOperation(dims=("time",)))
    payload = legacy.model_dump(mode="json")
    assert payload["schema_version"] == 3
    reparsed = parse_tool_provenance_spec(payload)
    assert reparsed is not None
    assert reparsed.model_dump(mode="json") == payload
    assert reparsed.kind == kind


def test_legacy_file_adaptation_preserves_typed_operations_and_executes(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    _scan().to_netcdf(path, engine="h5netcdf")
    legacy = _legacy_file(path).append_operations(AverageOperation(dims=("time",)))
    adapted = from_legacy_recipe(legacy)
    assert adapted.schema_version == 4
    xr.testing.assert_identical(_execute(adapted), _execute(legacy))
    xr.testing.assert_identical(_copied(adapted), _scan().qsel.mean("time"))


def test_legacy_named_inputs_adapt_without_rebinding_equal_names(
    tmp_path: pathlib.Path,
) -> None:
    first, second = _scan() / 1000.0, _scan(4) + 100.0
    paths = [tmp_path / "first.nc", tmp_path / "second.nc"]
    first.to_netcdf(paths[0], engine="h5netcdf")
    second.to_netcdf(paths[1], engine="h5netcdf")
    edge = _legacy_file(paths[0]).append_operations(
        AverageOperation(dims=("time", "eV"))
    )
    primary = _legacy_file(paths[1]).append_operations(AverageOperation(dims=("time",)))
    legacy = recipe(
        ReplayStep(
            operation=ShiftFromInputOperation(along="eV", negate=True),
            input_bindings={"shift": "edge"},
        ),
        start_label="Shift scan",
        active_name="result",
        primary_input="data",
        script_inputs=(
            ScriptInput(name="data", provenance_spec=primary),
            ScriptInput(name="edge", provenance_spec=edge),
        ),
    )
    adapted = from_legacy_recipe(legacy)
    expected = erlab.analysis.transform.shift(
        second.qsel.mean("time"), -first.qsel.mean(("time", "eV")), along="eV"
    )
    xr.testing.assert_identical(_execute(adapted), expected)
    xr.testing.assert_identical(_copied(adapted), expected)


@pytest.mark.parametrize("constructor", [full_data, public_data, selection])
def test_relative_legacy_view_requires_an_explicit_parent(constructor) -> None:
    legacy = constructor(AverageOperation(dims=("time",)))
    with pytest.raises((TypeError, ValueError)):
        from_legacy_recipe(legacy)
    adapted = from_legacy_recipe(legacy, parent=memory_recipe("scan"))
    xr.testing.assert_identical(
        _execute(adapted, {"scan": _scan()}), _scan().qsel.mean("time")
    )


@pytest.mark.parametrize("case", ["opaque", "legacy-context"])
def test_legacy_adapter_does_not_guess_opaque_execution_contracts(case: str) -> None:
    if case == "opaque":
        legacy = script(
            ScriptCodeOperation(label="Add", code="result = data + 1"),
            start_label="Old code",
            active_name="result",
            script_inputs=(
                ScriptInput(name="data", revision_id="data", follow_source=False),
            ),
        )
    else:
        legacy = recipe(
            ReplayStep(operation=AverageOperation(dims=("time",))),
            start_label="Average",
            active_name="result",
            primary_input="data",
            script_inputs=(
                ScriptInput(name="data", revision_id="data", follow_source=False),
            ),
        )
        step = ReplayStep.model_validate(
            {
                "operation": AverageOperation(dims=("time",)),
                "legacy_context": {"index": 0, "input_policy": "current"},
            }
        )
        legacy = legacy.model_copy(update={"steps": (step,)})
    with pytest.raises((TypeError, ValueError)):
        from_legacy_recipe(legacy)


def test_legacy_input_resolver_is_explicit() -> None:
    legacy = recipe(
        ReplayStep(operation=AverageOperation(dims=("time",))),
        start_label="Average",
        active_name="result",
        primary_input="data",
        script_inputs=(ScriptInput(name="data", node_uid="window-only"),),
    )
    with pytest.raises((TypeError, ValueError)):
        from_legacy_recipe(legacy)
    seen = []

    def resolve(item):
        seen.append(item.node_uid)
        return memory_recipe("owned")

    graph = from_legacy_recipe(legacy, input_resolver=resolve)
    assert seen == ["window-only"]
    xr.testing.assert_identical(
        _execute(graph, {"owned": _scan()}), _scan().qsel.mean("time")
    )


def test_builders_own_operation_values_and_preserve_sources() -> None:
    source = memory_recipe("owned")
    operation = AssignAttrsOperation(attrs={"state": {"original": True}})
    result = append_recipe(source, operation)
    before = result.model_dump_json()
    operation.attrs["state"]["original"] = False
    assert result.model_dump_json() == before
    assert len(source.nodes) == 1
    assert isinstance(source.nodes[0], MemorySourceNode)


def test_graft_rejects_a_recipe_without_an_input_placeholder() -> None:
    with pytest.raises((TypeError, ValueError)):
        graft_recipe(memory_recipe("first"), memory_recipe("second"))


def test_changing_a_legacy_kind_to_graph_revalidates_required_graph_fields() -> None:
    with pytest.raises((TypeError, ValueError)):
        full_data().model_copy(update={"kind": "graph", "schema_version": 4})


@pytest.mark.parametrize("factory", [memory_recipe, snapshot_recipe])
def test_owned_source_requires_a_nonempty_identity(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory("")


def _invalid_graph(case: str) -> ToolProvenanceSpec:
    source = MemorySourceNode(id="source", source_id="owned")
    nodes = [source]
    output, input_ref = ValueRef(node_id="source"), None
    extra = {}
    if case == "empty":
        nodes = []
    elif case == "duplicate":
        nodes.append(MemorySourceNode(id="source", source_id="different"))
    elif case == "output":
        output = ValueRef(node_id="missing")
    elif case in {"primary", "self-cycle", "cycle"}:
        parent = {
            "primary": "missing",
            "self-cycle": "operation",
            "cycle": "other",
        }[case]
        nodes.append(
            OperationNode(
                id="operation",
                operation=AverageOperation(dims=("time",)),
                data=ValueRef(node_id=parent),
            )
        )
        output = ValueRef(node_id="operation")
        if case == "cycle":
            nodes.append(
                OperationNode(
                    id="other",
                    operation=AverageOperation(dims=("alpha",)),
                    data=output,
                )
            )
    elif case in {"auxiliary", "missing-slot", "extra-slot"}:
        operation = ShiftFromInputOperation(along="eV")
        inputs = {"shift": ValueRef(node_id="missing")}
        if case == "missing-slot":
            inputs = {}
        elif case == "extra-slot":
            operation = AverageOperation(dims=("time",))
            inputs = {"shift": output}
        nodes.append(
            OperationNode(
                id="operation", operation=operation, data=output, inputs=inputs
            )
        )
        output = ValueRef(node_id="operation")
    elif case == "input-ref":
        input_ref = output
    elif case == "window-field":
        nodes = [{**source.model_dump(mode="json"), "node_uid": "window"}]
    else:
        extra["script_inputs"] = (ScriptInput(name="data"),)
    return ToolProvenanceSpec(
        schema_version=4,
        kind="graph",
        nodes=nodes,
        output=output,
        input_ref=input_ref,
        **extra,
    )


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "duplicate",
        "output",
        "primary",
        "auxiliary",
        "self-cycle",
        "cycle",
        "missing-slot",
        "extra-slot",
        "input-ref",
        "window-field",
        "legacy-field",
    ],
)
def test_graph_validation_rejects_ambiguous_or_incomplete_derivations(
    case: str,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _invalid_graph(case)


def test_invalid_python_is_reported_as_graph_validation_failure() -> None:
    with pytest.raises(ValueError, match="valid Python"):
        code_recipe("result = (", output_name="result")


def test_topological_order_uses_typed_references_not_storage_order() -> None:
    graph = ToolProvenanceSpec(
        schema_version=4,
        kind="graph",
        nodes=(
            OperationNode(
                id="operation",
                operation=AverageOperation(dims=("time",)),
                data=ValueRef(node_id="source"),
            ),
            MemorySourceNode(id="source", source_id="owned"),
        ),
        output=ValueRef(node_id="operation"),
    )
    assert [node.id for node in graph.topological_nodes()] == ["source", "operation"]
    xr.testing.assert_identical(
        _execute(graph, {"owned": _scan()}), _scan().qsel.mean("time")
    )
