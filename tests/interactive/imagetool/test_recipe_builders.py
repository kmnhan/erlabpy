import json
import pathlib
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive._code_trust import issue_execution_capability, new_document_trust
from erlab.interactive.imagetool._load_source import _register_local_callable_loader
from erlab.interactive.imagetool._provenance._execution import execute_replay_graph
from erlab.interactive.imagetool._provenance._graph import (
    ResolvedReplayInput,
    compile_replay_graph,
    emit_replay_code,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    file_load,
    full_data,
    parse_tool_provenance_spec,
    recipe,
    script,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    AverageOperation,
    ScriptCodeOperation,
    ShiftFromInputOperation,
)
from erlab.interactive.imagetool._provenance._recipes import (
    append_recipe,
    graft_recipe,
    producer_input,
    snapshot_input,
)


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


def _file_producer(path: pathlib.Path) -> ToolProvenanceSpec:
    _register_local_callable_loader(xr.load_dataarray)
    return file_load(
        start_label="Load scan",
        seed_code=(
            "import xarray as xr\n\n"
            f"derived = xr.load_dataarray({str(path)!r}, engine='h5netcdf')"
        ),
        file_load_source=FileLoadSource(
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
        ),
    )


def _execute(
    spec: ToolProvenanceSpec,
    resolver: Callable[[ScriptInput], ResolvedReplayInput | None] | None = None,
) -> xr.DataArray:
    return execute_replay_graph(
        compile_replay_graph(
            spec, structured_file_replay=True, live_input_resolver=resolver
        )
    )


@pytest.mark.parametrize("bound_shift", [False, True])
def test_append_and_graft_build_the_same_executable_recipe(
    tmp_path: pathlib.Path, bound_shift: bool
) -> None:
    data = _scan()
    path = tmp_path / "scan.nc"
    data.to_netcdf(path, engine="h5netcdf")
    source = producer_input(_file_producer(path), name="scan")
    steps = [ReplayStep(operation=AverageOperation(dims=("time",)))]
    offsets = xr.DataArray([0.0, 0.1, -0.1], dims="alpha", coords={"alpha": data.alpha})
    inputs = ()
    expected = data.qsel.mean("time")
    if bound_shift:
        inputs = (snapshot_input("edge-revision", name="edge"),)
        steps.append(
            ReplayStep(
                operation=ShiftFromInputOperation(along="eV", negate=True),
                input_bindings={"shift": "edge"},
            )
        )
        expected = erlab.analysis.transform.shift(expected, -offsets, along="eV")
    appended = append_recipe(
        source, steps, inputs=inputs, start_label="Process scan", active_name="result"
    )
    fragment = recipe(
        *steps,
        start_label="Process scan",
        active_name="result",
        primary_input="scan",
        script_inputs=(ScriptInput(name="scan"), *inputs),
    )
    grafted = graft_recipe(fragment, source)
    assert grafted == appended
    assert appended.schema_version == 3
    roundtripped = parse_tool_provenance_spec(json.loads(appended.model_dump_json()))
    assert roundtripped is not None
    assert roundtripped.model_dump_json() == appended.model_dump_json()
    assert fragment.script_inputs[0] == ScriptInput(name="scan")

    def resolve(item: ScriptInput) -> ResolvedReplayInput | None:
        if item.revision_id == "edge-revision":
            return ResolvedReplayInput(offsets, item)
        return None

    xr.testing.assert_identical(_execute(appended, resolve), expected)
    xr.testing.assert_identical(_execute(grafted, resolve), expected)
    xr.testing.assert_identical(_execute(roundtripped, resolve), expected)
    code = emit_replay_code(
        compile_replay_graph(appended, display=True), output_name="result"
    )
    namespace = {"edge": offsets, "erlab": erlab, "xr": xr}
    exec(code, namespace, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], expected)


def test_producer_reload_grows_while_snapshot_replay_keeps_captured_data(
    tmp_path: pathlib.Path,
) -> None:
    original = _scan()
    path = tmp_path / "scan.nc"
    original.to_netcdf(path, engine="h5netcdf")
    producer = _file_producer(path)
    step = ReplayStep(operation=AssignAttrsOperation(attrs={"processed": True}))
    current = append_recipe(
        producer_input(producer), (step,), start_label="Mark scan", active_name="result"
    )
    captured = append_recipe(
        snapshot_input("captured", producer=producer),
        (step,),
        start_label="Mark scan",
        active_name="result",
    )
    xr.testing.assert_identical(
        _execute(current), original.assign_attrs(processed=True)
    )
    updated = _scan(5) + 100.0
    updated.to_netcdf(path, engine="h5netcdf", mode="w")

    def resolve(item: ScriptInput) -> ResolvedReplayInput | None:
        if not item.follow_source and item.revision_id == "captured":
            return ResolvedReplayInput(original, item)
        return None

    xr.testing.assert_identical(_execute(current), updated.assign_attrs(processed=True))
    xr.testing.assert_identical(
        _execute(captured, resolve), original.assign_attrs(processed=True)
    )
    assert captured.script_inputs[0].parsed_provenance_spec() == producer
    assert current.script_inputs[0].node_uid is None
    assert current.script_inputs[0].revision_id is None


def test_recipe_builders_own_nested_values(tmp_path: pathlib.Path) -> None:
    producer = _file_producer(tmp_path / "scan.nc")
    source = producer_input(producer)
    auxiliary = snapshot_input("edge", name="edge", producer=producer)
    operation = AssignAttrsOperation(attrs={"stage": "original"})
    step = ReplayStep(operation=operation)
    result = append_recipe(
        source,
        (step,),
        inputs=(auxiliary,),
        start_label="Mark scan",
        active_name="result",
    )
    original = result.model_dump_json()
    assert producer.file_load_source is not None
    assert producer.file_load_source.replay_call is not None
    producer.file_load_source.replay_call.kwargs["engine"] = "changed"
    assert source.provenance_spec is not None
    source.provenance_spec["start_label"] = "Changed source"
    assert auxiliary.provenance_spec is not None
    auxiliary.provenance_spec["start_label"] = "Changed auxiliary"
    operation.attrs["stage"] = "changed"
    assert result.model_dump_json() == original


def test_append_preserves_following_input_checkpoints_and_source_transform(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    updated = _scan(5) + 100.0
    updated.to_netcdf(path, engine="h5netcdf")
    source = producer_input(_file_producer(path)).model_copy(
        update={
            "revision_id": "old-raw",
            "resolved_revision_id": "old-average",
            "node_uid": "source-window",
            "node_snapshot_token": "old-source-state",
            "source_spec": full_data(AverageOperation(dims=("time",))).model_dump(
                mode="json"
            ),
        }
    )
    result = append_recipe(
        source,
        (ReplayStep(operation=AssignAttrsOperation(attrs={"processed": True})),),
        start_label="Mark average",
        active_name="result",
    )
    assert result.script_inputs[0] == source
    xr.testing.assert_identical(
        _execute(result), updated.qsel.mean("time").assign_attrs(processed=True)
    )


def test_graft_renames_source_and_accepts_root_input() -> None:
    step = ReplayStep(operation=AverageOperation(dims=("time",)))
    fragment = script(
        start_label="Average scan",
        active_name="result",
        root_input=ScriptInput(name="scan"),
        steps=(step,),
    )
    source = snapshot_input("captured", name="original")
    grafted = graft_recipe(fragment, source)
    expected = append_recipe(
        source.model_copy(update={"name": "scan"}),
        (step,),
        start_label="Average scan",
        active_name="result",
    )
    assert grafted == expected
    assert source.name == "original"
    assert fragment.root_input == ScriptInput(name="scan")


@pytest.mark.parametrize(
    "source",
    [
        ScriptInput(name="data"),
        ScriptInput(name="data", node_uid="live-only"),
        ScriptInput(name="data", follow_source=False),
        ScriptInput(name="data", provenance_spec=full_data()),
    ],
)
def test_append_rejects_inputs_without_a_durable_source(source: ScriptInput) -> None:
    with pytest.raises((TypeError, ValueError)):
        append_recipe(
            source,
            (ReplayStep(operation=AverageOperation(dims=("time",))),),
            start_label="Average scan",
            active_name="result",
        )


def test_factories_reject_invalid_producers_and_checkpoint_ids() -> None:
    with pytest.raises((TypeError, ValueError)):
        producer_input(full_data())
    with pytest.raises((TypeError, ValueError)):
        snapshot_input("")


@pytest.mark.parametrize("seed_code", [None, "result = data"])
def test_producer_input_rejects_missing_script_sources(seed_code: str | None) -> None:
    with pytest.raises(ValueError, match="complete recipe"):
        producer_input(
            script(start_label="Source", active_name="result", seed_code=seed_code)
        )


def test_producer_input_keeps_self_contained_script_sources() -> None:
    producer = script(
        start_label="Create source",
        active_name="source",
        seed_code=(
            "import xarray as xr\n\n"
            "source = xr.DataArray([[1., 2.], [3., 4.]], dims=('alpha', 'eV'))"
        ),
    )
    result = append_recipe(
        producer_input(producer),
        (ReplayStep(operation=AverageOperation(dims=("alpha",))),),
        start_label="Average source",
        active_name="result",
    )
    actual = execute_replay_graph(
        compile_replay_graph(result, trusted_user_code=True),
        authorize=lambda entries: issue_execution_capability(
            new_document_trust(), entries
        )[1],
    )
    xr.testing.assert_identical(actual, xr.DataArray([2.0, 3.0], dims="eV"))


def test_graft_preserves_auxiliary_branch_with_the_same_primary_name(
    tmp_path: pathlib.Path,
) -> None:
    first = _scan() / 1000.0
    second = _scan(4) + 100.0
    first_path, second_path = tmp_path / "first.nc", tmp_path / "second.nc"
    first.to_netcdf(first_path, engine="h5netcdf")
    second.to_netcdf(second_path, engine="h5netcdf")
    edge = append_recipe(
        producer_input(_file_producer(first_path)),
        (ReplayStep(operation=AverageOperation(dims=("time", "eV"))),),
        start_label="Average first scan",
        active_name="edge",
    )
    prepared = append_recipe(
        producer_input(_file_producer(second_path)),
        (ReplayStep(operation=AverageOperation(dims=("time",))),),
        start_label="Average second scan",
        active_name="prepared",
    )
    fragment = recipe(
        ReplayStep(
            operation=ShiftFromInputOperation(along="eV", negate=True),
            input_bindings={"shift": "edge"},
        ),
        primary_input="data",
        script_inputs=(ScriptInput(name="data"), producer_input(edge, name="edge")),
        start_label="Shift scan",
        active_name="result",
    )
    grafted = graft_recipe(fragment, producer_input(prepared))
    assert grafted.script_inputs[1] == fragment.script_inputs[1]
    expected = erlab.analysis.transform.shift(
        second.qsel.mean("time"), -first.qsel.mean(("time", "eV")), along="eV"
    )
    xr.testing.assert_identical(_execute(grafted), expected)
    code = emit_replay_code(
        compile_replay_graph(grafted, display=True), output_name="result"
    )
    namespace = {"erlab": erlab, "xr": xr}
    exec(code, namespace, namespace)  # noqa: S102
    xr.testing.assert_identical(namespace["result"], expected)


@pytest.mark.parametrize("collision", ["primary", "auxiliary", "missing-binding"])
def test_append_rejects_ambiguous_or_missing_named_inputs(collision: str) -> None:
    source = snapshot_input("scan")
    inputs = (snapshot_input("edge", name="edge"),)
    if collision == "primary":
        inputs = (snapshot_input("edge", name="data"),)
    elif collision == "auxiliary":
        inputs = (*inputs, snapshot_input("other-edge", name="edge"))
    else:
        inputs = ()
    with pytest.raises((TypeError, ValueError)):
        append_recipe(
            source,
            (
                ReplayStep(
                    operation=ShiftFromInputOperation(along="eV"),
                    input_bindings={"shift": "edge"},
                ),
            ),
            inputs=inputs,
            start_label="Shift scan",
            active_name="result",
        )


@pytest.mark.parametrize("case", ["bound", "live", "code", "policy", "seed", "output"])
def test_graft_rejects_implicit_or_already_bound_primary_inputs(case: str) -> None:
    primary = ScriptInput(name="data")
    step = ReplayStep(operation=AverageOperation(dims=("time",)))
    if case == "bound":
        primary = snapshot_input("previous")
    elif case == "live":
        primary = ScriptInput(name="data", node_uid="previous")
    elif case == "code":
        step = ReplayStep(
            operation=ScriptCodeOperation(label="Copy", code="result = data")
        )
    elif case == "policy":
        step = step.model_copy(update={"input_policy": "current"})
    if case == "seed":
        fragment = script(
            start_label="Copy scan", seed_code="result = data", active_name="result"
        )
    else:
        fragment = script(
            start_label="Average scan",
            active_name=None if case == "output" else "result",
            primary_input="data",
            script_inputs=(primary,),
            steps=(step,),
        )
    with pytest.raises((TypeError, ValueError)):
        graft_recipe(fragment, snapshot_input("captured"))


def test_graft_rejects_a_file_root(tmp_path: pathlib.Path) -> None:
    with pytest.raises((TypeError, ValueError)):
        graft_recipe(_file_producer(tmp_path / "scan.nc"), snapshot_input("captured"))
