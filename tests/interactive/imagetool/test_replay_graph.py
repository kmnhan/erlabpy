import ast
import pathlib
import re
import textwrap
import types
import typing
from collections.abc import Collection, Mapping

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive.imagetool._provenance import _graph
from erlab.interactive.imagetool._provenance._code import (
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    _code_uses_name,
    _code_uses_name_any_scope,
    _nonuniform_restore_support_code,
    _replace_code_identifiers,
)
from erlab.interactive.imagetool._provenance._execution import (
    _script_provenance_validates,
    _script_trust_payload,
    execute_replay_graph,
    rebuild_script_provenance,
    replay_script_provenance,
    script_provenance_replayable,
    script_provenance_requires_trust,
    script_provenance_trust_key,
)
from erlab.interactive.imagetool._provenance._graph import (
    ReplayGraph,
    ReplayGraphError,
    _code_has_scoped_definition,
    _code_name_accesses,
    _compact_replay_temp_names,
    _file_seed_code_parts,
    _group_framework_imports,
    _import_binding_targets,
    _inline_single_use_replay_names,
    _is_future_import,
    _leading_top_level_imports,
    _operation_replay_code,
    _remove_noop_assignments,
    _replace_ast_names,
    _script_function_dependencies,
    _script_seed_file_load_parts,
    _simple_assignment_source_name,
    _single_assignment_output_name,
    _statement_scope_names,
    _validate_script_code_names,
    _validate_script_provenance,
    compile_replay_graph,
    emit_replay_code,
    script_inputs_code,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ReplayStage,
    ScriptInput,
    ToolProvenanceSpec,
    compose_full_provenance,
    file_load,
    full_data,
    public_data,
    script,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    AverageOperation,
    CoarsenOperation,
    CorrectWithEdgeOperation,
    DivideByCoordOperation,
    ImageDerivativeOperation,
    IselOperation,
    QSelOperation,
    RenameOperation,
    RestoreNonuniformDimsOperation,
    RotateOperation,
    ScriptCodeOperation,
    SelOperation,
    SortCoordOrderOperation,
    SqueezeOperation,
)


def _exec_generated_code(
    code: str, namespace_items: dict[str, typing.Any] | None = None
) -> dict[str, typing.Any]:
    namespace: dict[str, typing.Any] = {
        "erlab": erlab,
        "era": erlab.analysis,
        "np": np,
        "numpy": np,
        "xr": xr,
        "xarray": xr,
    }
    if namespace_items is not None:
        namespace.update(namespace_items)
    exec(code, namespace, namespace)  # noqa: S102
    return namespace


def _file_replay_source(
    path: pathlib.Path | str,
    *,
    selected_index: int = 0,
    load_code: str | None = None,
):
    return FileLoadSource(
        path=str(path),
        loader_label="xarray.load_dataarray",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        replay_call=FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            selected_index=selected_index,
        ),
        load_code=load_code,
    )


def _file_spec(path: pathlib.Path | str, *, selected_index: int = 0):
    return file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=_file_replay_source(path, selected_index=selected_index),
    )


def test_script_compiler_assigns_final_output_after_pending_seed() -> None:
    data = xr.DataArray(
        np.arange(3.0)[None, :],
        dims=("singleton", "x"),
    )
    spec = script(
        SqueezeOperation(),
        start_label="Start from data",
        seed_code="intermediate = data",
        active_name="result",
    )

    assert script_provenance_replayable(
        spec,
        external_input_names={"data"},
    )
    xr.testing.assert_identical(
        replay_script_provenance(spec, {"data": data}),
        data.squeeze(),
    )


def _erlab_file_spec(path: pathlib.Path | str, loader: str):
    return file_load(
        start_label=f"Load {path}",
        seed_code=(
            f"erlab.io.set_loader({loader!r})\nderived = erlab.io.load({str(path)!r})"
        ),
        file_load_source=FileLoadSource(
            path=str(path),
            loader_label="Loader",
            loader_text=loader,
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="erlab_loader",
                target=loader,
                selected_index=0,
            ),
        ),
    )


def _polarization_source(path: pathlib.Path) -> xr.DataArray:
    source = xr.DataArray(
        np.arange(12.0).reshape(2, 2, 3),
        dims=("pol", "energy", "k"),
        coords={"pol": ["LH", "LV"], "energy": [0.0, 1.0], "k": [0, 1, 2]},
    )
    source.to_netcdf(path)
    return source


def _assert_dense_replay_temps(code: str) -> None:
    temp_ids = sorted(
        {int(value) for value in re.findall(r"_itool_replay_(\d+)", code)}
    )
    if temp_ids:
        assert temp_ids == list(range(temp_ids[-1] + 1))


def test_replay_graph_low_level_validation_helpers() -> None:

    assert _code_uses_name("derived = data", "data")
    assert _code_uses_name_any_scope(
        "def identity(value):\n    _ = era\n    return value", "era"
    )
    assert not _code_uses_name("derived =", "data")
    assert _simple_assignment_source_name("target = source", "target") == "source"
    assert (
        _simple_assignment_source_name(
            "target: xr.DataArray = source",
            "target",
        )
        == "source"
    )
    assert _simple_assignment_source_name("target =", "target") is None
    assert (
        _simple_assignment_source_name("target = source\nother = source", "target")
        is None
    )

    module = ast.parse(
        """
@decorator
def helper(value=data_0, *, scale=data_1) -> data_2:
    return value

async def async_helper(value=data_3):
    return value

lambda_value = lambda value=data_4: value

@class_decorator
class Child(Base, metaclass=data_5):
    pass
"""
    )
    names = [_statement_scope_names(stmt) for stmt in module.body]
    loads = set().union(*(item.loads for item in names))
    stores = set().union(*(item.stores for item in names))
    assert {
        "decorator",
        "data_0",
        "data_1",
        "data_2",
        "data_3",
        "data_4",
        "class_decorator",
    }.issubset(loads)
    assert {"Base", "data_5"}.issubset(loads)
    assert {"helper", "async_helper", "lambda_value", "Child"}.issubset(stores)

    deps = _script_function_dependencies(
        "def helper():\n"
        "    def nested():\n"
        "        return missing\n"
        "    return nested()\n"
    )
    assert deps[("helper", 1)] == {"missing"}
    with pytest.raises(ReplayGraphError, match="unresolved name"):
        _validate_script_code_names(
            "def helper():\n    return missing\nresult = helper()",
            set(),
            {},
        )
    loop_names = {"axs", "profiles"}
    _validate_script_code_names(
        "for profile in profiles:\n    profile.plot(ax=axs, x='alpha')",
        loop_names,
        {},
    )
    assert "profile" in loop_names
    nested_loop_names = {"axs", "profiles", "show_profiles"}
    _validate_script_code_names(
        "if show_profiles:\n"
        "    for profile in profiles:\n"
        "        profile.plot(ax=axs, x='alpha')",
        nested_loop_names,
        {},
    )
    assert "profile" in nested_loop_names
    loop_else_names = {"items"}
    _validate_script_code_names(
        "for item in items:\n    pass\nelse:\n    derived = item",
        loop_else_names,
        {},
    )
    assert {"item", "derived"}.issubset(loop_else_names)
    comprehension_names = {"float", "profiles", "sum"}
    _validate_script_code_names(
        "line_color_values = [\n"
        '    float(profile.coords["sample_temp"].values.item())\n'
        "    for profile in profiles\n"
        "]\n"
        "profile_names = {profile.name for profile in profiles}\n"
        "profile_map = {profile.name: profile for profile in profiles}\n"
        "profile_total = sum(profile.sum() for profile in profiles)",
        comprehension_names,
        {},
    )
    assert {
        "line_color_values",
        "profile_names",
        "profile_map",
        "profile_total",
    }.issubset(comprehension_names)
    assert "profile" not in comprehension_names
    comprehension_condition_names = _statement_scope_names(
        ast.parse("values = [item for item in data if item > threshold]").body[0]
    )
    assert {"data", "threshold"}.issubset(comprehension_condition_names.loads)
    assert "item" not in comprehension_condition_names.loads
    generated_builtin_names = {
        *_SCRIPT_REPLAY_ALLOWED_BUILTINS,
        "values",
    }
    _validate_script_code_names(
        "indexed = [value for index, value in enumerate(values)]\n"
        "reordered = list(reversed(values))",
        generated_builtin_names,
        {},
    )
    assert {"indexed", "reordered"}.issubset(generated_builtin_names)
    with pytest.raises(ReplayGraphError, match="unresolved name"):
        _validate_script_code_names(
            "line_color_values = [missing + profile for profile in profiles]",
            {"profiles"},
            {},
        )
    with pytest.raises(ReplayGraphError, match="unresolved name"):
        _validate_script_code_names(
            "for holder.profile in profiles:\n    pass",
            {"profiles"},
            {},
        )
    with pytest.raises(ReplayGraphError, match="unresolved name"):
        _validate_script_code_names(
            "if use_left:\n    local_value = data\nelse:\n    derived = local_value",
            {"data", "use_left"},
            {},
        )
    unchanged_dependencies = {"helper": {"data"}}
    _validate_script_code_names(
        "if use_left:\n    left = data\nelse:\n    right = data",
        {"data", "use_left", "helper"},
        unchanged_dependencies,
    )
    assert unchanged_dependencies == {"helper": {"data"}}
    new_branch_dependencies: dict[str, set[str]] = {}
    _validate_script_code_names(
        "if use_left:\n"
        "    def choose():\n"
        "        return data\n"
        "else:\n"
        "    def choose():\n"
        "        return fallback\n"
        "derived = choose()",
        {"data", "fallback", "use_left"},
        new_branch_dependencies,
    )
    assert new_branch_dependencies["choose"] == {"data", "fallback"}
    exception_names = {"data", "ValueError"}
    _validate_script_code_names(
        "try:\n    data\nexcept ValueError as exc:\n    derived = exc",
        exception_names,
        {},
    )

    with pytest.raises(ReplayGraphError, match="Expected script"):
        _validate_script_provenance(full_data(SqueezeOperation()))
    with pytest.raises(ReplayGraphError, match="without active_name"):
        _validate_script_provenance(
            types.SimpleNamespace(kind="script", active_name=None)
        )
    with pytest.raises(ReplayGraphError, match="unsupported Import"):
        _validate_script_provenance(
            script(
                start_label="Run script",
                seed_code="import os",
                active_name="derived",
                script_inputs=(ScriptInput(name="data_0", label="Input"),),
            )
        )
    with pytest.raises(ReplayGraphError, match="no replay code"):
        _validate_script_provenance(
            script(
                AverageOperation(dims=("x",)),
                start_label="Run script",
                active_name="derived",
            )
        )
    with pytest.raises(ReplayGraphError, match="no replay code"):
        _validate_script_provenance(
            script(start_label="Run script", active_name="derived")
        )
    derivative_operation = ImageDerivativeOperation(
        method="diffn",
        kwargs={"coord": "x", "order": 2},
    )
    with pytest.raises(ReplayGraphError, match="no replay code"):
        _validate_script_provenance(
            script(
                derivative_operation,
                start_label="Run script",
                seed_code="derived = data",
                active_name="derived",
                script_inputs=(ScriptInput(name="data", label="Input"),),
            )
        )
    _validate_script_provenance(
        script(
            derivative_operation,
            ScriptCodeOperation(
                label="Use derivative output",
                code="derived = result",
                visible=False,
            ),
            start_label="Run script",
            seed_code="derived = data",
            active_name="derived",
            script_inputs=(ScriptInput(name="data", label="Input"),),
        )
    )
    with pytest.raises(ReplayGraphError, match="non-replayable"):
        _validate_script_provenance(
            script(
                ScriptCodeOperation(
                    label="Opaque",
                    code=None,
                    copyable=False,
                ),
                start_label="Run script",
                active_name="derived",
                script_inputs=(ScriptInput(name="data_0", label="Input"),),
            )
        )
    with pytest.raises(ReplayGraphError, match="no replay code"):
        _validate_script_provenance(
            script(
                start_label="Run script",
                active_name="derived",
                replay_stages=(
                    ReplayStage(
                        source_kind="full_data",
                        operations=(AverageOperation(dims=("x",)),),
                    ),
                ),
            )
        )
    invalid_stage_spec = script(
        start_label="Run script",
        seed_code="derived = 1",
        active_name="derived",
    ).model_copy(
        update={
            "replay_stages": (
                ReplayStage.model_construct(
                    source_kind="full_data",
                    operations=(
                        ScriptCodeOperation(
                            label="Opaque",
                            code=None,
                            copyable=False,
                        ),
                    ),
                ),
            )
        }
    )
    with pytest.raises(ReplayGraphError, match="non-replayable"):
        _validate_script_provenance(invalid_stage_spec)
    assert not script_provenance_replayable(None)
    external_input_spec = script(
        ScriptCodeOperation(
            label="Use external input",
            code="derived = data + 1",
        ),
        start_label="Run script",
        seed_code="derived = data",
        active_name="derived",
    )
    assert not script_provenance_replayable(external_input_spec)
    assert script_provenance_replayable(
        external_input_spec,
        external_input_names={"data"},
    )
    assert not _script_provenance_validates(
        script(
            start_label="Run script",
            seed_code="derived =",
            active_name="derived",
        ),
        strict_replay_code=False,
    )
    assert not _script_provenance_validates(
        script(
            ScriptCodeOperation(label="Broken", code="derived ="),
            start_label="Run script",
            seed_code="derived = 0",
            active_name="derived",
        ),
        strict_replay_code=False,
    )
    assert not _script_provenance_validates(
        None,
        strict_replay_code=True,
    )
    assert _single_assignment_output_name("derived: xr.DataArray = data") == "derived"
    assert _single_assignment_output_name("derived =") is None
    assert _single_assignment_output_name("obj.value = data") is None
    assert _single_assignment_output_name("first = data\nsecond = data") is None
    assert script_provenance_trust_key(None) is None


def test_replay_graph_script_context_binding_error_paths() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    invalid_context = ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        active_name="result",
        operations=(
            ScriptCodeOperation(
                label="Use pasted context",
                code="result = data + 1",
            ),
        ),
        script_context_bindings=[
            {"operation_index": 0, "names": ["data"]},
        ],
    )

    with pytest.raises(ReplayGraphError, match="no replay code"):
        _validate_script_provenance(invalid_context)

    rebound_input = ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        active_name="data_0",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
        operations=(
            ScriptCodeOperation(
                label="Offset rebound input",
                code="data_0 = derived + 1",
            ),
        ),
        script_context_bindings=[
            {"operation_index": 0, "names": ["derived"]},
        ],
    )
    rebound_graph = compile_replay_graph(
        rebound_input,
        external_inputs={"data_0": data},
    )
    rebound_display_graph = compile_replay_graph(
        rebound_input,
        display=True,
        external_inputs={"data_0": data},
    )
    xr.testing.assert_identical(
        execute_replay_graph(rebound_graph),
        data + 1,
    )
    assert rebound_display_graph.output_key is not None

    active_relay = ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="result = data",
        active_name="result",
        operations=(
            ScriptCodeOperation(
                label="Write alternate output",
                code="derived = result + 1",
            ),
        ),
    )
    active_graph = compile_replay_graph(
        active_relay,
        external_inputs={"data": data},
    )

    xr.testing.assert_identical(
        execute_replay_graph(active_graph),
        data,
    )
    assert _remove_noop_assignments("derived =") == "derived ="


def test_replay_graph_manual_error_and_cache_paths() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))

    graph = ReplayGraph()
    live_key = graph.add_node("live", "live_input", payload={"data": data})
    graph.output_key = live_key
    with pytest.raises(ReplayGraphError, match="Live inputs"):
        emit_replay_code(graph)
    xr.testing.assert_identical(
        execute_replay_graph(graph, cache={live_key: data + 1.0}),
        data + 1.0,
    )
    replayed = execute_replay_graph(graph)
    xr.testing.assert_identical(replayed, data)
    assert not np.shares_memory(replayed.data, data.data)

    relay_graph = ReplayGraph()
    relay_live_key = relay_graph.add_node("live", "live_input", payload={"data": data})
    relay_key = relay_graph.add_node("relay", "relay", parents=(relay_live_key,))
    relay_graph.output_key = relay_key
    xr.testing.assert_identical(execute_replay_graph(relay_graph), data)

    for load_code, message in (
        ("derived =", "not valid Python"),
        ("other = xr.DataArray([1.0], dims=('x',))", "does not assign"),
    ):
        file_graph = ReplayGraph()
        file_key = file_graph.add_node(
            f"file-{message}",
            "file_load",
            payload={
                "active_name": "derived",
                "load_code": load_code,
                "load_source": None,
            },
        )
        file_graph.output_key = file_key
        with pytest.raises(ReplayGraphError, match=message):
            emit_replay_code(file_graph)

    for codes, message in (
        (("other = xr.DataArray([1.0], dims=('x',))",), "did not create"),
        (("derived = 1",), "did not produce"),
    ):
        script_graph = ReplayGraph()
        script_key = script_graph.add_node(
            f"script-{message}",
            "script",
            payload={"bindings": (), "codes": codes, "active_name": "derived"},
        )
        script_graph.output_key = script_key
        with pytest.raises(ReplayGraphError, match=message):
            execute_replay_graph(script_graph)

    unknown_graph = ReplayGraph()
    unknown_key = unknown_graph.add_node("unknown", "unknown")
    unknown_graph.output_key = unknown_key
    with pytest.raises(ReplayGraphError, match="Unknown replay"):
        emit_replay_code(unknown_graph)
    with pytest.raises(ReplayGraphError, match="Unknown replay"):
        execute_replay_graph(unknown_graph)

    empty_graph = ReplayGraph()
    with pytest.raises(ReplayGraphError, match="no output"):
        emit_replay_code(empty_graph, output_name="derived")
    with pytest.raises(ReplayGraphError, match="no output"):
        execute_replay_graph(empty_graph)


def test_replay_graph_file_script_input_and_rebuild_edges(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    file_spec = _file_spec(path)

    setup_code, load_code = _file_seed_code_parts(
        "import erlab\nimport numpy as np\nderived = xr.load_dataarray('source.nc')",
        "derived",
    )
    assert setup_code == "import numpy as np"
    assert "xr.load_dataarray" in load_code
    for seed_code, message in (
        ("derived =", "not valid Python"),
        ("other = xr.DataArray([1.0], dims=('x',))", "does not assign"),
    ):
        with pytest.raises(ReplayGraphError, match=message):
            _file_seed_code_parts(seed_code, "derived")

    loaded_input = ScriptInput(
        name="loaded",
        label="Loaded source",
        provenance_spec=file_spec,
    )
    code = script_inputs_code((loaded_input,), display=False)
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["loaded"], data)
    with pytest.raises(ReplayGraphError, match="recorded source"):
        script_inputs_code(
            (ScriptInput(name="missing", label="Missing source"),),
            display=False,
        )

    with pytest.raises(ReplayGraphError, match="script-derived"):
        rebuild_script_provenance(file_spec)
    script_spec = script(
        ScriptCodeOperation(label="Add one", code="derived = data_0 + 1.0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Loaded source",
                provenance_spec=file_spec,
            ),
        ),
    )
    rebuilt, rebuilt_spec = rebuild_script_provenance(script_spec)
    xr.testing.assert_identical(rebuilt, data + 1.0)
    assert rebuilt_spec.script_inputs[0].node_uid is None

    with pytest.raises(ReplayGraphError, match="maximum reload depth"):
        rebuild_script_provenance(script_spec, depth=21)
    missing_spec = script(
        ScriptCodeOperation(label="Add one", code="derived = data_0 + 1.0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Closed input"),),
    )
    with pytest.raises(ReplayGraphError, match="not open"):
        rebuild_script_provenance(missing_spec)

    live_calls = 0
    initial_marker = "initial-marker"
    current_marker = "current-marker"
    live_input = ScriptInput(
        name="data_0",
        label="Live input",
        node_uid="uid-0",
        node_snapshot_token=initial_marker,
    )
    live_spec = script(
        ScriptCodeOperation(label="Double", code="derived = data_0 * 2.0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(live_input,),
    )

    def resolve_live(_script_input):
        nonlocal live_calls
        live_calls += 1
        return data, live_input.model_copy(
            update={"node_snapshot_token": current_marker}
        )

    live_rebuilt, live_rebuilt_spec = rebuild_script_provenance(
        live_spec,
        live_input_resolver=resolve_live,
    )
    xr.testing.assert_identical(live_rebuilt, data * 2.0)
    assert live_calls == 1
    assert live_rebuilt_spec.script_inputs[0].node_snapshot_token == current_marker

    source_data = xr.full_like(data, 1.0)
    displayed_data = xr.full_like(data, 10.0)
    source_nested = script(
        ScriptCodeOperation(label="Use source data", code="derived = data_0"),
        start_label="Use nested source input",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Source role",
                node_uid="shared-node",
                data_role="source",
            ),
        ),
    )
    displayed_nested = script(
        ScriptCodeOperation(label="Use displayed data", code="derived = data_0"),
        start_label="Use nested displayed input",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Displayed role",
                node_uid="shared-node",
                data_role="displayed",
            ),
        ),
    )
    mixed_role_spec = script(
        ScriptCodeOperation(label="Add inputs", code="derived = left + right"),
        start_label="Combine nested inputs",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="left",
                label="Source branch",
                provenance_spec=source_nested,
            ),
            ScriptInput(
                name="right",
                label="Displayed branch",
                provenance_spec=displayed_nested,
            ),
        ),
    )

    def resolve_role(script_input):
        if script_input.node_uid != "shared-node":
            return None
        resolved_data = (
            source_data if script_input.data_role == "source" else displayed_data
        )
        return resolved_data, script_input.model_copy(
            update={"node_snapshot_token": f"current-{script_input.data_role}"}
        )

    mixed_role_result, mixed_role_rebuilt = rebuild_script_provenance(
        mixed_role_spec,
        live_input_resolver=resolve_role,
    )
    xr.testing.assert_identical(mixed_role_result, source_data + displayed_data)
    rebuilt_displayed = mixed_role_rebuilt.script_inputs[1].parsed_provenance_spec()
    assert rebuilt_displayed is not None
    assert rebuilt_displayed.script_inputs[0].data_role == "displayed"

    miss_calls = 0
    duplicate_file_input = ScriptInput(
        name="data_0",
        label="Closed file input",
        node_uid="same-uid",
        provenance_spec=file_spec,
    )
    duplicate_file_spec = script(
        ScriptCodeOperation(label="Copy", code="derived = data_0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(duplicate_file_input, duplicate_file_input),
    )

    def miss_live(_script_input):
        nonlocal miss_calls
        miss_calls += 1
        return

    rebuilt_from_miss, _rebuilt_from_miss_spec = rebuild_script_provenance(
        duplicate_file_spec,
        live_input_resolver=miss_live,
    )
    xr.testing.assert_identical(rebuilt_from_miss, data)
    assert miss_calls == 2

    unsupported_nested = script(
        ScriptCodeOperation(label="Opaque", code=None, copyable=False),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    unsupported_input = ScriptInput(
        name="data_0",
        label="Unsupported nested input",
        provenance_spec=unsupported_nested,
    )
    unsupported_spec = script(
        ScriptCodeOperation(label="Copy", code="derived = data_0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(unsupported_input,),
    )
    with pytest.raises(ReplayGraphError, match="cannot be replayed"):
        rebuild_script_provenance(unsupported_spec)

    full_data_spec = script(
        ScriptCodeOperation(label="Copy", code="derived = data_0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Full data",
                provenance_spec=full_data(),
            ),
        ),
    )
    with pytest.raises(ReplayGraphError, match="reloadable"):
        rebuild_script_provenance(full_data_spec)


def test_replay_graph_operation_code_error_edges() -> None:
    class MissingReplayOperation:
        pass

    class MissingExpressionOperation:
        def replay_code(
            self,
            input_name: str,
            *,
            output_name: str | None = None,
            source_name: str | None = None,
            reserved_names: Collection[str] = (),
        ) -> str:
            raise NotImplementedError

    class Operation:
        def __init__(self, code: str | None) -> None:
            self._code = code

        def replay_code(
            self,
            input_name: str,
            *,
            output_name: str | None = None,
            source_name: str | None = None,
            reserved_names: Collection[str] = (),
        ) -> str | None:
            return self._code

    for operation, message in (
        (MissingReplayOperation(), "does not provide"),
        (MissingExpressionOperation(), "does not provide"),
        (Operation(None), "does not provide"),
        (Operation("derived ="), "not valid Python"),
        (Operation("other = data"), "does not assign"),
    ):
        with pytest.raises(ReplayGraphError, match=message):
            _operation_replay_code(
                operation,
                active_name="derived",
                context_name="data",
            )


def test_replay_graph_operation_code_uses_parameterized_names() -> None:
    class Operation:
        def replay_code(
            self,
            input_name: str,
            *,
            output_name: str | None = None,
            source_name: str | None = None,
            reserved_names: Collection[str] = (),
        ) -> str:
            assert input_name == "parent_data"
            assert output_name == "active_data"
            assert source_name == "source_data"
            assert not reserved_names
            return f"{output_name} = {input_name} + {source_name}"

    code = _operation_replay_code(
        Operation(),
        active_name="active_data",
        context_name="source_data",
        parent_name="parent_data",
    )

    assert code == "active_data = parent_data + source_data"


def test_replay_graph_emits_shared_file_and_operation_prefix(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "polarization.nc"
    source = _polarization_source(path)
    file_spec = _file_spec(path)
    shared_stage = full_data(AverageOperation(dims=("k",)))
    left_stage = selection(
        SelOperation(kwargs={"pol": "LH"}),
        SqueezeOperation(),
    )
    right_stage = selection(
        SelOperation(kwargs={"pol": "LV"}),
        SqueezeOperation(),
    )
    left_spec = compose_full_provenance(
        compose_full_provenance(file_spec, shared_stage),
        left_stage,
    )
    right_spec = compose_full_provenance(
        compose_full_provenance(file_spec, shared_stage),
        right_stage,
    )
    assert left_spec is not None
    assert right_spec is not None
    spec = script(
        ScriptCodeOperation(label="Subtract", code="derived = data_0 - data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="LH", provenance_spec=left_spec),
            ScriptInput(name="data_1", label="LV", provenance_spec=right_spec),
        ),
    )

    code = typing.cast("str", spec.display_code())

    assert code.count("xr.load_dataarray") == 1
    assert code.count(".qsel.mean") == 1
    namespace = _exec_generated_code(code)
    expected = left_stage.apply(shared_stage.apply(source)) - right_stage.apply(
        shared_stage.apply(source)
    )
    xr.testing.assert_identical(namespace["derived"], expected)


@pytest.mark.parametrize(
    "input_names",
    [("source_a", "source_b"), ("data_0", "data_1")],
)
def test_replay_graph_emits_one_readable_binding_for_shared_inputs(
    tmp_path: pathlib.Path,
    input_names: tuple[str, str],
) -> None:
    path = tmp_path / "shared.nc"
    source = xr.DataArray(np.arange(4.0), dims="x")
    source.to_netcdf(path)
    source_spec = _file_spec(path)
    first_name, second_name = input_names
    spec = script(
        ScriptCodeOperation(
            label="Add selected values",
            code=(f"result = {first_name}.isel(x=0) + {second_name}.isel(x=1)"),
        ),
        start_label="Run script",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name=first_name,
                label="First",
                provenance_spec=source_spec,
            ),
            ScriptInput(
                name=second_name,
                label="Second",
                provenance_spec=source_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())

    assert code.count("xr.load_dataarray") == 1
    assert code.count(".copy(deep=True)") == 2
    assert "_itool_replay_" not in code
    assert not any(
        isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Name)
        for statement in ast.parse(code).body
    )
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(
        namespace["result"], source.isel(x=0) + source.isel(x=1)
    )


def test_replay_graph_binds_distinct_structured_inputs_directly(
    tmp_path: pathlib.Path,
) -> None:
    left_path = tmp_path / "left.nc"
    right_path = tmp_path / "right.nc"
    left_source = xr.DataArray(np.arange(4.0), dims="x")
    right_source = xr.DataArray(np.arange(4.0) + 10.0, dims="x")
    left_source.to_netcdf(left_path)
    right_source.to_netcdf(right_path)
    left_spec = compose_full_provenance(
        _file_spec(left_path),
        full_data(IselOperation(kwargs={"x": slice(None)})),
    )
    right_spec = compose_full_provenance(
        _file_spec(right_path),
        full_data(IselOperation(kwargs={"x": slice(None)})),
    )
    assert left_spec is not None
    assert right_spec is not None
    spec = script(
        ScriptCodeOperation(label="Add sources", code="result = left + right"),
        start_label="Run script",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left", label="Left", provenance_spec=left_spec),
            ScriptInput(name="right", label="Right", provenance_spec=right_spec),
        ),
    )

    code = typing.cast("str", spec.display_code())

    assert code.count("xr.load_dataarray") == 2
    assert ".copy(deep=True)" not in code
    assert "loaded_data" not in code
    assert "processed_data" not in code
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["result"], left_source + right_source)


@pytest.mark.parametrize(
    "mutation_code",
    [
        "source_a = source_a + 10.0",
        "source_a.values[:] += 10.0",
        "alias = source_a\nalias.values[:] += 10.0",
        "source_a.copy(deep=False).values[:] += 10.0",
        ("alias = source_a if True else source_b\nalias.values[:] += 10.0"),
    ],
)
def test_replay_graph_preserves_shared_script_input_ownership(
    tmp_path: pathlib.Path,
    mutation_code: str,
) -> None:
    path = tmp_path / "shared.nc"
    source = xr.DataArray(np.arange(4.0), dims="x")
    source.to_netcdf(path)
    source_spec = _file_spec(path)
    spec = script(
        ScriptCodeOperation(
            label="Change one input",
            code=f"{mutation_code}\nresult = source_a + source_b",
        ),
        start_label="Run script",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="source_a",
                label="First",
                provenance_spec=source_spec,
            ),
            ScriptInput(
                name="source_b",
                label="Second",
                provenance_spec=source_spec,
            ),
        ),
    )

    expected = replay_script_provenance(spec, {}, trusted_user_code=True)
    code = typing.cast("str", spec.display_code())

    assert code.count("xr.load_dataarray") == 1
    assert code.count(".copy(deep=True)") == 2
    assert "_itool_replay_" not in code
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["result"], expected)
    xr.testing.assert_identical(expected, source * 2.0 + 10.0)


def test_replay_graph_isolates_mutation_across_shared_source_views(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "shared.nc"
    source = xr.DataArray(np.arange(4.0), dims="x")
    source.to_netcdf(path)
    source_spec = _file_spec(path)
    first_view = compose_full_provenance(
        source_spec,
        full_data(IselOperation(kwargs={"x": slice(None)})),
    )
    second_view = compose_full_provenance(
        source_spec,
        full_data(IselOperation(kwargs={"x": slice(0, None)})),
    )
    assert first_view is not None
    assert second_view is not None
    spec = script(
        ScriptCodeOperation(
            label="Change one view",
            code="source_a.values[:] += 10.0\nresult = source_a + source_b",
        ),
        start_label="Run script",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="source_a",
                label="First view",
                provenance_spec=first_view,
            ),
            ScriptInput(
                name="source_b",
                label="Second view",
                provenance_spec=second_view,
            ),
        ),
    )

    expected = replay_script_provenance(spec, {}, trusted_user_code=True)
    code = typing.cast("str", spec.display_code())

    assert code.count("xr.load_dataarray") == 1
    assert code.count(".copy(deep=True)") == 2
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["result"], expected)
    xr.testing.assert_identical(expected, source * 2.0 + 10.0)


def test_replay_graph_preserves_shared_script_input_identity(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "shared.nc"
    source = xr.DataArray(np.arange(4.0), dims="x")
    source.to_netcdf(path)
    source_spec = _file_spec(path)
    spec = script(
        ScriptCodeOperation(
            label="Observe input identity",
            code=("result = source_a + (100.0 if source_a is source_b else 0.0)"),
        ),
        start_label="Run script",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="source_a",
                label="First",
                provenance_spec=source_spec,
            ),
            ScriptInput(
                name="source_b",
                label="Second",
                provenance_spec=source_spec,
            ),
        ),
    )

    expected = replay_script_provenance(spec, {}, trusted_user_code=True)
    code = typing.cast("str", spec.display_code())

    assert code.count("xr.load_dataarray") == 1
    assert code.count(".copy(deep=True)") == 2
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["result"], expected)
    xr.testing.assert_identical(expected, source)


def test_replay_graph_replays_script_with_preserved_file_steps(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    source = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": [0, 1, 2], "y": [10, 20, 30, 40]},
        name="scan",
    )
    source.to_netcdf(path)
    file_spec = _file_spec(path).append_replay_stage(
        full_data(AverageOperation(dims=("x",)))
    )
    local = script(
        ScriptCodeOperation(
            label="Center profile",
            code="result = derived - derived.mean()",
        ),
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
    )

    spec = compose_full_provenance(file_spec, local)
    assert spec is not None
    assert spec.kind == "script"
    assert any(isinstance(step.operation, AverageOperation) for step in spec.steps)

    replayed = replay_script_provenance(spec, {})

    expected_input = AverageOperation(dims=("x",)).apply(source)
    xr.testing.assert_identical(replayed, expected_input - expected_input.mean())
    code = typing.cast("str", spec.display_code())
    assert code.count("xr.load_dataarray") == 1
    assert "result =" in code
    xr.testing.assert_identical(_exec_generated_code(code)["result"], replayed)


def test_replay_graph_display_code_preserves_script_mutation_order() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    spec = script(
        ScriptCodeOperation(label="Copy input", code="derived = data.copy()"),
        ScriptCodeOperation(
            label="Mutate input",
            code="data.values[:] = 10.0",
        ),
        ScriptCodeOperation(
            label="Combine arrays",
            code="derived = derived + data",
        ),
        start_label="Run script",
        active_name="derived",
    )

    replayed = replay_script_provenance(spec, {"data": data})
    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})

    xr.testing.assert_identical(namespace["derived"], replayed)


def test_replay_graph_applies_consecutive_restored_steps_as_one_chain() -> None:
    data = xr.DataArray(
        np.arange(20.0).reshape(4, 5),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 0.7, 1.5], "y": np.arange(5)},
    )
    source = selection(
        IselOperation(kwargs={"x": slice(1, None)}),
        SelOperation(kwargs={"y": slice(1, 3)}),
    )
    replay_spec = source.to_replay_spec()

    assert [step.input_policy for step in replay_spec.steps] == [
        "restored",
        "restored",
    ]
    xr.testing.assert_identical(
        replay_script_provenance(replay_spec, {"data": data}),
        source.apply(data),
    )


def test_replay_graph_composes_local_script_stage_after_script_parent() -> None:
    source = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": [0, 1, 2], "y": [10, 20, 30, 40]},
        name="scan",
    )
    parent = script(
        ScriptCodeOperation(
            label="Crop source",
            code="derived = derived.isel(x=slice(0, 2))",
        ),
        start_label="Run parent script",
        seed_code="derived = data",
        active_name="derived",
    )
    local = script(
        ScriptCodeOperation(
            label="Offset profile",
            code="result = derived + 1",
        ),
        start_label="Run local script",
        active_name="result",
        replay_stages=(
            ReplayStage.from_source_spec(full_data(AverageOperation(dims=("x",)))),
        ),
    )

    spec = compose_full_provenance(parent, local)
    assert spec is not None
    assert spec.kind == "script"
    assert [operation.op for operation in spec.operations] == [
        "script_code",
        "average",
        "script_code",
    ]

    replayed = replay_script_provenance(spec, {"data": source})

    expected = source.isel(x=slice(0, 2)).qsel.mean(("x",)) + 1
    xr.testing.assert_identical(replayed, expected)
    code = typing.cast("str", spec.display_code())
    assert code.startswith("result =")
    assert code.index(".isel(") < code.index(".qsel.mean(")
    xr.testing.assert_identical(
        _exec_generated_code(code, {"data": source})["result"],
        replayed,
    )


def test_replay_graph_dedupes_matching_script_file_seed(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    source = xr.DataArray(np.arange(4.0), dims=("x",), name="scan")
    source.to_netcdf(path)
    load_source = _file_replay_source(
        path,
        load_code=f"data = xr.load_dataarray({str(path)!r})",
    )
    file_spec = file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=load_source,
    )
    center_spec = script(
        ScriptCodeOperation(
            label="Extract center values",
            code="center_values = derived.mean('x')",
        ),
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        active_name="center_values",
        file_load_source=load_source,
    )
    corrected_spec = script(
        ScriptCodeOperation(
            label="Correct with center values",
            code="derived = data_0 - data_1",
        ),
        RenameOperation(name="corrected"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="ImageTool 4: scan",
                provenance_spec=file_spec,
            ),
            ScriptInput(
                name="data_1",
                label="ImageTool 4.0: center_values",
                provenance_spec=center_spec,
            ),
        ),
    )

    code = typing.cast("str", corrected_spec.display_code())
    namespace = _exec_generated_code(code)

    assert code.count("xr.load_dataarray") == 1
    assert ".rename('corrected')" in code
    xr.testing.assert_identical(
        namespace["derived"],
        (source - source.mean()).rename("corrected"),
    )


def test_replay_graph_script_seed_file_load_parts_rejects_mismatches(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    load_source = _file_replay_source(
        path,
        load_code=f"data = xr.load_dataarray({str(path)!r})",
    )
    seed_code = f"derived = xr.load_dataarray({str(path)!r})"

    assert (
        _script_seed_file_load_parts(
            "print(data)",
            active_name="derived",
            load_source=load_source,
        )
        is None
    )
    assert (
        _script_seed_file_load_parts(
            "derived =",
            active_name="derived",
            load_source=load_source,
        )
        is None
    )
    assert (
        _script_seed_file_load_parts(
            seed_code,
            active_name="derived",
            load_source=load_source.model_copy(update={"load_code": None}),
        )
        is None
    )
    assert (
        _script_seed_file_load_parts(
            seed_code,
            active_name="derived",
            load_source=load_source.model_copy(
                update={"load_code": "data.value = xr.load_dataarray('scan.nc')"}
            ),
        )
        is None
    )
    assert (
        _script_seed_file_load_parts(
            seed_code,
            active_name="derived",
            load_source=load_source.model_copy(
                update={"load_code": "setup = 1\ndata = xr.load_dataarray('scan.nc')"}
            ),
        )
        is None
    )


@pytest.mark.parametrize(
    "updated_load_source",
    [
        lambda path, other_path, source: source.model_copy(
            update={
                "path": str(other_path),
                "load_code": f"data = xr.load_dataarray({str(other_path)!r})",
            }
        ),
        lambda _path, _other_path, source: source.model_copy(
            update={"kwargs_text": "engine='h5netcdf'"}
        ),
        lambda _path, _other_path, source: source.model_copy(
            update={
                "replay_call": source.replay_call.model_copy(
                    update={
                        "selection": FileDataSelection(
                            kind="parsed_index",
                            value=1,
                        )
                    }
                )
            }
        ),
    ],
)
def test_replay_graph_keeps_distinct_file_load_sources_separate(
    tmp_path: pathlib.Path,
    updated_load_source,
) -> None:
    path = tmp_path / "scan.nc"
    other_path = tmp_path / "other.nc"
    source = xr.DataArray(np.arange(4.0), dims=("x",), name="scan")
    source.to_netcdf(path)
    (source + 10.0).to_netcdf(other_path)
    load_source = _file_replay_source(
        path,
        load_code=f"data = xr.load_dataarray({str(path)!r})",
    )
    other_load_source = updated_load_source(path, other_path, load_source)
    first_spec = file_load(
        start_label="Load first",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=load_source,
    )
    second_spec = script(
        ScriptCodeOperation(label="Copy", code="derived = derived"),
        start_label="Load second",
        seed_code=typing.cast("str", other_load_source.load_code).replace(
            "data =",
            "derived =",
            1,
        ),
        active_name="derived",
        file_load_source=other_load_source,
    )
    spec = script(
        ScriptCodeOperation(label="Add", code="derived = data_0 + data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                provenance_spec=second_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())

    assert code.count("xr.load_dataarray") == 2


def test_replay_graph_does_not_normalize_mismatched_script_file_seed(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    other_path = tmp_path / "other.nc"
    source = xr.DataArray(np.arange(4.0), dims=("x",), name="scan")
    source.to_netcdf(path)
    (source + 10.0).to_netcdf(other_path)
    load_source = _file_replay_source(
        path,
        load_code=f"data = xr.load_dataarray({str(path)!r})",
    )
    first_spec = file_load(
        start_label="Load first",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=load_source,
    )
    mismatched_script_spec = script(
        ScriptCodeOperation(label="Copy", code="derived = derived"),
        start_label="Load mismatched",
        seed_code=f"derived = xr.load_dataarray({str(other_path)!r})",
        active_name="derived",
        file_load_source=load_source,
    )
    spec = script(
        ScriptCodeOperation(label="Add", code="derived = data_0 + data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            ScriptInput(
                name="data_1",
                label="Mismatched",
                provenance_spec=mismatched_script_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())

    assert code.count("xr.load_dataarray") == 2


def test_replay_graph_handles_structured_script_operations(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("alpha", "eV"),
        coords={"alpha": [0.0, 1.0], "eV": [0.0, 1.0, 2.0]},
    )
    data.to_netcdf(path)
    spec = script(
        AverageOperation(dims=("alpha",)),
        start_label="Run script",
        seed_code="avg = data_0",
        active_name="avg",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Scan",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    assert ".qsel.mean" in code
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["avg"], data.qsel.mean("alpha"))

    graph = compile_replay_graph(spec)
    xr.testing.assert_identical(
        execute_replay_graph(graph),
        data.qsel.mean("alpha"),
    )


def test_replay_graph_emits_structured_script_operation_without_identity_relays(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("polarization", "eV"),
        coords={"polarization": [-1, 1], "eV": [0.0, 1.0]},
    )
    data.to_netcdf(path)
    spec = script(
        SelOperation(kwargs={"polarization": -1}),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Scan",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    lines = code.splitlines()

    assert not any(line.startswith("_itool_replay_") for line in lines)
    assert "derived = xr.load_dataarray" in code
    assert ".sel(polarization=-1)" in code
    xr.testing.assert_identical(
        _exec_generated_code(code)["derived"],
        data.sel(polarization=-1),
    )


def test_replay_graph_preserves_script_inputs_after_structured_operation() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = script(
        AverageOperation(dims=("x",)),
        ScriptCodeOperation(
            label="Use original input",
            code="derived = derived + data_0.qsel.average('x')",
        ),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )

    graph = compile_replay_graph(spec, external_inputs={"data_0": data})
    script_nodes = [node for node in graph.nodes if node.kind == "script"]

    assert script_nodes
    assert any(
        input_name == "data_0"
        for node in script_nodes
        for input_name, _key in node.payload["bindings"]
    )
    xr.testing.assert_identical(
        execute_replay_graph(graph),
        data.qsel.average("x") + data.qsel.average("x"),
    )


def test_replay_graph_keeps_structured_operations_in_opaque_script() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = script(
        AverageOperation(dims=("x",)),
        ScriptCodeOperation(
            label="Use temp",
            code="derived = derived + tmp.qsel.average('x') + data_0.qsel.average('x')",
        ),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )

    graph = compile_replay_graph(spec, external_inputs={"data_0": data})
    script_nodes = [node for node in graph.nodes if node.kind == "script"]

    assert len(script_nodes) == 1
    assert not any(node.kind == "operation" for node in graph.nodes)
    assert any(
        "derived = derived.qsel.mean(" in code
        for code in script_nodes[0].payload["codes"]
    )
    xr.testing.assert_identical(
        execute_replay_graph(graph),
        (data + 1).qsel.average("x")
        + (data + 1).qsel.average("x")
        + data.qsel.average("x"),
    )


def test_replay_graph_omits_cosmetic_coordinate_sort_operation() -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
    )
    spec = script(
        SortCoordOrderOperation(),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )

    graph = compile_replay_graph(spec, external_inputs={"data_0": data})
    script_nodes = [node for node in graph.nodes if node.kind == "script"]
    inlined_code = "\n".join(script_nodes[0].payload["codes"])

    assert script_provenance_replayable(spec)
    assert spec.operations == ()
    assert "sort_coord_order" not in inlined_code
    xr.testing.assert_identical(
        replay_script_provenance(spec, {"data_0": data}),
        data + 1,
    )


def test_replay_graph_omits_cosmetic_coordinate_sort_from_script_input_code(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
    )
    data.to_netcdf(path)
    spec = script(
        SortCoordOrderOperation(),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    graph = compile_replay_graph(spec, display=False)
    code = emit_replay_code(graph, output_name="derived")

    assert "sort_coord_order" not in code
    assert "derived = tmp" not in code
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["derived"], data + 1)


def test_replay_graph_shares_structured_console_alias_prefixes(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "polarization.nc"
    source = _polarization_source(path)
    averaged = script(
        AverageOperation(dims=("k",)),
        start_label="Run script",
        seed_code="avg = data_0",
        active_name="avg",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Scan",
                provenance_spec=_file_spec(path),
            ),
        ),
    )
    spec = script(
        ScriptCodeOperation(label="Subtract", code="derived = data_0 - data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="LH", provenance_spec=averaged),
            ScriptInput(name="data_1", label="LV", provenance_spec=averaged),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    graph = compile_replay_graph(spec)

    assert code.count("xr.load_dataarray") == 1
    assert code.count(".qsel.mean") == 1
    assert code.count(".copy(deep=True)") == 2
    assert (
        sum(
            node.kind == "operation" and node.payload["operation"].op == "average"
            for node in graph.nodes
        )
        == 1
    )
    expected = source.qsel.mean("k") - source.qsel.mean("k")
    xr.testing.assert_identical(_exec_generated_code(code)["derived"], expected)
    xr.testing.assert_identical(execute_replay_graph(graph), expected)


def test_replay_graph_display_normalizes_nested_derived_console_code(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "cd_map.nc"
    source = xr.DataArray(
        np.arange(2 * 6 * 10, dtype=float).reshape(2, 6, 10) + 1.0,
        dims=("polarization", "alpha", "eV"),
        coords={
            "polarization": [-1, 1],
            "alpha": np.linspace(-1.0, 1.0, 6),
            "eV": np.linspace(-0.5, 0.5, 10),
            "mesh_current": (
                ("alpha", "eV"),
                np.linspace(1.0, 2.0, 60).reshape(6, 10),
            ),
        },
    )
    source.to_netcdf(path)
    processed = compose_full_provenance(
        _file_spec(path),
        public_data(
            DivideByCoordOperation(coord_name="mesh_current"),
            CoarsenOperation(
                dim={"alpha": 3, "eV": 5},
                boundary="trim",
                side="left",
                coord_func="mean",
                reducer="mean",
            ),
        ),
    )
    assert processed is not None

    rc = script(
        SelOperation(kwargs={"polarization": -1}),
        start_label="Run ImageTool manager console code",
        seed_code="rc = data_0",
        active_name="rc",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Processed map",
                provenance_spec=processed,
            ),
        ),
    )
    lc = script(
        SelOperation(kwargs={"polarization": 1}),
        start_label="Run ImageTool manager console code",
        seed_code="lc = data_0",
        active_name="lc",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Processed map",
                provenance_spec=processed,
            ),
        ),
    )
    diff = script(
        ScriptCodeOperation(
            label="Evaluate console expression",
            code="derived = rc - lc",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="rc", label="console variable 'rc'", provenance_spec=rc),
            ScriptInput(name="lc", label="console variable 'lc'", provenance_spec=lc),
        ),
    )
    total = script(
        ScriptCodeOperation(
            label="Evaluate console expression",
            code="derived = rc + lc",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="rc", label="console variable 'rc'", provenance_spec=rc),
            ScriptInput(name="lc", label="console variable 'lc'", provenance_spec=lc),
        ),
    )
    ncd = script(
        ScriptCodeOperation(
            label="Evaluate console expression",
            code="ncd = data_1 / data_2",
        ),
        start_label="Run ImageTool manager console code",
        active_name="ncd",
        script_inputs=(
            ScriptInput(name="data_1", label="ImageTool 1", provenance_spec=diff),
            ScriptInput(name="data_2", label="ImageTool 2", provenance_spec=total),
        ),
    )

    code = typing.cast("str", ncd.display_code())
    namespace = _exec_generated_code(code)
    processed_data = (
        (source / source.mesh_current).coarsen(alpha=3, eV=5, boundary="trim").mean()
    )
    expected = (
        processed_data.sel(polarization=-1) - processed_data.sel(polarization=1)
    ) / (processed_data.sel(polarization=-1) + processed_data.sel(polarization=1))

    assert code.count("xr.load_dataarray") == 1
    assert "restore_nonuniform_dims" not in code
    assert len(re.findall(r"^rc =", code, flags=re.MULTILINE)) == 1
    assert len(re.findall(r"^lc =", code, flags=re.MULTILINE)) == 1
    assert len(re.findall(r"^data_1 =", code, flags=re.MULTILINE)) == 1
    assert len(re.findall(r"^data_2 =", code, flags=re.MULTILINE)) == 1
    assert "derived" not in code
    assert "ncd = data_1 / data_2" in code
    assigned_names = [
        statement.targets[0].id
        for statement in ast.parse(code).body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ]
    assert len(assigned_names) == len(set(assigned_names))
    _assert_dense_replay_temps(code)
    xr.testing.assert_identical(namespace["ncd"], expected)


def test_replay_graph_cleanup_helpers_cover_edge_cases() -> None:
    assert _compact_replay_temp_names("bad =") == "bad ="
    assert _code_has_scoped_definition("bad =")
    effectful_attributes = (
        "_itool_replay_0 = source.first\n"
        "other = source.second\n"
        "result = _itool_replay_0 + other"
    )
    assert _inline_single_use_replay_names(effectful_attributes) == effectful_attributes
    shadowed_temporary = (
        "_itool_replay_0 = source\n_itool_replay_0 = other\nresult = _itool_replay_0"
    )
    cleaned_shadowed_temporary = _inline_single_use_replay_names(shadowed_temporary)
    namespace = {"source": object(), "other": object()}
    exec(cleaned_shadowed_temporary, {}, namespace)  # noqa: S102
    assert namespace["result"] is namespace["other"]

    assert (
        _compact_replay_temp_names(
            "_itool_replay_4 = data\n_itool_replay_8 = _itool_replay_4 + 1"
        )
        == "_itool_replay_0 = data\n_itool_replay_1 = _itool_replay_0 + 1"
    )
    assert (
        _compact_replay_temp_names(
            "_itool_replay_0 = 'reserved'\n"
            "_itool_replay_4 = data\n"
            "result = _itool_replay_4"
        )
        == "_itool_replay_0 = 'reserved'\n"
        "_itool_replay_1 = data\n"
        "result = _itool_replay_1"
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            "def  _itool_replay_1():\n    return 1\nresult = _itool_replay_1()",
            1,
        ),
        (
            "class   _itool_replay_1:\n    value = 1\nresult = _itool_replay_1.value",
            1,
        ),
    ],
)
def test_replace_ast_names_preserves_definition_whitespace(
    code: str,
    expected: int,
) -> None:
    renamed = _replace_ast_names(
        code,
        ast.parse(code),
        {"_itool_replay_1": "script_result"},
    )
    namespace = _exec_generated_code(renamed)

    assert namespace["result"] == expected

    async_code = "async  def _itool_replay_1():\n    return 1\nresult = _itool_replay_1"
    renamed_async = _replace_ast_names(
        async_code,
        ast.parse(async_code),
        {"_itool_replay_1": "script_result"},
    )
    async_namespace = _exec_generated_code(renamed_async)
    assert async_namespace["result"].__name__ == "script_result"


def test_replace_code_identifiers_keeps_required_dotted_import_relay() -> None:
    code = (
        "import numpy.random\n\n"
        "def scalar():\n"
        "    return numpy.float64(1)\n\n"
        "numpy = scalar()"
    )

    renamed = _replace_code_identifiers(code, {"numpy": "result"})
    namespace = _exec_generated_code(renamed)

    assert namespace["result"] == np.float64(1)


@pytest.mark.parametrize(
    "body",
    [
        "if False:\n    numpy = 0\nresult = numpy.float64(1)",
        "numpy: object\nresult = numpy.float64(1)",
        "del numpy\nresult = 1",
    ],
)
def test_replace_code_identifiers_keeps_observable_dotted_import_relay(
    body: str,
) -> None:
    renamed = _replace_code_identifiers(
        f"import numpy.random\n{body}",
        {"numpy": "script_result"},
    )

    namespace = _exec_generated_code(renamed)

    assert namespace["result"] == 1


def test_replace_code_identifiers_preserves_renamed_dotted_import_binding() -> None:
    renamed = _replace_code_identifiers(
        "import numpy.random",
        {"numpy": "result"},
    )

    namespace = _exec_generated_code(renamed)

    assert namespace["result"] is np


def test_replace_code_identifiers_flattens_import_relay_in_function() -> None:
    code = (
        "def assign_result():\n"
        "    global numpy\n"
        "    import numpy.random\n"
        "    numpy = numpy.float64(1)\n\n"
        "assign_result()"
    )

    renamed = _replace_code_identifiers(code, {"numpy": "result"})
    namespace = _exec_generated_code(renamed)

    assert namespace["result"] == np.float64(1)


def test_replay_graph_emit_reports_script_rewrite_syntax_errors(monkeypatch) -> None:

    graph = ReplayGraph(display=True)
    source_key = graph.add_node(
        "source",
        "file_load",
        payload={
            "active_name": "loaded",
            "load_code": "loaded = data",
            "load_source": _file_replay_source("source.nc"),
        },
    )
    graph.output_key = graph.add_node(
        "script",
        "script",
        parents=(source_key,),
        payload={
            "codes": ("result = data_0",),
            "active_name": "result",
            "bindings": (("data_0", source_key),),
        },
    )

    original_replace = _replace_code_identifiers

    def _raise_on_input_replacement(code: str, replacements: Mapping[str, str]) -> str:
        if "data_0" in replacements:
            raise SyntaxError("bad script input")
        return original_replace(code, replacements)

    monkeypatch.setattr(
        _graph,
        "_replace_code_identifiers",
        _raise_on_input_replacement,
    )
    with pytest.raises(ReplayGraphError, match="Script replay code"):
        emit_replay_code(graph)

    graph = ReplayGraph(display=True)
    graph.output_key = graph.add_node(
        "script",
        "script",
        payload={
            "codes": ("bad =",),
            "active_name": "derived",
            "bindings": (),
        },
    )
    with pytest.raises(ReplayGraphError, match="Script replay code"):
        emit_replay_code(graph, output_name="result")


def test_replay_graph_display_promotes_ui_style_script_input_names(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    source = _polarization_source(path)
    left = compose_full_provenance(
        _file_spec(path),
        full_data(SelOperation(kwargs={"pol": "LH"})),
    )
    right = compose_full_provenance(
        _file_spec(path),
        full_data(SelOperation(kwargs={"pol": "LV"})),
    )
    assert left is not None
    assert right is not None
    spec = script(
        ScriptCodeOperation(
            label="Concatenate selected inputs",
            code="combined = xr.concat([left, right], dim='pol')",
        ),
        start_label="Run ImageTool manager UI action",
        active_name="combined",
        script_inputs=(
            ScriptInput(name="left", label="Selected left", provenance_spec=left),
            ScriptInput(name="right", label="Selected right", provenance_spec=right),
        ),
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code)

    assert code.count("xr.load_dataarray") == 1
    assert "data_0 =" not in code
    assert "data_1 =" not in code
    assert len(re.findall(r"^left =", code, flags=re.MULTILINE)) == 1
    assert len(re.findall(r"^right =", code, flags=re.MULTILINE)) == 1
    assert "combined = xr.concat([left, right], dim='pol')" in code
    _assert_dense_replay_temps(code)
    xr.testing.assert_identical(
        namespace["combined"],
        xr.concat(
            [source.sel(pol="LH"), source.sel(pol="LV")],
            dim="pol",
        ),
    )


def test_replay_graph_display_uses_watched_roots_as_raw_inputs() -> None:
    rc_data = xr.DataArray(np.arange(3.0), dims=("x",))
    lc_data = xr.DataArray(np.arange(3.0) + 10.0, dims=("x",))
    rc = script(
        start_label="Start from watched variable 'rc'",
        seed_code="derived = rc",
        active_name="derived",
    )
    lc = script(
        start_label="Start from watched variable 'lc'",
        seed_code="derived = lc",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(
            label="Subtract selected inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager action",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="ImageTool 0", provenance_spec=rc),
            ScriptInput(name="data_1", label="ImageTool 1", provenance_spec=lc),
        ),
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"rc": rc_data, "lc": lc_data})

    assert code == "derived = rc - lc"
    xr.testing.assert_identical(namespace["derived"], rc_data - lc_data)


def test_replay_graph_display_keeps_helpers_from_raw_seed_inputs() -> None:
    raw_data = xr.DataArray(np.arange(1.0, 4.0), dims=("x",))
    root = script(
        start_label="Start from watched variable 'raw'",
        seed_code="def normalize():\n    return raw / raw.max()\nderived = normalize()",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(
            label="Average normalized input",
            code="result = data_0.mean()",
        ),
        start_label="Run ImageTool manager action",
        active_name="result",
        script_inputs=(
            ScriptInput(name="data_0", label="ImageTool 0", provenance_spec=root),
        ),
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"raw": raw_data})

    assert "def normalize():" in code
    assert "raw / raw.max()" in code
    xr.testing.assert_identical(namespace["result"], (raw_data / raw_data.max()).mean())


def test_replay_graph_display_hides_internal_source_view_restore_only(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "nonuniform.nc"
    source = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 1.0], "y": [0.0, 1.0]},
    )
    source.to_netcdf(path)
    public_spec = compose_full_provenance(
        _file_spec(path),
        public_data(SelOperation(kwargs={"x": 0.2})),
    )
    assert public_spec is not None
    script_input = ScriptInput(
        name="selected",
        label="Selected nonuniform data",
        provenance_spec=public_spec,
    )

    replay_code = script_inputs_code((script_input,), display=False)
    display_code = script_inputs_code((script_input,), display=True)

    assert "restore_nonuniform_dims" not in replay_code
    assert "restore_nonuniform_dims" not in display_code
    xr.testing.assert_identical(
        _exec_generated_code(replay_code)["selected"],
        source.sel(x=0.2),
    )
    xr.testing.assert_identical(
        _exec_generated_code(display_code)["selected"],
        source.sel(x=0.2),
    )


def test_replay_graph_display_restores_dimensions_left_by_recorded_mapping(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "internal.nc"
    source = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x_idx", "y_idx"),
        coords={
            "x": ("x_idx", [0.0, 0.2, 1.0]),
            "y": ("y_idx", [0.0, 0.4, 1.5, 3.0]),
        },
    )
    source.to_netcdf(path)
    spec = script(
        start_label="Load ImageTool rendering data",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        active_name="derived",
        replay_stages=(
            ReplayStage(
                source_kind="full_data",
                operations=(
                    RestoreNonuniformDimsOperation(dimension_mapping={"x_idx": "x"}),
                ),
            ),
            ReplayStage(source_kind="public_data"),
        ),
    )

    runtime_graph = compile_replay_graph(spec)
    display_graph = compile_replay_graph(spec, display=True)
    display_code = emit_replay_code(display_graph, output_name="derived")
    expected = source.swap_dims({"x_idx": "x", "y_idx": "y"}).drop_vars(
        ("x_idx", "y_idx"), errors="ignore"
    )

    assert display_code.count("def _restore_image_tool_dimensions") == 1
    xr.testing.assert_identical(
        execute_replay_graph(runtime_graph),
        expected,
    )
    xr.testing.assert_identical(
        _exec_generated_code(display_code)["derived"],
        expected,
    )


@pytest.mark.parametrize("source_kind", ["public_data", "selection"])
@pytest.mark.parametrize("display", [False, True])
def test_replay_graph_restores_script_seeded_internal_source_views(
    source_kind: str,
    display: bool,
    tmp_path: pathlib.Path,
) -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 1.0], "y": [0.0, 1.0]},
    )
    path = tmp_path / "source.nc"
    source.to_netcdf(path)
    source_spec = {"public_data": public_data, "selection": selection}[source_kind](
        QSelOperation(kwargs={"x": 0.2})
    )
    spec = script(
        start_label="Create ImageTool rendering dimensions",
        seed_code="derived = erlab.utils.array._make_dims_uniform(data_0)",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Source data",
                provenance_spec=_file_spec(path),
            ),
        ),
        replay_stages=(ReplayStage.from_source_spec(source_spec),),
    )

    graph = compile_replay_graph(spec, display=display)
    code = emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    assert code.count("def _restore_image_tool_dimensions") == 1
    assert "erlab.utils.array._restore_nonuniform_dims" not in code
    assert "erlab.interactive.imagetool.slicer" not in code
    xr.testing.assert_identical(namespace["derived"], source.qsel(x=0.2))


@pytest.mark.parametrize("display", [False, True])
def test_replay_graph_restore_support_preserves_module_prologue(
    display: bool,
    tmp_path: pathlib.Path,
) -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 1.0], "y": [0.0, 1.0]},
    )
    path = tmp_path / "source.nc"
    source.to_netcdf(path)
    file_spec = file_load(
        start_label="Load source",
        seed_code=(
            '"""Load the replay source."""\n'
            "from __future__ import annotations\n"
            "import xarray as xr\n"
            f"derived = xr.load_dataarray({str(path)!r})"
        ),
        file_load_source=_file_replay_source(path),
    )
    spec = compose_full_provenance(
        file_spec,
        public_data(QSelOperation(kwargs={"x": 0.2})),
    )

    graph = compile_replay_graph(spec, display=display)
    code = emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    assert code.startswith(
        '"""Load the replay source."""\nfrom __future__ import annotations'
    )
    assert code.count("from __future__ import annotations") == 1
    assert code.count("import xarray as xr") == 1
    assert code.count("def _restore_image_tool_dimensions") == int(not display)
    assert code.index("from __future__ import annotations") < code.index(
        "import xarray as xr"
    )
    if not display:
        assert code.index("import xarray as xr") < code.index(
            "def _restore_image_tool_dimensions"
        )
    assert namespace["__doc__"] == "Load the replay source."
    xr.testing.assert_identical(namespace["derived"], source.qsel(x=0.2))


def test_replay_graph_restore_support_does_not_shadow_script_function(
    tmp_path: pathlib.Path,
) -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 1.0], "y": [0.0, 1.0]},
    )
    path = tmp_path / "source.nc"
    source.to_netcdf(path)
    spec = script(
        ScriptCodeOperation(
            label="Use script helper",
            code="derived = _restore_image_tool_dimensions(derived)",
        ),
        start_label="Create ImageTool rendering dimensions",
        seed_code=(
            "def _restore_image_tool_dimensions(array):\n"
            "    return array + 1\n"
            "derived = erlab.utils.array._make_dims_uniform(data_0)"
        ),
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Source data",
                provenance_spec=_file_spec(path),
            ),
        ),
        replay_stages=(
            ReplayStage.from_source_spec(public_data(QSelOperation(kwargs={"x": 0.2}))),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert "def _restore_image_tool_dimensions_2(array):" in code
    assert code.count("def _restore_image_tool_dimensions(array):") == 1
    xr.testing.assert_identical(namespace["derived"], source.qsel(x=0.2) + 1)


@pytest.mark.parametrize(
    "source",
    [
        xr.DataArray(
            np.arange(3.0),
            dims=("x_idx",),
            coords={"x": ("x_idx", [0.0, 0.2, 1.0])},
        ),
        xr.DataArray(
            np.arange(3.0),
            dims=("x_idx",),
            coords={"x": ("x_idx", [0.0, 1.0, 2.0])},
        ),
        xr.DataArray(
            [1.0],
            dims=("x_idx",),
            coords={"x": ("x_idx", [0.0])},
        ),
        xr.DataArray(np.arange(3.0), dims=("x_idx",)),
        xr.DataArray(
            np.arange(6.0).reshape(3, 2),
            dims=("x_idx", "y"),
            coords={"x": ("y", [0.0, 0.5])},
        ),
        xr.DataArray(
            np.arange(3.0),
            dims=("x_idx",),
            coords={"x": ("x_idx", ["left", "middle", "right"])},
        ),
        xr.DataArray(
            np.arange(9.0).reshape(3, 3),
            dims=("x_idx", "y_idx"),
            coords={
                "x": ("x_idx", [0.0, 0.0, 1.0]),
                "y": ("y_idx", [1.0, 0.4, -0.8]),
            },
        ),
    ],
    ids=(
        "nonuniform",
        "uniform-user-dimension",
        "singleton",
        "missing-coordinate",
        "wrong-coordinate-dimension",
        "non-numeric-coordinate",
        "multiple-constant-and-descending",
    ),
)
def test_generated_nonuniform_restore_matches_runtime(source: xr.DataArray) -> None:
    function_name = "restore_image_tool_dimensions"
    code = "\n".join(
        (
            _nonuniform_restore_support_code(function_name),
            f"derived = {function_name}(data)",
        )
    )

    namespace = _exec_generated_code(code, {"data": source})

    xr.testing.assert_identical(
        namespace["derived"],
        erlab.utils.array._restore_nonuniform_dims(source),
    )


def test_replay_graph_display_keeps_scoped_bindings_and_inlines_rebound_inputs(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    source = xr.DataArray(np.arange(3.0), dims=("x",))
    source.to_netcdf(path)
    script_input = ScriptInput(
        name="data_0",
        label="ImageTool 0",
        provenance_spec=_file_spec(path),
    )
    helper_spec = script(
        ScriptCodeOperation(
            label="Use helper",
            code="def offset():\n    return data_0 + 1\nresult = offset()",
        ),
        start_label="Run script",
        active_name="result",
        script_inputs=(script_input,),
    )
    rebound_spec = script(
        ScriptCodeOperation(
            label="Rebind input",
            code="data_0 = data_0 + 1\nresult = data_0 * 2",
        ),
        start_label="Run script",
        active_name="result",
        script_inputs=(script_input,),
    )

    helper_code = typing.cast("str", helper_spec.display_code())
    rebound_code = typing.cast("str", rebound_spec.display_code())

    assert len(re.findall(r"^data_0 =", helper_code, flags=re.MULTILINE)) == 1
    assert not re.search(r"^data_0 =", rebound_code, flags=re.MULTILINE)
    xr.testing.assert_identical(_exec_generated_code(helper_code)["result"], source + 1)
    xr.testing.assert_identical(
        _exec_generated_code(rebound_code)["result"],
        (source + 1) * 2,
    )


def test_replay_graph_display_keeps_alias_before_import_rebinding() -> None:
    source = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = script(
        ScriptCodeOperation(
            label="Rebind source name",
            code="import numpy as data\nresult = derived + 1",
        ),
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
    )

    replayed = replay_script_provenance(spec, {"data": source})
    code = typing.cast("str", spec.display_code())
    generated = _exec_generated_code(code, {"data": source})["result"]

    xr.testing.assert_identical(replayed, source + 1)
    xr.testing.assert_identical(generated, replayed)


def test_replay_graph_display_renames_imports_and_free_helper_references(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    source = xr.DataArray(np.arange(3.0), dims=("x",))
    source.to_netcdf(path)

    def branch(delta: int) -> ToolProvenanceSpec:
        return script(
            ScriptCodeOperation(
                label="Transform branch",
                code=(
                    "import numpy as result\n\n"
                    f"numeric = derived + result.float64({delta})\n\n"
                    "def result(script_result):\n"
                    "    return numeric + script_result\n\n"
                    "result = result(1)\n\n"
                    "def identity():\n"
                    "    return result\n\n"
                    "result = identity()"
                ),
            ),
            start_label="Run branch",
            seed_code="derived = data_0",
            active_name="result",
            script_inputs=(
                ScriptInput(
                    name="data_0",
                    label="Source",
                    provenance_spec=_file_spec(path),
                ),
            ),
        )

    spec = script(
        ScriptCodeOperation(label="Add branches", code="total = left + right"),
        start_label="Combine branches",
        active_name="total",
        script_inputs=(
            ScriptInput(
                name="left",
                label="Left branch",
                provenance_spec=branch(1),
            ),
            ScriptInput(
                name="right",
                label="Right branch",
                provenance_spec=branch(2),
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code)

    xr.testing.assert_identical(namespace["total"], (source + 2) + (source + 3))
    generated_copy_targets = {
        statement.targets[0].id
        for statement in ast.parse(code).body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id.startswith("data_0")
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "copy"
    }
    assert not generated_copy_targets


@pytest.mark.parametrize(
    "import_code",
    [
        "import numpy.random, numpy.linalg",
        "import numpy.random\nimport numpy.linalg",
    ],
)
def test_replay_graph_display_renames_dotted_import_root(
    tmp_path: pathlib.Path,
    import_code: str,
) -> None:
    path = tmp_path / "scan.nc"
    source = xr.DataArray(np.arange(3.0), dims=("x",))
    source.to_netcdf(path)

    def branch(delta: int) -> ToolProvenanceSpec:
        return script(
            ScriptCodeOperation(
                label="Transform branch",
                code=f"{import_code}\nnumpy = derived + numpy.float64({delta})",
            ),
            start_label="Run branch",
            seed_code="derived = data_0",
            active_name="numpy",
            script_inputs=(
                ScriptInput(
                    name="data_0",
                    label="Source",
                    provenance_spec=_file_spec(path),
                ),
            ),
        )

    spec = script(
        ScriptCodeOperation(label="Add branches", code="total = left + right"),
        start_label="Combine branches",
        active_name="total",
        script_inputs=(
            ScriptInput(
                name="left",
                label="Left branch",
                provenance_spec=branch(1),
            ),
            ScriptInput(
                name="right",
                label="Right branch",
                provenance_spec=branch(2),
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code)

    xr.testing.assert_identical(namespace["total"], (source + 1) + (source + 2))
    assert not any(
        isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Name)
        and statement.value.id == "numpy"
        and any(
            isinstance(target, ast.Name) and target.id != "numpy"
            for target in statement.targets
        )
        for statement in ast.parse(code).body
    )


@pytest.mark.parametrize(
    ("seed_code", "active_name"),
    [
        ("derived = data_0", "derived"),
        (None, "data_0"),
    ],
)
def test_replay_graph_structured_script_inputs_keep_execution_copy_boundary(
    seed_code: str | None,
    active_name: str,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = script(
        RenameOperation(name="renamed"),
        start_label="Run script",
        seed_code=seed_code,
        active_name=active_name,
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )

    graph = compile_replay_graph(spec, external_inputs={"data_0": data})
    replayed = execute_replay_graph(graph)

    assert any(node.kind == "relay" for node in graph.nodes)
    assert replayed.name == "renamed"
    assert not np.shares_memory(replayed.data, data.data)


def test_replay_graph_disables_numbagg_only_during_execution() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = script(
        ScriptCodeOperation(
            label="Record option",
            code=(
                "derived = data_0.copy()\n"
                "derived.attrs['use_numbagg_during_replay'] = "
                "xr.get_options()['use_numbagg']"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )
    graph = compile_replay_graph(spec, external_inputs={"data_0": data})

    with xr.set_options(use_numbagg=True):
        replayed = execute_replay_graph(graph)
        assert xr.get_options()["use_numbagg"] is True

    assert replayed.attrs["use_numbagg_during_replay"] is False


def test_replay_graph_display_preserves_whole_array_rename(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="source")
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    spec = script(
        RenameOperation(name="renamed"),
        start_label="Run script",
        active_name="data_0",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    graph = compile_replay_graph(
        spec,
        display=True,
    )
    code = emit_replay_code(graph, output_name="derived")

    assert ".rename('renamed')" in code
    xr.testing.assert_identical(
        _exec_generated_code(code)["derived"], data.rename("renamed")
    )


def test_replay_graph_display_preserves_structured_final_rename(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="source")
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    file_spec = compose_full_provenance(
        _file_spec(path),
        full_data(RenameOperation(name="renamed")),
    )
    live_spec = full_data(RenameOperation(name="renamed"))
    assert file_spec is not None

    file_code = typing.cast("str", file_spec.display_code())
    live_code = typing.cast("str", live_spec.display_code(parent_data=data))

    assert ".rename('renamed')" in file_code
    assert ".rename('renamed')" in live_code
    xr.testing.assert_identical(
        _exec_generated_code(file_code)["derived"],
        data.rename("renamed"),
    )
    xr.testing.assert_identical(
        _exec_generated_code(live_code, {"data": data})["derived"],
        data.rename("renamed"),
    )


def test_replay_graph_display_keeps_name_rename_before_script_code(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="source")
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    spec = script(
        RenameOperation(name="renamed"),
        ScriptCodeOperation(
            label="Use DataArray name",
            code="derived = derived.rename(derived.name + '_used')",
        ),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    graph = compile_replay_graph(
        spec,
        display=True,
    )
    code = emit_replay_code(graph, output_name="derived")

    assert ".rename(" in code
    xr.testing.assert_identical(
        _exec_generated_code(code)["derived"],
        data.rename("renamed_used"),
    )


def test_script_replayability_does_not_generate_structured_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = script(
        RenameOperation(name="renamed"),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(ScriptInput(name="data_0", label="Input"),),
    )

    def fail_derivation_entry(self):
        raise AssertionError("replayability checks must not generate copied code")

    monkeypatch.setattr(RenameOperation, "derivation_entry", fail_derivation_entry)

    assert script_provenance_replayable(spec)


def test_replay_graph_uses_existing_console_alias_for_script_code(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    data.to_netcdf(path)
    spec = script(
        ScriptCodeOperation(
            label="Rotate",
            code=(
                "derived = era.transform.rotate("
                "data_0, 0.0, axes=('x', 'y'), reshape=False)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Scan",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert "era = erlab.analysis" not in code
    xr.testing.assert_identical(
        namespace["derived"],
        erlab.analysis.transform.rotate(data, 0.0, axes=("x", "y"), reshape=False),
    )


def test_replay_graph_uses_existing_console_alias_for_structured_code(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    data.to_netcdf(path)
    spec = script(
        RotateOperation(
            angle=123.456,
            axes=("x", "y"),
            center=(1.234, 5.678),
            reshape=False,
            order=3,
        ),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Scan",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert "era = erlab.analysis" not in code
    xr.testing.assert_identical(
        namespace["derived"],
        erlab.analysis.transform.rotate(
            data,
            123.456,
            axes=("x", "y"),
            center=(1.234, 5.678),
            reshape=False,
            order=3,
        ),
    )


def test_replay_graph_keeps_structurally_distinct_file_loads() -> None:
    first = _file_spec("scan.h5", selected_index=0)
    second = _file_spec("scan.h5", selected_index=1)
    spec = script(
        ScriptCodeOperation(label="Subtract", code="derived = data_0 - data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="First", provenance_spec=first),
            ScriptInput(name="data_1", label="Second", provenance_spec=second),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count("xr.load_dataarray") == 2


def test_replay_graph_reuses_shared_loader_setup() -> None:
    spec = script(
        ScriptCodeOperation(
            label="Add",
            code="derived = data_0 + data_1",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=_erlab_file_spec("scan0.h5", "example"),
            ),
            ScriptInput(
                name="data_1",
                label="Second",
                provenance_spec=_erlab_file_spec("scan1.h5", "example"),
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert "import erlab" not in code
    assert code.count("erlab.io.set_loader('example')") == 1
    assert code.count("erlab.io.load") == 2


def test_replay_graph_reemits_stateful_setup_after_loader_change() -> None:
    spec = script(
        ScriptCodeOperation(
            label="Add",
            code="derived = data_0 + data_1 + data_2",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Alpha 0",
                provenance_spec=_erlab_file_spec("alpha0.h5", "alpha"),
            ),
            ScriptInput(
                name="data_1",
                label="Beta",
                provenance_spec=_erlab_file_spec("beta.h5", "beta"),
            ),
            ScriptInput(
                name="data_2",
                label="Alpha 1",
                provenance_spec=_erlab_file_spec("alpha1.h5", "alpha"),
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count("erlab.io.set_loader('alpha')") == 2
    assert code.count("erlab.io.set_loader('beta')") == 1
    assert code.index("erlab.io.load('alpha0.h5')") < code.index(
        "erlab.io.set_loader('beta')"
    )


def test_replay_graph_does_not_merge_operations_with_different_contexts() -> None:
    file_spec = _file_spec("scan.h5")
    first_spec = compose_full_provenance(
        file_spec,
        full_data(IselOperation(kwargs={"pol": 0})),
    )
    second_spec = compose_full_provenance(
        file_spec,
        selection(IselOperation(kwargs={"pol": 0})),
    )
    assert first_spec is not None
    assert second_spec is not None
    spec = script(
        ScriptCodeOperation(label="Subtract", code="derived = data_0 - data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="Full", provenance_spec=first_spec),
            ScriptInput(
                name="data_1",
                label="Selection",
                provenance_spec=second_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count(".isel") == 2


def test_replay_graph_script_nodes_are_not_deduplicated() -> None:
    first = script(
        start_label="Make first",
        seed_code="derived = xr.DataArray([1.0, 2.0], dims=['x'])",
        active_name="derived",
    )
    second = script(
        start_label="Make second",
        seed_code="derived = xr.DataArray([10.0, 20.0], dims=['x'])",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(label="Add", code="derived = data_0 + data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="data_0", label="First", provenance_spec=first),
            ScriptInput(name="data_1", label="Second", provenance_spec=second),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert code.count("xr.DataArray") == 2
    xr.testing.assert_identical(
        namespace["derived"],
        xr.DataArray([11.0, 22.0], dims=["x"]),
    )


def test_replay_graph_does_not_hoist_imports_from_user_script_code() -> None:
    spec = script(
        ScriptCodeOperation(
            label="User code",
            code=(
                "events.append('before')\n"
                "import statistics\n"
                "events.append('after')\n"
                "fig = statistics.fmean([1.0, 2.0])"
            ),
        ),
        start_label="Build result",
        seed_code="events = []",
        active_name="fig",
    )

    code = typing.cast("str", spec.display_code())
    assert code.index("events.append") < code.index("import statistics")
    namespace = _exec_generated_code(code)
    assert namespace["events"] == ["before", "after"]
    assert namespace["fig"] == 1.5


def test_group_framework_imports_preserves_conflicting_alias_bindings() -> None:
    code = _group_framework_imports(
        (
            (
                "import xarray as array_module\n"
                "first = array_module.DataArray([1.0], dims=['x'])",
                True,
            ),
            (
                "import numpy as array_module\nsecond = array_module.asarray([2.0])",
                True,
            ),
        )
    )

    assert code.index("import xarray") < code.index("first =")
    assert code.index("first =") < code.index("import numpy")
    assert code.index("import numpy") < code.index("second =")
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["first"], xr.DataArray([1.0], dims=["x"]))
    np.testing.assert_array_equal(namespace["second"], np.asarray([2.0]))


def test_group_framework_imports_deduplicates_canonical_aliases() -> None:
    code = _group_framework_imports(
        (
            ("import numpy as np\nfirst = np.asarray([1.0])", True),
            ("import numpy as np\nsecond = np.asarray([2.0])", True),
        )
    )

    assert code.count("import numpy as np") == 1
    namespace = _exec_generated_code(code)
    np.testing.assert_array_equal(namespace["first"], np.asarray([1.0]))
    np.testing.assert_array_equal(namespace["second"], np.asarray([2.0]))


def test_group_framework_imports_preserves_rebinding_before_import() -> None:
    code = _group_framework_imports(
        (
            ("np = 'sentinel'\nfirst = np", True),
            ("import numpy as np\nsecond = np.asarray([2.0])", True),
        )
    )

    assert code.index("first =") < code.index("import numpy as np")
    namespace = _exec_generated_code(code)
    assert namespace["first"] == "sentinel"
    np.testing.assert_array_equal(namespace["second"], np.asarray([2.0]))


def test_group_framework_imports_places_future_imports_first() -> None:
    code = _group_framework_imports(
        (
            ("import numpy as np\nfirst = np.asarray([1.0])", True),
            (
                "from __future__ import annotations\nsecond: MissingType | None = None",
                True,
            ),
        )
    )

    assert code.startswith("from __future__ import annotations\n")
    namespace = _exec_generated_code(code)
    assert namespace["__annotations__"] == {"second": "MissingType | None"}


def test_replay_import_helpers_preserve_nonhoistable_imports() -> None:
    source = (
        "import numpy as np\n"
        "from xarray import DataArray as Array\n"
        "result = Array(np.asarray([1.0]))"
    )
    imports, body = _leading_top_level_imports(source)

    assert imports == [
        ("import numpy as np", "import numpy as np"),
        (
            "from xarray import DataArray as Array",
            "from xarray import DataArray as Array",
        ),
    ]
    assert body == "result = Array(np.asarray([1.0]))"
    assert _leading_top_level_imports("if True:\n") == (
        [],
        "if True:\n",
    )
    assert _leading_top_level_imports("value = 1") == ([], "value = 1")
    commented_import = "import numpy as np  # preserve this comment\nvalue = np"
    assert _leading_top_level_imports(commented_import) == (
        [],
        commented_import,
    )

    assert _import_binding_targets("import numpy.linalg") == {"numpy": "module:numpy"}
    assert _import_binding_targets("import numpy.linalg as linalg, xarray as xr") == {
        "linalg": "module:numpy.linalg",
        "xr": "module:xarray",
    }
    assert _import_binding_targets("from ..package import item as alias") == {
        "alias": "from:..package:item"
    }
    assert _import_binding_targets("from __future__ import annotations") == {}
    assert _import_binding_targets("from package import *") is None
    assert (
        _import_binding_targets("from package import first as item, second as item")
        is None
    )
    assert _import_binding_targets("import numpy as module, xarray as module") is None
    assert _is_future_import("from __future__ import annotations")
    assert not _is_future_import("import numpy as np")


def test_replay_import_helpers_preserve_imports_without_ast_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_node = ast.Import(names=[ast.alias(name="numpy")])
    import_node.lineno = 1
    import_node.col_offset = 0
    import_node.end_lineno = None
    import_node.end_col_offset = None
    module = ast.Module(body=[import_node], type_ignores=[])
    monkeypatch.setattr(_graph.ast, "parse", lambda *_args, **_kwargs: module)

    assert _leading_top_level_imports("import numpy") == (
        [],
        "import numpy",
    )


def test_replay_import_helpers_track_names_bound_by_python_constructs() -> None:
    accessed, rebound = _code_name_accesses(
        textwrap.dedent(
            """\
            import numpy as np
            from package import *
            def helper():
                pass
            class Result:
                pass
            value = np
            del value
            try:
                raise error_type
            except error_type as caught:
                handled = caught
            try:
                raise error_type
            except:
                handled_without_name = True
            match subject:
                case [head, *tail] as whole:
                    matched = whole
                case {"key": item, **remaining}:
                    mapped = remaining
            """
        )
    )

    assert {
        "*",
        "Result",
        "caught",
        "head",
        "helper",
        "remaining",
        "tail",
        "value",
        "whole",
    } <= accessed
    assert {
        "*",
        "Result",
        "caught",
        "head",
        "helper",
        "remaining",
        "tail",
        "value",
        "whole",
    } <= rebound
    assert _code_name_accesses("if value:\n") == ({"*"}, {"*"})


def test_group_framework_imports_keeps_unsafe_and_rebound_imports_local() -> None:
    code = _group_framework_imports(
        (
            (
                "from __future__ import annotations\nfrom package import *\nfirst = 1",
                True,
            ),
            (
                "from __future__ import annotations\nsecond: MissingType | None = None",
                True,
            ),
            ("import numpy as np\nthird = np.asarray([3.0])", True),
        )
    )

    assert code.count("from __future__ import annotations") == 1
    assert code.index("from package import *") < code.index("first = 1")
    assert code.index("first = 1") < code.index("import numpy as np")
    assert code.index("import numpy as np") < code.index("third =")

    rebound_code = _group_framework_imports(
        (
            ("import numpy as np\nfirst = np.asarray([1.0])", True),
            ("np = 'changed'", False),
            ("import numpy as np\nsecond = np.asarray([2.0])", True),
        )
    )
    assert rebound_code.count("import numpy as np") == 2
    assert rebound_code.index("first =") < rebound_code.index("np = 'changed'")
    assert rebound_code.index("np = 'changed'") < rebound_code.rindex(
        "import numpy as np"
    )

    non_framework_code = "import numpy as np\nvalue = np.asarray([1.0])"
    assert (
        _group_framework_imports(((non_framework_code, False),)) == non_framework_code
    )
    assert (
        _group_framework_imports((("import numpy as np", True),))
        == "import numpy as np"
    )


def test_replay_graph_allows_for_loop_script_code() -> None:
    spec = script(
        ScriptCodeOperation(
            label="Build figure",
            code=(
                "fig = []\n"
                "profiles = list(profile_data.transpose('eV', ...))\n"
                "for profile in profiles:\n"
                "    fig.append(profile.sum())"
            ),
        ),
        start_label="Build figure",
        seed_code=(
            "profile_data = xr.DataArray("
            "np.arange(6.0).reshape(2, 3), "
            "dims=('eV', 'alpha'), "
            "coords={'alpha': [0.0, 1.0, 2.0]}"
            ")"
        ),
        active_name="fig",
    )

    graph = compile_replay_graph(spec, display=True)
    code = emit_replay_code(graph, output_name="fig")
    namespace = _exec_generated_code(code)

    assert "for profile in profiles:" in code
    assert len(namespace["fig"]) == 2
    xr.testing.assert_identical(namespace["fig"][0], xr.DataArray(3.0))
    xr.testing.assert_identical(namespace["fig"][1], xr.DataArray(12.0))


def test_replay_graph_allows_for_loop_with_script_input() -> None:
    profile_source = script(
        start_label="Make profile data",
        seed_code=(
            "profile_data = xr.DataArray("
            "np.arange(6.0).reshape(2, 3), "
            "dims=('eV', 'alpha'), "
            "coords={'alpha': [0.0, 1.0, 2.0]}"
            ")"
        ),
        active_name="profile_data",
    )
    spec = script(
        ScriptCodeOperation(
            label="Build figure",
            code=(
                "fig = []\n"
                "profiles = list(profile_data.transpose('eV', ...))\n"
                "for profile in profiles:\n"
                "    fig.append(profile.sum())"
            ),
        ),
        start_label="Build figure",
        active_name="fig",
        script_inputs=(
            ScriptInput(
                name="profile_data",
                label="Profile data",
                provenance_spec=profile_source,
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert "for profile in profiles:" in code
    assert len(namespace["fig"]) == 2
    xr.testing.assert_identical(namespace["fig"][0], xr.DataArray(3.0))
    xr.testing.assert_identical(namespace["fig"][1], xr.DataArray(12.0))


def test_replay_graph_allows_comprehension_with_script_input_source(
    tmp_path: pathlib.Path,
) -> None:
    source_data = xr.DataArray(
        np.asarray([[2.0, 4.0], [10.0, 14.0]]),
        dims=("sample_temp", "eV"),
        coords={
            "sample_temp": [10.0, 50.0],
            "eV": [-0.3, -0.2],
            "mesh_current": ("sample_temp", [2.0, 2.0]),
        },
        name="D10cu",
    )
    source_path = tmp_path / "source.nc"
    source_data.to_netcdf(source_path)
    source_spec = _file_spec(source_path).model_copy(
        update={
            "replay_stages": (
                ReplayStage(
                    source_kind="public_data",
                    operations=(DivideByCoordOperation(coord_name="mesh_current"),),
                ),
            ),
        }
    )
    spec = script(
        ScriptCodeOperation(
            label="Build figure",
            code=(
                "fig = []\n"
                "profile_data = data_3\n"
                "profiles = list(profile_data.transpose('sample_temp', ...))\n"
                "line_color_values = [\n"
                '    float(profile.coords["sample_temp"].values.item())\n'
                "    for profile in profiles\n"
                "]\n"
                "for profile, color in zip(\n"
                "    profiles,\n"
                "    line_color_values,\n"
                "    strict=True,\n"
                "):\n"
                "    fig.append(float(profile.sum().values) + color)"
            ),
        ),
        start_label="Build figure",
        active_name="fig",
        script_inputs=(
            ScriptInput(
                name="data_3",
                label="ImageTool 3: D10cu",
                provenance_spec=source_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert "for profile in profiles" in code
    assert namespace["fig"] == [13.0, 62.0]


def test_replay_graph_display_emits_user_code_blocked_by_replay_allowlist(
    tmp_path: pathlib.Path,
) -> None:
    source_data = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0.0, 1.0]})
    source_path = tmp_path / "source.nc"
    source_data.to_netcdf(source_path)
    source_spec = _file_spec(source_path)
    script_input = ScriptInput(
        name="data_0",
        label="ImageTool 0",
        provenance_spec=source_spec,
    )
    spec = script(
        ScriptCodeOperation(
            label="User code",
            code=("import os\nwith open(os.devnull):\n    pass\nderived = data_0 + 1"),
        ),
        start_label="Run user code",
        active_name="derived",
        script_inputs=(script_input,),
    )

    assert not script_provenance_replayable(spec)
    with pytest.raises(ReplayGraphError, match="unsupported Import"):
        compile_replay_graph(spec)

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert "import os" in code
    assert "with open(os.devnull):" in code
    xr.testing.assert_identical(namespace["derived"], source_data + 1)

    unresolved_spec = spec.model_copy(
        update={
            "operations": (
                ScriptCodeOperation(
                    label="User code",
                    code="derived = data_0 + missing",
                ),
            ),
        },
    )
    with pytest.raises(ReplayGraphError, match="unresolved name"):
        compile_replay_graph(unresolved_spec, display=True)


def test_replay_graph_trusted_user_code_executes_blocked_constructs() -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",))
    spec = script(
        ScriptCodeOperation(
            label="User code",
            code=(
                "import os\n"
                "with open(os.devnull):\n"
                "    pass\n"
                "derived = data + int(os.path.exists(os.devnull))"
            ),
        ),
        start_label="Run user code",
        active_name="derived",
        script_inputs=(ScriptInput(name="data", label="Input"),),
    )

    assert not script_provenance_replayable(spec)
    assert script_provenance_requires_trust(spec)
    with pytest.raises(ReplayGraphError, match="unsupported Import"):
        compile_replay_graph(spec, external_inputs={"data": data})

    result = replay_script_provenance(
        spec,
        {"data": data},
        trusted_user_code=True,
    )

    xr.testing.assert_identical(result, data + 1)


def test_replay_graph_trusted_user_code_still_validates_result_type() -> None:
    data = xr.DataArray([1.0], dims=("x",))
    spec = script(
        ScriptCodeOperation(label="Bad output", code="derived = 1"),
        start_label="Run user code",
        active_name="derived",
    )

    with pytest.raises(TypeError, match="did not produce"):
        replay_script_provenance(
            spec,
            {"data": data},
            trusted_user_code=True,
        )


def test_replay_graph_trusted_user_code_replays_nested_scripts(
    tmp_path: pathlib.Path,
) -> None:
    source = xr.DataArray([1.0, 2.0], dims=("x",))
    source_path = tmp_path / "source.nc"
    source.to_netcdf(source_path)
    source_spec = _file_spec(source_path)
    nested_spec = script(
        ScriptCodeOperation(
            label="User code",
            code="import os\nderived = data_0 + int(os.path.exists(os.devnull))",
        ),
        start_label="Run nested code",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=source_spec,
            ),
        ),
    )
    spec = script(
        ScriptCodeOperation(
            label="Use nested",
            code="derived = data_1 * 2",
        ),
        start_label="Run outer code",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data_1",
                label="Nested",
                provenance_spec=nested_spec,
            ),
        ),
    )

    assert script_provenance_replayable(spec)
    assert script_provenance_requires_trust(spec)
    trust_payload = _script_trust_payload(spec)
    assert trust_payload is not None
    assert trust_payload["inputs"][0]["name"] == "data_1"
    assert trust_payload["inputs"][0]["payload"]["operations"][0]["code"].startswith(
        "import os"
    )
    mixed_payload = _script_trust_payload(
        spec.model_copy(
            update={
                "operations": (
                    AverageOperation(dims=("x",)),
                    *spec.operations,
                )
            }
        )
    )
    assert mixed_payload is not None
    assert len(mixed_payload["operations"]) == len(trust_payload["operations"])
    assert script_provenance_trust_key(spec) is not None
    with pytest.raises(ReplayGraphError, match="recorded operation"):
        rebuild_script_provenance(spec)

    result, rebuilt = rebuild_script_provenance(
        spec,
        trusted_user_code=True,
    )

    assert rebuilt.kind == "script"
    xr.testing.assert_identical(result, (source + 1) * 2)


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("while True:\n    derived = data", "unsupported While"),
        (
            "try:\n    derived = data\nexcept Exception:\n    derived = data",
            "unsupported Try",
        ),
        ("with open('scan.nc') as handle:\n    derived = data", "unsupported With"),
        ("derived = globals()", "cannot call 'globals'"),
        ("derived = locals()", "cannot call 'locals'"),
    ],
)
def test_replay_graph_rejects_unsupported_script_constructs(
    code: str, message: str
) -> None:
    data = xr.DataArray([1.0], dims=("x",))
    spec = script(
        ScriptCodeOperation(label="Unsupported", code=code),
        start_label="Run script",
        active_name="derived",
    )

    with pytest.raises(ReplayGraphError, match=re.escape(message)):
        compile_replay_graph(spec, external_inputs={"data": data})


def test_replay_graph_raises_typed_errors_for_unsupported_script() -> None:
    data = xr.DataArray([1.0], dims=("x",))
    spec = script(
        ScriptCodeOperation(label="Unsupported", code="import os\nderived = data"),
        start_label="Run script",
        active_name="derived",
    )

    with pytest.raises(ReplayGraphError, match="unsupported Import"):
        compile_replay_graph(spec, external_inputs={"data": data})


def test_replay_graph_emits_correct_with_edge_operation_code(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "scan.nc"
    data = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0.0, 1.0]})
    data.to_netcdf(path)
    edge_fit = xr.Dataset({"center": ("x", [0.0, 1.0])})

    def correct_with_edge(data_arg, edge_fit_arg, *, shift_coords=True):
        xr.testing.assert_identical(edge_fit_arg, edge_fit)
        return data_arg.assign_attrs(shift_coords=shift_coords)

    monkeypatch.setattr(erlab.analysis.gold, "correct_with_edge", correct_with_edge)
    spec = compose_full_provenance(
        _file_spec(path),
        full_data(CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False)),
    )
    assert spec is not None

    graph = compile_replay_graph(spec)
    code = emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    assert namespace["derived"].attrs["shift_coords"] is False
    assert execute_replay_graph(graph).attrs["shift_coords"] is False


def test_replay_graph_execution_matches_emitted_code(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "source.nc"
    source = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [0, 1, 2]},
    )
    source.to_netcdf(path)
    spec = compose_full_provenance(
        _file_spec(path),
        full_data(AverageOperation(dims=("y",))),
    )
    assert spec is not None

    graph = compile_replay_graph(spec)
    replayed = execute_replay_graph(graph)
    code = emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    xr.testing.assert_identical(replayed, namespace["derived"])
