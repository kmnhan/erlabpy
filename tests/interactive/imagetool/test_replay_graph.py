import ast
import hashlib
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
import erlab.extensions._api as extension_api
from erlab.interactive._code_trust import issue_execution_capability, new_document_trust
from erlab.interactive.imagetool._load_source import _register_local_callable_loader
from erlab.interactive.imagetool._provenance import _execution, _graph
from erlab.interactive.imagetool._provenance._code import (
    _SCRIPT_REPLAY_ALLOWED_BUILTINS,
    _code_uses_name,
    _code_uses_name_any_scope,
    _nonuniform_restore_support_code,
    _replace_code_identifiers,
)
from erlab.interactive.imagetool._provenance._execution import (
    execute_replay_graph,
    rebuild_script_inputs,
    rebuild_script_provenance,
    replay_script_provenance,
    script_provenance_replayable,
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
    _simple_name_assignment,
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
    ExtensionRoutineOperation,
    GaussianFilterOperation,
    ImageDerivativeOperation,
    IselOperation,
    KspaceWorkFunctionOperation,
    QSelOperation,
    RenameOperation,
    RestoreNonuniformDimsOperation,
    RotateOperation,
    ScriptCodeOperation,
    SelOperation,
    SortCoordOrderOperation,
    SqueezeOperation,
    TransposeOperation,
)
from erlab.interactive.imagetool._provenance._trust import (
    provenance_replay_graph_code_trust_entries,
    provenance_requires_code_trust,
)


def _authorize_execution(entries: tuple[typing.Any, ...]) -> object:
    _trust, capability = issue_execution_capability(new_document_trust(), entries)
    if capability is None:  # pragma: no cover - local trust always issues one.
        raise RuntimeError("Could not issue test execution capability")
    return capability


def _exec_generated_code(
    code: str, namespace_items: dict[str, typing.Any] | None = None
) -> dict[str, typing.Any]:
    namespace = dict(namespace_items or {})
    exec(code, namespace, namespace)  # noqa: S102
    return namespace


def _extension_routine_operation() -> ExtensionRoutineOperation:
    return ExtensionRoutineOperation(
        script_name="local_routine.py",
        source_hash="a" * 64,
        routine_id="scale",
        routine_name="Scale",
        parameters={},
    )


def _registered_routine(
    script_path: pathlib.Path,
) -> extension_api._RegisteredScriptCapability:
    source_bytes = script_path.read_bytes()
    return extension_api._RegisteredScriptCapability(
        registered_path=script_path,
        script_name=script_path.name,
        source_hash=hashlib.sha256(source_bytes).hexdigest(),
        descriptor=erlab.extensions.RoutineDescriptor(
            id="scale",
            name="Scale",
            category="Other",
            summary="",
            function_name="scale",
        ),
        source_bytes=source_bytes,
    )


def _registered_loader(
    script_path: pathlib.Path,
) -> extension_api._RegisteredScriptCapability:
    source_bytes = script_path.read_bytes()
    return extension_api._RegisteredScriptCapability(
        registered_path=script_path,
        script_name=script_path.name,
        source_hash=hashlib.sha256(source_bytes).hexdigest(),
        descriptor=erlab.extensions.LoaderDescriptor(
            id="load_data",
            name="Load data",
            category="Other",
            summary="",
            function_name="load_data",
        ),
        source_bytes=source_bytes,
    )


def _file_replay_source(
    path: pathlib.Path | str,
    *,
    selected_index: int = 0,
    load_code: str | None = None,
):
    _register_local_callable_loader(xr.load_dataarray)
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
    seed_code = f"import xarray as xr\n\nderived = xr.load_dataarray({str(path)!r})"
    return file_load(
        start_label="Load source",
        seed_code=seed_code,
        file_load_source=_file_replay_source(
            path,
            selected_index=selected_index,
            load_code=seed_code.replace("derived =", "data =", 1),
        ),
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
        replay_script_provenance(spec, {"data": data}, authorize=_authorize_execution),
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
    assert _simple_name_assignment("target = source") == ("target", "source")
    assert _simple_name_assignment("target: xr.DataArray = source") is None
    assert _simple_name_assignment("target =") is None
    assert _simple_name_assignment("target = source\nother = source") is None

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
        active_name="derived",
    )
    assert not script_provenance_replayable(external_input_spec)
    assert script_provenance_replayable(
        external_input_spec,
        external_input_names={"data"},
    )
    assert _single_assignment_output_name("derived: xr.DataArray = data") == "derived"
    assert _single_assignment_output_name("derived =") is None
    assert _single_assignment_output_name("obj.value = data") is None
    assert _single_assignment_output_name("first = data\nsecond = data") is None


@pytest.mark.parametrize("alias", ["derived", "source_data", "seed_result"])
def test_replay_graph_removes_only_unused_simple_seed_aliases(alias: str) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    unused = script(
        ScriptCodeOperation(label="Offset input", code="result = watched + 1.0"),
        start_label="Start from watched data",
        seed_code=f"{alias} = watched",
        active_name="result",
    )
    unused_graph = compile_replay_graph(unused, display=True)
    unused_code = typing.cast("str", unused.display_code())

    assert not any(
        f"{alias} = watched" in node.payload.get("codes", ())
        for node in unused_graph.nodes
    )
    xr.testing.assert_identical(
        _exec_generated_code(unused_code, {"watched": data})["result"],
        data + 1.0,
    )

    used = script(
        ScriptCodeOperation(label="Add input", code=f"result = {alias} + {alias}"),
        start_label="Start from watched data",
        seed_code=f"{alias} = watched",
        active_name="result",
    )
    used_code = typing.cast("str", used.display_code())

    assert f"{alias} = watched" not in used_code
    assert "result = watched + watched" in used_code
    xr.testing.assert_identical(
        _exec_generated_code(used_code, {"watched": data})["result"],
        data + data,
    )


def test_replay_graph_preserves_annotated_seed_alias() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    spec = script(
        ScriptCodeOperation(label="Offset input", code="result = watched + 1.0"),
        start_label="Start from watched data",
        seed_code="source_data: object = watched",
        active_name="result",
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"watched": data})

    assert "source_data: object = watched" in code
    assert namespace["source_data"] is data
    xr.testing.assert_identical(namespace["result"], data + 1.0)


def test_replay_graph_omits_trivial_imports_for_one_framework_step() -> None:
    operation = ScriptCodeOperation(
        label="Copy data",
        code="result = xr.DataArray(np.asarray(data), dims=data.dims)",
        uses_implicit_framework_imports=True,
    )
    restored = ScriptCodeOperation.model_validate(operation.model_dump(mode="json"))
    spec = script(
        restored,
        start_label="Start from data",
        seed_code="result = data",
        active_name="result",
    )
    data = xr.DataArray([1.0, 2.0], dims="x")

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"data": data, "np": np, "xr": xr})

    assert restored.uses_implicit_framework_imports is True
    assert "import numpy" not in code
    assert "import xarray" not in code
    assert "result = data" not in code
    xr.testing.assert_identical(namespace["result"], data)


def test_replay_graph_imports_framework_once_for_composed_display_code() -> None:
    first = GaussianFilterOperation(sigma={"x": 0.5})
    second = GaussianFilterOperation(sigma={"x": 1.0})
    data = xr.DataArray([1.0, 2.0, 3.0], dims="x")
    spec = script(
        first,
        second,
        start_label="Start from data",
        seed_code="derived = data",
        active_name="derived",
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"data": data})

    assert code.count("import erlab.analysis as era") == 1
    xr.testing.assert_identical(
        namespace["derived"],
        second.apply(first.apply(data)),
    )


@pytest.mark.parametrize("active_name", ["era", "np", "xr"])
def test_replay_graph_reserves_framework_names_for_script_outputs(
    active_name: str,
) -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    spec = script(
        GaussianFilterOperation(sigma={"x": 0.5}),
        start_label="Start from data",
        seed_code=f"{active_name} = data",
        active_name=active_name,
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(
        code,
        {"data": data, "era": erlab.analysis},
    )

    assert "import erlab.analysis as era" not in code
    xr.testing.assert_identical(
        namespace[active_name],
        GaussianFilterOperation(sigma={"x": 0.5}).apply(data),
    )


def test_replay_graph_reuses_explicit_canonical_framework_import() -> None:
    spec = script(
        ScriptCodeOperation(
            label="Copy input",
            code="result = xr.DataArray(intermediate)",
            uses_implicit_framework_imports=True,
        ),
        start_label="Build data",
        seed_code="import xarray as xr\nintermediate = xr.DataArray([1.0])",
        active_name="result",
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code)

    assert code.count("import xarray as xr") == 1
    xr.testing.assert_identical(namespace["result"], xr.DataArray([1.0]))


def test_replay_graph_restores_framework_import_after_user_alias_collision() -> None:
    spec = script(
        ScriptCodeOperation(
            label="Copy input",
            code="result = xr.DataArray(intermediate)",
            uses_implicit_framework_imports=True,
        ),
        start_label="Start from data",
        seed_code="import types as xr\nintermediate = data",
        active_name="result",
    )
    data = xr.DataArray([1.0, 2.0], dims="x")

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"data": data})

    xr.testing.assert_identical(namespace["result"], data)


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
        trusted_user_code=True,
    )
    rebound_display_graph = compile_replay_graph(
        rebound_input,
        display=True,
        external_inputs={"data_0": data},
    )
    xr.testing.assert_identical(
        execute_replay_graph(rebound_graph, authorize=_authorize_execution),
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
        trusted_user_code=True,
    )

    xr.testing.assert_identical(
        execute_replay_graph(active_graph, authorize=_authorize_execution),
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
        script_graph = ReplayGraph(trusted_user_code=True)
        script_key = script_graph.add_node(
            f"script-{message}",
            "script",
            payload={"bindings": (), "codes": codes, "active_name": "derived"},
        )
        script_graph.output_key = script_key
        with pytest.raises(ReplayGraphError, match=message):
            execute_replay_graph(script_graph, authorize=_authorize_execution)

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


def test_replay_graph_partial_capability_reuses_cached_numeric_data() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    graph = ReplayGraph()
    input_key = graph.add_node("input", "live_input", payload={"data": data})
    external_operation = types.SimpleNamespace(
        op="model_fit",
        parameters={
            "dependent": types.SimpleNamespace(expr="2 * independent"),
        },
    )
    external_key = graph.add_node(
        "external-expression",
        "operation",
        parents=(input_key,),
        payload={"operation": external_operation},
    )
    local_key = graph.add_node(
        "local-script",
        "script",
        parents=(external_key,),
        cacheable=False,
        payload={
            "active_name": "derived",
            "bindings": (("cached", external_key),),
            "codes": ("derived = cached + 1",),
            "document_codes": ("derived = cached + 1",),
            "hoist_imports": (False,),
        },
    )
    graph.output_key = local_key
    entries = provenance_replay_graph_code_trust_entries(
        graph,
        location_prefix="runtime",
    )
    local_entries = tuple(
        entry for entry in entries if entry.feature == "erlab.provenance.script-code"
    )
    _trust, partial_capability = issue_execution_capability(
        new_document_trust(),
        local_entries,
    )
    assert partial_capability is not None

    with pytest.raises(ReplayGraphError, match="not trusted"):
        execute_replay_graph(graph, authorization=partial_capability)

    xr.testing.assert_identical(
        execute_replay_graph(
            graph,
            cache={external_key: data + 1},
            authorization=partial_capability,
        ),
        data + 2,
    )


def test_script_input_parsing_does_not_affect_model_equality() -> None:
    script_input = ScriptInput(
        name="data",
        label="Input data",
        provenance_spec=full_data(),
    )
    equivalent = ScriptInput.model_validate(script_input.model_dump())

    assert script_input == equivalent
    parsed = script_input.parsed_provenance_spec()
    assert parsed == full_data()
    assert script_input.parsed_provenance_spec() == parsed
    assert script_input == equivalent

    without_provenance = script_input.model_copy(update={"provenance_spec": None})
    assert without_provenance.parsed_provenance_spec() is None


def test_rebuild_script_inputs_controls_recorded_fallback(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))
    path = tmp_path / "recorded-input.nc"
    data.to_netcdf(path)
    source_spec = selection(TransposeOperation(dims=("y", "x")))
    fallback = compose_full_provenance(_file_spec(path), source_spec)
    assert fallback is not None
    script_input = ScriptInput(
        name="data",
        label="Closed input",
        node_uid="missing-node",
        source_spec=source_spec,
        provenance_spec=fallback,
    )
    expected = source_spec.apply(data)

    with pytest.raises(ReplayGraphError, match="not available in this Manager"):
        rebuild_script_inputs(
            (script_input,),
            live_input_resolver=lambda _input: None,
            allow_recorded=False,
        )

    resolved, refreshed = rebuild_script_inputs(
        (script_input,),
        live_input_resolver=lambda _input: None,
        authorize=lambda *_args: pytest.fail(
            "file fallback must not request script authorization"
        ),
        allow_recorded=True,
    )

    xr.testing.assert_identical(resolved["data"], expected)
    assert refreshed[0].node_uid is None

    fallback_spec = script(
        start_label="Copy transformed input",
        seed_code="derived = data",
        active_name="derived",
        script_inputs=(script_input,),
    )
    xr.testing.assert_identical(
        execute_replay_graph(compile_replay_graph(fallback_spec)),
        expected,
    )
    code = emit_replay_code(
        compile_replay_graph(fallback_spec, display=True),
        output_name="derived",
    )
    xr.testing.assert_identical(_exec_generated_code(code)["derived"], expected)

    external_spec = fallback_spec.model_copy(
        update={
            "script_inputs": (
                script_input.model_copy(update={"provenance_spec": None}),
            )
        }
    )
    external_code = emit_replay_code(
        compile_replay_graph(external_spec, display=True),
        output_name="derived",
    )
    xr.testing.assert_identical(
        _exec_generated_code(external_code, {"data": data})["derived"],
        expected,
    )

    resolved, refreshed = rebuild_script_inputs(
        (script_input,),
        live_input_resolver=lambda item: (expected, item),
        authorize=lambda *_args: pytest.fail(
            "live input must not authorize recorded provenance"
        ),
    )
    xr.testing.assert_identical(resolved["data"], expected)
    assert refreshed == (script_input,)

    with pytest.raises(ReplayGraphError, match="Duplicate named provenance input"):
        rebuild_script_inputs(
            (script_input, script_input),
            live_input_resolver=lambda _item: pytest.fail(
                "duplicate names must fail before resolution"
            ),
        )


@pytest.mark.parametrize(
    ("left_uid", "left_role", "right_uid", "right_role"),
    [
        ("shared-node", "source", "shared-node", "displayed"),
        ("left-node", "displayed", "right-node", "displayed"),
    ],
    ids=("data-role", "node-uid"),
)
def test_rebuild_script_provenance_cache_separates_live_inputs(
    left_uid,
    left_role,
    right_uid,
    right_role,
) -> None:
    source = xr.DataArray([[1.0, 3.0], [5.0, 7.0]], dims=("x", "y"))
    displayed = xr.DataArray([[10.0, 30.0], [50.0, 70.0]], dims=("x", "y"))
    data_by_ref = {
        (left_uid, left_role): source,
        (right_uid, right_role): displayed,
    }

    def nested(node_uid, data_role):
        return script(
            AverageOperation(dims=("x",)),
            start_label="Average input",
            active_name="data_0",
            script_inputs=(
                ScriptInput(
                    name="data_0",
                    label=data_role,
                    node_uid=node_uid,
                    node_snapshot_token=f"{left_uid}:{right_uid}:snapshot",
                    data_role=data_role,
                ),
            ),
        )

    spec = script(
        ScriptCodeOperation(label="Add", code="derived = left + right"),
        start_label="Add inputs",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="left",
                label="Source data",
                provenance_spec=nested(left_uid, left_role),
            ),
            ScriptInput(
                name="right",
                label="Displayed data",
                provenance_spec=nested(right_uid, right_role),
            ),
        ),
    )

    def resolve_live(script_input):
        data = data_by_ref.get((script_input.node_uid, script_input.data_role))
        return None if data is None else (data, script_input)

    result, _ = rebuild_script_provenance(
        spec,
        live_input_resolver=resolve_live,
        authorize=_authorize_execution,
    )

    xr.testing.assert_identical(result, source.mean("x") + displayed.mean("x"))


def test_rebuild_cache_separates_live_input_source_transforms() -> None:
    source = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("x", "y"))
    snapshot_token = f"snapshot-{id(source)}"

    def nested(start: int) -> ToolProvenanceSpec:
        source_spec = selection(IselOperation(kwargs={"x": slice(start, start + 2)}))
        return script(
            AverageOperation(dims=("y",)),
            start_label="Average selected input",
            active_name="data",
            script_inputs=(
                ScriptInput(
                    name="data",
                    node_uid="shared-node",
                    node_snapshot_token=snapshot_token,
                    source_spec=source_spec,
                ),
            ),
        )

    spec = script(
        ScriptCodeOperation(
            label="Stack selected inputs",
            code="result = xr.concat([left, right], dim='branch')",
        ),
        start_label="Stack selected inputs",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left", provenance_spec=nested(0)),
            ScriptInput(name="right", provenance_spec=nested(1)),
        ),
    )

    def resolve_live(script_input: ScriptInput):
        if script_input.node_uid != "shared-node":
            return None
        source_spec = script_input.parsed_source_spec()
        if source_spec is None:
            return source, script_input
        return source_spec.apply(source), script_input

    result, _ = rebuild_script_provenance(
        spec,
        live_input_resolver=resolve_live,
        authorize=_authorize_execution,
    )

    expected = xr.concat(
        [
            source.isel(x=slice(0, 2)).mean("y"),
            source.isel(x=slice(1, 3)).mean("y"),
        ],
        dim="branch",
    )
    xr.testing.assert_identical(result, expected)


def test_replay_cache_uses_resolved_live_input_snapshot() -> None:
    source = xr.DataArray([[1.0, 3.0], [5.0, 7.0]], dims=("x", "y"))
    displayed = xr.DataArray([[10.0, 30.0], [50.0, 70.0]], dims=("x", "y"))
    live_input = ScriptInput(
        name="data_0",
        label="Live input",
        node_uid="shared-node",
    )
    spec = script(
        AverageOperation(dims=("x",)),
        start_label="Average input",
        active_name="data_0",
        script_inputs=(live_input,),
    )
    replay_cache: dict[str, xr.DataArray] = {}

    def execute_with(data):
        resolved_input = live_input.model_copy(
            update={"node_snapshot_token": str(id(data))}
        )
        graph = compile_replay_graph(
            spec,
            live_input_resolver=lambda _item: (data, resolved_input),
        )
        return execute_replay_graph(graph, cache=replay_cache)

    xr.testing.assert_identical(execute_with(source), source.mean("x"))
    xr.testing.assert_identical(execute_with(displayed), displayed.mean("x"))


def test_replay_cache_separates_anonymous_external_inputs() -> None:
    source = xr.DataArray([[1.0, 3.0], [5.0, 7.0]], dims=("x", "y"))
    displayed = xr.DataArray([[10.0, 30.0], [50.0, 70.0]], dims=("x", "y"))
    replay_cache: dict[str, xr.DataArray] = {}

    for script_inputs in ((), (ScriptInput(name="data"),)):
        spec = script(
            AverageOperation(dims=("x",)),
            start_label="Average input",
            seed_code="result = data",
            active_name="result",
            script_inputs=script_inputs,
        )
        for data in (source, displayed):
            graph = compile_replay_graph(spec, external_inputs={"data": data})
            xr.testing.assert_identical(
                execute_replay_graph(graph, cache=replay_cache),
                data.mean("x"),
            )


def test_display_graph_uses_external_placeholders_for_unrecorded_inputs() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    weights = xr.DataArray([3.0, 4.0], dims="x")
    spec = script(
        ScriptCodeOperation(
            label="Apply weights",
            code="result = data * weights",
        ),
        start_label="Use external inputs",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="data",
                label="ImageTool 0",
                node_uid="data-node",
            ),
            ScriptInput(
                name="weights",
                label="ImageTool 1",
                node_uid="weights-node",
            ),
        ),
    )

    with pytest.raises(ReplayGraphError, match="recorded source provenance"):
        compile_replay_graph(spec)

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(
        code,
        {
            "data": data.copy(deep=True),
            "weights": weights.copy(deep=True),
        },
    )
    xr.testing.assert_identical(namespace["result"], data * weights)


def test_display_graph_reports_and_renames_required_caller_input() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    spec = script(
        ScriptCodeOperation(
            label="Offset source",
            code=(
                "def offset(source_data):\n"
                "    return data + source_data\n\n"
                "result = offset(1)"
            ),
        ),
        start_label="Use caller data",
        active_name="result",
    )

    graph = compile_replay_graph(spec, display=True)
    (source_key,) = graph.required_input_keys("data")
    code = emit_replay_code(
        graph,
        output_name="result",
        input_name_overrides={source_key: "source_data"},
    )

    assert graph.required_input_keys() == (source_key,)
    xr.testing.assert_identical(
        _exec_generated_code(code, {"source_data": data})["result"],
        data + 1,
    )


def test_display_graph_validates_input_name_overrides() -> None:
    spec = script(
        ScriptCodeOperation(label="Combine sources", code="result = left + right"),
        start_label="Use caller data",
        active_name="result",
    )
    graph = compile_replay_graph(spec, display=True)
    left_key, right_key = graph.required_input_keys()

    with pytest.raises(ReplayGraphError, match="unknown key"):
        emit_replay_code(graph, input_name_overrides={"missing": "source"})
    with pytest.raises(ReplayGraphError, match="valid Python identifier"):
        emit_replay_code(graph, input_name_overrides={left_key: "not valid"})
    with pytest.raises(ReplayGraphError, match="distinct names"):
        emit_replay_code(
            graph,
            input_name_overrides={left_key: "source", right_key: "source"},
        )
    with pytest.raises(ReplayGraphError, match="cannot use the requested name"):
        emit_replay_code(graph, input_name_overrides={left_key: "np"})


def test_display_graph_reports_no_input_for_nested_file_provenance() -> None:
    spec = script(
        ScriptCodeOperation(label="Offset source", code="result = data + 1"),
        start_label="Use recorded input",
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="data",
                label="File source",
                provenance_spec=_file_spec("scan.h5"),
            ),
        ),
    )

    graph = compile_replay_graph(spec, display=True)

    assert graph.required_input_keys() == ()
    assert "load_dataarray('scan.h5')" in emit_replay_code(
        graph,
        output_name="result",
    )


def test_display_graph_reuses_placeholder_for_same_live_input() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    snapshot_token = f"snapshot-{id(data)}"
    shared_input = {
        "label": "ImageTool 0",
        "node_uid": "shared-node",
        "node_snapshot_token": snapshot_token,
    }
    spec = script(
        ScriptCodeOperation(
            label="Add shared input",
            code="result = left + right",
        ),
        start_label="Use one external input twice",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left", **shared_input),
            ScriptInput(name="right", **shared_input),
        ),
    )

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"left": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["result"], data + data)


def test_display_graph_rejects_reserved_external_input_name() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    spec = script(
        start_label="Copy external input",
        seed_code="result = xr",
        active_name="result",
        script_inputs=(ScriptInput(name="xr"),),
    )

    xr.testing.assert_identical(
        replay_script_provenance(
            spec,
            {"xr": data},
            authorize=_authorize_execution,
        ),
        data,
    )
    with pytest.raises(ReplayGraphError, match="conflicts with a replay global"):
        compile_replay_graph(spec, display=True)
    assert spec.display_code() is None


def test_display_graph_does_not_mutate_external_input() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    original = data.copy(deep=True)
    spec = script(
        ScriptCodeOperation(
            label="Increment input",
            code="data[:] = data + 1.0\nresult = data",
        ),
        start_label="Use external input",
        active_name="result",
        script_inputs=(ScriptInput(name="data"),),
    )

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"data": data})

    xr.testing.assert_identical(namespace["result"], original + 1.0)
    xr.testing.assert_identical(data, original)


def test_rebuild_cache_separates_detached_recorded_inputs(
    tmp_path: pathlib.Path,
) -> None:
    source = xr.DataArray([[1.0, 3.0], [5.0, 7.0]], dims=("x", "y"))
    displayed = xr.DataArray([[10.0, 30.0], [50.0, 70.0]], dims=("x", "y"))
    source_path = tmp_path / "source.nc"
    displayed_path = tmp_path / "displayed.nc"
    source.to_netcdf(source_path)
    displayed.to_netcdf(displayed_path)

    def nested(path):
        return script(
            AverageOperation(dims=("x",)),
            start_label="Average input",
            active_name="data_0",
            script_inputs=(
                ScriptInput(
                    name="data_0",
                    label="Recorded input",
                    provenance_spec=_file_spec(path),
                ),
            ),
        )

    replay_cache: dict[str, xr.DataArray] = {}
    source_result, _ = rebuild_script_provenance(
        nested(source_path),
        live_input_resolver=lambda item: (source, item),
        cache=replay_cache,
    )
    displayed_result, _ = rebuild_script_provenance(
        nested(displayed_path),
        live_input_resolver=lambda item: (displayed, item),
        cache=replay_cache,
    )

    xr.testing.assert_identical(source_result, source.mean("x"))
    xr.testing.assert_identical(displayed_result, displayed.mean("x"))


def test_rebuild_executes_nested_script_once(monkeypatch) -> None:
    calls = 0

    def counted_random():
        nonlocal calls
        calls += 1
        return float(calls)

    monkeypatch.setattr(np.random, "random", counted_random)
    nested = script(
        ScriptCodeOperation(
            label="Create data",
            code=(
                "derived = xr.DataArray("
                "[np.random.random(), np.random.random()], dims=('x',))"
            ),
        ),
        start_label="Create input",
        active_name="derived",
    )
    spec = script(
        ScriptCodeOperation(label="Copy", code="derived = data"),
        start_label="Copy input",
        active_name="derived",
        script_inputs=(
            ScriptInput(
                name="data",
                label="Nested input",
                provenance_spec=nested,
            ),
        ),
    )

    result, _ = rebuild_script_provenance(spec, authorize=_authorize_execution)

    assert calls == 2
    xr.testing.assert_identical(
        result,
        xr.DataArray([1.0, 2.0], dims=("x",)),
    )


def test_rebuild_validates_all_inputs_before_nested_execution(monkeypatch) -> None:
    calls = 0

    def counted_random():
        nonlocal calls
        calls += 1
        return 1.0

    monkeypatch.setattr(np.random, "random", counted_random)
    nested = script(
        ScriptCodeOperation(
            label="Create data",
            code=("left = xr.DataArray([np.random.random(), 2.0], dims=('x',))"),
        ),
        start_label="Create left input",
        active_name="left",
    )
    invalid_script = script(
        start_label="Create right input",
        seed_code="right =",
        active_name="right",
    )
    spec = script(
        ScriptCodeOperation(label="Add inputs", code="result = left + right"),
        start_label="Add inputs",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left", provenance_spec=nested),
            ScriptInput(name="right", provenance_spec=invalid_script),
        ),
    )

    with pytest.raises(ReplayGraphError, match="not valid Python"):
        rebuild_script_provenance(spec, authorize=_authorize_execution)

    assert calls == 0


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
    rebuilt, rebuilt_spec = rebuild_script_provenance(
        script_spec, authorize=_authorize_execution
    )
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
    with pytest.raises(ReplayGraphError, match="recorded source"):
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
        authorize=_authorize_execution,
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
        authorize=_authorize_execution,
    )
    xr.testing.assert_identical(mixed_role_result, source_data + displayed_data)
    rebuilt_displayed = mixed_role_rebuilt.script_inputs[1].parsed_provenance_spec()
    assert rebuilt_displayed is not None
    assert rebuilt_displayed.script_inputs[0].data_role == "displayed"

    miss_calls = 0
    shared_file_input = ScriptInput(
        name="left",
        label="Closed file input",
        node_uid="same-uid",
        provenance_spec=file_spec,
    )
    shared_file_spec = script(
        ScriptCodeOperation(label="Add", code="derived = left + right"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            shared_file_input,
            shared_file_input.model_copy(update={"name": "right"}),
        ),
    )

    def miss_live(_script_input):
        nonlocal miss_calls
        miss_calls += 1
        return

    rebuilt_from_miss, _rebuilt_from_miss_spec = rebuild_script_provenance(
        shared_file_spec,
        live_input_resolver=miss_live,
        authorize=_authorize_execution,
    )
    xr.testing.assert_identical(rebuilt_from_miss, data * 2.0)
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
    with pytest.raises(ReplayGraphError, match="non-replayable"):
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
    with pytest.raises(ReplayGraphError, match="not self-contained"):
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

    expected = replay_script_provenance(spec, {}, authorize=_authorize_execution)
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

    expected = replay_script_provenance(spec, {}, authorize=_authorize_execution)
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

    expected = replay_script_provenance(spec, {}, authorize=_authorize_execution)
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

    replayed = replay_script_provenance(spec, {}, authorize=_authorize_execution)

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

    replayed = replay_script_provenance(
        spec, {"data": data}, authorize=_authorize_execution
    )
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

    replayed = replay_script_provenance(
        spec, {"data": source}, authorize=_authorize_execution
    )

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
        execute_replay_graph(graph, authorize=_authorize_execution),
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

    graph = compile_replay_graph(
        spec,
        external_inputs={"data_0": data},
        trusted_user_code=True,
    )
    script_nodes = [node for node in graph.nodes if node.kind == "script"]

    assert script_nodes
    assert any(
        input_name == "data_0"
        for node in script_nodes
        for input_name, _key in node.payload["bindings"]
    )
    xr.testing.assert_identical(
        execute_replay_graph(graph, authorize=_authorize_execution),
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

    graph = compile_replay_graph(
        spec,
        external_inputs={"data_0": data},
        trusted_user_code=True,
    )
    script_nodes = [node for node in graph.nodes if node.kind == "script"]

    assert len(script_nodes) == 1
    assert not any(node.kind == "operation" for node in graph.nodes)
    assert any(
        "derived = derived.qsel.mean(" in code
        for code in script_nodes[0].payload["codes"]
    )
    xr.testing.assert_identical(
        execute_replay_graph(graph, authorize=_authorize_execution),
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
        replay_script_provenance(
            spec, {"data_0": data}, authorize=_authorize_execution
        ),
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
    graph = compile_replay_graph(spec, trusted_user_code=True)

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
    xr.testing.assert_identical(
        execute_replay_graph(graph, authorize=_authorize_execution),
        expected,
    )


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
            "uses_implicit_framework_imports": (False,),
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
            "uses_implicit_framework_imports": (False,),
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

    runtime_graph = compile_replay_graph(spec, trusted_user_code=True)
    display_graph = compile_replay_graph(spec, display=True)
    display_code = emit_replay_code(display_graph, output_name="derived")
    runtime_expected = source.swap_dims({"x_idx": "x", "y_idx": "y"}).drop_vars(
        ("x_idx", "y_idx"), errors="ignore"
    )
    copied_expected = source.swap_dims({"x_idx": "x"}).drop_vars(
        "x_idx", errors="ignore"
    )

    assert "def _restore_image_tool_dimensions" not in display_code
    xr.testing.assert_identical(
        execute_replay_graph(runtime_graph, authorize=_authorize_execution),
        runtime_expected,
    )
    xr.testing.assert_identical(
        _exec_generated_code(display_code, {"xr": xr})["derived"],
        copied_expected,
    )


@pytest.mark.parametrize("source_kind", ["public_data", "selection"])
def test_replay_graph_runtime_restores_script_seeded_internal_source_views(
    source_kind: str,
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

    graph = compile_replay_graph(spec)
    code = emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code, {"erlab": erlab})

    assert code.count("def _restore_image_tool_dimensions") == 1
    assert "erlab.utils.array._restore_nonuniform_dims" not in code
    assert "erlab.interactive.imagetool.slicer" not in code
    xr.testing.assert_identical(
        execute_replay_graph(graph, authorize=_authorize_execution),
        source.qsel(x=0.2),
    )
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
    assert "def _restore_image_tool_dimensions" not in code
    assert code.index("from __future__ import annotations") < code.index(
        "import xarray as xr"
    )
    assert namespace["__doc__"] == "Load the replay source."
    xr.testing.assert_identical(namespace["derived"], source.qsel(x=0.2))


def test_extension_code_preserves_module_prologue(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "local_routine.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Create data",
        seed_code=(
            '"""Generated extension workflow."""\n'
            "from __future__ import annotations\n"
            "import xarray as xr\n"
            "annotation: MissingType | None = None\n"
            "result = xr.DataArray([3.0])"
        ),
        active_name="result",
    ).append_replay_stage(full_data(_extension_routine_operation()))

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code)

    assert code.startswith(
        '"""Generated extension workflow."""\nfrom __future__ import annotations'
    )
    assert namespace["__doc__"] == "Generated extension workflow."
    assert namespace["__annotations__"] == {"annotation": "MissingType | None"}
    xr.testing.assert_identical(namespace["result"], xr.DataArray([6.0]))


def test_extension_binding_named_like_internal_restore_helper(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "_restore_image_tool_dimensions.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    source = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("x_idx",),
        coords={"x": ("x_idx", [0.0, 0.2, 1.0])},
    )
    source_dict = source.to_dict()
    extension_operation = ExtensionRoutineOperation(
        script_name=script_path.name,
        source_hash="a" * 64,
        routine_id="scale",
        routine_name="Scale",
        parameters={},
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Create data",
        seed_code=(
            f"import xarray as xr\nresult = xr.DataArray.from_dict({source_dict!r})"
        ),
        active_name="result",
    ).append_replay_stage(
        full_data(extension_operation, RestoreNonuniformDimsOperation())
    )

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code)
    scale_call = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "scale"
    )
    scale_call_source = ast.get_source_segment(code, scale_call)

    assert scale_call_source is not None
    assert "\n" not in scale_call_source
    assert "def _restore_image_tool_dimensions" not in code
    xr.testing.assert_identical(namespace["result"], source * 2.0)


def test_extension_code_preserves_external_framework_alias_input(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "local_routine.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    source = xr.DataArray([1.0, 3.0, 2.0], dims=("x",))
    gaussian = GaussianFilterOperation(sigma={"x": 0.5})
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Use caller data",
        seed_code="result = era",
        active_name="result",
    ).append_replay_stage(full_data(_extension_routine_operation(), gaussian))

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"era": source})

    xr.testing.assert_identical(namespace["result"], gaussian.apply(source * 2.0))


def test_extension_code_hoists_nested_script_prologues(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "local_routine.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    data_path = tmp_path / "source.nc"
    xr.DataArray([4.0]).to_netcdf(data_path)
    nested_source = file_load(
        start_label="Load nested source",
        seed_code=(
            '"""Load the nested source."""\n'
            "from __future__ import annotations\n"
            "import xarray as xr\n"
            f"derived = xr.load_dataarray({str(data_path)!r})"
        ),
        file_load_source=_file_replay_source(data_path),
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Use nested source",
        seed_code=(
            '"""Process the nested source."""\n'
            "from __future__ import annotations\n"
            "annotation: int | None = None\n"
            "result = source"
        ),
        active_name="result",
        script_inputs=(
            ScriptInput(
                name="source",
                label="Nested source",
                provenance_spec=nested_source,
            ),
        ),
    ).append_replay_stage(full_data(_extension_routine_operation()))

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code)

    assert code.startswith(
        '"""Load the nested source."""\nfrom __future__ import annotations'
    )
    assert code.count("from __future__ import annotations") == 1
    string_expressions = [
        statement.value.value
        for statement in ast.parse(code).body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    ]
    assert string_expressions == [
        "Load the nested source.",
        "Process the nested source.",
    ]
    assert namespace["__doc__"] == "Load the nested source."
    assert namespace["__annotations__"] == {"annotation": "int | None"}
    xr.testing.assert_identical(namespace["result"], xr.DataArray([8.0]))


def test_extension_code_preserves_nested_load_script_input(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "local_routine.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Use caller data",
        seed_code=("def get_data():\n    return load_script\n\nresult = get_data()"),
        active_name="result",
    ).append_replay_stage(full_data(_extension_routine_operation()))
    source = xr.DataArray([3.0])

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"load_script": source})

    xr.testing.assert_identical(namespace["result"], source * 2.0)


def test_extension_code_preserves_later_load_script_input(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "local_routine.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    spec = script(
        ScriptCodeOperation(
            label="Use caller data",
            code="result = load_script",
        ),
        start_label="Create data",
        seed_code="result = xr.DataArray([1.0])",
        active_name="result",
    ).append_replay_stage(full_data(_extension_routine_operation()))
    source = xr.DataArray([3.0])

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"load_script": source, "xr": xr})

    xr.testing.assert_identical(namespace["result"], source * 2.0)


def test_extension_code_does_not_capture_later_local_load_script_binding(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "local_routine.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Use local data",
        seed_code=(
            "def get_data():\n"
            "    return load_script\n\n"
            "load_script = xr.DataArray([3.0])\n"
            "result = get_data()"
        ),
        active_name="result",
    ).append_replay_stage(full_data(_extension_routine_operation()))

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"xr": xr})

    xr.testing.assert_identical(namespace["result"], xr.DataArray([6.0]))


def test_replay_graph_preserves_user_restore_named_function(
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
            "derived = data_0"
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

    assert "def _restore_image_tool_dimensions_2(array):" not in code
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

    replayed = replay_script_provenance(
        spec, {"data": source}, authorize=_authorize_execution
    )
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

    graph = compile_replay_graph(
        spec,
        external_inputs={"data_0": data},
        trusted_user_code=True,
    )
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
    graph = compile_replay_graph(
        spec,
        external_inputs={"data_0": data},
        trusted_user_code=True,
    )

    with xr.set_options(use_numbagg=True):
        replayed = execute_replay_graph(graph, authorize=_authorize_execution)
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
    namespace = _exec_generated_code(code, {"era": erlab.analysis})

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
    namespace = _exec_generated_code(code, {"xr": xr})

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
    namespace = _exec_generated_code(code, {"np": np, "xr": xr})

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
    namespace = _exec_generated_code(code, {"np": np, "xr": xr})

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
    assert provenance_requires_code_trust(spec)
    with pytest.raises(ReplayGraphError, match="unsupported Import"):
        compile_replay_graph(spec, external_inputs={"data": data})

    result = replay_script_provenance(
        spec,
        {"data": data},
        authorize=_authorize_execution,
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
            authorize=_authorize_execution,
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
    assert provenance_requires_code_trust(spec)
    with pytest.raises(ReplayGraphError, match="not trusted"):
        rebuild_script_provenance(spec)

    result, rebuilt = rebuild_script_provenance(
        spec,
        authorize=_authorize_execution,
    )

    assert rebuilt.kind == "script"
    xr.testing.assert_identical(result, (source + 1) * 2)


def test_live_inputs_bypass_unreachable_unsafe_fallback() -> None:
    left = xr.DataArray([1.0, 2.0], dims="x")
    right = xr.DataArray([10.0, 20.0], dims="x")
    unsafe_right_spec = script(
        ScriptCodeOperation(
            label="Create right input",
            code="import os\nright = xr.DataArray([100.0, 200.0], dims='x')",
        ),
        start_label="Create right input",
        active_name="right",
    )
    spec = script(
        ScriptCodeOperation(label="Add inputs", code="result = left + right"),
        start_label="Add inputs",
        seed_code="result = left",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left", label="Left input"),
            ScriptInput(
                name="right",
                label="Right input",
                provenance_spec=unsafe_right_spec,
            ),
        ),
    )
    left_input, right_input = spec.script_inputs

    def resolve_live(script_input: ScriptInput):
        if script_input is left_input:
            return left, script_input
        if script_input is right_input:
            return right, script_input
        return None

    assert script_provenance_replayable(spec)
    assert not script_provenance_replayable(unsafe_right_spec)
    assert script_provenance_replayable(unsafe_right_spec, allow_code=True)
    assert script_provenance_replayable(
        spec,
        live_input_resolver=resolve_live,
    )

    result, _rebuilt_spec = rebuild_script_provenance(
        spec,
        live_input_resolver=resolve_live,
        authorize=_authorize_execution,
    )

    xr.testing.assert_identical(result, left + right)


@pytest.mark.parametrize("right_fallback", [None, full_data()])
def test_replay_rejects_unresolved_recorded_input(
    right_fallback: ToolProvenanceSpec | None,
) -> None:
    left = xr.DataArray([1.0, 2.0], dims="x")
    spec = script(
        ScriptCodeOperation(label="Add inputs", code="result = left + right"),
        start_label="Add inputs",
        seed_code="result = left",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left"),
            ScriptInput(name="right", provenance_spec=right_fallback),
        ),
    )

    assert script_provenance_replayable(
        spec,
        external_input_names={"left"},
    )
    with pytest.raises(ReplayGraphError):
        rebuild_script_provenance(
            spec,
            live_input_resolver=lambda item: (
                (left, item) if item.name == "left" else None
            ),
        )


def test_live_input_resolution_uses_exact_input_across_nested_scopes() -> None:
    live_shared = xr.DataArray([1.0, 2.0], dims="x")
    recorded_shared = script(
        ScriptCodeOperation(
            label="Create nested input",
            code=(
                "import os\n"
                "shared = xr.DataArray([10.0, 20.0], dims='x') "
                "+ int(os.path.exists(os.devnull))"
            ),
        ),
        start_label="Create nested input",
        active_name="shared",
    )
    nested_spec = script(
        ScriptCodeOperation(label="Copy nested input", code="nested = shared"),
        start_label="Copy nested input",
        active_name="nested",
        script_inputs=(
            ScriptInput(
                name="shared",
                label="Nested shared input",
                provenance_spec=recorded_shared,
            ),
        ),
    )
    spec = script(
        ScriptCodeOperation(
            label="Add inputs",
            code="derived = shared + nested",
        ),
        start_label="Add inputs",
        active_name="derived",
        script_inputs=(
            ScriptInput(name="shared", label="Live shared input"),
            ScriptInput(
                name="nested",
                label="Nested input",
                provenance_spec=nested_spec,
            ),
        ),
    )
    outer_shared_input = spec.script_inputs[0]

    def resolve_live(script_input: ScriptInput):
        if script_input is outer_shared_input:
            return live_shared, script_input
        return None

    assert not script_provenance_replayable(recorded_shared)
    assert script_provenance_replayable(recorded_shared, allow_code=True)

    result, _rebuilt_spec = rebuild_script_provenance(
        spec,
        live_input_resolver=resolve_live,
        authorize=_authorize_execution,
    )

    xr.testing.assert_identical(
        result,
        live_shared + xr.DataArray([11.0, 21.0], dims="x"),
    )


def test_rebuild_nested_live_inputs_uses_first_resolution() -> None:
    left = xr.DataArray([1.0, 2.0], dims="x")
    right = xr.DataArray([10.0, 20.0], dims="x")
    nested = script(
        ScriptCodeOperation(label="Add inputs", code="result = left + right"),
        start_label="Add nested inputs",
        seed_code="result = left",
        active_name="result",
        script_inputs=(ScriptInput(name="left"), ScriptInput(name="right")),
    )
    spec = script(
        start_label="Copy nested result",
        seed_code="result = nested",
        active_name="result",
        script_inputs=(ScriptInput(name="nested", provenance_spec=nested),),
    )
    calls: dict[str, int] = {}

    def resolve_live(script_input: ScriptInput):
        calls[script_input.name] = calls.get(script_input.name, 0) + 1
        if script_input.name == "nested":
            return None
        if calls[script_input.name] > 1:
            return None
        data = left if script_input.name == "left" else right
        return data, script_input

    result, _rebuilt = rebuild_script_provenance(
        spec,
        live_input_resolver=resolve_live,
        authorize=_authorize_execution,
    )

    xr.testing.assert_identical(result, left + right)
    assert calls == {"nested": 1, "left": 1, "right": 1}


def test_live_input_bypasses_invalid_recorded_fallback() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    live_input = ScriptInput(
        name="data",
        node_uid="live-data",
        provenance_spec={"kind": "invalid"},
    )
    spec = script(
        start_label="Copy live data",
        seed_code="result = data",
        active_name="result",
        script_inputs=(live_input,),
    )

    def resolve_live(script_input: ScriptInput):
        return (data, script_input) if script_input is live_input else None

    assert script_provenance_replayable(
        spec,
        live_input_resolver=resolve_live,
    )
    result, _rebuilt_spec = rebuild_script_provenance(
        spec,
        live_input_resolver=resolve_live,
        authorize=_authorize_execution,
    )

    xr.testing.assert_identical(result, data)


def test_external_input_bypasses_invalid_recorded_fallback() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    spec = script(
        start_label="Copy external data",
        seed_code="result = data",
        active_name="result",
        script_inputs=(ScriptInput(name="data", provenance_spec={"kind": "invalid"}),),
    )

    assert script_provenance_replayable(
        spec,
        external_input_names={"data"},
    )
    xr.testing.assert_identical(
        replay_script_provenance(
            spec,
            {"data": data},
            authorize=_authorize_execution,
        ),
        data,
    )


def test_invalid_later_fallback_precedes_any_input_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    loaded = xr.DataArray([1.0, 2.0], dims="x")
    load_calls: list[str] = []

    def load_file_source(load_source, **_kwargs) -> xr.DataArray:
        load_calls.append(load_source.path)
        return loaded

    monkeypatch.setattr(_execution, "_load_file_source_data", load_file_source)
    source_path = tmp_path / "unused.nc"
    source_path.touch()
    file_spec = _file_spec(source_path)
    invalid_fallback = script(
        ScriptCodeOperation(label="Invalid code", code="result ="),
        start_label="Copy invalid input",
        active_name="result",
        script_inputs=(ScriptInput(name="child", provenance_spec=file_spec),),
    )

    with pytest.raises(ReplayGraphError, match="not valid Python"):
        rebuild_script_inputs(
            (
                ScriptInput(name="loaded", provenance_spec=file_spec),
                ScriptInput(name="invalid", provenance_spec=invalid_fallback),
            )
        )

    assert load_calls == []


def test_missing_later_callable_precedes_any_file_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.nc"
    source_path.touch()
    loaded = xr.DataArray([1.0, 2.0], dims="x")
    load_calls: list[str] = []

    def load_file_source(load_source, **_kwargs) -> xr.DataArray:
        load_calls.append(load_source.path)
        return loaded

    monkeypatch.setattr(_execution, "_load_file_source_data", load_file_source)
    file_spec = _file_spec(source_path)
    load_source = file_spec.file_load_source
    assert load_source is not None
    replay_call = load_source.replay_call
    assert replay_call is not None
    missing_callable_spec = file_spec.model_copy(
        update={
            "file_load_source": load_source.model_copy(
                update={
                    "replay_call": replay_call.model_copy(
                        update={
                            "kind": "callable",
                            "target": "definitely_missing_replay_loader.load",
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(ReplayGraphError, match="loader is not available"):
        rebuild_script_inputs(
            (
                ScriptInput(name="first", provenance_spec=file_spec),
                ScriptInput(name="second", provenance_spec=missing_callable_spec),
            )
        )

    assert load_calls == []


def test_live_input_classification_enforces_rebuild_depth_limit() -> None:
    data = xr.DataArray([1.0, 2.0], dims="x")
    spec = script(
        ScriptCodeOperation(label="Add inputs", code="result = left + right"),
        start_label="Add live inputs",
        seed_code="result = left",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left", node_uid="left"),
            ScriptInput(name="right", node_uid="right"),
        ),
    )
    for level in range(21):
        spec = script(
            start_label=f"Copy nested input {level}",
            seed_code="result = child",
            active_name="result",
            script_inputs=(ScriptInput(name="child", provenance_spec=spec),),
        )

    def resolve_live(script_input: ScriptInput):
        if script_input.node_uid in {"left", "right"}:
            return data, script_input
        return None

    assert not script_provenance_replayable(
        spec,
        live_input_resolver=resolve_live,
    )
    with pytest.raises(ReplayGraphError, match="maximum reload depth"):
        rebuild_script_provenance(
            spec,
            live_input_resolver=resolve_live,
        )


def test_rebuild_preflight_work_scales_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = 12
    spec = script(
        start_label="Create leaf input",
        seed_code="result = xr.DataArray([1.0, 2.0], dims='x')",
        active_name="result",
    )
    for level in range(1, depth):
        spec = script(
            start_label=f"Copy nested input {level}",
            seed_code="result = child",
            active_name="result",
            script_inputs=(ScriptInput(name="child", provenance_spec=spec),),
        )

    compile_calls = 0
    compile_graph = _execution.compile_replay_graph

    def counted_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return compile_graph(*args, **kwargs)

    monkeypatch.setattr(_execution, "compile_replay_graph", counted_compile)

    result, _rebuilt = rebuild_script_provenance(
        spec,
        authorize=_authorize_execution,
    )

    xr.testing.assert_identical(result, xr.DataArray([1.0, 2.0], dims="x"))
    assert compile_calls <= 2 * depth


def test_external_input_name_does_not_leak_into_nested_fallback(
    tmp_path: pathlib.Path,
) -> None:
    outer_left = xr.DataArray([1.0, 2.0], dims="x")
    inner_left = xr.DataArray([100.0, 200.0], dims="x")
    inner_left_path = tmp_path / "inner-left.nc"
    inner_left.to_netcdf(inner_left_path)
    right_spec = script(
        ScriptCodeOperation(label="Offset input", code="right = left + 10.0"),
        start_label="Create right",
        active_name="right",
        script_inputs=(
            ScriptInput(name="left", provenance_spec=_file_spec(inner_left_path)),
        ),
    )
    spec = script(
        ScriptCodeOperation(label="Add inputs", code="result = left + right"),
        start_label="Add inputs",
        seed_code="result = left",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left"),
            ScriptInput(name="right", provenance_spec=right_spec),
        ),
    )
    expected = outer_left + inner_left + 10.0

    assert script_provenance_replayable(
        spec,
        external_input_names={"left"},
    )
    xr.testing.assert_identical(
        replay_script_provenance(
            spec,
            {"left": outer_left},
            authorize=_authorize_execution,
        ),
        expected,
    )
    code = typing.cast("str", spec.display_code())
    xr.testing.assert_identical(
        _exec_generated_code(code, {"left": outer_left})["result"],
        expected,
    )


def test_display_graph_rejects_nested_store_over_external_input() -> None:
    left = xr.DataArray([1.0, 2.0], dims="x")
    right = xr.DataArray([10.0, 20.0], dims="x")
    right_spec = script(
        ScriptCodeOperation(
            label="Create right",
            code="left = xr.DataArray([10.0, 20.0], dims='x')\nright = left",
        ),
        start_label="Create right",
        active_name="right",
    )
    spec = script(
        ScriptCodeOperation(label="Add inputs", code="result = left + right"),
        start_label="Add inputs",
        seed_code="result = left",
        active_name="result",
        script_inputs=(
            ScriptInput(name="left"),
            ScriptInput(name="right", provenance_spec=right_spec),
        ),
    )

    xr.testing.assert_identical(
        replay_script_provenance(
            spec,
            {"left": left},
            authorize=_authorize_execution,
        ),
        left + right,
    )
    with pytest.raises(ReplayGraphError, match="provenance name 'left'"):
        emit_replay_code(compile_replay_graph(spec, display=True))
    assert spec.display_code() is None


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


def test_script_file_source_replays_registered_extension_loader(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "local_loader.py"
    script_path.write_text(
        """from pathlib import Path
import xarray as xr
from erlab.extensions import loader

@loader()
def load_data(path: Path) -> xr.DataArray:
    return xr.DataArray([float(path.read_text())])
"""
    )
    data_path = tmp_path / "value.txt"
    data_path.write_text("4")
    load_source = FileLoadSource(
        path=str(data_path),
        loader_label="Extension Loader",
        loader_text="local_loader.py: Load data",
        kwargs_text="(none)",
        replay_call=FileReplayCall(
            kind="extension_loader",
            target=script_path.name,
            source_hash="a" * 64,
            capability_id="load_data",
            selection=FileDataSelection(kind="dataarray"),
        ),
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_loader(script_path),
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Load extension data",
        seed_code="derived = None",
        active_name="derived",
        file_load_source=load_source,
    )

    graph = compile_replay_graph(spec, display=True)
    code = emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    assert code.count("from erlab.extensions import load_script") == 1
    xr.testing.assert_identical(namespace["derived"], xr.DataArray([4.0]))


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("value = 1", (None, "", "value = 1")),
        (
            "from __future__ import annotations\nvalue = 1",
            (None, "from __future__ import annotations", "value = 1"),
        ),
        (
            '"""Module docs."""\nvalue = 1',
            ('"""Module docs."""', "", "value = 1"),
        ),
    ],
)
def test_partition_module_prologue_without_combined_docstring_and_future(
    code: str,
    expected: tuple[str | None, str, str],
) -> None:
    assert _graph._partition_module_prologue(code) == expected


def test_script_output_rename_handles_identity_and_annotated_binding() -> None:
    codes = ["result: object = data", "final = result"]

    assert (
        _graph._replace_script_output_identifiers(
            codes,
            previous_name="result",
            output_name="result",
        )
        == codes
    )
    assert _graph._replace_script_output_identifiers(
        codes,
        previous_name="result",
        output_name="processed",
    ) == ["processed: object = data", "final = processed"]


def test_extension_script_uses_safe_binding_for_invalid_module_name(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "123.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine(id="scale")
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    spec = script(
        _extension_routine_operation(),
        start_label="Create data",
        seed_code="result = data",
        active_name="result",
    )
    source = xr.DataArray([2.0])

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"data": source})

    assert "extension_script = load_script" in code
    xr.testing.assert_identical(namespace["result"], source * 2)


@pytest.mark.parametrize("active_name", ["era", "np", "xr"])
def test_script_code_flushes_before_framework_named_output(active_name: str) -> None:
    source = xr.DataArray([1.0, 2.0], dims="x")
    operation = GaussianFilterOperation(sigma={"x": 0.5})
    spec = script(
        ScriptCodeOperation(label="Use input", code=f"{active_name} = data + 0"),
        operation,
        start_label="Start from data",
        active_name=active_name,
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(
        code,
        {"data": source, "era": erlab.analysis},
    )

    xr.testing.assert_identical(namespace[active_name], operation.apply(source))


def test_script_output_rename_starts_at_an_earlier_conditional_binding() -> None:
    renamed = _graph._replace_script_output_identifiers(
        ["if False:\n    result = data\nresult = data + 1"],
        previous_name="result",
        output_name="processed",
    )
    source = xr.DataArray([1.0])

    namespace = _exec_generated_code(renamed[0], {"data": source})

    xr.testing.assert_identical(namespace["processed"], source + 1)


def test_script_output_rename_waits_for_a_later_binding() -> None:
    renamed = _graph._replace_script_output_identifiers(
        ["intermediate = data", "result = intermediate + 1"],
        previous_name="result",
        output_name="processed",
    )
    source = xr.DataArray([1.0])

    namespace = _exec_generated_code("\n".join(renamed), {"data": source})

    assert renamed[0] == "intermediate = data"
    xr.testing.assert_identical(namespace["processed"], source + 1)


def test_extension_code_avoids_reserved_caller_load_script_bindings(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "local_routine.py"
    script_path.write_text(
        """import xarray as xr
from erlab.extensions import routine

@routine()
def scale(data: xr.DataArray) -> xr.DataArray:
    return data * 2.0
"""
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    spec = ToolProvenanceSpec(
        kind="script",
        start_label="Use caller data",
        seed_code=(
            "input_data = None\n"
            "input_data_2 = None\n"
            "def get_data():\n"
            "    return load_script\n\n"
            "load_script = get_data()"
        ),
        active_name="load_script",
    ).append_replay_stage(full_data(_extension_routine_operation()))
    source = xr.DataArray([3.0])

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"load_script": source})

    assert "input_data_3 = load_script" in code
    xr.testing.assert_identical(namespace["result"], source * 2.0)


def test_replay_graph_avoids_reserved_copy_suffixes(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "shared.nc"
    source = xr.DataArray(np.arange(3.0), dims="x")
    source.to_netcdf(path)
    source_spec = _file_spec(path)

    def branch(delta: float) -> ToolProvenanceSpec:
        return script(
            ScriptCodeOperation(
                label="Change source",
                code=f"source_2 = None\nresult = source + {delta}",
            ),
            start_label="Run branch",
            active_name="result",
            script_inputs=(
                ScriptInput(
                    name="source",
                    label="Source",
                    provenance_spec=source_spec,
                ),
            ),
        )

    spec = script(
        ScriptCodeOperation(
            label="Combine branches",
            code="total = left + right",
        ),
        start_label="Combine branches",
        active_name="total",
        script_inputs=(
            ScriptInput(
                name="left",
                label="Left",
                provenance_spec=branch(1.0),
            ),
            ScriptInput(
                name="right",
                label="Right",
                provenance_spec=branch(2.0),
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code)

    assert "source_3 = loaded_data.copy(deep=True)" in code
    xr.testing.assert_identical(namespace["total"], source * 2 + 3)


def test_emit_replay_code_preserves_ambiguous_import_chunks() -> None:
    spec = script(
        ScriptCodeOperation(
            label="Calculate result",
            code="from math import *\nresult = sqrt(4)",
        ),
        start_label="Calculate result",
        active_name="result",
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code)

    assert namespace["result"] == 2.0


@pytest.mark.parametrize(
    "operation",
    [
        ScriptCodeOperation(
            label="Unavailable code",
            code="result = data",
            copyable=False,
        ),
        ScriptCodeOperation(label="Unavailable code", code=None),
    ],
)
def test_compile_replay_graph_rejects_noncopyable_script_code(
    operation: ScriptCodeOperation,
) -> None:
    spec = script(
        operation,
        start_label="Use input",
        active_name="result",
    )

    with pytest.raises(ReplayGraphError, match="non-replayable code"):
        compile_replay_graph(spec, display=True)


@pytest.mark.parametrize("method", ["isel", "qsel", "sel"])
def test_display_graph_omits_empty_legacy_selection_step(method: str) -> None:
    source = xr.DataArray([1.0, 2.0], dims="x", name="source")
    spec = script(
        ScriptCodeOperation(
            label=f"{method}()",
            code=f"derived = derived.{method}()",
        ),
        RenameOperation(name="renamed"),
        start_label="Use input",
        seed_code="derived = data",
        active_name="derived",
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"data": source})

    assert f".{method}()" not in code
    xr.testing.assert_identical(namespace["derived"], source.rename("renamed"))


def test_display_code_omits_alias_after_mutating_operation() -> None:
    source = xr.DataArray([1.0], attrs={"sample_workfunction": 4.5})
    spec = script(
        KspaceWorkFunctionOperation(work_function=4.4),
        start_label="Use input",
        seed_code="result = data",
        active_name="result",
    )

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"data": source})

    assert "result = data" not in code
    assert namespace["data"].kspace.work_function == 4.4
