import ast
import hashlib
import pathlib
import re
import symtable
import sys
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
    _CodeAnalyzer,
    _nonuniform_restore_support_code,
    _replace_code_identifiers,
    _simplify_display_code,
    _symbol_table_dependencies,
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
    SelectedProvenanceScope,
    _code_has_scoped_definition,
    _code_name_accesses,
    _compact_replay_temp_names,
    _file_seed_code_parts,
    _group_framework_imports,
    _import_binding_targets,
    _inline_adjacent_replay_assignments,
    _inline_simple_name_aliases,
    _inline_single_use_replay_names,
    _is_future_import,
    _leading_top_level_imports,
    _operation_replay_code,
    _remove_noop_assignments,
    _remove_unused_generated_copies,
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
    selected_provenance_code,
    selected_steps_code,
    selected_steps_spec,
)
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    FileLoadSource,
    FileReplayCall,
    ReplayStage,
    ReplayStep,
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _ProvenanceStepRef,
    compose_full_provenance,
    file_load,
    full_data,
    parse_script_inputs,
    public_data,
    rebase_script_inputs_node_uids,
    recipe,
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
    ShiftFromInputOperation,
    SortCoordOrderOperation,
    SqueezeOperation,
    TransposeOperation,
)
from erlab.interactive.imagetool._provenance._trust import (
    provenance_code_trust_entries,
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


def _selected_script_scope(
    code: str,
    input_name: str,
    output_name: str,
    links: dict[str, int],
) -> SelectedProvenanceScope:
    return SelectedProvenanceScope(
        selected_steps_spec(
            (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
            input_name=input_name,
            output_name=output_name,
        ),
        links,
    )


def test_captured_input_trust_inventory_matches_replay_bindings() -> None:
    source = xr.DataArray([1.0, 2.0], dims="x")
    original = source.copy(deep=True)
    spec = script(
        start_label="Apply copied script",
        active_name="result",
        primary_input="data",
        script_inputs=(
            ScriptInput(
                name="data",
                revision_id="captured",
                follow_source=False,
                provenance_spec=script(
                    ScriptCodeOperation(
                        label="Previous script", code="result = derived + 1"
                    ),
                    start_label="Previous data",
                    seed_code="derived = data",
                    active_name="result",
                ),
            ),
        ),
        steps=(
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Offset", code="result = derived + 2"
                ),
                context_names=("derived",),
            ),
        ),
    )

    def resolve(item: ScriptInput):
        return (source, item) if item.revision_id == "captured" else None

    entries = provenance_code_trust_entries(
        spec, location_prefix="copied", live_input_resolver=resolve
    )
    authorization = _authorize_execution(entries)
    graph = compile_replay_graph(spec, live_input_resolver=resolve)
    actual = execute_replay_graph(graph, authorize=lambda _entries: authorization)
    xr.testing.assert_identical(actual, source + 2)
    xr.testing.assert_identical(source, original)

    changed = spec.model_copy(
        update={
            "steps": (
                spec.steps[0].model_copy(
                    update={
                        "operation": ScriptCodeOperation(
                            label="Changed", code="result = derived + 3"
                        )
                    }
                ),
            )
        }
    )
    with pytest.raises(ReplayGraphError, match="not trusted"):
        execute_replay_graph(
            compile_replay_graph(changed, live_input_resolver=resolve),
            authorize=lambda _entries: authorization,
        )


def test_concrete_bound_input_operation_replays_without_stored_code() -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0.0, 1.0, 2.0]},
        name="data",
    )
    shift_data = xr.DataArray([0.0, 1.0], dims="y", coords={"y": [0, 1]})
    operation = ShiftFromInputOperation(along="x", negate=True)
    spec = script(
        start_label="Shift ImageTool data",
        active_name="shifted",
        script_inputs=(ScriptInput(name="data"), ScriptInput(name="shift")),
        primary_input="data",
        steps=(
            ReplayStep(
                operation=operation,
                input_bindings={"shift": "shift"},
            ),
        ),
    )

    reparsed = ToolProvenanceSpec.model_validate(spec.model_dump(mode="json"))
    assert reparsed == spec
    assert reparsed.primary_input == "data"
    assert reparsed.steps[0].input_bindings == {"shift": "shift"}
    assert provenance_requires_code_trust(reparsed) is False

    graph = compile_replay_graph(
        reparsed,
        external_inputs={"data": data, "shift": shift_data},
    )
    operation_node = next(node for node in graph.nodes if node.kind == "operation")
    assert operation_node.payload["input_bindings"] == (
        ("shift", operation_node.parents[2]),
    )
    expected = erlab.analysis.transform.shift(data, -shift_data, along="x")
    xr.testing.assert_identical(execute_replay_graph(graph), expected)

    display_graph = compile_replay_graph(reparsed, display=True)
    code = emit_replay_code(display_graph, output_name="shifted")
    namespace = _exec_generated_code(
        code,
        {"data": data, "shift": shift_data, "era": erlab.analysis},
    )
    xr.testing.assert_identical(namespace["shifted"], expected)

    prefix = reparsed._prefix_before_ref(
        _ProvenanceStepRef("operation", operation_index=0)
    )
    assert prefix.active_name == "data"
    assert tuple(item.name for item in prefix.script_inputs) == ("data",)
    prefix_graph = compile_replay_graph(
        prefix,
        external_inputs={"data": data},
    )
    xr.testing.assert_identical(execute_replay_graph(prefix_graph), data)

    start = reparsed._prefix_through_ref(_ProvenanceStepRef("start"))
    deleted = reparsed._replace_operation_ref(
        _ProvenanceStepRef("operation", operation_index=0),
        (),
    )
    for source_only in (start, deleted):
        assert source_only.active_name == "data"
        assert tuple(item.name for item in source_only.script_inputs) == ("data",)
        graph = compile_replay_graph(source_only, external_inputs={"data": data})
        xr.testing.assert_identical(execute_replay_graph(graph), data)

    replaced = reparsed._replace_operation_ref(
        _ProvenanceStepRef("operation", operation_index=0),
        (AverageOperation(dims=("y",)),),
    )
    assert tuple(item.name for item in replaced.script_inputs) == ("data",)


def test_bound_input_step_validates_its_slots_and_primary_input() -> None:
    operation = ShiftFromInputOperation(along="x")
    step = ReplayStep(operation=operation, input_bindings={"shift": "shift"})

    class _DuplicateSlotsOperation(ShiftFromInputOperation):
        def input_slots(self) -> tuple[str, ...]:
            return ("shift", "shift")

    with pytest.raises(ValueError, match="must match the operation input slots"):
        ReplayStep(operation=operation)
    with pytest.raises(ValueError, match="must match the operation input slots"):
        ReplayStep(operation=operation, input_bindings={"weights": "shift"})
    with pytest.raises(ValueError, match="must match the operation input slots"):
        step.model_copy(update={"input_bindings": {}})
    with pytest.raises(ValueError, match="must match the operation input slots"):
        step.model_copy(update={"operation": AverageOperation(dims=("x",))})
    with pytest.raises(ValueError, match="must match the operation input slots"):
        ReplayStep.model_validate({"operation": operation, "input_bindings": None})
    with pytest.raises(TypeError, match="must be a mapping"):
        ReplayStep.model_validate({"operation": operation, "input_bindings": []})
    with pytest.raises(ValueError, match="must not use None"):
        ReplayStep.model_validate(
            {"operation": operation, "input_bindings": {"shift": None}}
        )
    with pytest.raises(ValueError, match="distinct Python identifiers"):
        ReplayStep(
            operation=_DuplicateSlotsOperation(along="x"),
            input_bindings={"shift": "shift"},
        )
    with pytest.raises(ValueError, match="cannot define source input policies"):
        ReplayStep(
            operation=operation,
            input_bindings={"shift": "shift"},
            input_policy="current",
        )
    with pytest.raises(ValueError, match="primary input"):
        script(
            start_label="Shift",
            active_name="shifted",
            script_inputs=(ScriptInput(name="data"), ScriptInput(name="shift")),
            steps=(
                ReplayStep(
                    operation=operation,
                    input_bindings={"shift": "shift"},
                ),
            ),
        )
    with pytest.raises(ValueError, match="unknown script input"):
        script(
            start_label="Shift",
            active_name="shifted",
            script_inputs=(ScriptInput(name="data"),),
            primary_input="data",
            steps=(
                ReplayStep(
                    operation=operation,
                    input_bindings={"shift": "missing"},
                ),
            ),
        )
    spec = recipe(
        step,
        start_label="Shift",
        active_name="shifted",
        script_inputs=(ScriptInput(name="data"), ScriptInput(name="shift")),
        primary_input="data",
    )
    with pytest.raises(ValueError, match="cannot also define seed code"):
        spec.model_copy(update={"seed_code": "shifted = data"})
    assert spec.model_copy(update={"primary_input": "data"}) == spec
    with pytest.raises(ValueError, match="distinct script input names"):
        script(
            start_label="Duplicate inputs",
            active_name="derived",
            script_inputs=(ScriptInput(name="data"), ScriptInput(name="data")),
            primary_input="data",
        )
    with pytest.raises(ValueError, match="primary_input must name a script input"):
        script(
            start_label="Unknown primary input",
            active_name="derived",
            script_inputs=(ScriptInput(name="data"),),
            primary_input="missing",
        )
    with pytest.raises(ValueError, match="cannot define primary_input"):
        _file_spec("source.nc").model_copy(update={"primary_input": "data"})


def test_recipe_is_a_code_free_concrete_authoring_boundary() -> None:
    operation = ShiftFromInputOperation(along="x")
    shift_step = ReplayStep(
        operation=operation,
        input_bindings={"shift": "shift"},
    )
    spec = recipe(
        shift_step,
        start_label="Shift",
        active_name="shifted",
        script_inputs=(ScriptInput(name="data"), ScriptInput(name="shift")),
        primary_input="data",
    )

    assert spec.kind == "script"
    assert spec.seed_code is None
    assert spec.steps == (shift_step,)
    with pytest.raises(ValueError, match="must define replay steps"):
        recipe(
            start_label="Empty",
            active_name="result",
            script_inputs=(ScriptInput(name="data"),),
            primary_input="data",
        )
    with pytest.raises(TypeError, match="cannot contain script operations"):
        recipe(
            ReplayStep(
                operation=ScriptCodeOperation(label="Code", code="result = data")
            ),
            start_label="Code",
            active_name="result",
            script_inputs=(ScriptInput(name="data"),),
            primary_input="data",
        )
    with pytest.raises(ValueError, match="require direct replay steps"):
        recipe(
            ReplayStep(
                operation=AverageOperation(dims=("x",)),
                input_policy="current",
            ),
            start_label="Context-backed operation",
            active_name="result",
            script_inputs=(ScriptInput(name="data"),),
            primary_input="data",
        )


def test_replay_graph_rejects_unavailable_bound_and_nonlive_inputs() -> None:
    class _NonLiveOperation(ToolProvenanceOperation):
        live_applicable: typing.ClassVar[bool] = False

    bound = recipe(
        ReplayStep(
            operation=ShiftFromInputOperation(along="x"),
            input_bindings={"shift": "shift"},
        ),
        start_label="Shift",
        active_name="shifted",
        script_inputs=(ScriptInput(name="data"), ScriptInput(name="shift")),
        primary_input="data",
    )
    unavailable_bound = types.SimpleNamespace(
        kind="script",
        active_name=bound.active_name,
        primary_input=bound.primary_input,
        seed_code=None,
        script_inputs=(ScriptInput(name="data"),),
        script_context_bindings=(),
        steps=bound.steps,
    )
    with pytest.raises(ReplayGraphError, match="input binding is unavailable"):
        _validate_script_provenance(unavailable_bound)

    nonlive = script(
        start_label="Non-live operation",
        seed_code="derived = data",
        active_name="derived",
        steps=(ReplayStep(operation=_NonLiveOperation()),),
    )
    with pytest.raises(ReplayGraphError, match="non-replayable operation"):
        _validate_script_provenance(nonlive, external_input_names={"data"})


def test_compose_parent_into_concrete_recipe_primary_input(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray(
        np.arange(15.0).reshape(3, 5),
        dims=("y", "x"),
        coords={"y": np.arange(3), "x": np.arange(5.0)},
    )
    shift_data = xr.DataArray(
        [0.0, 0.2, -0.1],
        dims="y",
        coords={"y": np.arange(3)},
    )
    data_path = tmp_path / "data.nc"
    shift_path = tmp_path / "shift.nc"
    data.to_netcdf(data_path)
    shift_data.to_netcdf(shift_path)
    local = recipe(
        ReplayStep(
            operation=ShiftFromInputOperation(along="x"),
            input_bindings={"shift": "shift"},
        ),
        start_label="Shift",
        active_name="shifted",
        script_inputs=(
            ScriptInput(name="data"),
            ScriptInput(name="shift", provenance_spec=_file_spec(shift_path)),
        ),
        primary_input="data",
    )

    composed = compose_full_provenance(_file_spec(data_path), local)

    assert composed is not None
    assert composed.primary_input == "data"
    assert composed.seed_code is None
    assert not any(
        isinstance(operation, ScriptCodeOperation) for operation in composed.operations
    )
    primary = next(
        item for item in composed.script_inputs if item.name == composed.primary_input
    )
    assert primary.parsed_provenance_spec() == _file_spec(data_path)
    expected = erlab.analysis.transform.shift(data, shift_data, along="x")
    rebuilt, _ = rebuild_script_provenance(composed)
    xr.testing.assert_identical(rebuilt, expected)
    code = composed.derivation_code()
    assert code is not None
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["shifted"], expected)


def test_compose_concrete_recipe_with_concrete_descendants() -> None:
    data = xr.DataArray(
        np.arange(15.0).reshape(3, 5),
        dims=("y", "x"),
        coords={"y": np.arange(3), "x": np.arange(5.0)},
    )
    first_shift = xr.DataArray([0.0, 0.2, -0.1], dims="y", coords={"y": np.arange(3)})
    second_shift = xr.DataArray([0.1, 0.0, -0.2], dims="y", coords={"y": np.arange(3)})
    parent = recipe(
        ReplayStep(
            operation=ShiftFromInputOperation(along="x"),
            input_bindings={"shift": "first_shift"},
        ),
        start_label="First shift",
        active_name="first",
        script_inputs=(
            ScriptInput(name="data"),
            ScriptInput(name="first_shift"),
        ),
        primary_input="data",
    )
    local = recipe(
        ReplayStep(
            operation=ShiftFromInputOperation(along="x"),
            input_bindings={"shift": "second_shift"},
        ),
        start_label="Second shift",
        active_name="second",
        script_inputs=(
            ScriptInput(name="first"),
            ScriptInput(name="second_shift"),
        ),
        primary_input="first",
    )

    processed = compose_full_provenance(
        parent,
        full_data(GaussianFilterOperation(sigma={"x": 0.5})),
    )
    nested = compose_full_provenance(parent, local)

    assert processed is not None
    assert nested is not None
    assert [operation.op for operation in processed.operations] == [
        "shift_from_input",
        "gaussian_filter",
    ]
    assert not any(
        isinstance(operation, ScriptCodeOperation)
        for operation in (*processed.operations, *nested.operations)
    )
    nested_primary = next(
        item for item in nested.script_inputs if item.name == nested.primary_input
    )
    assert nested_primary.parsed_provenance_spec() == parent

    first = erlab.analysis.transform.shift(data, first_shift, along="x")
    expected_processed = erlab.analysis.image.gaussian_filter(first, sigma={"x": 0.5})
    processed_result = replay_script_provenance(
        processed,
        {"data": data, "first_shift": first_shift},
    )
    xr.testing.assert_identical(processed_result, expected_processed)
    processed_code = processed.derivation_code()
    assert processed_code is not None
    processed_namespace = _exec_generated_code(
        processed_code,
        {"data": data, "first_shift": first_shift, "era": erlab.analysis},
    )
    xr.testing.assert_identical(processed_namespace["derived"], expected_processed)

    expected_nested = erlab.analysis.transform.shift(first, second_shift, along="x")
    nested_code = nested.derivation_code()
    assert nested_code is not None
    nested_namespace = _exec_generated_code(
        nested_code,
        {
            "data": data,
            "first_shift": first_shift,
            "second_shift": second_shift,
            "era": erlab.analysis,
        },
    )
    xr.testing.assert_identical(nested_namespace["second"], expected_nested)


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


def test_canonical_replay_keys_stay_bounded_for_long_chains() -> None:
    first = _graph._canonical_key("source", {"left": 1, "right": 2})
    reordered = _graph._canonical_key("source", {"right": 2, "left": 1})

    assert first == reordered
    assert first.startswith("canonical-sha256:")
    assert len(first) == 81
    assert first != _graph._canonical_key("other", {"left": 1, "right": 2})

    keys: list[str] = []
    parent = first
    for index in range(256):
        parent = _graph._canonical_key(
            "operation",
            {
                "context": parent,
                "operation": {"index": index},
                "parent": parent,
            },
        )
        keys.append(parent)

    assert len(keys) == len(set(keys))
    assert {len(key) for key in keys} == {81}


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


def test_script_input_parsing_validates_boundary_values() -> None:
    with pytest.raises(TypeError, match="source spec must be"):
        ScriptInput(name="data", source_spec=object())
    with pytest.raises(TypeError, match="must be a live ToolProvenanceSpec"):
        ScriptInput(
            name="data",
            source_spec=script(start_label="Use data", active_name="data"),
        )

    assert parse_script_inputs(None) == ()
    for invalid in ('"data"', 1):
        with pytest.raises(TypeError, match="must be a sequence"):
            parse_script_inputs(invalid)

    script_inputs = (ScriptInput(name="data", node_uid="source"),)
    assert rebase_script_inputs_node_uids(script_inputs, {}) == script_inputs


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

    for recorded_spec, message in (
        (None, "does not contain recorded source provenance"),
        (full_data(), "does not contain reloadable script or file provenance"),
    ):
        with pytest.raises(ReplayGraphError, match=message):
            rebuild_script_inputs(
                (
                    ScriptInput(
                        name="detached",
                        label="Detached input",
                        provenance_spec=recorded_spec,
                    ),
                )
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
        return source, script_input

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


@pytest.mark.parametrize(
    "input_names",
    [
        {"left": "left", "right": "right", "reference": "reference"},
        {"left": "right", "right": "left", "reference": "reference"},
        {"left": "right", "right": "reference", "reference": "left"},
        {"left": "reference", "right": "left", "reference": "right"},
        {"left": "right", "right": "reference", "reference": "offsets"},
    ],
    ids=["identity", "swap", "cycle", "reverse-cycle", "chain"],
)
@pytest.mark.parametrize(
    "binding_scope", ["expression", "global", "default", "rebind", "self_assignment"]
)
@pytest.mark.parametrize("external_inputs", [False, True])
def test_display_graph_publishes_renamed_inputs_together(
    input_names: dict[str, str], binding_scope: str, external_inputs: bool
) -> None:
    expression = "left - right + reference * 10"
    if binding_scope == "expression":
        code = f"result = {expression}"
    elif binding_scope == "rebind":
        code = "left = right + 1\nresult = left + reference * 10"
    elif binding_scope == "self_assignment":
        code = f"left = left\nresult = {expression}"
    else:
        arguments = (
            "left=left, right=right, reference=reference"
            if binding_scope == "default"
            else ""
        )
        code = f"def combine({arguments}):\n    return {expression}\nresult = combine()"
    graph = compile_replay_graph(
        script(
            ScriptCodeOperation(label="Combine inputs", code=code),
            start_label="Use caller inputs",
            active_name="result",
            script_inputs=(
                tuple(ScriptInput(name=name) for name in input_names)
                if external_inputs
                else ()
            ),
        ),
        display=True,
    )
    overrides = {
        key: input_names[name]
        for name in input_names
        for key in graph.required_input_keys(name)
    }
    emitted = emit_replay_code(
        graph, output_name="result", input_name_overrides=overrides
    )
    namespace = {
        name: xr.DataArray(np.full((3, 2), value), dims=("x", "y"))
        for name, value in {
            "left": 10.0,
            "right": 3.0,
            "reference": 2.0,
            "offsets": 100.0,
        }.items()
    }
    expected = (
        namespace[input_names["right"]] + 1 + namespace[input_names["reference"]] * 10
        if binding_scope == "rebind"
        else namespace[input_names["left"]]
        - namespace[input_names["right"]]
        + namespace[input_names["reference"]] * 10
    )
    actual = _exec_generated_code(emitted, namespace)

    xr.testing.assert_identical(actual["result"], expected)
    assert ".copy(" not in emitted
    assert "_itool_replay_" not in emitted


@pytest.mark.parametrize("reverse_inputs", [False, True])
@pytest.mark.parametrize("external_inputs", [False, True])
def test_display_graph_captures_inputs_before_alias_chains(
    reverse_inputs: bool, external_inputs: bool
) -> None:
    input_names = {
        "left": "offsets",
        "right": "left",
        "reference": "right",
        "offsets": "reference",
    }
    if reverse_inputs:
        input_names = dict(reversed(tuple(input_names.items())))
    graph = compile_replay_graph(
        script(
            ScriptCodeOperation(
                label="Combine inputs",
                code="left = right + 1\nresult = left + reference * 10 + offsets * 100",
            ),
            start_label="Use caller inputs",
            active_name="result",
            script_inputs=(
                tuple(ScriptInput(name=name) for name in input_names)
                if external_inputs
                else ()
            ),
        ),
        display=True,
    )
    overrides = {
        key: input_names[name]
        for name in input_names
        for key in graph.required_input_keys(name)
    }
    emitted = emit_replay_code(
        graph, output_name="result", input_name_overrides=overrides
    )
    namespace = {
        name: xr.DataArray(np.full((3, 2), value), dims=("x", "y"))
        for name, value in {
            "left": 10.0,
            "right": 3.0,
            "reference": 2.0,
            "offsets": 100.0,
        }.items()
    }
    expected = (
        namespace["left"] + 1 + namespace["right"] * 10 + namespace["reference"] * 100
    )
    actual = _exec_generated_code(emitted, namespace)

    xr.testing.assert_identical(actual["result"], expected)
    assert ".copy(" not in emitted
    assert "_itool_replay_" not in emitted


@pytest.mark.parametrize("external_inputs", [False, True])
def test_display_graph_publishes_swapped_output_aliases_together(
    external_inputs: bool,
) -> None:
    graph = compile_replay_graph(
        script(
            ScriptCodeOperation(label="Combine", code="result = left - right"),
            start_label="Use caller inputs",
            active_name="result",
            script_inputs=(
                (ScriptInput(name="left"), ScriptInput(name="right"))
                if external_inputs
                else ()
            ),
        ),
        display=True,
    )
    (left_key,) = graph.required_input_keys("left")
    (right_key,) = graph.required_input_keys("right")
    graph.add_alias("left", right_key)
    graph.add_alias("right", left_key)
    code = emit_replay_code(graph, output_name="result", include_all_aliases=True)
    left = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "y"))
    right = left / 2
    actual = _exec_generated_code(code, {"left": left, "right": right})

    xr.testing.assert_identical(actual["result"], left - right)
    xr.testing.assert_identical(actual["left"], right)
    xr.testing.assert_identical(actual["right"], left)
    assert ".copy(" not in code


def test_display_graph_disambiguates_and_overrides_external_input_names() -> None:
    graph = ReplayGraph(display=True)
    script_code = "def add_values():\n    return left + right\n\nresult = add_values()"
    caller_key = graph.add_node(
        "caller",
        "caller_input",
        payload={"name": "source"},
    )
    external_key = graph.add_node(
        "external",
        "external_input",
        payload={"name": "source"},
    )
    graph.output_key = graph.add_node(
        "script",
        "script",
        parents=(caller_key, external_key),
        cacheable=False,
        payload={
            "active_name": "result",
            "bindings": (("left", caller_key), ("right", external_key)),
            "codes": (script_code,),
            "document_codes": (script_code,),
            "external_names": ((),),
            "hoist_imports": (False,),
            "uses_implicit_framework_imports": (False,),
        },
    )
    source = xr.DataArray([1.0, 2.0], dims="x")
    other = xr.DataArray([3.0, 4.0], dims="x")

    code = emit_replay_code(graph, output_name="result")
    namespace = _exec_generated_code(
        code,
        {"source": source, "source_2": other},
    )
    xr.testing.assert_identical(namespace["result"], source + other)

    overridden_code = emit_replay_code(
        graph,
        output_name="result",
        input_name_overrides={external_key: "weights"},
    )
    overridden = _exec_generated_code(
        overridden_code,
        {"source": source, "weights": other},
    )
    xr.testing.assert_identical(overridden["result"], source + other)


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


def test_display_graph_keeps_explicit_input_mutation() -> None:
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

    replayed = replay_script_provenance(
        spec, {"data": data}, authorize=_authorize_execution
    )
    xr.testing.assert_identical(replayed, original + 1.0)
    xr.testing.assert_identical(data, original)

    code = emit_replay_code(
        compile_replay_graph(spec, display=True),
        output_name="result",
    )
    namespace = _exec_generated_code(code, {"data": data})

    xr.testing.assert_identical(namespace["result"], original + 1.0)
    assert namespace["result"] is data
    assert ".copy(" not in code


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
    with pytest.raises(ReplayGraphError, match="reloadable script or file"):
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


def test_rebuild_script_inputs_rejects_unavailable_file_sources_before_loading(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "source.nc"
    path.touch()
    base_spec = _file_spec(path)
    load_source = base_spec.file_load_source
    assert load_source is not None

    extension_unavailable = base_spec.model_copy(
        update={
            "file_load_source": load_source.model_copy(
                update={
                    "replay_call": FileReplayCall(
                        kind="extension_loader",
                        target="missing_loader.py",
                        source_hash="a" * 64,
                        capability_id="load_data",
                        selection=FileDataSelection(kind="dataarray"),
                    )
                }
            )
        }
    )
    monkeypatch.setattr(
        _execution,
        "_registered_script_capability_status",
        lambda *_args: "disabled",
    )
    monkeypatch.setattr(
        _execution,
        "_load_file_source_data",
        lambda *_args, **_kwargs: pytest.fail(
            "unavailable sources must fail before loading"
        ),
    )

    for spec, message in (
        (_file_spec(tmp_path / "missing.nc"), "source file is not available"),
        (_erlab_file_spec(path, "missing-loader"), "loader is not available"),
        (extension_unavailable, "extension loader is not available"),
    ):
        with pytest.raises(ReplayGraphError, match=message):
            rebuild_script_inputs((ScriptInput(name="source", provenance_spec=spec),))


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


def test_live_input_classification_supports_deep_replay() -> None:
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
    for level in range(40):
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

    assert script_provenance_replayable(
        spec,
        live_input_resolver=resolve_live,
    )
    result, rebuilt = rebuild_script_provenance(
        spec,
        live_input_resolver=resolve_live,
        authorize=_authorize_execution,
    )
    xr.testing.assert_identical(result, data + data)
    assert rebuilt.kind == "script"


def test_replay_rejects_recorded_input_cycle() -> None:
    spec = script(
        start_label="Use child",
        active_name="child",
        primary_input="child",
        script_inputs=(ScriptInput(name="child"),),
    )
    with pytest.raises(ReplayGraphError, match="input cycle"):
        rebuild_script_provenance(spec, _input_provenance_resolver=lambda _item: spec)


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
        (
            "from math import *\nresult = left",
            "opaque namespace import",
        ),
        (
            "np = left\nresult = left",
            "overwrite replay global 'np'",
        ),
    ],
)
def test_display_graph_rejects_unsafe_multi_input_namespace_changes(
    code: str,
    message: str,
) -> None:
    spec = script(
        ScriptCodeOperation(label="Use input", code=code),
        start_label="Use input",
        active_name="result",
        script_inputs=(ScriptInput(name="left"),),
    )

    with pytest.raises(ReplayGraphError, match=message):
        emit_replay_code(compile_replay_graph(spec, display=True))


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


@pytest.mark.parametrize("input_alias", [None, "derived"])
def test_selected_steps_code_preserves_script_inputs_and_function_scope(
    input_alias: str | None,
) -> None:
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "y"), name="scan")
    reference = source + 1.0
    receiver = input_alias or "data"
    steps = (
        ReplayStep(
            operation=ScriptCodeOperation(
                label="Normalize against reference",
                code=(
                    "def normalize():\n"
                    f"    return {receiver} / reference\n"
                    "result = normalize()"
                ),
            )
        ),
        ReplayStep(operation=AverageOperation(dims=("x",))),
    )

    code = selected_steps_code(
        steps, input_name="data", output_name="result", input_alias=input_alias
    )
    namespace = {"data": source, "reference": reference}
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(
        namespace["result"], (source / reference).qsel.mean("x")
    )
    assert ".copy(" not in code


@pytest.mark.parametrize("function_scope", ["global", "default"])
def test_selected_notebook_scopes_publish_bindings_before_later_functions(
    function_scope: str,
) -> None:
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "y"))
    offsets = source + 3.0
    definition = (
        "def normalize():\n    return derived + 1"
        if function_scope == "global"
        else "def normalize(values=derived):\n    return values + 1"
    )
    scopes = (
        SelectedProvenanceScope(
            selected_steps_spec(
                (
                    ReplayStep(
                        operation=ScriptCodeOperation(
                            label="Normalize sample",
                            code=(
                                "events.append('A')\n"
                                f"{definition}\n"
                                "derived = normalize()"
                            ),
                        )
                    ),
                ),
                input_name="derived",
                output_name="derived",
            ),
            {},
        ),
        SelectedProvenanceScope(
            selected_steps_spec(
                (
                    ReplayStep(
                        operation=ScriptCodeOperation(
                            label="Prepare correction",
                            code="events.append('X')\ncorrection = offsets * 2",
                        )
                    ),
                ),
                input_name="offsets",
                output_name="correction",
            ),
            {},
        ),
        SelectedProvenanceScope(
            selected_steps_spec(
                (
                    ReplayStep(
                        operation=ScriptCodeOperation(
                            label="Scale normalized sample",
                            code="events.append('B')\nderived = normalize() * 2",
                        )
                    ),
                ),
                input_name="derived",
                output_name="derived",
            ),
            {"derived": 0},
        ),
    )
    code = selected_provenance_code(scopes)
    events: list[str] = []
    namespace = _exec_generated_code(
        code, {"derived": source, "offsets": offsets, "events": events}
    )

    expected = (source + (2 if function_scope == "global" else 1)) * 2
    xr.testing.assert_identical(namespace["derived"], expected)
    xr.testing.assert_identical(namespace["correction"], offsets * 2)
    assert events == ["A", "X", "B"]
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


@pytest.mark.parametrize("recorded", [False, True])
def test_selected_notebook_scopes_bind_current_or_recorded_inputs(
    recorded: bool,
) -> None:
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "y"))
    correction = source + 100.0
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare sample",
                    code="derived_input = data * 10\nderived = data + 1",
                )
            ),
        ),
        input_name="data",
        output_name="derived",
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Apply correction", code="shifted = data + derived"
                )
            ),
        ),
        input_name="data",
        output_name="shifted",
        script_inputs=(
            (
                ScriptInput(
                    name="derived", revision_id="correction", follow_source=False
                ),
            )
            if recorded
            else ()
        ),
    )
    code = selected_provenance_code(
        (SelectedProvenanceScope(first, {}), SelectedProvenanceScope(last, {"data": 0}))
    )
    namespace = _exec_generated_code(code, {"data": source, "derived": correction})

    expected_input = correction if recorded else source + 1
    xr.testing.assert_identical(namespace["shifted"], source + 1 + expected_input)
    xr.testing.assert_identical(namespace["derived_input"], source * 10)
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


@pytest.mark.parametrize("output_name", ["derived", "result"])
@pytest.mark.parametrize("correction_code", ["derived * 0", "derived - derived"])
def test_selected_notebook_scopes_restore_retained_function_globals(
    output_name: str,
    correction_code: str,
) -> None:
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    offsets = xr.DataArray([3.0, 4.0, 5.0], dims="x")

    def scope(code, input_name, output_name, links, input_alias=None):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
                input_alias=input_alias,
            ),
            links,
        )

    scopes = (
        scope(
            "def normalize():\n    return derived + 1\nderived = normalize()",
            "data",
            "derived",
            {},
            "derived",
        ),
        scope(
            f"derived = offsets * 10\ncorrection = {correction_code}",
            "offsets",
            "correction",
            {},
        ),
        scope(
            f"{output_name} = normalize() * 2 + correction",
            "derived",
            output_name,
            {"derived": 0, "correction": 1},
        ),
    )
    original_specs = tuple(scope.spec.model_dump(mode="json") for scope in scopes)
    code = selected_provenance_code(scopes)
    namespace = _exec_generated_code(code, {"data": source, "offsets": offsets})

    xr.testing.assert_identical(namespace[output_name], (source + 2) * 2)
    assert (
        tuple(scope.spec.model_dump(mode="json") for scope in scopes) == original_specs
    )
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


def test_selected_notebook_scopes_restore_recursive_function_alias_globals() -> None:
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    code = selected_provenance_code(
        (
            _selected_script_scope(
                "def normalize(values, count=1):\n"
                "    if count:\n"
                "        return normalize(values + 1, count - 1)\n"
                "    return values\n"
                "normalize_saved = normalize\nprepared = normalize(data)",
                "data",
                "prepared",
                {},
            ),
            _selected_script_scope(
                "normalize = offsets * 0\ncorrection = offsets * 0",
                "offsets",
                "correction",
                {},
            ),
            _selected_script_scope(
                "result = normalize_saved(prepared) + correction",
                "prepared",
                "result",
                {"prepared": 0, "correction": 1},
            ),
        )
    )
    namespace = _exec_generated_code(code, {"data": source, "offsets": source + 100})

    xr.testing.assert_identical(namespace["result"], source + 2)


@pytest.mark.parametrize("replace_function", [False, True])
def test_selected_notebook_scopes_keep_primary_namespace_versions(
    replace_function: bool,
) -> None:
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    offsets = xr.DataArray([3.0, 4.0, 5.0], dims="x")

    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    scopes = (
        scope(
            "def normalize():\n    return data + 1\nderived = data + 1",
            "data",
            "derived",
            {},
        ),
        scope(
            "data = offsets * 10\n"
            + (
                "normalize = data * 0\ncorrection = normalize - normalize"
                if replace_function
                else "correction = data - data"
            ),
            "offsets",
            "correction",
            {},
        ),
        scope(
            "result = normalize() + correction",
            "derived",
            "result",
            {"derived": 0, "correction": 1},
        ),
        scope(
            "data = data + 5\nfinal = normalize()",
            "result",
            "final",
            {"result": 2},
        ),
        scope("data = offsets * 100\nextra = data - data", "offsets", "extra", {}),
        scope(
            "last = normalize() + extra",
            "final",
            "last",
            {"final": 3, "extra": 4},
        ),
    )
    original_specs = tuple(scope.spec.model_dump(mode="json") for scope in scopes)
    code = selected_provenance_code(scopes)
    namespace = _exec_generated_code(code, {"data": source, "offsets": offsets})
    xr.testing.assert_identical(namespace["result"], source + 1)
    xr.testing.assert_identical(namespace["final"], source + 6)
    xr.testing.assert_identical(namespace["last"], source + 6)
    assert (
        tuple(scope.spec.model_dump(mode="json") for scope in scopes) == original_specs
    )
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


def test_selected_notebook_scopes_restore_globals_and_publish_inputs_together() -> None:
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    offsets = xr.DataArray([3.0, 4.0, 5.0], dims="x")

    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    code = selected_provenance_code(
        (
            scope(
                "def normalize():\n    return data + 1\nderived = data + 1",
                "data",
                "derived",
                {},
            ),
            scope("data = offsets * 2", "offsets", "data", {}),
            scope(
                "result = normalize() + correction",
                "derived",
                "result",
                {"derived": 0, "correction": 1},
            ),
        )
    )
    namespace = _exec_generated_code(code, {"data": source, "offsets": offsets})
    xr.testing.assert_identical(namespace["result"], source + 1 + offsets * 2)
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


@pytest.mark.parametrize(
    ("definition", "external_name", "scale", "offset"),
    [
        (
            "def compute():\n"
            "    edge = data.mean()\n"
            "    def normalize():\n"
            "        return data / edge\n"
            "    return normalize()",
            None,
            None,
            None,
        ),
        (
            "def compute(edge=data):\n"
            "    def normalize():\n"
            "        return edge + 1\n"
            "    return normalize()",
            None,
            1,
            1,
        ),
        ("def compute(edge=data):\n    return (lambda: edge + 1)()", None, 1, 1),
        (
            "def compute(edge=data):\n    return sum(edge + item for item in range(2))",
            None,
            2,
            1,
        ),
        (
            "def compute(edge=data):\n"
            "    return [(lambda: edge + item)() for item in range(2)][1]",
            None,
            1,
            1,
        ),
        (
            "def compute(edge=data):\n"
            "    def normalize():\n"
            "        nonlocal edge\n"
            "        edge = edge + 1\n"
            "        return edge\n"
            "    return normalize()",
            None,
            1,
            1,
        ),
        (
            "def compute(edge=data):\n"
            "    class Normalizer:\n"
            "        edge = 100\n"
            "        def normalize(self):\n"
            "            return edge + 1\n"
            "    return Normalizer().normalize()",
            None,
            1,
            1,
        ),
        (
            "def compute(edge=data):\n"
            "    class Normalizer:\n"
            "        def normalize(self):\n"
            "            return edge + int(__class__.__name__ == 'Normalizer')\n"
            "    return Normalizer().normalize()",
            None,
            1,
            1,
        ),
        (
            "def compute(edge=data):\n"
            "    def normalize():\n"
            "        global edge\n"
            "        return edge + 1\n"
            "    return normalize()",
            "edge",
            1,
            11,
        ),
        (
            "def compute(edge=data):\n"
            "    def normalize():\n"
            "        return reference + 1\n"
            "    return normalize()",
            "reference",
            1,
            21,
        ),
        (
            "def compute():\n"
            "    class Normalizer:\n"
            "        edge = 100\n"
            "        def normalize(self):\n"
            "            return edge + 1\n"
            "    return Normalizer().normalize()",
            "edge",
            1,
            11,
        ),
    ],
    ids=[
        "local",
        "parameter",
        "lambda",
        "generator",
        "lambda-in-comprehension",
        "nonlocal",
        "class-outer-closure",
        "class-cell",
        "explicit-global-shadow",
        "implicit-global",
        "class-method-global",
    ],
)
def test_selected_notebook_scopes_keep_lexical_closures_out_of_caller_inputs(
    definition: str, external_name: str | None, scale: int | None, offset: int | None
) -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    producer = scope(f"{definition}\nprepared = compute()", "data", "prepared", {})
    expected_inputs = {"data"} | ({external_name} if external_name else set())
    assert {item.name for item in producer.spec.script_inputs} == expected_inputs
    code = selected_provenance_code(
        (
            producer,
            scope(
                "edge = offsets * 0\nreference = offsets * 0\n"
                "item = offsets * 0\ncorrection = offsets * 0",
                "offsets",
                "correction",
                {},
            ),
            scope(
                "result = compute() + correction",
                "prepared",
                "result",
                {"prepared": 0, "correction": 1},
            ),
        )
    )
    source = xr.DataArray(np.arange(1.0, 7.0).reshape(3, 2), dims=("x", "eV"))
    inputs = {"data": source, "offsets": source + 100}
    if external_name:
        inputs[external_name] = source + (10 if external_name == "edge" else 20)
    namespace = _exec_generated_code(code, inputs)
    expected = (
        source / source.mean()
        if scale is None
        else source * scale + typing.cast("int", offset)
    )

    xr.testing.assert_identical(namespace["prepared"], expected)
    xr.testing.assert_identical(namespace["result"], expected)
    captures = {
        statement.targets[0].id
        for statement in ast.parse(code).body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id.endswith("_input")
    }
    assert captures == ({f"{external_name}_input"} if external_name else set())
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires type parameters")
@pytest.mark.parametrize(
    "definition",
    [
        "def compute[T](values=data):\n"
        "    def normalize():\n"
        "        return values + (T.__name__ == 'T')\n"
        "    return normalize()",
        "class Normalizer[T]:\n"
        "    def normalize(self, values):\n"
        "        return values + (T.__name__ == 'T')\n"
        "def compute():\n"
        "    return Normalizer().normalize(data)",
    ],
    ids=["function", "class"],
)
def test_selected_notebook_scopes_keep_type_parameters_in_lexical_scopes(
    definition: str,
) -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    producer = scope(f"{definition}\nprepared = compute()", "data", "prepared", {})
    assert {item.name for item in producer.spec.script_inputs} == {"data"}
    code = selected_provenance_code(
        (
            producer,
            scope(
                "T = offsets * 0\ncorrection = offsets * 0", "offsets", "correction", {}
            ),
            scope(
                "result = compute() + correction",
                "prepared",
                "result",
                {"prepared": 0, "correction": 1},
            ),
        )
    )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(code, {"data": source, "offsets": source + 100})

    xr.testing.assert_identical(namespace["result"], source + 1)
    assert "T_input" not in code
    assert ".copy(" not in code


def test_symbol_table_dependencies_keep_free_names_of_the_requested_scope() -> None:
    code = (
        "def outer(edge):\n"
        "    def inner():\n"
        "        return edge + 1\n"
        "    return inner\n"
        "result = outer(3)()"
    )
    module = symtable.symtable(code, "<closure>", "exec")
    outer = next(
        child for child in module.get_children() if child.get_name() == "outer"
    )
    inner = next(child for child in outer.get_children() if child.get_name() == "inner")
    namespace = _exec_generated_code(code)

    assert namespace["result"] == 4
    assert _symbol_table_dependencies(outer) == set()
    assert _symbol_table_dependencies(inner) == {"edge"}


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires type parameters")
def test_symbol_table_dependencies_keep_type_alias_parameters_in_lexical_scope() -> (
    None
):
    code = "type Pair[T] = tuple[T, T]\nresult = Pair.__name__"
    symbols = symtable.symtable(code, "<type alias>", "exec")
    namespace = _exec_generated_code(code)

    assert namespace["result"] == "Pair"
    assert _symbol_table_dependencies(symbols) == {"Pair", "tuple"}


def test_selected_notebook_scopes_do_not_capture_unused_inputs() -> None:
    spec = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(label="Step", code="result = offsets * 2")
            ),
        ),
        input_name="data",
        output_name="result",
    )
    code = selected_provenance_code(
        (SelectedProvenanceScope(spec, {}), SelectedProvenanceScope(spec, {}))
    )
    offsets = xr.DataArray([3.0, 4.0, 5.0], dims="x")
    namespace = _exec_generated_code(code, {"offsets": offsets})
    xr.testing.assert_identical(namespace["result"], offsets * 2)
    assert "data" not in code


@pytest.mark.parametrize(
    "invalid_code", ["result = (", "data = 1\nglobal data\nresult = data"]
)
@pytest.mark.parametrize("seed", [False, True])
def test_selected_notebook_scopes_report_invalid_code(
    invalid_code: str, seed: bool
) -> None:
    spec = script(
        start_label="Invalid code",
        active_name="result",
        primary_input=None if seed else "data",
        script_inputs=() if seed else (ScriptInput(name="data"),),
        seed_code=invalid_code if seed else None,
        steps=(
            ()
            if seed
            else (
                ReplayStep(
                    operation=ScriptCodeOperation(label="Invalid", code=invalid_code)
                ),
            )
        ),
    )
    with pytest.raises(
        ReplayGraphError, match="Script replay code is not valid Python"
    ):
        selected_provenance_code((SelectedProvenanceScope(spec, {}),))


@pytest.mark.parametrize(
    "overwrite",
    [
        "globals()['derived'] = offsets",
        "locals()['derived'] = offsets",
        "vars()['derived'] = offsets",
        "exec('derived = offsets')",
        "namespace = globals\nnamespace()['derived'] = offsets",
        "from builtins import exec as execute\nexecute('derived = offsets')",
        "import builtins as runtime\nruntime.exec('derived = offsets')",
        "def overwrite():\n    global derived\n    derived = offsets\noverwrite()",
    ],
)
def test_selected_notebook_scopes_reject_dynamic_namespace_crossings(
    overwrite: str,
) -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    auxiliary = scope(
        f"{overwrite}\ncorrection = offsets * 2", "offsets", "correction", {}
    )
    with pytest.raises(ReplayGraphError, match="opaque namespace"):
        selected_provenance_code(
            (
                scope("derived = data + 1", "data", "derived", {}),
                auxiliary,
                scope(
                    "derived = derived + correction",
                    "derived",
                    "derived",
                    {"derived": 0, "correction": 1},
                ),
            )
        )

    source = xr.DataArray([3.0])
    code = selected_provenance_code((auxiliary,))
    namespace = _exec_generated_code(code, {"offsets": source})
    xr.testing.assert_identical(namespace["correction"], source * 2)
    independent = scope("result = fresh * 3", "fresh", "result", {})
    code = selected_provenance_code((auxiliary, independent))
    namespace = _exec_generated_code(code, {"offsets": source, "fresh": source + 1})
    xr.testing.assert_identical(namespace["result"], (source + 1) * 3)


@pytest.mark.parametrize(
    "write", ["saved = data + 1", "del saved", "import math as saved"]
)
def test_selected_notebook_scopes_reject_deferred_global_writes(write: str) -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    definition = f"def prepare():\n    global saved\n    {write}"
    producer = scope(f"{definition}\nderived = data + 1", "data", "derived", {})
    auxiliary = scope(
        "saved = offsets * 10\n"
        "def correction_fn():\n    return saved * 0\n"
        "correction = correction_fn()",
        "offsets",
        "correction",
        {},
    )
    consumer = scope(
        "prepare()\nresult = saved + correction",
        "derived",
        "result",
        {"derived": 0, "correction": 1},
    )
    with pytest.raises(ReplayGraphError, match="opaque namespace"):
        selected_provenance_code((producer, auxiliary, consumer))

    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    code = selected_provenance_code((producer,))
    namespace = _exec_generated_code(code, {"data": source})
    xr.testing.assert_identical(namespace["derived"], source + 1)
    assert "saved" not in namespace
    standalone = scope(
        f"saved = data\n{definition}\nprepare()\nresult = data + 1",
        "data",
        "result",
        {},
    )
    code = selected_provenance_code((standalone,))
    namespace = _exec_generated_code(code, {"data": source})
    xr.testing.assert_identical(namespace["result"], source + 1)


def test_selected_notebook_scopes_allow_read_only_global_declarations() -> None:
    producer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Define",
                    code="def normalize():\n    global data\n    return data + 1\n"
                    "derived = data + 1",
                )
            ),
        ),
        input_name="data",
        output_name="derived",
    )
    consumer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(label="Use", code="result = normalize()")
            ),
        ),
        input_name="derived",
        output_name="result",
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(producer, {}),
            SelectedProvenanceScope(consumer, {"derived": 0}),
        )
    )
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    namespace = _exec_generated_code(code, {"data": source})
    xr.testing.assert_identical(namespace["result"], source + 1)


@pytest.mark.parametrize(
    "initialization", ["direct", "conditional", "augmented", "assertion"]
)
def test_selected_notebook_scopes_restore_only_live_incoming_names(
    initialization: str,
) -> None:
    conditional = initialization == "conditional"

    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    producer = scope(
        "saved = data + 1\n"
        + ("del saved\n" if initialization == "direct" else "")
        + "derived = data + 1",
        "data",
        "derived",
        {},
    )
    auxiliary = scope(
        "saved: xr.DataArray = offsets * 10\ncorrection = saved * 0",
        "offsets",
        "correction",
        {},
    )
    consumer = scope(
        (
            "if flag:\n    saved = derived + 1\n"
            if conditional
            else (
                "saved += 1\n"
                if initialization == "augmented"
                else (
                    "assert True, (saved := derived + 100)\n"
                    if initialization == "assertion"
                    else "saved = derived + 1\n"
                )
            )
        )
        + "result = saved + correction",
        "derived",
        "result",
        {"derived": 0, "correction": 1},
    )
    code = selected_provenance_code((producer, auxiliary, consumer))
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    offsets = xr.DataArray([3.0, 4.0, 5.0], dims="x")
    namespace = _exec_generated_code(
        code, {"data": source, "offsets": offsets, "flag": False, "xr": xr}
    )
    xr.testing.assert_identical(
        namespace["result"],
        source + (1 if conditional or initialization == "assertion" else 2),
    )
    if initialization == "direct":
        assert "saved_input" not in code
        live_consumer = scope(
            "result = saved + correction",
            "derived",
            "result",
            {"derived": 0, "correction": 1},
        )
        with pytest.raises(ReplayGraphError, match="binding 'saved' was deleted"):
            selected_provenance_code((producer, auxiliary, live_consumer))
        rebound = scope(
            "saved = data + 1\ndel saved\nsaved = data + 2\nderived = data + 1",
            "data",
            "derived",
            {},
        )
        code = selected_provenance_code((rebound, auxiliary, live_consumer))
        namespace = _exec_generated_code(
            code, {"data": source, "offsets": offsets, "xr": xr}
        )
        xr.testing.assert_identical(namespace["result"], source + 2)


def test_selected_notebook_scopes_preserve_bindings_across_bare_annotations() -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    code = selected_provenance_code(
        (
            scope("saved = data + 1\nderived = data + 1", "data", "derived", {}),
            scope(
                "saved: xr.DataArray\ncorrection = offsets * 0",
                "offsets",
                "correction",
                {},
            ),
            scope(
                "result = saved + correction",
                "derived",
                "result",
                {"derived": 0, "correction": 1},
            ),
        )
    )
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    offsets = xr.DataArray([3.0, 4.0, 5.0], dims="x")
    namespace = _exec_generated_code(
        code, {"data": source, "offsets": offsets, "xr": xr}
    )
    xr.testing.assert_identical(namespace["result"], source + 1)
    assert "saved_input" not in code


@pytest.mark.parametrize("flag", [None, False, True])
@pytest.mark.parametrize("binding", ["logical", "primary", "auxiliary", "caller"])
def test_selected_notebook_scopes_restore_bindings_needed_by_deletion(
    binding: str, flag: bool | None
) -> None:
    def scope(code, input_name, output_name, links, script_inputs=()):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
                script_inputs=script_inputs,
            ),
            links,
        )

    target = {"primary": "sample", "auxiliary": "reference"}.get(binding, "saved")
    deletion = f"del {target}"
    if flag is not None:
        deletion = f"if flag:\n    {deletion}"
    primary_input = "sample" if binding == "primary" else "prepared"
    links = {primary_input: 0, "correction": 1}
    if binding == "auxiliary":
        links["reference"] = 1
    code = selected_provenance_code(
        (
            scope("saved = data + 1\nprepared = data + 1", "data", "prepared", {}),
            scope("del saved\ncorrection = offsets * 0", "offsets", "correction", {}),
            scope(
                f"{deletion}\nresult = correction + 1",
                primary_input,
                "result",
                links,
                (ScriptInput(name="saved", revision_id="caller-saved"),)
                if binding == "caller"
                else (),
            ),
        )
    )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    inputs = {"data": source, "offsets": source + 100, "flag": flag}
    if binding == "caller":
        inputs["saved"] = source + 1000
    namespace = _exec_generated_code(code, inputs)

    xr.testing.assert_identical(namespace["result"], source * 0 + 1)
    if flag is False:
        expected = (
            source * 0
            if binding == "auxiliary"
            else source + (1000 if binding == "caller" else 1)
        )
        xr.testing.assert_identical(namespace[target], expected)
    else:
        assert target not in namespace
    captures = [
        statement
        for statement in ast.parse(code).body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id.endswith("_input")
    ]
    assert len(captures) == (1 if binding in {"logical", "caller"} else 0)
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


@pytest.mark.parametrize("deletion", ["del saved", "if flag:\n    del saved"])
@pytest.mark.parametrize("primary_scope", [False, True])
def test_selected_notebook_scopes_reject_deletion_of_known_missing_bindings(
    deletion: str, primary_scope: bool
) -> None:
    producer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare",
                    code="saved = data + 1\ndel saved\nprepared = data + 1",
                )
            ),
        ),
        input_name="data",
        output_name="prepared",
    )
    consumer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Delete", code=f"{deletion}\nresult = prepared + 1"
                )
            ),
        ),
        input_name="prepared",
        output_name="result",
    )
    with pytest.raises(ReplayGraphError, match="binding 'saved' was deleted"):
        selected_provenance_code(
            (
                SelectedProvenanceScope(producer, {}),
                SelectedProvenanceScope(
                    consumer, {"prepared": 0} if primary_scope else {}
                ),
            )
        )


@pytest.mark.parametrize("seed", [False, True])
def test_selected_code_keeps_explicit_alias_needed_by_deletion(seed: bool) -> None:
    steps = (
        ReplayStep(
            operation=ScriptCodeOperation(
                label="Delete alias", code="del saved\nresult = data + 1"
            )
        ),
    )
    spec = (
        script(
            start_label="Use alias",
            active_name="result",
            seed_code="saved = data",
            steps=steps,
        )
        if seed
        else selected_steps_spec(
            steps, input_name="data", output_name="result", input_alias="saved"
        )
    )
    code = selected_provenance_code((SelectedProvenanceScope(spec, {}),))
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(code, {"data": source})

    xr.testing.assert_identical(namespace["result"], source + 1)
    assert "saved" not in namespace
    assert ".copy(" not in code
    assert "_input" not in code


@pytest.mark.parametrize("read_alias", [False, True])
@pytest.mark.parametrize("cleanup", ["adjacent", "single", "simple", "copy", "display"])
def test_generated_code_cleanup_keeps_bindings_needed_by_deletion(
    cleanup: str, read_alias: bool
) -> None:
    alias = "_itool_replay_0"
    expression = alias if read_alias else "data"
    code = f"{alias} = data\nresult = {expression} + 1\ndel {alias}"
    if cleanup == "copy":
        cleaned = _remove_unused_generated_copies(code, {alias})
    elif cleanup == "display":
        cleaned = _simplify_display_code(code, inline_targets={alias})
    else:
        cleaned = {
            "adjacent": _inline_adjacent_replay_assignments,
            "single": _inline_single_use_replay_names,
            "simple": _inline_simple_name_aliases,
        }[cleanup](code)
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(cleaned, {"data": source})

    xr.testing.assert_identical(namespace["result"], source + 1)
    assert alias not in namespace


def test_script_name_validation_deletes_function_without_reading_dependencies() -> None:
    code = "def unused():\n    return missing\ndel unused\nresult = data + 1"
    available = {"data"}
    dependencies: dict[str, set[str]] = {}
    callers = _validate_script_code_names(code, available, dependencies)
    source = xr.DataArray([1.0, 2.0], dims="x")
    namespace = _exec_generated_code(code, {"data": source})

    xr.testing.assert_identical(namespace["result"], source + 1)
    assert not callers
    assert "unused" not in available
    assert not dependencies


def test_code_analysis_separates_deletion_presence_from_value_reads() -> None:
    code = "del saved\nsaved = data + 1\ndel saved\nresult = data + 1"
    analysis = _CodeAnalyzer()(code)
    source = xr.DataArray([1.0, 2.0], dims="x")
    namespace = _exec_generated_code(code, {"data": source, "saved": source + 10})

    xr.testing.assert_identical(namespace["result"], source + 1)
    assert analysis.incoming_bindings == {"saved"}
    assert analysis.incoming_reads == {"data"}
    assert not _code_uses_name(code, "saved")
    assert _code_uses_name_any_scope(code, "saved")
    assert not _code_uses_name_any_scope("def cleanup():\n    del era", "era")


@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        "values = [(saved := prepared + item) + saved for item in [1]]\n"
        "result = values[0]",
        "values = {(saved := item) + saved for item in [1]}\n"
        "result = prepared * 2 + values.pop()",
        "values = {(saved := item): prepared + saved for item in [1]}\n"
        "result = values[1] * 2",
        "values = [saved + saved for item in [1] "
        "if (saved := prepared + item) is not None]\nresult = values[0]",
        "values = [value + saved for item in [1] "
        "if (saved := prepared + item) is not None for value in [saved]]\n"
        "result = values[0]",
        "values = [[(saved := prepared + inner) + saved for inner in [item]][0] "
        "for item in [1]]\nresult = values[0]",
    ],
    ids=["list", "set", "dict", "filter", "later-iterable", "nested"],
)
def test_selected_code_keeps_eager_comprehension_binding_order(
    body: str, composed: bool
) -> None:
    consumer = _selected_script_scope(body, "prepared", "result", {"prepared": 0})
    assert {item.name for item in consumer.spec.script_inputs} == {"prepared"}
    analysis = _CodeAnalyzer()(body)
    assert analysis.incoming_reads == {"prepared"}
    assert not analysis.deferred_reads
    available = {"prepared"}
    assert not _validate_script_code_names(body, available, {})
    assert not {"item", "inner", "value"} & available
    code = (
        selected_provenance_code(
            (
                _selected_script_scope(
                    "saved = data\ndel saved\nprepared = data + 1",
                    "data",
                    "prepared",
                    {},
                ),
                consumer,
            )
        )
        if composed
        else selected_steps_code(
            consumer.spec.steps, input_name="prepared", output_name="result"
        )
    )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(code, {"data": source, "prepared": source + 1})

    xr.testing.assert_identical(namespace["result"], (source + 2) * 2)


@pytest.mark.parametrize(
    "body",
    [
        "values = [(saved := prepared + item) + saved for item in saved]",
        "values = [saved + saved for item in [1] "
        "if flag or (saved := prepared + item) is not None]",
    ],
    ids=["first-iterable", "short-circuit-filter"],
)
def test_selected_scopes_keep_incoming_comprehension_reads(body: str) -> None:
    assert "saved" in _CodeAnalyzer()(body).incoming_reads
    assert not _validate_script_code_names(body, {"prepared", "flag", "saved"}, {})
    with pytest.raises(ReplayGraphError, match="binding 'saved' was deleted"):
        selected_provenance_code(
            (
                _selected_script_scope(
                    "saved = data\ndel saved\nprepared = data + 1",
                    "data",
                    "prepared",
                    {},
                ),
                _selected_script_scope(
                    body + "\nresult = values[0]",
                    "prepared",
                    "result",
                    {"prepared": 0},
                ),
            )
        )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(
        body, {"prepared": source + 1, "saved": [1], "flag": False}
    )

    xr.testing.assert_identical(namespace["values"][0], (source + 2) * 2)


@pytest.mark.parametrize(
    "function_name", ["normalize", "listcomp", "setcomp", "dictcomp"]
)
def test_code_analysis_keeps_deferred_function_reads(function_name: str) -> None:
    code = f"def {function_name}():\n    return data + 1\nresult = {function_name}()"
    analysis = _CodeAnalyzer()(code)
    source = xr.DataArray([1.0, 2.0], dims="x")
    namespace = _exec_generated_code(code, {"data": source})

    xr.testing.assert_identical(namespace["result"], source + 1)
    assert analysis.deferred_reads == {"data"}


def test_trusted_script_replay_initializes_implicit_alias_needed_by_deletion() -> None:
    spec = script(
        start_label="Delete alias",
        active_name="result",
        seed_code="del era\nresult = xr.DataArray([1.0, 2.0], dims='x')",
    )
    graph = compile_replay_graph(spec, trusted_user_code=True)
    result = execute_replay_graph(graph, authorize=_authorize_execution)

    xr.testing.assert_identical(result, xr.DataArray([1.0, 2.0], dims="x"))


@pytest.mark.parametrize(
    ("body", "flag", "increment"),
    [
        ("if flag:\n    del saved\n    saved = data + 2", True, 2),
        ("if flag:\n    del saved\n    saved = data + 2", False, 1),
        ("if flag:\n    del saved", False, 1),
        ("for item in range(1):\n    del saved\n    saved = data + 2", True, 2),
        (
            "for item in range(1):\n    del saved\n    saved = data + 2\n"
            "    break\n    del saved",
            True,
            2,
        ),
        (
            "for item in range(1):\n    del saved\n    saved = data + 2\n"
            "    continue\n    del saved",
            True,
            2,
        ),
        ("while flag:\n    del saved\n    saved = data + 2\n    break", True, 2),
        ("while flag:\n    del saved", False, 1),
        ("try:\n    pass\nexcept ValueError as saved:\n    pass", True, 1),
        (
            "try:\n    raise ValueError()\nexcept ValueError:\n"
            "    del saved\n    saved = data + 2",
            True,
            2,
        ),
        ("try:\n    del saved\nfinally:\n    saved = data + 2", True, 2),
        (
            "try:\n    raise ValueError()\nexcept ValueError as saved:\n"
            "    pass\nfinally:\n    saved = data + 2",
            True,
            2,
        ),
        (
            "try:\n    del saved\nexcept ValueError:\n    pass\n"
            "else:\n    saved = data + 2",
            True,
            2,
        ),
        (
            "match flag:\n    case True:\n        del saved\n"
            "        saved = data + 2\n    case False:\n        pass",
            True,
            2,
        ),
        (
            "match flag:\n    case _:\n        del saved\n        saved = data + 2",
            True,
            2,
        ),
    ],
)
def test_selected_notebook_scopes_keep_ordered_compound_bindings(
    body: str, flag: bool, increment: int
) -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    code = selected_provenance_code(
        (
            scope(
                f"saved = data + 1\n{body}\nprepared = data + 1",
                "data",
                "prepared",
                {},
            ),
            scope(
                "saved = offsets * 10\ncorrection = saved * 0",
                "offsets",
                "correction",
                {},
            ),
            scope(
                "result = saved + correction",
                "prepared",
                "result",
                {"prepared": 0, "correction": 1},
            ),
        )
    )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(
        code, {"data": source, "offsets": source + 10, "flag": flag}
    )
    xr.testing.assert_identical(namespace["result"], source + increment)
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


@pytest.mark.parametrize(
    "initialization",
    [
        "if flag:\n    saved = derived + 2\nelse:\n    saved = derived + 2",
        "try:\n    saved = derived + 2\nfinally:\n    pass",
        "try:\n    pass\nfinally:\n    saved = derived + 2",
        "match flag:\n    case _:\n        saved = derived + 2",
        "for item in ():\n    pass\nelse:\n    saved = derived + 2",
    ],
)
def test_selected_notebook_scopes_do_not_restore_replaced_compound_bindings(
    initialization: str,
) -> None:
    producer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare",
                    code="saved = data + 1\ndel saved\nderived = data + 1",
                )
            ),
        ),
        input_name="data",
        output_name="derived",
    )
    consumer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Replace", code=f"{initialization}\nresult = saved + 0"
                )
            ),
        ),
        input_name="derived",
        output_name="result",
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(producer, {}),
            SelectedProvenanceScope(consumer, {"derived": 0}),
        )
    )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(code, {"data": source, "flag": False})
    xr.testing.assert_identical(namespace["result"], source + 3)
    assert "saved_input" not in code


@pytest.mark.parametrize(
    ("body", "flag", "increment"),
    [
        ("if flag:\n    saved = prepared + 2", False, 1),
        ("if flag:\n    saved = prepared + 2", True, 3),
        ("if flag:\n    del saved", False, 1),
        ("for item in ():\n    saved = prepared + 2", False, 1),
        ("values = [(saved := item) for item in ()]", False, 1),
        ("try:\n    pass\nexcept ValueError as saved:\n    pass", False, 1),
        ("assert True, (saved := prepared + 2)", False, 1),
    ],
)
def test_selected_notebook_scopes_publish_retained_conditional_bindings(
    body: str, flag: bool, increment: int
) -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    code = selected_provenance_code(
        (
            scope("saved = data + 1\nprepared = data + 1", "data", "prepared", {}),
            scope(
                "saved = offsets * 5\ncorrection = offsets * 0",
                "offsets",
                "correction",
                {},
            ),
            scope(
                f"{body}\nchild = prepared + 0", "prepared", "child", {"prepared": 0}
            ),
            scope(
                "result = saved + correction",
                "child",
                "result",
                {"child": 2, "correction": 1},
            ),
        )
    )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(
        code, {"data": source, "offsets": source + 100, "flag": flag}
    )
    xr.testing.assert_identical(namespace["result"], source + increment)
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize(
    "recovery",
    [
        "conditional",
        "delete_conditional",
        "branch_delete",
        "definite",
        "branch_definite",
    ],
)
def test_selected_notebook_scopes_do_not_retain_deleted_conditional_bindings(
    flag: bool, recovery: str
) -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    initialization = {
        "conditional": "if flag:\n    saved = prepared + 2\n",
        "delete_conditional": "del saved\nif flag:\n    saved = prepared + 2\n",
        "branch_delete": "if flag:\n    saved = prepared + 2\nelse:\n    del saved\n",
        "definite": "saved = prepared + 2\n",
        "branch_definite": (
            "if flag:\n    saved = prepared + 2\nelse:\n    saved = prepared + 2\n"
        ),
    }[recovery]
    scopes = [
        scope(
            "saved = data + 1\ndel saved\nprepared = data + 1", "data", "prepared", {}
        ),
        scope(
            "saved = offsets * 5\ncorrection = offsets * 0",
            "offsets",
            "correction",
            {},
        ),
        scope(
            initialization + "child = prepared + 0",
            "prepared",
            "child",
            {"prepared": 0},
        ),
    ]
    if recovery in {"delete_conditional", "branch_delete"}:
        with pytest.raises(ReplayGraphError, match="binding 'saved' was deleted"):
            selected_provenance_code(scopes)
        return
    code = selected_provenance_code(scopes)
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    inputs = {"data": source, "offsets": source + 100, "flag": flag}
    namespace = _exec_generated_code(code, inputs)
    xr.testing.assert_identical(namespace["child"], source + 1)
    assert "saved_input" not in code
    scopes.append(
        scope(
            "result = saved + correction",
            "child",
            "result",
            {"child": 2, "correction": 1},
        )
    )
    if recovery in {"conditional", "delete_conditional", "branch_delete"}:
        with pytest.raises(ReplayGraphError, match="binding 'saved' was deleted"):
            selected_provenance_code(scopes)
    else:
        code = selected_provenance_code(scopes)
        namespace = _exec_generated_code(code, inputs)
        xr.testing.assert_identical(namespace["result"], source + 3)


@pytest.mark.parametrize(
    "deletion",
    [
        "del saved",
        "if flag:\n    del saved\nelse:\n    del saved",
        "try:\n    pass\nfinally:\n    del saved",
        "try:\n    raise ValueError()\nexcept ValueError as saved:\n    pass",
        "match flag:\n    case _:\n        del saved",
    ],
)
def test_selected_notebook_scopes_reject_definite_compound_deletions(
    deletion: str,
) -> None:
    producer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Delete",
                    code=f"saved = data + 1\n{deletion}\nderived = data + 1",
                )
            ),
        ),
        input_name="data",
        output_name="derived",
    )
    consumer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(label="Use", code="result = saved + 0")
            ),
        ),
        input_name="derived",
        output_name="result",
    )
    with pytest.raises(ReplayGraphError, match="binding 'saved' was deleted"):
        selected_provenance_code(
            (
                SelectedProvenanceScope(producer, {}),
                SelectedProvenanceScope(consumer, {"derived": 0}),
            )
        )


@pytest.mark.parametrize(
    "code", ["saved: int", "del saved", "if flag:\n    del saved\nelse:\n    del saved"]
)
def test_script_name_validation_does_not_create_bindings(code: str) -> None:
    available = {"int", "flag"} | ({"saved"} if "del saved" in code else set())
    _validate_script_code_names(code, available, {})
    assert "saved" not in available
    with pytest.raises(ReplayGraphError, match="unresolved name 'saved'"):
        _validate_script_code_names("result = saved", available, {})


def test_selected_steps_code_keeps_definition_time_bindings() -> None:
    code = selected_steps_code(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Define",
                    code="def helper(value=(saved := data + 2)):\n    return value\n"
                    "result = saved + helper()",
                )
            ),
        ),
        input_name="data",
        output_name="result",
    )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "eV"))
    namespace = _exec_generated_code(code, {"data": source})
    xr.testing.assert_identical(namespace["result"], (source + 2) * 2)


@pytest.mark.parametrize(
    ("annotation", "missing"),
    [("container.value: float", "container"), ("container[index]: float", "index")],
)
def test_script_name_validation_reads_annotation_targets(
    annotation: str, missing: str
) -> None:
    available = {"container", "index", "float"}
    _validate_script_code_names(annotation, available, {})
    available.remove(missing)
    with pytest.raises(ReplayGraphError, match=f"unresolved name {missing!r}"):
        _validate_script_code_names(annotation, available, {})


@pytest.mark.parametrize(
    "statement",
    ["values = ((saved := item) for item in ())", "((saved := item) for item in ())"],
)
def test_script_name_validation_keeps_generator_bindings_deferred(
    statement: str,
) -> None:
    available: set[str] = set()
    code = f"saved = 1\ndel saved\n{statement}"
    _validate_script_code_names(code, available, {})
    namespace = _exec_generated_code(code)
    assert "saved" not in namespace
    assert "saved" not in available
    with pytest.raises(ReplayGraphError, match="unresolved name 'saved'"):
        _validate_script_code_names(
            f"{code}\nresult = saved", set(), {}, external_name_candidates=None
        )


def test_script_name_validation_does_not_readmit_deleted_callers() -> None:
    with pytest.raises(ReplayGraphError, match="unresolved name 'saved'"):
        _validate_script_code_names(
            "saved = data\ndel saved\nresult = saved",
            {"data"},
            {},
            external_name_candidates=None,
        )


def test_selected_notebook_scope_publishes_input_used_by_dynamic_access() -> None:
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    producer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare", code="prepared = data + 1"
                )
            ),
        ),
        input_name="data",
        output_name="prepared",
    )
    consumer = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Scale", code="result = globals()['source'] * 2"
                )
            ),
        ),
        input_name="source",
        output_name="result",
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(producer, {}),
            SelectedProvenanceScope(consumer, {"source": 0}),
        )
    )
    namespace = _exec_generated_code(code, {"data": source})
    xr.testing.assert_identical(namespace["result"], (source + 1) * 2)
    assert ".copy(" not in code


@pytest.mark.parametrize(
    ("setup", "overwrite"),
    [
        ("import builtins as runtime", "runtime.exec('derived = offsets')"),
        ("namespace = globals", "namespace()['derived'] = offsets"),
        ("values = ((saved := item) for item in ())", "next(values)"),
        ("namespace = globals()", "namespace['derived'] = offsets"),
        (
            "namespace = globals()\nnamespace['_scratch'] = None\n"
            "del namespace['_scratch']",
            "namespace['derived'] = offsets",
        ),
        (
            "import builtins as runtime\nexecute = runtime.exec",
            "execute('derived = offsets')",
        ),
        (
            "import builtins as runtime\nexecute = runtime.__dict__['exec']",
            "execute('derived = offsets')",
        ),
        (
            "import builtins as runtime\nexecute = getattr(runtime, 'exec')",
            "execute('derived = offsets')",
        ),
        (
            "import builtins as runtime\nnamespace = getattr(runtime, '__dict__')",
            "namespace['exec']('derived = offsets')",
        ),
        (
            "import builtins as runtime\nnamespace = globals\n"
            "runtime, namespace = namespace, runtime",
            "namespace.exec('derived = offsets')",
        ),
        (
            "namespace, shadow = globals, (globals := lambda: None)",
            "namespace()['derived'] = offsets",
        ),
        ("def namespace():\n    return globals()", "namespace()['derived'] = offsets"),
        ("namespace = lambda: globals()", "namespace()['derived'] = offsets"),
        (
            "def namespace(getter=globals):\n    return getter()",
            "namespace()['derived'] = offsets",
        ),
        (
            "namespace = lambda getter=globals: getter()",
            "namespace()['derived'] = offsets",
        ),
        (
            "def setup():\n    global namespace\n    namespace = globals()\nsetup()",
            "namespace['derived'] = offsets",
        ),
        (
            "def setup():\n    global runtime\n    import builtins as runtime\nsetup()",
            "runtime.exec('derived = offsets')",
        ),
    ],
)
def test_selected_notebook_scopes_reject_opaque_capability_producers(
    setup: str, overwrite: str
) -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    producer = scope(f"{setup}\nderived = data + 1", "data", "derived", {})
    consumer = scope(
        "result = derived + correction",
        "derived",
        "result",
        {"derived": 0, "correction": 1},
    )
    with pytest.raises(ReplayGraphError, match="opaque namespace"):
        selected_provenance_code(
            (
                producer,
                scope(
                    f"{overwrite}\ncorrection = offsets * 2",
                    "offsets",
                    "correction",
                    {},
                ),
                consumer,
            )
        )

    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    code = selected_provenance_code((producer,))
    namespace = _exec_generated_code(code, {"data": source})
    xr.testing.assert_identical(namespace["derived"], source + 1)
    with pytest.raises(ReplayGraphError, match="opaque namespace"):
        selected_provenance_code(
            (
                producer,
                scope("result = derived * 2", "derived", "result", {"derived": 0}),
            )
        )


def test_selected_notebook_scopes_allow_ordinary_dictionary_methods() -> None:
    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "eV"))
    consumer = scope(
        "result = derived + correction",
        "derived",
        "result",
        {"derived": 0, "correction": 1},
    )
    code = selected_provenance_code(
        (
            scope("derived = data + 1", "data", "derived", {}),
            scope(
                "runtime = {'offset': offsets}\n"
                "namespace = runtime\n"
                "execute = namespace.get\n"
                "correction = execute('offset') * 2",
                "offsets",
                "correction",
                {},
            ),
            consumer,
        )
    )
    namespace = _exec_generated_code(code, {"data": source, "offsets": source + 3})
    xr.testing.assert_identical(namespace["result"], source + 1 + (source + 3) * 2)


@pytest.mark.parametrize("output_name", ["derived", "np"])
def test_selected_notebook_scopes_keep_outputs_across_public_name_reuse(
    output_name: str,
) -> None:
    source = xr.DataArray([10.0])
    offsets = xr.DataArray([3.0])
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare sample", code=f"{output_name} = data + 1"
                )
            ),
        ),
        input_name="data",
        output_name=output_name,
    )
    auxiliary = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare correction", code=f"{output_name} = offsets * 10"
                )
            ),
        ),
        input_name="offsets",
        output_name=output_name,
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Apply correction", code="shifted = data + correction"
                )
            ),
        ),
        input_name="data",
        output_name="shifted",
        script_inputs=(ScriptInput(name="correction", revision_id="correction"),),
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(first, {}),
            SelectedProvenanceScope(auxiliary, {}),
            SelectedProvenanceScope(last, {"data": 0, "correction": 1}),
        )
    )
    namespace = _exec_generated_code(code, {"data": source, "offsets": offsets})

    xr.testing.assert_identical(namespace["shifted"], source + 1 + offsets * 10)
    xr.testing.assert_identical(namespace[output_name], offsets * 10)
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


def test_selected_notebook_scopes_publish_consumed_metadata_updates() -> None:
    source = xr.DataArray([10.0])
    first = selected_steps_spec(
        (ReplayStep(operation=KspaceWorkFunctionOperation(work_function=4.2)),),
        input_name="data",
        output_name="derived",
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Scale sample", code="result = data * 2"
                )
            ),
        ),
        input_name="data",
        output_name="result",
    )
    code = selected_provenance_code(
        (SelectedProvenanceScope(first, {}), SelectedProvenanceScope(last, {"data": 0}))
    )
    namespace = _exec_generated_code(code, {"data": source})

    xr.testing.assert_identical(namespace["result"], source * 2)
    assert namespace["derived"].kspace.analyzer_work_function == 4.2
    assert ".copy(" not in code


def test_selected_notebook_scopes_preserve_outputs_across_emitted_input_aliases() -> (
    None
):
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "y"))
    offsets = xr.DataArray([3.0, 4.0, 5.0], dims="x")

    def scope(code, input_name, output_name, links):
        return SelectedProvenanceScope(
            selected_steps_spec(
                (ReplayStep(operation=ScriptCodeOperation(label="Step", code=code)),),
                input_name=input_name,
                output_name=output_name,
            ),
            links,
        )

    code = selected_provenance_code(
        (
            scope("data = source + 1", "source", "data", {}),
            scope("original = offsets * 2", "offsets", "original", {}),
            scope(
                "def f():\n    return data * 10\ncorrection = f()",
                "data",
                "correction",
                {"data": 1},
            ),
            scope(
                "def g():\n    return original + correction\nresult = g()",
                "original",
                "result",
                {"original": 0, "correction": 2},
            ),
            scope(
                "final = correction + result",
                "correction",
                "final",
                {"correction": 1, "result": 3},
            ),
        )
    )
    namespace = _exec_generated_code(code, {"source": source, "offsets": offsets})

    xr.testing.assert_identical(namespace["result"], source + 1 + offsets * 20)
    xr.testing.assert_identical(
        namespace["final"], offsets * 2 + source + 1 + offsets * 20
    )
    assert ".copy(" not in code
    assert "_itool_replay_" not in code


def test_selected_notebook_scopes_capture_recorded_inputs_before_emitted_aliases() -> (
    None
):
    source = xr.DataArray(np.full((3, 2), 10.0), dims=("x", "y"))
    recorded = source + 100
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare sample", code="derived = source + 1"
                )
            ),
        ),
        input_name="source",
        output_name="derived",
    )
    auxiliary = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare correction",
                    code="def f():\n    return data * 2\ncorrection = f()",
                )
            ),
        ),
        input_name="data",
        output_name="correction",
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Apply correction", code="result = data + correction"
                )
            ),
        ),
        input_name="correction",
        output_name="result",
        script_inputs=(ScriptInput(name="data", revision_id="recorded"),),
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(first, {}),
            SelectedProvenanceScope(auxiliary, {"data": 0}),
            SelectedProvenanceScope(last, {"correction": 1}),
        )
    )
    namespace = _exec_generated_code(code, {"source": source, "data": recorded})

    xr.testing.assert_identical(namespace["result"], recorded + (source + 1) * 2)
    assert ".copy(" not in code


def test_selected_notebook_scopes_reserve_emitted_extension_bindings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = tmp_path / "original_input.py"
    script_path.write_text(
        "import xarray as xr\n"
        "from erlab.extensions import routine\n"
        "@routine(id='scale')\n"
        "def scale(data: xr.DataArray) -> xr.DataArray:\n    return data * 2\n"
    )
    monkeypatch.setattr(
        extension_api,
        "_resolve_registered_script_capability",
        lambda *_args: _registered_routine(script_path),
    )
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "y"))
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare sample", code="data = source + 1"
                )
            ),
        ),
        input_name="source",
        output_name="data",
    )
    extension = selected_steps_spec(
        (ReplayStep(operation=_extension_routine_operation()),),
        input_name="other",
        output_name="scaled",
    )
    alias = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Use scaled data",
                    code="def f():\n    return data + 1\ncorrection = f()",
                )
            ),
        ),
        input_name="data",
        output_name="correction",
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Combine inputs", code="result = original + correction"
                )
            ),
        ),
        input_name="original",
        output_name="result",
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(first, {}),
            SelectedProvenanceScope(extension, {}),
            SelectedProvenanceScope(alias, {"data": 1}),
            SelectedProvenanceScope(last, {"original": 0, "correction": 2}),
            SelectedProvenanceScope(extension, {}),
        )
    )
    namespace = _exec_generated_code(code, {"source": source, "other": source + 100})

    xr.testing.assert_identical(
        namespace["result"], source + 1 + (source + 100) * 2 + 1
    )
    assert isinstance(namespace["original_input"], erlab.extensions.LoadedScript)
    xr.testing.assert_identical(namespace["original_input_2"], source + 1)
    assert ".copy(" not in code


def test_selected_notebook_scopes_forward_captured_source_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def expression_code(self, input_name, *, source_name=None):
        return f"({input_name} + {source_name}).qsel.mean('x')"

    monkeypatch.setattr(AverageOperation, "expression_code", expression_code)
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "y"))
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare sample", code="derived = source + 1"
                )
            ),
        ),
        input_name="source",
        output_name="derived",
    )
    auxiliary = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare correction", code="derived = offsets * 2"
                )
            ),
        ),
        input_name="offsets",
        output_name="derived",
    )
    last = selected_steps_spec(
        (ReplayStep(operation=AverageOperation(dims=("x",))),),
        input_name="data",
        output_name="result",
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(first, {}),
            SelectedProvenanceScope(auxiliary, {}),
            SelectedProvenanceScope(last, {"data": 0}),
        )
    )
    namespace = _exec_generated_code(code, {"source": source, "offsets": source + 100})

    xr.testing.assert_identical(namespace["result"], ((source + 1) * 2).qsel.mean("x"))
    assert ".copy(" not in code


@pytest.mark.parametrize(
    "overwrite_code",
    [
        "import math as data",
        "from math import pi as data",
        "try:\n    raise ValueError()\nexcept ValueError as data:\n    pass",
        "match offsets.size:\n    case data:\n        pass",
    ],
)
def test_selected_notebook_scopes_preserve_outputs_across_namespace_writes(
    overwrite_code: str,
) -> None:
    source = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("x", "y"))
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare sample", code="data = source + 1"
                )
            ),
        ),
        input_name="source",
        output_name="data",
    )
    auxiliary = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare correction",
                    code=f"{overwrite_code}\ncorrection = offsets * 2",
                )
            ),
        ),
        input_name="offsets",
        output_name="correction",
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Combine inputs", code="result = data + correction"
                )
            ),
        ),
        input_name="data",
        output_name="result",
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(first, {}),
            SelectedProvenanceScope(auxiliary, {}),
            SelectedProvenanceScope(last, {"data": 0, "correction": 1}),
        )
    )
    namespace = _exec_generated_code(code, {"source": source, "offsets": source + 3})

    xr.testing.assert_identical(namespace["result"], source + 1 + (source + 3) * 2)
    assert ".copy(" not in code


@pytest.mark.parametrize("input_name", ["np", "xr", "len", "load_script"])
def test_selected_notebook_scopes_reject_protected_external_input_names(
    input_name: str,
) -> None:
    steps = (
        ReplayStep(
            operation=ScriptCodeOperation(
                label="Use input", code=f"result = {input_name}"
            )
        ),
    )
    with pytest.raises(ReplayGraphError, match="conflicts with a replay global"):
        selected_steps_code(steps, input_name=input_name, output_name="result")


def test_selected_notebook_scopes_ignore_local_bindings_when_capturing_outputs() -> (
    None
):
    source = xr.DataArray([10.0])
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(label="Prepare", code="data = source + 1")
            ),
        ),
        input_name="source",
        output_name="data",
    )
    auxiliary = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare correction",
                    code="def f(data):\n    return data * 2\ncorrection = f(offsets)",
                )
            ),
        ),
        input_name="offsets",
        output_name="correction",
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Combine", code="result = data + correction"
                )
            ),
        ),
        input_name="data",
        output_name="result",
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(first, {}),
            SelectedProvenanceScope(auxiliary, {}),
            SelectedProvenanceScope(last, {"data": 0, "correction": 1}),
        )
    )
    namespace = _exec_generated_code(code, {"source": source, "offsets": source + 3})

    xr.testing.assert_identical(namespace["result"], source + 1 + (source + 3) * 2)
    assert "data_input" not in code


def test_selected_notebook_scopes_reject_opaque_import_across_bound_values() -> None:
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare sample", code="array = source + 1"
                )
            ),
        ),
        input_name="source",
        output_name="array",
    )
    auxiliary = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare correction",
                    code="from numpy import *\ncorrection = offsets * 2",
                )
            ),
        ),
        input_name="offsets",
        output_name="correction",
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Combine inputs", code="result = data + correction"
                )
            ),
        ),
        input_name="data",
        output_name="result",
    )

    with pytest.raises(ReplayGraphError, match="opaque namespace import"):
        selected_provenance_code(
            (
                SelectedProvenanceScope(first, {}),
                SelectedProvenanceScope(auxiliary, {}),
                SelectedProvenanceScope(last, {"data": 0, "correction": 1}),
            )
        )

    source = xr.DataArray([10.0])
    code = selected_steps_code(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Offset sample",
                    code="from math import *\nresult = source + sqrt(4)",
                )
            ),
        ),
        input_name="source",
        output_name="result",
    )
    xr.testing.assert_identical(
        _exec_generated_code(code, {"source": source})["result"], source + 2
    )


def test_selected_notebook_scopes_share_framework_imports() -> None:
    source = xr.DataArray(
        np.arange(6.0).reshape(3, 2), dims=("x", "y"), coords={"x": [0.0, 1.0, 2.0]}
    )
    operation = GaussianFilterOperation(sigma={"x": 0.5})
    first = selected_steps_spec(
        (ReplayStep(operation=operation),), input_name="data", output_name="derived"
    )
    last = selected_steps_spec(
        (ReplayStep(operation=operation),), input_name="data", output_name="smoothed"
    )
    code = selected_provenance_code(
        (SelectedProvenanceScope(first, {}), SelectedProvenanceScope(last, {"data": 0}))
    )
    namespace = _exec_generated_code(code, {"data": source})

    xr.testing.assert_identical(
        namespace["smoothed"], operation.apply(operation.apply(source))
    )
    assert code.count("import erlab") == 1
    assert ".copy(" not in code


def test_selected_notebook_scopes_share_module_prologues() -> None:
    source = xr.DataArray([10.0])
    first = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Prepare sample",
                    code=(
                        '"""Prepare the sample."""\n'
                        "from __future__ import annotations\n"
                        "def normalize(value: MissingType = derived):\n"
                        "    return value + 1\n"
                        "derived = normalize()"
                    ),
                )
            ),
        ),
        input_name="derived",
        output_name="derived",
    )
    last = selected_steps_spec(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Scale sample",
                    code=(
                        '"""Scale the sample."""\n'
                        "from __future__ import annotations\n"
                        "derived = normalize() * 2"
                    ),
                )
            ),
        ),
        input_name="derived",
        output_name="derived",
    )
    code = selected_provenance_code(
        (
            SelectedProvenanceScope(first, {}),
            SelectedProvenanceScope(last, {"derived": 0}),
        )
    )
    namespace = _exec_generated_code(code, {"derived": source})

    xr.testing.assert_identical(namespace["derived"], (source + 1) * 2)
    assert namespace["__doc__"] == "Prepare the sample."
    assert code.count("from __future__ import annotations") == 1


def test_selected_steps_code_preserves_input_named_like_output() -> None:
    source = xr.DataArray([8.0])
    reference = xr.DataArray([2.0])
    code = selected_steps_code(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Divide by reference", code="derived = data / derived"
                )
            ),
        ),
        input_name="data",
        output_name="derived",
    )
    namespace = {"data": source, "derived": reference}
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(namespace["derived"], source / reference)
    assert ".copy(" not in code


@pytest.mark.parametrize(
    "expression", ["np.add(trigger.value, derived)", "trigger.value + np.abs(derived)"]
)
def test_selected_steps_code_preserves_input_before_effectful_property(
    expression: str,
) -> None:
    source = xr.DataArray([2.0])
    namespace = {"data": source, "np": np}

    class Trigger:
        @property
        def value(self) -> float:
            namespace["data"] = xr.DataArray([100.0])
            return 1.0

    namespace["trigger"] = Trigger()
    code = selected_steps_code(
        (
            ReplayStep(
                operation=ScriptCodeOperation(
                    label="Read input after property", code=f"derived = {expression}"
                )
            ),
        ),
        input_name="data",
        input_alias="derived",
        output_name="derived",
    )
    exec(code, namespace)  # noqa: S102

    xr.testing.assert_identical(namespace["derived"], source + 1.0)
    xr.testing.assert_identical(namespace["data"], xr.DataArray([100.0]))


def test_rebuild_captures_raw_and_source_checkpoints_on_cache_hits(
    tmp_path: pathlib.Path,
) -> None:
    from erlab.interactive.imagetool.manager._revisions import _ProvenanceRevisionStore

    data = xr.DataArray(np.arange(12.0).reshape(4, 3), dims=("x", "y"))
    path = tmp_path / "raw-checkpoint.nc"
    data.to_netcdf(path)
    source = selection(
        IselOperation(kwargs={"x": slice(1, None)}),
        IselOperation(kwargs={"x": slice(1, None)}),
    )
    spec = script(
        start_label="Use selected source",
        active_name="data",
        primary_input="data",
        script_inputs=(
            ScriptInput(
                name="data", provenance_spec=_file_spec(path), source_spec=source
            ),
        ),
    )
    cache = {}
    for _attempt in range(2):
        revisions = _ProvenanceRevisionStore()

        def capture(
            value: xr.DataArray, item: ScriptInput, *, revisions=revisions
        ) -> ScriptInput:
            revision_id = revisions.add(
                value,
                item.parsed_provenance_spec(),
                primary_revision_id=item.revision_id,
            )
            return item.model_copy(update={"revision_id": revision_id})

        result, rebuilt = rebuild_script_provenance(
            spec, cache=cache, _capture_input_revision=capture
        )
        owner = rebuilt.script_inputs[0]
        xr.testing.assert_identical(result, data.isel(x=slice(2, None)))
        xr.testing.assert_identical(revisions.resolve_data(owner.revision_id), data)
        xr.testing.assert_identical(
            revisions.resolve_data(owner.resolved_revision_id), result
        )
        intermediate = [
            revisions.resolve_data(record.revision_id)
            for record in revisions.reachable((owner.resolved_revision_id,))
        ]
        assert any(value.sizes["x"] == 3 for value in intermediate)
        if _attempt == 1:
            result.values[:] = -1
            xr.testing.assert_identical(revisions.resolve_data(owner.revision_id), data)


def test_composed_primary_source_runs_once(tmp_path: pathlib.Path) -> None:
    raw = xr.DataArray(np.arange(20.0).reshape(5, 4), dims=("x", "y"))
    path = tmp_path / "composed-source.nc"
    raw.to_netcdf(path)
    local = recipe(
        ReplayStep(operation=AverageOperation(dims=("y",))),
        start_label="Average source",
        active_name="average",
        primary_input="data",
        script_inputs=(
            ScriptInput(
                name="data",
                source_spec=selection(IselOperation(kwargs={"x": slice(1, None)})),
            ),
        ),
    )
    composed = compose_full_provenance(_file_spec(path), local)
    expected = raw.isel(x=slice(1, None)).mean("y")
    xr.testing.assert_identical(
        execute_replay_graph(compile_replay_graph(composed)), expected
    )
    code = emit_replay_code(
        compile_replay_graph(composed, display=True), output_name="average"
    )
    xr.testing.assert_identical(_exec_generated_code(code)["average"], expected)


def test_composition_preserves_pinned_root_provider(tmp_path: pathlib.Path) -> None:
    raw = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))
    path = tmp_path / "root-source.nc"
    raw.to_netcdf(path)
    owner = ScriptInput(
        name="base",
        revision_id="captured",
        follow_source=False,
        provenance_spec=_file_spec(path),
    )
    parent = script(start_label="Use saved input", active_name="base", root_input=owner)
    local = script(
        ScriptCodeOperation(label="Offset data", code="result = base + 2"),
        start_label="Offset data",
        active_name="result",
    )
    composed = compose_full_provenance(parent, local)
    path.unlink()
    graph = compile_replay_graph(
        composed,
        trusted_user_code=True,
        live_input_resolver=lambda item: _graph.ResolvedReplayInput(raw, item, "raw"),
    )
    xr.testing.assert_identical(
        execute_replay_graph(graph, authorize=_authorize_execution), raw + 2
    )
    assert composed.root_input == owner


@pytest.mark.parametrize("root", [False, True])
def test_input_source_extension_controls_reload_capability(
    tmp_path: pathlib.Path, root: bool
) -> None:
    raw = xr.DataArray([1.0], dims="x")
    path = tmp_path / "source-capability.nc"
    raw.to_netcdf(path)
    source = full_data(
        ExtensionRoutineOperation(
            script_name="routine.py",
            source_hash="a" * 64,
            routine_id="scale",
            routine_name="Scale",
            parameters={},
        )
    )
    item = ScriptInput(
        name="data", provenance_spec=_file_spec(path), source_spec=source
    )
    spec = script(
        start_label="Use input",
        active_name="data",
        **(
            {"root_input": item}
            if root
            else {"primary_input": "data", "script_inputs": (item,)}
        ),
    )
    for status in ("ready", "disabled"):
        calls = []

        def capability(
            script_name, source_hash, kind, capability_id, *, status=status, calls=calls
        ):
            calls.append((script_name, source_hash, kind, capability_id))
            return status

        assert _execution.can_reload_without_trust(
            spec, extension_status_resolver=capability
        ) == (status == "ready")
        assert calls == [("routine.py", "a" * 64, "routine", "scale")]


def test_raw_provider_replaces_stale_selected_checkpoint_ancestry() -> None:
    from erlab.interactive.imagetool.manager._revisions import _ProvenanceRevisionStore

    raw = xr.DataArray(np.arange(6.0), dims="x")
    revisions = _ProvenanceRevisionStore()
    raw_id = revisions.add(raw)
    owner = ScriptInput(
        name="data",
        revision_id=raw_id,
        resolved_revision_id="stale-selected",
        source_spec=selection(IselOperation(kwargs={"x": slice(1, None)})),
    )
    spec = script(
        start_label="Use selected input", active_name="data", root_input=owner
    )
    parents = []

    def capture(value: xr.DataArray, item: ScriptInput) -> ScriptInput:
        parents.append(item.revision_id)
        revision_id = revisions.add(
            value, item.parsed_provenance_spec(), primary_revision_id=item.revision_id
        )
        return item.model_copy(update={"revision_id": revision_id})

    result, rebuilt = rebuild_script_provenance(
        spec,
        live_input_resolver=lambda item: _graph.ResolvedReplayInput(raw, item, "raw"),
        _capture_input_revision=capture,
    )
    xr.testing.assert_identical(result, raw.isel(x=slice(1, None)))
    assert parents[0] == raw_id
    assert "stale-selected" not in parents
    assert rebuilt.root_input.revision_id == raw_id
    assert raw_id in {
        record.revision_id
        for record in revisions.reachable((rebuilt.root_input.resolved_revision_id,))
    }
