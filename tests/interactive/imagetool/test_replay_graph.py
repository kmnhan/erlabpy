import ast
import pathlib
import re
import types
import typing
from collections.abc import Mapping

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive.imagetool import _provenance_framework, _replay_graph, provenance


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
    return provenance.FileLoadSource(
        path=str(path),
        loader_label="xarray.load_dataarray",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        replay_call=provenance.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            selected_index=selected_index,
        ),
        load_code=load_code,
    )


def _file_spec(path: pathlib.Path | str, *, selected_index: int = 0):
    return provenance.file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=_file_replay_source(path, selected_index=selected_index),
    )


def _erlab_file_spec(path: pathlib.Path | str, loader: str):
    return provenance.file_load(
        start_label=f"Load {path}",
        seed_code=(
            f"erlab.io.set_loader({loader!r})\nderived = erlab.io.load({str(path)!r})"
        ),
        file_load_source=provenance.FileLoadSource(
            path=str(path),
            loader_label="Loader",
            loader_text=loader,
            kwargs_text="",
            replay_call=provenance.FileReplayCall(
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

    assert _replay_graph._code_uses_name("derived = data", "data")
    assert not _replay_graph._code_uses_name("derived =", "data")
    assert (
        _replay_graph._simple_assignment_source_name("target = source", "target")
        == "source"
    )
    assert (
        _replay_graph._simple_assignment_source_name(
            "target: xr.DataArray = source",
            "target",
        )
        == "source"
    )
    assert _replay_graph._simple_assignment_source_name("target =", "target") is None
    assert (
        _replay_graph._simple_assignment_source_name(
            "target = source\nother = source", "target"
        )
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
    names = [_replay_graph._statement_scope_names(stmt) for stmt in module.body]
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

    deps = _replay_graph._script_function_dependencies(
        "def helper():\n"
        "    def nested():\n"
        "        return missing\n"
        "    return nested()\n"
    )
    assert deps[("helper", 1)] == {"missing"}
    with pytest.raises(_replay_graph.ReplayGraphError, match="unresolved name"):
        _replay_graph._validate_script_code_names(
            "def helper():\n    return missing\nresult = helper()",
            set(),
            {},
        )
    loop_names = {"axs", "profiles"}
    _replay_graph._validate_script_code_names(
        "for profile in profiles:\n    profile.plot(ax=axs, x='alpha')",
        loop_names,
        {},
    )
    assert "profile" in loop_names
    nested_loop_names = {"axs", "profiles", "show_profiles"}
    _replay_graph._validate_script_code_names(
        "if show_profiles:\n"
        "    for profile in profiles:\n"
        "        profile.plot(ax=axs, x='alpha')",
        nested_loop_names,
        {},
    )
    assert "profile" in nested_loop_names
    loop_else_names = {"items"}
    _replay_graph._validate_script_code_names(
        "for item in items:\n    pass\nelse:\n    derived = item",
        loop_else_names,
        {},
    )
    assert {"item", "derived"}.issubset(loop_else_names)
    comprehension_names = {"float", "profiles", "sum"}
    _replay_graph._validate_script_code_names(
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
    comprehension_condition_names = _replay_graph._statement_scope_names(
        ast.parse("values = [item for item in data if item > threshold]").body[0]
    )
    assert {"data", "threshold"}.issubset(comprehension_condition_names.loads)
    assert "item" not in comprehension_condition_names.loads
    generated_builtin_names = {
        *_provenance_framework._SCRIPT_REPLAY_ALLOWED_BUILTINS,
        "values",
    }
    _replay_graph._validate_script_code_names(
        "indexed = [value for index, value in enumerate(values)]\n"
        "reordered = list(reversed(values))",
        generated_builtin_names,
        {},
    )
    assert {"indexed", "reordered"}.issubset(generated_builtin_names)
    with pytest.raises(_replay_graph.ReplayGraphError, match="unresolved name"):
        _replay_graph._validate_script_code_names(
            "line_color_values = [missing + profile for profile in profiles]",
            {"profiles"},
            {},
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="unresolved name"):
        _replay_graph._validate_script_code_names(
            "for holder.profile in profiles:\n    pass",
            {"profiles"},
            {},
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="unresolved name"):
        _replay_graph._validate_script_code_names(
            "if use_left:\n    local_value = data\nelse:\n    derived = local_value",
            {"data", "use_left"},
            {},
        )
    unchanged_dependencies = {"helper": {"data"}}
    _replay_graph._validate_script_code_names(
        "if use_left:\n    left = data\nelse:\n    right = data",
        {"data", "use_left", "helper"},
        unchanged_dependencies,
    )
    assert unchanged_dependencies == {"helper": {"data"}}
    new_branch_dependencies: dict[str, set[str]] = {}
    _replay_graph._validate_script_code_names(
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
    _replay_graph._validate_script_code_names(
        "try:\n    data\nexcept ValueError as exc:\n    derived = exc",
        exception_names,
        {},
    )

    with pytest.raises(_replay_graph.ReplayGraphError, match="Expected script"):
        _replay_graph._validate_script_provenance(
            provenance.full_data(provenance.SqueezeOperation())
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="without active_name"):
        _replay_graph._validate_script_provenance(
            types.SimpleNamespace(kind="script", active_name=None)
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="unsupported Import"):
        _replay_graph._validate_script_provenance(
            provenance.script(
                start_label="Run script",
                seed_code="import os",
                active_name="derived",
                script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
            )
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="no replay code"):
        _replay_graph._validate_script_provenance(
            provenance.script(
                provenance.AverageOperation(dims=("x",)),
                start_label="Run script",
                active_name="derived",
            )
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="no replay code"):
        _replay_graph._validate_script_provenance(
            provenance.script(start_label="Run script", active_name="derived")
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="non-replayable"):
        _replay_graph._validate_script_provenance(
            provenance.script(
                provenance.ScriptCodeOperation(
                    label="Opaque",
                    code=None,
                    copyable=False,
                ),
                start_label="Run script",
                active_name="derived",
                script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
            )
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="no replay code"):
        _replay_graph._validate_script_provenance(
            provenance.script(
                start_label="Run script",
                active_name="derived",
                replay_stages=(
                    provenance.ReplayStage(
                        source_kind="full_data",
                        operations=(provenance.AverageOperation(dims=("x",)),),
                    ),
                ),
            )
        )
    invalid_stage_spec = provenance.script(
        start_label="Run script",
        seed_code="derived = 1",
        active_name="derived",
    ).model_copy(
        update={
            "replay_stages": (
                provenance.ReplayStage.model_construct(
                    source_kind="full_data",
                    operations=(
                        provenance.ScriptCodeOperation(
                            label="Opaque",
                            code=None,
                            copyable=False,
                        ),
                    ),
                ),
            )
        }
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="non-replayable"):
        _replay_graph._validate_script_provenance(invalid_stage_spec)
    assert not _replay_graph.script_provenance_replayable(None)
    assert not _replay_graph._script_provenance_validates(
        provenance.script(
            start_label="Run script",
            seed_code="derived =",
            active_name="derived",
        ),
        strict_replay_code=False,
    )
    assert not _replay_graph._script_provenance_validates(
        provenance.script(
            provenance.ScriptCodeOperation(label="Broken", code="derived ="),
            start_label="Run script",
            seed_code="derived = data",
            active_name="derived",
        ),
        strict_replay_code=False,
    )
    assert _replay_graph._single_assignment_output_name("derived =") is None
    assert _replay_graph._single_assignment_output_name("obj.value = data") is None
    assert (
        _replay_graph._single_assignment_output_name("first = data\nsecond = data")
        is None
    )
    assert _replay_graph.script_provenance_trust_key(None) is None


def test_replay_graph_script_context_binding_error_paths() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    invalid_context = provenance.ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        active_name="result",
        operations=(
            provenance.ScriptCodeOperation(
                label="Use pasted context",
                code="result = data + 1",
            ),
        ),
        script_context_bindings=[
            {"operation_index": 0, "names": ["data"]},
        ],
    )

    with pytest.raises(_replay_graph.ReplayGraphError, match="no replay code"):
        _replay_graph._validate_script_provenance(invalid_context)

    rebound_input = provenance.ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        active_name="data_0",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
        operations=(
            provenance.ScriptCodeOperation(
                label="Offset rebound input",
                code="data_0 = derived + 1",
            ),
        ),
        script_context_bindings=[
            {"operation_index": 0, "names": ["derived"]},
        ],
    )
    rebound_graph = _replay_graph.compile_replay_graph(
        rebound_input,
        external_inputs={"data_0": data},
    )
    rebound_display_graph = _replay_graph.compile_replay_graph(
        rebound_input,
        display=True,
        external_inputs={"data_0": data},
    )
    xr.testing.assert_identical(
        _replay_graph.execute_replay_graph(rebound_graph),
        data + 1,
    )
    assert rebound_display_graph.output_key is not None

    active_relay = provenance.ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="result = data",
        active_name="result",
        operations=(
            provenance.ScriptCodeOperation(
                label="Write alternate output",
                code="derived = result + 1",
            ),
        ),
    )
    active_graph = _replay_graph.compile_replay_graph(
        active_relay,
        external_inputs={"data": data},
    )

    xr.testing.assert_identical(
        _replay_graph.execute_replay_graph(active_graph),
        data,
    )
    assert _replay_graph._remove_noop_assignments("derived =") == "derived ="


def test_replay_graph_manual_error_and_cache_paths() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))

    graph = _replay_graph.ReplayGraph()
    live_key = graph.add_node("live", "live_input", payload={"data": data})
    graph.output_key = live_key
    with pytest.raises(_replay_graph.ReplayGraphError, match="Live inputs"):
        _replay_graph.emit_replay_code(graph)
    xr.testing.assert_identical(
        _replay_graph.execute_replay_graph(graph, cache={live_key: data + 1.0}),
        data + 1.0,
    )
    replayed = _replay_graph.execute_replay_graph(graph)
    xr.testing.assert_identical(replayed, data)
    assert not np.shares_memory(replayed.data, data.data)

    relay_graph = _replay_graph.ReplayGraph()
    relay_live_key = relay_graph.add_node("live", "live_input", payload={"data": data})
    relay_key = relay_graph.add_node("relay", "relay", parents=(relay_live_key,))
    relay_graph.output_key = relay_key
    xr.testing.assert_identical(_replay_graph.execute_replay_graph(relay_graph), data)

    for load_code, message in (
        ("derived =", "not valid Python"),
        ("other = xr.DataArray([1.0], dims=('x',))", "does not assign"),
    ):
        file_graph = _replay_graph.ReplayGraph()
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
        with pytest.raises(_replay_graph.ReplayGraphError, match=message):
            _replay_graph.emit_replay_code(file_graph)

    for codes, message in (
        (("other = xr.DataArray([1.0], dims=('x',))",), "did not create"),
        (("derived = 1",), "did not produce"),
    ):
        script_graph = _replay_graph.ReplayGraph()
        script_key = script_graph.add_node(
            f"script-{message}",
            "script",
            payload={"bindings": (), "codes": codes, "active_name": "derived"},
        )
        script_graph.output_key = script_key
        with pytest.raises(_replay_graph.ReplayGraphError, match=message):
            _replay_graph.execute_replay_graph(script_graph)

    unknown_graph = _replay_graph.ReplayGraph()
    unknown_key = unknown_graph.add_node("unknown", "unknown")
    unknown_graph.output_key = unknown_key
    with pytest.raises(_replay_graph.ReplayGraphError, match="Unknown replay"):
        _replay_graph.emit_replay_code(unknown_graph)
    with pytest.raises(_replay_graph.ReplayGraphError, match="Unknown replay"):
        _replay_graph.execute_replay_graph(unknown_graph)

    empty_graph = _replay_graph.ReplayGraph()
    with pytest.raises(_replay_graph.ReplayGraphError, match="no output"):
        _replay_graph.emit_replay_code(empty_graph, output_name="derived")
    with pytest.raises(_replay_graph.ReplayGraphError, match="no output"):
        _replay_graph.execute_replay_graph(empty_graph)


def test_replay_graph_file_script_input_and_rebuild_edges(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    file_spec = _file_spec(path)

    setup_code, load_code = _replay_graph._file_seed_code_parts(
        "import erlab\nimport numpy as np\nderived = xr.load_dataarray('source.nc')",
        "derived",
    )
    assert setup_code == "import numpy as np"
    assert "xr.load_dataarray" in load_code
    for seed_code, message in (
        ("derived =", "not valid Python"),
        ("other = xr.DataArray([1.0], dims=('x',))", "does not assign"),
    ):
        with pytest.raises(_replay_graph.ReplayGraphError, match=message):
            _replay_graph._file_seed_code_parts(seed_code, "derived")

    loaded_input = provenance.ScriptInput(
        name="loaded",
        label="Loaded source",
        provenance_spec=file_spec,
    )
    code = _replay_graph.script_inputs_code((loaded_input,), display=False)
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["loaded"], data)
    with pytest.raises(_replay_graph.ReplayGraphError, match="recorded source"):
        _replay_graph.script_inputs_code(
            (provenance.ScriptInput(name="missing", label="Missing source"),),
            display=False,
        )

    with pytest.raises(_replay_graph.ReplayGraphError, match="script-derived"):
        _replay_graph.rebuild_script_provenance(file_spec)
    script_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Add one", code="derived = data_0 + 1.0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Loaded source",
                provenance_spec=file_spec,
            ),
        ),
    )
    rebuilt, rebuilt_spec = _replay_graph.rebuild_script_provenance(script_spec)
    xr.testing.assert_identical(rebuilt, data + 1.0)
    assert rebuilt_spec.script_inputs[0].node_uid is None

    with pytest.raises(_replay_graph.ReplayGraphError, match="maximum reload depth"):
        _replay_graph.rebuild_script_provenance(script_spec, depth=21)
    missing_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Add one", code="derived = data_0 + 1.0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Closed input"),),
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="not open"):
        _replay_graph.rebuild_script_provenance(missing_spec)

    live_calls = 0
    initial_marker = "initial-marker"
    current_marker = "current-marker"
    live_input = provenance.ScriptInput(
        name="data_0",
        label="Live input",
        node_uid="uid-0",
        node_snapshot_token=initial_marker,
    )
    live_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Double", code="derived = data_0 * 2.0"),
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

    live_rebuilt, live_rebuilt_spec = _replay_graph.rebuild_script_provenance(
        live_spec,
        live_input_resolver=resolve_live,
    )
    xr.testing.assert_identical(live_rebuilt, data * 2.0)
    assert live_calls == 1
    assert live_rebuilt_spec.script_inputs[0].node_snapshot_token == current_marker

    miss_calls = 0
    duplicate_file_input = provenance.ScriptInput(
        name="data_0",
        label="Closed file input",
        node_uid="same-uid",
        provenance_spec=file_spec,
    )
    duplicate_file_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Copy", code="derived = data_0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(duplicate_file_input, duplicate_file_input),
    )

    def miss_live(_script_input):
        nonlocal miss_calls
        miss_calls += 1
        return

    rebuilt_from_miss, _rebuilt_from_miss_spec = (
        _replay_graph.rebuild_script_provenance(
            duplicate_file_spec,
            live_input_resolver=miss_live,
        )
    )
    xr.testing.assert_identical(rebuilt_from_miss, data)
    assert miss_calls == 2

    unsupported_nested = provenance.script(
        provenance.ScriptCodeOperation(label="Opaque", code=None, copyable=False),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
    )
    unsupported_input = provenance.ScriptInput(
        name="data_0",
        label="Unsupported nested input",
        provenance_spec=unsupported_nested,
    )
    unsupported_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Copy", code="derived = data_0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(unsupported_input,),
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="cannot be replayed"):
        _replay_graph.rebuild_script_provenance(unsupported_spec)

    full_data_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Copy", code="derived = data_0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Full data",
                provenance_spec=provenance.full_data(),
            ),
        ),
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="reloadable"):
        _replay_graph.rebuild_script_provenance(full_data_spec)


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
        ) -> str | None:
            return self._code

    for operation, message in (
        (MissingReplayOperation(), "does not provide"),
        (MissingExpressionOperation(), "does not provide"),
        (Operation(None), "does not provide"),
        (Operation("derived ="), "not valid Python"),
        (Operation("other = data"), "does not assign"),
    ):
        with pytest.raises(_replay_graph.ReplayGraphError, match=message):
            _replay_graph._operation_replay_code(
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
        ) -> str:
            assert input_name == "parent_data"
            assert output_name == "active_data"
            assert source_name == "source_data"
            return f"{output_name} = {input_name} + {source_name}"

    code = _replay_graph._operation_replay_code(
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
    shared_stage = provenance.full_data(provenance.AverageOperation(dims=("k",)))
    left_stage = provenance.selection(
        provenance.SelOperation(kwargs={"pol": "LH"}),
        provenance.SqueezeOperation(),
    )
    right_stage = provenance.selection(
        provenance.SelOperation(kwargs={"pol": "LV"}),
        provenance.SqueezeOperation(),
    )
    left_spec = provenance.compose_full_provenance(
        provenance.compose_full_provenance(file_spec, shared_stage),
        left_stage,
    )
    right_spec = provenance.compose_full_provenance(
        provenance.compose_full_provenance(file_spec, shared_stage),
        right_stage,
    )
    assert left_spec is not None
    assert right_spec is not None
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Subtract", code="derived = data_0 - data_1"
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0", label="LH", provenance_spec=left_spec
            ),
            provenance.ScriptInput(
                name="data_1", label="LV", provenance_spec=right_spec
            ),
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


def test_replay_graph_replays_script_with_preserved_file_stage(
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
        provenance.full_data(provenance.AverageOperation(dims=("x",)))
    )
    local = provenance.script(
        provenance.ScriptCodeOperation(
            label="Center profile",
            code="result = derived - derived.mean()",
        ),
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
    )

    spec = provenance.compose_full_provenance(file_spec, local)
    assert spec is not None
    assert spec.kind == "script"
    assert len(spec.replay_stages) == 1
    assert isinstance(spec.replay_stages[0].operations[0], provenance.AverageOperation)

    replayed = provenance.replay_script_provenance(spec, {})

    expected_input = provenance.AverageOperation(dims=("x",)).apply(
        source,
        parent_data=source,
    )
    xr.testing.assert_identical(replayed, expected_input - expected_input.mean())
    code = typing.cast("str", spec.display_code())
    assert code.count("xr.load_dataarray") == 1
    assert "result =" in code
    xr.testing.assert_identical(_exec_generated_code(code)["result"], replayed)


def test_replay_graph_composes_local_script_stage_after_script_parent() -> None:
    source = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": [0, 1, 2], "y": [10, 20, 30, 40]},
        name="scan",
    )
    parent = provenance.script(
        provenance.ScriptCodeOperation(
            label="Crop source",
            code="derived = derived.isel(x=slice(0, 2))",
        ),
        start_label="Run parent script",
        seed_code="derived = data",
        active_name="derived",
    )
    local = provenance.script(
        provenance.ScriptCodeOperation(
            label="Offset profile",
            code="result = derived + 1",
        ),
        start_label="Run local script",
        active_name="result",
        replay_stages=(
            provenance.ReplayStage.from_source_spec(
                provenance.full_data(provenance.AverageOperation(dims=("x",)))
            ),
        ),
    )

    spec = provenance.compose_full_provenance(parent, local)
    assert spec is not None
    assert spec.kind == "script"
    assert spec.replay_stages == ()

    replayed = provenance.replay_script_provenance(spec, {"data": source})

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
    file_spec = provenance.file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=load_source,
    )
    center_spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Extract center values",
            code="center_values = derived.mean('x')",
        ),
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        active_name="center_values",
        file_load_source=load_source,
    )
    corrected_spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Correct with center values",
            code="derived = data_0 - data_1",
        ),
        provenance.RenameOperation(name="corrected"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="ImageTool 4: scan",
                provenance_spec=file_spec,
            ),
            provenance.ScriptInput(
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
        _replay_graph._script_seed_file_load_parts(
            "print(data)",
            active_name="derived",
            load_source=load_source,
        )
        is None
    )
    assert (
        _replay_graph._script_seed_file_load_parts(
            "derived =",
            active_name="derived",
            load_source=load_source,
        )
        is None
    )
    assert (
        _replay_graph._script_seed_file_load_parts(
            seed_code,
            active_name="derived",
            load_source=load_source.model_copy(update={"load_code": None}),
        )
        is None
    )
    assert (
        _replay_graph._script_seed_file_load_parts(
            seed_code,
            active_name="derived",
            load_source=load_source.model_copy(
                update={"load_code": "data.value = xr.load_dataarray('scan.nc')"}
            ),
        )
        is None
    )
    assert (
        _replay_graph._script_seed_file_load_parts(
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
                        "selection": provenance.FileDataSelection(
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
    first_spec = provenance.file_load(
        start_label="Load first",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=load_source,
    )
    second_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Copy", code="derived = derived"),
        start_label="Load second",
        seed_code=typing.cast("str", other_load_source.load_code).replace(
            "data =",
            "derived =",
            1,
        ),
        active_name="derived",
        file_load_source=other_load_source,
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(label="Add", code="derived = data_0 + data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            provenance.ScriptInput(
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
    first_spec = provenance.file_load(
        start_label="Load first",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=load_source,
    )
    mismatched_script_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Copy", code="derived = derived"),
        start_label="Load mismatched",
        seed_code=f"derived = xr.load_dataarray({str(other_path)!r})",
        active_name="derived",
        file_load_source=load_source,
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(label="Add", code="derived = data_0 + data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=first_spec,
            ),
            provenance.ScriptInput(
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
    spec = provenance.script(
        provenance.AverageOperation(dims=("alpha",)),
        start_label="Run script",
        seed_code="avg = data_0",
        active_name="avg",
        script_inputs=(
            provenance.ScriptInput(
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

    graph = _replay_graph.compile_replay_graph(spec)
    xr.testing.assert_identical(
        _replay_graph.execute_replay_graph(graph),
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
    spec = provenance.script(
        provenance.SelOperation(kwargs={"polarization": -1}),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
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
    spec = provenance.script(
        provenance.AverageOperation(dims=("x",)),
        provenance.ScriptCodeOperation(
            label="Use original input",
            code="derived = derived + data_0.qsel.average('x')",
        ),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
    )

    graph = _replay_graph.compile_replay_graph(spec, external_inputs={"data_0": data})
    script_nodes = [node for node in graph.nodes if node.kind == "script"]

    assert script_nodes
    assert any(
        input_name == "data_0"
        for node in script_nodes
        for input_name, _key in node.payload["bindings"]
    )
    xr.testing.assert_identical(
        _replay_graph.execute_replay_graph(graph),
        data.qsel.average("x") + data.qsel.average("x"),
    )


def test_replay_graph_keeps_structured_operations_in_opaque_script() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = provenance.script(
        provenance.AverageOperation(dims=("x",)),
        provenance.ScriptCodeOperation(
            label="Use temp",
            code="derived = derived + tmp.qsel.average('x') + data_0.qsel.average('x')",
        ),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
    )

    graph = _replay_graph.compile_replay_graph(spec, external_inputs={"data_0": data})
    script_nodes = [node for node in graph.nodes if node.kind == "script"]

    assert len(script_nodes) == 1
    assert not any(node.kind == "operation" for node in graph.nodes)
    assert any(
        "derived = derived.qsel.mean(" in code
        for code in script_nodes[0].payload["codes"]
    )
    xr.testing.assert_identical(
        _replay_graph.execute_replay_graph(graph),
        (data + 1).qsel.average("x")
        + (data + 1).qsel.average("x")
        + data.qsel.average("x"),
    )


def test_replay_graph_rebases_context_for_inlined_operation() -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
    )
    spec = provenance.script(
        provenance.SortCoordOrderOperation(),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
    )

    graph = _replay_graph.compile_replay_graph(spec, external_inputs={"data_0": data})
    script_nodes = [node for node in graph.nodes if node.kind == "script"]
    inlined_code = "\n".join(script_nodes[0].payload["codes"])

    assert provenance.script_provenance_replayable(spec)
    assert "data.coords" not in inlined_code
    assert "derived.coords.keys()" in inlined_code
    xr.testing.assert_identical(
        provenance.replay_script_provenance(spec, {"data_0": data}),
        erlab.utils.array.sort_coord_order(data + 1, (data + 1).coords.keys()),
    )


def test_replay_graph_rebases_context_in_script_input_code(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
    )
    data.to_netcdf(path)
    spec = provenance.script(
        provenance.SortCoordOrderOperation(),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    graph = _replay_graph.compile_replay_graph(spec, display=False)
    code = _replay_graph.emit_replay_code(graph, output_name="derived")

    assert "data.coords" not in code
    assert "derived.coords.keys()" in code
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(
        namespace["derived"],
        erlab.utils.array.sort_coord_order(data + 1, (data + 1).coords.keys()),
    )


def test_replay_graph_shares_structured_console_alias_prefixes(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "polarization.nc"
    source = _polarization_source(path)
    averaged = provenance.script(
        provenance.AverageOperation(dims=("k",)),
        start_label="Run script",
        seed_code="avg = data_0",
        active_name="avg",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Scan",
                provenance_spec=_file_spec(path),
            ),
        ),
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Subtract", code="derived = data_0 - data_1"
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(name="data_0", label="LH", provenance_spec=averaged),
            provenance.ScriptInput(name="data_1", label="LV", provenance_spec=averaged),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    graph = _replay_graph.compile_replay_graph(spec)

    assert code.count("xr.load_dataarray") == 1
    assert code.count(".qsel.mean") == 1
    assert ".copy(" not in code
    assert (
        sum(
            node.kind == "operation" and node.payload["operation"].op == "average"
            for node in graph.nodes
        )
        == 1
    )
    expected = source.qsel.mean("k") - source.qsel.mean("k")
    xr.testing.assert_identical(_exec_generated_code(code)["derived"], expected)
    xr.testing.assert_identical(_replay_graph.execute_replay_graph(graph), expected)


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
    processed = provenance.compose_full_provenance(
        _file_spec(path),
        provenance.public_data(
            provenance.DivideByCoordOperation(coord_name="mesh_current"),
            provenance.CoarsenOperation(
                dim={"alpha": 3, "eV": 5},
                boundary="trim",
                side="left",
                coord_func="mean",
                reducer="mean",
            ),
        ),
    )
    assert processed is not None

    rc = provenance.script(
        provenance.SelOperation(kwargs={"polarization": -1}),
        start_label="Run ImageTool manager console code",
        seed_code="rc = data_0",
        active_name="rc",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Processed map",
                provenance_spec=processed,
            ),
        ),
    )
    lc = provenance.script(
        provenance.SelOperation(kwargs={"polarization": 1}),
        start_label="Run ImageTool manager console code",
        seed_code="lc = data_0",
        active_name="lc",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Processed map",
                provenance_spec=processed,
            ),
        ),
    )
    diff = provenance.script(
        provenance.ScriptCodeOperation(
            label="Evaluate console expression",
            code="derived = rc - lc",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="rc", label="console variable 'rc'", provenance_spec=rc
            ),
            provenance.ScriptInput(
                name="lc", label="console variable 'lc'", provenance_spec=lc
            ),
        ),
    )
    total = provenance.script(
        provenance.ScriptCodeOperation(
            label="Evaluate console expression",
            code="derived = rc + lc",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="rc", label="console variable 'rc'", provenance_spec=rc
            ),
            provenance.ScriptInput(
                name="lc", label="console variable 'lc'", provenance_spec=lc
            ),
        ),
    )
    ncd = provenance.script(
        provenance.ScriptCodeOperation(
            label="Evaluate console expression",
            code="ncd = data_1 / data_2",
        ),
        start_label="Run ImageTool manager console code",
        active_name="ncd",
        script_inputs=(
            provenance.ScriptInput(
                name="data_1", label="ImageTool 1", provenance_spec=diff
            ),
            provenance.ScriptInput(
                name="data_2", label="ImageTool 2", provenance_spec=total
            ),
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
    assert "data_1 =" not in code
    assert "data_2 =" not in code
    assert "derived" not in code
    assert "ncd = (rc - lc) / (rc + lc)" in code
    _assert_dense_replay_temps(code)
    xr.testing.assert_identical(namespace["ncd"], expected)


def test_replay_graph_cleanup_helpers_cover_edge_cases() -> None:
    assert _replay_graph._inline_single_use_replay_expressions("bad =") == "bad ="
    assert _replay_graph._compact_replay_temp_names("bad =") == "bad ="
    assert _replay_graph._code_has_scoped_definition("bad =")

    assert (
        _replay_graph._inline_single_use_replay_expressions(
            "_itool_replay_1 = source.attr\nresult = _itool_replay_1 + 1"
        )
        == "result = source.attr + 1"
    )
    assert (
        _replay_graph._inline_single_use_replay_expressions(
            "_itool_replay_1 = source\nresult = 0"
        )
        == "_itool_replay_1 = source\nresult = 0"
    )
    assert (
        _replay_graph._inline_single_use_replay_expressions(
            "_itool_replay_1 = source\nresult = _itool_replay_1 + _itool_replay_1"
        )
        == "_itool_replay_1 = source\nresult = _itool_replay_1 + _itool_replay_1"
    )
    assert (
        _replay_graph._inline_single_use_replay_expressions(
            "_itool_replay_1 = source\n"
            "_itool_replay_1 = other\n"
            "result = _itool_replay_1"
        )
        == "_itool_replay_1 = source\nresult = other"
    )
    assert (
        _replay_graph._inline_single_use_replay_expressions(
            "_itool_replay_1 = source\nsource = other\nresult = _itool_replay_1"
        )
        == "_itool_replay_1 = source\nsource = other\nresult = _itool_replay_1"
    )
    assert (
        _replay_graph._inline_single_use_replay_expressions(
            "_itool_replay_1 = source.method()\nresult = _itool_replay_1"
        )
        == "_itool_replay_1 = source.method()\nresult = _itool_replay_1"
    )

    assert (
        _replay_graph._compact_replay_temp_names(
            "_itool_replay_4 = data\n_itool_replay_8 = _itool_replay_4 + 1"
        )
        == "_itool_replay_0 = data\n_itool_replay_1 = _itool_replay_0 + 1"
    )
    assert (
        _replay_graph._compact_replay_temp_names(
            "_itool_replay_0 = 'reserved'\n"
            "_itool_replay_4 = data\n"
            "result = _itool_replay_4"
        )
        == "_itool_replay_0 = 'reserved'\n"
        "_itool_replay_1 = data\n"
        "result = _itool_replay_1"
    )


def test_replay_graph_emit_reports_script_rewrite_syntax_errors(monkeypatch) -> None:

    graph = _replay_graph.ReplayGraph(display=True)
    source_key = graph.add_node(
        "source",
        "file_load",
        payload={"active_name": "loaded", "load_code": "loaded = data"},
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

    original_replace = _provenance_framework._replace_code_identifiers

    def _raise_on_input_replacement(code: str, replacements: Mapping[str, str]) -> str:
        if "data_0" in replacements:
            raise SyntaxError("bad script input")
        return original_replace(code, replacements)

    monkeypatch.setattr(
        _provenance_framework,
        "_replace_code_identifiers",
        _raise_on_input_replacement,
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="Script replay code"):
        _replay_graph.emit_replay_code(graph)

    graph = _replay_graph.ReplayGraph(display=True)
    graph.output_key = graph.add_node(
        "script",
        "script",
        payload={
            "codes": ("bad =",),
            "active_name": "derived",
            "bindings": (),
        },
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="Script replay code"):
        _replay_graph.emit_replay_code(graph, output_name="result")


def test_replay_graph_display_promotes_ui_style_script_input_names(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    source = _polarization_source(path)
    left = provenance.compose_full_provenance(
        _file_spec(path),
        provenance.full_data(provenance.SelOperation(kwargs={"pol": "LH"})),
    )
    right = provenance.compose_full_provenance(
        _file_spec(path),
        provenance.full_data(provenance.SelOperation(kwargs={"pol": "LV"})),
    )
    assert left is not None
    assert right is not None
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Concatenate selected inputs",
            code="combined = xr.concat([left, right], dim='pol')",
        ),
        start_label="Run ImageTool manager UI action",
        active_name="combined",
        script_inputs=(
            provenance.ScriptInput(
                name="left", label="Selected left", provenance_spec=left
            ),
            provenance.ScriptInput(
                name="right", label="Selected right", provenance_spec=right
            ),
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
    rc = provenance.script(
        start_label="Start from watched variable 'rc'",
        seed_code="derived = rc",
        active_name="derived",
    )
    lc = provenance.script(
        start_label="Start from watched variable 'lc'",
        seed_code="derived = lc",
        active_name="derived",
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Subtract selected inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager action",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0", label="ImageTool 0", provenance_spec=rc
            ),
            provenance.ScriptInput(
                name="data_1", label="ImageTool 1", provenance_spec=lc
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"rc": rc_data, "lc": lc_data})

    assert code == "derived = rc - lc"
    xr.testing.assert_identical(namespace["derived"], rc_data - lc_data)


def test_replay_graph_display_keeps_helpers_from_raw_seed_inputs() -> None:
    raw_data = xr.DataArray(np.arange(1.0, 4.0), dims=("x",))
    root = provenance.script(
        start_label="Start from watched variable 'raw'",
        seed_code="def normalize():\n    return raw / raw.max()\nderived = normalize()",
        active_name="derived",
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Average normalized input",
            code="result = data_0.mean()",
        ),
        start_label="Run ImageTool manager action",
        active_name="result",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0", label="ImageTool 0", provenance_spec=root
            ),
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
    public_spec = provenance.compose_full_provenance(
        _file_spec(path),
        provenance.public_data(provenance.SelOperation(kwargs={"x": 0.2})),
    )
    assert public_spec is not None
    script_input = provenance.ScriptInput(
        name="selected",
        label="Selected nonuniform data",
        provenance_spec=public_spec,
    )

    replay_code = _replay_graph.script_inputs_code((script_input,), display=False)
    display_code = _replay_graph.script_inputs_code((script_input,), display=True)

    assert "restore_nonuniform_dims" in replay_code
    assert "restore_nonuniform_dims" not in display_code
    xr.testing.assert_identical(
        _exec_generated_code(display_code)["selected"],
        source.sel(x=0.2),
    )


def test_replay_graph_display_keeps_bindings_for_scoped_or_rebound_inputs(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "scan.nc"
    source = xr.DataArray(np.arange(3.0), dims=("x",))
    source.to_netcdf(path)
    script_input = provenance.ScriptInput(
        name="data_0",
        label="ImageTool 0",
        provenance_spec=_file_spec(path),
    )
    helper_spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Use helper",
            code="def offset():\n    return data_0 + 1\nresult = offset()",
        ),
        start_label="Run script",
        active_name="result",
        script_inputs=(script_input,),
    )
    rebound_spec = provenance.script(
        provenance.ScriptCodeOperation(
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
    assert len(re.findall(r"^data_0 =", rebound_code, flags=re.MULTILINE)) == 2
    xr.testing.assert_identical(_exec_generated_code(helper_code)["result"], source + 1)
    xr.testing.assert_identical(
        _exec_generated_code(rebound_code)["result"],
        (source + 1) * 2,
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
    spec = provenance.script(
        provenance.RenameOperation(name="renamed"),
        start_label="Run script",
        seed_code=seed_code,
        active_name=active_name,
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
    )

    graph = _replay_graph.compile_replay_graph(spec, external_inputs={"data_0": data})
    replayed = _replay_graph.execute_replay_graph(graph)

    assert any(node.kind == "relay" for node in graph.nodes)
    assert replayed.name == "renamed"
    assert not np.shares_memory(replayed.data, data.data)


def test_replay_graph_disables_numbagg_only_during_execution() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Record option",
            code=(
                "derived = data_0.copy()\n"
                "derived.attrs['use_numbagg_during_replay'] = "
                "xr.get_options()['use_numbagg']"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
    )
    graph = _replay_graph.compile_replay_graph(spec, external_inputs={"data_0": data})

    with xr.set_options(use_numbagg=True):
        replayed = _replay_graph.execute_replay_graph(graph)
        assert xr.get_options()["use_numbagg"] is True

    assert replayed.attrs["use_numbagg_during_replay"] is False


def test_replay_graph_display_preserves_whole_array_rename(
    tmp_path: pathlib.Path,
) -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="source")
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    spec = provenance.script(
        provenance.RenameOperation(name="renamed"),
        start_label="Run script",
        active_name="data_0",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    graph = _replay_graph.compile_replay_graph(
        spec,
        display=True,
    )
    code = _replay_graph.emit_replay_code(graph, output_name="derived")

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
    file_spec = provenance.compose_full_provenance(
        _file_spec(path),
        provenance.full_data(provenance.RenameOperation(name="renamed")),
    )
    live_spec = provenance.full_data(provenance.RenameOperation(name="renamed"))
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
    spec = provenance.script(
        provenance.RenameOperation(name="renamed"),
        provenance.ScriptCodeOperation(
            label="Use DataArray name",
            code="derived = derived.rename(derived.name + '_used')",
        ),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=_file_spec(path),
            ),
        ),
    )

    graph = _replay_graph.compile_replay_graph(
        spec,
        display=True,
    )
    code = _replay_graph.emit_replay_code(graph, output_name="derived")

    assert ".rename(" in code
    xr.testing.assert_identical(
        _exec_generated_code(code)["derived"],
        data.rename("renamed_used"),
    )


def test_script_replayability_does_not_generate_structured_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = provenance.script(
        provenance.RenameOperation(name="renamed"),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="Input"),),
    )

    def fail_derivation_entry(self):
        raise AssertionError("replayability checks must not generate copied code")

    monkeypatch.setattr(
        provenance.RenameOperation, "derivation_entry", fail_derivation_entry
    )

    assert provenance.script_provenance_replayable(spec)


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
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Rotate",
            code=(
                "derived = era.transform.rotate("
                "data_0, 0.0, axes=('x', 'y'), reshape=False)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
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
    spec = provenance.script(
        provenance.RotateOperation(
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
            provenance.ScriptInput(
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
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Subtract", code="derived = data_0 - data_1"
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(name="data_0", label="First", provenance_spec=first),
            provenance.ScriptInput(
                name="data_1", label="Second", provenance_spec=second
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count("xr.load_dataarray") == 2


def test_replay_graph_reuses_shared_loader_setup() -> None:
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Add",
            code="derived = data_0 + data_1",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=_erlab_file_spec("scan0.h5", "example"),
            ),
            provenance.ScriptInput(
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
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Add",
            code="derived = data_0 + data_1 + data_2",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Alpha 0",
                provenance_spec=_erlab_file_spec("alpha0.h5", "alpha"),
            ),
            provenance.ScriptInput(
                name="data_1",
                label="Beta",
                provenance_spec=_erlab_file_spec("beta.h5", "beta"),
            ),
            provenance.ScriptInput(
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
    first_spec = provenance.compose_full_provenance(
        file_spec,
        provenance.full_data(provenance.IselOperation(kwargs={"pol": 0})),
    )
    second_spec = provenance.compose_full_provenance(
        file_spec,
        provenance.selection(provenance.IselOperation(kwargs={"pol": 0})),
    )
    assert first_spec is not None
    assert second_spec is not None
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Subtract", code="derived = data_0 - data_1"
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0", label="Full", provenance_spec=first_spec
            ),
            provenance.ScriptInput(
                name="data_1",
                label="Selection",
                provenance_spec=second_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count(".isel") == 2


def test_replay_graph_script_nodes_are_not_deduplicated() -> None:
    first = provenance.script(
        start_label="Make first",
        seed_code="derived = xr.DataArray([1.0, 2.0], dims=['x'])",
        active_name="derived",
    )
    second = provenance.script(
        start_label="Make second",
        seed_code="derived = xr.DataArray([10.0, 20.0], dims=['x'])",
        active_name="derived",
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(label="Add", code="derived = data_0 + data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(name="data_0", label="First", provenance_spec=first),
            provenance.ScriptInput(
                name="data_1", label="Second", provenance_spec=second
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert code.count("xr.DataArray") == 2
    xr.testing.assert_identical(
        namespace["derived"],
        xr.DataArray([11.0, 22.0], dims=["x"]),
    )


def test_replay_graph_allows_for_loop_script_code() -> None:
    spec = provenance.script(
        provenance.ScriptCodeOperation(
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

    graph = _replay_graph.compile_replay_graph(spec, display=True)
    code = _replay_graph.emit_replay_code(graph, output_name="fig")
    namespace = _exec_generated_code(code)

    assert "for profile in profiles:" in code
    assert len(namespace["fig"]) == 2
    xr.testing.assert_identical(namespace["fig"][0], xr.DataArray(3.0))
    xr.testing.assert_identical(namespace["fig"][1], xr.DataArray(12.0))


def test_replay_graph_allows_for_loop_with_script_input() -> None:
    profile_source = provenance.script(
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
    spec = provenance.script(
        provenance.ScriptCodeOperation(
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
            provenance.ScriptInput(
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
                provenance.ReplayStage(
                    source_kind="public_data",
                    operations=(
                        provenance.DivideByCoordOperation(coord_name="mesh_current"),
                    ),
                ),
            ),
        }
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
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
            provenance.ScriptInput(
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
    script_input = provenance.ScriptInput(
        name="data_0",
        label="ImageTool 0",
        provenance_spec=source_spec,
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="User code",
            code=("import os\nwith open(os.devnull):\n    pass\nderived = data_0 + 1"),
        ),
        start_label="Run user code",
        active_name="derived",
        script_inputs=(script_input,),
    )

    assert not provenance.script_provenance_replayable(spec)
    with pytest.raises(_replay_graph.ReplayGraphError, match="unsupported Import"):
        _replay_graph.compile_replay_graph(spec)

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert "import os" in code
    assert "with open(os.devnull):" in code
    xr.testing.assert_identical(namespace["derived"], source_data + 1)

    unresolved_spec = spec.model_copy(
        update={
            "operations": (
                provenance.ScriptCodeOperation(
                    label="User code",
                    code="derived = data_0 + missing",
                ),
            ),
        },
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="unresolved name"):
        _replay_graph.compile_replay_graph(unresolved_spec, display=True)


def test_replay_graph_trusted_user_code_executes_blocked_constructs() -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",))
    spec = provenance.script(
        provenance.ScriptCodeOperation(
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
        script_inputs=(provenance.ScriptInput(name="data", label="Input"),),
    )

    assert not provenance.script_provenance_replayable(spec)
    assert provenance.script_provenance_requires_trust(spec)
    with pytest.raises(_replay_graph.ReplayGraphError, match="unsupported Import"):
        _replay_graph.compile_replay_graph(spec, external_inputs={"data": data})

    result = provenance.replay_script_provenance(
        spec,
        {"data": data},
        trusted_user_code=True,
    )

    xr.testing.assert_identical(result, data + 1)


def test_replay_graph_trusted_user_code_still_validates_result_type() -> None:
    data = xr.DataArray([1.0], dims=("x",))
    spec = provenance.script(
        provenance.ScriptCodeOperation(label="Bad output", code="derived = 1"),
        start_label="Run user code",
        active_name="derived",
    )

    with pytest.raises(TypeError, match="did not produce"):
        provenance.replay_script_provenance(
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
    nested_spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="User code",
            code="import os\nderived = data_0 + int(os.path.exists(os.devnull))",
        ),
        start_label="Run nested code",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="Input",
                provenance_spec=source_spec,
            ),
        ),
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Use nested",
            code="derived = data_1 * 2",
        ),
        start_label="Run outer code",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_1",
                label="Nested",
                provenance_spec=nested_spec,
            ),
        ),
    )

    assert provenance.script_provenance_replayable(spec)
    assert provenance.script_provenance_requires_trust(spec)
    trust_payload = _replay_graph._script_trust_payload(spec)
    assert trust_payload is not None
    assert trust_payload["inputs"][0]["name"] == "data_1"
    assert trust_payload["inputs"][0]["payload"]["operations"][0]["code"].startswith(
        "import os"
    )
    assert _replay_graph.script_provenance_trust_key(spec) is not None
    with pytest.raises(_replay_graph.ReplayGraphError, match="recorded operation"):
        _replay_graph.rebuild_script_provenance(spec)

    result, rebuilt = _replay_graph.rebuild_script_provenance(
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
    spec = provenance.script(
        provenance.ScriptCodeOperation(label="Unsupported", code=code),
        start_label="Run script",
        active_name="derived",
    )

    with pytest.raises(_replay_graph.ReplayGraphError, match=re.escape(message)):
        _replay_graph.compile_replay_graph(spec, external_inputs={"data": data})


def test_replay_graph_raises_typed_errors_for_unsupported_script() -> None:
    data = xr.DataArray([1.0], dims=("x",))
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Unsupported", code="import os\nderived = data"
        ),
        start_label="Run script",
        active_name="derived",
    )

    with pytest.raises(_replay_graph.ReplayGraphError, match="unsupported Import"):
        _replay_graph.compile_replay_graph(spec, external_inputs={"data": data})


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
    spec = provenance.compose_full_provenance(
        _file_spec(path),
        provenance.full_data(
            provenance.CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False)
        ),
    )
    assert spec is not None

    graph = _replay_graph.compile_replay_graph(spec)
    code = _replay_graph.emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    assert namespace["derived"].attrs["shift_coords"] is False
    assert _replay_graph.execute_replay_graph(graph).attrs["shift_coords"] is False


def test_replay_graph_execution_matches_emitted_code(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "source.nc"
    source = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [0, 1, 2]},
    )
    source.to_netcdf(path)
    spec = provenance.compose_full_provenance(
        _file_spec(path),
        provenance.full_data(provenance.AverageOperation(dims=("y",))),
    )
    assert spec is not None

    graph = _replay_graph.compile_replay_graph(spec)
    replayed = _replay_graph.execute_replay_graph(graph)
    code = _replay_graph.emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    xr.testing.assert_identical(replayed, namespace["derived"])
