import ast
import pathlib
import re
import types
import typing

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive.imagetool import _replay_graph


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


def _file_replay_source(path: pathlib.Path | str, *, selected_index: int = 0):
    prov = erlab.interactive.imagetool.provenance
    return prov.FileLoadSource(
        path=str(path),
        loader_label="xarray.load_dataarray",
        loader_text="xarray.load_dataarray",
        kwargs_text="",
        replay_call=prov.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            selected_index=selected_index,
        ),
    )


def _file_spec(path: pathlib.Path | str, *, selected_index: int = 0):
    prov = erlab.interactive.imagetool.provenance
    return prov.file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=_file_replay_source(path, selected_index=selected_index),
    )


def _erlab_file_spec(path: pathlib.Path | str, loader: str):
    prov = erlab.interactive.imagetool.provenance
    return prov.file_load(
        start_label=f"Load {path}",
        seed_code=(
            f"erlab.io.set_loader({loader!r})\nderived = erlab.io.load({str(path)!r})"
        ),
        file_load_source=prov.FileLoadSource(
            path=str(path),
            loader_label="Loader",
            loader_text=loader,
            kwargs_text="",
            replay_call=prov.FileReplayCall(
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
    prov = erlab.interactive.imagetool.provenance

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

    with pytest.raises(_replay_graph.ReplayGraphError, match="Expected script"):
        _replay_graph._validate_script_provenance(
            prov.full_data(prov.SqueezeOperation())
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="without active_name"):
        _replay_graph._validate_script_provenance(
            types.SimpleNamespace(kind="script", active_name=None)
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="unsupported Import"):
        _replay_graph._validate_script_provenance(
            prov.script(
                start_label="Run script",
                seed_code="import os",
                active_name="derived",
                script_inputs=(prov.ScriptInput(name="data_0", label="Input"),),
            )
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="no replay code"):
        _replay_graph._validate_script_provenance(
            prov.script(
                prov.AverageOperation(dims=("x",)),
                start_label="Run script",
                active_name="derived",
            )
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="no replay code"):
        _replay_graph._validate_script_provenance(
            prov.script(start_label="Run script", active_name="derived")
        )
    with pytest.raises(_replay_graph.ReplayGraphError, match="non-replayable"):
        _replay_graph._validate_script_provenance(
            prov.script(
                prov.ScriptCodeOperation(
                    label="Opaque",
                    code=None,
                    copyable=False,
                ),
                start_label="Run script",
                active_name="derived",
                script_inputs=(prov.ScriptInput(name="data_0", label="Input"),),
            )
        )
    assert not _replay_graph.script_provenance_replayable(None)


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
    prov = erlab.interactive.imagetool.provenance
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

    loaded_input = prov.ScriptInput(
        name="loaded",
        label="Loaded source",
        provenance_spec=file_spec,
    )
    code = _replay_graph.script_inputs_code((loaded_input,), display=False)
    namespace = _exec_generated_code(code)
    xr.testing.assert_identical(namespace["loaded"], data)
    with pytest.raises(_replay_graph.ReplayGraphError, match="recorded source"):
        _replay_graph.script_inputs_code(
            (prov.ScriptInput(name="missing", label="Missing source"),),
            display=False,
        )

    with pytest.raises(_replay_graph.ReplayGraphError, match="script-derived"):
        _replay_graph.rebuild_script_provenance(file_spec)
    script_spec = prov.script(
        prov.ScriptCodeOperation(label="Add one", code="derived = data_0 + 1.0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
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
    missing_spec = prov.script(
        prov.ScriptCodeOperation(label="Add one", code="derived = data_0 + 1.0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(prov.ScriptInput(name="data_0", label="Closed input"),),
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="not open"):
        _replay_graph.rebuild_script_provenance(missing_spec)

    live_calls = 0
    initial_marker = "initial-marker"
    current_marker = "current-marker"
    live_input = prov.ScriptInput(
        name="data_0",
        label="Live input",
        node_uid="uid-0",
        node_snapshot_token=initial_marker,
    )
    live_spec = prov.script(
        prov.ScriptCodeOperation(label="Double", code="derived = data_0 * 2.0"),
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
    duplicate_file_input = prov.ScriptInput(
        name="data_0",
        label="Closed file input",
        node_uid="same-uid",
        provenance_spec=file_spec,
    )
    duplicate_file_spec = prov.script(
        prov.ScriptCodeOperation(label="Copy", code="derived = data_0"),
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

    unsupported_nested = prov.script(
        prov.ScriptCodeOperation(label="Opaque", code=None, copyable=False),
        start_label="Run script",
        active_name="derived",
        script_inputs=(prov.ScriptInput(name="data_0", label="Input"),),
    )
    unsupported_input = prov.ScriptInput(
        name="data_0",
        label="Unsupported nested input",
        provenance_spec=unsupported_nested,
    )
    unsupported_spec = prov.script(
        prov.ScriptCodeOperation(label="Copy", code="derived = data_0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(unsupported_input,),
    )
    with pytest.raises(_replay_graph.ReplayGraphError, match="cannot be replayed"):
        _replay_graph.rebuild_script_provenance(unsupported_spec)

    full_data_spec = prov.script(
        prov.ScriptCodeOperation(label="Copy", code="derived = data_0"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="Full data",
                provenance_spec=prov.full_data(),
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
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "polarization.nc"
    source = _polarization_source(path)
    file_spec = _file_spec(path)
    shared_stage = prov.full_data(prov.AverageOperation(dims=("k",)))
    left_stage = prov.selection(
        prov.SelOperation(kwargs={"pol": "LH"}),
        prov.SqueezeOperation(),
    )
    right_stage = prov.selection(
        prov.SelOperation(kwargs={"pol": "LV"}),
        prov.SqueezeOperation(),
    )
    left_spec = prov.compose_full_provenance(
        prov.compose_full_provenance(file_spec, shared_stage),
        left_stage,
    )
    right_spec = prov.compose_full_provenance(
        prov.compose_full_provenance(file_spec, shared_stage),
        right_stage,
    )
    assert left_spec is not None
    assert right_spec is not None
    spec = prov.script(
        prov.ScriptCodeOperation(label="Subtract", code="derived = data_0 - data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="LH", provenance_spec=left_spec),
            prov.ScriptInput(name="data_1", label="LV", provenance_spec=right_spec),
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


def test_replay_graph_handles_structured_script_operations(
    tmp_path: pathlib.Path,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("alpha", "eV"),
        coords={"alpha": [0.0, 1.0], "eV": [0.0, 1.0, 2.0]},
    )
    data.to_netcdf(path)
    spec = prov.script(
        prov.AverageOperation(dims=("alpha",)),
        start_label="Run script",
        seed_code="avg = data_0",
        active_name="avg",
        script_inputs=(
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("polarization", "eV"),
        coords={"polarization": [-1, 1], "eV": [0.0, 1.0]},
    )
    data.to_netcdf(path)
    spec = prov.script(
        prov.SelOperation(kwargs={"polarization": -1}),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = prov.script(
        prov.AverageOperation(dims=("x",)),
        prov.ScriptCodeOperation(
            label="Use original input",
            code="derived = derived + data_0.qsel.average('x')",
        ),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(prov.ScriptInput(name="data_0", label="Input"),),
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
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = prov.script(
        prov.AverageOperation(dims=("x",)),
        prov.ScriptCodeOperation(
            label="Use temp",
            code="derived = derived + tmp.qsel.average('x') + data_0.qsel.average('x')",
        ),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(prov.ScriptInput(name="data_0", label="Input"),),
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
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
    )
    spec = prov.script(
        prov.SortCoordOrderOperation(),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(prov.ScriptInput(name="data_0", label="Input"),),
    )

    graph = _replay_graph.compile_replay_graph(spec, external_inputs={"data_0": data})
    script_nodes = [node for node in graph.nodes if node.kind == "script"]
    inlined_code = "\n".join(script_nodes[0].payload["codes"])

    assert prov.script_provenance_replayable(spec)
    assert "data.coords" not in inlined_code
    assert "derived.coords.keys()" in inlined_code
    xr.testing.assert_identical(
        prov.replay_script_provenance(spec, {"data_0": data}),
        erlab.utils.array.sort_coord_order(data + 1, (data + 1).coords.keys()),
    )


def test_replay_graph_rebases_context_in_script_input_code(
    tmp_path: pathlib.Path,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]},
    )
    data.to_netcdf(path)
    spec = prov.script(
        prov.SortCoordOrderOperation(),
        start_label="Run script",
        seed_code="tmp = data_0 + 1\nderived = tmp",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "polarization.nc"
    source = _polarization_source(path)
    averaged = prov.script(
        prov.AverageOperation(dims=("k",)),
        start_label="Run script",
        seed_code="avg = data_0",
        active_name="avg",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="Scan",
                provenance_spec=_file_spec(path),
            ),
        ),
    )
    spec = prov.script(
        prov.ScriptCodeOperation(label="Subtract", code="derived = data_0 - data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="LH", provenance_spec=averaged),
            prov.ScriptInput(name="data_1", label="LV", provenance_spec=averaged),
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
    prov = erlab.interactive.imagetool.provenance
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
    processed = prov.compose_full_provenance(
        _file_spec(path),
        prov.public_data(
            prov.DivideByCoordOperation(coord_name="mesh_current"),
            prov.CoarsenOperation(
                dim={"alpha": 3, "eV": 5},
                boundary="trim",
                side="left",
                coord_func="mean",
                reducer="mean",
            ),
        ),
    )
    assert processed is not None

    rc = prov.script(
        prov.SelOperation(kwargs={"polarization": -1}),
        start_label="Run ImageTool manager console code",
        seed_code="rc = data_0",
        active_name="rc",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="Processed map",
                provenance_spec=processed,
            ),
        ),
    )
    lc = prov.script(
        prov.SelOperation(kwargs={"polarization": 1}),
        start_label="Run ImageTool manager console code",
        seed_code="lc = data_0",
        active_name="lc",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="Processed map",
                provenance_spec=processed,
            ),
        ),
    )
    diff = prov.script(
        prov.ScriptCodeOperation(
            label="Evaluate console expression",
            code="derived = rc - lc",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
                name="rc", label="console variable 'rc'", provenance_spec=rc
            ),
            prov.ScriptInput(
                name="lc", label="console variable 'lc'", provenance_spec=lc
            ),
        ),
    )
    total = prov.script(
        prov.ScriptCodeOperation(
            label="Evaluate console expression",
            code="derived = rc + lc",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
                name="rc", label="console variable 'rc'", provenance_spec=rc
            ),
            prov.ScriptInput(
                name="lc", label="console variable 'lc'", provenance_spec=lc
            ),
        ),
    )
    ncd = prov.script(
        prov.ScriptCodeOperation(
            label="Evaluate console expression",
            code="ncd = data_1 / data_2",
        ),
        start_label="Run ImageTool manager console code",
        active_name="ncd",
        script_inputs=(
            prov.ScriptInput(name="data_1", label="ImageTool 1", provenance_spec=diff),
            prov.ScriptInput(name="data_2", label="ImageTool 2", provenance_spec=total),
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
    prov = erlab.interactive.imagetool.provenance

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

    original_replace = prov._replace_code_identifiers

    def _raise_on_input_replacement(
        code: str, replacements: typing.Mapping[str, str]
    ) -> str:
        if "data_0" in replacements:
            raise SyntaxError("bad script input")
        return original_replace(code, replacements)

    monkeypatch.setattr(prov, "_replace_code_identifiers", _raise_on_input_replacement)
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
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "scan.nc"
    source = _polarization_source(path)
    left = prov.compose_full_provenance(
        _file_spec(path),
        prov.full_data(prov.SelOperation(kwargs={"pol": "LH"})),
    )
    right = prov.compose_full_provenance(
        _file_spec(path),
        prov.full_data(prov.SelOperation(kwargs={"pol": "LV"})),
    )
    assert left is not None
    assert right is not None
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Concatenate selected inputs",
            code="combined = xr.concat([left, right], dim='pol')",
        ),
        start_label="Run ImageTool manager UI action",
        active_name="combined",
        script_inputs=(
            prov.ScriptInput(name="left", label="Selected left", provenance_spec=left),
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    rc_data = xr.DataArray(np.arange(3.0), dims=("x",))
    lc_data = xr.DataArray(np.arange(3.0) + 10.0, dims=("x",))
    rc = prov.script(
        start_label="Start from watched variable 'rc'",
        seed_code="derived = rc",
        active_name="derived",
    )
    lc = prov.script(
        start_label="Start from watched variable 'lc'",
        seed_code="derived = lc",
        active_name="derived",
    )
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Subtract selected inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager action",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="ImageTool 0", provenance_spec=rc),
            prov.ScriptInput(name="data_1", label="ImageTool 1", provenance_spec=lc),
        ),
    )

    code = typing.cast("str", spec.display_code())
    namespace = _exec_generated_code(code, {"rc": rc_data, "lc": lc_data})

    assert code == "derived = rc - lc"
    xr.testing.assert_identical(namespace["derived"], rc_data - lc_data)


def test_replay_graph_display_keeps_helpers_from_raw_seed_inputs() -> None:
    prov = erlab.interactive.imagetool.provenance
    raw_data = xr.DataArray(np.arange(1.0, 4.0), dims=("x",))
    root = prov.script(
        start_label="Start from watched variable 'raw'",
        seed_code="def normalize():\n    return raw / raw.max()\nderived = normalize()",
        active_name="derived",
    )
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Average normalized input",
            code="result = data_0.mean()",
        ),
        start_label="Run ImageTool manager action",
        active_name="result",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="ImageTool 0", provenance_spec=root),
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
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "nonuniform.nc"
    source = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 1.0], "y": [0.0, 1.0]},
    )
    source.to_netcdf(path)
    public_spec = prov.compose_full_provenance(
        _file_spec(path),
        prov.public_data(prov.SelOperation(kwargs={"x": 0.2})),
    )
    assert public_spec is not None
    script_input = prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "scan.nc"
    source = xr.DataArray(np.arange(3.0), dims=("x",))
    source.to_netcdf(path)
    script_input = prov.ScriptInput(
        name="data_0",
        label="ImageTool 0",
        provenance_spec=_file_spec(path),
    )
    helper_spec = prov.script(
        prov.ScriptCodeOperation(
            label="Use helper",
            code="def offset():\n    return data_0 + 1\nresult = offset()",
        ),
        start_label="Run script",
        active_name="result",
        script_inputs=(script_input,),
    )
    rebound_spec = prov.script(
        prov.ScriptCodeOperation(
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
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(np.arange(3.0), dims=("x",))
    spec = prov.script(
        prov.RenameOperation(name="renamed"),
        start_label="Run script",
        seed_code=seed_code,
        active_name=active_name,
        script_inputs=(prov.ScriptInput(name="data_0", label="Input"),),
    )

    graph = _replay_graph.compile_replay_graph(spec, external_inputs={"data_0": data})
    replayed = _replay_graph.execute_replay_graph(graph)

    assert any(node.kind == "relay" for node in graph.nodes)
    assert replayed.name == "renamed"
    assert not np.shares_memory(replayed.data, data.data)


def test_replay_graph_display_skips_whole_array_rename(
    tmp_path: pathlib.Path,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="source")
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    spec = prov.script(
        prov.RenameOperation(name="renamed"),
        start_label="Run script",
        active_name="data_0",
        script_inputs=(
            prov.ScriptInput(
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

    assert ".rename(" not in code
    xr.testing.assert_identical(_exec_generated_code(code)["derived"], data)


def test_replay_graph_display_keeps_name_rename_before_script_code(
    tmp_path: pathlib.Path,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(np.arange(3.0), dims=("x",), name="source")
    path = tmp_path / "source.nc"
    data.to_netcdf(path)
    spec = prov.script(
        prov.RenameOperation(name="renamed"),
        prov.ScriptCodeOperation(
            label="Use DataArray name",
            code="derived = derived.rename(derived.name + '_used')",
        ),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    spec = prov.script(
        prov.RenameOperation(name="renamed"),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(prov.ScriptInput(name="data_0", label="Input"),),
    )

    def fail_derivation_entry(self):
        raise AssertionError("replayability checks must not generate copied code")

    monkeypatch.setattr(prov.RenameOperation, "derivation_entry", fail_derivation_entry)

    assert prov.script_provenance_replayable(spec)


def test_replay_graph_uses_existing_console_alias_for_script_code(
    tmp_path: pathlib.Path,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    data.to_netcdf(path)
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Rotate",
            code=(
                "derived = era.transform.rotate("
                "data_0, 0.0, axes=('x', 'y'), reshape=False)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "scan.nc"
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    data.to_netcdf(path)
    spec = prov.script(
        prov.RotateOperation(
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
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    first = _file_spec("scan.h5", selected_index=0)
    second = _file_spec("scan.h5", selected_index=1)
    spec = prov.script(
        prov.ScriptCodeOperation(label="Subtract", code="derived = data_0 - data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="First", provenance_spec=first),
            prov.ScriptInput(name="data_1", label="Second", provenance_spec=second),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count("xr.load_dataarray") == 2


def test_replay_graph_reuses_shared_loader_setup() -> None:
    prov = erlab.interactive.imagetool.provenance
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Add",
            code="derived = data_0 + data_1",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="First",
                provenance_spec=_erlab_file_spec("scan0.h5", "example"),
            ),
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Add",
            code="derived = data_0 + data_1 + data_2",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="Alpha 0",
                provenance_spec=_erlab_file_spec("alpha0.h5", "alpha"),
            ),
            prov.ScriptInput(
                name="data_1",
                label="Beta",
                provenance_spec=_erlab_file_spec("beta.h5", "beta"),
            ),
            prov.ScriptInput(
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
    prov = erlab.interactive.imagetool.provenance
    file_spec = _file_spec("scan.h5")
    first_spec = prov.compose_full_provenance(
        file_spec,
        prov.full_data(prov.IselOperation(kwargs={"pol": 0})),
    )
    second_spec = prov.compose_full_provenance(
        file_spec,
        prov.selection(prov.IselOperation(kwargs={"pol": 0})),
    )
    assert first_spec is not None
    assert second_spec is not None
    spec = prov.script(
        prov.ScriptCodeOperation(label="Subtract", code="derived = data_0 - data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="Full", provenance_spec=first_spec),
            prov.ScriptInput(
                name="data_1",
                label="Selection",
                provenance_spec=second_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count(".isel") == 2


def test_replay_graph_script_nodes_are_not_deduplicated() -> None:
    prov = erlab.interactive.imagetool.provenance
    first = prov.script(
        start_label="Make first",
        seed_code="derived = xr.DataArray([1.0, 2.0], dims=['x'])",
        active_name="derived",
    )
    second = prov.script(
        start_label="Make second",
        seed_code="derived = xr.DataArray([10.0, 20.0], dims=['x'])",
        active_name="derived",
    )
    spec = prov.script(
        prov.ScriptCodeOperation(label="Add", code="derived = data_0 + data_1"),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="First", provenance_spec=first),
            prov.ScriptInput(name="data_1", label="Second", provenance_spec=second),
        ),
    )

    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code)

    assert code.count("xr.DataArray") == 2
    xr.testing.assert_identical(
        namespace["derived"],
        xr.DataArray([11.0, 22.0], dims=["x"]),
    )


def test_replay_graph_raises_typed_errors_for_unsupported_script() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray([1.0], dims=("x",))
    spec = prov.script(
        prov.ScriptCodeOperation(label="Unsupported", code="import os\nderived = data"),
        start_label="Run script",
        active_name="derived",
    )

    with pytest.raises(_replay_graph.ReplayGraphError, match="unsupported Import"):
        _replay_graph.compile_replay_graph(spec, external_inputs={"data": data})


def test_replay_graph_emits_correct_with_edge_operation_code(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "scan.nc"
    data = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0.0, 1.0]})
    data.to_netcdf(path)
    edge_fit = xr.Dataset({"center": ("x", [0.0, 1.0])})

    def correct_with_edge(data_arg, edge_fit_arg, *, shift_coords=True):
        xr.testing.assert_identical(edge_fit_arg, edge_fit)
        return data_arg.assign_attrs(shift_coords=shift_coords)

    monkeypatch.setattr(erlab.analysis.gold, "correct_with_edge", correct_with_edge)
    spec = prov.compose_full_provenance(
        _file_spec(path),
        prov.full_data(
            prov.CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False)
        ),
    )
    assert spec is not None

    graph = _replay_graph.compile_replay_graph(spec)
    code = _replay_graph.emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    assert namespace["derived"].attrs["shift_coords"] is False
    assert _replay_graph.execute_replay_graph(graph).attrs["shift_coords"] is False


def test_replay_graph_execution_matches_emitted_code(tmp_path: pathlib.Path) -> None:
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "source.nc"
    source = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [0, 1, 2]},
    )
    source.to_netcdf(path)
    spec = prov.compose_full_provenance(
        _file_spec(path),
        prov.full_data(prov.AverageOperation(dims=("y",))),
    )
    assert spec is not None

    graph = _replay_graph.compile_replay_graph(spec)
    replayed = _replay_graph.execute_replay_graph(graph)
    code = _replay_graph.emit_replay_code(graph, output_name="derived")
    namespace = _exec_generated_code(code)

    xr.testing.assert_identical(replayed, namespace["derived"])
