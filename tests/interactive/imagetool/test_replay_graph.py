import pathlib
import typing

import numpy as np
import pytest
import xarray as xr

import erlab
from erlab.interactive.imagetool import _replay_graph


def _exec_generated_code(code: str) -> dict[str, typing.Any]:
    namespace: dict[str, typing.Any] = {
        "erlab": erlab,
        "np": np,
        "numpy": np,
        "xr": xr,
        "xarray": xr,
    }
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
            "import erlab\n\n"
            f"erlab.io.set_loader({loader!r})\n"
            f"derived = erlab.io.load({str(path)!r})"
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
    assert code.count(".qsel.average") == 1
    namespace = _exec_generated_code(code)
    expected = left_stage.apply(shared_stage.apply(source)) - right_stage.apply(
        shared_stage.apply(source)
    )
    xr.testing.assert_identical(namespace["derived"], expected)


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


def test_replay_graph_rejects_operations_without_replay_code() -> None:
    prov = erlab.interactive.imagetool.provenance
    spec = prov.compose_full_provenance(
        _file_spec("scan.h5"),
        prov.full_data(prov.RenameOperation(name="renamed")),
    )
    assert spec is not None

    graph = _replay_graph.compile_replay_graph(spec)

    with pytest.raises(_replay_graph.ReplayGraphError, match="replay code"):
        _replay_graph.emit_replay_code(graph)


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
