import ast
import json
import pathlib
import subprocess
import sys
import types
import typing

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

import erlab
from erlab.interactive.imagetool import provenance


def _exec_generated_code(
    code: str, namespace: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    locals_ns = dict(namespace)
    exec(  # noqa: S102
        code,
        {
            "np": np,
            "xr": xr,
            "erlab": erlab,
            "era": erlab.analysis,
        },
        locals_ns,
    )
    return locals_ns


def _generated_call_names(code: str) -> tuple[str, ...]:
    def _call_name(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
            if isinstance(node, ast.Call):
                return ".".join(reversed(parts))
        if not isinstance(node, ast.Name):
            return None
        parts.append(node.id)
        return ".".join(reversed(parts))

    return tuple(
        name
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        for name in [_call_name(node.func)]
        if name is not None
    )


def _base_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(24).reshape((3, 4, 2)),
        dims=("x", "y", "z"),
        coords={
            "x": [0.0, 1.0, 2.0],
            "y": [10.0, 11.0, 12.0, 13.0],
            "z": [5.0, 6.0],
            "x_alt": ("x", ["a", "b", "c"]),
        },
        name="data",
    )


def _kspace_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(27.0).reshape(3, 3, 3),
        dims=("alpha", "beta", "eV"),
        coords={
            "alpha": [-1.0, 0.0, 1.0],
            "beta": [-1.0, 0.0, 1.0],
            "eV": [-0.2, 0.0, 0.2],
            "xi": 0.0,
            "hv": 21.2,
        },
        attrs={
            "configuration": int(erlab.constants.AxesConfiguration.Type1),
            "inner_potential": 10.0,
            "sample_workfunction": 4.5,
        },
        name="anglemap",
    )


def _hashable_data() -> xr.DataArray:
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)),
        dims=(1, ("beta", 0)),
        name="data",
    )
    return data.assign_coords(
        {"coord_1": xr.DataArray([100.0, 101.0, 102.0], dims=[1])}
    )


def _string_key_data() -> xr.DataArray:
    return xr.DataArray(
        np.arange(12).reshape((3, 4)),
        dims=("k-space", ("beta", 0)),
        coords={"k-space": [0.0, 1.0, 2.0]},
        name="data",
    )


def test_provenance_import_keeps_analysis_targets_lazy() -> None:
    code = (
        "import sys\n"
        "from erlab.interactive.imagetool import provenance\n"
        "loaded = sorted("
        "name for name in sys.modules if name.startswith('erlab.analysis.') "
        "or name.startswith('scipy.interpolate') or name.startswith('scipy.linalg')"
        ")\n"
        "if loaded:\n"
        "    print('\\n'.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_public_provenance_module_registers_operations_in_fresh_process() -> None:
    code = (
        "import importlib.util\n"
        "from erlab.interactive.imagetool import provenance\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.provenance_operations') is None\n"
        "assert importlib.util.find_spec("
        "'erlab.interactive.imagetool.provenance_framework') is None\n"
        "spec = provenance.full_data(\n"
        "    provenance.AverageOperation(dims=('x',))\n"
        ").append_final_rename('avg')\n"
        "payload = spec.model_dump(mode='json')\n"
        "parsed = provenance.parse_tool_provenance_spec(payload)\n"
        "assert isinstance(parsed, provenance.ToolProvenanceSpec)\n"
        "assert isinstance(parsed.operations[0], provenance.AverageOperation)\n"
        "assert parsed.operations[-1] == provenance.RenameOperation(name='avg')\n"
        "assert isinstance(\n"
        "    provenance.parse_tool_provenance_operation("
        "{'op': 'rename', 'name': 'renamed'}),\n"
        "    provenance.RenameOperation,\n"
        ")\n"
        "assert provenance.selection(provenance.IselOperation(kwargs={'x': 0})).kind "
        "== 'selection'\n"
        "assert provenance.script(\n"
        "    provenance.ScriptCodeOperation(label='Step', code='derived = data'),\n"
        "    start_label='Run script',\n"
        "    active_name='derived',\n"
        ").kind == 'script'\n"
        "source = provenance.FileLoadSource(\n"
        "    path='scan.h5',\n"
        "    loader_label='xarray.load_dataarray',\n"
        "    loader_text='xarray.load_dataarray',\n"
        "    kwargs_text='',\n"
        "    replay_call=provenance.FileReplayCall(\n"
        "        kind='callable',\n"
        "        target='xarray.load_dataarray',\n"
        "        selected_index=0,\n"
        "    ),\n"
        ")\n"
        "file_spec = provenance.file_load(\n"
        "    start_label='Load data',\n"
        "    seed_code='derived = data',\n"
        "    file_load_source=source,\n"
        ")\n"
        "assert file_spec.kind == 'file'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_script_replay_keeps_unused_aliases_lazy() -> None:
    code = (
        "import sys\n"
        "import numpy as np\n"
        "import xarray as xr\n"
        "from erlab.interactive.imagetool import provenance\n"
        "data = xr.DataArray(np.arange(2.0), dims=('x',))\n"
        "spec = provenance.script(\n"
        "    provenance.ScriptCodeOperation(label='Subtract', "
        "code='derived = data_0 - data_1'),\n"
        "    start_label='Run script',\n"
        "    active_name='derived',\n"
        "    script_inputs=(\n"
        "        provenance.ScriptInput(name='data_0', label='A'),\n"
        "        provenance.ScriptInput(name='data_1', label='B'),\n"
        "    ),\n"
        ")\n"
        "provenance.replay_script_provenance(spec, {'data_0': data, 'data_1': data})\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('erlab.analysis.')\n"
        "    or name.startswith('erlab.plotting')\n"
        "    or name.startswith('matplotlib')\n"
        ")\n"
        "if loaded:\n"
        "    print('\\n'.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def _file_replay_source(
    path: typing.Any = "scan.h5",
    *,
    replay_call: typing.Any = None,
) -> typing.Any:
    if replay_call is None:
        replay_call = provenance.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={},
            selected_index=0,
        )
    return provenance.FileLoadSource(
        path=path,
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="(none)",
        replay_call=replay_call,
        load_code=None,
    )


def _file_provenance_spec(path: typing.Any = "scan.h5") -> typing.Any:
    return provenance.file_load(
        start_label="Load data from file 'scan.h5'",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(path),
    )


def _representative_structured_operations() -> tuple[
    provenance.ToolProvenanceOperation, ...
]:
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})
    vertices = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    return (
        provenance.QSelOperation(kwargs={"x": 1.0}),
        provenance.IselOperation(kwargs={"x": slice(0, 2)}),
        provenance.SelOperation(kwargs={"y": slice(10.0, 12.0)}),
        provenance.SortCoordOrderOperation(),
        provenance.SortByOperation(variables=("x",), ascending=False),
        provenance.SelectCoordOperation(coord_name="x"),
        provenance.TransposeOperation(dims=("y", "x", "z")),
        provenance.SqueezeOperation(),
        provenance.RenameOperation(name="renamed"),
        provenance.RestoreNonuniformDimsOperation(),
        provenance.RotateOperation(angle=0.0, axes=("x", "y"), center=(0.0, 10.0)),
        provenance.AverageOperation(dims=("x",)),
        provenance.QSelAggregationOperation(dims=("x",), func="sum"),
        provenance.InterpolationOperation(dim="x", values=[0.25, 0.75]),
        provenance.LeadingEdgeOperation(fraction=0.5, dim="x"),
        provenance.DivideByCoordOperation(coord_name="x"),
        provenance.GaussianFilterOperation(sigma={"x": 0.5}),
        provenance.NormalizeOperation(dims=("x",), mode="minmax"),
        provenance.CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        ),
        provenance.ThinOperation(mode="per_dim", factors={"x": 2}),
        provenance.SymmetrizeOperation(dim="x", center=1.0),
        provenance.SymmetrizeNfoldOperation(
            fold=4,
            axes=("x", "y"),
            center={"x": 1.0, "y": 10.0},
        ),
        provenance.CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False),
        provenance.SwapDimsOperation(mapping={"x": "x_alt"}),
        provenance.RenameDimsCoordsOperation(mapping={"x": "energy"}),
        provenance.AffineCoordOperation(coord_name="x", scale=2.0, offset=1.0),
        provenance.AssignCoordsOperation(coord_name="x", values=[0.0, 1.0, 2.0]),
        provenance.AssignScalarCoordOperation(coord_name="temperature", value=20.0),
        provenance.AssignCoord1DOperation(
            coord_name="temperature",
            dim="x",
            values=[1.0, 2.0, 3.0],
        ),
        provenance.AssignAttrsOperation(attrs={"sample": "test"}),
        provenance.KspaceConfigurationOperation(configuration=2),
        provenance.KspaceWorkFunctionOperation(work_function=4.2),
        provenance.KspaceInnerPotentialOperation(inner_potential=12.0),
        provenance.KspaceSetNormalOperation(alpha=1.5, beta=-0.5, delta=2.0),
        provenance.KspaceConvertOperation(
            bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
            resolution={"kx": 0.02, "ky": 0.02},
        ),
        provenance.SliceAlongPathOperation(
            vertices={"x": [0.0, 1.0], "y": [10.0, 11.0]},
            step_size=0.1,
            dim_name="path",
        ),
        provenance.MaskWithPolygonOperation(vertices=vertices, dims=("x", "y")),
    )


def test_operation_group_markers_round_trip_and_strip_partial_groups() -> None:
    operations = provenance.stamp_operation_group(
        (
            provenance.AverageOperation(dims=("x",)),
            provenance.SqueezeOperation(),
        ),
        kind="demo",
        group_id="group-1",
        focuses=("first", "second"),
    )

    assert provenance.operation_group_range(operations, 0, kind="demo") == (0, 2)
    assert provenance.operation_group_range(operations, 1, kind="demo") == (0, 2)
    assert operations[0].group is not None
    assert operations[0].group.focus == "first"
    assert "group" not in provenance.AverageOperation(dims=("x",)).model_dump(
        mode="json"
    )

    parsed = tuple(
        provenance.parse_tool_provenance_operation(operation.model_dump(mode="json"))
        for operation in operations
    )
    assert parsed == operations

    assert provenance.strip_partial_operation_groups(operations) == operations
    partial = provenance.strip_partial_operation_groups(operations[:1])
    assert partial[0].group is None

    scrambled = provenance.strip_partial_operation_groups(
        (operations[1], operations[0])
    )
    assert all(operation.group is None for operation in scrambled)

    restamped = provenance.restamp_operation_groups(operations)
    assert provenance.strip_operation_groups(
        restamped
    ) == provenance.strip_operation_groups(operations)
    assert provenance.operation_group_range(restamped, 0, kind="demo") == (0, 2)
    assert restamped[0].group is not None
    assert operations[0].group is not None
    assert restamped[0].group.id != operations[0].group.id

    adjacent = provenance.restamp_operation_groups(operations + operations)
    assert provenance.operation_group_range(adjacent, 0, kind="demo") == (0, 2)
    assert provenance.operation_group_range(adjacent, 2, kind="demo") == (2, 4)
    assert adjacent[0].group is not None
    assert adjacent[2].group is not None
    assert adjacent[0].group.id != adjacent[2].group.id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "", "id": "group", "index": 0, "size": 1},
        {"kind": "demo", "id": "", "index": 0, "size": 1},
        {"kind": "demo", "id": "group", "index": -1, "size": 1},
        {"kind": "demo", "id": "group", "index": 0, "size": 0},
        {"kind": "demo", "id": "group", "index": 1, "size": 1},
    ],
)
def test_operation_group_marker_rejects_invalid_metadata(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        provenance.OperationGroupMarker(**kwargs)


def test_operation_group_helpers_reject_broken_ranges() -> None:
    operations = provenance.stamp_operation_group(
        (
            provenance.AverageOperation(dims=("x",)),
            provenance.SqueezeOperation(),
        ),
        kind="demo",
        group_id="group-1",
    )
    plain = provenance.AverageOperation(dims=("y",))

    assert provenance.stamp_operation_group((), kind="demo") == ()
    with pytest.raises(ValueError, match="focuses must match"):
        provenance.stamp_operation_group(
            operations,
            kind="demo",
            focuses=("first",),
        )
    assert provenance.strip_operation_groups((plain,)) == (plain,)

    assert provenance.operation_group_range(operations, -1) is None
    assert provenance.operation_group_range(operations, len(operations)) is None
    assert provenance.operation_group_range((plain,), 0) is None
    assert provenance.operation_group_range(operations, 0, kind="other") is None
    assert provenance.operation_group_range((operations[1],), 0) is None

    neighbor = plain.model_copy(update={"group": operations[0].group})
    assert provenance.operation_group_range((*operations, neighbor), 0) is None

    restamped = provenance.restamp_operation_groups((operations[1],))
    assert restamped[0].group is None


def test_tool_provenance_codec_and_combinators() -> None:
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})
    encoded = provenance.encode_provenance_value(
        {"sel": slice(1.0, 2.0), "data": _base_data(), "edge_fit": edge_fit}
    )
    decoded = provenance.decode_provenance_value(encoded)

    assert decoded["sel"] == slice(1.0, 2.0)
    xr.testing.assert_identical(decoded["data"], _base_data())
    xr.testing.assert_identical(decoded["edge_fit"], edge_fit)

    hashable_encoded = provenance.encode_provenance_value(
        {1: slice(0.0, 1.0), ("beta", 0): {"nested": [1, 2, 3]}}
    )
    assert provenance._MAPPING_MARKER in hashable_encoded
    mapping_entries = hashable_encoded[provenance._MAPPING_MARKER]
    assert mapping_entries[0][0] == 1
    assert mapping_entries[1][0] == {provenance._TUPLE_MARKER: ["beta", 0]}
    assert provenance.decode_provenance_value(hashable_encoded) == {
        1: slice(0.0, 1.0),
        ("beta", 0): {"nested": [1, 2, 3]},
    }

    spec = provenance.full_data(
        provenance.AverageOperation(dims=("y",))
    ).append_final_rename("avg")
    trimmed = spec.drop_trailing_rename()
    replaced = spec.append_replacement_operations(
        provenance.ThinOperation(mode="global", factor=2)
    )

    assert [op.op for op in spec.operations] == ["average", "rename"]
    assert [op.op for op in trimmed.operations] == ["average"]
    assert [op.op for op in replaced.operations] == ["average", "thin"]

    with pytest.raises(ValidationError, match="Instance is frozen"):
        spec.kind = "selection"
    with pytest.raises(TypeError, match="ToolProvenanceOperation instances only"):
        erlab.interactive.imagetool.provenance.full_data(
            {"op": "average", "dims": ["y"]}
        )
    with pytest.raises(TypeError, match="ToolProvenanceOperation instances only"):
        spec.append_replacement_operations(
            {"op": "thin", "mode": "global", "factor": 2}
        )


def test_tool_provenance_parse_final_payload_and_migrate_legacy_schema() -> None:
    payload = {
        "schema_version": 1,
        "kind": "full_data",
        "operations": [
            {"op": "average", "dims": {provenance._TUPLE_MARKER: ["x"]}},
            {"op": "rename", "name": "avg"},
        ],
    }

    spec = provenance.parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 2
    assert [op.op for op in spec.operations] == ["average", "rename"]
    assert [entry.label for entry in spec.derivation_entries()] == [
        "Start from current parent ImageTool data",
        'Average(dims=("x",))',
        "rename('avg')",
    ]
    assert [entry.label for entry in spec.display_entries()] == [
        "Start from current parent ImageTool data",
        'Average(dims=("x",))',
    ]
    assert spec.derivation_code() == 'derived = data\nderived = derived.qsel.mean("x")'
    display_code = typing.cast("str", spec.display_code())
    assert ".rename(" not in display_code
    namespace = _exec_generated_code(display_code, {"data": _base_data()})
    xr.testing.assert_identical(
        namespace["derived"].rename(None),
        _base_data().qsel.mean("x").rename(None),
    )

    dumped = spec.model_dump(mode="json")
    assert dumped["schema_version"] == 2
    assert "active_name" in dumped
    assert dumped["active_name"] is None
    assert dumped["operations"][0]["op"] == "average"
    assert dumped["operations"][0]["dims"] == {provenance._TUPLE_MARKER: ["x"]}
    assert spec.to_replay_spec().active_name == "derived"

    with pytest.raises(ValidationError, match="Unknown provenance operation"):
        provenance.parse_tool_provenance_spec(
            {
                "kind": "full_data",
                "operations": [
                    {
                        "op": "transform",
                        "name": "average",
                        "kwargs": {"dims": ["x"]},
                    }
                ],
            }
        )

    with pytest.raises(
        ValidationError, match="script provenance specs must define `active_name`"
    ):
        provenance.parse_tool_provenance_spec(
            {
                "kind": "script",
                "start_label": "Start from current parent ImageTool data",
                "seed_code": "derived = data",
                "operations": [],
            }
        )

    with pytest.raises(
        TypeError, match="Serialized provenance operations must be a sequence"
    ):
        provenance.parse_tool_provenance_spec({"kind": "full_data", "operations": 1})

    with pytest.raises(
        TypeError, match="Serialized provenance operations must be a sequence"
    ):
        provenance.parse_tool_provenance_spec(
            {"kind": "full_data", "operations": {"op": "average", "dims": ["x"]}}
        )


def test_registered_provenance_define_operation_code_api() -> None:

    structured_operation_types = [
        operation_type
        for op, operation_type in provenance._OPERATION_TYPES.items()
        if op != "script_code"
    ]
    assert [
        operation_type
        for operation_type in structured_operation_types
        if "derivation_label" not in operation_type.__dict__
    ] == []
    assert [
        operation_type
        for operation_type in structured_operation_types
        if (
            operation_type.expression_code
            is provenance.ToolProvenanceOperation.expression_code
            and operation_type.statement_code
            is provenance.ToolProvenanceOperation.statement_code
        )
    ] == []


@pytest.mark.parametrize(
    "operation",
    [
        provenance.IselOperation(kwargs={"x": slice(0, 2)}),
        provenance.SelOperation(kwargs={"y": 11.0}),
        provenance.DivideByCoordOperation(coord_name="scale"),
        provenance.GaussianFilterOperation(sigma={"x": 0.5}),
        provenance.NormalizeOperation(
            dims=("x",),
            mode="minmax",
        ),
        provenance.CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        ),
        provenance.ThinOperation(
            mode="per_dim",
            factors={"y": 2},
        ),
    ],
)
def test_operation_replay_code_uses_requested_names(
    operation: erlab.interactive.imagetool.provenance.ToolProvenanceOperation,
) -> None:
    data = _base_data().drop_vars("x_alt").assign_coords(scale=("x", [1.0, 2.0, 3.0]))

    code = operation.replay_code("data", output_name="result", source_name="source")
    assert "derived" not in code

    namespace = _exec_generated_code(
        code,
        {
            "data": data.copy(deep=True),
            "source": data.copy(deep=True),
        },
    )
    result = namespace["result"]
    expected = operation.apply(data, parent_data=data)
    if isinstance(
        operation,
        provenance.DivideByCoordOperation,
    ):
        result = result.rename(None)
        expected = expected.rename(None)
        assert ".rename(" not in code
    xr.testing.assert_identical(result, expected)


def test_operation_replay_code_passes_source_context() -> None:
    parent = _base_data()
    child = parent.transpose("z", "x", "y")
    operation = provenance.SortCoordOrderOperation()

    code = operation.replay_code("child", output_name="result", source_name="parent")
    assert "parent.coords.keys()" in code
    assert "derived" not in code

    namespace = _exec_generated_code(
        code,
        {
            "child": child.copy(deep=True),
            "parent": parent.copy(deep=True),
        },
    )
    xr.testing.assert_identical(
        namespace["result"],
        operation.apply(child, parent_data=parent),
    )


def test_operation_code_base_edges() -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",))

    with pytest.raises(NotImplementedError):
        provenance.ToolProvenanceOperation().expression_code("data")
    with pytest.raises(NotImplementedError):
        provenance.ToolProvenanceOperation().statement_code(
            "data",
            output_name="derived",
        )
    with pytest.raises(NotImplementedError):
        provenance.KspaceWorkFunctionOperation(work_function=4.2).replay_code(
            "data",
            output_name=None,
        )

    assert (
        provenance.IselOperation(kwargs={"x": 0}).replay_code("data", output_name=None)
        == "data.isel(x=0)"
    )
    assert provenance._expression_receiver_code("data +") == "(data +)"
    assert (
        provenance._simplify_display_code(
            "derived = data\nfor item in []:\n    pass",
            inline_targets={"derived"},
        )
        == "for item in []:\n    pass"
    )
    xr.testing.assert_identical(
        provenance.NormalizeOperation(dims=()).apply(data, parent_data=data),
        data,
    )


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (
            provenance.NormalizeOperation(dims=("x",), mode="area"),
            'data / data.mean("x")',
        ),
        (
            provenance.NormalizeOperation(dims=("x",), mode="min"),
            'data - data.min("x")',
        ),
        (
            provenance.NormalizeOperation(dims=("x",), mode="min_area"),
            '(data - data.min("x")) / data.mean("x")',
        ),
    ],
)
def test_normalize_operation_expression_modes(
    operation: provenance.NormalizeOperation,
    expected: str,
) -> None:
    assert operation.expression_code("data") == expected


def test_roi_operation_derivation_labels() -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 1.0]])

    path_operation = provenance.SliceAlongPathOperation(
        vertices={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        step_size=0.1,
        dim_name="s",
    )
    mask_operation = provenance.MaskWithPolygonOperation(
        vertices=vertices,
        dims=("x", "y"),
    )

    assert path_operation.derivation_label().startswith("Slice Along ROI Path(")
    assert mask_operation.derivation_label().startswith("Mask with ROI(")


def test_operations_expression_code_chains_without_relay_assignments() -> None:
    data = _base_data().assign_coords(scale=("x", [1.0, 2.0, 3.0]))
    operations = (
        provenance.DivideByCoordOperation(coord_name="scale"),
        provenance.IselOperation(kwargs={"x": slice(0, 2)}),
    )

    code = provenance.operations_expression_code(operations, "data")
    assert code.startswith("(data / data.scale).isel(")
    assert ".rename(" not in code
    assert "derived" not in code

    namespace = _exec_generated_code(
        f"result = {code}",
        {"data": data.copy(deep=True)},
    )
    expected = operations[1].apply(
        operations[0].apply(data, parent_data=data),
        parent_data=data,
    )
    xr.testing.assert_identical(namespace["result"].rename(None), expected.rename(None))


def test_statement_operation_replay_code_mutates_working_copy() -> None:
    data = _kspace_data()
    operation = provenance.KspaceWorkFunctionOperation(work_function=4.2)

    code = operation.replay_code("data", output_name="result", source_name="data")

    assert "result = data.copy(deep=False)" in code
    assert "result.kspace.work_function = 4.2" in code
    assert "sample_workfunction" not in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    assert namespace["result"].kspace.work_function == pytest.approx(4.2)
    assert namespace["data"].kspace.work_function == pytest.approx(4.5)


def test_statement_operation_derivation_entry_omits_same_name_noop() -> None:
    operation = provenance.KspaceSetNormalOperation(alpha=1.5, beta=-0.5, delta=2.0)

    code = operation.derivation_entry().code

    assert code == "derived.kspace.set_normal(alpha=1.5, beta=-0.5, delta=2.0)"


def test_tool_provenance_mixed_statement_and_expression_display_code() -> None:
    data = _kspace_data()
    operations = (
        provenance.KspaceWorkFunctionOperation(work_function=4.2),
        provenance.KspaceSetNormalOperation(alpha=1.5, beta=-0.5, delta=2.0),
        provenance.KspaceConvertOperation(
            bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
            resolution={"kx": 0.02, "ky": 0.02},
        ),
    )
    spec = provenance.full_data(*operations).to_replay_spec()

    code = typing.cast("str", spec.display_code())

    assert "derived = data.copy(deep=False)" in code
    assert code.count(".copy(deep=False)") == 1
    assert "derived.kspace.work_function = 4.2" in code
    assert "derived.kspace.set_normal(" in code
    assert "derived = derived.kspace.convert(" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    expected = data.copy(deep=False)
    expected.kspace.work_function = 4.2
    expected.kspace.set_normal(alpha=1.5, beta=-0.5, delta=2.0)
    expected = expected.kspace.convert(
        bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
        resolution={"kx": 0.02, "ky": 0.02},
        silent=True,
    )
    xr.testing.assert_allclose(namespace["derived"], expected)
    assert namespace["data"].kspace.work_function == pytest.approx(4.5)


def test_tool_provenance_parse_legacy_file_script_metadata() -> None:
    payload = {
        "schema_version": 1,
        "kind": "script",
        "start_label": "Load data from file 'scan.h5'",
        "seed_code": "import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        "active_name": "derived",
        "file_load_source": {
            "path": "scan.h5",
            "loader_label": "Load Function",
            "loader_text": "xarray.load_dataarray",
            "kwargs_text": "(none)",
            "load_code": "import xarray\n\ndata = xarray.load_dataarray('scan.h5')",
        },
        "operations": [
            {
                "op": "script_code",
                "label": 'Average(dims=("x",))',
                "code": 'derived = derived.qsel.average("x")',
                "copyable": True,
            }
        ],
    }

    spec = provenance.parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 2
    assert spec.kind == "script"
    assert spec.file_load_source is not None
    assert spec.file_load_source.path == "scan.h5"
    assert spec.file_load_source.replay_call is None
    assert [operation.op for operation in spec.operations] == ["average"]
    assert spec.display_rows()[1].edit_ref == provenance._ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    assert spec.derivation_code() == (
        "import xarray\n\n"
        "derived = xarray.load_dataarray('scan.h5')\n"
        'derived = derived.qsel.mean("x")'
    )


def test_tool_provenance_apply_selection_and_xarray_operations() -> None:
    data = _base_data()

    nonuniform_public = xr.DataArray(
        np.arange(24).reshape((4, 3, 2)),
        dims=("alpha", "eV", "beta"),
        coords={
            "alpha": [0.0, 0.6, 1.7, 3.0],
            "eV": [-0.2, 0.0, 0.2],
            "beta": [1.0, 2.0],
        },
        name="data",
    )
    nonuniform = erlab.interactive.imagetool.slicer.make_dims_uniform(nonuniform_public)
    selection_spec = erlab.interactive.imagetool.provenance.selection(
        provenance.QSelOperation(kwargs={"beta": 2.0}),
        provenance.IselOperation(kwargs={"alpha": slice(1, 3)}),
        provenance.SortCoordOrderOperation(),
    )
    xr.testing.assert_identical(
        selection_spec.apply(nonuniform),
        nonuniform_public.qsel(beta=2.0).isel({"alpha": slice(1, 3)}),
    )

    transformed = erlab.interactive.imagetool.provenance.full_data(
        provenance.IselOperation(kwargs={"z": 0}),
        provenance.SelOperation(kwargs={"y": slice(11.0, 12.0)}),
        provenance.TransposeOperation(dims=("y", "x")),
        provenance.SqueezeOperation(),
        provenance.RenameOperation(name="done"),
    )
    xr.testing.assert_identical(
        transformed.apply(data),
        data.isel({"z": 0})
        .sel({"y": slice(11.0, 12.0)})
        .transpose("y", "x")
        .squeeze()
        .rename("done"),
    )

    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            provenance.AverageOperation(dims=("y",))
        ).apply(data),
        data.qsel.mean("y"),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            provenance.QSelAggregationOperation(
                dims=("y",),
                func="sum",
            )
        ).apply(data),
        data.qsel.sum("y"),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            provenance.CoarsenOperation(
                dim={"y": 2},
                boundary="trim",
                side="left",
                coord_func="mean",
                reducer="mean",
            )
        ).apply(data),
        data.coarsen(y=2, boundary="trim", side="left", coord_func="mean").mean(),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            provenance.ThinOperation(mode="global", factor=2)
        ).apply(data),
        data.thin(2),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            provenance.ThinOperation(mode="per_dim", factors={"x": 2})
        ).apply(data),
        data.thin({"x": 2}),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            provenance.SwapDimsOperation(mapping={"x": "x_alt"})
        ).apply(data),
        data.swap_dims({"x": "x_alt"}),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            provenance.RenameDimsCoordsOperation(mapping={"x": "kx", "x_alt": "label"})
        ).apply(data),
        data.rename({"x": "kx", "x_alt": "label"}),
    )

    assigned = erlab.interactive.imagetool.provenance.full_data(
        provenance.AssignCoordsOperation(
            coord_name="y", values=np.array([100.0, 101.0, 102.0, 103.0])
        )
    ).apply(data)
    expected_assigned = erlab.utils.array.sort_coord_order(
        data.assign_coords(
            {"y": data["y"].copy(data=np.array([100.0, 101.0, 102.0, 103.0]))}
        ),
        keys=data.coords.keys(),
        dims_first=False,
    )
    xr.testing.assert_identical(assigned, expected_assigned)


def test_tool_provenance_rename_dims_coords_round_trip_and_code() -> None:
    data = _string_key_data().assign_coords(
        {"coord-1": xr.DataArray([100.0, 101.0, 102.0], dims=["k-space"])}
    )
    operation = provenance.RenameDimsCoordsOperation(
        mapping={"k-space": "kx", "coord-1": "temperature"}
    )
    expected = data.rename({"k-space": "kx", "coord-1": "temperature"})

    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    assert parsed == operation
    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    spec = provenance.full_data(operation).to_replay_spec()
    code = spec.display_code(parent_data=data)
    assert code is not None
    assert ".rename(" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


def test_tool_provenance_interpolation_operation_round_trip_and_code() -> None:
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=("k-space", "y"),
        coords={"k-space": [0.0, 1.0, 2.0], "y": [10.0, 20.0]},
        name="data",
    )
    values = np.linspace(0.0, 2.0, 5)
    operation = provenance.InterpolationOperation(
        dim="k-space", values=values, method="linear"
    )
    expected = data.interp({"k-space": values}, method="linear")

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data, parent_data=data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    assert "Interpolate" in entry.label
    assert '.interp({"k-space": np.linspace' in entry.code
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    code = (
        provenance.full_data(operation).to_replay_spec().display_code(parent_data=data)
    )
    assert code is not None
    assert any(call.endswith(".interp") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


def test_tool_provenance_leading_edge_operation_round_trip_and_code() -> None:
    ev = np.linspace(0.0, 4.0, 5)
    data = xr.DataArray(
        np.vstack([4.0 - ev, 8.0 - 2.0 * ev, 2.0 - 0.5 * ev]),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": ev},
        name="data",
    )
    operation = provenance.LeadingEdgeOperation(
        fraction=0.5,
        dim="eV",
        direction="positive",
    )
    expected = erlab.analysis.interpolate.leading_edge(data)

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data, parent_data=data), expected)

    payload = provenance.full_data(operation).model_dump(mode="json")
    json.dumps(payload)
    reparsed_spec = provenance.parse_tool_provenance_spec(payload)
    assert reparsed_spec is not None
    xr.testing.assert_identical(reparsed_spec.apply(data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    assert "leading_edge" in entry.code
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    code = (
        provenance.full_data(operation).to_replay_spec().display_code(parent_data=data)
    )
    assert code is not None
    assert any(call.endswith(".leading_edge") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


@pytest.mark.parametrize(
    ("values", "expected_call"),
    [
        (np.array([100.0, 101.0, 102.0, 103.0]), "np.arange"),
        (np.linspace(0.25, 1.0, 4), "np.linspace"),
        (np.ones(4), "np.linspace"),
        (np.array([0.0, 0.5, 2.0, 3.0]), "np.array"),
    ],
)
def test_tool_provenance_assign_coords_replay_display_code(
    values: np.ndarray, expected_call: str
) -> None:
    data = _base_data()

    spec = provenance.full_data(
        provenance.AssignCoordsOperation(coord_name="y", values=values)
    ).to_replay_spec()

    code = spec.display_code(parent_data=data)
    assert code is not None
    call_names = _generated_call_names(code)
    assert any(call.endswith(".assign_coords") for call in call_names)
    assert "erlab.utils.array.sort_coord_order" not in call_names
    assert expected_call in call_names

    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_allclose(
        namespace["derived"],
        data.assign_coords({"y": data["y"].copy(data=values)}),
    )


def test_tool_provenance_assign_coords_single_value_uses_linspace() -> None:
    data = xr.DataArray(
        np.arange(1),
        dims=("x",),
        coords={"x": np.array([0.0])},
        name="data",
    )
    values = np.array([5.0])

    spec = provenance.full_data(
        provenance.AssignCoordsOperation(coord_name="x", values=values)
    ).to_replay_spec()

    code = spec.display_code(parent_data=data)
    assert code is not None
    call_names = _generated_call_names(code)
    assert "np.linspace" in call_names
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_allclose(
        namespace["derived"],
        data.assign_coords({"x": data["x"].copy(data=values)}),
    )


def test_tool_provenance_assign_scalar_coord_operation() -> None:
    data = _base_data()
    operation = provenance.AssignScalarCoordOperation(
        coord_name="temperature", value=21.5
    )
    expected = erlab.utils.array.sort_coord_order(
        data.assign_coords({"temperature": 21.5}),
        keys=data.coords.keys(),
        dims_first=False,
    )

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    assert parsed == operation

    code = (
        provenance.full_data(operation).to_replay_spec().display_code(parent_data=data)
    )
    assert code is not None
    assert any(call.endswith(".assign_coords") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"], data.assign_coords(temperature=21.5)
    )


def test_tool_provenance_nonfinite_coord_and_attr_code() -> None:
    data = _base_data()

    scalar_spec = provenance.full_data(
        provenance.AssignScalarCoordOperation(coord_name="temperature", value=np.nan)
    )
    scalar_code = typing.cast("str", scalar_spec.derivation_code())
    assert "np.nan" in scalar_code
    scalar_namespace = _exec_generated_code(scalar_code, {"data": data.copy(deep=True)})
    assert np.isnan(scalar_namespace["derived"].coords["temperature"].item())

    attrs_spec = provenance.full_data(
        provenance.AssignAttrsOperation(
            attrs={
                "offset": np.inf,
                "bad": np.nan,
                "complex": complex(float("nan"), float("inf")),
            }
        )
    )
    attrs_code = typing.cast("str", attrs_spec.derivation_code())
    assert "np.nan" in attrs_code
    assert "np.inf" in attrs_code
    assert "complex(np.nan, np.inf)" in attrs_code
    attrs_namespace = _exec_generated_code(attrs_code, {"data": data.copy(deep=True)})
    assert np.isinf(attrs_namespace["derived"].attrs["offset"])
    assert np.isnan(attrs_namespace["derived"].attrs["bad"])
    assert np.isnan(attrs_namespace["derived"].attrs["complex"].real)
    assert np.isinf(attrs_namespace["derived"].attrs["complex"].imag)

    coord_spec = provenance.full_data(
        provenance.AssignCoord1DOperation(
            coord_name="temperature",
            dim="x",
            values=np.array([np.nan, np.inf, -np.inf]),
        )
    )
    coord_code = typing.cast("str", coord_spec.derivation_code())
    assert "np.nan" in coord_code
    assert "np.inf" in coord_code
    coord_namespace = _exec_generated_code(coord_code, {"data": data.copy(deep=True)})
    np.testing.assert_equal(
        coord_namespace["derived"].coords["temperature"].values,
        np.array([np.nan, np.inf, -np.inf]),
    )


def test_tool_provenance_assign_1d_coord_operation() -> None:
    data = _base_data()
    values = np.array(["low", "mid", "high"])
    operation = provenance.AssignCoord1DOperation(
        coord_name="label",
        dim="x",
        values=values,
    )
    expected = erlab.utils.array.sort_coord_order(
        data.assign_coords({"label": ("x", values)}),
        keys=data.coords.keys(),
        dims_first=False,
    )

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    assert parsed == operation

    code = (
        provenance.full_data(operation).to_replay_spec().display_code(parent_data=data)
    )
    assert code is not None
    assert any(call.endswith(".assign_coords") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"], data.assign_coords(label=("x", values))
    )


def test_tool_provenance_assign_attrs_operation() -> None:
    data = _base_data().assign_attrs(source="old", count=1)
    attrs = {"source": "new", "flag": True, "meta": {"scan": 1}}
    operation = provenance.AssignAttrsOperation(attrs=attrs)
    expected = data.assign_attrs(attrs)

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    assert parsed == operation

    code = (
        provenance.full_data(operation).to_replay_spec().display_code(parent_data=data)
    )
    assert code is not None
    assert any(call.endswith(".assign_attrs") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


def test_kspace_configuration_operation_round_trip_and_code() -> None:
    data = _kspace_data()
    operation = provenance.KspaceConfigurationOperation(
        configuration=erlab.constants.AxesConfiguration.Type2
    )
    expected = data.kspace.as_configuration(erlab.constants.AxesConfiguration.Type2)

    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )

    assert parsed == operation
    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    assert operation.derivation_entry().label == "Set kspace configuration(2 Type2)"
    code = operation.replay_code("anglemap", output_name="converted")
    assert code == "converted = anglemap.kspace.as_configuration(2)"
    namespace = _exec_generated_code(code, {"anglemap": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["converted"], expected)


@pytest.mark.parametrize(
    ("operation", "attr_name", "expected"),
    [
        (
            provenance.KspaceWorkFunctionOperation(work_function=4.2),
            "sample_workfunction",
            4.2,
        ),
        (
            provenance.KspaceInnerPotentialOperation(inner_potential=12.0),
            "inner_potential",
            12.0,
        ),
    ],
)
def test_kspace_scalar_statement_operations_round_trip_and_code(
    operation: provenance.ToolProvenanceOperation,
    attr_name: str,
    expected: float,
) -> None:
    data = _kspace_data()
    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )

    result = parsed.apply(data, parent_data=data)

    assert result.attrs[attr_name] == pytest.approx(expected)
    assert data.attrs[attr_name] != pytest.approx(expected)
    code = parsed.replay_code("data", output_name="result")
    assert "result = data.copy(deep=False)" in code
    if attr_name == "sample_workfunction":
        assert "result.kspace.work_function =" in code
        assert "result.attrs" not in code
    else:
        assert "result.kspace.inner_potential =" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    assert namespace["result"].attrs[attr_name] == pytest.approx(expected)
    assert namespace["data"].attrs[attr_name] != pytest.approx(expected)


def test_kspace_set_normal_operation_round_trip_and_code() -> None:
    data = _kspace_data()
    operation = provenance.KspaceSetNormalOperation(
        alpha=1.5,
        beta=-0.5,
        delta=2.0,
    )

    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    result = parsed.apply(data, parent_data=data)

    assert result.kspace.offsets["delta"] == pytest.approx(2.0)
    assert result.kspace.offsets != data.kspace.offsets
    code = parsed.replay_code("data", output_name="result")
    assert "result = data.copy(deep=False)" in code
    assert "result.kspace.set_normal(alpha=1.5, beta=-0.5, delta=2.0)" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["result"], result)
    for key, value in data.kspace.offsets.items():
        assert namespace["data"].kspace.offsets[key] == pytest.approx(value)


def test_kspace_convert_operation_round_trip_and_code() -> None:
    data = _kspace_data()
    operation = provenance.KspaceConvertOperation(
        bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
        resolution={"kx": 0.02, "ky": 0.02},
    )
    expected = data.kspace.convert(
        bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
        resolution={"kx": 0.02, "ky": 0.02},
        silent=True,
    )

    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )

    assert parsed == operation
    xr.testing.assert_allclose(parsed.apply(data, parent_data=data), expected)
    code = parsed.replay_code("data", output_name="result")
    assert "result = data.kspace.convert(" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_allclose(namespace["result"], expected)


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            provenance.ConsoleCall(
                accessor_path=("kspace", "as_configuration"),
                args=(2,),
                display_code="data.kspace.as_configuration(2)",
                has_extra_tracked_inputs=False,
            ),
            provenance.KspaceConfigurationOperation(configuration=2),
        ),
        (
            provenance.ConsoleCall(
                accessor_path=("kspace", "set_normal"),
                args=(1.5, -0.5),
                kwargs={"delta": 2.0},
                display_code="data.kspace.set_normal(1.5, -0.5, delta=2.0)",
                has_extra_tracked_inputs=False,
            ),
            provenance.KspaceSetNormalOperation(alpha=1.5, beta=-0.5, delta=2.0),
        ),
        (
            provenance.ConsoleCall(
                accessor_path=("kspace", "convert"),
                args=(),
                kwargs={
                    "bounds": {"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
                    "resolution": {"kx": 0.02, "ky": 0.02},
                },
                display_code=(
                    "data.kspace.convert(bounds=bounds, resolution=resolution)"
                ),
                has_extra_tracked_inputs=False,
            ),
            provenance.KspaceConvertOperation(
                bounds={"kx": (-0.02, 0.02), "ky": (-0.02, 0.02)},
                resolution={"kx": 0.02, "ky": 0.02},
            ),
        ),
    ],
)
def test_kspace_operations_match_console_calls(
    call: provenance.ConsoleCall,
    operation: provenance.ToolProvenanceOperation,
) -> None:
    assert provenance.operation_from_console_call(call) == operation


def _expected_affine_coord(
    data: xr.DataArray, coord_name: str, scale: float, offset: float
) -> xr.DataArray:
    coord = data.coords[coord_name]
    return erlab.utils.array.sort_coord_order(
        data.assign_coords(
            {coord_name: coord.copy(data=scale * coord.values + offset)}
        ),
        keys=data.coords.keys(),
        dims_first=False,
    )


@pytest.mark.parametrize(
    ("data", "coord_name", "scale", "offset"),
    [
        (_base_data(), "y", 2.0, -1.0),
        (
            _base_data().assign_coords(temp=("x", [100.0, 200.0, 300.0])),
            "temp",
            0.5,
            1.0,
        ),
        (_base_data().assign_coords(temp=20.0), "temp", 1.5, -2.0),
        (
            _base_data().assign_coords({"beam current": ("x", [1.0, 2.0, 4.0])}),
            "beam current",
            3.0,
            0.25,
        ),
    ],
)
def test_tool_provenance_affine_coord_operation(
    data: xr.DataArray, coord_name: str, scale: float, offset: float
) -> None:
    operation = provenance.AffineCoordOperation(
        coord_name=coord_name,
        scale=scale,
        offset=offset,
    )

    expected = _expected_affine_coord(data, coord_name, scale, offset)
    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)

    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data, parent_data=data), expected)

    code = (
        provenance.full_data(operation).to_replay_spec().display_code(parent_data=data)
    )
    assert code is not None
    call_names = _generated_call_names(code)
    assert any(call.endswith(".assign_coords") for call in call_names)
    assert "erlab.utils.array.sort_coord_order" not in call_names
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"],
        data.assign_coords(
            {
                coord_name: data.coords[coord_name].copy(
                    data=scale * data.coords[coord_name].values + offset
                )
            }
        ),
    )


@pytest.mark.parametrize(
    ("scale", "offset", "expected_fragments", "forbidden_fragments"),
    [
        (1.0, -3.7, ("['y'].values - 3.7",), ("1.0 *", "+ -3.7")),
        (2.0, -3.7, ("2.0 *", "['y'].values - 3.7"), ("+ -3.7",)),
        (
            1.0,
            0.0,
            ("['y'].values)",),
            ("1.0 *", "+ 0.0", "- 0.0"),
        ),
    ],
)
def test_tool_provenance_affine_coord_display_code_formats_no_ops(
    scale: float,
    offset: float,
    expected_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
) -> None:
    data = _base_data()
    operation = provenance.AffineCoordOperation(
        coord_name="y",
        scale=scale,
        offset=offset,
    )

    code = (
        provenance.full_data(operation).to_replay_spec().display_code(parent_data=data)
    )
    assert code is not None
    for fragment in expected_fragments:
        assert fragment in code
    for fragment in forbidden_fragments:
        assert fragment not in code

    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"], _expected_affine_coord(data, "y", scale, offset)
    )


@pytest.mark.parametrize(
    ("scale", "offset"),
    [
        (np.nan, 0.0),
        (np.inf, 0.0),
        (1.0, np.nan),
        (1.0, -np.inf),
    ],
)
def test_tool_provenance_affine_coord_rejects_nonfinite_values(
    scale: float, offset: float
) -> None:
    with pytest.raises(
        ValidationError, match="affine coordinate scale and offset must be finite"
    ):
        provenance.AffineCoordOperation(
            coord_name="y",
            scale=scale,
            offset=offset,
        )


def test_tool_provenance_divide_by_coord_operation() -> None:
    data = _base_data().assign_coords(mesh_current=("x", [1.0, 2.0, 4.0]))

    spec = provenance.full_data(
        provenance.DivideByCoordOperation(coord_name="mesh_current")
    )
    expected = (data / data.mesh_current).rename(data.name)
    xr.testing.assert_identical(spec.apply(data), expected)
    code = spec.derivation_code()
    assert code is not None
    assert "derived.mesh_current" in code
    assert ".rename(" not in code
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], data / data.mesh_current)

    reparsed = provenance.parse_tool_provenance_spec(spec.model_dump(mode="json"))
    assert reparsed == spec
    xr.testing.assert_identical(reparsed.apply(data), expected)


def test_tool_provenance_divide_by_coord_fallback_code_and_broadcast() -> None:
    data = _base_data().assign_coords(
        {
            "mesh current": ("x", [1.0, 2.0, 4.0]),
            "mesh_map": (
                ("x", "y"),
                np.arange(12, dtype=float).reshape(3, 4) + 1.0,
            ),
            "mean": ("x", [1.0, 2.0, 4.0]),
        }
    )

    spaced_spec = provenance.full_data(
        provenance.DivideByCoordOperation(coord_name="mesh current")
    )
    spaced_code = spaced_spec.derivation_code()
    assert spaced_code is not None
    assert 'derived.coords["mesh current"]' in spaced_code
    assert ".rename(" not in spaced_code
    namespace = _exec_generated_code(spaced_code, {"data": data})
    xr.testing.assert_identical(
        namespace["derived"], data / data.coords["mesh current"]
    )

    conflict_spec = provenance.full_data(
        provenance.DivideByCoordOperation(coord_name="mean")
    )
    conflict_code = conflict_spec.derivation_code()
    assert conflict_code is not None
    assert 'derived.coords["mean"]' in conflict_code
    assert ".rename(" not in conflict_code
    namespace = _exec_generated_code(conflict_code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], data / data.coords["mean"])

    broadcast_spec = provenance.full_data(
        provenance.DivideByCoordOperation(coord_name="mesh_map")
    )
    xr.testing.assert_identical(
        broadcast_spec.apply(data), (data / data.coords["mesh_map"]).rename(data.name)
    )


def test_tool_provenance_divide_by_coord_rejects_zero_values() -> None:
    data = _base_data().assign_coords(mesh_current=("x", [1.0, 0.0, 4.0]))
    spec = provenance.full_data(
        provenance.DivideByCoordOperation(coord_name="mesh_current")
    )

    with pytest.raises(ValueError, match="zero values"):
        spec.apply(data)


def test_tool_provenance_public_data_replays_on_restored_nonuniform_dims() -> None:
    public = xr.DataArray(
        np.arange(20).reshape((5, 4)),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4)},
        name="data",
    )
    uniform = erlab.interactive.imagetool.slicer.make_dims_uniform(public)

    spec = provenance.public_data(
        provenance.CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        )
    )
    reparsed = provenance.parse_tool_provenance_spec(spec.model_dump(mode="json"))

    assert reparsed is not None
    assert reparsed.kind == "public_data"
    xr.testing.assert_identical(
        reparsed.apply(uniform),
        public.coarsen(x=2, boundary="trim", side="left", coord_func="mean").mean(),
    )

    display_code = reparsed.display_code(parent_data=uniform)
    assert display_code is not None
    assert "coarsen(x=2" in display_code
    assert "x_idx" not in display_code

    restored_spec = provenance.full_data(
        provenance.AverageOperation(dims=("y",)),
        provenance.RestoreNonuniformDimsOperation(),
    )
    reparsed_restored = provenance.parse_tool_provenance_spec(
        restored_spec.model_dump(mode="json")
    )

    assert reparsed_restored is not None
    xr.testing.assert_identical(
        reparsed_restored.apply(uniform),
        public.qsel.mean("y"),
    )
    restored_code = reparsed_restored.display_code(parent_data=uniform)
    assert restored_code is not None
    assert "restore_nonuniform_dims" in restored_code


def test_tool_provenance_preserves_hashable_dims_and_mapping_keys() -> None:
    data = _hashable_data()
    string_key_data = _string_key_data()

    qsel_spec = provenance.full_data(
        provenance.QSelOperation(kwargs={"k-space": 1.0, "k-space_width": 1.0})
    )
    assert qsel_spec.derivation_code() == (
        'derived = data\nderived = derived.qsel({"k-space": 1.0, "k-space_width": 1.0})'
    )
    xr.testing.assert_identical(
        qsel_spec.apply(string_key_data),
        string_key_data.qsel({"k-space": 1.0, "k-space_width": 1.0}),
    )

    isel_spec = provenance.full_data(provenance.IselOperation(kwargs={1: slice(1, 3)}))
    assert (
        isel_spec.derivation_code()
        == "derived = data\nderived = derived.isel({1: slice(1, 3)})"
    )
    xr.testing.assert_identical(isel_spec.apply(data), data.isel({1: slice(1, 3)}))

    transpose_spec = provenance.full_data(
        provenance.TransposeOperation(dims=(("beta", 0), 1))
    )
    assert (
        transpose_spec.derivation_code()
        == 'derived = data\nderived = derived.transpose(*(("beta", 0), 1))'
    )
    xr.testing.assert_identical(
        transpose_spec.apply(data), data.transpose(("beta", 0), 1)
    )

    average_spec = provenance.full_data(provenance.AverageOperation(dims=("k-space",)))
    assert (
        average_spec.derivation_code()
        == 'derived = data\nderived = derived.qsel.mean("k-space")'
    )
    xr.testing.assert_identical(
        average_spec.apply(string_key_data), string_key_data.qsel.mean("k-space")
    )

    tuple_average_spec = provenance.full_data(
        provenance.AverageOperation(dims=(("beta", 0),))
    )
    assert (
        tuple_average_spec.derivation_code()
        == 'derived = data\nderived = derived.qsel.mean((("beta", 0),))'
    )

    aggregate_spec = provenance.full_data(
        provenance.QSelAggregationOperation(dims=("k-space",), func="sum")
    )
    assert (
        aggregate_spec.derivation_code()
        == 'derived = data\nderived = derived.qsel.sum("k-space")'
    )
    xr.testing.assert_identical(
        aggregate_spec.apply(string_key_data), string_key_data.qsel.sum("k-space")
    )

    mean_aggregate_spec = provenance.full_data(
        provenance.QSelAggregationOperation(dims=(("beta", 0),), func="mean")
    )
    assert mean_aggregate_spec.derivation_code() == (
        'derived = data\nderived = derived.qsel.mean((("beta", 0),))'
    )

    dumped = aggregate_spec.model_dump(mode="json")
    assert dumped["operations"][0] == {
        "op": "qsel_aggregate",
        "dims": {provenance._TUPLE_MARKER: ["k-space"]},
        "func": "sum",
    }
    reparsed = provenance.parse_tool_provenance_spec(dumped)
    assert reparsed.operations[0] == aggregate_spec.operations[0]

    coarsen_spec = provenance.full_data(
        provenance.CoarsenOperation(
            dim={1: 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        )
    )
    assert coarsen_spec.derivation_code() == (
        'derived = data\nderived = derived.coarsen(dim={1: 2}, boundary="trim").mean()'
    )
    xr.testing.assert_identical(
        coarsen_spec.apply(data),
        data.coarsen(
            dim={1: 2}, boundary="trim", side="left", coord_func="mean"
        ).mean(),
    )

    thin_spec = provenance.full_data(
        provenance.ThinOperation(mode="per_dim", factors={1: 2})
    )
    assert (
        thin_spec.derivation_code() == "derived = data\nderived = derived.thin({1: 2})"
    )
    xr.testing.assert_identical(thin_spec.apply(data), data.thin({1: 2}))

    swap_spec = provenance.full_data(
        provenance.SwapDimsOperation(mapping={1: "coord_1"})
    )
    assert (
        swap_spec.derivation_code()
        == 'derived = data\nderived = derived.swap_dims({1: "coord_1"})'
    )
    xr.testing.assert_identical(swap_spec.apply(data), data.swap_dims({1: "coord_1"}))

    dumped = tuple_average_spec.model_dump(mode="json")
    assert dumped["operations"][0]["dims"] == {
        provenance._TUPLE_MARKER: [{provenance._TUPLE_MARKER: ["beta", 0]}]
    }

    coarsen_dump = coarsen_spec.model_dump(mode="json")
    assert coarsen_dump["operations"][0]["dim"] == {
        provenance._MAPPING_MARKER: [[1, 2]]
    }


def test_tool_provenance_display_entries_streamline_live_source() -> None:
    data = _base_data()

    hidden_spec = provenance.full_data(
        provenance.IselOperation(kwargs={}),
        provenance.SortCoordOrderOperation(),
        provenance.TransposeOperation(dims=data.dims),
        provenance.SqueezeOperation(),
    )
    assert [entry.label for entry in hidden_spec.display_entries(parent_data=data)] == [
        "Start from current parent ImageTool data"
    ]
    assert hidden_spec.display_code(parent_data=data) is None

    squeezed_spec = provenance.full_data(
        provenance.IselOperation(kwargs={"z": slice(0, 1)}),
        provenance.SqueezeOperation(),
    )
    squeezed_entries = squeezed_spec.display_entries(parent_data=data)
    assert squeezed_entries[0].label == "Start from current parent ImageTool data"
    assert squeezed_entries[-1].label == "squeeze()"
    squeezed_code = typing.cast("str", squeezed_spec.display_code(parent_data=data))
    squeezed_namespace = _exec_generated_code(
        squeezed_code,
        {"data": data.copy(deep=True)},
    )
    squeezed = squeezed_namespace["derived"]
    assert isinstance(squeezed, xr.DataArray)
    xr.testing.assert_identical(
        squeezed,
        data.isel(z=slice(0, 1)).squeeze(),
    )


def test_tool_provenance_display_entries_keep_ambiguous_script_steps() -> None:

    spec = provenance.script(
        provenance.ScriptCodeOperation(label="isel()", code="derived = derived.isel()"),
        provenance.ScriptCodeOperation(
            label="Sort coordinates to parent order",
            code=(
                "derived = erlab.utils.array.sort_coord_order("
                "derived, data.coords.keys())"
            ),
        ),
        provenance.ScriptCodeOperation(
            label="Custom coordinate-order step",
            code=(
                "derived = erlab.utils.array.sort_coord_order("
                "derived, data.coords.keys(), dims_first=False)"
            ),
        ),
        provenance.ScriptCodeOperation(
            label="transpose(('x', 'y', 'z'))",
            code="derived = derived.transpose(*('x', 'y', 'z'))",
        ),
        provenance.ScriptCodeOperation(
            label="squeeze()",
            code="derived = derived.squeeze()",
        ),
        start_label="Start from current analysis-tool input data",
        seed_code="derived = data",
    )

    code = spec.display_code()
    assert code is not None
    call_names = _generated_call_names(code)
    assert not any(call.endswith(".isel") for call in call_names)
    assert call_names.count("erlab.utils.array.sort_coord_order") == 1
    assert any(call.rsplit(".", maxsplit=1)[-1] == "transpose" for call in call_names)
    assert any(call.rsplit(".", maxsplit=1)[-1] == "squeeze" for call in call_names)
    namespace = _exec_generated_code(code, {"data": _base_data().copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(
        derived,
        _base_data().transpose(*("x", "y", "z")).squeeze(),
    )


def test_tool_provenance_display_keeps_name_rename_before_script_code() -> None:
    data = _base_data().rename("source")
    spec = provenance.script(
        provenance.RenameOperation(name="renamed"),
        provenance.ScriptCodeOperation(
            label="Use DataArray name",
            code="derived = derived.rename(derived.name + '_used')",
        ),
        start_label="Run script",
        seed_code="derived = data",
        active_name="derived",
    )

    code = typing.cast("str", spec.display_code())

    assert ".rename(" in code
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], data.rename("renamed_used"))


def test_imagetool_selection_source_binding_materializes_current_coordinates() -> None:
    original = _base_data()
    shifted = original.assign_coords(y=[20.0, 21.0, 22.0, 23.0])

    binned = provenance.ImageToolSelectionSourceBinding(
        selection_indexers={"y": slice(1, 4)},
        selection_binned_dims=("y",),
    )
    old_spec = binned.materialize(original)
    new_spec = binned.materialize(shifted)

    assert old_spec.operations[0].decoded_kwargs == {"y": 12.0, "y_width": 3.0}
    assert new_spec.operations[0].decoded_kwargs == {"y": 22.0, "y_width": 3.0}
    xr.testing.assert_identical(
        new_spec.apply(shifted), shifted.qsel(y=22.0, y_width=3.0)
    )

    unbinned = provenance.ImageToolSelectionSourceBinding(selection_indexers={"y": 2})
    unbinned_spec = unbinned.materialize(shifted)
    assert unbinned_spec.operations[0].decoded_kwargs == {"y": 22.0}
    xr.testing.assert_identical(unbinned_spec.apply(shifted), shifted.qsel(y=22.0))

    cropped = provenance.ImageToolSelectionSourceBinding(
        crop_sel_indexers={"y": slice(1, 3)}
    )
    cropped_spec = cropped.materialize(shifted)
    assert cropped_spec.operations[0].decoded_kwargs == {"y": slice(21.0, 22.0)}
    xr.testing.assert_identical(
        cropped_spec.apply(shifted),
        shifted.sel(y=slice(21.0, 22.0)),
    )


def test_imagetool_selection_source_binding_round_trips_and_reuses_operations() -> None:
    data = _base_data()
    binding = provenance.ImageToolSelectionSourceBinding(
        selection_mode="isel",
        selection_indexers={"z": 1},
        crop_sel_indexers={"x": slice(0, 3)},
        crop_isel_indexers={"y": slice(1, 3)},
        transpose_dims=("y", "x"),
        squeeze=True,
    )

    reparsed = provenance.ImageToolSelectionSourceBinding.model_validate(
        binding.model_dump(mode="json")
    )
    assert reparsed == binding

    spec = reparsed.materialize(data)
    assert [op.op for op in spec.operations] == [
        "isel",
        "sel",
        "isel",
        "sort_coord_order",
        "transpose",
        "squeeze",
    ]
    xr.testing.assert_identical(
        spec.apply(data),
        erlab.utils.array.sort_coord_order(
            data.isel(z=1).sel(x=slice(0.0, 2.0)).isel(y=slice(1, 3)),
            data.coords.keys(),
        )
        .transpose("y", "x")
        .squeeze(),
    )


def test_imagetool_selection_source_binding_validates_crop_indexers() -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
    )

    with pytest.raises(ValueError, match="Dimension `missing` not found"):
        provenance.ImageToolSelectionSourceBinding(
            crop_sel_indexers={"missing": slice(0, 1)}
        ).materialize(data)

    with pytest.raises(ValueError, match="Selection for dimension `x` is empty"):
        provenance.ImageToolSelectionSourceBinding(
            crop_sel_indexers={"x": slice(1, 1)}
        ).materialize(data)

    binding = provenance.ImageToolSelectionSourceBinding(
        crop_sel_indexers={"x": typing.cast("typing.Any", 1)}
    )
    spec = binding.materialize(data)
    sel_kwargs = next(op.decoded_kwargs for op in spec.operations if op.op == "sel")

    assert sel_kwargs == {"x": 1.0}


def test_tool_provenance_rejects_unsupported_hashables() -> None:
    class _UnsupportedHashable:
        def __hash__(self) -> int:
            return 0

    with pytest.raises(TypeError, match="provenance hashable fields only support"):
        provenance.AverageOperation(dims=(_UnsupportedHashable(),))


def test_tool_provenance_validation_helpers_and_error_branches() -> None:
    base_operation = provenance.ToolProvenanceOperation()

    assert provenance._format_derivation_value([1, 2]) == "(1, 2)"
    assert provenance._format_selection_step("isel", {}) == "derived = derived.isel()"
    assert provenance._simplify_display_code("if") == "if"
    assert provenance._simplify_display_code("") == ""
    assert (
        provenance._simplify_display_code("for item in []:\n    pass")
        == "for item in []:\n    pass"
    )
    assert provenance._simplify_display_code(
        "derived = data\nresult = derived + 1"
    ) == ("result = data + 1")
    simplified = provenance._simplify_display_code(
        "derived = data\nscale = 2\nresult = derived + scale"
    )
    simplified_namespace = _exec_generated_code(simplified, {"data": 3})
    assert simplified_namespace["result"] == 5
    assert "derived" not in simplified_namespace
    assert (
        provenance._simplify_display_code(
            "derived = data\nscale = 2\nleft, right = (data * scale, data + 1)",
            inline_targets={"derived"},
        )
        == "scale = 2\nleft, right = (data * scale, data + 1)"
    )
    assert (
        provenance._simplify_display_code(
            "left, right = (data - 1, data + 1)\n"
            "derived = left\n"
            "derived = derived.sel(x=0)"
        )
        == "left, right = (data - 1, data + 1)\nderived = left.sel(x=0)"
    )
    invalidated_namespace = _exec_generated_code(
        provenance._simplify_display_code(
            "derived = data + 1\ndata = other\nresult = derived"
        ),
        {"data": 3, "other": 10},
    )
    assert invalidated_namespace["result"] == 4
    rebased = provenance.rebase_default_replay_input(
        "derived = data\nscale = 2\nresult = derived + scale",
        "source_data",
    )
    rebased_namespace = _exec_generated_code(rebased, {"source_data": 3})
    assert rebased_namespace["result"] == 5
    assert "derived" not in rebased_namespace
    assert provenance.uses_default_replay_input("result = data + 1")
    assert not provenance.uses_default_replay_input("result = source_data + 1")
    helper_code = (
        "def normalize(data):\n"
        "    return data / data.max()\n"
        "\n"
        "derived = normalize(data_0)"
    )
    assert not provenance.uses_default_replay_input(helper_code)
    assert (
        provenance.rebase_default_replay_input(helper_code, "source_data")
        == helper_code
    )
    mixed_helper_code = (
        "def normalize(data):\n"
        "    return data / data.max()\n"
        "\n"
        "derived = normalize(data)"
    )
    rebased_helper_code = provenance.rebase_default_replay_input(
        mixed_helper_code, "source_data"
    )
    assert "def normalize(data):\n    return data / data.max()" in rebased_helper_code
    assert "derived = normalize(source_data)" in rebased_helper_code
    replaced_helper_code = provenance._replace_code_identifiers(
        "def normalize(data):\n    return data / data.max()\n\nderived = data",
        {"data": "source_data", "derived": "result"},
    )
    assert "def normalize(data):\n    return data / data.max()" in replaced_helper_code
    assert "result = source_data" in replaced_helper_code

    with pytest.raises(ValueError, match="Expected 2 items"):
        provenance._ensure_float_tuple([1.0], expected_len=2)
    with pytest.raises(TypeError, match="expected an array-like sequence"):
        provenance._coerce_float_sequence("not-a-sequence")
    with pytest.raises(TypeError, match="active_name must be a string"):
        provenance._validate_active_name(1)
    with pytest.raises(ValueError, match="active_name must be a valid"):
        provenance._validate_active_name("for")
    with pytest.raises(TypeError, match="expected a sequence"):
        provenance.ToolProvenanceOperation._coerce_hashable_tuple_field("x")
    with pytest.raises(ValueError, match="Expected 2 items"):
        provenance.ToolProvenanceOperation._coerce_hashable_tuple_field(
            [1], expected_len=2
        )
    assert provenance.ToolProvenanceOperation._coerce_hashable_mapping_field(None) == {}
    with pytest.raises(TypeError, match="expected a mapping"):
        provenance.ToolProvenanceOperation._coerce_hashable_mapping_field([("x", 1)])
    with pytest.raises(NotImplementedError):
        base_operation.apply(_base_data(), parent_data=_base_data())
    with pytest.raises(NotImplementedError):
        base_operation.derivation_entry()
    with pytest.raises(TypeError, match="must be mappings"):
        provenance.parse_tool_provenance_operation(1)
    with pytest.raises(TypeError, match="must include a string `op`"):
        provenance.parse_tool_provenance_operation({"op": 1})
    with pytest.raises(TypeError, match="array-like"):
        provenance.AssignCoordsOperation(coord_name="x", values=object())
    with pytest.raises(TypeError, match=r"xarray\.Dataset"):
        provenance.CorrectWithEdgeOperation(edge_fit=object())

    assert (
        provenance.ToolProvenanceSpec(kind="full_data", operations=None).operations
        == ()
    )
    with pytest.raises(ValidationError, match="must define `start_label`"):
        provenance.ToolProvenanceSpec(kind="script", active_name="derived")
    with pytest.raises(ValidationError, match="Only script or file provenance specs"):
        provenance.ToolProvenanceSpec(kind="full_data", start_label="bad")
    with pytest.raises(TypeError, match="Script and file provenance use"):
        provenance.script(
            start_label="Start", active_name="derived"
        )._display_operations()


def test_select_coord_operation_round_trips_and_applies() -> None:
    data = _base_data().assign_coords(temp=("x", [100.0, 200.0, 300.0]))
    operation = provenance.SelectCoordOperation(coord_name="temp")

    xr.testing.assert_identical(
        operation.apply(data, parent_data=data), data.coords["temp"]
    )

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], data.coords["temp"])

    parsed = provenance.parse_tool_provenance_operation(
        operation.model_dump(mode="json")
    )
    assert parsed == operation

    spec = provenance.public_data(operation)
    assert provenance.require_live_source_spec(spec) == spec
    xr.testing.assert_identical(spec.apply(data), data.coords["temp"])


def test_tool_provenance_remaining_operation_and_display_branches(monkeypatch) -> None:
    data = _base_data()

    xr.testing.assert_identical(
        provenance.full_data(provenance.TransposeOperation()).apply(data),
        data.transpose(*reversed(data.dims)),
    )
    assert provenance.TransposeOperation().derivation_entry().code == (
        "derived = derived.transpose(*reversed(derived.dims))"
    )
    assert provenance.SortCoordOrderOperation().derivation_entry().copyable is True
    assert (
        provenance.SelOperation(kwargs={"x": 1.0})
        .derivation_entry()
        .label.startswith("sel(")
    )
    rename_entry = provenance.RenameOperation(name="renamed").derivation_entry()
    assert rename_entry.copyable is True
    assert rename_entry.code is not None
    namespace = _exec_generated_code(
        rename_entry.code,
        {"derived": data.copy(deep=True)},
    )
    xr.testing.assert_identical(namespace["derived"], data.rename("renamed"))
    assert provenance.full_data().derivation_code() is None
    assert (
        provenance.script(start_label="Start", active_name="derived").display_code()
        is None
    )
    edge_entry = provenance.CorrectWithEdgeOperation(
        edge_fit=xr.Dataset(), shift_coords=True
    ).derivation_entry()
    assert edge_entry.copyable is True
    assert edge_entry.code is not None

    with pytest.raises(TypeError, match="script_code operations"):
        provenance.ScriptCodeOperation(label="Step", code="derived = data").apply(
            data, parent_data=data
        )
    with pytest.raises(ValidationError, match="thin global mode requires factor"):
        provenance.ThinOperation(mode="global")
    with pytest.raises(ValidationError, match="thin per_dim mode requires factors"):
        provenance.ThinOperation(mode="per_dim")
    assert provenance.ThinOperation(
        mode="global", factor=2
    ).derivation_entry().code == ("derived = derived.thin(2)")

    monkeypatch.setattr(
        erlab.interactive.utils,
        "generate_code",
        lambda *_args, assign=None, **_kwargs: (
            "generated()" if assign is None else f"{assign} = generated()"
        ),
    )
    assert (
        provenance.RotateOperation(angle=45.0, axes=("x", "y"), center=(0.0, 0.0))
        .derivation_entry()
        .code
        == "derived = generated()"
    )
    assert (
        provenance.SymmetrizeOperation(dim="x", center=0.0).derivation_entry().code
        == "derived = generated()"
    )
    assert (
        provenance.SymmetrizeNfoldOperation(fold=4, axes=("x", "y"))
        .derivation_entry()
        .code
        == "derived = generated()"
    )
    symmetrize_nfold_payload = provenance.SymmetrizeNfoldOperation(
        fold=4, axes=("x", "y"), center=(0.0, 0.0)
    ).model_dump(mode="json")
    assert provenance.parse_tool_provenance_operation(
        symmetrize_nfold_payload
    ) == provenance.SymmetrizeNfoldOperation(fold=4, axes=("x", "y"), center=(0.0, 0.0))

    assign_entry = provenance.AssignCoordsOperation(
        coord_name="x", values=np.array([2.0, 1.0, 0.0])
    ).derivation_entry()
    assert assign_entry.copyable is True
    assert "assign_coords" in typing.cast("str", assign_entry.code)

    ambiguous = provenance.full_data(
        provenance.SelOperation(kwargs={"missing": 0}),
        provenance.SqueezeOperation(),
    )
    assert [entry.label for entry in ambiguous.display_entries(parent_data=data)] == [
        "Start from current parent ImageTool data",
        "sel(missing=0)",
        "squeeze()",
    ]

    parent = provenance.script(
        start_label="Start from watched variable 'my_1d'",
        seed_code="derived = my_1d",
    )
    promoted = provenance.mark_promoted_1d_source(data.copy(deep=False))
    assert (
        provenance.compose_display_provenance(
            parent,
            provenance.selection(provenance.IselOperation(kwargs={"x": 0})),
            parent_data=promoted,
        )
        is not parent
    )
    assert (
        provenance.compose_display_provenance(
            parent,
            provenance.selection(provenance.AverageOperation(dims=("x",))),
            parent_data=promoted,
        )
        is not parent
    )
    assert (
        provenance.direct_replay_input_name(
            provenance.script(start_label="Start", seed_code="prepared = data")
        )
        is None
    )
    assert (
        provenance.direct_replay_input_name(
            provenance.script(start_label="Start", seed_code="derived = for")
        )
        is None
    )
    assert provenance.compose_full_provenance(parent, None) == parent


def test_append_display_operation_preserves_final_rename_for_live_sources() -> None:
    operation = provenance.NormalizeOperation(dims=("x",), mode="min")
    spec = provenance.full_data().append_final_rename("filtered")

    displayed = spec.append_display_operation(operation)

    assert [op.op for op in displayed.operations] == [
        "normalize",
        "rename",
    ]
    assert displayed.operations[-1] == provenance.RenameOperation(name="filtered")


def test_append_display_operation_rejects_non_live_sources() -> None:
    operation = provenance.NormalizeOperation(dims=("x",), mode="min")
    spec = provenance.script(
        start_label="Evaluate console expression",
        active_name="derived",
    )

    with pytest.raises(TypeError, match="live sources"):
        spec.append_display_operation(operation)


def test_tool_provenance_apply_analysis_operations(monkeypatch) -> None:
    data = _base_data()
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})
    calls: list[tuple[str, dict[str, object]]] = []

    def _record(name: str):
        def _inner(data_arg, *args, **kwargs):
            calls.append((name, {"args": args, "kwargs": kwargs}))
            return data_arg.assign_attrs(last_op=name)

        return _inner

    monkeypatch.setattr(erlab.analysis.transform, "rotate", _record("rotate"))
    monkeypatch.setattr(erlab.analysis.transform, "symmetrize", _record("symmetrize"))
    monkeypatch.setattr(
        erlab.analysis.transform, "symmetrize_nfold", _record("symmetrize_nfold")
    )
    monkeypatch.setattr(
        erlab.analysis.gold, "correct_with_edge", _record("correct_with_edge")
    )
    monkeypatch.setattr(
        erlab.analysis.interpolate, "slice_along_path", _record("slice_along_path")
    )
    monkeypatch.setattr(
        erlab.analysis.mask, "mask_with_polygon", _record("mask_with_polygon")
    )

    rotate_spec = erlab.interactive.imagetool.provenance.full_data(
        provenance.RotateOperation(
            angle=45.0, axes=("x", "y"), center=(0.5, 1.5), reshape=False, order=3
        )
    )
    assert rotate_spec.apply(data).attrs["last_op"] == "rotate"

    symmetrize_spec = erlab.interactive.imagetool.provenance.full_data(
        provenance.SymmetrizeOperation(
            dim="x", center=1.0, subtract=True, mode="valid", part="below"
        )
    )
    assert symmetrize_spec.apply(data).attrs["last_op"] == "symmetrize"

    symmetrize_nfold_spec = erlab.interactive.imagetool.provenance.full_data(
        provenance.SymmetrizeNfoldOperation(
            fold=4,
            axes=("x", "y"),
            center={"x": 1.0, "y": 11.0},
            reshape=True,
            order=2,
        )
    )
    assert symmetrize_nfold_spec.apply(data).attrs["last_op"] == "symmetrize_nfold"

    edge_spec = erlab.interactive.imagetool.provenance.full_data(
        provenance.CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False)
    )
    assert edge_spec.apply(data).attrs["last_op"] == "correct_with_edge"
    entries = edge_spec.derivation_entries()
    assert entries[-1].copyable is True
    assert entries[-1].code is not None
    assert edge_spec.derivation_code() is not None

    path_spec = erlab.interactive.imagetool.provenance.full_data(
        provenance.SliceAlongPathOperation(
            vertices={"x": [0.0, 1.0], "y": [10.0, 12.0]},
            step_size=0.5,
            dim_name="path",
        )
    )
    assert path_spec.apply(data).attrs["last_op"] == "slice_along_path"

    mask_spec = erlab.interactive.imagetool.provenance.full_data(
        provenance.MaskWithPolygonOperation(
            vertices=np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]),
            dims=("x", "y"),
            invert=True,
            drop=False,
        )
    )
    assert mask_spec.apply(data).attrs["last_op"] == "mask_with_polygon"

    call_names = [name for name, _ in calls]
    assert call_names == [
        "rotate",
        "symmetrize",
        "symmetrize_nfold",
        "correct_with_edge",
        "slice_along_path",
        "mask_with_polygon",
    ]
    assert calls[0][1]["kwargs"] == {
        "angle": 45.0,
        "axes": ("x", "y"),
        "center": (0.5, 1.5),
        "reshape": False,
        "order": 3,
    }


def test_tool_provenance_roundtrip_correct_with_edge(monkeypatch) -> None:
    data = _base_data()
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})

    monkeypatch.setattr(
        erlab.analysis.gold,
        "correct_with_edge",
        lambda data_arg, *_args, **_kwargs: data_arg.assign_attrs(
            last_op="correct_with_edge"
        ),
    )

    spec = provenance.full_data(
        provenance.CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False)
    )
    payload = spec.model_dump(mode="json")

    reparsed_operation = provenance.parse_tool_provenance_operation(
        payload["operations"][0]
    )
    assert isinstance(
        reparsed_operation,
        provenance.CorrectWithEdgeOperation,
    )
    xr.testing.assert_identical(reparsed_operation.decoded_edge_fit, edge_fit)

    reparsed_spec = erlab.interactive.imagetool.provenance.parse_tool_provenance_spec(
        payload
    )
    assert reparsed_spec is not None
    assert reparsed_spec.apply(data).attrs["last_op"] == "correct_with_edge"
    entries = reparsed_spec.derivation_entries()
    assert entries[-1].copyable is True
    assert entries[-1].code is not None
    namespace = _exec_generated_code(
        typing.cast("str", reparsed_spec.derivation_code()),
        {"data": data.copy(deep=True)},
    )
    assert namespace["derived"].attrs["last_op"] == "correct_with_edge"


def test_correct_with_edge_code_handles_nonfinite_dataset(monkeypatch) -> None:
    data = _base_data()
    edge_fit = xr.Dataset({"edge": ("x", [np.nan, np.inf, -np.inf])})

    def correct_with_edge(data_arg, edge_fit_arg, *, shift_coords=True):
        xr.testing.assert_identical(edge_fit_arg, edge_fit)
        return data_arg.assign_attrs(shift_coords=shift_coords)

    monkeypatch.setattr(erlab.analysis.gold, "correct_with_edge", correct_with_edge)
    spec = provenance.full_data(
        provenance.CorrectWithEdgeOperation(edge_fit=edge_fit, shift_coords=False)
    )
    code = typing.cast("str", spec.derivation_code())

    assert "np.nan" in code
    assert "np.inf" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    assert namespace["derived"].attrs["shift_coords"] is False


def test_tool_provenance_roundtrip_correct_with_edge_fit_dataset(
    gold, gold_fit_res
) -> None:
    spec = provenance.full_data(
        provenance.CorrectWithEdgeOperation(edge_fit=gold_fit_res, shift_coords=False)
    )

    payload = spec.model_dump(mode="json")
    json.dumps(payload)

    reparsed_operation = provenance.parse_tool_provenance_operation(
        payload["operations"][0]
    )
    assert isinstance(reparsed_operation, provenance.CorrectWithEdgeOperation)
    decoded = reparsed_operation.decoded_edge_fit
    xr.testing.assert_identical(
        decoded.drop_vars("modelfit_results"),
        gold_fit_res.drop_vars("modelfit_results"),
    )
    assert (
        decoded.modelfit_results.item().success
        == gold_fit_res.modelfit_results.item().success
    )

    reparsed_spec = provenance.parse_tool_provenance_spec(payload)
    assert reparsed_spec is not None
    xr.testing.assert_allclose(
        reparsed_spec.apply(gold),
        erlab.analysis.gold.correct_with_edge(gold, gold_fit_res, shift_coords=False),
    )


def test_tool_provenance_script_specs_reject_live_source() -> None:

    script_spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Fit current tool data",
            code="result = data.mean()",
        ),
        start_label="Start from current analysis-tool input data",
        seed_code="prepared = data.copy()",
    )
    reparsed_script = provenance.parse_tool_provenance_spec(
        script_spec.model_dump(mode="json")
    )

    assert reparsed_script is not None
    assert reparsed_script.derivation_entries()[0].label == (
        "Start from current analysis-tool input data"
    )
    assert reparsed_script.derivation_code() == (
        "prepared = data.copy()\nresult = data.mean()"
    )
    with pytest.raises(
        TypeError, match="source_spec must be a live ToolProvenanceSpec"
    ):
        provenance.require_live_source_spec(reparsed_script)


def test_tool_provenance_parse_restores_legacy_structured_script_steps() -> None:
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "seed_code": "derived = data",
        "active_name": "derived",
        "operations": [
            {
                "op": "script_code",
                "label": "Concatenate selected ImageTools",
                "code": "derived = left + right",
            },
            {
                "op": "script_code",
                "label": 'Sort By(variables=("x",), ascending=False)',
                "code": 'derived = derived.sortby("x", ascending=False)',
            },
            {
                "op": "script_code",
                "label": "Average(dims=('y',))",
                "code": 'derived = derived.qsel.mean("y")',
            },
        ],
    }

    parsed = provenance.parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == [
        "script_code",
        "sortby",
        "qsel_aggregate",
    ]
    rows = [
        row
        for row in parsed.display_rows()
        if row.replay_ref is not None and row.replay_ref.kind == "operation"
    ]
    assert [row.edit_ref is not None for row in rows] == [False, True, True]
    data = _base_data()
    replayed = provenance.replay_script_provenance(
        parsed,
        {
            "data": data.copy(deep=True),
            "left": data.copy(deep=True),
            "right": data.copy(deep=True),
        },
    )
    xr.testing.assert_identical(
        replayed,
        (data + data).sortby("x", ascending=False).qsel.mean("y"),
    )


def test_tool_provenance_parse_keeps_non_active_self_assignments_as_script() -> None:
    data = xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=("x",),
        coords={"x": [2.0, 1.0, 0.0]},
        name="scan",
    )
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "seed_code": "derived = data",
        "active_name": "derived",
        "operations": [
            {
                "op": "script_code",
                "label": "Prepare temporary data",
                "code": 'temp = derived.sortby("x", ascending=False)',
            },
            {
                "op": "script_code",
                "label": "Sort temporary data",
                "code": 'temp = temp.sortby("x")',
            },
        ],
    }

    parsed = provenance.parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == [
        "script_code",
        "script_code",
    ]
    replayed = provenance.replay_script_provenance(parsed, {"data": data})
    xr.testing.assert_identical(replayed, data)


def test_tool_provenance_parse_keeps_script_input_self_assignment_as_script() -> None:
    data = xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=("x",),
        coords={"x": [2.0, 1.0, 0.0]},
        name="scan",
    )
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "active_name": "derived",
        "script_inputs": [{"name": "data_0", "label": "ImageTool 0"}],
        "operations": [
            {
                "op": "script_code",
                "label": "Copy input",
                "code": "derived = data_0",
            },
            {
                "op": "script_code",
                "label": "Sort input alias",
                "code": 'data_0 = data_0.sortby("x")',
            },
        ],
    }

    parsed = provenance.parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == [
        "script_code",
        "script_code",
    ]
    replayed = provenance.replay_script_provenance(parsed, {"data_0": data})
    xr.testing.assert_identical(replayed, data)


def test_tool_provenance_parse_restores_active_alias_structured_steps() -> None:
    data = xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=("x",),
        coords={"x": [2.0, 1.0, 0.0]},
        name="scan",
    )
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "seed_code": "result = data",
        "active_name": "result",
        "operations": [
            {
                "op": "script_code",
                "label": "Sort active result",
                "code": 'result = result.sortby("x")',
            },
        ],
    }

    parsed = provenance.parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == ["sortby"]
    replayed = provenance.replay_script_provenance(parsed, {"data": data})
    xr.testing.assert_identical(replayed, data.sortby("x"))


def test_tool_provenance_parse_legacy_script_call_parser_edges() -> None:
    def parse_codes(
        *codes: str | None,
        seed_code: str | None = "derived = data",
        active_name: str = "derived",
        script_inputs: tuple[dict[str, str], ...] = (),
        copyable: bool = True,
    ) -> provenance.ToolProvenanceSpec:
        parsed = provenance.parse_tool_provenance_spec(
            {
                "kind": "script",
                "start_label": "Run ImageTool manager action",
                "seed_code": seed_code,
                "active_name": active_name,
                "script_inputs": script_inputs,
                "operations": [
                    {
                        "op": "script_code",
                        "label": "Legacy script step",
                        "code": code,
                        "copyable": copyable,
                    }
                    for code in codes
                ],
            }
        )
        assert parsed is not None
        return parsed

    parsed = parse_codes(
        "derived = derived.isel(x=slice(0, 2))",
        "derived = derived.coarsen(x=2).mean()",
        "derived = derived.qsel(x=1.0)",
    )
    assert [operation.op for operation in parsed.operations] == [
        "isel",
        "coarsen",
        "qsel",
    ]
    assert parsed.operations[0] == provenance.IselOperation(kwargs={"x": slice(0, 2)})
    assert parsed.operations[1] == provenance.CoarsenOperation(
        dim={"x": 2},
        boundary="exact",
        side="left",
        coord_func="mean",
        reducer="mean",
    )
    assert parsed.operations[2] == provenance.QSelOperation(kwargs={"x": 1.0})

    conservative_codes = (
        "derived = derived.isel(x=slice(0, stop))",
        'derived = derived.sortby(np.array(["x"]))',
        "derived = derived.sortby(np.array([unknown]))",
        "derived = derived.sortby(unknown)",
        "derived = derived.isel(**extra)",
        "derived = make_data()",
        'derived = factory().sortby("x")',
        'derived = factory().data.sortby("x")',
        'derived = other.sortby("x")',
        "derived = derived.unknown.accessor()",
        "derived = derived.coarsen(x=step).mean()",
        'derived = derived.sortby("x")\nextra = derived',
        'derived, other = derived.sortby("x"), other',
        "derived = 1",
        'other = derived.sortby("x")',
    )
    parsed = parse_codes(*conservative_codes)
    assert all(
        isinstance(operation, provenance.ScriptCodeOperation)
        for operation in parsed.operations
    )
    assert [operation.code for operation in parsed.operations] == list(
        conservative_codes
    )

    parsed = parse_codes(
        'derived = derived.sortby("x")',
        seed_code="derived =",
        script_inputs=({"name": "derived", "label": "ImageTool"},),
    )
    assert [operation.op for operation in parsed.operations] == ["sortby"]

    parsed = parse_codes('derived = derived.sortby("x")', seed_code=None)
    assert [operation.op for operation in parsed.operations] == ["script_code"]

    parsed = parse_codes('derived = derived.sortby("x")', copyable=False)
    assert [operation.op for operation in parsed.operations] == ["script_code"]

    parsed = parse_codes(None)
    assert [operation.op for operation in parsed.operations] == ["script_code"]

    parsed = parse_codes("derived =")
    assert [operation.op for operation in parsed.operations] == ["script_code"]


def test_tool_provenance_parse_restores_mixed_active_legacy_steps() -> None:
    data = _base_data().assign_coords(x=[2.0, 1.0, 0.0])
    payload = {
        "kind": "script",
        "start_label": "Run ImageTool manager action",
        "seed_code": "result = data",
        "active_name": "result",
        "operations": [
            {
                "op": "script_code",
                "label": "Prepare temporary data",
                "code": "temp = result + 1.0",
            },
            {
                "op": "script_code",
                "label": "Sort active result",
                "code": 'result = result.sortby("x", ascending=False)',
            },
            {
                "op": "script_code",
                "label": "Average active result",
                "code": 'result = result.qsel.mean("y")',
            },
        ],
    }

    parsed = provenance.parse_tool_provenance_spec(payload)

    assert parsed is not None
    assert [operation.op for operation in parsed.operations] == [
        "script_code",
        "sortby",
        "qsel_aggregate",
    ]
    replayed = provenance.replay_script_provenance(parsed, {"data": data})
    xr.testing.assert_identical(
        replayed,
        data.sortby("x", ascending=False).qsel.mean("y"),
    )


def test_current_structured_operations_round_trip_without_script_fallback() -> None:
    operations = _representative_structured_operations()
    assert {operation.op for operation in operations} == set(
        provenance._OPERATION_TYPES
    ) - {"script_code"}

    def assert_round_trip_operations(
        spec: provenance.ToolProvenanceSpec,
        expected_ops: tuple[str, ...],
    ) -> None:
        parsed = provenance.parse_tool_provenance_spec(spec.model_dump(mode="json"))
        assert parsed is not None
        if parsed.kind == "file":
            parsed_ops = tuple(
                operation
                for stage in parsed.replay_stages
                for operation in stage.operations
            )
        else:
            parsed_ops = parsed.operations
        assert tuple(operation.op for operation in parsed_ops) == expected_ops
        assert not any(
            isinstance(op, provenance.ScriptCodeOperation) for op in parsed_ops
        )

    for operation in operations:
        expected = (operation.op,)
        assert_round_trip_operations(provenance.full_data(operation), expected)
        assert_round_trip_operations(provenance.public_data(operation), expected)
        assert_round_trip_operations(provenance.selection(operation), expected)
        assert_round_trip_operations(
            provenance.full_data(operation).to_replay_spec(),
            expected,
        )
        assert_round_trip_operations(
            provenance.file_load(
                start_label="Load data",
                seed_code="derived = data",
                file_load_source=_file_replay_source(),
                replay_stages=(
                    provenance.ReplayStage(
                        source_kind="full_data",
                        operations=(operation,),
                    ),
                ),
            ),
            expected,
        )


def test_tool_replay_provenance_helpers_compose_parent_provenance() -> None:

    parent = provenance.selection(provenance.IselOperation(kwargs={"x": slice(0, 2)}))
    local = provenance.script(
        provenance.ScriptCodeOperation(
            label="Compute tool output",
            code="result = derived.mean()",
        ),
        start_label="Start from current tool input data",
    )

    composed = provenance.compose_full_provenance(parent, local)

    assert composed is not None
    assert composed.derivation_entries()[0].label == (
        "Start from selected parent ImageTool data"
    )
    assert composed.derivation_entries()[-1].label == "Compute tool output"
    assert composed.derivation_code() == (
        "derived = data\nderived = derived.isel(x=slice(0, 2))\nresult = derived.mean()"
    )

    assert provenance.compose_display_provenance(parent, provenance.full_data()) == (
        provenance.to_replay_provenance_spec(parent)
    )


def test_tool_provenance_compose_full_uses_parent_active_name_for_live_local() -> None:
    data = _base_data()
    parent = provenance.script(
        provenance.ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = data + 1",
        ),
        start_label="Start from current tool input data",
        active_name="result",
    )
    local = provenance.full_data(provenance.AverageOperation(dims=("x",)))

    composed = provenance.compose_full_provenance(parent, local)

    assert composed is not None
    code = composed.derivation_code()
    assert code == (
        'result = data + 1\nderived = result\nderived = derived.qsel.mean("x")'
    )
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, (data + 1).qsel.mean("x"))


def test_tool_provenance_compose_full_preserves_structured_live_steps() -> None:
    data = _base_data()
    parent = provenance.script(
        provenance.ScriptCodeOperation(
            label="Concatenate selected ImageTools",
            code="derived = data_0 + data_1",
        ),
        start_label="Run ImageTool manager action",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(name="data_0", label="ImageTool 0: scan"),
            provenance.ScriptInput(name="data_1", label="ImageTool 1: scan"),
        ),
    )
    local = provenance.full_data(
        provenance.SortByOperation(variables=("x",), ascending=False),
        provenance.AverageOperation(dims=("y",)),
    )

    composed = provenance.compose_full_provenance(parent, local)

    assert composed is not None
    assert [operation.op for operation in composed.operations] == [
        "script_code",
        "sortby",
        "average",
    ]
    rows = [
        row
        for row in composed.display_rows()
        if row.replay_ref is not None and row.replay_ref.kind == "operation"
    ]
    assert [row.edit_ref is not None for row in rows] == [False, True, True]
    derived = provenance.replay_script_provenance(
        composed,
        {
            "data_0": data.copy(deep=True),
            "data_1": data.copy(deep=True),
        },
    )
    xr.testing.assert_identical(
        derived,
        (data + data).sortby("x", ascending=False).qsel.mean("y"),
    )


def test_tool_provenance_script_context_binding_replays_current_output() -> None:
    data = _base_data()
    parent = provenance.script(
        provenance.ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = derived + 1",
        ),
        start_label="Start from current tool input data",
        seed_code="derived = data",
        active_name="result",
    )
    local = provenance.script(
        provenance.ScriptCodeOperation(
            label="Offset copied result",
            code="result = derived + 2",
        ),
        start_label="Start from current ImageTool data",
        active_name="result",
    )
    current = provenance.replay_script_provenance(parent, {"data": data})
    expected = provenance.replay_script_provenance(
        local,
        {"data": current, "derived": current},
    )

    composed = provenance.compose_full_provenance(
        parent,
        local,
        script_context_names=("data", "derived", "data"),
    )

    assert composed is not None
    assert [entry.label for entry in composed.derivation_entries()] == [
        "Start from current tool input data",
        "Compute intermediate result",
        "Offset copied result",
    ]
    assert [
        operation.derivation_entry().label for operation in composed.operations
    ] == [
        "Compute intermediate result",
        "Offset copied result",
    ]
    assert [
        binding.model_dump(mode="json") for binding in composed.script_context_bindings
    ] == [{"operation_index": 1, "names": ["data", "derived"]}]

    replayed = provenance.replay_script_provenance(composed, {"data": data})
    xr.testing.assert_identical(replayed, expected)
    code = typing.cast("str", composed.derivation_code())
    assert "Start from current ImageTool data" not in code
    assert "derived = derived" not in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    result = namespace["result"]
    assert isinstance(result, xr.DataArray)
    xr.testing.assert_identical(result, expected)


def test_tool_provenance_script_context_binding_validation() -> None:
    operation = provenance.ScriptCodeOperation(
        label="Offset copied result",
        code="result = derived + 2",
    )

    spec = provenance.ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
        operations=(operation,),
        script_context_bindings=[
            {"operation_index": 0, "names": ["data", "derived", "data"]},
        ],
    )
    assert [
        binding.model_dump(mode="json") for binding in spec.script_context_bindings
    ] == [{"operation_index": 0, "names": ["data", "derived"]}]
    assert (
        provenance.ToolProvenanceSpec(
            kind="script",
            start_label="Run script",
            active_name="result",
            operations=(operation,),
            script_context_bindings=None,
        ).script_context_bindings
        == ()
    )

    invalid_payloads: tuple[tuple[typing.Any, type[BaseException], str], ...] = (
        (
            [{"operation_index": True, "names": ["data"]}],
            TypeError,
            "operation index",
        ),
        (
            [{"operation_index": -1, "names": ["data"]}],
            ValidationError,
            "non-negative",
        ),
        (
            [{"operation_index": 0, "names": "data"}],
            TypeError,
            "names must be a sequence",
        ),
        (
            [{"operation_index": 0, "names": [None]}],
            ValidationError,
            "must not be None",
        ),
        (
            [{"operation_index": 0, "names": []}],
            ValidationError,
            "must not be empty",
        ),
        ("bad", TypeError, "must be a sequence"),
    )
    for bindings, exc_type, message in invalid_payloads:
        with pytest.raises(exc_type, match=message):
            provenance.ToolProvenanceSpec(
                kind="script",
                start_label="Run script",
                active_name="result",
                operations=(operation,),
                script_context_bindings=bindings,
            )

    with pytest.raises(ValidationError, match="operation boundary"):
        provenance.ToolProvenanceSpec(
            kind="script",
            start_label="Run script",
            active_name="result",
            operations=(operation,),
            script_context_bindings=[
                {"operation_index": 1, "names": ["data"]},
            ],
        )
    with pytest.raises(ValidationError, match="file provenance specs"):
        provenance.ToolProvenanceSpec(
            kind="file",
            start_label="Load source",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=_file_replay_source(),
            script_context_bindings=[
                {"operation_index": 0, "names": ["data"]},
            ],
        )
    with pytest.raises(ValidationError, match="Only script or file provenance specs"):
        provenance.ToolProvenanceSpec(
            kind="full_data",
            script_context_bindings=[
                {"operation_index": 0, "names": ["data"]},
            ],
        )


def test_tool_provenance_script_context_bindings_follow_operation_edits() -> None:
    first = provenance.ScriptCodeOperation(
        label="Compute intermediate result",
        code="result = derived + 1",
    )
    second = provenance.ScriptCodeOperation(
        label="Offset copied result",
        code="result = derived + 2",
    )
    average = provenance.AverageOperation(dims=("x",))
    spec = provenance.ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
        operations=(first, second, average),
        script_context_bindings=[
            {"operation_index": 1, "names": ["data", "derived"]},
            {"operation_index": 2, "names": ["data"]},
        ],
    )

    def binding_payloads(
        value: provenance.ToolProvenanceSpec,
    ) -> list[dict[str, typing.Any]]:
        return [
            binding.model_dump(mode="json") for binding in value.script_context_bindings
        ]

    expanded = spec._replace_operation_ref(
        provenance._ProvenanceStepRef("operation", operation_index=0),
        (first, provenance.SqueezeOperation()),
    )
    assert binding_payloads(expanded) == [
        {"operation_index": 2, "names": ["data", "derived"]},
        {"operation_index": 3, "names": ["data"]},
    ]

    replaced_at_binding = spec._replace_operation_ref(
        provenance._ProvenanceStepRef("operation", operation_index=1),
        (
            provenance.SqueezeOperation(),
            provenance.AssignAttrsOperation(attrs={"edited": True}),
        ),
    )
    assert binding_payloads(replaced_at_binding) == [
        {"operation_index": 1, "names": ["data", "derived"]},
        {"operation_index": 3, "names": ["data"]},
    ]

    deleted_last = spec._replace_operation_ref(
        provenance._ProvenanceStepRef("operation", operation_index=2),
        (),
    )
    assert binding_payloads(deleted_last) == [
        {"operation_index": 1, "names": ["data", "derived"]},
    ]

    through_second = spec._prefix_through_ref(
        provenance._ProvenanceStepRef("operation", operation_index=1)
    )
    assert binding_payloads(through_second) == [
        {"operation_index": 1, "names": ["data", "derived"]},
    ]
    before_second = spec._prefix_before_ref(
        provenance._ProvenanceStepRef("operation", operation_index=1)
    )
    assert binding_payloads(before_second) == []
    start_only = spec._prefix_through_ref(provenance._ProvenanceStepRef("start"))
    assert start_only.operations == ()
    assert start_only.script_context_bindings == ()


def test_tool_provenance_operation_group_replacement_preserves_script_context() -> None:
    grouped = provenance.stamp_operation_group(
        (
            provenance.ScriptCodeOperation(
                label="Offset copied result",
                code="result = derived + 2",
            ),
            provenance.AverageOperation(dims=("x",)),
        ),
        kind="demo",
        group_id="group-1",
    )
    spec = provenance.ToolProvenanceSpec(
        kind="script",
        start_label="Run script",
        seed_code="derived = data",
        active_name="result",
        operations=(
            provenance.ScriptCodeOperation(
                label="Compute intermediate result",
                code="result = derived + 1",
            ),
            *grouped,
        ),
        script_context_bindings=[
            {"operation_index": 1, "names": ["data", "derived"]},
        ],
    )

    replaced = spec._replace_operation_group_ref(
        provenance._ProvenanceStepRef("operation", operation_index=2),
        (provenance.SqueezeOperation(),),
        kind="demo",
    )

    assert [operation.op for operation in replaced.operations] == [
        "script_code",
        "squeeze",
    ]
    assert [
        binding.model_dump(mode="json") for binding in replaced.script_context_bindings
    ] == [{"operation_index": 1, "names": ["data", "derived"]}]


def test_tool_provenance_group_ref_helpers_cover_invalid_and_stage_refs() -> None:
    operations = provenance.stamp_operation_group(
        (
            provenance.AverageOperation(dims=("x",)),
            provenance.SqueezeOperation(),
        ),
        kind="demo",
    )
    spec = provenance.full_data(*operations)
    ref = provenance._ProvenanceStepRef("operation", operation_index=0)

    assert spec._operation_group_range_ref(ref, kind="demo") == (0, 2)
    assert (
        spec._operation_group_range_ref(
            provenance._ProvenanceStepRef("start"),
            kind="demo",
        )
        is None
    )
    with pytest.raises(ValueError, match="complete operation group"):
        spec._replace_operation_group_ref(ref, (), kind="other")
    with pytest.raises(ValueError, match="operation provenance row"):
        spec._replace_operation_range_ref(
            provenance._ProvenanceStepRef("start"),
            0,
            1,
            (),
        )
    with pytest.raises(ValueError, match="non-empty operation range"):
        spec._replace_operation_range_ref(ref, 1, 1, ())

    deleted = spec._delete_operation_group_ref(ref, kind="demo")
    assert deleted.operations == ()

    file_spec = _file_provenance_spec().append_replay_stage(provenance.full_data())
    stage_spec = file_spec.append_replay_stage(provenance.full_data(*operations))
    stage_ref = provenance._ProvenanceStepRef(
        "operation",
        operation_index=1,
        stage_index=1,
    )
    assert stage_spec._operation_group_range_ref(stage_ref, kind="demo") == (0, 2)
    replaced = stage_spec._replace_operation_group_ref(
        stage_ref,
        (provenance.ThinOperation(mode="per_dim", factors={"x": 2}),),
        kind="demo",
    )
    assert [stage.operations for stage in replaced.replay_stages] == [
        (),
        (provenance.ThinOperation(mode="per_dim", factors={"x": 2}),),
    ]
    assert (
        stage_spec._operation_group_range_ref(
            provenance._ProvenanceStepRef(
                "operation",
                operation_index=0,
                stage_index=3,
            ),
            kind="demo",
        )
        is None
    )


def test_tool_provenance_script_context_names_are_validation_only() -> None:
    parent = provenance.script(
        provenance.ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = derived + 1",
        ),
        start_label="Start from current tool input data",
        seed_code="derived = data",
        active_name="result",
    )
    local_script = provenance.script(
        provenance.ScriptCodeOperation(
            label="Offset copied result",
            code="result = derived + 2",
        ),
        start_label="Run pasted script",
        active_name="result",
    )
    local_structured = provenance.full_data(provenance.AverageOperation(dims=("x",)))

    with pytest.raises(ValueError, match="script context names"):
        provenance.compose_full_provenance(
            parent,
            local_script,
            script_context_names=(typing.cast("str", None),),
        )
    composed = provenance.compose_full_provenance(
        parent,
        local_structured,
        script_context_names=("data", "derived"),
    )

    assert composed is not None
    assert composed.script_context_bindings == ()


def test_file_load_source_replay_call_round_trips() -> None:

    xarray_source = provenance.FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text='engine="h5netcdf"',
        replay_call=provenance.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selection=provenance.FileDataSelection(kind="dataarray"),
            cast_float64=True,
        ),
        load_code='import xarray\n\ndata = xarray.load_dataarray("/tmp/scan.h5")',
    )
    parsed_xarray = provenance.FileLoadSource.model_validate(
        xarray_source.model_dump(mode="json")
    )
    assert parsed_xarray == xarray_source
    assert parsed_xarray.replay_call.kind == "callable"
    assert parsed_xarray.replay_call.target == "xarray.load_dataarray"
    assert parsed_xarray.replay_call.kwargs == {"engine": "h5netcdf"}
    assert parsed_xarray.replay_call.selection == provenance.FileDataSelection(
        kind="dataarray"
    )
    assert parsed_xarray.replay_call.cast_float64 is True

    legacy_call = provenance.FileReplayCall.model_validate(
        {
            "kind": "callable",
            "target": "xarray.load_dataarray",
            "selected_index": 2,
        }
    )
    assert legacy_call.selection == provenance.FileDataSelection(
        kind="parsed_index", value=2
    )
    assert "selected_index" not in legacy_call.model_dump(mode="json")

    erlab_source = provenance.FileLoadSource(
        path="data_002.h5",
        loader_label="Loader",
        loader_text="example",
        kwargs_text="(none)",
        replay_call=provenance.FileReplayCall(
            kind="erlab_loader",
            target="example",
            kwargs={},
            selection=provenance.FileDataSelection(kind="dataarray"),
        ),
        load_code="erlab.io.set_loader('example')\ndata = erlab.io.load(2)",
    )
    parsed_erlab = provenance.FileLoadSource.model_validate(
        erlab_source.model_dump(mode="json")
    )
    assert parsed_erlab == erlab_source
    assert parsed_erlab.replay_call.kind == "erlab_loader"
    assert parsed_erlab.replay_call.target == "example"


def test_file_provenance_validation_rejects_invalid_payloads() -> None:
    replay_stage = provenance.ReplayStage(source_kind="full_data")
    file_source = _file_replay_source()

    with pytest.raises(ValidationError, match="parsed file selection index"):
        provenance.FileReplayCall(
            kind="callable", target="xarray.load_dataarray", selected_index=-1
        )
    with pytest.raises(ValidationError, match="target"):
        provenance.FileReplayCall(kind="callable", target="", selected_index=0)

    bad_kwargs_call = provenance.FileReplayCall.model_construct(
        kind="callable",
        target="xarray.load_dataarray",
        kwargs={1: "bad"},
        selection=provenance.FileDataSelection(kind="dataarray"),
    )
    with pytest.raises(TypeError, match="string keys"):
        bad_kwargs_call._validate_replay_call()

    assert (
        provenance.ReplayStage(source_kind="full_data", operations=None).operations
        == ()
    )
    with pytest.raises(TypeError, match="replay stage operations"):
        provenance.ReplayStage(source_kind="full_data", operations=1)
    with pytest.raises(TypeError, match="script-only operations"):
        provenance.ReplayStage(
            source_kind="full_data",
            operations=[
                provenance.ScriptCodeOperation(
                    label="Generated", code="derived = derived"
                )
            ],
        )
    with pytest.raises(TypeError, match="source must not be None"):
        provenance.ReplayStage.from_source_spec(typing.cast("typing.Any", None))

    assert (
        provenance.ToolProvenanceSpec(
            kind="full_data", replay_stages=None
        ).replay_stages
        == ()
    )
    with pytest.raises(TypeError, match="Serialized replay stages"):
        provenance.ToolProvenanceSpec(kind="full_data", replay_stages=1)
    with pytest.raises(ValidationError, match="cannot define replay stages"):
        provenance.ToolProvenanceSpec(
            kind="script",
            start_label="Start",
            active_name="derived",
            replay_stages=[replay_stage],
        )

    with pytest.raises(ValidationError, match="must define `start_label`"):
        provenance.ToolProvenanceSpec(
            kind="file",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `seed_code`"):
        provenance.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            active_name="derived",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `active_name`"):
        provenance.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `file_load_source`"):
        provenance.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
        )
    with pytest.raises(ValidationError, match="must define `replay_call`"):
        provenance.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source.model_copy(update={"replay_call": None}),
        )
    with pytest.raises(ValidationError, match="cannot define operations"):
        provenance.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source,
            operations=[provenance.AverageOperation(dims=("x",))],
        )
    with pytest.raises(TypeError, match="Replay stages can only"):
        provenance.full_data().append_replay_stage(provenance.full_data())


def test_file_provenance_display_entries_keep_steps_after_stage_failure() -> None:
    spec = (
        _file_provenance_spec()
        .append_replay_stage(
            provenance.full_data(provenance.SelOperation(kwargs={"missing": 0}))
        )
        .append_replay_stage(provenance.full_data(provenance.SqueezeOperation()))
    )

    assert [entry.label for entry in spec.derivation_entries()] == [
        "Load data from file 'scan.h5'",
        "sel(missing=0)",
        "squeeze()",
    ]
    entries = spec.display_entries(parent_data=_base_data())
    assert [entry.label for entry in entries] == [
        "Load data from file 'scan.h5'",
        "sel(missing=0)",
        "squeeze()",
    ]
    code = spec.display_code(parent_data=_base_data())
    assert code is not None
    assert ".sel(missing=0)" in code
    assert ".squeeze()" in code


def test_tool_provenance_display_rows_expose_edit_and_replay_refs() -> None:
    file_spec = _file_provenance_spec().append_replay_stage(
        provenance.full_data(provenance.QSelAggregationOperation(dims=("x",)))
    )
    file_rows = file_spec.display_rows(parent_data=_base_data())

    assert file_rows[0].edit_ref == provenance._ProvenanceStepRef("file_load")
    assert file_rows[0].replay_ref == provenance._ProvenanceStepRef("file_load")
    assert file_rows[1].edit_ref == provenance._ProvenanceStepRef(
        "operation",
        operation_index=0,
        stage_index=0,
    )
    assert file_rows[1].replay_ref == file_rows[1].edit_ref

    live_spec = provenance.full_data(
        provenance.CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        )
    )
    live_rows = live_spec.display_rows(parent_data=_base_data(), scope="source")

    assert live_rows[0].scope == "source"
    assert live_rows[0].edit_ref is None
    assert live_rows[0].replay_ref == provenance._ProvenanceStepRef("start")
    assert live_rows[1].scope == "source"
    assert live_rows[1].edit_ref == provenance._ProvenanceStepRef(
        "operation",
        operation_index=0,
    )

    script_spec = provenance.script(
        provenance.ScriptCodeOperation(label="Run code", code="derived = data"),
        provenance.QSelAggregationOperation(dims=("x",)),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="source", label="Input"),),
    )
    script_rows = script_spec.display_rows()

    assert script_rows[1].edit_ref is None
    assert script_rows[1].replay_ref == provenance._ProvenanceStepRef(
        "script_input",
        script_input_index=0,
    )
    assert script_rows[2].edit_ref is None
    assert script_rows[2].replay_ref == provenance._ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    assert script_rows[3].edit_ref == provenance._ProvenanceStepRef(
        "operation",
        operation_index=1,
    )


def test_file_provenance_compose_fallbacks_and_replay_aliases() -> None:
    file_spec = _file_provenance_spec()

    assert provenance.replay_input_name(None) is None
    assert (
        provenance.script(start_label="Start", active_name="derived").derivation_code()
        is None
    )
    assert (
        provenance.script(
            start_label="Start",
            seed_code="derived = data",
            active_name="derived",
        ).display_code()
        is None
    )
    assert provenance.compose_full_provenance(None, None) is None
    local_replay = provenance.compose_full_provenance(
        None, provenance.full_data(provenance.AverageOperation(dims=("x",)))
    )
    assert local_replay is not None
    assert local_replay.kind == "script"

    assert (
        provenance.compose_full_provenance(file_spec, provenance.full_data())
        == file_spec
    )
    assert provenance._as_script_replay_spec(provenance.full_data()).kind == "script"

    script_local = provenance.script(
        provenance.ScriptCodeOperation(label="Offset", code="result = derived + 1"),
        start_label="Run generated code",
        seed_code="derived = data",
        active_name="result",
    )
    file_with_script = provenance.compose_full_provenance(file_spec, script_local)
    assert file_with_script is not None
    assert file_with_script.kind == "script"
    assert file_with_script.file_load_source == file_spec.file_load_source
    assert file_with_script.derivation_code() == (
        "import xarray\n\n"
        "derived = xarray.load_dataarray('scan.h5')\n"
        "result = derived + 1"
    )

    watched_parent = provenance.script(
        start_label="Start from watched variable 'watched_data'",
        seed_code="derived = watched_data",
        active_name="derived",
    )
    default_seed_local = provenance.script(
        provenance.ScriptCodeOperation(label="Mean", code="result = derived.mean()"),
        start_label="Use current parent output",
        seed_code="derived = data",
        active_name="result",
    )
    watched_composed = provenance.compose_full_provenance(
        watched_parent, default_seed_local
    )
    assert watched_composed is not None
    assert watched_composed.derivation_code() == (
        "derived = watched_data\nresult = derived.mean()"
    )
    assert watched_composed.display_code() == "result = watched_data.mean()"

    result_parent = provenance.script(
        provenance.ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = data + 1",
        ),
        start_label="Start",
        active_name="result",
    )
    no_seed_local = provenance.script(
        provenance.ScriptCodeOperation(label="Mean", code="result = derived.mean()"),
        start_label="Use parent result",
        active_name="derived",
    )
    result_composed = provenance.compose_full_provenance(result_parent, no_seed_local)
    assert result_composed is not None
    assert result_composed.derivation_code() == (
        "result = data + 1\nderived = result\nresult = derived.mean()"
    )

    promoted = provenance.mark_promoted_1d_source(_base_data().copy(deep=False))
    assert (
        provenance.compose_display_provenance(
            watched_parent,
            provenance.selection(
                provenance.IselOperation(), provenance.SortCoordOrderOperation()
            ),
            parent_data=promoted,
        )
        is not None
    )
    assert provenance.compose_display_provenance(
        watched_parent, None
    ) == provenance.to_replay_provenance_spec(watched_parent)


def test_script_provenance_supports_named_console_inputs() -> None:
    left = provenance.script(
        provenance.ScriptCodeOperation(
            label="Offset left input",
            code="data_0 = data_0 + 1.0",
        ),
        start_label="Load left",
        seed_code="data_0 = xr.DataArray([1.0, 2.0], dims=['x'])",
        active_name="data_0",
    )
    right = provenance.script(
        start_label="Load right",
        seed_code="data_1 = xr.DataArray([0.5, 1.5], dims=['x'])",
        active_name="data_1",
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Subtract console inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="ImageTool 0",
                node_uid="left",
                provenance_spec=left,
            ),
            provenance.ScriptInput(
                name="data_1",
                label="ImageTool 1",
                node_uid="right",
                provenance_spec=right,
            ),
        ),
    )

    reparsed = provenance.parse_tool_provenance_spec(spec.model_dump(mode="json"))

    assert reparsed == spec
    assert [entry.label for entry in spec.display_entries()] == [
        "Run ImageTool manager console code",
        "Use data_0 from ImageTool 0",
        "Use data_1 from ImageTool 1",
        "Subtract console inputs",
    ]
    rows = spec.display_rows()
    assert rows[1].children[0].entry.label == "Load left"
    assert rows[1].children[0].replay_ref == provenance._ProvenanceStepRef("start")
    assert rows[1].children[0].script_input_path == (0,)
    assert rows[1].children[1].entry.label == "Offset left input"
    assert rows[1].children[1].edit_ref is None
    assert rows[1].children[1].replay_ref == provenance._ProvenanceStepRef(
        "operation", operation_index=0
    )
    assert rows[1].children[1].script_input_path == (0,)
    assert rows[2].children[0].entry.label == "Load right"
    assert rows[2].children[0].script_input_path == (1,)
    assert rows[3].edit_ref is None
    assert rows[3].replay_ref == provenance._ProvenanceStepRef(
        "operation",
        operation_index=0,
    )
    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code, {})
    xr.testing.assert_identical(
        namespace["derived"],
        xr.DataArray([1.5, 1.5], dims=["x"]),
    )


def test_script_input_code_reuses_shared_file_replay_prefix(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "polarization.nc"
    source = xr.DataArray(
        np.arange(12.0).reshape(2, 2, 3),
        dims=("pol", "energy", "k"),
        coords={"pol": ["LH", "LV"], "energy": [0.0, 1.0], "k": [0, 1, 2]},
    )
    source.to_netcdf(path)
    file_spec = provenance.file_load(
        start_label="Load both polarizations",
        seed_code=f"import xarray\n\nderived = xarray.load_dataarray({str(path)!r})",
        file_load_source=provenance.FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )
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
            label="Subtract polarizations",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="ImageTool 0: LH",
                provenance_spec=left_spec,
            ),
            provenance.ScriptInput(
                name="data_1",
                label="ImageTool 1: LV",
                provenance_spec=right_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())

    assert code.count("xarray.load_dataarray") == 1
    assert code.count(".qsel.mean") == 1
    assert "data_0 =" not in code
    assert "data_1 =" not in code
    assert "restore_nonuniform_dims" not in code
    namespace = _exec_generated_code(code, {})
    expected = left_stage.apply(shared_stage.apply(source)) - right_stage.apply(
        shared_stage.apply(source)
    )
    xr.testing.assert_identical(namespace["derived"], expected)


def test_script_input_code_keeps_distinct_structured_replay_nodes() -> None:
    first = provenance.file_load(
        start_label="Load first",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(
            "scan.h5",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )
    second = provenance.file_load(
        start_label="Load second",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(
            "scan.h5",
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=1,
            ),
        ),
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Subtract inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0", label="ImageTool 0", provenance_spec=first
            ),
            provenance.ScriptInput(
                name="data_1",
                label="ImageTool 1",
                provenance_spec=second,
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count("xarray.load_dataarray") == 2


def test_script_input_dependency_refs_recurse_and_rebase() -> None:
    left_snapshot_id = "left-snapshot"
    right_snapshot_id = "right-snapshot"
    extra_snapshot_id = "extra-snapshot"
    nested = provenance.script(
        provenance.ScriptCodeOperation(
            label="Subtract console inputs",
            code="diff = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="diff",
        script_inputs=(
            provenance.ScriptInput(
                name="data_0",
                label="ImageTool 0",
                node_uid="old-left",
                node_snapshot_token=left_snapshot_id,
            ),
            provenance.ScriptInput(
                name="data_1",
                label="ImageTool 1",
                node_uid="old-right",
                node_snapshot_token=right_snapshot_id,
            ),
        ),
    )
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Add nested input",
            code="derived = diff + data_2",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(
                name="diff",
                label="console variable 'diff'",
                provenance_spec=nested,
            ),
            provenance.ScriptInput(
                name="data_2",
                label="ImageTool 2",
                node_uid="old-extra",
                node_snapshot_token=extra_snapshot_id,
            ),
        ),
    )

    refs = provenance.script_input_dependency_refs(spec)
    assert [(ref.name, ref.node_uid, ref.node_snapshot_token) for ref in refs] == [
        ("data_0", "old-left", left_snapshot_id),
        ("data_1", "old-right", right_snapshot_id),
        ("data_2", "old-extra", extra_snapshot_id),
    ]

    rebased = provenance.rebase_script_input_node_uids(
        spec,
        {
            "old-left": "new-left",
            "old-right": "new-right",
            "old-extra": "new-extra",
        },
    )

    assert [source.name for source in rebased.script_inputs] == ["diff", "data_2"]
    assert rebased.script_inputs[1].node_uid == "new-extra"
    assert rebased.script_inputs[1].label == "ImageTool 2"
    assert typing.cast("str", rebased.operations[-1].derivation_entry().code) == (
        "derived = diff + data_2"
    )
    assert [
        (ref.name, ref.node_uid, ref.node_snapshot_token)
        for ref in provenance.script_input_dependency_refs(rebased)
    ] == [
        ("data_0", "new-left", left_snapshot_id),
        ("data_1", "new-right", right_snapshot_id),
        ("data_2", "new-extra", extra_snapshot_id),
    ]
    assert provenance.script_input_dependency_refs(None) == ()
    assert provenance.rebase_script_input_node_uids(spec, {}) is spec
    with pytest.raises(TypeError, match="Expected provenance spec"):
        provenance.rebase_script_input_node_uids(None, {})


def test_low_level_provenance_replay_helper_branches() -> None:

    assert not provenance._is_whole_array_rename_entry(
        provenance.DerivationEntry("rename", "derived =")
    )
    assert not provenance._is_whole_array_rename_entry(
        provenance.DerivationEntry("rename", "derived = derived.rename('a', 'b')")
    )
    assert not provenance._is_whole_array_rename_entry(
        provenance.DerivationEntry(
            "rename", "derived = derived.rename(mapping={'x': 'y'})"
        )
    )
    assert provenance._is_whole_array_rename_entry(
        provenance.DerivationEntry(
            "rename",
            "derived = derived.rename(new_name_or_name_dict=None)",
        )
    )

    assert provenance._provenance_value_code({"left": (1,)}) == "{'left': (1,)}"
    with pytest.raises(TypeError, match="Cannot generate replay code"):
        provenance._provenance_value_code(object())
    with pytest.raises(TypeError, match="hashable fields"):
        provenance._normalize_provenance_hashable(object())
    assert provenance._encode_provenance_hashable(("x", 1)) == {
        provenance._TUPLE_MARKER: ["x", 1]
    }
    with pytest.raises(ValueError, match="Expected 2 items"):
        provenance._ensure_float_tuple([1], expected_len=2)
    for value in (1, "abc"):
        with pytest.raises(TypeError, match="array-like sequence"):
            provenance._coerce_float_sequence(value)
    assert provenance._format_selection_step("sel", {}) == "derived = derived.sel()"
    assert provenance._validate_active_name(None) is None
    for value, error in ((1, TypeError), ("class", ValueError)):
        with pytest.raises(error):
            provenance._validate_active_name(value)

    code = """
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
    statements = ast.parse(code).body
    assert provenance._statement_load_count(statements[0], "data_0") == 1
    assert provenance._statement_load_count(statements[0], "data_1") == 1
    assert provenance._statement_load_count(statements[0], "data_2") == 1
    assert (
        provenance._statement_store_count(
            statements[0], "helper", count_definition_names=True
        )
        == 1
    )
    assert provenance._statement_load_count(statements[1], "data_3") == 1
    assert provenance._statement_load_count(statements[2], "data_4") == 1
    assert provenance._statement_load_count(statements[3], "class_decorator") == 1
    assert provenance._statement_load_count(statements[3], "data_5") == 1
    assert (
        provenance._statement_store_count(
            statements[3], "Child", count_definition_names=True
        )
        == 1
    )
    assert "source_data" in provenance._replace_code_identifiers(
        code,
        {"data_0": "source_data"},
    )
    rebased = provenance.rebase_default_replay_input(
        """
def normalize(value=data) -> data:
    return value

lambda_value = lambda value=data: value

class Child(data):
    pass

derived = data
""",
        "source_data",
    )
    assert "source_data" in rebased
    assert (
        provenance.rebase_default_replay_input("derived = data", "not valid python(")
        == "derived = data"
    )
    assert provenance._simplify_display_code("derived =") == "derived ="
    assert provenance._simplify_display_code("") == ""
    assert provenance._simplify_display_code("for item in data:\n    pass") == (
        "for item in data:\n    pass"
    )
    assert provenance._simplify_display_code("left = right = data\nresult = left") == (
        "left = right = data\nresult = left"
    )
    assert (
        provenance._simplify_display_code("tmp = data\nresult = tmp") == "result = data"
    )
    assert (
        provenance._simplify_display_code(
            "tmp = data\nother = 1\nresult = tmp",
            inline_targets={"missing"},
        )
        == "tmp = data\nother = 1\nresult = tmp"
    )

    wrapped = staticmethod(lambda: None)
    assert any(
        path.endswith(".<lambda>") for path in provenance._callable_paths(wrapped)
    )
    assert (
        provenance._callable_paths(types.SimpleNamespace(__module__=1, __name__=2))
        == set()
    )

    with pytest.raises(ValidationError):
        provenance.ScriptInput(name=None, label="Input")
    with pytest.raises(TypeError, match="script input label"):
        provenance.ScriptInput(name="data_0", label=1)
    with pytest.raises(ValidationError):
        provenance.ScriptInput(name="data_0", label="   ")
    with pytest.raises(ValidationError):
        provenance.ScriptInput(name="data_0", label="Input", node_snapshot_token="")
    with pytest.raises(TypeError, match="script input provenance"):
        provenance.ScriptInput(name="data_0", label="Input", provenance_spec=object())
    with pytest.raises(TypeError, match="Serialized replay stages"):
        provenance.ToolProvenanceSpec(kind="full_data", replay_stages="bad")
    with pytest.raises(TypeError, match="Serialized script inputs"):
        provenance.ToolProvenanceSpec(
            kind="script",
            start_label="Run",
            active_name="derived",
            script_inputs="bad",
        )

    assert (
        provenance.script(start_label="Run", active_name="derived")._script_graph_code(
            display=True
        )
        is None
    )
    assert (
        provenance.full_data(
            provenance.ScriptCodeOperation(label="Opaque", code=None, copyable=False)
        ).derivation_code()
        is None
    )
    with pytest.raises(ValueError, match="not valid Python"):
        provenance._validate_script_replay_code("derived =")
    for code_snippet, message in (
        ("derived = __name__", "dunder names"),
        ("derived = data.__class__", "dunder attributes"),
        ("derived = open('path')", "cannot call"),
    ):
        with pytest.raises(ValueError, match=message):
            provenance._validate_script_replay_code(code_snippet)


def test_script_input_label_is_single_line_display_text() -> None:

    script_input = provenance.ScriptInput(
        name="data_0",
        label="  ImageTool 0:\n\n  processed data  ",
    )

    assert script_input.label == "ImageTool 0: processed data"
    with pytest.raises(ValidationError):
        provenance.ScriptInput(name="data_0", label="\n  \t")


def test_replay_script_provenance_uses_resolved_inputs_without_mutating() -> None:
    left = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0, 1]})
    right = xr.DataArray([0.5, 1.5], dims=("x",), coords={"x": [0, 1]})
    spec = provenance.script(
        provenance.ScriptCodeOperation(
            label="Mutate local input",
            code="data_0[0] = 10.0\nderived = data_0 - data_1",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(name="data_0", label="ImageTool 0"),
            provenance.ScriptInput(name="data_1", label="ImageTool 1"),
        ),
    )

    assert provenance.script_provenance_replayable(spec)
    result = provenance.replay_script_provenance(
        spec,
        {"data_0": left, "data_1": right},
    )

    xr.testing.assert_identical(
        result,
        xr.DataArray([9.5, 0.5], dims=("x",), coords={"x": [0, 1]}),
    )
    xr.testing.assert_identical(
        left,
        xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0, 1]}),
    )


def test_replay_script_provenance_rejects_unsupported_or_incomplete_code() -> None:
    data = xr.DataArray([1.0], dims=("x",))
    unsupported = provenance.script(
        provenance.ScriptCodeOperation(
            label="Unsupported",
            code="import os\nderived = data_0",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    incomplete = provenance.script(
        provenance.ScriptCodeOperation(label="Incomplete", code=None),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    missing_seed = provenance.script(
        provenance.AverageOperation(dims=("x",)),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    active_input = provenance.script(
        provenance.AverageOperation(dims=("x",)),
        start_label="Run script",
        active_name="data_0",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    rename_input = provenance.script(
        provenance.RenameOperation(name="renamed"),
        start_label="Run script",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    external_active_input = provenance.script(
        provenance.AverageOperation(dims=("x",)),
        start_label="Run script",
        active_name="data_0",
    )
    function_local_active = provenance.script(
        provenance.ScriptCodeOperation(
            label="Helper",
            code="def helper(data):\n    derived = data\n",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    missing_helper = provenance.script(
        provenance.ScriptCodeOperation(
            label="Missing helper",
            code="derived = helper(data_0)",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    missing_helper_global = provenance.script(
        provenance.ScriptCodeOperation(
            label="Missing helper global",
            code=(
                "def helper(data):\n    return data + scale\n\nderived = helper(data_0)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    captured_helper_global = provenance.script(
        provenance.ScriptCodeOperation(
            label="Captured helper global",
            code=(
                "scale = 2.0\n"
                "\n"
                "def helper(data):\n"
                "    return data + scale\n"
                "\n"
                "derived = helper(data_0)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    late_helper_global = provenance.script(
        provenance.ScriptCodeOperation(
            label="Late helper global",
            code=(
                "def helper(data):\n"
                "    return data + scale\n"
                "\n"
                "derived = helper(data_0)\n"
                "scale = 2.0"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    redefined_helper = provenance.script(
        provenance.ScriptCodeOperation(
            label="Redefined helper",
            code=(
                "def helper(data):\n"
                "    return data + missing\n"
                "\n"
                "def helper(data):\n"
                "    return data\n"
                "\n"
                "derived = helper(data_0)"
            ),
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )

    assert not provenance.script_provenance_replayable(unsupported)
    assert not provenance.script_provenance_replayable(incomplete)
    assert not provenance.script_provenance_replayable(missing_seed)
    assert not provenance.script_provenance_replayable(function_local_active)
    assert not provenance.script_provenance_replayable(missing_helper)
    assert not provenance.script_provenance_replayable(missing_helper_global)
    assert not provenance.script_provenance_replayable(late_helper_global)
    assert provenance.script_provenance_replayable(active_input)
    assert provenance.script_provenance_replayable(rename_input)
    assert provenance.script_provenance_replayable(captured_helper_global)
    assert provenance.script_provenance_replayable(redefined_helper)
    with pytest.raises(TypeError, match="unsupported Import"):
        provenance.replay_script_provenance(unsupported, {"data_0": data})
    with pytest.raises(ValueError, match="non-replayable"):
        provenance.replay_script_provenance(incomplete, {"data_0": data})
    with pytest.raises(TypeError, match="no replay code"):
        provenance.replay_script_provenance(missing_seed, {"data_0": data})
    with pytest.raises(TypeError, match="no replay code"):
        provenance.replay_script_provenance(function_local_active, {"data_0": data})
    with pytest.raises(TypeError, match="unresolved name 'helper'"):
        provenance.replay_script_provenance(missing_helper, {"data_0": data})
    with pytest.raises(TypeError, match="unresolved name 'scale'"):
        provenance.replay_script_provenance(missing_helper_global, {"data_0": data})
    with pytest.raises(TypeError, match="unresolved name 'scale'"):
        provenance.replay_script_provenance(late_helper_global, {"data_0": data})
    xr.testing.assert_identical(
        provenance.replay_script_provenance(active_input, {"data_0": data}),
        data.qsel.average("x"),
    )
    xr.testing.assert_identical(
        provenance.replay_script_provenance(rename_input, {"data_0": data}),
        data.rename("renamed"),
    )
    xr.testing.assert_identical(
        provenance.replay_script_provenance(external_active_input, {"data_0": data}),
        data.qsel.average("x"),
    )
    xr.testing.assert_identical(
        provenance.replay_script_provenance(captured_helper_global, {"data_0": data}),
        data + 2.0,
    )
    xr.testing.assert_identical(
        provenance.replay_script_provenance(redefined_helper, {"data_0": data}),
        data,
    )


def test_replay_script_provenance_accepts_console_module_aliases() -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
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
        script_inputs=(provenance.ScriptInput(name="data_0", label="ImageTool 0"),),
    )

    assert provenance.script_provenance_replayable(spec)
    xr.testing.assert_identical(
        provenance.replay_script_provenance(spec, {"data_0": data}),
        erlab.analysis.transform.rotate(data, 0.0, axes=("x", "y"), reshape=False),
    )


def test_console_pattern_expands_named_xarray_mapping_arguments() -> None:
    data = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0.0, 1.0]})

    qsel_operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            accessor_path=("qsel",),
            kwargs={"indexers": {"x": 1.0}},
            display_code="data.qsel(indexers={'x': 1.0})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    isel_operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            dataarray_method="isel",
            kwargs={"indexers": {"x": 1}},
            display_code="data.isel(indexers={'x': 1})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    sel_operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            dataarray_method="sel",
            kwargs={"indexers": {"x": 1.0}},
            display_code="data.sel(indexers={'x': 1.0})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    interp_operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            dataarray_method="interp",
            kwargs={"coords": {"x": [0.25, 0.75]}},
            display_code="data.interp(coords={'x': [0.25, 0.75]})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    rename_operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            dataarray_method="rename",
            kwargs={"new_name_or_name_dict": {"x": "energy"}},
            display_code="data.rename(new_name_or_name_dict={'x': 'energy'})",
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    multidim_coord_operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            dataarray_method="assign_coords",
            kwargs={"foo": (("x", "y"), np.ones((2, 2)))},
            display_code="data.assign_coords(foo=(('x', 'y'), values))",
            has_extra_tracked_inputs=False,
            receiver_data=xr.DataArray(np.ones((2, 2)), dims=("x", "y")),
        )
    )

    assert qsel_operation == provenance.QSelOperation(kwargs={"x": 1.0})
    assert isel_operation == provenance.IselOperation(kwargs={"x": 1})
    assert sel_operation == provenance.SelOperation(kwargs={"x": 1.0})
    assert interp_operation == provenance.InterpolationOperation(
        dim="x", values=[0.25, 0.75]
    )
    assert rename_operation == provenance.RenameDimsCoordsOperation(
        mapping={"x": "energy"}
    )
    assert multidim_coord_operation is None
    assert isinstance(qsel_operation, provenance.QSelOperation)
    assert isinstance(isel_operation, provenance.IselOperation)
    assert isinstance(sel_operation, provenance.SelOperation)
    assert isinstance(interp_operation, provenance.InterpolationOperation)
    assert isinstance(rename_operation, provenance.RenameDimsCoordsOperation)
    xr.testing.assert_identical(
        qsel_operation.apply(data, parent_data=data),
        data.qsel(indexers={"x": 1.0}),
    )
    xr.testing.assert_identical(
        isel_operation.apply(data, parent_data=data),
        data.isel(indexers={"x": 1}),
    )
    xr.testing.assert_identical(
        sel_operation.apply(data, parent_data=data),
        data.sel(indexers={"x": 1.0}),
    )
    xr.testing.assert_identical(
        interp_operation.apply(data, parent_data=data),
        data.interp(coords={"x": [0.25, 0.75]}),
    )
    xr.testing.assert_identical(
        rename_operation.apply(data, parent_data=data),
        data.rename(new_name_or_name_dict={"x": "energy"}),
    )


def test_console_pattern_matches_public_parameter_aliases() -> None:
    edge_fit = xr.Dataset({"center": ("x", [0.0, 1.0])})

    operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            func=erlab.analysis.gold.correct_with_edge,
            kwargs={"modelresult": edge_fit, "shift_coords": False},
            display_code="era.gold.correct_with_edge(data, modelresult=edge_fit)",
            has_extra_tracked_inputs=False,
        )
    )

    assert isinstance(operation, provenance.CorrectWithEdgeOperation)
    assert not operation.shift_coords
    xr.testing.assert_identical(operation.decoded_edge_fit, edge_fit)


def test_console_pattern_matches_new_replayable_operations() -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "eV"),
        coords={"x": [0.0, 1.0], "eV": [0.0, 1.0, 2.0]},
    )

    aggregate_operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            accessor_path=("qsel", "sum"),
            args=("x",),
            display_code='data.qsel.sum("x")',
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )
    leading_edge_operation = provenance.operation_from_console_call(
        provenance.ConsoleCall(
            func=erlab.analysis.interpolate.leading_edge,
            kwargs={"fraction": 0.25, "dim": "eV", "direction": "negative"},
            display_code=(
                "era.interpolate.leading_edge("
                "data, fraction=0.25, dim='eV', direction='negative')"
            ),
            has_extra_tracked_inputs=False,
            receiver_data=data,
        )
    )

    assert aggregate_operation == provenance.QSelAggregationOperation(
        dims=("x",), func="sum"
    )
    assert leading_edge_operation == provenance.LeadingEdgeOperation(
        fraction=0.25,
        dim="eV",
        direction="negative",
    )
    xr.testing.assert_identical(
        aggregate_operation.apply(data, parent_data=data),
        data.qsel.sum("x"),
    )


def test_sortby_operation_apply_code_and_console_calls() -> None:
    tuple_key = ("beta", 0)
    data = xr.DataArray(
        np.arange(12.0).reshape(4, 3),
        dims=("x", "y"),
        coords={
            "x": [2.0, 1.0, 2.0, 0.0],
            "y": [0.0, 1.0, 2.0],
            "sample temp": ("x", [20.0, 10.0, 15.0, 30.0]),
        },
        name="scan",
    )
    tuple_key_data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={
            "x": [0.0, 1.0, 2.0, 3.0],
            tuple_key: ("x", [20.0, 10.0, 15.0, 30.0]),
        },
        name="tuple_coord_scan",
    )

    single = provenance.SortByOperation(variables=("x",))
    multi = provenance.SortByOperation(
        variables=("x", "sample temp"),
        ascending=False,
    )
    non_identifier = provenance.SortByOperation(variables=("sample temp",))
    tuple_key_operation = provenance.SortByOperation(variables=(tuple_key,))

    assert provenance.SortByOperation(variables="x") == single
    with pytest.raises(TypeError, match="sortby variables must be coordinate names"):
        provenance.SortByOperation(variables=lambda darr: darr.x)
    assert multi.derivation_label().startswith("Sort By(")
    for operation in (single, multi, tuple_key_operation):
        assert (
            provenance.parse_tool_provenance_operation(
                operation.model_dump(mode="json")
            )
            == operation
        )

    xr.testing.assert_identical(single.apply(data, parent_data=data), data.sortby("x"))
    xr.testing.assert_identical(
        multi.apply(data, parent_data=data),
        data.sortby(["x", "sample temp"], ascending=False),
    )
    xr.testing.assert_identical(
        non_identifier.apply(data, parent_data=data),
        data.sortby("sample temp"),
    )
    xr.testing.assert_identical(
        tuple_key_operation.apply(tuple_key_data, parent_data=tuple_key_data),
        tuple_key_data.sortby(tuple_key),
    )

    code = f"derived = {multi.expression_code('data')}"
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"],
        data.sortby(["x", "sample temp"], ascending=False),
    )
    code = f"derived = {tuple_key_operation.expression_code('data')}"
    namespace = _exec_generated_code(code, {"data": tuple_key_data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], tuple_key_data.sortby(tuple_key))

    assert (
        provenance.operation_from_console_call(
            provenance.ConsoleCall(
                dataarray_method="sortby",
                args=("x",),
                display_code='data.sortby("x")',
                has_extra_tracked_inputs=False,
                receiver_data=data,
            )
        )
        == single
    )
    assert (
        provenance.operation_from_console_call(
            provenance.ConsoleCall(
                dataarray_method="sortby",
                kwargs={"variables": ["x", "sample temp"], "ascending": False},
                display_code=(
                    'data.sortby(variables=["x", "sample temp"], ascending=False)'
                ),
                has_extra_tracked_inputs=False,
                receiver_data=data,
            )
        )
        == multi
    )
    assert (
        provenance.operation_from_console_call(
            provenance.ConsoleCall(
                dataarray_method="sortby",
                args=(tuple_key,),
                display_code="data.sortby(('beta', 0))",
                has_extra_tracked_inputs=False,
                receiver_data=tuple_key_data,
            )
        )
        == tuple_key_operation
    )

    for args, kwargs in (
        ((), {"variables": "x", "ascending": "false"}),
        (("x",), {"variables": "y"}),
        ((), {}),
    ):
        assert (
            provenance.SortByOperation.from_console_call(
                provenance.ConsoleCall(
                    dataarray_method="sortby",
                    args=args,
                    kwargs=kwargs,
                    display_code="data.sortby(...)",
                    has_extra_tracked_inputs=False,
                    receiver_data=data,
                )
            )
            is None
        )

    for variables in (lambda darr: darr.x, data.x, [lambda darr: darr.x], [data.x], []):
        assert (
            provenance.SortByOperation.from_console_call(
                provenance.ConsoleCall(
                    dataarray_method="sortby",
                    args=(variables,),
                    display_code="data.sortby(...)",
                    has_extra_tracked_inputs=False,
                    receiver_data=data,
                )
            )
            is None
        )
    assert (
        provenance.SortByOperation.from_console_call(
            provenance.ConsoleCall(
                dataarray_method="sortby",
                args=("x",),
                display_code='data.sortby("x")',
                has_extra_tracked_inputs=True,
                receiver_data=data,
            )
        )
        is None
    )


def test_console_pattern_rejects_ambiguous_calls_and_expands_defaults() -> None:

    assert provenance._console_values_equal(np.nan, np.nan)
    assert provenance._console_mapping_values(
        (None,), {"coords": None, "x": 1}, mapping_kwargs=("coords",)
    ) == {"x": 1}
    assert provenance._console_mapping_values(
        ({"x": 1},), {"coords": {"y": 2}}, mapping_kwargs=("coords",)
    ) == {"x": 1, "y": 2}
    assert provenance._console_mapping_values((1, 2), {}) is None
    assert provenance._console_mapping_values((1,), {}) is None
    assert (
        provenance._console_mapping_values(
            (), {"coords": 1}, mapping_kwargs=("coords",)
        )
        is None
    )

    pattern = provenance.ConsoleOperationPattern(
        target="builtins.abs",
        fields=("value",),
        field_aliases={"old_value": "value"},
        defaults={"scale": 1},
        ignored_defaults={"drop": False},
    )

    assert (
        pattern.match(
            provenance.ConsoleCall(
                display_code="abs(3)",
                has_extra_tracked_inputs=True,
            )
        )
        is None
    )
    assert (
        pattern.match(
            provenance.ConsoleCall(
                display_code="abs(3)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        pattern.match(
            provenance.ConsoleCall(
                func=len,
                args=(3,),
                display_code="len(3)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        pattern.match(
            provenance.ConsoleCall(
                func=abs,
                kwargs={"old_value": 3, "value": 4},
                display_code="abs(old_value=3, value=4)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        pattern.match(
            provenance.ConsoleCall(
                func=abs,
                args=(3,),
                kwargs={"drop": True},
                display_code="abs(3, drop=True)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert pattern.match(
        provenance.ConsoleCall(
            func=abs,
            args=(3,),
            kwargs={"scale": 2, "drop": False},
            display_code="abs(3, scale=2, drop=False)",
            has_extra_tracked_inputs=False,
        )
    ) == {"value": 3, "scale": 2}

    assert (
        provenance.ConsoleOperationPattern(dataarray_method="isel").match(
            provenance.ConsoleCall(
                dataarray_method="sel",
                display_code="data.sel()",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        provenance.ConsoleOperationPattern().match(
            provenance.ConsoleCall(
                dataarray_method="isel",
                display_code="data.isel()",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        provenance.ConsoleOperationPattern(accessor_path=("qsel",)).match(
            provenance.ConsoleCall(
                accessor_path=("qsel", "mean"),
                display_code="data.qsel.mean()",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        provenance.ConsoleOperationPattern(fields=("required",)).match(
            provenance.ConsoleCall(
                display_code="data.call()",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        provenance.ConsoleOperationPattern().match(
            provenance.ConsoleCall(
                kwargs={"unexpected": 1},
                display_code="data.call(unexpected=1)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )
    assert (
        provenance.ConsoleOperationPattern(kwargs_field="mapping").match(
            provenance.ConsoleCall(
                args=(1,),
                display_code="data.call(1)",
                has_extra_tracked_inputs=False,
            )
        )
        is None
    )


def test_console_operations_match_branch_specific_calls() -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )

    def call(**kwargs: typing.Any) -> typing.Any:
        kwargs.setdefault("display_code", "data.call()")
        kwargs.setdefault("has_extra_tracked_inputs", False)
        kwargs.setdefault("receiver_data", data)
        return provenance.ConsoleCall(**kwargs)

    assert provenance.TransposeOperation.from_console_call(
        call(
            dataarray_method="transpose",
            args=("y", "x"),
            kwargs={"transpose_coords": True, "missing_dims": "raise"},
        )
    ) == provenance.TransposeOperation(dims=("y", "x"))
    assert (
        provenance.TransposeOperation.from_console_call(
            call(dataarray_method="transpose", kwargs={"transpose_coords": False})
        )
        is None
    )

    assert (
        provenance.SqueezeOperation.from_console_call(
            call(dataarray_method="squeeze", kwargs={"dim": None, "axis": None})
        )
        == provenance.SqueezeOperation()
    )
    assert (
        provenance.SqueezeOperation.from_console_call(
            call(dataarray_method="squeeze", kwargs={"drop": True})
        )
        is None
    )

    assert provenance.RenameOperation.from_console_call(
        call(dataarray_method="rename", args=("renamed",))
    ) == provenance.RenameOperation(name="renamed")
    assert (
        provenance.RenameOperation.from_console_call(
            call(dataarray_method="rename", args=("renamed",), kwargs={"bad": 1})
        )
        is None
    )

    assert provenance.AverageOperation.from_console_call(
        call(accessor_path=("qsel", "average"), kwargs={"dim": "x"})
    ) == provenance.AverageOperation(dims=("x",))
    assert (
        provenance.AverageOperation.from_console_call(
            call(accessor_path=("qsel", "average"), args=("x",), kwargs={"dim": "x"})
        )
        is None
    )

    assert provenance.QSelAggregationOperation.from_console_call(
        call(accessor_path=("qsel", "mean"), kwargs={"dim": ("x", "y")})
    ) == provenance.QSelAggregationOperation(dims=("x", "y"), func="mean")
    assert (
        provenance.QSelAggregationOperation.from_console_call(
            call(accessor_path=("qsel", "median"), kwargs={"dim": "x"})
        )
        is None
    )

    assert provenance.InterpolationOperation.from_console_call(
        call(
            dataarray_method="interp",
            args=({"x": [0.25, 0.75]},),
            kwargs={"method": "nearest", "assume_sorted": False, "kwargs": None},
        )
    ) == provenance.InterpolationOperation(
        dim="x", values=[0.25, 0.75], method="nearest"
    )
    for bad_call in (
        call(dataarray_method="interp", kwargs={"method": "cubic", "x": [0.5]}),
        call(dataarray_method="interp", kwargs={"x": [0.5], "assume_sorted": True}),
        call(dataarray_method="interp", args=({"x": [0.5], "y": [0.5]},)),
        call(dataarray_method="interp", kwargs={"x": [[0.0, 1.0]]}),
    ):
        assert provenance.InterpolationOperation.from_console_call(bad_call) is None

    assert provenance.CoarsenOperation.from_console_call(
        call(
            dataarray_method="coarsen",
            args=({"x": 2},),
            kwargs={"_reducer": "mean", "boundary": "trim"},
        )
    ) == provenance.CoarsenOperation(
        dim={"x": 2},
        boundary="trim",
        side="left",
        coord_func="mean",
        reducer="mean",
    )
    for bad_call in (
        call(dataarray_method="coarsen", kwargs={"x": 2}),
        call(dataarray_method="coarsen", args=(2,), kwargs={"_reducer": "mean"}),
        call(
            dataarray_method="coarsen",
            args=({"x": "bad"},),
            kwargs={"_reducer": "mean"},
        ),
        call(
            dataarray_method="coarsen",
            args=({"x": 2},),
            kwargs={"_reducer": "mean", "extra": object()},
        ),
    ):
        assert provenance.CoarsenOperation.from_console_call(bad_call) is None

    assert provenance.ThinOperation.from_console_call(
        call(dataarray_method="thin", args=(2,))
    ) == provenance.ThinOperation(mode="global", factor=2)
    assert provenance.ThinOperation.from_console_call(
        call(dataarray_method="thin", args=(None,), kwargs={"x": 2})
    ) == provenance.ThinOperation(mode="per_dim", factors={"x": 2})
    assert provenance.ThinOperation.from_console_call(
        call(dataarray_method="thin", args=({"x": 2},), kwargs={"y": 2})
    ) == provenance.ThinOperation(mode="per_dim", factors={"x": 2, "y": 2})
    for bad_call in (
        call(dataarray_method="thin", args=(1, 2)),
        call(dataarray_method="thin", args=(1,), kwargs={"x": 2}),
    ):
        assert provenance.ThinOperation.from_console_call(bad_call) is None

    assert provenance.RenameDimsCoordsOperation.from_console_call(
        call(dataarray_method="rename", kwargs={"new_name_or_name_dict": {"x": "kx"}})
    ) == provenance.RenameDimsCoordsOperation(mapping={"x": "kx"})
    assert (
        provenance.RenameDimsCoordsOperation.from_console_call(
            call(dataarray_method="rename", args=(None,))
        )
        is None
    )

    assigned = provenance.AssignCoordsOperation.from_console_call(
        call(dataarray_method="assign_coords", kwargs={"x": np.array([2.0, 3.0])})
    )
    assert isinstance(assigned, provenance.AssignCoordsOperation)
    np.testing.assert_allclose(assigned.decoded_values, [2.0, 3.0])
    assert provenance.AssignScalarCoordOperation.from_console_call(
        call(dataarray_method="assign_coords", kwargs={"temperature": 21.5})
    ) == provenance.AssignScalarCoordOperation(coord_name="temperature", value=21.5)
    assert provenance.AssignCoord1DOperation.from_console_call(
        call(
            dataarray_method="assign_coords",
            kwargs={"temperature": ("x", [100, 101])},
        )
    ) == provenance.AssignCoord1DOperation(
        coord_name="temperature", dim="x", values=[100, 101]
    )
    for bad_call in (
        call(dataarray_method="assign_coords", args=(1, 2)),
        call(dataarray_method="assign_coords", kwargs={"z": np.array([1.0, 2.0])}),
        call(
            dataarray_method="assign_coords",
            kwargs={"temperature": (["x"], [1, 2])},
        ),
        call(
            dataarray_method="assign_coords",
            kwargs={"temperature": ("x", np.ones((2, 2)))},
        ),
    ):
        assert provenance.AssignCoordsOperation.from_console_call(bad_call) is None
        assert provenance.AssignCoord1DOperation.from_console_call(bad_call) is None
    assert (
        provenance.AssignCoordsOperation.from_console_call(
            call(
                dataarray_method="assign_coords",
                kwargs={"x": ("x", [2.0, 3.0])},
            )
        )
        is None
    )


def test_file_replay_parses_supported_inputs_and_errors(tmp_path, monkeypatch) -> None:
    image = xr.DataArray(
        np.arange(6).reshape((2, 3)),
        dims=("row", "col"),
        name="image",
    )
    line = xr.DataArray(np.arange(3), dims=("energy",), name="line")
    five_dim = xr.DataArray(
        np.zeros((1, 2, 3, 1, 4)),
        dims=("a", "b", "c", "d", "e"),
        name="five_dim",
    )

    parsed_array = provenance._parse_replay_input(np.arange(6).reshape((2, 3)))
    assert len(parsed_array) == 1
    assert isinstance(parsed_array[0], xr.DataArray)

    dataset = xr.Dataset(
        {
            "line": line,
            "image": image,
            "five_dim": five_dim,
            "scalar": xr.DataArray(1.0),
        }
    )
    assert [darr.name for darr in provenance._parse_replay_input(dataset)] == [
        "line",
        "image",
        "five_dim",
    ]

    tree = xr.DataTree.from_dict({"leaf": xr.Dataset({"image": image})})
    assert [darr.name for darr in provenance._parse_replay_input(tree)] == ["image"]
    xr.testing.assert_identical(
        provenance._select_replay_input(
            dataset,
            provenance.FileDataSelection(kind="dataset_variable", value="image"),
        ),
        image,
    )
    xr.testing.assert_identical(
        provenance._select_replay_input(
            tree,
            provenance.FileDataSelection(kind="datatree_path", value="/leaf/image"),
        ),
        image,
    )
    assert provenance._select_replay_input(
        np.arange(6).reshape((2, 3)),
        provenance.FileDataSelection(kind="dataarray"),
    ).shape == (2, 3)

    with pytest.raises(ValueError, match="No valid data"):
        provenance._parse_replay_input([])
    with pytest.raises(ValueError, match="No valid data"):
        provenance._parse_replay_input(xr.Dataset({"scalar": xr.DataArray(1.0)}))
    with pytest.raises(TypeError, match="Unsupported input type list"):
        provenance._parse_replay_input([object()])
    with pytest.raises(KeyError, match="Selected file variable"):
        provenance._select_replay_input(
            dataset,
            provenance.FileDataSelection(kind="dataset_variable", value="missing"),
        )
    with pytest.raises(KeyError, match="Selected file DataTree path"):
        provenance._select_replay_input(
            tree,
            provenance.FileDataSelection(kind="datatree_path", value="/missing/image"),
        )

    assert (
        provenance._resolve_importable_callable("xarray.load_dataarray")
        is xr.load_dataarray
    )
    with pytest.raises(ValueError, match="must be dotted"):
        provenance._resolve_importable_callable("load")
    with pytest.raises(ModuleNotFoundError):
        provenance._resolve_importable_callable("missing_erlab_replay_loader.load")
    with pytest.raises(AttributeError):
        provenance._resolve_importable_callable("xarray.missing_loader.load")
    with pytest.raises(TypeError, match="not callable"):
        provenance._resolve_importable_callable("math.pi")

    broken_module = tmp_path / "broken_loader.py"
    broken_module.write_text(
        "import missing_erlab_replay_dependency\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="missing_erlab_replay_dependency"):
        provenance._resolve_importable_callable("broken_loader.load")

    source_file = tmp_path / "source.h5"
    image.to_netcdf(source_file, engine="h5netcdf")
    dataset_file = tmp_path / "dataset.h5"
    dataset.to_netcdf(dataset_file, engine="h5netcdf")
    xr.testing.assert_identical(
        provenance._load_file_source_data(
            _file_replay_source(
                dataset_file,
                replay_call=provenance.FileReplayCall(
                    kind="callable",
                    target="xarray.load_dataset",
                    kwargs={"engine": "h5netcdf"},
                    selection=provenance.FileDataSelection(
                        kind="dataset_variable",
                        value="image",
                    ),
                ),
            )
        ),
        image,
    )
    datatree_file = tmp_path / "tree.h5"
    tree.to_netcdf(datatree_file, engine="h5netcdf")
    xr.testing.assert_identical(
        provenance._load_file_source_data(
            _file_replay_source(
                datatree_file,
                replay_call=provenance.FileReplayCall(
                    kind="callable",
                    target="xarray.load_datatree",
                    kwargs={"engine": "h5netcdf"},
                    selection=provenance.FileDataSelection(
                        kind="datatree_path",
                        value="/leaf/image",
                    ),
                ),
            )
        ),
        image,
    )
    with pytest.raises(IndexError, match="out of range"):
        provenance._load_file_source_data(
            _file_replay_source(
                source_file,
                replay_call=provenance.FileReplayCall(
                    kind="callable",
                    target="xarray.load_dataarray",
                    kwargs={"engine": "h5netcdf"},
                    selected_index=1,
                ),
            )
        )
    with pytest.raises(ValueError, match="replay metadata"):
        provenance._load_file_source_data(
            provenance.FileLoadSource(
                path=source_file,
                loader_label="Load Function",
                loader_text="xarray.load_dataarray",
                kwargs_text="(none)",
            )
        )
    with pytest.raises(TypeError, match="Expected structured file provenance"):
        provenance.replay_file_provenance(provenance.full_data())
    with pytest.raises(TypeError, match="Expected structured file provenance"):
        provenance.replay_file_provenance(typing.cast("typing.Any", None))


def test_file_replay_uses_erlab_loader(example_loader, example_data_dir) -> None:
    del example_loader
    file_path = example_data_dir / "data_002.h5"
    spec = provenance.file_load(
        start_label="Load data from file 'data_002.h5'",
        seed_code="import erlab\n\nderived = erlab.io.load(2)",
        file_load_source=_file_replay_source(
            file_path,
            replay_call=provenance.FileReplayCall(
                kind="erlab_loader",
                target="example",
                kwargs={},
                selected_index=0,
            ),
        ),
    )

    xr.testing.assert_identical(
        provenance.replay_file_provenance(spec),
        erlab.io.loaders["example"].load(file_path),
    )


def test_file_provenance_composes_structured_stages_and_replays_modified_source(
    tmp_path,
) -> None:
    path = tmp_path / "source.h5"
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )
    data.to_netcdf(path, engine="h5netcdf")

    file_spec = provenance.file_load(
        start_label=f"Load data from file {path.name!r}",
        seed_code=(
            "import xarray\n\n"
            f"derived = xarray.load_dataarray({str(path)!r}, "
            'engine="h5netcdf").astype("float64")'
        ),
        file_load_source=provenance.FileLoadSource(
            path=path,
            loader_label="Load Function",
            loader_text="xarray.load_dataarray",
            kwargs_text='engine="h5netcdf"',
            replay_call=provenance.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                kwargs={"engine": "h5netcdf"},
                selected_index=0,
                cast_float64=True,
            ),
            load_code=(
                "import xarray\n\n"
                f"data = xarray.load_dataarray({str(path)!r}, "
                'engine="h5netcdf").astype("float64")'
            ),
        ),
    )
    first_stage = provenance.full_data(
        provenance.AverageOperation(dims=("x",)),
        provenance.RenameOperation(name="avg"),
    )
    second_stage = provenance.selection(
        provenance.IselOperation(kwargs={"y": slice(0, 2)}),
        provenance.RenameDimsCoordsOperation(mapping={"y": "energy"}),
        provenance.AssignCoordsOperation(
            coord_name="energy",
            values=np.array([10.0, 20.0]),
        ),
    )

    composed = provenance.compose_full_provenance(file_spec, first_stage)
    composed = provenance.compose_full_provenance(composed, second_stage)

    assert composed is not None
    assert composed.kind == "file"
    assert [stage.source_kind for stage in composed.replay_stages] == [
        "full_data",
        "selection",
    ]
    assert all(
        not isinstance(operation, provenance.ScriptCodeOperation)
        for stage in composed.replay_stages
        for operation in stage.operations
    )
    assert composed.display_entries()[0].label == "Load data from file 'source.h5'"
    assert any("Average" in entry.label for entry in composed.display_entries())

    code = composed.display_code()
    assert code is not None
    assert "import xarray" in code
    assert "xarray.load_dataarray" in code
    assert '.rename("avg")' not in code
    assert ".rename(y=" in code
    assert "data =" not in code
    namespace = _exec_generated_code(code, {})
    assert isinstance(namespace["derived"], xr.DataArray)

    updated = data + 100
    updated.to_netcdf(path, engine="h5netcdf")
    live_expected = second_stage.apply(first_stage.apply(updated.astype(np.float64)))
    xr.testing.assert_identical(
        provenance.replay_file_provenance(composed), live_expected
    )


def test_tool_provenance_compose_display_provenance_streamlines_live_source() -> None:
    parent = provenance.script(
        start_label="Start from watched variable 'my_data_name'",
        seed_code="derived = my_data_name",
    )
    source = provenance.selection(
        provenance.IselOperation(kwargs={"z": 0}),
        provenance.SortCoordOrderOperation(),
        provenance.SqueezeOperation(),
    )

    composed = provenance.compose_display_provenance(
        parent,
        source,
        parent_data=_base_data(),
    )

    assert composed is not None
    assert [operation.op for operation in composed.operations] == ["isel"]
    code = composed.display_code()
    assert code is not None
    namespace = _exec_generated_code(
        code,
        {"my_data_name": _base_data().copy(deep=True)},
    )
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, _base_data().isel(z=0))


def test_tool_provenance_display_compose_keeps_default_seed_without_parent() -> None:
    source = provenance.selection(
        provenance.IselOperation(kwargs={"z": 0}),
        provenance.SortCoordOrderOperation(),
        provenance.SqueezeOperation(),
    )

    composed = provenance.compose_display_provenance(
        None,
        source,
        parent_data=_base_data(),
    )

    assert composed is not None
    code = composed.display_code()
    assert code is not None
    namespace = _exec_generated_code(code, {"data": _base_data().copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, _base_data().isel(z=0))


def test_tool_provenance_direct_replay_input_name_requires_simple_seed() -> None:

    watched = provenance.script(
        start_label="Start from watched variable 'my_data'",
        seed_code="derived = my_data",
    )
    assert provenance.direct_replay_input_name(watched) == "my_data"
    watched_cast = provenance.script(
        start_label="Start from watched variable 'my_data'",
        seed_code="derived = my_data.astype(np.float64)",
    )
    assert provenance.direct_replay_input_name(watched_cast) == (
        "my_data.astype(np.float64)"
    )

    assert (
        provenance.direct_replay_input_name(
            provenance.script(
                start_label="Start from current parent ImageTool data",
                seed_code="derived = data",
            )
        )
        is None
    )
    assert (
        provenance.direct_replay_input_name(
            provenance.script(
                start_label="Start from watched variable 'my_data'",
                seed_code="derived = data.sel(x=0)",
            )
        )
        is None
    )


def test_tool_provenance_compose_display_replay_omits_synthetic_1d_squeeze() -> None:
    parent = provenance.script(
        start_label="Start from watched variable 'my_1d'",
        seed_code="derived = my_1d",
    )
    source = provenance.selection(
        provenance.SortCoordOrderOperation(),
        provenance.SqueezeOperation(),
    )
    parent_data = xr.DataArray(
        np.arange(5).reshape((5, 1)),
        dims=("x", "stack_dim"),
        coords={"x": np.arange(5), "stack_dim": [0]},
    )
    parent_data = provenance.mark_promoted_1d_source(parent_data)

    composed = provenance.compose_display_provenance(
        parent,
        source,
        parent_data=parent_data,
    )

    assert composed is not None
    code = composed.display_code()
    assert code is not None
    watched_data = xr.DataArray(
        np.arange(5),
        dims=("x",),
        coords={"x": np.arange(5)},
    )
    namespace = _exec_generated_code(code, {"my_1d": watched_data.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, watched_data)
    assert ".squeeze()" not in code
