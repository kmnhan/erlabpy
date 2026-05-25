import ast
import json
import pathlib
import typing

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

import erlab


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


def _file_replay_source(
    path: typing.Any = "scan.h5",
    *,
    replay_call: typing.Any = None,
) -> typing.Any:
    prov = erlab.interactive.imagetool.provenance
    if replay_call is None:
        replay_call = prov.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={},
            selected_index=0,
        )
    return prov.FileLoadSource(
        path=path,
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text="(none)",
        replay_call=replay_call,
        load_code=None,
    )


def _file_provenance_spec(path: typing.Any = "scan.h5") -> typing.Any:
    prov = erlab.interactive.imagetool.provenance
    return prov.file_load(
        start_label="Load data from file 'scan.h5'",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(path),
    )


def test_tool_provenance_codec_and_combinators() -> None:
    prov = erlab.interactive.imagetool.provenance
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})
    encoded = prov.encode_provenance_value(
        {"sel": slice(1.0, 2.0), "data": _base_data(), "edge_fit": edge_fit}
    )
    decoded = prov.decode_provenance_value(encoded)

    assert decoded["sel"] == slice(1.0, 2.0)
    xr.testing.assert_identical(decoded["data"], _base_data())
    xr.testing.assert_identical(decoded["edge_fit"], edge_fit)

    hashable_encoded = prov.encode_provenance_value(
        {1: slice(0.0, 1.0), ("beta", 0): {"nested": [1, 2, 3]}}
    )
    assert prov._MAPPING_MARKER in hashable_encoded
    mapping_entries = hashable_encoded[prov._MAPPING_MARKER]
    assert mapping_entries[0][0] == 1
    assert mapping_entries[1][0] == {prov._TUPLE_MARKER: ["beta", 0]}
    assert prov.decode_provenance_value(hashable_encoded) == {
        1: slice(0.0, 1.0),
        ("beta", 0): {"nested": [1, 2, 3]},
    }

    spec = prov.full_data(prov.AverageOperation(dims=("y",))).append_final_rename("avg")
    trimmed = spec.drop_trailing_rename()
    replaced = spec.append_replacement_operations(
        prov.ThinOperation(mode="global", factor=2)
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
    prov = erlab.interactive.imagetool.provenance
    payload = {
        "schema_version": 1,
        "kind": "full_data",
        "operations": [
            {"op": "average", "dims": {prov._TUPLE_MARKER: ["x"]}},
            {"op": "rename", "name": "avg"},
        ],
    }

    spec = prov.parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 2
    assert [op.op for op in spec.operations] == ["average", "rename"]
    assert [entry.label for entry in spec.derivation_entries()] == [
        "Start from current parent ImageTool data",
        'Average(dims=("x",))',
    ]
    assert spec.derivation_code() == 'derived = data\nderived = derived.qsel.mean("x")'

    dumped = spec.model_dump(mode="json")
    assert dumped["schema_version"] == 2
    assert "active_name" in dumped
    assert dumped["active_name"] is None
    assert dumped["operations"][0]["op"] == "average"
    assert dumped["operations"][0]["dims"] == {prov._TUPLE_MARKER: ["x"]}
    assert spec.to_replay_spec().active_name == "derived"

    with pytest.raises(ValidationError, match="Unknown provenance operation"):
        prov.parse_tool_provenance_spec(
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
        prov.parse_tool_provenance_spec(
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
        prov.parse_tool_provenance_spec({"kind": "full_data", "operations": 1})

    with pytest.raises(
        TypeError, match="Serialized provenance operations must be a sequence"
    ):
        prov.parse_tool_provenance_spec(
            {"kind": "full_data", "operations": {"op": "average", "dims": ["x"]}}
        )


def test_tool_provenance_parse_legacy_file_script_metadata() -> None:
    prov = erlab.interactive.imagetool.provenance
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

    spec = prov.parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 2
    assert spec.kind == "script"
    assert spec.file_load_source is not None
    assert spec.file_load_source.path == "scan.h5"
    assert spec.file_load_source.replay_call is None
    assert spec.derivation_code() == (
        "import xarray\n\n"
        "derived = xarray.load_dataarray('scan.h5')\n"
        'derived = derived.qsel.average("x")'
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
        erlab.interactive.imagetool.provenance.QSelOperation(kwargs={"beta": 2.0}),
        erlab.interactive.imagetool.provenance.IselOperation(
            kwargs={"alpha": slice(1, 3)}
        ),
        erlab.interactive.imagetool.provenance.SortCoordOrderOperation(),
    )
    xr.testing.assert_identical(
        selection_spec.apply(nonuniform),
        nonuniform_public.qsel(beta=2.0).isel({"alpha": slice(1, 3)}),
    )

    transformed = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.IselOperation(kwargs={"z": 0}),
        erlab.interactive.imagetool.provenance.SelOperation(
            kwargs={"y": slice(11.0, 12.0)}
        ),
        erlab.interactive.imagetool.provenance.TransposeOperation(dims=("y", "x")),
        erlab.interactive.imagetool.provenance.SqueezeOperation(),
        erlab.interactive.imagetool.provenance.RenameOperation(name="done"),
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
            erlab.interactive.imagetool.provenance.AverageOperation(dims=("y",))
        ).apply(data),
        data.qsel.mean("y"),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            erlab.interactive.imagetool.provenance.QSelAggregationOperation(
                dims=("y",),
                func="sum",
            )
        ).apply(data),
        data.qsel.sum("y"),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            erlab.interactive.imagetool.provenance.CoarsenOperation(
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
            erlab.interactive.imagetool.provenance.ThinOperation(
                mode="global", factor=2
            )
        ).apply(data),
        data.thin(2),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            erlab.interactive.imagetool.provenance.ThinOperation(
                mode="per_dim", factors={"x": 2}
            )
        ).apply(data),
        data.thin({"x": 2}),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            erlab.interactive.imagetool.provenance.SwapDimsOperation(
                mapping={"x": "x_alt"}
            )
        ).apply(data),
        data.swap_dims({"x": "x_alt"}),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            erlab.interactive.imagetool.provenance.RenameDimsCoordsOperation(
                mapping={"x": "kx", "x_alt": "label"}
            )
        ).apply(data),
        data.rename({"x": "kx", "x_alt": "label"}),
    )

    assigned = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.AssignCoordsOperation(
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
    prov = erlab.interactive.imagetool.provenance
    data = _string_key_data().assign_coords(
        {"coord-1": xr.DataArray([100.0, 101.0, 102.0], dims=["k-space"])}
    )
    operation = prov.RenameDimsCoordsOperation(
        mapping={"k-space": "kx", "coord-1": "temperature"}
    )
    expected = data.rename({"k-space": "kx", "coord-1": "temperature"})

    parsed = prov.parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation
    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    spec = prov.full_data(operation).to_replay_spec()
    code = spec.display_code(parent_data=data)
    assert code is not None
    assert ".rename(" in code
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


def test_tool_provenance_interpolation_operation_round_trip_and_code() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(
        np.arange(6).reshape((3, 2)).astype(float),
        dims=("k-space", "y"),
        coords={"k-space": [0.0, 1.0, 2.0], "y": [10.0, 20.0]},
        name="data",
    )
    values = np.linspace(0.0, 2.0, 5)
    operation = prov.InterpolationOperation(
        dim="k-space", values=values, method="linear"
    )
    expected = data.interp({"k-space": values}, method="linear")

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = prov.parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data, parent_data=data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    assert "Interpolate" in entry.label
    assert '.interp({"k-space": np.linspace' in entry.code
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    code = prov.full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".interp") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


def test_tool_provenance_leading_edge_operation_round_trip_and_code() -> None:
    prov = erlab.interactive.imagetool.provenance
    ev = np.linspace(0.0, 4.0, 5)
    data = xr.DataArray(
        np.vstack([4.0 - ev, 8.0 - 2.0 * ev, 2.0 - 0.5 * ev]),
        dims=("x", "eV"),
        coords={"x": np.arange(3), "eV": ev},
        name="data",
    )
    operation = prov.LeadingEdgeOperation(
        fraction=0.5,
        dim="eV",
        direction="positive",
    )
    expected = erlab.analysis.interpolate.leading_edge(data)

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = prov.parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data, parent_data=data), expected)

    payload = prov.full_data(operation).model_dump(mode="json")
    json.dumps(payload)
    reparsed_spec = prov.parse_tool_provenance_spec(payload)
    assert reparsed_spec is not None
    xr.testing.assert_identical(reparsed_spec.apply(data), expected)

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    assert "leading_edge" in entry.code
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)

    code = prov.full_data(operation).to_replay_spec().display_code(parent_data=data)
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
    prov = erlab.interactive.imagetool.provenance
    data = _base_data()

    spec = prov.full_data(
        prov.AssignCoordsOperation(coord_name="y", values=values)
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
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(
        np.arange(1),
        dims=("x",),
        coords={"x": np.array([0.0])},
        name="data",
    )
    values = np.array([5.0])

    spec = prov.full_data(
        prov.AssignCoordsOperation(coord_name="x", values=values)
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
    prov = erlab.interactive.imagetool.provenance
    data = _base_data()
    operation = prov.AssignScalarCoordOperation(coord_name="temperature", value=21.5)
    expected = erlab.utils.array.sort_coord_order(
        data.assign_coords({"temperature": 21.5}),
        keys=data.coords.keys(),
        dims_first=False,
    )

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = prov.parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation

    code = prov.full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".assign_coords") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"], data.assign_coords(temperature=21.5)
    )


def test_tool_provenance_assign_1d_coord_operation() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data()
    values = np.array(["low", "mid", "high"])
    operation = prov.AssignCoord1DOperation(
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
    parsed = prov.parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation

    code = prov.full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".assign_coords") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(
        namespace["derived"], data.assign_coords(label=("x", values))
    )


def test_tool_provenance_assign_attrs_operation() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data().assign_attrs(source="old", count=1)
    attrs = {"source": "new", "flag": True, "meta": {"scan": 1}}
    operation = prov.AssignAttrsOperation(attrs=attrs)
    expected = data.assign_attrs(attrs)

    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)
    parsed = prov.parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation

    code = prov.full_data(operation).to_replay_spec().display_code(parent_data=data)
    assert code is not None
    assert any(call.endswith(".assign_attrs") for call in _generated_call_names(code))
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], expected)


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
    prov = erlab.interactive.imagetool.provenance
    operation = prov.AffineCoordOperation(
        coord_name=coord_name,
        scale=scale,
        offset=offset,
    )

    expected = _expected_affine_coord(data, coord_name, scale, offset)
    xr.testing.assert_identical(operation.apply(data, parent_data=data), expected)

    parsed = prov.parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation
    xr.testing.assert_identical(parsed.apply(data, parent_data=data), expected)

    code = prov.full_data(operation).to_replay_spec().display_code(parent_data=data)
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


def test_tool_provenance_divide_by_coord_operation() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data().assign_coords(mesh_current=("x", [1.0, 2.0, 4.0]))

    spec = prov.full_data(prov.DivideByCoordOperation(coord_name="mesh_current"))
    xr.testing.assert_identical(spec.apply(data), data / data.mesh_current)
    code = spec.derivation_code()
    assert code is not None
    assert "derived.mesh_current" in code
    namespace = _exec_generated_code(code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], data / data.mesh_current)

    reparsed = prov.parse_tool_provenance_spec(spec.model_dump(mode="json"))
    assert reparsed == spec
    xr.testing.assert_identical(reparsed.apply(data), data / data.mesh_current)


def test_tool_provenance_divide_by_coord_fallback_code_and_broadcast() -> None:
    prov = erlab.interactive.imagetool.provenance
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

    spaced_spec = prov.full_data(prov.DivideByCoordOperation(coord_name="mesh current"))
    spaced_code = spaced_spec.derivation_code()
    assert spaced_code is not None
    assert 'derived.coords["mesh current"]' in spaced_code
    namespace = _exec_generated_code(spaced_code, {"data": data})
    xr.testing.assert_identical(
        namespace["derived"], data / data.coords["mesh current"]
    )

    conflict_spec = prov.full_data(prov.DivideByCoordOperation(coord_name="mean"))
    conflict_code = conflict_spec.derivation_code()
    assert conflict_code is not None
    assert 'derived.coords["mean"]' in conflict_code
    namespace = _exec_generated_code(conflict_code, {"data": data})
    xr.testing.assert_identical(namespace["derived"], data / data.coords["mean"])

    broadcast_spec = prov.full_data(prov.DivideByCoordOperation(coord_name="mesh_map"))
    xr.testing.assert_identical(
        broadcast_spec.apply(data), data / data.coords["mesh_map"]
    )


def test_tool_provenance_divide_by_coord_rejects_zero_values() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data().assign_coords(mesh_current=("x", [1.0, 0.0, 4.0]))
    spec = prov.full_data(prov.DivideByCoordOperation(coord_name="mesh_current"))

    with pytest.raises(ValueError, match="zero values"):
        spec.apply(data)


def test_tool_provenance_public_data_replays_on_restored_nonuniform_dims() -> None:
    prov = erlab.interactive.imagetool.provenance
    public = xr.DataArray(
        np.arange(20).reshape((5, 4)),
        dims=("x", "y"),
        coords={"x": [0.0, 0.2, 0.8, 1.4, 2.0], "y": np.arange(4)},
        name="data",
    )
    uniform = erlab.interactive.imagetool.slicer.make_dims_uniform(public)

    spec = prov.public_data(
        prov.CoarsenOperation(
            dim={"x": 2},
            boundary="trim",
            side="left",
            coord_func="mean",
            reducer="mean",
        )
    )
    reparsed = prov.parse_tool_provenance_spec(spec.model_dump(mode="json"))

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

    restored_spec = prov.full_data(
        prov.AverageOperation(dims=("y",)),
        prov.RestoreNonuniformDimsOperation(),
    )
    reparsed_restored = prov.parse_tool_provenance_spec(
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
    prov = erlab.interactive.imagetool.provenance
    data = _hashable_data()
    string_key_data = _string_key_data()

    qsel_spec = prov.full_data(
        prov.QSelOperation(kwargs={"k-space": 1.0, "k-space_width": 1.0})
    )
    assert qsel_spec.derivation_code() == (
        'derived = data\nderived = derived.qsel({"k-space": 1.0, "k-space_width": 1.0})'
    )
    xr.testing.assert_identical(
        qsel_spec.apply(string_key_data),
        string_key_data.qsel({"k-space": 1.0, "k-space_width": 1.0}),
    )

    isel_spec = prov.full_data(prov.IselOperation(kwargs={1: slice(1, 3)}))
    assert (
        isel_spec.derivation_code()
        == "derived = data\nderived = derived.isel({1: slice(1, 3)})"
    )
    xr.testing.assert_identical(isel_spec.apply(data), data.isel({1: slice(1, 3)}))

    transpose_spec = prov.full_data(prov.TransposeOperation(dims=(("beta", 0), 1)))
    assert (
        transpose_spec.derivation_code()
        == 'derived = data\nderived = derived.transpose(*(("beta", 0), 1))'
    )
    xr.testing.assert_identical(
        transpose_spec.apply(data), data.transpose(("beta", 0), 1)
    )

    average_spec = prov.full_data(prov.AverageOperation(dims=("k-space",)))
    assert (
        average_spec.derivation_code()
        == 'derived = data\nderived = derived.qsel.mean("k-space")'
    )
    xr.testing.assert_identical(
        average_spec.apply(string_key_data), string_key_data.qsel.mean("k-space")
    )

    tuple_average_spec = prov.full_data(prov.AverageOperation(dims=(("beta", 0),)))
    assert (
        tuple_average_spec.derivation_code()
        == 'derived = data\nderived = derived.qsel.mean((("beta", 0),))'
    )

    aggregate_spec = prov.full_data(
        prov.QSelAggregationOperation(dims=("k-space",), func="sum")
    )
    assert (
        aggregate_spec.derivation_code()
        == 'derived = data\nderived = derived.qsel.sum("k-space")'
    )
    xr.testing.assert_identical(
        aggregate_spec.apply(string_key_data), string_key_data.qsel.sum("k-space")
    )

    mean_aggregate_spec = prov.full_data(
        prov.QSelAggregationOperation(dims=(("beta", 0),), func="mean")
    )
    assert mean_aggregate_spec.derivation_code() == (
        'derived = data\nderived = derived.qsel.mean((("beta", 0),))'
    )

    dumped = aggregate_spec.model_dump(mode="json")
    assert dumped["operations"][0] == {
        "op": "qsel_aggregate",
        "dims": {prov._TUPLE_MARKER: ["k-space"]},
        "func": "sum",
    }
    reparsed = prov.parse_tool_provenance_spec(dumped)
    assert reparsed.operations[0] == aggregate_spec.operations[0]

    coarsen_spec = prov.full_data(
        prov.CoarsenOperation(
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

    thin_spec = prov.full_data(prov.ThinOperation(mode="per_dim", factors={1: 2}))
    assert (
        thin_spec.derivation_code() == "derived = data\nderived = derived.thin({1: 2})"
    )
    xr.testing.assert_identical(thin_spec.apply(data), data.thin({1: 2}))

    swap_spec = prov.full_data(prov.SwapDimsOperation(mapping={1: "coord_1"}))
    assert (
        swap_spec.derivation_code()
        == 'derived = data\nderived = derived.swap_dims({1: "coord_1"})'
    )
    xr.testing.assert_identical(swap_spec.apply(data), data.swap_dims({1: "coord_1"}))

    dumped = tuple_average_spec.model_dump(mode="json")
    assert dumped["operations"][0]["dims"] == {
        prov._TUPLE_MARKER: [{prov._TUPLE_MARKER: ["beta", 0]}]
    }

    coarsen_dump = coarsen_spec.model_dump(mode="json")
    assert coarsen_dump["operations"][0]["dim"] == {prov._MAPPING_MARKER: [[1, 2]]}


def test_tool_provenance_display_entries_streamline_live_source() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data()

    hidden_spec = prov.full_data(
        prov.IselOperation(kwargs={}),
        prov.SortCoordOrderOperation(),
        prov.TransposeOperation(dims=data.dims),
        prov.SqueezeOperation(),
    )
    assert [entry.label for entry in hidden_spec.display_entries(parent_data=data)] == [
        "Start from current parent ImageTool data"
    ]
    assert hidden_spec.display_code(parent_data=data) is None

    squeezed_spec = prov.full_data(
        prov.IselOperation(kwargs={"z": slice(0, 1)}),
        prov.SqueezeOperation(),
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
    prov = erlab.interactive.imagetool.provenance

    spec = prov.script(
        prov.ScriptCodeOperation(label="isel()", code="derived = derived.isel()"),
        prov.ScriptCodeOperation(
            label="Sort coordinates to parent order",
            code=(
                "derived = erlab.utils.array.sort_coord_order("
                "derived, data.coords.keys())"
            ),
        ),
        prov.ScriptCodeOperation(
            label="Custom coordinate-order step",
            code=(
                "derived = erlab.utils.array.sort_coord_order("
                "derived, data.coords.keys(), dims_first=False)"
            ),
        ),
        prov.ScriptCodeOperation(
            label="transpose(('x', 'y', 'z'))",
            code="derived = derived.transpose(*('x', 'y', 'z'))",
        ),
        prov.ScriptCodeOperation(
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


def test_imagetool_selection_source_binding_materializes_current_coordinates() -> None:
    prov = erlab.interactive.imagetool.provenance
    original = _base_data()
    shifted = original.assign_coords(y=[20.0, 21.0, 22.0, 23.0])

    binned = prov.ImageToolSelectionSourceBinding(
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

    unbinned = prov.ImageToolSelectionSourceBinding(selection_indexers={"y": 2})
    unbinned_spec = unbinned.materialize(shifted)
    assert unbinned_spec.operations[0].decoded_kwargs == {"y": 22.0}
    xr.testing.assert_identical(unbinned_spec.apply(shifted), shifted.qsel(y=22.0))

    cropped = prov.ImageToolSelectionSourceBinding(crop_sel_indexers={"y": slice(1, 3)})
    cropped_spec = cropped.materialize(shifted)
    assert cropped_spec.operations[0].decoded_kwargs == {"y": slice(21.0, 22.0)}
    xr.testing.assert_identical(
        cropped_spec.apply(shifted),
        shifted.sel(y=slice(21.0, 22.0)),
    )


def test_imagetool_selection_source_binding_round_trips_and_reuses_operations() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data()
    binding = prov.ImageToolSelectionSourceBinding(
        selection_mode="isel",
        selection_indexers={"z": 1},
        crop_sel_indexers={"x": slice(0, 3)},
        crop_isel_indexers={"y": slice(1, 3)},
        transpose_dims=("y", "x"),
        squeeze=True,
    )

    reparsed = prov.ImageToolSelectionSourceBinding.model_validate(
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
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
    )

    with pytest.raises(ValueError, match="Dimension `missing` not found"):
        prov.ImageToolSelectionSourceBinding(
            crop_sel_indexers={"missing": slice(0, 1)}
        ).materialize(data)

    with pytest.raises(ValueError, match="Selection for dimension `x` is empty"):
        prov.ImageToolSelectionSourceBinding(
            crop_sel_indexers={"x": slice(1, 1)}
        ).materialize(data)

    binding = prov.ImageToolSelectionSourceBinding(
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
        erlab.interactive.imagetool.provenance.AverageOperation(
            dims=(_UnsupportedHashable(),)
        )


def test_tool_provenance_validation_helpers_and_error_branches() -> None:
    prov = erlab.interactive.imagetool.provenance
    base_operation = prov.ToolProvenanceOperation()

    assert prov._format_derivation_value([1, 2]) == "(1, 2)"
    assert prov._format_selection_step("isel", {}) == "derived = derived.isel()"
    assert prov._simplify_display_code("if") == "if"
    assert prov._simplify_display_code("") == ""
    assert (
        prov._simplify_display_code("for item in []:\n    pass")
        == "for item in []:\n    pass"
    )
    assert prov._simplify_display_code("derived = data\nresult = derived + 1") == (
        "result = data + 1"
    )
    simplified = prov._simplify_display_code(
        "derived = data\nscale = 2\nresult = derived + scale"
    )
    simplified_namespace = _exec_generated_code(simplified, {"data": 3})
    assert simplified_namespace["result"] == 5
    assert "derived" not in simplified_namespace
    invalidated_namespace = _exec_generated_code(
        prov._simplify_display_code(
            "derived = data + 1\ndata = other\nresult = derived"
        ),
        {"data": 3, "other": 10},
    )
    assert invalidated_namespace["result"] == 4
    rebased = prov.rebase_default_replay_input(
        "derived = data\nscale = 2\nresult = derived + scale",
        "source_data",
    )
    rebased_namespace = _exec_generated_code(rebased, {"source_data": 3})
    assert rebased_namespace["result"] == 5
    assert "derived" not in rebased_namespace
    assert prov.uses_default_replay_input("result = data + 1")
    assert not prov.uses_default_replay_input("result = source_data + 1")

    with pytest.raises(ValueError, match="Expected 2 items"):
        prov._ensure_float_tuple([1.0], expected_len=2)
    with pytest.raises(TypeError, match="expected an array-like sequence"):
        prov._coerce_float_sequence("not-a-sequence")
    with pytest.raises(TypeError, match="active_name must be a string"):
        prov._validate_active_name(1)
    with pytest.raises(ValueError, match="active_name must be a valid"):
        prov._validate_active_name("for")
    with pytest.raises(TypeError, match="expected a sequence"):
        prov.ToolProvenanceOperation._coerce_hashable_tuple_field("x")
    with pytest.raises(ValueError, match="Expected 2 items"):
        prov.ToolProvenanceOperation._coerce_hashable_tuple_field([1], expected_len=2)
    assert prov.ToolProvenanceOperation._coerce_hashable_mapping_field(None) == {}
    with pytest.raises(TypeError, match="expected a mapping"):
        prov.ToolProvenanceOperation._coerce_hashable_mapping_field([("x", 1)])
    with pytest.raises(NotImplementedError):
        base_operation.apply(_base_data(), parent_data=_base_data())
    with pytest.raises(NotImplementedError):
        base_operation.derivation_entry()
    with pytest.raises(TypeError, match="must be mappings"):
        prov.parse_tool_provenance_operation(1)
    with pytest.raises(TypeError, match="must include a string `op`"):
        prov.parse_tool_provenance_operation({"op": 1})
    with pytest.raises(TypeError, match="array-like"):
        prov.AssignCoordsOperation(coord_name="x", values=object())
    with pytest.raises(TypeError, match=r"xarray\.Dataset"):
        prov.CorrectWithEdgeOperation(edge_fit=object())

    assert prov.ToolProvenanceSpec(kind="full_data", operations=None).operations == ()
    with pytest.raises(ValidationError, match="must define `start_label`"):
        prov.ToolProvenanceSpec(kind="script", active_name="derived")
    with pytest.raises(ValidationError, match="Only script or file provenance specs"):
        prov.ToolProvenanceSpec(kind="full_data", start_label="bad")
    with pytest.raises(TypeError, match="Script and file provenance use"):
        prov.script(start_label="Start", active_name="derived")._display_operations()


def test_select_coord_operation_round_trips_and_applies() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data().assign_coords(temp=("x", [100.0, 200.0, 300.0]))
    operation = prov.SelectCoordOperation(coord_name="temp")

    xr.testing.assert_identical(
        operation.apply(data, parent_data=data), data.coords["temp"]
    )

    entry = operation.derivation_entry()
    assert entry.copyable is True
    assert entry.code is not None
    namespace = _exec_generated_code(entry.code, {"derived": data.copy(deep=True)})
    xr.testing.assert_identical(namespace["derived"], data.coords["temp"])

    parsed = prov.parse_tool_provenance_operation(operation.model_dump(mode="json"))
    assert parsed == operation

    spec = prov.public_data(operation)
    assert prov.require_live_source_spec(spec) == spec
    xr.testing.assert_identical(spec.apply(data), data.coords["temp"])


def test_tool_provenance_remaining_operation_and_display_branches(monkeypatch) -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data()

    xr.testing.assert_identical(
        prov.full_data(prov.TransposeOperation()).apply(data),
        data.transpose(*reversed(data.dims)),
    )
    assert prov.TransposeOperation().derivation_entry().code == (
        "derived = derived.transpose(*reversed(derived.dims))"
    )
    assert prov.SortCoordOrderOperation().derivation_entry().copyable is True
    assert (
        prov.SelOperation(kwargs={"x": 1.0}).derivation_entry().label.startswith("sel(")
    )
    assert prov.full_data().derivation_code() is None
    assert (
        prov.script(start_label="Start", active_name="derived").display_code() is None
    )
    assert (
        prov.full_data(
            prov.CorrectWithEdgeOperation(edge_fit=xr.Dataset(), shift_coords=True)
        ).display_code()
        is None
    )

    with pytest.raises(TypeError, match="script_code operations"):
        prov.ScriptCodeOperation(label="Step", code="derived = data").apply(
            data, parent_data=data
        )
    with pytest.raises(ValidationError, match="thin global mode requires factor"):
        prov.ThinOperation(mode="global")
    with pytest.raises(ValidationError, match="thin per_dim mode requires factors"):
        prov.ThinOperation(mode="per_dim")
    assert prov.ThinOperation(mode="global", factor=2).derivation_entry().code == (
        "derived = derived.thin(2)"
    )

    monkeypatch.setattr(
        erlab.interactive.utils,
        "generate_code",
        lambda *_args, assign=None, **_kwargs: f"{assign} = generated()",
    )
    assert (
        prov.RotateOperation(angle=45.0, axes=("x", "y"), center=(0.0, 0.0))
        .derivation_entry()
        .code
        == "derived = generated()"
    )
    assert (
        prov.SymmetrizeOperation(dim="x", center=0.0).derivation_entry().code
        == "derived = generated()"
    )
    assert (
        prov.SymmetrizeNfoldOperation(fold=4, axes=("x", "y")).derivation_entry().code
        == "derived = generated()"
    )

    assign_entry = prov.AssignCoordsOperation(
        coord_name="x", values=np.array([2.0, 1.0, 0.0])
    ).derivation_entry()
    assert assign_entry.copyable is True
    assert "assign_coords" in typing.cast("str", assign_entry.code)

    ambiguous = prov.full_data(
        prov.SelOperation(kwargs={"missing": 0}),
        prov.SqueezeOperation(),
    )
    assert [entry.label for entry in ambiguous.display_entries(parent_data=data)] == [
        "Start from current parent ImageTool data",
        "sel(missing=0)",
        "squeeze()",
    ]

    parent = prov.script(
        start_label="Start from watched variable 'my_1d'",
        seed_code="derived = my_1d",
    )
    promoted = prov.mark_promoted_1d_source(data.copy(deep=False))
    assert (
        prov.compose_display_provenance(
            parent,
            prov.selection(prov.IselOperation(kwargs={"x": 0})),
            parent_data=promoted,
        )
        is not parent
    )
    assert (
        prov.compose_display_provenance(
            parent,
            prov.selection(prov.AverageOperation(dims=("x",))),
            parent_data=promoted,
        )
        is not parent
    )
    assert (
        prov.direct_replay_input_name(
            prov.script(start_label="Start", seed_code="prepared = data")
        )
        is None
    )
    assert (
        prov.direct_replay_input_name(
            prov.script(start_label="Start", seed_code="derived = for")
        )
        is None
    )
    assert prov.compose_full_provenance(parent, None) == parent


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
        erlab.interactive.imagetool.provenance.RotateOperation(
            angle=45.0, axes=("x", "y"), center=(0.5, 1.5), reshape=False, order=3
        )
    )
    assert rotate_spec.apply(data).attrs["last_op"] == "rotate"

    symmetrize_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.SymmetrizeOperation(
            dim="x", center=1.0, subtract=True, mode="valid", part="below"
        )
    )
    assert symmetrize_spec.apply(data).attrs["last_op"] == "symmetrize"

    symmetrize_nfold_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.SymmetrizeNfoldOperation(
            fold=4,
            axes=("x", "y"),
            center={"x": 1.0, "y": 11.0},
            reshape=True,
            order=2,
        )
    )
    assert symmetrize_nfold_spec.apply(data).attrs["last_op"] == "symmetrize_nfold"

    edge_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.CorrectWithEdgeOperation(
            edge_fit=edge_fit, shift_coords=False
        )
    )
    assert edge_spec.apply(data).attrs["last_op"] == "correct_with_edge"
    entries = edge_spec.derivation_entries()
    assert entries[-1].copyable is False
    assert entries[-1].code is None
    assert edge_spec.derivation_code() is None

    path_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.SliceAlongPathOperation(
            vertices={"x": [0.0, 1.0], "y": [10.0, 12.0]},
            step_size=0.5,
            dim_name="path",
        )
    )
    assert path_spec.apply(data).attrs["last_op"] == "slice_along_path"

    mask_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.MaskWithPolygonOperation(
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

    spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.CorrectWithEdgeOperation(
            edge_fit=edge_fit, shift_coords=False
        )
    )
    payload = spec.model_dump(mode="json")

    reparsed_operation = (
        erlab.interactive.imagetool.provenance.parse_tool_provenance_operation(
            payload["operations"][0]
        )
    )
    assert isinstance(
        reparsed_operation,
        erlab.interactive.imagetool.provenance.CorrectWithEdgeOperation,
    )
    xr.testing.assert_identical(reparsed_operation.decoded_edge_fit, edge_fit)

    reparsed_spec = erlab.interactive.imagetool.provenance.parse_tool_provenance_spec(
        payload
    )
    assert reparsed_spec is not None
    assert reparsed_spec.apply(data).attrs["last_op"] == "correct_with_edge"
    entries = reparsed_spec.derivation_entries()
    assert entries[-1].copyable is False
    assert entries[-1].code is None
    assert reparsed_spec.derivation_code() is None


def test_tool_provenance_roundtrip_correct_with_edge_fit_dataset(
    gold, gold_fit_res
) -> None:
    prov = erlab.interactive.imagetool.provenance
    spec = prov.full_data(
        prov.CorrectWithEdgeOperation(edge_fit=gold_fit_res, shift_coords=False)
    )

    payload = spec.model_dump(mode="json")
    json.dumps(payload)

    reparsed_operation = prov.parse_tool_provenance_operation(payload["operations"][0])
    assert isinstance(reparsed_operation, prov.CorrectWithEdgeOperation)
    decoded = reparsed_operation.decoded_edge_fit
    xr.testing.assert_identical(
        decoded.drop_vars("modelfit_results"),
        gold_fit_res.drop_vars("modelfit_results"),
    )
    assert (
        decoded.modelfit_results.item().success
        == gold_fit_res.modelfit_results.item().success
    )

    reparsed_spec = prov.parse_tool_provenance_spec(payload)
    assert reparsed_spec is not None
    xr.testing.assert_allclose(
        reparsed_spec.apply(gold),
        erlab.analysis.gold.correct_with_edge(gold, gold_fit_res, shift_coords=False),
    )


def test_tool_provenance_script_specs_reject_live_source() -> None:
    prov = erlab.interactive.imagetool.provenance

    script_spec = prov.script(
        prov.ScriptCodeOperation(
            label="Fit current tool data",
            code="result = data.mean()",
        ),
        start_label="Start from current analysis-tool input data",
        seed_code="prepared = data.copy()",
    )
    reparsed_script = prov.parse_tool_provenance_spec(
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
        prov.require_live_source_spec(reparsed_script)


def test_tool_replay_provenance_helpers_compose_parent_provenance() -> None:
    prov = erlab.interactive.imagetool.provenance

    parent = prov.selection(prov.IselOperation(kwargs={"x": slice(0, 2)}))
    local = prov.script(
        prov.ScriptCodeOperation(
            label="Compute tool output",
            code="result = derived.mean()",
        ),
        start_label="Start from current tool input data",
    )

    composed = prov.compose_full_provenance(parent, local)

    assert composed is not None
    assert composed.derivation_entries()[0].label == (
        "Start from selected parent ImageTool data"
    )
    assert composed.derivation_entries()[-1].label == "Compute tool output"
    assert composed.derivation_code() == (
        "derived = data\nderived = derived.isel(x=slice(0, 2))\nresult = derived.mean()"
    )

    assert prov.compose_display_provenance(parent, prov.full_data()) == (
        prov.to_replay_provenance_spec(parent)
    )


def test_tool_provenance_compose_full_uses_parent_active_name_for_live_local() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _base_data()
    parent = prov.script(
        prov.ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = data + 1",
        ),
        start_label="Start from current tool input data",
        active_name="result",
    )
    local = prov.full_data(prov.AverageOperation(dims=("x",)))

    composed = prov.compose_full_provenance(parent, local)

    assert composed is not None
    code = composed.derivation_code()
    assert code == (
        'result = data + 1\nderived = result\nderived = derived.qsel.mean("x")'
    )
    namespace = _exec_generated_code(code, {"data": data.copy(deep=True)})
    derived = namespace["derived"]
    assert isinstance(derived, xr.DataArray)
    xr.testing.assert_identical(derived, (data + 1).qsel.mean("x"))


def test_file_load_source_replay_call_round_trips() -> None:
    prov = erlab.interactive.imagetool.provenance

    xarray_source = prov.FileLoadSource(
        path="scan.h5",
        loader_label="Load Function",
        loader_text="xarray.load_dataarray",
        kwargs_text='engine="h5netcdf"',
        replay_call=prov.FileReplayCall(
            kind="callable",
            target="xarray.load_dataarray",
            kwargs={"engine": "h5netcdf"},
            selected_index=0,
            cast_float64=True,
        ),
        load_code='import xarray\n\ndata = xarray.load_dataarray("/tmp/scan.h5")',
    )
    parsed_xarray = prov.FileLoadSource.model_validate(
        xarray_source.model_dump(mode="json")
    )
    assert parsed_xarray == xarray_source
    assert parsed_xarray.replay_call.kind == "callable"
    assert parsed_xarray.replay_call.target == "xarray.load_dataarray"
    assert parsed_xarray.replay_call.kwargs == {"engine": "h5netcdf"}
    assert parsed_xarray.replay_call.cast_float64 is True

    erlab_source = prov.FileLoadSource(
        path="data_002.h5",
        loader_label="Loader",
        loader_text="example",
        kwargs_text="(none)",
        replay_call=prov.FileReplayCall(
            kind="erlab_loader",
            target="example",
            kwargs={},
            selected_index=0,
        ),
        load_code=(
            "import erlab\n\nerlab.io.set_loader('example')\ndata = erlab.io.load(2)"
        ),
    )
    parsed_erlab = prov.FileLoadSource.model_validate(
        erlab_source.model_dump(mode="json")
    )
    assert parsed_erlab == erlab_source
    assert parsed_erlab.replay_call.kind == "erlab_loader"
    assert parsed_erlab.replay_call.target == "example"


def test_file_provenance_validation_rejects_invalid_payloads() -> None:
    prov = erlab.interactive.imagetool.provenance
    replay_stage = prov.ReplayStage(source_kind="full_data")
    file_source = _file_replay_source()

    with pytest.raises(ValidationError, match="selected_index"):
        prov.FileReplayCall(
            kind="callable", target="xarray.load_dataarray", selected_index=-1
        )
    with pytest.raises(ValidationError, match="target"):
        prov.FileReplayCall(kind="callable", target="", selected_index=0)

    bad_kwargs_call = prov.FileReplayCall.model_construct(
        kind="callable",
        target="xarray.load_dataarray",
        kwargs={1: "bad"},
        selected_index=0,
    )
    with pytest.raises(TypeError, match="string keys"):
        bad_kwargs_call._validate_replay_call()

    assert prov.ReplayStage(source_kind="full_data", operations=None).operations == ()
    with pytest.raises(TypeError, match="replay stage operations"):
        prov.ReplayStage(source_kind="full_data", operations=1)
    with pytest.raises(TypeError, match="script-only operations"):
        prov.ReplayStage(
            source_kind="full_data",
            operations=[
                prov.ScriptCodeOperation(label="Generated", code="derived = derived")
            ],
        )
    with pytest.raises(TypeError, match="source must not be None"):
        prov.ReplayStage.from_source_spec(typing.cast("typing.Any", None))

    assert (
        prov.ToolProvenanceSpec(kind="full_data", replay_stages=None).replay_stages
        == ()
    )
    with pytest.raises(TypeError, match="Serialized replay stages"):
        prov.ToolProvenanceSpec(kind="full_data", replay_stages=1)
    with pytest.raises(ValidationError, match="cannot define replay stages"):
        prov.ToolProvenanceSpec(
            kind="script",
            start_label="Start",
            active_name="derived",
            replay_stages=[replay_stage],
        )

    with pytest.raises(ValidationError, match="must define `start_label`"):
        prov.ToolProvenanceSpec(
            kind="file",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `seed_code`"):
        prov.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            active_name="derived",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `active_name`"):
        prov.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            file_load_source=file_source,
        )
    with pytest.raises(ValidationError, match="must define `file_load_source`"):
        prov.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
        )
    with pytest.raises(ValidationError, match="must define `replay_call`"):
        prov.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source.model_copy(update={"replay_call": None}),
        )
    with pytest.raises(ValidationError, match="cannot define operations"):
        prov.ToolProvenanceSpec(
            kind="file",
            start_label="Load",
            seed_code="derived = data",
            active_name="derived",
            file_load_source=file_source,
            operations=[prov.AverageOperation(dims=("x",))],
        )
    with pytest.raises(TypeError, match="Replay stages can only"):
        prov.full_data().append_replay_stage(prov.full_data())


def test_file_provenance_display_entries_keep_steps_after_stage_failure() -> None:
    prov = erlab.interactive.imagetool.provenance
    spec = (
        _file_provenance_spec()
        .append_replay_stage(prov.full_data(prov.SelOperation(kwargs={"missing": 0})))
        .append_replay_stage(prov.full_data(prov.SqueezeOperation()))
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


def test_file_provenance_compose_fallbacks_and_replay_aliases() -> None:
    prov = erlab.interactive.imagetool.provenance
    file_spec = _file_provenance_spec()

    assert prov.replay_input_name(None) is None
    assert (
        prov.script(start_label="Start", active_name="derived").derivation_code()
        is None
    )
    assert (
        prov.script(
            start_label="Start",
            seed_code="derived = data",
            active_name="derived",
        ).display_code()
        is None
    )
    assert prov.compose_full_provenance(None, None) is None
    local_replay = prov.compose_full_provenance(
        None, prov.full_data(prov.AverageOperation(dims=("x",)))
    )
    assert local_replay is not None
    assert local_replay.kind == "script"

    assert prov.compose_full_provenance(file_spec, prov.full_data()) == file_spec
    assert prov._as_script_replay_spec(prov.full_data()).kind == "script"

    script_local = prov.script(
        prov.ScriptCodeOperation(label="Offset", code="result = derived + 1"),
        start_label="Run generated code",
        seed_code="derived = data",
        active_name="result",
    )
    file_with_script = prov.compose_full_provenance(file_spec, script_local)
    assert file_with_script is not None
    assert file_with_script.kind == "script"
    assert file_with_script.file_load_source == file_spec.file_load_source
    assert file_with_script.derivation_code() == (
        "import xarray\n\n"
        "derived = xarray.load_dataarray('scan.h5')\n"
        "result = derived + 1"
    )

    watched_parent = prov.script(
        start_label="Start from watched variable 'watched_data'",
        seed_code="derived = watched_data",
        active_name="derived",
    )
    default_seed_local = prov.script(
        prov.ScriptCodeOperation(label="Mean", code="result = derived.mean()"),
        start_label="Use current parent output",
        seed_code="derived = data",
        active_name="result",
    )
    watched_composed = prov.compose_full_provenance(watched_parent, default_seed_local)
    assert watched_composed is not None
    assert watched_composed.derivation_code() == (
        "derived = watched_data\nderived = watched_data\nresult = derived.mean()"
    )

    result_parent = prov.script(
        prov.ScriptCodeOperation(
            label="Compute intermediate result",
            code="result = data + 1",
        ),
        start_label="Start",
        active_name="result",
    )
    no_seed_local = prov.script(
        prov.ScriptCodeOperation(label="Mean", code="result = derived.mean()"),
        start_label="Use parent result",
        active_name="derived",
    )
    result_composed = prov.compose_full_provenance(result_parent, no_seed_local)
    assert result_composed is not None
    assert result_composed.derivation_code() == (
        "result = data + 1\nderived = result\nresult = derived.mean()"
    )

    promoted = prov.mark_promoted_1d_source(_base_data().copy(deep=False))
    assert (
        prov.compose_display_provenance(
            watched_parent,
            prov.selection(prov.IselOperation(), prov.SortCoordOrderOperation()),
            parent_data=promoted,
        )
        is not None
    )
    assert prov.compose_display_provenance(
        watched_parent, None
    ) == prov.to_replay_provenance_spec(watched_parent)


def test_script_provenance_supports_named_console_inputs() -> None:
    prov = erlab.interactive.imagetool.provenance
    left = prov.script(
        start_label="Load left",
        seed_code="data_0 = xr.DataArray([1.0, 2.0], dims=['x'])",
        active_name="data_0",
    )
    right = prov.script(
        start_label="Load right",
        seed_code="data_1 = xr.DataArray([0.5, 1.5], dims=['x'])",
        active_name="data_1",
    )
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Subtract console inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="ImageTool 0",
                node_uid="left",
                provenance_spec=left,
            ),
            prov.ScriptInput(
                name="data_1",
                label="ImageTool 1",
                node_uid="right",
                provenance_spec=right,
            ),
        ),
    )

    reparsed = prov.parse_tool_provenance_spec(spec.model_dump(mode="json"))

    assert reparsed == spec
    assert [entry.label for entry in spec.display_entries()] == [
        "Run ImageTool manager console code",
        "Use data_0 from ImageTool 0",
        "Use data_1 from ImageTool 1",
        "Subtract console inputs",
    ]
    code = typing.cast("str", spec.derivation_code())
    namespace = _exec_generated_code(code, {})
    xr.testing.assert_identical(
        namespace["derived"],
        xr.DataArray([0.5, 0.5], dims=["x"]),
    )


def test_script_input_code_reuses_shared_file_replay_prefix(
    tmp_path: pathlib.Path,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "polarization.nc"
    source = xr.DataArray(
        np.arange(12.0).reshape(2, 2, 3),
        dims=("pol", "energy", "k"),
        coords={"pol": ["LH", "LV"], "energy": [0.0, 1.0], "k": [0, 1, 2]},
    )
    source.to_netcdf(path)
    file_spec = prov.file_load(
        start_label="Load both polarizations",
        seed_code=f"import xarray\n\nderived = xarray.load_dataarray({str(path)!r})",
        file_load_source=prov.FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=prov.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )
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
        prov.ScriptCodeOperation(
            label="Subtract polarizations",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="ImageTool 0: LH",
                provenance_spec=left_spec,
            ),
            prov.ScriptInput(
                name="data_1",
                label="ImageTool 1: LV",
                provenance_spec=right_spec,
            ),
        ),
    )

    code = typing.cast("str", spec.display_code())

    assert code.count("xarray.load_dataarray") == 1
    assert code.count(".qsel.average") == 1
    assert "data_0 =" in code
    assert "data_1 =" in code
    namespace = _exec_generated_code(code, {})
    expected = left_stage.apply(shared_stage.apply(source)) - right_stage.apply(
        shared_stage.apply(source)
    )
    xr.testing.assert_identical(namespace["derived"], expected)


def test_script_input_code_keeps_distinct_structured_replay_nodes() -> None:
    prov = erlab.interactive.imagetool.provenance
    first = prov.file_load(
        start_label="Load first",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(
            "scan.h5",
            replay_call=prov.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )
    second = prov.file_load(
        start_label="Load second",
        seed_code="import xarray\n\nderived = xarray.load_dataarray('scan.h5')",
        file_load_source=_file_replay_source(
            "scan.h5",
            replay_call=prov.FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=1,
            ),
        ),
    )
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Subtract inputs",
            code="derived = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="ImageTool 0", provenance_spec=first),
            prov.ScriptInput(
                name="data_1",
                label="ImageTool 1",
                provenance_spec=second,
            ),
        ),
    )

    code = typing.cast("str", spec.derivation_code())

    assert code.count("xarray.load_dataarray") == 2


def test_script_input_dependency_refs_recurse_and_rebase() -> None:
    prov = erlab.interactive.imagetool.provenance
    left_snapshot_id = "left-snapshot"
    right_snapshot_id = "right-snapshot"
    extra_snapshot_id = "extra-snapshot"
    nested = prov.script(
        prov.ScriptCodeOperation(
            label="Subtract console inputs",
            code="diff = data_0 - data_1",
        ),
        start_label="Run ImageTool manager console code",
        active_name="diff",
        script_inputs=(
            prov.ScriptInput(
                name="data_0",
                label="ImageTool 0",
                node_uid="old-left",
                node_snapshot_token=left_snapshot_id,
            ),
            prov.ScriptInput(
                name="data_1",
                label="ImageTool 1",
                node_uid="old-right",
                node_snapshot_token=right_snapshot_id,
            ),
        ),
    )
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Add nested input",
            code="derived = diff + data_2",
        ),
        start_label="Run ImageTool manager console code",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(
                name="diff",
                label="console variable 'diff'",
                provenance_spec=nested,
            ),
            prov.ScriptInput(
                name="data_2",
                label="ImageTool 2",
                node_uid="old-extra",
                node_snapshot_token=extra_snapshot_id,
            ),
        ),
    )

    refs = prov.script_input_dependency_refs(spec)
    assert [(ref.name, ref.node_uid, ref.node_snapshot_token) for ref in refs] == [
        ("data_0", "old-left", left_snapshot_id),
        ("data_1", "old-right", right_snapshot_id),
        ("data_2", "old-extra", extra_snapshot_id),
    ]

    rebased = prov.rebase_script_input_node_uids(
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
        for ref in prov.script_input_dependency_refs(rebased)
    ] == [
        ("data_0", "new-left", left_snapshot_id),
        ("data_1", "new-right", right_snapshot_id),
        ("data_2", "new-extra", extra_snapshot_id),
    ]


def test_replay_script_provenance_uses_resolved_inputs_without_mutating() -> None:
    prov = erlab.interactive.imagetool.provenance
    left = xr.DataArray([1.0, 2.0], dims=("x",), coords={"x": [0, 1]})
    right = xr.DataArray([0.5, 1.5], dims=("x",), coords={"x": [0, 1]})
    spec = prov.script(
        prov.ScriptCodeOperation(
            label="Mutate local input",
            code="data_0[0] = 10.0\nderived = data_0 - data_1",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(
            prov.ScriptInput(name="data_0", label="ImageTool 0"),
            prov.ScriptInput(name="data_1", label="ImageTool 1"),
        ),
    )

    assert prov.script_provenance_replayable(spec)
    result = prov.replay_script_provenance(
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
    prov = erlab.interactive.imagetool.provenance
    data = xr.DataArray([1.0], dims=("x",))
    unsupported = prov.script(
        prov.ScriptCodeOperation(
            label="Unsupported",
            code="import os\nderived = data_0",
        ),
        start_label="Run script",
        active_name="derived",
        script_inputs=(prov.ScriptInput(name="data_0", label="ImageTool 0"),),
    )
    incomplete = prov.script(
        prov.ScriptCodeOperation(label="Incomplete", code=None),
        start_label="Run script",
        active_name="derived",
        script_inputs=(prov.ScriptInput(name="data_0", label="ImageTool 0"),),
    )

    assert not prov.script_provenance_replayable(unsupported)
    assert not prov.script_provenance_replayable(incomplete)
    with pytest.raises(TypeError, match="unsupported Import"):
        prov.replay_script_provenance(unsupported, {"data_0": data})
    with pytest.raises(ValueError, match="non-replayable"):
        prov.replay_script_provenance(incomplete, {"data_0": data})


def test_file_replay_parses_supported_inputs_and_errors(tmp_path, monkeypatch) -> None:
    prov = erlab.interactive.imagetool.provenance
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

    parsed_array = prov._parse_replay_input(np.arange(6).reshape((2, 3)))
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
    assert [darr.name for darr in prov._parse_replay_input(dataset)] == [
        "line",
        "image",
        "five_dim",
    ]

    tree = xr.DataTree.from_dict({"leaf": xr.Dataset({"image": image})})
    assert [darr.name for darr in prov._parse_replay_input(tree)] == ["image"]

    with pytest.raises(ValueError, match="No valid data"):
        prov._parse_replay_input([])
    with pytest.raises(ValueError, match="No valid data"):
        prov._parse_replay_input(xr.Dataset({"scalar": xr.DataArray(1.0)}))
    with pytest.raises(TypeError, match="Unsupported input type list"):
        prov._parse_replay_input([object()])

    assert (
        prov._resolve_importable_callable("xarray.load_dataarray") is xr.load_dataarray
    )
    with pytest.raises(ValueError, match="must be dotted"):
        prov._resolve_importable_callable("load")
    with pytest.raises(ModuleNotFoundError):
        prov._resolve_importable_callable("missing_erlab_replay_loader.load")
    with pytest.raises(AttributeError):
        prov._resolve_importable_callable("xarray.missing_loader.load")
    with pytest.raises(TypeError, match="not callable"):
        prov._resolve_importable_callable("math.pi")

    broken_module = tmp_path / "broken_loader.py"
    broken_module.write_text(
        "import missing_erlab_replay_dependency\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="missing_erlab_replay_dependency"):
        prov._resolve_importable_callable("broken_loader.load")

    source_file = tmp_path / "source.h5"
    image.to_netcdf(source_file, engine="h5netcdf")
    with pytest.raises(IndexError, match="out of range"):
        prov._load_file_source_data(
            _file_replay_source(
                source_file,
                replay_call=prov.FileReplayCall(
                    kind="callable",
                    target="xarray.load_dataarray",
                    kwargs={"engine": "h5netcdf"},
                    selected_index=1,
                ),
            )
        )
    with pytest.raises(ValueError, match="replay metadata"):
        prov._load_file_source_data(
            prov.FileLoadSource(
                path=source_file,
                loader_label="Load Function",
                loader_text="xarray.load_dataarray",
                kwargs_text="(none)",
            )
        )
    with pytest.raises(TypeError, match="Expected structured file provenance"):
        prov.replay_file_provenance(prov.full_data())
    with pytest.raises(TypeError, match="Expected structured file provenance"):
        prov.replay_file_provenance(typing.cast("typing.Any", None))


def test_file_replay_uses_erlab_loader(example_loader, example_data_dir) -> None:
    del example_loader
    prov = erlab.interactive.imagetool.provenance
    file_path = example_data_dir / "data_002.h5"
    spec = prov.file_load(
        start_label="Load data from file 'data_002.h5'",
        seed_code="import erlab\n\nderived = erlab.io.load(2)",
        file_load_source=_file_replay_source(
            file_path,
            replay_call=prov.FileReplayCall(
                kind="erlab_loader",
                target="example",
                kwargs={},
                selected_index=0,
            ),
        ),
    )

    xr.testing.assert_identical(
        prov.replay_file_provenance(spec),
        erlab.io.loaders["example"].load(file_path),
    )


def test_file_provenance_composes_structured_stages_and_replays_modified_source(
    tmp_path,
) -> None:
    prov = erlab.interactive.imagetool.provenance
    path = tmp_path / "source.h5"
    data = xr.DataArray(
        np.arange(12).reshape((3, 4)),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="scan",
    )
    data.to_netcdf(path, engine="h5netcdf")

    file_spec = prov.file_load(
        start_label=f"Load data from file {path.name!r}",
        seed_code=(
            "import xarray\n\n"
            f"derived = xarray.load_dataarray({str(path)!r}, "
            'engine="h5netcdf").astype("float64")'
        ),
        file_load_source=prov.FileLoadSource(
            path=path,
            loader_label="Load Function",
            loader_text="xarray.load_dataarray",
            kwargs_text='engine="h5netcdf"',
            replay_call=prov.FileReplayCall(
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
    first_stage = prov.full_data(
        prov.AverageOperation(dims=("x",)),
        prov.RenameOperation(name="avg"),
    )
    second_stage = prov.selection(
        prov.IselOperation(kwargs={"y": slice(0, 2)}),
        prov.RenameDimsCoordsOperation(mapping={"y": "energy"}),
        prov.AssignCoordsOperation(
            coord_name="energy",
            values=np.array([10.0, 20.0]),
        ),
    )

    composed = prov.compose_full_provenance(file_spec, first_stage)
    composed = prov.compose_full_provenance(composed, second_stage)

    assert composed is not None
    assert composed.kind == "file"
    assert [stage.source_kind for stage in composed.replay_stages] == [
        "full_data",
        "selection",
    ]
    assert all(
        not isinstance(operation, prov.ScriptCodeOperation)
        for stage in composed.replay_stages
        for operation in stage.operations
    )
    assert composed.display_entries()[0].label == "Load data from file 'source.h5'"
    assert any("Average" in entry.label for entry in composed.display_entries())

    code = composed.display_code()
    assert code is not None
    assert "import xarray" in code
    assert "xarray.load_dataarray" in code
    assert "data =" not in code
    namespace = _exec_generated_code(code, {})
    assert isinstance(namespace["derived"], xr.DataArray)

    updated = data + 100
    updated.to_netcdf(path, engine="h5netcdf")
    live_expected = second_stage.apply(first_stage.apply(updated.astype(np.float64)))
    xr.testing.assert_identical(prov.replay_file_provenance(composed), live_expected)


def test_tool_provenance_compose_display_provenance_streamlines_live_source() -> None:
    prov = erlab.interactive.imagetool.provenance
    parent = prov.script(
        start_label="Start from watched variable 'my_data_name'",
        seed_code="derived = my_data_name",
    )
    source = prov.selection(
        prov.IselOperation(kwargs={"z": 0}),
        prov.SortCoordOrderOperation(),
        prov.SqueezeOperation(),
    )

    composed = prov.compose_display_provenance(
        parent,
        source,
        parent_data=_base_data(),
    )

    assert composed is not None
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
    prov = erlab.interactive.imagetool.provenance
    source = prov.selection(
        prov.IselOperation(kwargs={"z": 0}),
        prov.SortCoordOrderOperation(),
        prov.SqueezeOperation(),
    )

    composed = prov.compose_display_provenance(
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
    prov = erlab.interactive.imagetool.provenance

    watched = prov.script(
        start_label="Start from watched variable 'my_data'",
        seed_code="derived = my_data",
    )
    assert prov.direct_replay_input_name(watched) == "my_data"
    watched_cast = prov.script(
        start_label="Start from watched variable 'my_data'",
        seed_code="derived = my_data.astype(np.float64)",
    )
    assert prov.direct_replay_input_name(watched_cast) == ("my_data.astype(np.float64)")

    assert (
        prov.direct_replay_input_name(
            prov.script(
                start_label="Start from current parent ImageTool data",
                seed_code="derived = data",
            )
        )
        is None
    )
    assert (
        prov.direct_replay_input_name(
            prov.script(
                start_label="Start from watched variable 'my_data'",
                seed_code="derived = data.sel(x=0)",
            )
        )
        is None
    )


def test_tool_provenance_compose_display_replay_omits_synthetic_1d_squeeze() -> None:
    prov = erlab.interactive.imagetool.provenance
    parent = prov.script(
        start_label="Start from watched variable 'my_1d'",
        seed_code="derived = my_1d",
    )
    source = prov.selection(
        prov.SortCoordOrderOperation(),
        prov.SqueezeOperation(),
    )
    parent_data = xr.DataArray(
        np.arange(5).reshape((5, 1)),
        dims=("x", "stack_dim"),
        coords={"x": np.arange(5), "stack_dim": [0]},
    )
    parent_data = prov.mark_promoted_1d_source(parent_data)

    composed = prov.compose_display_provenance(
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
