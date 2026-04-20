import typing

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

import erlab


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


def test_tool_provenance_parse_final_payload_and_reject_unreleased_legacy() -> None:
    prov = erlab.interactive.imagetool.provenance
    payload = {
        "kind": "full_data",
        "operations": [
            {"op": "average", "dims": {prov._TUPLE_MARKER: ["x"]}},
            {"op": "rename", "name": "avg"},
        ],
    }

    spec = prov.parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 1
    assert [op.op for op in spec.operations] == ["average", "rename"]
    assert [entry.label for entry in spec.derivation_entries()] == [
        "Start from current parent ImageTool data",
        'Average(dims=("x",))',
    ]
    assert (
        spec.derivation_code() == 'derived = data\nderived = derived.qsel.average("x")'
    )

    dumped = spec.model_dump(mode="json")
    assert dumped["schema_version"] == 1
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
        data.qsel.average("y"),
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


def test_tool_provenance_preserves_hashable_dims_and_mapping_keys() -> None:
    prov = erlab.interactive.imagetool.provenance
    data = _hashable_data()
    string_key_data = _string_key_data()

    qsel_spec = prov.full_data(
        prov.QSelOperation(kwargs={"k-space": 1.0, "k-space_width": 1.0})
    )
    assert qsel_spec.derivation_code() == (
        "derived = data\n"
        'derived = derived.qsel(**{"k-space": 1.0, "k-space_width": 1.0})'
    )
    xr.testing.assert_identical(
        qsel_spec.apply(string_key_data),
        string_key_data.qsel(**{"k-space": 1.0, "k-space_width": 1.0}),
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
        == 'derived = data\nderived = derived.qsel.average("k-space")'
    )
    xr.testing.assert_identical(
        average_spec.apply(string_key_data), string_key_data.qsel.average("k-space")
    )

    tuple_average_spec = prov.full_data(prov.AverageOperation(dims=(("beta", 0),)))
    assert (
        tuple_average_spec.derivation_code()
        == 'derived = data\nderived = derived.qsel.average((("beta", 0),))'
    )

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
    assert squeezed_code == "derived = data.isel(z=slice(0, 1)).squeeze()"


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

    assert [entry.label for entry in spec.display_entries()] == [
        "Start from current analysis-tool input data",
        "transpose(('x', 'y', 'z'))",
        "squeeze()",
    ]
    assert spec.display_code() == "derived = data.transpose(*('x', 'y', 'z')).squeeze()"


def test_tool_provenance_rejects_unsupported_hashables() -> None:
    class _UnsupportedHashable:
        def __hash__(self) -> int:
            return 0

    with pytest.raises(TypeError, match="provenance hashable fields only support"):
        erlab.interactive.imagetool.provenance.AverageOperation(
            dims=(_UnsupportedHashable(),)
        )


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


def test_tool_replay_provenance_helpers_compose_parent_lineage() -> None:
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
    assert composed.display_code() == "derived = my_data_name.isel(z=0)"


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
    assert composed.display_code() == "derived = data.isel(z=0)"


def test_tool_provenance_direct_replay_input_name_requires_simple_seed() -> None:
    prov = erlab.interactive.imagetool.provenance

    watched = prov.script(
        start_label="Start from watched variable 'my_data'",
        seed_code="derived = my_data",
    )
    assert prov.direct_replay_input_name(watched) == "my_data"

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
    assert composed.display_code() == "derived = my_1d"
