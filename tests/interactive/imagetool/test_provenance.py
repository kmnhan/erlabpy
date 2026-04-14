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


def test_tool_provenance_codec_and_combinators() -> None:
    edge_fit = xr.Dataset({"edge": ("x", [1.0, 2.0, 3.0])})
    encoded = erlab.interactive.imagetool.provenance.encode_provenance_value(
        {"sel": slice(1.0, 2.0), "data": _base_data(), "edge_fit": edge_fit}
    )
    decoded = erlab.interactive.imagetool.provenance.decode_provenance_value(encoded)

    assert decoded["sel"] == slice(1.0, 2.0)
    xr.testing.assert_identical(decoded["data"], _base_data())
    xr.testing.assert_identical(decoded["edge_fit"], edge_fit)

    spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.average("y")
    ).append_final_rename("avg")
    trimmed = spec.drop_trailing_rename()
    replaced = spec.append_replacement_operations(
        erlab.interactive.imagetool.provenance.thin(factor=2)
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
    payload = {
        "kind": "full_data",
        "operations": [
            {"op": "average", "dims": ["x"]},
            {"op": "rename", "name": "avg"},
        ],
    }

    spec = erlab.interactive.imagetool.provenance.parse_tool_provenance_spec(payload)

    assert spec is not None
    assert spec.schema_version == 1
    assert [op.op for op in spec.operations] == ["average", "rename"]
    assert [entry.label for entry in spec.derivation_entries()] == [
        "Start from current parent ImageTool data",
        "Average(dims=('x',))",
    ]
    assert (
        spec.derivation_code() == 'derived = data\nderived = derived.qsel.average("x")'
    )

    dumped = spec.model_dump(mode="json")
    assert dumped["schema_version"] == 1
    assert dumped["operations"][0]["op"] == "average"

    with pytest.raises(
        ValidationError, match="does not match any of the expected tags"
    ):
        erlab.interactive.imagetool.provenance.parse_tool_provenance_spec(
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
        erlab.interactive.imagetool.provenance.qsel(beta=2.0),
        erlab.interactive.imagetool.provenance.isel(alpha=slice(1, 3)),
        erlab.interactive.imagetool.provenance.sort_coord_order(),
    )
    xr.testing.assert_identical(
        selection_spec.apply(nonuniform),
        nonuniform_public.qsel(beta=2.0).isel({"alpha": slice(1, 3)}),
    )

    transformed = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.isel(z=0),
        erlab.interactive.imagetool.provenance.sel(y=slice(11.0, 12.0)),
        erlab.interactive.imagetool.provenance.transpose("y", "x"),
        erlab.interactive.imagetool.provenance.squeeze(),
        erlab.interactive.imagetool.provenance.rename("done"),
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
            erlab.interactive.imagetool.provenance.average("y")
        ).apply(data),
        data.qsel.average("y"),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            erlab.interactive.imagetool.provenance.coarsen(
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
            erlab.interactive.imagetool.provenance.thin(factor=2)
        ).apply(data),
        data.thin(2),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            erlab.interactive.imagetool.provenance.thin(factors={"x": 2})
        ).apply(data),
        data.thin({"x": 2}),
    )
    xr.testing.assert_identical(
        erlab.interactive.imagetool.provenance.full_data(
            erlab.interactive.imagetool.provenance.swap_dims({"x": "x_alt"})
        ).apply(data),
        data.swap_dims({"x": "x_alt"}),
    )

    assigned = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.assign_coords(
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
        erlab.interactive.imagetool.provenance.rotate(
            angle=45.0, axes=("x", "y"), center=(0.5, 1.5), reshape=False, order=3
        )
    )
    assert rotate_spec.apply(data).attrs["last_op"] == "rotate"

    symmetrize_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.symmetrize(
            dim="x", center=1.0, subtract=True, mode="valid", part="below"
        )
    )
    assert symmetrize_spec.apply(data).attrs["last_op"] == "symmetrize"

    symmetrize_nfold_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.symmetrize_nfold(
            fold=4,
            axes=("x", "y"),
            center={"x": 1.0, "y": 11.0},
            reshape=True,
            order=2,
        )
    )
    assert symmetrize_nfold_spec.apply(data).attrs["last_op"] == "symmetrize_nfold"

    edge_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.correct_with_edge(
            edge_fit=edge_fit, shift_coords=False
        )
    )
    assert edge_spec.apply(data).attrs["last_op"] == "correct_with_edge"
    entries = edge_spec.derivation_entries()
    assert entries[-1].copyable is False
    assert entries[-1].code is None
    assert edge_spec.derivation_code() is None

    path_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.slice_along_path(
            vertices={"x": [0.0, 1.0], "y": [10.0, 12.0]},
            step_size=0.5,
            dim_name="path",
        )
    )
    assert path_spec.apply(data).attrs["last_op"] == "slice_along_path"

    mask_spec = erlab.interactive.imagetool.provenance.full_data(
        erlab.interactive.imagetool.provenance.mask_with_polygon(
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
    assert calls[3][1]["args"] == (edge_fit,)
