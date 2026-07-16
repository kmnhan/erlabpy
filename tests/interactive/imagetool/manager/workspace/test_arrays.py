import contextlib
import json
import logging
import os
import pathlib
import types
import typing
import warnings

import h5py
import hdf5plugin
import numpy as np
import pytest
import xarray
import xarray as xr

import erlab
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from tests.interactive.imagetool.manager.workspace._support import (
    _assert_rich_workspace_attr,
    _hdf5_blosc2_level_codec,
    _hdf5_filter_ids,
    _rich_workspace_attr_value,
)


def test_workspace_h5py_helpers_reject_non_workspace_files(tmp_path) -> None:

    fname = tmp_path / "not-workspace.itws"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_group("0")

    with pytest.raises(ValueError, match="Not a valid workspace file"):
        workspace_arrays._workspace_live_root_group_copy_groups(fname)
    with pytest.raises(ValueError, match="Not a valid workspace file"):
        workspace_arrays._workspace_h5_paths_storage_size(fname, ("0",))
    with pytest.raises(ValueError, match="Not a valid workspace file"):
        workspace_arrays._workspace_live_h5_storage_size(fname)

    assert (
        workspace_storage._workspace_obsolete_estimate(tmp_path / "missing.itws") == 0
    )
    assert workspace_arrays._workspace_h5_object_storage_size(object()) == 0


def test_workspace_h5py_filter_matching_edge_cases(tmp_path) -> None:

    fname = tmp_path / "filters.h5"
    with h5py.File(fname, "w") as h5_file:
        plain = h5_file.create_dataset("plain", data=np.arange(3))
        compressed = h5_file.create_dataset(
            "compressed",
            data=np.arange(3),
            **hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            ),
        )
        group = h5_file.create_group("payload")
        gzip_data = group.create_dataset("data", data=np.arange(3), compression="gzip")
        metadata_group = h5_file.create_group("metadata")
        metadata_group.create_group("nested")

        assert workspace_arrays._workspace_h5py_blosc2_options_match((1, 2), (1, 2))
        assert not workspace_arrays._workspace_h5py_blosc2_options_match((1,), (2,))
        assert workspace_arrays._workspace_h5py_dataset_matches_encoding(plain, {})
        assert not workspace_arrays._workspace_h5py_dataset_matches_encoding(
            plain, {"compression": hdf5plugin.Blosc2.filter_id}
        )
        assert workspace_arrays._workspace_h5py_dataset_matches_encoding(
            compressed, {"compression": hdf5plugin.Blosc2.filter_id}
        )
        assert workspace_arrays._workspace_h5py_dataset_matches_encoding(
            compressed, workspace_arrays._workspace_blosc2_encoding("zstd1")
        )
        assert not workspace_arrays._workspace_h5py_dataset_matches_encoding(
            compressed, workspace_arrays._workspace_blosc2_encoding("blosclz3")
        )
        gzip_filter = workspace_arrays._workspace_h5py_filter_options(gzip_data)
        assert workspace_arrays._workspace_h5py_dataset_matches_encoding(
            gzip_data,
            {"compression": 1, "compression_opts": gzip_filter[1]},
        )
        assert not workspace_arrays._h5_group_matches_compression(
            h5_file, "missing", "none"
        )
        assert not workspace_arrays._h5_group_matches_compression(
            h5_file, "plain", "none"
        )
        assert workspace_arrays._h5_group_matches_compression(
            h5_file, "metadata", "none"
        )
        assert not workspace_arrays._workspace_h5_group_matches_compression_mode(
            h5_file,
            "missing",
            xr.Dataset({"data": ("x", np.arange(3))}),
            "none",
        )
        assert not workspace_arrays._workspace_h5_group_matches_compression_mode(
            h5_file,
            "payload",
            xr.Dataset({"missing": ("x", np.arange(3))}),
            "none",
        )
        assert not workspace_arrays._workspace_h5_group_matches_compression_mode(
            h5_file,
            "payload",
            xr.Dataset({"data": ("x", np.arange(3))}),
            "none",
        )


def test_workspace_h5py_copy_rebuilds_attrs_and_dimension_scales(tmp_path) -> None:

    class _FakeH5Type:
        def __init__(
            self,
            type_class: object,
            *,
            super_type: "_FakeH5Type | None" = None,
            member_types: tuple["_FakeH5Type", ...] = (),
        ) -> None:
            self._type_class = type_class
            self._super_type = super_type
            self._member_types = member_types
            self.closed = False

        def get_class(self) -> object:
            return self._type_class

        def get_super(self) -> "_FakeH5Type":
            if self._super_type is None:
                raise RuntimeError("missing super type")
            return self._super_type

        def get_nmembers(self) -> int:
            return len(self._member_types)

        def get_member_type(self, index: int) -> "_FakeH5Type":
            return self._member_types[index]

        def close(self) -> None:
            self.closed = True

    array_member = _FakeH5Type(h5py.h5t.REFERENCE)
    array_type = _FakeH5Type(h5py.h5t.ARRAY, super_type=array_member)
    assert workspace_arrays._workspace_h5py_type_contains_reference(array_type)
    assert array_member.closed

    plain_member = _FakeH5Type(h5py.h5t.INTEGER)
    compound_type = _FakeH5Type(h5py.h5t.COMPOUND, member_types=(plain_member,))
    assert not workspace_arrays._workspace_h5py_type_contains_reference(compound_type)
    assert plain_member.closed

    fname = tmp_path / "dimension-scales.h5"
    with h5py.File(fname, "w") as h5_file:
        source = h5_file.create_group("source")
        source.create_group("nested")
        source.create_dataset("plain", data=np.arange(2))
        source["plain"].attrs["_Netcdf4Coordinates"] = np.array([0, 1])
        source.create_dataset("scale_without_dimid", data=np.arange(2))
        source["scale_without_dimid"].attrs["CLASS"] = b"DIMENSION_SCALE"
        source["named_type"] = np.dtype("int32")

        scale = source.create_dataset("x", data=np.arange(2))
        scale.attrs["CLASS"] = b"DIMENSION_SCALE"
        scale.attrs["NAME"] = b"x"
        scale.attrs["_Netcdf4Dimid"] = np.int32(0)

        values = source.create_dataset("values", data=np.arange(2))
        values.attrs["_Netcdf4Coordinates"] = np.array([0])
        values_missing_scale = source.create_dataset(
            "values_missing_scale", data=np.arange(2)
        )
        values_missing_scale.attrs["_Netcdf4Coordinates"] = np.array([99])
        source.attrs["reference"] = values.ref
        source.attrs["reference_array"] = np.array([values.ref], dtype=h5py.ref_dtype)
        source.attrs["reference_compound"] = np.array(
            [(values.ref, 1)],
            dtype=np.dtype([("reference", h5py.ref_dtype), ("value", np.int32)]),
        )[0]

        target = h5_file.create_group("target")
        target.create_group("nested")
        target.create_dataset("plain", data=np.arange(2))
        target.create_dataset("scale_without_dimid", data=np.arange(2))
        target["named_type"] = np.dtype("int32")
        target.create_dataset("x", data=np.arange(2))
        target.create_dataset("values", data=np.arange(2))
        target.create_dataset("values_missing_scale", data=np.arange(2))

        assert workspace_arrays._workspace_h5py_attr_text(np.bytes_(b"x")) == "x"
        assert (
            workspace_arrays._workspace_h5py_attr_text(
                types.SimpleNamespace(decode=lambda: "decoded")
            )
            == "decoded"
        )
        assert workspace_arrays._workspace_h5py_attr_text("x") == "x"
        assert workspace_arrays._workspace_h5py_attr_text(object()) is None

        workspace_arrays._workspace_h5py_rebuild_dimension_scales(source, target)

        assert "reference" not in target.attrs
        assert "reference_array" not in target.attrs
        assert "reference_compound" not in target.attrs
        assert target["x"].attrs["_Netcdf4Dimid"] == 0
        assert len(target["values"].dims[0]) == 1


def test_copy_workspace_h5_group_to_open_file_edge_cases(tmp_path) -> None:

    fname = tmp_path / "copy.h5"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_dataset("dataset", data=np.arange(2))
        h5_file.create_group("source").create_dataset("data", data=np.arange(2))
        h5_file.create_group("target").create_group("source")

        assert not workspace_arrays._copy_workspace_h5_group_to_open_file(
            h5_file, h5_file, "missing", "target/missing", None
        )
        assert not workspace_arrays._copy_workspace_h5_group_to_open_file(
            h5_file, h5_file, "dataset", "target/dataset", None
        )
        assert workspace_arrays._copy_workspace_h5_group_to_open_file(
            h5_file,
            h5_file,
            "source",
            "target/source",
            {"title": "copied"},
        )
        assert h5_file["target/source"].attrs["title"] == "copied"


def test_write_workspace_dataset_group_h5py_cleans_failed_independent_items(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "independent-items.itws"
    saved_tool_data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    ds = xr.Dataset(
        {
            saved_tool_data_name: (
                (workspace_arrays._SAVED_TOOL_DATA_REFERENCE_DIM,),
                np.empty(0, dtype=np.float64),
            )
        }
    )
    monkeypatch.setattr(
        workspace_arrays,
        "_workspace_h5py_create_dataset",
        lambda *_args, **_kwargs: None,
    )

    assert not workspace_arrays._write_workspace_dataset_group_h5py(fname, "0/tool", ds)

    with h5py.File(fname, "r") as h5_file:
        assert "0/tool" not in h5_file


def test_workspace_dataset_encoding_compresses_only_large_numeric_payloads() -> None:
    import hdf5plugin

    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            ),
            "small": ("x", np.arange(512, dtype=np.float64)),
            "metadata": ("label", np.array(["a", "b"], dtype=object)),
        },
        coords={
            "x": np.linspace(-1.0, 1.0, 512),
            "y": np.linspace(-2.0, 2.0, 512),
            "label": ["a", "b"],
        },
    )

    encoding = workspace_arrays.workspace_dataset_encoding(ds)

    assert set(encoding) == {"large"}
    assert encoding["large"] == dict(
        hdf5plugin.Blosc2(
            cname="zstd",
            clevel=1,
            filters=hdf5plugin.Blosc2.SHUFFLE,
        )
    )


def test_workspace_dataset_encoding_supports_compression_modes() -> None:
    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        }
    )

    assert (
        workspace_arrays.workspace_dataset_encoding(ds, compression_mode="none") == {}
    )
    assert workspace_arrays.workspace_dataset_encoding(
        ds, compression_mode="blosclz3"
    ) == {
        "large": dict(
            hdf5plugin.Blosc2(
                cname="blosclz",
                clevel=3,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    }
    assert workspace_arrays.workspace_dataset_encoding(
        ds, compression_mode="zstd1"
    ) == {
        "large": dict(
            hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    }
    assert workspace_arrays.workspace_dataset_encoding(ds, compress=True) == {
        "large": dict(
            hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    }
    with pytest.raises(ValueError, match="Unknown workspace compression mode"):
        workspace_arrays.workspace_dataset_encoding(
            ds,
            compression_mode=typing.cast(
                "workspace_arrays.WorkspaceCompressionMode", "missing"
            ),
        )


def test_workspace_dataset_encoding_respects_compression_preference() -> None:
    ds = xr.Dataset(
        {
            "large": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        }
    )
    old_value = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        assert workspace_arrays.workspace_dataset_encoding(ds) == {}

        erlab.interactive.options["io/workspace/compression"] = "blosclz3"
        assert workspace_arrays.workspace_dataset_encoding(ds)["large"] == dict(
            hdf5plugin.Blosc2(
                cname="blosclz",
                clevel=3,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )

        erlab.interactive.options["io/workspace/compression"] = "zstd1"
        assert workspace_arrays.workspace_dataset_encoding(ds)["large"] == dict(
            hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_value


def test_workspace_dataset_encoding_persists_dask_chunksizes() -> None:
    data = xr.DataArray(
        np.arange(25, dtype=np.float64).reshape(5, 5),
        dims=("x", "y"),
    ).chunk({"x": (2, 3), "y": (4, 1)})
    ds = xr.Dataset({"data": data})

    assert workspace_arrays.workspace_dataset_encoding(ds, compress=False) == {
        "data": {"chunksizes": (2, 4)}
    }


def test_workspace_chunksizes_rejects_invalid_chunk_shapes() -> None:
    assert (
        workspace_arrays._workspace_chunksizes_for_dataarray(
            types.SimpleNamespace(chunks=((1,),), ndim=1, shape=(0,))
        )
        is None
    )
    assert (
        workspace_arrays._workspace_chunksizes_for_dataarray(
            types.SimpleNamespace(chunks=((0,),), ndim=1, shape=(5,))
        )
        is None
    )


def test_workspace_xarray_path_helpers_cover_fallbacks(monkeypatch, tmp_path) -> None:
    class _BadPath(os.PathLike):
        def __fspath__(self) -> str:
            raise TypeError

    assert workspace_arrays._normalized_file_path(object()) is None
    assert workspace_arrays._normalized_file_path(_BadPath()) is None
    assert workspace_arrays._normalized_file_path("") is None

    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert workspace_arrays._normalized_file_path(tmp_path / "workspace.itws") == str(
        tmp_path / "workspace.itws"
    )

    monkeypatch.setattr(workspace_arrays, "_normalized_file_path", lambda _path: None)
    lock = workspace_arrays._workspace_file_lock("fallback.itws")
    assert lock is workspace_arrays._workspace_file_lock("fallback.itws")

    def _raise_stat_oserror(_path: str):
        raise OSError

    monkeypatch.setattr(workspace_arrays.os, "stat", _raise_stat_oserror)
    assert workspace_arrays._workspace_file_identity("missing.itws") == (
        "missing.itws",
        0,
        0,
        0,
    )


def test_workspace_file_manager_uses_fsdecode_fallback(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_init(self, opener, *args, **kwargs):
        captured["opener"] = opener
        captured["args"] = args
        captured["kwargs"] = kwargs
        self._key = "fake-key"
        self._ref_counter = types.SimpleNamespace(decrement=lambda _key: None)
        self._cache = {}

    monkeypatch.setattr(
        workspace_arrays, "ensure_workspace_hdf5_filters_registered", lambda: None
    )
    monkeypatch.setattr(workspace_arrays, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(
        workspace_arrays, "_workspace_file_identity", lambda path: (path, 0, 0, 0)
    )
    monkeypatch.setattr(workspace_arrays.CachingFileManager, "__init__", _fake_init)

    file_manager = workspace_arrays.WorkspaceFileManager("fallback.itws")

    assert file_manager.workspace_path == "fallback.itws"
    assert captured["args"][0] == "fallback.itws"
    assert captured["kwargs"]["mode"] == "r+"


def test_open_workspace_dataset_uses_fsdecode_fallback(monkeypatch) -> None:
    calls: list[tuple[object, str, str | None]] = []

    class _FakeFileManager:
        def __init__(self, path: str) -> None:
            self.workspace_path = path

    def _fake_open(file_manager, group: str, *, chunks: str | None):
        calls.append((file_manager, group, chunks))
        return "dataset"

    monkeypatch.setattr(workspace_arrays, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(workspace_arrays, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        workspace_arrays, "_open_workspace_dataset_from_manager", _fake_open
    )

    assert (
        workspace_arrays.open_workspace_dataset("fallback.itws", "/0", chunks=None)
        == "dataset"
    )
    file_manager, group, chunks = calls[0]
    assert isinstance(file_manager, _FakeFileManager)
    assert file_manager.workspace_path == "fallback.itws"
    assert group == "/0"
    assert chunks is None


def test_open_workspace_datatree_closes_partial_groups_on_error(monkeypatch) -> None:
    closed: list[str] = []

    class _FakeDataset:
        def __init__(self, group_path: str) -> None:
            self.group_path = group_path

        def close(self) -> None:
            closed.append(self.group_path)

    class _FakeFileManager:
        workspace_path = "fallback.itws"

        def __init__(self, _path: str) -> None:
            pass

        def acquire_context(self):
            return contextlib.nullcontext(object())

    def _fake_open(_file_manager, group_path: str, *, chunks: str | None):
        if group_path == "/broken":
            raise RuntimeError("broken group")
        return _FakeDataset(group_path)

    monkeypatch.setattr(workspace_arrays, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(workspace_arrays, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        workspace_arrays,
        "_iter_h5netcdf_group_paths",
        lambda _h5_file: ("/", "/broken"),
    )
    monkeypatch.setattr(
        workspace_arrays, "_open_workspace_dataset_from_manager", _fake_open
    )

    with pytest.raises(RuntimeError, match="broken group"):
        workspace_arrays.open_workspace_datatree("fallback.itws", chunks="auto")

    assert closed == ["/"]


def test_open_workspace_datatree_reads_uncompressed_workspace(tmp_path) -> None:
    ds = xr.Dataset(
        {"data": (("x", "y"), np.arange(12, dtype=np.float64).reshape(3, 4))},
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    fname = tmp_path / "uncompressed.itws"
    try:
        tree.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)
    finally:
        tree.close()

    opened = workspace_arrays.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
    finally:
        opened.close()


def test_workspace_h5py_attrs_and_root_validation(tmp_path) -> None:

    assert workspace_arrays._h5py_attrs_to_dict({"name": b"value"}) == {"name": "value"}

    fname = tmp_path / "plain.h5"
    with h5py.File(fname, "w"):
        pass

    with pytest.raises(ValueError, match="Not a valid workspace file"):
        workspace_arrays._read_workspace_root_attrs_h5py(fname)


def test_replace_h5_attrs_drops_invalid_attr_names(tmp_path) -> None:

    fname = tmp_path / "replace-invalid-attrs.itws"
    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("0/imagetool")
        group.attrs["old"] = "removed"

        workspace_arrays._replace_h5_attrs(
            group.attrs,
            {"": "dropped", None: "dropped", "note": "", "valid": "kept"},
        )

        assert "old" not in group.attrs
        assert "" not in list(group.attrs)
        assert group.attrs["note"] == ""
        assert group.attrs["valid"] == "kept"


def test_replace_h5_attrs_encodes_non_native_attr_values(tmp_path) -> None:

    fname = tmp_path / "replace-rich-attrs.itws"
    rich_attr = _rich_workspace_attr_value()
    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("0/imagetool")

        workspace_arrays._replace_h5_attrs(
            group.attrs,
            {"Single Motor Scan": rich_attr, "valid": "kept"},
        )

        assert "Single Motor Scan" not in group.attrs
        assert workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR in group.attrs
        decoded = workspace_arrays._h5py_attrs_to_dict(group.attrs)
        assert decoded["valid"] == "kept"
        _assert_rich_workspace_attr(decoded["Single Motor Scan"])


def _assert_workspace_h5py_roundtrip(
    tmp_path: pathlib.Path, label: str, data: xr.DataArray
) -> tuple[xr.Dataset, xr.Dataset, pathlib.Path]:
    data_name = _ITOOL_DATA_NAME
    fname = tmp_path / f"{label}.itws"
    ds = data.rename(data_name).to_dataset()

    assert workspace_arrays._workspace_dataset_can_write_h5py(ds)
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=data_name,
    )
    assert loaded is not None

    opened = workspace_arrays.open_workspace_dataset(fname, "0/imagetool", chunks=None)
    try:
        opened_loaded = opened.load()
    finally:
        opened.close()
    xr.testing.assert_equal(loaded, opened_loaded)
    return loaded, opened_loaded, fname


def test_workspace_h5py_fast_path_roundtrips_scalar_coords(tmp_path) -> None:

    fname = tmp_path / "scalar-fast-path.itws"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(3.0), "temperature": 20.0},
        attrs={"coordinates": b""},
        name=_ITOOL_DATA_NAME,
    )
    ds = data.to_dataset()

    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=_ITOOL_DATA_NAME,
    )

    assert loaded is not None
    expected = data.copy()
    expected.attrs.pop("coordinates")
    xr.testing.assert_equal(
        loaded[_ITOOL_DATA_NAME],
        expected,
    )
    assert loaded.coords["temperature"].item() == 20.0
    with h5py.File(fname, "r") as h5_file:
        saved_data = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
        coordinates = saved_data.attrs["coordinates"]
    if isinstance(coordinates, bytes):
        coordinates = coordinates.decode()
    assert coordinates == "temperature"


def test_workspace_writer_encodes_saved_tool_spaced_associated_coord(
    tmp_path,
) -> None:
    data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={
            "x": np.arange(2.0),
            "y": np.arange(3.0),
            "Fake Motor": ("x", np.linspace(10.0, 20.0, 2)),
        },
        name=data_name,
    )
    fname = tmp_path / "saved-tool-spaced-coord.itws"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        workspace_arrays._write_workspace_dataset_group_to_file(
            fname, "0/tool", data.to_dataset()
        )

    assert not any("space in its name" in str(item.message) for item in caught)
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "0/tool", preferred_data_name=data_name
    )

    assert loaded is not None
    xr.testing.assert_equal(
        loaded[data_name].coords["Fake Motor"], data.coords["Fake Motor"]
    )


def test_workspace_h5py_fast_path_roundtrips_saved_tool_extra_blob(
    tmp_path,
) -> None:
    import hdf5plugin

    data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    primary = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={
            "x": np.arange(2.0),
            "y": np.arange(3.0),
            "temperature": ("x", np.linspace(100.0, 200.0, 2)),
            "Fake Motor": ("x", np.linspace(10.0, 20.0, 2)),
        },
        name="primary",
    )
    secondary = xr.DataArray(
        np.arange(200_000.0),
        dims=("z",),
        coords={"z": np.linspace(0.0, 1.0, 200_000)},
        name=None,
    )
    ds = primary.to_dataset(name=data_name)
    ds["data_1"] = erlab.interactive.utils._tool_data_to_blob(secondary, "data_1")
    fname = tmp_path / "saved-tool-extra-blob.itws"

    workspace_arrays._write_workspace_dataset_group_to_file(fname, "0/tool", ds)
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/tool",
        preferred_data_name=data_name,
    )

    assert loaded is not None
    assert workspace_arrays._workspace_dataset_can_write_h5py(
        imagetool_serialization.encode_private_coords(ds, data_name)
    )
    with h5py.File(fname, "r") as h5_file:
        assert hdf5plugin.Blosc2.filter_id in _hdf5_filter_ids(h5_file["0/tool/data_1"])
        assert _hdf5_blosc2_level_codec(h5_file["0/tool/data_1"]) == (1, 5)
    xr.testing.assert_identical(loaded[data_name], primary.rename(data_name))
    xr.testing.assert_equal(
        loaded[data_name].coords["Fake Motor"], primary.coords["Fake Motor"]
    )
    restored_secondary = erlab.interactive.utils._tool_data_from_blob(loaded["data_1"])
    xr.testing.assert_equal(restored_secondary, secondary)

    old_value = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "blosclz3"
        blosclz_fname = tmp_path / "blosclz-saved-tool-extra-blob.itws"
        workspace_arrays._write_workspace_dataset_group_to_file(
            blosclz_fname, "0/tool", ds
        )

        erlab.interactive.options["io/workspace/compression"] = "none"
        uncompressed_fname = tmp_path / "uncompressed-saved-tool-extra-blob.itws"
        workspace_arrays._write_workspace_dataset_group_to_file(
            uncompressed_fname, "0/tool", ds
        )
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_value
    with h5py.File(blosclz_fname, "r") as h5_file:
        assert _hdf5_blosc2_level_codec(h5_file["0/tool/data_1"]) == (3, 0)
    with h5py.File(uncompressed_fname, "r") as h5_file:
        assert hdf5plugin.Blosc2.filter_id not in _hdf5_filter_ids(
            h5_file["0/tool/data_1"]
        )


def test_workspace_h5py_fast_path_roundtrips_saved_tool_references(tmp_path) -> None:
    data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
    ds = xr.Dataset(
        {
            data_name: erlab.interactive.utils._tool_data_placeholder(),
            "data_1": erlab.interactive.utils._tool_data_placeholder(),
        },
        attrs={
            erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                {
                    data_name: {"kind": "manager_node", "node_uid": "uid-0"},
                    "data_1": {"kind": "manager_node", "node_uid": "uid-1"},
                }
            )
        },
    )
    fname = tmp_path / "saved-tool-references.itws"

    assert workspace_arrays._write_workspace_dataset_group_h5py(fname, "0/tool", ds)
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/tool",
        preferred_data_name=data_name,
    )

    assert loaded is not None
    assert set(loaded.data_vars) == {data_name, "data_1"}
    reference_dim = erlab.interactive.utils._SAVED_TOOL_DATA_REFERENCE_DIM
    assert loaded[data_name].dims == (reference_dim,)
    assert loaded["data_1"].dims == (reference_dim,)
    assert json.loads(
        loaded.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR]
    ) == json.loads(ds.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR])


def test_workspace_h5py_fast_path_roundtrips_associated_coords_and_xarray(
    tmp_path,
) -> None:

    data_name = _ITOOL_DATA_NAME
    base = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(3.0)},
    )

    divided = base.assign_coords(mesh_current=("x", [1.0, 2.0]))
    divided = divided / divided.coords["mesh_current"]
    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path, "divide-by-coord", divided
    )
    assert loaded.coords["mesh_current"].dims == ("x",)
    np.testing.assert_allclose(loaded.coords["mesh_current"], [1.0, 2.0])

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "two-dimensional-associated-coord",
        base.assign_coords(
            detector_norm=(("x", "y"), np.arange(6.0).reshape(2, 3) + 1.0)
        ),
    )
    assert loaded.coords["detector_norm"].dims == ("x", "y")

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "unicode-scalar-coord",
        base.assign_coords(label="sample"),
    )
    assert loaded.coords["label"].item() == "sample"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "unicode-associated-coord",
        base.assign_coords(label=("x", np.array(["left", "right"]))),
    )
    assert loaded.coords["label"].dtype.kind == "U"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "bytes-associated-coord",
        base.assign_coords(raw=("x", np.array([b"a", b"bb"], dtype="S2"))),
    )
    assert loaded.coords["raw"].dtype.kind == "S"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "datetime-associated-coord",
        xr.DataArray(
            np.arange(2.0),
            dims=("time",),
            coords={
                "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
                "event_time": (
                    "time",
                    np.array(["2024-02-01", "2024-02-02"], dtype="datetime64[D]"),
                ),
            },
        ),
    )
    assert loaded.coords["time"].dtype.kind == "M"
    assert loaded.coords["event_time"].dtype.kind == "M"

    loaded, _opened, _fname = _assert_workspace_h5py_roundtrip(
        tmp_path,
        "timedelta-associated-coord",
        xr.DataArray(
            np.arange(2.0),
            dims=("delay",),
            coords={
                "delay": np.array([0, 5], dtype="timedelta64[ms]"),
                "exposure": (
                    "delay",
                    np.array([1, 2], dtype="timedelta64[s]"),
                ),
            },
        ),
    )
    assert loaded.coords["delay"].dtype == np.dtype("timedelta64[ms]")
    assert loaded.coords["exposure"].dtype == np.dtype("timedelta64[s]")

    with h5py.File(_fname, "r") as h5_file:
        coordinates = h5_file["0/imagetool"][data_name].attrs["coordinates"]
    if isinstance(coordinates, bytes):
        coordinates = coordinates.decode()
    assert coordinates == "exposure"


def test_workspace_h5py_fast_path_keeps_numeric_since_units(tmp_path) -> None:
    data_name = _ITOOL_DATA_NAME
    fname = tmp_path / "numeric-since-units.itws"
    data = xr.DataArray(
        [1.0, 2.0],
        dims=("x",),
        coords={
            "x": [0.0, 1.0],
            "elapsed": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={"units": "seconds since start"},
            ),
        },
        name=data_name,
    )
    ds = data.to_dataset()

    assert workspace_arrays._workspace_dataset_can_write_h5py(ds)
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=data_name,
    )

    assert loaded is not None
    xr.testing.assert_equal(loaded[data_name], data)
    assert loaded.coords["elapsed"].attrs["units"] == "seconds since start"


def test_workspace_writer_roundtrips_non_native_attr_values_from_fast_path(
    tmp_path,
) -> None:

    data_name = _ITOOL_DATA_NAME
    fname = tmp_path / "rich-attrs-fast-path.itws"
    rich_attr = _rich_workspace_attr_value()
    data = xr.DataArray(
        np.arange(2.0),
        dims=("x",),
        coords={
            "x": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={"axis_config": rich_attr},
            ),
            "temperature": xr.DataArray(
                [20.0, 21.0],
                dims=("x",),
                attrs={"scan_config": rich_attr},
            ),
        },
        attrs={"Single Motor Scan": rich_attr},
        name=data_name,
    )
    ds = data.to_dataset()
    ds.attrs["dataset_config"] = rich_attr

    workspace_arrays._write_workspace_dataset_group_to_file(fname, "0/imagetool", ds)

    assert ds.attrs["dataset_config"] is rich_attr
    assert ds[data_name].attrs["Single Motor Scan"] is rich_attr
    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        saved_data = group[data_name]
        assert "dataset_config" not in group.attrs
        assert "Single Motor Scan" not in saved_data.attrs
        assert workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR in group.attrs
        assert workspace_format._WORKSPACE_ENCODED_ATTRS_ATTR in saved_data.attrs

    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "0/imagetool", preferred_data_name=data_name
    )
    assert loaded is not None
    _assert_rich_workspace_attr(loaded.attrs["dataset_config"])
    _assert_rich_workspace_attr(loaded[data_name].attrs["Single Motor Scan"])
    _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])
    _assert_rich_workspace_attr(loaded.coords["temperature"].attrs["scan_config"])

    opened = workspace_arrays.open_workspace_dataset(fname, "0/imagetool", chunks=None)
    try:
        restored = workspace_format._restore_workspace_dataset_attrs(opened.load())
    finally:
        opened.close()
    _assert_rich_workspace_attr(restored.attrs["dataset_config"])
    _assert_rich_workspace_attr(restored[data_name].attrs["Single Motor Scan"])
    _assert_rich_workspace_attr(restored.coords["x"].attrs["axis_config"])


def test_workspace_writer_drops_invalid_attr_names_from_fast_path(tmp_path) -> None:

    data_name = _ITOOL_DATA_NAME
    fname = tmp_path / "invalid-attrs-fast-path.itws"
    data = xr.DataArray(
        np.arange(2.0),
        dims=("x",),
        coords={
            "x": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={"": "dropped", "axis_note": ""},
            ),
            "temperature": xr.DataArray(
                [20.0, 21.0],
                dims=("x",),
                attrs={None: "dropped", "units": "K"},
            ),
        },
        attrs={"": "dropped", 1: "dropped", "note": ""},
        name=data_name,
    )
    ds = data.to_dataset()
    ds.attrs[""] = "dropped"
    ds.attrs["dataset_note"] = ""

    workspace_arrays._write_workspace_dataset_group_to_file(fname, "0/imagetool", ds)

    assert "" in ds.attrs
    assert "" in ds[data_name].attrs
    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        saved_data = group[data_name]

        assert "" not in list(group.attrs)
        assert group.attrs["dataset_note"] == ""
        assert "" not in list(saved_data.attrs)
        assert saved_data.attrs["note"] == ""
        assert "" not in list(group["x"].attrs)
        assert group["x"].attrs["axis_note"] == ""
        assert "" not in list(group["temperature"].attrs)
        assert group["temperature"].attrs["units"] == "K"

    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "0/imagetool", preferred_data_name=data_name
    )
    assert loaded is not None
    assert "" not in loaded.attrs
    assert loaded.attrs["dataset_note"] == ""
    assert "" not in loaded[data_name].attrs
    assert loaded[data_name].attrs["note"] == ""
    assert loaded.coords["temperature"].attrs["units"] == "K"


def test_workspace_writer_drops_invalid_attr_names_from_fallback(tmp_path) -> None:
    fname = tmp_path / "invalid-attrs-fallback.itws"
    rich_attr = _rich_workspace_attr_value()
    ds = xr.Dataset(
        {
            "left": xr.DataArray(
                [1.0, 2.0],
                dims=("x",),
                attrs={
                    "": "dropped",
                    "left_note": "",
                    "Single Motor Scan": rich_attr,
                },
            ),
            "right": ("x", [3.0, 4.0]),
        },
        coords={
            "x": xr.DataArray(
                [0.0, 1.0],
                dims=("x",),
                attrs={None: "dropped", "axis_note": "", "axis_config": rich_attr},
            )
        },
        attrs={"": "dropped", "dataset_note": "", "dataset_config": rich_attr},
    )

    workspace_arrays._write_workspace_dataset_group_to_file(fname, "0/tool", ds)

    opened = xr.open_dataset(fname, group="/0/tool", engine="h5netcdf")
    try:
        loaded = workspace_format._restore_workspace_dataset_attrs(opened.load())
    finally:
        opened.close()

    assert "" in ds.attrs
    assert "" in ds["left"].attrs
    assert "" not in loaded.attrs
    assert loaded.attrs["dataset_note"] == ""
    assert "" not in loaded["left"].attrs
    assert loaded["left"].attrs["left_note"] == ""
    _assert_rich_workspace_attr(loaded["left"].attrs["Single Motor Scan"])
    assert "" not in loaded.coords["x"].attrs
    assert loaded.coords["x"].attrs["axis_note"] == ""
    _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])
    _assert_rich_workspace_attr(loaded.attrs["dataset_config"])


def test_workspace_h5py_fast_path_rejects_invalid_payloads(
    caplog, monkeypatch, tmp_path
) -> None:
    data_name = _ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR

    assert not workspace_arrays._workspace_dataset_can_write_h5py(
        xr.Dataset(
            {
                data_name: ("x", [1.0]),
                "extra": ("x", [2.0]),
            },
            coords={"x": [0.0]},
        )
    )

    missing_private = xr.Dataset({data_name: ("x", [1.0])}, coords={"x": [0.0]})
    missing_private[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "missing", "dims": ["x"]}]
    )
    assert not workspace_arrays._workspace_dataset_can_write_h5py(missing_private)

    bad_private_dims = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "private": ("z", [2.0]),
        },
        coords={"x": [0.0], "z": [0.0]},
    )
    bad_private_dims[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["z"]}]
    )
    assert not workspace_arrays._workspace_dataset_can_write_h5py(bad_private_dims)

    assert not workspace_arrays._workspace_dataset_can_write_h5py(
        xr.Dataset(
            {data_name: ("x", [1.0])},
            coords={"x": np.array([object()], dtype=object)},
        )
    )

    bad_associated_dims = xr.Dataset(
        {data_name: ("x", [1.0])},
        coords={"x": [0.0], "z": [0.0], "bad": ("z", [1.0])},
    )
    assert not workspace_arrays._workspace_dataset_can_write_h5py(bad_associated_dims)

    import dask.array as da

    chunked_coord = xr.Dataset(
        {data_name: ("x", [1.0, 2.0])},
        coords={
            "x": [0.0, 1.0],
            "chunked": ("x", da.from_array(np.array([1.0, 2.0]), chunks=(1,))),
        },
    )
    assert not workspace_arrays._workspace_dataset_can_write_h5py(chunked_coord)

    monkeypatch.setattr(
        workspace_arrays, "_workspace_dataset_can_write_h5py", lambda _ds: True
    )
    assert not workspace_arrays._write_workspace_dataset_group_h5py(
        tmp_path / "no-data-name.itws", "0/imagetool", xr.Dataset()
    )

    bad_attrs = xr.Dataset({data_name: ("x", [1.0])}, coords={"x": [0.0]})
    bad_attrs.attrs["bad"] = object()
    fname = tmp_path / "bad-attrs.itws"
    with caplog.at_level(logging.WARNING, logger=workspace_format.logger.name):
        assert workspace_arrays._write_workspace_dataset_group_h5py(
            fname, "0/imagetool", bad_attrs
        )
    assert "unsupported value type object" in caplog.text

    with h5py.File(fname, "r") as h5_file:
        assert "0/imagetool" in h5_file
        assert "bad" not in h5_file["0/imagetool"].attrs


def test_workspace_h5py_reader_rejects_malformed_groups(tmp_path) -> None:

    data_name = _ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "malformed-reader.itws"

    with h5py.File(fname, "w") as h5_file:
        h5_file.create_dataset("not-a-group", data=np.arange(2.0))
        multi = h5_file.create_group("multi")
        multi.create_dataset("a", data=np.arange(2.0))
        multi.create_dataset("b", data=np.arange(2.0))
        no_dims = h5_file.create_group("no-dims")
        no_dims.create_dataset(data_name, data=np.arange(2.0))
        bad_scale = h5_file.create_group("bad-scale")
        scale = bad_scale.create_dataset("x", data=np.arange(4.0).reshape(2, 2))
        scale.make_scale("x")
        bad_data = bad_scale.create_dataset(data_name, data=np.arange(2.0))
        bad_data.dims[0].attach_scale(scale)
        missing_scalar = h5_file.create_group("missing-scalar")
        x = missing_scalar.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        missing_data = missing_scalar.create_dataset(data_name, data=np.arange(2.0))
        missing_data.dims[0].attach_scale(x)
        missing_data.attrs["coordinates"] = np.bytes_("missing")
        missing_private = h5_file.create_group("missing-private")
        x = missing_private.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        private_data = missing_private.create_dataset(data_name, data=np.arange(2.0))
        private_data.dims[0].attach_scale(x)
        private_data.attrs[private_attr] = json.dumps(
            [{"coord_name": "Fake Motor", "variable_name": "missing", "dims": ["x"]}]
        )
        bad_private = h5_file.create_group("bad-private")
        x = bad_private.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        private_data = bad_private.create_dataset(data_name, data=np.arange(2.0))
        private_data.dims[0].attach_scale(x)
        bad_coord = bad_private.create_dataset("private", data=np.arange(2.0))
        bad_coord.dims[0].attach_scale(x)
        private_data.attrs[private_attr] = json.dumps(
            [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["z"]}]
        )
        bad_associated_no_scale = h5_file.create_group("bad-associated-no-scale")
        x = bad_associated_no_scale.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_associated_no_scale.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        bad_associated_no_scale.create_dataset("associated", data=np.arange(2.0))
        bad_associated_length = h5_file.create_group("bad-associated-length")
        x = bad_associated_length.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_associated_length.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        associated = bad_associated_length.create_dataset(
            "associated", data=np.arange(3.0)
        )
        associated.dims[0].attach_scale(x)
        bad_associated_foreign_dim = h5_file.create_group("bad-associated-foreign-dim")
        x = bad_associated_foreign_dim.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        z = bad_associated_foreign_dim.create_dataset("z", data=np.arange(2.0))
        z.make_scale("z")
        data = bad_associated_foreign_dim.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "associated"
        associated = bad_associated_foreign_dim.create_dataset(
            "associated", data=np.arange(2.0)
        )
        associated.dims[0].attach_scale(z)
        bad_time = h5_file.create_group("bad-time-metadata")
        x = bad_time.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = bad_time.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "time"
        time = bad_time.create_dataset("time", data=np.arange(2, dtype=np.int64))
        time.dims[0].attach_scale(x)
        time.attrs["units"] = "days since not-a-date"
        time.attrs["calendar"] = "proleptic_gregorian"

    assert workspace_arrays._read_workspace_dataset_group_h5py(fname, "missing") is None
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(fname, "not-a-group")
        is None
    )
    assert workspace_arrays._read_workspace_dataset_group_h5py(fname, "multi") is None
    assert workspace_arrays._read_workspace_dataset_group_h5py(fname, "no-dims") is None
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(fname, "bad-scale") is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(fname, "missing-scalar")
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "missing-private", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-private", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-associated-no-scale", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-associated-length", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-associated-foreign-dim", preferred_data_name=data_name
        )
        is None
    )
    assert (
        workspace_arrays._read_workspace_dataset_group_h5py(
            fname, "bad-time-metadata", preferred_data_name=data_name
        )
        is None
    )


def test_workspace_h5py_reader_restores_legacy_spaced_coords(tmp_path) -> None:

    data_name = _ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "legacy-spaced-coord.itws"

    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("valid")
        x = group.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = group.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs["coordinates"] = "missing"
        fake = group.create_dataset("Fake Motor", data=np.arange(2.0) + 10.0)
        fake.dims[0].attach_scale(x)
        duplicate = h5_file.create_group("duplicate")
        x = duplicate.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = duplicate.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        data.attrs[private_attr] = json.dumps(
            [
                {
                    "coord_name": "Fake Motor",
                    "variable_name": "Fake Motor",
                    "dims": ["x"],
                }
            ]
        )
        fake = duplicate.create_dataset("Fake Motor", data=np.arange(2.0) + 20.0)
        fake.dims[0].attach_scale(x)
        invalid = h5_file.create_group("invalid")
        x = invalid.create_dataset("x", data=np.arange(2.0))
        x.make_scale("x")
        data = invalid.create_dataset(data_name, data=np.arange(2.0))
        data.dims[0].attach_scale(x)
        invalid.create_dataset("Fake Motor", data=np.arange(2.0) + 30.0)

    loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "valid", preferred_data_name=data_name
    )
    assert loaded is not None
    np.testing.assert_allclose(loaded.coords["Fake Motor"].values, [10.0, 11.0])
    duplicate_loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "duplicate", preferred_data_name=data_name
    )
    assert duplicate_loaded is not None
    np.testing.assert_allclose(
        duplicate_loaded.coords["Fake Motor"].values, [20.0, 21.0]
    )
    invalid_loaded = workspace_arrays._read_workspace_dataset_group_h5py(
        fname, "invalid", preferred_data_name=data_name
    )
    assert invalid_loaded is not None
    assert "Fake Motor" not in invalid_loaded


def test_workspace_h5py_writer_replaces_groups_and_preserves_attrs(tmp_path) -> None:

    data_name = _ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    fname = tmp_path / "writer-attrs.itws"
    ds = xr.Dataset(
        {
            data_name: (
                ("x", "y"),
                np.arange(4.0).reshape(2, 2),
                {"coordinates": "legacy"},
            ),
            "private": (("x",), np.arange(2.0), {"private_attr": "kept"}),
        },
        coords={
            "x": ("x", np.arange(2.0), {"axis_attr": "x"}),
            "y": ("y", np.arange(2.0), {"axis_attr": "y"}),
            "temperature": ((), 20.0, {"units": "K"}),
        },
    )
    ds[data_name].attrs[private_attr] = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["x"]}]
    )

    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )

    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        assert group["x"].attrs["axis_attr"] == "x"
        assert group["temperature"].attrs["units"] == "K"
        assert group["private"].attrs["private_attr"] == "kept"
        coordinates = group[data_name].attrs["coordinates"]
        if isinstance(coordinates, bytes):
            coordinates = coordinates.decode()
        assert coordinates == "legacy temperature"
