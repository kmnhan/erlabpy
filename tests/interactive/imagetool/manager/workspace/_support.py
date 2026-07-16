import json
import pathlib
import typing
from collections.abc import Mapping

import h5py
import hdf5plugin
import numpy as np
import pydantic
import xarray as xr

import erlab
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
from erlab.interactive.imagetool._provenance._model import (
    FileLoadSource,
    FileReplayCall,
    file_load,
)


class _AddedTimeChildState(pydantic.BaseModel):
    value: int = 0


class _AddedTimeChildTool(erlab.interactive.utils.ToolWindow[_AddedTimeChildState]):
    StateModel = _AddedTimeChildState
    tool_name = "added-time-child"

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = data
        self._status = _AddedTimeChildState()

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> _AddedTimeChildState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _AddedTimeChildState) -> None:
        self._status = status


class _WorkspaceSweepToolState(pydantic.BaseModel):
    value: int = 0
    mode: str = "idle"
    weights: list[float] = pydantic.Field(default_factory=list)
    options: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class _WorkspaceSweepChildTool(
    erlab.interactive.utils.ToolWindow[_WorkspaceSweepToolState]
):
    StateModel = _WorkspaceSweepToolState
    tool_name = "workspace-sweep-child"
    IMAGE_TOOL_OUTPUTS: typing.ClassVar = {
        "workspace-sweep.primary": erlab.interactive.utils.ToolImageOutputDefinition(
            data_method="_primary_output_data",
        )
    }

    def __init__(
        self,
        data: xr.DataArray,
        *,
        extra_data: xr.DataArray | None = None,
    ) -> None:
        super().__init__()
        self._data = data
        self._extra_data = (
            extra_data
            if extra_data is not None
            else data.copy(deep=True).rename("extra")
        )
        self._status = _WorkspaceSweepToolState()

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def extra_data(self) -> xr.DataArray:
        return self._extra_data

    @property
    def tool_status(self) -> _WorkspaceSweepToolState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _WorkspaceSweepToolState) -> None:
        self._status = status

    def _primary_output_data(self) -> xr.DataArray:
        return self._data

    def update_data(self, new_data: xr.DataArray) -> None:
        self._data = new_data

    def _persistence_data_items(self) -> Mapping[str, xr.DataArray]:
        return {
            imagetool_serialization.SAVED_TOOL_DATA_NAME: self._data,
            "auxiliary": self._extra_data,
        }

    def _restore_persistence_data_items(
        self, data_items: Mapping[str, xr.DataArray], ds: xr.Dataset
    ) -> None:
        del ds
        self._extra_data = data_items["auxiliary"]


class _WorkspaceSweepFigureTool(_WorkspaceSweepChildTool):
    manager_collection = "figures"
    tool_name = "workspace-sweep-figure"


class _WorkspaceManagerReferenceFigureTool(_WorkspaceSweepFigureTool):
    tool_name = "workspace-manager-reference-figure"

    def __init__(self, data: xr.DataArray, *, reference_uid: str | None = None) -> None:
        super().__init__(data)
        self._reference_uid = reference_uid

    def _tool_data_reference_payload(
        self, variable_name: str, data: xr.DataArray
    ) -> dict[str, typing.Any] | None:
        del data
        if (
            not self._save_tool_data_references
            or variable_name != imagetool_serialization.SAVED_TOOL_DATA_NAME
            or self._reference_uid is None
        ):
            return None
        allowed_uids = self._save_tool_data_reference_node_uids
        if allowed_uids is not None and self._reference_uid not in allowed_uids:
            return None
        return {"kind": "manager_node", "node_uid": self._reference_uid}


def _workspace_test_file_spec(path: pathlib.Path):
    return file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
        file_load_source=FileLoadSource(
            path=str(path),
            loader_label="xarray.load_dataarray",
            loader_text="xarray.load_dataarray",
            kwargs_text="",
            replay_call=FileReplayCall(
                kind="callable",
                target="xarray.load_dataarray",
                selected_index=0,
            ),
        ),
    )


def _open_external_hdf5_imagetool_data(
    fname: pathlib.Path, *, chunks: str | None = None
) -> xr.DataArray:
    open_kwargs: dict[str, typing.Any] = {
        "engine": "h5netcdf",
        "phony_dims": "sort",
    }
    if chunks is not None:
        open_kwargs["chunks"] = chunks
    tree = xr.open_datatree(fname, **open_kwargs)
    try:
        ds = typing.cast("xr.DataTree", tree["0/imagetool"]).to_dataset(inherit=False)
        return ds[next(iter(ds.data_vars))]
    finally:
        tree.close()


def _open_external_lazy_hdf5_imagetool_data(fname: pathlib.Path) -> xr.DataArray:
    return _open_external_hdf5_imagetool_data(fname, chunks="auto")


def _open_external_file_backed_hdf5_imagetool_data(
    fname: pathlib.Path,
) -> xr.DataArray:
    return _open_external_hdf5_imagetool_data(fname)


def _compute_first_value(darr: xr.DataArray) -> object:
    return darr.isel(dict.fromkeys(darr.dims, 0)).compute().item()


def _hdf5_filter_ids(dataset) -> list[int]:
    create_plist = dataset.id.get_create_plist()
    return [create_plist.get_filter(i)[0] for i in range(create_plist.get_nfilters())]


def _hdf5_blosc2_level_codec(dataset) -> tuple[int, int] | None:
    create_plist = dataset.id.get_create_plist()
    for index in range(create_plist.get_nfilters()):
        filter_id, _flags, cd_values, _name = create_plist.get_filter(index)
        if filter_id == hdf5plugin.Blosc2.filter_id:
            return cd_values[4], cd_values[6]
    return None


def _transaction_test_root_attrs(delta_save_count: int = 0) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 4,
        "root_order": [0],
        "nodes": [],
    }
    if delta_save_count > 0:
        manifest["transaction_protocol"] = (
            workspace_format._WORKSPACE_TRANSACTION_PROTOCOL
        )
        manifest["delta_save_count"] = delta_save_count
    return {
        "imagetool_workspace_schema_version": 4,
        workspace_format._WORKSPACE_MANIFEST_ATTR: json.dumps(manifest),
    }


def _transaction_test_dataset(value: float, *, title: str) -> xr.Dataset:
    ds = xr.Dataset({"data": ("x", np.array([value], dtype=np.float64))})
    ds.attrs["itool_title"] = title
    return ds


def _write_transaction_test_workspace(fname: pathlib.Path, value: float = 1.0) -> None:
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(value, title="old")}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()


def _assert_no_workspace_internal_groups(fname: pathlib.Path) -> None:

    with h5py.File(fname, "r") as h5_file:
        assert not any(
            workspace_format._is_workspace_internal_group_name(name) for name in h5_file
        )


def _rich_workspace_attr_value() -> dict[str, typing.Any]:
    return {
        "X Motor": "EPU Gap",
        "Start": np.float64(20.0),
        "Bidirect": False,
        "raw": b"\xff",
        "points": np.array([1, 2, 3], dtype=np.int16),
        "window": (None, complex(1.0, -2.0)),
    }


def _assert_rich_workspace_attr(value: typing.Any) -> None:
    assert isinstance(value, dict)
    assert value["X Motor"] == "EPU Gap"
    assert value["Start"] == np.float64(20.0)
    assert value["Bidirect"] is False
    assert value["raw"] == b"\xff"
    np.testing.assert_array_equal(value["points"], np.array([1, 2, 3], dtype=np.int16))
    assert value["window"] == (None, complex(1.0, -2.0))


def _request_workspace_save_and_wait(
    qtbot,
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    *,
    native: bool = True,
) -> bool:
    results: list[bool] = []
    requested = manager._workspace_controller.save(
        native=native,
        on_finished=results.append,
    )
    if not requested:
        return bool(results and results[-1])
    qtbot.wait_until(lambda: bool(results), timeout=10000)
    return results[-1]


def _request_workspace_save_as_and_wait(
    qtbot,
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    *,
    native: bool = True,
) -> bool:
    results: list[bool] = []
    requested = manager._workspace_controller.save_as(
        native=native,
        on_finished=results.append,
    )
    if not requested:
        return bool(results and results[-1])
    qtbot.wait_until(lambda: bool(results), timeout=10000)
    return results[-1]
