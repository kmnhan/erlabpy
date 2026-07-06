import contextlib
import datetime
import errno
import json
import logging
import os
import pathlib
import sys
import tempfile
import time
import types
import typing
import warnings
import weakref
from collections.abc import Callable, Iterable, Mapping

import h5py
import hdf5plugin
import numpy as np
import pydantic
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._qt_state as qt_state
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager as manager_module
import erlab.interactive.imagetool.manager._console as manager_console
import erlab.interactive.imagetool.manager._desktop as manager_desktop
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._modelview as manager_modelview
import erlab.interactive.imagetool.manager._provenance_edit as manager_provenance_edit
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace as manager_workspace
import erlab.interactive.imagetool.manager._workspace_io as manager_workspace_io
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
import erlab.interactive.imagetool.manager._xarray as manager_xarray
import erlab.interactive.imagetool.plot_items as imagetool_plot_items
import erlab.interactive.imagetool.viewer as imagetool_viewer
from erlab.interactive._fit1d import Fit1DTool
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive._options.schema import AppOptions
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool
from erlab.interactive.imagetool import itool, provenance
from erlab.interactive.imagetool.controls import ItoolColormapControls
from erlab.interactive.imagetool.manager import ImageToolManager, fetch, replace_data
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ChooseFromWorkspaceManifestDialog,
)

from .helpers import (
    _exec_generated_code,
    action_map_by_object_name,
    assert_fit_result_dataset_equivalent,
    assert_fit_result_list_equivalent,
    configure_goldtool_child,
    copy_full_code_for_uid,
    make_fit2d_child,
    select_child_tool,
    select_tools,
    set_transform_launch_mode,
    trigger_menu_action,
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


def _workspace_sweep_json(value: typing.Any) -> typing.Any:
    return json.loads(json.dumps(value))


def _workspace_sweep_spec_payload(
    spec: provenance.ToolProvenanceSpec | None,
) -> dict[str, typing.Any] | None:
    if spec is None:
        return None
    return typing.cast(
        "dict[str, typing.Any]",
        spec.model_dump(mode="json", exclude_none=True),
    )


def _workspace_sweep_binding_payload(
    binding: provenance.ImageToolSelectionSourceBinding | None,
) -> dict[str, typing.Any] | None:
    if binding is None:
        return None
    return typing.cast(
        "dict[str, typing.Any]",
        binding.model_dump(mode="json", exclude_none=True),
    )


def _workspace_sweep_window_state(widget: QtWidgets.QWidget) -> dict[str, typing.Any]:
    payload = qt_state.qt_window_state_payload(widget)
    payload.pop("geometry", None)
    return payload


def _workspace_sweep_data(name: str, *, offset: float = 0.0) -> xr.DataArray:
    x = np.linspace(0.0, 3.0, 4)
    y = np.linspace(-2.0, 2.0, 5)
    z = np.linspace(10.0, 12.0, 3)
    eV = np.linspace(-0.4, 0.4, 2)
    values = np.arange(x.size * y.size * z.size * eV.size, dtype=float).reshape(
        (x.size, y.size, z.size, eV.size)
    )
    return xr.DataArray(
        values + offset,
        dims=("x", "y", "z", "eV"),
        coords={
            "x": x,
            "y": y,
            "z": z,
            "eV": eV,
            "temperature": ("x", np.linspace(100.0, 160.0, x.size)),
            "Fake Motor": ("y", np.linspace(30.0, 50.0, y.size)),
        },
        attrs={"sample": name, "acquisition": {"pass": int(offset)}},
        name=name,
    )


def _configure_workspace_sweep_imagetool(
    tool: erlab.interactive.imagetool.ImageTool,
    *,
    source_file: pathlib.Path,
) -> None:
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_data = tool.slicer_area._data.copy(deep=False)
    source_data.attrs = {"sample": str(source_data.name)}
    source_data.to_netcdf(source_file, engine="h5netcdf")

    tool.setGeometry(33, 44, 540, 430)
    tool.controls_visible = False
    area = tool.slicer_area
    area.add_cursor("#336699", update=False)
    area.set_cursor_colors([QtGui.QColor("#123456"), QtGui.QColor("#abcdef")])
    area.set_current_cursor(1, update=False)
    area.set_value(axis=0, value=1.0, cursor=0, update=False)
    area.set_value(axis=1, value=0.0, cursor=0, update=False)
    area.set_value(axis=2, value=11.0, cursor=1, update=False)
    area.set_bin(axis=0, value=2, cursor=1, update=False)
    area.set_bin(axis=2, value=1, cursor=0, update=False)
    area.array_slicer.snap_to_data = True
    area.array_slicer.twin_coord_names = {"temperature"}
    area.array_slicer._cursor_color_params = (
        ("x",),
        "temperature",
        "viridis",
        False,
        100.0,
        160.0,
    )
    area.set_manual_limits({"x": [0.5, 2.5], "y": [-1.5, 1.5], "z": [10.0, 12.0]})
    area.set_axis_inverted("y", True)
    area.set_colormap(
        "viridis",
        gamma=1.7,
        reverse=True,
        high_contrast=True,
        zero_centered=True,
        levels_locked=True,
        levels=(5.0, 75.0),
        update=False,
    )
    area.main_image.add_roi()
    roi = area.main_image._roi_list[0]
    roi.setPoints([(0.5, -1.0), (1.5, 0.75), (2.5, -0.25)], closed=True)
    area.main_image._restore_guideline_state(
        {"count": 2, "angle": 33.0, "offset": (1.25, 0.5), "follow_cursor": False}
    )
    area._file_path = source_file
    area._load_func = (
        xr.load_dataarray,
        {"engine": "h5netcdf"},
        provenance.FileDataSelection(kind="dataarray"),
    )
    area.apply_filter_operation(
        provenance.GaussianFilterOperation(sigma={"x": 0.5}), update=False
    )


def _workspace_sweep_node_snapshot(
    node: typing.Any,
) -> dict[str, typing.Any]:
    if node.pending_workspace_payload is not None:
        assert node.materialize_pending_workspace_payload()
    window = node.window
    assert isinstance(window, QtWidgets.QWidget)
    snapshot: dict[str, typing.Any] = {
        "uid": node.uid,
        "parent_uid": node.parent_uid,
        "children": list(node._childtool_indices),
        "name": node.name,
        "display_text": node.display_text,
        "created_time": node.added_time_iso,
        "snapshot_token": node.snapshot_token,
        "is_imagetool": bool(node.is_imagetool),
        "visible": bool(window.isVisible()),
        "window_state": _workspace_sweep_window_state(window),
        "provenance_spec": _workspace_sweep_spec_payload(node.provenance_spec),
        "source_spec": _workspace_sweep_spec_payload(node.source_spec),
        "source_binding": _workspace_sweep_binding_payload(node.source_binding),
        "output_id": node.output_id,
        "source_state": node.source_state,
        "source_auto_update": node.source_auto_update,
    }
    if hasattr(node, "watched_metadata"):
        snapshot["watched_metadata"] = _workspace_sweep_json(node.watched_metadata())
        source_input_ndim = getattr(node, "source_input_ndim", None)
        snapshot["source_input_ndim"] = (
            None if source_input_ndim is None else int(source_input_ndim)
        )
    if node.is_imagetool:
        snapshot["imagetool_state"] = _workspace_sweep_json(node.slicer_area.state)
        snapshot["imagetool_data_name"] = node.slicer_area._data.name
    else:
        tool = typing.cast("_WorkspaceSweepChildTool", node.tool_window)
        snapshot["tool_status"] = tool.tool_status.model_dump(mode="json")
        snapshot["tool_data_name"] = tool.tool_data.name
        snapshot["tool_display_name"] = tool._tool_display_name
    return snapshot


def _workspace_sweep_runtime_snapshot(
    manager: ImageToolManager,
) -> dict[str, typing.Any]:
    controller = manager._workspace_controller
    return {
        "root_order": list(manager._workspace_root_indices()),
        "figure_uids": list(manager._tool_graph.figure_uids),
        "loader_state": controller._workspace_loader_state_snapshot(),
        "standalone_apps": controller._workspace_standalone_apps_snapshot(),
        "nodes": {
            uid: _workspace_sweep_node_snapshot(node)
            for uid, node in sorted(manager._tool_graph.nodes.items())
        },
    }


def _workspace_sweep_data_items(
    manager: ImageToolManager,
) -> dict[tuple[str, str], xr.DataArray]:
    data_items: dict[tuple[str, str], xr.DataArray] = {}
    for uid, node in sorted(manager._tool_graph.nodes.items()):
        if node.pending_workspace_payload is not None:
            assert manager._materialize_pending_workspace_payload(node)
        if node.is_imagetool:
            data, _state = node.slicer_area.persistence_data_and_state()
            data_items[(uid, "imagetool")] = data.copy(deep=True)
            continue
        tool = typing.cast("_WorkspaceSweepChildTool", node.tool_window)
        for name, data in tool._persistence_data_items().items():
            data_items[(uid, name)] = data.copy(deep=True)
    return data_items


def _workspace_sweep_assert_data_items_equal(
    actual: dict[tuple[str, str], xr.DataArray],
    expected: dict[tuple[str, str], xr.DataArray],
) -> None:
    assert set(actual) == set(expected)
    for key, expected_data in expected.items():
        try:
            xr.testing.assert_identical(actual[key], expected_data)
        except AssertionError as exc:
            raise AssertionError(f"{key!r} data mismatch") from exc


def _workspace_sweep_assert_snapshot_equal(
    actual: typing.Any, expected: typing.Any, path: str = "snapshot"
) -> None:
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), (
            f"{path}: expected mapping, got {type(actual).__name__}"
        )
        assert set(actual) == set(expected), (
            f"{path}: key mismatch "
            f"missing={sorted(set(expected) - set(actual))!r} "
            f"extra={sorted(set(actual) - set(expected))!r}"
        )
        for key in expected:
            _workspace_sweep_assert_snapshot_equal(
                actual[key], expected[key], f"{path}.{key}"
            )
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), (
            f"{path}: expected list, got {type(actual).__name__}"
        )
        assert len(actual) == len(expected), (
            f"{path}: length mismatch {len(actual)} != {len(expected)}"
        )
        for index, expected_item in enumerate(expected):
            _workspace_sweep_assert_snapshot_equal(
                actual[index], expected_item, f"{path}[{index}]"
            )
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _workspace_sweep_h5_attr_payload(
    fname: pathlib.Path, payload_path: str
) -> dict[str, typing.Any]:
    import h5py

    json_attrs = {
        "itool_state",
        "itool_window_state",
        "itool_provenance_spec",
        "manager_node_provenance_spec",
        "manager_node_live_source_spec",
        "manager_node_live_source_binding",
        "tool_state",
        "tool_window_state",
        "tool_source_spec",
        "tool_source_binding",
        "tool_input_provenance_spec",
        "tool_data_references",
    }
    with h5py.File(fname, "r") as h5_file:
        attrs = dict(h5_file[payload_path].attrs)
    payload: dict[str, typing.Any] = {}
    for key, value in attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        if key in json_attrs and isinstance(value, str):
            decoded = json.loads(value)
            if key in {"itool_window_state", "tool_window_state"}:
                decoded.pop("geometry", None)
            payload[key] = decoded
        else:
            payload[key] = value
    return payload


def test_tool_data_blob_ignores_stale_backend_encoding() -> None:
    data = xr.DataArray(
        np.arange(3.0),
        dims=("x",),
        coords={"x": [0.0, 1.0, 2.0]},
        name="secondary",
    )
    data.encoding["compression"] = "unknown"
    data.encoding["source"] = "stale-source.nc"
    data.coords["x"].encoding["compression"] = "unknown"

    blob = erlab.interactive.utils._tool_data_to_blob(data, "secondary")
    restored = erlab.interactive.utils._tool_data_from_blob(blob)

    xr.testing.assert_equal(restored, data)
    assert data.encoding["compression"] == "unknown"
    assert data.coords["x"].encoding["compression"] == "unknown"


def test_tool_data_blob_preserves_none_name() -> None:
    data = xr.DataArray(np.arange(3.0), dims=("x",))

    blob = erlab.interactive.utils._tool_data_to_blob(data, "secondary")
    restored = erlab.interactive.utils._tool_data_from_blob(blob)

    assert restored.name is None


def _wait_for_fit_idle(qtbot, tool: Fit1DTool, *, timeout: int = 10000) -> None:
    def _fit_idle() -> bool:
        if tool._fit_thread is not None:
            return False
        if isinstance(tool, Fit2DTool):
            return tool._fit_2d_total == 0 and not tool._fit_2d_indices
        return True

    qtbot.wait_until(_fit_idle, timeout=timeout)


def _workspace_test_file_spec(path: pathlib.Path):
    return provenance.file_load(
        start_label="Load source",
        seed_code=f"derived = xr.load_dataarray({str(path)!r})",
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


def _assert_manual_limits_view_ranges(
    area: typing.Any, expected_limits: dict[str, list[float]]
) -> None:
    matched_dims: set[str] = set()
    for plot_item in area.axes:
        view_ranges = plot_item.getViewBox().viewRange()
        for axis, axis_dim in enumerate(plot_item.axis_dims_uniform):
            if axis_dim in expected_limits:
                matched_dims.add(axis_dim)
                np.testing.assert_allclose(view_ranges[axis], expected_limits[axis_dim])
    assert matched_dims == set(expected_limits)


def test_workspace_file_suffix_helpers_collect_nested_inputs(tmp_path) -> None:
    first = _workspace_test_file_spec(tmp_path / "scan_a.h5")
    second = _workspace_test_file_spec(tmp_path / "scan_b.h5")
    third = _workspace_test_file_spec(tmp_path / "scan_c.h5")
    nested = provenance.script(
        start_label="Combine",
        seed_code="derived = data_0 + data_1",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(name="data_1", label="B", provenance_spec=second),
            provenance.ScriptInput(name="data_2", label="C", provenance_spec=third),
            provenance.ScriptInput(
                name="data_0", label="A duplicate", provenance_spec=first
            ),
        ),
    )
    combined = provenance.script(
        start_label="Combine nested",
        seed_code="derived = data_0",
        active_name="derived",
        script_inputs=(
            provenance.ScriptInput(name="data_0", label="A", provenance_spec=first),
            provenance.ScriptInput(
                name="nested", label="Nested", provenance_spec=nested
            ),
        ),
    )

    stems = manager_workspace_io._workspace_provenance_file_stems(combined)

    assert stems == ("scan_a", "scan_b", "scan_c")
    assert (
        manager_workspace_io._workspace_compact_file_suffix(stems)
        == " (scan_a, scan_b, +1)"
    )


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"itool_title": "2: manual (scan)", "itool_name": "scan"}, "manual"),
        ({"itool_title": "scan", "itool_name": ""}, None),
        ({"itool_title": "scan (scan)", "itool_name": "scan"}, None),
    ],
)
def test_workspace_legacy_title_migration_ignores_generated_file_labels(
    tmp_path,
    attrs,
    expected,
) -> None:
    ds = xr.Dataset(attrs=attrs)
    spec = _workspace_test_file_spec(tmp_path / "scan.h5")

    assert manager_workspace_io._legacy_saved_title_data_name(ds, spec) == expected


def test_manager_duplicate(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        manager.rename_imagetool(0, "renamed source")

        select_tools(manager, [0, 1])
        manager.duplicate_selected()
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=5000)

        # Check if the duplicated tools have the same data
        for i in range(2):
            original_tool = manager.get_imagetool(i)
            duplicated_tool = manager.get_imagetool(i + 2)

            assert original_tool.slicer_area._data.equals(
                duplicated_tool.slicer_area._data
            )
            assert (
                manager._tool_graph.root_wrappers[i].name
                == manager._tool_graph.root_wrappers[i + 2].name
            )


def test_manager_workspace_saves_added_time_for_all_node_kinds(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    root_added = datetime.datetime(
        2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=9))
    )
    child_added = datetime.datetime(
        2024, 1, 3, 4, 5, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))
    )
    tool_added = datetime.datetime(2024, 1, 4, 5, 6, 7, tzinfo=datetime.UTC)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root_index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            show=False,
            created_time=root_added,
        )
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(test_data + 1, _in_manager=True),
            root_index,
            show=False,
            created_time=child_added,
        )
        tool_uid = manager.add_childtool(
            _AddedTimeChildTool(test_data),
            root_index,
            show=False,
            created_time=tool_added,
        )

        fname = tmp_path / "added-time.itws"
        manager._save_workspace_document(fname, force_full=True)

    with h5py.File(fname, "r") as h5_file:
        assert h5_file["0/imagetool"].attrs[
            "manager_node_added_at"
        ] == root_added.isoformat(timespec="seconds")
        assert h5_file[f"0/childtools/{child_uid}/imagetool"].attrs[
            "manager_node_added_at"
        ] == child_added.isoformat(timespec="seconds")
        assert h5_file[f"0/childtools/{tool_uid}/tool"].attrs[
            "manager_node_added_at"
        ] == tool_added.isoformat(timespec="seconds")


def test_manager_workspace_load_preserves_added_time(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    root_added = datetime.datetime(
        2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=9))
    )
    child_added = datetime.datetime(
        2024, 1, 3, 4, 5, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=-5))
    )
    tool_added = datetime.datetime(2024, 1, 4, 5, 6, 7, tzinfo=datetime.UTC)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root_index = manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            show=False,
            created_time=root_added,
        )
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(test_data + 1, _in_manager=True),
            root_index,
            show=False,
            created_time=child_added,
        )
        tool_uid = manager.add_childtool(
            _AddedTimeChildTool(test_data),
            root_index,
            show=False,
            created_time=tool_added,
        )

        fname = tmp_path / "added-time-load.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )

        assert manager._tool_graph.root_wrappers[root_index].created_time == root_added
        assert manager._child_node(child_uid).created_time == child_added
        assert manager._child_node(tool_uid).created_time == tool_added


def test_manager_workspace_option_overrides_roundtrip_and_mark_dirty(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    overrides = {
        "colors/cmap/name": "viridis",
        "colors/max_rendered_abs_value": 12.0,
        "io/workspace/compression": "none",
        "figure/stylesheets": ["classic", "missing-style"],
    }

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            show=False,
        )
        manager._mark_workspace_clean()

        manager._set_workspace_option_overrides(overrides)

        assert manager.is_workspace_modified
        assert manager._workspace_state.options_modified
        assert manager.workspace_option_overrides() == overrides

        fname = tmp_path / "option-overrides.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._set_workspace_option_overrides({})
        manager._mark_workspace_clean()

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        assert manager.workspace_option_overrides() == overrides
        assert not manager._workspace_state.options_modified
        assert not manager.is_workspace_modified


def test_manager_workspace_option_override_helpers(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._mark_workspace_clean()

        manager.set_workspace_option_override("colors/cmap/name", "viridis")

        assert manager.workspace_option_overrides() == {"colors/cmap/name": "viridis"}
        assert manager.is_workspace_modified
        assert manager._workspace_state.options_modified

        manager.clear_workspace_option_override("colors/cmap/name")
        assert manager.workspace_option_overrides() == {}

        manager.clear_workspace_option_override("colors/cmap/name")
        assert manager.workspace_option_overrides() == {}


def test_manager_workspace_load_warns_for_unavailable_colormap(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    missing = "__erlab_missing_colormap__"
    dialogs: list[typing.Any] = []

    class _RecordingMessageDialog(QtWidgets.QDialog):
        def __init__(self, parent=None, **kwargs) -> None:
            super().__init__(parent)
            self.parent = parent
            self.kwargs = kwargs
            self._button_box = QtWidgets.QDialogButtonBox(self)
            dialogs.append(self)

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

        def text(self) -> str:
            return str(self.kwargs.get("text", ""))

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        tool = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        tool.slicer_area.set_colormap("viridis", gamma=1.5, reverse=True)
        manager.add_imagetool(tool, show=False)

        fname = tmp_path / "missing-cmap.itws"
        manager._save_workspace_document(fname, force_full=True)

        with h5py.File(fname, "a") as h5_file:
            state = json.loads(h5_file["0/imagetool"].attrs["itool_state"])
            state["color"]["cmap"] = missing
            h5_file["0/imagetool"].attrs["itool_state"] = json.dumps(state)

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )

        props = manager.get_imagetool(0).slicer_area.colormap_properties
        assert props["cmap"] == erlab.interactive.options.model.colors.cmap.name
        assert props["gamma"] == pytest.approx(1.5)
        assert props["reverse"] is True

        colormap_dialogs = [
            dialog
            for dialog in dialogs
            if dialog.kwargs.get("title") == "Unavailable Colormap"
        ]
        assert len(colormap_dialogs) == 1
        assert colormap_dialogs[0].parent is manager
        assert missing in colormap_dialogs[0].kwargs["informative_text"]
        assert "0" in colormap_dialogs[0].kwargs["informative_text"]


def test_standalone_imagetool_restore_unavailable_colormap_has_no_manager_dialog(
    qtbot,
    test_data,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = "__erlab_missing_colormap__"
    dialogs: list[typing.Any] = []

    class _RecordingMessageDialog:
        def __init__(self, *args, **kwargs) -> None:
            dialogs.append((args, kwargs))

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )

    tool = erlab.interactive.imagetool.ImageTool(test_data)
    qtbot.addWidget(tool)
    ds = tool.to_dataset()
    state = json.loads(ds.attrs["itool_state"])
    state["color"]["cmap"] = missing
    ds.attrs["itool_state"] = json.dumps(state)

    with pytest.warns(UserWarning, match="Failed to restore colormap settings"):
        restored = erlab.interactive.imagetool.ImageTool.from_dataset(ds)
    qtbot.addWidget(restored)

    assert dialogs == []


def test_manager_added_time_display_uses_zone_name_and_offset(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    added = datetime.datetime(
        2024,
        1,
        2,
        3,
        4,
        5,
        tzinfo=datetime.timezone(datetime.timedelta(hours=9), "KST"),
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            show=False,
            created_time=added,
        )

        node = manager._tool_graph.root_wrappers[0]
        expected = added.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%z)")
        assert node.added_time_display == expected
        assert node.added_time_iso == added.isoformat(timespec="seconds")
        assert (
            next(
                field.value for field in node.metadata_fields if field.label == "Added"
            )
            == expected
        )
        assert f"Added {expected}" in node.info_text


@pytest.mark.parametrize("saved_attr", [None, "not-a-date"], ids=["missing", "invalid"])
def test_manager_workspace_load_falls_back_for_legacy_or_invalid_added_time(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    saved_attr: str | None,
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True),
            show=False,
        )
        fname = tmp_path / "legacy-added-time.itws"
        manager._save_workspace_document(fname, force_full=True)

        with h5py.File(fname, "a") as h5_file:
            if saved_attr is None:
                del h5_file["0/imagetool"].attrs["manager_node_added_at"]
            else:
                h5_file["0/imagetool"].attrs["manager_node_added_at"] = saved_attr

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )

        loaded = manager._tool_graph.root_wrappers[0].created_time
        assert loaded.tzinfo is not None
        assert loaded.utcoffset() is not None


def test_qt_bytearray_base64_helpers_reject_invalid_values() -> None:
    value = QtCore.QByteArray(b"layout-state")
    encoded = erlab.interactive.utils._qt_bytearray_to_base64(value)

    decoded = erlab.interactive.utils._qt_bytearray_from_base64(encoded)
    assert decoded == value

    assert erlab.interactive.utils._qt_bytearray_from_base64(b"\xff") is None
    assert erlab.interactive.utils._qt_bytearray_from_base64("%%not-base64%%") is None
    assert erlab.interactive.utils._qt_bytearray_from_base64("") is None


def test_qt_window_state_helpers_parse_invalid_and_restore_rect(qtbot) -> None:
    assert qt_state.QtWindowState.model_validate({"rect": None}).rect is None
    with pytest.raises(pydantic.ValidationError):
        qt_state.QtWindowState.model_validate({"rect": [1, 2, 3]})

    assert qt_state.qt_bytearray_from_base64(object()) is None
    assert qt_state.parse_qt_window_state(b"\xff") is None
    assert qt_state.parse_qt_window_state("{") is None
    assert qt_state.parse_qt_window_state({"rect": [1, 2, 3]}) is None

    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    assert qt_state.restore_qt_window_state(
        widget, {"geometry": "", "rect": [10, 20, 123, 45]}
    )
    assert widget.geometry().getRect() == (10, 20, 123, 45)


def test_imagetool_dataset_uses_window_state_and_keeps_rect_fallback(
    qtbot, test_data
) -> None:
    tool = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
    qtbot.addWidget(tool)
    tool.resize(533, 477)
    ds = tool.to_dataset()
    window_state = json.loads(ds.attrs["itool_window_state"])
    assert window_state["geometry"]
    assert tuple(window_state["rect"]) == tuple(
        int(value) for value in tool.geometry().getRect()
    )
    assert "itool_qt_geometry" not in ds.attrs
    assert "itool_rect" not in ds.attrs

    restored = erlab.interactive.imagetool.ImageTool.from_dataset(ds, _in_manager=True)
    qtbot.addWidget(restored)
    assert restored.size() == tool.size()

    legacy_ds = ds.copy(deep=False)
    del legacy_ds.attrs["itool_window_state"]
    legacy_ds.attrs["itool_rect"] = tuple(
        int(value) for value in tool.geometry().getRect()
    )
    legacy_ds.attrs["itool_visible"] = True
    legacy = erlab.interactive.imagetool.ImageTool.from_dataset(
        legacy_ds, _in_manager=True
    )
    qtbot.addWidget(legacy)
    assert tuple(legacy.geometry().getRect()) == tuple(
        int(value) for value in legacy_ds.attrs["itool_rect"]
    )


def test_toolwindow_dataset_uses_window_state_and_keeps_rect_fallback(
    qtbot, test_data
) -> None:
    tool = _AddedTimeChildTool(test_data)
    qtbot.addWidget(tool)
    tool.resize(421, 333)
    ds = tool.to_dataset()
    window_state = json.loads(ds.attrs["tool_window_state"])
    assert window_state["geometry"]
    assert tuple(window_state["rect"]) == tuple(
        int(value) for value in tool.geometry().getRect()
    )
    assert "tool_qt_geometry" not in ds.attrs
    assert "tool_rect" not in ds.attrs

    restored = erlab.interactive.utils.ToolWindow.from_dataset(ds)
    qtbot.addWidget(restored)
    assert restored.size() == tool.size()

    legacy_ds = ds.copy(deep=False)
    del legacy_ds.attrs["tool_window_state"]
    legacy_ds.attrs["tool_rect"] = tuple(
        int(value) for value in tool.geometry().getRect()
    )
    legacy_ds.attrs["tool_visible"] = True
    legacy = erlab.interactive.utils.ToolWindow.from_dataset(legacy_ds)
    qtbot.addWidget(legacy)
    assert tuple(legacy.geometry().getRect()) == tuple(
        int(value) for value in legacy_ds.attrs["tool_rect"]
    )


def test_workspace_backing_uses_persistence_data_for_filtered_file_data(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    data = xr.DataArray(
        np.arange(25, dtype=float).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5, dtype=float), "y": np.arange(5, dtype=float)},
        name="scan",
    )
    data.to_netcdf(file_path, engine="h5netcdf")
    operation = erlab.interactive.imagetool.provenance.GaussianFilterOperation(
        sigma={"x": 1.0}
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        opened = xr.open_dataarray(file_path, engine="h5netcdf")
        try:
            itool(opened, manager=True)
            qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
            tool = manager.get_imagetool(0)
            tool.slicer_area.apply_filter_operation(operation, emit_edited=True)

            uid = manager._tool_graph.root_wrappers[0].uid

            def _fail_persistence_data_and_state(_self):
                raise AssertionError("metadata snapshots must not capture full state")

            with monkeypatch.context() as metadata_patch:
                metadata_patch.setattr(
                    imagetool_viewer.ImageSlicerArea,
                    "persistence_data_and_state",
                    _fail_persistence_data_and_state,
                )
                entry = next(
                    item
                    for item in manager._workspace_node_manifest_entries()
                    if item["uid"] == uid
                )
                snapshot = manager._workspace_data_backing_snapshot()

            assert entry["data_backing"] == "file_lazy"
            assert snapshot[uid][0] == "file_lazy"
            assert str(file_path.resolve()) in snapshot[uid][1]
        finally:
            opened.close()


def test_workspace_lightweight_backing_metadata_classifies_data_without_state(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "lazy.h5"
    file_data = xr.DataArray(
        np.arange(16, dtype=float).reshape((4, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(4)},
        name="lazy",
    )
    file_data.to_netcdf(file_path, engine="h5netcdf")

    opened = xr.open_dataarray(file_path, engine="h5netcdf")
    try:
        with manager_context() as manager:
            manager.show()
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

            memory_tool = itool(
                xr.DataArray(np.arange(9).reshape((3, 3)), dims=["x", "y"]),
                manager=False,
                execute=False,
            )
            dask_tool = itool(
                xr.DataArray(np.arange(16).reshape((4, 4)), dims=["x", "y"]).chunk(
                    {"x": 2, "y": 2}
                ),
                manager=False,
                execute=False,
                auto_compute=False,
            )
            file_tool = itool(opened, manager=False, execute=False)
            assert isinstance(memory_tool, erlab.interactive.imagetool.ImageTool)
            assert isinstance(dask_tool, erlab.interactive.imagetool.ImageTool)
            assert isinstance(file_tool, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(memory_tool, show=False)
            manager.add_imagetool(dask_tool, show=False)
            manager.add_imagetool(file_tool, show=False)

            memory_uid = manager._tool_graph.root_wrappers[0].uid
            dask_uid = manager._tool_graph.root_wrappers[1].uid
            file_uid = manager._tool_graph.root_wrappers[2].uid

            def _fail_persistence_data_and_state(_self):
                raise AssertionError("metadata snapshots must not capture full state")

            with monkeypatch.context() as metadata_patch:
                metadata_patch.setattr(
                    imagetool_viewer.ImageSlicerArea,
                    "persistence_data_and_state",
                    _fail_persistence_data_and_state,
                )
                entries = {
                    entry["uid"]: entry
                    for entry in manager._workspace_node_manifest_entries()
                }
                snapshot = manager._workspace_data_backing_snapshot()

            assert entries[memory_uid]["data_backing"] == "memory"
            assert entries[dask_uid]["data_backing"] == "dask"
            assert entries[file_uid]["data_backing"] == "file_lazy"
            assert snapshot[memory_uid] == ("memory", ())
            assert snapshot[dask_uid] == ("dask", ())
            assert snapshot[file_uid][0] == "file_lazy"
            assert str(file_path.resolve()) in snapshot[file_uid][1]
    finally:
        opened.close()


def test_manager_duplicate_goldtool_child(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)
        child.open_itool()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)

        select_child_tool(manager, child_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 2,
            timeout=5000,
        )

        duplicate_uid = next(
            uid
            for uid in manager._tool_graph.root_wrappers[0]._childtool_indices
            if uid != child_uid
        )
        duplicated = manager.get_childtool(duplicate_uid)

        assert isinstance(duplicated, GoldTool)
        assert duplicated is not child
        assert duplicated.tool_status == child.tool_status
        xr.testing.assert_identical(duplicated.corrected, child.corrected)

        duplicate_node = manager._child_node(duplicate_uid)
        qtbot.wait_until(
            lambda: len(duplicate_node._childtool_indices) == 1, timeout=5000
        )
        duplicate_output_uid = duplicate_node._childtool_indices[0]
        duplicate_output_node = manager._child_node(duplicate_output_uid)
        assert duplicate_output_node.output_id == "goldtool.corrected"
        assert duplicate_output_node.source_spec is None
        assert duplicate_output_node.provenance_spec is not None
        xr.testing.assert_identical(fetch(duplicate_output_uid), duplicated.corrected)

        monkeypatch.setattr(
            duplicated, "_prompt_existing_output_imagetool", lambda: "update"
        )
        duplicated.open_itool()
        assert duplicate_node._childtool_indices == [duplicate_output_uid]


def test_manager_sync(
    qtbot,
    move_and_compare_values,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=True, link_colors=True, manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2)

        win0, win1 = manager.get_imagetool(0), manager.get_imagetool(1)
        win0.show()
        win1.show()

        win1.slicer_area.set_colormap("RdYlBu", gamma=1.5)
        assert (
            win0.slicer_area._colormap_properties
            == win1.slicer_area._colormap_properties
        )

        move_and_compare_values(qtbot, win0, [12.0, 7.0, 6.0, 11.0], target_win=win1)

        # Transpose
        win0.slicer_area.transpose_main_image()
        move_and_compare_values(qtbot, win0, [12.0, 11.0, 6.0, 7.0], target_win=win1)

        # Set bin
        win1.slicer_area.set_bin(0, 2, update=False)
        win1.slicer_area.set_bin(1, 2, update=True)

        # Set all bins, same effect as above since we only have 1 cursor
        win1.slicer_area.set_bin_all(1, 2, update=True)

        move_and_compare_values(qtbot, win0, [9.0, 8.0, 3.0, 4.0], target_win=win1)

        # Change limits
        win0.slicer_area.main_image.getViewBox().setRange(xRange=[2, 3], yRange=[1, 2])
        # Trigger manual range propagation
        win0.slicer_area.main_image.getViewBox().sigRangeChangedManually.emit(
            win0.slicer_area.main_image.getViewBox().state["mouseEnabled"][:]
        )
        assert win1.slicer_area.main_image.getViewBox().viewRange() == [[2, 3], [1, 2]]


def test_manager_link_action_links_colors(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        select_tools(manager, [0, 1])
        qtbot.wait_until(lambda: manager.link_action.isEnabled())
        manager.link_action.trigger()

        proxy = manager.get_imagetool(0).slicer_area._linking_proxy
        assert proxy is not None
        assert proxy.link_colors is True

        control = (
            manager.get_imagetool(0).docks[1].widget().findChild(ItoolColormapControls)
        )
        assert control is not None
        control._set_gamma(1.5)
        assert manager.get_imagetool(1).slicer_area.colormap_properties[
            "gamma"
        ] == pytest.approx(1.5)

        manager.get_imagetool(0).slicer_area.undo()
        assert manager.get_imagetool(0).slicer_area.colormap_properties[
            "gamma"
        ] == pytest.approx(0.5)
        assert manager.get_imagetool(1).slicer_area.colormap_properties[
            "gamma"
        ] == pytest.approx(0.5)


def test_manager_workspace_io(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        # Add two tools
        itool([data, data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Open dtool for first tool
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        # Save and load workspace
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = f"{tmp_dir_name}/workspace.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(tmp_dir_name)
                dialog.selectFile(filename)
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText("workspace.itws")

            # Save workspace
            accept_dialog(
                lambda: manager.save(native=False),
                pre_call=_go_to_file,
            )
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
            assert manager.workspace_path == str(pathlib.Path(filename).resolve())
            assert not manager.is_workspace_modified

            # Load workspace
            accept_dialog(lambda: manager.load(native=False), pre_call=_go_to_file)

            # Check if the data is loaded
            assert manager.ntools == 2

            # Check if the child dtool is also loaded
            assert len(manager._tool_graph.root_wrappers[0]._childtools) == 1

            select_tools(manager, list(manager._tool_graph.root_wrappers.keys()))
            accept_dialog(manager.remove_action.trigger)
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_manager_workspace_roundtrip_restores_manager_layout(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        manager.resize(640, 520)
        manager.main_splitter.setSizes([220, 420])
        manager.right_splitter.setSizes([240, 140, 120])
        expected_layout = manager._workspace_layout_snapshot()
        assert "window_state" in expected_layout
        assert "geometry" not in expected_layout
        assert expected_layout["window_state"]["geometry"]
        expected_size = manager.size()
        expected_main_sizes = manager.main_splitter.sizes()
        expected_right_sizes = manager.right_splitter.sizes()

        fname = tmp_path / "manager-layout.itws"
        manager._save_workspace_document(fname, force_full=True)

        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest["manager_layout"] == expected_layout

        manager.resize(480, 500)
        manager.main_splitter.setSizes([120, 360])
        manager.right_splitter.setSizes([120, 250, 80])
        assert manager._workspace_layout_snapshot() != expected_layout

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(
            lambda: (
                manager.size() == expected_size
                and manager.main_splitter.sizes() == expected_main_sizes
                and manager.right_splitter.sizes() == expected_right_sizes
            ),
            timeout=5000,
        )
        assert not manager.is_workspace_modified


def test_manager_workspace_layout_only_save_updates_root_manifest_only(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "layout-only.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        delta_save_count = manager._workspace_state.delta_save_count

        manager.resize(manager.width() + 40, manager.height() + 30)
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        assert manager._workspace_state.layout_modified
        assert manager._dirty_details_text()
        manager._mark_workspace_layout_dirty()

        def _forbid_node_serialization(*_args, **_kwargs):
            raise AssertionError("layout-only save serialized a node")

        def _forbid_full_save(*_args, **_kwargs):
            raise AssertionError("layout-only save requested a full workspace write")

        monkeypatch.setattr(
            manager, "_serialize_workspace_node", _forbid_node_serialization
        )
        monkeypatch.setattr(
            manager_workspace, "_write_full_workspace_tree_file", _forbid_full_save
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == delta_save_count
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert "delta_save_count" not in manifest
        assert manifest["manager_layout"] == manager._workspace_layout_snapshot()


def test_manager_workspace_standalone_app_only_save_updates_root_manifest_only(
    qtbot,
    monkeypatch,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "standalone-only.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        delta_save_count = manager._workspace_state.delta_save_count

        manager.show_ptable()
        ptable = manager.ptable_window
        ptable.hv_edit.setText("150")
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        assert manager._workspace_state.layout_modified

        def _forbid_node_serialization(*_args, **_kwargs):
            raise AssertionError("standalone-only save serialized a node")

        def _forbid_full_save(*_args, **_kwargs):
            raise AssertionError(
                "standalone-only save requested a full workspace write"
            )

        monkeypatch.setattr(
            manager, "_serialize_workspace_node", _forbid_node_serialization
        )
        monkeypatch.setattr(
            manager_workspace, "_write_full_workspace_tree_file", _forbid_full_save
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == delta_save_count
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert "delta_save_count" not in manifest
        assert manifest["standalone_apps"]["apps"]["ptable"]["photon_energy"] == "150"


def test_manager_workspace_import_does_not_restore_manager_layout(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        manager.resize(640, 520)
        manager.main_splitter.setSizes([220, 420])
        manager.right_splitter.setSizes([240, 140, 120])
        import_fname = tmp_path / "import-layout.itws"
        manager._save_workspace_document(import_fname, force_full=True)

        manager.resize(480, 500)
        manager.main_splitter.setSizes([120, 360])
        manager.right_splitter.setSizes([120, 250, 80])
        current_layout = manager._workspace_layout_snapshot()

        import h5py

        with h5py.File(import_fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manager._from_h5py_workspace_file(
            import_fname, manifest, replace=False, mark_dirty=True
        )
        assert manager._workspace_layout_snapshot() == current_layout

        tree = manager_xarray.open_workspace_datatree(import_fname, chunks=None)
        assert manager._from_datatree(
            tree,
            replace=False,
            mark_dirty=True,
            select=False,
            workspace_file_path=import_fname,
        )
        assert manager._workspace_layout_snapshot() == current_layout


def test_manager_workspace_roundtrip_restores_loader_and_standalone_apps(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    example_data_dir: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    loader_name = next(
        name for name in erlab.io.loaders if erlab.io.loaders[name].file_dialog_methods
    )
    name_filter = next(iter(erlab.io.loaders[loader_name].file_dialog_methods))
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        manager._recent_directory = str(example_data_dir)
        manager._recent_name_filter = name_filter
        manager._recent_loader_kwargs_by_filter[name_filter] = {"single": True}
        manager._recent_loader_extensions_by_filter[name_filter] = {
            "coordinate_attrs": ["manager"]
        }

        manager.show_explorer()
        explorer = manager.explorer
        explorer.resize(682, 444)
        explorer.add_tab(root_path=example_data_dir, loader_name=loader_name)
        qtbot.wait_until(lambda: explorer.tab_widget.count() == 2, timeout=5000)
        explorer.tab_widget.setCurrentIndex(1)
        explorer.current_explorer._preview_check.setChecked(True)
        explorer.current_explorer._tree_view.sortByColumn(
            0, QtCore.Qt.SortOrder.DescendingOrder
        )
        explorer.current_explorer._loader_kwargs_by_name[loader_name] = {
            "single": False
        }
        explorer.current_explorer._loader_extensions_by_name[loader_name] = {
            "coordinate_attrs": ["explorer"]
        }
        expected_explorer_size = explorer.size()
        explorer.hide()
        qtbot.wait_until(lambda: not explorer.isVisible(), timeout=5000)

        manager.show_ptable()
        ptable = manager.ptable_window
        ptable.resize(825, 650)
        ptable._set_selection_state(
            [6, 8], current_atomic_number=8, anchor_atomic_number=6
        )
        ptable._plot_atomic_number = 6
        ptable._plot_target_user_selected = True
        ptable.hv_edit.setText("80")
        ptable.workfunction_edit.setText("4.5")
        ptable.max_harmonic_spin.setValue(3)
        ptable.notation_combo.setCurrentIndex(ptable.notation_combo.findData("iupac"))
        ptable._refresh_window_state(ensure_visible=False)

        fname = tmp_path / "loader-standalone.itws"
        manager._save_workspace_document(fname, force_full=True)
        expected_ptable_size = ptable.size()

        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest["loader_state"]["recent_directory"] == str(example_data_dir)
        assert manifest["loader_state"]["recent_name_filter"] == name_filter
        assert manifest["loader_state"]["manager_loader_kwargs_by_filter"][
            name_filter
        ] == {"single": True}
        app_state = manifest["standalone_apps"]["apps"]
        assert not app_state["explorer"]["window_state"]["visible"]
        assert app_state["explorer"]["active_tab"] == 1
        assert app_state["explorer"]["loader_kwargs_by_name"][loader_name] == {
            "single": False
        }
        assert app_state["ptable"]["window_state"]["visible"]
        assert app_state["ptable"]["window_state"]["rect"][2:] == [
            expected_ptable_size.width(),
            expected_ptable_size.height(),
        ]
        assert app_state["ptable"]["selected_atomic_numbers"] == [6, 8]

        manager._close_standalone_app("explorer")
        manager._close_standalone_app("ptable")
        manager._recent_directory = None
        manager._recent_name_filter = None
        manager._recent_loader_kwargs_by_filter.clear()
        manager._recent_loader_extensions_by_filter.clear()

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )

        assert manager._recent_directory == str(example_data_dir)
        assert manager._recent_name_filter == name_filter
        assert manager._recent_loader_kwargs_by_filter[name_filter] == {"single": True}
        assert manager._recent_loader_extensions_by_filter[name_filter] == {
            "coordinate_attrs": ["manager"]
        }

        assert "explorer" not in manager._standalone_app_windows
        assert manager._standalone_app_pending_states["explorer"]["active_tab"] == 1
        restored_ptable = manager.ptable_window
        assert restored_ptable.isVisible()
        assert restored_ptable.size().width() >= expected_ptable_size.width()
        assert restored_ptable.size().height() == expected_ptable_size.height()
        assert restored_ptable.selected_atomic_numbers == (6, 8)
        assert restored_ptable._plot_atomic_number == 6
        assert restored_ptable.hv_edit.text() == "80"
        assert restored_ptable.workfunction_edit.text() == "4.5"
        assert restored_ptable.max_harmonic == 3
        assert restored_ptable.current_notation == "iupac"

        manager.show_explorer()
        restored_explorer = manager.explorer
        qtbot.wait_until(restored_explorer.isVisible, timeout=5000)
        assert restored_explorer.size() == expected_explorer_size
        assert restored_explorer.tab_widget.count() == 2
        assert restored_explorer.tab_widget.currentIndex() == 1
        restored_tab = restored_explorer.current_explorer
        assert restored_tab is not None
        assert restored_tab.current_directory == example_data_dir
        assert restored_tab.loader_name == loader_name
        assert restored_tab._preview_check.isChecked()
        assert restored_tab._fs_model._sort_order == QtCore.Qt.SortOrder.DescendingOrder
        assert restored_explorer.loader_kwargs_by_name()[loader_name] == {
            "single": False
        }
        assert restored_explorer.loader_extensions_by_name()[loader_name] == {
            "coordinate_attrs": ["explorer"]
        }


def test_manager_workspace_roundtrip_restores_full_serializable_state(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    source_path = tmp_path / "source-root.h5"
    root_data = _workspace_sweep_data("root", offset=10.0)
    peer_data = _workspace_sweep_data("peer", offset=200.0)
    root_created = datetime.datetime(2024, 2, 3, 4, 5, 6, tzinfo=datetime.UTC)
    peer_created = datetime.datetime(2024, 2, 4, 5, 6, 7, tzinfo=datetime.UTC)
    child_created = datetime.datetime(2024, 2, 5, 6, 7, 8, tzinfo=datetime.UTC)
    tool_created = datetime.datetime(2024, 2, 6, 7, 8, 9, tzinfo=datetime.UTC)
    figure_created = datetime.datetime(2024, 2, 7, 8, 9, 10, tzinfo=datetime.UTC)
    root_marker = "snapshot-root"
    peer_marker = "snapshot-peer"
    child_marker = "snapshot-child-image"
    nested_marker = "snapshot-nested-image"
    child_tool_marker = "snapshot-child-tool"
    output_marker = "snapshot-output-image"
    figure_marker = "snapshot-figure-tool"

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        root_tool = erlab.interactive.imagetool.ImageTool(root_data, _in_manager=True)
        _configure_workspace_sweep_imagetool(root_tool, source_file=source_path)
        root_provenance = _workspace_test_file_spec(source_path)
        manager.add_imagetool(
            root_tool,
            show=False,
            uid="root-watched",
            snapshot_token=root_marker,
            created_time=root_created,
            watched_var=("root_data", "watch-root-uid"),
            watched_workspace_link_id=manager._workspace_state.link_id,
            watched_source_label="notebook-a",
            watched_source_uid="kernel-a",
            watched_connected=False,
            source_input_ndim=root_data.ndim,
            provenance_spec=root_provenance,
            source_spec=provenance.full_data(
                provenance.RenameOperation(name="root-live")
            ),
            source_auto_update=True,
            source_state="fresh",
        )
        manager.rename_imagetool(0, "root-renamed")

        peer_tool = erlab.interactive.imagetool.ImageTool(peer_data, _in_manager=True)
        peer_tool.setGeometry(7, 90, 792, 360)
        peer_tool.slicer_area.set_colormap("magma", gamma=0.8, update=False)
        manager.add_imagetool(
            peer_tool,
            show=True,
            uid="root-peer",
            snapshot_token=peer_marker,
            created_time=peer_created,
        )

        manager.link_imagetools(0, 1, link_colors=False)
        model = manager.tree_view._model
        assert model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            model.rowCount(),
            0,
            QtCore.QModelIndex(),
        )
        assert manager._tool_graph.displayed_indices == [1, 0]

        source_data, _state = root_tool.slicer_area.persistence_data_and_state()
        child_binding = provenance.ImageToolSelectionSourceBinding(
            selection_mode="isel",
            selection_indexers={"z": 1},
            transpose_dims=("x", "y", "eV"),
            squeeze=True,
        )
        child_source_spec = child_binding.materialize(source_data)
        child_data = child_source_spec.apply(source_data).rename("child-image")
        child_tool = erlab.interactive.imagetool.ImageTool(child_data, _in_manager=True)
        child_tool.setGeometry(80, 130, 696, 310)
        child_tool.slicer_area.set_colormap(
            "plasma", gamma=1.2, reverse=True, update=False
        )
        child_uid = manager.add_imagetool_child(
            child_tool,
            0,
            show=True,
            uid="child-image",
            snapshot_token=child_marker,
            created_time=child_created,
            provenance_spec=provenance.script(
                start_label="Build child image",
                seed_code="derived = data.isel(z=1).squeeze()",
                active_name="derived",
            ),
            source_spec=child_source_spec,
            source_binding=child_binding,
            source_auto_update=True,
            source_state="stale",
        )

        nested_data = child_data.copy(deep=True).rename("nested-image")
        nested_tool = erlab.interactive.imagetool.ImageTool(
            nested_data, _in_manager=True
        )
        nested_tool.setGeometry(80, 160, 696, 480)
        manager.add_imagetool_child(
            nested_tool,
            child_uid,
            show=True,
            uid="nested-image",
            snapshot_token=nested_marker,
            created_time=child_created + datetime.timedelta(seconds=1),
            provenance_spec=provenance.full_data(
                provenance.RenameOperation(name="nested-image")
            ),
        )

        tool_binding = provenance.ImageToolSelectionSourceBinding(
            selection_mode="isel",
            selection_indexers={"z": 0},
            transpose_dims=("x", "y", "eV"),
            squeeze=True,
        )
        tool_source_spec = tool_binding.materialize(source_data)
        tool_data = tool_source_spec.apply(source_data).rename("sweep-tool-data")
        extra_data = tool_data.mean("eV").rename("sweep-tool-extra")
        tool_data.attrs = {"sample": "sweep-tool-data"}
        extra_data.attrs = {"sample": "sweep-tool-extra"}
        child_window = _WorkspaceSweepChildTool(tool_data, extra_data=extra_data)
        child_window.setGeometry(160, 170, 360, 270)
        child_window._tool_display_name = "Sweep Child Display"
        child_window.tool_status = _WorkspaceSweepToolState(
            value=7,
            mode="configured",
            weights=[0.25, 0.5, 0.75],
            options={"snap": True, "labels": ["alpha", "beta"]},
        )
        child_window.set_source_binding(
            tool_source_spec,
            source_binding=tool_binding,
            auto_update=True,
            state="unavailable",
        )
        child_window.set_input_provenance_spec(root_provenance)
        child_tool_uid = manager.add_childtool(
            child_window,
            0,
            show=False,
            uid="sweep-child-tool",
            snapshot_token=child_tool_marker,
            created_time=tool_created,
        )

        output_data = child_window.output_imagetool_data("workspace-sweep.primary")
        assert output_data is not None
        output_tool = erlab.interactive.imagetool.ImageTool(
            output_data, _in_manager=True
        )
        output_tool.setGeometry(80, 200, 696, 480)
        output_tool.slicer_area.set_colormap(
            "cividis", gamma=0.9, reverse=False, update=False
        )
        output_child_uid = manager.add_imagetool_child(
            output_tool,
            child_tool_uid,
            show=True,
            uid="sweep-output-image",
            snapshot_token=output_marker,
            created_time=tool_created + datetime.timedelta(seconds=1),
            output_id="workspace-sweep.primary",
            source_auto_update=True,
            source_state="stale",
        )

        figure_data = _workspace_sweep_data("figure", offset=500.0).isel(eV=0)
        figure_data.attrs = {"sample": "figure"}
        figure_extra = figure_data.mean("z").rename("figure-extra")
        figure_extra.attrs = {"sample": "figure-extra"}
        figure_tool = _WorkspaceSweepFigureTool(
            figure_data.rename("figure-primary"),
            extra_data=figure_extra,
        )
        figure_tool.setGeometry(210, 220, 420, 320)
        figure_tool._tool_display_name = "Sweep Figure Display"
        figure_tool.tool_status = _WorkspaceSweepToolState(
            value=11,
            mode="figure",
            weights=[1.5],
            options={"layout": "standalone"},
        )
        figure_uid = manager.add_figuretool(
            figure_tool,
            show=False,
            uid="sweep-figure-tool",
            snapshot_token=figure_marker,
            created_time=figure_created,
        )

        name_filter = "Xarray DataArray (*.h5)"
        manager._recent_directory = str(tmp_path)
        manager._recent_name_filter = name_filter
        manager._recent_loader_kwargs_by_filter[name_filter] = {"single": True}
        manager._recent_loader_extensions_by_filter[name_filter] = {
            "coordinate_attrs": ["manager"]
        }
        manager._workspace_controller._loader_state = (
            manager_workspace.WorkspaceLoaderState(
                explorer_loader_kwargs_by_name={"xarray": {"single": False}},
                explorer_loader_extensions_by_name={
                    "xarray": {"coordinate_attrs": ["explorer"]}
                },
            )
        )
        manager._standalone_app_pending_states["explorer"] = {
            "window_state": {"visible": False, "rect": [10, 20, 333, 222]},
            "tabs": [
                {
                    "root_path": str(tmp_path),
                    "loader_name": None,
                    "preview": True,
                    "selected_paths": [str(source_path)],
                    "sort_column": 1,
                    "sort_order": "descending",
                    "splitter_sizes": [120, 240],
                    "preview_splitter_sizes": [80, 160],
                }
            ],
            "active_tab": 0,
            "loader_kwargs_by_name": {"xarray": {"single": False}},
            "loader_extensions_by_name": {"xarray": {"coordinate_attrs": ["explorer"]}},
        }
        manager.show_ptable()
        ptable = manager.ptable_window
        ptable._set_selection_state(
            [6, 8], current_atomic_number=8, anchor_atomic_number=6
        )
        ptable._plot_atomic_number = 6
        ptable._plot_target_user_selected = True
        ptable.hv_edit.setText("120.5")
        ptable.workfunction_edit.setText("4.35")
        ptable.max_harmonic_spin.setValue(4)
        ptable.notation_combo.setCurrentIndex(ptable.notation_combo.findData("iupac"))
        ptable._refresh_window_state(ensure_visible=False)
        qtbot.wait(100)

        fname = tmp_path / "maximal-state.itws"
        manager._save_workspace_document(fname, force_full=True)
        expected_snapshot = _workspace_sweep_runtime_snapshot(manager)
        expected_data_items = _workspace_sweep_data_items(manager)

        root_attrs = manager_workspace._read_workspace_root_attrs_h5py(fname)
        manifest = manager_workspace._workspace_manifest_from_attrs(root_attrs)
        assert root_attrs["imagetool_workspace_schema_version"] == 4
        assert manifest["root_order"] == [1, 0]
        assert manifest["workspace_link_id"] == manager._workspace_state.link_id
        assert manifest["loader_state"] == expected_snapshot["loader_state"]
        assert manifest["standalone_apps"] == expected_snapshot["standalone_apps"]

        entries = {entry["uid"]: entry for entry in manifest["nodes"]}
        assert set(entries) == {
            "root-watched",
            "root-peer",
            "child-image",
            "nested-image",
            child_tool_uid,
            output_child_uid,
            figure_uid,
        }
        assert entries["root-watched"]["path"] == "0"
        assert entries["root-peer"]["path"] == "1"
        assert entries["child-image"]["path"] == "0/childtools/child-image"
        assert (
            entries["nested-image"]["path"]
            == "0/childtools/child-image/childtools/nested-image"
        )
        assert entries[child_tool_uid]["kind"] == "tool"
        assert (
            entries[output_child_uid]["path"]
            == f"0/childtools/{child_tool_uid}/childtools/{output_child_uid}"
        )
        assert entries[figure_uid]["path"] == f"figures/{figure_uid}"
        linked_entries = [entry for entry in entries.values() if "link_group" in entry]
        assert {entry["uid"] for entry in linked_entries} == {
            "root-watched",
            "root-peer",
        }
        assert {entry["link_colors"] for entry in linked_entries} == {False}

        root_payload = _workspace_sweep_h5_attr_payload(fname, "0/imagetool")
        assert {
            "itool_state",
            "itool_title",
            "itool_name",
            "itool_window_state",
            "itool_provenance_spec",
            "manager_node_uid",
            "manager_node_kind",
            "manager_node_snapshot_token",
            "manager_node_added_at",
            "manager_node_provenance_spec",
            "manager_node_source_input_ndim",
            "manager_node_watched_varname",
            "manager_node_watched_uid",
            "manager_node_watched_workspace_link_id",
            "manager_node_watched_source_label",
            "manager_node_watched_source_uid",
            "manager_node_watched_connected",
            "manager_node_live_source_spec",
            "manager_node_source_state",
            "manager_node_source_auto_update",
        } <= set(root_payload)
        assert root_payload["manager_node_watched_connected"] is False
        assert root_payload["manager_node_source_input_ndim"] == root_data.ndim
        assert root_payload["itool_state"]["filter_operation"] == (
            provenance.GaussianFilterOperation(sigma={"x": 0.5}).model_dump(mode="json")
        )
        assert root_payload["itool_state"]["slice"]["snap_to_data"] is True
        assert root_payload["itool_state"]["slice"]["twin_coord_names"] == [
            "temperature"
        ]
        assert root_payload["itool_state"]["plotitem_states"][0]["roi_states"]
        assert (
            root_payload["itool_state"]["plotitem_states"][0]["guideline_state"][
                "count"
            ]
            == 2
        )

        child_payload = _workspace_sweep_h5_attr_payload(
            fname, "0/childtools/child-image/imagetool"
        )
        assert "manager_node_output_id" not in child_payload
        assert child_payload["manager_node_source_state"] == "stale"
        assert child_payload["manager_node_source_auto_update"] is True
        assert child_payload["manager_node_live_source_spec"] == (
            child_source_spec.model_dump(mode="json")
        )
        assert "manager_node_live_source_binding" not in child_payload

        output_payload = _workspace_sweep_h5_attr_payload(
            fname,
            f"0/childtools/{child_tool_uid}/childtools/{output_child_uid}/imagetool",
        )
        assert output_payload["manager_node_output_id"] == "workspace-sweep.primary"
        assert output_payload["manager_node_source_state"] == "stale"
        assert output_payload["manager_node_source_auto_update"] is True
        assert "manager_node_live_source_spec" not in output_payload
        assert "manager_node_live_source_binding" not in output_payload

        tool_payload = _workspace_sweep_h5_attr_payload(
            fname, "0/childtools/sweep-child-tool/tool"
        )
        assert tool_payload["tool_state"] == child_window.tool_status.model_dump(
            mode="json"
        )
        assert tool_payload["tool_source_state"] == "unavailable"
        assert tool_payload["tool_source_auto_update"] is True
        assert tool_payload["tool_source_spec"] == tool_source_spec.model_dump(
            mode="json"
        )
        assert "tool_source_binding" not in tool_payload
        with h5py.File(fname, "r") as h5_file:
            tool_group = h5_file["0/childtools/sweep-child-tool/tool"]
            assert "auxiliary" in tool_group
            assert (
                tool_group["auxiliary"].attrs[
                    manager_workspace._TOOL_DATA_BLOB_NAME_ATTR
                ]
                == extra_data.name
            )

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        qtbot.wait_until(lambda: figure_uid in manager._tool_graph.figure_uids)

        actual_data_items = _workspace_sweep_data_items(manager)
        actual_snapshot = _workspace_sweep_runtime_snapshot(manager)
        _workspace_sweep_assert_snapshot_equal(actual_snapshot, expected_snapshot)
        _workspace_sweep_assert_data_items_equal(actual_data_items, expected_data_items)
        assert not manager.is_workspace_modified


def test_manager_workspace_loader_state_does_not_create_explorer_app_state(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    example_data_dir: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    loader_name = next(
        name for name in erlab.io.loaders if erlab.io.loaders[name].file_dialog_methods
    )
    name_filter = next(iter(erlab.io.loaders[loader_name].file_dialog_methods))
    explorer_kwargs = {loader_name: {"single": False}}
    explorer_extensions = {loader_name: {"coordinate_attrs": ["explorer"]}}
    root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
    qtbot.addWidget(root)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        manager.add_imagetool(root, show=False)
        manager._recent_directory = str(example_data_dir)
        manager._recent_name_filter = name_filter
        manager._recent_loader_kwargs_by_filter[name_filter] = {"single": True}
        manager._workspace_controller._loader_state = (
            manager_workspace.WorkspaceLoaderState(
                explorer_loader_kwargs_by_name=explorer_kwargs,
                explorer_loader_extensions_by_name=explorer_extensions,
            )
        )

        fname = tmp_path / "loader-no-explorer.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest["loader_state"]["explorer_loader_kwargs_by_name"] == (
            explorer_kwargs
        )
        assert "explorer" not in manifest["standalone_apps"]["apps"]

        manager._workspace_controller._loader_state = (
            manager_workspace.WorkspaceLoaderState()
        )
        manager._recent_directory = None
        manager._recent_name_filter = None
        manager._recent_loader_kwargs_by_filter.clear()

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        assert "explorer" not in manager._standalone_app_pending_states

        manager._mark_workspace_layout_dirty()
        assert _request_workspace_save_and_wait(qtbot, manager)
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert "explorer" not in manifest["standalone_apps"]["apps"]
        assert manifest["loader_state"]["explorer_loader_kwargs_by_name"] == (
            explorer_kwargs
        )

        manager.show_explorer()
        explorer = manager.explorer
        assert (
            explorer.loader_kwargs_by_name()[loader_name]
            == explorer_kwargs[loader_name]
        )
        assert (
            explorer.loader_extensions_by_name()[loader_name]
            == explorer_extensions[loader_name]
        )


def test_workspace_loader_and_standalone_app_state_edge_cases(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller

        controller._restore_workspace_loader_state({"loader_state": []})
        controller._restore_workspace_loader_state(
            {"loader_state": {"manager_loader_kwargs_by_filter": []}}
        )

        loader_calls: list[
            tuple[dict[str, dict[str, typing.Any]], dict[str, dict[str, typing.Any]]]
        ] = []

        class _LoaderStateExplorer(QtWidgets.QWidget):
            def apply_loader_state(
                self,
                *,
                kwargs_by_name: dict[str, dict[str, typing.Any]],
                extensions_by_name: dict[str, dict[str, typing.Any]],
            ) -> None:
                loader_calls.append((kwargs_by_name, extensions_by_name))

        explorer = _LoaderStateExplorer()
        qtbot.addWidget(explorer)
        manager._standalone_app_windows["explorer"] = explorer
        controller._restore_workspace_loader_state(
            {
                "loader_state": {
                    "explorer_loader_kwargs_by_name": {"example": {"single": True}},
                    "explorer_loader_extensions_by_name": {
                        "example": {"coordinate_attrs": ["sample"]}
                    },
                }
            }
        )
        assert loader_calls == [
            (
                {"example": {"single": True}},
                {"example": {"coordinate_attrs": ["sample"]}},
            )
        ]
        manager._standalone_app_windows.pop("explorer")

        assert controller._validated_standalone_app_state("missing", {}) is None
        assert (
            controller._validated_standalone_app_state("explorer", {"tabs": "bad"})
            is None
        )

        manager._standalone_app_pending_states["ptable"] = {
            "window_state": {"visible": False},
            "selected_atomic_numbers": [1],
        }
        snapshot = controller._workspace_standalone_apps_snapshot()
        assert snapshot["apps"]["ptable"]["selected_atomic_numbers"] == [1]

        controller._restore_standalone_apps_state({"standalone_apps": []})
        controller._restore_standalone_apps_state({"standalone_apps": {"apps": []}})
        controller._restore_standalone_apps_state(
            {"standalone_apps": {"apps": {"missing": {}}}}
        )

        manager.show_ptable()
        ptable = manager.ptable_window
        assert ptable.isVisible()
        hidden_ptable_state = ptable.workspace_state_payload()
        hidden_ptable_state["window_state"]["visible"] = False
        controller._restore_standalone_apps_state(
            {"standalone_apps": {"apps": {"ptable": hidden_ptable_state}}}
        )
        assert not ptable.isVisible()

        widget = QtWidgets.QWidget()
        qtbot.addWidget(widget)
        manager._apply_standalone_app_state("missing", widget, {})


def test_manager_workspace_import_does_not_restore_standalone_apps(
    qtbot,
    tmp_path: pathlib.Path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
        manager.add_imagetool(root, show=False)
        manager.show_ptable()
        ptable = manager.ptable_window
        ptable.hv_edit.setText("120")
        ptable._set_selection_state(
            [8], current_atomic_number=8, anchor_atomic_number=8
        )
        ptable._refresh_window_state(ensure_visible=False)

        fname = tmp_path / "import-standalone.itws"
        manager._save_workspace_document(fname, force_full=True)

        ptable.hv_edit.setText("30")
        ptable._set_selection_state(
            [1], current_atomic_number=1, anchor_atomic_number=1
        )
        ptable._refresh_window_state(ensure_visible=False)

        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manager._from_h5py_workspace_file(
            fname, manifest, replace=False, mark_dirty=True
        )
        assert ptable.hv_edit.text() == "30"
        assert ptable.selected_atomic_numbers == (1,)

        tree = manager_xarray.open_workspace_datatree(fname, chunks=None)
        assert manager._from_datatree(
            tree,
            replace=False,
            mark_dirty=True,
            select=False,
            workspace_file_path=fname,
        )
        assert ptable.hv_edit.text() == "30"
        assert ptable.selected_atomic_numbers == (1,)


def test_manager_workspace_restore_layout_ignores_missing_or_invalid_values(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        layout = manager._workspace_layout_snapshot()

        manager._restore_workspace_layout(None)
        manager._restore_workspace_layout({})
        manager._restore_workspace_layout({"manager_layout": "invalid"})
        manager._restore_workspace_layout(
            {
                "manager_layout": {
                    "window_state": {"geometry": ""},
                    "main_splitter": "",
                    "right_splitter": "",
                }
            }
        )
        assert manager._workspace_layout_snapshot() == layout


def test_manager_workspace_preserves_link_groups(
    qtbot,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data, test_data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        fname = tmp_path / "linked.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert not manager.is_workspace_modified

        manager.link_imagetools(0, 1, link_colors=False)
        assert manager.is_workspace_modified
        manager._save_workspace_document(fname, force_full=True)

        manifest = manager_workspace._workspace_manifest_from_attrs(
            manager_workspace._read_workspace_root_attrs_h5py(fname)
        )
        linked_entries = [entry for entry in manifest["nodes"] if "link_group" in entry]
        assert {entry["path"] for entry in linked_entries} == {"0", "1"}
        assert {entry["link_group"] for entry in linked_entries} == {0}
        assert {entry["link_colors"] for entry in linked_entries} == {False}

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        loaded0 = manager.get_imagetool(0)
        loaded1 = manager.get_imagetool(1)
        proxy0 = loaded0.slicer_area._linking_proxy
        proxy1 = loaded1.slicer_area._linking_proxy
        assert proxy0 is not None
        assert proxy0 is proxy1
        assert proxy0.link_colors is False
        assert not manager.get_imagetool(2).slicer_area.is_linked
        assert not manager.is_workspace_modified


def test_manager_workspace_restore_hidden_memory_link_group_keeps_payload_pending(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.link_imagetools(0, 1, link_colors=False)
        fname = tmp_path / "hidden-memory-linked.itws"
        manager._save_workspace_document(fname, force_full=True)

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("link restore should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        for index, wrapper in enumerate(wrappers):
            assert wrapper.pending_workspace_memory_payload == (
                fname.resolve(),
                f"{index}/imagetool",
            )
            assert wrapper.imagetool is None
            assert wrapper.workspace_linked
            assert wrapper.workspace_link_colors is False

        assert wrappers[0].workspace_link_key == wrappers[1].workspace_link_key
        assert not manager.is_workspace_modified

        icon_colors: list[QtGui.QColor] = []
        original_icon = manager_modelview.qta.icon

        def _record_icon(name, *args, **kwargs):
            if name == "mdi6.link-variant":
                icon_colors.append(kwargs["color"])
            return original_icon(name, *args, **kwargs)

        monkeypatch.setattr(manager_modelview.qta, "icon", _record_icon)
        index = manager.tree_view._model.index(0, 0)
        option = manager.tree_view._delegate._option_for_index(manager.tree_view, index)
        canvas = QtGui.QPixmap(200, 32)
        canvas.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(canvas)
        try:
            manager.tree_view._delegate.paint(painter, option, index)
        finally:
            painter.end()

        assert icon_colors
        assert icon_colors[-1] != option.palette.color(QtGui.QPalette.ColorRole.Mid)


def test_manager_workspace_mixed_pending_link_badge_uses_group_color(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)

        fname = tmp_path / "mixed-hidden-memory-linked.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert wrappers[0].pending_workspace_memory_payload is not None
        assert wrappers[1].pending_workspace_memory_payload is not None
        link_key = wrappers[0].workspace_link_key
        assert link_key is not None
        assert wrappers[1].workspace_link_key == link_key

        manager.get_imagetool(0)
        assert wrappers[0].imagetool is not None
        assert wrappers[0].slicer_area._linking_proxy is None
        assert wrappers[1].pending_workspace_memory_payload is not None

        icon_colors: list[QtGui.QColor] = []
        original_icon = manager_modelview.qta.icon

        def _record_icon(name, *args, **kwargs):
            if name == "mdi6.link-variant":
                icon_colors.append(kwargs["color"])
            return original_icon(name, *args, **kwargs)

        monkeypatch.setattr(manager_modelview.qta, "icon", _record_icon)
        index = manager.tree_view._model.index(0, 0)
        option = manager.tree_view._delegate._option_for_index(manager.tree_view, index)
        canvas = QtGui.QPixmap(200, 32)
        canvas.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(canvas)
        try:
            manager.tree_view._delegate.paint(painter, option, index)
        finally:
            painter.end()

        assert icon_colors
        assert icon_colors[-1] == manager.color_for_workspace_link_key(link_key)
        assert icon_colors[-1] != option.palette.color(QtGui.QPalette.ColorRole.Mid)


def test_manager_update_actions_for_pending_memory_link_state_does_not_materialize(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()

        fname = tmp_path / "pending-actions.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("action refresh should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )

        select_tools(manager, [0, 1])
        manager._update_actions()

        assert manager.link_action.isEnabled()
        assert not manager.unlink_action.isEnabled()
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )

        manager.link_imagetools(0, 1, link_colors=False)
        manager._update_actions()

        assert not manager.link_action.isEnabled()
        assert manager.unlink_action.isEnabled()
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )


def test_manager_pending_memory_reload_unavailable_does_not_materialize(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(
            xr.DataArray(
                np.arange(25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            ),
            manager=False,
            execute=False,
        )
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-reload.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("reload availability should not materialize hidden memory data")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        select_tools(manager, [0])
        reload_candidates = manager._selected_reload_candidates()
        assert reload_candidates is not None
        assert reload_candidates[2] is not None
        assert manager._selected_reload_targets() is None
        manager._update_actions()

        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()
        assert wrapper.pending_workspace_memory_payload is not None


def test_manager_pending_memory_file_source_reload_available_without_materializing(
    qtbot,
    monkeypatch,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "reload-source.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(
            test_data,
            manager=False,
            execute=False,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-file-source-reload.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("reload availability should not materialize hidden memory data")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        select_tools(manager, [0])
        reload_candidates = manager._selected_reload_candidates()
        assert reload_candidates == ([0], {}, None)
        assert manager._selected_reload_targets() == ([0], {})
        manager._update_actions()

        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None


def test_manager_pending_memory_child_routes_reload_to_file_source_parent(
    qtbot,
    monkeypatch,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "parent-source.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(
            test_data,
            manager=False,
            execute=False,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        child_data = test_data.qsel.mean("alpha")
        child = itool(child_data, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_spec=provenance.full_data(
                provenance.AverageOperation(dims=("alpha",))
            ),
            source_state="stale",
        )
        child.hide()

        fname = tmp_path / "pending-file-source-child-reload.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        child_node = manager._child_node(child_uid)
        assert wrapper.pending_workspace_memory_payload is not None
        assert child_node.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("reload routing should not materialize hidden memory data")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        select_child_tool(manager, child_uid)
        assert manager._selected_reload_candidates() == ([0], {0: [child_uid]}, None)
        assert manager._selected_reload_targets() == ([0], {0: [child_uid]})
        manager._update_actions()

        assert manager.reload_action.isVisible()
        assert manager.reload_action.isEnabled()

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-origin child replay should not prompt"),
        )
        manager._update_info()
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert copied
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(
            namespace["derived"].rename(None),
            child_data.rename(None),
        )
        assert wrapper.pending_workspace_memory_payload is not None
        assert child_node.pending_workspace_memory_payload is not None


def test_manager_pending_memory_child_source_change_marks_stale_not_unavailable(
    qtbot,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(test_data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        child_data = test_data.qsel.mean("alpha")
        child = itool(child_data, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_spec=provenance.full_data(
                provenance.AverageOperation(dims=("alpha",))
            ),
        )
        child.hide()

        fname = tmp_path / "pending-child-source-stale.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        child_node = manager._child_node(child_uid)
        assert child_node.pending_workspace_memory_payload is not None
        assert child_node.source_state == "fresh"

        updated = test_data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 2
        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: child_node.source_state == "stale", timeout=5000)
        assert child_node.pending_workspace_memory_payload is not None
        assert child_node.imagetool is None


def test_manager_pending_memory_output_child_source_change_marks_stale(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12, dtype=np.float64).reshape((3, 4)),
        dims=["x", "y"],
        coords={"x": np.arange(3), "y": np.arange(4)},
        name="parent",
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        tool_window = _WorkspaceSweepChildTool(data)
        child_tool_uid = manager.add_childtool(tool_window, 0, show=False)
        output_data = tool_window.output_imagetool_data("workspace-sweep.primary")
        assert output_data is not None
        output_tool = itool(output_data, manager=False, execute=False)
        assert isinstance(output_tool, erlab.interactive.imagetool.ImageTool)
        output_uid = manager.add_imagetool_child(
            output_tool,
            child_tool_uid,
            show=False,
            output_id="workspace-sweep.primary",
            source_auto_update=True,
            source_state="fresh",
        )
        output_tool.hide()

        fname = tmp_path / "pending-output-child-source-change.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        output_node = manager._child_node(output_uid)
        parent_tool = manager.get_childtool(child_tool_uid)
        assert parent_tool is not None
        assert output_node.pending_workspace_memory_payload is not None
        assert output_node.imagetool is None
        assert output_node.source_auto_update is True

        def _fail_materialize_pending_payload(_node) -> bool:
            if _node is output_node or _node.is_imagetool:
                pytest.fail("hidden output child source change should not materialize")
            return True

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        typing.cast("_WorkspaceSweepChildTool", parent_tool)._data = data + 10.0

        assert not output_node.handle_parent_source_replaced(data + 10.0)
        assert output_node.source_state == "stale"
        assert output_node.pending_workspace_memory_payload is not None
        assert output_node.imagetool is None


def test_manager_link_imagetools_keeps_hidden_memory_payload_pending(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()

        fname = tmp_path / "pending-link.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("linking should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        manager.link_imagetools(0, 1, link_colors=False)

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert wrappers[0].workspace_link_key == wrappers[1].workspace_link_key
        for wrapper in wrappers:
            assert wrapper.pending_workspace_memory_payload is not None
            assert wrapper.workspace_linked
            assert wrapper.workspace_link_colors is False
            assert wrapper.uid in manager._workspace_state.dirty_state
            assert wrapper.uid not in manager._workspace_state.dirty_data


def test_manager_unlink_selected_keeps_hidden_memory_payload_pending(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)

        fname = tmp_path / "pending-unlink.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("unlinking should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        select_tools(manager, [0, 1])
        manager.unlink_selected(deselect=False)

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        for wrapper in wrappers:
            assert wrapper.pending_workspace_memory_payload is not None
            assert not wrapper.workspace_linked
            assert wrapper.uid in manager._workspace_state.dirty_state
            assert wrapper.uid not in manager._workspace_state.dirty_data


def test_manager_unlink_selected_prunes_pending_link_singleton(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)

        fname = tmp_path / "pending-unlink-singleton.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("unlinking should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        select_tools(manager, [0])
        manager.unlink_selected(deselect=False)

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        for wrapper in wrappers:
            assert wrapper.pending_workspace_memory_payload is not None
            assert not wrapper.workspace_linked
            assert wrapper.workspace_link_key is None
            assert wrapper.uid in manager._workspace_state.dirty_state
            assert wrapper.uid not in manager._workspace_state.dirty_data


def test_manager_unlink_selected_prunes_live_link_singleton(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
        manager.link_imagetools(0, 1, link_colors=False)
        fname = tmp_path / "live-unlink-singleton.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(wrapper.slicer_area.is_linked for wrapper in wrappers)

        select_tools(manager, [0])
        manager.unlink_selected(deselect=False)

        for wrapper in wrappers:
            assert not wrapper.slicer_area.is_linked
            assert not wrapper.workspace_linked
            assert wrapper.workspace_link_key is None
            assert wrapper.uid in manager._workspace_state.dirty_state


def test_manager_remove_pending_linked_root_prunes_partner_without_materializing(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)

        fname = tmp_path / "pending-remove-linked-root.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "removing linked rows should not materialize hidden memory data"
            )

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        removed_uid = manager._tool_graph.root_wrappers[0].uid
        manager.remove_imagetool(0)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        survivor = next(iter(manager._tool_graph.root_wrappers.values()))
        assert removed_uid not in manager._tool_graph.nodes
        assert survivor.pending_workspace_memory_payload is not None
        assert not survivor.workspace_linked
        assert survivor.workspace_link_key is None
        assert survivor.uid in manager._workspace_state.dirty_state
        assert survivor.uid not in manager._workspace_state.dirty_data

        manager._save_workspace_document(fname, force_full=True)
        manifest = manager_workspace._workspace_manifest_from_attrs(
            manager_workspace._read_workspace_root_attrs_h5py(fname)
        )
        assert all("link_group" not in entry for entry in manifest["nodes"])


def test_manager_remove_pending_linked_child_prunes_partner_without_materializing(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root_data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=("x", "y"))
        root = itool(root_data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        child_uids: list[str] = []
        for offset in (100, 200):
            child_data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=("x", "y"),
            )
            child = itool(child_data, manager=False, execute=False)
            assert isinstance(child, erlab.interactive.imagetool.ImageTool)
            child_uids.append(manager.add_imagetool_child(child, 0, show=False))
            child.hide()
        manager.link_imagetools(*child_uids, link_colors=False)

        fname = tmp_path / "pending-remove-linked-child.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "removing linked children should not materialize hidden memory data"
            )

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        manager._remove_childtool(child_uids[0])
        survivor = manager._child_node(child_uids[1])
        assert child_uids[0] not in manager._tool_graph.nodes
        assert survivor.pending_workspace_memory_payload is not None
        assert not survivor.workspace_linked
        assert survivor.workspace_link_key is None
        assert survivor.uid in manager._workspace_state.dirty_state
        assert survivor.uid not in manager._workspace_state.dirty_data

        manager._save_workspace_document(fname, force_full=True)
        manifest = manager_workspace._workspace_manifest_from_attrs(
            manager_workspace._read_workspace_root_attrs_h5py(fname)
        )
        assert all("link_group" not in entry for entry in manifest["nodes"])


def test_manager_save_updates_pending_linked_partner_state(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        coords = {"x": np.linspace(-1.0, 1.0, 5), "y": np.linspace(0.0, 0.4, 5)}
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
                coords=coords,
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)

        fname = tmp_path / "pending-linked-partner-state.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )
        materialized_calls = 0
        original_materialize = manager._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialized_calls
            materialized_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _count_materialize,
        )

        loaded = manager.get_imagetool(0).slicer_area
        loaded.set_index(0, 3)
        qtbot.wait_until(
            lambda: wrappers[1].uid in manager._workspace_state.dirty_state,
            timeout=5000,
        )

        assert materialized_calls == 1
        assert wrappers[1].pending_workspace_memory_payload is not None
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert materialized_calls == 1
        assert wrappers[1].pending_workspace_memory_payload is not None

        with h5py.File(fname, "r") as h5_file:
            saved_state = json.loads(h5_file["1/imagetool"].attrs["itool_state"])
        assert saved_state["slice"]["indices"][0][0] == 3

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        reloaded_partner = manager.get_imagetool(1).slicer_area
        assert reloaded_partner.array_slicer.get_index(0, 0) == 3


def test_manager_materializing_pending_linked_partner_uses_pending_state(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        coords = {"x": np.linspace(-1.0, 1.0, 5), "y": np.linspace(0.0, 0.4, 5)}
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
                coords=coords,
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)

        fname = tmp_path / "pending-linked-partner-materialize-state.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )

        loaded = manager.get_imagetool(0).slicer_area
        loaded.set_index(0, 3)
        qtbot.wait_until(
            lambda: wrappers[1].uid in manager._workspace_state.dirty_state,
            timeout=5000,
        )
        assert wrappers[1].pending_workspace_memory_payload is not None

        materialized_partner = manager.get_imagetool(1).slicer_area

        assert wrappers[1].pending_workspace_memory_payload is None
        assert materialized_partner.array_slicer.get_index(0, 0) == 3


def test_pending_link_state_operation_variants_update_saved_state(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(36, dtype=np.float64).reshape((6, 6)),
            dims=["x", "y"],
            coords={"x": np.arange(6.0), "y": np.arange(6.0)},
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        source = root.slicer_area
        controller = manager._workspace_controller

        def new_state(
            *, cursors: int = 1
        ) -> tuple[
            erlab.interactive.imagetool.slicer.ArraySlicer, dict[str, typing.Any]
        ]:
            array_slicer = erlab.interactive.imagetool.slicer.ArraySlicer(data, manager)
            for _ in range(cursors - 1):
                array_slicer.add_cursor(0, update=False)
            return array_slicer, {
                "slice": array_slicer.state,
                "current_cursor": 0,
                "cursor_colors": ["#ff0000"] * cursors,
            }

        array_slicer, state = new_state()
        try:
            assert not controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "refresh", {}
            )
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "view_all", {}
            )
            assert state["manual_limits"] == {}
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "center_all_cursors", {}
            )
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "center_cursor", {}
            )
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "set_current_cursor", {"cursor": 4}
            )
            assert state["current_cursor"] == 0
            assert not controller._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_axis_inverted",
                {"dim": "missing", "inverted": True},
            )
            assert controller._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_axis_inverted",
                {"dim": "x", "inverted": True},
            )
            assert state["axis_inversions"] == {"x": True}
            assert controller._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_axis_inverted",
                {"dim": "x", "inverted": False},
            )
            assert state["axis_inversions"] == {}
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "set_index", {"axis": 0, "value": 4}
            )
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "step_index", {"axis": 0, "value": -1}
            )
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "step_index_all", {"axis": 1, "value": 1}
            )
            assert controller._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_value",
                {"axis": 0, "value": 2.0, "uniform": True},
            )
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "set_bin", {"axis": 0, "value": 3}
            )
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "set_bin_all", {"axis": 1, "value": 3}
            )
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "swap_axes", {"ax1": 0, "ax2": 1}
            )
            assert controller._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_colormap",
                {
                    "cmap": "magma",
                    "gamma": 0.5,
                    "reverse": True,
                    "high_contrast": True,
                    "zero_centered": True,
                    "levels_locked": True,
                    "levels": (1.0, 5.0),
                },
            )
            assert state["color"]["cmap"] == "magma"
            assert state["color"]["levels"] == [1.0, 5.0]
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "toggle_snap", {}
            )
            assert state["slice"]["snap_to_data"] is True
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "toggle_snap", {"value": False}
            )
            assert state["slice"]["snap_to_data"] is False
            assert not controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "unknown_operation", {}
            )
        finally:
            array_slicer.deleteLater()

        array_slicer, state = new_state(cursors=2)
        try:
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "add_cursor", {"color": None}
            )
            assert len(state["cursor_colors"]) == 3
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "add_cursor", {"color": "#123456"}
            )
            assert state["cursor_colors"][-1] == "#123456"
            state["current_cursor"] = 2
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "remove_cursor", {"index": 1}
            )
            assert state["current_cursor"] == 1
        finally:
            array_slicer.deleteLater()

        array_slicer, state = new_state(cursors=3)
        try:
            state["current_cursor"] = 0
            assert controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "remove_cursor", {"index": 0}
            )
            assert state["current_cursor"] == 0
            assert controller._pending_link_default_cursor_color(
                source, 0, [QtGui.QColor(color).name() for color in source.COLORS]
            )
        finally:
            array_slicer.deleteLater()

        array_slicer, state = new_state()
        try:
            assert not controller._update_pending_link_state_for_operation(
                state, array_slicer, source, "remove_cursor", {"index": 0}
            )
        finally:
            array_slicer.deleteLater()


def test_pending_workspace_link_payload_helper_fallbacks(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class PendingNode:
        uid = "pending"
        is_imagetool = True

        def __init__(self) -> None:
            self.attrs: dict[str, typing.Any] | None = None
            self.pending_workspace_memory_payload: tuple[pathlib.Path, str] | None = (
                tmp_path / "source.itws",
                "0/imagetool",
            )

        @property
        def pending_workspace_payload_attrs(self) -> dict[str, typing.Any] | None:
            return None if self.attrs is None else dict(self.attrs)

        def update_pending_workspace_payload_attrs(
            self, attrs: Mapping[str, typing.Any]
        ) -> None:
            self.attrs = dict(attrs)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        data = xr.DataArray(
            np.arange(36, dtype=np.float64).reshape((6, 6)),
            dims=["x", "y"],
            coords={"x": np.arange(6.0), "y": np.arange(6.0)},
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        source = root.slicer_area
        source.array_slicer.set_index(0, 0, 2, update=False)

        node = PendingNode()
        assert not controller._update_pending_workspace_manual_limits(
            node, {"x": [0.0, 1.0]}
        )
        assert not controller._apply_pending_workspace_link_operation(
            source,
            node,
            "set_index",
            {"axis": 0, "value": 1},
            tuple(data.dims),
            True,
            False,
            None,
            False,
        )

        for raw_state in (b"\xff", "{bad-json"):
            node.attrs = {"itool_state": raw_state}
            assert not controller._update_pending_workspace_manual_limits(
                node, {"x": [0.0, 1.0]}
            )
            assert not controller._apply_pending_workspace_link_operation(
                source,
                node,
                "set_index",
                {"axis": 0, "value": 1},
                tuple(data.dims),
                True,
                False,
                None,
                False,
            )

        node.attrs = {"itool_state": json.dumps({})}
        node.pending_workspace_memory_payload = None
        assert not controller._update_pending_workspace_manual_limits(
            node, {"x": [0.0, 1.0]}
        )
        assert not controller._apply_pending_workspace_link_operation(
            source,
            node,
            "set_index",
            {"axis": 0, "value": 1},
            tuple(data.dims),
            True,
            False,
            None,
            False,
        )

        node.pending_workspace_memory_payload = (
            tmp_path / "source.itws",
            "0/imagetool",
        )

        def _raise_payload_read(*_args, **_kwargs):
            raise OSError("missing payload")

        monkeypatch.setattr(
            controller,
            "_read_workspace_imagetool_payload_dataset",
            _raise_payload_read,
        )
        assert not controller._update_pending_workspace_manual_limits(
            node, {"x": [0.0, 1.0]}
        )
        assert not controller._apply_pending_workspace_link_operation(
            source,
            node,
            "set_index",
            {"axis": 0, "value": 1},
            tuple(data.dims),
            True,
            False,
            None,
            False,
        )

        def _metadata_dataset(*_args, **_kwargs):
            return xr.Dataset({manager_workspace_io._ITOOL_DATA_NAME: data})

        monkeypatch.setattr(
            controller,
            "_read_workspace_imagetool_payload_dataset",
            _metadata_dataset,
        )
        node.attrs = {"itool_state": json.dumps({})}
        assert controller._update_pending_workspace_manual_limits(
            node, {"x": [0.0, 1.0], "missing": [2.0, 3.0]}
        )
        assert json.loads(node.attrs["itool_state"])["manual_limits"] == {
            "x": [0.0, 1.0]
        }

        target_slicer = erlab.interactive.imagetool.slicer.ArraySlicer(data, manager)
        try:
            node.attrs = {
                "itool_state": json.dumps(
                    {"slice": target_slicer.state, "current_cursor": 0}
                )
            }
            assert not controller._apply_pending_workspace_link_operation(
                source,
                node,
                "unknown_operation",
                {},
                tuple(data.dims),
                True,
                False,
                None,
                False,
            )
            assert controller._apply_pending_workspace_link_operation(
                source,
                node,
                "set_index",
                {"axis": 0, "value": 1},
                tuple(data.dims),
                True,
                False,
                None,
                False,
            )
            updated_state = json.loads(node.attrs["itool_state"])
            assert updated_state["slice"]["indices"][0][0] == 1
        finally:
            target_slicer.deleteLater()


def test_manager_save_updates_pending_linked_partner_manual_limits(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        coords = {"x": np.linspace(-1.0, 1.0, 5), "y": np.linspace(0.0, 0.4, 5)}
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
                coords=coords,
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)

        fname = tmp_path / "pending-linked-partner-manual-limits.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )
        materialized_calls = 0
        original_materialize = manager._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialized_calls
            materialized_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _count_materialize,
        )

        expected_limits = {"x": [-0.5, 0.5], "y": [0.1, 0.3]}
        loaded = manager.get_imagetool(0).slicer_area
        axis = loaded.axes[0]
        loaded.manual_limits = {**expected_limits, "missing": [0.0, 1.0]}
        loaded.propagate_limit_change(axis)
        qtbot.wait_until(
            lambda: wrappers[1].uid in manager._workspace_state.dirty_state,
            timeout=5000,
        )

        assert materialized_calls == 1
        assert wrappers[1].pending_workspace_memory_payload is not None
        pending_attrs = wrappers[1].pending_workspace_payload_attrs
        assert pending_attrs is not None
        pending_state = json.loads(pending_attrs["itool_state"])
        assert pending_state["manual_limits"] == expected_limits
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert materialized_calls == 1
        assert wrappers[1].pending_workspace_memory_payload is not None

        with h5py.File(fname, "r") as h5_file:
            saved_state = json.loads(h5_file["1/imagetool"].attrs["itool_state"])
        assert saved_state["manual_limits"] == expected_limits

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        reloaded_partner = manager.get_imagetool(1).slicer_area
        assert reloaded_partner.manual_limits == expected_limits


def test_manager_pending_linked_partner_respects_link_color_setting(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        coords = {"x": np.linspace(-1.0, 1.0, 5), "y": np.linspace(0.0, 0.4, 5)}
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
                coords=coords,
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)

        fname = tmp_path / "pending-linked-partner-color-state.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            original_partner_state = json.loads(
                h5_file["1/imagetool"].attrs["itool_state"]
            )
        new_cmap = (
            "viridis"
            if original_partner_state["color"]["cmap"] != "viridis"
            else "magma"
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )
        materialized_calls = 0
        original_materialize = manager._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialized_calls
            materialized_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _count_materialize,
        )

        loaded = manager.get_imagetool(0).slicer_area
        loaded.set_index(0, 3)
        loaded.set_colormap(cmap=new_cmap)
        qtbot.wait_until(
            lambda: wrappers[1].uid in manager._workspace_state.dirty_state,
            timeout=5000,
        )

        assert materialized_calls == 1
        assert wrappers[1].pending_workspace_memory_payload is not None
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert materialized_calls == 1
        assert wrappers[1].pending_workspace_memory_payload is not None

        with h5py.File(fname, "r") as h5_file:
            saved_partner_state = json.loads(
                h5_file["1/imagetool"].attrs["itool_state"]
            )
        assert saved_partner_state["slice"]["indices"][0][0] == 3
        assert saved_partner_state["color"] == original_partner_state["color"]


def test_manager_pending_linked_partner_saves_color_when_linked(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=True)

        fname = tmp_path / "pending-linked-partner-color-linked.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            original_partner_state = json.loads(
                h5_file["1/imagetool"].attrs["itool_state"]
            )
        new_cmap = (
            "viridis"
            if original_partner_state["color"]["cmap"] != "viridis"
            else "magma"
        )
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )
        materialized_calls = 0
        original_materialize = manager._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialized_calls
            materialized_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _count_materialize,
        )

        manager.get_imagetool(0).slicer_area.set_colormap(cmap=new_cmap)
        qtbot.wait_until(
            lambda: wrappers[1].uid in manager._workspace_state.dirty_state,
            timeout=5000,
        )

        assert materialized_calls == 1
        assert wrappers[1].pending_workspace_memory_payload is not None
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert materialized_calls == 1
        assert wrappers[1].pending_workspace_memory_payload is not None

        with h5py.File(fname, "r") as h5_file:
            saved_partner_state = json.loads(
                h5_file["1/imagetool"].attrs["itool_state"]
            )
        assert saved_partner_state["color"]["cmap"] == new_cmap


def test_manager_workspace_unlink_removes_saved_link_group(
    qtbot,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        fname = tmp_path / "unlinked.itws"
        manager.link_imagetools(0, 1)
        manager._save_workspace_document(fname, force_full=True)

        select_tools(manager, [0, 1])
        manager.unlink_selected()
        assert manager.is_workspace_modified
        manager._save_workspace_document(fname, force_full=True)

        manifest = manager_workspace._workspace_manifest_from_attrs(
            manager_workspace._read_workspace_root_attrs_h5py(fname)
        )
        assert all("link_group" not in entry for entry in manifest["nodes"])

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert not manager.get_imagetool(0).slicer_area.is_linked
        assert not manager.get_imagetool(1).slicer_area.is_linked
        assert not manager.is_workspace_modified


def test_manager_workspace_save_selection_cancel_does_not_write(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _RejectedChooseDialog:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._workspace_io,
        "_ChooseFromDataTreeDialog",
        _RejectedChooseDialog,
    )
    closed_trees: list[xr.DataTree] = []
    original_close = xr.DataTree.close

    def _close_spy(tree: xr.DataTree) -> None:
        closed_trees.append(tree)
        original_close(tree)

    monkeypatch.setattr(xr.DataTree, "close", _close_spy)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = pathlib.Path(tmp_dir_name) / "workspace.itws"
            manager._save_to_file(str(filename))

            assert not filename.exists()
            assert len(closed_trees) == 1


def test_manager_workspace_save_selection_encodes_rich_attrs(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    class _AcceptedChooseDialog:
        def __init__(self, _manager, tree: xr.DataTree, *, mode: str) -> None:
            assert mode == "save"
            self._tree_widget = QtWidgets.QTreeWidget()
            root_item = self._tree_widget.invisibleRootItem()
            for key in tree:
                item = QtWidgets.QTreeWidgetItem(root_item)
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, key)
                item.setCheckState(0, QtCore.Qt.CheckState.Checked)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._workspace_io,
        "_ChooseFromDataTreeDialog",
        _AcceptedChooseDialog,
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        rich_attr = _rich_workspace_attr_value()
        data = xr.DataArray(
            np.arange(4).reshape((2, 2)),
            dims=("x", "y"),
            attrs={"Single Motor Scan": rich_attr},
        )
        root = itool(data, link=False, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "selected-rich.itws"
        manager._save_to_file(str(fname))

    with h5py.File(fname, "r") as h5_file:
        saved_data = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
        assert "Single Motor Scan" not in saved_data.attrs
        decoded = manager_workspace._h5py_attrs_to_dict(saved_data.attrs)
    _assert_rich_workspace_attr(decoded["Single Motor Scan"])


def test_manager_workspace_load_selection_skips_unchecked_children(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _SelectedChooseDialog(
        erlab.interactive.imagetool.manager._workspace_io._ChooseFromDataTreeDialog
    ):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            root_item = self._tree_widget.topLevelItem(0)
            assert root_item is not None
            unchecked_child = root_item.child(1)
            assert unchecked_child is not None
            unchecked_child.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._workspace_io,
        "_ChooseFromDataTreeDialog",
        _SelectedChooseDialog,
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)

        child_uids: list[str] = []
        for offset in (1.0, 2.0):
            child_tool = itool(data + offset, manager=False, execute=False)
            assert isinstance(child_tool, erlab.interactive.imagetool.ImageTool)
            child_uids.append(manager.add_imagetool_child(child_tool, 0, show=False))

        tree = manager._to_datatree()
        try:
            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            manager._from_datatree(tree)
        finally:
            tree.close()

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == [
            child_uids[0]
        ]
        assert child_uids[1] not in manager._tool_graph.nodes


def test_manager_workspace_load_selection_skips_unchecked_figures(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _SelectedChooseDialog(
        erlab.interactive.imagetool.manager._workspace_io._ChooseFromDataTreeDialog
    ):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            root = self._tree_widget.invisibleRootItem()
            assert root is not None
            for index in range(root.childCount()):
                item = root.child(index)
                if (
                    item is not None
                    and item.data(0, QtCore.Qt.ItemDataRole.UserRole) == figure_path
                ):
                    item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
                    break
            else:
                pytest.fail("figure entry was not shown in the workspace load dialog")

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._workspace_io,
        "_ChooseFromDataTreeDialog",
        _SelectedChooseDialog,
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root_tool = itool(data, manager=False, execute=False)
        assert isinstance(root_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_tool, show=False)
        figure_uid = manager.add_figuretool(
            _WorkspaceSweepFigureTool(data + 1), show=False
        )
        figure_path = f"figures/{figure_uid}"

        tree = manager._to_datatree()
        try:
            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            manager._from_datatree(tree)
        finally:
            tree.close()

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert figure_uid not in manager._tool_graph.nodes


def test_manager_workspace_load_migrates_legacy_manual_title_to_data_name(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"], name="scan")
    file_path = tmp_path / "scan.h5"
    data.to_netcdf(file_path, engine="h5netcdf")
    root = itool(
        data,
        manager=False,
        execute=False,
        file_path=file_path,
        load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
    )
    assert isinstance(root, erlab.interactive.imagetool.ImageTool)
    qtbot.addWidget(root)
    ds = root.to_dataset()
    root.close()
    ds.attrs["itool_title"] = "legacy manual name (scan)"
    ds.attrs["itool_name"] = "scan"
    ds.attrs["manager_node_provenance_spec"] = ds.attrs["itool_provenance_spec"]

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._load_workspace_imagetool_dataset(
            ds,
            parent_target=None,
            node_path="0",
        )

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.name_of_imagetool(0) == "legacy manual name"
        assert manager.get_imagetool(0).slicer_area._data.name == "legacy manual name"


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
            manager_workspace._WORKSPACE_TRANSACTION_PROTOCOL
        )
        manifest["delta_save_count"] = delta_save_count
    return {
        "imagetool_workspace_schema_version": 4,
        manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(manifest),
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
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()


def _read_transaction_test_value(fname: pathlib.Path) -> float:
    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        ds = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        return float(ds["data"].item())
    finally:
        opened.close()


def _assert_no_workspace_internal_groups(fname: pathlib.Path) -> None:
    import h5py

    with h5py.File(fname, "r") as h5_file:
        assert not any(
            manager_workspace._is_workspace_internal_group_name(name)
            for name in h5_file
        )


def test_workspace_file_repack_payload_strips_delta_and_skips_internal_groups(
    tmp_path,
) -> None:
    import h5py

    fname = tmp_path / "file-repack.itws"
    _write_transaction_test_workspace(fname)
    manager_workspace._write_workspace_root_attrs_to_file(
        fname,
        _transaction_test_root_attrs(delta_save_count=3),
        replace=True,
    )
    with h5py.File(fname, "a") as h5_file:
        h5_file.create_group("__itws_pending_orphan")
        h5_file.create_dataset("root_dataset", data=np.arange(3))

    assert manager_workspace._workspace_live_root_group_copy_groups(fname) == (
        ("0", "0", None),
    )
    storage_size, existing_count = manager_workspace._workspace_h5_paths_storage_size(
        fname, ("0", "missing")
    )
    assert storage_size >= np.dtype(np.float64).itemsize
    assert existing_count == 1
    assert manager_workspace._workspace_live_h5_storage_size(fname) == storage_size
    assert manager_workspace._workspace_obsolete_estimate(fname) >= 0
    root_attrs, copy_groups = manager_workspace._workspace_file_repack_payload(fname)

    manifest = manager_workspace._workspace_manifest_from_attrs(root_attrs)
    assert "delta_save_count" not in manifest
    assert "transaction_protocol" not in manifest
    assert (
        manifest["schema_version"]
        == manager_workspace._current_workspace_schema_version()
    )
    assert manifest["erlab_version"] == erlab.__version__
    assert copy_groups == (("0", "0", None),)

    manager_workspace._write_full_workspace_tree_file(
        fname,
        None,
        root_attrs,
        copy_source=fname,
        copy_groups=copy_groups,
    )

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)
    with h5py.File(fname, "r") as h5_file:
        assert set(h5_file) == {"0"}
        manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert "delta_save_count" not in manifest


def test_workspace_h5py_helpers_reject_non_workspace_files(tmp_path) -> None:
    import h5py

    fname = tmp_path / "not-workspace.itws"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_group("0")

    with pytest.raises(ValueError, match="Not a valid workspace file"):
        manager_workspace._workspace_live_root_group_copy_groups(fname)
    with pytest.raises(ValueError, match="Not a valid workspace file"):
        manager_workspace._workspace_h5_paths_storage_size(fname, ("0",))
    with pytest.raises(ValueError, match="Not a valid workspace file"):
        manager_workspace._workspace_live_h5_storage_size(fname)

    assert (
        manager_workspace._workspace_obsolete_estimate(tmp_path / "missing.itws") == 0
    )
    assert manager_workspace._workspace_h5_object_storage_size(object()) == 0


def test_workspace_h5py_filter_matching_edge_cases(tmp_path) -> None:
    import h5py

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

        assert manager_workspace._workspace_h5py_blosc2_options_match((1, 2), (1, 2))
        assert not manager_workspace._workspace_h5py_blosc2_options_match((1,), (2,))
        assert manager_workspace._workspace_h5py_dataset_matches_encoding(plain, {})
        assert not manager_workspace._workspace_h5py_dataset_matches_encoding(
            plain, {"compression": hdf5plugin.Blosc2.filter_id}
        )
        assert manager_workspace._workspace_h5py_dataset_matches_encoding(
            compressed, {"compression": hdf5plugin.Blosc2.filter_id}
        )
        assert manager_workspace._workspace_h5py_dataset_matches_encoding(
            compressed, manager_xarray._workspace_blosc2_encoding("zstd1")
        )
        assert not manager_workspace._workspace_h5py_dataset_matches_encoding(
            compressed, manager_xarray._workspace_blosc2_encoding("blosclz3")
        )
        gzip_filter = manager_workspace._workspace_h5py_filter_options(gzip_data)
        assert manager_workspace._workspace_h5py_dataset_matches_encoding(
            gzip_data,
            {"compression": 1, "compression_opts": gzip_filter[1]},
        )
        assert not manager_workspace._h5_group_matches_compression(
            h5_file, "missing", "none"
        )
        assert not manager_workspace._h5_group_matches_compression(
            h5_file, "plain", "none"
        )
        assert manager_workspace._h5_group_matches_compression(
            h5_file, "metadata", "none"
        )
        assert not manager_workspace._workspace_h5_group_matches_compression_mode(
            h5_file,
            "missing",
            xr.Dataset({"data": ("x", np.arange(3))}),
            "none",
        )
        assert not manager_workspace._workspace_h5_group_matches_compression_mode(
            h5_file,
            "payload",
            xr.Dataset({"missing": ("x", np.arange(3))}),
            "none",
        )
        assert not manager_workspace._workspace_h5_group_matches_compression_mode(
            h5_file,
            "payload",
            xr.Dataset({"data": ("x", np.arange(3))}),
            "none",
        )


def test_workspace_h5py_copy_rebuilds_attrs_and_dimension_scales(tmp_path) -> None:
    import h5py

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
    assert manager_workspace._workspace_h5py_type_contains_reference(array_type)
    assert array_member.closed

    plain_member = _FakeH5Type(h5py.h5t.INTEGER)
    compound_type = _FakeH5Type(h5py.h5t.COMPOUND, member_types=(plain_member,))
    assert not manager_workspace._workspace_h5py_type_contains_reference(compound_type)
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

        assert manager_workspace._workspace_h5py_attr_text(np.bytes_(b"x")) == "x"
        assert (
            manager_workspace._workspace_h5py_attr_text(
                types.SimpleNamespace(decode=lambda: "decoded")
            )
            == "decoded"
        )
        assert manager_workspace._workspace_h5py_attr_text("x") == "x"
        assert manager_workspace._workspace_h5py_attr_text(object()) is None

        manager_workspace._workspace_h5py_rebuild_dimension_scales(source, target)

        assert "reference" not in target.attrs
        assert "reference_array" not in target.attrs
        assert "reference_compound" not in target.attrs
        assert target["x"].attrs["_Netcdf4Dimid"] == 0
        assert len(target["values"].dims[0]) == 1


def test_copy_workspace_h5_group_to_open_file_edge_cases(tmp_path) -> None:
    import h5py

    fname = tmp_path / "copy.h5"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_dataset("dataset", data=np.arange(2))
        h5_file.create_group("source").create_dataset("data", data=np.arange(2))
        h5_file.create_group("target").create_group("source")

        assert not manager_workspace._copy_workspace_h5_group_to_open_file(
            h5_file, h5_file, "missing", "target/missing", None
        )
        assert not manager_workspace._copy_workspace_h5_group_to_open_file(
            h5_file, h5_file, "dataset", "target/dataset", None
        )
        assert manager_workspace._copy_workspace_h5_group_to_open_file(
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
                (manager_workspace._SAVED_TOOL_DATA_REFERENCE_DIM,),
                np.empty(0, dtype=np.float64),
            )
        }
    )
    monkeypatch.setattr(
        manager_workspace,
        "_workspace_h5py_create_dataset",
        lambda *_args, **_kwargs: None,
    )

    assert not manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/tool", ds
    )

    import h5py

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

    encoding = manager_xarray.workspace_dataset_encoding(ds)

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

    assert manager_xarray.workspace_dataset_encoding(ds, compression_mode="none") == {}
    assert manager_xarray.workspace_dataset_encoding(
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
    assert manager_xarray.workspace_dataset_encoding(ds, compression_mode="zstd1") == {
        "large": dict(
            hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    }
    assert manager_xarray.workspace_dataset_encoding(ds, compress=True) == {
        "large": dict(
            hdf5plugin.Blosc2(
                cname="zstd",
                clevel=1,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )
    }
    with pytest.raises(ValueError, match="Unknown workspace compression mode"):
        manager_xarray.workspace_dataset_encoding(
            ds,
            compression_mode=typing.cast(
                "manager_xarray.WorkspaceCompressionMode", "missing"
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
        assert not manager_xarray.workspace_compression_enabled()
        assert manager_xarray.workspace_dataset_encoding(ds) == {}

        erlab.interactive.options["io/workspace/compression"] = "blosclz3"
        assert manager_xarray.workspace_dataset_encoding(ds)["large"] == dict(
            hdf5plugin.Blosc2(
                cname="blosclz",
                clevel=3,
                filters=hdf5plugin.Blosc2.SHUFFLE,
            )
        )

        erlab.interactive.options["io/workspace/compression"] = "zstd1"
        assert manager_xarray.workspace_dataset_encoding(ds)["large"] == dict(
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

    assert manager_xarray.workspace_dataset_encoding(ds, compress=False) == {
        "data": {"chunksizes": (2, 4)}
    }


def test_workspace_chunksizes_rejects_invalid_chunk_shapes() -> None:
    assert (
        manager_xarray._workspace_chunksizes_for_dataarray(
            types.SimpleNamespace(chunks=((1,),), ndim=1, shape=(0,))
        )
        is None
    )
    assert (
        manager_xarray._workspace_chunksizes_for_dataarray(
            types.SimpleNamespace(chunks=((0,),), ndim=1, shape=(5,))
        )
        is None
    )


def test_workspace_datatree_encoding_uses_group_paths() -> None:
    large_ds = xr.Dataset(
        {
            "data": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            )
        },
        coords={"x": np.arange(512, dtype=np.float64), "y": np.arange(512)},
    )
    small_ds = xr.Dataset({"data": ("x", np.arange(4, dtype=np.float64))})
    tree = xr.DataTree.from_dict({"0/imagetool": large_ds, "1/imagetool": small_ds})
    try:
        encoding = manager_xarray.workspace_datatree_encoding(tree)
    finally:
        tree.close()

    assert set(encoding) == {"/0/imagetool"}
    assert set(encoding["/0/imagetool"]) == {"data"}


def test_workspace_datatree_encoding_can_be_disabled() -> None:
    tree = xr.DataTree.from_dict(
        {
            "0/imagetool": xr.Dataset(
                {
                    "data": (
                        ("x", "y"),
                        np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                    )
                }
            )
        }
    )
    try:
        assert manager_xarray.workspace_datatree_encoding(tree, compress=False) == {}
    finally:
        tree.close()


def test_workspace_xarray_path_helpers_cover_fallbacks(monkeypatch, tmp_path) -> None:
    class _BadPath(os.PathLike):
        def __fspath__(self) -> str:
            raise TypeError

    assert manager_xarray._normalized_file_path(object()) is None
    assert manager_xarray._normalized_file_path(_BadPath()) is None
    assert manager_xarray._normalized_file_path("") is None

    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert manager_xarray._normalized_file_path(tmp_path / "workspace.itws") == str(
        tmp_path / "workspace.itws"
    )

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    lock = manager_xarray._workspace_file_lock("fallback.itws")
    assert lock is manager_xarray._workspace_file_lock("fallback.itws")

    def _raise_stat_oserror(_path: str):
        raise OSError

    monkeypatch.setattr(manager_xarray.os, "stat", _raise_stat_oserror)
    assert manager_xarray._workspace_file_identity("missing.itws") == (
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
        manager_xarray, "ensure_workspace_hdf5_filters_registered", lambda: None
    )
    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(
        manager_xarray, "_workspace_file_identity", lambda path: (path, 0, 0, 0)
    )
    monkeypatch.setattr(manager_xarray.CachingFileManager, "__init__", _fake_init)

    file_manager = manager_xarray.WorkspaceFileManager("fallback.itws")

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

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(manager_xarray, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        manager_xarray, "_open_workspace_dataset_from_manager", _fake_open
    )

    assert (
        manager_xarray.open_workspace_dataset("fallback.itws", "/0", chunks=None)
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

    monkeypatch.setattr(manager_xarray, "_normalized_file_path", lambda _path: None)
    monkeypatch.setattr(manager_xarray, "WorkspaceFileManager", _FakeFileManager)
    monkeypatch.setattr(
        manager_xarray, "_iter_h5netcdf_group_paths", lambda _h5_file: ("/", "/broken")
    )
    monkeypatch.setattr(
        manager_xarray, "_open_workspace_dataset_from_manager", _fake_open
    )

    with pytest.raises(RuntimeError, match="broken group"):
        manager_xarray.open_workspace_datatree("fallback.itws", chunks="auto")

    assert closed == ["/"]


def test_write_full_workspace_tree_file_compresses_payload_not_coords(
    tmp_path,
) -> None:
    import h5py
    import hdf5plugin

    ds = xr.Dataset(
        {
            "data": (
                ("x", "y"),
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
            ),
            "small": ("x", np.arange(512, dtype=np.int64)),
        },
        coords={
            "x": np.linspace(-1.0, 1.0, 512),
            "y": np.linspace(-2.0, 2.0, 512),
        },
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    fname = tmp_path / "compressed.itws"
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, {"imagetool_workspace_schema_version": 4}
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        assert hdf5plugin.Blosc2.filter_id in _hdf5_filter_ids(
            h5_file["0/imagetool/data"]
        )
        assert _hdf5_filter_ids(h5_file["0/imagetool/x"]) == []
        assert _hdf5_filter_ids(h5_file["0/imagetool/y"]) == []
        assert _hdf5_filter_ids(h5_file["0/imagetool/small"]) == []

    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
        xarray.testing.assert_equal(loaded["x"], ds["x"])
        xarray.testing.assert_equal(loaded["y"], ds["y"])
    finally:
        opened.close()


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

    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        loaded = typing.cast("xr.DataTree", opened["/0/imagetool"]).to_dataset(
            inherit=False
        )
        xarray.testing.assert_equal(loaded["data"], ds["data"])
    finally:
        opened.close()


def test_imagetool_private_coord_serialization_edge_cases() -> None:
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    private_prefix = imagetool_serialization._PRIVATE_COORD_VAR_PREFIX
    data_name = imagetool_serialization.ITOOL_DATA_NAME
    valid_payload = json.dumps(
        [{"coord_name": "Fake Motor", "variable_name": "private", "dims": ["x"]}]
    )

    assert imagetool_serialization.private_coord_records_from_attrs(
        {private_attr: valid_payload.encode()}
    ) == ({"coord_name": "Fake Motor", "variable_name": "private", "dims": ("x",)},)
    assert (
        imagetool_serialization.private_coord_records_from_attrs({private_attr: 1})
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: "{not-json"}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: json.dumps([[]])}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_records_from_attrs(
            {private_attr: json.dumps([{"coord_name": "Fake Motor", "dims": ["x"]}])}
        )
        == ()
    )
    assert (
        imagetool_serialization.private_coord_variable_names(
            xr.Dataset({"other": ("x", [1.0])})
        )
        == ()
    )

    ds = xr.Dataset(
        {
            data_name: ("x", np.arange(2.0)),
            f"{private_prefix}0": ("x", np.arange(2.0) + 10.0),
        },
        coords={"x": np.arange(2.0), "Fake Motor": ("x", np.arange(2.0) + 20.0)},
    )
    encoded = imagetool_serialization.encode_private_coords(ds)

    assert imagetool_serialization.private_coord_variable_names(encoded) == (
        f"{private_prefix}1",
    )
    restored = imagetool_serialization.restore_private_coords(encoded)
    xr.testing.assert_equal(restored.coords["Fake Motor"], ds.coords["Fake Motor"])


def test_imagetool_private_coord_restore_ignores_invalid_records() -> None:
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR
    data_name = imagetool_serialization.ITOOL_DATA_NAME
    missing_data = xr.Dataset({"other": ("x", [1.0])})

    assert imagetool_serialization.restore_private_coords(missing_data) is missing_data

    payload = json.dumps(
        [
            {"coord_name": "Missing", "variable_name": "missing", "dims": ["x"]},
            {"coord_name": "Bad Dims", "variable_name": "present", "dims": ["z"]},
        ]
    )
    encoded = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "present": ("z", [2.0]),
        },
        attrs={"root": "kept"},
    )
    encoded[data_name].attrs[private_attr] = payload

    restored = imagetool_serialization.restore_private_coords(encoded)

    assert private_attr not in restored[data_name].attrs
    assert "Missing" not in restored.coords
    assert "Bad Dims" not in restored.coords
    assert "present" in restored.data_vars

    legacy = xr.Dataset(
        {
            data_name: ("x", [1.0]),
            "plain": ("x", [2.0]),
            "Fake Motor": ("z", [3.0]),
        }
    )

    legacy_restored = imagetool_serialization.restore_private_coords(legacy)

    assert "plain" in legacy_restored.data_vars
    assert "Fake Motor" in legacy_restored.data_vars


def test_workspace_h5py_attrs_and_root_validation(tmp_path) -> None:
    import h5py

    assert manager_workspace._h5py_attrs_to_dict({"name": b"value"}) == {
        "name": "value"
    }

    fname = tmp_path / "plain.h5"
    with h5py.File(fname, "w"):
        pass

    with pytest.raises(ValueError, match="Not a valid workspace file"):
        manager_workspace._read_workspace_root_attrs_h5py(fname)


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


def test_workspace_attr_native_detection_handles_edge_types() -> None:
    assert manager_workspace._workspace_attr_value_writes_natively(b"ok")
    assert not manager_workspace._workspace_attr_value_writes_natively(b"\xff")
    assert not manager_workspace._workspace_attr_value_writes_natively(b"a\x00")
    assert manager_workspace._workspace_attr_value_writes_natively(
        np.array([1, 2], dtype=np.int16)
    )
    assert not manager_workspace._workspace_attr_value_writes_natively(
        np.array([object()], dtype=object)
    )
    assert manager_workspace._workspace_attr_value_writes_natively(np.float64(1.0))
    assert not manager_workspace._workspace_attr_value_writes_natively(
        np.datetime64("2024-01-01")
    )
    assert manager_workspace._workspace_attr_value_writes_natively(("left", "right"))
    assert manager_workspace._workspace_attr_value_writes_natively((b"left", b"right"))
    assert manager_workspace._workspace_attr_value_writes_natively(
        (np.bool_(True), complex(1.0, 2.0))
    )
    assert not manager_workspace._workspace_attr_value_writes_natively([1, "text"])
    assert not manager_workspace._workspace_attr_value_writes_natively(
        ("text", b"bytes")
    )
    assert not manager_workspace._workspace_attr_value_writes_natively(([1],))


def test_workspace_mixed_scalar_attrs_use_typed_encoding() -> None:
    attrs = {
        "mixed_list": [1, "text"],
        "mixed_tuple": ("text", b"bytes"),
        "native_numbers": [1, 2.0],
    }

    serializable = manager_workspace._workspace_serializable_attrs(attrs)

    assert "mixed_list" not in serializable
    assert "mixed_tuple" not in serializable
    assert serializable["native_numbers"] == [1, 2.0]
    restored = manager_workspace._restore_workspace_serialized_attrs(serializable)
    assert restored["mixed_list"] == [1, "text"]
    assert restored["mixed_tuple"] == ("text", b"bytes")
    assert restored["native_numbers"] == [1, 2.0]


def test_workspace_attr_typed_encoding_roundtrips_safe_values(caplog) -> None:
    import decimal
    import math

    value = {
        None: None,
        False: True,
        np.int64(3): 5,
        7: np.float64(2.5),
        1.5: math.inf,
        complex(1.0, -2.0): -math.inf,
        "nan": math.nan,
        b"\xff": b"\x00\xff",
        ("tuple", 2): [
            np.array([[1, 2], [3, 4]], dtype=np.int16),
            np.array([{"nested": (None, complex(3.0, 4.0))}], dtype=object),
        ],
    }

    decoded = manager_workspace._workspace_decode_attr_value(
        manager_workspace._workspace_encode_attr_value(value)
    )

    assert decoded[None] is None
    assert decoded[False] is True
    assert decoded[3] == 5
    assert decoded[7] == np.float64(2.5)
    assert decoded[1.5] == math.inf
    assert decoded[complex(1.0, -2.0)] == -math.inf
    assert math.isnan(decoded["nan"])
    assert decoded[b"\xff"] == b"\x00\xff"
    np.testing.assert_array_equal(
        decoded[("tuple", 2)][0], np.array([[1, 2], [3, 4]], dtype=np.int16)
    )
    assert decoded[("tuple", 2)][1][0]["nested"] == (None, complex(3.0, 4.0))

    with pytest.raises(TypeError, match="unsupported attr key type"):
        manager_workspace._workspace_encode_attr_key(["bad"])
    with pytest.raises(TypeError, match="unsupported numeric attr type"):
        manager_workspace._workspace_encode_attr_value(decimal.Decimal("1.0"))
    with pytest.raises(TypeError, match="must be a mapping"):
        manager_workspace._workspace_decode_attr_value([])
    with pytest.raises(TypeError, match="unknown workspace attr value kind"):
        manager_workspace._workspace_decode_attr_value({"kind": "unknown"})
    with pytest.raises(TypeError, match="not hashable"):
        manager_workspace._workspace_decode_attr_key({"kind": "list", "items": []})

    assert manager_workspace._workspace_encoded_attr_entries(b"\xff") is None
    assert manager_workspace._workspace_encoded_attr_entries(1) is None
    assert manager_workspace._workspace_encoded_attr_entries("{bad-json") is None
    assert (
        manager_workspace._workspace_encoded_attr_entries(
            json.dumps({"version": -1, "attrs": []})
        )
        is None
    )
    assert (
        manager_workspace._workspace_encoded_attr_entries(
            json.dumps(
                {
                    "version": manager_workspace._WORKSPACE_ENCODED_ATTRS_VERSION,
                    "attrs": [["too-short"]],
                }
            )
        )
        is None
    )

    invalid_payload = json.dumps(
        {
            "version": manager_workspace._WORKSPACE_ENCODED_ATTRS_VERSION,
            "attrs": [[{"kind": "list", "items": []}, {"kind": "str", "value": "x"}]],
        }
    )
    with caplog.at_level(logging.WARNING):
        restored = manager_workspace._restore_workspace_serialized_attrs(
            {manager_workspace._WORKSPACE_ENCODED_ATTRS_ATTR: invalid_payload}
        )
    assert restored == {}
    assert "Ignoring invalid encoded workspace attribute" in caplog.text


def test_replace_h5_attrs_drops_invalid_attr_names(tmp_path) -> None:
    import h5py

    fname = tmp_path / "replace-invalid-attrs.itws"
    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("0/imagetool")
        group.attrs["old"] = "removed"

        manager_workspace._replace_h5_attrs(
            group.attrs,
            {"": "dropped", None: "dropped", "note": "", "valid": "kept"},
        )

        assert "old" not in group.attrs
        assert "" not in list(group.attrs)
        assert group.attrs["note"] == ""
        assert group.attrs["valid"] == "kept"


def test_replace_h5_attrs_encodes_non_native_attr_values(tmp_path) -> None:
    import h5py

    fname = tmp_path / "replace-rich-attrs.itws"
    rich_attr = _rich_workspace_attr_value()
    with h5py.File(fname, "w") as h5_file:
        group = h5_file.create_group("0/imagetool")

        manager_workspace._replace_h5_attrs(
            group.attrs,
            {"Single Motor Scan": rich_attr, "valid": "kept"},
        )

        assert "Single Motor Scan" not in group.attrs
        assert manager_workspace._WORKSPACE_ENCODED_ATTRS_ATTR in group.attrs
        decoded = manager_workspace._h5py_attrs_to_dict(group.attrs)
        assert decoded["valid"] == "kept"
        _assert_rich_workspace_attr(decoded["Single Motor Scan"])


def _assert_workspace_h5py_roundtrip(
    tmp_path: pathlib.Path, label: str, data: xr.DataArray
) -> tuple[xr.Dataset, xr.Dataset, pathlib.Path]:
    data_name = manager_workspace_io._ITOOL_DATA_NAME
    fname = tmp_path / f"{label}.itws"
    ds = data.rename(data_name).to_dataset()

    assert manager_workspace._workspace_dataset_can_write_h5py(ds)
    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=data_name,
    )
    assert loaded is not None

    opened = manager_xarray.open_workspace_dataset(fname, "0/imagetool", chunks=None)
    try:
        opened_loaded = opened.load()
    finally:
        opened.close()
    xr.testing.assert_equal(loaded, opened_loaded)
    return loaded, opened_loaded, fname


def test_workspace_h5py_fast_path_roundtrips_scalar_coords(tmp_path) -> None:
    import h5py

    fname = tmp_path / "scalar-fast-path.itws"
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(3.0), "temperature": 20.0},
        attrs={"coordinates": b""},
        name=manager_workspace_io._ITOOL_DATA_NAME,
    )
    ds = data.to_dataset()

    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname,
        "0/imagetool",
        preferred_data_name=manager_workspace_io._ITOOL_DATA_NAME,
    )

    assert loaded is not None
    expected = data.copy()
    expected.attrs.pop("coordinates")
    xr.testing.assert_equal(
        loaded[manager_workspace_io._ITOOL_DATA_NAME],
        expected,
    )
    assert loaded.coords["temperature"].item() == 20.0
    with h5py.File(fname, "r") as h5_file:
        saved_data = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
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
        manager_workspace._write_workspace_dataset_group_to_file(
            fname, "0/tool", data.to_dataset()
        )

    assert not any("space in its name" in str(item.message) for item in caught)
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "0/tool", preferred_data_name=data_name
    )

    assert loaded is not None
    xr.testing.assert_equal(
        loaded[data_name].coords["Fake Motor"], data.coords["Fake Motor"]
    )


def test_workspace_h5py_fast_path_roundtrips_saved_tool_extra_blob(
    tmp_path,
) -> None:
    import h5py
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

    manager_workspace._write_workspace_dataset_group_to_file(fname, "0/tool", ds)
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname,
        "0/tool",
        preferred_data_name=data_name,
    )

    assert loaded is not None
    assert manager_workspace._workspace_dataset_can_write_h5py(
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
        manager_workspace._write_workspace_dataset_group_to_file(
            blosclz_fname, "0/tool", ds
        )

        erlab.interactive.options["io/workspace/compression"] = "none"
        uncompressed_fname = tmp_path / "uncompressed-saved-tool-extra-blob.itws"
        manager_workspace._write_workspace_dataset_group_to_file(
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

    assert manager_workspace._write_workspace_dataset_group_h5py(fname, "0/tool", ds)
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
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
    import h5py

    data_name = manager_workspace_io._ITOOL_DATA_NAME
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
    data_name = manager_workspace_io._ITOOL_DATA_NAME
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

    assert manager_workspace._workspace_dataset_can_write_h5py(ds)
    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    loaded = manager_workspace._read_workspace_dataset_group_h5py(
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
    import h5py

    data_name = manager_workspace_io._ITOOL_DATA_NAME
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

    manager_workspace._write_workspace_dataset_group_to_file(fname, "0/imagetool", ds)

    assert ds.attrs["dataset_config"] is rich_attr
    assert ds[data_name].attrs["Single Motor Scan"] is rich_attr
    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        saved_data = group[data_name]
        assert "dataset_config" not in group.attrs
        assert "Single Motor Scan" not in saved_data.attrs
        assert manager_workspace._WORKSPACE_ENCODED_ATTRS_ATTR in group.attrs
        assert manager_workspace._WORKSPACE_ENCODED_ATTRS_ATTR in saved_data.attrs

    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "0/imagetool", preferred_data_name=data_name
    )
    assert loaded is not None
    _assert_rich_workspace_attr(loaded.attrs["dataset_config"])
    _assert_rich_workspace_attr(loaded[data_name].attrs["Single Motor Scan"])
    _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])
    _assert_rich_workspace_attr(loaded.coords["temperature"].attrs["scan_config"])

    opened = manager_xarray.open_workspace_dataset(fname, "0/imagetool", chunks=None)
    try:
        restored = manager_workspace._restore_workspace_dataset_attrs(opened.load())
    finally:
        opened.close()
    _assert_rich_workspace_attr(restored.attrs["dataset_config"])
    _assert_rich_workspace_attr(restored[data_name].attrs["Single Motor Scan"])
    _assert_rich_workspace_attr(restored.coords["x"].attrs["axis_config"])


def test_workspace_writer_drops_invalid_attr_names_from_fast_path(tmp_path) -> None:
    import h5py

    data_name = manager_workspace_io._ITOOL_DATA_NAME
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

    manager_workspace._write_workspace_dataset_group_to_file(fname, "0/imagetool", ds)

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

    loaded = manager_workspace._read_workspace_dataset_group_h5py(
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

    manager_workspace._write_workspace_dataset_group_to_file(fname, "0/tool", ds)

    opened = xr.open_dataset(fname, group="/0/tool", engine="h5netcdf")
    try:
        loaded = manager_workspace._restore_workspace_dataset_attrs(opened.load())
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
    data_name = manager_workspace_io._ITOOL_DATA_NAME
    private_attr = imagetool_serialization._PRIVATE_COORDS_ATTR

    assert not manager_workspace._workspace_dataset_can_write_h5py(
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
    assert not manager_workspace._workspace_dataset_can_write_h5py(missing_private)

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
    assert not manager_workspace._workspace_dataset_can_write_h5py(bad_private_dims)

    assert not manager_workspace._workspace_dataset_can_write_h5py(
        xr.Dataset(
            {data_name: ("x", [1.0])},
            coords={"x": np.array([object()], dtype=object)},
        )
    )

    bad_associated_dims = xr.Dataset(
        {data_name: ("x", [1.0])},
        coords={"x": [0.0], "z": [0.0], "bad": ("z", [1.0])},
    )
    assert not manager_workspace._workspace_dataset_can_write_h5py(bad_associated_dims)

    import dask.array as da

    chunked_coord = xr.Dataset(
        {data_name: ("x", [1.0, 2.0])},
        coords={
            "x": [0.0, 1.0],
            "chunked": ("x", da.from_array(np.array([1.0, 2.0]), chunks=(1,))),
        },
    )
    assert not manager_workspace._workspace_dataset_can_write_h5py(chunked_coord)

    monkeypatch.setattr(
        manager_workspace, "_workspace_dataset_can_write_h5py", lambda _ds: True
    )
    assert not manager_workspace._write_workspace_dataset_group_h5py(
        tmp_path / "no-data-name.itws", "0/imagetool", xr.Dataset()
    )

    bad_attrs = xr.Dataset({data_name: ("x", [1.0])}, coords={"x": [0.0]})
    bad_attrs.attrs["bad"] = object()
    fname = tmp_path / "bad-attrs.itws"
    with caplog.at_level(logging.WARNING, logger=manager_workspace.logger.name):
        assert manager_workspace._write_workspace_dataset_group_h5py(
            fname, "0/imagetool", bad_attrs
        )
    assert "unsupported value type object" in caplog.text
    import h5py

    with h5py.File(fname, "r") as h5_file:
        assert "0/imagetool" in h5_file
        assert "bad" not in h5_file["0/imagetool"].attrs


def test_workspace_h5py_reader_rejects_malformed_groups(tmp_path) -> None:
    import h5py

    data_name = manager_workspace_io._ITOOL_DATA_NAME
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

    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "missing") is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "not-a-group")
        is None
    )
    assert manager_workspace._read_workspace_dataset_group_h5py(fname, "multi") is None
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "no-dims") is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "bad-scale") is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(fname, "missing-scalar")
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "missing-private", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-private", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-associated-no-scale", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-associated-length", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-associated-foreign-dim", preferred_data_name=data_name
        )
        is None
    )
    assert (
        manager_workspace._read_workspace_dataset_group_h5py(
            fname, "bad-time-metadata", preferred_data_name=data_name
        )
        is None
    )


def test_workspace_h5py_reader_restores_legacy_spaced_coords(tmp_path) -> None:
    import h5py

    data_name = manager_workspace_io._ITOOL_DATA_NAME
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

    loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "valid", preferred_data_name=data_name
    )
    assert loaded is not None
    np.testing.assert_allclose(loaded.coords["Fake Motor"].values, [10.0, 11.0])
    duplicate_loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "duplicate", preferred_data_name=data_name
    )
    assert duplicate_loaded is not None
    np.testing.assert_allclose(
        duplicate_loaded.coords["Fake Motor"].values, [20.0, 21.0]
    )
    invalid_loaded = manager_workspace._read_workspace_dataset_group_h5py(
        fname, "invalid", preferred_data_name=data_name
    )
    assert invalid_loaded is not None
    assert "Fake Motor" not in invalid_loaded


def test_workspace_h5py_writer_replaces_groups_and_preserves_attrs(tmp_path) -> None:
    import h5py

    data_name = manager_workspace_io._ITOOL_DATA_NAME
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

    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )
    assert manager_workspace._write_workspace_dataset_group_h5py(
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


def test_manager_load_workspace_dataset_ignores_invalid_saved_metadata(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": np.arange(2.0), "y": np.arange(2.0)},
    )
    saved = itool(data, manager=False, execute=False)
    qtbot.addWidget(saved)
    assert isinstance(saved, erlab.interactive.imagetool.ImageTool)
    ds = saved.to_dataset()
    ds.attrs["manager_node_uid"] = "loaded"
    ds.attrs["manager_node_provenance_spec"] = "{not-json"
    ds.attrs["manager_node_live_source_spec"] = "{not-json"
    ds.attrs["manager_node_live_source_binding"] = "{not-json"

    with manager_context() as manager:
        target = manager._load_workspace_imagetool_dataset(
            ds, parent_target=None, node_path="-1"
        )

        assert target in manager._tool_graph.root_wrappers
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        binding = provenance.ImageToolSelectionSourceBinding(
            selection_mode="isel",
            selection_indexers={"x": 0},
        )
        bound_ds = saved.to_dataset()
        bound_ds.attrs["manager_node_uid"] = "bound"
        bound_ds.attrs.pop("manager_node_live_source_spec", None)
        bound_ds.attrs["manager_node_live_source_binding"] = json.dumps(
            binding.model_dump(mode="json")
        )
        bound_ds.attrs.pop("itool_name", None)

        bound_target = manager._load_workspace_imagetool_dataset(
            bound_ds, parent_target=None, node_path="-2"
        )

        assert manager._node_for_target(bound_target).source_binding == binding
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)


def test_manager_load_workspace_tool_dataset_rejects_root_tool(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="data")
    tool = _AddedTimeChildTool(data)
    qtbot.addWidget(tool)
    ds = tool.to_dataset()

    with (
        manager_context() as manager,
        pytest.raises(ValueError, match="Workspace tool node has no parent"),
    ):
        manager._load_workspace_tool_dataset(ds, parent_target=None)


def test_manager_workspace_tool_data_reference_roundtrip(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [10.0, 11.0, 12.0]},
        name="source",
    )

    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        child = _AddedTimeChildTool(data.copy(deep=False))
        child.set_source_binding(provenance.full_data())
        child_uid = manager.add_childtool(child, 0, show=False)

        tree = manager._to_datatree()
        try:
            ds = typing.cast(
                "xr.DataTree", tree[f"0/childtools/{child_uid}/tool"]
            ).to_dataset(inherit=False)
            references = json.loads(
                ds.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR]
            )
            assert erlab.interactive.utils._SAVED_TOOL_DATA_NAME in references
            assert ds[erlab.interactive.utils._SAVED_TOOL_DATA_NAME].size == 0
            with pytest.raises(ValueError, match="parent data is unavailable"):
                erlab.interactive.utils.ToolWindow.from_dataset(ds)

            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            assert manager._from_datatree(
                tree, replace=True, mark_dirty=False, select=False
            )
        finally:
            tree.close()

        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, _AddedTimeChildTool)
        xr.testing.assert_identical(loaded_child.tool_data, data)


def test_manager_workspace_tool_data_reference_falls_back_on_shape_mismatch(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    parent_data = xr.DataArray(
        np.arange(50.0).reshape(2, 5, 5),
        dims=("z", "y", "x"),
        name="source",
    )
    child_data = parent_data.isel(z=0, drop=True).rename("source")

    with manager_context() as manager:
        root = itool(parent_data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        child = DerivativeTool(child_data)
        child.set_source_binding(provenance.full_data())
        child_uid = manager.add_childtool(child, 0, show=False)

        tree = manager._to_datatree()
        try:
            ds = typing.cast(
                "xr.DataTree", tree[f"0/childtools/{child_uid}/tool"]
            ).to_dataset(inherit=False)
            assert erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR not in ds.attrs
            xr.testing.assert_identical(
                ds[imagetool_serialization.SAVED_TOOL_DATA_NAME].rename(
                    child.tool_data.name
                ),
                child.tool_data,
            )
        finally:
            tree.close()


def test_manager_workspace_partially_loads_corrupted_child_with_warning(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dialogs: list[typing.Any] = []

    class _RecordingMessageDialog(QtWidgets.QDialog):
        def __init__(self, parent=None, **kwargs) -> None:
            super().__init__(parent)
            self.parent = parent
            self.kwargs = kwargs
            dialogs.append(self)

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )

    data = xr.DataArray(
        np.arange(25.0).reshape(5, 5),
        dims=("y", "x"),
        name="source",
    )
    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        child = DerivativeTool(data)
        child_uid = manager.add_childtool(child, 0, show=False)

        tree = manager._to_datatree()
        try:
            root_ds = typing.cast("xr.DataTree", tree["0/imagetool"]).to_dataset(
                inherit=False
            )
            child_ds = typing.cast(
                "xr.DataTree", tree[f"0/childtools/{child_uid}/tool"]
            ).to_dataset(inherit=False)
            child_ds = child_ds.copy(deep=True)
            saved_data_name = imagetool_serialization.SAVED_TOOL_DATA_NAME
            child_ds[saved_data_name] = xr.DataArray(
                np.arange(50.0).reshape(2, 5, 5),
                dims=("z", "y", "x"),
                name=saved_data_name,
            )
            corrupted_tree = xr.DataTree.from_dict(
                {
                    "0/imagetool": root_ds,
                    f"0/childtools/{child_uid}/tool": child_ds,
                }
            )
            corrupted_tree.attrs.update(tree.attrs)
        finally:
            tree.close()

        assert manager._from_datatree(
            corrupted_tree,
            replace=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == []

        assert manager._finish_workspace_file_load(True)

        partial_dialogs = [
            dialog
            for dialog in dialogs
            if dialog.kwargs.get("title") == "Workspace Partially Loaded"
        ]
        assert len(partial_dialogs) == 1
        dialog = partial_dialogs[0]
        assert dialog.parent is manager
        assert f"0/childtools/{child_uid}" in dialog.kwargs["informative_text"]
        assert "Input DataArray must be 2D" in dialog.kwargs["detailed_text"]


def test_manager_workspace_no_loaded_windows_error_without_skipped_nodes(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        controller = manager._workspace_controller
        controller._skipped_workspace_nodes = []
        with pytest.raises(ValueError, match="No workspace windows") as exc_info:
            controller._raise_no_workspace_windows_loaded()
        assert exc_info.value.__cause__ is None


def test_manager_load_workspace_figures_counts_loaded_and_skipped(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    tree = xr.DataTree.from_dict(
        {
            "figures/loaded": xr.Dataset(),
            "figures/skipped": xr.Dataset(),
        }
    )
    manifest = {
        "nodes": [
            {"path": "figures/missing"},
            {"path": "figures/loaded"},
            {"path": "figures/skipped"},
        ]
    }
    calls: list[str | None] = []

    def _fake_load_workspace_node_or_warn(*_args, **kwargs):
        calls.append(kwargs.get("node_path"))
        if kwargs.get("node_path") == "figures/skipped":
            return None
        return "figure-target"

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._workspace_controller,
            "_load_workspace_node_or_warn",
            _fake_load_workspace_node_or_warn,
        )
        assert manager._load_workspace_figures(tree, manifest=manifest) == 1

    assert calls == ["figures/loaded", "figures/skipped"]


def test_manager_from_h5py_workspace_manifest_validation(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "missing.itws"

    with manager_context() as manager:
        with pytest.raises(TypeError, match="missing node ordering"):
            manager._from_h5py_workspace_file(
                fname, {}, replace=False, mark_dirty=False
            )
        with pytest.raises(ValueError, match="no loadable nodes"):
            manager._from_h5py_workspace_file(
                fname,
                {
                    "nodes": [
                        [],
                        {"path": 0, "kind": "imagetool"},
                        {"path": "0", "kind": "unknown"},
                    ],
                    "root_order": [],
                },
                replace=False,
                mark_dirty=False,
            )
        with pytest.raises(ValueError, match="no loadable root nodes"):
            manager._from_h5py_workspace_file(
                fname,
                {
                    "nodes": [{"path": "0/childtools/tool", "kind": "tool"}],
                    "root_order": [],
                },
                replace=False,
                mark_dirty=False,
            )


def test_manager_from_h5py_workspace_falls_back_after_fast_read_error(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "fallback-load.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        def _raise_fast_read(*_args, **_kwargs):
            raise RuntimeError("fast path failed")

        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_dataset_group_h5py",
            _raise_fast_read,
        )

        assert manager._from_h5py_workspace_file(
            fname, manifest, replace=True, mark_dirty=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_load_workspace_file_falls_back_after_fast_path_error(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "load-fallback.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        def _raise_fast_load(*_args, **_kwargs) -> bool:
            raise RuntimeError("fast load failed")

        monkeypatch.setattr(manager, "_from_h5py_workspace_file", _raise_fast_load)

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._workspace_state.path == fname.resolve()


def test_manager_from_h5py_workspace_logs_restore_failure(
    qtbot,
    caplog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "restore-failure.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)

        def _raise_load(*_args, **_kwargs):
            raise RuntimeError("load failed")

        def _raise_restore(*_args, **_kwargs):
            raise RuntimeError("restore failed")

        monkeypatch.setattr(manager, "_load_workspace_imagetool_dataset", _raise_load)
        monkeypatch.setattr(
            manager._workspace_controller, "_restore_replaced_workspace", _raise_restore
        )

        with (
            caplog.at_level(logging.ERROR, logger=manager_workspace_io.logger.name),
            pytest.raises(ValueError, match="No workspace windows") as exc_info,
        ):
            manager._from_h5py_workspace_file(
                fname, manifest, replace=True, mark_dirty=False
            )
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "load failed"
        assert "Failed to restore previous workspace" in caplog.text
        assert "restore failed" in caplog.text


def test_manager_workspace_rebind_skips_missing_snapshot_and_keeps_chunks(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(16.0).reshape(4, 4), dims=("x", "y")).chunk(
            {"x": 2}
        )
        root = itool(data, manager=False, execute=False, auto_compute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid
        calls: list[typing.Any] = []

        def _fake_rebind_data(_fname, _uid, *, chunks):
            calls.append(chunks)
            return data

        monkeypatch.setattr(
            manager, "_workspace_rebind_data_for_uid", _fake_rebind_data
        )

        manager._rebind_workspace_backed_imagetools(
            tmp_path / "workspace.itws", backing_snapshot={}
        )
        assert calls == []

        manager._rebind_workspace_backed_imagetools(tmp_path / "workspace.itws")

        assert uid in manager._tool_graph.nodes
        assert calls == [{}]


def test_manager_workspace_full_save_copy_group_edge_cases(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace_path = tmp_path / "copy-groups.itws"
    workspace_path.touch()

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager._workspace_state.path = workspace_path
        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_root_attrs_h5py",
            lambda _path: (_ for _ in ()).throw(RuntimeError("metadata failed")),
        )
        assert manager._workspace_controller._workspace_full_save_copy_groups(
            xr.DataTree()
        ) == (None, ())

        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_root_attrs_h5py",
            lambda _path: {"imagetool_workspace_schema_version": 1},
        )
        assert manager._workspace_controller._workspace_full_save_copy_groups(
            xr.DataTree()
        ) == (None, ())

        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid
        tool = DerivativeTool(data)
        monkeypatch.setattr(tool, "can_save_and_load", lambda: False)
        manager.add_childtool(tool, 0, show=False)
        manager.add_childtool(_AddedTimeChildTool(data), 0, show=False)
        manager._mark_workspace_clean()
        tree = manager._to_datatree()
        monkeypatch.setitem(
            manager._tool_graph.nodes,
            "missing-tool",
            types.SimpleNamespace(
                is_imagetool=False,
                tool_window=None,
                pending_workspace_tool_payload=None,
            ),
        )
        try:
            manifest_without_identities = {
                "schema_version": manager_workspace._current_workspace_schema_version(),
                "nodes": [[]],
                "root_order": [0],
            }
            monkeypatch.setattr(
                manager_workspace,
                "_read_workspace_root_attrs_h5py",
                lambda _path: {
                    "imagetool_workspace_schema_version": (
                        manager_workspace._current_workspace_schema_version()
                    ),
                    manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(
                        manifest_without_identities
                    ),
                },
            )
            assert manager._workspace_controller._workspace_full_save_copy_groups(
                tree
            ) == (
                str(workspace_path),
                (),
            )

            manifest_with_missing_tree_payload = {
                "schema_version": manager_workspace._current_workspace_schema_version(),
                "nodes": [
                    [],
                    {"uid": uid, "kind": "imagetool", "path": "0"},
                ],
                "root_order": [0],
            }
            monkeypatch.setattr(
                manager_workspace,
                "_read_workspace_root_attrs_h5py",
                lambda _path: {
                    "imagetool_workspace_schema_version": (
                        manager_workspace._current_workspace_schema_version()
                    ),
                    manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(
                        manifest_with_missing_tree_payload
                    ),
                },
            )
            assert manager._workspace_controller._workspace_full_save_copy_groups(
                xr.DataTree()
            ) == (
                str(workspace_path),
                (),
            )
        finally:
            tree.close()
            manager._tool_graph.nodes.pop("missing-tool", None)


def test_prepare_workspace_transaction_promotes_missing_attr_fallback(
    tmp_path,
) -> None:
    fname = tmp_path / "fallback.itws"
    _write_transaction_test_workspace(fname)
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )
    rewrite_map: dict[str, tuple[str, dict[str, xr.Dataset]]] = {}

    group_operations, attr_updates = manager_workspace._prepare_workspace_transaction(
        fname,
        f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}fallback",
        f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}fallback",
        f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}fallback",
        rewrite_map,
        (("0/missing", {"itool_title": "new"}, fallback),),
        _transaction_test_root_attrs(delta_save_count=1),
    )

    assert rewrite_map == {"0": fallback}
    assert attr_updates == []
    assert group_operations[0]["group_path"] == "0"
    manager_workspace._recover_workspace_transactions(fname)
    _assert_no_workspace_internal_groups(fname)


def test_write_full_workspace_tree_file_skips_missing_copy_source_group(
    tmp_path,
) -> None:
    fname = tmp_path / "missing-copy-group.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(3.0, title="rewritten")}
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("missing/source", "0/imagetool", None),),
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 3.0


def test_write_full_workspace_tree_file_replaces_stale_root_attrs(tmp_path) -> None:
    import h5py

    fname = tmp_path / "root-attrs.itws"
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(1.0, title="old")}
    )
    tree.attrs["stale_workspace_attr"] = "remove me"
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        assert "stale_workspace_attr" not in h5_file.attrs
        manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest == {"schema_version": 4, "root_order": [0], "nodes": []}


def test_write_full_workspace_tree_file_local_path_uses_destination_temp(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "local.itws"
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(1.0, title="local")}
    )
    write_targets: list[pathlib.Path] = []
    original_write = manager_workspace._write_workspace_dataset_group_to_file

    def _record_write(target, *args, **kwargs):
        write_targets.append(pathlib.Path(target))
        return original_write(target, *args, **kwargs)

    monkeypatch.setattr(
        manager_workspace, "_write_workspace_dataset_group_to_file", _record_write
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: False
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert write_targets
    assert all(target.parent == fname.parent for target in write_targets)
    assert all(target.name.startswith(f"{fname.name}.tmp-") for target in write_targets)


def test_write_full_workspace_tree_file_cloud_path_uses_scratch_and_replace_first(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "Dropbox" / "cloud.itws"
    fname.parent.mkdir()
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(2.0, title="cloud")}
    )
    write_targets: list[pathlib.Path] = []
    replace_calls: list[tuple[pathlib.Path, pathlib.Path]] = []
    original_write = manager_workspace._write_workspace_dataset_group_to_file

    def _record_write(target, *args, **kwargs):
        write_targets.append(pathlib.Path(target))
        return original_write(target, *args, **kwargs)

    def _replace_by_copy(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        replace_calls.append((src_path, dst_path))
        dst_path.write_bytes(src_path.read_bytes())
        src_path.unlink()

    monkeypatch.setattr(
        manager_workspace, "_write_workspace_dataset_group_to_file", _record_write
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(manager_workspace.os, "replace", _replace_by_copy)
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert write_targets
    assert all(target.parent != fname.parent for target in write_targets)
    assert replace_calls == [(write_targets[0], fname)]
    assert _read_transaction_test_value(fname) == 2.0


def test_write_full_workspace_tree_file_copies_unchanged_payload_groups(
    monkeypatch,
    tmp_path,
) -> None:
    import h5py

    fname = tmp_path / "copy.itws"
    ds = xr.Dataset(
        {
            manager_workspace_io._ITOOL_DATA_NAME: (
                ("x", "y"),
                np.arange(12, dtype=np.float64).reshape(3, 4),
            )
        },
        coords={
            "x": np.arange(3, dtype=np.float64),
            "y": np.arange(4, dtype=np.float64),
        },
        attrs={
            "itool_title": "old",
            "manager_node_uid": "n0",
            "manager_node_kind": "imagetool",
        },
    )
    tree = xr.DataTree.from_dict({"0/imagetool": ds})
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    rewritten = ds.assign_attrs(
        {
            "itool_title": "new",
            "manager_node_uid": "n0",
            "manager_node_kind": "imagetool",
            "Single Motor Scan": _rich_workspace_attr_value(),
        }
    )
    tree = xr.DataTree.from_dict({"0/imagetool": rewritten})

    def _fail_to_netcdf(*_args, **_kwargs):
        raise AssertionError("unchanged payload should be copied with h5py")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _fail_to_netcdf)
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0/imagetool", "0/imagetool", dict(rewritten.attrs)),),
        )
    finally:
        tree.close()

    with h5py.File(fname, "r") as h5_file:
        group = h5_file["0/imagetool"]
        assert group.attrs["itool_title"] == "new"
        decoded_attrs = manager_workspace._h5py_attrs_to_dict(group.attrs)
        _assert_rich_workspace_attr(decoded_attrs["Single Motor Scan"])
        np.testing.assert_array_equal(
            group[manager_workspace_io._ITOOL_DATA_NAME][...],
            np.arange(12, dtype=np.float64).reshape(3, 4),
        )
    opened = manager_xarray.open_workspace_datatree(fname, chunks=None)
    try:
        xr.testing.assert_identical(
            opened["0/imagetool"].to_dataset()[manager_workspace_io._ITOOL_DATA_NAME],
            rewritten[manager_workspace_io._ITOOL_DATA_NAME],
        )
    finally:
        opened.close()


def test_write_full_workspace_tree_file_network_scratch_skips_copy_reuse(
    monkeypatch, tmp_path
) -> None:
    import shutil

    fname = tmp_path / "network-copy-reuse.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(3.0, title="rewritten")}
    )

    def _fail_copyfile(*_args, **_kwargs):
        raise AssertionError("network scratch save should not copy old workspace")

    def _replace_by_copy(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        dst_path.write_bytes(src_path.read_bytes())
        src_path.unlink()

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: True
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: False
    )
    monkeypatch.setattr(shutil, "copyfile", _fail_copyfile)
    monkeypatch.setattr(manager_workspace.os, "replace", _replace_by_copy)
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0/imagetool", "0/imagetool", None),),
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 3.0


def test_write_full_workspace_tree_file_scratch_exdev_fallback(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "fallback.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(4.0, title="fallback")}
    )
    original_replace = manager_workspace.os.replace
    replace_calls: list[tuple[pathlib.Path, pathlib.Path]] = []
    scratch_path: pathlib.Path | None = None

    def _replace_with_exdev(src, dst):
        nonlocal scratch_path
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        replace_calls.append((src_path, dst_path))
        if dst_path == fname and src_path.parent != fname.parent:
            scratch_path = src_path
            raise OSError(errno.EXDEV, "cross-device link")
        return original_replace(src, dst)

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(manager_workspace.os, "replace", _replace_with_exdev)
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 4.0
    assert scratch_path is not None
    assert not scratch_path.exists()
    assert len(replace_calls) == 2
    assert replace_calls[0] == (scratch_path, fname)
    assert replace_calls[1][0].parent == fname.parent
    assert replace_calls[1][1] == fname
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_write_full_workspace_tree_file_rejects_file_repack_on_network_path(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "network.itws"
    _write_transaction_test_workspace(fname)

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_high_risk", lambda *_: True
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda *_: True
    )

    with pytest.raises(ValueError, match="File-level workspace repack cannot run"):
        manager_workspace._write_full_workspace_tree_file(
            fname,
            None,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0", "0", None),),
        )
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(5.0, title="network")}
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            fname,
            tree,
            _transaction_test_root_attrs(),
            copy_source=fname,
            copy_groups=(("0", "0", None),),
        )
    finally:
        tree.close()
    assert _read_transaction_test_value(fname) == 5.0


def test_write_full_workspace_tree_file_scratch_replace_failure_preserves_old(
    monkeypatch, tmp_path
) -> None:
    fname = tmp_path / "replace-failure.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(5.0, title="failure")}
    )
    scratch_paths: list[pathlib.Path] = []

    def _fail_replace(src, dst):
        src_path = pathlib.Path(src)
        if pathlib.Path(dst) == fname:
            scratch_paths.append(src_path)
        raise OSError(errno.EPERM, "replace failed")

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(manager_workspace.os, "replace", _fail_replace)
    try:
        with pytest.raises(OSError, match="replace failed"):
            manager_workspace._write_full_workspace_tree_file(
                fname, tree, _transaction_test_root_attrs()
            )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 1.0
    assert scratch_paths
    assert all(not scratch_path.exists() for scratch_path in scratch_paths)
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_write_full_workspace_tree_file_scratch_copy_failure_cleans_destination_tmp(
    monkeypatch, tmp_path
) -> None:
    import shutil

    fname = tmp_path / "copy-failure.itws"
    _write_transaction_test_workspace(fname)
    tree = xr.DataTree.from_dict(
        {"0/imagetool": _transaction_test_dataset(6.0, title="failure")}
    )
    original_replace = manager_workspace.os.replace
    scratch_paths: list[pathlib.Path] = []

    def _replace_with_exdev(src, dst):
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        if dst_path == fname and src_path.parent != fname.parent:
            scratch_paths.append(src_path)
            raise OSError(errno.EXDEV, "cross-device link")
        return original_replace(src, dst)

    def _fail_copyfile(src, dst):
        pathlib.Path(dst).write_bytes(b"partial")
        raise OSError(errno.EIO, "copy failed")

    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_network_path", lambda _path: False
    )
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_likely_cloud_path", lambda _path: True
    )
    monkeypatch.setattr(manager_workspace.os, "replace", _replace_with_exdev)
    monkeypatch.setattr(shutil, "copyfile", _fail_copyfile)
    try:
        with pytest.raises(OSError, match="copy failed"):
            manager_workspace._write_full_workspace_tree_file(
                fname, tree, _transaction_test_root_attrs()
            )
    finally:
        tree.close()

    assert _read_transaction_test_value(fname) == 1.0
    assert scratch_paths
    assert all(not scratch_path.exists() for scratch_path in scratch_paths)
    assert not list(fname.parent.glob(f"{fname.name}.tmp-*"))


def test_workspace_recovery_discards_pending_only_transaction(tmp_path) -> None:
    fname = tmp_path / "pending-only.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "pendingonly"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"

    manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_restores_backup_before_pending_move(tmp_path) -> None:
    import h5py

    fname = tmp_path / "backup-before-pending.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "backuponly"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, _ = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        operation = group_operations[0]
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["group_path"]),
            typing.cast("str", operation["backup_path"]),
        )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_rolls_back_active_moved_before_commit(tmp_path) -> None:
    import h5py

    fname = tmp_path / "active-before-commit.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "activemoved"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    group_operations, _ = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        _transaction_test_root_attrs(delta_save_count=1),
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        operation = group_operations[0]
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["group_path"]),
            typing.cast("str", operation["backup_path"]),
        )
        manager_workspace._move_h5_path(
            h5_file,
            typing.cast("str", operation["pending_path"]),
            typing.cast("str", operation["group_path"]),
        )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 1.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_accepts_committed_before_cleanup(tmp_path) -> None:
    fname = tmp_path / "committed-before-cleanup.itws"
    _write_transaction_test_workspace(fname)
    rewrite = ("0", {"0/imagetool": _transaction_test_dataset(2.0, title="new")})
    rewrite_map = {"0": rewrite}
    txn_id = "committed"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    root_attrs = _transaction_test_root_attrs(delta_save_count=1)
    group_operations, attr_updates = manager_workspace._prepare_workspace_transaction(
        fname,
        txn_path,
        pending_root,
        backup_root,
        rewrite_map,
        (),
        root_attrs,
    )
    manager_workspace._write_workspace_transaction_pending_groups(
        fname, rewrite_map, pending_root
    )
    manager_workspace._commit_workspace_transaction(
        fname, txn_path, group_operations, attr_updates, root_attrs
    )

    manager_workspace._recover_workspace_transactions(fname)

    assert _read_transaction_test_value(fname) == 2.0
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_rolls_back_attr_only_transaction(tmp_path) -> None:
    import h5py

    fname = tmp_path / "attrs-before-commit.itws"
    _write_transaction_test_workspace(fname)
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )
    attr_update = ("0/imagetool", {"itool_title": "new"}, fallback)
    txn_id = "attrrollback"
    txn_path = f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}{txn_id}"
    pending_root = f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}{txn_id}"
    backup_root = f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}{txn_id}"
    root_attrs = _transaction_test_root_attrs(delta_save_count=1)
    _, attr_updates = manager_workspace._prepare_workspace_transaction(
        fname, txn_path, pending_root, backup_root, {}, (attr_update,), root_attrs
    )

    with h5py.File(fname, "a") as h5_file:
        manager_workspace._set_workspace_transaction_status(
            h5_file,
            txn_path,
            "committing",
        )
        manager_workspace._replace_h5_attrs(
            h5_file["0/imagetool"].attrs, attr_updates[0][1]
        )
        manager_workspace._write_root_attrs_to_open_workspace_file(h5_file, root_attrs)
        h5_file.flush()

    manager_workspace._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert h5_file["0/imagetool"].attrs["itool_title"] == "old"
        assert (
            manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs) == 0
        )
    _assert_no_workspace_internal_groups(fname)


def test_workspace_transaction_attr_update_encodes_non_native_values(tmp_path) -> None:
    import h5py

    fname = tmp_path / "rich-attrs-transaction.itws"
    _write_transaction_test_workspace(fname)
    fallback = (
        "0",
        {"0/imagetool": _transaction_test_dataset(2.0, title="fallback")},
    )
    rich_attr = _rich_workspace_attr_value()
    manager_workspace._write_workspace_transaction_file(
        fname,
        (),
        (
            (
                "0/imagetool",
                {"itool_title": "new", "Single Motor Scan": rich_attr},
                fallback,
            ),
        ),
        _transaction_test_root_attrs(delta_save_count=1),
    )

    with h5py.File(fname, "r") as h5_file:
        decoded_attrs = manager_workspace._h5py_attrs_to_dict(
            h5_file["0/imagetool"].attrs
        )
        assert decoded_attrs["itool_title"] == "new"
        _assert_rich_workspace_attr(decoded_attrs["Single Motor Scan"])
        assert (
            manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs) == 1
        )
    _assert_no_workspace_internal_groups(fname)


def test_workspace_recovery_cleans_orphan_internal_groups(tmp_path) -> None:
    import h5py

    fname = tmp_path / "orphan-internal.itws"
    _write_transaction_test_workspace(fname)
    with h5py.File(fname, "a") as h5_file:
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}orphan"
        )
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}orphan"
        )

    manager_workspace._recover_workspace_transactions(fname)

    _assert_no_workspace_internal_groups(fname)


def test_workspace_lock_path_uses_hidden_sidecar(tmp_path) -> None:
    fname = tmp_path / "example.itws"

    assert manager_workspace._workspace_lock_path(fname) == str(
        (tmp_path / ".example.itws.lock").resolve()
    )


def test_workspace_lock_conflict_is_reported(tmp_path) -> None:
    fname = tmp_path / "locked.itws"
    _write_transaction_test_workspace(fname)
    hidden_lock_path = pathlib.Path(manager_workspace._workspace_lock_path(fname))
    visible_lock_path = pathlib.Path(f"{fname.resolve()}.lock")
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    try:
        assert lock.staleLockTime() == 0
        assert hidden_lock_path.exists()
        assert not visible_lock_path.exists()
        with pytest.raises(BlockingIOError):
            manager_workspace._acquire_workspace_document_lock(fname)
    finally:
        lock.unlock()


def test_hide_workspace_lock_file_sets_macos_hidden_flag(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    lock_path = "/workspace/.workspace.itws.lock"
    regular_stat = types.SimpleNamespace(st_mode=0o100600)

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    monkeypatch.setattr(manager_workspace.os, "lstat", lambda _path: regular_stat)
    monkeypatch.setattr(
        manager_workspace.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    manager_workspace._hide_workspace_lock_file(lock_path)

    assert calls == [(lock_path, 0x8000)]


def test_hide_workspace_lock_file_skips_macos_symlink(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    symlink_stat = types.SimpleNamespace(st_mode=0o120777)

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    monkeypatch.setattr(manager_workspace.os, "lstat", lambda _path: symlink_stat)
    monkeypatch.setattr(
        manager_workspace.os,
        "chflags",
        lambda path, flags: calls.append((path, flags)),
        raising=False,
    )

    manager_workspace._hide_workspace_lock_file("/workspace/.workspace.itws.lock")

    assert calls == []


def test_workspace_lock_error_message_names_owner(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "busy-message.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    lock_info = manager_workspace._workspace_document_lock_info(fname)
    calls: list[dict[str, object]] = []

    def _critical(*args, **kwargs) -> int:
        calls.append({"args": args, "kwargs": kwargs})
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)
    try:
        manager_widgets._show_workspace_file_lock_error(None, fname)
    finally:
        lock.unlock()

    assert len(calls) == 1
    args = calls[0]["args"]
    assert isinstance(args, tuple)
    assert args[1] == "Workspace Already Open"
    assert args[2] == "This workspace is already open somewhere else."
    informative_text = args[3]
    assert isinstance(informative_text, str)
    assert fname.name in informative_text
    assert "lock" not in informative_text.casefold()
    if lock_info.owner:
        assert lock_info.owner in informative_text
    if lock_info.hostname:
        assert lock_info.hostname in informative_text
    detailed_text = calls[0]["kwargs"]["detailed_text"]
    assert isinstance(detailed_text, str)
    assert "Temporary workspace ownership marker:" in detailed_text
    assert lock_info.path in detailed_text


def test_workspace_lock_text_variants(tmp_path) -> None:
    app_only = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="user",
        hostname="",
        appname="ImageTool",
        pid=None,
    )
    pid_only = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="",
        hostname="",
        appname="",
        pid=123,
    )
    full_info = manager_workspace._WorkspaceDocumentLockInfo(
        path="marker",
        owner="user",
        hostname="workstation",
        appname="ImageTool",
        pid=123,
    )

    assert manager_widgets._workspace_lock_owner_text(app_only) == (
        "user using ImageTool"
    )
    assert manager_widgets._workspace_lock_owner_text(pid_only) == ("using process 123")
    assert manager_widgets._workspace_lock_owner_text(full_info) == (
        "user on workstation using ImageTool (process 123)"
    )

    def _raise_owner_details_failed() -> None:
        raise RuntimeError("owner details failed")

    def _details_from_active_exception() -> str:
        try:
            _raise_owner_details_failed()
        except RuntimeError:
            return manager_widgets._workspace_lock_details_text(
                tmp_path / "workspace.itws", full_info
            )

    details = _details_from_active_exception()

    assert "owner details failed" in details
    assert "Temporary workspace ownership marker: marker" in details


def test_workspace_window_title_placeholder_non_macos(monkeypatch) -> None:
    monkeypatch.setattr(manager_mainwindow.sys, "platform", "linux")

    assert manager_widgets._strip_workspace_modified_placeholder("Name[*]") == "Name"
    assert manager_widgets._window_title_with_modified_placeholder("Name[*]") == (
        "Name[*]"
    )


def test_manager_workspace_window_title_sets_file_path(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "normal.itws"
        manager._workspace_state.path = workspace
        manager._workspace_state.structure_modified = True
        file_path_calls: list[str] = []

        with monkeypatch.context() as patch:
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                lambda _manager, path: file_path_calls.append(path),
            )
            manager._update_workspace_window_title()

        assert file_path_calls == [str(workspace)]
        assert workspace.name in manager.windowTitle()
        assert manager.isWindowModified()


@pytest.mark.parametrize("dirty_kw", [{"data": True}, {"state": True}])
def test_manager_repeated_tool_dirty_event_defers_document_metadata_until_idle(
    monkeypatch,
    tmp_path,
    dirty_kw: dict[str, bool],
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "normal.itws"
        manager._workspace_state.path = workspace
        file_path_calls: list[str] = []
        node_modified_calls: list[tuple[str, bool]] = []

        monkeypatch.setattr(
            ImageToolManager,
            "setWindowFilePath",
            lambda _manager, path: file_path_calls.append(path),
        )
        monkeypatch.setattr(
            manager,
            "_set_node_window_modified",
            lambda uid, modified: node_modified_calls.append((uid, modified)),
        )

        manager._note_interaction_activity()
        assert manager._mark_workspace_dirty(uid="n1", **dirty_kw)
        assert not manager._mark_workspace_dirty(uid="n1", **dirty_kw)

        assert manager.is_workspace_modified
        assert file_path_calls == []
        assert node_modified_calls == []
        assert [event.uid for event in manager._workspace_state.dirty_events] == ["n1"]
        assert manager._workspace_state.dirty_generation == 1

        manager._flush_idle_work(force=True)

        assert file_path_calls == [str(workspace)]
        assert node_modified_calls == []


def test_manager_tool_dirty_event_escalates_state_to_data(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        assert manager._mark_workspace_dirty(uid="n1", state=True)
        assert manager._mark_workspace_dirty(uid="n1", data=True)

        assert manager._workspace_state.dirty_state == {"n1"}
        assert manager._workspace_state.dirty_data == {"n1"}
        assert [event.uid for event in manager._workspace_state.dirty_events] == [
            "n1",
            "n1",
        ]


def test_manager_workspace_window_title_clears_file_path_without_workspace(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = None
        manager._workspace_state.structure_modified = True
        file_path_calls: list[str] = []

        with monkeypatch.context() as patch:
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                lambda _manager, path: file_path_calls.append(path),
            )
            manager._update_workspace_window_title()

        assert file_path_calls == []
        assert "Untitled" in manager.windowTitle()
        assert not manager.isWindowModified()


def test_manager_workspace_window_title_sets_file_path_during_close(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "close-save.itws"
        manager._workspace_state.path = workspace
        manager._workspace_state.structure_modified = True
        previous_closing = manager._workspace_state.closing_document
        manager._workspace_state.closing_document = True
        file_path_calls: list[str] = []

        try:
            with monkeypatch.context() as patch:
                patch.setattr(
                    ImageToolManager,
                    "setWindowFilePath",
                    lambda _manager, path: file_path_calls.append(path),
                )
                manager._update_workspace_window_title()
        finally:
            manager._workspace_state.closing_document = previous_closing

        assert file_path_calls == [str(workspace)]
        assert workspace.name in manager.windowTitle()
        assert manager.isWindowModified()


def test_manager_loaded_workspace_association_updates_file_path(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "loaded.itws"
        workspace.touch()
        file_path_calls: list[str] = []

        with monkeypatch.context() as patch:
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                lambda _manager, path: file_path_calls.append(path),
            )
            with manager._workspace_document_access_context(workspace) as access:
                manager._associate_loaded_workspace_file(
                    access.path,
                    manager_workspace._current_workspace_schema_version(),
                    workspace_access=access,
                    rebind_data=False,
                )
            assert file_path_calls == [str(workspace.resolve())]

        assert manager.workspace_path == str(workspace.resolve())
        assert workspace.name in manager.windowTitle()
        assert not manager.isWindowModified()


def test_manager_loaded_workspace_association_rebinds_data_after_path_update(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "loaded-rebind.itws"
        workspace.touch()
        rebind_paths: list[pathlib.Path] = []

        monkeypatch.setattr(
            manager,
            "_rebind_workspace_backed_imagetools",
            lambda path: rebind_paths.append(pathlib.Path(path)),
        )
        with manager._workspace_document_access_context(workspace) as access:
            manager._associate_loaded_workspace_file(
                access.path,
                manager_workspace._current_workspace_schema_version(),
                workspace_access=access,
                rebind_data=True,
            )

        assert rebind_paths == [workspace.resolve()]


def test_manager_workspace_window_title_sets_file_path_for_close(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "linux-close.itws"
        manager._workspace_state.path = workspace
        previous_closing = manager._workspace_state.closing_document
        manager._workspace_state.closing_document = True
        file_path_calls: list[str] = []

        try:
            with monkeypatch.context() as patch:
                patch.setattr(
                    ImageToolManager,
                    "setWindowFilePath",
                    lambda _manager, path: file_path_calls.append(path),
                )
                manager._update_workspace_window_title()
        finally:
            manager._workspace_state.closing_document = previous_closing

        assert file_path_calls == [str(workspace)]


def test_manager_close_cancel_restores_workspace_document_closing_state(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        workspace = tmp_path / "cancel-close.itws"
        manager._workspace_state.path = workspace
        manager._workspace_state.structure_modified = True
        manager._workspace_state.closing_document = False
        event = QtGui.QCloseEvent()
        file_path_calls: list[str] = []

        with monkeypatch.context() as patch:
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                lambda _manager, path: file_path_calls.append(path),
            )
            patch.setattr(
                manager._workspace_controller,
                "_dirty_workspace_save_choice",
                lambda _message: "cancel",
            )
            manager._update_workspace_window_title()
            manager.closeEvent(event)

        assert file_path_calls == [str(workspace)]
        assert not event.isAccepted()
        assert not manager._workspace_state.closing_document


def test_manager_close_save_path_updates_file_path(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = tmp_path / "close-save.itws"
        manager._workspace_state.structure_modified = True
        save_closing_states: list[bool] = []
        save_callbacks: list[Callable[[bool], None]] = []
        file_path_calls: list[str] = []
        close_calls: list[str] = []

        def _save(
            *, native: bool = True, on_finished: Callable[[bool], None] | None = None
        ) -> bool:
            save_closing_states.append(manager._workspace_state.closing_document)
            if on_finished is not None:
                save_callbacks.append(on_finished)
            return True

        with monkeypatch.context() as patch:
            patch.setattr(
                ImageToolManager,
                "setWindowFilePath",
                lambda _manager, path: file_path_calls.append(path),
            )
            patch.setattr(
                QtWidgets.QMessageBox,
                "exec",
                lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
            )
            patch.setattr(manager._workspace_controller, "save", _save)
            patch.setattr(manager, "close", lambda: close_calls.append("close") or True)
            event = QtGui.QCloseEvent()
            manager.closeEvent(event)
            assert not event.isAccepted()
            manager._mark_workspace_clean()
            save_callbacks[0](True)

        assert save_closing_states == [True]
        assert file_path_calls == [str(manager._workspace_state.path)]
        assert close_calls == ["close"]
        assert not manager._workspace_state.closing_document


def test_workspace_controller_helper_branch_edges(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        controller = manager._workspace_controller

        invalid = xr.Dataset(attrs={"itool_state": b"not text"})
        assert (
            controller._dataset_without_missing_workspace_colormap(invalid, None)
            is invalid
        )
        invalid.attrs["itool_state"] = "{not-json"
        assert (
            controller._dataset_without_missing_workspace_colormap(invalid, None)
            is invalid
        )
        invalid.attrs["itool_state"] = json.dumps([])
        assert (
            controller._dataset_without_missing_workspace_colormap(invalid, None)
            is invalid
        )

        assert (
            controller._workspace_saved_uid_from_dataset(
                xr.Dataset(attrs={"manager_node_uid": b"node-1"})
            )
            == "node-1"
        )
        assert (
            controller._workspace_saved_uid_from_dataset(
                xr.Dataset(attrs={"manager_node_uid": b"\xff"})
            )
            is None
        )

        try:
            manager._tool_graph.nodes["parent"] = types.SimpleNamespace(parent_uid=None)
            manager._tool_graph.nodes["child"] = types.SimpleNamespace(
                parent_uid="parent"
            )
            manager._tool_graph.nodes["sibling"] = types.SimpleNamespace(
                parent_uid=None
            )
            manager._workspace_state.dirty_data.update({"parent", "child", "sibling"})
            with monkeypatch.context() as patch:
                patch.setattr(controller, "_workspace_node_path", lambda uid: uid)
                assert controller._workspace_highest_dirty_data_roots() == [
                    "parent",
                    "sibling",
                ]
        finally:
            for uid in ("parent", "child", "sibling"):
                manager._tool_graph.nodes.pop(uid, None)
            manager._workspace_state.dirty_data.difference_update(
                {"parent", "child", "sibling"}
            )

        calls: list[str] = []
        with monkeypatch.context() as patch:
            patch.setattr(
                controller, "_dirty_workspace_save_choice", lambda _message: "cancel"
            )
            assert not controller._run_after_dirty_workspace_saved_or_discarded(
                "action", lambda: calls.append("cancel") or True
            )
            patch.setattr(
                controller, "_dirty_workspace_save_choice", lambda _message: "clean"
            )
            assert controller._run_after_dirty_workspace_saved_or_discarded(
                "action", lambda: calls.append("clean") or True
            )
            patch.setattr(
                controller, "_dirty_workspace_save_choice", lambda _message: "discard"
            )
            assert not controller._run_after_dirty_workspace_saved_or_discarded(
                "action", lambda: calls.append("discard") or False
            )

        callbacks: list[Callable[[bool], None]] = []
        with monkeypatch.context() as patch:
            patch.setattr(
                controller, "_dirty_workspace_save_choice", lambda _message: "save"
            )
            patch.setattr(
                controller,
                "save",
                lambda *, native=True, on_finished=None: (
                    callbacks.append(typing.cast("Callable[[bool], None]", on_finished))
                    or True
                ),
            )
            clean_property = property(lambda _m: False)
            patch.setattr(type(manager), "is_workspace_modified", clean_property)
            assert controller._run_after_dirty_workspace_saved_or_discarded(
                "action", lambda: calls.append("saved") or True
            )
            callbacks[0](True)

        assert calls == ["clean", "discard", "saved"]


def test_manager_action_and_modelview_helper_branch_edges(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _DragEvent:
        def __init__(self, mime: QtCore.QMimeData | None) -> None:
            self._mime = mime
            self.accepted = False
            self.ignored = False

        def mimeData(self) -> QtCore.QMimeData | None:
            return self._mime

        def acceptProposedAction(self) -> None:
            self.accepted = True

        def ignore(self) -> None:
            self.ignored = True

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._actions_controller
        model = manager.tree_view._model

        widget = QtWidgets.QWidget()
        controller.add_widget(widget)
        assert widget.isVisible()
        widget.close()

        url_mime = QtCore.QMimeData()
        url_mime.setUrls([QtCore.QUrl.fromLocalFile(str(tmp_path / "data.itws"))])
        accept_event = _DragEvent(url_mime)
        controller.dragEnterEvent(typing.cast("QtGui.QDragEnterEvent", accept_event))
        assert accept_event.accepted

        reject_event = _DragEvent(QtCore.QMimeData())
        controller.dragEnterEvent(typing.cast("QtGui.QDragEnterEvent", reject_event))
        assert reject_event.ignored

        data = xr.DataArray(np.arange(4, dtype=float).reshape(2, 2), dims=("x", "y"))
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False)
        wrapper = manager._tool_graph.root_wrappers[0]

        removed: list[object] = []
        shown: list[str] = []
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_find_watched_idx", lambda _uid: None)
            patch.setattr(
                manager, "remove_imagetool", lambda index: removed.append(index)
            )
            patch.setattr(wrapper, "show", lambda: shown.append(wrapper.uid))
            controller._remove_watched(wrapper.uid)
            controller._show_watched(wrapper.uid)
        assert removed == [wrapper.index]
        assert shown == [wrapper.uid]

        child = _AddedTimeChildTool(data)
        child_uid = manager.add_childtool(child, 0, show=False)
        removed.clear()
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_find_watched_idx", lambda _uid: None)
            patch.setattr(manager, "_remove_childtool", lambda uid: removed.append(uid))
            controller._remove_watched(child_uid)
        assert removed == [child_uid]

        def mime_payload(payload: object) -> QtCore.QMimeData:
            mime = QtCore.QMimeData()
            raw = (
                payload
                if isinstance(payload, bytes)
                else json.dumps(payload).encode("utf-8")
            )
            mime.setData(manager_modelview._MIME, QtCore.QByteArray(raw))
            return mime

        assert model._decode_mime(mime_payload(b"\xff")) is None
        assert model._decode_mime(mime_payload({"parent_id": None})) is None
        assert model._decode_mime(mime_payload({"parent_id": 1, "rows": [0]})) is None
        assert (
            model._decode_mime(mime_payload({"parent_id": None, "rows": "0"})) is None
        )
        assert (
            model._decode_mime(mime_payload({"parent_id": None, "rows": [True]}))
            is None
        )
        assert (
            model._decode_mime(mime_payload({"parent_id": None, "rows": [1, 1]}))
            is None
        )
        assert model._decode_mime(
            mime_payload({"parent_id": None, "rows": [2, 0]})
        ) == {"parent_id": None, "rows": [0, 2]}
        assert model._contiguous_runs([]) == []
        assert model._contiguous_runs([2, 3, 4, 7, 8]) == [(2, 3), (7, 2)]

        assert not model.canDropMimeData(
            None,
            QtCore.Qt.DropAction.MoveAction,
            0,
            0,
            QtCore.QModelIndex(),
        )
        assert not model.canDropMimeData(
            QtCore.QMimeData(),
            QtCore.Qt.DropAction.MoveAction,
            0,
            0,
            QtCore.QModelIndex(),
        )
        assert not model.canDropMimeData(
            mime_payload({"parent_id": None, "rows": [0]}),
            QtCore.Qt.DropAction.CopyAction,
            0,
            0,
            QtCore.QModelIndex(),
        )
        assert not model.dropMimeData(
            mime_payload({"parent_id": None, "rows": [99]}),
            QtCore.Qt.DropAction.MoveAction,
            0,
            0,
            QtCore.QModelIndex(),
        )


def test_manager_close_ignores_active_save_and_async_compaction(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        saving_event = QtGui.QCloseEvent()
        manager.closeEvent(saving_event)

        assert not saving_event.isAccepted()

        manager._workspace_state.save_in_progress = False
        manager._workspace_state.path = tmp_path / "close-compact.itws"
        close_calls: list[str] = []
        compaction_callbacks: list[Callable[[], None]] = []

        def _compact_workspace_before_shutdown(
            *, on_finished: Callable[[], None]
        ) -> bool:
            compaction_callbacks.append(on_finished)
            return True

        with monkeypatch.context() as patch:
            patch.setattr(
                manager._workspace_controller,
                "_dirty_workspace_save_choice",
                lambda _message: "discard",
            )
            patch.setattr(
                manager._workspace_controller,
                "_compact_workspace_before_shutdown",
                _compact_workspace_before_shutdown,
            )
            patch.setattr(manager, "close", lambda: close_calls.append("close") or True)
            compacting_event = QtGui.QCloseEvent()
            manager.closeEvent(compacting_event)
            assert not compacting_event.isAccepted()
            assert len(compaction_callbacks) == 1
            compaction_callbacks[0]()

        assert close_calls == ["close"]


def test_manager_close_compacts_clean_delta_workspace(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = tmp_path / "close-no-compact.itws"
        manager._workspace_state.delta_save_count = 1
        compact_calls: list[str] = []

        monkeypatch.setattr(
            manager._workspace_controller,
            "_compact_workspace_before_shutdown",
            lambda **_kwargs: compact_calls.append("compact") or False,
        )

        assert manager.close()

    assert compact_calls == ["compact"]


def test_workspace_lock_error_message_without_owner(monkeypatch, tmp_path) -> None:
    fname = tmp_path / "busy-message.itws"
    calls: list[dict[str, object]] = []
    lock_info = manager_workspace._WorkspaceDocumentLockInfo(
        path=str(tmp_path / ".busy-message.itws.lock"),
        owner="",
        hostname="",
        appname="",
        pid=None,
    )

    def _critical(*args, **kwargs) -> int:
        calls.append({"args": args, "kwargs": kwargs})
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        manager_workspace, "_workspace_document_lock_info", lambda _fname: lock_info
    )
    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)

    manager_widgets._show_workspace_file_lock_error(None, fname)

    args = calls[0]["args"]
    assert isinstance(args, tuple)
    informative_text = args[3]
    assert isinstance(informative_text, str)
    assert informative_text == (
        "Close the other ImageTool Manager that has busy-message.itws open, "
        "then try again."
    )


def test_application_quit_filter_routes_quit_events(qtbot, monkeypatch) -> None:
    manager = QtWidgets.QWidget()
    qtbot.addWidget(manager)
    calls: list[str] = []
    manager.close = lambda: calls.append("close") or False
    event_filter = manager_widgets._ApplicationQuitFilter(manager)

    assert not event_filter.eventFilter(None, None)
    assert event_filter.eventFilter(None, QtCore.QEvent(QtCore.QEvent.Type.Quit))

    class _QuitKeyEvent(QtGui.QKeyEvent):
        def matches(self, key: QtGui.QKeySequence.StandardKey) -> bool:
            return key == QtGui.QKeySequence.StandardKey.Quit

    shortcut_event = _QuitKeyEvent(
        QtCore.QEvent.Type.ShortcutOverride,
        QtCore.Qt.Key.Key_Q,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert not event_filter.eventFilter(None, shortcut_event)

    key_event = _QuitKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Q,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )

    assert event_filter.eventFilter(None, key_event)
    assert key_event.isAccepted()
    assert calls == ["close", "close"]

    invalid_obj = QtCore.QObject()
    invalid_event = QtCore.QEvent(QtCore.QEvent.Type.Quit)
    original_qt_is_valid = erlab.interactive.utils.qt_is_valid

    def fake_qt_is_valid(*objects: object) -> bool:
        if any(obj is invalid_obj or obj is invalid_event for obj in objects):
            return False
        return original_qt_is_valid(*objects)

    monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", fake_qt_is_valid)
    assert not event_filter.eventFilter(
        invalid_obj,
        QtCore.QEvent(QtCore.QEvent.Type.Quit),
    )
    assert not event_filter.eventFilter(None, invalid_event)
    assert calls == ["close", "close"]

    class DeletedEvent:
        def type(self) -> QtCore.QEvent.Type:
            raise RuntimeError("wrapped C/C++ object has been deleted")

    monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda *objects: True)
    assert not event_filter.eventFilter(None, DeletedEvent())
    assert calls == ["close", "close"]


def test_workspace_document_access_releases_lock(tmp_path) -> None:
    class _FakeLock:
        def __init__(self) -> None:
            self.unlock_count = 0

        def unlock(self) -> None:
            self.unlock_count += 1

    lock = _FakeLock()
    access = manager_widgets._WorkspaceDocumentAccess(tmp_path / "workspace.itws", lock)

    assert access.take_lock() is lock
    access.release()
    assert lock.unlock_count == 0

    access = manager_widgets._WorkspaceDocumentAccess(tmp_path / "workspace.itws", lock)
    access.release()
    access.release()
    assert lock.unlock_count == 1


def test_choose_from_datatree_dialog_root_keys_skip_missing(qtbot) -> None:
    manager = QtWidgets.QWidget()
    manager.next_idx = 7
    qtbot.addWidget(manager)
    tree = xr.DataTree.from_dict(
        {
            "0/imagetool": xr.Dataset(
                attrs={"itool_title": "Loaded"},
            ),
            "figures/figure/tool": xr.Dataset(attrs={"tool_title": "Figure 1"}),
        }
    )
    try:
        dialog = _ChooseFromDataTreeDialog(
            manager,
            tree,
            mode="load",
            root_keys=("missing", "0"),
        )
        qtbot.addWidget(dialog)

        assert dialog._tree_widget.topLevelItemCount() == 2
        assert dialog._tree_widget.topLevelItem(0).text(0) == "7: Loaded"
        figure_item = dialog._tree_widget.topLevelItem(1)
        assert figure_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "figures/figure"
        dialog._uncheck_children()
        assert figure_item.checkState(0) == QtCore.Qt.CheckState.Unchecked
    finally:
        tree.close()


def test_choose_from_workspace_manifest_dialog_selected_paths(qtbot) -> None:
    manager = QtWidgets.QWidget()
    manager.next_idx = 7
    qtbot.addWidget(manager)
    manifest = {
        "root_order": ["0", "1"],
        "nodes": [
            {
                "path": "0",
                "kind": "imagetool",
                "display_name": "Root A",
            },
            {
                "path": "0/childtools/profile",
                "kind": "tool",
                "display_name": "Profile",
            },
            {
                "path": "1",
                "kind": "imagetool",
                "display_name": "Root B",
            },
            {
                "path": "figures/n17",
                "kind": "tool",
                "display_name": "Figure 1",
            },
        ],
    }
    dialog = _ChooseFromWorkspaceManifestDialog(manager, manifest)
    qtbot.addWidget(dialog)

    assert dialog._tree_widget.topLevelItemCount() == 3
    assert dialog._tree_widget.topLevelItem(0).text(0) == "7: Root A"
    assert dialog.item_for_path("0/childtools/profile") is not None
    root_item = dialog.item_for_path("0")
    assert root_item is not None
    child_item = dialog.item_for_path("0/childtools/profile")
    assert child_item is not None
    dialog._uncheck_children()

    assert dialog.selected_paths() == {"0", "1"}


def test_manager_workspace_save_as_locked_target_does_not_write(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-save-as.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    operation_errors: list[tuple[typing.Any, ...]] = []
    try:
        with manager_context() as manager:
            monkeypatch.setattr(
                manager, "_workspace_save_dialog", lambda *args, **kwargs: str(fname)
            )
            monkeypatch.setattr(
                manager,
                "_save_workspace_document",
                lambda *args, **kwargs: pytest.fail(
                    "Save As should lock the target before writing"
                ),
            )
            monkeypatch.setattr(
                manager,
                "_show_operation_error",
                lambda *args, **kwargs: operation_errors.append(args),
            )

            assert not manager._workspace_controller.save_as(native=False)
    finally:
        lock.unlock()

    assert operation_errors


def test_manager_workspace_save_as_reports_snapshot_error(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "write-error.itws"
    operation_errors: list[tuple[typing.Any, ...]] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager, "_workspace_save_dialog", lambda *args, **kwargs: str(fname)
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_full_save_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args, **kwargs: operation_errors.append(args),
        )

        assert not manager._workspace_controller.save_as(native=False)

    assert operation_errors == [
        (
            "Error while saving workspace",
            "An error occurred while saving the workspace file.",
        )
    ]


def test_manager_workspace_save_as_rejects_h5_target(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "workspace.h5"
    warnings: list[tuple[typing.Any, ...]] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager, "_workspace_save_dialog", lambda *args, **kwargs: str(fname)
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args)
        )

        assert not manager._workspace_controller.save_as(native=False)

    assert len(warnings) == 1
    assert warnings[0][1:] == (
        "Workspace Not Saved",
        "ImageTool Manager saves workspaces as .itws files.",
    )
    assert not fname.exists()


def test_manager_workspace_load_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-load.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            manager_workspace,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Load should lock the workspace before recovery")
            ),
        )
        with manager_context() as manager, pytest.raises(BlockingIOError):
            manager._load_workspace_file(
                fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )
    finally:
        lock.unlock()

    assert recovery_calls == []


def test_manager_workspace_path_lock_contract(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _FakeLock:
        def __init__(self) -> None:
            self.unlock_count = 0

        def unlock(self) -> None:
            self.unlock_count += 1

    with manager_context() as manager:
        current = (tmp_path / "current.itws").resolve()
        manager._workspace_state.path = current
        lock = _FakeLock()

        manager._set_workspace_path(current, workspace_lock=lock)

        assert lock.unlock_count == 1
        with pytest.raises(RuntimeError, match="pre-acquired document lock"):
            manager._set_workspace_path(tmp_path / "other.itws")


def test_manager_open_recent_menu_state_labels_and_clear(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace_a = tmp_path / "alpha" / "workspace.itws"
    workspace_b = tmp_path / "beta" / "workspace.itws"
    workspace_a.parent.mkdir()
    workspace_b.parent.mkdir()
    workspace_a.touch()
    workspace_b.touch()

    with manager_context() as manager:
        assert not manager.open_recent_menu.isEnabled()

        manager._workspace_controller._record_recent_workspace(workspace_a)
        manager._workspace_controller._record_recent_workspace(workspace_b)
        manager._workspace_controller._populate_open_recent_menu()

        assert manager.open_recent_menu.isEnabled()
        actions = action_map_by_object_name(manager.open_recent_menu)
        first_action = actions["manager_recent_workspace_action_0"]
        second_action = actions["manager_recent_workspace_action_1"]
        assert first_action.data() == str(workspace_b.resolve())
        assert second_action.data() == str(workspace_a.resolve())
        assert first_action.toolTip() == str(workspace_b.resolve())
        assert first_action.statusTip() == str(workspace_b.resolve())
        assert "manager_clear_recent_workspaces_action" in actions

        actions["manager_clear_recent_workspaces_action"].trigger()
        assert manager._workspace_controller._recent_workspace_paths() == []
        assert not manager.open_recent_menu.isEnabled()


def test_manager_open_recent_menu_stays_disabled_while_save_in_progress(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "recent.itws"
    workspace.touch()

    with manager_context() as manager:
        manager._workspace_controller._record_recent_workspace(workspace)
        assert manager.open_recent_menu.isEnabled()

        manager._workspace_state.save_in_progress = True
        manager._workspace_controller._refresh_open_recent_menu_action()
        assert not manager.open_recent_menu.isEnabled()
        manager._workspace_controller._populate_open_recent_menu()
        assert not manager.open_recent_menu.isEnabled()
        assert "manager_recent_workspace_action_0" in action_map_by_object_name(
            manager.open_recent_menu
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: pytest.fail(
                "Open Recent should not show dialogs during workspace save"
            ),
        )
        assert not manager.open_recent_workspace(tmp_path / "missing.itws")

        manager._workspace_state.save_in_progress = False
        manager._workspace_controller._refresh_open_recent_menu_action()
        assert manager.open_recent_menu.isEnabled()


def test_manager_recent_workspaces_dedupe_move_to_top_and_cap(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    paths = [tmp_path / f"workspace-{idx}.itws" for idx in range(12)]
    for path in paths:
        path.touch()

    with manager_context() as manager:
        for path in paths:
            manager._workspace_controller._record_recent_workspace(path)

        assert manager._workspace_controller._recent_workspace_paths() == [
            path.resolve() for path in reversed(paths[2:])
        ]

        manager._workspace_controller._record_recent_workspace(paths[6])

        assert manager._workspace_controller._recent_workspace_paths() == [
            paths[6].resolve(),
            paths[11].resolve(),
            paths[10].resolve(),
            paths[9].resolve(),
            paths[8].resolve(),
            paths[7].resolve(),
            paths[5].resolve(),
            paths[4].resolve(),
            paths[3].resolve(),
            paths[2].resolve(),
        ]


def test_manager_recent_workspace_normalization_and_settings(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "workspace.itws"
    data_file = tmp_path / "data.h5"
    workspace.touch()
    data_file.touch()

    assert ImageToolManager._normalize_recent_workspace_paths(
        [data_file, workspace, workspace]
    ) == [workspace.resolve()]

    with manager_context() as manager:
        settings = manager_widgets._manager_settings()
        settings.setValue(
            manager_widgets._RECENT_WORKSPACES_SETTINGS_KEY, str(workspace)
        )
        settings.sync()
        assert manager._workspace_controller._recent_workspace_paths() == [
            workspace.resolve()
        ]

        class _ObjectSettings:
            def sync(self) -> None:
                pass

            def value(self, _key, _default):
                return object()

        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._workspace_io,
            "_manager_settings",
            lambda: _ObjectSettings(),
        )
        assert manager._workspace_controller._recent_workspace_paths() == []


def test_manager_open_recent_workspace_flow(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    older = tmp_path / "older.itws"
    newer = tmp_path / "newer.itws"
    missing = tmp_path / "missing.itws"
    older.touch()
    newer.touch()

    with manager_context() as manager:
        manager._workspace_controller._set_recent_workspace_paths(
            [newer.resolve(), older.resolve()]
        )
        load_calls: list[tuple[pathlib.Path, dict[str, typing.Any]]] = []

        def _record_load(path, **kwargs):
            load_calls.append((pathlib.Path(path), kwargs))
            return True

        monkeypatch.setattr(manager, "_load_workspace_file", _record_load)
        manager._workspace_state.structure_modified = True
        monkeypatch.setattr(
            manager._workspace_controller,
            "_dirty_workspace_save_choice",
            lambda _message: "cancel",
        )

        assert not manager.open_recent_workspace(older)
        assert load_calls == []
        assert manager._workspace_controller._recent_workspace_paths() == [
            newer.resolve(),
            older.resolve(),
        ]

        manager._workspace_state.structure_modified = False
        monkeypatch.setattr(
            manager._workspace_controller,
            "_dirty_workspace_save_choice",
            lambda _message: "clean",
        )
        assert manager.open_recent_workspace(older)
        assert load_calls == [
            (
                older.resolve(),
                {
                    "replace": True,
                    "associate": True,
                    "mark_dirty": False,
                    "select": False,
                    "native": True,
                },
            )
        ]
        assert manager._workspace_controller._recent_workspace_paths() == [
            older.resolve(),
            newer.resolve(),
        ]

        missing_warnings: list[tuple[typing.Any, ...]] = []
        manager._workspace_controller._set_recent_workspace_paths(
            [missing.resolve(), older.resolve()]
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args: missing_warnings.append(args),
        )

        assert not manager.open_recent_workspace(missing)
        assert len(missing_warnings) == 1
        assert manager._workspace_controller._recent_workspace_paths() == [
            older.resolve()
        ]

        h5_workspace = tmp_path / "old-workspace.h5"
        h5_workspace.touch()
        confirm_calls: list[str] = []
        unsupported_warnings: list[tuple[typing.Any, ...]] = []
        manager._workspace_controller._set_recent_workspace_paths(
            [h5_workspace.resolve(), older.resolve()]
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_dirty_workspace_save_choice",
            lambda message: confirm_calls.append(message) or "clean",
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args: unsupported_warnings.append(args),
        )

        assert not manager.open_recent_workspace(h5_workspace)
        assert len(unsupported_warnings) == 1
        assert unsupported_warnings[0][1:] == (
            "Unsupported Workspace File",
            "ImageTool Manager opens workspace files with the .itws extension.",
        )
        assert confirm_calls == []
        assert manager._workspace_controller._recent_workspace_paths() == [
            older.resolve()
        ]


def test_manager_open_recent_workspace_reports_load_errors(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "broken.itws"
    workspace.touch()

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._workspace_controller,
            "_dirty_workspace_save_choice",
            lambda _message: "clean",
        )
        monkeypatch.setattr(
            manager,
            "_load_workspace_file",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        lock_errors: list[pathlib.Path] = []
        monkeypatch.setattr(
            manager_workspace,
            "_is_workspace_file_lock_error",
            lambda _exc: True,
        )
        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._workspace_io,
            "_show_workspace_file_lock_error",
            lambda _parent, path: lock_errors.append(pathlib.Path(path)),
        )
        assert not manager.open_recent_workspace(workspace)
        assert lock_errors == [workspace.resolve()]

        critical_messages: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager_workspace,
            "_is_workspace_file_lock_error",
            lambda _exc: False,
        )
        monkeypatch.setattr(
            erlab.interactive.utils.MessageDialog,
            "critical",
            lambda _parent, title, message: critical_messages.append((title, message)),
        )
        assert not manager.open_recent_workspace(workspace)
        assert critical_messages == [
            ("Error", "An error occurred while loading the workspace file.")
        ]


def test_manager_records_recent_workspace_accesses(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    opened = tmp_path / "opened.itws"
    saved = tmp_path / "saved.itws"
    imported = tmp_path / "imported.itws"
    for path in (opened, saved, imported):
        path.touch()

    with manager_context() as manager:
        with manager._workspace_document_access_context(opened) as access:
            manager._associate_loaded_workspace_file(
                opened,
                manager_workspace._current_workspace_schema_version(),
                workspace_access=access,
            )
        data = xr.DataArray(np.arange(4).reshape((2, 2)), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        monkeypatch.setattr(
            manager, "_workspace_save_dialog", lambda **_kwargs: str(saved)
        )
        monkeypatch.setattr(
            manager,
            "_rebind_workspace_backed_imagetools",
            lambda *_args, **_kwargs: None,
        )
        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)

        monkeypatch.setattr(QtWidgets.QFileDialog, "exec", lambda _dialog: True)
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "selectedFiles",
            lambda _dialog: [str(imported)],
        )
        monkeypatch.setattr(
            manager, "_load_workspace_file", lambda *_args, **_kwargs: True
        )
        assert manager.import_workspace(native=False)

        assert manager._workspace_controller._recent_workspace_paths()[:3] == [
            imported.resolve(),
            saved.resolve(),
            opened.resolve(),
        ]


def test_manager_registry_refreshes_use_heartbeat_controller(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    requests: list[tuple[str | None, bool]] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager._registry_heartbeat,
            "request_refresh",
            lambda workspace_path, *, coalesce_if_busy: requests.append(
                (workspace_path, coalesce_if_busy)
            ),
        )
        manager._refresh_manager_record()
        manager._registry_heartbeat_tick()

    assert requests == [(None, True), (None, False)]


def test_manager_records_packaged_workspace_with_desktop_shell(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    workspace = tmp_path / "workspace.itws"
    data_file = tmp_path / "data.h5"
    workspace.touch()
    data_file.touch()
    recorded: list[pathlib.Path] = []

    monkeypatch.setattr(
        manager_desktop,
        "record_recent_workspace",
        lambda path: recorded.append(pathlib.Path(path)),
    )

    with manager_context() as manager:
        monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", True)
        manager._workspace_controller._record_recent_workspace(workspace)
        manager._workspace_controller._record_recent_workspace(data_file)

    assert recorded == [workspace.resolve()]


def test_manager_startup_args_parse_flags_and_file_paths(tmp_path) -> None:
    workspace = tmp_path / "workspace.itws"
    data_file = tmp_path / "data.h5"
    workspace.touch()
    data_file.touch()

    startup_args = manager_module._parse_startup_args(
        [
            manager_desktop.OPEN_WORKSPACE_DIALOG_ARG,
            str(workspace),
            manager_desktop.NEW_MANAGER_WINDOW_ARG,
            "--ignored",
            str(tmp_path / "missing.itws"),
            str(data_file),
        ]
    )

    assert startup_args.open_workspace_dialog
    assert startup_args.force_new_manager
    assert startup_args.files == [workspace, data_file]


def test_manager_startup_pending_files_combines_file_open_events(tmp_path) -> None:
    event_file = tmp_path / "event.h5"
    argv_file = tmp_path / "argv.h5"

    assert manager_module._startup_pending_files([event_file], [argv_file]) == [
        event_file,
        argv_file,
    ]
    assert manager_module._startup_pending_files([event_file], [event_file]) == [
        event_file
    ]


def test_manager_startup_pending_files_keeps_unresolved_paths(
    monkeypatch, tmp_path
) -> None:
    event_file = tmp_path / "event.h5"
    argv_file = tmp_path / "argv.h5"

    def fail_resolve(self: pathlib.Path) -> pathlib.Path:
        raise OSError("unavailable")

    monkeypatch.setattr(pathlib.Path, "resolve", fail_resolve)

    assert manager_module._startup_pending_files([event_file], [argv_file]) == [
        event_file,
        argv_file,
    ]
    assert manager_module._startup_pending_files([event_file], [event_file]) == [
        event_file
    ]


def test_manager_startup_ignores_argv_files_with_existing_qapplication(
    monkeypatch, qapp, tmp_path
) -> None:
    data_file = tmp_path / "data.dat"
    data_file.touch()
    handled_files: list[pathlib.Path] = []

    if isinstance(qapp, manager_module._ManagerApp):
        qapp._pending_files.clear()

    class _FakeManager:
        def show(self) -> None:
            pass

        def activateWindow(self) -> None:
            pass

        def _handle_dropped_files(self, paths: list[pathlib.Path]) -> None:
            handled_files.extend(paths)

        def load(self) -> None:
            pass

    monkeypatch.setattr(
        manager_module.sys,
        "argv",
        ["erlab-imagetool-manager", str(data_file)],
    )
    monkeypatch.setattr(manager_module, "ImageToolManager", _FakeManager)
    monkeypatch.setattr(
        manager_module,
        "_try_forward_startup_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not forward embedded argv files")
        ),
    )

    try:
        manager_module.main(execute=False)
        assert handled_files == []
    finally:
        manager_module._manager_instance = None


def test_manager_startup_forward_files_to_resolved_manager(monkeypatch, tmp_path):
    data_file = tmp_path / "data.dat"
    data_file.touch()
    forwarded: list[tuple[list[pathlib.Path], int | None, int]] = []

    monkeypatch.setattr(
        manager_module,
        "manager_selection_info",
        lambda **_kwargs: {
            "resolved_index": 4,
            "needs_selection": False,
            "managers": [],
        },
    )
    monkeypatch.setattr(
        manager_module,
        "_load_in_manager_startup",
        lambda paths, *, target, timeout_ms: forwarded.append(
            (list(paths), target, timeout_ms)
        ),
    )

    assert manager_module._try_forward_startup_files([data_file])
    assert forwarded == [([data_file], 4, manager_module._STARTUP_FORWARD_TIMEOUT_MS)]


def test_manager_startup_forward_failure_falls_back_to_new_manager(
    monkeypatch, tmp_path, caplog
):
    data_file = tmp_path / "data.dat"
    data_file.touch()

    monkeypatch.setattr(
        manager_module,
        "manager_selection_info",
        lambda **_kwargs: {
            "resolved_index": 4,
            "needs_selection": False,
            "managers": [],
        },
    )
    monkeypatch.setattr(
        manager_module,
        "_load_in_manager_startup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gone")),
    )
    caplog.set_level(logging.INFO, logger=manager_module.logger.name)

    assert not manager_module._try_forward_startup_files([data_file])
    assert "Could not forward startup files" in caplog.text


def test_manager_startup_forward_workspace_files_stay_local(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "workspace.itws"
    workspace.touch()
    monkeypatch.setattr(
        manager_module,
        "manager_selection_info",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not inspect managers")
        ),
    )

    assert not manager_module._try_forward_startup_files([workspace])


def test_manager_startup_forward_invalid_h5_is_data_file(monkeypatch, tmp_path) -> None:
    data_file = tmp_path / "broken.h5"
    data_file.write_text("not hdf5")
    forwarded: list[int | None] = []

    monkeypatch.setattr(
        manager_module,
        "manager_selection_info",
        lambda **_kwargs: {
            "resolved_index": 2,
            "needs_selection": False,
            "managers": [],
        },
    )
    monkeypatch.setattr(
        manager_module,
        "_load_in_manager_startup",
        lambda paths, *, target, timeout_ms: forwarded.append(target),
    )

    assert manager_module._try_forward_startup_files([data_file])
    assert forwarded == [2]


def test_manager_startup_forward_no_live_manager_stays_local(
    monkeypatch, tmp_path
) -> None:
    data_file = tmp_path / "data.dat"
    data_file.touch()
    monkeypatch.setattr(
        manager_module,
        "manager_selection_info",
        lambda **_kwargs: {
            "resolved_index": None,
            "needs_selection": False,
            "managers": [],
        },
    )
    monkeypatch.setattr(
        manager_module,
        "_load_in_manager_startup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not forward")
        ),
    )

    assert not manager_module._try_forward_startup_files([data_file])


def test_manager_startup_forward_empty_and_uninspectable_state(
    monkeypatch, caplog
) -> None:
    assert not manager_module._try_forward_startup_files([])

    monkeypatch.setattr(
        manager_module,
        "manager_selection_info",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("locked")),
    )
    caplog.set_level(logging.DEBUG, logger=manager_module.logger.name)

    assert not manager_module._try_forward_startup_files([pathlib.Path("data.dat")])
    assert "Could not inspect live managers" in caplog.text


def test_manager_startup_h5_workspace_files_are_not_workspace_documents(
    monkeypatch, tmp_path
) -> None:
    import h5py

    h5_workspace = tmp_path / "workspace.h5"
    with h5py.File(h5_workspace, "w") as h5_file:
        h5_file.attrs["imagetool_workspace_schema_version"] = 5
    inspected: list[bool] = []
    monkeypatch.setattr(
        manager_module,
        "manager_selection_info",
        lambda **_kwargs: (
            inspected.append(True)
            or {
                "resolved_index": None,
                "needs_selection": False,
                "managers": [],
            }
        ),
    )

    assert not manager_module._startup_path_is_workspace(h5_workspace)
    assert not manager_module._try_forward_startup_files([h5_workspace])
    assert inspected == [True]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (manager_module._STARTUP_TARGET_NEW, False),
        (manager_module._STARTUP_TARGET_CANCEL, True),
        (3, True),
    ],
)
def test_manager_startup_forward_ambiguous_target_choices(
    monkeypatch, tmp_path, choice, expected
) -> None:
    data_file = tmp_path / "data.dat"
    data_file.touch()
    forwarded: list[int | None] = []
    info = {
        "resolved_index": None,
        "needs_selection": True,
        "managers": [{"index": 3}],
    }

    monkeypatch.setattr(
        manager_module, "manager_selection_info", lambda **_kwargs: info
    )
    monkeypatch.setattr(
        manager_module,
        "_choose_startup_manager_target",
        lambda selection_info, parent=None: choice,
    )
    monkeypatch.setattr(
        manager_module,
        "_load_in_manager_startup",
        lambda paths, *, target, timeout_ms: forwarded.append(target),
    )

    assert manager_module._try_forward_startup_files([data_file]) is expected
    assert forwarded == ([3] if choice == 3 else [])


@pytest.mark.parametrize(
    ("dialog_result", "selected_row", "expected"),
    [
        (
            QtWidgets.QDialog.DialogCode.Rejected,
            1,
            manager_module._STARTUP_TARGET_CANCEL,
        ),
        (
            QtWidgets.QDialog.DialogCode.Accepted,
            -1,
            manager_module._STARTUP_TARGET_CANCEL,
        ),
        (QtWidgets.QDialog.DialogCode.Accepted, 0, manager_module._STARTUP_TARGET_NEW),
        (QtWidgets.QDialog.DialogCode.Accepted, 1, 7),
        (QtWidgets.QDialog.DialogCode.Accepted, 2, 8),
    ],
)
def test_manager_startup_target_dialog_returns_selected_target(
    monkeypatch, qapp, tmp_path, dialog_result, selected_row, expected
) -> None:
    inspected: list[tuple[int, str, str]] = []
    workspace_path = tmp_path / "project.itws"

    def fake_exec(dialog: QtWidgets.QDialog):
        target_list = dialog.findChild(
            QtWidgets.QListWidget, "manager_startup_target_list"
        )
        assert target_list is not None
        assert target_list.count() == 3
        inspected.append(
            (
                target_list.currentRow(),
                target_list.item(1).text(),
                target_list.item(2).toolTip(),
            )
        )
        target_list.setCurrentRow(selected_row)
        return dialog_result

    monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec)

    target = manager_module._choose_startup_manager_target(
        {
            "managers": [
                {
                    "index": 7,
                    "host": "127.0.0.1",
                    "port": 45555,
                    "pid": 123,
                    "workspace_path": str(workspace_path),
                },
                {
                    "index": 8,
                    "host": "127.0.0.1",
                    "port": 45556,
                    "pid": 456,
                    "workspace_path": None,
                },
            ]
        }
    )

    assert target == expected
    assert inspected == [
        (
            0,
            "Manager #7 - project.itws",
            "Manager #8\nEndpoint: 127.0.0.1:45556\nProcess ID: 456",
        )
    ]


def test_manager_startup_open_workspace_dialog_schedules_load(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    calls: list[tuple[int, weakref.ReferenceType[object] | None, str]] = []

    def record_single_shot(interval: int, callback: Callable[[], None]) -> None:
        receiver = getattr(callback, "__self__", None)
        calls.append(
            (
                interval,
                weakref.ref(receiver) if receiver is not None else None,
                getattr(callback, "__name__", ""),
            )
        )

    monkeypatch.setattr(
        manager_module.sys,
        "argv",
        ["erlab-imagetool-manager", manager_desktop.OPEN_WORKSPACE_DIALOG_ARG],
    )
    monkeypatch.setattr(
        manager_module.QtCore.QTimer,
        "singleShot",
        record_single_shot,
    )

    with manager_context() as manager:
        assert any(
            interval == 0
            and receiver_ref is not None
            and receiver_ref() is manager
            and name == "load"
            for interval, receiver_ref, name in calls
        )


def test_manager_active_window_and_focus_restore_guards(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        active = QtWidgets.QWidget()
        origin = QtWidgets.QWidget()
        other = QtWidgets.QWidget()
        qtbot.addWidget(active)
        qtbot.addWidget(origin)
        qtbot.addWidget(other)
        origin.show()
        other.show()

        monkeypatch.setattr(
            QtWidgets.QApplication, "activeWindow", staticmethod(lambda: active)
        )
        monkeypatch.setattr(manager, "_node_uid_from_window", lambda _window: "uid")
        monkeypatch.setattr(
            erlab.interactive.utils, "qt_is_valid", lambda *_objs: False
        )
        assert manager._active_managed_window() is None

        monkeypatch.setattr(
            QtWidgets.QApplication, "activeWindow", staticmethod(lambda: other)
        )
        monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_objs: True)
        manager._restore_focus_after_workspace_save(origin)


def test_manager_compact_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(manager._workspace_controller, "save_as", lambda: True)
        assert manager.compact_workspace()

        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.save_in_progress = True
        assert not manager.compact_workspace()
        manager._workspace_state.save_in_progress = False

        operation_errors: list[tuple[typing.Any, ...]] = []
        focus_restores: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager,
            "_save_workspace_document",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("compact failed")
            ),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda *args: operation_errors.append(args),
        )
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restores.append(origin),
        )

        assert not manager.compact_workspace()
        assert operation_errors == [
            (
                "Error while compacting workspace",
                "An error occurred while compacting the workspace file.",
            )
        ]
        assert focus_restores == [None]


def test_manager_compact_workspace_copies_matching_groups(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "manual-repack.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        tree = manager._to_datatree()
        try:
            copy_source, copy_groups = (
                manager._workspace_controller._workspace_full_save_copy_groups(tree)
            )
        finally:
            tree.close()
        assert copy_source == str(fname)
        assert copy_groups

        original_write = manager_workspace._write_full_workspace_tree_file
        full_write_calls: list[
            tuple[
                str | os.PathLike[str] | None,
                tuple[tuple[str, str, dict[str, typing.Any] | None], ...],
            ]
        ] = []

        def _record_full_write(
            write_fname: str | os.PathLike[str],
            write_tree: xr.DataTree,
            root_attrs: Mapping[str, typing.Any],
            **kwargs: typing.Any,
        ) -> None:
            full_write_calls.append(
                (kwargs.get("copy_source"), tuple(kwargs.get("copy_groups", ())))
            )
            original_write(write_fname, write_tree, root_attrs, **kwargs)

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            _record_full_write,
        )

        assert manager.compact_workspace()

        copy_source, copy_groups = full_write_calls[-1]
        assert copy_source == str(fname)
        assert copy_groups


def test_manager_compact_workspace_reduces_internal_holes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_compression = erlab.interactive.options["io/workspace/compression"]
    erlab.interactive.options["io/workspace/compression"] = "none"
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            rng = np.random.default_rng(1234)
            data = xr.DataArray(
                rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8),
                dims=("x", "y"),
            )
            updated = xr.DataArray(
                rng.integers(0, 256, size=(2048, 2048), dtype=np.uint8),
                dims=("x", "y"),
            )

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            uid = manager._tool_graph.root_wrappers[0].uid

            fname = tmp_path / "hole-repack.itws"
            manager._save_workspace_document(fname, force_full=True)
            manager._adopt_workspace_path(fname)
            manager._mark_workspace_clean()
            size_full = fname.stat().st_size

            manager.get_imagetool(0).slicer_area.replace_source_data(
                updated,
                auto_compute=False,
            )
            manager._mark_node_data_dirty(uid)
            assert _request_workspace_save_and_wait(qtbot, manager)
            size_incremental = fname.stat().st_size

            monkeypatch.setattr(
                erlab.interactive.utils,
                "wait_dialog",
                lambda *args, **kwargs: contextlib.nullcontext(),
            )
            assert manager.compact_workspace()
            size_compact = fname.stat().st_size

            assert size_incremental > size_full + data.nbytes // 2
            assert size_compact < size_incremental - data.nbytes // 2
            _assert_no_workspace_internal_groups(fname)
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_compact_workspace_reapplies_compression_mode(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    old_compression = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                dims=("x", "y"),
            )

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            fname = tmp_path / "compression-repack.itws"
            manager._save_workspace_document(fname, force_full=True)
            manager._adopt_workspace_path(fname)
            manager._mark_workspace_clean()
            with h5py.File(fname, "r") as h5_file:
                assert (
                    _hdf5_blosc2_level_codec(
                        h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
                    )
                    is None
                )

            original_write = manager_workspace._write_full_workspace_tree_file
            full_write_calls: list[
                tuple[str | os.PathLike[str] | None, tuple[object, ...]]
            ] = []

            def _record_full_write(
                write_fname: str | os.PathLike[str],
                write_tree: xr.DataTree,
                root_attrs: Mapping[str, typing.Any],
                **kwargs: typing.Any,
            ) -> None:
                full_write_calls.append(
                    (kwargs.get("copy_source"), tuple(kwargs.get("copy_groups", ())))
                )
                original_write(write_fname, write_tree, root_attrs, **kwargs)

            erlab.interactive.options["io/workspace/compression"] = "zstd1"
            monkeypatch.setattr(
                erlab.interactive.utils,
                "wait_dialog",
                lambda *args, **kwargs: contextlib.nullcontext(),
            )
            monkeypatch.setattr(
                manager_workspace,
                "_write_full_workspace_tree_file",
                _record_full_write,
            )
            assert manager.compact_workspace()

            assert full_write_calls[-1] == (str(fname), ())
            with h5py.File(fname, "r") as h5_file:
                assert _hdf5_blosc2_level_codec(
                    h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
                ) == (1, 5)
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_shutdown_compact_preserves_existing_dataset_filters(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    old_compression = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                dims=("x", "y"),
            )

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            uid = manager._tool_graph.root_wrappers[0].uid

            fname = tmp_path / "shutdown-preserve-filters.itws"
            manager._save_workspace_document(fname, force_full=True)
            manager._adopt_workspace_path(fname)
            manager._mark_workspace_clean()
            manager._mark_node_data_dirty(uid)
            assert _request_workspace_save_and_wait(qtbot, manager)

            with h5py.File(fname, "r") as h5_file:
                assert (
                    _hdf5_blosc2_level_codec(
                        h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
                    )
                    is None
                )

            erlab.interactive.options["io/workspace/compression"] = "zstd1"
            monkeypatch.setattr(
                manager_workspace_io,
                "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
                1,
            )
            monkeypatch.setattr(
                manager_workspace_io,
                "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
                0.0,
            )

            assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)

            with h5py.File(fname, "r") as h5_file:
                assert (
                    _hdf5_blosc2_level_codec(
                        h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
                    )
                    is None
                )
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_shutdown_compaction_logs_failure(
    caplog,
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.delta_save_count = 1
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=100,
            replacement_delta_count=1,
        )

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_should_repack_before_shutdown",
            lambda: True,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_file_repack_snapshot",
            lambda generation: manager_workspace._WorkspaceSaveSnapshot(
                generation=generation,
                root_attrs=_transaction_test_root_attrs(delta_save_count=1),
                delta_save_count=0,
                file_repack=True,
            ),
        )

        with caplog.at_level(logging.ERROR):
            assert manager._workspace_controller._compact_workspace_before_shutdown()
            assert len(pool.workers) == 1
            pool.workers[0].finish(ok=False, error_text="compact failed")
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

    assert "Failed to compact workspace before shutdown" in caplog.text
    assert "compact failed" in caplog.text


def test_manager_workspace_save_dialog_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    calls: list[tuple[str, object]] = []
    original_file_dialog = QtWidgets.QFileDialog

    class _FakeFileDialog:
        AcceptMode = original_file_dialog.AcceptMode
        FileMode = original_file_dialog.FileMode
        Option = original_file_dialog.Option
        exec_result = 0

        def __init__(self, _parent, caption: str) -> None:
            calls.append(("caption", caption))

        def setAcceptMode(self, mode) -> None:
            calls.append(("accept", mode))

        def setFileMode(self, mode) -> None:
            calls.append(("file_mode", mode))

        def setNameFilter(self, name_filter: str) -> None:
            calls.append(("filter", name_filter))

        def setDefaultSuffix(self, suffix: str) -> None:
            calls.append(("suffix", suffix))

        def selectFile(self, fname: str) -> None:
            calls.append(("select", fname))

        def setDirectory(self, directory: str) -> None:
            calls.append(("directory", directory))

        def setOption(self, option) -> None:
            calls.append(("option", option))

        def exec(self) -> int:
            return self.exec_result

        def selectedFiles(self) -> list[str]:
            return [str(tmp_path / "selected.itws")]

    monkeypatch.setattr(QtWidgets, "QFileDialog", _FakeFileDialog)
    with manager_context() as manager:
        assert (
            manager._workspace_save_dialog(
                native=False, selected_file=tmp_path / "explicit.itws"
            )
            is None
        )
        assert ("select", str(tmp_path / "explicit.itws")) in calls

        _FakeFileDialog.exec_result = 1
        manager._workspace_state.path = tmp_path / "bound.itws"
        assert manager._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("select", str(tmp_path / "bound.itws")) in calls

        manager._workspace_state.path = None
        manager._recent_directory = None
        default_dir = tmp_path / "default"
        default_dir.mkdir()
        erlab.interactive.options.model = AppOptions().model_copy(
            update={
                "io": AppOptions().io.model_copy(
                    update={"default_directory": str(default_dir)}
                )
            }
        )
        assert manager._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("directory", str(default_dir)) in calls

        manager._recent_directory = str(tmp_path)
        assert manager._workspace_save_dialog(native=True) == str(
            tmp_path / "selected.itws"
        )
        assert ("directory", str(tmp_path)) in calls


def test_manager_dirty_workspace_save_choice_save_branch(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.path = pathlib.Path("dirty.itws")
        manager._workspace_state.structure_modified = True
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Save,
        )

        assert (
            manager._workspace_controller._dirty_workspace_save_choice(
                "Save before continuing."
            )
            == "save"
        )


def test_manager_legacy_itws_schema_save_helpers(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda _msg_box: QtWidgets.QMessageBox.StandardButton.Ok,
        )
        manager._show_legacy_workspace_upgrade_message(tmp_path / "legacy-schema.itws")

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: None)
        assert (
            manager._save_legacy_workspace_as_v4(tmp_path / "legacy-schema.itws")
            is None
        )

        dirty_reasons: list[str] = []
        monkeypatch.setattr(
            manager,
            "_save_legacy_workspace_as_v4",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            manager, "_mark_workspace_structure_dirty", dirty_reasons.append
        )
        manager._associate_loaded_workspace_file(
            tmp_path / "legacy-schema.itws",
            manager_workspace._WORKSPACE_LEGACY_SCHEMA_VERSION - 1,
        )

        assert manager._workspace_state.path is None
        assert manager._workspace_state.needs_full_save
        assert dirty_reasons == ["Legacy workspace needs conversion"]


class _DeferredWorkspaceSaveWorker:
    def __init__(
        self,
        _fname: str | os.PathLike[str],
        snapshot: manager_workspace._WorkspaceSaveSnapshot,
    ) -> None:
        self.signals = manager_workspace._WorkspaceSaveWorkerSignals()
        self._snapshot = snapshot

    def finish(
        self, *, ok: bool = True, elapsed: float = 0.0, error_text: str = ""
    ) -> None:
        self._snapshot.close()
        self.signals.finished.emit(ok, elapsed, error_text)


class _DeferredWorkspaceSaveThreadPool:
    def __init__(self) -> None:
        self.workers: list[_DeferredWorkspaceSaveWorker] = []

    def start(self, worker: _DeferredWorkspaceSaveWorker) -> None:
        self.workers.append(worker)


def _install_deferred_workspace_save_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> _DeferredWorkspaceSaveThreadPool:
    pool = _DeferredWorkspaceSaveThreadPool()
    monkeypatch.setattr(
        QtCore.QThreadPool, "globalInstance", staticmethod(lambda: pool)
    )
    monkeypatch.setattr(
        manager_workspace, "_WorkspaceSaveWorker", _DeferredWorkspaceSaveWorker
    )
    return pool


def _bind_dirty_workspace_for_save_test(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
    tmp_path: pathlib.Path,
) -> pathlib.Path:
    fname = tmp_path / "background-save.itws"
    fname.touch()
    manager._workspace_state.path = fname.resolve()
    manager._mark_workspace_clean()
    manager._workspace_state.mark_layout_dirty()
    return fname


def _workspace_save_test_snapshot(
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
) -> manager_workspace._WorkspaceSaveSnapshot:
    return manager_workspace._WorkspaceSaveSnapshot(
        generation=manager._workspace_state.dirty_generation,
        root_attrs={},
        delta_save_count=manager._workspace_state.delta_save_count + 1,
    )


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


def _compact_workspace_before_shutdown_and_wait(
    qtbot,
    manager: erlab.interactive.imagetool.manager.ImageToolManager,
) -> bool:
    completed = False

    def _mark_completed() -> None:
        nonlocal completed
        completed = True

    requested = manager._workspace_controller._compact_workspace_before_shutdown(
        on_finished=_mark_completed
    )
    if requested:
        qtbot.wait_until(lambda: completed, timeout=10000)
    return requested


def test_manager_async_save_request_error_paths(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _critical(*args, **kwargs) -> int:
        critical_calls.append(args)
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(erlab.interactive.utils.MessageDialog, "critical", _critical)
    with manager_context() as manager:
        manager._show_workspace_save_worker_error("Traceback text")
        assert critical_calls[-1][2] == (
            "An error occurred while saving the workspace file."
        )

        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.save_in_progress = True
        assert not manager._workspace_controller.save()

        manager._workspace_state.save_in_progress = False
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_snapshot",
            lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot failed")),
        )
        monkeypatch.setattr(
            manager, "_restore_focus_after_workspace_save", lambda _origin: None
        )
        callback_results: list[bool] = []
        assert not manager._workspace_controller.save(
            on_finished=callback_results.append
        )
        assert callback_results == [False]
        assert operation_errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: None)
        assert not manager._workspace_controller.save_as()


def test_manager_save_action_runs_workspace_save_in_background(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()
        fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        assert manager.workspace_path == str(fname.resolve())
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )
        monkeypatch.setattr(
            manager,
            "save_as",
            lambda **_kwargs: pytest.fail("Save action unexpectedly used Save As"),
        )

        manager.save_action.trigger()

        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        assert not manager.save_action.isEnabled()
        assert not manager.save_as_action.isEnabled()
        assert not manager.compact_workspace_action.isEnabled()
        assert not manager.import_workspace_action.isEnabled()
        assert manager.is_workspace_modified
        manager._update_actions()
        assert not manager.offload_action.isEnabled()
        manager.tree_view.deselect_all()
        manager._update_actions()
        assert not manager.offload_action.isEnabled()

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert manager.save_action.isEnabled()
        assert manager.save_as_action.isEnabled()
        assert manager.compact_workspace_action.isEnabled()
        assert manager.import_workspace_action.isEnabled()
        assert not manager.offload_action.isEnabled()
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_workspace_save_keeps_new_changes_and_queues_followup(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager._workspace_controller.save()
        manager._workspace_controller._mark_workspace_options_dirty()
        assert not manager._workspace_controller.save()

        assert len(pool.workers) == 1
        pool.workers[0].finish()
        qtbot.wait_until(lambda: len(pool.workers) == 2)
        assert manager._workspace_state.save_in_progress
        assert manager.is_workspace_modified

        pool.workers[1].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_workspace_save_keeps_duplicate_layout_change_dirty(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager._workspace_controller.save()
        assert len(pool.workers) == 1
        snapshot_generation = pool.workers[0]._snapshot.generation
        assert manager._workspace_state.mark_layout_dirty()
        assert manager._workspace_state.dirty_generation > snapshot_generation

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert manager.is_workspace_modified
        assert manager._workspace_state.layout_modified
        manager._workspace_state.path = None


def test_manager_background_full_save_preserves_post_snapshot_data_edit(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=("x", "y"),
            coords={"x": np.arange(5.0), "y": np.arange(5.0)},
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "post-snapshot-edit.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._rebind_workspace_backed_imagetools(fname, targets=[0], chunks={})
        slicer_area = root.slicer_area
        assert slicer_area.data_chunked
        manager._mark_workspace_clean()

        manager._mark_node_data_dirty(uid)
        manager._workspace_state.needs_full_save = True
        assert manager._workspace_controller.save()
        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        snapshot_generation = pool.workers[0]._snapshot.generation

        replacement = xr.DataArray(
            np.full((5, 5), 42.0),
            dims=("x", "y"),
            coords={"x": np.arange(5.0), "y": np.arange(5.0)},
            name="source",
        )
        slicer_area.replace_source_data(
            replacement,
            auto_compute=False,
            emit_edited=True,
        )
        assert any(
            event.uid == uid and event.data and event.generation > snapshot_generation
            for event in manager._workspace_state.dirty_events
        )

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        np.testing.assert_array_equal(slicer_area._data.values, replacement.values)
        assert not slicer_area.data_chunked
        assert manager.is_workspace_modified
        assert uid in manager._workspace_state.dirty_data
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_background_workspace_save_failure_restores_state(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        errors: list[str] = []
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        monkeypatch.setattr(manager, "_show_workspace_save_worker_error", errors.append)

        assert manager._workspace_controller.save()
        assert manager._workspace_state.save_in_progress

        pool.workers[0].finish(ok=False, error_text="worker boom")
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert errors
        assert "worker boom" in errors[-1]
        assert manager.save_action.isEnabled()
        assert manager.save_as_action.isEnabled()
        assert manager.compact_workspace_action.isEnabled()
        assert manager.import_workspace_action.isEnabled()
        assert manager.is_workspace_modified
        assert not operation_errors
        manager._mark_workspace_clean()
        manager._workspace_state.path = None


def test_manager_save_slot_requests_async_save(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        operation_errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: operation_errors.append((title, text)),
        )
        _fname = _bind_dirty_workspace_for_save_test(manager, tmp_path)
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_snapshot",
            lambda _path: _workspace_save_test_snapshot(manager),
        )

        assert manager.save() is None
        assert len(pool.workers) == 1
        assert manager._workspace_state.save_in_progress
        assert manager.is_workspace_modified

        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        assert not manager._workspace_state.save_in_progress
        assert not manager.is_workspace_modified
        assert not operation_errors
        manager._workspace_state.path = None


def test_manager_close_ignored_while_workspace_save_in_progress(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        event = QtGui.QCloseEvent()
        try:
            manager.closeEvent(event)
            assert not event.isAccepted()
        finally:
            manager._workspace_state.save_in_progress = False


def test_open_multiple_files_workspace_locks_before_recovery(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "locked-dropped.itws"
    _write_transaction_test_workspace(fname)
    lock = manager_workspace._acquire_workspace_document_lock(fname)
    lock_calls: list[pathlib.Path] = []
    recovery_calls: list[pathlib.Path] = []
    try:
        monkeypatch.setattr(
            manager_workspace,
            "_recover_workspace_transactions",
            lambda path: (
                recovery_calls.append(pathlib.Path(path))
                or pytest.fail("Dropped workspace should lock before recovery")
            ),
        )
        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._actions,
            "_show_workspace_file_lock_error",
            lambda _parent, locked_fname: lock_calls.append(pathlib.Path(locked_fname)),
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "file_loaders",
            lambda *args, **kwargs: pytest.fail(
                "locked workspace should not fall through to loaders"
            ),
        )

        with manager_context() as manager:
            manager.open_multiple_files([fname], try_workspace=True)
    finally:
        lock.unlock()

    assert recovery_calls == []
    assert lock_calls == [fname]


def test_open_multiple_files_loads_workspace_and_reads_metadata(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "open-multiple.itws"
        manager._save_workspace_document(fname, force_full=True)

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        manager.open_multiple_files([fname], try_workspace=True)

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.workspace_path == str(fname.resolve())


def test_manager_drop_h5_file_does_not_try_workspace(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "data.h5"
    calls: list[tuple[list[pathlib.Path], bool]] = []

    with manager_context() as manager:
        monkeypatch.setattr(
            manager,
            "open_multiple_files",
            lambda paths, *, try_workspace=False: calls.append(
                (list(paths), try_workspace)
            ),
        )

        manager._handle_dropped_files([fname])

    assert calls == [([fname], False)]


def test_workspace_high_risk_path_detection() -> None:
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path.home() / "OneDrive" / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_high_risk(
        pathlib.Path("//server/share/workspace.itws")
    )


def test_workspace_lock_error_detection_message_variants() -> None:
    transient = OSError(errno.EACCES, "resource temporarily unavailable")
    assert manager_workspace._is_workspace_file_lock_error(transient)
    assert manager_workspace._is_workspace_file_lock_error(
        RuntimeError("file is already open by another process")
    )
    assert manager_workspace._is_workspace_file_lock_error(
        RuntimeError("unable to lock file")
    )
    assert not manager_workspace._is_workspace_file_lock_error(
        OSError(errno.EINVAL, "resource temporarily unavailable")
    )


def test_hide_workspace_lock_file_windows_paths(monkeypatch) -> None:
    import ctypes

    calls: list[tuple[str, int]] = []

    class _Kernel32:
        @staticmethod
        def SetFileAttributesW(path: str, attrs: int) -> None:
            calls.append((path, attrs))

    monkeypatch.setattr(manager_workspace.sys, "platform", "win32")
    monkeypatch.setattr(manager_workspace.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    manager_workspace._hide_workspace_lock_file("missing-windll.itws.lock")
    assert calls == []

    monkeypatch.setattr(
        ctypes, "windll", types.SimpleNamespace(kernel32=_Kernel32()), raising=False
    )
    manager_workspace._hide_workspace_lock_file("hidden.itws.lock")
    assert calls == [("hidden.itws.lock", 0x2)]


def test_workspace_document_lock_info_without_lock(tmp_path) -> None:
    info = manager_workspace._workspace_document_lock_info(tmp_path / "free.itws")

    assert info.pid is None
    assert info.hostname == ""
    assert info.appname == ""


def test_workspace_metadata_helpers_cover_invalid_payloads() -> None:
    manifest_attrs = manager_workspace._workspace_root_attrs_payload(
        root_order=["1"],
        nodes=[{"path": "1"}],
        delta_save_count=2,
        erlab_version="test",
    )
    raw_manifest = manifest_attrs[manager_workspace._WORKSPACE_MANIFEST_ATTR]

    assert (
        manager_workspace._workspace_manifest_from_attrs(
            {manager_workspace._WORKSPACE_MANIFEST_ATTR: raw_manifest.encode()}
        )["delta_save_count"]
        == 2
    )
    manifest = manager_workspace._workspace_manifest_from_attrs(
        {manager_workspace._WORKSPACE_MANIFEST_ATTR: raw_manifest}
    )
    assert manager_workspace._workspace_manifest_repack_estimate(
        manifest, delta_save_count=2
    ) == (0, 0, True)
    assert manager_workspace._workspace_manifest_repack_estimate(
        {"delta_save_count": 2}, delta_save_count=2
    ) == (0, 0, False)
    assert manager_workspace._workspace_manifest_repack_estimate(
        None, delta_save_count=2
    ) == (0, 0, False)
    assert (
        manager_workspace._workspace_manifest_nonnegative_int(
            {"estimated_obsolete_bytes": "not-an-int"},
            "estimated_obsolete_bytes",
        )
        == 0
    )
    assert (
        manager_workspace._workspace_manifest_from_attrs(
            {manager_workspace._WORKSPACE_MANIFEST_ATTR: "{not-json"}
        )
        == {}
    )
    assert (
        manager_workspace._workspace_delta_save_count_from_attrs(
            {
                manager_workspace._WORKSPACE_MANIFEST_ATTR: (
                    '{"delta_save_count": "not-an-int"}'
                )
            }
        )
        == 0
    )
    with pytest.raises(ValueError, match="current workspace schema"):
        manager_workspace._compacted_workspace_root_attrs(
            {"imagetool_workspace_schema_version": 1}
        )
    assert manager_workspace._workspace_root_attrs_with_repack_estimate(
        {"imagetool_workspace_schema_version": 1},
        estimated_obsolete_bytes=1,
        replacement_delta_count=1,
    ) == {"imagetool_workspace_schema_version": 1}


def test_workspace_path_risk_detection_fallbacks(monkeypatch, tmp_path) -> None:
    def _raise_oserror(_path: pathlib.Path) -> pathlib.Path:
        raise OSError("resolve failed")

    monkeypatch.setattr(pathlib.Path, "resolve", _raise_oserror)
    assert manager_workspace._workspace_path_is_likely_cloud_path(
        tmp_path / "Dropbox" / "workspace.itws"
    )
    assert manager_workspace._workspace_path_is_likely_network_path(
        pathlib.Path("/net/server/workspace.itws")
    )

    monkeypatch.setattr(manager_workspace.sys, "platform", "darwin")
    assert manager_workspace._workspace_path_is_likely_network_path(
        pathlib.Path("/Volumes/share/workspace.itws")
    )


def test_workspace_requires_full_save_reasons(tmp_path) -> None:
    options = erlab.interactive.options
    old_incremental = options["io/workspace/use_incremental"]
    old_remote = options["io/workspace/incremental_save_on_remote"]
    existing = tmp_path / "existing.itws"
    existing.touch()
    try:
        options["io/workspace/use_incremental"] = False
        assert manager_workspace._workspace_requires_full_save(
            existing,
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )

        options["io/workspace/use_incremental"] = True
        options["io/workspace/incremental_save_on_remote"] = True
        assert manager_workspace._workspace_requires_full_save(
            tmp_path / "missing.itws",
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )
        for kwargs in (
            {"needs_full_save": True},
            {
                "schema_version": (
                    manager_workspace._current_workspace_schema_version() - 1
                )
            },
            {"structure_modified": True},
            {"has_dirty_added": True},
            {"has_dirty_removed": True},
        ):
            call_kwargs = {
                "needs_full_save": False,
                "schema_version": manager_workspace._current_workspace_schema_version(),
                "structure_modified": False,
                "has_dirty_added": False,
                "has_dirty_removed": False,
            }
            call_kwargs.update(kwargs)
            assert manager_workspace._workspace_requires_full_save(
                existing, **call_kwargs
            )
    finally:
        options["io/workspace/use_incremental"] = old_incremental
        options["io/workspace/incremental_save_on_remote"] = old_remote


def test_workspace_h5_transaction_helper_edge_cases(tmp_path) -> None:
    import h5py

    fname = tmp_path / "transaction-helpers.itws"
    with h5py.File(fname, "w") as h5_file:
        h5_file.attrs["imagetool_workspace_schema_version"] = (
            manager_workspace._current_workspace_schema_version()
        )
        assert manager_workspace._workspace_txn_attr_target(h5_file, "/missing") is None

        txn = h5_file.create_group(
            f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
        )
        txn_name = txn.name.strip("/")
        manager_workspace._restore_workspace_attr_backups(h5_file, txn)

        txn.attrs["operations"] = b'{"group_replacements": []}'
        assert manager_workspace._workspace_transaction_operations(txn) == {
            "group_replacements": []
        }
        txn.attrs["operations"] = "{not-json"
        assert manager_workspace._workspace_transaction_operations(txn) == {}

        txn.attrs["pending_root"] = b"__itws_pending_x"
        txn.attrs["backup_root"] = b"__itws_backup_x"
        assert manager_workspace._workspace_transaction_roots(txn) == (
            "__itws_pending_x",
            "__itws_backup_x",
        )

        manager_workspace._rollback_workspace_group_operations(
            h5_file, {"group_replacements": "not-a-list"}
        )
        manager_workspace._rollback_workspace_group_operations(
            h5_file,
            {"group_replacements": [None, {"group_path": 1, "backup_path": "x"}]},
        )

        target = h5_file.create_group("target")
        target.attrs["value"] = "old"
        txn.attrs["status"] = b"committing"
        txn.attrs["operations"] = json.dumps(
            {
                "group_replacements": [
                    {
                        "group_path": "target",
                        "backup_path": "missing-backup",
                        "old_exists": False,
                    }
                ]
            }
        )
        pending = h5_file.create_group("__itws_pending_x")
        pending.attrs["unused"] = True
        backup = h5_file.create_group("__itws_backup_x")
        backup.attrs["unused"] = True

        manager_workspace._recover_open_workspace_transaction(h5_file, txn.name)

        assert "target" not in h5_file
        assert "__itws_pending_x" not in h5_file
        assert "__itws_backup_x" not in h5_file
        assert txn_name not in h5_file


def test_recover_workspace_transactions_ignores_non_workspace_file(tmp_path) -> None:
    import h5py

    fname = tmp_path / "plain.h5"
    with h5py.File(fname, "w") as h5_file:
        h5_file.create_group(
            f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x"
        )

    manager_workspace._recover_workspace_transactions(fname)

    with h5py.File(fname, "r") as h5_file:
        assert f"{manager_workspace._WORKSPACE_TRANSACTION_GROUP_PREFIX}x" in h5_file


def test_validate_workspace_h5_file_rejects_non_workspace(tmp_path) -> None:
    import h5py

    fname = tmp_path / "invalid.h5"
    with h5py.File(fname, "w"):
        pass

    with pytest.raises(ValueError, match="not valid"):
        manager_workspace._validate_workspace_h5_file(fname)


def test_fsync_parent_directory_skips_non_posix(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(manager_workspace.os, "name", "nt")
    monkeypatch.setattr(
        manager_workspace.os,
        "open",
        lambda *args, **kwargs: pytest.fail("non-posix platforms should not fsync"),
    )

    manager_workspace._fsync_parent_directory(tmp_path / "workspace.itws")


def test_manager_workspace_v4_save_open_replaces_and_binds_path(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        fname = tmp_path / "bound.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            assert h5_file.attrs["imagetool_workspace_schema_version"] == 4
            manifest = json.loads(h5_file.attrs["imagetool_workspace_manifest"])
        assert manifest["schema_version"] == 4
        assert {node["uid"] for node in manifest["nodes"]} >= {
            manager._tool_graph.root_wrappers[0].uid,
            child_uid,
        }

        extra = itool(data + 2, manager=False, execute=False)
        assert isinstance(extra, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(extra, show=False)
        assert manager.ntools == 2

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == [child_uid]
        assert manager.get_imagetool(0).slicer_area._data.chunks is None
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0


def test_manager_workspace_h5py_fast_path_preserves_spaced_associated_coord(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": np.arange(5.0),
                "y": np.arange(5.0),
                "Fake Motor": (("x", "y"), np.arange(25.0).reshape((5, 5))),
            },
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)

        fname = tmp_path / "spaced-coord.itws"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager._save_workspace_document(fname, force_full=True)
        assert not any("space in its name" in str(item.message) for item in caught)

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        monkeypatch.setattr(
            manager_xarray,
            "open_workspace_dataset",
            lambda *args, **kwargs: pytest.fail(
                "spaced numeric coords should stay on the h5py fast path"
            ),
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is None
        assert "Fake Motor" in loaded.coords
        xarray.testing.assert_equal(
            loaded.coords["Fake Motor"], data.coords["Fake Motor"]
        )


def test_manager_workspace_import_appends_without_reassociation(
    qtbot,
    monkeypatch,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    choose_dialog_calls = {"count": 0}

    class _SelectSecondDialog(_ChooseFromWorkspaceManifestDialog):
        def __init__(self, *args, **kwargs) -> None:
            choose_dialog_calls["count"] += 1
            super().__init__(*args, **kwargs)
            first_item = self.item_for_path("0")
            assert first_item is not None
            first_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

        def exec(self) -> QtWidgets.QDialog.DialogCode:
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._workspace_io,
        "_ChooseFromWorkspaceManifestDialog",
        _SelectSecondDialog,
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        base_tool = itool(data, manager=False, execute=False)
        assert isinstance(base_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(base_tool, show=False)

        current_fname = tmp_path / "current.itws"
        manager._save_workspace_document(current_fname, force_full=True)
        manager._adopt_workspace_path(current_fname)
        manager._mark_workspace_clean()

        import_tool = itool(data + 1, manager=False, execute=False)
        assert isinstance(import_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(import_tool, show=False)
        import_fname = tmp_path / "import.itws"
        manager._save_workspace_document(import_fname, force_full=True)

        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        manager._mark_workspace_clean()

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(import_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(import_fname.name)

        accept_dialog(
            lambda: manager.import_workspace(native=False),
            pre_call=_go_to_file,
        )

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert choose_dialog_calls["count"] == 1
        assert manager.workspace_path == str(current_fname.resolve())
        assert manager.is_workspace_modified


def test_manager_workspace_import_ignored_while_save_in_progress(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        monkeypatch.setattr(
            QtWidgets,
            "QFileDialog",
            lambda *_args, **_kwargs: pytest.fail(
                "Import should not open a file dialog during workspace save"
            ),
        )

        assert not manager.import_workspace(native=False)

        manager._workspace_state.save_in_progress = False


def test_manager_workspace_load_ignored_while_save_in_progress(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._workspace_state.save_in_progress = True
        monkeypatch.setattr(
            QtWidgets,
            "QFileDialog",
            lambda *_args, **_kwargs: pytest.fail(
                "Open should not show a file dialog during workspace save"
            ),
        )

        assert not manager.load(native=False)

        manager._workspace_state.save_in_progress = False


def test_manager_workspace_selected_import_uses_manifest_fast_path(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root_a = itool(data, manager=False, execute=False)
        assert isinstance(root_a, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_a, show=False)
        child_uid = manager.add_childtool(
            _WorkspaceSweepChildTool(data + 10), 0, show=False
        )
        root_b = itool(data + 1, manager=False, execute=False)
        assert isinstance(root_b, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root_b, show=False)
        figure_uid = manager.add_figuretool(
            _WorkspaceSweepFigureTool(data + 20), show=False
        )

        fname = tmp_path / "manifest-selected-import.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        nodes = manifest["nodes"]
        paths = {
            str(entry["path"])
            for entry in nodes
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }
        root_paths = sorted(path for path in paths if "/" not in path)
        child_paths = sorted(path for path in paths if "/childtools/" in path)
        figure_paths = sorted(
            path
            for path in paths
            if path.startswith("figures/") and path.count("/") == 1
        )
        assert root_paths == ["0", "1"]
        assert child_paths == ["0/childtools/" + child_uid]
        assert figure_paths == ["figures/" + figure_uid]

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        class _SelectRootOnlyDialog(_ChooseFromWorkspaceManifestDialog):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._uncheck_children()
                for path in root_paths[1:] + figure_paths:
                    item = self.item_for_path(path)
                    assert item is not None
                    item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

        monkeypatch.setattr(
            manager_workspace_io,
            "_ChooseFromWorkspaceManifestDialog",
            _SelectRootOnlyDialog,
        )
        monkeypatch.setattr(
            manager_xarray,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "selected current-schema loads should use manifest h5py path"
            ),
        )

        assert manager._load_workspace_file(
            fname,
            replace=False,
            associate=False,
            mark_dirty=True,
            select=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == []
        assert figure_uid not in manager._tool_graph.nodes


def test_manager_workspace_save_as_preserves_live_in_memory_windows(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            child = itool(data + 1.0, manager=False, execute=False)
            assert isinstance(child, erlab.interactive.imagetool.ImageTool)
            child_uid = manager.add_imagetool_child(child, 0, show=False)

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager, "_load_workspace_file", _load_workspace_file_should_not_run
            )

            new_fname = tmp_path / "new.itws"

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

            assert manager.workspace_path == str(new_fname.resolve())
            assert not manager.is_workspace_modified
            assert manager.get_imagetool(0) is root
            assert manager._child_node(child_uid).imagetool is child
            assert manager._tool_graph.root_wrappers[0]._childtool_indices == [
                child_uid
            ]
            assert root.slicer_area._data.chunks is None
            assert child.slicer_area._data.chunks is None
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_offload_to_workspace_save_as_rebinds_root_as_dask(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)), dims=["x", "y"], name="source"
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()
        assert root.slicer_area._data.chunks is None

        fname = tmp_path / "offload.itws"

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(fname.name)

        results: list[bool] = []
        accept_dialog(
            lambda: results.append(manager.offload_to_workspace([0], native=False)),
            pre_call=_go_to_file,
        )

        assert results == [True]
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert manager.workspace_path == str(fname.resolve())
        assert not manager.is_workspace_modified

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert manager_xarray._normalized_file_path(rebound.encoding.get("source")) == (
            str(fname.resolve())
        )
        assert _compute_first_value(rebound) == 0.0

        manager._update_actions()
        assert not manager.offload_action.isEnabled()


def test_manager_workspace_load_reopens_offloaded_data_as_dask(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "offload-reopen.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is not None
        assert manager_xarray._normalized_file_path(loaded.encoding.get("source")) == (
            str(fname.resolve())
        )
        assert _compute_first_value(loaded) == 0.0


def test_manager_workspace_import_reopens_offloaded_data_as_dask(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "offload-import.itws"
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        loaded: list[bool] = []
        accept_dialog(
            lambda: loaded.append(
                manager._load_workspace_file(
                    fname,
                    replace=False,
                    associate=False,
                    mark_dirty=True,
                    select=True,
                )
            )
        )

        assert loaded == [True]
        loaded_data = manager.get_imagetool(0).slicer_area._data
        assert loaded_data.chunks is not None
        assert manager_xarray._normalized_file_path(
            loaded_data.encoding.get("source")
        ) == str(fname.resolve())
        assert _compute_first_value(loaded_data) == 0.0


def test_manager_workspace_load_reopens_offloaded_spaced_coord_data_as_dask(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25.0).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": np.arange(5.0),
                "y": np.arange(5.0),
                "Fake Motor": ("x", np.linspace(10.0, 20.0, 5)),
            },
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "offload-spaced-coord.itws"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager._save_workspace_document(fname, force_full=True)
        assert not any("space in its name" in str(item.message) for item in caught)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is not None
        assert "Fake Motor" in loaded.coords
        np.testing.assert_allclose(
            np.asarray(loaded.coords["Fake Motor"]),
            np.asarray(data.coords["Fake Motor"]),
        )


def test_manager_offload_to_workspace_saves_dirty_workspace_before_rebind(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "dirty-offload.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        updated = data + 10.0
        root.slicer_area.replace_source_data(
            updated, auto_compute=False, emit_edited=True
        )
        assert manager.is_workspace_modified

        original_save = manager._workspace_controller.save
        save_calls: list[bool] = []

        def _save(
            *,
            native: bool = True,
            on_finished: Callable[[bool], None] | None = None,
        ) -> bool:
            save_calls.append(native)
            return original_save(native=native, on_finished=on_finished)

        monkeypatch.setattr(manager._workspace_controller, "save", _save)

        assert manager.offload_to_workspace([0], native=False)
        assert save_calls == [False]
        qtbot.wait_until(lambda: root.slicer_area._data.chunks is not None)

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert _compute_first_value(rebound) == 10.0
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert saved[0, 0] == 10.0


def test_manager_compute_offloaded_workspace_data_marks_backing_dirty(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "compute-offloaded.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is not None
        assert not manager.is_workspace_modified

        root.slicer_area._compute_chunked()

        assert root.slicer_area._data.chunks is None
        assert uid in manager._workspace_state.dirty_data
        assert manager.is_workspace_modified

        select_tools(manager, [0])
        manager._update_actions()
        assert manager.offload_action.isEnabled()

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert saved.chunks is None


def test_manager_offload_to_workspace_save_cancel_or_failure_noop(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: None)
        assert not manager.offload_to_workspace([0], native=False)
        assert manager.workspace_path is None
        assert root.slicer_area._data.chunks is None

        fname = tmp_path / "failure-offload.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(manager._tool_graph.root_wrappers[0].uid)

        monkeypatch.setattr(
            manager._workspace_controller, "save", lambda **_kwargs: False
        )
        assert not manager.offload_to_workspace([0], native=False)
        assert root.slicer_area._data.chunks is None


def test_manager_offload_to_workspace_edge_paths(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[list[int]] = []
        monkeypatch.setattr(manager, "_selected_imagetool_targets", lambda: [0])
        monkeypatch.setattr(
            manager,
            "offload_to_workspace",
            lambda targets: calls.append(list(targets)) or True,
        )
        manager.offload_selected_to_workspace()
        assert calls == [[0]]

    with manager_context() as manager:
        assert not manager.offload_to_workspace([])

    fake_node = types.SimpleNamespace(
        is_imagetool=True,
        imagetool=object(),
        slicer_area=types.SimpleNamespace(data_chunked=False),
        pending_workspace_memory_payload=None,
    )

    with manager_context() as manager:
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: fake_node)
        monkeypatch.setattr(
            manager._workspace_controller, "save_as", lambda **_kwargs: False
        )
        assert not manager.offload_to_workspace([0], native=False)

    with manager_context() as manager:
        workspace = tmp_path / "offload-error.itws"
        manager._workspace_state.path = workspace
        manager._workspace_state.needs_full_save = False
        monkeypatch.setattr(manager, "_node_for_target", lambda _target: fake_node)
        monkeypatch.setattr(
            manager, "_active_managed_window", lambda: typing.cast("typing.Any", None)
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *_args, **_kwargs: contextlib.nullcontext(),
        )
        monkeypatch.setattr(
            manager,
            "_rebind_workspace_backed_imagetools",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        errors: list[tuple[str, str]] = []
        restored: list[object | None] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, message: errors.append((title, message)),
        )
        monkeypatch.setattr(
            manager, "_restore_focus_after_workspace_save", restored.append
        )

        assert not manager.offload_to_workspace([0], native=False)
        assert errors == [
            (
                "Error while offloading to workspace",
                "An error occurred while reconnecting data from the workspace file.",
            )
        ]
        assert restored == [None]


def test_manager_offload_to_workspace_preserves_child_source_state(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False, provenance_spec=provenance.full_data())

        child = itool(data.copy(deep=False), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(
            child,
            0,
            show=False,
            source_spec=provenance.full_data(),
            source_auto_update=True,
        )
        child_node = manager._child_node(child_uid)

        fname = tmp_path / "child-offload.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager.offload_to_workspace([0], native=False)
        assert manager.get_imagetool(0).slicer_area._data.chunks is not None
        assert child_node.source_state == "fresh"
        assert child.slicer_area._data.chunks is None

        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_actions()
        assert manager.offload_action.isEnabled()

        assert manager.offload_to_workspace([child_uid], native=False)
        assert child.slicer_area._data.chunks is not None
        assert child_node.source_state == "fresh"
        assert _compute_first_value(child.slicer_area._data) == 0.0

        manager._update_actions()
        assert not manager.offload_action.isEnabled()


def test_manager_manual_chunk_edits_persist_on_next_workspace_save(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "manual-chunks.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        root.slicer_area._set_chunks({"x": 2, "y": 3})

        assert root.slicer_area._data.chunks == ((2, 2, 1), (3, 2))
        assert uid in manager._workspace_state.dirty_data
        assert manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert saved.chunks is None

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert not manager.is_workspace_modified

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert saved.chunks == (2, 3)

        opened = manager_xarray.open_workspace_dataset(
            fname, manager._workspace_payload_path(uid), chunks={}
        )
        try:
            rebound = opened[manager_workspace_io._ITOOL_DATA_NAME]
            assert rebound.chunks == ((2, 2, 1), (3, 2))
        finally:
            opened.close()


def test_manager_workspace_full_save_preserves_non_dask_data(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            assert root.slicer_area._data.chunks is None

            fname = tmp_path / "full-save.itws"
            manager._save_workspace_document(fname, force_full=True)
            manager._adopt_workspace_path(fname)
            manager._mark_workspace_clean()
            manager._workspace_state.needs_full_save = True

            assert _request_workspace_save_and_wait(qtbot, manager)
            assert root.slicer_area._data.chunks is None
            assert _compute_first_value(root.slicer_area._data) == 0.0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_save_as_preserves_external_non_dask_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        old_fname = tmp_path / "old.h5"
        new_fname = tmp_path / "new.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            old_fname, engine="h5netcdf", invalid_netcdf=True
        )
        source = _open_external_file_backed_hdf5_imagetool_data(old_fname)
        assert source.chunks is None

        root = itool(source, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        live_data = manager.get_imagetool(0).slicer_area._data
        old_source = str(old_fname.resolve())
        assert (
            manager_xarray._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        rebound_source = manager_xarray._normalized_file_path(
            rebound.encoding.get("source")
        )
        assert rebound_source == old_source
        assert rebound_source != new_source
        assert rebound.chunks is None

        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_rebinds_workspace_non_dask_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        old_fname = tmp_path / "old.itws"
        new_fname = tmp_path / "new.itws"
        manager._save_workspace_document(old_fname, force_full=True)
        manager._adopt_workspace_path(old_fname)
        manager._rebind_workspace_backed_imagetools(old_fname, targets=[0], chunks=None)

        live_data = manager.get_imagetool(0).slicer_area._data
        old_source = str(old_fname.resolve())
        assert (
            manager_xarray._normalized_file_path(live_data.encoding.get("source"))
            == old_source
        )
        assert live_data.chunks is None

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        new_source = str(new_fname.resolve())
        assert (
            manager_xarray._normalized_file_path(rebound.encoding.get("source"))
            == new_source
        )
        assert rebound.chunks is None

        old_fname.unlink()
        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_preserves_manually_chunked_file_backed_data(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        old_fname = tmp_path / "old.h5"
        new_fname = tmp_path / "new.itws"
        xr.DataTree.from_dict({"0/imagetool": data.to_dataset(name="data")}).to_netcdf(
            old_fname, engine="h5netcdf", invalid_netcdf=True
        )
        source = _open_external_file_backed_hdf5_imagetool_data(old_fname)
        assert source.chunks is None

        root = itool(source, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.slicer_area.replace_source_data(
            root.slicer_area._data.chunk({"x": 2, "y": 2}),
            auto_compute=False,
        )
        assert root.slicer_area._data.chunks is not None

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(tmp_path))
            dialog.selectFile(str(new_fname))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText(new_fname.name)

        accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

        rebound = manager.get_imagetool(0).slicer_area._data
        assert rebound.chunks is not None
        assert manager_xarray._normalized_file_path(
            rebound.encoding.get("source")
        ) == str(new_fname.resolve())

        old_fname.unlink()
        assert _compute_first_value(rebound) == 0.0


def test_manager_workspace_save_as_rebinds_lazy_data_to_new_document(
    qtbot,
    accept_dialog,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            old_fname = tmp_path / "old.h5"
            new_fname = tmp_path / "new.itws"
            xr.DataTree.from_dict(
                {"0/imagetool": data.to_dataset(name="data")}
            ).to_netcdf(old_fname, engine="h5netcdf", invalid_netcdf=True)
            old_lazy = _open_external_lazy_hdf5_imagetool_data(old_fname)
            root.slicer_area.replace_source_data(old_lazy + 0, auto_compute=False)
            assert _compute_first_value(old_lazy) == 0

            def _load_workspace_file_should_not_run(*args, **kwargs):
                raise AssertionError("Save As should not reload the saved workspace")

            monkeypatch.setattr(
                manager, "_load_workspace_file", _load_workspace_file_should_not_run
            )

            def _go_to_file(dialog: QtWidgets.QFileDialog):
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(new_fname))
                focused = dialog.focusWidget()
                if isinstance(focused, QtWidgets.QLineEdit):
                    focused.setText(new_fname.name)

            accept_dialog(lambda: manager.save_as(native=False), pre_call=_go_to_file)
            qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)

            assert manager.workspace_path == str(new_fname.resolve())
            rebound = manager.get_imagetool(0).slicer_area._data
            assert rebound.chunks is not None
            assert manager_xarray._normalized_file_path(
                rebound.encoding.get("source")
            ) == str(new_fname.resolve())
            old_fname.unlink()
            assert _compute_first_value(rebound) == 0
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_workspace_dirty_markers_are_node_scoped(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        fname = tmp_path / "dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()
        assert not child.isWindowModified()

        manager._child_node(child_uid).name = "renamed child"
        assert manager.is_workspace_modified
        assert not manager.isWindowModified()
        assert not root.isWindowModified()
        assert not child.isWindowModified()

        manager._flush_idle_work(force=True)

        assert manager.isWindowModified()
        assert not root.isWindowModified()
        assert child.isWindowModified()
        details = manager._dirty_details_text()
        assert "State modified:\n- renamed child" in details
        assert "Data modified:" not in details


def test_manager_workspace_load_context_batches_secondary_ui_refreshes(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[tuple[str, object]] = []

        monkeypatch.setattr(manager, "_figure_uids", list)
        monkeypatch.setattr(
            manager,
            "_set_figures_tab_available",
            lambda available: calls.append(("figures", available)),
        )
        monkeypatch.setattr(
            manager._details_panel,
            "_update_info",
            lambda *, uid=None: calls.append(("info", uid)),
        )
        monkeypatch.setattr(
            manager._details_panel,
            "_update_actions",
            lambda: calls.append(("actions", None)),
        )
        monkeypatch.setattr(
            manager._lineage_controller,
            "_refresh_dependency_dependents",
            lambda uid: calls.append(("dependency", uid)),
        )
        update_figure_gallery_icon = manager._update_figure_gallery_icon

        def _update_figure_gallery_icon(uid: str) -> None:
            if manager._workspace_ui_refresh_defer_depth > 0:
                update_figure_gallery_icon(uid)
                return
            calls.append(("gallery", uid))

        monkeypatch.setattr(
            manager,
            "_update_figure_gallery_icon",
            _update_figure_gallery_icon,
        )
        refresh_figure_source_controls = manager._refresh_figure_source_controls

        def _refresh_figure_source_controls() -> None:
            if manager._workspace_ui_refresh_defer_depth > 0:
                refresh_figure_source_controls()
                return
            calls.append(("source_controls", None))

        monkeypatch.setattr(
            manager,
            "_refresh_figure_source_controls",
            _refresh_figure_source_controls,
        )

        with manager._workspace_load_context():
            manager._sync_figures_ui(select_uid="figure")
            manager._update_info(uid="figure")
            manager._update_info(uid="figure")
            manager._update_actions()
            manager._update_actions()
            manager._refresh_dependency_dependents("source")
            manager._refresh_dependency_dependents("source")
            manager._update_figure_gallery_icon("figure")
            manager._schedule_figure_gallery_icon_update("figure")
            manager._schedule_figure_gallery_icon_update("figure")
            manager._refresh_figure_source_controls()
            manager._refresh_figure_source_controls()
            assert calls == []

        assert calls == [
            ("figures", False),
            ("dependency", "source"),
            ("source_controls", None),
            ("gallery", "figure"),
            ("actions", None),
            ("info", "figure"),
        ]


def test_manager_workspace_save_clears_deferred_dirty_events(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "deferred-dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        QtCore.QTimer.singleShot(0, lambda: manager._mark_node_state_dirty(uid))
        manager._mark_node_state_dirty(uid)
        assert manager.is_workspace_modified
        assert not root.isWindowModified()

        manager._flush_idle_work(force=True)

        assert root.isWindowModified()

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(manager, "_active_managed_window", lambda: root)
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        manager._drain_workspace_deferred_events()
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()
        assert focus_restored == [root]


def test_manager_workspace_save_during_active_interaction_uses_dirty_state(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "active-save.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        manager._note_interaction_activity()
        manager._mark_node_state_dirty(uid)

        assert manager.is_workspace_modified
        assert not root.isWindowModified()

        assert _request_workspace_save_and_wait(qtbot, manager)

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()

        manager._flush_idle_work(force=True)

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_workspace_restore_event_drain_avoids_event_loop(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _Emitter(QtCore.QObject):
        sigRecord = QtCore.Signal()

    class _Receiver(QtCore.QObject):
        @QtCore.Slot()
        def record(self) -> None:
            calls.append("record")

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        calls: list[str] = []
        emitter = _Emitter(manager)
        receiver = _Receiver(manager)
        deferred_widget = QtWidgets.QWidget(manager)
        emitter.sigRecord.connect(
            receiver.record,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        emitter.sigRecord.emit()
        deferred_widget.deleteLater()

        with monkeypatch.context() as restore_drain_patch:
            restore_drain_patch.setattr(
                QtWidgets.QApplication,
                "processEvents",
                lambda *_args, **_kwargs: pytest.fail(
                    "workspace restore draining must not spin the event loop"
                ),
            )
            manager._drain_workspace_restore_events()

    assert calls == ["record"]
    assert not erlab.interactive.utils.qt_is_valid(deferred_widget)


def test_manager_workspace_save_drain_does_not_force_deferred_delete(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager, monkeypatch.context() as save_drain_patch:
        event_types: list[int] = []
        save_drain_patch.setattr(
            QtWidgets.QApplication,
            "sendPostedEvents",
            lambda _receiver, event_type: event_types.append(event_type),
        )
        save_drain_patch.setattr(
            QtWidgets.QApplication, "processEvents", lambda *_args, **_kwargs: None
        )

        manager._drain_workspace_deferred_events()

        assert event_types == [int(QtCore.QEvent.Type.MetaCall.value)] * 3


def test_manager_workspace_state_save_updates_attrs_without_full_rewrite(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "state-delta.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)

        original_transaction_write = manager_workspace._write_workspace_transaction_file
        attr_write_calls: list[str] = []

        def _record_transaction_write(
            _fname: str | os.PathLike[str],
            rewrite_groups: Iterable[tuple[str, dict[str, xr.Dataset]]],
            attr_updates: Iterable[
                tuple[
                    str,
                    dict[str, typing.Any],
                    tuple[str, dict[str, xr.Dataset]],
                ]
            ],
            root_attrs: Mapping[str, typing.Any],
            **kwargs: typing.Any,
        ) -> None:
            rewrite_groups = tuple(rewrite_groups)
            updates = tuple(attr_updates)
            assert rewrite_groups == ()
            attr_write_calls.extend(update[0] for update in updates)
            original_transaction_write(
                _fname, rewrite_groups, updates, root_attrs, **kwargs
            )

        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only Save should not rewrite the full workspace"
            ),
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_root_attrs_to_file",
            lambda *args, **kwargs: pytest.fail(
                "state-only Save should batch root attrs with node attrs"
            ),
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _record_transaction_write,
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert attr_write_calls == ["0/imagetool"]
        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_workspace_save_does_not_close_live_workspace_handles(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "live-handles.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)

        assert _request_workspace_save_and_wait(qtbot, manager)

        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)


def test_manager_workspace_save_preserves_live_lazy_readers_during_write(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25.0).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "live-lazy.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        root = manager.get_imagetool(0)
        uid = manager._tool_graph.root_wrappers[0].uid
        root.slicer_area.replace_source_data(
            manager._workspace_rebind_data_for_uid(fname, uid, chunks="auto"),
            auto_compute=False,
        )
        live_data = root.slicer_area._data
        assert live_data.chunks is not None
        assert _compute_first_value(live_data) == 0.0

        manager._mark_node_state_dirty(uid)
        original_write = manager_workspace._write_workspace_transaction_file
        computed_values: list[object] = []

        def _slow_write_workspace_transaction_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        def _compute_live_data() -> None:
            computed_values.append(live_data.isel({"x": 1, "y": 1}).compute().item())

        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _slow_write_workspace_transaction_file,
        )
        QtCore.QTimer.singleShot(10, _compute_live_data)

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert computed_values == [6.0]


def test_manager_workspace_slow_save_reports_status_after_background_write(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "slow-save.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(manager._tool_graph.root_wrappers[0].uid)

        original_write = manager_workspace._write_workspace_transaction_file

        def _slow_write_workspace_transaction_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        focus_restored: list[QtWidgets.QWidget | None] = []
        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._workspace_io,
            "_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _slow_write_workspace_transaction_file,
        )
        monkeypatch.setattr(manager, "_active_managed_window", lambda: root)
        monkeypatch.setattr(
            manager,
            "_restore_focus_after_workspace_save",
            lambda origin: focus_restored.append(origin),
        )

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._status_bar.currentMessage().startswith("Workspace saved")
        assert focus_restored == [root]


def test_manager_workspace_save_keeps_post_command_changes_dirty(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    pool = _install_deferred_workspace_save_worker(monkeypatch)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "post-command-dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)

        assert manager._workspace_controller.save()
        manager._mark_node_state_dirty(uid)
        pool.workers[0].finish()
        qtbot.wait_until(lambda: not manager._workspace_state.save_in_progress)
        assert manager.is_workspace_modified
        assert root.isWindowModified()
        details = manager._dirty_details_text()
        assert "State modified:" in details
        assert "Data modified:" not in details


def test_manager_workspace_compact_resets_delta_count_and_cleans_internal_groups(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest["estimated_obsolete_bytes"] == 0
        assert manifest["replacement_delta_count"] == 0

        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )

        assert manager.compact_workspace()
        assert manager._workspace_state.delta_save_count == 0
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        _assert_no_workspace_internal_groups(fname)
        with h5py.File(fname, "r") as h5_file:
            assert (
                manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs)
                == 0
            )
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_delta_save_tracks_repack_benefit(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "delta-benefit.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)

        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.estimated_obsolete_bytes > 0
        assert manager._workspace_state.replacement_delta_count == 1
        assert manager._workspace_state.repack_estimate_known
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert (
            manifest["estimated_obsolete_bytes"]
            == manager._workspace_state.estimated_obsolete_bytes
        )
        assert manifest["replacement_delta_count"] == 1

        manager._save_workspace_document(fname, force_full=True)

        assert manager._workspace_state.delta_save_count == 0
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
        assert "estimated_obsolete_bytes" not in manifest
        assert "replacement_delta_count" not in manifest


def test_manager_workspace_shutdown_compacts_clean_delta_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.estimated_obsolete_bytes > 0
        assert manager._workspace_state.replacement_delta_count == 1

        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )

        assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)

        assert manager._workspace_state.delta_save_count == 0
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        _assert_no_workspace_internal_groups(fname)
        with h5py.File(fname, "r") as h5_file:
            assert (
                manager_workspace._workspace_delta_save_count_from_attrs(h5_file.attrs)
                == 0
            )
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_shutdown_compact_uses_file_level_repack(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-file-repack.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.replacement_delta_count == 1

        def _fail_to_datatree() -> xr.DataTree:
            pytest.fail("Shutdown file-level repack should not serialize tools")

        monkeypatch.setattr(manager, "_to_datatree", _fail_to_datatree)
        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )

        assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)

        assert manager._workspace_state.delta_save_count == 0
        _assert_no_workspace_internal_groups(fname)
        with h5py.File(fname, "r") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
            assert "delta_save_count" not in manifest
            assert "transaction_protocol" not in manifest


def test_manager_workspace_shutdown_compact_skips_state_only_delta(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-state-skip.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.replacement_delta_count == 0

        monkeypatch.setattr(
            manager._workspace_controller,
            "_start_workspace_save_worker",
            lambda *args, **kwargs: pytest.fail(
                "State-only shutdown compaction should skip"
            ),
        )

        assert not manager._workspace_controller._compact_workspace_before_shutdown()

        assert manager._workspace_state.delta_save_count == 1


def test_manager_workspace_shutdown_compact_skips_below_threshold_data_delta(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-threshold-skip.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1
        assert manager._workspace_state.estimated_obsolete_bytes > 0
        assert manager._workspace_state.replacement_delta_count == 1

        monkeypatch.setattr(
            manager._workspace_controller,
            "_start_workspace_save_worker",
            lambda *args, **kwargs: pytest.fail(
                "Below-threshold shutdown compaction should skip"
            ),
        )

        assert not manager._workspace_controller._compact_workspace_before_shutdown()

        assert manager._workspace_state.delta_save_count == 1


def test_manager_workspace_shutdown_compact_scans_when_estimate_missing(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-missing-estimate.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        with h5py.File(fname, "a") as h5_file:
            manifest = manager_workspace._workspace_manifest_from_attrs(h5_file.attrs)
            manifest.pop("estimated_obsolete_bytes", None)
            manifest.pop("replacement_delta_count", None)
            h5_file.attrs[manager_workspace._WORKSPACE_MANIFEST_ATTR] = json.dumps(
                manifest
            )
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=0,
            replacement_delta_count=0,
            known=False,
        )

        scan_calls: list[pathlib.Path] = []

        def _fake_estimate(path: str | os.PathLike[str]) -> int:
            scan_calls.append(pathlib.Path(path))
            return 100

        monkeypatch.setattr(
            manager_workspace_io,
            "_workspace_obsolete_estimate",
            _fake_estimate,
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )

        assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)

        assert scan_calls == [fname.resolve()]
        assert manager._workspace_state.delta_save_count == 0


def test_manager_workspace_shutdown_compact_skips_if_file_repack_unavailable(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "shutdown-fallback-repack.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1

        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_file_repack_snapshot",
            lambda _generation: None,
        )

        def _fail_full_save_snapshot(
            _generation: int,
        ) -> manager_workspace._WorkspaceSaveSnapshot:
            pytest.fail("Shutdown compaction should not serialize as a fallback")

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_full_save_snapshot",
            _fail_full_save_snapshot,
        )

        assert not manager._workspace_controller._compact_workspace_before_shutdown()

        assert manager._workspace_state.delta_save_count == 1


def test_manager_workspace_shutdown_compact_runs_in_background(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "slow-shutdown-compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 1

        original_write = manager_workspace._write_full_workspace_tree_file

        def _slow_write_full_workspace_tree_file(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._workspace_io,
            "_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES",
            1,
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO",
            0.0,
        )
        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            _slow_write_full_workspace_tree_file,
        )
        assert _compact_workspace_before_shutdown_and_wait(qtbot, manager)
        assert manager._workspace_state.delta_save_count == 0


def test_manager_workspace_shutdown_compact_skips_dirty_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "dirty-shutdown-compact.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._workspace_state.delta_save_count = 1
        manager._mark_node_state_dirty(uid)

        monkeypatch.setattr(
            manager._workspace_controller,
            "_start_workspace_save_worker",
            lambda *args, **kwargs: pytest.fail(
                "Dirty shutdown compaction should not write discarded changes"
            ),
        )

        assert not manager._workspace_controller._compact_workspace_before_shutdown()


def test_manager_workspace_high_risk_path_forces_full_save_snapshot(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        fname = tmp_path / "high-risk.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        monkeypatch.setattr(
            manager_workspace, "_workspace_path_is_high_risk", lambda *_args: True
        )
        monkeypatch.setattr(
            manager_workspace,
            "_workspace_path_is_likely_network_path",
            lambda *_args: True,
        )

        snapshot = manager._workspace_save_snapshot(fname)
        try:
            assert snapshot.full_tree is not None
            assert snapshot.delta_save_count == 0
            assert snapshot.copy_groups == ()
        finally:
            snapshot.close()


def test_manager_workspace_save_snapshot_uses_compression_override(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        manager._set_workspace_option_overrides(
            {"io/workspace/compression": "blosclz3"}
        )
        assert manager._workspace_compression_mode() == "blosclz3"
        manager._workspace_state.delta_save_count = 2
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=128,
            replacement_delta_count=3,
            known=False,
        )
        root_attrs = manager._workspace_root_attrs_payload()
        manifest = manager_workspace._workspace_manifest_from_attrs(root_attrs)
        assert manifest["delta_save_count"] == 2
        assert "estimated_obsolete_bytes" not in manifest
        root_attrs = manager._workspace_root_attrs_payload(
            delta_save_count=1,
            estimated_obsolete_bytes=1024,
            replacement_delta_count=5,
            repack_estimate_known=True,
        )
        manifest = manager_workspace._workspace_manifest_from_attrs(root_attrs)
        assert manifest["estimated_obsolete_bytes"] == 1024
        assert manifest["replacement_delta_count"] == 5

        snapshot = manager._workspace_save_snapshot(tmp_path / "snapshot.itws")
        try:
            assert snapshot.compression_mode == "blosclz3"
        finally:
            snapshot.close()


def test_manager_workspace_delta_save_updates_repack_state(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        writes: list[tuple[object, ...]] = []

        def _snapshot(
            generation: int,
            root_attrs: dict[str, typing.Any],
            delta_save_count: int,
        ) -> manager_workspace._WorkspaceSaveSnapshot:
            return manager_workspace._WorkspaceSaveSnapshot(
                generation=generation,
                root_attrs=root_attrs,
                delta_save_count=delta_save_count,
                estimated_obsolete_bytes=256,
                replacement_delta_count=4,
                rewrite_groups=(),
                attr_updates=(),
            )

        def _record_write(*args, **kwargs) -> None:
            writes.append((*args, kwargs))

        monkeypatch.setattr(manager, "_workspace_delta_save_snapshot", _snapshot)
        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_transaction_file",
            _record_write,
        )

        manager._workspace_state.dirty_generation = 5
        manager._workspace_state.delta_save_count = 6
        manager._workspace_controller._save_workspace_delta(tmp_path / "delta.itws")

        assert writes
        assert manager._workspace_state.delta_save_count == 7
        assert manager._workspace_state.estimated_obsolete_bytes == 256
        assert manager._workspace_state.replacement_delta_count == 4


def test_manager_workspace_delta_snapshot_keeps_replacement_count_for_missing_groups(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller

        manager._workspace_state.path = tmp_path / "workspace.itws"
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=100,
            replacement_delta_count=2,
        )

        monkeypatch.setattr(
            manager,
            "_workspace_highest_dirty_data_roots",
            lambda: ("uid",),
        )
        monkeypatch.setattr(
            controller,
            "_workspace_rewrite_group_snapshot",
            lambda _uid: ("0", {}),
        )
        monkeypatch.setattr(manager, "_iter_descendant_uids", lambda _uid: ())
        monkeypatch.setattr(
            controller,
            "_workspace_manifest_node_uids",
            lambda _root_attrs: frozenset(),
        )
        monkeypatch.setattr(
            controller,
            "_workspace_stale_reference_rewrite_uids",
            lambda _manifest_uids: (),
        )
        monkeypatch.setattr(
            manager_workspace,
            "_workspace_h5_paths_storage_size",
            lambda *_args, **_kwargs: (256, 0),
        )

        snapshot = controller._workspace_delta_save_snapshot(1, {}, 3)

        assert snapshot.estimated_obsolete_bytes == 356
        assert snapshot.replacement_delta_count == 2


def test_manager_write_full_workspace_file_can_disable_group_reuse(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        tree = xr.DataTree()
        writes: list[dict[str, typing.Any]] = []

        monkeypatch.setattr(manager, "_to_datatree", lambda: tree)
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_full_save_copy_groups",
            lambda *_args, **_kwargs: pytest.fail("copy groups should not be reused"),
        )

        def _record_write(*_args, **kwargs) -> None:
            writes.append(kwargs)

        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            _record_write,
        )

        controller._write_full_workspace_file(
            tmp_path / "full.itws", reuse_unchanged_groups=False
        )

        assert writes
        assert writes[0]["copy_source"] is None
        assert writes[0]["copy_groups"] == ()


def test_workspace_manifest_first_helper_edge_cases(
    qtbot,
    tmp_path,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        with monkeypatch.context() as mp:
            mp.setattr(controller, "_workspace_node_path", lambda uid: uid)
            assert not controller._workspace_datatree_for_payload_uids(
                ("missing",)
            ).children
        assert (
            controller._workspace_full_save_manifest_entries(
                {
                    manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(
                        {"nodes": {"not": "a-list"}}
                    )
                }
            )
            == []
        )
        root_attrs = {
            manager_workspace._WORKSPACE_MANIFEST_ATTR: json.dumps(
                {
                    "nodes": [
                        None,
                        {"uid": uid, "kind": "unknown", "path": "0"},
                        {"uid": uid, "kind": "imagetool", "path": "0"},
                    ]
                }
            )
        }
        assert controller._workspace_full_save_manifest_entries(root_attrs) == [
            (uid, "imagetool", "0/imagetool")
        ]

        manager._workspace_state.dirty_added.add("missing")
        manager._workspace_state.dirty_data.add(uid)
        assert controller._workspace_full_save_dirty_payload_uids() == {uid}
        manager._workspace_state.dirty_added.clear()
        manager._workspace_state.dirty_data.clear()

        assert controller._workspace_full_save_source_identities() is None
        missing_workspace = tmp_path / "missing.itws"
        manager._workspace_state.path = missing_workspace
        assert controller._workspace_full_save_source_identities() is None

        workspace = tmp_path / "identity.itws"
        manifest = {
            "nodes": [
                "bad-entry",
                {"uid": uid, "kind": "tool", "path": 1},
                {"uid": uid, "kind": "imagetool", "path": "0"},
            ]
        }
        with h5py.File(workspace, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = (
                manager_workspace._current_workspace_schema_version()
            )
            h5_file.attrs[manager_workspace._WORKSPACE_MANIFEST_ATTR] = json.dumps(
                manifest
            )
        manager._workspace_state.path = workspace
        source = controller._workspace_full_save_source_identities()
        assert source is not None
        assert source[0] == workspace
        assert source[1] == {(uid, "imagetool"): "0/imagetool"}

        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_root_attrs_h5py",
            lambda _path: (_ for _ in ()).throw(OSError("boom")),
        )
        assert controller._workspace_full_save_source_identities() is None


def test_pending_workspace_imagetool_attrs_optional_metadata(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            root,
            show=False,
            watched_var=("watched_data", "watched-uid"),
            watched_workspace_link_id="workspace-link",
            watched_source_label="source label",
            watched_source_uid="source-uid",
            watched_connected=False,
            source_input_ndim=2,
            source_spec=provenance.full_data(),
            source_auto_update=True,
            source_state="stale",
            note="remember this",
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        wrapper.name = "pending-name"
        wrapper.set_pending_workspace_memory_payload(
            tmp_path / "source.itws",
            "0/imagetool",
            payload_attrs={
                "manager_node_note": "old note",
                "manager_node_provenance_spec": "old provenance",
                "manager_node_watched_workspace_link_id": "old link",
            },
        )

        attrs = manager._workspace_controller._pending_workspace_imagetool_attrs(
            wrapper
        )
        assert attrs["itool_name"] == "pending-name"
        assert attrs["manager_node_note"] == "remember this"
        assert attrs["manager_node_kind"] == "imagetool"
        assert attrs["manager_node_source_input_ndim"] == 2
        assert attrs["manager_node_watched_varname"] == "watched_data"
        assert attrs["manager_node_watched_uid"] == "watched-uid"
        assert attrs["manager_node_watched_workspace_link_id"] == "workspace-link"
        assert attrs["manager_node_watched_source_label"] == "source label"
        assert attrs["manager_node_watched_source_uid"] == "source-uid"
        assert attrs["manager_node_watched_connected"] is False
        assert attrs["manager_node_live_source_spec"]
        assert attrs["manager_node_source_state"] == "stale"
        assert attrs["manager_node_source_auto_update"] is True

        wrapper.note = ""
        wrapper.set_watched_binding("watched_data", "watched-uid")
        wrapper.set_source_input_ndim(None)
        wrapper._source_spec = None
        attrs = manager._workspace_controller._pending_workspace_imagetool_attrs(
            wrapper
        )
        assert "manager_node_note" not in attrs
        assert "manager_node_source_input_ndim" not in attrs
        assert "manager_node_watched_workspace_link_id" not in attrs
        assert "manager_node_watched_source_label" not in attrs
        assert "manager_node_watched_source_uid" not in attrs
        assert "manager_node_live_source_spec" not in attrs
        assert "manager_node_source_state" not in attrs
        assert "manager_node_source_auto_update" not in attrs


def test_wrapper_pending_workspace_branch_helpers(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        wrapper = manager._tool_graph.root_wrappers[0]

        assert wrapper.pending_workspace_payload_attrs is None
        wrapper.update_pending_workspace_payload_attrs({"itool_state": "{}"})
        assert wrapper.pending_workspace_payload_attrs is None
        assert wrapper.pending_workspace_preview_image() is None

        wrapper.set_pending_workspace_memory_payload(
            tmp_path / "source.itws", "0/imagetool"
        )
        assert wrapper._metadata_data() is None
        assert wrapper._pending_workspace_load_source_details() is None

        for raw_state in (b"\xff", "{not-json", json.dumps({"file_path": 1})):
            wrapper.update_pending_workspace_payload_attrs({"itool_state": raw_state})
            assert wrapper._pending_workspace_load_source_details() is None

        assert wrapper._load_func_from_serialized_state("bad") is None
        assert wrapper._load_func_from_serialized_state(["bad", {}, None]) is None
        assert (
            wrapper._load_func_from_serialized_state(["math:missing", {}, None]) is None
        )
        assert wrapper._load_func_from_serialized_state(["math:pi", {}, None]) is None
        assert (
            wrapper._load_func_from_serialized_state(["math:sqrt", {}, None])[
                0
            ].__name__
            == "sqrt"
        )
        assert wrapper._load_func_from_serialized_state(["da30", {}, None]) == (
            "da30",
            {},
            None,
        )

        original_tool = wrapper._imagetool
        wrapper._imagetool = None
        try:
            monkeypatch.setattr(
                wrapper,
                "_load_source_details",
                lambda: types.SimpleNamespace(load_code="data = 1"),
            )
            assert wrapper.load_source_code() == "data = 1"
            assert wrapper.load_source_code(assign="renamed") == "renamed = 1"
            with pytest.raises(ValueError, match="valid Python identifier"):
                wrapper.load_source_code(assign="bad name")
            monkeypatch.setattr(
                wrapper,
                "_load_source_details",
                lambda: types.SimpleNamespace(load_code="data ="),
            )
            assert wrapper.load_source_code(assign="renamed") is None
        finally:
            wrapper._imagetool = original_tool

        monkeypatch.setattr(
            wrapper, "materialize_pending_workspace_payload", lambda: False
        )
        with pytest.raises(ValueError, match="saved data"):
            wrapper.persistence_view()
        with pytest.raises(ValueError, match="saved data"):
            wrapper.current_source_data()
        wrapper.show()

        manager._workspace_state.loading_depth = 1
        try:
            wrapper._handle_source_data_replaced(data)
        finally:
            manager._workspace_state.loading_depth = 0
        assert wrapper.pending_workspace_memory_payload is not None

        assert wrapper.load_source_code() is None
        assert wrapper.persistence_data_backing() == ("memory", ())
        wrapper._handle_source_data_replaced(data)
        assert wrapper.pending_workspace_memory_payload is None

        empty_node = manager_wrapper._ManagedWindowNode(
            manager,
            "empty-node",
            None,
            None,
            window_kind="imagetool",
            created_time="2026-01-02T03:04:05+00:00",
        )
        try:
            assert "Added" in empty_node.info_text
            assert empty_node.persistence_data_backing() == (None, ())
        finally:
            empty_node.deleteLater()


def test_pending_workspace_actions_and_color_branches(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(offset, offset + 16, dtype=float).reshape(4, 4),
                dims=("x", "y"),
            )
            tool = itool(data, manager=False, execute=False)
            assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(tool, show=False)

        manager.link_imagetools(0, 1, link_colors=True)
        link_key = manager._tool_graph.root_wrappers[0].workspace_link_key
        assert link_key is not None
        linked_color = manager.color_for_linker(manager._link_registry.linkers[0])
        assert manager.color_for_workspace_link_key(link_key) == linked_color
        assert (
            manager.color_for_workspace_link_key("unknown-link-key")
            == (manager_mainwindow._LINKER_COLORS[0])
        )

        select_tools(manager, [0, 1])
        manager._unlink_selected_from_action()
        assert not manager._tool_graph.root_wrappers[0].workspace_linked
        assert not manager._tool_graph.root_wrappers[1].workspace_linked

        fake_node = types.SimpleNamespace(is_imagetool=False)
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_node_for_target", lambda _target: fake_node)
            patch.setattr(manager, "_selected_imagetool_targets", lambda: ("bad",))
            patch.setattr(manager, "_child_node", lambda _uid: fake_node)
            with pytest.raises(KeyError, match="not an ImageTool"):
                manager.link_imagetools("bad", "also_bad")
            with pytest.raises(KeyError, match="not an ImageTool"):
                manager.unlink_selected(deselect=False)
            with pytest.raises(KeyError, match="not an ImageTool"):
                manager.promote_child_imagetool("bad")
            with pytest.raises(KeyError, match="not an ImageTool"):
                manager.get_imagetool("bad")

        fake_pending_node = types.SimpleNamespace(
            is_imagetool=True,
            materialize_pending_workspace_payload=lambda: False,
        )
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_child_node", lambda _uid: fake_pending_node)
            with pytest.raises(RuntimeError, match="saved data"):
                manager.promote_child_imagetool("pending")
            patch.setattr(
                manager, "_node_for_target", lambda _target: fake_pending_node
            )
            with pytest.raises(ValueError, match="saved data"):
                manager.get_imagetool("pending")


def test_pending_workspace_reload_reason_branches(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        controller = manager._lineage_controller
        monkeypatch.setattr(provenance, "can_reload_without_trust", lambda _spec: False)
        monkeypatch.setattr(provenance, "has_file_load_source", lambda _spec: False)

        file_spec = types.SimpleNamespace(kind="file")
        monkeypatch.setattr(
            controller,
            "_file_load_source_unavailable_reason",
            lambda _spec, _label: "missing file",
        )
        assert (
            controller._pending_imagetool_reload_unavailable_reason(
                types.SimpleNamespace(
                    provenance_spec=file_spec,
                    _load_source_details=lambda: None,
                )
            )
            == "missing file"
        )

        script_spec = types.SimpleNamespace(kind="script")
        monkeypatch.setattr(
            provenance, "script_provenance_requires_trust", lambda _spec: True
        )
        trust_reason = controller._pending_imagetool_reload_unavailable_reason(
            types.SimpleNamespace(
                provenance_spec=script_spec,
                _load_source_details=lambda: None,
            )
        )
        assert trust_reason is not None
        assert "trust confirmation" in trust_reason

        monkeypatch.setattr(
            provenance, "script_provenance_requires_trust", lambda _spec: False
        )
        replay_reason = controller._pending_imagetool_reload_unavailable_reason(
            types.SimpleNamespace(
                provenance_spec=script_spec,
                _load_source_details=lambda: None,
            )
        )
        assert replay_reason is not None
        assert "cannot be reloaded automatically" in replay_reason

        missing_path = tmp_path / "missing.h5"
        missing_reason = controller._pending_imagetool_reload_unavailable_reason(
            types.SimpleNamespace(
                provenance_spec=None,
                _load_source_details=lambda: types.SimpleNamespace(
                    path=missing_path,
                    load_code=None,
                ),
            )
        )
        assert missing_reason is not None
        assert str(missing_path) in missing_reason

        existing_path = tmp_path / "scan.h5"
        existing_path.write_bytes(b"")
        assert (
            controller._pending_imagetool_reload_unavailable_reason(
                types.SimpleNamespace(
                    provenance_spec=None,
                    _load_source_details=lambda: types.SimpleNamespace(
                        path=existing_path,
                        load_code="data = load()",
                    ),
                )
            )
            is None
        )
        missing_loader_reason = controller._pending_imagetool_reload_unavailable_reason(
            types.SimpleNamespace(
                provenance_spec=None,
                _load_source_details=lambda: types.SimpleNamespace(
                    path=existing_path,
                    load_code=None,
                ),
            )
        )
        assert missing_loader_reason is not None
        assert "loader information" in missing_loader_reason


def test_pending_workspace_details_preview_and_viewer_restore_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(tool, show=False)

        updates: list[str] = []
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_selected_imagetool_targets", lambda: (0,))
            patch.setattr(manager, "_selected_tool_uids", lambda: ())
            patch.setattr(
                manager, "_update_info", lambda *, uid=None: updates.append(uid)
            )
            manager._details_panel._load_selected_preview_data()
            assert updates == []

            wrapper = manager._tool_graph.root_wrappers[0]
            wrapper.set_pending_workspace_memory_payload(tmp_path / "source.itws", "0")
            patch.setattr(
                wrapper,
                "materialize_pending_workspace_payload",
                lambda: True,
            )
            manager._details_panel._load_selected_preview_data()
            assert updates == [wrapper.uid]

            patch.setattr(manager, "_selected_imagetool_targets", lambda: (0, 1))
            manager._details_panel._load_selected_preview_data()
            assert updates == [wrapper.uid]

        slicer_area = tool.slicer_area
        fallback_sizes = [splitter.sizes() for splitter in slicer_area._splitters]
        assert slicer_area._normalize_splitter_sizes([]) == fallback_sizes
        slicer_area._pending_splitter_sizes = None
        slicer_area._pending_splitter_restore_queued = True
        slicer_area._apply_pending_splitter_restore()
        assert not slicer_area._pending_splitter_restore_queued
        slicer_area._deferred_show_restore_queued = True
        tool.hide()
        slicer_area._flush_deferred_show_restore()
        assert not slicer_area._deferred_show_restore_queued


def test_pending_workspace_provenance_edit_materialization_failures(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        node = types.SimpleNamespace(
            materialize_pending_workspace_payload=lambda: False
        )
        operation = typing.cast(
            "provenance.ToolProvenanceOperation", types.SimpleNamespace()
        )
        spec = provenance.full_data()
        data = xr.DataArray(np.arange(4), dims=("x",))

        with pytest.raises(RuntimeError, match="saved data"):
            manager._provenance_edit_controller._edit_active_filter(
                typing.cast("manager_wrapper._ManagedWindowNode", node),
                operation,
                manager_provenance_edit.dialogs.DataFilterDialog,
            )
        with pytest.raises(RuntimeError, match="saved data"):
            manager._provenance_edit_controller._replace_node_data(
                typing.cast("manager_wrapper._ManagedWindowNode", node),
                "display",
                data,
                spec,
                None,
            )


def test_workspace_save_worker_start_and_finish_error_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class Snapshot:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller

        start_errors: list[str] = []
        snapshot = Snapshot()
        monkeypatch.setattr(
            manager_workspace_io.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: None),
        )
        assert not controller._start_workspace_save_worker(
            tmp_path / "none.itws",
            typing.cast("manager_workspace._WorkspaceSaveSnapshot", snapshot),
            on_finished=lambda *_args: None,
            on_start_error=lambda: start_errors.append("none"),
        )
        assert snapshot.closed
        assert start_errors == ["none"]

        class RaisingPool:
            def start(self, _worker) -> None:
                raise RuntimeError("cannot start")

        snapshot = Snapshot()
        monkeypatch.setattr(
            manager_workspace_io.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: RaisingPool()),
        )
        assert not controller._start_workspace_save_worker(
            tmp_path / "raise.itws",
            typing.cast("manager_workspace._WorkspaceSaveSnapshot", snapshot),
            on_finished=lambda *_args: None,
            on_start_error=lambda: start_errors.append("raise"),
        )
        assert snapshot.closed
        assert start_errors == ["none", "raise"]
        assert not manager._workspace_state.save_in_progress
        assert controller._background_save_worker is None
        assert controller._background_save_receiver is None

        class RecordingPool:
            def __init__(self) -> None:
                self.worker = None

            def start(self, worker) -> None:
                self.worker = worker

        errors: list[tuple[str, str]] = []
        pool = RecordingPool()
        monkeypatch.setattr(
            manager_workspace_io.QtCore.QThreadPool,
            "globalInstance",
            staticmethod(lambda: pool),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        assert controller._start_workspace_save_worker(
            tmp_path / "finish.itws",
            typing.cast("manager_workspace._WorkspaceSaveSnapshot", Snapshot()),
            on_finished=lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert pool.worker is not None
        pool.worker.signals.finished.emit(True, 0.1, "")
        assert errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]
        assert not manager._workspace_state.save_in_progress


def test_background_workspace_save_finish_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        workspace_path = tmp_path / "queued.itws"
        manager._workspace_state.path = workspace_path
        manager._workspace_state.dirty_state.add("uid")
        controller._background_save_requested = True

        queued_callbacks: list[Callable[[], None]] = []
        finished: list[bool] = []
        monkeypatch.setattr(
            controller,
            "_finish_workspace_save_result",
            lambda **_kwargs: True,
        )
        monkeypatch.setattr(
            controller,
            "_current_workspace_document_path",
            lambda: workspace_path,
        )
        monkeypatch.setattr(
            manager_workspace_io.QtCore.QTimer,
            "singleShot",
            staticmethod(lambda _delay, callback: queued_callbacks.append(callback)),
        )
        controller._finish_background_workspace_save(
            document_id=manager._workspace_state.document_id,
            workspace_path=workspace_path,
            old_workspace_path=None,
            backing_snapshot={},
            snapshot=typing.cast(
                "manager_workspace._WorkspaceSaveSnapshot", types.SimpleNamespace()
            ),
            ok=True,
            worker_elapsed=0.1,
            error_text="",
            origin=None,
            snapshot_elapsed=0.0,
            started_at=time.perf_counter(),
            restore_focus=False,
            on_finished=finished.append,
        )
        assert len(queued_callbacks) == 1
        assert finished == [True]
        assert not controller._background_save_requested

        errors: list[tuple[str, str]] = []
        finished.clear()
        monkeypatch.setattr(
            controller,
            "_finish_workspace_save_result",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        controller._finish_background_workspace_save(
            document_id=manager._workspace_state.document_id,
            workspace_path=workspace_path,
            old_workspace_path=None,
            backing_snapshot={},
            snapshot=typing.cast(
                "manager_workspace._WorkspaceSaveSnapshot", types.SimpleNamespace()
            ),
            ok=True,
            worker_elapsed=0.1,
            error_text="",
            origin=None,
            snapshot_elapsed=0.0,
            started_at=time.perf_counter(),
            restore_focus=False,
            on_finished=finished.append,
        )
        assert finished == [False]
        assert errors == [
            (
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
        ]


def test_workspace_save_and_save_as_error_continuation_branches(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    def snapshot(generation: int = 0) -> manager_workspace._WorkspaceSaveSnapshot:
        return manager_workspace._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs={},
            delta_save_count=0,
            estimated_obsolete_bytes=0,
            replacement_delta_count=0,
            full_tree=xr.DataTree(),
        )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid
        controller = manager._workspace_controller

        finished: list[bool] = []
        manager._workspace_state.save_in_progress = True
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]
        manager._workspace_state.save_in_progress = False

        warnings: list[bool] = []
        monkeypatch.setattr(
            manager, "_workspace_save_dialog", lambda **_kwargs: tmp_path / "bad.txt"
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_show_itws_workspace_warning",
            lambda _parent: warnings.append(True),
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert warnings == [True]
        assert finished == [False]

        errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        monkeypatch.setattr(
            manager, "_workspace_save_dialog", lambda **_kwargs: tmp_path / "save.itws"
        )
        monkeypatch.setattr(
            controller,
            "_workspace_full_save_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]
        assert errors[-1] == (
            "Error while saving workspace",
            "An error occurred while saving the workspace file.",
        )

        monkeypatch.setattr(
            controller,
            "_workspace_full_save_snapshot",
            lambda generation, **_kwargs: snapshot(generation),
        )
        worker_errors: list[str] = []
        monkeypatch.setattr(
            manager, "_show_workspace_save_worker_error", worker_errors.append
        )

        def _fail_worker(_fname, _snapshot, *, on_finished, on_start_error=None):
            del on_start_error
            on_finished(False, 0.1, "write failed")
            return True

        monkeypatch.setattr(controller, "_start_workspace_save_worker", _fail_worker)
        finished.clear()
        assert controller.save_as(native=False, on_finished=finished.append)
        assert worker_errors == ["write failed"]
        assert finished == [False]

        manager._mark_workspace_clean()

        def _dirty_during_worker(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ):
            del on_start_error
            manager._mark_node_state_dirty(uid)
            on_finished(True, 0.1, "")
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _dirty_during_worker
        )
        finished.clear()
        assert controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]

        def _start_error_worker(_fname, _snapshot, *, on_finished, on_start_error=None):
            del on_finished
            if on_start_error is not None:
                on_start_error()
            return False

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _start_error_worker
        )
        finished.clear()
        assert not controller.save_as(native=False, on_finished=finished.append)
        assert finished == [False]

        manager._workspace_state.path = tmp_path / "current.itws"
        monkeypatch.setattr(
            controller, "_workspace_save_snapshot", lambda _path: snapshot()
        )
        finished.clear()
        assert not controller.save(on_finished=finished.append, restore_focus=False)
        assert finished == [False]


def test_workspace_save_completion_ignores_inactive_document(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        workspace_path = tmp_path / "stale-save.itws"
        manager._workspace_state.path = workspace_path.resolve()
        manager._workspace_state.delta_save_count = 7
        manager._workspace_state.mark_layout_dirty()

        snapshot = manager_workspace._WorkspaceSaveSnapshot(
            generation=manager._workspace_state.dirty_generation,
            root_attrs={},
            delta_save_count=99,
        )
        monkeypatch.setattr(
            controller, "_workspace_save_snapshot", lambda _path: snapshot
        )
        recorded_recent: list[pathlib.Path] = []
        monkeypatch.setattr(
            controller, "_record_recent_workspace", recorded_recent.append
        )

        def _finish_after_document_change(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ) -> bool:
            del on_start_error
            manager._workspace_state.advance_document_identity()
            manager._workspace_state.path = tmp_path / "replacement.itws"
            on_finished(True, 0.1, "")
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _finish_after_document_change
        )

        finished: list[bool] = []
        assert controller.save(on_finished=finished.append, restore_focus=False)

        assert finished == [False]
        assert manager._workspace_state.delta_save_count == 7
        assert manager.is_workspace_modified
        assert recorded_recent == []


def test_workspace_save_as_completion_ignores_inactive_document(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller
        target = tmp_path / "stale-save-as.itws"
        data = xr.DataArray(np.arange(9, dtype=float).reshape(3, 3), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        manager._workspace_state.mark_layout_dirty()

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: target)
        monkeypatch.setattr(
            controller,
            "_workspace_full_save_snapshot",
            lambda generation, **_kwargs: manager_workspace._WorkspaceSaveSnapshot(
                generation=generation,
                root_attrs={},
                delta_save_count=0,
            ),
        )

        def _finish_after_document_change(
            _fname, _snapshot, *, on_finished, on_start_error=None
        ) -> bool:
            del on_start_error
            manager._workspace_state.advance_document_identity()
            on_finished(True, 0.1, "")
            return True

        monkeypatch.setattr(
            controller, "_start_workspace_save_worker", _finish_after_document_change
        )

        finished: list[bool] = []
        assert controller.save_as(native=False, on_finished=finished.append)

        assert finished == [False]
        assert manager.workspace_path is None
        assert manager.is_workspace_modified


def test_manager_manifest_first_full_save_copies_clean_payloads(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(25, dtype=float).reshape(5, 5) + offset,
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-clean.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        monkeypatch.setattr(
            manager,
            "_serialize_workspace_node",
            lambda *_args, **_kwargs: pytest.fail(
                "Clean full save should copy payload groups"
            ),
        )

        manager._save_workspace_document(fname, force_full=True)

        with h5py.File(fname, "r") as h5_file:
            assert "0/imagetool" in h5_file
            assert "1/imagetool" in h5_file


def test_manager_manifest_first_full_save_serializes_missing_source_group(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(25, dtype=float).reshape(5, 5) + offset,
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-missing-source.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        with h5py.File(fname, "a") as h5_file:
            del h5_file["0/imagetool"]

        original_serialize = manager._serialize_workspace_node
        serialized_paths: list[str] = []

        def _record_serialize(*args, **kwargs) -> None:
            serialized_paths.append(args[2])
            original_serialize(*args, **kwargs)

        monkeypatch.setattr(manager, "_serialize_workspace_node", _record_serialize)

        manager._save_workspace_document(fname, force_full=True)

        assert serialized_paths == ["0"]
        with h5py.File(fname, "r") as h5_file:
            assert "0/imagetool" in h5_file
            assert "1/imagetool" in h5_file


def test_manager_manifest_first_full_save_preserves_clean_mismatched_compression(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_compression = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

            fname = tmp_path / "manifest-copy-mismatched-compression.itws"
            manager._save_workspace_document(fname, force_full=True)
            manager._adopt_workspace_path(fname)
            manager._mark_workspace_clean()
            with h5py.File(fname, "r") as h5_file:
                data_ds = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
                assert _hdf5_blosc2_level_codec(data_ds) is None

            writes: list[
                tuple[
                    str | os.PathLike[str] | None,
                    tuple[tuple[str, str, dict[str, typing.Any] | None], ...],
                ]
            ] = []
            original_write = manager_workspace._write_full_workspace_tree_file

            def _record_write(*args, **kwargs) -> None:
                writes.append(
                    (kwargs.get("copy_source"), tuple(kwargs.get("copy_groups", ())))
                )
                original_write(*args, **kwargs)

            erlab.interactive.options["io/workspace/compression"] = "zstd1"
            monkeypatch.setattr(
                manager,
                "_serialize_workspace_node",
                lambda *_args, **_kwargs: pytest.fail(
                    "Clean mismatched payload should be copied"
                ),
            )
            monkeypatch.setattr(
                manager_workspace,
                "_write_full_workspace_tree_file",
                _record_write,
            )

            manager._save_workspace_document(fname, force_full=True)

            assert writes[-1] == (
                str(fname),
                (("0/imagetool", "0/imagetool", None),),
            )
            with h5py.File(fname, "r") as h5_file:
                data_ds = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
                assert _hdf5_blosc2_level_codec(data_ds) is None
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


@pytest.mark.parametrize("force_fallback", [False, True])
def test_manager_full_save_rewrites_pending_dirty_state_on_compression_mismatch(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    force_fallback: bool,
) -> None:
    old_compression = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()

            fname = tmp_path / f"pending-compression-{force_fallback}.itws"
            manager._save_workspace_document(fname, force_full=True)
            with h5py.File(fname, "r") as h5_file:
                data_ds = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
                assert _hdf5_blosc2_level_codec(data_ds) is None
            assert manager._load_workspace_file(
                fname, replace=True, associate=True, mark_dirty=False, select=False
            )

            wrapper = manager._tool_graph.root_wrappers[0]
            assert wrapper.pending_workspace_memory_payload is not None
            wrapper.name = "renamed pending"
            assert wrapper.uid in manager._workspace_state.dirty_state
            assert wrapper.uid not in manager._workspace_state.dirty_data

            if force_fallback:
                monkeypatch.setattr(
                    manager._workspace_controller,
                    "_workspace_full_save_manifest_first_snapshot",
                    lambda *_args, **_kwargs: None,
                )

            erlab.interactive.options["io/workspace/compression"] = "zstd1"
            manager._save_workspace_document(
                fname,
                force_full=True,
                require_matching_compression=True,
            )

            with h5py.File(fname, "r") as h5_file:
                group = h5_file["0/imagetool"]
                data_ds = group[manager_workspace_io._ITOOL_DATA_NAME]
                assert _hdf5_blosc2_level_codec(data_ds) is not None
                assert group.attrs["itool_name"] == "renamed pending"
                np.testing.assert_array_equal(data_ds[...], data.values)
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_fallback_save_materializes_pending_without_wait_dialog(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    old_compression = erlab.interactive.options["io/workspace/compression"]
    try:
        erlab.interactive.options["io/workspace/compression"] = "none"
        with manager_context() as manager:
            qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
            data = xr.DataArray(
                np.arange(512 * 512, dtype=np.float64).reshape(512, 512),
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()

            fname = tmp_path / "fallback-save-pending-no-wait.itws"
            manager._save_workspace_document(fname, force_full=True)
            assert manager._load_workspace_file(
                fname, replace=True, associate=True, mark_dirty=False, select=False
            )

            wrapper = manager._tool_graph.root_wrappers[0]
            assert wrapper.pending_workspace_memory_payload is not None
            wrapper.name = "renamed pending"
            monkeypatch.setattr(
                manager._workspace_controller,
                "_workspace_full_save_manifest_first_snapshot",
                lambda *_args, **_kwargs: None,
            )
            messages: list[str] = []

            @contextlib.contextmanager
            def _record_wait_dialog(_parent, message):
                messages.append(message)
                yield types.SimpleNamespace(
                    set_message=lambda updated: messages.append(updated)
                )

            monkeypatch.setattr(
                erlab.interactive.utils, "wait_dialog", _record_wait_dialog
            )
            erlab.interactive.options["io/workspace/compression"] = "zstd1"

            manager._save_workspace_document(
                fname,
                force_full=True,
                require_matching_compression=True,
            )

            assert messages == []
            assert wrapper.pending_workspace_memory_payload is None
            with h5py.File(fname, "r") as h5_file:
                assert h5_file["0/imagetool"].attrs["itool_name"] == "renamed pending"
                np.testing.assert_array_equal(
                    h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME][...],
                    data.values,
                )
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_manager_manifest_first_full_save_serializes_only_dirty_data(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0, 100):
            data = xr.DataArray(
                np.arange(25, dtype=float).reshape(5, 5) + offset,
                dims=("x", "y"),
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-dirty.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        dirty_uid = manager._tool_graph.root_wrappers[0].uid
        updated = xr.DataArray(np.full((5, 5), 42.0), dims=("x", "y"))
        manager.get_imagetool(0).slicer_area.replace_source_data(
            updated, auto_compute=False
        )
        manager._mark_node_data_dirty(dirty_uid)

        original_serialize = manager._serialize_workspace_node
        serialized_paths: list[str] = []

        def _record_serialize(*args, **kwargs) -> None:
            serialized_paths.append(args[2])
            original_serialize(*args, **kwargs)

        original_write = manager_workspace._write_full_workspace_tree_file
        writes: list[tuple[tuple[str, str, dict[str, typing.Any] | None], ...]] = []

        def _record_write(*args, **kwargs) -> None:
            writes.append(tuple(kwargs.get("copy_groups", ())))
            original_write(*args, **kwargs)

        monkeypatch.setattr(manager, "_serialize_workspace_node", _record_serialize)
        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            _record_write,
        )

        manager._save_workspace_document(fname, force_full=True)

        assert serialized_paths == ["0"]
        assert ("1/imagetool", "1/imagetool", None) in writes[-1]
        assert all(group[1] != "0/imagetool" for group in writes[-1])


def test_manager_manifest_first_full_save_rewrites_dirty_state_attrs(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25, dtype=float).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-state.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()
        manager._tool_graph.root_wrappers[0].name = "renamed"

        original_serialize = manager._serialize_workspace_node
        serialized_paths: list[str] = []

        def _record_serialize(*args, **kwargs) -> None:
            serialized_paths.append(args[2])
            original_serialize(*args, **kwargs)

        original_write = manager_workspace._write_full_workspace_tree_file
        writes: list[tuple[tuple[str, str, dict[str, typing.Any] | None], ...]] = []

        def _record_write(*args, **kwargs) -> None:
            writes.append(tuple(kwargs.get("copy_groups", ())))
            original_write(*args, **kwargs)

        monkeypatch.setattr(manager, "_serialize_workspace_node", _record_serialize)
        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            _record_write,
        )

        manager._save_workspace_document(fname, force_full=True)

        assert serialized_paths == ["0"]
        assert all(group[1] != "0/imagetool" for group in writes[-1])
        with h5py.File(fname, "r") as h5_file:
            assert h5_file["0/imagetool"].attrs["itool_title"] == "renamed"


def test_manager_manifest_first_save_as_copies_clean_payloads(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25, dtype=float).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        source = tmp_path / "manifest-copy-source.itws"
        target = tmp_path / "manifest-copy-target.itws"
        manager._save_workspace_document(source, force_full=True)
        manager._adopt_workspace_path(source)
        manager._mark_workspace_clean()

        writes: list[
            tuple[
                str | os.PathLike[str] | None,
                tuple[tuple[str, str, dict[str, typing.Any] | None], ...],
            ]
        ] = []
        original_write = manager_workspace._write_full_workspace_tree_file

        def _record_write(*args, **kwargs) -> None:
            writes.append(
                (kwargs.get("copy_source"), tuple(kwargs.get("copy_groups", ())))
            )
            original_write(*args, **kwargs)

        monkeypatch.setattr(
            manager,
            "_serialize_workspace_node",
            lambda *_args, **_kwargs: pytest.fail(
                "Clean Save As should copy payload groups"
            ),
        )
        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: target)
        monkeypatch.setattr(
            manager_workspace,
            "_write_full_workspace_tree_file",
            _record_write,
        )

        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)

        assert manager.workspace_path == str(target.resolve())
        assert writes[-1] == (str(source), (("0/imagetool", "0/imagetool", None),))
        with h5py.File(target, "r") as h5_file:
            assert "0/imagetool" in h5_file


def test_manager_manifest_first_full_save_falls_back_without_usable_source(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25, dtype=float).reshape(5, 5), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "manifest-copy-fallback.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_root_attrs_h5py",
            lambda _path: {"imagetool_workspace_schema_version": 1},
        )

        def _fail_materializing_fallback() -> typing.NoReturn:
            pytest.fail("fallback full save should not call _to_datatree()")

        monkeypatch.setattr(manager, "_to_datatree", _fail_materializing_fallback)

        snapshot = manager._workspace_controller._workspace_full_save_snapshot(
            1, fname=fname
        )
        try:
            assert snapshot.full_tree is not None
            assert snapshot.copy_source is None
            assert snapshot.copy_groups == ()
            assert snapshot.copy_group_sources == ()
            ds = typing.cast(
                "xr.DataTree", snapshot.full_tree["0/imagetool"]
            ).to_dataset(inherit=False)
            assert manager_workspace_io._ITOOL_DATA_NAME in ds
        finally:
            snapshot.close()


def test_manager_associate_loaded_legacy_workspace_resets_repack_state(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        converted = tmp_path / "converted.itws"
        manager._workspace_state.path = converted.resolve()

        monkeypatch.setattr(
            manager_workspace,
            "_workspace_schema_requires_conversion",
            lambda _schema_version: True,
        )
        monkeypatch.setattr(
            manager,
            "_save_legacy_workspace_as_v4",
            lambda *_, **__: (str(converted), None),
        )

        manager._associate_loaded_workspace_file(
            tmp_path / "legacy.itws",
            1,
            delta_save_count=5,
            estimated_obsolete_bytes=100,
            replacement_delta_count=2,
            repack_estimate_known=False,
            rebind_data=False,
        )

        assert manager._workspace_state.path == converted
        assert manager._workspace_state.delta_save_count == 0
        assert manager._workspace_state.estimated_obsolete_bytes == 0
        assert manager._workspace_state.replacement_delta_count == 0
        assert manager._workspace_state.repack_estimate_known


def test_manager_workspace_file_repack_snapshot_guards(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller

        manager._workspace_state.path = None
        assert controller._workspace_file_repack_snapshot(1) is None

        manager._workspace_state.path = tmp_path / "network.itws"
        monkeypatch.setattr(
            manager_workspace,
            "_workspace_path_is_likely_network_path",
            lambda _path: True,
        )
        assert controller._workspace_file_repack_snapshot(1) is None

        monkeypatch.setattr(
            manager_workspace,
            "_workspace_path_is_likely_network_path",
            lambda _path: False,
        )
        monkeypatch.setattr(
            manager_workspace,
            "_workspace_file_repack_payload",
            lambda _path: (_ for _ in ()).throw(RuntimeError("repack failed")),
        )
        assert controller._workspace_file_repack_snapshot(1) is None


def test_manager_workspace_shutdown_repack_gate_edges(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller

        manager._workspace_state.path = None
        assert not controller._workspace_should_repack_before_shutdown()

        workspace_path = tmp_path / "workspace.itws"
        manager._workspace_state.path = workspace_path
        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=0,
            replacement_delta_count=0,
            known=False,
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_workspace_obsolete_estimate",
            lambda _path: (_ for _ in ()).throw(RuntimeError("estimate failed")),
        )
        assert not controller._workspace_should_repack_before_shutdown()

        manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=100,
            replacement_delta_count=1,
        )
        assert not controller._workspace_should_repack_before_shutdown()

        workspace_path.touch()
        assert not controller._workspace_should_repack_before_shutdown()


def test_workspace_remote_incremental_option_allows_delta_save(
    monkeypatch,
    tmp_path,
) -> None:
    options = erlab.interactive.options
    incremental_name = "io/workspace/incremental_save_on_remote"
    use_incremental_name = "io/workspace/use_incremental"
    old_remote_value = options[incremental_name]
    old_incremental_value = options[use_incremental_name]
    fname = tmp_path / "remote-incremental.itws"
    fname.touch()
    monkeypatch.setattr(
        manager_workspace, "_workspace_path_is_high_risk", lambda *_args: True
    )
    options[incremental_name] = True
    options[use_incremental_name] = True
    try:
        assert not manager_workspace._workspace_requires_full_save(
            fname,
            needs_full_save=False,
            schema_version=manager_workspace._current_workspace_schema_version(),
            structure_modified=False,
            has_dirty_added=False,
            has_dirty_removed=False,
        )
    finally:
        options[incremental_name] = old_remote_value
        options[use_incremental_name] = old_incremental_value


def test_manager_close_suppresses_child_visibility_dirty(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        qtbot.wait_until(root.isVisible)

        fname = tmp_path / "quit-clean.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._drain_workspace_deferred_events()
        manager._mark_workspace_clean()
        assert not manager.is_workspace_modified

        manager._workspace_state.closing_document = True
        root.hide()
        manager._workspace_state.closing_document = False
        manager._drain_workspace_deferred_events()

        assert not manager.is_workspace_modified
        assert not root.isWindowModified()


def test_manager_application_quit_filter_routes_quit_to_manager(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        calls: list[str] = []
        monkeypatch.setattr(
            manager,
            "close",
            lambda: calls.append("close") or False,
        )

        event = QtCore.QEvent(QtCore.QEvent.Type.Quit)
        assert manager._application_quit_filter is not None
        assert manager._application_quit_filter.eventFilter(None, event)
        assert calls == ["close"]


def test_manager_workspace_dirty_marker_not_saved_in_titles(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)), dims=["x", "y"], name="source"
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root_uid = manager._tool_graph.root_wrappers[0].uid
        tool = DerivativeTool(data)
        tool_uid = manager.add_childtool(tool, 0, show=False)

        root.setWindowTitle("stale root title[*]")
        manager._tool_graph.root_wrappers[0].update_title()
        assert "stale root title" not in manager._tool_graph.root_wrappers[0].label_text
        assert "[*]" not in manager._tool_graph.root_wrappers[0].label_text

        root.setWindowTitle("stale root title[*]")
        tool.setWindowTitle("stale tool title[*]")
        manager._set_node_window_modified(root_uid, True)
        manager._set_node_window_modified(tool_uid, True)

        expect_title_placeholder = sys.platform != "darwin"
        assert ("[*]" in root.windowTitle()) is expect_title_placeholder
        assert ("[*]" in tool.windowTitle()) is expect_title_placeholder
        assert (
            root.windowTitle()
            == manager_widgets._window_title_with_modified_placeholder(
                manager._tool_graph.root_wrappers[0].label_text
            )
        )
        assert (
            tool.windowTitle()
            == manager_widgets._window_title_with_modified_placeholder(
                f"{tool.tool_name}: {tool._tool_display_name}"
            )
        )

        fname = tmp_path / "titles.itws"
        manager._save_workspace_document(fname, force_full=True)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            root_title = h5_file["0/imagetool"].attrs["itool_title"]
            tool_title = h5_file[f"0/childtools/{tool_uid}/tool"].attrs["tool_title"]

        assert "[*]" not in root_title
        assert "[*]" not in tool_title
        assert root_title == "source"


def test_manager_workspace_full_save_drops_empty_attr_name(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            attrs={"": "dropped", "note": ""},
            name="data",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "empty-attr-name.itws"
        manager._save_workspace_document(fname, force_full=True)

        assert "" in root.slicer_area._data.attrs
        with h5py.File(fname, "r") as h5_file:
            saved_attrs = h5_file["0/imagetool"][
                manager_workspace_io._ITOOL_DATA_NAME
            ].attrs
            assert "" not in list(saved_attrs)
            assert saved_attrs["note"] == ""


def test_manager_workspace_full_save_roundtrips_non_native_data_attrs(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    rich_attr = _rich_workspace_attr_value()
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            coords={
                "x": xr.DataArray(
                    np.arange(5),
                    dims=("x",),
                    attrs={"axis_config": rich_attr},
                ),
                "y": np.arange(5),
            },
            attrs={"Single Motor Scan": rich_attr},
            name="data",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        live_rich_attr = root.slicer_area._data.attrs["Single Motor Scan"]
        live_axis_attr = root.slicer_area._data.coords["x"].attrs["axis_config"]

        fname = tmp_path / "rich-data-attrs.itws"
        manager._save_workspace_document(fname, force_full=True)

        assert root.slicer_area._data.attrs["Single Motor Scan"] is live_rich_attr
        assert root.slicer_area._data.coords["x"].attrs["axis_config"] is live_axis_attr
        assert (
            manager_workspace._WORKSPACE_ENCODED_ATTRS_ATTR
            not in root.slicer_area._data.attrs
        )
        with h5py.File(fname, "r") as h5_file:
            saved_data = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert "Single Motor Scan" not in saved_data.attrs
            assert manager_workspace._WORKSPACE_ENCODED_ATTRS_ATTR in saved_data.attrs

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded = manager.get_imagetool(0).slicer_area._data
        _assert_rich_workspace_attr(loaded.attrs["Single Motor Scan"])
        _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        tree = manager_xarray.open_workspace_datatree(fname, chunks=None)
        try:
            assert manager._from_datatree(
                tree,
                replace=True,
                mark_dirty=False,
                select=False,
                workspace_file_path=fname,
            )
        finally:
            tree.close()
        loaded = manager.get_imagetool(0).slicer_area._data
        _assert_rich_workspace_attr(loaded.attrs["Single Motor Scan"])
        _assert_rich_workspace_attr(loaded.coords["x"].attrs["axis_config"])


def test_manager_workspace_save_preserves_reordered_roots(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for value in range(3):
            data = xr.DataArray(
                np.full((5, 5), value), dims=["x", "y"], name=f"data_{value}"
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        fname = tmp_path / "ordered.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        model = manager.tree_view._model
        assert model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            model.rowCount(),
            0,
            QtCore.QModelIndex(),
        )
        assert manager._tool_graph.displayed_indices == [1, 2, 0]
        assert manager.is_workspace_modified
        assert _request_workspace_save_and_wait(qtbot, manager)

        with h5py.File(fname, "r") as h5_file:
            manifest = json.loads(h5_file.attrs["imagetool_workspace_manifest"])
        assert manifest["root_order"] == [1, 2, 0]

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        loaded_order = [
            int(manager.get_imagetool(index).slicer_area._data.values[0, 0])
            for index in manager._tool_graph.displayed_indices
        ]
        assert loaded_order == [1, 2, 0]


def test_manager_workspace_child_save_shortcuts_use_background_save(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        calls: list[bool] = []

        def _fake_save(*, native: bool = True) -> bool:
            calls.append(native)
            return True

        monkeypatch.setattr(manager, "save", _fake_save)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        root_shortcuts = root.findChildren(QtWidgets.QShortcut)
        root_save = [
            shortcut
            for shortcut in root_shortcuts
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(root_save) == 1
        root_save[0].activated.emit()

        child = itool(data + 1, manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool_child(child, 0, show=False)
        child_save = [
            shortcut
            for shortcut in child.findChildren(QtWidgets.QShortcut)
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(child_save) == 1
        child_save[0].activated.emit()

        tool = DerivativeTool(data)
        manager.add_childtool(tool, 0, show=False)
        tool_save = [
            shortcut
            for shortcut in tool.findChildren(QtWidgets.QShortcut)
            if shortcut.objectName() == "managerWorkspaceSaveShortcut"
        ]
        assert len(tool_save) == 1
        tool_save[0].activated.emit()

        assert calls == [True, True, True]


def test_manager_workspace_delta_save_splits_state_and_data_writes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "delta.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)

        dataset_writes: list[str | None] = []
        original_to_netcdf = xr.Dataset.to_netcdf

        def _to_netcdf_spy(self, *args, **kwargs):
            dataset_writes.append(kwargs.get("group"))
            return original_to_netcdf(self, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", _to_netcdf_spy)

        manager.rename_imagetool(0, "state only")
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert dataset_writes == []

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement)
        assert _request_workspace_save_and_wait(qtbot, manager)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_full_save_keeps_full_persistence_for_serialized_nodes(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        fname = tmp_path / "full-persistence.itws"
        manager._save_workspace_document(fname, force_full=True)

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 1
        root.slicer_area.replace_source_data(replacement)

        original = imagetool_viewer.ImageSlicerArea.persistence_data_and_state
        calls = 0

        def _persistence_data_and_state_spy(self):
            nonlocal calls
            calls += 1
            return original(self)

        monkeypatch.setattr(
            imagetool_viewer.ImageSlicerArea,
            "persistence_data_and_state",
            _persistence_data_and_state_spy,
        )

        manager._save_workspace_document(fname, force_full=True)

        assert calls >= 1


def test_manager_workspace_full_save_preserves_in_memory_backing_after_rebind(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "memory.itws"
        backing_snapshot = manager._workspace_data_backing_snapshot()
        manager._save_workspace_document(fname, force_full=True)
        manager._rebind_workspace_backed_imagetools(
            fname,
            backing_snapshot=backing_snapshot,
            old_workspace_path=None,
        )

        saved_data = manager.get_imagetool(0).slicer_area._data
        assert manager_xarray.dataarray_is_numpy_backed(saved_data)
        assert not manager_xarray.dataarray_is_file_backed(saved_data)


def test_manager_workspace_data_backing_snapshot_includes_pending_memory(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-backing-snapshot.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "backing snapshot should not materialize hidden memory payloads"
            )

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        assert manager._workspace_data_backing_snapshot()[wrapper.uid] == ("memory", ())
        assert wrapper.pending_workspace_memory_payload is not None


def test_manager_workspace_file_backed_data_can_load_into_memory(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    source_file = tmp_path / "source.itws"
    source = xr.DataArray(
        np.arange(25, dtype=np.float64).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
    tree = xr.DataTree.from_dict(
        {"0/imagetool": source.to_dataset(name=manager_workspace_io._ITOOL_DATA_NAME)}
    )
    try:
        manager_workspace._write_full_workspace_tree_file(
            source_file, tree, _transaction_test_root_attrs()
        )
    finally:
        tree.close()

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        file_backed = _open_external_file_backed_hdf5_imagetool_data(source_file)
        root = itool(file_backed, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        slicer_area = manager.get_imagetool(0).slicer_area
        assert slicer_area.data_file_backed
        assert slicer_area.data_loadable
        slicer_area._compute_chunked()

        assert manager_xarray.dataarray_is_numpy_backed(slicer_area._data)
        assert not slicer_area.data_file_backed


def test_manager_workspace_load_keeps_visible_saved_data_in_memory(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
            dims=["x", "y"],
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)

        fname = tmp_path / "load-visible-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded = manager.get_imagetool(0).slicer_area
        assert not loaded.data_chunked
        assert not loaded.data_file_backed
        assert loaded.data_loadable is False
        assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
        assert loaded._data.values.flags.writeable
        np.testing.assert_array_equal(loaded._data.values, data.values)


def test_manager_workspace_load_keeps_hidden_memory_payload_pending(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
            dims=["x", "y"],
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "load-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)

        def _fail_h5py_payload_read(*_args, **_kwargs):
            pytest.fail("hidden memory payload should not use fake h5py data")

        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_dataset_group_h5py",
            _fail_h5py_payload_read,
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload == (
            fname.resolve(),
            "0/imagetool",
        )
        assert wrapper.imagetool is None
        index = manager.tree_view._model.index(0, 0)
        option = manager.tree_view._delegate._option_for_index(manager.tree_view, index)
        _, dask_rect, _, _ = manager.tree_view._delegate._compute_icons_info(
            option, wrapper
        )
        assert dask_rect is None


def test_manager_pending_memory_sidebar_shows_metadata_without_materializing(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(20, dtype=np.float64).reshape((4, 5)),
            dims=["energy", "momentum"],
            coords={
                "energy": np.linspace(-1.0, 1.0, 4),
                "momentum": np.linspace(0.0, 0.4, 5),
                "work_function": 4.5,
            },
            attrs={"sample": "TiSe2", "temperature": 12.0},
            name="pending_metadata",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-sidebar-metadata.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "selecting pending sidebar metadata should not materialize data"
            )

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        select_tools(manager, [0])
        manager._update_info()

        info_text = manager.text_box.toPlainText()
        assert "pending_metadata" in info_text
        assert "energy" in info_text
        assert "-1 : 0.6667 : 1" in info_text
        assert "momentum" in info_text
        assert "0 : 0.1 : 0.4" in info_text
        assert "work_function" in info_text
        assert "4.5" in info_text
        assert "float64 [4]" not in info_text
        assert "float64 [5]" not in info_text
        assert "float64 scalar" not in info_text
        assert "sample" in info_text
        assert "TiSe2" in info_text
        assert "temperature" in info_text
        assert wrapper.pending_workspace_memory_payload is not None

        wrapper.name = "renamed_metadata"
        manager._update_info(uid=wrapper.uid)
        renamed_info_text = manager.text_box.toPlainText()
        assert "renamed_metadata" in renamed_info_text
        assert "pending_metadata" not in renamed_info_text
        assert wrapper.pending_workspace_memory_payload is not None

        assert manager.preview_widget.isVisible()
        assert not manager.preview_widget._pixmapitem.pixmap().isNull()
        load_button = manager.preview_widget.findChild(
            QtWidgets.QPushButton, "manager_pending_preview_load_button"
        )
        assert load_button is not None
        assert load_button.isHidden()


def test_pending_workspace_metadata_loads_only_coords() -> None:
    import dask.array as da

    data = xr.DataArray(
        da.from_array(np.arange(6, dtype=np.float64).reshape((2, 3)), chunks=(1, 3)),
        dims=["x", "y"],
        coords={
            "x": ("x", da.from_array(np.array([0.0, 1.0]), chunks=(1,))),
            "y": ("y", da.from_array(np.array([10.0, 11.0, 12.0]), chunks=(1,))),
            "temperature": da.from_array(np.array(12.0), chunks=()),
        },
    )

    controller_cls = manager_workspace_io._WorkspaceIOController
    loaded = controller_cls._pending_workspace_data_with_loaded_coords(data)

    assert loaded.chunks == data.chunks
    assert loaded.coords["x"].chunks is None
    assert loaded.coords["y"].chunks is None
    assert loaded.coords["temperature"].chunks is None
    np.testing.assert_allclose(loaded.coords["x"].values, [0.0, 1.0])
    np.testing.assert_allclose(loaded.coords["y"].values, [10.0, 11.0, 12.0])
    assert float(loaded.coords["temperature"].values) == 12.0


def test_pending_workspace_metadata_coord_load_failure_falls_back(
    tmp_path, monkeypatch
) -> None:
    fname = tmp_path / "pending-metadata-fallback.itws"
    data = xr.DataArray(
        np.arange(6, dtype=np.float64).reshape((2, 3)),
        dims=["x", "y"],
        coords={"x": [0.0, 1.0], "y": [10.0, 11.0, 12.0], "temperature": 12.0},
        name=manager_workspace_io._ITOOL_DATA_NAME,
    )
    ds = xr.Dataset(
        {manager_workspace_io._ITOOL_DATA_NAME: data},
        attrs={"itool_name": "pending"},
    )
    assert manager_workspace._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )

    def _fail_coord_load(_data: xr.DataArray) -> xr.DataArray:
        raise RuntimeError("coord read failed")

    monkeypatch.setattr(
        manager_workspace_io._WorkspaceIOController,
        "_pending_workspace_data_with_loaded_coords",
        staticmethod(_fail_coord_load),
    )
    node = types.SimpleNamespace(
        pending_workspace_memory_payload=(fname, "0/imagetool"),
        pending_workspace_payload_attrs=None,
        name="pending",
        added_time_display="Today",
    )

    controller_cls = manager_workspace_io._WorkspaceIOController
    info = controller_cls._pending_workspace_imagetool_info_text(node)
    assert info is not None
    assert "pending" in info
    assert "float64 [2]" in info
    assert "float64 [3]" in info
    assert "float64 scalar" in info
    assert node.pending_workspace_memory_payload == (fname, "0/imagetool")


def test_manager_pending_memory_file_source_full_code_uses_saved_load_code(
    qtbot,
    monkeypatch,
    tmp_path,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    file_path = tmp_path / "scan.h5"
    test_data.to_netcdf(file_path, engine="h5netcdf")

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root = itool(
            test_data,
            manager=False,
            execute=False,
            file_path=file_path,
            load_func=(xr.load_dataarray, {"engine": "h5netcdf"}, 0),
        )
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(
            provenance.full_data(provenance.AverageOperation(dims=("alpha",)))
        )

        fname = tmp_path / "pending-file-source-replay.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("copying pending full code should not materialize data")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-origin replay should not prompt"),
        )
        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )

        select_tools(manager, [0])
        manager._update_info()
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)

        assert copied
        assert not provenance.uses_default_replay_input(copied[-1])
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(
            namespace["derived"].rename(None),
            test_data.qsel.mean("alpha").rename(None),
        )
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None


def test_manager_pending_memory_preview_button_materializes_selected_node(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        original_values = []
        for offset in (0, 100):
            values = np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5))
            original_values.append(values)
            root = itool(
                xr.DataArray(values, dims=["x", "y"], name=f"pending_{offset}"),
                manager=False,
                execute=False,
            )
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()

        fname = tmp_path / "pending-preview-button.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )

        def _fail_pending_preview(_node):
            pytest.fail("multi-selection should not render pending previews")

        selection_model = manager.tree_view.selectionModel()
        blocker = QtCore.QSignalBlocker(selection_model)
        try:
            select_tools(manager, [0, 1])
        finally:
            del blocker
        monkeypatch.setattr(
            manager,
            "_pending_workspace_imagetool_preview_image",
            _fail_pending_preview,
        )
        monkeypatch.setattr(
            manager,
            "_pending_workspace_imagetool_preview_curve",
            _fail_pending_preview,
        )
        manager._update_info()
        selection_model.clearSelection()

        monkeypatch.setattr(
            manager,
            "_pending_workspace_imagetool_preview_image",
            lambda _node: None,
        )
        monkeypatch.setattr(
            manager,
            "_pending_workspace_imagetool_preview_curve",
            lambda _node: None,
        )

        select_tools(manager, [0])
        manager._update_info()
        load_button = manager.preview_widget.findChild(
            QtWidgets.QPushButton, "manager_pending_preview_load_button"
        )
        assert load_button is not None
        assert load_button.isVisible()

        qtbot.mouseClick(load_button, QtCore.Qt.MouseButton.LeftButton)
        qtbot.wait_until(
            lambda: wrappers[0].pending_workspace_memory_payload is None,
            timeout=5000,
        )

        assert wrappers[1].pending_workspace_memory_payload is not None
        assert not manager.is_workspace_modified
        loaded = manager.get_imagetool(0).slicer_area
        np.testing.assert_array_equal(loaded._data.values, original_values[0])
        qtbot.wait_until(
            lambda: not manager.preview_widget._pixmapitem.pixmap().isNull(),
            timeout=5000,
        )
        assert load_button.isHidden()


def test_manager_update_info_accepts_selected_root_uid(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for offset in (0, 100):
            root = itool(
                xr.DataArray(
                    np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                    dims=["x", "y"],
                    name=f"root_{offset}",
                ),
                manager=False,
                execute=False,
            )
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        rendered_uids: list[str] = []
        original_info_html = manager._node_info_html

        def _record_info_html(node) -> str:
            rendered_uids.append(node.uid)
            return original_info_html(node)

        monkeypatch.setattr(manager, "_node_info_html", _record_info_html)
        select_tools(manager, [0])
        rendered_uids.clear()

        manager._update_info(uid=wrappers[1].uid)
        assert rendered_uids == []

        manager._update_info(uid=wrappers[0].uid)
        assert rendered_uids == [wrappers[0].uid]


def test_manager_pending_memory_sidebar_renders_partial_preview_without_materializing(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        values = np.arange(4 * 5 * 6, dtype=np.float64).reshape((4, 5, 6))
        data = xr.DataArray(values, dims=["x", "y", "z"], name="partial_preview")
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        root.slicer_area.array_slicer.set_indices(0, [2, 3, 4], update=False)
        root.slicer_area.array_slicer.set_bins(0, [1, 1, 3], update=False)
        root.slicer_area.set_colormap(
            cmap="viridis",
            gamma=1.0,
            levels_locked=True,
            levels=(10.0, 90.0),
            update=False,
        )
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-partial-preview.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r+") as h5_file:
            attrs = h5_file["0/imagetool"].attrs
            state = json.loads(attrs["itool_state"])
            state["slice"]["indices"][0] = [2, 3, 999]
            attrs["itool_state"] = json.dumps(state)

        def _fail_full_payload_read(*_args, **_kwargs):
            pytest.fail("pending preview should not read the full payload")

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("pending preview should not materialize data")

        monkeypatch.setattr(
            manager_workspace,
            "_read_workspace_dataset_group_h5py",
            _fail_full_payload_read,
        )
        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None

        select_tools(manager, [0])
        manager._update_info()

        pixmap = manager.preview_widget._pixmapitem.pixmap()
        assert not pixmap.isNull()
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None
        assert not manager.is_workspace_modified
        cached = manager_wrapper._preview_image_for_node(wrapper)
        assert not cached[1].isNull()


def test_manager_pending_memory_partial_preview_read_cap_falls_back_to_load_button(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="too_large_for_preview",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-preview-cap.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES",
            0,
        )

        select_tools(manager, [0])
        manager._update_info()

        assert manager.preview_widget._pixmapitem.pixmap().isNull()
        load_button = manager.preview_widget.findChild(
            QtWidgets.QPushButton, "manager_pending_preview_load_button"
        )
        assert load_button is not None
        assert load_button.isVisible()


def test_manager_pending_memory_1d_preview_read_cap_falls_back_to_load_button(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.linspace(0.0, 1.0, 8),
            dims=["energy"],
            coords={"energy": np.linspace(-1.0, 1.0, 8)},
            name="spectrum",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-1d-preview-cap.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        monkeypatch.setattr(
            manager_workspace_io,
            "_PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES",
            0,
        )

        select_tools(manager, [0])
        manager._update_info()

        assert manager.preview_widget._pixmapitem.pixmap().isNull()
        assert manager.preview_widget._curve_data is None
        load_button = manager.preview_widget.findChild(
            QtWidgets.QPushButton, "manager_pending_preview_load_button"
        )
        assert load_button is not None
        assert load_button.isVisible()


def test_manager_pending_memory_1d_preview_uses_promoted_stack_dim(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.linspace(0.0, 1.0, 8),
            dims=["energy"],
            coords={"energy": np.linspace(-1.0, 1.0, 8)},
            name="spectrum",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        assert root.slicer_area.data.dims == ("energy", "stack_dim")
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-1d-preview.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None

        select_tools(manager, [0])
        manager._update_info()

        assert manager.preview_widget._pixmapitem.pixmap().isNull()
        assert manager.preview_widget._curve_data is not None
        assert not manager.preview_widget._curve_item.path().isEmpty()
        load_button = manager.preview_widget.findChild(
            QtWidgets.QPushButton, "manager_pending_preview_load_button"
        )
        assert load_button is not None
        assert load_button.isHidden()
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None


def test_manager_live_1d_preview_uses_curve(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.linspace(0.0, 1.0, 8),
            dims=["energy"],
            coords={"energy": np.linspace(-1.0, 1.0, 8)},
            name="live_spectrum",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        curve = manager_wrapper._preview_curve_from_imagetool(root)
        assert curve is not None
        np.testing.assert_allclose(curve[0], data.coords["energy"].values)
        np.testing.assert_allclose(curve[1], data.values)

        root.slicer_area.transpose_main_image()
        transposed_curve = manager_wrapper._preview_curve_from_imagetool(root)
        assert transposed_curve is not None
        np.testing.assert_allclose(transposed_curve[0], data.coords["energy"].values)
        np.testing.assert_allclose(transposed_curve[1], data.values)

        select_tools(manager, [0])
        manager._update_info()

        assert manager.preview_widget._pixmapitem.pixmap().isNull()
        assert manager.preview_widget._curve_data is not None
        assert not manager.preview_widget._curve_item.path().isEmpty()


def test_manager_live_1d_preview_reuses_rendered_dask_curve_without_computing(
    qtbot,
) -> None:
    da = pytest.importorskip("dask.array")
    from dask.callbacks import Callback

    dask_options = erlab.interactive.options.model.io.dask
    old_threshold = dask_options.compute_threshold
    object.__setattr__(dask_options, "compute_threshold", 0)
    try:
        values = np.linspace(0.0, 1.0, 8)
        initial_data = xr.DataArray(
            values,
            dims=["energy"],
            coords={"energy": np.linspace(-1.0, 1.0, 8)},
            name="live_dask_spectrum",
        )
        root = itool(initial_data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        qtbot.addWidget(root)

        data = xr.DataArray(
            da.from_array(values, chunks=(4,)),
            dims=["energy"],
            coords={"energy": np.linspace(-1.0, 1.0, 8)},
            name="live_dask_spectrum",
        )
        root.slicer_area.set_data(data, auto_compute=False)
        assert root.slicer_area._data.chunks is not None
        root.slicer_area._update_if_delayed()

        computed_keys: list[object] = []
        with Callback(pretask=lambda key, _dsk, _state: computed_keys.append(key)):
            curve = manager_wrapper._preview_curve_from_imagetool(root)

        assert curve is not None
        assert computed_keys == []
        np.testing.assert_allclose(curve[0], data.coords["energy"].values)
        np.testing.assert_allclose(curve[1], values)
    finally:
        object.__setattr__(dask_options, "compute_threshold", old_threshold)


def test_manager_curve_preview_widget_handles_constant_nonfinite_and_decimation(
    qtbot,
) -> None:
    preview = manager_widgets._SingleImagePreview()
    qtbot.addWidget(preview)
    assert bool(preview.renderHints() & QtGui.QPainter.RenderHint.Antialiasing)
    assert bool(preview.renderHints() & QtGui.QPainter.RenderHint.SmoothPixmapTransform)

    preview.setCurve(
        np.arange(5, dtype=np.float64),
        np.array([1.0, 2.0, np.nan, np.inf, 1.0]),
    )
    assert preview._curve_data is not None
    assert not preview._curve_item.path().isEmpty()
    assert preview._curve_item.pen().isCosmetic()
    assert preview._curve_item.pen().color() == manager_widgets._summary_accent_color()

    preview.setCurve(np.arange(4, dtype=np.float64), np.ones(4, dtype=np.float64))
    assert preview._curve_data is not None
    assert not preview._curve_item.path().isEmpty()
    assert preview.scene() is not None
    assert preview.scene().sceneRect() == QtCore.QRectF(0.0, 0.0, 1.0, 1.0)

    long_x = np.arange(manager_widgets._CURVE_PREVIEW_MAX_POINTS * 3, dtype=np.float64)
    long_y = np.sin(long_x / 50.0)
    decimated = manager_widgets._curve_preview_data(long_x, long_y)
    assert decimated is not None
    assert decimated[0].size <= manager_widgets._CURVE_PREVIEW_MAX_POINTS

    pixmap = QtGui.QPixmap(32, 16)
    pixmap.fill(QtCore.Qt.GlobalColor.white)
    preview.setPixmap(pixmap)
    assert preview._curve_data is None
    assert preview._curve_item.isVisible() is False
    assert preview.scene() is not None
    assert preview.scene().sceneRect() == QtCore.QRectF(0.0, 0.0, 32.0, 16.0)


def test_pending_workspace_preview_and_metadata_reader_fallbacks(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        controller = manager._workspace_controller

        assert controller._pending_preview_axis_state((4, 5), {}) is None
        assert (
            controller._pending_preview_axis_state(
                (4, 5), {"slice": {"bins": [], "indices": []}}
            )
            is None
        )
        assert controller._pending_preview_axis_state(
            (4, 5),
            {"slice": {"bins": [["bad", 3]], "indices": [["bad", 99]]}},
        ) == ((1, 2), (1, 3), (False, True))
        assert controller._pending_preview_axis_state(
            (4, 5),
            {"slice": {"bins": [[0, -2]], "indices": [[2, 3]]}},
        ) == ((2, 3), (1, 1), (False, False))
        assert controller._pending_preview_axis_state(
            (4, 5),
            {
                "current_cursor": 1,
                "slice": {"bins": [[1, 1], [1, 1]], "indices": [[0, 0]]},
            },
        ) == ((1, 2), (1, 1), (False, False))
        assert (
            controller._pending_preview_axis_state(
                (4, 5), {"slice": {"bins": ["bad"], "indices": ["bad"]}}
            )
            is None
        )

        fname = tmp_path / "pending-preview-reader-fallbacks.itws"
        data = xr.DataArray(
            np.arange(4 * 5 * 6, dtype=np.float64).reshape((4, 5, 6)),
            dims=["x", "y", "z"],
            coords={"x": np.arange(4), "y": np.arange(5), "z": np.arange(6)},
            name=manager_workspace_io._ITOOL_DATA_NAME,
        )
        state = {
            "slice": {
                "dims": ["x", "y", "z"],
                "bins": [[1, 1, 3]],
                "indices": [[2, 3, 4]],
            },
            "current_cursor": 0,
            "color": {
                "cmap": "viridis",
                "levels_locked": True,
                "levels": [10.0, 90.0],
            },
        }
        ds = xr.Dataset(
            {manager_workspace_io._ITOOL_DATA_NAME: data},
            attrs={"itool_state": json.dumps(state), "itool_name": "pending"},
        )
        assert manager_workspace._write_workspace_dataset_group_h5py(
            fname, "0/imagetool", ds
        )

        node = types.SimpleNamespace(
            pending_workspace_memory_payload=(fname, "0/imagetool"),
            pending_workspace_payload_attrs=None,
            name="pending",
            added_time_display="Today",
        )
        info = controller._pending_workspace_imagetool_info_text(node)
        assert info is not None
        assert "pending" in info
        preview = controller._pending_workspace_imagetool_preview_image(node)
        assert preview is not None
        assert preview[1].isNull() is False

        nonuniform_fname = tmp_path / "pending-preview-nonuniform-index.itws"
        nonuniform_data = xr.DataArray(
            np.arange(4 * 5 * 6, dtype=np.float64).reshape((4, 5, 6)),
            dims=["alpha_idx", "eV", "z"],
            coords={
                "alpha_idx": np.arange(4, dtype=np.float32),
                "alpha": ("alpha_idx", np.array([-2.0, -0.4, 0.2, 2.1])),
                "eV": np.arange(5, dtype=np.float64),
                "z": np.arange(6, dtype=np.float64),
            },
            name=manager_workspace_io._ITOOL_DATA_NAME,
        )
        nonuniform_state = {
            "slice": {
                "dims": ["alpha_idx", "eV", "z"],
                "bins": [[1, 1, 2]],
                "indices": [[2, 3, 4]],
            },
            "current_cursor": 0,
            "color": {"cmap": "viridis"},
        }
        nonuniform_ds = xr.Dataset(
            {manager_workspace_io._ITOOL_DATA_NAME: nonuniform_data},
            attrs={
                "itool_state": json.dumps(nonuniform_state),
                "itool_name": "nonuniform",
            },
        )
        assert manager_workspace._write_workspace_dataset_group_h5py(
            nonuniform_fname, "0/imagetool", nonuniform_ds
        )
        with h5py.File(nonuniform_fname, "r+") as h5_file:
            group = h5_file["0/imagetool"]
            physical_coord = group.create_dataset(
                "alpha_physical_scale", data=np.array([-2.0, -0.4, 0.2, 2.1])
            )
            physical_coord.make_scale("alpha")
            data_dataset = group[manager_workspace_io._ITOOL_DATA_NAME]
            data_dataset.dims[0].attach_scale(physical_coord)
            assert len(list(data_dataset.dims[0].keys())) > 1
        node.pending_workspace_memory_payload = (nonuniform_fname, "0/imagetool")
        node.pending_workspace_payload_attrs = None
        nonuniform_preview = controller._pending_workspace_imagetool_preview_image(node)
        assert nonuniform_preview is not None
        assert nonuniform_preview[1].isNull() is False

        permuted_fname = tmp_path / "pending-preview-permuted-index.itws"
        permuted_data = xr.DataArray(
            np.arange(3 * 5 * 4, dtype=np.float64).reshape((3, 5, 4)),
            dims=["sample_temp", "eV", "alpha"],
            coords={
                "sample_temp": np.array([249.4, 251.2, 253.8]),
                "eV": np.linspace(-0.5, 0.5, 5),
                "alpha": np.linspace(-2.0, 2.0, 4),
            },
            name=manager_workspace_io._ITOOL_DATA_NAME,
        )
        permuted_state = {
            "slice": {
                "dims": ["alpha", "eV", "sample_temp_idx"],
                "bins": [[1, 1, 1]],
                "indices": [[2, 3, 1]],
            },
            "current_cursor": 0,
            "color": {"cmap": "viridis"},
        }
        permuted_ds = xr.Dataset(
            {manager_workspace_io._ITOOL_DATA_NAME: permuted_data},
            attrs={
                "itool_state": json.dumps(permuted_state),
                "itool_name": "permuted",
            },
        )
        assert manager_workspace._write_workspace_dataset_group_h5py(
            permuted_fname, "0/imagetool", permuted_ds
        )
        with h5py.File(permuted_fname, "r") as h5_file:
            data_dataset = h5_file[
                f"0/imagetool/{manager_workspace_io._ITOOL_DATA_NAME}"
            ]
            assert controller._pending_preview_dataset_dims(
                data_dataset, ("alpha", "eV", "sample_temp_idx")
            ) == (("sample_temp_idx", "eV", "alpha"), None)
        node.pending_workspace_memory_payload = (permuted_fname, "0/imagetool")
        node.pending_workspace_payload_attrs = None
        permuted_preview = controller._pending_workspace_imagetool_preview_image(node)
        assert permuted_preview is not None
        assert permuted_preview[1].isNull() is False

        missing_node = types.SimpleNamespace(
            pending_workspace_memory_payload=(tmp_path / "missing.itws", "0/imagetool"),
            pending_workspace_payload_attrs=None,
            name="missing",
            added_time_display="Today",
        )
        assert controller._pending_workspace_imagetool_info_text(missing_node) is None
        assert (
            controller._pending_workspace_imagetool_preview_image(missing_node) is None
        )

        no_pending = types.SimpleNamespace(
            pending_workspace_memory_payload=None,
            pending_workspace_payload_attrs=None,
            name="missing",
            added_time_display="Today",
        )
        assert controller._pending_workspace_imagetool_info_text(no_pending) is None
        assert controller._pending_workspace_imagetool_preview_image(no_pending) is None

        malformed_payload_attrs = [
            {"itool_state": json.dumps([])},
            {"itool_state": json.dumps({"slice": None})},
            {"itool_state": json.dumps({"slice": {"dims": "xy"}})},
            {
                "itool_state": json.dumps(
                    {
                        "slice": {
                            "dims": ["missing", "y", "z"],
                            "bins": [[1, 1, 1]],
                            "indices": [[0, 0, 0]],
                        }
                    }
                )
            },
            {
                "itool_state": json.dumps(
                    {"slice": {"dims": ["x", "y", "z"], "bins": [], "indices": []}}
                ).encode()
            },
        ]
        for attrs in malformed_payload_attrs:
            node.pending_workspace_payload_attrs = attrs
            assert controller._pending_workspace_imagetool_preview_image(node) is None

        empty_fname = tmp_path / "pending-preview-empty.itws"
        with h5py.File(empty_fname, "w") as h5_file:
            h5_file.create_group("0/imagetool")
        node.pending_workspace_memory_payload = (empty_fname, "0/imagetool")
        node.pending_workspace_payload_attrs = None
        assert controller._pending_workspace_imagetool_preview_image(node) is None


def test_pending_preview_slicer_helpers_match_array_slicer(qtbot) -> None:
    data = xr.DataArray(
        np.arange(5 * 6 * 7 * 8, dtype=np.float64).reshape((5, 6, 7, 8)),
        dims=["a", "b", "c", "d"],
    )
    array_slicer = erlab.interactive.imagetool.slicer.ArraySlicer(
        data, QtCore.QObject()
    )
    array_slicer.set_indices(0, [2, 3, 0, 7], update=False)
    array_slicer.set_bins(0, [1, 1, 3, 5], update=False)

    hidden_axes = erlab.interactive.imagetool.slicer._hidden_axes_for_display(
        data.ndim, (0, 1)
    )
    assert hidden_axes == array_slicer._hidden_axes_for_disp((0, 1))
    selection, reduction_axes, any_binned, all_binned = (
        erlab.interactive.imagetool.slicer._reduced_axes_selection(
            data.shape,
            hidden_axes,
            array_slicer.get_indices(0),
            array_slicer.get_bins(0),
            array_slicer.get_binned(0),
        )
    )

    assert any_binned
    assert all_binned
    assert selection[2] == array_slicer._bin_slice(0, 2)
    assert selection[3] == array_slicer._bin_slice(0, 3)
    assert reduction_axes == (2, 3)


def test_manager_pending_memory_hover_preview_does_not_materialize(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-hover-preview.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("hover preview should not materialize pending data")

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        manager.preview_action.setChecked(True)
        delegate = manager.tree_view._delegate
        delegate._force_hover = True
        index = manager.tree_view._model.index(0, 0)
        option = delegate._option_for_index(manager.tree_view, index)
        canvas = QtGui.QPixmap(200, 32)
        canvas.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(canvas)
        try:
            delegate.paint(painter, option, index)
        finally:
            painter.end()
            delegate._force_hover = False

        assert wrapper.pending_workspace_memory_payload is not None
        assert not delegate.preview_popup.isVisible()


def test_manager_workspace_open_coalesces_pending_memory_wait_dialogs(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-wait-dialogs.itws"
        manager._save_workspace_document(fname, force_full=True)
        messages: list[str] = []

        @contextlib.contextmanager
        def _record_wait_dialog(_parent, message):
            messages.append(message)
            yield types.SimpleNamespace(
                set_message=lambda updated: messages.append(updated)
            )

        monkeypatch.setattr(erlab.interactive.utils, "wait_dialog", _record_wait_dialog)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]

        assert wrapper.pending_workspace_memory_payload is not None
        assert messages.count("Loading workspace...") == 1
        assert "Loading ImageTool data..." not in messages

        messages.clear()
        assert manager._materialize_pending_workspace_payload(wrapper)

        assert messages == ["Loading ImageTool data..."]


def test_hidden_workspace_toolwindows_restore_pending_until_shown(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=("x", "y"),
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root_uid = manager._tool_graph.root_wrappers[0].uid
        root.hide()

        child = _AddedTimeChildTool(data.rename("child"))
        child._tool_display_name = "Hidden child"
        child_uid = manager.add_childtool(child, 0, show=False)
        child.hide()

        figure = _WorkspaceManagerReferenceFigureTool(
            data.rename("figure"),
            reference_uid=root_uid,
        )
        figure._tool_display_name = "Hidden figure"
        figure_uid = manager.add_figuretool(figure, show=False)
        figure.hide()

        fname = tmp_path / "hidden-toolwindows-pending.itws"
        manager._save_workspace_document(fname, force_full=True)

        original_from_dataset = erlab.interactive.utils.ToolWindow.from_dataset.__func__
        constructed: list[str] = []

        def _record_from_dataset(cls, ds, *args, **kwargs):
            constructed.append(str(ds.attrs.get("tool_display_name", "")))
            return original_from_dataset(cls, ds, *args, **kwargs)

        monkeypatch.setattr(
            erlab.interactive.utils.ToolWindow,
            "from_dataset",
            classmethod(_record_from_dataset),
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        assert constructed == []
        root_node = manager._tool_graph.root_wrappers[0]
        child_node = manager._child_node(child_uid)
        figure_node = manager._child_node(figure_uid)

        assert root_node.pending_workspace_memory_payload is not None
        assert child_node.pending_workspace_tool_payload is not None
        assert child_node.tool_window is None
        assert figure_node.pending_workspace_tool_payload is not None
        assert figure_node.tool_window is None
        assert figure_uid in manager._figure_uids()

        child_node.show()

        assert constructed == ["Hidden child"]
        assert child_node.pending_workspace_tool_payload is None
        assert child_node.tool_window is not None
        assert root_node.pending_workspace_memory_payload is not None


def test_full_save_copies_unopened_pending_toolwindows_without_construction(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=("x", "y"),
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root_uid = manager._tool_graph.root_wrappers[0].uid
        root.hide()

        child = _AddedTimeChildTool(data.rename("child"))
        child._tool_display_name = "Pending child"
        child_uid = manager.add_childtool(child, 0, show=False)
        child.hide()

        figure = _WorkspaceManagerReferenceFigureTool(
            data.rename("figure"),
            reference_uid=root_uid,
        )
        figure._tool_display_name = "Pending figure"
        figure_uid = manager.add_figuretool(figure, show=False)
        figure.hide()

        fname = tmp_path / "pending-toolwindow-source.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        child_node = manager._child_node(child_uid)
        figure_node = manager._child_node(figure_uid)
        assert child_node.pending_workspace_tool_payload is not None
        assert figure_node.pending_workspace_tool_payload is not None

        def _fail_from_dataset(cls, ds, *args, **kwargs):
            del cls, ds, args, kwargs
            pytest.fail("pending ToolWindow save should not construct the tool")

        monkeypatch.setattr(
            erlab.interactive.utils.ToolWindow,
            "from_dataset",
            classmethod(_fail_from_dataset),
        )

        saved = tmp_path / "pending-toolwindow-copy.itws"
        manager._save_workspace_document(saved, force_full=True)

        assert child_node.pending_workspace_tool_payload is not None
        assert figure_node.pending_workspace_tool_payload is not None
        with h5py.File(saved, "r") as h5_file:
            assert f"0/childtools/{child_uid}/tool" in h5_file
            assert f"figures/{figure_uid}/tool" in h5_file


def test_pending_toolwindow_reference_availability_rejects_unsupported_kind(
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="source")

    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root_uid = manager._tool_graph.root_wrappers[0].uid

        child_uid = manager.add_childtool(_AddedTimeChildTool(data), 0, show=False)
        child_node = manager._child_node(child_uid)
        child_node.set_pending_workspace_payload(
            "tool",
            tmp_path / "source.itws",
            f"0/childtools/{child_uid}/tool",
            payload_attrs={
                erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                    {"data": {"kind": "manager_node", "node_uid": root_uid}}
                )
            },
        )
        assert (
            manager._workspace_controller._pending_workspace_tool_references_available(
                child_node
            )
        )

        child_node.update_pending_workspace_payload_attrs(
            {
                erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                    {"data": {"kind": "future_reference", "node_uid": root_uid}}
                )
            }
        )
        assert not (
            manager._workspace_controller._pending_workspace_tool_references_available(
                child_node
            )
        )

        child_node.update_pending_workspace_payload_attrs(
            {
                erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                    {"data": {"kind": "manager_node", "node_uid": "missing"}}
                )
            }
        )
        assert not (
            manager._workspace_controller._pending_workspace_tool_references_available(
                child_node
            )
        )


def test_hidden_toolwindow_with_missing_saved_class_is_skipped(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _AcceptingMessageDialog(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            super().__init__()

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _AcceptingMessageDialog
    )

    data = xr.DataArray(np.arange(4.0), dims=("x",), name="source")
    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        child = _AddedTimeChildTool(data)
        child_uid = manager.add_childtool(child, 0, show=False)
        child.hide()

        fname = tmp_path / "missing-hidden-tool-class.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r+") as h5_file:
            h5_file[f"0/childtools/{child_uid}/tool"].attrs["tool_cls_qualname"] = (
                "erlab.interactive._missing_tool_module:MissingTool"
            )

        skipped: list[tuple[str | None, Exception]] = []
        original_record = manager._workspace_controller._record_skipped_workspace_node

        def _record_skipped(node_path: str | None, exc: Exception) -> None:
            skipped.append((node_path, exc))
            original_record(node_path, exc)

        monkeypatch.setattr(
            manager._workspace_controller,
            "_record_skipped_workspace_node",
            _record_skipped,
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        assert child_uid not in manager._tool_graph.nodes
        assert skipped
        assert skipped[0][0] == f"0/childtools/{child_uid}"
        assert isinstance(skipped[0][1], ModuleNotFoundError)


def test_pending_toolwindow_source_metadata_decodes_saved_state(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        binding = provenance.ImageToolSelectionSourceBinding(
            selection_mode="isel",
            selection_indexers={"x": 0},
            transpose_dims=("y",),
            squeeze=True,
        )
        attrs = {
            erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR: json.dumps(
                binding.model_dump(mode="json")
            ),
            erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR: b"stale",
            erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR: True,
        }

        source_spec, source_binding, auto_update, state = (
            manager._workspace_controller._workspace_tool_source_metadata(attrs)
        )

        assert source_spec is None
        assert source_binding == binding
        assert auto_update is True
        assert state == "stale"

        attrs[erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR] = "not-valid"
        *_, state = manager._workspace_controller._workspace_tool_source_metadata(attrs)
        assert state == "fresh"


def test_manager_get_imagetool_materializes_hidden_memory_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "get-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        loaded = manager.get_imagetool(0).slicer_area

        assert wrapper.pending_workspace_memory_payload is None
        assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
        np.testing.assert_array_equal(loaded._data.values, data.values)


def test_manager_persistence_view_materializes_hidden_memory_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "persistence-view-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        persistence = wrapper.persistence_view()

        assert wrapper.pending_workspace_memory_payload is None
        assert persistence.data is not None
        assert persistence.state is not None
        assert persistence.data_backing == "memory"
        xr.testing.assert_identical(persistence.data, data)


def test_manager_console_namespace_materializes_hidden_memory_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "console-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        namespace = manager_console.ToolNamespace(wrapper)
        loaded = namespace.data

        assert wrapper.pending_workspace_memory_payload is None
        np.testing.assert_array_equal(loaded.values, data.values)


def test_manager_figure_operation_materializes_hidden_memory_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "figure-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        manager._figure_operations_from_image_targets((0,), ("saved",))

        assert wrapper.pending_workspace_memory_payload is None
        np.testing.assert_array_equal(
            manager.get_imagetool(0).slicer_area._data.values,
            data.values,
        )


def test_manager_duplicate_pending_memory_uses_saved_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "duplicate-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        duplicate_index = manager.duplicate_imagetool(0)

        assert wrapper.pending_workspace_memory_payload is None
        duplicated = manager.get_imagetool(duplicate_index).slicer_area
        assert manager_xarray.dataarray_is_numpy_backed(duplicated._data)
        np.testing.assert_array_equal(duplicated._data.values, data.values)


def test_manager_promote_pending_child_memory_uses_saved_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        root_data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="root",
        )
        child_data = (root_data + 100.0).rename("child")

        root = itool(root_data, manager=False, execute=False)
        child = itool(child_data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        child_uid = manager.add_imagetool_child(child, 0, show=False)
        root.hide()
        child.hide()

        fname = tmp_path / "promote-child-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        child_node = manager._child_node(child_uid)
        assert child_node.pending_workspace_memory_payload is not None

        promoted_index = manager.promote_child_imagetool(child_uid)
        promoted = manager.get_imagetool(promoted_index).slicer_area

        assert (
            manager._tool_graph.root_wrappers[
                promoted_index
            ].pending_workspace_memory_payload
            is None
        )
        np.testing.assert_array_equal(promoted._data.values, child_data.values)


def test_manager_reload_selected_materializes_hidden_memory_payload(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "reload-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        reloaded_values: list[np.ndarray] = []

        def _reload(area):
            reloaded_values.append(np.asarray(area._data.values).copy())
            return True

        monkeypatch.setattr(
            imagetool_viewer.ImageSlicerArea,
            "_reload",
            _reload,
        )
        monkeypatch.setattr(
            manager,
            "_selected_reload_candidates",
            lambda: ([0], {}, None),
        )

        manager.reload_selected()

        assert wrapper.pending_workspace_memory_payload is None
        assert len(reloaded_values) == 1
        np.testing.assert_array_equal(reloaded_values[0], data.values)


def test_manager_active_filter_edit_materializes_hidden_memory_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    restored: list[provenance.ToolProvenanceOperation] = []

    class _FilterDialog(manager_provenance_edit.dialogs.DataFilterDialog):
        def __init__(self, slicer_area) -> None:
            QtWidgets.QDialog.__init__(self)
            self.slicer_area = slicer_area

        def restore_filter_operation(self, operation) -> None:
            restored.append(operation)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Rejected)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "active-filter-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        operation = provenance.GaussianFilterOperation(sigma={"x": 1.0})

        manager._provenance_edit_controller._edit_active_filter(
            wrapper,
            operation,
            _FilterDialog,
        )

        assert wrapper.pending_workspace_memory_payload is None
        assert restored == [operation]
        assert manager_xarray.dataarray_is_numpy_backed(wrapper.slicer_area._data)
        np.testing.assert_array_equal(wrapper.slicer_area._data.values, data.values)


def test_manager_workspace_show_materializes_hidden_memory_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
            name="saved",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        root.slicer_area.set_index(1, 3)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "show-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        manager.show_imagetool(0)
        qtbot.wait_until(lambda: manager.get_imagetool(0).isVisible())

        loaded = manager.get_imagetool(0).slicer_area
        assert wrapper.pending_workspace_memory_payload is None
        assert loaded.data_loadable is False
        assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
        np.testing.assert_array_equal(loaded._data.values, data.values)
        assert loaded.array_slicer.get_value(0, 1) == 3.0


def test_manager_workspace_child_tool_reference_keeps_pending_parent_unmaterialized(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="parent",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        child_data = data.copy(deep=True).rename("child")
        child = _WorkspaceSweepChildTool(child_data)
        child.set_source_binding(provenance.full_data())
        child_uid = manager.add_childtool(child, 0, show=False)

        fname = tmp_path / "pending-parent-child-reference.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            references = json.loads(
                h5_file[f"0/childtools/{child_uid}/tool"].attrs["tool_data_references"]
            )
        assert references[imagetool_serialization.SAVED_TOOL_DATA_NAME] == {
            "kind": "parent_source"
        }

        original_materialize = manager._materialize_pending_workspace_payload
        controller_cls = manager_workspace_io._WorkspaceIOController
        original_read = controller_cls._read_workspace_imagetool_payload_dataset

        def _fail_materialize_pending_payload(_node) -> bool:
            if _node.is_imagetool:
                pytest.fail("tool reference restore should not materialize parent data")
            return original_materialize(_node)

        def _fail_eager_pending_read(cls, workspace_path, payload_path, *, load_data):
            del cls
            if load_data:
                pytest.fail(
                    "tool reference restore should not eagerly load parent data"
                )
            return original_read(workspace_path, payload_path, load_data=load_data)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )
        monkeypatch.setattr(
            controller_cls,
            "_read_workspace_imagetool_payload_dataset",
            classmethod(_fail_eager_pending_read),
        )
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, _WorkspaceSweepChildTool)
        assert loaded_child.tool_data.name == child_data.name
        assert loaded_child.tool_data.dims == child_data.dims
        assert loaded_child.tool_data.chunks is not None
        np.testing.assert_array_equal(loaded_child.tool_data.values, child_data.values)


def test_manager_workspace_tool_manager_node_reference_keeps_pending_source(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
            name="source",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()
        wrapper = manager._tool_graph.root_wrappers[0]

        figure_uid = manager.add_figuretool(
            _WorkspaceManagerReferenceFigureTool(
                data.copy(deep=False), reference_uid=wrapper.uid
            ),
            show=False,
        )

        fname = tmp_path / "pending-manager-node-reference.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            figure_group = h5_file[f"figures/{figure_uid}/tool"]
            references = json.loads(figure_group.attrs["tool_data_references"])
            assert references[imagetool_serialization.SAVED_TOOL_DATA_NAME] == {
                "kind": "manager_node",
                "node_uid": wrapper.uid,
            }
            assert figure_group[imagetool_serialization.SAVED_TOOL_DATA_NAME].shape == (
                0,
            )

        original_materialize = manager._materialize_pending_workspace_payload
        controller_cls = manager_workspace_io._WorkspaceIOController
        original_read = controller_cls._read_workspace_imagetool_payload_dataset

        def _fail_materialize_pending_payload(_node) -> bool:
            if _node.is_imagetool:
                pytest.fail(
                    "tool manager-node restore should not materialize source data"
                )
            return original_materialize(_node)

        def _fail_eager_pending_read(cls, workspace_path, payload_path, *, load_data):
            del cls
            if load_data:
                pytest.fail(
                    "tool manager-node restore should not eagerly load source data"
                )
            return original_read(workspace_path, payload_path, load_data=load_data)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )
        monkeypatch.setattr(
            controller_cls,
            "_read_workspace_imagetool_payload_dataset",
            classmethod(_fail_eager_pending_read),
        )
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        loaded_wrapper = manager._tool_graph.root_wrappers[0]
        assert loaded_wrapper.pending_workspace_memory_payload is not None
        pending_preview = loaded_wrapper.pending_workspace_preview_image()
        assert pending_preview is not None
        assert not pending_preview[1].isNull()

        loaded_figure = manager.get_childtool(figure_uid)
        assert isinstance(loaded_figure, _WorkspaceManagerReferenceFigureTool)
        assert loaded_figure.tool_data.name == data.name
        assert loaded_figure.tool_data.dims == data.dims
        assert loaded_figure.tool_data.chunks is not None
        np.testing.assert_array_equal(loaded_figure.tool_data.values, data.values)


def test_manager_save_as_rebinds_live_tool_reference_dataset(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()
        wrapper = manager._tool_graph.root_wrappers[0]
        figure_uid = manager.add_figuretool(
            _WorkspaceManagerReferenceFigureTool(
                data.copy(deep=False), reference_uid=wrapper.uid
            ),
            show=False,
        )

        source = tmp_path / "reference-source.itws"
        target = tmp_path / "reference-target.itws"
        manager._save_workspace_document(source, force_full=True)
        assert manager._load_workspace_file(
            source, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded_wrapper = manager._tool_graph.root_wrappers[0]
        assert loaded_wrapper.pending_workspace_memory_payload is not None

        loaded_figure = manager.get_childtool(figure_uid)
        assert isinstance(loaded_figure, _WorkspaceManagerReferenceFigureTool)
        source_paths = manager_xarray.dataarray_source_paths(loaded_figure.tool_data)
        assert str(source.resolve()) in source_paths

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: target)
        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)

        rebound_paths = manager_xarray.dataarray_source_paths(loaded_figure.tool_data)
        assert str(target.resolve()) in rebound_paths
        assert str(source.resolve()) not in rebound_paths
        assert loaded_wrapper.pending_workspace_memory_payload == (
            target.resolve(),
            "0/imagetool",
        )
        np.testing.assert_array_equal(loaded_figure.tool_data.values, data.values)


def test_manager_save_as_rebind_failure_keeps_old_live_references(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()
        wrapper = manager._tool_graph.root_wrappers[0]
        figure_uid = manager.add_figuretool(
            _WorkspaceManagerReferenceFigureTool(
                data.copy(deep=False), reference_uid=wrapper.uid
            ),
            show=False,
        )

        source = tmp_path / "reference-failure-source.itws"
        target = tmp_path / "reference-failure-target.itws"
        manager._save_workspace_document(source, force_full=True)
        assert manager._load_workspace_file(
            source, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded_wrapper = manager._tool_graph.root_wrappers[0]
        loaded_figure = manager.get_childtool(figure_uid)
        assert isinstance(loaded_figure, _WorkspaceManagerReferenceFigureTool)
        original_paths = manager_xarray.dataarray_source_paths(loaded_figure.tool_data)
        assert str(source.resolve()) in original_paths

        def _fail_replace(_data_items, _ds) -> None:
            raise RuntimeError("replacement failed")

        errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            loaded_figure, "_replace_persistence_data_items", _fail_replace
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: target)
        manager._mark_node_state_dirty(figure_uid)

        assert not _request_workspace_save_as_and_wait(qtbot, manager, native=False)
        assert errors[-1] == (
            "Workspace file saved but live references were not updated",
            "The workspace file was saved, but live tool data could not be "
            "updated to use the saved file. Reopen the workspace to continue "
            "from the saved version.",
        )
        assert manager._workspace_state.path == source.resolve()
        assert manager.is_workspace_modified
        assert loaded_wrapper.pending_workspace_memory_payload == (
            source.resolve(),
            "0/imagetool",
        )
        assert manager_xarray.dataarray_source_paths(loaded_figure.tool_data) == (
            original_paths
        )


def test_manager_compact_rebind_failure_keeps_workspace_modified(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()
        wrapper = manager._tool_graph.root_wrappers[0]
        figure_uid = manager.add_figuretool(
            _WorkspaceManagerReferenceFigureTool(
                data.copy(deep=False), reference_uid=wrapper.uid
            ),
            show=False,
        )

        source = tmp_path / "reference-compact-failure-source.itws"
        manager._save_workspace_document(source, force_full=True)
        assert manager._load_workspace_file(
            source, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded_figure = manager.get_childtool(figure_uid)
        assert isinstance(loaded_figure, _WorkspaceManagerReferenceFigureTool)
        assert not manager.is_workspace_modified

        def _fail_replace(_data_items, _ds) -> None:
            raise RuntimeError("replacement failed")

        errors: list[tuple[str, str]] = []
        monkeypatch.setattr(
            loaded_figure, "_replace_persistence_data_items", _fail_replace
        )
        monkeypatch.setattr(
            manager,
            "_show_operation_error",
            lambda title, text: errors.append((title, text)),
        )
        monkeypatch.setattr(
            erlab.interactive.utils,
            "wait_dialog",
            lambda *args, **kwargs: contextlib.nullcontext(),
        )

        assert not manager.compact_workspace()
        assert errors[-1] == (
            "Workspace file saved but live references were not updated",
            "The workspace file was saved, but live tool data could not be "
            "updated to use the saved file. Reopen the workspace to continue "
            "from the saved version.",
        )
        assert manager._workspace_state.path == source.resolve()
        assert manager.is_workspace_modified
        assert manager._workspace_state.needs_full_save
        assert "Live workspace data references need refresh" in (
            manager._workspace_state.structure_reasons
        )


def test_manager_save_as_does_not_revive_stale_live_tool_references(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="source",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()
        wrapper = manager._tool_graph.root_wrappers[0]
        figure_uid = manager.add_figuretool(
            _WorkspaceManagerReferenceFigureTool(
                data.copy(deep=False), reference_uid=wrapper.uid
            ),
            show=False,
        )

        source = tmp_path / "stale-reference-source.itws"
        target = tmp_path / "stale-reference-target.itws"
        manager._save_workspace_document(source, force_full=True)
        assert manager._load_workspace_file(
            source, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded_figure = manager.get_childtool(figure_uid)
        assert isinstance(loaded_figure, _WorkspaceManagerReferenceFigureTool)
        assert str(source.resolve()) in manager_xarray.dataarray_source_paths(
            loaded_figure.tool_data
        )

        loaded_figure._reference_uid = None
        manager._mark_node_state_dirty(figure_uid)
        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: target)

        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)
        rebound_paths = manager_xarray.dataarray_source_paths(loaded_figure.tool_data)
        assert str(target.resolve()) in rebound_paths
        assert str(source.resolve()) not in rebound_paths
        assert (
            manager._node_for_target(figure_uid)._workspace_tool_data_references == {}
        )
        np.testing.assert_array_equal(loaded_figure.tool_data.values, data.values)


def test_manager_save_as_repoints_imported_pending_payload(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="imported",
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        source = tmp_path / "import-source.itws"
        target = tmp_path / "import-target.itws"
        manager._save_workspace_document(source, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        base_data = xr.DataArray(np.arange(4.0).reshape((2, 2)), dims=["x", "y"])
        base_root = itool(base_data, manager=False, execute=False)
        assert isinstance(base_root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(base_root, show=False)
        base = tmp_path / "import-base.itws"
        manager._save_workspace_document(base, force_full=True)
        manager._adopt_workspace_path(base)
        manager._mark_workspace_clean()

        assert manager._load_workspace_file(
            source, replace=False, associate=False, mark_dirty=True, select=False
        )
        wrapper = next(
            node
            for node in manager._tool_graph.root_wrappers.values()
            if node.pending_workspace_memory_payload is not None
        )
        assert wrapper.pending_workspace_memory_payload == (
            source.resolve(),
            "0/imagetool",
        )
        payload_path = manager._workspace_payload_path(wrapper.uid)

        monkeypatch.setattr(manager, "_workspace_save_dialog", lambda **_kwargs: target)
        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)
        assert wrapper.pending_workspace_memory_payload == (
            target.resolve(),
            payload_path,
        )

        source.unlink()
        manager.show_imagetool(wrapper.index)
        qtbot.wait_until(lambda: manager.get_imagetool(wrapper.index).isVisible())
        np.testing.assert_array_equal(
            manager.get_imagetool(wrapper.index).slicer_area._data.values,
            data.values,
        )


def test_manager_workspace_embedded_child_tool_keeps_parent_payload_pending(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="parent",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        child_data = (data + 10.0).rename("embedded")
        child_uid = manager.add_childtool(
            _WorkspaceSweepChildTool(child_data), 0, show=False
        )

        fname = tmp_path / "pending-parent-embedded-child.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "r") as h5_file:
            assert (
                "tool_data_references"
                not in h5_file[f"0/childtools/{child_uid}/tool"].attrs
            )

        materialize_calls = 0
        original_materialize = manager._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialize_calls
            materialize_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _count_materialize,
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert materialize_calls == 0
        assert wrapper.pending_workspace_memory_payload is not None
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, _WorkspaceSweepChildTool)
        assert loaded_child.tool_data.name == child_data.name
        assert loaded_child.tool_data.dims == child_data.dims
        np.testing.assert_array_equal(loaded_child.tool_data.values, child_data.values)


def test_manager_workspace_replacing_pending_memory_data_clears_pending_payload(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="original",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "replace-pending-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        replacement = xr.DataArray(
            np.full((5, 5), 42.0),
            dims=["x", "y"],
            name="replacement",
        )
        tool = manager.get_imagetool(0)
        tool.slicer_area.replace_source_data(replacement)

        assert wrapper.pending_workspace_memory_payload is None
        assert wrapper.uid in manager._workspace_state.dirty_data
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert wrapper.pending_workspace_memory_payload is None

        with h5py.File(fname, "r") as h5_file:
            group = h5_file["0/imagetool"]
            assert group.attrs["itool_name"] == "replacement"
            np.testing.assert_array_equal(
                group[manager_workspace_io._ITOOL_DATA_NAME][...],
                replacement.values,
            )

        manager.show_imagetool(0)
        qtbot.wait_until(lambda: manager.get_imagetool(0).isVisible())
        loaded = manager.get_imagetool(0).slicer_area
        np.testing.assert_array_equal(loaded._data.values, replacement.values)


def test_manager_workspace_attr_update_keeps_pending_hidden_memory_unmaterialized(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="original",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "save-pending-hidden-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        materialize_calls = 0
        original_materialize = manager._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialize_calls
            materialize_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _count_materialize,
        )

        wrapper.name = "renamed"
        update = manager._workspace_attr_update_snapshot(wrapper.uid)

        assert update is not None
        payload_path, attrs, (_node_path, constructor) = update
        assert materialize_calls == 0
        assert wrapper.pending_workspace_memory_payload is not None
        assert attrs["itool_name"] == "renamed"
        assert payload_path == "0/imagetool"
        assert constructor == {}

        assert _request_workspace_save_and_wait(qtbot, manager)
        assert materialize_calls == 0
        assert wrapper.pending_workspace_memory_payload is not None

        with h5py.File(fname, "r") as h5_file:
            group = h5_file["0/imagetool"]
            assert group.attrs["itool_name"] == "renamed"
            np.testing.assert_array_equal(
                group[manager_workspace_io._ITOOL_DATA_NAME][...],
                data.values,
            )


def test_manager_network_full_save_keeps_pending_hidden_memory_unmaterialized(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            name="original",
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "network-full-save-pending-memory.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "network full save should not materialize hidden memory payloads"
            )

        monkeypatch.setattr(
            manager,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )
        monkeypatch.setattr(
            manager_workspace,
            "_workspace_path_is_likely_network_path",
            lambda _path: True,
        )

        wrapper.name = "network-renamed"
        manager._workspace_controller._write_full_workspace_file(fname)
        assert wrapper.pending_workspace_memory_payload is not None

        with h5py.File(fname, "r") as h5_file:
            group = h5_file["0/imagetool"]
            assert group.attrs["itool_name"] == "network-renamed"
            np.testing.assert_array_equal(
                group[manager_workspace_io._ITOOL_DATA_NAME][...],
                data.values,
            )


def test_manager_workspace_load_uses_h5py_fast_path(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "h5py-fast-load.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        monkeypatch.setattr(
            manager_xarray,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "simple v4 load should not open the workspace DataTree"
            ),
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        node = manager._node_for_target(0)
        if node.pending_workspace_memory_payload is not None:
            assert manager._materialize_pending_workspace_payload(node)
        loaded = manager.get_imagetool(0).slicer_area
        assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
        assert not loaded.data_chunked
        assert not loaded.data_file_backed
        np.testing.assert_array_equal(loaded._data.values, data.values)


def test_manager_h5py_workspace_load_defers_hidden_imagetool_refresh_and_profiles(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "profiled-h5py-load.itws"
        manager._save_workspace_document(fname, force_full=True)
        root_attrs = manager_workspace._read_workspace_root_attrs_h5py(fname)
        _, _, manifest = manager_workspace._workspace_file_metadata_from_attrs(
            root_attrs
        )
        assert manifest is not None

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        original_restore_state = imagetool_viewer.ImageSlicerArea._restore_state
        original_refresh_all = imagetool_viewer.ImageSlicerArea.refresh_all
        restoring_state: list[imagetool_viewer.ImageSlicerArea] = []
        refresh_during_restore: list[imagetool_viewer.ImageSlicerArea] = []

        def _record_restore_state(self, *args, **kwargs):
            restoring_state.append(self)
            try:
                return original_restore_state(self, *args, **kwargs)
            finally:
                restoring_state.pop()

        def _record_refresh_all(self):
            if restoring_state and restoring_state[-1] is self:
                refresh_during_restore.append(self)
            return original_refresh_all(self)

        monkeypatch.setattr(
            imagetool_viewer.ImageSlicerArea,
            "_restore_state",
            _record_restore_state,
        )
        monkeypatch.setattr(
            imagetool_viewer.ImageSlicerArea,
            "refresh_all",
            _record_refresh_all,
        )

        profiler = manager_workspace_io._WorkspaceLoadProfiler(fname)
        assert manager._from_h5py_workspace_file(
            fname,
            manifest,
            replace=True,
            mark_dirty=False,
            profiler=profiler,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        loaded = manager.get_imagetool(0).slicer_area
        assert loaded not in refresh_during_restore

        assert profiler._durations["imagetool widget restore"] > 0.0
        assert profiler._durations["imagetool manager registration"] > 0.0
        assert "imagetool state restore: layout" in profiler._durations
        assert "imagetool state refresh" not in profiler._durations


def test_manager_h5py_workspace_load_defers_hidden_secondary_plot_widgets(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        for index in range(4):
            manager.add_imagetool(
                erlab.interactive.imagetool.ImageTool(
                    _workspace_sweep_data(f"window_{index}"),
                    _in_manager=True,
                ),
                show=False,
            )

        fname = tmp_path / "lazy-secondary-plots.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        constructed_display_axes: list[tuple[int, ...] | tuple[int, int]] = []
        original_init = imagetool_plot_items.ItoolGraphicsLayoutWidget.__init__

        def _record_graphics_layout_init(self, *args, **kwargs) -> None:
            constructed_display_axes.append(tuple(kwargs["display_axis"]))
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(
            imagetool_plot_items.ItoolGraphicsLayoutWidget,
            "__init__",
            _record_graphics_layout_init,
        )

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 4, timeout=5000)

        loaded_areas = []
        for index in range(manager.ntools):
            wrapper = manager._tool_graph.root_wrappers[index]
            assert wrapper.pending_workspace_memory_payload is not None
            assert wrapper.imagetool is None
        assert loaded_areas == []
        assert constructed_display_axes == []

        manager.show_imagetool(0)
        first_tool = manager.get_imagetool(0)
        qtbot.wait_until(
            lambda: first_tool.slicer_area._secondary_plots_materialized,
            timeout=5000,
        )
        first_area = first_tool.slicer_area
        assert first_area._plot_widgets_constructed == 8
        assert first_area._secondary_plot_materialization_duration > 0.0
        assert len(first_area.axes) == 8
        assert constructed_display_axes == [
            (0, 1),
            (0,),
            (0, 2),
            (3,),
            (2,),
            (3, 2),
            (2, 1),
            (1,),
        ]


def test_hidden_2d_workspace_preview_does_not_build_invalid_secondary_plots(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = erlab.interactive.imagetool.ImageTool(data, _in_manager=True)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "hidden-2d-preview.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        constructed_display_axes: list[tuple[int, ...] | tuple[int, int]] = []
        original_init = imagetool_plot_items.ItoolGraphicsLayoutWidget.__init__

        def _record_graphics_layout_init(self, *args, **kwargs) -> None:
            constructed_display_axes.append(tuple(kwargs["display_axis"]))
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(
            imagetool_plot_items.ItoolGraphicsLayoutWidget,
            "__init__",
            _record_graphics_layout_init,
        )

        assert manager._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_node = manager._node_for_target(0)

        assert loaded_node.imagetool is None
        assert constructed_display_axes == []

        ratio, pixmap = manager_wrapper._preview_image_for_node(loaded_node)
        assert isinstance(ratio, float)
        assert isinstance(pixmap, QtGui.QPixmap)
        assert np.isnan(ratio)
        assert pixmap.isNull()
        assert loaded_node.pending_workspace_memory_payload is not None


def test_manager_workspace_load_preserves_transposed_inverted_axis_limits(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(125, dtype=np.float64).reshape((5, 5, 5)),
            dims=["x", "y", "z"],
            coords={"x": np.arange(5.0), "y": np.arange(5.0), "z": np.arange(5.0)},
        )
        expected_limits = {"x": [1.0, 3.0], "y": [0.0, 2.0], "z": [2.0, 4.0]}

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        root.slicer_area.set_manual_limits(expected_limits)
        root.slicer_area.set_axis_inverted("x", True)
        root.slicer_area.transpose_main_image()
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "transposed-inverted-limits.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        monkeypatch.setattr(
            manager_xarray,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "simple v4 load should not open the workspace DataTree"
            ),
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        node = manager._node_for_target(0)
        if node.pending_workspace_memory_payload is not None:
            assert manager._materialize_pending_workspace_payload(node)
        loaded = manager.get_imagetool(0).slicer_area
        assert loaded.data.dims == ("y", "x", "z")
        assert loaded.manual_limits == expected_limits
        _assert_manual_limits_view_ranges(loaded, expected_limits)
        assert loaded.axis_inversions == {"x": True}
        assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
        np.testing.assert_array_equal(
            loaded._data.transpose(*data.dims).values, data.values
        )


def test_manager_workspace_load_h5py_fast_path_falls_back_per_payload(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25, dtype=np.float64).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5), "label": "sample"},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "h5py-group-fallback.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        monkeypatch.setattr(
            manager_xarray,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "unsupported payload should fall back by group, not by whole tree"
            ),
        )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded = manager.get_imagetool(0).slicer_area
        assert manager_xarray.dataarray_is_numpy_backed(loaded._data)
        assert not loaded.data_chunked
        assert not loaded.data_file_backed
        assert loaded._data.coords["label"].item() == "sample"


def test_manager_workspace_rejects_external_xarray_reader_for_active_workspace(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-state.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        lazy = _open_external_lazy_hdf5_imagetool_data(fname)
        manager.get_imagetool(0).slicer_area.set_data(lazy, auto_compute=False)
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0
        manager._mark_workspace_clean()

        errors: list[str] = []
        monkeypatch.setattr(manager, "_show_workspace_save_worker_error", errors.append)

        manager.rename_imagetool(0, "lazy state")
        assert not _request_workspace_save_and_wait(qtbot, manager)
        assert errors
        with contextlib.suppress(Exception):
            lazy.close()


def test_manager_workspace_lazy_data_delta_save_uses_pending_group_before_replacing(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-data.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        manager.get_imagetool(0).slicer_area.replace_source_data(
            replacement, auto_compute=False
        )
        assert _request_workspace_save_and_wait(qtbot, manager)
        assert list(tmp_path.glob("lazy-data.itws.delta-*")) == []

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert saved[0, 0] == 10


def test_manager_workspace_same_file_lazy_data_delta_save_does_not_deadlock(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(512 * 512, dtype=np.float64).reshape((512, 512)),
            dims=["x", "y"],
            coords={"x": np.arange(512), "y": np.arange(512)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "same-file-lazy.itws"
        manager._save_workspace_document(fname, force_full=True)
        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        manager._rebind_workspace_backed_imagetools(fname, targets=[0], chunks={})
        assert manager.get_imagetool(0).slicer_area.data_chunked
        manager.get_imagetool(0).slicer_area._set_chunks({"x": 128, "y": 64})

        uid = manager._tool_graph.root_wrappers[0].uid
        manager._mark_node_data_dirty(uid)
        assert _request_workspace_save_and_wait(qtbot, manager)

        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert saved.chunks == (128, 64)


def test_manager_workspace_lazy_data_delta_pending_failure_preserves_old_group(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "lazy-failure.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        replacement = data.copy(deep=True)
        replacement.data = np.asarray(replacement.data) + 10
        root.slicer_area.replace_source_data(replacement, auto_compute=False)

        def _write_partial_pending_then_raise(
            fname: str | os.PathLike[str],
            _constructor: Mapping[str, xr.Dataset],
            _group_path: str,
            pending_path: str,
        ) -> None:
            with manager_workspace._open_workspace_h5_file_for_update(fname) as h5_file:
                h5_file.create_group(pending_path)
            raise RuntimeError("pending write failed")

        monkeypatch.setattr(
            manager_workspace,
            "_write_workspace_constructor_groups_to_pending",
            _write_partial_pending_then_raise,
        )
        monkeypatch.setattr(
            manager, "_show_workspace_save_worker_error", lambda *args: None
        )

        assert not _request_workspace_save_and_wait(qtbot, manager)
        import h5py

        with h5py.File(fname, "r") as h5_file:
            saved = h5_file["0/imagetool"][manager_workspace_io._ITOOL_DATA_NAME]
            assert saved[0, 0] == 0
            assert not any(
                manager_workspace._is_workspace_internal_group_name(name)
                for name in h5_file
            )


def test_manager_workspace_stale_pending_groups_do_not_poison_open_or_save(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "stale-pending.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager.rename_imagetool(0, "cleaned")
        assert _request_workspace_save_and_wait(qtbot, manager)
        with h5py.File(fname, "r") as h5_file:
            assert not any(
                manager_workspace._is_workspace_internal_group_name(name)
                for name in h5_file
            )


def test_manager_workspace_load_dialog_skips_stale_internal_groups(
    qtbot,
    accept_dialog,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "stale-dialog.itws"
        manager._save_workspace_document(fname, force_full=True)
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{manager_workspace._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        tree = manager_xarray.open_workspace_datatree(fname, chunks="auto")
        accept_dialog(
            lambda: manager._from_datatree(
                tree,
                replace=True,
                mark_dirty=False,
                select=True,
            )
        )

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_workspace_replace_load_failure_restores_previous_workspace(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        current_fname = tmp_path / "current.itws"
        manager._save_workspace_document(current_fname, force_full=True)
        manager._adopt_workspace_path(current_fname)
        manager._mark_workspace_clean()

        broken_fname = tmp_path / "broken.itws"
        with h5py.File(broken_fname, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
            h5_file.create_group("0")

        with pytest.raises(ValueError, match="No workspace windows") as exc_info:
            manager._load_workspace_file(
                broken_fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "Workspace node" in str(exc_info.value.__cause__)

        assert manager.workspace_path == str(current_fname.resolve())
        assert manager.ntools == 1
        xarray.testing.assert_equal(manager.get_imagetool(0).slicer_area._data, data)
        assert not manager.is_workspace_modified


def test_manager_workspace_replace_load_failure_uses_clean_file_backup(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        current_fname = tmp_path / "current-clean.itws"
        manager._save_workspace_document(current_fname, force_full=True)
        manager._adopt_workspace_path(current_fname)
        manager._mark_workspace_clean()

        def _fail_to_datatree(*_args, **_kwargs):
            raise AssertionError("clean associated replace load should use file backup")

        monkeypatch.setattr(manager, "_to_datatree", _fail_to_datatree)

        broken_fname = tmp_path / "broken-clean.itws"
        with h5py.File(broken_fname, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
            h5_file.create_group("0")

        with pytest.raises(ValueError, match="No workspace windows"):
            manager._load_workspace_file(
                broken_fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )

        assert manager.workspace_path == str(current_fname.resolve())
        assert manager.ntools == 1
        xarray.testing.assert_equal(manager.get_imagetool(0).slicer_area._data, data)
        assert not manager.is_workspace_modified


def test_manager_workspace_replace_load_failure_keeps_dirty_memory_backup(
    qtbot,
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        uid = manager._tool_graph.root_wrappers[0].uid

        current_fname = tmp_path / "current-dirty.itws"
        manager._save_workspace_document(current_fname, force_full=True)
        manager._adopt_workspace_path(current_fname)
        manager._mark_workspace_clean()
        manager._mark_node_state_dirty(uid)
        assert manager.is_workspace_modified

        to_datatree_calls = 0
        original_to_datatree = manager._to_datatree

        def _record_to_datatree(*args, **kwargs):
            nonlocal to_datatree_calls
            to_datatree_calls += 1
            return original_to_datatree(*args, **kwargs)

        monkeypatch.setattr(manager, "_to_datatree", _record_to_datatree)

        broken_fname = tmp_path / "broken-dirty.itws"
        with h5py.File(broken_fname, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
            h5_file.create_group("0")

        with pytest.raises(ValueError, match="No workspace windows"):
            manager._load_workspace_file(
                broken_fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )

        assert to_datatree_calls == 1
        assert manager.workspace_path == str(current_fname.resolve())
        assert manager.ntools == 1
        xarray.testing.assert_equal(manager.get_imagetool(0).slicer_area._data, data)
        assert manager.is_workspace_modified


def test_manager_workspace_load_visible_windows_stays_clean_after_events(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=True)
        qtbot.wait_until(root.isVisible)

        fname = tmp_path / "visible.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        for _ in range(3):
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents()

        assert not manager.is_workspace_modified


def test_manager_workspace_roundtrip_preserves_controls_visibility(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        manager._mark_workspace_clean()
        assert not manager.is_workspace_modified

        root.mnb.action_dict["toggleControlsAct"].trigger()
        assert not root.controls_visible
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)

        fname = tmp_path / "controls.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        assert not manager.is_workspace_modified

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        restored = manager.get_imagetool(0)
        assert not restored.controls_visible
        assert not restored.mnb.action_dict["toggleControlsAct"].isChecked()
        assert restored.slicer_area.state["controls_visible"] is False
        assert not manager.is_workspace_modified


def test_manager_workspace_delta_save_persists_geometry_changes(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    import h5py

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        fname = tmp_path / "geometry.itws"
        manager._save_workspace_document(fname, force_full=True)
        manager._adopt_workspace_path(fname)
        manager._mark_workspace_clean()

        root.setGeometry(12, 34, 321, 234)
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)
        expected_rect = tuple(root.geometry().getRect())

        assert _request_workspace_save_and_wait(qtbot, manager)
        with h5py.File(fname, "r") as h5_file:
            saved_state = json.loads(h5_file["0/imagetool"].attrs["itool_window_state"])
            saved_rect = tuple(int(value) for value in saved_state["rect"])
        assert saved_rect == expected_rect


def test_manager_workspace_roundtrip_goldtool_child(
    qtbot,
    monkeypatch,
    gold,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(gold, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = GoldTool(gold.copy(deep=True), data_name="gold_input")
        child.set_source_binding(erlab.interactive.imagetool.provenance.full_data())
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)

        expected_status = child.tool_status.model_copy(deep=True)
        expected_corrected = child.corrected.copy(deep=True)
        expected_source_spec = child.source_spec
        child.open_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert (
            tree[f"0/childtools/{child_uid}/tool"].attrs["manager_node_uid"]
            == child_uid
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._tool_graph.root_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, GoldTool)
        assert loaded_child.source_spec == expected_source_spec
        assert loaded_child.tool_status == expected_status
        xr.testing.assert_identical(loaded_child.corrected, expected_corrected)
        loaded_child_node = manager._child_node(child_uid)
        assert loaded_child_node._childtool_indices == [output_uid]
        loaded_output_node = manager._child_node(output_uid)
        assert loaded_output_node.output_id == "goldtool.corrected"
        assert loaded_output_node.source_spec is None
        assert loaded_output_node.provenance_spec is not None
        assert loaded_output_node.provenance_spec.active_name == "corrected"
        xr.testing.assert_identical(fetch(output_uid), expected_corrected)

        monkeypatch.setattr(
            loaded_child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        loaded_child.open_itool()
        assert loaded_child_node._childtool_indices == [output_uid]


def test_manager_workspace_roundtrip_dtool_child(
    qtbot,
    monkeypatch,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, DerivativeTool)

        expected_status = child.tool_status.model_copy(deep=True)
        expected_result = child.result.T.copy(deep=True)
        expected_source_spec = child.source_spec
        child.open_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert (
            tree[f"0/childtools/{child_uid}/tool"].attrs["manager_node_uid"]
            == child_uid
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._tool_graph.root_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, DerivativeTool)
        assert loaded_child.source_spec == expected_source_spec
        assert loaded_child.tool_status == expected_status
        xr.testing.assert_identical(loaded_child.result.T, expected_result)
        loaded_child_node = manager._child_node(child_uid)
        assert loaded_child_node._childtool_indices == [output_uid]
        loaded_output_node = manager._child_node(output_uid)
        assert loaded_output_node.output_id == "dtool.result"
        assert loaded_output_node.source_spec is None
        assert loaded_output_node.provenance_spec is not None
        assert loaded_output_node.provenance_spec.active_name == "result"
        xr.testing.assert_identical(fetch(output_uid), expected_result)

        monkeypatch.setattr(
            loaded_child, "_prompt_existing_output_imagetool", lambda: "update"
        )
        loaded_child.open_itool()
        assert loaded_child_node._childtool_indices == [output_uid]


def test_manager_workspace_roundtrip_fit1d_child(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        t = np.linspace(0.0, 4.0, 25)
        data = xr.DataArray(
            3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
        )
        params = exp_decay_model.make_params(n0=2.0, tau=1.0)
        child = erlab.interactive.ftool(
            data, model=exp_decay_model, params=params, execute=False
        )
        assert isinstance(child, Fit1DTool)
        child_uid = manager.add_childtool(child, 0, show=False)

        assert child._run_fit()
        qtbot.wait_until(lambda: child._last_result_ds is not None, timeout=10000)
        _wait_for_fit_idle(qtbot, child)
        assert child._last_result_ds is not None
        expected_fit_ds = child._last_result_ds.copy(deep=True)
        expected_status = child.tool_status.model_dump()

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit1DTool)
        assert loaded_child._last_result_ds is not None
        assert_fit_result_dataset_equivalent(
            loaded_child._last_result_ds, expected_fit_ds
        )
        assert loaded_child.tool_status.model_dump() == expected_status
        assert loaded_child._fit_is_current
        assert loaded_child.save_button.isEnabled()
        assert loaded_child.copy_button.isEnabled()

        warnings: list[tuple[str, str]] = []
        monkeypatch.setattr(
            loaded_child,
            "_show_warning",
            lambda title, text: warnings.append((title, text)),
        )
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        assert "modelfit" in copied
        assert not warnings


def test_manager_workspace_roundtrip_fit2d_child(
    qtbot,
    monkeypatch,
    exp_decay_model,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child_uid, child = make_fit2d_child(manager, 0, exp_decay_model)
        child.timeout_spin.setValue(30.0)
        child.nfev_spin.setValue(0)
        child.y_index_spin.setValue(child.y_min_spin.value())
        child._run_fit_2d("up")
        qtbot.wait_until(
            lambda: all(ds is not None for ds in child._result_ds_full),
            timeout=10000,
        )
        _wait_for_fit_idle(qtbot, child)
        expected_results = [
            None if ds is None else ds.copy(deep=True) for ds in child._result_ds_full
        ]
        expected_status = child.tool_status.model_dump()

        tree = manager._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._load_workspace_node(typing.cast("xr.DataTree", node))

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit2DTool)
        assert all(ds is not None for ds in loaded_child._result_ds_full)
        assert_fit_result_list_equivalent(
            loaded_child._result_ds_full, expected_results
        )
        assert loaded_child.tool_status.model_dump() == expected_status
        assert loaded_child._fit_is_current
        assert loaded_child.copy_full_button.isEnabled()
        assert loaded_child.save_full_button.isEnabled()

        warnings: list[tuple[str, str]] = []
        monkeypatch.setattr(
            loaded_child,
            "_show_warning",
            lambda title, text: warnings.append((title, text)),
        )
        manager.tree_view.clearSelection()
        select_child_tool(manager, child_uid)
        manager._update_info(uid=child_uid)
        copied = copy_full_code_for_uid(monkeypatch, manager, child_uid)
        assert "modelfit" in copied
        assert not warnings


def test_manager_workspace_roundtrip_fit2d_child_with_spaced_axis(
    qtbot,
    exp_decay_model,
    test_data,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(test_data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        t = np.linspace(0.0, 4.0, 25)
        motor = np.arange(3.0)
        data = xr.DataArray(
            np.stack(
                [((1.0 + 0.5 * idx) * np.exp(-t / 2.0)) for idx in motor],
                axis=0,
            ),
            dims=("Fake Motor", "t"),
            coords={
                "Fake Motor": motor,
                "t": t,
                "Sample Motor": ("Fake Motor", motor + 10.0),
            },
            name="decay2d",
        )
        params = exp_decay_model.make_params(n0=1.0, tau=1.0)
        child = erlab.interactive.ftool(
            data, model=exp_decay_model, params=params, execute=False
        )
        assert isinstance(child, Fit2DTool)
        child_uid = manager.add_childtool(child, 0, show=False)
        child.timeout_spin.setValue(30.0)
        child.nfev_spin.setValue(0)
        child.y_index_spin.setValue(child.y_min_spin.value())
        child._run_fit_2d("up")
        qtbot.wait_until(
            lambda: all(ds is not None for ds in child._result_ds_full),
            timeout=10000,
        )
        _wait_for_fit_idle(qtbot, child)
        assert child.current_provenance_spec() is not None

        fname = tmp_path / "fit2d-spaced-axis.itws"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager._save_workspace_document(fname, force_full=True)

        assert not any("space in its name" in str(item.message) for item in caught)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        assert manager._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_child = manager.get_childtool(child_uid)
        assert isinstance(loaded_child, Fit2DTool)
        assert loaded_child.tool_data.dims[0] == "Fake Motor"
        xr.testing.assert_equal(
            loaded_child.tool_data.coords["Fake Motor"], data.coords["Fake Motor"]
        )
        xr.testing.assert_equal(
            loaded_child.tool_data.coords["Sample Motor"], data.coords["Sample Motor"]
        )
        assert loaded_child.current_provenance_spec() is not None


def test_manager_workspace_roundtrip_recursive_nested_imagetools(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(60).reshape((3, 4, 5)).astype(float),
        dims=["x", "y", "z"],
        coords={"x": np.arange(3), "y": np.arange(4), "z": np.arange(5)},
        name="scan",
    )

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        itool(data, link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        def _nest_average(dialog) -> None:
            dialog.dim_checks["x"].setChecked(True)
            set_transform_launch_mode(dialog, "nest")

        accept_dialog(manager.get_imagetool(0).mnb._average, pre_call=_nest_average)

        root_wrapper = manager._tool_graph.root_wrappers[0]
        qtbot.wait_until(
            lambda: len(root_wrapper._childtool_indices) == 1, timeout=5000
        )
        child_uid = root_wrapper._childtool_indices[0]
        child_node = manager._child_node(child_uid)
        child_spec = child_node.source_spec

        child_tool = manager.get_imagetool(child_uid)
        child_tool.slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        tool_uid = child_node._childtool_indices[0]

        tree = manager._to_datatree()
        assert tree.attrs["imagetool_workspace_schema_version"] == 3
        assert (
            tree[f"0/childtools/{child_uid}/imagetool"].attrs["manager_node_uid"]
            == child_uid
        )
        assert (
            tree[f"0/childtools/{child_uid}/childtools/{tool_uid}/tool"].attrs[
                "manager_node_uid"
            ]
            == tool_uid
        )

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            filename = pathlib.Path(tmp_dir_name) / "workspace.itws"
            tree.to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)

            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            loaded = manager_xarray.open_workspace_datatree(filename, chunks="auto")
            try:
                assert manager._is_datatree_workspace(loaded)
                assert loaded.attrs["imagetool_workspace_schema_version"] == 3
                for node in loaded.values():
                    manager._load_workspace_node(typing.cast("xr.DataTree", node))
            finally:
                loaded.close()

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        loaded_root = manager._tool_graph.root_wrappers[0]
        assert loaded_root._childtool_indices == [child_uid]

        loaded_child = manager._child_node(child_uid)
        assert loaded_child.source_spec == child_spec
        assert loaded_child._childtool_indices == [tool_uid]

        updated = data.copy(deep=True)
        updated.data = np.asarray(updated.data) * 4

        with qtbot.wait_signal(manager._sigDataReplaced):
            replace_data(0, updated)

        qtbot.wait_until(lambda: loaded_child.source_state == "stale", timeout=5000)
        assert loaded_child._update_from_parent_source() is True
        xr.testing.assert_identical(
            manager.get_imagetool(child_uid).slicer_area._data.rename(None),
            updated.qsel.mean("x").rename(None),
        )


def test_manager_workspace_rejects_h5_workspace_file(
    qtbot,
    datadir,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    legacy_workspace = tmp_path / "manager_workspace_legacy.h5"
    legacy_workspace.write_bytes((datadir / "manager_workspace_legacy.h5").read_bytes())
    original_legacy_bytes = legacy_workspace.read_bytes()

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        with pytest.raises(ValueError, match=r"\.itws"):
            manager._load_workspace_file(
                legacy_workspace,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )

        assert manager.ntools == 0
        assert legacy_workspace.read_bytes() == original_legacy_bytes
