import base64
import contextlib
import datetime
import json
import logging
import pathlib
import types
import typing
import weakref
from collections.abc import Callable, Mapping

import h5py
import numpy as np
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
import erlab.interactive.imagetool.manager._lineage as manager_lineage
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._modelview as manager_modelview
import erlab.interactive.imagetool.manager._provenance_edit as manager_provenance_edit
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._loading as workspace_loading
import erlab.interactive.imagetool.manager._workspace._pending as workspace_pending
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
import erlab.interactive.imagetool.plot_items as imagetool_plot_items
import erlab.interactive.imagetool.viewer as imagetool_viewer
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    compose_display_provenance,
    full_data,
    script,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    AverageOperation,
    BoxcarFilterOperation,
    GaussianFilterOperation,
    ImageToolSelectionSourceBinding,
    RenameOperation,
    SortCoordOrderOperation,
    SqueezeOperation,
)
from erlab.interactive.imagetool.manager import ImageToolManager, replace_data
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ChooseFromWorkspaceManifestDialog,
)
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    adopt_workspace_path,
    select_child_tool,
    select_tools,
    trigger_menu_action,
)

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool.manager._workspace import (
        _controller as workspace_controller,
    )
from tests.interactive.imagetool.manager.workspace._support import (
    _AddedTimeChildTool,
    _open_external_file_backed_hdf5_imagetool_data,
    _request_workspace_save_and_wait,
    _transaction_test_root_attrs,
    _workspace_test_file_spec,
    _WorkspaceManagerReferenceFigureTool,
    _WorkspaceSweepChildTool,
    _WorkspaceSweepFigureTool,
    _WorkspaceSweepToolState,
)


def _workspace_sweep_json(value: typing.Any) -> typing.Any:
    return json.loads(json.dumps(value))


def _workspace_sweep_spec_payload(
    spec: ToolProvenanceSpec | None,
) -> dict[str, typing.Any] | None:
    if spec is None:
        return None
    return typing.cast(
        "dict[str, typing.Any]",
        spec.model_dump(mode="json", exclude_none=True),
    )


def _workspace_sweep_binding_payload(
    binding: ImageToolSelectionSourceBinding | None,
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
        FileDataSelection(kind="dataarray"),
    )
    area.apply_filter_operation(GaussianFilterOperation(sigma={"x": 0.5}), update=False)


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
        "source_snapshot_token": node.source_snapshot_token,
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
    saver = controller.saving
    return {
        "root_order": list(
            manager._workspace_controller.saving._workspace_root_indices()
        ),
        "figure_uids": list(manager._tool_graph.figure_uids),
        "loader_state": saver._workspace_loader_state_snapshot(),
        "standalone_apps": saver._workspace_standalone_apps_snapshot(),
        "nodes": {
            uid: _workspace_sweep_node_snapshot(node)
            for uid, node in sorted(manager._tool_graph.nodes.items())
        },
    }


def _workspace_sweep_data_items(
    manager: ImageToolManager,
) -> dict[tuple[str, str], xr.DataArray]:
    data_items: dict[tuple[str, str], xr.DataArray] = {}
    loader = manager._workspace_controller.loading
    for uid, node in sorted(manager._tool_graph.nodes.items()):
        if node.pending_workspace_payload is not None:
            assert loader.pending._materialize_pending_workspace_payload(node)
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )

        assert manager._tool_graph.root_wrappers[root_index].created_time == root_added
        assert manager._child_node(child_uid).created_time == child_added
        assert manager._child_node(tool_uid).created_time == tool_added


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        with h5py.File(fname, "a") as h5_file:
            state = json.loads(h5_file["0/imagetool"].attrs["itool_state"])
            state["color"]["cmap"] = missing
            h5_file["0/imagetool"].attrs["itool_state"] = json.dumps(state)

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        with h5py.File(fname, "a") as h5_file:
            if saved_attr is None:
                del h5_file["0/imagetool"].attrs["manager_node_added_at"]
            else:
                h5_file["0/imagetool"].attrs["manager_node_added_at"] = saved_attr

        assert manager._workspace_controller.loading._load_workspace_file(
            fname,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )

        loaded = manager._tool_graph.root_wrappers[0].created_time
        assert loaded.tzinfo is not None
        assert loaded.utcoffset() is not None


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
        expected_layout = (
            manager._workspace_controller.saving._workspace_layout_snapshot()
        )
        assert "window_state" in expected_layout
        assert "geometry" not in expected_layout
        assert expected_layout["window_state"]["geometry"]
        expected_size = manager.size()
        expected_main_sizes = manager.main_splitter.sizes()
        expected_right_sizes = manager.right_splitter.sizes()

        fname = tmp_path / "manager-layout.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest["manager_layout"] == expected_layout

        manager.resize(480, 500)
        manager.main_splitter.setSizes([120, 360])
        manager.right_splitter.setSizes([120, 250, 80])
        assert (
            manager._workspace_controller.saving._workspace_layout_snapshot()
            != expected_layout
        )

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            import_fname, force_full=True
        )

        manager.resize(480, 500)
        manager.main_splitter.setSizes([120, 360])
        manager.right_splitter.setSizes([120, 250, 80])
        current_layout = (
            manager._workspace_controller.saving._workspace_layout_snapshot()
        )

        import h5py

        with h5py.File(import_fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert manager._workspace_controller.loading._from_h5py_workspace_file(
            import_fname, manifest, replace=False, mark_dirty=True
        )
        assert (
            manager._workspace_controller.saving._workspace_layout_snapshot()
            == current_layout
        )

        tree = workspace_arrays.open_workspace_datatree(import_fname, chunks=None)
        assert manager._workspace_controller.loading._from_datatree(
            tree,
            replace=False,
            mark_dirty=True,
            select=False,
            workspace_file_path=import_fname,
        )
        assert (
            manager._workspace_controller.saving._workspace_layout_snapshot()
            == current_layout
        )


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        expected_ptable_size = ptable.size()

        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
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

        assert manager._workspace_controller.loading._load_workspace_file(
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
    root_source_marker = "snapshot-root-source"

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
            source_snapshot_token=root_source_marker,
            created_time=root_created,
            watched_var=("root_data", "watch-root-uid"),
            watched_workspace_link_id=manager._workspace_state.link_id,
            watched_source_label="notebook-a",
            watched_source_uid="kernel-a",
            watched_connected=False,
            source_input_ndim=root_data.ndim,
            provenance_spec=root_provenance,
            source_spec=full_data(RenameOperation(name="root-live")),
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
        child_binding = ImageToolSelectionSourceBinding(
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
            provenance_spec=script(
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
            provenance_spec=full_data(RenameOperation(name="nested-image")),
        )

        tool_binding = ImageToolSelectionSourceBinding(
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
            workspace_format.WorkspaceLoaderState(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        expected_snapshot = _workspace_sweep_runtime_snapshot(manager)
        expected_data_items = _workspace_sweep_data_items(manager)

        root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(fname)
        manifest = workspace_format._workspace_manifest_from_attrs(root_attrs)
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
            "manager_node_source_snapshot_token",
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
            GaussianFilterOperation(sigma={"x": 0.5}).model_dump(mode="json")
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
                    workspace_arrays._TOOL_DATA_BLOB_NAME_ATTR
                ]
                == extra_data.name
            )

        assert manager._workspace_controller.loading._load_workspace_file(
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
            workspace_format.WorkspaceLoaderState(
                explorer_loader_kwargs_by_name=explorer_kwargs,
                explorer_loader_extensions_by_name=explorer_extensions,
            )
        )

        fname = tmp_path / "loader-no-explorer.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert manifest["loader_state"]["explorer_loader_kwargs_by_name"] == (
            explorer_kwargs
        )
        assert "explorer" not in manifest["standalone_apps"]["apps"]

        manager._workspace_controller._loader_state = (
            workspace_format.WorkspaceLoaderState()
        )
        manager._recent_directory = None
        manager._recent_name_filter = None
        manager._recent_loader_kwargs_by_filter.clear()

        assert manager._workspace_controller.loading._load_workspace_file(
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
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
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
        loader = controller.loading
        saver = controller.saving

        loader._restore_workspace_loader_state({"loader_state": []})
        loader._restore_workspace_loader_state(
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
        loader._restore_workspace_loader_state(
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
        snapshot = saver._workspace_standalone_apps_snapshot()
        assert snapshot["apps"]["ptable"]["selected_atomic_numbers"] == [1]

        loader._restore_standalone_apps_state({"standalone_apps": []})
        loader._restore_standalone_apps_state({"standalone_apps": {"apps": []}})
        loader._restore_standalone_apps_state(
            {"standalone_apps": {"apps": {"missing": {}}}}
        )

        manager.show_ptable()
        ptable = manager.ptable_window
        assert ptable.isVisible()
        hidden_ptable_state = ptable.workspace_state_payload()
        hidden_ptable_state["window_state"]["visible"] = False
        loader._restore_standalone_apps_state(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        ptable.hv_edit.setText("30")
        ptable._set_selection_state(
            [1], current_atomic_number=1, anchor_atomic_number=1
        )
        ptable._refresh_window_state(ensure_visible=False)

        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
        assert manager._workspace_controller.loading._from_h5py_workspace_file(
            fname, manifest, replace=False, mark_dirty=True
        )
        assert ptable.hv_edit.text() == "30"
        assert ptable.selected_atomic_numbers == (1,)

        tree = workspace_arrays.open_workspace_datatree(fname, chunks=None)
        assert manager._workspace_controller.loading._from_datatree(
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
        layout = manager._workspace_controller.saving._workspace_layout_snapshot()

        manager._workspace_controller.loading._restore_workspace_layout(None)
        manager._workspace_controller.loading._restore_workspace_layout({})
        manager._workspace_controller.loading._restore_workspace_layout(
            {"manager_layout": "invalid"}
        )
        manager._workspace_controller.loading._restore_workspace_layout(
            {
                "manager_layout": {
                    "window_state": {"geometry": ""},
                    "main_splitter": "",
                    "right_splitter": "",
                }
            }
        )
        assert (
            manager._workspace_controller.saving._workspace_layout_snapshot() == layout
        )


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("link restore should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("action refresh should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("reload availability should not materialize hidden memory data")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        fname = tmp_path / "pending-file-source-reload.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("reload availability should not materialize hidden memory data")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
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
            source_spec=full_data(AverageOperation(dims=("alpha",))),
            source_state="stale",
        )
        child.hide()

        fname = tmp_path / "pending-file-source-child-reload.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        child_node = manager._child_node(child_uid)
        assert wrapper.pending_workspace_memory_payload is not None
        assert child_node.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("reload routing should not materialize hidden memory data")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
            source_spec=full_data(AverageOperation(dims=("alpha",))),
        )
        child.hide()

        fname = tmp_path / "pending-child-source-stale.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
            manager._workspace_controller.loading.pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("linking should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("unlinking should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("unlinking should not materialize hidden memory payloads")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "removing linked rows should not materialize hidden memory data"
            )

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        manifest = workspace_format._workspace_manifest_from_attrs(
            workspace_arrays._read_workspace_root_attrs_h5py(fname)
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "removing linked children should not materialize hidden memory data"
            )

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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

        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        manifest = workspace_format._workspace_manifest_from_attrs(
            workspace_arrays._read_workspace_root_attrs_h5py(fname)
        )
        assert all("link_group" not in entry for entry in manifest["nodes"])


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        loader = controller.loading
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
        assert not loader.pending._update_pending_workspace_manual_limits(
            node, {"x": [0.0, 1.0]}
        )
        assert not loader.pending._apply_pending_workspace_link_operation(
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
            assert not loader.pending._update_pending_workspace_manual_limits(
                node, {"x": [0.0, 1.0]}
            )
            assert not loader.pending._apply_pending_workspace_link_operation(
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
        assert not loader.pending._update_pending_workspace_manual_limits(
            node, {"x": [0.0, 1.0]}
        )
        assert not loader.pending._apply_pending_workspace_link_operation(
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
            controller.loading,
            "_read_workspace_imagetool_payload_dataset",
            _raise_payload_read,
        )
        assert not loader.pending._update_pending_workspace_manual_limits(
            node, {"x": [0.0, 1.0]}
        )
        assert not loader.pending._apply_pending_workspace_link_operation(
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
            return xr.Dataset({_ITOOL_DATA_NAME: data})

        monkeypatch.setattr(
            controller.loading,
            "_read_workspace_imagetool_payload_dataset",
            _metadata_dataset,
        )
        node.attrs = {"itool_state": json.dumps({})}
        assert loader.pending._update_pending_workspace_manual_limits(
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
            assert not loader.pending._apply_pending_workspace_link_operation(
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
            assert loader.pending._apply_pending_workspace_link_operation(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "r") as h5_file:
            original_partner_state = json.loads(
                h5_file["1/imagetool"].attrs["itool_state"]
            )
        new_cmap = (
            "viridis"
            if original_partner_state["color"]["cmap"] != "viridis"
            else "magma"
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(
            wrapper.pending_workspace_memory_payload is not None for wrapper in wrappers
        )
        materialized_calls = 0
        pending_payloads = manager._workspace_controller.loading.pending
        original_materialize = pending_payloads._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialized_calls
            materialized_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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


def test_manager_workspace_load_selection_skips_unchecked_children(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _SelectedChooseDialog(_ChooseFromDataTreeDialog):
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
        workspace_loading,
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

        tree = manager._workspace_controller.saving._to_datatree()
        try:
            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            manager._workspace_controller.loading._from_datatree(tree)
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
    class _SelectedChooseDialog(_ChooseFromDataTreeDialog):
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
        workspace_loading,
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

        tree = manager._workspace_controller.saving._to_datatree()
        try:
            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            manager._workspace_controller.loading._from_datatree(tree)
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
        load_func=(
            xr.load_dataarray,
            {"engine": "h5netcdf"},
            FileDataSelection(kind="dataarray"),
        ),
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
        manager._workspace_controller.loading._load_workspace_imagetool_dataset(
            ds,
            parent_target=None,
            node_path="0",
        )

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.name_of_imagetool(0) == "legacy manual name"
        assert manager.get_imagetool(0).slicer_area._data.name == "legacy manual name"


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
        manager._workspace_controller.loading._load_workspace_tool_dataset(
            ds, parent_target=None
        )


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

        tree = manager._workspace_controller.saving._to_datatree()
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

        assert manager._workspace_controller.loading._from_datatree(
            corrupted_tree,
            replace=True,
            mark_dirty=False,
            select=False,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == []

        assert manager._workspace_controller.loading._finish_workspace_file_load(True)

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
        loader = controller.loading
        controller._skipped_workspace_nodes = []
        with pytest.raises(ValueError, match="No workspace windows") as exc_info:
            loader._raise_no_workspace_windows_loaded()
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
            manager._workspace_controller.loading,
            "_load_workspace_node_or_warn",
            _fake_load_workspace_node_or_warn,
        )
        assert (
            manager._workspace_controller.loading._load_workspace_figures(
                tree, manifest=manifest
            )
            == 1
        )

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
            manager._workspace_controller.loading._from_h5py_workspace_file(
                fname, {}, replace=False, mark_dirty=False
            )
        with pytest.raises(ValueError, match="no loadable nodes"):
            manager._workspace_controller.loading._from_h5py_workspace_file(
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
            manager._workspace_controller.loading._from_h5py_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        def _raise_fast_read(*_args, **_kwargs):
            raise RuntimeError("fast path failed")

        monkeypatch.setattr(
            workspace_arrays,
            "_read_workspace_dataset_group_h5py",
            _raise_fast_read,
        )

        assert manager._workspace_controller.loading._from_h5py_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        def _raise_fast_load(*_args, **_kwargs) -> bool:
            raise RuntimeError("fast load failed")

        monkeypatch.setattr(
            manager._workspace_controller.loading,
            "_from_h5py_workspace_file",
            _raise_fast_load,
        )

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)

        def _raise_load(*_args, **_kwargs):
            raise RuntimeError("load failed")

        def _raise_restore(*_args, **_kwargs):
            raise RuntimeError("restore failed")

        loader = manager._workspace_controller.loading
        monkeypatch.setattr(loader, "_load_workspace_imagetool_dataset", _raise_load)
        monkeypatch.setattr(loader, "_restore_replaced_workspace", _raise_restore)

        with (
            caplog.at_level(logging.ERROR, logger=workspace_loading.logger.name),
            pytest.raises(ValueError, match="No workspace windows") as exc_info,
        ):
            manager._workspace_controller.loading._from_h5py_workspace_file(
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
            manager._workspace_controller.loading,
            "_workspace_rebind_data_for_uid",
            _fake_rebind_data,
        )

        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            tmp_path / "workspace.itws", backing_snapshot={}
        )
        assert calls == []

        manager._workspace_controller.loading._rebind_workspace_backed_imagetools(
            tmp_path / "workspace.itws"
        )

        assert uid in manager._tool_graph.nodes
        assert calls == [{}]


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
            with manager._workspace_controller._workspace_document_access_context(
                workspace
            ) as access:
                manager._workspace_controller._associate_loaded_workspace_file(
                    access.path,
                    workspace_format._current_workspace_schema_version(),
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
            manager._workspace_controller.loading,
            "_rebind_workspace_backed_imagetools",
            lambda path: rebind_paths.append(pathlib.Path(path)),
        )
        with manager._workspace_controller._workspace_document_access_context(
            workspace
        ) as access:
            manager._workspace_controller._associate_loaded_workspace_file(
                access.path,
                workspace_format._current_workspace_schema_version(),
                workspace_access=access,
                rebind_data=True,
            )

        assert rebind_paths == [workspace.resolve()]


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
            manager._workspace_controller._update_workspace_window_title()
            manager.closeEvent(event)

        assert file_path_calls == [str(workspace)]
        assert not event.isAccepted()
        assert not manager._workspace_state.closing_document


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
            manager._workspace_controller.loading,
            "_load_workspace_file",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        lock_errors: list[pathlib.Path] = []
        monkeypatch.setattr(
            workspace_storage,
            "_is_workspace_file_lock_error",
            lambda _exc: True,
        )
        monkeypatch.setattr(
            erlab.interactive.imagetool.manager._workspace._controller,
            "_show_workspace_file_lock_error",
            lambda _parent, path: lock_errors.append(pathlib.Path(path)),
        )
        assert not manager.open_recent_workspace(workspace)
        assert lock_errors == [workspace.resolve()]

        critical_messages: list[tuple[str, str]] = []
        monkeypatch.setattr(
            workspace_storage,
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
        assert manager._workspace_controller._active_managed_window() is None

        monkeypatch.setattr(
            QtWidgets.QApplication, "activeWindow", staticmethod(lambda: other)
        )
        monkeypatch.setattr(erlab.interactive.utils, "qt_is_valid", lambda *_objs: True)
        manager._workspace_controller._restore_focus_after_workspace_save(origin)


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        manager.open_multiple_files([fname], try_workspace=True)

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.workspace_path == str(fname.resolve())


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
        workspace_loading,
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
        manager._workspace_controller.saving._save_workspace_document(
            current_fname, force_full=True
        )
        adopt_workspace_path(manager, current_fname)
        manager._workspace_controller._mark_workspace_clean()

        import_tool = itool(data + 1, manager=False, execute=False)
        assert isinstance(import_tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(import_tool, show=False)
        import_fname = tmp_path / "import.itws"
        manager._workspace_controller.saving._save_workspace_document(
            import_fname, force_full=True
        )

        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        manager._workspace_controller._mark_workspace_clean()

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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "r") as h5_file:
            manifest = workspace_format._workspace_manifest_from_attrs(h5_file.attrs)
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
            workspace_loading,
            "_ChooseFromWorkspaceManifestDialog",
            _SelectRootOnlyDialog,
        )
        monkeypatch.setattr(
            workspace_arrays,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "selected current-schema loads should use manifest h5py path"
            ),
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            fname,
            replace=False,
            associate=False,
            mark_dirty=True,
            select=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager._tool_graph.root_wrappers[0]._childtool_indices == []
        assert figure_uid not in manager._tool_graph.nodes


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
            manager._figure_collection,
            "_set_available",
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
        update_figure_gallery_icon = manager._figure_collection.update_gallery_icon

        def _update_figure_gallery_icon(uid: str) -> None:
            if manager._workspace_ui_refresh_defer_depth > 0:
                update_figure_gallery_icon(uid)
                return
            calls.append(("gallery", uid))

        monkeypatch.setattr(
            manager._figure_collection,
            "update_gallery_icon",
            _update_figure_gallery_icon,
        )
        refresh_figure_source_controls = (
            manager._figure_workflows._refresh_figure_source_controls
        )

        def _refresh_figure_source_controls() -> None:
            if manager._workspace_ui_refresh_defer_depth > 0:
                refresh_figure_source_controls()
                return
            calls.append(("source_controls", None))

        monkeypatch.setattr(
            manager._figure_workflows,
            "_refresh_figure_source_controls",
            _refresh_figure_source_controls,
        )

        with manager._workspace_controller._workspace_load_context():
            manager._figure_collection.sync(select_uid="figure")
            manager._update_info(uid="figure")
            manager._update_info(uid="figure")
            manager._update_actions()
            manager._update_actions()
            manager._refresh_dependency_dependents("source")
            manager._refresh_dependency_dependents("source")
            manager._figure_collection.update_gallery_icon("figure")
            manager._figure_collection.update_gallery_icon("figure")
            manager._figure_collection.update_gallery_icon("figure")
            manager._figure_workflows._refresh_figure_source_controls()
            manager._figure_workflows._refresh_figure_source_controls()
            assert calls == []

        assert calls == [
            ("figures", False),
            ("dependency", "source"),
            ("source_controls", None),
            ("gallery", "figure"),
            ("actions", None),
            ("info", "figure"),
        ]


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
            manager._workspace_controller._drain_workspace_restore_events()

    assert calls == ["record"]
    assert not erlab.interactive.utils.qt_is_valid(deferred_widget)


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
            source_spec=full_data(),
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

        attrs = manager._workspace_controller.saving._pending_workspace_imagetool_attrs(
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
        attrs = manager._workspace_controller.saving._pending_workspace_imagetool_attrs(
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

        dataarray_selection = FileDataSelection(kind="dataarray")
        serialized_selection = dataarray_selection.model_dump(mode="json")
        assert wrapper._load_func_from_serialized_state("bad") is None
        assert wrapper._load_func_from_serialized_state(["bad", {}, None]) is None
        assert (
            wrapper._load_func_from_serialized_state(
                ["math:missing", {}, serialized_selection]
            )
            is None
        )
        assert (
            wrapper._load_func_from_serialized_state(
                ["math:pi", {}, serialized_selection]
            )
            is None
        )
        assert (
            wrapper._load_func_from_serialized_state(
                ["math:sqrt", {}, serialized_selection]
            )[0].__name__
            == "sqrt"
        )
        assert wrapper._load_func_from_serialized_state(
            ["da30", {}, serialized_selection]
        ) == ("da30", {}, dataarray_selection)
        assert wrapper._load_func_from_serialized_state(["da30", {}, 0]) == (
            "da30",
            {},
            FileDataSelection(kind="parsed_index", value=0),
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
        monkeypatch.setattr(
            manager_lineage, "can_reload_without_trust", lambda _spec: False
        )
        monkeypatch.setattr(
            manager_lineage, "has_file_load_source", lambda _spec: False
        )

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
            manager_lineage,
            "script_provenance_requires_trust",
            lambda _spec: True,
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
            manager_lineage,
            "script_provenance_requires_trust",
            lambda _spec: False,
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
        operation = typing.cast("ToolProvenanceOperation", types.SimpleNamespace())
        spec = full_data()
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "backing snapshot should not materialize hidden memory payloads"
            )

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        assert manager._workspace_controller.loading._workspace_data_backing_snapshot()[
            wrapper.uid
        ] == ("memory", ())
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
        {"0/imagetool": source.to_dataset(name=_ITOOL_DATA_NAME)}
    )
    try:
        workspace_storage._write_full_workspace_tree_file(
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

        assert workspace_arrays.dataarray_is_numpy_backed(slicer_area._data)
        assert not slicer_area.data_file_backed


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        def _fail_h5py_payload_read(*_args, **_kwargs):
            pytest.fail("hidden memory payload should not use fake h5py data")

        monkeypatch.setattr(
            workspace_arrays,
            "_read_workspace_dataset_group_h5py",
            _fail_h5py_payload_read,
        )

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail(
                "selecting pending sidebar metadata should not materialize data"
            )

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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

    pending_cls = workspace_pending._PendingWorkspacePayloads
    loaded = pending_cls._pending_workspace_data_with_loaded_coords(data)

    assert loaded.chunks == data.chunks
    assert loaded.coords["x"].chunks is None
    assert loaded.coords["y"].chunks is None
    assert loaded.coords["temperature"].chunks is None
    np.testing.assert_allclose(loaded.coords["x"].values, [0.0, 1.0])
    np.testing.assert_allclose(loaded.coords["y"].values, [10.0, 11.0, 12.0])
    assert float(loaded.coords["temperature"].values) == 12.0


def test_pending_workspace_data_roles_match_materialized_filtered_nonuniform_data(
    qtbot, tmp_path
) -> None:
    data = xr.DataArray(
        np.arange(4 * 5 * 3, dtype=np.float64).reshape((4, 5, 3)),
        dims=("alpha", "eV", "sample_temp"),
        coords={
            "alpha": np.linspace(-2.0, 2.0, 4),
            "eV": np.linspace(-0.5, 0.5, 5),
            "sample_temp": np.array([249.4, 251.2, 253.8]),
        },
        name="pending_nonuniform_order",
    )
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    operation = BoxcarFilterOperation(size={"sample_temp": 3})
    tool.slicer_area.apply_filter_operation(operation)
    saved = tool.to_dataset()
    state = json.loads(saved.attrs["itool_state"])
    assert tuple(state["slice"]["dims"]) == (
        "alpha",
        "eV",
        "sample_temp_idx",
    )

    stored = xr.Dataset(
        {
            _ITOOL_DATA_NAME: saved[_ITOOL_DATA_NAME].transpose(
                "sample_temp", "eV", "alpha"
            )
        },
        attrs=dict(saved.attrs),
    )
    fname = tmp_path / "pending-nonuniform-saved-dim-order.itws"
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", stored
    )
    node = types.SimpleNamespace(
        pending_workspace_memory_payload=(fname, "0/imagetool"),
        pending_workspace_payload_attrs=None,
        name=data.name,
        added_time_display="Today",
        _finalize_script_input_data=lambda data: data.copy(deep=False),
    )
    loader = workspace_loading._WorkspaceLoader(
        typing.cast("ImageToolManager", None),
        typing.cast("workspace_controller._WorkspaceController", None),
    )
    reference_datasets = {}
    try:
        pending_source = loader.pending._pending_workspace_lazy_source_data(
            node,
            data_role="source",
            reference_datasets=reference_datasets,
        )
        pending_displayed = loader.pending._pending_workspace_lazy_source_data(
            node,
            data_role="displayed",
            reference_datasets=reference_datasets,
        )
        assert pending_source.dims == data.dims
        assert pending_displayed.dims == pending_source.dims
        assert all(not str(dim).endswith("_idx") for dim in pending_source.dims)
        assert pending_source.chunks is not None
        pending_source = pending_source.compute()
        pending_displayed = pending_displayed.compute()
        info = loader.pending._pending_workspace_imagetool_info_text(node)
        assert info is not None
        assert "sample_temp_idx" not in info
    finally:
        loader._close_workspace_reference_datasets(reference_datasets)
    assert node.pending_workspace_memory_payload == (fname, "0/imagetool")

    loader_cls = workspace_loading._WorkspaceLoader
    loaded_ds = loader_cls._read_workspace_imagetool_payload_dataset(
        fname, "0/imagetool", load_data=True
    )
    restored = erlab.interactive.imagetool.ImageTool.from_dataset(loaded_ds)
    qtbot.addWidget(restored)
    try:
        materialized_source, _state = restored.slicer_area.persistence_data_and_state()
        xr.testing.assert_identical(pending_source, materialized_source)
        xr.testing.assert_identical(
            pending_displayed, restored.slicer_area.displayed_data
        )
        xr.testing.assert_identical(
            restored.slicer_area.displayed_data,
            operation.apply(materialized_source, parent_data=materialized_source),
        )
    finally:
        loaded_ds.close()


def test_pending_workspace_1d_roles_match_materialized_provenance_input(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(5, dtype=float),
        dims=("x",),
        coords={"x": np.arange(5, dtype=float)},
        name="watched_1d",
    )
    parent_spec = script(
        start_label="Start from watched variable 'watched_1d'",
        seed_code="derived = watched_1d",
    )
    source_spec = selection(SortCoordOrderOperation(), SqueezeOperation())

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        tool = itool(data, manager=False, execute=False)
        assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(
            tool,
            show=False,
            source_input_ndim=1,
            provenance_spec=parent_spec,
        )
        tool.hide()
        wrapper = manager._tool_graph.root_wrappers[0]
        materialized = {
            role: wrapper.data_for_role(role).copy(deep=True)
            for role in ("source", "displayed")
        }

        fname = tmp_path / "pending-1d-role-parity.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        loader = manager._workspace_controller.loading
        pending = {
            role: loader._workspace_tool_reference_source_data(
                0,
                data_role=role,
                owner_node=wrapper,
            ).compute()
            for role in ("source", "displayed")
        }
        for role in ("source", "displayed"):
            xr.testing.assert_identical(pending[role], materialized[role])
            composed = compose_display_provenance(
                parent_spec,
                source_spec,
                parent_data=pending[role],
            )
            assert composed is not None
            code = composed.display_code()
            assert code is not None
            assert ".squeeze()" not in code
            namespace = _exec_generated_code(code, {"watched_1d": data.copy(deep=True)})
            xr.testing.assert_identical(namespace["derived"], data)

        manager.get_imagetool(0)
        for role in ("source", "displayed"):
            xr.testing.assert_identical(wrapper.data_for_role(role), pending[role])


def test_pending_workspace_filter_validation(monkeypatch) -> None:
    pending_cls = workspace_pending._PendingWorkspacePayloads
    data = xr.DataArray(
        np.arange(6, dtype=np.float64).reshape((2, 3)),
        dims=("x", "y"),
    )

    assert pending_cls._apply_pending_workspace_filter(data, None) is data
    with pytest.raises(TypeError, match="Invalid pending filter operation"):
        pending_cls._apply_pending_workspace_filter(data, object())

    class _FakeOperation:
        def __init__(self, result: xr.DataArray) -> None:
            self._result = result

        def apply(
            self, _data: xr.DataArray, *, parent_data: xr.DataArray
        ) -> xr.DataArray:
            assert parent_data is data
            return self._result

    def _set_filter_result(result: xr.DataArray) -> None:
        monkeypatch.setattr(
            workspace_pending,
            "parse_tool_provenance_operation",
            lambda _payload: _FakeOperation(result),
        )

    _set_filter_result(data.mean("x"))
    with pytest.raises(ValueError, match="changed data dimensions"):
        pending_cls._apply_pending_workspace_filter(data, {})

    _set_filter_result(
        xr.DataArray(
            np.arange(8, dtype=np.float64).reshape((2, 4)),
            dims=("x", "y"),
        )
    )
    with pytest.raises(ValueError, match="changed data shape"):
        pending_cls._apply_pending_workspace_filter(data, {})

    _set_filter_result(data.transpose("y", "x"))
    filtered = pending_cls._apply_pending_workspace_filter(data, {})
    xr.testing.assert_identical(filtered, data)


def test_pending_workspace_metadata_coord_load_failure_falls_back(
    tmp_path,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    fname = tmp_path / "pending-metadata-fallback.itws"
    data = xr.DataArray(
        np.arange(6, dtype=np.float64).reshape((2, 3)),
        dims=["x", "y"],
        coords={"x": [0.0, 1.0], "y": [10.0, 11.0, 12.0], "temperature": 12.0},
        name=_ITOOL_DATA_NAME,
    )
    ds = xr.Dataset(
        {_ITOOL_DATA_NAME: data},
        attrs={"itool_name": "pending"},
    )
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", ds
    )

    def _fail_coord_load(_data: xr.DataArray) -> xr.DataArray:
        raise RuntimeError("coord read failed")

    monkeypatch.setattr(
        workspace_pending._PendingWorkspacePayloads,
        "_pending_workspace_data_with_loaded_coords",
        staticmethod(_fail_coord_load),
    )
    node = types.SimpleNamespace(
        pending_workspace_memory_payload=(fname, "0/imagetool"),
        pending_workspace_payload_attrs=None,
        name="pending",
        added_time_display="Today",
    )

    with manager_context() as manager:
        pending_payloads = manager._workspace_controller.loading.pending
        info = pending_payloads._pending_workspace_imagetool_info_text(node)
    assert info is not None
    assert "pending" in info
    assert "float64 [2]" in info
    assert "float64 [3]" in info
    assert "float64 scalar" in info
    assert node.pending_workspace_memory_payload == (fname, "0/imagetool")


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
            manager._workspace_controller.loading.pending,
            "_pending_workspace_imagetool_preview_image",
            _fail_pending_preview,
        )
        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
            "_pending_workspace_imagetool_preview_curve",
            _fail_pending_preview,
        )
        manager._update_info()
        selection_model.clearSelection()

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
            "_pending_workspace_imagetool_preview_image",
            lambda _node: None,
        )
        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
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
            workspace_arrays,
            "_read_workspace_dataset_group_h5py",
            _fail_full_payload_read,
        )
        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        monkeypatch.setattr(
            workspace_pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        monkeypatch.setattr(
            workspace_pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
    monkeypatch,
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

    sparse_y = np.full_like(long_x, np.nan)
    sparse_y[::1000] = 1.0
    sparse_decimated = manager_widgets._curve_preview_data(long_x, sparse_y)
    assert sparse_decimated is not None
    assert sparse_decimated[0].size <= manager_widgets._CURVE_PREVIEW_MAX_POINTS

    assert manager_widgets._curve_preview_data(np.arange(2.0), np.arange(3.0)) is None
    assert (
        manager_widgets._curve_preview_data(
            np.array([np.nan, np.inf]), np.array([1.0, np.nan])
        )
        is None
    )
    preview.setCurve(np.arange(2.0), np.arange(3.0))
    assert preview._curve_data is None
    assert not preview.isVisible()

    preview.setCurve(np.ones(4, dtype=np.float64), np.arange(4, dtype=np.float64))
    assert preview._curve_data is not None
    assert not preview._curve_item.path().isEmpty()

    preview._curve_data = None
    preview._rebuild_curve_paths()
    assert preview._curve_data is None

    preview._curve_data = (np.array([np.nan]), np.array([1.0]))
    preview._rebuild_curve_paths()
    assert preview._curve_data is None

    pixmap = QtGui.QPixmap(32, 16)
    pixmap.fill(QtCore.Qt.GlobalColor.white)
    preview.setPixmap(pixmap)
    assert preview._curve_data is None
    assert preview._curve_item.isVisible() is False
    assert preview.scene() is not None
    assert preview.scene().sceneRect() == QtCore.QRectF(0.0, 0.0, 32.0, 16.0)

    preview.setPixmap(QtGui.QPixmap())
    assert not preview.isVisible()
    assert preview.scene() is not None
    assert preview.scene().sceneRect() == QtCore.QRectF()

    with monkeypatch.context() as patch:
        patch.setattr(erlab.interactive.utils, "_apply_qt_accent_color", lambda _c: "")
        assert manager_widgets._summary_accent_color().isValid()


def test_manager_wrapper_preview_curve_handles_unavailable_live_items(
    monkeypatch,
) -> None:
    valid_objects: set[int] = set()
    monkeypatch.setattr(
        erlab.interactive.utils,
        "qt_is_valid",
        lambda obj: id(obj) in valid_objects,
    )

    class _FakeArea:
        def __init__(
            self,
            main_image: object | None,
            coord_values: np.ndarray | None = None,
        ) -> None:
            self._main_image = main_image
            if coord_values is None:
                coord_values = np.arange(3.0)
            self.array_slicer = types.SimpleNamespace(
                values_of_dim=lambda _dim: coord_values
            )

        def _update_if_delayed(self) -> None:
            return

        @property
        def main_image(self) -> object:
            if self._main_image is None:
                raise RuntimeError
            return self._main_image

    def _tool(main_image: object | None) -> types.SimpleNamespace:
        return types.SimpleNamespace(slicer_area=_FakeArea(main_image))

    def _tool_with_coords(
        main_image: object | None, coord_values: np.ndarray
    ) -> types.SimpleNamespace:
        return types.SimpleNamespace(slicer_area=_FakeArea(main_image, coord_values))

    assert manager_wrapper._preview_curve_from_imagetool(None) is None
    assert manager_wrapper._preview_curve_from_imagetool(_tool(None)) is None

    invalid_image = types.SimpleNamespace(slicer_data_items=[])
    assert manager_wrapper._preview_curve_from_imagetool(_tool(invalid_image)) is None

    empty_image = types.SimpleNamespace(slicer_data_items=[])
    valid_objects.add(id(empty_image))
    assert manager_wrapper._preview_curve_from_imagetool(_tool(empty_image)) is None

    invalid_item = types.SimpleNamespace()
    item_image = types.SimpleNamespace(slicer_data_items=[invalid_item])
    valid_objects.add(id(item_image))
    assert manager_wrapper._preview_curve_from_imagetool(_tool(item_image)) is None

    missing_values_item = types.SimpleNamespace(image=None)
    image_without_values = types.SimpleNamespace(
        is_image=True,
        slicer_data_items=[missing_values_item],
    )
    valid_objects.update({id(image_without_values), id(missing_values_item)})
    assert (
        manager_wrapper._preview_curve_from_imagetool(_tool(image_without_values))
        is None
    )

    bad_shape_item = types.SimpleNamespace(image=np.array(1.0))
    bad_shape_image = types.SimpleNamespace(
        is_image=True,
        slicer_data_items=[bad_shape_item],
    )
    valid_objects.update({id(bad_shape_image), id(bad_shape_item)})
    assert manager_wrapper._preview_curve_from_imagetool(_tool(bad_shape_image)) is None

    one_dimensional_image_item = types.SimpleNamespace(image=np.arange(3.0)[:, None])
    one_dimensional_image = types.SimpleNamespace(
        is_image=True,
        slicer_data_items=[one_dimensional_image_item],
        axis_dims=(),
        display_axis=(),
    )
    valid_objects.update({id(one_dimensional_image), id(one_dimensional_image_item)})
    curve = manager_wrapper._preview_curve_from_imagetool(_tool(one_dimensional_image))
    assert curve is not None
    np.testing.assert_allclose(curve[0], np.arange(3.0))
    np.testing.assert_allclose(curve[1], np.arange(3.0))

    image_with_mismatched_coords = types.SimpleNamespace(
        is_image=True,
        slicer_data_items=[one_dimensional_image_item],
        axis_dims=("energy",),
        display_axis=(),
    )
    valid_objects.add(id(image_with_mismatched_coords))
    curve = manager_wrapper._preview_curve_from_imagetool(
        _tool_with_coords(image_with_mismatched_coords, np.arange(2.0))
    )
    assert curve is not None
    np.testing.assert_allclose(curve[0], np.arange(3.0))
    np.testing.assert_allclose(curve[1], np.arange(3.0))

    class _RaisingCurveItem:
        def getData(self) -> tuple[np.ndarray, np.ndarray]:
            raise RuntimeError

    raising_item = _RaisingCurveItem()
    raising_curve_image = types.SimpleNamespace(
        is_image=False,
        slicer_data_items=[raising_item],
    )
    valid_objects.update({id(raising_curve_image), id(raising_item)})
    assert (
        manager_wrapper._preview_curve_from_imagetool(_tool(raising_curve_image))
        is None
    )

    none_data_item = types.SimpleNamespace(getData=lambda: (None, np.arange(3.0)))
    none_data_image = types.SimpleNamespace(
        is_image=False,
        slicer_data_items=[none_data_item],
    )
    valid_objects.update({id(none_data_image), id(none_data_item)})
    assert manager_wrapper._preview_curve_from_imagetool(_tool(none_data_image)) is None

    valid_curve_item = types.SimpleNamespace(
        getData=lambda: (np.arange(3.0), np.arange(3.0))
    )
    valid_curve_image = types.SimpleNamespace(
        is_image=False,
        slicer_data_items=[valid_curve_item],
    )
    valid_objects.update({id(valid_curve_image), id(valid_curve_item)})
    curve = manager_wrapper._preview_curve_from_imagetool(_tool(valid_curve_image))
    assert curve is not None
    np.testing.assert_allclose(curve[0], np.arange(3.0))
    np.testing.assert_allclose(curve[1], np.arange(3.0))

    class _RaisingPendingCurve:
        pending_workspace_memory_payload = object()

        def pending_workspace_preview_curve(self) -> None:
            raise RuntimeError

    assert manager_wrapper._preview_curve_for_node(_RaisingPendingCurve()) is None

    class _RaisingImageToolNode:
        @property
        def imagetool(self) -> None:
            raise RuntimeError

    assert manager_wrapper._preview_curve_for_node(_RaisingImageToolNode()) is None


def test_pending_toolwindow_metadata_and_preview_helpers(
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
        loader = controller.loading
        workspace = tmp_path / "pending-tool-metadata.itws"
        with h5py.File(workspace, "w") as h5_file:
            payload = h5_file.create_group("payload/tool")
            payload.attrs["tool_visible"] = False
            payload.attrs["tool_display_name"] = "Saved Tool"

        assert (
            workspace_loading._workspace_payload_window_visible_h5py(
                workspace, "payload/tool", "tool"
            )
            is False
        )
        assert (
            workspace_loading._workspace_payload_window_visible_h5py(
                workspace, "missing", "tool"
            )
            is None
        )
        payload_attrs = workspace_loading._workspace_payload_attrs_h5py(
            workspace, "payload/tool"
        )
        assert payload_attrs is not None
        assert payload_attrs["tool_display_name"] == "Saved Tool"
        assert (
            workspace_loading._workspace_payload_attrs_h5py(workspace, "missing")
            is None
        )

        pixmap = QtGui.QPixmap(8, 4)
        pixmap.fill(QtCore.Qt.GlobalColor.white)
        png_bytes = QtCore.QByteArray()
        buffer = QtCore.QBuffer(png_bytes)
        buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        assert pixmap.save(buffer, "PNG")
        encoded_png = base64.b64encode(bytes(png_bytes)).decode("ascii")

        node = manager_wrapper._ManagedWindowNode(
            manager,
            "pending-tool",
            None,
            None,
            window_kind="tool",
            name="pending-tool",
        )
        node.set_pending_workspace_payload(
            "tool",
            workspace,
            "payload/tool",
            payload_attrs={
                "tool_cls_qualname": b"example.module:Outer.NestedTool",
                "tool_data_name": b"source_data",
                "tool_source_state": b"stale",
                "figure_composer_preview_cache_png": encoded_png,
            },
        )

        assert node.type_badge_text == "NestedTool"
        fields = {field.label: field.value for field in node.metadata_fields}
        assert fields["Kind"] == "NestedTool"
        assert (
            loader._workspace_tool_display_name_from_attrs(
                {"tool_cls_qualname": b"example.module:Outer.NestedTool"}
            )
            == "NestedTool"
        )
        assert (
            loader._workspace_tool_display_name_from_attrs(
                {"tool_display_name": b"Display Tool"}
            )
            == "Display Tool"
        )
        assert (
            loader._workspace_tool_display_name_from_attrs({"tool_title": "Window[*]"})
            == "Window"
        )
        assert loader._workspace_tool_display_name_from_attrs({}) == "ToolWindow"
        assert (
            loader._workspace_tool_source_state_from_attrs(
                {"tool_source_state": b"stale"}
            )
            == "stale"
        )
        assert (
            loader._workspace_tool_source_state_from_attrs(
                {"tool_source_state": b"unknown"}
            )
            == "fresh"
        )

        text = loader.pending._pending_workspace_tool_info_text(node)
        assert text is not None
        assert "NestedTool" in text
        assert "source_data" in text
        assert "stale" in text
        assert loader.pending._pending_workspace_info_text(node) == text

        preview = node.pending_workspace_tool_preview_image()
        assert preview is not None
        assert preview[0] == 0.5
        assert not preview[1].isNull()
        assert node.cached_pending_workspace_tool_preview_image() == preview

        with monkeypatch.context() as patch:
            patch.setattr(manager, "_selected_imagetool_targets", lambda: ())
            patch.setattr(manager, "_selected_tool_uids", lambda: (node.uid,))
            patch.setattr(manager, "_node_for_target", lambda _target: node)
            manager._details_panel._update_info()
        assert manager.preview_widget.isVisible()
        assert not manager.preview_widget._pixmapitem.pixmap().isNull()

        node.update_pending_workspace_payload_attrs(
            {
                "tool_display_name": b"Bytes Preview Tool",
                "figure_composer_preview_cache_png": encoded_png.encode(),
            }
        )
        bytes_preview = loader.pending._pending_workspace_tool_preview_image(node)
        assert bytes_preview is not None
        assert not bytes_preview[1].isNull()

        node.update_pending_workspace_payload_attrs(
            {
                "tool_display_name": b"Display Tool",
                "tool_data_name": "<none-value>",
                "figure_composer_preview_cache_png": "not valid base64",
            }
        )
        assert node.type_badge_text == "Display Tool"
        text_without_data = loader.pending._pending_workspace_tool_info_text(node)
        assert text_without_data is not None
        assert "Data:" not in text_without_data
        assert loader.pending._pending_workspace_tool_preview_image(node) is None

        updates: list[str | None] = []
        with monkeypatch.context() as patch:
            patch.setattr(manager, "_selected_imagetool_targets", lambda: ())
            patch.setattr(manager, "_selected_tool_uids", lambda: (node.uid,))
            patch.setattr(manager, "_node_for_target", lambda _target: node)
            patch.setattr(manager, "_child_node", lambda _uid: node)
            patch.setattr(
                manager,
                "_update_info",
                lambda *, uid=None: updates.append(uid),
            )
            patch.setattr(node, "materialize_pending_workspace_payload", lambda: True)
            manager._details_panel._update_info()
            manager._details_panel._load_selected_preview_data()
        load_button = manager.preview_widget.findChild(
            QtWidgets.QPushButton, "manager_pending_preview_load_button"
        )
        assert load_button is not None
        assert load_button.isVisible()
        assert updates == [node.uid]

        node_without_pending = manager_wrapper._ManagedWindowNode(
            manager,
            "not-pending-tool",
            None,
            None,
            window_kind="tool",
            name="not-pending-tool",
        )
        node_without_pending.update_pending_workspace_payload_attrs(
            {"tool_display_name": "ignored"}
        )
        assert node_without_pending.type_badge_text is None
        assert node_without_pending.pending_workspace_preview_curve() is None

        no_pending = types.SimpleNamespace(
            pending_workspace_tool_payload=None,
            pending_workspace_payload_kind=None,
        )
        assert loader.pending._pending_workspace_tool_info_text(no_pending) is None
        assert loader.pending._pending_workspace_tool_preview_image(no_pending) is None
        assert loader.pending._pending_workspace_info_text(no_pending) is None


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
        loader = controller.loading

        assert loader.pending._pending_preview_axis_state((4, 5), {}) is None
        assert (
            loader.pending._pending_preview_axis_state(
                (4, 5), {"slice": {"bins": [], "indices": []}}
            )
            is None
        )
        assert loader.pending._pending_preview_axis_state(
            (4, 5),
            {"slice": {"bins": [["bad", 3]], "indices": [["bad", 99]]}},
        ) == ((1, 2), (1, 3), (False, True))
        assert loader.pending._pending_preview_axis_state(
            (4, 5),
            {"slice": {"bins": [[0, -2]], "indices": [[2, 3]]}},
        ) == ((2, 3), (1, 1), (False, False))
        assert loader.pending._pending_preview_axis_state(
            (4, 5),
            {
                "current_cursor": 1,
                "slice": {"bins": [[1, 1], [1, 1]], "indices": [[0, 0]]},
            },
        ) == ((1, 2), (1, 1), (False, False))
        assert (
            loader.pending._pending_preview_axis_state(
                (4, 5), {"slice": {"bins": ["bad"], "indices": ["bad"]}}
            )
            is None
        )

        fname = tmp_path / "pending-preview-reader-fallbacks.itws"
        data = xr.DataArray(
            np.arange(4 * 5 * 6, dtype=np.float64).reshape((4, 5, 6)),
            dims=["x", "y", "z"],
            coords={"x": np.arange(4), "y": np.arange(5), "z": np.arange(6)},
            name=_ITOOL_DATA_NAME,
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
            {_ITOOL_DATA_NAME: data},
            attrs={"itool_state": json.dumps(state), "itool_name": "pending"},
        )
        assert workspace_arrays._write_workspace_dataset_group_h5py(
            fname, "0/imagetool", ds
        )

        node = types.SimpleNamespace(
            pending_workspace_memory_payload=(fname, "0/imagetool"),
            pending_workspace_payload_attrs=None,
            name="pending",
            added_time_display="Today",
        )
        info = loader.pending._pending_workspace_imagetool_info_text(node)
        assert info is not None
        assert "pending" in info
        preview = loader.pending._pending_workspace_imagetool_preview_image(node)
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
            name=_ITOOL_DATA_NAME,
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
            {_ITOOL_DATA_NAME: nonuniform_data},
            attrs={
                "itool_state": json.dumps(nonuniform_state),
                "itool_name": "nonuniform",
            },
        )
        assert workspace_arrays._write_workspace_dataset_group_h5py(
            nonuniform_fname, "0/imagetool", nonuniform_ds
        )
        with h5py.File(nonuniform_fname, "r+") as h5_file:
            group = h5_file["0/imagetool"]
            physical_coord = group.create_dataset(
                "alpha_physical_scale", data=np.array([-2.0, -0.4, 0.2, 2.1])
            )
            physical_coord.make_scale("alpha")
            data_dataset = group[_ITOOL_DATA_NAME]
            data_dataset.dims[0].attach_scale(physical_coord)
            assert len(list(data_dataset.dims[0].keys())) > 1
        node.pending_workspace_memory_payload = (nonuniform_fname, "0/imagetool")
        node.pending_workspace_payload_attrs = None
        nonuniform_preview = loader.pending._pending_workspace_imagetool_preview_image(
            node
        )
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
            name=_ITOOL_DATA_NAME,
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
            {_ITOOL_DATA_NAME: permuted_data},
            attrs={
                "itool_state": json.dumps(permuted_state),
                "itool_name": "permuted",
            },
        )
        assert workspace_arrays._write_workspace_dataset_group_h5py(
            permuted_fname, "0/imagetool", permuted_ds
        )
        with h5py.File(permuted_fname, "r") as h5_file:
            data_dataset = h5_file[f"0/imagetool/{_ITOOL_DATA_NAME}"]
            assert loader.pending._pending_preview_dataset_dims(
                data_dataset, ("alpha", "eV", "sample_temp_idx")
            ) == (("sample_temp_idx", "eV", "alpha"), None)
        node.pending_workspace_memory_payload = (permuted_fname, "0/imagetool")
        node.pending_workspace_payload_attrs = None
        permuted_preview = loader.pending._pending_workspace_imagetool_preview_image(
            node
        )
        assert permuted_preview is not None
        assert permuted_preview[1].isNull() is False

        missing_node = types.SimpleNamespace(
            pending_workspace_memory_payload=(tmp_path / "missing.itws", "0/imagetool"),
            pending_workspace_payload_attrs=None,
            name="missing",
            added_time_display="Today",
        )
        assert (
            loader.pending._pending_workspace_imagetool_info_text(missing_node) is None
        )
        assert (
            loader.pending._pending_workspace_imagetool_preview_image(missing_node)
            is None
        )

        no_pending = types.SimpleNamespace(
            pending_workspace_memory_payload=None,
            pending_workspace_payload_attrs=None,
            name="missing",
            added_time_display="Today",
        )
        assert loader.pending._pending_workspace_imagetool_info_text(no_pending) is None
        assert (
            loader.pending._pending_workspace_imagetool_preview_image(no_pending)
            is None
        )

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
            assert (
                loader.pending._pending_workspace_imagetool_preview_image(node) is None
            )

        empty_fname = tmp_path / "pending-preview-empty.itws"
        with h5py.File(empty_fname, "w") as h5_file:
            h5_file.create_group("0/imagetool")
        node.pending_workspace_memory_payload = (empty_fname, "0/imagetool")
        node.pending_workspace_payload_attrs = None
        assert loader.pending._pending_workspace_imagetool_preview_image(node) is None


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        def _fail_materialize_pending_payload(_node) -> bool:
            pytest.fail("hover preview should not materialize pending data")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        messages: list[str] = []

        @contextlib.contextmanager
        def _record_wait_dialog(_parent, message):
            messages.append(message)
            yield types.SimpleNamespace(
                set_message=lambda updated: messages.append(updated)
            )

        monkeypatch.setattr(erlab.interactive.utils, "wait_dialog", _record_wait_dialog)
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        wrapper = manager._tool_graph.root_wrappers[0]

        assert wrapper.pending_workspace_memory_payload is not None
        assert messages.count("Loading workspace...") == 1
        assert "Loading ImageTool data..." not in messages

        messages.clear()
        loader = manager._workspace_controller.loading
        assert loader.pending._materialize_pending_workspace_payload(wrapper)

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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

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

        assert manager._workspace_controller.loading._load_workspace_file(
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
        saver = manager._workspace_controller.saving
        assert saver._pending_workspace_tool_references_available(child_node)

        child_node.update_pending_workspace_payload_attrs(
            {
                erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR: json.dumps(
                    {"data": {"kind": "future_reference", "node_uid": root_uid}}
                )
            }
        )
        assert not (
            manager._workspace_controller.saving._pending_workspace_tool_references_available(
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
            manager._workspace_controller.saving._pending_workspace_tool_references_available(
                child_node
            )
        )


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        loaded = manager.get_imagetool(0).slicer_area

        assert wrapper.pending_workspace_memory_payload is None
        assert workspace_arrays.dataarray_is_numpy_backed(loaded._data)
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        manager._figure_workflows._figure_operations_from_image_targets(
            (0,), ("saved",)
        )

        assert wrapper.pending_workspace_memory_payload is None
        np.testing.assert_array_equal(
            manager.get_imagetool(0).slicer_area._data.values,
            data.values,
        )


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
    restored: list[ToolProvenanceOperation] = []

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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        operation = GaussianFilterOperation(sigma={"x": 1.0})

        manager._provenance_edit_controller._edit_active_filter(
            wrapper,
            operation,
            _FilterDialog,
        )

        assert wrapper.pending_workspace_memory_payload is None
        assert restored == [operation]
        assert workspace_arrays.dataarray_is_numpy_backed(wrapper.slicer_area._data)
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None
        manager.show_imagetool(0)
        qtbot.wait_until(lambda: manager.get_imagetool(0).isVisible())

        loaded = manager.get_imagetool(0).slicer_area
        assert wrapper.pending_workspace_memory_payload is None
        assert loaded.data_loadable is False
        assert workspace_arrays.dataarray_is_numpy_backed(loaded._data)
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
        child.set_source_binding(full_data())
        child_uid = manager.add_childtool(child, 0, show=False)

        fname = tmp_path / "pending-parent-child-reference.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "r") as h5_file:
            references = json.loads(
                h5_file[f"0/childtools/{child_uid}/tool"].attrs["tool_data_references"]
            )
        assert references[imagetool_serialization.SAVED_TOOL_DATA_NAME] == {
            "kind": "parent_source"
        }

        pending_payloads = manager._workspace_controller.loading.pending
        original_materialize = pending_payloads._materialize_pending_workspace_payload
        loader_cls = workspace_loading._WorkspaceLoader
        original_read = loader_cls._read_workspace_imagetool_payload_dataset

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
            manager._workspace_controller.loading.pending,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )
        monkeypatch.setattr(
            loader_cls,
            "_read_workspace_imagetool_payload_dataset",
            classmethod(_fail_eager_pending_read),
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
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

        pending_payloads = manager._workspace_controller.loading.pending
        original_materialize = pending_payloads._materialize_pending_workspace_payload
        loader_cls = workspace_loading._WorkspaceLoader
        original_read = loader_cls._read_workspace_imagetool_payload_dataset

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
            manager._workspace_controller.loading.pending,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )
        monkeypatch.setattr(
            loader_cls,
            "_read_workspace_imagetool_payload_dataset",
            classmethod(_fail_eager_pending_read),
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "r") as h5_file:
            assert (
                "tool_data_references"
                not in h5_file[f"0/childtools/{child_uid}/tool"].attrs
            )

        materialize_calls = 0
        pending_payloads = manager._workspace_controller.loading.pending
        original_materialize = pending_payloads._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialize_calls
            materialize_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
            "_materialize_pending_workspace_payload",
            _count_materialize,
        )

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
                group[_ITOOL_DATA_NAME][...],
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        materialize_calls = 0
        pending_payloads = manager._workspace_controller.loading.pending
        original_materialize = pending_payloads._materialize_pending_workspace_payload

        def _count_materialize(node):
            nonlocal materialize_calls
            materialize_calls += 1
            return original_materialize(node)

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
            "_materialize_pending_workspace_payload",
            _count_materialize,
        )

        wrapper.name = "renamed"
        update = manager._workspace_controller.saving._workspace_attr_update_snapshot(
            wrapper.uid
        )

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
                group[_ITOOL_DATA_NAME][...],
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        monkeypatch.setattr(
            workspace_arrays,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "simple v4 load should not open the workspace DataTree"
            ),
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        node = manager._node_for_target(0)
        if node.pending_workspace_memory_payload is not None:
            loader = manager._workspace_controller.loading
            assert loader.pending._materialize_pending_workspace_payload(node)
        loaded = manager.get_imagetool(0).slicer_area
        assert workspace_arrays.dataarray_is_numpy_backed(loaded._data)
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(fname)
        _, _, manifest = workspace_format._workspace_file_metadata_from_attrs(
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

        profiler = workspace_loading._WorkspaceLoadProfiler(fname)
        assert manager._workspace_controller.loading._from_h5py_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
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

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
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

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        monkeypatch.setattr(
            workspace_arrays,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "simple v4 load should not open the workspace DataTree"
            ),
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        node = manager._node_for_target(0)
        if node.pending_workspace_memory_payload is not None:
            loader = manager._workspace_controller.loading
            assert loader.pending._materialize_pending_workspace_payload(node)
        loaded = manager.get_imagetool(0).slicer_area
        assert loaded.data.dims == ("y", "x", "z")
        assert loaded.manual_limits == expected_limits
        _assert_manual_limits_view_ranges(loaded, expected_limits)
        assert loaded.axis_inversions == {"x": True}
        assert workspace_arrays.dataarray_is_numpy_backed(loaded._data)
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        monkeypatch.setattr(
            workspace_arrays,
            "open_workspace_datatree",
            lambda *args, **kwargs: pytest.fail(
                "unsupported payload should fall back by group, not by whole tree"
            ),
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded = manager.get_imagetool(0).slicer_area
        assert workspace_arrays.dataarray_is_numpy_backed(loaded._data)
        assert not loaded.data_chunked
        assert not loaded.data_file_backed
        assert loaded._data.coords["label"].item() == "sample"


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "a") as h5_file:
            h5_file.create_group(
                f"{workspace_format._WORKSPACE_PENDING_GROUP_PREFIX}stale"
            )
            h5_file.create_group(
                f"{workspace_format._WORKSPACE_BACKUP_GROUP_PREFIX}stale"
            )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        tree = workspace_arrays.open_workspace_datatree(fname, chunks="auto")
        accept_dialog(
            lambda: manager._workspace_controller.loading._from_datatree(
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
        manager._workspace_controller.saving._save_workspace_document(
            current_fname, force_full=True
        )
        adopt_workspace_path(manager, current_fname)
        manager._workspace_controller._mark_workspace_clean()

        broken_fname = tmp_path / "broken.itws"
        with h5py.File(broken_fname, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
            h5_file.create_group("0")

        with pytest.raises(ValueError, match="No workspace windows") as exc_info:
            manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            current_fname, force_full=True
        )
        adopt_workspace_path(manager, current_fname)
        manager._workspace_controller._mark_workspace_clean()

        def _fail_to_datatree(*_args, **_kwargs):
            raise AssertionError("clean associated replace load should use file backup")

        monkeypatch.setattr(
            manager._workspace_controller.saving, "_to_datatree", _fail_to_datatree
        )

        broken_fname = tmp_path / "broken-clean.itws"
        with h5py.File(broken_fname, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
            h5_file.create_group("0")

        with pytest.raises(ValueError, match="No workspace windows"):
            manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            current_fname, force_full=True
        )
        adopt_workspace_path(manager, current_fname)
        manager._workspace_controller._mark_workspace_clean()
        manager._workspace_controller._mark_node_state_dirty(uid)
        assert manager.is_workspace_modified

        to_datatree_calls = 0
        original_to_datatree = manager._workspace_controller.saving._to_datatree

        def _record_to_datatree(*args, **kwargs):
            nonlocal to_datatree_calls
            to_datatree_calls += 1
            return original_to_datatree(*args, **kwargs)

        monkeypatch.setattr(
            manager._workspace_controller.saving, "_to_datatree", _record_to_datatree
        )

        broken_fname = tmp_path / "broken-dirty.itws"
        with h5py.File(broken_fname, "w") as h5_file:
            h5_file.attrs["imagetool_workspace_schema_version"] = 4
            h5_file.create_group("0")

        with pytest.raises(ValueError, match="No workspace windows"):
            manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        for _ in range(3):
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents()

        assert not manager.is_workspace_modified
