import datetime
import json
import logging
import pathlib
import typing
import weakref
from collections.abc import Callable, Mapping

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._qt_state as qt_state
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager as manager_module
import erlab.interactive.imagetool.manager._desktop as manager_desktop
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._loading as workspace_loading
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.plot_items as imagetool_plot_items
import erlab.interactive.imagetool.viewer as imagetool_viewer
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._load_source import _serialize_loader_kwargs
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ToolProvenanceSpec,
    full_data,
    script,
)
from erlab.interactive.imagetool._provenance._operations import (
    GaussianFilterOperation,
    ImageToolSelectionSourceBinding,
    RenameOperation,
)
from erlab.interactive.imagetool.manager import ImageToolManager
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ChooseFromWorkspaceManifestDialog,
)
from tests.interactive.imagetool.manager.helpers import adopt_workspace_path
from tests.interactive.imagetool.manager.workspace._support import (
    _AddedTimeChildTool,
    _request_workspace_save_and_wait,
    _workspace_test_file_spec,
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
        metadata_source = erlab.io.metadata.ExcelMetadataSource(
            tmp_path / "metadata.xlsx",
            sheet_name="Measurements",
            file_name_column="File",
            coordinate_mapping={"Temperature": "sample_temp"},
        )
        shared_loader_kwargs = {
            "single": True,
            "metadata": metadata_source,
        }
        shared_loader_extensions = {"coordinate_attrs": ["shared"]}
        manager._set_shared_loader_options(
            loader_name,
            shared_loader_kwargs,
            shared_loader_extensions,
        )

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
        assert explorer.loader_kwargs_by_name()[loader_name] == shared_loader_kwargs
        assert (
            explorer.loader_extensions_by_name()[loader_name]
            == shared_loader_extensions
        )
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
        saved_manager_kwargs = manifest["loader_state"][
            "manager_loader_kwargs_by_filter"
        ][name_filter]
        assert saved_manager_kwargs["single"] is True
        assert (
            saved_manager_kwargs["metadata"]["__erlab_spreadsheet_metadata_source__"][
                "sheet_name"
            ]
            == "Measurements"
        )
        app_state = manifest["standalone_apps"]["apps"]
        assert not app_state["explorer"]["window_state"]["visible"]
        assert app_state["explorer"]["active_tab"] == 1
        saved_explorer_kwargs = app_state["explorer"]["loader_kwargs_by_name"][
            loader_name
        ]
        assert saved_explorer_kwargs["single"] is True
        assert (
            saved_explorer_kwargs["metadata"]["__erlab_spreadsheet_metadata_source__"][
                "sheet_name"
            ]
            == "Measurements"
        )
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
        restored_manager_kwargs = manager._recent_loader_kwargs_by_filter[name_filter]
        assert restored_manager_kwargs["single"] is True
        restored_metadata = restored_manager_kwargs["metadata"]
        assert isinstance(restored_metadata, erlab.io.metadata.ExcelMetadataSource)
        assert restored_metadata.sheet_name == "Measurements"
        assert restored_metadata.coordinate_mapping == {"Temperature": "sample_temp"}
        assert (
            manager._recent_loader_extensions_by_filter[name_filter]
            == shared_loader_extensions
        )

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
        restored_explorer_kwargs = restored_explorer.loader_kwargs_by_name()[
            loader_name
        ]
        assert restored_explorer_kwargs["single"] is True
        assert isinstance(
            restored_explorer_kwargs["metadata"],
            erlab.io.metadata.ExcelMetadataSource,
        )
        assert (
            restored_explorer.loader_extensions_by_name()[loader_name]
            == shared_loader_extensions
        )


def test_manager_workspace_roundtrip_restores_full_serializable_state(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:

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

    loader_name = next(
        name for name in erlab.io.loaders if erlab.io.loaders[name].file_dialog_methods
    )
    name_filter = next(iter(erlab.io.loaders[loader_name].file_dialog_methods))
    explorer_kwargs = {loader_name: {"single": True}}
    explorer_extensions = {loader_name: {"coordinate_attrs": ["shared"]}}
    root = erlab.interactive.imagetool.ImageTool(test_data, _in_manager=True)
    qtbot.addWidget(root)

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        manager.add_imagetool(root, show=False)
        manager._recent_directory = str(example_data_dir)
        manager._recent_name_filter = name_filter
        manager._set_shared_loader_options(
            loader_name,
            explorer_kwargs[loader_name],
            explorer_extensions[loader_name],
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
        manager._recent_loader_extensions_by_filter.clear()

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


def test_workspace_loader_snapshot_keeps_runtime_metadata_source(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        source = erlab.io.metadata.ExcelMetadataSource(
            tmp_path / "metadata.xlsx",
            sheet_name="Measurements",
            file_name_column="File",
            coordinate_mapping={"Temperature": "sample_temp"},
        )
        manager._set_shared_loader_options(
            "example",
            {"metadata": source},
            {},
        )

        payload = (
            manager._workspace_controller.saving._workspace_loader_state_snapshot()
        )

        serialized_metadata = payload["explorer_loader_kwargs_by_name"]["example"][
            "metadata"
        ]
        assert "__erlab_spreadsheet_metadata_source__" in serialized_metadata
        runtime_metadata = (
            manager._workspace_controller._loader_state.explorer_loader_kwargs_by_name[
                "example"
            ]["metadata"]
        )
        assert runtime_metadata is source
        shared_kwargs, _shared_extensions = manager._shared_loader_state()
        assert shared_kwargs["example"]["metadata"] is source


def test_workspace_loader_restore_skips_invalid_metadata_entries(
    qtbot,
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    valid_source = erlab.io.metadata.ExcelMetadataSource(
        tmp_path / "metadata.xlsx",
        file_name_column="File",
        coordinate_mapping={"Temperature": "sample_temp"},
    )
    valid_kwargs = _serialize_loader_kwargs({"single": True, "metadata": valid_source})
    invalid_kwargs = {
        "metadata": {
            "__erlab_spreadsheet_metadata_source__": {
                "type": "google_sheets",
                "share_url": "not a Google Sheets link",
            }
        }
    }

    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        with caplog.at_level(logging.WARNING, logger=workspace_loading.__name__):
            manager._workspace_controller.loading._restore_workspace_loader_state(
                {
                    "loader_state": {
                        "recent_name_filter": "Valid files (*.dat)",
                        "manager_loader_kwargs_by_filter": {
                            "Valid files (*.dat)": valid_kwargs,
                            "Invalid files (*.bad)": invalid_kwargs,
                        },
                        "explorer_loader_kwargs_by_name": {
                            "example": valid_kwargs,
                            "broken": invalid_kwargs,
                        },
                    }
                }
            )

        manager_kwargs = manager._recent_loader_kwargs_by_filter
        assert set(manager_kwargs) == {"Valid files (*.dat)"}
        assert isinstance(
            manager_kwargs["Valid files (*.dat)"]["metadata"],
            erlab.io.metadata.ExcelMetadataSource,
        )
        explorer_kwargs = (
            manager._workspace_controller._loader_state.explorer_loader_kwargs_by_name
        )
        assert set(explorer_kwargs) == {"example"}
        assert isinstance(
            explorer_kwargs["example"]["metadata"],
            erlab.io.metadata.ExcelMetadataSource,
        )
        assert (
            sum(
                record.name == workspace_loading.__name__
                and record.levelno == logging.WARNING
                for record in caplog.records
            )
            == 2
        )


def test_manager_workspace_import_does_not_restore_standalone_apps(
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


def test_manager_workspace_roundtrips_child_plot_appearance(
    qtbot,
    tmp_path: pathlib.Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
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
        manager._workspace_controller._mark_workspace_clean()

        histogram = child.hists[0]
        histogram.gradient.setColorMap(
            pg.colormap.get("plasma", source="matplotlib", skipCache=True)
        )
        histogram.gradient.sigGradientChangeFinished.emit(histogram.gradient)
        histogram.region.lines[0].sigDragged.emit(histogram.region.lines[0])
        histogram.setLevels(3.0, 18.0)
        histogram.sigLevelChangeFinished.emit(histogram)
        qtbot.wait_until(lambda: manager.is_workspace_modified)
        expected_positions, expected_colors = histogram.gradient.colorMap().getStops(
            pg.ColorMap.BYTE
        )

        fname = tmp_path / "child-plot-appearance.itws"
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

        node = manager._child_node(child_uid)
        if node.pending_workspace_payload is not None:
            assert node.materialize_pending_workspace_payload()
        restored = node.tool_window
        assert isinstance(restored, DerivativeTool)
        assert restored.hists[0].getLevels() == pytest.approx((3.0, 18.0))
        actual_positions, actual_colors = (
            restored.hists[0].gradient.colorMap().getStops(pg.ColorMap.BYTE)
        )
        np.testing.assert_allclose(actual_positions, expected_positions)
        np.testing.assert_array_equal(actual_colors, expected_colors)
        assert not manager.is_workspace_modified


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
        manager._data_ingress.open_multiple_files([fname], try_workspace=True)

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
