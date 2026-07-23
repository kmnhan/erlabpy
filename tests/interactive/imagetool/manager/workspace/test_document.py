import contextlib
import datetime
import json
import logging
import pathlib
import tempfile
import types
import typing
import warnings
from collections.abc import Callable

import h5py
import numpy as np
import pytest
import xarray
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.manager as manager_module
import erlab.interactive.imagetool.manager._desktop as manager_desktop
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._modelview as manager_modelview
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.viewer as imagetool_viewer
from erlab.interactive._fit1d import Fit1DTool
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._provenance._model import full_data
from erlab.interactive.imagetool._provenance._operations import GaussianFilterOperation
from erlab.interactive.imagetool.manager import ImageToolManager, fetch, replace_data
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ChooseFromWorkspaceManifestDialog,
)
from erlab.interactive.imagetool.manager._workspace import (
    _controller as workspace_controller,
)
from tests.interactive.imagetool.manager.helpers import (
    action_map_by_object_name,
    adopt_workspace_path,
    assert_fit_result_dataset_equivalent,
    assert_fit_result_list_equivalent,
    configure_goldtool_child,
    copy_full_code_for_uid,
    make_fit2d_child,
    select_child_tool,
    select_tools,
    set_transform_launch_mode,
)
from tests.interactive.imagetool.manager.workspace._support import (
    _AddedTimeChildTool,
    _compute_first_value,
    _open_external_lazy_hdf5_imagetool_data,
    _request_workspace_save_and_wait,
    _request_workspace_save_as_and_wait,
)


def _wait_for_fit_idle(qtbot, tool: Fit1DTool, *, timeout: int = 10000) -> None:
    def _fit_idle() -> bool:
        if tool._fit_thread is not None:
            return False
        if isinstance(tool, Fit2DTool):
            return tool._fit_2d_total == 0 and not tool._fit_2d_indices
        return True

    qtbot.wait_until(_fit_idle, timeout=timeout)


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
        manager._workspace_controller._mark_workspace_clean()

        manager._set_workspace_option_overrides(overrides)

        assert manager.is_workspace_modified
        assert manager._workspace_state.options_modified
        assert manager.workspace_option_overrides() == overrides

        fname = tmp_path / "option-overrides.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        manager._set_workspace_option_overrides({})
        manager._workspace_controller._mark_workspace_clean()

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller._mark_workspace_clean()

        manager.set_workspace_option_override("colors/cmap/name", "viridis")

        assert manager.workspace_option_overrides() == {"colors/cmap/name": "viridis"}
        assert manager.is_workspace_modified
        assert manager._workspace_state.options_modified

        manager.clear_workspace_option_override("colors/cmap/name")
        assert manager.workspace_option_overrides() == {}

        manager.clear_workspace_option_override("colors/cmap/name")
        assert manager.workspace_option_overrides() == {}


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
        assert next(
            field.value for field in node.metadata_fields if field.label == "Size"
        ) == erlab.utils.formatting.format_nbytes(test_data.nbytes)
        assert expected not in node.info_text
        assert "Size " not in node.info_text


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
    operation = GaussianFilterOperation(sigma={"x": 1.0})

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
                    for item in (
                        manager._workspace_controller.saving._workspace_node_manifest_entries()
                    )
                    if item["uid"] == uid
                )
                loader = manager._workspace_controller.loading
                snapshot = loader._workspace_data_backing_snapshot()

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
                    for entry in (
                        manager._workspace_controller.saving._workspace_node_manifest_entries()
                    )
                }
                loader = manager._workspace_controller.loading
                snapshot = loader._workspace_data_backing_snapshot()

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

        control = manager.get_imagetool(0).colormap_controls
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


def test_workspace_controller(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert not manager.is_workspace_modified

        manager.link_imagetools(0, 1, link_colors=False)
        assert manager.is_workspace_modified
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )

        manifest = workspace_format._workspace_manifest_from_attrs(
            workspace_arrays._read_workspace_root_attrs_h5py(fname)
        )
        linked_entries = [entry for entry in manifest["nodes"] if "link_group" in entry]
        assert {entry["path"] for entry in linked_entries} == {"0", "1"}
        assert {entry["link_group"] for entry in linked_entries} == {0}
        assert {entry["link_colors"] for entry in linked_entries} == {False}

        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._mark_workspace_clean()

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(2)]
        assert all(wrapper.slicer_area.is_linked for wrapper in wrappers)

        select_tools(manager, [0])
        manager.unlink_selected(deselect=False)

        for wrapper in wrappers:
            assert not wrapper.slicer_area.is_linked
            assert not wrapper.workspace_linked
            assert wrapper.workspace_link_key is None
            assert wrapper.uid in manager._workspace_state.dirty_state


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
        child.set_source_binding(full_data())
        child_uid = manager.add_childtool(child, 0, show=False)

        tree = manager._workspace_controller.saving._to_datatree()
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

            assert manager._workspace_controller.loading._from_datatree(
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
        child.set_source_binding(full_data())
        child_uid = manager.add_childtool(child, 0, show=False)

        tree = manager._workspace_controller.saving._to_datatree()
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
            manager._workspace_controller._update_workspace_window_title()

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
            manager._workspace_controller,
            "_set_node_window_modified",
            lambda uid, modified: node_modified_calls.append((uid, modified)),
        )

        manager._note_interaction_activity()
        assert manager._workspace_controller._mark_workspace_dirty(uid="n1", **dirty_kw)
        assert not manager._workspace_controller._mark_workspace_dirty(
            uid="n1", **dirty_kw
        )

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
        assert manager._workspace_controller._mark_workspace_dirty(uid="n1", state=True)
        assert manager._workspace_controller._mark_workspace_dirty(uid="n1", data=True)

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
            manager._workspace_controller._update_workspace_window_title()

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
                manager._workspace_controller._update_workspace_window_title()
        finally:
            manager._workspace_state.closing_document = previous_closing

        assert file_path_calls == [str(workspace)]
        assert workspace.name in manager.windowTitle()
        assert manager.isWindowModified()


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
                manager._workspace_controller._update_workspace_window_title()
        finally:
            manager._workspace_state.closing_document = previous_closing

        assert file_path_calls == [str(workspace)]


def test_workspace_controller_helper_branch_edges(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        controller = manager._workspace_controller
        loader = controller.loading
        saver = controller.saving

        invalid = xr.Dataset(attrs={"itool_state": b"not text"})
        assert (
            loader._dataset_without_missing_workspace_colormap(invalid, None) is invalid
        )
        invalid.attrs["itool_state"] = "{not-json"
        assert (
            loader._dataset_without_missing_workspace_colormap(invalid, None) is invalid
        )
        invalid.attrs["itool_state"] = json.dumps([])
        assert (
            loader._dataset_without_missing_workspace_colormap(invalid, None) is invalid
        )

        assert (
            loader._workspace_saved_uid_from_dataset(
                xr.Dataset(attrs={"manager_node_uid": b"node-1"})
            )
            == "node-1"
        )
        assert (
            loader._workspace_saved_uid_from_dataset(
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
                patch.setattr(saver, "_workspace_node_path", lambda uid: uid)
                assert saver._workspace_highest_dirty_data_roots() == [
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

    assert workspace_controller._WorkspaceController._normalize_recent_workspace_paths(
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
            erlab.interactive.imagetool.manager._workspace._controller,
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

        monkeypatch.setattr(
            manager._workspace_controller.loading, "_load_workspace_file", _record_load
        )
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
        with manager._workspace_controller._workspace_document_access_context(
            opened
        ) as access:
            manager._workspace_controller._associate_loaded_workspace_file(
                opened,
                workspace_format._current_workspace_schema_version(),
                workspace_access=access,
            )
        data = xr.DataArray(np.arange(4).reshape((2, 2)), dims=("x", "y"))
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: str(saved),
        )
        monkeypatch.setattr(
            manager._workspace_controller.loading,
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
            manager._workspace_controller.loading,
            "_load_workspace_file",
            lambda *_args, **_kwargs: True,
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
        manager._workspace_controller._refresh_manager_record()
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
            manager._data_ingress,
            "open_multiple_files",
            lambda paths, *, try_workspace=False: calls.append(
                (list(paths), try_workspace)
            ),
        )

        manager._handle_dropped_files([fname])

    assert calls == [([fname], False)]


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
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
        assert not any("space in its name" in str(item.message) for item in caught)

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        monkeypatch.setattr(
            workspace_arrays,
            "open_workspace_dataset",
            lambda *args, **kwargs: pytest.fail(
                "spaced numeric coords should stay on the h5py fast path"
            ),
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded = manager.get_imagetool(0).slicer_area._data
        assert loaded.chunks is None
        assert "Fake Motor" in loaded.coords
        xarray.testing.assert_equal(
            loaded.coords["Fake Motor"], data.coords["Fake Motor"]
        )


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
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
        details = manager._workspace_controller._dirty_details_text()
        assert "State modified:\n- renamed child" in details
        assert "Data modified:" not in details


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        manager._workspace_controller._drain_workspace_deferred_events()
        manager._workspace_controller._mark_workspace_clean()
        assert not manager.is_workspace_modified

        manager._workspace_state.closing_document = True
        root.hide()
        manager._workspace_state.closing_document = False
        manager._workspace_controller._drain_workspace_deferred_events()

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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        lazy = _open_external_lazy_hdf5_imagetool_data(fname)
        manager.get_imagetool(0).slicer_area.set_data(lazy, auto_compute=False)
        assert _compute_first_value(manager.get_imagetool(0).slicer_area._data) == 0
        manager._workspace_controller._mark_workspace_clean()

        errors: list[str] = []
        monkeypatch.setattr(manager, "_show_workspace_save_worker_error", errors.append)

        manager.rename_imagetool(0, "lazy state")
        assert not _request_workspace_save_and_wait(qtbot, manager)
        assert errors
        with contextlib.suppress(Exception):
            lazy.close()


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
        manager._workspace_controller._mark_workspace_clean()
        assert not manager.is_workspace_modified

        root.mnb.action_dict["toggleControlsAct"].trigger()
        assert not root.controls_visible
        qtbot.wait_until(lambda: manager.is_workspace_modified, timeout=5000)

        fname = tmp_path / "controls.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        adopt_workspace_path(manager, fname)
        assert not manager.is_workspace_modified

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        restored = manager.get_imagetool(0)
        assert not restored.controls_visible
        assert not restored.mnb.action_dict["toggleControlsAct"].isChecked()
        assert restored.slicer_area.state["controls_visible"] is False
        assert not manager.is_workspace_modified


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
        child.set_source_binding(full_data())
        child_uid = manager.add_childtool(child, 0, show=False)
        configure_goldtool_child(child, fitted=True, spline=True)

        expected_status = child.tool_status.model_copy(deep=True)
        expected_corrected = child.corrected.copy(deep=True)
        expected_source_spec = child.source_spec
        child.open_itool()
        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]

        tree = manager._workspace_controller.saving._to_datatree()
        assert (
            tree[f"0/childtools/{child_uid}/tool"].attrs["manager_node_uid"]
            == child_uid
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

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

        tree = manager._workspace_controller.saving._to_datatree()
        assert (
            tree[f"0/childtools/{child_uid}/tool"].attrs["manager_node_uid"]
            == child_uid
        )

        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

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

        tree = manager._workspace_controller.saving._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

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

        tree = manager._workspace_controller.saving._to_datatree()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        for node in tree.values():
            manager._workspace_controller.loading._load_workspace_node(
                typing.cast("xr.DataTree", node)
            )

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
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )

        assert not any("space in its name" in str(item.message) for item in caught)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        assert manager._workspace_controller.loading._load_workspace_file(
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

        tree = manager._workspace_controller.saving._to_datatree()
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

            loaded = workspace_arrays.open_workspace_datatree(filename, chunks="auto")
            try:
                assert manager._workspace_controller.loading._is_datatree_workspace(
                    loaded
                )
                assert loaded.attrs["imagetool_workspace_schema_version"] == 3
                for node in loaded.values():
                    manager._workspace_controller.loading._load_workspace_node(
                        typing.cast("xr.DataTree", node)
                    )
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
            manager._workspace_controller.loading._load_workspace_file(
                legacy_workspace,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )

        assert manager.ntools == 0
        assert legacy_workspace.read_bytes() == original_legacy_bytes
