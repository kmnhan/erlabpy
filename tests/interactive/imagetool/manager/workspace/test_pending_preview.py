"""Pending workspace metadata and lightweight preview tests."""

from __future__ import annotations

import base64
import json
import types
import typing

import h5py
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._loading as workspace_loading
import erlab.interactive.imagetool.manager._workspace._pending as workspace_pending
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
import erlab.interactive.imagetool.plot_items as imagetool_plot_items
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._model import full_data
from tests.interactive.imagetool.manager.helpers import select_tools

if typing.TYPE_CHECKING:
    from collections.abc import Callable


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
        assert "Added" not in info_text
        assert "Size " not in info_text
        fields = {field.label: field.value for field in wrapper.metadata_fields}
        assert fields["Size"] == erlab.utils.formatting.format_nbytes(data.nbytes)
        assert not wrapper._workspace_reference_datasets
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
        info, data_size = pending_payloads._pending_workspace_imagetool_info(node)
    assert info is not None
    assert data_size == data.nbytes
    assert "pending" in info
    assert "float64 [2]" in info
    assert "float64 [3]" in info
    assert "float64 scalar" in info
    assert "Added" not in info
    assert "Size " not in info
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
        assert "Added" not in text
        assert loader.pending._pending_workspace_info(node) == (text, None)

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
        assert loader.pending._pending_workspace_info(no_pending) == (None, None)


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
        info, data_size = loader.pending._pending_workspace_imagetool_info(node)
        assert info is not None
        assert data_size == data.nbytes
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
        assert loader.pending._pending_workspace_imagetool_info(missing_node) == (
            None,
            None,
        )
        assert (
            loader.pending._pending_workspace_imagetool_preview_image(missing_node)
            is None
        )

        missing_data_fname = tmp_path / "pending-preview-missing-data.itws"
        assert workspace_arrays._write_workspace_dataset_group_h5py(
            missing_data_fname,
            "0/imagetool",
            xr.Dataset({"other": xr.DataArray([1.0], dims="x")}),
        )
        missing_data_node = types.SimpleNamespace(
            pending_workspace_memory_payload=(
                missing_data_fname,
                "0/imagetool",
            ),
            pending_workspace_payload_attrs=None,
            name="missing",
        )
        assert loader.pending._pending_workspace_imagetool_info(missing_data_node) == (
            None,
            None,
        )

        no_pending = types.SimpleNamespace(
            pending_workspace_memory_payload=None,
            pending_workspace_payload_attrs=None,
            name="missing",
            added_time_display="Today",
        )
        assert loader.pending._pending_workspace_imagetool_info(no_pending) == (
            None,
            None,
        )
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
