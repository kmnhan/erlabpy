"""Pending workspace persistence and rebinding tests."""

from __future__ import annotations

import contextlib
import json
import types
import typing

import h5py
import numpy as np
import pytest
import xarray as xr
from qtpy import QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._loading as workspace_loading
import erlab.interactive.imagetool.manager._workspace._pending as workspace_pending
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._model import FileDataSelection, full_data
from erlab.interactive.imagetool._provenance._operations import (
    AverageOperation,
    ImageToolSelectionSourceBinding,
)
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    adopt_workspace_path,
    select_tools,
    trigger_menu_action,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.manager._workspace import (
        _controller as workspace_controller,
    )

from erlab.interactive.imagetool._provenance._code import uses_default_replay_input
from tests.interactive.imagetool.manager.workspace._support import (
    _AddedTimeChildTool,
    _hdf5_blosc2_level_codec,
    _request_workspace_save_and_wait,
    _request_workspace_save_as_and_wait,
    _WorkspaceManagerReferenceFigureTool,
)


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

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        reloaded_partner = manager.get_imagetool(1).slicer_area
        assert reloaded_partner.array_slicer.get_index(0, 0) == 3


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
        loader = controller.loading

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
            assert not loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "refresh", {}
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "view_all", {}
            )
            assert state["manual_limits"] == {}
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "center_all_cursors", {}
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "center_cursor", {}
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "set_current_cursor", {"cursor": 4}
            )
            assert state["current_cursor"] == 0
            assert not loader.pending._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_axis_inverted",
                {"dim": "missing", "inverted": True},
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_axis_inverted",
                {"dim": "x", "inverted": True},
            )
            assert state["axis_inversions"] == {"x": True}
            assert loader.pending._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_axis_inverted",
                {"dim": "x", "inverted": False},
            )
            assert state["axis_inversions"] == {}
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "set_index", {"axis": 0, "value": 4}
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "step_index", {"axis": 0, "value": -1}
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "step_index_all", {"axis": 1, "value": 1}
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state,
                array_slicer,
                source,
                "set_value",
                {"axis": 0, "value": 2.0, "uniform": True},
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "set_bin", {"axis": 0, "value": 3}
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "set_bin_all", {"axis": 1, "value": 3}
            )
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "swap_axes", {"ax1": 0, "ax2": 1}
            )
            assert loader.pending._update_pending_link_state_for_operation(
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
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "toggle_snap", {}
            )
            assert state["slice"]["snap_to_data"] is True
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "toggle_snap", {"value": False}
            )
            assert state["slice"]["snap_to_data"] is False
            assert not loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "unknown_operation", {}
            )
        finally:
            array_slicer.deleteLater()

        array_slicer, state = new_state(cursors=2)
        try:
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "add_cursor", {"color": None}
            )
            assert len(state["cursor_colors"]) == 3
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "add_cursor", {"color": "#123456"}
            )
            assert state["cursor_colors"][-1] == "#123456"
            state["current_cursor"] = 2
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "remove_cursor", {"index": 1}
            )
            assert state["current_cursor"] == 1
        finally:
            array_slicer.deleteLater()

        array_slicer, state = new_state(cursors=3)
        try:
            state["current_cursor"] = 0
            assert loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "remove_cursor", {"index": 0}
            )
            assert state["current_cursor"] == 0
            assert loader.pending._pending_link_default_cursor_color(
                source, 0, [QtGui.QColor(color).name() for color in source.COLORS]
            )
        finally:
            array_slicer.deleteLater()

        array_slicer, state = new_state()
        try:
            assert not loader.pending._update_pending_link_state_for_operation(
                state, array_slicer, source, "remove_cursor", {"index": 0}
            )
        finally:
            array_slicer.deleteLater()


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

        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )
        reloaded_partner = manager.get_imagetool(1).slicer_area
        assert reloaded_partner.manual_limits == expected_limits


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
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
            with h5py.File(fname, "r") as h5_file:
                data_ds = h5_file["0/imagetool"][_ITOOL_DATA_NAME]
                assert _hdf5_blosc2_level_codec(data_ds) is None
            assert manager._workspace_controller.loading._load_workspace_file(
                fname, replace=True, associate=True, mark_dirty=False, select=False
            )

            wrapper = manager._tool_graph.root_wrappers[0]
            assert wrapper.pending_workspace_memory_payload is not None
            wrapper.name = "renamed pending"
            assert wrapper.uid in manager._workspace_state.dirty_state
            assert wrapper.uid not in manager._workspace_state.dirty_data

            if force_fallback:
                monkeypatch.setattr(
                    manager._workspace_controller.saving,
                    "_workspace_full_save_manifest_first_snapshot",
                    lambda *_args, **_kwargs: None,
                )

            erlab.interactive.options["io/workspace/compression"] = "zstd1"
            manager._workspace_controller.saving._save_workspace_document(
                fname,
                force_full=True,
                require_matching_compression=True,
            )

            with h5py.File(fname, "r") as h5_file:
                group = h5_file["0/imagetool"]
                data_ds = group[_ITOOL_DATA_NAME]
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
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
            assert manager._workspace_controller.loading._load_workspace_file(
                fname, replace=True, associate=True, mark_dirty=False, select=False
            )

            wrapper = manager._tool_graph.root_wrappers[0]
            assert wrapper.pending_workspace_memory_payload is not None
            wrapper.name = "renamed pending"
            monkeypatch.setattr(
                manager._workspace_controller.saving,
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

            manager._workspace_controller.saving._save_workspace_document(
                fname,
                force_full=True,
                require_matching_compression=True,
            )

            assert messages == []
            assert wrapper.pending_workspace_memory_payload is None
            with h5py.File(fname, "r") as h5_file:
                assert h5_file["0/imagetool"].attrs["itool_name"] == "renamed pending"
                np.testing.assert_array_equal(
                    h5_file["0/imagetool"][_ITOOL_DATA_NAME][...],
                    data.values,
                )
    finally:
        erlab.interactive.options["io/workspace/compression"] = old_compression


def test_pending_workspace_lazy_source_data_matches_saved_semantic_order(
    qtbot, tmp_path
) -> None:
    data = xr.DataArray(
        np.arange(2 * 3 * 4, dtype=np.float64).reshape((2, 3, 4)),
        dims=("x", "hv", "y"),
        coords={
            "x": np.array([0.0, 1.0]),
            "hv": np.array([10.0, 20.0, 30.0]),
            "y": np.array([-1.0, 0.0, 1.0, 2.0]),
        },
        name="pending_order",
    )
    tool = erlab.interactive.imagetool.ImageTool(data)
    qtbot.addWidget(tool)
    saved = tool.to_dataset()
    state = json.loads(saved.attrs["itool_state"])
    assert tuple(state["slice"]["dims"]) == data.dims

    stored = xr.Dataset(
        {_ITOOL_DATA_NAME: saved[_ITOOL_DATA_NAME].transpose("hv", "y", "x")},
        attrs=dict(saved.attrs),
    )
    fname = tmp_path / "pending-saved-dim-order.itws"
    assert workspace_arrays._write_workspace_dataset_group_h5py(
        fname, "0/imagetool", stored
    )
    node = types.SimpleNamespace(
        pending_workspace_memory_payload=(fname, "0/imagetool"),
        pending_workspace_payload_attrs=None,
        name="pending_order",
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
        assert pending_source.chunks is not None
        pending_source = pending_source.compute()
        pending_displayed = pending_displayed.compute()
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
    finally:
        loaded_ds.close()


def test_pending_workspace_saved_dim_order_handles_invalid_state(monkeypatch) -> None:
    pending_cls = workspace_pending._PendingWorkspacePayloads
    data = xr.DataArray(
        np.arange(6, dtype=np.float64).reshape((2, 3)),
        dims=("x", "y"),
    )
    valid_state = {"slice": {"dims": ["y", "x"]}}

    reordered = pending_cls._pending_workspace_data_with_saved_dim_order(
        data, {"itool_state": json.dumps(valid_state).encode()}
    )
    assert reordered.dims == ("y", "x")
    np.testing.assert_array_equal(reordered.values, data.transpose("y", "x").values)

    assert (
        pending_cls._pending_workspace_data_with_saved_dim_order(
            data, {"itool_state": b"\xff"}
        )
        is data
    )
    assert (
        pending_cls._pending_workspace_data_with_saved_dim_order(
            data, {"itool_state": "{"}
        )
        is data
    )
    assert (
        pending_cls._pending_workspace_data_with_saved_dim_order(
            data, {"itool_state": "[]"}
        )
        is data
    )
    assert (
        pending_cls._pending_workspace_data_with_saved_dim_order(
            data, {"itool_state": json.dumps({"slice": []})}
        )
        is data
    )
    assert (
        pending_cls._pending_workspace_data_with_saved_dim_order(
            data, {"itool_state": json.dumps({"slice": {"dims": "xy"}})}
        )
        is data
    )

    def _raise_transpose(self: xr.DataArray, *_args, **_kwargs) -> xr.DataArray:
        raise ValueError("bad dim order")

    monkeypatch.setattr(xr.DataArray, "transpose", _raise_transpose)
    assert (
        pending_cls._pending_workspace_data_with_saved_dim_order(
            data, {"itool_state": json.dumps(valid_state)}
        )
        is data
    )


def test_pending_workspace_source_data_decodes_saved_state_attrs(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data_name = _ITOOL_DATA_NAME
    payload = xr.Dataset(
        {data_name: xr.DataArray(np.arange(3.0), dims=("x",), name=data_name)}
    )

    node = types.SimpleNamespace(
        name="restored",
        pending_workspace_payload_attrs={
            "itool_state": json.dumps({"filter_operation": None}).encode()
        },
        _finalize_script_input_data=lambda data: data.copy(deep=False),
    )
    with manager_context() as manager:
        loader = manager._workspace_controller.loading
        monkeypatch.setattr(
            loader,
            "_workspace_imagetool_reference_dataset",
            lambda *_args, **_kwargs: payload,
        )

        loaded = loader.pending._pending_workspace_lazy_source_data(node)
        assert loaded.name == "restored"
        np.testing.assert_array_equal(loaded.values, np.arange(3.0))

        node.pending_workspace_payload_attrs = {"itool_state": "{"}
        assert (
            loader.pending._pending_workspace_lazy_source_data(node).name == "restored"
        )

        node.pending_workspace_payload_attrs = {"itool_state": "[]"}
        assert (
            loader.pending._pending_workspace_lazy_source_data(node).name == "restored"
        )


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
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()

        node = manager._tool_graph.root_wrappers[0]
        node.set_detached_provenance(full_data(AverageOperation(dims=("alpha",))))

        fname = tmp_path / "pending-file-source-replay.itws"
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
            pytest.fail("copying pending full code should not materialize data")

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
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
        assert not uses_default_replay_input(copied[-1])
        namespace = _exec_generated_code(copied[-1], {})
        xr.testing.assert_identical(
            namespace["derived"].rename(None),
            test_data.qsel.mean("alpha").rename(None),
        )
        assert wrapper.pending_workspace_memory_payload is not None
        assert wrapper.imagetool is None


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            saved, force_full=True
        )

        assert child_node.pending_workspace_tool_payload is not None
        assert figure_node.pending_workspace_tool_payload is not None
        with h5py.File(saved, "r") as h5_file:
            assert f"0/childtools/{child_uid}/tool" in h5_file
            assert f"figures/{figure_uid}/tool" in h5_file


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        with h5py.File(fname, "r+") as h5_file:
            h5_file[f"0/childtools/{child_uid}/tool"].attrs["tool_cls_qualname"] = (
                "erlab.interactive._missing_tool_module:MissingTool"
            )

        skipped: list[tuple[str | None, Exception]] = []
        original_record = (
            manager._workspace_controller.loading._record_skipped_workspace_node
        )

        def _record_skipped(node_path: str | None, exc: Exception) -> None:
            skipped.append((node_path, exc))
            original_record(node_path, exc)

        monkeypatch.setattr(
            manager._workspace_controller.loading,
            "_record_skipped_workspace_node",
            _record_skipped,
        )

        assert manager._workspace_controller.loading._load_workspace_file(
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
        binding = ImageToolSelectionSourceBinding(
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
            manager._workspace_controller.loading.pending._workspace_tool_source_metadata(
                attrs
            )
        )

        assert source_spec is None
        assert source_binding == binding
        assert auto_update is True
        assert state == "stale"

        attrs[erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR] = "not-valid"
        *_, state = (
            manager._workspace_controller.loading.pending._workspace_tool_source_metadata(
                attrs
            )
        )
        assert state == "fresh"


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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrapper = manager._tool_graph.root_wrappers[0]
        assert wrapper.pending_workspace_memory_payload is not None

        duplicate_index = manager.duplicate_imagetool(0)

        assert wrapper.pending_workspace_memory_payload is None
        duplicated = manager.get_imagetool(duplicate_index).slicer_area
        assert workspace_arrays.dataarray_is_numpy_backed(duplicated._data)
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
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            source, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded_wrapper = manager._tool_graph.root_wrappers[0]
        assert loaded_wrapper.pending_workspace_memory_payload is not None

        loaded_figure = manager.get_childtool(figure_uid)
        assert isinstance(loaded_figure, _WorkspaceManagerReferenceFigureTool)
        source_paths = workspace_arrays.dataarray_source_paths(loaded_figure.tool_data)
        assert str(source.resolve()) in source_paths

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target,
        )
        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)

        rebound_paths = workspace_arrays.dataarray_source_paths(loaded_figure.tool_data)
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
        manager._workspace_controller.saving._save_workspace_document(
            source, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded_wrapper = manager._tool_graph.root_wrappers[0]
        loaded_figure = manager.get_childtool(figure_uid)
        assert isinstance(loaded_figure, _WorkspaceManagerReferenceFigureTool)
        original_paths = workspace_arrays.dataarray_source_paths(
            loaded_figure.tool_data
        )
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
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target,
        )
        manager._workspace_controller._mark_node_state_dirty(figure_uid)

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
        assert workspace_arrays.dataarray_source_paths(loaded_figure.tool_data) == (
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
        manager._workspace_controller.saving._save_workspace_document(
            source, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
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
        manager._workspace_controller.saving._save_workspace_document(
            source, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            source, replace=True, associate=True, mark_dirty=False, select=False
        )
        loaded_figure = manager.get_childtool(figure_uid)
        assert isinstance(loaded_figure, _WorkspaceManagerReferenceFigureTool)
        assert str(source.resolve()) in workspace_arrays.dataarray_source_paths(
            loaded_figure.tool_data
        )

        loaded_figure._reference_uid = None
        manager._workspace_controller._mark_node_state_dirty(figure_uid)
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target,
        )

        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)
        rebound_paths = workspace_arrays.dataarray_source_paths(loaded_figure.tool_data)
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
        manager._workspace_controller.saving._save_workspace_document(
            source, force_full=True
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        base_data = xr.DataArray(np.arange(4.0).reshape((2, 2)), dims=["x", "y"])
        base_root = itool(base_data, manager=False, execute=False)
        assert isinstance(base_root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(base_root, show=False)
        base = tmp_path / "import-base.itws"
        manager._workspace_controller.saving._save_workspace_document(
            base, force_full=True
        )
        adopt_workspace_path(manager, base)
        manager._workspace_controller._mark_workspace_clean()

        assert manager._workspace_controller.loading._load_workspace_file(
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
        payload_path = manager._workspace_controller.saving._workspace_payload_path(
            wrapper.uid
        )

        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: target,
        )
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
                "network full save should not materialize hidden memory payloads"
            )

        monkeypatch.setattr(
            manager._workspace_controller.loading.pending,
            "_materialize_pending_workspace_payload",
            _fail_materialize_pending_payload,
        )
        monkeypatch.setattr(
            workspace_storage,
            "_workspace_path_is_likely_network_path",
            lambda _path: True,
        )

        wrapper.name = "network-renamed"
        manager._workspace_controller.saving._write_full_workspace_file(fname)
        assert wrapper.pending_workspace_memory_payload is not None

        with h5py.File(fname, "r") as h5_file:
            group = h5_file["0/imagetool"]
            assert group.attrs["itool_name"] == "network-renamed"
            np.testing.assert_array_equal(
                group[_ITOOL_DATA_NAME][...],
                data.values,
            )
