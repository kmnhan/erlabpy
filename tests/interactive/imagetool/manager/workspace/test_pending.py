"""Pending workspace data ownership and materialization tests."""

from __future__ import annotations

import contextlib
import json
import types
import typing

import h5py
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool._serialization as imagetool_serialization
import erlab.interactive.imagetool.dialogs as imagetool_dialogs
import erlab.interactive.imagetool.manager._console as manager_console
import erlab.interactive.imagetool.manager._lineage as manager_lineage
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._modelview as manager_modelview
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._loading as workspace_loading
import erlab.interactive.imagetool.manager._workspace._pending as workspace_pending
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._wrapper as manager_wrapper
import erlab.interactive.imagetool.viewer as imagetool_viewer
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ToolProvenanceOperation,
    compose_display_provenance,
    full_data,
    script,
    selection,
)
from erlab.interactive.imagetool._provenance._operations import (
    AverageOperation,
    BoxcarFilterOperation,
    GaussianFilterOperation,
    SortCoordOrderOperation,
    SqueezeOperation,
)
from erlab.interactive.imagetool.manager import replace_data
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    select_child_tool,
    select_tools,
    trigger_menu_action,
)

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Mapping

    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.manager._workspace import (
        _controller as workspace_controller,
    )
from tests.interactive.imagetool.manager.workspace._support import (
    _AddedTimeChildTool,
    _open_external_file_backed_hdf5_imagetool_data,
    _request_workspace_save_and_wait,
    _transaction_test_root_attrs,
    _WorkspaceManagerReferenceFigureTool,
    _WorkspaceSweepChildTool,
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


def test_manager_workspace_link_badge_colors_do_not_depend_on_materialization(
    qtbot,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100, 200, 300):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
            root.hide()
        manager.link_imagetools(0, 1, link_colors=False)
        manager.link_imagetools(2, 3, link_colors=False)

        fname = tmp_path / "pending-link-badge-colors.itws"
        manager._workspace_controller.saving._save_workspace_document(
            fname, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            fname, replace=True, associate=True, mark_dirty=False, select=False
        )

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(4)]
        first_key = wrappers[0].workspace_link_key
        second_key = wrappers[2].workspace_link_key
        assert first_key is not None
        assert second_key is not None
        assert first_key != second_key
        first_color = manager.color_for_workspace_link_key(first_key)
        second_color = manager.color_for_workspace_link_key(second_key)
        color_cache = manager._workspace_link_color_indices
        assert first_color != second_color
        assert color_cache is not None

        manager.get_imagetool(2)
        manager.get_imagetool(3)

        assert wrappers[0].pending_workspace_memory_payload is not None
        assert wrappers[1].pending_workspace_memory_payload is not None
        second_proxy = wrappers[2].slicer_area._linking_proxy
        assert second_proxy is not None
        assert manager.color_for_workspace_link_key(first_key) == first_color
        assert manager.color_for_workspace_link_key(second_key) == second_color
        assert manager.color_for_linker(second_proxy) == second_color
        assert manager._workspace_link_color_indices is color_cache


def test_manager_workspace_link_badge_color_cache_reuses_and_invalidates(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in (0, 100, 200, 300):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
        manager.link_imagetools(0, 1, link_colors=False)
        manager.link_imagetools(2, 3, link_colors=False)

        wrappers = [manager._tool_graph.root_wrappers[index] for index in range(4)]
        first_key = wrappers[0].workspace_link_key
        second_key = wrappers[2].workspace_link_key
        assert first_key is not None
        assert second_key is not None

        reconcile_calls = 0
        original_reconcile = manager._reconcile_workspace_link_color_cache

        def _count_reconcile() -> dict[str, int]:
            nonlocal reconcile_calls
            reconcile_calls += 1
            return original_reconcile()

        monkeypatch.setattr(
            manager,
            "_reconcile_workspace_link_color_cache",
            _count_reconcile,
        )
        manager._invalidate_workspace_link_color_cache()

        first_color = manager.color_for_workspace_link_key(first_key)
        second_color = manager.color_for_workspace_link_key(second_key)
        color_cache = manager._workspace_link_color_indices
        assert first_color != second_color
        assert reconcile_calls == 1
        assert color_cache is not None
        for _ in range(3):
            assert manager.color_for_workspace_link_key(first_key) == first_color
            assert manager.color_for_workspace_link_key(second_key) == second_color
        assert manager._workspace_link_color_indices is color_cache
        assert reconcile_calls == 1

        with monkeypatch.context() as patch:
            patch.setattr(
                manager,
                "color_for_linker",
                lambda _proxy: pytest.fail(
                    "managed badge painting should use its structural link key"
                ),
            )
            index = manager.tree_view._model.index(2, 0)
            option = manager.tree_view._delegate._option_for_index(
                manager.tree_view, index
            )
            canvas = QtGui.QPixmap(200, 32)
            canvas.fill(QtGui.QColor("white"))
            painter = QtGui.QPainter(canvas)
            try:
                manager.tree_view._delegate.paint(painter, option, index)
            finally:
                painter.end()
        assert manager._workspace_link_color_indices is color_cache
        assert reconcile_calls == 1

        manager._actions_controller.unlink_imagetool_nodes(wrappers[:2])
        assert manager._workspace_link_color_cache_dirty
        assert manager.color_for_workspace_link_key(second_key) == second_color
        assert reconcile_calls == 2

        manager.link_imagetools(0, 1, link_colors=False)
        first_key = wrappers[0].workspace_link_key
        assert first_key is not None
        assert manager._workspace_link_color_cache_dirty
        assert manager.color_for_workspace_link_key(
            first_key
        ) != manager.color_for_workspace_link_key(second_key)
        assert reconcile_calls == 3


def test_manager_workspace_link_badge_colors_survive_member_removal(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        for offset in range(5):
            data = xr.DataArray(
                np.arange(offset, offset + 25, dtype=np.float64).reshape((5, 5)),
                dims=["x", "y"],
            )
            root = itool(data, manager=False, execute=False)
            assert isinstance(root, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(root, show=False)
        manager.link_imagetools(0, 2, 4, link_colors=False)
        manager.link_imagetools(1, 3, link_colors=False)

        first_key = manager._tool_graph.root_wrappers[0].workspace_link_key
        second_key = manager._tool_graph.root_wrappers[1].workspace_link_key
        assert first_key is not None
        assert second_key is not None
        first_color = manager.color_for_workspace_link_key(first_key)
        second_color = manager.color_for_workspace_link_key(second_key)
        assert first_color != second_color

        manager.remove_imagetool(0)

        assert manager._tool_graph.root_wrappers[2].workspace_link_key == first_key
        assert manager._tool_graph.root_wrappers[4].workspace_link_key == first_key
        assert manager._workspace_link_color_cache_dirty
        assert manager.color_for_workspace_link_key(first_key) == first_color
        assert manager.color_for_workspace_link_key(second_key) == second_color
        assert not manager._workspace_link_color_cache_dirty


def test_manager_workspace_link_badge_palette_exhaustion_recovers_free_color() -> None:
    palette_size = len(manager_mainwindow._LINKER_COLORS)
    assert palette_size > 1
    link_keys = [f"group-{index}" for index in range(palette_size + 1)]
    nodes = {
        f"node-{index}": types.SimpleNamespace(workspace_link_key=link_key)
        for index, link_key in enumerate(link_keys)
    }
    manager = types.SimpleNamespace(
        _tool_graph=types.SimpleNamespace(nodes=nodes),
        _workspace_link_color_indices={},
        _workspace_link_color_cache_dirty=True,
    )

    initial_indices = (
        manager_mainwindow.ImageToolManager._reconcile_workspace_link_color_cache(
            manager
        )
    )

    assert len(set(initial_indices.values())) == palette_size
    assert initial_indices[link_keys[0]] == initial_indices[link_keys[-1]]

    nodes.pop("node-1")
    recovered_indices = (
        manager_mainwindow.ImageToolManager._reconcile_workspace_link_color_cache(
            manager
        )
    )

    assert len(set(recovered_indices.values())) == palette_size
    assert recovered_indices[link_keys[-1]] == initial_indices[link_keys[1]]
    assert recovered_indices[link_keys[0]] == initial_indices[link_keys[0]]
    for link_key in link_keys[2:-1]:
        assert recovered_indices[link_key] == initial_indices[link_key]


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

        removed_node = manager._child_node(child_uids[0])
        pending = removed_node.pending_workspace_memory_payload
        attrs = removed_node.pending_workspace_payload_attrs
        assert pending is not None
        assert attrs is not None
        cached_slicer = erlab.interactive.imagetool.slicer.ArraySlicer(
            root_data, removed_node
        )
        removed_node._pending_workspace_link_slicer_cache = (
            pending,
            str(attrs["itool_state"]),
            cached_slicer,
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
        assert removed_node._pending_workspace_link_slicer_cache is None
        qtbot.wait_until(
            lambda: not erlab.interactive.utils.qt_is_valid(cached_slicer),
            timeout=5000,
        )
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
    class PendingNode(QtCore.QObject):
        uid = "pending"
        is_imagetool = True

        def __init__(self, parent: QtCore.QObject) -> None:
            super().__init__(parent)
            self.attrs: dict[str, typing.Any] | None = None
            self.pending_workspace_memory_payload: tuple[pathlib.Path, str] | None = (
                tmp_path / "source.itws",
                "0/imagetool",
            )
            self._pending_workspace_link_slicer_cache = None

        @property
        def pending_workspace_payload_attrs(self) -> dict[str, typing.Any] | None:
            return None if self.attrs is None else dict(self.attrs)

        def update_pending_workspace_payload_attrs(
            self, attrs: Mapping[str, typing.Any]
        ) -> None:
            self.attrs = dict(attrs)

        def _clear_pending_workspace_link_slicer_cache(self) -> None:
            cache = self._pending_workspace_link_slicer_cache
            self._pending_workspace_link_slicer_cache = None
            if cache is not None and erlab.interactive.utils.qt_is_valid(cache[2]):
                cache[2].deleteLater()

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

        node = PendingNode(manager)
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

            cached_slicer = node._pending_workspace_link_slicer_cache[2]
            updated_state["slice"]["indices"][0][0] = 4
            updated_state["slice"]["values"][0][0] = 4.0
            resynced_state_json = json.dumps(updated_state)
            node.attrs = {"itool_state": resynced_state_json}
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
            cache = node._pending_workspace_link_slicer_cache
            assert cache == (
                node.pending_workspace_memory_payload,
                resynced_state_json,
                cached_slicer,
            )
            assert cached_slicer.get_index(0, 0) == 4

            node._clear_pending_workspace_link_slicer_cache()
            with monkeypatch.context() as patch:
                patch.setattr(
                    controller.loading,
                    "_read_workspace_imagetool_payload_dataset",
                    lambda *_args, **_kwargs: xr.Dataset({"other": data}),
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
            assert node._pending_workspace_link_slicer_cache is None

            with monkeypatch.context() as patch:
                patch.setattr(
                    erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy,
                    "convert_args",
                    lambda _self, *_args, **_kwargs: None,
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

            node._clear_pending_workspace_link_slicer_cache()
            with monkeypatch.context() as patch:
                patch.setattr(
                    loader.pending,
                    "_update_pending_link_state_for_operation",
                    lambda *_args, **_kwargs: True,
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
        finally:
            node._clear_pending_workspace_link_slicer_cache()
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
            manager.color_for_workspace_link_key("unknown-link-key").getRgb()[:3]
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
                imagetool_dialogs.DataFilterDialog,
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

    class _FilterDialog(imagetool_dialogs.DataFilterDialog):
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
