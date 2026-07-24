import json
import typing
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._figurecomposer._tool as figurecomposer_tool_module
import erlab.interactive._stylesheets
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._saving as workspace_saving
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
from erlab.interactive._figurecomposer import (
    FigureComposerTool,
    FigureOperationState,
    FigureRecipeState,
    FigureSourceState,
)
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._model import FileDataSelection
from erlab.interactive.imagetool.manager._figurecomposer import _collection
from tests.interactive.imagetool.manager.helpers import (
    _exec_generated_code,
    activate_widget_shortcut,
    adopt_workspace_path,
    select_child_tool,
    select_tools,
    trigger_menu_action,
)
from tests.interactive.imagetool.manager.workspace._support import (
    _request_workspace_save_as_and_wait,
)

from ._common import (
    _materialized_figure_tool,
    _restored_figure_composer_from_netcdf,
    _selection_shortcut_sequences,
    _SourcePickerDummyTool,
)


class _GalleryPreviewTool(_SourcePickerDummyTool):
    manager_collection = "figures"
    tool_name = "gallery-preview"

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__(data)
        self._preview = QtGui.QPixmap(40, 20)
        self._preview.fill(QtGui.QColor("red"))

    @property
    def preview_pixmap(self) -> QtGui.QPixmap:
        return self._preview


def _figure_pane(manager):
    pane = manager._figure_collection.pane
    assert pane is not None
    return pane


def test_manager_figures_ui_is_lazy_and_figures_survive_source_removal(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        assert not manager.left_tabs.tabBar().isVisible()
        assert not manager.left_tabs.isTabVisible(1)
        assert not hasattr(manager, "figure_tab")
        assert not hasattr(manager, "figure_view_controls")

        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        select_tools(manager, [0])
        manager.create_figure_action.trigger()
        assert len(manager._tool_graph.figure_uids) == 1
        figure_uid = manager._tool_graph.figure_uids[0]
        assert manager.left_tabs.tabBar().isVisible()
        assert manager.left_tabs.indexOf(_figure_pane(manager)) == 1
        assert manager.left_tabs.isTabVisible(1)
        assert figure_uid in manager._tool_graph.figure_uids
        assert figure_uid not in manager._tool_graph.root_wrappers[0]._childtool_indices

        manager.remove_imagetool(0)
        assert figure_uid in manager._tool_graph.nodes
        assert _figure_pane(manager).list_widget.count() == 1
        assert manager.dependency_status_for_uid(figure_uid) == "missing"
        assert manager.left_tabs.tabBar().isVisible()

        manager._remove_childtool(figure_uid)
        assert figure_uid not in manager._tool_graph.nodes
        assert not manager.left_tabs.tabBar().isVisible()
        assert not manager.left_tabs.isTabVisible(1)
        assert not hasattr(manager, "figure_tab")


def test_manager_tool_selection_clears_stale_figure_selection_details(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None

        manager._figure_collection.select_uid(figure_uid)

        assert manager._selected_tool_uids() == [figure_uid]

        select_tools(manager, [0])

        root_uid = manager._tool_graph.root_wrappers[0].uid
        assert manager._selected_imagetool_targets() == [0]
        assert manager._selected_tool_uids() == []
        assert manager._selected_figure_uids() == []
        assert manager._metadata_node_uid == root_uid


def test_manager_figures_tab_does_not_set_empty_minimum_width(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        manager.show()
        empty_width = manager.left_tabs.minimumSizeHint().width()

        assert manager.left_tabs.count() == 1
        assert not hasattr(manager, "figure_tab")

        figure_uid = manager.add_figuretool(FigureComposerTool(data), show=False)

        assert manager.left_tabs.count() == 2
        assert manager.left_tabs.indexOf(_figure_pane(manager)) == 1
        manager._figure_collection._show_menu(QtCore.QPoint())
        menu = manager._figure_collection._menu
        assert isinstance(menu, QtWidgets.QMenu)
        qtbot.wait_until(menu.isVisible, timeout=5000)

        manager._remove_childtool(figure_uid)
        qtbot.wait_until(lambda: manager._figure_collection._menu is None, timeout=5000)

        assert manager.left_tabs.count() == 1
        assert not hasattr(manager, "figure_tab")
        assert manager.left_tabs.minimumSizeHint().width() == empty_width

        manager._figure_collection.clear_selection_from_tree()
        manager._figure_collection._selection_changed()
        manager._figure_collection._show_item(QtWidgets.QListWidgetItem("removed"))
        manager._figure_collection._show_menu(QtCore.QPoint())


def test_manager_figure_menu_helpers_release_stale_wrappers(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._figure_collection.close_menu()
        assert manager._figure_collection._menu is None

        menu = QtWidgets.QMenu(manager)
        manager._figure_collection._menu = menu
        manager._figure_collection.close_menu()
        assert manager._figure_collection._menu is None

        stale_menu = QtWidgets.QMenu(manager)
        manager._figure_collection._menu = stale_menu
        original_qt_is_valid = erlab.interactive.utils.qt_is_valid

        def fake_qt_is_valid(*objects: object) -> bool:
            if any(obj is stale_menu for obj in objects):
                return False
            return original_qt_is_valid(*objects)

        with monkeypatch.context() as patch:
            patch.setattr(erlab.interactive.utils, "qt_is_valid", fake_qt_is_valid)
            manager._figure_collection.close_menu()
            assert manager._figure_collection._menu is None
            manager._figure_collection._release_menu(stale_menu)


def test_manager_figure_menu_releases_menu_without_viewport(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        manager.add_figuretool(FigureComposerTool(data), show=False)
        monkeypatch.setattr(
            type(_figure_pane(manager).list_widget), "viewport", lambda _self: None
        )

        manager._figure_collection._show_menu(QtCore.QPoint())

        assert manager._figure_collection._menu is None


@pytest.mark.parametrize(
    ("platform", "rename_shortcut", "show_shortcut", "expected_shortcuts"),
    [
        (
            "darwin",
            "Return",
            "Ctrl+Down",
            {"Return", "Enter", "Ctrl+Down"},
        ),
        (
            "linux",
            "F2",
            "Return",
            {"F2", "Return", "Enter"},
        ),
    ],
)
def test_manager_figure_list_selection_shortcuts_are_platform_native(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    platform: str,
    rename_shortcut: str,
    show_shortcut: str,
    expected_shortcuts: set[str],
) -> None:
    monkeypatch.setattr(manager_mainwindow.sys, "platform", platform)
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )

    with manager_context() as manager:
        figure_uid = manager.add_figuretool(FigureComposerTool(data), show=False)
        manager._figure_collection.select_uid(figure_uid)

        assert manager.show_action.shortcut().isEmpty()
        assert (
            _selection_shortcut_sequences(_figure_pane(manager).list_widget)
            == expected_shortcuts
        )

        activate_widget_shortcut(_figure_pane(manager).list_widget, rename_shortcut)
        qtbot.wait_until(
            lambda: (
                _figure_pane(manager).list_widget.state()
                == QtWidgets.QAbstractItemView.State.EditingState
            ),
            timeout=5000,
        )
        editor = _figure_pane(manager).list_widget.findChild(QtWidgets.QLineEdit)
        assert editor is not None
        editor.setText(f"{platform}_figure")
        qtbot.keyClick(editor, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: manager._child_node(figure_uid).name == f"{platform}_figure",
            timeout=5000,
        )

        shown: list[str] = []
        monkeypatch.setattr(manager, "show_childtool", shown.append)
        activate_widget_shortcut(_figure_pane(manager).list_widget, show_shortcut)
        qtbot.wait_until(lambda: shown == [figure_uid], timeout=5000)


def test_manager_figures_gallery_view_preserves_selection_and_persists(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        first_uid = manager.add_figuretool(FigureComposerTool(data), show=False)
        second_uid = manager.add_figuretool(FigureComposerTool(data), show=False)
        manager._figure_collection.select_uid(first_uid)

        assert (
            _figure_pane(manager).list_widget.viewMode()
            == QtWidgets.QListView.ViewMode.IconMode
        )
        assert (
            _figure_pane(manager).list_widget.verticalScrollMode()
            == QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        assert _figure_pane(manager).gallery_size_combo.isVisible()
        assert (
            _figure_pane(manager).list_button.toolButtonStyle()
            == QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        assert (
            _figure_pane(manager).gallery_button.toolButtonStyle()
            == QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        assert not _figure_pane(manager).list_button.icon().isNull()
        assert not _figure_pane(manager).gallery_button.icon().isNull()

        _figure_pane(manager).list_button.click()
        assert (
            _figure_pane(manager).list_widget.viewMode()
            == QtWidgets.QListView.ViewMode.ListMode
        )
        assert _figure_pane(manager).gallery_size_combo.isHidden()
        _figure_pane(manager).gallery_button.click()

        assert (
            _figure_pane(manager).list_widget.viewMode()
            == QtWidgets.QListView.ViewMode.IconMode
        )
        assert (
            _figure_pane(manager).list_widget.verticalScrollMode()
            == QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        assert _figure_pane(manager).gallery_size_combo.isVisible()
        assert manager._selected_figure_uids() == [first_uid]
        for row, uid in enumerate((first_uid, second_uid)):
            item = _figure_pane(manager).list_widget.item(row)
            assert item is not None
            assert item.data(QtCore.Qt.ItemDataRole.UserRole) == uid
            assert not item.icon().isNull()

        old_grid_size = _figure_pane(manager).list_widget.gridSize()
        large_index = _figure_pane(manager).gallery_size_combo.findData("large")
        assert large_index >= 0
        _figure_pane(manager).gallery_size_combo.setCurrentIndex(large_index)
        assert (
            _figure_pane(manager).list_widget.gridSize().width() > old_grid_size.width()
        )
        assert manager._selected_figure_uids() == [first_uid]

        shown: list[str] = []
        monkeypatch.setattr(manager, "show_childtool", shown.append)
        manager._figure_collection._show_item(_figure_pane(manager).list_widget.item(0))
        assert shown == [first_uid]

        _figure_pane(manager).list_button.click()
        assert (
            _figure_pane(manager).list_widget.viewMode()
            == QtWidgets.QListView.ViewMode.ListMode
        )
        assert _figure_pane(manager).gallery_size_combo.isHidden()
        _figure_pane(manager).gallery_button.click()

    with manager_context() as restored_manager:
        restored_uid = restored_manager.add_figuretool(
            FigureComposerTool(data), show=False
        )
        assert _figure_pane(restored_manager).gallery_button.isChecked()
        assert (
            _figure_pane(restored_manager).gallery_size_combo.currentData() == "large"
        )
        assert (
            _figure_pane(restored_manager).list_widget.viewMode()
            == QtWidgets.QListView.ViewMode.IconMode
        )
        item = _figure_pane(restored_manager).list_widget.item(0)
        assert item is not None
        assert item.data(QtCore.Qt.ItemDataRole.UserRole) == restored_uid
        assert not item.icon().isNull()


def test_manager_figures_gallery_reuses_cached_preview_for_size_changes(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        manager.add_figuretool(figure_tool, show=False)
        assert figure_tool.refresh_preview_pixmap(allow_offscreen=True) is not None

        def fail_preview_update(*_args, **_kwargs) -> None:
            pytest.fail("gallery thumbnail updates must not render the recipe")

        monkeypatch.setattr(
            figure_tool, "request_preview_pixmap_update", fail_preview_update
        )
        monkeypatch.setattr(figure_tool, "refresh_preview_pixmap", fail_preview_update)

        _figure_pane(manager).gallery_button.click()
        old_grid_size = _figure_pane(manager).list_widget.gridSize()
        size_name = (
            "large"
            if _figure_pane(manager).gallery_size_combo.currentData() != "large"
            else "small"
        )
        size_index = _figure_pane(manager).gallery_size_combo.findData(size_name)
        assert size_index >= 0
        _figure_pane(manager).gallery_size_combo.setCurrentIndex(size_index)

        assert _figure_pane(manager).list_widget.gridSize() != old_grid_size


def test_figure_composer_persists_compact_preview_cache_without_rendering(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    first = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="first",
    )
    second = xr.DataArray(
        np.arange(4.0, 8.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="second",
    )
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="First"),
            FigureSourceState(name="second", label="Second"),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)
    assert tool.refresh_preview_pixmap(allow_offscreen=True) is not None

    def fail_render(*_args, **_kwargs) -> None:
        pytest.fail("saving a preview cache must not render the figure")

    monkeypatch.setattr(tool, "refresh_preview_pixmap", fail_render)
    monkeypatch.setattr(tool, "_cache_live_canvas_preview", fail_render)
    monkeypatch.setattr(tool, "_fallback_preview_pixmap", fail_render)

    ds = tool.to_dataset()
    encoded_cache = ds.attrs.get(
        figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR
    )
    assert isinstance(encoded_cache, str)
    max_encoded_cache_size = 4 * (
        (figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_MAX_BYTES + 2) // 3
    )
    assert len(encoded_cache) <= max_encoded_cache_size

    monkeypatch.setattr(figurecomposer_tool_module, "_render_preview", fail_render)
    restored = _restored_figure_composer_from_netcdf(tool, qtbot, tmp_path)
    restored_preview = restored.preview_pixmap
    assert restored_preview is not None
    assert not restored.preview_pixmap_stale
    xr.testing.assert_identical(restored.source_data()["second"], second)
    assert (
        restored_preview.width()
        <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE[0]
    )
    assert (
        restored_preview.height()
        <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE[1]
    )
    thumbnail = restored._preview_thumbnail_pixmap(QtCore.QSize(64, 64))
    assert thumbnail is not None
    assert not thumbnail.isNull()


def test_figure_composer_visible_restore_queues_auto_redraw(
    qtbot,
    monkeypatch,
) -> None:
    first = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="first",
    )
    second = xr.DataArray(
        np.arange(4.0, 8.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="second",
    )
    tool = FigureComposerTool.from_sources(
        {"first": first, "second": second},
        sources=(
            FigureSourceState(name="first", label="First"),
            FigureSourceState(name="second", label="Second"),
        ),
        primary_source="first",
    )
    qtbot.addWidget(tool)

    render_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_render(*args, **kwargs) -> None:
        render_calls.append((args, kwargs))

    monkeypatch.setattr(figurecomposer_tool_module, "_render_preview", record_render)

    ds = tool.to_dataset()
    window_state = json.loads(ds.attrs["tool_window_state"])
    window_state["visible"] = True
    ds.attrs["tool_window_state"] = json.dumps(window_state)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(ds)
    qtbot.addWidget(restored)

    qtbot.wait_until(lambda: bool(render_calls), timeout=5000)
    assert render_calls == [((restored,), {"show_window": True})]


def test_figure_composer_deferred_restore_delays_visible_redraw(
    qtbot,
    monkeypatch,
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    tool = FigureComposerTool.from_sources(
        {"line": data},
        sources=(FigureSourceState(name="line", label="line"),),
        operations=(FigureOperationState.line(label="line", source="line"),),
        primary_source="line",
    )
    qtbot.addWidget(tool)
    ds = tool.to_dataset()
    window_state = json.loads(ds.attrs["tool_window_state"])
    window_state["visible"] = True
    ds.attrs["tool_window_state"] = json.dumps(window_state)

    render_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_render(*args, **kwargs) -> None:
        render_calls.append((args, kwargs))

    monkeypatch.setattr(figurecomposer_tool_module, "_render_preview", record_render)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        ds,
        _defer_restore_work=True,
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    assert render_calls == []

    restored.show()

    qtbot.wait_until(lambda: bool(render_calls), timeout=5000)
    assert render_calls == [((restored,), {"show_window": True})]


def _line_figure_composer_restore_dataset(qtbot):
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    tool = FigureComposerTool.from_sources(
        {"line": data},
        sources=(FigureSourceState(name="line", label="line"),),
        operations=(FigureOperationState.line(label="line", source="line"),),
        primary_source="line",
    )
    qtbot.addWidget(tool)
    return tool.to_dataset(), data, tool.tool_status


def _record_figure_composer_editor_updates(monkeypatch) -> list[FigureComposerTool]:
    editor_calls: list[FigureComposerTool] = []
    original = FigureComposerTool._update_operation_editor

    def record_editor_update(self: FigureComposerTool) -> None:
        editor_calls.append(self)
        original(self)

    monkeypatch.setattr(
        FigureComposerTool, "_update_operation_editor", record_editor_update
    )
    return editor_calls


def test_figure_composer_deferred_restore_delays_operation_editor(
    qtbot,
    monkeypatch,
) -> None:
    ds, _data, _status = _line_figure_composer_restore_dataset(qtbot)
    editor_calls = _record_figure_composer_editor_updates(monkeypatch)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        ds,
        _defer_restore_work=True,
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    assert editor_calls == []

    restored.show()

    qtbot.wait_until(lambda: editor_calls == [restored], timeout=5000)


def test_figure_composer_generated_code_flushes_deferred_operation_editor(
    qtbot,
    monkeypatch,
) -> None:
    ds, _data, _status = _line_figure_composer_restore_dataset(qtbot)
    editor_calls = _record_figure_composer_editor_updates(monkeypatch)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        ds,
        _defer_restore_work=True,
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, FigureComposerTool)
    assert editor_calls == []

    code = restored.generated_code()

    assert "line" in code
    assert editor_calls == [restored]


def test_figure_composer_save_skips_deferred_operation_editor(
    qtbot,
    monkeypatch,
) -> None:
    ds, data, status = _line_figure_composer_restore_dataset(qtbot)

    with monkeypatch.context() as patch:

        def fail_editor_update(_self: FigureComposerTool) -> None:
            pytest.fail(
                "saving hidden deferred Figure Composer should not build editor"
            )

        patch.setattr(
            FigureComposerTool, "_update_operation_editor", fail_editor_update
        )
        restored = erlab.interactive.utils.ToolWindow.from_dataset(
            ds,
            _defer_restore_work=True,
        )
        qtbot.addWidget(restored)
        assert isinstance(restored, FigureComposerTool)
        saved = restored.to_dataset()
        restored._discard_restore_work(
            key=figurecomposer_tool_module._RESTORE_OPERATION_EDITOR_KEY
        )

    loaded = erlab.interactive.utils.ToolWindow.from_dataset(saved)
    qtbot.addWidget(loaded)
    assert isinstance(loaded, FigureComposerTool)
    assert loaded.tool_status.model_dump(mode="json") == status.model_dump(mode="json")
    xr.testing.assert_identical(loaded.source_data()["line"], data)


def test_figure_composer_flushes_restore_work_before_user_outputs(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    calls: list[str] = []

    monkeypatch.setattr(tool, "_flush_restore_work", lambda: calls.append("flush"))
    monkeypatch.setattr(
        tool.operation_editor,
        "flush_pending_commits",
        lambda: calls.append("commit"),
    )
    monkeypatch.setattr(tool, "_warn_invalid_operation_targets", lambda: False)
    monkeypatch.setattr(
        erlab.interactive._figurecomposer._codegen,
        "generated_code",
        lambda _tool: "figure_code",
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda code: calls.append(f"copy:{code}"),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", ""),
    )

    assert tool.generated_code() == "figure_code"
    tool.copy_code()
    tool.export_figure()
    tool.current_provenance_spec(flush_deferred_restore=False)

    assert calls == [
        "flush",
        "commit",
        "flush",
        "flush",
        "commit",
        "copy:figure_code",
        "flush",
        "commit",
    ]


def test_figure_composer_skips_preview_cache_when_unrendered(
    qtbot, monkeypatch
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    tool = FigureComposerTool(data)
    qtbot.addWidget(tool)
    assert tool.preview_pixmap is None

    def fail_render(*_args, **_kwargs) -> None:
        pytest.fail("saving without a preview cache must not render the figure")

    monkeypatch.setattr(tool, "refresh_preview_pixmap", fail_render)
    monkeypatch.setattr(tool, "_cache_live_canvas_preview", fail_render)
    monkeypatch.setattr(tool, "_fallback_preview_pixmap", fail_render)

    ds = tool.to_dataset()

    assert figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_ATTR not in ds.attrs
    assert tool.preview_pixmap is None


def test_manager_workspace_restores_figure_gallery_preview_cache(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    workspace_path = tmp_path / "figure-preview-cache.itws"
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        assert figure_tool.refresh_preview_pixmap(allow_offscreen=True) is not None

        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded_node = manager._child_node(figure_uid)
        assert loaded_node.tool_window is None
        assert loaded_node.pending_workspace_tool_payload is not None
        pending_preview = loaded_node.pending_workspace_tool_preview_image()
        assert pending_preview is not None
        assert not pending_preview[1].isNull()
        loaded_node.show()
        loaded_tool = loaded_node.tool_window
        assert isinstance(loaded_tool, FigureComposerTool)
        assert loaded_tool.preview_pixmap is not None
        assert not loaded_tool.preview_pixmap_stale

        _figure_pane(manager).gallery_button.click()
        item = manager._figure_collection.item_for_uid(figure_uid)
        assert item is not None
        assert not item.icon().isNull()


def test_manager_figures_gallery_helpers_handle_invalid_sources(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )

    class FakeSettings:
        def value(self, key: str, default: str) -> object:
            if key.endswith("view_mode"):
                return "invalid"
            if key.endswith("gallery_thumbnail_size"):
                return "huge"
            return 1

        def setValue(self, key: str, value: str) -> None:
            pass

    monkeypatch.setattr(_collection, "_manager_settings", lambda: FakeSettings())
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        assert _figure_pane(manager).gallery_button.isChecked()
        assert _figure_pane(manager).gallery_size_combo.currentData() == "medium"
        _figure_pane(manager).gallery_button.click()

        assert not manager._figure_collection._gallery_icon("missing").isNull()

        high_dpi_preview = QtGui.QPixmap(400, 100)
        high_dpi_preview.setDevicePixelRatio(2.0)
        high_dpi_preview.fill(QtGui.QColor("red"))
        high_dpi_thumbnail = manager._figure_collection._thumbnail_pixmap(
            high_dpi_preview
        )
        high_dpi_image = high_dpi_thumbnail.toImage().convertToFormat(
            QtGui.QImage.Format.Format_ARGB32
        )
        red_pixels: list[tuple[int, int]] = []
        for y_pos in range(high_dpi_image.height()):
            for x_pos in range(high_dpi_image.width()):
                color = high_dpi_image.pixelColor(x_pos, y_pos)
                if color.red() > 220 and color.green() < 40 and color.blue() < 40:
                    red_pixels.append((x_pos, y_pos))
        assert red_pixels
        min_x = min(x_pos for x_pos, _y_pos in red_pixels)
        max_x = max(x_pos for x_pos, _y_pos in red_pixels)
        min_y = min(y_pos for _x_pos, y_pos in red_pixels)
        max_y = max(y_pos for _x_pos, y_pos in red_pixels)
        assert high_dpi_thumbnail.size() == _figure_pane(manager).list_widget.iconSize()
        assert max_x - min_x + 1 == high_dpi_thumbnail.width()
        assert abs((max_y - min_y + 1) - round(high_dpi_thumbnail.width() / 4)) <= 1
        assert abs(((min_y + max_y) / 2) - ((high_dpi_thumbnail.height() - 1) / 2)) <= 1

        requested: list[None] = []
        monkeypatch.setattr(
            figure_tool,
            "request_preview_pixmap_update",
            lambda: requested.append(None),
        )
        figure_tool._preview_pixmap_stale = True
        assert not manager._figure_collection._gallery_icon(figure_uid).isNull()
        assert requested == []


def test_manager_figures_gallery_uses_generic_tool_preview(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
    with manager_context() as manager:
        figure_uid = manager.add_figuretool(_GalleryPreviewTool(data), show=False)
        pane = _figure_pane(manager)
        pane.gallery_button.click()
        item = manager._figure_collection.item_for_uid(figure_uid)
        assert item is not None

        icon_image = item.icon().pixmap(pane.list_widget.iconSize()).toImage()
        center = icon_image.pixelColor(
            icon_image.width() // 2, icon_image.height() // 2
        )
        assert center.red() > 220
        assert center.green() < 40
        assert center.blue() < 40


def test_manager_figures_gallery_updates_one_icon_from_preview_signal(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        _figure_pane(manager).gallery_button.click()
        item = manager._figure_collection.item_for_uid(figure_uid)
        assert item is not None
        old_cache_key = item.icon().cacheKey()

        pixmap = QtGui.QPixmap(32, 24)
        pixmap.fill(QtGui.QColor("red"))
        figure_tool._preview_pixmap_cache = pixmap
        figure_tool._preview_pixmap_generation += 1
        figure_tool._preview_thumbnail_cache.clear()
        figure_tool._preview_pixmap_stale = False

        def fail_sync(*_args, **_kwargs) -> None:
            pytest.fail("preview updates should update the changed gallery item only")

        with monkeypatch.context() as patch:
            patch.setattr(manager._figure_collection, "sync", fail_sync)
            figure_tool.sigInfoChanged.emit()

        qtbot.wait_until(lambda: item.icon().cacheKey() != old_cache_key, timeout=1000)


def test_manager_figure_selection_defers_preview_generation(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        refresh_calls: list[None] = []
        request_calls: list[int] = []

        def record_refresh() -> None:
            refresh_calls.append(None)

        def record_request(*, delay_ms: int = 250) -> None:
            request_calls.append(delay_ms)

        monkeypatch.setattr(figure_tool, "refresh_preview_pixmap", record_refresh)
        monkeypatch.setattr(
            figure_tool, "request_preview_pixmap_update", record_request
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        manager._figure_collection.select_uid(figure_uid)

        assert refresh_calls == []
        assert request_calls == []
        assert not manager.preview_widget.isVisible()

        qtbot.waitUntil(lambda: request_calls == [0], timeout=1000)
        assert refresh_calls == []


def test_manager_figure_selection_keeps_preview_aspect_ratio(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0),
        dims=("x",),
        coords={"x": np.arange(4.0)},
        name="line",
    )
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        preview_pixmap = QtGui.QPixmap(160, 80)
        preview_pixmap.fill(QtGui.QColor("red"))
        figure_tool._preview_pixmap_cache = preview_pixmap
        figure_tool._preview_pixmap_generation += 1
        figure_tool._preview_thumbnail_cache.clear()
        figure_tool._preview_pixmap_stale = False
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        manager.preview_widget.resize(120, 300)
        manager._figure_collection.select_uid(figure_uid)

        assert manager.preview_widget.isVisible()
        transform = manager.preview_widget.transform()
        assert transform.m11() == pytest.approx(transform.m22())


def test_manager_copy_full_code_for_file_backed_figure_composer_sources(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": np.arange(3), "y": np.arange(3)},
        name="first",
    )
    second = first + 10.0
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    first.to_netcdf(first_path, engine="h5netcdf")
    second.to_netcdf(second_path, engine="h5netcdf")

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        itool(
            first,
            manager=True,
            file_path=first_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        itool(
            second,
            manager=True,
            file_path=second_path,
            load_func=(
                xr.load_dataarray,
                {"engine": "h5netcdf"},
                FileDataSelection(kind="dataarray"),
            ),
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        manager.tree_view.clearSelection()
        select_child_tool(manager, figure_uid)
        manager._update_info(uid=figure_uid)
        assert manager.metadata_derivation_list.topLevelItemCount() == 4
        assert manager.metadata_derivation_list.count() == 6
        child_counts: list[int] = []
        for row in range(manager.metadata_derivation_list.topLevelItemCount()):
            item = manager.metadata_derivation_list.topLevelItem(row)
            assert item is not None
            child_counts.append(item.childCount())
        assert child_counts == [0, 1, 1, 0]

        copied: list[str] = []
        monkeypatch.setattr(
            erlab.interactive.utils,
            "copy_to_clipboard",
            lambda text: copied.append(text) or text,
        )
        monkeypatch.setattr(
            manager,
            "_prompt_replay_input_name",
            lambda _node: pytest.fail("file-backed figure replay should not prompt"),
        )
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert copied
        namespace = _exec_generated_code(copied[-1], {})
        assert isinstance(namespace["fig"], Figure)


def test_manager_copy_full_code_for_memory_figure_reports_unavailable(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dialogs: list[typing.Any] = []

    class _RecordingMessageDialog:
        def __init__(self, parent=None, **kwargs) -> None:
            self.parent = parent
            self.kwargs = kwargs
            dialogs.append(self)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        manager.tree_view.clearSelection()
        select_child_tool(manager, figure_uid)
        manager._update_info(uid=figure_uid)
        assert manager.metadata_derivation_list.count() == 3

        copied: list[str] = []
        monkeypatch.setattr(erlab.interactive.utils, "copy_to_clipboard", copied.append)
        menu = manager._build_metadata_derivation_menu()
        assert menu is not None
        trigger_menu_action(menu, manager._metadata_copy_full_action)
        assert not copied
        assert len(dialogs) == 1
        assert (
            dialogs[0].kwargs["icon_pixmap"]
            == QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
        )
        assert dialogs[0].kwargs["text"]
        assert dialogs[0].kwargs["informative_text"]
        assert dialogs[0].kwargs["detailed_text"]


def test_manager_workspace_figure_sources_save_as_references(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        name="first",
    )
    second = xr.DataArray(
        np.arange(9.0, 18.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        name="second",
    )

    with manager_context() as manager:
        for data in (first, second):
            tool = erlab.interactive.itool(data, manager=False, execute=False)
            assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(tool, show=False)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None

        tree = manager._workspace_controller.saving._to_datatree()
        try:
            ds = typing.cast(
                "xr.DataTree", tree[f"figures/{figure_uid}/tool"]
            ).to_dataset(inherit=False)
            references = json.loads(
                ds.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR]
            )

            assert erlab.interactive.utils._SAVED_TOOL_DATA_NAME in references
            assert "second" in references
            assert ds[erlab.interactive.utils._SAVED_TOOL_DATA_NAME].size == 0
            assert ds["second"].size == 0
            assert workspace_arrays._workspace_dataset_can_write_h5py(ds)

            source_data_by_uid = {
                references[erlab.interactive.utils._SAVED_TOOL_DATA_NAME][
                    "node_uid"
                ]: first,
                references["second"]["node_uid"]: second,
            }
            corrupt_ds = ds.drop_vars("second")
            restored_from_reference = erlab.interactive.utils.ToolWindow.from_dataset(
                corrupt_ds,
                _tool_data_reference_resolver=lambda reference: source_data_by_uid.get(
                    reference.get("node_uid")
                ),
            )
            qtbot.addWidget(restored_from_reference)
            assert isinstance(restored_from_reference, FigureComposerTool)
            xr.testing.assert_identical(
                restored_from_reference.source_data()["second"], second
            )

            fname = tmp_path / "figure-source-references.itws"
            manager._workspace_controller.saving._save_workspace_document(
                fname, force_full=True
            )
            saved_ds = workspace_arrays._read_workspace_dataset_group_h5py(
                fname,
                f"figures/{figure_uid}/tool",
                preferred_data_name=erlab.interactive.utils._SAVED_TOOL_DATA_NAME,
            )
            assert saved_ds is not None
            assert "second" in saved_ds

            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

            assert manager._from_datatree(
                tree, replace=True, mark_dirty=False, select=False
            )
            manager.remove_all_tools()
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
            assert manager._workspace_controller.loading._load_workspace_file(
                fname,
                replace=True,
                associate=False,
                mark_dirty=False,
                select=False,
            )
            loaded_node = manager._tool_graph.nodes[figure_uid]
            assert loaded_node.pending_workspace_tool_payload is not None
            loaded_tool = _materialized_figure_tool(manager, figure_uid)
            xr.testing.assert_identical(loaded_tool.source_data()["second"], second)
        finally:
            tree.close()

        restored = _materialized_figure_tool(manager, figure_uid)
        source_data = restored.source_data()
        xr.testing.assert_identical(source_data["first"], first)
        xr.testing.assert_identical(source_data["second"], second)


def test_manager_figure_workspace_reference_helper_edges(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        dims=("x", "y"),
        name="data",
    )

    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None

        controller = manager._workspace_controller
        loader = controller.loading
        saver = controller.saving
        root_node = manager._node_for_target(0)
        figure_node = manager._child_node(figure_uid)
        stale_revision = "stale-snapshot"
        current_reference = {
            "kind": "manager_node",
            "node_uid": root_node.uid,
            "node_snapshot_token": root_node.snapshot_token,
            "data_role": "displayed",
        }

        assert not controller._tool_data_reference_matches_current_snapshot(
            {"kind": "manager_node", "node_uid": ""}
        )
        assert not controller._tool_data_reference_matches_current_snapshot(
            {
                "kind": "manager_node",
                "node_uid": root_node.uid,
                "data_role": "invalid",
            }
        )

        def unavailable_data(_data_role: str) -> xr.DataArray:
            raise RuntimeError("unavailable")

        with monkeypatch.context() as context:
            context.setattr(root_node, "data_for_role", unavailable_data)
            assert not controller._tool_data_reference_matches_current_data(
                current_reference, data
            )

        snapshot = workspace_saving._WorkspaceSaveSnapshot(
            generation=0,
            root_attrs={},
            delta_save_count=0,
            serialized_tool_data_references=(
                ("missing", {}),
                (root_node.uid, {}),
            ),
        )
        controller._commit_saved_tool_data_references(snapshot)

        invalid_tool_dataset = xr.Dataset(
            attrs={"manager_node_kind": "tool", "manager_node_uid": ""}
        )
        assert saver._serialized_tool_data_references((invalid_tool_dataset,)) == ()
        assert figure_uid not in saver._workspace_stale_reference_rewrite_uids(
            frozenset(manager._tool_graph.nodes)
        )

        assert (
            loader._saved_workspace_reference_source_data(
                figure_node,
                snapshot_token=stale_revision,
                data_role="displayed",
                owner_node=None,
                reference_datasets={},
            )
            is None
        )
        assert (
            loader._saved_workspace_reference_source_data(
                root_node,
                snapshot_token=stale_revision,
                data_role="displayed",
                owner_node=None,
                reference_datasets={},
            )
            is None
        )

        workspace_path = tmp_path / "reference-helper-edges.itws"
        saver._save_workspace_document(workspace_path, force_full=True)
        adopt_workspace_path(manager, workspace_path)
        reference_datasets: dict[tuple[Path, str], xr.Dataset] = {}
        try:
            assert (
                loader._saved_workspace_reference_source_data(
                    root_node,
                    snapshot_token=stale_revision,
                    data_role="displayed",
                    owner_node=figure_node,
                    reference_datasets=reference_datasets,
                )
                is None
            )
            _parent_data, resolver = loader._workspace_tool_restore_references(
                xr.Dataset(),
                parent_target=None,
                owner_node=figure_node,
                reference_datasets=reference_datasets,
            )
            xr.testing.assert_identical(
                resolver(
                    {
                        "kind": "manager_node",
                        "node_uid": root_node.uid,
                        "data_role": "displayed",
                    }
                ),
                root_node.data_for_role("displayed"),
            )
            xr.testing.assert_identical(
                resolver(
                    {
                        **current_reference,
                        "node_snapshot_token": stale_revision,
                    }
                ),
                root_node.data_for_role("displayed"),
            )
        finally:
            loader._close_workspace_reference_datasets(reference_datasets)


def test_manager_workspace_embeds_figure_snapshot_after_source_transpose(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("kx", "eV"),
        coords={
            "kx": [-0.5, 0.0, 0.5],
            "eV": [-1.0, -0.5, 0.0, 0.5],
        },
        name="cut",
    )

    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = _materialized_figure_tool(manager, figure_uid)
        primary_source = figure_tool.tool_status.primary_source
        captured = figure_tool.source_data()[primary_source]
        assert captured.dims == data.dims

        [operation] = figure_tool.tool_status.operations
        figure_tool.tool_status = figure_tool.tool_status.model_copy(
            update={"operations": (operation.model_copy(update={"transpose": True}),)}
        )
        source_state = figure_tool.source_states()[0]
        root_node = manager._tool_graph.root_wrappers[0]
        assert source_state.node_snapshot_token == root_node.snapshot_token

        workspace_path = tmp_path / "figure-source-transpose.itws"
        monkeypatch.setattr(
            manager._workspace_controller,
            "_workspace_save_dialog",
            lambda **_kwargs: workspace_path,
        )
        assert _request_workspace_save_as_and_wait(qtbot, manager, native=False)
        saved_figure = workspace_arrays._read_workspace_dataset_group_h5py(
            workspace_path,
            f"figures/{figure_uid}/tool",
            preferred_data_name=erlab.interactive.utils._SAVED_TOOL_DATA_NAME,
        )
        assert saved_figure is not None
        saved_references = FigureComposerTool._saved_tool_data_references(saved_figure)
        assert erlab.interactive.utils._SAVED_TOOL_DATA_NAME in saved_references
        assert (
            manager._child_node(figure_uid)._workspace_tool_data_references
            == saved_references
        )
        manager._workspace_controller._mark_workspace_clean()

        root.slicer_area.transpose_main_image()

        assert root_node.data_for_role("displayed").dims == ("eV", "kx")
        assert root_node.snapshot_token != source_state.node_snapshot_token
        assert figure_tool.source_data()[primary_source].dims == data.dims

        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=False
        )
        rewritten_figure = workspace_arrays._read_workspace_dataset_group_h5py(
            workspace_path,
            f"figures/{figure_uid}/tool",
            preferred_data_name=erlab.interactive.utils._SAVED_TOOL_DATA_NAME,
        )
        assert rewritten_figure is not None
        rewritten_references = FigureComposerTool._saved_tool_data_references(
            rewritten_figure
        )
        assert erlab.interactive.utils._SAVED_TOOL_DATA_NAME not in rewritten_references
        assert (
            rewritten_figure[erlab.interactive.utils._SAVED_TOOL_DATA_NAME].size
            == data.size
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )

        restored = _materialized_figure_tool(manager, figure_uid)
        restored_source = restored.source_data()[restored.tool_status.primary_source]
        xr.testing.assert_identical(restored_source, data)
        assert restored.tool_status.operations[0].transpose is True

        figure = Figure()
        figurecomposer_tool_module._render_into_figure(
            restored, figure, sync_visible=False
        )
        assert figure.axes[0].images[0].get_array().shape == data.T.shape


def test_manager_failed_save_keeps_last_saved_figure_references(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("kx", "eV"),
        coords={
            "kx": [-0.5, 0.0, 0.5],
            "eV": [-1.0, -0.5, 0.0, 0.5],
        },
        name="cut",
    )

    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        _materialized_figure_tool(manager, figure_uid)
        figure_node = manager._child_node(figure_uid)

        workspace_path = tmp_path / "failed-figure-source-transpose.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        adopt_workspace_path(manager, workspace_path)
        saved_references = dict(figure_node._workspace_tool_data_references)
        assert erlab.interactive.utils._SAVED_TOOL_DATA_NAME in saved_references
        manager._workspace_controller._mark_workspace_clean()

        root.slicer_area.transpose_main_image()

        def _raise_write_error(*_args, **_kwargs) -> None:
            raise RuntimeError("write failed")

        monkeypatch.setattr(
            workspace_storage,
            "_write_workspace_transaction_file",
            _raise_write_error,
        )

        with pytest.raises(RuntimeError, match="write failed"):
            manager._workspace_controller.saving._save_workspace_document(
                workspace_path, force_full=False
            )

        assert figure_node._workspace_tool_data_references == saved_references


def test_manager_full_save_fallback_preserves_figure_source_snapshot(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("kx", "eV"),
        coords={
            "kx": [-0.5, 0.0, 0.5],
            "eV": [-1.0, -0.5, 0.0, 0.5],
        },
        name="cut",
    )

    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure = _materialized_figure_tool(manager, figure_uid)
        source_name = figure.tool_status.primary_source

        source_path = tmp_path / "full-save-source.itws"
        manager._workspace_controller.saving._save_workspace_document(
            source_path, force_full=True
        )
        adopt_workspace_path(manager, source_path)
        manager._workspace_controller._mark_workspace_clean()

        root.slicer_area.transpose_main_image()
        monkeypatch.setattr(
            manager._workspace_controller.saving,
            "_workspace_full_save_manifest_first_snapshot",
            lambda *_args, **_kwargs: None,
        )
        target_path = tmp_path / "full-save-target.itws"
        manager._workspace_controller.saving._save_workspace_document(
            target_path, force_full=True
        )

        saved_figure = workspace_arrays._read_workspace_dataset_group_h5py(
            target_path,
            f"figures/{figure_uid}/tool",
            preferred_data_name=erlab.interactive.utils._SAVED_TOOL_DATA_NAME,
        )
        assert saved_figure is not None
        assert (
            erlab.interactive.utils._SAVED_TOOL_DATA_NAME
            not in FigureComposerTool._saved_tool_data_references(saved_figure)
        )
        assert manager._child_node(figure_uid)._workspace_tool_data_references == {}

        assert manager._workspace_controller.loading._load_workspace_file(
            target_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        restored = _materialized_figure_tool(manager, figure_uid)
        xr.testing.assert_identical(restored.source_data()[source_name], data)


def test_manager_incremental_save_preserves_pending_figure_snapshot(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("kx", "eV"),
        coords={"kx": [-0.5, 0.0, 0.5], "eV": [-1.0, -0.5, 0.0, 0.5]},
        name="pending_cut",
    )

    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None

        workspace_path = tmp_path / "pending-figure-source-transpose.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=True,
            mark_dirty=False,
            select=False,
        )

        source_node = manager._tool_graph.root_wrappers[0]
        figure_node = manager._child_node(figure_uid)
        assert figure_node.pending_workspace_tool_payload is not None
        if source_node.pending_workspace_memory_payload is not None:
            assert source_node.materialize_pending_workspace_payload()
        assert source_node.imagetool is not None
        assert figure_node.pending_workspace_tool_payload is not None

        source_node.slicer_area.transpose_main_image()
        assert source_node.data_for_role("displayed").dims == ("eV", "kx")

        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=False
        )
        rewritten_figure = workspace_arrays._read_workspace_dataset_group_h5py(
            workspace_path,
            f"figures/{figure_uid}/tool",
            preferred_data_name=erlab.interactive.utils._SAVED_TOOL_DATA_NAME,
        )
        assert rewritten_figure is not None
        assert (
            erlab.interactive.utils._SAVED_TOOL_DATA_NAME
            not in FigureComposerTool._saved_tool_data_references(rewritten_figure)
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        restored = _materialized_figure_tool(manager, figure_uid)
        restored_source = restored.source_data()[restored.tool_status.primary_source]
        xr.testing.assert_identical(restored_source, data)


def test_manager_pending_figure_source_reference_uses_saved_imagetool_dim_order(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
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

    with manager_context() as manager:
        root = itool(data, manager=False, execute=False)
        assert isinstance(root, erlab.interactive.imagetool.ImageTool)
        manager.add_imagetool(root, show=False)
        root.hide()
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None

        workspace_path = tmp_path / "pending-figure-source-order.itws"
        manager._workspace_controller.saving._save_workspace_document(
            workspace_path, force_full=True
        )
        saved_ds = workspace_arrays._read_workspace_dataset_group_h5py(
            workspace_path,
            "0/imagetool",
            preferred_data_name=_ITOOL_DATA_NAME,
        )
        assert saved_ds is not None
        stored = xr.Dataset(
            {_ITOOL_DATA_NAME: saved_ds[_ITOOL_DATA_NAME].transpose("hv", "y", "x")},
            attrs=dict(saved_ds.attrs),
        )
        with h5py.File(workspace_path, "r+") as h5_file:
            del h5_file["0/imagetool"]
        assert workspace_arrays._write_workspace_dataset_group_h5py(
            workspace_path, "0/imagetool", stored
        )

        assert manager._workspace_controller.loading._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )

        source_node = manager._tool_graph.root_wrappers[0]
        assert source_node.pending_workspace_memory_payload is not None
        figure_node = manager._child_node(figure_uid)
        assert figure_node.pending_workspace_tool_payload is not None
        figure_node.show()
        loaded_figure = figure_node.tool_window
        assert isinstance(loaded_figure, FigureComposerTool)
        source = loaded_figure.source_data()[loaded_figure.tool_status.primary_source]
        assert source.dims == data.dims
        assert source.chunks is not None
        np.testing.assert_array_equal(source.values, data.values)
        assert source_node.pending_workspace_memory_payload is not None


def test_manager_workspace_delta_snapshot_rewrites_stale_figure_source_references(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        name="first",
    )
    second = xr.DataArray(
        np.arange(9.0, 18.0).reshape(3, 3),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        name="second",
    )

    with manager_context() as manager:
        for data in (first, second):
            tool = erlab.interactive.itool(data, manager=False, execute=False)
            assert isinstance(tool, erlab.interactive.imagetool.ImageTool)
            manager.add_imagetool(tool, show=False)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = typing.cast(
            "FigureComposerTool", manager._child_node(figure_uid).tool_window
        )
        tree = manager._workspace_controller.saving._to_datatree()
        try:
            saved_ds = typing.cast(
                "xr.DataTree", tree[f"figures/{figure_uid}/tool"]
            ).to_dataset(inherit=False)
            saved_refs = json.loads(
                saved_ds.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR]
            )
            assert "second" in saved_refs
            assert saved_ds["second"].size == 0
        finally:
            tree.close()

        stale_uid = "missing-source-node"
        current_recipe = figure_tool.tool_status
        figure_tool.tool_status = current_recipe.model_copy(
            update={
                "sources": tuple(
                    source.model_copy(update={"node_uid": stale_uid})
                    if source.name == "second"
                    else source
                    for source in current_recipe.sources
                )
            }
        )
        manager._mark_workspace_layout_dirty()

        snapshot = manager._workspace_controller.saving._workspace_delta_save_snapshot(
            manager._workspace_state.dirty_generation,
            manager._workspace_controller.saving._workspace_root_attrs_payload(
                delta_save_count=1
            ),
            1,
        )
        try:
            rewrite_map = dict(snapshot.rewrite_groups)
            constructor = rewrite_map[f"figures/{figure_uid}"]
            rewritten_ds = constructor[f"figures/{figure_uid}/tool"]
            rewritten_refs = json.loads(
                rewritten_ds.attrs[erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR]
            )
            assert all(
                reference.get("node_uid") != stale_uid
                for reference in rewritten_refs.values()
            )
            assert "second" not in rewritten_refs
            assert rewritten_ds["second"].size > 0
            rewritten_state = FigureRecipeState.model_validate_json(
                rewritten_ds.attrs["tool_state"]
            )
            rewritten_source = next(
                source for source in rewritten_state.sources if source.name == "second"
            )
            assert rewritten_source.node_uid is None
            assert rewritten_source.node_snapshot_token is None
        finally:
            snapshot.close()


def test_manager_workspace_restores_figures_ui(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        itool(
            xr.DataArray(
                np.arange(4.0).reshape(2, 2),
                dims=("x", "y"),
                name="map",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        workspace = manager._workspace_controller.saving._to_datatree()
        try:
            manager.remove_all_tools()
            assert not manager.left_tabs.tabBar().isVisible()

            restored = manager._from_datatree(
                workspace, replace=True, mark_dirty=False, select=False
            )
            assert restored is True
            assert _figure_pane(manager).list_widget.count() == 1
            assert manager.left_tabs.tabBar().isVisible()
            assert manager.left_tabs.isTabVisible(1)
        finally:
            workspace.close()


def test_manager_workspace_close_serializes_all_figures(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uids: list[str] = []
        for _ in range(2):
            uid = manager.create_figure_from_targets((0,), show=False)
            assert uid is not None
            figure_uids.append(uid)

        workspace = manager._workspace_controller.saving._to_datatree(close=True)
        try:
            assert "figures" in workspace
            figures = typing.cast("xr.DataTree", workspace["figures"])
            assert all(uid in figures for uid in figure_uids)
            assert manager.ntools == 0
            assert manager._tool_graph.figure_uids == []
        finally:
            workspace.close()
