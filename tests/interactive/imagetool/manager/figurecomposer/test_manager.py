# ruff: noqa: F403, F405

from collections.abc import Iterable

import h5py
import pydantic

import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive.imagetool._figurecomposer_adapter import (
    build_figure_composer_operation,
)
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool._provenance._model import FileDataSelection
from erlab.interactive.imagetool.manager._figurecomposer import _controller, _dialogs

from ._common import *


class _SourcePickerDummyState(pydantic.BaseModel):
    value: int = 0


class _SourcePickerDummyTool(
    erlab.interactive.utils.ToolWindow[_SourcePickerDummyState]
):
    StateModel = _SourcePickerDummyState
    tool_name = "source-picker-dummy"

    def __init__(self, data: xr.DataArray) -> None:
        super().__init__()
        self._data = data
        self._status = _SourcePickerDummyState()

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    @property
    def tool_status(self) -> _SourcePickerDummyState:
        return self._status

    @tool_status.setter
    def tool_status(self, status: _SourcePickerDummyState) -> None:
        self._status = status


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
    pane = manager._figure_controller.pane
    assert pane is not None
    return pane


def _source_picker_item(
    dialog: QtWidgets.QDialog, uid: str
) -> QtWidgets.QTreeWidgetItem:
    tree = dialog.findChild(QtWidgets.QTreeWidget, "managerFigureSourcePickerTree")
    assert tree is not None
    root = tree.invisibleRootItem()
    stack = [root.child(index) for index in range(root.childCount())]
    while stack:
        item = stack.pop()
        assert item is not None
        if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == uid:
            return item
        stack.extend(item.child(index) for index in range(item.childCount()))
    raise AssertionError(f"missing picker row for {uid!r}")


def test_manager_figure_operation_source_name_mapping_updates_all_fields() -> None:
    method_x = FigureMethodPlotValueState(source="old", kind="data")
    method_y = FigureMethodPlotValueState(source="other", kind="data")
    operation = FigureOperationState.method(
        family=FigureMethodFamily.AXES,
        name="plot",
    ).model_copy(
        update={
            "sources": ("old", "other"),
            "map_selections": (
                FigureDataSelectionState(source="old", qsel={"x": 0.0}),
            ),
            "line_source": "old",
            "hv_overlay_source": "old",
            "method_plot_x": method_x,
            "method_plot_y": method_y,
        }
    )

    assert (
        manager_mainwindow.ImageToolManager._figure_operation_with_source_names(
            operation, {}
        )
        is operation
    )
    renamed = manager_mainwindow.ImageToolManager._figure_operation_with_source_names(
        operation, {"old": "new"}
    )

    assert renamed.sources == ("new", "other")
    assert renamed.map_selections == (
        FigureDataSelectionState(source="new", qsel={"x": 0.0}),
    )
    assert renamed.line_source == "new"
    assert renamed.hv_overlay_source == "new"
    assert renamed.method_plot_x == method_x.model_copy(update={"source": "new"})
    assert renamed.method_plot_y is method_y


def test_manager_default_figure_seed_uses_public_nonuniform_dims(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    public = xr.DataArray(
        np.arange(24.0).reshape(4, 2, 3),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0, 60.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1],
        },
        name="map",
    )
    internal = erlab.utils.array._make_dims_uniform(public)

    with manager_context() as manager:
        operation = manager._make_figure_operations_for_sources(
            {"data": internal},
            setup=FigureSubplotsState(),
        )[0]

    assert operation.kind == FigureOperationKind.PLOT_SLICES
    assert operation.slice_dim == "sample_temp"
    assert "sample_temp_idx" not in operation.model_dump_json()


def test_manager_default_figure_seed_keeps_mixed_higher_dimensional_sources(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    profile = xr.DataArray(np.arange(4.0), dims=("eV",), name="profile")
    image_stack = xr.DataArray(
        np.arange(24.0).reshape(3, 2, 4),
        dims=("sample_temp", "alpha", "eV"),
        coords={
            "sample_temp": [10.0, 15.0, 30.0],
            "alpha": [0.0, 1.0],
            "eV": [-0.1, 0.0, 0.1, 0.2],
        },
        name="map",
    )

    with manager_context() as manager:
        line_operation, map_operation = manager._make_figure_operations_for_sources(
            {"profile": profile, "map": image_stack},
            setup=FigureSubplotsState(nrows=2, ncols=1),
        )

    assert line_operation.kind == FigureOperationKind.LINE
    assert line_operation.line_source == "profile"
    assert line_operation.axes.axes == ((0, 0),)
    assert map_operation.kind == FigureOperationKind.PLOT_SLICES
    assert map_operation.sources == ("map",)
    assert map_operation.axes.axes == ((1, 0),)
    assert map_operation.slice_dim == "sample_temp"
    assert map_operation.slice_values == (15.0,)


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

        manager._figure_controller.select_uid(figure_uid)

        assert manager._selected_tool_uids() == [figure_uid]

        select_tools(manager, [0])

        root_uid = manager._tool_graph.root_wrappers[0].uid
        assert manager._selected_imagetool_targets() == [0]
        assert manager._selected_tool_uids() == []
        assert manager._selected_figure_uids() == []
        assert manager._metadata_node_uid == root_uid


def test_imagetool_plot_with_matplotlib_warns_for_uneditable_selection(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(_unsupported_plot_slices_data(), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)
        _set_unsupported_plot_slices_cursor_state(tool)

        warnings: list[tuple[QtWidgets.QWidget | None, str, str]] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda parent, title, text: warnings.append((parent, title, text)),
        )

        tool.slicer_area.images[0].plot_with_matplotlib()

        assert warnings
        assert len(manager._tool_graph.figure_uids) == 0


def test_manager_figure_action_warns_for_uneditable_plot_slices_selection(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(_unsupported_plot_slices_data(), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)
        _set_unsupported_plot_slices_cursor_state(tool)
        select_tools(manager, [0])

        warnings: list[tuple[QtWidgets.QWidget | None, str, str]] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda parent, title, text: warnings.append((parent, title, text)),
        )

        manager.create_figure_action.trigger()

        assert warnings
        assert len(manager._tool_graph.figure_uids) == 0


def test_manager_append_figure_warns_for_uneditable_plot_slices_selection(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(_unsupported_plot_slices_data(), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        source_tool = manager.get_imagetool(0)
        _set_unsupported_plot_slices_cursor_state(source_tool)

        figure_data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="existing",
        )
        figure_tool = FigureComposerTool(
            figure_data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="existing", label="existing"),),
                operations=(),
                primary_source="existing",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        warnings: list[tuple[QtWidgets.QWidget | None, str, str]] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda parent, title, text: warnings.append((parent, title, text)),
        )

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )

        assert appended is False
        assert warnings
        assert figure_tool.tool_status.operations == ()


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
        manager._figure_controller._show_menu(QtCore.QPoint())
        menu = manager._figure_controller._menu
        assert isinstance(menu, QtWidgets.QMenu)
        qtbot.wait_until(menu.isVisible, timeout=5000)

        manager._remove_childtool(figure_uid)
        qtbot.wait_until(lambda: manager._figure_controller._menu is None, timeout=5000)

        assert manager.left_tabs.count() == 1
        assert not hasattr(manager, "figure_tab")
        assert manager.left_tabs.minimumSizeHint().width() == empty_width

        manager._figure_controller.clear_selection_from_tree()
        manager._figure_controller._selection_changed()
        manager._figure_controller._show_item(QtWidgets.QListWidgetItem("removed"))
        manager._figure_controller._show_menu(QtCore.QPoint())


def test_manager_figure_menu_helpers_release_stale_wrappers(
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager._figure_controller.close_menu()
        assert manager._figure_controller._menu is None

        menu = QtWidgets.QMenu(manager)
        manager._figure_controller._menu = menu
        manager._figure_controller.close_menu()
        assert manager._figure_controller._menu is None

        stale_menu = QtWidgets.QMenu(manager)
        manager._figure_controller._menu = stale_menu
        original_qt_is_valid = erlab.interactive.utils.qt_is_valid

        def fake_qt_is_valid(*objects: object) -> bool:
            if any(obj is stale_menu for obj in objects):
                return False
            return original_qt_is_valid(*objects)

        with monkeypatch.context() as patch:
            patch.setattr(erlab.interactive.utils, "qt_is_valid", fake_qt_is_valid)
            manager._figure_controller.close_menu()
            assert manager._figure_controller._menu is None
            manager._figure_controller._release_menu(stale_menu)


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

        manager._figure_controller._show_menu(QtCore.QPoint())

        assert manager._figure_controller._menu is None


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
        manager._figure_controller.select_uid(figure_uid)

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
        manager._figure_controller.select_uid(first_uid)

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
        manager._figure_controller._show_item(_figure_pane(manager).list_widget.item(0))
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
    monkeypatch.setattr(tool, "_canvas_preview_pixmap", fail_render)
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
        <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.width()
    )
    assert (
        restored_preview.height()
        <= figurecomposer_tool_module._PERSISTED_PREVIEW_CACHE_SIZE.height()
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


def _materialized_figure_tool(
    manager: erlab.interactive.imagetool.manager.ImageToolManager, figure_uid: str
) -> FigureComposerTool:
    node = manager._child_node(figure_uid)
    assert node.materialize_pending_workspace_payload()
    tool = node.tool_window
    assert isinstance(tool, FigureComposerTool)
    return tool


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
    monkeypatch.setattr(tool, "_canvas_preview_pixmap", fail_render)
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

        manager._save_workspace_document(workspace_path, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        assert manager._load_workspace_file(
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
        item = manager._figure_controller.item_for_uid(figure_uid)
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

    monkeypatch.setattr(_controller, "_manager_settings", lambda: FakeSettings())
    with manager_context() as manager:
        figure_tool = FigureComposerTool(data)
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        assert _figure_pane(manager).gallery_button.isChecked()
        assert _figure_pane(manager).gallery_size_combo.currentData() == "medium"
        _figure_pane(manager).gallery_button.click()

        assert not manager._figure_controller._gallery_icon("missing").isNull()

        high_dpi_preview = QtGui.QPixmap(400, 100)
        high_dpi_preview.setDevicePixelRatio(2.0)
        high_dpi_preview.fill(QtGui.QColor("red"))
        high_dpi_thumbnail = manager._figure_controller._thumbnail_pixmap(
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
        assert not manager._figure_controller._gallery_icon(figure_uid).isNull()
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
        item = manager._figure_controller.item_for_uid(figure_uid)
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
        item = manager._figure_controller.item_for_uid(figure_uid)
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
            patch.setattr(manager._figure_controller, "sync", fail_sync)
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

        manager._figure_controller.select_uid(figure_uid)

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
        manager._figure_controller.select_uid(figure_uid)

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


def test_manager_figure_action_new_target_creates_second_figure(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0).reshape(2, 2),
                dims=("x", "y"),
                name="map",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        first_uid = manager.create_figure_from_targets((0,), show=False)
        assert first_uid is not None
        select_tools(manager, [0])

        class FakeFigureDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (first_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_NEW

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeFigureDialog)

        manager.create_figure_action.trigger()

        assert len(manager._tool_graph.figure_uids) == 2
        assert first_uid in manager._tool_graph.figure_uids


def test_manager_figure_action_appends_to_selected_subplots_axes(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0),
            dims=("x",),
            coords={"x": np.arange(4.0)},
            name="line",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        select_tools(manager, [0])

        class FakeFigureDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_ADD_STEP

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes=((0, 1),))

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeFigureDialog)

        manager.create_figure_action.trigger()

        assert len(manager._tool_graph.figure_uids) == 1
        assert len(figure_tool.tool_status.operations) == 1
        assert figure_tool.tool_status.operations[0].axes.axes == ((0, 1),)


def test_manager_figure_action_appends_to_selected_gridspec_axes(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0),
            dims=("x",),
            coords={"x": np.arange(4.0)},
            name="line",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(
                    layout_mode="gridspec",
                    gridspec=FigureGridSpecLayoutState(
                        root=FigureGridSpecGridState(
                            grid_id="root",
                            nrows=1,
                            ncols=1,
                            axes=(
                                FigureGridSpecAxesState(
                                    axes_id="axis-a",
                                    span=FigureGridSpecSpanState(
                                        row_start=0,
                                        row_stop=1,
                                        col_start=0,
                                        col_stop=1,
                                    ),
                                ),
                            ),
                        )
                    ),
                ),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        select_tools(manager, [0])

        class FakeFigureDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_ADD_STEP

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes_ids=("axis-a",))

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeFigureDialog)

        manager.create_figure_action.trigger()

        assert len(figure_tool.tool_status.operations) == 1
        assert figure_tool.tool_status.operations[0].axes.axes_ids == ("axis-a",)


def test_manager_auto_names_figures_numerically(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="map",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        manager._tool_graph.root_wrappers[0].slicer_area.axes[0].plot_with_matplotlib()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 1, timeout=5000
        )
        first_uid = manager._tool_graph.figure_uids[0]
        assert manager._child_node(first_uid).display_text == "Figure 1"

        second_uid = manager.create_figure_from_targets((0,), show=False)
        assert second_uid is not None
        assert manager._child_node(second_uid).display_text == "Figure 2"

        manager._child_node(first_uid).name = "Published figure"
        assert manager._child_node(first_uid).display_text == "Published figure"

        third_uid = manager.create_figure_from_targets((0,), show=False)
        assert third_uid is not None
        assert manager._child_node(third_uid).display_text == "Figure 3"

        manager._remove_childtool(second_uid)
        fourth_uid = manager.create_figure_from_targets((0,), show=False)
        assert fourth_uid is not None
        assert manager._child_node(fourth_uid).display_text == "Figure 4"

        preserved_tool = FigureComposerTool(data)
        preserved_tool._tool_display_name = "ImageTool plot"
        preserved_uid = manager.add_figuretool(preserved_tool, show=False)
        assert manager._child_node(preserved_uid).display_text == "ImageTool plot"

        explicit_uid = manager.create_figure_from_targets(
            (0,), title="Custom figure", show=False
        )
        assert explicit_uid is not None
        assert manager._child_node(explicit_uid).display_text == "Custom figure"

        fifth_uid = manager.create_figure_from_targets((0,), show=False)
        assert fifth_uid is not None
        assert manager._child_node(fifth_uid).display_text == "Figure 5"

        unnamed_tool = FigureComposerTool(data)
        unnamed_uid = manager.add_figuretool(unnamed_tool, show=False)
        assert manager._child_node(unnamed_uid).display_text == "Figure 6"


def test_manager_duplicate_figure_assigns_unique_display_name_and_keeps_state(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    def uid_for_name(
        manager: erlab.interactive.imagetool.manager.ImageToolManager, name: str
    ) -> str:
        matches = [
            uid
            for uid in manager._tool_graph.figure_uids
            if manager._child_node(uid).display_text == name
        ]
        assert len(matches) == 1
        return matches[0]

    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="map",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        first_uid = manager.create_figure_from_targets((0,), show=False)
        assert first_uid is not None
        assert manager._child_node(first_uid).display_text == "Figure 1"

        original_tool = manager._child_node(first_uid).tool_window
        assert isinstance(original_tool, FigureComposerTool)
        original_status = original_tool.tool_status

        manager._figure_controller.select_uid(first_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 2, timeout=5000
        )

        auto_copy_uid = uid_for_name(manager, "Figure 2")
        assert manager._selected_figure_uids() == [auto_copy_uid]
        auto_copy_tool = manager._child_node(auto_copy_uid).tool_window
        assert isinstance(auto_copy_tool, FigureComposerTool)
        assert auto_copy_tool.tool_status == original_status
        assert auto_copy_tool.tool_data.identical(original_tool.tool_data)

        manager._child_node(first_uid).name = "Band map"

        manager._figure_controller.select_uid(first_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 3, timeout=5000
        )

        first_custom_copy_uid = uid_for_name(manager, "Band map copy")
        assert manager._selected_figure_uids() == [first_custom_copy_uid]

        manager._figure_controller.select_uid(first_uid)
        manager.duplicate_selected()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.figure_uids) == 4, timeout=5000
        )

        second_custom_copy_uid = uid_for_name(manager, "Band map copy 2")
        assert manager._selected_figure_uids() == [second_custom_copy_uid]


def test_manager_create_figure_uses_first_selected_main_image_state(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(20.0).reshape(2, 2, 5) - 4.0,
            dims=("eV", "kx", "ky"),
            coords={
                "eV": [0.0, 1.0],
                "kx": [0.0, 1.0],
                "ky": [-2.0, -1.0, 0.0, 1.0, 2.0],
            },
            name="first",
        )
        second = xr.DataArray(
            np.arange(20.0).reshape(2, 2, 5),
            dims=("eV", "kx", "ky"),
            coords={
                "eV": [0.0, 1.0],
                "kx": [0.0, 1.0],
                "ky": [-2.0, -1.0, 0.0, 1.0, 2.0],
            },
            name="second",
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(first, _in_manager=True),
            show=False,
        )
        manager.add_imagetool(
            erlab.interactive.imagetool.ImageTool(second, _in_manager=True),
            show=False,
        )

        first_tool = manager.get_imagetool(0)
        first_tool.slicer_area.set_value(axis=2, value=1.0, cursor=0)
        first_tool.slicer_area.set_bin(axis=2, value=3, cursor=0)
        first_tool.slicer_area.set_colormap(
            "magma",
            gamma=0.75,
            reverse=True,
            high_contrast=True,
            zero_centered=True,
            levels_locked=True,
            levels=(-2.0, 4.0),
        )
        second_tool = manager.get_imagetool(1)
        second_tool.slicer_area.set_colormap("viridis", gamma=0.25)
        vmin, vmax = first_tool.slicer_area.colormap_properties["levels"]
        expected_first = build_figure_composer_operation(
            first_tool.slicer_area.images[0], source_name="first"
        )
        expected_second = build_figure_composer_operation(
            second_tool.slicer_area.images[0], source_name="second"
        )

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = typing.cast(
            "FigureComposerTool", manager._child_node(figure_uid).tool_window
        )
        operation = figure_tool.tool_status.operations[0]
        assert operation.sources == ("first_selected", "second_selected")
        sources = {source.name: source for source in figure_tool.source_states()}
        assert sources["first_selected"].selection_source == "first"
        assert sources["first_selected"].qsel == expected_first.map_selections[0].qsel
        assert sources["second_selected"].selection_source == "second"
        assert sources["second_selected"].qsel == expected_second.map_selections[0].qsel
        assert operation.order == "F"
        assert figure_tool.tool_status.setup.nrows == 1
        assert figure_tool.tool_status.setup.ncols == 2
        assert operation.slice_dim is None
        assert operation.slice_values == ()
        assert operation.slice_width is None
        assert operation.slice_kwargs == {}
        assert operation.transpose == expected_first.transpose
        assert operation.xlim == expected_first.xlim
        assert operation.ylim == expected_first.ylim
        assert operation.crop == expected_first.crop
        assert operation.axis == expected_first.axis
        assert operation.cmap is None
        assert operation.same_limits is False
        assert operation.norm_name == "PowerNorm"
        assert operation.norm_gamma is None
        assert operation.vcenter is None
        assert operation.halfrange is None
        assert operation.panel_styles_enabled
        styles = {
            (style.map_index, style.slice_index): style
            for style in operation.panel_styles
        }
        assert set(styles) == {(0, 0), (1, 0)}
        assert styles[(0, 0)].cmap == "magma_r"
        assert styles[(0, 0)].norm_name == "CenteredInversePowerNorm"
        assert styles[(0, 0)].norm_gamma == pytest.approx(0.75)
        assert styles[(0, 0)].vcenter == pytest.approx(0.5 * (vmin + vmax))
        assert styles[(0, 0)].halfrange == pytest.approx(0.5 * (vmax - vmin))
        assert styles[(1, 0)].cmap == "viridis"
        assert styles[(1, 0)].norm_name is None
        assert styles[(1, 0)].norm_gamma == pytest.approx(0.25)


def test_manager_create_figure_from_2d_data_ignores_autorange_startup_limits(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(25.0).reshape(5, 5),
        dims=("eV", "alpha"),
        coords={
            "eV": np.linspace(10.0, 14.0, 5),
            "alpha": np.linspace(20.0, 24.0, 5),
        },
        name="map",
    )

    with manager_context() as manager:
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = typing.cast(
            "FigureComposerTool", manager._child_node(figure_uid).tool_window
        )
        operation = figure_tool.tool_status.operations[0]
        assert operation.xlim is None
        assert operation.ylim is None

        figure = plt.figure()
        try:
            figurecomposer_rendering._render_into_figure(
                figure_tool, figure, sync_visible=False
            )
            assert figure_tool._operation_render_errors == {}
            assert any(axis.images for axis in figure.axes)
        finally:
            plt.close(figure)


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

        tree = manager._to_datatree()
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
            manager._save_workspace_document(fname, force_full=True)
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
            assert manager._load_workspace_file(
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
        manager._save_workspace_document(workspace_path, force_full=True)
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

        assert manager._load_workspace_file(
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
        tree = manager._to_datatree()
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

        snapshot = manager._workspace_delta_save_snapshot(
            manager._workspace_state.dirty_generation,
            manager._workspace_root_attrs_payload(delta_save_count=1),
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


def test_manager_plot_slices_setup_honors_order_for_horizontal_seeding() -> None:
    operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data",),
        slice_dim="eV",
        slice_values=(0.0, 1.0, 2.0),
    )
    assert manager_mainwindow.ImageToolManager._figure_plot_slices_grid_shape(
        operation
    ) == (1, 3)

    multi_source_operation = FigureOperationState.plot_slices(
        label="plot_slices",
        sources=("data_0", "data_1", "data_2"),
        slice_dim="eV",
        slice_values=(0.0,),
    ).model_copy(update={"order": "F"})
    assert manager_mainwindow.ImageToolManager._figure_plot_slices_grid_shape(
        multi_source_operation
    ) == (1, 3)


def test_manager_figure_operation_helpers_cover_multi_image_edges() -> None:
    manager = typing.cast(
        "manager_mainwindow.ImageToolManager",
        manager_mainwindow.ImageToolManager.__new__(
            manager_mainwindow.ImageToolManager
        ),
    )
    first_image = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("y", "x"))
    second_image = xr.DataArray(np.arange(4.0, 8.0).reshape(2, 2), dims=("y", "x"))

    operations = (
        manager_mainwindow.ImageToolManager._make_figure_operations_for_sources(
            manager,
            {"first": first_image, "second": second_image},
            setup=FigureSubplotsState(nrows=2, ncols=1),
        )
    )

    assert [operation.kind for operation in operations] == [
        FigureOperationKind.PLOT_ARRAY,
        FigureOperationKind.PLOT_ARRAY,
    ]
    assert [operation.axes.axes for operation in operations] == [
        ((0, 0),),
        ((1, 0),),
    ]

    split_by_axes_id = (
        manager_mainwindow.ImageToolManager._figure_operations_with_append_axes(
            typing.cast("tuple[typing.Any, ...]", operations),
            FigureAxesSelectionState(axes_ids=("left", "right")),
        )
    )
    assert [operation.axes.axes_ids for operation in split_by_axes_id] == [
        ("left",),
        ("right",),
    ]


def test_manager_figure_image_target_helpers_cover_plot_slices_edges(
    monkeypatch,
) -> None:
    plot_slices = FigureOperationState.plot_slices(
        label="slice",
        sources=("old",),
        slice_dim="eV",
        slice_values=(0.0,),
    )
    plot_array = FigureOperationState.plot_array(label="array", source="old")

    class _FakePlot:
        is_image = True

        def __init__(self, operation: FigureOperationState) -> None:
            self.operation = operation

    nodes = {
        "slice_a": types.SimpleNamespace(
            imagetool=types.SimpleNamespace(
                slicer_area=types.SimpleNamespace(axes=(_FakePlot(plot_slices),))
            )
        ),
        "slice_b": types.SimpleNamespace(
            imagetool=types.SimpleNamespace(
                slicer_area=types.SimpleNamespace(axes=(_FakePlot(plot_slices),))
            )
        ),
        "array": types.SimpleNamespace(
            imagetool=types.SimpleNamespace(
                slicer_area=types.SimpleNamespace(axes=(_FakePlot(plot_array),))
            )
        ),
    }
    manager = typing.cast(
        "manager_mainwindow.ImageToolManager",
        types.SimpleNamespace(
            _node_for_target=lambda target: nodes[target],
            get_imagetool=lambda target: nodes[target].imagetool,
        ),
    )
    monkeypatch.setattr(
        "erlab.interactive.imagetool._figurecomposer_adapter.build_figure_composer_operation",
        lambda plot, *, source_name: plot.operation.model_copy(
            update={"sources": (source_name,)}
        ),
    )

    assert (
        manager_mainwindow.ImageToolManager._figure_operations_from_image_targets(
            manager,
            ("slice_a", "array"),
            ("first", "second"),
        )
        is None
    )
    combined = (
        manager_mainwindow.ImageToolManager._figure_operations_from_image_targets(
            manager,
            ("slice_a", "slice_b"),
            ("first", "second"),
        )
    )

    assert combined is not None
    assert len(combined) == 1
    assert combined[0].kind == FigureOperationKind.PLOT_SLICES
    assert combined[0].sources == ("first", "second")
    assert combined[0].order == "F"


def test_manager_append_to_gridspec_figure_uses_axes_ids(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        axis_a = FigureGridSpecAxesState(
            axes_id="axis-a",
            label="panel",
            span=FigureGridSpecSpanState(
                row_start=0,
                row_stop=1,
                col_start=0,
                col_stop=1,
            ),
        )
        axis_b = FigureGridSpecAxesState(
            axes_id="axis-b",
            label="panel",
            span=FigureGridSpecSpanState(
                row_start=0,
                row_stop=1,
                col_start=1,
                col_stop=2,
            ),
        )
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(
                    layout_mode="gridspec",
                    gridspec=FigureGridSpecLayoutState(
                        root=FigureGridSpecGridState(
                            grid_id="root",
                            nrows=1,
                            ncols=2,
                            axes=(axis_a, axis_b),
                        )
                    ),
                ),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        operation = FigureOperationState.line(
            label="line",
            source="line",
            axes=FigureAxesSelectionState(),
        )

        dialog = _dialogs._AppendFigureTargetDialog(manager, (figure_uid,), operation)

        assert dialog.selector_stack.currentWidget() is dialog.gridspec_axes_selector
        assert dialog.gridspec_axes_selector.axes_ids() == ("axis-a", "axis-b")
        assert dialog.axes_selection() == FigureAxesSelectionState(axes_ids=("axis-a",))

        dialog.gridspec_axes_selector.set_selected_axes_ids(("axis-b",), emit=True)

        assert dialog.selected_target() == (
            figure_uid,
            FigureAxesSelectionState(axes_ids=("axis-b",)),
        )


def test_manager_append_to_subplots_figure_uses_axes_selector(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        operation = FigureOperationState.plot_slices(
            label="plot_slices",
            sources=("line",),
            axes=FigureAxesSelectionState(),
        )

        dialog = _dialogs._AppendFigureTargetDialog(manager, (figure_uid,), operation)

        assert dialog.selector_stack.currentWidget() is dialog.axes_selector
        assert dialog.axes_selector.selected_axes() == ((0, 0), (0, 1))

        dialog.axes_selector.set_selected_axes(((0, 1),), emit=True)

        assert dialog.selected_target() == (
            figure_uid,
            FigureAxesSelectionState(axes=((0, 1),)),
        )

        dialog.axes_selector.resize(dialog.axes_selector.sizeHint())
        qtbot.mouseClick(
            dialog.axes_selector,
            QtCore.Qt.MouseButton.LeftButton,
            pos=dialog.axes_selector._add_pill_rect("row").center(),
        )
        assert figure_tool.tool_status.setup.nrows == 2
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 1),))

        dialog.axes_selector.resize(dialog.axes_selector.sizeHint())
        qtbot.mouseClick(
            dialog.axes_selector,
            QtCore.Qt.MouseButton.LeftButton,
            pos=dialog.axes_selector._add_pill_rect("column").center(),
        )
        assert figure_tool.tool_status.setup.ncols == 3
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 1),))


def test_manager_figure_target_dialog_defaults_to_add_step_without_selected_figure(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
        )

        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_STEP
        assert not dialog.selector_stack.isHidden()
        assert dialog.selected_target() == (
            figure_uid,
            FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        )
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert button.isEnabled()

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_NEW)
        )

        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_NEW
        assert dialog.selector_stack.isHidden()
        assert dialog.selected_target() is None
        assert button.isEnabled()


def test_manager_figure_target_dialog_defaults_to_replace_selected_single_source(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="Line Source"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
            source_count=1,
            selected_figure_uid=figure_uid,
        )

        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE
        assert dialog.selected_source_alias() == "line"
        assert dialog.source_combo.currentData() == "line"
        assert dialog.selector_stack.isHidden()
        assert not dialog.source_combo.isHidden()
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert button.isEnabled()
        assert dialog._figure_source_count(None) == 0
        assert dialog._figure_source_count("missing") == 0

        class EmptyFigureNode:
            uid = "empty-figure"
            tool_window = None

        with monkeypatch.context() as context:
            context.setattr(manager, "_child_node", lambda _uid: EmptyFigureNode())
            assert dialog._figure_source_count(figure_uid) == 0


def test_manager_figure_target_dialog_switches_and_repairs_axes_selection(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        first_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        second_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        first_uid = manager.add_figuretool(first_tool, show=False)
        second_uid = manager.add_figuretool(second_tool, show=False)

        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (first_uid, second_uid),
            FigureOperationState.line(label="line", source="line"),
            allow_new_figure=True,
        )

        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_STEP
        assert not dialog.selector_stack.isHidden()
        ok_button = dialog.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        assert ok_button is not None
        assert ok_button.isEnabled()

        dialog.figure_combo.setCurrentIndex(dialog.figure_combo.findData(first_uid))
        assert dialog.figure_uid() == first_uid
        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_STEP
        assert dialog.selector_stack.currentWidget() is dialog.axes_selector
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 0),))

        dialog._select_all_axes()
        assert dialog.axes_selection() == FigureAxesSelectionState(
            axes=((0, 0), (0, 1))
        )
        dialog._clear_axes()
        assert dialog.axes_selection() is None
        assert not ok_button.isEnabled()
        dialog._select_all_axes()
        assert ok_button.isEnabled()
        assert dialog.selected_target() == (
            first_uid,
            FigureAxesSelectionState(axes=((0, 0), (0, 1))),
        )

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_ADD_SOURCE)
        )
        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_ADD_SOURCE
        assert dialog.selected_target() is None
        assert dialog.selector_stack.isHidden()
        assert ok_button.isEnabled()

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_REPLACE_SOURCE)
        )
        assert dialog.selected_action() == _dialogs._FIGURE_DIALOG_REPLACE_SOURCE
        assert dialog.selected_source_alias() == "line"
        assert dialog.selector_stack.isHidden()
        assert not dialog.source_combo.isHidden()
        assert ok_button.isEnabled()
        dialog.source_combo.setCurrentIndex(-1)
        dialog._selection_changed()
        assert not ok_button.isEnabled()
        dialog.source_combo.setCurrentIndex(0)
        dialog._selection_changed()
        assert ok_button.isEnabled()

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_ADD_STEP)
        )
        dialog.figure_combo.setCurrentIndex(dialog.figure_combo.findData(second_uid))
        assert dialog.axes_selection() == FigureAxesSelectionState(axes=((0, 0),))

        dialog.figure_combo.setItemData(dialog.figure_combo.currentIndex(), "missing")
        dialog._figure_changed()
        assert dialog.axes_selection() is None
        assert not ok_button.isEnabled()
        dialog._select_all_axes()
        dialog._grow_subplot_grid("row")

        dialog.figure_combo.setItemData(dialog.figure_combo.currentIndex(), None)
        assert dialog.figure_uid() == first_uid


def test_manager_figure_target_dialog_disables_replace_for_multiple_sources(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        figure_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        dialog = _dialogs._AppendFigureTargetDialog(
            manager,
            (figure_uid,),
            None,
            allow_new_figure=True,
            source_count=2,
            selected_figure_uid=figure_uid,
        )

        dialog.action_combo.setCurrentIndex(
            dialog.action_combo.findData(_dialogs._FIGURE_DIALOG_REPLACE_SOURCE)
        )
        button = dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        assert button is not None
        assert not button.isEnabled()
        assert not dialog.status_label.isHidden()


def test_manager_prompt_append_figure_target_auto_and_cancel_paths(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(np.arange(4.0), dims=("x",), name="line")
        assert manager._prompt_append_figure_target(None) is None
        assert manager._prompt_append_figure_target(None, figure_uid="missing") is None

        single_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        single_uid = manager.add_figuretool(single_tool, show=False)
        assert manager._append_single_axis_selection(single_uid) == (
            FigureAxesSelectionState(axes=((0, 0),))
        )
        assert manager._prompt_append_figure_target(None) == (
            single_uid,
            FigureAxesSelectionState(axes=((0, 0),)),
        )

        wide_tool = FigureComposerTool(
            data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        wide_uid = manager.add_figuretool(wide_tool, show=False)

        class RejectDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
            ) -> None:
                assert figure_uids == (wide_uid,)
                assert allow_new_figure is False

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Rejected

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", RejectDialog)
        assert manager._prompt_append_figure_target(None, figure_uid=wide_uid) is None


def test_manager_child_imagetool_gets_figure_context_actions(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    def action_names(tool: erlab.interactive.imagetool.ImageTool) -> set[str]:
        names: set[str] = set()
        for plot in tool.slicer_area.axes:
            menu = plot.vb.getMenu(None)
            assert menu is not None
            names.update(action.objectName() for action in menu.actions())
        return names

    with manager_context() as manager:
        itool(
            xr.DataArray(
                np.arange(4.0).reshape(2, 2),
                dims=("x", "y"),
                name="map",
            ),
            manager=True,
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        child = itool(
            xr.DataArray(
                np.arange(4.0),
                dims=("x",),
                coords={"x": np.arange(4.0)},
                name="line",
            ),
            manager=False,
            execute=False,
        )
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        assert all(plot.vb.menu is None for plot in child.slicer_area.axes)
        assert all(
            plot._plot_with_matplotlib_action is None for plot in child.slicer_area.axes
        )

        manager.add_imagetool_child(child, 0, show=False)

        assert all(plot.vb.menu is None for plot in child.slicer_area.axes)
        assert "itool_plot_with_matplotlib_action" in action_names(child)
        assert "itool_append_to_figure_action" in action_names(child)
        main_plot = child.slicer_area.axes[0]
        main_plot.vb.setMenuEnabled(False)
        assert main_plot.vb.menu is None
        main_plot.vb.setMenuEnabled(True)
        rebuilt_menu = main_plot.vb.getMenu(None)
        assert rebuilt_menu is not None
        rebuilt_action_names = {
            action.objectName() for action in rebuilt_menu.actions()
        }
        assert "itool_plot_with_matplotlib_action" in rebuilt_action_names
        assert "itool_append_to_figure_action" in rebuilt_action_names


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
        workspace = manager._to_datatree()
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

        workspace = manager._to_datatree(close=True)
        try:
            assert "figures" in workspace
            figures = typing.cast("xr.DataTree", workspace["figures"])
            assert all(uid in figures for uid in figure_uids)
            assert manager.ntools == 0
            assert manager._tool_graph.figure_uids == []
        finally:
            workspace.close()


def test_manager_append_operation_to_existing_figure(
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
        figure = manager._child_node(figure_uid).tool_window
        assert isinstance(figure, FigureComposerTool)
        operation_count = len(figure.tool_status.operations)
        source_name = figure.tool_status.sources[0].name

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            operation=FigureOperationState.line(label="overlay", source=source_name),
            show=False,
        )

        assert appended is True
        assert len(figure.tool_status.operations) == operation_count + 1
        assert figure.tool_status.operations[-1].kind.value == "line"
        assert figure.tool_status.operations[-1].line_source == source_name


def test_manager_explicit_figure_operations_use_readable_source_aliases(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(8.0).reshape(2, 4),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": np.arange(4.0)},
            name="sample map",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        script_name = manager._script_input_name_for_node(manager._node_for_target(0))
        assert script_name != "sample_map"
        figure_uid = manager.create_figure_from_targets(
            (0,),
            operation=FigureOperationState.plot_array(label="plot", source=script_name),
            show=False,
        )

        assert figure_uid is not None
        figure = manager._child_node(figure_uid).tool_window
        assert isinstance(figure, FigureComposerTool)
        assert tuple(figure.source_data()) == ("sample_map",)
        assert figure.tool_status.operations[-1].sources == ("sample_map",)


def test_manager_custom_figure_code_uses_readable_source_aliases(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(8.0).reshape(2, 4),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": np.arange(4.0)},
            name="sample map",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        script_name = manager._script_input_name_for_node(manager._node_for_target(0))
        assert script_name != "sample_map"
        for argument in ("operation", "custom_code"):
            code = f"fig.__dict__['{argument}_total'] = float({script_name}.sum())"
            kwargs: dict[str, typing.Any]
            if argument == "operation":
                kwargs = {
                    "operation": FigureOperationState.custom(
                        label="summary",
                        code=code,
                        trusted=True,
                    )
                }
            else:
                kwargs = {"custom_code": code}

            figure_uid = manager.create_figure_from_targets((0,), show=False, **kwargs)

            assert figure_uid is not None
            figure = manager._child_node(figure_uid).tool_window
            assert isinstance(figure, FigureComposerTool)
            [custom_operation] = figure.tool_status.operations
            assert script_name not in custom_operation.code
            assert "sample_map" in custom_operation.code
            figurecomposer_rendering._render_into_figure(
                figure, figure.figure, sync_visible=False
            )
            assert figure._operation_render_errors == {}
            assert figure.figure.__dict__[f"{argument}_total"] == float(data.sum())

        ambiguous_code = f"{script_name} = {script_name}.mean()"
        for kwargs in (
            {
                "operation": FigureOperationState.custom(
                    label="ambiguous",
                    code=ambiguous_code,
                    trusted=True,
                )
            },
            {"custom_code": ambiguous_code},
        ):
            with pytest.raises(
                FigureComposerInputError,
                match="also binds",
            ):
                manager.create_figure_from_targets((0,), show=False, **kwargs)


def test_manager_append_explicit_operation_uses_conflict_free_source_alias(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        for offset in (0.0, 10.0):
            itool(
                xr.DataArray(
                    np.arange(8.0).reshape(2, 4) + offset,
                    dims=("x", "y"),
                    coords={"x": [0.0, 1.0], "y": np.arange(4.0)},
                    name="sample map",
                ),
                manager=True,
            )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure = manager._child_node(figure_uid).tool_window
        assert isinstance(figure, FigureComposerTool)
        assert tuple(figure.source_data()) == ("sample_map",)

        script_name = manager._script_input_name_for_node(manager._node_for_target(1))
        assert script_name != "sample_map_2"
        appended = manager.append_figure_from_targets(
            (1,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            operation=FigureOperationState.plot_array(
                label="overlay", source=script_name
            ),
            show=False,
        )

        assert appended is True
        assert tuple(figure.source_data()) == ("sample_map", "sample_map_2")
        assert figure.tool_status.operations[-1].sources == ("sample_map_2",)


def test_manager_append_momentum_source_seeds_bz_overlay(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 0.0, 1.0], "ky": [-2.0, 0.0, 2.0]},
        name="momentum",
    )
    with manager_context() as manager:
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="existing",
        )
        figure_tool = FigureComposerTool(
            figure_data,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=1),
                sources=(FigureSourceState(name="existing", label="existing"),),
                operations=(),
                primary_source="existing",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )

        assert appended is True
        assert [operation.kind for operation in figure_tool.tool_status.operations] == [
            FigureOperationKind.PLOT_ARRAY,
            FigureOperationKind.BZ_OVERLAY,
        ]
        assert figure_tool.tool_status.operations[-1].axes.axes == ((0, 0),)


def test_manager_create_explicit_plot_slices_fills_axes(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="map",
    )
    with manager_context() as manager:
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        figure_uid = manager.create_figure_from_targets(
            (0,),
            operation=FigureOperationState.plot_slices(
                label="plot", sources=("map",), axes=FigureAxesSelectionState()
            ),
            show=False,
        )

        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        assert figure_tool.tool_status.operations[0].axes.axes == ((0, 0),)


def test_manager_ktool_output_figure_seeds_bz_overlay(
    qtbot,
    anglemap,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    from erlab.interactive.kspace import KspaceTool

    with manager_context() as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool(anglemap.qsel(eV=-0.1), link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.open_in_ktool()
        qtbot.wait_until(
            lambda: len(manager._tool_graph.root_wrappers[0]._childtools) == 1,
            timeout=5000,
        )

        child_uid = manager._tool_graph.root_wrappers[0]._childtool_indices[0]
        child = manager.get_childtool(child_uid)
        assert isinstance(child, KspaceTool)
        child._avec = erlab.lattice.abc2avec(2.0, 3.0, 4.0, 90.0, 100.0, 110.0)
        child.centering_combo.setCurrentText("I")
        child.rot_spin.setValue(15.0)
        child.kz_spin.setValue(0.5)
        child.points_check.setChecked(True)
        qtbot.wait_until(lambda: child.bz_group.isEnabled(), timeout=5000)
        child.bz_group.setChecked(True)
        child.show_converted()

        child_node = manager._child_node(child_uid)
        qtbot.wait_until(lambda: len(child_node._childtool_indices) == 1, timeout=5000)
        output_uid = child_node._childtool_indices[0]
        manager._child_node(output_uid).name = "converted"

        figure_uid = manager.create_figure_from_targets((output_uid,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operations = figure_tool.tool_status.operations

        assert [operation.kind for operation in operations] == [
            FigureOperationKind.PLOT_ARRAY,
            FigureOperationKind.BZ_OVERLAY,
        ]
        bz_operation = operations[1]
        assert np.isclose(bz_operation.bz_a, 2.0)
        assert np.isclose(bz_operation.bz_b, 3.0)
        assert np.isclose(bz_operation.bz_c, 4.0)
        assert bz_operation.bz_centering_type == "I"
        assert bz_operation.bz_angle == 15.0
        assert bz_operation.bz_kz_pi_over_c == 0.5
        assert bz_operation.bz_vertices is True
        assert bz_operation.bz_midpoints is True


def test_manager_bz_overlay_ignores_converted_output_without_ktool_parent() -> None:
    class FakeNode:
        output_id = "ktool.converted_output"

    class FakeParent:
        tool_window = object()

    class FakeManager:
        def _node_for_target(self, target: int | str) -> FakeNode:
            assert target == "converted"
            return FakeNode()

        def _parent_node(self, node: FakeNode) -> FakeParent:
            assert isinstance(node, FakeNode)
            return FakeParent()

    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("kx", "ky"),
        coords={"kx": [-1.0, 1.0], "ky": [-2.0, 2.0]},
        name="momentum",
    )
    axes = FigureAxesSelectionState(axes=((0, 0),))

    assert (
        manager_mainwindow.ImageToolManager._figure_bz_overlay_operation_from_targets(
            FakeManager(),
            ("first", "second"),
            {"momentum": data},
            axes=axes,
        )
        is None
    )
    assert (
        manager_mainwindow.ImageToolManager._figure_bz_overlay_operation_from_targets(
            FakeManager(),
            ("converted",),
            {"first": data, "second": data},
            axes=axes,
        )
        is None
    )

    assert (
        manager_mainwindow.ImageToolManager._figure_bz_overlay_operation_from_target(
            FakeManager(),
            "converted",
            data,
            axes=axes,
        )
        is None
    )


def test_manager_append_operation_uses_axes_dialog_selection(
    qtbot,
    monkeypatch,
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

        figure_tool = FigureComposerTool(
            xr.DataArray(np.arange(4.0), dims=("x",), name="line"),
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="line", label="line"),),
                operations=(),
                primary_source="line",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)

        class FakeAppendDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState,
            ) -> None:
                assert figure_uids == (figure_uid,)

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes=((0, 1),))

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeAppendDialog)

        appended = manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            operation=FigureOperationState.line(label="overlay", source="line"),
            show=False,
        )

        assert appended is True
        assert figure_tool.tool_status.operations[-1].axes.axes == ((0, 1),)


def test_manager_figure_action_replace_source_keeps_recipe_steps(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operation_count = len(figure_tool.tool_status.operations)

        select_tools(manager, [1])

        class FakeReplaceDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_REPLACE_SOURCE

            def figure_uid(self) -> str:
                return figure_uid

            def selected_source_alias(self) -> str:
                return "first"

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeReplaceDialog)

        manager.create_figure_action.trigger()

        assert len(figure_tool.tool_status.operations) == operation_count
        xr.testing.assert_identical(figure_tool.source_data()["first"], second)
        [source] = figure_tool.source_states()
        assert source.name == "first"
        assert source.node_uid == manager._node_for_target(1).uid
        assert "second" not in figure_tool.source_data()


def test_manager_figure_sources_reveal_associated_imagetool_rows(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2), dims=("x", "y"), name="first"
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2), dims=("x", "y"), name="second"
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        assert not manager._reveal_figure_sources("missing", ("first",))

        def raise_reveal_error(_source_name: str) -> bool:
            raise RuntimeError("source is unavailable")

        original_reveal_available = figure_tool._source_reveal_available_callback
        figure_tool._source_reveal_available_callback = raise_reveal_error
        assert not figure_tool._source_reveal_available("first")
        figure_tool._source_reveal_available_callback = original_reveal_available

        figure_tool.source_panel.set_selected_names(("first",), current_name="first")
        figure_tool.source_panel.duplicate_requested.emit(("first",))
        source_names = {source.name for source in figure_tool.source_states()}
        figure_tool.source_panel.set_selected_names(
            tuple(source_names), current_name="first"
        )
        figure_tool._refresh_source_controls()
        assert figure_tool.source_panel.reveal_sources_button.isEnabled()
        assert not figure_tool.source_panel.reveal_sources_button.autoRaise()
        assert (
            figure_tool.source_panel.reveal_sources_button.toolButtonStyle()
            == QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )

        figure_tool.source_panel.reveal_sources_button.click()

        assert manager.tree_view.selected_imagetool_indices == [0, 1]
        assert manager.left_tabs.currentWidget() is manager.tree_view

        manager.tree_view.clearSelection()
        figure_tool.source_panel.source_list.context_menu_requested.emit(
            QtCore.QPoint(0, 0)
        )
        reveal_action = figure_tool.source_panel.source_list.findChild(
            QtGui.QAction, "figureComposerContextRevealSourceAction"
        )
        assert reveal_action is not None
        source_menu = reveal_action.parent()
        assert isinstance(source_menu, QtWidgets.QMenu)
        assert reveal_action.isEnabled()
        reveal_action.trigger()
        assert manager.tree_view.selected_imagetool_indices == [0, 1]
        source_menu.close()

        manager.remove_imagetool(1)
        figure_tool.source_panel.set_selected_names(("second",), current_name="second")
        figure_tool._refresh_source_controls()
        assert not figure_tool.source_panel.reveal_sources_button.isEnabled()
        figure_tool.source_panel.reveal_requested.emit(("second",))


def test_manager_figure_source_row_refresh_updates_only_selected_source(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        original_second = figure_tool.source_data()["second"]
        operations = figure_tool.tool_status.operations
        manager._mark_workspace_clean()

        updated_first = first.copy(data=np.asarray(first.data) + 100.0, deep=True)
        updated_first.name = "first updated"
        updated_second = second.copy(data=np.asarray(second.data) + 200.0, deep=True)
        updated_second.name = "second updated"
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_first, manager=True, replace=0)
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_second, manager=True, replace=1)

        figure_tool.source_panel.set_selected_names(("first",), current_name="first")
        figure_tool._refresh_source_controls()
        assert figure_tool.source_panel.refresh_sources_button.isEnabled()
        with qtbot.wait_signal(figure_tool.sigDataChanged, timeout=5000):
            figure_tool.source_panel.refresh_sources_button.click()

        source_data = figure_tool.source_data()
        xr.testing.assert_identical(source_data["first"], updated_first)
        xr.testing.assert_identical(source_data["second"], original_second)
        [source_0, source_1] = figure_tool.source_states()
        assert source_0.name == "first"
        assert source_0.node_uid == manager._node_for_target(0).uid
        assert source_1.name == "second"
        assert figure_tool.tool_status.operations == operations
        assert "first" in figure_tool.generated_code()

        _render_figure_composer_rgba(figure_tool)
        namespace = _exec_generated_code(
            figure_tool.generated_code(),
            {"first": updated_first, "second": original_second},
        )
        assert isinstance(namespace["fig"], Figure)
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        manager.remove_imagetool(0)
        figure_tool._refresh_source_controls()
        assert not figure_tool.source_panel.refresh_sources_button.isEnabled()
        first_item = figure_tool.source_panel.source_list.topLevelItem(0)
        assert first_item is not None
        assert first_item.icon(0).isNull()


def test_manager_figure_refresh_sources_updates_live_sources_and_skips_detached(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        detached = xr.DataArray(
            np.array([1.0, 2.0]),
            dims=("x",),
            coords={"x": [0.0, 1.0]},
            name="detached",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0, 1), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        figure_tool.add_sources(
            (FigureSourceState(name="detached", label="Detached"),),
            {"detached": detached},
        )
        operations = figure_tool.tool_status.operations
        manager._mark_workspace_clean()

        updated_first = first.copy(data=np.asarray(first.data) + 100.0, deep=True)
        updated_first.name = "first updated"
        updated_second = second.copy(data=np.asarray(second.data) + 200.0, deep=True)
        updated_second.name = "second updated"
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_first, manager=True, replace=0)
        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(updated_second, manager=True, replace=1)

        figure_tool.source_panel.set_selected_names(
            ("first", "second", "detached"), current_name="first"
        )
        figure_tool._refresh_source_controls()
        assert figure_tool.source_panel.refresh_sources_button.isEnabled()
        figure_tool.source_panel.refresh_sources_button.click()

        source_data = figure_tool.source_data()
        xr.testing.assert_identical(source_data["first"], updated_first)
        xr.testing.assert_identical(source_data["second"], updated_second)
        xr.testing.assert_identical(source_data["detached"], detached)
        assert figure_tool.tool_status.operations == operations
        assert not figure_tool.source_panel.source_status_label.isHidden()
        assert figure_tool._operation_render_errors == {}
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]


def test_manager_figure_sources_use_readable_unique_aliases(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    arrays = (
        xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"), name="sample_map"),
        xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            name="reference map",
        ),
        xr.DataArray(
            np.arange(8.0, 12.0).reshape(2, 2),
            dims=("x", "y"),
            name="sample_map",
        ),
        xr.DataArray(
            np.arange(12.0, 16.0).reshape(2, 2),
            dims=("x", "y"),
            name=" !!! ",
        ),
    )

    with manager_context() as manager:
        for data in arrays:
            itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == len(arrays), timeout=5000)

        figure_uid = manager.create_figure_from_targets(range(len(arrays)), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)

        source_names = tuple(source.name for source in figure_tool.source_states())
        assert source_names == (
            "sample_map",
            "reference_map",
            "sample_map_2",
            "data_3",
        )
        assert tuple(figure_tool.source_data()) == source_names
        reference_item = figure_tool.source_panel.source_list.topLevelItem(1)
        assert reference_item is not None
        assert (
            reference_item.data(0, QtCore.Qt.ItemDataRole.UserRole) == "reference_map"
        )
        assert "Original name: reference map" in reference_item.toolTip(0)


def test_manager_figure_source_only_append_uses_readable_conflict_suffix(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(np.arange(4.0).reshape(2, 2), dims=("x", "y"), name="map")
    second = xr.DataArray(
        np.arange(4.0, 8.0).reshape(2, 2), dims=("x", "y"), name="map"
    )

    with manager_context() as manager:
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)

        _resolved_targets, sources, source_data = manager._figure_sources_from_targets(
            (1,)
        )
        manager._add_sources_to_figure(figure_uid, sources, source_data, show=False)

        assert tuple(source.name for source in figure_tool.source_states()) == (
            "map",
            "map_2",
        )
        xr.testing.assert_identical(figure_tool.source_data()["map_2"], second)


def test_manager_figure_source_helper_edge_contracts(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="data",
        )
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        image_uid = manager._node_for_target(0).uid
        child = itool(data.sum("y"), manager=False, execute=False)
        assert isinstance(child, erlab.interactive.imagetool.ImageTool)
        child_uid = manager.add_imagetool_child(child, 0, show=False)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        select_child_tool(manager, figure_uid)
        assert manager._selected_figure_uid_for_figure_dialog() == figure_uid

        source_states = figure_tool.source_states()
        source_data = figure_tool.source_data()
        source_alias = source_states[0].name
        replacement_source = FigureSourceState(
            name="replacement",
            label="replacement",
            node_uid=image_uid,
        )
        assert not manager._add_sources_to_figure(
            "missing",
            source_states,
            source_data,
            show=False,
        )
        assert manager._figure_source_state(figure_tool, "missing") is None
        assert manager._figure_source_live_node("missing", source_alias) is None
        assert not manager._refresh_figure_source(figure_uid, "missing")
        assert not manager._replace_figure_source(
            figure_uid,
            source_alias,
            (),
            {},
            show=False,
        )
        assert not manager._replace_figure_source(
            figure_uid,
            source_alias,
            (replacement_source,),
            {},
            show=False,
        )

        with monkeypatch.context() as context:
            context.setattr(
                manager,
                "_is_figure_uid",
                lambda uid: uid in {figure_uid, child_uid},
            )
            assert not manager._add_sources_to_figure(
                child_uid,
                source_states,
                source_data,
                show=False,
            )
            assert not manager._replace_figure_source(
                child_uid,
                source_alias,
                (replacement_source,),
                {"replacement": data},
                show=False,
            )
            assert manager._figure_source_live_node(child_uid, source_alias) is None
            assert not manager._refresh_figure_source(child_uid, source_alias)

        with monkeypatch.context() as context:
            context.setattr(figure_tool, "replace_source", lambda *_args: False)
            assert not manager._replace_figure_source(
                figure_uid,
                source_alias,
                (replacement_source,),
                {"replacement": data},
                show=False,
            )
            assert not manager._refresh_figure_source(figure_uid, source_alias)

        with monkeypatch.context() as context:
            context.setattr(
                manager,
                "_figure_source_live_node",
                lambda *_args: manager._node_for_target(0),
            )
            context.setattr(manager, "_is_figure_uid", lambda uid: uid == child_uid)
            assert not manager._refresh_figure_source(child_uid, source_alias)

        with monkeypatch.context() as context:
            context.setattr(manager, "_figure_uids", lambda: ("missing",))
            manager._refresh_figure_source_controls()

        with monkeypatch.context() as context:
            context.setattr(manager, "_selected_figure_source_targets", lambda: (0,))
            context.setattr(
                manager,
                "_figure_sources_from_targets",
                lambda _targets: ((), (), {}),
            )
            manager.create_figure_from_selection()

        dialog_events: list[str] = []

        class AliasNoneDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                _figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid == figure_uid
                dialog_events.append("init")

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_REPLACE_SOURCE

            def selected_source_alias(self) -> str | None:
                dialog_events.append("alias")
                return None

        with monkeypatch.context() as context:
            context.setattr(manager, "_selected_figure_source_targets", lambda: (0,))
            context.setattr(manager, "_figure_uids", lambda: (figure_uid,))
            context.setattr(
                manager,
                "_figure_sources_from_targets",
                lambda _targets: ((0,), source_states, source_data),
            )
            context.setattr(
                manager,
                "_selected_figure_uid_for_figure_dialog",
                lambda: figure_uid,
            )
            context.setattr(
                _dialogs,
                "_AppendFigureTargetDialog",
                AliasNoneDialog,
            )
            manager.create_figure_from_selection()
        assert dialog_events == ["init", "alias"]


def test_manager_figure_action_add_source_only_keeps_recipe_steps(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operation_count = len(figure_tool.tool_status.operations)
        workspace_path = tmp_path / "add-source-only-delta.itws"
        manager._save_workspace_document(workspace_path, force_full=True)

        select_tools(manager, [1])

        class FakeSourceOnlyDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 1
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_ADD_SOURCE

            def figure_uid(self) -> str:
                return figure_uid

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeSourceOnlyDialog)

        manager.create_figure_action.trigger()

        assert len(figure_tool.tool_status.operations) == operation_count
        xr.testing.assert_identical(figure_tool.source_data()["second"], second)
        assert {source.name for source in figure_tool.source_states()} == {
            "first",
            "second",
        }
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        manager._save_workspace_document(workspace_path)
        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded_node = manager._child_node(figure_uid)
        loaded_tool = loaded_node.tool_window
        if loaded_tool is None:
            assert loaded_node.pending_workspace_tool_payload is not None
            loaded_node.show()
            loaded_tool = loaded_node.tool_window
        assert isinstance(loaded_tool, FigureComposerTool)
        xr.testing.assert_identical(loaded_tool.source_data()["second"], second)


def test_manager_figure_source_picker_selects_imagetool_rows_only(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    root_data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="root",
    )
    child_data = root_data + 10.0
    with manager_context() as manager:
        itool(root_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        root_uid = manager._tool_graph.root_wrappers[0].uid
        dummy_uid = manager.add_childtool(
            _SourcePickerDummyTool(root_data.rename("dummy")),
            0,
            show=False,
        )
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(
                child_data.rename("child"), _in_manager=True
            ),
            dummy_uid,
            show=False,
        )

        dialog = _dialogs._FigureSourcePickerDialog(
            manager, prechecked_uids=(root_uid, child_uid)
        )
        qtbot.addWidget(dialog)
        root_item = _source_picker_item(dialog, root_uid)
        dummy_item = _source_picker_item(dialog, dummy_uid)
        child_item = _source_picker_item(dialog, child_uid)

        checkable = QtCore.Qt.ItemFlag.ItemIsUserCheckable
        assert root_item.flags() & checkable
        assert child_item.flags() & checkable
        assert not dummy_item.flags() & checkable
        assert root_item.checkState(0) == QtCore.Qt.CheckState.Checked
        assert child_item.checkState(0) == QtCore.Qt.CheckState.Checked
        assert dialog.selected_targets() == (root_uid, child_uid)

        assert dummy_item.isExpanded()
        dummy_item.setExpanded(False)
        dialog.search_edit.setText("child")
        assert not root_item.isHidden()
        assert not dummy_item.isHidden()
        assert not child_item.isHidden()
        assert dummy_item.isExpanded()
        assert dialog.selected_targets() == (root_uid, child_uid)
        dialog.search_edit.clear()
        assert not dummy_item.isExpanded()
        assert root_item.checkState(0) == QtCore.Qt.CheckState.Checked
        assert child_item.checkState(0) == QtCore.Qt.CheckState.Checked

        root_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        child_item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        ok_button = dialog.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        assert ok_button is not None
        assert not ok_button.isEnabled()

        assert manager.create_figure_from_targets((dummy_uid,), show=False) is None
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        assert not manager.append_figure_from_targets(
            (dummy_uid,), figure_uid=figure_uid, show=False
        )
        resolved_targets, sources, source_data = manager._figure_sources_from_targets(
            (dummy_uid, child_uid)
        )
        assert resolved_targets == (child_uid,)
        assert len(sources) == len(source_data) == 1


def test_figure_sources_add_button_adds_imagetool_sources(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="first",
    )
    second = first + 10.0
    with manager_context() as manager:
        itool(first, manager=True)
        itool(second.rename("second"), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        operation_count = len(figure_tool.tool_status.operations)
        second_uid = manager._tool_graph.root_wrappers[1].uid
        select_tools(manager, [1])
        manager._mark_workspace_clean()

        class RejectingPicker:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                *,
                prechecked_uids: Iterable[str] = (),
            ) -> None:
                assert tuple(prechecked_uids) == (second_uid,)

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Rejected

        monkeypatch.setattr(_dialogs, "_FigureSourcePickerDialog", RejectingPicker)
        figure_tool.source_panel.add_source_button.click()

        assert len(figure_tool.tool_status.operations) == operation_count
        assert {source.name for source in figure_tool.source_states()} == {"first"}
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid not in snapshot["dirty_data"]
        assert figure_uid not in snapshot["dirty_state"]

        class AcceptingPicker(RejectingPicker):
            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_targets(self) -> tuple[str, ...]:
                return (second_uid,)

        monkeypatch.setattr(_dialogs, "_FigureSourcePickerDialog", AcceptingPicker)
        with qtbot.wait_signal(figure_tool.sigDataChanged, timeout=5000):
            figure_tool.source_panel.add_source_button.click()

        assert len(figure_tool.tool_status.operations) == operation_count
        xr.testing.assert_identical(
            figure_tool.source_data()["second"], second.rename("second")
        )
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]


def test_manager_figure_source_add_reports_partial_rejection(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="first",
    )
    second = first.copy(data=np.arange(4.0, 8.0).reshape(2, 2)).rename("second")
    incompatible = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("z", "y"),
        coords={"z": [0.0, 1.0], "y": [0.0, 1.0]},
        name="replacement",
    )
    with manager_context() as manager:
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        figure_tool.source_panel.set_selected_names(("first",), current_name="first")
        figure_tool.source_panel.selection_dimension_requested.emit(
            ("first",), "x", "qsel", "0.0", ""
        )
        original_first = figure_tool.source_data()["first"]
        original_operations = figure_tool.tool_status.operations

        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(incompatible, manager=True, replace=0)
        manager._mark_workspace_clean()

        assert manager._add_imagetool_sources_to_figure(figure_uid, (0, 1), show=False)

        xr.testing.assert_identical(figure_tool.source_data()["first"], original_first)
        xr.testing.assert_identical(figure_tool.source_data()["second"], second)
        assert figure_tool.tool_status.operations == original_operations
        assert not figure_tool.source_panel.source_status_label.isHidden()
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        manager._mark_workspace_clean()
        assert manager.append_figure_from_targets(
            (0, 1),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )
        assert figure_tool.tool_status.operations == original_operations
        assert not figure_tool.source_panel.source_status_label.isHidden()
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]


def test_manager_rejected_source_update_does_not_mark_dirty_or_add_step(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    data = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="data",
    )
    incompatible = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("z", "y"),
        coords={"z": [0.0, 1.0], "y": [0.0, 1.0]},
        name="replacement",
    )
    with manager_context() as manager:
        itool(data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        figure_tool.source_panel.set_selected_names(("data",), current_name="data")
        figure_tool.source_panel.selection_dimension_requested.emit(
            ("data",), "x", "qsel", "0.0", ""
        )
        original_data = figure_tool.source_data()["data"]
        original_operations = figure_tool.tool_status.operations

        with qtbot.wait_signal(manager._sigDataReplaced, timeout=5000):
            itool(incompatible, manager=True, replace=0)
        manager._mark_workspace_clean()

        assert not manager._add_imagetool_sources_to_figure(
            figure_uid, (0,), show=False
        )
        xr.testing.assert_identical(figure_tool.source_data()["data"], original_data)
        assert not figure_tool.source_panel.source_status_label.isHidden()
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid not in snapshot["dirty_data"]
        assert figure_uid not in snapshot["dirty_state"]

        assert not manager.append_figure_from_targets(
            (0,),
            figure_uid=figure_uid,
            axes_selection=FigureAxesSelectionState(axes=((0, 0),)),
            show=False,
        )

        xr.testing.assert_identical(figure_tool.source_data()["data"], original_data)
        assert figure_tool.tool_status.operations == original_operations
        assert not figure_tool.source_panel.source_status_label.isHidden()
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid not in snapshot["dirty_data"]
        assert figure_uid not in snapshot["dirty_state"]


def test_figure_sources_drag_mime_adds_root_and_child_imagetools(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    first = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("x", "y"),
        coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
        name="first",
    )
    second = first + 10.0
    child_data = first + 20.0
    with manager_context() as manager:
        itool(first, manager=True)
        itool(second.rename("second"), manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        child_uid = manager.add_imagetool_child(
            erlab.interactive.imagetool.ImageTool(
                child_data.rename("child"), _in_manager=True
            ),
            0,
            show=False,
        )
        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())
        second_uid = manager._tool_graph.root_wrappers[1].uid
        second_mime = model.mimeData([model.index(1, 0)])

        assert manager.tree_view.figure_source_uids_from_mime(second_mime) == (
            second_uid,
        )
        assert figure_tool._source_drop_available(second_mime)
        assert figure_tool._add_sources_from_mime(second_mime)
        xr.testing.assert_identical(
            figure_tool.source_data()["second"], second.rename("second")
        )

        child_mime = model.mimeData([model._row_index(child_uid)])
        assert manager.tree_view.figure_source_uids_from_mime(child_mime) == (
            child_uid,
        )
        window = figure_tool.figure_window
        assert window._handle_source_drag_event(None) is False
        assert figure_tool._source_drop_available(child_mime)
        assert figure_tool._add_sources_from_mime(child_mime)
        child_source_name = next(
            source.name
            for source in figure_tool.source_states()
            if source.node_uid == child_uid
        )
        xr.testing.assert_identical(
            figure_tool.source_data()[child_source_name], child_data.rename("child")
        )

        assert not manager._add_imagetool_sources_to_figure(
            figure_uid, (figure_uid,), show=False
        )
        assert not manager._request_add_sources_to_figure("missing-figure")
        assert not manager._add_figure_sources_from_mime(figure_uid, QtCore.QMimeData())

        original_add_sources = manager._add_sources_to_figure
        monkeypatch.setattr(
            manager,
            "_add_sources_to_figure",
            lambda *_args, **_kwargs: False,
        )
        assert not manager._add_imagetool_sources_to_figure(
            figure_uid, (second_uid,), show=False
        )
        monkeypatch.setattr(manager, "_add_sources_to_figure", original_add_sources)

        source_names = tuple(figure_tool.source_data())
        assert manager._add_imagetool_sources_to_figure(
            figure_uid, (second_uid, figure_uid), show=False
        )
        assert tuple(figure_tool.source_data()) == source_names
        xr.testing.assert_identical(
            figure_tool.source_data()["second"], second.rename("second")
        )
        assert not figure_tool.source_panel.source_status_label.isHidden()


def test_manager_figure_remove_unused_source_persists_workspace(
    qtbot,
    tmp_path: Path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        figure_uid = manager.create_figure_from_targets((0,), show=False)
        assert figure_uid is not None
        figure_tool = manager._child_node(figure_uid).tool_window
        assert isinstance(figure_tool, FigureComposerTool)
        _resolved_targets, sources, source_data = manager._figure_sources_from_targets(
            (1,)
        )
        [source] = sources
        manager._add_sources_to_figure(figure_uid, sources, source_data, show=False)
        assert source.name in figure_tool.source_data()
        manager._mark_workspace_clean()

        figure_tool.source_panel.set_selected_names(
            (source.name,), current_name=source.name
        )
        figure_tool._refresh_source_controls()
        assert figure_tool.source_panel.remove_selected_source_button.isEnabled()
        with qtbot.wait_signal(figure_tool.sigDataChanged, timeout=5000):
            figure_tool.source_panel.remove_selected_source_button.click()

        assert source.name not in figure_tool.source_data()
        assert source.name not in {
            source.name for source in figure_tool.source_states()
        }
        snapshot = manager._workspace_state_snapshot()
        assert figure_uid in snapshot["dirty_data"]
        assert figure_uid in snapshot["dirty_state"]

        workspace_path = tmp_path / "remove-unused-figure-source.itws"
        manager._save_workspace_document(workspace_path, force_full=True)
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
        assert manager._load_workspace_file(
            workspace_path,
            replace=True,
            associate=False,
            mark_dirty=False,
            select=False,
        )
        loaded_node = manager._child_node(figure_uid)
        assert loaded_node.pending_workspace_tool_payload is not None
        loaded_tool = _materialized_figure_tool(manager, figure_uid)
        assert source.name not in loaded_tool.source_data()
        assert source.name not in {
            source.name for source in loaded_tool.source_states()
        }


def test_manager_figure_action_multi_source_append_preserves_image_colormaps(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        first = xr.DataArray(
            np.arange(4.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="first",
        )
        second = xr.DataArray(
            np.arange(4.0, 8.0).reshape(2, 2),
            dims=("x", "y"),
            coords={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            name="second",
        )
        itool(first, manager=True)
        itool(second, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.get_imagetool(0).slicer_area.set_colormap("magma")
        manager.get_imagetool(1).slicer_area.set_colormap("viridis", reverse=True)

        figure_tool = FigureComposerTool(
            first,
            recipe=FigureRecipeState(
                setup=FigureSubplotsState(nrows=1, ncols=2),
                sources=(FigureSourceState(name="seed", label="seed"),),
                operations=(),
                primary_source="seed",
            ),
        )
        figure_uid = manager.add_figuretool(figure_tool, show=False)
        select_tools(manager, [0, 1])

        class FakeAppendDialog:
            def __init__(
                self,
                _manager: erlab.interactive.imagetool.manager.ImageToolManager,
                figure_uids: tuple[str, ...],
                _operation: FigureOperationState | None,
                *,
                allow_new_figure: bool = False,
                source_count: int = 1,
                selected_figure_uid: str | None = None,
            ) -> None:
                assert figure_uids == (figure_uid,)
                assert allow_new_figure is True
                assert source_count == 2
                assert selected_figure_uid is None

            def exec(self) -> QtWidgets.QDialog.DialogCode:
                return QtWidgets.QDialog.DialogCode.Accepted

            def selected_action(self) -> str:
                return _dialogs._FIGURE_DIALOG_ADD_STEP

            def selected_target(self) -> tuple[str, FigureAxesSelectionState]:
                return figure_uid, FigureAxesSelectionState(axes=((0, 0), (0, 1)))

        monkeypatch.setattr(_dialogs, "_AppendFigureTargetDialog", FakeAppendDialog)

        manager.create_figure_action.trigger()

        appended = figure_tool.tool_status.operations[-2:]
        assert [operation.kind for operation in appended] == [
            FigureOperationKind.PLOT_ARRAY,
            FigureOperationKind.PLOT_ARRAY,
        ]
        assert [operation.sources for operation in appended] == [
            ("first",),
            ("second",),
        ]
        assert [operation.axes.axes for operation in appended] == [
            ((0, 0),),
            ((0, 1),),
        ]
        assert [operation.cmap for operation in appended] == ["magma", "viridis_r"]
        assert all(not operation.panel_styles_enabled for operation in appended)
