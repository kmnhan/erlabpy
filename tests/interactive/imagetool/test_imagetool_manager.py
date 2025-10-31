import concurrent.futures
import json
import logging
import tempfile
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
import xarray.testing
from IPython.core.interactiveshell import InteractiveShell
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.derivative import DerivativeTool
from erlab.interactive.fermiedge import GoldTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool.manager import ImageToolManager, fetch, load_in_manager
from erlab.interactive.imagetool.manager._dialogs import (
    _ConcatDialog,
    _NameFilterDialog,
    _RenameDialog,
)
from erlab.interactive.imagetool.manager._modelview import (
    _MIME,
    _ImageToolWrapperItemDelegate,
    _ImageToolWrapperItemModel,
)
from erlab.interactive.imagetool.manager._server import _remove_idx, _show_idx

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_data():
    return xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5), "eV": np.arange(5)},
    )


def select_tools(
    manager: ImageToolManager, indices: list[int], deselect: bool = False
) -> None:
    selection_model = manager.tree_view.selectionModel()

    for index in indices:
        qmodelindex = manager.tree_view._model._row_index(index)
        selection_model.select(
            QtCore.QItemSelection(qmodelindex, qmodelindex),
            QtCore.QItemSelectionModel.SelectionFlag.Deselect
            if deselect
            else QtCore.QItemSelectionModel.SelectionFlag.Select,
        )


def select_child_tool(
    manager: ImageToolManager, uid: str, deselect: bool = False
) -> None:
    selection_model = manager.tree_view.selectionModel()

    qmodelindex = manager.tree_view._model._row_index(uid)
    selection_model.select(
        QtCore.QItemSelection(qmodelindex, qmodelindex),
        QtCore.QItemSelectionModel.SelectionFlag.Deselect
        if deselect
        else QtCore.QItemSelectionModel.SelectionFlag.Select,
    )


def make_drop_event(filename: str) -> QtGui.QDropEvent:
    mime_data = QtCore.QMimeData()
    mime_data.setUrls([QtCore.QUrl.fromLocalFile(filename)])
    return QtGui.QDropEvent(
        QtCore.QPointF(0.0, 0.0),
        QtCore.Qt.DropAction.CopyAction,
        mime_data,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def bring_manager_to_top(bot, manager):
    with bot.waitExposed(manager):
        manager.hide_all()  # Prevent windows from obstructing the manager
        manager.activateWindow()
        manager.raise_()


@pytest.mark.parametrize("use_socket", [False, True], ids=["no_socket", "socket"])
def test_manager(
    qtbot,
    accept_dialog,
    test_data,
    use_socket,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context(use_socket=use_socket) as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        logger.info("Manager is running, adding test data")
        test_data.qshow(manager=True)

        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        logger.info("Confirmed tool is added, checking data")
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        logger.info("Checking data retrieval via fetch")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fetch, 0)

            # Let the GUI thread keep processing events while waiting
            qtbot.waitUntil(lambda: fut.done(), timeout=10000)

            result = fut.result()

        xr.testing.assert_identical(result, test_data)

        logger.info("Confirmed fetch works, adding more tools")
        # Add two tools
        for tool in itool(
            [test_data, test_data], link=False, execute=False, manager=False
        ):
            tool.move_to_manager()

        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Linking
        select_tools(manager, [1, 2])
        manager.link_selected()
        manager.tree_view.refresh(None)

        # Unlinking one unlinks both
        select_tools(manager, [1])
        manager.unlink_selected()
        manager.tree_view.refresh(None)
        assert not manager.get_imagetool(1).slicer_area.is_linked
        assert not manager.get_imagetool(2).slicer_area.is_linked

        # Linking again
        select_tools(manager, [1, 2])
        manager.link_selected()
        manager.tree_view.refresh(None)
        assert manager.get_imagetool(1).slicer_area.is_linked
        assert manager.get_imagetool(2).slicer_area.is_linked

        # Archiving and unarchiving
        manager._imagetool_wrappers[1].archive()
        manager._imagetool_wrappers[1].touch_archive()
        assert manager._imagetool_wrappers[1].archived
        manager._imagetool_wrappers[1].unarchive()
        assert not manager._imagetool_wrappers[1].archived

        # Toggle visibility
        geometry = manager.get_imagetool(1).geometry()
        manager._imagetool_wrappers[1].hide()
        assert not manager.get_imagetool(1).isVisible()
        manager._imagetool_wrappers[1].show()
        assert manager.get_imagetool(1).geometry() == geometry

        # Removing archived tool
        manager._imagetool_wrappers[0].archive()
        manager.remove_imagetool(0)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Batch renaming
        select_tools(manager, [1, 2])

        def _handle_renaming(dialog: _RenameDialog):
            dialog._new_name_lines[1].setText("new_name_1")
            dialog._new_name_lines[2].setText("new_name_2")

        accept_dialog(manager.rename_action.trigger, pre_call=_handle_renaming)
        assert manager._imagetool_wrappers[1].name == "new_name_1"
        assert manager._imagetool_wrappers[2].name == "new_name_2"

        # Rename single
        select_tools(manager, [2], deselect=True)
        select_tools(manager, [1])
        manager.rename_action.trigger()

        qtbot.wait_until(
            lambda: manager.tree_view.state()
            == QtWidgets.QAbstractItemView.State.EditingState,
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor(), QtWidgets.QLineEdit)
        delegate._current_editor().setText("new_name_1_single")
        qtbot.keyClick(delegate._current_editor(), QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: manager._imagetool_wrappers[1].name == "new_name_1_single",
            timeout=5000,
        )

        # Select single tool
        select_tools(manager, [1])

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager._update_info()

        # Archive & unarchive single
        manager.archive_action.trigger()
        qtbot.wait_until(lambda: manager._imagetool_wrappers[1].archived, timeout=5000)

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager._update_info()

        manager._imagetool_wrappers[1].unarchive()

        # Batch archiving with show/hide
        select_tools(manager, [1])
        manager.archive_action.trigger()

        select_tools(manager, [1, 2])

        # Hide non-archived window, does nothing to archived window
        manager.hide_action.trigger()
        # Unarchive the archived one and show both
        manager.show_action.trigger()

        assert not manager._imagetool_wrappers[1].archived
        assert not manager._imagetool_wrappers[2].archived
        assert manager.get_imagetool(1).isVisible()
        assert manager.get_imagetool(2).isVisible()

        # Add third tool
        xr.concat(
            [
                manager.get_imagetool(1).slicer_area._data,
                manager.get_imagetool(2).slicer_area._data,
            ],
            "concat_dim",
        ).qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Update info panel
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_tools(manager, [1, 2, 3])
        manager._update_info()

        # Show goldtool
        logger.info("Opening goldtool")
        manager.get_imagetool(3).slicer_area.images[2].open_in_goldtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 1, timeout=5000
        )
        assert isinstance(
            next(iter(manager._imagetool_wrappers[3]._childtools.values())), GoldTool
        )
        logger.info("Confirmed goldtool is added")

        # Trigger paint event
        manager.tree_view.expandAll()

        # Test rename goldtool
        goldtool_uid: str = manager._imagetool_wrappers[3]._childtool_indices[0]

        # Bring manager to top
        manager.tree_view.clearSelection()
        bring_manager_to_top(qtbot, manager)
        select_child_tool(manager, goldtool_uid)

        manager.tree_view.edit(manager.tree_view._model._row_index(goldtool_uid))
        qtbot.wait_until(
            lambda: manager.tree_view.state()
            == QtWidgets.QAbstractItemView.State.EditingState,
            timeout=5000,
        )
        delegate = manager.tree_view.itemDelegate()
        assert isinstance(delegate, _ImageToolWrapperItemDelegate)
        assert isinstance(delegate._current_editor(), QtWidgets.QLineEdit)
        delegate._current_editor().setText("new_goldtool_name")
        qtbot.keyClick(delegate._current_editor(), QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(
            lambda: next(
                iter(manager._imagetool_wrappers[3]._childtools.values())
            )._tool_display_name
            == "new_goldtool_name",
            timeout=5000,
        )

        # Close goldtool
        logger.info("Closing goldtool")
        manager._remove_childtool(
            next(iter(manager._imagetool_wrappers[3]._childtools.keys()))
        )

        # Show dtool
        logger.info("Opening dtool")
        manager.get_imagetool(3).slicer_area.images[2].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 1, timeout=5000
        )
        assert isinstance(
            next(iter(manager._imagetool_wrappers[3]._childtools.values())),
            DerivativeTool,
        )
        logger.info("Confirmed dtool is added")
        manager.tree_view.expandAll()
        tool_uid: str = manager._imagetool_wrappers[3]._childtool_indices[0]

        # Show dtool
        manager.show_childtool(tool_uid)

        # Tool and parent
        logger.info("Checking parent and childtool retrieval")
        tool, idx = manager._get_childtool_and_parent(tool_uid)
        assert isinstance(tool, DerivativeTool)
        assert idx == 3

        # Check dtool info printing
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_child_tool(manager, tool_uid)
        manager._update_info(uid=tool_uid)

        # Duplicate dtool
        logger.info("Duplicating dtool")
        bring_manager_to_top(qtbot, manager)
        manager.tree_view.clearSelection()
        select_child_tool(manager, tool_uid)
        manager.duplicate_selected()
        manager.tree_view.refresh(None)

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 2, timeout=5000
        )

        # Check calling invalid indices
        logger.info("Checking invalid index handling")
        parent_qindex = manager.tree_view._model._row_index(3)
        assert manager.tree_view._model.index(1, 0, parent_qindex).isValid()
        assert not manager.tree_view._model.index(4, 0, parent_qindex).isValid()

        valid_but_wrong_pointer_type = manager.tree_view._model.createIndex(
            parent_qindex.row(), parent_qindex.column(), "invalid data"
        )
        assert not manager.tree_view._model.index(
            1, 0, valid_but_wrong_pointer_type
        ).isValid()

        # Close dtools
        logger.info("Closing dtools")
        for uid in list(manager._imagetool_wrappers[3]._childtools.keys()):
            manager._remove_childtool(uid)

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[3]._childtools) == 0, timeout=5000
        )
        logger.info("Confirmed dtools are removed")

        # Bring manager to top
        logger.info("Testing mouse events")
        with qtbot.waitExposed(manager):
            manager.hide_all()  # Prevent windows from obstructing the manager
            manager.activateWindow()
            manager.raise_()
            manager.preview_action.setChecked(True)

        # Test mouse hover over list view
        # This may not work on all systems due to the way the mouse events are generated
        delegate._force_hover = True

        first_index = manager.tree_view.model().index(0, 0)
        first_rect_center = manager.tree_view.visualRect(first_index).center()
        qtbot.mouseMove(manager.tree_view.viewport())
        qtbot.mouseMove(manager.tree_view.viewport(), first_rect_center)
        qtbot.mouseMove(
            manager.tree_view.viewport(), first_rect_center - QtCore.QPoint(10, 10)
        )
        qtbot.mouseMove(manager.tree_view.viewport())  # move to blank should hide popup

        # Remove third tool
        select_tools(manager, [3])
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Test concatenate
        concat_data = xr.concat(
            [
                manager.get_imagetool(1).slicer_area._data,
                manager.get_imagetool(2).slicer_area._data,
            ],
            "concat_dim",
        )
        select_tools(manager, [1, 2])
        accept_dialog(manager.concat_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(3).slicer_area._data, concat_data
        )

        # Test concatenate (remove originals)
        select_tools(manager, [1, 2])

        def _handle_concat(dialog: _ConcatDialog):
            dialog._remove_original_check.setChecked(True)

        accept_dialog(manager.concat_action.trigger, pre_call=_handle_concat)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        xr.testing.assert_identical(
            manager.get_imagetool(4).slicer_area._data, concat_data
        )

        # Remove all selected
        select_tools(manager, [3, 4])
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        # Show about dialog
        accept_dialog(manager.about)


def test_manager_replace(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool(test_data, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        # Replace data in the tool
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(test_data**2, manager=True, replace=0)

        assert manager.get_imagetool(0).array_slicer.point_value(0) == 144.0

        # Replacing 1 should create a new tool
        itool(test_data**2, manager=True, replace=1)
        qtbot.wait_until(lambda: manager.ntools == 2)
        assert manager.get_imagetool(1).array_slicer.point_value(0) == 144.0

        # Negative indexing
        with qtbot.wait_signal(manager._sigDataReplaced):
            itool(test_data, manager=True, replace=-1)

        assert manager.get_imagetool(1).array_slicer.point_value(0) == 12.0


def test_manager_reindex(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3)

        assert manager._displayed_indices == [0, 1, 2]

        # Remove tool at index 1
        manager.remove_imagetool(1)
        qtbot.wait_until(lambda: manager.ntools == 2)

        assert manager._displayed_indices == [0, 2]

        # Reindex
        manager.reindex_action.trigger()
        qtbot.wait_until(lambda: manager._displayed_indices == [0, 1], timeout=5000)


def test_manager_server_show_remove(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Show tool at index 0
        _show_idx(0)

        # Remove tool at index 0
        _remove_idx(0)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)


def test_manager_duplicate(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Open a tool with the manager
        itool([test_data, test_data], manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

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
                manager._imagetool_wrappers[i].name
                == manager._imagetool_wrappers[i + 2].name
            )


def test_manager_sync(
    qtbot,
    move_and_compare_values,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        itool([test_data, test_data], link=True, link_colors=True, manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2)

        win0, win1 = manager.get_imagetool(0), manager.get_imagetool(1)

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


def test_manager_workspace_io(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        # Add two tools
        itool([data, data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Open dtool for first tool
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
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
                chained_dialogs=2,
            )

            # Load workspace
            accept_dialog(
                lambda: manager.load(native=False),
                pre_call=_go_to_file,
                chained_dialogs=2,
            )

            # Check if the data is loaded
            assert manager.ntools == 4

            # Check if the child dtool is also loaded
            assert len(manager._imagetool_wrappers[2]._childtools) == 1

            select_tools(manager, list(manager._imagetool_wrappers.keys()))
            accept_dialog(manager.remove_action.trigger)
            qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_manager_workspace_load_legacy(
    qtbot,
    accept_dialog,
    datadir,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()

        def _go_to_file(dialog: QtWidgets.QFileDialog):
            dialog.setDirectory(str(datadir))
            dialog.selectFile(str(datadir / "manager_workspace_legacy.h5"))
            focused = dialog.focusWidget()
            if isinstance(focused, QtWidgets.QLineEdit):
                focused.setText("manager_workspace_legacy.h5")

        # Load workspace
        accept_dialog(
            lambda: manager.load(native=False), pre_call=_go_to_file, chained_dialogs=2
        )

        # Check if the data is loaded
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        # Check data
        xr.testing.assert_identical(
            manager.get_imagetool(0).slicer_area._data,
            test_data,
        )

        select_tools(manager, list(manager._imagetool_wrappers.keys()))
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_drop_mimedata(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)
        manager.show()
        data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

        # Add three tools
        logger.info("Adding three tools")
        itool([data, data, data], link=False, manager=True)
        qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

        # Add three childtools to the first tool
        logger.info("Adding three childtools to the first tool")
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )
        logger.info("First childtool added")
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 2, timeout=5000
        )
        logger.info("Second childtool added")
        manager.get_imagetool(0).slicer_area.images[0].open_in_dtool()
        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 3, timeout=5000
        )
        logger.info("Third childtool added")
        manager.hide_all()  # Prevent windows from obstructing the manager

        # Check mimedata
        model = typing.cast("_ImageToolWrapperItemModel", manager.tree_view.model())

        # Drop None
        assert not model.canDropMimeData(None, 0, 0, 0, QtCore.QModelIndex())

        # Drop invalid mime type
        mime = QtCore.QMimeData()
        mime.setData("wrong/type", QtCore.QByteArray(b"{}"))
        assert not model.dropMimeData(
            mime, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Drop invalid payload (not json)
        mime = QtCore.QMimeData()
        mime.setData(_MIME, QtCore.QByteArray(b"not json"))
        assert not model.dropMimeData(
            mime, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        mime = QtCore.QMimeData()

        # Drop invalid payload (missing keys)
        mime.setData(
            _MIME,
            QtCore.QByteArray(
                json.dumps({"invalid_key": "invalid_value"}).encode("utf-8")
            ),
        )
        assert not model.dropMimeData(
            mime, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Drop with invalid action
        assert not model.dropMimeData(
            model.mimeData([model.index(0, 0)]),  # valid mimedata
            QtCore.Qt.DropAction.CopyAction,  # invalid action
            0,
            0,
            QtCore.QModelIndex(),
        )

        # Test single selection, top-level drops
        mime_single = model.mimeData([model.index(0, 0)])
        assert model.canDropMimeData(
            mime_single, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )
        # No-op drop
        assert not model.dropMimeData(
            mime_single, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Drop at the end
        assert model.dropMimeData(
            mime_single,
            QtCore.Qt.DropAction.MoveAction,
            model.rowCount(),
            0,
            QtCore.QModelIndex(),
        )

        # Check new order
        assert manager._displayed_indices == [1, 2, 0]

        # No-op drop (drop on itself)
        assert not model.dropMimeData(
            model.mimeData([model.index(0, 0)]),
            QtCore.Qt.DropAction.MoveAction,
            0,
            0,
            model.index(0, 0),
        )

        # Check unchanged
        assert manager._displayed_indices == [1, 2, 0]

        # Test move child tool
        parent_wrapper = model.manager._imagetool_wrappers[0]
        child_uid: str = parent_wrapper._childtool_indices[0]
        old_order = list(parent_wrapper._childtool_indices)
        parent_index: QtCore.QModelIndex = model._row_index(0)
        child_index: QtCore.QModelIndex = model._row_index(child_uid)

        # Drop to different parent
        logger.info("Testing drop to different parent")
        assert not model.dropMimeData(
            model.mimeData([child_index]),
            QtCore.Qt.DropAction.MoveAction,
            0,
            0,
            model._row_index(1),  # Different parent
        )

        # Drop to different position in the same parent
        logger.info("Testing drop to different position in the same parent")
        assert model.dropMimeData(
            model.mimeData([child_index]),
            QtCore.Qt.DropAction.MoveAction,
            2,
            0,
            parent_index,
        )

        assert list(parent_wrapper._childtool_indices) == [
            old_order[1],
            old_order[0],
            old_order[2],
        ]

        # Test multiple selection
        logger.info("Testing multiple selection drop")
        mime_multiple = model.mimeData([model.index(0, 0), model.index(0, 1)])
        assert model.canDropMimeData(
            mime_multiple, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )
        model.dropMimeData(
            mime_multiple, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Test mixed top-level and childtool selection (should be rejected)
        logger.info("Testing mixed top-level and childtool selection drop")
        parent_wrapper = model.manager._imagetool_wrappers[0]
        child_uid: str = parent_wrapper._childtool_indices[0]
        mime_mixed = model.mimeData([model._row_index(0), model._row_index(child_uid)])
        assert not model.dropMimeData(
            mime_mixed, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
        )

        # Test invalid mimedata
        logger.info("Testing invalid mimedata decoding")
        invalid_mime = QtCore.QMimeData()
        invalid_mime.setData(
            _MIME,
            QtCore.QByteArray(json.dumps({"invalid": "dictionary"}).encode("utf-8")),
        )
        assert model._decode_mime(invalid_mime) is None

        logger.info("Testing invalid mimedata decoding with non-dict payload")
        invalid_mime = QtCore.QMimeData()
        invalid_mime.setData(
            _MIME,
            QtCore.QByteArray(json.dumps("not a dict").encode("utf-8")),
        )
        assert model._decode_mime(invalid_mime) is None


def test_treeview(qtbot, accept_dialog, test_data) -> None:
    manager = ImageToolManager()

    qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

    test_data.qshow(manager=True)
    test_data.qshow(manager=True)
    qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

    manager.raise_()
    manager.activateWindow()

    model = manager.tree_view._model
    assert model.supportedDropActions() == QtCore.Qt.DropAction.MoveAction
    first_row_rect = manager.tree_view.visualRect(model.index(0, 0))

    # Click on first row
    qtbot.mouseMove(manager.tree_view.viewport(), first_row_rect.center())
    qtbot.mousePress(
        manager.tree_view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=first_row_rect.center(),
    )
    assert manager.tree_view.selected_imagetool_indices == [0]

    # Show context menu
    manager.tree_view._show_menu(first_row_rect.center())
    menu = None
    for tl in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(tl, QtWidgets.QMenu):
            menu = tl
            break
    assert isinstance(menu, QtWidgets.QMenu)

    accept_dialog(manager.close)
    qtbot.wait_until(lambda: not erlab.interactive.imagetool.manager.is_running())


@pytest.mark.parametrize("mode", ["dragdrop", "ask"])
def test_manager_open_files(
    qtbot,
    accept_dialog,
    test_data,
    mode,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager, tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/data.h5"
        test_data.to_netcdf(filename, engine="h5netcdf")

        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        with qtbot.waitExposed(manager):
            manager.show()
            manager.activateWindow()

        if mode == "dragdrop":
            mime_data = QtCore.QMimeData()
            mime_data.setUrls([QtCore.QUrl.fromLocalFile(filename)])
            evt = QtGui.QDropEvent(
                QtCore.QPointF(0.0, 0.0),
                QtCore.Qt.DropAction.CopyAction,
                mime_data,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )

            # Simulate drag and drop
            def trigger_drop():
                return manager.dropEvent(evt)
        else:

            def trigger_drop():
                return load_in_manager([filename], loader_name=None)

        accept_dialog(trigger_drop)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        xarray.testing.assert_identical(
            manager.get_imagetool(0).slicer_area.data, test_data
        )

        # Simulate drag and drop with wrong filter, retry with correct filter
        # Dialogs created are:
        # select loader → failed alert → retry → select loader
        def _choose_wrong_filter(dialog: _NameFilterDialog):
            assert dialog._valid_name_filters[0] == "xarray HDF5 Files (*.h5)"
            dialog._button_group.buttons()[-1].setChecked(True)

        def _choose_correct_filter(dialog: _NameFilterDialog):
            dialog._button_group.buttons()[0].setChecked(True)

        accept_dialog(
            trigger_drop,
            pre_call=[_choose_wrong_filter, None, None, _choose_correct_filter],
            chained_dialogs=4,
        )
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        xarray.testing.assert_identical(
            manager.get_imagetool(1).slicer_area.data, test_data
        )


def test_manager_console(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        data = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["x", "y"],
            coords={"x": np.arange(5), "y": np.arange(5)},
        )

        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        manager.activateWindow()

        manager._data_recv([data, data], kwargs={"link": True, "link_colors": True})
        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        # Open console
        manager.toggle_console()
        qtbot.wait_until(manager.console.isVisible, timeout=5000)

        def _get_last_output_contents():
            return manager.console._console_widget.kernel_manager.kernel.shell.user_ns[
                "_"
            ]

        # Test delayed import
        manager.console._console_widget.execute("era")
        assert _get_last_output_contents() == erlab.analysis

        # Test repr
        manager.console._console_widget.execute("tools")
        assert str(_get_last_output_contents()) == "0: \n1: "
        manager.console._console_widget.execute("tools[0]")

        # Select all
        select_tools(manager, list(manager._imagetool_wrappers.keys()))
        manager.console._console_widget.execute("tools.selected_data")
        assert _get_last_output_contents() == [
            wrapper.imagetool.slicer_area._data
            for wrapper in manager._imagetool_wrappers.values()
        ]

        # Test storing with ipython
        accept_dialog(manager.store_action.trigger)
        manager.console._console_widget.execute(r"%store -d data_0 data_1")

        # Test calling wrapped methods
        manager.console._console_widget.execute("tools[0].archive()")
        qtbot.wait_until(lambda: manager._imagetool_wrappers[0].archived, timeout=5000)

        # Test setting data
        manager.console._console_widget.execute(
            "tools[1].data = xr.DataArray("
            "np.arange(25).reshape((5, 5)) * 2, "
            "dims=['x', 'y'], "
            "coords={'x': np.arange(5), 'y': np.arange(5)}"
            ")",
        )
        xr.testing.assert_identical(manager.get_imagetool(1).slicer_area.data, data * 2)

        # Remove all tools
        select_tools(manager, list(manager._imagetool_wrappers.keys()))
        accept_dialog(manager.remove_action.trigger)
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

        # Test repr
        manager.console._console_widget.execute("tools")
        assert str(_get_last_output_contents()) == "No tools"

        # Test magic command: itool
        manager.console._console_widget.kernel_manager.kernel.shell.user_ns[
            "example_data"
        ] = xr.DataArray(
            np.arange(25).reshape((5, 5)),
            dims=["alpha", "eV"],
            coords={"alpha": np.arange(5), "eV": np.arange(5)},
        )
        manager.console._console_widget.execute(r"%itool example_data --cmap viridis")
        qtbot.wait_until(lambda: manager.ntools == 1)
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 12.0

        # Destroy console
        manager.console._console_widget.shutdown_kernel()
        InteractiveShell.clear_instance()
