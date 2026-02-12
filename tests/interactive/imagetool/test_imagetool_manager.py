import concurrent.futures
import io
import json
import logging
import pathlib
import pickle
import sys
import tempfile
import time
import types
import typing
import webbrowser
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
import xarray.testing
import zmq
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
from erlab.interactive.imagetool.manager._server import (
    HOST_IP,
    PORT,
    AddDataPacket,
    Response,
    _recv_multipart,
    _remove_idx,
    _show_idx,
)

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
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("new_name_1_single")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
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
        assert isinstance(delegate._current_editor, QtWidgets.QLineEdit)
        delegate._current_editor.setText("new_goldtool_name")
        qtbot.keyClick(delegate._current_editor, QtCore.Qt.Key.Key_Return)
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


def test_remove_from_window_shortcut(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        tool = manager.get_imagetool(0)

        with qtbot.waitExposed(tool):
            tool.activateWindow()
            tool.raise_()
            tool.setFocus()

        assert tool.remove_act.isVisible()

        accept_dialog(lambda: qtbot.keyClick(tool, QtCore.Qt.Key.Key_Delete))
        qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)


def test_remove_childtool_delete_shortcut(
    qtbot,
    accept_dialog,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        test_data.qshow(manager=True)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        parent_tool = manager.get_imagetool(0)
        parent_tool.slicer_area.images[0].open_in_dtool()

        qtbot.wait_until(
            lambda: len(manager._imagetool_wrappers[0]._childtools) == 1, timeout=5000
        )
        wrapper = manager._imagetool_wrappers[0]
        uid, child = next(iter(wrapper._childtools.items()))

        with qtbot.waitExposed(child):
            child.activateWindow()
            child.raise_()
            child.setFocus()

        accept_dialog(lambda: qtbot.keyClick(child, QtCore.Qt.Key.Key_Delete))
        qtbot.wait_until(lambda: uid not in wrapper._childtools, timeout=5000)


def test_manager_multi_data_not_shown(
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

        itool([test_data, test_data], manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
        assert not manager.get_imagetool(0).isVisible()
        assert not manager.get_imagetool(1).isVisible()


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

        # Try reload
        with qtbot.wait_signal(manager.get_imagetool(0).slicer_area.sigDataChanged):
            manager.get_imagetool(0).slicer_area.reload()

        # Try archive
        manager._imagetool_wrappers[0].archive()
        qtbot.wait_until(lambda: manager._imagetool_wrappers[0].archived, timeout=5000)

        # Unarchive
        manager._imagetool_wrappers[0].unarchive()
        qtbot.wait_until(
            lambda: not manager._imagetool_wrappers[0].archived, timeout=5000
        )

        # Simulate drag and drop with wrong filter, retry with correct filter
        # Dialogs created are:
        # select loader → failed alert → retry → select loader
        def _choose_wrong_filter(dialog: _NameFilterDialog):
            assert (
                next(iter(dialog._valid_loaders.keys())) == "xarray HDF5 Files (*.h5)"
            )
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


def test_manager_hover_tooltip(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    monkeypatch,
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        manager.activateWindow()

        itool([test_data, test_data], link=True, manager=True)

        qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

        manager.get_imagetool(0).slicer_area._auto_chunk()
        manager.get_imagetool(1).slicer_area._auto_chunk()

        view = manager.tree_view

        model = view._model
        delegate = view._delegate

        index = model.index(0, 0)  # first tool
        option = QtWidgets.QStyleOptionViewItem()
        delegate.initStyleOption(option, index)
        _, dask_rect, link_rect, _ = delegate._compute_icons_info(
            option, index.internalPointer()
        )

        text = None

        def fake_show_text(pos, s, *args, **kwargs):
            nonlocal text
            text = s

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", fake_show_text)

        # Hover over dask icon
        pos = dask_rect.center()
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)

        assert handled
        assert text == "Dask-backed data (chunked array)"

        # Hover over link icon
        text = None
        pos = link_rect.center()
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)

        assert handled
        assert text == "Linked (#0)"

        # Hover outside icons
        text = None
        pos = dask_rect.topRight() + QtCore.QPoint(2, 0)
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip, pos, view.viewport().mapToGlobal(pos)
        )
        handled = delegate.helpEvent(event, view, option, index)
        assert not handled
        assert text is None


def test_warning_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        warning_logger = logging.getLogger("test.warning.history")
        warning_logger.warning("First warning")
        warning_logger.warning("Second warning")

        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 2)
        qtbot.wait_until(
            lambda: all(warning.isVisible() for warning in manager._alert_dialogs)
        )

        texts = [notification.text() for notification in manager._alert_dialogs]
        assert any("First warning" in text for text in texts)
        assert any("Second warning" in text for text in texts)

        clear_all_button = manager._alert_dialogs[-1].findChild(
            QtWidgets.QPushButton, "warningDismissAllButton"
        )
        assert clear_all_button is not None

        qtbot.mouseClick(clear_all_button, QtCore.Qt.MouseButton.LeftButton, delay=10)
        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 0)


def test_warning_alert_suppressed_by_log_flag(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        suppressed = logging.LogRecord(
            name="test.warning.suppressed",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="suppressed warning",
            args=(),
            exc_info=None,
        )
        suppressed.suppress_ui_alert = True
        manager._warning_handler.emit(suppressed)
        QtWidgets.QApplication.processEvents()
        assert manager._alert_dialogs == []

        regular = logging.LogRecord(
            name="test.warning.regular",
            level=logging.WARNING,
            pathname=__file__,
            lineno=0,
            msg="regular warning",
            args=(),
            exc_info=None,
        )
        manager._warning_handler.emit(regular)
        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 1)

        manager._clear_all_alerts()
        QtWidgets.QApplication.processEvents()


def test_error_creating_imagetool_does_not_duplicate_alert_dialog(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )

    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        try:
            raise RuntimeError("boom")  # noqa: TRY301
        except RuntimeError:
            manager._error_creating_imagetool()

        QtWidgets.QApplication.processEvents()

        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_data_recv_dataset_creation_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _raise_from_dataset(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow.ImageTool,
        "from_dataset",
        staticmethod(_raise_from_dataset),
    )

    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        ds = xr.Dataset({"v": xr.DataArray(np.ones((2, 2)), dims=("x", "y"))})
        flags = manager._data_recv([ds], {})

        QtWidgets.QApplication.processEvents()

        assert flags == [False]
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_data_recv_dataarray_creation_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _raise_imagetool(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager._mainwindow,
        "ImageTool",
        _raise_imagetool,
    )

    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        flags = manager._data_recv([xr.DataArray(np.ones((2, 2)), dims=("x", "y"))], {})

        QtWidgets.QApplication.processEvents()

        assert flags == [False]
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_load_workspace_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []
    exec_calls = {"count": 0}

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _fake_exec(self):
        exec_calls["count"] += 1
        return exec_calls["count"] == 1

    def _raise_open_datatree(*args, **kwargs):
        raise RuntimeError("broken workspace")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", _fake_exec)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "selectedFiles",
        lambda self: ["broken_workspace.itws"],
    )
    monkeypatch.setattr(xr, "open_datatree", _raise_open_datatree)

    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        ImageToolManager.load(manager, native=False)

        QtWidgets.QApplication.processEvents()

        assert exec_calls["count"] >= 2  # One retry after the failure path.
        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_open_multiple_files_workspace_error_no_duplicate_alert(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    critical_calls: list[tuple[typing.Any, ...]] = []

    def _fake_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return int(QtWidgets.QDialog.DialogCode.Accepted)

    def _fake_open_datatree(*args, **kwargs):
        return object()

    def _raise_from_datatree(*args, **kwargs):
        raise RuntimeError("cannot restore workspace")

    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        staticmethod(_fake_critical),
    )
    monkeypatch.setattr(xr, "open_datatree", _fake_open_datatree)

    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        monkeypatch.setattr(manager, "_is_datatree_workspace", lambda *args: True)
        monkeypatch.setattr(manager, "_from_datatree", _raise_from_datatree)

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = pathlib.Path(tmp_dir) / "workspace.itws"
            p.write_text("placeholder", encoding="utf-8")
            manager.open_multiple_files([p], try_workspace=True)

        QtWidgets.QApplication.processEvents()

        assert len(critical_calls) == 1
        assert manager._alert_dialogs == []


def test_manager_progressbar_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        message = "Load data:   8%|8         | 1/12 [00:00<00:07,  1.54it/s]"
        manager._show_alert("INFO", logging.INFO, message, "")

        assert 12 in manager._progress_bars
        pbar = manager._progress_bars[12]
        assert pbar.labelText() == "Load data"
        assert pbar.value() == 1
        assert manager._alert_dialogs == []

        manager._show_alert(
            "INFO",
            logging.INFO,
            "Load data:  50%|##        | 6/12 [00:00<00:07,  1.54it/s]",
            "",
        )
        assert manager._progress_bars[12] is pbar
        assert pbar.value() == 6

        pbar.close()


def test_manager_alert_icons(
    qtbot,
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    recorded_icons: list[QtWidgets.QStyle.StandardPixmap | None] = []

    class _RecordingMessageDialog(erlab.interactive.utils.MessageDialog):
        def __init__(self, *args, icon_pixmap=None, **kwargs):
            recorded_icons.append(icon_pixmap)
            super().__init__(*args, icon_pixmap=icon_pixmap, **kwargs)

    monkeypatch.setattr(
        erlab.interactive.utils, "MessageDialog", _RecordingMessageDialog
    )

    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())

        manager._show_alert("INFO", logging.INFO, "info", "")
        manager._show_alert("WARNING", logging.WARNING, "warning", "")
        manager._show_alert("ERROR", logging.ERROR, "error", "")

        assert recorded_icons == [
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation,
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical,
        ]

        manager._clear_all_alerts()
        QtWidgets.QApplication.processEvents()


def test_uncaught_exception_alert(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        assert sys.excepthook == manager._handle_uncaught_exception
        manager._previous_excepthook = lambda *exc: None

        try:
            raise RuntimeError("boom")  # noqa: TRY301
        except RuntimeError:
            exc_info = sys.exc_info()

        sys.excepthook(*exc_info)

        qtbot.wait_until(lambda: len(manager._alert_dialogs) == 1)
        qtbot.wait_until(manager._alert_dialogs[0].isVisible)

        text = manager._alert_dialogs[0].text()
        detailed_text = manager._alert_dialogs[0].detailedText()

        assert "ERROR" in manager._alert_dialogs[0].windowTitle()
        assert "An unexpected error occurred" in text
        assert detailed_text is not None
        assert "boom" in detailed_text
        assert "RuntimeError" in detailed_text


def test_manager_reload(
    qtbot,
    example_loader,
    example_data_dir,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        manager.activateWindow()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        load_in_manager(
            [example_data_dir / "data_006.h5"], loader_name=example_loader.name
        )
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        # Try reload
        with qtbot.wait_signal(manager.get_imagetool(0).slicer_area.sigDataChanged):
            manager.get_imagetool(0).slicer_area.reload()

        # Try archive
        manager._imagetool_wrappers[0].archive()
        qtbot.wait_until(lambda: manager._imagetool_wrappers[0].archived, timeout=5000)

        # Unarchive
        manager._imagetool_wrappers[0].unarchive()
        qtbot.wait_until(
            lambda: not manager._imagetool_wrappers[0].archived, timeout=5000
        )

        # Try reload again
        with qtbot.wait_signal(manager.get_imagetool(0).slicer_area.sigDataChanged):
            manager.get_imagetool(0).slicer_area.reload()


def make_dataarray_unpicklable(darr):
    mod = types.ModuleType("temp_mod_for_apply_ufunc")
    exec(  # noqa: S102
        "def myfunc(dat):\n    return dat * 2\n",
        mod.__dict__,
    )
    sys.modules[mod.__name__] = mod
    from temp_mod_for_apply_ufunc import myfunc

    darr = xr.apply_ufunc(myfunc, darr.chunk(), vectorize=True, dask="parallelized")
    return darr, mod.__name__


def _create_frames(
    obj: dict[str, typing.Any], pickler_cls: type[pickle.Pickler]
) -> list[memoryview]:
    buffers: list[pickle.PickleBuffer] = []  # out-of-band frames will be appended here
    bio = io.BytesIO()

    p = pickler_cls(bio, protocol=5, buffer_callback=buffers.append)
    p.dump(obj)
    header = memoryview(bio.getbuffer())
    return [header] + [memoryview(b) for b in buffers]


def test_manager_cloudpickle(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context(use_socket=True) as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Test with unpicklable data
        darr, modname = make_dataarray_unpicklable(test_data)
        del sys.modules[modname]
        erlab.interactive.imagetool.manager.show_in_manager(darr)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        logger.info("Confirmed tool is added, checking data")
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 24.0

        # Pickle data first and remove module before trying unpickle
        darr, modname = make_dataarray_unpicklable(test_data)
        content = AddDataPacket(
            packet_type="add", data_list=darr, arguments={}
        ).model_dump(exclude_unset=True)
        ctx = zmq.Context.instance()
        sock: zmq.Socket = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        try:
            sock.connect(f"tcp://{HOST_IP}:{PORT}")
            frames = _create_frames(content, pickler_cls=pickle.Pickler)
            del sys.modules[modname]
            sock.send_multipart(frames, copy=False)

            timeout_seconds = 5.0
            start_time = time.time()
            while True:
                try:
                    response = Response(**_recv_multipart(sock, flags=zmq.NOBLOCK))
                except zmq.Again as e:
                    if time.time() - start_time > timeout_seconds:
                        raise TimeoutError(
                            "Timed out waiting for response from ZeroMQ socket."
                        ) from e
                else:
                    break
            assert response.status == "unpickle-failed"
        finally:
            sock.close()


@pytest.mark.parametrize(
    (
        "old_version",
        "new_version",
        "button_text",
        "expected_url",
        "expected_title",
        "expected_info",
    ),
    [
        (
            "",
            "1.2.3",
            "Open Release Notes",
            "https://github.com/kmnhan/erlabpy/releases",
            "ImageTool Manager Installed",
            "Welcome to ImageTool Manager! You are using version 1.2.3.",
        ),
        (
            "1.0.0",
            "1.1.0",
            "Open Documentation",
            "https://erlabpy.readthedocs.io/en/stable/user-guide/interactive/imagetool.html",
            "ImageTool Manager Updated",
            "ImageTool Manager has been successfully updated from version 1.0.0 to "
            "1.1.0.",
        ),
    ],
)
def test_manager_updated_opens_links(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    monkeypatch,
    old_version,
    new_version,
    button_text,
    expected_url,
    expected_title,
    expected_info,
) -> None:
    opened: list[str] = []

    def _open(url: str) -> bool:
        opened.append(url)
        return True

    def _accept_call(dialog: QtWidgets.QMessageBox) -> None:
        assert dialog.text() == expected_title
        assert dialog.informativeText() == expected_info
        for button in dialog.buttons():
            if button.text() == button_text:
                button.click()
                return
        pytest.fail(f"Button {button_text!r} not found.")

    monkeypatch.setattr(webbrowser, "open", _open)

    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()
        accept_dialog(
            lambda: manager.updated(old_version=old_version, new_version=new_version),
            accept_call=_accept_call,
        )

    assert opened == [expected_url]
