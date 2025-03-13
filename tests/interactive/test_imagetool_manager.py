import tempfile

import numpy as np
import pytest
import xarray as xr
import xarray.testing
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.fermiedge import GoldTool
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool.manager import ImageToolManager
from erlab.interactive.imagetool.manager._dialogs import (
    _NameFilterDialog,
    _RenameDialog,
)
from erlab.interactive.imagetool.manager._modelview import (
    _ImageToolWrapperItemDelegate,
    _ImageToolWrapperListModel,
)


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
    selection_model = manager.list_view.selectionModel()

    for index in indices:
        qmodelindex = manager.list_view._model._row_index(index)
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


@pytest.mark.parametrize("use_socket", [True, False], ids=["socket", "no_socket"])
def test_manager(qtbot, accept_dialog, test_data, use_socket) -> None:
    erlab.interactive.imagetool.manager._always_use_socket = use_socket

    erlab.interactive.imagetool.manager.main(execute=False)
    manager = erlab.interactive.imagetool.manager._manager_instance

    qtbot.addWidget(manager)
    qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

    test_data.qshow(manager=True)
    qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

    assert manager.get_tool(0).array_slicer.point_value(0) == 12.0

    # Add two tools
    for tool in itool([test_data, test_data], link=False, execute=False, manager=False):
        tool.move_to_manager()

    qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

    # Linking
    select_tools(manager, [1, 2])
    manager.link_selected()
    manager.list_view.refresh(None)

    # Unlinking one unlinks both
    select_tools(manager, [1])
    manager.unlink_selected()
    manager.list_view.refresh(None)
    assert not manager.get_tool(1).slicer_area.is_linked
    assert not manager.get_tool(2).slicer_area.is_linked

    # Linking again
    select_tools(manager, [1, 2])
    manager.link_selected()
    manager.list_view.refresh(None)
    assert manager.get_tool(1).slicer_area.is_linked
    assert manager.get_tool(2).slicer_area.is_linked

    # Archiving and unarchiving
    manager._tool_wrappers[1].archive()
    manager._tool_wrappers[1].touch_archive()
    assert manager._tool_wrappers[1].archived
    manager._tool_wrappers[1].unarchive()
    assert not manager._tool_wrappers[1].archived

    # Toggle visibility
    geometry = manager.get_tool(1).geometry()
    manager._tool_wrappers[1].close()
    assert not manager.get_tool(1).isVisible()
    manager._tool_wrappers[1].show()
    assert manager.get_tool(1).geometry() == geometry

    # Removing archived tool
    manager._tool_wrappers[0].archive()
    manager.remove_tool(0)
    qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

    # Batch renaming
    select_tools(manager, [1, 2])

    def _handle_renaming(dialog: _RenameDialog):
        dialog._new_name_lines[0].setText("new_name_1")
        dialog._new_name_lines[1].setText("new_name_2")

    _handler = accept_dialog(manager.rename_action.trigger, pre_call=_handle_renaming)
    assert manager._tool_wrappers[1].name == "new_name_1"
    assert manager._tool_wrappers[2].name == "new_name_2"

    # Rename single
    select_tools(manager, [2], deselect=True)
    select_tools(manager, [1])
    manager.rename_action.trigger()

    qtbot.wait_until(
        lambda: manager.list_view.state()
        == QtWidgets.QAbstractItemView.State.EditingState,
        timeout=5000,
    )
    delegate = manager.list_view.itemDelegate()
    assert isinstance(delegate, _ImageToolWrapperItemDelegate)
    assert isinstance(delegate._current_editor(), QtWidgets.QLineEdit)
    delegate._current_editor().setText("new_name_1_single")
    qtbot.keyClick(delegate._current_editor(), QtCore.Qt.Key.Key_Return)
    qtbot.wait_until(
        lambda: manager._tool_wrappers[1].name == "new_name_1_single", timeout=5000
    )

    # Batch archiving
    select_tools(manager, [1])
    manager.archive_action.trigger()
    manager._tool_wrappers[1].unarchive()

    # Show and hide windows including archived ones
    select_tools(manager, [1])
    manager.archive_action.trigger()

    select_tools(manager, [1, 2])
    manager.hide_action.trigger()  # Hide non-archived window, does nothing to archived
    manager.show_action.trigger()  # Unarchive the archived one and show both

    assert not manager._tool_wrappers[1].archived
    assert not manager._tool_wrappers[2].archived
    assert manager.get_tool(1).isVisible()
    assert manager.get_tool(2).isVisible()

    # Select tools
    select_tools(manager, [1, 2])
    _handler = accept_dialog(manager.concat_action.trigger)
    qtbot.wait_until(lambda: manager.ntools == 3, timeout=5000)

    xr.testing.assert_identical(
        manager.get_tool(3).slicer_area._data,
        xr.concat(
            [
                manager.get_tool(1).slicer_area._data,
                manager.get_tool(2).slicer_area._data,
            ],
            "concat_dim",
        ),
    )

    # Show goldtool
    manager.get_tool(3).slicer_area.images[2].open_in_goldtool()
    assert isinstance(next(iter(manager._additional_windows.values())), GoldTool)

    # Close goldtool
    next(iter(manager._additional_windows.values())).close()

    # Bring manager to top
    with qtbot.waitExposed(manager):
        manager.hide_all()  # Prevent windows from obstructing the manager
        manager.preview_action.setChecked(True)
        manager.activateWindow()
        manager.raise_()

    # Test mouse hover over list view
    # This may not work on all systems due to the way the mouse events are generated
    delegate._force_hover = True

    first_index = manager.list_view.model().index(0)
    first_rect_center = manager.list_view.visualRect(first_index).center()
    qtbot.mouseMove(manager.list_view.viewport())
    qtbot.mouseMove(manager.list_view.viewport(), first_rect_center)
    qtbot.mouseMove(
        manager.list_view.viewport(), first_rect_center - QtCore.QPoint(10, 10)
    )
    qtbot.mouseMove(manager.list_view.viewport())  # move to blank should hide popup

    # Remove all selected
    select_tools(manager, [1, 2, 3])
    _handler = accept_dialog(manager.remove_action.trigger)
    qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

    # Run garbage collection
    manager.gc_action.trigger()

    # Show about dialog
    _handler = accept_dialog(manager.about)

    manager.close()
    erlab.interactive.imagetool.manager._manager_instance = None
    erlab.interactive.imagetool.manager._always_use_socket = False


def test_manager_sync(qtbot, move_and_compare_values, test_data) -> None:
    manager = ImageToolManager()
    qtbot.addWidget(manager)
    qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

    itool([test_data, test_data], link=True, link_colors=True, manager=True)

    qtbot.wait_until(lambda: manager.ntools == 2)

    win0, win1 = manager.get_tool(0), manager.get_tool(1)

    win1.slicer_area.set_colormap("RdYlBu", gamma=1.5)
    assert (
        win0.slicer_area._colormap_properties == win1.slicer_area._colormap_properties
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

    manager.remove_all_tools()
    qtbot.wait_until(lambda: manager.ntools == 0)
    manager.close()


def test_manager_workspace_io(qtbot, accept_dialog) -> None:
    manager = ImageToolManager()

    qtbot.addWidget(manager)
    qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

    # Add two tools
    itool([data, data], link=False, manager=True)
    qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/workspace.h5"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("workspace.h5")

    # Save workspace
    _handler = accept_dialog(lambda: manager.save(native=False), pre_call=_go_to_file)

    # Load workspace
    _handler = accept_dialog(lambda: manager.load(native=False), pre_call=_go_to_file)

    # Check if the data is loaded
    assert manager.ntools == 4

    select_tools(manager, list(manager._tool_wrappers.keys()))
    _handler = accept_dialog(manager.remove_action.trigger)
    qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)
    manager.close()
    tmp_dir.cleanup()


def test_can_drop_mime_data(qtbot) -> None:
    manager = ImageToolManager()
    model = _ImageToolWrapperListModel(manager)

    mime_data = QtCore.QMimeData()
    mime_data.setData("application/json", QtCore.QByteArray())
    assert model.canDropMimeData(
        mime_data, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
    )
    manager.close()


def test_listview(qtbot, accept_dialog, test_data) -> None:
    manager = ImageToolManager()

    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

    test_data.qshow(manager=True)
    test_data.qshow(manager=True)
    qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

    manager.raise_()
    manager.activateWindow()

    model = manager.list_view._model
    assert model.supportedDropActions() == QtCore.Qt.DropAction.MoveAction
    first_row_rect = manager.list_view.rectForIndex(model.index(0))

    # Click on first row
    qtbot.mouseMove(manager.list_view.viewport(), first_row_rect.center())
    qtbot.mousePress(
        manager.list_view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=first_row_rect.center(),
    )
    assert manager.list_view.selected_tool_indices == [0]

    # Show context menu
    manager.list_view._show_menu(first_row_rect.center())
    menu = None
    for tl in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(tl, QtWidgets.QMenu):
            menu = tl
            break
    assert isinstance(menu, QtWidgets.QMenu)

    _handler = accept_dialog(manager.close)
    qtbot.wait_until(lambda: not erlab.interactive.imagetool.manager.is_running())


def test_manager_drag_drop_files(qtbot, accept_dialog, test_data) -> None:
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/data.h5"
    test_data.to_netcdf(filename, engine="h5netcdf")

    manager = ImageToolManager()
    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

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
    _handler = accept_dialog(lambda: manager.dropEvent(evt))
    qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
    xarray.testing.assert_identical(manager.get_tool(0).slicer_area.data, test_data)

    # Simulate drag and drop with wrong filter, retry with correct filter
    # Dialogs created are:
    # select loader → failed alert → retry → select loader
    def _choose_wrong_filter(dialog: _NameFilterDialog):
        assert dialog._valid_name_filters[0] == "xarray HDF5 Files (*.h5)"
        dialog._button_group.buttons()[-1].setChecked(True)

    def _choose_correct_filter(dialog: _NameFilterDialog):
        dialog._button_group.buttons()[0].setChecked(True)

    _handler = accept_dialog(
        lambda: manager.dropEvent(evt),
        pre_call=[_choose_wrong_filter, None, None, _choose_correct_filter],
        chained_dialogs=4,
    )
    qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)
    xarray.testing.assert_identical(manager.get_tool(1).slicer_area.data, test_data)

    # Cleanup
    manager.remove_all_tools()
    qtbot.wait_until(lambda: manager.ntools == 0)
    manager.close()
    tmp_dir.cleanup()


def test_manager_console(qtbot, accept_dialog) -> None:
    manager = ImageToolManager()
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )

    qtbot.addWidget(manager)
    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    manager._data_recv([data, data], kwargs={"link": True, "link_colors": True})
    qtbot.wait_until(lambda: manager.ntools == 2, timeout=5000)

    # Open console
    manager.toggle_console()
    qtbot.wait_until(manager.console.isVisible, timeout=5000)

    def _get_last_output_contents():
        return manager.console._console_widget.kernel_manager.kernel.shell.user_ns["_"]

    # Test delayed import
    manager.console._console_widget.execute("era")
    assert _get_last_output_contents() == erlab.analysis

    # Test repr
    manager.console._console_widget.execute("tools")
    assert str(_get_last_output_contents()) == "0: \n1: "
    manager.console._console_widget.execute("tools[0]")

    # Select all
    select_tools(manager, list(manager._tool_wrappers.keys()))
    manager.console._console_widget.execute("tools.selected_data")
    assert _get_last_output_contents() == [
        wrapper.tool.slicer_area._data for wrapper in manager._tool_wrappers.values()
    ]

    # Test storing with ipython
    _handler = accept_dialog(manager.store_action.trigger)
    manager.console._console_widget.execute(r"%store -d data_0 data_1")

    # Test calling wrapped methods
    manager.console._console_widget.execute("tools[0].archive()")
    qtbot.wait_until(lambda: manager._tool_wrappers[0].archived, timeout=5000)

    # Test setting data
    manager.console._console_widget.execute(
        "tools[1].data = xr.DataArray("
        "np.arange(25).reshape((5, 5)) * 2, "
        "dims=['x', 'y'], "
        "coords={'x': np.arange(5), 'y': np.arange(5)}"
        ")"
    )
    xr.testing.assert_identical(manager.get_tool(1).slicer_area.data, data * 2)

    # Remove all tools
    select_tools(manager, list(manager._tool_wrappers.keys()))
    _handler = accept_dialog(manager.remove_action.trigger)
    qtbot.wait_until(lambda: manager.ntools == 0, timeout=5000)

    # Test repr
    manager.console._console_widget.execute("tools")
    assert str(_get_last_output_contents()) == "No tools"
    manager.close()
