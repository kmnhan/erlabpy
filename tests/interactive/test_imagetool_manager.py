import tempfile
import time

import numpy as np
import pytest
import xarray as xr
import xarray.testing
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool.manager import (
    ImageToolManager,
    _ImageToolWrapperItemDelegate,
    _ImageToolWrapperListModel,
    _NameFilterDialog,
    _RenameDialog,
)


@pytest.fixture
def data():
    return xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
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


@pytest.mark.parametrize("use_socket", [True, False])
def test_manager(qtbot, accept_dialog, data, use_socket) -> None:
    erlab.interactive.imagetool.manager._always_use_socket = use_socket

    erlab.interactive.imagetool.manager.main(execute=False)
    manager = erlab.interactive.imagetool.manager._manager_instance

    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    data.qshow()

    t0 = time.perf_counter()
    while True:
        if manager.ntools > 0:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    assert manager.get_tool(0).array_slicer.point_value(0) == 12.0

    # Add two tools
    itool([data, data], link=False)
    while True:
        if manager.ntools == 3:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    # Linking
    select_tools(manager, [1, 2])
    manager.link_selected()

    # Unlinking one unlinks both
    select_tools(manager, [1])
    manager.unlink_selected()
    assert not manager.get_tool(1).slicer_area.is_linked
    assert not manager.get_tool(2).slicer_area.is_linked

    # Linking again
    select_tools(manager, [1, 2])
    manager.link_selected()
    assert manager.get_tool(1).slicer_area.is_linked
    assert manager.get_tool(2).slicer_area.is_linked

    # Archiving and unarchiving
    manager._tool_wrappers[1].archive()
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
    qtbot.waitUntil(lambda: manager.ntools == 2, timeout=2000)

    # Batch renaming
    select_tools(manager, [1, 2])

    def _handle_renaming(dialog: _RenameDialog):
        dialog._new_name_lines[0].setText("new_name_1")
        dialog._new_name_lines[1].setText("new_name_2")

    accept_dialog(manager.rename_action.trigger, pre_call=_handle_renaming)
    assert manager._tool_wrappers[1].name == "new_name_1"
    assert manager._tool_wrappers[2].name == "new_name_2"

    # Rename single
    select_tools(manager, [2], deselect=True)
    select_tools(manager, [1])
    manager.rename_action.trigger()

    qtbot.waitUntil(
        lambda: manager.list_view.state()
        == QtWidgets.QAbstractItemView.State.EditingState,
        timeout=2000,
    )
    delegate = manager.list_view.itemDelegate()
    assert isinstance(delegate, _ImageToolWrapperItemDelegate)
    assert isinstance(delegate._current_editor(), QtWidgets.QLineEdit)
    delegate._current_editor().setText("new_name_1_single")
    qtbot.keyClick(delegate._current_editor(), QtCore.Qt.Key.Key_Return)
    qtbot.waitUntil(
        lambda: manager._tool_wrappers[1].name == "new_name_1_single", timeout=2000
    )

    # Batch archiving
    select_tools(manager, [1])
    manager.archive_action.trigger()
    manager._tool_wrappers[1].unarchive()

    # GC action
    manager.gc_action.trigger()

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
    accept_dialog(manager.concat_action.trigger)
    qtbot.waitUntil(lambda: manager.ntools == 3, timeout=2000)

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

    # Remove all selected
    select_tools(manager, [1, 2, 3])
    accept_dialog(manager.remove_action.trigger)
    qtbot.waitUntil(lambda: manager.ntools == 0, timeout=2000)

    # Show about dialog
    accept_dialog(manager.about)

    manager.close()
    erlab.interactive.imagetool.manager._manager_instance = None
    erlab.interactive.imagetool.manager._always_use_socket = False


def test_manager_sync(qtbot, move_and_compare_values, data) -> None:
    manager = ImageToolManager()

    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    itool([data, data], link=True, link_colors=True, use_manager=True)

    t0 = time.perf_counter()
    while True:
        if manager.ntools == 2:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    win0, win1 = manager.get_tool(0), manager.get_tool(1)

    win1.slicer_area.set_colormap("ColdWarm", gamma=1.5)
    assert (
        win0.slicer_area._colormap_properties == win1.slicer_area._colormap_properties
    )

    move_and_compare_values(qtbot, win0, [12.0, 7.0, 6.0, 11.0], target_win=win1)

    # Transpose
    qtbot.keyClick(win1, QtCore.Qt.Key.Key_T)
    move_and_compare_values(qtbot, win0, [12.0, 11.0, 6.0, 7.0], target_win=win1)

    # Set bin
    win1.slicer_area.set_bin(0, 2, update=False)
    win1.slicer_area.set_bin(1, 2, update=True)

    # Set all bins, same effect as above since we only have 1 cursor
    win1.slicer_area.set_bin_all(1, 2, update=True)

    move_and_compare_values(qtbot, win0, [9.0, 8.0, 3.0, 4.0], target_win=win1)

    manager.remove_tool(0)
    manager.remove_tool(1)
    manager.close()


def test_manager_workspace_io(qtbot, accept_dialog) -> None:
    manager = ImageToolManager()

    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])

    # Add two tools
    t0 = time.perf_counter()
    itool([data, data], link=False)
    while True:
        if manager.ntools == 2:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/workspace.h5"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("workspace.h5")

    # Save workspace
    accept_dialog(lambda: manager.save(native=False), pre_call=_go_to_file)

    # Load workspace
    accept_dialog(lambda: manager.load(native=False), pre_call=_go_to_file)

    # Check if the data is loaded
    assert manager.ntools == 4

    select_tools(manager, list(manager._tool_wrappers.keys()))
    accept_dialog(manager.remove_action.trigger)
    qtbot.waitUntil(lambda: manager.ntools == 0, timeout=2000)
    manager.close()


def test_can_drop_mime_data(qtbot) -> None:
    manager = ImageToolManager()
    model = _ImageToolWrapperListModel(manager)

    mime_data = QtCore.QMimeData()
    mime_data.setData("application/json", QtCore.QByteArray())
    assert model.canDropMimeData(
        mime_data, QtCore.Qt.DropAction.MoveAction, 0, 0, QtCore.QModelIndex()
    )
    manager.close()


def test_listview(qtbot, accept_dialog, data) -> None:
    manager = ImageToolManager()

    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    data.qshow()
    data.qshow()
    qtbot.waitUntil(lambda: manager.ntools == 2, timeout=2000)

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
    menu.close()

    accept_dialog(manager.close)


def test_manager_drag_drop_files(qtbot, accept_dialog, data) -> None:
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/data.h5"
    data.to_netcdf(filename, engine="h5netcdf")

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
    accept_dialog(lambda: manager.dropEvent(evt))
    qtbot.waitUntil(lambda: manager.ntools == 1, timeout=2000)
    xarray.testing.assert_identical(manager.get_tool(0).slicer_area.data, data)

    # Simulate drag and drop with wrong filter, retry with correct filter
    # Dialogs created are:
    # select loader → failed alert → retry → select loader
    def _choose_wrong_filter(dialog: _NameFilterDialog):
        assert dialog._valid_name_filters[0] == "xarray HDF5 Files (*.h5)"
        dialog._button_group.buttons()[-1].setChecked(True)

    def _choose_correct_filter(dialog: _NameFilterDialog):
        dialog._button_group.buttons()[0].setChecked(True)

    accept_dialog(
        lambda: manager.dropEvent(evt),
        pre_call=[_choose_wrong_filter, None, None, _choose_correct_filter],
        chained_dialogs=4,
    )
    qtbot.waitUntil(lambda: manager.ntools == 2, timeout=2000)
    xarray.testing.assert_identical(manager.get_tool(1).slicer_area.data, data)

    # Cleanup
    manager.remove_tool(0)
    accept_dialog(manager.close)
    tmp_dir.cleanup()


def test_manager_console(qtbot, accept_dialog, data) -> None:
    manager = ImageToolManager()

    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    itool([data, data], link=True, link_colors=True, use_manager=True)
    qtbot.waitUntil(lambda: manager.ntools == 2, timeout=2000)

    # Open console
    manager.toggle_console()
    qtbot.waitUntil(manager.console.isVisible, timeout=2000)

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

    # Test calling wrapped methods
    manager.console._console_widget.execute("tools[0].archive()")
    qtbot.waitUntil(lambda: manager._tool_wrappers[0].archived, timeout=2000)

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
    accept_dialog(manager.remove_action.trigger)
    qtbot.waitUntil(lambda: manager.ntools == 0, timeout=2000)

    # Test repr
    manager.console._console_widget.execute("tools")
    assert str(_get_last_output_contents()) == "No tools"
    manager.close()
