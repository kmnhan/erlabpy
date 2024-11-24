import tempfile
import time

import numpy as np
import xarray as xr
import xarray.testing
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool.manager import (
    ImageToolManager,
    _NameFilterDialog,
    _RenameDialog,
)


def test_manager(qtbot, accept_dialog):
    win = ImageToolManager()

    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    data.qshow()

    t0 = time.perf_counter()
    while True:
        if win.ntools > 0:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    assert win.get_tool(0).array_slicer.point_value(0) == 12.0

    # Add two tools
    itool([data, data], link=False)
    while True:
        if win.ntools == 3:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    # Linking
    win.tool_options[1].check.setChecked(True)
    win.tool_options[2].check.setChecked(True)
    win.link_selected()

    # Unlinking one unlinks both
    win.tool_options[1].check.setChecked(True)
    win.unlink_selected()
    assert not win.get_tool(1).slicer_area.is_linked
    assert not win.get_tool(2).slicer_area.is_linked

    # Linking again
    win.tool_options[1].check.setChecked(True)
    win.tool_options[2].check.setChecked(True)
    win.link_selected()
    assert win.get_tool(1).slicer_area.is_linked
    assert win.get_tool(2).slicer_area.is_linked

    # Archiving and unarchiving
    win.tool_options[1].archive()
    win.tool_options[1].toggle_archive()
    assert not win.tool_options[1].archived

    # Toggle visibility
    geometry = win.get_tool(1).geometry()
    win.tool_options[1].toggle_visibility()
    assert not win.get_tool(1).isVisible()
    win.tool_options[1].toggle_visibility()
    assert win.get_tool(1).geometry() == geometry

    # Removing archived tool
    win.tool_options[0].archive()
    win.remove_tool(0)
    qtbot.waitUntil(lambda: win.ntools == 2, timeout=2000)

    # Batch renaming
    win.tool_options[1].check.setChecked(True)
    win.tool_options[2].check.setChecked(True)

    def _handle_renaming(dialog: _RenameDialog):
        dialog._new_name_lines[0].setText("new_name_1")
        dialog._new_name_lines[1].setText("new_name_2")

    accept_dialog(win.rename_action.trigger, pre_call=_handle_renaming)
    assert win.tool_options[1].name == "new_name_1"
    assert win.tool_options[2].name == "new_name_2"

    # Batch archiving
    win.tool_options[1].check.setChecked(True)
    win.archive_action.trigger()
    win.tool_options[1].unarchive()

    # GC action
    win.gc_action.trigger()

    # Remove all checked
    win.tool_options[1].check.setChecked(True)
    win.tool_options[2].check.setChecked(True)
    accept_dialog(win.close_action.trigger)
    qtbot.waitUntil(lambda: win.ntools == 0, timeout=2000)

    win.close()


def test_manager_sync(qtbot, move_and_compare_values):
    manager = ImageToolManager()

    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()

    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
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


def test_manager_workspace_io(qtbot, accept_dialog):
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

    for opt in manager.tool_options.values():
        opt.check.setChecked(True)

    accept_dialog(manager.close_action.trigger)
    qtbot.waitUntil(lambda: manager.ntools == 0, timeout=2000)
    manager.close()


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


def test_manager_drag_drop(qtbot, accept_dialog):
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["x", "y"],
        coords={"x": np.arange(5), "y": np.arange(5)},
    )
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
    manager.remove_tool(1)
    manager.close()
    tmp_dir.cleanup()
