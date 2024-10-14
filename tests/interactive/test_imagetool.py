import tempfile
import time
from collections.abc import Callable

import numpy as np
import pyperclip
import pytest
import xarray as xr
import xarray.testing
from numpy.testing import assert_almost_equal
from qtpy import QtCore, QtWidgets

import erlab.analysis.transform
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool.manager import ImageToolManager


def accept_dialog(
    dialog_trigger: Callable, time_out: int = 5, pre_call: Callable | None = None
) -> QtWidgets.QDialog:
    """Accept a dialog during testing.

    If there is no dialog, it waits until one is created for a maximum of 5 seconds (by
    default). Adapted from `this issue comment on pytest-qt
    <https://github.com/pytest-dev/pytest-qt/issues/256#issuecomment-1915675942>`_.

    Parameters
    ----------
    dialog_trigger
        Callable that triggers the dialog creation.
    time_out
        Maximum time (seconds) to wait for the dialog creation.
    pre_call
        Callable that takes the dialog as a single argument. If provided, it is executed
        before calling ``.accept()`` on the dialog.
    """
    dialog = None
    start_time = time.time()

    # Helper function to catch the dialog instance and hide it
    def dialog_creation():
        # Wait for the dialog to be created or timeout
        nonlocal dialog
        while dialog is None and time.time() - start_time < time_out:
            dialog = QtWidgets.QApplication.activeModalWidget()

        # Avoid errors when dialog is not created
        if isinstance(dialog, QtWidgets.QDialog):
            if pre_call is not None:
                pre_call(dialog)
            if (
                isinstance(dialog, QtWidgets.QMessageBox)
                and dialog.defaultButton() is not None
            ):
                dialog.defaultButton().click()
            else:
                dialog.accept()

    # Create a thread to get the dialog instance and call dialog_creation trigger
    QtCore.QTimer.singleShot(1, dialog_creation)
    dialog_trigger()

    assert isinstance(
        dialog, QtWidgets.QDialog
    ), f"No dialog was created after {time_out} seconds. Dialog type: {type(dialog)}"


def move_and_compare_values(qtbot, win, expected, cursor=0, target_win=None):
    if target_win is None:
        target_win = win
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])

    # Move left
    qtbot.keyClick(
        target_win, QtCore.Qt.Key.Key_Left, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[1])

    # Move down
    qtbot.keyClick(
        target_win, QtCore.Qt.Key.Key_Down, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[2])

    # Move right
    qtbot.keyClick(
        target_win, QtCore.Qt.Key.Key_Right, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[3])

    # Move up
    qtbot.keyClick(
        target_win, QtCore.Qt.Key.Key_Up, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])


@pytest.mark.parametrize("val_dtype", [np.float32, np.float64, np.int32, np.int64])
@pytest.mark.parametrize("coord_dtype", [np.float32, np.float64, np.int32, np.int64])
def test_itool_dtypes(qtbot, val_dtype, coord_dtype):
    data = xr.DataArray(
        np.arange(25).reshape((5, 5)).astype(val_dtype),
        dims=["x", "y"],
        coords={
            "x": np.arange(5, dtype=coord_dtype),
            "y": np.arange(5, dtype=coord_dtype),
        },
    )
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])
    win.close()
    del win


def test_itool_load(qtbot):
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/data.h5"

    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    data.to_netcdf(filename, engine="h5netcdf")

    win = itool(np.zeros((2, 2)), execute=False)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("data.h5")

    accept_dialog(lambda: win.mnb._open_file(native=False), pre_call=_go_to_file)

    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

    win.close()

    tmp_dir.cleanup()


def test_itool_save(qtbot):
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/data.h5"

    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False)

    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("data.h5")

    accept_dialog(lambda: win.mnb._export_file(native=False), pre_call=_go_to_file)

    win.close()

    xr.testing.assert_equal(data, xr.load_dataarray(filename))
    tmp_dir.cleanup()


def test_itool(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

    # Snap
    qtbot.keyClick(win, QtCore.Qt.Key.Key_S)

    # Transpose
    qtbot.keyClick(win, QtCore.Qt.Key.Key_T)
    move_and_compare_values(qtbot, win, [12.0, 11.0, 6.0, 7.0])

    # Set bin
    win.array_slicer.set_bin(0, 0, 2, update=False)
    win.array_slicer.set_bin(0, 1, 2, update=True)
    move_and_compare_values(qtbot, win, [9.0, 8.0, 3.0, 4.0])

    # Test code generation
    assert win.array_slicer.qsel_code(0, (0,)) == ".qsel(x=1.5, x_width=2.0)"

    # Set colormap and gamma
    win.slicer_area.set_colormap(
        "ColdWarm", gamma=1.5, reverse=True, high_contrast=True, zero_centered=True
    )

    # Lock levels
    win.slicer_area.lock_levels(True)
    win.slicer_area.levels = (1.0, 23.0)
    assert win.slicer_area._colorbar.cb._copy_limits() == str((1.0, 23.0))

    # Test color limits editor
    clw = win.slicer_area._colorbar.cb._clim_menu.actions()[0].defaultWidget()
    assert clw.min_spin.value() == win.slicer_area.levels[0]
    assert clw.max_spin.value() == win.slicer_area.levels[1]
    clw.min_spin.setValue(1.0)
    assert clw.min_spin.value() == 1.0
    clw.max_spin.setValue(2.0)
    assert clw.max_spin.value() == 2.0
    clw.rst_btn.click()
    assert win.slicer_area.levels == (0.0, 24.0)
    win.slicer_area.levels = (1.0, 23.0)
    win.slicer_area.lock_levels(False)

    # Undo and redo
    win.slicer_area.undo()
    qtbot.keyClick(win, QtCore.Qt.Key.Key_Z, QtCore.Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(
        win,
        QtCore.Qt.Key.Key_Z,
        QtCore.Qt.KeyboardModifier.ControlModifier
        | QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    win.slicer_area.redo()

    # Check restoring the state works
    old_state = dict(win.slicer_area.state)
    win.slicer_area.state = old_state

    # Add and remove cursor
    win.slicer_area.add_cursor()
    expected_state = {
        "color": {
            "cmap": "ColdWarm",
            "gamma": 1.5,
            "reverse": True,
            "high_contrast": True,
            "zero_centered": True,
            "levels_locked": False,
        },
        "slice": {
            "dims": ("y", "x"),
            "bins": [[2, 2], [2, 2]],
            "indices": [[2, 2], [2, 2]],
            "values": [[2.0, 2.0], [2.0, 2.0]],
            "snap_to_data": True,
        },
        "current_cursor": 1,
        "manual_limits": {"x": [-0.5, 4.5], "y": [-0.5, 4.5]},
        "splitter_sizes": list(old_state["splitter_sizes"]),
        "cursor_colors": ["#cccccc", "#ffff00"],
    }
    assert win.slicer_area.state == expected_state
    win.slicer_area.remove_current_cursor()
    assert win.slicer_area.state == old_state

    # See if restoring the state works for the second cursor
    win.slicer_area.state = expected_state
    assert win.slicer_area.state == expected_state

    # Setting data
    win.slicer_area.set_data(data.rename("new_data"))
    assert win.windowTitle() == "new_data"

    win.close()

    del win


def test_itool_tools(qtbot, gold):
    win = itool(gold, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    # Test code generation
    assert win.slicer_area.images[0].selection_code == ""

    # Open goldtool from main image
    win.slicer_area.images[0].open_in_goldtool()
    assert isinstance(win.slicer_area.images[0]._goldtool, QtWidgets.QWidget)
    win.slicer_area.images[0]._goldtool.close()

    # Open dtool from main image
    win.slicer_area.images[0].open_in_dtool()
    assert isinstance(win.slicer_area.images[0]._dtool, QtWidgets.QWidget)
    win.slicer_area.images[0]._dtool.close()

    win.close()
    del win


def test_itool_ds(qtbot):
    # If no 2D to 4D data is present in given Dataset, ValueError is raised
    with pytest.raises(
        ValueError, match="No valid data for ImageTool found in the Dataset"
    ):
        itool(
            xr.Dataset(
                {
                    "data1d": xr.DataArray(np.arange(5), dims=["x"]),
                    "data0d": 1,
                }
            ),
            execute=False,
        )

    data = xr.Dataset(
        {
            "data1d": xr.DataArray(np.arange(5), dims=["x"]),
            "a": xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]),
            "b": xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]),
        }
    )
    wins = itool(data, execute=False, link=True)
    assert isinstance(wins, list)
    assert len(wins) == 2

    qtbot.addWidget(wins[0])
    qtbot.addWidget(wins[1])

    with qtbot.waitExposed(wins[0]):
        wins[0].show()
    with qtbot.waitExposed(wins[1]):
        wins[1].show()

    assert wins[0].windowTitle() == "a"
    assert wins[1].windowTitle() == "b"

    # Check if properly linked
    assert wins[0].slicer_area._linking_proxy == wins[1].slicer_area._linking_proxy

    wins[0].slicer_area.unlink()
    wins[1].slicer_area.unlink()
    wins[0].close()
    wins[1].close()

    del wins


def test_value_update(qtbot):
    win = itool(
        xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]), execute=False
    )

    new_vals = -np.arange(25).reshape((5, 5)).astype(float)
    win.slicer_area.update_values(new_vals)
    assert_almost_equal(win.array_slicer.point_value(0), -12.0)

    with pytest.raises(ValueError, match="DataArray dimensions do not match"):
        win.slicer_area.update_values(
            xr.DataArray(np.arange(24).reshape((2, 2, 6)), dims=["x", "y", "z"])
        )
    with pytest.raises(ValueError, match="DataArray dimensions do not match"):
        win.slicer_area.update_values(
            xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "z"])
        )
    with pytest.raises(ValueError, match="DataArray shape does not match"):
        win.slicer_area.update_values(
            xr.DataArray(np.arange(24).reshape((4, 6)), dims=["x", "y"])
        )
    with pytest.raises(ValueError, match="^Data shape does not match.*"):
        win.slicer_area.update_values(np.arange(24).reshape((4, 6)))

    win.close()


def test_sync(qtbot):
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


def test_manager(qtbot):
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
    win.tool_options[1].unarchive()

    # Removing archived tool
    win.tool_options[0].archive()
    win.remove_tool(0)
    qtbot.waitUntil(lambda: win.ntools == 2, timeout=2000)

    # Remove all checked
    win.tool_options[1].check.setChecked(True)
    win.tool_options[2].check.setChecked(True)
    accept_dialog(win.close_selected)
    qtbot.waitUntil(lambda: win.ntools == 0, timeout=2000)

    win.close()


def test_itool_rotate(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)).astype(float), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    # Test dialog
    def _set_dialog_params(dialog):
        dialog.angle_spin.setValue(60.0)
        dialog.reshape_check.setChecked(True)
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._rotate, pre_call=_set_dialog_params)

    # Check if the data is rotated
    xarray.testing.assert_allclose(
        win.slicer_area._data,
        erlab.analysis.transform.rotate(data, angle=60.0, reshape=True),
    )

    # Test guidelines
    win.slicer_area.set_data(data)
    win.slicer_area.main_image.set_guidelines(3)
    assert win.slicer_area.main_image.is_guidelines_visible

    win.slicer_area.main_image._guidelines_items[0].setAngle(90.0 - 30.0)
    win.slicer_area.main_image._guidelines_items[-1].setPos((3.0, 3.1))

    def _set_dialog_params(dialog):
        assert dialog.angle_spin.value() == 30.0
        assert dialog.center_spins[0].value() == 3.0
        assert dialog.center_spins[1].value() == 3.1
        dialog.copy_button.click()
        dialog.reshape_check.setChecked(True)
        dialog.new_window_check.setChecked(False)

    accept_dialog(win.mnb._rotate, pre_call=_set_dialog_params)

    # Check if the data is rotated
    xarray.testing.assert_allclose(
        win.slicer_area._data,
        erlab.analysis.transform.rotate(
            data, angle=30.0, center=(3.0, 3.1), reshape=True
        ),
    )

    # Test copy button
    assert pyperclip.paste().startswith("era.transform.rotate")

    # Transpose should remove guidelines
    qtbot.keyClick(win, QtCore.Qt.Key.Key_T)
    assert not win.slicer_area.main_image.is_guidelines_visible

    win.close()
