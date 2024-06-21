import time

import numpy as np
import pytest
import xarray as xr
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool.manager import ImageToolManager
from numpy.testing import assert_almost_equal
from qtpy import QtCore


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


def test_itool(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()
        win.raise_()

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
        "ColdWarm", gamma=1.5, reversed=True, high_contrast=True, zero_centered=True
    )

    # Lock levels
    win.slicer_area.lock_levels(True)
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
            "reversed": True,
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


def test_value_update(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)
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


def test_sync(qtbot):
    manager = ImageToolManager()

    qtbot.addWidget(manager)

    with qtbot.waitExposed(manager):
        manager.show()
        manager.activateWindow()
        manager.raise_()

    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    itool([data, data], link=True, link_colors=True, use_manager=True)

    t0 = time.perf_counter()
    while True:
        if len(manager._tools) == 2:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    win0, win1 = manager._tools["0"], manager._tools["1"]

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

    manager.remove_tool("0")
    manager.remove_tool("1")
    manager.close()


def test_manager(qtbot):
    win = ImageToolManager()

    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()
        win.raise_()

    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    data.qshow()

    t0 = time.perf_counter()
    while True:
        if len(win._tools) > 0:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    assert win._tools["0"].array_slicer.point_value(0) == 12.0

    # Add two tools
    itool([data, data], link=False)
    while True:
        if len(win._tools) == 3:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    # Linking
    win.tool_options["1"].check.setChecked(True)
    win.tool_options["2"].check.setChecked(True)
    win.link_selected()

    # Unlinking one unlinks both
    win.tool_options["1"].check.setChecked(True)
    win.unlink_selected()
    assert ~win._tools["1"].slicer_area.is_linked
    assert ~win._tools["2"].slicer_area.is_linked

    # Linking again
    win.tool_options["1"].check.setChecked(True)
    win.tool_options["2"].check.setChecked(True)
    win.link_selected()
    assert win._tools["1"].slicer_area.is_linked
    assert win._tools["2"].slicer_area.is_linked

    # Archiving and unarchiving
    win.archive("1")
    win.unarchive("1")

    # Removing archived tool
    win.archive("0")
    win.remove_tool("0")

    # Remove for cleanup
    win.remove_tool("1")
    win.remove_tool("2")

    win.close()
