import numpy as np
import pytest
import xarray as xr
from erlab.interactive.imagetool import itool
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

    # Set colormap and gamma
    win.slicer_area.set_colormap(
        "ColdWarm", gamma=1.5, reversed=True, highContrast=True, zeroCentered=True
    )

    # Lock levels
    win.slicer_area.lock_levels(True)
    win.slicer_area.lock_levels(False)

    # Add and remove cursor
    win.slicer_area.add_cursor()
    win.slicer_area.remove_current_cursor()


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
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool([data, data], link=True, link_colors=True, execute=False)
    for w in win:
        qtbot.addWidget(w)

        with qtbot.waitExposed(w):
            w.show()
            w.activateWindow()
            w.raise_()

    win[1].slicer_area.set_colormap("ColdWarm", gamma=1.5)
    assert (
        win[0].slicer_area.colormap_properties == win[1].slicer_area.colormap_properties
    )

    move_and_compare_values(qtbot, win[0], [12.0, 7.0, 6.0, 11.0], target_win=win[1])

    # Transpose
    qtbot.keyClick(win[1], QtCore.Qt.Key.Key_T)
    move_and_compare_values(qtbot, win[0], [12.0, 11.0, 6.0, 7.0], target_win=win[1])

    # Set bin
    win[1].slicer_area.set_bin(0, 2, update=False)
    win[1].slicer_area.set_bin(1, 2, update=True)

    # Set all bins, same effect as above since we only have 1 cursor
    win[1].slicer_area.set_bin_all(1, 2, update=True)

    move_and_compare_values(qtbot, win[0], [9.0, 8.0, 3.0, 4.0], target_win=win[1])
