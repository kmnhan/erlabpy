import numpy as np
import xarray as xr
from erlab.interactive.imagetool import itool
from numpy.testing import assert_almost_equal
from qtpy import QtCore


def move_and_compare_values(qtbot, win, expected, cursor=0):
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])

    # Move left
    qtbot.keyClick(
        win, QtCore.Qt.Key.Key_Left, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[1])

    # Move down
    qtbot.keyClick(
        win, QtCore.Qt.Key.Key_Down, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[2])

    # Move right
    qtbot.keyClick(
        win, QtCore.Qt.Key.Key_Right, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[3])

    # Move up
    qtbot.keyClick(win, QtCore.Qt.Key.Key_Up, QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert_almost_equal(win.array_slicer.point_value(cursor), expected[0])


def test_itool(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"])
    win = itool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitActive(win):
        win.show()
        win.activateWindow()
        win.raise_()

    move_and_compare_values(qtbot, win, [12.0, 7.0, 6.0, 11.0])

    # Transpose
    qtbot.keyClick(win, QtCore.Qt.Key.Key_T)
    move_and_compare_values(qtbot, win, [12.0, 11.0, 6.0, 7.0])

    # Set bin
    win.array_slicer.set_bin(0, 0, 2, update=False)
    win.array_slicer.set_bin(0, 1, 2, update=True)
    move_and_compare_values(qtbot, win, [9.0, 8.0, 3.0, 4.0])

    win.slicer_area.set_colormap("viridis", gamma=1.5)
