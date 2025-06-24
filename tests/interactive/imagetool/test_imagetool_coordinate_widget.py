import numpy as np
import pytest

from erlab.interactive.imagetool.dialogs import _CoordinateWidget


def test_coordinate_widget_initialization(qtbot):
    arr = np.linspace(0, 10, 6)
    widget = _CoordinateWidget(arr)
    qtbot.addWidget(widget)
    assert widget.spin0.value() == 0
    assert widget.spin1.value() == 10
    assert widget.table.rowCount() == 6
    assert np.allclose(widget.new_coord, arr)


def test_coordinate_widget_mode_switch(qtbot):
    arr = np.linspace(1, 5, 5)
    widget = _CoordinateWidget(arr)
    qtbot.addWidget(widget)
    # Switch to Delta mode
    widget.mode_combo.setCurrentText("Delta")
    assert widget.spin1.value() == 1.0
    # Switch back to End mode
    widget.mode_combo.setCurrentText("End")
    assert widget.spin1.value() == 5.0


def test_coordinate_widget_update_table(qtbot):
    arr = np.linspace(2, 4, 3)
    widget = _CoordinateWidget(arr)
    qtbot.addWidget(widget)
    widget.spin0.setValue(10)
    widget.spin1.setValue(20)
    widget.mode_combo.setCurrentText("End")
    widget.update_table()
    vals = widget._current_values_end
    for i in range(3):
        item = widget.table.item(i, 0)
        assert float(item.text()) == pytest.approx(vals[i])


def test_coordinate_widget_reset(qtbot):
    arr = np.linspace(0, 2, 3)
    widget = _CoordinateWidget(arr)
    qtbot.addWidget(widget)
    widget.spin0.setValue(5)
    widget.spin1.setValue(7)
    widget.reset()
    assert widget.spin0.value() == 0
    assert widget.spin1.value() == 2


def test_coordinate_widget_set_old_coord(qtbot):
    arr = np.array([1, 2, 3, 4])
    widget = _CoordinateWidget(np.zeros(4))
    qtbot.addWidget(widget)
    widget.set_old_coord(arr)
    assert np.allclose(widget._old_coord, arr)
    assert widget.table.rowCount() == 4


def test_coordinate_widget_new_coord_edit(qtbot):
    arr = np.arange(3)
    widget = _CoordinateWidget(arr)
    qtbot.addWidget(widget)
    # Edit table values
    widget.table.item(0, 0).setText("10")
    widget.table.item(1, 0).setText("20")
    widget.table.item(2, 0).setText("30")
    vals = widget.new_coord
    assert np.allclose(vals, [10, 20, 30])
