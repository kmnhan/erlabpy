import numpy as np
import pytest

from erlab.interactive.imagetool._dialog_widgets import (
    CoordinateEditorWidget,
    CoordinateGridWidget,
)


def _affine_preview_values(widget: CoordinateEditorWidget) -> np.ndarray:
    return np.array(
        [
            float(widget.affine_table.item(row, 1).text())
            for row in range(widget.affine_table.rowCount())
        ]
    )


def test_coordinate_widget_initialization(qtbot):
    arr = np.linspace(0, 10, 6)
    widget = CoordinateEditorWidget(arr)
    qtbot.addWidget(widget)
    assert widget.spin0.value() == 0
    assert widget.spin1.value() == 10
    assert widget.table.rowCount() == 6
    assert np.allclose(widget.new_coord, arr)


def test_coordinate_widget_mode_switch(qtbot):
    arr = np.linspace(1, 5, 5)
    widget = CoordinateEditorWidget(arr)
    qtbot.addWidget(widget)
    # Switch to Delta mode
    widget.mode_combo.setCurrentText("Delta")
    assert widget.spin1.value() == 1.0
    # Switch back to End mode
    widget.mode_combo.setCurrentText("End")
    assert widget.spin1.value() == 5.0


def test_coordinate_widget_update_table(qtbot):
    arr = np.linspace(2, 4, 3)
    widget = CoordinateEditorWidget(arr)
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
    widget = CoordinateEditorWidget(arr)
    qtbot.addWidget(widget)
    widget.spin0.setValue(5)
    widget.spin1.setValue(7)
    widget.reset()
    assert widget.spin0.value() == 0
    assert widget.spin1.value() == 2


def test_coordinate_widget_set_old_coord(qtbot):
    arr = np.array([1, 2, 3, 4])
    widget = CoordinateEditorWidget(np.zeros(4))
    qtbot.addWidget(widget)
    widget.set_old_coord(arr)
    assert np.allclose(widget._old_coord, arr)
    assert widget.table.rowCount() == 4


def test_coordinate_widget_new_coord_edit(qtbot):
    arr = np.arange(3)
    widget = CoordinateEditorWidget(arr)
    qtbot.addWidget(widget)
    # Edit table values
    widget.table.item(0, 0).setText("10")
    widget.table.item(1, 0).setText("20")
    widget.table.item(2, 0).setText("30")
    vals = widget.new_coord
    assert np.allclose(vals, [10, 20, 30])


def test_coordinate_widget_affine_identity(qtbot):
    arr = np.linspace(0, 10, 6)
    widget = CoordinateEditorWidget(arr)
    qtbot.addWidget(widget)
    widget.edit_mode_tabs.setCurrentIndex(1)
    assert widget.use_affine_transform
    assert widget.affine_scale == 1.0
    assert widget.affine_offset == 0.0
    assert np.allclose(widget.affine_coord, arr)
    assert np.allclose(_affine_preview_values(widget), arr)


def test_coordinate_widget_affine_scale_offset(qtbot):
    arr = np.array([1.0, 2.0, 4.0])
    widget = CoordinateEditorWidget(arr)
    qtbot.addWidget(widget)
    widget.edit_mode_tabs.setCurrentIndex(1)
    widget.scale_spin.setValue(2.0)
    assert np.allclose(widget.affine_coord, [2.0, 4.0, 8.0])
    widget.offset_spin.setValue(-0.5)
    assert np.allclose(widget.affine_coord, [1.5, 3.5, 7.5])
    assert np.allclose(_affine_preview_values(widget), [1.5, 3.5, 7.5])


def test_coordinate_widget_affine_offset_only(qtbot):
    arr = np.array([1.0, 2.0, 4.0])
    widget = CoordinateEditorWidget(arr)
    qtbot.addWidget(widget)
    widget.edit_mode_tabs.setCurrentIndex(1)
    widget.offset_spin.setValue(3.0)
    assert np.allclose(widget.affine_coord, [4.0, 5.0, 7.0])


def test_coordinate_widget_affine_reset(qtbot):
    arr = np.array([1.0, 2.0, 4.0])
    widget = CoordinateEditorWidget(arr)
    qtbot.addWidget(widget)
    widget.edit_mode_tabs.setCurrentIndex(1)
    widget.scale_spin.setValue(3.0)
    widget.offset_spin.setValue(5.0)
    widget.reset()
    assert widget.use_affine_transform
    assert widget.affine_scale == 1.0
    assert widget.affine_offset == 0.0
    assert np.allclose(widget.affine_coord, arr)


def test_coordinate_widget_affine_scalar(qtbot):
    widget = CoordinateEditorWidget(np.asarray(2.0))
    qtbot.addWidget(widget)
    widget.edit_mode_tabs.setCurrentIndex(1)
    widget.scale_spin.setValue(4.0)
    widget.offset_spin.setValue(1.0)
    assert widget.use_affine_transform
    assert np.allclose(np.atleast_1d(widget.affine_coord), [9.0])


def test_coordinate_grid_widget_variable_count(qtbot):
    widget = CoordinateGridWidget(
        np.array([0.0, 2.0]),
        editable_count=True,
        preserve_shape=False,
        require_complete=True,
        numeric_reference=True,
        disable_singleton_controls=False,
        reset_table_to_reference=False,
        update_table_on_mode_changed=True,
    )
    qtbot.addWidget(widget)

    widget.count_spin.setValue(5)
    assert widget.table.rowCount() == 5
    assert np.allclose(widget.new_coord, np.linspace(0.0, 2.0, 5))

    widget.mode_combo.setCurrentText("Delta")
    widget.spin0.setValue(1.0)
    widget.spin1.setValue(0.5)
    widget.count_spin.setValue(4)
    assert np.allclose(widget.new_coord, [1.0, 1.5, 2.0, 2.5])


def test_coordinate_grid_widget_requires_complete_values(qtbot):
    widget = CoordinateGridWidget(
        np.array([0.0, 1.0]),
        editable_count=True,
        preserve_shape=False,
        require_complete=True,
        numeric_reference=True,
    )
    qtbot.addWidget(widget)

    widget.table.item(0, 0).setText("")
    with pytest.raises(ValueError, match="Missing value in row 0"):
        _ = widget.new_coord
