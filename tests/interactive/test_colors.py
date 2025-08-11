from qtpy import QtGui

from erlab.interactive.colors import ColorCycleDialog


def test_ColorCycleDialog_flow(qtbot):
    # Initial and default colors
    orig_colors = [QtGui.QColor("red"), QtGui.QColor("green"), QtGui.QColor("blue")]
    default_colors = [
        QtGui.QColor("magenta"),
        QtGui.QColor("cyan"),
        QtGui.QColor("yellow"),
    ]
    dialog = ColorCycleDialog(orig_colors, default_colors=default_colors)
    qtbot.addWidget(dialog)

    # Initial state
    assert dialog.colors == tuple(orig_colors)

    # Modify a color
    dialog.color_btns[0].setColor(QtGui.QColor("yellow"))
    dialog._update_color()
    assert dialog.colors[0] == QtGui.QColor("yellow")

    # Reset restores originals
    dialog.reset()
    assert dialog.colors == tuple(orig_colors)

    # Set from colormap
    dialog.cmap_combo.setCurrentText("viridis")
    dialog.reverse_check.setChecked(True)
    dialog.start_spin.setValue(0.2)
    dialog.stop_spin.setValue(0.8)
    dialog.set_from_cmap()
    assert len(dialog.colors) == 3
    assert all(isinstance(c, QtGui.QColor) for c in dialog.colors)

    # Restore defaults then reset back to original
    dialog.restore_defaults()
    assert dialog.colors == tuple(default_colors)
    dialog.reset()
    assert dialog.colors == tuple(orig_colors)

    # Accept emits sigAccepted with current (original) colors
    with qtbot.waitSignal(dialog.sigAccepted, timeout=1000) as blocker:
        dialog.accept()
    assert blocker.args[0] == tuple(orig_colors)


def test_ColorCycleDialog_preview(qtbot):
    colors = [QtGui.QColor("red"), QtGui.QColor("green")]
    dialog = ColorCycleDialog(colors, preview_cursors=True)
    qtbot.addWidget(dialog)

    # Preview components created
    assert len(dialog.lines) == 2
    assert len(dialog.spans) == 2

    # Color change updates preview pen
    new_color = QtGui.QColor("black")
    dialog.color_btns[0].setColor(new_color)
    dialog._update_color()
    assert dialog.curves[0].opts["pen"].color().name() == new_color.name()
