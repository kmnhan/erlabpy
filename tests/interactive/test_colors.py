from qtpy import QtGui

from erlab.interactive.colors import ColorCycleDialog


def test_ColorCycleDialog_basic(qtbot):
    # Create a ColorCycleDialog with 3 colors
    colors = [QtGui.QColor("red"), QtGui.QColor("green"), QtGui.QColor("blue")]
    dialog = ColorCycleDialog(colors)
    qtbot.addWidget(dialog)
    assert dialog.colors == tuple(colors)
    # Change a color via the color button
    dialog.color_btns[0].setColor(QtGui.QColor("yellow"))
    dialog._update_color()
    assert dialog.colors[0] == QtGui.QColor("yellow")
    # Reset should restore original colors
    dialog.reset()
    assert dialog.colors == tuple(colors)
    # Accept should emit sigAccepted
    with qtbot.waitSignal(dialog.sigAccepted, timeout=1000) as blocker:
        dialog.accept()
    assert blocker.args[0] == tuple(colors)


def test_ColorCycleDialog_set_from_cmap(qtbot):
    colors = [QtGui.QColor("red"), QtGui.QColor("green"), QtGui.QColor("blue")]
    dialog = ColorCycleDialog(colors)
    qtbot.addWidget(dialog)
    dialog.cmap_combo.setCurrentText("viridis")
    dialog.reverse_check.setChecked(True)
    dialog.start_spin.setValue(0.2)
    dialog.stop_spin.setValue(0.8)
    dialog.set_from_cmap()
    # Should have 3 colors, all QColor
    assert len(dialog.colors) == 3
    assert all(isinstance(c, QtGui.QColor) for c in dialog.colors)


def test_ColorCycleDialog_restore_defaults(qtbot):
    orig_colors = [QtGui.QColor("red"), QtGui.QColor("green"), QtGui.QColor("blue")]
    default_colors = [
        QtGui.QColor("magenta"),
        QtGui.QColor("cyan"),
        QtGui.QColor("yellow"),
    ]
    dialog = ColorCycleDialog(orig_colors, default_colors=default_colors)
    qtbot.addWidget(dialog)
    dialog.restore_defaults()
    assert dialog.colors == tuple(default_colors)
    dialog.reset()
    assert dialog.colors == tuple(orig_colors)


def test_ColorCycleDialog_preview_cursors(qtbot):
    colors = [QtGui.QColor("red"), QtGui.QColor("green")]
    dialog = ColorCycleDialog(colors, preview_cursors=True)
    qtbot.addWidget(dialog)
    # Should have lines and spans for preview
    assert len(dialog.lines) == 2
    assert len(dialog.spans) == 2
    # Changing color updates preview pens
    dialog.color_btns[0].setColor(QtGui.QColor("black"))
    dialog._update_color()
    assert dialog.curves[0].opts["pen"].color().name() == QtGui.QColor("black").name()
