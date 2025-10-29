import pytest
from qtpy import QtGui

import erlab
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


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_is_dark_mode(monkeypatch, qtbot, dark):
    colors_mod = erlab.interactive.colors

    # Stub QtCore.Qt.ColorScheme with distinct sentinels
    class _Dark: ...

    class _Light: ...

    class _ColorScheme:
        Dark = _Dark()
        Light = _Light()

    class _Qt:
        ColorScheme = _ColorScheme()

    class _QtCore:
        Qt = _Qt()

    monkeypatch.setattr(colors_mod, "QtCore", _QtCore, raising=False)

    # Stub styleHints to return Light
    class DummyHints:
        def colorScheme(self):
            return _ColorScheme.Dark if dark else _ColorScheme.Light

    monkeypatch.setattr(
        colors_mod.QtGui.QGuiApplication,
        "styleHints",
        lambda: DummyHints(),
        raising=True,
    )

    assert colors_mod.is_dark_mode() == dark


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_is_dark_mode_fallback(monkeypatch, qtbot, dark):
    colors_mod = erlab.interactive.colors
    # Force fallback path
    monkeypatch.setattr(
        colors_mod.QtGui.QGuiApplication, "styleHints", lambda: None, raising=True
    )

    # Stub QPalette to control lightness
    class DummyColor:
        def __init__(self, lightness):
            self._l = lightness

        def lightness(self):
            return self._l

    class DummyPalette:
        class ColorRole:
            WindowText = object()
            Window = object()

        def color(self, role):
            if role is self.ColorRole.WindowText:
                return DummyColor(200) if dark else DummyColor(100)
            return DummyColor(100) if dark else DummyColor(200)

    monkeypatch.setattr(colors_mod.QtGui, "QPalette", DummyPalette, raising=True)

    assert colors_mod.is_dark_mode() == dark
