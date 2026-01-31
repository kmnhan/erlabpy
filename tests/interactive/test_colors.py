import numpy as np
import pytest
from qtpy import QtGui, QtWidgets

import erlab
from erlab.interactive._options import options
from erlab.interactive.colors import (
    BetterColorBarItem,
    BetterImageItem,
    ColorCycleDialog,
    ColorMapComboBox,
    pg_colormap_names,
    pg_colormap_powernorm,
)


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


def test_pg_colormap_powernorm_sets_metadata() -> None:
    cmap = pg_colormap_powernorm(
        "viridis", gamma=0.6, reverse=True, high_contrast=True, zero_centered=False
    )

    attrs = getattr(cmap, "_erlab_attrs", {})
    assert attrs["gamma"] == 0.6
    assert attrs["reverse"] is True
    assert attrs["high_contrast"] is True
    assert attrs["zero_centered"] is False


def _colorbar_edit_widget(colorbar: BetterColorBarItem) -> QtWidgets.QWidget:
    cmap_menu = getattr(colorbar, "_cmap_menu", None)
    assert cmap_menu is not None
    actions = cmap_menu.actions()
    assert actions
    widget = actions[0].defaultWidget()
    assert widget is not None
    return widget


def test_colorbar_edit_widget_populates_from_primary_image(qtbot):
    data = np.arange(16, dtype=float).reshape(4, 4)
    image = BetterImageItem(data)
    image.set_colormap("magma", gamma=0.6, reverse=True, high_contrast=True)

    colorbar = BetterColorBarItem(image=image)
    edit_widget = _colorbar_edit_widget(colorbar)
    qtbot.addWidget(edit_widget)

    edit_widget.setVisible(True)

    assert colorbar._cmap_menu.title() == "Edit colormap"
    assert edit_widget._cmap_combo.currentText() == "magma"
    assert edit_widget._gamma_widget.value() == pytest.approx(0.6)
    assert edit_widget._reversed_check.isChecked()
    assert edit_widget._high_contrast_check.isChecked()


def test_colorbar_edit_widget_applies_changes_to_images(qtbot):
    data = np.linspace(0, 1, 25, dtype=float).reshape(5, 5)
    images = [BetterImageItem(data + offset) for offset in (0.0, 1.0)]
    for img in images:
        img.set_colormap("viridis", gamma=1.0)

    colorbar = BetterColorBarItem(image=images)
    edit_widget = _colorbar_edit_widget(colorbar)
    qtbot.addWidget(edit_widget)
    edit_widget.setVisible(True)

    edit_widget._cmap_combo.setCurrentText("plasma")
    edit_widget._gamma_widget.setValue(0.4)
    edit_widget._reversed_check.setChecked(True)
    edit_widget._high_contrast_check.setChecked(True)
    QtWidgets.QApplication.processEvents()

    for img in images:
        cmap = img._colorMap
        attrs = getattr(cmap, "_erlab_attrs", {})
        assert cmap.name == "plasma"
        assert attrs["gamma"] == pytest.approx(0.4)
        assert attrs["reverse"] is True
        assert attrs["high_contrast"] is True


def test_colormap_combobox_populates_on_show(qtbot):
    names = pg_colormap_names("matplotlib", exclude_local=True)
    assert names
    default = names[0]

    combo = ColorMapComboBox()
    qtbot.addWidget(combo)
    combo.setDefaultCmap(default)

    assert combo.count() == 0
    combo.show()

    qtbot.wait_until(lambda: combo.count() > 0, timeout=2000)
    assert combo.currentText() == default


def test_pg_colormap_names_respects_runtime_exclude():
    names = pg_colormap_names("matplotlib", exclude_local=True)
    assert names
    target = names[0]

    try:
        options["colors/cmap/exclude"] = [target]
        assert target not in pg_colormap_names("matplotlib", exclude_local=True)

        options["colors/cmap/exclude"] = []
        assert target in pg_colormap_names("matplotlib", exclude_local=True)
    finally:
        options.restore()
