import numpy as np
import pyqtgraph as pg
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


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"gamma": 0.6}, id="normal-gamma"),
        pytest.param({"gamma": 0.01}, id="aggressive-gamma"),
        pytest.param({"gamma": 0.6, "reverse": True}, id="reverse"),
        pytest.param({"gamma": 0.6, "high_contrast": True}, id="high-contrast"),
        pytest.param({"gamma": 0.6, "zero_centered": True}, id="zero-centered"),
    ],
)
def test_lut_powernorm_matches_dense_powernorm(kwargs) -> None:
    colors_mod = erlab.interactive.colors

    dense = pg_colormap_powernorm("viridis", **kwargs)
    lut_cmap = colors_mod._pg_colormap_powernorm_lut("viridis", **kwargs)

    assert lut_cmap.name == "viridis"
    assert np.array_equal(lut_cmap.getStops()[1], dense.getStops()[1])


def test_better_image_item_lut_matches_dense_powernorm() -> None:
    kwargs = {
        "gamma": 0.01,
        "reverse": True,
        "high_contrast": True,
        "zero_centered": True,
    }
    image = BetterImageItem(np.arange(16, dtype=float).reshape(4, 4))
    image.set_colormap("magma", **kwargs)

    dense = pg_colormap_powernorm("magma", **kwargs)

    assert np.array_equal(image.lut, dense.getStops()[1])


def test_lut_powernorm_returns_read_only_shared_arrays() -> None:
    colors_mod = erlab.interactive.colors

    cmap = colors_mod._pg_colormap_powernorm_lut("viridis", gamma=0.6)
    pos, lut = cmap.getStops()

    assert not pos.flags.writeable
    assert not lut.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        lut[0, 0] = lut[0, 0]


def test_better_image_item_shares_named_powernorm_lut() -> None:
    data = np.arange(16, dtype=float).reshape(4, 4)
    images = [BetterImageItem(data), BetterImageItem(data)]

    for image in images:
        image.set_colormap("viridis", gamma=0.6, reverse=True)

    assert images[0].lut is images[1].lut
    assert images[0]._colorMap.getStops()[1] is images[1]._colorMap.getStops()[1]


def test_better_image_item_custom_colormap_bypasses_named_lut_cache() -> None:
    data = np.arange(16, dtype=float).reshape(4, 4)
    custom = pg.ColorMap(
        [0.0, 0.5, 1.0],
        [(0, 0, 0, 255), (128, 32, 16, 255), (255, 255, 255, 255)],
        name="custom",
    )
    original_pos = custom.pos.copy()
    original_color = custom.color.copy()
    images = [BetterImageItem(data), BetterImageItem(data)]

    for image in images:
        image.set_colormap(custom, gamma=0.6, reverse=True)

    assert images[0]._colorMap.name == "custom"
    assert images[0].lut is not images[1].lut
    assert np.array_equal(custom.pos, original_pos)
    assert np.array_equal(custom.color, original_color)


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


def test_colorbar_keeps_menu_widget_action_references() -> None:
    image = BetterImageItem(np.arange(16, dtype=float).reshape(4, 4))
    image.set_colormap("viridis", gamma=1.0)
    colorbar = BetterColorBarItem(image=image)

    assert colorbar._clim_menu.actions()[0] is colorbar._clim_action
    assert colorbar._clim_action.defaultWidget() is colorbar._clim_widget
    assert colorbar._center_zero_action in colorbar.vb.menu.actions()

    assert colorbar._cmap_menu.actions()[0] is colorbar._cmap_action
    assert colorbar._cmap_action.defaultWidget() is colorbar._cmap_widget


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
