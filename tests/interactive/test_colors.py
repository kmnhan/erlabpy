import numpy as np
import pyqtgraph as pg
import pytest
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._colormap as interactive_colormap
from erlab.interactive._options import options
from erlab.interactive.colors import (
    BetterColorBarItem,
    BetterImageItem,
    ColorCycleDialog,
    ColorMapComboBox,
    matplotlib_colormap_name,
    pg_colormap_from_name,
    pg_colormap_names,
    pg_colormap_powernorm,
    pg_colormap_to_QPixmap,
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


def test_colorbar_lazy_context_menu_access_paths(qtbot, monkeypatch) -> None:
    image = BetterImageItem(np.arange(16, dtype=float).reshape(4, 4))
    image.set_colormap("viridis", gamma=1.0)
    colorbar = BetterColorBarItem(image=image, _defer_context_menu_setup=True)
    widget = pg.PlotWidget(plotItem=colorbar)
    qtbot.addWidget(widget)

    with monkeypatch.context() as patch:
        patch.setattr(colorbar, "_menu_for_context_setup", lambda: None)
        assert colorbar._ensure_context_menu() is None

    menu = colorbar.getMenu()
    assert menu is colorbar.ctrlMenu
    assert colorbar._ensure_context_menu() is colorbar._context_menu
    assert colorbar.getContextMenus(None) is None


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


def test_colorbar_handle_drag_updates_large_data_levels(qtbot):
    mn, mx = 1e11, 1e15
    image = BetterImageItem(np.array([[mn, mx]], dtype=float))
    image.set_colormap("viridis", gamma=1.0)
    colorbar = BetterColorBarItem(image=image)
    widget = pg.PlotWidget(plotItem=colorbar)
    qtbot.addWidget(widget)
    widget.resize(100, 400)
    widget.show()
    QtWidgets.QApplication.processEvents()

    colorbar.setSpanRegion((mn, mx))
    QtWidgets.QApplication.processEvents()

    start = widget.mapFromScene(colorbar.vb.mapViewToScene(pg.Point(0.5, mn)))
    start += QtCore.QPoint(0, -3)
    end = QtCore.QPoint(start.x(), start.y() - 30)

    qtbot.mousePress(widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(widget.viewport(), pos=end)
    qtbot.mouseRelease(widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=end)
    QtWidgets.QApplication.processEvents()

    lower, upper = colorbar.spanRegion()
    assert lower > mn
    assert upper == pytest.approx(mx)
    assert image.getLevels() == pytest.approx((lower, upper))


def test_colorbar_image_maps_to_large_data_limits(qtbot):
    mn, mx = 1e11, 1e15
    image = BetterImageItem(np.array([[mn, mx]], dtype=float))
    image.set_colormap("viridis", gamma=1.0)
    colorbar = BetterColorBarItem(image=image)
    widget = pg.PlotWidget(plotItem=colorbar)
    qtbot.addWidget(widget)
    widget.resize(100, 400)
    widget.show()
    QtWidgets.QApplication.processEvents()

    mapped_rect = colorbar._colorbar.mapRectToParent(colorbar._colorbar.boundingRect())

    assert mapped_rect.left() == pytest.approx(0.0)
    assert mapped_rect.width() == pytest.approx(1.0)
    assert mapped_rect.top() == pytest.approx(mn)
    assert mapped_rect.bottom() == pytest.approx(mx)


def test_colorbar_limit_change_preserves_data_levels(qtbot):
    image = BetterImageItem(np.arange(100, dtype=float).reshape(10, 10))
    image.set_colormap("viridis", gamma=1.0)
    image.setLevels((2.0, 8.0))
    colorbar = BetterColorBarItem(image=image, limits=(0.0, 10.0))
    widget = pg.PlotWidget(plotItem=colorbar)
    qtbot.addWidget(widget)
    widget.show()
    QtWidgets.QApplication.processEvents()

    colorbar.setSpanRegion((2.0, 8.0))
    colorbar.setLimits((0.0, 20.0))
    QtWidgets.QApplication.processEvents()

    assert colorbar.spanRegion() == pytest.approx((2.0, 8.0))
    assert image.getLevels() == pytest.approx((2.0, 8.0))


def test_colorbar_normalized_helpers_handle_invalid_limits(qtbot, monkeypatch):
    image = BetterImageItem(np.array([[0.0, 1.0]], dtype=float))
    image.set_colormap("viridis", gamma=1.0)
    colorbar = BetterColorBarItem(image=image)
    widget = pg.PlotWidget(plotItem=colorbar)
    qtbot.addWidget(widget)

    colorbar._fixedlimits = (np.inf, np.inf)
    monkeypatch.setattr(colorbar._colorbar, "width", lambda: 0)
    monkeypatch.setattr(colorbar._colorbar, "height", lambda: 0)

    normalized = colorbar._normalized_transform()
    colorbar_image = colorbar._colorbar_image_transform()

    assert normalized.map(QtCore.QPointF(0.0, 0.0)).y() == pytest.approx(0.0)
    assert normalized.map(QtCore.QPointF(0.0, 1.0)).y() == pytest.approx(1.0)
    assert colorbar_image.mapRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0)) == (
        QtCore.QRectF(0.0, 0.0, 1.0, 1.0)
    )

    colorbar._fixedlimits = (5.0, 5.0)

    assert colorbar._level_to_span_unit(np.nan) == 0.0
    assert colorbar._level_to_span_unit(6.0) == 0.0
    assert colorbar._span_unit_to_level(0.5) == 5.0


def test_colorbar_nonfinite_limits_and_missing_colormap_metadata() -> None:
    image = BetterImageItem(np.zeros((2, 2)))
    colorbar = BetterColorBarItem()

    assert colorbar.colormap_properties is None

    colorbar._primary_image = lambda: None
    assert colorbar.colormap_properties is None

    colorbar._primary_image = lambda: image
    image.setImage(np.full((2, 2), np.nan), autoLevels=False)
    image.setLevels((np.nan, np.nan))
    with pytest.warns(RuntimeWarning, match="All-NaN slice"):
        assert colorbar.limits == (0.0, 1.0)
    assert colorbar.colormap_properties is None


def test_colorbar_syncs_unset_levels_and_reset_paths(qtbot):
    image = BetterImageItem(np.arange(100, dtype=float).reshape(10, 10))
    image.set_colormap("viridis", gamma=1.0)
    colorbar = BetterColorBarItem(image=image, limits=(0.0, 10.0))
    widget = pg.PlotWidget(plotItem=colorbar)
    qtbot.addWidget(widget)
    widget.show()
    QtWidgets.QApplication.processEvents()

    colorbar.setSpanRegion((2.0, 8.0))
    image.setLevels(None)
    colorbar.setLimits((0.0, 20.0))

    assert colorbar.spanRegion() == pytest.approx((2.0, 8.0))
    assert image.getLevels() is None

    colorbar.image_changed()
    assert image.getLevels() == pytest.approx((2.0, 8.0))

    colorbar.reset_levels()
    assert colorbar.spanRegion() == pytest.approx((0.0, 20.0))
    assert image.getLevels() == pytest.approx((0.0, 20.0))

    image.setLevels((4.0, 12.0))
    colorbar.image_level_changed()

    assert colorbar.spanRegion() == pytest.approx((4.0, 12.0))


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


def test_colormap_combobox_ignores_unavailable_text(qtbot):
    names = pg_colormap_names("matplotlib", exclude_local=True)
    assert names
    valid = names[0]
    missing = "__erlab_missing_colormap__"

    combo = ColorMapComboBox()
    qtbot.addWidget(combo)
    combo.setDefaultCmap(valid)
    combo.show()

    qtbot.wait_until(lambda: combo.count() > 0, timeout=2000)
    combo.setCurrentText(valid)
    combo.setCurrentText(missing)

    assert combo.currentText() == valid


def test_colormap_combobox_exposes_matplotlib_name_for_colorcet(qtbot):
    pytest.importorskip("colorcet")

    combo = ColorMapComboBox()
    qtbot.addWidget(combo)
    combo.load_all()

    combo.setCurrentText("CET_C1")

    assert combo.currentText() == "CET_C1"
    assert combo.current_matplotlib_name() == "cet_CET_C1"

    combo.setCurrentText("cet_CET_C1")

    assert combo.currentText() == "CET_C1"
    assert combo.current_matplotlib_name() == "cet_CET_C1"


def test_matplotlib_colormap_name_checks_again_after_loading(monkeypatch) -> None:
    available: set[str] = set()

    def has_colormap(name: str) -> bool:
        return name in available

    def load_colormaps() -> None:
        available.update({"late_cmap", "cet_late_colorcet"})

    token = interactive_colormap._ALL_COLORMAPS_LOADED.set(False)
    try:
        monkeypatch.setattr(
            interactive_colormap, "_matplotlib_has_colormap", has_colormap
        )
        monkeypatch.setattr(interactive_colormap, "load_all_colormaps", load_colormaps)

        assert matplotlib_colormap_name("late_cmap") == "late_cmap"
        available.clear()
        assert matplotlib_colormap_name("late_colorcet") == "cet_late_colorcet"
    finally:
        interactive_colormap._ALL_COLORMAPS_LOADED.reset(token)


def test_colormap_combobox_explicit_icon_and_fallback_helpers(qtbot) -> None:
    combo = ColorMapComboBox()
    qtbot.addWidget(combo)

    assert combo._find_colormap_index(None) == -1

    pixmap = QtGui.QPixmap(16, 16)
    pixmap.fill(QtGui.QColor("red"))
    with QtCore.QSignalBlocker(combo):
        combo._add_colormap_item("viridis", QtGui.QIcon(pixmap))

    assert combo.itemData(0) == "viridis"
    assert not combo.itemIcon(0).isNull()

    combo.clear()
    combo.addItem("viridis")
    combo.setCurrentIndex(0)
    assert combo.current_matplotlib_name() == "viridis"


def test_colormap_combobox_load_thumbnail_ignores_invalid_index(qtbot) -> None:
    combo = ColorMapComboBox()
    qtbot.addWidget(combo)

    combo.load_thumbnail(-1)

    assert combo.count() == 0


def test_colormap_thumbnail_pixmap_uses_string_cache(qtbot) -> None:
    erlab.interactive.colors._cached_colormap_qpixmap.cache_clear()

    first = pg_colormap_to_QPixmap("viridis")
    second = pg_colormap_to_QPixmap("viridis")

    assert not first.isNull()
    assert not second.isNull()
    cache_info = erlab.interactive.colors._cached_colormap_qpixmap.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1


def test_colormap_thumbnail_pixmap_normalizes_colorcet_cache_key(qtbot) -> None:
    pytest.importorskip("colorcet")

    erlab.interactive.colors._cached_colormap_qpixmap.cache_clear()

    display_pixmap = pg_colormap_to_QPixmap("CET_C1")
    matplotlib_pixmap = pg_colormap_to_QPixmap("cet_CET_C1")

    assert not display_pixmap.isNull()
    assert not matplotlib_pixmap.isNull()
    cache_info = erlab.interactive.colors._cached_colormap_qpixmap.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1


def test_colormap_combobox_load_all_only_renders_current_thumbnail(
    qtbot, monkeypatch
) -> None:
    rendered: list[str] = []

    def record_thumbnail(name, *args, **kwargs):
        rendered.append(name)
        return QtGui.QPixmap(64, 16)

    monkeypatch.setattr(
        erlab.interactive.colors, "pg_colormap_to_QPixmap", record_thumbnail
    )
    combo = ColorMapComboBox()
    qtbot.addWidget(combo)
    combo.setDefaultCmap("viridis")
    combo.ensure_populated()
    rendered.clear()

    combo.load_all()

    assert combo.count() == len(pg_colormap_names("all", exclude_local=True))
    assert combo.currentText() == "viridis"
    assert not combo.itemIcon(combo.currentIndex()).isNull()
    assert rendered == ["viridis"]

    other_index = next(i for i in range(combo.count()) if i != combo.currentIndex())
    combo.load_thumbnail(other_index)

    assert rendered == ["viridis", combo.itemText(other_index)]


def test_colormap_combobox_blocked_updates_keep_thumbnail_and_signal_state(
    qtbot,
) -> None:
    combo = ColorMapComboBox()
    qtbot.addWidget(combo)
    combo.setDefaultCmap("viridis")

    assert combo.signalsBlocked()
    combo.ensure_populated()
    assert not combo.signalsBlocked()
    assert combo.currentText() == "viridis"
    assert not combo.itemIcon(combo.currentIndex()).isNull()

    with QtCore.QSignalBlocker(combo):
        combo.setCurrentText("magma")
    assert not combo.signalsBlocked()
    assert combo.currentText() == "magma"
    assert not combo.itemIcon(combo.currentIndex()).isNull()


def test_colormap_combobox_close_blocks_teardown_signals(qtbot) -> None:
    combo = ColorMapComboBox()
    qtbot.addWidget(combo)
    combo.show()

    assert combo.close()
    assert combo.signalsBlocked()


def test_colormap_combobox_popup_populates_once_before_loading_thumbnails(
    qtbot, monkeypatch
) -> None:
    rendered: list[str] = []

    def record_thumbnail(name, *args, **kwargs):
        rendered.append(name)
        return QtGui.QPixmap(64, 16)

    monkeypatch.setattr(
        erlab.interactive.colors, "pg_colormap_to_QPixmap", record_thumbnail
    )
    monkeypatch.setattr(QtWidgets.QComboBox, "showPopup", lambda self: None)
    combo = ColorMapComboBox()
    qtbot.addWidget(combo)

    combo.showPopup()
    count = combo.count()

    assert count == len(pg_colormap_names("matplotlib", exclude_local=True))
    assert rendered == [combo.itemText(i) for i in range(count)]
    assert all(not combo.itemIcon(i).isNull() for i in range(count))

    combo._populate()

    assert combo.count() == count


def test_colormap_combobox_load_all_keeps_current_selection(qtbot) -> None:
    combo = ColorMapComboBox()
    qtbot.addWidget(combo)
    combo.ensure_populated()
    if combo.count() < 2:
        pytest.skip("Need multiple colormaps to test selection preservation")
    combo.setCurrentIndex(1)
    current_name = combo.current_matplotlib_name()

    combo.load_all()

    assert combo.current_matplotlib_name() == current_name


def test_colormap_comboboxes_share_thumbnail_cache(qtbot) -> None:
    erlab.interactive.colors._cached_colormap_qpixmap.cache_clear()
    first = ColorMapComboBox()
    second = ColorMapComboBox()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.ensure_populated()
    second.ensure_populated()

    first.load_thumbnail(0)
    second.load_thumbnail(0)

    cache_info = erlab.interactive.colors._cached_colormap_qpixmap.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1


def test_colormap_thumbnail_pixmap_does_not_cache_colormap_objects(qtbot) -> None:
    erlab.interactive.colors._cached_colormap_qpixmap.cache_clear()
    cmap = pg_colormap_from_name("viridis")

    first = pg_colormap_to_QPixmap(cmap)
    second = pg_colormap_to_QPixmap(cmap)

    assert not first.isNull()
    assert not second.isNull()
    cache_info = erlab.interactive.colors._cached_colormap_qpixmap.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 0


def test_colormap_thumbnail_pixmap_respects_skip_cache_false(
    qtbot, monkeypatch
) -> None:
    original = erlab.interactive.colors.pg_colormap_from_name
    calls: list[tuple[str, bool]] = []

    def record_colormap_lookup(name: str, skipCache: bool = True):
        calls.append((name, skipCache))
        return original(name, skipCache=skipCache)

    monkeypatch.setattr(
        erlab.interactive.colors, "pg_colormap_from_name", record_colormap_lookup
    )
    erlab.interactive.colors._cached_colormap_qpixmap.cache_clear()

    pixmap = pg_colormap_to_QPixmap("viridis", skipCache=False)

    assert not pixmap.isNull()
    assert calls == [("viridis", False)]
    cache_info = erlab.interactive.colors._cached_colormap_qpixmap.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 0


def test_colormap_combobox_missing_default_load_all_falls_back(qtbot):
    missing = "__erlab_missing_colormap__"

    combo = ColorMapComboBox()
    qtbot.addWidget(combo)
    combo.setDefaultCmap(missing)
    combo.show()

    qtbot.wait_until(lambda: combo.count() > 0, timeout=2000)
    combo.load_all()

    assert combo.currentText() != missing


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
