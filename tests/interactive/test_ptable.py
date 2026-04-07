import html
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
import xraydb
from qtpy import QtCore, QtGui, QtTest, QtWidgets

import erlab
import erlab.interactive.ptable.__main__
import erlab.interactive.ptable._window
from erlab.interactive import ptable as launch_ptable
from erlab.interactive.ptable import PeriodicTableWindow
from erlab.interactive.ptable._inspector import CompactElementChip, RichTextHeaderView
from erlab.interactive.ptable._metadata import (
    CATEGORY_COLORS,
    CATEGORY_REFERENCES,
    ELEMENT_CATEGORIES,
    ELEMENT_POSITIONS,
    GROUND_STATE_CONFIGURATIONS,
    configuration_to_html,
)
from erlab.interactive.ptable._plot import CrossSectionPlot, EdgeTickAxisItem
from erlab.interactive.ptable._shared import (
    _blend_colors,
    _chip_secondary_text_color,
    _format_mass,
)


def _show_window(qtbot, win: PeriodicTableWindow) -> None:
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()


@pytest.fixture(autouse=True)
def isolate_ptable_settings(monkeypatch, tmp_path) -> None:
    settings_path = tmp_path / "ptable-settings.ini"

    def _test_settings() -> QtCore.QSettings:
        return QtCore.QSettings(str(settings_path), QtCore.QSettings.Format.IniFormat)

    monkeypatch.setattr(
        erlab.interactive.ptable._window,
        "_get_ptable_settings",
        _test_settings,
    )


def _viewport_pos_for_table_child(
    win: PeriodicTableWindow, child: QtWidgets.QWidget
) -> QtCore.QPoint:
    local_pos = child.mapTo(win.periodic_table, child.rect().center())
    scene_pos = win.table_view._proxy.mapToScene(QtCore.QPointF(local_pos))
    return win.table_view.mapFromScene(scene_pos)


def _viewport_rect_for_table_child(
    win: PeriodicTableWindow, child: QtWidgets.QWidget
) -> QtCore.QRect:
    scene_rect = win.table_view._proxy.mapRectToScene(QtCore.QRectF(child.geometry()))
    return win.table_view.mapFromScene(scene_rect).boundingRect()


def _table_child_is_visible_in_viewport(
    win: PeriodicTableWindow, child: QtWidgets.QWidget
) -> bool:
    viewport_rect = win.table_view.viewport().rect().adjusted(1, 1, -1, -1)
    return viewport_rect.contains(_viewport_rect_for_table_child(win, child))


def _table_child_center_is_visible_in_viewport(
    win: PeriodicTableWindow, child: QtWidgets.QWidget
) -> bool:
    scene_rect = win.table_view._proxy.mapRectToScene(QtCore.QRectF(child.geometry()))
    view_center = win.table_view.mapFromScene(scene_rect.center())
    return win.table_view.viewport().rect().adjusted(1, 1, -1, -1).contains(view_center)


def _search_completion_texts(win: PeriodicTableWindow) -> list[str]:
    model = win.search_completer.model()
    assert model is not None
    return [str(model.index(row, 0).data()) for row in range(model.rowCount())]


def _card_display_matches_target(card) -> bool:
    return (
        card._style_animation.state() == QtCore.QAbstractAnimation.State.Stopped
        and card._display_fill_color == card._fill_color
        and card._display_border_color == card._border_color
        and abs(card._display_border_width - float(card._border_width)) < 1e-6
        and abs(card._display_lift_opacity - card._animation_target_style.lift_opacity)
        < 1e-6
    )


def _legend_entry_display_matches_target(entry) -> bool:
    return (
        entry._style_animation.state() == QtCore.QAbstractAnimation.State.Stopped
        and entry._display_label_color
        == entry.label.palette().color(QtGui.QPalette.ColorRole.WindowText)
        and entry._display_marker_color
        == entry.marker.palette().color(QtGui.QPalette.ColorRole.Window)
        and entry._display_label_color == entry._animation_target_style.label_color
        and entry._display_marker_color == entry._animation_target_style.marker_color
    )


def _hue_distance(left: QtGui.QColor, right: QtGui.QColor) -> int:
    raw_distance = abs(left.hslHue() - right.hslHue())
    return min(raw_distance, 360 - raw_distance)


def _cam02_distance(
    left: QtGui.QColor,
    right: QtGui.QColor,
    *,
    cvd_type: str | None = None,
    severity: int = 80,
) -> float:
    colorspacious = pytest.importorskip("colorspacious")
    left_rgb = np.array([left.redF(), left.greenF(), left.blueF()])
    right_rgb = np.array([right.redF(), right.greenF(), right.blueF()])
    if cvd_type is not None:
        cvd_space = {
            "name": "sRGB1+CVD",
            "cvd_type": cvd_type,
            "severity": severity,
        }
        left_rgb = np.clip(
            colorspacious.cspace_convert(left_rgb, "sRGB1", cvd_space), 0, 1
        )
        right_rgb = np.clip(
            colorspacious.cspace_convert(right_rgb, "sRGB1", cvd_space), 0, 1
        )
    left_cam = colorspacious.cspace_convert(left_rgb, "sRGB1", "CAM02-UCS")
    right_cam = colorspacious.cspace_convert(right_rgb, "sRGB1", "CAM02-UCS")
    return float(np.linalg.norm(left_cam - right_cam))


def _hover_sequence_between_cards(
    win: PeriodicTableWindow,
    start_card: QtWidgets.QWidget,
    end_card: QtWidgets.QWidget,
) -> list[str | None]:
    app = QtWidgets.QApplication.instance()
    assert app is not None

    start_pos = _viewport_pos_for_table_child(win, start_card)
    end_pos = _viewport_pos_for_table_child(win, end_card)
    if start_pos.x() != end_pos.x():
        raise AssertionError("Hover path helper expects vertically aligned cards")

    step = 1 if end_pos.y() >= start_pos.y() else -1
    seen: list[str | None] = []
    last: str | None | object = object()

    for y in range(start_pos.y(), end_pos.y() + step, step):
        QtTest.QTest.mouseMove(
            win.table_view.viewport(), QtCore.QPoint(start_pos.x(), y)
        )
        app.processEvents()
        current = win.current_record.symbol if win.current_record else None
        if current != last:
            seen.append(current)
            last = current

    return seen


def _hover_sequence_between_widgets(
    container: QtWidgets.QWidget,
    start_widget: QtWidgets.QWidget,
    end_widget: QtWidgets.QWidget,
    current_value: Callable[[], str | None],
) -> list[str | None]:
    app = QtWidgets.QApplication.instance()
    assert app is not None

    start_pos = start_widget.mapTo(container, start_widget.rect().center())
    end_pos = end_widget.mapTo(container, end_widget.rect().center())
    if start_pos.x() != end_pos.x():
        raise AssertionError("Hover path helper expects vertically aligned widgets")

    step = 1 if end_pos.y() >= start_pos.y() else -1
    seen: list[str | None] = []
    last: str | None | object = object()

    for y in range(start_pos.y(), end_pos.y() + step, step):
        QtTest.QTest.mouseMove(container, QtCore.QPoint(start_pos.x(), y))
        app.processEvents()
        current = current_value()
        if current != last:
            seen.append(current)
            last = current

    return seen


def _hover_sequence_from_widget_to_point(
    container: QtWidgets.QWidget,
    start_widget: QtWidgets.QWidget,
    end_pos: QtCore.QPoint,
    current_value: Callable[[], str | None],
) -> list[str | None]:
    app = QtWidgets.QApplication.instance()
    assert app is not None

    start_pos = start_widget.mapTo(container, start_widget.rect().center())
    if start_pos.x() != end_pos.x():
        raise AssertionError("Hover path helper expects a vertically aligned path")

    step = 1 if end_pos.y() >= start_pos.y() else -1
    seen: list[str | None] = []
    last: str | None | object = object()

    for y in range(start_pos.y(), end_pos.y() + step, step):
        QtTest.QTest.mouseMove(container, QtCore.QPoint(start_pos.x(), y))
        app.processEvents()
        current = current_value()
        if current != last:
            seen.append(current)
            last = current

    return seen


def _move_cross_section_hover(
    plot: CrossSectionPlot,
    *,
    photon_energy: float,
    sigma: float,
) -> None:
    view_box = plot.plot_item.getViewBox()
    scene_pos = view_box.mapViewToScene(
        QtCore.QPointF(np.log10(photon_energy), np.log10(sigma))
    )
    plot._handle_plot_hover((scene_pos,))
    QtWidgets.QApplication.processEvents()


def _fake_cross_sections(_symbol: str) -> dict[str, xr.DataArray]:
    hv = np.array([5.0, 10.0, 20.0, 100.0, 1000.0, 3000.0])
    return {
        "2p": xr.DataArray(
            np.array([0.004, 0.01, 0.2, 0.08, 0.01, 0.004]),
            coords={"hv": hv},
            dims=["hv"],
        ),
        "3d": xr.DataArray(
            np.array([0.006, 0.02, 0.5, 0.2, 0.02, 0.006]),
            coords={"hv": hv},
            dims=["hv"],
        ),
    }


def _many_cross_sections(_symbol: str) -> dict[str, xr.DataArray]:
    hv = np.array([10.0, 20.0, 100.0, 1000.0])
    labels = ("3s", "3p", "3d", "4s", "4p", "4d", "4f", "5p")
    return {
        label: xr.DataArray(
            np.linspace(0.02 + index * 0.01, 0.2 + index * 0.01, hv.size),
            coords={"hv": hv},
            dims=["hv"],
        )
        for index, label in enumerate(labels)
    }


def _many_total_cross_section(_symbol: str) -> xr.DataArray:
    hv = np.array([10.0, 20.0, 100.0, 1000.0])
    return xr.DataArray(
        np.array([1.2, 0.9, 0.6, 0.3]),
        coords={"hv": hv},
        dims=["hv"],
    )


def _preview_persistence_cross_sections(symbol: str) -> dict[str, xr.DataArray]:
    hv = np.array([10.0, 20.0, 100.0, 1000.0])
    values_by_symbol = {
        "Au": {
            "2p": np.array([0.04, 0.08, 0.12, 0.16]),
            "3d": np.array([0.06, 0.1, 0.14, 0.18]),
        },
        "H": {
            "2p": np.array([0.03, 0.06, 0.1, 0.13]),
            "4s": np.array([0.02, 0.05, 0.08, 0.11]),
        },
    }
    values = values_by_symbol.get(symbol, values_by_symbol["Au"])
    return {
        label: xr.DataArray(series, coords={"hv": hv}, dims=["hv"])
        for label, series in values.items()
    }


def _preview_persistence_total_cross_section(symbol: str) -> xr.DataArray:
    hv = np.array([10.0, 20.0, 100.0, 1000.0])
    values_by_symbol = {
        "Au": np.array([0.7, 0.55, 0.4, 0.25]),
        "H": np.array([0.45, 0.35, 0.24, 0.15]),
    }
    return xr.DataArray(
        values_by_symbol.get(symbol, values_by_symbol["Au"]),
        coords={"hv": hv},
        dims=["hv"],
    )


def _mismatched_cross_sections(_symbol: str) -> dict[str, xr.DataArray]:
    return {
        "2p": xr.DataArray(
            np.array([0.12, 0.2, 0.18, 0.1]),
            coords={"hv": np.array([10.0, 20.0, 60.0, 200.0])},
            dims=["hv"],
        ),
        "3d": xr.DataArray(
            np.array([0.24, 0.28, 0.21, 0.11]),
            coords={"hv": np.array([15.0, 30.0, 90.0, 300.0])},
            dims=["hv"],
        ),
    }


def _mismatched_total_cross_section(_symbol: str) -> xr.DataArray:
    return xr.DataArray(
        np.array([1.8, 1.2, 0.8, 0.5]),
        coords={"hv": np.array([20.0, 40.0, 80.0, 160.0])},
        dims=["hv"],
    )


def _dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#111827"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#f9fafb"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#0f172a"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#1f2937"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#f9fafb"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#1f2937"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#f9fafb"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#60a5fa"))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#94a3b8"))
    palette.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor("#4b5563"))
    return palette


def _light_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#f8fafc"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#0f172a"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#0f172a"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#0f172a"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#2563eb"))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#64748b"))
    palette.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor("#94a3b8"))
    return palette


def test_category_colors_keep_lanthanoids_and_actinoids_related() -> None:
    lanthanoid = QtGui.QColor(CATEGORY_COLORS["lanthanoid"])
    actinoid = QtGui.QColor(CATEGORY_COLORS["actinoid"])

    assert _hue_distance(lanthanoid, actinoid) <= 8
    assert abs(lanthanoid.lightness() - actinoid.lightness()) >= 18
    assert 10.0 <= _cam02_distance(lanthanoid, actinoid) <= 16.0
    assert _cam02_distance(lanthanoid, actinoid, cvd_type="deuteranomaly") >= 12.0
    assert _cam02_distance(lanthanoid, actinoid, cvd_type="protanomaly") >= 11.0
    assert _cam02_distance(lanthanoid, actinoid, cvd_type="tritanomaly") >= 13.0


@pytest.mark.parametrize(
    ("left_category", "right_category", "minimum_distances"),
    [
        (
            "other_metal",
            "noble_gas",
            {
                "normal": 20.0,
                "deuteranomaly": 17.0,
                "protanomaly": 20.0,
                "tritanomaly": 20.0,
            },
        ),
        (
            "transition_metal",
            "halogen",
            {
                "normal": 20.0,
                "deuteranomaly": 10.0,
                "protanomaly": 10.0,
                "tritanomaly": 20.0,
            },
        ),
        (
            "metalloid",
            "noble_gas",
            {
                "normal": 15.0,
                "deuteranomaly": 10.0,
                "protanomaly": 9.0,
                "tritanomaly": 15.0,
            },
        ),
        (
            "nonmetal",
            "actinoid",
            {
                "normal": 20.0,
                "deuteranomaly": 10.0,
                "protanomaly": 10.0,
                "tritanomaly": 35.0,
            },
        ),
    ],
)
def test_category_colors_preserve_accessible_spacing(
    left_category: str,
    right_category: str,
    minimum_distances: dict[str, float],
) -> None:
    left = QtGui.QColor(CATEGORY_COLORS[left_category])
    right = QtGui.QColor(CATEGORY_COLORS[right_category])

    assert _cam02_distance(left, right) >= minimum_distances["normal"]
    assert (
        _cam02_distance(left, right, cvd_type="deuteranomaly")
        >= minimum_distances["deuteranomaly"]
    )
    assert (
        _cam02_distance(left, right, cvd_type="protanomaly")
        >= minimum_distances["protanomaly"]
    )
    assert (
        _cam02_distance(left, right, cvd_type="tritanomaly")
        >= minimum_distances["tritanomaly"]
    )


def test_element_categories_follow_requested_display_classification() -> None:
    assert ELEMENT_CATEGORIES[1] == "nonmetal"
    assert ELEMENT_CATEGORIES[79] == "transition_metal"
    assert all(
        ELEMENT_CATEGORIES[number] == "other_metal"
        for number in (13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 116)
    )
    assert all(
        ELEMENT_CATEGORIES[number] == "metalloid" for number in (5, 14, 32, 33, 51, 52)
    )
    assert all(
        ELEMENT_CATEGORIES[number] == "nonmetal" for number in (1, 6, 7, 8, 15, 16, 34)
    )
    assert all(ELEMENT_CATEGORIES[number] == "lanthanoid" for number in range(57, 72))
    assert all(ELEMENT_CATEGORIES[number] == "actinoid" for number in range(89, 104))


def test_ptable_launcher_and_search_highlight(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = launch_ptable(execute=False)
    _show_window(qtbot, win)

    assert len(win.periodic_table.cards) == 118
    assert win.splitter.orientation() == QtCore.Qt.Orientation.Vertical
    top_layout = win.top_panel.layout()
    assert top_layout is not None
    assert top_layout.contentsMargins().left() == 0
    assert top_layout.contentsMargins().top() == 0
    assert top_layout.spacing() == 0
    assert top_layout.indexOf(win.inspector.table_container) == 0
    assert top_layout.indexOf(win.inspector.side_panel) == 1
    assert len(win.category_legend.entry_labels) == 10
    assert len(win.category_legend.marker_frames) == 10
    assert "Noble gases" in "".join(
        label.text() for label in win.category_legend.entry_labels
    )
    table_panel = win.findChild(QtWidgets.QFrame, "ptable-table-panel")
    assert table_panel is not None
    assert table_panel.styleSheet() == ""
    assert table_panel.autoFillBackground() is True
    assert (
        table_panel.palette().color(QtGui.QPalette.ColorRole.Window)
        == win._theme.table_surface
    )
    table_layout = table_panel.layout()
    assert table_layout is not None
    assert table_layout.contentsMargins().left() == 0
    assert table_layout.contentsMargins().top() == 0
    root_layout = win.central.layout()
    assert root_layout is not None
    assert root_layout.contentsMargins().left() <= 8
    assert root_layout.contentsMargins().top() <= 8
    assert root_layout.spacing() <= 8
    empty_summary_height = win.inspector.summary_frame.height()
    table_container_layout = win.inspector.table_container.layout()
    assert table_container_layout is not None
    assert table_container_layout.indexOf(win.table_panel) == 0
    assert win.table_view.styleSheet() == ""
    assert win.table_view.backgroundBrush().color() == win._theme.table_surface
    assert win.periodic_table.autoFillBackground() is True
    assert (
        win.periodic_table.palette().color(QtGui.QPalette.ColorRole.Window)
        == win._theme.table_surface
    )
    assert win._theme.table_surface.lightness() > win._theme.window.lightness()
    assert win.category_legend.parent() is win.periodic_table
    periodic_table_layout = win.periodic_table.layout()
    assert periodic_table_layout is not None
    assert periodic_table_layout.horizontalSpacing() == 0
    assert periodic_table_layout.verticalSpacing() == 0
    group1_index = periodic_table_layout.indexOf(win.periodic_table.group_labels[0])
    group2_index = periodic_table_layout.indexOf(win.periodic_table.group_labels[1])
    group3_index = periodic_table_layout.indexOf(win.periodic_table.group_labels[2])
    group13_index = periodic_table_layout.indexOf(win.periodic_table.group_labels[12])
    assert periodic_table_layout.getItemPosition(group1_index)[:2] == (0, 1)
    assert periodic_table_layout.getItemPosition(group2_index)[:2] == (1, 2)
    assert periodic_table_layout.getItemPosition(group3_index)[:2] == (3, 3)
    assert periodic_table_layout.getItemPosition(group13_index)[:2] == (1, 13)
    assert periodic_table_layout.cellRect(2, 3).height() == (
        periodic_table_layout.cellRect(3, 3).height()
    )
    assert periodic_table_layout.cellRect(3, 3).height() == (
        periodic_table_layout.cellRect(4, 3).height()
    )
    assert (
        win.periodic_table.group_labels[0].alignment()
        & QtCore.Qt.AlignmentFlag.AlignBottom
    )
    assert set(win.category_legend.entries) == {
        "alkali_metal",
        "alkaline_earth_metal",
        "transition_metal",
        "other_metal",
        "metalloid",
        "nonmetal",
        "halogen",
        "noble_gas",
        "lanthanoid",
        "actinoid",
    }
    assert win.periodic_table.group_labels[0].font().pointSizeF() >= 18.0
    assert win.periodic_table.period_labels[0].font().pointSizeF() >= 18.0
    assert win.category_legend.entry_labels[0].font().pointSizeF() >= 13.0
    assert win.category_legend.entry_labels[0].text() == "Alkali metals"
    assert win.category_legend.entry_labels[0].palette().color(
        QtGui.QPalette.ColorRole.WindowText
    ) == _blend_colors(win._theme.muted_text, win._theme.text, 0.74)
    assert win.category_legend.entry_labels[0].font().weight() >= int(
        QtGui.QFont.Weight.ExtraBold
    )
    assert (
        win.category_legend.marker_frames[0].width()
        == win.category_legend.entries["alkali_metal"].width()
    )
    assert (
        win.category_legend.marker_frames[0].frameShape()
        == QtWidgets.QFrame.Shape.NoFrame
    )
    legend_layout = win.category_legend.layout()
    assert legend_layout is not None
    assert legend_layout.horizontalSpacing() >= 1
    assert legend_layout.verticalSpacing() >= 1
    assert (
        win.category_legend.geometry().top()
        < periodic_table_layout.cellRect(2, 3).top()
    )
    assert legend_layout.itemAtPosition(0, 0) is not None
    assert legend_layout.itemAtPosition(0, 2) is not None
    assert legend_layout.itemAtPosition(1, 2) is not None
    assert legend_layout.itemAtPosition(0, 5) is not None
    assert legend_layout.itemAtPosition(0, 9) is not None
    assert legend_layout.itemAtPosition(0, 11) is not None
    assert legend_layout.itemAtPosition(0, 12) is not None
    assert win.category_legend.reference_dialog.isVisible() is False

    noble_gas_entry = win.category_legend.entries["noble_gas"]
    assert noble_gas_entry.layout() is None
    assert noble_gas_entry.label.parent() is noble_gas_entry
    assert noble_gas_entry.marker.parent() is noble_gas_entry
    assert noble_gas_entry.height() >= noble_gas_entry.marker.height()
    noble_gas_fill = noble_gas_entry.marker.palette().color(
        QtGui.QPalette.ColorRole.Window
    )
    assert (
        noble_gas_entry.marker.palette().color(QtGui.QPalette.ColorRole.Window)
        == noble_gas_fill
    )
    noble_gas_entry.hovered.emit("noble_gas")
    assert win.periodic_table.cards[2].is_legend_match is True
    assert win.periodic_table.cards[10].is_legend_match is True
    assert win.periodic_table.cards[1].is_legend_match is False
    assert noble_gas_entry.is_active is True
    qtbot.waitUntil(
        lambda: (
            noble_gas_entry.marker.palette().color(QtGui.QPalette.ColorRole.Window)
            != noble_gas_fill
        )
    )

    noble_gas_entry.unhovered.emit("noble_gas")
    assert win.periodic_table.cards[2].is_legend_match is False
    assert win.periodic_table.cards[10].is_legend_match is False
    assert noble_gas_entry.is_active is False
    qtbot.waitUntil(
        lambda: (
            noble_gas_entry.marker.palette().color(QtGui.QPalette.ColorRole.Window)
            == noble_gas_fill
        )
    )

    hydrogen = win.periodic_table.cards[1]
    hydrogen_layout = hydrogen.layout()
    assert hydrogen_layout is not None
    assert hydrogen_layout.spacing() == 0
    assert hydrogen_layout.contentsMargins().top() == 4
    assert hydrogen_layout.contentsMargins().left() == 6
    assert hydrogen_layout.contentsMargins().bottom() == 4
    assert hydrogen_layout.itemAt(0).layout() is not None
    assert hydrogen_layout.itemAt(1).spacerItem() is not None
    assert hydrogen_layout.itemAt(hydrogen_layout.count() - 1).spacerItem() is not None
    assert hydrogen_layout.stretch(1) == 1
    assert hydrogen_layout.stretch(hydrogen_layout.count() - 1) == 1
    assert hydrogen.width() == hydrogen.height()
    assert hydrogen.width() == 120
    assert hydrogen.symbol_label.text() == "H"
    assert hydrogen.atomic_number_label.text() == "1"
    hydrogen_base_fill = QtGui.QColor(
        CATEGORY_COLORS[ELEMENT_CATEGORIES[hydrogen.record.atomic_number]]
    )
    hydrogen_previous_fill = _blend_colors(
        hydrogen_base_fill, win._theme.table_surface, 0.16
    )
    assert hydrogen._fill_color != hydrogen_base_fill
    assert hydrogen._fill_color.saturation() > hydrogen_previous_fill.saturation()
    assert hydrogen.atomic_number_label.alignment() & QtCore.Qt.AlignmentFlag.AlignLeft
    assert hydrogen.atomic_number_label.alignment() & QtCore.Qt.AlignmentFlag.AlignTop
    assert hydrogen.symbol_label.alignment() & QtCore.Qt.AlignmentFlag.AlignRight
    assert hydrogen.symbol_label.alignment() & QtCore.Qt.AlignmentFlag.AlignTop
    assert (
        hydrogen.atomic_number_label.x() <= hydrogen_layout.contentsMargins().left() + 4
    )
    assert (
        hydrogen.atomic_number_label.y() <= hydrogen_layout.contentsMargins().top() + 4
    )
    assert hydrogen.atomic_number_label.font().pointSizeF() >= 13.5
    assert hydrogen.symbol_label.font().pointSizeF() >= 38.0
    assert hydrogen._CORNER_RADIUS == 0.0
    assert hydrogen._draw_border is False
    assert hydrogen._border_width == 0
    assert hydrogen.name_label.font().pointSizeF() >= 12.6
    assert hydrogen.mass_label.font().pointSizeF() >= 12.6
    assert hydrogen.config_label.font().pointSizeF() >= 10.4
    assert hydrogen.config_label.wordWrap() is False
    assert hydrogen.symbol_label.palette().color(
        QtGui.QPalette.ColorRole.WindowText
    ) == hydrogen.name_label.palette().color(QtGui.QPalette.ColorRole.WindowText)
    assert hydrogen.mass_label.palette().color(
        QtGui.QPalette.ColorRole.WindowText
    ) == hydrogen.config_label.palette().color(QtGui.QPalette.ColorRole.WindowText)
    assert hydrogen.atomic_number_label.palette().color(
        QtGui.QPalette.ColorRole.WindowText
    ) != hydrogen.symbol_label.palette().color(QtGui.QPalette.ColorRole.WindowText)
    assert hydrogen.mass_label.palette().color(
        QtGui.QPalette.ColorRole.WindowText
    ) != hydrogen.symbol_label.palette().color(QtGui.QPalette.ColorRole.WindowText)
    assert hydrogen.name_label.text() == xraydb.atomic_name(1).title()
    assert hydrogen.mass_label.text() == _format_mass(xraydb.atomic_mass(1))
    assert "<sup>1</sup>" in hydrogen.config_label.text()
    tooltip_html = hydrogen.toolTip()
    secondary_text = _chip_secondary_text_color(win._theme).name()
    assert tooltip_html.startswith("<qt>")
    assert "<b>Hydrogen (H)</b>" in tooltip_html
    assert f"color: {win._theme.text.name()};" in tooltip_html
    assert f"color: {secondary_text};" in tooltip_html
    assert "<b>Category:</b></td><td" in tooltip_html
    assert "<b>Atomic number:</b></td><td" in tooltip_html
    assert (
        f"<b>Atomic mass:</b></td><td style='color: {win._theme.text.name()};'>"
        f"&nbsp;{_format_mass(xraydb.atomic_mass(1))} u" in tooltip_html
    )
    assert (
        "<b>Electron configuration:</b></td><td style='color: "
        f"{win._theme.text.name()};'>&nbsp;1<i>s</i><sup>1</sup>" in tooltip_html
    )
    mercury = win.periodic_table.cards[80]
    mercury_layout = mercury.layout()
    assert mercury_layout is not None
    mercury_available_width = (
        mercury.width()
        - mercury_layout.contentsMargins().left()
        - mercury_layout.contentsMargins().right()
    )
    assert mercury.config_label.sizeHint().width() <= mercury_available_width
    assert (
        win.inspector.summary_cards_grid.horizontalSpacing()
        == win.inspector.summary_cards_grid.verticalSpacing()
    )
    assert win.inspector.summary_cards_grid.horizontalSpacing() == 6
    assert win.inspector.summary_cards_grid.verticalSpacing() == 6

    gold_fill = QtGui.QColor(CATEGORY_COLORS[ELEMENT_CATEGORIES[79]]).name()

    assert win.selected_atomic_number is None
    assert win.selected_atomic_numbers == ()
    assert win.current_record is None
    assert (
        win.inspector.summary_stack.currentWidget() is win.inspector.summary_empty_page
    )
    assert empty_summary_height < win.inspector._summary_max_fixed_height
    assert win.inspector.mode_label.text() == "No selection"
    assert win.inspector.copy_values_button.isEnabled() is False
    assert win.inspector.copy_table_button.isEnabled() is False
    win.search_edit.setText("gold")

    assert win.selected_atomic_number is None
    assert win.selected_atomic_numbers == ()
    assert win.periodic_table.cards[79].is_search_match
    assert win.periodic_table.cards[79]._border_width == 5
    assert win.periodic_table.cards[79]._border_color == win._theme.search_accent
    assert win.periodic_table.cards[79]._fill_color.name() != gold_fill
    assert _search_completion_texts(win)[0] == "Au - Gold"
    assert (
        win.inspector.summary_stack.currentWidget() is win.inspector.summary_empty_page
    )
    assert "Select an element" in win.inspector.summary_empty_label.text()
    assert win.inspector.minimumHeight() >= 300
    assert (
        win.inspector.copy_values_button.palette().color(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.ButtonText,
        )
        == win._theme.disabled_text
    )

    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)
    assert win.inspector.summary_frame.height() == empty_summary_height
    chip_previous_fill = _blend_colors(hydrogen_base_fill, win._theme.panel, 0.16)
    assert (
        hydrogen.atomic_number_label.palette()
        .color(QtGui.QPalette.ColorRole.WindowText)
        .name()
        == win.inspector._summary_cards[0]
        .atomic_number_label.palette()
        .color(QtGui.QPalette.ColorRole.WindowText)
        .name()
    )
    assert (
        hydrogen.mass_label.palette().color(QtGui.QPalette.ColorRole.WindowText).name()
        == win.inspector._summary_cards[0]
        .mass_label.palette()
        .color(QtGui.QPalette.ColorRole.WindowText)
        .name()
    )
    assert (
        hydrogen.config_label.palette()
        .color(QtGui.QPalette.ColorRole.WindowText)
        .name()
        == win.inspector._summary_cards[0]
        .config_label.palette()
        .color(QtGui.QPalette.ColorRole.WindowText)
        .name()
    )
    assert win.inspector._summary_cards[0].styleSheet() == ""
    assert win.inspector._summary_cards[0]._border_width == 0
    assert win.inspector._summary_cards[0]._border_color.alpha() == 0
    assert (
        win.inspector._summary_cards[0]._fill_color.saturation()
        > chip_previous_fill.saturation()
    )
    win._clear_selection()
    assert (
        win.inspector.levels_table.verticalHeader()
        .palette()
        .color(QtGui.QPalette.ColorRole.ButtonText)
        == win._theme.text
    )
    header_layout = win.header.layout()
    inspector_layout = win.inspector.layout()
    side_layout = win.inspector.side_panel.layout()
    bottom_layout = win.inspector.bottom_panel.layout()
    levels_layout = win.inspector.levels_frame.layout()
    levels_header_layout = win.inspector._levels_header_layout
    levels_controls_layout = win.inspector.levels_controls_frame.layout()
    assert header_layout is not None
    assert inspector_layout is not None
    assert side_layout is not None
    assert bottom_layout is not None
    assert levels_layout is not None
    assert levels_header_layout is not None
    assert levels_controls_layout is not None
    assert inspector_layout.contentsMargins().left() == 0
    assert inspector_layout.contentsMargins().top() == 0
    assert inspector_layout.spacing() == 0
    assert inspector_layout.indexOf(win.inspector.splitter) == 0
    assert header_layout.indexOf(win.search_edit) >= 0
    assert header_layout.indexOf(win.notation_frame) >= 0
    assert header_layout.indexOf(win.search_edit) < header_layout.indexOf(
        win.notation_frame
    )
    assert header_layout.indexOf(win.photon_energy_edit) == -1
    assert header_layout.indexOf(win.workfunction_edit) == -1
    assert header_layout.indexOf(win.harmonic_frame) == -1
    assert side_layout.contentsMargins().left() == 0
    assert side_layout.contentsMargins().top() == 0
    assert side_layout.spacing() <= 10
    assert bottom_layout.contentsMargins().left() == 0
    assert bottom_layout.contentsMargins().top() == 0
    assert bottom_layout.spacing() == 0
    assert side_layout.indexOf(win.inspector.summary_frame) >= 0
    assert side_layout.indexOf(win.inspector.summary_levels_separator) >= 0
    assert side_layout.indexOf(win.inspector.plot_frame) >= 0
    assert bottom_layout.indexOf(win.inspector.levels_plot_separator) >= 0
    assert bottom_layout.indexOf(win.inspector.levels_frame) >= 0
    assert levels_layout.indexOf(win.inspector.levels_stack) >= 0
    assert side_layout.indexOf(win.inspector.summary_frame) < side_layout.indexOf(
        win.inspector.summary_levels_separator
    )
    assert side_layout.indexOf(
        win.inspector.summary_levels_separator
    ) < side_layout.indexOf(win.inspector.plot_frame)
    assert bottom_layout.indexOf(
        win.inspector.levels_plot_separator
    ) < bottom_layout.indexOf(win.inspector.levels_frame)
    assert levels_header_layout.indexOf(win.inspector.levels_title) >= 0
    assert levels_header_layout.indexOf(win.inspector.levels_controls_frame) >= 0
    assert levels_header_layout.indexOf(win.inspector.copy_values_button) >= 0
    assert levels_header_layout.indexOf(win.inspector.copy_table_button) >= 0
    assert levels_header_layout.indexOf(
        win.inspector.levels_title
    ) < levels_header_layout.indexOf(win.inspector.levels_controls_frame)
    assert levels_header_layout.indexOf(
        win.inspector.levels_controls_frame
    ) < levels_header_layout.indexOf(win.inspector.copy_values_button)
    assert levels_header_layout.indexOf(
        win.inspector.copy_values_button
    ) < levels_header_layout.indexOf(win.inspector.copy_table_button)
    assert win.photon_energy_label.parent() is win.inspector.levels_controls_frame
    assert win.photon_energy_edit.parent() is win.inspector.levels_controls_frame
    assert win.workfunction_label.parent() is win.inspector.levels_controls_frame
    assert win.workfunction_edit.parent() is win.inspector.levels_controls_frame
    assert win.harmonic_frame.parent() is win.inspector.levels_controls_frame
    assert (
        win.inspector.levels_controls_frame.geometry().left()
        - win.inspector.levels_title.geometry().right()
        >= 8
    )
    assert (
        win.inspector.copy_values_button.geometry().left()
        > win.inspector.levels_controls_frame.geometry().right()
    )
    assert levels_controls_layout.indexOf(win.photon_energy_label) >= 0
    assert levels_controls_layout.indexOf(
        win.photon_energy_label
    ) < levels_controls_layout.indexOf(win.photon_energy_edit)
    assert levels_controls_layout.indexOf(win.photon_energy_edit) >= 0
    assert levels_controls_layout.indexOf(win.workfunction_label) >= 0
    assert levels_controls_layout.indexOf(
        win.photon_energy_edit
    ) < levels_controls_layout.indexOf(win.workfunction_label)
    assert levels_controls_layout.indexOf(
        win.workfunction_label
    ) < levels_controls_layout.indexOf(win.workfunction_edit)
    assert levels_controls_layout.indexOf(win.workfunction_edit) >= 0
    assert levels_controls_layout.indexOf(win.harmonic_frame) >= 0
    assert levels_controls_layout.indexOf(
        win.workfunction_edit
    ) < levels_controls_layout.indexOf(win.harmonic_frame)
    assert getattr(win.photon_energy_edit, "suffix", None) == "eV"
    assert getattr(win.workfunction_edit, "suffix", None) == "eV"
    for frame in (
        win.inspector.side_panel,
        win.inspector.bottom_panel,
        win.inspector.summary_frame,
        win.inspector.levels_frame,
        win.inspector.levels_controls_frame,
        win.inspector.plot_frame,
    ):
        frame_layout = frame.layout()
        assert frame_layout is not None
        assert frame_layout.contentsMargins().left() <= 10
        assert frame_layout.contentsMargins().top() <= 10
        assert frame.frameShape() == QtWidgets.QFrame.Shape.NoFrame
    for separator in (
        win.inspector.summary_levels_separator,
        win.inspector.levels_plot_separator,
    ):
        assert separator.frameShape() == QtWidgets.QFrame.Shape.NoFrame
        assert separator.minimumHeight() == 1
        assert separator.maximumHeight() == 1
        assert separator.autoFillBackground() is True
        assert (
            separator.palette().color(QtGui.QPalette.ColorRole.Window)
            == win._theme.border_soft
        )
    assert win.inspector.plot_frame.minimumWidth() == 320
    assert win.inspector.plot_frame.maximumWidth() == 320
    assert (
        win.inspector.summary_frame.minimumHeight()
        == win.inspector.summary_frame.height()
    )
    assert (
        win.inspector.summary_frame.maximumHeight()
        == win.inspector.summary_frame.height()
    )
    assert win.inspector.cross_section_plot.minimumWidth() == 300
    assert win.inspector.cross_section_plot.maximumWidth() == 300
    assert (
        win.inspector.cross_section_plot.plot_widget.minimumHeight()
        == CrossSectionPlot._PLOT_WIDGET_MIN_HEIGHT
    )
    assert win.inspector.plot_frame.minimumHeight() > 0
    assert win.minimumHeight() == win.inspector.splitter.minimumSizeHint().height()

    win.close()


def test_ptable_dark_mode_theme(
    qtbot,
    monkeypatch,
) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None

    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    original_palette = QtGui.QPalette(app.palette())
    app.setPalette(_dark_palette())
    try:
        win = PeriodicTableWindow()
        _show_window(qtbot, win)

        assert win._theme.is_dark is True
        assert win._theme.window.lightness() < 128
        assert win.header.styleSheet() == ""
        assert win.inspector.levels_table.styleSheet() == ""
        assert win.central.styleSheet() == ""
        assert (
            win.central.palette().color(QtGui.QPalette.ColorRole.Window)
            == win._theme.window
        )
        assert (
            win.inspector.palette().color(QtGui.QPalette.ColorRole.Window)
            == win._theme.panel
        )
        assert (
            win.periodic_table.palette().color(QtGui.QPalette.ColorRole.Window)
            == win._theme.table_surface
        )
        assert win._theme.table_surface.lightness() < win._theme.window.lightness()
        assert (
            win.inspector.levels_table.palette().color(QtGui.QPalette.ColorRole.Base)
            == win._theme.panel
        )
        assert (
            win.inspector.cross_section_plot.plot_widget.backgroundBrush()
            .color()
            .lightness()
            < 128
        )
        hydrogen = win.periodic_table.cards[1]
        hydrogen_base_fill = QtGui.QColor(
            CATEGORY_COLORS[ELEMENT_CATEGORIES[hydrogen.record.atomic_number]]
        )
        hydrogen_previous_fill = _blend_colors(
            hydrogen_base_fill, win._theme.table_surface, 0.28
        )
        assert (
            hydrogen.symbol_label.palette().color(QtGui.QPalette.ColorRole.WindowText)
            == win._theme.text
        )
        assert (
            hydrogen.name_label.palette().color(QtGui.QPalette.ColorRole.WindowText)
            == win._theme.text
        )
        atomic_color = hydrogen.atomic_number_label.palette().color(
            QtGui.QPalette.ColorRole.WindowText
        )
        mass_color = hydrogen.mass_label.palette().color(
            QtGui.QPalette.ColorRole.WindowText
        )
        config_color = hydrogen.config_label.palette().color(
            QtGui.QPalette.ColorRole.WindowText
        )
        assert atomic_color != win._theme.text
        assert mass_color == config_color
        assert hydrogen._fill_color != hydrogen_base_fill
        assert hydrogen._fill_color.saturation() > hydrogen_previous_fill.saturation()
        win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)
        chip_previous_fill = _blend_colors(hydrogen_base_fill, win._theme.panel, 0.28)
        assert len(win.inspector._summary_cards) == 1
        assert win.inspector._summary_cards[0].styleSheet() == ""
        assert win.inspector._summary_cards[0]._border_width == 0
        assert win.inspector._summary_cards[0]._border_color.alpha() == 0
        assert (
            hydrogen.atomic_number_label.palette()
            .color(QtGui.QPalette.ColorRole.WindowText)
            .name()
            == win.inspector._summary_cards[0]
            .atomic_number_label.palette()
            .color(QtGui.QPalette.ColorRole.WindowText)
            .name()
        )
        assert (
            hydrogen.mass_label.palette()
            .color(QtGui.QPalette.ColorRole.WindowText)
            .name()
            == win.inspector._summary_cards[0]
            .mass_label.palette()
            .color(QtGui.QPalette.ColorRole.WindowText)
            .name()
        )
        assert (
            hydrogen.config_label.palette()
            .color(QtGui.QPalette.ColorRole.WindowText)
            .name()
            == win.inspector._summary_cards[0]
            .config_label.palette()
            .color(QtGui.QPalette.ColorRole.WindowText)
            .name()
        )
        assert (
            win.inspector._summary_cards[0]._fill_color.saturation()
            > chip_previous_fill.saturation()
        )

        win.close()
    finally:
        app.setPalette(original_palette)


def test_ptable_category_legend_dialog_rethemes_live_on_palette_change(
    qtbot,
    monkeypatch,
) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None

    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    original_palette = QtGui.QPalette(app.palette())
    app.setPalette(_light_palette())
    try:
        win = PeriodicTableWindow()
        _show_window(qtbot, win)

        dialog = win.category_legend.reference_dialog
        qtbot.mouseClick(
            win.category_legend.entries["noble_gas"],
            QtCore.Qt.MouseButton.LeftButton,
        )
        qtbot.waitUntil(dialog.isVisible)

        light_bg = QtGui.QColor(dialog._background_color)
        light_text = dialog.title_label.palette().color(
            QtGui.QPalette.ColorRole.WindowText
        )
        assert win._theme.is_dark is False

        app.setPalette(_dark_palette())
        qtbot.waitUntil(lambda: win._theme.is_dark is True)
        qtbot.waitUntil(
            lambda: (
                dialog._background_color != light_bg
                and dialog.title_label.palette().color(
                    QtGui.QPalette.ColorRole.WindowText
                )
                != light_text
            )
        )

        dark_bg = QtGui.QColor(dialog._background_color)
        dark_text = dialog.title_label.palette().color(
            QtGui.QPalette.ColorRole.WindowText
        )
        assert dark_bg.lightness() < light_bg.lightness()
        assert dark_text == win._theme.text

        app.setPalette(_light_palette())
        qtbot.waitUntil(lambda: win._theme.is_dark is False)
        qtbot.waitUntil(
            lambda: (
                dialog._background_color != dark_bg
                and dialog.title_label.palette().color(
                    QtGui.QPalette.ColorRole.WindowText
                )
                != dark_text
            )
        )

        assert dialog._background_color.lightness() > dark_bg.lightness()
        assert (
            dialog.title_label.palette().color(QtGui.QPalette.ColorRole.WindowText)
            == win._theme.text
        )

        win.close()
    finally:
        app.setPalette(original_palette)


def test_ptable_hover_preview_and_click_lock(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    helium = win.periodic_table.cards[2]
    initial_height = win.inspector.height()
    initial_plot_width = win.inspector.plot_frame.width()
    helium.hovered.emit(2)

    assert win.current_record.symbol == "He"
    assert win.inspector.mode_label.text() == "Preview"
    assert (
        win.inspector.summary_stack.currentWidget() is win.inspector.summary_cards_page
    )
    assert len(win.inspector._summary_cards) == 1
    assert win.inspector._summary_cards[0].symbol_label.text() == "He"
    assert win.inspector.levels_table.horizontalHeader().isVisible() is True
    assert win.inspector.levels_table.rowCount() == 1
    assert win.inspector.levels_table.verticalHeaderItem(0).text() == "He"
    assert win.inspector.levels_table.horizontalHeaderItem(0).text() == "1s"
    assert win.inspector.levels_table.item(0, 0).text() == "10"
    assert win.inspector.plot_frame.width() == initial_plot_width
    assert win.inspector.cross_section_plot._last_state == ("He", "orbital", None, 1)

    helium.selected.emit(2, QtCore.Qt.KeyboardModifier.NoModifier)

    assert win.selected_atomic_number == 2
    assert win.selected_atomic_numbers == (2,)
    assert win.inspector.mode_label.text() == "Selected"
    assert win.periodic_table.cards[2].is_selected
    assert len(win.inspector._summary_cards) == 1
    assert win.inspector._summary_cards[0].symbol_label.text() == "He"
    assert win.inspector.height() == initial_height
    assert win.inspector.plot_frame.width() == initial_plot_width
    assert win.inspector.cross_section_plot._last_state == ("He", "orbital", None, 1)

    win._handle_card_unhovered(2)

    assert win.current_record.symbol == "He"
    assert win.inspector.mode_label.text() == "Selected"
    assert win.inspector.height() == initial_height
    assert win.inspector.plot_frame.width() == initial_plot_width

    win.close()


def test_ptable_card_style_animation_tracks_target_and_settles(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_binding_energy",
        lambda _symbol: {"1s": 10.0},
        raising=False,
    )
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    hydrogen = win.periodic_table.cards[1]
    assert _card_display_matches_target(hydrogen)

    previous_fill = QtGui.QColor(hydrogen._display_fill_color)
    previous_border = QtGui.QColor(hydrogen._display_border_color)
    previous_border_width = hydrogen._display_border_width

    win.periodic_table.set_hovered_atomic_number(1)

    assert hydrogen.is_hovered is True
    assert hydrogen._border_color == win._theme.hover_accent
    assert hydrogen._border_width == 2
    assert hydrogen._display_lift_opacity == 0.0

    qtbot.waitUntil(
        lambda: (
            hydrogen._display_fill_color != previous_fill
            or hydrogen._display_border_color != previous_border
            or abs(hydrogen._display_border_width - previous_border_width) > 1e-6
            or hydrogen._display_lift_opacity > 0.0
        )
    )
    qtbot.waitUntil(lambda: _card_display_matches_target(hydrogen))

    assert hydrogen._display_fill_color == hydrogen._fill_color
    assert hydrogen._display_border_color == hydrogen._border_color
    assert abs(hydrogen._display_border_width - float(hydrogen._border_width)) < 1e-6
    assert hydrogen._display_lift_opacity > 0.0

    win.close()


def test_ptable_card_style_animation_retargets_on_theme_change(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_binding_energy",
        lambda _symbol: {"1s": 10.0},
        raising=False,
    )
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    hydrogen = win.periodic_table.cards[1]
    previous_display_border = QtGui.QColor(hydrogen._display_border_color)

    hydrogen.set_hover_state(True)

    assert hydrogen.is_hovered is True
    assert hydrogen._border_color == win._theme.hover_accent
    qtbot.waitUntil(lambda: hydrogen._display_border_color != previous_display_border)

    display_before_theme_change = QtGui.QColor(hydrogen._display_border_color)
    updated_theme = replace(
        win._theme,
        accent=QtGui.QColor("#0284c7"),
        hover_accent=QtGui.QColor("#e11d48"),
        search_accent=QtGui.QColor("#f97316"),
        border=QtGui.QColor("#475569"),
    )

    hydrogen.apply_theme(updated_theme)

    assert hydrogen._border_color == updated_theme.hover_accent
    assert hydrogen._display_border_color == display_before_theme_change
    qtbot.waitUntil(lambda: _card_display_matches_target(hydrogen))
    assert hydrogen._display_border_color == updated_theme.hover_accent

    win.close()


def test_ptable_summary_panel_does_not_grow_when_window_gets_taller(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win._handle_card_selected(2, QtCore.Qt.KeyboardModifier.NoModifier)

    initial_summary_height = win.inspector.summary_frame.height()
    initial_plot_height = win.inspector.plot_frame.height()

    win.resize(win.width(), win.height() + 240)
    QtWidgets.QApplication.processEvents()

    assert win.inspector.summary_frame.height() == initial_summary_height
    assert win.inspector.plot_frame.height() > initial_plot_height

    win.resize(win.width(), 650)
    QtWidgets.QApplication.processEvents()

    assert win.inspector.summary_frame.height() == initial_summary_height

    win.close()


def test_ptable_cross_section_plot_keeps_minimum_height_when_window_shrinks(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win._handle_card_selected(2, QtCore.Qt.KeyboardModifier.NoModifier)

    win.resize(win.width(), 400)
    QtWidgets.QApplication.processEvents()

    assert win.height() == win.minimumHeight()
    assert (
        win.inspector.cross_section_plot.plot_widget.height()
        >= CrossSectionPlot._PLOT_WIDGET_MIN_HEIGHT
    )
    assert (
        win.inspector.cross_section_plot.visibleRegion().boundingRect().height()
        >= win.inspector.cross_section_plot.plot_widget.height()
    )

    win.close()


def test_ptable_plot_legend_does_not_overlap_when_window_shrinks(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _many_cross_sections)
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _many_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.processEvents()

    plot = win.inspector.cross_section_plot
    assert plot.legend_label.isVisible()
    assert plot.minimumHeight() > plot.plot_widget.minimumHeight()

    win.resize(win.width(), 400)
    QtWidgets.QApplication.processEvents()

    assert win.height() == win.minimumHeight()
    assert (
        plot.minimumHeight()
        >= plot.plot_widget.minimumHeight() + plot.legend_label.height()
    )
    assert plot.legend_label.geometry().top() > plot._stack.geometry().bottom()

    win.close()


def test_ptable_vertical_hover_tracking_is_directionally_stable(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    downward = _hover_sequence_between_cards(
        win,
        win.periodic_table.cards[51],
        win.periodic_table.cards[115],
    )
    upward = _hover_sequence_between_cards(
        win,
        win.periodic_table.cards[115],
        win.periodic_table.cards[51],
    )

    assert downward == ["Sb", "Bi", "Mc"]
    assert upward == ["Mc", "Bi", "Sb"]

    win.close()


def test_ptable_category_legend_hover_tracking_is_directionally_stable(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    legend = win.category_legend
    downward = _hover_sequence_between_widgets(
        legend,
        legend.entries["lanthanoid"],
        legend.entries["actinoid"],
        lambda: legend._active_category,
    )
    upward = _hover_sequence_between_widgets(
        legend,
        legend.entries["actinoid"],
        legend.entries["lanthanoid"],
        lambda: legend._active_category,
    )

    assert downward == ["lanthanoid", None, "actinoid"]
    assert upward == ["actinoid", None, "lanthanoid"]

    win.close()


def test_ptable_category_legend_click_opens_and_retargets_reference_dialog(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    legend = win.category_legend
    dialog = legend.reference_dialog
    assert dialog is win.reference_dialog
    noble_gas = legend.entries["noble_gas"]
    halogen = legend.entries["halogen"]

    qtbot.mouseClick(noble_gas, QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(dialog.isVisible)

    assert legend._reference_category == "noble_gas"
    assert dialog.title_label.text() == CATEGORY_REFERENCES["noble_gas"].title
    assert dialog.body_label.text() == CATEGORY_REFERENCES["noble_gas"].blurb
    citation_html = html.unescape(dialog.citation_label.text())
    assert "<table" in citation_html
    assert "<ol" not in citation_html
    for reference in CATEGORY_REFERENCES["noble_gas"].references:
        assert html.unescape(reference.citation_html or "") in citation_html
        assert reference.url in citation_html
    assert "<i>Compendium of Chemical Terminology (the Gold Book)</i>" in citation_html
    assert dialog.parent() is win
    assert dialog.isModal() is True
    assert dialog.windowModality() == QtCore.Qt.WindowModality.WindowModal
    assert dialog._background_color.alpha() == 255
    assert dialog._background_color != QtCore.Qt.GlobalColor.transparent
    base_point_size = dialog.font().pointSizeF()
    assert base_point_size >= 15.0
    assert dialog.title_label.font().pointSizeF() == pytest.approx(base_point_size)
    assert dialog.body_label.font().pointSizeF() == pytest.approx(base_point_size)
    assert dialog.citation_label.font().pointSizeF() == pytest.approx(base_point_size)
    assert dialog.title_label.font().weight() >= QtGui.QFont.Weight.Bold
    assert noble_gas.is_active is True
    noble_gas_width = dialog.width()

    qtbot.mouseClick(halogen, QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: (
            dialog.isVisible()
            and dialog.title_label.text() == CATEGORY_REFERENCES["halogen"].title
        )
    )

    assert dialog.isVisible() is True
    assert legend._reference_category == "halogen"
    assert dialog.title_label.text() == CATEGORY_REFERENCES["halogen"].title
    assert dialog.width() == noble_gas_width
    assert halogen.is_active is True

    qtbot.mouseClick(halogen, QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not dialog.isVisible())
    assert legend._reference_category is None

    win.close()


def test_ptable_category_legend_reference_dialog_closes_on_escape(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    legend = win.category_legend
    dialog = legend.reference_dialog
    qtbot.mouseClick(legend.entries["noble_gas"], QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(dialog.isVisible)

    qtbot.mouseClick(win.search_edit, QtCore.Qt.MouseButton.LeftButton)
    assert dialog.isVisible() is True
    assert legend._reference_category == "noble_gas"

    qtbot.mouseClick(legend.entries["halogen"], QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(dialog.isVisible)

    QtTest.QTest.keyClick(dialog, QtCore.Qt.Key.Key_Escape)
    qtbot.waitUntil(lambda: not dialog.isVisible())
    assert legend._reference_category is None

    win.close()


def test_ptable_category_legend_reference_links_open_official_reference_urls(
    qtbot,
    monkeypatch,
) -> None:
    opened_urls: list[str] = []

    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)
    monkeypatch.setattr(
        QtGui.QDesktopServices,
        "openUrl",
        lambda url: opened_urls.append(url.toString()) or True,
    )

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    legend = win.category_legend
    dialog = legend.reference_dialog
    qtbot.mouseClick(
        legend.entries["transition_metal"], QtCore.Qt.MouseButton.LeftButton
    )
    qtbot.waitUntil(dialog.isVisible)

    target_url = CATEGORY_REFERENCES["transition_metal"].references[-1].url
    dialog.citation_label.linkActivated.emit(target_url)

    assert opened_urls == [target_url]
    qtbot.waitUntil(lambda: not dialog.isVisible())

    win.close()


def test_ptable_category_legend_click_does_not_change_selection_or_inspector_state(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    original_selection = win.selected_atomic_numbers
    original_current = win.current_record.symbol if win.current_record else None
    original_mode = win.inspector.mode_label.text()
    original_plot_state = win.inspector.cross_section_plot._last_state

    qtbot.mouseClick(
        win.category_legend.entries["noble_gas"],
        QtCore.Qt.MouseButton.LeftButton,
    )
    QtWidgets.QApplication.processEvents()

    assert win.selected_atomic_numbers == original_selection
    assert (
        win.current_record.symbol if win.current_record else None
    ) == original_current
    assert win.inspector.mode_label.text() == original_mode
    assert win.inspector.cross_section_plot._last_state == original_plot_state

    win.close()


def test_ptable_category_legend_hover_preview_still_works_with_reference_dialog(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    legend = win.category_legend
    qtbot.mouseClick(legend.entries["noble_gas"], QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(legend.reference_dialog.isVisible)

    legend.entries["halogen"].hovered.emit("halogen")
    QtWidgets.QApplication.processEvents()

    assert legend.reference_dialog.isVisible() is True
    assert legend.entries["noble_gas"].is_active is True
    assert legend.entries["halogen"].is_active is True
    assert win.periodic_table.cards[17].is_legend_match is True
    assert win.periodic_table.cards[2].is_legend_match is False

    legend.entries["halogen"].unhovered.emit("halogen")
    QtWidgets.QApplication.processEvents()

    assert legend.entries["noble_gas"].is_active is True
    assert legend.entries["halogen"].is_active is False
    assert win.periodic_table.cards[17].is_legend_match is False

    win.close()


def test_ptable_category_legend_entry_animation_tracks_active_state(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_binding_energy",
        lambda _symbol: {"1s": 10.0},
        raising=False,
    )
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    entry = win.category_legend.entries["noble_gas"]
    base_marker = entry.marker.palette().color(QtGui.QPalette.ColorRole.Window)
    base_label = entry.label.palette().color(QtGui.QPalette.ColorRole.WindowText)

    entry.hovered.emit("noble_gas")

    assert entry.is_active is True
    assert win.periodic_table.cards[2].is_legend_match is True
    qtbot.waitUntil(
        lambda: (
            entry._display_marker_color != base_marker
            or entry._display_label_color != base_label
        )
    )
    qtbot.waitUntil(lambda: _legend_entry_display_matches_target(entry))
    assert entry._display_marker_color != base_marker
    assert entry._display_label_color != base_label

    entry.unhovered.emit("noble_gas")

    assert entry.is_active is False
    qtbot.waitUntil(
        lambda: (
            entry.marker.palette().color(QtGui.QPalette.ColorRole.Window) == base_marker
            and entry.label.palette().color(QtGui.QPalette.ColorRole.WindowText)
            == base_label
        )
    )
    assert entry.marker.palette().color(QtGui.QPalette.ColorRole.Window) == base_marker
    assert (
        entry.label.palette().color(QtGui.QPalette.ColorRole.WindowText) == base_label
    )

    win.close()


def test_ptable_hover_preview_does_not_flicker_back_to_selection(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win._handle_card_selected(51, QtCore.Qt.KeyboardModifier.NoModifier)

    sequence = _hover_sequence_between_cards(
        win,
        win.periodic_table.cards[51],
        win.periodic_table.cards[115],
    )

    assert sequence == ["Sb", "Bi", "Mc"]

    win.close()


def test_ptable_notation_toggle_and_photon_energy_display(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_edge",
        lambda _symbol: {"2p1/2": 120.0, "2p3/2": 74.0, "3d5/2": 5.0},
    )
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow(photon_energy=100.0, workfunction=4.5)
    _show_window(qtbot, win)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)

    assert win.notation_label.text() == "Notation:"
    assert isinstance(win.notation_combo, QtWidgets.QComboBox)
    assert win.notation_combo.count() == 2
    assert win.notation_combo.itemText(0) == "Orbital"
    assert win.notation_combo.itemText(1) == "X-ray level"
    assert win.notation_combo.itemData(0) == "orbital"
    assert win.notation_combo.itemData(1) == "iupac"
    assert win.notation_combo.currentText() == "Orbital"
    assert win.notation_combo.toolTip() == "Choose how energy levels are labeled."
    assert (
        win.notation_combo.sizeAdjustPolicy()
        == QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
    )

    assert win.inspector.levels_table.horizontalHeader().isVisible() is True
    assert win.inspector.levels_table.rowCount() == 2
    assert win.inspector.levels_table.verticalHeaderItem(0).text() == "H"
    assert win.inspector.levels_table.verticalHeaderItem(1).text() == "H KE"
    assert win.inspector.levels_table.horizontalHeaderItem(0).text() == "2p1/2"
    assert (
        win.inspector.levels_table.horizontalHeaderItem(0).data(
            QtCore.Qt.ItemDataRole.UserRole
        )
        == "2<i>p</i><sub>1/2</sub>"
    )
    assert win.inspector.levels_table.item(0, 0).text() == "120"
    assert win.inspector.levels_table.item(1, 0).text() == ""
    assert win.inspector.levels_table.item(1, 1).text() == "21.5"
    assert (
        win.inspector.levels_table.item(0, 1).background().color()
        == win.inspector._theme.edge_match_bg
    )
    assert (
        win.inspector.levels_table.item(1, 1).background().color()
        == win.inspector._theme.harmonic_match_bgs[0]
    )
    assert (
        win.inspector.levels_table.selectionBehavior()
        == QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
    )
    assert (
        win.inspector.levels_table.selectionMode()
        == QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
    )

    binding_item = win.inspector.levels_table.item(0, 0)
    assert binding_item is not None
    assert not binding_item.isSelected()

    qtbot.mouseClick(
        win.inspector.levels_table.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=win.inspector.levels_table.visualItemRect(binding_item).center(),
    )

    assert binding_item.isSelected()
    assert win.inspector.levels_table.currentItem() is binding_item

    win.notation_combo.setCurrentIndex(win.notation_combo.findData("iupac"))

    assert win.current_notation == "iupac"
    assert win.notation_combo.currentText() == "X-ray level"
    assert win.inspector.levels_table.horizontalHeaderItem(0).text() == "L2"
    assert (
        win.inspector.levels_table.horizontalHeaderItem(0).data(
            QtCore.Qt.ItemDataRole.UserRole
        )
        == "L<sub>2</sub>"
    )

    assert win.inspector.cross_section_plot.legend_labels == ("L2,3", "M4,5", "Total")
    assert win.inspector.cross_section_plot.photon_line_energy == 100.0
    assert win.inspector.cross_section_plot.photon_line_energies == (100.0,)

    win.close()


def test_ptable_notation_persists_across_windows(qtbot) -> None:
    first = PeriodicTableWindow()
    _show_window(qtbot, first)

    assert first.current_notation == "orbital"

    first.notation_combo.setCurrentIndex(first.notation_combo.findData("iupac"))

    assert first.current_notation == "iupac"

    first.close()

    second = PeriodicTableWindow()
    _show_window(qtbot, second)

    assert second.current_notation == "iupac"

    second.close()

    explicit = PeriodicTableWindow(notation="orbital")
    _show_window(qtbot, explicit)

    assert explicit.current_notation == "orbital"

    explicit.close()

    with pytest.raises(
        ValueError, match="notation must be either 'orbital' or 'iupac'"
    ):
        PeriodicTableWindow(notation="xps")

    restored = PeriodicTableWindow()
    _show_window(qtbot, restored)

    assert restored.current_notation == "iupac"

    restored.close()


def test_ptable_harmonic_rows_expand_from_spinbox(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_edge",
        lambda _symbol: {"1s": 150.0},
    )
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)

    assert win.harmonic_frame.isEnabled() is True
    assert win.max_harmonic_spin.isEnabled() is True
    assert win.inspector.levels_table.rowCount() == 2
    assert win.inspector.levels_table.verticalHeaderItem(1).text() == "H KE"

    win.max_harmonic_spin.setValue(2)

    assert win.inspector.levels_table.rowCount() == 3
    assert win.inspector.levels_table.verticalHeaderItem(1).text() == "H KE (1hv)"
    assert win.inspector.levels_table.verticalHeaderItem(2).text() == "H KE (2hv)"
    assert win.inspector.levels_table.item(0, 0).text() == "150"
    assert (
        win.inspector.levels_table.item(0, 0).background().color()
        == win.inspector._theme.edge_match_bg
    )
    assert win.inspector.levels_table.item(1, 0).text() == ""
    assert win.inspector.levels_table.item(2, 0).text() == "10"
    assert (
        win.inspector.levels_table.item(2, 0).background().color()
        == win.inspector._theme.harmonic_match_bgs[1]
    )

    win.close()


def test_ptable_copy_actions_and_invalid_inputs(
    qtbot,
    monkeypatch,
) -> None:
    copied: list[str] = []

    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_edge",
        lambda _symbol: {"1s": 12.0, "2p3/2": 3.5},
    )
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda content: copied.append(
            "\n".join(content) if isinstance(content, list) else content
        ),
    )

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)

    assert win.photon_energy_label.text() == "hν"
    assert win.workfunction_label.text() == "Φ"
    assert win.photon_energy_edit.placeholderText() == ""
    assert win.workfunction_edit.placeholderText() == "0"
    assert win.photon_energy_edit.isClearButtonEnabled() is True
    assert win.workfunction_edit.isClearButtonEnabled() is True
    assert getattr(win.photon_energy_edit, "suffix", None) == "eV"
    assert getattr(win.workfunction_edit, "suffix", None) == "eV"
    initial_photon_margin = win.photon_energy_edit.textMargins().right()
    initial_photon_width = win.photon_energy_edit.width()
    initial_workfunction_width = win.workfunction_edit.width()
    assert win.photon_energy_label.toolTip().startswith("<qt>")
    assert "<i>" in win.photon_energy_label.toolTip()
    assert "<sub>" in win.photon_energy_label.toolTip()
    assert "−" in win.photon_energy_label.toolTip()
    assert win.photon_energy_edit.toolTip() == win.photon_energy_label.toolTip()
    assert "Spectrometer work function" in win.workfunction_label.toolTip()
    assert "Leave blank to treat Φ as 0 eV." in win.workfunction_label.toolTip()
    assert win.workfunction_label.toolTip().startswith("<qt>")
    assert "<i>" in win.workfunction_label.toolTip()
    assert "<sub>" in win.workfunction_label.toolTip()
    assert "−" in win.workfunction_label.toolTip()
    assert win.workfunction_edit.toolTip() == win.workfunction_label.toolTip()

    assert win.harmonic_frame.isEnabled() is False
    assert win.max_harmonic_spin.isEnabled() is False
    assert (
        win.max_harmonic_label.palette().color(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.WindowText,
        )
        == win._theme.disabled_text
    )
    assert win.inspector.plot_target_combo.isVisible() is False
    assert win.inspector.levels_table.horizontalHeader().isVisible() is True
    assert win.inspector.levels_table.rowCount() == 1
    qtbot.mouseClick(win.inspector.copy_values_button, QtCore.Qt.MouseButton.LeftButton)
    qtbot.mouseClick(win.inspector.copy_table_button, QtCore.Qt.MouseButton.LeftButton)

    assert copied[0] == "12\n3.5"
    assert copied[1].splitlines()[0] == "Metric\t1s\t2p3/2"

    win.photon_energy_edit.setText("12345678901234567890123456789012345678901234567890")
    qtbot.waitUntil(
        lambda: any(
            button.isVisible()
            for button in win.photon_energy_edit.findChildren(QtWidgets.QAbstractButton)
            if button.parent() is win.photon_energy_edit
        )
    )

    photon_suffix = next(
        label
        for label in win.photon_energy_edit.findChildren(QtWidgets.QLabel)
        if label.objectName() == "suffix"
    )
    photon_cursor_right = (
        win.photon_energy_edit.cursorRect().x()
        + win.photon_energy_edit.cursorRect().width()
    )

    assert win.photon_energy_edit.textMargins().right() == initial_photon_margin
    assert (
        photon_suffix.geometry().x() - photon_cursor_right
        <= win.photon_energy_edit.fontMetrics().horizontalAdvance(" ")
    )

    win.photon_energy_edit.setText("invalid")
    win.workfunction_edit.setText("-1")

    assert win.photon_energy_edit.width() == initial_photon_width
    assert win.workfunction_edit.width() == initial_workfunction_width
    assert win.photon_energy is None
    assert win.photon_energy_edit.property("invalid") is True
    assert win.workfunction_edit.property("invalid") is True
    assert win.harmonic_frame.isEnabled() is False
    assert win.max_harmonic_spin.isEnabled() is False
    assert win.inspector.levels_table.rowCount() == 1

    win.close()


def test_ptable_levels_table_match_highlights_retheme_live_on_palette_change(
    qtbot,
    monkeypatch,
) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None

    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 150.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    original_palette = QtGui.QPalette(app.palette())
    app.setPalette(_light_palette())
    try:
        win = PeriodicTableWindow(photon_energy=80.0, max_harmonic=2)
        _show_window(qtbot, win)
        win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)

        binding_item = win.inspector.levels_table.item(0, 0)
        second_harmonic_item = win.inspector.levels_table.item(2, 0)
        assert binding_item is not None
        assert second_harmonic_item is not None
        assert binding_item.background().color() == win.inspector._theme.edge_match_bg
        assert (
            second_harmonic_item.background().color()
            == win.inspector._theme.harmonic_match_bgs[1]
        )

        light_binding = QtGui.QColor(binding_item.background().color())
        light_second_harmonic = QtGui.QColor(second_harmonic_item.background().color())

        app.setPalette(_dark_palette())
        qtbot.waitUntil(lambda: win._theme.is_dark is True)
        qtbot.waitUntil(
            lambda: (
                binding_item.background().color() != light_binding
                and second_harmonic_item.background().color() != light_second_harmonic
            )
        )

        assert binding_item.background().color() == win.inspector._theme.edge_match_bg
        assert (
            second_harmonic_item.background().color()
            == win.inspector._theme.harmonic_match_bgs[1]
        )
        assert (
            binding_item.background().color()
            != second_harmonic_item.background().color()
        )

        win.close()
    finally:
        app.setPalette(original_palette)


def test_ptable_missing_data_states_and_blank_configuration(
    qtbot,
    monkeypatch,
) -> None:
    def _binding(symbol: str) -> dict[str, float]:
        if symbol == "Ds":
            return {}
        return {"1s": 8.0}

    def _cross_section(symbol: str) -> dict[str, xr.DataArray]:
        if symbol == "Ds":
            raise KeyError(symbol)
        return _fake_cross_sections(symbol)

    monkeypatch.setattr(erlab.analysis.xps, "get_edge", _binding)
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _cross_section)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    darmstadtium = win.periodic_table.cards[110]
    assert darmstadtium.config_label.text() == ""

    win._handle_card_selected(110, QtCore.Qt.KeyboardModifier.NoModifier)

    assert (
        win.inspector.levels_stack.currentWidget() is win.inspector.levels_empty_label
    )
    assert "unavailable" in win.inspector.levels_empty_label.text().lower()
    assert (
        win.inspector.cross_section_plot._stack.currentWidget()
        is win.inspector.cross_section_plot.empty_label
    )
    assert "unavailable" in win.inspector.cross_section_plot.empty_label.text().lower()
    assert (
        win.inspector.cross_section_plot.empty_label.palette().color(
            QtGui.QPalette.ColorRole.WindowText
        )
        == win._theme.muted_text
    )

    win.close()


def test_ptable_plot_ranges_and_metadata_snapshot(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)
    plot = win.inspector.cross_section_plot

    assert (
        win.table_view.horizontalScrollBarPolicy()
        == QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert (
        win.table_view.verticalScrollBarPolicy()
        == QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert win.table_view.horizontalScrollBar().maximum() == 0
    assert win.table_view.verticalScrollBar().maximum() == 0
    assert plot.log_x is True
    assert plot.log_y is True
    assert plot.x_range_eV == (10.0, 1500.0)
    assert plot.y_range == (1e-3, 100.0)
    assert plot.plot_item.getViewBox().state["mouseEnabled"] == [True, True]
    assert plot.plot_item.getViewBox().state["mouseMode"] == pg.ViewBox.PanMode
    assert plot.x_tick_labels == ("10", "100", "1000")
    assert plot.y_tick_labels == ("0.001", "0.01", "0.1", "1", "10", "100")
    assert plot.x_minor_tick_count > 0
    assert plot.y_minor_tick_count > 0
    assert plot.y_axis_label_html == "\u03c3<sub>abs</sub> (Mb/atom)"
    assert set(plot.plotted_labels) == {"2p", "3d", "Total"}
    assert plot.photon_line_energy == 80.0
    assert plot.photon_line_energies == (80.0,)
    initial_pen_colors = [
        plot._series_pen(index).color().name()
        for index in range(min(6, len(plot._LINE_COLORS)))
    ]
    assert len(set(initial_pen_colors)) == len(initial_pen_colors)
    assert plot._series_pen(0).style() == QtCore.Qt.PenStyle.SolidLine
    assert (
        plot._series_pen(len(plot._LINE_COLORS)).style() == QtCore.Qt.PenStyle.DashLine
    )
    assert (
        plot._series_pen(0).color().name()
        == plot._series_pen(len(plot._LINE_COLORS)).color().name()
    )
    bottom_axis = plot.plot_item.getAxis("bottom")
    left_axis = plot.plot_item.getAxis("left")
    top_axis = plot.plot_item.getAxis("top")
    right_axis = plot.plot_item.getAxis("right")
    assert isinstance(bottom_axis, EdgeTickAxisItem)
    assert isinstance(left_axis, EdgeTickAxisItem)
    assert isinstance(top_axis, EdgeTickAxisItem)
    assert isinstance(right_axis, EdgeTickAxisItem)
    assert bottom_axis.pen().color() == win._theme.text
    assert bottom_axis.tickPen().color() == win._theme.text
    assert bottom_axis.textPen().color() == win._theme.text
    assert left_axis.pen().color() == win._theme.text
    assert left_axis.tickPen().color() == win._theme.text
    assert left_axis.textPen().color() == win._theme.text
    assert bottom_axis.grid == 255
    assert left_axis.grid == 255
    assert (
        plot.legend_label.palette().color(QtGui.QPalette.ColorRole.WindowText)
        == win._theme.text
    )
    expected_hover_color = QtGui.QColor(win._theme.accent)
    expected_hover_color.setAlpha(248 if win._theme.is_dark else 232)
    hover_vertical_pen = plot._hover_vertical_line.opts["pen"]
    hover_horizontal_pen = plot._hover_horizontal_line.opts["pen"]
    assert hover_vertical_pen.color() == expected_hover_color
    assert hover_horizontal_pen.color() == expected_hover_color
    assert hover_vertical_pen.widthF() == pytest.approx(1.4)
    assert hover_horizontal_pen.widthF() == pytest.approx(1.4)
    assert hover_vertical_pen.style() == QtCore.Qt.PenStyle.DashLine
    assert hover_horizontal_pen.style() == QtCore.Qt.PenStyle.DashLine
    assert top_axis.isVisible() is True
    assert right_axis.isVisible() is True
    assert top_axis.style["showValues"] is False
    assert right_axis.style["showValues"] is False
    assert bottom_axis.fixedHeight is None
    assert left_axis.fixedWidth is None
    assert bottom_axis.style["hideOverlappingLabels"] == 64
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    try:
        bottom_specs = bottom_axis.generateDrawSpecs(painter)
        left_specs = left_axis.generateDrawSpecs(painter)
        top_specs = top_axis.generateDrawSpecs(painter)
        right_specs = right_axis.generateDrawSpecs(painter)
    finally:
        painter.end()
    assert bottom_specs is not None
    assert left_specs is not None
    assert top_specs is not None
    assert right_specs is not None
    bottom_line, _, _ = bottom_specs
    left_line, _, _ = left_specs
    top_line, _, _ = top_specs
    right_line, _, _ = right_specs
    bottom_bounds = bottom_axis.mapRectFromParent(bottom_axis.geometry())
    left_bounds = left_axis.mapRectFromParent(left_axis.geometry())
    top_bounds = top_axis.mapRectFromParent(top_axis.geometry())
    right_bounds = right_axis.mapRectFromParent(right_axis.geometry())
    assert bottom_line[1].y() == pytest.approx(bottom_bounds.top())
    assert bottom_line[2].y() == pytest.approx(bottom_bounds.top())
    assert bottom_line[1].x() == pytest.approx(bottom_bounds.left())
    assert bottom_line[2].x() == pytest.approx(bottom_bounds.right())
    assert left_line[1].x() == pytest.approx(left_bounds.right())
    assert left_line[2].x() == pytest.approx(left_bounds.right())
    assert left_line[1].y() == pytest.approx(left_bounds.top())
    assert left_line[2].y() == pytest.approx(left_bounds.bottom())
    assert top_line[1].y() == pytest.approx(top_bounds.bottom())
    assert top_line[2].y() == pytest.approx(top_bounds.bottom())
    assert top_line[1].x() == pytest.approx(top_bounds.left())
    assert top_line[2].x() == pytest.approx(top_bounds.right())
    assert right_line[1].x() == pytest.approx(right_bounds.left())
    assert right_line[2].x() == pytest.approx(right_bounds.left())
    assert right_line[1].y() == pytest.approx(right_bounds.top())
    assert right_line[2].y() == pytest.approx(right_bounds.bottom())
    assert (
        bottom_axis.boundingRect().left()
        < bottom_axis.mapRectFromParent(bottom_axis.geometry()).left()
    )
    assert (
        bottom_axis.boundingRect().right()
        > bottom_axis.mapRectFromParent(bottom_axis.geometry()).right()
    )
    assert (
        left_axis.boundingRect().top()
        < left_axis.mapRectFromParent(left_axis.geometry()).top()
    )
    assert (
        left_axis.boundingRect().bottom()
        > left_axis.mapRectFromParent(left_axis.geometry()).bottom()
    )

    assert ELEMENT_POSITIONS[57] == (6, 3)
    assert ELEMENT_POSITIONS[58] == (9, 4)
    assert ELEMENT_CATEGORIES[10] == "noble_gas"
    assert GROUND_STATE_CONFIGURATIONS["Na"] == "[Ne] 3s^1"
    assert GROUND_STATE_CONFIGURATIONS["Pt"] == "[Xe] 4f^14 5d^9 6s^1"
    assert GROUND_STATE_CONFIGURATIONS["Tl"] == "[Hg] 6p^1"
    assert GROUND_STATE_CONFIGURATIONS["Pb"] == "[Hg] 6p^2"
    assert GROUND_STATE_CONFIGURATIONS["Rn"] == "[Hg] 6p^6"
    assert GROUND_STATE_CONFIGURATIONS["U"] == "[Rn] 5f^3 6d^1 7s^2"
    assert (
        configuration_to_html("[Ar] 3d^10 4s^2") == "[Ar]3d<sup>10</sup>4s<sup>2</sup>"
    )

    win.close()


def test_ptable_detached_f_block_is_separated_from_main_table(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    layout = win.periodic_table.layout()
    assert isinstance(layout, QtWidgets.QGridLayout)
    assert (
        layout.rowMinimumHeight(win.periodic_table._SERIES_GAP_ROW)
        == win.periodic_table._SERIES_GAP_HEIGHT
    )
    assert (
        win.periodic_table.cards[58].geometry().top()
        > win.periodic_table.cards[104].geometry().bottom()
    )
    assert (
        win.periodic_table.cards[90].geometry().top()
        > win.periodic_table.cards[104].geometry().bottom()
    )

    win.close()


def test_ptable_table_view_enters_scroll_mode_below_legacy_minimum_size(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win.resize(900, 650)
    QtWidgets.QApplication.processEvents()

    assert win.width() == 900
    assert win.height() == 650
    assert (
        win.table_view.horizontalScrollBar().maximum() > 0
        or win.table_view.verticalScrollBar().maximum() > 0
    )

    win.close()


def test_ptable_table_view_scale_does_not_shrink_past_legacy_minimum(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win.resize(1180, 760)
    QtWidgets.QApplication.processEvents()
    baseline_scale = win.table_view.transform().m11()

    win.resize(800, 600)
    QtWidgets.QApplication.processEvents()

    assert win.table_view.transform().m11() == pytest.approx(baseline_scale, rel=1e-2)
    assert win.table_view.transform().m22() == pytest.approx(baseline_scale, rel=1e-2)

    win.close()


def test_ptable_uses_direct_total_cross_section(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)

    render_data = win.inspector.cross_section_plot._get_render_data("H")
    expected_total = erlab.analysis.xps.get_total_cross_section("H")
    expected_hv = np.asarray(expected_total.hv.values, dtype=np.float64)
    expected_sigma = np.asarray(expected_total.values, dtype=np.float64)
    mask = (
        np.isfinite(expected_hv)
        & np.isfinite(expected_sigma)
        & (expected_hv >= 10.0)
        & (expected_hv <= 1500.0)
        & (expected_sigma > 0.0)
    )
    np.testing.assert_allclose(render_data.total_hv, expected_hv[mask])
    np.testing.assert_allclose(render_data.total_sigma, expected_sigma[mask])

    win.close()


def test_ptable_plot_legend_wraps_and_hover_emphasizes_curve(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _many_cross_sections)
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _many_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    plot = win.inspector.cross_section_plot
    assert len(plot.legend_label.entry_widgets) == 9
    assert plot.legend_label.height() > plot.legend_label.entry_widgets[0].height()

    legend_layout = plot.legend_label.layout()
    assert legend_layout is not None
    assert legend_layout.horizontalSpacing() == 0
    assert legend_layout.verticalSpacing() == 0
    last_entry = plot.legend_label.entry_widgets[-1]
    last_index = legend_layout.indexOf(last_entry)
    assert legend_layout.getItemPosition(last_index)[0] > 0
    assert legend_layout.getItemPosition(last_index)[0] == 1
    assert legend_layout.getItemPosition(last_index)[1] == 2
    assert legend_layout.getItemPosition(last_index)[3] == 2
    assert plot.legend_label.entry_widgets[2].label.text() == "3<i>d</i>"
    assert (
        plot.legend_label.entry_widgets[2].label.textFormat()
        == QtCore.Qt.TextFormat.RichText
    )

    first_entry = plot.legend_label.entry_widgets[0]
    first_label = first_entry.label_text
    second_label = plot.legend_label.entry_widgets[1].label_text
    base_pen = QtGui.QPen(plot._curve_items[first_label][1])
    other_base_pen = QtGui.QPen(plot._curve_items[second_label][1])
    first_pos = first_entry.mapTo(plot, first_entry.rect().center())

    QtTest.QTest.mouseMove(plot, first_pos)
    QtWidgets.QApplication.processEvents()

    highlighted_pen = plot._curve_items[first_label][0].opts["pen"]
    dimmed_pen = plot._curve_items[second_label][0].opts["pen"]
    assert plot._active_legend_label == first_label
    assert highlighted_pen.widthF() > base_pen.widthF()
    assert dimmed_pen.color().alpha() < other_base_pen.color().alpha()
    assert first_entry._active is True

    QtTest.QTest.mouseMove(
        plot,
        QtCore.QPoint(first_pos.x(), plot._stack.geometry().center().y()),
    )
    QtWidgets.QApplication.processEvents()

    restored_pen = plot._curve_items[first_label][0].opts["pen"]
    assert plot._active_legend_label is None
    assert restored_pen.widthF() == base_pen.widthF()
    assert first_entry._active is False

    win.close()


def test_ptable_plot_legend_click_toggles_multi_selection(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _many_cross_sections)
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _many_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    plot = win.inspector.cross_section_plot
    entries = {entry.label_text: entry for entry in plot.legend_label.entry_widgets}
    first_entry = entries["3s"]
    second_entry = entries["3p"]
    third_label = "3d"
    base_first_pen = QtGui.QPen(plot._curve_items["3s"][1])
    base_second_pen = QtGui.QPen(plot._curve_items["3p"][1])
    base_third_pen = QtGui.QPen(plot._curve_items[third_label][1])

    qtbot.mouseClick(first_entry, QtCore.Qt.MouseButton.LeftButton)

    first_pen = plot._curve_items["3s"][0].opts["pen"]
    second_pen = plot._curve_items["3p"][0].opts["pen"]
    assert plot._toggled_legend_keys == {"3s"}
    assert plot._active_legend_label is None
    assert first_pen.widthF() > base_first_pen.widthF()
    assert second_pen.color().alpha() < base_second_pen.color().alpha()
    assert first_entry._active is True
    assert second_entry._active is False

    qtbot.mouseClick(second_entry, QtCore.Qt.MouseButton.LeftButton)

    first_pen = plot._curve_items["3s"][0].opts["pen"]
    second_pen = plot._curve_items["3p"][0].opts["pen"]
    third_pen = plot._curve_items[third_label][0].opts["pen"]
    assert plot._toggled_legend_keys == {"3s", "3p"}
    assert first_pen.widthF() > base_first_pen.widthF()
    assert second_pen.widthF() > base_second_pen.widthF()
    assert third_pen.color().alpha() < base_third_pen.color().alpha()
    assert first_entry._active is True
    assert second_entry._active is True

    qtbot.mouseClick(first_entry, QtCore.Qt.MouseButton.LeftButton)

    first_pen = plot._curve_items["3s"][0].opts["pen"]
    second_pen = plot._curve_items["3p"][0].opts["pen"]
    assert plot._toggled_legend_keys == {"3p"}
    assert plot._active_legend_label is None
    assert first_pen.color().alpha() < base_first_pen.color().alpha()
    assert second_pen.widthF() > base_second_pen.widthF()
    assert first_entry._active is False
    assert second_entry._active is True

    win.close()


def test_ptable_plot_legend_toggles_persist_across_hover_preview(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_cross_section",
        _preview_persistence_cross_sections,
    )
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _preview_persistence_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    plot = win.inspector.cross_section_plot
    entries = {entry.label_text: entry for entry in plot.legend_label.entry_widgets}
    qtbot.mouseClick(entries["2p"], QtCore.Qt.MouseButton.LeftButton)
    qtbot.mouseClick(entries["3d"], QtCore.Qt.MouseButton.LeftButton)

    assert plot._toggled_legend_keys == {"2p", "3d"}
    assert entries["2p"]._active is True
    assert entries["3d"]._active is True

    win._set_hover_atomic_number(1)
    QtWidgets.QApplication.processEvents()

    plot = win.inspector.cross_section_plot
    hover_entries = {
        entry.label_text: entry for entry in plot.legend_label.entry_widgets
    }
    hover_2p_pen = plot._curve_items["2p"][0].opts["pen"]
    hover_4s_pen = plot._curve_items["4s"][0].opts["pen"]
    hover_2p_base_pen = QtGui.QPen(plot._curve_items["2p"][1])
    hover_4s_base_pen = QtGui.QPen(plot._curve_items["4s"][1])

    assert plot.legend_labels == ("2p", "4s", "Total")
    assert plot._toggled_legend_keys == {"2p", "3d"}
    assert hover_entries["2p"]._active is True
    assert hover_entries["4s"]._active is False
    assert hover_2p_pen.widthF() > hover_2p_base_pen.widthF()
    assert hover_4s_pen.color().alpha() < hover_4s_base_pen.color().alpha()

    win._set_hover_atomic_number(None)
    QtWidgets.QApplication.processEvents()

    plot = win.inspector.cross_section_plot
    restored_entries = {
        entry.label_text: entry for entry in plot.legend_label.entry_widgets
    }
    restored_2p_pen = plot._curve_items["2p"][0].opts["pen"]
    restored_3d_pen = plot._curve_items["3d"][0].opts["pen"]
    restored_2p_base_pen = QtGui.QPen(plot._curve_items["2p"][1])
    restored_3d_base_pen = QtGui.QPen(plot._curve_items["3d"][1])

    assert plot.legend_labels == ("2p", "3d", "Total")
    assert plot._toggled_legend_keys == {"2p", "3d"}
    assert restored_entries["2p"]._active is True
    assert restored_entries["3d"]._active is True
    assert restored_2p_pen.widthF() > restored_2p_base_pen.widthF()
    assert restored_3d_pen.widthF() > restored_3d_base_pen.widthF()

    win.close()


def test_ptable_plot_legend_toggles_ignore_missing_preview_series(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_cross_section",
        _preview_persistence_cross_sections,
    )
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _preview_persistence_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    plot = win.inspector.cross_section_plot
    entries = {entry.label_text: entry for entry in plot.legend_label.entry_widgets}
    qtbot.mouseClick(entries["3d"], QtCore.Qt.MouseButton.LeftButton)

    assert plot._toggled_legend_keys == {"3d"}

    win._set_hover_atomic_number(1)
    QtWidgets.QApplication.processEvents()

    plot = win.inspector.cross_section_plot
    hover_entries = {
        entry.label_text: entry for entry in plot.legend_label.entry_widgets
    }
    hover_2p_pen = plot._curve_items["2p"][0].opts["pen"]
    hover_4s_pen = plot._curve_items["4s"][0].opts["pen"]
    hover_2p_base_pen = QtGui.QPen(plot._curve_items["2p"][1])
    hover_4s_base_pen = QtGui.QPen(plot._curve_items["4s"][1])

    assert plot.legend_labels == ("2p", "4s", "Total")
    assert plot._toggled_legend_keys == {"3d"}
    assert hover_entries["2p"]._active is False
    assert hover_entries["4s"]._active is False
    assert hover_2p_pen.widthF() == hover_2p_base_pen.widthF()
    assert hover_4s_pen.widthF() == hover_4s_base_pen.widthF()

    win._set_hover_atomic_number(None)
    QtWidgets.QApplication.processEvents()

    plot = win.inspector.cross_section_plot
    restored_entries = {
        entry.label_text: entry for entry in plot.legend_label.entry_widgets
    }

    assert plot.legend_labels == ("2p", "3d", "Total")
    assert plot._toggled_legend_keys == {"3d"}
    assert restored_entries["3d"]._active is True

    win.close()


def test_ptable_plot_legend_toggles_survive_notation_switch(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_cross_section",
        _preview_persistence_cross_sections,
    )
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _preview_persistence_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    plot = win.inspector.cross_section_plot
    entries = {entry.label_text: entry for entry in plot.legend_label.entry_widgets}
    qtbot.mouseClick(entries["2p"], QtCore.Qt.MouseButton.LeftButton)

    assert plot._toggled_legend_keys == {"2p"}
    assert entries["2p"]._active is True

    win.notation_combo.setCurrentIndex(win.notation_combo.findData("iupac"))
    QtWidgets.QApplication.processEvents()

    plot = win.inspector.cross_section_plot
    iupac_entries = {
        entry.label_text: entry for entry in plot.legend_label.entry_widgets
    }
    iupac_pen = plot._curve_items["L2,3"][0].opts["pen"]
    iupac_base_pen = QtGui.QPen(plot._curve_items["L2,3"][1])
    other_pen = plot._curve_items["M4,5"][0].opts["pen"]
    other_base_pen = QtGui.QPen(plot._curve_items["M4,5"][1])

    assert plot.legend_labels == ("L2,3", "M4,5", "Total")
    assert plot._toggled_legend_keys == {"2p"}
    assert iupac_entries["L2,3"]._active is True
    assert iupac_entries["M4,5"]._active is False
    assert iupac_pen.widthF() > iupac_base_pen.widthF()
    assert other_pen.color().alpha() < other_base_pen.color().alpha()

    win.close()


def test_ptable_plot_legend_hover_tracking_is_directionally_stable(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _many_cross_sections)
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _many_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    legend = win.inspector.cross_section_plot.legend_label
    entries = {entry.label_text: entry for entry in legend.entry_widgets}
    downward = _hover_sequence_between_widgets(
        legend,
        entries["3s"],
        entries["4f"],
        lambda: legend._active_label,
    )
    upward = _hover_sequence_between_widgets(
        legend,
        entries["4f"],
        entries["3s"],
        lambda: legend._active_label,
    )

    assert downward == ["3s", "4f"]
    assert upward == ["4f", "3s"]

    win.close()


def test_ptable_plot_legend_upward_hover_exit_clears_active_label(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _many_cross_sections)
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _many_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    plot = win.inspector.cross_section_plot
    entries = {entry.label_text: entry for entry in plot.legend_label.entry_widgets}
    sequence = _hover_sequence_from_widget_to_point(
        plot,
        entries["4f"],
        QtCore.QPoint(
            entries["4f"].mapTo(plot, entries["4f"].rect().center()).x(),
            plot._stack.geometry().center().y(),
        ),
        lambda: plot._active_legend_label,
    )

    assert sequence[0] == "4f"
    assert sequence[-1] is None
    assert plot._active_legend_label is None
    assert entries["4f"]._active is False

    win.close()


def test_ptable_plot_hover_crosshair_snaps_to_nearest_sampled_point(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_binding_energy",
        lambda _symbol: {"1s": 5.0},
        raising=False,
    )
    monkeypatch.setattr(
        erlab.analysis.xps, "get_cross_section", _mismatched_cross_sections
    )
    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_total_cross_section",
        _mismatched_total_cross_section,
    )

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)

    plot = win.inspector.cross_section_plot
    assert not hasattr(plot, "_hover_overlay")

    _move_cross_section_hover(plot, photon_energy=19.0, sigma=0.11)

    assert plot._hover_energy_eV == 20.0
    assert plot._hover_cursor_sigma == 0.2
    assert plot._hover_vertical_line.isVisible()
    assert plot._hover_horizontal_line.isVisible()
    assert plot._hover_x_badge.text() == "20.00"
    assert plot._hover_y_badge.text() == plot._format_hover_sigma(0.2)
    assert plot._active_legend_label is None

    _move_cross_section_hover(plot, photon_energy=19.0, sigma=0.25)

    assert plot._hover_energy_eV == 15.0
    assert plot._hover_cursor_sigma == 0.24
    assert plot._hover_x_badge.text() == "15.00"
    assert plot._hover_y_badge.text() == plot._format_hover_sigma(0.24)

    first_cursor_sigma = plot._hover_cursor_sigma
    assert first_cursor_sigma is not None

    _move_cross_section_hover(plot, photon_energy=19.0, sigma=10.0)

    assert plot._hover_energy_eV == 20.0
    assert plot._hover_cursor_sigma == 1.8
    assert plot._hover_cursor_sigma != first_cursor_sigma
    assert plot._hover_y_badge.text() == plot._format_hover_sigma(
        plot._hover_cursor_sigma
    )

    _move_cross_section_hover(plot, photon_energy=29.0, sigma=0.33)

    assert plot._hover_energy_eV == 30.0
    assert plot._hover_cursor_sigma == 0.28
    assert plot._hover_y_badge.text() == plot._format_hover_sigma(0.28)

    plot.eventFilter(
        plot.plot_widget.viewport(),
        QtCore.QEvent(QtCore.QEvent.Type.Leave),
    )

    assert plot._hover_energy_eV is None
    assert plot._hover_vertical_line.isVisible() is False
    assert plot._hover_horizontal_line.isVisible() is False
    assert plot._hover_x_badge.isHidden()
    assert plot._hover_y_badge.isHidden()

    _move_cross_section_hover(plot, photon_energy=19.0, sigma=0.11)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)

    assert plot._hover_energy_eV is None
    assert plot._hover_x_badge.isHidden()
    assert plot._hover_y_badge.isHidden()

    win.close()


def test_ptable_multi_select_summary_and_plot_picker(
    qtbot,
    monkeypatch,
) -> None:
    binding_levels = {
        "H": {"1s": 12.0},
        "He": {"1s": 20.0},
        "Li": {"1s": 30.0},
        "Be": {"1s": 40.0},
        "B": {"1s": 50.0},
        "C": {"1s": 60.0},
        "N": {"1s": 70.0},
    }

    monkeypatch.setattr(
        erlab.analysis.xps, "get_edge", lambda symbol: binding_levels[symbol]
    )
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)
    plot_height_single = win.inspector.cross_section_plot.height()
    summary_margins = win.inspector._summary_layout.contentsMargins()
    summary_spacing = win.inspector._summary_layout.spacing()
    summary_row_spacing = win.inspector.summary_cards_grid.verticalSpacing()
    summary_base_height = (
        summary_margins.top()
        + win.inspector._summary_header_layout.sizeHint().height()
        + summary_spacing
        + summary_margins.bottom()
    )
    one_row_summary_height = summary_base_height + CompactElementChip._DETAILED_HEIGHT
    two_row_summary_height = (
        summary_base_height
        + (2 * CompactElementChip._DETAILED_HEIGHT)
        + summary_row_spacing
    )

    assert win.inspector.summary_frame.height() == one_row_summary_height

    win._handle_card_selected(2, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (1, 2)
    assert win.selected_atomic_number == 2
    assert (
        win.inspector.summary_stack.currentWidget() is win.inspector.summary_cards_page
    )
    assert win.inspector.mode_label.text() == "2 selected"
    assert len(win.inspector._summary_cards) == 2
    assert (
        win.inspector._summary_cards[0].height() == CompactElementChip._DETAILED_HEIGHT
    )
    assert win.inspector._summary_cards[0].width() == 92
    assert win.inspector._summary_cards[0].mass_label.isHidden() is False
    assert win.inspector._summary_cards[0].config_label.isHidden() is False
    assert win.inspector._summary_cards[0].config_label.wordWrap() is False
    assert (
        win.inspector._summary_cards[0].mass_label.text()
        == _format_mass(win.inspector._summary_cards[0].record.mass) + " u"
    )
    assert win.inspector._summary_cards[0].config_label.text() == configuration_to_html(
        win.inspector._summary_cards[0].record.configuration
    )
    chip_layout = win.inspector._summary_cards[0].layout()
    assert chip_layout is not None
    assert chip_layout.contentsMargins().top() == 4
    assert chip_layout.contentsMargins().bottom() == 7
    assert chip_layout.spacing() == 0
    assert win.inspector._summary_cards[0].symbol_label.contentsMargins().top() == -1
    chip_text_width = (
        win.inspector._summary_cards[0].width()
        - chip_layout.contentsMargins().left()
        - chip_layout.contentsMargins().right()
    )
    assert win.inspector._summary_cards[0].name_label.font().pointSizeF() > 11.0
    assert win.inspector._summary_cards[0].mass_label.font().pointSizeF() > 10.2
    assert win.inspector._summary_cards[0].config_label.font().pointSizeF() > 9.8
    assert (
        win.inspector._summary_cards[0].name_label.sizeHint().width() <= chip_text_width
    )
    assert (
        win.inspector._summary_cards[0].mass_label.sizeHint().width() <= chip_text_width
    )
    assert (
        win.inspector._summary_cards[0].config_label.sizeHint().width()
        <= chip_text_width
    )
    assert win.inspector.plot_target_combo.isVisible() is True
    assert win.inspector.plot_target_combo.currentData() == 2
    assert win.inspector.plot_target_combo.width() == 72
    assert [
        win.inspector.plot_target_combo.itemText(index)
        for index in range(win.inspector.plot_target_combo.count())
    ] == ["H", "He"]
    assert win.inspector.cross_section_plot.height() == plot_height_single
    assert win.inspector.summary_frame.height() == one_row_summary_height
    assert (
        2 * CompactElementChip._DETAILED_HEIGHT
        + win.inspector.summary_cards_grid.verticalSpacing()
        == 3 * CompactElementChip._COMPACT_SIZE.height()
        + 2 * win.inspector.summary_cards_grid.verticalSpacing()
    )

    win.inspector.plot_target_combo.setCurrentIndex(0)

    assert win.inspector.plot_target_combo.currentData() == 1
    assert win.inspector.cross_section_plot._last_state == ("H", "orbital", 80.0, 1)

    win._handle_card_selected(3, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (1, 2, 3)
    assert win.selected_atomic_number == 3
    assert win.inspector.plot_target_combo.currentData() == 1
    assert (
        win.inspector._summary_cards[0].height() == CompactElementChip._DETAILED_HEIGHT
    )
    assert win.inspector._summary_cards[0].width() == 92
    assert win.inspector.summary_frame.width() == 320
    assert win.inspector.summary_frame.height() == one_row_summary_height
    third_card_index = win.inspector.summary_cards_grid.indexOf(
        win.inspector._summary_cards[2]
    )
    assert win.inspector.summary_cards_grid.getItemPosition(third_card_index)[:2] == (
        0,
        2,
    )

    win._handle_card_hovered(4)

    assert win.current_record.symbol == "Li"
    assert (
        win.inspector.summary_stack.currentWidget() is win.inspector.summary_cards_page
    )
    assert win.inspector.mode_label.text() == "3 selected"
    assert win.inspector.plot_target_combo.currentData() == 1

    win._handle_card_selected(4, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (1, 2, 3, 4)
    assert win.inspector.mode_label.text() == "4 selected"
    assert (
        win.inspector._summary_cards[0].height() == CompactElementChip._DETAILED_HEIGHT
    )
    assert win.inspector._summary_cards[0].width() == 92
    assert win.inspector._summary_cards[0].mass_label.isHidden() is False
    assert win.inspector._summary_cards[0].config_label.isHidden() is False
    assert win.inspector.summary_frame.height() == two_row_summary_height
    fourth_card_index = win.inspector.summary_cards_grid.indexOf(
        win.inspector._summary_cards[3]
    )
    assert win.inspector.summary_cards_grid.getItemPosition(fourth_card_index)[:2] == (
        1,
        0,
    )
    QtWidgets.QApplication.processEvents()
    summary_viewport = win.inspector.summary_cards_scroll.viewport()
    assert summary_viewport is not None
    assert (
        win.inspector._summary_cards[3].geometry().bottom() <= summary_viewport.height()
    )
    horizontal_gap = win.inspector._summary_cards[1].geometry().left() - (
        win.inspector._summary_cards[0].geometry().left()
        + win.inspector._summary_cards[0].width()
    )
    vertical_gap = win.inspector._summary_cards[3].geometry().top() - (
        win.inspector._summary_cards[0].geometry().top()
        + win.inspector._summary_cards[0].height()
    )
    assert horizontal_gap == vertical_gap
    assert horizontal_gap == win.inspector.summary_cards_grid.horizontalSpacing()
    assert win.inspector.plot_target_combo.currentData() == 1

    win._handle_card_selected(5, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (1, 2, 3, 4, 5)
    assert win.inspector.mode_label.text() == "5 selected"
    assert (
        win.inspector._summary_cards[0].height() == CompactElementChip._DETAILED_HEIGHT
    )
    assert win.inspector._summary_cards[0].mass_label.isHidden() is False
    assert win.inspector._summary_cards[0].config_label.isHidden() is False

    win._handle_card_selected(7, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (1, 2, 3, 4, 5, 7)
    assert win.inspector.mode_label.text() == "6 selected"
    assert (
        win.inspector._summary_cards[0].height() == CompactElementChip._DETAILED_HEIGHT
    )
    assert win.inspector._summary_cards[0].mass_label.isHidden() is False
    assert win.inspector._summary_cards[0].config_label.isHidden() is False
    assert win.inspector.plot_target_combo.currentData() == 1

    win._handle_card_selected(6, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (1, 2, 3, 4, 5, 7, 6)
    assert win.inspector.mode_label.text() == "7 selected"
    assert win.inspector._summary_cards[0].height() == 58
    assert win.inspector._summary_cards[0].width() == 92
    assert win.inspector._summary_cards[0].mass_label.isHidden() is True
    assert win.inspector._summary_cards[0].config_label.isHidden() is True
    assert win.inspector.plot_target_combo.currentData() == 1
    seventh_card_index = win.inspector.summary_cards_grid.indexOf(
        win.inspector._summary_cards[6]
    )
    assert win.inspector.summary_cards_grid.getItemPosition(seventh_card_index)[:2] == (
        2,
        0,
    )

    win._handle_card_selected(6, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (1, 2, 3, 4, 5, 7)
    assert win.inspector.mode_label.text() == "6 selected"
    assert (
        win.inspector._summary_cards[0].height() == CompactElementChip._DETAILED_HEIGHT
    )
    assert win.inspector._summary_cards[0].mass_label.isHidden() is False
    assert win.inspector._summary_cards[0].config_label.isHidden() is False
    assert win.inspector.plot_target_combo.currentData() == 1

    win._handle_card_selected(4, QtCore.Qt.KeyboardModifier.ControlModifier)
    win._handle_card_selected(5, QtCore.Qt.KeyboardModifier.ControlModifier)
    win._handle_card_selected(7, QtCore.Qt.KeyboardModifier.ControlModifier)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (2, 3)
    assert win.inspector.plot_target_combo.currentData() == 3
    assert (
        win.inspector._summary_cards[0].height() == CompactElementChip._DETAILED_HEIGHT
    )

    win._handle_card_selected(2, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (3,)
    assert win.selected_atomic_number == 3
    assert win.inspector.plot_target_combo.isVisible() is False
    assert win.inspector.cross_section_plot.height() == plot_height_single

    win._handle_card_selected(3, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == ()
    assert win.selected_atomic_number is None
    assert (
        win.inspector.summary_stack.currentWidget() is win.inspector.summary_empty_page
    )
    assert win.inspector.mode_label.text() == "No selection"

    win.close()


def test_ptable_multi_select_levels_table_clipboard_and_notation(
    qtbot,
    monkeypatch,
) -> None:
    copied: list[str] = []
    binding_levels = {
        "H": {"2p3/2": 3.5, "1s": 12.0},
        "He": {"2p1/2": 7.0},
        "Li": {},
    }

    monkeypatch.setattr(
        erlab.analysis.xps,
        "get_edge",
        lambda symbol: binding_levels.get(symbol, {"1s": 5.0}),
    )
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "copy_to_clipboard",
        lambda content: copied.append(
            "\n".join(content) if isinstance(content, list) else content
        ),
    )

    win = PeriodicTableWindow(photon_energy=20.0, workfunction=1.0)
    _show_window(qtbot, win)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)

    win._handle_card_selected(2, QtCore.Qt.KeyboardModifier.ControlModifier)
    win._handle_card_selected(3, QtCore.Qt.KeyboardModifier.ControlModifier)
    win.max_harmonic_spin.setValue(2)

    assert win.selected_atomic_numbers == (1, 2, 3)
    assert win.inspector.levels_table.horizontalHeader().isVisible() is True
    assert win.inspector.levels_table.columnCount() == 3
    assert win.inspector.levels_table.rowCount() == 9
    assert [
        win.inspector.levels_table.horizontalHeaderItem(index).text()
        for index in range(win.inspector.levels_table.columnCount())
    ] == ["1s", "2p1/2", "2p3/2"]
    assert isinstance(
        win.inspector.levels_table.horizontalHeader(),
        RichTextHeaderView,
    )
    assert (
        win.inspector.levels_table.horizontalHeaderItem(0).data(
            QtCore.Qt.ItemDataRole.UserRole
        )
        == "1<i>s</i>"
    )
    assert (
        win.inspector.levels_table.horizontalHeaderItem(1).data(
            QtCore.Qt.ItemDataRole.UserRole
        )
        == "2<i>p</i><sub>1/2</sub>"
    )
    assert win.inspector.levels_table.verticalHeaderItem(0).text() == "H"
    assert win.inspector.levels_table.verticalHeaderItem(1).text() == "H KE (1hv)"
    assert win.inspector.levels_table.verticalHeaderItem(2).text() == "H KE (2hv)"
    assert win.inspector.levels_table.verticalHeaderItem(6).text() == "Li"
    assert win.inspector.levels_table.item(0, 0).text() == "12"
    assert (
        win.inspector.levels_table.item(0, 0).background().color()
        == win.inspector._theme.edge_match_bg
    )
    assert win.inspector.levels_table.item(1, 0).text() == "7"
    assert (
        win.inspector.levels_table.item(1, 0).background().color()
        == win.inspector._theme.harmonic_match_bgs[0]
    )
    assert win.inspector.levels_table.item(2, 0).text() == "27"
    assert (
        win.inspector.levels_table.item(2, 0).background().color()
        == win.inspector._theme.harmonic_match_bgs[1]
    )
    assert win.inspector.levels_table.item(3, 0).text() == ""
    assert win.inspector.levels_table.item(4, 1).text() == "12"
    assert win.inspector.levels_table.item(5, 1).text() == "32"
    assert win.inspector.levels_table.item(6, 2).text() == ""
    assert win.inspector.levels_table.item(8, 2).text() == ""

    qtbot.mouseClick(win.inspector.copy_values_button, QtCore.Qt.MouseButton.LeftButton)
    qtbot.mouseClick(win.inspector.copy_table_button, QtCore.Qt.MouseButton.LeftButton)

    assert copied[0].splitlines()[0] == "Element\tMetric\t1s\t2p1/2\t2p3/2"
    assert copied[0].splitlines()[1] == "H\tAbsorption edge (eV)\t12\t\t3.5"
    assert copied[0].splitlines()[3] == "Li\tAbsorption edge (eV)\t\t\t"
    assert copied[1].splitlines()[3] == "H\tKinetic energy @ 2hv (eV)\t27\t\t35.5"
    assert copied[1].splitlines()[5] == "He\tKinetic energy @ 1hv (eV)\t\t12\t"

    win.notation_combo.setCurrentIndex(win.notation_combo.findData("iupac"))

    assert [
        win.inspector.levels_table.horizontalHeaderItem(index).text()
        for index in range(win.inspector.levels_table.columnCount())
    ] == ["K", "L2", "L3"]
    assert (
        win.inspector.levels_table.horizontalHeaderItem(1).data(
            QtCore.Qt.ItemDataRole.UserRole
        )
        == "L<sub>2</sub>"
    )

    win.close()


def test_ptable_harmonic_plot_markers_refresh_when_only_order_changes(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 5.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow(photon_energy=80.0)
    _show_window(qtbot, win)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)

    plot = win.inspector.cross_section_plot
    assert plot._last_state == ("H", "orbital", 80.0, 1)
    assert plot.photon_line_energies == (80.0,)

    win.max_harmonic_spin.setValue(2)

    assert plot._last_state == ("H", "orbital", 80.0, 2)
    assert plot.photon_line_energy == 80.0
    assert plot.photon_line_energies == (80.0, 160.0)

    win.close()


def test_ptable_search_autocomplete_updates_selection(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.NoModifier)

    win.search_edit.setText("gol")

    assert win.selected_atomic_numbers == (1,)
    assert win.search_completer.popup().isVisible() is False
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 0
    assert _search_completion_texts(win)[0] == "Au - Gold"

    completion_index = win.search_completer.model().index(0, 0)
    win.search_completer.activated[QtCore.QModelIndex].emit(completion_index)

    assert win.search_edit.text() == "Au"
    assert win.selected_atomic_numbers == (79,)
    assert win.selected_atomic_number == 79
    assert win.periodic_table.cards[79].is_selected is True
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 0
    assert win.periodic_table.cards[79].is_search_match is False
    assert _search_completion_texts(win) == ["Au - Gold"]
    assert win.table_view.hasFocus() is True

    win.close()


def test_ptable_search_activation_scrolls_selected_element_into_view(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win.resize(900, 650)
    QtWidgets.QApplication.processEvents()

    helium = win.periodic_table.cards[2]
    initial_scroll_values = (
        win.table_view.horizontalScrollBar().value(),
        win.table_view.verticalScrollBar().value(),
    )
    assert _table_child_center_is_visible_in_viewport(win, helium) is False

    win.search_edit.setText("heli")
    completion_index = win.search_completer.model().index(0, 0)
    win.search_completer.activated[QtCore.QModelIndex].emit(completion_index)

    qtbot.waitUntil(lambda: win.selected_atomic_number == 2)
    qtbot.waitUntil(lambda: _table_child_center_is_visible_in_viewport(win, helium))
    assert (
        win.table_view.horizontalScrollBar().value(),
        win.table_view.verticalScrollBar().value(),
    ) != initial_scroll_values

    win.close()


def test_ptable_search_popup_mouse_click_selects_without_keyboard_highlight(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win.search_edit.setFocus()
    qtbot.keyClicks(win.search_edit, "gol")

    popup = win.search_completer.popup()
    qtbot.waitUntil(popup.isVisible)
    assert popup.currentIndex().isValid() is False

    first_item = popup.item(0)
    assert first_item is not None
    qtbot.mouseClick(
        popup.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=popup.visualItemRect(first_item).center(),
    )

    qtbot.waitUntil(lambda: win.selected_atomic_numbers == (79,))
    assert win.search_edit.text() == "Au"
    assert popup.isVisible() is False

    win.close()


def test_ptable_search_popup_mouse_click_selects_via_item_view_press_signal(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    def _base_mouse_press_event(
        self: QtWidgets.QListWidget,
        event: QtGui.QMouseEvent | None,
    ) -> None:
        QtWidgets.QListWidget.mousePressEvent(self, event)

    monkeypatch.setattr(
        erlab.interactive.ptable._window._SearchPopup,
        "mousePressEvent",
        _base_mouse_press_event,
    )

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win.search_edit.setFocus()
    qtbot.keyClicks(win.search_edit, "gol")

    popup = win.search_completer.popup()
    qtbot.waitUntil(popup.isVisible)

    first_item = popup.item(0)
    assert first_item is not None
    qtbot.mouseClick(
        popup.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=popup.visualItemRect(first_item).center(),
    )

    qtbot.waitUntil(lambda: win.selected_atomic_numbers == (79,))
    assert win.search_edit.text() == "Au"
    assert popup.isVisible() is False

    win.close()


def test_ptable_search_popup_mouse_click_ignores_event_filter_geometry_miss(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)
    monkeypatch.setattr(
        erlab.interactive.ptable._window._SearchPopup,
        "frameGeometry",
        lambda self: QtCore.QRect(-1, -1, 0, 0),
    )

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win.search_edit.setFocus()
    qtbot.keyClicks(win.search_edit, "gol")

    popup = win.search_completer.popup()
    qtbot.waitUntil(popup.isVisible)

    first_item = popup.item(0)
    assert first_item is not None
    qtbot.mouseClick(
        popup.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=popup.visualItemRect(first_item).center(),
    )

    qtbot.waitUntil(lambda: win.selected_atomic_numbers == (79,))
    assert win.search_edit.text() == "Au"
    assert popup.isVisible() is False

    win.close()


def test_ptable_search_popup_hover_tracks_and_clears_active_completion(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win.search_edit.setFocus()
    qtbot.keyClicks(win.search_edit, "cu")

    popup = win.search_completer.popup()
    qtbot.waitUntil(popup.isVisible)
    assert popup.currentRow() == -1
    assert win.periodic_table.cards[29].is_search_match is True
    assert win.periodic_table.cards[96].is_search_match is True
    assert win.periodic_table.cards[80].is_search_match is True

    second_item = popup.item(1)
    assert second_item is not None
    qtbot.mouseMove(popup.viewport(), popup.visualItemRect(second_item).center())

    qtbot.waitUntil(lambda: popup.currentRow() == 1)
    assert win.periodic_table.cards[29].is_search_match is False
    assert win.periodic_table.cards[96].is_search_match is True
    assert win.periodic_table.cards[80].is_search_match is False

    qtbot.mouseMove(win.search_edit, win.search_edit.rect().center())

    qtbot.waitUntil(lambda: popup.currentRow() == -1)
    assert win.periodic_table.cards[29].is_search_match is True
    assert win.periodic_table.cards[96].is_search_match is True
    assert win.periodic_table.cards[80].is_search_match is True

    win.close()


def test_ptable_search_popup_mouse_click_matches_keyboard_activation(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win.search_edit.setFocus()
    qtbot.keyClicks(win.search_edit, "cu")

    popup = win.search_completer.popup()
    qtbot.waitUntil(popup.isVisible)
    qtbot.keyClick(win.search_edit, QtCore.Qt.Key.Key_Down)

    second_item = popup.item(1)
    assert second_item is not None
    qtbot.mouseClick(
        popup.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=popup.visualItemRect(second_item).center(),
    )

    qtbot.waitUntil(lambda: win.selected_atomic_numbers == (96,))
    assert win.search_edit.text() == "Cm"
    assert win.selected_atomic_number == 96
    assert popup.isVisible() is False
    assert win.table_view.hasFocus() is True

    win.close()


def test_ptable_symbol_list_search_selects_multiple_elements(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win.search_edit.setFocus()
    qtbot.keyClicks(win.search_edit, "f cl br")

    popup = win.search_completer.popup()
    qtbot.waitUntil(popup.isVisible)

    assert _search_completion_texts(win)[0] == "Select: F, Cl, Br"
    assert win.periodic_table.cards[9].is_search_match is True
    assert win.periodic_table.cards[17].is_search_match is True
    assert win.periodic_table.cards[35].is_search_match is True
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 3

    completion_index = win.search_completer.model().index(0, 0)
    win.search_completer.activated[QtCore.QModelIndex].emit(completion_index)

    assert win.search_edit.text() == "F, Cl, Br"
    assert win.selected_atomic_numbers == (9, 17, 35)
    assert win.selected_atomic_number == 35
    assert win.periodic_table.cards[9].is_selected is True
    assert win.periodic_table.cards[17].is_selected is True
    assert win.periodic_table.cards[35].is_selected is True
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 0

    win.close()


def test_ptable_search_keyboard_completion_tracks_active_target(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win.search_edit.setFocus()
    qtbot.keyClicks(win.search_edit, "cu")

    popup = win.search_completer.popup()
    qtbot.waitUntil(popup.isVisible)

    assert _search_completion_texts(win)[:3] == [
        "Cu - Copper",
        "Cm - Curium",
        "Hg - Mercury",
    ]
    assert popup.currentIndex().isValid() is False
    assert win.periodic_table.cards[29].is_search_match is True
    assert win.periodic_table.cards[96].is_search_match is True
    assert win.periodic_table.cards[80].is_search_match is True
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 3

    qtbot.keyClick(win.search_edit, QtCore.Qt.Key.Key_Down)

    qtbot.waitUntil(lambda: win._search_completion_row == 0)
    assert popup.currentIndex().isValid() is True
    assert popup.currentIndex().row() == 0
    assert win.periodic_table.cards[29].is_search_match is True
    assert win.periodic_table.cards[96].is_search_match is False
    assert win.periodic_table.cards[80].is_search_match is False
    assert popup.isVisible() is True

    qtbot.keyClick(win.search_edit, QtCore.Qt.Key.Key_Return)

    qtbot.waitUntil(lambda: win.selected_atomic_number == 29)
    assert win.selected_atomic_numbers == (29,)
    assert win.periodic_table.cards[29].is_selected is True
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 0
    assert win.search_edit.text() == "Cu"

    win.close()


def test_ptable_search_highlight_clears_when_dropdown_closes(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win.search_edit.setFocus()
    qtbot.keyClicks(win.search_edit, "cu")

    popup = win.search_completer.popup()
    qtbot.waitUntil(popup.isVisible)
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 3

    win.table_view.setFocus()

    qtbot.waitUntil(lambda: popup.isVisible() is False)
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 0

    win.search_edit.setFocus()

    qtbot.waitUntil(popup.isVisible)
    assert win.periodic_table.cards[29].is_search_match is True
    assert win.periodic_table.cards[96].is_search_match is True
    assert win.periodic_table.cards[80].is_search_match is True
    assert sum(card.is_search_match for card in win.periodic_table.cards.values()) == 3

    win.close()


def test_ptable_find_shortcut_focuses_search_bar(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win.search_edit.setText("cu")
    qtbot.waitUntil(win.search_completer.popup().isVisible)

    win.table_view.setFocus()
    qtbot.waitUntil(lambda: win.search_completer.popup().isVisible() is False)
    assert win.table_view.hasFocus() is True

    win.find_shortcut.activated.emit()

    qtbot.waitUntil(win.search_edit.hasFocus)
    qtbot.waitUntil(win.search_completer.popup().isVisible)
    assert win.search_edit.selectedText() == "cu"

    win.close()


def test_ptable_close_shortcut_hides_window(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    class _TrackingPeriodicTableWindow(PeriodicTableWindow):
        def __init__(self) -> None:
            self.close_event_count = 0
            super().__init__()

        def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
            self.close_event_count += 1
            super().closeEvent(event)

    win = _TrackingPeriodicTableWindow()
    _show_window(qtbot, win)

    win.search_edit.setFocus()
    qtbot.waitUntil(win.search_edit.hasFocus)
    assert (
        win.close_shortcut.key().toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        == "Ctrl+W"
    )

    qtbot.keyClick(
        win.search_edit,
        QtCore.Qt.Key.Key_W,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )

    qtbot.waitUntil(lambda: win.isVisible() is False)
    assert win.close_event_count == 0

    win.close()


def test_ptable_keyboard_navigation_and_background_clear(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win.table_view.setFocus()
    qtbot.keyClick(win.table_view, QtCore.Qt.Key.Key_Right)

    assert win.selected_atomic_numbers == (1,)
    assert win.selected_atomic_number == 1
    assert win.periodic_table.cards[1].is_current is True

    qtbot.keyClick(win.table_view, QtCore.Qt.Key.Key_Right)

    assert win.selected_atomic_numbers == (2,)
    assert win.selected_atomic_number == 2

    qtbot.keyClick(
        win.table_view,
        QtCore.Qt.Key.Key_Right,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
    )

    assert win.selected_atomic_numbers == (2, 3)
    assert win.selected_atomic_number == 3
    assert win.periodic_table.cards[3].is_current is True
    assert win.periodic_table.cards[3]._border_width == 5

    background_pos = _viewport_pos_for_table_child(
        win, win.periodic_table.group_labels[5]
    )
    qtbot.mouseClick(
        win.table_view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=background_pos,
    )

    assert win.selected_atomic_numbers == ()
    assert win.selected_atomic_number is None
    assert win.inspector.mode_label.text() == "No selection"

    qtbot.keyClick(win.table_view, QtCore.Qt.Key.Key_Right)
    assert win.selected_atomic_numbers == (1,)

    qtbot.keyClick(win.table_view, QtCore.Qt.Key.Key_Escape)

    assert win.selected_atomic_numbers == ()
    assert win.selected_atomic_number is None

    win.close()


def test_ptable_keyboard_navigation_scrolls_current_element_into_view(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)
    win.resize(900, 650)
    QtWidgets.QApplication.processEvents()

    initial_scroll_values = (
        win.table_view.horizontalScrollBar().value(),
        win.table_view.verticalScrollBar().value(),
    )

    win.table_view.navigate_requested.emit(
        int(QtCore.Qt.Key.Key_Left),
        QtCore.Qt.KeyboardModifier.NoModifier,
    )

    qtbot.waitUntil(lambda: win.selected_atomic_number is not None)
    assert win.selected_atomic_number is not None
    assert (
        win.table_view.horizontalScrollBar().value(),
        win.table_view.verticalScrollBar().value(),
    ) != initial_scroll_values
    assert _table_child_center_is_visible_in_viewport(
        win, win.periodic_table.cards[win.selected_atomic_number]
    )

    win.close()


def test_ptable_shift_click_selects_rectangular_region(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win._handle_card_selected(79, QtCore.Qt.KeyboardModifier.NoModifier)
    win._handle_card_selected(46, QtCore.Qt.KeyboardModifier.ShiftModifier)

    assert win.selected_atomic_numbers == (46, 47, 78, 79)
    assert win.selected_atomic_number == 46
    assert win.periodic_table.cards[46].is_selected is True
    assert win.periodic_table.cards[47].is_selected is True
    assert win.periodic_table.cards[78].is_selected is True
    assert win.periodic_table.cards[79].is_selected is True
    assert win.periodic_table.cards[48].is_selected is False
    assert win.inspector.mode_label.text() == "4 selected"

    win.close()


def test_ptable_summary_chips_are_sorted_by_atomic_number(
    qtbot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(erlab.analysis.xps, "get_edge", lambda _symbol: {"1s": 10.0})
    monkeypatch.setattr(erlab.analysis.xps, "get_cross_section", _fake_cross_sections)

    win = PeriodicTableWindow()
    _show_window(qtbot, win)

    win._handle_card_selected(3, QtCore.Qt.KeyboardModifier.NoModifier)
    win._handle_card_selected(1, QtCore.Qt.KeyboardModifier.ControlModifier)
    win._handle_card_selected(2, QtCore.Qt.KeyboardModifier.ControlModifier)

    assert win.selected_atomic_numbers == (3, 1, 2)
    assert [card.record.atomic_number for card in win.inspector._summary_cards] == [
        1,
        2,
        3,
    ]
    assert [card.symbol_label.text() for card in win.inspector._summary_cards] == [
        "H",
        "He",
        "Li",
    ]

    win.close()


def test_ptable_module_main(monkeypatch) -> None:
    called: list[bool] = []

    monkeypatch.setattr(
        erlab.interactive.ptable.__main__,
        "ptable",
        lambda: called.append(True),
    )

    erlab.interactive.ptable.__main__.main()

    assert called == [True]
