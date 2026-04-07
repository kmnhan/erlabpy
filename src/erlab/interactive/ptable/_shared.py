from __future__ import annotations

import functools
import html
import re
from dataclasses import dataclass

import numpy as np
import xraydb
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.ptable._metadata import (
    ELEMENT_CATEGORIES,
    ELEMENT_POSITIONS,
    GROUND_STATE_CONFIGURATIONS,
)

_EDGE_IUPAC_LABELS = {
    value: key for key, value in erlab.analysis.xps.IUPAC_TO_XPS.items()
}
_CROSS_SECTION_RE = re.compile(r"(?P<n>\d+)(?P<orb>[spdfg])")
_ORBITAL_ORDER = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
_SHELL_LETTERS = "KLMNOPQ"
_IUPAC_GROUP_INDEX = {"s": "1", "p": "2,3", "d": "4,5", "f": "6,7", "g": "8,9"}
_INSPECTOR_WIDTH = 560
_SYMBOL_LIST_SPLIT_RE = re.compile(r"[\s,]+")
_LEGEND_CATEGORY_ORDER = [
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
]
_XPS_LABEL_RE = re.compile(r"(?P<n>\d+)(?P<orb>[spdfg])(?P<j>\d+(?:/\d+)?)?$")
_IUPAC_LABEL_RE = re.compile(r"(?P<shell>[KLMNOPQ])(?P<index>\d+(?:,\d+)?)?$")


@dataclass(frozen=True)
class ThemeColors:
    is_dark: bool
    window: QtGui.QColor
    panel: QtGui.QColor
    panel_alt: QtGui.QColor
    table_surface: QtGui.QColor
    input_bg: QtGui.QColor
    border: QtGui.QColor
    border_soft: QtGui.QColor
    text: QtGui.QColor
    muted_text: QtGui.QColor
    disabled_text: QtGui.QColor
    accent: QtGui.QColor
    hover_accent: QtGui.QColor
    search_accent: QtGui.QColor
    invalid_bg: QtGui.QColor
    invalid_border: QtGui.QColor
    edge_match_bg: QtGui.QColor
    harmonic_match_bgs: tuple[QtGui.QColor, ...]
    plot_grid: QtGui.QColor
    plot_total: QtGui.QColor
    plot_marker: QtGui.QColor


@dataclass(frozen=True)
class ElementRecord:
    atomic_number: int
    symbol: str
    name: str
    mass: float
    category: str
    row: int
    column: int
    configuration: str


@dataclass(frozen=True)
class CrossSectionSeries:
    subshell: str
    hv: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class CrossSectionRenderData:
    series: tuple[CrossSectionSeries, ...]
    total_hv: np.ndarray | None
    total_sigma: np.ndarray | None
    empty_message: str | None = None


@dataclass(frozen=True)
class _FittedSymbolFont:
    font: QtGui.QFont
    top_margin: int


def _effective_point_size(font: QtGui.QFont) -> float:
    point_size = font.pointSizeF()
    if point_size > 0.0:
        return point_size
    resolved = float(QtGui.QFontInfo(font).pointSizeF())
    return resolved if resolved > 0.0 else 12.0


def _fit_symbol_font(
    base_font: QtGui.QFont,
    text: str,
    *,
    max_width: float,
    max_height: float,
    preferred_point_size: float,
    minimum_point_size: float,
    step: float = 0.2,
) -> _FittedSymbolFont:
    fitted_font = QtGui.QFont(base_font)
    fitted_font.setBold(True)
    if text == "":
        fitted_font.setPointSizeF(max(preferred_point_size, minimum_point_size))
        return _FittedSymbolFont(fitted_font, 0)

    width_limit = max(1.0, max_width)
    height_limit = max(1.0, max_height)
    point_size = max(preferred_point_size, minimum_point_size)

    while point_size >= minimum_point_size - 1e-6:
        fitted_font.setPointSizeF(point_size)
        ink_rect = QtGui.QFontMetricsF(fitted_font).tightBoundingRect(text)
        if (
            ink_rect.width() <= width_limit + 1e-6
            and ink_rect.height() <= height_limit + 1e-6
        ):
            break
        point_size -= step
    else:
        fitted_font.setPointSizeF(minimum_point_size)

    metrics = QtGui.QFontMetricsF(fitted_font)
    ink_rect = metrics.tightBoundingRect(text)
    bounding_rect = metrics.boundingRect(text)
    top_headroom = max(0.0, ink_rect.top() - bounding_rect.top())
    max_negative_top_margin = max(1, round(height_limit * 0.08))
    top_margin = -min(max_negative_top_margin, max(0, round(top_headroom * 0.2)))
    return _FittedSymbolFont(fitted_font, top_margin)


def _blend_colors(
    base: QtGui.QColor,
    overlay: QtGui.QColor,
    ratio: float,
) -> QtGui.QColor:
    ratio = min(max(ratio, 0.0), 1.0)
    inverse = 1.0 - ratio
    return QtGui.QColor(
        int(base.red() * inverse + overlay.red() * ratio),
        int(base.green() * inverse + overlay.green() * ratio),
        int(base.blue() * inverse + overlay.blue() * ratio),
        int(base.alpha() * inverse + overlay.alpha() * ratio),
    )


def _boost_saturation(color: QtGui.QColor, amount: float) -> QtGui.QColor:
    amount = min(max(amount, 0.0), 1.0)
    hsv_color = QtGui.QColor(color).convertTo(QtGui.QColor.Spec.Hsv)
    hue, saturation, value, alpha = hsv_color.getHsv()
    if hue is None or saturation is None or value is None or alpha is None or hue < 0:
        return QtGui.QColor(color)
    boosted_saturation = min(255, int(saturation + (255 - saturation) * amount))
    return QtGui.QColor.fromHsv(hue, boosted_saturation, value, alpha)


def _css_rgba(color: QtGui.QColor) -> str:
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()})"


def _set_foreground(widget: QtWidgets.QWidget, color: QtGui.QColor) -> None:
    palette = QtGui.QPalette(widget.palette())
    for role in (
        QtGui.QPalette.ColorRole.WindowText,
        QtGui.QPalette.ColorRole.Text,
        QtGui.QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(role, color)
    widget.setPalette(palette)


def _set_background(
    widget: QtWidgets.QWidget,
    color: QtGui.QColor,
    *,
    role: QtGui.QPalette.ColorRole = QtGui.QPalette.ColorRole.Window,
) -> None:
    palette = QtGui.QPalette(widget.palette())
    palette.setColor(role, color)
    widget.setPalette(palette)
    widget.setAutoFillBackground(role == QtGui.QPalette.ColorRole.Window)


def _palette_accent(palette: QtGui.QPalette) -> QtGui.QColor:
    if hasattr(QtGui.QPalette.ColorRole, "Accent"):  # pragma: no branch
        return palette.accent().color()
    return palette.color(QtGui.QPalette.ColorRole.Highlight)


def _theme_colors() -> ThemeColors:
    palette = QtWidgets.QApplication.palette()
    window = palette.color(QtGui.QPalette.ColorRole.Window)
    base = palette.color(QtGui.QPalette.ColorRole.Base)
    text = palette.color(QtGui.QPalette.ColorRole.WindowText)
    border_source = palette.color(QtGui.QPalette.ColorRole.Mid)
    accent = _palette_accent(palette)
    placeholder = palette.color(QtGui.QPalette.ColorRole.PlaceholderText)
    disabled = palette.color(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.WindowText,
    )

    palette_is_dark = text.lightness() > window.lightness()
    is_dark = erlab.interactive.colors.is_dark_mode() or palette_is_dark

    muted_text = (
        placeholder
        if placeholder.isValid() and placeholder != text
        else _blend_colors(text, window, 0.42 if is_dark else 0.5)
    )
    panel = _blend_colors(base, text, 0.06 if is_dark else 0.015)
    panel_alt = _blend_colors(window, text, 0.12 if is_dark else 0.03)
    table_surface = _blend_colors(
        window,
        QtGui.QColor("#000000") if is_dark else QtGui.QColor("#ffffff"),
        0.14 if is_dark else 0.34,
    )
    input_bg = _blend_colors(base, text, 0.03 if is_dark else 0.0)
    border = _blend_colors(border_source, text, 0.18 if is_dark else 0.08)
    border_soft = _blend_colors(border, panel, 0.35 if is_dark else 0.25)
    hover_accent = _blend_colors(accent, QtGui.QColor("#14b8a6"), 0.35)
    search_accent = _blend_colors(accent, QtGui.QColor("#f59e0b"), 0.55)
    invalid_border = _blend_colors(accent, QtGui.QColor("#ef4444"), 0.85)
    invalid_bg = QtGui.QColor("#3f1d1d") if is_dark else QtGui.QColor("#fef2f2")
    edge_match_bg = QtGui.QColor("#16351f") if is_dark else QtGui.QColor("#dcfce7")
    harmonic_match_bgs = tuple(
        QtGui.QColor(color)
        for color in (
            (
                "#172554",
                "#431407",
                "#2e1065",
                "#500724",
                "#083344",
                "#1a2e05",
                "#450a0a",
                "#4a044e",
                "#042f2e",
                "#1e1b4b",
            )
            if is_dark
            else (
                "#dbeafe",
                "#ffedd5",
                "#ede9fe",
                "#fce7f3",
                "#cffafe",
                "#ecfccb",
                "#fee2e2",
                "#fae8ff",
                "#ccfbf1",
                "#e0e7ff",
            )
        )
    )
    plot_grid = QtGui.QColor(border)
    plot_grid.setAlpha(120 if is_dark else 90)

    return ThemeColors(
        is_dark=is_dark,
        window=window,
        panel=panel,
        panel_alt=panel_alt,
        table_surface=table_surface,
        input_bg=input_bg,
        border=border,
        border_soft=border_soft,
        text=text,
        muted_text=muted_text,
        disabled_text=disabled,
        accent=accent,
        hover_accent=hover_accent,
        search_accent=search_accent,
        invalid_bg=invalid_bg,
        invalid_border=invalid_border,
        edge_match_bg=edge_match_bg,
        harmonic_match_bgs=harmonic_match_bgs,
        plot_grid=plot_grid,
        plot_total=text,
        plot_marker=QtGui.QColor("#ef4444"),
    )


def _chip_secondary_text_color(theme: ThemeColors) -> QtGui.QColor:
    if not theme.is_dark:
        return theme.muted_text
    return _blend_colors(theme.muted_text, theme.text, 0.28)


def _chip_detail_text_color(theme: ThemeColors) -> QtGui.QColor:
    secondary = _chip_secondary_text_color(theme)
    return _blend_colors(secondary, theme.text, 0.25)


def _format_mass(value: float) -> str:
    return np.format_float_positional(float(value), precision=6, trim="-")


def _format_energy(value: float) -> str:
    return np.format_float_positional(float(value), precision=4, trim="-")


def _parse_positive_float(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0.0 else None


def _cross_section_sort_key(label: str) -> tuple[int, int]:
    match = _CROSS_SECTION_RE.fullmatch(label)
    if match is None:
        return (999, 999)
    return (int(match.group("n")), _ORBITAL_ORDER[match.group("orb")])


def _aggregate_iupac_label(subshell: str) -> str:
    match = _CROSS_SECTION_RE.fullmatch(subshell)
    if match is None:
        return subshell
    shell_index = int(match.group("n")) - 1
    shell = (
        _SHELL_LETTERS[shell_index] if 0 <= shell_index < len(_SHELL_LETTERS) else "?"
    )
    return f"{shell}{_IUPAC_GROUP_INDEX[match.group('orb')]}"


def _edge_label(label: str, notation: str) -> str:
    if notation == "iupac":
        return _EDGE_IUPAC_LABELS.get(label, label)
    return label


def _cross_section_label(label: str, notation: str) -> str:
    if notation == "iupac":
        return _aggregate_iupac_label(label)
    return label


def _edge_sort_key(label: str) -> tuple[int, int, float, str]:
    match = _XPS_LABEL_RE.fullmatch(label)
    if match is None:
        return (999, 999, 999.0, label)
    j_value = 0.0
    if match.group("j") is not None:
        numerator, _, denominator = match.group("j").partition("/")
        j_value = float(numerator) / float(denominator or "1")
    return (
        int(match.group("n")),
        _ORBITAL_ORDER[match.group("orb")],
        j_value,
        label,
    )


def _has_keyboard_modifier(
    modifiers: object,
    modifier: QtCore.Qt.KeyboardModifier,
) -> bool:
    raw_modifier_value = getattr(modifiers, "value", None)
    modifier_value = (
        raw_modifier_value
        if isinstance(raw_modifier_value, int)
        else modifiers
        if isinstance(modifiers, int)
        else None
    )
    raw_target_value = getattr(modifier, "value", None)
    if not isinstance(raw_target_value, int):
        return False
    target_value = raw_target_value
    return isinstance(modifier_value, int) and bool(modifier_value & target_value)


def _is_toggle_selection_modifier(modifiers: object) -> bool:
    return _has_keyboard_modifier(
        modifiers, QtCore.Qt.KeyboardModifier.ControlModifier
    ) or _has_keyboard_modifier(modifiers, QtCore.Qt.KeyboardModifier.MetaModifier)


def _search_records(query: str) -> list[ElementRecord]:
    normalized = query.strip().lower()
    if normalized == "":
        return []

    exact_symbol: list[ElementRecord] = []
    prefixed: list[ElementRecord] = []
    substring: list[ElementRecord] = []
    for record in _element_records().values():
        symbol_lower = record.symbol.lower()
        name_lower = record.name.lower()
        symbol_match = normalized in symbol_lower
        name_match = normalized in name_lower
        if not (symbol_match or name_match):
            continue
        if normalized == symbol_lower:
            exact_symbol.append(record)
        elif symbol_lower.startswith(normalized) or name_lower.startswith(normalized):
            prefixed.append(record)
        else:
            substring.append(record)

    return [*exact_symbol, *prefixed, *substring]


def _parse_symbol_selection_query(query: str) -> tuple[ElementRecord, ...]:
    normalized = query.strip()
    if normalized == "":
        return ()

    tokens = [token for token in _SYMBOL_LIST_SPLIT_RE.split(normalized) if token]
    if len(tokens) < 2:
        return ()

    symbol_lookup = {
        record.symbol.lower(): record for record in _element_records().values()
    }
    records: list[ElementRecord] = []
    seen_atomic_numbers: set[int] = set()
    for token in tokens:
        record = symbol_lookup.get(token.lower())
        if record is None:
            return ()
        if record.atomic_number in seen_atomic_numbers:
            continue
        seen_atomic_numbers.add(record.atomic_number)
        records.append(record)

    if len(records) < 2:
        return ()
    return tuple(records)


@functools.cache
def _navigation_atomic_numbers() -> tuple[int, ...]:
    return tuple(
        atomic_number
        for atomic_number, _ in sorted(
            _element_records().items(),
            key=lambda item: (item[1].row, item[1].column, item[0]),
        )
    )


def _selection_range(
    start_atomic_number: int, end_atomic_number: int
) -> tuple[int, ...]:
    navigation = _navigation_atomic_numbers()
    start_index = navigation.index(start_atomic_number)
    end_index = navigation.index(end_atomic_number)
    if start_index <= end_index:
        return navigation[start_index : end_index + 1]
    return navigation[end_index : start_index + 1]


def _rectangular_selection_range(
    start_atomic_number: int, end_atomic_number: int
) -> tuple[int, ...]:
    records = _element_records()
    start_record = records[start_atomic_number]
    end_record = records[end_atomic_number]
    min_row = min(start_record.row, end_record.row)
    max_row = max(start_record.row, end_record.row)
    min_column = min(start_record.column, end_record.column)
    max_column = max(start_record.column, end_record.column)
    return tuple(
        atomic_number
        for atomic_number, record in sorted(
            records.items(),
            key=lambda item: (item[1].row, item[1].column, item[0]),
        )
        if (
            min_row <= record.row <= max_row
            and min_column <= record.column <= max_column
        )
    )


def _adjacent_atomic_number(
    current_atomic_number: int,
    key: int,
) -> int | None:
    records = _element_records()
    current_record = records[current_atomic_number]
    direction = {
        int(QtCore.Qt.Key.Key_Left): (-1, 0),
        int(QtCore.Qt.Key.Key_Right): (1, 0),
        int(QtCore.Qt.Key.Key_Up): (0, -1),
        int(QtCore.Qt.Key.Key_Down): (0, 1),
    }.get(key)
    if direction is None:
        return None
    dx, dy = direction
    if dy == 0:
        same_row_candidates = [
            record
            for record in records.values()
            if record.row == current_record.row
            and (
                (dx < 0 and record.column < current_record.column)
                or (dx > 0 and record.column > current_record.column)
            )
        ]
        if same_row_candidates:
            return (
                max(same_row_candidates, key=lambda record: record.column).atomic_number
                if dx < 0
                else min(
                    same_row_candidates, key=lambda record: record.column
                ).atomic_number
            )
    else:
        vertical_candidates = [
            record
            for record in records.values()
            if (dy < 0 and record.row < current_record.row)
            or (dy > 0 and record.row > current_record.row)
        ]
        if vertical_candidates:
            return min(
                vertical_candidates,
                key=lambda record: (
                    abs(record.row - current_record.row),
                    abs(record.column - current_record.column),
                    record.row,
                    record.column,
                ),
            ).atomic_number

    navigation = _navigation_atomic_numbers()
    current_index = navigation.index(current_atomic_number)
    neighbor_index = current_index + (
        -1 if key in {int(QtCore.Qt.Key.Key_Left), int(QtCore.Qt.Key.Key_Up)} else 1
    )
    if 0 <= neighbor_index < len(navigation):
        return navigation[neighbor_index]
    return None


def _minor_log_ticks(
    low: float,
    high: float,
    *,
    include_upper: bool = False,
) -> tuple[tuple[float, str], ...]:
    ticks: list[tuple[float, str]] = []
    start_exp = int(np.floor(np.log10(low)))
    end_exp = int(np.ceil(np.log10(high)))
    for exponent in range(start_exp, end_exp + 1):
        base = 10.0**exponent
        for mantissa in range(2, 10):
            value = mantissa * base
            if value < low or (value > high if include_upper else value >= high):
                continue
            ticks.append((float(np.log10(value)), ""))
    return tuple(ticks)


def _rich_orbital_label_html(label: str) -> str:
    xps_match = _XPS_LABEL_RE.fullmatch(label)
    if xps_match is not None:
        orbital = html.escape(xps_match.group("orb"))
        html_label = f"{xps_match.group('n')}<i>{orbital}</i>"
        if xps_match.group("j") is not None:
            html_label += f"<sub>{html.escape(xps_match.group('j'))}</sub>"
        return html_label

    iupac_match = _IUPAC_LABEL_RE.fullmatch(label)
    if iupac_match is not None:
        html_label = html.escape(iupac_match.group("shell"))
        if iupac_match.group("index") is not None:
            html_label += f"<sub>{html.escape(iupac_match.group('index'))}</sub>"
        return html_label

    return html.escape(label)


@functools.cache
def _element_records() -> dict[int, ElementRecord]:
    records: dict[int, ElementRecord] = {}
    for atomic_number in range(1, 119):
        symbol = xraydb.atomic_symbol(atomic_number)
        row, column = ELEMENT_POSITIONS[atomic_number]
        records[atomic_number] = ElementRecord(
            atomic_number=atomic_number,
            symbol=symbol,
            name=xraydb.atomic_name(atomic_number).title(),
            mass=float(xraydb.atomic_mass(atomic_number)),
            category=ELEMENT_CATEGORIES[atomic_number],
            row=row,
            column=column,
            configuration=GROUND_STATE_CONFIGURATIONS.get(symbol, ""),
        )
    return records
