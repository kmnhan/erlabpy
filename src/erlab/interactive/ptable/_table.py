from __future__ import annotations

import html
import re
import webbrowser
from dataclasses import dataclass

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.ptable._metadata import (
    CATEGORY_COLORS,
    CATEGORY_REFERENCES,
    CATEGORY_TITLES,
    CategoryReference,
    configuration_to_html,
)
from erlab.interactive.ptable._shared import (
    _LEGEND_CATEGORY_ORDER,
    ElementRecord,
    ThemeColors,
    _blend_colors,
    _boost_saturation,
    _chip_detail_text_color,
    _chip_secondary_text_color,
    _css_rgba,
    _element_records,
    _format_mass,
    _set_background,
    _set_foreground,
    _theme_colors,
)

_CONFIG_TOOLTIP_PATTERN = re.compile(
    r"(?P<shell>\d+)(?P<orb>[spdfg])(?P<sup><sup>\d+</sup>)?"
)


def _popup_font(*, minimum_point_size: float = 15.0) -> QtGui.QFont:
    font = QtGui.QFont(QtWidgets.QApplication.font())
    font.setPointSizeF(max(font.pointSizeF(), minimum_point_size))
    return font


def _configuration_tooltip_html(configuration: str) -> str:
    config_html = configuration_to_html(configuration)
    if not config_html:
        return ""
    return _CONFIG_TOOLTIP_PATTERN.sub(
        lambda match: (
            f"{match.group('shell')}<i>{match.group('orb')}</i>"
            f"{match.group('sup') or ''}"
        ),
        config_html,
    )


@dataclass(frozen=True)
class _ElementCardDisplayStyle:
    fill_color: QtGui.QColor
    border_color: QtGui.QColor
    border_width: float
    lift_opacity: float


@dataclass(frozen=True)
class _LegendEntryDisplayStyle:
    label_color: QtGui.QColor
    marker_color: QtGui.QColor


@dataclass(frozen=True)
class _LegendEntryLayout:
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1
    vertical_label: bool = False


_LEGEND_ENTRY_LAYOUTS: dict[str, _LegendEntryLayout] = {
    "alkali_metal": _LegendEntryLayout(0, 0, row_span=2, vertical_label=True),
    "alkaline_earth_metal": _LegendEntryLayout(0, 1, row_span=2, vertical_label=True),
    "lanthanoid": _LegendEntryLayout(0, 2, column_span=3),
    "actinoid": _LegendEntryLayout(1, 2, column_span=3),
    "transition_metal": _LegendEntryLayout(0, 5, row_span=2, column_span=2),
    "other_metal": _LegendEntryLayout(0, 7, row_span=2, vertical_label=True),
    "metalloid": _LegendEntryLayout(0, 8, row_span=2, vertical_label=True),
    "nonmetal": _LegendEntryLayout(0, 9, row_span=2, column_span=2),
    "halogen": _LegendEntryLayout(0, 11, row_span=2, vertical_label=True),
    "noble_gas": _LegendEntryLayout(0, 12, row_span=2, vertical_label=True),
}


def _legend_fill_color(
    theme: ThemeColors, category: str, *, active: bool = False
) -> QtGui.QColor:
    fill_color = _boost_saturation(
        _blend_colors(
            QtGui.QColor(CATEGORY_COLORS[category]),
            theme.table_surface,
            0.28 if theme.is_dark else 0.16,
        ),
        0.18 if theme.is_dark else 0.22,
    )
    if active:
        fill_color = _boost_saturation(fill_color, 0.18)
        fill_color = _blend_colors(fill_color, theme.accent, 0.12)
    return fill_color


def _legend_label_color(theme: ThemeColors, *, active: bool) -> QtGui.QColor:
    return _blend_colors(theme.muted_text, theme.text, 0.86 if active else 0.74)


class _LegendTextLabel(QtWidgets.QLabel):
    def __init__(
        self,
        text: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        vertical: bool = False,
    ) -> None:
        super().__init__(text, parent)
        self._vertical = vertical
        self.setWordWrap(not vertical)

    def set_vertical(self, vertical: bool) -> None:
        if self._vertical == vertical:
            return
        self._vertical = vertical
        self.setWordWrap(not vertical)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        if not self._vertical:
            return hint
        return QtCore.QSize(hint.height(), hint.width())

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        if not self._vertical:
            return hint
        return QtCore.QSize(hint.height(), hint.width())

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        if not self._vertical:
            super().paintEvent(event)
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        painter.translate(self.width(), 0)
        painter.rotate(90)
        style = self.style()
        if style is None:
            style = QtWidgets.QApplication.style()
        if style is None:  # pragma: no cover
            return
        style.drawItemText(
            painter,
            QtCore.QRect(0, 0, self.height(), self.width()),
            int(self.alignment()),
            self.palette(),
            self.isEnabled(),
            self.text(),
            self.foregroundRole(),
        )


class _CategoryReferenceDialog(QtWidgets.QDialog):
    closed = QtCore.Signal()
    _MAX_TEXT_COLUMNS = 48

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = _theme_colors()
        self._category: str | None = None
        self._reference: CategoryReference | None = None
        self._applying_theme = False
        self._background_color = QtGui.QColor(self._theme.panel_alt)
        self._border_color = QtGui.QColor(self._theme.border)
        self.setObjectName("ptable-category-reference-dialog")
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setModal(True)
        self.setSizeGripEnabled(True)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)

        base_font = _popup_font()
        self.setFont(base_font)

        title_font = QtGui.QFont(base_font)
        title_font.setWeight(QtGui.QFont.Weight.Bold)

        body_font = QtGui.QFont(base_font)

        citation_font = QtGui.QFont(base_font)

        self.title_label = QtWidgets.QLabel(self)
        self.title_label.setFont(title_font)
        self.title_label.setWordWrap(True)
        self.title_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.title_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        layout.addWidget(self.title_label)

        self.body_label = QtWidgets.QLabel(self)
        self.body_label.setFont(body_font)
        self.body_label.setWordWrap(True)
        self.body_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.body_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        layout.addWidget(self.body_label)

        self.citation_label = QtWidgets.QLabel(self)
        self.citation_label.setFont(citation_font)
        self.citation_label.setWordWrap(True)
        self.citation_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.citation_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.citation_label.setOpenExternalLinks(False)
        self.citation_label.linkActivated.connect(self._open_url)
        self.citation_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        layout.addWidget(self.citation_label)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            parent=self,
        )
        self.button_box.rejected.connect(self.reject)
        self.button_box.accepted.connect(self.accept)
        layout.addWidget(self.button_box)

        self._apply_native_layout_metrics()
        self._apply_content_width_limits()
        self.apply_theme(self._theme)
        self.hide()

    def apply_theme(self, theme: ThemeColors) -> None:
        if self._applying_theme:
            return
        self._applying_theme = True
        self._theme = theme
        try:
            background = _blend_colors(
                theme.panel_alt,
                theme.window,
                0.18 if theme.is_dark else 0.42,
            )
            self._background_color = QtGui.QColor(background)
            self._border_color = QtGui.QColor(theme.border)
            palette = QtGui.QPalette(self.palette())
            palette.setColor(QtGui.QPalette.ColorRole.Window, self._background_color)
            palette.setColor(QtGui.QPalette.ColorRole.WindowText, theme.text)
            self.setPalette(palette)
            self.setStyleSheet(
                f"""
                QDialog#ptable-category-reference-dialog {{
                    background: {_css_rgba(self._background_color)};
                    border: 1px solid {_css_rgba(self._border_color)};
                }}
                """
            )
            _set_foreground(self.title_label, theme.text)
            _set_foreground(self.body_label, theme.text)
            _set_foreground(self.citation_label, theme.muted_text)
            self._refresh_reference_html()
            self.adjustSize()
        finally:
            self._applying_theme = False

    def show_for_category(self, category: str) -> None:
        reference = CATEGORY_REFERENCES[category]
        self.apply_theme(_theme_colors())
        self._reference = reference
        self._category = category
        self.setWindowTitle(reference.title)
        self.title_label.setText(reference.title)
        self.body_label.setText(reference.blurb)
        self._refresh_reference_html()
        self.adjustSize()
        self._center_on_owner()
        self.show()
        self.raise_()
        self.activateWindow()

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if event is None:
            return
        if event.type() in {
            QtCore.QEvent.Type.ApplicationFontChange,
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._apply_native_layout_metrics()
            self._apply_content_width_limits()
        if (
            event.type()
            in {
                QtCore.QEvent.Type.ApplicationPaletteChange,
                QtCore.QEvent.Type.PaletteChange,
                QtCore.QEvent.Type.StyleChange,
            }
            and not self._applying_theme
        ):
            self.apply_theme(_theme_colors())

    def hideEvent(self, event: QtGui.QHideEvent | None) -> None:
        super().hideEvent(event)
        self.closed.emit()

    def _refresh_reference_html(self) -> None:
        if self._reference is None:
            self.citation_label.clear()
            return
        text_color = self._theme.muted_text.name()
        link_color = self._theme.accent.name()
        rows = "".join(
            (
                "<tr>"
                f"<td valign='top' style='padding: 0 3px 4px 0; color: {text_color}; "
                "white-space: nowrap;'>"
                f"{index}.</td>"
                "<td valign='top' style='padding: 0 0 4px 0;'>"
                f"<a href='{html.escape(source.url, quote=True)}' "
                "style='text-decoration: none; "
                f"color: {link_color};'>"
                f"{source.citation_html or html.escape(source.citation)}</a></td>"
                "</tr>"
            )
            for index, source in enumerate(self._reference.references, start=1)
        )
        self.citation_label.setText(
            "".join(
                (
                    "<div>",
                    (
                        "<table cellspacing='0' cellpadding='0' "
                        "style='margin: 0; padding: 0;'>"
                    ),
                    f"{rows}</table>",
                    "</div>",
                )
            )
        )

    def _open_url(self, url: str) -> None:
        if not url:
            return
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)):
            webbrowser.open(url)
        self.close()

    def _apply_native_layout_metrics(self) -> None:
        layout = self.layout()
        if not isinstance(layout, QtWidgets.QVBoxLayout):  # pragma: no cover
            return
        style = self.style()
        if style is None:
            style = QtWidgets.QApplication.style()
        if style is None:  # pragma: no cover
            return
        margins = (
            style.pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_LayoutLeftMargin, None, self
            ),
            style.pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_LayoutTopMargin, None, self
            ),
            style.pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_LayoutRightMargin, None, self
            ),
            style.pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_LayoutBottomMargin, None, self
            ),
        )
        spacing = style.pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_LayoutVerticalSpacing,
            None,
            self,
        )
        if spacing < 0:
            spacing = style.layoutSpacing(
                QtWidgets.QSizePolicy.ControlType.Label,
                QtWidgets.QSizePolicy.ControlType.Label,
                QtCore.Qt.Orientation.Vertical,
                None,
                self,
            )
        layout.setContentsMargins(*margins)
        layout.setSpacing(max(spacing, 0))

    def _apply_content_width_limits(self) -> None:
        if not hasattr(self, "body_label"):
            return
        font_metrics = QtGui.QFontMetrics(self.body_label.font())
        content_width = font_metrics.horizontalAdvance("M" * self._MAX_TEXT_COLUMNS)
        for widget in (self.title_label, self.body_label, self.citation_label):
            widget.setMinimumWidth(content_width)

    @staticmethod
    def _available_geometry(anchor: QtCore.QPoint) -> QtCore.QRect:
        screen = QtGui.QGuiApplication.screenAt(anchor)
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:  # pragma: no cover
            return QtCore.QRect(anchor, QtCore.QSize(1, 1))
        return screen.availableGeometry().adjusted(8, 8, -8, -8)

    def _center_on_owner(self) -> None:
        owner = self.parent()
        owner_widget = owner if isinstance(owner, QtWidgets.QWidget) else None
        owner_window = owner_widget.window() if owner_widget is not None else None
        if owner_window is not None and owner_window.isVisible():
            anchor = owner_window.mapToGlobal(owner_window.rect().center())
        else:
            anchor = QtGui.QCursor.pos()
        available = self._available_geometry(anchor)
        size = self.sizeHint()
        x = anchor.x() - (size.width() // 2)
        y = anchor.y() - (size.height() // 2)
        x = max(available.left(), min(x, available.right() - size.width()))
        y = max(available.top(), min(y, available.bottom() - size.height()))
        self.move(QtCore.QPoint(x, y))


class ElementCard(QtWidgets.QFrame):
    hovered = QtCore.Signal(int)
    unhovered = QtCore.Signal(int)
    selected = QtCore.Signal(int, object)
    _CORNER_RADIUS = 0.0
    _ANIMATION_DURATION_MS = 140

    def __init__(
        self,
        record: ElementRecord,
        *,
        card_width: int = 120,
        card_height: int = 120,
        contents_margins: tuple[int, int, int, int] = (2, 0, 2, 0),
        top_stretch: int = 1,
        bottom_stretch: int = 2,
        symbol_font_delta: float = 26.0,
        symbol_font_min: float = 38.0,
        secondary_font_delta: float = 3.0,
        secondary_font_min: float = 12.6,
        config_font_delta: float = 1.6,
        config_font_min: float = 10.4,
        interactive: bool = True,
        draw_border: bool = True,
        chip_text_layout: bool = False,
        object_name_prefix: str = "element-card",
    ) -> None:
        super().__init__()
        self.record = record
        self._interactive = interactive
        self._draw_border = draw_border
        self._chip_text_layout = chip_text_layout
        self.is_selected: bool = False
        self.is_current: bool = False
        self.is_hovered: bool = False
        self.is_legend_match: bool = False
        self.is_search_match: bool = False
        self._theme = _theme_colors()
        self._fill_color = QtGui.QColor(CATEGORY_COLORS[record.category])
        self._border_color = self._theme.border
        self._border_width = 1
        self._display_fill_color = QtGui.QColor(self._fill_color)
        self._display_border_color = QtGui.QColor(self._border_color)
        self._display_border_width = float(self._border_width)
        self._display_lift_opacity = 0.0
        self._style_animation = QtCore.QVariantAnimation(self)
        self._style_animation.setStartValue(0.0)
        self._style_animation.setEndValue(1.0)
        self._style_animation.setDuration(self._ANIMATION_DURATION_MS)
        self._style_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._style_animation.valueChanged.connect(
            self._handle_style_animation_value_changed
        )
        self._style_animation.finished.connect(self._handle_style_animation_finished)
        self._animation_start_style = _ElementCardDisplayStyle(
            fill_color=QtGui.QColor(self._display_fill_color),
            border_color=QtGui.QColor(self._display_border_color),
            border_width=self._display_border_width,
            lift_opacity=self._display_lift_opacity,
        )
        self._animation_target_style = _ElementCardDisplayStyle(
            fill_color=QtGui.QColor(self._display_fill_color),
            border_color=QtGui.QColor(self._display_border_color),
            border_width=self._display_border_width,
            lift_opacity=self._display_lift_opacity,
        )

        self.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
            if interactive
            else QtCore.Qt.CursorShape.ArrowCursor
        )
        self.setObjectName(f"{object_name_prefix}-{record.atomic_number}")
        self.setFixedSize(card_width, card_height)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(*contents_margins)
        layout.setSpacing(0)

        symbol_font = QtGui.QFont(self.font())
        symbol_font.setPointSizeF(
            max(symbol_font.pointSizeF() + symbol_font_delta, symbol_font_min)
        )
        symbol_font.setBold(True)

        secondary_font = QtGui.QFont(self.font())
        secondary_font.setPointSizeF(
            max(
                secondary_font.pointSizeF() + secondary_font_delta,
                secondary_font_min,
            )
        )

        config_font = QtGui.QFont(self.font())
        config_font.setPointSizeF(
            max(config_font.pointSizeF() + config_font_delta, config_font_min)
        )

        atomic_number_font = QtGui.QFont(self.font())
        atomic_number_font.setPointSizeF(
            max(config_font.pointSizeF() + 0.8, config_font_min + 0.8) * 1.5
        )
        atomic_number_font.setWeight(QtGui.QFont.Weight.Bold)

        self.atomic_number_label = QtWidgets.QLabel(self)
        self.atomic_number_label.setFont(atomic_number_font)
        self.atomic_number_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.atomic_number_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.atomic_number_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.symbol_label = QtWidgets.QLabel(self)
        self.symbol_label.setFont(symbol_font)
        self.symbol_label.setAlignment(
            (QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)
            if chip_text_layout
            else QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.symbol_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.name_label = QtWidgets.QLabel(self)
        self.name_label.setFont(secondary_font)
        self.name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.name_label.setWordWrap(True)
        self.name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        self.mass_label = QtWidgets.QLabel(_format_mass(record.mass), self)
        self.mass_label.setFont(secondary_font)
        self.mass_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.mass_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.config_label = QtWidgets.QLabel(self)
        self.config_label.setFont(config_font)
        self.config_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.config_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.config_label.setWordWrap(False)
        self.config_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        if chip_text_layout:
            top_row = QtWidgets.QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)
            top_row.setSpacing(4)
            top_row.addWidget(
                self.atomic_number_label,
                0,
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            )
            top_row.addStretch(1)
            top_row.addWidget(
                self.symbol_label,
                0,
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop,
            )
            layout.addLayout(top_row)
        else:
            layout.addStretch(top_stretch)
            layout.addWidget(self.symbol_label)
        layout.addStretch(top_stretch)
        layout.addWidget(self.name_label)
        layout.addWidget(self.mass_label)
        layout.addWidget(self.config_label)
        layout.addStretch(bottom_stretch)

        for label in (
            self.atomic_number_label,
            self.symbol_label,
            self.name_label,
            self.mass_label,
            self.config_label,
        ):
            label.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
            )
            _set_foreground(label, QtGui.QColor("#0f172a"))
        self.set_record(record, animate=False)

    def apply_theme(self, theme: ThemeColors) -> None:
        self._theme = theme
        self.setToolTip(self._tooltip_html())
        self._refresh_style()

    def set_record(self, record: ElementRecord, *, animate: bool = False) -> None:
        self.record = record
        object_prefix = self.objectName().rsplit("-", 1)[0]
        self.setObjectName(f"{object_prefix}-{record.atomic_number}")
        self.symbol_label.setText(record.symbol)
        self.name_label.setText(record.name)
        self.mass_label.setText(_format_mass(record.mass))
        self.config_label.setText(configuration_to_html(record.configuration))
        self.atomic_number_label.setText(str(record.atomic_number))
        self._position_atomic_number_label()
        self.setToolTip(self._tooltip_html())
        self._refresh_style(animate=animate)

    def _tooltip_html(self) -> str:
        primary_color = self._theme.text.name()
        secondary_color = _chip_secondary_text_color(self._theme).name()
        configuration_html = _configuration_tooltip_html(self.record.configuration)
        category_label = html.escape(CATEGORY_TITLES[self.record.category])
        title_html = (
            f"<div style='margin-bottom: 4px; color: {primary_color};'>"
            f"<b>{html.escape(self.record.name)} "
            f"({html.escape(self.record.symbol)})</b></div>"
        )

        def row_html(label: str, value_html: str) -> str:
            return (
                "<tr>"
                f"<td style='color: {secondary_color};'><b>{label}</b></td>"
                f"<td style='color: {primary_color};'>&nbsp;{value_html}</td>"
                "</tr>"
            )

        rows_html = "".join(
            (
                row_html("Category:", category_label),
                row_html("Atomic number:", str(self.record.atomic_number)),
                row_html("Atomic mass:", f"{_format_mass(self.record.mass)} u"),
                row_html("Electron configuration:", configuration_html),
            )
        )
        return (
            "<qt>"
            f"<div style='white-space: nowrap; color: {primary_color};'>"
            f"{title_html}"
            "<table cellspacing='0' cellpadding='0'>"
            f"{rows_html}"
            "</table>"
            "</div>"
            "</qt>"
        )

    def _position_atomic_number_label(self) -> None:
        if self._chip_text_layout:
            return
        layout = self.layout()
        if layout is None:
            return
        margins = layout.contentsMargins()
        self.atomic_number_label.adjustSize()
        self.atomic_number_label.move(margins.left() + 2, margins.top() + 2)
        self.atomic_number_label.raise_()

    @staticmethod
    def _card_text_color(fill_color: QtGui.QColor) -> QtGui.QColor:
        if fill_color.lightness() >= 148:
            return QtGui.QColor("#0f172a")
        return QtGui.QColor("#f8fafc")

    def _apply_label_colors(self, fill_color: QtGui.QColor) -> None:
        if self._chip_text_layout:
            secondary = _chip_secondary_text_color(self._theme)
            detail = _chip_detail_text_color(self._theme)
            _set_foreground(self.symbol_label, self._theme.text)
            _set_foreground(self.name_label, self._theme.text)
            _set_foreground(self.mass_label, detail)
            _set_foreground(self.config_label, detail)
            _set_foreground(self.atomic_number_label, secondary)
            return
        primary = self._card_text_color(fill_color)
        secondary = _blend_colors(primary, fill_color, 0.34)
        tertiary = _blend_colors(primary, fill_color, 0.5)
        _set_foreground(self.symbol_label, primary)
        _set_foreground(self.name_label, primary)
        _set_foreground(self.mass_label, secondary)
        _set_foreground(self.config_label, secondary)
        _set_foreground(self.atomic_number_label, tertiary)

    def _base_fill_color(self) -> QtGui.QColor:
        fill_color = QtGui.QColor(CATEGORY_COLORS[self.record.category])
        if self._chip_text_layout:
            return _boost_saturation(
                _blend_colors(
                    fill_color,
                    self._theme.table_surface,
                    0.28 if self._theme.is_dark else 0.16,
                ),
                0.18 if self._theme.is_dark else 0.22,
            )
        return fill_color

    def _should_animate_style_transitions(self) -> bool:
        window = self.window()
        return self.isVisible() and window is not None and window.isVisible()

    def _current_display_style(self) -> _ElementCardDisplayStyle:
        return _ElementCardDisplayStyle(
            fill_color=QtGui.QColor(self._display_fill_color),
            border_color=QtGui.QColor(self._display_border_color),
            border_width=self._display_border_width,
            lift_opacity=self._display_lift_opacity,
        )

    @staticmethod
    def _style_equals(
        left: _ElementCardDisplayStyle, right: _ElementCardDisplayStyle
    ) -> bool:
        return (
            left.fill_color == right.fill_color
            and left.border_color == right.border_color
            and abs(left.border_width - right.border_width) < 1e-6
            and abs(left.lift_opacity - right.lift_opacity) < 1e-6
        )

    def _apply_display_style(self, style: _ElementCardDisplayStyle) -> None:
        self._display_fill_color = QtGui.QColor(style.fill_color)
        self._display_border_color = QtGui.QColor(style.border_color)
        self._display_border_width = style.border_width
        self._display_lift_opacity = style.lift_opacity
        self.update()

    def _animate_to_style(self, target_style: _ElementCardDisplayStyle) -> None:
        current_style = self._current_display_style()
        self._animation_start_style = current_style
        self._animation_target_style = target_style
        if self._style_equals(current_style, target_style):
            self._style_animation.stop()
            self._apply_display_style(target_style)
            return
        self._style_animation.stop()
        self._style_animation.start()

    def _handle_style_animation_value_changed(self, value: object) -> None:
        progress = float(value) if isinstance(value, (int, float)) else 1.0
        start_style = self._animation_start_style
        target_style = self._animation_target_style
        self._apply_display_style(
            _ElementCardDisplayStyle(
                fill_color=_blend_colors(
                    start_style.fill_color,
                    target_style.fill_color,
                    progress,
                ),
                border_color=_blend_colors(
                    start_style.border_color,
                    target_style.border_color,
                    progress,
                ),
                border_width=(
                    start_style.border_width
                    + (target_style.border_width - start_style.border_width) * progress
                ),
                lift_opacity=(
                    start_style.lift_opacity
                    + (target_style.lift_opacity - start_style.lift_opacity) * progress
                ),
            )
        )

    def _handle_style_animation_finished(self) -> None:
        self._apply_display_style(self._animation_target_style)

    def set_selected_state(self, selected: bool) -> None:
        if self.is_selected == selected:
            return
        self.is_selected = selected
        self._refresh_style()

    def set_current_state(self, current: bool) -> None:
        if self.is_current == current:
            return
        self.is_current = current
        self._refresh_style()

    def set_hover_state(self, hovered: bool) -> None:
        if self.is_hovered == hovered:
            return
        self.is_hovered = hovered
        self._refresh_style()

    def set_search_match_state(self, search_match: bool) -> None:
        if self.is_search_match == search_match:
            return
        self.is_search_match = search_match
        self._refresh_style()

    def set_legend_match_state(self, legend_match: bool) -> None:
        if self.is_legend_match == legend_match:
            return
        self.is_legend_match = legend_match
        self._refresh_style()

    def _target_style(self) -> _ElementCardDisplayStyle:
        theme = self._theme
        fill_color = self._base_fill_color()
        lift_opacity = 0.0
        if self._draw_border:
            if self.is_selected and self.is_current:
                border_color = _blend_colors(theme.text, theme.accent, 0.75)
                border_width = 5
                fill_color = _blend_colors(
                    fill_color,
                    theme.accent,
                    0.22 if theme.is_dark else 0.3,
                )
                lift_opacity = 0.22 if theme.is_dark else 0.16
            elif self.is_selected:
                border_color = theme.accent
                border_width = 3
                lift_opacity = 0.12 if theme.is_dark else 0.09
            elif self.is_current:
                border_color = _blend_colors(theme.text, theme.accent, 0.35)
                border_width = 3
                lift_opacity = 0.1 if theme.is_dark else 0.08
            elif self.is_hovered:
                border_color = theme.hover_accent
                border_width = 2
                lift_opacity = 0.16 if theme.is_dark else 0.12
            elif self.is_legend_match:
                border_color = theme.accent
                border_width = 2
                lift_opacity = 0.09 if theme.is_dark else 0.07
            elif self.is_search_match:
                border_color = theme.search_accent
                border_width = 5
                fill_color = _blend_colors(
                    fill_color,
                    theme.search_accent,
                    0.42 if theme.is_dark else 0.58,
                )
                lift_opacity = 0.2 if theme.is_dark else 0.14
            else:
                border_color = theme.border
                border_width = 1
            if self.is_legend_match:
                fill_color = _blend_colors(
                    fill_color,
                    QtGui.QColor("#ffffff"),
                    0.18 if theme.is_dark else 0.32,
                )
                lift_opacity = max(lift_opacity, 0.1 if theme.is_dark else 0.07)
        else:
            if self.is_selected and self.is_current:
                border_color = _blend_colors(theme.text, theme.accent, 0.75)
                border_width = 5
                fill_color = _blend_colors(
                    fill_color,
                    theme.accent,
                    0.36 if theme.is_dark else 0.5,
                )
                lift_opacity = 0.22 if theme.is_dark else 0.16
            elif self.is_selected:
                border_color = theme.accent
                border_width = 3
                fill_color = _blend_colors(
                    fill_color,
                    theme.accent,
                    0.28 if theme.is_dark else 0.38,
                )
                lift_opacity = 0.12 if theme.is_dark else 0.09
            elif self.is_current:
                border_color = _blend_colors(theme.text, theme.accent, 0.35)
                border_width = 3
                fill_color = _blend_colors(
                    fill_color,
                    theme.text,
                    0.1 if theme.is_dark else 0.16,
                )
                lift_opacity = 0.1 if theme.is_dark else 0.08
            elif self.is_hovered:
                border_color = theme.hover_accent
                border_width = 2
                fill_color = _blend_colors(
                    fill_color,
                    theme.hover_accent,
                    0.18 if theme.is_dark else 0.26,
                )
                lift_opacity = 0.16 if theme.is_dark else 0.12
            elif self.is_legend_match:
                border_color = theme.accent
                border_width = 2
                fill_color = _blend_colors(
                    fill_color,
                    theme.accent,
                    0.15 if theme.is_dark else 0.22,
                )
                lift_opacity = 0.09 if theme.is_dark else 0.07
            elif self.is_search_match:
                border_color = theme.search_accent
                border_width = 5
                fill_color = _blend_colors(
                    fill_color,
                    theme.search_accent,
                    0.46 if theme.is_dark else 0.64,
                )
                lift_opacity = 0.2 if theme.is_dark else 0.14
            else:
                border_color = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
                border_width = 0
        return _ElementCardDisplayStyle(
            fill_color=fill_color,
            border_color=border_color,
            border_width=float(border_width),
            lift_opacity=lift_opacity,
        )

    def _refresh_style(self, *, animate: bool | None = None) -> None:
        target_style = self._target_style()
        self._fill_color = target_style.fill_color
        self._border_color = target_style.border_color
        self._border_width = round(target_style.border_width)
        self._apply_label_colors(target_style.fill_color)
        if animate is None:
            animate = self._should_animate_style_transitions()
        if animate:
            self._animate_to_style(target_style)
        else:
            self._style_animation.stop()
            self._animation_start_style = target_style
            self._animation_target_style = target_style
            self._apply_display_style(target_style)

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = QtCore.QRectF(self.rect())
        if self._display_border_width > 0:
            rect = rect.adjusted(
                self._display_border_width / 2.0,
                self._display_border_width / 2.0,
                -self._display_border_width / 2.0,
                -self._display_border_width / 2.0,
            )
            pen = QtGui.QPen(self._display_border_color)
            pen.setWidthF(self._display_border_width)
            painter.setPen(pen)
        else:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(self._display_fill_color))
        if self._CORNER_RADIUS > 0.0:
            painter.drawRoundedRect(rect, self._CORNER_RADIUS, self._CORNER_RADIUS)
        else:
            painter.drawRect(rect)
        if self._display_lift_opacity > 0.0:
            inner_rect = rect.adjusted(
                3.0 + self._display_border_width * 0.25,
                3.0 + self._display_border_width * 0.25,
                -(3.0 + self._display_border_width * 0.25),
                -(3.0 + self._display_border_width * 0.25),
            )
            if inner_rect.isValid():
                inner_rect.setHeight(max(0.0, inner_rect.height() * 0.42))
                if inner_rect.height() > 0.0 and inner_rect.width() > 0.0:
                    lift_color = QtGui.QColor("#ffffff")
                    lift_color.setAlphaF(
                        min(
                            0.28,
                            max(
                                0.0,
                                self._display_lift_opacity
                                * (0.95 if self._theme.is_dark else 0.8),
                            ),
                        )
                    )
                    transparent_lift = QtGui.QColor(lift_color)
                    transparent_lift.setAlpha(0)
                    gradient = QtGui.QLinearGradient(
                        inner_rect.topLeft(), inner_rect.bottomLeft()
                    )
                    gradient.setColorAt(0.0, lift_color)
                    gradient.setColorAt(1.0, transparent_lift)
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.setBrush(QtGui.QBrush(gradient))
                    if self._CORNER_RADIUS > 0.0:
                        painter.drawRoundedRect(
                            inner_rect, self._CORNER_RADIUS, self._CORNER_RADIUS
                        )
                    else:
                        painter.drawRect(inner_rect)
        super().paintEvent(event)

    def enterEvent(self, event: QtGui.QEnterEvent | None) -> None:
        if self._interactive:
            self.hovered.emit(self.record.atomic_number)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent | None) -> None:
        if self._interactive:
            self.unhovered.emit(self.record.atomic_number)
        super().leaveEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        self._position_atomic_number_label()
        super().resizeEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if (
            event is not None
            and self._interactive
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self.selected.emit(self.record.atomic_number, event.modifiers())
        super().mousePressEvent(event)


class PeriodicTableWidget(QtWidgets.QWidget):
    hovered = QtCore.Signal(int)
    unhovered = QtCore.Signal(int)
    selected = QtCore.Signal(int, object)
    background_clicked = QtCore.Signal(object)
    _SERIES_GAP_ROW = 8
    _SERIES_GAP_HEIGHT = 28

    def __init__(self) -> None:
        super().__init__()
        self.cards: dict[int, ElementCard] = {}
        self.group_labels: list[QtWidgets.QLabel] = []
        self.period_labels: list[QtWidgets.QLabel] = []
        self.series_labels: list[QtWidgets.QLabel] = []
        self._selected_atomic_numbers: set[int] = set()
        self._current_atomic_number: int | None = None
        self._hovered_atomic_number: int | None = None
        self._active_legend_category: str | None = None
        self._legend_matches: set[int] = set()
        self._search_matches: set[int] = set()

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)

        axis_font = QtGui.QFont(self.font())
        axis_font.setPointSizeF(max(axis_font.pointSizeF() + 7.0, 18.0))
        axis_font.setBold(True)

        top_rows_by_group = {
            group: min(
                record.row
                for record in _element_records().values()
                if record.column == group
            )
            for group in range(1, 19)
        }
        for group in range(1, 19):
            label = QtWidgets.QLabel(str(group), self)
            label.setFont(axis_font)
            label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignHCenter
                | QtCore.Qt.AlignmentFlag.AlignBottom
            )
            layout.addWidget(label, max(0, top_rows_by_group[group] - 1), group)
            self.group_labels.append(label)

        for period in range(1, 8):
            label = QtWidgets.QLabel(str(period), self)
            label.setFont(axis_font)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, period, 0)
            self.period_labels.append(label)

        layout.setRowMinimumHeight(self._SERIES_GAP_ROW, self._SERIES_GAP_HEIGHT)

        for row, title in ((9, "Lanthanoids"), (10, "Actinoids")):
            label = QtWidgets.QLabel(title, self)
            label.setFont(axis_font)
            label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            label.setContentsMargins(0, 0, 8, 0)
            layout.addWidget(label, row, 0, 1, 4)
            self.series_labels.append(label)

        self.category_legend = CategoryLegend(self)
        self.category_legend.category_hovered.connect(self.set_active_legend_category)

        for record in _element_records().values():
            card = ElementCard(
                record,
                contents_margins=(6, 4, 6, 4),
                top_stretch=1,
                bottom_stretch=1,
                draw_border=False,
                chip_text_layout=True,
            )
            card.hovered.connect(self.hovered)
            card.unhovered.connect(self.unhovered)
            card.selected.connect(self.selected)
            self.cards[record.atomic_number] = card
            layout.addWidget(card, record.row, record.column)

        layout.setColumnMinimumWidth(0, 48)
        self.apply_theme(_theme_colors())
        self._position_category_legend()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._position_category_legend()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self._position_category_legend()

    def apply_theme(self, theme: ThemeColors) -> None:
        _set_background(self, theme.table_surface)
        for label in self.group_labels:
            _set_foreground(label, theme.muted_text)
        for label in self.period_labels:
            _set_foreground(label, theme.muted_text)
        for row_label in self.series_labels:
            _set_foreground(row_label, theme.muted_text)
        self.category_legend.apply_theme(theme)
        for card in self.cards.values():
            card.apply_theme(theme)

    def set_search_matches(self, matches: set[int]) -> None:
        if matches == self._search_matches:
            return
        for atomic_number in self._search_matches - matches:
            self.cards[atomic_number].set_search_match_state(False)
        for atomic_number in matches - self._search_matches:
            self.cards[atomic_number].set_search_match_state(True)
        self._search_matches = set(matches)

    def set_selected_atomic_numbers(self, atomic_numbers: set[int]) -> None:
        if atomic_numbers == self._selected_atomic_numbers:
            return
        for atomic_number in self._selected_atomic_numbers - atomic_numbers:
            self.cards[atomic_number].set_selected_state(False)
        for atomic_number in atomic_numbers - self._selected_atomic_numbers:
            self.cards[atomic_number].set_selected_state(True)
        self._selected_atomic_numbers = set(atomic_numbers)

    def set_current_atomic_number(self, atomic_number: int | None) -> None:
        if atomic_number == self._current_atomic_number:
            return
        if self._current_atomic_number is not None:
            self.cards[self._current_atomic_number].set_current_state(False)
        if atomic_number is not None:
            self.cards[atomic_number].set_current_state(True)
        self._current_atomic_number = atomic_number

    def set_hovered_atomic_number(self, atomic_number: int | None) -> None:
        if atomic_number == self._hovered_atomic_number:
            return
        if self._hovered_atomic_number is not None:
            self.cards[self._hovered_atomic_number].set_hover_state(False)
        if atomic_number is not None:
            self.cards[atomic_number].set_hover_state(True)
        self._hovered_atomic_number = atomic_number

    def set_active_legend_category(self, category: object) -> None:
        active_category = category if isinstance(category, str) else None
        if active_category == self._active_legend_category:
            return

        if active_category is None:
            new_matches: set[int] = set()
        else:
            new_matches = {
                atomic_number
                for atomic_number, card in self.cards.items()
                if card.record.category == active_category
            }

        for atomic_number in self._legend_matches - new_matches:
            self.cards[atomic_number].set_legend_match_state(False)
        for atomic_number in new_matches - self._legend_matches:
            self.cards[atomic_number].set_legend_match_state(True)

        self._active_legend_category = active_category
        self._legend_matches = new_matches
        self.category_legend.set_active_category(active_category)

    def _position_category_legend(self) -> None:
        layout = self.layout()
        if not isinstance(layout, QtWidgets.QGridLayout):  # pragma: no cover
            return

        top_left = layout.cellRect(1, 3).topLeft()
        bottom_right = layout.cellRect(3, 12).bottomRight()
        available_rect = QtCore.QRect(top_left, bottom_right).adjusted(6, 8, -6, -8)
        if available_rect.isNull():
            return

        legend_size = self.category_legend.sizeHint().boundedTo(available_rect.size())
        legend_rect = QtWidgets.QStyle.alignedRect(
            self.layoutDirection(),
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
            legend_size,
            available_rect,
        )
        self.category_legend.setGeometry(legend_rect)
        self.category_legend.raise_()


class PeriodicTableView(QtWidgets.QGraphicsView):
    background_clicked = QtCore.Signal(object)
    navigate_requested = QtCore.Signal(int, object)
    clear_requested = QtCore.Signal()
    hovered_atomic_number_changed = QtCore.Signal(object)
    _FIT_MARGIN_PX = 4.0
    _ENSURE_VISIBLE_MARGIN_PX = 16

    def __init__(
        self,
        table_widget: PeriodicTableWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.table_widget = table_widget
        self._hovered_atomic_number: int | None = None
        self._minimum_scale = 0.0
        self._is_scroll_mode = False

        self._scene = QtWidgets.QGraphicsScene(self)
        proxy = self._scene.addWidget(table_widget)
        if proxy is None:  # pragma: no cover
            raise RuntimeError("Failed to create table widget proxy")
        self._proxy = proxy
        self.setScene(self._scene)

        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        viewport = self.viewport()
        if viewport is not None:
            viewport.setMouseTracking(True)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.TextAntialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.apply_theme(_theme_colors())

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._fit_table()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self._fit_table()

    def set_minimum_scale_for_viewport(
        self, viewport_size: QtCore.QSize | QtCore.QSizeF
    ) -> None:
        self._minimum_scale = self._fit_scale_for_viewport_size(viewport_size)
        self._fit_table()

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            card = self._card_at_viewport_pos(event.position().toPoint())
            if card is None:
                self.background_clicked.emit(event.modifiers())
                self.setFocus()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is not None:
            card = self._card_at_viewport_pos(event.position().toPoint())
            atomic_number = None if card is None else card.record.atomic_number
            self._set_hovered_atomic_number(atomic_number)
        super().mouseMoveEvent(event)

    def viewportEvent(self, event: QtCore.QEvent | None) -> bool:
        if event is not None and event.type() == QtCore.QEvent.Type.Leave:
            self._set_hovered_atomic_number(None)
        return super().viewportEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            super().keyPressEvent(event)
            return
        if event.key() in {
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
        }:
            self.navigate_requested.emit(int(event.key()), event.modifiers())
            event.accept()
            return
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.clear_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _fit_table(self) -> None:
        rect = self._proxy.sceneBoundingRect()
        if rect.isNull():
            return
        self.setSceneRect(rect)
        fit_scale = self._fit_scale_for_viewport_size(self.maximumViewportSize())
        viewport = self.viewport()
        if viewport is None:  # pragma: no cover
            return
        current_center = self.mapToScene(viewport.rect().center())

        self.resetTransform()
        if fit_scale >= self._minimum_scale:
            self._is_scroll_mode = False
            self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            return

        self._is_scroll_mode = True
        self.scale(self._minimum_scale, self._minimum_scale)
        self.centerOn(
            current_center if rect.contains(current_center) else rect.center()
        )

    def ensure_atomic_number_visible(self, atomic_number: int | None) -> None:
        if atomic_number is None:
            return
        card = self.table_widget.cards.get(atomic_number)
        if card is None:
            return
        scene_rect = self._proxy.mapRectToScene(QtCore.QRectF(card.geometry()))
        if scene_rect.isNull():
            return
        if self._is_scroll_mode:
            self.centerOn(scene_rect.center())
        self.ensureVisible(
            scene_rect,
            self._ENSURE_VISIBLE_MARGIN_PX,
            self._ENSURE_VISIBLE_MARGIN_PX,
        )

    def apply_theme(self, theme: ThemeColors) -> None:
        self._scene.setBackgroundBrush(theme.table_surface)
        self.setBackgroundBrush(theme.table_surface)
        viewport = self.viewport()
        if viewport is not None:
            _set_background(viewport, theme.table_surface)

    def clear_hover_tracking(self) -> None:
        self._hovered_atomic_number = None

    def _fit_scale_for_viewport_size(
        self, viewport_size: QtCore.QSize | QtCore.QSizeF
    ) -> float:
        rect = self._proxy.sceneBoundingRect()
        if rect.isNull():
            return 0.0

        width = max(float(viewport_size.width()) - self._FIT_MARGIN_PX, 0.0)
        height = max(float(viewport_size.height()) - self._FIT_MARGIN_PX, 0.0)
        if width <= 0.0 or height <= 0.0:
            return 0.0
        return min(width / rect.width(), height / rect.height())

    def _set_hovered_atomic_number(self, atomic_number: int | None) -> None:
        if atomic_number == self._hovered_atomic_number:
            return
        self._hovered_atomic_number = atomic_number
        self.hovered_atomic_number_changed.emit(atomic_number)

    def _card_at_viewport_pos(self, position: QtCore.QPoint) -> ElementCard | None:
        scene_pos = self.mapToScene(position)
        local_pos = self._proxy.mapFromScene(scene_pos).toPoint()
        widget = self.table_widget.childAt(local_pos)
        return widget if isinstance(widget, ElementCard) else None


class LegendEntry(QtWidgets.QWidget):
    hovered = QtCore.Signal(str)
    unhovered = QtCore.Signal(str)
    clicked = QtCore.Signal(str)
    _ANIMATION_DURATION_MS = 140
    _CORNER_RADIUS_PX = 0
    _INACTIVE_BORDER_WIDTH_PX = 1.0
    _ACTIVE_BORDER_WIDTH_PX = 2.0

    def __init__(
        self,
        category: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        vertical_label: bool = False,
    ) -> None:
        super().__init__(parent)
        self.category = category
        self.is_active = False
        self._vertical_label = vertical_label
        self._theme = _theme_colors()
        self._display_label_color = _legend_label_color(self._theme, active=False)
        self._display_marker_color = _legend_fill_color(self._theme, self.category)
        self._style_animation = QtCore.QVariantAnimation(self)
        self._style_animation.setStartValue(0.0)
        self._style_animation.setEndValue(1.0)
        self._style_animation.setDuration(self._ANIMATION_DURATION_MS)
        self._style_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._style_animation.valueChanged.connect(
            self._handle_style_animation_value_changed
        )
        self._style_animation.finished.connect(self._handle_style_animation_finished)
        self._animation_start_style = _LegendEntryDisplayStyle(
            label_color=QtGui.QColor(self._display_label_color),
            marker_color=QtGui.QColor(self._display_marker_color),
        )
        self._animation_target_style = _LegendEntryDisplayStyle(
            label_color=QtGui.QColor(self._display_label_color),
            marker_color=QtGui.QColor(self._display_marker_color),
        )
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(34, 28)

        self.marker = QtWidgets.QFrame(self)
        self.marker.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.marker.setLineWidth(0)
        self.marker.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.marker.lower()

        legend_font = QtGui.QFont(self.font())
        legend_font.setPointSizeF(max(legend_font.pointSizeF() + 2.9, 13.6))
        legend_font.setWeight(QtGui.QFont.Weight.ExtraBold)

        self.label = _LegendTextLabel(
            CATEGORY_TITLES[category], self, vertical=vertical_label
        )
        self.label.setFont(legend_font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.apply_theme(self._theme, animate=False)

    def set_active(self, active: bool) -> None:
        if self.is_active == active:
            return
        self.is_active = active
        self.apply_theme(self._theme)

    def _should_animate_style_transitions(self) -> bool:
        window = self.window()
        return self.isVisible() and window is not None and window.isVisible()

    def _current_display_style(self) -> _LegendEntryDisplayStyle:
        return _LegendEntryDisplayStyle(
            label_color=QtGui.QColor(self._display_label_color),
            marker_color=QtGui.QColor(self._display_marker_color),
        )

    @staticmethod
    def _style_equals(
        left: _LegendEntryDisplayStyle, right: _LegendEntryDisplayStyle
    ) -> bool:
        return (
            left.label_color == right.label_color
            and left.marker_color == right.marker_color
        )

    def _target_style(self, theme: ThemeColors) -> _LegendEntryDisplayStyle:
        return _LegendEntryDisplayStyle(
            label_color=_legend_label_color(theme, active=self.is_active),
            marker_color=_legend_fill_color(
                theme, self.category, active=self.is_active
            ),
        )

    def _apply_display_style(self, style: _LegendEntryDisplayStyle) -> None:
        self._display_label_color = QtGui.QColor(style.label_color)
        self._display_marker_color = QtGui.QColor(style.marker_color)
        _set_foreground(self.label, self._display_label_color)
        palette = QtGui.QPalette(self.marker.palette())
        palette.setColor(QtGui.QPalette.ColorRole.Window, self._display_marker_color)
        self.marker.setPalette(palette)
        self.marker.setAutoFillBackground(True)
        border_color = (
            _blend_colors(self._theme.accent, self._display_marker_color, 0.28)
            if self.is_active
            else _blend_colors(
                self._theme.table_surface, self._display_marker_color, 0.5
            )
        )
        self.marker.setStyleSheet(
            f"""
            QFrame {{
                background-color: {_css_rgba(self._display_marker_color)};
                border: {
                self._ACTIVE_BORDER_WIDTH_PX
                if self.is_active
                else self._INACTIVE_BORDER_WIDTH_PX
            }px solid {_css_rgba(border_color)};
                border-radius: {self._CORNER_RADIUS_PX}px;
            }}
            """
        )
        self.update()

    def _animate_to_style(self, target_style: _LegendEntryDisplayStyle) -> None:
        current_style = self._current_display_style()
        self._animation_start_style = current_style
        self._animation_target_style = target_style
        if self._style_equals(current_style, target_style):
            self._style_animation.stop()
            self._apply_display_style(target_style)
            return
        self._style_animation.stop()
        self._style_animation.start()

    def _handle_style_animation_value_changed(self, value: object) -> None:
        progress = float(value) if isinstance(value, (int, float)) else 1.0
        start_style = self._animation_start_style
        target_style = self._animation_target_style
        self._apply_display_style(
            _LegendEntryDisplayStyle(
                label_color=_blend_colors(
                    start_style.label_color,
                    target_style.label_color,
                    progress,
                ),
                marker_color=_blend_colors(
                    start_style.marker_color,
                    target_style.marker_color,
                    progress,
                ),
            )
        )

    def _handle_style_animation_finished(self) -> None:
        self._apply_display_style(self._animation_target_style)

    def apply_theme(self, theme: ThemeColors, *, animate: bool | None = None) -> None:
        self._theme = theme
        target_style = self._target_style(theme)
        if animate is None:
            animate = self._should_animate_style_transitions()
        if animate:
            self._animate_to_style(target_style)
        else:
            self._style_animation.stop()
            self._animation_start_style = target_style
            self._animation_target_style = target_style
            self._apply_display_style(target_style)

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.marker.setGeometry(self.rect())
        if self._vertical_label:
            self.label.setGeometry(self.rect().adjusted(4, 3, -4, -3))
            return
        self.label.setGeometry(self.rect().adjusted(6, 3, -6, -3))

    def enterEvent(self, event: QtGui.QEnterEvent | None) -> None:
        self.hovered.emit(self.category)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent | None) -> None:
        self.unhovered.emit(self.category)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        self.hovered.emit(self.category)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if (
            event is not None
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.rect().contains(
                event.position().toPoint()
                if hasattr(event, "position")
                else event.pos()
            )
        ):
            self.clicked.emit(self.category)
            event.accept()
        super().mousePressEvent(event)


class CategoryLegend(QtWidgets.QWidget):
    category_hovered = QtCore.Signal(object)
    _COLUMN_COUNT = 13

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.entries: dict[str, LegendEntry] = {}
        self.entry_labels: list[QtWidgets.QLabel] = []
        self.marker_frames: list[QtWidgets.QFrame] = []
        self._active_category: str | None = None
        self._reference_category: str | None = None
        self.reference_dialog: _CategoryReferenceDialog | None = None
        self.setMouseTracking(True)
        self.installEventFilter(self)

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(1)
        layout.setVerticalSpacing(1)

        for category in _LEGEND_CATEGORY_ORDER:
            spec = _LEGEND_ENTRY_LAYOUTS[category]
            entry = LegendEntry(category, self, vertical_label=spec.vertical_label)
            entry.installEventFilter(self)
            entry.hovered.connect(self._handle_entry_hovered)
            entry.unhovered.connect(self._handle_entry_unhovered)
            entry.clicked.connect(self._handle_entry_clicked)
            layout.addWidget(
                entry,
                spec.row,
                spec.column,
                spec.row_span,
                spec.column_span,
            )
            self.entries[category] = entry
            self.entry_labels.append(entry.label)
            self.marker_frames.append(entry.marker)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        for column in range(self._COLUMN_COUNT):
            layout.setColumnStretch(column, 1)
        self.apply_theme(_theme_colors())

    def apply_theme(self, theme: ThemeColors) -> None:
        for entry in self.entries.values():
            entry.apply_theme(theme)
        if self.reference_dialog is not None:
            self.reference_dialog.apply_theme(theme)

    def set_reference_dialog(self, dialog: _CategoryReferenceDialog) -> None:
        if self.reference_dialog is dialog:
            return
        if self.reference_dialog is not None:
            blocker = QtCore.QSignalBlocker(self.reference_dialog)
            self.reference_dialog.close()
            del blocker
            self.reference_dialog.closed.disconnect(
                self._handle_reference_dialog_closed
            )
        self.reference_dialog = dialog
        self.reference_dialog.closed.connect(self._handle_reference_dialog_closed)
        self.reference_dialog.apply_theme(_theme_colors())

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(770, 184)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(650, 156)

    def set_active_category(self, category: str | None) -> None:
        if category == self._active_category:
            return
        self._active_category = category
        self._refresh_entry_states()

    def _entry_at_cursor(self) -> LegendEntry | None:
        local_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        if not self.rect().contains(local_pos):
            return None
        widget = self.childAt(local_pos)
        return widget if isinstance(widget, LegendEntry) else None

    def _set_hovered_category(self, category: str | None) -> None:
        if category == self._active_category:
            return
        self.set_active_category(category)
        self.category_hovered.emit(category)

    def _sync_hovered_category_from_cursor(self) -> None:
        entry = self._entry_at_cursor()
        self._set_hovered_category(None if entry is None else entry.category)

    def _refresh_entry_states(self) -> None:
        for entry_category, entry in self.entries.items():
            entry.set_active(
                entry_category == self._active_category
                or entry_category == self._reference_category
            )

    def _close_reference_dialog(self) -> None:
        self._reference_category = None
        self._refresh_entry_states()
        if self.reference_dialog is not None:
            self.reference_dialog.close()

    def _show_reference_dialog(self, category: str) -> None:
        if self._reference_category != category or self.reference_dialog is None:
            return
        self.reference_dialog.show_for_category(category)

    def cleanup(self) -> None:
        self._close_reference_dialog()

    def eventFilter(
        self,
        watched: QtCore.QObject | None,
        event: QtCore.QEvent | None,
    ) -> bool:
        if (
            event is not None
            and event.type()
            in {
                QtCore.QEvent.Type.Enter,
                QtCore.QEvent.Type.Leave,
                QtCore.QEvent.Type.MouseMove,
            }
            and (watched is self or isinstance(watched, LegendEntry))
        ):
            self._sync_hovered_category_from_cursor()
        return super().eventFilter(watched, event)

    def _handle_entry_hovered(self, category: str) -> None:
        self._set_hovered_category(category)

    def _handle_entry_unhovered(self, category: str) -> None:
        if self._active_category == category:
            self._set_hovered_category(None)

    def _handle_entry_clicked(self, category: str) -> None:
        if (
            self.reference_dialog is not None
            and self._reference_category == category
            and self.reference_dialog.isVisible()
        ):
            self._close_reference_dialog()
            return
        self._reference_category = category
        self._refresh_entry_states()
        if self.reference_dialog is None:
            return

        def _show_reference_dialog() -> None:
            self._show_reference_dialog(category)

        erlab.interactive.utils.single_shot(
            self,
            0,
            _show_reference_dialog,
            self.reference_dialog,
        )

    def _handle_reference_dialog_closed(self) -> None:
        if self._reference_category is None:
            return
        self._reference_category = None
        self._refresh_entry_states()
