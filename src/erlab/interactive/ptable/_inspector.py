from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.ptable._metadata import (
    CATEGORY_COLORS,
    CATEGORY_TITLES,
    configuration_to_html,
)
from erlab.interactive.ptable._plot import CrossSectionPlot, RichTextItemDelegate
from erlab.interactive.ptable._shared import (
    ElementRecord,
    ThemeColors,
    _blend_colors,
    _boost_saturation,
    _chip_detail_text_color,
    _chip_secondary_text_color,
    _edge_label,
    _edge_sort_key,
    _effective_point_size,
    _element_records,
    _element_symbols,
    _fit_symbol_font,
    _FittedSymbolFont,
    _format_energy,
    _format_mass,
    _rich_orbital_label_html,
    _set_background,
    _set_foreground,
    _theme_colors,
)


@dataclass(frozen=True)
class _LevelsTableRow:
    label: str
    values: tuple[float | None, ...]
    highlights: tuple[bool, ...]
    highlight_order: int = 0


class CompactElementChip(QtWidgets.QFrame):
    _COMPACT_SIZE = QtCore.QSize(92, 58)
    _DETAILED_HEIGHT = 90
    _COMPACT_CONTENT_MARGINS = (6, 4, 6, 4)
    _DETAILED_CONTENT_MARGINS = (6, 4, 6, 7)
    _COMPACT_CONTENT_SPACING = 1
    _DETAILED_CONTENT_SPACING = 0
    _CORNER_RADIUS = 8.0
    _FONT_FIT_STEP = 0.2
    _FONT_SCALE_STEP = 0.02
    _MIN_LAYOUT_SCALE = 0.5
    _NAME_FONT_MIN = 9.0
    _NAME_FONT_MAX = 13.0
    _DETAIL_FONT_MIN = 8.4
    _DETAIL_FONT_MAX = 12.2
    _CONFIG_FONT_MIN = 7.8
    _CONFIG_FONT_MAX = 11.2
    _ATOMIC_FONT_ABS_MIN = 7.0
    _NAME_FONT_ABS_MIN = 6.8
    _DETAIL_FONT_ABS_MIN = 6.4
    _CONFIG_FONT_ABS_MIN = 6.0
    _SYMBOL_FONT_ABS_MIN = 15.0
    _DETAILED_SYMBOL_STACK_RATIO = 0.44
    _COMPACT_SYMBOL_STACK_RATIO = 0.7
    _SHARED_SYMBOL_LAYOUT_CACHE: ClassVar[
        dict[tuple[str, str, float, float, float, bool], tuple[QtGui.QFont, int]]
    ] = {}

    def __init__(self, record: ElementRecord, *, detailed: bool = False) -> None:
        super().__init__()
        self.record = record
        self._detailed = detailed
        self._theme = _theme_colors()
        self._fill_color = QtGui.QColor(CATEGORY_COLORS[record.category])
        self._border_color = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
        self._border_width = 0
        self.setObjectName(f"element-summary-chip-{record.atomic_number}")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(
            *(
                self._DETAILED_CONTENT_MARGINS
                if self._detailed
                else self._COMPACT_CONTENT_MARGINS
            )
        )
        layout.setSpacing(
            self._DETAILED_CONTENT_SPACING
            if self._detailed
            else self._COMPACT_CONTENT_SPACING
        )
        self._top_row_layout = top_row = QtWidgets.QHBoxLayout()

        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(3 if self._detailed else 4)
        layout.addLayout(top_row)

        self.atomic_number_label = QtWidgets.QLabel(self)
        self.atomic_number_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        top_row.addWidget(self.atomic_number_label)
        top_row.addStretch(1)

        self.symbol_label = QtWidgets.QLabel(self)
        self.symbol_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop
        )
        top_row.addWidget(self.symbol_label)

        self.name_label = QtWidgets.QLabel(self)
        self.name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.name_label.setWordWrap(False)
        layout.addWidget(self.name_label, 1)

        self.mass_label = QtWidgets.QLabel(self)
        self.mass_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.mass_label.setWordWrap(False)
        layout.addWidget(self.mass_label)

        self.config_label = QtWidgets.QLabel(self)
        self.config_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.config_label.setWordWrap(False)
        self.config_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.config_label)

        self.mass_label.setVisible(self._detailed)
        self.config_label.setVisible(self._detailed)
        self.set_chip_width(self._COMPACT_SIZE.width())
        self.set_record(record)
        self.apply_theme(self._theme)

    @staticmethod
    def _font_with_delta(
        base_font: QtGui.QFont,
        *,
        delta: float,
        minimum_point_size: float,
        bold: bool = False,
    ) -> QtGui.QFont:
        font = QtGui.QFont(base_font)
        font.setPointSizeF(
            max(_effective_point_size(base_font) + delta, minimum_point_size)
        )
        font.setBold(bold)
        return font

    @staticmethod
    def _label_height_for_width(label: QtWidgets.QLabel, width: int) -> int:
        if label.isHidden():
            return 0
        if label.hasHeightForWidth():
            return label.heightForWidth(width)
        return label.sizeHint().height()

    def _rebuild_font_templates(self) -> None:
        base_font = QtGui.QFont(self.font())
        self._name_font_template = self._font_with_delta(
            base_font,
            delta=-0.1,
            minimum_point_size=self._NAME_FONT_MIN,
        )
        self._detail_font_template = self._font_with_delta(
            base_font,
            delta=-0.6,
            minimum_point_size=self._DETAIL_FONT_MIN,
        )
        self._config_font_template = self._font_with_delta(
            base_font,
            delta=-1.0,
            minimum_point_size=self._CONFIG_FONT_MIN,
        )

    def _visible_detail_labels(self) -> list[QtWidgets.QLabel]:
        return [
            label
            for label in (self.name_label, self.mass_label, self.config_label)
            if label.isVisible()
        ]

    def _content_geometry(self) -> tuple[int, int, int]:
        layout = self.layout()
        if layout is None:
            return (1, 1, 0)
        margins = layout.contentsMargins()
        text_width = max(16, self.width() - margins.left() - margins.right())
        contents_height = max(1, self.height() - margins.top() - margins.bottom())
        visible_item_count = 1 + len(self._visible_detail_labels())
        spacing_total = layout.spacing() * max(visible_item_count - 1, 0)
        return (text_width, contents_height, spacing_total)

    def _lower_text_height(self, text_width: int) -> int:
        return sum(
            self._label_height_for_width(label, text_width)
            for label in self._visible_detail_labels()
        )

    def _detail_height_budget(self) -> float:
        _, contents_height, spacing_total = self._content_geometry()
        symbol_ratio = (
            self._DETAILED_SYMBOL_STACK_RATIO
            if self._detailed
            else self._COMPACT_SYMBOL_STACK_RATIO
        )
        return max(1.0, (contents_height - spacing_total) * (1.0 - symbol_ratio))

    def _symbol_height_budget(self) -> float:
        _, contents_height, spacing_total = self._content_geometry()
        symbol_ratio = (
            self._DETAILED_SYMBOL_STACK_RATIO
            if self._detailed
            else self._COMPACT_SYMBOL_STACK_RATIO
        )
        return max(1.0, (contents_height - spacing_total) * symbol_ratio)

    def _atomic_number_font(
        self, base_font: QtGui.QFont, *, scale: float
    ) -> QtGui.QFont:
        base_point_size = _effective_point_size(base_font)
        atomic_preferred = max(base_point_size + 1.0, 9.0)
        atomic_font = QtGui.QFont(base_font)
        atomic_font.setPointSizeF(
            max(self._ATOMIC_FONT_ABS_MIN, atomic_preferred * scale)
        )
        atomic_font.setBold(True)
        return atomic_font

    def _atomic_number_width_reserve(self, base_font: QtGui.QFont) -> int:
        metrics = QtGui.QFontMetricsF(self._atomic_number_font(base_font, scale=1.0))
        widest = max(
            metrics.horizontalAdvance(str(record.atomic_number))
            for record in _element_records().values()
        )
        return round(widest)

    def _symbol_layout_limits(self, base_font: QtGui.QFont) -> tuple[float, float]:
        text_width, _, _ = self._content_geometry()
        available_height = max(1.0, self._symbol_height_budget())
        available_width = max(
            1,
            text_width
            - self._atomic_number_width_reserve(base_font)
            - self._top_row_layout.spacing(),
        )
        return (float(available_width), float(available_height))

    def _shared_symbol_font(self, base_font: QtGui.QFont) -> _FittedSymbolFont:
        symbol_width, symbol_height = self._symbol_layout_limits(base_font)
        cache_key = (
            base_font.toString(),
            QtGui.QFontInfo(base_font).family(),
            round(_effective_point_size(base_font), 3),
            round(symbol_width, 3),
            round(symbol_height, 3),
            self._detailed,
        )
        cached = self._SHARED_SYMBOL_LAYOUT_CACHE.get(cache_key)
        if cached is None:
            preferred_symbol_size = max(_effective_point_size(base_font) + 12.0, 22.0)
            fitted = _fit_symbol_font(
                base_font,
                _element_symbols(),
                max_width=symbol_width,
                max_height=symbol_height,
                preferred_point_size=max(
                    self._SYMBOL_FONT_ABS_MIN,
                    preferred_symbol_size,
                ),
                minimum_point_size=self._SYMBOL_FONT_ABS_MIN,
                step=self._FONT_FIT_STEP,
            )
            cached = (QtGui.QFont(fitted.font), fitted.top_margin)
            self._SHARED_SYMBOL_LAYOUT_CACHE[cache_key] = cached
        return _FittedSymbolFont(QtGui.QFont(cached[0]), cached[1])

    def _details_fit_for_current_fonts(self) -> bool:
        text_width, _, _ = self._content_geometry()
        return self._lower_text_height(text_width) <= self._detail_height_budget() + 1.0

    def _apply_text_fonts_for_scale(self, scale: float) -> None:
        self._rebuild_font_templates()
        base_font = QtGui.QFont(self.font())
        self.atomic_number_label.setFont(
            self._atomic_number_font(base_font, scale=scale)
        )

        text_width, _, _ = self._content_geometry()
        name_max = max(
            self._NAME_FONT_ABS_MIN,
            min(self._NAME_FONT_MAX, self._name_font_template.pointSizeF()) * scale,
        )
        detail_max = max(
            self._DETAIL_FONT_ABS_MIN,
            min(self._DETAIL_FONT_MAX, self._detail_font_template.pointSizeF()) * scale,
        )
        config_max = max(
            self._CONFIG_FONT_ABS_MIN,
            min(self._CONFIG_FONT_MAX, self._config_font_template.pointSizeF()) * scale,
        )
        self.name_label.setFont(
            self._fit_plain_text_font(
                self._name_font_template,
                self.name_label.text(),
                max_width=text_width,
                min_size=min(self._NAME_FONT_ABS_MIN, name_max),
                max_size=name_max,
            )
        )
        self.mass_label.setFont(
            self._fit_plain_text_font(
                self._detail_font_template,
                self.mass_label.text(),
                max_width=text_width,
                min_size=min(self._DETAIL_FONT_ABS_MIN, detail_max),
                max_size=detail_max,
            )
        )
        self.config_label.setFont(
            self._fit_rich_text_font(
                self._config_font_template,
                self.config_label.text(),
                max_width=text_width,
                min_size=min(self._CONFIG_FONT_ABS_MIN, config_max),
                max_size=config_max,
            )
        )
        _, symbol_height = self._symbol_layout_limits(base_font)
        fitted_symbol = self._shared_symbol_font(base_font)
        self.symbol_label.setFont(fitted_symbol.font)
        self.symbol_label.setContentsMargins(0, fitted_symbol.top_margin, 0, 0)
        self.symbol_label.setFixedHeight(max(1, round(symbol_height)))
        layout = self.layout()
        if layout is not None:
            layout.invalidate()

    def set_record(self, record: ElementRecord) -> None:
        self.record = record
        self.atomic_number_label.setText(str(record.atomic_number))
        self.symbol_label.setText(record.symbol)
        self.name_label.setText(record.name)
        self.mass_label.setText(f"{_format_mass(record.mass)} u")
        self.config_label.setText(configuration_to_html(record.configuration))
        self._update_text_fonts()
        self.setToolTip(
            f"{record.name} ({record.symbol})\n"
            f"Atomic number: {record.atomic_number}\n"
            f"Category: {CATEGORY_TITLES[record.category]}"
        )

    def set_chip_width(self, width: int) -> None:
        height = (
            self._DETAILED_HEIGHT if self._detailed else self._COMPACT_SIZE.height()
        )
        self.setFixedSize(width, height)
        self._update_text_fonts()

    def _fit_plain_text_font(
        self,
        template: QtGui.QFont,
        text: str,
        *,
        max_width: int,
        min_size: float,
        max_size: float,
    ) -> QtGui.QFont:
        fitted = QtGui.QFont(template)
        point_size = max_size
        while point_size >= min_size:
            fitted.setPointSizeF(point_size)
            if QtGui.QFontMetricsF(fitted).horizontalAdvance(text) <= max_width:
                return fitted
            point_size -= self._FONT_FIT_STEP
        fitted.setPointSizeF(min_size)
        return fitted

    def _fit_rich_text_font(
        self,
        template: QtGui.QFont,
        html_text: str,
        *,
        max_width: int,
        min_size: float,
        max_size: float,
    ) -> QtGui.QFont:
        fitted = QtGui.QFont(template)
        point_size = max_size
        while point_size >= min_size:
            fitted.setPointSizeF(point_size)
            document = QtGui.QTextDocument()
            document.setDefaultFont(fitted)
            document.setDocumentMargin(0.0)
            document.setHtml(html_text)
            if document.idealWidth() <= max_width:
                return fitted
            point_size -= self._FONT_FIT_STEP
        fitted.setPointSizeF(min_size)
        return fitted

    def _update_text_fonts(self) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return
        chosen_scale = self._MIN_LAYOUT_SCALE
        self._apply_text_fonts_for_scale(1.0)
        if self._details_fit_for_current_fonts():
            return
        scale = 1.0 - self._FONT_SCALE_STEP
        while scale >= self._MIN_LAYOUT_SCALE - 1e-6:
            self._apply_text_fonts_for_scale(scale)
            chosen_scale = scale
            if self._details_fit_for_current_fonts():
                break
            scale -= self._FONT_SCALE_STEP
        self._apply_text_fonts_for_scale(chosen_scale)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if event is None:
            return
        if event.type() in {
            QtCore.QEvent.Type.ApplicationFontChange,
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._update_text_fonts()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self._update_text_fonts()

    def _apply_text_colors(self) -> None:
        secondary = _chip_secondary_text_color(self._theme)
        detail = _chip_detail_text_color(self._theme)
        colors = (
            secondary,
            self._theme.text,
            self._theme.text,
            detail,
            detail,
        )
        label_colors = (
            (self.atomic_number_label, colors[0]),
            (self.symbol_label, colors[1]),
            (self.name_label, colors[2]),
            (self.mass_label, colors[3]),
            (self.config_label, colors[4]),
        )
        for label, color in label_colors:
            _set_foreground(label, color)

    def apply_theme(self, theme: ThemeColors) -> None:
        self._theme = theme
        self._fill_color = _boost_saturation(
            _blend_colors(
                QtGui.QColor(CATEGORY_COLORS[self.record.category]),
                theme.panel,
                0.28 if theme.is_dark else 0.16,
            ),
            0.18 if theme.is_dark else 0.22,
        )
        self._border_color = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
        self._border_width = 0
        self._apply_text_colors()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setBrush(self._fill_color)
        if self._border_width > 0:
            painter.setPen(QtGui.QPen(self._border_color, self._border_width))
        else:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, self._CORNER_RADIUS, self._CORNER_RADIUS)
        super().paintEvent(event)


class RichTextHeaderView(QtWidgets.QHeaderView):
    def paintSection(
        self,
        painter: QtGui.QPainter | None,
        rect: QtCore.QRect,
        logical_index: int,
    ) -> None:
        if painter is None:
            return
        model = self.model()
        if model is None:
            super().paintSection(painter, rect, logical_index)
            return
        html_text = model.headerData(
            logical_index,
            self.orientation(),
            QtCore.Qt.ItemDataRole.UserRole,
        )
        if not isinstance(html_text, str):
            super().paintSection(painter, rect, logical_index)
            return

        option = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(option)
        if hasattr(self, "initStyleOptionForIndex"):
            self.initStyleOptionForIndex(option, logical_index)
        option.rect = rect
        option.text = ""

        style = self.style()
        if style is None:
            return
        style.drawControl(
            QtWidgets.QStyle.ControlElement.CE_Header,
            option,
            painter,
            self,
        )

        text_rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_HeaderLabel,
            option,
            self,
        )
        document = QtGui.QTextDocument()
        document.setDefaultFont(self.font())
        document.setDocumentMargin(0.0)
        color = self.palette().color(QtGui.QPalette.ColorRole.ButtonText).name()
        document.setHtml(f"<span style='color: {color};'>{html_text}</span>")

        painter.save()
        painter.translate(
            text_rect.left() + (text_rect.width() - document.idealWidth()) / 2.0,
            text_rect.top() + (text_rect.height() - document.size().height()) / 2.0,
        )
        document.drawContents(
            painter, QtCore.QRectF(0, 0, text_rect.width(), text_rect.height())
        )
        painter.restore()

    def sectionSizeFromContents(self, logical_index: int) -> QtCore.QSize:
        size = super().sectionSizeFromContents(logical_index)
        model = self.model()
        if model is None:
            return size
        html_text = model.headerData(
            logical_index,
            self.orientation(),
            QtCore.Qt.ItemDataRole.UserRole,
        )
        if not isinstance(html_text, str):
            return size

        document = QtGui.QTextDocument()
        document.setDefaultFont(self.font())
        document.setDocumentMargin(0.0)
        document.setHtml(html_text)
        return QtCore.QSize(
            max(size.width(), int(document.idealWidth()) + 12),
            max(size.height(), int(document.size().height()) + 10),
        )


class ElementInspector(QtWidgets.QWidget):
    plot_target_changed = QtCore.Signal(int)
    _SIDEBAR_WIDTH = 320
    _PLOT_FRAME_WIDTH = 320
    _PLOT_CONTENT_WIDTH = 300
    _PLOT_TARGET_COMBO_WIDTH = 72
    _SUMMARY_COMPACT_WIDTH = 320
    _SUMMARY_COLUMNS = 3
    _SUMMARY_DETAILED_MAX_ROWS = 2
    _SUMMARY_COMPACT_VISIBLE_ROWS = 3
    _SUMMARY_CARD_SPACING = 6

    def __init__(self) -> None:
        super().__init__()
        self._theme = _theme_colors()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        layout.addWidget(self.splitter, 1)

        self.top_panel = QtWidgets.QWidget(self.splitter)
        self.top_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._top_layout = QtWidgets.QHBoxLayout(self.top_panel)
        self._top_layout.setContentsMargins(0, 0, 0, 0)
        self._top_layout.setSpacing(0)
        self.splitter.addWidget(self.top_panel)

        self.table_container = QtWidgets.QFrame(self.top_panel)
        self.table_container.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.table_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._table_container_layout = QtWidgets.QVBoxLayout(self.table_container)
        self._table_container_layout.setContentsMargins(0, 0, 0, 0)
        self._table_container_layout.setSpacing(0)
        self._top_layout.addWidget(self.table_container, 1)

        self.side_panel = QtWidgets.QFrame(self.top_panel)
        self.side_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.side_panel.setFixedWidth(self._SIDEBAR_WIDTH)
        self.side_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        side_layout = QtWidgets.QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)
        self._top_layout.addWidget(self.side_panel, 0)

        self.summary_frame = QtWidgets.QFrame(self.side_panel)
        self.summary_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.summary_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._summary_layout = QtWidgets.QVBoxLayout(self.summary_frame)
        self._summary_layout.setContentsMargins(10, 10, 10, 10)
        self._summary_layout.setSpacing(8)

        self._summary_header_layout = QtWidgets.QHBoxLayout()
        self._summary_header_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_header_layout.setSpacing(0)
        self._summary_header_layout.addStretch(1)
        self._summary_layout.addLayout(self._summary_header_layout)

        self.mode_label = QtWidgets.QLabel(self.summary_frame)
        self.mode_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        mode_font = QtGui.QFont(self.font())
        mode_font.setPointSizeF(max(mode_font.pointSizeF() + 0.5, 12.0))
        mode_font.setBold(True)
        self.mode_label.setFont(mode_font)
        mode_height = max(
            self._label_height_for_text(text, self.mode_label)
            for text in ("No selection", "Preview", "Selected", "999 selected")
        )
        self.mode_label.setFixedHeight(mode_height)
        self._summary_header_layout.addWidget(self.mode_label)
        self.summary_stack = QtWidgets.QStackedWidget(self.summary_frame)
        self._summary_layout.addWidget(self.summary_stack, 1)

        self.summary_empty_page = QtWidgets.QWidget(self.summary_frame)
        empty_summary_layout = QtWidgets.QVBoxLayout(self.summary_empty_page)
        empty_summary_layout.setContentsMargins(0, 0, 0, 0)
        empty_summary_layout.setSpacing(8)
        self.summary_empty_label = QtWidgets.QLabel(self.summary_empty_page)
        self.summary_empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.summary_empty_label.setWordWrap(True)
        self._summary_empty_placeholder = CompactElementChip(
            _element_records()[1], detailed=True
        )
        placeholder_policy = self._summary_empty_placeholder.sizePolicy()
        placeholder_policy.setRetainSizeWhenHidden(True)
        self._summary_empty_placeholder.setSizePolicy(placeholder_policy)
        self._summary_empty_placeholder.hide()
        empty_summary_layout.addStretch(1)
        empty_summary_layout.addWidget(self.summary_empty_label)
        empty_summary_layout.addWidget(
            self._summary_empty_placeholder,
            0,
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
        )
        empty_summary_layout.addStretch(1)
        self.summary_stack.addWidget(self.summary_empty_page)

        self.summary_cards_page = QtWidgets.QWidget(self.summary_frame)
        summary_cards_layout = QtWidgets.QVBoxLayout(self.summary_cards_page)
        summary_cards_layout.setContentsMargins(0, 0, 0, 0)
        summary_cards_layout.setSpacing(8)

        self.summary_cards_scroll = QtWidgets.QScrollArea(self.summary_cards_page)
        self.summary_cards_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.summary_cards_scroll.setWidgetResizable(True)
        self.summary_cards_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.summary_cards_container = QtWidgets.QWidget(self.summary_cards_scroll)
        self.summary_cards_grid = QtWidgets.QGridLayout(self.summary_cards_container)
        self.summary_cards_grid.setContentsMargins(0, 0, 0, 0)
        self.summary_cards_grid.setHorizontalSpacing(self._SUMMARY_CARD_SPACING)
        self.summary_cards_grid.setVerticalSpacing(self._SUMMARY_CARD_SPACING)
        self.summary_cards_grid.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self.summary_cards_scroll.setWidget(self.summary_cards_container)
        summary_cards_layout.addWidget(self.summary_cards_scroll, 1)
        self.summary_stack.addWidget(self.summary_cards_page)
        self.summary_frame.ensurePolished()
        self._summary_max_fixed_height = self._summary_fixed_height_for_cards(
            self._SUMMARY_COLUMNS * self._SUMMARY_DETAILED_MAX_ROWS,
            detailed_cards=True,
        )
        self.summary_frame.setFixedHeight(self._summary_max_fixed_height)

        side_layout.addWidget(self.summary_frame, 0)
        self.summary_levels_separator = self._create_section_separator(
            "summary-levels", QtCore.Qt.Orientation.Horizontal
        )
        side_layout.addWidget(self.summary_levels_separator, 0)

        self.plot_frame = QtWidgets.QFrame(self.side_panel)
        self.plot_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.plot_frame.setFixedWidth(self._PLOT_FRAME_WIDTH)
        self.plot_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        plot_layout = QtWidgets.QVBoxLayout(self.plot_frame)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        plot_layout.setSpacing(8)

        self._plot_header_layout = QtWidgets.QHBoxLayout()
        self._plot_header_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_header_layout.setSpacing(8)
        plot_layout.addLayout(self._plot_header_layout)

        self.plot_title = QtWidgets.QLabel(
            "Photoionization cross sections", self.plot_frame
        )
        self.plot_title.setMinimumWidth(0)
        self.plot_title.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._plot_header_layout.addWidget(self.plot_title, 1)

        self.plot_target_combo = QtWidgets.QComboBox(self.plot_frame)
        self.plot_target_combo.setObjectName("ptable-plot-target")
        self.plot_target_combo.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.plot_target_combo.setMinimumContentsLength(2)
        self.plot_target_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo_policy = self.plot_target_combo.sizePolicy()
        combo_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        self.plot_target_combo.setSizePolicy(combo_policy)
        self.plot_target_combo.setFixedWidth(self._PLOT_TARGET_COMBO_WIDTH)
        self.plot_target_combo.currentIndexChanged.connect(
            self._emit_plot_target_changed
        )
        self.plot_target_combo.setVisible(False)
        self._plot_header_layout.addWidget(self.plot_target_combo, 0)

        self.cross_section_plot = CrossSectionPlot()
        self.cross_section_plot.setFixedWidth(self._PLOT_CONTENT_WIDTH)
        self.cross_section_plot.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.cross_section_plot.minimum_height_changed.connect(
            self._sync_plot_frame_minimum_height
        )
        plot_layout.addWidget(
            self.cross_section_plot, 1, QtCore.Qt.AlignmentFlag.AlignHCenter
        )
        self._sync_plot_frame_minimum_height()

        side_layout.addWidget(self.plot_frame, 1)

        self.bottom_panel = QtWidgets.QFrame(self.splitter)
        self.bottom_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.bottom_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        bottom_layout = QtWidgets.QVBoxLayout(self.bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self.levels_plot_separator = self._create_section_separator(
            "levels-plot", QtCore.Qt.Orientation.Horizontal
        )
        bottom_layout.addWidget(self.levels_plot_separator, 0)

        self.levels_frame = QtWidgets.QFrame(self.bottom_panel)
        self.levels_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.levels_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        levels_layout = QtWidgets.QVBoxLayout(self.levels_frame)
        levels_layout.setContentsMargins(10, 10, 10, 10)
        levels_layout.setSpacing(8)

        levels_header = QtWidgets.QHBoxLayout()
        levels_layout.addLayout(levels_header)
        self._levels_header_layout = levels_header

        self.levels_title = QtWidgets.QLabel(
            "X-ray absorption edges", self.levels_frame
        )
        self.levels_title.setMinimumWidth(0)
        self.levels_title.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        levels_header.addWidget(self.levels_title)
        levels_header.addStretch(1)
        self._apply_section_header_fonts()

        self.levels_controls_frame = QtWidgets.QFrame(self.levels_frame)
        self.levels_controls_frame.setObjectName("ptable-levels-controls")
        self.levels_controls_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.levels_controls_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.levels_controls_layout = QtWidgets.QHBoxLayout(self.levels_controls_frame)
        self.levels_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.levels_controls_layout.setSpacing(8)
        levels_header.addWidget(self.levels_controls_frame, 1)
        levels_header.addSpacing(8)

        self.copy_values_button = QtWidgets.QPushButton(
            "Copy absorption edges", self.levels_frame
        )
        self.copy_table_button = QtWidgets.QPushButton("Copy table", self.levels_frame)
        self.copy_values_button.setToolTip("Copy only the absorption edge values.")
        self.copy_table_button.setToolTip("Copy the visible table with labels.")
        for button in (self.copy_values_button, self.copy_table_button):
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setAutoDefault(False)
            button.setDefault(False)
        levels_header.addWidget(self.copy_values_button)
        levels_header.addWidget(self.copy_table_button)

        self.levels_stack = QtWidgets.QStackedWidget(self.levels_frame)
        levels_layout.addWidget(self.levels_stack)

        self.levels_table = QtWidgets.QTableWidget(self.levels_frame)
        self.levels_table.setRowCount(3)
        self.levels_table.setVerticalHeaderLabels(
            ["Label", "Absorption edge (eV)", "Kinetic energy (eV)"]
        )
        self.levels_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.levels_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.levels_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.levels_table.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.levels_table.setAlternatingRowColors(True)
        self.levels_table.setShowGrid(False)
        self.levels_table.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.levels_table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.levels_table.setItemDelegateForRow(
            0, RichTextItemDelegate(self.levels_table)
        )
        self.levels_table.setHorizontalHeader(
            RichTextHeaderView(QtCore.Qt.Orientation.Horizontal, self.levels_table)
        )
        horizontal_header = self.levels_table.horizontalHeader()
        vertical_header = self.levels_table.verticalHeader()
        if horizontal_header is None or vertical_header is None:  # pragma: no cover
            raise RuntimeError("Levels table headers were not created")
        self._levels_horizontal_header = horizontal_header
        self._levels_vertical_header = vertical_header
        self._levels_vertical_header.setDefaultSectionSize(34)
        self._levels_horizontal_header.setVisible(False)
        self._levels_horizontal_header.setStretchLastSection(False)
        self._levels_horizontal_header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.levels_stack.addWidget(self.levels_table)

        self.levels_empty_label = QtWidgets.QLabel(self.levels_frame)
        self.levels_empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.levels_empty_label.setWordWrap(True)
        self.levels_empty_label.setMargin(12)
        self.levels_stack.addWidget(self.levels_empty_label)

        bottom_layout.addWidget(self.levels_frame, 1)
        self.splitter.addWidget(self.bottom_panel)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setSizes([560, 340])
        self.section_separators = (
            self.summary_levels_separator,
            self.levels_plot_separator,
        )

        self._single_level_rows: list[tuple[str, float, dict[int, float]]] = []
        self._single_level_has_ke = False
        self._single_level_harmonic_orders: tuple[int, ...] = ()
        self._multi_level_columns: tuple[str, ...] = ()
        self._multi_level_rows: list[tuple[str, str, tuple[float | None, ...]]] = []
        self._levels_table_rows: tuple[_LevelsTableRow, ...] = ()
        self._summary_atomic_numbers: tuple[int, ...] = ()
        self._summary_cards: list[CompactElementChip] = []
        self._edge_cache: dict[str, dict[str, float]] = {}
        self.copy_values_button.clicked.connect(self.copy_edges)
        self.copy_table_button.clicked.connect(self.copy_table)
        self.copy_values_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)
        self.apply_theme(self._theme)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if event is None:
            return
        if event.type() in {
            QtCore.QEvent.Type.ApplicationFontChange,
            QtCore.QEvent.Type.FontChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._apply_section_header_fonts()

    def set_table_panel(self, widget: QtWidgets.QWidget) -> None:
        while self._table_container_layout.count():
            item = self._table_container_layout.takeAt(0)
            if item is None:
                continue
            child = item.widget()
            if child is not None and child is not widget:
                child.hide()
        self._table_container_layout.addWidget(widget, 1)

    @staticmethod
    def _label_height_for_text(text: str, prototype: QtWidgets.QLabel) -> int:
        label = QtWidgets.QLabel(text)
        label.setFont(prototype.font())
        label.setAlignment(prototype.alignment())
        label.ensurePolished()
        return label.sizeHint().height()

    def _apply_section_header_fonts(self) -> None:
        section_font = QtGui.QFont(self.font())
        section_font.setBold(True)
        self.plot_title.setFont(QtGui.QFont(section_font))
        self.levels_title.setFont(QtGui.QFont(section_font))
        self._sync_plot_frame_minimum_height()

    def _summary_fixed_height_for_cards(
        self, card_count: int, *, detailed_cards: bool
    ) -> int:
        if card_count <= 0:
            return self._summary_max_fixed_height
        if detailed_cards:
            visible_rows = min(
                self._SUMMARY_DETAILED_MAX_ROWS,
                max(
                    1, (card_count + self._SUMMARY_COLUMNS - 1) // self._SUMMARY_COLUMNS
                ),
            )
            card_height = CompactElementChip._DETAILED_HEIGHT
        else:
            visible_rows = self._SUMMARY_COMPACT_VISIBLE_ROWS
            card_height = CompactElementChip._COMPACT_SIZE.height()
        row_spacing = self.summary_cards_grid.verticalSpacing()
        rows_height = (visible_rows * card_height) + (
            row_spacing * max(visible_rows - 1, 0)
        )
        margins = self._summary_layout.contentsMargins()
        return (
            margins.top()
            + self._summary_header_layout.sizeHint().height()
            + self._summary_layout.spacing()
            + rows_height
            + margins.bottom()
        )

    def _set_summary_fixed_height(
        self, card_count: int, *, detailed_cards: bool
    ) -> None:
        self.summary_frame.setFixedHeight(
            self._summary_fixed_height_for_cards(
                card_count, detailed_cards=detailed_cards
            )
        )
        self.summary_frame.updateGeometry()

    def _sync_plot_frame_minimum_height(self, _height: int | None = None) -> None:
        plot_layout = self.plot_frame.layout()
        if not isinstance(plot_layout, QtWidgets.QVBoxLayout):  # pragma: no cover
            return
        self._plot_frame_min_height = (
            plot_layout.contentsMargins().top()
            + plot_layout.contentsMargins().bottom()
            + plot_layout.spacing()
            + self._plot_header_layout.sizeHint().height()
            + self.cross_section_plot.minimumHeight()
        )
        self.plot_frame.setMinimumHeight(self._plot_frame_min_height)
        self.plot_frame.updateGeometry()

    def _create_section_separator(
        self, name: str, orientation: QtCore.Qt.Orientation
    ) -> QtWidgets.QFrame:
        separator = QtWidgets.QFrame(self)
        separator.setObjectName(f"ptable-inspector-separator-{name}")
        separator.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        if orientation == QtCore.Qt.Orientation.Horizontal:
            separator.setFixedHeight(1)
            separator.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        else:
            separator.setFixedWidth(1)
            separator.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        return separator

    def apply_theme(self, theme: ThemeColors) -> None:
        self._theme = theme
        for frame in (
            self,
            self.side_panel,
            self.bottom_panel,
            self.summary_frame,
            self.levels_frame,
            self.levels_controls_frame,
            self.plot_frame,
        ):
            _set_background(frame, theme.panel)
            _set_foreground(frame, theme.text)
        for separator in self.section_separators:
            _set_background(separator, theme.border_soft)
        _set_background(self.summary_cards_container, theme.panel)
        summary_viewport = self.summary_cards_scroll.viewport()
        if summary_viewport is not None:
            _set_background(summary_viewport, theme.panel)
        _set_foreground(self.summary_empty_label, theme.muted_text)
        _set_foreground(self.mode_label, theme.accent)
        _set_foreground(self.levels_title, theme.text)
        _set_foreground(self.plot_title, theme.text)
        _set_foreground(self.levels_empty_label, theme.muted_text)
        for card in self._summary_cards:
            card.apply_theme(theme)
        for button in (self.copy_values_button, self.copy_table_button):
            palette = QtGui.QPalette(button.palette())
            palette.setColor(QtGui.QPalette.ColorRole.ButtonText, theme.text)
            palette.setColor(QtGui.QPalette.ColorRole.WindowText, theme.text)
            palette.setColor(
                QtGui.QPalette.ColorGroup.Disabled,
                QtGui.QPalette.ColorRole.ButtonText,
                theme.disabled_text,
            )
            palette.setColor(
                QtGui.QPalette.ColorGroup.Disabled,
                QtGui.QPalette.ColorRole.WindowText,
                theme.disabled_text,
            )
            button.setPalette(palette)

        combo_palette = QtGui.QPalette(self.plot_target_combo.palette())
        combo_palette.setColor(QtGui.QPalette.ColorRole.Base, theme.input_bg)
        combo_palette.setColor(QtGui.QPalette.ColorRole.Button, theme.panel_alt)
        combo_palette.setColor(QtGui.QPalette.ColorRole.Text, theme.text)
        combo_palette.setColor(QtGui.QPalette.ColorRole.WindowText, theme.text)
        combo_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, theme.text)
        self.plot_target_combo.setPalette(combo_palette)

        table_palette = QtGui.QPalette(self.levels_table.palette())
        table_palette.setColor(QtGui.QPalette.ColorRole.Base, theme.panel)
        table_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, theme.panel_alt)
        table_palette.setColor(QtGui.QPalette.ColorRole.Text, theme.text)
        table_palette.setColor(QtGui.QPalette.ColorRole.WindowText, theme.text)
        table_palette.setColor(QtGui.QPalette.ColorRole.Button, theme.panel)
        table_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, theme.text)
        table_palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.ButtonText,
            theme.disabled_text,
        )
        table_palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.WindowText,
            theme.disabled_text,
        )
        table_palette.setColor(QtGui.QPalette.ColorRole.Mid, theme.border_soft)
        self.levels_table.setPalette(table_palette)
        self._levels_horizontal_header.setPalette(table_palette)
        self._levels_vertical_header.setPalette(table_palette)
        self._reapply_levels_table_highlights()
        self.cross_section_plot.apply_theme(theme)

    def set_elements(
        self,
        records: tuple[ElementRecord, ...],
        *,
        active_record: ElementRecord | None,
        plot_record: ElementRecord | None,
        notation: str,
        hv: float | None,
        workfunction: float,
        max_harmonic: int,
        preview: bool,
    ) -> None:
        if active_record is None:
            self._set_empty_summary()
            self._set_levels_empty(
                "Select an element or hover to preview its x-ray absorption edges."
            )
            self._update_plot_target_combo(records, None)
            self.cross_section_plot.clear_element(
                "Select an element or hover to preview photoionization cross sections."
            )
            return

        summary_records = (
            tuple(sorted(records, key=lambda record: record.atomic_number))
            if len(records) > 1 and not preview
            else (active_record,)
        )
        summary_mode = (
            "Preview"
            if preview
            else "Selected"
            if len(summary_records) == 1
            else f"{len(summary_records)} selected"
        )
        self._set_summary_cards(summary_records, summary_mode)
        self._update_levels(
            records,
            active_record,
            notation,
            hv,
            workfunction,
            max_harmonic,
            preview=preview,
        )
        self._update_plot_target_combo(
            records, None if plot_record is None else plot_record.atomic_number
        )
        if plot_record is None:
            self.cross_section_plot.clear_element(
                "Select an element or hover to preview photoionization cross sections."
            )
            return
        self.cross_section_plot.set_element(
            plot_record.symbol,
            notation,
            hv,
            max_harmonic,
        )

    def _set_empty_summary(self) -> None:
        self.summary_stack.setCurrentWidget(self.summary_empty_page)
        self.mode_label.setText("No selection")
        self.summary_empty_label.setText(
            "Select an element or hover to preview its details."
        )
        self._set_summary_fixed_height(1, detailed_cards=True)
        self.summary_frame.updateGeometry()

    def _set_summary_cards(
        self,
        records: tuple[ElementRecord, ...],
        mode_text: str,
    ) -> None:
        atomic_numbers = tuple(record.atomic_number for record in records)
        detailed_cards = (
            len(records) <= self._SUMMARY_COLUMNS * self._SUMMARY_DETAILED_MAX_ROWS
        )
        self._set_summary_fixed_height(len(records), detailed_cards=detailed_cards)
        self.summary_stack.setCurrentWidget(self.summary_cards_page)
        self.mode_label.setText(mode_text)
        if atomic_numbers != self._summary_atomic_numbers:
            self._clear_summary_cards()
            cards: list[CompactElementChip] = []
            start_column = (
                (self._SUMMARY_COLUMNS - len(records)) // 2
                if len(records) < self._SUMMARY_COLUMNS
                else 0
            )
            for record in records:
                card = CompactElementChip(record, detailed=detailed_cards)
                card.apply_theme(self._theme)
                cards.append(card)
            for index, card in enumerate(cards):
                self._summary_cards.append(card)
                row = index // self._SUMMARY_COLUMNS
                column = (
                    start_column + index
                    if row == 0 and len(records) < self._SUMMARY_COLUMNS
                    else index % self._SUMMARY_COLUMNS
                )
                self.summary_cards_grid.addWidget(
                    card,
                    row,
                    column,
                    QtCore.Qt.AlignmentFlag.AlignHCenter
                    | QtCore.Qt.AlignmentFlag.AlignTop,
                )
            self._summary_atomic_numbers = atomic_numbers

    def _clear_summary_cards(self) -> None:
        while self.summary_cards_grid.count():
            item = self.summary_cards_grid.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._summary_cards = []

    def _update_plot_target_combo(
        self,
        records: tuple[ElementRecord, ...],
        plot_atomic_number: int | None,
    ) -> None:
        is_multi = len(records) > 1 and plot_atomic_number is not None
        self.plot_target_combo.setVisible(is_multi)
        if not is_multi:
            self.plot_target_combo.blockSignals(True)
            self.plot_target_combo.clear()
            self.plot_target_combo.blockSignals(False)
            return
        self.plot_target_combo.blockSignals(True)
        self.plot_target_combo.clear()
        for record in records:
            self.plot_target_combo.addItem(record.symbol, record.atomic_number)
        target_index = self.plot_target_combo.findData(plot_atomic_number)
        if target_index >= 0:
            self.plot_target_combo.setCurrentIndex(target_index)
        self.plot_target_combo.blockSignals(False)

    def _emit_plot_target_changed(self, index: int) -> None:
        atomic_number = self.plot_target_combo.itemData(index)
        if atomic_number is None:
            return
        self.plot_target_changed.emit(int(atomic_number))

    def _get_edges(self, symbol: str) -> dict[str, float]:
        edges = self._edge_cache.get(symbol)
        if edges is None:
            try:
                edges = dict(erlab.analysis.xps.get_edge(symbol))
            except KeyError:
                edges = {}
            self._edge_cache[symbol] = edges
        return edges

    def _update_levels(
        self,
        records: tuple[ElementRecord, ...],
        active_record: ElementRecord,
        notation: str,
        hv: float | None,
        workfunction: float,
        max_harmonic: int,
        *,
        preview: bool,
    ) -> None:
        if len(records) > 1 and not preview:
            self._update_multi_levels(
                records,
                notation,
                hv,
                workfunction,
                max_harmonic,
            )
            return
        self._update_single_levels(
            active_record,
            notation,
            hv,
            workfunction,
            max_harmonic,
        )

    def _set_levels_empty(self, message: str) -> None:
        self._single_level_rows = []
        self._single_level_has_ke = False
        self._single_level_harmonic_orders = ()
        self._multi_level_columns = ()
        self._multi_level_rows = []
        self._levels_table_rows = ()
        self.levels_empty_label.setText(message)
        self.levels_stack.setCurrentWidget(self.levels_empty_label)
        self.copy_values_button.setEnabled(False)
        self.copy_table_button.setEnabled(False)

    def _levels_highlight_color(self, highlight_order: int) -> QtGui.QColor:
        if highlight_order <= 0:
            return self._theme.edge_match_bg
        return self._theme.harmonic_match_bgs[
            (highlight_order - 1) % len(self._theme.harmonic_match_bgs)
        ]

    def _reapply_levels_table_highlights(self) -> None:
        default_brush = QtGui.QBrush()
        for row_index, row in enumerate(self._levels_table_rows):
            highlight_brush = QtGui.QBrush(
                self._levels_highlight_color(row.highlight_order)
            )
            for column_index, is_highlighted in enumerate(row.highlights):
                item = self.levels_table.item(row_index, column_index)
                if item is None:
                    continue
                item.setBackground(highlight_brush if is_highlighted else default_brush)

    def _core_level_edges(
        self,
        edges: dict[str, float],
        *,
        hv: float,
        workfunction: float,
        max_harmonic: int,
    ) -> dict[str, erlab.analysis.xps.CoreLevelEdge]:
        return {
            label: erlab.analysis.xps.CoreLevelEdge.from_edge(
                edge,
                hv=hv,
                workfunction=workfunction,
                max_harmonic=max_harmonic,
            )
            for label, edge in edges.items()
        }

    def _populate_levels_table(
        self,
        columns: tuple[str, ...],
        rows: list[_LevelsTableRow],
    ) -> None:
        self.levels_stack.setCurrentWidget(self.levels_table)
        self.levels_table.clearContents()
        self.levels_table.setColumnCount(len(columns))
        self.levels_table.setRowCount(len(rows))
        self._levels_table_rows = tuple(rows)
        for column_index, display_label in enumerate(columns):
            header_item = QtWidgets.QTableWidgetItem(display_label)
            header_item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                _rich_orbital_label_html(display_label),
            )
            header_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.levels_table.setHorizontalHeaderItem(column_index, header_item)
        horizontal_header = self.levels_table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setVisible(True)
        for row_index in range(self.levels_table.rowCount()):
            self.levels_table.setRowHidden(row_index, False)

        self.levels_table.setVerticalHeaderLabels([row.label for row in rows])
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row.values):
                item = QtWidgets.QTableWidgetItem(
                    "" if value is None else _format_energy(value)
                )
                item.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignCenter
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                self.levels_table.setItem(row_index, column_index, item)

        self._reapply_levels_table_highlights()
        self.levels_table.resizeColumnsToContents()
        self.copy_values_button.setEnabled(True)
        self.copy_table_button.setEnabled(True)

    def _update_single_levels(
        self,
        record: ElementRecord,
        notation: str,
        hv: float | None,
        workfunction: float,
        max_harmonic: int,
    ) -> None:
        levels = self._get_edges(record.symbol)
        self._multi_level_columns = ()
        self._multi_level_rows = []
        if not levels:
            self._set_levels_empty(
                "Core-level absorption edges are unavailable for this element."
            )
            return

        rows = sorted(levels.items(), key=lambda item: item[1], reverse=True)
        self._single_level_rows = []
        show_ke = hv is not None
        harmonic_orders = tuple(range(1, max_harmonic + 1)) if show_ke else ()
        self._single_level_has_ke = show_ke
        self._single_level_harmonic_orders = harmonic_orders
        columns = tuple(_edge_label(label, notation) for label, _ in rows)
        edge_values: list[float | None] = []
        edge_highlights: list[bool] = []
        harmonic_kinetic_values: dict[int, list[float | None]] = {
            order: [] for order in harmonic_orders
        }
        harmonic_kinetic_highlights: dict[int, list[bool]] = {
            order: [] for order in harmonic_orders
        }
        core_edges = (
            self._core_level_edges(
                levels,
                hv=hv,
                workfunction=workfunction,
                max_harmonic=max_harmonic,
            )
            if hv is not None
            else {}
        )

        for label, edge in rows:
            display_label = _edge_label(label, notation)
            kinetic_energies = {} if hv is None else core_edges[label].kinetic_energies
            self._single_level_rows.append(
                (display_label, edge, dict(kinetic_energies))
            )
            highlight = any(energy > 0.0 for energy in kinetic_energies.values())
            edge_values.append(edge)
            edge_highlights.append(highlight)
            for order in harmonic_orders:
                kinetic_energy = kinetic_energies[order]
                harmonic_kinetic_values[order].append(
                    None if kinetic_energy < 0.0 else kinetic_energy
                )
                harmonic_kinetic_highlights[order].append(kinetic_energy > 0.0)

        table_rows = [
            _LevelsTableRow(
                record.symbol,
                tuple(edge_values),
                tuple(edge_highlights),
                highlight_order=0,
            )
        ]
        if show_ke and max_harmonic == 1:
            table_rows.append(
                _LevelsTableRow(
                    f"{record.symbol} KE",
                    tuple(harmonic_kinetic_values[1]),
                    tuple(harmonic_kinetic_highlights[1]),
                    highlight_order=1,
                )
            )
        elif show_ke:
            table_rows.extend(
                _LevelsTableRow(
                    f"{record.symbol} KE ({order}hv)",
                    tuple(harmonic_kinetic_values[order]),
                    tuple(harmonic_kinetic_highlights[order]),
                    highlight_order=order,
                )
                for order in harmonic_orders
            )
        self._populate_levels_table(columns, table_rows)

    def _update_multi_levels(
        self,
        records: tuple[ElementRecord, ...],
        notation: str,
        hv: float | None,
        workfunction: float,
        max_harmonic: int,
    ) -> None:
        edge_maps = {
            record.symbol: self._get_edges(record.symbol) for record in records
        }
        if not any(edge_maps.values()):
            self._set_levels_empty(
                "Core-level absorption edges are unavailable for the selected elements."
            )
            return

        columns_raw = sorted(
            {label for levels in edge_maps.values() for label in levels},
            key=_edge_sort_key,
        )
        self._single_level_rows = []
        self._single_level_has_ke = False
        self._single_level_harmonic_orders = ()
        self._multi_level_columns = tuple(
            _edge_label(label, notation) for label in columns_raw
        )
        self._multi_level_rows = []
        show_ke = hv is not None
        harmonic_orders = tuple(range(1, max_harmonic + 1)) if show_ke else ()
        table_rows: list[_LevelsTableRow] = []
        for record in records:
            levels = edge_maps[record.symbol]
            edge_values: list[float | None] = []
            edge_highlights: list[bool] = []
            harmonic_kinetic_values: dict[int, list[float | None]] = {
                order: [] for order in harmonic_orders
            }
            harmonic_kinetic_highlights: dict[int, list[bool]] = {
                order: [] for order in harmonic_orders
            }
            core_edges = (
                self._core_level_edges(
                    levels,
                    hv=hv,
                    workfunction=workfunction,
                    max_harmonic=max_harmonic,
                )
                if hv is not None
                else {}
            )
            for label in columns_raw:
                edge = levels.get(label)
                edge_values.append(edge)
                kinetic_energies = (
                    {}
                    if edge is None or hv is None
                    else core_edges[label].kinetic_energies
                )
                highlight = any(energy > 0.0 for energy in kinetic_energies.values())
                edge_highlights.append(highlight)
                for order in harmonic_orders:
                    kinetic_energy = kinetic_energies.get(order)
                    harmonic_kinetic_values[order].append(
                        None
                        if kinetic_energy is None or kinetic_energy < 0.0
                        else kinetic_energy
                    )
                    harmonic_kinetic_highlights[order].append(
                        kinetic_energy is not None and kinetic_energy > 0.0
                    )

            self._multi_level_rows.append(
                (record.symbol, "Absorption edge (eV)", tuple(edge_values))
            )
            table_rows.append(
                _LevelsTableRow(
                    record.symbol,
                    tuple(edge_values),
                    tuple(edge_highlights),
                    highlight_order=0,
                )
            )
            if show_ke and max_harmonic == 1:
                self._multi_level_rows.append(
                    (
                        record.symbol,
                        "Kinetic energy (eV)",
                        tuple(harmonic_kinetic_values[1]),
                    )
                )
                table_rows.append(
                    _LevelsTableRow(
                        f"{record.symbol} KE",
                        tuple(harmonic_kinetic_values[1]),
                        tuple(harmonic_kinetic_highlights[1]),
                        highlight_order=1,
                    )
                )
            elif show_ke:
                for order in harmonic_orders:
                    self._multi_level_rows.append(
                        (
                            record.symbol,
                            f"Kinetic energy @ {order}hv (eV)",
                            tuple(harmonic_kinetic_values[order]),
                        )
                    )
                    table_rows.append(
                        _LevelsTableRow(
                            f"{record.symbol} KE ({order}hv)",
                            tuple(harmonic_kinetic_values[order]),
                            tuple(harmonic_kinetic_highlights[order]),
                            highlight_order=order,
                        )
                    )

        self._populate_levels_table(self._multi_level_columns, table_rows)

    def copy_edges(self) -> None:
        if self._multi_level_rows:
            lines = [
                "Element\tMetric\t" + "\t".join(self._multi_level_columns),
                *(
                    f"{element_label}\t{metric_label}\t"
                    + "\t".join(
                        "" if value is None else _format_energy(value)
                        for value in values
                    )
                    for element_label, metric_label, values in self._multi_level_rows
                    if metric_label == "Absorption edge (eV)"
                ),
            ]
            erlab.interactive.utils.copy_to_clipboard(lines)
            return
        if not self._single_level_rows:
            return
        erlab.interactive.utils.copy_to_clipboard(
            [_format_energy(edge) for _, edge, _ in self._single_level_rows]
        )

    def copy_table(self) -> None:
        if self._multi_level_rows:
            lines = [
                "Element\tMetric\t" + "\t".join(self._multi_level_columns),
                *(
                    f"{element_label}\t{metric_label}\t"
                    + "\t".join(
                        "" if value is None else _format_energy(value)
                        for value in values
                    )
                    for element_label, metric_label, values in self._multi_level_rows
                ),
            ]
            erlab.interactive.utils.copy_to_clipboard(lines)
            return
        if not self._single_level_rows:
            return
        lines = [
            "Metric\t" + "\t".join(label for label, _, _ in self._single_level_rows)
        ]
        lines.append(
            "Absorption edge (eV)\t"
            + "\t".join(_format_energy(edge) for _, edge, _ in self._single_level_rows)
        )
        if self._single_level_has_ke:
            if self._single_level_harmonic_orders == (1,):
                lines.append(
                    "Kinetic energy (eV)\t"
                    + "\t".join(
                        ""
                        if kinetic_energies[1] < 0.0
                        else _format_energy(kinetic_energies[1])
                        for _, _, kinetic_energies in self._single_level_rows
                    )
                )
            else:
                for order in self._single_level_harmonic_orders:
                    lines.append(
                        f"Kinetic energy @ {order}hv (eV)\t"
                        + "\t".join(
                            ""
                            if kinetic_energies[order] < 0.0
                            else _format_energy(kinetic_energies[order])
                            for _, _, kinetic_energies in self._single_level_rows
                        )
                    )
        erlab.interactive.utils.copy_to_clipboard(lines)
