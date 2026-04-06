from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.ptable._shared import (
    CrossSectionRenderData,
    CrossSectionSeries,
    ThemeColors,
    _cross_section_label,
    _cross_section_sort_key,
    _css_rgba,
    _minor_log_ticks,
    _rich_orbital_label_html,
    _set_foreground,
    _theme_colors,
)

_TOTAL_SERIES_KEY = "__total__"


class RichTextItemDelegate(QtWidgets.QStyledItemDelegate):
    def paint(
        self,
        painter: QtGui.QPainter | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        options = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        html_text = index.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(html_text, str):
            super().paint(painter, option, index)
            return
        if painter is None:
            return

        style = options.widget.style()
        if style is None:
            style = QtWidgets.QApplication.style()
        if style is None:
            return
        text = options.text
        options.text = ""
        style.drawControl(
            QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
            options,
            painter,
            options.widget,
        )

        document = QtGui.QTextDocument()
        document.setDefaultFont(options.font)
        document.setDocumentMargin(0.0)
        document.setHtml(html_text)

        text_rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText,
            options,
            options.widget,
        )
        painter.save()
        painter.translate(
            text_rect.left(),
            text_rect.top() + (text_rect.height() - document.size().height()) / 2.0,
        )
        document.drawContents(
            painter, QtCore.QRectF(0, 0, text_rect.width(), text_rect.height())
        )
        painter.restore()
        options.text = text

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        html_text = index.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(html_text, str):
            return super().sizeHint(option, index)

        document = QtGui.QTextDocument()
        document.setDefaultFont(option.font)
        document.setDocumentMargin(0.0)
        document.setHtml(html_text)
        return QtCore.QSize(
            max(
                super().sizeHint(option, index).width(), int(document.idealWidth()) + 8
            ),
            max(30, int(document.size().height()) + 10),
        )


class EdgeTickAxisItem(pg.AxisItem):
    """Axis item that preserves edge label space and aligns spines with grid edges."""

    def boundingRect(self) -> QtCore.QRectF:
        hide_overlapping_labels = self.style["hideOverlappingLabels"]
        if hide_overlapping_labels is True:
            margin = 0
        elif hide_overlapping_labels is False:
            margin = 15
        else:
            try:
                margin = int(hide_overlapping_labels)
            except ValueError:
                margin = 0

        rect = self.mapRectFromParent(self.geometry())
        linked_view = self.linkedView()
        if linked_view is not None and self.grid is not False:
            rect = rect | linked_view.mapRectToItem(self, linked_view.boundingRect())

        tick_length = self.style["tickLength"]
        if self.orientation == "left":
            rect = rect.adjusted(0, -margin, -min(0, tick_length), margin)
        elif self.orientation == "right":
            rect = rect.adjusted(min(0, tick_length), -margin, 0, margin)
        elif self.orientation == "top":
            rect = rect.adjusted(-margin, 0, margin, -min(0, tick_length))
        elif self.orientation == "bottom":
            rect = rect.adjusted(-margin, min(0, tick_length), margin, 0)
        return rect

    def generateDrawSpecs(
        self, p: QtGui.QPainter
    ) -> (
        tuple[
            tuple[QtGui.QPen, QtCore.QPointF, QtCore.QPointF],
            list[tuple[QtGui.QPen, pg.Point, pg.Point]],
            list[tuple[QtCore.QRectF, int, str]],
        ]
        | None
    ):
        specs = super().generateDrawSpecs(p)
        if specs is None:
            return None

        if self.grid is False or self.linkedView() is None:
            return specs

        axis_spec, tick_specs, text_specs = specs
        pen, start, end = axis_spec
        bounds = self.mapRectFromParent(self.geometry())
        start_point = QtCore.QPointF(start)
        end_point = QtCore.QPointF(end)

        if self.orientation == "left":
            start_point.setX(bounds.right())
            end_point.setX(bounds.right())
            start_point.setY(min(max(start_point.y(), bounds.top()), bounds.bottom()))
            end_point.setY(min(max(end_point.y(), bounds.top()), bounds.bottom()))
        elif self.orientation == "right":
            start_point.setX(bounds.left())
            end_point.setX(bounds.left())
            start_point.setY(min(max(start_point.y(), bounds.top()), bounds.bottom()))
            end_point.setY(min(max(end_point.y(), bounds.top()), bounds.bottom()))
        elif self.orientation == "top":
            start_point.setY(bounds.bottom())
            end_point.setY(bounds.bottom())
            start_point.setX(min(max(start_point.x(), bounds.left()), bounds.right()))
            end_point.setX(min(max(end_point.x(), bounds.left()), bounds.right()))
        elif self.orientation == "bottom":
            start_point.setY(bounds.top())
            end_point.setY(bounds.top())
            start_point.setX(min(max(start_point.x(), bounds.left()), bounds.right()))
            end_point.setX(min(max(end_point.x(), bounds.left()), bounds.right()))

        return (pen, start_point, end_point), tick_specs, text_specs


class LegendSwatch(QtWidgets.QWidget):
    def __init__(
        self,
        pen: QtGui.QPen,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pen = QtGui.QPen(pen)
        self.setFixedSize(16, 10)

    def set_pen(self, pen: QtGui.QPen) -> None:
        self._pen = QtGui.QPen(pen)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(self._pen)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setWidthF(max(pen.widthF(), 1.8))
        painter.setPen(pen)
        mid_y = self.rect().center().y()
        painter.drawLine(
            QtCore.QPointF(1.5, mid_y),
            QtCore.QPointF(self.width() - 1.5, mid_y),
        )


class CurveLegendEntry(QtWidgets.QFrame):
    hovered = QtCore.Signal(str)
    unhovered = QtCore.Signal(str)
    clicked = QtCore.Signal(str)

    def __init__(
        self,
        label: str,
        pen: QtGui.QPen,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.label_text = label
        self._theme = _theme_colors()
        self._active = False
        self.setObjectName("curve-legend-entry")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self.swatch = LegendSwatch(pen, self)
        self.swatch.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        layout.addWidget(self.swatch, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.label = QtWidgets.QLabel(label, self)
        entry_font = QtGui.QFont(self.font())
        entry_font.setPointSizeF(max(entry_font.pointSizeF() - 0.6, 8.0))
        self.label.setFont(entry_font)
        self.label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.label.setText(_rich_orbital_label_html(label))
        self.label.setWordWrap(False)
        self.label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        layout.addWidget(self.label, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.apply_theme(self._theme)

    def set_pen(self, pen: QtGui.QPen) -> None:
        self.swatch.set_pen(pen)

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self.apply_theme(self._theme)

    def apply_theme(self, theme: ThemeColors) -> None:
        self._theme = theme
        if self._active:
            background = theme.panel_alt.name()
            border = theme.accent.name()
        else:
            background = "transparent"
            border = "transparent"
        self.setStyleSheet(
            f"""
            QFrame#curve-legend-entry {{
                background-color: {background};
                border: 1px solid {border};
                border-radius: 4px;
            }}
            """
        )
        _set_foreground(self.label, theme.text)

    def enterEvent(self, event: QtGui.QEnterEvent | None) -> None:
        self.hovered.emit(self.label_text)
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent | None) -> None:
        self.unhovered.emit(self.label_text)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent | None) -> None:
        self.hovered.emit(self.label_text)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:
        event_pos = (
            event.position().toPoint()
            if event is not None and hasattr(event, "position")
            else event.pos()
            if event is not None
            else QtCore.QPoint()
        )
        if (
            event is not None
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.rect().contains(event_pos)
        ):
            self.clicked.emit(self.label_text)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CurveLegend(QtWidgets.QWidget):
    entry_hovered = QtCore.Signal(str)
    entry_unhovered = QtCore.Signal(str)
    entry_toggled = QtCore.Signal(str, bool)
    _COLUMN_COUNT = 6

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = _theme_colors()
        self.entry_widgets: list[CurveLegendEntry] = []
        self._active_label: str | None = None
        self._toggled_labels: set[str] = set()
        self.setMouseTracking(True)
        self._layout = QtWidgets.QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(0)
        self._layout.setVerticalSpacing(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.entry_widgets = []
        self._active_label = None
        self._toggled_labels = set()

    def set_entries(self, entries: list[tuple[str, QtGui.QPen]]) -> None:
        self.clear()
        row = 0
        column = 0
        for label, pen in entries:
            entry = CurveLegendEntry(label, pen, self)
            entry.apply_theme(self._theme)
            entry.hovered.connect(self._handle_entry_hovered)
            entry.unhovered.connect(self._handle_entry_unhovered)
            entry.clicked.connect(self._handle_entry_clicked)
            self.entry_widgets.append(entry)
            column_span = 2 if label == "Total" else 1
            if column + column_span > self._COLUMN_COUNT:
                row += 1
                column = 0
            self._layout.addWidget(
                entry,
                row,
                column,
                1,
                column_span,
            )
            column += column_span
            if column >= self._COLUMN_COUNT:
                row += 1
                column = 0
        for column in range(self._COLUMN_COUNT):
            self._layout.setColumnStretch(column, 1)
        self._update_entry_states()

    def set_toggled_labels(self, labels: set[str]) -> None:
        available_labels = {entry.label_text for entry in self.entry_widgets}
        sanitized_labels = labels & available_labels
        if sanitized_labels == self._toggled_labels:
            return
        self._toggled_labels = sanitized_labels
        self._update_entry_states()

    def apply_theme(self, theme: ThemeColors) -> None:
        self._theme = theme
        _set_foreground(self, theme.text)
        for entry in self.entry_widgets:
            entry.apply_theme(theme)

    def _update_entry_states(self) -> None:
        active_labels = set(self._toggled_labels)
        if self._active_label is not None:
            active_labels.add(self._active_label)
        for entry in self.entry_widgets:
            entry.set_active(entry.label_text in active_labels)

    def _set_hovered_label(self, label: str | None) -> None:
        if label == self._active_label:
            return
        previous = self._active_label
        self._active_label = label
        self._update_entry_states()
        if label is None:
            if previous is not None:
                self.entry_unhovered.emit(previous)
            return
        self.entry_hovered.emit(label)

    def _handle_entry_hovered(self, label: str) -> None:
        self._set_hovered_label(label)

    def _handle_entry_unhovered(self, label: str) -> None:
        if self._active_label == label:
            self._set_hovered_label(None)

    def _handle_entry_clicked(self, label: str) -> None:
        next_toggled_labels = set(self._toggled_labels)
        toggled = label not in next_toggled_labels
        if toggled:
            next_toggled_labels.add(label)
        else:
            next_toggled_labels.discard(label)
        self.set_toggled_labels(next_toggled_labels)
        self.entry_toggled.emit(label, toggled)
        self._set_hovered_label(None)

    def leaveEvent(self, event: QtCore.QEvent | None) -> None:
        self._set_hovered_label(None)
        super().leaveEvent(event)

    def hideEvent(self, event: QtGui.QHideEvent | None) -> None:
        self._set_hovered_label(None)
        super().hideEvent(event)


class CrossSectionPlot(QtWidgets.QWidget):
    minimum_height_changed = QtCore.Signal(int)
    _X_RANGE = (10.0, 1500.0)
    _Y_RANGE = (1e-3, 100.0)
    _HOVER_OFFSET = 10
    _PLOT_WIDGET_MIN_HEIGHT = 140
    _LINE_COLORS = (
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#17BECF",
        "#BCBD22",
        "#7F7F7F",
    )
    _LINE_STYLES = (
        QtCore.Qt.PenStyle.SolidLine,
        QtCore.Qt.PenStyle.DashLine,
        QtCore.Qt.PenStyle.DotLine,
        QtCore.Qt.PenStyle.DashDotLine,
    )
    _X_TICKS = ((1.0, "10"), (2.0, "100"), (3.0, "1000"))
    _Y_TICKS = (
        (-3.0, "0.001"),
        (-2.0, "0.01"),
        (-1.0, "0.1"),
        (0.0, "1"),
        (1.0, "10"),
        (2.0, "100"),
    )
    _Y_LABEL_HTML = "\u03c3<sub>abs</sub> (Mb/atom)"

    def __init__(self) -> None:
        super().__init__()
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._theme = _theme_colors()

        self._stack = QtWidgets.QStackedWidget(self)
        axis_items = {
            "bottom": EdgeTickAxisItem("bottom"),
            "left": EdgeTickAxisItem("left"),
            "top": EdgeTickAxisItem("top"),
            "right": EdgeTickAxisItem("right"),
        }
        self.plot_widget = pg.PlotWidget(self, axisItems=axis_items)
        self.plot_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.setMouseTracking(True)
        self.plot_widget.viewport().setMouseTracking(True)
        self.plot_widget.viewport().installEventFilter(self)
        self.plot_widget.hideButtons()
        self.plot_widget.setMinimumHeight(self._PLOT_WIDGET_MIN_HEIGHT)

        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot_item.setLogMode(x=True, y=True)
        self.plot_item.showGrid(x=True, y=True, alpha=0.22)
        self.plot_item.getViewBox().setDefaultPadding(0.0)
        self.plot_item.enableAutoRange(x=False, y=False)
        self._set_fixed_ranges()
        self._hover_vertical_line = pg.PlotDataItem(
            pen=pg.mkPen("#000000"),
            antialias=False,
            autoDownsample=False,
        )
        self._hover_horizontal_line = pg.PlotDataItem(
            pen=pg.mkPen("#000000"),
            antialias=False,
            autoDownsample=False,
        )
        self._hover_plot_items = (
            self._hover_vertical_line,
            self._hover_horizontal_line,
        )
        self._ensure_hover_plot_items()

        self._hover_x_badge = QtWidgets.QLabel(self.plot_widget.viewport())
        self._hover_y_badge = QtWidgets.QLabel(self.plot_widget.viewport())
        fixed_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )
        fixed_font.setPointSizeF(max(fixed_font.pointSizeF() - 0.4, 8.5))
        for widget in (self._hover_x_badge, self._hover_y_badge):
            widget.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
            )
            widget.hide()
        self._hover_x_badge.setFont(fixed_font)
        self._hover_y_badge.setFont(fixed_font)
        self._hover_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            delay=1 / 120,
            rateLimit=120,
            slot=self._handle_plot_hover,
        )

        self.empty_label = QtWidgets.QLabel(self)
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setMargin(12)

        self._stack.addWidget(self.plot_widget)
        self._stack.addWidget(self.empty_label)
        self._stack.setMinimumHeight(self._PLOT_WIDGET_MIN_HEIGHT)
        self._layout.addWidget(self._stack, 1)

        self.legend_label = CurveLegend(self)
        self.legend_label.entry_hovered.connect(self._handle_legend_hovered)
        self.legend_label.entry_unhovered.connect(self._handle_legend_unhovered)
        self.legend_label.entry_toggled.connect(self._handle_legend_toggled)
        self._layout.addWidget(self.legend_label)
        self.legend_labels: tuple[str, ...] = ()
        self.plotted_labels: tuple[str, ...] = ()
        self.photon_line_energy: float | None = None
        self.photon_line_energies: tuple[float, ...] = ()
        self.x_range_eV: tuple[float, float] = self._X_RANGE
        self.y_range: tuple[float, float] = self._Y_RANGE
        self.x_tick_labels: tuple[str, ...] = tuple(label for _, label in self._X_TICKS)
        self.y_tick_labels: tuple[str, ...] = tuple(label for _, label in self._Y_TICKS)
        self.x_minor_tick_count = len(_minor_log_ticks(*self._X_RANGE))
        self.y_minor_tick_count = len(_minor_log_ticks(*self._Y_RANGE))
        self.y_axis_label_html: str = self._Y_LABEL_HTML
        self.log_x: bool = True
        self.log_y: bool = True
        self._data_cache: dict[str, CrossSectionRenderData] = {}
        self._curve_items: dict[str, tuple[pg.PlotDataItem, QtGui.QPen]] = {}
        self._curve_key_by_label: dict[str, str] = {}
        self._curve_label_by_key: dict[str, str] = {}
        self._active_legend_label: str | None = None
        self._toggled_legend_keys: set[str] = set()
        self._hover_series_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._hover_energy_grid = np.empty(0, dtype=np.float64)
        self._hover_log_energy_grid = np.empty(0, dtype=np.float64)
        self._hover_energy_eV: float | None = None
        self._hover_cursor_sigma: float | None = None
        self._hover_log_y: float | None = None
        self._last_state: tuple[str, str, float | None, int] | None = None
        self._update_minimum_height()
        self.apply_theme(self._theme)

    def minimumSizeHint(self) -> QtCore.QSize:
        width = max(
            self._stack.minimumSizeHint().width(),
            self.legend_label.minimumSizeHint().width(),
        )
        return QtCore.QSize(width, self.minimumHeight())

    def _update_minimum_height(self) -> None:
        legend_height = self.legend_label.sizeHint().height()
        if not self.legend_label.isVisible():
            legend_height = 0
        minimum_height = self._PLOT_WIDGET_MIN_HEIGHT + legend_height
        if legend_height > 0:
            minimum_height += self._layout.spacing()
        if self.minimumHeight() == minimum_height:
            return
        self.setMinimumHeight(minimum_height)
        self.updateGeometry()
        self.minimum_height_changed.emit(minimum_height)

    def _set_fixed_ranges(self) -> None:
        self.plot_item.setXRange(*np.log10(self._X_RANGE), padding=0.0)
        self.plot_item.setYRange(*np.log10(self._Y_RANGE), padding=0.0)

    @classmethod
    def _series_pen(cls, index: int) -> QtGui.QPen:
        color = cls._LINE_COLORS[index % len(cls._LINE_COLORS)]
        style = cls._LINE_STYLES[
            (index // len(cls._LINE_COLORS)) % len(cls._LINE_STYLES)
        ]
        return pg.mkPen(color=color, width=1.8, style=style)

    @staticmethod
    def _format_hover_energy(value: float) -> str:
        return f"{float(value):.2f}"

    @staticmethod
    def _format_hover_sigma(value: float) -> str:
        return np.format_float_positional(float(value), precision=3, trim="-")

    def _apply_hover_theme(self, theme: ThemeColors) -> None:
        guide_color = QtGui.QColor(theme.accent)
        guide_color.setAlpha(248 if theme.is_dark else 232)
        guide_pen = pg.mkPen(
            guide_color,
            width=1.4,
            style=QtCore.Qt.PenStyle.DashLine,
        )
        for item in self._hover_plot_items:
            item.setPen(guide_pen)
            item.setZValue(8)

        badge_bg = QtGui.QColor(theme.accent)
        badge_bg.setAlpha(232)
        badge_border = QtGui.QColor(theme.border)
        badge_text = (
            QtGui.QColor("#ffffff")
            if badge_bg.lightness() < 148
            else QtGui.QColor("#111111")
        )
        badge_style = f"""
            QLabel {{
                background-color: {_css_rgba(badge_bg)};
                border: 1px solid {_css_rgba(badge_border)};
                border-radius: 6px;
                color: {badge_text.name()};
                padding: 4px 8px;
            }}
        """
        self._hover_x_badge.setStyleSheet(badge_style)
        self._hover_y_badge.setStyleSheet(badge_style)

    def _ensure_hover_plot_items(self) -> None:
        for item in self._hover_plot_items:
            if item.scene() is None:
                self.plot_item.addItem(item, ignoreBounds=True)
            item.hide()

    def _reset_hover_lookup(self) -> None:
        self._hover_series_lookup = {}
        self._hover_energy_grid = np.empty(0, dtype=np.float64)
        self._hover_log_energy_grid = np.empty(0, dtype=np.float64)

    def _reset_hover_display(self) -> None:
        self._hover_energy_eV = None
        self._hover_cursor_sigma = None
        self._hover_log_y = None
        for item in self._hover_plot_items:
            item.hide()
            item.setData([], [])
        for widget in (self._hover_x_badge, self._hover_y_badge):
            widget.hide()
            widget.clear()

    def _set_hover_lookup(
        self,
        series: list[tuple[str, np.ndarray, np.ndarray]],
    ) -> None:
        self._reset_hover_lookup()
        if not series:
            return
        self._hover_series_lookup = {
            label: (
                np.asarray(hv, dtype=np.float64),
                np.asarray(sigma, dtype=np.float64),
            )
            for label, hv, sigma in series
        }
        energy_grid = np.unique(
            np.concatenate([hv for hv, _ in self._hover_series_lookup.values()])
        )
        self._hover_energy_grid = energy_grid.astype(np.float64, copy=False)
        self._hover_log_energy_grid = np.log10(self._hover_energy_grid)

    @staticmethod
    def _point_to_polyline_distance_sq(
        log_x: float,
        log_y: float,
        log_hv: np.ndarray,
        log_sigma: np.ndarray,
    ) -> float:
        if log_hv.size == 0:
            return float("inf")

        point = np.array([log_x, log_y], dtype=np.float64)
        if log_hv.size == 1:
            return float(
                np.sum(
                    np.square(
                        np.array([log_hv[0], log_sigma[0]], dtype=np.float64) - point
                    )
                )
            )

        segment_start = np.column_stack((log_hv[:-1], log_sigma[:-1]))
        segment_end = np.column_stack((log_hv[1:], log_sigma[1:]))
        segment_delta = segment_end - segment_start
        segment_lengths_sq = np.sum(np.square(segment_delta), axis=1)
        point_delta = point - segment_start

        projection = np.zeros(segment_lengths_sq.shape, dtype=np.float64)
        nonzero_segments = segment_lengths_sq > 0.0
        projection[nonzero_segments] = np.clip(
            np.sum(
                point_delta[nonzero_segments] * segment_delta[nonzero_segments],
                axis=1,
            )
            / segment_lengths_sq[nonzero_segments],
            0.0,
            1.0,
        )

        projected_points = segment_start + segment_delta * projection[:, np.newaxis]
        distances_sq = np.sum(np.square(projected_points - point), axis=1)
        if np.any(~nonzero_segments):
            distances_sq[~nonzero_segments] = np.sum(
                np.square(point_delta[~nonzero_segments]), axis=1
            )
        return float(np.min(distances_sq))

    def _hover_point_on_nearest_line(
        self, log_x: float, log_y: float
    ) -> tuple[float, float] | None:
        best_point: tuple[float, float] | None = None
        best_line_distance = np.inf
        best_point_distance = np.inf

        for label in self.plotted_labels:
            series = self._hover_series_lookup.get(label)
            if series is None:
                continue
            hv, sigma = series
            log_hv = np.log10(hv)
            log_sigma = np.log10(sigma)
            line_distance = self._point_to_polyline_distance_sq(
                log_x, log_y, log_hv, log_sigma
            )
            point_distances = np.square(log_hv - log_x) + np.square(log_sigma - log_y)
            if point_distances.size == 0:
                continue

            point_index = int(np.argmin(point_distances))
            point_distance = float(point_distances[point_index])
            if line_distance > best_line_distance or (
                np.isclose(line_distance, best_line_distance)
                and point_distance >= best_point_distance
            ):
                continue
            best_line_distance = line_distance
            best_point_distance = point_distance
            best_point = (float(hv[point_index]), float(sigma[point_index]))

        return best_point

    def _plot_viewport_rect(self) -> QtCore.QRect:
        scene_rect = self.plot_item.getViewBox().sceneBoundingRect()
        top_left = self.plot_widget.mapFromScene(scene_rect.topLeft())
        bottom_right = self.plot_widget.mapFromScene(scene_rect.bottomRight())
        return QtCore.QRect(top_left, bottom_right).normalized()

    def _position_hover_widgets(self) -> None:
        if self._hover_energy_eV is None or self._hover_log_y is None:
            return
        view_box = self.plot_item.getViewBox()
        crosshair_scene = view_box.mapViewToScene(
            QtCore.QPointF(np.log10(self._hover_energy_eV), self._hover_log_y)
        )
        crosshair_point = self.plot_widget.mapFromScene(crosshair_scene)
        plot_rect = self._plot_viewport_rect()
        viewport_rect = self.plot_widget.viewport().rect()

        x_badge_x = max(
            plot_rect.left(),
            min(
                crosshair_point.x() - (self._hover_x_badge.width() // 2),
                plot_rect.right() - self._hover_x_badge.width(),
            ),
        )
        x_badge_y = min(
            plot_rect.bottom() + self._HOVER_OFFSET,
            viewport_rect.bottom() - self._hover_x_badge.height(),
        )
        self._hover_x_badge.move(x_badge_x, x_badge_y)

        y_badge_x = max(
            plot_rect.left() - self._hover_y_badge.width() - self._HOVER_OFFSET,
            0,
        )
        y_badge_y = max(
            plot_rect.top(),
            min(
                crosshair_point.y() - (self._hover_y_badge.height() // 2),
                plot_rect.bottom() - self._hover_y_badge.height(),
            ),
        )
        self._hover_y_badge.move(y_badge_x, y_badge_y)

    def _update_hover_display(self) -> None:
        if (
            self._hover_energy_eV is None
            or self._hover_cursor_sigma is None
            or self._hover_log_y is None
        ):
            self._reset_hover_display()
            return
        self._ensure_hover_plot_items()
        view_box = self.plot_item.getViewBox()
        x_range, y_range = view_box.viewRange()
        x_span = np.power(10.0, np.asarray(x_range, dtype=np.float64))
        y_span = np.power(10.0, np.asarray(y_range, dtype=np.float64))
        self._hover_vertical_line.setData(
            [self._hover_energy_eV, self._hover_energy_eV],
            y_span,
        )
        self._hover_horizontal_line.setData(
            x_span,
            [self._hover_cursor_sigma, self._hover_cursor_sigma],
        )
        self._hover_vertical_line.show()
        self._hover_horizontal_line.show()

        self._hover_x_badge.setText(self._format_hover_energy(self._hover_energy_eV))
        self._hover_y_badge.setText(self._format_hover_sigma(self._hover_cursor_sigma))
        for widget in (self._hover_x_badge, self._hover_y_badge):
            widget.adjustSize()
            widget.show()
        self._position_hover_widgets()

    def _handle_plot_hover(self, event: tuple[QtCore.QPointF] | object) -> None:
        if (
            self._stack.currentWidget() is not self.plot_widget
            or not self._hover_series_lookup
            or self._hover_log_energy_grid.size == 0
        ):
            self._reset_hover_display()
            return
        scene_pos = event[0] if isinstance(event, tuple) else event
        if not isinstance(scene_pos, QtCore.QPointF):
            self._reset_hover_display()
            return

        view_box = self.plot_item.getViewBox()
        if not view_box.sceneBoundingRect().contains(scene_pos):
            self._reset_hover_display()
            return
        view_pos = view_box.mapSceneToView(scene_pos)
        log_x = float(view_pos.x())
        log_y = float(view_pos.y())
        if not np.isfinite(log_x) or not np.isfinite(log_y):
            self._reset_hover_display()
            return
        if (
            log_x < self._hover_log_energy_grid[0]
            or log_x > self._hover_log_energy_grid[-1]
            or log_y < np.log10(self._Y_RANGE[0])
            or log_y > np.log10(self._Y_RANGE[1])
        ):
            self._reset_hover_display()
            return

        hover_point = self._hover_point_on_nearest_line(log_x, log_y)
        if hover_point is None:
            self._reset_hover_display()
            return

        self._hover_energy_eV, self._hover_cursor_sigma = hover_point
        self._hover_log_y = float(np.log10(self._hover_cursor_sigma))
        self._update_hover_display()

    def apply_theme(self, theme: ThemeColors) -> None:
        self._theme = theme
        self.plot_widget.setBackground(theme.panel.name())
        self.plot_item.showGrid(x=True, y=True, alpha=1.0)
        self.plot_item.setLabel("bottom", "Photon energy (eV)", color=theme.text.name())
        self.plot_item.setLabel(
            "left",
            self._Y_LABEL_HTML,
            color=theme.text.name(),
        )
        bottom_axis = self.plot_item.getAxis("bottom")
        left_axis = self.plot_item.getAxis("left")
        top_axis = self.plot_item.getAxis("top")
        right_axis = self.plot_item.getAxis("right")
        bottom_axis.setStyle(
            autoExpandTextSpace=True,
            autoReduceTextSpace=False,
            hideOverlappingLabels=64,
            tickTextOffset=8,
        )
        left_axis.setStyle(
            autoExpandTextSpace=True,
            autoReduceTextSpace=False,
            hideOverlappingLabels=36,
            tickTextOffset=8,
        )
        top_axis.setStyle(
            autoExpandTextSpace=False,
            autoReduceTextSpace=True,
            showValues=False,
        )
        right_axis.setStyle(
            autoExpandTextSpace=False,
            autoReduceTextSpace=True,
            showValues=False,
        )
        # Let pyqtgraph size the labeled axes from their current text metrics so
        # the label-to-tick gap stays compact instead of inheriting extra padding.
        bottom_axis.setHeight(None)
        left_axis.setWidth(None)
        top_axis.setHeight(14)
        right_axis.setWidth(14)
        tick_levels = {
            "bottom": [
                list(self._X_TICKS),
                list(_minor_log_ticks(*self._X_RANGE)),
            ],
            "top": [
                list(self._X_TICKS),
                list(_minor_log_ticks(*self._X_RANGE)),
            ],
            "left": [
                list(self._Y_TICKS),
                list(_minor_log_ticks(*self._Y_RANGE, include_upper=False)),
            ],
            "right": [
                list(self._Y_TICKS),
                list(_minor_log_ticks(*self._Y_RANGE, include_upper=False)),
            ],
        }
        for axis_name, ticks in tick_levels.items():
            axis = self.plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(theme.text))
            axis.setTickPen(pg.mkPen(theme.text))
            axis.setTextPen(pg.mkPen(theme.text))
            axis.setTicks(ticks)
        _set_foreground(self.empty_label, theme.muted_text)
        self.legend_label.apply_theme(theme)
        self._apply_hover_theme(theme)

        previous_state = self._last_state
        self._last_state = None
        if previous_state is not None:
            self.set_element(*previous_state)

    def _visible_toggled_legend_labels(self) -> set[str]:
        return {
            label
            for key, label in self._curve_label_by_key.items()
            if key in self._toggled_legend_keys
        }

    def _sync_visible_toggled_legend_labels(self) -> None:
        self.legend_label.set_toggled_labels(self._visible_toggled_legend_labels())

    def _apply_curve_emphasis(self) -> None:
        emphasized_labels = self._visible_toggled_legend_labels()
        if self._active_legend_label is not None:
            emphasized_labels.add(self._active_legend_label)
        for label, (item, base_pen) in self._curve_items.items():
            pen = QtGui.QPen(base_pen)
            if not emphasized_labels:
                item.setPen(pen)
                item.setZValue(1)
                continue
            if label in emphasized_labels:
                pen.setWidthF(max(base_pen.widthF() + 1.6, 3.2))
                item.setZValue(3)
            else:
                color = QtGui.QColor(pen.color())
                color.setAlpha(120 if self._theme.is_dark else 150)
                pen.setColor(color)
                item.setZValue(1)
            item.setPen(pen)

    def _handle_legend_hovered(self, label: str) -> None:
        self._active_legend_label = label
        self._apply_curve_emphasis()

    def _handle_legend_unhovered(self, label: str) -> None:
        if self._active_legend_label == label:
            self._active_legend_label = None
            self._apply_curve_emphasis()

    def _handle_legend_toggled(self, label: str, toggled: bool) -> None:
        series_key = self._curve_key_by_label.get(label)
        if series_key is None:
            return
        next_toggled_legend_keys = set(self._toggled_legend_keys)
        if toggled:
            next_toggled_legend_keys.add(series_key)
        else:
            next_toggled_legend_keys.discard(series_key)
        self._toggled_legend_keys = next_toggled_legend_keys
        self._sync_visible_toggled_legend_labels()
        self._apply_curve_emphasis()

    def eventFilter(
        self,
        watched: QtCore.QObject | None,
        event: QtCore.QEvent | None,
    ) -> bool:
        if watched is self.plot_widget.viewport() and event is not None:
            if event.type() in {
                QtCore.QEvent.Type.Leave,
                QtCore.QEvent.Type.Hide,
            }:
                self._reset_hover_display()
            elif (
                event.type() == QtCore.QEvent.Type.Resize
                and self._hover_energy_eV is not None
            ):
                self._position_hover_widgets()
        return super().eventFilter(watched, event)

    def _get_render_data(self, symbol: str) -> CrossSectionRenderData:
        cached = self._data_cache.get(symbol)
        if cached is not None:
            return cached

        try:
            cross_sections = erlab.analysis.xps.get_cross_section(symbol)
            total_cross_section = erlab.analysis.xps.get_total_cross_section(symbol)
        except KeyError:
            cross_sections = {}
            total_cross_section = None

        if not cross_sections and total_cross_section is None:
            render_data = CrossSectionRenderData(
                series=(),
                total_hv=None,
                total_sigma=None,
                empty_message=(
                    "Photoionization cross sections are unavailable for this element."
                ),
            )
            self._data_cache[symbol] = render_data
            return render_data

        series: list[CrossSectionSeries] = []

        for subshell, data in sorted(
            cross_sections.items(), key=lambda item: _cross_section_sort_key(item[0])
        ):
            hv = np.asarray(data.hv.values, dtype=np.float64)
            sigma = np.asarray(data.values, dtype=np.float64)
            mask = (
                np.isfinite(hv)
                & np.isfinite(sigma)
                & (hv >= self._X_RANGE[0])
                & (hv <= self._X_RANGE[1])
                & (sigma > 0.0)
            )
            if not np.any(mask):
                continue

            hv_visible = hv[mask]
            sigma_visible = sigma[mask]
            series.append(
                CrossSectionSeries(
                    subshell=subshell,
                    hv=hv_visible,
                    sigma=sigma_visible,
                )
            )

        total_hv = None
        total_sigma = None
        if total_cross_section is not None:
            hv = np.asarray(total_cross_section.hv.values, dtype=np.float64)
            sigma = np.asarray(total_cross_section.values, dtype=np.float64)
            mask = (
                np.isfinite(hv)
                & np.isfinite(sigma)
                & (hv >= self._X_RANGE[0])
                & (hv <= self._X_RANGE[1])
                & (sigma > 0.0)
            )
            if np.any(mask):
                total_hv = hv[mask]
                total_sigma = sigma[mask]

        if not series and total_hv is None:
            render_data = CrossSectionRenderData(
                series=(),
                total_hv=None,
                total_sigma=None,
                empty_message=(
                    "Photoionization cross sections are unavailable in the requested "
                    "10-1500 eV range for this element."
                ),
            )
            self._data_cache[symbol] = render_data
            return render_data

        render_data = CrossSectionRenderData(
            series=tuple(series),
            total_hv=total_hv,
            total_sigma=total_sigma,
        )
        self._data_cache[symbol] = render_data
        return render_data

    def set_element(
        self,
        symbol: str,
        notation: str,
        photon_energy: float | None,
        max_harmonic: int = 1,
    ) -> None:
        state = (symbol, notation, photon_energy, max_harmonic)
        if state == self._last_state:
            return

        render_data = self._get_render_data(symbol)
        self.plot_item.clear()
        self._ensure_hover_plot_items()
        self._reset_hover_display()
        self._reset_hover_lookup()
        self.legend_label.clear()
        self.legend_label.setVisible(False)
        self.legend_labels = ()
        self.plotted_labels = ()
        self.photon_line_energy = None
        self.photon_line_energies = ()
        self._curve_items = {}
        self._curve_key_by_label = {}
        self._curve_label_by_key = {}
        self._active_legend_label = None

        if render_data.empty_message is not None:
            self.empty_label.setText(render_data.empty_message)
            self._stack.setCurrentWidget(self.empty_label)
            self._last_state = state
            return

        self._stack.setCurrentWidget(self.plot_widget)
        legend_labels: list[str] = []
        legend_entries: list[tuple[str, QtGui.QPen]] = []
        hover_series: list[tuple[str, np.ndarray, np.ndarray]] = []

        for index, series in enumerate(render_data.series):
            label = _cross_section_label(series.subshell, notation)
            pen = self._series_pen(index)
            item = self.plot_item.plot(
                series.hv,
                series.sigma,
                pen=pen,
                name=label,
                antialias=False,
                autoDownsample=True,
            )
            legend_labels.append(label)
            self._curve_items[label] = (item, QtGui.QPen(pen))
            self._curve_key_by_label[label] = series.subshell
            self._curve_label_by_key[series.subshell] = label
            legend_entries.append((label, QtGui.QPen(pen)))
            hover_series.append((label, series.hv, series.sigma))

        if render_data.total_hv is not None and render_data.total_sigma is not None:
            total_pen = pg.mkPen(self._theme.plot_total, width=2.8)
            item = self.plot_item.plot(
                render_data.total_hv,
                render_data.total_sigma,
                pen=total_pen,
                name="Total",
                antialias=False,
                autoDownsample=True,
            )
            legend_labels.append("Total")
            self._curve_items["Total"] = (item, QtGui.QPen(total_pen))
            self._curve_key_by_label["Total"] = _TOTAL_SERIES_KEY
            self._curve_label_by_key[_TOTAL_SERIES_KEY] = "Total"
            legend_entries.append(("Total", QtGui.QPen(total_pen)))
            hover_series.append(
                ("Total", render_data.total_hv, render_data.total_sigma)
            )

        harmonic_marker_energies: list[float] = []
        if photon_energy is not None:
            for order in range(1, max_harmonic + 1):
                marker_energy = photon_energy * order
                if not self._X_RANGE[0] <= marker_energy <= self._X_RANGE[1]:
                    continue
                self.plot_item.plot(
                    [marker_energy, marker_energy],
                    [self._Y_RANGE[0], self._Y_RANGE[1]],
                    pen=pg.mkPen(
                        self._theme.plot_marker,
                        width=1.2 if order == 1 else 0.9,
                        style=(
                            QtCore.Qt.PenStyle.DashLine
                            if order == 1
                            else QtCore.Qt.PenStyle.DashDotLine
                        ),
                    ),
                    antialias=False,
                    autoDownsample=False,
                )
                harmonic_marker_energies.append(marker_energy)
            self.photon_line_energies = tuple(harmonic_marker_energies)
            if photon_energy in self.photon_line_energies:
                self.photon_line_energy = photon_energy

        self._set_fixed_ranges()
        self.legend_labels = tuple(legend_labels)
        self.plotted_labels = tuple(legend_labels)
        self.legend_label.set_entries(legend_entries)
        self.legend_label.setVisible(bool(legend_entries))
        self._sync_visible_toggled_legend_labels()
        self._set_hover_lookup(hover_series)
        self._update_minimum_height()
        self._apply_curve_emphasis()
        self._last_state = state

    def clear_element(self, message: str) -> None:
        self.plot_item.clear()
        self._ensure_hover_plot_items()
        self._reset_hover_display()
        self._reset_hover_lookup()
        self.legend_label.clear()
        self.legend_label.setVisible(False)
        self.legend_labels = ()
        self.plotted_labels = ()
        self.photon_line_energy = None
        self.photon_line_energies = ()
        self._curve_items = {}
        self._curve_key_by_label = {}
        self._curve_label_by_key = {}
        self._active_legend_label = None
        self._toggled_legend_keys = set()
        self.empty_label.setText(message)
        self._stack.setCurrentWidget(self.empty_label)
        self._update_minimum_height()
        self._last_state = None
