"""Visual controls and geometry helpers for arranging managed windows."""

from __future__ import annotations

import math
import typing

from qtpy import QtCore, QtGui, QtWidgets

if typing.TYPE_CHECKING:
    from collections.abc import Sequence


_LayoutMode = typing.Literal["grid_row", "grid_column", "row", "column"]

_LAYOUT_LABELS: dict[_LayoutMode, str] = {
    "grid_row": "Grid by Row",
    "grid_column": "Grid by Column",
    "row": "Arrange by Row",
    "column": "Arrange by Column",
}


def _partition_axis(
    origin: int, extent: int, count: int, spacing: int
) -> list[tuple[int, int]]:
    available = extent - spacing * (count - 1)
    if available <= 0:
        raise ValueError("Spacing leaves no room for windows")
    base, remainder = divmod(available, count)
    output: list[tuple[int, int]] = []
    position = origin
    for index in range(count):
        size = base + (index < remainder)
        output.append((position, size))
        position += size + spacing
    return output


def window_layout_rects(
    bounds: QtCore.QRect,
    count: int,
    mode: _LayoutMode,
    primary_count: int,
    spacing: int,
    reverse: bool,
) -> list[QtCore.QRect]:
    """Return equal-cell outer-frame rectangles in traversal order."""
    if count < 1:
        raise ValueError("Window count must be positive")
    if primary_count < 1:
        raise ValueError("Primary layout count must be positive")
    if spacing < 0:
        raise ValueError("Window spacing cannot be negative")
    if mode == "grid_row":
        columns = min(count, primary_count)
        rows = math.ceil(count / columns)
    elif mode == "grid_column":
        rows = min(count, primary_count)
        columns = math.ceil(count / rows)
    elif mode == "row":
        rows, columns = 1, count
    else:
        rows, columns = count, 1

    x_parts = _partition_axis(bounds.x(), bounds.width(), columns, spacing)
    y_parts = _partition_axis(bounds.y(), bounds.height(), rows, spacing)
    output: list[QtCore.QRect] = []
    for index in range(count):
        if mode in {"grid_row", "row"}:
            row, column = divmod(index, columns)
        else:
            column, row = divmod(index, rows)
        if reverse:
            column = columns - 1 - column
        x, width = x_parts[column]
        y, height = y_parts[row]
        output.append(QtCore.QRect(x, y, width, height))
    return output


def _layout_icon(
    widget: QtWidgets.QWidget,
    mode: _LayoutMode,
    *,
    reverse: bool = False,
) -> QtGui.QIcon:
    logical_size = QtCore.QSize(32, 28)
    ratio = max(1.0, widget.devicePixelRatioF())
    pixmap = QtGui.QPixmap(
        round(logical_size.width() * ratio), round(logical_size.height() * ratio)
    )
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if reverse and mode != "column":
            painter.translate(logical_size.width(), 0)
            painter.scale(-1, 1)
        color = widget.palette().color(QtGui.QPalette.ColorRole.ButtonText)
        painter.setBrush(color)
        if mode == "grid_row":
            painter.translate(3.5, 1)
        elif mode == "grid_column":
            painter.translate(3, 1)
        elif mode == "row":
            painter.translate(3, 0)
        else:
            painter.translate(0, 2)
        painter.setPen(
            QtGui.QPen(
                color,
                2.0,
                QtCore.Qt.PenStyle.SolidLine,
                QtCore.Qt.PenCapStyle.FlatCap,
                QtCore.Qt.PenJoinStyle.MiterJoin,
            )
        )
        if mode == "grid_row":
            painter.drawPolyline(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(5.5, 5.5),
                        QtCore.QPointF(20.5, 5.5),
                        QtCore.QPointF(5.5, 20.5),
                        QtCore.QPointF(16, 20.5),
                    ]
                )
            )
        elif mode == "grid_column":
            painter.drawPolyline(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(5.5, 5.5),
                        QtCore.QPointF(5.5, 20.5),
                        QtCore.QPointF(20.5, 5.5),
                        QtCore.QPointF(20.5, 18),
                    ]
                )
            )
        painter.setPen(QtCore.Qt.PenStyle.NoPen)

        if mode == "grid_row":
            for rectangle in (
                QtCore.QRectF(2, 2, 7, 7),
                QtCore.QRectF(17, 2, 7, 7),
                QtCore.QRectF(2, 17, 7, 7),
            ):
                painter.drawRoundedRect(rectangle, 1, 1)
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(24, 20.5),
                        QtCore.QPointF(16, 14.5),
                        QtCore.QPointF(16, 26.5),
                    ]
                )
            )
        elif mode == "grid_column":
            for rectangle in (
                QtCore.QRectF(2, 2, 7, 7),
                QtCore.QRectF(2, 17, 7, 7),
                QtCore.QRectF(17, 2, 7, 7),
            ):
                painter.drawRoundedRect(rectangle, 1, 1)
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(20.5, 25),
                        QtCore.QPointF(14, 18),
                        QtCore.QPointF(27, 18),
                    ]
                )
            )
        elif mode == "row":
            painter.drawRect(QtCore.QRectF(6, 13, 13, 2))
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(27, 14),
                        QtCore.QPointF(19, 8),
                        QtCore.QPointF(19, 20),
                    ]
                )
            )
            painter.drawRoundedRect(QtCore.QRectF(2, 10, 8, 8), 1, 1)
        else:
            painter.drawRect(QtCore.QRectF(15, 5, 2, 12.5))
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(16, 26),
                        QtCore.QPointF(10, 17.5),
                        QtCore.QPointF(22, 17.5),
                    ]
                )
            )
            painter.drawRoundedRect(QtCore.QRectF(12, 1, 8, 8), 1, 1)
    finally:
        painter.end()
    return QtGui.QIcon(pixmap)


def _direction_icon(widget: QtWidgets.QWidget, *, reverse: bool) -> QtGui.QIcon:
    size = QtCore.QSize(18, 18)
    ratio = max(1.0, widget.devicePixelRatioF())
    pixmap = QtGui.QPixmap(round(size.width() * ratio), round(size.height() * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = widget.palette().color(QtGui.QPalette.ColorRole.ButtonText)
        painter.translate(0.75 if reverse else -0.75, 0)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        if reverse:
            painter.drawRect(QtCore.QRectF(6, 7.5, 9, 3))
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(2, 9),
                        QtCore.QPointF(7, 4.5),
                        QtCore.QPointF(7, 13.5),
                    ]
                )
            )
        else:
            painter.drawRect(QtCore.QRectF(3, 7.5, 9, 3))
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(16, 9),
                        QtCore.QPointF(11, 4.5),
                        QtCore.QPointF(11, 13.5),
                    ]
                )
            )
    finally:
        painter.end()
    return QtGui.QIcon(pixmap)


class _WindowLayoutDialog(QtWidgets.QDialog):
    """Illustrator-inspired controls for arranging selected windows."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Arrange Selected Windows")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._count_initialized = False

        main_layout = QtWidgets.QVBoxLayout(self)
        options = QtWidgets.QFormLayout()
        options.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        layout_widget = QtWidgets.QWidget(self)
        layout_row = QtWidgets.QHBoxLayout(layout_widget)
        layout_row.setContentsMargins(0, 0, 0, 0)
        layout_row.setSpacing(4)
        self.layout_group = QtWidgets.QButtonGroup(self)
        self.layout_group.setExclusive(True)
        self.layout_buttons: dict[_LayoutMode, QtWidgets.QToolButton] = {}
        for mode, label in _LAYOUT_LABELS.items():
            button = QtWidgets.QToolButton(layout_widget)
            button.setObjectName(f"manager_window_layout_{mode}_button")
            button.setProperty("layoutMode", mode)
            button.setCheckable(True)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setIconSize(QtCore.QSize(24, 22))
            button.setFixedSize(32, 32)
            button.setAccessibleName(label)
            button.setToolTip(f"Arrange selected windows using {label.lower()}.")
            self.layout_group.addButton(button)
            self.layout_buttons[mode] = button
            layout_row.addWidget(button)
        layout_row.addStretch()
        options.addRow("Layout", layout_widget)
        self.layout_buttons["grid_row"].setChecked(True)
        self.layout_group.buttonToggled.connect(self._layout_changed)

        direction_widget = QtWidgets.QWidget(self)
        direction_layout = QtWidgets.QHBoxLayout(direction_widget)
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.setSpacing(4)
        self.direction_group = QtWidgets.QButtonGroup(self)
        self.direction_group.setExclusive(True)
        self.direction_buttons: dict[bool, QtWidgets.QToolButton] = {}
        for reverse in (True, False):
            label = "Right to Left" if reverse else "Left to Right"
            button = QtWidgets.QToolButton(direction_widget)
            button.setObjectName(
                "manager_window_layout_reverse_button"
                if reverse
                else "manager_window_layout_forward_button"
            )
            button.setProperty("layoutReverse", reverse)
            button.setCheckable(True)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setIconSize(QtCore.QSize(18, 18))
            button.setFixedSize(32, 32)
            button.setAccessibleName(label)
            button.setToolTip(f"Arrange selected windows {label.lower()}.")
            button.setIcon(_direction_icon(self, reverse=reverse))
            self.direction_group.addButton(button)
            self.direction_buttons[reverse] = button
            direction_layout.addWidget(button)
        direction_layout.addStretch()
        self.direction_buttons[False].setChecked(True)
        self.direction_group.buttonToggled.connect(self._order_changed)
        options.addRow("Layout order", direction_widget)

        self.primary_count_spin = QtWidgets.QSpinBox(self)
        self.primary_count_spin.setObjectName("manager_window_layout_primary_count")
        self.primary_count_spin.setRange(1, 2)
        self.primary_count_spin.setValue(2)
        self.primary_count_spin.setMaximumWidth(100)
        self.primary_count_label = QtWidgets.QLabel("Columns", self)
        options.addRow(self.primary_count_label, self.primary_count_spin)

        self.spacing_spin = QtWidgets.QSpinBox(self)
        self.spacing_spin.setObjectName("manager_window_layout_spacing")
        self.spacing_spin.setRange(0, 256)
        self.spacing_spin.setValue(0)
        self.spacing_spin.setSuffix(" px")
        self.spacing_spin.setMaximumWidth(100)
        options.addRow("Spacing", self.spacing_spin)
        main_layout.addLayout(options)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)
        self._refresh_layout_icons()
        self._layout_changed()

    @property
    def layout_mode(self) -> _LayoutMode:
        button = self.layout_group.checkedButton()
        if button is None:  # pragma: no cover
            return "grid_row"
        return typing.cast("_LayoutMode", button.property("layoutMode"))

    @property
    def reverse_order(self) -> bool:
        button = self.direction_group.checkedButton()
        return bool(button is not None and button.property("layoutReverse"))

    @property
    def primary_count(self) -> int:
        return self.primary_count_spin.value()

    @property
    def spacing(self) -> int:
        return self.spacing_spin.value()

    def set_window_count(self, count: int) -> None:
        count = max(2, count)
        self.primary_count_spin.setMaximum(count)
        if not self._count_initialized:
            self.primary_count_spin.setValue(math.ceil(math.sqrt(count)))
            self._count_initialized = True
        elif self.primary_count_spin.value() > count:
            self.primary_count_spin.setValue(count)

    @QtCore.Slot()
    @QtCore.Slot(QtWidgets.QAbstractButton, bool)
    def _layout_changed(
        self,
        _button: QtWidgets.QAbstractButton | None = None,
        checked: bool = True,
    ) -> None:
        if not checked:
            return
        mode = self.layout_mode
        single_axis = mode in {"row", "column"}
        self.primary_count_label.setText("Rows" if mode == "grid_column" else "Columns")
        self.primary_count_label.setEnabled(not single_axis)
        self.primary_count_spin.setEnabled(not single_axis)

    @QtCore.Slot(QtWidgets.QAbstractButton, bool)
    def _order_changed(self, _button: QtWidgets.QAbstractButton, checked: bool) -> None:
        if checked:
            self._refresh_layout_icons()

    def _refresh_layout_icons(self) -> None:
        for mode, button in self.layout_buttons.items():
            button.setIcon(_layout_icon(self, mode, reverse=self.reverse_order))

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if event is not None and event.type() in {
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._refresh_layout_icons()
            for reverse, button in self.direction_buttons.items():
                button.setIcon(_direction_icon(self, reverse=reverse))


def frame_margins(window: QtWidgets.QWidget) -> QtCore.QMargins:
    """Return the current native frame margins around a top-level widget."""
    frame = window.frameGeometry()
    client = window.geometry()
    return QtCore.QMargins(
        client.left() - frame.left(),
        client.top() - frame.top(),
        frame.right() - client.right(),
        frame.bottom() - client.bottom(),
    )


def frame_rects_fit_windows(
    windows: Sequence[QtWidgets.QWidget], frame_rects: Sequence[QtCore.QRect]
) -> bool:
    """Return whether each frame cell can contain its window's minimum size."""
    for window, rectangle in zip(windows, frame_rects, strict=True):
        minimum = window.minimumSize()
        margins = frame_margins(window)
        if (
            rectangle.width() < minimum.width() + margins.left() + margins.right()
            or rectangle.height() < minimum.height() + margins.top() + margins.bottom()
        ):
            return False
    return True
