from qtpy import QtCore, QtGui, QtWidgets


class _Separator(QtWidgets.QWidget):
    """Draw a subtle, palette-aware one-pixel separator."""

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Horizontal,
        parent: QtWidgets.QWidget | None = None,
        *,
        inset: int = 0,
    ) -> None:
        super().__init__(parent)
        if inset < 0:
            raise ValueError("Separator inset cannot be negative")
        self._orientation = orientation
        self._inset = inset
        if orientation == QtCore.Qt.Orientation.Horizontal:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        else:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )

    @property
    def orientation(self) -> QtCore.Qt.Orientation:
        """Return the direction in which the separator extends."""
        return self._orientation

    @property
    def inset(self) -> int:
        """Return the space left at both ends of the separator."""
        return self._inset

    def sizeHint(self) -> QtCore.QSize:
        length = 2 * self._inset + 1
        if self._orientation == QtCore.Qt.Orientation.Horizontal:
            return QtCore.QSize(length, 1)
        return QtCore.QSize(1, length)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        del event
        if self._orientation == QtCore.Qt.Orientation.Horizontal:
            if self.width() <= 2 * self._inset:
                return
            center = self.height() / 2
            start = QtCore.QPointF(self._inset, center)
            end = QtCore.QPointF(self.width() - self._inset, center)
        else:
            if self.height() <= 2 * self._inset:
                return
            center = self.width() / 2
            start = QtCore.QPointF(center, self._inset)
            end = QtCore.QPointF(center, self.height() - self._inset)

        color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
        background = self.palette().color(QtGui.QPalette.ColorRole.Window)
        color.setAlphaF(0.18 if background.lightnessF() < 0.5 else 0.14)

        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(color)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.drawLine(start, end)
        painter.end()


class _CenteredIconToolButton(QtWidgets.QToolButton):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)

    @staticmethod
    def _visible_pixmap_rect(pixmap: QtGui.QPixmap) -> QtCore.QRectF:
        dpr = pixmap.devicePixelRatioF()
        if dpr <= 0.0:
            dpr = 1.0
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        min_x = image.width()
        min_y = image.height()
        max_x = -1
        max_y = -1
        for y in range(image.height()):
            for x in range(image.width()):
                if image.pixelColor(x, y).alpha() <= 8:
                    continue
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        if max_x < 0 or max_y < 0:
            return QtCore.QRectF(
                QtCore.QPointF(0.0, 0.0),
                QtCore.QSizeF(pixmap.width() / dpr, pixmap.height() / dpr),
            )
        return QtCore.QRectF(
            min_x / dpr,
            min_y / dpr,
            (max_x - min_x + 1) / dpr,
            (max_y - min_y + 1) / dpr,
        )

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        side = max(hint.width(), hint.height())
        return QtCore.QSize(side, side)

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        side = max(hint.width(), hint.height())
        return QtCore.QSize(side, side)

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        del event
        option = QtWidgets.QStyleOptionToolButton()
        self.initStyleOption(option)
        icon = QtGui.QIcon(option.icon)
        icon_size = QtCore.QSize(option.iconSize)
        option.icon = QtGui.QIcon()
        option.text = ""

        painter = QtWidgets.QStylePainter(self)
        style = self.style() or QtWidgets.QApplication.style()
        if style is not None:
            style.drawComplexControl(
                QtWidgets.QStyle.ComplexControl.CC_ToolButton,
                option,
                painter,
                self,
            )

        if icon.isNull() or not icon_size.isValid():
            return
        mode = (
            QtGui.QIcon.Mode.Normal if self.isEnabled() else QtGui.QIcon.Mode.Disabled
        )
        state = QtGui.QIcon.State.On if self.isChecked() else QtGui.QIcon.State.Off
        pixmap = icon.pixmap(icon_size, mode, state)
        if pixmap.isNull():
            return
        dpr = pixmap.devicePixelRatioF()
        if dpr <= 0.0:
            dpr = 1.0
        pixmap_size = QtCore.QSizeF(pixmap.width() / dpr, pixmap.height() / dpr)
        visible_rect = self._visible_pixmap_rect(pixmap)
        button_center = QtCore.QRectF(self.rect()).center()
        visible_center = visible_rect.center()
        target = QtCore.QRectF(
            QtCore.QPointF(
                button_center.x() - visible_center.x(),
                button_center.y() - visible_center.y(),
            ),
            pixmap_size,
        )
        painter.drawPixmap(target, pixmap, QtCore.QRectF(pixmap.rect()))
