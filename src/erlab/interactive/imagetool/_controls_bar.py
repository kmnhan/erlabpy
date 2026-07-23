"""Responsive control strip for ImageTool windows."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive._widgets import _Separator


class _ControlsBar(QtWidgets.QScrollArea):
    """Horizontally scrollable strip of ImageTool control groups."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("itoolControlsBar")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidgetResizable(False)
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self._contents = QtWidgets.QWidget()
        self._contents.setObjectName("itoolControlsContents")
        self._layout = QtWidgets.QHBoxLayout(self._contents)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)
        self._layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        self.setWidget(self._contents)

        self._previous_button = self._create_scroll_button(
            "itoolControlsPreviousButton",
            QtCore.Qt.ArrowType.LeftArrow,
            "Show previous controls",
            -1,
        )
        self._next_button = self._create_scroll_button(
            "itoolControlsNextButton",
            QtCore.Qt.ArrowType.RightArrow,
            "Show more controls",
            1,
        )

        scroll_bar = typing.cast("QtWidgets.QScrollBar", self.horizontalScrollBar())
        scroll_bar.rangeChanged.connect(self._update_navigation)
        scroll_bar.valueChanged.connect(self._update_navigation)
        self._update_navigation()

    def _create_scroll_button(
        self,
        object_name: str,
        arrow_type: QtCore.Qt.ArrowType,
        description: str,
        direction: int,
    ) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self.viewport())
        button.setObjectName(object_name)
        button.setArrowType(arrow_type)
        button.setToolTip(description)
        button.setAccessibleName(description)
        button.setAutoRepeat(True)
        button.setAutoRepeatDelay(300)
        button.setAutoRepeatInterval(80)
        button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        button.clicked.connect(lambda: self._scroll_page(direction))
        return button

    def add_group(self, widget: QtWidgets.QWidget, accessible_name: str) -> None:
        """Append a control group, separated from preceding groups by a line."""
        if self._layout.count():
            separator = _Separator(QtCore.Qt.Orientation.Vertical, self._contents)
            separator.setObjectName("itoolControlsSeparator")
            self._layout.addWidget(separator)

        widget.setAccessibleName(accessible_name)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._layout.addWidget(widget)

    def viewportSizeHint(self) -> QtCore.QSize:
        return self._contents.sizeHint()

    def sizeHint(self) -> QtCore.QSize:
        return self.viewportSizeHint()

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = self.sizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._position_navigation()
        self._update_navigation()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self._position_navigation()
        self._update_navigation()

    def wheelEvent(self, event: QtGui.QWheelEvent | None) -> None:
        scroll_bar = typing.cast("QtWidgets.QScrollBar", self.horizontalScrollBar())
        if event is None or scroll_bar.maximum() == scroll_bar.minimum():
            super().wheelEvent(event)
            return

        pixel_delta = event.pixelDelta()
        delta = pixel_delta.x()
        if delta == 0:
            delta = pixel_delta.y()
        if delta == 0:
            angle_delta = event.angleDelta()
            delta = angle_delta.x()
            if delta == 0:
                delta = angle_delta.y()
            delta = round(delta / 120 * scroll_bar.singleStep())
        if delta:
            scroll_bar.setValue(scroll_bar.value() - delta)
            event.accept()
            return
        super().wheelEvent(event)

    def _scroll_page(self, direction: int) -> None:
        scroll_bar = typing.cast("QtWidgets.QScrollBar", self.horizontalScrollBar())
        viewport = typing.cast("QtWidgets.QWidget", self.viewport())
        step = max(1, viewport.width() - self._next_button.width())
        scroll_bar.setValue(scroll_bar.value() + direction * step)

    def _position_navigation(self) -> None:
        viewport = typing.cast("QtWidgets.QWidget", self.viewport())
        style = self.style() or QtWidgets.QApplication.style()
        if style is None:  # pragma: no cover
            return
        width = style.pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent, None, self
        )
        width = max(16, width)
        height = viewport.height()
        self._previous_button.setGeometry(0, 0, width, height)
        self._next_button.setGeometry(
            max(0, viewport.width() - width), 0, width, height
        )
        self._previous_button.raise_()
        self._next_button.raise_()

    def _update_navigation(self, *args: int) -> None:
        del args
        scroll_bar = typing.cast("QtWidgets.QScrollBar", self.horizontalScrollBar())
        has_overflow = scroll_bar.maximum() > scroll_bar.minimum()
        self._previous_button.setVisible(
            has_overflow and scroll_bar.value() > scroll_bar.minimum()
        )
        self._next_button.setVisible(
            has_overflow and scroll_bar.value() < scroll_bar.maximum()
        )
