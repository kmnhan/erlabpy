"""Numeric range controls for the angle calculator."""

import math
import typing

import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

import erlab


class DraggableCurveItem(pg.PlotCurveItem):
    """Report drags on the curve without translating the graphics item."""

    sigDragStarted = QtCore.Signal(QtCore.QPointF)
    sigDragged = QtCore.Signal(QtCore.QPointF)
    sigDragFinished = QtCore.Signal()

    def __init__(self, *args, pen=None, hover_pen=None, **kwargs):
        self._normal_pen = (
            pg.mkPen("#d55e00", width=3) if pen is None else pg.mkPen(pen)
        )
        self._hover_pen = (
            pg.mkPen("#ff8050", width=4) if hover_pen is None else pg.mkPen(hover_pen)
        )
        self._dragging = False
        self._hovering = False
        kwargs.setdefault("connect", "finite")
        super().__init__(*args, pen=self._normal_pen, **kwargs)
        self.setClickable(True, width=12)

    def viewTransformChanged(self):
        self._mouseShape = None
        super().viewTransformChanged()

    def cancelDrag(self) -> None:
        """Clear an interrupted drag without emitting drag signals."""
        if not self._dragging:
            return
        self._dragging = False
        self._hovering = False
        self._update_drag_style()

    def _update_drag_style(self):
        self.setPen(
            self._hover_pen if self._hovering or self._dragging else self._normal_pen
        )

    def hoverEvent(self, event):
        self._hovering = (
            not event.isExit()
            and self.mouseShape().contains(event.pos())
            and event.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        )
        self._update_drag_style()

    def mouseDragEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        if event.isStart():
            if not self.mouseShape().contains(event.buttonDownPos()):
                event.ignore()
                return
            self._dragging = True
            self._update_drag_style()
            self.sigDragStarted.emit(QtCore.QPointF(event.buttonDownPos()))
        if not self._dragging:
            event.ignore()
            return
        event.accept()
        self.sigDragged.emit(QtCore.QPointF(event.pos()))
        if event.isFinish():
            self._dragging = False
            self._hovering = self.mouseShape().contains(event.pos())
            self._update_drag_style()
            self.sigDragFinished.emit()


class RichTextCheckBox(QtWidgets.QCheckBox):
    """Keep native checkbox behavior with a formatted, mouse-transparent label."""

    def __init__(self, text: str, rich_text: str, parent=None):
        super().__init__(text, parent)
        self._label = QtWidgets.QLabel(rich_text, self)
        self._label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._label.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._label.setForegroundRole(self.foregroundRole())
        self._label.setMargin(0)
        self._label.setIndent(0)
        self._label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

    def sizeHint(self) -> QtCore.QSize:
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        return (
            typing.cast("QtWidgets.QStyle", self.style())
            .sizeFromContents(
                QtWidgets.QStyle.ContentsType.CT_CheckBox,
                option,
                self._label.sizeHint(),
                self,
            )
            .expandedTo(super().sizeHint())
        )

    def hitButton(self, position: QtCore.QPoint) -> bool:
        return super().hitButton(position) or self._label.geometry().contains(position)

    def paintEvent(self, event):
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        style = self.style()
        contents = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_CheckBoxContents, option, self
        )
        self._label.setGeometry(
            QtWidgets.QStyle.alignedRect(
                self.layoutDirection(),
                QtCore.Qt.AlignmentFlag.AlignLeading
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
                self._label.sizeHint(),
                contents,
            )
        )
        focus_rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_CheckBoxFocusRect, option, self
        )
        focused = bool(option.state & QtWidgets.QStyle.StateFlag.State_HasFocus)
        option.text = ""
        option.state &= ~QtWidgets.QStyle.StateFlag.State_HasFocus
        painter = QtWidgets.QStylePainter(self)
        painter.drawControl(QtWidgets.QStyle.ControlElement.CE_CheckBox, option)
        if focused:
            focus = QtWidgets.QStyleOptionFocusRect()
            focus.initFrom(self)
            focus.rect = focus_rect.united(self._label.geometry())
            focus.backgroundColor = self.palette().window().color()
            painter.drawPrimitive(
                QtWidgets.QStyle.PrimitiveElement.PE_FrameFocusRect, focus
            )


class _RangeSlider(QtWidgets.QSlider):
    """A native horizontal slider with two independently movable endpoints."""

    sigValueChanged = QtCore.Signal(object)
    sigDragStarted = QtCore.Signal()
    sigDragFinished = QtCore.Signal()
    sigDragCanceled = QtCore.Signal()
    sigEditingFinished = QtCore.Signal()

    def __init__(
        self, bounds, values, parent=None, *, logarithmic=False, snap_to_integer=False
    ):
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setRange(0, 1_000_000)
        self.setMinimumWidth(90)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._logarithmic = logarithmic
        self._snap_to_integer = snap_to_integer
        self._bounds = bounds
        self._values = values
        self._selected = 0
        self._drag: str | None = None
        self._press_values = values
        self._press_value = 0.0
        self._offset = 0.0

    def values(self):
        return self._values

    def setBounds(self, bounds):
        self._bounds = tuple(sorted(float(value) for value in bounds))
        self.setValues(self._values)
        self.update()

    def setValues(self, values):
        values = tuple(
            sorted(max(self._bounds[0], min(self._bounds[1], float(v))) for v in values)
        )
        if values != self._values:
            self._values = values
            self.update()
            self.sigValueChanged.emit(values)

    def _option(self, index):
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        lower, upper = self._bounds
        value = self._values[index]
        if self._logarithmic:
            lower, upper, value = math.log(lower), math.log(upper), math.log(value)
        span = upper - lower
        option.sliderPosition = (
            round((value - lower) / span * self.maximum()) if span else 0
        )
        option.sliderValue = option.sliderPosition
        option.activeSubControls = QtWidgets.QStyle.SubControl.SC_SliderHandle
        if index != self._selected:
            option.state &= ~QtWidgets.QStyle.StateFlag.State_HasFocus
        if self._drag == "handle" and index == self._selected:
            option.state |= QtWidgets.QStyle.StateFlag.State_Sunken
        return option

    def _handle_rect(self, index):
        return self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            self._option(index),
            QtWidgets.QStyle.SubControl.SC_SliderHandle,
            self,
        )

    def _value_at(self, x):
        option = self._option(0)
        option.sliderPosition = 0
        first = (
            self.style()
            .subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderHandle,
                self,
            )
            .center()
            .x()
        )
        option.sliderPosition = self.maximum()
        last = (
            self.style()
            .subControlRect(
                QtWidgets.QStyle.ComplexControl.CC_Slider,
                option,
                QtWidgets.QStyle.SubControl.SC_SliderHandle,
                self,
            )
            .center()
            .x()
        )
        fraction = (x - first) / (last - first) if last != first else 0.0
        lower, upper = self._bounds
        if self._logarithmic:
            fraction = max(0.0, min(1.0, fraction))
            return math.exp(math.log(lower) + fraction * math.log(upper / lower))
        return lower + fraction * (upper - lower)

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        option = self._option(self._selected)
        option.sliderPosition = option.sliderValue = option.minimum
        option.subControls = QtWidgets.QStyle.SubControl.SC_SliderGroove
        painter.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_Slider, option)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            option,
            QtWidgets.QStyle.SubControl.SC_SliderGroove,
            self,
        )
        left, right = sorted(self._handle_rect(i).center().x() for i in (0, 1))
        height = min(
            groove.height(),
            max(
                3,
                self.style().pixelMetric(
                    QtWidgets.QStyle.PixelMetric.PM_DefaultFrameWidth, option, self
                )
                * 2,
            ),
        )
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(option.palette.highlight())
        painter.drawRoundedRect(
            QtCore.QRectF(left, groove.center().y() - height / 2, right - left, height),
            height / 2,
            height / 2,
        )
        for index in (1 - self._selected, self._selected):
            option = self._option(index)
            option.subControls = QtWidgets.QStyle.SubControl.SC_SliderHandle
            painter.drawComplexControl(
                QtWidgets.QStyle.ComplexControl.CC_Slider, option
            )

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        x = event.position().x()
        handles = [self._handle_rect(index) for index in (0, 1)]
        hits = [
            index
            for index, rect in enumerate(handles)
            if rect.contains(event.position().toPoint())
        ]
        self.sigDragStarted.emit()
        self._press_values = self._values
        self._press_value = self._value_at(x)
        if hits:
            self._selected = (
                hits[0] if len(hits) == 1 else int(x >= handles[0].center().x())
            )
            self._drag = "handle"
            self._offset = self._press_value - self._values[self._selected]
        elif (
            min(rect.center().x() for rect in handles)
            < x
            < max(rect.center().x() for rect in handles)
        ):
            self._drag = "interval"
        else:
            self._selected = min(
                (0, 1), key=lambda index: abs(x - handles[index].center().x())
            )
            self._drag = "handle"
            self._offset = 0.0
            self._move_handle(self._press_value)
        self.update()
        event.accept()

    def _move_handle(self, value):
        values = list(self._values)
        if self._snap_to_integer:
            value = round(value)
            value = (
                min(value, math.floor(values[1]))
                if self._selected == 0
                else max(value, math.ceil(values[0]))
            )
        values[self._selected] = (
            min(value, values[1]) if self._selected == 0 else max(value, values[0])
        )
        self.setValues(values)

    def mouseMoveEvent(self, event):
        if self._drag is None or not event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        value = self._value_at(event.position().x())
        if self._drag == "interval":
            delta = max(
                self._bounds[0] - self._press_values[0],
                min(self._bounds[1] - self._press_values[1], value - self._press_value),
            )
            values = tuple(endpoint + delta for endpoint in self._press_values)
            if self._snap_to_integer:
                values = tuple(round(endpoint) for endpoint in values)
            self.setValues(values)
        else:
            self._move_handle(value - self._offset)
        event.accept()

    def mouseReleaseEvent(self, event):
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._drag is not None
        ):
            self._finish_drag()
            event.accept()
        else:
            event.ignore()

    def cancelDrag(self) -> None:
        # History restoration owns the final value and must discard later moves.
        self._drag = None
        self.update()

    def _finish_drag(self) -> None:
        if getattr(self, "_drag", None) is not None:
            self.cancelDrag()
            self.sigDragFinished.emit()
            self.sigEditingFinished.emit()

    def event(self, event):
        if event.type() in (
            QtCore.QEvent.Type.FocusOut,
            QtCore.QEvent.Type.Hide,
            QtCore.QEvent.Type.UngrabMouse,
            QtCore.QEvent.Type.WindowDeactivate,
        ) or (
            event.type() == QtCore.QEvent.Type.EnabledChange and not self.isEnabled()
        ):
            self._finish_drag()
        return super().event(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key.Key_Escape and self._drag is not None:
            self.cancelDrag()
            self.setValues(self._press_values)
            self.sigDragCanceled.emit()
        elif key in (
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_PageDown,
            QtCore.Qt.Key.Key_PageUp,
            QtCore.Qt.Key.Key_Home,
            QtCore.Qt.Key.Key_End,
        ):
            if key == QtCore.Qt.Key.Key_Home:
                value = self._bounds[0]
            elif key == QtCore.Qt.Key.Key_End:
                value = self._bounds[1]
            else:
                step = (
                    10
                    if key in (QtCore.Qt.Key.Key_PageDown, QtCore.Qt.Key.Key_PageUp)
                    else 1
                )
                if key in (
                    QtCore.Qt.Key.Key_Left,
                    QtCore.Qt.Key.Key_Down,
                    QtCore.Qt.Key.Key_PageDown,
                ):
                    step = -step
                value = self._values[self._selected] + step
            self._move_handle(value)
            self.sigEditingFinished.emit()
        else:
            event.ignore()
            return
        event.accept()

    def wheelEvent(self, event):
        # Leave scrolling to the surrounding settings panel.
        event.ignore()

    def focusInEvent(self, event):
        if event.reason() == QtCore.Qt.FocusReason.TabFocusReason:
            self._selected = 0
        elif event.reason() == QtCore.Qt.FocusReason.BacktabFocusReason:
            self._selected = 1
        super().focusInEvent(event)

    def focusNextPrevChild(self, forward):
        if self.hasFocus() and (
            (forward and self._selected == 0) or (not forward and self._selected == 1)
        ):
            self._selected = 1 - self._selected
            self.update()
            return True
        return super().focusNextPrevChild(forward)


class NumericRangeControl(QtWidgets.QWidget):
    """Edit a numeric interval with two handles and unit-aware endpoints."""

    sigValueChanged = QtCore.Signal(object)
    sigDragStarted = QtCore.Signal()
    sigDragFinished = QtCore.Signal()
    sigDragCanceled = QtCore.Signal()
    sigEditingFinished = QtCore.Signal()

    def __init__(
        self,
        bounds=(-90.0, 90.0),
        values=(-15.0, 15.0),
        parent=None,
        *,
        suffix="°",
        decimals=2,
        significant_digits: int | None = None,
        step=0.1,
        logarithmic=False,
        snap_to_integer=False,
    ):
        super().__init__(parent)
        self._bounds = tuple(sorted(float(value) for value in bounds))
        self._decimals = decimals
        self.lower_spin, self.upper_spin = (
            erlab.interactive.utils.BetterSpinBox(
                self,
                compact=False,
                decimals=decimals if significant_digits is None else significant_digits,
                significant=significant_digits is not None,
                suffix=suffix,
                minimum=self._bounds[0],
                maximum=self._bounds[1],
                singleStep=step,
            )
            for _ in range(2)
        )
        for spin in (self.lower_spin, self.upper_spin):
            spin.editingFinished.connect(self.sigEditingFinished.emit)
        self.slider = _RangeSlider(
            self._bounds,
            values,
            self,
            logarithmic=logarithmic,
            snap_to_integer=snap_to_integer,
        )
        self.lower_spin.setToolTip("Minimum value")
        self.upper_spin.setToolTip("Maximum value")
        QtWidgets.QWidget.setTabOrder(self.lower_spin, self.upper_spin)
        QtWidgets.QWidget.setTabOrder(self.upper_spin, self.slider)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(0)
        layout.addWidget(self.lower_spin, 0, 0)
        layout.addWidget(self.upper_spin, 0, 1)
        layout.addWidget(self.slider, 1, 0, 1, 2)
        self._values = (float("nan"), float("nan"))
        self.setValues(values)
        self.lower_spin.valueChanged.connect(self._spin_changed)
        self.upper_spin.valueChanged.connect(self._spin_changed)
        self.slider.sigValueChanged.connect(self.setValues)
        self.slider.sigDragStarted.connect(self.sigDragStarted.emit)
        self.slider.sigDragFinished.connect(self.sigDragFinished.emit)
        self.slider.sigDragCanceled.connect(self.sigDragCanceled.emit)
        self.slider.sigEditingFinished.connect(self.sigEditingFinished.emit)

    def values(self) -> tuple[float, float]:
        return self._values

    def setBounds(self, bounds: tuple[float, float]) -> None:
        self._bounds = tuple(sorted(float(value) for value in bounds))
        with (
            QtCore.QSignalBlocker(self.lower_spin),
            QtCore.QSignalBlocker(self.upper_spin),
            QtCore.QSignalBlocker(self.slider),
        ):
            self.lower_spin.setRange(*self._bounds)
            self.upper_spin.setRange(*self._bounds)
            self.slider.setBounds(self._bounds)
        self.setValues(self._values)

    def setValues(self, values: tuple[float, float]) -> None:
        lower, upper = sorted(
            round(
                max(self._bounds[0], min(self._bounds[1], float(value))), self._decimals
            )
            for value in values
        )
        changed = (lower, upper) != self._values
        self._values = (lower, upper)
        with (
            QtCore.QSignalBlocker(self.lower_spin),
            QtCore.QSignalBlocker(self.upper_spin),
            QtCore.QSignalBlocker(self.slider),
        ):
            self.lower_spin.setValue(lower)
            self.upper_spin.setValue(upper)
            self.slider.setValues((lower, upper))
        if changed:
            self.sigValueChanged.emit(self._values)

    def _spin_changed(self) -> None:
        self.setValues((self.lower_spin.value(), self.upper_spin.value()))
