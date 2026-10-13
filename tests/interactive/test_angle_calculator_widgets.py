import pytest
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.angle_calculator._widgets import (
    NumericRangeControl,
    RichTextCheckBox,
)


@pytest.mark.parametrize(
    "direction",
    [QtCore.Qt.LayoutDirection.LeftToRight, QtCore.Qt.LayoutDirection.RightToLeft],
)
def test_rich_checkbox_interaction(qtbot, direction):
    widget = RichTextCheckBox("kz", "<i>k</i><sub>z</sub>")
    qtbot.addWidget(widget)
    widget.setLayoutDirection(direction)
    widget.resize(widget.sizeHint())
    widget.show()
    qtbot.waitExposed(widget)
    widget.activateWindow()
    QtWidgets.QApplication.setActiveWindow(widget)
    widget.setFocus()
    qtbot.waitUntil(widget.hasFocus)
    label = widget._label
    endpoint = label.geometry().center()
    endpoint.setX(
        label.geometry().right() - 1
        if direction == QtCore.Qt.LayoutDirection.LeftToRight
        else label.geometry().left() + 1
    )
    qtbot.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton, pos=endpoint)
    assert widget.isChecked()
    qtbot.keyClick(widget, QtCore.Qt.Key.Key_Space)
    assert not widget.isChecked()
    widget.setEnabled(False)
    qtbot.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton, pos=endpoint)
    assert not widget.isChecked()

    font = widget.font()
    font.setPointSizeF(font.pointSizeF() * 2)
    widget.setFont(font)
    widget.resize(widget.sizeHint())
    widget.setEnabled(True)
    qtbot.mouseClick(
        widget, QtCore.Qt.MouseButton.LeftButton, pos=label.geometry().center()
    )
    assert widget.isChecked()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((-90, 90), (-90, 90)),
        ((30, -20), (-20, 30)),
        ((4, 4), (4, 4)),
        ((-100, 200), (-90, 90)),
    ],
)
def test_angle_range(qtbot, values, expected):
    widget = NumericRangeControl()
    qtbot.addWidget(widget)
    received = []
    widget.sigValueChanged.connect(received.append)
    widget.setValues(values)
    assert widget.values() == expected
    assert widget.slider.values() == expected
    assert (widget.lower_spin.value(), widget.upper_spin.value()) == expected
    assert received == [expected]
    widget.setValues(values)
    assert received == [expected]


def test_angle_range_interaction(qtbot):
    widget = NumericRangeControl()
    qtbot.addWidget(widget)
    received = []
    widget.sigValueChanged.connect(received.append)
    widget.lower_spin.setValue(-22)
    assert received == [(-22, 15)]
    with qtbot.waitSignal(widget.sigEditingFinished):
        widget.upper_spin.editingFinished.emit()


@pytest.mark.parametrize("endpoint", [0, 1])
def test_range_rounds_spins_when_interval_is_unchanged(qtbot, endpoint):
    widget = NumericRangeControl()
    qtbot.addWidget(widget)
    received = []
    widget.sigValueChanged.connect(received.append)
    spin = (widget.lower_spin, widget.upper_spin)[endpoint]
    expected = spin.value()
    spin.lineEdit().setText(str(expected + 0.001))
    spin.editingFinished.emit()
    assert spin.value() == expected
    assert widget.values() == widget.slider.values() == (-15, 15)
    assert received == []


@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize("commit", [QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Tab])
def test_range_typing_updates_only_after_commit(qtbot, endpoint, commit):
    widget = NumericRangeControl(
        bounds=(5, 1000),
        values=(20, 80),
        suffix=" eV",
        step=1.0,
        snap_to_integer=True,
    )
    qtbot.addWidget(widget)
    widget.show()
    widget.activateWindow()
    QtWidgets.QApplication.setActiveWindow(widget)
    spin = (widget.lower_spin, widget.upper_spin)[endpoint]
    spin.setFocus()
    qtbot.waitUntil(spin.hasFocus)
    received = []
    widget.sigValueChanged.connect(received.append)
    spin.selectAll()
    qtbot.keyClicks(spin, ("21.2", "90.7")[endpoint])
    assert received == []
    assert widget.values() == widget.slider.values() == (20, 80)

    qtbot.keyClick(spin, commit)
    expected = (21.2, 80) if endpoint == 0 else (20, 90.7)
    assert widget.values() == widget.slider.values() == expected
    assert received == [expected]

    spin.setFocus()
    qtbot.keyClick(spin, QtCore.Qt.Key.Key_Up)
    stepped = (22.2, 80) if endpoint == 0 else (20, 91.7)
    assert widget.values() == widget.slider.values() == stepped
    assert received == [expected, stepped]
    # Match the calculator's focus cleanup before widget deletion.
    spin.clearFocus()


def _move_with_button(slider, position):
    local = QtCore.QPointF(position)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        local,
        QtCore.QPointF(slider.mapToGlobal(position)),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(slider, event)


@pytest.mark.parametrize("index", [0, 1])
def test_native_range_handle_drag(qtbot, index):
    widget = NumericRangeControl()
    qtbot.addWidget(widget)
    widget.resize(500, 50)
    widget.show()
    slider = widget.slider
    assert isinstance(slider, QtWidgets.QSlider)
    start = slider._handle_rect(index).center()
    received = []
    widget.sigValueChanged.connect(received.append)
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    end = start + QtCore.QPoint(-25 if index == 0 else 25, 0)
    _move_with_button(slider, end)
    assert received
    assert widget.values()[1 - index] == (-15, 15)[1 - index]
    assert widget.values()[index] < -15 if index == 0 else widget.values()[index] > 15
    with qtbot.waitSignal(widget.sigEditingFinished):
        qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert slider.values() == widget.values()
    assert (widget.lower_spin.value(), widget.upper_spin.value()) == pytest.approx(
        widget.values(), abs=0.001
    )


@pytest.mark.parametrize(("direction", "expected"), [(-1, (-90, -60)), (1, (60, 90))])
def test_native_range_interval_drag(qtbot, direction, expected):
    widget = NumericRangeControl()
    qtbot.addWidget(widget)
    widget.resize(500, 50)
    widget.show()
    slider = widget.slider
    start = (slider._handle_rect(0).center() + slider._handle_rect(1).center()) / 2
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    end = start + QtCore.QPoint(direction * 1000, 0)
    _move_with_button(slider, end)
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert widget.values() == pytest.approx(expected)


def test_native_range_keyboard_and_cancel(qtbot):
    widget = NumericRangeControl(values=(-12.5, 12.5))
    qtbot.addWidget(widget)
    widget.resize(500, 50)
    widget.show()
    slider = widget.slider
    widget.activateWindow()
    QtWidgets.QApplication.setActiveWindow(widget)
    slider.setFocus()
    qtbot.waitUntil(slider.hasFocus)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Right)
    assert widget.values() == (-11.5, 12.5)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Home)
    assert widget.values() == (-90, 12.5)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Tab)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_End)
    assert widget.values() == (-90, 90)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_PageDown)
    assert widget.values() == (-90, 80)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Home)
    assert widget.values() == (-90, -90)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Right)
    assert widget.values() == (-90, -89)
    widget.setValues((-10, 10))
    events = []
    widget.sigDragStarted.connect(lambda: events.append("started"))
    widget.sigDragFinished.connect(lambda: events.append("finished"))
    widget.sigDragCanceled.connect(lambda: events.append("canceled"))
    widget.sigEditingFinished.connect(lambda: events.append("edited"))
    start = slider._handle_rect(1).center()
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    _move_with_button(slider, start + QtCore.QPoint(30, 0))
    assert widget.values() != (-10, 10)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Escape)
    assert widget.values() == (-10, 10)
    _move_with_button(slider, start + QtCore.QPoint(60, 0))
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    assert widget.values() == (-10, 10)
    assert events == ["started", "canceled"]


def test_native_range_focus_scroll_and_collapse(qtbot):
    widget = NumericRangeControl(values=(0, 0))
    qtbot.addWidget(widget)
    widget.resize(500, 50)
    widget.show()
    widget.activateWindow()
    QtWidgets.QApplication.setActiveWindow(widget)
    slider = widget.slider
    widget.upper_spin.setFocus()
    qtbot.waitUntil(widget.upper_spin.hasFocus)
    qtbot.keyClick(widget.upper_spin, QtCore.Qt.Key.Key_Backtab)
    qtbot.waitUntil(widget.lower_spin.hasFocus)
    qtbot.keyClick(widget.lower_spin, QtCore.Qt.Key.Key_Backtab)
    qtbot.waitUntil(slider.hasFocus)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Right)
    assert widget.values() == (0, 1)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Backtab)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Left)
    assert widget.values() == (-1, 1)
    widget.setValues((0, 0))
    start = slider._handle_rect(0).center()
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    end = start + QtCore.QPoint(30, 0)
    _move_with_button(slider, end)
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert widget.values()[0] == 0
    assert widget.values()[1] > 0
    values = widget.values()
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(start),
        QtCore.QPointF(slider.mapToGlobal(start)),
        QtCore.QPoint(),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(slider, wheel)
    assert not wheel.isAccepted()
    assert widget.values() == values
    widget.setEnabled(False)
    assert not slider.grab().isNull()


def test_angle_range_changing_bounds(qtbot):
    widget = NumericRangeControl(values=(-60, 70))
    qtbot.addWidget(widget)
    received = []
    widget.sigValueChanged.connect(received.append)
    widget.setBounds((-45, 45))
    assert received == [(-45, 45)]
    assert widget.values() == widget.slider.values() == (-45, 45)
    widget.lower_spin.setValue(-90)
    widget.upper_spin.setValue(90)
    assert (widget.lower_spin.value(), widget.upper_spin.value()) == (-45, 45)
    widget.slider.setValues((-90, 90))
    assert widget.slider.values() == (-45, 45)
    widget.setBounds((-90, 90))
    assert received == [(-45, 45)]
    widget.slider.setValues((-80, 85))
    assert widget.values() == (-80, 85)
    assert (widget.lower_spin.value(), widget.upper_spin.value()) == (-80, 85)


def test_angle_range_precision(qtbot):
    widget = NumericRangeControl()
    qtbot.addWidget(widget)
    widget.setValues((-12.3456, 23.4567))
    assert widget.values() == widget.slider.values() == (-12.35, 23.46)
    assert (widget.lower_spin.value(), widget.upper_spin.value()) == (-12.35, 23.46)


def test_photon_energy_range_logarithmic_scale(qtbot):
    control = NumericRangeControl(
        bounds=(5, 1000),
        values=(5, 1000),
        suffix=" eV",
        decimals=3,
        significant_digits=4,
        step=1.0,
        logarithmic=True,
        snap_to_integer=True,
    )
    qtbot.addWidget(control)
    control.resize(360, 70)
    control.show()
    assert control.lower_spin.minimum() == 5
    assert control.upper_spin.minimum() == 5
    slider = control.slider
    midpoint = (
        slider._handle_rect(0).center().x() + slider._handle_rect(1).center().x()
    ) / 2
    assert slider._value_at(midpoint) == pytest.approx((5 * 1000) ** 0.5)
    assert slider._value_at(-100000) == pytest.approx(5)
    assert slider._value_at(100000) == pytest.approx(1000)
    control.setValues((6.23456, 12.34567))
    assert control.values() == (6.235, 12.346)
    control.upper_spin.setValue(6.235)
    assert control.values() == (6.235, 6.235)
    control.setValues((1, 4.999))
    assert control.values() == slider.values() == (5, 5)
    control.setValues((1000.001, 1001))
    assert control.values() == slider.values() == (1000, 1000)


@pytest.mark.parametrize(
    ("value", "displayed_value"),
    [(6.001, 6.001), (60.123, 60.12), (600.123, 600.1), (1000, 1000)],
)
def test_range_significant_display_preserves_values(qtbot, value, displayed_value):
    control = NumericRangeControl(
        bounds=(5, 1000),
        values=(value, value),
        suffix=" eV",
        decimals=3,
        significant_digits=4,
        step=1.0,
    )
    qtbot.addWidget(control)
    for spin in (control.lower_spin, control.upper_spin):
        assert isinstance(spin, erlab.interactive.utils.BetterSpinBox)
        assert spin.valueFromText(spin.text()) == displayed_value
        spin.editingFinished.emit()
        assert spin.value() == value
        assert spin.singleStep() == 1.0
    assert control.values() == control.slider.values() == (value, value)


@pytest.mark.parametrize("gesture", ["lower", "upper", "groove", "interval"])
def test_photon_energy_slider_integer_mouse_gestures(qtbot, gesture):
    control = NumericRangeControl(
        bounds=(5, 1000),
        values=(21.2, 80.7),
        decimals=3,
        logarithmic=True,
        snap_to_integer=True,
    )
    qtbot.addWidget(control)
    control.resize(700, 70)
    control.show()
    slider = control.slider
    lower, upper = (slider._handle_rect(i).center() for i in (0, 1))
    if gesture == "interval":
        start = (lower + upper) / 2
    elif gesture == "groove":
        start = upper + QtCore.QPoint(25, 0)
    else:
        start = lower if gesture == "lower" else upper
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    end = start + QtCore.QPoint(23, 0)
    _move_with_button(slider, end)
    updated = control.values()
    assert updated != (21.2, 80.7)
    if gesture == "interval":
        assert all(value.is_integer() for value in updated)
    else:
        index = 0 if gesture == "lower" else 1
        assert updated[index].is_integer()
        assert updated[1 - index] == (21.2, 80.7)[1 - index]
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Escape)
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert control.values() == slider.values() == (21.2, 80.7)


def test_photon_energy_slider_integer_keys_and_fractional_limits(qtbot):
    control = NumericRangeControl(
        bounds=(5, 1000),
        values=(21.2, 80.7),
        decimals=3,
        logarithmic=True,
        snap_to_integer=True,
    )
    qtbot.addWidget(control)
    control.show()
    slider = control.slider
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Right)
    assert control.values() == (22, 80.7)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_End)
    assert control.values() == (80, 80.7)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Home)
    assert control.values() == (5, 80.7)
    control.setValues((21.2, 80.7))
    qtbot.mouseClick(
        slider, QtCore.Qt.MouseButton.LeftButton, pos=slider._handle_rect(1).center()
    )
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Home)
    assert control.values() == (21.2, 22)


@pytest.fixture
def draggable_curve(qtbot):
    import pyqtgraph as pg

    from erlab.interactive.angle_calculator._widgets import DraggableCurveItem

    widget = pg.GraphicsLayoutWidget()
    qtbot.addWidget(widget)
    widget.resize(600, 400)
    plot = widget.addPlot()
    curve = DraggableCurveItem([0, 1, 2], [0, 0, 0])
    plot.addItem(curve)
    plot.setRange(xRange=(-1, 3), yRange=(-1, 1), padding=0)
    widget.show()
    qtbot.waitExposed(widget)
    return widget, plot, curve


class _CurveDragEvent:
    def __init__(self, position, *, start=False, finish=False, down=(1, 0), left=True):
        self.position = QtCore.QPointF(*position)
        self.down = QtCore.QPointF(*down)
        self.start = start
        self.finish = finish
        self.left = left
        self.accepted = False

    def pos(self):
        return self.position

    def buttonDownPos(self):
        return self.down

    def button(self):
        return (
            QtCore.Qt.MouseButton.LeftButton
            if self.left
            else QtCore.Qt.MouseButton.RightButton
        )

    def isStart(self):
        return self.start

    def isFinish(self):
        return self.finish

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


class _CurveHoverEvent:
    def __init__(self, position=(1, 0), *, exiting=False, available=True):
        self.position = QtCore.QPointF(*position)
        self.exit = exiting
        self.available = available
        self.requested = []

    def pos(self):
        return self.position

    def isExit(self):
        return self.exit

    def acceptDrags(self, button):
        self.requested.append(button)
        return self.available


def test_draggable_curve_hover_and_drag(draggable_curve):
    _, _, curve = draggable_curve
    starts, positions, finishes = [], [], []
    curve.sigDragStarted.connect(starts.append)
    curve.sigDragged.connect(positions.append)
    curve.sigDragFinished.connect(lambda: finishes.append(True))
    hover = _CurveHoverEvent()
    curve.hoverEvent(hover)
    assert hover.requested == [QtCore.Qt.MouseButton.LeftButton]
    assert curve.opts["pen"].widthF() == 4
    for event in (
        _CurveDragEvent((1, 0), left=False, start=True),
        _CurveDragEvent((1, 0), down=(1, 0.5), start=True),
        _CurveDragEvent((1, 0)),
    ):
        curve.mouseDragEvent(event)
        assert not event.accepted
    assert not starts
    start = _CurveDragEvent((1.1, 0.2), start=True)
    curve.mouseDragEvent(start)
    assert start.accepted
    assert starts == [QtCore.QPointF(1, 0)]
    assert positions == [QtCore.QPointF(1.1, 0.2)]
    curve.hoverEvent(_CurveHoverEvent(exiting=True))
    assert curve.opts["pen"].widthF() == 4
    move = _CurveDragEvent((3, 2))
    curve.mouseDragEvent(move)
    assert move.accepted
    end = _CurveDragEvent((3, 3), finish=True)
    curve.mouseDragEvent(end)
    assert end.accepted
    assert finishes == [True]
    assert positions[-2:] == [QtCore.QPointF(3, 2), QtCore.QPointF(3, 3)]
    assert curve.pos() == QtCore.QPointF()
    assert curve.opts["pen"].widthF() == 3
    curve.hoverEvent(_CurveHoverEvent(available=False))
    assert curve.opts["pen"].widthF() == 3
    outside = _CurveHoverEvent((1, 0.5))
    curve.hoverEvent(outside)
    assert not outside.requested
    curve.hoverEvent(_CurveHoverEvent())
    curve.hoverEvent(_CurveHoverEvent(exiting=True))
    curve.mouseDragEvent(_CurveDragEvent((1, 0), start=True))
    curve.mouseDragEvent(_CurveDragEvent((1, 0), finish=True))
    assert curve.opts["pen"].widthF() == 4


def test_draggable_curve_finite_gaps_and_zoom(draggable_curve):
    import numpy as np

    _, plot, curve = draggable_curve
    curve.setData([0, 0.5, 1, 1.5, 2], [0, 0, np.nan, 0, 0])
    assert curve.mouseShape().contains(QtCore.QPointF(0.25, 0))
    assert not curve.mouseShape().contains(QtCore.QPointF(1, 0))
    gap = _CurveDragEvent((1, 0), down=(1, 0), start=True)
    curve.mouseDragEvent(gap)
    assert not gap.accepted
    initial_shape = curve.mouseShape()
    plot.setYRange(-10, 10, padding=0)
    QtWidgets.QApplication.processEvents()
    assert (
        curve.mouseShape().boundingRect().height()
        > initial_shape.boundingRect().height()
    )


def test_draggable_curve_real_scene_drag_and_pan(qtbot, draggable_curve):
    widget, plot, curve = draggable_curve
    starts, positions, finishes = [], [], []
    curve.sigDragStarted.connect(starts.append)
    curve.sigDragged.connect(positions.append)
    curve.sigDragFinished.connect(lambda: finishes.append(True))
    viewport = widget.viewport()
    start = widget.mapFromScene(curve.mapToScene(QtCore.QPointF(1, 0)))
    end = start + QtCore.QPoint(40, -40)
    qtbot.mouseMove(viewport, start)
    qtbot.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    qtbot.wait(30)
    _move_with_button(viewport, end)
    qtbot.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert len(starts) == len(finishes) == 1
    assert len(positions) >= 2
    assert starts[0].x() == pytest.approx(1, abs=0.02)
    assert positions[-1].x() > starts[0].x()
    assert positions[-1].y() > starts[0].y()
    assert curve.pos() == QtCore.QPointF()
    before = plot.viewRange()
    outside = widget.mapFromScene(curve.mapToScene(QtCore.QPointF(1, 0.6)))
    qtbot.mouseMove(viewport, outside)
    qtbot.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=outside)
    qtbot.wait(30)
    _move_with_button(viewport, outside + QtCore.QPoint(40, 0))
    qtbot.mouseRelease(
        viewport, QtCore.Qt.MouseButton.LeftButton, pos=outside + QtCore.QPoint(40, 0)
    )
    assert len(starts) == 1
    assert plot.viewRange() != before


@pytest.mark.parametrize("values", [[], [float("nan")] * 3])
def test_draggable_curve_empty_path(draggable_curve, values):
    import numpy as np

    _, _, curve = draggable_curve
    curve.setData(np.arange(len(values)), np.array(values))
    assert curve.mouseShape().isEmpty()
    hover = _CurveHoverEvent()
    curve.hoverEvent(hover)
    assert not hover.requested
    drag = _CurveDragEvent((1, 0), start=True)
    curve.mouseDragEvent(drag)
    assert not drag.accepted


def test_draggable_curve_cancel_and_restart(draggable_curve):
    _, _, curve = draggable_curve
    starts, positions, finishes = [], [], []
    curve.sigDragStarted.connect(starts.append)
    curve.sigDragged.connect(positions.append)
    curve.sigDragFinished.connect(lambda: finishes.append(True))
    original_cursor = curve.cursor().shape()
    curve.hoverEvent(_CurveHoverEvent())
    curve.mouseDragEvent(_CurveDragEvent((1.1, 0), start=True))
    assert curve.opts["pen"].widthF() == 4
    curve.cancelDrag()
    assert curve.opts["pen"].widthF() == 3
    assert curve.cursor().shape() == original_cursor
    for event in (
        _CurveDragEvent((1.2, 0)),
        _CurveDragEvent((1.3, 0), finish=True),
    ):
        curve.mouseDragEvent(event)
        assert not event.accepted
    assert starts == [QtCore.QPointF(1, 0)]
    assert positions == [QtCore.QPointF(1.1, 0)]
    assert not finishes
    event = _CurveDragEvent((1.4, 0), start=True)
    curve.mouseDragEvent(event)
    assert event.accepted
    assert len(starts) == 2
    assert positions[-1] == QtCore.QPointF(1.4, 0)
    curve.mouseDragEvent(_CurveDragEvent((1.5, 0), finish=True))
    assert finishes == [True]
    assert curve.opts["pen"].widthF() == 4
    curve.cancelDrag()
    assert curve.opts["pen"].widthF() == 4
    assert finishes == [True]


@pytest.mark.parametrize("target", ["handle", "interval", "groove"])
def test_native_range_gesture_signal_order(qtbot, target):
    control = NumericRangeControl()
    qtbot.addWidget(control)
    control.resize(500, 70)
    control.show()
    slider = control.slider
    lower = slider._handle_rect(0).center()
    upper = slider._handle_rect(1).center()
    start = {
        "handle": lower,
        "interval": (lower + upper) / 2,
        "groove": lower - QtCore.QPoint(30, 0),
    }[target]
    events = []
    control.sigDragStarted.connect(lambda: events.append("start"))
    control.sigValueChanged.connect(lambda _: events.append("change"))
    control.sigDragFinished.connect(lambda: events.append("finish"))
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    end = start + QtCore.QPoint(-20, 0)
    _move_with_button(slider, end)
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert events[0] == "start"
    assert events[-1] == "finish"
    assert set(events[1:-1]) == {"change"}


@pytest.mark.parametrize("interrupt", ["focus", "hide", "ungrab", "disable", "cancel"])
def test_native_range_interrupted_gesture(qtbot, interrupt):
    control = NumericRangeControl()
    qtbot.addWidget(control)
    control.resize(500, 70)
    control.show()
    slider = control.slider
    finishes = []
    control.sigDragFinished.connect(lambda: finishes.append(True))
    start = slider._handle_rect(0).center()
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    _move_with_button(slider, start + QtCore.QPoint(-20, 0))
    if interrupt == "cancel":
        slider.cancelDrag()
    elif interrupt == "disable":
        slider.setEnabled(False)
    else:
        event_type = {
            "focus": QtCore.QEvent.Type.FocusOut,
            "hide": QtCore.QEvent.Type.Hide,
            "ungrab": QtCore.QEvent.Type.UngrabMouse,
        }[interrupt]
        QtWidgets.QApplication.sendEvent(slider, QtCore.QEvent(event_type))
    values = control.values()
    _move_with_button(slider, start + QtCore.QPoint(-40, 0))
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    assert control.values() == values
    assert finishes == ([] if interrupt == "cancel" else [True])
