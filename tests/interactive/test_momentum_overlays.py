import json
import sys

import numpy as np
import pydantic
import pyqtgraph as pg
import pytest
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive._momentum_overlays import _geometry
from erlab.interactive._momentum_overlays._editor import (
    _POCKET_MIME_TYPE,
    MomentumOverlayEditor,
)
from erlab.interactive._momentum_overlays._state import (
    ImageState,
    MomentumOverlayState,
    PocketState,
)
from erlab.interactive._momentum_overlays._widgets import ImageROI, PocketROI
from erlab.interactive.utils import qt_is_valid


@pytest.fixture
def pocket_clipboard(qapp):
    return qapp.clipboard()


@pytest.fixture
def overlays(qtbot):
    view = pg.GraphicsLayoutWidget()
    qtbot.addWidget(view)
    primary, projected = view.addPlot(row=0, col=0), view.addPlot(row=0, col=1)
    for plot in (primary, projected):
        plot.setRange(xRange=(-0.2, 0.2), yRange=(-0.2, 0.2), padding=0)
    editor = MomentumOverlayEditor(primary, projected_plot=projected)
    qtbot.addWidget(editor)
    editor.set_projection(lambda x, y: (x, y), lambda x, y: (x, y), token=0)
    yield editor
    editor.set_active(False)


@pytest.fixture
def image_files(tmp_path):
    paths = []
    for i, size in enumerate(((12, 8), (6, 12))):
        image = QtGui.QImage(*size, QtGui.QImage.Format.Format_RGBA8888)
        image.fill(QtGui.QColor(240, 40 + i * 80, 60, 255))
        path = tmp_path / f"image{i}.png"
        assert image.save(str(path))
        paths.append(path)
    return paths


@pytest.mark.parametrize("rotation", [0, 37, -110, 180])
def test_pocket_geometry(qtbot, rotation):
    roi = PocketROI((2, -3), (0.4, 0.2), rotation)
    center, radii, angle = roi.geometry()
    assert center == pytest.approx((2, -3))
    assert radii == pytest.approx((0.4, 0.2))
    assert angle == rotation
    roi.setGeometry((-4, 7), (0.1, 0.5), rotation + 30)
    assert roi.geometry()[0] == pytest.approx((-4, 7))
    assert roi.geometry()[1] == pytest.approx((0.1, 0.5))
    assert roi.geometry()[2] == rotation + 30
    roi.setGeometry((2, 3), (0, -2), 0)
    assert roi.geometry()[1] == pytest.approx((1e-6, 1e-6))


def test_pocket_handles(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    roi = PocketROI((2, -3), (1, 1), 25)
    plot.addItem(roi)
    scale = next(handle for handle in roi.handles if handle["type"] == "s")
    target = roi.mapToParent(pg.Point(3, 2.5))
    with qtbot.waitSignal(roi.sigRegionChanged):
        roi.movePoint(scale["item"], target)
    assert roi.geometry()[0] == pytest.approx((2, -3))
    assert roi.geometry()[1][0] != pytest.approx(roi.geometry()[1][1])
    rotate = next(handle for handle in roi.handles if handle["type"] == "r")
    with qtbot.waitSignal(roi.sigRegionChangeFinished):
        roi.movePoint(rotate["item"], pg.Point(2, -1))
    assert roi.geometry()[0] == pytest.approx((2, -3))
    assert roi.angle() == pytest.approx(90)


def test_pocket_selection_and_menu(qtbot):
    roi = PocketROI((0, 0), (1, 1))
    with qtbot.waitSignal(roi.sigSelected) as selected:
        roi.sigClicked.emit(roi, None)
    assert selected.args == [roi]
    menu = roi.getMenu()
    assert menu is roi.getMenu()
    action = next(a for a in menu.actions() if a.data() == "remove_pocket")
    assert action.objectName() == "remove_pocket"
    with qtbot.assertNotEmitted(roi.sigRemoveRequested):
        menu.close()
    with qtbot.waitSignal(roi.sigRemoveRequested) as removed:
        action.trigger()
    assert removed.args == [roi]
    roi.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    roi._emitRemoveRequest()


@pytest.mark.parametrize("rotation", [0, 37, -110, 180])
def test_image_geometry_and_handle(qtbot, rotation):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    roi = ImageROI((3, -4), 2, 0.4, rotation=rotation)
    plot.addItem(roi)
    assert roi.geometry()[0] == pytest.approx((3, -4))
    assert roi.geometry()[1] == 2
    assert roi.rotatable
    assert roi.geometry()[2] == rotation
    assert len(roi.handles) == 2
    target = roi.mapToParent(pg.Point(4, 1))
    with qtbot.waitSignal(roi.sigRegionChangeFinished):
        roi.movePoint(roi.handles[0]["item"], target)
    assert roi.geometry()[0] == pytest.approx((3, -4))
    assert roi.geometry()[1] > 2
    assert roi.size().y() / roi.size().x() == pytest.approx(0.4)
    roi.translate((1, 2))
    assert roi.geometry()[0] == pytest.approx((4, -2))
    roi.setGeometry((-5, 8), 3, aspect=2)
    assert roi.geometry()[0] == pytest.approx((-5, 8))
    assert tuple(roi.size()) == pytest.approx((3, 6))
    assert roi.geometry()[2] == rotation
    rotate = next(handle for handle in roi.handles if handle["type"] == "r")
    with qtbot.waitSignal(roi.sigRegionChangeFinished):
        roi.movePoint(rotate["item"], pg.Point(-5, 11))
    assert roi.geometry()[0] == pytest.approx((-5, 8))
    assert roi.geometry()[2] % 360 == pytest.approx(90)
    assert tuple(roi.size()) == pytest.approx((3, 6))
    roi.setGeometry((2, -3), 4, rotation=-25)
    assert roi.geometry()[0] == pytest.approx((2, -3))
    assert roi.geometry()[2] == -25
    assert tuple(roi.size()) == pytest.approx((4, 8))
    roi.setGeometry((0, 0), 0)
    assert roi.geometry()[1] == 1e-6


def test_overlays_pockets(overlays, qtbot):
    roi = overlays.add_pocket()
    pocket = overlays.state.pockets[0]
    for name, value in (
        ("pocket_x", 0.1),
        ("pocket_y", 0.03),
        ("pocket_rx", 0.02),
        ("pocket_ry", 0.01),
        ("pocket_rotation", 30),
    ):
        overlays.inputs[name].setValue(value)
    assert roi.geometry()[0] == pytest.approx((0.1, 0.03))
    assert roi.geometry()[1] == pytest.approx((0.02, 0.01))
    assert roi.angle() == 30
    roi.translate((0.01, -0.02))
    assert pocket.center == pytest.approx((0.11, 0.01))
    assert overlays.inputs["pocket_x"].value() == pytest.approx(0.11)
    overlays.group_combo.setCurrentText("4")
    overlays.redraw()

    def visible_points():
        points = np.vstack(
            [
                np.column_stack(curves[0].getData())
                for curves in overlays._contours
                if curves[0].isVisible()
            ]
        )
        return points[np.isfinite(points).all(axis=1)]

    expected = np.vstack(
        _geometry.expand_pocket(pocket.center, pocket.radii, pocket.rotation, "4")
    )
    np.testing.assert_allclose(visible_points(), expected)
    overlays.pocket_symmetry.setChecked(False)
    overlays.redraw()
    np.testing.assert_allclose(
        visible_points(),
        _geometry.ellipse_points(pocket.center, pocket.radii, pocket.rotation),
    )
    overlays.pocket_visible.setChecked(False)
    overlays.redraw()
    assert not roi.isVisible()
    overlays.pocket_visible.setChecked(True)
    other = overlays.add_pocket()
    roi.sigSelected.emit(roi)
    assert overlays._selected_pocket == pocket.uid
    with qtbot.waitSignal(roi.sigRemoveRequested):
        roi.getMenu().remAct.trigger()
    assert len(overlays.state.pockets) == 1
    assert next(iter(overlays._pocket_rois.values())) is other
    overlays.delete_button.click()
    assert not overlays.state.pockets
    assert not overlays.inputs["pocket_x"].isEnabled()


@pytest.mark.parametrize("source", ["drag", "restore"])
@pytest.mark.parametrize(
    ("control", "field", "value"),
    [
        ("pocket_x", "center", 0.2),
        ("pocket_y", "center", -0.2),
        ("pocket_rx", "radii", 0.03),
        ("pocket_ry", "radii", 0.04),
        ("pocket_rotation", "rotation", 24),
        ("pocket_visible", "visible", False),
        ("pocket_symmetry", "symmetry", False),
        ("group_combo", "point_group", "4"),
        ("symmetry_angle", "symmetry_angle", 15),
    ],
)
def test_pocket_edit_preserves_other_fields(
    overlays, qtbot, source, control, field, value
):
    roi = overlays.add_pocket()
    center = (0.012345678, -0.023456789)
    radii = (0.034567891, 0.045678912)
    rotation = 12.345678
    if source == "drag":
        roi.setGeometry(center, radii, rotation)
    else:
        restored = overlays.state.model_copy(deep=True)
        restored.pockets[0].center = center
        restored.pockets[0].radii = radii
        restored.pockets[0].rotation = rotation
        overlays.restore_state(restored)
        roi = overlays._pocket_rois[restored.pockets[0].uid]
    before = overlays.state.model_copy(deep=True)
    expected = before.model_copy(deep=True)
    pocket = expected.pockets[0]
    if field in ("center", "radii"):
        previous = getattr(pocket, field)
        index = 0 if control in ("pocket_x", "pocket_rx") else 1
        setattr(
            pocket, field, (value, previous[1]) if index == 0 else (previous[0], value)
        )
    else:
        setattr(pocket, field, value)
    with qtbot.waitSignal(overlays.sigStateChanged):
        if control in overlays.inputs:
            overlays.inputs[control].setValue(value)
        elif control == "group_combo":
            overlays.group_combo.setCurrentText(value)
        else:
            getattr(overlays, control).setChecked(value)
    assert overlays.state == expected
    assert roi.geometry()[0] == pytest.approx(pocket.center, abs=1e-14)
    assert roi.geometry()[1] == pytest.approx(pocket.radii, abs=1e-14)
    assert roi.geometry()[2] == pocket.rotation


@pytest.mark.parametrize("source", ["drag", "restore"])
@pytest.mark.parametrize(
    ("control", "field", "value"),
    [
        ("image_x", "center", 0.2),
        ("image_y", "center", -0.2),
        ("image_width", "width", 0.812345678),
        ("image_rotation", "rotation", 24),
        ("image_opacity", "opacity", 0.3),
        ("image_visible", "visible", False),
        ("image_invert", "invert", True),
        ("image_black_transparent", "black_transparent", False),
        ("image_black_threshold", "black_threshold", 20),
    ],
)
def test_image_edit_preserves_other_fields(
    overlays, image_files, qtbot, source, control, field, value
):
    overlays.load_image(str(image_files[0]))
    overlays.image_black_transparent.setChecked(True)
    center = (0.012345678, -0.023456789)
    width = 0.345678912
    rotation = 12.345678
    if source == "drag":
        overlays._image_roi.setGeometry(center, width, rotation=rotation)
    else:
        restored = overlays.state.model_copy(deep=True)
        restored.image.center = center
        restored.image.width = width
        restored.image.rotation = rotation
        restored.image.opacity = 0.654321
        overlays.restore_state(restored)
    expected = overlays.state.model_copy(deep=True)
    image = expected.image
    if field == "center":
        image.center = (
            (value, image.center[1])
            if control == "image_x"
            else (image.center[0], value)
        )
    else:
        setattr(image, field, value)
    with qtbot.waitSignal(overlays.sigStateChanged):
        if control in overlays.inputs:
            overlays.inputs[control].setValue(value)
        else:
            getattr(overlays, control).setChecked(value)
    assert overlays.state == expected
    assert overlays._image_roi.geometry()[0] == pytest.approx(image.center, abs=1e-14)
    assert overlays._image_roi.geometry()[1] == image.width
    assert overlays._image_roi.geometry()[2] == image.rotation


def test_overlays_image_dialogs(overlays, image_files, accept_dialog):
    accept_dialog(
        overlays.image_button.click, accept_call=lambda dialog: dialog.reject()
    )
    assert overlays.state.image is None

    def select_image(dialog, path):
        dialog.setDirectory(str(path.parent))
        dialog.selectFile(str(path))
        dialog.findChild(QtWidgets.QLineEdit, "fileNameEdit").setText(path.name)

    accept_dialog(
        overlays.image_button.click,
        pre_call=lambda dialog: select_image(dialog, image_files[0]),
        accept_call=lambda dialog: (
            dialog.findChild(QtWidgets.QDialogButtonBox)
            .button(QtWidgets.QDialogButtonBox.StandardButton.Open)
            .click()
        ),
    )
    assert overlays._source_image.shape == (8, 12, 4)
    overlays.inputs["image_x"].setValue(0.04)
    overlays.inputs["image_width"].setValue(0.4)
    overlays.inputs["image_opacity"].setValue(0.3)
    overlays.inputs["image_rotation"].setValue(29)
    overlays.image_locked.setChecked(True)
    overlays.image_invert.setChecked(True)
    overlays.image_black_transparent.setChecked(True)
    overlays.inputs["image_black_threshold"].setValue(20)
    previous = overlays.state.image.model_copy()
    accept_dialog(
        overlays.image_button.click, accept_call=lambda dialog: dialog.reject()
    )
    assert overlays.state.image == previous
    accept_dialog(
        overlays.image_button.click,
        pre_call=lambda dialog: select_image(dialog, image_files[1]),
        accept_call=lambda dialog: (
            dialog.findChild(QtWidgets.QDialogButtonBox)
            .button(QtWidgets.QDialogButtonBox.StandardButton.Open)
            .click()
        ),
    )
    assert overlays._source_image.shape == (12, 6, 4)
    assert overlays.state.image.center == previous.center
    assert overlays.state.image.width == previous.width
    assert overlays.state.image.opacity == previous.opacity
    assert overlays.state.image.invert is True
    assert overlays.state.image.black_transparent is True
    assert overlays.state.image.black_threshold == 20
    assert overlays.state.image.rotation == previous.rotation
    assert overlays.state.image.locked
    assert not overlays._image_roi.isVisible()
    assert overlays._image_roi.geometry()[2] == pytest.approx(previous.rotation)
    assert overlays._image_roi.size().y() / overlays._image_roi.size().x() == 2


@pytest.mark.parametrize(
    ("width", "display"),
    [(1e-5, "0.00001"), (0.3, "0.3"), (0.3456789, "0.3457"), (1000.0, "1000")],
)
def test_image_width_significant_display(overlays, image_files, qtbot, width, display):
    overlays.load_image(str(image_files[0]))
    spin = overlays.inputs["image_width"]
    spin.setValue(width)
    assert spin.valueFromText(spin.text()) == float(display)
    assert spin.value() == width
    assert overlays.state.image.width == width
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        spin.interpretText()
    assert spin.value() == width
    overlays.inputs["image_opacity"].setValue(0.5)
    assert overlays.state.image.width == width
    saved = json.loads(overlays.state.model_dump_json())
    spin.setValue(0.1)
    overlays.restore_state(saved)
    assert spin.value() == width
    assert overlays.state.image.width == width


def test_image_lock_controls_and_restore(overlays, image_files, qtbot):
    assert not overlays.image_locked.isEnabled()
    assert not overlays.image_locked.isChecked()
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        overlays.image_locked.setChecked(True)
        overlays.image_locked.setChecked(False)
    overlays.load_image(str(image_files[0]))
    assert overlays.image_locked.isEnabled()
    assert not overlays.state.image.locked
    # A lock toggle must preserve the full precision of geometry set by dragging.
    overlays._image_roi.translate((0.012345678, -0.023456789))
    overlays._draw_image(32)
    before = overlays.state.model_copy(deep=True)
    pixels = overlays.momentum_image.image.copy()
    projected = overlays.projected_image.image.copy()
    with qtbot.waitSignal(overlays.sigStateChanged):
        overlays.image_locked.click()
    expected = before.model_copy(deep=True)
    expected.image.locked = True
    assert overlays.state == expected
    assert not overlays._image_roi.isVisible()
    assert all(not handle["item"].isVisible() for handle in overlays._image_roi.handles)
    overlays._draw_image(32)
    assert overlays.momentum_image.isVisible()
    assert overlays.projected_image.isVisible()
    np.testing.assert_array_equal(overlays.momentum_image.image, pixels)
    np.testing.assert_array_equal(overlays.projected_image.image, projected)

    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        overlays.restore_state(json.loads(expected.model_dump_json()))
    assert overlays.image_locked.isChecked()
    assert not overlays._image_roi.isVisible()
    for name, value in (
        ("image_x", 0.05),
        ("image_y", -0.02),
        ("image_width", 0.4),
        ("image_rotation", 24),
    ):
        assert overlays.inputs[name].isEnabled()
        overlays.inputs[name].setValue(value)
    assert overlays._image_roi.geometry()[0] == pytest.approx((0.05, -0.02))
    assert overlays._image_roi.geometry()[1] == pytest.approx(0.4)
    assert overlays._image_roi.geometry()[2] == pytest.approx(24)
    assert overlays.state.image.locked
    assert not overlays._image_roi.isVisible()
    overlays.set_available(False)
    overlays.set_available(True)
    assert not overlays._image_roi.isVisible()
    overlays.image_visible.setChecked(False)
    overlays.image_locked.click()
    assert not overlays._image_roi.isVisible()
    overlays.image_visible.setChecked(True)
    overlays.redraw()
    assert overlays._image_roi.isVisible()
    assert all(handle["item"].isVisible() for handle in overlays._image_roi.handles)

    legacy = expected.model_dump()
    del legacy["image"]["locked"]
    overlays.restore_state(legacy)
    assert not overlays.state.image.locked
    assert not overlays.image_locked.isChecked()
    assert overlays._image_roi.isVisible()
    overlays.image_locked.click()
    overlays.restore_state(MomentumOverlayState())
    assert not overlays.image_locked.isEnabled()
    assert not overlays.image_locked.isChecked()
    assert overlays._image_roi is None


@pytest.mark.parametrize("gesture", ["drag", "scale", "rotate"])
def test_image_lock_blocks_mouse_gestures(
    overlays, image_files, qtbot, monkeypatch, gesture
):
    # Deliver each synthetic move instead of dropping rapid events between frames.
    monkeypatch.setitem(pg.CONFIG_OPTIONS, "mouseRateLimit", 0)
    overlays.load_image(str(image_files[0]))
    plot = overlays.momentum_plot
    plot.setMouseEnabled(x=False, y=False)
    view = plot.scene().views()[0]
    view.resize(1000, 500)
    with qtbot.waitExposed(view):
        view.show()
    overlays.redraw()
    roi = overlays._image_roi

    def drag():
        if gesture == "drag":
            start = roi.mapToScene(roi.size() * 0.5)
        else:
            handle = next(
                handle
                for handle in roi.handles
                if handle["type"] == ("s" if gesture == "scale" else "r")
            )
            start = handle["item"].scenePos()
        start = view.mapFromScene(start)
        viewport = view.viewport()
        qtbot.mouseMove(viewport, pos=start)
        qtbot.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
        for delta in (QtCore.QPoint(15, -15), QtCore.QPoint(30, -30)):
            end = start + delta
            event = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(end),
                QtCore.QPointF(viewport.mapToGlobal(end)),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            QtWidgets.QApplication.sendEvent(viewport, event)
        qtbot.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)

    overlays.image_locked.click()
    before = overlays.state.model_copy(deep=True)
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        drag()
    assert overlays.state == before
    overlays.image_locked.click()
    with qtbot.waitSignal(overlays.sigStateChanged):
        drag()
    if gesture == "drag":
        assert overlays.state.image.center != before.image.center
    elif gesture == "scale":
        assert overlays.state.image.width != before.image.width
    else:
        assert overlays.state.image.rotation != before.image.rotation


def test_overlays_image_error_dialog(overlays, accept_dialog, monkeypatch, tmp_path):
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(invalid), "")
    )
    accept_dialog(overlays.image_button.click)
    assert overlays.state.image is None


def test_overlays_extreme_pocket_drag_state(overlays):
    roi = overlays.add_pocket()
    roi.translate((150, -150))
    assert roi.geometry()[0] == pytest.approx((100, -100))
    scale = next(handle for handle in roi.handles if handle["type"] == "s")
    roi.movePoint(scale["item"], roi.mapToParent(roi.size() * 10000))
    assert roi.geometry()[1] == pytest.approx((100, 100))
    center_local = roi.size() * 0.5
    roi.movePoint(
        scale["item"], roi.mapToParent(center_local + type(center_local)(1e-7, 1e-7))
    )
    assert roi.geometry()[1] == pytest.approx((1e-5, 1e-5))
    roi.setAngle(1085)
    assert roi.geometry()[2] == pytest.approx(5)
    saved = MomentumOverlayState.model_validate(overlays.state.model_dump(mode="json"))
    pocket = saved.pockets[0]
    assert pocket.center == pytest.approx(roi.geometry()[0])
    assert pocket.radii == pytest.approx(roi.geometry()[1])
    assert pocket.rotation == pytest.approx(roi.geometry()[2])


def test_overlays_extreme_image_drag_state(overlays, image_files):
    overlays.load_image(str(image_files[1]))
    roi = overlays._image_roi
    roi.translate((-150, 150))
    assert roi.geometry()[0] == pytest.approx((-100, 100))
    roi.movePoint(roi.handles[0]["item"], roi.mapToParent(roi.size() * 10000))
    assert roi.geometry()[1] == pytest.approx(1000)
    center_local = roi.size() * 0.5
    roi.movePoint(
        roi.handles[0]["item"], roi.mapToParent(center_local + roi.size() * 1e-10)
    )
    assert roi.geometry()[1] == pytest.approx(1e-5)
    saved = MomentumOverlayState.model_validate(overlays.state.model_dump(mode="json"))
    assert saved.image.center == pytest.approx(roi.geometry()[0])
    assert saved.image.width == pytest.approx(roi.geometry()[1])
    payload = saved.model_dump(mode="json")
    payload["image"]["center"] = [101, 0]
    with pytest.raises(pydantic.ValidationError):
        MomentumOverlayState.model_validate(payload)


def test_overlays_per_pocket_point_groups(overlays):
    window = overlays
    assert not window.group_combo.isEnabled()
    window.add_pocket()
    first = window.state.pockets[0]
    window.inputs["pocket_x"].setValue(0.08)
    window.group_combo.setCurrentText("m")
    window.symmetry_spin.setValue(15)
    window.add_pocket()
    second = window.state.pockets[1]
    assert second.point_group == first.point_group == "m"
    assert second.symmetry_angle == first.symmetry_angle == 15
    window.inputs["pocket_y"].setValue(0.06)
    window.group_combo.setCurrentText("3")
    window.symmetry_spin.setValue(27)
    assert first.point_group == "m"
    assert first.symmetry_angle == 15
    window._pocket_rois[first.uid].sigSelected.emit(window._pocket_rois[first.uid])
    assert window.group_combo.currentText() == "m"
    assert window.symmetry_spin.value() == 15
    window.group_combo.setCurrentText("4")
    window.redraw()
    for pocket, curves in zip((first, second), window._contours, strict=True):
        points = np.column_stack(curves[0].getData())
        expected = np.vstack(
            _geometry.expand_pocket(
                pocket.center,
                pocket.radii,
                pocket.rotation,
                pocket.point_group,
                pocket.symmetry_angle,
            )
        )
        np.testing.assert_allclose(points[np.isfinite(points).all(axis=1)], expected)
    payload = window.state.model_dump(mode="json")
    window.restore_state(payload)
    assert [p.model_dump() for p in window.state.pockets] == [
        first.model_dump(),
        second.model_dump(),
    ]
    assert window._selected_pocket == first.uid
    assert window.group_combo.currentText() == first.point_group
    window._pocket_rois[first.uid].sigSelected.emit(window._pocket_rois[first.uid])
    window.add_pocket()
    assert window.state.pockets[-1].point_group == "4"
    assert window.state.pockets[-1].symmetry_angle == 15


def test_overlays_pocket_group_after_delete(overlays):
    window = overlays
    window.add_pocket()
    window.group_combo.setCurrentText("6mm")
    window.delete_button.click()
    assert not window.group_combo.isEnabled()
    assert not window.symmetry_spin.isEnabled()
    window.add_pocket()
    assert window.state.pockets[0].point_group == "6mm"


def test_overlay_image_warp_cache(overlays, image_files):
    calls = []

    def mapping(x, y):
        calls.append(np.shape(x))
        return x, y

    overlays.set_projection(lambda x, y: (x, y), mapping, token=1)
    overlays.load_image(str(image_files[0]))
    overlays._draw_image(32)
    assert calls == [(32, 32)]
    assert overlays.projected_image.image.shape == (32, 32, 4)
    assert np.any(overlays.projected_image.image[..., 3])
    overlays._image_roi.translate((0.01, 0.02))
    assert overlays.state.image.center == pytest.approx((0.01, 0.02))
    overlays._draw_image(32)
    assert calls == [(32, 32)]
    overlays._draw_image(48)
    overlays._draw_image(32)
    assert calls == [(32, 32), (48, 48)]
    overlays.set_projection(lambda x, y: (x, y), mapping, token=2)
    overlays._draw_image(32)
    assert calls == [(32, 32), (48, 48), (32, 32)]
    overlays._refine_image()
    assert overlays.projected_image.image.shape == (480, 480, 4)
    overlays.image_visible.setChecked(False)
    overlays._draw_image(32)
    assert not overlays.projected_image.isVisible()
    assert not overlays.momentum_image.isVisible()
    assert not overlays._image_roi.isVisible()
    with pytest.raises(ValueError, match="not a readable image"):
        overlays.load_image(str(image_files[0].with_suffix(".missing")))


def test_overlay_restore_is_atomic(overlays, image_files):
    roi = overlays.add_pocket()
    overlays.load_image(str(image_files[0]))
    state = overlays.state.model_copy(deep=True)
    source = overlays._source_image.copy()
    with pytest.raises(ValueError, match="cannot be decoded"):
        overlays.restore_state(
            MomentumOverlayState(image=ImageState(png_base64="iVBORw0KGgo="))
        )
    assert overlays.state == state
    assert overlays._pocket_rois[state.pockets[0].uid] is roi
    np.testing.assert_array_equal(overlays._source_image, source)


def test_overlay_projection_breaks_discontinuities(overlays):
    def mapping(x, y):
        return np.where(np.asarray(x) >= 0, 179.0, -179.0), y

    overlays.set_projection(mapping, None, token=1, max_step=(90.0, 90.0))
    overlays.add_pocket()
    overlays.redraw()
    x, _ = overlays._contours[0][1].getData()
    assert np.isnan(x).sum() > 1
    assert np.all(np.abs(np.diff(x)[np.isfinite(np.diff(x))]) <= 90)


def test_overlay_availability_preserves_state(overlays, image_files):
    roi = overlays.add_pocket()
    overlays.load_image(str(image_files[0]))
    state = overlays.state.model_copy(deep=True)
    overlays.set_available(False)
    assert not overlays.isEnabled()
    assert not roi.isVisible()
    assert not overlays.momentum_image.isVisible()
    assert not overlays.projected_image.isVisible()
    overlays.set_available(True)
    assert overlays.isEnabled()
    assert roi.isVisible()
    assert overlays.momentum_image.isVisible()
    assert overlays.projected_image.isVisible()
    assert overlays.state == state


def test_explicit_pocket_state(overlays):
    state = PocketState(center=(0.1, 0.2), radii=(0.3, 0.4))
    roi = overlays.add_pocket(state=state)
    assert overlays.state.pockets == [state]
    assert roi.geometry()[0] == pytest.approx(state.center)
    with pytest.raises(ValueError, match="unique"):
        overlays.add_pocket(state=state)
    assert overlays.state.pockets == [state]


@pytest.mark.parametrize("add_pocket", [False, True])
def test_overlay_items_do_not_change_data_auto_range(overlays, image_files, add_pocket):
    plots = (overlays.momentum_plot, overlays.projected_plot)
    for plot in plots:
        plot.plot([-1, 1], [-1, 1])
        plot.enableAutoRange()
        plot.getViewBox().updateAutoRange()
    expected = [np.asarray(plot.viewRange()) for plot in plots]
    overlays.load_image(str(image_files[0]))
    if add_pocket:
        overlays.add_pocket(state=PocketState(center=(5, 5)))
    for _ in range(5):
        overlays.redraw()
        for plot, limits in zip(plots, expected, strict=True):
            plot.getViewBox().updateAutoRange()
            np.testing.assert_allclose(plot.viewRange(), limits)


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("transparent", [False, True])
@pytest.mark.parametrize("threshold", [0, 19, 20, 255])
def test_image_color_processing_in_both_views_and_saved_state(
    overlays, tmp_path, invert, transparent, threshold
):
    assert not overlays.image_invert.isEnabled()
    assert not overlays.image_black_transparent.isEnabled()
    assert not overlays.inputs["image_black_threshold"].isEnabled()
    assert not overlays.layout().isRowVisible(overlays.inputs["image_black_threshold"])
    path = tmp_path / "colors.png"
    colors = np.array(
        [
            [[0, 0, 0, 255], [255, 255, 255, 128]],
            [[20, 10, 5, 160], [235, 245, 250, 200]],
        ],
        dtype=np.uint8,
    )
    image = QtGui.QImage(2, 2, QtGui.QImage.Format.Format_RGBA8888)
    for y in range(2):
        for x in range(2):
            image.setPixelColor(x, y, QtGui.QColor(*map(int, colors[y, x])))
    assert image.save(str(path))
    overlays.load_image(str(path))
    overlays.inputs["image_width"].setValue(0.4)
    original = overlays.state.image.png_base64
    assert overlays.image_invert.isEnabled()
    assert overlays.image_black_transparent.isEnabled()
    overlays.image_invert.setChecked(invert)
    overlays.image_black_transparent.setChecked(transparent)
    overlays.inputs["image_black_threshold"].setValue(threshold)
    assert overlays.inputs["image_black_threshold"].isEnabled() is transparent
    assert (
        overlays.layout().isRowVisible(overlays.inputs["image_black_threshold"])
        is transparent
    )
    expected = colors.copy()
    if invert:
        expected[..., :3] = [
            [[255, 255, 255], [0, 0, 0]],
            [[235, 245, 250], [20, 10, 5]],
        ]
    if transparent:
        expected[0, int(invert), 3] = 0
        if threshold >= 20:
            expected[1, int(invert), 3] = 0
        if threshold == 255:
            expected[..., 3] = 0
    overlays._draw_image(2)
    np.testing.assert_array_equal(overlays.momentum_image.image, np.flipud(expected))
    projected = np.flipud(overlays.projected_image.image)
    np.testing.assert_array_equal(projected[..., 3], expected[..., 3])
    visible = expected[..., 3] > 0
    np.testing.assert_array_equal(projected[visible], expected[visible])
    np.testing.assert_array_equal(overlays._source_image, colors)
    assert overlays.state.image.png_base64 == original
    saved = overlays.state.model_dump(mode="json")
    overlays.restore_state({})
    assert not overlays.image_invert.isEnabled()
    assert not overlays.image_black_transparent.isEnabled()
    assert not overlays.inputs["image_black_threshold"].isEnabled()
    assert not overlays.layout().isRowVisible(overlays.inputs["image_black_threshold"])
    overlays.restore_state(saved)
    assert overlays.image_invert.isChecked() is invert
    assert overlays.image_black_transparent.isChecked() is transparent
    assert overlays.inputs["image_black_threshold"].value() == threshold
    assert (
        overlays.layout().isRowVisible(overlays.inputs["image_black_threshold"])
        is transparent
    )
    np.testing.assert_array_equal(overlays.momentum_image.image, np.flipud(expected))
    overlays.image_invert.setChecked(False)
    overlays.image_black_transparent.setChecked(False)
    np.testing.assert_array_equal(overlays.momentum_image.image, np.flipud(colors))
    for name in ("invert", "black_transparent", "black_threshold"):
        del saved["image"][name]
    overlays.restore_state(saved)
    assert not overlays.image_invert.isChecked()
    assert not overlays.image_black_transparent.isChecked()
    assert overlays.inputs["image_black_threshold"].value() == 0
    np.testing.assert_array_equal(overlays.momentum_image.image, np.flipud(colors))


@pytest.mark.parametrize("threshold", [-1, 256, 0.5])
def test_image_transparency_threshold_validation(threshold):
    with pytest.raises(pydantic.ValidationError):
        ImageState(png_base64="iVBORw0KGgo=", black_threshold=threshold)


@pytest.mark.parametrize("reorder", [False, True])
def test_overlay_restore_preserves_selected_pocket(overlays, reorder):
    for _ in range(3):
        overlays.add_button.click()
    overlays.pocket_list.setCurrentRow(0)
    selected_uid = overlays._selected_pocket
    state = overlays.state.model_copy(deep=True)
    if reorder:
        state.pockets.reverse()
    overlays.restore_state(state)
    assert overlays._selected_pocket == selected_uid
    assert (
        overlays.pocket_list.currentItem().data(QtCore.Qt.ItemDataRole.UserRole)
        == selected_uid
    )
    overlays.inputs["pocket_x"].setValue(0.2)
    assert {pocket.uid: pocket.center[0] for pocket in overlays.state.pockets} == {
        pocket.uid: 0.2 if pocket.uid == selected_uid else 0.0
        for pocket in state.pockets
    }
    # Removing the old selection must leave a valid selection, then none when empty.
    state.pockets = [pocket for pocket in state.pockets if pocket.uid != selected_uid]
    overlays.restore_state(state)
    assert overlays._selected_pocket in {pocket.uid for pocket in state.pockets}
    overlays.restore_state(MomentumOverlayState())
    assert overlays._selected_pocket is None
    assert not overlays.inputs["pocket_x"].isEnabled()


@pytest.mark.parametrize("accept", [True, False])
@pytest.mark.parametrize("trigger", ["menu", "keyboard"])
def test_native_pocket_rename(overlays, qtbot, accept, trigger):
    overlays.add_pocket()
    overlays.show()
    overlays.activateWindow()
    pocket_list = overlays.pocket_list
    pocket_list.setFocus()
    qtbot.waitUntil(pocket_list.hasFocus)
    item = pocket_list.currentItem()
    before = overlays.state.model_copy(deep=True)
    assert item.flags() & QtCore.Qt.ItemFlag.ItemIsEditable
    if trigger == "menu":
        overlays.rename_pocket_action.trigger()
    else:
        qtbot.keyClick(
            pocket_list,
            QtCore.Qt.Key.Key_Return
            if sys.platform == "darwin"
            else QtCore.Qt.Key.Key_F2,
        )
    edit = pocket_list.findChild(QtWidgets.QLineEdit)
    assert edit is not None
    edit.setText("  Electron Γ  ")
    changes = []
    overlays.sigStateChanged.connect(lambda: changes.append(1))
    if accept:
        with qtbot.waitSignal(overlays.sigStateChanged):
            qtbot.keyClick(edit, QtCore.Qt.Key.Key_Return)
    else:
        qtbot.keyClick(edit, QtCore.Qt.Key.Key_Escape)
    expected = before.model_copy(deep=True)
    if accept:
        expected.pockets[0].name = "Electron Γ"
    assert overlays.state == expected
    assert item.text() == expected.pockets[0].name
    assert len(changes) == int(accept)
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        item.setText("   ")
        item.setText(expected.pockets[0].name)
    assert overlays.state == expected
    assert item.text() == expected.pockets[0].name


def test_pocket_names_persist_and_survive_deletion(overlays):
    overlays.restore_state({"pockets": [{}, {}]})
    first_name, second_name = [p.name for p in overlays.state.pockets]
    assert first_name
    assert second_name
    assert first_name != second_name
    overlays.pocket_list.item(1).setText("Hole pocket")
    overlays.pocket_list.setCurrentRow(0)
    overlays.delete_button.click()
    assert overlays.state.pockets[0].name == "Hole pocket"
    overlays.add_pocket()
    overlays.add_pocket()
    names = [p.name for p in overlays.state.pockets]
    assert len(set(names)) == 3
    saved = overlays.state.model_dump(mode="json")
    overlays.restore_state({})
    overlays.restore_state(saved)
    assert overlays.state.model_dump(mode="json") == saved
    assert [overlays.pocket_list.item(i).text() for i in range(3)] == names


def test_pocket_restore_keeps_remaining_selection(overlays, qtbot):
    for _ in range(3):
        overlays.add_pocket()
    overlays.pocket_list.setCurrentRow(1)
    overlays.pocket_list.item(0).setSelected(True)
    saved = overlays.state.model_copy(deep=True)
    saved.pockets.pop(1)
    overlays.restore_state(saved)
    assert overlays._selected_pocket == saved.pockets[0].uid
    assert overlays.pocket_list.currentItem() is overlays.pocket_list.item(0)
    assert overlays.pocket_list.selectedItems() == [overlays.pocket_list.item(0)]
    # Display updates during restoration must not become name edits.
    overlays._restoring = True
    try:
        with qtbot.assertNotEmitted(overlays.sigStateChanged):
            overlays.pocket_list.item(0).setText("Restoring display")
        assert overlays.state == saved
    finally:
        overlays._restoring = False
    overlays.restore_state(saved)
    assert overlays.pocket_list.item(0).text() == saved.pockets[0].name


def test_pocket_list_context_menu_and_multiple_selection(
    overlays, qtbot, pocket_clipboard
):
    overlays.show()
    pocket_list = overlays.pocket_list

    def open_menu(position):
        event = QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Reason.Mouse,
            position,
            pocket_list.viewport().mapToGlobal(position),
        )
        QtWidgets.QApplication.sendEvent(pocket_list.viewport(), event)
        qtbot.waitUntil(overlays.pocket_menu.isVisible)
        assert set(overlays.pocket_menu.actions()) == {
            overlays.rename_pocket_action,
            overlays.copy_pockets_action,
            overlays.paste_pockets_action,
        }

    open_menu(QtCore.QPoint(5, 5))
    assert not any(action.isEnabled() for action in overlays.pocket_menu.actions())
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        qtbot.keyClick(overlays.pocket_menu, QtCore.Qt.Key.Key_Escape)
        overlays._rename_pocket()
        overlays._copy_pockets()
        overlays._paste_pockets()
    for _ in range(3):
        overlays.add_pocket()
    pocket_list.setCurrentRow(0)
    pocket_list.item(1).setSelected(True)
    selected = [p.uid for p in overlays.state.pockets[:2]]
    open_menu(pocket_list.visualItemRect(pocket_list.item(1)).center())
    assert len(pocket_list.selectedItems()) == 2
    assert not overlays.rename_pocket_action.isEnabled()
    assert overlays.copy_pockets_action.isEnabled()
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        overlays._rename_pocket()
        qtbot.mouseClick(
            overlays.pocket_menu,
            QtCore.Qt.MouseButton.LeftButton,
            pos=overlays.pocket_menu.actionGeometry(
                overlays.copy_pockets_action
            ).center(),
        )
    payload = json.loads(pocket_clipboard.mimeData().data(_POCKET_MIME_TYPE).data())
    assert [p["uid"] for p in payload["pockets"]] == selected
    saved = overlays.state.model_copy(deep=True)
    overlays.restore_state(saved)
    assert {
        item.data(QtCore.Qt.ItemDataRole.UserRole)
        for item in pocket_list.selectedItems()
    } == set(selected)
    changes = []
    overlays.sigStateChanged.connect(lambda: changes.append(1))
    overlays.delete_button.click()
    assert len(changes) == 1
    assert overlays.state.pockets == saved.pockets[2:]
    overlays.restore_state({})
    open_menu(QtCore.QPoint(5, 5))
    assert overlays.paste_pockets_action.isEnabled()
    qtbot.mouseClick(
        overlays.pocket_menu,
        QtCore.Qt.MouseButton.LeftButton,
        pos=overlays.pocket_menu.actionGeometry(overlays.paste_pockets_action).center(),
    )
    assert len(overlays.state.pockets) == 2
    assert len(pocket_list.selectedItems()) == 2
    assert len(changes) == 2
    pocket_list.clearSelection()
    open_menu(pocket_list.visualItemRect(pocket_list.item(0)).center())
    assert pocket_list.selectedItems() == [pocket_list.item(0)]
    assert overlays.rename_pocket_action.isEnabled()
    overlays.pocket_menu.close()


@pytest.mark.parametrize(
    "payload",
    [
        b"not JSON",
        b"[]",
        b'{"version":2,"pockets":[{}]}',
        b'{"pockets":[{}]}',
        b'{"version":1,"pockets":[]}',
        b'{"version":1,"pockets":[{}, {"radii":[-1,1]}]}',
        b'{"version":1,"pockets":[{}, {"color":"invalid color"}]}',
        b'{"version":1,"pockets":[{}, {"color":""}]}',
        b'{"version":1,"pockets":[{}, {"color":"#12"}]}',
        b'{"version":1,"pockets":[{}, {"color":"#12345"}]}',
    ],
)
def test_invalid_pocket_clipboard_is_atomic(overlays, qtbot, pocket_clipboard, payload):
    overlays.add_pocket()
    before = overlays.state.model_copy(deep=True)
    mime = QtCore.QMimeData()
    mime.setData(_POCKET_MIME_TYPE, payload)
    pocket_clipboard.setMimeData(mime)
    assert not overlays.paste_pockets_action.isEnabled()
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        overlays._paste_pockets()
    assert overlays.state == before
    assert set(overlays._pocket_rois) == {before.pockets[0].uid}


def test_pocket_clipboard_availability_and_lifetime(
    overlays, pocket_clipboard, monkeypatch, qtbot
):
    overlays.add_pocket()
    overlays.copy_pockets_action.trigger()
    assert overlays.paste_pockets_action.isEnabled()
    overlays.set_available(False)
    assert not any(action.isEnabled() for action in overlays.pocket_menu.actions())
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        overlays._rename_pocket()
        overlays._copy_pockets()
        overlays._paste_pockets()
    overlays.set_available(True)
    assert all(action.isEnabled() for action in overlays.pocket_menu.actions())
    with monkeypatch.context() as patch:
        patch.setattr(QtWidgets.QApplication, "clipboard", lambda: None)
        overlays._update_pocket_actions()
        assert not overlays.paste_pockets_action.isEnabled()
        with qtbot.assertNotEmitted(overlays.sigStateChanged):
            overlays._copy_pockets()
            overlays._paste_pockets()
    pocket_clipboard.setText("Ordinary clipboard text")
    assert not overlays.paste_pockets_action.isEnabled()
    closed = MomentumOverlayEditor(overlays.momentum_plot)
    qtbot.addWidget(closed)
    closed.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    # The global clipboard must not call into an editor after Qt deletes it.
    pocket_clipboard.setText("Clipboard change after editor deletion")


def test_pocket_clipboard_shortcuts_and_native_text_editing(
    overlays, qtbot, pocket_clipboard
):
    overlays.add_pocket(state=PocketState(name="First pocket", color="r"))
    overlays.show()
    overlays.activateWindow()
    pocket_list = overlays.pocket_list
    pocket_list.setFocus()
    qtbot.waitUntil(pocket_list.hasFocus)
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        qtbot.keySequence(pocket_list, QtGui.QKeySequence.StandardKey.Copy)
    assert pocket_clipboard.mimeData().hasFormat(_POCKET_MIME_TYPE)
    with qtbot.waitSignal(overlays.sigStateChanged):
        qtbot.keySequence(pocket_list, QtGui.QKeySequence.StandardKey.Paste)
    assert len(overlays.state.pockets) == 2
    assert overlays.state.pockets[0].uid != overlays.state.pockets[1].uid
    assert overlays.state.pockets[1].color == "r"
    overlays.rename_pocket_action.trigger()
    edit = pocket_list.findChild(QtWidgets.QLineEdit)
    assert edit is not None
    edit.setText("Text edit")
    edit.selectAll()
    edit.setFocus()
    qtbot.waitUntil(edit.hasFocus)
    before = overlays.state.model_copy(deep=True)
    # A name editor keeps Qt's text shortcuts instead of pasting another pocket.
    with qtbot.assertNotEmitted(overlays.sigStateChanged):
        qtbot.keySequence(edit, QtGui.QKeySequence.StandardKey.Copy)
        edit.setCursorPosition(len(edit.text()))
        qtbot.keySequence(edit, QtGui.QKeySequence.StandardKey.Paste)
        assert edit.text() == "Text editText edit"
        qtbot.keyClick(edit, QtCore.Qt.Key.Key_Escape)
    assert overlays.state == before


@pytest.fixture
def overlay_window(qtbot, anglemap):
    from erlab.interactive.angle_calculator._window import AngleCalculatorWindow
    from erlab.interactive.kspace import KspaceTool

    def make(kind):
        if kind == "ktool":
            data = anglemap.isel(alpha=slice(0, 3), beta=slice(0, 3), eV=slice(0, 3))
            window = KspaceTool(data.copy(deep=True).assign_coords(xi=0.0))
            window.tabWidget.setCurrentWidget(window.overlay_tab)
        else:
            window = AngleCalculatorWindow()
            scroll = next(
                scroll
                for scroll in window.tabs.findChildren(QtWidgets.QScrollArea)
                if scroll.isAncestorOf(window.overlays)
            )
            window.tabs.setCurrentWidget(scroll)
        qtbot.addWidget(window)
        return window

    return make


@pytest.mark.parametrize("kind", ["ktool", "calculator"])
def test_overlay_image_filename_wrapped_height(
    overlay_window, image_files, qtbot, kind
):
    path = image_files[0].with_name(
        "a long image filename with several words that must remain visible "
        "after wrapping to multiple lines in a narrow overlay panel.png"
    )
    path.write_bytes(image_files[0].read_bytes())
    win = overlay_window(kind)
    scroll = next(
        scroll
        for scroll in win.findChildren(QtWidgets.QScrollArea)
        if scroll.isAncestorOf(win.overlays)
    )
    win.show()
    win.overlays.load_image(str(path))
    label = win.overlays.image_name
    for width in (340, 260):
        scroll.setFixedWidth(width)
        qtbot.wait(20)
        qtbot.waitUntil(
            lambda: (
                label.height() >= label.heightForWidth(label.width())
                and label.height() > label.fontMetrics().height()
            )
        )
        assert label.text() == path.name
        assert scroll.horizontalScrollBar().maximum() == 0
        assert label.geometry().bottom() < win.overlays.image_locked.geometry().top()


@pytest.mark.parametrize("kind", ["ktool", "calculator"])
@pytest.mark.parametrize("close", [False, True])
@pytest.mark.parametrize("process_before_undo", [False, True])
def test_pocket_menu_removal_history(
    overlay_window, qtbot, kind, close, process_before_undo
):
    win = overlay_window(kind)
    win.show()
    roi = win.overlays.add_pocket()
    win._flush_pending_history_write()
    before = win._history_state
    uid = before.overlays.pockets[0].uid
    scene = roi.scene()
    action = roi.getMenu().remAct
    action.trigger()
    # Record the edit before the next command, but leave the graphics item in
    # its scene until the menu event handler has returned.
    assert not win.overlays.state.pockets
    assert roi.scene() is scene
    after = win._history_state
    if close:
        assert win.close()
        assert not win._history_write_pending
    if process_before_undo:
        qtbot.wait_until(lambda: not qt_is_valid(roi))
    if close:
        win.show()
    win.undo()
    assert win._history_state == before
    replacement = win.overlays._pocket_rois[uid]
    assert replacement is not roi
    # Pending cleanup must target the retired object, even when Undo restores
    # a pocket with the same UID before the event loop runs.
    qtbot.wait_until(lambda: not qt_is_valid(roi))
    assert win._history_state == before
    assert qt_is_valid(replacement)
    assert replacement.scene() is scene
    win.redo()
    assert win._history_state == after


@pytest.mark.parametrize("kind", ["ktool", "calculator"])
def test_pocket_menu_removal_menu_lifetime(overlay_window, qtbot, kind):
    win = overlay_window(kind)
    win.show()
    roi = win.overlays.add_pocket()
    win._flush_pending_history_write()
    before = win._history_state
    menu = roi.getMenu()
    menu.popup(win.mapToGlobal(QtCore.QPoint(60, 60)))
    qtbot.wait_until(menu.isVisible)
    qtbot.keyClick(menu, QtCore.Qt.Key.Key_Escape)
    assert not menu.isVisible()
    assert win._history_state == before
    menu.popup(win.mapToGlobal(QtCore.QPoint(60, 60)))
    qtbot.wait_until(menu.isVisible)
    menu.setActiveAction(roi.remove_action)
    qtbot.keyClick(menu, QtCore.Qt.Key.Key_Return)
    assert not menu.isVisible()
    assert not win.overlays.state.pockets
    qtbot.wait_until(lambda: not qt_is_valid(roi))
    win.undo()
    assert win._history_state == before


@pytest.mark.parametrize("kind", ["ktool", "calculator"])
def test_pocket_menu_removal_window_deleted(overlay_window, qtbot, kind):
    win = overlay_window(kind)
    roi = win.overlays.add_pocket()
    roi.getMenu().remAct.trigger()
    win.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert not qt_is_valid(win)
    assert not qt_is_valid(roi)
    # Run the queued graphics cleanup after the entire host has been deleted.
    QtWidgets.QApplication.processEvents()


@pytest.mark.parametrize("deleted", ["editor", "plot", "roi"])
def test_pocket_menu_removal_cleanup(qtbot, deleted):
    view = pg.PlotWidget()
    qtbot.addWidget(view)
    plot = view.getPlotItem()
    overlays = MomentumOverlayEditor(plot)
    qtbot.addWidget(overlays)
    roi = overlays.add_pocket()
    notifications = []
    overlays.sigStateChanged.connect(lambda: notifications.append(None))
    action = roi.getMenu().remAct
    action.trigger()
    action.trigger()
    assert not overlays.state.pockets
    assert len(notifications) == 1
    target = {"editor": overlays, "plot": plot, "roi": roi}[deleted]
    target.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert not qt_is_valid(target)
    QtWidgets.QApplication.processEvents()
    qtbot.wait_until(lambda: not qt_is_valid(roi))
    if deleted == "editor":
        assert roi not in plot.items


@pytest.mark.parametrize("edit", ["add", "move", "remove"])
def test_inactive_overlay_edits_notify(overlays, edit):
    roi = overlays.add_pocket()
    overlays.set_active(False)
    notifications = []
    overlays.sigStateChanged.connect(
        lambda: notifications.append(overlays.state.model_copy(deep=True))
    )
    if edit == "add":
        overlays.add_pocket()
        assert len(overlays.state.pockets) == 2
    elif edit == "move":
        roi.translate((0.1, 0.2))
        assert overlays.state.pockets[0].center == pytest.approx((0.1, 0.2))
    else:
        roi.getMenu().remAct.trigger()
        assert not overlays.state.pockets
    assert notifications == [overlays.state]
    assert not overlays._redraw_timer.isActive()
    assert not overlays._refine_timer.isActive()


@pytest.mark.parametrize("kind", ["ktool", "calculator"])
def test_image_lock_history(overlay_window, image_files, monkeypatch, qtbot, kind):
    from erlab.accessors.kspace import MomentumAccessor

    win = overlay_window(kind)
    win.overlays.load_image(str(image_files[0]))
    win._flush_pending_history_write()
    before = win._history_state

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("Locking an overlay must not convert measured data.")

    with monkeypatch.context() as patch:
        patch.setattr(MomentumAccessor, "convert", unexpected_conversion)
        win.overlays.image_locked.click()
        win._flush_pending_history_write()
        qtbot.wait(200)
    expected = before.model_copy(deep=True)
    expected.overlays.image.locked = True
    assert win._history_state == expected
    assert not win.overlays._image_roi.isVisible()
    win.undo()
    assert win._history_state == before
    assert not win.overlays.image_locked.isChecked()
    assert win.overlays._image_roi.isVisible()
    win.redo()
    assert win._history_state == expected
    assert win.overlays.image_locked.isChecked()
    assert not win.overlays._image_roi.isVisible()


@pytest.mark.parametrize("kind", ["ktool", "calculator"])
@pytest.mark.parametrize("overlay", ["pocket", "image"])
@pytest.mark.parametrize(
    ("finish", "previous_edit"),
    [
        ("release", None),
        ("undo", None),
        ("close", None),
        ("release", "pending"),
        ("cancel", "redo"),
        ("unchanged", "redo"),
    ],
)
def test_overlay_drag_history(
    overlay_window,
    image_files,
    qtbot,
    monkeypatch,
    kind,
    overlay,
    finish,
    previous_edit,
):
    monkeypatch.setitem(pg.CONFIG_OPTIONS, "mouseRateLimit", 0)
    win = overlay_window(kind)
    if overlay == "pocket":
        roi = win.overlays.add_pocket(
            state=PocketState(
                center=(0.012345678, -0.023456789),
                radii=(0.07, 0.05),
                rotation=12.345678,
            )
        )
    else:
        win.overlays.load_image(str(image_files[0]))
        roi = win.overlays._image_roi
        roi.setGeometry((0.012345678, -0.023456789), 0.14, rotation=12.345678)
    win._flush_pending_history_write()
    initial = win.overlays.state.model_copy(deep=True)
    win.resize(1500, 700)
    win.show()
    qtbot.waitExposed(win)
    qtbot.wait(30)
    plot = win.overlays.momentum_plot
    plot.setRange(xRange=(-0.3, 0.3), yRange=(-0.3, 0.3), padding=0)
    qtbot.wait(30)
    if previous_edit is not None:
        win.overlays.inputs[f"{overlay}_rotation"].setValue(25)
        prior_edit = win.overlays.state.model_copy(deep=True)
        if previous_edit == "redo":
            win.undo()
            roi = (
                next(iter(win.overlays._pocket_rois.values()))
                if overlay == "pocket"
                else win.overlays._image_roi
            )
    before = win.overlays.state.model_copy(deep=True)
    view = plot.scene().views()[0]
    viewport = view.viewport()
    # Pick the ROI body away from its resize and rotation handles.
    start = view.mapFromScene(roi.mapToScene(roi.size() * pg.Point(0.5, 0.25)))
    qtbot.mouseMove(viewport, pos=start + QtCore.QPoint(20, 20))
    qtbot.mouseMove(viewport, pos=start)
    qtbot.wait(20)
    qtbot.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    end = start
    if finish == "unchanged":
        # A gesture can start without changing its geometry.
        roi.sigRegionChangeStarted.emit(roi)
    if finish != "unchanged":
        for delta in (20, 40):
            end = start + QtCore.QPoint(delta, -delta)
            QtWidgets.QApplication.sendEvent(
                viewport,
                QtGui.QMouseEvent(
                    QtCore.QEvent.Type.MouseMove,
                    QtCore.QPointF(end),
                    QtCore.QPointF(viewport.mapToGlobal(end)),
                    QtCore.Qt.MouseButton.NoButton,
                    QtCore.Qt.MouseButton.LeftButton,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                ),
            )
            # A pause longer than the history debounce must remain one gesture.
            qtbot.wait(200)
            current = (
                win.overlays.state.pockets[0]
                if overlay == "pocket"
                else win.overlays.state.image
            )
            original = before.pockets[0] if overlay == "pocket" else before.image
            assert current.center != original.center
    after = win.overlays.state.model_copy(deep=True)
    if finish in ("release", "unchanged"):
        qtbot.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
        if finish == "unchanged":
            roi.sigRegionChangeFinished.emit(roi)
    elif finish == "undo":
        win.undo()
    elif finish == "close":
        assert win.close()
    else:
        qtbot.mouseClick(viewport, QtCore.Qt.MouseButton.RightButton, pos=end)
    if finish in ("release", "close"):
        win.undo()
    assert win.overlays.state == before
    qtbot.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    qtbot.wait(200)
    assert win.overlays.state == before
    if previous_edit == "pending":
        win.undo()
        assert win.overlays.state == initial
        win.redo()
        assert win.overlays.state == before
    win.redo()
    assert win.overlays.state == (prior_edit if previous_edit == "redo" else after)


@pytest.mark.parametrize("kind", ["ktool", "calculator"])
@pytest.mark.parametrize("overlay", ["pocket", "image"])
def test_overlay_drag_state_restore(overlay_window, image_files, qtbot, kind, overlay):
    win = overlay_window(kind)
    if overlay == "pocket":
        roi = win.overlays.add_pocket()
    else:
        win.overlays.load_image(str(image_files[0]))
        roi = win.overlays._image_roi
    win._flush_pending_history_write()
    restored = win.overlays.state.model_copy(deep=True)
    workspace = (
        win.workspace_state_payload() if kind == "calculator" else win.tool_status
    )
    roi.sigRegionChangeStarted.emit(roi)
    roi.setPos(roi.pos() + pg.Point(0.01, 0.02), finish=False)
    if kind == "calculator":
        win.restore_workspace_state(workspace)
    else:
        # KspaceTool reconstructs workspace settings through its status setter.
        with win._history_suppressed():
            win.tool_status = workspace
        win._reset_history_stack()
    # Old queued motion and release events must not mutate the restored state.
    roi.setPos(roi.pos() + pg.Point(0.03, 0.04), finish=False)
    roi.sigRegionChangeFinished.emit(roi)
    qtbot.wait(200)
    assert win.overlays.state == restored
    assert not win.undoable
    assert not win.redoable
    current_roi = (
        next(iter(win.overlays._pocket_rois.values()))
        if overlay == "pocket"
        else win.overlays._image_roi
    )
    assert current_roi.geometry()[0] == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize("kind", ["ktool", "calculator"])
@pytest.mark.parametrize("field", ["pocket_x", "image_width", "name"])
def test_overlay_close_commits_focused_editor(
    overlay_window, qtbot, image_files, kind, field
):
    win = overlay_window(kind)
    win.overlays.add_pocket()
    win.overlays.load_image(str(image_files[0]))
    win._flush_pending_history_write()
    before = win._history_state
    history_length = len(win._prev_states)
    win.show()
    win.activateWindow()
    if field == "name":
        win.overlays.rename_pocket_action.trigger()
        editor = win.overlays.pocket_list.findChild(QtWidgets.QLineEdit)
        text = "Renamed pocket"
    else:
        editor = win.overlays.inputs[field]
        text = "0.2" if field == "pocket_x" else "0.81234567"
    assert editor is not None
    editor.setFocus()
    qtbot.wait_until(editor.hasFocus)
    editor.selectAll()
    qtbot.keyClicks(editor, text)
    assert win._history_state == before
    assert win.close()
    expected = before.model_copy(deep=True)
    if field == "name":
        expected.overlays.pockets[0].name = text
    elif field == "pocket_x":
        expected.overlays.pockets[0].center = (float(text), 0.0)
    else:
        expected.overlays.image.width = float(text)
    assert win._history_state == expected
    assert len(win._prev_states) == history_length + 1
    assert not win._history_write_pending
    assert not win._history_write_timer.isActive()
    assert not win.overlays._redraw_timer.isActive()
    assert not win.overlays._refine_timer.isActive()
    win.show()
    qtbot.wait(200)
    assert len(win._prev_states) == history_length + 1
    win.undo()
    assert win._history_state == before
    win.redo()
    assert win._history_state == expected


@pytest.mark.parametrize("source_kind", ["ktool", "calculator"])
@pytest.mark.parametrize("target_kind", ["ktool", "calculator"])
def test_pocket_clipboard_interoperability_and_history(
    overlay_window, pocket_clipboard, monkeypatch, qtbot, source_kind, target_kind
):
    from erlab.accessors.kspace import MomentumAccessor

    source, target = overlay_window(source_kind), overlay_window(target_kind)

    conversions = []
    convert = MomentumAccessor.convert

    def track_convert(*args, **kwargs):
        conversions.append(1)
        return convert(*args, **kwargs)

    monkeypatch.setattr(MomentumAccessor, "convert", track_convert)
    for pocket in (
        PocketState(name="Electron Γ", center=(0.1, -0.2), rotation=35),
        PocketState(
            name="Hole M",
            center=(-0.3, 0.4),
            radii=(0.1, 0.2),
            point_group="6mm",
            symmetry_angle=20,
            visible=False,
            symmetry=False,
            color="#c12345",
        ),
    ):
        source.overlays.add_pocket(state=pocket)
    source.overlays.pocket_list.selectAll()
    source._flush_pending_history_write()
    source_state = source._history_state
    with qtbot.assertNotEmitted(source.overlays.sigStateChanged):
        source.overlays.copy_pockets_action.trigger()
    assert not source._history_write_pending
    before = target._history_state
    assert target.overlays.paste_pockets_action.isEnabled()
    target.overlays.paste_pockets_action.trigger()
    target._flush_pending_history_write()
    pasted = target._history_state
    assert not conversions
    source_pockets = source.overlays.state.pockets
    target_pockets = target.overlays.state.pockets
    assert [p.model_dump(exclude={"uid"}) for p in target_pockets] == [
        p.model_dump(exclude={"uid"}) for p in source_pockets
    ]
    assert not {p.uid for p in source_pockets}.intersection(
        p.uid for p in target_pockets
    )
    target.undo_action.trigger()
    assert target._history_state == before
    target.redo_action.trigger()
    assert target._history_state == pasted
    conversion_count = len(conversions)
    target.overlays.pocket_list.item(0).setText("Renamed pocket")
    target._flush_pending_history_write()
    assert len(conversions) == conversion_count
    renamed = target._history_state
    target.undo_action.trigger()
    assert target._history_state == pasted
    target.redo_action.trigger()
    assert target._history_state == renamed
    conversion_count = len(conversions)
    target.overlays.paste_pockets_action.trigger()
    target._flush_pending_history_write()
    assert len({p.uid for p in target.overlays.state.pockets}) == 4
    assert len(target.overlays.pocket_list.selectedItems()) == 2
    target.overlays.inputs["pocket_x"].setValue(0.6)
    assert source._history_state == source_state
    assert target.overlays.state.pockets[1].center == source_pockets[1].center
    assert len(conversions) == conversion_count


def test_explicit_translations_refresh_and_restore_lattice(overlays):
    overlays.add_pocket()
    overlays.pocket_symmetry.setChecked(False)
    overlays.set_lattice(np.eye(2), (1, 0))
    overlays.redraw()
    initial_cache = overlays._pocket_cache
    assert np.count_nonzero(np.isnan(overlays._contours[0][0].getData()[0])) == 3
    shifts = np.array([[0.0, 0.0], [0.7, -0.3]])
    overlays.set_translations(shifts)
    assert overlays._redraw_timer.isActive()
    shifts[1] = [9, 9]
    overlays.redraw()
    assert overlays._pocket_cache is not initial_cache
    pocket = overlays.state.pockets[0]
    expected = _geometry.ellipse_points(pocket.center, pocket.radii, pocket.rotation)
    points = np.column_stack(overlays._contours[0][0].getData())
    np.testing.assert_allclose(points[:181], expected)
    np.testing.assert_allclose(points[182:363], expected + np.array([0.7, -0.3]))
    explicit_cache = overlays._pocket_cache
    overlays.set_translations([[0, 0], [0.7, -0.3]])
    assert not overlays._redraw_timer.isActive()
    overlays.redraw()
    assert overlays._pocket_cache is explicit_cache
    overlays.set_translations(None)
    overlays.redraw()
    assert np.count_nonzero(np.isnan(overlays._contours[0][0].getData()[0])) == 3
    overlays.set_translations(np.empty((0, 2)))
    overlays.redraw()
    assert not any(primary.isVisible() for primary, _ in overlays._contours)
    # An unchanged lattice must still reset explicit translation mode.
    overlays.set_lattice(np.eye(2), (1, 0))
    assert overlays._redraw_timer.isActive()
    overlays.redraw()
    assert overlays._contours[0][0].isVisible()
    assert np.count_nonzero(np.isnan(overlays._contours[0][0].getData()[0])) == 3
    overlays.set_lattice(np.eye(2), (1, 0))
    overlays.set_translations(None)
    assert not overlays._redraw_timer.isActive()


@pytest.mark.parametrize(
    "shifts", [[], [1, 2], [[0]], [[0, 1, 2]], [[np.nan, 0]], [[0, np.inf]]]
)
def test_overlay_translation_validation(overlays, shifts):
    with pytest.raises(ValueError, match="translations"):
        overlays.set_translations(shifts)
