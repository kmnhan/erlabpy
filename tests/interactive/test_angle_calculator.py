import contextlib
import dataclasses
import importlib
import runpy

import numpy as np
import pydantic
import pytest
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._momentum_overlays import _geometry
from erlab.interactive._momentum_overlays._state import (
    ImageState,
    MomentumOverlayState,
    PocketState,
)
from erlab.interactive.angle_calculator import _calculation as calc
from erlab.interactive.angle_calculator._state import AngleCalculatorState
from erlab.interactive.angle_calculator._window import AngleCalculatorWindow


@pytest.fixture
def angle_calculator(qtbot):
    window = AngleCalculatorWindow()
    qtbot.addWidget(window)
    return window


def test_angle_calculator_marker_survives_resize(angle_calculator, qtbot):
    angle_calculator.show()
    qtbot.waitExposed(angle_calculator)
    angle_calculator.cut_marker.setValue(3)
    angle_calculator.resize(1200, 700)
    qtbot.wait(20)
    assert angle_calculator.current_spin.value() == pytest.approx(3)
    for plot in (angle_calculator.momentum_plot, angle_calculator.angle_plot):
        assert np.isfinite(plot.viewRange()).all()


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


def test_angle_calculator_launcher(qtbot, monkeypatch):
    calls = []

    @contextlib.contextmanager
    def setup(execute):
        calls.append(execute)
        yield

    monkeypatch.setattr(erlab.interactive.utils, "setup_qapp", setup)
    window = erlab.interactive.angle_calculator(execute=False)
    qtbot.addWidget(window)
    assert window.isVisible()
    assert calls == [False]
    window.close()
    runpy.run_module("erlab.interactive.angle_calculator", run_name="__main__")
    assert calls == [False, True]
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(widget, AngleCalculatorWindow):
            qtbot.addWidget(widget)
            widget.close()


def test_angle_calculator_module_callable(qtbot):
    module = importlib.import_module("erlab.interactive.angle_calculator")
    window = module(execute=False)
    qtbot.addWidget(window)
    assert isinstance(window, AngleCalculatorWindow)
    assert window.isVisible()
    namespace_window = erlab.interactive.angle_calculator(execute=False)
    qtbot.addWidget(namespace_window)
    assert isinstance(namespace_window, AngleCalculatorWindow)
    assert namespace_window.isVisible()


def test_angle_calculator_modes_and_marker(angle_calculator):
    angle_calculator.slit_combo.setCurrentIndex(1)
    angle_calculator.inputs["polar"].setValue(7)
    angle_calculator.inputs["tilt"].setValue(-9)
    angle_calculator.scan_control.setValues((-8, 12))
    angle_calculator.current_spin.setValue(4)
    angle_calculator.da_check.setChecked(False)
    assert angle_calculator._state.geometry.configuration == 2
    assert angle_calculator.inputs["polar"].isEnabled()
    assert not angle_calculator.inputs["tilt"].isEnabled()
    angle_calculator.scan_control.setValues((-3, 6))
    angle_calculator.cut_marker.setValue(5)
    assert angle_calculator.current_spin.value() == 5
    assert angle_calculator._state.scan_positions["tilt"] == 5
    angle_calculator.slit_combo.setCurrentIndex(0)
    assert angle_calculator._state.geometry.configuration == 1
    assert not angle_calculator.inputs["polar"].isEnabled()
    assert angle_calculator.inputs["tilt"].isEnabled()
    angle_calculator.scan_control.setValues((-2, 3))
    angle_calculator.current_spin.setValue(2)
    angle_calculator.da_check.setChecked(True)
    assert angle_calculator._state.geometry.configuration == 3
    assert angle_calculator.scan_control.values() == (-8, 12)
    assert angle_calculator.current_spin.value() == 4
    assert angle_calculator.inputs["polar"].value() == 7
    assert angle_calculator.inputs["tilt"].value() == -9
    angle_calculator.slit_combo.setCurrentIndex(1)
    assert angle_calculator._state.geometry.configuration == 4
    angle_calculator.da_check.setChecked(False)
    assert angle_calculator.scan_control.values() == (-3, 6)
    assert angle_calculator.current_spin.value() == 5
    angle_calculator.scan_control.setValues((1, 1))
    assert (
        angle_calculator.current_spin.value()
        == angle_calculator.cut_marker.value()
        == 1
    )
    angle_calculator._redraw()
    x, y = angle_calculator.scan_curves[1].getData()
    assert np.all(y == 1)
    assert np.ptp(x) > 0


def test_angle_calculator_lattice(angle_calculator):
    angle_calculator.inputs["lattice_a"].setValue(5)
    assert angle_calculator._state.lattice_a == pytest.approx(5)
    old = angle_calculator._state
    angle_calculator.inputs["lattice_b"].setValue(0)
    assert angle_calculator._state == old
    angle_calculator.inputs["lattice_a"].setValue(6)
    assert angle_calculator._state.lattice_a == 6
    assert angle_calculator._state.lattice_b == old.lattice_b
    assert angle_calculator.inputs["lattice_b"].value() == old.lattice_b
    angle_calculator.inputs["lattice_b"].setValue(4)
    assert angle_calculator._state.lattice_b == pytest.approx(4)
    angle_calculator._redraw()
    angle_calculator.bz_check.setChecked(False)
    angle_calculator._redraw()
    assert all(not curve.isVisible() for curve in angle_calculator.bz_curves)


def test_angle_calculator_nonreduced_lattice(angle_calculator):
    state = AngleCalculatorState(
        lattice_a=3.5,
        lattice_b=np.hypot(7, 3.5),
        lattice_gamma=np.rad2deg(np.arctan2(3.5, 7)),
        momentum_range=((-2, 2), (-2, 2)),
    )
    angle_calculator.restore_workspace_state(state)
    momentum = np.column_stack(angle_calculator.bz_curves[0].getData())
    assert np.isfinite(momentum).any()
    restored_state = AngleCalculatorState.model_validate(
        angle_calculator.workspace_state_payload()
    )
    for field in ("lattice_a", "lattice_b", "lattice_gamma"):
        assert getattr(restored_state, field) == getattr(state, field)
    alpha, beta = angle_calculator.bz_curves[1].getData()
    projected = np.column_stack(
        calc.angles_to_momentum(alpha, beta, angle_calculator._state.geometry)
    )
    valid = np.isfinite(projected).all(axis=1)
    assert valid.any()
    np.testing.assert_allclose(projected[valid], momentum[valid], atol=1e-10)


def test_angle_calculator_overlay_projection_updates(angle_calculator, image_files):
    window = angle_calculator
    window.overlays.load_image(str(image_files[0]))
    for energy in (6.0, 8.0):
        window.inputs["photon_energy"].setValue(energy)
        window._redraw()
        window.overlays._draw_image(32)
        assert window.overlays._projection_token == window._state.geometry
        limits = window.angle_plot.viewRange()
        axes = [
            np.linspace(lo, hi, 32, endpoint=False) + (hi - lo) / 64
            for lo, hi in limits
        ]
        expected_grid = calc.angles_to_momentum(
            *np.meshgrid(*axes), window._state.geometry
        )
        expected = _geometry.warp_image(
            window.overlays._source_image, *expected_grid, (0, 0), 0.3
        )
        np.testing.assert_array_equal(window.overlays.projected_image.image, expected)


def test_angle_calculator_workspace_embedded_image(
    angle_calculator, image_files, qtbot
):
    angle_calculator.overlays.load_image(str(image_files[0]))
    angle_calculator.overlays.add_pocket()
    angle_calculator.overlays.inputs["image_rotation"].setValue(63)
    angle_calculator.overlays.image_locked.setChecked(True)
    angle_calculator.overlays.image_invert.setChecked(True)
    angle_calculator.overlays.image_black_transparent.setChecked(True)
    angle_calculator.overlays.inputs["image_black_threshold"].setValue(25)
    angle_calculator.overlays.inputs["pocket_rotation"].setValue(42)
    angle_calculator.scan_control.setValues((-2, 9))
    angle_calculator.current_spin.setValue(3)
    payload = angle_calculator.workspace_state_payload()
    expected_image = angle_calculator.overlays._source_image.copy()
    image_files[0].unlink()
    restored = AngleCalculatorWindow()
    qtbot.addWidget(restored)
    restored.restore_workspace_state(AngleCalculatorState.model_validate(payload))
    np.testing.assert_array_equal(restored.overlays._source_image, expected_image)
    assert restored.overlays.state.pockets == angle_calculator.overlays.state.pockets
    assert restored.scan_control.values() == (-2, 9)
    assert restored.current_spin.value() == 3
    assert (
        restored.overlays._image_roi.geometry()
        == angle_calculator.overlays._image_roi.geometry()
    )
    assert restored.overlays.inputs["image_rotation"].value() == 63
    assert restored.overlays.state.image.locked
    assert restored.overlays.image_locked.isChecked()
    assert not restored.overlays._image_roi.isVisible()
    assert restored.overlays.image_invert.isChecked()
    assert restored.overlays.image_black_transparent.isChecked()
    assert restored.overlays.inputs["image_black_threshold"].value() == 25
    restored.restore_workspace_state(AngleCalculatorState())
    assert not restored.overlays._pocket_rois
    assert restored.overlays._source_image is None
    assert restored.overlays._image_roi is None


def test_angle_calculator_close_deferred(angle_calculator, qtbot):
    angle_calculator.show()
    roi = angle_calculator.overlays.add_pocket()
    roi.getMenu().remAct.trigger()
    angle_calculator.close()
    assert not angle_calculator._redraw_timer.isActive()
    assert not angle_calculator._pocket_redraw_timer.isActive()
    assert not angle_calculator.overlays._refine_timer.isActive()
    qtbot.wait(10)
    angle_calculator._redraw()
    angle_calculator._redraw_pockets()
    angle_calculator.overlays._refine_image()
    angle_calculator.overlays._schedule_redraw()
    assert not angle_calculator._redraw_timer.isActive()
    assert not angle_calculator.overlays._refine_timer.isActive()
    angle_calculator.show()
    assert not angle_calculator._closed
    assert angle_calculator._redraw_timer.isActive()


def test_angle_calculator_collapsed_scan_preserves_image(angle_calculator, image_files):
    angle_calculator.overlays.load_image(str(image_files[0]))
    angle_calculator.da_check.setChecked(False)
    angle_calculator.scan_control.setValues((0, 0))
    angle_calculator._redraw()
    assert angle_calculator.overlays.momentum_image.isVisible()
    assert angle_calculator.overlays.projected_image.isVisible()
    assert np.any(angle_calculator.overlays.projected_image.image[..., 3])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("geometry.configuration", 5),
        ("geometry.photon_energy", 0),
        ("geometry.photon_energy", 1000.001),
        ("geometry.binding_energy", 1),
        ("geometry.binding_energy", -100001),
        ("geometry.analyzer_work_function", -1),
        ("geometry.sample_work_function", -1),
        ("geometry.inner_potential", -1),
        ("geometry.inner_potential", 50.001),
        ("hv_range", (90, 20)),
        ("hv_range", (0, 30)),
        ("centering", "X"),
        ("out_of_plane_range", ((0, 0), (1, 5))),
        ("geometry.analyzer_work_function", 101),
        ("geometry.polar", 91),
        ("geometry.tilt", -91),
        ("geometry.azimuth", 361),
        ("geometry.photon_energy", float("nan")),
        ("alpha_range", (2, -2)),
        ("polar_range", (-91, 0)),
        ("deflector_range", (-46, 0)),
        ("deflector_range", (0, 46)),
        ("scan_positions", {"polar": 0}),
        ("scan_positions", {"polar": 30, "tilt": 0, "deflector": 0}),
        ("lattice_gamma", 180),
        ("overlays.pockets", [{"point_group": "5"}]),
        ("overlays.pockets", [{"symmetry_angle": 361}]),
        ("angle_range", ((0, 0), (-1, 1))),
        ("overlays.pockets", [{"uid": "same"}, {"uid": "same"}]),
        ("overlays.pockets", [{"radii": [0, 1]}]),
        ("overlays.pockets", [{"radii": [101, 1]}]),
        ("overlays.pockets", [{"center": [101, 0]}]),
        ("overlays.pockets", [{"rotation": 361}]),
    ],
)
def test_angle_calculator_rejects_invalid_state(field, value):
    payload = AngleCalculatorState().model_dump(mode="json")
    target = payload
    *parents, name = field.split(".")
    for parent in parents:
        target = target[parent]
    target[name] = value
    with pytest.raises(pydantic.ValidationError):
        AngleCalculatorState.model_validate(payload)


@pytest.mark.parametrize("encoded", ["invalid base64", "eA=="])
def test_angle_calculator_rejects_invalid_embedded_image(encoded):
    with pytest.raises(pydantic.ValidationError):
        ImageState(png_base64=encoded)


def test_angle_calculator_pockets_and_ranges(angle_calculator):
    angle_calculator.overlays.add_pocket(state=None)
    angle_calculator.overlays.inputs["pocket_x"].setValue(3)
    angle_calculator.alpha_control.setValues((-4, 12))
    assert angle_calculator._state.alpha_range == (-4, 12)
    angle_calculator.inputs["photon_energy"].setValue(0.1)
    assert np.isfinite(angle_calculator.momentum_plot.viewRange()).all()
    angle_calculator.da_check.setChecked(False)
    angle_calculator.slit_combo.setCurrentIndex(0)
    angle_calculator.scan_control.setValues((-1, 2))
    angle_calculator._redraw()
    assert np.isfinite(angle_calculator.scan_curves[1].getData()).all()


@pytest.mark.parametrize("bz_visible", [True, False])
def test_angle_calculator_pocket_copies_follow_view(angle_calculator, bz_visible):
    win = angle_calculator
    win.restore_workspace_state(
        AngleCalculatorState(
            lattice_a=2 * np.pi,
            lattice_b=2 * np.pi,
            lattice_c=2 * np.pi,
            lattice_gamma=90,
        )
    )
    win.overlays.add_pocket()
    win.bz_check.setChecked(bz_visible)
    for plot in (win.momentum_plot, win.angle_plot):
        plot.setAspectLocked(False)
        plot.setRange(xRange=(-0.1, 0.1), yRange=(-0.1, 0.1), padding=0)
    win._redraw()
    np.testing.assert_allclose(win.overlays._translations, [[0, 0]], atol=1e-12)

    win.momentum_plot.setRange(xRange=(1.8, 2.2), yRange=(-0.1, 0.1), padding=0)
    win._redraw()
    shifts = win.overlays._translations
    assert np.any(np.all(np.isclose(shifts, [2, 0]), axis=1))
    assert not np.any(np.all(np.isclose(shifts, [-2, 0]), axis=1))

    # Planar guides remain periodic outside the photoemission horizon.
    win.momentum_plot.setRange(xRange=(7.8, 8.2), yRange=(-0.1, 0.1), padding=0)
    win._redraw()
    assert np.any(np.all(np.isclose(win.overlays._translations, [8, 0]), axis=1))

    win.momentum_plot.setRange(xRange=(-2.2, 2.2), yRange=(-2.2, 2.2), padding=0)
    win._redraw()
    shifts = win.overlays._translations
    for expected in ([2, 0], [-2, 0], [0, 2], [0, -2]):
        assert np.any(np.all(np.isclose(shifts, expected), axis=1))


def test_angle_calculator_pocket_copies_face_centered(angle_calculator):
    win = angle_calculator
    lattice = 2 * np.pi * np.eye(3)
    win.restore_workspace_state(
        AngleCalculatorState(
            lattice_a=2 * np.pi,
            lattice_b=2 * np.pi,
            lattice_c=2 * np.pi,
            lattice_gamma=90,
            centering="F",
        )
    )
    win.overlays.add_pocket()
    basis = erlab.lattice.to_reciprocal(erlab.lattice.to_primitive(lattice, "F"))
    assert np.linalg.matrix_rank(basis[:2, :2]) == 1
    win.momentum_plot.setAspectLocked(False)
    win.momentum_plot.setRange(xRange=(-2.5, 2.5), yRange=(-2.5, 2.5), padding=0)
    win.angle_plot.setRange(xRange=(-0.1, 0.1), yRange=(-0.1, 0.1), padding=0)
    shifts = win._pocket_translations(basis)
    for expected in ([2, 0], [-2, 0], [0, 2], [0, -2]):
        assert np.any(np.all(np.isclose(shifts, expected), axis=1))
    assert len(np.unique(np.round(shifts, 10), axis=0)) == len(shifts)


def test_angle_calculator_skew_pocket_copy_survives_another_pocket(angle_calculator):
    win = angle_calculator
    basis = np.array([[1, 0, 0], [5 * np.cos(np.pi / 6), 2.5, 0], [0, 0, 1]])
    win.restore_workspace_state(
        AngleCalculatorState(
            lattice_a=4 * np.pi,
            lattice_b=0.8 * np.pi,
            lattice_c=2 * np.pi,
            lattice_gamma=150,
            lattice_rotation=-60,
        )
    )
    center = 0.49 * (basis[0, :2] + basis[1, :2])
    win.overlays.add_pocket(
        state=PocketState(center=tuple(center), radii=(0.01, 0.01), symmetry=False)
    )
    visible_center = center + basis[0, :2]
    angle_center = calc.momentum_to_angles(*visible_center, win._state.geometry)
    for plot, point in zip(
        (win.momentum_plot, win.angle_plot), (visible_center, angle_center), strict=True
    ):
        plot.setAspectLocked(False)
        plot.setRange(
            xRange=(point[0] - 0.01, point[0] + 0.01),
            yRange=(point[1] - 0.01, point[1] + 0.01),
            padding=0,
        )
    win._redraw()
    assert np.any(np.all(np.isclose(win.overlays._translations, [1, 0]), axis=1))
    win.overlays.add_pocket(
        state=PocketState(center=(4, 0), radii=(0.01, 0.01), symmetry=False)
    )
    win._redraw()
    assert np.any(np.all(np.isclose(win.overlays._translations, [1, 0]), axis=1))


def test_angle_calculator_bounds_budget_includes_skew_candidates(angle_calculator):
    basis = np.array([[1, 0, 0], [100, 0.1, 0], [0, 0, 1]])
    surface = (0.49 * (basis[0] + basis[1]))[None, None, :]
    assert not angle_calculator._surface_within_budget(basis, surface)


def test_angle_calculator_pocket_copies_empty_invalid_dense(angle_calculator):
    win = angle_calculator
    basis = np.eye(3)
    np.testing.assert_array_equal(win._pocket_translations(basis), [[0, 0]])
    win.overlays.add_pocket()
    pockets = tuple(win.overlays.state.pockets)
    views = (((-1.0, 1.0), (-1.0, 1.0)), ((-0.1, 0.1), (-0.1, 0.1)))
    assert win._visible_pocket_translations(basis * 1e-4, pockets, views) is None
    win.inputs["binding_energy"].setValue(-100)
    np.testing.assert_array_equal(
        win._visible_pocket_translations(basis, pockets, views), [[0, 0]]
    )


def test_angle_calculator_pocket_copies_refresh_after_edit(angle_calculator):
    win = angle_calculator
    win.overlays.add_pocket()
    for plot in (win.momentum_plot, win.angle_plot):
        plot.setAspectLocked(False)
        plot.setRange(xRange=(-0.1, 0.1), yRange=(-0.1, 0.1), padding=0)
    basis = np.eye(3)
    initial = win._pocket_translations(basis)
    np.testing.assert_allclose(initial, [[0, 0]], atol=1e-12)
    assert win._pocket_translations(basis) is initial
    win.overlays.inputs["pocket_x"].setValue(2)
    shifted = win._pocket_translations(basis)
    assert np.any(np.all(np.isclose(shifted, [-2, 0]), axis=1))
    win.inputs["binding_energy"].setValue(-100)
    np.testing.assert_array_equal(win._pocket_translations(basis), [[0, 0]])


def test_angle_calculator_restore_nondefault_controls(angle_calculator, qtbot):
    payload = AngleCalculatorState().model_dump(mode="json")
    payload.update(
        alpha_range=[-3, 4],
        polar_range=[-2, 8],
        scan_positions={"polar": 3, "tilt": 0, "deflector": 0},
        bz_visible=False,
        overlays={"pockets": [{"point_group": "m", "symmetry_angle": 30}]},
    )
    payload["geometry"].update(configuration=1, polar=2, tilt=3, azimuth=7)
    with qtbot.assertNotEmitted(angle_calculator.sigStateChanged):
        angle_calculator.restore_workspace_state(
            AngleCalculatorState.model_validate(payload)
        )
    assert angle_calculator.alpha_control.values() == (-3, 4)
    assert angle_calculator.scan_control.values() == (-2, 8)
    assert angle_calculator.current_spin.value() == 3
    assert angle_calculator._state.geometry.polar == 2
    assert angle_calculator.inputs["polar"].value() == 2
    assert angle_calculator.overlays.symmetry_spin.value() == 30
    assert not angle_calculator.bz_check.isChecked()


def test_angle_calculator_corrupt_embedded_image(angle_calculator):
    state = AngleCalculatorState(
        overlays=MomentumOverlayState(image=ImageState(png_base64="iVBORw0KGgo="))
    )
    with pytest.raises(ValueError, match="cannot be decoded"):
        angle_calculator.restore_workspace_state(state)
    assert not angle_calculator._restoring
    angle_calculator.restore_workspace_state(AngleCalculatorState())
    assert angle_calculator.overlays._source_image is None


def test_angle_calculator_rejects_nonfinite_geometry_instance():
    with pytest.raises(pydantic.ValidationError, match="must be finite"):
        AngleCalculatorState(geometry=calc.Geometry(photon_energy=float("nan")))


def test_angle_calculator_lattice_uses_ktool_settings(qtbot):
    options = erlab.interactive.options.model
    options.ktool.bz.default_a = 4.7
    options.ktool.bz.default_b = 6.2
    options.ktool.bz.default_gamma = 105
    options.ktool.bz.default_rot = 28
    erlab.interactive.options.model = options
    window = AngleCalculatorWindow()
    qtbot.addWidget(window)
    expected = {
        "lattice_a": 4.7,
        "lattice_b": 6.2,
        "lattice_c": 6.01,
        "lattice_alpha": 90,
        "lattice_beta": 90,
        "lattice_gamma": 105,
        "lattice_rotation": 28,
    }
    assert window._lattice_parameters() == expected
    assert window.inputs["lattice_a"].value() == 4.7
    assert window.inputs["lattice_b"].value() == 6.2
    assert window.inputs["lattice_gamma"].value() == 105
    assert window.inputs["lattice_rotation"].value() == 28
    saved = window.workspace_state_payload()
    options.ktool.bz.default_a = 8
    erlab.interactive.options.model = options
    restored = AngleCalculatorWindow()
    qtbot.addWidget(restored)
    assert restored.inputs["lattice_a"].value() == 8
    restored.restore_workspace_state(saved)
    assert restored._lattice_parameters() == expected


def test_angle_calculator_initial_defaults(angle_calculator):
    assert angle_calculator._state.geometry == calc.Geometry(
        configuration=3,
        analyzer_work_function=4.3,
        photon_energy=90.0,
    )
    assert angle_calculator.da_check.isChecked()
    assert angle_calculator.slit_combo.currentIndex() == 0
    for widget in (
        angle_calculator.inputs["sample_work_function"],
        angle_calculator.cutoff_check,
        angle_calculator.correction_check,
        angle_calculator._laser_arpes_summary,
    ):
        assert not angle_calculator.geometry_layout.isRowVisible(widget)


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
@pytest.mark.parametrize("photon_energy", [6.0, 21.2, 90.0])
@pytest.mark.parametrize("plot_index", [0, 1])
def test_angle_calculator_initial_plot_range(
    qtbot, configuration, photon_energy, plot_index
):
    options = erlab.interactive.options.model.model_copy(deep=True)
    options.angle_calculator.configuration = configuration
    options.angle_calculator.photon_energy = photon_energy
    window = AngleCalculatorWindow(options=options)
    qtbot.addWidget(window)
    with qtbot.waitExposed(window):
        window.show()
    qtbot.wait(30)
    plot = (window.momentum_plot, window.angle_plot)[plot_index]
    points = np.array(window.scan_curves[plot_index].getData())
    limits = np.array(plot.viewRange())
    assert np.isfinite(points).all()
    assert np.all(limits[:, 0] < points.min(axis=1))
    assert np.all(limits[:, 1] > points.max(axis=1))
    # One axis can expand to keep equal scales in a rectangular view.
    assert np.min(np.ptp(limits, axis=1) / np.ptp(points, axis=1)) < 1.5
    window.inputs["photon_energy"].setValue(40.0)
    window.alpha_control.setValues((-5, 5))
    window.scan_control.setValues((-10, 10))
    window._redraw()
    np.testing.assert_allclose(plot.viewRange(), limits)


def test_angle_calculator_initial_momentum_range_without_emission(qtbot):
    options = erlab.interactive.options.model.model_copy(deep=True)
    options.angle_calculator.photon_energy = 0.001
    window = AngleCalculatorWindow(options=options)
    qtbot.addWidget(window)
    limits = np.array(window.momentum_plot.viewRange())
    assert np.isfinite(limits).all()
    assert np.all(np.ptp(limits, axis=1) > 0)


@pytest.mark.parametrize("plot_name", ["momentum", "angle"])
def test_angle_calculator_restores_manual_plot_range(
    angle_calculator, qtbot, plot_name
):
    getattr(angle_calculator, f"{plot_name}_plot").setRange(
        xRange=(2, 3), yRange=(-1, 1), padding=0
    )
    saved = angle_calculator.workspace_state_payload()
    restored = AngleCalculatorWindow()
    qtbot.addWidget(restored)
    restored.restore_workspace_state(saved)
    # The visible range can expand when axis labels change the view's aspect ratio.
    np.testing.assert_allclose(
        getattr(restored, f"{plot_name}_plot").getViewBox().targetRange(),
        saved[f"{plot_name}_range"],
    )


def test_angle_calculator_restores_geometry_over_user_defaults(angle_calculator, qtbot):
    angle_calculator.slit_combo.setCurrentIndex(1)
    angle_calculator.da_check.setChecked(False)
    angle_calculator.inputs["analyzer_work_function"].setValue(4.6)
    angle_calculator.inputs["photon_energy"].setValue(40)
    saved = angle_calculator.workspace_state_payload()
    restored = AngleCalculatorWindow()
    qtbot.addWidget(restored)
    restored.restore_workspace_state(saved)
    assert restored._state.geometry == angle_calculator._state.geometry
    assert restored.slit_combo.currentIndex() == 1
    assert not restored.da_check.isChecked()
    for name in ("analyzer_work_function", "photon_energy"):
        assert restored.inputs[name].value() == saved["geometry"][name]


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
def test_angle_calculator_restore_disables_incompatible_correction(
    angle_calculator, configuration
):
    window = angle_calculator
    payload = window.workspace_state_payload()
    payload["geometry"].update(
        configuration=configuration,
        photon_energy=6.0,
        binding_energy=-0.2,
        analyzer_work_function=4.6,
        sample_work_function=5.99,
        low_energy_correction=True,
        polar=13.0,
        tilt=7.0,
        azimuth=17.0,
    )
    payload["cutoff_visible"] = True
    window.restore_workspace_state(payload)
    restored = window.workspace_state_payload()
    assert restored["geometry"]["sample_work_function"] == 5.99
    assert not restored["geometry"]["low_energy_correction"]
    assert restored["cutoff_visible"]
    assert restored["geometry"]["polar"] == 13.0
    assert restored["geometry"]["tilt"] == 7.0
    params = {"delta": 17.0}
    if configuration in (1, 2):
        params["xi"] = 7.0 if configuration == 1 else 13.0
    else:
        params.update(chi=13.0, xi=7.0)
    alpha, beta = window.scan_curves[1].getData()
    expected = erlab.analysis.kspace.get_kconv_forward(configuration)(
        alpha, beta, 1.2, **params
    )
    np.testing.assert_allclose(window.scan_curves[0].getData(), expected, atol=1e-14)
    np.testing.assert_allclose(
        window.overlays._display_to_momentum(alpha, beta), expected, atol=1e-14
    )
    cut_alpha = np.linspace(*window._state.alpha_range, 181)
    expected_cut = erlab.analysis.kspace.get_kconv_forward(configuration)(
        cut_alpha, window.current_spin.value(), 1.2, **params
    )
    np.testing.assert_allclose(window.cut_curve.getData(), expected_cut, atol=1e-14)


@pytest.mark.parametrize("configuration", [1, 2, 3, 4])
def test_angle_calculator_laser_correction_gate_and_history(
    angle_calculator, configuration, image_files
):
    win = angle_calculator
    win.slit_combo.setCurrentIndex((configuration - 1) % 2)
    win.da_check.setChecked(configuration > 2)
    if configuration in (1, 2):
        win.scan_control.setValues((0, 0))
        # The stored, inactive mechanical angle does not set the current geometry.
        win.inputs["polar" if configuration == 1 else "tilt"].setValue(12)
    win.inputs["photon_energy"].setValue(6)
    win.inputs["sample_work_function"].setValue(5.9)
    win.inputs["analyzer_work_function"].setValue(4.65)
    win.overlays.load_image(str(image_files[0]))
    win._redraw()
    standard = np.array(win.cut_curve.getData())
    win._flush_pending_history_write()
    assert win.correction_check.isEnabled()
    win.correction_check.setChecked(True)
    win._redraw()
    corrected = np.array(win.cut_curve.getData())
    assert not np.allclose(standard, corrected, equal_nan=True)
    assert win.overlays.projected_image.isVisible() is (configuration > 2)
    assert not win._laser_arpes_summary.isHidden()
    win.undo()
    assert not win.correction_check.isChecked()
    assert win._laser_arpes_summary.isHidden()
    np.testing.assert_allclose(win.cut_curve.getData(), standard)
    win.redo()
    np.testing.assert_allclose(win.cut_curve.getData(), corrected)
    fixed_angle = "tilt" if configuration == 1 else "polar"
    win.inputs[fixed_angle].setValue(1)
    assert not win.correction_check.isEnabled()
    assert not win.correction_check.isChecked()
    assert not win._state.geometry.low_energy_correction
    win.undo()
    assert win.correction_check.isEnabled()
    assert win.correction_check.isChecked()
    if configuration in (1, 2):
        win.scan_control.setValues((0, 1))
        assert not win.correction_check.isEnabled()
        assert not win.correction_check.isChecked()


@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize("work_functions", [(5.9, 4.65), (4.65, 5.9), (4.65, 4.65)])
def test_angle_calculator_sample_escape_limit(
    angle_calculator, correction, work_functions
):
    win = angle_calculator
    win.inputs["photon_energy"].setValue(6)
    win.inputs["sample_work_function"].setValue(work_functions[0])
    win.inputs["analyzer_work_function"].setValue(work_functions[1])
    win.correction_check.setChecked(correction)
    win.cutoff_check.setChecked(True)
    win._redraw()
    sample_energy, analyzer_energy = 6 - np.array(work_functions)
    momenta = np.array(win.cutoff_curves[0].getData())
    angles = np.array(win.cutoff_curves[1].getData())
    np.testing.assert_allclose(
        np.hypot(*momenta), erlab.constants.rel_kconv * np.sqrt(sample_energy)
    )
    if sample_energy > analyzer_energy:
        assert np.isnan(angles).all()
    else:
        expected = np.rad2deg(
            np.arctan2(
                2 * np.sqrt(sample_energy), np.sqrt(analyzer_energy - sample_energy)
            )
            if correction
            else np.arcsin(np.sqrt(sample_energy / analyzer_energy))
        )
        np.testing.assert_allclose(np.hypot(*angles), expected, atol=2e-5)
    win.cutoff_check.setChecked(False)
    win._redraw()
    assert all(not curve.isVisible() for curve in win.cutoff_curves)


@pytest.mark.parametrize("work_function", [6.0, 7.0])
def test_angle_calculator_clears_sample_escape_limit(angle_calculator, work_function):
    win = angle_calculator
    win.inputs["photon_energy"].setValue(6)
    win.cutoff_check.setChecked(True)
    win._redraw()
    assert all(curve.getData()[0] is not None for curve in win.cutoff_curves)
    win.inputs["sample_work_function"].setValue(work_function)
    win._redraw()
    assert all(curve.getData() == (None, None) for curve in win.cutoff_curves)
    assert np.isfinite(win.scan_curves[0].getData()).all()


@pytest.mark.parametrize("work_function", [6.0, 7.0])
def test_angle_calculator_nonpositive_kinetic_energy(angle_calculator, work_function):
    window = angle_calculator
    window.inputs["photon_energy"].setValue(6.0)
    window.inputs["analyzer_work_function"].setValue(work_function)
    window._redraw()
    assert np.isfinite(window.scan_curves[1].getData()).all()
    assert np.isnan(window.scan_curves[0].getData()).all()
    assert np.isnan(window.cut_curve.getData()).all()
    assert np.isfinite(window.momentum_plot.viewRange()).all()


def test_angle_calculator_image_rotation(angle_calculator, tmp_path):
    source = QtGui.QImage(3, 2, QtGui.QImage.Format.Format_RGBA8888)
    for row in range(2):
        for col in range(3):
            source.setPixelColor(
                col, row, QtGui.QColor(40 + 80 * col, 20 + 180 * row, 0)
            )
    path = tmp_path / "asymmetric.png"
    assert source.save(str(path))
    angle_calculator.overlays.load_image(str(path))
    angle_calculator.overlays._draw_image(64)
    original = angle_calculator.overlays.projected_image.image.copy()
    angle_calculator.overlays.inputs["image_x"].setValue(0.03)
    angle_calculator.overlays.inputs["image_y"].setValue(-0.02)
    angle_calculator.overlays.inputs["image_rotation"].setValue(37)
    angle_calculator.overlays._draw_image(64)
    assert not np.array_equal(original, angle_calculator.overlays.projected_image.image)
    center = angle_calculator.overlays.momentum_image.mapToParent(
        QtCore.QPointF(1.5, 1)
    )
    assert (center.x(), center.y()) == pytest.approx((0.03, -0.02))
    angle = np.deg2rad(37)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    corner = angle_calculator.overlays.momentum_image.mapToParent(QtCore.QPointF(0, 0))
    expected_corner = np.array([0.03, -0.02]) + rotation @ np.array([-0.15, -0.1])
    np.testing.assert_allclose((corner.x(), corner.y()), expected_corner)
    assert angle_calculator.overlays._image_roi.geometry()[0] == pytest.approx(
        (0.03, -0.02)
    )
    assert angle_calculator.overlays._image_roi.geometry()[2] == 37
    angle_calculator.overlays._image_roi.setGeometry((0.03, -0.02), 0.3, rotation=63)
    assert angle_calculator.overlays.state.image.rotation == 63
    assert angle_calculator.overlays.inputs["image_rotation"].value() == 63
    angle_calculator.overlays._draw_image(64)
    grid = next(
        grid
        for key, grid in angle_calculator.overlays._grid_cache.items()
        if key[1] == 64
    )
    expected_image = _geometry.warp_image(
        angle_calculator.overlays._source_image, *grid, (0.03, -0.02), 0.3, rotation=63
    )
    np.testing.assert_array_equal(
        angle_calculator.overlays.projected_image.image, expected_image
    )


def test_angle_calculator_deflector_limits(angle_calculator):
    window = angle_calculator
    window.scan_control.setValues((-90, 90))
    assert window.scan_control.values() == (-45, 45)
    window.current_spin.setValue(90)
    assert window.current_spin.value() == 45
    window.da_check.setChecked(False)
    window.scan_control.setValues((-70, 65))
    window.current_spin.setValue(60)
    window.da_check.setChecked(True)
    assert window.scan_control.values() == (-45, 45)
    assert window.current_spin.value() == 45
    window.da_check.setChecked(False)
    assert window.scan_control.values() == (-70, 65)
    assert window.current_spin.value() == 60
    window.restore_workspace_state(window.workspace_state_payload())
    window.da_check.setChecked(True)
    assert window.scan_control.values() == (-45, 45)


@pytest.mark.parametrize("alpha_range", [(-60, 60), (-10, 5), (5, 55), (-55, -5)])
def test_angle_calculator_slit_marker_tracks_view(angle_calculator, qtbot, alpha_range):
    window = angle_calculator
    window.show()
    qtbot.waitExposed(window)
    window.current_spin.setValue(5)
    window._redraw()
    window.cut_marker.boundingRect()
    view = window.angle_plot.getViewBox()
    view.setRange(xRange=alpha_range, yRange=(-30, 30), padding=0)
    # A paint can cache the line's bounds before the deferred plot update.
    window.cut_marker.boundingRect()
    for redraw in (False, True):
        if redraw:
            window._redraw()
        bounds = window.cut_marker.mapRectToParent(window.cut_marker.boundingRect())
        expected = np.clip(window._state.alpha_range, *view.viewRange()[0])
        np.testing.assert_allclose((bounds.left(), bounds.right()), expected)
        assert bounds.center().y() == pytest.approx(5)
    window.alpha_control.setValues((-8, 12))
    window._redraw()
    bounds = window.cut_marker.mapRectToParent(window.cut_marker.boundingRect())
    expected = np.clip((-8, 12), *view.viewRange()[0])
    np.testing.assert_allclose((bounds.left(), bounds.right()), expected)


def test_angle_calculator_compact_scan_modes(angle_calculator, qtbot):
    window = angle_calculator
    window.show()
    qtbot.waitExposed(window)
    window.inputs["polar"].setValue(7.12)
    window.inputs["tilt"].setValue(-9.34)
    for configuration, scanned_angle in (
        (1, "polar"),
        (2, "tilt"),
        (3, None),
        (4, None),
    ):
        window.da_check.setChecked(configuration > 2)
        window.slit_combo.setCurrentIndex((configuration - 1) % 2)
        assert window.scan_control.isVisible()
        assert window.scan_label.isVisible() == (scanned_angle is None)
        for angle in ("polar", "tilt"):
            assert window.inputs[angle].isVisible() == (angle != scanned_angle)
        window.scan_control.lower_spin.setValue(-12.34)
        assert window.scan_control.slider.values()[0] == -12.34
    assert window.inputs["polar"].value() == 7.12
    assert window.inputs["tilt"].value() == -9.34
    for spin in window.inputs.values():
        if spin.suffix() == "°":
            assert spin.decimals() == 2


@pytest.mark.parametrize(
    "edit",
    [
        lambda win: win.inputs["photon_energy"].setValue(40),
        lambda win: win.inputs["binding_energy"].setValue(-0.4),
        lambda win: win.inputs["azimuth"].setValue(12),
        lambda win: win.da_check.toggle(),
        lambda win: win.slit_combo.setCurrentIndex(1),
        lambda win: win.alpha_control.setValues((-9, 11)),
        lambda win: win.scan_control.setValues((-8, 10)),
        lambda win: win.cut_marker.setValue(4),
        lambda win: win.inputs["lattice_a"].setValue(5),
        lambda win: win.inputs["lattice_c"].setValue(7),
        lambda win: win.inputs["lattice_alpha"].setValue(85),
        lambda win: win.inputs["lattice_beta"].setValue(95),
        lambda win: win.inputs["inner_potential"].setValue(15),
        lambda win: win.centering_combo.setCurrentIndex(1),
        lambda win: win.out_of_plane_check.toggle(),
        lambda win: win.photon_energy_control.setValues((30, 100)),
        lambda win: win.inputs["lattice_rotation"].setValue(17),
        lambda win: win.bz_check.toggle(),
    ],
)
def test_angle_calculator_settings_history(angle_calculator, edit, qtbot):
    win = angle_calculator
    initial = win._history_state
    edit(win)
    changed = win._history_state
    assert changed != initial
    qtbot.wait_until(win.undo_action.isEnabled)
    assert win.undoable
    with qtbot.waitSignal(win.sigStateChanged):
        win.undo_action.trigger()
    assert win._history_state == initial
    assert win.redo_action.isEnabled()
    win.redo_action.trigger()
    assert win._history_state == changed
    assert not win.redoable


def test_angle_calculator_overlay_history(angle_calculator, image_files):
    win = angle_calculator
    editor = win.overlays
    snapshots = [win._history_state]
    for edit in (
        editor.add_button.click,
        lambda: editor.inputs["pocket_x"].setValue(0.2),
        lambda: editor.load_image(str(image_files[0])),
        lambda: editor.image_invert.setChecked(True),
        lambda: editor.image_black_transparent.setChecked(True),
        lambda: editor.inputs["image_black_threshold"].setValue(20),
        lambda: editor.load_image(str(image_files[1])),
        editor.delete_button.click,
    ):
        edit()
        assert win._flush_pending_history_write()
        snapshots.append(win._history_state)
    for path in image_files:
        path.unlink()
    for expected in reversed(snapshots[:-1]):
        win.undo()
        assert win._history_state == expected
    for expected in snapshots[1:]:
        win.redo()
        assert win._history_state == expected
    win.undo()
    editor.inputs["pocket_y"].setValue(0.3)
    win._flush_pending_history_write()
    assert not win.redoable


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize(
    "control", ["geometry", "range", "photon_range", "lattice", "overlay"]
)
def test_angle_calculator_numeric_history_shortcuts(
    angle_calculator, qtbot, commit, control
):
    from qtpy import QtTest

    win = angle_calculator
    if control == "photon_range":
        win.out_of_plane_check.setChecked(True)
    win.overlays.add_button.click()
    win._flush_pending_history_write()
    spin, value, tab_index = {
        "geometry": (win.inputs["photon_energy"], "60.123", 0),
        "range": (win.scan_control.upper_spin, "10", 0),
        "photon_range": (win.photon_energy_control.lower_spin, "60.123", 0),
        "lattice": (win.inputs["lattice_a"], "5", 1),
        "overlay": (win.overlays.inputs["pocket_x"], "0.2", 2),
    }[control]
    win.tabs.setCurrentIndex(tab_index)
    win.show()
    qtbot.waitExposed(win)
    win.activateWindow()
    # QTest sends keys directly. Set their Qt shortcut context without waiting
    # for native focus, which other xdist workers can hold on the same display.
    QtWidgets.QApplication.setActiveWindow(win)
    initial = win._history_state
    spin.setFocus()
    qtbot.wait_until(spin.hasFocus)
    spin.selectAll()
    qtbot.keyClicks(spin, value)
    if commit:
        qtbot.keyClick(spin, QtCore.Qt.Key.Key_Return)
        qtbot.wait_until(lambda: not win._history_write_pending)
    QtTest.QTest.keySequence(spin, win.undo_action.shortcut())
    assert win._history_state == initial
    QtTest.QTest.keySequence(spin, win.redo_action.shortcut())
    assert spin.value() == float(value)


@pytest.mark.parametrize("pending", [False, True])
def test_angle_calculator_history_preserves_view_and_workspace_baseline(
    angle_calculator, qtbot, pending
):
    win = angle_calculator
    win.show()
    qtbot.wait(50)
    initial = win._history_state
    win.inputs["photon_energy"].setValue(40)
    # Select the history state before processing layout and paint events. Their
    # duration must not decide whether the 150 ms history timer has fired.
    win._history_write_timer.stop()
    if not pending:
        win._flush_pending_history_write()
    win.momentum_plot.setRange(xRange=(-1, 1), yRange=(-2, 2), padding=0)
    win.angle_plot.setRange(xRange=(-10, 10), yRange=(-20, 20), padding=0)
    win.out_of_plane_plot.setRange(xRange=(-12, 15), yRange=(2, 6), padding=0)
    win.resize(1200, 740)
    qtbot.wait(50)
    workspace = win.workspace_state_payload()
    assert win._history_write_pending is pending
    win.undo()
    assert win._history_state == initial
    after = win.workspace_state_payload()
    for name in ("momentum_range", "angle_range", "out_of_plane_range", "window_state"):
        assert after[name] == workspace[name]
    assert win.redoable
    win.restore_workspace_state(workspace)
    assert win.inputs["photon_energy"].value() == 40
    assert not win.undoable
    assert not win.redoable
    assert not win._history_write_timer.isActive()
    win.momentum_plot.setRange(xRange=(-0.5, 0.5), padding=0)
    win.resize(1300, 760)
    win.workspace_state_payload()
    qtbot.wait(200)
    assert not win.undoable
    assert not win._history_write_pending


def test_angle_calculator_pending_history_close_and_reopen(angle_calculator, qtbot):
    win = angle_calculator
    win.show()
    initial = win._history_state
    win.inputs["photon_energy"].setValue(40)
    assert win._history_write_pending
    win.close()
    assert not win._history_write_timer.isActive()
    assert win.undoable
    win.show()
    qtbot.wait(200)
    assert len(win._prev_states) == 2
    win.undo()
    assert win._history_state == initial


@pytest.mark.parametrize("shortcut", [False, True])
def test_angle_calculator_close_commits_focused_numeric_edit(
    angle_calculator, qtbot, shortcut
):
    win = angle_calculator
    win.show()
    win.activateWindow()
    spin = win.inputs["photon_energy"]
    spin.setValue(91.0)
    win._flush_pending_history_write()
    initial = win._history_state
    spin.setFocus()
    qtbot.wait_until(spin.hasFocus)
    spin.selectAll()
    qtbot.keyClicks(spin, "21.2")
    assert win._history_state == initial
    if shortcut:
        qtbot.keySequence(spin, QtGui.QKeySequence("Ctrl+W"))
    else:
        assert win.close()
    assert not win.isVisible()
    expected = initial.model_copy(
        update={"geometry": dataclasses.replace(initial.geometry, photon_energy=21.2)}
    )
    assert win._history_state == expected
    assert len(win._prev_states) == 3
    assert not win._history_write_pending
    assert not win._history_write_timer.isActive()
    assert not win._redraw_timer.isActive()
    win.show()
    qtbot.wait(200)
    assert len(win._prev_states) == 3
    win.undo()
    assert win._history_state == initial
    win.redo()
    assert win._history_state == expected


def test_angle_calculator_overlay_history_does_not_recalculate_coverage(
    angle_calculator, qtbot, monkeypatch
):
    win = angle_calculator
    win.show()
    qtbot.waitExposed(win)
    views = tuple(
        plot.getViewBox()
        for plot in (win.momentum_plot, win.angle_plot, win.out_of_plane_plot)
    )
    # Range changes redraw coverage independently. Measure only overlay edits.
    for view in views:
        view.sigRangeChanged.disconnect(win._view_changed)
    try:
        win._redraw_timer.stop()
        calls = []
        for curve in (*win.scan_curves, win.cut_curve):
            monkeypatch.setattr(
                curve, "setData", lambda *args, **kwargs: calls.append(None)
            )
        win.overlays.add_button.click()
        win.overlays.inputs["pocket_x"].setValue(0.2)
        qtbot.wait_until(win.undo_action.isEnabled)
        qtbot.wait_until(lambda: not win._pocket_redraw_timer.isActive())
        assert calls == []
        assert win._history_state.overlays.pockets[0].center[0] == 0.2
    finally:
        for view in views:
            view.sigRangeChanged.connect(win._view_changed)


def test_angle_calculator_history_preserves_pocket_selection(angle_calculator):
    win = angle_calculator
    for _ in range(2):
        win.overlays.add_button.click()
        win._flush_pending_history_write()
    win.overlays.pocket_list.setCurrentRow(0)
    uid = win.overlays._selected_pocket
    win.overlays.inputs["pocket_x"].setValue(0.1)
    win._flush_pending_history_write()
    for action, x in ((win.undo_action, 0.0), (win.redo_action, 0.1)):
        action.trigger()
        assert win.overlays._selected_pocket == uid
        assert win.overlays.inputs["pocket_x"].value() == x
    win.undo()
    win.overlays.inputs["pocket_x"].setValue(0.2)
    assert [pocket.center for pocket in win.overlays.state.pockets] == [
        (0.2, 0.0),
        (0.0, 0.0),
    ]


@pytest.mark.parametrize(
    "parameters",
    [
        (5.0, 5.00001, 6.01, 90.0, 90.0, 0.115),
        (5.0, 5.00001, 6.01, 90.0, 90.0, 179.885),
        (0.005, 0.009, 6.01, 80.0, 95.0, 110.0),
        (1000.0, 1000.0, 6.01, 90.0, 90.0, 90.0),
        (
            4.123456789,
            5.234567891,
            7.345678912,
            80.123456789,
            95.234567891,
            110.345678912,
        ),
    ],
)
def test_angle_calculator_rotation_preserves_lattice_geometry(
    angle_calculator, parameters
):
    win = angle_calculator
    fields = dict(
        zip(
            (f"lattice_{name}" for name in ("a", "b", "c", "alpha", "beta", "gamma")),
            parameters,
            strict=True,
        )
    )
    win.restore_workspace_state(AngleCalculatorState(**fields, bz_visible=False))
    before = erlab.lattice.abc2avec(*parameters)
    initial = win._history_state
    win._flush_pending_history_write()
    for name, value in fields.items():
        spin = win.inputs[name]
        assert spin.minimum() <= value <= spin.maximum()
        assert spin.value() == pytest.approx(value, abs=0.5 * 10 ** -spin.decimals())
    for target in (17.0, -63.0, 121.0):
        win.inputs["lattice_rotation"].setValue(target)
        assert win._lattice_parameters() == {**fields, "lattice_rotation": target}
        theta = np.deg2rad(target)
        rotation = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        np.testing.assert_allclose(
            win._reciprocal_basis(),
            2 * np.pi * np.linalg.inv(before).T @ rotation,
            rtol=1e-12,
            atol=1e-12,
        )
    final = win._history_state
    win.undo()
    assert win._history_state == initial
    win.redo()
    assert win._history_state == final
    win.restore_workspace_state(win.workspace_state_payload())
    assert win._history_state == final


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lattice_a", 6.0),
        ("lattice_b", 7.0),
        ("lattice_c", 8.0),
        ("lattice_alpha", 80.0),
        ("lattice_beta", 100.0),
        ("lattice_gamma", 23.0),
    ],
)
def test_angle_calculator_parameter_edits_preserve_other_parameters(
    angle_calculator, field, value
):
    win = angle_calculator
    win.restore_workspace_state(
        AngleCalculatorState(
            lattice_a=4.123456789,
            lattice_b=5.234567891,
            lattice_c=7.345678912,
            lattice_alpha=80.123456789,
            lattice_beta=95.234567891,
            lattice_gamma=110.345678912,
            lattice_rotation=17.123456789,
            bz_visible=False,
        )
    )
    before = win._lattice_parameters()
    win.inputs[field].setValue(value)
    after = win._lattice_parameters()
    assert after == {**before, field: value}
    final = win._history_state
    win.undo()
    assert win._lattice_parameters() == before
    win.redo()
    assert win._history_state == final
    win.restore_workspace_state(win.workspace_state_payload())
    assert win._lattice_parameters() == after


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lattice_a", 0.0),
        ("lattice_b", 0.0),
        ("lattice_gamma", 0.0),
        ("lattice_gamma", 180.0),
    ],
)
def test_angle_calculator_rejects_degenerate_lattice_parameters(
    angle_calculator, field, value
):
    win = angle_calculator
    initial = win._history_state
    win.inputs[field].setValue(value)
    assert win._history_state == initial
    assert not win._history_write_pending
    assert win.lattice_status.text()
    win.inputs[field].setValue(
        {"lattice_a": 6.0, "lattice_b": 7.0, "lattice_gamma": 90.0}[field]
    )
    assert win._history_state != initial
    assert not win.lattice_status.text()


def test_angle_calculator_full_cell_defaults(qtbot):
    from erlab.interactive._options.schema import AppOptions

    parameters = {
        "a": 4.0,
        "b": 5.0,
        "c": 7.0,
        "alpha": 80.0,
        "beta": 95.0,
        "gamma": 110.0,
    }
    options = AppOptions.model_validate(
        {
            "ktool": {
                "bz": {
                    **{f"default_{key}": value for key, value in parameters.items()},
                    "default_rot": 23.0,
                    "default_centering": "F",
                }
            }
        }
    )
    window = AngleCalculatorWindow(options=options)
    qtbot.addWidget(window)
    for name, value in parameters.items():
        assert getattr(window._state, f"lattice_{name}") == value
    for name, value in parameters.items():
        assert window.inputs[f"lattice_{name}"].value() == pytest.approx(value)
    assert window._state.centering == "F"
    assert window.centering_combo.currentText() == "F"
    assert window._state.lattice_rotation == 23.0


def test_angle_calculator_full_cell_workspace(angle_calculator):
    state = AngleCalculatorState(
        lattice_a=4.123456789,
        lattice_b=5.234567891,
        lattice_c=7.345678912,
        lattice_alpha=80.123456789,
        lattice_beta=95.234567891,
        lattice_gamma=110.345678912,
        lattice_rotation=27,
        centering="I",
        geometry=calc.Geometry(inner_potential=15),
        hv_range=(35, 95),
        out_of_plane_visible=True,
        out_of_plane_range=((-12, 18), (2, 6)),
    )
    angle_calculator.restore_workspace_state(state)
    restored = AngleCalculatorState.model_validate(
        angle_calculator.workspace_state_payload()
    )
    for field in (
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "lattice_alpha",
        "lattice_beta",
        "lattice_gamma",
        "lattice_rotation",
        "centering",
        "geometry",
        "hv_range",
        "out_of_plane_visible",
    ):
        assert getattr(restored, field) == getattr(state, field)
    np.testing.assert_allclose(restored.out_of_plane_range, state.out_of_plane_range)
    assert angle_calculator.photon_energy_control.values() == (35, 95)
    assert angle_calculator.out_of_plane_check.isChecked()


def test_angle_calculator_out_of_plane_controls(angle_calculator, qtbot):
    window = angle_calculator
    window.show()
    qtbot.waitExposed(window)
    assert not window.out_of_plane_plot.isVisible()
    window.out_of_plane_check.setChecked(True)
    window._redraw()
    assert window.out_of_plane_plot.isVisible()
    window.current_spin.setValue(9)
    window.inputs["inner_potential"].setValue(17)
    window.photon_energy_control.setValues((30, 95))
    window._redraw()
    alpha, kz = window.out_of_plane_current_curve.getData()
    assert np.isfinite(kz).all()
    expected = calc.angles_to_momentum3d(alpha, 9, window._state.geometry)[2]
    np.testing.assert_allclose(kz, expected)
    assert np.ptp(alpha) > 0
    scan_alpha, scan_kz = window.out_of_plane_scan_curve.getData()
    assert np.isfinite(scan_kz).any()
    lower = calc.angles_to_momentum3d(scan_alpha, 9, window._state.geometry, 30)[2]
    upper = calc.angles_to_momentum3d(scan_alpha, 9, window._state.geometry, 95)[2]
    valid = np.isfinite(scan_kz)
    assert np.all(scan_kz[valid] >= lower[valid] - 1e-12)
    assert np.all(scan_kz[valid] <= upper[valid] + 1e-12)
    assert window._state.hv_range == (30, 95)
    window.inputs["inner_potential"].setValue(22)
    window._redraw()
    np.testing.assert_allclose(
        window.out_of_plane_current_curve.getData()[1] ** 2 - kz**2,
        erlab.constants.rel_kconv**2 * 5,
    )
    previous_kz = window.out_of_plane_current_curve.getData()[1].copy()
    window.current_spin.setValue(0)
    window._redraw()
    assert not np.allclose(window.out_of_plane_current_curve.getData()[1], previous_kz)
    window.out_of_plane_check.setChecked(False)
    assert not window.out_of_plane_plot.isVisible()


def test_angle_calculator_uses_shared_surface_bz(angle_calculator, monkeypatch):
    calls = []
    original = erlab.lattice.get_surface_bz

    def surface_bz(basis, x, y, surface, **kwargs):
        calls.append((np.asarray(basis), np.asarray(surface)))
        return original(basis, x, y, surface, **kwargs)

    monkeypatch.setattr(erlab.lattice, "get_surface_bz", surface_bz)
    window = angle_calculator
    window.centering_combo.setCurrentText("F")
    window.out_of_plane_check.setChecked(True)
    window._redraw()
    assert len(calls) >= 2
    cell = erlab.lattice.abc2avec(
        *(
            getattr(window._state, f"lattice_{name}")
            for name in ("a", "b", "c", "alpha", "beta", "gamma")
        )
    )
    primitive = erlab.lattice.to_primitive(cell, "F")
    expected = erlab.lattice.to_reciprocal(primitive)
    for basis, surface in calls:
        np.testing.assert_allclose(basis, expected, atol=1e-12)
        assert surface.shape[-1] == 3
        assert np.isfinite(surface[..., 2]).any()


def test_angle_calculator_collapsed_photon_range_history(angle_calculator):
    window = angle_calculator
    window.out_of_plane_check.setChecked(True)
    window._flush_pending_history_write()
    previous = window._state.hv_range
    window.photon_energy_control.setValues((45, 45))
    window._redraw()
    alpha, kz = window.out_of_plane_scan_curve.getData()
    np.testing.assert_allclose(
        kz, calc.angles_to_momentum3d(alpha, 0, window._state.geometry, 45)[2]
    )
    payload = window.workspace_state_payload()
    window.undo()
    assert window._state.hv_range == previous
    window.redo()
    assert window._state.hv_range == (45, 45)
    window.restore_workspace_state(payload)
    assert window.photon_energy_control.values() == (45, 45)


def test_angle_calculator_out_of_plane_hide_preserves_range(angle_calculator, qtbot):
    window = angle_calculator
    window.out_of_plane_check.setChecked(True)
    window.out_of_plane_plot.setRange(xRange=(-12, 18), yRange=(2, 6), padding=0)
    before = window.out_of_plane_plot.viewRange()
    window.out_of_plane_check.setChecked(False)
    window.out_of_plane_check.setChecked(True)
    np.testing.assert_allclose(window.out_of_plane_plot.viewRange(), before)
    window.out_of_plane_check.setChecked(False)
    restored = AngleCalculatorWindow()
    qtbot.addWidget(restored)
    restored.restore_workspace_state(window.workspace_state_payload())
    restored.out_of_plane_check.setChecked(True)
    np.testing.assert_allclose(restored.out_of_plane_plot.viewRange(), before)


@pytest.mark.parametrize("control", ["slider", "spinbox"])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_angle_calculator_photon_range_preserves_view(
    angle_calculator, qtbot, control, endpoint
):
    window = angle_calculator
    window.show()
    qtbot.waitExposed(window)
    window.out_of_plane_check.setChecked(True)
    window.photon_energy_control.setValues((20, 80))
    window.out_of_plane_plot.setRange(xRange=(-8, 12), yRange=(3, 4), padding=0)
    window._flush_pending_history_changes()
    before = np.array(window.out_of_plane_plot.viewRange())
    initial = window.photon_energy_control.values()

    if control == "slider":
        slider = window.photon_energy_control.slider
        start = slider._handle_rect(endpoint).center()
        end = start + QtCore.QPoint(-10 if endpoint == 0 else 10, 0)
        qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
        QtWidgets.QApplication.sendEvent(
            slider,
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                QtCore.QPointF(end),
                QtCore.QPointF(slider.mapToGlobal(end)),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            ),
        )
        qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    else:
        spin = (
            window.photon_energy_control.lower_spin,
            window.photon_energy_control.upper_spin,
        )[endpoint]
        qtbot.keyClick(spin, QtCore.Qt.Key.Key_Up)

    qtbot.waitUntil(lambda: not window._redraw_timer.isActive())
    updated = window.photon_energy_control.values()
    assert updated[endpoint] != initial[endpoint]
    assert updated[1 - endpoint] == initial[1 - endpoint]
    assert window._state.hv_range == updated
    np.testing.assert_allclose(window.out_of_plane_plot.viewRange(), before)
    for curve, energy in zip(window._hv_edge_curves, updated, strict=True):
        alpha, kz = curve.getData()
        np.testing.assert_allclose(
            kz, calc.angles_to_momentum3d(alpha, 0, window._state.geometry, energy)[2]
        )
    window._flush_pending_history_changes()
    window.undo()
    assert window.photon_energy_control.values() == initial
    np.testing.assert_allclose(window.out_of_plane_plot.viewRange(), before)
    window.redo()
    assert window.photon_energy_control.values() == updated
    np.testing.assert_allclose(window.out_of_plane_plot.viewRange(), before)


def test_angle_calculator_dense_cell_warning(angle_calculator, monkeypatch):
    window = angle_calculator
    calls = []
    original = erlab.lattice.get_surface_bz

    def surface_bz(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(erlab.lattice, "get_surface_bz", surface_bz)
    window.restore_workspace_state(
        AngleCalculatorState(
            lattice_a=1000,
            lattice_b=1000,
            lattice_c=1000,
            lattice_gamma=90,
            momentum_range=((-2, 2), (-2, 2)),
        )
    )
    assert calls == []
    assert window.bz_status.text()
    assert window.bz_curves[0].getData()[0] is None
    window.restore_workspace_state(
        AngleCalculatorState(momentum_range=((-1, 1), (-1, 1)))
    )
    assert calls
    assert not window.bz_status.text()


def test_angle_calculator_grazing_out_of_plane_surface(angle_calculator, monkeypatch):
    window = angle_calculator
    surfaces = []
    original = erlab.lattice.get_surface_bz

    def surface_bz(basis, x, y, surface, **kwargs):
        surfaces.append(np.asarray(surface).copy())
        return original(basis, x, y, surface, **kwargs)

    window.out_of_plane_check.setChecked(True)
    window._redraw()
    monkeypatch.setattr(erlab.lattice, "get_surface_bz", surface_bz)
    window.out_of_plane_plot.setRange(xRange=(-90, 90), yRange=(1, 7), padding=0)
    window._redraw()
    assert len(surfaces) == 1
    surface = surfaces[0]
    assert np.isnan(surface[:, (0, -1)]).all()
    assert np.nanmax(np.linalg.norm(surface, axis=-1)) < 7
    assert not window.bz_status.text()
    assert np.isfinite(window.out_of_plane_bz_curve.getData()[0]).any()
    assert np.isfinite(window.out_of_plane_current_curve.getData()[1]).all()


def test_angle_calculator_photon_coverage_threshold_and_binding(angle_calculator):
    window = angle_calculator
    window.out_of_plane_check.setChecked(True)
    window.photon_energy_control.setValues((5, 20))
    window.inputs["analyzer_work_function"].setValue(6)
    window.out_of_plane_plot.setRange(xRange=(-20, 20), yRange=(1, 3), padding=0)
    window._redraw()
    coverage = window._hv_fill.image[..., 3].copy()
    assert np.count_nonzero(coverage) > 0
    assert not coverage[:, (0, -1)].any()
    assert np.isnan(window._hv_edge_curves[0].getData()[1]).all()
    window.inputs["binding_energy"].setValue(-5)
    window._redraw()
    shifted = window._hv_fill.image[..., 3]
    assert 0 < np.count_nonzero(shifted) < np.count_nonzero(coverage)
    assert np.all(shifted <= coverage)


def _drag_curve_point(window, parameter, value, alpha=3.0):
    if parameter == "beta":
        return QtCore.QPointF(
            *calc.angles_to_momentum(alpha, value, window._state.geometry)
        )
    kz = calc.angles_to_momentum3d(
        alpha, window.current_spin.value(), window._state.geometry, value
    )[2]
    return QtCore.QPointF(alpha, float(kz))


def _assert_drag_curve_projection(window):
    window._redraw()
    kx, ky = window.cut_curve.getData()
    alpha, beta = calc.momentum_to_angles(kx, ky, window._state.geometry)
    np.testing.assert_allclose(beta, window.current_spin.value(), atol=1e-8)
    np.testing.assert_allclose(
        (kx, ky),
        calc.angles_to_momentum(
            alpha, window.current_spin.value(), window._state.geometry
        ),
        atol=1e-10,
    )
    assert window.cut_marker.value() == window.current_spin.value()
    assert (
        window._state.scan_positions[window._scan_name()] == window.current_spin.value()
    )
    alpha, kz = window.out_of_plane_current_curve.getData()
    np.testing.assert_allclose(
        kz,
        calc.angles_to_momentum3d(
            alpha, window.current_spin.value(), window._state.geometry
        )[2],
    )
    assert (
        window._state.geometry.photon_energy == window.inputs["photon_energy"].value()
    )


@pytest.mark.parametrize("parameter", ["beta", "photon_energy"])
@pytest.mark.parametrize(
    ("configuration", "corrected"),
    [(1, False), (2, False), (3, False), (4, False), (3, True), (4, True)],
)
def test_angle_calculator_curve_drag_mapping(
    angle_calculator, parameter, configuration, corrected
):
    window = angle_calculator
    window.restore_workspace_state(
        AngleCalculatorState(
            geometry=calc.Geometry(
                configuration=configuration,
                photon_energy=6 if corrected else 90,
                sample_work_function=4.7,
                low_energy_correction=corrected,
                azimuth=17,
            ),
            out_of_plane_visible=True,
            bz_visible=False,
        )
    )
    curve = (
        window.cut_curve if parameter == "beta" else window.out_of_plane_current_curve
    )
    spin = window.current_spin if parameter == "beta" else window.inputs[parameter]
    initial = spin.value()
    # A press beside the curve must preserve its parameter offset.
    offset = 0.1 if corrected else 0.4
    press = _drag_curve_point(window, parameter, initial + offset)
    curve.sigDragStarted.emit(press)
    curve.sigDragged.emit(press)
    assert spin.value() == pytest.approx(initial, abs=1e-6)
    target = initial + (1 if parameter == "beta" else 2)
    curve.sigDragged.emit(_drag_curve_point(window, parameter, target + offset))
    curve.sigDragFinished.emit()
    assert spin.value() == pytest.approx(target, abs=1e-6)
    _assert_drag_curve_projection(window)


@pytest.mark.parametrize("parameter", ["beta", "photon_energy"])
def test_angle_calculator_curve_drag_bounds_and_invalid(angle_calculator, parameter):
    window = angle_calculator
    window.out_of_plane_check.setChecked(True)
    window.bz_check.setChecked(False)
    curve = (
        window.cut_curve if parameter == "beta" else window.out_of_plane_current_curve
    )
    spin = window.current_spin if parameter == "beta" else window.inputs[parameter]
    if parameter == "photon_energy":
        spin.setRange(30, 100)
    original = spin.value()
    for invalid in (QtCore.QPointF(float("nan"), 0), QtCore.QPointF(1e6, -1e6)):
        curve.sigDragStarted.emit(invalid)
        curve.sigDragged.emit(_drag_curve_point(window, parameter, original + 1))
        curve.sigDragFinished.emit()
        assert spin.value() == original
    curve.sigDragStarted.emit(_drag_curve_point(window, parameter, original))
    for value, expected in (
        (spin.maximum() + 2, spin.maximum()),
        (spin.minimum() - 2, spin.minimum()),
    ):
        curve.sigDragged.emit(_drag_curve_point(window, parameter, value))
        assert spin.value() == pytest.approx(expected)
        curve.sigDragged.emit(QtCore.QPointF(float("nan"), 0))
        assert spin.value() == pytest.approx(expected)
    curve.sigDragFinished.emit()
    _assert_drag_curve_projection(window)


@pytest.mark.parametrize("parameter", ["beta", "photon_energy"])
@pytest.mark.parametrize("finish", ["release", "undo", "close"])
def test_angle_calculator_curve_drag_history(
    angle_calculator, qtbot, parameter, finish
):
    window = angle_calculator
    window.restore_workspace_state(
        AngleCalculatorState(
            out_of_plane_visible=True,
            bz_visible=False,
            momentum_range=((-1.5, 1.5), (-1.5, 1.5)),
        )
    )
    curve = (
        window.cut_curve if parameter == "beta" else window.out_of_plane_current_curve
    )
    spin = window.current_spin if parameter == "beta" else window.inputs[parameter]
    before = window._history_state
    initial = spin.value()
    curve.sigDragStarted.emit(_drag_curve_point(window, parameter, initial))
    for delta in (1, 2):
        curve.sigDragged.emit(_drag_curve_point(window, parameter, initial + delta))
        qtbot.wait(180)
        assert not window.undoable
    after = window._history_state
    assert after != before
    if finish == "release":
        curve.sigDragFinished.emit()
    elif finish == "close":
        window.close()
    else:
        window.undo_action.trigger()
    if finish != "undo":
        window.undo_action.trigger()
    assert window._history_state == before
    assert not window.undoable
    # Events queued after an interrupted gesture must not change restored state.
    curve.sigDragged.emit(_drag_curve_point(window, parameter, initial + 3))
    curve.sigDragFinished.emit()
    assert window._history_state == before
    window.redo_action.trigger()
    assert window._history_state == after
    assert not window.redoable
    _assert_drag_curve_projection(window)


@pytest.mark.parametrize(
    ("finish", "previous_edit"),
    [
        ("release", None),
        ("undo", None),
        ("close", None),
        ("restore", None),
        ("release", "pending"),
        ("cancel", "redo"),
    ],
)
def test_angle_calculator_marker_drag_history(
    angle_calculator, qtbot, monkeypatch, finish, previous_edit
):
    import pyqtgraph as pg

    monkeypatch.setitem(pg.CONFIG_OPTIONS, "mouseRateLimit", 0)
    window = angle_calculator
    window.restore_workspace_state(
        AngleCalculatorState(
            bz_visible=False,
            angle_range=((-20, 20), (-10, 10)),
        )
    )
    window.resize(1500, 600)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    initial = window._history_state
    if previous_edit is not None:
        window.inputs["photon_energy"].setValue(80)
        prior_edit = window._history_state
        if previous_edit == "redo":
            window.undo()
    before = window._history_state
    plot = window.angle_plot
    view = plot.scene().views()[0]
    viewport = view.viewport()
    vb = plot.getViewBox()
    start = view.mapFromScene(vb.mapViewToScene(QtCore.QPointF(3, 0)))
    qtbot.mouseMove(viewport, pos=start + QtCore.QPoint(20, 20))
    qtbot.mouseMove(viewport, pos=start)
    qtbot.wait(20)
    qtbot.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    for value in (1, 2):
        end = view.mapFromScene(vb.mapViewToScene(QtCore.QPointF(3, value)))
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
        qtbot.wait(200)
        assert window.current_spin.value() == pytest.approx(value, abs=0.1)
        if previous_edit != "pending":
            assert not window.undoable
    after = window._history_state
    if finish == "release":
        qtbot.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    elif finish == "undo":
        window.undo_action.trigger()
    elif finish == "close":
        window.close()
    elif finish == "cancel":
        qtbot.mouseClick(viewport, QtCore.Qt.MouseButton.RightButton, pos=end)
    else:
        window.restore_workspace_state(AngleCalculatorState(**before.model_dump()))
    if finish not in ("undo", "restore", "cancel"):
        window.undo_action.trigger()
    assert window._history_state == before
    assert window.undoable == (previous_edit == "pending")
    qtbot.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert window._history_state == before
    if previous_edit == "pending":
        window.undo_action.trigger()
        assert window._history_state == initial
        assert not window.undoable
        window.redo_action.trigger()
        assert window._history_state == before
    if finish == "restore":
        assert not window.redoable
    else:
        window.redo_action.trigger()
        assert window._history_state == (prior_edit if finish == "cancel" else after)
        assert not window.redoable


@pytest.mark.parametrize("parameter", ["beta", "photon_energy"])
def test_angle_calculator_curve_mouse_drag(
    angle_calculator, qtbot, monkeypatch, parameter
):
    import pyqtgraph as pg

    monkeypatch.setitem(pg.CONFIG_OPTIONS, "mouseRateLimit", 0)
    window = angle_calculator
    window.restore_workspace_state(
        AngleCalculatorState(
            out_of_plane_visible=True,
            bz_visible=False,
            momentum_range=((-1.5, 1.5), (-1.5, 1.5)),
        )
    )
    window.resize(1500, 600)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    plot = window.momentum_plot if parameter == "beta" else window.out_of_plane_plot
    spin = window.current_spin if parameter == "beta" else window.inputs[parameter]
    initial = spin.value()
    view = plot.scene().views()[0]
    viewport = view.viewport()
    vb = plot.getViewBox()
    before_range = np.array(vb.viewRange())
    start = view.mapFromScene(
        vb.mapViewToScene(_drag_curve_point(window, parameter, initial))
    )
    qtbot.mouseMove(viewport, pos=start + QtCore.QPoint(20, 20))
    qtbot.mouseMove(viewport, pos=start)
    qtbot.wait(20)
    qtbot.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    for delta in (1, 2):
        target = initial + delta * (2 if parameter == "beta" else 10)
        end = view.mapFromScene(
            vb.mapViewToScene(_drag_curve_point(window, parameter, target))
        )
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
    qtbot.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert spin.value() == pytest.approx(target, abs=0.6)
    np.testing.assert_allclose(vb.viewRange(), before_range)
    _assert_drag_curve_projection(window)
    window.undo_action.trigger()
    assert spin.value() == initial
    assert not window.undoable


@pytest.mark.parametrize("parameter", ["beta", "photon_energy"])
def test_angle_calculator_curve_drag_workspace_restore(angle_calculator, parameter):
    window = angle_calculator
    curve = (
        window.cut_curve if parameter == "beta" else window.out_of_plane_current_curve
    )
    spin = window.current_spin if parameter == "beta" else window.inputs[parameter]
    initial = spin.value()
    curve.sigDragStarted.emit(_drag_curve_point(window, parameter, initial))
    curve.sigDragged.emit(_drag_curve_point(window, parameter, initial + 2))
    restored = AngleCalculatorState(
        geometry=calc.Geometry(photon_energy=40),
        scan_positions={"polar": 0, "tilt": 0, "deflector": -3},
        out_of_plane_visible=True,
        bz_visible=False,
    )
    window.restore_workspace_state(restored)
    baseline = window._history_state
    curve.sigDragged.emit(_drag_curve_point(window, parameter, initial + 3))
    curve.sigDragFinished.emit()
    assert window._history_state == baseline
    assert window.current_spin.value() == -3
    assert window.inputs["photon_energy"].value() == 40
    assert not window.undoable
    _assert_drag_curve_projection(window)


def _move_range_slider(slider, position):
    QtWidgets.QApplication.sendEvent(
        slider,
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(position),
            QtCore.QPointF(slider.mapToGlobal(position)),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    )


@pytest.mark.parametrize(
    "name", ["alpha_control", "scan_control", "photon_energy_control"]
)
@pytest.mark.parametrize("finish", ["release", "undo", "close", "hide", "restore"])
def test_angle_calculator_range_drag_history(angle_calculator, qtbot, name, finish):
    window = angle_calculator
    window.restore_workspace_state(
        AngleCalculatorState(out_of_plane_visible=True, bz_visible=False)
    )
    window.resize(1500, 700)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(30)
    control = getattr(window, name)
    slider = control.slider
    before = window._history_state
    start = slider._handle_rect(0).center()
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    for offset in (-10, -20):
        end = start + QtCore.QPoint(offset, 0)
        _move_range_slider(slider, end)
        qtbot.wait(200)
        assert not window.undoable
        assert not window._history_write_timer.isActive()
    after = window._history_state
    assert after != before
    if finish == "release":
        qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    elif finish == "undo":
        window.undo_action.trigger()
    elif finish == "close":
        window.close()
    elif finish == "hide":
        window.hide()
    else:
        window.restore_workspace_state(AngleCalculatorState(**before.model_dump()))
    if finish not in ("undo", "restore"):
        window.undo_action.trigger()
    assert window._history_state == before
    assert not window.undoable
    _move_range_slider(slider, start + QtCore.QPoint(-30, 0))
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert window._history_state == before
    assert window._range_drag is None
    assert not window._history_write_timer.isActive()
    if finish == "restore":
        assert not window.redoable
    else:
        window.redo_action.trigger()
        assert window._history_state == after
        assert not window.redoable


def test_angle_calculator_range_drag_boundaries(angle_calculator, qtbot):
    window = angle_calculator
    window.show()
    qtbot.waitExposed(window)
    control = window.alpha_control
    slider = control.slider
    states = [window._history_state]
    # A pending spin edit remains separate from the following mouse gesture.
    control.lower_spin.setValue(-20)
    states.append(window._history_state)
    for offset in (-10, -20):
        start = slider._handle_rect(0).center()
        qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
        end = start + QtCore.QPoint(offset, 0)
        _move_range_slider(slider, end)
        qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
        states.append(window._history_state)
    control.upper_spin.setValue(25)
    states.append(window._history_state)
    for state in reversed(states[:-1]):
        window.undo()
        assert window._history_state == state
    assert not window.undoable
    for state in states[1:]:
        window.redo()
        assert window._history_state == state


@pytest.mark.parametrize("cancel", [False, True])
def test_angle_calculator_range_drag_no_change(angle_calculator, qtbot, cancel):
    window = angle_calculator
    window.show()
    qtbot.waitExposed(window)
    control = window.alpha_control
    slider = control.slider
    window.current_spin.setValue(5)
    window.undo()
    assert window.redoable
    before = window._history_state
    start = slider._handle_rect(0).center()
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    if cancel:
        _move_range_slider(slider, start + QtCore.QPoint(-20, 0))
        qtbot.wait(200)
        qtbot.keyClick(slider, QtCore.Qt.Key.Key_Escape)
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    assert window._history_state == before
    assert not window.undoable
    assert not window.undo_action.isEnabled()
    assert window.redoable
    assert window.redo_action.isEnabled()
    assert window._range_drag is None


@pytest.mark.parametrize(
    ("configuration", "correction"),
    [(1, False), (2, False), (3, False), (4, False), (1, True), (2, True)],
)
def test_angle_calculator_scan_drag_cancel_restores_settings(
    angle_calculator, qtbot, configuration, correction
):
    window = angle_calculator
    window.restore_workspace_state(
        AngleCalculatorState(
            geometry=calc.Geometry(configuration=configuration), bz_visible=False
        )
    )
    window.resize(1500, 700)
    window.show()
    qtbot.waitExposed(window)
    if correction:
        window.scan_control.setValues((0.0, 0.0))
        window.correction_check.setChecked(True)
    else:
        window.current_spin.setValue(10.0)
    window._flush_pending_history_changes()
    before = window._history_state
    window.inputs["photon_energy"].setValue(before.geometry.photon_energy + 1)
    redo_state = window._history_state
    window.undo()
    history = tuple(window._prev_states), tuple(window._next_states)
    assert window.redoable

    slider = window.scan_control.slider
    start = slider._handle_rect(1).center()
    end = (
        start + QtCore.QPoint(20, 0)
        if correction
        else (start + slider._handle_rect(0).center()) / 2
    )
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    _move_range_slider(slider, end)
    qtbot.wait(200)
    if correction:
        assert not window._state.geometry.low_energy_correction
    else:
        assert window.current_spin.value() < 10.0
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Escape)
    _move_range_slider(slider, end + QtCore.QPoint(20, 0))
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    qtbot.wait(200)

    assert window._history_state == before
    assert (tuple(window._prev_states), tuple(window._next_states)) == history
    assert window.undo_action.isEnabled() == (len(history[0]) > 1)
    assert window.redo_action.isEnabled()
    assert window.current_spin.value() == (0.0 if correction else 10.0)
    assert window.cut_marker.value() == window.current_spin.value()
    assert window.correction_check.isChecked() == correction
    assert window._range_drag is None
    assert not window._history_write_timer.isActive()
    window.redo()
    assert window._history_state == redo_state


def test_angle_calculator_photon_range_integer_slider(angle_calculator, qtbot):
    window = angle_calculator
    window.restore_workspace_state(
        AngleCalculatorState(hv_range=(20.7, 80.7), out_of_plane_visible=True)
    )
    window.resize(1500, 700)
    window.show()
    qtbot.waitExposed(window)
    control = window.photon_energy_control
    assert control.lower_spin.minimum() == control.upper_spin.minimum() == 5
    control.lower_spin.setFocus()
    control.lower_spin.selectAll()
    qtbot.keyClicks(control.lower_spin, "21.2")
    qtbot.keyClick(control.lower_spin, QtCore.Qt.Key.Key_Return)
    assert control.values() == (21.2, 80.7)
    assert window._state.hv_range == (21.2, 80.7)
    window._flush_pending_history_changes()
    slider = control.slider
    start = slider._handle_rect(0).center()
    qtbot.mousePress(slider, QtCore.Qt.MouseButton.LeftButton, pos=start)
    end = start + QtCore.QPoint(20, 0)
    _move_range_slider(slider, end)
    qtbot.mouseRelease(slider, QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert control.values()[0].is_integer()
    assert window._state.hv_range == control.values()
    window.undo_action.trigger()
    assert control.values() == (21.2, 80.7)
    qtbot.keyClick(slider, QtCore.Qt.Key.Key_Home)
    assert control.values() == (5, 80.7)
    control.upper_spin.setValue(1)
    assert control.values() == slider.values() == (5, 5)


@pytest.mark.parametrize(
    ("value", "displayed_value"),
    [(6.001, 6.001), (60.123, 60.12), (600.123, 600.1), (1000, 1000)],
)
def test_angle_calculator_photon_energy_significant_display(
    angle_calculator, value, displayed_value
):
    window = angle_calculator
    spin = window.inputs["photon_energy"]
    assert isinstance(spin, erlab.interactive.utils.BetterSpinBox)
    spin.setValue(value)
    assert spin.valueFromText(spin.text()) == displayed_value
    spin.editingFinished.emit()
    assert spin.value() == window._state.geometry.photon_energy == value
    state = AngleCalculatorState.model_validate(window.workspace_state_payload())
    spin.setValue(90)
    window.restore_workspace_state(state)
    assert spin.value() == window._state.geometry.photon_energy == value
    assert spin.valueFromText(spin.text()) == displayed_value


@pytest.mark.parametrize("value", [1000, 1000.001])
def test_angle_calculator_photon_energy_maximum(angle_calculator, value):
    window = angle_calculator
    control = window.photon_energy_control
    current = window.inputs["photon_energy"]
    assert current.maximum() == 1000
    assert control.lower_spin.maximum() == control.upper_spin.maximum() == 1000
    current.setValue(value)
    control.upper_spin.setValue(value)
    control.lower_spin.setValue(value)
    assert current.value() == 1000
    assert control.values() == control.slider.values() == (1000, 1000)
    control.slider.setValues((value, value))
    assert control.slider.values() == (1000, 1000)
    state = AngleCalculatorState.model_validate(window.workspace_state_payload())
    assert state.geometry.photon_energy == 1000
    assert state.hv_range == (1000, 1000)
    assert AngleCalculatorState.model_validate_json(state.model_dump_json()) == state


@pytest.mark.parametrize("value", [50, 50.001])
def test_angle_calculator_inner_potential_maximum(angle_calculator, value):
    spin = angle_calculator.inputs["inner_potential"]
    spin.setValue(value)
    assert spin.value() == 50
    state = AngleCalculatorState.model_validate(
        angle_calculator.workspace_state_payload()
    )
    assert state.geometry.inner_potential == 50
    assert AngleCalculatorState.model_validate_json(state.model_dump_json()) == state
