import time

import numpy as np
import pyperclip
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import erlab.lattice
from erlab.interactive.bzplot import BZPlotter
from erlab.interactive.curvefittingtool import edctool, mdctool
from erlab.interactive.derivative import dtool
from erlab.interactive.fermiedge import goldtool
from erlab.interactive.imagetool.manager import ImageToolManager
from erlab.interactive.kspace import ktool


def test_goldtool(qtbot, gold) -> None:
    win = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()
    win.params_edge.widgets["# CPU"].setValue(1)
    win.params_edge.widgets["Fast"].setChecked(True)

    with qtbot.waitSignal(win.fitter.sigFinished):
        win.params_edge.widgets["go"].click()

    win.params_poly.widgets["copy"].click()
    assert (
        pyperclip.paste()
        == """modelresult = era.gold.poly(
    gold,
    angle_range=(-13.5, 13.5),
    eV_range=(-0.204, 0.276),
    fast=True,
)"""
    )


def test_dtool(qtbot) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]).astype(
        np.float64
    )
    win = dtool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    win.tab_widget.setCurrentIndex(0)
    win.interp_group.setChecked(False)
    win.smooth_group.setChecked(True)
    assert (
        win.copy_code()
        == """_processed = data.copy()
for _ in range(1):
\t_processed = era.image.gaussian_filter(_processed, sigma={\"y\": 1.0, \"x\": 1.0})
result = _processed.differentiate('y').differentiate('y')"""
    )

    win.smooth_group.setChecked(False)
    assert win.copy_code() == "result = data.differentiate('y').differentiate('y')"

    win.tab_widget.setCurrentIndex(1)
    assert win.copy_code() == "result = era.image.scaled_laplace(data, factor=1.0)"

    win.tab_widget.setCurrentIndex(2)
    assert win.copy_code() == "result = era.image.curvature(data, a0=1.0, factor=1.0)"

    win.tab_widget.setCurrentIndex(3)
    assert win.copy_code() == "result = era.image.minimum_gradient(data)"


def test_ktool_compatible(anglemap) -> None:
    cut = anglemap.qsel(beta=-8.3)
    data_4d = anglemap.expand_dims("x", 2)
    data_3d_without_alpha = data_4d.qsel(alpha=-8.3)

    for data in (cut, data_4d, data_3d_without_alpha):
        with pytest.raises(
            ValueError, match="Data is not compatible with the interactive tool."
        ):
            data.kspace.interactive()


@pytest.mark.parametrize("constant_energy", [True, False])
def test_ktool(qtbot, anglemap, constant_energy) -> None:
    anglemap = anglemap.copy()

    if constant_energy:
        anglemap = anglemap.qsel(eV=-0.1)

    win = ktool(
        anglemap,
        avec=erlab.lattice.abc2avec(6.97, 6.97, 8.685, 90, 90, 120),
        rotate_bz=30.0,
        cmap="terrain_r",
        execute=False,
    )

    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()

    win._offset_spins["delta"].setValue(30.0)
    win._offset_spins["xi"].setValue(20.0)
    win._offset_spins["beta"].setValue(10.0)

    assert (
        win.copy_code()
        == """anglemap.kspace.offsets = {"delta": 30.0, "xi": 20.0, "beta": 10.0}
anglemap_kconv = anglemap.kspace.convert()"""
    )
    win.add_circle_btn.click()
    roi = win._roi_list[0]
    roi.getMenu()
    roi.set_position((0.1, 0.1), 0.2)
    assert roi.get_position() == (0.1, 0.1, 0.2)

    roi_control_widget = roi._pos_menu.actions()[0].defaultWidget()
    roi_control_widget.x_spin.setValue(0.0)
    roi_control_widget.y_spin.setValue(0.2)
    roi_control_widget.r_spin.setValue(0.3)
    assert roi.get_position() == (0.0, 0.2, 0.3)

    anglemap.kspace.offsets = {"delta": 30.0, "xi": 20.0, "beta": 10.0}

    if "eV" in anglemap.dims:
        anglemap_kconv = anglemap.kspace.convert()
    else:
        anglemap_kconv = anglemap.kspace.convert()

    # Show imagetool
    win.show_converted()
    xr.testing.assert_identical(win._itool.slicer_area.data, anglemap_kconv)
    win._itool.close()

    # Start manager
    manager = ImageToolManager()
    qtbot.addWidget(manager)

    # Show in manager
    win.show_converted()
    t0 = time.perf_counter()
    while True:
        if manager.ntools == 1:
            break
        assert time.perf_counter() - t0 < 20
        qtbot.wait(10)

    manager.remove_tool(0)
    assert manager.ntools == 0
    manager.close()
    win.close()


def test_curvefittingtool(qtbot) -> None:
    data = xr.DataArray(np.arange(25), dims=["x"]).astype(np.float64)
    win_mdc = mdctool(data, execute=False)
    win_edc = edctool(data, execute=False)
    qtbot.addWidget(win_edc)
    qtbot.addWidget(win_mdc)


def test_bzplot(qtbot) -> None:
    win = BZPlotter(execute=False)
    qtbot.addWidget(win)
    qtbot.addWidget(win.controls)

    with qtbot.waitExposed(win):
        win.show()

    with qtbot.waitExposed(win.controls):
        win.controls.show()

    win.controls.params_latt.set_values(
        a=3.0, b=3.0, c=6.0, alpha=90.0, beta=90.0, gamma=120.0
    )
    win.controls.params_latt.widgets["apply"].click()

    assert_allclose(win.controls.params_avec.values["a1x"], 3.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a1y"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a1z"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a2x"], -1.5, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a2z"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a3x"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a3y"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_avec.values["a3z"], 6.0, atol=1e-15)
    assert_allclose(
        win.controls.params_bvec.values["b1y"], 2 * np.pi / np.sqrt(3) / 3, atol=1e-15
    )
    assert_allclose(win.controls.params_bvec.values["b1z"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_bvec.values["b2x"], 0.0, atol=1e-15)
    assert_allclose(
        win.controls.params_bvec.values["b2y"], 4 * np.pi / np.sqrt(3) / 3, atol=1e-15
    )
    assert_allclose(win.controls.params_bvec.values["b2z"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_bvec.values["b3x"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_bvec.values["b3x"], 0.0, atol=1e-15)
    assert_allclose(win.controls.params_bvec.values["b3y"], 0.0, atol=1e-15)

    win.controls.close()
    win.close()
