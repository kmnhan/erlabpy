import numpy as np
import xarray as xr
from erlab.interactive.bzplot import BZPlotter
from erlab.interactive.curvefittingtool import edctool, mdctool
from erlab.interactive.derivative import dtool
from erlab.interactive.fermiedge import goldtool
from erlab.interactive.kspace import ktool
from numpy.testing import assert_allclose


def test_goldtool(qtbot, gold):
    win = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()
        win.raise_()


def test_dtool(qtbot):
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]).astype(
        np.float64
    )
    win = dtool(data, execute=False)
    qtbot.addWidget(win)

    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()
        win.raise_()

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


def test_ktool(qtbot, anglemap):
    win = ktool(anglemap, execute=False)
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()
        win.activateWindow()
        win.raise_()

    win._offset_spins["delta"].setValue(30.0)
    win._offset_spins["xi"].setValue(20.0)
    win._offset_spins["beta"].setValue(10.0)

    assert (
        win.copy_code()
        == """anglemap.kspace.offsets = {"delta": 30.0, "xi": 20.0, "beta": 10.0}
anglemap_kconv = anglemap.kspace.convert()"""
    )


def test_curvefittingtool(qtbot):
    data = xr.DataArray(np.arange(25), dims=["x"]).astype(np.float64)
    win_mdc = mdctool(data, execute=False)
    win_edc = edctool(data, execute=False)
    qtbot.addWidget(win_edc)
    qtbot.addWidget(win_mdc)


def test_bzplot(qtbot):
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
