import numpy as np
import pyperclip
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import erlab
from erlab.interactive.bzplot import BZPlotter
from erlab.interactive.curvefittingtool import edctool, mdctool
from erlab.interactive.derivative import DerivativeTool, dtool
from erlab.interactive.fermiedge import ResolutionTool, goldtool, restool
from erlab.interactive.kspace import KspaceTool, ktool
from erlab.io.exampledata import generate_gold_edge


def test_goldtool(qtbot, gold) -> None:
    win = goldtool(gold, execute=False)
    qtbot.addWidget(win)
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


def test_restool(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win._guess_temp()
    qtbot.wait_until(lambda: win.fit_params["temp"] == 100.0, timeout=1000)

    win._guess_center()
    win.res_spin.setValue(0.02)
    win.live_check.setChecked(True)
    win.y0_spin.setValue(-12.0)
    win.x0_spin.setValue(-0.3)
    win.x1_spin.setValue(0.3)

    qtbot.wait_until(lambda: isinstance(win._result_ds, xr.Dataset), timeout=1000)

    for k, v in {
        "eV_range": (-0.3, 0.3),
        "temp": 100.0,
        "resolution": 0.02,
        "center": -0.01037,
        "bkg_slope": False,
    }.items():
        assert win.fit_params[k] == v

    def check_generated_code(w: ResolutionTool) -> None:
        namespace = {"era": erlab.analysis, "gold": gold, "result": None}
        code = "result = " + w.copy_code().replace("quick_resolution", "quick_fit")
        exec(code, {"__builtins__": {"slice": slice}}, namespace)  # noqa: S102

        xr.testing.assert_identical(
            w._result_ds.drop_vars("modelfit_results"),
            namespace["result"].drop_vars("modelfit_results"),
        )

    check_generated_code(win)
    win.close()


@pytest.mark.parametrize("method_idx", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("interpmode", ["interp", "nointerp"])
@pytest.mark.parametrize(
    ("smoothmode", "nsmooth"),
    [
        ("none", 1),
        ("gaussian", 1),
        ("gaussian", 3),
        ("boxcar", 1),
        ("boxcar", 3),
    ],
)
def test_dtool(qtbot, interpmode, smoothmode, nsmooth, method_idx) -> None:
    data = xr.DataArray(np.arange(25).reshape((5, 5)), dims=["x", "y"]).astype(
        np.float64
    )
    win: DerivativeTool = dtool(data, execute=False)
    qtbot.addWidget(win)

    def check_generated_code(w: DerivativeTool) -> None:
        namespace = {"era": erlab.analysis, "data": data, "np": np, "result": None}
        exec(w.copy_code(), {"__builtins__": {"range": range}}, namespace)  # noqa: S102
        xr.testing.assert_identical(w.result, namespace["result"])

    win.interp_group.setChecked(interpmode == "interp")
    win.smooth_group.setChecked(smoothmode != "none")
    win.sn_spin.setValue(nsmooth)

    match smoothmode:
        case "gaussian":
            win.smooth_combo.setCurrentIndex(0)
        case "boxcar":
            win.smooth_combo.setCurrentIndex(1)

    win.tab_widget.setCurrentIndex(method_idx)
    check_generated_code(win)
    win.close()


def test_ktool_compatible(anglemap) -> None:
    cut = anglemap.qsel(beta=-8.3)
    data_4d = anglemap.expand_dims("x", 2)
    data_3d_without_alpha = data_4d.qsel(alpha=-8.3)

    for data in (cut, data_4d, data_3d_without_alpha):
        with pytest.raises(
            ValueError, match="Data is not compatible with the interactive tool."
        ):
            data.kspace.interactive()


@pytest.mark.parametrize("wf", ["wf_auto", "wf_manual"])
@pytest.mark.parametrize("kind", ["map", "const_energy"])
@pytest.mark.parametrize("assignment", ["before", "after"])
def test_ktool(qtbot, anglemap, wf, kind, assignment) -> None:
    offset_dict = {"delta": 30.0, "xi": 20.0, "beta": 10.0}

    anglemap = anglemap.copy()

    def assign_attrs():
        anglemap.kspace.offsets = offset_dict
        if wf != "wf_auto":
            anglemap.kspace.work_function = 4.0

    if kind != "map":
        anglemap = anglemap.qsel(eV=-0.1)

    if assignment == "before":
        assign_attrs()

    win = ktool(
        anglemap,
        avec=erlab.lattice.abc2avec(6.97, 6.97, 8.685, 90, 90, 120),
        rotate_bz=30.0,
        cmap="terrain_r",
        execute=False,
    )
    qtbot.addWidget(win)

    for k, v in offset_dict.items():
        if assignment == "before":
            assert np.isclose(win._offset_spins[k].value(), v)
        else:
            win._offset_spins[k].setValue(v)

    if wf != "wf_auto":
        if assignment == "before":
            assert np.isclose(win._offset_spins["wf"].value(), 4.0)
        else:
            win._offset_spins["wf"].setValue(4.0)

    if assignment != "before":
        assign_attrs()
    anglemap_kconv = anglemap.kspace.convert()

    def _check_code_kconv(w: KspaceTool):
        namespace = {"anglemap": anglemap}
        exec(w.copy_code(), {"__builtins__": {}}, namespace)  # noqa: S102
        xr.testing.assert_identical(anglemap_kconv, namespace["anglemap_kconv"])

    _check_code_kconv(win)

    # Test ROI
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

    # Show imagetool
    win.show_converted()
    xr.testing.assert_identical(win._itool.slicer_area.data, anglemap_kconv)
    win._itool.close()
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
