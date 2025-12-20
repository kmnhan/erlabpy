import tempfile

import joblib
import pytest
import scipy
import xarray as xr
import xarray_lmfit as xlm
from qtpy import QtWidgets

import erlab
from erlab.interactive.fermiedge import GoldTool, ResolutionTool, goldtool, restool
from erlab.io.exampledata import generate_gold_edge


@pytest.mark.parametrize("fast", [True, False], ids=["StepB", "FD"])
def test_goldtool(
    qtbot, gold, fast, gold_fit_res, gold_fit_res_fd, accept_dialog, monkeypatch
) -> None:
    # Force joblib to avoid loky while Qt is running
    monkeypatch.setattr(joblib.parallel, "DEFAULT_BACKEND", "threading")
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    win.params_edge.widgets["Fast"].setChecked(fast)
    win.params_edge.widgets["# CPU"].setValue(1)

    expected = gold_fit_res if fast else gold_fit_res_fd

    with qtbot.wait_signal(win.sigUpdated, timeout=20000):
        win.params_edge.widgets["go"].click()

    def check_generated_code(w: GoldTool) -> None:
        namespace = {"era": erlab.analysis, "gold": gold}
        exec(w.gen_code("poly"), {"__builtins__": {}}, namespace)  # noqa: S102

        xr.testing.assert_identical(
            w.result.drop_vars("modelfit_results"),
            namespace["modelresult"].drop_vars("modelfit_results"),
        )
        xr.testing.assert_identical(
            w.result.drop_vars("modelfit_results"),
            expected.drop_vars("modelfit_results"),
        )

    check_generated_code(win)

    # Test save fit
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/fit_save.nc"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("fit_save.nc")

    # Save fit
    accept_dialog(win._save_poly_fit, pre_call=_go_to_file)

    # Check saved file
    xr.testing.assert_identical(
        expected.drop_vars("modelfit_results"),
        xlm.load_fit(filename).drop_vars("modelfit_results"),
    )

    # Move to spline tab
    win.params_tab.setCurrentIndex(1)

    assert isinstance(win.result, scipy.interpolate.BSpline)

    tmp_dir.cleanup()


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

    qtbot.wait_until(lambda: isinstance(win._result_ds, xr.Dataset), timeout=10000)

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

    # Test tool save & restore
    tmp_dir = tempfile.TemporaryDirectory()
    filename = f"{tmp_dir.name}/tool_save.h5"
    win.to_file(filename)

    win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, ResolutionTool)

    assert win.tool_status == win_restored.tool_status
    assert str(win_restored.info_text) == str(win.info_text)

    tmp_dir.cleanup()
