import tempfile
import time

import joblib
import pytest
import scipy
import xarray as xr
import xarray_lmfit as xlm
from qtpy import QtGui, QtWidgets

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
    filename = f"{tmp_dir.name}/fit_save.h5"

    def _go_to_file(dialog: QtWidgets.QFileDialog):
        dialog.setDirectory(tmp_dir.name)
        dialog.selectFile(filename)
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("fit_save.h5")

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


def test_goldtool_roi_limits_descending_coords(qtbot, gold) -> None:
    gold_desc = gold.isel(alpha=slice(None, None, -1), eV=slice(None, None, -1))
    win: GoldTool = goldtool(gold_desc, execute=False)
    qtbot.addWidget(win)

    win.params_roi.modify_roi(x0=-10.0, x1=10.0, y0=-0.5, y1=0.2)
    x0, y0, x1, y1 = win.roi_limits_ordered
    assert x0 == pytest.approx(10.0)
    assert x1 == pytest.approx(-10.0)
    assert y0 == pytest.approx(0.2)
    assert y1 == pytest.approx(-0.5)


def test_restool(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)
    win.timeout_spin.setValue(5.0)

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


def test_restool_timeout_cleans_up_worker(qtbot, monkeypatch) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    def _slow_quick_fit(*args, **kwargs) -> xr.Dataset:
        iter_cb = kwargs["iter_cb"]
        while True:
            if iter_cb():
                raise RuntimeError("timed out")
            time.sleep(0.01)

    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _slow_quick_fit)

    win.live_check.setChecked(True)
    win.timeout_spin.setValue(0.1)
    win.do_fit()

    qtbot.wait_until(lambda: "Fit timed out" in win.overview_label.text(), timeout=2000)

    assert not win.live_check.isChecked()
    assert win._fit_thread is None
    assert win._result_ds is None


def test_restool_close_event_ignored_if_fit_thread_stuck(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _StuckThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.interrupted = False
            self.wait_timeout_ms: int | None = None

        def cancel(self) -> None:
            self.cancel_called = True

        def isRunning(self) -> bool:
            return True

        def requestInterruption(self) -> None:
            self.interrupted = True

        def wait(self, timeout_ms: int) -> bool:
            self.wait_timeout_ms = timeout_ms
            return False

    stuck_thread = _StuckThread()
    win._fit_thread = stuck_thread  # type: ignore[assignment]

    event = QtGui.QCloseEvent()
    assert event.isAccepted()
    win.closeEvent(event)
    assert not event.isAccepted()

    assert stuck_thread.cancel_called
    assert stuck_thread.interrupted
    assert stuck_thread.wait_timeout_ms == 5000


def test_restool_clear_fit_preview_keeps_line_when_live(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win.edc_fit.setData(x=[0.0, 1.0], y=[1.0, 2.0])
    win.live_check.setChecked(True)
    win._clear_fit_preview()

    x, y = win.edc_fit.getData()
    assert x is not None
    assert y is not None
    assert len(x) == 2
    assert len(y) == 2


def test_restool_clear_fit_preview_clears_line_when_not_live(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win.edc_fit.setData(x=[0.0, 1.0], y=[1.0, 2.0])
    win.live_check.setChecked(False)
    win._clear_fit_preview()

    x, y = win.edc_fit.getData()
    assert x is None
    assert y is None


def test_restool_invalidate_displayed_fit_clears_stale_overlay(
    qtbot, monkeypatch
) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win._fit_signature_displayed = ("old",)
    win.edc_fit.setData(x=[0.0, 1.0], y=[1.0, 2.0])
    monkeypatch.setattr(win, "_current_fit_signature", lambda: ("new",))

    signature = win._invalidate_displayed_fit_if_stale()

    assert signature == ("new",)
    assert win._fit_signature_current == ("new",)
    assert win._fit_signature_displayed is None
    x, y = win.edc_fit.getData()
    assert x is None
    assert y is None


def test_restool_handle_fit_success_ignores_stale_result(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win._result_ds = None
    win._fit_signature_current = ("current",)
    win._handle_fit_success(xr.Dataset(), 0.1, ("stale",))
    assert win._result_ds is None


def test_restool_do_fit_queues_when_thread_object_exists(qtbot, monkeypatch) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _ThreadPlaceholder:
        def cancel(self) -> None:
            return

        def requestInterruption(self) -> None:
            return

        def wait(self, timeout_ms: int) -> bool:
            return True

    called: list[bool] = []

    def _start_fit_worker() -> bool:
        called.append(True)
        return True

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)
    win._fit_thread = _ThreadPlaceholder()  # type: ignore[assignment]
    win._fit_queued = False

    win.do_fit()

    assert win._fit_queued
    assert not called


def test_restool_ranges_descending_coords(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    gold_desc = gold.isel(alpha=slice(None, None, -1), eV=slice(None, None, -1))
    win = restool(gold_desc, execute=False)
    qtbot.addWidget(win)

    win.x0_spin.setValue(-0.3)
    win.x1_spin.setValue(0.3)
    win.y0_spin.setValue(-12.0)
    win.y1_spin.setValue(12.0)

    assert win.x_range == pytest.approx((0.3, -0.3))
    assert win.y_range == pytest.approx((12.0, -12.0))
