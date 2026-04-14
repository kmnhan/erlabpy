import json
import tempfile
import time
import typing

import joblib
import numpy as np
import pytest
import scipy
import xarray as xr
import xarray_lmfit as xlm
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.fermiedge import (
    GoldTool,
    ResolutionFitThread,
    ResolutionTool,
    goldtool,
    restool,
)
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


def test_goldtool_update_data_invalidates_fit_and_can_refit(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    win.params_edge.widgets["T (K)"].setValue(45.0)
    degree_widget = typing.cast("QtWidgets.QSpinBox", win.params_poly.widgets["Degree"])
    with QtCore.QSignalBlocker(win.params_poly):
        degree_widget.setValue(3)
    win.params_roi.modify_roi(x0=-8.0, x1=8.0, y0=-0.2, y1=0.1)
    win.params_poly.setDisabled(False)
    win.params_spl.setDisabled(False)
    win.params_tab.setDisabled(False)
    win.edge_center = gold.mean("eV")
    win.edge_stderr = xr.ones_like(win.edge_center)
    win.result = xr.Dataset()

    called: list[bool] = []
    monkeypatch.setattr(win, "perform_edge_fit", lambda: called.append(True))

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.05
    win.update_data(new_gold)

    assert win.params_edge.values["T (K)"] == pytest.approx(45.0)
    assert win.params_poly.values["Degree"] == 3
    assert win.result is None
    assert win.params_tab.isEnabled() is False
    assert not called

    win.edge_center = new_gold.mean("eV")
    win.edge_stderr = xr.ones_like(win.edge_center)
    win.result = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)

    newer_gold = new_gold.copy(deep=True)
    newer_gold.data = np.asarray(newer_gold.data) * 1.02
    win.update_data(newer_gold)

    assert called == [True]


def test_goldtool_update_data_ignores_late_results_from_aborted_task(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    class _DummyTask:
        def __init__(self) -> None:
            self.aborted = False

        def abort_fit(self) -> None:
            self.aborted = True

    stale_task = _DummyTask()
    win._fit_task = stale_task

    perform_called: list[bool] = []
    monkeypatch.setattr(win, "perform_fit", lambda: perform_called.append(True))

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.01
    win.update_data(new_gold)

    assert stale_task.aborted is True
    assert win._fit_task is None
    assert not hasattr(win, "edge_center")
    assert not perform_called

    stale_center = gold.mean("eV")
    stale_stderr = xr.ones_like(stale_center)
    win.post_fit(stale_center, stale_stderr, task=stale_task)

    assert not hasattr(win, "edge_center")
    assert not perform_called


def test_goldtool_update_data_defers_until_fit_worker_drains(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    class _DummyTask:
        def __init__(self) -> None:
            self.aborted = False

        def abort_fit(self) -> None:
            self.aborted = True

    stale_task = _DummyTask()
    win._fit_task = stale_task

    active_counts = iter((1, 0))
    monkeypatch.setattr(
        win._threadpool,
        "activeThreadCount",
        lambda: next(active_counts, 0),
    )

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.02
    win.update_data(new_gold)

    assert stale_task.aborted is True
    assert win._pending_update_request is not None
    xr.testing.assert_identical(win.data, gold)

    win._pending_update_timer.stop()
    win._flush_pending_update()

    assert win._pending_update_request is None
    xr.testing.assert_identical(win.data, new_gold)


def test_goldtool_auto_source_update_stays_stale_until_deferred_refresh_applies(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)
    win.set_source_binding(
        erlab.interactive.imagetool.provenance.full_data(),
        auto_update=True,
    )

    class _DummyTask:
        def __init__(self) -> None:
            self.aborted = False

        def abort_fit(self) -> None:
            self.aborted = True

    stale_task = _DummyTask()
    win._fit_task = stale_task

    active_counts = iter((1, 0))
    monkeypatch.setattr(
        win._threadpool,
        "activeThreadCount",
        lambda: next(active_counts, 0),
    )

    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.02
    win.handle_parent_source_replaced(new_gold)

    assert stale_task.aborted is True
    assert win.source_state == "stale"
    xr.testing.assert_identical(win.data, gold)

    win._pending_update_timer.stop()
    win._flush_pending_update()

    assert win.source_state == "fresh"
    xr.testing.assert_identical(win.data, new_gold)


def test_goldtool_close_event_ignored_if_threadpool_does_not_quiesce(
    qtbot, gold, monkeypatch
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    monkeypatch.setattr(
        type(win),
        "_wait_for_threadpool",
        staticmethod(lambda *args, **kwargs: False),
    )

    event = QtGui.QCloseEvent()
    assert event.isAccepted()
    win.closeEvent(event)
    assert not event.isAccepted()


def test_goldtool_update_data_clamps_roi_to_non_empty_bounds(qtbot, gold) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    win.params_roi.modify_roi(x0=-12.0, x1=-8.0, y0=-0.45, y1=-0.3)
    narrowed = gold.sel(alpha=slice(-2.5, 2.5), eV=slice(-0.1, 0.05))

    win.update_data(narrowed)

    x0, y0, x1, y1 = win.params_roi.roi_limits
    xmin, ymin, xmax, ymax = win.params_roi.max_bounds
    assert xmin <= x0 < x1 <= xmax
    assert y0 < y1 <= ymax
    assert y0 >= ymin - 1e-3

    sel_x0, sel_y0, sel_x1, sel_y1 = win.roi_limits_ordered
    selected = win.data.sel(
        {win._along_dim: slice(sel_x0, sel_x1), "eV": slice(sel_y0, sel_y1)}
    )
    assert selected.sizes[win._along_dim] > 0
    assert selected.sizes["eV"] > 0


def test_goldtool_validate_update_data_transposes_and_rejects_invalid(
    qtbot, gold
) -> None:
    win: GoldTool = goldtool(gold, execute=False)
    qtbot.addWidget(win)

    transposed = win.validate_update_data(gold.transpose(*reversed(gold.dims)))
    assert transposed.dims[0] == "eV"

    with pytest.raises(ValueError, match="2D DataArray with an `eV` dimension"):
        win.validate_update_data(xr.DataArray(np.arange(5), dims=("alpha",)))


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
        namespace = {"era": erlab.analysis, "gold": gold, "data": gold, "result": None}
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


def test_restool_loads_legacy_saved_state_without_source_update_flag(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        filename = f"{tmp_dir_name}/tool_save.h5"
        win.to_file(filename)

        with xr.load_dataset(filename, engine="h5netcdf") as ds:
            legacy_ds = ds.load()

        legacy_state = json.loads(legacy_ds.attrs["tool_state"])
        legacy_state.pop("refit_on_source_update", None)
        legacy_ds.attrs["tool_state"] = json.dumps(legacy_state)
        legacy_ds.to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)

        win_restored = erlab.interactive.utils.ToolWindow.from_file(filename)
        qtbot.addWidget(win_restored)
        assert isinstance(win_restored, ResolutionTool)
        assert win_restored.tool_status.refit_on_source_update is False
        assert win_restored.refit_on_source_update_check.isChecked() is False


def test_restool_update_data_invalidates_fit_and_can_refit(qtbot, monkeypatch) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win.x0_spin.setValue(-0.2)
    win.x1_spin.setValue(0.2)
    win.y0_spin.setValue(-10.0)
    win.y1_spin.setValue(10.0)
    win.live_check.setChecked(True)
    win.temp_spin.setValue(90.0)
    win.fix_temp_check.setChecked(False)
    win.center_spin.setValue(-0.01)
    win.fix_center_check.setChecked(True)
    win.res_spin.setValue(0.015)
    win.fix_res_check.setChecked(True)
    win.slope_check.setChecked(False)
    win.timeout_spin.setValue(2.5)
    win.nfev_spin.setValue(250)
    win.refit_on_source_update_check.setChecked(False)
    win._result_ds = xr.Dataset()

    fit_inputs: list[xr.DataArray] = []
    monkeypatch.setattr(
        win, "do_fit", lambda: fit_inputs.append(win.averaged_edc.copy(deep=True))
    )

    status = win.tool_status
    new_gold = gold.copy(deep=True)
    new_gold.data = np.asarray(new_gold.data) * 1.03
    win.update_data(new_gold)

    expected = status.model_copy(
        update={"results": ("No fit results", "—", "—", "—", "—")}
    )
    assert win.tool_status == expected
    assert win._result_ds is None
    xr.testing.assert_identical(
        win.averaged_edc,
        new_gold.sel({win.y_dim: slice(*win.y_range)}).mean(win.y_dim),
    )
    assert not fit_inputs

    win._result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)
    newer_gold = new_gold.copy(deep=True)
    newer_gold.data = np.asarray(newer_gold.data) * 1.01
    win.update_data(newer_gold)

    xr.testing.assert_identical(
        fit_inputs[0], newer_gold.sel({win.y_dim: slice(*win.y_range)}).mean(win.y_dim)
    )


def test_restool_update_data_returns_false_if_fit_thread_stays_alive(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)
    original = win.tool_data.copy(deep=True)

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

    updated = gold.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.01

    assert win.update_data(updated) is False
    assert stuck_thread.cancel_called
    assert stuck_thread.interrupted
    assert stuck_thread.wait_timeout_ms == win.BACKGROUND_TASK_TIMEOUT_MS
    xr.testing.assert_identical(win.tool_data, original)
    win._fit_thread = None


def test_restool_update_data_auto_refit_after_waiting_cancelled_thread(
    qtbot, monkeypatch
) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _FinishedThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.interrupted = False
            self.wait_timeout_ms: int | None = None
            self.deleted = False

        def cancel(self) -> None:
            self.cancel_called = True

        def requestInterruption(self) -> None:
            self.interrupted = True

        def wait(self, timeout_ms: int) -> bool:
            self.wait_timeout_ms = timeout_ms
            return True

        def deleteLater(self) -> None:
            self.deleted = True

    old_thread = _FinishedThread()
    win._fit_thread = old_thread  # type: ignore[assignment]
    win._result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)

    started: list[bool] = []

    def _start_fit_worker() -> bool:
        started.append(True)
        assert win._fit_thread is None
        return True

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    updated = gold.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.01

    assert win.update_data(updated) is True
    assert started == [True]
    assert old_thread.cancel_called
    assert old_thread.interrupted
    assert old_thread.wait_timeout_ms == win.BACKGROUND_TASK_TIMEOUT_MS
    assert old_thread.deleted is True


def test_restool_queue_fit_action_ignores_stale_thread(qtbot) -> None:
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

        def deleteLater(self) -> None:
            return

    stale_thread = _ThreadPlaceholder()
    current_thread = _ThreadPlaceholder()
    win._fit_thread = typing.cast("typing.Any", current_thread)

    called: list[str] = []
    win._queue_fit_action(
        typing.cast("typing.Any", stale_thread), lambda: called.append("stale")
    )
    assert called == []
    assert win._pending_fit_action is None


def test_restool_validate_update_data_transposes_and_rejects_invalid(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    transposed = win.validate_update_data(gold.transpose(*reversed(gold.dims)))
    assert transposed.dims[-1] == "eV"

    with pytest.raises(ValueError, match="2D and have an 'eV' dimension"):
        win.validate_update_data(xr.DataArray(np.arange(5), dims=("alpha",)))


def test_restool_fit_thread_loads_result_before_emit(monkeypatch) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})

    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )

    class _DummyResult:
        def __init__(self) -> None:
            self.loaded = False

        def load(self):
            self.loaded = True
            return self

    dummy = _DummyResult()

    def _quick_fit(*_args, **_kwargs):
        return dummy

    captured: dict[str, object | None] = {"result": None, "elapsed": None}

    def _on_finished(result, elapsed: float) -> None:
        captured["result"] = result
        captured["elapsed"] = elapsed

    thread.sigFinished.connect(_on_finished)
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)

    thread.run()

    assert dummy.loaded
    assert captured["result"] is dummy
    assert isinstance(captured["elapsed"], float)


def test_restool_fit_thread_cancel_sets_event() -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})
    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )
    thread.cancel()
    assert thread._cancel.is_set()


def test_restool_fit_thread_runtimeerror_interruption_treated_as_cancelled(
    monkeypatch,
) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})

    class _ThreadWithInterruptedWrapper(ResolutionFitThread):
        def isInterruptionRequested(self) -> bool:
            raise RuntimeError("thread wrapper deleted")

    thread = _ThreadWithInterruptedWrapper(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )

    events = {"cancelled": False}

    def _quick_fit(*_args, **kwargs):
        kwargs["iter_cb"]()
        raise RuntimeError("cancel")

    thread.sigCancelled.connect(lambda: events.__setitem__("cancelled", True))
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)

    thread.run()
    assert events["cancelled"]


def test_restool_fit_thread_error_emits_errored(monkeypatch) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})
    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )
    events = {"errored": False}

    def _quick_fit(*_args, **_kwargs):
        raise RuntimeError("boom")

    thread.sigErrored.connect(lambda _msg: events.__setitem__("errored", True))
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)
    thread.run()
    assert events["errored"]


def test_restool_fit_thread_cancelled_after_success_emits_cancelled(
    monkeypatch,
) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})
    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )

    class _DummyResult:
        def load(self):
            return self

    events = {"cancelled": False}

    def _quick_fit(*_args, **kwargs):
        thread._cancel.set()
        kwargs["iter_cb"]()
        return _DummyResult()

    thread.sigCancelled.connect(lambda: events.__setitem__("cancelled", True))
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)
    thread.run()
    assert events["cancelled"]


def test_restool_fit_thread_timed_out_after_success_emits_timeout(
    monkeypatch,
) -> None:
    eV = np.linspace(-0.2, 0.2, 31)
    data = xr.DataArray(np.exp(-(eV**2)), dims=("eV",), coords={"eV": eV})
    thread = ResolutionFitThread(
        data,
        {
            "eV_range": (-0.1, 0.1),
            "method": "least_squares",
            "temp": 100.0,
            "resolution": 0.02,
            "center": 0.0,
            "fix_temp": True,
            "fix_center": False,
            "fix_resolution": False,
            "bkg_slope": True,
            "max_nfev": 5,
        },
        timeout=1.0,
    )

    class _DummyResult:
        def load(self):
            return self

    events = {"timed_out": False}

    def _quick_fit(*_args, **kwargs):
        kwargs["iter_cb"]()
        return _DummyResult()

    timer = iter([0.0, 2.0, 3.0])
    monkeypatch.setattr(
        "erlab.interactive.fermiedge.time.perf_counter", lambda: next(timer)
    )
    thread.sigTimedOut.connect(lambda _elapsed: events.__setitem__("timed_out", True))
    monkeypatch.setattr(erlab.analysis.gold, "quick_fit", _quick_fit)
    thread.run()
    assert events["timed_out"]


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


def test_restool_cancel_fit_waits_without_timeout(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.interrupted = False
            self.wait_args: tuple[object, ...] | None = None
            self.deleted = False

        def cancel(self) -> None:
            self.cancel_called = True

        def isRunning(self) -> bool:
            return True

        def requestInterruption(self) -> None:
            self.interrupted = True

        def wait(self, *args) -> bool:
            self.wait_args = args
            return True

        def deleteLater(self) -> None:
            self.deleted = True

    dummy_thread = _DummyThread()
    win._fit_thread = dummy_thread  # type: ignore[assignment]

    assert win._cancel_fit(wait=True, timeout_ms=None)
    assert dummy_thread.cancel_called
    assert dummy_thread.interrupted
    assert dummy_thread.wait_args == ()
    assert dummy_thread.deleted


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

        def deleteLater(self) -> None:
            return

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


def test_restool_queue_fit_action_drops_when_cancel_requested(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    called = {"value": False}
    win._fit_cancel_requested = True
    win._pending_fit_action = None
    win._queue_fit_action(None, lambda: called.__setitem__("value", True))  # type: ignore[arg-type]

    assert not called["value"]
    assert win._pending_fit_action is None


def test_restool_queue_fit_action_runs_immediately_when_no_thread(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    called = {"value": False}
    win._fit_cancel_requested = False
    win._fit_thread = None
    win._pending_fit_action = None
    win._queue_fit_action(None, lambda: called.__setitem__("value", True))  # type: ignore[arg-type]

    assert called["value"]
    assert win._pending_fit_action is None


def test_restool_start_fit_worker_returns_false_when_thread_exists(qtbot) -> None:
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

        def deleteLater(self) -> None:
            return

    win._fit_thread = _ThreadPlaceholder()  # type: ignore[assignment]
    assert not win._start_fit_worker()
    win._fit_thread = None


def test_restool_start_fit_worker_sets_signature_when_missing(
    qtbot, monkeypatch
) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    class _DummySignal:
        def connect(self, *_args, **_kwargs) -> None:
            return

    class _DummyThread:
        def __init__(self, *_args, **_kwargs) -> None:
            self.finished = _DummySignal()
            self.sigFinished = _DummySignal()
            self.sigTimedOut = _DummySignal()
            self.sigErrored = _DummySignal()
            self.sigCancelled = _DummySignal()
            self.started = False

        def cancel(self) -> None:
            return

        def requestInterruption(self) -> None:
            return

        def wait(self, timeout_ms: int) -> bool:
            return True

        def start(self) -> None:
            self.started = True

    monkeypatch.setattr("erlab.interactive.fermiedge.ResolutionFitThread", _DummyThread)
    monkeypatch.setattr(win, "_current_fit_signature", lambda: ("sig",))
    win._fit_thread = None
    win._fit_signature_current = None
    win._fit_cancel_requested = True

    assert win._start_fit_worker()
    assert isinstance(win._fit_thread, _DummyThread)
    assert win._fit_cancel_requested is False
    assert win._fit_thread.started
    win._fit_thread = None


def test_restool_handle_fit_cancelled_returns_none(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)
    assert win._handle_fit_cancelled() is None


def test_restool_handle_fit_timeout_ignores_stale_signature(qtbot) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)
    win._result_ds = xr.Dataset({"x": xr.DataArray([1.0])})
    win._fit_signature_displayed = ("current",)
    win.live_check.setChecked(True)

    win._fit_signature_current = ("current",)
    win._handle_fit_timeout(0.1, ("stale",))

    assert win._result_ds is not None
    assert win._fit_signature_displayed == ("current",)
    assert win.live_check.isChecked()


def test_restool_handle_fit_error_updates_ui_for_current_signature(
    qtbot, monkeypatch
) -> None:
    gold = generate_gold_edge(
        edge_coeffs=(0.0, 0.0, 0.0), background_coeffs=(5.0, 0.0, -2e-3), seed=1
    )
    win = restool(gold, execute=False)
    qtbot.addWidget(win)

    win._fit_signature_current = ("current",)
    win._fit_signature_displayed = ("current",)
    win._result_ds = xr.Dataset({"x": xr.DataArray([1.0])})
    win.live_check.setChecked(True)
    win.edc_fit.setData(x=[0.0, 1.0], y=[1.0, 2.0])

    failed = {"text": None}
    monkeypatch.setattr(
        win, "_fit_failed", lambda text: failed.__setitem__("text", text)
    )

    win._handle_fit_error("boom", ("current",))

    assert win._result_ds is None
    assert win._fit_signature_displayed is None
    assert not win.live_check.isChecked()
    x, y = win.edc_fit.getData()
    assert x is None
    assert y is None
    assert failed["text"] == "Fit failed"


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
