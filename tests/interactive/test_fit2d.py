import contextlib
import os
import re

import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._fit2d import Fit2DTool
from tests._qt_helpers import signal_receiver_count


def _make_1d_data() -> xr.DataArray:
    x = np.linspace(-1.0, 1.0, 11)
    data = np.exp(-(x**2))
    return xr.DataArray(data, dims=("x",), coords={"x": x}, name="spec")


def _make_2d_data() -> xr.DataArray:
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(0.0, 2.0, 3)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    data = np.exp(-(xx**2)) * (1.0 + 0.1 * yy)
    return xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x}, name="map")


def _configure_fit2d_for_tests(
    win: Fit2DTool, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str | None]]]:
    # Avoid flaky one-second fit timeout under coverage instrumentation.
    win.timeout_spin.setValue(30.0)

    warnings: list[tuple[str, str]] = []
    errors: list[tuple[str, str, str | None]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    def _error(title: str, text: str, detailed_text: str | None = None) -> None:
        errors.append((title, text, detailed_text))

    monkeypatch.setattr(win, "_show_warning", _warn)
    monkeypatch.setattr(win, "_show_error", _error)
    return warnings, errors


def test_ftool_2d_fill_and_transpose(qtbot, accept_dialog) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_index_spin.setValue(0)
    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "2.0", QtCore.Qt.ItemDataRole.EditRole)

    win.y_index_spin.setValue(1)
    win._fill_params_from_prev()
    assert win.param_model.param_at(0).value == pytest.approx(2.0)

    original_dims = win._data_full.dims
    accept_dialog(win._transpose)
    assert win._data_full.dims == (original_dims[1], original_dims[0])
    assert win._y_dim_name == win._data_full.dims[0]


def test_fit2d_tool_status_restore(qtbot, exp_decay_model) -> None:
    data = _make_2d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_index_spin.setValue(0)
    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "2.0", QtCore.Qt.ItemDataRole.EditRole)

    win.y_index_spin.setValue(1)
    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "3.0", QtCore.Qt.ItemDataRole.EditRole)

    status = win.tool_status
    assert status.state2d is not None

    win_restored = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win_restored)
    win_restored.tool_status = status

    win_restored.y_index_spin.setValue(0)
    assert win_restored.param_model.param_at(0).value == pytest.approx(2.0)
    win_restored.y_index_spin.setValue(1)
    assert win_restored.param_model.param_at(0).value == pytest.approx(3.0)


def test_fit2d_tool_status_overlay_and_limits(qtbot, exp_decay_model) -> None:
    data = _make_2d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_min_spin.setValue(1)
    win.y_max_spin.setValue(2)
    param_name = win.param_plot_combo.itemText(0)
    win.param_plot_combo.setCurrentText(param_name)
    win.param_plot_overlay_check.setChecked(True)

    status = win.tool_status
    assert status.state2d is not None
    assert status.state2d.y_limits == (1, 2)
    assert status.state2d.param_plot_overlay_states.get(param_name) is True

    win_restored = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win_restored)
    win_restored.tool_status = status

    assert win_restored.y_min_spin.value() == 1
    assert win_restored.y_max_spin.value() == 2
    win_restored.param_plot_combo.setCurrentText(param_name)
    assert win_restored.param_plot_overlay_check.isChecked() is True


def test_fit2d_overlay_legend_sync(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    param_name = win.param_plot_combo.itemText(0)
    win.param_plot_combo.setCurrentText(param_name)
    win.param_plot_overlay_check.setChecked(True)
    win._update_param_plot_overlays()

    errbar, scatter = win._param_plot_overlay_items[param_name]
    scatter.setVisible(False)

    class _Sample:
        def __init__(self, item):
            self.item = item

    win._on_image_legend_sample_clicked(_Sample(scatter))
    qtbot.waitUntil(lambda: errbar.isVisible() is False)
    assert win._param_plot_overlay_states[param_name] is False
    assert win.param_plot_overlay_check.isChecked() is False

    scatter.setVisible(True)
    win._sync_overlay_visibility(param_name, scatter, errbar)
    assert errbar.isVisible() is True
    assert win._param_plot_overlay_states[param_name] is True
    assert win.param_plot_overlay_check.isChecked() is True


def test_fit2d_update_param_plot_overlays_paths(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    names = [win.param_plot_combo.itemText(i) for i in range(2)]
    for name in names:
        win._param_plot_overlay_states[name] = True
    win._update_param_plot_overlays()
    assert win.image_plot_legend.isVisible() is True
    assert set(win._param_plot_overlay_items.keys()) == set(names)

    win._param_plot_overlay_states[names[0]] = False
    win._update_param_plot_overlays()
    assert names[0] not in win._param_plot_overlay_items
    legend_names = {item[1].text for item in win.image_plot_legend.items}
    assert names[0] not in legend_names

    for name in names:
        win._param_plot_overlay_states[name] = False
    win._update_param_plot_overlays()
    assert not win._param_plot_overlay_items
    assert win.image_plot_legend.isVisible() is False


def test_fit2d_run_fit(qtbot, exp_decay_model, monkeypatch) -> None:
    t = np.linspace(0.0, 4.0, 25)
    y = np.arange(3)
    data = np.stack([((1.0 + 0.5 * idx) * np.exp(-t / 2.0)) for idx in y], axis=0)
    data = xr.DataArray(data, dims=("y", "t"), coords={"y": y, "t": t}, name="decay2d")

    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    warnings, errors = _configure_fit2d_for_tests(win, monkeypatch)

    win.y_index_spin.setValue(win.y_min_spin.value())
    win.nfev_spin.setValue(0)
    win._run_fit_2d("up")
    qtbot.waitUntil(
        lambda: all(ds is not None for ds in win._result_ds_full), timeout=10000
    )

    assert all(ds is not None for ds in win._result_ds_full)
    assert not warnings
    assert not errors

    code = win._copy_code_full()
    assert "modelfit" in code
    assert ".isel(" in code


def test_fit2d_update_data_preserves_state_and_refit(
    qtbot, exp_decay_model, monkeypatch
) -> None:
    data = _make_2d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_index_spin.setValue(1)
    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "2.0", QtCore.Qt.ItemDataRole.EditRole)
    win.y_min_spin.setValue(1)
    win.y_max_spin.setValue(2)
    win.refit_on_source_update_check.setChecked(False)
    win._last_result_ds = xr.Dataset()

    called: list[bool] = []
    monkeypatch.setattr(win, "_run_fit", lambda: called.append(True) or True)

    status = win.tool_status
    new_data = data.copy(deep=True)
    new_data.data = np.asarray(new_data.data) * 1.1
    win.update_data(new_data)

    assert win.tool_status == status
    xr.testing.assert_identical(win.tool_data, new_data)
    assert win._fit_is_current is False
    assert not called

    win._last_result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)
    newer_data = new_data.copy(deep=True)
    newer_data.data = np.asarray(newer_data.data) * 1.05
    win.update_data(newer_data)

    assert called == [True]


def test_fit2d_update_data_resizes_slice_state_and_keeps_param_sync(
    qtbot, exp_decay_model
) -> None:
    data = _make_2d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_index_spin.setValue(2)
    win.y_min_spin.setValue(1)
    win.y_max_spin.setValue(2)

    new_data = data.isel(y=slice(0, 2)).copy(deep=True)
    new_data.data = np.asarray(new_data.data) * 1.1
    win.update_data(new_data)

    assert win._current_idx == 1
    assert len(win._params_full) == 2
    assert len(win._params_from_coord_full) == 2
    assert win.y_index_spin.maximum() == 1
    xr.testing.assert_identical(win.tool_data, new_data)

    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "3.0", QtCore.Qt.ItemDataRole.EditRole)
    assert win._params_full[win._current_idx] is not None
    assert win._params_full[win._current_idx]["n0"].value == pytest.approx(3.0)


def test_fit2d_update_data_preserves_initial_params_full_for_reset_all(
    qtbot, exp_decay_model, monkeypatch
) -> None:
    data = _make_2d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    first_params = win._params.copy()
    first_params["n0"].set(value=1.0)
    second_params = win._params.copy()
    second_params["n0"].set(value=2.0)
    win._params_full = [first_params.copy(), second_params.copy(), None]
    win._initial_params_full = [
        first_params.copy(),
        second_params.copy(),
        win._params.copy(),
    ]

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.1
    win.update_data(updated)

    assert win._initial_params_full is not None
    assert win._initial_params_full[0]["n0"].value == pytest.approx(1.0)
    assert win._initial_params_full[1]["n0"].value == pytest.approx(2.0)

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    win._reset_params_all()

    assert win._params_full[0] is not None
    assert win._params_full[1] is not None
    assert win._params_full[0]["n0"].value == pytest.approx(1.0)
    assert win._params_full[1]["n0"].value == pytest.approx(2.0)


def test_fit2d_update_data_invalid_input_keeps_existing_ui(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    old_central = win.centralWidget()
    bad_data = _make_1d_data()

    with pytest.raises(ValueError, match="2D DataArray"):
        win.update_data(bad_data)

    assert win.centralWidget() is old_central
    assert old_central is not None
    assert old_central.parent() is not None
    xr.testing.assert_identical(win.tool_data, data)


def test_fit2d_update_data_returns_false_if_fit_thread_stays_alive(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

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
    old_central = win.centralWidget()

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.1

    assert win.update_data(updated) is False
    assert stuck_thread.cancel_called
    assert stuck_thread.interrupted
    assert stuck_thread.wait_timeout_ms == win.BACKGROUND_TASK_TIMEOUT_MS
    assert win.centralWidget() is old_central
    assert old_central is not None
    assert old_central.parent() is not None
    xr.testing.assert_identical(win.tool_data, data)


def test_fit2d_rebuild_paths_keep_fit_finished_receivers_constant(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    initial_receivers = signal_receiver_count(win, win.sigFitFinished, "sigFitFinished")

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.1
    win.update_data(updated)
    assert (
        signal_receiver_count(win, win.sigFitFinished, "sigFitFinished")
        == initial_receivers
    )

    win._do_transpose()
    assert (
        signal_receiver_count(win, win.sigFitFinished, "sigFitFinished")
        == initial_receivers
    )


def test_fit2d_update_data_auto_refit_after_waiting_cancelled_thread(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

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
    win._last_result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)

    started: list[bool] = []

    def _start_fit_worker(*args, **kwargs) -> bool:
        started.append(True)
        assert win._fit_thread is None
        return True

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.1

    assert win.update_data(updated) is True
    assert started == [True]
    assert old_thread.cancel_called
    assert old_thread.interrupted
    assert old_thread.wait_timeout_ms == win.BACKGROUND_TASK_TIMEOUT_MS
    assert old_thread.deleted is True


def test_fit2d_next_step_is_deferred(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    class _DummyResult:
        nfev = 1

    class _DummyResults:
        def compute(self):
            return self

        def item(self):
            return _DummyResult()

    class _DummyDataset:
        modelfit_results = _DummyResults()

    started_steps: list[int] = []

    monkeypatch.setattr(win, "_set_fit_ds", lambda result_ds, t0: win._params)
    monkeypatch.setattr(win, "_fill_params_from", lambda *args, **kwargs: None)
    monkeypatch.setattr(win, "_show_warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(win, "_show_error", lambda *args, **kwargs: None)

    def _start_fit_worker(
        fit_data,
        params,
        *,
        multi,
        step=0,
        total=0,
        on_success,
        on_timeout,
        on_error,
    ) -> bool:
        del fit_data, params, multi, total, on_timeout, on_error
        started_steps.append(step)
        win._fit_start_time = 0.0
        if step == 1:
            on_success(_DummyDataset())
            return True
        return False

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    win.y_index_spin.setValue(win.y_min_spin.value())
    win._run_fit_2d("up")

    assert started_steps == [1]
    qtbot.waitUntil(lambda: started_steps == [1, 2], timeout=1000)


def test_fit2d_cancelled_before_deferred_next_step_stops_sequence(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    class _DummyResult:
        nfev = 1

    class _DummyResults:
        def compute(self):
            return self

        def item(self):
            return _DummyResult()

    class _DummyDataset:
        modelfit_results = _DummyResults()

    started_steps: list[int] = []

    monkeypatch.setattr(win, "_set_fit_ds", lambda result_ds, t0: win._params)
    monkeypatch.setattr(win, "_fill_params_from", lambda *args, **kwargs: None)
    monkeypatch.setattr(win, "_show_warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(win, "_show_error", lambda *args, **kwargs: None)

    def _start_fit_worker(
        fit_data,
        params,
        *,
        multi,
        step=0,
        total=0,
        on_success,
        on_timeout,
        on_error,
    ) -> bool:
        del fit_data, params, multi, total, on_timeout, on_error
        started_steps.append(step)
        win._fit_start_time = 0.0
        if step == 1:
            on_success(_DummyDataset())
            return True
        return False

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    win.y_index_spin.setValue(win.y_min_spin.value())
    win._run_fit_2d("up")
    assert started_steps == [1]

    assert win._cancel_fit()
    qtbot.waitUntil(
        lambda: win._fit_2d_total == 0 and not win._fit_2d_indices,
        timeout=1000,
    )
    assert started_steps == [1]


def test_fit2d_open_saved_fit_dataset(qtbot, exp_decay_model, monkeypatch) -> None:
    t = np.linspace(0.0, 4.0, 25)
    y = np.arange(3)
    data = np.stack([((1.0 + 0.5 * idx) * np.exp(-t / 2.0)) for idx in y], axis=0)
    data = xr.DataArray(data, dims=("y", "t"), coords={"y": y, "t": t}, name="decay2d")

    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    warnings, errors = _configure_fit2d_for_tests(win, monkeypatch)

    win.y_index_spin.setValue(win.y_min_spin.value())
    win.nfev_spin.setValue(0)
    win._run_fit_2d("up")
    qtbot.waitUntil(
        lambda: all(ds is not None for ds in win._result_ds_full), timeout=10000
    )
    assert all(ds is not None for ds in win._result_ds_full)
    assert not warnings
    assert not errors

    full_ds = xr.concat(
        win._result_ds_full,
        dim=win._y_dim_name,
        data_vars="all",
        coords="minimal",
        compat="override",
        join="override",
        combine_attrs="override",
    )
    win_restored = erlab.interactive.ftool(full_ds, execute=False)
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, Fit2DTool)

    assert win_restored._fit_is_current
    assert all(ds is not None for ds in win_restored._result_ds_full)
    assert win_restored.copy_button.isEnabled()
    assert win_restored.save_button.isEnabled()
    assert win_restored.copy_full_button.isEnabled()
    assert win_restored.save_full_button.isEnabled()


def test_fit2d_full_save_and_param_plot(qtbot, exp_decay_model, monkeypatch) -> None:
    t = np.linspace(0.0, 4.0, 25)
    y = np.arange(3)
    data = np.stack([((1.0 + 0.5 * idx) * np.exp(-t / 2.0)) for idx in y], axis=0)
    data = xr.DataArray(data, dims=("y", "t"), coords={"y": y, "t": t}, name="decay2d")

    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    warnings, errors = _configure_fit2d_for_tests(win, monkeypatch)

    win.y_index_spin.setValue(win.y_min_spin.value())
    win.nfev_spin.setValue(0)
    win._run_fit_2d("up")
    qtbot.waitUntil(
        lambda: all(ds is not None for ds in win._result_ds_full), timeout=10000
    )

    combo_items = {
        win.param_plot_combo.itemText(i) for i in range(win.param_plot_combo.count())
    }
    assert set(win._model.param_names).issubset(combo_items)

    saved = {}

    @contextlib.contextmanager
    def _wait_stub(parent, message):
        yield None

    def _save_stub(ds, parent=None):
        saved["ds"] = ds

    monkeypatch.setattr(erlab.interactive.utils, "wait_dialog", _wait_stub)
    monkeypatch.setattr(erlab.interactive.utils, "save_fit_ui", _save_stub)
    win._save_fit_full()
    assert win._y_dim_name in saved["ds"].dims
    assert (
        saved["ds"].sizes[win._y_dim_name]
        == win.y_max_spin.value() - win.y_min_spin.value() + 1
    )
    assert not warnings
    assert not errors


def test_fit2d_fit_cancelled_stops_sequence(qtbot, exp_decay_model) -> None:
    data = _make_2d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._fit_2d_indices = [0, 1]
    win._fit_2d_total = 2
    win._fit_running_multi = True
    win._set_fit_running(True, multi=True, step=1, total=2)
    win._fit_cancelled()

    assert win._fit_2d_indices == []
    assert win._fit_2d_total == 0
    assert win._fit_running_multi is False


def test_fit2d_fill_params_extrapolate(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._current_idx = 2
    base_params = win._params.copy()
    param_name = "p0_center"
    base_params[param_name].set(value=1.0)
    next_params = base_params.copy()
    next_params[param_name].set(value=2.0)
    win._params_full[0] = base_params
    win._params_full[1] = next_params

    win._fill_params_from(1, mode="extrapolate")
    assert win.param_model.param_at(
        win.param_model._param_names.index(param_name)
    ).value == pytest.approx(3.0)


def test_fit2d_reset_params_all(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._result_ds_full = [
        xr.Dataset({"dummy": xr.DataArray([1])}) for _ in win._result_ds_full
    ]
    win._params_from_coord_full = [
        {"p0_center": "x"} for _ in win._params_from_coord_full
    ]

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    win._reset_params_all()
    assert all(ds is None for ds in win._result_ds_full)
    assert all(not mapping for mapping in win._params_from_coord_full)


def test_fit2d_copy_code_full_inconsistent_expr_warning(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    params1 = win._params.copy()
    params2 = win._params.copy()
    params1["p0_center"].set(expr="p0_width / 2")
    params2["p0_center"].set(expr="p0_width / 3")
    win._params["p0_center"].set(expr="p0_width / 2")

    win._params_full = [params1, params2]
    win._result_ds_full = [xr.Dataset(), xr.Dataset()]
    win.y_min_spin.setValue(0)
    win.y_max_spin.setValue(1)

    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)
    code = win._copy_code_full()
    assert code == ""
    assert warnings


def test_fit2d_copy_code_full_missing_fit_warning(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._params_full = [win._params.copy(), win._params.copy()]
    win._result_ds_full = [xr.Dataset(), None]
    win.y_min_spin.setValue(0)
    win.y_max_spin.setValue(1)

    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)
    code = win._copy_code_full()
    assert code == ""
    assert warnings


def test_fit2d_copy_code_full_inconsistent_params_warning(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    params1 = win._params.copy()
    params2 = win._params.copy()
    del params2["p0_height"]

    win._params_full = [params1, params2]
    win._result_ds_full = [xr.Dataset(), xr.Dataset()]
    win.y_min_spin.setValue(0)
    win.y_max_spin.setValue(1)

    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)
    code = win._copy_code_full()
    assert code == ""
    assert warnings


def test_fit2d_run_fit_2d_while_running(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)
    monkeypatch.setattr(win, "_fit_running", lambda: True)
    win._run_fit_2d("up")
    assert warnings


def test_fit2d_y_values_no_coord(qtbot) -> None:
    y = np.arange(3)
    data = xr.DataArray(np.ones((3, 5)), dims=("y", "x"))
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    vals = win._y_values()
    assert np.allclose(vals, y)


def test_fit2d_update_param_plot_with_results(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    params = win._params.copy()
    params["p0_center"].set(value=0.25)
    params["p0_center"].stderr = 0.1
    win._params_full = [params.copy() for _ in range(len(win._params_full))]
    win._result_ds_full = [xr.Dataset() for _ in range(len(win._result_ds_full))]

    win.param_plot_combo.setCurrentText("p0_center")
    win._update_param_plot()
    assert win.param_plot_scatter.points() is not None


def test_fit2d_param_plot_dataarray_context_actions(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    center_name = "p0_center"
    params_0 = win._params.copy()
    params_1 = win._params.copy()
    params_2 = win._params.copy()
    params_0[center_name].set(value=0.1)
    params_1[center_name].set(value=0.2)
    params_2[center_name].set(value=0.3)
    params_0[center_name].stderr = 0.01
    params_1[center_name].stderr = 0.02
    params_2[center_name].stderr = None
    win._params_full = [params_0, params_1, params_2]
    win.param_plot_combo.setCurrentText(center_name)

    values = win._param_plot_dataarray(center_name)
    stderr = win._param_plot_dataarray(center_name, stderr=True)
    np.testing.assert_allclose(values.values, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(stderr.values, [0.01, 0.02, 0.0])
    assert values.name == f"{center_name}_values"
    assert stderr.name == f"{center_name}_stderr"

    saved: list[xr.DataArray] = []
    shown: list[xr.DataArray] = []
    monkeypatch.setattr(
        win.param_plot,
        "_save_dataarray_as_hdf5",
        lambda da: saved.append(da.copy(deep=True)),
    )
    monkeypatch.setattr(
        win,
        "_show_dataarray_in_itool",
        lambda da: shown.append(da.copy(deep=True)),
    )

    win.param_plot._save_parameter_values()
    win.param_plot._save_parameter_stderr()
    win.param_plot._show_parameter_values()
    win.param_plot._show_parameter_stderr()

    assert [da.name for da in saved] == [
        f"{center_name}_values",
        f"{center_name}_stderr",
    ]
    assert [da.name for da in shown] == [
        f"{center_name}_values",
        f"{center_name}_stderr",
    ]


def test_fit2d_show_dataarray_in_itool_respects_manager_state(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    calls: list[tuple[xr.DataArray, bool | None, bool | None]] = []
    return_widget = QtWidgets.QWidget()
    qtbot.addWidget(return_widget)

    def _itool_stub(
        data: xr.DataArray, *, manager: bool | None = None, execute: bool | None = None
    ) -> QtWidgets.QWidget:
        calls.append((data, manager, execute))
        return return_widget

    monkeypatch.setattr(erlab.interactive, "itool", _itool_stub)
    monkeypatch.setattr(win, "_is_in_manager", lambda: True)

    da = xr.DataArray(np.arange(3.0), dims=("y",), coords={"y": np.arange(3)})
    win._show_dataarray_in_itool(da)
    assert calls
    assert calls[0][1] is True
    assert calls[0][2] is False


def test_fit2d_param_plot_context_actions_missing_selection(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    warnings: list[tuple[str, str]] = []
    saved: list[xr.DataArray] = []
    shown: list[xr.DataArray] = []
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )
    monkeypatch.setattr(
        win.param_plot, "_save_dataarray_as_hdf5", lambda da: saved.append(da)
    )
    monkeypatch.setattr(win, "_show_dataarray_in_itool", lambda da: shown.append(da))

    win.param_plot_combo.clear()
    win.param_plot._save_parameter_values()
    win.param_plot._show_parameter_values()
    win.param_plot._save_parameter_stderr()
    win.param_plot._show_parameter_stderr()

    assert len(warnings) == 4
    assert all(title == "No parameter selected" for title, _ in warnings)
    assert not saved
    assert not shown


def test_fit2d_param_plot_context_actions_no_data_available(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )

    win._params_full = [None for _ in win._params_full]
    win.param_plot_combo.setCurrentText("p0_center")
    win.param_plot._show_parameter_values()

    assert warnings
    assert warnings[-1][0] == "No data available"


def test_fit2d_param_plot_save_dataarray_as_hdf5(
    qtbot, accept_dialog, tmp_path
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    filename = tmp_path / "param_values.h5"
    da = xr.DataArray(np.arange(3.0), dims=("y",), coords={"y": np.arange(3)})

    def _go_to_file(dialog: QtWidgets.QFileDialog) -> None:
        dialog.setDirectory(str(tmp_path))
        dialog.selectFile(str(filename))
        focused = dialog.focusWidget()
        if isinstance(focused, QtWidgets.QLineEdit):
            focused.setText("param_values.h5")

    accept_dialog(
        lambda: win.param_plot._save_dataarray_as_hdf5(da), pre_call=_go_to_file
    )
    loaded = xr.load_dataarray(filename, engine="h5netcdf")
    xr.testing.assert_identical(da, loaded)


def test_fit2d_param_plot_save_dataarray_as_hdf5_branches(
    qtbot, monkeypatch, tmp_path
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    real_dialog = QtWidgets.QFileDialog
    captured_dirs: list[str] = []

    class _DialogStub:
        AcceptMode = real_dialog.AcceptMode
        FileMode = real_dialog.FileMode
        Option = real_dialog.Option

        def __init__(self, *args, **kwargs) -> None:
            self._selected = [str(tmp_path / "unused.h5")]

        def setAcceptMode(self, mode) -> None:
            self._accept_mode = mode

        def setFileMode(self, mode) -> None:
            self._file_mode = mode

        def setNameFilters(self, name_filters) -> None:
            self._name_filters = name_filters

        def setDefaultSuffix(self, suffix) -> None:
            self._suffix = suffix

        def setOption(self, *args, **kwargs) -> None:
            self._option = (args, kwargs)

        def setDirectory(self, directory: str) -> None:
            captured_dirs.append(directory)

        def exec(self) -> bool:
            return False

        def selectedFiles(self) -> list[str]:
            return self._selected

    monkeypatch.setattr(QtWidgets, "QFileDialog", _DialogStub)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)

    monkeypatch.setattr(pg.PlotItem, "lastFileDir", str(tmp_path), raising=False)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "_get_recent_directory",
        lambda: str(tmp_path / "recent"),
    )

    win.param_plot._save_dataarray_as_hdf5(xr.DataArray([1.0], name=""))
    win.param_plot._save_dataarray_as_hdf5(xr.DataArray([2.0], name=5))
    assert captured_dirs[-2].endswith("data.h5")
    assert captured_dirs[-1].endswith("5.h5")

    monkeypatch.setattr(pg.PlotItem, "lastFileDir", "", raising=False)
    win.param_plot._save_dataarray_as_hdf5(xr.DataArray([3.0], name="named"))
    assert captured_dirs[-1] == str(tmp_path / "recent" / "named.h5")

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_get_recent_directory", lambda: ""
    )
    win.param_plot._save_dataarray_as_hdf5(xr.DataArray([4.0], name="cwdname"))
    assert captured_dirs[-1] == os.path.join(os.getcwd(), "cwdname.h5")


def test_fit2d_is_in_manager_false_when_no_manager(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    monkeypatch.setattr(erlab.interactive.imagetool.manager, "_manager_instance", None)
    assert win._is_in_manager() is False


def test_fit2d_is_in_manager_node_lookup(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    class _Manager:
        def __init__(self, managed: object | None) -> None:
            self._managed = managed

        def _node_uid_from_window(self, widget) -> str | None:
            if widget is self._managed:
                return "y"
            return None

    manager = _Manager(win)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    assert win._is_in_manager() is True

    manager = _Manager(None)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    assert win._is_in_manager() is False


def test_fit2d_show_dataarray_in_itool_non_widget_return(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    monkeypatch.setattr(erlab.interactive, "itool", lambda *args, **kwargs: None)
    da = xr.DataArray(np.arange(3.0), dims=("y",), coords={"y": np.arange(3)})
    win._show_dataarray_in_itool(da)
    assert not hasattr(win, "_itool")


def test_fit2d_y_range_slice(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_min_spin.setValue(0)
    win.y_max_spin.setValue(1)
    sl = win._y_range_slice()
    assert sl.start == 0
    assert sl.stop == 2


def test_fit2d_y_bounds_descending_coords(qtbot) -> None:
    data = _make_2d_data().isel(y=slice(None, None, -1))
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_min_spin.setValue(0)
    win.y_max_spin.setValue(1)

    assert win.y_min_line.bounds()[0] <= win.y_min_line.bounds()[1]
    assert win.y_max_line.bounds()[0] <= win.y_max_line.bounds()[1]

    y_vals = data["y"].values
    assert win.y_min_line.value() == pytest.approx(y_vals[0])
    assert win.y_max_line.value() == pytest.approx(y_vals[1])


def test_fit2d_param_plot_options_update(qtbot, exp_decay_model) -> None:
    data = _make_2d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.set_model(win._make_model_from_choice("MultiPeakModel"))
    combo_items = {
        win.param_plot_combo.itemText(i) for i in range(win.param_plot_combo.count())
    }
    assert set(win._model.param_names).issubset(combo_items)


def test_fit2d_init_validation_errors(qtbot, exp_decay_model) -> None:
    data_1d = _make_1d_data()
    with pytest.raises(ValueError, match="`data` must be a 2D DataArray"):
        Fit2DTool(data_1d)  # type: ignore[arg-type]

    data = _make_2d_data()
    params_da = {"n0": xr.DataArray([1.0], dims=("z",))}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Some parameters are dependent on dimension `z`, which does not match the "
            "independent dimension of the data (`y`)."
        ),
    ):
        Fit2DTool(data, params=params_da)  # type: ignore[arg-type]

    params_da = {"n0": xr.DataArray([1.0, 2.0], dims=("y",))}
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of parameter sets does not match the size of the "
            "independent dimension of the data."
        ),
    ):
        Fit2DTool(data, params=params_da)  # type: ignore[arg-type]


def test_fit2d_tool_data_and_refresh_multipeak_model(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    assert win.tool_data is win._data_full

    other_idx = 0 if win._current_idx != 0 else 1
    win._params_full[other_idx] = win._params.copy()
    win._params_from_coord_full[other_idx] = {"missing_param": "x"}
    win._refresh_multipeak_model()
    assert win._params_full[other_idx] is not None
    assert "missing_param" not in win._params_from_coord_full[other_idx]


def test_fit2d_refresh_contents_from_index_updates(qtbot) -> None:
    data = _make_2d_data().assign_coords(temp=300.0)
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    idx = win._current_idx
    win._initial_params_full = [win._params.copy() for _ in win._params_full]
    win._params_full[idx] = win._params.copy()
    win._params_from_coord_full[idx] = {"p0_center": "temp"}

    class _Result:
        nfev = 1
        redchi = 1.0
        rsquared = None
        aic = 1.0
        bic = 2.0

    win._result_ds_full[idx] = xr.Dataset(
        {"modelfit_results": xr.DataArray(_Result(), dims=())}
    )
    win._refresh_contents_from_index()
    assert win._params["p0_center"].value == pytest.approx(300.0)


def test_fit2d_fill_params_none_and_invalid(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._current_idx = 1
    original = win._params.copy()
    win._params_full[1] = original.copy()
    win._fill_params_from(0, mode="none")
    assert win._params_full[1]["p0_center"].value == original["p0_center"].value

    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)
    win._fill_params_from(-1)
    assert warnings


def test_fit2d_set_model_merge_params_across_indices(qtbot) -> None:
    """Test that set_model with merge_params=True merges params at all indices."""
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    # Set specific param values at different indices
    win.y_index_spin.setValue(0)
    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "5.0", QtCore.Qt.ItemDataRole.EditRole)

    win.y_index_spin.setValue(1)
    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "10.0", QtCore.Qt.ItemDataRole.EditRole)

    # Change model with merge_params=True
    new_model = win._make_model_from_choice("MultiPeakModel")
    win.set_model(new_model, merge_params=True)

    # Verify params were merged at all indices
    win.y_index_spin.setValue(0)
    assert win.param_model.param_at(0).value == pytest.approx(5.0)

    win.y_index_spin.setValue(1)
    assert win.param_model.param_at(0).value == pytest.approx(10.0)


def test_fit2d_refresh_multipeak_model_merges_params(qtbot) -> None:
    """Test that _refresh_multipeak_model merges params across all indices."""
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    # Set params at current index and another index
    win.y_index_spin.setValue(0)
    center_row = win.param_model._param_names.index("p0_center")
    index = win.param_model.index(center_row, 1)
    assert win.param_model.setData(index, "0.3", QtCore.Qt.ItemDataRole.EditRole)
    win._update_params_full()

    win.y_index_spin.setValue(1)
    index = win.param_model.index(center_row, 1)
    assert win.param_model.setData(index, "0.5", QtCore.Qt.ItemDataRole.EditRole)
    win._update_params_full()

    # Change number of peaks (triggers _refresh_multipeak_model)
    win.npeaks_spin.setValue(2)

    # Both indices should have preserved their p0_center values
    win.y_index_spin.setValue(0)
    center_row = win.param_model._param_names.index("p0_center")
    assert win.param_model.param_at(center_row).value == pytest.approx(0.3)

    win.y_index_spin.setValue(1)
    assert win.param_model.param_at(center_row).value == pytest.approx(0.5)
