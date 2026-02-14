import lmfit
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive import _fit1d as fit1d
from erlab.interactive._fit1d import (
    Fit1DTool,
    _ParameterEditDelegate,
    _ParameterTableModel,
)


def _make_1d_data() -> xr.DataArray:
    x = np.linspace(-1.0, 1.0, 11)
    data = np.exp(-(x**2))
    return xr.DataArray(data, dims=("x",), coords={"x": x}, name="spec")


def test_fit1d_fit_domain_descending_coords(qtbot) -> None:
    x = np.linspace(1.0, -1.0, 21)
    data = xr.DataArray(np.exp(-(x**2)), dims=("x",), coords={"x": x})
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win.domain_min_spin.setValue(-0.5)
    win.domain_max_spin.setValue(0.5)

    domain = win._fit_domain()
    assert domain == pytest.approx((0.5, -0.5))

    fit_data = win._fit_data_raw()
    assert fit_data.size > 0
    assert fit_data["x"].values[0] == pytest.approx(0.5)
    assert fit_data["x"].values[-1] == pytest.approx(-0.5)


def test_ftool_1d_param_edit_and_state(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "1.5", QtCore.Qt.ItemDataRole.EditRole)
    assert win.param_model.param_at(0).value == pytest.approx(1.5)

    vary_index = win.param_model.index(0, 5)
    assert win.param_model.setData(
        vary_index, QtCore.Qt.CheckState.Unchecked, QtCore.Qt.ItemDataRole.EditRole
    )
    assert win.param_model.param_at(0).vary is False

    win.normalize_check.setChecked(True)
    win.domain_min_spin.setValue(-0.5)
    win.domain_max_spin.setValue(0.5)

    status = win.tool_status
    win_restored = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win_restored)
    win_restored.tool_status = status
    assert win_restored.tool_status.model_dump() == status.model_dump()

    code = win.copy_code()
    assert "modelfit" in code
    assert "MultiPeakModel" in code


def test_fit1d_undo_redo(qtbot, exp_decay_model) -> None:
    t = np.linspace(0.0, 2.0, 11)
    data = xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="decay")
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)

    index = win.param_model.index(0, 1)
    start_value = win.param_model.param_at(0).value
    updated_value = start_value + 0.5

    assert win.param_model.setData(
        index, f"{updated_value}", QtCore.Qt.ItemDataRole.EditRole
    )
    assert win.undoable

    win.undo()
    assert win.param_model.param_at(0).value == pytest.approx(start_value)
    assert win.redoable

    win.redo()
    assert win.param_model.param_at(0).value == pytest.approx(updated_value)


def test_fit1d_run_fit(qtbot, exp_decay_model) -> None:
    t = np.linspace(0.0, 4.0, 25)
    data = xr.DataArray(
        3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
    )
    params = exp_decay_model.make_params(n0=2.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)

    assert win._run_fit()
    qtbot.waitUntil(lambda: win._last_result_ds is not None, timeout=10000)
    assert win._last_result_ds is not None
    assert win._fit_is_current

    result = win._last_result_ds.modelfit_results.compute().item()
    assert result.params["n0"].value == pytest.approx(3.0, rel=1e-2)
    assert result.params["tau"].value == pytest.approx(2.0, rel=1e-2)


def test_fit1d_open_saved_fit_dataset(qtbot, exp_decay_model) -> None:
    t = np.linspace(0.0, 4.0, 25)
    data = xr.DataArray(
        3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
    )
    params = exp_decay_model.make_params(n0=2.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)

    assert win._run_fit()
    qtbot.waitUntil(lambda: win._last_result_ds is not None, timeout=10000)
    fit_ds = win._last_result_ds
    assert fit_ds is not None

    win_restored = erlab.interactive.ftool(fit_ds, execute=False)
    qtbot.addWidget(win_restored)

    assert win_restored._last_result_ds is not None
    assert win_restored._fit_is_current
    assert win_restored.save_button.isEnabled()
    assert win_restored.copy_button.isEnabled()
    assert isinstance(win_restored._model, type(exp_decay_model))


def test_parameter_table_model_and_delegate(qtbot) -> None:
    params = lmfit.Parameters()
    params.add("amp", value=1.0, min=-1.0, max=2.0, vary=True)
    params.add("expr_param", value=2.0, expr="2*amp")
    params_from_coord = {"amp": "temp"}
    model = _ParameterTableModel(params, params_from_coord)

    assert model.rowCount() == 2
    assert model.columnCount() == 6
    assert (
        model.headerData(0, QtCore.Qt.Orientation.Horizontal) == model._COLUMN_NAMES[0]
    )

    value_index = model.index(0, 1)
    assert model.setData(value_index, "1.5", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].value == pytest.approx(1.5)

    min_index = model.index(0, 3)
    max_index = model.index(0, 4)
    assert model.setData(min_index, "-inf", QtCore.Qt.ItemDataRole.EditRole)
    assert model.setData(max_index, "inf", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].min == -np.inf
    assert params["amp"].max == np.inf

    vary_index = model.index(0, 5)
    assert model.setData(
        vary_index, QtCore.Qt.CheckState.Unchecked, QtCore.Qt.ItemDataRole.EditRole
    )
    assert params["amp"].vary is False
    assert (
        model.setData(value_index, "not-a-number", QtCore.Qt.ItemDataRole.EditRole)
        is False
    )
    assert params["amp"].value == pytest.approx(1.5)

    expr_index = model.index(1, 1)
    flags = model.flags(expr_index)
    assert not (flags & QtCore.Qt.ItemFlag.ItemIsEditable)
    assert model.data(model.index(1, 5), QtCore.Qt.ItemDataRole.CheckStateRole) is None

    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    delegate = _ParameterEditDelegate(model, parent)
    editor = delegate.createEditor(parent, None, value_index)
    assert isinstance(editor, QtWidgets.QLineEdit)
    delegate.setEditorData(editor, value_index)
    assert editor.text() == model.edit_value_string(0, 1)
    editor.setText("2.5")
    delegate.setModelData(editor, model, value_index)
    assert params["amp"].value == pytest.approx(2.5)


def test_fit1d_guess_components_and_slider(qtbot) -> None:
    x = np.linspace(-1.0, 1.0, 101)
    data = np.exp(-((x - 0.2) ** 2) / (2.0 * 0.3**2))
    darr = xr.DataArray(data, dims=("x",), coords={"x": x, "temp": 300.0}, name="gauss")
    win = erlab.interactive.ftool(darr, execute=False)
    qtbot.addWidget(win)

    win._guess_params()
    assert win._params["p0_center"].value == pytest.approx(0.2, abs=0.05)

    win.components_check.setChecked(True)
    win._update_fit_curve()
    assert win.component_curves
    assert any(curve.isVisible() for curve in win.component_curves.values())

    center_row = win.param_model._param_names.index("p0_center")
    win.param_view.selectRow(center_row)
    param = win.param_model.param_at(center_row)
    initial_value = float(param.value)
    vmin, vmax, _ = win._slider_range(param.value, param)
    win.param_value_slider.setValue(int(0.8 * win._slider_steps))
    assert vmin <= param.value <= vmax
    assert param.value != pytest.approx(initial_value)

    win.param_mode_combo.setCurrentText("Take from 'temp'")
    assert win._params_from_coord[param.name] == "temp"
    assert param.value == pytest.approx(300.0)
    assert win.param_value_spin.isReadOnly()
    win.param_mode_combo.setCurrentText("Manual")
    assert param.name not in win._params_from_coord


def test_fit1d_multiple_fits_and_save(qtbot, exp_decay_model, monkeypatch) -> None:
    t = np.linspace(0.0, 4.0, 25)
    data = xr.DataArray(
        3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
    )
    params = exp_decay_model.make_params(n0=2.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)

    win._run_fit_multiple(2)
    qtbot.waitUntil(
        lambda: (not win._fit_running()) and (win._fit_multi_total is None),
        timeout=10000,
    )
    assert win._fit_is_current
    assert win.fit_multi_button.text() == "Fit ×20"

    saved = {}

    def _save_stub(ds, parent=None):
        saved["ds"] = ds
        saved["parent"] = parent

    monkeypatch.setattr(erlab.interactive.utils, "save_fit_ui", _save_stub)
    win._save_fit()
    assert saved["ds"] is not None


def test_fit1d_user_and_file_models(qtbot, tmp_path) -> None:
    x = np.linspace(0.0, 1.0, 5)
    data = xr.DataArray(np.zeros_like(x), dims=("x",), coords={"x": x}, name="line")

    def _linear(x, slope=1.0, offset=0.0):
        return slope * x + offset

    user_model = lmfit.Model(_linear)
    win = erlab.interactive.ftool(data, model=user_model, execute=False)
    qtbot.addWidget(win)
    assert win.model_combo.currentData(role=QtCore.Qt.ItemDataRole.UserRole) == "__user"

    if not hasattr(lmfit.model, "save_model"):
        pytest.skip("lmfit.model.save_model is unavailable in this environment.")

    model_path = tmp_path / "model.sav"
    lmfit.model.save_model(user_model, model_path)
    loaded_model = lmfit.model.load_model(model_path)
    win.set_model(loaded_model, model_load_path=str(model_path))
    assert win.model_combo.currentData(role=QtCore.Qt.ItemDataRole.UserRole) == "__file"
    code = win.copy_code()
    assert "load_model" in code
    assert str(model_path) in code


def test_fit1d_expression_editing_and_validation(qtbot, monkeypatch) -> None:
    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(np.exp(-(x**2)), dims=("x",), coords={"x": x}, name="spec")
    model = lmfit.models.VoigtModel()
    win = erlab.interactive.ftool(data, model=model, execute=False)
    qtbot.addWidget(win)

    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)

    gamma_row = win.param_model._param_names.index("gamma")
    gamma_param = win.param_model.param_at(gamma_row)
    original_expr = gamma_param.expr

    win._set_param_expr(gamma_row, "sigma**")
    assert gamma_param.expr == original_expr
    assert warnings

    warnings.clear()
    win._set_param_expr(gamma_row, "2*sigma")
    assert gamma_param.expr == "2*sigma"

    win._clear_param_expr(gamma_row)
    assert gamma_param.expr is None

    fwhm_row = win.param_model._param_names.index("fwhm")
    fwhm_param = win.param_model.param_at(fwhm_row)
    old_fwhm_expr = fwhm_param.expr
    win._set_param_expr(fwhm_row, "sigma")
    assert fwhm_param.expr == old_fwhm_expr


def test_fit1d_peak_lines_follow_components(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win.components_check.setChecked(True)
    win._update_fit_curve()

    assert win._peak_lines
    assert isinstance(win._peak_lines[0], fit1d._PeakPositionLine)

    key = win._model._prefix or win._model._name
    comp_name = f"{key}_p0"
    curve = win.component_curves[comp_name]
    curve_pen = curve.opts.get("pen")
    line = win._peak_lines[0]
    line_pen = line.pen() if callable(getattr(line, "pen", None)) else line.pen
    assert curve_pen.color() == line_pen.color()


def test_fit1d_prompt_expr_lock_and_apply(qtbot, monkeypatch) -> None:
    x = np.linspace(-1.0, 1.0, 21)
    data = xr.DataArray(np.exp(-(x**2)), dims=("x",), coords={"x": x}, name="spec")
    model = lmfit.models.VoigtModel()
    win = erlab.interactive.ftool(data, model=model, execute=False)
    qtbot.addWidget(win)

    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)

    fwhm_row = win.param_model._param_names.index("fwhm")
    fwhm_param = win.param_model.param_at(fwhm_row)
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: ("sigma", True)
    )
    win._prompt_param_expr(fwhm_row)
    assert fwhm_param.expr is not None
    assert warnings

    gamma_row = win.param_model._param_names.index("gamma")
    gamma_param = win.param_model.param_at(gamma_row)
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: ("2*sigma", True)
    )
    win._prompt_param_expr(gamma_row)
    assert gamma_param.expr == "2*sigma"

    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: ("", True)
    )
    win._prompt_param_expr(gamma_row)
    assert gamma_param.expr is None


def test_fit1d_slider_messages_for_expr_and_nonfinite(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    center_row = win.param_model._param_names.index("p0_center")
    param = win.param_model.param_at(center_row)
    param.expr = "1"
    win.param_view.selectRow(center_row)
    win._on_param_selected(win.param_model.index(center_row, 0), QtCore.QModelIndex())
    win._refresh_slider_from_model()
    assert "expr:" in win.expr_label.text()

    param.expr = None
    param.value = np.inf
    win._refresh_slider_from_model()
    assert "value:" in win.expr_label.text()


def test_fit1d_fit_cancelled_clears_multi(qtbot, exp_decay_model) -> None:
    t = np.linspace(0.0, 2.0, 11)
    data = xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="decay")
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)

    win._fit_multi_total = 3
    win._fit_running_multi = True
    win._set_fit_running(True, multi=True, step=1, total=3)
    win._fit_cancelled()

    assert win._fit_multi_total is None
    assert win._fit_running_multi is False
    assert not win.cancel_fit_button.isEnabled()


def test_fit1d_set_param_expr_clears_coord_and_vary(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    center_row = win.param_model._param_names.index("p0_center")
    param = win.param_model.param_at(center_row)
    win._params_from_coord[param.name] = "x"
    win._set_param_expr(center_row, "1")

    assert param.expr == "1"
    assert param.vary is False
    assert param.name not in win._params_from_coord


def test_fit1d_peak_lines_clear_when_components_off(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win.components_check.setChecked(True)
    win._update_fit_curve()
    assert win._peak_lines

    win.components_check.setChecked(False)
    win._update_peak_lines(win._x_values())
    assert not win._peak_lines


def test_fit1d_model_controls_and_stats(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win.set_model(win._make_model_from_choice("PolynomialModel"), merge_params=True)
    assert win.polynomial_group.isVisible()
    win.poly_degree_spin.setValue(3)
    win._refresh_polynomial_model()
    assert isinstance(win._model, erlab.analysis.fit.models.PolynomialModel)

    win.set_model(win._make_model_from_choice("MultiPeakModel"), merge_params=True)
    win._sync_multipeak_controls()
    assert isinstance(win._model, erlab.analysis.fit.models.MultiPeakModel)

    class _Result:
        nfev = 20
        redchi = 1.0
        rsquared = 0.9
        aic = 1.0
        bic = 2.0

    win.nfev_spin.setValue(10)
    win._set_fit_stats(_Result(), elapsed=1.23)
    assert "1.23" in win.elapsed_value.text()

    win._set_fit_stats(None)
    assert win.elapsed_value.text() == "—"


def test_fit1d_fit_error_paths(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    errors: list[tuple[str, str]] = []

    def _error(title: str, text: str, detailed_text: str | None = None) -> None:
        errors.append((title, text))

    monkeypatch.setattr(win, "_show_error", _error)

    win._fit_timed_out(0.0)
    assert errors

    errors.clear()
    win._fit_errored("trace")
    assert errors


def test_parameter_table_formatting_helpers() -> None:
    params = lmfit.Parameters()
    params.add("amp", value=1.0)
    params["amp"].stderr = 0.1
    model = _ParameterTableModel(params, {})
    tooltip = model._param_tooltip(params["amp"])
    assert "±" in tooltip

    assert model._parse_bound("", default=-np.inf) == -np.inf
    assert model._parse_bound("inf", default=-np.inf) == np.inf
    assert model._parse_bound("-inf", default=np.inf) == -np.inf
    assert model._format_value(np.nan) == "nan"
    assert model._format_value(np.inf) == "inf"
    assert model._format_value(-np.inf) == "-inf"
    assert model._format_scientific(np.nan) == "nan"
    assert model._format_scientific(np.inf) == "inf"
    assert model._format_scientific(-np.inf) == "-inf"


def test_fit1d_nonfinite_params_update(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win._params["p0_center"].set(value=np.inf)
    win._update_fit_curve()
    assert win._last_fit_y is None
    assert win._last_residual is None


def test_fit1d_set_slider_position_invalid(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    center_row = win.param_model._param_names.index("p0_center")
    win.param_view.selectRow(center_row)
    win._on_param_selected(win.param_model.index(center_row, 0), QtCore.QModelIndex())
    win._set_slider_position(np.inf, 0.0, 1.0)
    assert not win.param_value_slider.isEnabled()


def test_fit1d_set_param_expr_without_asteval(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win._params._asteval = None
    center_row = win.param_model._param_names.index("p0_center")
    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)
    win._set_param_expr(center_row, "1")
    assert win.param_model.param_at(center_row).expr == "1"
    assert not warnings


def test_fit1d_set_model_resets_param_coord(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win._params_from_coord["p0_center"] = "x"
    win.set_model(
        win._make_model_from_choice("MultiPeakModel"),
        reset_params_from_coord=True,
    )
    assert win._params_from_coord == {}


def test_fit1d_slider_range_with_bounds(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    row = win.param_model._param_names.index("p0_center")
    param = win.param_model.param_at(row)
    param.set(min=0.1, max=0.2)
    vmin, vmax, width = win._slider_range(param.value, param)
    assert vmin <= vmax
    assert vmin <= param.max
    assert vmax >= param.min
    assert width > 0


def test_fit1d_component_legend_toggle(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win.components_check.setChecked(True)
    win._update_fit_curve()
    assert win.component_curves
    assert win.legend.items

    win.components_check.setChecked(False)
    win._update_component_curves(np.array([]))
    assert not any(curve.isVisible() for curve in win.component_curves.values())


def test_fit1d_queue_fit_action_no_thread(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    called: list[bool] = []

    def _action() -> None:
        called.append(True)

    win._fit_thread = None
    win._queue_fit_action(_action)
    assert called


def test_fit1d_param_is_func_arg_prefix(qtbot) -> None:
    data = _make_1d_data()
    model = lmfit.models.VoigtModel(prefix="v_")
    win = erlab.interactive.ftool(data, model=model, execute=False)
    qtbot.addWidget(win)

    row = win.param_model._param_names.index("v_gamma")
    param = win.param_model.param_at(row)
    assert win._param_is_func_arg(param)


def test_fit1d_fit_cancelled_single(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win._fit_running_multi = False
    win._set_fit_running(True, multi=False)
    win._fit_cancelled()
    assert win.fit_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


def test_fit1d_cancel_fit_waits_for_thread(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
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
            return True

    dummy_thread = _DummyThread()
    win._fit_thread = dummy_thread  # type: ignore[assignment]

    assert win._cancel_fit(wait=True)
    assert dummy_thread.cancel_called
    assert dummy_thread.interrupted
    assert dummy_thread.wait_timeout_ms == 5000


def test_fit1d_close_event_ignored_if_thread_does_not_stop(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _StuckThread:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

        def isRunning(self) -> bool:
            return True

        def requestInterruption(self) -> None:
            return

        def wait(self, timeout_ms: int) -> bool:
            return False

    stuck_thread = _StuckThread()
    win._fit_thread = stuck_thread  # type: ignore[assignment]

    event = QtGui.QCloseEvent()
    assert event.isAccepted()
    win.closeEvent(event)
    assert not event.isAccepted()
    assert stuck_thread.cancel_called


def test_fit1d_slider_drag_updates_value(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    row = win.param_model._param_names.index("p0_center")
    win.param_view.selectRow(row)
    win._on_param_selected(win.param_model.index(row, 0), QtCore.QModelIndex())
    win._slider_dragging = True
    win._slider_drag_range = (0.0, 1.0)
    win._on_slider_moved(win._slider_steps // 2)
    assert 0.0 <= win.param_model.param_at(row).value <= 1.0


def test_fit1d_param_is_editable_false(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    row = win.param_model._param_names.index("p0_center")
    param = win.param_model.param_at(row)
    win._params_from_coord[param.name] = "x"
    assert not win._param_is_editable(param.name)
    del win._params_from_coord[param.name]
    param.expr = "1"
    assert not win._param_is_editable(param.name)


def test_fit1d_param_expr_from_hint(qtbot) -> None:
    data = _make_1d_data()
    model = lmfit.models.VoigtModel()
    win = erlab.interactive.ftool(data, model=model, execute=False)
    qtbot.addWidget(win)

    row = win.param_model._param_names.index("fwhm")
    param = win.param_model.param_at(row)
    assert win._param_expr_from_hint(param)


def test_python_code_editor_indent_unindent(qtbot) -> None:
    editor = fit1d._PythonCodeEditor()
    qtbot.addWidget(editor)
    editor.setPlainText("a\nb")
    cursor = editor.textCursor()
    cursor.select(QtGui.QTextCursor.SelectionType.Document)
    editor._indent(cursor, 2)
    assert editor.toPlainText().startswith("  a")
    cursor.select(QtGui.QTextCursor.SelectionType.Document)
    editor._unindent(cursor, 2)
    assert editor.toPlainText().startswith("a")


def test_python_code_editor_keypress_tab_and_backtab(qtbot) -> None:
    editor = fit1d._PythonCodeEditor()
    qtbot.addWidget(editor)
    editor.setPlainText("x")
    cursor = editor.textCursor()
    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
    editor.setTextCursor(cursor)

    tab_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Tab,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    editor.keyPressEvent(tab_event)
    assert editor.toPlainText().startswith(" " * editor.TAB_SPACES)

    backtab_event = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Backtab,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    editor.keyPressEvent(backtab_event)
    assert editor.toPlainText().startswith("x")


def test_expression_init_script_dialog_get_script(qtbot) -> None:
    dialog = fit1d._ExpressionInitScriptDialog()
    qtbot.addWidget(dialog)
    dialog.text_edit.setPlainText("x = 1")
    assert dialog.get_script() == "x = 1"
    dialog.text_edit.setPlainText("   ")
    assert dialog.get_script() is None


def test_fit_worker_run_and_cancel(qtbot, exp_decay_model) -> None:
    t = np.linspace(0.0, 1.0, 11)
    data = xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="decay")
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    worker = fit1d._FitWorker(
        data,
        "t",
        exp_decay_model,
        params,
        max_nfev=5,
        method="least_squares",
        timeout=1.0,
    )

    signals: dict[str, bool] = {"finished": False}

    def _on_finished(_result):
        signals["finished"] = True

    worker.sigFinished.connect(_on_finished)
    worker.cancel()
    worker.run()
    assert signals["finished"]


def test_fit_worker_timeout_and_cancelled(qtbot, exp_decay_model, monkeypatch) -> None:
    t = np.linspace(0.0, 1.0, 11)
    data = xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="decay")
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)

    worker = fit1d._FitWorker(
        data,
        "t",
        exp_decay_model,
        params,
        max_nfev=5,
        method="least_squares",
        timeout=1.0,
    )

    events: dict[str, bool] = {"timed_out": False, "cancelled": False}

    def _modelfit_cancel(*_args, **kwargs):
        worker._cancel.set()
        kwargs["iter_cb"]()
        raise RuntimeError("cancel")

    def _modelfit_timeout(*_args, **kwargs):
        kwargs["iter_cb"]()
        raise RuntimeError("timeout")

    def _on_timeout():
        events["timed_out"] = True

    def _on_cancel():
        events["cancelled"] = True

    worker.sigTimedOut.connect(_on_timeout)
    worker.sigCancelled.connect(_on_cancel)

    worker._cancel.set()
    monkeypatch.setattr(data.xlm, "modelfit", _modelfit_cancel)
    worker.run()
    assert events["cancelled"]

    events["cancelled"] = False
    events["timed_out"] = False
    timer = iter([0.0, 2.0])
    monkeypatch.setattr(fit1d.time, "perf_counter", lambda: next(timer))
    monkeypatch.setattr(data.xlm, "modelfit", _modelfit_timeout)
    worker.run()
    assert events["timed_out"]


def test_fit_worker_runtimeerror_interruption_treated_as_cancelled(
    qtbot, exp_decay_model, monkeypatch
) -> None:
    t = np.linspace(0.0, 1.0, 11)
    data = xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="decay")
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)

    class _ThreadWithInterruptedWrapper(fit1d._FitWorker):
        def isInterruptionRequested(self) -> bool:
            raise RuntimeError("thread wrapper deleted")

    worker = _ThreadWithInterruptedWrapper(
        data,
        "t",
        exp_decay_model,
        params,
        max_nfev=5,
        method="least_squares",
        timeout=1.0,
    )

    cancelled = {"value": False}

    def _modelfit(*_args, **kwargs):
        kwargs["iter_cb"]()
        raise RuntimeError("cancel")

    worker.sigCancelled.connect(lambda: cancelled.__setitem__("value", True))
    monkeypatch.setattr(data.xlm, "modelfit", _modelfit)

    worker.run()
    assert cancelled["value"]


def test_fit_worker_loads_result_before_emit(
    qtbot, exp_decay_model, monkeypatch
) -> None:
    t = np.linspace(0.0, 1.0, 11)
    data = xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="decay")
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)

    worker = fit1d._FitWorker(
        data,
        "t",
        exp_decay_model,
        params,
        max_nfev=5,
        method="least_squares",
        timeout=1.0,
    )

    class _DummyResult:
        def __init__(self) -> None:
            self.loaded = False

        def load(self):
            self.loaded = True
            return self

    dummy = _DummyResult()

    def _modelfit(*_args, **_kwargs):
        return dummy

    finished: dict[str, object | None] = {"result": None}

    def _on_finished(result) -> None:
        finished["result"] = result

    worker.sigFinished.connect(_on_finished)
    monkeypatch.setattr(data.xlm, "modelfit", _modelfit)

    worker.run()

    assert dummy.loaded
    assert finished["result"] is dummy


def test_fit1d_finalize_fit_thread_cancelled_deletes_thread(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self.deleted = False

        def deleteLater(self) -> None:
            self.deleted = True

    thread = _DummyThread()
    cancelled = {"value": False}

    win._fit_thread = thread  # type: ignore[assignment]
    win._pending_fit_action = None
    win._fit_cancel_requested = True
    win._fit_cancelled = lambda: cancelled.__setitem__("value", True)  # type: ignore[method-assign]

    win._finalize_fit_thread()

    assert cancelled["value"]
    assert thread.deleted
    assert win._fit_thread is None
    assert not win._fit_cancel_requested


def test_snap_cursor_line_value(qtbot) -> None:
    line = fit1d._SnapCursorLine(pos=1.5, angle=90, movable=True)
    qtbot.addWidget(QtWidgets.QWidget())
    assert line.value() == pytest.approx(1.5)


def test_parameter_edit_delegate_non_line_edit(qtbot) -> None:
    params = lmfit.Parameters()
    params.add("amp", value=1.0)
    model = _ParameterTableModel(params, {})
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    delegate = _ParameterEditDelegate(model, parent)

    index = model.index(0, 1)
    editor = QtWidgets.QSpinBox(parent)
    delegate.setEditorData(editor, index)
    delegate.setModelData(editor, model, index)


def test_parameter_table_model_params_property() -> None:
    params = lmfit.Parameters()
    params.add("amp", value=1.0)
    model = _ParameterTableModel(params, {})
    assert model.params is params


def test_peak_position_line_refresh_hide(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win._params["p0_center"].set(value=np.nan)
    win.components_check.setChecked(True)
    win._update_fit_curve()
    line = win._peak_lines[0]
    line.refresh_pos()
    assert not line.isVisible()


def test_parameter_table_edit_value_strings_and_tooltip() -> None:
    params = lmfit.Parameters()
    params.add("amp", value=1.0, min=-1.0, max=2.0)
    params.add("expr_param", value=2.0, expr="2*amp")
    model = _ParameterTableModel(params, {})

    assert model.param_name(0) == "amp"
    assert model.edit_value_string(0, 2) == ""
    assert model.edit_value_string(0, 3) == "-1"
    assert model.edit_value_string(0, 4) == "2"

    tooltip = model.data(model.index(1, 0), QtCore.Qt.ItemDataRole.ToolTipRole)
    assert "expr:" in tooltip


def test_fit1d_model_choice_changed_paths(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    warnings: list[tuple[str, str]] = []

    def _error(title: str, text: str, detailed_text: str | None = None) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_error", _error)

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", "")
    )
    file_idx = win.model_combo.findData("__file")
    with QtCore.QSignalBlocker(win.model_combo):
        win.model_combo.setCurrentIndex(file_idx)
    win._on_model_choice_changed(file_idx)

    user_idx = win.model_combo.findData("__user")
    with QtCore.QSignalBlocker(win.model_combo):
        win.model_combo.setCurrentIndex(user_idx)
    win._on_model_choice_changed(user_idx)

    def _fail_make(_label: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(win, "_make_model_from_choice", _fail_make)
    model_idx = win.model_combo.findData("MultiPeakModel")
    with QtCore.QSignalBlocker(win.model_combo):
        win.model_combo.setCurrentIndex(model_idx)
    win._on_model_choice_changed(model_idx)
    assert warnings


# Tests for _merge_params and related static helper methods


def test_merge_params_param_basename_no_prefix() -> None:
    """Test _param_basename without model prefix."""
    model = lmfit.models.GaussianModel()
    assert Fit1DTool._param_basename("amplitude", model) == "amplitude"
    assert Fit1DTool._param_basename("center", model) == "center"


def test_merge_params_param_basename_with_prefix() -> None:
    """Test _param_basename with model prefix."""
    model = lmfit.models.GaussianModel(prefix="g1_")
    assert Fit1DTool._param_basename("g1_amplitude", model) == "amplitude"
    assert Fit1DTool._param_basename("g1_center", model) == "center"
    # Non-matching prefix should return unchanged
    assert Fit1DTool._param_basename("other_param", model) == "other_param"


def test_merge_params_expr_is_from_hint_no_hints() -> None:
    """Test _param_expr_is_from_hint when model has no hints."""

    def simple_func(x, a=1.0):
        return a * x

    model = lmfit.Model(simple_func)
    assert Fit1DTool._param_expr_is_from_hint("a", model) is False


def test_merge_params_expr_is_from_hint_with_hints() -> None:
    """Test _param_expr_is_from_hint with model that has expression hints."""
    # VoigtModel has fwhm with expr hint
    model = lmfit.models.VoigtModel()
    assert Fit1DTool._param_expr_is_from_hint("fwhm", model) is True
    assert Fit1DTool._param_expr_is_from_hint("amplitude", model) is False


def test_merge_params_is_model_func_arg() -> None:
    """Test _param_is_model_func_arg identifies function arguments."""

    def custom_func(x, real_arg=1.0, another_arg=2.0):
        return real_arg * x + another_arg

    model = lmfit.Model(custom_func)
    assert Fit1DTool._param_is_model_func_arg("real_arg", model) is True
    assert Fit1DTool._param_is_model_func_arg("another_arg", model) is True
    # Independent var should not be considered
    assert Fit1DTool._param_is_model_func_arg("x", model) is False


def test_merge_params_is_model_func_arg_with_prefix() -> None:
    """Test _param_is_model_func_arg with prefixed model."""
    model = lmfit.models.GaussianModel(prefix="p1_")
    assert Fit1DTool._param_is_model_func_arg("p1_amplitude", model) is True
    assert Fit1DTool._param_is_model_func_arg("p1_center", model) is True


def test_merge_params_is_valid_param_simple_model() -> None:
    """Test _is_valid_param with a simple model (no expr hints)."""

    def simple_func(x, a=1.0, b=2.0):
        return a * x + b

    model = lmfit.Model(simple_func)
    params = model.make_params()
    assert Fit1DTool._is_valid_param(params["a"], model) is True
    assert Fit1DTool._is_valid_param(params["b"], model) is True


def test_merge_params_is_valid_param_with_derived_params() -> None:
    """Test _is_valid_param with model that has derived parameters."""
    # VoigtModel has 'fwhm' as a derived parameter (expr from hints, not func arg)
    model = lmfit.models.VoigtModel()
    params = model.make_params()
    # sigma, gamma, center, amplitude are func args -> valid
    assert Fit1DTool._is_valid_param(params["sigma"], model) is True
    assert Fit1DTool._is_valid_param(params["gamma"], model) is True
    assert Fit1DTool._is_valid_param(params["center"], model) is True
    assert Fit1DTool._is_valid_param(params["amplitude"], model) is True
    # fwhm has expr from hints and is NOT a func arg -> invalid
    assert Fit1DTool._is_valid_param(params["fwhm"], model) is False


def test_merge_params_is_valid_param_not_in_model() -> None:
    """Test _is_valid_param returns False for params not in model."""

    def simple_func(x, a=1.0):
        return a * x

    model = lmfit.Model(simple_func)
    other_param = lmfit.Parameter(name="not_in_model", value=1.0)
    assert Fit1DTool._is_valid_param(other_param, model) is False


def test_merge_params_can_evaluate_expr_valid() -> None:
    """Test _can_evaluate_expr with valid expressions."""
    params = lmfit.Parameters()
    params.add("a", value=1.0)
    params.add("b", value=2.0)
    assert Fit1DTool._can_evaluate_expr("a + b", params) is True
    assert Fit1DTool._can_evaluate_expr("2*a", params) is True


def test_merge_params_can_evaluate_expr_invalid() -> None:
    """Test _can_evaluate_expr with invalid expressions."""
    params = lmfit.Parameters()
    params.add("a", value=1.0)
    # Reference to non-existent parameter
    assert Fit1DTool._can_evaluate_expr("a + nonexistent", params) is False
    # Syntax error
    assert Fit1DTool._can_evaluate_expr("a +", params) is False
    # Empty expression
    assert Fit1DTool._can_evaluate_expr("", params) is False
    assert Fit1DTool._can_evaluate_expr(None, params) is False


def test_merge_params_basic() -> None:
    """Test _merge_params copies values for matching valid params."""

    def simple_func(x, a=1.0, b=2.0):
        return a * x + b

    old_model = lmfit.Model(simple_func)
    new_model = lmfit.Model(simple_func)
    old_params = old_model.make_params(a=5.0, b=10.0)
    old_params["a"].min = 0.0
    old_params["a"].max = 20.0
    old_params["a"].vary = False
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    assert new_params["a"].value == pytest.approx(5.0)
    assert new_params["a"].min == pytest.approx(0.0)
    assert new_params["a"].max == pytest.approx(20.0)
    assert new_params["a"].vary is False
    assert new_params["b"].value == pytest.approx(10.0)


def test_merge_params_skips_new_params_not_in_old() -> None:
    """Test _merge_params leaves new params at defaults if not in old."""

    def old_func(x, a=1.0):
        return a * x

    def new_func(x, a=1.0, b=2.0):
        return a * x + b

    old_model = lmfit.Model(old_func)
    new_model = lmfit.Model(new_func)
    old_params = old_model.make_params(a=5.0)
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    assert new_params["a"].value == pytest.approx(5.0)
    assert new_params["b"].value == pytest.approx(2.0)  # default


def test_merge_params_skips_invalid_old_params() -> None:
    """Test _merge_params skips params that were invalid in old model."""
    # VoigtModel has fwhm as derived (invalid) param
    old_model = lmfit.models.VoigtModel()
    new_model = lmfit.models.VoigtModel()
    old_params = old_model.make_params()
    old_params["sigma"].value = 0.5
    # fwhm is derived, but let's try to set it anyway
    # (in practice, this would be constrained by the model)
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    # sigma should be merged (it's valid)
    assert new_params["sigma"].value == pytest.approx(0.5)
    # fwhm should NOT be merged (it's invalid/derived)
    # It should have the default expression from the model
    assert new_params["fwhm"].expr is not None


def test_merge_params_transfers_expression() -> None:
    """Test _merge_params transfers expressions when evaluable."""

    def simple_func(x, a=1.0, b=2.0):
        return a * x + b

    old_model = lmfit.Model(simple_func)
    new_model = lmfit.Model(simple_func)
    old_params = old_model.make_params()
    old_params["b"].expr = "2*a"
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    assert new_params["b"].expr == "2*a"


def test_merge_params_drops_unevaluable_expression() -> None:
    """Test _merge_params falls back to value if expression can't evaluate."""

    def old_func(x, a=1.0, b=2.0, c=3.0):
        return a * x + b + c

    def new_func(x, a=1.0, b=2.0):
        return a * x + b

    old_model = lmfit.Model(old_func)
    new_model = lmfit.Model(new_func)
    old_params = old_model.make_params()
    # Expression references 'c' which won't exist in new model
    # When expr is set, the value is computed from expr (c + 1 = 4.0)
    old_params["b"].expr = "c + 1"
    old_params["b"].min = -10.0
    old_params["b"].max = 10.0
    # The value will be 4.0 (from c + 1 = 3 + 1)
    expected_value = old_params["b"].value
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    # Expression can't be evaluated, so value and bounds should be copied instead
    assert new_params["b"].expr is None
    assert new_params["b"].value == pytest.approx(expected_value)
    assert new_params["b"].min == pytest.approx(-10.0)
    assert new_params["b"].max == pytest.approx(10.0)


def test_merge_params_preserves_new_model_expressions() -> None:
    """Test _merge_params doesn't overwrite new model's expressions."""
    # VoigtModel has built-in expression for fwhm
    old_model = lmfit.models.VoigtModel()
    new_model = lmfit.models.VoigtModel()
    old_params = old_model.make_params()
    new_params = new_model.make_params()
    original_fwhm_expr = new_params["fwhm"].expr

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    # fwhm expression should be preserved (not overwritten)
    assert new_params["fwhm"].expr == original_fwhm_expr


def test_merge_params_different_models() -> None:
    """Test _merge_params between different model types with common params."""
    # Both Gaussian and Lorentzian have center, sigma (or gamma), amplitude
    old_model = lmfit.models.GaussianModel()
    new_model = lmfit.models.LorentzianModel()
    old_params = old_model.make_params(center=1.0, sigma=0.5, amplitude=2.0)
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    # center and amplitude exist in both and should be merged
    assert new_params["center"].value == pytest.approx(1.0)
    assert new_params["amplitude"].value == pytest.approx(2.0)
    # sigma only exists in Gaussian, not Lorentzian (which has sigma too actually)
    # But let's verify sigma is merged since both have it
    assert new_params["sigma"].value == pytest.approx(0.5)


def test_merge_params_with_prefix_basic() -> None:
    """Test _merge_params with prefixed parameters."""
    old_model = lmfit.models.GaussianModel(prefix="g1_")
    new_model = lmfit.models.GaussianModel(prefix="g1_")
    old_params = old_model.make_params()
    old_params["g1_center"].value = 2.5
    old_params["g1_sigma"].value = 0.3
    old_params["g1_amplitude"].value = 10.0
    old_params["g1_center"].min = 0.0
    old_params["g1_center"].max = 5.0
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    assert new_params["g1_center"].value == pytest.approx(2.5)
    assert new_params["g1_sigma"].value == pytest.approx(0.3)
    assert new_params["g1_amplitude"].value == pytest.approx(10.0)
    assert new_params["g1_center"].min == pytest.approx(0.0)
    assert new_params["g1_center"].max == pytest.approx(5.0)


def test_merge_params_with_prefix_expression_transfer() -> None:
    """Test _merge_params transfers expressions with prefixed parameters."""
    old_model = lmfit.models.GaussianModel(prefix="g1_")
    new_model = lmfit.models.GaussianModel(prefix="g1_")
    old_params = old_model.make_params()
    old_params["g1_sigma"].expr = "g1_center / 10"
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    assert new_params["g1_sigma"].expr == "g1_center / 10"


def test_merge_params_with_prefix_derived_params_skipped() -> None:
    """Test _merge_params skips derived params in prefixed models."""
    # VoigtModel has fwhm and gamma as derived parameters (both have expr hints)
    old_model = lmfit.models.VoigtModel(prefix="v1_")
    new_model = lmfit.models.VoigtModel(prefix="v1_")
    old_params = old_model.make_params()
    old_params["v1_sigma"].value = 0.5
    old_params["v1_center"].value = 1.5
    old_params["v1_amplitude"].value = 3.0
    new_params = new_model.make_params()
    original_fwhm_expr = new_params["v1_fwhm"].expr
    original_gamma_expr = new_params["v1_gamma"].expr

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    # Valid params (sigma, center, amplitude) should be merged
    assert new_params["v1_sigma"].value == pytest.approx(0.5)
    assert new_params["v1_center"].value == pytest.approx(1.5)
    assert new_params["v1_amplitude"].value == pytest.approx(3.0)
    # fwhm and gamma (both derived) should keep their expressions
    assert new_params["v1_fwhm"].expr == original_fwhm_expr
    assert new_params["v1_gamma"].expr == original_gamma_expr


def test_merge_params_different_prefixes_no_merge() -> None:
    """Test _merge_params doesn't merge when prefixes differ."""
    old_model = lmfit.models.GaussianModel(prefix="g1_")
    new_model = lmfit.models.GaussianModel(prefix="g2_")
    old_params = old_model.make_params()
    old_params["g1_center"].value = 5.0
    new_params = new_model.make_params()
    default_center = new_params["g2_center"].value

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    # No matching param names, so nothing should be merged
    assert new_params["g2_center"].value == pytest.approx(default_center)


def test_merge_params_prefix_expression_unevaluable() -> None:
    """Test _merge_params falls back to value when prefixed expr can't evaluate."""

    def custom_func(x, a=1.0, b=2.0, c=3.0):
        return a * x + b + c

    old_model = lmfit.Model(custom_func, prefix="p1_")

    # Remove 'c' from new model by creating a different function
    def custom_func_no_c(x, a=1.0, b=2.0):
        return a * x + b

    new_model = lmfit.Model(custom_func_no_c, prefix="p1_")
    old_params = old_model.make_params()
    # Expression references p1_c which won't exist in new model
    old_params["p1_b"].expr = "p1_c + 1"
    old_params["p1_b"].min = -5.0
    old_params["p1_b"].max = 15.0
    expected_value = old_params["p1_b"].value  # Should be 4.0 (c+1 = 3+1)
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    # Expression can't be evaluated (p1_c doesn't exist), so value is used
    assert new_params["p1_b"].expr is None
    assert new_params["p1_b"].value == pytest.approx(expected_value)
    assert new_params["p1_b"].min == pytest.approx(-5.0)
    assert new_params["p1_b"].max == pytest.approx(15.0)


def test_merge_params_composite_model_with_prefixes() -> None:
    """Test _merge_params with composite models having different prefixes."""
    # Create composite models
    g1 = lmfit.models.GaussianModel(prefix="g1_")
    g2 = lmfit.models.GaussianModel(prefix="g2_")
    old_model = g1 + g2
    new_model = g1 + g2

    old_params = old_model.make_params()
    old_params["g1_center"].value = 1.0
    old_params["g1_sigma"].value = 0.1
    old_params["g2_center"].value = 2.0
    old_params["g2_sigma"].value = 0.2
    new_params = new_model.make_params()

    Fit1DTool._merge_params(old_params, new_params, old_model, new_model)

    # Both prefixed params should be merged
    assert new_params["g1_center"].value == pytest.approx(1.0)
    assert new_params["g1_sigma"].value == pytest.approx(0.1)
    assert new_params["g2_center"].value == pytest.approx(2.0)
    assert new_params["g2_sigma"].value == pytest.approx(0.2)
