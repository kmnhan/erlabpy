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
    fit_data = data.sel(x=slice(0.5, -0.5))
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
