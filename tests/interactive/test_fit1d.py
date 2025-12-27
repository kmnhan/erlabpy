from types import SimpleNamespace

import lmfit
import numpy as np
import xarray as xr
from lmfit.models import GaussianModel
from qtpy import QtCore

from erlab.interactive import fit1d
from erlab.interactive._fit1d import Fit1DTool, _ParameterTableModel


def _make_data() -> xr.DataArray:
    x = np.linspace(-1.0, 1.0, 81)
    y = np.exp(-((x - 0.1) ** 2) / 0.05) + 0.05
    return xr.DataArray(y, coords={"x": x}, dims=["x"], name="signal")


def _make_params() -> lmfit.Parameters:
    params = lmfit.Parameters()
    params.add("amp", value=1.0, min=0.0, max=2.0, vary=True)
    params.add("offset", value=0.5, min=-np.inf, max=np.inf, vary=False)
    params.add("linked", expr="amp")
    return params


def test_fit1d_tool_status_roundtrip(qtbot) -> None:
    data = _make_data()
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    win.npeaks_spin.setValue(2)
    win.peak_shape_combo.setCurrentText("gaussian")
    win.fd_check.setChecked(True)
    win.background_combo.setCurrentText("polynomial")
    win.degree_spin.setValue(3)
    win.convolve_check.setChecked(True)
    win.domain_min_spin.setValue(float(data.x.values[10]))
    win.domain_max_spin.setValue(float(data.x.values[-10]))
    win.components_check.setChecked(True)
    win.timeout_spin.setValue(3.5)
    win.nfev_spin.setValue(321)
    win.method_combo.setCurrentText("powell")

    status = win.tool_status
    assert status.multipeak is not None
    assert status.multipeak.npeaks == 2
    assert status.multipeak.peak_shape == "gaussian"
    assert status.multipeak.background == "polynomial"
    assert status.show_components is True
    assert status.method == "powell"
    assert status.domain is not None

    win.npeaks_spin.setValue(1)
    win.peak_shape_combo.setCurrentText("lorentzian")
    win.background_combo.setCurrentText("linear")
    win.components_check.setChecked(False)

    win.tool_status = status

    assert win.npeaks_spin.value() == 2
    assert win.peak_shape_combo.currentText() == "gaussian"
    assert win.background_combo.currentText() == "polynomial"
    assert win.components_check.isChecked() is True
    assert win.method_combo.currentText() == "powell"
    assert win._fit_domain() is not None


def test_fit1d_copy_code_domain(qtbot) -> None:
    data = _make_data()
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    win.domain_min_spin.setValue(float(data.x.values[10]))
    win.domain_max_spin.setValue(float(data.x.values[-10]))
    code = win.copy_code()

    assert "data_fit = data.sel(" in code
    assert "slice(" in code


def test_fit1d_run_fit_updates_stats(qtbot) -> None:
    data = _make_data()
    model = GaussianModel()
    win = fit1d(data, model=model, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    win.nfev_spin.setValue(200)
    win._run_fit()

    assert win.nfev_out_value.text() != "\u2014"
    assert win.redchi_value.text() != "\u2014"
    assert win.copy_button.isEnabled() is True

    stderr_values = []
    for row in range(win.param_model.rowCount()):
        index = win.param_model.index(row, 2)
        stderr_values.append(win.param_model.data(index))
    assert any(value not in ("", None) for value in stderr_values)


def test_parameter_table_model_setdata_clamps_and_vary() -> None:
    params = _make_params()
    model = _ParameterTableModel(params)

    min_index = model.index(0, 3)
    assert model.setData(min_index, "3", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].min == 3.0
    assert params["amp"].max == 3.0
    assert params["amp"].value == 3.0

    value_index = model.index(0, 1)
    assert model.setData(value_index, "2.5", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].value == 3.0

    max_index = model.index(0, 4)
    assert model.setData(max_index, "4", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].max == 4.0

    min_index = model.index(0, 3)
    assert model.setData(min_index, "0", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].min == 0.0

    assert model.setData(value_index, "2.5", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].value == 2.5

    vary_index = model.index(0, 5)
    assert model.setData(
        vary_index,
        QtCore.Qt.CheckState.Unchecked,
        QtCore.Qt.ItemDataRole.CheckStateRole,
    )
    assert params["amp"].vary is False

    invalid_index = model.index(0, 1)
    assert (
        model.setData(invalid_index, "not-a-number", QtCore.Qt.ItemDataRole.EditRole)
        is False
    )


def test_parameter_table_model_expr_param_flags() -> None:
    params = _make_params()
    model = _ParameterTableModel(params)

    expr_index = model.index(2, 5)
    flags = model.flags(expr_index)
    assert not (flags & QtCore.Qt.ItemFlag.ItemIsEnabled)
    assert model.data(expr_index, QtCore.Qt.ItemDataRole.CheckStateRole) is None
    assert (
        model.setData(
            expr_index,
            QtCore.Qt.CheckState.Unchecked,
            QtCore.Qt.ItemDataRole.CheckStateRole,
        )
        is False
    )

    value_index = model.index(2, 1)
    assert not (model.flags(value_index) & QtCore.Qt.ItemFlag.ItemIsEditable)


def test_parameter_table_model_parse_bound_variants() -> None:
    assert _ParameterTableModel._parse_bound(None, default=5.0) == 5.0
    assert _ParameterTableModel._parse_bound("", default=-np.inf) == -np.inf
    assert _ParameterTableModel._parse_bound("inf", default=0.0) == np.inf
    assert _ParameterTableModel._parse_bound("-inf", default=0.0) == -np.inf


def test_parameter_table_model_display_and_tooltip() -> None:
    params = _make_params()
    params["amp"].stderr = 0.1
    model = _ParameterTableModel(params)

    name_index = model.index(0, 0)
    assert model.data(name_index) == "amp"

    value_index = model.index(0, 1)
    assert model.data(value_index) == _ParameterTableModel._format_value(
        params["amp"].value
    )

    stderr_index = model.index(0, 2)
    assert model.data(stderr_index) == _ParameterTableModel._format_scientific(0.1)

    min_index = model.index(1, 3)
    max_index = model.index(1, 4)
    assert model.data(min_index) == "-inf"
    assert model.data(max_index) == "inf"

    tooltip = model.data(value_index, QtCore.Qt.ItemDataRole.ToolTipRole)
    assert "value:" in tooltip
    assert "min:" in tooltip
    assert "max:" in tooltip
    assert "vary:" in tooltip


def test_parameter_table_model_format_helpers() -> None:
    assert _ParameterTableModel._format_value(np.nan) == "nan"
    assert _ParameterTableModel._format_value(np.inf) == "inf"
    assert _ParameterTableModel._format_value(-np.inf) == "-inf"
    assert _ParameterTableModel._format_scientific(np.nan) == "nan"
    assert _ParameterTableModel._format_scientific(np.inf) == "inf"
    assert _ParameterTableModel._format_scientific(-np.inf) == "-inf"
    assert _ParameterTableModel._format_bound(np.inf, default="inf") == "inf"
    assert _ParameterTableModel._format_bound(-np.inf, default="-inf") == "-inf"
    assert _ParameterTableModel._format_bound(1.2345, default="inf") == "1.2345"


def test_fit1d_merge_params_respects_expr() -> None:
    old_params = lmfit.Parameters()
    old_params.add("amp", value=1.5, min=0.0, max=2.5, vary=False)
    old_params.add("offset", value=0.1)

    new_params = lmfit.Parameters()
    new_params.add("amp", value=9.0, min=-1.0, max=10.0, vary=True)
    new_params.add("offset", expr="amp")
    new_params.add("extra", value=3.0)

    Fit1DTool._merge_params(old_params, new_params)
    assert new_params["amp"].value == 1.5
    assert new_params["amp"].min == 0.0
    assert new_params["amp"].max == 2.5
    assert new_params["amp"].vary is False
    assert new_params["offset"].expr == "amp"


def test_fit1d_domain_brushes_and_segmented(qtbot) -> None:
    xvals = np.array([0.0, 1.0, 3.0, 4.0])
    data = xr.DataArray(np.arange(xvals.size), coords={"x": xvals}, dims=["x"])
    model = GaussianModel()
    win = fit1d(
        data,
        model=model,
        execute=False,
        data_name="data",
        model_name="model",
    )
    qtbot.addWidget(win)

    assert win._auto_segmented(convolve=False) is False
    assert win._auto_segmented(convolve=True) is True

    win.domain_min_spin.setValue(float(xvals[1]))
    win.domain_max_spin.setValue(float(xvals[2]))
    brushes = win._domain_brushes(xvals)
    assert brushes is not None
    mask = (xvals >= xvals[1]) & (xvals <= xvals[2])
    tint_count = sum(brush.color().alpha() == 50 for brush in brushes)
    assert tint_count == int(mask.sum())


def test_fit1d_domain_sync_and_fit_domain(qtbot) -> None:
    xvals = np.linspace(0.0, 1.0, 11)
    data = xr.DataArray(np.arange(xvals.size), coords={"x": xvals}, dims=["x"])
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    assert win._fit_domain() is None

    win.domain_min_spin.setValue(float(xvals[2]))
    win.domain_max_spin.setValue(float(xvals[8]))
    win._sync_lines_from_domain()
    assert win._fit_domain() == (float(xvals[2]), float(xvals[8]))

    win.fit_line_min.setValue(float(xvals[9]))
    win.fit_line_max.setValue(float(xvals[1]))
    win._sync_domain_from_lines()
    assert win.domain_min_spin.value() <= win.domain_max_spin.value()


def test_fit1d_copy_code_variants(qtbot) -> None:
    data = _make_data()
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    win._argnames["data"] = "data name"
    win._argnames["model"] = "model-name"
    win.domain_min_spin.setValue(float(data.x.values[5]))
    win.domain_max_spin.setValue(float(data.x.values[-5]))

    params = lmfit.Parameters()
    params.add("valid", value=1.0, min=-np.inf, max=np.inf, vary=True)
    params.add("fixed", value=2.0, min=0.0, max=3.0, vary=False)
    params.add("linked", expr="valid")
    win._params = params

    code = win.copy_code()
    assert "model = era.fit.models.MultiPeakModel" in code
    assert "data_fit = data.sel(" in code
    assert "make_params(" in code
    assert "vary': False" in code


def test_fit1d_copy_code_no_params(qtbot) -> None:
    data = _make_data()
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)
    win._params = None

    code = win.copy_code()
    assert "params = model.make_params()" in code
    assert "params=params" in code


def test_fit1d_fit_data_fallback(monkeypatch, qtbot) -> None:
    data = _make_data()
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    win.domain_min_spin.setValue(float(data.x.values[10]))
    win.domain_max_spin.setValue(float(data.x.values[-10]))

    def _raise(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(xr.DataArray, "sel", _raise)
    sliced = win._fit_data()
    assert sliced.size < data.size
    assert sliced.dims == data.dims


def test_fit1d_parameter_values_and_helpers(qtbot) -> None:
    data = _make_data()
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    assert win._parameter_values()
    assert win._has_non_finite_params() is False

    nan_params = win._params.copy()
    first_param = next(iter(nan_params))
    nan_params[first_param].value = np.nan
    win._params = nan_params
    assert win._has_non_finite_params() is True
    win._update_fit_curve()
    assert win._last_fit_y is None
    assert win._last_residual is None

    win._params = None
    assert win._parameter_values() == {}


def test_fit1d_set_fit_stats_and_elapsed(qtbot) -> None:
    data = _make_data()
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    win.nfev_spin.setValue(10)
    result = SimpleNamespace(
        nfev=10,
        redchi=0.5,
        rsquared=0.9,
        aic=1.2,
        bic=2.3,
    )
    win._set_fit_stats(result, elapsed=1.23)
    assert "color:#d62728" in win.nfev_out_label.text()
    assert win.elapsed_value.text() != "\u2014"

    win._set_elapsed_status(2.34, timed_out=True)
    assert "color:#d62728" in win.elapsed_value.text()


def test_fit1d_residuals_use_cached_result(qtbot) -> None:
    data = _make_data()
    win = fit1d(data, execute=False, data_name="data", model_name="model")
    qtbot.addWidget(win)

    params = win._params.copy()
    best_fit = np.ones_like(data.values)
    win._last_result = SimpleNamespace(params=params.copy(), best_fit=best_fit)
    win._params = params.copy()
    win._fit_is_current = True

    residuals = win._residuals_from_result(win._x_values())
    np.testing.assert_allclose(residuals, data.values - best_fit)
