import contextlib
import re

import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._fit2d import Fit2DTool


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


def test_ftool_2d_fill_and_transpose(qtbot) -> None:
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
    win._transpose()
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


def test_fit2d_run_fit(qtbot, exp_decay_model) -> None:
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

    win.y_index_spin.setValue(win.y_min_spin.value())
    win.nfev_spin.setValue(0)
    win._run_fit_2d("up")
    qtbot.waitUntil(
        lambda: all(ds is not None for ds in win._result_ds_full), timeout=10000
    )

    assert all(ds is not None for ds in win._result_ds_full)

    code = win._copy_code_full()
    assert "modelfit" in code
    assert ".isel(" in code


def test_fit2d_open_saved_fit_dataset(qtbot, exp_decay_model) -> None:
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

    win.y_index_spin.setValue(win.y_min_spin.value())
    win.nfev_spin.setValue(0)
    win._run_fit_2d("up")
    qtbot.waitUntil(
        lambda: all(ds is not None for ds in win._result_ds_full), timeout=10000
    )
    assert all(ds is not None for ds in win._result_ds_full)

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
