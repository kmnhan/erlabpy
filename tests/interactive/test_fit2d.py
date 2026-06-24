import contextlib
import json
import os
import re
import types
import warnings

import lmfit
import numpy as np
import pyqtgraph as pg
import pytest
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive._fit2d as fit2d_module
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive.imagetool import provenance
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


def _assert_fit_result_dataset_equivalent(
    actual: xr.Dataset, expected: xr.Dataset
) -> None:
    xr.testing.assert_identical(
        actual.drop_vars("modelfit_results"),
        expected.drop_vars("modelfit_results"),
    )
    actual_result = actual.modelfit_results.compute().item()
    expected_result = expected.modelfit_results.compute().item()
    assert type(actual_result.model) is type(expected_result.model)
    assert list(actual_result.params.keys()) == list(expected_result.params.keys())
    for name, expected_param in expected_result.params.items():
        actual_param = actual_result.params[name]
        assert actual_param.value == pytest.approx(expected_param.value)
        if expected_param.stderr is None:
            assert actual_param.stderr is None
        else:
            assert actual_param.stderr == pytest.approx(expected_param.stderr)
        assert actual_param.expr == expected_param.expr
        assert actual_param.vary == expected_param.vary


def _fit_result_dataset(params, *, nfev: int = 1) -> xr.Dataset:
    params = params.copy()
    param_args = ", ".join(("x", *params.keys()))
    namespace = {"np": np}
    exec(  # noqa: S102
        f"def _model_func({param_args}):\n    return np.zeros_like(x, dtype=float)\n",
        namespace,
    )
    model = lmfit.Model(namespace["_model_func"])
    result = lmfit.model.ModelResult(
        model,
        params,
        data=np.zeros(3),
        fcn_args=(np.arange(3, dtype=float),),
        max_nfev=nfev,
    )
    result.params = params.copy()
    result.nfev = nfev
    result.redchi = 1.0
    result.rsquared = 0.9
    result.aic = 1.0
    result.bic = 2.0

    return xr.Dataset({"modelfit_results": xr.DataArray(result, dims=())})


def _placeholder_fit_result_dataset(params) -> xr.Dataset:
    class _Result:
        def __init__(self) -> None:
            self.params = params.copy()
            self.nfev = 1
            self.redchi = 1.0
            self.rsquared = 0.9
            self.aic = 1.0
            self.bic = 2.0

    return xr.Dataset({"modelfit_results": xr.DataArray(_Result(), dims=())})


def _assert_fit_result_list_equivalent(
    actual: list[xr.Dataset | None], expected: list[xr.Dataset | None]
) -> None:
    assert len(actual) == len(expected)
    for actual_ds, expected_ds in zip(actual, expected, strict=True):
        if expected_ds is None:
            assert actual_ds is None
            continue
        assert actual_ds is not None
        _assert_fit_result_dataset_equivalent(actual_ds, expected_ds)


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


def _seed_fit2d_full_results(win: Fit2DTool, model, params) -> None:
    for idx in range(len(win._result_ds_full)):
        fit_data = win._data_full.isel({win._y_dim_name: idx})
        fit_ds = fit_data.xlm.modelfit(
            win._coord_name,
            model=model,
            params=params,
            max_nfev=10,
        ).load()
        result = fit_ds.modelfit_results.compute().item()
        win._result_ds_full[idx] = fit_ds
        win._params_full[idx] = result.params.copy()
        win._params_from_coord_full[idx] = {}

    win._set_current_index(0)
    win._fit_is_current = True
    win._update_full_fit_saveable()
    win._update_param_plot_options()


def _seed_fit2d_param_results(win: Fit2DTool, params_list) -> None:
    win._params_full = [params.copy() for params in params_list]
    win._result_ds_full = [_fit_result_dataset(params) for params in params_list]
    win._update_param_plot_options()


def _lmfit_json_with_callable_pyversion(
    payload: str, callable_name: str, pyversion: str = "3.13"
) -> str:
    decoded = json.loads(payload)

    def _set_pyversion(value: object) -> bool:
        changed = False
        if isinstance(value, dict):
            if (
                value.get("__class__") == "Callable"
                and value.get("__name__") == callable_name
            ):
                value["pyversion"] = pyversion
                changed = True
            for item in value.values():
                changed = _set_pyversion(item) or changed
        elif isinstance(value, list):
            for item in value:
                changed = _set_pyversion(item) or changed
        return changed

    assert _set_pyversion(decoded)
    return json.dumps(decoded)


def _saved_ftool_dataset_with_callable_pyversion(
    ds: xr.Dataset, callable_name: str, pyversion: str = "3.13"
) -> xr.Dataset:
    ds = ds.copy()
    state = json.loads(ds.attrs["tool_state"])
    state["model_state"][1] = _lmfit_json_with_callable_pyversion(
        state["model_state"][1], callable_name, pyversion
    )
    ds.attrs["tool_state"] = json.dumps(state)

    result_var = Fit2DTool._PERSISTED_FIT_RESULT_VAR
    if result_var not in ds:
        return ds
    sparse = xr.load_dataset(
        memoryview(np.asarray(ds[result_var].values, dtype=np.uint8).tobytes()),
        engine="h5netcdf",
    )
    for var in sparse.data_vars:
        if str(var).endswith("modelfit_results"):
            attrs = sparse[var].attrs.copy()
            patched = xr.apply_ufunc(
                lambda text: _lmfit_json_with_callable_pyversion(
                    str(text), callable_name, pyversion
                ),
                sparse[var],
                vectorize=True,
                output_dtypes=[str],
            )
            patched.attrs = attrs
            sparse[var] = patched
    blob = sparse.to_netcdf(path=None, engine="h5netcdf", invalid_netcdf=True)
    ds[result_var] = xr.DataArray(
        np.frombuffer(blob, dtype=np.uint8).copy(),
        dims=(Fit2DTool._PERSISTED_FIT_RESULT_DIM,),
    )
    return ds


def _assert_modelresult_params_equivalent(actual, expected) -> None:
    assert list(actual.params.keys()) == list(expected.params.keys())
    for name, expected_param in expected.params.items():
        actual_param = actual.params[name]
        assert actual_param.value == pytest.approx(expected_param.value)
        assert actual_param.expr == expected_param.expr
        assert actual_param.vary == expected_param.vary
    np.testing.assert_allclose(actual.best_fit, expected.best_fit)


def _make_erlab_callable_case(
    name: str,
) -> tuple[str, object, xr.DataArray, object]:
    x = np.linspace(-1.0, 1.0, 25)
    y = np.arange(2)
    match name:
        case "multipeak":
            model = erlab.analysis.fit.models.MultiPeakModel(
                fd=False,
                background="none",
                convolve=False,
            )
            params = model.make_params(
                p0_center=0.0,
                p0_width=0.35,
                p0_height=1.2,
            )
            callable_name = "MultiPeakFunction"
        case "polynomial":
            model = erlab.analysis.fit.models.PolynomialModel(degree=2)
            params = model.make_params(c0=1.0, c1=0.25, c2=-0.15)
            callable_name = "PolynomialFunction"
        case _:
            raise ValueError(name)
    rows = [model.eval(params, x=x) * (1.0 + 0.05 * idx) for idx in y]
    data = xr.DataArray(
        np.stack(rows, axis=0),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name=f"{name}_map",
    )
    return callable_name, model, data, params


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


def test_fit2d_update_data_preserves_transpose_orientation(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._do_transpose()
    win.y_index_spin.setValue(2)

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.1
    win.update_data(updated)

    xr.testing.assert_identical(win.tool_data, updated.transpose("x", "y"))
    assert win._y_dim_name == "x"
    assert win.y_index_spin.value() == 2
    assert win.y_index_spin.maximum() == updated.sizes["x"] - 1


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


def test_fit2d_restore_uses_saved_voigt_params_before_defaults(qtbot) -> None:
    data = _make_2d_data()
    model = erlab.analysis.fit.models.MultiPeakModel(
        peak_shapes="voigt",
        fd=False,
        background="linear",
        convolve=False,
    )
    params = model.make_params(
        const_bkg=0.0,
        lin_bkg=0.0,
        p0_center=0.1,
        p0_sigma=0.15,
        p0_gamma=0.2,
        p0_amplitude=1.3,
    )
    win = erlab.interactive.ftool(data, model=model, params=params, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    win.param_plot_combo.setCurrentText("p0_width")
    win.param_plot_overlay_check.setChecked(True)

    status = win.tool_status

    win_restored = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, Fit2DTool)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        win_restored.tool_status = status

    assert win_restored.tool_status.params == status.params
    assert [
        win_restored.param_plot_combo.itemText(i)
        for i in range(win_restored.param_plot_combo.count())
    ] == []
    assert not win_restored.param_plot_combo.isEnabled()
    assert not win_restored.param_plot_overlay_check.isChecked()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        win_roundtripped = erlab.interactive.utils.ToolWindow.from_dataset(
            win.to_dataset()
        )
    qtbot.addWidget(win_roundtripped)
    assert isinstance(win_roundtripped, Fit2DTool)
    assert win_roundtripped.tool_status.params == status.params
    assert win_roundtripped.param_plot_combo.count() == 0
    assert not win_roundtripped.param_plot_combo.isEnabled()
    assert not win_roundtripped.param_plot_overlay_check.isChecked()


@pytest.mark.parametrize("case_name", ["multipeak", "polynomial"])
def test_fit2d_persistence_suppresses_successful_erlab_callable_warning(
    qtbot, case_name: str
) -> None:
    callable_name, model, data, params = _make_erlab_callable_case(case_name)
    win = erlab.interactive.ftool(data, model=model, params=params, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    fit_ds = win._data.xlm.modelfit(
        win._coord_name,
        model=model,
        params=params,
        max_nfev=10,
    ).load()
    expected_result = fit_ds.modelfit_results.compute().item()
    win._last_result_ds = fit_ds
    win._result_ds_full[win._current_idx] = fit_ds
    win._params = expected_result.params.copy()
    win._params_full[win._current_idx] = win._params.copy()
    win._fit_is_current = True
    expected_params = win.tool_status.params

    saved = _saved_ftool_dataset_with_callable_pyversion(
        win.to_dataset(), callable_name
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = erlab.interactive.utils.ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit2DTool)

    assert [
        warning
        for warning in caught
        if "Could not unpack dill-encoded callable" in str(warning.message)
    ] == []
    assert type(restored._model.func) is type(model.func)
    assert restored.tool_status.params == expected_params

    restored_result_ds = restored._result_ds_full[restored._current_idx]
    assert restored_result_ds is not None
    restored_result = restored_result_ds.modelfit_results.compute().item()
    assert type(restored_result.model.func) is type(model.func)
    _assert_modelresult_params_equivalent(restored_result, expected_result)


def test_fit2d_status_and_persistence_preserve_transpose_orientation(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._do_transpose()
    win.y_index_spin.setValue(3)
    status = win.tool_status

    win_restored = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, Fit2DTool)
    win_restored.tool_status = status

    xr.testing.assert_identical(win_restored.tool_data, data.transpose("x", "y"))
    assert win_restored.y_index_spin.value() == 3

    win_roundtripped = erlab.interactive.utils.ToolWindow.from_dataset(win.to_dataset())
    qtbot.addWidget(win_roundtripped)
    assert isinstance(win_roundtripped, Fit2DTool)
    xr.testing.assert_identical(win_roundtripped.tool_data, data.transpose("x", "y"))
    assert win_roundtripped.y_index_spin.value() == 3

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) + 1.0
    win_roundtripped.update_data(updated)
    xr.testing.assert_identical(win_roundtripped.tool_data, updated.transpose("x", "y"))


def test_fit2d_saved_dims_ignore_missing_or_incompatible_state() -> None:
    data = _make_2d_data()

    assert Fit2DTool._data_with_saved_dims(data, None) is data
    assert (
        Fit2DTool._data_with_saved_dims(
            data, types.SimpleNamespace(data_dims_full=("x", "missing"))
        )
        is data
    )


def test_fit2d_tool_status_overlay_and_limits(qtbot, exp_decay_model) -> None:
    data = _make_2d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    _seed_fit2d_param_results(
        win, [params.copy() for _ in range(len(win._params_full))]
    )
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
    assert win_restored.param_plot_combo.count() == 0
    assert win_restored.param_plot_overlay_check.isChecked() is False


def test_fit2d_overlay_legend_sync(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    _seed_fit2d_param_results(
        win, [win._params.copy() for _ in range(len(win._params_full))]
    )
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

    _seed_fit2d_param_results(
        win, [win._params.copy() for _ in range(len(win._params_full))]
    )
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


def test_fit2d_full_provenance_handles_spaced_fit_axis(qtbot) -> None:
    x = np.linspace(-1.0, 1.0, 5)
    motor = np.array([10.0, 11.0, 12.0])
    data = xr.DataArray(
        np.ones((3, 5)),
        dims=("Fake Motor", "x"),
        coords={"Fake Motor": motor, "x": x},
        name="derived_crop",
    )
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    centers = [0.1, 0.2, 0.3]
    params_full = []
    for value in centers:
        params = win._params.copy()
        params["p0_center"].set(value=value)
        params_full.append(params)
    win._params_full = params_full
    win._result_ds_full = [xr.Dataset() for _ in params_full]
    win.y_min_spin.setValue(0)
    win.y_max_spin.setValue(len(params_full) - 1)

    assert win.current_provenance_spec() is not None
    prelude = win._detached_full_copy_prelude(input_name="derived_crop")
    assert prelude is not None
    assert "fit_data" not in prelude

    display_code = win.current_provenance_spec().display_code()
    assert display_code is not None
    assert "fit_data" not in display_code
    assert ".xlm.modelfit" in display_code

    namespace = {"derived_crop": data}
    exec(  # noqa: S102
        prelude,
        {
            "__builtins__": {"dict": dict, "slice": slice},
            "era": erlab.analysis,
            "xr": xr,
        },
        namespace,
    )
    center_param = namespace["params"]["p0_center"]
    assert isinstance(center_param, xr.DataArray)
    assert center_param.dims == ("Fake Motor",)
    np.testing.assert_allclose(center_param.values, centers)
    xr.testing.assert_equal(
        center_param.coords["Fake Motor"], data.coords["Fake Motor"]
    )


def test_fit2d_file_roundtrip_preserves_spaced_associated_coord(
    qtbot, tmp_path
) -> None:
    data = _make_2d_data().assign_coords(
        {"Fake Motor": ("y", np.linspace(10.0, 12.0, 3))}
    )
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    fname = tmp_path / "fit2d-spaced-associated-coord.h5"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        win.to_file(fname)

    assert not any("space in its name" in str(item.message) for item in caught)
    restored = erlab.interactive.utils.ToolWindow.from_file(fname)
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit2DTool)
    xr.testing.assert_equal(
        restored.tool_data.coords["Fake Motor"], data.coords["Fake Motor"]
    )


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

    assert win.update_data(updated) is False
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

    started_steps: list[int] = []

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
            on_success(_fit_result_dataset(win._params))
            return True
        return False

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    win.y_index_spin.setValue(win.y_min_spin.value())
    win._run_fit_2d("up")

    assert started_steps == [1]
    qtbot.waitUntil(lambda: started_steps == [1, 2], timeout=1000)


def test_fit2d_next_step_requests_paint_before_deferred_next_step(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    events: list[str] = []

    monkeypatch.setattr(win, "_update_fit_curve", lambda: events.append("curve"))
    monkeypatch.setattr(
        win, "_refresh_slider_from_model", lambda: events.append("slider")
    )

    def _update_param_plot(*, notify: bool = True) -> None:
        events.append(f"plot-{notify}")

    monkeypatch.setattr(win, "_update_param_plot", _update_param_plot)
    monkeypatch.setattr(win, "_fit_2d_live_refresh_due", lambda: True)
    monkeypatch.setattr(win, "_request_fit_step_paint", lambda: events.append("paint"))
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
        events.append(f"start-{step}")
        win._fit_start_time = 0.0
        if step == 1:
            on_success(_fit_result_dataset(win._params))
            return True
        return False

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    win.y_index_spin.setValue(win.y_min_spin.value())
    events.clear()
    win._run_fit_2d("up")

    assert events == ["start-1", "curve", "slider", "plot-False", "paint"]
    qtbot.waitUntil(lambda: "start-2" in events, timeout=1000)
    paint_after_first_start = events.index("paint", events.index("start-1"))
    assert paint_after_first_start < events.index("start-2")


def test_fit2d_paints_once_between_finished_step_and_next_worker(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    events: list[str] = []

    monkeypatch.setattr(win, "_fit_2d_live_refresh_due", lambda: True)
    monkeypatch.setattr(win, "_request_fit_step_paint", lambda: events.append("paint"))
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
        events.append(f"start-{step}")
        win._fit_start_time = 0.0
        if step == 1:
            on_success(_fit_result_dataset(win._params))
            return True
        return False

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    win.y_index_spin.setValue(win.y_min_spin.value())
    win._run_fit_2d("up")
    qtbot.waitUntil(lambda: events[-1:] == ["start-2"], timeout=1000)

    assert events == ["start-1", "paint", "start-2"]


def test_fit2d_sequence_throttles_expensive_live_refreshes(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    events: list[str] = []
    clock_values = [100.0, 100.05, 100.30]

    def _monotonic() -> float:
        return clock_values.pop(0) if clock_values else 100.10

    monkeypatch.setattr(fit2d_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(
        win, "_update_param_plot_options", lambda: events.append("options")
    )

    def _update_param_plot(*, notify: bool = True) -> None:
        events.append(f"plot-{notify}")

    monkeypatch.setattr(win, "_update_param_plot", _update_param_plot)
    monkeypatch.setattr(win, "_request_fit_step_paint", lambda: events.append("paint"))
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
        events.append(f"start-{step}")
        win._fit_start_time = 0.0
        on_success(_fit_result_dataset(win._params))
        return True

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    win.y_index_spin.setValue(win.y_min_spin.value())
    events.clear()
    win._run_fit_2d("up")
    qtbot.waitUntil(
        lambda: win._fit_2d_total == 0 and not win._fit_2d_indices,
        timeout=1000,
    )

    assert events == [
        "start-1",
        "start-2",
        "options",
        "plot-False",
        "paint",
        "start-3",
        "options",
        "plot-True",
    ]


def test_fit2d_sequence_skips_visible_refresh_for_hidden_steps(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_index_spin.setValue(win.y_min_spin.value())

    clock_values = [100.0, 100.05, 100.30]

    def _monotonic() -> float:
        return clock_values.pop(0) if clock_values else 100.10

    refresh_modes: list[tuple[bool, bool, bool]] = []
    original_refresh = win._refresh_contents_from_index

    def _refresh_contents_from_index(
        *,
        mark_fit_stale: bool = True,
        update_widgets: bool = True,
        elapsed: float | None = None,
        emit_info: bool = True,
        emit_param_changed: bool = True,
    ) -> None:
        refresh_modes.append((update_widgets, emit_info, emit_param_changed))
        original_refresh(
            mark_fit_stale=mark_fit_stale,
            update_widgets=update_widgets,
            elapsed=elapsed,
            emit_info=emit_info,
            emit_param_changed=emit_param_changed,
        )

    started_steps: list[int] = []
    monkeypatch.setattr(fit2d_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(
        win, "_refresh_contents_from_index", _refresh_contents_from_index
    )
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
        on_success(_fit_result_dataset(win._params))
        return True

    monkeypatch.setattr(win, "_start_fit_worker", _start_fit_worker)

    win._run_fit_2d("up")
    qtbot.waitUntil(
        lambda: win._fit_2d_total == 0 and not win._fit_2d_indices,
        timeout=1000,
    )

    assert started_steps == [1, 2, 3]
    assert sum(not update for update, _, _ in refresh_modes) > 0
    assert (True, False, False) in refresh_modes
    assert refresh_modes[-1] == (True, True, True)
    assert win.y_index_spin.value() == win.y_max_spin.value()
    assert win._write_history is True


def test_fit2d_set_fit_ds_updates_slice_state_before_fit_finished(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    idx = win._current_idx
    param_name = next(iter(win._params))
    params = win._params.copy()
    expected_value = params[param_name].value + 0.25
    params[param_name].set(value=expected_value)
    result_ds = _fit_result_dataset(params)

    param_changed: list[None] = []
    events: list[str] = []
    win.param_model.sigParamsChanged.connect(lambda: param_changed.append(None))
    win.sigFitFinished.connect(lambda params: events.append("finished"))
    monkeypatch.setattr(
        win, "_update_param_plot", lambda *, notify=True: events.append("plot")
    )

    win._set_fit_ds(result_ds, 0.0)

    assert param_changed == []
    assert events == ["plot", "finished"]
    assert win._params_full[idx][param_name].value == pytest.approx(expected_value)
    assert win._result_ds_full[idx] is win._last_result_ds


def test_fit2d_fit_step_paint_widgets_skip_invalid_entries(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.y_value_spin = object()
    duplicate = win.fit_down_button
    win.fit_up_button = duplicate

    widgets = win._fit_step_paint_widgets()

    assert all(isinstance(widget, QtWidgets.QWidget) for widget in widgets)
    assert sum(widget is duplicate for widget in widgets) == 1


def test_fit2d_sequence_state_and_history_edges(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._fit_2d_total = 2
    win._sync_fit_result_state()
    assert win._fit_2d_param_plot_refresh_pending

    win._fit_2d_last_live_refresh = fit2d_module.time.monotonic()
    assert not win._fit_2d_live_refresh_due()

    replaced: list[bool] = []
    monkeypatch.setattr(win, "_replace_last_state", lambda: replaced.append(True))
    win._write_history = True
    win._begin_fit_2d_sequence_history()
    assert win._fit_2d_sequence_write_history is True
    assert win._write_history is False
    win._begin_fit_2d_sequence_history()
    win._finish_fit_2d_sequence_history()
    assert win._write_history is True
    assert replaced == [True]

    events: list[str] = []
    monkeypatch.setattr(
        win, "_update_param_plot_options", lambda: events.append("options")
    )
    monkeypatch.setattr(
        win,
        "_update_param_plot",
        lambda *, notify=True: events.append(f"plot-{notify}"),
    )

    win._fit_2d_param_plot_refresh_pending = False
    win._flush_fit_2d_sequence_param_plot()
    assert events == []
    win._flush_fit_2d_sequence_param_plot(force=True, notify=False)
    assert events == ["options", "plot-False"]


def test_fit2d_sequence_view_live_refresh_edges(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    events: list[str] = []
    monkeypatch.setattr(
        win,
        "_refresh_contents_from_index",
        lambda **kwargs: events.append(f"refresh-{kwargs['emit_info']}"),
    )
    monkeypatch.setattr(
        win,
        "_flush_fit_2d_sequence_param_plot",
        lambda *, notify=True, force=False: events.append(f"plot-{notify}"),
    )

    win._fit_2d_live_refresh_pending = False
    win._sync_fit_2d_sequence_view(0, full=False)
    assert events == []

    win._fit_2d_live_refresh_pending = True
    win._sync_fit_2d_sequence_view(0, full=False)
    assert events == ["refresh-False", "plot-False"]

    events.clear()
    win._fit_2d_total = 0
    monkeypatch.setattr(
        Fit2DTool.__mro__[1],
        "_defer_next_fit_step",
        lambda _self, callback: events.append("super") or callback(),
    )
    win._defer_next_fit_step(lambda: events.append("callback"))
    assert events == ["super", "callback"]


def test_fit2d_cancelled_before_deferred_next_step_stops_sequence(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    started_steps: list[int] = []

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
            on_success(_fit_result_dataset(win._params))
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


def test_fit2d_persistence_roundtrip_preserves_fit_results(
    qtbot, exp_decay_model
) -> None:
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

    _seed_fit2d_full_results(win, exp_decay_model, params)
    expected_results = [
        None if ds is None else ds.copy(deep=True) for ds in win._result_ds_full
    ]
    expected_status = win.tool_status.model_dump()

    win_restored = erlab.interactive.utils.ToolWindow.from_dataset(win.to_dataset())
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, Fit2DTool)

    assert win_restored._fit_is_current
    assert all(ds is not None for ds in win_restored._result_ds_full)
    _assert_fit_result_list_equivalent(win_restored._result_ds_full, expected_results)
    assert win_restored.tool_status.model_dump() == expected_status
    assert win_restored.copy_full_button.isEnabled()
    assert win_restored.save_full_button.isEnabled()
    assert win_restored.current_provenance_spec() is not None


def test_fit2d_irregular_current_slice_disables_unsafe_segments(
    qtbot, monkeypatch
) -> None:
    sample_temp = np.array([0.0, 1.0, 2.7, 4.1, 7.6, 8.2], dtype=float)
    alpha = np.array([0.0, 1.0])
    data = np.vstack(
        [
            np.exp(-((sample_temp - 3.0) ** 2) / 5.0),
            np.exp(-((sample_temp - 4.0) ** 2) / 5.0),
        ]
    )
    darr = xr.DataArray(
        data,
        dims=("alpha", "sample_temp"),
        coords={"alpha": alpha, "sample_temp": sample_temp},
        name="cut",
    )
    model = erlab.analysis.fit.models.MultiPeakModel(
        npeaks=1,
        peak_shapes="lorentzian",
        convolve=True,
        segmented=True,
        oversample=3,
    )
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        Fit2DTool,
        "_show_error",
        lambda _self, title, text: errors.append((title, text)),
    )

    win = erlab.interactive.ftool(darr, model=model, execute=False)
    qtbot.addWidget(win)

    assert isinstance(win, Fit2DTool)
    assert win._data.dims == ("sample_temp",)
    assert win._model.func.convolve
    assert not win._model.func.segmented
    win._update_fit_curve()
    assert errors == []
    assert win._last_residual is not None
    assert win._last_residual.shape == sample_temp.shape


def test_fit2d_persistence_roundtrip_preserves_sparse_results(
    qtbot, exp_decay_model
) -> None:
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

    _seed_fit2d_full_results(win, exp_decay_model, params)

    win._result_ds_full[1] = None
    win.y_index_spin.setValue(0)
    win._mark_fit_stale()
    expected_results = [
        None if ds is None else ds.copy(deep=True) for ds in win._result_ds_full
    ]
    expected_status = win.tool_status.model_dump()

    win_restored = erlab.interactive.utils.ToolWindow.from_dataset(win.to_dataset())
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, Fit2DTool)

    assert [ds is not None for ds in win_restored._result_ds_full] == [
        True,
        False,
        True,
    ]
    _assert_fit_result_list_equivalent(win_restored._result_ds_full, expected_results)
    assert win_restored.tool_status.model_dump() == expected_status
    assert win_restored._fit_is_current is False
    assert not win_restored.copy_full_button.isEnabled()
    assert not win_restored.save_full_button.isEnabled()
    assert win_restored.current_provenance_spec() is None


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


def test_fit2d_full_copy_fit_data_name_with_domain_and_normalization(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.domain_min_line.setValue(-0.5)
    win.domain_max_line.setValue(0.5)
    win.normalize_check.setChecked(True)

    lines: list[str] = []
    result_name = win._full_copy_fit_data_name("data", lines=lines)

    assert result_name == "data_crop_norm"
    assert len(lines) == 2
    assert lines[0].startswith("data_crop = data.sel(")
    assert ".isel(" in lines[0]
    assert lines[1] == 'data_crop_norm = data_crop / data_crop.mean("x")'
    assert win._full_copy_fit_data_name("data") == "data_crop_norm"


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
    assert not code
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
    assert not code
    assert warnings


def test_fit2d_copy_code_full_inconsistent_params_warning(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    params1 = win._params.copy()
    params2 = win._params.copy()
    del params2["p0_center"]

    win._params_full = [params1, params2]
    win._result_ds_full = [xr.Dataset(), xr.Dataset()]
    win.y_min_spin.setValue(0)
    win.y_max_spin.setValue(1)

    warnings: list[tuple[str, str]] = []

    def _warn(title: str, text: str) -> None:
        warnings.append((title, text))

    monkeypatch.setattr(win, "_show_warning", _warn)
    code = win._copy_code_full()
    assert not code
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


def test_fit2d_invalid_bound_edit_warns_without_param_update(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    param = win.param_model.param_at(0)
    param.set(value=0.0, min=-1.0, max=2.0)
    changed: list[bool] = []
    warnings: list[tuple[str, str]] = []
    win.param_model.sigParamsChanged.connect(lambda: changed.append(True))
    monkeypatch.setattr(
        win,
        "_show_warning",
        lambda title, text: warnings.append((title, text)),
    )

    min_index = win.param_model.index(0, 3)
    assert not win.param_model.setData(
        min_index, "2.0", QtCore.Qt.ItemDataRole.EditRole
    )

    assert (param.value, param.min, param.max) == (0.0, -1.0, 2.0)
    assert changed == []
    assert warnings


def test_fit2d_start_error_resets_sequence_state(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    errors: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        win,
        "_show_error",
        lambda title, text, detailed_text=None: errors.append(
            (title, text, detailed_text)
        ),
    )
    param = win.param_model.param_at(0)
    param.min = 1.0
    param.max = 1.0

    win._run_fit_2d("up")

    assert errors
    assert win._fit_thread is None
    assert win._fit_cancel_requested is False
    assert win._fit_2d_total == 0
    assert win._fit_2d_indices == []
    assert win.fit_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


def test_fit2d_preparation_error_resets_sequence_state(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    errors: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        win,
        "_show_error",
        lambda title, text, detailed_text=None: errors.append(
            (title, text, detailed_text)
        ),
    )

    def _raise_fit_data() -> xr.DataArray:
        raise RuntimeError("unexpected preparation failure")

    monkeypatch.setattr(win, "_fit_data", _raise_fit_data)

    win._run_fit_2d("up")

    assert errors
    assert win._fit_thread is None
    assert win._fit_cancel_requested is False
    assert win._fit_2d_total == 0
    assert win._fit_2d_indices == []
    assert win.fit_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


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
    _seed_fit2d_param_results(
        win, [params.copy() for _ in range(len(win._params_full))]
    )

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
    _seed_fit2d_param_results(win, [params_0, params_1, params_2])
    win.param_plot_combo.setCurrentText(center_name)

    values = win._param_plot_dataarray(center_name)
    stderr = win._param_plot_dataarray(center_name, stderr=True)
    np.testing.assert_allclose(values.values, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(stderr.values, [0.01, 0.02, 0.0])
    assert values.name == f"{center_name}_values"
    assert stderr.name == f"{center_name}_stderr"

    saved: list[xr.DataArray] = []
    shown: list[tuple[xr.DataArray, str]] = []
    monkeypatch.setattr(
        win.param_plot,
        "_save_dataarray_as_hdf5",
        lambda da: saved.append(da.copy(deep=True)),
    )
    monkeypatch.setattr(
        win,
        "_show_dataarray_in_itool",
        lambda da, *, output_id=None: shown.append((da.copy(deep=True), output_id)),
    )

    win.param_plot._save_parameter_values()
    win.param_plot._save_parameter_stderr()
    win.param_plot._show_parameter_values()
    win.param_plot._show_parameter_stderr()

    assert [da.name for da in saved] == [
        f"{center_name}_values",
        f"{center_name}_stderr",
    ]
    assert [da.name for da, _ in shown] == [
        f"{center_name}_values",
        f"{center_name}_stderr",
    ]
    assert [output_id for _, output_id in shown] == [
        Fit2DTool._parameter_output_id(Fit2DTool.Output.PARAMETER_VALUES, center_name),
        Fit2DTool._parameter_output_id(Fit2DTool.Output.PARAMETER_STDERR, center_name),
    ]


def test_fit2d_parameter_output_provenance_uses_distinct_active_names(qtbot) -> None:
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
    _seed_fit2d_param_results(win, [params_0, params_1, params_2])
    win.param_plot_combo.setCurrentText(center_name)

    values = win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES)
    stderr = win.output_imagetool_data(Fit2DTool.Output.PARAMETER_STDERR)
    assert values is not None
    assert stderr is not None
    values_output_id = Fit2DTool._parameter_output_id(
        Fit2DTool.Output.PARAMETER_VALUES, center_name
    )
    stderr_output_id = Fit2DTool._parameter_output_id(
        Fit2DTool.Output.PARAMETER_STDERR, center_name
    )

    values_spec = win.output_imagetool_provenance(
        Fit2DTool.Output.PARAMETER_VALUES, values
    )
    stderr_spec = win.output_imagetool_provenance(
        Fit2DTool.Output.PARAMETER_STDERR, stderr
    )

    assert values_spec is not None
    assert stderr_spec is not None
    assert values_spec.active_name == "parameter_values"
    assert stderr_spec.active_name == "parameter_stderr"

    values_code = values_spec.display_code()
    stderr_code = stderr_spec.display_code()
    assert values_code is not None
    assert stderr_code is not None
    assert ".modelfit_coefficients.sel(param='p0_center')" in values_code
    assert ".modelfit_stderr.sel(param='p0_center')" in stderr_code
    assert ".rename(" not in values_code
    assert ".rename(" not in stderr_code

    win.param_plot_combo.setCurrentText("p0_width")
    bound_values = win.output_imagetool_data(values_output_id)
    bound_stderr = win.output_imagetool_data(stderr_output_id)
    assert bound_values is not None
    assert bound_stderr is not None
    xr.testing.assert_identical(bound_values, values)
    xr.testing.assert_identical(bound_stderr, stderr)

    bound_values_spec = win.output_imagetool_provenance(values_output_id, bound_values)
    assert bound_values_spec is not None
    bound_values_code = bound_values_spec.display_code()
    assert bound_values_code is not None
    assert ".modelfit_coefficients.sel(param='p0_center')" in bound_values_code
    assert ".modelfit_coefficients.sel(param='p0_width')" not in bound_values_code

    malformed_id = f"{Fit2DTool.Output.PARAMETER_VALUES.value}:"
    missing_id = Fit2DTool._parameter_output_id(
        Fit2DTool.Output.PARAMETER_VALUES, "does_not_exist"
    )
    assert win.output_imagetool_data(malformed_id) is None
    assert win.output_imagetool_data(missing_id) is None
    assert win.output_imagetool_provenance(malformed_id, values) is None
    assert win.output_imagetool_provenance(missing_id, values) is None


def test_fit2d_parameter_output_resolution_edges(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    center_name = "p0_center"
    params = win._params.copy()
    params[center_name].set(value=0.1)
    params[center_name].stderr = 0.01
    _seed_fit2d_param_results(
        win, [params.copy() for _ in range(len(win._params_full))]
    )
    win.param_plot_combo.setCurrentText(center_name)

    with pytest.raises(ValueError, match="Fit2DTool parameter output"):
        Fit2DTool._parameter_output_id("not-an-output", center_name)  # type: ignore[arg-type]
    assert Fit2DTool._parameter_output_parts("other.output") is None
    with pytest.raises(ValueError, match="does not define ImageTool output"):
        win._image_output_definition("other.output")

    values = win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES)
    assert values is not None
    values_output_id = Fit2DTool._parameter_output_id(
        Fit2DTool.Output.PARAMETER_VALUES, center_name
    )

    with pytest.raises(ValueError, match="does not define ImageTool output"):
        win.output_imagetool_data("other.output")
    with pytest.raises(ValueError, match="does not define ImageTool output"):
        win.output_imagetool_provenance("other.output", values)

    with monkeypatch.context() as patch:
        patch.setattr(win, "_full_fit_expression", lambda **_kwargs: "")
        assert win.output_imagetool_provenance(values_output_id, values) is None

    direct_input = provenance.script(
        start_label="Start from watched data",
        seed_code="derived = watched_data",
        active_name="derived",
    )
    win.set_input_provenance_spec(direct_input)
    direct_spec = win.output_imagetool_provenance(values_output_id, values)
    assert direct_spec is not None
    assert direct_spec.start_label == "Start from watched data"

    with monkeypatch.context() as patch:
        patch.setattr(provenance, "to_replay_provenance_spec", lambda _spec: None)
        with pytest.raises(RuntimeError, match="Could not convert local provenance"):
            win.output_imagetool_provenance(values_output_id, values)

    win.param_plot_combo.clear()
    assert win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES) is None
    assert win.output_imagetool_data(Fit2DTool.Output.PARAMETER_STDERR) is None
    assert (
        win.output_imagetool_provenance(Fit2DTool.Output.PARAMETER_VALUES, values)
        is None
    )
    assert (
        win.output_imagetool_provenance(Fit2DTool.Output.PARAMETER_STDERR, values)
        is None
    )


def test_fit2d_show_dataarray_in_itool_uses_detached_launcher(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    calls: list[tuple[xr.DataArray, object]] = []
    return_widget = QtWidgets.QWidget()
    qtbot.addWidget(return_widget)

    def _launch_stub(
        data: xr.DataArray,
        *,
        provenance_spec: object,
    ) -> QtWidgets.QWidget:
        calls.append((data, provenance_spec))
        return return_widget

    monkeypatch.setattr(win, "_launch_detached_output_imagetool", _launch_stub)

    da = xr.DataArray(np.arange(3.0), dims=("y",), coords={"y": np.arange(3)})
    win._show_dataarray_in_itool(da)
    assert calls
    assert calls[0][1] is None
    assert win._itool is return_widget


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


def test_fit2d_param_plot_rejects_cached_guess_params(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )

    guessed = win._params.copy()
    guessed["p0_center"].set(value=0.25)
    win._params_full = [guessed.copy() for _ in win._params_full]
    win._result_ds_full = [None for _ in win._result_ds_full]
    win._update_param_plot_options()

    assert win.param_plot_combo.count() == 0
    assert not win.param_plot_combo.isEnabled()
    assert not win.param_plot_overlay_check.isEnabled()
    assert win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES) is None
    assert (
        win.output_imagetool_data(
            Fit2DTool._parameter_output_id(
                Fit2DTool.Output.PARAMETER_VALUES, "p0_center"
            )
        )
        is None
    )

    win.param_plot_combo.setCurrentText("p0_center")
    win.param_plot._show_parameter_values()

    assert warnings
    assert warnings[-1][0] == "No parameter selected"


def test_fit2d_param_plot_rejects_placeholder_result_objects(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    guessed = win._params.copy()
    guessed["p0_center"].set(value=0.25)
    win._params_full = [guessed.copy() for _ in win._params_full]
    win._result_ds_full = [
        _placeholder_fit_result_dataset(guessed) for _ in win._result_ds_full
    ]
    win._update_param_plot_options()

    assert win.param_plot_combo.count() == 0
    assert not win.param_plot_combo.isEnabled()
    assert not win.param_plot_overlay_check.isEnabled()
    assert win._param_plot_names() == []
    assert win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES) is None
    assert (
        win.output_imagetool_data(
            Fit2DTool._parameter_output_id(
                Fit2DTool.Output.PARAMETER_VALUES, "p0_center"
            )
        )
        is None
    )


def test_fit2d_param_plot_rejects_unfitted_model_results(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    guessed = win._params.copy()
    guessed["p0_center"].set(value=0.25)
    win._params_full = [guessed.copy() for _ in win._params_full]
    win._result_ds_full = [
        _fit_result_dataset(guessed, nfev=0) for _ in win._result_ds_full
    ]
    win._update_param_plot_options()

    assert win._param_plot_names() == []
    assert win.param_plot_combo.count() == 0
    assert not win.param_plot_combo.isEnabled()
    assert win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES) is None
    assert (
        win.output_imagetool_data(
            Fit2DTool._parameter_output_id(
                Fit2DTool.Output.PARAMETER_VALUES, "p0_center"
            )
        )
        is None
    )
    assert win._param_plot_dataarray("p0_center").size == 0


def test_fit2d_index_changes_do_not_expose_guess_params(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._params_full[0] = win._params.copy()
    win._set_current_index(0)
    win._update_param_plot_options()

    assert win._params_full[0] is not None
    assert win.param_plot_combo.count() == 0
    assert win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES) is None

    fitted = win._params.copy()
    fitted["p0_center"].set(value=0.5)
    win._result_ds_full[0] = _fit_result_dataset(fitted)
    win._update_param_plot_options()

    assert "p0_center" in {
        win.param_plot_combo.itemText(i) for i in range(win.param_plot_combo.count())
    }


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


def test_fit2d_param_plot_save_dataarray_as_hdf5_handles_write_error(
    qtbot, monkeypatch, tmp_path
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    class _DialogStub:
        AcceptMode = QtWidgets.QFileDialog.AcceptMode
        FileMode = QtWidgets.QFileDialog.FileMode
        Option = QtWidgets.QFileDialog.Option

        def __init__(self, *args, **kwargs) -> None:
            self._init_args = (args, kwargs)

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
            self._directory = directory

        def exec(self) -> bool:
            return True

        def selectedFiles(self) -> list[str]:
            return [str(tmp_path / "locked.h5")]

    def _raise_write_error(self, *args, **kwargs) -> None:
        raise BlockingIOError("locked")

    critical_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(QtWidgets, "QFileDialog", _DialogStub)
    monkeypatch.setattr(xr.DataArray, "to_netcdf", _raise_write_error)
    monkeypatch.setattr(pg.PlotItem, "lastFileDir", str(tmp_path / "previous"))
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda *args, **kwargs: critical_calls.append((args, kwargs)) or 0,
    )

    win.param_plot._save_dataarray_as_hdf5(xr.DataArray([1.0], name="data"))

    assert len(critical_calls) == 1
    assert pg.PlotItem.lastFileDir == str(tmp_path / "previous")


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

    monkeypatch.setattr(
        win, "_launch_detached_output_imagetool", lambda *args, **kwargs: None
    )
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
    assert win.param_plot_combo.count() == 0

    _seed_fit2d_param_results(
        win, [win._params.copy() for _ in range(len(win._params_full))]
    )
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
