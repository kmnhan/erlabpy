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
from erlab.interactive import _fit1d as fit1d_module
from erlab.interactive._code_trust import (
    create_entry,
    issue_execution_capability,
    new_document_trust,
    untrusted_document_trust,
)
from erlab.interactive._code_trust._api import _document_trust_after_save
from erlab.interactive._code_trust._payloads import store_code_payload_entries
from erlab.interactive._figurecomposer import (
    FigureAxesSelectionState,
    FigureMethodFamily,
    FigureMethodPlotValueState,
)
from erlab.interactive._fit2d import Fit2DTool
from erlab.interactive.imagetool._provenance._execution import replay_script_provenance
from erlab.interactive.imagetool._provenance._model import ScriptInput, script
from erlab.interactive.imagetool._provenance._operations import (
    ModelFitOperation,
    ScriptCodeOperation,
)
from tests._qt_helpers import signal_receiver_count


def _exec_generated_code(code: str, **namespace_items: object) -> dict[str, object]:
    namespace = dict(namespace_items)
    exec(code, namespace, namespace)  # noqa: S102
    return namespace


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


def _make_linear_fit2d_tool(
    qtbot, *, expression: bool = False
) -> tuple[Fit2DTool, lmfit.Model, lmfit.Parameters]:
    model = lmfit.models.LinearModel()
    params = model.make_params(slope=1.0, intercept=2.0 if expression else 0.0)
    if expression:
        params["intercept"].expr = "2 * slope"
    tool = erlab.interactive.ftool(
        _make_2d_data(), model=model, params=params, execute=False
    )
    qtbot.addWidget(tool)
    if not isinstance(tool, Fit2DTool):  # pragma: no cover
        raise TypeError("Expected Fit2DTool")
    return tool, model, params


def test_fit2d_inherits_reviewed_model_file_loading(
    qtbot, tmp_path, monkeypatch
) -> None:
    tool, _model, _params = _make_linear_fit2d_tool(qtbot, expression=True)
    _set_signed_fit_trust(tool)
    candidate = lmfit.models.ExpressionModel("slope * x + intercept")
    serialized = candidate.dumps()
    model_path = tmp_path / "candidate.model"
    model_path.write_text(serialized)
    file_index = tool.model_combo.findData("__file")
    events: list[str] = []
    original_loads = lmfit.model.Model.loads

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(model_path), ""),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "confirm_code_trust",
        lambda *_args, **_kwargs: events.append("review") or True,
    )

    def tracked_loads(model, text, *args, **kwargs):
        events.append("decode")
        assert text == serialized
        return original_loads(model, text, *args, **kwargs)

    monkeypatch.setattr(lmfit.model.Model, "loads", tracked_loads)
    with QtCore.QSignalBlocker(tool.model_combo):
        tool.model_combo.setCurrentIndex(file_index)

    tool._on_model_choice_changed(file_index)

    assert events == ["review", "decode"]
    assert isinstance(tool._model, lmfit.models.ExpressionModel)
    assert tool._model.expr == candidate.expr
    assert tool._model_load_path == str(model_path)
    assert _trust_allows_local_code_edit(tool._document_trust)


def _set_signed_fit_trust(tool: Fit2DTool) -> None:
    manifest = tool._current_code_trust_manifest()
    assert manifest is not None
    signed = _document_trust_after_save(
        new_document_trust(),
        manifest,
        saved_trusted_lineage=True,
        signature_stored=True,
    )
    tool.set_document_trust(signed, notify=False)


def _authorize_execution(entries):
    _trust, capability = issue_execution_capability(new_document_trust(), entries)
    if capability is None:  # pragma: no cover - local trust always authorizes.
        raise RuntimeError("Could not authorize test execution")
    return capability


def _trust_allows_local_code_edit(trust) -> bool:
    entry = create_entry("test.local-edit", "test", "result = source")
    return issue_execution_capability(trust, (entry,))[1] is not None


def test_fit_dataset_settings_require_common_values(monkeypatch) -> None:
    model = lmfit.models.LinearModel()

    class _Result:
        def __init__(self, scale_covar: bool, *, weighted: bool = False) -> None:
            self.model = model
            self.scale_covar = scale_covar
            self.weights = np.ones(2) if weighted else None

    common = xr.Dataset(
        {
            "modelfit_results": xr.DataArray(
                [_Result(False), _Result(False)], dims=("fit",)
            )
        }
    )
    mixed = common.copy()
    mixed["modelfit_results"] = xr.DataArray(
        [_Result(False), _Result(True)], dims=("fit",)
    )
    missing = xr.Dataset(
        {"modelfit_results": xr.DataArray([_Result(False), object()], dims=("fit",))}
    )
    weighted = xr.Dataset(
        {
            "modelfit_results": xr.DataArray(
                [_Result(False, weighted=True), _Result(False, weighted=True)],
                dims=("fit",),
            )
        }
    )
    mixed_weighted = xr.Dataset(
        {
            "modelfit_results": xr.DataArray(
                [_Result(False), _Result(False, weighted=True)],
                dims=("fit",),
            )
        }
    )

    assert fit2d_module._fit_dataset_settings(common) == (False, False)
    assert fit2d_module._fit_dataset_settings(weighted) == (False, True)
    assert fit2d_module._fit_dataset_settings(mixed_weighted) == (False, None)
    assert fit2d_module._fit_dataset_settings(mixed) == (None, False)
    assert fit2d_module._fit_dataset_settings(missing) == (None, None)
    assert fit2d_module._fit_dataset_settings(xr.Dataset()) == (None, None)

    original_compute = xr.DataArray.compute

    def _raise_compute(self, **kwargs):
        if self.name == "modelfit_results":
            raise RuntimeError("cannot load result")
        return original_compute(self, **kwargs)

    monkeypatch.setattr(xr.DataArray, "compute", _raise_compute)
    assert fit2d_module._fit_dataset_settings(common) == (None, None)


def test_ftool_uncertainty_name_fallback(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    uncertainty = xr.full_like(data, 0.2)

    def _raise_argname(*_args, **_kwargs):
        raise RuntimeError("name unavailable")

    monkeypatch.setattr(fit2d_module.varname, "argname", _raise_argname)
    win = erlab.interactive.ftool(data, uncertainty=uncertainty, execute=False)
    qtbot.addWidget(win)

    assert win.tool_status.uncertainty_name == "uncertainty"

    model = lmfit.models.LinearModel()
    fit_ds = data.xlm.modelfit("x", model=model, params=model.make_params()).load()
    restored = erlab.interactive.ftool(fit_ds, execute=False)
    qtbot.addWidget(restored)
    assert restored.tool_status.data_name == "(fit_result)['modelfit_data']"

    named = erlab.interactive.ftool(fit_ds, data_name="saved_fit", execute=False)
    qtbot.addWidget(named)
    assert named.tool_status.data_name == "(saved_fit)['modelfit_data']"


def _assert_fit_result_dataset_equivalent(
    actual: xr.Dataset,
    expected: xr.Dataset,
    *,
    require_model_type: bool = True,
) -> None:
    xr.testing.assert_identical(
        actual.drop_vars("modelfit_results"),
        expected.drop_vars("modelfit_results"),
    )
    actual_result = actual.modelfit_results.compute().item()
    expected_result = expected.modelfit_results.compute().item()
    if require_model_type:
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
    actual: list[xr.Dataset | None],
    expected: list[xr.Dataset | None],
    *,
    require_model_type: bool = True,
) -> None:
    assert len(actual) == len(expected)
    for actual_ds, expected_ds in zip(actual, expected, strict=True):
        if expected_ds is None:
            assert actual_ds is None
            continue
        assert actual_ds is not None
        _assert_fit_result_dataset_equivalent(
            actual_ds,
            expected_ds,
            require_model_type=require_model_type,
        )


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
        fit_ds = win._fit_result_with_range(fit_ds)
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
    win._result_ds_full = [
        win._fit_result_with_range(_fit_result_dataset(params))
        for params in params_list
    ]
    win._fit_is_current = True
    win._update_full_fit_saveable()
    win._update_param_plot_options()


def _fit_slices_with_ranges(
    win: Fit2DTool, fit_ranges: list[tuple[float, float]]
) -> None:
    for index, fit_range in enumerate(fit_ranges):
        win._set_current_index(index)
        win.domain_min_spin.setValue(fit_range[0])
        win.domain_max_spin.setValue(fit_range[1])
        fit_ds = (
            win._fit_data()
            .xlm.modelfit(
                win._coord_name,
                model=win._model,
                params=win._params,
                method=win.method_combo.currentText(),
            )
            .load()
        )
        win._last_result_ds = fit_ds
        result = fit_ds.modelfit_results.compute().item()
        win._params = result.params.copy()
        win._sync_fit_result_state()
    win._fit_is_current = True
    win._update_full_fit_saveable()


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
    store_code_payload_entries(
        ds.attrs,
        (Fit2DTool._fit_result_code_trust_entry(ds[result_var].values),),
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


def test_fit2d_update_inputs_preserves_transpose_orientation(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win._do_transpose()
    win.y_index_spin.setValue(2)

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) * 1.1
    win.update_inputs({"data": updated})

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
            win.to_dataset(), _code_trust=new_document_trust()
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
        restored = erlab.interactive.utils.ToolWindow.from_dataset(
            saved, _code_trust=new_document_trust()
        )
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

    win.cbar.set_colormap("plasma", 0.6, reverse=True, high_contrast=True)
    win.cbar.setSpanRegion((0.5, 1.0))
    win._do_transpose()
    win.y_index_spin.setValue(3)
    status = win.tool_status
    assert win.cbar.colormap_properties == {
        "cmap": "plasma",
        "gamma": pytest.approx(0.6),
        "reverse": True,
        "high_contrast": True,
        "zero_centered": False,
    }
    assert win.cbar.spanRegion() == pytest.approx((0.5, 1.0))

    win_restored = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, Fit2DTool)
    win_restored.tool_status = status

    xr.testing.assert_identical(win_restored.tool_data, data.transpose("x", "y"))
    assert win_restored.y_index_spin.value() == 3

    win_roundtripped = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(), _code_trust=new_document_trust()
    )
    qtbot.addWidget(win_roundtripped)
    assert isinstance(win_roundtripped, Fit2DTool)
    xr.testing.assert_identical(win_roundtripped.tool_data, data.transpose("x", "y"))
    assert win_roundtripped.y_index_spin.value() == 3
    assert win_roundtripped.cbar.colormap_properties == win.cbar.colormap_properties
    assert win_roundtripped.cbar.spanRegion() == pytest.approx((0.5, 1.0))

    updated = data.copy(deep=True)
    updated.data = np.asarray(updated.data) + 1.0
    win_roundtripped.update_inputs({"data": updated})
    xr.testing.assert_identical(win_roundtripped.tool_data, updated.transpose("x", "y"))
    qtbot.wait_until(
        lambda: win_roundtripped.cbar.spanRegion() == pytest.approx((0.5, 1.0))
    )


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


def test_fit2d_weighted_fit_broadcasts_uncertainty(
    qtbot, exp_decay_model, monkeypatch
) -> None:
    t = np.linspace(0.0, 4.0, 25)
    y = np.arange(3)
    data = xr.DataArray(
        np.stack([((1.0 + 0.5 * idx) * np.exp(-t / 2.0)) for idx in y]),
        dims=("y", "t"),
        coords={"y": y, "t": t},
        name="decay2d",
    )
    uncertainty = xr.DataArray(
        np.linspace(0.1, 0.2, t.size),
        dims=("t",),
        coords={"t": t},
        name="sigma",
    )
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data,
        model=exp_decay_model,
        params=params,
        uncertainty=uncertainty,
        data_name="decay2d",
        model_name="model",
        uncertainty_name="sigma.copy()",
        execute=False,
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    warnings, errors = _configure_fit2d_for_tests(win, monkeypatch)

    assert win.uncertainty is uncertainty
    assert not win.scale_covar_check.isChecked()
    np.testing.assert_allclose(win.data_errorbar.opts["top"], uncertainty)

    win.y_index_spin.setValue(win.y_min_spin.value())
    win.nfev_spin.setValue(0)
    win._run_fit_2d("up")
    qtbot.waitUntil(
        lambda: all(ds is not None for ds in win._result_ds_full), timeout=10000
    )

    for result_ds in win._result_ds_full:
        assert result_ds is not None
        result = result_ds.modelfit_results.compute().item()
        np.testing.assert_allclose(result.weights, 1.0 / uncertainty.values)
        assert result.scale_covar is False
    assert not warnings
    assert not errors

    unnormalized_namespace = {
        "decay2d": data,
        "model": exp_decay_model,
        "sigma": uncertainty,
        "xr": xr,
    }
    exec(win._copy_code_full(), unnormalized_namespace)  # noqa: S102
    unnormalized_replay = unnormalized_namespace["result"]
    for result in unnormalized_replay.modelfit_results.compute().values:
        np.testing.assert_allclose(result.weights, 1.0 / uncertainty.values)

    win.normalize_check.setChecked(True)
    namespace = dict(unnormalized_namespace)
    code = win._copy_code_full()
    assert "input_uncertainty" not in code
    exec(code, namespace)  # noqa: S102
    replayed = namespace["result"]
    for idx, result in enumerate(replayed.modelfit_results.compute().values):
        np.testing.assert_allclose(
            result.weights,
            abs(data.isel(y=idx).mean().item()) / uncertainty.values,
        )
        assert result.scale_covar is False

    slice_namespace = {
        "decay2d": data,
        "model": exp_decay_model,
        "sigma": uncertainty,
    }
    exec(win.copy_code_1d(), slice_namespace)  # noqa: S102
    replayed_slice = slice_namespace["result"]
    slice_result = replayed_slice.modelfit_results.compute().item()
    assert any(
        np.allclose(
            slice_result.weights,
            abs(data.isel(y=idx).mean().item()) / uncertainty.values,
        )
        for idx in range(data.sizes["y"])
    )

    assert win.current_provenance_spec() is None
    assert win.detached_output_imagetool_provenance(data) is None
    assert win._parameter_model_fit_operation("n0", stderr=False) is None
    values_output_id = Fit2DTool._parameter_output_id(
        Fit2DTool.Output.PARAMETER_VALUES,
        "n0",
    )
    values = win.output_imagetool_data(values_output_id)
    assert values is not None
    assert win.output_imagetool_provenance(values_output_id, values) is None

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(),
        _code_trust=new_document_trust(),
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit2DTool)
    xr.testing.assert_identical(restored.uncertainty, uncertainty)
    assert restored.scale_covar_check.isChecked() is False


def test_fit2d_managed_uncertainty_persistence_preserves_dimensions(qtbot) -> None:
    data = _make_2d_data()
    uncertainty = xr.full_like(data, 0.2).rename("sigma")
    win = erlab.interactive.ftool(data, uncertainty=uncertainty, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    bindings = (ScriptInput(name="data"), ScriptInput(name="uncertainty"))
    win.set_script_inputs(bindings, primary_input="data")

    items = win._persistence_data_items()
    assert "uncertainty" in items
    xr.testing.assert_identical(items["uncertainty"], uncertainty)
    assert items["uncertainty"].dims == data.dims

    restored = erlab.interactive.utils.ToolWindow.from_dataset(win.to_dataset())
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit2DTool)
    assert restored.script_inputs == bindings
    xr.testing.assert_identical(restored.uncertainty, uncertainty)
    assert restored.uncertainty is not None
    assert restored.uncertainty.dims == data.dims


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
    prelude = win._detached_full_copy_prelude(primary_input="derived_crop")
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


def test_fit2d_update_inputs_preserves_state_and_refit(
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
    win.update_inputs({"data": new_data})

    assert win.tool_status == status
    xr.testing.assert_identical(win.tool_data, new_data)
    assert win._fit_is_current is False
    assert not called

    win._last_result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)
    newer_data = new_data.copy(deep=True)
    newer_data.data = np.asarray(newer_data.data) * 1.05
    win.update_inputs({"data": newer_data})

    assert called == [True]


def test_fit2d_update_inputs_resizes_slice_state_and_keeps_param_sync(
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
    win.update_inputs({"data": new_data})

    assert win._current_idx == 1
    assert len(win._params_full) == 2
    assert len(win._params_from_coord_full) == 2
    assert win.y_index_spin.maximum() == 1
    xr.testing.assert_identical(win.tool_data, new_data)

    index = win.param_model.index(0, 1)
    assert win.param_model.setData(index, "3.0", QtCore.Qt.ItemDataRole.EditRole)
    assert win._params_full[win._current_idx] is not None
    assert win._params_full[win._current_idx]["n0"].value == pytest.approx(3.0)


def test_fit2d_update_inputs_preserves_initial_params_full_for_reset_all(
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
    win.update_inputs({"data": updated})

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


def test_fit2d_validate_update_inputs_invalid_input_keeps_existing_ui(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    old_central = win.centralWidget()
    bad_data = _make_1d_data()

    with pytest.raises(ValueError, match="2D DataArray"):
        win.validate_update_inputs({"data": bad_data})

    assert win.centralWidget() is old_central
    assert old_central is not None
    assert old_central.parent() is not None
    xr.testing.assert_identical(win.tool_data, data)


def test_fit2d_apply_inputs_returns_false_if_fit_thread_stays_alive(qtbot) -> None:
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

    script_input = ScriptInput(name="data")
    win.set_script_inputs((script_input,), primary_input="data")
    assert win._apply_inputs({"data": updated}, (script_input,)) is False
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
    win.update_inputs({"data": updated})
    assert (
        signal_receiver_count(win, win.sigFitFinished, "sigFitFinished")
        == initial_receivers
    )

    win._do_transpose()
    assert (
        signal_receiver_count(win, win.sigFitFinished, "sigFitFinished")
        == initial_receivers
    )


def test_fit2d_apply_inputs_auto_refit_after_waiting_cancelled_thread(
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

    script_input = ScriptInput(name="data")
    win.set_script_inputs((script_input,), primary_input="data")
    assert win._apply_inputs({"data": updated}, (script_input,)) is False
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

    def _refresh_contents_from_index(
        *,
        mark_fit_stale: bool = True,
        update_widgets: bool = True,
        elapsed: float | None = None,
        emit_info: bool = True,
        emit_param_changed: bool = True,
    ) -> None:
        del mark_fit_stale, elapsed
        refresh_modes.append((update_widgets, emit_info, emit_param_changed))

    started_steps: list[int] = []
    pending_callbacks = []
    monkeypatch.setattr(fit2d_module.time, "monotonic", _monotonic)
    monkeypatch.setattr(
        win, "_refresh_contents_from_index", _refresh_contents_from_index
    )
    monkeypatch.setattr(win, "_show_warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(win, "_show_error", lambda *args, **kwargs: None)

    def _queue_single_shot(receiver, msec, callback, *guards) -> None:
        assert receiver is win
        assert msec == 0
        assert not guards
        pending_callbacks.append(callback)

    monkeypatch.setattr(erlab.interactive.utils, "single_shot", _queue_single_shot)

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
    while pending_callbacks:
        pending_callbacks.pop(0)()

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
    original_replace_last_state = win._replace_last_state

    def record_replace_last_state() -> None:
        replaced.append(True)
        original_replace_last_state()

    monkeypatch.setattr(win, "_replace_last_state", record_replace_last_state)
    initial_revision = win.provenance_revision
    win._write_history = True
    win._begin_fit_2d_sequence_history()
    assert win._fit_2d_sequence_write_history is True
    assert win._write_history is False
    win._begin_fit_2d_sequence_history()
    win._write_state()
    win._write_state()
    assert win.provenance_revision == initial_revision
    win._finish_fit_2d_sequence_history()
    assert win._write_history is True
    assert replaced == [True]
    assert win.provenance_revision == initial_revision + 1

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

    win.scale_covar_check.setChecked(False)
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
    assert not win_restored.scale_covar_check.isChecked()


def test_fit2d_open_weighted_saved_fit_dataset(
    qtbot, exp_decay_model, monkeypatch
) -> None:
    t = np.linspace(0.0, 4.0, 25)
    y = np.arange(3)
    data = xr.DataArray(
        np.stack([((1.0 + 0.5 * idx) * np.exp(-t / 2.0)) for idx in y], axis=0),
        dims=("y", "t"),
        coords={"y": y, "t": t},
        name="decay2d",
    )
    weights = xr.DataArray(np.linspace(0.3, 2.7, t.size), dims="t", coords={"t": t})
    assert not np.array_equal(weights, 1.0 / (1.0 / weights))
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    weighted_fit_ds = data.xlm.modelfit(
        "t",
        model=exp_decay_model,
        params=params,
        weights=weights,
        scale_covar=False,
    ).load()

    win = erlab.interactive.ftool(weighted_fit_ds, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    xr.testing.assert_identical(
        win._direct_weights_full, weighted_fit_ds.modelfit_weights
    )
    np.testing.assert_array_equal(win._fit_weights(), weights)
    assert win.uncertainty is not None
    xr.testing.assert_allclose(win.uncertainty, 1.0 / weights)
    np.testing.assert_allclose(win.data_errorbar.opts["top"], 1.0 / weights)
    assert win._fit_is_current
    assert win.copy_full_button.isEnabled()
    assert win.save_full_button.isEnabled()

    namespace = {
        "weighted_fit_ds": weighted_fit_ds,
        "model": exp_decay_model,
        "xr": xr,
    }
    exec(win._copy_code_full(), namespace)  # noqa: S102
    xr.testing.assert_identical(
        namespace["result"].modelfit_weights,
        weighted_fit_ds.modelfit_weights,
    )
    for result in namespace["result"].modelfit_results.compute().values:
        np.testing.assert_array_equal(result.weights, weights)

    workspace_restored = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(),
        _code_trust=new_document_trust(),
    )
    qtbot.addWidget(workspace_restored)
    assert isinstance(workspace_restored, Fit2DTool)
    xr.testing.assert_identical(
        workspace_restored._direct_weights_full,
        weighted_fit_ds.modelfit_weights,
    )
    np.testing.assert_array_equal(workspace_restored._fit_weights(), weights)

    win.normalize_check.setChecked(True)
    normalized_namespace = {
        "weighted_fit_ds": weighted_fit_ds,
        "model": exp_decay_model,
        "xr": xr,
    }
    exec(win._copy_code_full(), normalized_namespace)  # noqa: S102
    for idx, result in enumerate(
        normalized_namespace["result"].modelfit_results.compute().values
    ):
        norm = abs(data.isel(y=idx).mean("t").item())
        np.testing.assert_allclose(result.weights, weights * norm)
    expected_normalized_weights = (weights * abs(data.mean("t"))).rename(
        "modelfit_weights"
    )
    xr.testing.assert_identical(
        normalized_namespace["result"].modelfit_weights,
        expected_normalized_weights,
    )
    assert workspace_restored._fit_is_current
    updated_data = data * 1.01
    assert workspace_restored.update_inputs({"data": updated_data})
    xr.testing.assert_identical(
        workspace_restored._direct_weights_full,
        weighted_fit_ds.modelfit_weights,
    )
    np.testing.assert_array_equal(workspace_restored._fit_weights(), weights)

    y_weights = xr.DataArray([0.5, 0.7, 0.9], dims="y", coords={"y": y})
    y_weighted_fit_ds = data.xlm.modelfit(
        "t",
        model=exp_decay_model,
        params=params,
        weights=y_weights,
        scale_covar=False,
    ).load()
    y_weighted = erlab.interactive.ftool(y_weighted_fit_ds, execute=False)
    qtbot.addWidget(y_weighted)
    assert isinstance(y_weighted, Fit2DTool)
    xr.testing.assert_identical(
        y_weighted._direct_weights_full,
        y_weighted_fit_ds.modelfit_weights,
    )
    xr.testing.assert_equal(y_weighted._fit_weights(), y_weights.isel(y=1))
    y_weighted.y_min_spin.setValue(1)
    y_weighted.y_max_spin.setValue(2)
    y_weighted_namespace = {
        "y_weighted_fit_ds": y_weighted_fit_ds,
        "model": exp_decay_model,
        "xr": xr,
    }
    exec(y_weighted._copy_code_full(), y_weighted_namespace)  # noqa: S102
    xr.testing.assert_identical(
        y_weighted_namespace["result"].modelfit_weights,
        y_weighted_fit_ds.modelfit_weights.isel(y=slice(1, 3)),
    )

    crop_x = np.linspace(-1.0, 1.0, 9)
    crop_data = xr.DataArray(
        np.stack((2.0 * crop_x + 1.0, 3.0 * crop_x - 1.0)),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": crop_x},
    )
    crop_weights = xr.DataArray(
        np.linspace(0.5, 1.5, crop_x.size), dims="x", coords={"x": crop_x}
    )
    crop_model = lmfit.models.LinearModel()
    crop_params = crop_model.make_params(slope=1.0, intercept=0.0)
    cropped = erlab.interactive.ftool(
        crop_data,
        model=crop_model,
        params=crop_params,
        data_name="crop_data",
        model_name="crop_model",
        execute=False,
    )
    qtbot.addWidget(cropped)
    assert isinstance(cropped, Fit2DTool)
    monkeypatch.setattr(
        cropped,
        "_show_warning",
        lambda title, text: pytest.fail(f"{title}: {text}"),
    )
    cropped._set_direct_weights(crop_weights, weights_name="crop_weights")
    direct_weight_lines: list[str] = []
    direct_weight_name = cropped._full_copy_fit_direct_weights_name(
        "crop_data", "crop_model", lines=direct_weight_lines
    )
    assert direct_weight_name is not None
    direct_weight_namespace = {"crop_weights": crop_weights}
    exec("\n".join(direct_weight_lines), direct_weight_namespace)  # noqa: S102
    xr.testing.assert_identical(
        direct_weight_namespace[direct_weight_name], crop_weights
    )
    _fit_slices_with_ranges(cropped, [(-0.5, 0.5), (-0.5, 0.5)])
    cropped_namespace = {
        "crop_data": crop_data,
        "crop_model": crop_model,
        "crop_weights": crop_weights,
        "xr": xr,
    }
    exec(cropped._copy_code_full(), cropped_namespace)  # noqa: S102
    xr.testing.assert_identical(
        cropped_namespace["result"].modelfit_weights,
        crop_weights.sel(x=slice(-0.5, 0.5)).rename("modelfit_weights"),
    )

    legacy = erlab.interactive.ftool(
        weighted_fit_ds.drop_vars("modelfit_weights"), execute=False
    )
    qtbot.addWidget(legacy)
    assert isinstance(legacy, Fit2DTool)
    assert legacy.uncertainty is None
    assert legacy.scale_covar_check.isChecked()
    assert not legacy._fit_is_current
    assert not legacy.copy_full_button.isEnabled()
    assert not legacy.save_full_button.isEnabled()


@pytest.mark.parametrize(
    ("weight_kind", "normalize"),
    [
        pytest.param("scalar", False, id="scalar"),
        pytest.param("fit", False, id="fit-dimension"),
        pytest.param("slice", False, id="slice-dimension"),
        pytest.param("full", False, id="two-dimensional"),
        pytest.param("fit", True, id="normalized-fit-dimension"),
    ],
)
def test_fit2d_save_full_preserves_direct_weight_dimensions(
    qtbot, monkeypatch, weight_kind, normalize
) -> None:
    x = np.linspace(-1.0, 1.0, 9)
    y = np.array([10.0, 20.0])
    data = xr.DataArray(
        np.stack((2.0 * x, 3.0 * x + 2.0)),
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    fit_weights = xr.DataArray(np.linspace(0.5, 1.5, x.size), dims="x", coords={"x": x})
    match weight_kind:
        case "scalar":
            weights = xr.DataArray(0.5)
        case "fit":
            weights = fit_weights
        case "slice":
            weights = xr.DataArray([0.5, 0.8], dims="y", coords={"y": y})
        case "full":
            weights = xr.DataArray(
                np.outer([0.5, 0.8], fit_weights.values),
                dims=("y", "x"),
                coords={"y": y, "x": x},
            )
        case _:
            raise ValueError(f"Unexpected weight kind: {weight_kind}")

    model = lmfit.models.LinearModel()
    fit_ds = data.xlm.modelfit(
        "x",
        model=model,
        params=model.make_params(slope=1.0, intercept=0.0),
        weights=weights,
        scale_covar=False,
    ).load()
    win = erlab.interactive.ftool(fit_ds, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    warnings, errors = _configure_fit2d_for_tests(win, monkeypatch)

    if normalize:
        win.normalize_check.setChecked(True)
        win.domain_min_spin.setValue(-0.5)
        win.domain_max_spin.setValue(0.5)
        win.y_index_spin.setValue(win.y_min_spin.value())
        win.nfev_spin.setValue(0)
        win._run_fit_2d("up")
        qtbot.waitUntil(lambda: not win._fit_2d_sequence_active(), timeout=10000)

    saved_fits: list[xr.Dataset] = []

    @contextlib.contextmanager
    def _wait_stub(_parent, _message):
        yield None

    monkeypatch.setattr(erlab.interactive.utils, "wait_dialog", _wait_stub)
    monkeypatch.setattr(
        erlab.interactive.utils,
        "save_fit_ui",
        lambda fit_result, *, parent: saved_fits.append(fit_result),
    )
    win._save_fit_full()

    assert len(saved_fits) == 1
    saved_weights = saved_fits[0].modelfit_weights
    if normalize:
        fit_data = data.sel(x=slice(-0.5, 0.5))
        normalization = abs(fit_data.mean("x"))
        normalization = normalization.where(~np.isclose(normalization, 0.0), 1.0)
        expected = (fit_weights.sel(x=fit_data.x) * normalization).rename(
            "modelfit_weights"
        )
        assert saved_weights.dims == expected.dims == ("x", "y")
        np.testing.assert_allclose(saved_weights.values, expected.values)
    else:
        expected = fit_ds.modelfit_weights
        assert saved_weights.dims == expected.dims
        np.testing.assert_array_equal(saved_weights.values, expected.values)
    for dim in expected.dims:
        np.testing.assert_array_equal(
            saved_weights.coords[dim].values, expected.coords[dim].values
        )
    reopened = erlab.interactive.ftool(saved_fits[0], execute=False)
    qtbot.addWidget(reopened)
    assert isinstance(reopened, Fit2DTool)
    assert reopened._fit_is_current
    xr.testing.assert_identical(reopened._direct_weights_full, saved_weights)
    assert not warnings
    assert not errors


def test_fit2d_rejects_ambiguous_internal_weighting(qtbot) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    weighting = xr.ones_like(data)
    with pytest.raises(ValueError, match="Only one"):
        win._init_full_data_state(
            data,
            uncertainty=weighting,
            direct_weights=weighting,
            data_name="data",
        )


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

    win_restored = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(), _code_trust=new_document_trust()
    )
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

    win_restored = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(), _code_trust=new_document_trust()
    )
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


def test_fit2d_deferred_restore_preserves_raw_fit_blob_until_needed(
    qtbot, exp_decay_model, monkeypatch
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
    saved = win.to_dataset()
    calls = []
    fail_next_deserialize = {"value": False}
    original = erlab.interactive.utils._deserialize_fit_dataset_blob

    def _tracked_deserialize(blob):
        calls.append(np.asarray(blob).size)
        if fail_next_deserialize["value"]:
            fail_next_deserialize["value"] = False
            raise RuntimeError("fit deserialize failed")
        return original(blob)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "_deserialize_fit_dataset_blob",
        _tracked_deserialize,
    )

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved,
        _defer_restore_work=True,
        _code_trust=new_document_trust(),
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit2DTool)
    assert calls == []
    assert restored._serialized_fit_result_blob is not None
    assert restored._pending_persisted_fit_is_current is True

    resaved = restored.to_dataset()
    assert calls == []
    np.testing.assert_array_equal(
        resaved[Fit2DTool._PERSISTED_FIT_RESULT_VAR].values,
        saved[Fit2DTool._PERSISTED_FIT_RESULT_VAR].values,
    )

    fail_next_deserialize["value"] = True
    with pytest.raises(RuntimeError, match="fit deserialize failed"):
        restored._flush_restore_work()
    assert len(calls) == 1
    assert restored._serialized_fit_result_blob is not None
    assert restored._pending_persisted_fit_is_current is True

    resaved_after_failure = restored.to_dataset()
    assert len(calls) == 1
    np.testing.assert_array_equal(
        resaved_after_failure[Fit2DTool._PERSISTED_FIT_RESULT_VAR].values,
        saved[Fit2DTool._PERSISTED_FIT_RESULT_VAR].values,
    )

    restored._flush_restore_work()

    assert len(calls) == 2
    assert restored._pending_persisted_fit_is_current is None
    _assert_fit_result_list_equivalent(restored._result_ds_full, expected_results)


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

    expected_param_names = set(win._model.param_names)

    def _param_plot_combo_items() -> set[str]:
        return {
            win.param_plot_combo.itemText(i)
            for i in range(win.param_plot_combo.count())
        }

    qtbot.waitUntil(
        lambda: expected_param_names.issubset(_param_plot_combo_items()),
        timeout=10000,
    )

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


def test_fit2d_fill_expression_commits_local_lineage(qtbot) -> None:
    tool, _model, params = _make_linear_fit2d_tool(qtbot, expression=True)
    source_params = params.copy()
    destination_params = params.copy()
    destination_params["intercept"].expr = None
    tool._params_full[0] = source_params
    tool._params_full[tool._current_idx] = destination_params
    tool._refresh_contents_from_index()
    tool._refresh_fit_code_entries()
    _set_signed_fit_trust(tool)

    tool._fill_params_from(0, mode="previous")

    current = tool._params_full[tool._current_idx]
    assert current is not None
    assert current["intercept"].expr == "2 * slope"
    assert _trust_allows_local_code_edit(tool._document_trust)
    assert tool._current_fit_execution_allowed()


def test_fit2d_blocked_fill_expression_does_not_mutate(
    qtbot,
    monkeypatch,
) -> None:
    tool, _model, params = _make_linear_fit2d_tool(qtbot, expression=True)
    source_params = params.copy()
    destination_params = params.copy()
    destination_params["intercept"].expr = None
    tool._params_full[0] = source_params
    tool._params_full[tool._current_idx] = destination_params
    tool._refresh_contents_from_index()
    tool._refresh_fit_code_entries()
    previous_entries = tool._fit_code_entries
    previous_params = tool._params_full[tool._current_idx]
    calls: list[None] = []
    copy_calls: list[lmfit.Parameters] = []
    original_copy = lmfit.Parameters.copy

    @contextlib.contextmanager
    def blocked_edit(*_args, **_kwargs):
        calls.append(None)
        yield None

    monkeypatch.setattr(tool, "_local_code_edit", blocked_edit)
    monkeypatch.setattr(
        lmfit.Parameters,
        "copy",
        lambda current: copy_calls.append(current) or original_copy(current),
    )

    tool._fill_params_from(0, mode="previous")

    assert calls == [None]
    assert copy_calls == []
    assert tool._params_full[tool._current_idx] is previous_params
    assert previous_params is not None
    assert previous_params["intercept"].expr is None
    assert tool._fit_code_entries == previous_entries


def test_fit2d_fill_expression_requires_model_context(qtbot, monkeypatch) -> None:
    model = lmfit.models.ExpressionModel("slope * x + intercept")
    params = model.make_params(slope=1.0, intercept=2.0)
    params["intercept"].expr = "2 * slope"
    tool = erlab.interactive.ftool(
        _make_2d_data(),
        model=model,
        params=params,
        execute=False,
    )
    qtbot.addWidget(tool)
    assert isinstance(tool, Fit2DTool)
    source_params = params.copy()
    destination_params = params.copy()
    destination_params["intercept"].expr = None
    tool._params_full[0] = source_params
    tool._params_full[tool._current_idx] = destination_params
    tool._refresh_contents_from_index()
    tool._refresh_fit_code_entries()
    candidate_params_full = tool._params_full.copy()
    candidate_params_full[tool._current_idx] = source_params
    candidate_entries = tool._fit_code_entries_with_params_full(candidate_params_full)
    parameter_entries = tuple(
        entry
        for entry in candidate_entries
        if entry.feature == tool._PARAMETER_CODE_TRUST_FEATURE
    )
    _trust, parameter_only_capability = issue_execution_capability(
        new_document_trust(),
        parameter_entries,
    )
    assert parameter_only_capability is not None
    copy_calls: list[lmfit.Parameters] = []
    original_copy = lmfit.Parameters.copy

    @contextlib.contextmanager
    def parameter_only_edit(*_args, **_kwargs):
        yield parameter_only_capability

    monkeypatch.setattr(tool, "_local_code_edit", parameter_only_edit)
    monkeypatch.setattr(
        lmfit.Parameters,
        "copy",
        lambda current: copy_calls.append(current) or original_copy(current),
    )

    tool._fill_params_from(0, mode="previous")

    assert copy_calls == []
    current = tool._params_full[tool._current_idx]
    assert current is destination_params
    assert current["intercept"].expr is None


def test_fit2d_reset_expression_commits_local_lineage(qtbot, monkeypatch) -> None:
    tool, _model, params = _make_linear_fit2d_tool(qtbot, expression=True)
    tool._params_full = [params.copy() for _ in tool._params_full]
    tool._initial_params = params.copy()
    tool._initial_params["intercept"].expr = None
    tool._refresh_contents_from_index()
    tool._refresh_fit_code_entries()
    _set_signed_fit_trust(tool)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    tool._reset_params_all()

    assert all(
        params is not None and params["intercept"].expr is None
        for params in tool._params_full
    )
    assert _trust_allows_local_code_edit(tool._document_trust)


def test_fit2d_blocked_reset_expression_does_not_mutate(
    qtbot,
    monkeypatch,
) -> None:
    tool, _model, params = _make_linear_fit2d_tool(qtbot)
    initial = params.copy()
    initial["intercept"].expr = "2 * slope"
    tool._initial_params_full = [initial.copy() for _ in tool._params_full]
    marker = xr.Dataset({"marker": xr.DataArray(1)})
    tool._result_ds_full[0] = marker
    tool._params_from_coord_full[0] = {"slope": "y"}
    previous_entries = tool._fit_code_entries
    previous_params_full = tool._params_full
    previous_result_full = tool._result_ds_full
    previous_sources_full = tool._params_from_coord_full
    copy_calls: list[lmfit.Parameters] = []
    original_copy = lmfit.Parameters.copy

    @contextlib.contextmanager
    def blocked_edit(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(tool, "_local_code_edit", blocked_edit)
    monkeypatch.setattr(
        lmfit.Parameters,
        "copy",
        lambda current: copy_calls.append(current) or original_copy(current),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    tool._reset_params_all()

    assert copy_calls == []
    assert tool._params_full is previous_params_full
    assert tool._result_ds_full is previous_result_full
    assert tool._params_from_coord_full is previous_sources_full
    assert tool._result_ds_full[0] is marker
    assert tool._params_from_coord_full[0] == {"slope": "y"}
    assert tool._fit_code_entries == previous_entries


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


def test_fit2d_uniform_slice_ranges_keep_single_crop_code(qtbot) -> None:
    x = np.linspace(-1.0, 1.0, 9)
    data = xr.DataArray(
        np.stack((x**2 + 0.1 * x, x**2 - 0.2 * x)),
        dims=("slice", "x"),
        coords={"slice": [10.0, 20.0], "x": x},
    )
    model = erlab.analysis.fit.models.PolynomialModel(degree=1)
    win = erlab.interactive.ftool(
        data,
        model=model,
        params=model.make_params(c0=0.0, c1=0.0),
        execute=False,
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.domain_min_spin.setValue(-0.6)
    win.domain_max_spin.setValue(0.6)
    expected_lines: list[str] = []
    expected_name = win._full_copy_fit_data_name("data", lines=expected_lines)

    _fit_slices_with_ranges(win, [(-0.6, 0.6), (-0.6, 0.6)])
    actual_lines: list[str] = []
    actual_name = win._full_copy_fit_data_name("data", lines=actual_lines)

    assert actual_name == expected_name == "data_crop"
    assert (
        actual_lines
        == expected_lines
        == ["data_crop = data.sel(x=slice(-0.6, 0.6)).isel(slice=slice(0, 2))"]
    )


def test_fit2d_records_range_from_dispatched_fit_data(qtbot) -> None:
    x = np.linspace(-1.0, 1.0, 9)
    data = xr.DataArray(
        np.stack((x**2 + 0.1 * x, x**2 - 0.2 * x)),
        dims=("slice", "x"),
        coords={"slice": [10.0, 20.0], "x": x},
    )
    model = erlab.analysis.fit.models.PolynomialModel(degree=1)
    win = erlab.interactive.ftool(
        data,
        model=model,
        params=model.make_params(c0=0.0, c1=0.0),
        execute=False,
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    win.domain_min_spin.setValue(-0.6)
    win.domain_max_spin.setValue(0.6)
    dispatched_data = win._fit_data()
    result_ds = dispatched_data.xlm.modelfit(
        win._coord_name,
        model=win._model,
        params=win._params,
    ).load()

    win.domain_min_spin.setValue(-1.0)
    win.domain_max_spin.setValue(0.0)
    win._set_fit_ds(result_ds, 0.0)

    assert win._last_result_ds is not None
    assert win._fit_result_range(win._last_result_ds) == (-0.6, 0.6)

    overwritten = win._fit_result_with_range(win._last_result_ds, (-0.5, 0.5))
    assert win._fit_result_range(overwritten) == (-0.5, 0.5)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("slice_dim", ["slice", "bound"])
def test_fit2d_mixed_slice_ranges_copy_and_output_provenance(
    qtbot, descending, slice_dim
) -> None:
    x = np.linspace(-1.0, 1.0, 9)
    if descending:
        x = x[::-1]
    data = xr.DataArray(
        np.stack((x**2 + 0.1 * x, x**2 - 0.2 * x)),
        dims=(slice_dim, "x"),
        coords={slice_dim: [10.0, 20.0], "x": x},
        name="spectrum",
    )
    model = erlab.analysis.fit.models.PolynomialModel(degree=1)
    win = erlab.interactive.ftool(
        data,
        model=model,
        params=model.make_params(c0=0.0, c1=0.0),
        data_name="source_spectrum",
        execute=False,
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    fit_ranges = [(-1.0, 0.0), (0.0, 1.0)]
    _fit_slices_with_ranges(win, fit_ranges)
    assert [
        win._fit_result_range(result_ds)
        for result_ds in win._result_ds_full
        if result_ds is not None
    ] == fit_ranges

    current_spec = win.current_provenance_spec()
    assert current_spec is not None
    current_code = current_spec.display_code()
    assert current_code is not None
    copied_code = win._copy_code_full()
    assert copied_code

    for code in (current_code, copied_code):
        namespace = _exec_generated_code(
            code,
            source_spectrum=data,
            era=erlab.analysis,
            np=np,
            xr=xr,
        )
        result = namespace["result"]
        assert isinstance(result, xr.Dataset)
        for index, fit_range in enumerate(fit_ranges):
            actual_x = result["x"].where(
                result.modelfit_data.isel({slice_dim: index}).notnull(), drop=True
            )
            expected_x = x[(x >= fit_range[0]) & (x <= fit_range[1])]
            np.testing.assert_allclose(actual_x, expected_x)

    win.param_plot_combo.setCurrentText("c1")
    values = win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES)
    assert values is not None
    output_spec = win.output_imagetool_provenance(
        Fit2DTool.Output.PARAMETER_VALUES,
        values,
    )
    assert output_spec is not None
    assert isinstance(output_spec.operations[-1], ScriptCodeOperation)
    replayed = replay_script_provenance(
        output_spec,
        {"source_spectrum": data},
        authorize=_authorize_execution,
    )
    xr.testing.assert_allclose(replayed, values)

    direct_weights = xr.DataArray(
        np.linspace(0.5, 1.5, x.size), dims="x", coords={"x": x}
    )
    win._set_direct_weights(direct_weights, weights_name="direct_weights")
    namespace = {
        "source_spectrum": data,
        "direct_weights": direct_weights,
        "era": erlab.analysis,
        "xr": xr,
    }
    exec(win._copy_code_full(), namespace)  # noqa: S102
    xr.testing.assert_identical(
        namespace["result"].modelfit_weights,
        direct_weights.rename("modelfit_weights"),
    )
    xr.testing.assert_identical(
        win._direct_weights_for_full_fit_results(fit_ranges),
        direct_weights.rename("modelfit_weights"),
    )


@pytest.mark.parametrize("descending", [False, True])
def test_fit2d_mixed_slice_ranges_persistence_roundtrip(
    qtbot, monkeypatch, descending
) -> None:
    x = np.linspace(-1.0, 1.0, 9)
    if descending:
        x = x[::-1]
    data = xr.DataArray(
        np.stack((x**2 + 0.1 * x, x**2 - 0.2 * x)),
        dims=("slice", "x"),
        coords={"slice": [10.0, 20.0], "x": x},
        name="spectrum",
    )
    model = erlab.analysis.fit.models.PolynomialModel(degree=1)
    win = erlab.interactive.ftool(
        data,
        model=model,
        params=model.make_params(c0=0.0, c1=0.0),
        data_name="source_spectrum",
        execute=False,
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    fit_ranges = [(-1.0, 0.0), (0.0, 1.0)]
    _fit_slices_with_ranges(win, fit_ranges)
    expected_results = [
        result_ds.copy(deep=True)
        for result_ds in win._result_ds_full
        if result_ds is not None
    ]

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(), _code_trust=new_document_trust()
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit2DTool)
    assert [
        restored._fit_result_range(result_ds)
        for result_ds in restored._result_ds_full
        if result_ds is not None
    ] == fit_ranges
    for index, (actual, expected) in enumerate(
        zip(restored._result_ds_full, expected_results, strict=True)
    ):
        assert actual is not None
        xr.testing.assert_identical(
            actual.drop_vars("modelfit_results"),
            expected.drop_vars("modelfit_results"),
        )
        expected_x = x[(x >= fit_ranges[index][0]) & (x <= fit_ranges[index][1])]
        np.testing.assert_allclose(actual["x"], expected_x)

    restored_spec = restored.current_provenance_spec()
    assert restored_spec is not None
    restored_code = restored_spec.display_code()
    assert restored_code is not None
    namespace = _exec_generated_code(
        restored_code,
        source_spectrum=data,
        era=erlab.analysis,
        np=np,
        xr=xr,
    )
    restored_result = namespace["result"]
    assert isinstance(restored_result, xr.Dataset)
    for index, fit_range in enumerate(fit_ranges):
        actual_x = restored_result["x"].where(
            restored_result.modelfit_data.isel(slice=index).notnull(), drop=True
        )
        expected_x = x[(x >= fit_range[0]) & (x <= fit_range[1])]
        np.testing.assert_allclose(actual_x, expected_x)

    saved_fits: list[xr.Dataset] = []
    monkeypatch.setattr(
        erlab.interactive.utils,
        "save_fit_ui",
        lambda fit_ds, *, parent: saved_fits.append(fit_ds),
    )
    win._save_fit_full()
    assert len(saved_fits) == 1

    reopened = erlab.interactive.ftool(saved_fits[0], execute=False)
    qtbot.addWidget(reopened)
    assert isinstance(reopened, Fit2DTool)
    assert [
        reopened._fit_result_range(result_ds)
        for result_ds in reopened._result_ds_full
        if result_ds is not None
    ] == fit_ranges
    for index, result_ds in enumerate(reopened._result_ds_full):
        assert result_ds is not None
        expected_x = x[(x >= fit_ranges[index][0]) & (x <= fit_ranges[index][1])]
        np.testing.assert_allclose(result_ds["x"], expected_x)

    reopened._set_current_index(0)
    reopened.domain_min_spin.setValue(-0.5)
    reopened.domain_max_spin.setValue(0.0)
    refit_data = reopened._fit_data()
    assert float(refit_data.coords[Fit2DTool._FIT_RANGE_MIN_COORD]) == -0.5
    assert float(refit_data.coords[Fit2DTool._FIT_RANGE_MAX_COORD]) == 0.0
    refit_ds = refit_data.xlm.modelfit(
        reopened._coord_name,
        model=reopened._model,
        params=reopened._params,
    ).load()
    reopened._set_fit_ds(refit_ds, 0.0)
    assert reopened._last_result_ds is not None
    assert reopened._fit_result_range(reopened._last_result_ds) == (-0.5, 0.0)


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
    np.testing.assert_allclose(stderr.values, [0.01, 0.02, np.nan])
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
    action_names = {action.objectName() for action in win.param_plot.vb.menu.actions()}
    assert "fit2dParamPlotOpenInFtoolAction" in action_names
    assert "fit2dParamPlotAddToFigureAction" in action_names


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
    weighted_fit_values = win.output_imagetool_data(
        Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT
    )
    stderr = win.output_imagetool_data(Fit2DTool.Output.PARAMETER_STDERR)
    assert values is not None
    assert weighted_fit_values is not None
    assert stderr is not None
    np.testing.assert_allclose(values.values, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(weighted_fit_values.values, [0.1, 0.2, np.nan])
    np.testing.assert_allclose(stderr.values, [0.01, 0.02, np.nan])
    values_output_id = Fit2DTool._parameter_output_id(
        Fit2DTool.Output.PARAMETER_VALUES, center_name
    )
    weighted_fit_values_output_id = Fit2DTool._parameter_output_id(
        Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT, center_name
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
    weighted_fit_values_spec = win.output_imagetool_provenance(
        weighted_fit_values_output_id, weighted_fit_values
    )

    assert values_spec is not None
    assert stderr_spec is not None
    assert weighted_fit_values_spec is not None
    assert values_spec.active_name == "parameter_values"
    assert stderr_spec.active_name == "parameter_stderr"
    assert weighted_fit_values_spec.active_name == "parameter_values"
    assert isinstance(values_spec.operations[-1], ModelFitOperation)
    assert isinstance(stderr_spec.operations[-1], ModelFitOperation)
    assert isinstance(weighted_fit_values_spec.operations[-1], ScriptCodeOperation)

    values_code = values_spec.display_code()
    stderr_code = stderr_spec.display_code()
    weighted_fit_values_code = weighted_fit_values_spec.display_code()
    assert values_code is not None
    assert stderr_code is not None
    assert weighted_fit_values_code is not None
    assert ".modelfit_coefficients.sel(" in values_code
    assert ".modelfit_stderr.sel(" in stderr_code
    assert "param='p0_center'" in values_code
    assert "param='p0_center'" in stderr_code
    assert "fit_data = " in values_code
    assert "fit_data = " in stderr_code
    assert "_itool_replay_" not in values_code
    assert "_itool_replay_" not in stderr_code
    assert "result =" not in values_code
    assert "result =" not in stderr_code
    assert "imagetool" not in values_code
    assert "imagetool" not in stderr_code
    assert max(map(len, values_code.splitlines())) <= 88
    assert max(map(len, stderr_code.splitlines())) <= 88
    assert max(map(len, weighted_fit_values_code.splitlines())) <= 88

    fit_data = data.isel(y=slice(0, 3))
    expected_values = values_spec.operations[-1].apply(fit_data)
    expected_stderr = stderr_spec.operations[-1].apply(fit_data)
    values_namespace = _exec_generated_code(values_code, data=data)
    stderr_namespace = _exec_generated_code(stderr_code, data=data)
    weighted_fit_values_namespace = _exec_generated_code(
        weighted_fit_values_code,
        data=data,
        era=erlab.analysis,
        xr=xr,
    )
    xr.testing.assert_identical(values_namespace["parameter_values"], expected_values)
    xr.testing.assert_identical(stderr_namespace["parameter_stderr"], expected_stderr)
    expected_weighted_values = expected_values.where(
        np.isfinite(expected_values)
        & np.isfinite(expected_stderr)
        & (expected_stderr > 0)
    )
    xr.testing.assert_identical(
        weighted_fit_values_namespace["parameter_values"], expected_weighted_values
    )

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
    assert "param='p0_center'" in bound_values_code
    assert "param='p0_width'" not in bound_values_code

    win._set_uncertainty(xr.ones_like(win._data_full))
    assert win.output_imagetool_provenance(values_output_id, values) is None
    win._set_uncertainty(None)

    win.scale_covar_check.setChecked(False)
    win.scale_covar_check.setChecked(True)
    assert not win._fit_is_current
    assert win.output_imagetool_provenance(values_output_id, values) is None

    malformed_id = f"{Fit2DTool.Output.PARAMETER_VALUES.value}:"
    missing_id = Fit2DTool._parameter_output_id(
        Fit2DTool.Output.PARAMETER_VALUES, "does_not_exist"
    )
    assert win.output_imagetool_data(malformed_id) is None
    assert win.output_imagetool_data(missing_id) is None
    assert win.output_imagetool_provenance(malformed_id, values) is None
    assert win.output_imagetool_provenance(missing_id, values) is None


def test_fit2d_parameter_output_provenance_preserves_managed_uncertainty(
    qtbot,
) -> None:
    x = np.linspace(-1.0, 1.0, 9)
    y = np.array([0.0, 1.0])
    data = xr.DataArray(
        np.stack((1.0 + 2.0 * x + x**2, 2.0 - x + 0.5 * x**2)),
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="data",
    )
    uncertainty = xr.DataArray(
        np.broadcast_to(np.linspace(0.1, 0.5, x.size), data.shape),
        dims=data.dims,
        coords=data.coords,
        name="uncertainty",
    )
    model = erlab.analysis.fit.models.PolynomialModel(degree=1)
    params = model.make_params(c0=0.0, c1=0.0)
    win = erlab.interactive.ftool(
        data,
        model=model,
        params=params,
        uncertainty=uncertainty,
        data_name="data",
        uncertainty_name="uncertainty",
        scale_covar=True,
        execute=False,
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)
    win.set_script_inputs(
        (ScriptInput(name="data"), ScriptInput(name="uncertainty")),
        primary_input="data",
    )

    fitted_params = []
    for _ in y:
        fitted = params.copy()
        fitted["c1"].stderr = 0.1
        fitted_params.append(fitted)
    _seed_fit2d_param_results(win, fitted_params)
    win.param_plot_combo.setCurrentText("c1")

    expected_fit = data.xlm.modelfit(
        "x",
        model=model,
        params=params,
        weights=1.0 / uncertainty,
        method=win.method_combo.currentText(),
        scale_covar=True,
    ).load()
    expected_values = expected_fit.modelfit_coefficients.sel(
        param="c1", drop=True
    ).rename("c1_values")
    expected_stderr = (
        expected_fit.modelfit_stderr.sel(param="c1", drop=True)
        .where(lambda error: np.isfinite(error) & (error > 0))
        .rename("c1_stderr")
    )
    expected_weighted_values = expected_values.where(
        np.isfinite(expected_values)
        & np.isfinite(expected_stderr)
        & (expected_stderr > 0)
    )

    expected_outputs = {
        Fit2DTool.Output.PARAMETER_VALUES: expected_values,
        Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT: expected_weighted_values,
        Fit2DTool.Output.PARAMETER_STDERR: expected_stderr,
    }
    for output, expected in expected_outputs.items():
        output_id = win._parameter_output_id(output, "c1")
        output_data = win.output_imagetool_data(output_id)
        assert output_data is not None
        spec = win.output_imagetool_provenance(output_id, output_data)
        assert spec is not None
        assert isinstance(spec.operations[-1], ScriptCodeOperation)
        replayed = replay_script_provenance(
            spec,
            {"data": data, "uncertainty": uncertainty},
            authorize=_authorize_execution,
        )
        xr.testing.assert_allclose(replayed, expected)


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
        patch.setattr(win, "_full_fit_parameter_specs", lambda **_kwargs: None)
        assert win.output_imagetool_provenance(values_output_id, values) is None

    direct_input = script(
        start_label="Start from watched data",
        seed_code="derived = watched_data",
        active_name="derived",
    )
    win.set_script_inputs(
        (
            ScriptInput(
                name="derived",
                provenance_spec=direct_input.model_dump(mode="json"),
            ),
        ),
        primary_input="derived",
    )
    direct_spec = win.output_imagetool_provenance(values_output_id, values)
    assert direct_spec is not None
    assert direct_spec.script_inputs == win.script_inputs
    assert direct_spec.start_label == "Start from current fit-tool input data"
    direct_code = direct_spec.display_code()
    assert direct_code is not None
    assert "watched_data.isel" in direct_code
    direct_namespace = _exec_generated_code(direct_code, watched_data=data)
    expected_direct = direct_spec.operations[-1].apply(data.isel(y=slice(0, 3)))
    xr.testing.assert_identical(direct_namespace["parameter_values"], expected_direct)

    with monkeypatch.context() as patch:
        patch.setattr(win, "_infer_model_choice", lambda _model: "ExpressionModel")
        fallback_spec = win.output_imagetool_provenance(values_output_id, values)
    assert fallback_spec is not None
    assert isinstance(fallback_spec.operations[-1], ScriptCodeOperation)
    fallback_code = fallback_spec.display_code()
    assert fallback_code is not None
    assert "import lmfit" in fallback_code
    assert "imagetool" not in fallback_code

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


def test_fit2d_parameter_output_provenance_preserves_standalone_data_name(
    qtbot,
) -> None:
    source_spectrum = _make_2d_data()
    win = erlab.interactive.ftool(
        source_spectrum,
        data_name="source_spectrum",
        execute=False,
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    center_name = "p0_center"
    params = win._params.copy()
    _seed_fit2d_param_results(
        win,
        [params.copy() for _ in range(len(win._params_full))],
    )
    win.param_plot_combo.setCurrentText(center_name)
    values = win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES)
    assert values is not None

    spec = win.output_imagetool_provenance(
        Fit2DTool._parameter_output_id(
            Fit2DTool.Output.PARAMETER_VALUES,
            center_name,
        ),
        values,
    )
    assert spec is not None
    code = spec.display_code()
    assert code is not None
    assert "source_spectrum.isel" in code

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        namespace = _exec_generated_code(code, source_spectrum=source_spectrum)
    assert spec.active_name is not None
    actual = namespace[spec.active_name]
    assert isinstance(actual, xr.DataArray)
    xr.testing.assert_allclose(actual, values)


def test_fit2d_expression_model_parameter_output_provenance_executes(qtbot) -> None:
    x = np.linspace(-1.0, 1.0, 11)
    y = np.arange(3)
    slopes = np.array([2.0, 4.0, 6.0])
    intercepts = np.array([1.0, 2.0, 3.0])
    data = xr.DataArray(
        intercepts[:, None] + slopes[:, None] * x,
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    model = lmfit.models.ExpressionModel("slope * x + intercept")
    win = erlab.interactive.ftool(
        data,
        model=model,
        data_name="source_spectrum",
        execute=False,
    )
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    params_list = []
    for slope, intercept in zip(slopes, intercepts, strict=True):
        params = win._params.copy()
        params["slope"].set(value=slope)
        params["intercept"].set(value=intercept)
        params_list.append(params)
    _seed_fit2d_param_results(win, params_list)
    win.param_plot_combo.setCurrentText("slope")

    values = win.output_imagetool_data(Fit2DTool.Output.PARAMETER_VALUES)
    assert values is not None
    spec = win.output_imagetool_provenance(
        Fit2DTool.Output.PARAMETER_VALUES,
        values,
    )
    assert spec is not None
    assert isinstance(spec.operations[-1], ScriptCodeOperation)
    code = spec.display_code()
    assert code is not None
    assert "import lmfit" in code
    assert "source_spectrum.isel" in code
    assert "imagetool" not in code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        replayed = replay_script_provenance(
            spec,
            {"source_spectrum": data},
            authorize=_authorize_execution,
        )
    xr.testing.assert_allclose(replayed, values)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        namespace = _exec_generated_code(code, source_spectrum=data, xr=xr)
    actual = namespace["parameter_values"]
    assert isinstance(actual, xr.DataArray)
    xr.testing.assert_allclose(actual, values)


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
    win.param_plot._open_parameter_values_in_ftool()
    win.param_plot._add_parameter_plot_to_figure()

    assert len(warnings) == 6
    assert not saved
    assert not shown


def _seed_param_plot_for_figure(
    win: Fit2DTool,
    param_name: str,
    *,
    stderrs: tuple[float | None, float | None, float | None] = (0.01, 0.02, 0.03),
) -> None:
    params_0 = win._params.copy()
    params_1 = win._params.copy()
    params_2 = win._params.copy()
    for idx, (params, stderr) in enumerate(
        zip((params_0, params_1, params_2), stderrs, strict=True), start=1
    ):
        params[param_name].set(value=0.1 * idx)
        params[param_name].stderr = stderr
    _seed_fit2d_param_results(win, [params_0, params_1, params_2])
    win.param_plot_combo.setCurrentText(param_name)


class _FakeFigureNode:
    def __init__(self, script_name: str) -> None:
        self.script_name = script_name


_DEFAULT_FIGURE_PROMPT = object()


class _FakeFigureManager:
    def __init__(
        self,
        managed: Fit2DTool,
        *,
        figure_uids: tuple[str, ...] = (),
        append_return: bool = True,
        prompt_return: object = _DEFAULT_FIGURE_PROMPT,
        weighted_ftool_return: str | None = "weighted_ftool",
    ) -> None:
        self._managed = managed
        self._figure_uids_value = figure_uids
        self._append_return = append_return
        self._weighted_ftool_return = weighted_ftool_return
        if prompt_return is _DEFAULT_FIGURE_PROMPT:
            prompt_return = (
                (
                    figure_uids[0],
                    FigureAxesSelectionState(axes=((0, 0),)),
                )
                if figure_uids
                else None
            )
        self._prompt_return = prompt_return
        self.nodes = {
            "values_target": _FakeFigureNode("parameter_values"),
            "stderr_target": _FakeFigureNode("parameter_stderr"),
        }
        self.prompt_calls: list[object] = []
        self.create_calls: list[tuple[tuple[str, ...], object, str | None]] = []
        self.append_calls: list[
            tuple[tuple[str, ...], str | None, object | None, object]
        ] = []
        self.weighted_ftool_calls: list[tuple[str, str]] = []
        self.remove_calls: list[str] = []

    def _node_uid_from_window(self, widget) -> str | None:
        return "ftool" if widget is self._managed else None

    def _node_for_target(self, target: str) -> _FakeFigureNode:
        return self.nodes[target]

    def _script_input_name_for_node(self, node: _FakeFigureNode) -> str:
        return node.script_name

    def _figure_uids(self) -> tuple[str, ...]:
        return self._figure_uids_value

    def _choose_figure_append_target(self, operation):
        self.prompt_calls.append(operation)
        return self._prompt_return

    def create_figure_from_targets(
        self,
        targets,
        *,
        operation=None,
        title: str | None = None,
    ) -> str:
        self.create_calls.append((tuple(targets), operation, title))
        self._figure_uids_value = ("figure_new",)
        return "figure_new"

    def append_figure_from_targets(
        self,
        targets,
        *,
        figure_uid: str | None = None,
        axes_selection=None,
        operation=None,
    ) -> bool:
        self.append_calls.append(
            (tuple(targets), figure_uid, axes_selection, operation)
        )
        return self._append_return

    def open_weighted_ftool(self, values_target: str, stderr_target: str) -> str | None:
        self.weighted_ftool_calls.append((values_target, stderr_target))
        return self._weighted_ftool_return

    def _remove_childtool(self, target: str) -> None:
        self.remove_calls.append(target)


def test_fit2d_param_plot_open_parameter_values_uses_managed_outputs(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    param_name = "p0_center"
    _seed_param_plot_for_figure(win, param_name, stderrs=(0.01, None, 0.03))
    manager = _FakeFigureManager(win)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    output_calls: list[tuple[str, Fit2DTool.Output, xr.DataArray]] = []

    def _target_stub(param: str, output: Fit2DTool.Output, da: xr.DataArray) -> str:
        output_calls.append((param, output, da.copy(deep=True)))
        return (
            "values_target"
            if output == Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT
            else "stderr_target"
        )

    param_data_calls = 0
    param_plot_data = win._param_plot_data

    def _param_plot_data(param: str):
        nonlocal param_data_calls
        param_data_calls += 1
        return param_plot_data(param)

    monkeypatch.setattr(win, "_param_plot_data", _param_plot_data)
    monkeypatch.setattr(win, "_parameter_output_target", _target_stub)

    win.param_plot._open_parameter_values_in_ftool()

    assert [(param, output, da.name) for param, output, da in output_calls] == [
        (
            param_name,
            Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT,
            f"{param_name}_values",
        ),
        (param_name, Fit2DTool.Output.PARAMETER_STDERR, f"{param_name}_stderr"),
    ]
    np.testing.assert_allclose(output_calls[0][2].values, [0.1, np.nan, 0.3])
    np.testing.assert_allclose(output_calls[1][2].values, [0.01, np.nan, 0.03])
    assert param_data_calls == 1
    np.testing.assert_allclose(
        win._param_plot_dataarray(param_name).values, [0.1, 0.2, 0.3]
    )
    assert manager.weighted_ftool_calls == [("values_target", "stderr_target")]
    assert not manager.create_calls
    assert not manager.append_calls


def test_fit2d_param_plot_open_parameter_values_requires_managed_context(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    _seed_param_plot_for_figure(win, "p0_center")
    monkeypatch.setattr(erlab.interactive.imagetool.manager, "_manager_instance", None)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win.param_plot,
        "_current_param_dataarrays",
        lambda **_kwargs: pytest.fail("parameter data was computed without a manager"),
    )
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )

    win.param_plot._open_parameter_values_in_ftool()

    assert len(warnings) == 1


def test_fit2d_param_plot_open_parameter_values_does_not_launch_without_valid_stderr(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    param_name = "p0_center"
    _seed_param_plot_for_figure(win, param_name, stderrs=(None, None, None))
    manager = _FakeFigureManager(win)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    output_calls: list[object] = []
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_parameter_output_target", lambda *_args: output_calls.append(None)
    )
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )

    win.param_plot._open_parameter_values_in_ftool()

    assert len(warnings) == 1
    assert not output_calls
    assert not manager.weighted_ftool_calls


@pytest.mark.parametrize(
    ("failed_output", "values_preexisting", "expected_removals"),
    [
        (Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT, False, []),
        (Fit2DTool.Output.PARAMETER_STDERR, False, ["values_target"]),
        (Fit2DTool.Output.PARAMETER_STDERR, True, []),
    ],
)
def test_fit2d_param_plot_open_parameter_values_does_not_launch_on_output_failure(
    qtbot,
    monkeypatch,
    failed_output: Fit2DTool.Output,
    values_preexisting: bool,
    expected_removals: list[str],
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    _seed_param_plot_for_figure(win, "p0_center")
    manager = _FakeFigureManager(win)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    output_calls: list[Fit2DTool.Output] = []

    def _target_stub(
        _param: str, output: Fit2DTool.Output, _data: xr.DataArray
    ) -> str | None:
        output_calls.append(output)
        if output == failed_output:
            return None
        return (
            "values_target"
            if output == Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT
            else None
        )

    if values_preexisting:
        values_output_id = win._parameter_output_id(
            Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT, "p0_center"
        )
        monkeypatch.setattr(
            win,
            "_output_imagetool_target",
            lambda output_id: (
                "values_target" if output_id == values_output_id else None
            ),
        )
    monkeypatch.setattr(win, "_parameter_output_target", _target_stub)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )

    win.param_plot._open_parameter_values_in_ftool()

    expected_calls = [Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT]
    if failed_output == Fit2DTool.Output.PARAMETER_STDERR:
        expected_calls.append(Fit2DTool.Output.PARAMETER_STDERR)
    assert output_calls == expected_calls
    assert manager.remove_calls == expected_removals
    assert len(warnings) == 1
    assert not manager.weighted_ftool_calls


def test_fit2d_param_plot_open_parameter_values_cleans_up_failed_launch(
    qtbot, monkeypatch
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    param_name = "p0_center"
    _seed_param_plot_for_figure(win, param_name)
    manager = _FakeFigureManager(win, weighted_ftool_return=None)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    monkeypatch.setattr(
        win,
        "_parameter_output_target",
        lambda _param, output, _data: (
            "values_target"
            if output == Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT
            else "stderr_target"
        ),
    )
    cleared_output_ids: list[str] = []
    monkeypatch.setattr(
        win, "_clear_output_imagetool_target", cleared_output_ids.append
    )

    win.param_plot._open_parameter_values_in_ftool()

    assert manager.weighted_ftool_calls == [("values_target", "stderr_target")]
    assert manager.remove_calls == ["values_target", "stderr_target"]
    assert cleared_output_ids == [
        win._parameter_output_id(
            Fit2DTool.Output.PARAMETER_VALUES_FOR_WEIGHTED_FIT, param_name
        ),
        win._parameter_output_id(Fit2DTool.Output.PARAMETER_STDERR, param_name),
    ]


def test_fit2d_param_plot_add_to_figure_requires_manager(qtbot, monkeypatch) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    param_name = "p0_center"
    _seed_param_plot_for_figure(win, param_name)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )
    monkeypatch.setattr(erlab.interactive.imagetool.manager, "_manager_instance", None)

    win.param_plot._add_parameter_plot_to_figure()

    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("figure_uids", "expected_create", "append_return"),
    [
        ((), True, True),
        (("figure_existing",), False, True),
        (("figure_existing",), False, False),
    ],
)
def test_fit2d_param_plot_add_to_figure_manager_paths(
    qtbot,
    monkeypatch,
    figure_uids: tuple[str, ...],
    expected_create: bool,
    append_return: bool,
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    param_name = "p0_center"
    _seed_param_plot_for_figure(win, param_name)
    manager = _FakeFigureManager(
        win, figure_uids=figure_uids, append_return=append_return
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )
    output_calls: list[tuple[str, Fit2DTool.Output, str]] = []

    def _target_stub(
        param: str, output: Fit2DTool.Output, da: xr.DataArray
    ) -> str | None:
        output_calls.append((param, output, str(da.name)))
        if output == Fit2DTool.Output.PARAMETER_VALUES:
            return "values_target"
        return "stderr_target"

    monkeypatch.setattr(win, "_parameter_output_target", _target_stub)

    win.param_plot._add_parameter_plot_to_figure()

    assert output_calls == [
        (param_name, Fit2DTool.Output.PARAMETER_VALUES, f"{param_name}_values"),
        (param_name, Fit2DTool.Output.PARAMETER_STDERR, f"{param_name}_stderr"),
    ]
    if expected_create:
        assert len(manager.create_calls) == 1
        assert not manager.append_calls
        targets, operation, title = manager.create_calls[0]
        assert title == f"{param_name} parameter plot"
    else:
        assert len(manager.append_calls) == 1
        assert not manager.create_calls
        targets, figure_uid, axes_selection, operation = manager.append_calls[0]
        assert len(manager.prompt_calls) == 1
        assert manager.prompt_calls[0].method_name == "errorbar"
        assert figure_uid == "figure_existing"
        assert axes_selection == FigureAxesSelectionState(axes=((0, 0),))
    if append_return:
        assert not warnings
    else:
        assert len(warnings) == 1
    assert targets == ("values_target", "stderr_target")
    assert operation.method_family == FigureMethodFamily.AXES
    assert operation.method_name == "errorbar"
    assert operation.method_plot_data_mode == "from_data"
    assert operation.method_plot_x == FigureMethodPlotValueState(
        source="parameter_values", kind="coord", name=win._y_dim_name
    )
    assert operation.method_plot_y == FigureMethodPlotValueState(
        source="parameter_values", kind="data"
    )
    assert operation.method_plot_xerr is None
    assert operation.method_plot_yerr == FigureMethodPlotValueState(
        source="parameter_stderr", kind="data"
    )
    assert operation.method_kwargs["label"] == param_name


def test_fit2d_param_plot_add_to_figure_cancel_does_not_open_outputs(
    qtbot,
    monkeypatch,
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    param_name = "p0_center"
    _seed_param_plot_for_figure(win, param_name)
    manager = _FakeFigureManager(
        win,
        figure_uids=("figure_existing",),
        prompt_return=None,
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    output_calls: list[tuple[str, Fit2DTool.Output, str]] = []
    monkeypatch.setattr(
        win,
        "_parameter_output_target",
        lambda param, output, da: output_calls.append((param, output, str(da.name))),
    )

    win.param_plot._add_parameter_plot_to_figure()

    assert len(manager.prompt_calls) == 1
    assert manager.prompt_calls[0].method_name == "errorbar"
    assert output_calls == []
    assert not manager.create_calls
    assert not manager.append_calls


def test_fit2d_param_plot_add_to_figure_warns_when_outputs_fail(
    qtbot,
    monkeypatch,
) -> None:
    data = _make_2d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit2DTool)

    _seed_param_plot_for_figure(win, "p0_center")
    manager = _FakeFigureManager(win)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager, "_manager_instance", manager
    )
    monkeypatch.setattr(win, "_parameter_output_target", lambda *_args: None)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win, "_show_warning", lambda title, text: warnings.append((title, text))
    )

    win.param_plot._add_parameter_plot_to_figure()

    assert len(warnings) == 1
    assert not manager.create_calls
    assert not manager.append_calls


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

    assert len(warnings) == 1


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


def test_fit2d_init_broadcast_params(qtbot, exp_decay_model) -> None:
    data = _make_2d_data()
    n0_values = xr.DataArray([1.0, 2.0, 3.0], dims=("y",)).chunk({"y": 1})

    win = Fit2DTool(
        data,
        model=exp_decay_model,
        params={
            "n0": {"value": n0_values, "min": 0.0},
            "tau": {"value": 2.0, "vary": False},
        },
    )
    qtbot.addWidget(win)

    assert [params["n0"].value for params in win._params_full] == [1.0, 2.0, 3.0]
    assert all(params["n0"].min == 0.0 for params in win._params_full)
    assert all(params["tau"].value == 2.0 for params in win._params_full)
    assert all(not params["tau"].vary for params in win._params_full)


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


def test_fit2d_blocked_model_edit_does_not_merge_parameters(qtbot, monkeypatch) -> None:
    win, model, _params = _make_linear_fit2d_tool(qtbot, expression=True)
    win.set_document_trust(untrusted_document_trust())
    saved_params = [
        None if params is None else win._serialize_params(params)
        for params in win._params_full
    ]
    monkeypatch.setattr(
        win,
        "_merge_params",
        lambda *_args, **_kwargs: pytest.fail("blocked edit evaluated parameters"),
    )

    win._refresh_multipeak_model()

    assert win._model is model
    assert [
        None if params is None else win._serialize_params(params)
        for params in win._params_full
    ] == saved_params


def test_fit2d_saved_expressions_wait_for_document_approval(qtbot, monkeypatch) -> None:
    source, _model, _params = _make_linear_fit2d_tool(qtbot, expression=True)
    saved = source.to_dataset()
    saved_status = Fit2DTool.StateModel.model_validate_json(saved.attrs["tool_state"])
    deserialize_calls: list[object] = []
    original_deserialize = Fit2DTool._deserialize_params

    def tracked_deserialize(state, **kwargs):
        deserialize_calls.append(state)
        return original_deserialize(state, **kwargs)

    monkeypatch.setattr(
        Fit2DTool,
        "_deserialize_params",
        staticmethod(tracked_deserialize),
    )

    restored = erlab.interactive.utils.ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit2DTool)

    assert deserialize_calls == []
    assert restored._pending_fit_status == saved_status

    def fail_local_edit(*_args, **_kwargs):
        raise AssertionError("stored status retried as a local edit")

    monkeypatch.setattr(restored, "_local_code_edit", fail_local_edit)

    restored.set_document_trust(new_document_trust())

    assert restored._pending_fit_status is None
    assert deserialize_calls
    assert restored._params["intercept"].expr == "2 * slope"
    assert type(restored)._code_trust_entries_from_status(restored.tool_status) == type(
        source
    )._code_trust_entries_from_status(saved_status)


def test_fit2d_status_edit_authorizes_per_slice_expression_decode(
    qtbot, monkeypatch
) -> None:
    tool, _model, _params = _make_linear_fit2d_tool(qtbot, expression=True)
    status = tool.tool_status
    state2d = status.state2d
    if state2d is None:  # pragma: no cover - Fit2D status always has 2D state.
        raise RuntimeError("Fit2D status does not contain per-slice state")
    template = next(params for params in state2d.params_full if params is not None)
    added = [tuple(item) for item in template]
    intercept_index = next(
        index for index, item in enumerate(added) if item[0] == "intercept"
    )
    intercept = list(added[intercept_index])
    intercept[3] = "3 * slope"
    added[intercept_index] = tuple(intercept)
    params_full = list(state2d.params_full)
    params_full[0] = added
    changed = status.model_copy(
        update={"state2d": state2d.model_copy(update={"params_full": params_full})}
    )
    _set_signed_fit_trust(tool)
    boundary_lineage: list[bool] = []
    edit_scope_active = False
    original_deserialize = tool._deserialize_params
    original_local_code_edit = tool._local_code_edit

    @contextlib.contextmanager
    def tracked_local_code_edit(*args, **kwargs):
        nonlocal edit_scope_active
        with original_local_code_edit(*args, **kwargs) as capability:
            edit_scope_active = True
            try:
                yield capability
            finally:
                edit_scope_active = False

    def tracked_deserialize(state, **kwargs):
        boundary_lineage.append(edit_scope_active)
        return original_deserialize(state, **kwargs)

    monkeypatch.setattr(tool, "_local_code_edit", tracked_local_code_edit)
    monkeypatch.setattr(tool, "_deserialize_params", tracked_deserialize)

    tool.tool_status = changed

    assert boundary_lineage
    assert all(boundary_lineage)
    assert tool._params_full[0] is not None
    assert tool._params_full[0]["intercept"].expr == "3 * slope"
    assert _trust_allows_local_code_edit(tool._document_trust)


def test_fit2d_sequence_serializes_full_results_once(qtbot, monkeypatch) -> None:
    tool, model, params = _make_linear_fit2d_tool(qtbot)
    data = tool.tool_data

    result_ds = data.isel(y=0).xlm.modelfit("x", model=model, params=params).load()
    serialized_datasets: list[xr.Dataset] = []
    invalidation_calls = 0
    original_invalidate = tool._invalidate_fit_result_payload

    def tracked_invalidate() -> None:
        nonlocal invalidation_calls
        invalidation_calls += 1
        original_invalidate()

    def tracked_serialize(dataset: xr.Dataset) -> np.ndarray:
        serialized_datasets.append(dataset)
        return np.array([1], dtype=np.uint8)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "_serialize_fit_dataset_blob",
        tracked_serialize,
    )
    monkeypatch.setattr(tool, "_invalidate_fit_result_payload", tracked_invalidate)
    tool._fit_2d_total = data.sizes["y"]
    for index in range(data.sizes["y"]):
        tool._store_fit_2d_sequence_result(index, result_ds, 0.0)

    assert serialized_datasets == []
    assert invalidation_calls == data.sizes["y"]

    tool._fit_2d_last_completed_idx = data.sizes["y"] - 1
    tool._finish_fit_2d_sequence()

    assert len(serialized_datasets) == 1
    assert invalidation_calls == data.sizes["y"]


def test_fit2d_multi_fit_caches_updated_full_result(qtbot) -> None:
    tool, model, params = _make_linear_fit2d_tool(qtbot)
    data = tool.tool_data

    current_data = data.isel(y=tool._current_idx)
    old_result = current_data.xlm.modelfit(
        "x", model=model, params=params, max_nfev=1
    ).load()
    new_result = (
        (current_data + 5.0)
        .xlm.modelfit("x", model=model, params=params, max_nfev=1)
        .load()
    )
    tool._last_result_ds = old_result
    tool._sync_fit_result_state(notify=False)
    tool._cache_fit_result_payload()

    tool._store_multi_fit_result(new_result, fit1d_module.time.perf_counter())
    tool._sync_multi_fit_view(full=True)
    expected_results = [
        None if result is None else result.copy(deep=True)
        for result in tool._result_ds_full
    ]

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        tool.to_dataset(), _code_trust=new_document_trust()
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit2DTool)
    _assert_fit_result_list_equivalent(
        restored._result_ds_full,
        expected_results,
        require_model_type=False,
    )


def test_fit2d_numeric_results_and_redraw_reuse_cached_code_inventory(
    qtbot, monkeypatch
) -> None:
    x = np.linspace(-1.0, 1.0, 5)
    y = np.arange(128.0)
    data = xr.DataArray(
        y[:, None] + x[None, :],
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="map",
    )
    model = lmfit.models.ExpressionModel("slope * x + intercept")
    params = model.make_params(slope=1.0, intercept=0.0)
    tool = erlab.interactive.ftool(data, model=model, params=params, execute=False)
    qtbot.addWidget(tool)
    assert isinstance(tool, Fit2DTool)
    _set_signed_fit_trust(tool)
    signed_trust = tool._document_trust
    admitted_entries = tool._fit_code_entries
    tool._begin_fit_2d_sequence_history()
    original_tool_status = Fit2DTool.tool_status

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("redraw rebuilt or reauthorized the code inventory")

    monkeypatch.setattr(
        Fit2DTool,
        "_code_trust_entries_from_status",
        classmethod(unexpected_call),
    )
    monkeypatch.setattr(
        Fit2DTool,
        "tool_status",
        property(unexpected_call, original_tool_status.fset),
    )
    monkeypatch.setattr(
        Fit2DTool,
        "_serialize_params",
        staticmethod(unexpected_call),
    )
    monkeypatch.setattr(Fit2DTool, "_build_fit_code_entries", unexpected_call)
    monkeypatch.setattr(tool, "_issue_code_execution_capability", unexpected_call)

    result_ds = (
        data.isel(y=0).xlm.modelfit("x", model=model, params=params, max_nfev=1).load()
    )
    for index in range(data.sizes["y"]):
        tool._store_fit_2d_sequence_result(index, result_ds, 0.0)
    for index in range(0, data.sizes["y"], 11):
        tool._set_current_index(index)
        tool._update_fit_curve()

    assert tool._fit_code_entries is admitted_entries
    assert tool._document_trust == signed_trust
    assert tool._current_fit_execution_allowed()


def test_fit2d_rebuild_keeps_new_result_blob_not_previous_blob(qtbot) -> None:
    tool, model, params = _make_linear_fit2d_tool(qtbot)
    data = tool.tool_data
    previous_blob = np.array([17, 29, 41], dtype=np.uint8)
    tool._serialized_fit_result_blob = previous_blob
    replacement = data.xlm.modelfit("x", model=model, params=params, max_nfev=1).load()

    tool._restore_from_fit_dataset(replacement)

    assert tool._serialized_fit_result_blob is not None
    assert not np.array_equal(tool._serialized_fit_result_blob, previous_blob)
    saved = tool.to_dataset()
    assert np.array_equal(
        saved[tool._PERSISTED_FIT_RESULT_VAR].values,
        tool._serialized_fit_result_blob,
    )


def test_fit2d_transpose_discards_owned_result_blob(qtbot) -> None:
    tool, model, params = _make_linear_fit2d_tool(qtbot)
    _seed_fit2d_full_results(tool, model, params)
    tool._cache_fit_result_payload()
    assert tool._serialized_fit_result_blob is not None

    tool._do_transpose()

    assert tool._serialized_fit_result_blob is None
    assert all(result is None for result in tool._result_ds_full)
    assert tool._PERSISTED_FIT_RESULT_VAR not in tool.to_dataset()


def test_fit2d_reset_discards_pending_result_restore(qtbot, monkeypatch) -> None:
    tool, model, params = _make_linear_fit2d_tool(qtbot, expression=True)
    _seed_fit2d_full_results(tool, model, params)
    tool._cache_fit_result_payload()
    blob = np.array(tool._serialized_fit_result_blob, copy=True)
    deserialize_calls: list[int] = []
    original_deserialize = erlab.interactive.utils._deserialize_fit_dataset_blob

    def tracked_deserialize(blob):
        deserialize_calls.append(np.asarray(blob).size)
        return original_deserialize(blob)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "_deserialize_fit_dataset_blob",
        tracked_deserialize,
    )

    tool._last_result_ds = None
    tool._result_ds_full = [None] * len(tool._result_ds_full)
    tool.set_document_trust(untrusted_document_trust(), notify=False)
    tool._restore_persisted_fit_result_blob(blob, fit_is_current=True)
    assert np.array_equal(tool._serialized_fit_result_blob, blob)
    assert tool._pending_persisted_fit_is_current is True
    assert deserialize_calls == []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    tool._reset_params_all()

    assert tool._serialized_fit_result_blob is None
    assert tool._pending_persisted_fit_is_current is None

    tool.set_document_trust(new_document_trust())

    assert deserialize_calls == []
    assert tool._last_result_ds is None
    assert all(result is None for result in tool._result_ds_full)
    assert tool._PERSISTED_FIT_RESULT_VAR not in tool.to_dataset()


def test_fit2d_source_replacement_discards_pending_result_retry(qtbot) -> None:
    tool, _safe_model, _safe_params = _make_linear_fit2d_tool(qtbot)
    data = tool.tool_data
    unsafe_model = lmfit.models.ExpressionModel("slope * x + intercept")
    unsafe_params = unsafe_model.make_params(slope=1.0, intercept=0.0)
    unsafe_result = (
        data.isel(y=0)
        .xlm.modelfit("x", model=unsafe_model, params=unsafe_params, max_nfev=1)
        .load()
    )
    blob = erlab.interactive.utils._serialize_fit_dataset_blob(unsafe_result)
    tool.set_document_trust(untrusted_document_trust(), notify=False)

    tool._restore_persisted_fit_result_blob(blob, fit_is_current=True)
    assert np.array_equal(tool._serialized_fit_result_blob, blob)
    assert tool._pending_persisted_fit_is_current is True
    assert all(result is None for result in tool._result_ds_full)

    assert tool.update_data(data + 1.0)
    tool.set_document_trust(new_document_trust())

    assert tool._serialized_fit_result_blob is None
    assert tool._pending_persisted_fit_is_current is None
    assert all(result is None for result in tool._result_ds_full)
    assert tool._PERSISTED_FIT_RESULT_VAR not in tool.to_dataset()
