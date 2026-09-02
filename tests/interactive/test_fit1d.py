import base64
import contextlib
import gc
import json
import pickle
import threading
import weakref

import lmfit
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive import _fit1d as fit1d
from erlab.interactive import _fit_code_trust as fit_code_trust
from erlab.interactive._code_trust import (
    authorize_document_execution,
    create_entry,
    document_trust_has_trusted_lineage,
    execution_capability_allows,
    external_document_trust,
    issue_local_edit_capability,
    new_document_trust,
    untrusted_document_trust,
)
from erlab.interactive._code_trust._api import _document_trust_after_save
from erlab.interactive._fit1d import (
    Fit1DTool,
    _ParameterEditDelegate,
    _ParameterTableModel,
)
from erlab.interactive._fit_code_trust import (
    lmfit_expression_model_code_entries,
    lmfit_model_code_entry,
    lmfit_model_safe_parameter_expressions,
    lmfit_parameter_expression_entries,
    lmfit_result_code_entry,
)
from erlab.interactive.imagetool._provenance._model import ScriptInput
from tests._qt_helpers import signal_receiver_count


def _make_1d_data() -> xr.DataArray:
    x = np.linspace(-1.0, 1.0, 11)
    data = np.exp(-(x**2))
    return xr.DataArray(data, dims=("x",), coords={"x": x}, name="spec")


def _make_linear_fit1d_tool(
    qtbot, *, expression: bool = False
) -> tuple[Fit1DTool, xr.DataArray, lmfit.Model, lmfit.Parameters]:
    data = _make_1d_data()
    model = lmfit.models.LinearModel()
    params = model.make_params(slope=1.0, intercept=2.0 if expression else 0.0)
    if expression:
        params["intercept"].expr = "2 * slope"
    tool = erlab.interactive.ftool(data, model=model, params=params, execute=False)
    qtbot.addWidget(tool)
    if not isinstance(tool, Fit1DTool):  # pragma: no cover
        raise TypeError("Expected Fit1DTool")
    return tool, data, model, params


def _set_signed_fit_trust(tool: Fit1DTool) -> None:
    manifest = tool._current_code_trust_manifest()
    assert manifest is not None
    signed = _document_trust_after_save(
        new_document_trust(),
        manifest,
        saved_trusted_lineage=True,
        signature_stored=True,
    )
    tool.set_document_trust(signed, notify=False)


def _trust_allows_local_code_edit(trust) -> bool:
    entry = create_entry("test.local-edit", "test", "result = source")
    return authorize_document_execution(trust, (entry,))[1]


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


def _fit_result_dataset(params: lmfit.Parameters, *, nfev: int = 1) -> xr.Dataset:
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


@pytest.mark.parametrize(
    "load",
    [
        pytest.param(lambda: object(), id="no-model"),
        pytest.param(
            lambda: xr.Dataset(
                {
                    "modelfit_results": xr.DataArray(
                        np.empty((0,), dtype=object), dims=("empty",)
                    )
                }
            ),
            id="empty-result",
        ),
    ],
)
def test_lmfit_ftool_restore_rejects_missing_restored_model(load) -> None:
    with pytest.raises(TypeError, match="Restored lmfit model is missing"):
        fit1d._load_lmfit_for_ftool_restore(load)


def test_lmfit_ftool_restore_rejects_non_callable_model_func() -> None:
    class BadModel:
        func = None

    with pytest.raises(TypeError, match="non-callable function"):
        fit1d._load_lmfit_for_ftool_restore(BadModel)


def test_lmfit_ftool_restore_rejects_non_iterable_params() -> None:
    model = lmfit.models.LinearModel()

    with pytest.raises(TypeError, match="parameters are not iterable"):
        fit1d._load_lmfit_for_ftool_restore(lambda: model, params=object())


def test_lmfit_ftool_restore_rejects_incompatible_params() -> None:
    model = lmfit.models.LinearModel()

    with pytest.raises(ValueError, match="missing saved parameters"):
        fit1d._load_lmfit_for_ftool_restore(lambda: model, params=[])


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
    assert isinstance(win._model, erlab.analysis.fit.models.MultiPeakModel)
    assert win._model.func._peak_shapes == ["voigt"]
    assert not win._model.func.convolve
    assert win.peak_shape_combo.currentText() == "voigt"
    assert not win.convolve_check.isChecked()
    assert not win.oversample_spin.isEnabled()

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
    restored_status = win_restored.tool_status.model_dump()
    expected_status = status.model_dump()
    restored_status.pop("model_state")
    expected_status.pop("model_state")
    assert restored_status == expected_status
    assert isinstance(win_restored._model, erlab.analysis.fit.models.MultiPeakModel)
    assert win_restored._model.func._peak_shapes == ["voigt"]
    assert not win_restored._model.func.convolve

    code = win.copy_code()
    assert "modelfit" in code
    assert "MultiPeakModel" in code
    assert "fit_data" not in code


def test_fit1d_copy_code_executes_with_notebook_aliases(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    code = win.copy_code()
    assert "import erlab.analysis as era" not in code
    namespace = {"era": erlab.analysis, "data": data}
    exec(code, namespace, namespace)  # noqa: S102

    assert isinstance(namespace["result"], xr.Dataset)


def test_fit1d_uncertainty_defaults_and_error_bars(qtbot) -> None:
    data = _make_1d_data()
    uncertainty = xr.full_like(data, 0.2).rename("sigma")

    unweighted = erlab.interactive.ftool(data, execute=False)
    weighted = erlab.interactive.ftool(
        data,
        uncertainty=uncertainty,
        uncertainty_name="sigma",
        execute=False,
    )
    overridden = erlab.interactive.ftool(
        data,
        uncertainty=uncertainty,
        scale_covar=True,
        execute=False,
    )
    for win in (unweighted, weighted, overridden):
        qtbot.addWidget(win)

    assert unweighted.scale_covar_check.objectName() == "ftoolScaleCovarCheck"
    assert unweighted.scale_covar_check.isChecked()
    assert unweighted.current_provenance_spec() is not None
    assert not unweighted.normalize_residuals_check.isEnabled()
    assert not weighted.scale_covar_check.isChecked()
    assert weighted.normalize_residuals_check.objectName() == (
        "ftoolNormalizeResidualsCheck"
    )
    assert weighted.normalize_residuals_check.isChecked()
    assert overridden.scale_covar_check.isChecked()
    assert weighted.current_provenance_spec() is None
    np.testing.assert_allclose(weighted.data_errorbar.opts["top"], uncertainty)
    np.testing.assert_allclose(weighted.data_errorbar.opts["bottom"], uncertainty)

    xvals = weighted._x_values()
    residuals = weighted._normalized_data_values() - weighted._model_eval_values(xvals)
    np.testing.assert_allclose(weighted._last_residual, residuals / uncertainty)
    weighted.normalize_residuals_check.setChecked(False)
    np.testing.assert_allclose(weighted._last_residual, residuals)

    weighted.scale_covar_check.setChecked(True)
    assert weighted.tool_status.scale_covar

    weighted._set_fit_running(True, multi=False)
    assert not weighted.scale_covar_check.isEnabled()
    weighted._set_fit_running(False, multi=False)
    assert weighted.scale_covar_check.isEnabled()


def test_fit1d_uncertainty_validation(qtbot) -> None:
    data = _make_1d_data()
    misaligned = xr.DataArray(
        np.ones(data.size),
        dims=("x",),
        coords={"x": np.arange(data.size)},
    )
    extra_dim = xr.DataArray(np.ones(data.size), dims=("other",))

    with pytest.raises(TypeError, match=r"xarray\.DataArray"):
        erlab.interactive.ftool(  # type: ignore[arg-type]
            data, uncertainty=np.ones(data.size), execute=False
        )
    with pytest.raises(ValueError, match="align exactly"):
        erlab.interactive.ftool(data, uncertainty=misaligned, execute=False)
    with pytest.raises(ValueError, match="subset"):
        erlab.interactive.ftool(data, uncertainty=extra_dim, execute=False)
    with pytest.raises(TypeError, match="real numeric"):
        erlab.interactive.ftool(
            data,
            uncertainty=xr.full_like(data, "invalid", dtype=object),
            execute=False,
        )
    with pytest.raises(TypeError, match="real numeric"):
        erlab.interactive.ftool(
            data,
            uncertainty=xr.DataArray(
                np.ones(data.size, dtype=complex),
                dims=data.dims,
                coords=data.coords,
            ),
            execute=False,
        )

    for invalid_value in (0.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and strictly positive"):
            erlab.interactive.ftool(
                data,
                uncertainty=xr.full_like(data, invalid_value),
                execute=False,
            )

    data_with_nan = data.copy()
    data_with_nan[0] = np.nan
    uncertainty_with_nan = xr.full_like(data, 0.2)
    uncertainty_with_nan[0] = np.nan
    win = erlab.interactive.ftool(
        data_with_nan,
        uncertainty=uncertainty_with_nan,
        execute=False,
    )
    qtbot.addWidget(win)


def test_fit1d_direct_weights_validation() -> None:
    data = _make_1d_data()

    assert fit1d._validate_weights_input(data, None) is None
    scalar = xr.DataArray(2.0)
    assert fit1d._validate_weights_input(data, scalar) is scalar
    zero = xr.DataArray(0.0)
    assert fit1d._validate_weights_input(data, zero) is zero
    xr.testing.assert_identical(
        fit1d._broadcast_weights(data, scalar), xr.full_like(data, 2.0).rename(None)
    )

    with pytest.raises(TypeError, match=r"xarray\.DataArray"):
        fit1d._validate_weights_input(data, np.ones(data.size))
    with pytest.raises(TypeError, match="real numeric"):
        fit1d._validate_weights_input(data, xr.full_like(data, "invalid", dtype=object))
    with pytest.raises(TypeError, match="real numeric"):
        fit1d._validate_weights_input(data, xr.full_like(data, 1.0, dtype=complex))
    with pytest.raises(ValueError, match="unexpected dimensions"):
        fit1d._validate_weights_input(
            data, xr.DataArray(np.ones((data.size, 2)), dims=("x", "extra"))
        )
    with pytest.raises(ValueError, match="align exactly"):
        fit1d._validate_weights_input(
            data,
            xr.DataArray(
                np.ones(data.size),
                dims="x",
                coords={"x": data.x + 1.0},
            ),
        )
    for invalid_value in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            fit1d._validate_weights_input(data, xr.full_like(data, invalid_value))


def test_fit1d_rejects_ambiguous_internal_weighting(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    weighting = xr.ones_like(data)
    with pytest.raises(ValueError, match="Only one"):
        win._reset_fit_state(
            data,
            win._model,
            win._params,
            uncertainty=weighting,
            direct_weights=weighting,
            data_name="data",
            model_name="model",
        )

    tiny_weights = xr.full_like(data, np.nextafter(0.0, 1.0))
    win._set_direct_weights(tiny_weights)
    xr.testing.assert_identical(win._fit_weights(), tiny_weights)
    assert np.isnan(win.data_errorbar.opts["top"]).all()


def test_fit1d_scalar_uncertainty_broadcasts(qtbot, exp_decay_model) -> None:
    t = np.linspace(0.0, 4.0, 25)
    data = xr.DataArray(
        3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
    )
    uncertainty = xr.DataArray(0.2, name="sigma")
    win = erlab.interactive.ftool(
        data,
        model=exp_decay_model,
        params=exp_decay_model.make_params(n0=2.0, tau=1.0),
        uncertainty=uncertainty,
        data_name="decay",
        model_name="model",
        uncertainty_name="xr.DataArray(0.2)",
        execute=False,
    )
    qtbot.addWidget(win)

    np.testing.assert_allclose(win.data_errorbar.opts["top"], 0.2)
    weights = win._fit_weights()
    assert weights is not None
    xr.testing.assert_allclose(weights, xr.full_like(data, 5.0))

    prelude = win._copy_prelude()
    assert "input_uncertainty" not in prelude
    namespace = {
        "decay": data,
        "model": exp_decay_model,
        "xr": xr,
    }
    exec(  # noqa: S102
        f"{prelude}\nreplayed = {win._fit_expression()}",
        namespace,
    )
    replayed_result = namespace["replayed"].modelfit_results.compute().item()
    np.testing.assert_allclose(replayed_result.weights, 5.0)


def test_fit1d_generated_uncertainty_name_does_not_shadow_data(
    qtbot, exp_decay_model
) -> None:
    t = np.linspace(0.0, 4.0, 25)
    fit_uncertainty = xr.DataArray(
        3.0 * np.exp(-t / 2.0),
        dims=("t",),
        coords={"t": t},
        name="fit_uncertainty",
    )
    sigma = xr.DataArray(0.2)
    win = erlab.interactive.ftool(
        fit_uncertainty,
        model=exp_decay_model,
        params=exp_decay_model.make_params(n0=2.0, tau=1.0),
        uncertainty=sigma,
        data_name="fit_uncertainty",
        model_name="model",
        uncertainty_name="sigma.copy()",
        execute=False,
    )
    qtbot.addWidget(win)

    namespace = {
        "fit_uncertainty": fit_uncertainty,
        "model": exp_decay_model,
        "sigma": sigma,
    }
    exec(  # noqa: S102
        f"{win._copy_prelude()}\nreplayed = {win._fit_expression()}",
        namespace,
    )
    replayed = namespace["replayed"]
    xr.testing.assert_identical(
        replayed.modelfit_data,
        fit_uncertainty.rename("modelfit_data"),
    )
    np.testing.assert_allclose(
        replayed.modelfit_results.compute().item().weights,
        5.0,
    )


def test_fit1d_weighted_fit_and_generated_code(qtbot, exp_decay_model) -> None:
    t = np.linspace(0.0, 4.0, 25)
    data = xr.DataArray(
        3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
    )
    uncertainty = xr.DataArray(
        np.linspace(0.1, 0.2, t.size),
        dims=("t",),
        coords={"t": t},
        name="sigma",
    )
    params = exp_decay_model.make_params(n0=2.0, tau=1.0)
    win = erlab.interactive.ftool(
        data,
        model=exp_decay_model,
        params=params,
        uncertainty=uncertainty,
        data_name="decay",
        model_name="model",
        uncertainty_name="sigma",
        execute=False,
    )
    qtbot.addWidget(win)
    win.normalize_check.setChecked(True)
    win.domain_min_spin.setValue(0.5)
    win.domain_max_spin.setValue(3.5)
    win.nfev_spin.setValue(0)

    fit_uncertainty = win._fit_uncertainty()
    assert fit_uncertainty is not None
    expected_uncertainty = uncertainty.sel(t=slice(0.5, 3.5)) / abs(
        data.sel(t=slice(0.5, 3.5)).mean()
    )
    xr.testing.assert_allclose(fit_uncertainty, expected_uncertainty)

    assert win._run_fit()
    qtbot.waitUntil(lambda: win._last_result_ds is not None, timeout=10000)
    assert win._last_result_ds is not None
    result = win._last_result_ds.modelfit_results.compute().item()
    np.testing.assert_allclose(result.weights, 1.0 / expected_uncertainty.values)
    assert result.scale_covar is False

    win.scale_covar_check.setChecked(True)
    assert win._run_fit()
    qtbot.waitUntil(lambda: not win._fit_running(), timeout=10000)
    assert win._last_result_ds is not None
    assert win._last_result_ds.modelfit_results.compute().item().scale_covar is True
    win.scale_covar_check.setChecked(False)

    namespace = {
        "decay": data,
        "model": exp_decay_model,
        "sigma": uncertainty,
    }
    exec(  # noqa: S102
        f"{win._copy_prelude()}\nreplayed = {win._fit_expression()}",
        namespace,
    )
    replayed_result = namespace["replayed"].modelfit_results.compute().item()
    np.testing.assert_allclose(
        replayed_result.weights, 1.0 / expected_uncertainty.values
    )
    assert replayed_result.scale_covar is False

    win._run_fit_multiple(1)
    qtbot.waitUntil(lambda: not win._fit_running(), timeout=10000)
    assert win._last_result_ds is not None
    multi_result = win._last_result_ds.modelfit_results.compute().item()
    np.testing.assert_allclose(
        multi_result.weights,
        1.0 / expected_uncertainty.values,
    )


def test_fit1d_uncertainty_persistence_roundtrip(qtbot) -> None:
    data = _make_1d_data()
    uncertainty = xr.full_like(data, 0.2).rename("sigma")
    win = erlab.interactive.ftool(
        data,
        uncertainty=uncertainty,
        uncertainty_name="sigma",
        execute=False,
    )
    qtbot.addWidget(win)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(),
        _code_trust=new_document_trust(),
    )
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit1DTool)
    xr.testing.assert_identical(restored.uncertainty, uncertainty)
    assert restored.tool_status.uncertainty_name == "sigma"
    assert restored.scale_covar_check.isChecked() is False


def test_fit1d_managed_uncertainty_uses_named_persistence_input(qtbot) -> None:
    data = _make_1d_data()
    uncertainty = xr.full_like(data, 0.2).rename("sigma")
    win = erlab.interactive.ftool(data, uncertainty=uncertainty, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)
    bindings = (ScriptInput(name="data"), ScriptInput(name="uncertainty"))
    win.set_script_inputs(bindings, primary_input="data", state="stale")

    items = win._persistence_data_items()
    assert "uncertainty" in items
    assert fit1d._PERSISTED_UNCERTAINTY_VAR not in items
    xr.testing.assert_identical(items["uncertainty"], uncertainty)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(win.to_dataset())
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit1DTool)
    assert restored.script_inputs == bindings
    xr.testing.assert_identical(restored.uncertainty, uncertainty)

    replacement_data = data + 1.0
    replacement_uncertainty = uncertainty + 0.1
    win._replace_persistence_data_items(
        {
            erlab.interactive.utils._SAVED_TOOL_DATA_NAME: replacement_data,
            "uncertainty": replacement_uncertainty,
        },
        xr.Dataset(),
    )
    xr.testing.assert_identical(win.tool_data, replacement_data)
    xr.testing.assert_identical(win.uncertainty, replacement_uncertainty)
    assert win.script_inputs == bindings
    assert win.source_state == "stale"


def test_fit1d_legacy_uncertainty_and_direct_weights_persistence(qtbot) -> None:
    data = _make_1d_data()
    uncertainty = xr.full_like(data, 0.2).rename("sigma")
    legacy = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(legacy)
    assert isinstance(legacy, Fit1DTool)
    legacy._restore_persistence_data_items(
        {fit1d._PERSISTED_UNCERTAINTY_VAR: uncertainty}, xr.Dataset()
    )
    xr.testing.assert_identical(legacy.uncertainty, uncertainty)

    weights = xr.full_like(data, 2.0).rename("weights")
    weighted = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(weighted)
    assert isinstance(weighted, Fit1DTool)
    weighted._set_direct_weights(weights)
    items = weighted._persistence_data_items()
    assert fit1d._PERSISTED_WEIGHTS_VAR in items
    assert "uncertainty" not in items
    xr.testing.assert_identical(items[fit1d._PERSISTED_WEIGHTS_VAR], weights)

    restored = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(restored)
    assert isinstance(restored, Fit1DTool)
    restored._restore_persistence_data_items(items, xr.Dataset())
    xr.testing.assert_identical(restored._direct_weights, weights)


def test_fit1d_tool_status_without_saved_params_uses_model_defaults(
    qtbot, exp_decay_model
) -> None:
    t = np.linspace(0.0, 2.0, 11)
    data = xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="decay")
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)

    status = win.tool_status.model_copy(update={"params": []})
    win_restored = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win_restored)
    win_restored.tool_status = status

    assert list(win_restored._params) == ["n0", "tau"]
    assert win_restored.tool_status.params


def test_fit1d_saved_model_restore_waits_for_document_approval(
    qtbot, monkeypatch
) -> None:
    data = _make_1d_data()
    model = lmfit.Model(lambda x: x, independent_vars=["x"])
    source = erlab.interactive.ftool(data, model=model, execute=False)
    qtbot.addWidget(source)
    saved = source.to_dataset()
    saved_status = Fit1DTool.StateModel.model_validate_json(saved.attrs["tool_state"])

    calls: list[str] = []
    original_loads = lmfit.model.Model.loads

    def tracked_loads(model, state, **kwargs):
        calls.append(state)
        return original_loads(model, state, **kwargs)

    monkeypatch.setattr(lmfit.model.Model, "loads", tracked_loads)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)

    assert isinstance(restored, Fit1DTool)
    assert calls == []
    assert not document_trust_has_trusted_lineage(restored._document_trust)
    assert restored._saved_tool_status() == saved_status

    restored.set_document_trust(new_document_trust())

    assert calls == [saved_status.model_state[1]]
    assert restored.tool_status.model_name == saved_status.model_name


def test_fit1d_saved_builtin_model_restores_without_approval(
    qtbot, monkeypatch
) -> None:
    source = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(source)
    saved = source.to_dataset()
    saved_status = Fit1DTool.StateModel.model_validate_json(saved.attrs["tool_state"])
    calls: list[str] = []
    original_loads = lmfit.model.Model.loads

    def tracked_loads(model, state, **kwargs):
        calls.append(state)
        return original_loads(model, state, **kwargs)

    monkeypatch.setattr(lmfit.model.Model, "loads", tracked_loads)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)

    assert isinstance(restored, Fit1DTool)
    assert calls == [saved_status.model_state[1]]
    assert restored._pending_fit_status is None
    assert restored._fit_code_entries == ()


@pytest.mark.parametrize("ndim", [1, 2])
def test_signed_parameterless_model_restore_preserves_saved_identity(
    qtbot, monkeypatch, ndim: int
) -> None:
    data = _make_1d_data()
    if ndim == 2:
        data = xr.concat([data, data + 1.0], dim="y")
    source = erlab.interactive.ftool(
        data,
        model=lmfit.Model(lambda x: x, independent_vars=["x"]),
        execute=False,
    )
    qtbot.addWidget(source)
    assert not source._params
    _set_signed_fit_trust(source)
    signed_trust = source._document_trust
    saved_model_state = source._serialized_model_state
    saved_entries = source._fit_code_entries
    saved = source.to_dataset()
    dump_calls = 0
    original_loads = lmfit.model.Model.loads

    def tracked_loads(model, state, **kwargs):
        loaded = original_loads(model, state, **kwargs)

        def unexpected_dump(*_args, **_kwargs):
            nonlocal dump_calls
            dump_calls += 1
            return lmfit.models.ExpressionModel("changed * x").dumps()

        monkeypatch.setattr(loaded, "dumps", unexpected_dump)
        return loaded

    monkeypatch.setattr(lmfit.model.Model, "loads", tracked_loads)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved,
        _code_trust=signed_trust,
    )
    qtbot.addWidget(restored)

    assert isinstance(restored, Fit1DTool)
    assert dump_calls == 0
    assert restored._serialized_model_state == saved_model_state
    assert restored._fit_code_entries == saved_entries
    assert restored._document_trust == signed_trust


def test_fit1d_saved_model_class_does_not_import_document_target(
    qtbot, monkeypatch, tmp_path
) -> None:
    module_name = "unregistered_saved_fit_model"
    marker = tmp_path / "imported.txt"
    (tmp_path / f"{module_name}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "class SavedModel:\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    source = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(source)
    saved = source.to_dataset()
    status = Fit1DTool.StateModel.model_validate_json(saved.attrs["tool_state"])
    saved.attrs["tool_state"] = status.model_copy(
        update={
            "model_state": (
                f"{module_name}:SavedModel",
                status.model_state[1],
            )
        }
    ).model_dump_json()

    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        saved,
        _code_trust=new_document_trust(),
    )
    qtbot.addWidget(restored)

    assert isinstance(restored, Fit1DTool)
    assert not marker.exists()
    assert type(restored._model).__module__ != module_name


@pytest.mark.parametrize(
    ("model", "expected_code"),
    [
        pytest.param(lmfit.models.ExponentialModel(), None, id="lmfit-library"),
        pytest.param(erlab.analysis.fit.models.TLLModel(), None, id="erlab-library"),
        pytest.param(
            erlab.analysis.fit.models.MultiPeakModel(),
            None,
            id="erlab-embedded-multipeak",
        ),
        pytest.param(
            erlab.analysis.fit.models.MultiPeakModel(
                2,
                "voigt gaussian",
                oversample=64,
                segmented=True,
            ),
            None,
            id="erlab-embedded-multipeak-options",
        ),
        pytest.param(
            erlab.analysis.fit.models.MultiPeakModel(
                fd=False,
                background="shirley",
                convolve=False,
            ),
            None,
            id="erlab-embedded-multipeak-shirley",
        ),
        pytest.param(
            erlab.analysis.fit.models.MultiPeakModel(
                fd=False,
                background="constant",
                convolve=False,
            ),
            None,
            id="erlab-embedded-multipeak-constant",
        ),
        pytest.param(
            erlab.analysis.fit.models.MultiPeakModel(
                fd=False,
                background="polynomial",
                degree=3,
                convolve=False,
            ),
            None,
            id="erlab-embedded-multipeak-polynomial",
        ),
        pytest.param(
            erlab.analysis.fit.models.PolynomialModel(),
            None,
            id="erlab-embedded-polynomial",
        ),
        pytest.param(
            erlab.analysis.fit.models.FermiEdge2dModel(),
            None,
            id="erlab-embedded-fermi-edge-2d",
        ),
        pytest.param(
            erlab.analysis.fit.models.StepEdgeModel(),
            None,
            id="erlab-embedded-step-edge",
        ),
        pytest.param(
            lmfit.models.ExpressionModel("amplitude * exp(-x / decay)"),
            "amplitude * exp(-x / decay)",
            id="expression",
        ),
        pytest.param(
            lmfit.Model(lambda x, amplitude: amplitude * x),
            "serialized-callable",
            id="custom-callable",
        ),
        pytest.param(
            erlab.analysis.fit.models.FermiEdgeModel(),
            None,
            id="embedded-library-callable",
        ),
    ],
)
def test_lmfit_code_trust_classifies_models_without_loading(
    model: lmfit.Model, expected_code: str | None
) -> None:
    entry = lmfit_model_code_entry(
        model.dumps(),
        feature="test.lmfit-model",
        location="model",
        model_reference=fit1d._model_class_reference(type(model)),
    )

    if expected_code is None:
        assert entry is None
    else:
        assert entry is not None
        assert expected_code in entry.code


def test_lmfit_code_trust_rejects_modified_embedded_library_model() -> None:
    model = erlab.analysis.fit.models.MultiPeakModel(
        2,
        "voigt voigt",
        fd=False,
        background="none",
        convolve=False,
    )
    reference = fit1d._model_class_reference(type(model))
    serialized = json.loads(model.dumps())
    serialized["value"][0]["funcdef"]["value"] += "modified"

    entry = lmfit_model_code_entry(
        json.dumps(serialized),
        feature="test.lmfit-model",
        location="model",
        model_reference=reference,
    )
    wrong_reference_entry = lmfit_model_code_entry(
        model.dumps(),
        feature="test.lmfit-model",
        location="model",
        model_reference="lmfit.model:Model",
    )

    assert entry is not None
    assert wrong_reference_entry is not None
    assert "serialized-callable" in entry.code
    assert "serialized-callable" in wrong_reference_entry.code


def test_lmfit_code_trust_serialized_helper_guards() -> None:
    class EmptySerializedModel:
        def dumps(self) -> str:
            return "{}"

    assert fit_code_trust._wrapped_list({"__class__": "List", "value": ()}) is None
    assert fit_code_trust._serialized_model_dict(object()) is None
    assert fit_code_trust._serialized_model_dict(EmptySerializedModel()) is None
    assert fit_code_trust._common_model_kwargs({}) is None


@pytest.mark.parametrize(
    "state",
    [
        {"oversample": None},
        {"other": 3},
    ],
)
def test_lmfit_code_trust_rejects_non_scalar_pickle_state(
    state: dict[str, object],
) -> None:
    payload = base64.b64encode(pickle.dumps(state)).decode()

    assert fit_code_trust._pickle_scalar_after_key(payload, "oversample") is None


@pytest.mark.parametrize(
    "case",
    [
        "non-string-callable",
        "non-string-name",
        "no-peaks",
        "non-contiguous-peaks",
        "unknown-shape",
        "partial-fermi-dirac",
        "non-contiguous-polynomial",
        "high-degree-polynomial",
    ],
)
def test_lmfit_code_trust_rejects_invalid_multipeak_options(case: str) -> None:
    model = erlab.analysis.fit.models.MultiPeakModel(
        1,
        "voigt",
        fd=False,
        background="none",
        convolve=False,
    )
    item = json.loads(model.dumps())["value"][0]
    names = item["param_root_names"]["value"]

    if case == "non-string-callable":
        item["funcdef"]["value"] = 1
    elif case == "non-string-name":
        names.append(1)
    elif case == "no-peaks":
        names.clear()
    elif case == "non-contiguous-peaks":
        names[:] = [name.replace("p0_", "p1_") for name in names]
    elif case == "unknown-shape":
        names[:] = ["p0_center"]
        item["param_hints"].clear()
    elif case == "partial-fermi-dirac":
        names.append("temp")
    elif case == "non-contiguous-polynomial":
        names.append("c1")
    elif case == "high-degree-polynomial":
        names.extend(f"c{index}" for index in range(12))

    assert fit_code_trust._multipeak_candidate_options(item) is None


def test_lmfit_code_trust_known_model_candidate_guards() -> None:
    common = {"name": "model", "prefix": "", "nan_policy": "raise"}
    wrapped = {"__class__": "List", "value": ["c0", "c1"]}

    assert not list(
        fit_code_trust._known_model_candidates({"funcname": "PolynomialFunction"})
    )
    assert not list(
        fit_code_trust._known_model_candidates(
            {**common, "funcname": "PolynomialFunction"}
        )
    )
    assert not list(
        fit_code_trust._known_model_candidates(
            {
                **common,
                "funcname": "PolynomialFunction",
                "param_root_names": {
                    "__class__": "List",
                    "value": ["c0", "c2"],
                },
            }
        )
    )
    assert not list(
        fit_code_trust._known_model_candidates(
            {
                **common,
                "funcname": "PolynomialFunction",
                "param_root_names": {
                    "__class__": "List",
                    "value": [f"c{index}" for index in range(22)],
                },
            }
        )
    )
    assert not list(
        fit_code_trust._known_model_candidates(
            {**common, "funcname": "FermiEdge2dFunction"}
        )
    )
    assert not list(
        fit_code_trust._known_model_candidates(
            {
                **common,
                "funcname": "FermiEdge2dFunction",
                "param_root_names": {"__class__": "List", "value": []},
            }
        )
    )
    assert not list(
        fit_code_trust._known_model_candidates(
            {
                **common,
                "funcname": "FermiEdge2dFunction",
                "param_root_names": {
                    "__class__": "List",
                    "value": [f"c{index}" for index in range(22)],
                },
            }
        )
    )
    candidates = list(
        fit_code_trust._known_model_candidates(
            {
                **common,
                "funcname": "FermiEdge2dFunction",
                "param_root_names": wrapped,
            }
        )
    )

    assert len(candidates) == 1
    assert isinstance(candidates[0], erlab.analysis.fit.models.FermiEdge2dModel)


@pytest.mark.parametrize("serialized", ["{", "[]"])
def test_lmfit_code_trust_known_model_match_rejects_invalid_json(
    serialized: str,
) -> None:
    assert fit_code_trust._compute_known_model_match(serialized, None) is None


def test_lmfit_code_trust_known_model_match_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_candidate_error(_item: dict[str, object]):
        raise RuntimeError("candidate construction failed")

    monkeypatch.setattr(
        fit_code_trust, "_known_model_candidates", raise_candidate_error
    )

    assert fit_code_trust._compute_known_model_match("{}", None) is None


def test_lmfit_code_trust_payload_match_uses_serialized_memo_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = {"funcdef": {}}
    serialized_item = json.dumps(
        item,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    match = ("test:Model", ())
    memo = {(serialized_item, None): (item, match)}

    def unexpected_match(*_args):
        pytest.fail("The exact payload memo entry was not used")

    monkeypatch.setattr(fit_code_trust, "_known_model_match", unexpected_match)

    assert fit_code_trust._payload_model_match(item, None, memo) == match


def test_lmfit_code_trust_payload_match_stores_serialized_memo_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = {"funcdef": {}}
    serialized_item = json.dumps(
        item,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    match = ("test:Model", ())
    memo = {}

    monkeypatch.setattr(fit_code_trust, "_known_model_match", lambda *_args: match)

    assert fit_code_trust._payload_model_match(item, None, memo) == match
    assert memo == {(serialized_item, None): (item, match)}


def test_lmfit_code_trust_rejects_non_json_model_items() -> None:
    payload_item = {"funcdef": {"value": "callable"}, "bad": float("nan")}
    serialized_item = {
        "funcname": "CustomFunction",
        "funcdef": {},
        "param_hints": {},
        "bad": float("nan"),
    }

    assert fit_code_trust._payload_model_match(payload_item, None, {}) is None
    assert fit_code_trust._safe_model_matches(
        None,
        serialized_items=(("root", serialized_item, None),),
    ) == ({}, frozenset())


def test_lmfit_code_trust_omits_exact_model_parameter_hints() -> None:
    model = erlab.analysis.fit.models.MultiPeakModel(
        2,
        "voigt voigt",
        fd=False,
        background="none",
        convolve=False,
    )
    reference = fit1d._model_class_reference(type(model))
    safe_expressions = lmfit_model_safe_parameter_expressions(model.dumps(), reference)
    expressions = [
        *((name, param.expr) for name, param in model.make_params().items()),
        ("p1_sigma", "p0_sigma"),
    ]

    entries = lmfit_parameter_expression_entries(
        expressions,
        feature="test.lmfit-parameter",
        location_prefix="parameters",
        safe_expressions=safe_expressions,
    )

    assert [(entry.location, entry.code) for entry in entries] == [
        ("parameters/p1_sigma", "p0_sigma")
    ]


def test_expression_model_local_entry_matches_saved_model_identity() -> None:
    expression = "amplitude * exp(-x / decay)"
    model = lmfit.models.ExpressionModel(expression)
    local_entries = lmfit_expression_model_code_entries(
        expression,
        "constant = 1",
        feature="test.lmfit-model",
        location="model",
    )
    saved_entry = lmfit_model_code_entry(
        model.dumps(),
        feature="test.lmfit-model",
        location="model",
    )

    assert saved_entry is not None
    assert local_entries[0] == saved_entry
    assert local_entries[1].code == "constant = 1"


def test_lmfit_code_trust_tracks_every_saved_parameter_expression() -> None:
    assert (
        lmfit_parameter_expression_entries(
            None,
            feature="test.lmfit-parameter",
            location_prefix="parameters",
        )
        == ()
    )
    entries = lmfit_parameter_expression_entries(
        [
            ("plain", None),
            ("constrained", "2 * plain"),
        ],
        feature="test.lmfit-parameter",
        location_prefix="parameters",
    )

    assert len(entries) == 1
    assert entries[0].location == "parameters/constrained"
    assert entries[0].code == "2 * plain"
    assert entries[0].context == {"parameter": "constrained"}


@pytest.mark.parametrize(
    ("serialized", "review_text"),
    [
        ("{", "Invalid serialized lmfit content"),
        (json.dumps({"params": "{"}), "Invalid serialized lmfit parameters"),
    ],
)
def test_lmfit_code_trust_fails_closed_for_malformed_json(
    serialized: str, review_text: str
) -> None:
    entry = lmfit_model_code_entry(
        serialized,
        feature="test.lmfit-model",
        location="model",
    )

    assert entry is not None
    assert review_text in entry.code


def test_lmfit_code_trust_hashes_non_json_executable_details() -> None:
    serialized = json.dumps(
        {
            "__class__": "Callable",
            "__name__": "custom_model",
            "importer": "unavailable.user_module",
            "value": float("nan"),
        }
    )

    entry = lmfit_model_code_entry(
        serialized,
        feature="test.lmfit-model",
        location="model",
    )

    assert entry is not None
    assert "serialized-callable" in entry.code
    assert isinstance(entry.context["payload_sha256"], str)


def test_lmfit_parameter_expression_locations_are_stable_and_escaped() -> None:
    original = lmfit_parameter_expression_entries(
        [("first/parameter", "scale * 2"), ("second", "scale * 3")],
        feature="test.lmfit-parameter",
        location_prefix="parameters",
    )
    inserted = lmfit_parameter_expression_entries(
        [
            ("added", "scale * 4"),
            ("first/parameter", "scale * 2"),
            ("second", "scale * 3"),
        ],
        feature="test.lmfit-parameter",
        location_prefix="parameters",
    )

    assert original == inserted[1:]
    assert [entry.location for entry in inserted] == [
        "parameters/added",
        "parameters/first%2Fparameter",
        "parameters/second",
    ]


def _custom_lmfit_json(*, dill_value: str, numeric_value: float) -> str:
    return json.dumps(
        {
            "chisqr": numeric_value,
            "model": {
                "__class__": "Callable",
                "__name__": "custom_model",
                "importer": "unavailable.user_module",
                "pyversion": "3.14",
                "value": dill_value,
            },
        }
    )


@pytest.mark.parametrize("payload_kind", ["model", "result"])
def test_lmfit_entries_digest_only_executable_fragments(payload_kind: str) -> None:
    def entry(*, dill_value: str, numeric_value: float):
        serialized = _custom_lmfit_json(
            dill_value=dill_value, numeric_value=numeric_value
        )
        if payload_kind == "model":
            return lmfit_model_code_entry(
                serialized,
                feature="test.lmfit-model",
                location="model",
            )
        payload = xr.Dataset(
            {
                "modelfit_results": xr.DataArray(serialized),
                "numeric_result": xr.DataArray(numeric_value),
            }
        ).to_netcdf(path=None, engine="h5netcdf")
        return lmfit_result_code_entry(
            bytes(payload),
            feature="test.lmfit-result",
            location="fit-result",
        )

    original = entry(dill_value="dill-a", numeric_value=1.0)

    assert original is not None
    assert original == entry(dill_value="dill-a", numeric_value=2.0)
    assert original != entry(dill_value="dill-b", numeric_value=1.0)


def test_lmfit_code_trust_classifies_result_payload_without_loading_it() -> None:
    x = np.linspace(0.0, 2.0, 11)
    model = lmfit.models.ExponentialModel()
    params = model.make_params(amplitude=1.0, decay=1.0)
    result = model.fit(np.exp(-x), params, x=x)
    payload = xr.Dataset({"modelfit_results": xr.DataArray(result.dumps())}).to_netcdf(
        path=None, engine="h5netcdf"
    )

    entry = lmfit_result_code_entry(
        bytes(payload),
        feature="test.lmfit-result",
        location="fit-result",
    )

    assert entry is None


def test_lmfit_code_trust_accepts_exact_local_multipeak_result() -> None:
    x = np.linspace(-1.0, 1.0, 21)
    model = erlab.analysis.fit.models.MultiPeakModel(
        1,
        "voigt",
        fd=False,
        background="none",
        convolve=False,
    )
    params = model.make_params(
        p0_center=0.0,
        p0_sigma=0.2,
        p0_gamma=0.2,
        p0_amplitude=1.0,
    )
    result = model.fit(model.eval(params, x=x), params, x=x)
    payload = xr.Dataset({"modelfit_results": xr.DataArray(result.dumps())}).to_netcdf(
        path=None, engine="h5netcdf"
    )

    entry = lmfit_result_code_entry(
        bytes(payload),
        feature="test.lmfit-result",
        location="fit-result",
    )

    assert entry is None


@pytest.mark.parametrize("transient_miss", [False, True])
def test_lmfit_result_code_trust_memoizes_confirmed_model_match(
    monkeypatch, transient_miss: bool
) -> None:
    x = np.linspace(-1.0, 1.0, 21)
    model = erlab.analysis.fit.models.MultiPeakModel(
        1,
        "voigt",
        fd=False,
        background="none",
        convolve=False,
    )
    params = model.make_params(
        p0_center=0.0,
        p0_sigma=0.2,
        p0_gamma=0.2,
        p0_amplitude=1.0,
    )
    result = model.fit(model.eval(params, x=x), params, x=x)
    payload = xr.Dataset(
        {"modelfit_results": xr.DataArray([result.dumps()] * 10, dims="row")}
    ).to_netcdf(path=None, engine="h5netcdf")
    expressions = tuple(
        (name, expression)
        for name, param in model.make_params().items()
        if isinstance((expression := param._expr), str) and expression.strip()
    )
    match = (fit1d._model_class_reference(type(model)), expressions)
    calls: list[tuple[str, str | None]] = []

    def model_match(serialized_item: str, model_reference: str | None):
        calls.append((serialized_item, model_reference))
        if transient_miss and len(calls) > 1:
            return match
        return None

    monkeypatch.setattr(fit_code_trust, "_known_model_match", model_match)

    entry = lmfit_result_code_entry(
        bytes(payload),
        feature="test.lmfit-result",
        location="fit-result",
    )

    assert len(calls) == 2
    assert (entry is None) is transient_miss


@pytest.mark.parametrize("tampered_first", [False, True])
def test_lmfit_result_code_trust_memo_requires_exact_model_match(
    tampered_first: bool,
) -> None:
    x = np.linspace(-1.0, 1.0, 21)
    model = erlab.analysis.fit.models.MultiPeakModel(
        1,
        "voigt",
        fd=False,
        background="none",
        convolve=False,
    )
    params = model.make_params(
        p0_center=0.0,
        p0_sigma=0.2,
        p0_gamma=0.2,
        p0_amplitude=1.0,
    )
    result = model.fit(model.eval(params, x=x), params, x=x)
    serialized = result.dumps()
    tampered = json.loads(serialized)
    tampered["model"]["value"][0]["funcdef"]["pyversion"] = "modified"
    serialized_tampered = json.dumps(tampered)
    results = [serialized_tampered, serialized]
    if not tampered_first:
        results.reverse()
    payload = xr.Dataset(
        {"modelfit_results": xr.DataArray(results, dims="row")}
    ).to_netcdf(path=None, engine="h5netcdf")

    entry = lmfit_result_code_entry(
        bytes(payload),
        feature="test.lmfit-result",
        location="fit-result",
    )

    assert entry is not None
    assert "serialized-callable" in entry.code


def test_lmfit_code_trust_fails_closed_for_invalid_result_payloads() -> None:
    missing_results = xr.Dataset({"other": xr.DataArray(1)}).to_netcdf(
        path=None, engine="h5netcdf"
    )
    non_string_result = xr.Dataset({"modelfit_results": xr.DataArray(1)}).to_netcdf(
        path=None, engine="h5netcdf"
    )

    for payload in (b"not-netcdf", bytes(missing_results), bytes(non_string_result)):
        entry = lmfit_result_code_entry(
            payload,
            feature="test.lmfit-result",
            location="fit-result",
        )
        assert entry is not None
        assert "Unrecognized serialized lmfit result" in entry.code


def test_fit1d_saved_library_model_and_result_restore_without_approval(qtbot) -> None:
    x = np.linspace(0.0, 2.0, 25)
    values = 1.2 * np.exp(-x / 0.8) + 0.005 * np.sin(5.0 * x)
    data = xr.DataArray(values, dims=("x",), coords={"x": x})
    model = lmfit.models.ExponentialModel()
    source = erlab.interactive.ftool(
        data,
        model=model,
        params=model.make_params(amplitude=1.0, decay=1.0),
        execute=False,
    )
    qtbot.addWidget(source)
    assert source._run_fit()
    qtbot.waitUntil(lambda: source._last_result_ds is not None, timeout=10000)

    restored = erlab.interactive.utils.ToolWindow.from_dataset(source.to_dataset())
    qtbot.addWidget(restored)

    assert isinstance(restored, Fit1DTool)
    assert restored._pending_fit_status is None
    assert restored._pending_persisted_fit_is_current is None
    assert restored._model.func is lmfit.lineshapes.exponential
    assert restored._last_result_ds is not None


def test_fit1d_code_trust_tracks_parameter_expressions(qtbot) -> None:
    tool = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(tool)
    status = tool.tool_status
    parameter_state = list(status.params[0])
    parameter_state[3] = "other_parameter + 1"
    changed_params = list(status.params)
    changed_params[0] = tuple(parameter_state)
    changed = status.model_copy(update={"params": changed_params})

    baseline_entries = tuple(type(tool)._code_trust_entries_from_status(status))
    changed_entries = tuple(type(tool)._code_trust_entries_from_status(changed))

    assert baseline_entries != changed_entries
    expression_entry = next(
        entry
        for entry in changed_entries
        if entry.feature == tool._PARAMETER_CODE_TRUST_FEATURE
    )
    assert expression_entry.location == f"parameters/{parameter_state[0]}"
    assert expression_entry.code == "other_parameter + 1"
    assert expression_entry.context == {"parameter": parameter_state[0]}


def test_fit1d_empty_model_state_does_not_bypass_expression_trust(
    qtbot, monkeypatch
) -> None:
    source = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(source)
    saved = source.to_dataset()
    status = Fit1DTool.StateModel.model_validate_json(saved.attrs["tool_state"])
    parameter_state = list(status.params[0])
    parameter_state[3] = "1 + 1"
    params = list(status.params)
    params[0] = tuple(parameter_state)
    modified_status = status.model_copy(
        update={
            "model_state": (status.model_state[0], ""),
            "normalize_mean": not status.normalize_mean,
            "params": params,
        }
    )
    saved.attrs["tool_state"] = modified_status.model_dump_json()

    calls: list[object] = []
    original = Fit1DTool._deserialize_params

    def tracked_deserialize(state, **kwargs):
        calls.append(state)
        return original(state, **kwargs)

    monkeypatch.setattr(
        Fit1DTool,
        "_deserialize_params",
        staticmethod(tracked_deserialize),
    )

    restored = erlab.interactive.utils.ToolWindow.from_dataset(saved)
    qtbot.addWidget(restored)

    assert calls == []
    assert not document_trust_has_trusted_lineage(restored._document_trust)
    assert restored.normalize_check.isChecked() is modified_status.normalize_mean
    assert tuple(type(restored)._code_trust_entries_from_status(modified_status))


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
    assert win._flush_pending_history_write()
    assert win.undoable

    win.undo()
    assert win.param_model.param_at(0).value == pytest.approx(start_value)
    assert win.redoable

    win.redo()
    assert win.param_model.param_at(0).value == pytest.approx(updated_value)


def test_fit1d_undo_redo_repairs_equal_bound_params(
    qtbot, exp_decay_model, monkeypatch
) -> None:
    t = np.linspace(0.0, 2.0, 11)
    data = xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="decay")
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)

    good_status = win.tool_status
    bad_params = list(good_status.params or [])
    bad_param_state = list(bad_params[0])
    param_name = str(bad_param_state[0])
    bad_param_state[1] = 1.0
    bad_param_state[2] = True
    bad_param_state[4] = 1.0
    bad_param_state[5] = 1.0
    bad_params[0] = tuple(bad_param_state)
    bad_status = good_status.model_copy(update={"params": bad_params})
    current_status = good_status.model_copy(
        update={"normalize_mean": not good_status.normalize_mean}
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        win,
        "_show_warning",
        lambda title, text: warnings.append((title, text)),
    )

    win._flush_pending_history_write()
    win._prev_states.clear()
    win._prev_states.extend([good_status, bad_status, current_status])
    win._next_states.clear()
    win.undo()

    repaired = win._params[param_name]
    assert repaired.value == pytest.approx(1.0)
    assert repaired.min < repaired.value < repaired.max
    assert repaired.vary is False
    assert warnings

    win._prev_states.clear()
    win._prev_states.append(good_status)
    win._next_states.clear()
    win._next_states.append(bad_status)
    win.redo()

    repaired = win._params[param_name]
    assert repaired.value == pytest.approx(1.0)
    assert repaired.min < repaired.value < repaired.max
    assert repaired.vary is False
    assert len(warnings) == 2


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
    win.scale_covar_check.setChecked(False)

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
    assert not win_restored.scale_covar_check.isChecked()

    overridden = erlab.interactive.ftool(fit_ds, scale_covar=True, execute=False)
    qtbot.addWidget(overridden)
    assert overridden.scale_covar_check.isChecked()
    assert not overridden._fit_is_current
    assert not overridden.save_button.isEnabled()
    assert not overridden.copy_button.isEnabled()

    supplied_uncertainty = erlab.interactive.ftool(
        fit_ds,
        uncertainty=xr.full_like(data, 0.2),
        execute=False,
    )
    qtbot.addWidget(supplied_uncertainty)
    assert not supplied_uncertainty._fit_is_current
    assert not supplied_uncertainty.save_button.isEnabled()
    assert not supplied_uncertainty.copy_button.isEnabled()

    scaled_fit_ds = data.xlm.modelfit(
        "t",
        model=exp_decay_model,
        params=params,
        scale_covar=True,
    ).load()
    scaled_with_uncertainty = erlab.interactive.ftool(
        scaled_fit_ds,
        uncertainty=xr.full_like(data, 0.2),
        execute=False,
    )
    qtbot.addWidget(scaled_with_uncertainty)
    assert not scaled_with_uncertainty.scale_covar_check.isChecked()
    assert not scaled_with_uncertainty._fit_is_current

    weights = xr.DataArray(
        np.linspace(0.3, 2.7, data.size), dims="t", coords={"t": data.t}
    )
    assert not np.array_equal(weights, 1.0 / (1.0 / weights))
    weighted_fit_ds = data.xlm.modelfit(
        "t",
        model=exp_decay_model,
        params=params,
        weights=weights,
        scale_covar=False,
    ).load()
    weighted_restored = erlab.interactive.ftool(weighted_fit_ds, execute=False)
    qtbot.addWidget(weighted_restored)
    xr.testing.assert_identical(
        weighted_restored._direct_weights, weighted_fit_ds.modelfit_weights
    )
    np.testing.assert_array_equal(weighted_restored._fit_weights(), weights)
    assert weighted_restored.uncertainty is not None
    xr.testing.assert_allclose(weighted_restored.uncertainty, 1.0 / weights)
    np.testing.assert_allclose(
        weighted_restored.data_errorbar.opts["top"], 1.0 / weights
    )
    assert weighted_restored._fit_is_current
    assert weighted_restored.save_button.isEnabled()
    assert weighted_restored.copy_button.isEnabled()

    namespace = {"weighted_fit_ds": weighted_fit_ds, "model": exp_decay_model}
    exec(  # noqa: S102
        f"{weighted_restored._copy_prelude()}\n"
        f"replayed = {weighted_restored._fit_expression()}",
        namespace,
    )
    np.testing.assert_array_equal(
        namespace["replayed"].modelfit_results.compute().item().weights,
        weights,
    )

    workspace_restored = erlab.interactive.utils.ToolWindow.from_dataset(
        weighted_restored.to_dataset(), _code_trust=new_document_trust()
    )
    qtbot.addWidget(workspace_restored)
    assert isinstance(workspace_restored, Fit1DTool)
    xr.testing.assert_identical(
        workspace_restored._direct_weights, weighted_fit_ds.modelfit_weights
    )
    np.testing.assert_array_equal(workspace_restored._fit_weights(), weights)

    weighted_restored.domain_min_spin.setValue(0.5)
    weighted_restored.domain_max_spin.setValue(3.5)
    cropped_data = weighted_restored._fit_data_raw()
    np.testing.assert_array_equal(
        weighted_restored._fit_weights(),
        weights.sel(t=cropped_data.t),
    )

    weighted_restored._direct_weights_name = "fit_weights"
    collision_namespace = {
        "weighted_fit_ds": weighted_fit_ds,
        "model": exp_decay_model,
        "fit_weights": weights,
    }
    exec(  # noqa: S102
        f"{weighted_restored._copy_prelude()}\n"
        f"replayed = {weighted_restored._fit_expression()}",
        collision_namespace,
    )
    np.testing.assert_array_equal(
        collision_namespace["replayed"].modelfit_results.compute().item().weights,
        weights.sel(t=cropped_data.t),
    )
    assert workspace_restored._fit_is_current
    updated_data = data * 1.01
    assert workspace_restored.update_inputs({"data": updated_data})
    xr.testing.assert_identical(
        workspace_restored._direct_weights, weighted_fit_ds.modelfit_weights
    )
    np.testing.assert_array_equal(workspace_restored._fit_weights(), weights)

    legacy_weighted = erlab.interactive.ftool(
        weighted_fit_ds.drop_vars("modelfit_weights"), execute=False
    )
    qtbot.addWidget(legacy_weighted)
    assert legacy_weighted.uncertainty is None
    assert legacy_weighted.scale_covar_check.isChecked()
    assert not legacy_weighted._fit_is_current
    assert not legacy_weighted.save_button.isEnabled()
    assert not legacy_weighted.copy_button.isEnabled()

    invalid_weighted_ds = weighted_fit_ds.copy()
    invalid_weighted_ds["modelfit_weights"] = -weights
    invalid_weighted = erlab.interactive.ftool(invalid_weighted_ds, execute=False)
    qtbot.addWidget(invalid_weighted)
    assert invalid_weighted.uncertainty is None
    assert not invalid_weighted._fit_is_current

    zero_weights = weights.copy()
    zero_weights[5] = 0.0
    zero_weighted_ds = data.xlm.modelfit(
        "t",
        model=exp_decay_model,
        params=params,
        weights=zero_weights,
        scale_covar=False,
    ).load()
    zero_weighted = erlab.interactive.ftool(zero_weighted_ds, execute=False)
    qtbot.addWidget(zero_weighted)
    xr.testing.assert_identical(
        zero_weighted._direct_weights, zero_weighted_ds.modelfit_weights
    )
    np.testing.assert_array_equal(zero_weighted._fit_weights(), zero_weights)
    assert zero_weighted._fit_is_current
    assert np.isnan(zero_weighted.data_errorbar.opts["top"][5])
    np.testing.assert_allclose(
        np.delete(zero_weighted.data_errorbar.opts["top"], 5),
        np.delete(1.0 / zero_weights.where(zero_weights != 0).values, 5),
    )

    scalar_weights = xr.DataArray(0.3)
    scalar_fit_ds = data.xlm.modelfit(
        "t",
        model=exp_decay_model,
        params=params,
        weights=scalar_weights,
        scale_covar=False,
    ).load()
    scalar_restored = erlab.interactive.ftool(scalar_fit_ds, execute=False)
    qtbot.addWidget(scalar_restored)
    xr.testing.assert_identical(
        scalar_restored._direct_weights, scalar_fit_ds.modelfit_weights
    )
    xr.testing.assert_identical(
        scalar_restored._fit_weights(), scalar_fit_ds.modelfit_weights
    )
    np.testing.assert_allclose(
        scalar_restored.data_errorbar.opts["top"], 1.0 / scalar_weights
    )
    scalar_namespace = {
        "scalar_fit_ds": scalar_fit_ds,
        "model": exp_decay_model,
    }
    exec(  # noqa: S102
        f"{scalar_restored._copy_prelude()}\n"
        f"replayed = {scalar_restored._fit_expression()}",
        scalar_namespace,
    )
    xr.testing.assert_identical(
        scalar_namespace["replayed"].modelfit_weights,
        scalar_fit_ds.modelfit_weights,
    )
    scalar_restored.normalize_check.setChecked(True)
    norm = scalar_restored._fit_normalization_factor()
    assert norm is not None
    norm = abs(norm)
    xr.testing.assert_identical(
        scalar_restored._fit_weights(), scalar_fit_ds.modelfit_weights * norm
    )
    np.testing.assert_allclose(
        scalar_restored._normalized_direct_weights_values(),
        np.full(data.size, scalar_weights.item() * norm),
    )
    np.testing.assert_allclose(
        scalar_restored.data_errorbar.opts["top"],
        np.full(data.size, 1.0 / scalar_weights.item() / norm),
    )
    scalar_namespace = {
        "scalar_fit_ds": scalar_fit_ds,
        "model": exp_decay_model,
    }
    exec(  # noqa: S102
        f"{scalar_restored._copy_prelude()}\n"
        f"replayed = {scalar_restored._fit_expression()}",
        scalar_namespace,
    )
    np.testing.assert_allclose(
        scalar_namespace["replayed"].modelfit_results.compute().item().weights,
        np.full(data.size, scalar_weights.item() * norm),
    )
    assert scalar_namespace["replayed"].modelfit_weights.dims == ()
    xr.testing.assert_allclose(
        scalar_namespace["replayed"].modelfit_weights,
        scalar_fit_ds.modelfit_weights * norm,
    )


def test_fit1d_persistence_roundtrip_preserves_fit_result(
    qtbot, exp_decay_model
) -> None:
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
    expected_fit_ds = win._last_result_ds.copy(deep=True)
    expected_status = win.tool_status.model_dump()

    win_restored = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(), _code_trust=new_document_trust()
    )
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, Fit1DTool)

    assert win_restored._last_result_ds is not None
    _assert_fit_result_dataset_equivalent(win_restored._last_result_ds, expected_fit_ds)
    assert win_restored.tool_status.model_dump() == expected_status
    assert win_restored._fit_is_current
    assert win_restored.save_button.isEnabled()
    assert win_restored.copy_button.isEnabled()


def test_fit1d_persistence_roundtrip_preserves_stale_fit(
    qtbot, exp_decay_model
) -> None:
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

    index = win.param_model.index(0, 1)
    current_value = float(win.param_model.data(index, QtCore.Qt.ItemDataRole.EditRole))
    assert win.param_model.setData(
        index, f"{current_value + 0.5}", QtCore.Qt.ItemDataRole.EditRole
    )
    assert win._last_result_ds is not None
    assert win._fit_is_current is False
    expected_fit_ds = win._last_result_ds.copy(deep=True)
    expected_status = win.tool_status.model_dump()

    win_restored = erlab.interactive.utils.ToolWindow.from_dataset(
        win.to_dataset(), _code_trust=new_document_trust()
    )
    qtbot.addWidget(win_restored)
    assert isinstance(win_restored, Fit1DTool)

    assert win_restored._last_result_ds is not None
    _assert_fit_result_dataset_equivalent(win_restored._last_result_ds, expected_fit_ds)
    assert win_restored.tool_status.model_dump() == expected_status
    assert win_restored._fit_is_current is False
    assert not win_restored.save_button.isEnabled()
    assert not win_restored.copy_button.isEnabled()


def test_fit1d_persisted_result_digest_blocks_swapped_payload(
    qtbot, monkeypatch
) -> None:
    t = np.linspace(0.0, 4.0, 25)
    data = xr.DataArray(
        3.0 * np.exp(-t / 2.0), dims=("t",), coords={"t": t}, name="decay"
    )
    model = lmfit.models.ExponentialModel()
    source = erlab.interactive.ftool(
        data,
        model=model,
        params=model.make_params(amplitude=2.0, decay=1.0),
        execute=False,
    )
    qtbot.addWidget(source)
    assert source._run_fit()
    qtbot.waitUntil(lambda: source._last_result_ds is not None, timeout=10000)
    qtbot.waitUntil(lambda: not source._fit_running(), timeout=10000)

    tampered = source.to_dataset().copy(deep=True)
    payload = np.asarray(tampered[source._PERSISTED_FIT_RESULT_VAR].values).copy()
    payload[-1] ^= np.uint8(1)
    tampered[source._PERSISTED_FIT_RESULT_VAR] = xr.DataArray(
        payload,
        dims=(source._PERSISTED_FIT_RESULT_DIM,),
    )

    decode_calls: list[None] = []

    def fail_if_decoded(_payload):
        decode_calls.append(None)
        raise AssertionError("tampered fit result was decoded")

    monkeypatch.setattr(
        erlab.interactive.utils,
        "_deserialize_fit_dataset_blob",
        fail_if_decoded,
    )
    restored = erlab.interactive.utils.ToolWindow.from_dataset(
        tampered,
        _code_trust=new_document_trust(),
        _defer_restore_work=True,
    )
    qtbot.addWidget(restored)

    assert decode_calls == []
    assert not document_trust_has_trusted_lineage(restored._document_trust)

    duplicate = restored.duplicate()
    qtbot.addWidget(duplicate)
    assert decode_calls == []
    assert not document_trust_has_trusted_lineage(duplicate._document_trust)


def test_fit1d_update_inputs_preserves_state_and_refit(
    qtbot, exp_decay_model, monkeypatch
):
    data = _make_1d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    win = erlab.interactive.ftool(
        data, model=exp_decay_model, params=params, execute=False
    )
    qtbot.addWidget(win)

    win.normalize_check.setChecked(True)
    win.components_check.setChecked(True)
    win.domain_min_spin.setValue(-0.5)
    win.domain_max_spin.setValue(0.5)
    win.timeout_spin.setValue(3.0)
    win.nfev_spin.setValue(250)
    win.refit_on_source_update_check.setChecked(False)
    win._last_result_ds = xr.Dataset()

    called: list[bool] = []
    monkeypatch.setattr(win, "_run_fit", lambda: called.append(True) or True)

    status = win.tool_status
    new_data = xr.DataArray(
        np.linspace(0.5, 1.5, data.size),
        dims=data.dims,
        coords=data.coords,
        name=data.name,
    )
    win.update_inputs({"data": new_data})

    assert win.tool_status == status
    xr.testing.assert_identical(win.tool_data, new_data)
    assert win._fit_is_current is False
    assert not called

    win._last_result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)
    newer_data = new_data.copy(deep=True)
    newer_data.data = np.asarray(newer_data.data) * 1.1
    win.update_inputs({"data": newer_data})

    assert called == [True]


def test_fit1d_validate_update_inputs_invalid_input_keeps_existing_ui(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    old_central = win.centralWidget()
    bad_data = xr.DataArray(np.arange(6).reshape((2, 3)), dims=("y", "x"))

    with pytest.raises(ValueError, match="1D DataArray"):
        win.validate_update_inputs({"data": bad_data})

    uncertainty = xr.full_like(data, 0.2)
    validated = win.validate_update_inputs({"data": data, "uncertainty": uncertainty})
    xr.testing.assert_identical(validated["data"], data)
    xr.testing.assert_identical(validated["uncertainty"], uncertainty)

    win._set_direct_weights(xr.ones_like(data))
    with pytest.raises(ValueError, match="align exactly"):
        win.validate_update_inputs(
            {"data": data.assign_coords(x=np.asarray(data.x) + 1.0)}
        )

    assert win.centralWidget() is old_central
    assert old_central is not None
    assert old_central.parent() is not None
    xr.testing.assert_identical(win.tool_data, data)


def test_fit1d_update_inputs_drops_missing_coord_backed_param_bindings(qtbot) -> None:
    data = _make_1d_data().assign_coords(offset=1.25)
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    win.param_view.selectRow(0)
    qtbot.waitUntil(lambda: win._current_row == 0)

    param_name = win.param_model.param_name(0)
    win._params_from_coord[param_name] = "offset"
    win._params[param_name].value = float(data["offset"].values)
    win._params[param_name].vary = False
    win.param_model.set_params(win._params, win._params_from_coord)

    value_index = win.param_model.index(0, 1)
    assert not (win.param_model.flags(value_index) & QtCore.Qt.ItemFlag.ItemIsEditable)

    new_data = data.drop_vars("offset")
    win.update_inputs({"data": new_data})

    assert param_name not in win._params_from_coord
    value_index = win.param_model.index(0, 1)
    assert win.param_model.flags(value_index) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert win.param_mode_combo.currentText() == "Manual"


def test_fit1d_apply_inputs_returns_false_if_fit_thread_stays_alive(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    class _StuckThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.join_timeout: float | None = None

        def cancel(self) -> None:
            self.cancel_called = True

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    stuck_thread = _StuckThread()
    win._fit_thread = stuck_thread  # type: ignore[assignment]
    old_central = win.centralWidget()

    new_data = xr.DataArray(
        np.linspace(0.5, 1.5, data.size),
        dims=data.dims,
        coords=data.coords,
        name=data.name,
    )

    script_input = ScriptInput(name="data")
    win.set_script_inputs((script_input,), primary_input="data")
    assert win._apply_inputs({"data": new_data}, (script_input,)) is False
    assert stuck_thread.cancel_called
    assert stuck_thread.join_timeout == win.BACKGROUND_TASK_TIMEOUT_MS / 1000
    assert win.centralWidget() is old_central
    assert old_central is not None
    assert old_central.parent() is not None
    xr.testing.assert_identical(win.tool_data, data)


def test_fit1d_update_inputs_keeps_fit_finished_receivers_constant(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    initial_receivers = signal_receiver_count(win, win.sigFitFinished, "sigFitFinished")

    for scale in (1.1, 1.2, 1.3):
        updated = data.copy(deep=True)
        updated.data = np.asarray(data.data) * scale
        win.update_inputs({"data": updated})
        assert (
            signal_receiver_count(win, win.sigFitFinished, "sigFitFinished")
            == initial_receivers
        )


def test_fit1d_apply_inputs_auto_refit_after_waiting_cancelled_thread(
    qtbot, monkeypatch
) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    class _FinishedThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.join_timeout: float | None = None

        def cancel(self) -> None:
            self.cancel_called = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

        def is_alive(self) -> bool:
            return False

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
    assert old_thread.join_timeout == win.BACKGROUND_TASK_TIMEOUT_MS / 1000


def test_fit1d_refit_start_failure_commits_published_input(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    updated = data * 1.1
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)
    win._last_result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)
    monkeypatch.setattr(win, "_run_fit", lambda: False)
    old_snapshot = "old"
    old_binding = ScriptInput(name="data", node_snapshot_token=old_snapshot)
    refreshed_binding = old_binding.model_copy(update={"node_snapshot_token": "new"})
    win.set_script_inputs((old_binding,), primary_input="data")

    assert win._apply_inputs({"data": updated}, (refreshed_binding,)) is True

    assert win.source_state == "fresh"
    assert win.script_inputs == (refreshed_binding,)
    assert win._pending_script_inputs is None
    xr.testing.assert_identical(win.tool_data, updated)


@pytest.mark.parametrize("terminal", ["error", "timeout"])
def test_fit1d_async_refit_terminal_commits_published_input(
    qtbot, monkeypatch, terminal: str
) -> None:
    data = _make_1d_data()
    updated = data * 1.1
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)
    win._last_result_ds = xr.Dataset()
    win.refit_on_source_update_check.setChecked(True)
    monkeypatch.setattr(win, "_run_fit", lambda: True)
    monkeypatch.setattr(win, "_show_error", lambda *_args, **_kwargs: None)
    old_snapshot = "old"
    old_binding = ScriptInput(name="data", node_snapshot_token=old_snapshot)
    refreshed_binding = old_binding.model_copy(update={"node_snapshot_token": "new"})
    win.set_script_inputs((old_binding,), primary_input="data")

    assert win._apply_inputs({"data": updated}, (refreshed_binding,)) is False
    assert win._source_refresh_deferred is True
    if terminal == "error":
        win._fit_errored("fit failed")
    else:
        win._fit_timed_out(fit1d.time.perf_counter())

    assert win.source_state == "fresh"
    assert win.script_inputs == (refreshed_binding,)
    assert win._pending_script_inputs is None
    assert win._source_refresh_deferred is False
    xr.testing.assert_identical(win.tool_data, updated)


def test_fit1d_finalize_ignores_stale_thread(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    class _ThreadPlaceholder:
        pass

    stale_thread = _ThreadPlaceholder()
    current_thread = _ThreadPlaceholder()
    win._fit_thread = current_thread  # type: ignore[assignment]

    win._finalize_fit_thread(stale_thread)  # type: ignore[arg-type]

    assert win._fit_thread is current_thread
    win._fit_thread = None


def test_parameter_table_model_and_delegate(qtbot) -> None:
    params = lmfit.Parameters()
    params.add("amp", value=1.0, min=-1.0, max=2.0, vary=True)
    params.add("expr_param", value=2.0, expr="2*amp")
    params_from_coord = {"amp": "temp"}
    model = _ParameterTableModel(params, params_from_coord)

    assert model.rowCount() == 2
    assert model.columnCount() == 6
    assert model.rowCount(model.index(0, 0)) == 0
    assert model.columnCount(model.index(0, 0)) == 0
    assert (
        model.headerData(0, QtCore.Qt.Orientation.Horizontal) == model._COLUMN_NAMES[0]
    )
    assert model.headerData(99, QtCore.Qt.Orientation.Horizontal) is None

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


def test_parameter_table_model_rejects_invalid_bounds() -> None:
    params = lmfit.Parameters()
    params.add("amp", value=1.0, min=-1.0, max=2.0, vary=True)
    model = _ParameterTableModel(params, {})

    changed: list[bool] = []
    invalid_bounds: list[str] = []
    model.sigParamsChanged.connect(lambda: changed.append(True))
    model.sigInvalidBounds.connect(invalid_bounds.append)

    min_index = model.index(0, 3)
    max_index = model.index(0, 4)
    original = (params["amp"].value, params["amp"].min, params["amp"].max)

    assert not model.setData(min_index, "2.0", QtCore.Qt.ItemDataRole.EditRole)
    assert (params["amp"].value, params["amp"].min, params["amp"].max) == original
    assert changed == []
    assert invalid_bounds == ["amp"]

    invalid_bounds.clear()
    assert not model.setData(max_index, "-1.0", QtCore.Qt.ItemDataRole.EditRole)
    assert (params["amp"].value, params["amp"].min, params["amp"].max) == original
    assert changed == []
    assert invalid_bounds == ["amp"]

    assert model.setData(max_index, "0.5", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].value == pytest.approx(0.5)
    assert params["amp"].min == pytest.approx(-1.0)
    assert params["amp"].max == pytest.approx(0.5)
    assert len(changed) == 1

    params["amp"].set(value=0.0, min=-1.0, max=2.0)
    assert model.setData(min_index, "0.5", QtCore.Qt.ItemDataRole.EditRole)
    assert params["amp"].value == pytest.approx(0.5)
    assert params["amp"].min == pytest.approx(0.5)
    assert params["amp"].max == pytest.approx(2.0)


def test_fit1d_invalid_bound_edit_warns_without_history(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    param = win.param_model.param_at(0)
    param.set(value=0.0, min=-1.0, max=2.0)
    win._reset_history_stack()
    history_len = len(win._prev_states)
    warnings: list[tuple[str, str]] = []
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
    assert len(win._prev_states) == history_len
    assert warnings


def test_fit1d_parameter_context_menu_builds_actions(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)

    index = win.param_model.index(0, 0)
    menus: list[QtWidgets.QMenu] = []
    exec_positions: list[QtCore.QPoint] = []

    class _TrackingMenu(QtWidgets.QMenu):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            menus.append(self)

        def exec(self, pos):
            exec_positions.append(pos)

    monkeypatch.setattr(win.param_view, "indexAt", lambda _pos: index)
    monkeypatch.setattr(fit1d.QtWidgets, "QMenu", _TrackingMenu)

    win._show_param_menu(QtCore.QPoint(1, 1))

    assert len(menus) == 1
    assert [action.text() for action in menus[0].actions()] == [
        "Set expression…",
        "Clear expression",
    ]
    assert exec_positions


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


def test_fit1d_nonuniform_convolved_preview_opens_without_error(
    qtbot, monkeypatch
) -> None:
    x = np.r_[np.linspace(-1.0, -0.2, 41), np.linspace(0.2, 1.0, 41)]
    data = np.exp(-((x - 0.25) ** 2) / (2.0 * 0.2**2))
    darr = xr.DataArray(data, dims=("x",), coords={"x": x}, name="nonuniform")
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        fit1d.Fit1DTool,
        "_show_error",
        lambda _self, title, text: errors.append((title, text)),
    )

    win = erlab.interactive.ftool(darr, execute=False)
    qtbot.addWidget(win)

    assert errors == []
    assert not win._model.func.convolve
    assert win._auto_segmented(convolve=True)
    win.convolve_check.setChecked(True)
    assert isinstance(
        win._model.func, erlab.analysis.fit.functions.dynamic.MultiPeakFunction
    )
    assert win._model.func.convolve
    assert win._model.func.segmented
    win._update_fit_curve()
    assert errors == []
    assert win._last_fit_y is not None


def test_fit1d_irregular_convolved_model_disables_unsafe_segments(
    qtbot, monkeypatch
) -> None:
    x = np.array([0.0, 1.0, 2.7, 4.1, 7.6, 8.2], dtype=float)
    data = np.exp(-((x - 4.0) ** 2) / 5.0)
    darr = xr.DataArray(data, dims=("sample_temp",), coords={"sample_temp": x})
    model = erlab.analysis.fit.models.MultiPeakModel(
        npeaks=1,
        peak_shapes="lorentzian",
        convolve=True,
        segmented=True,
        oversample=3,
    )
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        fit1d.Fit1DTool,
        "_show_error",
        lambda _self, title, text: errors.append((title, text)),
    )

    win = erlab.interactive.ftool(darr, model=model, execute=False)
    qtbot.addWidget(win)

    assert win._model.func.convolve
    assert not win._model.func.segmented
    assert not win._auto_segmented(convolve=True)
    win._update_fit_curve()
    assert errors == []
    assert win._last_residual is not None
    assert win._last_residual.shape == data.shape


def test_fit1d_restore_irregular_segmented_model_does_not_crash(
    qtbot, monkeypatch
) -> None:
    x = np.array([0.0, 1.0, 2.7, 4.1, 7.6, 8.2], dtype=float)
    data = np.exp(-((x - 4.0) ** 2) / 5.0)
    darr = xr.DataArray(data, dims=("sample_temp",), coords={"sample_temp": x})
    saved_model = erlab.analysis.fit.models.MultiPeakModel(
        npeaks=1,
        peak_shapes="lorentzian",
        convolve=True,
        segmented=True,
        oversample=3,
    )
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        fit1d.Fit1DTool,
        "_show_error",
        lambda _self, title, text: errors.append((title, text)),
    )
    win = erlab.interactive.ftool(darr, execute=False)
    qtbot.addWidget(win)

    status = win.tool_status.model_copy(
        update={
            "model_state": (
                f"{saved_model.__class__.__module__}:"
                f"{saved_model.__class__.__qualname__}",
                saved_model.dumps(),
            ),
            "params": win._serialize_params(saved_model.make_params()),
        }
    )
    win.tool_status = status

    assert win._model.func.convolve
    assert not win._model.func.segmented
    win._update_fit_curve()
    assert errors == []
    assert win._last_residual is not None
    assert win._last_residual.shape == data.shape


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


def test_fit1d_next_multi_step_is_deferred(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    started_steps: list[int] = []

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

    win._run_fit_multiple(2)

    assert started_steps == [1]
    qtbot.waitUntil(lambda: started_steps == [1, 2], timeout=1000)


def test_fit1d_multi_step_requests_paint_before_deferred_next_step(
    qtbot, monkeypatch
) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    events: list[str] = []

    monkeypatch.setattr(win, "_fit_multi_live_refresh_due", lambda: True)
    monkeypatch.setattr(win, "_request_fit_step_paint", lambda: events.append("paint"))
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

    win._run_fit_multiple(2)

    assert events == ["start-1", "paint"]
    qtbot.waitUntil(lambda: events == ["start-1", "paint", "start-2"], timeout=1000)


def test_fit1d_multi_fit_throttles_visible_refreshes(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    clock_values = [100.0, 100.05, 100.30]

    def _monotonic() -> float:
        return clock_values.pop(0) if clock_values else 100.10

    events: list[str] = []
    started_steps: list[int] = []
    monkeypatch.setattr(fit1d.time, "monotonic", _monotonic)
    monkeypatch.setattr(win, "_update_fit_curve", lambda: events.append("curve"))
    monkeypatch.setattr(
        win, "_refresh_slider_from_model", lambda: events.append("slider")
    )

    def _set_fit_stats(*args, **kwargs) -> None:
        events.append(f"stats-{kwargs.get('emit_info', True)}")

    monkeypatch.setattr(win, "_set_fit_stats", _set_fit_stats)
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

    win._run_fit_multiple(3)
    qtbot.waitUntil(
        lambda: win._fit_multi_total is None and not win._fit_running(),
        timeout=1000,
    )

    assert started_steps == [1, 2, 3]
    assert events == [
        "curve",
        "slider",
        "stats-False",
        "curve",
        "slider",
        "stats-True",
    ]
    assert win._write_history is True


def test_fit1d_fit_step_paint_repaints_targets_without_queued_update(
    qtbot, monkeypatch
) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    events: list[str] = []

    class _PaintWidget(QtWidgets.QWidget):
        def __init__(self, name: str, *, visible: bool = True) -> None:
            super().__init__()
            self._name = name
            self._visible = visible

        def update(self) -> None:
            events.append(f"update-{self._name}")

        def repaint(self) -> None:
            events.append(f"repaint-{self._name}")

        def isVisible(self) -> bool:
            return self._visible

    widgets = (
        _PaintWidget("plot"),
        _PaintWidget("hidden", visible=False),
        _PaintWidget("params"),
    )
    for widget in widgets:
        qtbot.addWidget(widget)
    monkeypatch.setattr(win, "_fit_step_paint_widgets", lambda: widgets)
    monkeypatch.setattr(
        QtWidgets.QApplication,
        "processEvents",
        lambda *args, **kwargs: events.append("process"),
    )

    win._request_fit_step_paint()

    assert events == [
        "repaint-plot",
        "repaint-params",
    ]


def test_fit1d_fit_step_paint_widgets_skip_invalid_entries(qtbot) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    win.aic_value = object()
    duplicate = win.bic_value
    win.fit_multi_button = duplicate

    widgets = win._fit_step_paint_widgets()

    assert all(isinstance(widget, QtWidgets.QWidget) for widget in widgets)
    assert sum(widget is duplicate for widget in widgets) == 1


def test_fit1d_fit_progress_paint_widgets_skip_invalid_entries(qtbot) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    win.redchi_value = object()
    duplicate = win.bic_value
    win.fit_multi_button = duplicate

    widgets = win._fit_progress_paint_widgets()

    assert all(isinstance(widget, QtWidgets.QWidget) for widget in widgets)
    assert sum(widget is duplicate for widget in widgets) == 1


def test_fit1d_request_fit_progress_paint_repaints_visible_widgets(
    qtbot, monkeypatch
) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    events: list[str] = []

    class _PaintWidget(QtWidgets.QWidget):
        def __init__(self, name: str, *, visible: bool = True) -> None:
            super().__init__()
            self._name = name
            self._visible = visible

        def repaint(self) -> None:
            events.append(self._name)

        def isVisible(self) -> bool:
            return self._visible

    widgets = (
        _PaintWidget("visible"),
        _PaintWidget("hidden", visible=False),
    )
    for widget in widgets:
        qtbot.addWidget(widget)
    monkeypatch.setattr(win, "_fit_progress_paint_widgets", lambda: widgets)

    win._request_fit_progress_paint()

    assert events == ["visible"]


def test_fit1d_set_fit_ds_updates_without_param_changed_signal(
    qtbot, monkeypatch
) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    param_name = next(iter(win._params))
    params = win._params.copy()
    expected_value = params[param_name].value + 0.25
    params[param_name].set(value=expected_value)
    result_ds = _fit_result_dataset(params, nfev=7)

    param_changed: list[None] = []
    events: list[str] = []
    stats_nfev: list[int] = []
    win.param_model.sigParamsChanged.connect(lambda: param_changed.append(None))
    monkeypatch.setattr(win, "_update_fit_curve", lambda: events.append("curve"))
    monkeypatch.setattr(
        win, "_refresh_slider_from_model", lambda: events.append("slider")
    )

    def _set_fit_stats(result, *, elapsed=None) -> None:
        del elapsed
        stats_nfev.append(result.nfev)

    monkeypatch.setattr(win, "_set_fit_stats", _set_fit_stats)

    win._set_fit_ds(result_ds, 0.0)

    assert param_changed == []
    assert win.param_model.params[param_name].value == pytest.approx(expected_value)
    assert events == ["curve", "slider"]
    assert stats_nfev == [7]


def test_fit1d_cancelled_before_deferred_multi_step_stops_sequence(
    qtbot, monkeypatch
) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    started_steps: list[int] = []

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

    win._run_fit_multiple(2)
    assert started_steps == [1]

    assert win._cancel_fit()
    qtbot.waitUntil(
        lambda: win._fit_multi_total is None and not win._fit_cancel_requested,
        timeout=1000,
    )
    assert started_steps == [1]
    assert win._fit_running_multi is False


def test_fit1d_defer_next_fit_step_refreshes_due_live_view(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    events: list[str] = []
    win._fit_multi_total = 2
    monkeypatch.setattr(win, "_fit_multi_live_refresh_due", lambda: True)
    monkeypatch.setattr(
        win, "_sync_multi_fit_view", lambda **_kwargs: events.append("sync")
    )
    monkeypatch.setattr(win, "_request_fit_step_paint", lambda: events.append("paint"))
    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda _parent, _delay, callback: callback(),
    )

    win._defer_next_fit_step(lambda: events.append("callback"))
    win._fit_multi_total = None

    assert events == ["sync", "paint", "callback"]


def test_fit1d_defer_next_fit_step_paints_single_sequence(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    events: list[str] = []
    monkeypatch.setattr(win, "_request_fit_step_paint", lambda: events.append("paint"))
    monkeypatch.setattr(
        erlab.interactive.utils,
        "single_shot",
        lambda _parent, _delay, callback: callback(),
    )

    win._defer_next_fit_step(lambda: events.append("callback"))

    assert events == ["paint", "callback"]


def test_fit1d_fit_curve_residual_success_and_failure(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    fit_calls: list[tuple[np.ndarray, np.ndarray]] = []
    residual_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _Curve:
        def __init__(self, calls) -> None:
            self._calls = calls

        def setData(self, *args, **kwargs) -> None:
            self._calls.append((args, kwargs))

    win.fit_curve = _Curve(fit_calls)
    win.residual_curve = _Curve(residual_calls)
    xvals = np.array([0.0, 1.0, 2.0])
    monkeypatch.setattr(win, "_x_values", lambda: xvals)
    monkeypatch.setattr(win, "_has_non_finite_params", lambda: False)
    monkeypatch.setattr(win, "_model_eval_values", lambda values: np.zeros_like(values))
    monkeypatch.setattr(
        win, "_residuals_from_result", lambda _values: np.array([0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(win, "_domain_brushes", lambda _values: None)
    monkeypatch.setattr(win, "_update_component_curves", lambda _values: None)
    monkeypatch.setattr(win, "_update_peak_lines", lambda _values: None)

    win._update_fit_curve()

    np.testing.assert_allclose(residual_calls[-1][0][0], xvals)
    np.testing.assert_allclose(residual_calls[-1][0][1], [0.1, 0.2, 0.3])
    assert win._last_residual is not None

    def raise_residuals(_values):
        raise RuntimeError("bad residuals")

    monkeypatch.setattr(win, "_residuals_from_result", raise_residuals)
    win._update_fit_curve()

    assert residual_calls[-1][0] == ([], [])
    assert win._last_residual is None


def test_fit1d_model_eval_y_independent_shape_mismatch(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    captured: list[dict[str, np.ndarray]] = []

    class _Model:
        independent_vars = ("x", "y")

        def eval(self, *, params, **kwargs):
            del params
            captured.append(kwargs)
            return np.array([1.0, 2.0])

    win._model = _Model()
    monkeypatch.setattr(win, "_normalized_data_values", lambda: np.ones(3))

    with pytest.raises(ValueError, match="2 values for 3 x values"):
        win._model_eval_values(np.arange(3.0))

    assert "y" in captured[0]


def test_fit1d_segmented_convolution_support_edges(monkeypatch) -> None:
    assert not Fit1DTool._x_values_support_segmented_convolution(np.array([0.0, 1.0]))
    assert not Fit1DTool._x_values_support_segmented_convolution(
        np.array([0.0, np.nan, 1.0])
    )

    import erlab.utils._array_jit as array_jit

    monkeypatch.setattr(
        array_jit,
        "_split_uniform_segments",
        lambda _values: (_ for _ in ()).throw(RuntimeError("bad segments")),
    )
    assert not Fit1DTool._x_values_support_segmented_convolution(
        np.array([0.0, 0.5, 1.0])
    )


def test_fit1d_equal_bound_param_state_repair_edges(monkeypatch) -> None:
    assert Fit1DTool._repair_equal_bound_param_state(("amp",), []) is None
    assert (
        Fit1DTool._repair_equal_bound_param_state(
            ("amp", 1.0, True, None, "bad", "bad"), []
        )
        is None
    )
    assert (
        Fit1DTool._repair_equal_bound_param_state(
            ("amp", 1.0, True, None, 0.0, 1.0), []
        )
        is None
    )

    with pytest.raises(ValueError, match="not enough values"):
        Fit1DTool._deserialize_params([("amp", 1.0, True, None, 2.0, 1.0)])

    param = lmfit.Parameter(name="amp", value=1.0, min=0.0, max=2.0)
    bad_state = list(param.__getstate__())
    bad_state[4] = 1.0
    bad_state[5] = 1.0
    monkeypatch.setattr(
        Fit1DTool,
        "_repair_equal_bound_param_state",
        staticmethod(lambda _state, _repaired_bounds: None),
    )
    with pytest.raises(ValueError, match="min == max"):
        Fit1DTool._deserialize_params([tuple(bad_state)])


def test_fit1d_multi_fit_history_and_sync_edges(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    win._fit_multi_last_live_refresh = fit1d.time.monotonic()
    assert not win._fit_multi_live_refresh_due()

    replaced: list[bool] = []
    original_replace_last_state = win._replace_last_state

    def record_replace_last_state(*_args) -> None:
        replaced.append(True)
        original_replace_last_state()

    monkeypatch.setattr(win, "_replace_last_state", record_replace_last_state)
    initial_revision = win.provenance_revision
    win._write_history = True
    win._begin_fit_multi_history()
    assert win._fit_multi_sequence_write_history is True
    assert win._write_history is False
    win._write_state()
    win._write_state()
    assert win.provenance_revision == initial_revision
    win._finish_fit_multi_history()
    assert win._write_history is True
    assert replaced == [True]
    assert win.provenance_revision == initial_revision + 1

    result_ds = _fit_result_dataset(win._params, nfev=4)
    returned_params = win._store_multi_fit_result(result_ds, fit1d.time.perf_counter())
    assert returned_params is win._params
    assert win._fit_multi_live_refresh_pending is True
    assert win._fit_multi_refresh_pending is True

    events: list[str] = []
    monkeypatch.setattr(win, "_update_fit_curve", lambda *_args: events.append("curve"))
    monkeypatch.setattr(
        win, "_refresh_slider_from_model", lambda *_args: events.append("slider")
    )
    monkeypatch.setattr(
        win,
        "_set_fit_stats",
        lambda _result, *, elapsed=None, emit_info=True: events.append(
            f"stats-{emit_info}"
        ),
    )
    monkeypatch.setattr(
        win,
        "_sync_fit_result_state",
        lambda *_args, notify=True: events.append(f"sync-{notify}"),
    )
    monkeypatch.setattr(
        win,
        "_mark_fit_fresh",
        lambda *_args, emit_info=True: events.append(f"fresh-{emit_info}"),
    )
    monkeypatch.setattr(
        win, "finalize_source_refresh", lambda *_args: events.append("source")
    )

    win._fit_multi_refresh_pending = False
    win._sync_multi_fit_view(full=True)
    assert events == []

    win._fit_multi_refresh_pending = True
    win._fit_multi_live_refresh_pending = True
    win._source_refresh_deferred = True
    win._sync_multi_fit_view(full=True)
    assert events == [
        "curve",
        "slider",
        "stats-True",
        "sync-True",
        "fresh-True",
        "source",
    ]

    events.clear()
    win._fit_multi_live_refresh_pending = False
    win._sync_multi_fit_view()
    assert events == []


def test_fit1d_completion_action_exception_recovers(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    assert isinstance(win, Fit1DTool)

    events: list[str] = []

    class _Thread:
        _outcome = fit1d._FitWorkerOutcome("success", result=xr.Dataset())

    thread = _Thread()
    win._fit_thread = thread
    monkeypatch.setattr(win, "_fit_errored", lambda _message: events.append("error"))
    monkeypatch.setattr(win, "_fit_cancelled", lambda: events.append("cancelled"))

    def fail(_result) -> None:
        raise RuntimeError("bad action")

    win._fit_worker_callbacks[thread] = fit1d._FitWorkerCallbacks(
        on_success=fail,
        on_timeout=lambda: None,
        on_error=lambda _message: None,
    )
    win._finalize_fit_thread(thread)

    assert events == ["error", "cancelled"]


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


def test_fit1d_model_file_reviews_exact_candidate_before_decode(
    qtbot, tmp_path, monkeypatch
) -> None:
    tool, _data, _model, _params = _make_linear_fit1d_tool(qtbot, expression=True)
    _set_signed_fit_trust(tool)
    candidate = lmfit.models.ExpressionModel("amplitude * x + offset")
    serialized = candidate.dumps()
    model_path = tmp_path / "candidate.model"
    model_path.write_text(serialized)
    file_index = tool.model_combo.findData("__file")
    events: list[str] = []
    reviewed_manifests = []
    review_options: list[dict[str, object]] = []
    original_loads = lmfit.model.Model.loads

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(model_path), ""),
    )

    def confirm_candidate(_parent, manifest, **kwargs) -> bool:
        events.append("review")
        reviewed_manifests.append(manifest)
        review_options.append(kwargs)
        model_path.write_text(
            lmfit.models.ExpressionModel("different * x + baseline").dumps()
        )
        return True

    def tracked_loads(model, text, *args, **kwargs):
        events.append("decode")
        assert text == serialized
        return original_loads(model, text, *args, **kwargs)

    monkeypatch.setattr(
        erlab.interactive.utils, "confirm_code_trust", confirm_candidate
    )
    monkeypatch.setattr(lmfit.model.Model, "loads", tracked_loads)
    with QtCore.QSignalBlocker(tool.model_combo):
        tool.model_combo.setCurrentIndex(file_index)

    tool._on_model_choice_changed(file_index)

    assert events == ["review", "decode"]
    assert len(reviewed_manifests) == 1
    assert reviewed_manifests[0].entries == (
        tool._model_code_trust_entry(("", serialized)),
    )
    assert review_options[0]["object_name"] == (
        "fit_model_file_code_trust_review_dialog"
    )
    assert isinstance(tool._model, lmfit.models.ExpressionModel)
    assert tool._model.expr == candidate.expr
    assert tool._model_load_path == str(model_path)
    assert _trust_allows_local_code_edit(tool._document_trust)


@pytest.mark.parametrize(
    ("outcome", "expected_reviews", "expected_decodes"),
    [
        pytest.param("cancel", 0, 0, id="file-dialog-cancel"),
        pytest.param("deny", 1, 0, id="review-deny"),
        pytest.param("entry-mutation", 1, 0, id="candidate-entry-mutation"),
        pytest.param("failure", 1, 1, id="decode-failure"),
    ],
)
def test_fit1d_model_file_cancel_deny_and_failure_preserve_state(
    qtbot,
    tmp_path,
    monkeypatch,
    outcome: str,
    expected_reviews: int,
    expected_decodes: int,
) -> None:
    tool, _data, old_model, _params = _make_linear_fit1d_tool(qtbot)
    old_trust = tool._document_trust
    old_model_state = tool._serialized_model_state
    candidate = lmfit.models.ExpressionModel("amplitude * x + offset")
    model_path = tmp_path / "candidate.model"
    model_path.write_text(candidate.dumps())
    file_index = tool.model_combo.findData("__file")
    review_calls = 0
    decode_calls = 0
    errors: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (
            ("", "") if outcome == "cancel" else (str(model_path), "")
        ),
    )

    def confirm_candidate(_parent, manifest, **_kwargs) -> bool:
        nonlocal review_calls
        review_calls += 1
        if outcome == "entry-mutation":
            manifest.entries[0].context["changed-during-review"] = True
        return outcome != "deny"

    def tracked_loads(*_args, **_kwargs):
        nonlocal decode_calls
        decode_calls += 1
        raise RuntimeError("invalid model")

    monkeypatch.setattr(
        erlab.interactive.utils, "confirm_code_trust", confirm_candidate
    )
    monkeypatch.setattr(lmfit.model.Model, "loads", tracked_loads)
    monkeypatch.setattr(tool, "_show_error", lambda *args: errors.append(args))
    with QtCore.QSignalBlocker(tool.model_combo):
        tool.model_combo.setCurrentIndex(file_index)

    tool._on_model_choice_changed(file_index)

    assert review_calls == expected_reviews
    assert decode_calls == expected_decodes
    assert bool(errors) is (outcome == "failure")
    assert tool._model is old_model
    assert tool._serialized_model_state == old_model_state
    assert tool._model_load_path is None
    assert tool._document_trust == old_trust


def test_fit1d_model_file_application_failure_preserves_state(
    qtbot, tmp_path, monkeypatch
) -> None:
    tool, _data, old_model, _params = _make_linear_fit1d_tool(qtbot, expression=True)
    _set_signed_fit_trust(tool)
    old_trust = tool._document_trust
    old_params = tool._params
    old_model_state = tool._serialized_model_state
    old_entries = tool._fit_code_entries
    candidate = lmfit.models.ExpressionModel("amplitude * x + offset")
    model_path = tmp_path / "candidate.model"
    model_path.write_text(candidate.dumps())
    file_index = tool.model_combo.findData("__file")
    original_loads = lmfit.model.Model.loads
    errors: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(model_path), ""),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "confirm_code_trust",
        lambda *_args, **_kwargs: True,
    )

    def tracked_loads(model, text, *args, **kwargs):
        loaded = original_loads(model, text, *args, **kwargs)

        def fail_make_params(*_args, **_kwargs):
            raise RuntimeError("parameter creation failed")

        monkeypatch.setattr(loaded, "make_params", fail_make_params)
        return loaded

    monkeypatch.setattr(lmfit.model.Model, "loads", tracked_loads)
    monkeypatch.setattr(tool, "_show_error", lambda *args: errors.append(args))
    with QtCore.QSignalBlocker(tool.model_combo):
        tool.model_combo.setCurrentIndex(file_index)

    tool._on_model_choice_changed(file_index)

    assert errors
    assert tool._model is old_model
    assert tool._params is old_params
    assert tool._serialized_model_state == old_model_state
    assert tool._model_load_path is None
    assert tool._fit_code_entries is old_entries
    assert tool._document_trust == old_trust


def test_fit1d_model_file_keeps_reviewed_state_without_second_dump(
    qtbot, tmp_path, monkeypatch
) -> None:
    tool, _data, _old_model, _params = _make_linear_fit1d_tool(qtbot, expression=True)
    _set_signed_fit_trust(tool)
    candidate = lmfit.models.ExpressionModel("reviewed * x + offset")
    serialized = candidate.dumps()
    model_path = tmp_path / "candidate.model"
    model_path.write_text(serialized)
    different_serialized = lmfit.models.ExpressionModel(
        "changed * x + offset", init_script="constant = 1"
    ).dumps()
    file_index = tool.model_combo.findData("__file")
    review_calls = 0
    dump_calls = 0
    original_loads = lmfit.model.Model.loads

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(model_path), ""),
    )

    def confirm_candidate(*_args, **_kwargs) -> bool:
        nonlocal review_calls
        review_calls += 1
        return True

    def tracked_dumps(*args, **kwargs):
        nonlocal dump_calls
        dump_calls += 1
        return different_serialized

    def tracked_loads(model, text, *args, **kwargs):
        loaded = original_loads(model, text, *args, **kwargs)
        monkeypatch.setattr(loaded, "dumps", tracked_dumps)
        return loaded

    monkeypatch.setattr(
        erlab.interactive.utils, "confirm_code_trust", confirm_candidate
    )
    monkeypatch.setattr(
        lmfit.model.Model,
        "loads",
        tracked_loads,
    )
    with QtCore.QSignalBlocker(tool.model_combo):
        tool.model_combo.setCurrentIndex(file_index)

    tool._on_model_choice_changed(file_index)

    assert review_calls == 1
    assert dump_calls == 0
    assert isinstance(tool._model, lmfit.models.ExpressionModel)
    assert tool._model.expr == candidate.expr
    assert tool._serialized_model_state == (
        fit1d._model_class_reference(type(tool._model)),
        serialized,
    )
    assert tool._model_load_path == str(model_path)
    assert _trust_allows_local_code_edit(tool._document_trust)


def test_fit1d_safe_model_file_loads_without_review(
    qtbot, tmp_path, monkeypatch
) -> None:
    tool, _data, _model, _params = _make_linear_fit1d_tool(qtbot)
    serialized = lmfit.models.LinearModel().dumps()
    model_path = tmp_path / "linear.model"
    model_path.write_text(serialized)
    file_index = tool.model_combo.findData("__file")
    original_loads = lmfit.model.Model.loads
    decoded: list[str] = []

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(model_path), ""),
    )
    monkeypatch.setattr(
        erlab.interactive.utils,
        "confirm_code_trust",
        lambda *_args, **_kwargs: pytest.fail("safe model requested code review"),
    )

    def tracked_loads(model, text, *args, **kwargs):
        decoded.append(text)
        return original_loads(model, text, *args, **kwargs)

    monkeypatch.setattr(lmfit.model.Model, "loads", tracked_loads)
    with QtCore.QSignalBlocker(tool.model_combo):
        tool.model_combo.setCurrentIndex(file_index)

    tool._on_model_choice_changed(file_index)

    assert decoded == [serialized]
    assert tool._model_load_path == str(model_path)
    assert (
        tool.model_combo.currentData(role=QtCore.Qt.ItemDataRole.UserRole) == "__file"
    )


def test_fit1d_model_file_review_does_not_admit_unrelated_code(
    qtbot, tmp_path, monkeypatch
) -> None:
    old_model = lmfit.models.ExpressionModel("slope * x + intercept")
    old_params = old_model.make_params(slope=1.0, intercept=0.0)
    old_params["intercept"].expr = "2 * slope"
    tool = erlab.interactive.ftool(
        _make_1d_data(), model=old_model, params=old_params, execute=False
    )
    qtbot.addWidget(tool)
    tool.set_document_trust(untrusted_document_trust())
    old_trust = tool._document_trust
    candidate = lmfit.models.ExpressionModel("amplitude * x + offset")
    serialized = candidate.dumps()
    model_path = tmp_path / "candidate.model"
    model_path.write_text(serialized)
    file_index = tool.model_combo.findData("__file")
    reviewed_manifests = []

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(model_path), ""),
    )

    def confirm_candidate(_parent, manifest, **_kwargs) -> bool:
        reviewed_manifests.append(manifest)
        return True

    monkeypatch.setattr(
        erlab.interactive.utils, "confirm_code_trust", confirm_candidate
    )
    with QtCore.QSignalBlocker(tool.model_combo):
        tool.model_combo.setCurrentIndex(file_index)

    tool._on_model_choice_changed(file_index)

    assert len(reviewed_manifests) == 1
    assert len(reviewed_manifests[0].entries) == 1
    assert "2 * slope" not in reviewed_manifests[0].entries[0].code
    assert tool._model is old_model
    assert tool._params["intercept"].expr == "2 * slope"
    assert tool._document_trust == old_trust


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


def test_fit1d_run_fit_start_error_resets_buttons(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

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

    assert win._run_fit() is False

    assert errors
    assert win._fit_thread is None
    assert win._fit_cancel_requested is False
    assert win.fit_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


def test_fit1d_start_worker_start_exception_resets_buttons(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    errors: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        win,
        "_show_error",
        lambda title, text, detailed_text=None: errors.append(
            (title, text, detailed_text)
        ),
    )

    class _FailingWorker:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(fit1d, "_FitWorker", _FailingWorker)

    assert not win._start_fit_worker(
        win._fit_data(),
        win._params,
        multi=False,
        on_success=lambda _result: None,
        on_timeout=lambda: None,
        on_error=lambda _message: None,
    )

    assert errors
    assert not fit1d._running_fit_workers
    assert win._fit_thread is None
    assert win._fit_cancel_requested is False
    assert win.fit_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


def test_fit1d_workers_use_one_scale_covar_snapshot(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)

    captured: list[bool] = []

    class _Worker:
        def __init__(self, *_args, scale_covar: bool, **_kwargs) -> None:
            captured.append(scale_covar)

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(fit1d, "_FitWorker", _Worker)
    callbacks = {
        "on_success": lambda _result: None,
        "on_timeout": lambda: None,
        "on_error": lambda _message: None,
    }

    win.scale_covar_check.setChecked(False)
    assert win._start_fit_worker(win._fit_data(), win._params, multi=True, **callbacks)
    first_worker = win._fit_thread
    assert not win.scale_covar_check.isEnabled()

    win.scale_covar_check.setChecked(True)
    win._fit_thread = None
    assert win._start_fit_worker(win._fit_data(), win._params, multi=True, **callbacks)
    second_worker = win._fit_thread

    assert captured == [False, False]
    win._fit_poll_timer.stop()
    for worker in (first_worker, second_worker):
        if worker is not None:
            fit1d._release_running_fit_worker(worker)
    win._fit_thread = None
    win._fit_is_current = True
    win._set_fit_running(False, multi=True)
    assert win.scale_covar_check.isEnabled()
    assert not win._fit_is_current


def test_fit1d_run_fit_preparation_error_resets_buttons(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

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

    assert win._run_fit() is False

    assert errors
    assert win._fit_thread is None
    assert win._fit_cancel_requested is False
    assert win.fit_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


def test_fit1d_run_fit_multiple_preparation_error_finishes(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

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

    win._run_fit_multiple(2)

    assert errors
    assert win._fit_multi_total is None
    assert win._fit_thread is None
    assert win._fit_cancel_requested is False
    assert win.fit_multi_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


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


def test_fit1d_completion_action_requires_callbacks(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _Thread:
        _outcome = fit1d._FitWorkerOutcome("success", result=xr.Dataset())

    with pytest.raises(RuntimeError, match="callbacks are unavailable"):
        win._fit_worker_completion_action(_Thread())  # type: ignore[arg-type]


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


def test_fit1d_cancel_fit_without_thread_restores_idle_state(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    win._set_fit_running(True, multi=False)
    win._fit_cancel_requested = True

    assert win._cancel_fit()

    assert win._fit_cancel_requested is False
    assert win.fit_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


def test_fit1d_cancel_fit_without_wait_leaves_worker_for_polling(qtbot) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)

    class _Thread:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

    thread = _Thread()
    win._fit_thread = thread  # type: ignore[assignment]
    try:
        assert win._cancel_fit()

        assert thread.cancel_called
        assert win._fit_thread is thread
        assert win._fit_cancel_requested
        assert not win.cancel_fit_button.isEnabled()
    finally:
        win._fit_thread = None
        win._fit_cancel_requested = False
        win._fit_cancelled()


def test_fit1d_cancel_fit_waits_for_thread(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.join_timeout: float | None = None

        def cancel(self) -> None:
            self.cancel_called = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

        def is_alive(self) -> bool:
            return False

    dummy_thread = _DummyThread()
    win._fit_thread = dummy_thread  # type: ignore[assignment]

    assert win._cancel_fit(wait=True)
    assert dummy_thread.cancel_called
    assert dummy_thread.join_timeout == 5.0


def test_fit1d_cancel_fit_waits_without_timeout(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.join_timeout: float | None = 1.0

        def cancel(self) -> None:
            self.cancel_called = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

        def is_alive(self) -> bool:
            return False

    dummy_thread = _DummyThread()
    win._fit_thread = dummy_thread  # type: ignore[assignment]

    assert win._cancel_fit(wait=True, timeout_ms=None)
    assert dummy_thread.cancel_called
    assert dummy_thread.join_timeout is None


def test_fit1d_close_event_ignored_if_thread_does_not_stop(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _StuckThread:
        def __init__(self) -> None:
            self.cancel_called = False
            self.running = True

        def cancel(self) -> None:
            self.cancel_called = True

        def is_alive(self) -> bool:
            return self.running

        def join(self, timeout: float | None = None) -> None:
            return None

    stuck_thread = _StuckThread()
    win._fit_thread = stuck_thread  # type: ignore[assignment]
    fit1d._register_running_fit_worker(stuck_thread, win)  # type: ignore[arg-type]

    try:
        event = QtGui.QCloseEvent()
        assert event.isAccepted()
        win.closeEvent(event)
        assert not event.isAccepted()
        assert stuck_thread.cancel_called
        assert fit1d._running_fit_workers.get(stuck_thread) is win
    finally:
        fit1d._release_running_fit_worker(stuck_thread)  # type: ignore[arg-type]
        stuck_thread.running = False
        win._fit_thread = None
        win.close()


def test_fit1d_app_quit_waits_for_worker_completion(qtbot) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    win._stop_fit_before_app_quit()

    class _Thread:
        def __init__(self) -> None:
            self.cancelled = False
            self.join_timeout: float | None = 1.0

        def cancel(self) -> None:
            self.cancelled = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

        def is_alive(self) -> bool:
            return False

    thread = _Thread()
    win._fit_thread = thread  # type: ignore[assignment]
    fit1d._register_running_fit_worker(thread, win)  # type: ignore[arg-type]
    try:
        win._stop_fit_before_app_quit()

        assert thread.cancelled
        assert thread.join_timeout is None
        assert win._fit_thread is None
        assert thread not in fit1d._running_fit_workers
    finally:
        fit1d._release_running_fit_worker(thread)  # type: ignore[arg-type]
        win._fit_thread = None


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


def test_fit_worker_honors_cancel_before_run(qtbot, exp_decay_model) -> None:
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

    worker.cancel()
    worker.run()
    assert worker._outcome.kind == "cancelled"


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

    def _modelfit_cancel(*_args, **kwargs):
        worker._cancel.set()
        kwargs["iter_cb"]()
        raise RuntimeError("cancel")

    def _modelfit_timeout(*_args, **kwargs):
        kwargs["iter_cb"]()
        raise RuntimeError("timeout")

    worker._cancel.set()
    monkeypatch.setattr(data.xlm, "modelfit", _modelfit_cancel)
    worker.run()
    assert worker._outcome.kind == "cancelled"

    worker = fit1d._FitWorker(
        data,
        "t",
        exp_decay_model,
        params,
        max_nfev=5,
        method="least_squares",
        timeout=1.0,
    )
    timer = iter([0.0, 2.0])
    monkeypatch.setattr(fit1d.time, "perf_counter", lambda: next(timer))
    monkeypatch.setattr(data.xlm, "modelfit", _modelfit_timeout)
    worker.run()
    assert worker._outcome.kind == "timed_out"


def test_fit_worker_records_loaded_result(qtbot, exp_decay_model, monkeypatch) -> None:
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

    monkeypatch.setattr(data.xlm, "modelfit", _modelfit)

    worker.run()

    assert dummy.loaded
    assert worker._outcome.result is dummy


@pytest.mark.parametrize("outcome", ["error", "cancelled", "timed_out"])
def test_fit_worker_records_each_terminal_outcome(
    qtbot, exp_decay_model, monkeypatch, outcome
) -> None:
    data = _make_1d_data()
    worker = fit1d._FitWorker(
        data,
        "x",
        exp_decay_model,
        exp_decay_model.make_params(n0=1.0, tau=1.0),
        max_nfev=5,
        method="least_squares",
        timeout=1.0,
    )

    class _Result:
        def load(self):
            return self

    def _modelfit(*_args, **kwargs):
        if outcome == "error":
            raise RuntimeError("fit failed")
        if outcome == "cancelled":
            worker._cancel.set()
        kwargs["iter_cb"]()
        return _Result()

    if outcome == "timed_out":
        elapsed = iter([0.0, 2.0])
        monkeypatch.setattr(fit1d.time, "perf_counter", lambda: next(elapsed))
    monkeypatch.setattr(data.xlm, "modelfit", _modelfit)

    worker.run()

    assert worker._outcome.kind == outcome
    assert (worker._outcome.error is not None) is (outcome == "error")


def test_running_fit_worker_registry_tracks_concurrent_workers(
    qtbot, exp_decay_model
) -> None:
    data = _make_1d_data()
    params = exp_decay_model.make_params(n0=1.0, tau=1.0)
    workers = [
        fit1d._FitWorker(
            data,
            "x",
            exp_decay_model,
            params,
            max_nfev=5,
            method="least_squares",
            timeout=1.0,
        )
        for _ in range(2)
    ]
    owners = [QtWidgets.QWidget(), QtWidgets.QWidget()]
    for owner in owners:
        qtbot.addWidget(owner)

    try:
        assert not fit1d._running_fit_workers
        fit1d._register_running_fit_worker(workers[0], owners[0])
        fit1d._register_running_fit_worker(workers[0], owners[0])
        fit1d._register_running_fit_worker(workers[1], owners[1])
        assert fit1d._running_fit_workers == dict(zip(workers, owners, strict=True))

        fit1d._release_running_fit_worker(workers[0])
        fit1d._release_running_fit_worker(workers[0])
        assert fit1d._running_fit_workers == {workers[1]: owners[1]}

        fit1d._release_running_fit_worker(workers[1])
        assert not fit1d._running_fit_workers
    finally:
        for worker in workers:
            fit1d._release_running_fit_worker(worker)


def test_fit_worker_is_registered_before_python_thread_start(
    qtbot, monkeypatch
) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    start_state: list[QtWidgets.QWidget | None] = []
    monkeypatch.setattr(win, "_show_error", lambda *_args, **_kwargs: None)

    def _start(thread) -> None:
        start_state.append(fit1d._running_fit_workers.get(thread))

    monkeypatch.setattr(threading.Thread, "start", _start)
    try:
        assert win._run_fit()
        thread = win._fit_thread
        assert thread is not None
        assert start_state == [win]
        assert fit1d._running_fit_workers == {thread: win}

        win._fit_cancel_requested = True
        win._finalize_fit_thread(thread)
        assert not fit1d._running_fit_workers
    finally:
        win._fit_poll_timer.stop()
        if win._fit_thread is not None:
            fit1d._release_running_fit_worker(win._fit_thread)
            win._fit_thread = None


def test_fit_worker_completion_is_finalized_on_gui_thread(qtbot, monkeypatch) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    win._poll_fit_worker()
    completion_threads: list[QtCore.QThread] = []
    original_completion_action = win._fit_worker_completion_action

    def _completion_action(thread):
        completion_threads.append(QtCore.QThread.currentThread())
        return original_completion_action(thread)

    monkeypatch.setattr(win, "_fit_worker_completion_action", _completion_action)

    assert win._run_fit()
    qtbot.waitUntil(lambda: win._fit_thread is None, timeout=5000)

    assert completion_threads == [QtWidgets.QApplication.instance().thread()]


@pytest.mark.parametrize("alive", [True, False])
def test_fit_worker_poll_uses_current_worker(qtbot, alive) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)
    results: list[xr.Dataset] = []

    class _DummyThread:
        def __init__(self) -> None:
            self._outcome = fit1d._FitWorkerOutcome("success", result=xr.Dataset())
            self.joined = False

        def is_alive(self) -> bool:
            return alive

        def join(self) -> None:
            self.joined = True

    thread = _DummyThread()
    win._fit_thread = thread  # type: ignore[assignment]
    win._fit_worker_callbacks[thread] = fit1d._FitWorkerCallbacks(
        on_success=results.append,
        on_timeout=lambda: None,
        on_error=lambda _message: None,
    )
    fit1d._register_running_fit_worker(thread, win)  # type: ignore[arg-type]
    try:
        win._fit_poll_timer.start()
        win._poll_fit_worker()

        assert len(results) == int(not alive)
        assert thread.joined is (not alive)
        assert (win._fit_thread is None) is (not alive)
        assert (thread not in fit1d._running_fit_workers) is (not alive)
        assert win._fit_poll_timer.isActive() is alive
    finally:
        win._fit_poll_timer.stop()
        fit1d._release_running_fit_worker(thread)  # type: ignore[arg-type]
        win._fit_thread = None


@pytest.mark.parametrize("outcome", ["success", "error", "cancelled", "timed_out"])
def test_fit_worker_outcomes_dispatch_on_gui_thread(
    qtbot, monkeypatch, outcome
) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)
    worker_ready = threading.Event()
    worker_holder: dict[str, fit1d._FitWorker] = {}
    events: list[tuple[str, QtCore.QThread]] = []

    class _Result:
        def load(self):
            return self

    def _modelfit(*_args, **kwargs):
        assert worker_ready.wait(timeout=5)
        worker = worker_holder["worker"]
        if outcome == "error":
            raise RuntimeError("fit failed")
        if outcome == "cancelled":
            worker.cancel()
        elif outcome == "timed_out":
            worker._timeout = 1e-9
        kwargs["iter_cb"]()
        return _Result()

    def _record_event(name: str) -> None:
        events.append((name, QtCore.QThread.currentThread()))

    monkeypatch.setattr(data.xlm, "modelfit", _modelfit)
    monkeypatch.setattr(win, "_fit_cancelled", lambda: _record_event("cancelled"))

    assert win._start_fit_worker(
        data,
        win._params,
        multi=False,
        on_success=lambda _result: _record_event("success"),
        on_timeout=lambda: _record_event("timed_out"),
        on_error=lambda _message: _record_event("error"),
    )
    worker = win._fit_thread
    assert worker is not None
    worker_holder["worker"] = worker
    worker_ready.set()

    qtbot.waitUntil(lambda: win._fit_thread is None, timeout=5000)

    assert events == [(outcome, QtWidgets.QApplication.instance().thread())]


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (fit1d._FitWorkerOutcome("success"), "returned no result"),
        (fit1d._FitWorkerOutcome("error"), "returned no error details"),
        (fit1d._FitWorkerOutcome("not_finished"), "without a terminal outcome"),
    ],
)
def test_fit_worker_completion_rejects_incomplete_outcome(
    qtbot, outcome, message
) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)

    class _Thread:
        _outcome = outcome

    thread = _Thread()
    win._fit_worker_callbacks[thread] = fit1d._FitWorkerCallbacks(
        on_success=lambda _result: None,
        on_timeout=lambda: None,
        on_error=lambda _message: None,
    )

    with pytest.raises(RuntimeError, match=message):
        win._fit_worker_completion_action(thread)  # type: ignore[arg-type]


def test_fit1d_finalize_fit_thread_cancelled_releases_thread(qtbot) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self._outcome = fit1d._FitWorkerOutcome("success", result=xr.Dataset())

    thread = _DummyThread()
    cancelled = {"value": False}

    win._fit_thread = thread  # type: ignore[assignment]
    win._fit_cancel_requested = True
    win._fit_cancelled = lambda: cancelled.__setitem__("value", True)  # type: ignore[method-assign]
    fit1d._register_running_fit_worker(thread, win)  # type: ignore[arg-type]

    try:
        win._finalize_fit_thread(thread)  # type: ignore[arg-type]
    finally:
        fit1d._release_running_fit_worker(thread)  # type: ignore[arg-type]

    assert cancelled["value"]
    assert win._fit_thread is None
    assert not win._fit_cancel_requested


def test_fit1d_sequential_workers_keep_owner_rooted(qtbot) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self._outcome = fit1d._FitWorkerOutcome("success", result=xr.Dataset())

    first = _DummyThread()
    second = _DummyThread()
    fit1d._register_running_fit_worker(first, win)  # type: ignore[arg-type]

    def _start_second(_result) -> None:
        win._fit_thread = second  # type: ignore[assignment]
        fit1d._register_running_fit_worker(second, win)  # type: ignore[arg-type]

    win._fit_thread = first  # type: ignore[assignment]
    win._fit_worker_callbacks[first] = fit1d._FitWorkerCallbacks(
        on_success=_start_second,
        on_timeout=lambda: None,
        on_error=lambda _message: None,
    )
    try:
        win._finalize_fit_thread(first)  # type: ignore[arg-type]

        assert win._fit_thread is second
        assert fit1d._running_fit_workers == {second: win}
    finally:
        fit1d._release_running_fit_worker(first)  # type: ignore[arg-type]
        fit1d._release_running_fit_worker(second)  # type: ignore[arg-type]
        win._fit_thread = None


def test_fit_worker_survives_forced_collection_while_running(
    qtbot, monkeypatch
) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    worker_started = threading.Event()
    release_worker = threading.Event()
    outcomes: list[xr.Dataset] = []
    execution_threads: list[threading.Thread] = []

    def _modelfit(*_args, **_kwargs) -> xr.Dataset:
        execution_threads.append(threading.current_thread())
        worker_started.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("the test did not release the fit worker")
        return xr.Dataset()

    monkeypatch.setattr(data.xlm, "modelfit", _modelfit)

    assert win._start_fit_worker(
        data,
        win._params,
        multi=False,
        on_success=outcomes.append,
        on_timeout=lambda: None,
        on_error=lambda _message: None,
    )
    thread = win._fit_thread
    assert thread is not None
    assert isinstance(thread, threading.Thread)
    assert not isinstance(thread, QtCore.QThread)
    owner_ref = weakref.ref(win)
    thread_ref = weakref.ref(thread)
    del thread
    del win
    try:
        qtbot.waitUntil(worker_started.is_set, timeout=5000)
        assert execution_threads == [thread_ref()]
        for _ in range(3):
            gc.collect()
        retained_owner = owner_ref()
        retained_thread = thread_ref()
        assert retained_owner is not None
        assert retained_thread is not None
        assert fit1d._running_fit_workers == {retained_thread: retained_owner}
        del retained_owner
        del retained_thread

        release_worker.set()
        qtbot.waitUntil(
            lambda: (
                thread_ref() is None or thread_ref() not in fit1d._running_fit_workers
            ),
            timeout=5000,
        )
        assert len(outcomes) == 1
        xr.testing.assert_identical(outcomes[0], xr.Dataset())
    finally:
        release_worker.set()
        retained_owner = owner_ref()
        if retained_owner is not None:
            if retained_owner._fit_thread is not None:
                retained_owner._cancel_fit(wait=True, timeout_ms=None)
            retained_owner.close()


def test_fit1d_finalize_stale_and_idle_threads(qtbot) -> None:
    win = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self._outcome = fit1d._FitWorkerOutcome("success", result=xr.Dataset())

    current = _DummyThread()
    stale = _DummyThread()
    win._fit_thread = current  # type: ignore[assignment]
    fit1d._register_running_fit_worker(stale, win)  # type: ignore[arg-type]
    try:
        win._finalize_fit_thread(stale)  # type: ignore[arg-type]
        assert win._fit_thread is current
        assert stale not in fit1d._running_fit_workers

        fit1d._register_running_fit_worker(current, win)  # type: ignore[arg-type]
        win._fit_worker_callbacks[current] = fit1d._FitWorkerCallbacks(
            on_success=lambda _result: None,
            on_timeout=lambda: None,
            on_error=lambda _message: None,
        )
        win._fit_cancel_requested = False
        win._finalize_fit_thread(current)  # type: ignore[arg-type]
        assert win._fit_thread is None
        assert current not in fit1d._running_fit_workers
    finally:
        fit1d._release_running_fit_worker(stale)  # type: ignore[arg-type]
        fit1d._release_running_fit_worker(current)  # type: ignore[arg-type]
        win._fit_thread = None


def test_fit1d_finalize_fit_thread_action_error_restores_idle(
    qtbot, monkeypatch
) -> None:
    data = _make_1d_data()
    win = erlab.interactive.ftool(data, execute=False)
    qtbot.addWidget(win)

    class _DummyThread:
        def __init__(self) -> None:
            self._outcome = fit1d._FitWorkerOutcome("success", result=xr.Dataset())

    errors: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        win,
        "_show_error",
        lambda title, text, detailed_text=None: errors.append(
            (title, text, detailed_text)
        ),
    )

    def _raise_action(_result) -> None:
        raise RuntimeError("post-processing failed")

    thread = _DummyThread()
    win._set_fit_running(True, multi=False)
    win._fit_thread = thread  # type: ignore[assignment]
    win._fit_worker_callbacks[thread] = fit1d._FitWorkerCallbacks(
        on_success=_raise_action,
        on_timeout=lambda: None,
        on_error=lambda _message: None,
    )
    win._fit_cancel_requested = False

    win._finalize_fit_thread(thread)  # type: ignore[arg-type]

    assert errors
    assert win._fit_thread is None
    assert win._fit_cancel_requested is False
    assert win.fit_button.isEnabled()
    assert not win.cancel_fit_button.isEnabled()


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


def test_fit1d_trust_revocation_blocks_expression_execution(qtbot, monkeypatch) -> None:
    tool, _data, _model, _params = _make_linear_fit1d_tool(qtbot, expression=True)
    tool.set_document_trust(untrusted_document_trust())
    assert tool._params["intercept"].expr == "2 * slope"

    def fail_if_executed(*_args, **_kwargs):
        raise AssertionError("untrusted fit code was executed")

    monkeypatch.setattr(tool._model, "eval", fail_if_executed)
    monkeypatch.setattr(tool._model, "guess", fail_if_executed)
    monkeypatch.setattr(fit1d, "_FitWorker", fail_if_executed)
    monkeypatch.setattr(lmfit.Parameter, "_getval", fail_if_executed)

    value_index = tool.param_model.index(
        tool.param_model._param_names.index("intercept"), 1
    )
    assert tool.param_model.data(value_index) == "2"
    tool._update_fit_curve()
    tool._guess_params()
    assert not tool._run_fit()


def test_fit1d_untrusted_host_attachment_clears_cached_capability(qtbot) -> None:
    tool, _data, _model, _params = _make_linear_fit1d_tool(qtbot, expression=True)
    assert tool._current_fit_execution_allowed()
    assert tool._fit_execution_capability is not None

    host_trust = untrusted_document_trust()
    tool._set_code_trust_host(
        lambda *_args, **_kwargs: None,
        local_edit_context=lambda *_args, **_kwargs: contextlib.nullcontext(None),
        state_getter=lambda: host_trust,
    )

    assert tool._fit_execution_capability is None
    assert not tool._current_fit_execution_allowed()


def test_fit1d_result_deserialization_waits_for_approval(qtbot, monkeypatch) -> None:
    tool, data, _safe_model, _safe_params = _make_linear_fit1d_tool(qtbot)

    unsafe_model = lmfit.models.ExpressionModel("slope * x + intercept")
    unsafe_params = unsafe_model.make_params(slope=1.0, intercept=0.0)
    result_ds = data.xlm.modelfit("x", model=unsafe_model, params=unsafe_params).load()
    blob = erlab.interactive.utils._serialize_fit_dataset_blob(result_ds)
    deserialize_calls: list[np.ndarray] = []
    original_deserialize = erlab.interactive.utils._deserialize_fit_dataset_blob

    def tracked_deserialize(payload):
        deserialize_calls.append(np.asarray(payload))
        return original_deserialize(payload)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "_deserialize_fit_dataset_blob",
        tracked_deserialize,
    )

    tool.set_document_trust(untrusted_document_trust())
    tool._restore_persisted_fit_result_blob(blob, fit_is_current=True)

    assert deserialize_calls == []
    assert tool._last_result_ds is None
    assert np.array_equal(tool._serialized_fit_result_blob, blob)
    assert tool._pending_persisted_fit_is_current is True

    tool.set_document_trust(new_document_trust())

    assert len(deserialize_calls) == 1
    assert tool._pending_persisted_fit_is_current is None
    assert tool._last_result_ds is not None
    _assert_fit_result_dataset_equivalent(
        tool._last_result_ds,
        result_ds,
        require_model_type=False,
    )


def test_fit1d_safe_result_can_serialize_in_untrusted_document(
    qtbot, monkeypatch
) -> None:
    tool, data, model, params = _make_linear_fit1d_tool(qtbot)
    tool._last_result_ds = data.xlm.modelfit("x", model=model, params=params).load()
    tool._fit_is_current = True
    tool._cache_fit_result_payload()
    assert not tuple(tool._code_trust_payload_entries())

    tool.set_document_trust(untrusted_document_trust())
    tool._serialized_fit_result_blob = None
    original_serialize = erlab.interactive.utils._serialize_fit_dataset_blob
    serialize_calls: list[xr.Dataset] = []

    def tracked_serialize(dataset: xr.Dataset) -> np.ndarray:
        serialize_calls.append(dataset)
        return original_serialize(dataset)

    monkeypatch.setattr(
        erlab.interactive.utils,
        "_serialize_fit_dataset_blob",
        tracked_serialize,
    )

    saved = tool.to_dataset()

    assert len(serialize_calls) == 1
    assert tool._PERSISTED_FIT_RESULT_VAR in saved


def test_fit1d_local_code_edits_use_one_scoped_capability(qtbot, monkeypatch) -> None:
    model = lmfit.models.ExpressionModel("a * x + b")
    tool = erlab.interactive.ftool(
        _make_1d_data(),
        model=model,
        params=model.make_params(a=1.0, b=0.0),
        execute=False,
    )
    qtbot.addWidget(tool)
    _set_signed_fit_trust(tool)
    tool.expr_edit.setPlainText("scale * x + offset")
    tool.expr_init_script_dialog.text_edit.setPlainText("constant = 1")

    original_init = lmfit.models.ExpressionModel.__init__
    boundary_events: list[str] = []
    edit_scope_active = False
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

    def tracked_init(expression_model, *args, **kwargs):
        assert edit_scope_active
        boundary_events.append("model")
        original_init(expression_model, *args, **kwargs)

    monkeypatch.setattr(tool, "_local_code_edit", tracked_local_code_edit)
    monkeypatch.setattr(lmfit.models.ExpressionModel, "__init__", tracked_init)
    expression_index = tool.model_combo.findData("ExpressionModel")
    with QtCore.QSignalBlocker(tool.model_combo):
        tool.model_combo.setCurrentIndex(expression_index)

    tool._on_model_choice_changed(expression_index)

    assert boundary_events == ["model"]
    assert isinstance(tool._model, lmfit.models.ExpressionModel)
    assert tool._model.expr == "scale * x + offset"

    _set_signed_fit_trust(tool)
    tool.expr_edit.setPlainText("2 * scale * x + offset")
    tool._refresh_expression_model()

    assert boundary_events == ["model", "model"]
    assert tool._model.expr == "2 * scale * x + offset"

    _set_signed_fit_trust(tool)
    original_validate = tool._validate_param_expr

    def tracked_validate(param, expression):
        assert edit_scope_active
        boundary_events.append("parameter-edit")
        return original_validate(param, expression)

    monkeypatch.setattr(tool, "_validate_param_expr", tracked_validate)
    tool._set_param_expr(tool.param_model._param_names.index("offset"), "2 * scale")

    assert boundary_events == ["model", "model", "parameter-edit"]
    assert tool._params["offset"].expr == "2 * scale"
    assert tool._current_fit_execution_allowed()


def test_fit1d_first_local_code_edit_promotes_code_free_external_document(
    qtbot,
) -> None:
    model = lmfit.models.LinearModel()
    tool = erlab.interactive.ftool(
        _make_1d_data(), model=model, params=model.make_params(), execute=False
    )
    qtbot.addWidget(tool)
    tool.set_document_trust(external_document_trust(None), notify=False)
    assert not _trust_allows_local_code_edit(tool._document_trust)

    tool.expr_edit.setPlainText("a * x + b")
    tool._refresh_expression_model()

    assert isinstance(tool._model, lmfit.models.ExpressionModel)
    assert tool._model.expr == "a * x + b"
    assert _trust_allows_local_code_edit(tool._document_trust)


def test_fit1d_model_option_edit_preserves_trusted_lineage(qtbot, monkeypatch) -> None:
    tool = erlab.interactive.ftool(_make_1d_data(), execute=False)
    qtbot.addWidget(tool)
    _set_signed_fit_trust(tool)
    npeaks = tool.npeaks_spin.value() + 1
    redraws = 0
    original_update = tool._update_fit_curve

    def tracked_update() -> None:
        nonlocal redraws
        redraws += 1
        original_update()

    monkeypatch.setattr(tool, "_update_fit_curve", tracked_update)

    tool.npeaks_spin.setValue(npeaks)

    assert tool._model.func.npeaks == npeaks
    assert _trust_allows_local_code_edit(tool._document_trust)
    assert redraws == 1


def test_fit1d_stored_status_rejects_partial_host_capability(
    qtbot, monkeypatch
) -> None:
    model = lmfit.models.ExpressionModel("a * x + b")
    params = model.make_params(a=1.0, b=0.0)
    params["b"].expr = "2 * a"
    tool = erlab.interactive.ftool(
        _make_1d_data(), model=model, params=params, execute=False
    )
    qtbot.addWidget(tool)
    entries = tool._fit_code_entries
    assert len(entries) >= 2
    trust = untrusted_document_trust()
    _prospective, partial_capability = issue_local_edit_capability(
        trust,
        entries,
        edited_entries=(entries[0],),
    )
    assert execution_capability_allows(partial_capability, (entries[0],))
    assert not execution_capability_allows(partial_capability, entries)
    tool._set_code_trust_host(
        lambda *_args, **_kwargs: partial_capability,
        local_edit_context=lambda *_args, **_kwargs: contextlib.nullcontext(
            partial_capability
        ),
        state_getter=lambda: trust,
    )
    status = tool.tool_status
    monkeypatch.setattr(
        fit1d,
        "_load_lmfit_for_ftool_restore",
        lambda *_args, **_kwargs: pytest.fail(
            "partially authorized status was decoded"
        ),
    )

    tool._restoring_from_dataset = True
    try:
        tool.tool_status = status
    finally:
        tool._restoring_from_dataset = False

    assert tool._pending_fit_status is status
    assert not tool._current_fit_execution_allowed()


def test_fit1d_local_edit_does_not_authorize_untrusted_mixed_content(
    qtbot, monkeypatch
) -> None:
    model = lmfit.models.ExpressionModel("a * x + b")
    params = model.make_params(a=1.0, b=0.0)
    params["b"].expr = "2 * a"
    tool = erlab.interactive.ftool(
        _make_1d_data(), model=model, params=params, execute=False
    )
    qtbot.addWidget(tool)
    tool.set_document_trust(untrusted_document_trust())
    original_expression = tool._model.expr
    tool.expr_edit.setPlainText("3 * a * x + b")

    monkeypatch.setattr(
        lmfit.models.ExpressionModel,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("blocked candidate was evaluated"),
    )

    tool._refresh_expression_model()

    assert tool._model.expr == original_expression
    assert not document_trust_has_trusted_lineage(tool._document_trust)


def test_fit1d_failed_local_expression_validation_rolls_back_signed_trust(
    qtbot, monkeypatch
) -> None:
    model = lmfit.models.ExpressionModel("a * x + b")
    tool = erlab.interactive.ftool(
        _make_1d_data(),
        model=model,
        params=model.make_params(a=1.0, b=0.0),
        execute=False,
    )
    qtbot.addWidget(tool)
    _set_signed_fit_trust(tool)
    assert not _trust_allows_local_code_edit(tool._document_trust)
    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(tool, "_show_warning", lambda *args: warnings.append(args))

    tool._set_param_expr(tool.param_model._param_names.index("b"), "missing +")

    assert warnings
    assert tool._params["b"].expr is None
    assert not _trust_allows_local_code_edit(tool._document_trust)


def test_fit1d_clear_parameter_expression_is_local_edit(qtbot) -> None:
    tool, _data, _model, _params = _make_linear_fit1d_tool(qtbot, expression=True)
    _set_signed_fit_trust(tool)
    assert not _trust_allows_local_code_edit(tool._document_trust)

    tool._clear_param_expr(tool.param_model._param_names.index("intercept"))

    assert tool._params["intercept"].expr is None
    assert _trust_allows_local_code_edit(tool._document_trust)


def test_fit1d_status_parameter_expression_change_is_local_edit(qtbot) -> None:
    tool, _data, _model, _params = _make_linear_fit1d_tool(qtbot, expression=True)
    status = tool.tool_status
    params = list(status.params)
    intercept_index = next(
        index for index, state in enumerate(params) if state[0] == "intercept"
    )
    intercept = list(params[intercept_index])
    intercept[3] = "3 * slope"
    params[intercept_index] = tuple(intercept)
    changed = status.model_copy(update={"params": params})
    _set_signed_fit_trust(tool)

    tool.tool_status = changed

    assert tool._params["intercept"].expr == "3 * slope"
    assert _trust_allows_local_code_edit(tool._document_trust)


def test_fit1d_expression_model_code_generation_does_not_run_init_script(qtbot) -> None:
    model = lmfit.models.ExpressionModel("a * x")
    tool = erlab.interactive.ftool(
        _make_1d_data(),
        model=model,
        params=model.make_params(a=1.0),
        execute=False,
    )
    qtbot.addWidget(tool)
    tool.expr_init_script_dialog.text_edit.setPlainText(
        "raise RuntimeError('generated code must not run while copying')"
    )

    _data_name, _model_name, lines = tool._make_model_code("data")

    assert lines


def test_fit1d_numeric_fit_result_reuses_code_inventory(qtbot, monkeypatch) -> None:
    data = _make_1d_data()
    model = lmfit.models.ExpressionModel("a * x + b")
    params = model.make_params(a=1.0, b=0.0)
    tool = erlab.interactive.ftool(data, model=model, params=params, execute=False)
    qtbot.addWidget(tool)
    _set_signed_fit_trust(tool)
    admitted_entries = tool._fit_code_entries
    result_ds = data.xlm.modelfit("x", model=model, params=params, max_nfev=1).load()
    invalidation_calls = 0
    original_invalidate = tool._invalidate_fit_result_payload

    def tracked_invalidate() -> None:
        nonlocal invalidation_calls
        invalidation_calls += 1
        original_invalidate()

    monkeypatch.setattr(tool, "_invalidate_fit_result_payload", tracked_invalidate)

    tool._set_fit_ds(result_ds, fit1d.time.perf_counter())

    assert invalidation_calls == 1
    assert tool._fit_code_entries is admitted_entries
    assert tool._current_fit_execution_allowed()


def test_fit1d_source_replacement_discards_pending_result_retry(qtbot) -> None:
    tool, data, _safe_model, _safe_params = _make_linear_fit1d_tool(qtbot)
    unsafe_model = lmfit.models.ExpressionModel("a * x + b")
    unsafe_result = data.xlm.modelfit(
        "x",
        model=unsafe_model,
        params=unsafe_model.make_params(a=1.0, b=0.0),
        max_nfev=1,
    ).load()
    blob = erlab.interactive.utils._serialize_fit_dataset_blob(unsafe_result)

    tool.set_document_trust(untrusted_document_trust(), notify=False)

    tool._restore_persisted_fit_result_blob(blob, fit_is_current=True)
    assert np.array_equal(tool._serialized_fit_result_blob, blob)
    assert tool._pending_persisted_fit_is_current is True
    assert tool._last_result_ds is None

    assert tool.update_inputs({"data": data + 1.0})
    tool.set_document_trust(new_document_trust())

    assert tool._serialized_fit_result_blob is None
    assert tool._pending_persisted_fit_is_current is None
    assert tool._last_result_ds is None
    assert tool._PERSISTED_FIT_RESULT_VAR not in tool.to_dataset()
