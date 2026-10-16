import json

import lmfit
import lmfit.jsonutils
import numpy as np
import pytest

from erlab.analysis.fit import models
from erlab.analysis.fit._serialization import (
    dumps_model,
    dumps_result,
    inspect_descriptor,
    inspect_model_reference,
    loads_model,
    loads_result,
)


def _assert_parameters(actual, expected) -> None:
    assert list(actual) == list(expected)
    for name, param in actual.items():
        other = expected[name]
        assert (param.min, param.max, param.vary, param.expr) == (
            other.min,
            other.max,
            other.vary,
            other.expr,
        )
        np.testing.assert_allclose(param.value, other.value, equal_nan=True)


@pytest.mark.parametrize(
    "model",
    [
        models.MultiPeakModel(2, ["voigt", "gaussian"], prefix="p_"),
        models.MultiPeakModel(
            21,
            fd=False,
            background="polynomial",
            degree=11,
            oversample=65,
            segmented=True,
        ),
        models.PolynomialModel(degree=3, prefix="b_"),
        models.FermiEdge2dModel(degree=3),
        models.FermiDiracModel(),
        models.FermiEdgeModel(),
        models.StepEdgeModel(),
        models.TLLModel(),
        lmfit.models.GaussianModel(prefix="g_"),
        lmfit.models.LinearModel(prefix="l_"),
        lmfit.models.StepModel(form="erf"),
    ],
)
def test_builtin_model_roundtrip_has_no_callable_blob(model, monkeypatch) -> None:
    model.set_param_hint(
        model._param_root_names[0], value=2.0, min=-np.inf, max=np.inf, vary=False
    )
    serialized = dumps_model(model)
    item = json.loads(serialized)["value"][0]
    assert set(item["funcdef"]) == {"__erlab_builtin_model__", "config"}

    def reject_dill(_value):
        raise AssertionError("Built-in model must not decode a callable")

    monkeypatch.setattr(lmfit.jsonutils.dill, "loads", reject_dill)
    restored = loads_model(serialized)
    assert type(restored) is type(model)
    assert restored.name == model.name
    assert restored.prefix == model.prefix
    assert restored.nan_policy == model.nan_policy
    assert restored.independent_vars == model.independent_vars
    assert restored.opts == model.opts
    assert restored.param_hints == model.param_hints
    _assert_parameters(restored.make_params(), model.make_params())
    assert inspect_model_reference(json.loads(serialized)) == (
        f"{type(model).__module__}:{type(model).__qualname__}"
    )


def test_composite_and_result_roundtrip_preserves_each_builtin(monkeypatch) -> None:
    x = np.linspace(-1.0, 1.0, 31)
    first = models.MultiPeakModel(
        fd=False, convolve=False, background="none", prefix="first_"
    )
    second = models.MultiPeakModel(
        2, "gaussian", fd=False, convolve=False, background="constant", prefix="second_"
    )
    model = first + second + lmfit.models.LinearModel(prefix="b_")
    params = model.make_params()
    y = model.eval(params, x=x)
    result = model.fit(y, params, x=x, max_nfev=2)
    serialized = dumps_result(result)

    def reject_dill(_value):
        raise AssertionError("Built-in result must not decode a callable")

    monkeypatch.setattr(lmfit.jsonutils.dill, "loads", reject_dill)
    restored = loads_result(serialized)
    actual_models = [
        restored.model.left.left,
        restored.model.left.right,
        restored.model.right,
    ]
    expected_models = [model.left.left, model.left.right, model.right]
    assert [type(item) for item in actual_models] == [
        type(item) for item in expected_models
    ]
    assert actual_models[0].func.npeaks == 1
    assert actual_models[1].func.npeaks == 2
    _assert_parameters(restored.params, result.params)
    _assert_parameters(restored.init_params, result.init_params)
    np.testing.assert_allclose(restored.best_fit, result.best_fit)
    np.testing.assert_allclose(restored.init_fit, result.init_fit)
    for actual, expected in zip(actual_models, expected_models, strict=True):
        assert actual.eval_components(params=restored.params, x=x).keys() == (
            expected.eval_components(params=result.params, x=x).keys()
        )
    assert (
        inspect_model_reference(json.loads(serialized)) == "lmfit.model:CompositeModel"
    )
    assert "<erlab builtin " not in dumps_result(restored)


def test_descriptor_inspection_does_not_evaluate_saved_hints() -> None:
    model = models.MultiPeakModel(prefix="test_")
    item = json.loads(dumps_model(model))["value"][0]
    item["param_hints"]["p0_center"] = {"expr": "missing_user_function()"}
    builtin = inspect_descriptor(item)
    assert type(builtin) is models.MultiPeakModel
    assert builtin.prefix == "test_"
    assert "expr" not in builtin.param_hints.get("p0_center", {})


def test_custom_model_and_modified_builtin_keep_native_serialization() -> None:
    class CustomModel(models.MultiPeakModel):
        pass

    custom = CustomModel()
    modified = models.MultiPeakModel()
    modified.func._peak_funcs = [lambda x: x]
    for model in (custom, modified, lmfit.models.ExpressionModel("a*x")):
        serialized = dumps_model(model)
        assert "__erlab_builtin_model__" not in serialized
        assert inspect_model_reference(json.loads(serialized)) is None


def test_released_model_normalizes_after_authorized_load() -> None:
    model = models.MultiPeakModel(fd=False, convolve=False, background="none")
    restored = loads_model(model.dumps())
    assert type(restored) is models.MultiPeakModel
    assert "__erlab_builtin_model__" in dumps_model(restored)


@pytest.mark.parametrize(
    "change", ["unknown-model", "callable-options", "bad-config", "parameter-options"]
)
def test_invalid_descriptor_does_not_load_callable(change, monkeypatch) -> None:
    model = (
        lmfit.models.GaussianModel()
        if change == "parameter-options"
        else models.MultiPeakModel()
    )
    item = json.loads(dumps_model(model))["value"][0]
    if change == "unknown-model":
        item["funcdef"]["__erlab_builtin_model__"] = "saved.module:CustomModel"
    elif change == "callable-options":
        item["opts"] = {"__class__": "Callable", "value": "untrusted"}
    elif change == "bad-config":
        item["funcdef"]["config"]["npeaks"] = "two"
    else:
        item["opts"] = {"__class__": "Dict", "sigma": 1.0}

    def reject_dill(_value):
        raise AssertionError("Invalid descriptors must not decode a callable")

    monkeypatch.setattr(lmfit.jsonutils.dill, "loads", reject_dill)
    with pytest.raises((TypeError, ValueError)):
        inspect_descriptor(item)


@pytest.mark.parametrize("result_payload", [False, True])
def test_authorized_normalization_preserves_custom_leaf_bytes(
    result_payload, monkeypatch
) -> None:
    model = models.MultiPeakModel(
        fd=False, convolve=False, background="none", prefix="peak_"
    ) + lmfit.Model(lambda x, scale: scale * x, prefix="custom_")
    if result_payload:
        x = np.linspace(-1.0, 1.0, 11)
        params = model.make_params(custom_scale=2.0)
        source = model.fit(model.eval(params, x=x), params, x=x, max_nfev=2)
        load, dump = loads_result, dumps_result
    else:
        source = model
        load, dump = loads_model, dumps_model
    original = source.dumps()
    restored = load(original)

    def reject_redump():
        raise AssertionError("Authorized custom callable must retain its saved bytes")

    monkeypatch.setattr(restored, "dumps", reject_redump)
    normalized = dump(restored, serialized=original)
    original_state = json.loads(original)
    normalized_state = json.loads(normalized)
    if result_payload:
        original_state = original_state["model"]
        normalized_state = normalized_state["model"]
    assert (
        normalized_state["value"][0]["value"][0]["funcdef"].get(
            "__erlab_builtin_model__"
        )
        == "MultiPeakModel"
    )
    assert normalized_state["value"][1] == original_state["value"][1]
