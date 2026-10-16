"""Data-only storage for installed ERLab fitting models."""

from __future__ import annotations

import copy
import functools
import json
import typing

import lmfit
import lmfit.jsonutils
import numpy as np

import erlab.analysis.fit.models as fit_models
from erlab.analysis.fit.functions.dynamic import (
    FermiEdge2dFunction,
    MultiPeakFunction,
    PolynomialFunction,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

_MARKER = "__erlab_builtin_model__"
_MODEL_CLASSES = {
    name: getattr(fit_models, name)
    for name in (
        "BCSGapModel",
        "DynesModel",
        "FermiDiracModel",
        "FermiEdge2dModel",
        "FermiEdgeModel",
        "MultiPeakModel",
        "PolynomialModel",
        "StepEdgeModel",
        "SymmetrizedGapModel",
        "TLLModel",
    )
}
_MODEL_CLASSES.update(
    (f"lmfit.{name}", getattr(lmfit.models, name))
    for name in (
        "ConstantModel",
        "ComplexConstantModel",
        "LinearModel",
        "QuadraticModel",
        "SineModel",
        "GaussianModel",
        "Gaussian2dModel",
        "LorentzianModel",
        "SplitLorentzianModel",
        "VoigtModel",
        "PseudoVoigtModel",
        "MoffatModel",
        "Pearson4Model",
        "Pearson7Model",
        "StudentsTModel",
        "BreitWignerModel",
        "LognormalModel",
        "DampedOscillatorModel",
        "DampedHarmonicOscillatorModel",
        "ExponentialGaussianModel",
        "SkewedGaussianModel",
        "SkewedVoigtModel",
        "ThermalDistributionModel",
        "BoseModel",
        "FermiModel",
        "DoniachModel",
        "PowerLawModel",
        "ExponentialModel",
        "StepModel",
        "RectangleModel",
    )
)
_DYNAMIC_CLASSES = (MultiPeakFunction, PolynomialFunction, FermiEdge2dFunction)


def _data_only(value: typing.Any) -> bool:
    if value is None or type(value) in (str, int, float, bool):
        return True
    if isinstance(value, list):
        return all(_data_only(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("__class__") not in {
        None,
        "Dict",
        "List",
        "Tuple",
        "NDArray",
        "Complex",
        "UFloat",
    }:
        return False
    if value.get("__class__") == "NDArray":
        try:
            if np.dtype(value["__dtype__"]).hasobject:
                return False
        except (KeyError, TypeError):
            return False
    return all(isinstance(key, str) and _data_only(item) for key, item in value.items())


def _decode_data(value: typing.Any) -> typing.Any:
    if not _data_only(value):
        raise ValueError("Built-in model metadata must contain only data")
    return lmfit.jsonutils.decode4js(copy.deepcopy(value))


def _config(name: str, func: typing.Any) -> dict[str, typing.Any]:
    if name == "MultiPeakModel":
        return {
            "npeaks": int(func.npeaks),
            "peak_shapes": list(func._peak_shapes),
            "fd": bool(func.fd),
            "background": func.background,
            "degree": int(getattr(func, "bkg_degree", 2)),
            "convolve": bool(func.convolve),
            "oversample": int(func.oversample),
            "segmented": bool(func.segmented),
        }
    if name == "PolynomialModel":
        return {"degree": int(func.degree)}
    if name == "FermiEdge2dModel":
        return {"degree": int(func.poly.degree)}
    return {}


def _validate_config(name: str, value: typing.Any) -> dict[str, typing.Any]:
    if not isinstance(value, dict):
        raise TypeError("Built-in model configuration must be a mapping")
    expected: dict[str, type] = {}
    if name == "MultiPeakModel":
        expected = {
            "npeaks": int,
            "peak_shapes": list,
            "fd": bool,
            "background": str,
            "degree": int,
            "convolve": bool,
            "oversample": int,
            "segmented": bool,
        }
    elif name in {"PolynomialModel", "FermiEdge2dModel"}:
        expected = {"degree": int}
    if value.keys() != expected.keys() or any(
        type(value[key]) is not kind for key, kind in expected.items()
    ):
        raise ValueError("Invalid built-in model configuration")
    if name == "MultiPeakModel" and (
        not all(isinstance(shape, str) for shape in value["peak_shapes"])
        or value["background"]
        not in {"none", "constant", "linear", "polynomial", "shirley"}
    ):
        raise ValueError("Invalid peak shapes or background")
    return value


def inspect_descriptor(item: typing.Any) -> lmfit.Model | None:
    """Build an installed model without applying saved parameter hints."""
    definition = item.get("funcdef") if isinstance(item, dict) else None
    if not isinstance(definition, dict) or _MARKER not in definition:
        return None
    name = definition[_MARKER]
    if (
        not isinstance(name, str)
        or name not in _MODEL_CLASSES
        or set(definition) != {_MARKER, "config"}
    ):
        raise ValueError("Invalid built-in model descriptor")
    config = _validate_config(name, definition["config"])
    metadata = {
        key: _decode_data(item.get(key))
        for key in ("name", "prefix", "nan_policy", "independent_vars", "opts")
    }
    if (
        not isinstance(metadata["name"], str)
        or not isinstance(metadata["prefix"], str)
        or metadata["nan_policy"] not in {"raise", "omit", "propagate"}
        or not isinstance(metadata["independent_vars"], list)
        or not all(isinstance(value, str) for value in metadata["independent_vars"])
        or not isinstance(metadata["opts"], dict)
        or not isinstance(item.get("param_hints"), dict)
        or any(
            key != "__class__" and not isinstance(hint, dict)
            for key, hint in item["param_hints"].items()
        )
    ):
        raise ValueError("Invalid built-in model metadata")
    options = metadata.pop("opts")
    if options.keys() & (metadata.keys() | config.keys()):
        raise ValueError("Duplicate built-in model options")
    try:
        model = _MODEL_CLASSES[name](**config, **metadata, **options)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid built-in model constructor settings") from exc
    if item.get("funcname") != model.func.__name__:
        raise ValueError("Built-in model function does not match its descriptor")
    return model


@functools.cache
def _static_functions() -> tuple[tuple[str, typing.Any], ...]:
    return tuple(
        (name, cls().func)
        for name, cls in _MODEL_CLASSES.items()
        if name not in {"MultiPeakModel", "PolynomialModel", "FermiEdge2dModel"}
    )


def _builtin_name(model: lmfit.Model) -> str | None:
    for name, cls in _MODEL_CLASSES.items():
        if type(model) is cls:
            return name
    if type(model) is not lmfit.Model:
        return None
    for func_class, name in zip(
        _DYNAMIC_CLASSES,
        ("MultiPeakModel", "PolynomialModel", "FermiEdge2dModel"),
        strict=True,
    ):
        if type(model.func) is func_class:
            return name
    return next(
        (name for name, func in _static_functions() if model.func is func), None
    )


def _same_state(left: typing.Any, right: typing.Any) -> bool:
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if type(left) in (str, int, float, bool, type(None)):
        return bool(left == right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_state(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list | tuple):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) in _DYNAMIC_CLASSES:
        if type(left) is MultiPeakFunction and "peak_argnames" in vars(left):
            _ = right.peak_argnames
        return _same_state(vars(left), vars(right))
    return False


def _state_parts(state: typing.Any) -> list[typing.Any]:
    if (
        not isinstance(state, dict)
        or state.get("__class__") not in {"Tuple", "List"}
        or not isinstance(parts := state.get("value"), list)
        or len(parts) != 3
    ):
        raise ValueError("Invalid serialized lmfit model")
    return parts


def inspect_model_reference(value: typing.Any) -> str | None:
    """Return the fixed root class of a model that contains descriptors."""
    if isinstance(value, dict) and "model" in value:
        value = value["model"]
    left, right, op = _state_parts(value)
    if right is not None or op is not None:
        if inspect_model_reference(left) or inspect_model_reference(right):
            return "lmfit.model:CompositeModel"
        return None
    model = inspect_descriptor(left)
    if model is None:
        return None
    return f"{type(model).__module__}:{type(model).__qualname__}"


def _normalize_state(state: typing.Any, model: lmfit.Model) -> bool:
    if type(model) is lmfit.model.CompositeModel:
        parts = _state_parts(state)
        left = _normalize_state(parts[0], model.left)
        right = _normalize_state(parts[1], model.right)
        return left or right
    name = _builtin_name(model)
    if name is None:
        return False
    item = _state_parts(state)[0]
    descriptor = {_MARKER: name, "config": _config(name, model.func)}
    candidate_item = {**item, "funcdef": descriptor}
    try:
        candidate = inspect_descriptor(candidate_item)
    except (TypeError, ValueError):
        return False
    if candidate is None or not _same_state(model.func, candidate.func):
        return False
    item["funcdef"] = descriptor
    return True


def dumps_model(model: lmfit.Model, *, serialized: str | None = None) -> str:
    """Serialize built-in model functions as constructor settings."""
    if serialized is None:
        serialized = model.dumps()
    value = json.loads(serialized)
    return json.dumps(value) if _normalize_state(value, model) else serialized


def dumps_result(
    result: lmfit.model.ModelResult, *, serialized: str | None = None
) -> str:
    """Serialize a result with data-only built-in model definitions."""
    if serialized is None:
        serialized = result.dumps()
    value = json.loads(serialized)
    return (
        json.dumps(value)
        if _normalize_state(value["model"], result.model)
        else serialized
    )


def _prepare_load(
    state: typing.Any,
    functions: dict[str, typing.Any],
) -> Iterator[lmfit.Model | None]:
    left, right, op = parts = _state_parts(state)
    if right is not None or op is not None:
        yield from _prepare_load(left, functions)
        yield from _prepare_load(right, functions)
        return
    model = inspect_descriptor(left)
    if model is None:
        yield None
        return
    alias = f"<erlab builtin {len(functions)}>"
    functions[alias] = model.func
    parts[0] = {**left, "funcname": alias, "funcdef": alias}
    yield model


def _restore_classes(model: lmfit.Model, classes: Iterator[lmfit.Model | None]) -> None:
    if isinstance(model, lmfit.model.CompositeModel):
        _restore_classes(model.left, classes)
        _restore_classes(model.right, classes)
        model._parse_params()
        return
    builtin = next(classes)
    if builtin is not None:
        model.__class__ = type(builtin)
        model.def_vals = builtin.def_vals
        model.independent_vars_defvals = builtin.independent_vars_defvals
        for name, value in vars(builtin).items():
            if name not in vars(model):
                setattr(model, name, value)
    elif type(model) is lmfit.Model and type(model.func) is MultiPeakFunction:
        model.__class__ = fit_models.MultiPeakModel


def loads_model(serialized: str) -> lmfit.Model:
    """Load model settings after the caller authorizes any custom code."""
    value = json.loads(serialized)
    functions: dict[str, typing.Any] = {}
    classes = tuple(_prepare_load(value, functions))
    model = lmfit.Model(lambda x: x).loads(json.dumps(value), funcdefs=functions)
    _restore_classes(model, iter(classes))
    return model


def loads_result(serialized: str) -> lmfit.model.ModelResult:
    """Load result settings after the caller authorizes any custom code."""
    value = json.loads(serialized)
    functions: dict[str, typing.Any] = {}
    classes = tuple(_prepare_load(value["model"], functions))
    aliases = tuple(functions)
    result = lmfit.model.ModelResult(lmfit.Model(lambda x: x), lmfit.Parameters())
    result.loads(json.dumps(value), funcdefs=functions)
    _restore_classes(result.model, iter(classes))
    for params in (result.params, result.init_params):
        for alias in aliases:
            params._asteval.symtable.pop(alias, None)
    return result
