"""Color map and normalization helpers for Figure Composer."""

from __future__ import annotations

import typing

import matplotlib.colors as mcolors

import erlab
import erlab.interactive.utils
import erlab.plotting as eplt
from erlab.interactive._figurecomposer._defaults import _current_options
from erlab.interactive._figurecomposer._state import (
    _POWER_NORM_NAME,
    FigureOperationState,
)
from erlab.interactive._figurecomposer._text import _code_kwargs

_MATPLOTLIB_NORM_NAMES = frozenset(
    {
        "Normalize",
        "PowerNorm",
        "LogNorm",
        "TwoSlopeNorm",
    }
)
_ERLAB_NORM_NAMES = frozenset(
    {
        "InversePowerNorm",
        "TwoSlopePowerNorm",
        "TwoSlopeInversePowerNorm",
        "CenteredPowerNorm",
        "CenteredInversePowerNorm",
    }
)
_NORM_CHOICES = (
    "PowerNorm",
    "Normalize",
    "LogNorm",
    "TwoSlopeNorm",
    "InversePowerNorm",
    "TwoSlopePowerNorm",
    "TwoSlopeInversePowerNorm",
    "CenteredPowerNorm",
    "CenteredInversePowerNorm",
)
_NORM_KWARG_FIELDS = {
    "Normalize": ("vmin", "vmax", "clip"),
    "PowerNorm": ("gamma", "vmin", "vmax", "clip"),
    "LogNorm": ("vmin", "vmax", "clip"),
    "TwoSlopeNorm": ("vcenter", "vmin", "vmax"),
    "InversePowerNorm": ("gamma", "vmin", "vmax", "clip"),
    "TwoSlopePowerNorm": ("gamma", "vcenter", "vmin", "vmax"),
    "TwoSlopeInversePowerNorm": ("gamma", "vcenter", "vmin", "vmax"),
    "CenteredPowerNorm": ("gamma", "vcenter", "halfrange", "clip"),
    "CenteredInversePowerNorm": ("gamma", "vcenter", "halfrange", "clip"),
}
_ZERO_VCENTER_NORMS = frozenset(
    {
        "TwoSlopeNorm",
        "TwoSlopePowerNorm",
        "TwoSlopeInversePowerNorm",
        "CenteredPowerNorm",
        "CenteredInversePowerNorm",
    }
)
_UNKNOWN_NORM_KWARG_FIELDS = ("gamma",)
_NORM_STRUCTURED_KWARG_FIELDS = {
    "gamma": "norm_gamma",
    "vmin": "vmin",
    "vmax": "vmax",
    "vcenter": "vcenter",
    "halfrange": "halfrange",
    "clip": "norm_clip",
}


def _effective_norm_name(name: str | None) -> str:
    return name or _POWER_NORM_NAME


def _norm_name_from_combo_text(text: str) -> str:
    return text


def _norm_combo_text(name: str | None) -> str:
    return _effective_norm_name(name)


def _norm_module_prefix(name: str) -> str:
    if name in _MATPLOTLIB_NORM_NAMES:
        return "mcolors"
    if name in _ERLAB_NORM_NAMES:
        return "eplt"
    return "eplt"


def _norm_combo_choices(name: str | None) -> tuple[str, ...]:
    effective_name = _effective_norm_name(name)
    if effective_name in _NORM_CHOICES:
        return _NORM_CHOICES
    return (*_NORM_CHOICES, effective_name)


def _norm_kwarg_fields(name: str | None) -> tuple[str, ...]:
    return _NORM_KWARG_FIELDS.get(
        _effective_norm_name(name), _UNKNOWN_NORM_KWARG_FIELDS
    )


def _norm_float_value(value: typing.Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _norm_updates_from_kwargs(
    kwargs: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    updates: dict[str, typing.Any] = {}
    remaining: dict[str, typing.Any] = {}
    for key, value in kwargs.items():
        field = _NORM_STRUCTURED_KWARG_FIELDS.get(key)
        if field is None:
            remaining[key] = value
        elif field == "norm_clip":
            updates[field] = None if value is None else bool(value)
        else:
            updates[field] = _norm_float_value(value)
    updates["norm_kwargs"] = remaining
    return updates


def _cmap_base_and_reverse(cmap: str | None) -> tuple[str, bool]:
    if cmap is None:
        return _current_options().colors.cmap.name, False
    if cmap.endswith("_r"):
        return cmap[:-2], True
    return cmap, False


def _cmap_with_reverse(base: str, reverse: bool) -> str | None:
    if not base:
        return None
    base = base.removesuffix("_r")
    if reverse:
        return f"{base}_r"
    return base


def _norm_constructor_kwargs(operation: FigureOperationState) -> dict[str, typing.Any]:
    name = _effective_norm_name(operation.norm_name)
    fields = _NORM_KWARG_FIELDS.get(name)
    if fields is None:
        return dict(operation.norm_kwargs)

    kwargs: dict[str, typing.Any] = {}
    if "gamma" in fields:
        if operation.norm_gamma is not None:
            kwargs["gamma"] = operation.norm_gamma
        elif operation.gamma is not None:
            kwargs["gamma"] = operation.gamma
        elif "gamma" not in operation.norm_kwargs:
            kwargs["gamma"] = 1.0
    for field in ("vmin", "vmax", "vcenter", "halfrange"):
        if field not in fields:
            continue
        value = getattr(operation, field)
        if value is not None:
            kwargs[field] = value
        elif (
            name == "TwoSlopeNorm"
            and field == "vcenter"
            and "vcenter" not in operation.norm_kwargs
        ):
            kwargs[field] = 0.0
    if "clip" in fields and operation.norm_clip is not None:
        kwargs["clip"] = operation.norm_clip
    for key, value in operation.norm_kwargs.items():
        kwargs.setdefault(key, value)
    return kwargs


def _use_powernorm_plot_kwargs(operation: FigureOperationState) -> bool:
    return (
        _effective_norm_name(operation.norm_name) == "PowerNorm"
        and operation.norm_clip is None
        and not operation.norm_kwargs
    )


def _norm_object(operation: FigureOperationState) -> object:
    effective_name = _effective_norm_name(operation.norm_name)
    module = mcolors if _norm_module_prefix(effective_name) == "mcolors" else eplt
    norm_cls = getattr(module, effective_name)
    if effective_name not in _NORM_KWARG_FIELDS:
        norm_kwargs = dict(operation.norm_kwargs)
        if operation.norm_gamma is None:
            return norm_cls(**norm_kwargs)
        return norm_cls(operation.norm_gamma, **norm_kwargs)
    return norm_cls(**_norm_constructor_kwargs(operation))


def _norm_code(operation: FigureOperationState) -> str:
    effective_name = _effective_norm_name(operation.norm_name)
    prefix = _norm_module_prefix(effective_name)
    if effective_name not in _NORM_KWARG_FIELDS:
        args = ""
        if operation.norm_gamma is not None:
            args = erlab.interactive.utils._parse_single_arg(operation.norm_gamma)
        kwargs = _code_kwargs(operation.norm_kwargs)
        if args and kwargs:
            args += f", {kwargs}"
        elif kwargs:
            args = kwargs
        return f"{prefix}.{effective_name}({args})"

    kwargs = _norm_constructor_kwargs(operation)
    return f"{prefix}.{effective_name}({_code_kwargs(kwargs)})"
