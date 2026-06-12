"""Shared ``fig.subplots_adjust`` value handling."""

from __future__ import annotations

import contextlib
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

SUBPLOTS_ADJUST_KEYS = ("left", "bottom", "right", "top", "wspace", "hspace")
SUBPLOTS_ADJUST_SPINBOX_MINIMUM = 0.0
SUBPLOTS_ADJUST_SPINBOX_MAXIMUM = 1.0
SUBPLOTS_ADJUST_SPINBOX_DECIMALS = 3
SUBPLOTS_ADJUST_SPINBOX_STEP = 0.005

_SUBPLOTS_ADJUST_DELTA = 10.0**-SUBPLOTS_ADJUST_SPINBOX_DECIMALS
_SUBPLOTS_ADJUST_PAIRS = (("left", "right"), ("bottom", "top"))
_SUBPLOTS_ADJUST_BORDER_KEYS = ("left", "bottom", "right", "top")


def subplots_adjust_spinbox_range(
    key: str, values: Mapping[str, typing.Any]
) -> tuple[float, float]:
    """Return the valid editor range for a ``subplots_adjust`` key."""
    minimum = SUBPLOTS_ADJUST_SPINBOX_MINIMUM
    maximum = SUBPLOTS_ADJUST_SPINBOX_MAXIMUM
    if key == "left" and (right := _float_value(values.get("right"))) is not None:
        maximum = min(maximum, right - _SUBPLOTS_ADJUST_DELTA)
    elif key == "right" and (left := _float_value(values.get("left"))) is not None:
        minimum = max(minimum, left + _SUBPLOTS_ADJUST_DELTA)
    elif key == "bottom" and (top := _float_value(values.get("top"))) is not None:
        maximum = min(maximum, top - _SUBPLOTS_ADJUST_DELTA)
    elif key == "top" and (bottom := _float_value(values.get("bottom"))) is not None:
        minimum = max(minimum, bottom + _SUBPLOTS_ADJUST_DELTA)
    if minimum > maximum:
        if key in {"left", "bottom"}:
            maximum = minimum
        else:
            minimum = maximum
    return minimum, maximum


def normalize_subplots_adjust_kwargs(
    kwargs: Mapping[str, typing.Any],
    *,
    defaults: Mapping[str, typing.Any] | None = None,
    changed_key: str | None = None,
) -> dict[str, typing.Any]:
    """Clamp ``subplots_adjust`` kwargs so Matplotlib paired limits stay valid."""
    normalized = dict(kwargs)
    for key in _SUBPLOTS_ADJUST_BORDER_KEYS:
        if (value := _float_value(normalized.get(key))) is not None:
            normalized[key] = _clamp(value)
    for lower_key, upper_key in _SUBPLOTS_ADJUST_PAIRS:
        lower = _float_value(normalized.get(lower_key))
        upper = _float_value(normalized.get(upper_key))
        lower_from_kwargs = lower is not None
        upper_from_kwargs = upper is not None
        if lower is None and defaults is not None:
            lower = _float_value(defaults.get(lower_key))
        if upper is None and defaults is not None:
            upper = _float_value(defaults.get(upper_key))
        if lower is None or upper is None or lower < upper:
            continue
        if lower_from_kwargs and not upper_from_kwargs:
            normalized[lower_key] = max(
                SUBPLOTS_ADJUST_SPINBOX_MINIMUM,
                upper - _SUBPLOTS_ADJUST_DELTA,
            )
            continue
        if upper_from_kwargs and not lower_from_kwargs:
            normalized[upper_key] = min(
                SUBPLOTS_ADJUST_SPINBOX_MAXIMUM,
                lower + _SUBPLOTS_ADJUST_DELTA,
            )
            continue
        lower, upper = _repair_pair(
            lower,
            upper,
            changed_key=changed_key,
            lower_key=lower_key,
            upper_key=upper_key,
        )
        normalized[lower_key] = lower
        normalized[upper_key] = upper
    return normalized


def _float_value(value: typing.Any) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(value)
    return None


def _clamp(value: float) -> float:
    return min(
        SUBPLOTS_ADJUST_SPINBOX_MAXIMUM,
        max(SUBPLOTS_ADJUST_SPINBOX_MINIMUM, value),
    )


def _repair_pair(
    lower: float,
    upper: float,
    *,
    changed_key: str | None,
    lower_key: str,
    upper_key: str,
) -> tuple[float, float]:
    if changed_key == upper_key:
        upper = min(SUBPLOTS_ADJUST_SPINBOX_MAXIMUM, lower + _SUBPLOTS_ADJUST_DELTA)
        if lower < upper:
            return lower, upper
        lower = max(SUBPLOTS_ADJUST_SPINBOX_MINIMUM, upper - _SUBPLOTS_ADJUST_DELTA)
        return lower, upper
    lower = max(SUBPLOTS_ADJUST_SPINBOX_MINIMUM, upper - _SUBPLOTS_ADJUST_DELTA)
    if lower < upper:
        return lower, upper
    upper = min(SUBPLOTS_ADJUST_SPINBOX_MAXIMUM, lower + _SUBPLOTS_ADJUST_DELTA)
    return lower, upper
