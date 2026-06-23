"""Coordinate-colormap helpers for Figure Composer line outputs."""

from __future__ import annotations

import math
import typing

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from erlab.interactive._figurecomposer._defaults import _current_options
from erlab.interactive._figurecomposer._norms import _cmap_with_reverse

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from erlab.interactive._figurecomposer._state import FigureOperationState


_GENERIC_CONTEXT_FIELDS = frozenset(("index", "number", "source", "dim", "value"))
LINE_COLOR_CMAP_TRIM_MAX = 0.49


def line_colormap_active(operation: FigureOperationState) -> bool:
    return operation.line_color_mode == "coordinate"


def effective_line_color_coord(
    operation: FigureOperationState, default: str | None
) -> str | None:
    return operation.line_color_coord or default


def effective_line_color_cmap(operation: FigureOperationState) -> str:
    base = operation.line_color_cmap or _current_options().colors.cmap.name
    return _cmap_with_reverse(base, operation.line_color_cmap_reverse) or base


def effective_line_color_cmap_trim(
    operation: FigureOperationState,
) -> tuple[float, float]:
    return _validate_line_color_cmap_trim(
        operation.line_color_cmap_trim_lower,
        operation.line_color_cmap_trim_upper,
    )


def line_color_cmap_trim_control_values(
    operation: FigureOperationState,
) -> tuple[float, float]:
    try:
        return effective_line_color_cmap_trim(operation)
    except ValueError:
        return (0.0, 0.0)


def _validate_line_color_cmap_trim(
    lower: float,
    upper: float,
) -> tuple[float, float]:
    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Line colormap trim values must be finite")
    if lower < 0.0 or upper < 0.0 or lower + upper >= 1.0:
        raise ValueError(
            "Line colormap trim values must be at least 0 and sum to less than 1"
        )
    return lower, upper


def numeric_context_field_names(
    contexts: Sequence[Mapping[str, typing.Any]],
) -> tuple[str, ...]:
    names: list[str] = []
    for context in contexts:
        for name in context:
            if name in _GENERIC_CONTEXT_FIELDS or name in names:
                continue
            try:
                values_from_contexts(contexts, name, item_name="line")
            except ValueError:
                continue
            names.append(name)
    return tuple(names)


def values_from_contexts(
    contexts: Sequence[Mapping[str, typing.Any]],
    coord: str | None,
    *,
    item_name: str,
) -> tuple[float, ...]:
    if not coord:
        raise ValueError(f"Choose a coordinate to color {item_name}s")
    values: list[float] = []
    for context in contexts:
        if coord not in context:
            raise ValueError(
                f"Cannot color {item_name}s by {coord!r}: "
                f"the coordinate is missing for one or more {item_name}s"
            )
        try:
            value = float(context[coord])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot color {item_name}s by {coord!r}: "
                "values must be numeric scalars"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Cannot color {item_name}s by {coord!r}: "
                "values must be finite numeric scalars"
            )
        values.append(value)
    return tuple(values)


def colors_from_values(
    values: Sequence[float],
    cmap: str,
    *,
    trim: tuple[float, float] = (0.0, 0.0),
) -> tuple[tuple[float, float, float, float], ...]:
    if not values:
        return ()
    trim_lower, trim_upper = _validate_line_color_cmap_trim(*trim)
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        norm = mcolors.Normalize(vmin=vmin - 0.5, vmax=vmax + 0.5)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    normalized = np.asarray(norm(values), dtype=float)
    if trim_lower or trim_upper:
        normalized = trim_lower + (1.0 - trim_lower - trim_upper) * normalized
    mapped = np.asarray(plt.get_cmap(cmap)(normalized), dtype=float)
    return tuple(
        (float(color[0]), float(color[1]), float(color[2]), float(color[3]))
        for color in mapped
    )


def colormap_code_lines(
    value_expression: str,
    cmap: str,
    *,
    trim: tuple[float, float] = (0.0, 0.0),
    values_name: str = "line_color_values",
    colors_name: str = "line_colors",
) -> list[str]:
    trim_lower, trim_upper = _validate_line_color_cmap_trim(*trim)
    vmin_name = f"{values_name}_vmin"
    vmax_name = f"{values_name}_vmax"
    norm_name = f"{values_name}_norm"
    color_values = f"{norm_name}({values_name})"
    if trim_lower or trim_upper:
        scale = 1.0 - trim_lower - trim_upper
        color_values = f"{trim_lower!r} + {scale!r} * {color_values}"
        color_code = [
            f"    {colors_name} = plt.get_cmap({cmap!r})(",
            f"        {color_values}",
            "    )",
        ]
    else:
        color_code = [f"    {colors_name} = plt.get_cmap({cmap!r})({color_values})"]
    return [
        f"{values_name} = {value_expression}",
        f"if {values_name}:",
        f"    {vmin_name} = min({values_name})",
        f"    {vmax_name} = max({values_name})",
        f"    if {vmin_name} == {vmax_name}:",
        f"        {norm_name} = mcolors.Normalize(",
        f"            vmin={vmin_name} - 0.5,",
        f"            vmax={vmax_name} + 0.5,",
        "        )",
        "    else:",
        f"        {norm_name} = mcolors.Normalize(",
        f"            vmin={vmin_name},",
        f"            vmax={vmax_name},",
        "        )",
        *color_code,
        "else:",
        f"    {colors_name} = []",
    ]
