"""Shared line-style helpers for Figure Composer operations."""

from __future__ import annotations

import ast
import typing

import matplotlib.lines
import matplotlib.markers

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from erlab.interactive._figurecomposer._model._state import FigureOperationState


def _is_no_style_token(value: str) -> bool:
    return value == "" or value.isspace() or value.lower() == "none"


def _style_options(values: Iterable[typing.Any]) -> tuple[str, ...]:
    options = ["none"]
    for value in values:
        if not isinstance(value, str):
            continue
        if _is_no_style_token(value):
            continue
        if value not in options:
            options.append(value)
    return tuple(options)


LINE_STYLE_OPTIONS = _style_options(matplotlib.lines.lineStyles)
LINE_MARKER_OPTIONS = _style_options(matplotlib.markers.MarkerStyle.markers)
LINE_STYLE_DEFAULT_LABEL = "None"
CONTROLLED_LINE_KW_KEYS = frozenset(
    (
        "c",
        "color",
        "ls",
        "linestyle",
        "lw",
        "linewidth",
        "marker",
        "ms",
        "markersize",
        "mfc",
        "markerfacecolor",
        "mec",
        "markeredgecolor",
    )
)


def color_kw_value_from_text(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] in "[(":
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped
        if isinstance(value, list):
            return tuple(value)
        return value
    return stripped


def line_kw_value(
    operation: FigureOperationState, key: str, *aliases: str
) -> typing.Any:
    for candidate in (key, *aliases):
        if candidate in operation.line_kw:
            return operation.line_kw[candidate]
    return None


def line_kw_text(operation: FigureOperationState, key: str, *aliases: str) -> str:
    value = line_kw_value(operation, key, *aliases)
    return "" if value is None else str(value)


def line_kw_style_value(
    operation: FigureOperationState, key: str, *aliases: str
) -> str | None:
    value = line_kw_value(operation, key, *aliases)
    return normalize_style_value(value)


def normalize_style_value(value: typing.Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return "none" if _is_no_style_token(text) else text


def line_kw_float(
    operation: FigureOperationState, key: str, *aliases: str
) -> float | None:
    value = line_kw_value(operation, key, *aliases)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extra_line_kw(operation: FigureOperationState) -> dict[str, typing.Any]:
    return {
        key: value
        for key, value in operation.line_kw.items()
        if key not in CONTROLLED_LINE_KW_KEYS
    }
