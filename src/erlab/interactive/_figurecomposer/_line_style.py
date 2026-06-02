"""Shared line-style helpers for Figure Composer operations."""

from __future__ import annotations

import ast
import typing

import matplotlib.lines
import matplotlib.markers
from qtpy import QtWidgets

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from erlab.interactive._figurecomposer._state import FigureOperationState
    from erlab.interactive._figurecomposer._tool import FigureComposerTool


def _style_options(values: Iterable[typing.Any]) -> tuple[str, ...]:
    options = [""]
    for value in values:
        if not isinstance(value, str):
            continue
        if value not in options:
            options.append(value)
    return tuple(options)


LINE_STYLE_OPTIONS = _style_options(matplotlib.lines.lineStyles)
LINE_MARKER_OPTIONS = _style_options(matplotlib.markers.MarkerStyle.markers)
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


class _LineStyleDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def setToolTip(self, text: str | None) -> None:
        super().setToolTip(text)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setToolTip(text)


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


def optional_positive_spinbox(
    value: float | None,
    *,
    parent: QtWidgets.QWidget,
) -> QtWidgets.QDoubleSpinBox:
    spinbox = _LineStyleDoubleSpinBox(parent)
    spinbox.setRange(0.0, 1_000_000.0)
    spinbox.setDecimals(3)
    spinbox.setSingleStep(0.5)
    spinbox.setSpecialValueText("default")
    spinbox.setKeyboardTracking(False)
    spinbox.setValue(0.0 if value is None else float(value))
    return spinbox


def optional_positive_spinbox_value(value: float) -> float | None:
    return None if value == 0.0 else float(value)


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


def update_current_line_kw(
    tool: FigureComposerTool,
    key: str,
    value: typing.Any,
    *,
    aliases: tuple[str, ...] = (),
    clear_legacy_cmap: bool = False,
) -> None:
    if tool._updating_controls:
        return

    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        line_kw = dict(operation.line_kw)
        for candidate in (key, *aliases):
            line_kw.pop(candidate, None)
        if value is not None:
            line_kw[key] = value
        updates: dict[str, typing.Any] = {"line_kw": line_kw}
        if clear_legacy_cmap:
            updates["cmap"] = None
        return operation.model_copy(update=updates)

    tool._update_operations(update_operation)


def update_current_extra_line_kw(
    tool: FigureComposerTool, extra_line_kw: dict[str, typing.Any]
) -> None:
    if tool._updating_controls:
        return

    def update_operation(
        _operation_index: int, operation: FigureOperationState
    ) -> FigureOperationState:
        line_kw = {
            key: value
            for key, value in operation.line_kw.items()
            if key in CONTROLLED_LINE_KW_KEYS
        }
        line_kw.update(extra_line_kw)
        return operation.model_copy(update={"line_kw": line_kw})

    tool._update_operations(update_operation)
