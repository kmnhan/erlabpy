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


def configure_style_combo(
    combo: QtWidgets.QComboBox,
    options: tuple[str, ...],
    current: str | None,
) -> None:
    combo.clear()
    combo.addItem(LINE_STYLE_DEFAULT_LABEL, None)
    for value in options:
        combo.addItem(value, value)
    set_style_combo_value(combo, current)


def set_style_combo_value(combo: QtWidgets.QComboBox, current: str | None) -> None:
    normalized = normalize_style_value(current)
    for index in range(combo.count()):
        if combo.itemData(index) == normalized:
            combo.setCurrentIndex(index)
            return
    if normalized is not None:
        combo.addItem(normalized, normalized)
        combo.setCurrentIndex(combo.count() - 1)


def style_combo_value(combo: QtWidgets.QComboBox) -> str | None:
    return typing.cast("str | None", combo.currentData())


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
    clear_stale_cmap: bool = False,
    clear_stale_line_colormap: bool = False,
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
        if clear_stale_cmap:
            updates["cmap"] = None
        if clear_stale_line_colormap:
            updates["line_color_mode"] = "manual"
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
