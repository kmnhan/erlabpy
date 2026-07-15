"""Qt controls and editor mutations for Figure Composer line styles."""

from __future__ import annotations

import typing

from qtpy import QtWidgets

from erlab.interactive._figurecomposer._line_style import (
    CONTROLLED_LINE_KW_KEYS,
    LINE_STYLE_DEFAULT_LABEL,
    normalize_style_value,
)

if typing.TYPE_CHECKING:
    from erlab.interactive._figurecomposer._model._state import FigureOperationState
    from erlab.interactive._figurecomposer._ui._operation_editor import (
        FigureOperationEditor,
    )


class _LineStyleDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def setToolTip(self, text: str | None) -> None:
        super().setToolTip(text)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setToolTip(text)


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


def update_current_line_kw(
    editor: FigureOperationEditor,
    key: str,
    value: typing.Any,
    *,
    aliases: tuple[str, ...] = (),
    clear_stale_cmap: bool = False,
    clear_stale_line_colormap: bool = False,
) -> None:
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

    editor.request_transform(update_operation)


def update_current_extra_line_kw(
    editor: FigureOperationEditor, extra_line_kw: dict[str, typing.Any]
) -> None:
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

    editor.request_transform(update_operation)
