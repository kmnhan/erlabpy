from __future__ import annotations

import base64
import binascii
import json
import typing

import pydantic
from qtpy import QtCore, QtWidgets

__all__ = [
    "QtWindowState",
    "parse_qt_window_state",
    "qt_bytearray_from_base64",
    "qt_bytearray_to_base64",
    "qt_window_state",
    "qt_window_state_json",
    "qt_window_state_payload",
    "restore_qt_window_state",
]


class QtWindowState(pydantic.BaseModel):
    """JSON-safe Qt top-level window state."""

    # Native Qt geometry preserves frame/window-manager state when Qt can restore it.
    geometry: str | None = None
    # Rect is a portable fallback when native Qt geometry cannot be restored.
    rect: tuple[int, int, int, int] | None = None
    # Visibility is tracked separately so hidden restored windows can stay hidden.
    visible: bool = False

    model_config = pydantic.ConfigDict(extra="ignore")

    @pydantic.field_validator("rect", mode="before")
    @classmethod
    def _validate_rect(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("window rect must contain four integers")
        return tuple(int(item) for item in value)


def qt_bytearray_to_base64(value: QtCore.QByteArray) -> str:
    return base64.b64encode(value.data()).decode("ascii")


def qt_bytearray_from_base64(value: object) -> QtCore.QByteArray | None:
    if isinstance(value, bytes):
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    elif isinstance(value, str):
        text = value
    else:
        return None

    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError):
        return None
    if not raw:
        return None
    return QtCore.QByteArray(raw)


def qt_window_state(widget: QtWidgets.QWidget) -> QtWindowState:
    return QtWindowState(
        geometry=qt_bytearray_to_base64(widget.saveGeometry()),
        rect=widget.geometry().getRect(),
        visible=bool(widget.isVisible()),
    )


def qt_window_state_payload(widget: QtWidgets.QWidget) -> dict[str, typing.Any]:
    return qt_window_state(widget).model_dump(mode="json", exclude_none=True)


def qt_window_state_json(widget: QtWidgets.QWidget) -> str:
    return qt_window_state(widget).model_dump_json(exclude_none=True)


def parse_qt_window_state(value: object) -> QtWindowState | None:
    if isinstance(value, QtWindowState):
        return value
    if isinstance(value, bytes):
        try:
            value = value.decode()
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    try:
        return QtWindowState.model_validate(value)
    except pydantic.ValidationError:
        return None


def restore_qt_window_state(
    widget: QtWidgets.QWidget, state: QtWindowState | object
) -> bool:
    parsed = parse_qt_window_state(state)
    if parsed is None:
        return False

    restored = False
    geometry = qt_bytearray_from_base64(parsed.geometry)
    if geometry is not None:
        restored = bool(widget.restoreGeometry(geometry))
    if not restored and parsed.rect is not None:
        widget.setGeometry(*parsed.rect)
        restored = True
    return restored
