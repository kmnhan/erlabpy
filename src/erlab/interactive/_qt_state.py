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
    # Rect retains the client size for files whose native geometry cannot be restored.
    rect: tuple[int, int, int, int] | None = None
    # Visibility is tracked separately so hidden restored windows can stay hidden.
    visible: bool = False

    model_config = pydantic.ConfigDict(extra="ignore")


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
    # Qt gives a never-shown widget placeholder geometry. Keep that geometry
    # unset so the first show can use sizeHint() and native screen constraints.
    has_explicit_size = widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_Resized)
    has_been_shown = widget.testAttribute(
        QtCore.Qt.WidgetAttribute.WA_WState_ExplicitShowHide
    ) and not widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_PendingResizeEvent)
    if not (has_explicit_size or has_been_shown):
        return QtWindowState(visible=bool(widget.isVisible()))

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
    """Restore native Qt geometry or a safe client-size fallback.

    Qt corrects the screen position when it accepts native geometry. A rectangle
    fallback restores only its size because its position can refer to an unavailable
    screen or use incompatible window-frame coordinates.
    """
    parsed = parse_qt_window_state(state)
    if parsed is None:
        return False

    geometry = qt_bytearray_from_base64(parsed.geometry)
    if geometry is not None and widget.restoreGeometry(geometry):
        return True

    if parsed.rect is None:
        return False

    _, _, width, height = parsed.rect
    maximum_width = widget.maximumWidth()
    maximum_height = widget.maximumHeight()
    screen = widget.screen()
    if screen is not None:
        available_size = screen.availableGeometry().size()
        maximum_width = min(maximum_width, available_size.width())
        maximum_height = min(maximum_height, available_size.height())
    width = min(width, maximum_width)
    height = min(height, maximum_height)
    if width <= 0 or height <= 0:
        return False

    # Saved positions are not portable across screens, display scales, or window
    # managers. Keep the current position so Qt can place an unshown window safely.
    widget.resize(width, height)
    return True
