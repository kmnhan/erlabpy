from __future__ import annotations

import atexit
import importlib
import typing

import pyperclip

if typing.TYPE_CHECKING:
    from collections.abc import Callable

QTCORE_EXIT_CLEANUP: Callable[..., object] | None = None
_original_atexit_register = atexit.register


def _capture_pyqt_exit_notifier(
    func: Callable[..., object], *args: object, **kwargs: object
) -> Callable[..., object]:
    global QTCORE_EXIT_CLEANUP

    if getattr(func, "__name__", None) == "_qtcore_cleanup":
        QTCORE_EXIT_CLEANUP = func
    return _original_atexit_register(func, *args, **kwargs)


atexit.register = _capture_pyqt_exit_notifier
try:
    qtpy = importlib.import_module("qtpy")
    QtCore = importlib.import_module("qtpy.QtCore")
    QtGui = importlib.import_module("qtpy.QtGui")
    QtWidgets = importlib.import_module("qtpy.QtWidgets")
finally:
    atexit.register = _original_atexit_register

API_NAME = qtpy.API_NAME


class InMemoryClipboard(QtCore.QObject):
    Mode = QtGui.QClipboard.Mode

    dataChanged = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._mime_data = QtCore.QMimeData()
        self._pixmap: QtGui.QPixmap | None = None

    def clear(self, mode: QtGui.QClipboard.Mode = Mode.Clipboard) -> None:
        if mode != self.Mode.Clipboard:
            return
        self._mime_data = QtCore.QMimeData()
        self._pixmap = None
        self.dataChanged.emit()

    def setMimeData(
        self,
        mime_data: QtCore.QMimeData,
        mode: QtGui.QClipboard.Mode = Mode.Clipboard,
    ) -> None:
        if mode != self.Mode.Clipboard:
            return
        self._mime_data = mime_data
        self._pixmap = None
        self.dataChanged.emit()

    def mimeData(
        self,
        mode: QtGui.QClipboard.Mode = Mode.Clipboard,
    ) -> QtCore.QMimeData:
        if mode != self.Mode.Clipboard:
            return QtCore.QMimeData()
        return self._mime_data

    def setText(self, text: str, mode: QtGui.QClipboard.Mode = Mode.Clipboard) -> None:
        if mode != self.Mode.Clipboard:
            return
        mime_data = QtCore.QMimeData()
        mime_data.setText(text)
        self.setMimeData(mime_data)

    def text(self, mode: QtGui.QClipboard.Mode = Mode.Clipboard) -> str:
        if mode != self.Mode.Clipboard:
            return ""
        return self._mime_data.text()

    def setPixmap(
        self,
        pixmap: QtGui.QPixmap,
        mode: QtGui.QClipboard.Mode = Mode.Clipboard,
    ) -> None:
        if mode != self.Mode.Clipboard:
            return
        self._mime_data = QtCore.QMimeData()
        self._pixmap = QtGui.QPixmap(pixmap)
        self.dataChanged.emit()

    def pixmap(self, mode: QtGui.QClipboard.Mode = Mode.Clipboard) -> QtGui.QPixmap:
        if mode != self.Mode.Clipboard or self._pixmap is None:
            return QtGui.QPixmap()
        return QtGui.QPixmap(self._pixmap)


_ACTIVE_TEST_QT_CLIPBOARD = InMemoryClipboard()


def reset_test_qt_clipboard() -> InMemoryClipboard:
    global _ACTIVE_TEST_QT_CLIPBOARD

    _ACTIVE_TEST_QT_CLIPBOARD = InMemoryClipboard()
    return _ACTIVE_TEST_QT_CLIPBOARD


def _test_qt_clipboard() -> InMemoryClipboard:
    return _ACTIVE_TEST_QT_CLIPBOARD


# Install this before test collection. Calls through the Python Qt API stay isolated
# during collection, fixture setup, fixture teardown, and conftest reloads.
QtWidgets.QApplication.clipboard = staticmethod(_test_qt_clipboard)


_ACTIVE_TEST_TEXT_CLIPBOARD = ""


def reset_test_text_clipboard() -> None:
    global _ACTIVE_TEST_TEXT_CLIPBOARD

    _ACTIVE_TEST_TEXT_CLIPBOARD = ""


def _copy_test_text(content: object) -> None:
    global _ACTIVE_TEST_TEXT_CLIPBOARD

    _ACTIVE_TEST_TEXT_CLIPBOARD = str(content)


def _paste_test_text() -> str:
    return _ACTIVE_TEST_TEXT_CLIPBOARD


# Install these before test collection for the same reason as the Qt clipboard.
pyperclip.copy = _copy_test_text
pyperclip.paste = _paste_test_text


def signal_receiver_count(obj: QtCore.QObject, signal: object, signal_name: str) -> int:
    try:
        return obj.receivers(signal)
    except TypeError:
        return obj.receivers(_encoded_signal_signature(obj, signal, signal_name))


def _encoded_signal_signature(
    obj: QtCore.QObject, signal: object, signal_name: str
) -> str:
    from_signal = getattr(QtCore.QMetaMethod, "fromSignal", None)
    if from_signal is not None:
        method = from_signal(signal)
        if method.isValid():
            return f"2{bytes(method.methodSignature()).decode()}"

    meta = obj.metaObject()

    for method_index in range(meta.methodCount()):
        method = meta.method(method_index)
        if method.methodType() != QtCore.QMetaMethod.MethodType.Signal:
            continue

        signature = bytes(method.methodSignature()).decode()
        if signature.startswith(f"{signal_name}("):
            return f"2{signature}"

    raise ValueError(f"Could not resolve Qt signal signature for {signal_name!r}")
