from __future__ import annotations

import atexit
import importlib
import typing

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
    QtWidgets = importlib.import_module("qtpy.QtWidgets")
finally:
    atexit.register = _original_atexit_register

API_NAME = qtpy.API_NAME


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
