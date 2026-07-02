import contextlib
import logging
from collections.abc import Iterator

import pytest
from qtpy import QtCore, QtWidgets

from erlab.interactive.imagetool.manager._mainwindow import (
    _WarningEmitter,
    _WarningNotificationHandler,
)


@pytest.fixture
def warning_emitter(qapp: QtWidgets.QApplication) -> Iterator[_WarningEmitter]:
    emitter = _WarningEmitter()
    try:
        yield emitter
    finally:
        with contextlib.suppress(RuntimeError, TypeError):
            emitter.warning_received.disconnect()
        with contextlib.suppress(RuntimeError):
            emitter.deleteLater()
        QtWidgets.QApplication.sendPostedEvents(
            emitter, QtCore.QEvent.Type.DeferredDelete
        )
        qapp.processEvents()


def test_warning_handler_formats_message(warning_emitter: _WarningEmitter) -> None:
    emitter = warning_emitter
    handler = _WarningNotificationHandler(emitter)

    received: list[tuple[str, int, str, str]] = []

    def _capture(levelname, levelno, message, traceback_msg):
        received.append((levelname, levelno, message, traceback_msg))

    emitter.warning_received.connect(_capture)

    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )

    handler.emit(record)

    assert received
    assert received[0][2] == "hello world"


def test_warning_handler_formats_exc_info(warning_emitter: _WarningEmitter) -> None:
    emitter = warning_emitter
    handler = _WarningNotificationHandler(emitter)

    received: list[tuple[str, int, str, str]] = []

    def _capture(levelname, levelno, message, traceback_msg):
        received.append((levelname, levelno, message, traceback_msg))

    emitter.warning_received.connect(_capture)

    try:
        raise ValueError("boom")  # noqa: TRY301
    except ValueError:
        exc_info = logging.sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="error happened",
        args=(),
        exc_info=exc_info,
    )

    handler.emit(record)

    assert received
    assert "ValueError" in received[0][3]
