import logging

from erlab.interactive.imagetool.manager._mainwindow import (
    _WarningEmitter,
    _WarningNotificationHandler,
)


def test_warning_handler_formats_message(qtbot) -> None:
    emitter = _WarningEmitter()
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
