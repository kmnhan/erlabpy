"""Logging helpers for the ImageTool manager application."""

from __future__ import annotations

import io
import logging
import pathlib
import sys
from logging.handlers import RotatingFileHandler

from qtpy import QtCore

import erlab


class _StreamToLogger(io.TextIOBase):  # pragma: no cover
    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buffer: str) -> int:
        for line in buffer.splitlines():
            line = line.strip()
            if line:
                self.logger.log(self.level, line)
        return len(buffer)

    def flush(self) -> None:
        return None


def _log_directory() -> pathlib.Path:
    location = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.AppDataLocation
    )
    if not location:  # pragma: no cover
        location = str(pathlib.Path.home() / ".erlab" / "logs")

    path = pathlib.Path(location)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_file_path() -> pathlib.Path:
    """Return the path to the log file used by ImageTool Manager."""
    return _log_directory() / "imagetool-manager.log"


def configure_logging() -> None:
    """Configure logging for the ImageTool Manager application."""
    root_logger = logging.getLogger()

    log_path = get_log_file_path()

    file_handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    if (
        root_logger.level == logging.NOTSET or root_logger.level > logging.INFO
    ):  # pragma: no cover
        root_logger.setLevel(logging.INFO)
    root_logger.info("Writing ImageTool Manager logs to %s", log_path)

    logging.captureWarnings(True)

    if erlab.utils.misc._IS_PACKAGED:  # pragma: no cover
        sys.stdout = _StreamToLogger(
            logging.getLogger("imagetool.stdout"), logging.INFO
        )
        sys.stderr = _StreamToLogger(
            logging.getLogger("imagetool.stderr"), logging.ERROR
        )
        root_logger.info("Logging redirected to %s", log_path)
