__all__ = ["PORT", "_ManagerServer", "_save_pickle"]

import contextlib
import logging
import os
import pickle
import socket
import struct
import threading
import time
from typing import Any

from qtpy import QtCore

from erlab.interactive.utils import _coverage_resolve_trace

logger = logging.getLogger(__name__)

PORT: int = int(os.getenv("ITOOL_MANAGER_PORT", "45555"))
"""Port number for the manager server.

The default port number 45555 can be overridden by setting the environment variable
``ITOOL_MANAGER_PORT``.
"""


def _save_pickle(obj: Any, filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(obj, file, protocol=-1)


def _load_pickle(filename: str) -> Any:
    with open(filename, "rb") as file:
        return pickle.load(file)


def _recv_all(conn, size):
    data = b""
    while len(data) < size:
        part = conn.recv(size - len(data))
        data += part
    return data


class _ManagerServer(QtCore.QThread):
    sigReceived = QtCore.Signal(list, dict)

    def __init__(self) -> None:
        super().__init__()
        self.stopped = threading.Event()

    @_coverage_resolve_trace
    def run(self) -> None:
        self.stopped.clear()

        logger.debug("Starting server...")
        soc = socket.socket()
        soc.bind(("127.0.0.1", PORT))
        soc.setblocking(False)
        soc.listen()

        logger.info("Server is listening...")

        while not self.stopped.is_set():
            try:
                conn, _ = soc.accept()
            except BlockingIOError:
                time.sleep(0.01)
                continue

            conn.setblocking(True)
            logger.debug("Connection accepted")
            # Receive the size of the data first
            data_size = struct.unpack(">L", _recv_all(conn, 4))[0]

            # Receive the data
            kwargs = _recv_all(conn, data_size)
            try:
                kwargs = pickle.loads(kwargs)
                logger.debug("Received data: %s", kwargs)

                files = kwargs.pop("__filename")
                self.sigReceived.emit([_load_pickle(f) for f in files], kwargs)
                logger.debug("Emitted loaded data")

                # Clean up temporary files
                for f in files:
                    os.remove(f)
                    dirname = os.path.dirname(f)
                    if os.path.isdir(dirname):
                        with contextlib.suppress(OSError):
                            os.rmdir(dirname)
                logger.debug("Cleaned up temporary files")

            except (
                pickle.UnpicklingError,
                AttributeError,
                EOFError,
                ImportError,
                IndexError,
            ):
                logger.exception("Failed to unpickle received data")

            conn.close()
            logger.debug("Connection closed")

        soc.close()
