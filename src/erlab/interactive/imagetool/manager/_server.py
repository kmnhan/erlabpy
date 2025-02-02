"""Server that listens to incoming data."""

from __future__ import annotations

__all__ = ["PORT", "load_in_manager", "show_in_manager"]

import contextlib
import errno
import logging
import os
import pathlib
import pickle
import socket
import struct
import tempfile
import threading
import time
import typing
import uuid

import xarray as xr
from qtpy import QtCore

import erlab
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME

if typing.TYPE_CHECKING:
    from collections.abc import Collection

    import numpy.typing as npt


logger = logging.getLogger(__name__)

PORT: int = int(os.getenv("ITOOL_MANAGER_PORT", "45555"))
"""Port number for the manager server.

The default port number 45555 can be overridden by setting the environment variable
``ITOOL_MANAGER_PORT``.
"""


def _save_pickle(obj: typing.Any, filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(obj, file, protocol=-1)


def _load_pickle(filename: str) -> typing.Any:
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
    sigLoadRequested = QtCore.Signal(list, str)

    def __init__(self) -> None:
        super().__init__()
        self.stopped = threading.Event()

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

            # "__filename" contains the list of paths to pickled data
            # If "__loader_name" key is present, "__filename" will be a list of paths to
            # raw data files instead
            try:
                kwargs = pickle.loads(kwargs)
                logger.debug("Received data: %s", kwargs)

                files = kwargs.pop("__filename")
                loader_name = kwargs.pop("__loader_name", None)
                if loader_name is not None:
                    self.sigLoadRequested.emit(files, loader_name)
                    logger.debug("Emitted file and loader info")
                else:
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


def _send_dict(contents: dict[str, typing.Any]) -> None:
    contents = pickle.dumps(contents, protocol=-1)

    logger.debug("Connecting to server")
    client_socket = socket.socket()
    client_socket.connect(("localhost", PORT))

    logger.debug("Sending data")
    # Send the size of the data first
    client_socket.sendall(struct.pack(">L", len(contents)))
    client_socket.sendall(contents)
    client_socket.close()

    logger.debug("Data sent successfully")


def load_in_manager(paths: typing.Iterable[str | os.PathLike], loader_name: str):
    """Load and display data in the ImageToolManager.

    Parameters
    ----------
    paths
        List of paths containing the data to be displayed in the ImageTool window.
    loader_name
        Name of the loader to use to load the data. The loader must be registered in
        :attr:`erlab.io.loaders`.

    """
    if not erlab.interactive.imagetool.manager.is_running():
        raise RuntimeError(
            "ImageToolManager is not running. Please start the ImageToolManager "
            "application before using this function"
        )

    path_list: list[str] = []
    for p in paths:
        path = pathlib.Path(p)
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        path_list.append(str(path))

    loader: str = erlab.io.loaders[
        loader_name
    ].name  # Trigger exception if loader_name is not registered

    if (
        erlab.interactive.imagetool.manager._manager_instance is not None
        and not erlab.interactive.imagetool.manager._always_use_socket
    ):
        # If the manager is running in the same process, directly pass the data
        erlab.interactive.imagetool.manager._manager_instance._data_load(
            path_list, loader
        )
        return

    _send_dict({"__filename": path_list, "__loader_name": loader})


def show_in_manager(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
    **kwargs,
) -> None:
    """Create and display ImageTool windows in the ImageToolManager.

    Parameters
    ----------
    data
        The data to be displayed in the ImageTool window. See :func:`itool
        <erlab.interactive.imagetool.itool>` for more information.
    link
        Whether to enable linking between multiple ImageTool windows, by default
        `False`.
    link_colors
        Whether to link the color maps between multiple linked ImageTool windows, by
        default `True`.
    **kwargs
        Keyword arguments passed onto :class:`ImageTool
        <erlab.interactive.imagetool.ImageTool>`.

    """
    if not erlab.interactive.imagetool.manager.is_running():
        raise RuntimeError(
            "ImageToolManager is not running. Please start the ImageToolManager "
            "application before using this function"
        )

    logger.debug("Parsing input data into DataArrays")

    if isinstance(data, xr.Dataset) and _ITOOL_DATA_NAME in data:
        # Dataset created with ImageTool.to_dataset()
        input_data: list[xr.DataArray] | list[xr.Dataset] = [data]
    else:
        input_data = erlab.interactive.imagetool.core._parse_input(data)

    if (
        erlab.interactive.imagetool.manager._manager_instance is not None
        and not erlab.interactive.imagetool.manager._always_use_socket
    ):
        # If the manager is running in the same process, directly pass the data
        erlab.interactive.imagetool.manager._manager_instance._data_recv(
            input_data, kwargs
        )
        return

    # Save the data to a temporary file
    logger.debug("Pickling data to temporary files")
    tmp_dir = tempfile.mkdtemp(prefix="erlab_manager_")

    files: list[str] = []

    for dat in input_data:
        fname = str(uuid.uuid4())
        fname = os.path.join(tmp_dir, fname)
        _save_pickle(dat, fname)
        files.append(fname)

    kwargs["__filename"] = files

    _send_dict(kwargs)
