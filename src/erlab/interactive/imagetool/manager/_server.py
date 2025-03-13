"""Server that listens to incoming data."""

from __future__ import annotations

__all__ = ["PORT", "is_running", "load_in_manager", "show_in_manager"]

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
    sigLoadRequested = QtCore.Signal(list, str, dict)

    def __init__(self) -> None:
        super().__init__()
        self.stopped = threading.Event()

    def _handle_client(self, conn: socket.socket) -> None:
        with conn:
            # If no data is received in 2 seconds, close the connection
            conn.settimeout(2.0)

            # Receive the size of the data first
            data_size_bytes = _recv_all(conn, 4)
            data_size = struct.unpack(">L", data_size_bytes)[0]

            # Receive the actual data
            kwargs_bytes = _recv_all(conn, data_size)

            try:
                kwargs: dict = pickle.loads(kwargs_bytes)
                logger.debug("Received data: %s", kwargs)

                if "__ping" in kwargs:
                    logger.debug("Received ping")
                    return

                # kwargs["__filename"] contains the list of paths to pickled data

                # If "__loader_name" key is present, kwargs["__filename"] will be a list
                # of paths to raw data files instead

                files = kwargs.pop("__filename")
                loader_name = kwargs.pop("__loader_name", None)

                if loader_name is not None:
                    self.sigLoadRequested.emit(files, loader_name, kwargs)
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

    def run(self) -> None:
        self.stopped.clear()

        logger.debug("Starting server...")
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            soc.bind(("127.0.0.1", PORT))
            soc.setblocking(True)
            soc.listen()
            logger.info("Server is listening...")

            while not self.stopped.is_set():
                try:
                    conn, _ = soc.accept()
                except Exception:
                    logger.exception("Unexpected error while accepting connection")
                    continue
                else:
                    logger.debug("Connection accepted")
                    threading.Thread(
                        target=self._handle_client, args=(conn,), daemon=True
                    ).start()
        except Exception:
            logger.exception("Server encountered an error")
        finally:
            soc.close()
            logger.debug("Socket closed")


def _send_dict(contents: dict[str, typing.Any]) -> None:
    contents = pickle.dumps(contents, protocol=-1)

    logger.debug("Connecting to server")
    client_socket = socket.socket()

    try:
        client_socket.connect(("localhost", PORT))
    except Exception:
        logger.exception("Failed to connect to server")
    else:
        # Send the size of the data first
        logger.debug("Sending data")
        client_socket.sendall(struct.pack(">L", len(contents)))
        client_socket.sendall(contents)
    finally:
        client_socket.close()

    logger.debug("Data sent successfully")


def _ping_server() -> bool:
    """Ping the ImageToolManager server.

    Returns
    -------
    bool
        True if the ping was successful, False otherwise.
    """
    client_socket = socket.socket()

    to_send = pickle.dumps({"__ping": True}, protocol=-1)
    try:
        client_socket.connect(("localhost", PORT))
    except (TimeoutError, ConnectionRefusedError):
        return False
    else:
        client_socket.sendall(struct.pack(">L", len(to_send)))
        client_socket.sendall(to_send)
    finally:
        client_socket.close()
    return True


def is_running() -> bool:
    """Check whether an instance of ImageToolManager is active.

    Returns
    -------
    bool
        True if an instance of ImageToolManager is running, False otherwise.
    """
    if erlab.interactive.imagetool.manager._manager_instance is not None:
        return True
    return _ping_server()


def load_in_manager(
    paths: typing.Iterable[str | os.PathLike], loader_name: str, **load_kwargs
) -> None:
    """Load and display data in the ImageToolManager.

    Parameters
    ----------
    paths
        List of paths containing the data to be displayed in the ImageTool window.
    loader_name
        Name of the loader to use to load the data. The loader must be registered in
        :attr:`erlab.io.loaders`.
    **load_kwargs
        Additional keyword arguments passed onto the load method of the loader.
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
            path_list, loader, load_kwargs
        )
        return

    load_kwargs["__filename"] = path_list
    load_kwargs["__loader_name"] = loader

    _send_dict(load_kwargs)


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
        _save_pickle(dat.load(), fname)
        files.append(fname)

    kwargs["__filename"] = files

    _send_dict(kwargs)
