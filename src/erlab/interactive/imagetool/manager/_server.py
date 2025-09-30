"""Server that listens to incoming data."""

from __future__ import annotations

__all__ = [
    "HOST_IP",
    "PORT",
    "PORT_WATCH",
    "fetch",
    "is_running",
    "load_in_manager",
    "replace_data",
    "show_in_manager",
    "unwatch_data",
    "watch_data",
]

import errno
import functools
import io
import logging
import os
import pathlib
import pickle
import threading
import typing

import numpy as np
import pydantic
import xarray as xr
import zmq
from qtpy import QtCore

import erlab
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME

if typing.TYPE_CHECKING:
    import types
    from collections.abc import Callable, Collection, Iterable

    import numpy.typing as npt


if np.lib.NumpyVersion(np.__version__) < "2.3.0":  # pragma: no cover
    # Patch numpy to add compatibility for cases where ImageToolManager is running on
    # numpy <2.3 and the client is using numpy >=2.3.

    # This whole block can be removed when we drop support for numpy <2.3

    import importlib

    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        _numeric: types.ModuleType = importlib.import_module("numpy._core.numeric")
    else:
        _numeric = importlib.import_module("numpy.core.numeric")

    _orig_frombuffer = _numeric._frombuffer

    def _frombuffer_compat(buf, dtype, shape, order, axis_order=None):
        if order == "K" and axis_order is not None:
            return (
                np.frombuffer(buf, dtype=dtype)
                .reshape(shape, order="C")
                .transpose(axis_order)
            )
        return _orig_frombuffer(buf, dtype, shape, order)

    _numeric._frombuffer = _frombuffer_compat  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


PORT: int = int(os.getenv("ITOOL_MANAGER_PORT", "45555"))
"""Port number for the manager server.

The default port number 45555 can be overridden by setting the environment variable
``ITOOL_MANAGER_PORT``.
"""

PORT_WATCH: int = int(os.getenv("ITOOL_MANAGER_PORT_WATCH", "45556"))
"""Port number for the manager server's watch channel.

The default port number 45556 can be overridden by setting the environment variable
``ITOOL_MANAGER_PORT_WATCH``.
"""

HOST_IP: str = str(os.getenv("ITOOL_MANAGER_HOST", "localhost"))
"""Host IP address for the manager server.

The default host IP address "localhost" can be overridden by setting the environment
variable ``ITOOL_MANAGER_HOST``, enabling remote connections.
"""


class _BasePacket(pydantic.BaseModel):
    """Base class for packets sent to the server."""

    model_config = {"arbitrary_types_allowed": True}


class AddDataPacket(_BasePacket):
    """Packet structure for adding new data to the manager."""

    packet_type: typing.Literal["add"]
    data_list: list[xr.DataArray] | list[xr.Dataset]
    arguments: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class OpenFilesPacket(_BasePacket):
    """Packet structure for opening files in the manager."""

    packet_type: typing.Literal["open"]
    filename_list: list[str]
    loader_name: str
    arguments: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class ReplacePacket(_BasePacket):
    """Packet structure for replacing data in existing ImageTool windows."""

    packet_type: typing.Literal["replace"]
    data_list: list[xr.DataArray]
    replace_idxs: list[int]


class WatchPacket(_BasePacket):
    """Packet structure for watching a variable."""

    packet_type: typing.Literal["watch"]
    watched_info: tuple[str, str]
    watched_data: xr.DataArray


class CommandPacket(_BasePacket):
    """Packet structure for sending commands to the server."""

    packet_type: typing.Literal["command"]
    command: typing.Literal[
        "ping",
        "get-data",
        "remove-idx",
        "show-idx",
        "remove-uid",
        "show-uid",
        "unwatch-uid",
    ]
    command_arg: str | int | None = None


PacketVariant = typing.Annotated[
    AddDataPacket | OpenFilesPacket | ReplacePacket | WatchPacket | CommandPacket,
    pydantic.Field(discriminator="packet_type"),
]

Packet: pydantic.TypeAdapter = pydantic.TypeAdapter(PacketVariant)


class Response(pydantic.BaseModel):
    """Response for server replies."""

    status: typing.Literal["ok", "error"]
    data: xr.DataArray | None = None

    model_config = {"arbitrary_types_allowed": True}


def _send_multipart(sock: zmq.Socket, obj: typing.Any, **kwargs) -> None:
    """Send a Python object as a multipart ZeroMQ message using pickle protocol 5."""
    buffers: list[pickle.PickleBuffer] = []  # out-of-band frames will be appended here
    bio = io.BytesIO()
    p = pickle.Pickler(bio, protocol=5, buffer_callback=buffers.append)
    p.dump(obj)
    header = memoryview(bio.getbuffer())
    frames = [header] + [memoryview(b) for b in buffers]
    sock.send_multipart(frames, copy=False, **kwargs)


def _recv_multipart(sock: zmq.Socket, **kwargs) -> typing.Any:
    """Receive a multipart ZeroMQ message and reconstruct the Python object."""
    parts = sock.recv_multipart(copy=False, **kwargs)
    return pickle.loads(parts[0].buffer, buffers=(p.buffer for p in parts[1:]))


def _query_zmq(payload: PacketVariant) -> Response:
    """Client side function to send a packet to the server and receive a response.

    Parameters
    ----------
    payload
        The packet to send to the server. The packet formats are defined as pydantic
        models.

    """
    ctx = zmq.Context.instance()

    sock: zmq.Socket = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    try:
        logger.info("Connecting to server...")
        sock.connect(f"tcp://{HOST_IP}:{PORT}")
    except Exception:
        logger.exception("Failed to connect to server")
    else:
        logger.debug("Sending request %s", payload)
        _send_multipart(sock, payload.model_dump(exclude_unset=True))
        # Wait for a response
        try:
            response = Response(**_recv_multipart(sock))
        except Exception:
            logger.exception("Failed to receive response from server")
            response = Response(status="error")
    finally:
        sock.close()

    return response


_UNSET = object()


class _WatcherServer(QtCore.QThread):
    def __init__(self) -> None:
        super().__init__()
        self.stopped = threading.Event()

        self._ret_val: typing.Any = _UNSET
        self._mutex = QtCore.QMutex()
        self._cv = QtCore.QWaitCondition()

    @QtCore.Slot(str, str, str)
    def send_parameters(
        self,
        varname: str,
        uid: str,
        event: typing.Literal["updated", "removed", "shutdown"],
    ) -> None:
        with QtCore.QMutexLocker(self._mutex):
            self._ret_val = (varname, uid, event)
            self._cv.wakeAll()

    def run(self) -> None:
        self.stopped.clear()
        logger.debug("Starting watcher server...")

        ctx = zmq.Context.instance()
        sock: zmq.Socket = ctx.socket(zmq.PUB)

        try:
            sock.bind(f"tcp://*:{PORT_WATCH}")
            logger.info("Watcher server is listening...")

            while not self.stopped.is_set():
                with QtCore.QMutexLocker(self._mutex):
                    while self._ret_val is _UNSET:
                        self._cv.wait(self._mutex)
                    varname, uid, event = self._ret_val
                    self._ret_val = _UNSET

                if event == "shutdown":
                    break

                logger.debug(
                    "Sending watched variable update for %s:%s[%s]", varname, uid, event
                )
                sock.send_json({"varname": varname, "uid": uid, "event": event})

        except Exception:
            logger.exception("Watcher server encountered an error")
        finally:
            sock.close()
            logger.debug("Watcher socket closed")


class _ManagerServer(QtCore.QThread):
    sigReceived = QtCore.Signal(list, dict)
    sigLoadRequested = QtCore.Signal(list, str, dict)
    sigReplaceRequested = QtCore.Signal(list, list)
    sigWatchedVarChanged = QtCore.Signal(str, str, object)

    sigDataRequested = QtCore.Signal(object)
    sigRemoveIndex = QtCore.Signal(int)
    sigShowIndex = QtCore.Signal(int)
    sigRemoveUID = QtCore.Signal(str)
    sigShowUID = QtCore.Signal(str)
    sigUnwatchUID = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.stopped = threading.Event()

        self._ret_val: typing.Any = _UNSET
        self._mutex = QtCore.QMutex()
        self._cv = QtCore.QWaitCondition()

    @QtCore.Slot(object)
    def set_return_value(self, value: typing.Any) -> None:
        with QtCore.QMutexLocker(self._mutex):
            self._ret_val = value
            self._cv.wakeAll()

    def run(self) -> None:
        self.stopped.clear()
        logger.debug("Starting server...")

        ctx = zmq.Context.instance()
        sock: zmq.Socket = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.RCVHWM, 0)

        try:
            sock.bind(f"tcp://*:{PORT}")
            logger.info("Server is listening...")

            while not self.stopped.is_set():
                try:
                    payload = Packet.validate_python(
                        _recv_multipart(sock, flags=zmq.NOBLOCK)
                    )
                except zmq.Again:
                    self.msleep(10)
                    continue
                except Exception:
                    logger.exception("Failed to parse incoming packet")
                    _send_multipart(sock, {"status": "error"})
                    continue

                logger.debug("Received payload with type %s", payload.packet_type)
                match payload.packet_type:
                    case "add":
                        self.sigReceived.emit(payload.data_list, payload.arguments)
                    case "open":
                        self.sigLoadRequested.emit(
                            payload.filename_list,
                            payload.loader_name,
                            payload.arguments,
                        )
                    case "watch":
                        self.sigWatchedVarChanged.emit(
                            *payload.watched_info, payload.watched_data
                        )
                    case "replace":
                        self.sigReplaceRequested.emit(
                            payload.data_list, payload.replace_idxs
                        )
                    case "command":
                        logger.info("Processing command: %s", payload.command)
                        match payload.command:
                            case "ping":
                                pass
                            case "get-data":
                                self.sigDataRequested.emit(payload.command_arg)
                                logger.info("Getting data...")
                                with QtCore.QMutexLocker(self._mutex):
                                    while self._ret_val is _UNSET:
                                        self._cv.wait(self._mutex)
                                    data = self._ret_val
                                    self._ret_val = _UNSET

                                if data.chunks is not None:
                                    data = data.compute()

                                logger.info("Data obtained, sending response...")
                                _send_multipart(sock, {"status": "ok", "data": data})
                                logger.debug("Response sent")

                                continue
                            case "remove-idx":
                                self.sigRemoveIndex.emit(int(payload.command_arg))
                            case "show-idx":
                                self.sigShowIndex.emit(int(payload.command_arg))
                            case "remove-uid":
                                self.sigRemoveUID.emit(str(payload.command_arg))
                            case "show-uid":
                                self.sigShowUID.emit(str(payload.command_arg))
                            case "unwatch-uid":
                                self.sigUnwatchUID.emit(str(payload.command_arg))

                logger.info("Sending response...")
                _send_multipart(sock, {"status": "ok"})
                logger.debug("Response sent")

        except Exception:
            logger.exception("Server encountered an error")
        finally:
            sock.close()
            logger.debug("Socket closed")


def _ping_server(timeout_ms: int = 100) -> bool:
    """Ping the ImageToolManager server to check if it is running."""
    ctx = zmq.Context.instance()
    sock: zmq.Socket = ctx.socket(zmq.REQ)
    # Timeouts in milliseconds
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        sock.connect(f"tcp://{HOST_IP}:{PORT}")
        # Send a minimal ping command and expect an OK
        _send_multipart(sock, {"packet_type": "command", "command": "ping"})
        return Response(**_recv_multipart(sock)).status == "ok"
    except Exception:
        return False
    finally:
        sock.close()


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


def _manager_running(func: Callable) -> Callable:
    """Decorate a function to ensure that the ImageToolManager is running."""

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if not erlab.interactive.imagetool.manager.is_running():
            raise RuntimeError(
                "ImageToolManager is not running. Please start the ImageToolManager "
                "application before using this function"
            )
        return func(*args, **kwargs)

    return _wrapper


@_manager_running
def load_in_manager(
    paths: Iterable[str | os.PathLike], loader_name: str | None = None, **load_kwargs
) -> Response | None:
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
    path_list: list[str] = []
    for p in paths:
        path = pathlib.Path(p)
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        path_list.append(str(path))

    if loader_name is None:
        loader: str = "ask"
    else:
        loader = erlab.io.loaders[
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
        return None

    return _query_zmq(
        OpenFilesPacket(
            packet_type="open",
            filename_list=path_list,
            loader_name=loader,
            arguments=load_kwargs,
        )
    )


@_manager_running
def show_in_manager(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree
    | None,
    **kwargs,
) -> Response | None:
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
    logger.debug("Parsing input data into DataArrays")

    if isinstance(data, xr.Dataset) and _ITOOL_DATA_NAME in data:
        # Dataset created with ImageTool.to_dataset()
        input_data: list[xr.DataArray] | list[xr.Dataset] = [data]
    elif data is None:
        input_data = []
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
        return None

    return _query_zmq(
        AddDataPacket(packet_type="add", data_list=input_data, arguments=kwargs)
    )


@_manager_running
def replace_data(
    index: int | Collection[int],
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
) -> Response:
    """Replace data in existing ImageTool windows.

    Parameters
    ----------
    index
        Index or indices of the ImageTool windows to replace data in. If ``data``
        corresponds to a single ImageTool window, for instance a single DataArray,
        ``index`` is allowed to be a single integer. If ``data`` corresponds to multiple
        ImageTool windows, for instance a list of DataArrays, ``index`` must be a list
        of integers with the same length as ``data``. All indices must be valid
        indices of existing ImageTool windows.
    data
        Data to replace the existing data in the ImageTool windows. See :func:`itool
        <erlab.interactive.imagetool.itool>` for more information.
    """
    data_list = erlab.interactive.imagetool.core._parse_input(data)

    if isinstance(index, int):
        index = [index]

    if len(data_list) != len(index):
        raise ValueError(
            f"Mismatch between number of data ({len(data_list)}) "
            f"and provided indices ({len(index)})"
        )

    return _query_zmq(
        ReplacePacket(
            packet_type="replace", data_list=data_list, replace_idxs=list(index)
        )
    )


@_manager_running
def watch_data(varname: str, uid: str, data: xr.DataArray, show: bool = False) -> None:
    """Add or update a watched variable in the ImageToolManager.

    Parameters
    ----------
    varname
        Name of the watched variable.
    uid
        Unique identifier for the watched variable.
    data
        New data for the watched variable.
    show
        If `True`, bring the corresponding ImageTool window to the front. Default is
        `False`. If this is a new variable being watched, a new ImageTool window will be
        created and shown regardless of this parameter.
    """
    if data.chunks is not None:
        # TODO: there might be a better way to handle this like directly serializing as
        # dask, but for now just compute to numpy
        data = data.compute()

    _query_zmq(
        WatchPacket(packet_type="watch", watched_info=(varname, uid), watched_data=data)
    )
    if show:
        _query_zmq(
            CommandPacket(packet_type="command", command="show-uid", command_arg=uid)
        )


@_manager_running
def unwatch_data(uid: str, remove: bool = False) -> Response:
    """Cancel watching a variable in the ImageToolManager.

    Parameters
    ----------
    uid
        Unique identifier for the watched variable.
    remove
        Whether to remove the corresponding ImageTool window when unwatching. Default is
        `False`.
    """
    if remove:
        return _query_zmq(
            CommandPacket(packet_type="command", command="remove-uid", command_arg=uid)
        )
    return _query_zmq(
        CommandPacket(packet_type="command", command="unwatch-uid", command_arg=uid)
    )


def _remove_idx(index: int) -> Response:
    """Remove the ImageTool window at the given index.

    Parameters
    ----------
    index
        Index of the ImageTool window to close.
    """
    return _query_zmq(
        CommandPacket(packet_type="command", command="remove-idx", command_arg=index)
    )


def _show_idx(index: int) -> Response:
    """Show the ImageTool window at the given index.

    Parameters
    ----------
    index
        Index of the ImageTool window to show.
    """
    return _query_zmq(
        CommandPacket(packet_type="command", command="show-idx", command_arg=index)
    )


@_manager_running
def fetch(index: int | str) -> xr.DataArray | None:
    """Get data from the ImageTool window at the given index.

    Parameters
    ----------
    index
        Index of the ImageTool window to get data from, or the unique identifier (UID)
        of a watched variable (used internally).

    Returns
    -------
    xr.DataArray or None
        The data in the ImageTool window at the given index, or `None` if the index is
        invalid or the data cannot be retrieved.
    """
    if (
        erlab.interactive.imagetool.manager._manager_instance is not None
        and not erlab.interactive.imagetool.manager._always_use_socket
    ):
        return (
            erlab.interactive.imagetool.manager._manager_instance._get_imagetool_data(
                index
            )
        )
    return _query_zmq(
        CommandPacket(packet_type="command", command="get-data", command_arg=index)
    ).data
