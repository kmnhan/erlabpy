"""Server that listens to incoming data."""

from __future__ import annotations

__all__ = [
    "HOST_IP",
    "PORT",
    "PORT_WATCH",
    "ImageToolManagerAmbiguousError",
    "ImageToolManagerHandle",
    "ImageToolManagerNotFoundError",
    "ImageToolManagerRegistry",
    "ImageToolManagerTimeoutError",
    "clear_default_manager",
    "fetch",
    "get_default_manager",
    "is_running",
    "load_in_manager",
    "manager_selection_info",
    "managers",
    "replace_data",
    "set_default_manager",
    "show_in_manager",
    "use_manager",
    "watch_info",
]


import errno
import html
import io
import logging
import os
import pathlib
import pickle
import threading
import typing
import warnings

import numpy as np
import pydantic
import xarray as xr
import zmq
import zmq.utils.monitor
from qtpy import QtCore

import erlab
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME
from erlab.interactive.imagetool.manager._registry import (
    ImageToolManagerAmbiguousError,
    ImageToolManagerNotFoundError,
    _ManagerRecord,
    _normalize_manager_index,
    clear_default_manager,
    get_default_manager,
    live_manager_records,
    manager_selection_info,
    resolve_manager_record,
    set_default_manager,
    use_manager,
)

if typing.TYPE_CHECKING:
    import types
    from collections.abc import Collection, Iterable, Iterator

    import numpy.typing as npt


if np.lib.NumpyVersion(np.__version__) < "2.3.0":  # pragma: no cover
    # Patch numpy to add compatibility for cases where ImageToolManager is running on
    # numpy <2.3 and the client is using numpy >=2.3.

    # Note: does not cover the case where ImageToolManager is running on numpy >=2.3
    # and the client is using numpy <2.3

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

ZMQ_TIMEOUT_MS: int = int(os.getenv("ITOOL_MANAGER_ZMQ_TIMEOUT_MS", "15000"))
"""Timeout (ms) for client-side ZMQ requests.

The default timeout is 15000 ms and can be overridden by setting the environment
variable ``ITOOL_MANAGER_ZMQ_TIMEOUT_MS``. Set to 0 or a negative value to disable
the timeout.
"""


class _BasePacket(pydantic.BaseModel):
    """Base class for packets sent to the server."""

    pickler_kind: typing.Literal["pickle", "cloudpickle"] = "pickle"

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }


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
    replace_idxs: list[int | str]


class WatchPacket(_BasePacket):
    """Packet structure for watching a variable."""

    packet_type: typing.Literal["watch"]
    watched_info: tuple[str, str]
    watched_data: xr.DataArray
    watched_metadata: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


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
        "watch-info",
    ]
    command_arg: str | int | None = None


class UnpickleFailedPacket(_BasePacket):
    """Packet structure for data that failed to unpickle."""

    packet_type: typing.Literal["unpickle-failed"]


PacketVariant = typing.Annotated[
    AddDataPacket
    | OpenFilesPacket
    | ReplacePacket
    | WatchPacket
    | CommandPacket
    | UnpickleFailedPacket,
    pydantic.Field(discriminator="packet_type"),
]

Packet: pydantic.TypeAdapter = pydantic.TypeAdapter(PacketVariant)


class Response(pydantic.BaseModel):
    """Response for server replies."""

    status: typing.Literal["ok", "error", "unpickle-failed"]
    data: xr.DataArray | None = None
    watch_info: dict[str, typing.Any] | None = None
    pickler_kind: typing.Literal["pickle", "cloudpickle"] = "pickle"

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }


def _send_multipart(
    sock: zmq.Socket,
    obj: dict[str, typing.Any],
    *,
    use_cloudpickle: bool = False,
    max_frame_size: int = 2**30,
    **kwargs,
) -> None:
    """Send a Python object as a multipart ZeroMQ message using pickle protocol 5.

    Parameters
    ----------
    max_frame_size
        Upper bound for each multipart frame. Messages exceeding this are split into
        smaller chunks. Defaults to 2**30 to stay below the 2 GiB ZMQ ceiling.
    """
    pickler_cls = pickle.Pickler
    if use_cloudpickle:
        import cloudpickle

        pickler_cls = cloudpickle.Pickler

    buffers: list[pickle.PickleBuffer] = []  # out-of-band frames will be appended here
    bio = io.BytesIO()

    p = pickler_cls(bio, protocol=5, buffer_callback=buffers.append)
    try:
        p.dump(obj)
    except Exception:
        if use_cloudpickle:
            raise
        # Retry with cloudpickle
        _send_multipart(sock, obj, use_cloudpickle=True, **kwargs)
        return

    pickler_kind = b"cloudpickle" if use_cloudpickle else b"pickle"
    header = memoryview(bio.getbuffer())
    frames: list[memoryview[int] | bytes] = [pickler_kind, header]
    frames.extend(memoryview(buffer) for buffer in buffers)

    def _normalize_frame(frame: memoryview | bytes) -> memoryview | bytes:
        if not isinstance(frame, memoryview):
            return frame
        if frame.ndim == 1 and frame.format == "B":
            return frame
        try:
            return frame.cast("B")
        except TypeError:
            return memoryview(bytes(frame))

    def _frame_size(frame: memoryview | bytes) -> int:
        return frame.nbytes if isinstance(frame, memoryview) else len(frame)

    normalized_frames = [_normalize_frame(frame) for frame in frames]

    if all(_frame_size(frame) <= max_frame_size for frame in normalized_frames):
        sock.send_multipart(normalized_frames, copy=False, **kwargs)
        return

    chunked_frames: list[memoryview | bytes] = [pickler_kind, memoryview(b"chunked-v1")]
    for frame in normalized_frames[1:]:
        frame_len = _frame_size(frame)
        chunk_count = (frame_len + max_frame_size - 1) // max_frame_size
        chunk_header = memoryview(f"{chunk_count}:{frame_len}".encode())
        chunked_frames.append(chunk_header)
        start = 0
        for _ in range(chunk_count):
            end = min(start + max_frame_size, frame_len)
            chunked_frames.append(frame[start:end])
            start = end

    sock.send_multipart(chunked_frames, copy=False, **kwargs)


def _recv_multipart(sock: zmq.Socket, **kwargs) -> dict[str, typing.Any]:
    """Receive a multipart ZeroMQ message and reconstruct the Python object."""
    parts = sock.recv_multipart(copy=False, **kwargs)
    pickler_kind = b"pickle"
    if parts[0].bytes in {b"pickle", b"cloudpickle"}:
        pickler_kind = parts[0].bytes
        parts = parts[1:]

    if parts and parts[0].bytes == b"chunked-v1":
        parts = parts[1:]
        rebuilt_parts: list[memoryview] = []
        idx = 0
        while idx < len(parts):
            try:
                chunk_info = parts[idx].bytes
                idx += 1
                chunk_count_str, total_len_str = chunk_info.split(b":", maxsplit=1)
                chunk_count = int(chunk_count_str)
                total_len = int(total_len_str)
            except Exception as exc:  # pragma: no cover - defensive parsing
                raise RuntimeError(
                    "Malformed chunk header in multipart message"
                ) from exc

            if idx + chunk_count > len(parts):  # pragma: no cover - defensive parsing
                raise RuntimeError("Incomplete chunked multipart message")

            if chunk_count == 1:
                rebuilt_parts.append(parts[idx].buffer)
                idx += 1
                continue

            combined = bytearray(total_len)
            offset = 0
            for _ in range(chunk_count):
                chunk = parts[idx].buffer
                combined[offset : offset + len(chunk)] = chunk
                offset += len(chunk)
                idx += 1
            rebuilt_parts.append(memoryview(combined))
        parts = rebuilt_parts

    def _as_buffer(frame: typing.Any) -> memoryview:
        if isinstance(frame, memoryview):
            return frame
        return memoryview(frame)

    try:
        payload_dict = pickle.loads(
            _as_buffer(parts[0]),
            buffers=(
                p.buffer if hasattr(p, "buffer") else _as_buffer(p) for p in parts[1:]
            ),
        )
    except (AttributeError, ModuleNotFoundError) as e:
        logger.info("Unpickling failed with %s: %s", type(e).__name__, e)
        payload_dict = {
            "packet_type": "unpickle-failed",
            "status": "unpickle-failed",
        }
    payload_dict["pickler_kind"] = pickler_kind.decode()
    return payload_dict


class ImageToolManagerTimeoutError(TimeoutError):
    """Custom timeout error for ImageToolManager operations."""

    def __init__(self) -> None:
        super().__init__(
            f"Timed out waiting for ImageToolManager response after "
            f"{ZMQ_TIMEOUT_MS} ms. Adjust via the environment variable "
            "ITOOL_MANAGER_ZMQ_TIMEOUT_MS."
        )


def _query_zmq(
    payload: PacketVariant,
    *,
    target: int | None = None,
    record: _ManagerRecord | None = None,
) -> Response:
    """Client side function to send a packet to the server and receive a response.

    Parameters
    ----------
    payload
        The packet to send to the server. The packet formats are defined as pydantic
        models.

    """
    record = resolve_manager_record(target) if record is None else record

    ctx = zmq.Context.instance()

    sock: zmq.Socket = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        logger.debug("Connecting to server...")
        sock.connect(f"tcp://{record.host}:{record.port}")
    except Exception:
        logger.exception("Failed to connect to server")
    else:
        logger.debug("Sending request %s", payload)
        payload_dict = payload.model_dump(exclude_unset=True)
        _send_multipart(sock, payload_dict)

        # Wait for a response
        if ZMQ_TIMEOUT_MS > 0 and not poller.poll(ZMQ_TIMEOUT_MS):
            raise ImageToolManagerTimeoutError
        response = Response(**_recv_multipart(sock))
        if response.status == "unpickle-failed":
            logger.debug("Retrying with cloudpickle...")
            _send_multipart(sock, payload_dict, use_cloudpickle=True)
            if ZMQ_TIMEOUT_MS > 0 and not poller.poll(ZMQ_TIMEOUT_MS):
                raise ImageToolManagerTimeoutError
            response = Response(**_recv_multipart(sock))
    finally:
        sock.close()

    return response


_UNSET = object()


class _WatcherServer(QtCore.QThread):
    def __init__(self, port: int | None = None) -> None:
        super().__init__()
        self.stopped = threading.Event()
        self.port = PORT_WATCH if port is None else port
        self._bound_event = threading.Event()
        self._bound_port: int | None = None
        self._bind_error: Exception | None = None

        self._ret_val: typing.Any = _UNSET
        self._mutex = QtCore.QMutex()
        self._cv = QtCore.QWaitCondition()

    def wait_until_bound(self, timeout_ms: int = 5000) -> int:
        if not self._bound_event.wait(timeout_ms / 1000):
            raise TimeoutError("Timed out waiting for watcher server to bind")
        if self._bind_error is not None:
            raise RuntimeError("Watcher server failed to bind") from self._bind_error
        if self._bound_port is None:  # pragma: no cover - defensive
            raise RuntimeError("Watcher server did not report a bound port")
        return self._bound_port

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

    def stop(self, timeout_ms: int = 5000) -> None:
        self.stopped.set()
        self.send_parameters("", "", "shutdown")
        if self.isRunning() and not self.wait(timeout_ms):
            logger.warning("Watcher server did not stop within timeout")

    def run(self) -> None:
        self.stopped.clear()
        logger.debug("Starting watcher server...")

        ctx = zmq.Context.instance()
        sock: zmq.Socket = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 0)

        try:
            try:
                if self.port == 0:
                    self.port = int(sock.bind_to_random_port("tcp://*"))
                else:
                    sock.bind(f"tcp://*:{self.port}")
            except Exception as exc:
                self._bind_error = exc
                self._bound_event.set()
                logger.debug("Watcher server failed to bind", exc_info=True)
                return
            self._bound_port = self.port
            self._bound_event.set()
            logger.debug("Watcher server is listening on port %s...", self.port)

            while not self.stopped.is_set():
                with QtCore.QMutexLocker(self._mutex):
                    while self._ret_val is _UNSET:
                        self._cv.wait(self._mutex, 100)
                        if self.stopped.is_set():
                            break
                    if self._ret_val is _UNSET:
                        continue
                    varname, uid, event = self._ret_val
                    self._ret_val = _UNSET

                if event == "shutdown":
                    break

                logger.debug(
                    "Sending watched variable update for %s:%s[%s]", varname, uid, event
                )
                sock.send_json({"varname": varname, "uid": uid, "event": event})

        except Exception as exc:
            if not self._bound_event.is_set():
                self._bind_error = exc
                self._bound_event.set()
            logger.exception("Watcher server encountered an error")
        finally:
            sock.close()
            logger.debug("Watcher socket closed")


class _ManagerServer(QtCore.QThread):
    sigReceived = QtCore.Signal(list, dict)
    sigLoadRequested = QtCore.Signal(list, str, dict)
    sigReplaceRequested = QtCore.Signal(list, list)
    sigWatchedVarChanged = QtCore.Signal(str, str, object, object)

    sigDataRequested = QtCore.Signal(object)
    sigWatchInfoRequested = QtCore.Signal()
    sigRemoveIndex = QtCore.Signal(int)
    sigShowIndex = QtCore.Signal(int)
    sigRemoveUID = QtCore.Signal(str)
    sigShowUID = QtCore.Signal(str)
    sigUnwatchUID = QtCore.Signal(str)

    def __init__(self, port: int | None = None) -> None:
        super().__init__()
        self.stopped = threading.Event()
        self.port = PORT if port is None else port
        self._bound_event = threading.Event()
        self._bound_port: int | None = None
        self._bind_error: Exception | None = None

        self._ret_val: typing.Any = _UNSET
        self._mutex = QtCore.QMutex()
        self._cv = QtCore.QWaitCondition()

    def wait_until_bound(self, timeout_ms: int = 5000) -> int:
        if not self._bound_event.wait(timeout_ms / 1000):
            raise TimeoutError("Timed out waiting for manager server to bind")
        if self._bind_error is not None:
            raise RuntimeError("Manager server failed to bind") from self._bind_error
        if self._bound_port is None:  # pragma: no cover - defensive
            raise RuntimeError("Manager server did not report a bound port")
        return self._bound_port

    @QtCore.Slot(object)
    def set_return_value(self, value: typing.Any) -> None:
        with QtCore.QMutexLocker(self._mutex):
            self._ret_val = value
            self._cv.wakeAll()

    def stop(self, timeout_ms: int = 5000) -> None:
        self.stopped.set()
        with QtCore.QMutexLocker(self._mutex):
            self._cv.wakeAll()
        if self.isRunning() and not self.wait(timeout_ms):
            logger.warning("Manager server did not stop within timeout")

    def _wait_for_return_value(self) -> typing.Any:
        with QtCore.QMutexLocker(self._mutex):
            while self._ret_val is _UNSET:
                if self.stopped.is_set():
                    break
                self._cv.wait(self._mutex, 100)
            value = self._ret_val
            self._ret_val = _UNSET
            return value

    def run(self) -> None:
        self.stopped.clear()
        logger.debug("Starting server...")

        ctx = zmq.Context.instance()
        sock: zmq.Socket = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        sock.setsockopt(zmq.RCVTIMEO, 100)

        try:
            try:
                if self.port == 0:
                    self.port = int(sock.bind_to_random_port("tcp://*"))
                else:
                    sock.bind(f"tcp://*:{self.port}")
            except Exception as exc:
                self._bind_error = exc
                self._bound_event.set()
                logger.debug("Server failed to bind", exc_info=True)
                return
            self._bound_port = self.port
            self._bound_event.set()
            logger.debug("Server is listening on port %s...", self.port)

            while not self.stopped.is_set():
                try:
                    payload = Packet.validate_python(_recv_multipart(sock))
                except zmq.Again:
                    continue
                except Exception:
                    logger.exception("Failed to parse incoming packet")
                    _send_multipart(sock, {"status": "error"})
                    continue
                else:
                    if (
                        payload.packet_type == "unpickle-failed"
                        and payload.pickler_kind == "pickle"
                    ):
                        _send_multipart(sock, {"status": "unpickle-failed"})
                        continue

                logger.debug("Received payload with type %s", payload.packet_type)
                if not (
                    payload.packet_type == "command"
                    and payload.command in {"get-data", "watch-info"}
                ):
                    # For commands without return payloads, send an immediate OK
                    # response before dispatching work to the GUI thread.
                    logger.debug("Sending response...")
                    _send_multipart(sock, {"status": "ok"})
                    logger.debug("Response sent")

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
                            *payload.watched_info,
                            payload.watched_data,
                            payload.watched_metadata,
                        )
                    case "replace":
                        self.sigReplaceRequested.emit(
                            payload.data_list, payload.replace_idxs
                        )
                    case "command":
                        logger.debug("Processing command: %s", payload.command)
                        match payload.command:
                            case "get-data":
                                self.sigDataRequested.emit(payload.command_arg)
                                logger.debug("Getting data...")
                                data = self._wait_for_return_value()
                                if data is _UNSET:
                                    logger.debug(
                                        "Server stopping while waiting for data"
                                    )
                                    _send_multipart(sock, {"status": "error"})
                                    break

                                logger.debug("Data obtained, sending response...")
                                _send_multipart(
                                    sock,
                                    {"status": "ok", "data": data},
                                    use_cloudpickle=bool(
                                        payload.pickler_kind == "cloudpickle"
                                    ),
                                )
                                logger.debug("Response sent")

                                continue
                            case "watch-info":
                                self.sigWatchInfoRequested.emit()
                                logger.debug("Getting watched-variable info...")
                                info = self._wait_for_return_value()
                                if info is _UNSET:
                                    logger.debug(
                                        "Server stopping while waiting for "
                                        "watched-variable info"
                                    )
                                    _send_multipart(sock, {"status": "error"})
                                    break

                                logger.debug(
                                    "Watched-variable info obtained, sending "
                                    "response..."
                                )
                                _send_multipart(
                                    sock,
                                    {"status": "ok", "watch_info": info},
                                )
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

        except Exception as exc:
            if not self._bound_event.is_set():
                self._bind_error = exc
                self._bound_event.set()
            logger.exception("Server encountered an error")
        finally:
            sock.close()
            logger.debug("Socket closed")


def _direct_manager_for_target(target: int | None):
    if target is not None:
        target = _normalize_manager_index(target, label="target")
    manager = erlab.interactive.imagetool.manager._manager_instance
    if manager is None or erlab.interactive.imagetool.manager._always_use_socket:
        return None
    local_index = getattr(manager, "manager_index", None)
    if target is not None:
        return manager if local_index == target else None
    default_index = get_default_manager(validate=False)
    if default_index is not None and default_index != local_index:
        return None
    return manager


class ImageToolManagerHandle:
    """Handle for a live ImageTool manager."""

    index: int
    pid: int
    host: str
    port: int
    watch_port: int
    started: str
    version: str
    workspace_path: str | None
    is_default: bool

    def __init__(
        self,
        *,
        index: int,
        pid: int,
        host: str,
        port: int,
        watch_port: int,
        started: str,
        version: str,
        workspace_path: str | None = None,
        is_default: bool = False,
    ) -> None:
        self.index = index
        self.pid = pid
        self.host = host
        self.port = port
        self.watch_port = watch_port
        self.started = started
        self.version = version
        self.workspace_path = workspace_path
        self.is_default = is_default

    @classmethod
    def _from_record(
        cls, record: _ManagerRecord, *, is_default: bool = False
    ) -> ImageToolManagerHandle:
        return cls(
            index=record.index,
            pid=record.pid,
            host=record.host,
            port=record.port,
            watch_port=record.watch_port,
            started=record.started,
            version=record.version,
            workspace_path=record.workspace_path,
            is_default=is_default,
        )

    @property
    def endpoint(self) -> str:
        """Request endpoint used by this manager."""
        return f"{self.host}:{self.port}"

    def __repr__(self) -> str:
        default = ", default" if self.is_default else ""
        workspace = (
            "" if self.workspace_path is None else f" workspace={self.workspace_path!r}"
        )
        return (
            f"<ImageToolManager #{self.index}{default} {self.endpoint} "
            f"pid={self.pid}{workspace}>"
        )

    def show(self, data: typing.Any, **kwargs: typing.Any) -> Response | None:
        """Create ImageTool windows in this manager."""
        return erlab.interactive.imagetool.manager.show_in_manager(
            data, target=self.index, **kwargs
        )

    def load(
        self,
        paths: Iterable[str | os.PathLike],
        loader_name: str | None = None,
        **load_kwargs: typing.Any,
    ) -> Response | None:
        """Load files into this manager."""
        return erlab.interactive.imagetool.manager.load_in_manager(
            paths, loader_name=loader_name, target=self.index, **load_kwargs
        )

    def replace(
        self,
        index: int | str | Collection[int | str],
        data: typing.Any,
    ) -> Response:
        """Replace ImageTool data in this manager."""
        return erlab.interactive.imagetool.manager.replace_data(
            index, data, target=self.index
        )

    def fetch(self, index: int | str) -> xr.DataArray | None:
        """Fetch ImageTool data from this manager."""
        return erlab.interactive.imagetool.manager.fetch(index, target=self.index)

    def watch(self, *varnames: str, **kwargs: typing.Any) -> tuple[str, ...]:
        """Watch variables in this manager."""
        return erlab.interactive.imagetool.manager.watch(
            *varnames, target=self.index, **kwargs
        )

    def watch_info(self) -> dict[str, typing.Any]:
        """Return watched-variable metadata for this manager."""
        return erlab.interactive.imagetool.manager.watch_info(target=self.index)

    def use(self) -> int:
        """Set this manager as the default for the current Python process."""
        return erlab.interactive.imagetool.manager.set_default_manager(self.index)


class ImageToolManagerRegistry:
    """Live ImageTool manager registry.

    The registry is a lightweight view over the live manager registry. It refreshes
    whenever it is displayed, iterated, or indexed.

    .. versionadded:: 3.22.0
    """

    def _snapshot(self) -> tuple[ImageToolManagerHandle, ...]:
        default_index = get_default_manager(validate=False)
        return tuple(
            ImageToolManagerHandle._from_record(
                record, is_default=record.index == default_index
            )
            for record in live_manager_records()
        )

    def __iter__(self) -> Iterator[ImageToolManagerHandle]:
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, index: int) -> ImageToolManagerHandle:
        index = _normalize_manager_index(index, label="index")
        for manager in self._snapshot():
            if manager.index == index:
                return manager
        raise KeyError(index)

    def keys(self) -> tuple[int, ...]:
        """Return live manager indexes."""
        return tuple(manager.index for manager in self._snapshot())

    def values(self) -> tuple[ImageToolManagerHandle, ...]:
        """Return live manager handles."""
        return self._snapshot()

    def items(self) -> tuple[tuple[int, ImageToolManagerHandle], ...]:
        """Return ``(index, handle)`` pairs for live managers."""
        return tuple((manager.index, manager) for manager in self._snapshot())

    def _table_rows(self) -> list[tuple[str, ...]]:
        rows: list[tuple[str, ...]] = [
            (
                "Index",
                "Default",
                "Workspace",
                "PID",
                "Endpoint",
                "Watch Port",
                "Started",
                "Version",
            )
        ]
        rows.extend(
            (
                f"#{manager.index}",
                "yes" if manager.is_default else "",
                manager.workspace_path or "",
                str(manager.pid),
                manager.endpoint,
                str(manager.watch_port),
                manager.started,
                manager.version,
            )
            for manager in self._snapshot()
        )
        return rows

    def __repr__(self) -> str:
        rows = self._table_rows()
        if len(rows) == 1:
            return "No live ImageTool managers."

        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        formatted_rows = [
            " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            for row in rows
        ]
        separator = "-+-".join("-" * width for width in widths)
        return "\n".join([formatted_rows[0], separator, *formatted_rows[1:]])

    def _repr_html_(self) -> str:
        rows = self._table_rows()
        if len(rows) == 1:
            return "<p>No live ImageTool managers.</p>"
        escaped_rows = [tuple(html.escape(cell) for cell in row) for row in rows]
        return erlab.utils.formatting.format_html_table(escaped_rows, header_rows=1)


managers: ImageToolManagerRegistry = ImageToolManagerRegistry()
"""Live ImageTool manager registry."""


def is_running(target: int | None = None) -> bool:
    """Check whether an instance of ImageToolManager is active.

    Parameters
    ----------
    target
        Optional 0-based manager index to check. If omitted, return ``True`` when
        any manager is live.

    Returns
    -------
    bool
        True if an instance of ImageToolManager is running, False otherwise.
    """
    if target is not None:
        target = _normalize_manager_index(target, label="target")
    manager = erlab.interactive.imagetool.manager._manager_instance
    local_index = (
        getattr(manager, "manager_index", None) if manager is not None else None
    )
    if local_index is not None and (target is None or local_index == target):
        return True
    records = live_manager_records()
    if target is not None:
        return any(record.index == target for record in records)
    return bool(records)


def load_in_manager(
    paths: Iterable[str | os.PathLike],
    loader_name: str | None = None,
    *,
    target: int | None = None,
    **load_kwargs,
) -> Response | None:
    """Load and display data in the ImageToolManager.

    Parameters
    ----------
    paths
        List of paths containing the data to be displayed in the ImageTool window.
    loader_name
        Name of the loader to use to load the data. The loader must be registered in
        :attr:`erlab.io.loaders`.
    target
        Optional 0-based manager index. If omitted, the current process default is
        used. If no default is set, the only live manager is used. If multiple
        managers are live, an ambiguity error is raised.
    **load_kwargs
        Additional keyword arguments passed onto the load method of the loader.

    Returns
    -------
    Response or None
        Manager response for socket calls. Returns ``None`` when the manager is in
        the same Python process and the data is passed directly.

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

    direct_manager = _direct_manager_for_target(target)
    if direct_manager is not None:
        # If the manager is running in the same process, directly pass the data
        direct_manager._data_load(path_list, loader, load_kwargs)
        return None

    record = resolve_manager_record(target)
    return _query_zmq(
        OpenFilesPacket(
            packet_type="open",
            filename_list=path_list,
            loader_name=loader,
            arguments=load_kwargs,
        ),
        record=record,
    )


def show_in_manager(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree
    | None,
    target: int | None = None,
    **kwargs,
) -> Response | None:
    """Create and display ImageTool windows in the ImageToolManager.

    Parameters
    ----------
    data
        The data to be displayed in the ImageTool window. See :func:`itool
        <erlab.interactive.imagetool.itool>` for more information.
    target
        Optional 0-based manager index. If omitted, the current process default is
        used. If no default is set, the only live manager is used. If multiple
        managers are live, an ambiguity error is raised.
    link
        Whether to enable linking between multiple ImageTool windows, by default
        `False`.
    link_colors
        Whether to link the color maps between multiple linked ImageTool windows, by
        default `True`.
    **kwargs
        Keyword arguments passed onto :class:`ImageTool
        <erlab.interactive.imagetool.ImageTool>`.

    Returns
    -------
    Response or None
        Manager response for socket calls. Returns ``None`` when the manager is in
        the same Python process and the data is passed directly.

    """
    logger.debug("Parsing input data into DataArrays")

    if isinstance(data, xr.Dataset) and _ITOOL_DATA_NAME in data:
        # Dataset created with ImageTool.to_dataset()
        input_data: list[xr.DataArray] | list[xr.Dataset] = [data]
    elif data is None:
        input_data = []
    else:
        input_data = erlab.interactive.imagetool.viewer._parse_input(data)

    direct_manager = _direct_manager_for_target(target)
    if direct_manager is not None:
        # If the manager is running in the same process, directly pass the data
        direct_manager._data_recv(input_data, kwargs)
        return None

    record = resolve_manager_record(target)
    return _query_zmq(
        AddDataPacket(packet_type="add", data_list=input_data, arguments=kwargs),
        record=record,
    )


def replace_data(
    index: int | str | Collection[int | str],
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
    *,
    target: int | None = None,
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
    target
        Optional 0-based manager index. If omitted, the current process default is
        used. If no default is set, the only live manager is used. If multiple
        managers are live, an ambiguity error is raised.

    Returns
    -------
    Response
        Manager response.
    """
    data_list = erlab.interactive.imagetool.viewer._parse_input(data)

    if isinstance(index, (int, str)):
        index = [index]

    if len(data_list) != len(index):
        raise ValueError(
            f"Mismatch between number of data ({len(data_list)}) "
            f"and provided indices ({len(index)})"
        )

    direct_manager = _direct_manager_for_target(target)
    if direct_manager is not None:
        direct_manager._data_replace(data_list, list(index))
        return Response(status="ok")

    record = resolve_manager_record(target)
    return _query_zmq(
        ReplacePacket(
            packet_type="replace", data_list=data_list, replace_idxs=list(index)
        ),
        record=record,
    )


def _watch_data(
    varname: str,
    uid: str,
    data: xr.DataArray,
    show: bool = False,
    *,
    target: int | None = None,
    watched_metadata: dict[str, typing.Any] | None = None,
) -> None:
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
    target
        Optional 0-based manager index. If omitted, the current process default is
        used. If no default is set, the only live manager is used. If multiple
        managers are live, an ambiguity error is raised.
    """
    if watched_metadata is None:
        watched_metadata = {}

    record = resolve_manager_record(target)
    _query_zmq(
        WatchPacket(
            packet_type="watch",
            watched_info=(varname, uid),
            watched_data=data,
            watched_metadata=watched_metadata,
        ),
        record=record,
    )
    if show:
        _query_zmq(
            CommandPacket(packet_type="command", command="show-uid", command_arg=uid),
            record=record,
        )


def _unwatch_data(
    uid: str, remove: bool = False, *, target: int | None = None
) -> Response:
    """Cancel watching a variable in the ImageToolManager.

    Parameters
    ----------
    uid
        Unique identifier for the watched variable.
    remove
        Whether to remove the corresponding ImageTool window when unwatching. Default is
        `False`.
    target
        Optional 0-based manager index. If omitted, the current process default is
        used. If no default is set, the only live manager is used. If multiple
        managers are live, an ambiguity error is raised.
    """
    record = resolve_manager_record(target)
    if remove:
        return _query_zmq(
            CommandPacket(packet_type="command", command="remove-uid", command_arg=uid),
            record=record,
        )
    return _query_zmq(
        CommandPacket(packet_type="command", command="unwatch-uid", command_arg=uid),
        record=record,
    )


def _watch_info(*, target: int | None = None) -> dict[str, typing.Any]:
    """Return watched-variable metadata for the target manager."""
    direct_manager = _direct_manager_for_target(target)
    if direct_manager is not None:
        return direct_manager._watch_info()
    record = resolve_manager_record(target)
    response = _query_zmq(
        CommandPacket(packet_type="command", command="watch-info"),
        record=record,
    )
    return response.watch_info or {}


def watch_info(*, target: int | None = None) -> dict[str, typing.Any]:
    """Return watched-variable metadata for the target manager.

    This is primarily intended for editor integrations and the watcher restore
    workflow.

    .. versionadded:: 3.22.0
    """
    return _watch_info(target=target)


def watch_data(varname: str, uid: str, data: xr.DataArray, show: bool = False) -> None:
    """Compatibility wrapper for :func:`_watch_data`.

    .. deprecated:: 3.20.0
       Use :func:`erlab.interactive.imagetool.manager.watch` instead.
    """
    warnings.warn(
        "`watch_data` is deprecated and will become private. "
        "Use `erlab.interactive.imagetool.manager.watch` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _watch_data(varname, uid, data, show=show)


def unwatch_data(uid: str, remove: bool = False) -> Response:
    """Compatibility wrapper for :func:`_unwatch_data`.

    .. deprecated:: 3.20.0
       Use :func:`erlab.interactive.imagetool.manager.watch` with ``stop`` options
       instead.
    """
    warnings.warn(
        "`unwatch_data` is deprecated and will become private. "
        "Use `erlab.interactive.imagetool.manager.watch(..., stop=True)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _unwatch_data(uid, remove=remove)


def _remove_idx(index: int, *, target: int | None = None) -> Response:
    """Remove the ImageTool window at the given index.

    Parameters
    ----------
    index
        Index of the ImageTool window to close.
    """
    return _query_zmq(
        CommandPacket(packet_type="command", command="remove-idx", command_arg=index),
        target=target,
    )


def _show_idx(index: int, *, target: int | None = None) -> Response:
    """Show the ImageTool window at the given index.

    Parameters
    ----------
    index
        Index of the ImageTool window to show.
    """
    return _query_zmq(
        CommandPacket(packet_type="command", command="show-idx", command_arg=index),
        target=target,
    )


def fetch(index: int | str, *, target: int | None = None) -> xr.DataArray | None:
    """Get data from the ImageTool window at the given index.

    Parameters
    ----------
    index
        Index of the ImageTool window to get data from, or the unique identifier (UID)
        of a watched variable (used internally).
    target
        Optional 0-based manager index. If omitted, the current process default is
        used. If no default is set, the only live manager is used. If multiple
        managers are live, an ambiguity error is raised.

    Returns
    -------
    xr.DataArray or None
        The data in the ImageTool window at the given index, or `None` if the index is
        invalid or the data cannot be retrieved.
    """
    direct_manager = _direct_manager_for_target(target)
    if direct_manager is not None:
        return direct_manager._get_imagetool_data(index)
    record = resolve_manager_record(target)
    return _query_zmq(
        CommandPacket(packet_type="command", command="get-data", command_arg=index),
        record=record,
    ).data
