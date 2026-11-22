"""Keep one private interpreter available for recorded extension work."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import pickle
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import typing
from dataclasses import dataclass
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge

import cloudpickle

from erlab.extensions._models import ExtensionExecutionError

if typing.TYPE_CHECKING:
    import io
    from collections.abc import Callable

    from erlab._provenance._settings import ExecutionSettings

_T = typing.TypeVar("_T")
_IDLE_SECONDS = 5.0
_STARTUP_SECONDS = 90.0
_WORKER_ARG = "--erlab-extension-worker"
_AUTH_ENV = "ERLAB_EXTENSION_WORKER_AUTH"


@dataclass(frozen=True)
class WorkerContext:
    settings: ExecutionSettings
    search_path: tuple[str, ...]
    directory: str


class WorkerStopped(ExtensionExecutionError):
    """The interpreter stopped before it returned a complete reply."""

    def __init__(self, message="The extension process stopped. Run the routine again."):
        super().__init__(message)


def _worker_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, _WORKER_ARG]
    return [sys.executable, "-P", "-m", "erlab.extensions._subprocess", _WORKER_ARG]


def _authenticate(
    channel: socket.socket, key: bytes, *, parent: bool, timeout: float = 5.0
) -> None:
    # Use multiprocessing's mutual authentication before reading any pickle.
    # A local connection that never answers must not block the worker's startup.
    def abort_handshake() -> None:
        with contextlib.suppress(OSError):
            channel.shutdown(socket.SHUT_RDWR)

    timer = threading.Timer(timeout, abort_handshake)
    timer.daemon = True
    timer.start()
    try:
        with Connection(channel.dup().detach()) as auth:
            if parent:
                deliver_challenge(auth, key)
                answer_challenge(auth, key)
            else:
                answer_challenge(auth, key)
                deliver_challenge(auth, key)
    finally:
        timer.cancel()
    channel.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _request(requests: typing.BinaryIO, replies: typing.BinaryIO) -> bool:
    try:
        function, args = pickle.load(requests)
    except EOFError:
        return False
    try:
        message = (True, function(*args))
    except BaseException as error:
        error.add_note(traceback.format_exc())
        message = (False, error)
    cloudpickle.dump(message, replies)
    replies.flush()
    return True


def serve_worker(port: int) -> None:
    from erlab.extensions._routine import _run_with_settings

    if sys.stdout is not None and sys.stderr is not None:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    key = bytes.fromhex(os.environ.pop(_AUTH_ENV))
    with (
        socket.create_connection(("127.0.0.1", port)) as channel,
        channel.makefile("rb") as requests,
        channel.makefile("wb") as replies,
    ):
        _authenticate(channel, key, parent=False)
        context = pickle.load(requests)
        while True:
            # A task or its operands can import scientific helpers while unpickling.
            os.chdir(context.directory)
            sys.path[:] = context.search_path
            if not _run_with_settings(context.settings, _request, requests, replies):
                return


class SubprocessWorker:
    """Serve one request at a time and release the interpreter after idle use."""

    def __init__(self, context: WorkerContext) -> None:
        self._context = context
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._requests: typing.BinaryIO | None = None
        self._replies: typing.BinaryIO | None = None
        self._idle: threading.Timer | None = None
        self._deadline = 0.0

    def _start(self) -> None:
        # The ephemeral listener is local to this machine and exists only while
        # this child connects. Nothing depends on a GUI executable having stdio.
        key = secrets.token_bytes(32)
        paths = os.pathsep.join(
            os.path.abspath(path) for path in self._context.search_path
        )
        with socket.create_server(("127.0.0.1", 0)) as listener:
            process = subprocess.Popen(
                [*_worker_command(), str(listener.getsockname()[1])],
                env={**os.environ, "PYTHONPATH": paths, _AUTH_ENV: key.hex()},
                stdin=subprocess.DEVNULL,
            )
            self._process = process
            deadline = time.monotonic() + _STARTUP_SECONDS
            try:
                while process.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    listener.settimeout(min(0.2, remaining))
                    try:
                        channel, _ = listener.accept()
                    except TimeoutError:
                        continue
                    with channel:
                        try:
                            _authenticate(
                                channel, key, parent=True, timeout=min(5, remaining)
                            )
                        except (multiprocessing.AuthenticationError, EOFError, OSError):
                            continue
                        self._requests = channel.makefile("wb")
                        self._replies = channel.makefile("rb")
                        pickle.dump(self._context, self._requests)
                        return
            except BaseException:
                process.kill()
                self._close()
                raise
            timed_out = process.poll() is None
            if timed_out:
                process.kill()
            self._close()
            if timed_out:
                raise WorkerStopped(
                    "The extension interpreter did not connect before the startup "
                    "timeout. Check its startup logs and run the routine again."
                )
            raise WorkerStopped

    def _close(self) -> None:
        if self._idle is not None:
            self._idle.cancel()
            self._idle = None
        process, self._process = self._process, None
        if self._requests is not None:
            with contextlib.suppress(OSError):
                self._requests.close()
            self._requests = None
        if self._replies is not None:
            self._replies.close()
            self._replies = None
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # A finished routine may have left a non-daemonic thread running.
                process.kill()
                process.wait()

    def close(self) -> None:
        with self._lock:
            self._close()

    def disown_after_fork(self) -> None:
        # An I/O thread may have held a buffered stream lock at fork. Close the
        # raw descriptors without flushing, waiting for, or signalling the parent.
        for stream in (self._requests, self._replies):
            if stream is not None:
                typing.cast("io.BufferedReader | io.BufferedWriter", stream).raw.close()

    def _schedule_expiry(self, process: subprocess.Popen, delay: float) -> None:
        self._idle = threading.Timer(delay, self._expire, (process,))
        self._idle.daemon = True
        self._idle.start()

    def _expire(self, process: subprocess.Popen) -> None:
        with self._lock:
            if process is not self._process:
                return
            remaining = self._deadline - time.monotonic()
            if remaining > 0:
                self._schedule_expiry(process, remaining)
            else:
                self._close()

    def run(self, function: Callable[..., _T], *args: typing.Any) -> _T:
        with self._lock:
            if self._process is None:
                self._start()
            process = typing.cast("subprocess.Popen", self._process)
            requests = typing.cast("typing.BinaryIO", self._requests)
            replies = typing.cast("typing.BinaryIO", self._replies)
            try:
                cloudpickle.dump((function, args), requests)
                requests.flush()
                ok, value = pickle.load(replies)
            except (EOFError, ConnectionError) as error:
                self._close()
                raise WorkerStopped from error
            except BaseException:
                # A partial request cannot be reused. Stop its interpreter.
                process.kill()
                self._close()
                raise
            finally:
                if self._process is not None:
                    self._deadline = time.monotonic() + _IDLE_SECONDS
                    if self._idle is None:
                        self._schedule_expiry(process, _IDLE_SECONDS)
            if not ok:
                raise value
            return value


if __name__ == "__main__":
    serve_worker(int(sys.argv[2]))
