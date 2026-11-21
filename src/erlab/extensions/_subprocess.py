"""Keep one private interpreter available for recorded extension work."""

from __future__ import annotations

import contextlib
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
import typing

import cloudpickle

from erlab.extensions._models import ExtensionExecutionError

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab._provenance._settings import ExecutionSettings

_T = typing.TypeVar("_T")
_IDLE_SECONDS = 5.0


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


def _serve() -> None:
    from erlab.extensions._routine import _run_with_settings

    # Keep an independent result pipe. The user's extension can write to standard
    # output from Python or from native code without corrupting the result.
    with (
        os.fdopen(os.dup(sys.stdin.fileno()), "rb") as requests,
        os.fdopen(os.dup(sys.stdout.fileno()), "wb") as replies,
        open(os.devnull, "r+b") as empty_input,
    ):
        os.dup2(empty_input.fileno(), sys.stdin.fileno())
        output_fd = empty_input.fileno() if sys.stderr is None else sys.stderr.fileno()
        os.dup2(output_fd, sys.stdout.fileno())
        settings = pickle.load(requests)
        # Loading a Dask task can import a helper module. Install the settings
        # before deserializing either the callable or its deferred dependencies.
        while _run_with_settings(settings, _request, requests, replies):
            pass


class SubprocessWorker:
    """Serve one request at a time and release the interpreter after idle use."""

    def __init__(self, settings: ExecutionSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._idle: threading.Timer | None = None
        self._deadline = 0.0

    def _close(self) -> None:
        if self._idle is not None:
            self._idle.cancel()
            self._idle = None
        process, self._process = self._process, None
        if process is not None:
            # Closing the parent pipe also closes the worker when its owning Dask
            # process exits before the idle timer fires.
            with contextlib.suppress(BrokenPipeError):
                typing.cast("typing.BinaryIO", process.stdin).close()
            typing.cast("typing.BinaryIO", process.stdout).close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # A finished routine may have left a non-daemonic thread running.
                process.kill()
                process.wait()

    def close(self) -> None:
        with self._lock:
            self._close()

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
            starting = self._process is None
            if self._process is None:
                # Notebook sessions and editable checkouts can add import paths
                # after interpreter startup. Give the worker that same search path.
                paths = os.pathsep.join(os.path.abspath(path) for path in sys.path)
                self._process = subprocess.Popen(
                    [sys.executable, "-m", "erlab.extensions._subprocess"],
                    env={**os.environ, "PYTHONPATH": paths},
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
            process = self._process
            requests = typing.cast("typing.BinaryIO", process.stdin)
            replies = typing.cast("typing.BinaryIO", process.stdout)
            try:
                if starting:
                    pickle.dump(self._settings, requests)
                cloudpickle.dump((function, args), requests)
                requests.flush()
                ok, value = pickle.load(replies)
            except (EOFError, BrokenPipeError) as error:
                self._close()
                raise ExtensionExecutionError(
                    "The extension process stopped. Run the routine again."
                ) from error
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
    _serve()
