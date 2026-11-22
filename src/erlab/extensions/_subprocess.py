"""Keep one private interpreter available for recorded extension work."""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
import typing
from dataclasses import dataclass

import cloudpickle

from erlab.extensions._models import ExtensionExecutionError

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab._provenance._settings import ExecutionSettings

_T = typing.TypeVar("_T")
_IDLE_SECONDS = 5.0


@dataclass(frozen=True)
class WorkerContext:
    settings: ExecutionSettings
    search_path: tuple[str, ...]
    directory: str


class WorkerStopped(ExtensionExecutionError):
    """The interpreter stopped before it returned a complete reply."""

    message = "The extension process stopped. Run the routine again."


def _write_request(stream, function, args) -> None:
    cloudpickle.dump((function, args), stream)


def _result(reply):
    ok, value = reply
    if not ok:
        raise value
    return value


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


def _service(context: WorkerContext, requests, replies) -> bool:
    from erlab.extensions._routine import _run_with_settings

    # A task or its operands can import scientific helpers while unpickling.
    os.chdir(context.directory)
    sys.path[:] = context.search_path
    return _run_with_settings(context.settings, _request, requests, replies)


def _buffered_reply(context: WorkerContext, payload: bytes) -> bytes:
    replies = io.BytesIO()
    _service(context, io.BytesIO(payload), replies)
    return replies.getvalue()


def _serve() -> None:
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
        context = pickle.load(requests)
        while _service(context, requests, replies):
            pass


class SubprocessWorker:
    """Serve one request at a time and release the interpreter after idle use."""

    def __init__(self, context: WorkerContext) -> None:
        self._context = context
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

    def disown_after_fork(self) -> None:
        # An I/O thread may have held a buffered stream lock at fork. Close the
        # raw descriptors without flushing, waiting for, or signalling the parent.
        if (process := self._process) is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    typing.cast(
                        "io.BufferedReader | io.BufferedWriter", stream
                    ).raw.close()

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
                paths = os.pathsep.join(
                    os.path.abspath(path) for path in self._context.search_path
                )
                self._process = subprocess.Popen(
                    [sys.executable, "-P", "-m", "erlab.extensions._subprocess"],
                    env={**os.environ, "PYTHONPATH": paths},
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
            process = self._process
            requests = typing.cast("typing.BinaryIO", process.stdin)
            replies = typing.cast("typing.BinaryIO", process.stdout)
            try:
                if starting:
                    pickle.dump(self._context, requests)
                _write_request(requests, function, args)
                requests.flush()
                reply = pickle.load(replies)
            except (EOFError, BrokenPipeError) as error:
                self._close()
                raise WorkerStopped(WorkerStopped.message) from error
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
            return _result(reply)


class FrozenWorker:
    """Serve the same requests through the packaged multiprocessing entry point."""

    def __init__(self, context: WorkerContext) -> None:
        from multiprocessing.spawn import prepare

        from joblib.externals.loky import ProcessPoolExecutor

        if multiprocessing.current_process().daemon:
            raise ExtensionExecutionError(
                "This packaged Dask worker cannot start an extension interpreter. "
                "Use Dask's threaded scheduler or start a non-daemonic worker."
            )
        self._context = context
        self._executor = ProcessPoolExecutor(
            max_workers=1,
            context=multiprocessing.get_context("spawn"),
            # PyInstaller clears multiprocessing's original directory. Spawn
            # then turns an empty search path entry into None. Restore the path
            # before importing ERLab or any scientific library to unpickle a job.
            initializer=prepare,
            initargs=(
                {"sys_path": list(context.search_path), "dir": context.directory},
            ),
            timeout=_IDLE_SECONDS,
        )

    def close(self) -> None:
        self._executor.shutdown()

    def run(self, function: Callable[..., _T], *args: typing.Any) -> _T:
        from joblib.externals.loky import BrokenProcessPool

        request = io.BytesIO()
        _write_request(request, function, args)
        try:
            response = self._executor.submit(
                _buffered_reply, self._context, request.getvalue()
            ).result()
        except BrokenProcessPool as error:
            raise WorkerStopped(WorkerStopped.message) from error
        return _result(pickle.loads(response))


if __name__ == "__main__":
    _serve()
