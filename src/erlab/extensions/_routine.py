"""Run a recorded extension without changing the caller's xarray options."""

from __future__ import annotations

import contextvars
import copy
import os
import sys
import threading
import typing

import dask
import numpy as np
import xarray as xr

from erlab.extensions._api import _invoke_routine, _load_script_bytes
from erlab.extensions._models import ExtensionExecutionError, ExtensionImportError
from erlab.extensions._subprocess import (
    FrozenWorker,
    SubprocessWorker,
    WorkerContext,
    WorkerStopped,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from erlab._provenance._settings import ExecutionSettings

_T = typing.TypeVar("_T")
_worker_settings: contextvars.ContextVar[ExecutionSettings | None] = (
    contextvars.ContextVar("erlab_extension_settings", default=None)
)


def _run_with_settings(
    settings: ExecutionSettings, function: Callable[..., _T], *args: typing.Any
) -> _T:
    token = _worker_settings.set(settings)
    try:
        with settings.scope(), dask.config.set(scheduler="synchronous"):
            return function(*args)
    finally:
        _worker_settings.reset(token)


class RoutineProcesses:
    """Share one settings-isolated interpreter across active Manager sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._admission = threading.Lock()
        self._worker: FrozenWorker | SubprocessWorker | None = None
        self._owners = 0
        self._context: WorkerContext | None = None

    def acquire(self) -> None:
        with self._lock:
            self._owners += 1

    def release(self) -> None:
        with self._lock:
            self._owners -= 1
            if self._owners or self._worker is None:
                return
        # A worker can still be publishing its last result. Release it after that
        # job drains, without waiting for python shutdown on the GUI thread.
        threading.Thread(target=self._close_unowned, daemon=True).start()

    def _close_unowned(self) -> None:
        with self._admission:
            with self._lock:
                if self._owners:
                    return
            self._retire()

    def _retire(self) -> None:
        with self._lock:
            worker, self._worker = self._worker, None
        if worker is not None:
            worker.close()

    def _after_fork(self) -> None:
        if isinstance(self._worker, SubprocessWorker):
            self._worker.disown_after_fork()
        self._worker = None
        self._lock = threading.Lock()
        self._admission = threading.Lock()

    def run(
        self,
        settings: ExecutionSettings,
        function: Callable[..., _T],
        *args: typing.Any,
    ) -> _T:
        # Bound queued IPC payloads. Native subprocess pipes also work in
        # daemonic distributed workers and do not need a resource tracker.
        with self._admission:
            context = WorkerContext(settings, tuple(sys.path), os.getcwd())
            if self._context != context:
                self._retire()
                self._context = context
            with self._lock:
                if self._worker is None:
                    worker_type = (
                        FrozenWorker
                        if getattr(sys, "frozen", False)
                        else SubprocessWorker
                    )
                    self._worker = worker_type(context)
                worker = self._worker
            try:
                return worker.run(function, *args)
            except WorkerStopped:
                self._retire()
                raise


routine_processes = RoutineProcesses()


if hasattr(os, "register_at_fork"):
    # Resolve the current owner when the child starts. Its locks and file
    # descriptors belong to the parent process.
    os.register_at_fork(after_in_child=lambda: routine_processes._after_fork())


def _run_isolated(
    settings: ExecutionSettings, function: Callable[..., _T], *args: typing.Any
) -> _T:
    # A nested extension or a chunk from an earlier result already owns this
    # interpreter. Compute its dependencies here to avoid waiting on our own worker.
    if _worker_settings.get() == settings:
        return function(*args)
    return routine_processes.run(settings, function, *args)


def _copied_xindexes(data: xr.DataArray) -> dict[typing.Hashable, xr.Index]:
    """Copy each unique xarray index without splitting shared index groups."""
    copies: dict[int, xr.Index] = {}
    indexes: dict[typing.Hashable, xr.Index] = {}
    for name, index in data.xindexes.items():
        index_key = id(index)
        copied = copies.get(index_key)
        if copied is None:
            copied = index.copy(deep=True)
            copies[index_key] = copied
        indexes[name] = copied
    return indexes


def _resident_array(variable: xr.Variable) -> np.ndarray | None:
    # IndexVariable always wraps an in-memory pandas index. Xarray does not
    # report a MultiIndex adapter as in memory on some versions.
    if variable._in_memory or isinstance(variable, xr.IndexVariable):
        values = variable.data
        if isinstance(values, np.ndarray):
            return values
    return None


def _readonly_variable(variable: xr.Variable) -> xr.Variable:
    result = variable.to_base_variable().copy(deep=False)
    if (values := _resident_array(variable)) is not None:
        view = values.view()
        view.flags.writeable = False
        result.data = view
    result.attrs = copy.deepcopy(variable.attrs)
    result.encoding = copy.deepcopy(variable.encoding)
    return result


def _readonly_array(data: xr.DataArray) -> xr.DataArray:
    """Protect NumPy inputs and metadata without reading lazy backends."""
    coordinates = {
        name: _readonly_variable(variable)
        for name, variable in data.coords.variables.items()
    }
    result = xr.DataArray(
        _readonly_variable(data.variable),
        coords=xr.Coordinates(coordinates, indexes=_copied_xindexes(data)),
        name=data.name,
    )
    result.encoding = copy.deepcopy(data.encoding)
    for variable in result.coords.variables.values():
        if (values := _resident_array(variable)) is not None:
            values.flags.writeable = False
    return result


def _detached_routine_output(
    output: xr.DataArray, input_data: xr.DataArray
) -> xr.DataArray:
    """Detach writable result buffers from input and read-only arrays."""
    input_buffers = tuple(
        values
        for variable in (input_data.variable, *input_data.coords.variables.values())
        if (values := _resident_array(variable)) is not None
    )

    def detached(variable: xr.Variable) -> xr.Variable:
        if (values := _resident_array(variable)) is None:
            return variable
        if not values.flags.writeable or any(
            np.may_share_memory(values, source) for source in input_buffers
        ):
            return variable.to_base_variable().copy(data=values.copy(), deep=False)
        return variable

    data = detached(output.variable)
    coords = {
        name: detached(variable) for name, variable in output.coords.variables.items()
    }
    if data is output.variable and all(
        variable is output.coords.variables[name] for name, variable in coords.items()
    ):
        return output
    result = xr.DataArray(
        data,
        coords=xr.Coordinates(coords, indexes=_copied_xindexes(output)),
        name=output.name,
    )
    result.encoding = dict(output.encoding)
    return result


def _invoke(
    source: bytes,
    path: Path,
    source_hash: str | None,
    routine_id: str,
    parameters: Mapping[str, typing.Any],
    data: xr.DataArray,
    settings: ExecutionSettings,
) -> xr.DataArray:
    try:
        loaded = _load_script_bytes(source, path, expected_source_hash=source_hash)
    except (KeyboardInterrupt, SystemExit) as error:
        raise ExtensionImportError(f"Importing extension {path} stopped") from error
    try:
        result = _invoke_routine(loaded, _readonly_array(data), routine_id, parameters)
        result = _detached_routine_output(result, data)
    except (KeyboardInterrupt, SystemExit) as error:
        raise ExtensionExecutionError(f"Routine {routine_id!r} stopped") from error
    finally:
        # A later call can use different settings at module initialization. The
        # returned Dask graph owns any functions it still needs. With this private
        # module unregistered cloudpickle transports those functions by value.
        sys.modules.pop(loaded.erlab.module.__name__, None)
    from erlab.extensions._chunks import bind_result

    return bind_result(result, settings)


def run_recorded_routine(
    data: xr.DataArray,
    *,
    source: bytes,
    path: Path,
    source_hash: str | None,
    routine_id: str,
    parameters: Mapping[str, typing.Any],
    settings: ExecutionSettings,
) -> xr.DataArray:
    return _run_isolated(
        settings,
        _invoke,
        source,
        path,
        source_hash,
        routine_id,
        parameters,
        data,
        settings,
    )
