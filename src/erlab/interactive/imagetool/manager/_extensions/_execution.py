"""Serialized in-process execution for ImageTool Manager extensions."""

from __future__ import annotations

import collections.abc
import contextlib
import copy
import dataclasses
import hashlib
import logging
import pathlib
import re
import sys
import threading
import time
import traceback
import typing
import uuid
from collections import deque

import numpy as np
import xarray as xr
from qtpy import QtCore

import erlab
from erlab.extensions import (
    ExtensionExecutionError,
    LoadedScript,
    LoaderDescriptor,
    RoutineDescriptor,
    load_script,
)
from erlab.extensions._api import _CapabilityStatus, _coerce_call_parameters
from erlab.extensions._models import _require_finite_parameter_values
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool._provenance._model import (
    compose_display_provenance,
    full_data,
)
from erlab.interactive.imagetool._provenance._operations import (
    ExtensionRoutineOperation,
)
from erlab.interactive.imagetool.manager._extensions._catalog import (
    _ExtensionCatalogConflictError,
    _ExtensionCatalogStore,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionCatalogModel,
    _ExtensionSource,
    _source_loader_name_filters,
)
from erlab.io.dataloader import LoaderBase

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from erlab.interactive.imagetool._provenance._model import ToolProvenanceOperation
    from erlab.interactive.imagetool.manager._extensions._catalog import (
        _ExtensionCatalog,
    )
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager

logger = logging.getLogger(__name__)


def _manager_module_name(
    manager_session_id: str,
    extension_id: str,
    source_hash: str,
) -> str:
    """Return an import name isolated to one manager and source snapshot."""
    safe_session_id = re.sub(r"\W", "_", manager_session_id)
    safe_extension_id = re.sub(r"\W", "_", extension_id)
    extension_token = hashlib.sha256(extension_id.encode()).hexdigest()[:12]
    return (
        f"_erlab_extension_{safe_session_id}_{safe_extension_id}_"
        f"{extension_token}_{source_hash}"
    )


def _remove_manager_modules(manager_session_id: str) -> None:
    """Release script modules after their owning manager stops all workers."""
    safe_session_id = re.sub(r"\W", "_", manager_session_id)
    prefix = f"_erlab_extension_{safe_session_id}_"
    for module_name in tuple(sys.modules):
        if module_name.startswith(prefix):
            sys.modules.pop(module_name, None)


def _cached_script(
    modules: dict[tuple[str, str], LoadedScript],
    *,
    extension_id: str,
    source_hash: str,
    source_path: pathlib.Path,
    module_name: str,
) -> LoadedScript:
    """Import one exact script once during a manager execution lifetime."""
    key = (extension_id, source_hash)
    loaded = modules.get(key)
    if loaded is None:
        loaded = load_script(
            source_path,
            module_name=module_name,
            expected_source_hash=source_hash,
        )
        modules[key] = loaded
    return loaded


def _extension_error(error: BaseException, action: str) -> Exception:
    """Convert process-control exceptions from extension code into normal failures."""
    if isinstance(error, Exception):
        return error
    return ExtensionExecutionError(
        f"Extension {action} stopped with {type(error).__name__}"
    )


@dataclasses.dataclass(frozen=True)
class _ExtensionRoutineJob:
    """Pinned source and input identity for one queued routine call."""

    job_id: str
    extension_id: str
    extension_name: str
    source_hash: str
    routine: RoutineDescriptor
    source_path: pathlib.Path
    parameters: dict[str, typing.Any]
    input_uid: str
    input_snapshot: str
    input_data: xr.DataArray
    catalog_generation: int
    catalog_store: _ExtensionCatalogStore = dataclasses.field(repr=False, compare=False)


@dataclasses.dataclass(frozen=True)
class _ExtensionRoutineResult:
    job: _ExtensionRoutineJob
    output: xr.DataArray | None
    duration: float
    status: typing.Literal["success", "failed", "discarded"]
    traceback_text: str | None = None


@dataclasses.dataclass
class _ExtensionRoutineWaiter:
    """Own the nested event loop for one synchronous manager replay call."""

    loop: QtCore.QEventLoop
    result: _ExtensionRoutineResult | None = None


@dataclasses.dataclass(frozen=True)
class _ExtensionLoaderCall:
    """Pinned decorated loader callable passed to existing file workers."""

    manager_session_id: str
    catalog_generation: int
    extension_id: str
    extension_name: str
    source_hash: str
    loader_id: str
    descriptor: LoaderDescriptor
    source_path: pathlib.Path
    executor: Callable[
        [_ExtensionLoaderCall, pathlib.Path, dict[str, typing.Any]],
        xr.DataArray | xr.Dataset | xr.DataTree,
    ] = dataclasses.field(repr=False, compare=False)

    @property
    def manager_loader_name(self) -> str:
        return f"{self.extension_id}:{self.loader_id}"

    def __call__(self, path: pathlib.Path, **parameters: typing.Any) -> typing.Any:
        _require_finite_parameter_values(parameters)
        return self.executor(self, pathlib.Path(path), dict(parameters))

    @property
    def __name__(self) -> str:
        return "load"

    def _invoke(
        self,
        path: pathlib.Path,
        parameters: Mapping[str, typing.Any],
        script_modules: dict[tuple[str, str], LoadedScript],
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        """Import and invoke this pinned loader from the extension worker."""
        started = time.perf_counter()
        fields = {
            "manager_session_id": self.manager_session_id,
            "catalog_generation": self.catalog_generation,
            "extension_id": self.extension_id,
            "extension_source_hash": self.source_hash,
            "capability_id": self.loader_id,
            "extension_source": str(self.source_path),
            "parameters": parameters,
            "file_path": str(path),
        }
        try:
            logger.info("Importing extension loader source", extra=fields)
            loaded = _cached_script(
                script_modules,
                extension_id=self.extension_id,
                source_hash=self.source_hash,
                source_path=self.source_path,
                module_name=_manager_module_name(
                    self.manager_session_id,
                    self.extension_id,
                    self.source_hash,
                ),
            )
            entry = loaded.loaders.get(self.loader_id)
            entry = _require_loader_entry(self, entry)
            values = _coerce_call_parameters(entry[1], parameters)
            logger.info("Invoking extension loader", extra=fields)
            result = _require_loader_output(entry[1](path, **values))
        except BaseException:
            logger.exception(
                "Extension loader failed",
                extra={
                    **fields,
                    "extension_status": "failed",
                    "duration_seconds": time.perf_counter() - started,
                    "suppress_ui_alert": True,
                },
            )
            raise
        logger.info(
            "Extension loader finished",
            extra={
                **fields,
                "duration_seconds": time.perf_counter() - started,
                "extension_status": "success",
                "output": _loader_output_log_fields(result),
            },
        )
        return result


class _ExtensionLoaderSignals(QtCore.QObject):
    """Signal bridge that keeps synchronous GUI waits responsive."""

    finished = QtCore.Signal()


class _ExtensionLoaderWorker(QtCore.QRunnable):
    """Run one synchronous file-loader request on the extension thread pool."""

    def __init__(
        self,
        call: _ExtensionLoaderCall,
        path: pathlib.Path,
        parameters: dict[str, typing.Any],
        catalog_store: _ExtensionCatalogStore,
        script_modules: dict[tuple[str, str], LoadedScript],
    ) -> None:
        super().__init__()
        self.call = call
        self.path = path
        self.parameters = parameters
        self.catalog_store = catalog_store
        self.script_modules = script_modules
        self.signals = _ExtensionLoaderSignals()
        self.done = threading.Event()
        self.output: xr.DataArray | xr.Dataset | xr.DataTree | None = None
        self.error: Exception | None = None
        self._state_lock = threading.Lock()
        self._started = False
        self._cancelled = False

    def cancel_if_pending(self) -> None:
        """Release a waiting caller only when this worker has not started."""
        cancelled = False
        with self._state_lock:
            if self._started or self._cancelled:
                return
            self._cancelled = True
            self.error = ExtensionExecutionError(
                "The queued extension loader was canceled during manager shutdown"
            )
            self.done.set()
            cancelled = True
        if cancelled:
            self.signals.finished.emit()

    def run(self) -> None:
        with self._state_lock:
            if self._cancelled:
                return
            self._started = True
        try:
            try:
                status = self.catalog_store.capability_status(
                    self.call.extension_id,
                    self.call.source_hash,
                    "loader",
                    self.call.loader_id,
                )
            except KeyError:
                status = "missing-source"
            if status != "ready":
                self.error = ExtensionExecutionError(
                    f"The extension loader is unavailable: {status}"
                )
                logger.info(
                    "Discarding unavailable queued extension loader",
                    extra={
                        "manager_session_id": self.call.manager_session_id,
                        "catalog_generation": self.call.catalog_generation,
                        "extension_id": self.call.extension_id,
                        "extension_source_hash": self.call.source_hash,
                        "capability_id": self.call.loader_id,
                        "extension_status": f"{status}-before-start",
                    },
                )
            else:
                self.output = self.call._invoke(
                    self.path, self.parameters, self.script_modules
                )
        except BaseException as error:
            self.error = _extension_error(error, "loader execution")
        finally:
            self.done.set()
            self.signals.finished.emit()


class _ExtensionValidationWorker(QtCore.QRunnable):
    """Validate and enable one source on the serialized extension queue."""

    def __init__(
        self,
        extension_id: str,
        source_hash: str,
        expected_record_generation: int,
        *,
        manager_session_id: str,
        catalog_store: _ExtensionCatalogStore,
        script_modules: dict[tuple[str, str], LoadedScript],
        check_loader_filter_conflicts: bool = True,
        enable_extension: bool = True,
    ) -> None:
        super().__init__()
        self.extension_id = extension_id
        self.source_hash = source_hash
        self.expected_record_generation = expected_record_generation
        self.manager_session_id = manager_session_id
        self.catalog_store = catalog_store
        self.script_modules = script_modules
        self.check_loader_filter_conflicts = check_loader_filter_conflicts
        self.enable_extension = enable_extension
        self.signals = _ExtensionLoaderSignals()
        self.done = threading.Event()
        self.output: _ExtensionCatalogModel | None = None
        self.error: Exception | None = None
        self._state_lock = threading.Lock()
        self._started = False
        self._cancelled = False

    def cancel_if_pending(self) -> None:
        """Release the GUI wait if shutdown removes this queued validation."""
        cancelled = False
        with self._state_lock:
            if self._started or self._cancelled:
                return
            self._cancelled = True
            self.error = ExtensionExecutionError(
                "The queued extension validation was canceled during manager shutdown"
            )
            self.done.set()
            cancelled = True
        if cancelled:
            self.signals.finished.emit()

    def run(self) -> None:
        with self._state_lock:
            if self._cancelled:
                return
            self._started = True
        started = time.perf_counter()
        fields: dict[str, typing.Any] = {
            "manager_session_id": self.manager_session_id,
            "catalog_generation": None,
            "extension_id": self.extension_id,
            "extension_source_hash": None,
            "capability_id": None,
            "extension_source": None,
        }
        try:
            catalog = self.catalog_store.read()
            record = catalog.extensions.get(self.extension_id)
            fields.update(
                {
                    "catalog_generation": catalog.generation,
                    "extension_source_hash": self.source_hash,
                    "extension_source": (
                        None
                        if record is None
                        else str(
                            self.catalog_store.executable_source_path(
                                self.extension_id, self.source_hash
                            )
                        )
                    ),
                }
            )
            logger.info("Importing extension source for validation", extra=fields)
            self.output = _validate_extension_source(
                self.catalog_store,
                self.extension_id,
                source_hash=self.source_hash,
                expected_record_generation=self.expected_record_generation,
                manager_session_id=self.manager_session_id,
                script_modules=self.script_modules,
                check_loader_filter_conflicts=self.check_loader_filter_conflicts,
                enable_extension=self.enable_extension,
            )
        except BaseException as error:
            self.error = _extension_error(error, "validation")
            logger.exception(
                "Extension validation failed",
                extra={
                    **fields,
                    "duration_seconds": time.perf_counter() - started,
                    "extension_status": "failed",
                    "suppress_ui_alert": True,
                },
            )
        else:
            logger.info(
                "Extension validation finished",
                extra={
                    **fields,
                    "duration_seconds": time.perf_counter() - started,
                    "extension_status": "success",
                },
            )
        finally:
            self.done.set()
            self.signals.finished.emit()


class _DecoratedLoaderAdapter(LoaderBase):
    """Expose one pinned decorated loader to a manager-owned Data Explorer."""

    name = "_decorated_extension"
    description = ""
    always_single = True
    skip_validate = True

    def __init__(self, call: _ExtensionLoaderCall) -> None:
        self._extension_call = call
        self.name = call.manager_loader_name
        self.description = call.descriptor.summary
        self.extensions = set(call.descriptor.extensions) or None  # type: ignore[misc]

    @property
    def extension_id(self) -> str:
        return self._extension_call.extension_id

    @property
    def source_hash(self) -> str:
        return self._extension_call.source_hash

    @property
    def loader_id(self) -> str:
        return self._extension_call.loader_id

    @property
    def source_path(self) -> pathlib.Path:
        return self._extension_call.source_path

    @property
    def file_dialog_methods(
        self,
    ) -> dict[str, tuple[Callable[..., typing.Any], dict[str, typing.Any]]]:
        patterns = " ".join(f"*{value}" for value in self.descriptor.extensions) or "*"
        return {f"{self.descriptor.name} ({patterns})": (self.load, {})}

    @property
    def descriptor(self) -> LoaderDescriptor:
        return self._extension_call.descriptor

    def load_single(  # type: ignore[override]
        self,
        file_path: str | pathlib.Path,
        *,
        without_values: bool = False,
        **parameters: typing.Any,
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        del without_values
        return self._extension_call(pathlib.Path(file_path), **parameters)


class _ExtensionWorkerSignals(QtCore.QObject):
    started = QtCore.Signal()
    finished = QtCore.Signal(object)


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


def _readonly_array(data: xr.DataArray) -> xr.DataArray:
    """Return an isolated xarray wrapper backed by read-only NumPy views."""
    values = data.data
    if isinstance(values, np.ndarray):
        values = values.view()
        values.flags.writeable = False
    coordinates: dict[typing.Hashable, xr.Variable] = {}
    for name, coordinate in data.coords.items():
        coordinate_data = coordinate.data
        if isinstance(coordinate_data, np.ndarray):
            coordinate_data = coordinate_data.view()
            coordinate_data.flags.writeable = False
        coordinates[name] = xr.Variable(
            coordinate.dims,
            coordinate_data,
            attrs=copy.deepcopy(coordinate.attrs),
            encoding=copy.deepcopy(coordinate.encoding),
        )
    result = xr.DataArray(
        xr.Variable(
            data.dims,
            values,
            attrs=copy.deepcopy(data.attrs),
        ),
        coords=xr.Coordinates(coordinates, indexes=_copied_xindexes(data)),
        name=data.name,
    )
    result.encoding = copy.deepcopy(data.encoding)
    for coordinate in result.coords.values():
        if isinstance(coordinate.data, np.ndarray):
            coordinate.data.flags.writeable = False
    return result


def _detached_routine_output(
    output: xr.DataArray, input_data: xr.DataArray
) -> xr.DataArray:
    """Detach result buffers that are read-only or still alias the routine input."""
    input_buffers = tuple(
        value
        for value in (
            input_data.data,
            *(coordinate.data for coordinate in input_data.coords.values()),
        )
        if isinstance(value, np.ndarray)
    )

    def detached(values: typing.Any) -> tuple[typing.Any, bool]:
        if not isinstance(values, np.ndarray):
            return values, False
        if values.flags.writeable and not any(
            np.may_share_memory(values, source) for source in input_buffers
        ):
            return values, False
        return values.copy(), True

    values, data_changed = detached(output.data)
    coordinates: dict[typing.Hashable, xr.Variable] = {}
    coordinates_changed = False
    for name, coordinate in output.coords.items():
        coordinate_values, changed = detached(coordinate.data)
        coordinates_changed |= changed
        coordinates[name] = xr.Variable(
            coordinate.dims,
            coordinate_values,
            attrs=dict(coordinate.attrs),
            encoding=dict(coordinate.encoding),
        )
    if not data_changed and not coordinates_changed:
        return output
    result = xr.DataArray(
        xr.Variable(
            output.dims,
            values,
            attrs=dict(output.attrs),
        ),
        coords=xr.Coordinates(coordinates, indexes=_copied_xindexes(output)),
        name=output.name,
    )
    result.encoding = dict(output.encoding)
    return result


def _require_loader_entry(
    call: _ExtensionLoaderCall,
    entry: tuple[LoaderDescriptor, Callable[..., typing.Any]] | None,
) -> tuple[LoaderDescriptor, Callable[..., typing.Any]]:
    if entry is None:
        raise ExtensionExecutionError(
            f"Loader {call.loader_id!r} is missing from the registered source"
        )
    return entry


def _require_loader_output(
    value: typing.Any,
) -> xr.DataArray | xr.Dataset | xr.DataTree:
    if isinstance(value, (xr.DataArray, xr.Dataset, xr.DataTree)):
        return value
    raise ExtensionExecutionError(
        f"Loader returned {type(value).__name__}; expected an xarray object"
    )


def _array_log_fields(data: xr.DataArray) -> dict[str, typing.Any]:
    return {
        "type": type(data).__name__,
        "dimensions": tuple(str(dim) for dim in data.dims),
        "shape": data.shape,
        "dtype": str(data.dtype),
    }


def _xarray_log_fields(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
) -> dict[str, typing.Any]:
    fields: dict[str, typing.Any] = {"type": type(data).__name__}
    if isinstance(data, xr.DataArray):
        fields.update(_array_log_fields(data))
    elif isinstance(data, xr.Dataset):
        fields.update(
            {
                "dimensions": tuple(str(dim) for dim in data.dims),
                "shape": tuple(data.sizes.values()),
                "dtype": tuple(str(value.dtype) for value in data.data_vars.values()),
            }
        )
    else:
        sizes = getattr(data, "sizes", {})
        if isinstance(sizes, collections.abc.Mapping):
            fields.update(
                {
                    "dimensions": tuple(str(dim) for dim in sizes),
                    "shape": tuple(sizes.values()),
                }
            )
    return fields


def _loader_output_log_fields(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
) -> dict[str, typing.Any]:
    return _xarray_log_fields(data)


def _require_routine(
    loaded: erlab.extensions.LoadedScript, routine_id: str
) -> tuple[RoutineDescriptor, Callable[..., typing.Any]]:
    entry = loaded.routines.get(routine_id)
    if entry is None:
        raise ExtensionExecutionError(
            f"Routine {routine_id!r} is missing from the registered source"
        )
    return entry


def _require_dataarray(value: typing.Any) -> xr.DataArray:
    if not isinstance(value, xr.DataArray):
        raise ExtensionExecutionError(
            f"Routine returned {type(value).__name__}; expected DataArray"
        )
    return value


def _reject_builtin_loader_filter_conflicts(
    catalog: _ExtensionCatalogModel,
    extension_id: str,
    source: _ExtensionSource,
) -> None:
    """Reject filters that would hide a built-in file loader."""
    candidate_filters = set(_source_loader_name_filters(source))
    if not candidate_filters:
        return
    builtin_filters = set(erlab.interactive.utils.file_loaders())
    conflicts = sorted(candidate_filters.intersection(builtin_filters))
    if conflicts:
        joined = ", ".join(repr(value) for value in conflicts)
        raise _ExtensionCatalogConflictError(
            f"Extension {extension_id!r} conflicts with built-in file dialog "
            f"filters: {joined}"
        )


def _validate_extension_source(
    catalog_store: _ExtensionCatalogStore,
    extension_id: str,
    *,
    source_hash: str,
    expected_record_generation: int,
    manager_session_id: str,
    script_modules: dict[tuple[str, str], LoadedScript],
    check_loader_filter_conflicts: bool = True,
    enable_extension: bool = True,
) -> _ExtensionCatalogModel:
    """Import the current source, then record its validated descriptors."""
    catalog = catalog_store.read()
    record = catalog.extensions.get(extension_id)
    if record is None:
        raise KeyError(extension_id)
    if (
        source_hash != record.source.source_hash
        or record.record_generation != expected_record_generation
    ):
        raise _ExtensionCatalogConflictError(
            f"Extension {extension_id!r} changed before validation"
        )
    try:
        loaded = _cached_script(
            script_modules,
            extension_id=extension_id,
            source_hash=source_hash,
            source_path=catalog_store.executable_source_path(extension_id, source_hash),
            module_name=_manager_module_name(
                manager_session_id, extension_id, source_hash
            ),
        )
        routines = tuple(item[0] for item in loaded.routines.values())
        loaders = tuple(item[0] for item in loaded.loaders.values())
        validated_source = record.source.model_copy(
            update={
                "routines": routines,
                "loaders": loaders,
            }
        )
        if check_loader_filter_conflicts:
            _reject_builtin_loader_filter_conflicts(
                catalog, extension_id, validated_source
            )
        return catalog_store.enable_validated_source(
            extension_id,
            source_hash=source_hash,
            expected_record_generation=expected_record_generation,
            routines=routines,
            loaders=loaders,
            enable_extension=enable_extension,
        )
    except BaseException:
        with contextlib.suppress(_ExtensionCatalogConflictError):
            catalog_store.record_validation_failure(
                extension_id,
                source_hash=source_hash,
                expected_record_generation=expected_record_generation,
                import_error=traceback.format_exc(),
            )
        raise


class _ExtensionRoutineWorker(QtCore.QRunnable):
    """Run one pinned source snapshot without accessing Qt widgets."""

    def __init__(
        self,
        job: _ExtensionRoutineJob,
        *,
        manager_session_id: str,
        catalog_store: _ExtensionCatalogStore,
        script_modules: dict[tuple[str, str], LoadedScript],
    ) -> None:
        super().__init__()
        self.job = job
        self.manager_session_id = manager_session_id
        self.catalog_store = catalog_store
        self.script_modules = script_modules
        self.signals = _ExtensionWorkerSignals()
        self.result: _ExtensionRoutineResult | None = None
        self._started = threading.Event()

    @property
    def started(self) -> bool:
        """Return whether this job passed its queued-state enablement check."""
        return self._started.is_set()

    def discard_pending(self) -> None:
        """Finish a job that the thread pool removed before execution."""
        result = _ExtensionRoutineResult(
            job=self.job,
            output=None,
            duration=0.0,
            status="discarded",
        )
        self.result = result
        self.signals.finished.emit(result)

    def run(self) -> None:
        started = time.perf_counter()
        output: xr.DataArray | None = None
        status: typing.Literal["success", "failed", "discarded"] = "failed"
        traceback_text: str | None = None
        fields: dict[str, typing.Any] = {
            "manager_session_id": self.manager_session_id,
            "catalog_generation": self.job.catalog_generation,
            "extension_id": self.job.extension_id,
            "extension_source_hash": self.job.source_hash,
            "capability_id": self.job.routine.id,
            "extension_source": str(self.job.source_path),
            "parameters": self.job.parameters,
            "input_uid": self.job.input_uid,
            "input_snapshot": self.job.input_snapshot,
            "input": _array_log_fields(self.job.input_data),
        }
        try:
            try:
                capability_status = self.catalog_store.capability_status(
                    self.job.extension_id,
                    self.job.source_hash,
                    "routine",
                    self.job.routine.id,
                )
            except KeyError:
                capability_status = "missing-source"
            if capability_status != "ready":
                status = "discarded"
                logger.info(
                    "Discarding unavailable queued extension routine",
                    extra={
                        **fields,
                        "extension_status": f"{capability_status}-before-start",
                    },
                )
                return
            self._started.set()
            self.signals.started.emit()
            logger.info("Importing extension source", extra=fields)
            module_name = _manager_module_name(
                self.manager_session_id,
                self.job.extension_id,
                self.job.source_hash,
            )
            loaded = _cached_script(
                self.script_modules,
                extension_id=self.job.extension_id,
                source_hash=self.job.source_hash,
                source_path=self.job.source_path,
                module_name=module_name,
            )
            entry = _require_routine(loaded, self.job.routine.id)
            parameters = _coerce_call_parameters(entry[1], self.job.parameters)
            logger.info("Invoking extension routine", extra=fields)
            result = _require_dataarray(
                entry[1](_readonly_array(self.job.input_data), **parameters)
            )
            result = _detached_routine_output(result, self.job.input_data)
            erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(result)
            output = result
            status = "success"
            fields["output"] = _array_log_fields(result)
        except BaseException:
            traceback_text = traceback.format_exc()
            logger.exception(
                "Extension routine failed",
                extra={
                    **fields,
                    "duration_seconds": time.perf_counter() - started,
                    "extension_status": "failed",
                    "suppress_ui_alert": True,
                },
            )
        finally:
            duration = time.perf_counter() - started
            if status == "success":
                logger.info(
                    "Extension routine finished",
                    extra={
                        **fields,
                        "duration_seconds": duration,
                        "extension_status": status,
                    },
                )
            result = _ExtensionRoutineResult(
                job=self.job,
                output=output,
                duration=duration,
                status=status,
                traceback_text=traceback_text,
            )
            self.result = result
            self.signals.finished.emit(result)


class _ExtensionExecutionController(QtCore.QObject):
    """Own one serialized extension queue for one manager lifetime.

    Running code keeps its pinned input and source snapshot until completion. Queued
    jobs recheck their owning catalog before they start. Shutdown stops admission,
    removes queued jobs, and waits for the one active worker.
    """

    queue_changed = QtCore.Signal()

    def __init__(
        self,
        manager: ImageToolManager,
        catalog: _ExtensionCatalog,
    ) -> None:
        super().__init__(manager)
        self._manager = manager
        self._catalog = catalog
        self._manager_session_id = manager._manager_record.internal_id
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._script_modules: dict[tuple[str, str], LoadedScript] = {}
        self._pending: deque[_ExtensionRoutineJob] = deque()
        self._active: tuple[_ExtensionRoutineJob, _ExtensionRoutineWorker] | None = None
        self._routine_waiters: dict[str, _ExtensionRoutineWaiter] = {}
        self._blocking_tasks: set[
            _ExtensionLoaderWorker | _ExtensionValidationWorker
        ] = set()
        self._blocking_tasks_lock = threading.Lock()
        self._accepting = True
        self._shutdown_complete = False
        self._finished_slot = self._finished
        self._started_slot = self._routine_started

    @property
    def active(self) -> _ExtensionRoutineJob | None:
        active = self._active
        if active is None or not active[1].started:
            return None
        return active[0]

    @property
    def queued(self) -> tuple[_ExtensionRoutineJob, ...]:
        active = self._active
        dispatched = () if active is None or active[1].started else (active[0],)
        return (*dispatched, *self._pending)

    def uses_extension(self, extension_id: str) -> bool:
        """Return whether this manager retains work for one extension."""
        active = self._active
        if active is not None and active[0].extension_id == extension_id:
            return True
        if any(job.extension_id == extension_id for job in self._pending):
            return True
        with self._blocking_tasks_lock:
            return any(
                (
                    task.extension_id
                    if isinstance(task, _ExtensionValidationWorker)
                    else task.call.extension_id
                )
                == extension_id
                for task in self._blocking_tasks
            )

    def run_loader(
        self,
        call: _ExtensionLoaderCall,
        path: pathlib.Path,
        parameters: dict[str, typing.Any],
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        """Run a loader synchronously on this manager's extension thread pool."""
        task = _ExtensionLoaderWorker(
            call,
            path,
            parameters,
            self._catalog.store,
            self._script_modules,
        )
        self._run_blocking_task(task)
        if task.output is None:
            raise ExtensionExecutionError("The extension loader returned no result")
        return task.output

    def validate_and_enable(
        self,
        extension_id: str,
        *,
        expected_record_generation: int,
    ) -> _ExtensionCatalogModel:
        """Validate the current catalog source on the extension thread."""
        record = self._catalog.store.read().extensions.get(extension_id)
        if record is None:
            raise KeyError(extension_id)
        return self.validate_source(
            extension_id,
            record.source.source_hash,
            expected_record_generation=expected_record_generation,
            enable_extension=True,
        )

    def validate_source(
        self,
        extension_id: str,
        source_hash: str,
        *,
        expected_record_generation: int,
        enable_extension: bool,
    ) -> _ExtensionCatalogModel:
        """Validate one registered source on the manager extension thread."""
        catalog_store = self._catalog.store
        catalog = catalog_store.read()
        record = catalog.extensions.get(extension_id)
        if record is None:
            raise KeyError(extension_id)
        if record.record_generation != expected_record_generation:
            raise _ExtensionCatalogConflictError(
                f"Extension {extension_id!r} changed before validation"
            )
        task = _ExtensionValidationWorker(
            extension_id,
            source_hash,
            expected_record_generation,
            manager_session_id=self._manager_session_id,
            catalog_store=catalog_store,
            script_modules=self._script_modules,
            check_loader_filter_conflicts=enable_extension,
            enable_extension=enable_extension,
        )
        self._run_blocking_task(task, wait_message="Validating extension...")
        if task.output is None:
            raise ExtensionExecutionError("Extension validation returned no result")
        return task.output

    def _run_blocking_task(
        self,
        task: _ExtensionLoaderWorker | _ExtensionValidationWorker,
        *,
        wait_message: str | None = None,
    ) -> None:
        """Retain one synchronous task until its exact completion signal arrives."""
        with self._blocking_tasks_lock:
            if not self._accepting:
                raise ExtensionExecutionError("Extension execution is shutting down")
            self._blocking_tasks.add(task)
            try:
                self._pool.start(task)
            except BaseException:
                self._blocking_tasks.discard(task)
                raise
        self.queue_changed.emit()
        try:
            if QtCore.QThread.currentThread() == self.thread():
                loop = QtCore.QEventLoop()
                quit_loop = loop.quit
                task.signals.finished.connect(quit_loop)
                try:
                    if not task.done.is_set():
                        if wait_message is None:
                            loop.exec(
                                QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                            )
                        else:
                            with erlab.interactive.utils.wait_dialog(
                                self._manager, wait_message
                            ):
                                loop.exec(
                                    QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                                )
                finally:
                    with contextlib.suppress(TypeError, RuntimeError):
                        task.signals.finished.disconnect(quit_loop)
            else:
                task.done.wait()
        finally:
            with self._blocking_tasks_lock:
                self._blocking_tasks.discard(task)
            self.queue_changed.emit()
        if task.error is not None:
            raise task.error

    def queue_routine(
        self,
        *,
        extension_id: str,
        routine_id: str,
        parameters: Mapping[str, typing.Any],
        target: int | str,
    ) -> str:
        if not self._accepting:
            raise RuntimeError("Extension execution is shutting down")
        node = self._manager._node_for_target(target)
        data = node.data_for_role("displayed")
        job = self._routine_job(
            extension_id=extension_id,
            source_hash=None,
            routine_id=routine_id,
            parameters=parameters,
            input_data=data,
            input_uid=node.uid,
            input_snapshot=node.snapshot_token,
        )
        self._pending.append(job)
        self.queue_changed.emit()
        self._start_next()
        return job.job_id

    def run_operation(
        self, operation: ToolProvenanceOperation, data: xr.DataArray
    ) -> xr.DataArray:
        """Replay one pinned routine through this manager's serialized queue."""
        if not isinstance(operation, ExtensionRoutineOperation):
            raise TypeError("Expected extension routine provenance")
        if QtCore.QThread.currentThread() != self.thread():
            raise ExtensionExecutionError(
                "Manager extension replay must start on the manager thread"
            )
        if not self._accepting:
            raise ExtensionExecutionError("Extension execution is shutting down")
        if self._routine_waiters:
            raise ExtensionExecutionError("Another extension replay is in progress")
        job = self._routine_job(
            extension_id=operation.extension_id,
            source_hash=operation.source_hash,
            routine_id=operation.routine_id,
            parameters=operation.parameters,
            input_data=data,
            input_uid="provenance-replay",
            input_snapshot="pinned-source",
        )
        waiter = _ExtensionRoutineWaiter(QtCore.QEventLoop())
        self._routine_waiters[job.job_id] = waiter
        self._pending.append(job)
        self.queue_changed.emit()
        self._start_next()
        waiter.loop.exec(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        self._routine_waiters.pop(job.job_id, None)
        result = waiter.result
        if result is None:
            raise ExtensionExecutionError("Extension replay ended without a result")
        if result.status == "discarded":
            raise ExtensionExecutionError("The extension is not enabled")
        if result.status == "failed" or result.output is None:
            raise ExtensionExecutionError(
                f"Routine {operation.routine_id!r} could not complete"
            )
        return result.output

    def _routine_job(
        self,
        *,
        extension_id: str,
        source_hash: str | None,
        routine_id: str,
        parameters: Mapping[str, typing.Any],
        input_data: xr.DataArray,
        input_uid: str,
        input_snapshot: str,
    ) -> _ExtensionRoutineJob:
        """Pin catalog state and input identity before queue admission."""
        _require_finite_parameter_values(parameters)
        catalog_store = self._catalog.store
        catalog = catalog_store.read()
        record = catalog.extensions.get(extension_id)
        pinned_source_hash = (
            None
            if record is None
            else record.source.source_hash
            if source_hash is None
            else source_hash
        )
        source = (
            record.source
            if record is not None and pinned_source_hash == record.source.source_hash
            else None
        )
        global_status: _CapabilityStatus = "missing-source"
        if source_hash is not None:
            with contextlib.suppress(KeyError):
                global_status = catalog_store.capability_status(
                    extension_id,
                    source_hash,
                    "routine",
                    routine_id,
                )
        if record is None or not record.enabled:
            raise ExtensionExecutionError("The extension is not enabled")
        if source_hash is not None and global_status != "ready":
            raise ExtensionExecutionError(
                f"The extension source is unavailable: {global_status}"
            )
        if pinned_source_hash is None or source is None or not source.approved:
            raise ExtensionExecutionError("The extension source is not available")
        routine = next(
            (item for item in source.routines if item.id == routine_id), None
        )
        if routine is None:
            raise ExtensionExecutionError(f"Routine {routine_id!r} is not available")
        return _ExtensionRoutineJob(
            job_id=uuid.uuid4().hex,
            extension_id=extension_id,
            extension_name=record.name,
            source_hash=pinned_source_hash,
            routine=routine,
            source_path=catalog_store.executable_source_path(
                extension_id, pinned_source_hash
            ),
            parameters=dict(parameters),
            input_uid=input_uid,
            input_snapshot=input_snapshot,
            input_data=input_data,
            catalog_generation=catalog.generation,
            catalog_store=catalog_store,
        )

    @QtCore.Slot(str)
    def remove_queued(self, job_id: str) -> None:
        active = self._active
        if (
            active is not None
            and active[0].job_id == job_id
            and self._pool.tryTake(active[1])
        ):
            active[1].discard_pending()
            return
        removed = tuple(job for job in self._pending if job.job_id == job_id)
        kept = deque(job for job in self._pending if job.job_id != job_id)
        if len(kept) == len(self._pending):
            return
        self._pending = kept
        for job in removed:
            waiter = self._routine_waiters.pop(job.job_id, None)
            if waiter is None:
                continue
            waiter.result = _ExtensionRoutineResult(
                job=job,
                output=None,
                duration=0.0,
                status="discarded",
            )
            waiter.loop.quit()
        self.queue_changed.emit()

    def _start_next(self) -> None:
        if self._active is not None or not self._accepting:
            return
        if self._pending:
            job = self._pending.popleft()
            worker = _ExtensionRoutineWorker(
                job,
                manager_session_id=self._manager_session_id,
                catalog_store=job.catalog_store,
                script_modules=self._script_modules,
            )
            worker.signals.finished.connect(self._finished_slot)
            worker.signals.started.connect(self._started_slot)
            self._active = (job, worker)
            self.queue_changed.emit()
            self._pool.start(worker)
            return
        self.queue_changed.emit()

    @QtCore.Slot(object)
    def _finished(self, result: _ExtensionRoutineResult) -> None:
        active = self._active
        if active is None or active[0].job_id != result.job.job_id:
            return
        active[1].signals.finished.disconnect(self._finished_slot)
        active[1].signals.started.disconnect(self._started_slot)
        self._active = None
        waiter = self._routine_waiters.pop(result.job.job_id, None)
        if waiter is not None:
            waiter.result = result
            waiter.loop.quit()
        elif erlab.interactive.utils.qt_is_valid(self._manager):
            if result.status == "failed":
                erlab.interactive.utils.MessageDialog.critical(
                    self._manager,
                    "Extension Error",
                    f"{result.job.routine.name} could not complete.",
                    detailed_text=result.traceback_text,
                )
            elif result.output is not None:
                self._insert_if_current(result)
        self.queue_changed.emit()
        self._start_next()

    @QtCore.Slot()
    def _routine_started(self) -> None:
        self.queue_changed.emit()

    def _insert_if_current(self, result: _ExtensionRoutineResult) -> None:
        node = self._manager._tool_graph.nodes.get(result.job.input_uid)
        if node is None or node.snapshot_token != result.job.input_snapshot:
            logger.info(
                "Discarding stale extension result",
                extra={
                    "manager_session_id": self._manager_session_id,
                    "extension_id": result.job.extension_id,
                    "extension_source_hash": result.job.source_hash,
                    "capability_id": result.job.routine.id,
                    "input_uid": result.job.input_uid,
                    "input_snapshot": result.job.input_snapshot,
                    "extension_status": "stale-input",
                },
            )
            return
        operation = ExtensionRoutineOperation(
            extension_id=result.job.extension_id,
            source_hash=result.job.source_hash,
            routine_id=result.job.routine.id,
            extension_name=result.job.extension_name,
            routine_name=result.job.routine.name,
            parameters=result.job.parameters,
        )
        provenance = compose_display_provenance(
            node.displayed_provenance_spec,
            full_data(operation),
            parent_data=result.job.input_data,
        )
        tool = ImageTool(
            result.output,
            options_model=self._manager.effective_interactive_options,
        )
        self._manager.add_imagetool(
            tool,
            activate=True,
            provenance_spec=provenance,
            replay_source_data=result.job.input_data,
        )

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        with self._blocking_tasks_lock:
            self._accepting = False
            blocking_tasks = tuple(self._blocking_tasks)
        pending = tuple(self._pending)
        self._pending.clear()
        for job in pending:
            waiter = self._routine_waiters.pop(job.job_id, None)
            if waiter is None:
                continue
            waiter.result = _ExtensionRoutineResult(
                job=job,
                output=None,
                duration=0.0,
                status="discarded",
            )
            waiter.loop.quit()
        self.queue_changed.emit()
        for task in blocking_tasks:
            task.cancel_if_pending()
        self._pool.clear()
        if self._active is not None or self._pool.activeThreadCount() > 0:
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Waiting for extension code to finish..."
            ):
                self._pool.waitForDone()
            active = self._active
            if active is not None:
                with contextlib.suppress(TypeError, RuntimeError):
                    active[1].signals.finished.disconnect(self._finished_slot)
                with contextlib.suppress(TypeError, RuntimeError):
                    active[1].signals.started.disconnect(self._started_slot)
                waiter = self._routine_waiters.pop(active[0].job_id, None)
                if waiter is not None:
                    waiter.result = active[1].result or _ExtensionRoutineResult(
                        job=active[0],
                        output=None,
                        duration=0.0,
                        status="discarded",
                    )
                    waiter.loop.quit()
                self._active = None
        with self._blocking_tasks_lock:
            self._blocking_tasks.clear()
        self._script_modules.clear()
        _remove_manager_modules(self._manager_session_id)
        self._shutdown_complete = True
