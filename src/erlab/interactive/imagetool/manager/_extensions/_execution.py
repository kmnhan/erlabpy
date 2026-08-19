"""Serialized in-process execution for ImageTool Manager extensions."""

from __future__ import annotations

import collections.abc
import contextlib
import copy
import dataclasses
import hashlib
import logging
import os
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
)
from erlab.extensions._api import (
    _CapabilityStatus,
    _coerce_call_parameters,
    _load_script_bytes,
)
from erlab.extensions._models import _require_finite_parameter_values, _script_name_key
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool._provenance._model import (
    compose_display_provenance,
    full_data,
)
from erlab.interactive.imagetool._provenance._operations import (
    ExtensionRoutineOperation,
)
from erlab.interactive.imagetool.manager._extensions._catalog import (
    _capability_descriptor,
    _capability_status,
    _ExtensionCatalogConflictError,
    _ExtensionCatalogError,
    _ExtensionCatalogStore,
    _PinnedScript,
)
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionCatalogModel,
    _script_loader_name_filters,
    _ScriptRecord,
)
from erlab.io.dataloader import LoaderBase

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from erlab.interactive.imagetool._provenance._model import ToolProvenanceOperation
    from erlab.interactive.imagetool.manager._extensions._catalog import (
        _ExtensionCatalog,
    )
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager

logger = logging.getLogger(__name__)


def _manager_module_name(
    manager_session_id: str,
    script_name: str,
    source_hash: str,
    registered_path: pathlib.Path,
) -> str:
    """Return an import name isolated to one manager and local script path."""
    safe_session_id = re.sub(r"\W", "_", manager_session_id)
    safe_script_name = re.sub(r"\W", "_", script_name)
    script_token = hashlib.sha256(_script_name_key(script_name).encode()).hexdigest()[
        :12
    ]
    path_token = hashlib.sha256(os.fspath(registered_path).encode()).hexdigest()[:12]
    return (
        f"_erlab_extension_{safe_session_id}_{safe_script_name}_"
        f"{script_token}_{path_token}_{source_hash}"
    )


def _remove_manager_modules(manager_session_id: str) -> None:
    """Release script modules after their owning manager stops all workers."""
    safe_session_id = re.sub(r"\W", "_", manager_session_id)
    prefix = f"_erlab_extension_{safe_session_id}_"
    for module_name in tuple(sys.modules):
        if module_name.startswith(prefix):
            sys.modules.pop(module_name, None)


def _cached_script(
    modules: dict[tuple[str, str, str], LoadedScript],
    *,
    snapshot: _PinnedScript,
    module_name: str,
) -> LoadedScript:
    """Import one exact script once during a manager execution lifetime."""
    key = (
        _script_name_key(snapshot.record.script_name),
        snapshot.record.source_hash,
        os.fspath(snapshot.registered_path),
    )
    loaded = modules.get(key)
    if loaded is None:
        loaded = _load_script_bytes(
            snapshot.source_bytes,
            snapshot.registered_path,
            module_name=module_name,
            expected_source_hash=snapshot.record.source_hash,
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


class _ExtensionSourceLoadFailure(ExtensionExecutionError):
    """Identify a failure that occurred before extension function invocation."""


@dataclasses.dataclass(frozen=True)
class _ExecutionCapability:
    """One authoritative execution admission snapshot."""

    status: _CapabilityStatus
    snapshot: _PinnedScript | None = None
    descriptor: RoutineDescriptor | LoaderDescriptor | None = None


def _resolve_execution_script(
    catalog_store: _ExtensionCatalogStore,
    script_name: str,
    source_hash: str,
    *,
    source_is_healthy: Callable[[str, str], bool],
) -> _ExecutionCapability:
    """Resolve one verified local-script snapshot and manager-local health."""
    try:
        snapshot = catalog_store.resolve_script(script_name, source_hash)
    except _ExtensionCatalogConflictError:
        return _ExecutionCapability("hash-mismatch")
    except (FileNotFoundError, KeyError, _ExtensionCatalogError):
        return _ExecutionCapability("missing-source")
    status: _CapabilityStatus = (
        "ready" if source_is_healthy(script_name, source_hash) else "validation-failed"
    )
    return _ExecutionCapability(status, snapshot)


def _resolve_execution_capability(
    catalog_store: _ExtensionCatalogStore,
    script_name: str,
    source_hash: str,
    kind: typing.Literal["routine", "loader"],
    capability_id: str,
    *,
    source_is_healthy: Callable[[str, str], bool],
) -> _ExecutionCapability:
    """Resolve one capability from one verified local-script snapshot."""
    script = _resolve_execution_script(
        catalog_store,
        script_name,
        source_hash,
        source_is_healthy=source_is_healthy,
    )
    snapshot = script.snapshot
    if snapshot is None:
        return script
    status = _capability_status(snapshot, kind, capability_id)
    if script.status == "validation-failed":
        status = script.status
    descriptor = _capability_descriptor(snapshot, kind, capability_id)
    return _ExecutionCapability(status, snapshot, descriptor)


@dataclasses.dataclass(frozen=True)
class _ExtensionRoutineJob:
    """Pinned source and input identity for one queued routine call."""

    job_id: str
    snapshot: _PinnedScript
    routine: RoutineDescriptor
    parameters: dict[str, typing.Any]
    input_uid: str
    input_snapshot: str
    input_data: xr.DataArray

    @property
    def script_name(self) -> str:
        return self.snapshot.record.script_name

    @property
    def source_hash(self) -> str:
        return self.snapshot.record.source_hash


@dataclasses.dataclass(frozen=True)
class _ExtensionRoutineResult:
    job: _ExtensionRoutineJob
    output: xr.DataArray | None
    duration: float
    status: typing.Literal["success", "failed", "discarded"]
    traceback_text: str | None = None
    source_failure: bool = False


@dataclasses.dataclass
class _ExtensionRoutineWaiter:
    """Own the nested event loop for one synchronous manager replay call."""

    loop: QtCore.QEventLoop
    result: _ExtensionRoutineResult | None = None


@dataclasses.dataclass
class _ReplaySourceCapture:
    """Stage exact sources until the caller publishes replayed data.

    A nested capture merges into its parent. Only a published outer capture writes
    recovery bytes into workspace state. This keeps failed validation and canceled
    replay attempts from changing the document.
    """

    publication_checker: Callable[[_ReplaySourceCapture], None] = dataclasses.field(
        repr=False
    )
    permits: dict[
        tuple[str, str, typing.Literal["routine", "loader"], str],
        _PinnedScript,
    ] = dataclasses.field(default_factory=dict)
    checked: bool = False
    published: bool = False

    def require_current_for_publication(self) -> None:
        """Recheck every staged capability before the caller changes manager data."""
        self.publication_checker(self)
        self.checked = True

    def publish(self) -> None:
        """Mark the staged sources as belonging to successfully published data."""
        if self.permits and not self.checked:
            raise RuntimeError(
                "Replay capabilities must be checked before result publication"
            )
        self.published = True


@dataclasses.dataclass(frozen=True)
class _ExtensionLoaderCall:
    """Pinned decorated loader callable passed to existing file workers."""

    manager_session_id: str
    snapshot: _PinnedScript
    loader_id: str
    descriptor: LoaderDescriptor
    executor: Callable[
        [_ExtensionLoaderCall, pathlib.Path, dict[str, typing.Any]],
        xr.DataArray | xr.Dataset | xr.DataTree,
    ] = dataclasses.field(repr=False, compare=False)
    publication_checker: Callable[[_ExtensionLoaderCall], None] = dataclasses.field(
        repr=False, compare=False
    )
    publication_recorder: Callable[[_ExtensionLoaderCall], None] = dataclasses.field(
        repr=False, compare=False
    )

    @property
    def script_name(self) -> str:
        return self.snapshot.record.script_name

    @property
    def source_hash(self) -> str:
        return self.snapshot.record.source_hash

    @property
    def registered_path(self) -> pathlib.Path:
        return self.snapshot.registered_path

    @property
    def manager_loader_name(self) -> str:
        return f"{self.script_name}:{self.loader_id}"

    def require_current_for_publication(self) -> None:
        """Reject a result if this exact loader is no longer executable."""
        self.publication_checker(self)

    def record_publication(self) -> None:
        """Retain this verified source after Manager insertion succeeds."""
        self.publication_recorder(self)

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
        script_modules: dict[tuple[str, str, str], LoadedScript],
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        """Import and invoke this pinned loader from the extension worker."""
        started = time.perf_counter()
        fields = {
            "manager_session_id": self.manager_session_id,
            "catalog_generation": self.snapshot.catalog_generation,
            "extension_script_name": self.script_name,
            "extension_source_hash": self.source_hash,
            "capability_id": self.loader_id,
            "extension_source": str(self.registered_path),
            "parameters": parameters,
            "file_path": str(path),
        }
        try:
            try:
                logger.info("Importing extension loader source", extra=fields)
                loaded = _cached_script(
                    script_modules,
                    snapshot=self.snapshot,
                    module_name=_manager_module_name(
                        self.manager_session_id,
                        self.script_name,
                        self.source_hash,
                        self.registered_path,
                    ),
                )
                entry = loaded.erlab.loaders.get(self.loader_id)
                entry = _require_loader_entry(self, entry)
            except BaseException as error:
                raise _ExtensionSourceLoadFailure(
                    f"Extension loader {self.loader_id!r} could not be imported"
                ) from error
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
                "output": _xarray_log_fields(result),
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
        script_modules: dict[tuple[str, str, str], LoadedScript],
        *,
        source_is_healthy: Callable[[str, str], bool],
    ) -> None:
        super().__init__()
        self.call = call
        self.path = path
        self.parameters = parameters
        self.catalog_store = catalog_store
        self.script_modules = script_modules
        self.source_is_healthy = source_is_healthy
        self.signals = _ExtensionLoaderSignals()
        self.done = threading.Event()
        self.output: xr.DataArray | xr.Dataset | xr.DataTree | None = None
        self.error: Exception | None = None
        self.traceback_text: str | None = None
        self.source_failure = False
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
            resolution = _resolve_execution_capability(
                self.catalog_store,
                self.call.script_name,
                self.call.source_hash,
                "loader",
                self.call.loader_id,
                source_is_healthy=self.source_is_healthy,
            )
            if resolution.status != "ready":
                self.error = ExtensionExecutionError(
                    "The extension loader is unavailable in this manager session"
                )
                logger.info(
                    "Discarding unavailable queued extension loader",
                    extra={
                        "manager_session_id": self.call.manager_session_id,
                        "catalog_generation": self.call.snapshot.catalog_generation,
                        "extension_script_name": self.call.script_name,
                        "extension_source_hash": self.call.source_hash,
                        "capability_id": self.call.loader_id,
                        "extension_status": f"{resolution.status}-before-start",
                    },
                )
            else:
                self.output = self.call._invoke(
                    self.path, self.parameters, self.script_modules
                )
        except BaseException as error:
            self.traceback_text = traceback.format_exc()
            self.source_failure = isinstance(error, _ExtensionSourceLoadFailure)
            self.error = _extension_error(error, "loader execution")
        finally:
            self.done.set()
            self.signals.finished.emit()


class _ExtensionValidationWorker(QtCore.QRunnable):
    """Validate and enable one source on the serialized extension queue."""

    def __init__(
        self,
        script_name: str,
        source_hash: str,
        expected_record_generation: int,
        *,
        manager_session_id: str,
        catalog_store: _ExtensionCatalogStore,
        script_modules: dict[tuple[str, str, str], LoadedScript],
        check_loader_filter_conflicts: bool = True,
        enable_script: bool = True,
        persist_result: bool = True,
    ) -> None:
        super().__init__()
        self.script_name = script_name
        self.source_hash = source_hash
        self.expected_record_generation = expected_record_generation
        self.manager_session_id = manager_session_id
        self.catalog_store = catalog_store
        self.script_modules = script_modules
        self.check_loader_filter_conflicts = check_loader_filter_conflicts
        self.enable_script = enable_script
        self.persist_result = persist_result
        self.signals = _ExtensionLoaderSignals()
        self.done = threading.Event()
        self.output: _ExtensionCatalogModel | None = None
        self.error: Exception | None = None
        self.traceback_text: str | None = None
        self.source_failure = False
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
            "extension_script_name": self.script_name,
            "extension_source_hash": self.source_hash,
            "capability_id": None,
            "extension_source": None,
        }
        try:
            snapshot = self.catalog_store.resolve_script(
                self.script_name, self.source_hash
            )
            fields.update(
                {
                    "catalog_generation": snapshot.catalog_generation,
                    "extension_source": str(snapshot.registered_path),
                }
            )
            logger.info("Importing extension source for validation", extra=fields)
            self.output = _validate_script_snapshot(
                self.catalog_store,
                snapshot,
                expected_record_generation=self.expected_record_generation,
                manager_session_id=self.manager_session_id,
                script_modules=self.script_modules,
                check_loader_filter_conflicts=self.check_loader_filter_conflicts,
                enable_script=self.enable_script,
                persist_result=self.persist_result,
            )
        except BaseException as error:
            self.traceback_text = traceback.format_exc()
            self.source_failure = not isinstance(
                error,
                (_ExtensionCatalogError, FileNotFoundError, KeyError),
            )
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
    def script_name(self) -> str:
        return self._extension_call.script_name

    @property
    def source_hash(self) -> str:
        return self._extension_call.source_hash

    @property
    def loader_id(self) -> str:
        return self._extension_call.loader_id

    @property
    def registered_path(self) -> pathlib.Path:
        return self._extension_call.registered_path

    def require_current_for_publication(self) -> None:
        """Reject a loaded result if the registered loader changed."""
        self._extension_call.require_current_for_publication()

    def record_publication(self) -> None:
        """Retain the verified loader source after Manager insertion succeeds."""
        self._extension_call.record_publication()

    @property
    def file_dialog_methods(
        self,
    ) -> dict[str, tuple[Callable[..., typing.Any], dict[str, typing.Any]]]:
        patterns = " ".join(f"*{value}" for value in self.descriptor.extensions) or "*"
        return {f"{self.descriptor.name} ({patterns})": (self.load_for_manager, {})}

    @property
    def descriptor(self) -> LoaderDescriptor:
        return self._extension_call.descriptor

    def load_for_manager(
        self, file_path: str | pathlib.Path, **parameters: typing.Any
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        """Keep extension parameters outside ``LoaderBase.load`` controls."""
        result = self.load(file_path, load_kwargs=parameters)
        if isinstance(result, xr.DataArray | xr.Dataset | xr.DataTree):
            return result
        raise ExtensionExecutionError(
            "An extension loader must return one xarray object"
        )

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


def _xarray_log_fields(
    data: xr.DataArray | xr.Dataset | xr.DataTree,
) -> dict[str, typing.Any]:
    fields: dict[str, typing.Any] = {"type": type(data).__name__}
    if isinstance(data, xr.DataArray):
        fields.update(
            {
                "dimensions": tuple(str(dim) for dim in data.dims),
                "shape": data.shape,
                "dtype": str(data.dtype),
            }
        )
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


def _require_routine(
    loaded: erlab.extensions.LoadedScript, routine_id: str
) -> tuple[RoutineDescriptor, Callable[..., typing.Any]]:
    entry = loaded.erlab.routines.get(routine_id)
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
    record: _ScriptRecord,
) -> None:
    """Reject filters that would hide a built-in file loader."""
    candidate_filters = set(_script_loader_name_filters(record))
    if not candidate_filters:
        return
    builtin_filters = set(erlab.interactive.utils.file_loaders())
    conflicts = sorted(candidate_filters.intersection(builtin_filters))
    if conflicts:
        joined = ", ".join(repr(value) for value in conflicts)
        raise _ExtensionCatalogConflictError(
            f"Script {record.script_name!r} conflicts with built-in file dialog "
            f"filters: {joined}"
        )


def _validate_script_snapshot(
    catalog_store: _ExtensionCatalogStore,
    snapshot: _PinnedScript,
    *,
    expected_record_generation: int,
    manager_session_id: str,
    script_modules: dict[tuple[str, str, str], LoadedScript],
    check_loader_filter_conflicts: bool = True,
    enable_script: bool = True,
    persist_result: bool = True,
) -> _ExtensionCatalogModel:
    """Validate one pinned local script, then commit its descriptors."""
    record = snapshot.record
    if record.record_generation != expected_record_generation:
        raise _ExtensionCatalogConflictError(
            f"Script {record.script_name!r} changed before validation"
        )
    loaded = _cached_script(
        script_modules,
        snapshot=snapshot,
        module_name=_manager_module_name(
            manager_session_id,
            record.script_name,
            record.source_hash,
            snapshot.registered_path,
        ),
    )
    routines = tuple(item[0] for item in loaded.erlab.routines.values())
    loaders = tuple(item[0] for item in loaded.erlab.loaders.values())
    validated_record = record.model_copy(
        update={
            "routines": routines,
            "loaders": loaders,
        }
    )
    if check_loader_filter_conflicts:
        _reject_builtin_loader_filter_conflicts(validated_record)
    if not persist_result:
        return catalog_store.read()
    return catalog_store.commit_script_validation(
        record.script_name,
        source_hash=record.source_hash,
        expected_record_generation=expected_record_generation,
        routines=routines,
        loaders=loaders,
        enable_script=enable_script,
    )


class _ExtensionRoutineWorker(QtCore.QRunnable):
    """Run one pinned source snapshot without accessing Qt widgets."""

    def __init__(
        self,
        job: _ExtensionRoutineJob,
        *,
        manager_session_id: str,
        catalog_store: _ExtensionCatalogStore,
        script_modules: dict[tuple[str, str, str], LoadedScript],
        source_is_healthy: Callable[[str, str], bool],
    ) -> None:
        super().__init__()
        self.job = job
        self.manager_session_id = manager_session_id
        self.catalog_store = catalog_store
        self.script_modules = script_modules
        self.source_is_healthy = source_is_healthy
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
        source_failure = False
        fields: dict[str, typing.Any] = {
            "manager_session_id": self.manager_session_id,
            "catalog_generation": self.job.snapshot.catalog_generation,
            "extension_script_name": self.job.script_name,
            "extension_source_hash": self.job.source_hash,
            "capability_id": self.job.routine.id,
            "extension_source": str(self.job.snapshot.registered_path),
            "parameters": self.job.parameters,
            "input_uid": self.job.input_uid,
            "input_snapshot": self.job.input_snapshot,
            "input": _xarray_log_fields(self.job.input_data),
        }
        try:
            resolution = _resolve_execution_capability(
                self.catalog_store,
                self.job.script_name,
                self.job.source_hash,
                "routine",
                self.job.routine.id,
                source_is_healthy=self.source_is_healthy,
            )
            if resolution.status != "ready":
                status = "discarded"
                logger.info(
                    "Discarding unavailable queued extension routine",
                    extra={
                        **fields,
                        "extension_status": f"{resolution.status}-before-start",
                    },
                )
                return
            self._started.set()
            self.signals.started.emit()
            logger.info("Importing extension source", extra=fields)
            module_name = _manager_module_name(
                self.manager_session_id,
                self.job.script_name,
                self.job.source_hash,
                self.job.snapshot.registered_path,
            )
            try:
                loaded = _cached_script(
                    self.script_modules,
                    snapshot=self.job.snapshot,
                    module_name=module_name,
                )
                entry = _require_routine(loaded, self.job.routine.id)
            except BaseException as error:
                source_failure = True
                raise _ExtensionSourceLoadFailure(
                    f"Extension routine {self.job.routine.id!r} could not be imported"
                ) from error
            parameters = _coerce_call_parameters(entry[1], self.job.parameters)
            logger.info("Invoking extension routine", extra=fields)
            result = _require_dataarray(
                entry[1](_readonly_array(self.job.input_data), **parameters)
            )
            result = _detached_routine_output(result, self.job.input_data)
            erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(result)
            output = result
            status = "success"
            fields["output"] = _xarray_log_fields(result)
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
                source_failure=source_failure,
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
    validation_changed = QtCore.Signal()

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
        self._script_modules: dict[tuple[str, str, str], LoadedScript] = {}
        self._pending: deque[_ExtensionRoutineJob] = deque()
        self._active: tuple[_ExtensionRoutineJob, _ExtensionRoutineWorker] | None = None
        self._routine_waiters: dict[str, _ExtensionRoutineWaiter] = {}
        self._replay_source_captures: list[_ReplaySourceCapture] = []
        self._blocking_tasks: set[
            _ExtensionLoaderWorker | _ExtensionValidationWorker
        ] = set()
        self._blocking_tasks_lock = threading.Lock()
        self._validation_errors: dict[tuple[str, str], str] = {}
        self._validation_errors_lock = threading.Lock()
        self._accepting = True
        self._shutdown_complete = False
        self._finished_slot = self._finished
        self._started_slot = self._routine_started

    @property
    def validation_errors(self) -> dict[tuple[str, str], str]:
        """Return manager-local failures keyed by normalized script and source."""
        with self._validation_errors_lock:
            return dict(self._validation_errors)

    def validation_error(self, script_name: str, source_hash: str) -> str | None:
        """Return a source failure observed only by this manager process."""
        with self._validation_errors_lock:
            return self._validation_errors.get(
                (_script_name_key(script_name), source_hash)
            )

    def _source_is_healthy(self, script_name: str, source_hash: str) -> bool:
        return self.validation_error(script_name, source_hash) is None

    def _set_validation_error(
        self, script_name: str, source_hash: str, detail: str | None
    ) -> None:
        key = (_script_name_key(script_name), source_hash)
        changed = False
        with self._validation_errors_lock:
            if detail is None:
                changed = self._validation_errors.pop(key, None) is not None
            elif self._validation_errors.get(key) != detail:
                self._validation_errors[key] = detail
                changed = True
        if changed:
            self.validation_changed.emit()

    def prune_validation_errors(self, catalog: _ExtensionCatalogModel) -> None:
        """Discard failures for sources that are no longer current."""
        current = {
            (_script_name_key(record.script_name), record.source_hash)
            for record in catalog.extensions.values()
        }
        with self._validation_errors_lock:
            for key in tuple(self._validation_errors):
                if key not in current:
                    self._validation_errors.pop(key)

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

    def uses_script(self, script_name: str) -> bool:
        """Return whether this manager retains work for one registered script."""
        script_key = _script_name_key(script_name)
        active = self._active
        if active is not None and _script_name_key(active[0].script_name) == script_key:
            return True
        if any(
            _script_name_key(job.script_name) == script_key for job in self._pending
        ):
            return True
        with self._blocking_tasks_lock:
            return any(
                _script_name_key(
                    task.script_name
                    if isinstance(task, _ExtensionValidationWorker)
                    else task.call.script_name
                )
                == script_key
                for task in self._blocking_tasks
            )

    def capability_status(
        self,
        script_name: str,
        source_hash: str,
        kind: typing.Literal["routine", "loader"],
        capability_id: str,
    ) -> _CapabilityStatus:
        """Return the executable state from one verified local snapshot."""
        return _resolve_execution_capability(
            self._catalog.store,
            script_name,
            source_hash,
            kind,
            capability_id,
            source_is_healthy=self._source_is_healthy,
        ).status

    def ready_routines(
        self,
        script_name: str,
        source_hash: str,
    ) -> tuple[RoutineDescriptor, ...]:
        """Return ready routines after one verified local-script read."""
        snapshot = self._ready_script_snapshot(script_name, source_hash)
        if snapshot is None:
            return ()
        return tuple(
            descriptor
            for descriptor in snapshot.record.routines
            if _capability_status(snapshot, "routine", descriptor.id) == "ready"
        )

    def ready_loader_calls(
        self,
        script_name: str,
        source_hash: str,
    ) -> tuple[_ExtensionLoaderCall, ...]:
        """Pin all ready loaders from one verified local-script read."""
        snapshot = self._ready_script_snapshot(script_name, source_hash)
        if snapshot is None:
            return ()
        return tuple(
            self._loader_call_from_snapshot(snapshot, descriptor)
            for descriptor in snapshot.record.loaders
            if _capability_status(snapshot, "loader", descriptor.id) == "ready"
        )

    def _ready_script_snapshot(
        self,
        script_name: str,
        source_hash: str,
    ) -> _PinnedScript | None:
        """Resolve one ready script without importing it."""
        resolution = _resolve_execution_script(
            self._catalog.store,
            script_name,
            source_hash,
            source_is_healthy=self._source_is_healthy,
        )
        snapshot = resolution.snapshot
        if snapshot is None or resolution.status != "ready":
            return None
        record = snapshot.record
        if not record.approved or not record.enabled:
            return None
        return snapshot

    def loader_call(
        self,
        script_name: str,
        source_hash: str,
        loader_id: str,
    ) -> _ExtensionLoaderCall:
        """Pin one executable loader for Manager file-ingress paths."""
        self._require_workspace_extension_publication()
        resolution = _resolve_execution_capability(
            self._catalog.store,
            script_name,
            source_hash,
            "loader",
            loader_id,
            source_is_healthy=self._source_is_healthy,
        )
        if (
            resolution.status != "ready"
            or resolution.snapshot is None
            or not isinstance(resolution.descriptor, LoaderDescriptor)
        ):
            raise ExtensionExecutionError(
                f"The extension loader is not available: {resolution.status}"
            )
        return self._loader_call_from_snapshot(
            resolution.snapshot, resolution.descriptor
        )

    def _loader_call_from_snapshot(
        self,
        snapshot: _PinnedScript,
        descriptor: LoaderDescriptor,
    ) -> _ExtensionLoaderCall:
        """Build one loader call from an already verified snapshot."""
        return _ExtensionLoaderCall(
            manager_session_id=self._manager_session_id,
            snapshot=snapshot,
            loader_id=descriptor.id,
            descriptor=descriptor,
            executor=self.run_loader,
            publication_checker=self.require_loader_publication,
            publication_recorder=self._record_loader_publication,
        )

    def _record_loader_publication(self, call: _ExtensionLoaderCall) -> None:
        """Retain one loader source only after a caller publishes its result."""
        self._manager._workspace_state.extension_scripts.remember_verified_source(
            call.script_name,
            call.source_hash,
            call.snapshot.source_bytes,
        )

    @contextlib.contextmanager
    def capture_replay_sources(self) -> Iterator[_ReplaySourceCapture]:
        """Capture synchronous replay sources until final Manager publication.

        The caller must call :meth:`_ReplaySourceCapture.publish` only after it
        inserts or replaces the replay result. An exception or an unpublished exit
        discards the staged sources. A successful nested capture merges into its
        parent so only the outer publication changes workspace state.
        """
        capture = _ReplaySourceCapture(publication_checker=self._check_replay_capture)
        self._replay_source_captures.append(capture)
        completed = False
        try:
            yield capture
            completed = True
        finally:
            if (
                not self._replay_source_captures
                or self._replay_source_captures[-1] is not capture
            ):
                raise RuntimeError("replay source captures must exit in stack order")
            self._replay_source_captures.pop()
            if completed and capture.published and self._replay_source_captures:
                parent = self._replay_source_captures[-1]
                parent.permits.update(capture.permits)
                parent.checked = False
            elif completed and capture.published:
                snapshots = {
                    (
                        _script_name_key(snapshot.record.script_name),
                        snapshot.record.source_hash,
                    ): snapshot
                    for snapshot in capture.permits.values()
                }
                for snapshot in snapshots.values():
                    self._manager._workspace_state.extension_scripts.remember_verified_source(
                        snapshot.record.script_name,
                        snapshot.record.source_hash,
                        snapshot.source_bytes,
                    )

    def _check_replay_capture(self, capture: _ReplaySourceCapture) -> None:
        """Reject publication when any capability in a replay chain is stale."""
        for (
            _script_key,
            _source_hash,
            kind,
            capability_id,
        ), snapshot in capture.permits.items():
            self._require_current_capability(
                snapshot.record.script_name,
                snapshot.record.source_hash,
                kind,
                capability_id,
            )

    def stage_replay_source(
        self,
        snapshot: _PinnedScript,
        kind: typing.Literal["routine", "loader"],
        capability_id: str,
    ) -> None:
        """Stage one capability when synchronous replay runs inside a capture."""
        if not self._replay_source_captures:
            return
        key = (
            _script_name_key(snapshot.record.script_name),
            snapshot.record.source_hash,
            kind,
            capability_id,
        )
        capture = self._replay_source_captures[-1]
        capture.permits[key] = snapshot
        capture.checked = False

    def run_loader(
        self,
        call: _ExtensionLoaderCall,
        path: pathlib.Path,
        parameters: dict[str, typing.Any],
    ) -> xr.DataArray | xr.Dataset | xr.DataTree:
        """Run a loader synchronously on this manager's extension thread pool."""
        self._require_workspace_extension_publication()
        task = _ExtensionLoaderWorker(
            call,
            path,
            parameters,
            self._catalog.store,
            self._script_modules,
            source_is_healthy=self._source_is_healthy,
        )
        try:
            self._run_blocking_task(task)
        except BaseException:
            if task.source_failure:
                self._set_validation_error(
                    call.script_name, call.source_hash, task.traceback_text
                )
            raise
        if task.output is None:
            raise ExtensionExecutionError("The extension loader returned no result")
        self._set_validation_error(call.script_name, call.source_hash, None)
        call.require_current_for_publication()
        self.stage_replay_source(call.snapshot, "loader", call.loader_id)
        return task.output

    def _require_current_capability(
        self,
        script_name: str,
        source_hash: str,
        kind: typing.Literal["routine", "loader"],
        capability_id: str,
    ) -> None:
        """Reject output when its capability changed or stopped before delivery."""
        self._require_workspace_extension_publication()
        if not self._accepting:
            raise ExtensionExecutionError("Extension execution is shutting down")
        resolution = _resolve_execution_capability(
            self._catalog.store,
            script_name,
            source_hash,
            kind,
            capability_id,
            source_is_healthy=self._source_is_healthy,
        )
        if resolution.status != "ready" or resolution.snapshot is None:
            raise ExtensionExecutionError(
                f"The extension result became unavailable: {resolution.status}"
            )
        self._manager._extensions._canonicalize_workspace_script_name(
            resolution.snapshot.record.script_name,
            resolution.snapshot.record.source_hash,
        )

    def _require_workspace_extension_publication(self) -> None:
        """Reject extension results that cannot be serialized without data loss."""
        scripts = self._manager._workspace_state.extension_scripts
        if (
            scripts.opaque_requirement_container is not None
            or scripts.opaque_source_container is not None
        ):
            raise ExtensionExecutionError(
                "This workspace contains unsupported extension metadata. "
                "Extension results cannot be added without losing recovery data."
            )

    def require_loader_publication(self, call: _ExtensionLoaderCall) -> None:
        """Reject a loader result if its pinned capability is no longer current."""
        self._require_current_capability(
            call.script_name,
            call.source_hash,
            "loader",
            call.loader_id,
        )

    def validate_script(
        self,
        script_name: str,
        source_hash: str,
        *,
        expected_record_generation: int,
        enable_script: bool = True,
        persist_result: bool = True,
    ) -> _ExtensionCatalogModel:
        """Validate one exact registered script on the extension thread."""
        task = _ExtensionValidationWorker(
            script_name,
            source_hash,
            expected_record_generation,
            manager_session_id=self._manager_session_id,
            catalog_store=self._catalog.store,
            script_modules=self._script_modules,
            check_loader_filter_conflicts=enable_script,
            enable_script=enable_script,
            persist_result=persist_result,
        )
        try:
            self._run_blocking_task(task, wait_message="Validating extension...")
        except BaseException:
            if task.source_failure:
                self._set_validation_error(
                    script_name, source_hash, task.traceback_text
                )
            raise
        if task.output is None:
            raise ExtensionExecutionError("Extension validation returned no result")
        self._set_validation_error(script_name, source_hash, None)
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
        script_name: str,
        source_hash: str,
        routine_id: str,
        parameters: Mapping[str, typing.Any],
        target: int | str,
    ) -> str:
        if not self._accepting:
            raise RuntimeError("Extension execution is shutting down")
        node = self._manager._node_for_target(target)
        data = node.data_for_role("displayed")
        job = self._routine_job(
            script_name=script_name,
            source_hash=source_hash,
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
            script_name=operation.script_name,
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
        self._require_current_capability(
            job.script_name,
            job.source_hash,
            "routine",
            job.routine.id,
        )
        self.stage_replay_source(job.snapshot, "routine", job.routine.id)
        return result.output

    def _routine_job(
        self,
        *,
        script_name: str,
        source_hash: str,
        routine_id: str,
        parameters: Mapping[str, typing.Any],
        input_data: xr.DataArray,
        input_uid: str,
        input_snapshot: str,
    ) -> _ExtensionRoutineJob:
        """Pin catalog state and input identity before queue admission."""
        self._require_workspace_extension_publication()
        _require_finite_parameter_values(parameters)
        if self._catalog.load_error is not None:
            raise ExtensionExecutionError("The extension catalog is unavailable")
        catalog_store = self._catalog.store
        resolution = _resolve_execution_capability(
            catalog_store,
            script_name,
            source_hash,
            "routine",
            routine_id,
            source_is_healthy=self._source_is_healthy,
        )
        if resolution.status != "ready" or resolution.snapshot is None:
            raise ExtensionExecutionError(
                f"The extension is not enabled or available: {resolution.status}"
            )
        routine = resolution.descriptor
        if not isinstance(routine, RoutineDescriptor):
            raise ExtensionExecutionError(f"Routine {routine_id!r} is not available")
        return _ExtensionRoutineJob(
            job_id=uuid.uuid4().hex,
            snapshot=resolution.snapshot,
            routine=routine,
            parameters=dict(parameters),
            input_uid=input_uid,
            input_snapshot=input_snapshot,
            input_data=input_data,
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
                catalog_store=self._catalog.store,
                script_modules=self._script_modules,
                source_is_healthy=self._source_is_healthy,
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
        if result.source_failure:
            self._set_validation_error(
                result.job.script_name,
                result.job.source_hash,
                result.traceback_text,
            )
        elif result.status == "success":
            self._set_validation_error(
                result.job.script_name, result.job.source_hash, None
            )
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
        input_is_current = (
            node is not None and node.snapshot_token == result.job.input_snapshot
        )
        if not self._accepting or not input_is_current:
            if not self._accepting:
                final_status = "shutdown-after-finish"
            else:
                final_status = "stale-input"
            logger.info(
                "Discarding stale extension result",
                extra={
                    "manager_session_id": self._manager_session_id,
                    "extension_script_name": result.job.script_name,
                    "extension_source_hash": result.job.source_hash,
                    "capability_id": result.job.routine.id,
                    "input_uid": result.job.input_uid,
                    "input_snapshot": result.job.input_snapshot,
                    "extension_status": final_status,
                },
            )
            return
        if node is None:  # pragma: no cover - narrowed by input_is_current above.
            return
        operation = ExtensionRoutineOperation(
            script_name=result.job.script_name,
            source_hash=result.job.source_hash,
            routine_id=result.job.routine.id,
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
        node = self._manager._tool_graph.nodes.get(result.job.input_uid)
        try:
            self._require_current_capability(
                result.job.script_name,
                result.job.source_hash,
                "routine",
                result.job.routine.id,
            )
        except ExtensionExecutionError as error:
            publication_error = str(error)
        else:
            publication_error = None
        input_is_current = (
            node is not None and node.snapshot_token == result.job.input_snapshot
        )
        if publication_error is not None or not input_is_current:
            tool.close()
            tool.deleteLater()
            final_status = (
                publication_error
                if publication_error is not None
                else "stale-input-before-insert"
            )
            logger.info(
                "Discarding stale extension result",
                extra={
                    "manager_session_id": self._manager_session_id,
                    "extension_script_name": result.job.script_name,
                    "extension_source_hash": result.job.source_hash,
                    "capability_id": result.job.routine.id,
                    "input_uid": result.job.input_uid,
                    "input_snapshot": result.job.input_snapshot,
                    "extension_status": final_status,
                },
            )
            return
        self._manager.add_imagetool(
            tool,
            activate=True,
            provenance_spec=provenance,
            replay_source_data=result.job.input_data,
        )
        self._manager._workspace_state.extension_scripts.remember_verified_source(
            result.job.script_name,
            result.job.source_hash,
            result.job.snapshot.source_bytes,
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
