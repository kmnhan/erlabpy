"""Serialized in-process execution for ImageTool Manager extensions."""

from __future__ import annotations

import collections.abc
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import inspect
import logging
import pathlib
import re
import sys
import threading
import time
import traceback
import types
import typing
import uuid
from collections import deque

import numpy as np
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
from erlab.extensions import (
    ExtensionExecutionError,
    LoadedScript,
    LoaderDescriptor,
    RoutineDescriptor,
    load_script,
)
from erlab.extensions._api import (
    _CAPABILITY_ATTRIBUTE,
    _coerce_call_parameters,
    _descriptor_for,
    _module_capabilities,
    _resolve_loader_method,
)
from erlab.extensions._entry_points import (
    _entry_point_revision,
    _load_entry_point_value,
)
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
    _EnvironmentLoaderMethod,
    _ExtensionCatalogModel,
    _ExtensionRevision,
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
    revision_hash: str,
) -> str:
    """Return an import name isolated to one manager and source revision."""
    safe_session_id = re.sub(r"\W", "_", manager_session_id)
    safe_extension_id = re.sub(r"\W", "_", extension_id)
    extension_token = hashlib.sha256(extension_id.encode()).hexdigest()[:12]
    return (
        f"_erlab_extension_{safe_session_id}_{safe_extension_id}_"
        f"{extension_token}_{revision_hash}"
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
    revision_hash: str,
    source_path: pathlib.Path,
    module_name: str,
) -> LoadedScript:
    """Import one exact script once during a manager execution lifetime."""
    key = (extension_id, revision_hash)
    loaded = modules.get(key)
    if loaded is None:
        loaded = load_script(
            source_path,
            module_name=module_name,
            expected_revision=revision_hash,
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
    revision_hash: str
    routine: RoutineDescriptor
    source_path: pathlib.Path | None
    source_type: typing.Literal["script", "environment-package"]
    entry_point_group: str | None
    entry_point_name: str | None
    entry_point_value: str | None
    parameters: dict[str, typing.Any]
    input_uid: str
    input_snapshot: str
    input_data: xr.DataArray
    catalog_generation: int


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
    revision_hash: str
    loader_id: str
    descriptor: LoaderDescriptor
    source_path: pathlib.Path | None
    source_type: typing.Literal["script", "environment-package"]
    executor: Callable[
        [_ExtensionLoaderCall, pathlib.Path, dict[str, typing.Any]],
        xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | list[xr.DataArray | xr.Dataset | xr.DataTree],
    ] = dataclasses.field(repr=False, compare=False)
    entry_point_group: str | None = None
    entry_point_name: str | None = None
    entry_point_value: str | None = None
    loader_method: str | None = None
    loader_always_single: bool = True

    @property
    def manager_loader_name(self) -> str:
        if self.entry_point_group == "erlab.io.loaders":
            return self.loader_id
        return f"{self.extension_id}:{self.loader_id}"

    @property
    def uses_standard_loader_options(self) -> bool:
        return self.entry_point_group == "erlab.io.loaders"

    def __call__(self, path: pathlib.Path, **parameters: typing.Any) -> typing.Any:
        return self.executor(self, pathlib.Path(path), dict(parameters))

    @property
    def __name__(self) -> str:
        return self.loader_method or "load"

    def _invoke(
        self,
        path: pathlib.Path,
        parameters: Mapping[str, typing.Any],
        script_modules: dict[tuple[str, str], LoadedScript],
    ) -> (
        xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | list[xr.DataArray | xr.Dataset | xr.DataTree]
    ):
        """Import and invoke this pinned loader from the extension worker."""
        started = time.perf_counter()
        fields = {
            "manager_session_id": self.manager_session_id,
            "catalog_generation": self.catalog_generation,
            "extension_id": self.extension_id,
            "extension_revision": self.revision_hash,
            "capability_id": self.loader_id,
            "extension_source": str(self.source_path or self.entry_point_value),
            "parameters": parameters,
            "file_path": str(path),
        }
        try:
            logger.info("Importing extension loader revision", extra=fields)
            if self.source_type == "script":
                loaded = _cached_script(
                    script_modules,
                    extension_id=self.extension_id,
                    revision_hash=self.revision_hash,
                    source_path=_require_loader_source(self),
                    module_name=_manager_module_name(
                        self.manager_session_id,
                        self.extension_id,
                        self.revision_hash,
                    ),
                )
                entry = loaded.loaders.get(self.loader_id)
            else:
                entry = _environment_loader(self)
            entry = _require_loader_entry(self, entry)
            values = (
                _coerce_call_parameters(entry[1], parameters)
                if getattr(entry[1], _CAPABILITY_ATTRIBUTE, None) is not None
                else dict(parameters)
            )
            logger.info("Invoking extension loader", extra=fields)
            result = _require_loader_output(
                entry[1](path, **values),
                allow_multiple=self.uses_standard_loader_options,
            )
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
        self.output: (
            xr.DataArray
            | xr.Dataset
            | xr.DataTree
            | list[xr.DataArray | xr.Dataset | xr.DataTree]
            | None
        ) = None
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
            record = self.catalog_store.read().extensions.get(self.call.extension_id)
            if record is None or record.removed or not record.enabled:
                self.error = ExtensionExecutionError("The extension is not enabled")
                logger.info(
                    "Discarding disabled queued extension loader",
                    extra={
                        "manager_session_id": self.call.manager_session_id,
                        "catalog_generation": self.call.catalog_generation,
                        "extension_id": self.call.extension_id,
                        "extension_revision": self.call.revision_hash,
                        "capability_id": self.call.loader_id,
                        "extension_status": "disabled-before-start",
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
    """Validate and enable one revision on the serialized extension queue."""

    def __init__(
        self,
        extension_id: str,
        revision_hash: str,
        expected_record_generation: int,
        *,
        manager_session_id: str,
        catalog_store: _ExtensionCatalogStore,
        script_modules: dict[tuple[str, str], LoadedScript],
    ) -> None:
        super().__init__()
        self.extension_id = extension_id
        self.revision_hash = revision_hash
        self.expected_record_generation = expected_record_generation
        self.manager_session_id = manager_session_id
        self.catalog_store = catalog_store
        self.script_modules = script_modules
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
            "extension_revision": None,
            "capability_id": None,
            "extension_source": None,
        }
        try:
            catalog = self.catalog_store.read()
            record = catalog.extensions.get(self.extension_id)
            fields.update(
                {
                    "catalog_generation": catalog.generation,
                    "extension_revision": self.revision_hash,
                    "extension_source": (
                        None
                        if record is None
                        else (
                            str(
                                self.catalog_store.source_path(
                                    self.extension_id, self.revision_hash
                                )
                            )
                            if record.source_type == "script"
                            else (
                                None
                                if (
                                    revision := record.revisions.get(self.revision_hash)
                                )
                                is None
                                else revision.entry_point_value
                            )
                        )
                    ),
                }
            )
            logger.info("Importing extension revision for validation", extra=fields)
            self.output = _validate_extension_revision(
                self.catalog_store,
                self.extension_id,
                revision_hash=self.revision_hash,
                expected_record_generation=self.expected_record_generation,
                manager_session_id=self.manager_session_id,
                script_modules=self.script_modules,
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
        self.always_single = call.loader_always_single

    @property
    def extension_id(self) -> str:
        return self._extension_call.extension_id

    @property
    def revision_hash(self) -> str:
        return self._extension_call.revision_hash

    @property
    def loader_id(self) -> str:
        return self._extension_call.loader_id

    @property
    def loader_method(self) -> str | None:
        return self._extension_call.loader_method

    @property
    def source_path(self) -> pathlib.Path | None:
        return self._extension_call.source_path

    @property
    def source_type(self) -> typing.Literal["script", "environment-package"]:
        return self._extension_call.source_type

    @property
    def entry_point_group(self) -> str | None:
        return self._extension_call.entry_point_group

    @property
    def entry_point_name(self) -> str | None:
        return self._extension_call.entry_point_name

    @property
    def file_dialog_methods(
        self,
    ) -> dict[str, tuple[Callable[..., typing.Any], dict[str, typing.Any]]]:
        patterns = " ".join(f"*{value}" for value in self.descriptor.extensions) or "*"
        return {f"{self.descriptor.name} ({patterns})": (self.load, {})}

    @property
    def descriptor(self) -> LoaderDescriptor:
        return self._extension_call.descriptor

    @property
    def uses_standard_loader_options(self) -> bool:
        return self._extension_call.uses_standard_loader_options

    def load(self, identifier: typing.Any, *args: typing.Any, **kwargs: typing.Any):
        """Preserve installed ``LoaderBase.load`` behavior through the worker."""
        if self.uses_standard_loader_options:
            if args:
                raise TypeError("Loader arguments after the path must use keywords")
            return self._extension_call(pathlib.Path(identifier), **kwargs)
        return super().load(identifier, *args, **kwargs)

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
    """Return a shallow xarray copy backed by read-only NumPy views."""
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
            attrs=dict(coordinate.attrs),
            encoding=dict(coordinate.encoding),
        )
    result = xr.DataArray(
        xr.Variable(
            data.dims,
            values,
            attrs=dict(data.attrs),
            encoding=dict(data.encoding),
        ),
        coords=xr.Coordinates(coordinates, indexes=_copied_xindexes(data)),
        name=data.name,
    )
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
    return xr.DataArray(
        xr.Variable(
            output.dims,
            values,
            attrs=dict(output.attrs),
            encoding=dict(output.encoding),
        ),
        coords=xr.Coordinates(coordinates, indexes=_copied_xindexes(output)),
        name=output.name,
    )


def _require_loader_source(call: _ExtensionLoaderCall) -> pathlib.Path:
    if call.source_path is None:
        raise ExtensionExecutionError("Script revision source is missing")
    return call.source_path


def _require_loader_entry(
    call: _ExtensionLoaderCall,
    entry: tuple[LoaderDescriptor, Callable[..., typing.Any]] | None,
) -> tuple[LoaderDescriptor, Callable[..., typing.Any]]:
    if entry is None:
        raise ExtensionExecutionError(
            f"Loader {call.loader_id!r} is missing from the revision"
        )
    return entry


def _require_loader_output(
    value: typing.Any, *, allow_multiple: bool
) -> (
    xr.DataArray
    | xr.Dataset
    | xr.DataTree
    | list[xr.DataArray | xr.Dataset | xr.DataTree]
):
    if isinstance(value, (xr.DataArray, xr.Dataset, xr.DataTree)):
        return value
    if (
        allow_multiple
        and isinstance(value, list)
        and all(
            isinstance(item, (xr.DataArray, xr.Dataset, xr.DataTree)) for item in value
        )
    ):
        return value
    expected = (
        "an xarray object or a list of xarray objects"
        if allow_multiple
        else ("an xarray object")
    )
    raise ExtensionExecutionError(
        f"Loader returned {type(value).__name__}; expected {expected}"
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
    data: xr.DataArray
    | xr.Dataset
    | xr.DataTree
    | list[xr.DataArray | xr.Dataset | xr.DataTree],
) -> dict[str, typing.Any]:
    if isinstance(data, list):
        return {
            "type": "list",
            "items": tuple(_xarray_log_fields(item) for item in data),
        }
    return _xarray_log_fields(data)


def _require_routine(
    loaded: erlab.extensions.LoadedScript, routine_id: str
) -> tuple[RoutineDescriptor, Callable[..., typing.Any]]:
    entry = loaded.routines.get(routine_id)
    if entry is None:
        raise ExtensionExecutionError(
            f"Routine {routine_id!r} is missing from the revision"
        )
    return entry


def _require_dataarray(value: typing.Any) -> xr.DataArray:
    if not isinstance(value, xr.DataArray):
        raise ExtensionExecutionError(
            f"Routine returned {type(value).__name__}; expected DataArray"
        )
    return value


def _require_script_source(job: _ExtensionRoutineJob) -> pathlib.Path:
    if job.source_path is None:
        raise ExtensionExecutionError("Script revision source is missing")
    return job.source_path


def _environment_routine(
    job: _ExtensionRoutineJob,
) -> tuple[RoutineDescriptor, Callable[..., typing.Any]]:
    for entry_point in importlib.metadata.entry_points().select(
        group=job.entry_point_group or ""
    ):
        if (
            entry_point.name != job.entry_point_name
            or entry_point.value != job.entry_point_value
        ):
            continue
        if not _environment_revision_matches(entry_point, job.revision_hash):
            continue
        value = _load_entry_point_value(entry_point, job.revision_hash)
        if isinstance(value, types.ModuleType):
            routines, _loaders = _module_capabilities(value)
            entry = routines.get(job.routine.id)
        elif (
            callable(value) and getattr(value, _CAPABILITY_ATTRIBUTE, None) is not None
        ):
            descriptor = _descriptor_for(value, getattr(value, _CAPABILITY_ATTRIBUTE))
            entry = (
                (descriptor, value)
                if isinstance(descriptor, RoutineDescriptor)
                and descriptor.id == job.routine.id
                else None
            )
        else:
            entry = None
        if entry is None:
            break
        return entry
    raise ExtensionExecutionError(
        f"Environment routine {job.routine.id!r} is no longer available"
    )


def _environment_loader(
    call: _ExtensionLoaderCall,
) -> tuple[erlab.extensions.LoaderDescriptor, Callable[..., typing.Any]] | None:
    for entry_point in importlib.metadata.entry_points().select(
        group=call.entry_point_group or ""
    ):
        if (
            entry_point.name != call.entry_point_name
            or entry_point.value != call.entry_point_value
        ):
            continue
        if not _environment_revision_matches(entry_point, call.revision_hash):
            continue
        value = _load_entry_point_value(entry_point, call.revision_hash)
        if entry_point.group == "erlab.io.loaders":
            if isinstance(value, type) and issubclass(value, LoaderBase):
                loader_instance = value()
            elif isinstance(value, LoaderBase):
                loader_instance = value
            else:
                return None
            if loader_instance.name != call.loader_id:
                return None
            return call.descriptor, _resolve_loader_method(
                loader_instance.load, call.loader_method
            )
        if isinstance(value, types.ModuleType):
            _routines, loaders = _module_capabilities(value)
            return loaders.get(call.loader_id)
        if callable(value) and getattr(value, _CAPABILITY_ATTRIBUTE, None) is not None:
            descriptor = _descriptor_for(value, getattr(value, _CAPABILITY_ATTRIBUTE))
            if (
                isinstance(descriptor, erlab.extensions.LoaderDescriptor)
                and descriptor.id == call.loader_id
            ):
                return descriptor, value
        break
    return None


def _environment_revision_matches(
    entry_point: importlib.metadata.EntryPoint, expected_revision: str
) -> bool:
    return _entry_point_revision(entry_point) == expected_revision


def _loader_method_reference(loader: LoaderBase, method: Callable) -> str | None:
    """Return a stable reference for one installed loader dialog callable."""
    if getattr(method, "__self__", None) is loader:
        name = getattr(method, "__name__", None)
        if not isinstance(name, str) or not callable(getattr(loader, name, None)):
            raise TypeError("Loader file-dialog methods must have a stable name")
        return None if name == "load" else name
    module = getattr(method, "__module__", None)
    qualname = getattr(method, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not isinstance(qualname, str)
        or "<locals>" in qualname
    ):
        raise TypeError("Loader file-dialog callables must be importable functions")
    return f"{module}.{qualname}"


def _environment_capabilities(
    catalog_store: _ExtensionCatalogStore,
    revision: _ExtensionRevision,
) -> tuple[
    tuple[RoutineDescriptor, ...],
    tuple[LoaderDescriptor, ...],
    bool | None,
    tuple[_EnvironmentLoaderMethod, ...],
]:
    """Import and validate capabilities from one exact installed entry point."""
    entry_point = catalog_store._entry_point_for_revision(revision)
    value = _load_entry_point_value(entry_point, revision.source_hash)
    if entry_point.group == "erlab.io.loaders":
        loader_type = value if isinstance(value, type) else type(value)
        if not issubclass(loader_type, LoaderBase):
            raise TypeError("erlab.io.loaders entry points must provide LoaderBase")
        loader_instance = value() if isinstance(value, type) else value
        descriptor = LoaderDescriptor(
            id=loader_instance.name,
            name=loader_instance.name.replace("_", " ").title(),
            category="Environment",
            summary=(inspect.getdoc(loader_type) or "").split("\n", maxsplit=1)[0],
            function_name="load",
            extensions=tuple(sorted(loader_instance.extensions or ())),
        )
        dialog_methods = tuple(
            _EnvironmentLoaderMethod(
                name_filter=name_filter,
                method=_loader_method_reference(loader_instance, method),
                defaults=defaults,
            )
            for name_filter, (method, defaults) in (
                loader_instance.file_dialog_methods.items()
            )
        )
        return (), (descriptor,), loader_instance.always_single, dialog_methods
    if isinstance(value, types.ModuleType):
        routines, loaders = _module_capabilities(value)
    elif callable(value) and isinstance(
        getattr(value, _CAPABILITY_ATTRIBUTE, None), collections.abc.Mapping
    ):
        descriptor = _descriptor_for(value, getattr(value, _CAPABILITY_ATTRIBUTE))
        routines = (
            {descriptor.id: (descriptor, value)}
            if isinstance(descriptor, RoutineDescriptor)
            else {}
        )
        loaders = (
            {descriptor.id: (descriptor, value)}
            if isinstance(descriptor, LoaderDescriptor)
            else {}
        )
    else:
        raise TypeError(
            "erlab.extensions entry points must provide a decorated function or module"
        )
    if not routines and not loaders:
        raise TypeError("The environment entry point has no capabilities")
    return (
        tuple(item[0] for item in routines.values()),
        tuple(item[0] for item in loaders.values()),
        None,
        (),
    )


def _validate_extension_revision(
    catalog_store: _ExtensionCatalogStore,
    extension_id: str,
    *,
    revision_hash: str,
    expected_record_generation: int,
    manager_session_id: str,
    script_modules: dict[tuple[str, str], LoadedScript],
) -> _ExtensionCatalogModel:
    """Import one revision, then atomically record its validated descriptors."""
    catalog = catalog_store.read()
    record = catalog.extensions.get(extension_id)
    if record is None:
        raise KeyError(extension_id)
    if (
        record.current_revision != revision_hash
        or record.record_generation != expected_record_generation
    ):
        raise _ExtensionCatalogConflictError(
            f"Extension {extension_id!r} changed before validation"
        )
    try:
        loader_always_single: bool | None = None
        loader_dialog_methods: tuple[_EnvironmentLoaderMethod, ...] = ()
        if record.source_type == "script":
            loaded = _cached_script(
                script_modules,
                extension_id=extension_id,
                revision_hash=revision_hash,
                source_path=catalog_store.source_path(extension_id, revision_hash),
                module_name=_manager_module_name(
                    manager_session_id, extension_id, revision_hash
                ),
            )
            routines = tuple(item[0] for item in loaded.routines.values())
            loaders = tuple(item[0] for item in loaded.loaders.values())
        else:
            (
                routines,
                loaders,
                loader_always_single,
                loader_dialog_methods,
            ) = _environment_capabilities(
                catalog_store, record.revisions[revision_hash]
            )
    except BaseException:
        with contextlib.suppress(_ExtensionCatalogConflictError):
            catalog_store.record_validation_failure(
                extension_id,
                revision_hash=revision_hash,
                expected_record_generation=expected_record_generation,
                import_error=traceback.format_exc(),
            )
        raise
    return catalog_store.enable_validated_revision(
        extension_id,
        revision_hash=revision_hash,
        expected_record_generation=expected_record_generation,
        routines=routines,
        loaders=loaders,
        loader_always_single=loader_always_single,
        loader_dialog_methods=loader_dialog_methods,
    )


class _ExtensionRoutineWorker(QtCore.QRunnable):
    """Run one pinned revision without accessing Qt widgets."""

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
            "extension_revision": self.job.revision_hash,
            "capability_id": self.job.routine.id,
            "extension_source": (
                str(self.job.source_path)
                if self.job.source_path is not None
                else self.job.entry_point_value
            ),
            "parameters": self.job.parameters,
            "input_uid": self.job.input_uid,
            "input_snapshot": self.job.input_snapshot,
            "input": _array_log_fields(self.job.input_data),
        }
        try:
            record = self.catalog_store.read().extensions.get(self.job.extension_id)
            enabled = record is not None and not record.removed and record.enabled
            if not enabled:
                status = "discarded"
                logger.info(
                    "Discarding disabled queued extension routine",
                    extra={
                        **fields,
                        "extension_status": "disabled-before-start",
                    },
                )
                return
            self._started.set()
            self.signals.started.emit()
            logger.info("Importing extension revision", extra=fields)
            if self.job.source_type == "script":
                source_path = _require_script_source(self.job)
                module_name = _manager_module_name(
                    self.manager_session_id,
                    self.job.extension_id,
                    self.job.revision_hash,
                )
                loaded = _cached_script(
                    self.script_modules,
                    extension_id=self.job.extension_id,
                    revision_hash=self.job.revision_hash,
                    source_path=source_path,
                    module_name=module_name,
                )
                entry = _require_routine(loaded, self.job.routine.id)
            else:
                entry = _environment_routine(self.job)
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


class _ExtensionProgressDialog(QtWidgets.QDialog):
    """Non-modal queue view. Closing it does not cancel the active call."""

    remove_requested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("manager_extension_progress_dialog")
        self.setWindowTitle("Extension Jobs")
        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setObjectName("manager_extension_job_list")
        layout.addWidget(self.list_widget)
        remove_button = QtWidgets.QPushButton("Remove Queued Job", self)
        remove_button.setObjectName("manager_extension_remove_job_button")
        remove_button.clicked.connect(self._remove_selected)
        layout.addWidget(remove_button)

    def set_jobs(
        self,
        active: _ExtensionRoutineJob | None,
        queued: tuple[_ExtensionRoutineJob, ...],
    ) -> None:
        self.list_widget.clear()
        for job, active_job in (
            *((job, True) for job in (() if active is None else (active,))),
            *((job, False) for job in queued),
        ):
            item = QtWidgets.QListWidgetItem(
                f"{job.routine.name} — {'running' if active_job else 'queued'}"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, job.job_id)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, active_job)
            self.list_widget.addItem(item)

    @QtCore.Slot()
    def _remove_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item is None or bool(item.data(QtCore.Qt.ItemDataRole.UserRole + 1)):
            return
        self.remove_requested.emit(str(item.data(QtCore.Qt.ItemDataRole.UserRole)))


class _ExtensionExecutionController(QtCore.QObject):
    """Own one serialized extension queue for one manager lifetime.

    Running code keeps its pinned input and source revision until completion. Queued
    jobs recheck application enablement before they start. Shutdown stops admission,
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
        self._progress_dialog = _ExtensionProgressDialog(manager)
        self._remove_queued_slot = self.remove_queued
        self._finished_slot = self._finished
        self._started_slot = self._routine_started
        self._refresh_progress_slot = self._refresh_progress
        self._progress_dialog.remove_requested.connect(self._remove_queued_slot)
        self.queue_changed.connect(self._refresh_progress_slot)

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

    def show_progress(self) -> None:
        self._refresh_progress()
        self._progress_dialog.show()
        self._progress_dialog.raise_()

    def run_loader(
        self,
        call: _ExtensionLoaderCall,
        path: pathlib.Path,
        parameters: dict[str, typing.Any],
    ) -> (
        xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | list[xr.DataArray | xr.Dataset | xr.DataTree]
    ):
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
        """Validate one catalog revision on the manager extension thread."""
        catalog = self._catalog.store.read()
        record = catalog.extensions.get(extension_id)
        if record is None:
            raise KeyError(extension_id)
        if record.record_generation != expected_record_generation:
            raise _ExtensionCatalogConflictError(
                f"Extension {extension_id!r} changed before validation"
            )
        task = _ExtensionValidationWorker(
            extension_id,
            record.current_revision,
            expected_record_generation,
            manager_session_id=self._manager_session_id,
            catalog_store=self._catalog.store,
            script_modules=self._script_modules,
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
            revision_hash=None,
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
            revision_hash=operation.revision_hash,
            routine_id=operation.routine_id,
            parameters=operation.parameters,
            input_data=data,
            input_uid="provenance-replay",
            input_snapshot="pinned-revision",
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
        revision_hash: str | None,
        routine_id: str,
        parameters: Mapping[str, typing.Any],
        input_data: xr.DataArray,
        input_uid: str,
        input_snapshot: str,
    ) -> _ExtensionRoutineJob:
        """Pin catalog state and input identity before queue admission."""
        catalog = self._catalog.store.read()
        record = catalog.extensions.get(extension_id)
        if record is None or record.removed or not record.enabled:
            raise ExtensionExecutionError("The extension is not enabled")
        pinned_revision = (
            record.current_revision if revision_hash is None else revision_hash
        )
        revision = record.revisions.get(pinned_revision)
        if revision is None or not revision.approved:
            raise ExtensionExecutionError("The extension revision is not available")
        routine = next(
            (item for item in revision.routines if item.id == routine_id), None
        )
        if routine is None:
            raise ExtensionExecutionError(f"Routine {routine_id!r} is not available")
        return _ExtensionRoutineJob(
            job_id=uuid.uuid4().hex,
            extension_id=extension_id,
            extension_name=record.name,
            revision_hash=pinned_revision,
            routine=routine,
            source_path=(
                self._catalog.store.source_path(extension_id, pinned_revision)
                if record.source_type == "script"
                else None
            ),
            source_type=record.source_type,
            entry_point_group=revision.entry_point_group,
            entry_point_name=revision.entry_point_name,
            entry_point_value=revision.entry_point_value,
            parameters=dict(parameters),
            input_uid=input_uid,
            input_snapshot=input_snapshot,
            input_data=input_data,
            catalog_generation=catalog.generation,
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
                    "extension_revision": result.job.revision_hash,
                    "capability_id": result.job.routine.id,
                    "input_uid": result.job.input_uid,
                    "input_snapshot": result.job.input_snapshot,
                    "extension_status": "stale-input",
                },
            )
            return
        operation = ExtensionRoutineOperation(
            extension_id=result.job.extension_id,
            revision_hash=result.job.revision_hash,
            routine_id=result.job.routine.id,
            extension_name=result.job.extension_name,
            routine_name=result.job.routine.name,
            source_type=result.job.source_type,
            function_name=result.job.routine.function_name,
            source_path=(
                None if result.job.source_path is None else str(result.job.source_path)
            ),
            entry_point_group=result.job.entry_point_group,
            entry_point_name=result.job.entry_point_name,
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

    @QtCore.Slot()
    def _refresh_progress(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self._progress_dialog):
            return
        self._progress_dialog.set_jobs(self.active, self.queued)

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
        if erlab.interactive.utils.qt_is_valid(self._progress_dialog):
            with contextlib.suppress(TypeError, RuntimeError):
                self._progress_dialog.remove_requested.disconnect(
                    self._remove_queued_slot
                )
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
        with contextlib.suppress(TypeError, RuntimeError):
            self.queue_changed.disconnect(self._refresh_progress_slot)
        self._script_modules.clear()
        _remove_manager_modules(self._manager_session_id)
        self._shutdown_complete = True
