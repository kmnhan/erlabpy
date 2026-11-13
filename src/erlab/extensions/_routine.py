"""Execute a registered analysis routine in a Manager-owned Python process."""

from __future__ import annotations

import copy
import functools
import typing

import cloudpickle
import numpy as np
import xarray as xr

from erlab.extensions import ExtensionExecutionError, LoadedScript, RoutineDescriptor
from erlab.extensions._api import _coerce_call_parameters, _load_script_bytes

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable


class _ExtensionSourceLoadFailure(ExtensionExecutionError):
    """Identify a failure that occurred before extension function invocation."""


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


def _require_routine(
    loaded: LoadedScript, routine_id: str
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


@functools.cache
def _load_routine_script(
    source: bytes, path: pathlib.Path, module_name: str, source_hash: str
) -> LoadedScript:
    loaded = _load_script_bytes(
        source, path, module_name=module_name, expected_source_hash=source_hash
    )
    # Lazy results can contain functions from this process-local module.
    cloudpickle.register_pickle_by_value(loaded.erlab.module)
    return loaded


def run_routine(
    source: bytes,
    path: pathlib.Path,
    module_name: str,
    source_hash: str,
    routine_id: str,
    parameters: dict[str, typing.Any],
    settings: dict[str, bool | str],
    data: xr.DataArray,
) -> xr.DataArray:
    with xr.set_options(**settings):
        try:
            loaded = _load_routine_script(source, path, module_name, source_hash)
            _, function = _require_routine(loaded, routine_id)
        except BaseException as error:
            raise _ExtensionSourceLoadFailure(
                f"Extension routine {routine_id!r} could not be imported"
            ) from error
        kwargs = _coerce_call_parameters(function, parameters)
        value = _require_dataarray(function(_readonly_array(data), **kwargs))
        return _detached_routine_output(value, data)
