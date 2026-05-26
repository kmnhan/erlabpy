"""Private persistence helpers for ImageTool datasets."""

from __future__ import annotations

import contextlib
import json
import typing

import xarray as xr

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

ITOOL_DATA_NAME: str = "<erlab-itool-data>"
SAVED_TOOL_DATA_NAME: str = "<saved-tool-data>"

# NetCDF stores non-dimension coordinate names in a whitespace-delimited
# ``coordinates`` attribute.  A coord named "Fake Motor" cannot round-trip there,
# so ImageTool stores those associated coords as private variables plus this map.
_PRIVATE_COORDS_ATTR = "_erlab_imagetool_private_coords"
_PRIVATE_COORD_VAR_PREFIX = "__erlab_imagetool_coord_"


def coord_name_needs_private_storage(name: Hashable) -> bool:
    return isinstance(name, str) and any(char.isspace() for char in name)


def _private_coord_variable_name(existing: set[Hashable], index: int) -> str:
    while True:
        name = f"{_PRIVATE_COORD_VAR_PREFIX}{index}"
        if name not in existing:
            return name
        index += 1


def _decode_attr(value: object) -> object:
    if isinstance(value, bytes):
        with contextlib.suppress(UnicodeDecodeError):
            return value.decode()
    return value


def private_coord_records_from_attrs(
    attrs: Mapping[Hashable, object],
) -> tuple[dict[str, typing.Any], ...] | None:
    raw = _decode_attr(attrs.get(_PRIVATE_COORDS_ATTR))
    if raw is None:
        return None
    if not isinstance(raw, str):
        return ()
    with contextlib.suppress(json.JSONDecodeError):
        payload = json.loads(raw)
        if isinstance(payload, list):
            records: list[dict[str, typing.Any]] = []
            for item in payload:
                if not isinstance(item, dict):
                    return ()
                coord_name = item.get("coord_name")
                variable_name = item.get("variable_name")
                dims = item.get("dims")
                if (
                    not isinstance(coord_name, str)
                    or not isinstance(variable_name, str)
                    or not isinstance(dims, list)
                    or not all(isinstance(dim, str) for dim in dims)
                ):
                    return ()
                records.append(
                    {
                        "coord_name": coord_name,
                        "variable_name": variable_name,
                        "dims": tuple(dims),
                    }
                )
            return tuple(records)
    return ()


def private_coord_variable_names(
    ds: xr.Dataset,
    data_name: Hashable = ITOOL_DATA_NAME,
) -> tuple[str, ...]:
    if data_name not in ds.data_vars:
        return ()
    records = private_coord_records_from_attrs(ds[data_name].attrs)
    if not records:
        return ()
    return tuple(str(record["variable_name"]) for record in records)


def _coord_dims_fit_data(coord: xr.DataArray, data_array: xr.DataArray) -> bool:
    data_dims = set(data_array.dims)
    return all(dim in data_dims for dim in coord.dims)


def encode_private_coords(
    ds: xr.Dataset,
    data_name: Hashable = ITOOL_DATA_NAME,
) -> xr.Dataset:
    if data_name not in ds.data_vars:
        return ds

    data_array = ds[data_name]
    coord_names = [
        name
        for name, coord in ds.coords.items()
        if isinstance(name, str)
        and name not in data_array.dims
        and coord_name_needs_private_storage(name)
        and _coord_dims_fit_data(coord, data_array)
    ]
    if not coord_names:
        return ds

    encoded = ds.copy(deep=False)
    records: list[dict[str, typing.Any]] = []
    existing: set[Hashable] = set(encoded.variables)
    for index, coord_name in enumerate(coord_names):
        coord = encoded.coords[coord_name]
        variable_name = _private_coord_variable_name(existing, index)
        existing.add(variable_name)
        records.append(
            {
                "coord_name": coord_name,
                "variable_name": variable_name,
                "dims": list(coord.dims),
            }
        )

        # Keep the public coordinate name out of NetCDF's whitespace-delimited
        # coordinate list; the private variable name is safe and restored on load.
        private_coord = xr.DataArray(
            coord.data,
            dims=coord.dims,
            attrs=dict(coord.attrs),
            name=variable_name,
        )
        private_coord.encoding.update(coord.encoding)
        encoded = encoded.drop_vars(coord_name)
        encoded[variable_name] = private_coord

    data_attrs = dict(encoded[data_name].attrs)
    data_attrs[_PRIVATE_COORDS_ATTR] = json.dumps(records, separators=(",", ":"))
    encoded[data_name].attrs = data_attrs
    return encoded


def _legacy_spaced_coord_records(
    ds: xr.Dataset,
    data_name: Hashable,
) -> tuple[dict[str, typing.Any], ...]:
    data_dims = set(ds[data_name].dims)
    records: list[dict[str, typing.Any]] = []
    for variable_name, variable in ds.data_vars.items():
        if variable_name == data_name:
            continue
        if not coord_name_needs_private_storage(variable_name):
            continue
        if not all(dim in data_dims for dim in variable.dims):
            continue
        records.append(
            {
                "coord_name": variable_name,
                "variable_name": variable_name,
                "dims": tuple(variable.dims),
            }
        )
    return tuple(records)


def restore_private_coords(
    ds: xr.Dataset,
    data_name: Hashable = ITOOL_DATA_NAME,
) -> xr.Dataset:
    if data_name not in ds.data_vars:
        return ds

    restored = ds.copy(deep=False)
    data_attrs = dict(restored[data_name].attrs)
    records = private_coord_records_from_attrs(data_attrs)
    if records is None:
        records = _legacy_spaced_coord_records(restored, data_name)
    else:
        data_attrs.pop(_PRIVATE_COORDS_ATTR, None)
        restored[data_name].attrs = data_attrs

    drop_names: list[Hashable] = []
    data_dims = set(restored[data_name].dims)
    for record in records:
        coord_name = record["coord_name"]
        variable_name = record["variable_name"]
        dims = tuple(record["dims"])
        if variable_name not in restored.data_vars:
            continue
        variable = restored[variable_name]
        if len(dims) != variable.ndim or not all(dim in data_dims for dim in dims):
            continue

        if variable_name == coord_name:
            restored = restored.set_coords(variable_name)
        else:
            coord = xr.DataArray(
                variable.data,
                dims=dims,
                attrs=dict(variable.attrs),
                name=coord_name,
            )
            coord.encoding.update(variable.encoding)
            restored = restored.assign_coords({coord_name: coord})
            drop_names.append(variable_name)

    if drop_names:
        restored = restored.drop_vars(drop_names)
    return restored
