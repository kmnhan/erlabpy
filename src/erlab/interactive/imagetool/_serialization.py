"""Private persistence helpers for ImageTool datasets."""

from __future__ import annotations

import base64
import collections.abc
import contextlib
import json
import math
import numbers
import typing

import numpy as np
import xarray as xr

from erlab.interactive import _persistence_constants

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Mapping


# Coordinates with whitespace in their names use private variables plus a name map.
PRIVATE_COORDS_ATTR = "_erlab_imagetool_private_coords"

_STALE_BACKEND_ENCODING_KEYS = frozenset(
    (
        "chunksizes",
        "compression",
        "compression_opts",
        "contiguous",
        "fletcher32",
        "original_shape",
        "preferred_chunks",
        "shuffle",
        "source",
    )
)


def _tool_attr_writes_natively(value: typing.Any) -> bool:
    if isinstance(value, str):
        return "\x00" not in value
    if isinstance(value, np.generic) and value.dtype.kind in "Mm":
        return False
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "U":
            return False
        # Let HDF5 retain its existing handling of compound and special dtypes.
        # These dtypes must not pass through the typed codec's raw-byte encoding.
        if value.dtype.kind == "V" or value.dtype.metadata is not None:
            return True
        if value.dtype.kind == "O":
            return bool(value.size) and all(
                isinstance(item, str | bytes) and _tool_attr_writes_natively(item)
                for item in value.flat
            )
    if isinstance(value, list | tuple):
        if _structured_attr_sequence(value):
            return True
        return all(_tool_attr_writes_natively(item) for item in value) and (
            attr_value_writes_natively(value)
        )
    return attr_value_writes_natively(value)


def _structured_attr_sequence(value: typing.Any) -> bool:
    return isinstance(value, np.void) or (
        isinstance(value, list | tuple)
        and bool(value)
        and all(_structured_attr_sequence(item) for item in value)
    )


def _validate_tool_attr_value(value: typing.Any) -> None:
    """Reject NumPy metadata that the shared workspace codec cannot preserve."""
    if isinstance(value, np.ndarray | np.generic):
        if (
            value.dtype.kind == "V"
            or (value.dtype.hasobject and value.dtype.kind != "O")
            or value.dtype.metadata is not None
        ):
            raise TypeError(
                "structured or variable-width NumPy attributes and dtype metadata "
                "are unsupported"
            )
        if value.dtype.kind == "O":
            for item in np.asarray(value).flat:
                if isinstance(item, list | tuple | np.ndarray):
                    raise TypeError(
                        "sequence elements in object array attributes are unsupported"
                    )
                _validate_tool_attr_value(item)
    elif isinstance(value, collections.abc.Mapping):
        for key, item in value.items():
            _validate_tool_attr_value(key)
            _validate_tool_attr_value(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _validate_tool_attr_value(item)


def _encode_tool_attrs(
    attrs: Mapping[Hashable, typing.Any], *, root: bool
) -> tuple[dict[str, typing.Any], list[list[dict[str, typing.Any]]]]:
    native: dict[str, typing.Any] = {}
    encoded: list[list[dict[str, typing.Any]]] = []
    for key, value in attrs.items():
        if not isinstance(key, str) or not key:
            raise TypeError("Tool attribute names must be non-empty strings")
        try:
            writes_natively = not (
                root and key in _persistence_constants.TOOL_ATTR_TRANSPORT_KEYS
            ) and (_tool_attr_writes_natively(value))
        except RecursionError as exc:
            raise TypeError(
                f"Cannot serialize tool attribute {key!r}: unsupported cyclic sequence"
            ) from exc
        if writes_natively:
            native[key] = value
            continue
        if root and key.startswith(("tool_", "erlab_")):
            raise TypeError(f"Tool control attribute {key!r} must use native metadata")
        try:
            _validate_tool_attr_value(value)
            encoded.append([encode_attr_key(key), encode_attr_value(value)])
        except (TypeError, RecursionError) as exc:
            raise TypeError(f"Cannot serialize tool attribute {key!r}: {exc}") from exc
    return native, encoded


def prepare_tool_dataset(
    ds: xr.Dataset, *, strip_backend_encoding: bool = False
) -> xr.Dataset:
    """Prepare tool file metadata without copying or computing array values."""
    prepared = ds.copy(deep=False)
    prepared.attrs, root_attrs = _encode_tool_attrs(ds.attrs, root=True)
    variable_attrs: dict[str, list[list[dict[str, typing.Any]]]] = {}
    for name, variable in prepared.variables.items():
        variable.attrs, encoded = _encode_tool_attrs(variable.attrs, root=False)
        if encoded:
            if not isinstance(name, str):
                raise TypeError("Saved tool variable names must be strings")
            variable_attrs[name] = encoded
        if strip_backend_encoding:
            variable.encoding = {
                key: value
                for key, value in variable.encoding.items()
                if key not in _STALE_BACKEND_ENCODING_KEYS
            }
    if root_attrs or variable_attrs:
        prepared.attrs[_persistence_constants.TOOL_ATTRS_VERSION_ATTR] = (
            _persistence_constants.TOOL_ATTRS_VERSION
        )
        prepared.attrs[_persistence_constants.TOOL_ENCODED_ATTRS_ATTR] = json.dumps(
            {"dataset": root_attrs, "variables": variable_attrs},
            separators=(",", ":"),
        )
    return prepared


def _decode_tool_attrs(
    attrs: dict[Hashable, typing.Any], entries: object, *, root: bool
) -> None:
    if not isinstance(entries, list):
        raise TypeError("Encoded tool attributes must be a list")
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("Invalid encoded tool attribute entry")
        try:
            key = decode_attr_key(entry[0])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid encoded tool attribute name") from exc
        if not isinstance(key, str) or not key or encode_attr_key(key) != entry[0]:
            raise ValueError("Invalid encoded tool attribute name")
        if key in attrs or (root and key.startswith(("tool_", "erlab_"))):
            raise ValueError(f"Encoded tool attribute cannot replace metadata {key!r}")
        try:
            value = decode_attr_value(entry[1])
            _validate_tool_attr_value(value)
            encoded = encode_attr_value(value)
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError(
                f"Invalid encoded tool attribute value for {key!r}"
            ) from exc
        if encoded != entry[1]:
            raise ValueError(f"Invalid encoded tool attribute value for {key!r}")
        attrs[key] = value


def restore_tool_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Decode a marked tool file once, before restoring private coordinates."""
    if _persistence_constants.TOOL_ATTRS_VERSION_ATTR not in ds.attrs:
        return ds
    version = ds.attrs[_persistence_constants.TOOL_ATTRS_VERSION_ATTR]
    if (
        isinstance(version, bool)
        or not isinstance(version, int | np.integer)
        or version != _persistence_constants.TOOL_ATTRS_VERSION
    ):
        raise ValueError(f"Unsupported tool attribute encoding version: {version!r}")
    try:
        payload = json.loads(ds.attrs[_persistence_constants.TOOL_ENCODED_ATTRS_ATTR])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid encoded tool attributes: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"dataset", "variables"}:
        raise ValueError("Invalid tool attribute envelope")
    variables = payload["variables"]
    if not isinstance(variables, dict):
        raise TypeError("Encoded tool variable attributes must be a mapping")
    restored = ds.copy(deep=False)
    for key in _persistence_constants.TOOL_ATTR_TRANSPORT_KEYS:
        restored.attrs.pop(key)
    _decode_tool_attrs(restored.attrs, payload["dataset"], root=True)
    for name, entries in variables.items():
        if name not in restored.variables:
            raise ValueError(f"Encoded attributes refer to missing variable {name!r}")
        _decode_tool_attrs(restored[name].attrs, entries, root=False)
    return restored


def coord_name_needs_private_storage(name: Hashable) -> bool:
    return isinstance(name, str) and any(char.isspace() for char in name)


def _private_coord_variable_name(existing: set[Hashable], index: int) -> str:
    while True:
        name = f"__erlab_imagetool_coord_{index}"
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
    raw = _decode_attr(attrs.get(PRIVATE_COORDS_ATTR))
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
    data_name: Hashable = _persistence_constants.ITOOL_DATA_NAME,
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
    data_name: Hashable = _persistence_constants.ITOOL_DATA_NAME,
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
    data_attrs[PRIVATE_COORDS_ATTR] = json.dumps(records, separators=(",", ":"))
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
    data_name: Hashable = _persistence_constants.ITOOL_DATA_NAME,
) -> xr.Dataset:
    if data_name not in ds.data_vars:
        return ds

    restored = ds.copy(deep=False)
    data_attrs = dict(restored[data_name].attrs)
    records = private_coord_records_from_attrs(data_attrs)
    if records is None:
        records = _legacy_spaced_coord_records(restored, data_name)
    else:
        data_attrs.pop(PRIVATE_COORDS_ATTR, None)
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


def attr_value_writes_natively(value: typing.Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, bytes):
        return b"\x00" not in value and _bytes_are_utf8(value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "biufcSU"
    if isinstance(value, np.generic):
        return isinstance(value, (np.number, np.bool_))
    if isinstance(value, bool | int | float | complex):
        return True
    if isinstance(value, list | tuple):
        return _attr_sequence_writes_natively(value)
    return False


def _attr_sequence_writes_natively(
    value: list[typing.Any] | tuple[typing.Any, ...],
) -> bool:
    if not value:
        return True
    if all(isinstance(item, str) for item in value):
        return True
    if all(
        isinstance(item, bytes) and b"\x00" not in item and _bytes_are_utf8(item)
        for item in value
    ):
        return True
    return all(_attr_numeric_scalar_writes_natively(item) for item in value)


def _attr_numeric_scalar_writes_natively(value: typing.Any) -> bool:
    if isinstance(value, np.generic):
        return isinstance(value, (np.number, np.bool_))
    return isinstance(value, bool | int | float | complex)


def _bytes_are_utf8(value: bytes) -> bool:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def encode_attr_key(value: typing.Any) -> dict[str, typing.Any]:
    if isinstance(value, np.generic):
        if value.dtype.kind in "SUMm":
            return encode_attr_value(value)
        value = value.item()
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", **_encode_float(value)}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": _encode_float(value.real),
            "imag": _encode_float(value.imag),
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [encode_attr_key(item) for item in value],
        }
    raise TypeError(f"unsupported attr key type {type(value).__name__!r}")


def decode_attr_key(value: typing.Any) -> typing.Hashable:
    decoded = decode_attr_value(value)
    if not isinstance(decoded, collections.abc.Hashable):
        raise TypeError(f"decoded attr key is not hashable: {type(decoded).__name__!r}")
    return decoded


def encode_attr_value(value: typing.Any) -> dict[str, typing.Any]:
    if isinstance(value, np.ndarray):
        return _encode_array(value, kind="ndarray")
    if isinstance(value, np.str_ | np.bytes_):
        array = np.asarray(value)
        if value.dtype.itemsize == 0:
            # np.asarray expands an empty string scalar to one NUL code unit.
            array = np.ndarray((), dtype=value.dtype)
        # Legacy numpy_scalar records do not distinguish empty strings from NULs.
        return _encode_array(array, kind="numpy_string")
    if isinstance(value, np.generic):
        return _encode_array(np.asarray(value), kind="numpy_scalar")
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", **_encode_float(value)}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": _encode_float(value.real),
            "imag": _encode_float(value.imag),
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [encode_attr_value(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [encode_attr_value(item) for item in value],
        }
    if isinstance(value, collections.abc.Mapping):
        return {
            "kind": "dict",
            "items": [
                [encode_attr_key(key), encode_attr_value(item)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, numbers.Number):
        raise TypeError(f"unsupported numeric attr type {type(value).__name__!r}")
    raise TypeError(f"unsupported attr value type {type(value).__name__!r}")


def decode_attr_value(value: typing.Any) -> typing.Any:
    if not isinstance(value, collections.abc.Mapping):
        raise TypeError("encoded workspace attr value must be a mapping")
    kind = value.get("kind")
    match kind:
        case "none":
            return None
        case "bool":
            return bool(value["value"])
        case "int":
            return int(value["value"])
        case "float":
            return _decode_float(value)
        case "complex":
            return complex(
                _decode_float(value["real"]),
                _decode_float(value["imag"]),
            )
        case "str":
            return str(value["value"])
        case "bytes":
            return base64.b64decode(str(value["value"]).encode("ascii"))
        case "list":
            return [decode_attr_value(item) for item in value["items"]]
        case "tuple":
            return tuple(decode_attr_value(item) for item in value["items"])
        case "dict":
            return {
                decode_attr_key(key): decode_attr_value(item)
                for key, item in value["items"]
            }
        case "ndarray":
            return _decode_array(value)
        case "numpy_scalar" | "numpy_string":
            array = _decode_array(value)
            if array.ndim != 0:
                raise ValueError("encoded NumPy scalar must have an empty shape")
            if kind == "numpy_scalar":
                # Retain the decoding semantics of existing workspace records.
                return array[()]
            # Indexing a string array strips trailing NULs from the scalar.
            if array.dtype.kind == "S":
                return np.bytes_(array.tobytes())
            if array.dtype.kind == "U":
                data = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
                return np.str_(data.decode("utf-32-le", errors="surrogatepass"))
            raise TypeError("encoded NumPy string must have a string dtype")
        case _:
            raise TypeError(f"unknown workspace attr value kind {kind!r}")


def _encode_float(value: float) -> dict[str, typing.Any]:
    if math.isnan(value):
        return {"special": "nan"}
    if math.isinf(value):
        return {"special": "inf" if value > 0 else "-inf"}
    return {"value": value}


def _decode_float(value: Mapping[str, typing.Any]) -> float:
    special = value.get("special")
    if special == "nan":
        # Distinct NaN mapping keys must not share the same object identity.
        return float("nan")
    if special == "inf":
        return math.inf
    if special == "-inf":
        return -math.inf
    return float(value["value"])


def _encode_array(value, *, kind: str) -> dict[str, typing.Any]:
    array = np.asarray(value)
    payload: dict[str, typing.Any] = {
        "kind": kind,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    if array.dtype.kind == "O":
        payload["items"] = encode_attr_value(array.tolist())
        return payload
    contiguous = np.ascontiguousarray(array)
    payload["data"] = base64.b64encode(contiguous.tobytes()).decode("ascii")
    return payload


def _decode_array(value: Mapping[str, typing.Any]):
    dtype = np.dtype(typing.cast("str", value["dtype"]))
    shape = tuple(int(size) for size in typing.cast("list[typing.Any]", value["shape"]))
    if "items" in value:
        items = decode_attr_value(value["items"])
        return np.asarray(items, dtype=object).reshape(shape)
    data = base64.b64decode(str(value["data"]).encode("ascii"))
    if dtype.kind in "SU" and dtype.itemsize == 0:
        if data:
            raise ValueError("encoded zero-width string array must have no data")
        return np.ndarray(shape, dtype=dtype)
    return np.frombuffer(data, dtype=dtype).copy().reshape(shape)
