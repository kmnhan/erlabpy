"""Workspace schema, manifest, and metadata serialization.

The root manifest is authoritative for ordering and manager-level state. HDF5 group
names identify payload locations; they do not define the complete workspace schema.
"""

from __future__ import annotations

import base64
import collections.abc
import contextlib
import json
import logging
import math
import numbers
import pathlib
import typing

import numpy as np
import pydantic

import erlab

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator, Mapping, MutableMapping

    import xarray as xr

logger = logging.getLogger(__name__)

_WORKSPACE_SCHEMA_VERSION = 4
_WORKSPACE_LEGACY_SCHEMA_VERSION = 3
_WORKSPACE_MANIFEST_ATTR = "imagetool_workspace_manifest"
_WORKSPACE_TRANSACTION_PROTOCOL = "recoverable-delta-v1"
_WORKSPACE_PENDING_GROUP_PREFIX = "__itws_pending_"
_WORKSPACE_BACKUP_GROUP_PREFIX = "__itws_backup_"
_WORKSPACE_TRANSACTION_GROUP_PREFIX = "__itws_txn_"
_WORKSPACE_ENCODED_ATTRS_ATTR = "_erlab_workspace_encoded_attrs"
_WORKSPACE_ENCODED_ATTRS_VERSION = 1
_WORKSPACE_REPLAY_SOURCE_BLOB_NAME = "<manager-replay-source-data>"


class WorkspaceLoaderState(pydantic.BaseModel):
    recent_directory: str | None = None
    # QFileDialog name filter selected in manager file-open flows.
    recent_name_filter: str | None = None
    # Manager file-open options are keyed by name filter, not loader name.
    manager_loader_kwargs_by_filter: dict[str, dict[str, typing.Any]] = pydantic.Field(
        default_factory=dict
    )
    # Loader extensions are saved separately because extend_loader applies them later.
    manager_loader_extensions_by_filter: dict[str, dict[str, typing.Any]] = (
        pydantic.Field(default_factory=dict)
    )
    # Data Explorer tabs select loaders by name, so their options use loader names.
    explorer_loader_kwargs_by_name: dict[str, dict[str, typing.Any]] = pydantic.Field(
        default_factory=dict
    )
    explorer_loader_extensions_by_name: dict[str, dict[str, typing.Any]] = (
        pydantic.Field(default_factory=dict)
    )

    model_config = pydantic.ConfigDict(extra="ignore")


class StandaloneAppsState(pydantic.BaseModel):
    schema_version: int = 1
    # Keys are manager standalone app ids such as "explorer" and "ptable".
    apps: dict[str, dict[str, typing.Any]] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(extra="ignore")


class WorkspaceOptionOverridesState(pydantic.BaseModel):
    schema_version: int = 1
    overrides: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(extra="ignore")


def _workspace_manifest_from_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> dict[str, typing.Any]:
    raw_manifest = attrs.get(_WORKSPACE_MANIFEST_ATTR)
    if isinstance(raw_manifest, bytes):
        raw_manifest = raw_manifest.decode()
    if isinstance(raw_manifest, str):
        with contextlib.suppress(json.JSONDecodeError):
            manifest = json.loads(raw_manifest)
            if isinstance(manifest, dict):
                return manifest
    return {}


def _iter_workspace_manifest_node_entries(
    manifest: Mapping[str, typing.Any] | None,
) -> Iterator[Mapping[str, typing.Any]]:
    if manifest is None:
        return
    nodes = manifest.get("nodes", ())
    if not isinstance(nodes, list):
        return
    for entry in nodes:
        if isinstance(entry, collections.abc.Mapping):
            yield entry


def _workspace_manifest_payload_entries(
    manifest: Mapping[str, typing.Any] | None,
) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for entry in _iter_workspace_manifest_node_entries(manifest):
        uid = entry.get("uid")
        kind = entry.get("kind")
        path = entry.get("path")
        if (
            isinstance(uid, str)
            and isinstance(kind, str)
            and kind in {"imagetool", "tool"}
            and isinstance(path, str)
        ):
            entries.append((uid, kind, f"{path}/{kind}"))
    return entries


def _decode_workspace_attr_text(value: object) -> str | None:
    if isinstance(value, bytes):
        with contextlib.suppress(UnicodeDecodeError):
            value = value.decode()
    if isinstance(value, str) and value:
        return value
    return None


def _workspace_delta_save_count_from_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> int:
    value = _workspace_manifest_from_attrs(attrs).get("delta_save_count", 0)
    with contextlib.suppress(TypeError, ValueError):
        return max(0, int(value))
    return 0


def _workspace_manifest_nonnegative_int(
    manifest: Mapping[str, typing.Any], key: str
) -> int:
    value = manifest.get(key, 0)
    with contextlib.suppress(TypeError, ValueError):
        return max(0, int(value))
    return 0


def _workspace_manifest_repack_estimate(
    manifest: Mapping[str, typing.Any] | None,
    *,
    delta_save_count: int,
) -> tuple[int, int, bool]:
    if delta_save_count <= 0:
        return 0, 0, True
    if manifest is None:
        return 0, 0, False
    has_estimate = (
        "estimated_obsolete_bytes" in manifest and "replacement_delta_count" in manifest
    )
    return (
        _workspace_manifest_nonnegative_int(manifest, "estimated_obsolete_bytes"),
        _workspace_manifest_nonnegative_int(manifest, "replacement_delta_count"),
        has_estimate,
    )


def _workspace_file_metadata_from_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> tuple[int, int, dict[str, typing.Any] | None]:
    schema_version = int(attrs.get("imagetool_workspace_schema_version", 1))
    manifest = None
    if schema_version >= _WORKSPACE_SCHEMA_VERSION:
        manifest = _workspace_manifest_from_attrs(attrs) or None
    return schema_version, _workspace_delta_save_count_from_attrs(attrs), manifest


def _compacted_workspace_root_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> dict[str, typing.Any]:
    schema_version, _delta_save_count, manifest = _workspace_file_metadata_from_attrs(
        attrs
    )
    if schema_version != _WORKSPACE_SCHEMA_VERSION or manifest is None:
        raise ValueError(
            "File-level workspace repack requires current workspace schema"
        )
    compacted_manifest = dict(manifest)
    compacted_manifest["schema_version"] = _WORKSPACE_SCHEMA_VERSION
    compacted_manifest["erlab_version"] = erlab.__version__
    compacted_manifest.pop("delta_save_count", None)
    compacted_manifest.pop("transaction_protocol", None)
    compacted_manifest.pop("estimated_obsolete_bytes", None)
    compacted_manifest.pop("replacement_delta_count", None)

    root_attrs = {str(key): value for key, value in attrs.items()}
    root_attrs["imagetool_workspace_schema_version"] = _WORKSPACE_SCHEMA_VERSION
    root_attrs[_WORKSPACE_MANIFEST_ATTR] = json.dumps(compacted_manifest)
    root_attrs["erlab_version"] = erlab.__version__
    return root_attrs


def _workspace_root_attrs_with_repack_estimate(
    attrs: Mapping[typing.Any, typing.Any],
    *,
    estimated_obsolete_bytes: int,
    replacement_delta_count: int,
    repack_estimate_known: bool = True,
) -> dict[str, typing.Any]:
    schema_version, delta_save_count, manifest = _workspace_file_metadata_from_attrs(
        attrs
    )
    if schema_version != _WORKSPACE_SCHEMA_VERSION or manifest is None:
        return {str(key): value for key, value in attrs.items()}
    updated_manifest = dict(manifest)
    if delta_save_count > 0 and repack_estimate_known:
        updated_manifest["estimated_obsolete_bytes"] = max(
            0, int(estimated_obsolete_bytes)
        )
        updated_manifest["replacement_delta_count"] = max(
            0, int(replacement_delta_count)
        )
    else:
        updated_manifest.pop("estimated_obsolete_bytes", None)
        updated_manifest.pop("replacement_delta_count", None)

    root_attrs = {str(key): value for key, value in attrs.items()}
    root_attrs[_WORKSPACE_MANIFEST_ATTR] = json.dumps(updated_manifest)
    return root_attrs


def _current_workspace_schema_version() -> int:
    return _WORKSPACE_SCHEMA_VERSION


def _workspace_path_is_itws(
    fname: str | os.PathLike[str],
) -> bool:
    return pathlib.Path(fname).suffix.lower() == ".itws"


def _require_itws_workspace_path(fname: str | os.PathLike[str], message: str) -> None:
    if not _workspace_path_is_itws(fname):
        raise ValueError(message)


def _set_legacy_workspace_schema(
    attrs: MutableMapping[typing.Hashable, typing.Any],
) -> None:
    attrs["imagetool_workspace_schema_version"] = _WORKSPACE_LEGACY_SCHEMA_VERSION


def _workspace_schema_requires_conversion(schema_version: int) -> bool:
    return schema_version < _WORKSPACE_LEGACY_SCHEMA_VERSION


def _workspace_schema_requires_full_save(schema_version: int) -> bool:
    return (
        _WORKSPACE_LEGACY_SCHEMA_VERSION <= schema_version < _WORKSPACE_SCHEMA_VERSION
    )


def _workspace_root_attrs_payload(
    *,
    root_order: Iterable[typing.Any],
    nodes: Iterable[Mapping[str, typing.Any]],
    delta_save_count: int,
    erlab_version: str,
    workspace_link_id: str | None = None,
    manager_layout: Mapping[str, typing.Any] | None = None,
    loader_state: Mapping[str, typing.Any] | None = None,
    standalone_apps: Mapping[str, typing.Any] | None = None,
    option_overrides: Mapping[str, typing.Any] | None = None,
    acquisition_context: Mapping[str, typing.Any] | None = None,
    estimated_obsolete_bytes: int = 0,
    replacement_delta_count: int = 0,
    repack_estimate_known: bool = True,
) -> dict[str, typing.Any]:
    manifest: dict[str, typing.Any] = {
        "schema_version": _WORKSPACE_SCHEMA_VERSION,
        "erlab_version": erlab_version,
        # HDF5 group order is not the manager's authoritative top-level order.
        "root_order": list(root_order),
        # Stable manager UIDs map to payload groups and link/data metadata here.
        "nodes": list(nodes),
    }
    if workspace_link_id is not None:
        # Scopes watched-variable/source links to this workspace document.
        manifest["workspace_link_id"] = workspace_link_id
    if manager_layout is not None:
        # Stored at root so layout-only saves can avoid rewriting tool payloads.
        manifest["manager_layout"] = dict(manager_layout)
    if loader_state is not None:
        # Manager loader choices are independent of standalone Data Explorer state.
        manifest["loader_state"] = dict(loader_state)
    if standalone_apps is not None:
        # Restored on full workspace open; imports intentionally ignore app windows.
        manifest["standalone_apps"] = dict(standalone_apps)
    if option_overrides is not None:
        # Sparse interactive settings overrides portable with this workspace.
        manifest["interactive_option_overrides"] = dict(option_overrides)
    if acquisition_context is not None:
        # Context defaults are workspace-scoped and independent of loader choices.
        manifest["acquisition_context"] = dict(acquisition_context)
    if delta_save_count > 0:
        # Marks files written through the recoverable in-place delta-save protocol.
        manifest["transaction_protocol"] = _WORKSPACE_TRANSACTION_PROTOCOL
        manifest["delta_save_count"] = int(delta_save_count)
        if repack_estimate_known:
            manifest["estimated_obsolete_bytes"] = max(0, int(estimated_obsolete_bytes))
            manifest["replacement_delta_count"] = max(0, int(replacement_delta_count))
    return {
        "imagetool_workspace_schema_version": _WORKSPACE_SCHEMA_VERSION,
        _WORKSPACE_MANIFEST_ATTR: json.dumps(manifest),
        "erlab_version": erlab_version,
    }


def _is_workspace_internal_group_name(name: typing.Any) -> bool:
    return str(name).startswith(
        (
            _WORKSPACE_PENDING_GROUP_PREFIX,
            _WORKSPACE_BACKUP_GROUP_PREFIX,
            _WORKSPACE_TRANSACTION_GROUP_PREFIX,
        )
    )


def _workspace_root_keys(
    tree: typing.Any, manifest: Mapping[str, typing.Any] | None
) -> list[str]:
    root_keys: list[str] = []
    if manifest is not None:
        raw_root_order = manifest.get("root_order", ())
        if isinstance(raw_root_order, list):
            root_keys.extend(
                str(item)
                for item in raw_root_order
                if str(item) not in root_keys
                and str(item) != "figures"
                and not _is_workspace_internal_group_name(item)
            )
    root_keys.extend(
        str(key)
        for key in tree
        if str(key) not in root_keys
        and str(key) != "figures"
        and not _is_workspace_internal_group_name(key)
    )
    return root_keys


def _workspace_file_is_workspace(h5_file) -> bool:
    if "imagetool_workspace_schema_version" in h5_file.attrs:
        return True
    return h5_file.attrs.get("is_itool_workspace", 0) == 1


def _workspace_serializable_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> dict[str, typing.Any]:
    serializable: dict[str, typing.Any] = {}
    encoded_entries: list[list[typing.Any]] = []
    for key, value in attrs.items():
        if not isinstance(key, str) or not key:
            continue
        if key == _WORKSPACE_ENCODED_ATTRS_ATTR:
            existing_entries = _workspace_encoded_attr_entries(value)
            if existing_entries is not None:
                encoded_entries.extend(existing_entries)
                continue
        if (
            key != _WORKSPACE_ENCODED_ATTRS_ATTR
            and _workspace_attr_value_writes_natively(value)
        ):
            serializable[key] = value
            continue
        try:
            encoded_entries.append(
                [_workspace_encode_attr_key(key), _workspace_encode_attr_value(value)]
            )
        except TypeError:
            logger.warning(
                "Dropping workspace attribute %r with unsupported value type %s",
                key,
                type(value).__name__,
            )
    if encoded_entries:
        serializable[_WORKSPACE_ENCODED_ATTRS_ATTR] = json.dumps(
            {
                "version": _WORKSPACE_ENCODED_ATTRS_VERSION,
                "attrs": encoded_entries,
            },
            separators=(",", ":"),
        )
    return serializable


def _workspace_attr_value_writes_natively(value: typing.Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, bytes):
        return b"\x00" not in value and _workspace_bytes_are_utf8(value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "biufcSU"
    if isinstance(value, np.generic):
        return isinstance(value, (np.number, np.bool_))
    if isinstance(value, bool | int | float | complex):
        return True
    if isinstance(value, list | tuple):
        return _workspace_attr_sequence_writes_natively(value)
    return False


def _workspace_attr_sequence_writes_natively(
    value: list[typing.Any] | tuple[typing.Any, ...],
) -> bool:
    if not value:
        return True
    if all(isinstance(item, str) for item in value):
        return True
    if all(
        isinstance(item, bytes)
        and b"\x00" not in item
        and _workspace_bytes_are_utf8(item)
        for item in value
    ):
        return True
    return all(_workspace_attr_numeric_scalar_writes_natively(item) for item in value)


def _workspace_attr_numeric_scalar_writes_natively(value: typing.Any) -> bool:
    if isinstance(value, np.generic):
        return isinstance(value, (np.number, np.bool_))
    return isinstance(value, bool | int | float | complex)


def _workspace_bytes_are_utf8(value: bytes) -> bool:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _workspace_encode_attr_key(value: typing.Any) -> dict[str, typing.Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", **_workspace_encode_float(value)}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": _workspace_encode_float(value.real),
            "imag": _workspace_encode_float(value.imag),
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
            "items": [_workspace_encode_attr_key(item) for item in value],
        }
    raise TypeError(f"unsupported attr key type {type(value).__name__!r}")


def _workspace_decode_attr_key(value: typing.Any) -> typing.Hashable:
    decoded = _workspace_decode_attr_value(value)
    if not isinstance(decoded, collections.abc.Hashable):
        raise TypeError(f"decoded attr key is not hashable: {type(decoded).__name__!r}")
    return decoded


def _workspace_encode_attr_value(value: typing.Any) -> dict[str, typing.Any]:
    if isinstance(value, np.ndarray):
        return _workspace_encode_array(value, kind="ndarray")
    if isinstance(value, np.generic):
        return _workspace_encode_array(np.asarray(value), kind="numpy_scalar")
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", **_workspace_encode_float(value)}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": _workspace_encode_float(value.real),
            "imag": _workspace_encode_float(value.imag),
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
            "items": [_workspace_encode_attr_value(item) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_workspace_encode_attr_value(item) for item in value],
        }
    if isinstance(value, collections.abc.Mapping):
        return {
            "kind": "dict",
            "items": [
                [_workspace_encode_attr_key(key), _workspace_encode_attr_value(item)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, numbers.Number):
        raise TypeError(f"unsupported numeric attr type {type(value).__name__!r}")
    raise TypeError(f"unsupported attr value type {type(value).__name__!r}")


def _workspace_decode_attr_value(value: typing.Any) -> typing.Any:
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
            return _workspace_decode_float(value)
        case "complex":
            return complex(
                _workspace_decode_float(value["real"]),
                _workspace_decode_float(value["imag"]),
            )
        case "str":
            return str(value["value"])
        case "bytes":
            return base64.b64decode(str(value["value"]).encode("ascii"))
        case "list":
            return [_workspace_decode_attr_value(item) for item in value["items"]]
        case "tuple":
            return tuple(_workspace_decode_attr_value(item) for item in value["items"])
        case "dict":
            return {
                _workspace_decode_attr_key(key): _workspace_decode_attr_value(item)
                for key, item in value["items"]
            }
        case "ndarray":
            return _workspace_decode_array(value)
        case "numpy_scalar":
            return _workspace_decode_array(value)[()]
        case _:
            raise TypeError(f"unknown workspace attr value kind {kind!r}")


def _workspace_encode_float(value: float) -> dict[str, typing.Any]:
    if math.isnan(value):
        return {"special": "nan"}
    if math.isinf(value):
        return {"special": "inf" if value > 0 else "-inf"}
    return {"value": value}


def _workspace_decode_float(value: Mapping[str, typing.Any]) -> float:
    special = value.get("special")
    if special == "nan":
        return math.nan
    if special == "inf":
        return math.inf
    if special == "-inf":
        return -math.inf
    return float(value["value"])


def _workspace_encode_array(value, *, kind: str) -> dict[str, typing.Any]:
    array = np.asarray(value)
    payload: dict[str, typing.Any] = {
        "kind": kind,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    if array.dtype.kind == "O":
        payload["items"] = _workspace_encode_attr_value(array.tolist())
        return payload
    contiguous = np.ascontiguousarray(array)
    payload["data"] = base64.b64encode(contiguous.tobytes()).decode("ascii")
    return payload


def _workspace_decode_array(value: Mapping[str, typing.Any]):
    dtype = np.dtype(typing.cast("str", value["dtype"]))
    shape = tuple(int(size) for size in typing.cast("list[typing.Any]", value["shape"]))
    if "items" in value:
        items = _workspace_decode_attr_value(value["items"])
        return np.asarray(items, dtype=object).reshape(shape)
    data = base64.b64decode(str(value["data"]).encode("ascii"))
    return np.frombuffer(data, dtype=dtype).copy().reshape(shape)


def _workspace_encoded_attr_entries(value: typing.Any) -> list[list[typing.Any]] | None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(value, str):
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != _WORKSPACE_ENCODED_ATTRS_VERSION
        or not isinstance(payload.get("attrs"), list)
    ):
        return None
    entries = payload["attrs"]
    if not all(
        isinstance(entry, list) and len(entry) == 2
        for entry in typing.cast("list[typing.Any]", entries)
    ):
        return None
    return typing.cast("list[list[typing.Any]]", entries)


def _restore_workspace_serialized_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> dict[typing.Any, typing.Any]:
    encoded_entries = _workspace_encoded_attr_entries(
        attrs.get(_WORKSPACE_ENCODED_ATTRS_ATTR)
    )
    if encoded_entries is None:
        return dict(attrs)
    restored = {
        key: value
        for key, value in attrs.items()
        if key != _WORKSPACE_ENCODED_ATTRS_ATTR
    }
    for key_payload, value_payload in encoded_entries:
        try:
            key = _workspace_decode_attr_key(key_payload)
            value = _workspace_decode_attr_value(value_payload)
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Ignoring invalid encoded workspace attribute", exc_info=True
            )
            continue
        if isinstance(key, str) and key:
            restored[key] = value
    return restored


def _sanitize_workspace_attr_names(ds: xr.Dataset) -> xr.Dataset:
    sanitized = ds.copy(deep=False)
    sanitized.attrs = _workspace_serializable_attrs(sanitized.attrs)
    for variable in sanitized.variables.values():
        variable.attrs = _workspace_serializable_attrs(variable.attrs)
    return sanitized


def _restore_workspace_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    restored = ds.copy(deep=False)
    restored.attrs = _restore_workspace_serialized_attrs(restored.attrs)
    for variable in restored.variables.values():
        variable.attrs = _restore_workspace_serialized_attrs(variable.attrs)
    return restored
