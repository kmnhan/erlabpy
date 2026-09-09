"""Workspace schema, manifest, and metadata serialization.

The root manifest is authoritative for ordering and manager-level state. HDF5 group
names identify payload locations; they do not define the complete workspace schema.
"""

from __future__ import annotations

import collections.abc
import contextlib
import copy
import json
import logging
import pathlib
import typing

import pydantic

from erlab.extensions._models import _script_name_key, _validate_source_hash
from erlab.interactive import _persistence_constants
from erlab.interactive.imagetool import _serialization

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator, Mapping, MutableMapping

    import xarray as xr

logger = logging.getLogger(__name__)


class _WorkspaceEmbeddedScriptEntry(pydantic.BaseModel):
    """One verified script source object owned by a workspace document."""

    script_name: str
    source_hash: str
    object_id: str

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("script_name")
    @classmethod
    def _valid_script_name(cls, value: str) -> str:
        _script_name_key(value)
        return value

    @pydantic.field_validator("source_hash")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return _validate_source_hash(value)

    @pydantic.model_validator(mode="after")
    def _valid_object_id(self) -> typing.Self:
        if self.object_id != f"extension-source-{self.source_hash}":
            raise ValueError("embedded script object ID does not match its source")
        return self


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
    # Shared manager/Data Explorer options use loader names. The field name is retained
    # for compatibility with existing workspace manifests.
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
    raw_manifest = attrs.get(_persistence_constants.WORKSPACE_MANIFEST_ATTR)
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
        path = entry.get("payload_path")
        if not isinstance(path, str):
            node_path = entry.get("path")
            path = (
                f"{node_path}/{kind}"
                if isinstance(node_path, str) and isinstance(kind, str)
                else None
            )
        if (
            isinstance(uid, str)
            and isinstance(kind, str)
            and kind in {"imagetool", "tool"}
            and isinstance(path, str)
        ):
            entries.append((uid, kind, path))
    return entries


def _workspace_manifest_payload_path(
    manifest: Mapping[str, typing.Any] | None,
    uid: str,
) -> str | None:
    """Return the immutable payload path for one manifest node."""
    for entry_uid, _kind, payload_path in _workspace_manifest_payload_entries(manifest):
        if entry_uid == uid:
            return payload_path
    return None


def _workspace_manifest_legacy_reader_rebindings(
    manifest: Mapping[str, typing.Any] | None,
) -> dict[str, str]:
    """Map old node payload paths to their current immutable objects."""
    mappings: dict[str, str] = {}
    for entry in _iter_workspace_manifest_node_entries(manifest):
        node_path = entry.get("path")
        kind = entry.get("kind")
        object_id = entry.get("payload_object_id")
        if (
            isinstance(node_path, str)
            and kind in {"imagetool", "tool"}
            and isinstance(object_id, str)
            and object_id
        ):
            mappings[f"/{node_path.strip('/')}/{kind}"] = object_id
    return mappings


def _decode_workspace_attr_text(value: object) -> str | None:
    if isinstance(value, bytes):
        with contextlib.suppress(UnicodeDecodeError):
            value = value.decode()
    if isinstance(value, str) and value:
        return value
    return None


def _workspace_file_metadata_from_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> tuple[int, dict[str, typing.Any] | None]:
    schema_version = int(attrs.get("imagetool_workspace_schema_version", 1))
    manifest = None
    # Schema 4 introduced the root manifest.
    if schema_version >= 4:
        manifest = _workspace_manifest_from_attrs(attrs) or None
    return schema_version, manifest


def _current_workspace_schema_version() -> int:
    return _persistence_constants.WORKSPACE_SCHEMA_VERSION


def _workspace_schema_uses_immutable_generations(schema_version: int) -> bool:
    """Return whether a readable schema uses immutable generation storage."""
    return 5 <= schema_version <= _persistence_constants.WORKSPACE_SCHEMA_VERSION


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
    attrs["imagetool_workspace_schema_version"] = (
        _persistence_constants.WORKSPACE_LEGACY_SCHEMA_VERSION
    )


def _workspace_schema_requires_conversion(schema_version: int) -> bool:
    return schema_version < _persistence_constants.WORKSPACE_LEGACY_SCHEMA_VERSION


def _workspace_manifest_payload(
    *,
    root_order: Iterable[typing.Any],
    nodes: Iterable[Mapping[str, typing.Any]],
    erlab_version: str,
    workspace_link_id: str | None = None,
    manager_layout: Mapping[str, typing.Any] | None = None,
    loader_state: Mapping[str, typing.Any] | None = None,
    standalone_apps: Mapping[str, typing.Any] | None = None,
    option_overrides: Mapping[str, typing.Any] | None = None,
    acquisition_context: Mapping[str, typing.Any] | None = None,
    extension_requirements: Iterable[typing.Any] | None = None,
    embedded_extension_sources: Iterable[typing.Any] | None = None,
) -> dict[str, typing.Any]:
    manifest: dict[str, typing.Any] = {
        "schema_version": _persistence_constants.WORKSPACE_SCHEMA_VERSION,
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
    if extension_requirements is not None:
        # Requirements describe exact code without importing it during inspection.
        manifest["extension_requirements"] = copy.deepcopy(list(extension_requirements))
    if embedded_extension_sources is not None:
        # Embedded code is document-owned recovery material, not an execution source.
        manifest["embedded_extension_sources"] = copy.deepcopy(
            list(embedded_extension_sources)
        )
    return manifest


def _is_workspace_internal_group_name(name: typing.Any) -> bool:
    return str(name).startswith(
        (
            _persistence_constants.WORKSPACE_OBJECTS_GROUP,
            _persistence_constants.WORKSPACE_STAGING_GROUP,
            _persistence_constants.WORKSPACE_GENERATIONS_GROUP,
            *_persistence_constants.WORKSPACE_LEGACY_TEMP_GROUP_PREFIXES,
            _persistence_constants.WORKSPACE_TRANSACTION_GROUP_PREFIX,
        )
    )


def _workspace_manifest_attrs(
    attrs: Mapping[typing.Any, typing.Any],
) -> list[list[dict[str, typing.Any]]]:
    """Encode dataset attributes for storage inside a JSON manifest."""
    encoded: list[list[dict[str, typing.Any]]] = []
    for key, value in attrs.items():
        try:
            encoded.append(
                [
                    _serialization.encode_attr_key(key),
                    _serialization.encode_attr_value(value),
                ]
            )
        except TypeError:
            logger.warning(
                "Dropping workspace attribute %r with unsupported value type %s",
                key,
                type(value).__name__,
            )
    return encoded


def _restore_workspace_manifest_attrs(
    payload: object,
) -> dict[typing.Hashable, typing.Any]:
    """Decode dataset attributes stored in a generation manifest."""
    if not isinstance(payload, list):
        raise TypeError("Workspace manifest attributes must be a list")
    attrs: dict[typing.Hashable, typing.Any] = {}
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError("Workspace manifest attribute entry is invalid")
        attrs[_serialization.decode_attr_key(item[0])] = (
            _serialization.decode_attr_value(item[1])
        )
    return attrs


def _workspace_root_keys(
    tree: typing.Any, manifest: Mapping[str, typing.Any] | None
) -> list[str]:
    root_keys: list[str] = []
    root_key_set: set[str] = set()
    if manifest is not None:
        raw_root_order = manifest.get("root_order", ())
        if isinstance(raw_root_order, list):
            for item in raw_root_order:
                key = str(item)
                if (
                    key not in root_key_set
                    and key != "figures"
                    and not _is_workspace_internal_group_name(item)
                ):
                    root_keys.append(key)
                    root_key_set.add(key)
    for item in tree:
        key = str(item)
        if (
            key not in root_key_set
            and key != "figures"
            and not _is_workspace_internal_group_name(item)
        ):
            root_keys.append(key)
            root_key_set.add(key)
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
        if key == _persistence_constants.WORKSPACE_ENCODED_ATTRS_ATTR:
            existing_entries = _workspace_encoded_attr_entries(value)
            if existing_entries is not None:
                encoded_entries.extend(existing_entries)
                continue
        if (
            key != _persistence_constants.WORKSPACE_ENCODED_ATTRS_ATTR
            and _serialization.attr_value_writes_natively(value)
        ):
            serializable[key] = value
            continue
        try:
            encoded_entries.append(
                [
                    _serialization.encode_attr_key(key),
                    _serialization.encode_attr_value(value),
                ]
            )
        except TypeError:
            logger.warning(
                "Dropping workspace attribute %r with unsupported value type %s",
                key,
                type(value).__name__,
            )
    if encoded_entries:
        serializable[_persistence_constants.WORKSPACE_ENCODED_ATTRS_ATTR] = json.dumps(
            {
                "version": _persistence_constants.WORKSPACE_ENCODED_ATTRS_VERSION,
                "attrs": encoded_entries,
            },
            separators=(",", ":"),
        )
    return serializable


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
        or payload.get("version")
        != _persistence_constants.WORKSPACE_ENCODED_ATTRS_VERSION
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
        attrs.get(_persistence_constants.WORKSPACE_ENCODED_ATTRS_ATTR)
    )
    if encoded_entries is None:
        return dict(attrs)
    restored = {
        key: value
        for key, value in attrs.items()
        if key != _persistence_constants.WORKSPACE_ENCODED_ATTRS_ATTR
    }
    for key_payload, value_payload in encoded_entries:
        try:
            key = _serialization.decode_attr_key(key_payload)
            value = _serialization.decode_attr_value(value_payload)
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
    return _serialization.restore_private_coords(restored)
