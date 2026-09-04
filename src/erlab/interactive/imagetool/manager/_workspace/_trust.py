"""Workspace adapter for durable executable-code trust."""

from __future__ import annotations

import json
import pathlib
import typing
from collections.abc import Mapping

import erlab
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
from erlab.interactive._code_trust import (
    create_manifest,
    document_path_is_trusted,
    relocate_manifest_entries,
)
from erlab.interactive._code_trust._payloads import (
    CODE_PAYLOAD_ENTRIES_ATTR,
    store_code_payload_entries,
)
from erlab.interactive._saved_tools import resolve_saved_tool_class
from erlab.interactive.imagetool._provenance._model import (
    ToolProvenanceSpec,
    parse_tool_provenance_spec,
)
from erlab.interactive.imagetool._provenance._trust import provenance_code_trust_entries

if typing.TYPE_CHECKING:
    import os

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.utils import ToolWindow

WORKSPACE_CODE_TRUST_DOMAIN = "erlab.workspace"
WORKSPACE_CODE_TRUST_POLICY_VERSION = 7
_MANIFEST_ID = (WORKSPACE_CODE_TRUST_DOMAIN, WORKSPACE_CODE_TRUST_POLICY_VERSION)
_MANAGER_LIVE_SOURCE_SPEC_ATTR = "manager_node_live_source_spec"
_MANAGER_PROVENANCE_SPEC_ATTR = "manager_node_provenance_spec"
_ITOOL_PROVENANCE_SPEC_ATTR = "itool_provenance_spec"
_TOOL_SOURCE_SPEC_ATTR = "tool_source_spec"
_TOOL_SCRIPT_INPUTS_ATTR = "tool_script_inputs"
_TOOL_PRIMARY_INPUT_ATTR = "tool_primary_input"
_TOOL_DATA_REFERENCES_ATTR = "tool_data_references"
_PROVENANCE_ATTRS = (_MANAGER_PROVENANCE_SPEC_ATTR, _ITOOL_PROVENANCE_SPEC_ATTR)
_SOURCE_ATTRS = {
    _MANAGER_LIVE_SOURCE_SPEC_ATTR: "source",
    _TOOL_SOURCE_SPEC_ATTR: "tool-source",
}
_ALL_SOURCE_ATTRS = tuple(_SOURCE_ATTRS.items())
_CODE_TRUST_ATTRS = frozenset(
    (
        *_PROVENANCE_ATTRS,
        *_SOURCE_ATTRS,
        CODE_PAYLOAD_ENTRIES_ATTR,
        "tool_cls_qualname",
        "tool_state",
        _TOOL_SCRIPT_INPUTS_ATTR,
        _TOOL_PRIMARY_INPUT_ATTR,
        _TOOL_DATA_REFERENCES_ATTR,
    )
)


def workspace_path_is_trusted(path: str | os.PathLike[str]) -> bool:
    """Return whether an ``.itws`` path matches the user trusted-folder policy."""
    document = pathlib.Path(path)
    if document.suffix.lower() != ".itws":
        return False
    security = erlab.interactive.options.model.security
    return document_path_is_trusted(
        WORKSPACE_CODE_TRUST_DOMAIN,
        document,
        (
            (WORKSPACE_CODE_TRUST_DOMAIN, folder)
            for folder in security.trusted_workspace_folders
        ),
    )


def _decoded_json_mapping(value: typing.Any) -> Mapping[str, typing.Any] | None:
    if not isinstance(value, str):
        return None
    try:
        payload = json.loads(value)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _entry_attrs(entry: Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    payload = entry.get("payload_attrs")
    if isinstance(payload, dict):
        return {key: payload[key] for key in _CODE_TRUST_ATTRS if key in payload}
    if not isinstance(payload, list):
        raise TypeError("Workspace node manifest must contain payload attributes")
    attrs: dict[str, typing.Any] = {}
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError("Workspace manifest attribute entry is invalid")
        encoded_key, encoded_value = item
        if not isinstance(encoded_key, Mapping) or encoded_key.get("kind") != "str":
            continue
        key = encoded_key.get("value")
        if key not in _CODE_TRUST_ATTRS:
            continue
        if (
            not isinstance(encoded_value, Mapping)
            or encoded_value.get("kind") != "str"
            or not isinstance(encoded_value.get("value"), str)
        ):
            raise TypeError(f"Workspace code trust attribute {key!r} must be a string")
        attrs[typing.cast("str", key)] = encoded_value["value"]
    return attrs


def _tool_code_trust_from_attrs(
    attrs: Mapping[str, typing.Any],
):
    identifier = attrs.get("tool_cls_qualname")
    if identifier is None:
        return None
    if not isinstance(identifier, str):
        raise TypeError("Workspace tool class identifier must be a string")
    tool_state = attrs.get("tool_state")
    if not isinstance(tool_state, str):
        raise TypeError("Workspace tool state must be JSON text")
    try:
        tool_cls = typing.cast("type[ToolWindow]", resolve_saved_tool_class(identifier))
        status = tool_cls.StateModel.model_validate_json(tool_state)
        manifest_attrs = dict(attrs)
        manifest_attrs.pop(_TOOL_SOURCE_SPEC_ATTR, None)
        return tool_cls._code_trust_manifest_from_saved_metadata(status, manifest_attrs)
    except Exception as exc:
        raise TypeError("Workspace tool trust metadata could not be inspected") from exc


def _node_code_trust_entries(
    path: str,
    attrs: Mapping[str, typing.Any],
    tool_manifest,
    *,
    provenance_specs=(),
    source_specs=(),
    saved_provenance_attrs=_PROVENANCE_ATTRS,
    saved_source_attrs=_ALL_SOURCE_ATTRS,
):
    """Return deduplicated provenance and relocated tool entries for one node."""
    entries = []
    entry_identities: set[bytes] = set()
    seen_specs: list[tuple[str, ToolProvenanceSpec]] = []
    located_specs = [
        *((f"{path}/provenance", spec) for spec in provenance_specs),
        *(
            (f"{path}/provenance", _decoded_json_mapping(attrs.get(attr)))
            for attr in saved_provenance_attrs
        ),
        *source_specs,
    ]
    located_specs.extend(
        (
            f"{path}/{segment}/provenance",
            _decoded_json_mapping(attrs.get(attr)),
        )
        for attr, segment in saved_source_attrs
    )
    for location, provenance_spec in located_specs:
        parsed_spec = parse_tool_provenance_spec(provenance_spec)
        if parsed_spec is None or any(
            location == previous_location and parsed_spec == previous_spec
            for previous_location, previous_spec in seen_specs
        ):
            continue
        seen_specs.append((location, parsed_spec))
        for entry in provenance_code_trust_entries(
            parsed_spec,
            location_prefix=location,
        ):
            identity = entry.document_identity()
            if identity in entry_identities:
                continue
            entry_identities.add(identity)
            entries.append(entry)
    if tool_manifest is not None:
        entries.extend(relocate_manifest_entries(tool_manifest, location_prefix=path))
    return entries


def workspace_code_trust_manifest(
    workspace_manifest: Mapping[str, typing.Any],
    *,
    selected_paths: set[str] | None = None,
):
    """Build the signed executable manifest from workspace metadata only."""
    entries = []
    for node_entry in workspace_format._iter_workspace_manifest_node_entries(
        workspace_manifest
    ):
        path = node_entry.get("path")
        if not isinstance(path, str) or (
            selected_paths is not None and path not in selected_paths
        ):
            continue
        attrs = _entry_attrs(node_entry)
        is_imagetool = (
            node_entry.get("kind") == "imagetool" and "tool_cls_qualname" not in attrs
        )
        if not is_imagetool:
            attrs.pop(_MANAGER_PROVENANCE_SPEC_ATTR, None)
        tool_manifest = _tool_code_trust_from_attrs(attrs)
        entries += _node_code_trust_entries(
            path,
            attrs,
            tool_manifest,
            saved_provenance_attrs=(
                _PROVENANCE_ATTRS if is_imagetool else (_ITOOL_PROVENANCE_SPEC_ATTR,)
            ),
        )
    return create_manifest(*_MANIFEST_ID, entries)


def current_workspace_code_trust_manifest(
    manager: ImageToolManager,
):
    """Build a review manifest from current metadata without reading data arrays."""
    entries = []
    node_path = manager._workspace_controller.saving._workspace_node_path
    for uid in sorted(manager._tool_graph.nodes, key=node_path):
        node = manager._tool_graph.nodes[uid]
        path = node_path(uid)
        tool = node.tool_window
        tool_manifest = None if tool is None else tool._current_code_trust_manifest()
        attrs = node.pending_workspace_payload_attrs or {}
        if tool is None:
            tool_manifest = _tool_code_trust_from_attrs(attrs)
        source_spec = node.source_spec
        source_attr, source_segment = (
            (_MANAGER_LIVE_SOURCE_SPEC_ATTR, "source")
            if node.is_imagetool
            else (_TOOL_SOURCE_SPEC_ATTR, "tool-source")
        )
        include_source = node.is_imagetool or tool is None or tool_manifest is None
        entries += _node_code_trust_entries(
            path,
            attrs,
            tool_manifest,
            provenance_specs=(
                (node.passive_displayed_provenance_spec,) if node.is_imagetool else ()
            ),
            saved_provenance_attrs=_PROVENANCE_ATTRS if node.is_imagetool else (),
            source_specs=(
                ((f"{path}/{source_segment}/provenance", source_spec),)
                if include_source
                else ()
            ),
            saved_source_attrs=(
                ((source_attr, source_segment),)
                if tool is None and source_spec is None
                else ()
            ),
        )
    return create_manifest(*_MANIFEST_ID, entries)


def inspect_pending_workspace_code_payloads(manager: ImageToolManager) -> None:
    """Inspect legacy opaque payloads when the user requests trust review."""
    updates = []
    for node in manager._tool_graph.nodes.values():
        pending = node.pending_workspace_tool_payload
        attrs = node.pending_workspace_payload_attrs
        if pending is None:
            continue
        if attrs is None:
            raise TypeError("Workspace tool payload metadata is missing")
        if CODE_PAYLOAD_ENTRIES_ATTR in attrs:
            continue
        identifier = attrs.get("tool_cls_qualname")
        if not isinstance(identifier, str):
            raise TypeError("Workspace tool class identifier must be a string")
        tool_cls = resolve_saved_tool_class(identifier)
        if not issubclass(tool_cls, erlab.interactive.utils.ToolWindow):
            raise TypeError("Workspace tool class is not a ToolWindow subclass")
        base_tool_cls = erlab.interactive.utils.ToolWindow
        saved_inspector = tool_cls._code_trust_payload_entries_from_saved_dataset
        base_saved_inspector = (
            base_tool_cls._code_trust_payload_entries_from_saved_dataset
        )
        if getattr(saved_inspector, "__func__", saved_inspector) is getattr(
            base_saved_inspector, "__func__", base_saved_inspector
        ):
            if (
                tool_cls._code_trust_payload_entries
                is not base_tool_cls._code_trust_payload_entries
                or tool_cls._code_trust_payload_entries_from_dataset
                is not base_tool_cls._code_trust_payload_entries_from_dataset
            ):
                raise TypeError(
                    "Workspace tool opaque payload cannot be inspected without "
                    "loading the tool"
                )
            entries = ()
        else:
            workspace_path, payload_path = pending
            opened = workspace_arrays.open_workspace_dataset(
                workspace_path,
                payload_path,
                chunks={},
            )
            try:
                inspected_entries = saved_inspector(opened)
                if inspected_entries is None:
                    raise TypeError(
                        "Workspace tool saved-payload inspector did not return entries"
                    )
                entries = tuple(inspected_entries)
            finally:
                opened.close()
        updated_attrs = dict(attrs)
        store_code_payload_entries(updated_attrs, entries)
        updates.append((node, updated_attrs))

    for node, attrs in updates:
        node.update_pending_workspace_payload_attrs(attrs)
