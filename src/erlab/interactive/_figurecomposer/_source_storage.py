"""Source manifests shared by workspace and clipboard storage."""

from __future__ import annotations

import dataclasses
import json
import typing

import erlab
from erlab.interactive import _persistence_constants
from erlab.interactive._figurecomposer._model._state import FigureRecipeState
from erlab.interactive.imagetool._serialization import decode_attr_key, encode_attr_key

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr

    from erlab.interactive._figurecomposer._model._state import FigureSourceState


SOURCE_MANIFEST_ATTR = "figure_composer_source_manifest"


def embedded_workspace_sources(
    attrs: Mapping[str, typing.Any],
) -> dict[str, FigureSourceState]:
    """Find captured array inputs that still have an embedded workspace payload."""
    if attrs.get("tool_cls_qualname") != (
        "erlab.interactive._figurecomposer._tool:FigureComposerTool"
    ):
        return {}
    tool_cls = erlab.interactive.utils.ToolWindow
    try:
        recipe = FigureRecipeState.model_validate_json(attrs["tool_state"])
        references = tool_cls._saved_tool_data_references_from_metadata(attrs)
    except (KeyError, TypeError, ValueError):
        return {}
    sources = {}
    for source in recipe.sources:
        if source.kind != "array" or not source.node_uid or not source.use_node_data:
            continue
        variable_name = (
            _persistence_constants.SAVED_TOOL_DATA_NAME
            if source.name == recipe.primary_source
            else source.name
        )
        if variable_name not in references:
            sources[variable_name] = source
    return sources


def restore_workspace_source_name(
    attrs: Mapping[str, typing.Any],
    source_name: str,
    variable_name: str,
    data: xr.DataArray,
) -> xr.DataArray:
    """Restore the captured array name from current or legacy workspace metadata."""
    source_manifest = attrs.get(SOURCE_MANIFEST_ATTR)
    if source_manifest is not None:
        names = json.loads(source_manifest)["sources"][source_name]["array_names"]
        return data.rename(decode_attr_key(names["data"]))
    if variable_name == _persistence_constants.SAVED_TOOL_DATA_NAME:
        data_name = attrs.get("tool_data_name")
        if data_name == _persistence_constants.NONE_TOOL_DATA_NAME:
            data_name = None
        return data.rename(data_name)
    return data


@dataclasses.dataclass(frozen=True)
class SourceBundle:
    manifest: dict[str, typing.Any]
    arrays: dict[str, xr.DataArray]


def encode_source_bundle(
    payloads: Mapping[str, object], *, primary_source: str | None = None
) -> SourceBundle:
    from erlab.interactive._figurecomposer._sources import (
        encode_source_payload,
        source_payload_kind,
    )

    entries: dict[str, typing.Any] = {}
    arrays: dict[str, xr.DataArray] = {}
    reserved = set(payloads) | {_persistence_constants.SAVED_TOOL_DATA_NAME}
    for source_index, (name, payload) in enumerate(payloads.items()):
        encoded = encode_source_payload(payload)
        slots: dict[str, str] = {}
        names: dict[str, dict[str, typing.Any]] = {}
        for slot_index, (slot, data) in enumerate(encoded.arrays.items()):
            if slot == "data":
                variable = (
                    _persistence_constants.SAVED_TOOL_DATA_NAME
                    if name == primary_source
                    else name
                )
            else:
                variable = f"__figure_source_{source_index}_{slot_index}"
                while variable in reserved:
                    variable += "_"
            reserved.add(variable)
            slots[slot] = variable
            names[slot] = encode_attr_key(data.name)
            arrays[variable] = data
        entries[name] = {
            "kind": source_payload_kind(payload),
            "payload": encoded.metadata,
            "arrays": slots,
            "array_names": names,
        }
    return SourceBundle({"version": 1, "sources": entries}, arrays)


def decode_source_bundle(
    manifest: Mapping[str, typing.Any],
    arrays: Mapping[str, xr.DataArray],
    *,
    skip_missing_arrays: bool = False,
) -> dict[str, object]:
    from erlab.interactive._figurecomposer._sources import (
        EncodedSourcePayload,
        decode_source_payload,
    )

    if manifest.get("version") != 1:
        raise ValueError("Unsupported figure source manifest version.")
    entries = manifest.get("sources", {})
    if not isinstance(entries, dict):
        raise TypeError("Invalid figure source manifest.")
    restored: dict[str, object] = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise TypeError("Invalid figure source entry.")
        slots = entry.get("arrays", {})
        array_names = entry.get("array_names", {})
        if not isinstance(slots, dict) or not isinstance(array_names, dict):
            raise TypeError("Invalid figure source array slots.")
        if any(variable not in arrays for variable in slots.values()):
            if skip_missing_arrays:
                continue
            raise ValueError(f"Stored arrays for source {name!r} are unavailable.")
        source_arrays = {}
        for slot, variable in slots.items():
            array_name = array_names.get(slot, arrays[variable].name)
            if isinstance(array_name, dict):
                array_name = decode_attr_key(array_name)
            source_arrays[slot] = arrays[variable].rename(array_name)
        restored[name] = decode_source_payload(
            entry["kind"], EncodedSourcePayload(entry["payload"], source_arrays)
        )
    return restored
