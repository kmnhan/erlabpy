"""Workspace metadata and immutable objects for provenance revisions."""

from __future__ import annotations

import functools
import json
import typing
from collections.abc import Mapping

from erlab.interactive import _persistence_constants
from erlab.interactive.imagetool import _serialization
from erlab.interactive.imagetool._provenance._model import parse_tool_provenance_spec
from erlab.interactive.imagetool.manager._revisions import (
    _ProvenanceRevision,
    _ProvenanceRevisionStore,
    _script_input_revision_ids,
    _spec_revision_ids,
    requires_legacy_replay_source,
)
from erlab.interactive.imagetool.manager._workspace import _arrays as workspace_arrays
from erlab.interactive.imagetool.manager._workspace import _format as workspace_format
from erlab.interactive.imagetool.manager._workspace._storage import (
    _WorkspaceObjectWrite,
)
from erlab.interactive.imagetool.manager._workspace._store import WorkspaceStore

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    import xarray as xr

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode


def retained_revision_ids(manager: ImageToolManager) -> frozenset[str]:
    ids: set[str] = set()
    for node in manager._tool_graph.nodes.values():
        if node._current_revision_id is not None:
            ids.add(node._current_revision_id)
        ids.update(
            revision_id
            for role, (token, revision_id) in node._revision_ids_by_role.items()
            if token == node.snapshot_token_for_role(role)
        )
        ids.update(_spec_revision_ids(node.passive_displayed_provenance_spec))
        ids.update(_spec_revision_ids(node.source_spec))
        ids.update(_script_input_revision_ids(node.active_tool_script_inputs))
    return frozenset(ids)


def revision_manifest_entry(
    store: _ProvenanceRevisionStore, record: _ProvenanceRevision
) -> dict[str, typing.Any]:
    spec = store.normalize_provenance(record.provenance_spec)
    return {
        "revision_id": record.revision_id,
        "checkpoint_object_id": (
            record.checkpoint_object_id if record.checkpoint is not None else None
        ),
        "provenance_spec": (None if spec is None else spec.model_dump(mode="json")),
        "input_revisions": dict(record.input_revisions),
        "primary_revision_id": record.primary_revision_id,
        "source_owner_uid": record.source_owner_uid,
    }


def revision_object_writes(
    store: _ProvenanceRevisionStore, records: Iterable[_ProvenanceRevision]
) -> tuple[_WorkspaceObjectWrite, ...]:
    writes = []
    for record in records:
        checkpoint = record.checkpoint
        if checkpoint is None:
            continue
        if checkpoint.source_file is not None and checkpoint.source_path is not None:
            writes.append(
                _WorkspaceObjectWrite(
                    record.checkpoint_object_id,
                    source_file=checkpoint.source_file,
                    source_path=checkpoint.source_path,
                )
            )
            continue
        data = store.resolve_data(record.revision_id)
        dataset = data.to_dataset(name=_persistence_constants.ITOOL_DATA_NAME)
        dataset.attrs["revision_data_name"] = json.dumps(
            _serialization.encode_attr_value(data.name)
        )
        dataset.attrs["revision_coordinate_names"] = json.dumps(
            [_serialization.encode_attr_key(name) for name in data.coords]
        )
        dataset = workspace_format._sanitize_workspace_attr_names(
            _serialization.encode_private_coords(dataset)
        )
        writes.append(
            _WorkspaceObjectWrite(record.checkpoint_object_id, dataset=dataset)
        )
    return tuple(writes)


def _read_revision_checkpoint(path: str, group: str) -> xr.DataArray:
    dataset = workspace_arrays.open_workspace_dataset(path, group, chunks={})
    dataset = workspace_format._restore_workspace_dataset_attrs(dataset)
    data = dataset[_persistence_constants.ITOOL_DATA_NAME]
    coordinate_names = {
        _serialization.decode_attr_key(name)
        for name in json.loads(dataset.attrs["revision_coordinate_names"])
    }
    data = data.drop_vars(
        [name for name in data.coords if name not in coordinate_names]
    )
    data.name = _serialization.decode_attr_value(
        json.loads(dataset.attrs["revision_data_name"])
    )
    return data


def normalize_workspace_nodes(manager: ImageToolManager) -> None:
    for node in manager._tool_graph.nodes.values():
        if not node.is_imagetool:
            continue
        spec = node.provenance_spec
        seed_id = None
        if requires_legacy_replay_source(spec) and node.replay_source_data is not None:
            seed_id = manager._provenance_revisions.add(node.replay_source_data)
        normalized = manager._provenance_revisions.normalize_provenance(
            spec, replay_source_revision_id=seed_id
        )
        if normalized != spec or seed_id is not None:
            node.adopt_normalized_provenance(normalized)
        if (
            node.imagetool is not None
            and node._current_revision_id is not None
            and node.current_revision_id is None
        ):
            node.revision_id_for_role("source")


def saved_provenance_matches_node(
    node: _ManagedWindowNode, attrs_payload: object
) -> bool:
    if not node.is_imagetool:
        return True
    if attrs_payload is None:
        return False
    try:
        attrs = (
            workspace_format._restore_workspace_manifest_attrs(attrs_payload)
            if isinstance(attrs_payload, list)
            else attrs_payload
        )
        if not isinstance(attrs, Mapping):
            return False
        raw = attrs.get(_persistence_constants.MANAGER_PROVENANCE_SPEC_ATTR)
        saved = parse_tool_provenance_spec(json.loads(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return False
    else:
        return saved == node.provenance_spec


def bind_saved_revision_sources(
    store: _ProvenanceRevisionStore,
    manifest: Mapping[str, typing.Any],
    path: str | os.PathLike[str],
) -> None:
    for entry in manifest.get("provenance_revisions", ()):
        revision_id = entry["revision_id"]
        checkpoint_id = entry.get("checkpoint_object_id")
        if checkpoint_id is not None and revision_id in store:
            source_path = WorkspaceStore.object_path(checkpoint_id)
            store.bind_checkpoint_source(
                revision_id,
                str(path),
                source_path,
                provider=functools.partial(
                    _read_revision_checkpoint, str(path), source_path
                ),
            )


def restore_revision_store(
    store: _ProvenanceRevisionStore,
    manifest: Mapping[str, typing.Any],
    path: str | os.PathLike[str],
) -> None:
    entries = manifest.get("provenance_revisions", [])
    if not isinstance(entries, list):
        raise TypeError("Workspace provenance revisions must be a list")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Workspace provenance revision must be an object")
        revision_id = entry.get("revision_id")
        if not isinstance(revision_id, str) or not revision_id:
            raise ValueError("Workspace provenance revision has no ID")
        checkpoint_id = entry.get("checkpoint_object_id")
        provider = None
        source_path = None
        if checkpoint_id is not None:
            if checkpoint_id != f"revision-{revision_id}":
                raise ValueError("Workspace revision checkpoint ID is not canonical")
            source_path = WorkspaceStore.object_path(checkpoint_id)
            provider = functools.partial(
                _read_revision_checkpoint, str(path), source_path
            )
        inputs = entry.get("input_revisions", {})
        if not isinstance(inputs, Mapping) or any(
            not isinstance(name, str) or not isinstance(ref, str)
            for name, ref in inputs.items()
        ):
            raise TypeError("Workspace revision inputs must map names to revision IDs")
        primary = entry.get("primary_revision_id")
        if primary is not None and not isinstance(primary, str):
            raise TypeError(
                "Workspace revision primary reference must be a revision ID"
            )
        source_owner_uid = entry.get("source_owner_uid")
        if source_owner_uid is not None and not isinstance(source_owner_uid, str):
            raise TypeError("Workspace revision source owner must be a node UID")
        spec = store.normalize_provenance(
            parse_tool_provenance_spec(entry.get("provenance_spec"))
        )
        inputs = dict(inputs)
        existing = store.get(revision_id)
        if existing is not None:
            existing_metadata = revision_manifest_entry(store, existing)
            loaded_metadata = {
                "revision_id": revision_id,
                "checkpoint_object_id": checkpoint_id,
                "provenance_spec": None
                if spec is None
                else spec.model_dump(mode="json"),
                "input_revisions": inputs,
                "primary_revision_id": primary,
                "source_owner_uid": source_owner_uid,
            }
            # Serialized NaN operation values compare equal across JSON decoding.
            if json.dumps(existing_metadata, sort_keys=True) != json.dumps(
                loaded_metadata, sort_keys=True
            ):
                raise ValueError("Conflicting workspace provenance revision ID")
            continue
        store.add(
            spec=spec,
            revision_id=revision_id,
            input_revisions=inputs,
            primary_revision_id=primary,
            source_owner_uid=source_owner_uid,
            provider=provider,
            source_file=str(path) if source_path is not None else None,
            source_path=source_path,
        )
