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
    _spec_revision_ids,
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


def retained_revision_ids(manager: ImageToolManager) -> frozenset[str]:
    ids: set[str] = set()
    for node in manager._tool_graph.nodes.values():
        if node.current_revision_id is not None:
            ids.add(node.current_revision_id)
        ids.update(_spec_revision_ids(node.passive_displayed_provenance_spec))
        ids.update(_spec_revision_ids(node.source_spec))
    return frozenset(ids)


def revision_manifest_entry(record: _ProvenanceRevision) -> dict[str, typing.Any]:
    return {
        "revision_id": record.revision_id,
        "checkpoint_object_id": (
            record.checkpoint_object_id if record.checkpoint is not None else None
        ),
        "provenance_spec": (
            None
            if record.provenance_spec is None
            else record.provenance_spec.model_dump(mode="json")
        ),
        "input_revisions": dict(record.input_revisions),
        "primary_revision_id": record.primary_revision_id,
        "replay_source_revision_id": record.replay_source_revision_id,
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
        spec = parse_tool_provenance_spec(entry.get("provenance_spec"))
        replay_source = entry.get("replay_source_revision_id")
        if replay_source is not None and not isinstance(replay_source, str):
            raise TypeError("Workspace revision replay source must be a revision ID")
        existing = store.get(revision_id)
        if existing is not None:
            if revision_manifest_entry(existing) != dict(entry):
                raise ValueError("Conflicting workspace provenance revision ID")
            continue
        store.add(
            spec=spec,
            revision_id=revision_id,
            input_revisions=inputs,
            primary_revision_id=primary,
            replay_source_revision_id=replay_source,
            provider=provider,
            source_file=str(path) if source_path is not None else None,
            source_path=source_path,
        )
