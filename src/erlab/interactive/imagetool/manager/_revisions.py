"""Immutable data and recipe revisions independent of managed windows."""

from __future__ import annotations

import copy
import json
import typing
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace

import xarray as xr

from erlab.interactive.imagetool._provenance._model import (
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    parse_tool_provenance_spec,
)
from erlab.interactive.imagetool._provenance._operations import ScriptCodeOperation
from erlab.interactive.imagetool._provenance._trust import (
    script_replay_source_input_names,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@dataclass(frozen=True)
class _RevisionCheckpoint:
    provider: Callable[[], xr.DataArray]
    source_file: str | None = None
    source_path: str | None = None


def _capture_checkpoint(data: xr.DataArray) -> tuple[_RevisionCheckpoint, xr.DataArray]:
    # A deep copy alone preserves lazy graphs that can read mutable external data.
    protected = data.copy(deep=True).load()

    def _copy_snapshot() -> xr.DataArray:
        return protected.copy(deep=True)

    return _RevisionCheckpoint(_copy_snapshot), protected


@dataclass(frozen=True)
class _ProvenanceRevision:
    revision_id: str
    provenance_spec: ToolProvenanceSpec | None
    input_revisions: tuple[tuple[str, str], ...] = ()
    primary_revision_id: str | None = None
    source_owner_uid: str | None = None
    checkpoint: _RevisionCheckpoint | None = None

    @property
    def checkpoint_object_id(self) -> str:
        return f"revision-{self.revision_id}"


def _spec_revision_ids(spec: ToolProvenanceSpec | None) -> frozenset[str]:
    if spec is None:
        return frozenset()
    ids: set[str] = set()

    def _visit(payload: object) -> None:
        if isinstance(payload, Mapping):
            inputs = payload.get("script_inputs", ())
            if isinstance(inputs, (list, tuple)):
                for item in inputs:
                    if isinstance(item, Mapping):
                        revision_id = item.get("revision_id")
                        if isinstance(revision_id, str) and revision_id:
                            ids.add(revision_id)
            for value in payload.values():
                _visit(value)
        elif isinstance(payload, (list, tuple)):
            for value in payload:
                _visit(value)

    _visit(spec.model_dump(mode="python"))
    return frozenset(ids)


def _same_provenance(
    left: ToolProvenanceSpec | None, right: ToolProvenanceSpec | None
) -> bool:
    """Compare recorded recipes, including serialized NaN operation values."""
    return json.dumps(
        None if left is None else left.model_dump(mode="json"), sort_keys=True
    ) == json.dumps(
        None if right is None else right.model_dump(mode="json"), sort_keys=True
    )


def requires_legacy_replay_source(spec: ToolProvenanceSpec | None) -> bool:
    if spec is None:
        return False
    if spec.kind in {"full_data", "public_data", "selection"}:
        return True
    if spec.kind != "script":
        return False
    if spec.script_inputs:
        return any(
            item.node_uid is None
            and item.node_snapshot_token is not None
            and item.revision_id is None
            for item in spec.script_inputs
        )
    return bool(script_replay_source_input_names(spec))


def _script_input_revision_ids(inputs: Iterable[ScriptInput]) -> frozenset[str]:
    ids: set[str] = set()
    for item in inputs:
        if item.revision_id is not None:
            ids.add(item.revision_id)
        ids.update(_spec_revision_ids(item.parsed_provenance_spec()))
        ids.update(_spec_revision_ids(item.parsed_source_spec()))
    return frozenset(ids)


class _ProvenanceRevisionStore:
    def __init__(self) -> None:
        self._records: dict[str, _ProvenanceRevision] = {}
        self._data: dict[str, xr.DataArray] = {}

    def __contains__(self, revision_id: object) -> bool:
        return revision_id in self._records

    def add(
        self,
        data: xr.DataArray | None = None,
        spec: ToolProvenanceSpec | None = None,
        *,
        input_revisions: Mapping[str, str] | Iterable[tuple[str, str]] = (),
        primary_revision_id: str | None = None,
        source_owner_uid: str | None = None,
        revision_id: str | None = None,
        provider: Callable[[], xr.DataArray] | None = None,
        source_file: str | None = None,
        source_path: str | None = None,
    ) -> str:
        revision_id = uuid.uuid4().hex if revision_id is None else revision_id
        if (
            not revision_id
            or "/" in revision_id
            or "\x00" in revision_id
            or revision_id in {".", ".."}
        ):
            raise ValueError("Invalid provenance revision ID")
        if revision_id in self._records:
            raise ValueError(f"Provenance revision {revision_id!r} already exists")
        if data is not None and provider is not None:
            raise ValueError("Supply revision data or a lazy provider, not both")
        if data is not None and not isinstance(data, xr.DataArray):
            raise TypeError("Revision data must be a DataArray")
        if provider is not None and not callable(provider):
            raise TypeError("Revision checkpoint provider must be callable")
        if (source_file is None) != (source_path is None):
            raise ValueError("Checkpoint source requires a file and a group path")
        if source_file is not None and provider is None:
            raise ValueError("Checkpoint source requires a lazy provider")
        parsed_spec = parse_tool_provenance_spec(spec)
        inputs = (
            tuple(input_revisions.items())
            if isinstance(input_revisions, Mapping)
            else tuple(input_revisions)
        )
        if len({name for name, _ in inputs}) != len(inputs):
            raise ValueError("Revision input names must be unique")
        if any(
            not isinstance(name, str) or not name or not isinstance(ref, str) or not ref
            for name, ref in inputs
        ):
            raise ValueError("Revision inputs require a name and revision ID")
        owned_spec = None if parsed_spec is None else parsed_spec.model_copy(deep=True)
        checkpoint = (
            None
            if provider is None
            else _RevisionCheckpoint(provider, source_file, source_path)
        )
        protected = None
        if data is not None:
            checkpoint, protected = _capture_checkpoint(data)
        record = _ProvenanceRevision(
            revision_id=revision_id,
            provenance_spec=owned_spec,
            input_revisions=inputs,
            primary_revision_id=primary_revision_id,
            source_owner_uid=source_owner_uid,
            checkpoint=checkpoint,
        )
        self._records[revision_id] = record
        if protected is not None:
            self._data[revision_id] = protected
        return revision_id

    @property
    def revision_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def get(self, revision_id: str) -> _ProvenanceRevision | None:
        record = self._records.get(revision_id)
        if record is None:
            return None
        return replace(record, provenance_spec=copy.deepcopy(record.provenance_spec))

    def has_data(self, revision_id: str) -> bool:
        record = self._records.get(revision_id)
        return record is not None and record.checkpoint is not None

    def resolve_data(self, revision_id: str) -> xr.DataArray:
        record = self._records[revision_id]
        if record.checkpoint is None:
            raise ValueError(f"Revision {revision_id!r} has no data checkpoint")
        if revision_id not in self._data:
            data = record.checkpoint.provider()
            if not isinstance(data, xr.DataArray):
                raise TypeError("Revision checkpoint must provide a DataArray")
            self._data[revision_id] = data.copy(deep=True)
        return self._data[revision_id].copy(deep=True)

    def reachable(self, revision_ids: Iterable[str]) -> tuple[_ProvenanceRevision, ...]:
        pending = list(revision_ids)
        visited: set[str] = set()
        while pending:
            revision_id = pending.pop()
            if revision_id in visited:
                continue
            record = self._records.get(revision_id)
            if record is None:
                # Keep unavailable references in recipes and named input metadata.
                # Their missing checkpoints cannot block retention of known data.
                continue
            visited.add(revision_id)
            pending.extend(ref for _, ref in record.input_revisions)
            if record.primary_revision_id is not None:
                pending.append(record.primary_revision_id)
            pending.extend(_spec_revision_ids(record.provenance_spec))
        return tuple(
            replace(
                self._records[revision_id],
                provenance_spec=copy.deepcopy(
                    self._records[revision_id].provenance_spec
                ),
            )
            for revision_id in sorted(visited)
        )

    def copy(self) -> _ProvenanceRevisionStore:
        result = _ProvenanceRevisionStore()
        result._records = dict(self._records)
        result._data = dict(self._data)
        return result

    def stage(self) -> _ProvenanceRevisionStore:
        return _ProvenanceRevisionStore()

    def merge(self, staged: _ProvenanceRevisionStore) -> None:
        conflicts = self.revision_ids & staged.revision_ids
        if conflicts:
            raise ValueError(
                f"Provenance revisions already exist: {', '.join(sorted(conflicts))}"
            )
        self._records.update(staged._records)
        self._data.update(staged._data)

    def bind_checkpoint_source(
        self,
        revision_id: str,
        source_file: str,
        source_path: str,
        *,
        provider: Callable[[], xr.DataArray] | None = None,
    ) -> None:
        record = self._records[revision_id]
        if record.checkpoint is None:
            raise ValueError("Cannot bind a revision without a data checkpoint")
        self._records[revision_id] = replace(
            record,
            checkpoint=replace(
                record.checkpoint,
                provider=record.checkpoint.provider if provider is None else provider,
                source_file=source_file,
                source_path=source_path,
            ),
        )
        if provider is not None:
            self._data.pop(revision_id, None)

    def discard_unreachable(self, revision_ids: Iterable[str]) -> None:
        retained = {record.revision_id for record in self.reachable(revision_ids)}
        self._records = {
            key: record for key, record in self._records.items() if key in retained
        }
        self._data = {key: data for key, data in self._data.items() if key in retained}

    def normalize_provenance(
        self,
        spec: ToolProvenanceSpec | None,
        *,
        replay_source_revision_id: str | None = None,
    ) -> ToolProvenanceSpec | None:
        if spec is None:
            return None
        if replay_source_revision_id is not None and not self.has_data(
            replay_source_revision_id
        ):
            raise ValueError("Legacy replay source revision has no checkpoint")
        inputs = []
        for item in spec.script_inputs:
            updates: dict[str, typing.Any] = {}
            nested = item.parsed_provenance_spec()
            if nested is not None:
                normalized = self.normalize_provenance(nested)
                updates["provenance_spec"] = (
                    None if normalized is None else normalized.model_dump(mode="json")
                )
            if item.node_uid is None and item.node_snapshot_token is not None:
                updates["node_snapshot_token"] = None
                updates["follow_source"] = False
                if item.revision_id is None and replay_source_revision_id is not None:
                    updates["revision_id"] = replay_source_revision_id
            inputs.append(
                item.model_copy(update=updates, deep=True) if updates else item
            )
        if replay_source_revision_id is not None and spec.kind in {
            "full_data",
            "public_data",
            "selection",
        }:
            if spec.is_live_source:
                steps = ReplayStep.from_source_spec(spec)
            else:
                source_start = ToolProvenanceSpec(kind=spec.kind)
                policy: typing.Literal["current", "restored"] = (
                    "current" if spec.kind == "full_data" else "restored"
                )
                steps = (
                    *ReplayStep.from_source_spec(source_start),
                    *(
                        ReplayStep(
                            operation=operation,
                            input_policy=policy if operation.live_applicable else None,
                            context_names=(
                                ("data", "derived", "parent_data")
                                if isinstance(operation, ScriptCodeOperation)
                                else ()
                            ),
                        )
                        for operation in spec.operations
                    ),
                )
            return ToolProvenanceSpec.model_validate(
                {
                    "kind": "script",
                    "start_label": "Saved",
                    "active_name": "derived" if steps else "data",
                    "primary_input": "data",
                    "script_inputs": (
                        ScriptInput(
                            name="data",
                            label="Saved",
                            revision_id=replay_source_revision_id,
                            follow_source=False,
                        ),
                    ),
                    "steps": steps,
                }
            )
        if (
            replay_source_revision_id is not None
            and spec.kind == "script"
            and not inputs
        ):
            inputs.extend(
                ScriptInput(
                    name=name,
                    label="Saved",
                    revision_id=replay_source_revision_id,
                    follow_source=False,
                )
                for name in script_replay_source_input_names(spec)
            )
        return spec.model_copy(update={"script_inputs": tuple(inputs)}, deep=True)
