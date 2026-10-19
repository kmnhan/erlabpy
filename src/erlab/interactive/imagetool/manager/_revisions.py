"""Immutable data and recipe revisions independent of managed windows."""

from __future__ import annotations

import copy
import json
import typing
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace

import xarray as xr

from erlab.interactive.imagetool._provenance._model import (
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    _normalize_legacy_script_input,
    parse_tool_provenance_spec,
)
from erlab.interactive.imagetool._provenance._operations import ScriptCodeOperation
from erlab.interactive.imagetool._provenance._trust import (
    remap_script_input_provenance,
    script_replay_source_input_names,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping


@dataclass(frozen=True)
class _RevisionCheckpointSource:
    file_path: str
    group_path: str
    reader: Callable[[str, str], xr.DataArray]
    ownership: object | None = None

    def read_data(self) -> xr.DataArray:
        return self.reader(self.file_path, self.group_path)


@dataclass(frozen=True)
class _RevisionCheckpoint:
    source: _RevisionCheckpointSource | None = None


@dataclass(frozen=True)
class _ProvenanceRevision:
    revision_id: str
    provenance_spec: ToolProvenanceSpec | None
    primary_revision_id: str | None = None
    source_owner_uid: str | None = None
    checkpoint: _RevisionCheckpoint | None = None

    @property
    def checkpoint_object_id(self) -> str:
        return f"revision-{self.revision_id}"


@dataclass(frozen=True)
class _BoundProvenanceRevision:
    """A shared checkpoint with the recipe bound to one owning reference."""

    revision: _ProvenanceRevision
    provenance_spec: ToolProvenanceSpec | None

    @property
    def revision_id(self) -> str:
        return self.revision.revision_id


def _spec_revision_ids(spec: ToolProvenanceSpec | None) -> frozenset[str]:
    return (
        frozenset() if spec is None else _script_input_revision_ids(spec.replay_inputs)
    )


def _remap_provenance_revision_ids(
    spec: ToolProvenanceSpec, revision_ids: Mapping[str, str]
) -> ToolProvenanceSpec:
    """Retarget captured inputs without changing their recipes or live bindings."""
    if not revision_ids:
        return spec
    inputs = tuple(
        remap_script_input_provenance(
            item,
            lambda nested: _remap_provenance_revision_ids(nested, revision_ids),
            revision_ids=revision_ids,
        )
        for item in spec.replay_inputs
    )
    return (
        spec
        if all(
            item is previous
            for item, previous in zip(inputs, spec.replay_inputs, strict=True)
        )
        else spec.with_replay_inputs(inputs)
    )


def _same_bound_provenance(
    left: ToolProvenanceSpec | None, right: ToolProvenanceSpec | None
) -> bool:
    """Compare recipes while allowing captured inputs to bind to other nodes."""

    def normalized(spec: ToolProvenanceSpec | None) -> object:
        if spec is None:
            return None
        payload = spec.model_dump(mode="json")
        input_payloads = payload["script_inputs"]
        if spec.root_input is not None:
            input_payloads = [payload["root_input"], *input_payloads]
        for item, item_payload in zip(spec.replay_inputs, input_payloads, strict=True):
            item_payload["node_uid"] = None
            item_payload["provenance_spec"] = normalized(item.parsed_provenance_spec())
            item_payload["source_spec"] = normalized(item.parsed_source_spec())
        return payload

    return json.dumps(normalized(left), sort_keys=True) == json.dumps(
        normalized(right), sort_keys=True
    )


def requires_legacy_replay_source(spec: ToolProvenanceSpec | None) -> bool:
    if spec is None:
        return False
    if spec.kind in {"full_data", "public_data", "selection"}:
        return True
    if spec.kind != "script":
        return False
    if spec.replay_inputs:
        return any(
            item.node_uid is None
            and item.node_snapshot_token is not None
            and item.revision_id is None
            for item in spec.replay_inputs
        )
    return bool(script_replay_source_input_names(spec))


def _script_input_revision_ids(inputs: Iterable[ScriptInput]) -> frozenset[str]:
    ids: set[str] = set()
    pending = list(inputs)
    while pending:
        item = pending.pop()
        if item.revision_id is not None:
            ids.add(item.revision_id)
        if item.resolved_revision_id is not None:
            ids.add(item.resolved_revision_id)
        provenance = item.parsed_provenance_spec()
        if provenance is not None:
            pending.extend(provenance.replay_inputs)
    return frozenset(ids)


class _ProvenanceRevisionStore:
    def __init__(self) -> None:
        self._records: dict[str, _ProvenanceRevision] = {}
        self._data: dict[str, xr.DataArray] = {}
        self._retained_roots: dict[object, set[str]] = {}

    def __contains__(self, revision_id: object) -> bool:
        return revision_id in self._records

    def add(
        self,
        data: xr.DataArray | None = None,
        spec: ToolProvenanceSpec | None = None,
        *,
        primary_revision_id: str | None = None,
        source_owner_uid: str | None = None,
        revision_id: str | None = None,
        source: _RevisionCheckpointSource | None = None,
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
        if data is not None and source is not None:
            raise ValueError("Supply revision data or a checkpoint source, not both")
        if data is not None and not isinstance(data, xr.DataArray):
            raise TypeError("Revision data must be a DataArray")
        parsed_spec = parse_tool_provenance_spec(spec)
        parsed_spec = self.normalize_provenance(parsed_spec)
        owned_spec = None if parsed_spec is None else parsed_spec.model_copy(deep=True)
        checkpoint = None if source is None else _RevisionCheckpoint(source)
        protected = None
        if data is not None:
            # A deep copy alone preserves graphs that can read mutable external data.
            protected = data.copy(deep=True).load()
            checkpoint = _RevisionCheckpoint()
        record = _ProvenanceRevision(
            revision_id=revision_id,
            provenance_spec=owned_spec,
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

    @property
    def retained_roots(self) -> frozenset[str]:
        return frozenset().union(*self._retained_roots.values())

    @contextmanager
    def retain(self, revision_ids: Iterable[str]) -> Iterator[set[str]]:
        token = object()
        roots = set(revision_ids)
        self._retained_roots[token] = roots
        try:
            yield roots
        finally:
            del self._retained_roots[token]

    def get(self, revision_id: str) -> _ProvenanceRevision | None:
        record = self._records.get(revision_id)
        if record is None:
            return None
        return replace(record, provenance_spec=copy.deepcopy(record.provenance_spec))

    @contextmanager
    def normalize_provenance_revisions(
        self,
        remap: Callable[[ToolProvenanceSpec], ToolProvenanceSpec],
        *,
        roots: Iterable[str],
    ) -> Iterator[Mapping[str, str]]:
        """Publish normalized revisions and their dependents under new identities."""
        records = {
            record.revision_id: self._records[record.revision_id]
            for record in self.reachable(roots)
        }
        normalized = {
            revision_id: (
                None
                if record.provenance_spec is None
                else remap(record.provenance_spec)
            )
            for revision_id, record in records.items()
        }
        changed = {
            revision_id
            for revision_id, spec in normalized.items()
            if spec != records[revision_id].provenance_spec
        }
        dependents: dict[str, set[str]] = {}
        for revision_id, record in records.items():
            dependencies = set(_spec_revision_ids(normalized[revision_id]))
            if record.primary_revision_id is not None:
                dependencies.add(record.primary_revision_id)
            for dependency in dependencies:
                dependents.setdefault(dependency, set()).add(revision_id)
        pending = list(changed)
        while pending:
            for revision_id in dependents.get(pending.pop(), ()):
                if revision_id not in changed:
                    changed.add(revision_id)
                    pending.append(revision_id)
        revision_ids = {revision_id: uuid.uuid4().hex for revision_id in changed}
        staged = self.stage()
        for previous_id, revision_id in revision_ids.items():
            previous = self._records[previous_id]
            spec = normalized[previous_id]
            staged._records[revision_id] = replace(
                previous,
                revision_id=revision_id,
                provenance_spec=(
                    None
                    if spec is None
                    else _remap_provenance_revision_ids(spec, revision_ids)
                ),
                primary_revision_id=(
                    None
                    if previous.primary_revision_id is None
                    else revision_ids.get(
                        previous.primary_revision_id, previous.primary_revision_id
                    )
                ),
            )
            if previous_id in self._data:
                staged._data[revision_id] = self._data[previous_id]
        with self.retain((*revision_ids, *revision_ids.values())):
            self.merge(staged)
            try:
                yield revision_ids
            except Exception:
                for revision_id in staged.revision_ids:
                    self._records.pop(revision_id, None)
                    self._data.pop(revision_id, None)
                raise

    def bind(
        self, revision_id: str, provenance_spec: ToolProvenanceSpec | None
    ) -> _BoundProvenanceRevision:
        """Resolve a checkpoint without replacing its owner's input bindings."""
        record = self.get(revision_id)
        if record is None:
            raise KeyError(revision_id)
        if not _same_bound_provenance(record.provenance_spec, provenance_spec):
            raise ValueError("The bound recipe does not match the captured revision")
        return _BoundProvenanceRevision(record, copy.deepcopy(provenance_spec))

    def normalize_input(self, item: ScriptInput) -> ScriptInput:
        """Normalize a released input and its nested fallback."""
        item = _normalize_legacy_script_input(item)
        normalized = self.normalize_provenance(item.parsed_provenance_spec())
        return item.model_copy(
            update={
                "provenance_spec": (
                    None if normalized is None else normalized.model_dump(mode="json")
                ),
            },
            deep=True,
        )

    def has_data(self, revision_id: str) -> bool:
        record = self._records.get(revision_id)
        return record is not None and record.checkpoint is not None

    def resolve_data(self, revision_id: str) -> xr.DataArray:
        record = self._records[revision_id]
        if record.checkpoint is None:
            raise ValueError(f"Revision {revision_id!r} has no data checkpoint")
        if revision_id not in self._data:
            source = typing.cast("_RevisionCheckpointSource", record.checkpoint.source)
            data = source.read_data()
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
        source: _RevisionCheckpointSource,
    ) -> None:
        record = self._records[revision_id]
        if record.checkpoint is None:
            raise ValueError("Cannot bind a revision without a data checkpoint")
        self._records[revision_id] = replace(
            record,
            checkpoint=_RevisionCheckpoint(source),
        )
        self._data.pop(revision_id, None)

    def discard_unreachable(self, revision_ids: Iterable[str]) -> None:
        retained = {
            record.revision_id
            for record in self.reachable(set(revision_ids) | self.retained_roots)
        }
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
        inputs = []
        for item in spec.replay_inputs:
            normalized = self.normalize_input(item)
            if item.node_uid is None and item.node_snapshot_token is not None:
                normalized = normalized.model_copy(
                    update={
                        "node_snapshot_token": None,
                        "follow_source": False,
                        "revision_id": normalized.revision_id
                        or replay_source_revision_id,
                    }
                )
            inputs.append(normalized)
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
                    "root_input": ScriptInput(
                        name="data",
                        label="Saved",
                        revision_id=replay_source_revision_id,
                        follow_source=False,
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
        return spec.with_replay_inputs(inputs)
