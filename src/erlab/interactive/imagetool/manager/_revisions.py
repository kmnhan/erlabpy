"""Immutable data and recipe revisions independent of managed windows."""

from __future__ import annotations

import copy
import json
import typing
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace

import xarray as xr

from erlab.interactive.imagetool._provenance._model import (
    ReplayStep,
    ScriptInput,
    ToolProvenanceSpec,
    _normalize_legacy_script_input,
    compose_full_provenance,
    parse_tool_provenance_spec,
)
from erlab.interactive.imagetool._provenance._operations import ScriptCodeOperation
from erlab.interactive.imagetool._provenance._trust import (
    script_replay_source_input_names,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


@dataclass(frozen=True)
class _RevisionCheckpointSource:
    file_path: str
    group_path: str
    reader: Callable[[str, str], xr.DataArray]
    ownership: object | None = None

    def __post_init__(self) -> None:
        if not self.file_path or not self.group_path:
            raise ValueError("Checkpoint source requires a file and a group path")
        if not callable(self.reader):
            raise TypeError("Checkpoint source reader must be callable")

    def read_data(self) -> xr.DataArray:
        return self.reader(self.file_path, self.group_path)


@dataclass(frozen=True)
class _RevisionCheckpoint:
    source: _RevisionCheckpointSource | None = None

    def __post_init__(self) -> None:
        if self.source is not None and not isinstance(
            self.source, _RevisionCheckpointSource
        ):
            raise TypeError("Revision source must be a checkpoint source")


@dataclass(frozen=True)
class _ProvenanceRevision:
    revision_id: str
    provenance_spec: ToolProvenanceSpec | None
    extra_input_revisions: tuple[tuple[str, str], ...] = ()
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


def _provenance_input_revisions(spec: ToolProvenanceSpec | None) -> dict[str, str]:
    return {
        item.name: item.revision_id
        for item in (() if spec is None else spec.replay_inputs)
        if item.revision_id is not None
    }


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
                        for field in ("revision_id", "resolved_revision_id"):
                            revision_id = item.get(field)
                            if isinstance(revision_id, str) and revision_id:
                                ids.add(revision_id)
            root = payload.get("root_input")
            if isinstance(root, Mapping):
                for field in ("revision_id", "resolved_revision_id"):
                    revision_id = root.get(field)
                    if isinstance(revision_id, str) and revision_id:
                        ids.add(revision_id)
            for value in payload.values():
                _visit(value)
        elif isinstance(payload, (list, tuple)):
            for value in payload:
                _visit(value)

    _visit(spec.model_dump(mode="python"))
    return frozenset(ids)


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
    for item in inputs:
        if item.revision_id is not None:
            ids.add(item.revision_id)
        if item.resolved_revision_id is not None:
            ids.add(item.resolved_revision_id)
        ids.update(_spec_revision_ids(item.parsed_provenance_spec()))
        ids.update(_spec_revision_ids(item.parsed_source_spec()))
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
        extra_input_revisions: Mapping[str, str] | Iterable[tuple[str, str]] = (),
        primary_revision_id: str | None = None,
        source_owner_uid: str | None = None,
        revision_id: str | None = None,
        source: _RevisionCheckpointSource | None = None,
        normalize: bool = True,
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
        if normalize:
            parsed_spec = self.normalize_provenance(parsed_spec)
        inputs = (
            tuple(extra_input_revisions.items())
            if isinstance(extra_input_revisions, Mapping)
            else tuple(extra_input_revisions)
        )
        if len({name for name, _ in inputs}) != len(inputs):
            raise ValueError("Revision input names must be unique")
        if any(
            not isinstance(name, str) or not name or not isinstance(ref, str) or not ref
            for name, ref in inputs
        ):
            raise ValueError("Revision inputs require a name and revision ID")
        owned_spec = None if parsed_spec is None else parsed_spec.model_copy(deep=True)
        ordinary_inputs = _provenance_input_revisions(owned_spec)
        checkpoint = None if source is None else _RevisionCheckpoint(source)
        protected = None
        if data is not None:
            # A deep copy alone preserves graphs that can read mutable external data.
            protected = data.copy(deep=True).load()
            checkpoint = _RevisionCheckpoint()
        record = _ProvenanceRevision(
            revision_id=revision_id,
            provenance_spec=owned_spec,
            extra_input_revisions=tuple(
                (name, ref) for name, ref in inputs if ordinary_inputs.get(name) != ref
            ),
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

    def _normalization_store(
        self, references: _ProvenanceRevisionStore | None
    ) -> _ProvenanceRevisionStore:
        result = _ProvenanceRevisionStore()
        if references is not None:
            result._records.update(references._records)
        result._records.update(self._records)
        return result

    def normalize_input(
        self,
        item: ScriptInput,
        *,
        references: _ProvenanceRevisionStore | None = None,
    ) -> ScriptInput:
        """Convert saved resolved fallbacks to the explicit raw-input contract."""
        if references is not None:
            return self._normalization_store(references).normalize_input(item)
        original = item
        bound = item.parsed_provenance_spec()
        source = item.parsed_source_spec()
        item = _normalize_legacy_script_input(item)
        if original.input_version == 1 and source is not None and bound is not None:
            # Saved wrappers and their already imported canonical forms use the
            # same boundary. Compare these forms without looking up references,
            # so cyclic record references cannot recurse during migration.
            syntax_normalizer = _ProvenanceRevisionStore()

            def same_recipe(
                left: ToolProvenanceSpec | None, right: ToolProvenanceSpec | None
            ) -> bool:
                return _same_bound_provenance(
                    syntax_normalizer.normalize_provenance(left),
                    syntax_normalizer.normalize_provenance(right),
                )

            record = self._records.get(original.revision_id or "")
            record_is_resolved = False
            if record is not None:
                identity_source = source.kind == "full_data" and not source.operations
                if not identity_source and same_recipe(record.provenance_spec, bound):
                    record_is_resolved = True
                    # The saved revision captured the selected result. Its raw
                    # boundary can be recovered from the exact source suffix.
                    item = item.model_copy(
                        update={
                            "revision_id": (
                                item.revision_id
                                if item.revision_id != original.revision_id
                                else None
                            ),
                            "resolved_revision_id": original.revision_id,
                        }
                    )
                else:
                    raw = record.provenance_spec
                    if (
                        raw is not None
                        and raw.kind == "script"
                        and tuple(value.name for value in raw.replay_inputs)
                        == tuple(value.name for value in bound.replay_inputs)
                    ):
                        # Workspace UID rebasing changes each owner's bindings,
                        # while the checkpoint retains its original recipe.
                        raw = raw.with_replay_inputs(bound.replay_inputs)
                    if same_recipe(compose_full_provenance(raw, source), bound):
                        item = item.model_copy(
                            update={
                                "fallback_role": "raw",
                                "revision_id": original.revision_id,
                                "resolved_revision_id": original.resolved_revision_id,
                                "provenance_spec": (
                                    None if raw is None else raw.model_dump(mode="json")
                                ),
                            }
                        )
                    elif item.resolved_revision_id == original.revision_id:
                        item = item.model_copy(
                            update={
                                "resolved_revision_id": original.resolved_revision_id
                            }
                        )
            raw_record = self._records.get(item.revision_id or "")
            if (
                item.revision_id != original.revision_id
                and raw_record is not None
                and not same_recipe(
                    raw_record.provenance_spec, item.parsed_provenance_spec()
                )
            ):
                # The old outer reference does not prove that the nested recipe
                # belongs to its checkpoint. Keep its complete resolved fallback.
                item = original.model_copy(
                    update={
                        "input_version": 2,
                        "fallback_role": "resolved",
                        "revision_id": None
                        if record_is_resolved
                        else original.revision_id,
                        "resolved_revision_id": original.revision_id
                        if record_is_resolved
                        else original.resolved_revision_id,
                    }
                )
        normalized = self.normalize_provenance(item.parsed_provenance_spec())
        return item.model_copy(
            update={
                "input_version": 2,
                "provenance_spec": (
                    None if normalized is None else normalized.model_dump(mode="json")
                ),
            },
            deep=True,
        )

    def normalize_records(
        self, *, references: _ProvenanceRevisionStore | None = None
    ) -> None:
        """Normalize recipes after all imported revision references are visible."""
        # Read the same original records for every migration. Mutation order must
        # not determine whether a reference denotes raw or selected data.
        normalizer = self._normalization_store(references)
        records = {}
        for revision_id, record in tuple(self._records.items()):
            spec = normalizer.normalize_provenance(record.provenance_spec)
            ordinary_inputs = _provenance_input_revisions(spec)
            records[revision_id] = replace(
                record,
                provenance_spec=spec,
                extra_input_revisions=tuple(
                    (name, ref)
                    for name, ref in record.extra_input_revisions
                    if ordinary_inputs.get(name) != ref
                ),
            )
        self._records = records

    def has_data(self, revision_id: str) -> bool:
        record = self._records.get(revision_id)
        return record is not None and record.checkpoint is not None

    def resolve_data(self, revision_id: str) -> xr.DataArray:
        record = self._records[revision_id]
        if record.checkpoint is None:
            raise ValueError(f"Revision {revision_id!r} has no data checkpoint")
        if revision_id not in self._data:
            if record.checkpoint.source is None:
                raise ValueError(f"Revision {revision_id!r} has no captured data")
            data = record.checkpoint.source.read_data()
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
            pending.extend(ref for _, ref in record.extra_input_revisions)
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
        references: _ProvenanceRevisionStore | None = None,
    ) -> ToolProvenanceSpec | None:
        if references is not None:
            return self._normalization_store(references).normalize_provenance(
                spec, replay_source_revision_id=replay_source_revision_id
            )
        if spec is None:
            return None
        if replay_source_revision_id is not None and not self.has_data(
            replay_source_revision_id
        ):
            raise ValueError("Legacy replay source revision has no checkpoint")
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
        normalized = spec.with_replay_inputs(inputs)
        if (
            normalized.kind == "script"
            and normalized.root_input is None
            and normalized.primary_input is not None
            and len(normalized.script_inputs) == 1
            and normalized.seed_code is None
            and normalized.start_label in {"Use saved input", "Use source input"}
        ):
            normalized = normalized.model_copy(
                update={
                    "root_input": normalized.script_inputs[0],
                    "primary_input": None,
                    "script_inputs": (),
                }
            )
        return normalized
