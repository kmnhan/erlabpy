"""Immutable data and recipe revisions independent of managed windows."""

from __future__ import annotations

import copy
import typing
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace

import xarray as xr

from erlab.interactive.imagetool._provenance._model import (
    ToolProvenanceSpec,
    parse_tool_provenance_spec,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@dataclass(frozen=True)
class _RevisionCheckpoint:
    provider: Callable[[], xr.DataArray]
    source_file: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class _ProvenanceRevision:
    revision_id: str
    provenance_spec: ToolProvenanceSpec | None
    input_revisions: tuple[tuple[str, str], ...] = ()
    primary_revision_id: str | None = None
    replay_source_revision_id: str | None = None
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
        replay_source_revision_id: str | None = None,
        revision_id: str | None = None,
        provider: Callable[[], xr.DataArray] | None = None,
        source_file: str | None = None,
        source_path: str | None = None,
        provenance_spec: ToolProvenanceSpec | None = None,
    ) -> str:
        revision_id = revision_id or uuid.uuid4().hex
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
        if (source_file is None) != (source_path is None):
            raise ValueError("Checkpoint source requires a file and a group path")
        if spec is not None and provenance_spec is not None:
            raise ValueError("Supply only one revision provenance spec")
        parsed_spec = parse_tool_provenance_spec(spec or provenance_spec)
        checkpoint = None
        if data is not None:
            self._data[revision_id] = data.copy(deep=True)

            def provider() -> xr.DataArray:
                return self._data[revision_id].copy(deep=True)

        if provider is not None:
            checkpoint = _RevisionCheckpoint(provider, source_file, source_path)
        elif source_file is not None:
            raise ValueError("Checkpoint source requires a lazy provider")
        inputs = (
            tuple(input_revisions.items())
            if isinstance(input_revisions, Mapping)
            else tuple(input_revisions)
        )
        if len({name for name, _ in inputs}) != len(inputs):
            raise ValueError("Revision input names must be unique")
        if any(not name or not ref for name, ref in inputs):
            raise ValueError("Revision inputs require a name and revision ID")
        self._records[revision_id] = _ProvenanceRevision(
            revision_id=revision_id,
            provenance_spec=(
                None if parsed_spec is None else parsed_spec.model_copy(deep=True)
            ),
            input_revisions=inputs,
            primary_revision_id=primary_revision_id,
            replay_source_revision_id=replay_source_revision_id,
            checkpoint=checkpoint,
        )
        return revision_id

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
                raise KeyError(f"Missing provenance revision {revision_id!r}")
            visited.add(revision_id)
            pending.extend(ref for _, ref in record.input_revisions)
            if record.primary_revision_id is not None:
                pending.append(record.primary_revision_id)
            if record.replay_source_revision_id is not None:
                pending.append(record.replay_source_revision_id)
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

    def discard_unreachable(self, revision_ids: Iterable[str]) -> None:
        retained = {record.revision_id for record in self.reachable(revision_ids)}
        self._records = {
            key: record for key, record in self._records.items() if key in retained
        }
        self._data = {key: data for key, data in self._data.items() if key in retained}

    def freeze_legacy_inputs(
        self,
        spec: ToolProvenanceSpec | None,
        data: xr.DataArray | None = None,
        *,
        provider: Callable[[], xr.DataArray] | None = None,
        replay_source_revision_id: str | None = None,
    ) -> ToolProvenanceSpec | None:
        if spec is None:
            return None
        inputs = tuple(
            item
            for item in spec.script_inputs
            if item.uses_owner_replay_source and item.revision_id is None
        )
        if not inputs:
            return spec
        if replay_source_revision_id is None:
            if data is None and provider is None:
                return spec
            replay_source_revision_id = self.add(data, provider=provider)
        elif not self.has_data(replay_source_revision_id):
            raise ValueError("Legacy replay source revision has no checkpoint")
        return spec.model_copy(
            update={
                "script_inputs": tuple(
                    item.model_copy(update={"revision_id": replay_source_revision_id})
                    if item in inputs
                    else item
                    for item in spec.script_inputs
                )
            },
            deep=True,
        )
