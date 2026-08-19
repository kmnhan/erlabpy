"""Workspace state owned by the ImageTool manager."""

from __future__ import annotations

__all__ = [
    "_ManagerWorkspaceState",
    "_WorkspaceDirtyEvent",
    "_WorkspaceScriptState",
    "_WorkspaceStateSnapshot",
]

import contextlib
import copy
import hashlib
import typing
import uuid
from dataclasses import dataclass

from erlab.extensions._models import _script_name_key
from erlab.interactive.imagetool.manager._workspace._format import (
    _current_workspace_schema_version,
    _WorkspaceEmbeddedScriptEntry,
)

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Collection, Iterable, Iterator, Mapping

    from qtpy import QtCore

    from erlab.interactive.imagetool.manager._extensions._models import (
        _WorkspaceScriptRequirement,
    )


@dataclass(frozen=True)
class _WorkspaceDirtyEvent:
    generation: int
    uid: str | None = None
    data: bool = False
    state: bool = False
    added: bool = False
    removed: str | None = None
    structure: str | None = None
    layout: bool = False
    options: bool = False
    context: bool = False


@dataclass(frozen=True)
class _OpaqueManifestContainer:
    """One malformed or future manifest field preserved without interpretation."""

    value: typing.Any


class _WorkspaceScriptState:
    """Own extension-script state that belongs to one workspace document.

    Verified sources are recovery material only. They never participate in script
    resolution or execution. Opaque payloads and objects stay separate so Save As
    can preserve content that this ERLab version cannot validate.
    """

    def __init__(
        self,
        requirements: Iterable[_WorkspaceScriptRequirement] = (),
        *,
        verified_sources: Mapping[
            tuple[str, str], tuple[_WorkspaceEmbeddedScriptEntry, bytes]
        ]
        | None = None,
        opaque_requirement_payloads: Iterable[typing.Any] = (),
        opaque_source_payloads: Iterable[typing.Any] = (),
        opaque_objects: Mapping[str, tuple[bytes, str | None]] | None = None,
        explicit_sources: Collection[tuple[str, str]] = frozenset(),
        opaque_requirement_container: _OpaqueManifestContainer | None = None,
        opaque_source_container: _OpaqueManifestContainer | None = None,
    ) -> None:
        self.requirements = tuple(requirements)
        self.verified_sources = dict(verified_sources or {})
        self.opaque_requirement_payloads = tuple(
            copy.deepcopy(tuple(opaque_requirement_payloads))
        )
        self.opaque_source_payloads = tuple(
            copy.deepcopy(tuple(opaque_source_payloads))
        )
        self.opaque_objects = dict(opaque_objects or {})
        self.explicit_sources = set(explicit_sources)
        self.opaque_requirement_container = copy.deepcopy(opaque_requirement_container)
        self.opaque_source_container = copy.deepcopy(opaque_source_container)
        self._validate_script_names()
        self._validate_verified_sources()

    def _validate_script_names(self) -> None:
        """Reject ambiguous spellings of one portable script filename."""
        names: dict[str, str] = {}
        script_names = (
            *(requirement.script_name for requirement in self.requirements),
            *(entry.script_name for entry, _source in self.verified_sources.values()),
        )
        for script_name in script_names:
            key = _script_name_key(script_name)
            previous = names.setdefault(key, script_name)
            if previous != script_name:
                raise ValueError(
                    "workspace scripts have ambiguous filenames: "
                    f"{previous!r} and {script_name!r}"
                )

    def _validate_verified_sources(self) -> None:
        """Require each recovery entry and byte snapshot to have one identity."""
        for key, (entry, source) in self.verified_sources.items():
            if key != (entry.script_name, entry.source_hash):
                raise ValueError("embedded script source key does not match its entry")
            if hashlib.sha256(source).hexdigest() != entry.source_hash:
                raise ValueError("embedded script source does not match its hash")
            opaque = self.opaque_objects.get(entry.object_id)
            if opaque is not None and opaque != (
                source,
                "extension-python-source-v1",
            ):
                raise ValueError(
                    "verified script source conflicts with an opaque object"
                )
        if not self.explicit_sources.issubset(self.verified_sources):
            raise ValueError("explicit script sources must have verified bytes")

    def copy(self) -> _WorkspaceScriptState:
        """Return an independent copy for document rollback."""
        return type(self)(
            self.requirements,
            verified_sources=self.verified_sources,
            opaque_requirement_payloads=self.opaque_requirement_payloads,
            opaque_source_payloads=self.opaque_source_payloads,
            opaque_objects=self.opaque_objects,
            explicit_sources=self.explicit_sources,
            opaque_requirement_container=self.opaque_requirement_container,
            opaque_source_container=self.opaque_source_container,
        )

    def has_same_content(self, other: _WorkspaceScriptState) -> bool:
        """Return whether two document snapshots contain the same script state."""
        return (
            self.requirements == other.requirements
            and self.verified_sources == other.verified_sources
            and self.opaque_requirement_payloads == other.opaque_requirement_payloads
            and self.opaque_source_payloads == other.opaque_source_payloads
            and self.opaque_objects == other.opaque_objects
            and self.explicit_sources == other.explicit_sources
            and self.opaque_requirement_container == other.opaque_requirement_container
            and self.opaque_source_container == other.opaque_source_container
        )

    def replace(self, other: _WorkspaceScriptState) -> None:
        """Replace all document-owned script state from one validated snapshot."""
        replacement = other.copy()
        self.requirements = replacement.requirements
        self.verified_sources = replacement.verified_sources
        self.opaque_requirement_payloads = replacement.opaque_requirement_payloads
        self.opaque_source_payloads = replacement.opaque_source_payloads
        self.opaque_objects = replacement.opaque_objects
        self.explicit_sources = replacement.explicit_sources
        self.opaque_requirement_container = replacement.opaque_requirement_container
        self.opaque_source_container = replacement.opaque_source_container

    def clear(self) -> None:
        """Remove all script state from the current document."""
        self.replace(type(self)())

    def merge(self, other: _WorkspaceScriptState) -> None:
        """Merge imported script state without rewriting opaque content."""
        self._require_mergeable_containers(other)
        verified_sources = dict(self.verified_sources)
        for key, incoming in other.verified_sources.items():
            existing = verified_sources.get(key)
            if existing is not None and existing != incoming:
                raise ValueError(f"conflicting embedded script source for {key!r}")
            verified_sources[key] = incoming

        opaque_objects = dict(self.opaque_objects)
        for object_id, incoming in other.opaque_objects.items():
            existing = opaque_objects.get(object_id)
            if existing is not None and existing != incoming:
                raise ValueError(f"conflicting opaque workspace object {object_id!r}")
            opaque_objects[object_id] = incoming

        for entry, source in verified_sources.values():
            opaque = opaque_objects.get(entry.object_id)
            if opaque is not None and opaque != (
                source,
                "extension-python-source-v1",
            ):
                raise ValueError(
                    "opaque workspace object conflicts with a verified script source"
                )

        replacement = type(self)(
            (*self.requirements, *other.requirements),
            verified_sources=verified_sources,
            opaque_requirement_payloads=(
                *self.opaque_requirement_payloads,
                *other.opaque_requirement_payloads,
            ),
            opaque_source_payloads=(
                *self.opaque_source_payloads,
                *other.opaque_source_payloads,
            ),
            opaque_objects=opaque_objects,
            explicit_sources=self.explicit_sources | other.explicit_sources,
            opaque_requirement_container=(
                self.opaque_requirement_container
                if self.opaque_requirement_container is not None
                else other.opaque_requirement_container
            ),
            opaque_source_container=(
                self.opaque_source_container
                if self.opaque_source_container is not None
                else other.opaque_source_container
            ),
        )
        self.replace(replacement)

    def _require_mergeable_containers(self, other: _WorkspaceScriptState) -> None:
        """Reject imports whose malformed containers cannot be combined exactly."""
        merge_states = (
            (
                "requirement",
                self.opaque_requirement_container,
                other.opaque_requirement_container,
                (*self.requirements, *self.opaque_requirement_payloads),
                (*other.requirements, *other.opaque_requirement_payloads),
            ),
            (
                "embedded source",
                self.opaque_source_container,
                other.opaque_source_container,
                (*self.verified_sources, *self.opaque_source_payloads),
                (*other.verified_sources, *other.opaque_source_payloads),
            ),
        )
        for (
            label,
            current_container,
            incoming_container,
            current_items,
            incoming_items,
        ) in merge_states:
            if current_container is None and incoming_container is None:
                continue
            if (
                current_items
                or incoming_items
                or (
                    current_container is not None
                    and incoming_container is not None
                    and current_container != incoming_container
                )
            ):
                raise ValueError(
                    f"cannot merge an opaque workspace {label} container exactly"
                )

    def rebase_nodes(self, uid_map: Mapping[str, str]) -> None:
        """Rebase all validated references before incoming nodes are restored."""
        if not uid_map:
            return
        self.requirements = tuple(
            requirement.model_copy(
                update={
                    "referencing_nodes": tuple(
                        uid_map.get(uid, uid) for uid in requirement.referencing_nodes
                    )
                }
            )
            for requirement in self.requirements
        )

    def remove_node_references(self, node_uids: Iterable[str]) -> None:
        """Remove only references for nodes that the user explicitly deleted."""
        removed = set(node_uids)
        if not removed:
            return
        remaining_requirements: list[_WorkspaceScriptRequirement] = []
        for requirement in self.requirements:
            if not requirement.referencing_nodes:
                remaining_requirements.append(requirement)
                continue
            remaining = tuple(
                uid for uid in requirement.referencing_nodes if uid not in removed
            )
            if not remaining:
                continue
            remaining_requirements.append(
                requirement.model_copy(update={"referencing_nodes": remaining})
            )
        self.requirements = tuple(remaining_requirements)

    def remap_script(
        self,
        previous_name: str,
        source_hash: str,
        new_name: str,
    ) -> None:
        """Atomically change one script filename across validated state."""
        previous_key = _script_name_key(previous_name)
        new_key = _script_name_key(new_name)
        verified_sources: dict[
            tuple[str, str], tuple[_WorkspaceEmbeddedScriptEntry, bytes]
        ] = {}
        remapped_source_keys: set[tuple[str, str]] = set()
        for key, (entry, source) in self.verified_sources.items():
            script_name, key_hash = key
            matches_source = (
                key_hash == source_hash
                and _script_name_key(script_name) == previous_key
            )
            if (
                not matches_source
                and key_hash == source_hash
                and previous_key != new_key
                and _script_name_key(script_name) == new_key
            ):
                raise ValueError(f"embedded script source already uses {new_name!r}")
            if not matches_source:
                verified_sources[key] = (entry, source)
                continue
            remapped_key = (new_name, source_hash)
            remapped_entry = _WorkspaceEmbeddedScriptEntry(
                script_name=new_name,
                source_hash=source_hash,
                object_id=entry.object_id,
            )
            existing = verified_sources.get(remapped_key)
            candidate = (remapped_entry, source)
            if existing is not None and existing != candidate:
                raise ValueError(f"conflicting embedded script source for {new_name!r}")
            verified_sources[remapped_key] = candidate
            remapped_source_keys.add(key)

        requirements = tuple(
            requirement.model_copy(update={"script_name": new_name})
            if requirement.source_hash == source_hash
            and _script_name_key(requirement.script_name) == previous_key
            else requirement
            for requirement in self.requirements
        )
        explicit_sources = {
            (new_name, source_hash) if key in remapped_source_keys else key
            for key in self.explicit_sources
        }
        replacement = type(self)(
            requirements,
            verified_sources=verified_sources,
            opaque_requirement_payloads=self.opaque_requirement_payloads,
            opaque_source_payloads=self.opaque_source_payloads,
            opaque_objects=self.opaque_objects,
            explicit_sources=explicit_sources,
            opaque_requirement_container=self.opaque_requirement_container,
            opaque_source_container=self.opaque_source_container,
        )
        self.replace(replacement)

    def remember_verified_source(
        self,
        script_name: str,
        source_hash: str,
        source: bytes,
        *,
        explicit: bool = False,
    ) -> _WorkspaceEmbeddedScriptEntry | None:
        """Retain exact local bytes when they do not conflict with opaque content.

        A conflicting opaque object remains authoritative. In that case, this
        method does not retain the new source and returns ``None``. This makes
        result publication independent of malformed recovery content.
        """
        entry = _WorkspaceEmbeddedScriptEntry(
            script_name=script_name,
            source_hash=source_hash,
            object_id=f"extension-source-{source_hash}",
        )
        if hashlib.sha256(source).hexdigest() != source_hash:
            raise ValueError("embedded script source does not match its hash")
        script_key = _script_name_key(script_name)
        known_names = (
            *(requirement.script_name for requirement in self.requirements),
            *(
                stored_entry.script_name
                for stored_entry, _stored_source in self.verified_sources.values()
            ),
        )
        if any(
            _script_name_key(known_name) == script_key and known_name != script_name
            for known_name in known_names
        ):
            return None
        existing = self.verified_sources.get((script_name, source_hash))
        candidate = (entry, bytes(source))
        if existing is not None and existing != candidate:
            raise ValueError("embedded script source conflicts with retained bytes")
        opaque = self.opaque_objects.get(entry.object_id)
        if opaque is not None and opaque != (source, "extension-python-source-v1"):
            return None
        self.verified_sources[(script_name, source_hash)] = candidate
        if explicit:
            self.explicit_sources.add((script_name, source_hash))
        return entry

    def requirement_manifest_value(
        self, requirements: Iterable[_WorkspaceScriptRequirement]
    ) -> typing.Any:
        """Combine current validated dependencies with exact opaque payloads."""
        validated = tuple(requirements)
        if self.opaque_requirement_container is not None:
            return copy.deepcopy(self.opaque_requirement_container.value)
        return [
            *(item.model_dump(mode="json") for item in validated),
            *copy.deepcopy(self.opaque_requirement_payloads),
        ]

    def source_manifest_value(
        self, required_sources: Collection[tuple[str, str]]
    ) -> typing.Any:
        """Serialize reachable verified sources and all opaque source payloads."""
        included_sources = set(required_sources) | self.explicit_sources
        verified = tuple(
            entry
            for key, (entry, _source) in self.verified_sources.items()
            if key in included_sources
        )
        if self.opaque_source_container is not None:
            return copy.deepcopy(self.opaque_source_container.value)
        return [
            *(item.model_dump(mode="json") for item in verified),
            *copy.deepcopy(self.opaque_source_payloads),
        ]

    @property
    def has_opaque_content(self) -> bool:
        """Return whether this document contains unvalidated extension content."""
        return bool(
            self.opaque_requirement_payloads
            or self.opaque_source_payloads
            or self.opaque_objects
            or self.opaque_requirement_container is not None
            or self.opaque_source_container is not None
        )

    @property
    def has_content(self) -> bool:
        """Return whether this document owns any extension-script state."""
        return bool(
            self.requirements
            or self.verified_sources
            or self.explicit_sources
            or self.has_opaque_content
        )

    @property
    def has_opaque_requirements(self) -> bool:
        """Return whether unknown requirement data can contain node references."""
        return bool(
            self.opaque_requirement_payloads
            or self.opaque_requirement_container is not None
        )


class _WorkspaceStateSnapshot(typing.TypedDict):
    path: pathlib.Path | None
    document_id: str
    link_id: str
    node_uid_counter: int
    structure_modified: bool
    dirty_added: frozenset[str]
    dirty_data: frozenset[str]
    dirty_state: frozenset[str]
    dirty_removed: tuple[str, ...]
    structure_reasons: tuple[str, ...]
    layout_modified: bool
    options_modified: bool
    option_overrides: dict[str, typing.Any]
    context_modified: bool
    acquisition_context: dict[str, typing.Any]
    metadata_editor_layout: dict[str, typing.Any]
    dirty_generation: int
    dirty_events: tuple[_WorkspaceDirtyEvent, ...]
    schema_version: int
    save_as_only: bool
    degraded_reasons: tuple[str, ...]
    extension_scripts: _WorkspaceScriptState


class _ManagerWorkspaceState:
    """Own mutable workspace bookkeeping without Qt/UI side effects."""

    def __init__(self) -> None:
        self.path: pathlib.Path | None = None
        self.document_id: str = uuid.uuid4().hex
        self.link_id: str = uuid.uuid4().hex
        self.loading_depth: int = 0
        self.saving_depth: int = 0
        self.structure_modified: bool = False
        self.dirty_added: set[str] = set()
        self.dirty_data: set[str] = set()
        self.dirty_state: set[str] = set()
        self.dirty_removed: list[str] = []
        self.structure_reasons: list[str] = []
        self.layout_modified: bool = False
        self.options_modified: bool = False
        self.option_overrides: dict[str, typing.Any] = {}
        self.context_modified: bool = False
        self.acquisition_context: dict[str, typing.Any] = {}
        self.metadata_editor_layout: dict[str, typing.Any] = {}
        self.dirty_generation: int = 0
        self.dirty_events: list[_WorkspaceDirtyEvent] = []
        self.save_in_progress: bool = False
        self.schema_version: int = _current_workspace_schema_version()
        self.lock: QtCore.QLockFile | None = None
        self.closing_document: bool = False
        self.save_as_only: bool = False
        self.degraded_reasons: tuple[str, ...] = ()
        self.extension_scripts = _WorkspaceScriptState()

    def is_modified(self, *, has_nodes: bool) -> bool:
        if (
            self.path is None
            and not has_nodes
            and not self.context_modified
            and not self.extension_scripts.has_content
        ):
            return False
        return (
            self.structure_modified
            or self.layout_modified
            or bool(self.dirty_added)
            or bool(self.dirty_data)
            or bool(self.dirty_state)
            or bool(self.dirty_removed)
            or self.options_modified
            or self.context_modified
        )

    def apply_dirty_event(self, event: _WorkspaceDirtyEvent) -> bool:
        dirty_changed = False
        if event.uid is not None:
            already_added = event.uid in self.dirty_added
            already_data = event.uid in self.dirty_data
            already_state = event.uid in self.dirty_state
            if event.added and not already_added:
                self.dirty_added.add(event.uid)
                dirty_changed = True
            elif event.data and not (already_added or already_data):
                self.dirty_data.add(event.uid)
                dirty_changed = True
            elif event.state and not (already_added or already_state):
                self.dirty_state.add(event.uid)
                dirty_changed = True
        if event.removed is not None:
            self.dirty_removed.append(event.removed)
            self.structure_modified = True
            dirty_changed = True
        if event.structure is not None:
            self.structure_reasons.append(event.structure)
            self.structure_modified = True
            dirty_changed = True
        if event.layout and not self.layout_modified:
            self.layout_modified = True
            dirty_changed = True
        if event.options and not self.options_modified:
            self.options_modified = True
            dirty_changed = True
        if event.context and not self.context_modified:
            self.context_modified = True
            dirty_changed = True
        return dirty_changed

    def mark_dirty(self, event: _WorkspaceDirtyEvent) -> bool:
        dirty_changed = self.apply_dirty_event(event)
        if dirty_changed or (
            self.save_in_progress
            and (
                event.uid is not None
                or event.removed is not None
                or event.structure is not None
                or event.layout
                or event.options
                or event.context
            )
        ):
            self.dirty_generation = event.generation
            self.dirty_events.append(event)
            return True
        return False

    def mark_layout_dirty(self) -> bool:
        return self.mark_dirty(
            _WorkspaceDirtyEvent(
                generation=self.dirty_generation + 1,
                layout=True,
            )
        )

    def mark_options_dirty(self) -> bool:
        return self.mark_dirty(
            _WorkspaceDirtyEvent(
                generation=self.dirty_generation + 1,
                options=True,
            )
        )

    def mark_context_dirty(self) -> bool:
        return self.mark_dirty(
            _WorkspaceDirtyEvent(
                generation=self.dirty_generation + 1,
                context=True,
            )
        )

    def mark_clean(self) -> None:
        self.structure_modified = False
        self.layout_modified = False
        self.options_modified = False
        self.context_modified = False
        self.dirty_added.clear()
        self.dirty_data.clear()
        self.dirty_state.clear()
        self.dirty_removed.clear()
        self.structure_reasons.clear()
        self.dirty_events.clear()

    def advance_document_identity(self) -> None:
        self.document_id = uuid.uuid4().hex

    @contextlib.contextmanager
    def load_context(self) -> Iterator[None]:
        self.loading_depth += 1
        try:
            yield
        finally:
            self.loading_depth -= 1

    def snapshot(self, *, node_uid_counter: int) -> _WorkspaceStateSnapshot:
        return {
            "path": self.path,
            "document_id": self.document_id,
            "link_id": self.link_id,
            "node_uid_counter": node_uid_counter,
            "structure_modified": self.structure_modified,
            "dirty_added": frozenset(self.dirty_added),
            "dirty_data": frozenset(self.dirty_data),
            "dirty_state": frozenset(self.dirty_state),
            "dirty_removed": tuple(self.dirty_removed),
            "structure_reasons": tuple(self.structure_reasons),
            "layout_modified": self.layout_modified,
            "options_modified": self.options_modified,
            "option_overrides": dict(self.option_overrides),
            "context_modified": self.context_modified,
            "acquisition_context": copy.deepcopy(self.acquisition_context),
            "metadata_editor_layout": copy.deepcopy(self.metadata_editor_layout),
            "dirty_generation": self.dirty_generation,
            "dirty_events": tuple(self.dirty_events),
            "schema_version": self.schema_version,
            "save_as_only": self.save_as_only,
            "degraded_reasons": self.degraded_reasons,
            "extension_scripts": self.extension_scripts.copy(),
        }

    def restore(self, snapshot: _WorkspaceStateSnapshot) -> set[str]:
        self.path = snapshot["path"]
        self.document_id = snapshot["document_id"]
        self.link_id = snapshot["link_id"]
        self.structure_modified = snapshot["structure_modified"]
        self.dirty_added = set(snapshot["dirty_added"])
        self.dirty_data = set(snapshot["dirty_data"])
        self.dirty_state = set(snapshot["dirty_state"])
        self.dirty_removed = list(snapshot["dirty_removed"])
        self.structure_reasons = list(snapshot["structure_reasons"])
        self.layout_modified = snapshot["layout_modified"]
        self.options_modified = snapshot["options_modified"]
        self.option_overrides = dict(snapshot["option_overrides"])
        self.context_modified = snapshot["context_modified"]
        self.acquisition_context = copy.deepcopy(snapshot["acquisition_context"])
        self.metadata_editor_layout = copy.deepcopy(snapshot["metadata_editor_layout"])
        self.dirty_generation = snapshot["dirty_generation"]
        self.dirty_events = list(snapshot["dirty_events"])
        self.schema_version = snapshot["schema_version"]
        self.save_as_only = snapshot["save_as_only"]
        self.degraded_reasons = snapshot["degraded_reasons"]
        self.extension_scripts.replace(snapshot["extension_scripts"])
        return self.dirty_added | self.dirty_data | self.dirty_state
