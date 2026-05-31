"""Workspace state owned by the ImageTool manager."""

from __future__ import annotations

__all__ = ["_ManagerWorkspaceState", "_WorkspaceStateSnapshot"]

import contextlib
import typing
import uuid

from erlab.interactive.imagetool.manager import _workspace as _manager_workspace

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterator

    from qtpy import QtCore


class _WorkspaceStateSnapshot(typing.TypedDict):
    path: pathlib.Path | None
    link_id: str
    needs_full_save: bool
    node_uid_counter: int
    structure_modified: bool
    dirty_added: frozenset[str]
    dirty_data: frozenset[str]
    dirty_state: frozenset[str]
    dirty_removed: tuple[str, ...]
    structure_reasons: tuple[str, ...]
    dirty_generation: int
    dirty_events: tuple[_manager_workspace._WorkspaceDirtyEvent, ...]
    delta_save_count: int
    schema_version: int


class _ManagerWorkspaceState:
    """Own mutable workspace bookkeeping without Qt/UI side effects."""

    def __init__(self) -> None:
        self.path: pathlib.Path | None = None
        self.link_id: str = uuid.uuid4().hex
        self.loading_depth: int = 0
        self.saving_depth: int = 0
        self.structure_modified: bool = False
        self.dirty_added: set[str] = set()
        self.dirty_data: set[str] = set()
        self.dirty_state: set[str] = set()
        self.dirty_removed: list[str] = []
        self.structure_reasons: list[str] = []
        self.needs_full_save: bool = False
        self.dirty_generation: int = 0
        self.dirty_events: list[_manager_workspace._WorkspaceDirtyEvent] = []
        self.save_in_progress: bool = False
        self.delta_save_count: int = 0
        self.schema_version: int = (
            _manager_workspace._current_workspace_schema_version()
        )
        self.lock: QtCore.QLockFile | None = None
        self.closing_document: bool = False

    def is_modified(self, *, has_nodes: bool) -> bool:
        if self.path is None and not has_nodes:
            return False
        return (
            self.structure_modified
            or bool(self.dirty_added)
            or bool(self.dirty_data)
            or bool(self.dirty_state)
            or bool(self.dirty_removed)
        )

    def apply_dirty_event(self, event: _manager_workspace._WorkspaceDirtyEvent) -> bool:
        dirty_changed = False
        if event.uid is not None:
            if event.added:
                self.dirty_added.add(event.uid)
            elif event.data:
                self.dirty_data.add(event.uid)
            elif event.state:
                self.dirty_state.add(event.uid)
            if event.added or event.data or event.state:
                dirty_changed = True
        if event.removed is not None:
            self.dirty_removed.append(event.removed)
            self.structure_modified = True
            dirty_changed = True
        if event.structure is not None:
            self.structure_reasons.append(event.structure)
            self.structure_modified = True
            dirty_changed = True
        return dirty_changed

    def mark_dirty(self, event: _manager_workspace._WorkspaceDirtyEvent) -> bool:
        if self.apply_dirty_event(event):
            self.dirty_generation = event.generation
            self.dirty_events.append(event)
            return True
        return False

    def mark_clean(self) -> None:
        self.structure_modified = False
        self.dirty_added.clear()
        self.dirty_data.clear()
        self.dirty_state.clear()
        self.dirty_removed.clear()
        self.structure_reasons.clear()
        self.dirty_events.clear()

    def restore_dirty_events(
        self, events: list[_manager_workspace._WorkspaceDirtyEvent]
    ) -> None:
        self.mark_clean()
        for event in events:
            self.apply_dirty_event(event)
        self.dirty_events = events

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
            "link_id": self.link_id,
            "needs_full_save": self.needs_full_save,
            "node_uid_counter": node_uid_counter,
            "structure_modified": self.structure_modified,
            "dirty_added": frozenset(self.dirty_added),
            "dirty_data": frozenset(self.dirty_data),
            "dirty_state": frozenset(self.dirty_state),
            "dirty_removed": tuple(self.dirty_removed),
            "structure_reasons": tuple(self.structure_reasons),
            "dirty_generation": self.dirty_generation,
            "dirty_events": tuple(self.dirty_events),
            "delta_save_count": self.delta_save_count,
            "schema_version": self.schema_version,
        }

    def restore(self, snapshot: _WorkspaceStateSnapshot) -> set[str]:
        self.path = snapshot["path"]
        self.link_id = snapshot["link_id"]
        self.needs_full_save = snapshot["needs_full_save"]
        self.structure_modified = snapshot["structure_modified"]
        self.dirty_added = set(snapshot["dirty_added"])
        self.dirty_data = set(snapshot["dirty_data"])
        self.dirty_state = set(snapshot["dirty_state"])
        self.dirty_removed = list(snapshot["dirty_removed"])
        self.structure_reasons = list(snapshot["structure_reasons"])
        self.dirty_generation = snapshot["dirty_generation"]
        self.dirty_events = list(snapshot["dirty_events"])
        self.delta_save_count = snapshot["delta_save_count"]
        self.schema_version = snapshot["schema_version"]
        return self.dirty_added | self.dirty_data | self.dirty_state
