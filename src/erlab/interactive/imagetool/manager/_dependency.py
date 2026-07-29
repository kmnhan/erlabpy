"""Dependency bookkeeping for managed ImageTool nodes."""

from __future__ import annotations

from erlab.interactive.imagetool._provenance._model import (
    ScriptInputDependencyRef,
    script_input_dependency_refs,
)

__all__ = ["_DependencyStatus", "_ManagerDependencyTracker"]

import typing

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool.manager._tool_graph import _ManagerToolGraph

_DependencyStatus = typing.Literal["current", "changed", "missing"]


class _ManagerDependencyTracker:
    """Track script-input dependencies and deferred source refresh requests."""

    def __init__(self, graph: _ManagerToolGraph) -> None:
        self._graph = graph
        self._ref_cache: dict[
            str,
            tuple[
                int,
                tuple[
                    ScriptInputDependencyRef,
                    ...,
                ],
            ],
        ] = {}
        self._source_uids_by_dependent: dict[str, set[str]] = {}
        self._dependents_by_source_uid: dict[str, set[str]] = {}
        self._unindexed_uids: set[str] = set()
        self._status_cache: dict[str, _DependencyStatus | None] = {}
        self._pending_source_refresh_targets: dict[str, set[str]] = {}

    def note_uid(self, uid: str) -> None:
        self._unindexed_uids.add(uid)
        for dependent_uid in self._dependents_by_source_uid.get(uid, ()):
            self._status_cache.pop(dependent_uid, None)

    def invalidate_uid(self, uid: str) -> None:
        self._remove_reverse_refs(uid)
        self._ref_cache.pop(uid, None)
        self._status_cache.pop(uid, None)
        if uid in self._graph.nodes:
            self._unindexed_uids.add(uid)

    def refs_for_uid(self, uid: str) -> tuple[ScriptInputDependencyRef, ...]:
        node = self._graph.nodes.get(uid)
        if node is None:
            self.invalidate_uid(uid)
            return ()
        spec = (
            node.passive_displayed_provenance_spec
            if node.tool_window is not None
            else node.provenance_spec
        )
        if spec is None:
            self._remove_reverse_refs(uid)
            self._ref_cache.pop(uid, None)
            self._unindexed_uids.discard(uid)
            self._status_cache[uid] = None
            return ()
        spec_id = id(spec)
        cached = self._ref_cache.get(uid)
        if cached is not None and cached[0] == spec_id:
            self._unindexed_uids.discard(uid)
            return cached[1]
        refs = script_input_dependency_refs(spec)
        self._remove_reverse_refs(uid)
        source_uids = {ref.node_uid for ref in refs}
        self._source_uids_by_dependent[uid] = source_uids
        for source_uid in source_uids:
            self._dependents_by_source_uid.setdefault(source_uid, set()).add(uid)
        self._ref_cache[uid] = (spec_id, refs)
        self._unindexed_uids.discard(uid)
        self._status_cache.pop(uid, None)
        return refs

    def status_for_uid(self, uid: str) -> _DependencyStatus | None:
        if uid in self._status_cache:
            return self._status_cache[uid]
        refs = self.refs_for_uid(uid)
        if not refs:
            self._status_cache[uid] = None
            return None

        changed = False
        for ref in refs:
            parent = self._graph.nodes.get(ref.node_uid)
            if parent is None:
                self._status_cache[uid] = "missing"
                return "missing"
            if (
                ref.node_snapshot_token is not None
                and parent.snapshot_token_for_role(ref.data_role)
                != ref.node_snapshot_token
            ):
                changed = True
        status: _DependencyStatus = "changed" if changed else "current"
        self._status_cache[uid] = status
        return status

    def dependent_uids(self, uid: str) -> list[str]:
        self._index_pending_uids()
        dependents = self._dependents_by_source_uid.get(uid, set())
        for dependent_uid in dependents:
            self._status_cache.pop(dependent_uid, None)
        return [node_uid for node_uid in self._graph.nodes if node_uid in dependents]

    def clear_uid(self, uid: str) -> None:
        self.invalidate_uid(uid)
        self._unindexed_uids.discard(uid)
        self._pending_source_refresh_targets.pop(uid, None)
        for blocker_uid, target_uids in list(
            self._pending_source_refresh_targets.items()
        ):
            target_uids.discard(uid)
            if not target_uids:
                self._pending_source_refresh_targets.pop(blocker_uid, None)

    def _index_pending_uids(self) -> None:
        for uid in tuple(self._unindexed_uids):
            self.refs_for_uid(uid)

    def _remove_reverse_refs(self, dependent_uid: str) -> None:
        source_uids = self._source_uids_by_dependent.pop(dependent_uid, set())
        for source_uid in source_uids:
            dependents = self._dependents_by_source_uid.get(source_uid)
            if dependents is None:
                continue
            dependents.discard(dependent_uid)
            if not dependents:
                self._dependents_by_source_uid.pop(source_uid, None)

    def queue_source_refresh(self, blocker_uid: str, target_uid: str) -> None:
        self._pending_source_refresh_targets.setdefault(blocker_uid, set()).add(
            target_uid
        )

    def pop_source_refreshes(self, blocker_uid: str) -> set[str]:
        return self._pending_source_refresh_targets.pop(blocker_uid, set())

    def has_pending_source_refreshes(self) -> bool:
        return bool(self._pending_source_refresh_targets)
