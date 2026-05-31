"""Dependency bookkeeping for managed ImageTool nodes."""

from __future__ import annotations

__all__ = ["_DependencyStatus", "_ManagerDependencyTracker"]

import typing

from erlab.interactive.imagetool import provenance_framework

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
                    provenance_framework.ScriptInputDependencyRef,
                    ...,
                ],
            ],
        ] = {}
        self._pending_source_refresh_targets: dict[str, set[str]] = {}

    def refs_for_uid(
        self, uid: str
    ) -> tuple[provenance_framework.ScriptInputDependencyRef, ...]:
        node = self._graph.nodes.get(uid)
        if node is None or node.provenance_spec is None:
            self._ref_cache.pop(uid, None)
            return ()
        spec_id = id(node.provenance_spec)
        cached = self._ref_cache.get(uid)
        if cached is not None and cached[0] == spec_id:
            return cached[1]
        refs = provenance_framework.script_input_dependency_refs(node.provenance_spec)
        self._ref_cache[uid] = (spec_id, refs)
        return refs

    def status_for_uid(self, uid: str) -> _DependencyStatus | None:
        refs = self.refs_for_uid(uid)
        if not refs:
            return None

        changed = False
        for ref in refs:
            parent = self._graph.nodes.get(ref.node_uid)
            if parent is None:
                return "missing"
            if (
                ref.node_snapshot_token is not None
                and parent.snapshot_token != ref.node_snapshot_token
            ):
                changed = True
        return "changed" if changed else "current"

    def dependent_uids(self, uid: str) -> list[str]:
        return [
            node_uid
            for node_uid in self._graph.nodes
            if node_uid != uid
            and any(ref.node_uid == uid for ref in self.refs_for_uid(node_uid))
        ]

    def clear_uid(self, uid: str) -> None:
        self._ref_cache.pop(uid, None)
        self._pending_source_refresh_targets.pop(uid, None)
        for blocker_uid, target_uids in list(
            self._pending_source_refresh_targets.items()
        ):
            target_uids.discard(uid)
            if not target_uids:
                self._pending_source_refresh_targets.pop(blocker_uid, None)

    def queue_source_refresh(self, blocker_uid: str, target_uid: str) -> None:
        self._pending_source_refresh_targets.setdefault(blocker_uid, set()).add(
            target_uid
        )

    def pop_source_refreshes(self, blocker_uid: str) -> set[str]:
        return self._pending_source_refresh_targets.pop(blocker_uid, set())

    def has_pending_source_refreshes(self) -> bool:
        return bool(self._pending_source_refresh_targets)
