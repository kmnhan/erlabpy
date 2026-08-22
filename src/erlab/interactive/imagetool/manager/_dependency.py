"""Dependency bookkeeping for managed ImageTool nodes."""

from __future__ import annotations

import typing

from erlab.interactive.imagetool._provenance._model import (
    ScriptInputDependencyRef,
    script_input_dependency_refs,
    script_inputs_dependency_refs,
)

__all__ = ["_DependencyStatus", "_ManagerDependencyTracker"]

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from erlab.interactive.imagetool._provenance._model import ScriptInput
    from erlab.interactive.imagetool.manager._tool_graph import _ManagerToolGraph
    from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode

_DependencyStatus = typing.Literal["current", "changed", "missing"]
_SourceState = typing.Literal["fresh", "stale", "unavailable"]


def _combine_source_states(states: Iterable[_SourceState]) -> _SourceState:
    """Return the most severe state from a set of source states."""
    aggregate: _SourceState = "fresh"
    for state in states:
        if state == "unavailable":
            return "unavailable"
        if state == "stale":
            aggregate = "stale"
    return aggregate


def _effective_script_input_dependency_refs(
    script_inputs: Sequence[ScriptInput],
) -> tuple[ScriptInputDependencyRef, ...]:
    """Return the live dependencies selected by named input resolution."""
    refs: list[ScriptInputDependencyRef] = []
    for script_input in script_inputs:
        if script_input.node_uid:
            refs.extend(script_inputs_dependency_refs((script_input,)))
            continue
        fallback = script_input.parsed_provenance_spec()
        if fallback is not None:
            refs.extend(_effective_script_input_dependency_refs(fallback.script_inputs))
    return tuple(refs)


class _ManagerDependencyTracker:
    """Track script-input dependencies and deferred source refresh requests."""

    def __init__(self, graph: _ManagerToolGraph) -> None:
        self._graph = graph
        self._ref_cache: dict[
            str,
            tuple[
                _ManagedWindowNode,
                int,
                tuple[
                    ScriptInputDependencyRef,
                    ...,
                ],
            ],
        ] = {}
        self._source_uids_by_dependent: dict[str, set[str]] = {}
        self._dependents_by_source_uid: dict[str, dict[str, None]] = {}
        self._unindexed_uids: dict[str, None] = {}
        self._status_cache: dict[str, _DependencyStatus | None] = {}
        self._pending_source_refresh_targets: dict[str, dict[str, bool]] = {}

    def note_uid(self, uid: str) -> None:
        self._unindexed_uids[uid] = None
        for dependent_uid in self._dependents_by_source_uid.get(uid, ()):
            self._status_cache.pop(dependent_uid, None)

    def invalidate_uid(self, uid: str) -> None:
        self._remove_reverse_refs(uid)
        self._ref_cache.pop(uid, None)
        self._status_cache.pop(uid, None)
        if uid in self._graph.nodes:
            self._unindexed_uids[uid] = None
        else:
            self._unindexed_uids.pop(uid, None)

    def refs_for_uid(self, uid: str) -> tuple[ScriptInputDependencyRef, ...]:
        node = self._graph.nodes.get(uid)
        if node is None:
            self.invalidate_uid(uid)
            return ()
        if node.is_imagetool:
            script_inputs = ()
            spec = node.provenance_spec
        else:
            script_inputs = node.tool_script_inputs
            spec = None if script_inputs else node.provenance_spec
            if spec is not None:
                script_inputs = spec.script_inputs
        if spec is None and not script_inputs:
            self._remove_reverse_refs(uid)
            self._ref_cache.pop(uid, None)
            self._unindexed_uids.pop(uid, None)
            self._status_cache[uid] = None
            return ()
        cached = self._ref_cache.get(uid)
        if (
            cached is not None
            and cached[0] is node
            and cached[1] == node.provenance_revision
        ):
            self._unindexed_uids.pop(uid, None)
            return cached[2]
        refs = (
            _effective_script_input_dependency_refs(script_inputs)
            if script_inputs
            else script_input_dependency_refs(spec)
        )
        self._remove_reverse_refs(uid)
        source_uids = {ref.node_uid for ref in refs}
        self._source_uids_by_dependent[uid] = source_uids
        for source_uid in source_uids:
            self._dependents_by_source_uid.setdefault(source_uid, {})[uid] = None
        self._ref_cache[uid] = (node, node.provenance_revision, refs)
        self._unindexed_uids.pop(uid, None)
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
        # Registration and provenance signals defer index work so that repeated
        # changes can coalesce. Complete that bounded pending work before a
        # reverse lookup so a source update cannot miss a newly added edge.
        for pending_uid in tuple(self._unindexed_uids):
            self.refs_for_uid(pending_uid)
        dependents = self._dependents_by_source_uid.get(uid, {})
        for dependent_uid in dependents:
            self._status_cache.pop(dependent_uid, None)
        return list(dependents)

    def clear_uid(self, uid: str) -> None:
        self.invalidate_uid(uid)
        self._unindexed_uids.pop(uid, None)
        self._pending_source_refresh_targets.pop(uid, None)
        for blocker_uid, target_uids in list(
            self._pending_source_refresh_targets.items()
        ):
            target_uids.pop(uid, None)
            if not target_uids:
                self._pending_source_refresh_targets.pop(blocker_uid, None)

    def _remove_reverse_refs(self, dependent_uid: str) -> None:
        source_uids = self._source_uids_by_dependent.pop(dependent_uid, set())
        for source_uid in source_uids:
            dependents = self._dependents_by_source_uid.get(source_uid)
            if dependents is None:
                continue
            dependents.pop(dependent_uid, None)
            if not dependents:
                self._dependents_by_source_uid.pop(source_uid, None)

    def queue_source_refresh(
        self,
        blocker_uid: str,
        target_uid: str,
        *,
        automatic: bool = False,
    ) -> None:
        """Queue a deferred refresh continuation.

        A manual continuation remains manual when an automatic update later queues
        the same target. This lets a disabled automatic-update setting cancel only
        automatic self-refreshes.
        """
        target_uids = self._pending_source_refresh_targets.setdefault(blocker_uid, {})
        existing = target_uids.get(target_uid)
        target_uids[target_uid] = (
            automatic if existing is None else existing and automatic
        )

    def pop_source_refreshes(self, blocker_uid: str) -> set[str]:
        return set(self.pop_source_refresh_intents(blocker_uid))

    def pop_source_refresh_intents(self, blocker_uid: str) -> dict[str, bool]:
        """Return deferred targets mapped to whether they are automatic."""
        return self._pending_source_refresh_targets.pop(blocker_uid, {})

    def source_refresh_queued(self, blocker_uid: str, target_uid: str) -> bool:
        return target_uid in self._pending_source_refresh_targets.get(blocker_uid, ())

    def discard_source_refreshes(self, blocker_uids: Iterable[str]) -> None:
        """Discard deferred continuation chains below the given blockers."""
        pending = list(blocker_uids)
        seen: set[str] = set()
        while pending:
            blocker_uid = pending.pop()
            if blocker_uid in seen:
                continue
            seen.add(blocker_uid)
            pending.extend(self.pop_source_refreshes(blocker_uid))

    def discard_source_refresh_chain(self, uid: str) -> None:
        """Discard queued refreshes that reach or depend on a failed target."""
        pending = [uid]
        seen: set[str] = set()
        while pending:
            target_uid = pending.pop()
            if target_uid in seen:
                continue
            seen.add(target_uid)
            pending.extend(self.pop_source_refreshes(target_uid))
            for blocker_uid, target_uids in list(
                self._pending_source_refresh_targets.items()
            ):
                target_uids.pop(target_uid, None)
                if not target_uids:
                    self._pending_source_refresh_targets.pop(blocker_uid, None)

    def has_pending_source_refreshes(self) -> bool:
        return bool(self._pending_source_refresh_targets)
