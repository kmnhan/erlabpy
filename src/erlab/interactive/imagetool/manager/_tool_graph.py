"""Managed ImageTool node graph."""

from __future__ import annotations

__all__ = ["_ManagerToolGraph"]

import typing

from erlab.interactive.imagetool.manager._node_change import _ManagedNodeChange
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from qtpy import QtWidgets


class _ManagerToolGraph:
    """Own root/child manager nodes and their display order."""

    def __init__(
        self,
        node_changed_callback: Callable[[str | None, _ManagedNodeChange], None]
        | None = None,
    ) -> None:
        self.root_wrappers: dict[int, _ImageToolWrapper] = {}
        self.nodes: dict[str, _ImageToolWrapper | _ManagedWindowNode] = {}
        self.displayed_indices: list[int] = []
        self.figure_uids: list[str] = []
        self._figure_uid_set: set[str] = set()
        self._imagetool_count: int = 0
        self._structure_generation: int = 0
        self._node_changed_callback = node_changed_callback
        self._node_uid_counter: int = 0

    @property
    def ntools(self) -> int:
        return len(self.root_wrappers)

    @property
    def nimagetools(self) -> int:
        return self._imagetool_count

    @property
    def structure_generation(self) -> int:
        return self._structure_generation

    @property
    def next_index(self) -> int:
        return max(self.root_wrappers.keys(), default=-1) + 1

    @property
    def uid_counter(self) -> int:
        return self._node_uid_counter

    def restore_uid_counter(self, value: int) -> None:
        self._node_uid_counter = int(value)

    def next_uid(self, preferred: str | None = None) -> str:
        if preferred is not None and preferred not in self.nodes:
            self.consume_uid(preferred)
            return preferred
        while True:
            uid = f"n{self._node_uid_counter}"
            self._node_uid_counter += 1
            if uid not in self.nodes:
                return uid

    def consume_uid(self, uid: str) -> None:
        if uid.startswith("n") and uid[1:].isdigit():
            self._node_uid_counter = max(self._node_uid_counter, int(uid[1:]) + 1)

    def node(self, target: int | str) -> _ImageToolWrapper | _ManagedWindowNode:
        if isinstance(target, int):
            return self.root_wrappers[target]
        return self.nodes[target]

    def child(self, uid: str) -> _ManagedWindowNode:
        node = self.nodes[uid]
        if isinstance(node, _ImageToolWrapper):
            raise KeyError(f"{uid!r} refers to a root ImageTool")
        return node

    def parent(
        self, node: _ManagedWindowNode
    ) -> _ImageToolWrapper | _ManagedWindowNode:
        if node.parent_uid is None:
            raise KeyError(f"Node {node.uid!r} has no parent")
        return self.nodes[node.parent_uid]

    def root_for_uid(self, uid: str) -> _ImageToolWrapper:
        node = self.node(uid)
        while not isinstance(node, _ImageToolWrapper):
            node = self.parent(node)
        return node

    def node_path(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[int, ...] | None:
        """Return the root index and child rows that currently address a node."""
        child_rows: list[int] = []
        current = node
        while current.parent_uid is not None:
            parent = self.nodes.get(current.parent_uid)
            if parent is None or current.uid not in parent._childtool_indices:
                return None
            child_rows.append(parent._childtool_indices.index(current.uid))
            current = parent

        for root_index, wrapper in self.root_wrappers.items():
            if wrapper is current:
                return (root_index, *reversed(child_rows))
        return None

    def uid_from_window(self, widget: object) -> str | None:
        for uid, node in self.nodes.items():
            if node.window is widget:
                return uid
        return None

    def is_figure_uid(self, uid: str) -> bool:
        return uid in self._figure_uid_set

    def _structure_changed(self) -> None:
        self._structure_generation += 1
        self.notify_node_change(None, _ManagedNodeChange.PRESENTATION)

    def notify_node_change(
        self,
        uid: str | None,
        change: _ManagedNodeChange,
    ) -> None:
        """Publish derived-state changes through the graph's owner callback."""
        if change == _ManagedNodeChange.NONE:
            return
        if uid is not None and uid not in self.nodes:
            return
        if self._node_changed_callback is not None:
            self._node_changed_callback(uid, change)

    def register_root(self, wrapper: _ImageToolWrapper) -> None:
        self._require_available_uid(wrapper.uid)
        if wrapper.index in self.root_wrappers:
            raise ValueError(
                f"Root ImageTool index {wrapper.index} is already registered"
            )
        self.root_wrappers[wrapper.index] = wrapper
        self.nodes[wrapper.uid] = wrapper
        self._imagetool_count += 1
        self._structure_changed()

    def register_child(self, node: _ManagedWindowNode) -> None:
        if (
            node.tool_window is not None
            and node.tool_window.manager_collection == "figures"
        ):
            raise ValueError("Figure nodes must use register_figure()")
        self._require_available_uid(node.uid)
        if node.parent_uid is None:
            raise ValueError("Child nodes must have a registered parent")
        parent = self.nodes.get(node.parent_uid)
        if parent is None:
            raise ValueError(f"Parent node UID {node.parent_uid!r} is not registered")
        if node.uid in parent._childtool_indices:
            raise ValueError(
                f"Child node UID {node.uid!r} is already registered with its parent"
            )
        parent.add_child_reference(node.uid, node.window)
        self.nodes[node.uid] = node
        if node.is_imagetool:
            self._imagetool_count += 1
        self._structure_changed()

    def register_figure(self, node: _ManagedWindowNode) -> None:
        if node.is_imagetool or (
            node.tool_window is not None
            and node.tool_window.manager_collection != "figures"
        ):
            raise ValueError("Only figure tools can be registered as figure nodes")
        self._require_available_uid(node.uid)
        if node.uid in self._figure_uid_set:
            raise ValueError(f"Figure node UID {node.uid!r} is already registered")
        self.nodes[node.uid] = node
        self.figure_uids.append(node.uid)
        self._figure_uid_set.add(node.uid)
        self._structure_changed()

    def _require_available_uid(self, uid: str) -> None:
        if uid in self.nodes:
            raise ValueError(f"Managed node UID {uid!r} is already registered")

    def update_node_window_reference(self, node: _ManagedWindowNode) -> None:
        """Synchronize the parent cache after a node replaces its window."""
        if self.nodes.get(node.uid) is not node or node.parent_uid is None:
            return
        parent = self.nodes.get(node.parent_uid)
        if parent is None:
            return
        if node.window is None:
            parent._childtools.pop(node.uid, None)
        else:
            parent._childtools[node.uid] = node.window

    def replace_child_references(
        self,
        uid: str,
        child_uids: list[str],
        childtools: dict[str, QtWidgets.QWidget],
    ) -> None:
        node = self.nodes[uid]
        node._childtool_indices = list(child_uids)
        node._childtools = dict(childtools)
        self._structure_changed()

    def add_child_reference(
        self,
        parent_uid: str,
        child_uid: str,
        window: QtWidgets.QWidget | None,
    ) -> None:
        parent = self.nodes[parent_uid]
        was_present = child_uid in parent._childtool_indices
        parent.add_child_reference(child_uid, window)
        if not was_present:
            self._structure_changed()

    def remove_child_references(
        self, parent_uid: str, child_uids: Iterable[str]
    ) -> None:
        parent = self.nodes[parent_uid]
        removed = False
        for child_uid in child_uids:
            if child_uid not in parent._childtool_indices:
                continue
            parent.remove_child_reference(child_uid)
            removed = True
        if removed:
            self._structure_changed()

    def replace_child_order(self, parent_uid: str, child_uids: list[str]) -> None:
        self.nodes[parent_uid]._childtool_indices = list(child_uids)
        self._structure_changed()

    def unregister_node(
        self, uid: str
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        node = self.nodes.pop(uid, None)
        if node is None:
            return None
        if uid in self._figure_uid_set:
            self._figure_uid_set.remove(uid)
            self.figure_uids.remove(uid)
        if node.is_imagetool:
            self._imagetool_count -= 1
        if node.parent_uid is not None:
            parent = self.nodes.get(node.parent_uid)
            if parent is not None:
                parent.remove_child_reference(uid)
        self._structure_changed()
        return node

    def unregister_root(self, index: int) -> _ImageToolWrapper | None:
        wrapper = self.root_wrappers.pop(index, None)
        if wrapper is None:
            return None
        self.nodes.pop(wrapper.uid, None)
        self._imagetool_count -= 1
        self._structure_changed()
        return wrapper

    def descendant_uids(self, uid: str) -> list[str]:
        descendants: list[str] = []
        stack = [uid]
        while stack:
            current = stack.pop()
            node = self.nodes.get(current)
            if node is None:
                continue
            for child_uid in node._childtool_indices:
                descendants.append(child_uid)
                stack.append(child_uid)
        return descendants

    def subtree_uids(self, uid: str) -> list[str]:
        return [uid, *self.descendant_uids(uid)]

    def root_indices_for_workspace(self) -> tuple[int, ...]:
        displayed = [idx for idx in self.displayed_indices if idx in self.root_wrappers]
        remaining = [
            idx for idx in self.root_wrappers if idx not in self.displayed_indices
        ]
        return (*displayed, *remaining)

    def insert_root_order(self, index: int, row: int | None = None) -> None:
        if row is None:
            row = len(self.displayed_indices)
        self.displayed_indices.insert(row, index)
        self._structure_changed()

    def remove_root_rows(self, row: int, count: int) -> None:
        del self.displayed_indices[row : row + count]
        self._structure_changed()

    def clear_root_order(self) -> None:
        self.displayed_indices.clear()
        self._structure_changed()

    def move_root_rows(self, moves: Iterable[tuple[int, int]]) -> None:
        for src, dest in moves:
            self.displayed_indices.insert(dest, self.displayed_indices.pop(src))
        self._structure_changed()

    def remove_child_rows(self, parent_uid: str, row: int, count: int) -> None:
        del self.nodes[parent_uid]._childtool_indices[row : row + count]
        self._structure_changed()

    def move_child_rows(
        self, parent_uid: str, moves: Iterable[tuple[int, int]]
    ) -> None:
        child_uids = self.nodes[parent_uid]._childtool_indices
        for src, dest in moves:
            child_uids.insert(dest, child_uids.pop(src))
        self._structure_changed()

    def reindex_roots(self) -> None:
        new_root_wrappers: dict[int, _ImageToolWrapper] = {}
        displayed_indices = list(self.displayed_indices)
        for row_idx, tool_idx in enumerate(displayed_indices):
            self.displayed_indices[row_idx] = row_idx
            self.root_wrappers[tool_idx]._index = row_idx
            new_root_wrappers[row_idx] = self.root_wrappers[tool_idx]
        self.root_wrappers = new_root_wrappers
        self._structure_changed()
