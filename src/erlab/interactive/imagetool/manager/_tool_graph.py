"""Managed ImageTool node graph."""

from __future__ import annotations

__all__ = ["_ManagerToolGraph"]

import contextlib
import typing

from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from qtpy import QtWidgets


class _ManagerToolGraph:
    """Own root/child manager nodes and their display order."""

    def __init__(self) -> None:
        self.root_wrappers: dict[int, _ImageToolWrapper] = {}
        self.nodes: dict[str, _ImageToolWrapper | _ManagedWindowNode] = {}
        self.displayed_indices: list[int] = []
        self.figure_uids: list[str] = []
        self._node_uid_counter: int = 0

    @property
    def ntools(self) -> int:
        return len(self.root_wrappers)

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

    def uid_from_window(self, widget: object) -> str | None:
        for uid, node in self.nodes.items():
            if node.window is widget:
                return uid
        return None

    def register_root(self, wrapper: _ImageToolWrapper) -> None:
        self.root_wrappers[wrapper.index] = wrapper
        self.nodes[wrapper.uid] = wrapper

    def register_child(self, node: _ManagedWindowNode) -> None:
        self.nodes[node.uid] = node
        self.parent(node).add_child_reference(
            node.uid, typing.cast("QtWidgets.QWidget", node.window)
        )

    def register_figure(self, node: _ManagedWindowNode) -> None:
        self.nodes[node.uid] = node
        if node.uid not in self.figure_uids:
            self.figure_uids.append(node.uid)

    def replace_child_references(
        self,
        uid: str,
        child_uids: list[str],
        childtools: dict[str, QtWidgets.QWidget],
    ) -> None:
        node = self.nodes[uid]
        node._childtool_indices = child_uids
        node._childtools = childtools

    def unregister_node(
        self, uid: str
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        node = self.nodes.pop(uid, None)
        if node is None:
            return None
        with contextlib.suppress(ValueError):
            self.figure_uids.remove(uid)
        if node.parent_uid is not None:
            parent = self.nodes.get(node.parent_uid)
            if parent is not None:
                parent.remove_child_reference(uid)
        return node

    def unregister_root(self, index: int) -> _ImageToolWrapper | None:
        wrapper = self.root_wrappers.pop(index, None)
        if wrapper is None:
            return None
        self.nodes.pop(wrapper.uid, None)
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

    def remove_root_rows(self, row: int, count: int) -> None:
        del self.displayed_indices[row : row + count]

    def clear_root_order(self) -> None:
        self.displayed_indices.clear()

    def move_root_rows(self, moves: Iterable[tuple[int, int]]) -> None:
        for src, dest in moves:
            self.displayed_indices.insert(dest, self.displayed_indices.pop(src))

    def remove_child_rows(self, parent_uid: str, row: int, count: int) -> None:
        del self.nodes[parent_uid]._childtool_indices[row : row + count]

    def move_child_rows(
        self, parent_uid: str, moves: Iterable[tuple[int, int]]
    ) -> None:
        child_uids = self.nodes[parent_uid]._childtool_indices
        for src, dest in moves:
            child_uids.insert(dest, child_uids.pop(src))

    def reindex_roots(self) -> None:
        new_root_wrappers: dict[int, _ImageToolWrapper] = {}
        displayed_indices = list(self.displayed_indices)
        for row_idx, tool_idx in enumerate(displayed_indices):
            self.displayed_indices[row_idx] = row_idx
            self.root_wrappers[tool_idx]._index = row_idx
            new_root_wrappers[row_idx] = self.root_wrappers[tool_idx]
        self.root_wrappers = new_root_wrappers
