"""Managed-node change notifications."""

from __future__ import annotations

__all__ = ["_ManagedNodeChange"]

import enum


class _ManagedNodeChange(enum.Flag):
    """Cross-component state derived from a managed node."""

    NONE = 0
    # Rebuild this node's reverse dependency-index entries when the manager is idle.
    DEPENDENCY_INDEX = enum.auto()
    # Refresh the visible provenance status after dependency indexing completes.
    PROVENANCE_DISPLAY = enum.auto()
    # Refresh labels and collection placement derived from the node.
    PRESENTATION = enum.auto()
    # Refresh only the node's manager-tree row. Do not rebuild selected details.
    ROW = enum.auto()
    # Refresh expensive tool information and previews when the manager is idle.
    INFO = enum.auto()
    # Refresh derivation details for this node or a selected descendant.
    DERIVATION = enum.auto()
    # Refresh nodes whose displayed status or labels refer to this node.
    DEPENDENTS = enum.auto()
    PROVENANCE = DEPENDENCY_INDEX | PROVENANCE_DISPLAY
