"""Metadata refresh helpers for ImageTool manager nodes."""

from __future__ import annotations

__all__ = ["_ManagerToolMetadataQueue"]

import typing

from qtpy import QtCore

if typing.TYPE_CHECKING:
    from collections.abc import Callable


class _ManagerToolMetadataQueue:
    """Debounce expensive metadata refreshes for bursty tool info changes."""

    def __init__(
        self,
        parent: QtCore.QObject,
        flush_callback: Callable[[set[str]], None],
    ) -> None:
        self._flush_callback = flush_callback
        self._pending_uids: set[str] = set()
        self._timer = QtCore.QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self.flush)

    @property
    def pending_uids(self) -> frozenset[str]:
        return frozenset(self._pending_uids)

    def schedule(self, uid: str) -> None:
        self._pending_uids.add(uid)
        self._timer.start()

    def flush(self) -> None:
        pending = self._pending_uids
        self._pending_uids = set()
        if pending:
            self._flush_callback(pending)

    def set_interval(self, msec: int) -> None:
        self._timer.setInterval(msec)

    def is_active(self) -> bool:
        return self._timer.isActive()
