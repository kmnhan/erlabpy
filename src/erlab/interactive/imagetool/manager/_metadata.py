"""Selected-details refresh helpers for ImageTool manager nodes."""

from __future__ import annotations

__all__ = ["_ManagerDetailsRefreshQueue"]

import functools
import typing

from qtpy import QtCore

if typing.TYPE_CHECKING:
    from collections.abc import Callable

_DETAILS_REFRESH_DELAY_MS = 300


class _ManagerDetailsRefreshQueue:
    """Debounce expensive selected-details refreshes for bursty node changes."""

    def __init__(
        self,
        parent: QtCore.QObject,
        flush_callback: Callable[[set[str]], None],
        *,
        idle_scheduler: (
            Callable[[tuple[str, str], Callable[[], None]], None] | None
        ) = None,
    ) -> None:
        self._flush_callback = flush_callback
        self._idle_scheduler = idle_scheduler
        self._pending_uids: set[str] = set()
        self._generation = 0
        self._timer = QtCore.QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_DETAILS_REFRESH_DELAY_MS)
        self._timer.timeout.connect(self._request_flush)

    @property
    def pending_uids(self) -> frozenset[str]:
        return frozenset(self._pending_uids)

    def schedule(self, uid: str) -> None:
        self._pending_uids.add(uid)
        self._generation += 1
        self._timer.start()

    def flush(self) -> None:
        self._generation += 1
        self._timer.stop()
        self._flush_pending()

    def _flush_pending(self) -> None:
        pending = self._pending_uids
        self._pending_uids = set()
        if pending:
            self._flush_callback(pending)

    def _request_flush(self) -> None:
        generation = self._generation
        if self._idle_scheduler is None:
            self._flush_generation(generation)
            return
        self._idle_scheduler(
            ("details-refresh-debounce", "flush"),
            functools.partial(self._flush_generation, generation),
        )

    def _flush_generation(self, generation: int) -> None:
        if generation != self._generation:
            return
        self._flush_pending()

    def set_interval(self, msec: int) -> None:
        self._timer.setInterval(msec)

    def is_active(self) -> bool:
        return self._timer.isActive()
