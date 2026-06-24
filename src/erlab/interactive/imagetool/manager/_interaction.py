"""Idle-aware manager work scheduling."""

from __future__ import annotations

__all__ = ["_ManagerInteractionGate"]

import collections
import logging
import os
import time
import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable

logger = logging.getLogger(__name__)

_IDLE_QUIET_INTERVAL_MS = 150
_IDLE_BATCH_BUDGET_MS = 6


def _manager_perf_timing_enabled() -> bool:
    return os.environ.get("ERLAB_MANAGER_PERF_TIMING") == "1"


class _ManagerInteractionGate(QtCore.QObject):
    """Keep secondary manager work out of active direct manipulation."""

    _ACTIVE_EVENT_TYPES = frozenset(
        {
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QEvent.Type.MouseButtonDblClick,
            QtCore.QEvent.Type.Wheel,
            QtCore.QEvent.Type.KeyPress,
            QtCore.QEvent.Type.KeyRelease,
            QtCore.QEvent.Type.InputMethod,
        }
    )
    _EDITOR_FOCUS_TYPES = frozenset(
        {
            QtCore.QEvent.Type.FocusIn,
            QtCore.QEvent.Type.FocusOut,
        }
    )
    _EDITOR_WIDGET_TYPES = (
        QtWidgets.QAbstractSpinBox,
        QtWidgets.QAbstractSlider,
        QtWidgets.QComboBox,
        QtWidgets.QLineEdit,
        QtWidgets.QPlainTextEdit,
        QtWidgets.QTextEdit,
    )

    def __init__(self, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._roots: weakref.WeakSet[QtWidgets.QWidget] = weakref.WeakSet()
        self._pending_work: collections.OrderedDict[Hashable, Callable[[], None]] = (
            collections.OrderedDict()
        )
        self._active: bool = False
        self._quiet_timer = QtCore.QTimer(self)
        self._quiet_timer.setSingleShot(True)
        self._quiet_timer.setInterval(_IDLE_QUIET_INTERVAL_MS)
        self._quiet_timer.timeout.connect(self._interaction_settled)
        self._work_timer = QtCore.QTimer(self)
        self._work_timer.setSingleShot(True)
        self._work_timer.setInterval(0)
        self._work_timer.timeout.connect(self._run_pending_work)
        self._batch_budget_ms = _IDLE_BATCH_BUDGET_MS
        self._queued_since_flush = 0
        self._replaced_since_flush = 0
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    @property
    def is_active(self) -> bool:
        return self._active or self._quiet_timer.isActive()

    @property
    def pending_keys(self) -> tuple[Hashable, ...]:
        return tuple(self._pending_work)

    def set_quiet_interval(self, msec: int) -> None:
        self._quiet_timer.setInterval(msec)

    def set_batch_budget(self, msec: int) -> None:
        self._batch_budget_ms = max(1, int(msec))

    def register_window(self, widget: QtWidgets.QWidget | None) -> None:
        if widget is not None:
            self._roots.add(widget)

    def unregister_window(self, widget: QtWidgets.QWidget | None) -> None:
        if widget is not None:
            self._roots.discard(widget)

    def note_activity(self) -> None:
        self._active = True
        self._work_timer.stop()
        self._quiet_timer.start()

    def queue_work(
        self,
        key: Hashable,
        callback: Callable[[], None],
        *,
        require_idle: bool = True,
    ) -> None:
        if _manager_perf_timing_enabled():
            self._queued_since_flush += 1
            if key in self._pending_work:
                self._replaced_since_flush += 1
        self._pending_work[key] = callback
        if require_idle and self.is_active:
            return
        self._schedule_pending_work()

    def flush(
        self,
        *,
        key_prefix: Hashable | None = None,
        force: bool = False,
    ) -> None:
        if self.is_active and not force:
            return
        start_time = time.perf_counter() if _manager_perf_timing_enabled() else 0.0
        flushed = 0
        keys = list(self._pending_work)
        for key in keys:
            if key_prefix is not None and not self._key_matches_prefix(key, key_prefix):
                continue
            callback = self._pending_work.pop(key, None)
            if callback is not None:
                flushed += 1
                self._run_callback(callback)
        self._log_flush_stats("forced", flushed, start_time)

    def eventFilter(
        self, obj: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if event is not None:
            try:
                marks_activity = self._event_marks_activity(obj, event)
            except RuntimeError:
                marks_activity = False
            if marks_activity:
                self.note_activity()
        return super().eventFilter(obj, event)

    def _event_marks_activity(
        self, obj: QtCore.QObject | None, event: QtCore.QEvent
    ) -> bool:
        if not self._is_managed_object(obj):
            return False
        event_type = event.type()
        if event_type == QtCore.QEvent.Type.MouseMove:
            return isinstance(event, QtGui.QMouseEvent) and bool(event.buttons())
        if event_type in self._ACTIVE_EVENT_TYPES:
            return True
        return event_type in self._EDITOR_FOCUS_TYPES and isinstance(
            obj, self._EDITOR_WIDGET_TYPES
        )

    def _is_managed_object(self, obj: QtCore.QObject | None) -> bool:
        if not isinstance(obj, QtWidgets.QWidget):
            return False
        for root in tuple(self._roots):
            try:
                if obj is root or root.isAncestorOf(obj):
                    return True
            except RuntimeError:
                self._roots.discard(root)
        return False

    def _interaction_settled(self) -> None:
        self._active = False
        self._schedule_pending_work()

    def _schedule_pending_work(self) -> None:
        if self.is_active or not self._pending_work or self._work_timer.isActive():
            return
        self._work_timer.start()

    def _run_pending_work(self) -> None:
        if self.is_active:
            return
        start_time = time.perf_counter() if _manager_perf_timing_enabled() else 0.0
        flushed = 0
        deadline = time.monotonic() + self._batch_budget_ms / 1000.0
        while self._pending_work and not self.is_active:
            _key, callback = self._pending_work.popitem(last=False)
            flushed += 1
            self._run_callback(callback)
            if time.monotonic() >= deadline:
                break
        self._log_flush_stats("idle", flushed, start_time)
        self._schedule_pending_work()

    def _log_flush_stats(self, mode: str, flushed: int, start_time: float) -> None:
        if not _manager_perf_timing_enabled() or flushed == 0:
            return
        logger.debug(
            "Manager %s work flush ran %d callbacks in %.1f ms "
            "(remaining=%d queued=%d replaced=%d)",
            mode,
            flushed,
            (time.perf_counter() - start_time) * 1000.0,
            len(self._pending_work),
            self._queued_since_flush,
            self._replaced_since_flush,
        )
        self._queued_since_flush = 0
        self._replaced_since_flush = 0

    @staticmethod
    def _run_callback(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            logger.exception("Idle manager work failed")

    @staticmethod
    def _key_matches_prefix(key: Hashable, prefix: Hashable) -> bool:
        if key == prefix:
            return True
        return (
            isinstance(key, tuple)
            and isinstance(prefix, tuple)
            and len(key) >= len(prefix)
            and key[: len(prefix)] == prefix
        )
