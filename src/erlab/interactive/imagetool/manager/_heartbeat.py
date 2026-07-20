"""Background heartbeat for the live ImageTool manager registry."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
import typing

from qtpy import QtCore

import erlab
from erlab.interactive.imagetool.manager._registry import (
    ImageToolManagerRegistryLockError,
    refresh_manager_record,
)

_HEARTBEAT_LOCK_TIMEOUT_MS = 100
_HEARTBEAT_THREAD_STOP_TIMEOUT_MS = 2000
_LOCK_FAILURE_WARNING_THRESHOLD = 10
_LOCK_FAILURE_WARNING_PERIOD = 20
_REGISTRY_FAILURE_WARNING_PERIOD = 20
_NO_PENDING_WORKSPACE_PATH = object()

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _HeartbeatResult:
    generation: int
    ok: bool
    error_text: str = ""
    lock_failed: bool = False


class _RegistryHeartbeatWorker(QtCore.QObject):
    finished = QtCore.Signal(object)

    @QtCore.Slot(int, str, object, int)
    def refresh(
        self,
        generation: int,
        internal_id: str,
        workspace_path: object,
        lock_timeout_ms: int,
    ) -> None:
        normalized_workspace_path = (
            workspace_path if isinstance(workspace_path, str) else None
        )
        try:
            refresh_manager_record(
                internal_id,
                workspace_path=normalized_workspace_path,
                lock_timeout_ms=lock_timeout_ms,
            )
        except ImageToolManagerRegistryLockError as exc:
            self.finished.emit(
                _HeartbeatResult(
                    generation,
                    ok=False,
                    error_text=str(exc),
                    lock_failed=True,
                )
            )
        except Exception as exc:
            self.finished.emit(
                _HeartbeatResult(generation, ok=False, error_text=str(exc))
            )
        else:
            self.finished.emit(_HeartbeatResult(generation, ok=True))


_ORPHANED_HEARTBEAT_OBJECTS: list[tuple[QtCore.QThread, _RegistryHeartbeatWorker]] = []


class _RegistryHeartbeatController(QtCore.QObject):
    _sigRefreshRequested = QtCore.Signal(int, str, object, int)

    def __init__(
        self,
        internal_id: str,
        *,
        lock_timeout_ms: int = _HEARTBEAT_LOCK_TIMEOUT_MS,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._internal_id = internal_id
        self._lock_timeout_ms = lock_timeout_ms
        self._generation = 0
        self._in_flight_generation: int | None = None
        self._pending_workspace_path: str | None | object = _NO_PENDING_WORKSPACE_PATH
        self._stopping = False
        self._consecutive_lock_failures = 0
        self._consecutive_registry_failures = 0
        thread = QtCore.QThread()
        thread.setObjectName("ImageToolManagerRegistryHeartbeat")
        worker = _RegistryHeartbeatWorker()
        worker.moveToThread(thread)
        self._thread: QtCore.QThread | None = thread
        self._worker: _RegistryHeartbeatWorker | None = worker
        self._sigRefreshRequested.connect(
            worker.refresh,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        worker.finished.connect(
            self._refresh_finished,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        thread.finished.connect(worker.deleteLater)
        thread.start()

    @property
    def is_busy(self) -> bool:
        return self._in_flight_generation is not None

    @property
    def is_stopping(self) -> bool:
        return self._stopping

    def request_refresh(
        self,
        workspace_path: str | None,
        *,
        coalesce_if_busy: bool,
    ) -> None:
        if self._stopping:
            return
        if self._in_flight_generation is not None:
            if coalesce_if_busy:
                self._pending_workspace_path = workspace_path
            return
        self._start_refresh(workspace_path)

    def stop(self) -> None:
        if self._stopping:
            return
        self._pending_workspace_path = _NO_PENDING_WORKSPACE_PATH
        idle = self._wait_until_idle(_HEARTBEAT_THREAD_STOP_TIMEOUT_MS)
        self._stopping = True
        self._in_flight_generation = None
        thread = self._thread
        if thread is None:
            return
        if idle:
            self._disconnect_worker_signals()
        thread.quit()
        if idle and thread.wait(_HEARTBEAT_THREAD_STOP_TIMEOUT_MS):
            self._worker = None
            self._thread = None
            return
        if self._worker is not None:
            self._retain_orphaned_thread(thread, self._worker)
        logger.warning(
            "ImageTool manager registry heartbeat thread did not stop promptly",
            extra={"suppress_ui_alert": True},
        )
        self._worker = None
        self._thread = None

    def _wait_until_idle(self, timeout_ms: int) -> bool:
        """Let an in-flight refresh finish before tearing down the worker thread."""
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000
        while self._in_flight_generation is not None:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                return False
            app = QtCore.QCoreApplication.instance()
            if app is not None:
                # The worker completion is a queued metacall. Processing every event
                # here can re-enter unrelated UI teardown while the controller stops.
                QtCore.QCoreApplication.sendPostedEvents(
                    None, int(QtCore.QEvent.Type.MetaCall.value)
                )
            QtCore.QThread.msleep(min(10, remaining_ms))
        return True

    def _start_refresh(self, workspace_path: str | None) -> None:
        thread = self._thread
        if thread is None or not thread.isRunning():
            logger.warning(
                "ImageTool manager registry heartbeat thread is unavailable",
                extra={"suppress_ui_alert": True},
            )
            return
        self._generation += 1
        self._in_flight_generation = self._generation
        self._sigRefreshRequested.emit(
            self._generation,
            self._internal_id,
            workspace_path,
            self._lock_timeout_ms,
        )

    @QtCore.Slot(object)
    def _refresh_finished(self, result: object) -> None:
        if not isinstance(result, _HeartbeatResult):
            return
        if (
            self._stopping
            or not erlab.interactive.utils.qt_is_valid(self)
            or result.generation != self._in_flight_generation
        ):
            return
        self._in_flight_generation = None
        self._handle_result(result)
        if (
            self._pending_workspace_path is not _NO_PENDING_WORKSPACE_PATH
            and not self._stopping
        ):
            workspace_path = typing.cast("str | None", self._pending_workspace_path)
            self._pending_workspace_path = _NO_PENDING_WORKSPACE_PATH
            self._start_refresh(workspace_path)

    def _handle_result(self, result: _HeartbeatResult) -> None:
        if result.ok:
            self._consecutive_lock_failures = 0
            self._consecutive_registry_failures = 0
            return
        if result.lock_failed:
            self._consecutive_lock_failures += 1
            self._consecutive_registry_failures = 0
            self._log_lock_failure(result.error_text)
            return
        self._consecutive_lock_failures = 0
        self._consecutive_registry_failures += 1
        self._log_registry_failure(result.error_text)

    def _log_lock_failure(self, error_text: str) -> None:
        count = self._consecutive_lock_failures
        if count == _LOCK_FAILURE_WARNING_THRESHOLD or (
            count > _LOCK_FAILURE_WARNING_THRESHOLD
            and count % _LOCK_FAILURE_WARNING_PERIOD == 0
        ):
            logger.warning(
                "Could not lock ImageTool manager registry for %d heartbeat attempts: "
                "%s",
                count,
                error_text,
                extra={"suppress_ui_alert": True},
            )
            return
        logger.debug(
            "Could not lock ImageTool manager registry for heartbeat refresh: %s",
            error_text,
            extra={"suppress_ui_alert": True},
        )

    def _log_registry_failure(self, error_text: str) -> None:
        count = self._consecutive_registry_failures
        if count == 1 or count % _REGISTRY_FAILURE_WARNING_PERIOD == 0:
            logger.warning(
                "Could not refresh ImageTool manager registry record: %s",
                error_text,
                extra={"suppress_ui_alert": True},
            )
            return
        logger.debug(
            "Could not refresh ImageTool manager registry record: %s",
            error_text,
            extra={"suppress_ui_alert": True},
        )

    def _disconnect_worker_signals(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if not erlab.interactive.utils.qt_is_valid(worker):
            return
        with contextlib.suppress(TypeError, RuntimeError, SystemError):
            self._sigRefreshRequested.disconnect(worker.refresh)
        with contextlib.suppress(TypeError, RuntimeError, SystemError):
            worker.finished.disconnect(self._refresh_finished)

    @staticmethod
    def _retain_orphaned_thread(
        thread: QtCore.QThread,
        worker: _RegistryHeartbeatWorker,
    ) -> None:
        orphan = (thread, worker)
        _ORPHANED_HEARTBEAT_OBJECTS.append(orphan)

        def release_orphan() -> None:
            with contextlib.suppress(ValueError):
                _ORPHANED_HEARTBEAT_OBJECTS.remove(orphan)

        thread.finished.connect(release_orphan)
        if not thread.isRunning():
            release_orphan()
