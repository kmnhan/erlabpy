import concurrent.futures
import contextlib
import dataclasses
import json
import logging
import subprocess
import sys
import threading
import typing
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
import zmq
from qtpy import QtCore

import erlab
import erlab.interactive.imagetool._itool as itool_mod
import erlab.interactive.imagetool.manager._heartbeat as manager_heartbeat
import erlab.interactive.imagetool.manager._registry as manager_registry
import erlab.interactive.imagetool.manager._server as manager_server
import erlab.interactive.imagetool.manager._widgets as manager_widgets
from erlab.interactive.imagetool import itool
from erlab.interactive.imagetool._magic import _normalize_manager_target_args
from erlab.interactive.imagetool.manager import ImageToolManager, fetch
from erlab.interactive.imagetool.manager._server import Response, _WatcherServer

from .helpers import _use_isolated_manager_registry


def test_manager_selection_info_single_manager(
    tmp_path,
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    manager_mod = erlab.interactive.imagetool.manager
    with manager_context() as manager:
        info = manager_mod.manager_selection_info()

        assert info["reason"] == "single"
        assert info["resolved_index"] == manager.manager_index
        assert info["needs_selection"] is False
        assert info["default_index"] is None
        assert info["managers"][0]["index"] == manager.manager_index
        assert info["managers"][0]["workspace_path"] is None

        workspace_path = tmp_path / "selected.itws"
        manager._adopt_workspace_path(workspace_path)
        qtbot.waitUntil(
            lambda: not manager._registry_heartbeat.is_busy,
            timeout=5000,
        )
        info = manager_mod.manager_selection_info()
        assert info["managers"][0]["workspace_path"] == str(workspace_path.resolve())

        manager_mod.set_default_manager(manager.manager_index)
        info = manager_mod.manager_selection_info()

        assert info["reason"] == "default"
        assert info["default_index"] == manager.manager_index
        assert info["managers"][0]["is_default"] is True


def test_manager_server_threads_are_explicitly_managed(
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        assert manager.server.parent() is None
        assert manager.watcher_server.parent() is None
        assert manager.server.isRunning()
        assert manager.watcher_server.isRunning()


def test_manager_registry_object_repr_and_mapping(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)
    manager_mod = erlab.interactive.imagetool.manager

    assert not manager_mod.managers
    assert repr(manager_mod.managers) == "No live ImageTool managers."
    assert "No live ImageTool managers" in manager_mod.managers._repr_html_()

    records = [
        registry.activate_manager_record(
            registry.reserve_manager_record(host="localhost").internal_id,
            port=45555 + idx * 2,
            watch_port=45556 + idx * 2,
        )
        for idx in range(2)
    ]
    workspace_path = str(tmp_path / "workspace.itws")
    registry.refresh_manager_record(
        records[0].internal_id, workspace_path=workspace_path
    )
    manager_mod.set_default_manager(records[1].index)

    handles = tuple(manager_mod.managers)
    assert len(manager_mod.managers) == 2
    assert [manager.index for manager in handles] == [0, 1]
    assert handles[0].workspace_path == workspace_path
    assert handles[1].workspace_path is None
    assert manager_mod.managers.keys() == (0, 1)
    assert [manager.index for manager in manager_mod.managers.values()] == [0, 1]
    assert [
        (index, manager.index) for index, manager in manager_mod.managers.items()
    ] == [(0, 0), (1, 1)]
    assert manager_mod.managers[1].is_default is True

    with pytest.raises(KeyError):
        manager_mod.managers[2]
    with pytest.raises(TypeError, match="Manager index must be an integer"):
        manager_mod.managers[True]

    text = repr(manager_mod.managers)
    assert "Index" in text
    assert "Default" in text
    assert "Workspace" in text
    assert "#0" in text
    assert "#1" in text
    assert "localhost:45555" in text
    assert workspace_path in text
    assert "yes" in text
    assert "ManagerInfo(" not in text
    assert "internal_id" not in text

    html = manager_mod.managers._repr_html_()
    assert "<table" in html
    assert "#0" in html
    assert workspace_path in html
    assert "ManagerInfo(" not in html
    assert "internal_id" not in html
    assert f"workspace={workspace_path!r}" in repr(manager_mod.managers[0])


def test_manager_registry_lock_timeout_is_configurable(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    timeouts: list[int] = []

    class RecordingLockFile:
        def __init__(self, _path: str) -> None:
            return

        def setStaleLockTime(self, _timeout_ms: int) -> None:
            return

        def tryLock(self, timeout_ms: int) -> bool:
            timeouts.append(timeout_ms)
            return True

        def unlock(self) -> None:
            return

    monkeypatch.setattr(registry.QtCore, "QLockFile", RecordingLockFile)

    with registry._registry_lock(timeout_ms=123):
        pass

    assert timeouts == [123]


def test_manager_registry_lock_failure_uses_specific_error(
    monkeypatch, tmp_path
) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)

    class FailingLockFile:
        def __init__(self, _path: str) -> None:
            return

        def setStaleLockTime(self, _timeout_ms: int) -> None:
            return

        def tryLock(self, _timeout_ms: int) -> bool:
            return False

        def error(self) -> str:
            return "locked"

        def unlock(self) -> None:
            return

    monkeypatch.setattr(registry.QtCore, "QLockFile", FailingLockFile)

    with (
        pytest.raises(registry.ImageToolManagerRegistryLockError, match="locked"),
        registry._registry_lock(timeout_ms=1),
    ):
        pass


def test_manager_registry_refresh_uses_requested_lock_timeout(
    monkeypatch, tmp_path
) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    timeouts: list[int] = []

    @contextlib.contextmanager
    def capture_lock(timeout_ms: int = registry._LOCK_TIMEOUT_MS):
        timeouts.append(timeout_ms)
        yield

    monkeypatch.setattr(registry, "_registry_lock", capture_lock)
    monkeypatch.setattr(registry, "_active_records_unlocked", list)

    registry.refresh_manager_record("missing", lock_timeout_ms=321)

    assert timeouts == [321]


def test_manager_handle_forwards_to_public_api(
    monkeypatch, tmp_path, test_data
) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)
    record = registry.activate_manager_record(
        registry.reserve_manager_record(host="localhost").internal_id,
        port=45555,
        watch_port=45556,
    )
    manager_mod = erlab.interactive.imagetool.manager
    handle = manager_mod.managers[record.index]
    calls: list[tuple[str, tuple[typing.Any, ...], dict[str, typing.Any]]] = []

    def record_call(name: str, result: typing.Any):
        def inner(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            calls.append((name, args, kwargs))
            return result

        return inner

    monkeypatch.setattr(manager_mod, "show_in_manager", record_call("show", "shown"))
    monkeypatch.setattr(manager_mod, "load_in_manager", record_call("load", "loaded"))
    monkeypatch.setattr(manager_mod, "replace_data", record_call("replace", "done"))
    monkeypatch.setattr(manager_mod, "fetch", record_call("fetch", test_data))
    monkeypatch.setattr(manager_mod, "watch", record_call("watch", ("data",)))
    monkeypatch.setattr(
        manager_mod, "set_default_manager", record_call("use", record.index)
    )

    assert handle.show(test_data, link=True) == "shown"
    assert handle.load(("path.dat",), loader_name="example") == "loaded"
    assert handle.replace(0, test_data) == "done"
    xr.testing.assert_identical(handle.fetch(0), test_data)
    assert handle.watch("data") == ("data",)
    assert handle.use() == record.index

    assert calls[0][0] == "show"
    assert calls[0][1][0] is test_data
    assert calls[0][2] == {"target": record.index, "link": True}
    assert calls[1] == (
        "load",
        (("path.dat",),),
        {"loader_name": "example", "target": record.index},
    )
    assert calls[2][0] == "replace"
    assert calls[2][1][0] == 0
    assert calls[2][1][1] is test_data
    assert calls[2][2] == {"target": record.index}
    assert calls[3] == ("fetch", (0,), {"target": record.index})
    assert calls[4] == ("watch", ("data",), {"target": record.index})
    assert calls[5] == ("use", (record.index,), {})


def test_manager_registry_hides_starting_records(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)

    record = registry.reserve_manager_record(host="localhost")

    assert record.state == "starting"
    assert registry.live_manager_records() == ()
    assert registry.manager_selection_info()["reason"] == "none"

    ready_record = registry.activate_manager_record(
        record.internal_id, port=45555, watch_port=45556
    )

    assert ready_record.state == "ready"
    assert [item.index for item in registry.live_manager_records()] == [record.index]
    assert registry.manager_selection_info()["reason"] == "single"


def test_manager_registry_handles_invalid_records_and_paths(
    monkeypatch, tmp_path
) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setenv("ITOOL_MANAGER_REGISTRY", str(tmp_path / "custom.json"))

    assert registry._default_registry_path() == tmp_path / "custom.json"
    assert registry._ManagerRecord.from_dict({"state": "unknown"}) is None
    assert registry._ManagerRecord.from_dict({"state": "ready"}) is None

    registry._REGISTRY_PATH.write_text("{", encoding="utf-8")
    assert registry._read_records_unlocked() == []

    registry._REGISTRY_PATH.write_text('{"records": []}', encoding="utf-8")
    assert registry._read_records_unlocked() == []

    record = registry._ManagerRecord(
        internal_id="abc",
        index=0,
        pid=123,
        host="localhost",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
        heartbeat=100.0,
    )
    registry._REGISTRY_PATH.write_text(
        json.dumps([1, {"state": "unknown"}, dataclasses.asdict(record)]),
        encoding="utf-8",
    )

    assert registry._read_records_unlocked() == [record]


def test_manager_registry_write_open_and_commit_failures(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)

    class _OpenFailSaveFile:
        def __init__(self, _path: str) -> None:
            return None

        def open(self, _mode) -> bool:
            return False

        def errorString(self) -> str:
            return "open failed"

    monkeypatch.setattr(registry.QtCore, "QSaveFile", _OpenFailSaveFile)
    with pytest.raises(registry.ImageToolManagerRegistryError, match="open failed"):
        registry._write_records_unlocked([])

    class _CommitFailSaveFile:
        def __init__(self, _path: str) -> None:
            return None

        def open(self, _mode) -> bool:
            return True

        def write(self, payload: bytes) -> int:
            return len(payload)

        def commit(self) -> bool:
            return False

        def errorString(self) -> str:
            return "commit failed"

    monkeypatch.setattr(registry.QtCore, "QSaveFile", _CommitFailSaveFile)
    with pytest.raises(registry.ImageToolManagerRegistryError, match="commit failed"):
        registry._write_records_unlocked([])


def test_manager_registry_process_and_activity_helpers(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    record = registry._ManagerRecord(
        internal_id="abc",
        index=0,
        pid=999999,
        host="localhost",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
        heartbeat=100.0,
    )

    assert registry._pid_exists(0) is False
    with pytest.raises(ValueError, match="Manager index must be >= 0"):
        registry._normalize_manager_index(-1, label="index")

    monkeypatch.setattr(registry.os, "kill", lambda *_args: None)
    assert registry._pid_exists(999999) is True
    monkeypatch.setattr(
        registry.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError),
    )
    assert registry._pid_exists(999999) is False
    monkeypatch.setattr(
        registry.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(PermissionError),
    )
    assert registry._pid_exists(999999) is True
    monkeypatch.setattr(
        registry.os, "kill", lambda *_args: (_ for _ in ()).throw(OSError)
    )
    assert registry._pid_exists(999999) is False

    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: False)
    assert registry._record_is_active(record, now=101.0) is False

    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: False)
    assert registry._record_is_active(record, now=101.0) is True
    assert registry._record_is_active(record, now=200.0) is False


def test_manager_registry_does_not_reuse_holes(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)

    records = [
        registry.activate_manager_record(
            registry.reserve_manager_record(host="localhost").internal_id,
            port=45555 + idx * 2,
            watch_port=45556 + idx * 2,
        )
        for idx in range(3)
    ]

    registry.unregister_manager_record(records[1].internal_id)
    new_record = registry.reserve_manager_record(host="localhost")

    assert [record.index for record in records] == [0, 1, 2]
    assert new_record.index == 3

    for record in (*records[::2], new_record):
        registry.unregister_manager_record(record.internal_id)

    assert registry.reserve_manager_record(host="localhost").index == 0


def test_manager_registry_removes_stale_starting_records(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    current_time = 1000.0
    monkeypatch.setattr(registry.time, "time", lambda: current_time)

    registry.reserve_manager_record(host="localhost")
    current_time += registry._STARTUP_GRACE_S + 1.0

    assert registry.live_manager_records() == ()
    with registry._registry_lock():
        assert registry._read_records_unlocked() == []


def test_manager_registry_target_resolution_edges(monkeypatch, tmp_path) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    monkeypatch.setattr(registry, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(registry, "_is_tcp_port_open", lambda *_args: True)

    with pytest.raises(ValueError, match="No live ImageTool manager"):
        registry.set_default_manager(0)
    with pytest.raises(registry.ImageToolManagerNotFoundError, match="not running"):
        registry.resolve_manager_record()

    records = [
        registry.activate_manager_record(
            registry.reserve_manager_record(host="localhost").internal_id,
            port=45555 + idx * 2,
            watch_port=45556 + idx * 2,
        )
        for idx in range(2)
    ]

    with pytest.raises(registry.ImageToolManagerNotFoundError, match="index 99"):
        registry.resolve_manager_record(99)
    with pytest.raises(registry.ImageToolManagerAmbiguousError, match="Multiple"):
        registry.resolve_manager_record()

    registry._default_manager_index = 99
    assert registry.get_default_manager() is None
    registry._default_manager_index = 99
    info = registry.manager_selection_info()
    assert info["reason"] == "multiple"
    assert info["default_index"] is None

    registry.set_default_manager(records[0].index)
    assert registry.resolve_manager_record() == records[0]
    with registry.use_manager(records[1].index):
        assert registry.get_default_manager() == records[1].index
    assert registry.get_default_manager() == records[0].index

    with (
        pytest.raises(RuntimeError, match="boom"),
        registry.use_manager(records[1].index),
    ):
        raise RuntimeError("boom")
    assert registry.get_default_manager() == records[0].index


def test_manager_registry_write_failure_preserves_existing_file(
    monkeypatch, tmp_path
) -> None:
    registry = _use_isolated_manager_registry(monkeypatch, tmp_path)
    registry._REGISTRY_PATH.write_text("[]\n", encoding="utf-8")
    cancelled: list[bool] = []

    class _FailingSaveFile:
        def __init__(self, _path: str) -> None:
            return None

        def open(self, _mode) -> bool:
            return True

        def write(self, payload: bytes) -> int:
            return len(payload) - 1

        def cancelWriting(self) -> None:
            cancelled.append(True)

        def errorString(self) -> str:
            return "partial write"

    monkeypatch.setattr(registry.QtCore, "QSaveFile", _FailingSaveFile)

    with pytest.raises(registry.ImageToolManagerRegistryError, match="partial write"):
        registry._write_records_unlocked([])

    assert cancelled == [True]
    assert registry._REGISTRY_PATH.read_text(encoding="utf-8") == "[]\n"


def test_manager_registry_lock_assigns_unique_concurrent_indexes(tmp_path) -> None:
    registry_path = tmp_path / "concurrent-managers.json"
    worker_code = """
import pathlib
import sys
import time

import erlab.interactive.imagetool.manager._registry as registry

registry._REGISTRY_PATH = pathlib.Path(sys.argv[1])
registry._LOCK_PATH = registry._REGISTRY_PATH.with_suffix(
    registry._REGISTRY_PATH.suffix + ".lock"
)
record = registry.reserve_manager_record(host="localhost")
print(record.index, flush=True)
time.sleep(1.0)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker_code, str(registry_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]

    outputs = [process.communicate(timeout=15) for process in processes]
    indexes = sorted(int(stdout.strip()) for stdout, _stderr in outputs)

    assert indexes == [0, 1, 2]
    for process, (_stdout, stderr) in zip(processes, outputs, strict=True):
        assert process.returncode == 0, stderr


def test_manager_registry_heartbeat_runs_refresh_off_gui_thread(
    qtbot,
    monkeypatch,
) -> None:
    gui_thread = QtCore.QThread.currentThread()
    calls: list[tuple[QtCore.QThread, str, str | None, int]] = []

    def refresh_record(
        internal_id: str,
        *,
        workspace_path: str | None | object = None,
        lock_timeout_ms: int,
    ) -> None:
        calls.append(
            (
                QtCore.QThread.currentThread(),
                internal_id,
                workspace_path if isinstance(workspace_path, str) else None,
                lock_timeout_ms,
            )
        )

    monkeypatch.setattr(manager_heartbeat, "refresh_manager_record", refresh_record)
    controller = manager_heartbeat._RegistryHeartbeatController("manager-id")
    try:
        controller.request_refresh("workspace.itws", coalesce_if_busy=False)
        qtbot.waitUntil(lambda: bool(calls) and not controller.is_busy, timeout=2000)
    finally:
        controller.stop()

    assert len(calls) == 1
    assert calls[0][0] != gui_thread
    assert calls[0][1:] == (
        "manager-id",
        "workspace.itws",
        manager_heartbeat._HEARTBEAT_LOCK_TIMEOUT_MS,
    )


def test_manager_registry_heartbeat_skips_ticks_and_coalesces_workspace_refreshes(
    qtbot,
    monkeypatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[tuple[str | None, int]] = []

    def refresh_record(
        _internal_id: str,
        *,
        workspace_path: str | None | object = None,
        lock_timeout_ms: int,
    ) -> None:
        calls.append(
            (
                workspace_path if isinstance(workspace_path, str) else None,
                lock_timeout_ms,
            )
        )
        if len(calls) == 1:
            started.set()
            release.wait(2)

    monkeypatch.setattr(manager_heartbeat, "refresh_manager_record", refresh_record)
    controller = manager_heartbeat._RegistryHeartbeatController("manager-id")
    try:
        controller.request_refresh("initial.itws", coalesce_if_busy=False)
        qtbot.waitUntil(started.is_set, timeout=1000)
        assert controller.is_busy

        controller.request_refresh("timer-skipped.itws", coalesce_if_busy=False)
        controller.request_refresh("workspace-updated.itws", coalesce_if_busy=True)
        release.set()

        qtbot.waitUntil(
            lambda: len(calls) == 2 and not controller.is_busy,
            timeout=2000,
        )
    finally:
        release.set()
        controller.stop()

    assert calls == [
        ("initial.itws", manager_heartbeat._HEARTBEAT_LOCK_TIMEOUT_MS),
        ("workspace-updated.itws", manager_heartbeat._HEARTBEAT_LOCK_TIMEOUT_MS),
    ]


def test_manager_registry_heartbeat_logs_failures_without_modal_alerts(
    qtbot,
    monkeypatch,
    caplog,
) -> None:
    failures = [
        manager_registry.ImageToolManagerRegistryLockError("locked"),
        manager_registry.ImageToolManagerRegistryError("write failed"),
    ]

    def refresh_record(
        _internal_id: str,
        *,
        workspace_path: str | None | object = None,
        lock_timeout_ms: int,
    ) -> None:
        raise failures.pop(0)

    monkeypatch.setattr(manager_heartbeat, "refresh_manager_record", refresh_record)
    controller = manager_heartbeat._RegistryHeartbeatController("manager-id")
    try:
        with caplog.at_level(logging.DEBUG, logger=manager_heartbeat.logger.name):
            controller.request_refresh(None, coalesce_if_busy=False)
            qtbot.waitUntil(lambda: not controller.is_busy, timeout=1000)
            controller.request_refresh(None, coalesce_if_busy=False)
            qtbot.waitUntil(lambda: not controller.is_busy, timeout=1000)
    finally:
        controller.stop()

    debug_records = [
        record
        for record in caplog.records
        if record.message.startswith(
            "Could not lock ImageTool manager registry for heartbeat refresh"
        )
    ]
    warning_records = [
        record
        for record in caplog.records
        if record.message.startswith(
            "Could not refresh ImageTool manager registry record"
        )
    ]
    assert len(debug_records) == 1
    assert debug_records[0].suppress_ui_alert is True
    assert len(warning_records) == 1
    assert warning_records[0].suppress_ui_alert is True


def test_manager_registry_heartbeat_stop_is_safe_while_refresh_is_in_flight(
    qtbot,
    monkeypatch,
) -> None:
    started = threading.Event()

    def refresh_record(
        _internal_id: str,
        *,
        workspace_path: str | None | object = None,
        lock_timeout_ms: int,
    ) -> None:
        started.set()
        QtCore.QThread.msleep(50)

    monkeypatch.setattr(manager_heartbeat, "refresh_manager_record", refresh_record)
    controller = manager_heartbeat._RegistryHeartbeatController("manager-id")

    controller.request_refresh(None, coalesce_if_busy=False)
    qtbot.waitUntil(started.is_set, timeout=1000)
    controller.stop()

    assert controller.is_stopping
    assert not controller.is_busy
    assert controller._thread is None
    assert controller._worker is None


def test_manager_registry_heartbeat_controller_edge_paths(
    qtbot,
    caplog,
) -> None:
    controller = manager_heartbeat._RegistryHeartbeatController("manager-id")
    try:
        controller._in_flight_generation = 3
        controller._refresh_finished(object())
        assert controller.is_busy

        controller._refresh_finished(manager_heartbeat._HeartbeatResult(2, ok=True))
        assert controller.is_busy

        with caplog.at_level(logging.WARNING, logger=manager_heartbeat.logger.name):
            controller._handle_result(
                manager_heartbeat._HeartbeatResult(3, ok=False, lock_failed=True)
            )
            controller._consecutive_lock_failures = (
                manager_heartbeat._LOCK_FAILURE_WARNING_THRESHOLD - 1
            )
            controller._handle_result(
                manager_heartbeat._HeartbeatResult(
                    3,
                    ok=False,
                    error_text="locked",
                    lock_failed=True,
                )
            )

        assert any(
            record.message.startswith("Could not lock ImageTool manager registry for")
            and record.suppress_ui_alert is True
            for record in caplog.records
        )

        with caplog.at_level(logging.DEBUG, logger=manager_heartbeat.logger.name):
            controller._handle_result(
                manager_heartbeat._HeartbeatResult(
                    3,
                    ok=False,
                    error_text="write failed",
                )
            )
            controller._handle_result(
                manager_heartbeat._HeartbeatResult(
                    3,
                    ok=False,
                    error_text="write failed again",
                )
            )

        assert any(
            record.levelno == logging.DEBUG
            and record.message.startswith(
                "Could not refresh ImageTool manager registry record"
            )
            and record.suppress_ui_alert is True
            for record in caplog.records
        )

        controller.stop()
        controller.request_refresh("ignored.itws", coalesce_if_busy=True)
        assert not controller.is_busy

        controller._stopping = False
        with caplog.at_level(logging.WARNING, logger=manager_heartbeat.logger.name):
            controller.request_refresh("missing-thread.itws", coalesce_if_busy=False)

        assert any(
            record.message
            == "ImageTool manager registry heartbeat thread is unavailable"
            and record.suppress_ui_alert is True
            for record in caplog.records
        )
        controller._disconnect_worker_signals()
    finally:
        controller.stop()


def test_manager_registry_heartbeat_retains_orphan_until_thread_finishes(
    qtbot,
) -> None:
    thread = QtCore.QThread()
    worker = manager_heartbeat._RegistryHeartbeatWorker()
    worker.moveToThread(thread)
    thread.start()
    manager_heartbeat._RegistryHeartbeatController._retain_orphaned_thread(
        thread,
        worker,
    )
    assert (thread, worker) in manager_heartbeat._ORPHANED_HEARTBEAT_OBJECTS

    thread.quit()
    assert thread.wait(1000)
    qtbot.waitUntil(
        lambda: (thread, worker) not in manager_heartbeat._ORPHANED_HEARTBEAT_OBJECTS,
        timeout=1000,
    )


def test_manager_registry_heartbeat_logs_thread_stop_timeout(
    qtbot,
    monkeypatch,
    caplog,
) -> None:
    controller = manager_heartbeat._RegistryHeartbeatController("manager-id")
    thread = controller._thread
    assert thread is not None
    original_wait = thread.wait
    monkeypatch.setattr(thread, "wait", lambda _timeout_ms: False)

    with caplog.at_level(logging.WARNING, logger=manager_heartbeat.logger.name):
        controller.stop()

    assert controller._thread is None
    assert controller._worker is None
    assert any(
        record.message
        == "ImageTool manager registry heartbeat thread did not stop promptly"
        and record.suppress_ui_alert is True
        for record in caplog.records
    )

    monkeypatch.setattr(thread, "wait", original_wait)
    thread.quit()
    assert thread.wait(1000)
    qtbot.waitUntil(
        lambda: all(
            orphan_thread is not thread
            for orphan_thread, _worker in manager_heartbeat._ORPHANED_HEARTBEAT_OBJECTS
        ),
        timeout=1000,
    )


def test_itool_magic_manager_target_normalization() -> None:
    assert _normalize_manager_target_args("-m data") == "-m data"
    assert _normalize_manager_target_args("-m 1 data") == "-m --manager-index 1 data"
    assert (
        _normalize_manager_target_args("--manager=2 data")
        == "--manager --manager-index 2 data"
    )


def test_itool_manager_accepts_index_like_values(monkeypatch, test_data) -> None:
    calls: list[dict[str, typing.Any]] = []

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "is_running",
        lambda target=None: True,
    )
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "show_in_manager",
        lambda _data, **kwargs: calls.append(kwargs),
    )

    itool(test_data, manager=np.int64(2))
    itool(test_data, manager=True)

    assert calls[0]["target"] == 2
    assert calls[1]["target"] is None


def test_itool_manager_invalid_index_object_and_unavailable_fallback(
    monkeypatch, test_data
) -> None:
    warnings: list[str] = []
    running_targets: list[int | None] = []

    class _DummyImageTool:
        def __init__(self, data, **_kwargs) -> None:
            self.data = data

        def show(self) -> None:
            return None

        def activateWindow(self) -> None:
            return None

        def raise_(self) -> None:
            return None

    monkeypatch.setattr(itool_mod, "_parse_input", lambda _data: [test_data])
    monkeypatch.setattr(erlab.interactive.imagetool, "ImageTool", _DummyImageTool)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "is_running",
        lambda target=None: running_targets.append(target) or False,
    )
    monkeypatch.setattr(
        erlab.utils.misc,
        "emit_user_level_warning",
        lambda message: warnings.append(message),
    )

    direct_tool = itool(test_data, manager=object(), execute=False)
    fallback_tool = itool(test_data, manager=3, execute=False)

    assert isinstance(direct_tool, _DummyImageTool)
    assert isinstance(fallback_tool, _DummyImageTool)
    assert running_targets == [3]
    assert warnings == [
        "The manager is not running. Opening the ImageTool window(s) directly."
    ]


def test_manager_target_validation_rejects_bool(monkeypatch, tmp_path) -> None:
    _use_isolated_manager_registry(monkeypatch, tmp_path)

    assert erlab.interactive.imagetool.manager.is_running(np.int64(0)) is False
    with pytest.raises(TypeError, match="Manager target must be an integer"):
        erlab.interactive.imagetool.manager.is_running(True)


def test_multi_manager_integer_targets(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    manager_mod = erlab.interactive.imagetool.manager
    with manager_context() as manager0:
        manager1 = ImageToolManager()
        qtbot.addWidget(manager1, before_close_func=lambda w: w.remove_all_tools())
        try:
            qtbot.wait_until(
                lambda: (
                    manager0.server.isRunning()
                    and manager0.watcher_server.isRunning()
                    and manager1.server.isRunning()
                    and manager1.watcher_server.isRunning()
                ),
                timeout=5000,
            )

            info = manager_mod.manager_selection_info()
            assert info["reason"] == "multiple"
            assert info["needs_selection"] is True
            assert [item["index"] for item in info["managers"]] == [
                manager0.manager_index,
                manager1.manager_index,
            ]

            test_data.qshow(manager=manager0.manager_index)
            qtbot.wait_until(lambda: manager0.ntools == 1, timeout=5000)
            assert manager1.ntools == 0

            (test_data + 1).qshow(manager=manager1.manager_index)
            qtbot.wait_until(lambda: manager1.ntools == 1, timeout=5000)
            assert manager0.ntools == 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fetch, 0, target=manager1.manager_index)
                qtbot.waitUntil(lambda: fut.done(), timeout=10000)
                fetched = fut.result()
            xr.testing.assert_identical(fetched, test_data + 1)
        finally:
            manager1.remove_all_tools()
            manager1.close()
            qtbot.wait_until(
                lambda: (
                    not manager1.server.isRunning()
                    and not manager1.watcher_server.isRunning()
                ),
                timeout=5000,
            )
            manager1.deleteLater()


def test_watcher_server_run_stops_cleanly_without_pending_payload(monkeypatch) -> None:
    class _DummySocket:
        def __init__(self) -> None:
            self.bound: list[str] = []
            self.closed = False
            self.sent: list[dict[str, str]] = []

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, address: str) -> None:
            self.bound.append(address)

        def send_json(self, payload: dict[str, str]) -> None:
            self.sent.append(payload)

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _DummySocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _DummySocket:
            return self._socket

    class _DummyCondition:
        def __init__(self, server: _WatcherServer) -> None:
            self.server = server
            self.calls = 0

        def __enter__(self) -> typing.Self:
            return self

        def __exit__(self, *args) -> None:
            return None

        def wait(self, _timeout: float) -> bool:
            self.calls += 1
            self.server.stopped.set()
            return True

        def notify_all(self) -> None:
            return None

    socket = _DummySocket()
    monkeypatch.setattr(
        zmq.Context, "instance", staticmethod(lambda: _DummyContext(socket))
    )

    server = _WatcherServer()
    condition = _DummyCondition(server)
    server._condition = condition

    server.run()

    assert condition.calls == 1
    assert socket.bound == [f"tcp://*:{erlab.interactive.imagetool.manager.PORT_WATCH}"]
    assert socket.sent == []
    assert socket.closed


def test_manager_server_wait_until_bound_errors() -> None:
    watcher_server = _WatcherServer()
    with pytest.raises(TimeoutError, match="watcher server"):
        watcher_server.wait_until_bound(timeout_ms=1)
    watcher_error = RuntimeError("watcher bind failed")
    watcher_server._bind_error = watcher_error
    watcher_server._bound_event.set()
    with pytest.raises(RuntimeError, match="Watcher server failed") as exc_info:
        watcher_server.wait_until_bound(timeout_ms=1)
    assert exc_info.value.__cause__ is watcher_error

    manager = manager_server._ManagerServer()
    with pytest.raises(TimeoutError, match="manager server"):
        manager.wait_until_bound(timeout_ms=1)
    manager_error = RuntimeError("manager bind failed")
    manager._bind_error = manager_error
    manager._bound_event.set()
    with pytest.raises(RuntimeError, match="Manager server failed") as exc_info:
        manager.wait_until_bound(timeout_ms=1)
    assert exc_info.value.__cause__ is manager_error


def test_server_thread_stop_wait_is_cooperative(monkeypatch) -> None:
    class _FinishingThread:
        def __init__(self) -> None:
            self.running = True
            self.waits: list[int] = []

        def isRunning(self) -> bool:
            return self.running

        def wait(self, timeout_ms: int) -> bool:
            self.waits.append(timeout_ms)
            self.running = False
            return True

    finishing_thread = _FinishingThread()
    assert manager_server._wait_for_qthread_to_stop(finishing_thread, 100)
    assert finishing_thread.waits
    assert 0 < finishing_thread.waits[0] <= 10

    class _RunningThread:
        def isRunning(self) -> bool:
            return True

        def wait(self, timeout_ms: int) -> bool:
            raise AssertionError("wait should not be called after the timeout")

    monotonic_values = iter([0.0, 0.002])
    monkeypatch.setattr(
        manager_server.time, "monotonic", lambda: next(monotonic_values)
    )

    assert not manager_server._wait_for_qthread_to_stop(_RunningThread(), 1)


def test_server_stop_logs_when_cooperative_wait_times_out(monkeypatch, caplog) -> None:
    caplog.set_level(logging.WARNING, logger=manager_server.__name__)
    monkeypatch.setattr(
        manager_server, "_wait_for_qthread_to_stop", lambda *_args: False
    )
    monkeypatch.setattr(_WatcherServer, "isRunning", lambda _self: True)
    monkeypatch.setattr(manager_server._ManagerServer, "isRunning", lambda _self: True)

    watcher_server = _WatcherServer()
    watcher_server.stop(timeout_ms=1)

    manager = manager_server._ManagerServer()
    manager.stop(timeout_ms=1)

    assert "Watcher server did not stop within timeout" in caplog.text
    assert "Manager server did not stop within timeout" in caplog.text


def test_manager_server_stop_wakes_receive_loop(qtbot) -> None:
    server = manager_server._ManagerServer(port=0)
    server.start()
    try:
        assert server.wait_until_bound(timeout_ms=5000) > 0
        server.stop(timeout_ms=1000)
        assert not server.isRunning()
    finally:
        if server.isRunning():
            server.stop(timeout_ms=1000)
        server.wait(1000)
        server.deleteLater()


def test_widgets_controller_stop_servers_disconnects_and_deletes() -> None:
    class _ServerDouble(QtCore.QObject):
        sigReceived = QtCore.Signal(list, dict)
        sigLoadRequested = QtCore.Signal(list, str, dict)
        sigReplaceRequested = QtCore.Signal(list, list)
        sigWatchedVarChanged = QtCore.Signal(str, str, object, object)
        sigDataRequested = QtCore.Signal(object)
        sigWatchInfoRequested = QtCore.Signal()
        sigRemoveIndex = QtCore.Signal(int)
        sigShowIndex = QtCore.Signal(int)
        sigRemoveUID = QtCore.Signal(str)
        sigShowUID = QtCore.Signal(str)
        sigUnwatchUID = QtCore.Signal(str)

        def __init__(self) -> None:
            super().__init__()
            self.running = True
            self.deleted = False
            self.return_values: list[object] = []

        def isRunning(self) -> bool:
            return self.running

        def stop(self) -> None:
            self.running = False

        def deleteLater(self) -> None:
            self.deleted = True

        def set_return_value(self, value: object) -> None:
            self.return_values.append(value)

    class _WatcherDouble(QtCore.QObject):
        def __init__(self) -> None:
            super().__init__()
            self.running = True
            self.deleted = False
            self.parameters: list[tuple[str, str, str]] = []

        def isRunning(self) -> bool:
            return self.running

        def stop(self) -> None:
            self.running = False

        def deleteLater(self) -> None:
            self.deleted = True

        def send_parameters(self, varname: str, uid: str, event: str) -> None:
            self.parameters.append((varname, uid, event))

    class _ManagerDouble(QtCore.QObject):
        _sigReplyData = QtCore.Signal(object)
        _sigWatchedDataEdited = QtCore.Signal(str, str, str)

        def __init__(self) -> None:
            super().__init__()
            self.server = _ServerDouble()
            self.watcher_server = _WatcherDouble()
            self.calls: list[tuple[str, object]] = []

        def _record(self, name: str, *args) -> None:
            self.calls.append((name, args))

        def _data_recv(self, *args) -> None:
            self._record("data_recv", *args)

        def _data_load(self, *args) -> None:
            self._record("data_load", *args)

        def _data_replace(self, *args) -> None:
            self._record("data_replace", *args)

        def _send_imagetool_data(self, *args) -> None:
            self._record("send_imagetool_data", *args)

        def _send_watch_info(self) -> None:
            self._record("send_watch_info")

        def remove_imagetool(self, *args) -> None:
            self._record("remove_imagetool", *args)

        def show_imagetool(self, *args) -> None:
            self._record("show_imagetool", *args)

        def _remove_watched(self, *args) -> None:
            self._record("remove_watched", *args)

        def _show_watched(self, *args) -> None:
            self._record("show_watched", *args)

        def _data_unwatch(self, *args) -> None:
            self._record("data_unwatch", *args)

        def _data_watched_update(self, *args) -> None:
            self._record("data_watched_update", *args)

    manager = _ManagerDouble()
    server = manager.server
    watcher_server = manager.watcher_server

    for signal, slot in (
        (server.sigReceived, manager._data_recv),
        (server.sigLoadRequested, manager._data_load),
        (server.sigReplaceRequested, manager._data_replace),
        (server.sigDataRequested, manager._send_imagetool_data),
        (server.sigWatchInfoRequested, manager._send_watch_info),
        (manager._sigReplyData, server.set_return_value),
        (server.sigRemoveIndex, manager.remove_imagetool),
        (server.sigShowIndex, manager.show_imagetool),
        (server.sigRemoveUID, manager._remove_watched),
        (server.sigShowUID, manager._show_watched),
        (server.sigUnwatchUID, manager._data_unwatch),
        (server.sigWatchedVarChanged, manager._data_watched_update),
        (manager._sigWatchedDataEdited, watcher_server.send_parameters),
    ):
        signal.connect(slot)

    manager_widgets._WidgetsController(manager)._stop_servers()

    assert not server.running
    assert server.deleted
    assert not watcher_server.running
    assert watcher_server.deleted

    server.sigRemoveIndex.emit(0)
    manager._sigReplyData.emit("value")
    manager._sigWatchedDataEdited.emit("data", "uid", "updated")

    assert manager.calls == []
    assert server.return_values == []
    assert watcher_server.parameters == []


def test_manager_server_bind_failures_close_socket(monkeypatch) -> None:
    class _FailingSocket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, _address: str) -> None:
            raise RuntimeError("bind failed")

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _FailingSocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _FailingSocket:
            return self._socket

    watcher_socket = _FailingSocket()
    monkeypatch.setattr(
        zmq.Context,
        "instance",
        staticmethod(lambda: _DummyContext(watcher_socket)),
    )
    watcher_server = _WatcherServer(port=45556)
    watcher_server.run()
    assert watcher_server._bound_event.is_set()
    assert isinstance(watcher_server._bind_error, RuntimeError)
    assert watcher_socket.closed

    manager_socket = _FailingSocket()
    monkeypatch.setattr(
        zmq.Context,
        "instance",
        staticmethod(lambda: _DummyContext(manager_socket)),
    )
    manager = manager_server._ManagerServer(port=45555)
    manager.run()
    assert manager._bound_event.is_set()
    assert isinstance(manager._bind_error, RuntimeError)
    assert manager_socket.closed


def test_manager_server_wait_for_return_value_stops_cleanly() -> None:
    manager = manager_server._ManagerServer()
    manager.stopped.set()

    assert manager._wait_for_return_value() is manager_server._UNSET


def test_manager_server_run_returns_watch_info(monkeypatch) -> None:
    sent: list[dict[str, typing.Any]] = []

    class _DummySocket:
        def __init__(self) -> None:
            self.closed = False
            self.bound: list[str] = []

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, address: str) -> None:
            self.bound.append(address)

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _DummySocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _DummySocket:
            return self._socket

    socket = _DummySocket()
    manager = manager_server._ManagerServer(port=45555)
    watch_info = {"workspace_link_id": "workspace-1", "watched": []}

    monkeypatch.setattr(
        zmq.Context, "instance", staticmethod(lambda: _DummyContext(socket))
    )
    monkeypatch.setattr(
        manager_server,
        "_recv_multipart",
        lambda _socket: {"packet_type": "command", "command": "watch-info"},
    )
    monkeypatch.setattr(
        manager_server,
        "_send_multipart",
        lambda _socket, payload, **_kwargs: sent.append(payload),
    )
    manager.sigWatchInfoRequested.connect(
        lambda: (manager.set_return_value(watch_info), manager.stopped.set())
    )

    manager.run()

    assert sent == [{"status": "ok", "watch_info": watch_info}]
    assert socket.bound == ["tcp://*:45555"]
    assert socket.closed


def test_manager_server_run_errors_when_data_request_stops(monkeypatch) -> None:
    sent: list[dict[str, typing.Any]] = []

    class _DummySocket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, _address: str) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _DummySocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _DummySocket:
            return self._socket

    socket = _DummySocket()
    manager = manager_server._ManagerServer(port=45555)

    monkeypatch.setattr(
        zmq.Context, "instance", staticmethod(lambda: _DummyContext(socket))
    )
    monkeypatch.setattr(
        manager_server,
        "_recv_multipart",
        lambda _socket: {
            "packet_type": "command",
            "command": "get-data",
            "command_arg": 0,
        },
    )
    monkeypatch.setattr(
        manager_server,
        "_send_multipart",
        lambda _socket, payload, **_kwargs: sent.append(payload),
    )
    manager.sigDataRequested.connect(lambda _arg: manager.stopped.set())

    manager.run()

    assert sent == []
    assert socket.closed


def test_manager_server_run_errors_when_watch_info_request_stops(monkeypatch) -> None:
    sent: list[dict[str, typing.Any]] = []

    class _DummySocket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, _address: str) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _DummyContext:
        def __init__(self, socket: _DummySocket) -> None:
            self._socket = socket

        def socket(self, *_args) -> _DummySocket:
            return self._socket

    socket = _DummySocket()
    manager = manager_server._ManagerServer(port=45555)

    monkeypatch.setattr(
        zmq.Context, "instance", staticmethod(lambda: _DummyContext(socket))
    )
    monkeypatch.setattr(
        manager_server,
        "_recv_multipart",
        lambda _socket: {"packet_type": "command", "command": "watch-info"},
    )
    monkeypatch.setattr(
        manager_server,
        "_send_multipart",
        lambda _socket, payload, **_kwargs: sent.append(payload),
    )
    manager.sigWatchInfoRequested.connect(lambda: manager.stopped.set())

    manager.run()

    assert sent == []
    assert socket.closed


def test_manager_server_client_helpers_target_socket_branches(
    monkeypatch, tmp_path, test_data
) -> None:
    calls: list[tuple[typing.Any, dict[str, typing.Any]]] = []
    record = manager_registry._ManagerRecord(
        internal_id="abc",
        index=2,
        pid=123,
        host="localhost",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
        heartbeat=100.0,
    )

    monkeypatch.setattr(erlab.interactive.imagetool.manager, "_manager_instance", None)
    monkeypatch.setattr(
        manager_server, "resolve_manager_record", lambda target=None, **_kwargs: record
    )

    def _query_zmq(payload, **kwargs):
        calls.append((payload, kwargs))
        if (
            isinstance(payload, manager_server.CommandPacket)
            and payload.command == "watch-info"
        ):
            return Response(status="ok", watch_info={"watched": []})
        return Response(status="ok", data=test_data)

    monkeypatch.setattr(manager_server, "_query_zmq", _query_zmq)

    path = tmp_path / "data.dat"
    path.write_text("data", encoding="utf-8")
    manager_server.load_in_manager((path,), target=2)
    manager_server.show_in_manager(None, target=2)
    manager_server.replace_data(0, test_data, target=2)
    manager_server._watch_data("data", "uid", test_data, show=True, target=2)
    assert manager_server._unwatch_data("uid", target=2).status == "ok"
    assert manager_server._unwatch_data("uid", remove=True, target=2).status == "ok"
    xr.testing.assert_identical(manager_server.fetch("uid", target=2), test_data)
    assert manager_server.watch_info(target=2) == {"watched": []}
    with pytest.warns(DeprecationWarning, match="watch_data"):
        manager_server.watch_data("data", "uid", test_data)
    with pytest.warns(DeprecationWarning, match="unwatch_data"):
        assert manager_server.unwatch_data("uid").status == "ok"

    with pytest.raises(FileNotFoundError):
        manager_server.load_in_manager((tmp_path / "missing.dat",), target=2)
    with pytest.raises(ValueError, match="Mismatch"):
        manager_server.replace_data([0, 1], test_data, target=2)

    packet_types = [payload.packet_type for payload, _kwargs in calls]
    assert packet_types == [
        "open",
        "add",
        "replace",
        "watch",
        "command",
        "command",
        "command",
        "command",
        "command",
        "watch",
        "command",
    ]
    assert all(kwargs["record"] == record for _payload, kwargs in calls)


def test_startup_file_forwarding_uses_private_timeout(monkeypatch, tmp_path) -> None:
    calls: list[tuple[typing.Any, dict[str, typing.Any]]] = []
    record = manager_registry._ManagerRecord(
        internal_id="mgr",
        index=2,
        pid=123,
        host="127.0.0.1",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
        heartbeat=100.0,
    )
    data_file = tmp_path / "data.dat"
    data_file.write_text("data", encoding="utf-8")

    monkeypatch.setattr(erlab.interactive.imagetool.manager, "_manager_instance", None)
    monkeypatch.setattr(
        manager_server, "resolve_manager_record", lambda target=None, **_kwargs: record
    )
    monkeypatch.setattr(
        manager_server,
        "_query_zmq",
        lambda payload, **kwargs: (
            calls.append((payload, kwargs)) or Response(status="ok")
        ),
    )

    manager_server._load_in_manager_startup(
        (data_file,),
        target=2,
        timeout_ms=1234,
    )

    [(payload, kwargs)] = calls
    assert payload.packet_type == "open"
    assert payload.loader_name == "ask"
    assert payload.filename_list == [str(data_file)]
    assert kwargs["record"] == record
    assert kwargs["timeout_ms"] == 1234


def test_manager_handle_watch_info_delegates_to_target(monkeypatch) -> None:
    handle = manager_server.ImageToolManagerHandle(
        index=7,
        pid=123,
        host="localhost",
        port=45555,
        watch_port=45556,
        started="2026-05-08T10:00:00",
        version="3.22.0",
    )

    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "watch_info",
        lambda *, target=None: {"target": target},
    )

    assert handle.watch_info() == {"target": 7}
