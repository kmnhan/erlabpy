import threading
import typing
from collections.abc import Callable

import IPython
import numpy as np
import pytest
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._watcher._core as watcher_core
import erlab.interactive.imagetool.manager._watcher._ipython as watcher_ipy
from erlab.interactive.imagetool.manager import _watcher as watcher_mod
from erlab.interactive.imagetool.manager._watcher._core import _Watcher


class FakeIOLoop:
    def add_callback(self, func, *args, **kwargs):
        # Execute immediately for tests
        func(*args, **kwargs)


class FakeKernel:
    def __init__(self):
        self.io_loop = FakeIOLoop()


class FakeShell:
    def __init__(self, with_kernel=True):
        self.user_ns = {}
        if with_kernel:
            self.kernel = FakeKernel()


@pytest.fixture
def fake_shell():
    return FakeShell(with_kernel=True)


@pytest.fixture
def patch_manager(monkeypatch):
    state = {
        "watched": {},
        "last_watch_calls": [],
        "last_unwatch_calls": [],
        "fetch_map": {},
    }

    def watch_data(varname, uid, darr, show=False):
        state["watched"][uid] = darr
        state["last_watch_calls"].append((varname, uid, darr, show))

    def unwatch_data(uid, remove=False):
        state["watched"].pop(uid, None)
        state["last_unwatch_calls"].append((uid, remove))

    def fetch(uid):
        return state["fetch_map"].get(uid)

    # Patch manager attributes on the already-imported module namespace
    manager = erlab.interactive.imagetool.manager
    monkeypatch.setattr(manager, "_watch_data", watch_data, raising=False)
    monkeypatch.setattr(manager, "_unwatch_data", unwatch_data, raising=False)
    monkeypatch.setattr(manager, "watch_data", watch_data, raising=False)
    monkeypatch.setattr(manager, "unwatch_data", unwatch_data, raising=False)
    monkeypatch.setattr(manager, "fetch", fetch, raising=False)
    monkeypatch.setattr(manager, "HOST_IP", "127.0.0.1", raising=False)
    # monkeypatch.setattr(manager, "PORT_WATCH", 0, raising=False)

    return state


def _ensure_fake_thread_attr(watcher: _Watcher):
    # Ensure shutdown/__del__ won't fail due to missing _watcher_thread
    if not hasattr(watcher, "_watcher_thread"):
        watcher._watcher_thread = threading.Thread(target=lambda: None)


def test_watch_and_maybe_push_and_delete(fake_shell, patch_manager, monkeypatch):
    # Prepare DataArray in user namespace
    da1 = xr.DataArray(np.array([1, 2, 3]), dims=("x",))
    fake_shell.user_ns["a"] = da1

    watcher = _Watcher(fake_shell)
    _ensure_fake_thread_attr(watcher)

    # Avoid starting real thread
    def fake_start_thread():
        watcher._thread_started = True
        _ensure_fake_thread_attr(watcher)

    monkeypatch.setattr(watcher, "start_thread", fake_start_thread)

    # Initial watch
    watcher.watch("a")
    assert "a" in watcher.watched_vars
    assert len(patch_manager["last_watch_calls"]) == 1
    varname, uid, darr, show = patch_manager["last_watch_calls"][-1]
    assert varname == "a"
    assert show is False
    assert isinstance(darr, xr.DataArray)

    with pytest.raises(NameError, match="'asdf' not found"):
        watcher.watch("asdf")

    # Change content to trigger _maybe_push
    da2 = xr.DataArray(np.array([1, 99, 3]), dims=("x",))
    fake_shell.user_ns["a"] = da2
    watcher._last_send = 0.0  # bypass rate limit
    watcher._maybe_push()

    assert len(patch_manager["last_watch_calls"]) == 2
    _, uid2, darr2, show2 = patch_manager["last_watch_calls"][-1]
    assert uid2 == watcher.watched_vars["a"]["uid"]
    assert np.array_equal(darr2.values, da2.values)
    assert show2 is False

    # Change type -> should unwatch and remove
    fake_shell.user_ns["a"] = "not a DataArray"
    watcher._last_send = 0.0
    watcher._maybe_push()

    assert "a" not in watcher.watched_vars
    assert len(patch_manager["last_unwatch_calls"]) == 1
    unwatch_uid, remove_flag = patch_manager["last_unwatch_calls"][-1]
    assert unwatch_uid == uid
    assert remove_flag is False

    # Already watched -> should show
    fake_shell.user_ns["a"] = da2
    watcher.watch("a")
    watcher.watch("a")  # watch again
    assert len(patch_manager["last_watch_calls"]) == 4
    _, _, _, show3 = patch_manager["last_watch_calls"][-1]
    assert show3 is True

    # Try pushing non-watched var -> no action
    watcher._push_to_gui("nonwatched", xr.DataArray())

    watcher.shutdown()


def test_recv_loop_updated_event_applies_update(fake_shell, patch_manager, monkeypatch):
    # Setup watched var
    da1 = xr.DataArray(np.array([0.0, 1.0]), dims=("x",))
    fake_shell.user_ns["a"] = da1
    watcher = _Watcher(fake_shell)
    _ensure_fake_thread_attr(watcher)

    # Avoid starting internal thread in watch()
    monkeypatch.setattr(watcher, "start_thread", lambda: None)
    watcher.watch("a")
    uid = watcher.watched_vars["a"]["uid"]

    # Provide fetched value for update
    updated_da = xr.DataArray(np.array([10.0, 20.0, 30.0]), dims=("x",))
    patch_manager["fetch_map"][uid] = updated_da

    # Fake zmq context/socket to emit one 'updated' event
    class FakeSocket:
        def __init__(self):
            self._sent = False

        def setsockopt(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def recv_json(self):
            if not self._sent:
                self._sent = True
                return {"varname": "a", "uid": uid, "event": "updated"}
            raise RuntimeError("stop")  # break loop

        def close(self):
            pass

    class FakeContext:
        def socket(self, *_):
            return FakeSocket()

    # Replace Context.instance() classmethod
    monkeypatch.setattr(
        watcher_core.zmq.Context, "instance", classmethod(lambda cls: FakeContext())
    )

    # Run recv loop in a thread
    t = threading.Thread(target=watcher._recv_loop, daemon=True)
    t.start()
    t.join(timeout=2.0)

    # Verify that variable was updated from manager.fetch
    assert isinstance(fake_shell.user_ns["a"], xr.DataArray)
    assert np.array_equal(fake_shell.user_ns["a"].values, updated_da.values)

    watcher.shutdown()


def test_recv_loop_removed_event_removes_watch(fake_shell, patch_manager, monkeypatch):
    da1 = xr.DataArray(np.array([5, 6]), dims=("x",))
    fake_shell.user_ns["b"] = da1
    watcher = _Watcher(fake_shell)
    _ensure_fake_thread_attr(watcher)
    monkeypatch.setattr(watcher, "start_thread", lambda: None)
    watcher.watch("b")
    uid = watcher.watched_vars["b"]["uid"]

    class FakeSocket:
        def __init__(self):
            self._sent = False

        def setsockopt(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def recv_json(self):
            if not self._sent:
                self._sent = True
                return {"varname": "b", "uid": uid, "event": "removed"}
            raise RuntimeError("stop")

        def close(self):
            pass

    class FakeContext:
        def socket(self, *_):
            return FakeSocket()

    monkeypatch.setattr(
        watcher_core.zmq.Context, "instance", classmethod(lambda cls: FakeContext())
    )

    t = threading.Thread(target=watcher._recv_loop, daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert "b" not in watcher.watched_vars

    watcher.shutdown()


def test_stop_watching_all_with_remove(fake_shell, patch_manager, monkeypatch):
    da1 = xr.DataArray(np.array([1]), dims=("x",))
    da2 = xr.DataArray(np.array([2]), dims=("x",))
    fake_shell.user_ns.update({"a": da1, "b": da2})

    watcher = _Watcher(fake_shell)
    _ensure_fake_thread_attr(watcher)
    # Avoid starting real thread
    monkeypatch.setattr(
        watcher,
        "start_thread",
        lambda: (
            _ensure_fake_thread_attr(watcher),
            setattr(watcher, "_thread_started", True),
        ),
    )

    watcher.watch("a")
    watcher.watch("b")

    assert set(watcher.watched_vars.keys()) == {"a", "b"}

    watcher.stop_watching_all(remove=True)

    # Both should be unwatched with remove=True
    assert watcher.watched_vars == {}
    assert len(patch_manager["last_unwatch_calls"]) >= 2
    for _, remove_flag in patch_manager["last_unwatch_calls"][-2:]:
        assert remove_flag is True

    watcher._apply_update_now("asdf", "nonexistent")  # should not raise

    watcher.shutdown()


def test_watch_api_works_without_ipython_extension(patch_manager, monkeypatch):
    namespace = {
        "a": xr.DataArray(np.array([1, 2, 3]), dims=("x",)),
    }

    monkeypatch.setattr(_Watcher, "start_thread", lambda self: None)
    monkeypatch.setattr(_Watcher, "start_polling", lambda self, interval_s=0.25: None)

    try:
        watched = watcher_mod.watch("a", namespace=namespace)
        assert watched == ("a",)
        assert watcher_mod.watched_variables(namespace=namespace) == ("a",)

        namespace["a"] = xr.DataArray(np.array([9, 8, 7]), dims=("x",))
        watcher, _ = watcher_core._get_or_create_watcher(namespace=namespace)
        watcher._last_send = 0.0
        watcher_mod.maybe_push(namespace=namespace)

        assert len(patch_manager["last_watch_calls"]) == 2
        assert np.array_equal(
            patch_manager["last_watch_calls"][-1][2].values, namespace["a"].values
        )

        watcher_mod.watch("a", namespace=namespace, stop=True, remove=True)
        assert watcher_mod.watched_variables(namespace=namespace) == ()
        assert patch_manager["last_unwatch_calls"][-1][1] is True
    finally:
        watcher_mod.shutdown(namespace=namespace)


def test_watch_magic_delegates_to_watch_api(
    ip_shell: IPython.InteractiveShell, monkeypatch
):
    calls = []

    def fake_watch(*varnames, **kwargs):
        calls.append((varnames, kwargs))
        return ()

    monkeypatch.setattr(watcher_ipy, "watch", fake_watch)

    ip_shell.run_line_magic("watch", "darr")
    assert calls[-1][0] == ("darr",)
    assert calls[-1][1]["shell"] is ip_shell
    assert calls[-1][1]["stop"] is False
    assert calls[-1][1]["remove"] is False

    ip_shell.run_line_magic("watch", "-d darr")
    assert calls[-1][0] == ("darr",)
    assert calls[-1][1]["stop"] is True
    assert calls[-1][1]["remove"] is False

    ip_shell.run_line_magic("watch", "-x darr")
    assert calls[-1][0] == ("darr",)
    assert calls[-1][1]["stop"] is True
    assert calls[-1][1]["remove"] is True

    ip_shell.run_line_magic("watch", "-z")
    assert calls[-1][0] == ()
    assert calls[-1][1]["stop_all"] is True
    assert calls[-1][1]["remove"] is False

    ip_shell.run_line_magic("watch", "-xz")
    assert calls[-1][0] == ()
    assert calls[-1][1]["stop_all"] is True
    assert calls[-1][1]["remove"] is True


def test_watch_magic_lists_currently_watched_variables(
    ip_shell: IPython.InteractiveShell, monkeypatch
):
    messages = []

    monkeypatch.setattr(
        watcher_ipy, "watched_variables", lambda **kwargs: ("alpha", "beta")
    )
    monkeypatch.setattr(
        watcher_ipy,
        "_display_message",
        lambda message, html=None: messages.append((message, html)),
    )

    ip_shell.run_line_magic("watch", "")

    assert len(messages) == 1
    assert "Currently watched variables" in messages[0][0]
    assert "alpha" in messages[0][0]
    assert "beta" in messages[0][0]


def test_watch_api_fallback_namespace_without_ipython(patch_manager, monkeypatch):
    monkeypatch.setattr(watcher_ipy, "_safe_get_ipython_shell", lambda: None)
    monkeypatch.setattr(_Watcher, "start_thread", lambda self: None)
    monkeypatch.setattr(_Watcher, "start_polling", lambda self, interval_s=0.25: None)

    globals()["fallback_darr"] = xr.DataArray(np.array([1, 2, 3]), dims=("x",))

    try:
        watched = watcher_mod.watch("fallback_darr")
        assert watched == ("fallback_darr",)
        assert watcher_mod.watch() == ("fallback_darr",)

        # Trigger fallback path in maybe_push/shutdown when no shell/namespace is given.
        watcher_mod.maybe_push()
        watcher_mod.shutdown()
        assert watcher_mod.watched_variables() == ()

        # No watcher left: should be a no-op.
        watcher_mod.maybe_push()
        watcher_mod.shutdown()
    finally:
        globals().pop("fallback_darr", None)
        watcher_mod.shutdown(namespace=globals())


def test_watcher_type_error_and_push_failure_cleanup(fake_shell, monkeypatch):
    fake_shell.user_ns["not_da"] = object()
    watcher = _Watcher(fake_shell)

    with pytest.raises(TypeError, match=r"is not an xarray\.DataArray"):
        watcher.watch("not_da")

    fake_shell.user_ns["a"] = xr.DataArray(np.array([1, 2, 3]), dims=("x",))
    monkeypatch.setattr(watcher, "start_thread", lambda: None)
    monkeypatch.setattr(
        erlab.interactive.imagetool.manager,
        "_watch_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        watcher.watch("a")

    assert "a" not in watcher.watched_vars
    watcher.shutdown()


def test_start_polling_and_poll_loop_branches(fake_shell, monkeypatch):
    watcher = _Watcher(fake_shell)

    with pytest.raises(ValueError, match="interval_s must be > 0"):
        watcher.start_polling(0)

    started = {"count": 0}

    class FakeThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            started["count"] += 1

        def is_alive(self):
            return False

    monkeypatch.setattr(watcher_core.threading, "Thread", FakeThread)
    watcher.start_polling(0.1)
    assert started["count"] == 1

    # Already-running poll thread path.
    watcher._poll_thread = type(
        "AliveThread",
        (),
        {"is_alive": lambda self: True, "join": lambda self, timeout=None: None},
    )()
    watcher.start_polling(0.2)
    assert started["count"] == 1

    calls = {"count": 0}

    class FakeStop:
        def __init__(self):
            self._flags = [False, True]

        def wait(self, _):
            return self._flags.pop(0)

        def set(self):
            return None

    watcher._poll_stop = FakeStop()

    def _count_call() -> None:
        calls["count"] += 1

    monkeypatch.setattr(watcher, "_maybe_push", _count_call)
    watcher._poll_loop()
    assert calls["count"] == 1


def test_callback_registration_failure_and_enable_auto_push_error(
    fake_shell, monkeypatch
):
    watcher = _Watcher(fake_shell)
    key = 101

    class FailingEvents:
        def register(self, *_):
            raise RuntimeError("register failed")

    fake_shell.events = FailingEvents()
    assert (
        watcher_core._register_post_run_cell_callback(fake_shell, watcher, key) is False
    )
    assert key not in watcher_core._POST_RUN_CELL_CALLBACKS

    # events without unregister should still be handled gracefully.
    shell_without_unregister = type("ShellNoUnregister", (), {"events": object()})()
    watcher_core._POST_RUN_CELL_CALLBACKS[key] = (
        shell_without_unregister,
        lambda: None,
    )
    watcher_core._unregister_post_run_cell_callback(key)
    assert key not in watcher_core._POST_RUN_CELL_CALLBACKS

    monkeypatch.setattr(watcher_ipy, "_safe_get_ipython_shell", lambda: None)
    with pytest.raises(RuntimeError, match="No active IPython shell found"):
        watcher_mod.enable_ipython_auto_push()


def test_manager_watch_transport_wrappers_are_deprecated(monkeypatch):
    manager = erlab.interactive.imagetool.manager
    calls = {"watch": None, "unwatch": None}

    def _fake_watch(varname, uid, data, show=False):
        calls["watch"] = (varname, uid, data, show)

    def _fake_unwatch(uid, remove=False):
        calls["unwatch"] = (uid, remove)
        return "ok"

    monkeypatch.setattr(manager, "_watch_data", _fake_watch, raising=False)
    monkeypatch.setattr(manager, "_unwatch_data", _fake_unwatch, raising=False)

    darr = xr.DataArray(np.array([1, 2]), dims=("x",))
    with pytest.deprecated_call(match="watch_data"):
        manager.watch_data("darr", "uid-1", darr, show=True)
    assert calls["watch"] == ("darr", "uid-1", darr, True)

    with pytest.deprecated_call(match="unwatch_data"):
        response = manager.unwatch_data("uid-1", remove=True)
    assert response == "ok"
    assert calls["unwatch"] == ("uid-1", True)


def test_watcher_real(
    qtbot,
    ip_shell: IPython.InteractiveShell,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    monkeypatch,
):
    with manager_context() as manager:
        qtbot.addWidget(manager, before_close_func=lambda w: w.remove_all_tools())
        manager.show()

        darr = xr.DataArray(
            np.arange(125).reshape((5, 5, 5)),
            dims=["alpha", "beta", "eV"],
            coords={"alpha": np.arange(5), "beta": np.arange(5), "eV": np.arange(5)},
        )
        ip_shell.user_ns.update({"darr": darr})

        watcher = ip_shell.magics_manager.registry.get("WatcherMagics")._watcher

        # Try watching
        ip_shell.run_line_magic("watch", "darr")
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        # Check watched
        assert manager._imagetool_wrappers[0].watched

        # Get selection code
        assert (
            manager.get_imagetool(0).slicer_area.main_image.get_selection_code()
            == "darr.qsel(eV=2.0)"
        )

        # Watched tooltip
        text = None

        def fake_show_text(pos, s, *args, **kwargs):
            nonlocal text
            text = s

        monkeypatch.setattr(QtWidgets.QToolTip, "showText", fake_show_text)

        index = manager.tree_view._model.index(0, 0)  # first tool
        option = QtWidgets.QStyleOptionViewItem()
        manager.tree_view._delegate.initStyleOption(option, index)
        _, _, _, watched_rect = manager.tree_view._delegate._compute_icons_info(
            option, index.internalPointer()
        )
        pos = watched_rect.center()
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            pos,
            manager.tree_view.viewport().mapToGlobal(pos),
        )
        handled = manager.tree_view._delegate.helpEvent(
            event, manager.tree_view, option, index
        )

        assert handled
        assert text == "Variable synced with IPython"

        # Update data
        with qtbot.wait_signal(manager.server.sigWatchedVarChanged, timeout=10000):
            watcher._last_send = 0  # Bypass rate limit for deterministic update
            ip_shell.user_ns["darr"] = darr * 2
            watcher._maybe_push()

        xr.testing.assert_equal(
            manager.get_imagetool(0).slicer_area._data, ip_shell.user_ns["darr"]
        )

        # Modify data from manager side
        manager.get_imagetool(0).slicer_area.set_data(darr + 10)

        with qtbot.wait_signal(manager._sigWatchedDataEdited):
            manager._imagetool_wrappers[0]._trigger_watched_update()

        qtbot.wait(1000)  # wait for async update to complete
        xr.testing.assert_equal(ip_shell.user_ns["darr"], darr + 10)

        # Unwatch
        ip_shell.run_line_magic("watch", "-d darr")
        qtbot.wait_until(lambda: not manager._imagetool_wrappers[0].watched)

        # Watch again
        ip_shell.run_line_magic("watch", "darr")
        qtbot.wait_until(lambda: manager.ntools == 2)
        assert manager._imagetool_wrappers[1].watched

        # Remove watched
        ip_shell.run_line_magic("watch", "-x darr")
        qtbot.wait_until(lambda: manager.ntools == 1)

        # Watch again
        ip_shell.run_line_magic("watch", "darr")
        qtbot.wait_until(lambda: manager.ntools == 2)
        assert manager._imagetool_wrappers[1].watched

        # Stop watching all
        ip_shell.run_line_magic("watch", "-z")
        qtbot.wait_until(lambda: not manager._imagetool_wrappers[1].watched)

        # Watch again
        ip_shell.run_line_magic("watch", "darr")
        qtbot.wait_until(lambda: manager.ntools == 3)
        assert manager._imagetool_wrappers[2].watched

        # Stop watching and close all watched
        ip_shell.run_line_magic("watch", "-xz")
        qtbot.wait_until(lambda: manager.ntools == 2)

        watcher.shutdown()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0)
