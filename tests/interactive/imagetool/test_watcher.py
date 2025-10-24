import threading
import typing

import numpy as np
import pytest
import xarray as xr

from erlab.interactive.imagetool.manager import _watcher as watcher_mod
from erlab.interactive.imagetool.manager._watcher import _Watcher

if typing.TYPE_CHECKING:
    import IPython


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
    manager = watcher_mod.erlab.interactive.imagetool.manager
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
        watcher_mod.zmq.Context, "instance", classmethod(lambda cls: FakeContext())
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
        watcher_mod.zmq.Context, "instance", classmethod(lambda cls: FakeContext())
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


def test_watcher_real(qtbot, manager_context):
    with manager_context() as manager:
        qtbot.addWidget(manager)
        manager.show()

        # Start IPython session
        from IPython.testing.globalipapp import start_ipython

        ip_session: IPython.InteractiveShell = start_ipython()

        # Load extension
        ip_session.run_line_magic("load_ext", "erlab.interactive")

        darr = xr.DataArray(
            np.arange(125).reshape((5, 5, 5)),
            dims=["alpha", "beta", "eV"],
            coords={"alpha": np.arange(5), "beta": np.arange(5), "eV": np.arange(5)},
        )
        ip_session.user_ns.update({"darr": darr})

        watcher = ip_session.magics_manager.registry.get("WatcherMagics")._watcher

        # Try watching
        ip_session.run_line_magic("watch", "darr")
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

        # Check watched
        assert manager._imagetool_wrappers[0].watched

        # Get selection code
        assert (
            manager.get_imagetool(0).slicer_area.main_image.selection_code
            == ".qsel(eV=2.0)"
        )

        # Update data
        ip_session.user_ns["darr"] = darr**2
        with qtbot.wait_signal(manager.server.sigWatchedVarChanged):
            watcher._maybe_push()

        xr.testing.assert_equal(
            manager.get_imagetool(0).slicer_area._data, ip_session.user_ns["darr"]
        )

        # Modify data from manager side
        manager.get_imagetool(0).slicer_area.set_data(darr + 10)

        with qtbot.wait_signal(manager._sigWatchedDataEdited):
            manager._imagetool_wrappers[0]._trigger_watched_update()

        qtbot.wait(1000)  # wait for async update to complete
        xr.testing.assert_equal(ip_session.user_ns["darr"], darr + 10)

        # Unwatch
        ip_session.run_line_magic("watch", "-d darr")
        qtbot.wait_until(lambda: not manager._imagetool_wrappers[0].watched)

        # Watch again
        ip_session.run_line_magic("watch", "darr")
        qtbot.wait_until(lambda: manager.ntools == 2)
        assert manager._imagetool_wrappers[1].watched

        # Remove watched
        ip_session.run_line_magic("watch", "-x darr")
        qtbot.wait_until(lambda: manager.ntools == 1)

        # Watch again
        ip_session.run_line_magic("watch", "darr")
        qtbot.wait_until(lambda: manager.ntools == 2)
        assert manager._imagetool_wrappers[1].watched

        # Stop watching all
        ip_session.run_line_magic("watch", "-z")
        qtbot.wait_until(lambda: not manager._imagetool_wrappers[1].watched)

        # Watch again
        ip_session.run_line_magic("watch", "darr")
        qtbot.wait_until(lambda: manager.ntools == 3)
        assert manager._imagetool_wrappers[2].watched

        # Stop watching and close all watched
        ip_session.run_line_magic("watch", "-xz")
        qtbot.wait_until(lambda: manager.ntools == 2)

        watcher.shutdown()
        manager.remove_all_tools()
        qtbot.wait_until(lambda: manager.ntools == 0)
        manager.close()

    # Unload the extension and clean up
    ip_session.run_line_magic("unload_ext", "erlab.interactive")
    ip_session.user_ns.clear()  # Clear the user namespace

    ip_session.clear_instance()
    del start_ipython.already_called
