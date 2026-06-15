import dataclasses
import gc
import io
import logging
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import time
import types
import typing
import webbrowser
from collections.abc import Callable

import pytest
import xarray as xr
import zmq
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager.__main__ as manager_main
import erlab.interactive.imagetool.manager._desktop as manager_desktop
import erlab.interactive.imagetool.manager._mainwindow as manager_mainwindow
import erlab.interactive.imagetool.manager._widgets as manager_widgets
import erlab.interactive.imagetool.manager._workspace_io as manager_workspace_io
from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
from erlab.interactive.imagetool.manager import load_in_manager
from erlab.interactive.imagetool.manager._server import (
    AddDataPacket,
    Response,
    _recv_multipart,
)
from erlab.interactive.imagetool.manager._widgets import _WorkspacePropertiesState
from erlab.interactive.ptable import PeriodicTableWindow

from .helpers import action_map_by_object_name, menu_map_by_object_name

logger = logging.getLogger(__name__)


def test_manager_main_cache_directory_uses_qstandardpaths(
    tmp_path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache-root"
    requested_locations: list[QtCore.QStandardPaths.StandardLocation] = []

    def _writable_location(
        location: QtCore.QStandardPaths.StandardLocation,
    ) -> str:
        requested_locations.append(location)
        return str(cache_root)

    monkeypatch.setattr(
        manager_main.QtCore.QStandardPaths, "writableLocation", _writable_location
    )

    cache_dir = manager_main._cache_directory()

    assert requested_locations == [QtCore.QStandardPaths.StandardLocation.CacheLocation]
    assert cache_dir == cache_root / "dev.kmnhan.erlabpy.imagetoolmanager"
    assert cache_dir.is_dir()


def test_manager_main_cache_directory_falls_back_to_home_cache(
    tmp_path, monkeypatch
) -> None:
    home_dir = tmp_path / "home"

    monkeypatch.setattr(
        manager_main.QtCore.QStandardPaths, "writableLocation", lambda location: ""
    )
    monkeypatch.setattr(manager_main.pathlib.Path, "home", lambda: home_dir)

    cache_dir = manager_main._cache_directory()

    assert cache_dir == home_dir / ".cache" / "dev.kmnhan.erlabpy.imagetoolmanager"
    assert cache_dir.is_dir()


def test_manager_main_mpl_cache_directory(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(
        manager_main.QtCore.QStandardPaths,
        "writableLocation",
        lambda location: str(cache_root),
    )

    mpl_cache_dir = manager_main._mpl_cache_directory()

    assert (
        mpl_cache_dir
        == cache_root / "dev.kmnhan.erlabpy.imagetoolmanager" / "matplotlib"
    )
    assert mpl_cache_dir.is_dir()


def test_manager_main_configure_packaged_runtime_caches_skips_source_launch(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.setattr(manager_main.sys, "frozen", True, raising=False)
    monkeypatch.delattr(manager_main.sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(manager_main.sys, "pycache_prefix", None)
    monkeypatch.setattr(
        manager_main.QtCore.QStandardPaths,
        "writableLocation",
        lambda location: str(tmp_path / "cache-root"),
    )

    manager_main._configure_packaged_runtime_caches()

    assert "MPLCONFIGDIR" not in os.environ
    assert "NUMBA_CACHE_DIR" not in os.environ
    assert "PYTHONPYCACHEPREFIX" not in os.environ
    assert manager_main.sys.pycache_prefix is None


def test_manager_main_configure_packaged_runtime_caches_sets_packaged_env(
    tmp_path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache-root"
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.setattr(manager_main.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        manager_main.sys, "_MEIPASS", str(tmp_path / "_MEI"), raising=False
    )
    monkeypatch.setattr(manager_main.sys, "pycache_prefix", None)
    monkeypatch.setattr(manager_main.sys, "dont_write_bytecode", True)
    fake_numba_config = types.ModuleType("numba.core.config")
    fake_numba_config.CACHE_DIR = ""
    monkeypatch.setitem(
        manager_main.sys.modules, "numba.core.config", fake_numba_config
    )
    monkeypatch.setattr(
        manager_main.QtCore.QStandardPaths,
        "writableLocation",
        lambda location: str(cache_root),
    )

    manager_main._configure_packaged_runtime_caches()

    cache_dir = cache_root / "dev.kmnhan.erlabpy.imagetoolmanager"
    mpl_cache_dir = cache_dir / "matplotlib"
    numba_cache_dir = cache_dir / "numba"
    pycache_dir = cache_dir / "python-pycache"
    assert os.environ["MPLCONFIGDIR"] == str(mpl_cache_dir)
    assert os.environ["NUMBA_CACHE_DIR"] == str(numba_cache_dir)
    assert os.environ["PYTHONPYCACHEPREFIX"] == str(pycache_dir)
    assert manager_main.sys.pycache_prefix == str(pycache_dir)
    assert manager_main.sys.dont_write_bytecode is False
    assert str(numba_cache_dir) == fake_numba_config.CACHE_DIR
    assert mpl_cache_dir.is_dir()
    assert numba_cache_dir.is_dir()
    assert pycache_dir.is_dir()


def test_manager_reload(
    qtbot,
    example_loader,
    example_data_dir,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show()
        manager.activateWindow()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            load_in_manager(
                [example_data_dir / "data_006.h5"], loader_name=example_loader.name
            )
            qtbot.wait_until(
                lambda: manager.ntools == 1 and len(manager._file_handlers) == 0,
                timeout=5000,
            )
        finally:
            if gc_enabled:
                gc.enable()

        # Try reload
        with qtbot.wait_signal(manager.get_imagetool(0).slicer_area.sigDataChanged):
            manager.get_imagetool(0).slicer_area.reload()

        # Try reload again
        with qtbot.wait_signal(manager.get_imagetool(0).slicer_area.sigDataChanged):
            manager.get_imagetool(0).slicer_area.reload()


def make_dataarray_unpicklable(darr):
    mod = types.ModuleType("temp_mod_for_apply_ufunc")
    exec(  # noqa: S102
        "def myfunc(dat):\n    return dat * 2\n",
        mod.__dict__,
    )
    sys.modules[mod.__name__] = mod
    from temp_mod_for_apply_ufunc import myfunc

    darr = xr.apply_ufunc(myfunc, darr.chunk(), vectorize=True, dask="parallelized")
    return darr, mod.__name__


def _create_frames(
    obj: dict[str, typing.Any], pickler_cls: type[pickle.Pickler]
) -> list[memoryview]:
    buffers: list[pickle.PickleBuffer] = []  # out-of-band frames will be appended here
    bio = io.BytesIO()

    p = pickler_cls(bio, protocol=5, buffer_callback=buffers.append)
    p.dump(obj)
    header = memoryview(bio.getbuffer())
    return [header] + [memoryview(b) for b in buffers]


def test_manager_cloudpickle(
    qtbot,
    test_data,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context(use_socket=True) as manager:
        manager.show()
        qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

        # Test with unpicklable data
        darr, modname = make_dataarray_unpicklable(test_data)
        del sys.modules[modname]
        erlab.interactive.imagetool.manager.show_in_manager(darr)
        qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)
        logger.info("Confirmed tool is added, checking data")
        assert manager.get_imagetool(0).array_slicer.point_value(0) == 24.0

        # Pickle data first and remove module before trying unpickle
        darr, modname = make_dataarray_unpicklable(test_data)
        content = AddDataPacket(
            packet_type="add", data_list=darr, arguments={}
        ).model_dump(exclude_unset=True)
        ctx = zmq.Context.instance()
        sock: zmq.Socket = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.RCVHWM, 0)
        try:
            sock.connect(
                "tcp://"
                f"{erlab.interactive.imagetool.manager.HOST_IP}:"
                f"{manager._manager_record.port}"
            )
            frames = _create_frames(content, pickler_cls=pickle.Pickler)
            del sys.modules[modname]
            sock.send_multipart(frames, copy=False)

            timeout_seconds = 5.0
            start_time = time.time()
            while True:
                try:
                    response = Response(**_recv_multipart(sock, flags=zmq.NOBLOCK))
                except zmq.Again as e:
                    if time.time() - start_time > timeout_seconds:
                        raise TimeoutError(
                            "Timed out waiting for response from ZeroMQ socket."
                        ) from e
                else:
                    break
            assert response.status == "unpickle-failed"
        finally:
            sock.close()


@pytest.mark.parametrize(
    (
        "old_version",
        "new_version",
        "button_text",
        "expected_url",
        "expected_title",
        "expected_info",
    ),
    [
        (
            "",
            "1.2.3",
            "Open Release Notes",
            "https://github.com/kmnhan/erlabpy/releases",
            "ImageTool Manager Installed",
            "Welcome to ImageTool Manager! You are using version 1.2.3.",
        ),
        (
            "1.0.0",
            "1.1.0",
            "Open Documentation",
            "https://erlabpy.readthedocs.io/en/stable/user-guide/interactive/imagetool.html",
            "ImageTool Manager Updated",
            "ImageTool Manager has been successfully updated from version 1.0.0 to "
            "1.1.0.",
        ),
    ],
)
def test_manager_updated_opens_links(
    qtbot,
    accept_dialog,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
    monkeypatch,
    old_version,
    new_version,
    button_text,
    expected_url,
    expected_title,
    expected_info,
) -> None:
    opened: list[str] = []

    def _open(url: str) -> bool:
        opened.append(url)
        return True

    def _accept_call(dialog: QtWidgets.QMessageBox) -> None:
        assert dialog.text() == expected_title
        assert dialog.informativeText() == expected_info
        for button in dialog.buttons():
            if button.text() == button_text:
                button.click()
                return
        pytest.fail(f"Button {button_text!r} not found.")

    monkeypatch.setattr(webbrowser, "open", _open)

    with manager_context() as manager:
        manager.show()
        accept_dialog(
            lambda: manager.updated(old_version=old_version, new_version=new_version),
            accept_call=_accept_call,
        )

    assert opened == [expected_url]


def test_manager_workspace_properties_action_uses_current_state(
    monkeypatch,
    tmp_path,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    dialog_calls: list[
        tuple[str | None, _WorkspacePropertiesState, QtWidgets.QWidget | None]
    ] = []

    class _FakeWorkspacePropertiesDialog:
        def __init__(
            self,
            workspace_path: str | None,
            *,
            state: _WorkspacePropertiesState,
            parent: QtWidgets.QWidget | None = None,
        ) -> None:
            dialog_calls.append((workspace_path, state, parent))

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

    monkeypatch.setattr(
        manager_workspace_io,
        "_WorkspacePropertiesDialog",
        _FakeWorkspacePropertiesDialog,
    )

    with manager_context() as manager:
        manager.workspace_properties_action.trigger()
        assert dialog_calls[-1][0] is None
        assert not dialog_calls[-1][1].is_modified
        assert dialog_calls[-1][1].top_level_window_count == 0
        assert dialog_calls[-1][2] is manager

        workspace_path = tmp_path / "workspace.itws"
        workspace_path.touch()
        manager._adopt_workspace_path(workspace_path)

        manager.workspace_properties_action.trigger()
        assert dialog_calls[-1][0] == str(workspace_path.resolve())
        assert not dialog_calls[-1][1].is_modified
        assert dialog_calls[-1][1].top_level_window_count == 0
        assert dialog_calls[-1][2] is manager


def test_manager_standalone_app_menus(
    monkeypatch,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    launched: list[bool] = []
    theme_icon_names: list[str] = []

    def _record_theme_icon(name: str, *_args: typing.Any) -> QtGui.QIcon:
        theme_icon_names.append(name)
        return QtGui.QIcon()

    monkeypatch.setattr(QtGui.QIcon, "fromTheme", staticmethod(_record_theme_icon))
    monkeypatch.setattr(
        manager_widgets,
        "_launch_new_manager_instance",
        lambda: launched.append(True),
    )

    with manager_context() as manager:
        menus = menu_map_by_object_name(manager.menu_bar)

        assert "manager_file_menu" in menus
        assert "manager_apps_menu" in menus
        file_menu = menus["manager_file_menu"]
        file_action_names = [
            action.objectName() if not action.isSeparator() else ""
            for action in file_menu.actions()
        ]
        assert file_action_names[:10] == [
            "manager_open_workspace_action",
            "manager_open_recent_menu_action",
            "manager_save_workspace_action",
            "manager_save_workspace_as_action",
            "manager_compact_workspace_action",
            "manager_workspace_properties_action",
            "",
            "manager_add_data_files_action",
            "manager_add_windows_from_workspace_action",
            "manager_explorer_action",
        ]
        file_actions = action_map_by_object_name(file_menu)
        expected_open_shortcut = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open)
        assert file_actions["manager_open_workspace_action"] is manager.load_action
        assert manager.load_action.shortcut().toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        ) == expected_open_shortcut.toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        assert file_actions["manager_open_recent_menu_action"] is (
            manager.open_recent_menu.menuAction()
        )
        assert file_actions["manager_add_data_files_action"] is manager.open_action
        assert manager.open_action.shortcut().isEmpty()
        assert (
            file_actions["manager_add_windows_from_workspace_action"]
            is manager.import_workspace_action
        )
        assert (
            file_actions["manager_workspace_properties_action"]
            is manager.workspace_properties_action
        )
        assert manager.workspace_properties_action.menuRole() == (
            QtWidgets.QAction.MenuRole.NoRole
        )
        assert (
            manager.workspace_properties_action.shortcut().toString(
                QtGui.QKeySequence.SequenceFormat.PortableText
            )
            == "Alt+Return"
        )
        assert file_actions["manager_explorer_action"] is manager.explorer_action
        assert file_actions["manager_new_instance_action"].shortcut().isEmpty()
        file_actions["manager_new_instance_action"].trigger()
        assert launched == [True]

        apps_actions = action_map_by_object_name(menus["manager_apps_menu"])
        assert apps_actions["manager_ptable_action"] is manager.ptable_action
        assert (
            apps_actions["manager_ptable_action"]
            .shortcut()
            .toString(QtGui.QKeySequence.SequenceFormat.PortableText)
            == "Ctrl+Shift+P"
        )
        assert {
            "document-open",
            "document-open-recent",
            "document-save",
            "document-save-as",
            "document-properties",
            "list-add",
            "window-new",
            "applications-science",
        }.issubset(theme_icon_names)


def test_manager_macos_dock_menu_actions(
    monkeypatch,
    qtbot,
) -> None:
    class _DockMenuManager(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self._macos_dock_menu: QtWidgets.QMenu | None = None
            self.load_calls: list[bool] = []
            self.new_manager_calls: list[bool] = []

        def load(self) -> None:
            self.load_calls.append(True)

        def open_new_manager_instance(self) -> None:
            self.new_manager_calls.append(True)

    dock_menus: list[QtWidgets.QMenu] = []
    deleted_menus: list[QtWidgets.QMenu] = []
    original_delete_later = manager_desktop.QtWidgets.QMenu.deleteLater

    monkeypatch.setattr(manager_desktop.sys, "platform", "darwin")
    monkeypatch.setattr(
        manager_desktop.QtWidgets.QMenu,
        "setAsDockMenu",
        lambda menu: dock_menus.append(menu),
        raising=False,
    )
    monkeypatch.setattr(
        manager_desktop.QtWidgets.QMenu,
        "deleteLater",
        lambda menu: (deleted_menus.append(menu), original_delete_later(menu)),
    )

    manager = _DockMenuManager()
    qtbot.addWidget(manager)

    dock_menu = manager_desktop.install_macos_dock_menu(manager)

    assert dock_menu is not None
    assert dock_menus == [dock_menu]
    assert manager._macos_dock_menu is dock_menu

    actions = action_map_by_object_name(dock_menu)
    actions["manager_dock_open_workspace_action"].trigger()
    actions["manager_dock_new_manager_action"].trigger()

    assert manager.load_calls == [True]
    assert manager.new_manager_calls == [True]

    manager_desktop.uninstall_macos_dock_menu(manager)

    assert deleted_menus == [dock_menu]
    assert manager._macos_dock_menu is None


def test_manager_desktop_configure_process_platform_branch(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        manager_desktop, "set_windows_app_user_model_id", lambda: calls.append("appid")
    )
    monkeypatch.setattr(
        manager_desktop, "install_windows_jump_list", lambda: calls.append("jump-list")
    )

    monkeypatch.setattr(manager_desktop.sys, "platform", "linux")
    manager_desktop.configure_process()
    assert calls == []

    monkeypatch.setattr(manager_desktop.sys, "platform", "win32")
    manager_desktop.configure_process()
    assert calls == ["appid", "jump-list"]


def test_manager_macos_dock_menu_skip_and_failure_paths(
    qtbot, monkeypatch, caplog
) -> None:
    class _FakeManager(QtWidgets.QWidget):
        def load(self) -> None:
            pass

        def open_new_manager_instance(self) -> None:
            pass

    manager = _FakeManager()
    qtbot.addWidget(manager)

    monkeypatch.setattr(manager_desktop.sys, "platform", "linux")
    assert manager_desktop.install_macos_dock_menu(manager) is None

    deleted_menus: list[QtWidgets.QMenu] = []

    def _raise_dock_menu(_menu) -> None:
        raise RuntimeError("dock unavailable")

    monkeypatch.setattr(manager_desktop.sys, "platform", "darwin")
    monkeypatch.setattr(
        manager_desktop.QtWidgets.QMenu,
        "setAsDockMenu",
        _raise_dock_menu,
        raising=False,
    )
    monkeypatch.setattr(
        manager_desktop.QtWidgets.QMenu,
        "deleteLater",
        lambda menu: deleted_menus.append(menu),
    )
    caplog.set_level(logging.DEBUG, logger=manager_desktop.logger.name)

    assert manager_desktop.install_macos_dock_menu(manager) is None
    assert len(deleted_menus) == 1
    assert "Could not install macOS Dock menu" in caplog.text


def test_manager_desktop_record_macos_recent_workspace(
    monkeypatch, caplog, tmp_path
) -> None:
    workspace = tmp_path / "workspace.itws"
    data_file = tmp_path / "data.h5"
    recorded: list[pathlib.Path] = []

    monkeypatch.setattr(manager_desktop.sys, "platform", "darwin")
    monkeypatch.setattr(
        manager_desktop,
        "_record_macos_recent_document",
        lambda path: recorded.append(pathlib.Path(path)),
    )

    manager_desktop.record_recent_workspace(data_file)
    manager_desktop.record_recent_workspace(workspace)
    assert recorded == [workspace.resolve()]

    caplog.set_level(logging.DEBUG, logger=manager_desktop.logger.name)
    monkeypatch.setattr(
        manager_desktop,
        "_record_macos_recent_document",
        lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    manager_desktop.record_recent_workspace(workspace)
    assert "Could not record workspace with desktop shell" in caplog.text


def test_manager_desktop_macos_recent_document_import_paths(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "workspace.itws"
    noted_urls: list[tuple[str, str]] = []

    class _Controller:
        def noteNewRecentDocumentURL_(self, url) -> None:
            noted_urls.append(url)

    class _NSDocumentController:
        @staticmethod
        def sharedDocumentController():
            return _Controller()

    class _NSURL:
        @staticmethod
        def fileURLWithPath_(path):
            return ("url", path)

    appkit = types.ModuleType("AppKit")
    appkit.NSDocumentController = _NSDocumentController
    foundation = types.ModuleType("Foundation")
    foundation.NSURL = _NSURL
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setitem(sys.modules, "Foundation", foundation)

    manager_desktop._record_macos_recent_document(workspace)
    assert noted_urls == [("url", str(workspace))]

    fallback_paths: list[pathlib.Path] = []
    monkeypatch.setitem(sys.modules, "AppKit", None)
    monkeypatch.setitem(sys.modules, "Foundation", None)
    monkeypatch.setattr(
        manager_desktop,
        "_record_macos_recent_document_ctypes",
        lambda path: fallback_paths.append(path),
    )

    manager_desktop._record_macos_recent_document(workspace)
    assert fallback_paths == [workspace]


def test_manager_desktop_helpers_record_windows_recent_workspace(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "workspace.itws"
    data_file = tmp_path / "data.h5"
    calls: list[tuple[str, object, object | None]] = []

    class _Shell32:
        def SetCurrentProcessExplicitAppUserModelID(self, app_id):
            calls.append(("appid", app_id.value, None))
            return 0

        def SHAddToRecentDocs(self, flags, path):
            calls.append(("recent", flags, path.value))

    monkeypatch.setattr(manager_desktop.sys, "platform", "win32")
    monkeypatch.setattr(
        manager_desktop.ctypes,
        "windll",
        types.SimpleNamespace(shell32=_Shell32()),
        raising=False,
    )

    manager_desktop.set_windows_app_user_model_id()
    manager_desktop.record_recent_workspace(workspace)
    manager_desktop.record_recent_workspace(data_file)

    assert calls == [
        ("appid", manager_desktop.APP_USER_MODEL_ID, None),
        ("recent", manager_desktop._SHARD_PATHW, str(workspace.resolve())),
    ]


def test_manager_desktop_windows_app_id_error_paths(monkeypatch, caplog) -> None:
    class _ReturningShell32:
        def SetCurrentProcessExplicitAppUserModelID(self, _app_id):
            return 5

    class _RaisingShell32:
        def SetCurrentProcessExplicitAppUserModelID(self, _app_id):
            raise OSError("shell unavailable")

    monkeypatch.setattr(manager_desktop.sys, "platform", "linux")
    manager_desktop.set_windows_app_user_model_id()

    caplog.set_level(logging.DEBUG, logger=manager_desktop.logger.name)
    monkeypatch.setattr(manager_desktop.sys, "platform", "win32")
    monkeypatch.setattr(
        manager_desktop.ctypes,
        "windll",
        types.SimpleNamespace(shell32=_ReturningShell32()),
        raising=False,
    )
    manager_desktop.set_windows_app_user_model_id()
    assert "SetCurrentProcessExplicitAppUserModelID returned 5" in caplog.text

    monkeypatch.setattr(
        manager_desktop.ctypes,
        "windll",
        types.SimpleNamespace(shell32=_RaisingShell32()),
        raising=False,
    )
    manager_desktop.set_windows_app_user_model_id()
    assert "Could not set Windows AppUserModelID" in caplog.text


def test_manager_desktop_windows_recent_workspace_error_path(
    monkeypatch, caplog, tmp_path
) -> None:
    class _RaisingShell32:
        def SHAddToRecentDocs(self, _flags, _path) -> None:
            raise OSError("recent unavailable")

    monkeypatch.setattr(manager_desktop.sys, "platform", "win32")
    monkeypatch.setattr(
        manager_desktop.ctypes,
        "windll",
        types.SimpleNamespace(shell32=_RaisingShell32()),
        raising=False,
    )
    caplog.set_level(logging.DEBUG, logger=manager_desktop.logger.name)

    manager_desktop.record_recent_workspace(tmp_path / "workspace.itws")

    assert "Could not record workspace with desktop shell" in caplog.text


def test_manager_desktop_windows_jump_list_import_and_error_paths(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(manager_desktop.sys, "platform", "linux")
    manager_desktop.install_windows_jump_list()

    caplog.set_level(logging.DEBUG, logger=manager_desktop.logger.name)
    monkeypatch.setattr(manager_desktop.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pythoncom", None)
    manager_desktop.install_windows_jump_list()
    assert "pywin32 is unavailable" in caplog.text

    pythoncom = types.ModuleType("pythoncom")
    pythoncom.CLSCTX_INPROC_SERVER = "inproc"

    def _raise_create_instance(*_args):
        raise RuntimeError("COM unavailable")

    pythoncom.CoCreateInstance = _raise_create_instance
    win32com = types.ModuleType("win32com")
    propsys_package = types.ModuleType("win32com.propsys")
    propsys = types.SimpleNamespace(
        IID_IPropertyStore="property-store",
        PROPVARIANTType=lambda value: ("variant", value),
    )
    pscon = types.SimpleNamespace(PKEY_Title="title")
    propsys_package.propsys = propsys
    propsys_package.pscon = pscon
    shell_package = types.ModuleType("win32com.shell")
    shell = types.SimpleNamespace(
        CLSID_DestinationList="destination-list",
        IID_ICustomDestinationList="destination-list-iface",
        CLSID_EnumerableObjectCollection="collection",
        IID_IObjectCollection="collection-iface",
        CLSID_ShellLink="shell-link",
        IID_IShellLink="shell-link-iface",
    )
    shell_package.shell = shell
    monkeypatch.setitem(sys.modules, "pythoncom", pythoncom)
    monkeypatch.setitem(sys.modules, "win32com", win32com)
    monkeypatch.setitem(sys.modules, "win32com.propsys", propsys_package)
    monkeypatch.setitem(sys.modules, "win32com.propsys.propsys", propsys)
    monkeypatch.setitem(sys.modules, "win32com.propsys.pscon", pscon)
    monkeypatch.setitem(sys.modules, "win32com.shell", shell_package)
    monkeypatch.setitem(sys.modules, "win32com.shell.shell", shell)

    manager_desktop.install_windows_jump_list()
    assert "Could not install Windows Jump List tasks" in caplog.text


def test_manager_desktop_windows_jump_list_installs_tasks(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(manager_desktop.sys, "platform", "win32")
    monkeypatch.setattr(manager_desktop.sys, "executable", r"C:\App\manager.exe")

    shell = types.SimpleNamespace(
        CLSID_DestinationList="destination-list",
        IID_ICustomDestinationList="destination-list-iface",
        CLSID_EnumerableObjectCollection="collection",
        IID_IObjectCollection="collection-iface",
        CLSID_ShellLink="shell-link",
        IID_IShellLink="shell-link-iface",
    )
    propsys = types.SimpleNamespace(
        IID_IPropertyStore="property-store",
        PROPVARIANTType=lambda value: ("variant", value),
    )
    pscon = types.SimpleNamespace(PKEY_Title="title")

    class _DestinationList:
        def SetAppID(self, app_id) -> None:
            calls.append(("appid", app_id))

        def BeginList(self) -> None:
            calls.append(("begin", None))

        def AppendKnownCategory(self, category) -> None:
            calls.append(("category", category))

        def AddUserTasks(self, collection) -> None:
            calls.append(("tasks", len(collection.links)))

        def CommitList(self) -> None:
            calls.append(("commit", None))

    class _Collection:
        def __init__(self) -> None:
            self.links: list[_ShellLink] = []

        def AddObject(self, link) -> None:
            self.links.append(link)

    class _PropertyStore:
        def SetValue(self, key, value) -> None:
            calls.append(("property", (key, value)))

        def Commit(self) -> None:
            calls.append(("property-commit", None))

    class _ShellLink:
        def SetPath(self, path) -> None:
            calls.append(("path", path))

        def SetArguments(self, arguments) -> None:
            calls.append(("arguments", arguments))

        def SetDescription(self, description) -> None:
            calls.append(("description", description))

        def SetIconLocation(self, path, index) -> None:
            calls.append(("icon", (path, index)))

        def QueryInterface(self, interface):
            calls.append(("query", interface))
            return _PropertyStore()

    pythoncom = types.ModuleType("pythoncom")
    pythoncom.CLSCTX_INPROC_SERVER = "inproc"

    def _create_instance(clsid, *_args):
        if clsid == shell.CLSID_DestinationList:
            return _DestinationList()
        if clsid == shell.CLSID_EnumerableObjectCollection:
            return _Collection()
        if clsid == shell.CLSID_ShellLink:
            return _ShellLink()
        raise AssertionError(f"unexpected CLSID {clsid!r}")

    pythoncom.CoCreateInstance = _create_instance
    win32com = types.ModuleType("win32com")
    propsys_package = types.ModuleType("win32com.propsys")
    propsys_package.propsys = propsys
    propsys_package.pscon = pscon
    shell_package = types.ModuleType("win32com.shell")
    shell_package.shell = shell
    monkeypatch.setitem(sys.modules, "pythoncom", pythoncom)
    monkeypatch.setitem(sys.modules, "win32com", win32com)
    monkeypatch.setitem(sys.modules, "win32com.propsys", propsys_package)
    monkeypatch.setitem(sys.modules, "win32com.propsys.propsys", propsys)
    monkeypatch.setitem(sys.modules, "win32com.propsys.pscon", pscon)
    monkeypatch.setitem(sys.modules, "win32com.shell", shell_package)
    monkeypatch.setitem(sys.modules, "win32com.shell.shell", shell)

    manager_desktop.install_windows_jump_list()

    assert ("appid", manager_desktop.APP_USER_MODEL_ID) in calls
    assert ("category", manager_desktop._KDC_RECENT) in calls
    assert ("tasks", len(manager_desktop._WINDOWS_JUMP_LIST_TASKS)) in calls
    assert (
        "arguments",
        manager_desktop.OPEN_WORKSPACE_DIALOG_ARG,
    ) in calls
    assert ("arguments", manager_desktop.NEW_MANAGER_WINDOW_ARG) in calls
    assert ("property", ("title", ("variant", "Open Workspace…"))) in calls
    assert ("commit", None) in calls


def test_manager_windows_jump_list_task_specs() -> None:
    assert manager_desktop._WINDOWS_JUMP_LIST_TASKS == (
        (
            "Open Workspace…",
            (manager_desktop.OPEN_WORKSPACE_DIALOG_ARG,),
            "Open an ImageTool workspace file",
        ),
        (
            "New Manager Window",
            (manager_desktop.NEW_MANAGER_WINDOW_ARG,),
            "Open another ImageTool Manager window",
        ),
    )


def test_manager_packaging_declares_desktop_integration_metadata() -> None:
    installer_text = pathlib.Path("manager.iss").read_text()
    spec_text = pathlib.Path("manager.spec").read_text()

    assert '#define MyAppUserModelID "dev.kmnhan.erlabpy.imagetoolmanager"' in (
        installer_text
    )
    assert 'ValueName: "AppUserModelID"' in installer_text
    assert 'AppUserModelID: "{#MyAppUserModelID}"' in installer_text
    assert '"win32com.propsys.propsys"' in spec_text
    assert '"win32com.shell.shell"' in spec_text


def test_launch_new_manager_instance_uses_detached_source_process(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, typing.Any]]] = []

    monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", False)
    monkeypatch.setattr(manager_widgets.sys, "platform", "linux")
    monkeypatch.setattr(manager_widgets.sys, "executable", "/env/bin/python")
    monkeypatch.setattr(
        manager_widgets.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    manager_widgets._launch_new_manager_instance()

    assert calls == [
        (
            ["/env/bin/python", "-m", "erlab.interactive.imagetool.manager"],
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
                "start_new_session": True,
            },
        )
    ]


def test_launch_new_manager_instance_uses_macos_app_bundle(
    monkeypatch, tmp_path
) -> None:
    calls: list[tuple[list[str], dict[str, typing.Any]]] = []
    app_bundle = tmp_path / "ImageTool Manager.app"
    executable = app_bundle / "Contents" / "MacOS" / "ImageTool Manager"
    executable.parent.mkdir(parents=True)
    executable.touch()

    monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", True)
    monkeypatch.setattr(manager_widgets.sys, "platform", "darwin")
    monkeypatch.setattr(manager_widgets.sys, "executable", str(executable))
    monkeypatch.setattr(
        manager_widgets.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    manager_widgets._launch_new_manager_instance()

    assert calls == [
        (
            ["/usr/bin/open", "-n", str(app_bundle.resolve())],
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
                "start_new_session": True,
            },
        )
    ]


def test_launch_new_manager_instance_uses_windows_detached_flags(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, typing.Any]]] = []

    monkeypatch.setattr(erlab.utils.misc, "_IS_PACKAGED", False)
    monkeypatch.setattr(manager_widgets.sys, "platform", "win32")
    monkeypatch.setattr(manager_widgets.sys, "executable", r"C:\env\python.exe")
    monkeypatch.setattr(
        manager_widgets.subprocess, "DETACHED_PROCESS", 8, raising=False
    )
    monkeypatch.setattr(
        manager_widgets.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        512,
        raising=False,
    )
    monkeypatch.setattr(
        manager_widgets.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    manager_widgets._launch_new_manager_instance()

    assert calls == [
        (
            [r"C:\env\python.exe", "-m", "erlab.interactive.imagetool.manager"],
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
                "creationflags": 520,
            },
        )
    ]


def test_open_new_manager_instance_shows_error_dialog(monkeypatch) -> None:
    dialogs: list[tuple[object, str, str]] = []

    monkeypatch.setattr(
        manager_widgets,
        "_launch_new_manager_instance",
        lambda: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    monkeypatch.setattr(
        erlab.interactive.utils.MessageDialog,
        "critical",
        lambda parent, *, title, text: dialogs.append((parent, title, text)),
    )

    parent = typing.cast("manager_mainwindow.ImageToolManager", object())
    manager_widgets._WidgetsController(parent).open_new_manager_instance()

    assert dialogs == [
        (
            parent,
            "New Manager Window",
            "Could not open another ImageTool Manager window.",
        )
    ]


def test_manager_explorer_launcher_reuses_instance_and_opens_directory_tabs(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _TrackingTabbedExplorer(_TabbedExplorer):
        def __init__(self, *args, **kwargs) -> None:
            self.close_event_count = 0
            super().__init__(*args, **kwargs)

        def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
            self.close_event_count += 1
            super().closeEvent(event)

    with (
        manager_context() as manager,
        tempfile.TemporaryDirectory() as recent_dir,
        tempfile.TemporaryDirectory() as dropped_dir,
    ):
        manager._recent_directory = recent_dir
        spec = manager._standalone_app_specs["explorer"]
        manager._standalone_app_specs["explorer"] = dataclasses.replace(
            spec,
            factory=lambda: _TrackingTabbedExplorer(
                root_path=manager._recent_directory,
                loader_name=manager._recent_loader_name,
            ),
        )

        manager.ensure_explorer_initialized()
        explorer = manager.explorer

        assert isinstance(explorer, _TrackingTabbedExplorer)
        assert hasattr(manager, "explorer")
        assert explorer.tab_widget.count() == 1

        explorer.close_tab(0)
        qtbot.wait_until(lambda: not explorer.isVisible())
        assert explorer.close_event_count == 0
        manager.show_explorer()
        qtbot.wait_until(explorer.isVisible)
        assert manager.explorer is explorer

        explorer.hide()
        manager.show_explorer()
        qtbot.wait_until(explorer.isVisible)
        assert manager.explorer is explorer

        explorer.close()
        qtbot.wait_until(lambda: not explorer.isVisible())
        manager.show_explorer()
        qtbot.wait_until(explorer.isVisible)
        assert manager.explorer is explorer

        manager.open_multiple_files([pathlib.Path(dropped_dir)], try_workspace=True)

        qtbot.wait_until(lambda: explorer.tab_widget.count() == 2)
        assert explorer.current_explorer is not None
        assert explorer.current_explorer.current_directory == pathlib.Path(dropped_dir)


def test_manager_ptable_launcher_reuses_instance_without_affecting_tree(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    class _TrackingPeriodicTableWindow(PeriodicTableWindow):
        def __init__(self) -> None:
            self.close_event_count = 0
            super().__init__()

        def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
            self.close_event_count += 1
            super().closeEvent(event)

    with manager_context() as manager:
        initial_ntools = manager.ntools
        initial_rows = manager.tree_view.model().rowCount(QtCore.QModelIndex())
        spec = manager._standalone_app_specs["ptable"]
        manager._standalone_app_specs["ptable"] = dataclasses.replace(
            spec,
            factory=_TrackingPeriodicTableWindow,
        )

        manager.show_ptable()
        ptable = manager.ptable_window

        qtbot.wait_until(ptable.isVisible)
        assert isinstance(ptable, _TrackingPeriodicTableWindow)
        assert manager.ntools == initial_ntools
        assert manager.tree_view.model().rowCount(QtCore.QModelIndex()) == initial_rows

        ptable.search_edit.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        qtbot.keyClick(
            ptable.search_edit,
            QtCore.Qt.Key.Key_W,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        qtbot.wait_until(lambda: not ptable.isVisible())
        manager.show_ptable()
        qtbot.wait_until(ptable.isVisible)
        assert manager.ptable_window is ptable
        assert ptable.close_event_count == 0

        ptable.hide()
        manager.show_ptable()
        qtbot.wait_until(ptable.isVisible)
        assert manager.ptable_window is ptable

        ptable.close()
        qtbot.wait_until(lambda: not ptable.isVisible())
        manager.show_ptable()
        qtbot.wait_until(ptable.isVisible)
        assert manager.ptable_window is ptable
        assert manager.ntools == initial_ntools


def test_manager_close_event_closes_standalone_apps(
    qtbot,
    manager_context: Callable[
        ..., typing.ContextManager[erlab.interactive.imagetool.manager.ImageToolManager]
    ],
) -> None:
    with manager_context() as manager:
        manager.show_explorer()
        manager.show_ptable()

        explorer = manager.explorer
        ptable = manager.ptable_window

        qtbot.wait_until(explorer.isVisible)
        qtbot.wait_until(ptable.isVisible)

        manager.close()
        QtWidgets.QApplication.sendPostedEvents(None, 0)
        QtWidgets.QApplication.processEvents()

        assert manager._standalone_app_windows == {}
        assert not erlab.interactive.utils.qt_is_valid(explorer, ptable)
