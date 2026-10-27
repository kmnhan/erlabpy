from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform not in {"darwin", "win32"},
    reason="Native desktop APIs require macOS or Windows",
)

_QT_SETUP = """
import sys
from qtpy import QtCore, QtGui, QtWidgets

app = QtWidgets.QApplication([])
expected_platform = "cocoa" if sys.platform == "darwin" else "windows"
assert app.platformName() == expected_platform, app.platformName()
"""


def _run_native_script(
    tmp_path: pathlib.Path, script: str, *, native_qt: bool = False
) -> None:
    if native_qt:
        expected_platform = "cocoa" if sys.platform == "darwin" else "windows"
        if os.environ.get("QT_QPA_PLATFORM") != expected_platform:
            pytest.skip(
                f"Native Qt coverage requires QT_QPA_PLATFORM={expected_platform}"
            )
        script = _QT_SETUP + textwrap.dedent(script)
    environment = os.environ.copy()
    source_path = pathlib.Path(__file__).resolve().parents[2] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_path), environment.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_window_cursor_wait_and_font(tmp_path) -> None:
    _run_native_script(
        tmp_path,
        """
        from matplotlib.backend_bases import cursors
        from matplotlib.backends.backend_qt import FigureCanvasQT
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from erlab.interactive import _shortcut_sequences
        from erlab.interactive import utils

        window = QtWidgets.QWidget()
        window.resize(320, 180)
        window.show()
        app.processEvents()
        flags = window.windowFlags()
        window.hide()
        utils.activate_window(window)
        app.processEvents()
        assert window.isVisible()
        assert window.windowFlags() == flags
        window.showMinimized()
        app.processEvents()
        utils.activate_window(window)
        app.processEvents()
        assert window.isVisible() and not window.isMinimized()

        utils.set_widget_cursor(window, QtCore.Qt.CursorShape.SizeAllCursor)
        if sys.platform == "darwin":
            assert not window.testAttribute(QtCore.Qt.WidgetAttribute.WA_SetCursor)
        else:
            assert window.cursor().shape() == QtCore.Qt.CursorShape.SizeAllCursor
        utils.set_widget_cursor(window, None)
        # Top-level Qt windows retain WA_SetCursor after unsetCursor().
        assert window.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor

        utils.patch_macos_matplotlib_qt_cursor()
        canvas = FigureCanvasQTAgg(Figure(figsize=(2, 2)))
        canvas.draw()
        canvas.set_cursor(cursors.MOVE)
        if sys.platform == "darwin":
            assert hasattr(FigureCanvasQT, utils._MPL_QT_CURSOR_PATCH_ATTR)
            assert not canvas.testAttribute(QtCore.Qt.WidgetAttribute.WA_SetCursor)

        with utils.wait_dialog(window, "Native platform check") as indicator:
            expected = (
                utils._WaitDialog if sys.platform == "darwin" else utils._WaitPanel
            )
            assert isinstance(indicator, expected)
            indicator.set_message("Updated native platform check")
            app.processEvents()
            assert indicator.isVisible()
        assert utils._WAIT_DIALOG_DEPTH == 0
        assert window.isEnabled()

        font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.GeneralFont
        )
        assert QtGui.QFontInfo(font).family()
        assert QtGui.QRawFont.fromFont(font).isValid()
        assert QtGui.QFontMetrics(font).horizontalAdvance("ImageTool α") > 0
        sequence = QtGui.QKeySequence(_shortcut_sequences.WORKSPACE_SAVE)
        assert not sequence.isEmpty()
        assert sequence.toString(QtGui.QKeySequence.SequenceFormat.NativeText)
        shortcut = QtGui.QShortcut(sequence, window)
        assert shortcut.key() == sequence

        canvas.close()
        window.close()
        app.processEvents()
        """,
        native_qt=True,
    )


def test_native_registry_process_liveness(tmp_path) -> None:
    _run_native_script(
        tmp_path,
        """
        import subprocess
        import sys
        from erlab.interactive.imagetool.manager._registry import _pid_exists

        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE,
        )
        try:
            assert _pid_exists(child.pid)
            child.communicate(b"x", timeout=10)
            assert child.returncode == 0
            assert not _pid_exists(child.pid)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
        """,
    )


def test_native_workspace_lock_file_is_hidden(tmp_path) -> None:
    _run_native_script(
        tmp_path,
        """
        import pathlib
        import stat
        import sys
        from erlab.interactive.imagetool.manager._workspace._storage import (
            _hide_workspace_lock_file,
        )

        path = pathlib.Path("workspace.itws.lock")
        path.write_bytes(b"temporary native lock")
        _hide_workspace_lock_file(str(path.resolve()))
        if sys.platform == "darwin":
            assert path.stat().st_flags & stat.UF_HIDDEN
        else:
            assert path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN
        """,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process desktop APIs")
def test_native_windows_process_desktop_integration(tmp_path) -> None:
    _run_native_script(
        tmp_path,
        """
        import ctypes
        import logging
        import os
        from erlab.interactive.imagetool.manager import _desktop

        class CaptureErrors(logging.Handler):
            def emit(self, record):
                assert record.exc_info is None, record.getMessage()

        _desktop.logger.setLevel(logging.DEBUG)
        _desktop.logger.addHandler(CaptureErrors())
        app_id = "dev.kmnhan.erlabpy.native-test"
        _desktop.set_windows_app_user_model_id(app_id)
        shell32 = ctypes.WinDLL("shell32")
        getter = shell32.GetCurrentProcessExplicitAppUserModelID
        getter.argtypes = [ctypes.POINTER(ctypes.c_wchar_p)]
        getter.restype = ctypes.c_long
        actual = ctypes.c_wchar_p()
        assert getter(ctypes.byref(actual)) == 0
        try:
            assert actual.value == app_id
        finally:
            free = ctypes.WinDLL("ole32").CoTaskMemFree
            free.argtypes = [ctypes.c_void_p]
            free.restype = None
            free(ctypes.cast(actual, ctypes.c_void_p))
        _desktop.allow_foreground_activation(os.getpid())
        """,
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Dock menu API")
def test_native_macos_dock_menu_lifetime(tmp_path) -> None:
    _run_native_script(
        tmp_path,
        """
        from erlab.interactive.imagetool.manager import _desktop
        from erlab.interactive.utils import qt_is_valid

        class Window(QtWidgets.QWidget):
            def load(self):
                raise AssertionError("The test must not open a menu action")

            def open_new_manager_instance(self):
                raise AssertionError("The test must not launch a Manager")

        window = Window()
        menu = _desktop.install_macos_dock_menu(window)
        assert menu is not None and qt_is_valid(menu)
        assert window._macos_dock_menu is menu
        assert menu.parent() is window
        assert len(menu.actions()) == 2
        _desktop.uninstall_macos_dock_menu(window)
        assert window._macos_dock_menu is None
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
        assert not qt_is_valid(menu)
        window.close()
        """,
        native_qt=True,
    )
