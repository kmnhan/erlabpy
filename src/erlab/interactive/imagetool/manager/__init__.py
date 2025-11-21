"""Manager for multiple ImageTool windows.

.. image:: ../images/manager_light.png
    :align: center
    :alt: ImageToolManager window screenshot
    :class: only-light
    :width: 600px

.. only:: format_html

    .. image:: ../images/manager_dark.png
        :align: center
        :alt: ImageToolManager window screenshot
        :class: only-dark
        :width: 600px

This module provides a GUI application for managing multiple ImageTool windows. The
application can be started by running the script `itool-manager` from the command line
in the environment where the package is installed.

Python scripts communicate with the manager using a ZeroMQ connection with the default
port number 45555. The port number can be changed by setting the environment variable
``ITOOL_MANAGER_PORT``.

"""

from __future__ import annotations

__all__ = [
    "HOST_IP",
    "PORT",
    "PORT_WATCH",
    "ImageToolManager",
    "fetch",
    "get_log_file_path",
    "is_running",
    "load_in_manager",
    "main",
    "replace_data",
    "show_in_manager",
    "unwatch_data",
    "watch_data",
]


import logging
import os
import pathlib
import shutil
import sys
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.manager._logging import (
    configure_logging,
    get_log_file_path,
)
from erlab.interactive.imagetool.manager._mainwindow import _ICON_PATH, ImageToolManager
from erlab.interactive.imagetool.manager._server import (
    HOST_IP,
    PORT,
    PORT_WATCH,
    fetch,
    is_running,
    load_in_manager,
    replace_data,
    show_in_manager,
    unwatch_data,
    watch_data,
)
from erlab.interactive.utils import MessageDialog

logger = logging.getLogger(__name__)


_manager_instance: ImageToolManager | None = None
"""Reference to the running manager instance."""

_always_use_socket: bool = False
"""Internal flag to use sockets within same process for test coverage."""


class _ManagerApp(QtWidgets.QApplication):
    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        self._pending_files: list[pathlib.Path] = []

    def event(self, e: QtCore.QEvent | None) -> bool:  # pragma: no cover
        if e and e.type() == QtCore.QEvent.Type.FileOpen:
            # Happens both at first launch (before your window shows)
            # and when the app is already running

            file_event = typing.cast("QtGui.QFileOpenEvent", e)
            if file_event.url().isLocalFile():
                self._handle_open_file(pathlib.Path(file_event.url().toLocalFile()))
                return True
        return super().event(e)

    def _handle_open_file(self, path: pathlib.Path) -> None:  # pragma: no cover
        if _manager_instance:
            _manager_instance._handle_dropped_files([path])
        else:
            self._pending_files.append(path)


def _get_updater_settings() -> QtCore.QSettings:  # pragma: no cover
    """Get the QSettings object for the updater."""
    return QtCore.QSettings(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        "erlabpy",
        "imagetool-manager-updater",
    )


def _cleanup_update_tmp_dirs(settings) -> None:
    """Clean up any leftover temporary update directories from previous runs."""
    tmp_dirs = settings.value("update_tmp_dirs", "")
    tmp_dir_list = tmp_dirs.strip().split(",") if tmp_dirs else []
    for d in tmp_dir_list:
        p = pathlib.Path(d)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    settings.setValue("update_tmp_dirs", "")


def main(execute: bool = True) -> None:
    """Start the ImageToolManager application.

    Running ``itool-manager`` from a shell will invoke this function.
    """
    global _manager_instance

    file_args = [pathlib.Path(f) for f in sys.argv[1:] if pathlib.Path(f).exists()]
    # Files passed as command-line arguments
    # Also handles opening files on Windows

    if file_args and is_running():  # pragma: no cover
        load_in_manager(file_args)
        return

    qapp = typing.cast(
        "QtWidgets.QApplication | None", QtWidgets.QApplication.instance()
    )
    if not qapp:
        qapp = _ManagerApp(sys.argv)
        qapp.setStyle("Fusion")
        qapp.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)

        if file_args:
            qapp._pending_files.extend(file_args)

    if (
        sys.platform != "darwin" or not erlab.utils.misc._IS_PACKAGED
    ):  # pragma: no branch
        # Ignore if running in a PyInstaller bundle on macOS
        qapp.setWindowIcon(QtGui.QIcon(_ICON_PATH))
        qapp.setApplicationName("ImageTool Manager")
        qapp.setApplicationDisplayName("ImageTool Manager")
        qapp.setApplicationVersion(erlab.__version__)

    while is_running():  # pragma: no branch
        if (
            sys.platform == "darwin" and erlab.utils.misc._IS_PACKAGED
        ):  # pragma: no cover
            # If another instance is detected and a file event is queued, sends files to
            # that instance (macOS packaged app only)

            # Populate _pending_files if FileOpenEvent(s) occurred
            qapp.processEvents()

            if isinstance(qapp, _ManagerApp) and qapp._pending_files:
                load_in_manager(qapp._pending_files)
                return

        dialog = MessageDialog(
            parent=None,
            title="",
            text="An instance of ImageToolManager is already running.",
            informative_text="Retry after closing the existing instance.",
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        if os.environ.get("PYTEST_VERSION"):  # pragma: no cover
            # Automatically confirm on test fail to avoid blocking
            timer = QtCore.QTimer(dialog)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda dlg=dialog: dlg.reject())
            timer.start(5000)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            break
    else:
        configure_logging()

        _manager_instance = ImageToolManager()
        _manager_instance.show()
        _manager_instance.activateWindow()

        if isinstance(qapp, _ManagerApp):  # pragma: no cover
            pending = qapp._pending_files.copy()
            if pending:
                _manager_instance._handle_dropped_files(pending)
                qapp._pending_files.clear()

        if erlab.utils.misc._IS_PACKAGED:  # pragma: no cover
            # Handle cleanup after a successful application update
            updater_settings = _get_updater_settings()
            new_version = str(erlab.__version__)
            old_version = updater_settings.value("version_before_update", "")
            if old_version != new_version:
                _manager_instance.updated(old_version, new_version)
                _cleanup_update_tmp_dirs(updater_settings)
                updater_settings.setValue("version_before_update", new_version)
                updater_settings.sync()

            # Suppress warnings on console initialization
            os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

        if execute:
            qapp.exec()
            _manager_instance = None


def _get_recent_directory() -> str:
    """Return the most recent directory used by the ImageToolManager.

    Used internally to set the default directory for various file dialogs in tools
    launched inside the manager. Returns an empty string if no directory has been set
    yet, or if the manager is running in a different process.

    """
    if (
        _manager_instance is not None
        and _manager_instance._recent_directory is not None
    ):
        return str(_manager_instance._recent_directory)
    return ""
