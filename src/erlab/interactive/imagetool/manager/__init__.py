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
    "is_running",
    "load_in_manager",
    "main",
    "replace_data",
    "show_in_manager",
    "unwatch_data",
    "watch_data",
]


import logging
import pathlib
import sys
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
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
            _manager_instance.open_multiple_files(
                [path], try_workspace=(path.suffix == "h5")
            )
        else:
            self._pending_files.append(path)


def main(execute: bool = True) -> None:
    """Start the ImageToolManager application.

    Running ``itool-manager`` from a shell will invoke this function.
    """
    global _manager_instance

    file_args = [pathlib.Path(f) for f in sys.argv[1:] if pathlib.Path(f).exists()]
    # Files passed as command-line arguments
    # This also handles opening files from Windows

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
        qapp.setApplicationName("imagetool-manager")
        qapp.setApplicationDisplayName("ImageTool Manager")
        qapp.setApplicationVersion(erlab.__version__)

    while is_running():  # pragma: no branch
        dialog = MessageDialog(
            parent=None,
            title="",
            text="An instance of ImageToolManager is already running.",
            informative_text="Retry after closing the existing instance.",
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            break
    else:
        _manager_instance = ImageToolManager()
        _manager_instance.show()
        _manager_instance.activateWindow()

        if isinstance(qapp, _ManagerApp) and qapp._pending_files:  # pragma: no cover
            _manager_instance.open_multiple_files(
                qapp._pending_files,
                try_workspace=all(
                    file_path.suffix == ".h5" for file_path in qapp._pending_files
                ),
            )
            qapp._pending_files.clear()

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
