"""Manager for multiple ImageTool windows.

This module provides a GUI application for managing multiple ImageTool windows. The
application can be started by running the script `itool-manager` from the command line
in the environment where the package is installed.

Python scripts communicate with the manager using a socket connection with the default
port number 45555. The port number can be changed by setting the environment variable
``ITOOL_MANAGER_PORT``.

"""

from __future__ import annotations

__all__ = ["PORT", "ImageToolManager", "is_running", "main", "show_in_manager"]


import logging
import sys
from typing import cast

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.manager._mainwindow import (
    _ICON_PATH,
    _SHM_NAME,
    ImageToolManager,
)
from erlab.interactive.imagetool.manager._server import PORT, show_in_manager

logger = logging.getLogger(__name__)


_manager_instance: ImageToolManager | None = None
"""Reference to the running manager instance."""

_always_use_socket: bool = False
"""Internal flag to use sockets within same process for test coverage."""


class _InitDialog(QtWidgets.QDialog):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.label = QtWidgets.QLabel(
            "An instance of ImageToolManager is already running.\n"
            "Retry after closing the existing instance."
        )
        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.label)
        layout.addWidget(self.buttonBox)


def is_running() -> bool:
    """Check whether an instance of ImageToolManager is active.

    Returns
    -------
    bool
        True if an instance of ImageToolManager is running, False otherwise.
    """
    if sys.platform != "win32":
        # Shared memory is removed on crash only on Windows
        unix_fix_shared_mem = QtCore.QSharedMemory(_SHM_NAME)
        if unix_fix_shared_mem.attach():
            # Detaching will release the shared memory if no other process is attached
            unix_fix_shared_mem.detach()

    # If attaching succeeds, another instance is running
    return QtCore.QSharedMemory(_SHM_NAME).attach()


def main(execute: bool = True) -> None:
    """Start the ImageToolManager application.

    Running ``itool-manager`` from a shell will invoke this function.
    """
    global _manager_instance

    qapp = cast(QtWidgets.QApplication | None, QtWidgets.QApplication.instance())
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    qapp.setStyle("Fusion")
    qapp.setWindowIcon(QtGui.QIcon(_ICON_PATH))
    qapp.setApplicationName("imagetool-manager")
    qapp.setApplicationDisplayName("ImageTool Manager")
    qapp.setApplicationVersion(erlab.__version__)

    while is_running():
        dialog = _InitDialog()
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            break
    else:
        _manager_instance = ImageToolManager()
        _manager_instance.show()
        _manager_instance.activateWindow()
        if execute:
            qapp.exec()
            _manager_instance = None
