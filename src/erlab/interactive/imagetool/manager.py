"""Manager for multiple ImageTool windows."""

from __future__ import annotations

import os
import pickle
import socket
import struct
import sys
import tempfile
import threading
import time
import uuid
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any

from qtpy import QtCore, QtGui, QtWidgets

import erlab.io
from erlab.interactive.imagetool import ImageTool, _parse_input
from erlab.interactive.imagetool.controls import IconButton
from erlab.interactive.imagetool.core import SlicerLinkProxy

if TYPE_CHECKING:
    from collections.abc import Collection

    import numpy.typing as npt
    import xarray

PORT: int = int(os.getenv("ITOOL_MANAGER_PORT", "45555"))
"""Port number for the ImageToolManager server

The default port number is 45555. This can be changed by setting the environment
variable `ITOOL_MANAGER_PORT`.
"""

_SHM_NAME: str = "__enforce_single_itoolmanager"
"""Name of the shared memory object that enforces single instance of ImageToolManager"""


class ItoolManagerParseError(Exception):
    """Raised when the data received from the client cannot be parsed."""


class _ManagerServer(QtCore.QThread):
    sigReceived = QtCore.Signal(list, dict)

    def __init__(self):
        super().__init__()
        self.stopped = threading.Event()

    def run(self):
        self.stopped.clear()
        soc = socket.socket()
        soc.bind(("127.0.0.1", PORT))
        soc.setblocking(0)
        soc.listen(1)
        print("Server is listening...")

        while not self.stopped.is_set():
            try:
                conn, _ = soc.accept()
            except BlockingIOError:
                time.sleep(0.01)
                continue

            conn.setblocking(1)
            # Receive the size of the data first
            data_size = struct.unpack(">L", _recv_all(conn, 4))[0]

            # Receive the data
            kwargs = _recv_all(conn, data_size)
            try:
                kwargs = pickle.loads(kwargs)
                files = kwargs.pop("__filename")
                self.sigReceived.emit([erlab.io.load_hdf5(f) for f in files], kwargs)
                for f in files:
                    os.remove(f)
                    dirname = os.path.dirname(f)
                    if os.path.isdir(dirname):
                        try:
                            os.rmdir(dirname)
                        except OSError:
                            pass
            except (
                pickle.UnpicklingError,
                AttributeError,
                EOFError,
                ImportError,
                IndexError,
            ) as e:
                print("Failed to unpickle data:", e)

            conn.close()


class ImageToolOptionsWidget(QtWidgets.QWidget):
    def __init__(
        self, manager: ImageToolManagerGUI, name: str, tool: QtWidgets.QMainWindow
    ):
        super().__init__()
        self.manager: ImageToolManagerGUI = manager
        self.name: str = name
        self.tool: QtWidgets.QMainWindow = tool
        self._setup_gui()

    def _setup_gui(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.label = QtWidgets.QLabel(self.tool.windowTitle())

        self.show_btn = IconButton("mdi6.eye")
        self.show_btn.clicked.connect(self.show_clicked)

        self.hide_btn = IconButton("mdi6.eye-off")
        self.hide_btn.clicked.connect(self.tool.close)

        self.close_btn = IconButton("mdi6.close")
        # self.close_btn = IconButton("mdi6.trash-can")
        self.close_btn.clicked.connect(self.close_clicked)

        self.archive_btn = IconButton(
            on="mdi6.archive-arrow-down", off="mdi6.archive-arrow-up"
        )
        self.archive_btn.toggled.connect(self.toggle_archive)

        for btn in (self.show_btn, self.hide_btn, self.archive_btn, self.close_btn):
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed
            )

        layout.addWidget(self.label)
        layout.addWidget(self.show_btn)
        layout.addWidget(self.hide_btn)
        layout.addWidget(self.close_btn)
        layout.addWidget(self.archive_btn)

    @QtCore.Slot()
    def show_clicked(self):
        self.tool.show()
        self.tool.activateWindow()

    @QtCore.Slot()
    def close_clicked(self):
        self.manager.remove_tool(self.name)

    def archive(self):
        self.show_btn.setDisabled(True)
        self.hide_btn.setDisabled(True)

    def unarchive(self):
        self.show_btn.setDisabled(False)
        self.hide_btn.setDisabled(False)

    def toggle_archive(self):
        if self.archive_btn.isChecked():
            self.archive()
        else:
            self.unarchive()


class ImageToolManagerGUI(QtWidgets.QMainWindow):
    sigNumChanged = QtCore.Signal()

    def __init__(self: ImageToolManagerGUI):
        super().__init__()
        self.tools: dict[str, ImageTool] = {}
        self.tool_options: dict[str, ImageToolOptionsWidget] = {}
        self.linkers: list[SlicerLinkProxy] = []
        self.next_idx: int = 0

        self.options = QtWidgets.QWidget()
        self.options_layout = QtWidgets.QVBoxLayout()
        # self.options_layout.setContentsMargins(0, 0, 0, 0)
        self.options.setLayout(self.options_layout)

        self.options_layout.addWidget(QtWidgets.QLabel("List of windows"))

        self.setCentralWidget(self.options)

    def add_tool(self, tool: ImageTool, activate: bool = False) -> str:
        old = tool.windowTitle()
        title = f"{self.next_idx}"
        if old != "":
            title += f": {old}"
        self.next_idx += 1
        tool.setWindowTitle(title)

        self.tools[title] = tool
        self.tool_options[title] = ImageToolOptionsWidget(self, title, tool)
        self.options_layout.addWidget(self.tool_options[title])

        self.sigNumChanged.emit()

        tool.show()

        if activate:
            tool.activateWindow()
            tool.raise_()
        return title

    def cleanup_linkers(self):
        for linker in list(self.linkers):
            if linker.num_children <= 1:
                linker.unlink_all()
                self.linkers.remove(linker)

    def remove_tool(self, title: str):
        tool = self.tools.pop(title)
        opt = self.tool_options.pop(title)

        tool.slicer_area.unlink()
        tool.close()
        self.options_layout.removeWidget(opt)
        del tool, opt

        self.cleanup_linkers()
        self.sigNumChanged.emit()

    def link_tools(self, *titles, link_colors: bool = True):
        linker = SlicerLinkProxy(
            *[self.tools[t].slicer_area for t in titles], link_colors=link_colors
        )
        self.linkers.append(linker)

    @property
    def ntools(self) -> int:
        return len(self.tools)


class ImageToolManager(ImageToolManagerGUI):
    def __init__(self):
        super().__init__()
        self.server = _ManagerServer()
        self.server.sigReceived.connect(self.data_recv)
        self.server.start()

        self._shm = shared_memory.SharedMemory(name=_SHM_NAME, create=True, size=1)

    @QtCore.Slot(list, dict)
    def data_recv(self, data: list[xarray.DataArray], kwargs: dict[str, Any]):
        link = kwargs.pop("link", False)
        link_colors = kwargs.pop("link_colors", True)
        titles: list[str] = [
            self.add_tool(ImageTool(d, **kwargs), activate=True) for d in data
        ]

        if link:
            self.link_tools(*titles, link_colors=link_colors)

    def closeEvent(self, event: QtGui.QCloseEvent | None):
        if self.ntools != 0:
            ret = QtWidgets.QMessageBox.question(
                self, "Do you want to close?", "All windows will be closed."
            )
            if not ret == QtWidgets.QMessageBox.StandardButton.Yes:
                if event:
                    event.ignore()
                return

            for tool in self.tools.values():
                tool.close()

        # Clean up shared memory
        self._shm.close()
        self._shm.unlink()

        # Stop the server
        self.server.stopped.set()
        self.server.wait()

        super().closeEvent(event)


class _InitDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setLayout(QtWidgets.QVBoxLayout())

        self.label = QtWidgets.QLabel(
            "An instance of ImageToolManager is already running.\n"
            "Retry after closing the existing instance."
        )
        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.buttonBox)


def _recv_all(conn, size):
    data = b""
    while len(data) < size:
        part = conn.recv(size - len(data))
        data += part
    return data


def is_running() -> bool:
    """Check whether an instance of ImageToolManager is active.

    Returns
    -------
    bool
        True if an instance of ImageToolManager is running, False otherwise.
    """
    try:
        shm = shared_memory.SharedMemory(name=_SHM_NAME, create=True, size=1)
    except FileExistsError:
        return True
    else:
        shm.unlink()
        return False


def main():
    """Start the ImageToolManager application.

    Running ``itool-manager`` from a shell will invoke this function.
    """
    qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")

    while is_running():
        dialog = _InitDialog()
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            break
    else:
        win = ImageToolManager()
        win.show()
        win.activateWindow()
        qapp.exec()


def show_in_manager(
    data: Collection[xarray.DataArray | npt.NDArray]
    | xarray.DataArray
    | npt.NDArray
    | xarray.Dataset,
    **kwargs,
):
    """Create and display ImageTool windows in the ImageToolManager.

    Parameters
    ----------
    data
        The data to be displayed in the ImageTool window. See :func:`itool
        <erlab.interactive.imagetool.itool>` for more information.
    data
        Array-like object or a sequence of such object with 2 to 4 dimensions. See
        notes.
    link
        Whether to enable linking between multiple ImageTool windows, by default
        `False`.
    link_colors
        Whether to link the color maps between multiple linked ImageTool windows, by
        default `True`.
    **kwargs
        Keyword arguments passed onto :class:`ImageTool
        <erlab.interactive.imagetool.ImageTool>`.

    """
    client_socket = socket.socket()
    client_socket.connect(("localhost", PORT))

    darr_list: list[xarray.DataArray] = _parse_input(data)

    # Save the data to a temporary file
    tmp_dir = tempfile.mkdtemp()

    files: list[str] = []

    for darr in darr_list:
        fname = str(uuid.uuid4())
        fname = os.path.join(tmp_dir, fname)
        erlab.io.save_as_hdf5(darr, fname, igor_compat=False)
        files.append(fname)

    kwargs["__filename"] = files

    # Serialize kwargs dict into a byte stream
    kwargs = pickle.dumps(kwargs)

    # Send the size of the data first
    client_socket.sendall(struct.pack(">L", len(kwargs)))

    client_socket.sendall(kwargs)

    client_socket.close()


if __name__ == "__main__":
    main()
