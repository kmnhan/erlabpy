"""Manager for multiple ImageTool windows."""

from __future__ import annotations

import functools
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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import qtawesome as qta
from qtpy import QtCore, QtGui, QtWidgets

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


_LINKER_COLORS: tuple[QtGui.QColor, ...] = (
    QtGui.QColor(76, 114, 176),
    QtGui.QColor(221, 132, 82),
    QtGui.QColor(85, 168, 104),
    QtGui.QColor(196, 78, 82),
    QtGui.QColor(129, 114, 179),
    QtGui.QColor(147, 120, 96),
    QtGui.QColor(218, 139, 195),
    QtGui.QColor(140, 140, 140),
    QtGui.QColor(204, 185, 116),
    QtGui.QColor(100, 181, 205),
)


class ItoolManagerParseError(Exception):
    """Raised when the data received from the client cannot be parsed."""


def _coverage_resolve_trace(fn):
    # https://github.com/nedbat/coveragepy/issues/686#issuecomment-634932753
    @functools.wraps(fn)
    def _wrapped_for_coverage(*args, **kwargs):
        if threading._trace_hook:
            sys.settrace(threading._trace_hook)
        fn(*args, **kwargs)

    return _wrapped_for_coverage


def _save_pickle(obj: Any, filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def _load_pickle(filename: str) -> Any:
    with open(filename, "rb") as file:
        return pickle.load(file)


class _ManagerServer(QtCore.QThread):
    sigReceived = QtCore.Signal(list, dict)

    def __init__(self):
        super().__init__()
        self.stopped = threading.Event()

    @_coverage_resolve_trace
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
                self.sigReceived.emit([_load_pickle(f) for f in files], kwargs)
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


class _QHLine(QtWidgets.QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)


class ImageToolOptionsWidget(QtWidgets.QWidget):
    def __init__(self, manager: ImageToolManagerGUI, name: str, tool: ImageTool):
        super().__init__()
        self.manager: ImageToolManagerGUI = manager
        self.name: str = name
        self.tool: ImageTool = tool

        self.manager.sigLinkersChanged.connect(self.update_link_icon)

        self.tool.installEventFilter(self)  # Detect visibility changes
        self._setup_gui()

    def eventFilter(self, obj, event):
        if obj == self.tool and (
            event.type() == QtCore.QEvent.Type.Show
            or event.type() == QtCore.QEvent.Type.Hide
            or event.type() == QtCore.QEvent.Type.WindowStateChange
        ):
            self.visibility_changed()
        return super().eventFilter(obj, event)

    def _setup_gui(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.check = QtWidgets.QCheckBox(self.tool.windowTitle())
        self.check.toggled.connect(self.manager.changed)

        self.link_icon = qta.IconWidget("mdi6.link-variant", opacity=0.0)

        self.visibility_btn = IconButton(on="mdi6.eye", off="mdi6.eye-off")
        self.visibility_btn.toggled.connect(self.toggle_visibility)

        self.close_btn = IconButton("mdi6.trash-can")
        self.close_btn.clicked.connect(self.close_clicked)

        self.archive_btn = IconButton(
            on="mdi6.archive-arrow-down", off="mdi6.archive-arrow-up"
        )
        self.archive_btn.toggled.connect(self.toggle_archive)

        for btn in (
            self.link_icon,
            self.visibility_btn,
            self.archive_btn,
            self.close_btn,
        ):
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed
            )

        layout.addWidget(self.check)
        layout.addStretch()
        layout.addWidget(self.link_icon)
        layout.addWidget(self.visibility_btn)
        layout.addWidget(self.close_btn)
        layout.addWidget(self.archive_btn)

    @QtCore.Slot()
    def update_link_icon(self):
        if self.tool.slicer_area.is_linked:
            self.link_icon.setIcon(
                qta.icon(
                    "mdi6.link-variant",
                    color=self.manager.color_for_linker(
                        self.tool.slicer_area._linking_proxy
                    ),
                )
            )
        else:
            self.link_icon.setIcon(qta.icon("mdi6.link-variant", opacity=0.0))

    @QtCore.Slot()
    def visibility_changed(self):
        self.visibility_btn.blockSignals(True)
        self.visibility_btn.setChecked(self.tool.isVisible())
        self.visibility_btn.blockSignals(False)

    @QtCore.Slot()
    def toggle_visibility(self):
        if self.tool.isVisible():
            self.tool.close()
        else:
            self.tool.show()
            self.tool.activateWindow()

    @QtCore.Slot()
    def show_clicked(self):
        self.tool.show()
        self.tool.activateWindow()

    @QtCore.Slot()
    def close_clicked(self):
        self.manager.remove_tool(self.name)

    def archive(self):
        self.manager.archive(self.name)

    def unarchive(self):
        self.manager.unarchive(self.name)

    def toggle_archive(self):
        if self.archive_btn.isChecked():
            self.archive()
        else:
            self.unarchive()


class ImageToolManagerGUI(QtWidgets.QMainWindow):
    sigLinkersChanged = QtCore.Signal()
    sigReloadLinkers = QtCore.Signal()

    def __init__(self: ImageToolManagerGUI):
        super().__init__()
        self._tools: dict[str, ImageTool | str] = {}
        self.tool_options: dict[str, ImageToolOptionsWidget] = {}
        self.linkers: list[SlicerLinkProxy] = []
        self.next_idx: int = 0

        self.titlebar = QtWidgets.QWidget()
        self.titlebar_layout = QtWidgets.QHBoxLayout()
        self.titlebar_layout.setContentsMargins(0, 0, 0, 0)
        self.titlebar.setLayout(self.titlebar_layout)

        self.add_button = IconButton("mdi6.plus")
        self.add_button.clicked.connect(self.add_new)

        self.link_button = IconButton("mdi6.link-variant")
        self.link_button.clicked.connect(self.link_selected)

        self.unlink_button = IconButton("mdi6.link-variant-off")
        self.unlink_button.clicked.connect(self.unlink_selected)

        self.titlebar_layout.addWidget(QtWidgets.QLabel("ImageTool windows"))
        self.titlebar_layout.addStretch()
        self.titlebar_layout.addWidget(self.add_button)
        self.titlebar_layout.addWidget(self.link_button)
        self.titlebar_layout.addWidget(self.unlink_button)

        self.options = QtWidgets.QWidget()
        self.options_layout = QtWidgets.QVBoxLayout()
        self.options.setLayout(self.options_layout)

        self.options_layout.addWidget(self.titlebar)
        self.options_layout.addWidget(_QHLine())
        self.options_layout.addStretch()

        # Temporary directory for storing archived data
        self._tmp_dir = tempfile.TemporaryDirectory()

        # Store most recent name filter for new windows
        self._recent_name_filter: str | None = None

        self.setCentralWidget(self.options)
        self.sigLinkersChanged.connect(self.changed)
        self.sigReloadLinkers.connect(self.cleanup_linkers)
        self.changed()

    @property
    def cache_dir(self) -> str:
        """Name of the cache directory where archived data are stored."""
        return self._tmp_dir.name

    @property
    def selected_tool_names(self) -> list[str]:
        """Names of currently checked tools."""
        return [t for t, opt in self.tool_options.items() if opt.check.isChecked()]

    def get_tool(self, title: str) -> ImageTool:
        if not isinstance(self._tools[title], ImageTool):
            self.unarchive(title)
        return cast(ImageTool, self._tools[title])

    def deselect_all(self):
        """Clear selection."""
        for opt in self.tool_options.values():
            opt.check.setChecked(False)

    @QtCore.Slot()
    def add_new(self):
        """Add a new ImageTool window and open a file dialog."""
        tool = ImageTool(np.zeros((2, 2)))
        self.add_tool(tool, activate=True)

        tool.mnb._open_file(name_filter=self._recent_name_filter)
        self._recent_name_filter = tool.mnb._recent_name_filter

    def color_for_linker(self, linker: SlicerLinkProxy) -> QtGui.QColor:
        idx = self.linkers.index(linker)
        return _LINKER_COLORS[idx % len(_LINKER_COLORS)]

    def add_tool(self, tool: ImageTool, activate: bool = False) -> str:
        old = tool.windowTitle()
        title = f"{self.next_idx}"
        if old != "":
            title += f": {old}"
        self.next_idx += 1
        tool.setWindowTitle(title)

        opt = ImageToolOptionsWidget(self, title, tool)

        self._tools[title] = tool
        self.tool_options[title] = opt
        self.options_layout.insertWidget(self.options_layout.count() - 1, opt)

        self.sigReloadLinkers.emit()

        tool.show()

        if activate:
            tool.activateWindow()
            tool.raise_()

        return title

    @QtCore.Slot()
    def changed(self) -> None:
        selection = self.selected_tool_names
        match len(selection):
            case 0:
                self.link_button.setDisabled(True)
                self.unlink_button.setDisabled(True)
                return
            case 1:
                self.link_button.setDisabled(True)
            case _:
                self.link_button.setDisabled(False)

        is_linked: list[bool] = [
            self.get_tool(title).slicer_area.is_linked for title in selection
        ]
        self.unlink_button.setEnabled(any(is_linked))

        if all(is_linked):
            proxies = [
                self.get_tool(title).slicer_area._linking_proxy for title in selection
            ]
            if all(p == proxies[0] for p in proxies):
                self.link_button.setEnabled(False)

    def remove_tool(self, title: str):
        tool = self._tools.pop(title)
        opt = self.tool_options.pop(title)

        if not isinstance(tool, ImageTool):
            os.remove(tool)  # Delete pickle file
        else:
            tool.slicer_area.unlink()
            tool.close()

        self.options_layout.removeWidget(opt)
        del tool, opt

        self.sigReloadLinkers.emit()

    def archive(self, title: str):
        tool = self._tools[title]
        if not isinstance(tool, ImageTool):
            return

        opt = self.tool_options[title]

        fname: str = os.path.join(self.cache_dir, str(uuid.uuid4()))
        tool.to_pickle(fname)

        self._tools[title] = fname
        tool.slicer_area.unlink()
        tool.close()

        opt.check.setChecked(False)
        opt.check.setDisabled(True)
        opt.visibility_btn.setDisabled(True)
        del tool
        self.sigReloadLinkers.emit()

    def unarchive(self, title: str):
        pickle_path = self._tools[title]
        if isinstance(pickle_path, ImageTool):
            return

        tool = ImageTool.from_pickle(pickle_path)
        tool.setWindowTitle(title)

        self._tools[title] = tool
        opt = self.tool_options[title]

        opt.check.setDisabled(False)
        opt.visibility_btn.setDisabled(False)
        self.sigReloadLinkers.emit()

    @QtCore.Slot()
    def cleanup_linkers(self):
        for linker in list(self.linkers):
            if linker.num_children <= 1:
                linker.unlink_all()
                self.linkers.remove(linker)
        self.sigLinkersChanged.emit()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    @QtCore.Slot(bool, bool)
    def link_selected(self, link_colors: bool = True, deselect: bool = True):
        self.unlink_selected(deselect=False)
        self.link_tools(*self.selected_tool_names, link_colors=link_colors)
        if deselect:
            self.deselect_all()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    def unlink_selected(self, deselect: bool = True):
        for title in self.selected_tool_names:
            self.get_tool(title).slicer_area.unlink()
        self.sigReloadLinkers.emit()
        if deselect:
            self.deselect_all()

    def link_tools(self, *titles, link_colors: bool = True):
        linker = SlicerLinkProxy(
            *[self.get_tool(t).slicer_area for t in titles], link_colors=link_colors
        )
        self.linkers.append(linker)
        self.sigReloadLinkers.emit()

    @property
    def ntools(self) -> int:
        return len(self._tools)


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

            for tool in list(self._tools.keys()):
                self.remove_tool(tool)

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
        _save_pickle(darr, fname)
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
