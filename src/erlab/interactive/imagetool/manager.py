"""Manager for multiple ImageTool windows."""

from __future__ import annotations

import functools
import gc
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

    from erlab.interactive.imagetool.core import ImageSlicerArea

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


class ItoolManagerParseError(Exception):
    """Raised when the data received from the client cannot be parsed."""


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
    def __init__(self, manager: ImageToolManagerGUI, index: int, tool: ImageTool):
        super().__init__()
        self._tool: ImageTool | None = None

        self.manager: ImageToolManagerGUI = manager
        self.index: int = index
        self._archived_fname: str | None = None
        self._recent_geometry: QtCore.QRect | None = None

        self.tool = tool

        self.manager.sigLinkersChanged.connect(self.update_link_icon)
        self._setup_gui()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

    @property
    def tool(self) -> ImageTool | None:
        return self._tool

    @tool.setter
    def tool(self, value: ImageTool | None):
        if self._tool is None:
            if self._archived_fname is not None:
                # Remove the archived file
                os.remove(self._archived_fname)
        else:
            # Close and cleanup existing tool
            self.slicer_area.unlink()
            self._tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self._tool.removeEventFilter(self)
            self._tool.slicer_area.set_data(np.zeros((2, 2)))
            self._tool.destroyed.connect(self._destroyed_callback)
            self._tool.close()

        if value is not None:
            # Install event filter to detect visibility changes
            value.installEventFilter(self)
            value.sigTitleChanged.connect(self.update_title)

        self._tool = value

    @property
    def slicer_area(self) -> ImageSlicerArea:
        if self.tool is None:
            raise ValueError("ImageTool is not available")
        return self.tool.slicer_area

    @property
    def archived(self) -> bool:
        return self._tool is None

    def eventFilter(self, obj, event):
        if obj == self.tool and (
            event.type() == QtCore.QEvent.Type.Show
            or event.type() == QtCore.QEvent.Type.Hide
            or event.type() == QtCore.QEvent.Type.WindowStateChange
        ):
            self.visibility_changed()
        return super().eventFilter(obj, event)

    def _destroyed_callback(self):
        print("DESTROYED!")
        self.manager.sigReloadLinkers.emit()

    def _setup_gui(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.check = QtWidgets.QCheckBox(cast(ImageTool, self.tool).windowTitle())
        self.check.toggled.connect(self.manager.changed)

        self.link_icon = qta.IconWidget("mdi6.link-variant", opacity=0.0)

        self.visibility_btn = IconButton(
            on="mdi6.eye-outline", off="mdi6.eye-off-outline"
        )
        self.visibility_btn.toggled.connect(self.toggle_visibility)
        self.visibility_btn.setToolTip("Show/Hide")

        self.close_btn = IconButton("mdi6.window-close")
        self.close_btn.clicked.connect(lambda: self.manager.remove_tool(self.index))
        self.close_btn.setToolTip("Close")

        self.archive_btn = IconButton(
            on="mdi6.archive-outline", off="mdi6.archive-off-outline"
        )
        self.archive_btn.toggled.connect(self.toggle_archive)
        self.archive_btn.setToolTip("Archive/Unarchive")

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
    @QtCore.Slot(str)
    def update_title(self, title: str | None = None):
        if not self.archived:
            if title is None:
                title = cast(ImageTool, self.tool).windowTitle()

            new_title = f"{self.index}"
            if title != "":
                new_title += f": {title}"
            cast(ImageTool, self.tool).setWindowTitle(new_title)
            self.check.setText(new_title)

    @QtCore.Slot()
    def update_link_icon(self):
        if self.archived or not self.slicer_area.is_linked:
            self.link_icon.setIcon(qta.icon("mdi6.link-variant", opacity=0.0))
        else:
            self.link_icon.setIcon(
                qta.icon(
                    "mdi6.link-variant",
                    color=self.manager.color_for_linker(
                        self.slicer_area._linking_proxy
                    ),
                )
            )

    @QtCore.Slot()
    def visibility_changed(self):
        self._recent_geometry = self.tool.geometry()
        self.visibility_btn.blockSignals(True)
        self.visibility_btn.setChecked(self.tool.isVisible())
        self.visibility_btn.blockSignals(False)

    @QtCore.Slot()
    def toggle_visibility(self):
        if self.tool.isVisible():
            self.tool.close()
        else:
            if self._recent_geometry is not None:
                self.tool.setGeometry(self._recent_geometry)
            self.tool.show()
            self.tool.activateWindow()

    @QtCore.Slot()
    def show_clicked(self):
        self.tool.show()
        self.tool.activateWindow()

    @QtCore.Slot()
    def close_tool(self):
        self.tool = None

    @QtCore.Slot()
    def archive(self):
        if not self.archived:
            self._archived_fname = os.path.join(
                self.manager.cache_dir, str(uuid.uuid4())
            )
            self.tool.to_pickle(self._archived_fname)

            self.close_tool()
            self.check.blockSignals(True)
            self.check.setChecked(False)
            self.check.setDisabled(True)
            self.check.blockSignals(False)
            self.visibility_btn.blockSignals(True)
            self.visibility_btn.setChecked(False)
            self.visibility_btn.setDisabled(True)
            self.visibility_btn.blockSignals(False)
            gc.collect()

    @QtCore.Slot()
    def unarchive(self):
        if self.archived:
            self.check.setDisabled(False)
            self.visibility_btn.setDisabled(False)

            self.tool = ImageTool.from_pickle(self._archived_fname)
            self.tool.show()

            self.manager.sigReloadLinkers.emit()

    @QtCore.Slot()
    def toggle_archive(self):
        if self.archived:
            self.unarchive()
        else:
            self.archive()


class ImageToolManagerGUI(QtWidgets.QMainWindow):
    sigLinkersChanged = QtCore.Signal()
    sigReloadLinkers = QtCore.Signal()

    def __init__(self: ImageToolManagerGUI):
        super().__init__()
        self.setWindowTitle("ImageTool Manager")
        self.tool_options: dict[int, ImageToolOptionsWidget] = {}
        self.linkers: list[SlicerLinkProxy] = []

        self.titlebar = QtWidgets.QWidget()
        self.titlebar_layout = QtWidgets.QHBoxLayout()
        self.titlebar_layout.setContentsMargins(0, 0, 0, 0)
        self.titlebar.setLayout(self.titlebar_layout)

        self.add_button = IconButton("mdi6.folder-open-outline")
        self.add_button.clicked.connect(self.add_new)
        self.add_button.setToolTip("New ImageTool from file")

        self.close_button = IconButton("mdi6.close-box-multiple-outline")
        self.close_button.clicked.connect(self.close_selected)
        self.close_button.setToolTip("Close selected windows")

        self.link_button = IconButton("mdi6.link-variant")
        self.link_button.clicked.connect(self.link_selected)
        self.link_button.setToolTip("Link selected windows")

        self.unlink_button = IconButton("mdi6.link-variant-off")
        self.unlink_button.clicked.connect(self.unlink_selected)
        self.unlink_button.setToolTip("Unlink selected windows")

        self.titlebar_layout.addWidget(self.add_button)
        self.titlebar_layout.addWidget(self.close_button)
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

        # Store most recent name filter and directory for new windows
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None

        self.setCentralWidget(self.options)
        self.sigLinkersChanged.connect(self.changed)
        self.sigReloadLinkers.connect(self.cleanup_linkers)
        self.changed()

    @property
    def cache_dir(self) -> str:
        """Name of the cache directory where archived data are stored."""
        return self._tmp_dir.name

    @property
    def selected_tool_indices(self) -> list[int]:
        """Names of currently checked tools."""
        return [t for t, opt in self.tool_options.items() if opt.check.isChecked()]

    @property
    def ntools(self) -> int:
        return len(self.tool_options)

    @property
    def next_idx(self) -> int:
        return max(self.tool_options.keys(), default=-1) + 1

    def get_tool(self, index: int, unarchive: bool = True) -> ImageTool:
        if index not in self.tool_options:
            raise KeyError(f"Tool of index '{index}' not found")

        opt = self.tool_options[index]
        if opt.archived:
            if unarchive:
                opt.unarchive()
            else:
                raise KeyError(f"Tool of index '{index}' is archived")
        return cast(ImageTool, opt.tool)

    def deselect_all(self):
        """Clear selection."""
        for opt in self.tool_options.values():
            opt.check.setChecked(False)

    @QtCore.Slot()
    def add_new(self):
        """Add a new ImageTool window and open a file dialog."""
        tool = ImageTool(np.zeros((2, 2)))
        self.add_tool(tool, activate=True)

        tool.mnb._open_file(
            name_filter=self._recent_name_filter, directory=self._recent_directory
        )
        self._recent_name_filter = tool.mnb._recent_name_filter
        self._recent_directory = tool.mnb._recent_directory

    def color_for_linker(self, linker: SlicerLinkProxy) -> QtGui.QColor:
        idx = self.linkers.index(linker)
        return _LINKER_COLORS[idx % len(_LINKER_COLORS)]

    def add_tool(self, tool: ImageTool, activate: bool = False) -> int:
        index = int(self.next_idx)
        opt = ImageToolOptionsWidget(self, index, tool)
        self.tool_options[index] = opt
        opt.update_title()

        self.options_layout.insertWidget(self.options_layout.count() - 1, opt)

        self.sigReloadLinkers.emit()

        tool.show()

        if activate:
            tool.activateWindow()
            tool.raise_()

        return index

    @QtCore.Slot()
    def changed(self) -> None:
        selection = self.selected_tool_indices
        self.close_button.setEnabled(len(selection) != 0)

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
            self.get_tool(index).slicer_area.is_linked for index in selection
        ]
        self.unlink_button.setEnabled(any(is_linked))

        if all(is_linked):
            proxies = [
                self.get_tool(index).slicer_area._linking_proxy for index in selection
            ]
            if all(p == proxies[0] for p in proxies):
                self.link_button.setEnabled(False)

    def remove_tool(self, index: int):
        opt = self.tool_options.pop(index)
        if not opt.archived:
            cast(ImageTool, opt.tool).removeEventFilter(opt)

        self.options_layout.removeWidget(opt)
        opt.close_tool()
        opt.close()
        del opt
        gc.collect()

    @QtCore.Slot()
    def cleanup_linkers(self):
        for linker in list(self.linkers):
            if linker.num_children <= 1:
                linker.unlink_all()
                self.linkers.remove(linker)
        self.sigLinkersChanged.emit()

    @QtCore.Slot()
    def close_selected(self):
        checked_names = self.selected_tool_indices
        ret = QtWidgets.QMessageBox.question(
            self,
            "Close selected windows?",
            "1 selected window will be closed."
            if len(checked_names) == 1
            else f"{len(checked_names)} selected windows will be closed.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            for name in checked_names:
                self.remove_tool(name)

    @QtCore.Slot()
    @QtCore.Slot(bool)
    @QtCore.Slot(bool, bool)
    def link_selected(self, link_colors: bool = True, deselect: bool = True):
        self.unlink_selected(deselect=False)
        self.link_tools(*self.selected_tool_indices, link_colors=link_colors)
        if deselect:
            self.deselect_all()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    def unlink_selected(self, deselect: bool = True):
        for index in self.selected_tool_indices:
            self.get_tool(index).slicer_area.unlink()
        self.sigReloadLinkers.emit()
        if deselect:
            self.deselect_all()

    def link_tools(self, *indices, link_colors: bool = True):
        linker = SlicerLinkProxy(
            *[self.get_tool(t).slicer_area for t in indices], link_colors=link_colors
        )
        self.linkers.append(linker)
        self.sigReloadLinkers.emit()


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
        indices: list[int] = [
            self.add_tool(ImageTool(d, **kwargs), activate=True) for d in data
        ]

        if link:
            self.link_tools(*indices, link_colors=link_colors)

    def closeEvent(self, event: QtGui.QCloseEvent | None):
        if self.ntools != 0:
            if self.ntools == 1:
                msg = "1 remaining window will be closed."
            else:
                msg = f"All {self.ntools} remaining windows will be closed."

            ret = QtWidgets.QMessageBox.question(self, "Do you want to close?", msg)
            if not ret == QtWidgets.QMessageBox.StandardButton.Yes:
                if event:
                    event.ignore()
                return

            for tool in list(self.tool_options.keys()):
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
