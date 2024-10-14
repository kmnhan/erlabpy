"""Manager for multiple ImageTool windows.

This module provides a GUI application for managing multiple ImageTool windows. The
application can be started by running the script `itool-manager` from the command line
in the environment where the package is installed.

Python scripts communicate with the manager using a socket connection with the default
port number 45555. The port number can be changed by setting the environment variable
``ITOOL_MANAGER_PORT``.

"""

from __future__ import annotations

__all__ = [
    "PORT",
    "ImageToolManager",
    "is_running",
    "main",
    "show_in_manager",
]

import contextlib
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
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.imagetool import ImageTool, _parse_input
from erlab.interactive.imagetool.controls import IconButton
from erlab.interactive.imagetool.core import SlicerLinkProxy
from erlab.interactive.utils import _coverage_resolve_trace

if TYPE_CHECKING:
    from collections.abc import Collection

    import numpy.typing as npt
    import xarray

    from erlab.interactive.imagetool.core import ImageSlicerArea

PORT: int = int(os.getenv("ITOOL_MANAGER_PORT", "45555"))
"""Port number for the manager server.

The default port number 45555 can be overridden by setting the environment variable
``ITOOL_MANAGER_PORT``.
"""

_SHM_NAME: str = "__enforce_single_itoolmanager"
"""Name of the sharedmemory object that enforces single instance of ImageToolManager.

If a shared memory object with this name exists, it means that an instance is running.
"""


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
"""Colors for different linkers."""


def _save_pickle(obj: Any, filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(obj, file, protocol=-1)


def _load_pickle(filename: str) -> Any:
    with open(filename, "rb") as file:
        return pickle.load(file)


def _recv_all(conn, size):
    data = b""
    while len(data) < size:
        part = conn.recv(size - len(data))
        data += part
    return data


class ItoolManagerParseError(Exception):
    """Raised when the data received from the client cannot be parsed."""


class _ManagerServer(QtCore.QThread):
    sigReceived = QtCore.Signal(list, dict)

    def __init__(self) -> None:
        super().__init__()
        self.stopped = threading.Event()

    @_coverage_resolve_trace
    def run(self) -> None:
        self.stopped.clear()

        soc = socket.socket()
        soc.bind(("127.0.0.1", PORT))
        soc.setblocking(False)
        soc.listen()
        print("Server is listening...")

        while not self.stopped.is_set():
            try:
                conn, _ = soc.accept()
            except BlockingIOError:
                time.sleep(0.01)
                continue

            conn.setblocking(True)
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
                        with contextlib.suppress(OSError):
                            os.rmdir(dirname)
            except (
                pickle.UnpicklingError,
                AttributeError,
                EOFError,
                ImportError,
                IndexError,
            ) as e:
                print("Failed to unpickle data:", e)

            conn.close()

        soc.close()


class _QHLine(QtWidgets.QFrame):
    """Horizontal line widget."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)


class _ImageToolOptionsWidget(QtWidgets.QWidget):
    def __init__(
        self, manager: _ImageToolManagerGUI, index: int, tool: ImageTool
    ) -> None:
        super().__init__()
        self._tool: ImageTool | None = None

        self.manager: _ImageToolManagerGUI = manager
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
    def tool(self, value: ImageTool | None) -> None:
        if self._tool is None:
            if self._archived_fname is not None:
                # Remove the archived file
                os.remove(self._archived_fname)
        else:
            # Close and cleanup existing tool
            self._tool.slicer_area.unlink()
            self._tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self._tool.removeEventFilter(self)
            self._tool.sigTitleChanged.disconnect(self.update_title)
            self._tool.slicer_area.set_data(
                xr.DataArray(np.zeros((2, 2)), name=self._tool.slicer_area.data.name)
            )
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

    def _destroyed_callback(self) -> None:
        self.manager.sigReloadLinkers.emit()

    def _setup_gui(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.check = QtWidgets.QCheckBox(cast(ImageTool, self.tool).windowTitle())
        self.check.toggled.connect(self.manager._update_button_state)

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
    def update_title(self, title: str | None = None) -> None:
        if not self.archived:
            if title is None:
                title = cast(ImageTool, self.tool).windowTitle()

            new_title = f"{self.index}"
            if title != "":
                new_title += f": {title}"
            cast(ImageTool, self.tool).setWindowTitle(new_title)
            self.check.setText(new_title)

    @QtCore.Slot()
    def update_link_icon(self) -> None:
        if self.archived or not self.slicer_area.is_linked:
            self.link_icon.setIcon(qta.icon("mdi6.link-variant", opacity=0.0))
        else:
            self.link_icon.setIcon(
                qta.icon(
                    "mdi6.link-variant",
                    color=self.manager.color_for_linker(
                        cast(SlicerLinkProxy, self.slicer_area._linking_proxy)
                    ),
                )
            )

    @QtCore.Slot()
    def visibility_changed(self) -> None:
        if self.archived:
            return
        tool = cast(ImageTool, self.tool)
        self._recent_geometry = tool.geometry()
        self.visibility_btn.blockSignals(True)
        self.visibility_btn.setChecked(tool.isVisible())
        self.visibility_btn.blockSignals(False)

    @QtCore.Slot()
    def toggle_visibility(self) -> None:
        if self.archived:
            return
        tool = cast(ImageTool, self.tool)
        if tool.isVisible():
            tool.close()
        else:
            if self._recent_geometry is not None:
                tool.setGeometry(self._recent_geometry)
            tool.show()
            tool.activateWindow()

    @QtCore.Slot()
    def show_clicked(self) -> None:
        if self.archived:
            return
        tool = cast(ImageTool, self.tool)
        tool.show()
        tool.activateWindow()

    @QtCore.Slot()
    def close_tool(self) -> None:
        self.tool = None

    @QtCore.Slot()
    def archive(self) -> None:
        if not self.archived:
            self._archived_fname = os.path.join(
                self.manager.cache_dir, str(uuid.uuid4())
            )
            cast(ImageTool, self.tool).to_pickle(self._archived_fname)

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
    def unarchive(self) -> None:
        if self.archived:
            self.check.setDisabled(False)
            self.visibility_btn.setDisabled(False)

            self.tool = ImageTool.from_pickle(cast(str, self._archived_fname))
            self.tool.show()

            self.manager.sigReloadLinkers.emit()

    @QtCore.Slot()
    def toggle_archive(self) -> None:
        if self.archived:
            self.unarchive()
        else:
            self.archive()


class _ImageToolManagerGUI(QtWidgets.QMainWindow):
    sigLinkersChanged = QtCore.Signal()
    sigReloadLinkers = QtCore.Signal()

    def __init__(self: _ImageToolManagerGUI) -> None:
        super().__init__()
        self.setWindowTitle("ImageTool Manager")
        self.tool_options: dict[int, _ImageToolOptionsWidget] = {}
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
        self.sigLinkersChanged.connect(self._update_button_state)
        self.sigReloadLinkers.connect(self._cleanup_linkers)
        self._update_button_state()

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
        """Number of ImageTool windows being handled by the manager."""
        return len(self.tool_options)

    @property
    def next_idx(self) -> int:
        """Index for the next ImageTool window."""
        return max(self.tool_options.keys(), default=-1) + 1

    def get_tool(self, index: int, unarchive: bool = True) -> ImageTool:
        """Get the ImageTool object corresponding to the index.

        Parameters
        ----------
        index
            Index of the ImageTool window to retrieve.
        unarchive
            Whether to unarchive the tool if it is archived, by default `True`. If set
            to `False`, an error will be raised if the tool is archived.

        Returns
        -------
        ImageTool
            The ImageTool object corresponding to the index.
        """
        if index not in self.tool_options:
            raise KeyError(f"Tool of index '{index}' not found")

        opt = self.tool_options[index]
        if opt.archived:
            if unarchive:
                opt.unarchive()
            else:
                raise KeyError(f"Tool of index '{index}' is archived")
        return cast(ImageTool, opt.tool)

    def deselect_all(self) -> None:
        """Clear selection."""
        for opt in self.tool_options.values():
            opt.check.setChecked(False)

    @QtCore.Slot()
    def add_new(self) -> None:
        """Add a new ImageTool window and open a file dialog."""
        tool = ImageTool(np.zeros((2, 2)))
        self.add_tool(tool, activate=True)

        tool.mnb._open_file(
            name_filter=self._recent_name_filter, directory=self._recent_directory
        )
        self._recent_name_filter = tool.mnb._recent_name_filter
        self._recent_directory = tool.mnb._recent_directory

    def color_for_linker(self, linker: SlicerLinkProxy) -> QtGui.QColor:
        """Get the color that should represent the given linker."""
        idx = self.linkers.index(linker)
        return _LINKER_COLORS[idx % len(_LINKER_COLORS)]

    def add_tool(self, tool: ImageTool, activate: bool = False) -> int:
        """Add a new ImageTool window to the manager and show it.

        Parameters
        ----------
        tool
            ImageTool object to be added.
        activate
            Whether to focus on the window after adding, by default `False`.
        """
        index = int(self.next_idx)
        opt = _ImageToolOptionsWidget(self, index, tool)
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
    def _update_button_state(self) -> None:
        """Update the state of the buttons based on the current selection."""
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

    def remove_tool(self, index: int) -> None:
        opt = self.tool_options.pop(index)
        if not opt.archived:
            cast(ImageTool, opt.tool).removeEventFilter(opt)

        self.options_layout.removeWidget(opt)
        opt.close_tool()
        opt.close()
        del opt
        gc.collect()

    @QtCore.Slot()
    def _cleanup_linkers(self) -> None:
        """Remove linkers with one or no children."""
        for linker in list(self.linkers):
            if linker.num_children <= 1:
                linker.unlink_all()
                self.linkers.remove(linker)
        self.sigLinkersChanged.emit()

    @QtCore.Slot()
    def close_selected(self) -> None:
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
    def link_selected(self, link_colors: bool = True, deselect: bool = True) -> None:
        self.unlink_selected(deselect=False)
        self.link_tools(*self.selected_tool_indices, link_colors=link_colors)
        if deselect:
            self.deselect_all()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    def unlink_selected(self, deselect: bool = True) -> None:
        for index in self.selected_tool_indices:
            self.get_tool(index).slicer_area.unlink()
        self.sigReloadLinkers.emit()
        if deselect:
            self.deselect_all()

    def link_tools(self, *indices, link_colors: bool = True) -> None:
        linker = SlicerLinkProxy(
            *[self.get_tool(t).slicer_area for t in indices], link_colors=link_colors
        )
        self.linkers.append(linker)
        self.sigReloadLinkers.emit()


class ImageToolManager(_ImageToolManagerGUI):
    """The ImageToolManager window.

    This class implements a GUI application for managing multiple ImageTool windows.

    Users do not need to create an instance of this class directly. Instead, use the
    command line script ``itool-manager`` or the function :func:`main
    <erlab.interactive.imagetool.manager.main>` to start the application.

    """

    def __init__(self) -> None:
        super().__init__()
        self.server = _ManagerServer()
        self.server.sigReceived.connect(self.data_recv)
        self.server.start()

        self._shm = shared_memory.SharedMemory(name=_SHM_NAME, create=True, size=1)

    @QtCore.Slot(list, dict)
    def data_recv(self, data: list[xarray.DataArray], kwargs: dict[str, Any]) -> None:
        """
        Slot function to receive data from the server.

        Args:
            data (list[xarray.DataArray]): The data received from the server.
            kwargs (dict[str, Any]): Additional keyword arguments.

        Returns
        -------
            None

        """
        link = kwargs.pop("link", False)
        link_colors = kwargs.pop("link_colors", True)
        indices: list[int] = [
            self.add_tool(ImageTool(d, **kwargs), activate=True) for d in data
        ]

        if link:
            self.link_tools(*indices, link_colors=link_colors)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        """
        Event handler for the close event.

        Args:
            event (QtGui.QCloseEvent | None): The close event.

        Returns
        -------
            None

        """
        if self.ntools != 0:
            if self.ntools == 1:
                msg = "1 remaining window will be closed."
            else:
                msg = f"All {self.ntools} remaining windows will be closed."

            ret = QtWidgets.QMessageBox.question(self, "Do you want to close?", msg)
            if ret != QtWidgets.QMessageBox.StandardButton.Yes:
                if event:
                    event.ignore()
                return

            for tool in list(self.tool_options.keys()):
                self.remove_tool(tool)

        # Clean up temporary directory
        self._tmp_dir.cleanup()

        # Clean up shared memory
        self._shm.close()
        self._shm.unlink()

        # Stop the server
        self.server.stopped.set()
        self.server.wait()

        super().closeEvent(event)


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
    try:
        shm = shared_memory.SharedMemory(name=_SHM_NAME, create=True, size=1)
    except FileExistsError:
        return True
    else:
        shm.unlink()
        return False


def main() -> None:
    """Start the ImageToolManager application.

    Running ``itool-manager`` from a shell will invoke this function.
    """
    qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")
    qapp.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), "icon.png")))

    while is_running():
        dialog = _InitDialog()
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
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
) -> None:
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
    if not is_running():
        raise RuntimeError(
            "ImageToolManager is not running. Please start the ImageToolManager "
            "application before using this function"
        )

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
    kwargs = pickle.dumps(kwargs, protocol=-1)

    # Send the size of the data first
    client_socket.sendall(struct.pack(">L", len(kwargs)))
    client_socket.sendall(kwargs)
    client_socket.close()


if __name__ == "__main__":
    main()
