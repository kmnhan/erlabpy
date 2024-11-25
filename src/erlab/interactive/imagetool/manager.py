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

import contextlib
import gc
import os
import pathlib
import pickle
import socket
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
import weakref
from collections.abc import ValuesView
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import qtawesome as qta
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.imagetool import ImageTool, _parse_input
from erlab.interactive.imagetool.core import SlicerLinkProxy
from erlab.interactive.utils import (
    IconActionButton,
    IconButton,
    _coverage_resolve_trace,
    file_loaders,
    wait_dialog,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

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

                # Clean up temporary files
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
            ):
                print(
                    f"Failed to unpickle data due to the following error:\n"
                    f"{traceback.format_exc()}"
                )

            conn.close()

        soc.close()


class _QHLine(QtWidgets.QFrame):
    """Horizontal line widget."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)


class _RenameDialog(QtWidgets.QDialog):
    def __init__(self, manager: ImageToolManager, original_names: list[str]) -> None:
        super().__init__(manager)
        self.setWindowTitle("Rename selected tools")
        self._manager = weakref.ref(manager)

        self._layout = QtWidgets.QGridLayout()
        self.setLayout(self._layout)

        self._new_name_lines: list[QtWidgets.QLineEdit] = []

        for i, name in enumerate(original_names):
            line_new = QtWidgets.QLineEdit(name)
            line_new.setPlaceholderText("New name")
            self._layout.addWidget(QtWidgets.QLabel(name), i, 0)
            self._layout.addWidget(QtWidgets.QLabel("→"), i, 1)
            self._layout.addWidget(line_new, i, 2)
            self._new_name_lines.append(line_new)

        fm = self._new_name_lines[0].fontMetrics()
        max_width = max(
            fm.boundingRect(line.text()).width() for line in self._new_name_lines
        )
        for line in self._new_name_lines:
            line.setMinimumWidth(max_width + 10)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self._layout.addWidget(button_box)

    def new_names(self) -> list[str]:
        return [w.text() for w in self._new_name_lines]

    def accept(self) -> None:
        manager = self._manager()
        if manager is not None:
            for index, new_name in zip(
                manager.selected_tool_indices, self.new_names(), strict=True
            ):
                manager.rename_tool(index, new_name)
        super().accept()


class _NameFilterDialog(QtWidgets.QDialog):
    def __init__(self, parent: ImageToolManager, valid_name_filters: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Loader")

        self._valid_name_filters = valid_name_filters

        layout = QtWidgets.QVBoxLayout(self)
        self._button_group = QtWidgets.QButtonGroup(self)

        for i, name in enumerate(valid_name_filters):
            radio_button = QtWidgets.QRadioButton(name)
            self._button_group.addButton(radio_button, i)
            layout.addWidget(radio_button)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def check_filter(self, name_filter: str | None) -> None:
        self._button_group.buttons()[
            self._valid_name_filters.index(name_filter)
            if name_filter in self._valid_name_filters
            else 0
        ].setChecked(True)

    def checked_filter(self) -> str:
        return self._valid_name_filters[self._button_group.checkedId()]


class _ImageToolOptionsWidget(QtWidgets.QWidget):
    def __init__(self, manager: ImageToolManager, index: int, tool: ImageTool) -> None:
        super().__init__()
        self._tool: ImageTool | None = None

        self._manager = weakref.ref(manager)
        self.index: int = index
        self._archived_fname: str | None = None
        self._recent_geometry: QtCore.QRect | None = None

        self.tool = tool

        self.manager.sigLinkersChanged.connect(self.update_link_icon)
        self._setup_gui()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

    @property
    def manager(self) -> ImageToolManager:
        _manager = self._manager()
        if _manager:
            return _manager
        raise LookupError("Parent was destroyed")

    @property
    def tool(self) -> ImageTool | None:
        return self._tool

    @tool.setter
    def tool(self, value: ImageTool | None) -> None:
        if self._tool is None:
            if self._archived_fname is not None:
                # Remove the archived file
                os.remove(self._archived_fname)
                self._archived_fname = None
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

    @property
    def name(self) -> str:
        return self.check.text().removeprefix(f"{self.index}").removeprefix(": ")

    @name.setter
    def name(self, title: str) -> None:
        new_title = f"{self.index}"
        if title != "":
            new_title += f": {title}"
        cast(ImageTool, self.tool).setWindowTitle(new_title)
        self.check.setText(new_title)

    def eventFilter(self, obj, event):
        if obj == self.tool and (
            event.type() == QtCore.QEvent.Type.Show
            or event.type() == QtCore.QEvent.Type.Hide
            or event.type() == QtCore.QEvent.Type.WindowStateChange
        ):
            self.visibility_changed()
        return super().eventFilter(obj, event)

    def _destroyed_callback(self) -> None:
        self.manager._sigReloadLinkers.emit()

    def _setup_gui(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.check = QtWidgets.QCheckBox(cast(ImageTool, self.tool).windowTitle())
        self.check.toggled.connect(self.manager._update_action_state)

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
        self.archive_btn.setToolTip(
            "Archive/Unarchive"
            "\n"
            "Archived windows use minimal resources, but cannot be interacted with."
        )

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
            self.name = title

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
        tool = cast(ImageTool, self.tool)
        self._recent_geometry = tool.geometry()
        self.visibility_btn.blockSignals(True)
        self.visibility_btn.setChecked(tool.isVisible())
        self.visibility_btn.blockSignals(False)

    @QtCore.Slot()
    def toggle_visibility(self) -> None:
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
            with wait_dialog(self.manager, "Archiving..."):
                cast(ImageTool, self.tool).to_file(self._archived_fname)

            self.close_tool()
            self.check.blockSignals(True)
            self.check.setChecked(False)
            self.check.setDisabled(True)
            self.check.blockSignals(False)
            self.visibility_btn.blockSignals(True)
            self.visibility_btn.setChecked(False)
            self.visibility_btn.setDisabled(True)
            self.visibility_btn.blockSignals(False)

            if not self.archive_btn.isChecked():
                self.archive_btn.blockSignals(True)
                self.archive_btn.setChecked(True)
                self.archive_btn.blockSignals(False)

    @QtCore.Slot()
    def unarchive(self) -> None:
        if self.archived:
            self.check.setDisabled(False)
            self.visibility_btn.setDisabled(False)

            if self.archive_btn.isChecked():
                self.archive_btn.blockSignals(True)
                self.archive_btn.setChecked(False)
                self.archive_btn.blockSignals(False)

            with wait_dialog(self.manager, "Unarchiving..."):
                self.tool = ImageTool.from_file(cast(str, self._archived_fname))
            self.tool.show()

            self.manager._sigReloadLinkers.emit()

    @QtCore.Slot()
    def toggle_archive(self) -> None:
        if self.archived:
            self.unarchive()
        else:
            self.archive()


class ImageToolManager(QtWidgets.QMainWindow):
    """The ImageToolManager window.

    This class implements a GUI application for managing multiple ImageTool windows.

    Users do not need to create an instance of this class directly. Instead, use the
    command line script ``itool-manager`` or the function :func:`main
    <erlab.interactive.imagetool.manager.main>` to start the application.


    Signals
    -------
    sigLinkersChanged()
        Signal emitted when the linker state is changed.

    """

    sigLinkersChanged = QtCore.Signal()  #: :meta private:
    _sigReloadLinkers = QtCore.Signal()  #: Emitted when linker state needs refreshing

    def __init__(self: ImageToolManager) -> None:
        super().__init__()
        self.setWindowTitle("ImageTool Manager")

        menu_bar: QtWidgets.QMenuBar = cast(QtWidgets.QMenuBar, self.menuBar())

        self.tool_options: dict[int, _ImageToolOptionsWidget] = {}
        self.linkers: list[SlicerLinkProxy] = []

        self.titlebar = QtWidgets.QWidget()
        self.titlebar_layout = QtWidgets.QHBoxLayout()
        self.titlebar_layout.setContentsMargins(0, 0, 0, 0)
        self.titlebar.setLayout(self.titlebar_layout)

        self.gc_action = QtWidgets.QAction("Run Garbage Collection", self)
        self.gc_action.triggered.connect(self.garbage_collect)
        self.gc_action.setToolTip("Run garbage collection to free up memory")

        self.open_action = QtWidgets.QAction("&Open File...", self)
        self.open_action.triggered.connect(self.open)
        self.open_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self.open_action.setToolTip("Open file(s) in ImageTool")

        self.save_action = QtWidgets.QAction("&Save Workspace As...", self)
        self.save_action.setToolTip("Save all windows to a single file")
        self.save_action.triggered.connect(self.save)

        self.load_action = QtWidgets.QAction("&Open Workspace...", self)
        self.load_action.setToolTip("Restore windows from a file")
        self.load_action.triggered.connect(self.load)

        self.close_action = QtWidgets.QAction("Close Selected", self)
        self.close_action.triggered.connect(self.close_selected)
        self.close_action.setToolTip("Close selected windows")

        self.rename_action = QtWidgets.QAction("Rename Selected", self)
        self.rename_action.triggered.connect(self.rename_selected)
        self.rename_action.setToolTip("Rename selected windows")

        self.link_action = QtWidgets.QAction("Link Selected", self)
        self.link_action.triggered.connect(self.link_selected)
        self.link_action.setToolTip("Link selected windows")

        self.unlink_action = QtWidgets.QAction("Unlink Selected", self)
        self.unlink_action.triggered.connect(self.unlink_selected)
        self.unlink_action.setToolTip("Unlink selected windows")

        self.archive_action = QtWidgets.QAction("Archive Selected", self)
        self.archive_action.triggered.connect(self.archive_selected)
        self.archive_action.setToolTip("Archive selected windows")

        file_menu: QtWidgets.QMenu = cast(QtWidgets.QMenu, menu_bar.addMenu("&File"))
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.gc_action)

        edit_menu: QtWidgets.QMenu = cast(QtWidgets.QMenu, menu_bar.addMenu("&Edit"))
        edit_menu.addAction(self.close_action)
        edit_menu.addAction(self.archive_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.rename_action)
        edit_menu.addAction(self.link_action)
        edit_menu.addAction(self.unlink_action)

        self.open_button = IconActionButton(
            self.open_action,
            "mdi6.folder-open-outline",
        )
        self.close_button = IconActionButton(
            self.close_action,
            "mdi6.close-box-multiple-outline",
        )
        self.rename_button = IconActionButton(
            self.rename_action,
            "mdi6.rename",
        )
        self.link_button = IconActionButton(
            self.link_action,
            "mdi6.link-variant",
        )
        self.unlink_button = IconActionButton(
            self.unlink_action,
            "mdi6.link-variant-off",
        )

        self.titlebar_layout.addWidget(self.open_button)
        self.titlebar_layout.addWidget(self.close_button)
        self.titlebar_layout.addWidget(self.rename_button)
        self.titlebar_layout.addWidget(self.link_button)
        self.titlebar_layout.addWidget(self.unlink_button)

        self.options = QtWidgets.QWidget()
        self.options_layout = QtWidgets.QVBoxLayout()
        self.options.setLayout(self.options_layout)

        self.options_layout.addWidget(self.titlebar)
        self.options_layout.addWidget(_QHLine())
        self.options_layout.addStretch()

        # Temporary directory for storing archived data
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="erlab_archive_")

        # Store most recent name filter and directory for new windows
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None

        self.setCentralWidget(self.options)
        self.sigLinkersChanged.connect(self._update_action_state)
        self._sigReloadLinkers.connect(self._cleanup_linkers)
        self._update_action_state()

        self.server: _ManagerServer = _ManagerServer()
        self.server.sigReceived.connect(self.data_recv)
        self.server.start()

        self._shm = shared_memory.SharedMemory(name=_SHM_NAME, create=True, size=1)

        self.setMinimumWidth(300)
        self.setMinimumHeight(200)

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
        """Get the ImageTool object corresponding to the given index.

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

    @QtCore.Slot()
    def deselect_all(self) -> None:
        """Clear selection."""
        for opt in self.tool_options.values():
            opt.check.setChecked(False)

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

        self._sigReloadLinkers.emit()

        tool.show()

        if activate:
            tool.activateWindow()
            tool.raise_()

        return index

    @QtCore.Slot()
    def _update_action_state(self) -> None:
        """Update the state of the actions based on the current selection."""
        selection = self.selected_tool_indices

        something_selected: bool = len(selection) != 0

        self.close_action.setEnabled(something_selected)
        self.rename_action.setEnabled(something_selected)
        self.archive_action.setEnabled(something_selected)

        match len(selection):
            case 0:
                self.link_action.setDisabled(True)
                self.unlink_action.setDisabled(True)
                return
            case 1:
                self.link_action.setDisabled(True)
            case _:
                self.link_action.setDisabled(False)

        is_linked: list[bool] = [
            self.get_tool(index).slicer_area.is_linked for index in selection
        ]
        self.unlink_action.setEnabled(any(is_linked))

        if all(is_linked):
            proxies = [
                self.get_tool(index).slicer_area._linking_proxy for index in selection
            ]
            if all(p == proxies[0] for p in proxies):
                self.link_action.setEnabled(False)

    def remove_tool(self, index: int) -> None:
        """Remove the ImageTool window corresponding to the given index."""
        opt = self.tool_options.pop(index)
        if not opt.archived:
            cast(ImageTool, opt.tool).removeEventFilter(opt)

        self.options_layout.removeWidget(opt)
        opt.close_tool()
        opt.close()
        del opt

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
        """Close selected ImageTool windows."""
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
    def rename_selected(self) -> None:
        """Rename selected ImageTool windows."""
        dialog = _RenameDialog(
            self, [self.tool_options[i].name for i in self.selected_tool_indices]
        )
        dialog.exec()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    @QtCore.Slot(bool, bool)
    def link_selected(self, link_colors: bool = True, deselect: bool = True) -> None:
        """Link selected ImageTool windows."""
        self.unlink_selected(deselect=False)
        self.link_tools(*self.selected_tool_indices, link_colors=link_colors)
        if deselect:
            self.deselect_all()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    def unlink_selected(self, deselect: bool = True) -> None:
        """Unlink selected ImageTool windows."""
        for index in self.selected_tool_indices:
            self.get_tool(index).slicer_area.unlink()
        self._sigReloadLinkers.emit()
        if deselect:
            self.deselect_all()

    @QtCore.Slot()
    def archive_selected(self) -> None:
        """Archive selected ImageTool windows."""
        for index in self.selected_tool_indices:
            self.tool_options[index].archive()

    def rename_tool(self, index: int, new_name: str) -> None:
        """Rename the ImageTool window corresponding to the given index."""
        self.tool_options[index].name = new_name

    def link_tools(self, *indices, link_colors: bool = True) -> None:
        """Link the ImageTool windows corresponding to the given indices."""
        linker = SlicerLinkProxy(
            *[self.get_tool(t).slicer_area for t in indices], link_colors=link_colors
        )
        self.linkers.append(linker)
        self._sigReloadLinkers.emit()

    @QtCore.Slot()
    def garbage_collect(self) -> None:
        """Run garbage collection to free up memory."""
        gc.collect()

    def _to_datatree(self, close: bool = False) -> xr.DataTree:
        """Convert the current state of the manager to a DataTree object."""
        constructor: dict[str, xr.Dataset] = {}
        for index in tuple(self.tool_options.keys()):
            ds = self.get_tool(index).to_dataset()
            ds.attrs["itool_title"] = (
                ds.attrs["itool_title"].removeprefix(f"{index}").removeprefix(": ")
            )
            constructor[str(index)] = ds
            if close:
                self.remove_tool(index)
        tree = xr.DataTree.from_dict(constructor)
        tree.attrs["is_itool_workspace"] = 1
        return tree

    def _from_datatree(self, tree: xr.DataTree) -> None:
        """Restore the state of the manager from a DataTree object."""
        if not self._is_datatree_workspace(tree):
            raise ValueError("Not a valid workspace file")
        for node in cast(ValuesView[xr.DataTree], (tree.values())):
            self.add_tool(
                ImageTool.from_dataset(node.to_dataset(inherit=False), _in_manager=True)
            )

    def _is_datatree_workspace(self, tree: xr.DataTree) -> bool:
        """Check if the given DataTree object is a valid workspace file."""
        return tree.attrs.get("is_itool_workspace", 0) == 1

    @QtCore.Slot()
    def save(self, *, native: bool = True) -> None:
        """Save the current state of the manager to a file.

        Parameters
        ----------
        native
            Whether to use the native file dialog, by default `True`. This option is
            used when testing the application to ensure reproducibility.
        """
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("xarray HDF5 Files (*.h5)")
        dialog.setDefaultSuffix("h5")
        if self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            fname = dialog.selectedFiles()[0]
            self._recent_directory = os.path.dirname(fname)
            with wait_dialog(self, "Saving workspace..."):
                self._to_datatree().to_netcdf(
                    fname, engine="h5netcdf", invalid_netcdf=True
                )

    @QtCore.Slot()
    def load(self, *, native: bool = True) -> None:
        """Load the state of the manager from a file.

        Parameters
        ----------
        native
            Whether to use the native file dialog, by default `True`. This option is
            used when testing the application to ensure reproducibility.
        """
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("xarray HDF5 Files (*.h5)")
        if self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            fname = dialog.selectedFiles()[0]
            self._recent_directory = os.path.dirname(fname)
            try:
                with wait_dialog(self, "Loading workspace..."):
                    self._from_datatree(xr.open_datatree(fname, engine="h5netcdf"))
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    "An error occurred while loading the workspace file:\n\n"
                    f"{type(e).__name__}: {e}",
                    QtWidgets.QMessageBox.StandardButton.Ok,
                )
                self.load()

    @QtCore.Slot()
    def open(self, *, native: bool = True) -> None:
        """Open files in a new ImageTool window.

        Parameters
        ----------
        native
            Whether to use the native file dialog, by default `True`. This option is
            used when testing the application to ensure reproducibility.
        """
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        valid_loaders: dict[str, tuple[Callable, dict]] = file_loaders()
        dialog.setNameFilters(valid_loaders.keys())
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if self._recent_name_filter is not None:
            dialog.selectNameFilter(self._recent_name_filter)
        if self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)

        if dialog.exec():
            file_names = dialog.selectedFiles()
            self._recent_name_filter = dialog.selectedNameFilter()
            self._recent_directory = os.path.dirname(file_names[0])
            func, kwargs = valid_loaders[self._recent_name_filter]
            self._add_from_multiple_files(
                loaded=[],
                queued=[pathlib.Path(f) for f in file_names],
                failed=[],
                func=func,
                kwargs=kwargs,
                retry_callback=lambda _: self.open(),
            )

    @QtCore.Slot(list, dict)
    def data_recv(self, data: list[xarray.DataArray], kwargs: dict[str, Any]) -> None:
        """Slot function to receive data from the server.

        DataArrays passed to this function are displayed in new ImageTool windows which
        are added to the manager.

        Parameters
        ----------
        data
            A list of xarray.DataArray objects received from the server.
        kwargs
            Additional keyword arguments received from the server.

        """
        link = kwargs.pop("link", False)
        link_colors = kwargs.pop("link_colors", True)
        indices: list[int] = []
        kwargs["_in_manager"] = True

        for d in data:
            try:
                indices.append(self.add_tool(ImageTool(d, **kwargs), activate=True))
            except Exception as e:
                self._error_creating_tool(e)

        if link:
            self.link_tools(*indices, link_colors=link_colors)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None):
        if event:
            mime_data: QtCore.QMimeData | None = event.mimeData()
            if mime_data and mime_data.hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent | None):
        if event:
            mime_data: QtCore.QMimeData | None = event.mimeData()
            if mime_data and mime_data.hasUrls():
                urls = mime_data.urls()
                file_paths: list[pathlib.Path] = [
                    pathlib.Path(url.toLocalFile()) for url in urls
                ]
                extensions: set[str] = {file_path.suffix for file_path in file_paths}
                if len(extensions) != 1:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Error",
                        "Multiple file types are not supported in a single "
                        "drag-and-drop operation.",
                    )
                    return

                msg = f"Loading {'file' if len(file_paths) == 1 else 'files'}..."
                with wait_dialog(self, msg):
                    self.open_multiple_files(
                        file_paths, try_workspace=extensions == {".h5"}
                    )

    def _show_loaded_info(
        self,
        loaded: list[pathlib.Path],
        canceled: list[pathlib.Path],
        failed: list[pathlib.Path],
        retry_callback: Callable[[list[pathlib.Path]], Any],
    ) -> None:
        """Show a message box with information about the loaded files.

        Nothing is shown if all files were successfully loaded.

        Parameters
        ----------
        loaded
            List of successfully loaded files.
        canceled
            List of files that were aborted before trying to load.
        failed
            List of files that failed to load.
        retry_callback
            Callback function to retry loading the failed files. It should accept a list
            of path objects as its only argument.

        """
        n_done, n_fail = len(loaded), len(failed)
        if n_fail == 0:
            return

        message = f"Loaded {n_done} file"
        if n_done != 1:
            message += "s"
        message = message + f" with {n_fail} failure"
        if n_fail != 1:
            message += "s"
        message += "."

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setText(message)

        loaded_str = "\n".join(p.name for p in loaded)
        if loaded_str:
            loaded_str = f"Loaded:\n{loaded_str}\n\n"

        failed_str = "\n".join(p.name for p in failed)
        if failed_str:
            failed_str = f"Failed:\n{failed_str}\n\n"

        canceled_str = "\n".join(p.name for p in canceled)
        if canceled_str:
            canceled_str = f"Canceled:\n{canceled_str}\n\n"
        msg_box.setDetailedText(f"{loaded_str}{failed_str}{canceled_str}")

        msg_box.setInformativeText("Do you want to retry loading the failed files?")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Retry
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Retry)
        if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Retry:
            retry_callback(failed)

    def open_multiple_files(
        self, queued: list[pathlib.Path], try_workspace: bool = False
    ) -> None:
        """Open multiple files in the manager."""
        n_files: int = int(len(queued))
        loaded: list[pathlib.Path] = []
        failed: list[pathlib.Path] = []

        if try_workspace:
            for p in list(queued):
                try:
                    dt = xr.open_datatree(p, engine="h5netcdf")
                except Exception:
                    pass
                else:
                    if self._is_datatree_workspace(dt):
                        self._from_datatree(dt)
                        queued.remove(p)
                        loaded.append(p)

        if len(queued) == 0:
            return

        # Get loaders applicable to input files
        valid_loaders: dict[str, tuple[Callable, dict]] = file_loaders(queued)

        if len(valid_loaders) == 0:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"The selected {'file' if n_files == 1 else 'files'} "
                f"with extension '{queued[0].suffix}' is not supported by "
                "any available plugin.",
            )
            return

        if len(valid_loaders) == 1:
            func, kargs = next(iter(valid_loaders.values()))
        else:
            valid_name_filters: list[str] = list(valid_loaders.keys())

            dialog = _NameFilterDialog(self, valid_name_filters)
            dialog.check_filter(self._recent_name_filter)

            if dialog.exec():
                selected = dialog.checked_filter()
                func, kargs = valid_loaders[selected]
                self._recent_name_filter = selected
            else:
                return

        self._add_from_multiple_files(
            loaded, queued, failed, func, kargs, self.open_multiple_files
        )

    def _error_creating_tool(self, e: Exception) -> None:
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg_box.setText("An error occurred while creating the ImageTool window.")
        msg_box.setInformativeText("The data may be incompatible with ImageTool.")
        msg_box.setDetailedText(f"{type(e).__name__}: {e}")
        msg_box.exec()

    def _add_from_multiple_files(
        self,
        loaded: list[pathlib.Path],
        queued: list[pathlib.Path],
        failed: list[pathlib.Path],
        func: Callable,
        kwargs: dict[str, Any],
        retry_callback: Callable,
    ) -> None:
        for p in list(queued):
            queued.remove(p)
            try:
                data = func(p, **kwargs)
            except Exception as e:
                failed.append(p)

                msg_box = QtWidgets.QMessageBox(self)
                msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                msg_box.setText(f"Failed to load {p.name}")
                msg_box.setInformativeText(
                    "Do you want to skip this file and continue loading?"
                )
                msg_box.setStandardButtons(
                    QtWidgets.QMessageBox.StandardButton.Abort
                    | QtWidgets.QMessageBox.StandardButton.Yes
                )
                msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)
                msg_box.setDetailedText(f"{type(e).__name__}: {e}")
                match msg_box.exec():
                    case QtWidgets.QMessageBox.StandardButton.Yes:
                        continue
                    case QtWidgets.QMessageBox.StandardButton.Abort:
                        break
            else:
                try:
                    tool = ImageTool(np.zeros((2, 2)), _in_manager=True)
                    tool._recent_name_filter = self._recent_name_filter
                    tool._recent_directory = self._recent_directory
                    tool.slicer_area.set_data(data, file_path=p)
                except Exception as e:
                    failed.append(p)
                    self._error_creating_tool(e)
                else:
                    loaded.append(p)
                    self.add_tool(tool, activate=True)

        self._show_loaded_info(loaded, queued, failed, retry_callback=retry_callback)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        """Properly clear all resources before closing the application."""
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
        shm.close()
        shm.unlink()
        return False


def main() -> None:
    """Start the ImageToolManager application.

    Running ``itool-manager`` from a shell will invoke this function.
    """
    qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")

    qapp.setWindowIcon(
        QtGui.QIcon(
            os.path.join(
                os.path.dirname(__file__),
                "icon.icns" if sys.platform == "darwin" else "icon.png",
            )
        )
    )
    qapp.setApplicationDisplayName("ImageTool Manager")

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
    tmp_dir = tempfile.mkdtemp(prefix="erlab_manager_")

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
