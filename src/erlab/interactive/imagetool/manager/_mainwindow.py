from __future__ import annotations

import gc
import logging
import os
import pathlib
import platform
import sys
import tempfile
import threading
import typing
import uuid

import numpy as np
import pyqtgraph
import qtpy
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _NameFilterDialog,
    _RenameDialog,
    _StoreDialog,
)
from erlab.interactive.imagetool.manager._io import _MultiFileHandler
from erlab.interactive.imagetool.manager._modelview import _ImageToolWrapperTreeView
from erlab.interactive.imagetool.manager._server import (
    _ManagerServer,
    _WatcherServer,
    show_in_manager,
)
from erlab.interactive.imagetool.manager._wrapper import _ImageToolWrapper

if typing.TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


_SHM_NAME: str = "__enforce_single_itoolmanager"
"""Name of `QtCore.QSharedMemory` that enforces single instance of ImageToolManager.

No longer used starting from v3.8.2, but kept for backward compatibility.
"""

_ICON_PATH = os.path.join(
    os.path.dirname(__file__), "icon.icns" if sys.platform == "darwin" else "icon.png"
)
"""Path to the icon file for the manager window."""


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

_manager_instance: ImageToolManager | None = None
"""Reference to the running manager instance."""

_always_use_socket: bool = False
"""Internal flag to use sockets within same process for test coverage."""


class ItoolManagerParseError(Exception):
    """Raised when the data received from the client cannot be parsed."""


class _SingleImagePreview(QtWidgets.QGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._pixmapitem = typing.cast(
            "QtWidgets.QGraphicsPixmapItem",
            typing.cast("QtWidgets.QGraphicsScene", self.scene()).addPixmap(
                QtGui.QPixmap()
            ),
        )
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.setToolTip("Main image preview")

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmapitem.setPixmap(pixmap)
        self.fitInView(self._pixmapitem)

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.fitInView(self._pixmapitem)

    def wheelEvent(self, event: QtGui.QWheelEvent | None) -> None:
        # Disable scrolling by ignoring wheel events
        if event:
            event.ignore()


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

    _sigDataReplaced = QtCore.Signal()  #: :meta private:
    # Signal emitted when data is replaced in the manager, for testing purposes.

    _sigReplyData = QtCore.Signal(object)  #: :meta private:
    # Signal emitted to reply data requests.

    _sigWatchedDataEdited = QtCore.Signal(str, str, str)  #: :meta private:
    # Signal emitted to notify ipython watchers of data changes.

    def __init__(self) -> None:
        super().__init__()
        self.server: _ManagerServer = _ManagerServer()
        self.server.sigReceived.connect(self._data_recv)
        self.server.sigLoadRequested.connect(self._data_load)
        self.server.sigReplaceRequested.connect(self._data_replace)

        self.server.sigDataRequested.connect(self._send_imagetool_data)
        self._sigReplyData.connect(self.server.set_return_value)
        self.server.sigRemoveIndex.connect(self.remove_imagetool)
        self.server.sigShowIndex.connect(self.show_imagetool)
        self.server.sigRemoveUID.connect(self._remove_watched)
        self.server.sigShowUID.connect(self._show_watched)
        self.server.sigUnwatchUID.connect(self._data_unwatch)
        self.server.sigWatchedVarChanged.connect(self._data_watched_update)
        self.server.start()

        self.watcher_server: _WatcherServer = _WatcherServer()
        self._sigWatchedDataEdited.connect(self.watcher_server.send_parameters)
        self.watcher_server.start()

        # Shared memory for detecting multiple instances
        # No longer used starting from v3.8.2, but kept for backward compatibility
        self._shm = QtCore.QSharedMemory(_SHM_NAME)
        self._shm.create(1)  # Create segment so that it can be attached to

        self.setWindowTitle("ImageTool Manager")

        menu_bar: QtWidgets.QMenuBar = typing.cast("QtWidgets.QMenuBar", self.menuBar())

        self._imagetool_wrappers: dict[int, _ImageToolWrapper] = {}
        self._displayed_indices: list[int] = []
        self._linkers: list[erlab.interactive.imagetool.core.SlicerLinkProxy] = []

        # Stores additional analysis tools opened from child ImageTool windows
        self._additional_windows: dict[str, QtWidgets.QWidget] = {}

        # Initialize actions
        self.settings_action = QtWidgets.QAction("Settings", self)
        self.settings_action.triggered.connect(self.open_settings)
        self.settings_action.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
        self.settings_action.setToolTip("Open settings")

        self.show_action = QtWidgets.QAction("Show", self)
        self.show_action.triggered.connect(self.show_selected)
        self.show_action.setToolTip("Show selected windows")

        self.hide_action = QtWidgets.QAction("Hide", self)
        self.hide_action.triggered.connect(self.hide_selected)
        self.hide_action.setShortcut(QtGui.QKeySequence.StandardKey.Close)
        self.hide_action.setToolTip("Hide selected windows")

        self.gc_action = QtWidgets.QAction("Run Garbage Collection", self)
        self.gc_action.triggered.connect(self.garbage_collect)
        self.gc_action.setToolTip("Run garbage collection to free up memory")

        self.open_action = QtWidgets.QAction("&Open File...", self)
        self.open_action.triggered.connect(self.open)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_action.setToolTip("Open file(s) in ImageTool")

        self.save_action = QtWidgets.QAction("&Save Workspace As...", self)
        self.save_action.setToolTip("Save all windows to a single file")
        self.save_action.triggered.connect(self.save)

        self.load_action = QtWidgets.QAction("&Open Workspace...", self)
        self.load_action.setToolTip("Restore windows from a file")
        self.load_action.triggered.connect(self.load)

        self.remove_action = QtWidgets.QAction("Remove", self)
        self.remove_action.triggered.connect(self.remove_selected)
        self.remove_action.setShortcut(QtGui.QKeySequence.StandardKey.Delete)
        self.remove_action.setToolTip("Remove selected windows")

        self.rename_action = QtWidgets.QAction("Rename", self)
        self.rename_action.triggered.connect(self.rename_selected)
        self.rename_action.setToolTip("Rename selected windows")

        self.duplicate_action = QtWidgets.QAction("Duplicate", self)
        self.duplicate_action.triggered.connect(self.duplicate_selected)
        self.duplicate_action.setToolTip("Duplicate selected windows")

        self.reindex_action = QtWidgets.QAction("Reset Index", self)
        self.reindex_action.triggered.connect(self.reindex)
        self.reindex_action.setToolTip("Reset indices of all windows")

        self.link_action = QtWidgets.QAction("Link", self)
        self.link_action.triggered.connect(self.link_selected)
        self.link_action.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        self.link_action.setToolTip("Link selected windows")

        self.unlink_action = QtWidgets.QAction("Unlink", self)
        self.unlink_action.triggered.connect(self.unlink_selected)
        self.unlink_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+L"))
        self.unlink_action.setToolTip("Unlink selected windows")

        self.archive_action = QtWidgets.QAction("Archive", self)
        self.archive_action.triggered.connect(self.archive_selected)
        self.archive_action.setToolTip("Archive selected windows")

        self.unarchive_action = QtWidgets.QAction("Unarchive", self)
        self.unarchive_action.triggered.connect(self.unarchive_selected)
        self.unarchive_action.setToolTip("Unarchive selected windows")

        self.console_action = QtWidgets.QAction("Console", self)
        self.console_action.triggered.connect(self.toggle_console)
        self.console_action.setShortcut(
            QtGui.QKeySequence("Meta+`" if sys.platform == "darwin" else "Ctrl+`")
        )
        self.console_action.setToolTip("Toggle console window")

        self.preview_action = QtWidgets.QAction("Preview on Hover", self)
        self.preview_action.setCheckable(True)
        self.preview_action.setToolTip("Show preview on hover")

        self.about_action = QtWidgets.QAction("About", self)
        self.about_action.triggered.connect(self.about)

        self.store_action = QtWidgets.QAction("Store with IPython", self)
        self.store_action.triggered.connect(self.store_selected)
        self.store_action.setToolTip("Store selected data with IPython")

        self.explorer_action = QtWidgets.QAction("Data Explorer", self)
        self.explorer_action.triggered.connect(self.show_explorer)
        self.explorer_action.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        self.explorer_action.setToolTip("Show the data explorer window")

        self.concat_action = QtWidgets.QAction("Concatenate", self)
        self.concat_action.triggered.connect(self.concat_selected)
        self.concat_action.setToolTip("Concatenate data in selected windows")

        self.reload_action = QtWidgets.QAction("Reload Data", self)
        self.reload_action.triggered.connect(self.reload_selected)
        self.reload_action.setToolTip("Reload data from file for selected windows")
        self.reload_action.setVisible(False)

        self.unwatch_action = QtWidgets.QAction("Stop Watching", self)
        self.unwatch_action.triggered.connect(self.unwatch_selected)
        self.unwatch_action.setToolTip("Stop watching selected windows")
        self.unwatch_action.setVisible(False)

        # Populate menu bar
        file_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", menu_bar.addMenu("&File")
        )
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.explorer_action)
        file_menu.addSeparator()
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.store_action)
        file_menu.addSeparator()
        file_menu.addAction(self.gc_action)
        file_menu.addSeparator()
        file_menu.addAction(self.about_action)
        file_menu.addAction(self.settings_action)

        edit_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", menu_bar.addMenu("&Edit")
        )
        edit_menu.addAction(self.reindex_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.concat_action)
        edit_menu.addAction(self.duplicate_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.show_action)
        edit_menu.addAction(self.hide_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.remove_action)
        edit_menu.addAction(self.archive_action)
        edit_menu.addAction(self.unarchive_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.rename_action)
        edit_menu.addAction(self.link_action)
        edit_menu.addAction(self.unlink_action)

        view_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", menu_bar.addMenu("&View")
        )
        view_menu.addAction(self.console_action)
        view_menu.addSeparator()
        view_menu.addAction(self.preview_action)
        view_menu.addSeparator()

        # Initialize sidebar buttons linked to actions
        self.open_button = erlab.interactive.utils.IconActionButton(
            self.open_action, "mdi6.folder-file"
        )
        self.remove_button = erlab.interactive.utils.IconActionButton(
            self.remove_action, "mdi6.window-close"
        )
        self.rename_button = erlab.interactive.utils.IconActionButton(
            self.rename_action, "mdi6.rename"
        )
        self.link_button = erlab.interactive.utils.IconActionButton(
            self.link_action, "mdi6.link-variant"
        )
        self.unlink_button = erlab.interactive.utils.IconActionButton(
            self.unlink_action, "mdi6.link-variant-off"
        )
        self.preview_button = erlab.interactive.utils.IconActionButton(
            self.preview_action, on="mdi6.eye", off="mdi6.eye-off"
        )

        # Initialize GUI
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # Construct left side of splitter
        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        main_splitter.addWidget(left_container)

        titlebar = QtWidgets.QWidget()
        titlebar_layout = QtWidgets.QVBoxLayout()
        titlebar.setLayout(titlebar_layout)
        titlebar_layout.addWidget(self.open_button)
        titlebar_layout.addWidget(self.remove_button)
        titlebar_layout.addWidget(self.rename_button)
        titlebar_layout.addWidget(self.link_button)
        titlebar_layout.addWidget(self.unlink_button)
        titlebar_layout.addStretch()
        left_layout.addWidget(titlebar)

        self.tree_view = _ImageToolWrapperTreeView(self)
        self.tree_view._selection_model.selectionChanged.connect(self._update_actions)
        self.tree_view._selection_model.selectionChanged.connect(self._update_info)
        self.tree_view._model.dataChanged.connect(self._update_info)
        left_layout.addWidget(self.tree_view)

        # Construct right side of splitter
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_splitter.addWidget(right_splitter)

        self.text_box = QtWidgets.QTextEdit(self)
        self.text_box.setReadOnly(True)
        right_splitter.addWidget(self.text_box)

        self.preview_widget = _SingleImagePreview(self)
        right_splitter.addWidget(self.preview_widget)

        # Set initial splitter sizes
        right_splitter.setSizes([300, 100])
        main_splitter.setSizes([100, 150])

        # Temporary directory for storing archived data
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="erlab_archive_")

        # Store most recent name filter and directory for new windows
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None

        self.sigLinkersChanged.connect(self._update_actions)
        self.sigLinkersChanged.connect(self.tree_view.refresh)
        self._sigReloadLinkers.connect(self._cleanup_linkers)
        self._update_actions()
        self._update_info()

        # Golden ratio :)
        self.setMinimumWidth(301)
        self.setMinimumHeight(487)
        self.resize(487, 487)

        # Install event filter for keyboard shortcuts
        self._kb_filter = erlab.interactive.utils.KeyboardEventFilter(self)
        self.text_box.installEventFilter(self._kb_filter)

        # File handlers for multithreaded file loading
        self._file_handlers: set[_MultiFileHandler] = set()

        # Initialize status bar
        self._status_bar.showMessage("")

    @property
    def cache_dir(self) -> str:
        """Name of the cache directory where archived data are stored."""
        return self._tmp_dir.name

    @property
    def ntools(self) -> int:
        """Number of ImageTool windows being handled by the manager."""
        return len(self._imagetool_wrappers)

    @property
    def next_idx(self) -> int:
        """Index for the next window."""
        return max(self._imagetool_wrappers.keys(), default=-1) + 1

    @property
    def _status_bar(self) -> QtWidgets.QStatusBar:
        return typing.cast("QtWidgets.QStatusBar", self.statusBar())

    @QtCore.Slot()
    def about(self) -> None:
        """Show the about dialog."""
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIconPixmap(QtGui.QIcon(_ICON_PATH).pixmap(64, 64))
        msg_box.setText("About ImageTool Manager")

        version_info = {
            "erlab": erlab.__version__,
            "numpy": np.__version__,
            "xarray": xr.__version__,
            "pyqtgraph": pyqtgraph.__version__,
            "Qt": f"{qtpy.API_NAME} {qtpy.QT_VERSION}",
            "Python": platform.python_version(),
            "OS": platform.platform(),
        }
        msg_box.setInformativeText(
            "\n".join(f"{k}: {v}" for k, v in version_info.items())
        )
        msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        copy_btn = msg_box.addButton(
            "Copy", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        msg_box.exec()

        if msg_box.clickedButton() == copy_btn:
            cb = QtWidgets.QApplication.clipboard()
            if cb:
                cb.setText(msg_box.informativeText())

    @QtCore.Slot()
    def reindex(self) -> None:
        """Reset indices of ImageTool windows to be consecutive in displayed order."""
        lock = getattr(self, "_reindex_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._reindex_lock = lock

        with lock:
            new_imagetool_wrappers: dict[int, _ImageToolWrapper] = {}
            displayed_indices = list(self._displayed_indices)
            for row_idx, tool_idx in enumerate(displayed_indices):
                self._displayed_indices[row_idx] = row_idx
                self._imagetool_wrappers[tool_idx]._index = row_idx
                new_imagetool_wrappers[row_idx] = self._imagetool_wrappers[tool_idx]
            self._imagetool_wrappers = new_imagetool_wrappers

        self.tree_view.refresh()

    def get_imagetool(self, index: int, unarchive: bool = True) -> ImageTool:
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
        if index not in self._imagetool_wrappers:
            raise KeyError(f"Tool of index '{index}' not found")

        wrapper = self._imagetool_wrappers[index]
        if wrapper.archived:
            if unarchive:
                wrapper.unarchive()
            else:
                raise KeyError(f"Tool of index '{index}' is archived")
        return typing.cast("ImageTool", wrapper.imagetool)

    def color_for_linker(
        self, linker: erlab.interactive.imagetool.core.SlicerLinkProxy
    ) -> QtGui.QColor:
        """Get the color that should represent the given linker."""
        idx = self._linkers.index(linker)
        return _LINKER_COLORS[idx % len(_LINKER_COLORS)]

    def add_imagetool(
        self,
        tool: ImageTool,
        activate: bool = False,
        *,
        watched_var: tuple[str, str] | None = None,
    ) -> int:
        """Add a new ImageTool window to the manager and show it.

        Parameters
        ----------
        tool
            ImageTool object to be added.
        activate
            Whether to focus on the window after adding, by default `False`.
        watched_var
            If the tool is created from a watched variable, this should be a tuple of
            the variable name and its unique ID.

        Returns
        -------
        int
            Index of the added ImageTool window.
        """
        index = int(self.next_idx)
        wrapper = _ImageToolWrapper(self, index, tool, watched_var=watched_var)
        self._imagetool_wrappers[index] = wrapper
        wrapper.update_title()

        self._sigReloadLinkers.emit()

        tool.show()

        if activate:
            tool.activateWindow()
            tool.raise_()

        # Add to view after initialization
        self.tree_view.imagetool_added(index)

        return index

    @QtCore.Slot()
    @QtCore.Slot(str)
    def _update_info(self, uid: str | None = None) -> None:
        """Update the information text box.

        If a string ``uid`` is provided, the function will update the info box only if
        the given ``uid`` is the only selected child tool.
        """
        selected_imagetools = self.tree_view.selected_imagetool_indices
        selected_childtools = self.tree_view.selected_childtool_uids

        n_itool: int = len(selected_imagetools)
        n_total: int = n_itool + len(selected_childtools)

        if (uid is not None) and ((n_total != 1) or (uid not in selected_childtools)):
            return

        match n_total:
            case 0:
                self.text_box.setPlainText("Select a window to view its information.")
                self.preview_widget.setVisible(False)

            case 1:
                if n_itool > 0:
                    wrapper = self._imagetool_wrappers[selected_imagetools[0]]
                    self.text_box.setHtml(wrapper.info_text)
                    self.preview_widget.setPixmap(wrapper._preview_image[1])
                    self.preview_widget.setVisible(True)
                else:
                    childtool = self.get_childtool(selected_childtools[0])
                    info: str = erlab.interactive.utils._apply_qt_accent_color(
                        childtool.info_text
                    )
                    self.text_box.setHtml(info)
                    image_item = childtool.preview_imageitem
                    if image_item is None:
                        self.preview_widget.setVisible(False)
                    else:
                        self.preview_widget.setPixmap(
                            image_item.getPixmap().transformed(
                                QtGui.QTransform().scale(1.0, -1.0)
                            )
                        )
                        self.preview_widget.setVisible(True)

            case _:
                self.text_box.setHtml(
                    "<p><b>Selected ImageTool windows</b></p>"
                    + "<br>".join(
                        self._imagetool_wrappers[i].label_text
                        for i in selected_imagetools
                    )
                )
                self.preview_widget.setVisible(False)

    @QtCore.Slot()
    def _update_actions(self) -> None:
        """Update the state of the actions based on the current selection."""
        selection_archived: list[int] = []
        selection_unarchived: list[int] = []
        selection_watched: list[int] = []
        selection_children: list[str] = list(self.tree_view.selected_childtool_uids)

        for s in self.tree_view.selected_imagetool_indices:
            wrapper = self._imagetool_wrappers[s]
            if wrapper.archived:
                selection_archived.append(s)
            else:
                selection_unarchived.append(s)
            if wrapper.watched:
                selection_watched.append(s)

        selection_all = selection_archived + selection_unarchived

        something_selected: bool = len(selection_all) != 0
        multiple_selected: bool = len(selection_all) > 1
        only_unarchived: bool = len(selection_archived) == 0
        only_archived: bool = len(selection_unarchived) == 0

        self.show_action.setEnabled(something_selected)
        self.hide_action.setEnabled(something_selected)
        self.remove_action.setEnabled(something_selected)
        self.rename_action.setEnabled(something_selected and only_unarchived)
        self.duplicate_action.setEnabled(something_selected)
        self.archive_action.setEnabled(something_selected and only_unarchived)
        self.unarchive_action.setEnabled(something_selected and only_archived)
        self.concat_action.setEnabled(multiple_selected)
        self.store_action.setEnabled(something_selected and only_unarchived)

        self.reload_action.setVisible(
            something_selected
            and only_unarchived
            and all(
                self._imagetool_wrappers[s].slicer_area.reloadable
                for s in selection_unarchived
            )
        )
        self.unwatch_action.setVisible(
            something_selected and len(selection_watched) == len(selection_all)
        )

        self.link_action.setDisabled(only_archived)
        self.unlink_action.setDisabled(only_archived)

        if len(selection_children) != 0:
            self.show_action.setEnabled(True)
            self.hide_action.setEnabled(True)
            self.remove_action.setEnabled(True)
            self.duplicate_action.setEnabled(True)

        if only_unarchived:
            match len(selection_unarchived):
                case 0:
                    self.link_action.setDisabled(True)
                    self.unlink_action.setDisabled(True)
                    return
                case 1:
                    self.link_action.setDisabled(True)
                case _:
                    self.link_action.setDisabled(False)

            is_linked: list[bool] = [
                self.get_imagetool(index).slicer_area.is_linked
                for index in selection_unarchived
            ]
            self.unlink_action.setEnabled(any(is_linked))

            if all(is_linked):
                proxies = [
                    self.get_imagetool(index).slicer_area._linking_proxy
                    for index in selection_unarchived
                ]
                if all(p == proxies[0] for p in proxies):  # pragma: no branch
                    self.link_action.setEnabled(False)

    @QtCore.Slot(int)
    def remove_imagetool(self, index: int) -> None:
        """Remove the ImageTool window corresponding to the given index."""
        # Remove all child tools first
        for uid in list(self._imagetool_wrappers[index]._childtool_indices):
            self._remove_childtool(uid)

        self.tree_view.imagetool_removed(index)
        wrapper = self._imagetool_wrappers.pop(index)
        if not wrapper.archived:
            typing.cast("ImageTool", wrapper.imagetool).removeEventFilter(wrapper)

        wrapper.dispose()
        del wrapper

    def remove_all_tools(self) -> None:
        """Remove all ImageTool windows."""
        for index in tuple(self._imagetool_wrappers.keys()):
            self.remove_imagetool(index)

    @QtCore.Slot(int)
    def show_imagetool(self, index: int) -> None:
        """Show the ImageTool window corresponding to the given index."""
        if index in self._imagetool_wrappers:
            self._imagetool_wrappers[index].show()

    @QtCore.Slot()
    def _cleanup_linkers(self) -> None:
        """Remove linkers with one or no children."""
        for linker in list(self._linkers):
            if linker.num_children <= 1:
                linker.unlink_all()
                self._linkers.remove(linker)
        self.sigLinkersChanged.emit()

    @QtCore.Slot()
    def show_selected(self) -> None:
        """Show selected windows."""
        index_list = self.tree_view.selected_imagetool_indices

        require_unarchive = any(
            self._imagetool_wrappers[i].archived for i in index_list
        )
        if require_unarchive:
            # This is just to display the wait dialog for unarchiving.
            # If this part is removed, the showing will just hang until the unarchiving
            # is finished without any feedback.
            self.unarchive_selected()

        for index in index_list:
            self.show_imagetool(index)

        uid_list = self.tree_view.selected_childtool_uids

        for uid in uid_list:
            self.show_childtool(uid)

    @QtCore.Slot()
    def hide_selected(self) -> None:
        """Hide selected ImageTool windows."""
        for index in self.tree_view.selected_imagetool_indices:
            self._imagetool_wrappers[index].close()
        for uid in self.tree_view.selected_childtool_uids:
            self.get_childtool(uid).hide()

    @QtCore.Slot()
    def hide_all(self) -> None:
        """Hide all ImageTool windows."""
        for tool in self._imagetool_wrappers.values():
            tool.close()

    @QtCore.Slot()
    def reload_selected(self) -> None:
        """Reload data in selected ImageTool windows."""
        for index in self.tree_view.selected_imagetool_indices:
            self._imagetool_wrappers[index].slicer_area.reload()

    @QtCore.Slot()
    def remove_selected(self) -> None:
        """Discard selected ImageTool windows."""
        indices = list(self.tree_view.selected_imagetool_indices)
        child_uids = list(self.tree_view.selected_childtool_uids)

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Remove selected windows?")

        count: int = len(indices)
        num_selected_children: int = len(child_uids)
        num_implicit_children: int = 0
        for i in indices:
            for uid in self._imagetool_wrappers[i]._childtool_indices:
                if uid not in child_uids:
                    num_implicit_children += 1

        text = f"{count} selected ImageTool window{'' if count == 1 else 's'}"
        if num_implicit_children > 0:
            text += (
                f", along with {num_implicit_children} associated child tool"
                f"{'' if num_implicit_children == 1 else 's'}"
            )
        if num_selected_children > 0:
            text += (
                f" and {num_selected_children} selected child tool"
                f"{'' if num_selected_children == 1 else 's'}"
            )
        text += " will be removed."

        msg_box.setInformativeText(text)
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

        if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
            for name in indices:
                self.remove_imagetool(name)
            for uid in child_uids:
                self._remove_childtool(uid)

    @QtCore.Slot()
    def rename_selected(self) -> None:
        """Rename selected ImageTool windows."""
        selected = self.tree_view.selected_imagetool_indices
        if len(selected) == 1:
            self.tree_view.edit(self.tree_view._model._row_index(selected[0]))
            return
        dialog = _RenameDialog(
            self, [self._imagetool_wrappers[i].name for i in selected]
        )
        dialog.exec()

    @QtCore.Slot()
    def duplicate_selected(self) -> None:
        """Duplicate selected windows."""
        indices = list(self.tree_view.selected_imagetool_indices)
        child_uids = list(self.tree_view.selected_childtool_uids)
        self.tree_view.deselect_all()

        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self.tree_view.selectionModel()
        )
        for index in indices:
            new_index = self.duplicate_imagetool(index)

            qmodelindex = self.tree_view._model._row_index(new_index)

            selection_model.select(
                QtCore.QItemSelection(qmodelindex, qmodelindex),
                QtCore.QItemSelectionModel.SelectionFlag.Select,
            )

        for uid in child_uids:
            new_uid = self.duplicate_childtool(uid)

            qmodelindex = self.tree_view._model._row_index(new_uid)

            selection_model.select(
                QtCore.QItemSelection(qmodelindex, qmodelindex),
                QtCore.QItemSelectionModel.SelectionFlag.Select,
            )

    @QtCore.Slot()
    @QtCore.Slot(bool)
    @QtCore.Slot(bool, bool)
    def link_selected(self, link_colors: bool = True, deselect: bool = True) -> None:
        """Link selected ImageTool windows."""
        self.unlink_selected(deselect=False)
        self.link_imagetools(
            *self.tree_view.selected_imagetool_indices, link_colors=link_colors
        )
        if deselect:
            self.tree_view.deselect_all()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    def unlink_selected(self, deselect: bool = True) -> None:
        """Unlink selected ImageTool windows."""
        for index in self.tree_view.selected_imagetool_indices:
            self.get_imagetool(index).slicer_area.unlink()
        self._sigReloadLinkers.emit()
        if deselect:
            self.tree_view.deselect_all()

    @QtCore.Slot()
    def archive_selected(self) -> None:
        """Archive selected ImageTool windows."""
        with erlab.interactive.utils.wait_dialog(self, "Archiving..."):
            for index in self.tree_view.selected_imagetool_indices:
                self._imagetool_wrappers[index].archive()

    @QtCore.Slot()
    def unarchive_selected(self) -> None:
        """Unarchive selected ImageTool windows."""
        with erlab.interactive.utils.wait_dialog(self, "Unarchiving..."):
            for index in self.tree_view.selected_imagetool_indices:
                self._imagetool_wrappers[index].unarchive()

    @QtCore.Slot()
    def concat_selected(self) -> None:
        """Concatenate the selected data using :func:`xarray.concat`."""
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Concatenate",
            "Dimension name:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            "concat_dim",
        )

        if ok and text:
            try:
                show_in_manager(
                    xr.concat(
                        [
                            self.get_imagetool(index).slicer_area._data
                            for index in self.tree_view.selected_imagetool_indices
                        ],
                        dim=text,
                    )
                )
            except Exception:
                logger.exception("Error while concatenating data")
                erlab.interactive.utils.show_traceback(
                    self,
                    "Error",
                    "An error occurred while concatenating data.",
                )
                return

    @QtCore.Slot()
    def store_selected(self) -> None:
        self.ensure_console_initialized()
        dialog = _StoreDialog(self, self.tree_view.selected_imagetool_indices)
        dialog.exec()

    @QtCore.Slot()
    def unwatch_selected(self) -> None:
        """Unwatch selected ImageTool windows."""
        for index in self.tree_view.selected_imagetool_indices:
            self._imagetool_wrappers[index].unwatch()

    def rename_imagetool(self, index: int, new_name: str) -> None:
        """Rename the ImageTool window corresponding to the given index."""
        self._imagetool_wrappers[index].name = new_name

    def duplicate_imagetool(self, index: int) -> int:
        """Duplicate the ImageTool window corresponding to the given index.

        Parameters
        ----------
        index
            Index of the ImageTool window to duplicate.

        Returns
        -------
        int
            Index of the newly created ImageTool window.
        """
        return self.add_imagetool(
            self.get_imagetool(index).duplicate(_in_manager=True), activate=True
        )

    def duplicate_childtool(self, uid: str) -> str:
        """Duplicate the child tool corresponding to the given UID.

        Parameters
        ----------
        uid
            UID of the child tool to duplicate.

        Returns
        -------
        str
            UID of the newly created child tool.
        """
        tool, idx = self._get_childtool_and_parent(uid)
        return self.add_childtool(tool.duplicate(), idx)

    def link_imagetools(self, *indices, link_colors: bool = True) -> None:
        """Link the ImageTool windows corresponding to the given indices."""
        linker = erlab.interactive.imagetool.core.SlicerLinkProxy(
            *[self.get_imagetool(t).slicer_area for t in indices],
            link_colors=link_colors,
        )
        self._linkers.append(linker)
        self._sigReloadLinkers.emit()

    def name_of_imagetool(self, index: int) -> str:
        """Get the name of the ImageTool window corresponding to the given index."""
        return self._imagetool_wrappers[index].name

    def label_of_imagetool(self, index: int) -> str:
        """Get the label of the ImageTool window corresponding to the given index."""
        return self._imagetool_wrappers[index].label_text

    @QtCore.Slot()
    def garbage_collect(self) -> None:
        """Run garbage collection to free up memory."""
        gc.collect()  # pragma: no cover

    def _to_datatree(
        self, close: bool = False, include_children: bool = True
    ) -> xr.DataTree:
        """Convert the current state of the manager to a DataTree object."""
        constructor: dict[str, xr.Dataset] = {}
        for index in tuple(self._imagetool_wrappers.keys()):
            ds = self.get_imagetool(index).to_dataset()
            ds.attrs["itool_title"] = (
                ds.attrs["itool_title"].removeprefix(f"{index}").removeprefix(": ")
            )
            constructor[f"{index}/imagetool"] = ds
            if include_children:
                wrapper = self._imagetool_wrappers[index]
                for i, uid in enumerate(wrapper._childtool_indices):
                    childtool = wrapper._childtools[uid]
                    if childtool.can_save_and_load():
                        constructor[f"{index}/childtools/child{i}"] = (
                            childtool.to_dataset()
                        )
            if close:
                self.remove_imagetool(index)
        tree = xr.DataTree.from_dict(constructor)
        tree.attrs["imagetool_workspace_schema_version"] = 2
        return tree

    def _from_datatree(self, tree: xr.DataTree) -> None:
        """Restore the state of the manager from a DataTree object."""
        with erlab.interactive.utils.wait_dialog(self, "Loading workspace..."):
            if not self._is_datatree_workspace(tree):
                raise ValueError("Not a valid workspace file")

            schema_version = tree.attrs.get("imagetool_workspace_schema_version", 1)
            match schema_version:
                case 1:
                    # Legacy format, only contains imagetools at the root level
                    tree = self._parse_datatree_compat_v1(tree)
                case 2:
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported workspace schema version {schema_version}, "
                        "file may be from a newer version of erlab"
                    )

            dialog = _ChooseFromDataTreeDialog(self, tree, mode="load")
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                for i, node in enumerate(tree.values()):
                    if dialog.imagetool_selected(i):
                        ds = (
                            typing.cast("xr.DataTree", node["imagetool"])
                            .to_dataset(inherit=False)
                            .compute()
                        )
                        new_idx: int = self.add_imagetool(
                            ImageTool.from_dataset(ds, _in_manager=True)
                        )
                        if not ds.attrs.get("itool_visible", True):
                            self.get_imagetool(new_idx).hide()

                    if "childtools" in node:
                        for j, child_node in enumerate(
                            typing.cast("xr.DataTree", node["childtools"]).values()
                        ):
                            if dialog.childtool_selected(i, j):
                                ds = (
                                    typing.cast("xr.DataTree", child_node)
                                    .to_dataset(inherit=False)
                                    .compute()
                                )
                                uid = self.add_childtool(
                                    erlab.interactive.utils.ToolWindow.from_dataset(ds),
                                    new_idx,
                                )
                                if not ds.attrs.get("tool_visible", True):
                                    self.get_childtool(uid).hide()
            tree.close()

    def _parse_datatree_compat_v1(self, tree: xr.DataTree) -> xr.DataTree:
        """Restore the state of the manager from a DataTree object.

        This is for the legacy format where only imagetools are stored at the root level
        (saved from erlab v3.14.1 and earlier).
        """
        return xr.DataTree.from_dict(
            {f"{i}/imagetool": node.dataset for i, node in tree.items()}
        )

    def _is_datatree_workspace(self, tree: xr.DataTree) -> bool:
        """Check if the given DataTree object is a valid workspace file."""
        if "imagetool_workspace_schema_version" in tree.attrs:
            return True
        # Legacy format
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
            with erlab.interactive.utils.wait_dialog(self, "Saving workspace..."):
                self._save_to_file(fname)

    def _save_to_file(self, fname: str):
        tree: xr.DataTree = self._to_datatree()
        dialog = _ChooseFromDataTreeDialog(self, tree, mode="save")
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            for i, key in enumerate(list(tree.keys())):
                if not dialog.imagetool_selected(i):
                    del tree[key]
                    continue

                node = tree[key]
                if "childtools" in node:
                    for j, child_key in enumerate(
                        list(typing.cast("xr.DataTree", node["childtools"]).keys())
                    ):
                        if not dialog.childtool_selected(i, j):
                            del node["childtools"][child_key]
                    if len(node["childtools"]) == 0:
                        del node["childtools"]
        tree.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)

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
                self._from_datatree(xr.open_datatree(fname, engine="h5netcdf"))
            except Exception:
                logger.exception("Error while loading workspace")
                erlab.interactive.utils.show_traceback(
                    self,
                    "Error",
                    "An error occurred while loading the workspace file.",
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
        valid_loaders: dict[str, tuple[Callable, dict]] = (
            erlab.interactive.utils.file_loaders()
        )
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
    def _data_recv(
        self,
        data: list[xr.DataArray] | list[xr.Dataset],
        kwargs: dict[str, typing.Any],
        *,
        watched_var: tuple[str, str] | None = None,
    ) -> list[bool]:
        """Slot function to receive data from the server.

        DataArrays passed to this function are displayed in new ImageTool windows which
        are added to the manager.

        Parameters
        ----------
        data
            A list of xarray.DataArray objects representing the data.

            Also accepts a list of xarray.Dataset objects created with
            ``ImageTool.to_dataset()``, in which case all other parameters are ignored.
        kwargs
            Additional keyword arguments to be passed to the ImageTool.
        watched_var
            If the tool is created from a watched variable, this should be a tuple of
            the variable name and its unique ID.

        Returns
        -------
        flags : list of bool
            List of flags indicating whether the data was successfully received.
        """
        flags: list[bool] = []
        if erlab.utils.misc.is_sequence_of(data, xr.Dataset):
            for ds in data:
                try:
                    self.add_imagetool(
                        ImageTool.from_dataset(ds, _in_manager=True), activate=True
                    )
                except Exception as e:
                    flags.append(False)
                    self._error_creating_imagetool(e)
                else:
                    flags.append(True)
            return flags

        link = kwargs.pop("link", False)
        link_colors = kwargs.pop("link_colors", True)
        indices: list[int] = []
        kwargs["_in_manager"] = True

        for d in data:
            try:
                indices.append(
                    self.add_imagetool(
                        ImageTool(d, **kwargs), activate=True, watched_var=watched_var
                    )
                )
            except Exception as e:
                flags.append(False)
                self._error_creating_imagetool(e)
            else:
                flags.append(True)

        if link:
            self.link_imagetools(*indices, link_colors=link_colors)

        return flags

    @QtCore.Slot(list, str, dict)
    def _data_load(
        self, paths: list[str], loader_name: str, kwargs: dict[str, typing.Any]
    ) -> None:
        """Load data from the given files using the specified loader."""
        self._add_from_multiple_files(
            [],
            [pathlib.Path(p) for p in paths],
            [],
            func=erlab.io.loaders[loader_name].load,
            kwargs=kwargs,
            retry_callback=lambda _: self._data_load(paths, loader_name),
        )

    @QtCore.Slot(list, list)
    def _data_replace(self, data_list: list[xr.DataArray], indices: list[int]) -> None:
        """Replace data in the ImageTool windows with the given data."""
        for darr, idx in zip(data_list, indices, strict=True):
            if idx < 0:
                # Negative index counts from the end
                idx = sorted(self._imagetool_wrappers.keys())[idx]
            elif idx == self.next_idx:
                # If not yet created, add new tool
                self._data_recv([darr], {})
                continue
            self.get_imagetool(idx).slicer_area.set_data(darr)
        self._sigDataReplaced.emit()

    def _find_watched_idx(self, uid: str) -> int | None:
        """Find the index of the watched ImageTool corresponding to the given UID."""
        for k, v in self._imagetool_wrappers.items():
            if v._watched_uid == uid:
                return k
        return None

    @QtCore.Slot(str)
    def _remove_watched(self, uid: str) -> None:
        """Remove the ImageTool corresponding to the given watched variable UID."""
        idx = self._find_watched_idx(uid)
        if idx is not None:  # pragma: no branch
            self.remove_imagetool(idx)

    @QtCore.Slot(str)
    def _show_watched(self, uid: str) -> None:
        """Show the ImageTool corresponding to the given watched variable UID."""
        idx = self._find_watched_idx(uid)
        if idx is not None:
            self.show_imagetool(idx)

    @QtCore.Slot(str, str, object)
    def _data_watched_update(self, varname: str, uid: str, darr: xr.DataArray) -> None:
        """Update ImageTool window corresponding to the given watched variable."""
        idx = self._find_watched_idx(uid)
        if idx is None:
            # If the tool does not exist, create a new one
            self._data_recv([darr], {}, watched_var=(varname, uid))
        else:
            # Update data in the existing tool
            self.get_imagetool(idx).slicer_area.set_data(darr)

    @QtCore.Slot(str)
    def _data_unwatch(self, uid: str) -> None:
        idx = self._find_watched_idx(uid)
        if idx is not None:
            # Convert the tool to a normal one
            self._imagetool_wrappers[idx].unwatch()

    @QtCore.Slot(object)
    def _get_imagetool_data(self, index_or_uid: int | str) -> xr.DataArray | None:
        """Request data from the ImageTool window corresponding to the given index."""
        if isinstance(index_or_uid, str):
            index = self._find_watched_idx(index_or_uid)
        else:
            index = index_or_uid

        if index not in self._imagetool_wrappers:
            return None
        return self.get_imagetool(index).slicer_area._data

    @QtCore.Slot(object)
    def _send_imagetool_data(self, index_or_uid: int | str) -> None:
        """Send data of the ImageTool window corresponding to the given index."""
        self._sigReplyData.emit(self._get_imagetool_data(index_or_uid))

    def ensure_console_initialized(self) -> None:
        """Ensure that the console window is initialized."""
        if not hasattr(self, "console"):
            from erlab.interactive.imagetool.manager._console import (
                _ImageToolManagerJupyterConsole,
            )

            self.console = _ImageToolManagerJupyterConsole(self)

    @QtCore.Slot()
    def toggle_console(self) -> None:
        """Toggle the console window."""
        self.ensure_console_initialized()
        if self.console.isVisible():
            self.console.hide()
        else:
            self.console.show()
            self.console.activateWindow()
            self.console.raise_()
            self.console._console_widget._control.setFocus()

    @property
    def _recent_loader_name(self) -> str | None:
        """Name of the most recently used loader."""
        if self._recent_name_filter is not None:
            for k in erlab.io.loaders:
                if self._recent_name_filter in erlab.io.loaders[k].file_dialog_methods:
                    return k
        return None

    def ensure_explorer_initialized(self) -> None:
        """Ensure that the data explorer window is initialized."""
        if not hasattr(self, "explorer"):
            from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer

            self.explorer = _TabbedExplorer(
                root_path=self._recent_directory, loader_name=self._recent_loader_name
            )

    @QtCore.Slot()
    def show_explorer(self) -> None:
        """Show data explorer window."""
        self.ensure_explorer_initialized()
        self.explorer.show()
        self.explorer.activateWindow()
        self.explorer.raise_()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None) -> None:
        """Handle drag-and-drop operations entering the window."""
        if event:
            mime_data: QtCore.QMimeData | None = event.mimeData()
            if mime_data and mime_data.hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        """Handle drag-and-drop operations dropping files into the window."""
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

                self.open_multiple_files(
                    file_paths, try_workspace=extensions == {".h5"}
                )

    def _show_loaded_info(
        self,
        loaded: list[pathlib.Path],
        canceled: list[pathlib.Path],
        failed: list[pathlib.Path],
        retry_callback: Callable[[list[pathlib.Path]], typing.Any],
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
        loaded, canceled, failed = (
            list(dict.fromkeys(loaded)),
            list(dict.fromkeys(canceled)),
            list(dict.fromkeys(failed)),
        )  # Remove duplicate entries

        n_done, n_fail = len(loaded), len(failed)

        status_msg = f"Loaded {n_done} {'file' if n_done == 1 else 'files'}"
        self._status_bar.showMessage(status_msg, 5000)

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
        n_files: int = len(queued)
        loaded: list[pathlib.Path] = []
        failed: list[pathlib.Path] = []

        if try_workspace:
            for p in list(queued):
                try:
                    dt = xr.open_datatree(p, engine="h5netcdf")
                except Exception:
                    logger.debug("Failed to open %s as datatree workspace", p)
                else:
                    if self._is_datatree_workspace(dt):
                        try:
                            self._from_datatree(dt)
                        except Exception:
                            logger.exception("Error while loading workspace")
                            erlab.interactive.utils.show_traceback(
                                self,
                                "Error",
                                "An error occurred while loading the workspace file.",
                            )
                        finally:
                            queued.remove(p)
                            loaded.append(p)

        if len(queued) == 0:
            return

        # Get loaders applicable to input files
        valid_loaders: dict[str, tuple[Callable, dict]] = (
            erlab.interactive.utils.file_loaders(queued)
        )

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
            self._recent_name_filter = next(iter(valid_loaders.keys()))
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

    def _error_creating_imagetool(self, e: Exception) -> None:
        logger.exception("Error creating ImageTool window")
        erlab.interactive.utils.show_traceback(
            self,
            "Error",
            "An error occurred while creating the ImageTool window. ",
            "The data may be incompatible with ImageTool.",
        )

    def _add_from_multiple_files(
        self,
        loaded: list[pathlib.Path],
        queued: list[pathlib.Path],
        failed: list[pathlib.Path],
        func: Callable,
        kwargs: dict[str, typing.Any],
        retry_callback: Callable,
    ) -> None:
        handler = _MultiFileHandler(self, queued, func, kwargs)
        self._file_handlers.add(handler)

        def _finished_callback() -> None:
            self._show_loaded_info(
                loaded + handler.loaded,
                handler.queued,
                failed + handler.failed,
                retry_callback=retry_callback,
            )
            self._file_handlers.remove(handler)

        handler.sigFinished.connect(_finished_callback)
        handler.start()

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        """Save a reference to an additional window widget.

        This is mainly used for handling tool windows such as goldtool and dtool opened
        from child ImageTool windows. This way, they can stay open even when the
        ImageTool that opened them is archived or removed.

        All additional windows are closed when the manager is closed.

        Only pass widgets that are not associated with a parent widget.

        Parameters
        ----------
        widget
            The widget to add.
        """
        uid = str(uuid.uuid4())
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self._additional_windows[uid] = widget  # Store reference to prevent gc
        widget.destroyed.connect(lambda: self._additional_windows.pop(uid, None))
        widget.show()

    def add_childtool(
        self, tool: erlab.interactive.utils.ToolWindow, index: int
    ) -> str:
        """Register a child tool window.

        This is mainly used for handling tool windows such as goldtool and dtool opened
        from child ImageTool windows.

        Parameters
        ----------
        tool
            The tool window to add.
        index
            Index of the parent ImageTool window.
        """
        uid = self._imagetool_wrappers[index]._add_childtool(tool)
        self.tree_view.childtool_added(uid, index)
        return uid

    def _add_childtool_from_slicerarea(
        self,
        tool: erlab.interactive.utils.ToolWindow,
        parent_slicer_area: erlab.interactive.imagetool.core.ImageSlicerArea,
    ) -> None:
        for idx, wrapper in self._imagetool_wrappers.items():
            if wrapper.slicer_area is parent_slicer_area:
                self.add_childtool(tool, idx)
                return

        # The parent slicer area is not owned by this manager; just keep track of it
        self.add_widget(tool)

    def _get_childtool_and_parent(
        self, uid: str
    ) -> tuple[erlab.interactive.utils.ToolWindow, int]:
        """Get the child tool window and parent index corresponding to the given UID.

        Parameters
        ----------
        uid
            The unique ID of the child tool to get.

        Returns
        -------
        ToolWindow
            The child tool window corresponding to the given UID.
        int
            The index of the parent ImageTool window.
        """
        for idx, wrapper in self._imagetool_wrappers.items():
            if uid in wrapper._childtool_indices:
                return wrapper._childtools[uid], idx
        raise KeyError(f"No child tool with UID {uid} found")

    def get_childtool(self, uid: str) -> erlab.interactive.utils.ToolWindow:
        """Get the child tool window corresponding to the given UID.

        Parameters
        ----------
        uid
            The unique ID of the child tool to get.

        Returns
        -------
        ToolWindow
            The child tool window corresponding to the given UID.
        """
        return self._get_childtool_and_parent(uid)[0]

    def show_childtool(self, uid: str) -> None:
        """Show the child tool window corresponding to the given UID."""
        childtool = self.get_childtool(uid)

        if sys.platform == "win32":  # pragma: no cover
            # On Windows, window flags must be set to bring the window to the top
            childtool.setWindowFlags(
                childtool.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint
            )
            childtool.show()
            childtool.setWindowFlags(
                childtool.windowFlags() & ~QtCore.Qt.WindowType.WindowStaysOnTopHint
            )
        childtool.show()
        childtool.show()
        childtool.activateWindow()
        childtool.raise_()

    def _remove_childtool(self, uid: str) -> None:
        """Unregister a child tool window.

        Parameters
        ----------
        uid
            The unique ID of the child tool to remove.
        """
        self.tree_view.childtool_removed(uid)

        for wrapper in self._imagetool_wrappers.values():
            if uid in wrapper._childtools:
                wrapper._remove_childtool(uid)
                return

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        """Event filter that intercepts select all and copy shortcuts.

        For some operating systems, shortcuts are often intercepted by actions in the
        menu bar. This filter ensures that the shortcuts work as expected when the
        target widget has focus.
        """
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.ShortcutOverride
            and isinstance(obj, QtWidgets.QWidget)
            and obj.hasFocus()
        ):
            event = typing.cast("QtGui.QKeyEvent", event)
            if event.matches(QtGui.QKeySequence.StandardKey.SelectAll) or event.matches(
                QtGui.QKeySequence.StandardKey.Copy
            ):
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _stop_servers(self) -> None:
        """Stop the server thread properly."""
        if self.server.isRunning():  # pragma: no branch
            self.server.stopped.set()
            self.server.wait()
        if self.watcher_server.isRunning():  # pragma: no branch
            self.watcher_server.stopped.set()
            self._sigWatchedDataEdited.emit("", "", "shutdown")
            self.watcher_server.wait()

    # def __del__(self):
    # """Ensure proper cleanup of server thread when the manager is deleted."""
    # self._stop_server()

    @QtCore.Slot()
    def open_settings(self) -> None:
        """Open the settings dialog for the ImageTool manager."""
        dialog = erlab.interactive._options.OptionDialog(self)
        dialog.exec()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        """Handle proper termination of resources before closing the application."""
        if self.ntools != 0:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setText("Close ImageTool Manager?")
            msg_box.setInformativeText(
                "1 remaining window will be removed."
                if self.ntools == 1
                else f"All {self.ntools} remaining windows will be removed."
            )
            msg_box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel
            )
            msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

            if msg_box.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
                if event:
                    event.ignore()
                return

        # Remove all ImageTool windows
        self.remove_all_tools()

        for widget in dict(self._additional_windows).values():
            widget.close()

        # Remove event filters (problematic in CI)
        self.text_box.removeEventFilter(self._kb_filter)
        self.tree_view._delegate._cleanup_filter()

        if hasattr(self, "console"):
            self.console._console_widget.shutdown_kernel()
            self.console.close()

        if hasattr(self, "explorer"):
            self.explorer.close()

        # Clean up temporary directory
        self._tmp_dir.cleanup()

        # Properly close the server
        self._stop_servers()

        super().closeEvent(event)
