from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import pathlib
import platform
import sys
import tempfile
import threading
import traceback
import typing
import uuid
from dataclasses import dataclass

import numpy as np
import pyqtgraph
import qtpy
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._dask import DaskMenu
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ConcatDialog,
    _is_loader_func,
    _NameFilterDialog,
    _RenameDialog,
    _StoreDialog,
)
from erlab.interactive.imagetool.manager._io import _MultiFileHandler
from erlab.interactive.imagetool.manager._logging import get_log_file_path
from erlab.interactive.imagetool.manager._modelview import _ImageToolWrapperTreeView
from erlab.interactive.imagetool.manager._server import _ManagerServer, _WatcherServer
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _LoadSourceDetails,
    _ManagedWindowNode,
    _MetadataField,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.ptable import PeriodicTableWindow

logger = logging.getLogger(__name__)

_METADATA_DERIVATION_CODE_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
_METADATA_DERIVATION_COPYABLE_ROLE = _METADATA_DERIVATION_CODE_ROLE + 1


class _WarningEmitter(QtCore.QObject):
    warning_received = QtCore.Signal(str, int, str, str)


class _MetadataDerivationListWidget(QtWidgets.QListWidget):
    copy_requested = QtCore.Signal()
    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class _ElidedInteractiveLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def __init__(
        self,
        text: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        elide_mode: QtCore.Qt.TextElideMode = QtCore.Qt.TextElideMode.ElideMiddle,
    ) -> None:
        super().__init__(parent)
        self._full_text = text
        self._elide_mode = elide_mode
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._update_elided_text()

    @property
    def full_text(self) -> str:
        return self._full_text

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._update_elided_text()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def setFont(self, font: QtGui.QFont) -> None:
        super().setFont(font)
        self._update_elided_text()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _update_elided_text(self) -> None:
        width = max(self.contentsRect().width(), 0)
        if width <= 0:
            super().setText("")
            return
        super().setText(
            self.fontMetrics().elidedText(self._full_text, self._elide_mode, width)
        )


class _LoadSourceArgumentsEdit(QtWidgets.QPlainTextEdit):
    _MAX_VISIBLE_ROWS = 4
    _VERTICAL_PADDING = 12

    def setPlainText(self, text: str | None) -> None:
        super().setPlainText("" if text is None else text)
        self._update_fixed_height()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._update_fixed_height()

    def _visual_row_count(self) -> int:
        document = typing.cast("QtGui.QTextDocument", self.document())
        row_count = 0
        block = document.firstBlock()
        while block.isValid():
            layout = typing.cast("QtGui.QTextLayout", block.layout())
            row_count += max(1, layout.lineCount())
            block = block.next()
        return row_count

    def _update_fixed_height(self) -> None:
        row_count = max(1, min(self._MAX_VISIBLE_ROWS, self._visual_row_count()))
        self.setFixedHeight(
            row_count * self.fontMetrics().lineSpacing() + self._VERTICAL_PADDING
        )


class _LoadSourceDetailsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        details: _LoadSourceDetails,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data Source")
        self.setModal(True)
        self.setMinimumWidth(500)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        mono_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )

        title_label = QtWidgets.QLabel(details.path.name, self)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(8)
        form_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        layout.addLayout(form_layout)

        self.path_edit = QtWidgets.QLineEdit(str(details.path), self)
        self.path_edit.setReadOnly(True)
        self.path_edit.setFont(mono_font)
        form_layout.addRow("File", self.path_edit)

        self.loader_edit = QtWidgets.QLineEdit(details.loader_text, self)
        self.loader_edit.setReadOnly(True)
        self.loader_edit.setFont(mono_font)
        form_layout.addRow(details.loader_label, self.loader_edit)

        self.kwargs_edit = _LoadSourceArgumentsEdit(self)
        self.kwargs_edit.setReadOnly(True)
        self.kwargs_edit.setFont(mono_font)
        self.kwargs_edit.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.kwargs_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.kwargs_edit.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        self.kwargs_edit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.kwargs_highlighter = erlab.interactive.utils.PythonHighlighter(
            self.kwargs_edit.document()
        )
        self.kwargs_edit.setPlainText(details.kwargs_text)
        form_layout.addRow("Load Arguments", self.kwargs_edit)

        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setCenterButtons(False)
        self.copy_path_button = self.button_box.addButton(
            "Copy Path", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        assert self.copy_path_button is not None
        self.copy_path_button.clicked.connect(
            lambda: erlab.interactive.utils.copy_to_clipboard(self.path_edit.text())
        )
        self.copy_code_button = self.button_box.addButton(
            "Copy Load Code", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        assert self.copy_code_button is not None
        self.copy_code_button.setEnabled(details.load_code is not None)
        self.copy_code_button.setToolTip(
            "Copy a replayable loading snippet for this source."
            if details.load_code is not None
            else "Load code is unavailable for this source."
        )
        self.copy_code_button.clicked.connect(
            lambda: (
                erlab.interactive.utils.copy_to_clipboard(details.load_code)
                if details.load_code is not None
                else None
            )
        )
        close_button = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        assert close_button is not None
        close_button.clicked.connect(self.accept)
        layout.addWidget(self.button_box)
        self.adjustSize()


def _check_message_is_progressbar(message: str) -> bool:
    """Check if a log message contains a progress bar.

    example: "Title:   8%|8         | 1/12 [00:00<00:07,  1.54it/s]"
    """
    return "|" in message and "%" in message and message.endswith("]")


class _WarningNotificationHandler(logging.Handler):
    def __init__(self, emitter: _WarningEmitter):
        super().__init__(level=logging.WARNING)
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "suppress_ui_alert", False):
            # A dedicated dialog already surfaced this handled condition.
            return

        traceback_header = "Traceback (most recent call last):"
        traceback_msg = ""
        try:
            message = record.getMessage()

            if message.strip() == traceback_header:
                # Ignore messages that are only the traceback header
                return

            if traceback_header in message:
                message, traceback_msg = message.split(traceback_header, 1)
                traceback_msg = traceback_header + traceback_msg
            elif (
                not _check_message_is_progressbar(message)
                and record.levelno == logging.ERROR
                and not record.exc_info
            ):
                # This should have been handled by stderr logger already, so just ignore
                return

            if record.exc_info:
                traceback_msg = "".join(traceback.format_exception(*record.exc_info))

            if traceback_msg:
                traceback_msg = erlab.interactive.utils._format_traceback(traceback_msg)

            # emit_user_level_warning adds "<sys>:0: " when triggerd from GUI actions
            message = message.strip().replace("<sys>:0: ", "")

        except Exception:  # pragma: no cover
            self.handleError(record)
            return
        self._emitter.warning_received.emit(
            record.levelname, record.levelno, message, traceback_msg
        )


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

_WATCHED_VAR_COLORS: tuple[QtGui.QColor, ...] = (
    QtGui.QColor(72, 120, 208),
    QtGui.QColor(238, 133, 74),
    QtGui.QColor(106, 204, 100),
    QtGui.QColor(214, 95, 95),
    QtGui.QColor(149, 108, 180),
)
"""Colors for watched variables from different kernels."""

_manager_instance: ImageToolManager | None = None
"""Reference to the running manager instance."""

_always_use_socket: bool = False
"""Internal flag to use sockets within same process for test coverage."""


class ItoolManagerParseError(Exception):
    """Raised when the data received from the client cannot be parsed."""


@dataclass(frozen=True)
class _StandaloneAppSpec:
    key: str
    menu: typing.Literal["file", "apps"]
    text: str
    tooltip: str
    factory: Callable[[], QtWidgets.QWidget]
    shortcut: str | None = None
    icon_name: str | None = None


class _SingleImagePreview(QtWidgets.QGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        _scene = QtWidgets.QGraphicsScene(self)
        self.setScene(_scene)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._pixmapitem = typing.cast(
            "QtWidgets.QGraphicsPixmapItem", _scene.addPixmap(QtGui.QPixmap())
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

        # Initialize warning notifications
        self._warning_emitter = _WarningEmitter(self)
        self._warning_emitter.warning_received.connect(self._show_alert)
        self._warning_handler = _WarningNotificationHandler(self._warning_emitter)
        logging.getLogger().addHandler(self._warning_handler)
        self._alert_dialogs: list[erlab.interactive.utils.MessageDialog] = []
        self._ignored_warning_messages: set[str] = set()

        # Setup uncaught exception handler
        self._previous_excepthook = sys.excepthook
        sys.excepthook = self._handle_uncaught_exception

        # Setup servers and connect signals
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

        self.menu_bar: QtWidgets.QMenuBar = typing.cast(
            "QtWidgets.QMenuBar", self.menuBar()
        )

        self._imagetool_wrappers: dict[int, _ImageToolWrapper] = {}
        self._all_nodes: dict[str, _ImageToolWrapper | _ManagedWindowNode] = {}
        self._pending_source_refresh_targets: dict[str, set[str]] = {}
        self._displayed_indices: list[int] = []
        self._node_uid_counter: int = 0
        self._linkers: list[erlab.interactive.imagetool.viewer.SlicerLinkProxy] = []

        # Stores additional analysis tools opened from child ImageTool windows
        self._additional_windows: dict[str, QtWidgets.QWidget] = {}
        self._standalone_app_windows: dict[str, QtWidgets.QWidget] = {}
        self._standalone_app_specs: dict[str, _StandaloneAppSpec] = {
            "explorer": _StandaloneAppSpec(
                key="explorer",
                menu="file",
                text="Data Explorer",
                tooltip="Show the data explorer window",
                shortcut="Ctrl+E",
                icon_name="drive-harddisk",
                factory=self._create_explorer_window,
            ),
            "ptable": _StandaloneAppSpec(
                key="ptable",
                menu="apps",
                text="Periodic Table",
                tooltip="Show the periodic table window",
                shortcut="Ctrl+Shift+P",
                factory=self._create_ptable_window,
            ),
        }
        self._standalone_app_actions: dict[str, QtWidgets.QAction] = {}

        # Store progress bar widgets
        self._progress_bars: dict[int, QtWidgets.QProgressDialog] = {}

        # Deferred updates while removing multiple windows
        self._bulk_remove_depth: int = 0
        self._pending_linker_reload: bool = False

        # Initialize actions
        self.settings_action = QtWidgets.QAction("Settings", self)
        self.settings_action.triggered.connect(self.open_settings)
        self.settings_action.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
        self.settings_action.setToolTip("Open settings")

        self.show_action = QtWidgets.QAction("Show", self)
        self.show_action.triggered.connect(self.show_selected)
        self.show_action.setShortcut(
            "Return"
            if sys.platform == "darwin"
            else QtGui.QKeySequence.StandardKey.InsertParagraphSeparator
        )
        self.show_action.setToolTip("Show selected windows")

        self.hide_action = QtWidgets.QAction("Hide", self)
        self.hide_action.triggered.connect(self.hide_selected)
        self.hide_action.setShortcut("Ctrl+W")
        self.hide_action.setToolTip("Hide selected windows")

        self.gc_action = QtWidgets.QAction("Run Garbage Collection", self)
        self.gc_action.triggered.connect(self.garbage_collect)
        self.gc_action.setToolTip("Run garbage collection to free up memory")
        self.gc_action.setIcon(QtGui.QIcon.fromTheme("user-trash"))

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

        self.promote_action = QtWidgets.QAction("Promote Window", self)
        self.promote_action.triggered.connect(self.promote_selected)
        self.promote_action.setToolTip(
            "Promote the selected nested ImageTool to a top-level window"
        )

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
        self.console_action.setShortcut(QtGui.QKeySequence("Ctrl+J"))
        self.console_action.setToolTip("Toggle console window")
        self.console_action.setIcon(QtGui.QIcon.fromTheme("utilities-terminal"))

        self.preview_action = QtWidgets.QAction("Preview on Hover", self)
        self.preview_action.setCheckable(True)
        self.preview_action.setToolTip("Show preview on hover")

        self.store_action = QtWidgets.QAction("Store with IPython", self)
        self.store_action.triggered.connect(self.store_selected)
        self.store_action.setToolTip("Store selected data with IPython")

        self.explorer_action = self._create_standalone_app_action("explorer")
        self.ptable_action = self._create_standalone_app_action("ptable")

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

        self.source_update_action = QtWidgets.QAction("Automatic Updates...", self)
        self.source_update_action.triggered.connect(self.show_selected_source_updates)
        self.source_update_action.setToolTip(
            "Turn automatic updates on or off for the selected child window"
        )
        self.source_update_action.setVisible(False)

        self.about_action = QtWidgets.QAction("About", self)
        self.about_action.setIcon(QtGui.QIcon.fromTheme("help-about"))
        self.about_action.triggered.connect(self.about)

        self.check_update_action = QtWidgets.QAction("Check for Updates", self)
        self.check_update_action.setMenuRole(
            QtWidgets.QAction.MenuRole.ApplicationSpecificRole
        )
        self.check_update_action.triggered.connect(self.check_for_updates)
        self.check_update_action.setVisible(erlab.utils.misc._IS_PACKAGED)

        release_notes_action, open_docs_action, report_issue_action = (
            erlab.interactive.utils.make_help_actions(self)
        )

        self.open_log_folder_action = QtWidgets.QAction("Open Log Directory", self)
        self.open_log_folder_action.triggered.connect(self.open_log_directory)

        # Populate menu bar
        file_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&File")
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
        file_menu.addAction(self.show_action)
        file_menu.addAction(self.hide_action)
        file_menu.addSeparator()
        file_menu.addAction(self.remove_action)
        file_menu.addAction(self.archive_action)
        file_menu.addAction(self.unarchive_action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)

        edit_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Edit")
        )
        edit_menu.addAction(self.reindex_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.concat_action)
        edit_menu.addAction(self.duplicate_action)
        edit_menu.addAction(self.promote_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.rename_action)
        edit_menu.addAction(self.link_action)
        edit_menu.addAction(self.unlink_action)

        view_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&View")
        )
        view_menu.addAction(self.console_action)
        view_menu.addSeparator()
        view_menu.addAction(self.preview_action)
        view_menu.addSeparator()

        apps_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Apps")
        )
        apps_menu.addAction(self.ptable_action)

        self._dask_menu = DaskMenu(self, "Dask")
        self.menu_bar.addMenu(self._dask_menu)

        help_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Help")
        )
        help_menu.addAction(self.about_action)
        help_menu.addAction(self.check_update_action)
        help_menu.addAction(release_notes_action)
        help_menu.addSeparator()
        help_menu.addAction(open_docs_action)
        help_menu.addAction(report_issue_action)
        help_menu.addSeparator()
        help_menu.addAction(self.open_log_folder_action)

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
            self.preview_action, on="ph.eye", off="ph.eye-slash"
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

        self.metadata_group = QtWidgets.QFrame(self)
        self.metadata_group.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.metadata_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        metadata_layout = QtWidgets.QVBoxLayout(self.metadata_group)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setSpacing(4)
        self.metadata_group.setLayout(metadata_layout)

        self.metadata_details_widget = QtWidgets.QWidget(self.metadata_group)
        self.metadata_details_layout = QtWidgets.QGridLayout(
            self.metadata_details_widget
        )
        self.metadata_details_layout.setContentsMargins(0, 0, 0, 0)
        self.metadata_details_layout.setHorizontalSpacing(8)
        self.metadata_details_layout.setVerticalSpacing(2)
        self.metadata_details_layout.setColumnStretch(1, 1)
        self.metadata_details_widget.setLayout(self.metadata_details_layout)
        self.metadata_details_widget.setVisible(False)
        metadata_layout.addWidget(self.metadata_details_widget)
        self._metadata_detail_labels: dict[str, QtWidgets.QLabel] = {}
        self._metadata_monospace_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )

        self.metadata_derivation_list = _MetadataDerivationListWidget(
            self.metadata_group
        )
        self.metadata_derivation_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.metadata_derivation_list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.metadata_derivation_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.metadata_derivation_list.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.metadata_derivation_list.setTextElideMode(
            QtCore.Qt.TextElideMode.ElideRight
        )
        self.metadata_derivation_list.setUniformItemSizes(True)
        self.metadata_derivation_list.setAlternatingRowColors(False)
        self.metadata_derivation_list.copy_requested.connect(
            self._copy_selected_derivation_code
        )
        self.metadata_derivation_list.context_menu_requested.connect(
            self._show_metadata_derivation_menu
        )
        self.metadata_derivation_list.setVisible(False)
        metadata_layout.addWidget(self.metadata_derivation_list)
        right_splitter.addWidget(self.metadata_group)

        # Set initial splitter sizes
        right_splitter.setSizes([280, 140, 96])
        main_splitter.setSizes([100, 150])

        # Temporary directory for storing archived data
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="erlab_archive_")

        # Store most recent name filter and directory for new windows
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None
        self._recent_loader_extensions_by_filter: dict[str, dict[str, typing.Any]] = {}
        self._metadata_full_code: str | None = None
        self._metadata_copy_selected_action = QtGui.QAction("Copy Selected Code", self)
        self._metadata_copy_selected_action.triggered.connect(
            self._copy_selected_derivation_code
        )
        self._metadata_copy_full_action = QtGui.QAction("Copy Full Code", self)
        self._metadata_copy_full_action.triggered.connect(
            self._copy_full_derivation_code
        )

        self.sigLinkersChanged.connect(self._update_actions)
        self.sigLinkersChanged.connect(self.tree_view.refresh)
        self._sigReloadLinkers.connect(self._request_reload_linkers)
        self._update_actions()
        self._update_info()

        # Golden ratio :)
        self.setMinimumWidth(301)
        self.setMinimumHeight(487)
        self.resize(487, 487)

        # Install event filter for keyboard shortcuts
        self._kb_filter = erlab.interactive.utils.KeyboardEventFilter(self)
        for widget in (
            self.text_box,
            self.metadata_derivation_list,
        ):
            widget.installEventFilter(self._kb_filter)

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

    def _next_node_uid(self, preferred: str | None = None) -> str:
        if preferred is not None and preferred not in self._all_nodes:
            self._consume_node_uid(preferred)
            return preferred
        while True:
            uid = f"n{self._node_uid_counter}"
            self._node_uid_counter += 1
            if uid not in self._all_nodes:
                return uid

    def _consume_node_uid(self, uid: str) -> None:
        if uid.startswith("n") and uid[1:].isdigit():
            self._node_uid_counter = max(self._node_uid_counter, int(uid[1:]) + 1)

    def _register_root_wrapper(self, wrapper: _ImageToolWrapper) -> None:
        self._all_nodes[wrapper.uid] = wrapper

    def _register_child_node(self, node: _ManagedWindowNode) -> None:
        self._all_nodes[node.uid] = node
        parent = self._parent_node(node)
        parent.add_child_reference(
            node.uid, typing.cast("QtWidgets.QWidget", node.window)
        )

    def _unregister_node(self, uid: str) -> None:
        node = self._all_nodes.pop(uid, None)
        if node is None:
            return
        self._pending_source_refresh_targets.pop(uid, None)
        for blocker_uid, target_uids in list(
            self._pending_source_refresh_targets.items()
        ):
            target_uids.discard(uid)
            if not target_uids:
                self._pending_source_refresh_targets.pop(blocker_uid, None)
        if node.parent_uid is not None:
            parent = typing.cast(
                "_ImageToolWrapper | _ManagedWindowNode",
                self._all_nodes.get(node.parent_uid),
            )
            if parent is not None:
                parent.remove_child_reference(uid)

    def _node_for_target(
        self, target: int | str
    ) -> _ImageToolWrapper | _ManagedWindowNode:
        if isinstance(target, int):
            return self._imagetool_wrappers[target]
        return self._all_nodes[target]

    def _child_node(self, uid: str) -> _ManagedWindowNode:
        node = self._all_nodes[uid]
        if isinstance(node, _ImageToolWrapper):
            raise KeyError(f"{uid!r} refers to a root ImageTool")
        return node

    def _parent_node(
        self, node: _ManagedWindowNode
    ) -> _ImageToolWrapper | _ManagedWindowNode:
        if node.parent_uid is None:
            raise KeyError(f"Node {node.uid!r} has no parent")
        return self._all_nodes[node.parent_uid]

    def _root_wrapper_for_uid(self, uid: str) -> _ImageToolWrapper:
        node = self._node_for_target(uid)
        while not isinstance(node, _ImageToolWrapper):
            node = self._parent_node(node)
        return node

    def _node_uid_from_window(self, widget: QtWidgets.QWidget) -> str | None:
        for uid, node in self._all_nodes.items():
            if node.window is widget:
                return uid
        return None

    def _is_imagetool_target(self, target: int | str) -> bool:
        node = self._node_for_target(target)
        return node.is_imagetool

    def _target_is_archived(self, target: int | str) -> bool:
        node = self._node_for_target(target)
        return node.archived

    def _selected_imagetool_targets(self) -> list[int | str]:
        targets: list[int | str] = list(self.tree_view.selected_imagetool_indices)
        targets.extend(
            uid
            for uid in self.tree_view.selected_childtool_uids
            if self._is_imagetool_target(uid)
        )
        return targets

    def _selected_tool_uids(self) -> list[str]:
        return [
            uid
            for uid in self.tree_view.selected_childtool_uids
            if not self._is_imagetool_target(uid)
        ]

    def _selected_promotable_child_imagetool_uid(self) -> str | None:
        child_imagetool_uids = [
            uid
            for uid in self.tree_view.selected_childtool_uids
            if self._is_imagetool_target(uid)
        ]
        if len(child_imagetool_uids) != 1:
            return None
        if self.tree_view.selected_imagetool_indices or self._selected_tool_uids():
            return None
        uid = child_imagetool_uids[0]
        if self._target_is_archived(uid):
            return None
        return uid

    def _selected_source_update_child_uid(self) -> str | None:
        selected_child_uids = self.tree_view.selected_childtool_uids
        if len(selected_child_uids) != 1 or self.tree_view.selected_imagetool_indices:
            return None
        uid = selected_child_uids[0]
        try:
            node = self._child_node(uid)
        except KeyError:
            return None
        return uid if node.has_source_binding else None

    @QtCore.Slot()
    def show_selected_source_updates(self) -> None:
        """Show automatic update controls for the selected child window."""
        uid = self._selected_source_update_child_uid()
        if uid is None:
            return
        self._child_node(uid).show_source_update_dialog(parent=self)

    def _child_targets_of(self, target: int | str) -> list[str]:
        return list(self._node_for_target(target)._childtool_indices)

    def _iter_descendant_uids(self, uid: str) -> list[str]:
        descendants: list[str] = []
        stack = [uid]
        while stack:
            current = stack.pop()
            node = self._all_nodes.get(current)
            if not isinstance(node, _ManagedWindowNode):
                continue
            for child_uid in node._childtool_indices:
                descendants.append(child_uid)
                stack.append(child_uid)
        return descendants

    def _refresh_source_chain_to_uid(self, uid: str) -> bool:
        """Refresh stale ancestors before refreshing a managed child node."""
        try:
            node = self._child_node(uid)
        except KeyError:
            return False

        refresh_chain = [node]
        while True:
            try:
                parent = self._parent_node(node)
            except KeyError:
                return False
            if isinstance(parent, _ImageToolWrapper):
                break
            refresh_chain.append(parent)
            node = parent

        for node in reversed(refresh_chain):
            current_uid = node.uid

            if not node.has_source_binding or node.source_state == "fresh":
                continue
            if node.archived:
                self._mark_descendants_source_state(current_uid, node.source_state)
                return False

            updated = node._update_from_parent_source()
            if updated and node.source_state == "fresh":
                continue
            tool = node.tool_window
            if (
                tool is not None
                and tool.source_state == "stale"
                and getattr(tool, "_source_refresh_deferred", False)
            ):
                self._pending_source_refresh_targets.setdefault(current_uid, set()).add(
                    uid
                )
                return False
            if node.source_state != "fresh":
                self._mark_descendants_source_state(current_uid, node.source_state)
            return False

        try:
            return self._child_node(uid).source_state == "fresh"
        except KeyError:
            return False

    def _resume_pending_source_refreshes(self, uid: str) -> None:
        target_uids = self._pending_source_refresh_targets.pop(uid, set())
        for target_uid in list(target_uids):
            if target_uid not in self._all_nodes:
                continue
            self._refresh_source_chain_to_uid(target_uid)

    def _parent_source_data_for_uid(self, uid: str) -> xr.DataArray:
        node = self._child_node(uid)
        parent = self._parent_node(node)
        return parent.current_source_data()

    def _mark_descendants_source_state(
        self,
        uid: str,
        state: _ManagedWindowNode._source_state_type,
    ) -> None:
        for child_uid in self._iter_descendant_uids(uid):
            node = self._child_node(child_uid)
            if node.tool_window is not None and node.tool_window.has_source_binding:
                node.tool_window._set_source_state(state)
            elif node.has_source_binding:
                node._set_source_state(state)

    def _mark_descendants_source_unavailable(self, uid: str) -> None:
        self._mark_descendants_source_state(uid, "unavailable")

    def _propagate_source_change_from_uid(
        self, uid: str, parent_data: xr.DataArray | None = None
    ) -> None:
        if parent_data is None:
            try:
                parent_data = self._node_for_target(uid).current_source_data()
            except Exception:
                self._mark_descendants_source_unavailable(uid)
                return
        for child_uid in list(self._node_for_target(uid)._childtool_indices):
            try:
                child = self._child_node(child_uid)
            except KeyError:
                continue
            updated = child.handle_parent_source_replaced(parent_data)
            self.tree_view.refresh(child_uid)
            if child.archived:
                continue
            if updated:
                self._propagate_source_change_from_uid(child_uid)
            elif child.source_state != "fresh":
                self._mark_descendants_source_state(child_uid, child.source_state)

    def _remove_uid_target(self, uid: str) -> None:
        if uid not in self._all_nodes:
            return
        subtree = [*self._iter_descendant_uids(uid), uid]
        subtree.reverse()
        for child_uid in subtree:
            child = typing.cast("_ManagedWindowNode", self._all_nodes.get(child_uid))
            if child is None:
                continue
            self._unregister_node(child_uid)
            if child.tool_window is not None:
                child.tool_window.set_source_parent_fetcher(None)
                child.tool_window.set_input_provenance_parent_fetcher(None)
            child.dispose()
        self.tree_view.childtool_removed(uid)

    @property
    def _status_bar(self) -> QtWidgets.QStatusBar:
        return typing.cast("QtWidgets.QStatusBar", self.statusBar())

    def _make_icon_msgbox(self) -> QtWidgets.QMessageBox:
        """Create a QMessageBox with the application icon."""
        msg_box = QtWidgets.QMessageBox(self)
        style = self.style()
        if style is not None:  # pragma: no branch
            icon_size = (
                style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_MessageBoxIconSize)
                or 48
            )
            msg_box.setIconPixmap(self.windowIcon().pixmap(icon_size, icon_size))
        return msg_box

    @QtCore.Slot()
    def about(self) -> None:
        """Show the about dialog."""
        import h5netcdf
        import xarray_lmfit

        msg_box = self._make_icon_msgbox()

        version_info = {
            "numpy": np.__version__,
            "xarray": xr.__version__,
            "h5netcdf": h5netcdf.__version__,
            "xarray-lmfit": xarray_lmfit.__version__,
            "pyqtgraph": pyqtgraph.__version__,
            "Qt": f"{qtpy.API_NAME} {qtpy.QT_VERSION}",
            "Python": platform.python_version(),
            "OS": platform.platform(),
        }
        if erlab.utils.misc._IS_PACKAGED:  # pragma: no cover
            version_info["Location"] = os.path.dirname(sys.executable).removesuffix(
                "/Contents/MacOS"
            )
        version_info_str = "\n".join(f"{k}: {v}" for k, v in version_info.items())
        msg_box.setText(f"ImageTool Manager {erlab.__version__}")
        msg_box.setInformativeText(version_info_str)
        msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        copy_btn = msg_box.addButton(
            "Copy", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        msg_box.exec()

        if msg_box.clickedButton() == copy_btn:
            erlab.interactive.utils.copy_to_clipboard(
                f"erlab: {erlab.__version__}\n" + version_info_str
            )

    def updated(self, old_version: str, new_version: str) -> None:
        """Notify the user that the application has been updated."""
        msg_box = self._make_icon_msgbox()
        if old_version == "":
            # First time installation
            msg_box.setText("ImageTool Manager Installed")
            msg_box.setInformativeText(
                f"Welcome to ImageTool Manager! You are using version {new_version}.",
            )
        else:
            msg_box.setText("ImageTool Manager Updated")
            msg_box.setInformativeText(
                "ImageTool Manager has been successfully updated from version "
                f"{old_version} to {new_version}.",
            )
        msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        release_notes_btn = msg_box.addButton(
            "Open Release Notes", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        documentation_btn = msg_box.addButton(
            "Open Documentation", QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        msg_box.exec()

        import webbrowser

        clicked_button = msg_box.clickedButton()
        if clicked_button == release_notes_btn:
            webbrowser.open("https://github.com/kmnhan/erlabpy/releases")
        elif clicked_button == documentation_btn:
            webbrowser.open(
                "https://erlabpy.readthedocs.io/en/stable/user-guide/interactive/imagetool.html"
            )

    @QtCore.Slot()
    def open_log_directory(self) -> None:
        """Open the log directory in the system file explorer."""
        erlab.utils.misc.open_in_file_manager(get_log_file_path().parent)

    def _parse_progressbar(self, message: str):
        """Parse and display a progress bar from a log message."""
        title_part, _, info_part = message.split("|", 2)
        title: str = title_part.split(":", 1)[0].strip()
        current: int = int(info_part.split("/", 1)[0].strip())
        total: int = int(info_part.split("/", 1)[1].split("[", 1)[0].strip())

        if total in self._progress_bars:
            pbar = self._progress_bars[total]
        else:
            pbar = QtWidgets.QProgressDialog(self)
            pbar.setLabelText(title)
            pbar.setMinimum(0)
            pbar.setMaximum(total)
            pbar.setAutoReset(True)
            pbar.setAutoClose(True)
            pbar.setWindowModality(QtCore.Qt.WindowModality.NonModal)
            pbar.setWindowFlags(
                pbar.windowFlags()
                | QtCore.Qt.WindowType.Tool
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
            )
            pbar.findChild(QtWidgets.QProgressBar).setFormat("%p% (%v/%m)")
            self._progress_bars[total] = pbar
            pbar.show()
            pbar.raise_()
            pbar.activateWindow()

        pbar.setValue(current)

    @QtCore.Slot(str, int, str, str)
    def _show_alert(
        self, levelname: str, levelno: int, message: str, formatted_traceback: str
    ) -> None:
        """Show a non-intrusive warning message in a floating window."""
        if _check_message_is_progressbar(message):
            try:
                self._parse_progressbar(message)
            except Exception:  # pragma: no cover
                logger.exception("Failed to parse progress bar message %r", message)
            else:
                return

        if message in self._ignored_warning_messages:
            return

        if levelno >= logging.ERROR:
            icon_pixmap = QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical
        elif levelno >= logging.WARNING:
            icon_pixmap = QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
        else:
            icon_pixmap = QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation

        dialog = erlab.interactive.utils.MessageDialog(
            self,
            title=levelname,
            text=message,
            detailed_text=formatted_traceback,
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=icon_pixmap,
        )
        dialog.setModal(False)
        dialog.setWindowFlags(
            dialog.windowFlags()
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )

        ignore_btn = QtWidgets.QPushButton("Ignore", dialog)
        ignore_btn.setObjectName("warningIgnoreButton")
        ignore_btn.clicked.connect(
            lambda *, msg=message: self._ignore_warning_message(msg)
        )
        ignore_btn.setDefault(False)
        ignore_btn.setAutoDefault(False)
        dialog._button_box.addButton(
            ignore_btn, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )

        dismiss_btn = QtWidgets.QPushButton("Dismiss All", dialog)
        dismiss_btn.setObjectName("warningDismissAllButton")
        dismiss_btn.clicked.connect(self._clear_all_alerts)
        dismiss_btn.setDefault(False)
        dismiss_btn.setAutoDefault(False)
        dialog._button_box.addButton(
            dismiss_btn, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self._alert_dialogs.append(dialog)
        dialog.finished.connect(lambda *, d=dialog: self._unregister_alert(d))

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ignore_warning_message(self, message: str) -> None:
        """Ignore future warnings with the same message for this session."""
        self._ignored_warning_messages.add(message)
        for notification in list(self._alert_dialogs):
            if notification.text() == message:
                notification.close()

    def _unregister_alert(self, alert: erlab.interactive.utils.MessageDialog) -> None:
        with contextlib.suppress(ValueError):  # pragma: no cover - defensive cleanup
            self._alert_dialogs.remove(alert)

    @QtCore.Slot()
    def _clear_all_alerts(self) -> None:
        for notification in list(self._alert_dialogs):
            notification.close()

    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Show a dialog for uncaught exceptions and log them."""
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.error(
                "An unexpected error occurred.",
                exc_info=(exc_type, exc_value, exc_traceback),
            )
        if self._previous_excepthook is not None:  # pragma: no branch
            with contextlib.suppress(Exception):
                self._previous_excepthook(exc_type, exc_value, exc_traceback)

    @property
    def _reindex_lock(self) -> threading.Lock:
        """Lock for reindexing operation."""
        if not hasattr(self, "__reindex_lock"):
            lock = threading.Lock()
            self.__reindex_lock = lock
        return self.__reindex_lock

    @QtCore.Slot()
    def reindex(self) -> None:
        """Reset indices of ImageTool windows to be consecutive in displayed order."""
        with self._reindex_lock:
            new_imagetool_wrappers: dict[int, _ImageToolWrapper] = {}
            displayed_indices = list(self._displayed_indices)
            for row_idx, tool_idx in enumerate(displayed_indices):
                self._displayed_indices[row_idx] = row_idx
                self._imagetool_wrappers[tool_idx]._index = row_idx
                new_imagetool_wrappers[row_idx] = self._imagetool_wrappers[tool_idx]
            self._imagetool_wrappers = new_imagetool_wrappers

        self.tree_view.refresh()

    def get_imagetool(self, index: int | str, unarchive: bool = True) -> ImageTool:
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
        node = self._node_for_target(index)
        if not node.is_imagetool:
            raise KeyError(f"Target {index!r} is not an ImageTool")

        wrapper = node
        if wrapper.archived:
            if unarchive:
                wrapper.unarchive()
            else:
                raise KeyError(f"Tool of target '{index}' is archived")
        tool = wrapper.imagetool
        if tool is None or not erlab.interactive.utils.qt_is_valid(tool):
            raise KeyError(f"Tool of target '{index}' is not available")
        return tool

    def color_for_linker(
        self, linker: erlab.interactive.imagetool.viewer.SlicerLinkProxy
    ) -> QtGui.QColor:
        """Get the color that should represent the given linker."""
        idx = self._linkers.index(linker)
        return _LINKER_COLORS[idx % len(_LINKER_COLORS)]

    def add_imagetool(
        self,
        tool: ImageTool,
        *,
        show: bool = True,
        activate: bool = False,
        watched_var: tuple[str, str] | None = None,
        source_input_ndim: int | None = None,
        source_input_dtype: np.dtype[typing.Any] | str | None = None,
        uid: str | None = None,
        provenance_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
    ) -> int:
        """Add a new ImageTool window to the manager and show it.

        Parameters
        ----------
        tool
            ImageTool object to be added.
        show
            Whether to show the window after adding, by default `True`.
        activate
            Whether to focus on the window after adding, by default `False`.
        watched_var
            If the tool is created from a watched variable, this should be a tuple of
            the variable name and its unique ID.
        source_input_ndim
            Original dimensionality of the bound source before ImageTool-specific
            promotion (for example, promoted 1D inputs).

        Returns
        -------
        int
            Index of the added ImageTool window.
        """
        if provenance_spec is not None:
            tool.set_provenance_spec(provenance_spec)
        index = int(self.next_idx)
        wrapper = _ImageToolWrapper(
            self,
            index,
            self._next_node_uid(uid),
            tool,
            watched_var=watched_var,
            source_input_ndim=source_input_ndim,
            source_input_dtype=source_input_dtype,
            provenance_spec=provenance_spec,
            source_spec=source_spec,
            source_auto_update=source_auto_update,
            source_state=source_state,
        )
        self._imagetool_wrappers[index] = wrapper
        self._register_root_wrapper(wrapper)
        wrapper.update_title()

        self._sigReloadLinkers.emit()

        if show:
            tool.show()

        if activate:
            tool.activateWindow()
            tool.raise_()

        # Add to view after initialization
        self.tree_view.imagetool_added(index)

        return index

    @QtCore.Slot()
    def _node_info_html(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        return node.info_text

    def _clear_metadata(self) -> None:
        self._metadata_full_code = None
        with QtCore.QSignalBlocker(self.metadata_derivation_list):
            self.metadata_derivation_list.clear()
        self._set_metadata_fields([])
        self._update_metadata_pane()

    def _set_metadata_node(self, node: _ImageToolWrapper | _ManagedWindowNode) -> None:
        self._metadata_full_code = node.derivation_code
        self._set_metadata_fields(node.metadata_fields)

        with QtCore.QSignalBlocker(self.metadata_derivation_list):
            self.metadata_derivation_list.clear()
            for entry in node.derivation_entries:
                item = QtWidgets.QListWidgetItem(entry.label)
                item.setToolTip(entry.label)
                item.setData(_METADATA_DERIVATION_CODE_ROLE, entry.code)
                item.setData(_METADATA_DERIVATION_COPYABLE_ROLE, entry.copyable)
                if not entry.copyable:
                    item.setForeground(
                        self.palette().color(QtGui.QPalette.ColorRole.Mid)
                    )
                    if entry.code is None and not entry.label.startswith("Start from "):
                        item.setToolTip("Replay code is unavailable for this step.")
                self.metadata_derivation_list.addItem(item)
        self._update_metadata_pane()

    def _set_metadata_fields(self, fields: list[_MetadataField]) -> None:
        while self.metadata_details_layout.count():
            item = self.metadata_details_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._metadata_detail_labels.clear()

        for row, field in enumerate(fields):
            key_label = QtWidgets.QLabel(field.label, self.metadata_details_widget)
            key_label.setStyleSheet("color: palette(mid);")
            value_label: QtWidgets.QLabel
            if field.details is not None:
                value_label = _ElidedInteractiveLabel(
                    field.value,
                    self.metadata_details_widget,
                )
                value_label.setStyleSheet("color: palette(link);")
                value_label.set_full_text(field.value)
                value_label.clicked.connect(
                    lambda d=field.details: self._show_load_source_details(d)
                )
            else:
                value_label = QtWidgets.QLabel(
                    field.value, self.metadata_details_widget
                )
                value_label.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                )
                value_label.setWordWrap(field.wrap)
                value_label.setToolTip(field.value)
                value_label.setMinimumWidth(0)
            if field.monospace:
                value_label.setFont(self._metadata_monospace_font)
            self.metadata_details_layout.addWidget(key_label, row, 0)
            self.metadata_details_layout.addWidget(value_label, row, 1)
            self._metadata_detail_labels[field.label] = value_label

    def _show_load_source_details(self, details: _LoadSourceDetails) -> None:
        _LoadSourceDetailsDialog(details, self).exec()

    def _update_metadata_pane(self) -> None:
        has_details = bool(self._metadata_detail_labels)
        derivation_count = self.metadata_derivation_list.count()

        self.metadata_group.setVisible(has_details or derivation_count > 0)
        self.metadata_details_widget.setVisible(has_details)
        self.metadata_derivation_list.setVisible(derivation_count > 0)

        if derivation_count == 0:
            self.metadata_derivation_list.setMinimumHeight(0)
            self.metadata_derivation_list.setMaximumHeight(0)
            return

        row_height = self.metadata_derivation_list.sizeHintForRow(0)
        if row_height <= 0:
            row_height = self.fontMetrics().height() + 8
        visible_rows = min(derivation_count, 4)
        frame = self.metadata_derivation_list.frameWidth() * 2
        height = visible_rows * row_height + frame + 4
        self.metadata_derivation_list.setMinimumHeight(height)
        self.metadata_derivation_list.setMaximumHeight(height)

    def _selected_derivation_items(self) -> list[QtWidgets.QListWidgetItem]:
        items = list(self.metadata_derivation_list.selectedItems())
        if not items:
            current_item = self.metadata_derivation_list.currentItem()
            if current_item is not None:
                items = [current_item]
        return sorted(items, key=self.metadata_derivation_list.row)

    def _selected_derivation_code(self) -> str | None:
        codes: list[str] = []
        for item in self._selected_derivation_items():
            if not bool(item.data(_METADATA_DERIVATION_COPYABLE_ROLE)):
                continue
            code = typing.cast("str | None", item.data(_METADATA_DERIVATION_CODE_ROLE))
            if code:
                codes.append(code)
        if not codes:
            return None
        return "\n".join(codes)

    def _build_metadata_derivation_menu(self) -> QtWidgets.QMenu | None:
        if self.metadata_derivation_list.count() == 0:
            return None

        menu = QtWidgets.QMenu(self.metadata_derivation_list)
        selected_code = self._selected_derivation_code()
        self._metadata_copy_selected_action.setEnabled(bool(selected_code))
        menu.addAction(self._metadata_copy_selected_action)
        if self._metadata_full_code:
            self._metadata_copy_full_action.setEnabled(True)
            menu.addAction(self._metadata_copy_full_action)
        return menu

    @QtCore.Slot(QtCore.QPoint)
    def _show_metadata_derivation_menu(self, pos: QtCore.QPoint) -> None:
        if self.metadata_derivation_list.itemAt(pos) is None:
            return
        menu = self._build_metadata_derivation_menu()
        if menu is None:
            return
        viewport = self.metadata_derivation_list.viewport()
        if viewport is None:
            return
        menu.exec(viewport.mapToGlobal(pos))

    @QtCore.Slot()
    def _copy_selected_derivation_code(self) -> None:
        code = self._selected_derivation_code()
        if code:
            erlab.interactive.utils.copy_to_clipboard(code)

    @QtCore.Slot()
    def _copy_full_derivation_code(self) -> None:
        if self._metadata_full_code:
            erlab.interactive.utils.copy_to_clipboard(self._metadata_full_code)

    @QtCore.Slot()
    def _update_info(self, *, uid: str | None = None) -> None:
        """Update the information text box.

        If a string ``uid`` is provided, the function will update the info box only if
        the given ``uid`` is the only selected child tool.
        """
        selected_imagetools = self._selected_imagetool_targets()
        selected_childtools = self._selected_tool_uids()

        n_itool: int = len(selected_imagetools)
        n_total: int = n_itool + len(selected_childtools)

        selected_child_ids = list(selected_childtools)
        if uid is not None and n_itool == 1:
            target = selected_imagetools[0]
            if isinstance(target, str):
                selected_child_ids.append(target)

        if (uid is not None) and ((n_total != 1) or (uid not in selected_child_ids)):
            return

        match n_total:
            case 0:
                self.text_box.setPlainText("Select a window to view its information.")
                self._clear_metadata()
                self.preview_widget.setVisible(False)

            case 1:
                selected_target: int | str
                if n_itool > 0:
                    selected_target = selected_imagetools[0]
                else:
                    selected_target = selected_childtools[0]

                node = self._node_for_target(selected_target)
                self.text_box.setHtml(self._node_info_html(node))
                self._set_metadata_node(node)

                if node.is_imagetool:
                    self.preview_widget.setPixmap(node._preview_image[1])
                    self.preview_widget.setVisible(True)
                    return

                image_item = (
                    None
                    if node.tool_window is None
                    else node.tool_window.preview_imageitem
                )
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
                        self._node_for_target(i).display_text
                        for i in selected_imagetools
                    )
                )
                self._clear_metadata()
                self.preview_widget.setVisible(False)

    @QtCore.Slot()
    def _update_actions(self) -> None:
        """Update the state of the actions based on the current selection."""
        selection_children = self._selected_tool_uids()
        imagetool_targets = self._selected_imagetool_targets()
        promotable_child_uid = self._selected_promotable_child_imagetool_uid()
        source_update_child_uid = self._selected_source_update_child_uid()

        selection_archived: list[int | str] = []
        selection_unarchived: list[int | str] = []
        selection_watched: list[int] = []

        for target in imagetool_targets:
            node = self._node_for_target(target)
            if node.archived:
                selection_archived.append(target)
            else:
                selection_unarchived.append(target)
            if isinstance(node, _ImageToolWrapper) and node.watched:
                selection_watched.append(node.index)

        something_selected = bool(imagetool_targets or selection_children)
        root_imagetool_count = len(self.tree_view.selected_imagetool_indices)
        total_selected = len(imagetool_targets) + len(selection_children)
        single_selected = total_selected == 1
        multiple_root_imagetools_selected = (
            root_imagetool_count > 1 and root_imagetool_count == total_selected
        )
        multiple_selected = len(imagetool_targets) > 1
        only_unarchived = len(selection_archived) == 0
        only_archived = len(selection_unarchived) == 0 and len(imagetool_targets) > 0

        self.show_action.setEnabled(something_selected)
        self.hide_action.setEnabled(something_selected)
        self.remove_action.setEnabled(something_selected)
        self.rename_action.setEnabled(
            only_unarchived and (single_selected or multiple_root_imagetools_selected)
        )
        self.duplicate_action.setEnabled(something_selected)
        self.promote_action.setEnabled(promotable_child_uid is not None)
        self.archive_action.setEnabled(
            bool(imagetool_targets) and only_unarchived and len(selection_children) == 0
        )
        self.unarchive_action.setEnabled(
            bool(imagetool_targets) and only_archived and len(selection_children) == 0
        )
        self.concat_action.setEnabled(
            multiple_selected and len(selection_children) == 0
        )
        self.store_action.setEnabled(
            bool(self.tree_view.selected_imagetool_indices) and only_unarchived
        )

        self.reload_action.setVisible(
            bool(imagetool_targets)
            and only_unarchived
            and len(selection_children) == 0
            and all(
                self.get_imagetool(s).slicer_area.reloadable
                for s in selection_unarchived
            )
        )
        self.unwatch_action.setVisible(
            bool(imagetool_targets)
            and len(selection_watched) == len(imagetool_targets)
            and len(selection_children) == 0
            and all(
                isinstance(self._node_for_target(s), _ImageToolWrapper)
                for s in imagetool_targets
            )
        )
        self.source_update_action.setVisible(source_update_child_uid is not None)
        self.source_update_action.setEnabled(source_update_child_uid is not None)

        if not imagetool_targets or selection_children or not only_unarchived:
            self.link_action.setDisabled(True)
            self.unlink_action.setDisabled(True)
            return

        self.link_action.setDisabled(len(selection_unarchived) <= 1)
        is_linked = [
            self.get_imagetool(index).slicer_area.is_linked
            for index in selection_unarchived
        ]
        self.unlink_action.setEnabled(any(is_linked))

        if len(selection_unarchived) > 1 and all(is_linked):
            proxies = [
                self.get_imagetool(index).slicer_area._linking_proxy
                for index in selection_unarchived
            ]
            if all(p == proxies[0] for p in proxies):  # pragma: no branch
                self.link_action.setEnabled(False)

    @QtCore.Slot(int)
    def remove_imagetool(self, index: int, *, update_view: bool = True) -> None:
        """Remove the ImageTool window corresponding to the given index."""
        if index not in self._imagetool_wrappers:
            return
        wrapper = self._imagetool_wrappers[index]
        descendant_uids = list(wrapper._childtool_indices)
        if update_view:
            self.tree_view.imagetool_removed(index)

        for uid in list(descendant_uids):
            self._remove_uid_target(uid)

        self._imagetool_wrappers.pop(index)
        self._all_nodes.pop(wrapper.uid, None)
        wrapper.dispose()
        wrapper.deleteLater()

    @contextlib.contextmanager
    def _bulk_remove_context(self):
        outermost = self._bulk_remove_depth == 0
        self._bulk_remove_depth += 1
        if outermost:
            self._pending_linker_reload = False
            self.setUpdatesEnabled(False)
            self.tree_view.setUpdatesEnabled(False)
        try:
            yield
        finally:
            self._bulk_remove_depth -= 1
            if outermost:
                self.tree_view.setUpdatesEnabled(True)
                self.setUpdatesEnabled(True)

                if self._pending_linker_reload:
                    self._pending_linker_reload = False
                    self._cleanup_linkers()

                self._update_actions()
                self._update_info()

    def _remove_imagetools(
        self,
        indices: list[int | str],
        *,
        child_uids: list[str] | None = None,
        clear_view: bool = False,
    ) -> None:
        root_indices: list[int] = []
        child_targets: list[str] = []
        covered_child_uids: set[str] = set()
        for target in indices:
            if isinstance(target, int):
                root_indices.append(target)
                wrapper = self._imagetool_wrappers.get(target)
                if wrapper is not None:
                    direct_children = list(wrapper._childtool_indices)
                    covered_child_uids.update(direct_children)
                    iter_descendants = getattr(self, "_iter_descendant_uids", None)
                    if callable(iter_descendants):
                        for child_uid in direct_children:
                            covered_child_uids.update(iter_descendants(child_uid))
            else:
                child_targets.append(target)

        for uid in child_uids or []:
            if uid not in covered_child_uids and uid not in child_targets:
                child_targets.append(uid)

        if len(root_indices) == 0 and len(child_targets) == 0:
            return

        with self._bulk_remove_context():
            if clear_view:
                self.tree_view.clear_imagetools()

            for index in root_indices:
                self.remove_imagetool(index, update_view=not clear_view)

            for uid in child_targets:
                self._remove_childtool(uid)

    def remove_all_tools(self) -> None:
        """Remove all ImageTool windows."""
        self._remove_imagetools(list(self._imagetool_wrappers.keys()), clear_view=True)

    @QtCore.Slot(int)
    def show_imagetool(self, index: int) -> None:
        """Show the ImageTool window corresponding to the given index."""
        if index in self._imagetool_wrappers:  # pragma: no branch
            self._imagetool_wrappers[index].show()

    @QtCore.Slot()
    def _request_reload_linkers(self) -> None:
        if self._bulk_remove_depth > 0:
            self._pending_linker_reload = True
            return

        self._cleanup_linkers()

    @QtCore.Slot()
    def _cleanup_linkers(self) -> None:
        """Remove linkers with one or no children."""
        for linker in list(self._linkers):
            if linker.num_children <= 1:
                linker.unlink_all()
                self._linkers.remove(linker)
        self.sigLinkersChanged.emit()

    @property
    def explorer(self) -> _TabbedExplorer:
        widget = self._standalone_app_windows.get("explorer")
        if widget is None or not erlab.interactive.utils.qt_is_valid(widget):
            raise AttributeError("Data explorer is not initialized.")
        return typing.cast("_TabbedExplorer", widget)

    @property
    def ptable_window(self) -> PeriodicTableWindow:
        widget = self._standalone_app_windows.get("ptable")
        if widget is None or not erlab.interactive.utils.qt_is_valid(widget):
            raise AttributeError("Periodic table window is not initialized.")
        return typing.cast("PeriodicTableWindow", widget)

    def _create_explorer_window(self) -> QtWidgets.QWidget:
        from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer

        return _TabbedExplorer(
            root_path=self._recent_directory, loader_name=self._recent_loader_name
        )

    def _create_ptable_window(self) -> QtWidgets.QWidget:
        from erlab.interactive.ptable import PeriodicTableWindow

        return PeriodicTableWindow()

    def _preferred_name_filter(
        self, valid_loaders: dict[str, tuple[Callable, dict]]
    ) -> str | None:
        if self._recent_name_filter in valid_loaders:
            return self._recent_name_filter

        default_loader = erlab.interactive.options.model.io.default_loader
        if default_loader == "None" or default_loader not in erlab.io.loaders:
            return None

        return next(
            (
                name_filter
                for name_filter in erlab.io.loaders[default_loader].file_dialog_methods
                if name_filter in valid_loaders
            ),
            None,
        )

    def _select_loader_options(
        self,
        valid_loaders: dict[str, tuple[Callable, dict]],
        name_filter: str | None = None,
        *,
        sample_paths: Iterable[str | pathlib.Path] | None = None,
    ) -> tuple[str, Callable, dict[str, typing.Any]] | None:
        dialog = _NameFilterDialog(
            self,
            valid_loaders,
            loader_extensions=self._recent_loader_extensions_by_filter,
            sample_paths=sample_paths,
        )
        dialog.check_filter(name_filter or self._preferred_name_filter(valid_loaders))

        if not dialog.exec():
            return None

        selected_filter, func, kwargs = dialog.checked_filter()
        self._recent_name_filter = selected_filter
        loader_extensions = kwargs.get("loader_extensions", {})
        self._recent_loader_extensions_by_filter[selected_filter] = (
            loader_extensions.copy() if isinstance(loader_extensions, dict) else {}
        )
        return selected_filter, func, kwargs

    def _create_standalone_app_action(self, key: str) -> QtWidgets.QAction:
        spec = self._standalone_app_specs[key]
        action = QtWidgets.QAction(spec.text, self)
        action.triggered.connect(
            lambda _checked=False, app_key=key: self._show_standalone_app(app_key)
        )
        if spec.shortcut is not None:
            action.setShortcut(spec.shortcut)
        action.setToolTip(spec.tooltip)
        if spec.icon_name is not None:
            action.setIcon(QtGui.QIcon.fromTheme(spec.icon_name))
        self._standalone_app_actions[key] = action
        return action

    def _create_standalone_app_window(self, key: str) -> QtWidgets.QWidget:
        widget = self._standalone_app_specs[key].factory()
        widget.destroyed.connect(
            lambda _=None, app_key=key: self._standalone_app_windows.pop(app_key, None)
        )
        self._standalone_app_windows[key] = widget
        return widget

    def _ensure_standalone_app(self, key: str) -> QtWidgets.QWidget:
        widget = self._standalone_app_windows.get(key)
        if widget is None or not erlab.interactive.utils.qt_is_valid(widget):
            widget = self._create_standalone_app_window(key)
        return widget

    def _show_standalone_app(self, key: str) -> QtWidgets.QWidget:
        widget = self._ensure_standalone_app(key)
        widget.show()
        widget.activateWindow()
        widget.raise_()
        return widget

    def _close_standalone_apps(self) -> None:
        for widget in tuple(self._standalone_app_windows.values()):
            if erlab.interactive.utils.qt_is_valid(widget):
                widget.close()
                widget.deleteLater()
        self._standalone_app_windows.clear()

    @QtCore.Slot()
    def show_selected(self) -> None:
        """Show selected windows."""
        index_list = self._selected_imagetool_targets()

        require_unarchive = any(self._target_is_archived(i) for i in index_list)
        if require_unarchive:  # pragma: no branch
            # This is just to display the wait dialog for unarchiving.
            # If this part is removed, the showing will just hang until the unarchiving
            # is finished without any feedback.
            self.unarchive_selected()

        for index in index_list:
            self._node_for_target(index).show()

        uid_list = self._selected_tool_uids()

        for uid in uid_list:
            self.show_childtool(uid)

    @QtCore.Slot()
    def hide_selected(self) -> None:
        """Hide selected windows."""
        for index in self._selected_imagetool_targets():
            self._node_for_target(index).hide()
        for uid in self._selected_tool_uids():
            self.get_childtool(uid).hide()

    @QtCore.Slot()
    def hide_all(self) -> None:
        """Hide all windows."""
        for node in self._all_nodes.values():
            node.hide()

    @QtCore.Slot()
    def reload_selected(self) -> None:
        """Reload data in selected ImageTool windows."""
        for index in self._selected_imagetool_targets():
            self._node_for_target(index).reload()

    @QtCore.Slot()
    def remove_selected(self) -> None:
        """Discard selected ImageTool windows."""
        indices = list(self._selected_imagetool_targets())
        child_uids = list(self._selected_tool_uids())

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Remove selected windows?")

        count: int = len(indices)
        num_selected_children: int = len(child_uids)
        num_implicit_children: int = 0
        for i in indices:
            for uid in self._child_targets_of(i):
                if uid not in child_uids:  # pragma: no branch
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
            self._remove_imagetools(indices, child_uids=child_uids)

    @property
    def _rename_dialog(self) -> _RenameDialog:
        if not hasattr(self, "__rename_dialog"):
            self.__rename_dialog = _RenameDialog(self)
        return self.__rename_dialog

    @QtCore.Slot()
    def rename_selected(self) -> None:
        """Rename selected ImageTool windows."""
        selected_images = self._selected_imagetool_targets()
        selected_tools = self._selected_tool_uids()
        if len(selected_images) + len(selected_tools) == 1:
            target = selected_images[0] if selected_images else selected_tools[0]
            self.tree_view.edit(self.tree_view._model._row_index(target))
            return

        if selected_tools or any(
            not isinstance(target, int) for target in selected_images
        ):
            return

        dlg = self._rename_dialog
        root_selected = typing.cast("list[int]", selected_images)
        dlg.set_names(
            root_selected, [self._imagetool_wrappers[i].name for i in root_selected]
        )
        dlg.open()

    @QtCore.Slot()
    def duplicate_selected(self) -> None:
        """Duplicate selected windows."""
        indices = list(self._selected_imagetool_targets())
        child_uids = list(self._selected_tool_uids())
        self.tree_view.deselect_all()

        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self.tree_view.selectionModel()
        )
        try:
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
        except Exception:
            self._show_operation_error(
                "Error while duplicating selected windows",
                "An error occurred while duplicating the selected window.",
            )

    @QtCore.Slot()
    def promote_selected(self) -> None:
        """Promote the selected nested ImageTool to a top-level window."""
        uid = self._selected_promotable_child_imagetool_uid()
        if uid is None:
            return

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Promote selected ImageTool to a top-level window?")
        msg_box.setInformativeText(
            "This will detach the ImageTool from its parent. Live update linkage and "
            "automatic updates from the parent will be removed, but existing "
            "provenance and derivation metadata will be retained as detached history."
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)

        if msg_box.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self.promote_child_imagetool(uid)

    @QtCore.Slot(str)
    def promote_child_imagetool(self, uid: str) -> int:
        """Promote the nested ImageTool identified by ``uid`` to a top-level row."""
        node = self._child_node(uid)
        if not node.is_imagetool:
            raise KeyError(f"Target {uid!r} is not an ImageTool")
        if node.archived:
            raise KeyError(f"Target {uid!r} is archived")

        row_index = self.tree_view._model._row_index(uid)
        was_expanded = row_index.isValid() and self.tree_view.isExpanded(row_index)

        promoted_window = node.take_window()
        if not isinstance(promoted_window, ImageTool):
            raise TypeError(f"Unable to detach ImageTool window for {uid!r}")

        childtool_indices = list(node._childtool_indices)
        childtools = dict(node._childtools)
        created_time = node._created_time
        recent_geometry = node._recent_geometry
        provenance_spec = node.provenance_spec

        self.tree_view.childtool_removed(uid)
        self._unregister_node(uid)

        new_index = self.add_imagetool(
            promoted_window,
            show=False,
            uid=uid,
            provenance_spec=provenance_spec,
        )
        wrapper = self._imagetool_wrappers[new_index]
        wrapper._created_time = created_time
        wrapper._recent_geometry = recent_geometry
        wrapper._childtool_indices = childtool_indices
        wrapper._childtools = childtools
        if wrapper.imagetool is not None:
            wrapper.imagetool.setWindowTitle(wrapper.label_text)
        node.deleteLater()

        promoted_index = self.tree_view._model._row_index(new_index)
        if was_expanded:
            self.tree_view.expand(promoted_index)
        self.tree_view.deselect_all()
        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self.tree_view.selectionModel()
        )
        selection_model.select(
            QtCore.QItemSelection(promoted_index, promoted_index),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )
        self.tree_view.setCurrentIndex(promoted_index)
        self.tree_view.scrollTo(promoted_index)
        self.tree_view.refresh(new_index)
        self._update_actions()
        return new_index

    @QtCore.Slot()
    @QtCore.Slot(bool)
    @QtCore.Slot(bool, bool)
    def link_selected(self, link_colors: bool = True, deselect: bool = True) -> None:
        """Link selected ImageTool windows."""
        self.unlink_selected(deselect=False)
        self.link_imagetools(
            *self._selected_imagetool_targets(), link_colors=link_colors
        )
        if deselect:
            self.tree_view.deselect_all()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    def unlink_selected(self, deselect: bool = True) -> None:
        """Unlink selected ImageTool windows."""
        for index in self._selected_imagetool_targets():
            self.get_imagetool(index).slicer_area.unlink()
        self._sigReloadLinkers.emit()
        if deselect:
            self.tree_view.deselect_all()

    @QtCore.Slot()
    def archive_selected(self) -> None:
        """Archive selected ImageTool windows."""
        with erlab.interactive.utils.wait_dialog(self, "Archiving..."):
            for index in self._selected_imagetool_targets():
                self._node_for_target(index).archive()

    @QtCore.Slot()
    def unarchive_selected(self) -> None:
        """Unarchive selected ImageTool windows."""
        with erlab.interactive.utils.wait_dialog(self, "Unarchiving..."):
            for index in self._selected_imagetool_targets():
                self._node_for_target(index).unarchive()

    @property
    def _concat_dialog(self) -> _ConcatDialog:
        if not hasattr(self, "__concat_dialog"):
            self.__concat_dialog = _ConcatDialog(self)
        return self.__concat_dialog

    @QtCore.Slot()
    def concat_selected(self) -> None:
        """Concatenate the selected data using :func:`xarray.concat`."""
        dlg = self._concat_dialog
        dlg.open()

    @QtCore.Slot()
    def store_selected(self) -> None:
        self.ensure_console_initialized()
        dialog = _StoreDialog(
            self,
            [
                target
                for target in self._selected_imagetool_targets()
                if isinstance(target, int)
            ],
        )
        dialog.exec()

    @QtCore.Slot()
    def unwatch_selected(self) -> None:
        """Unwatch selected ImageTool windows."""
        for index in self.tree_view.selected_imagetool_indices:
            self._imagetool_wrappers[index].unwatch()

    def rename_imagetool(self, index: int, new_name: str) -> None:
        """Rename the ImageTool window corresponding to the given index."""
        self._imagetool_wrappers[index].name = new_name

    def _duplicate_subtree(
        self, target: int | str, *, parent_override: int | str | None = None
    ) -> int | str:
        node = self._node_for_target(target)
        if node.is_imagetool:
            duplicated_window = self.get_imagetool(target).duplicate(_in_manager=True)
            if isinstance(node, _ImageToolWrapper):
                new_target: int | str = self.add_imagetool(
                    duplicated_window,
                    activate=True,
                    source_input_ndim=node.source_input_ndim,
                    provenance_spec=node.provenance_spec,
                    source_spec=node.source_spec,
                    source_auto_update=node.source_auto_update,
                    source_state=node.source_state,
                )
            else:
                parent_target = (
                    parent_override
                    if parent_override is not None
                    else (self._parent_node(node).uid)
                )
                new_target = self.add_imagetool_child(
                    duplicated_window,
                    parent_target,
                    activate=True,
                    provenance_spec=node.provenance_spec,
                    source_spec=node.source_spec,
                    source_auto_update=node.source_auto_update,
                    source_state=node.source_state,
                    output_id=node.output_id,
                )
        else:
            tool = typing.cast("erlab.interactive.utils.ToolWindow", node.tool_window)
            parent_target = (
                parent_override
                if parent_override is not None
                else self._parent_node(node).uid
            )
            new_target = self.add_childtool(tool.duplicate(), parent_target)

        for child_uid in node._childtool_indices:
            self._duplicate_subtree(child_uid, parent_override=new_target)
        return new_target

    def duplicate_imagetool(self, index: int | str) -> int | str:
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
        return self._duplicate_subtree(index)

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
        duplicated = self._duplicate_subtree(uid)
        if isinstance(duplicated, str):
            return duplicated
        raise TypeError("Expected duplicated child target to remain nested")

    def link_imagetools(self, *indices: int | str, link_colors: bool = True) -> None:
        """Link the ImageTool windows corresponding to the given indices."""
        linker = erlab.interactive.imagetool.viewer.SlicerLinkProxy(
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

    def _annotate_workspace_dataset(
        self,
        ds: xr.Dataset,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        kind: typing.Literal["imagetool", "tool"],
    ) -> xr.Dataset:
        ds.attrs["manager_node_uid"] = node.uid
        ds.attrs["manager_node_kind"] = kind
        if node.provenance_spec is not None:
            ds.attrs["manager_node_provenance_spec"] = json.dumps(
                node.provenance_spec.model_dump(mode="json")
            )
        if isinstance(node, _ImageToolWrapper) and node.source_input_ndim is not None:
            ds.attrs["manager_node_source_input_ndim"] = int(node.source_input_ndim)
        output_id = node.output_id
        if kind == "imagetool" and output_id is not None:
            ds.attrs["manager_node_output_id"] = output_id
        if kind == "imagetool" and node.source_spec is not None:
            ds.attrs["manager_node_live_source_spec"] = json.dumps(
                node.source_spec.model_dump(mode="json")
            )
        if kind == "imagetool" and (
            node.source_spec is not None or output_id is not None
        ):
            ds.attrs["manager_node_source_state"] = node.source_state
            ds.attrs["manager_node_source_auto_update"] = bool(node.source_auto_update)
        return ds

    def _serialize_workspace_node(
        self,
        constructor: dict[str, xr.Dataset],
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: str,
        *,
        include_children: bool,
    ) -> None:
        if node.is_imagetool:
            target: int | str = (
                node.index if isinstance(node, _ImageToolWrapper) else node.uid
            )
            ds = self.get_imagetool(target).to_dataset()
            if isinstance(node, _ImageToolWrapper):
                ds.attrs["itool_title"] = (
                    ds.attrs["itool_title"]
                    .removeprefix(f"{node.index}")
                    .removeprefix(": ")
                )
            constructor[f"{path}/imagetool"] = self._annotate_workspace_dataset(
                ds, node, kind="imagetool"
            )
        else:
            tool = typing.cast("erlab.interactive.utils.ToolWindow", node.tool_window)
            if not tool.can_save_and_load():
                return
            constructor[f"{path}/tool"] = self._annotate_workspace_dataset(
                tool.to_dataset(), node, kind="tool"
            )

        if not include_children:
            return
        for child_uid in node._childtool_indices:
            child = self._child_node(child_uid)
            self._serialize_workspace_node(
                constructor,
                child,
                f"{path}/childtools/{child_uid}",
                include_children=include_children,
            )

    def _to_datatree(
        self, close: bool = False, include_children: bool = True
    ) -> xr.DataTree:
        """Convert the current state of the manager to a DataTree object."""
        constructor: dict[str, xr.Dataset] = {}
        for index in tuple(self._imagetool_wrappers.keys()):
            self._serialize_workspace_node(
                constructor,
                self._imagetool_wrappers[index],
                str(index),
                include_children=include_children,
            )
            if close:
                self.remove_imagetool(index)
        tree = xr.DataTree.from_dict(constructor)
        tree.attrs["imagetool_workspace_schema_version"] = 3
        return tree

    def _load_workspace_node(
        self,
        node_tree: xr.DataTree,
        *,
        parent_target: int | str | None = None,
        selection_item: QtWidgets.QTreeWidgetItem | None = None,
    ) -> int | str:
        if "imagetool" in node_tree:
            ds = typing.cast("xr.DataTree", node_tree["imagetool"]).to_dataset(
                inherit=False
            )
            uid = ds.attrs.get("manager_node_uid")
            provenance_spec = ds.attrs.get("manager_node_provenance_spec")
            live_source_spec = ds.attrs.get("manager_node_live_source_spec")
            parse_provenance_spec = (
                erlab.interactive.imagetool.provenance.parse_tool_provenance_spec
            )
            parsed_provenance_spec = None
            if provenance_spec is not None:
                try:
                    provenance_payload = typing.cast(
                        "Mapping[str, typing.Any]",
                        json.loads(provenance_spec),
                    )
                    parsed_provenance_spec = parse_provenance_spec(provenance_payload)
                except Exception:
                    logger.warning(
                        "Ignoring invalid saved manager provenance for node %s",
                        uid,
                        exc_info=True,
                    )
            parsed_source_spec = None
            if live_source_spec is not None:
                try:
                    source_payload = typing.cast(
                        "Mapping[str, typing.Any]",
                        json.loads(live_source_spec),
                    )
                    parsed_source_spec = (
                        erlab.interactive.imagetool.provenance.require_live_source_spec(
                            parse_provenance_spec(source_payload)
                        )
                    )
                except Exception:
                    logger.warning(
                        "Ignoring invalid saved manager source provenance for node %s",
                        uid,
                        exc_info=True,
                    )
            kwargs: dict[str, typing.Any] = {
                "uid": uid,
                "provenance_spec": parsed_provenance_spec,
                "source_spec": parsed_source_spec,
                "output_id": ds.attrs.get("manager_node_output_id"),
                "source_auto_update": bool(
                    ds.attrs.get("manager_node_source_auto_update", False)
                ),
                "source_state": typing.cast(
                    "_ManagedWindowNode._source_state_type",
                    ds.attrs.get("manager_node_source_state", "fresh"),
                ),
            }
            tool = ImageTool.from_dataset(ds, _in_manager=True)
            target: int | str
            if parent_target is None:
                kwargs.pop("output_id", None)
                kwargs["source_input_ndim"] = typing.cast(
                    "int | None",
                    ds.attrs.get("manager_node_source_input_ndim"),
                )
                target = self.add_imagetool(
                    tool, show=ds.attrs.get("itool_visible", True), **kwargs
                )
            else:
                target = self.add_imagetool_child(
                    tool,
                    parent_target,
                    show=ds.attrs.get("itool_visible", True),
                    **kwargs,
                )
        elif "tool" in node_tree:
            ds = typing.cast("xr.DataTree", node_tree["tool"]).to_dataset(inherit=False)
            target = self.add_childtool(
                erlab.interactive.utils.ToolWindow.from_dataset(ds),
                typing.cast("int | str", parent_target),
                show=ds.attrs.get("tool_visible", True),
                uid=ds.attrs.get("manager_node_uid"),
            )
        else:
            raise ValueError("Workspace node has no supported window payload")

        if "childtools" in node_tree:
            for i, child_node in enumerate(
                typing.cast("xr.DataTree", node_tree["childtools"]).values()
            ):
                child_item = (
                    selection_item.child(i) if selection_item is not None else None
                )
                if (
                    child_item is not None
                    and child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked
                ):
                    continue
                self._load_workspace_node(
                    typing.cast("xr.DataTree", child_node),
                    parent_target=target,
                    selection_item=child_item,
                )
        return target

    def _from_datatree(self, tree: xr.DataTree) -> None:
        """Restore the state of the manager from a DataTree object."""
        opened_tree = tree
        try:
            if not self._is_datatree_workspace(tree):
                raise ValueError("Not a valid workspace file")

            schema_version = tree.attrs.get("imagetool_workspace_schema_version", 1)
            match schema_version:
                case 1:
                    tree = self._parse_datatree_compat_v1(tree)
                case 2:
                    tree = self._parse_datatree_compat_v2(tree)
                case 3:
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported workspace schema version {schema_version}, "
                        "file may be from a newer version of erlab"
                    )

            dialog = _ChooseFromDataTreeDialog(self, tree, mode="load")
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                with erlab.interactive.utils.wait_dialog(self, "Loading workspace..."):
                    for i, node in enumerate(tree.values()):
                        item = dialog._tree_widget.topLevelItem(i)
                        if dialog.imagetool_selected(i):  # pragma: no branch
                            self._load_workspace_node(
                                typing.cast("xr.DataTree", node),
                                selection_item=item,
                            )
        finally:
            tree.close()
            if tree is not opened_tree:
                opened_tree.close()

    def _parse_datatree_compat_v1(self, tree: xr.DataTree) -> xr.DataTree:
        """Restore the state of the manager from a DataTree object.

        This is for the legacy format where only imagetools are stored at the root level
        (saved from erlab v3.14.1 and earlier).
        """
        return xr.DataTree.from_dict(
            {f"{i}/imagetool": node.dataset for i, node in tree.items()}
        )

    def _parse_datatree_compat_v2(self, tree: xr.DataTree) -> xr.DataTree:
        constructor: dict[str, xr.Dataset] = {}
        for key, node in tree.items():
            constructor[f"{key}/imagetool"] = typing.cast(
                "xr.DataTree", node["imagetool"]
            ).to_dataset(inherit=False)
            if "childtools" in node:
                for child_key, child_node in typing.cast(
                    "xr.DataTree", node["childtools"]
                ).items():
                    constructor[f"{key}/childtools/{child_key}/tool"] = typing.cast(
                        "xr.DataTree", child_node
                    ).to_dataset(inherit=False)
        converted = xr.DataTree.from_dict(constructor)
        converted.attrs["imagetool_workspace_schema_version"] = 3
        return converted

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
        dialog.setNameFilter("ImageTool Workspace Files (*.itws)")
        dialog.setDefaultSuffix("itws")
        if self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            fname = dialog.selectedFiles()[0]
            self._recent_directory = os.path.dirname(fname)
            try:
                self._save_to_file(fname)
            except Exception:
                self._show_operation_error(
                    "Error while saving workspace",
                    "An error occurred while saving the workspace file.",
                )

    def _save_to_file(self, fname: str):
        tree: xr.DataTree = self._to_datatree()
        try:
            dialog = _ChooseFromDataTreeDialog(self, tree, mode="save")
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return

            def _prune(node: xr.DataTree, item: QtWidgets.QTreeWidgetItem) -> None:
                if "childtools" not in node:
                    return
                child_tree = typing.cast("xr.DataTree", node["childtools"])
                for i in reversed(range(item.childCount())):
                    child_item = typing.cast("QtWidgets.QTreeWidgetItem", item.child(i))
                    child_key = str(child_item.data(0, QtCore.Qt.ItemDataRole.UserRole))
                    if child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked:
                        del child_tree[child_key]
                        continue
                    _prune(
                        typing.cast("xr.DataTree", child_tree[child_key]), child_item
                    )
                if len(child_tree) == 0:
                    del node["childtools"]

            root_item = dialog._tree_widget.invisibleRootItem()
            if root_item is None:
                return
            for i in reversed(range(root_item.childCount())):
                item = typing.cast("QtWidgets.QTreeWidgetItem", root_item.child(i))
                key = str(item.data(0, QtCore.Qt.ItemDataRole.UserRole))
                if item.checkState(0) == QtCore.Qt.CheckState.Unchecked:
                    del tree[key]
                    continue
                _prune(typing.cast("xr.DataTree", tree[key]), item)
            with erlab.interactive.utils.wait_dialog(self, "Saving workspace..."):
                tree.to_netcdf(fname, engine="h5netcdf", invalid_netcdf=True)
        finally:
            tree.close()

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
        dialog.setNameFilters(
            ["ImageTool Workspace Files (*.itws)", "xarray HDF5 Files (*.h5)"]
        )
        if self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():  # pragma: no branch
            fname = dialog.selectedFiles()[0]
            self._recent_directory = os.path.dirname(fname)
            try:
                self._from_datatree(
                    xr.open_datatree(
                        fname, engine="h5netcdf", chunks="auto", phony_dims="sort"
                    )
                )
            except Exception:
                logger.exception(
                    "Error while loading workspace",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self,
                    "Error",
                    "An error occurred while loading the workspace file.",
                )
                self.load(native=native)

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

        preferred_name_filter = self._preferred_name_filter(valid_loaders)
        if preferred_name_filter is not None:
            dialog.selectNameFilter(preferred_name_filter)
        if self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)

        if dialog.exec():
            file_names = dialog.selectedFiles()
            self._recent_name_filter = dialog.selectedNameFilter()
            self._recent_directory = os.path.dirname(file_names[0])
            func, kwargs = valid_loaders[self._recent_name_filter]
            if _is_loader_func(func):
                selected = self._select_loader_options(
                    {self._recent_name_filter: (func, kwargs)},
                    self._recent_name_filter,
                    sample_paths=file_names,
                )
                if selected is None:
                    return
                self._recent_name_filter, func, kwargs = selected
            self._add_from_multiple_files(
                loaded=[],
                queued=[pathlib.Path(f) for f in file_names],
                failed=[],
                func=func,
                kwargs=kwargs,
                retry_callback=lambda _: self.open(native=native),
            )

    @QtCore.Slot(list, dict)
    def _data_recv(
        self,
        data: list[xr.DataArray] | list[xr.Dataset],
        kwargs: dict[str, typing.Any],
        *,
        watched_var: tuple[str, str] | None = None,
        show: bool | None = None,
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
        show
            Whether to show the created windows. By default, only show if `data`
            contains only one DataArray.

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
                except Exception:
                    flags.append(False)
                    logger.exception(
                        "Error creating ImageTool window",
                        extra={"suppress_ui_alert": True},
                    )
                    self._error_creating_imagetool()
                else:
                    flags.append(True)
            return flags

        link = kwargs.pop("link", False)
        link_colors = kwargs.pop("link_colors", True)
        indices: list[int] = []
        kwargs["_in_manager"] = True

        load_func = kwargs.pop("load_func", None)
        if show is None:
            show = len(data) == 1

        for i, d in enumerate(data):
            # Set index-specific load function if provided
            this_load_func = (*load_func[:2], i) if load_func else None
            try:
                indices.append(
                    self.add_imagetool(
                        ImageTool(d, **kwargs, load_func=this_load_func),
                        show=show,
                        activate=show,
                        watched_var=watched_var,
                        source_input_ndim=d.ndim,
                        source_input_dtype=d.dtype,
                    )
                )
                if watched_var is not None:
                    # Refresh title to include variable name
                    self.get_imagetool(indices[-1])._update_title()
            except Exception:
                flags.append(False)
                logger.exception(
                    "Error creating ImageTool window",
                    extra={"suppress_ui_alert": True},
                )
                self._error_creating_imagetool()
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
        if loader_name == "ask":
            self._handle_dropped_files([pathlib.Path(p) for p in paths])
            return

        self._add_from_multiple_files(
            [],
            [pathlib.Path(p) for p in paths],
            [],
            func=erlab.io.loaders[loader_name].load,
            kwargs=kwargs,
            retry_callback=lambda _: self._data_load(paths, loader_name, kwargs),
        )

    @QtCore.Slot(list, list)
    def _data_replace(
        self, data_list: list[xr.DataArray], indices: list[int | str]
    ) -> None:
        """Replace data in the ImageTool windows with the given data."""
        for darr, idx in zip(data_list, indices, strict=True):
            if isinstance(idx, int) and idx < 0:
                # Negative index counts from the end
                idx = sorted(self._imagetool_wrappers.keys())[idx]
            elif isinstance(idx, int) and idx == self.next_idx:
                # If not yet created, add new tool
                self._data_recv([darr], {})
                continue
            self.get_imagetool(idx).slicer_area.replace_source_data(darr)
        self._sigDataReplaced.emit()

    def _find_watched_idx(self, uid: str) -> int | None:
        """Find the index of the watched ImageTool corresponding to the given UID."""
        for k, v in self._imagetool_wrappers.items():
            if v._watched_uid == uid:
                return k
        return None

    def color_for_watched_var_kernel(self, kernel_uid: str) -> QtGui.QColor:
        """Return a different color for different source kernels."""
        all_kernel_uids = tuple(
            typing.cast("str", v._watched_uid).removeprefix(f"{v._watched_varname} ")
            for v in self._imagetool_wrappers.values()
            if v.watched
        )
        idx = all_kernel_uids.index(kernel_uid)
        return _WATCHED_VAR_COLORS[idx % len(_WATCHED_VAR_COLORS)]

    @QtCore.Slot(str)
    def _remove_watched(self, uid: str) -> None:
        """Remove the ImageTool corresponding to the given watched variable UID."""
        idx = self._find_watched_idx(uid)
        if idx is not None:  # pragma: no branch
            self.remove_imagetool(idx)
            return
        if uid in self._all_nodes:
            if isinstance(self._all_nodes[uid], _ImageToolWrapper):
                self.remove_imagetool(
                    typing.cast("_ImageToolWrapper", self._all_nodes[uid]).index
                )
            else:
                self._remove_childtool(uid)

    @QtCore.Slot(str)
    def _show_watched(self, uid: str) -> None:
        """Show the ImageTool corresponding to the given watched variable UID."""
        idx = self._find_watched_idx(uid)
        if idx is not None:
            self.show_imagetool(idx)
            return
        if uid in self._all_nodes:
            self._node_for_target(uid).show()

    @QtCore.Slot(str, str, object)
    def _data_watched_update(self, varname: str, uid: str, darr: xr.DataArray) -> None:
        """Update ImageTool window corresponding to the given watched variable."""
        idx = self._find_watched_idx(uid)
        if idx is None:
            # If the tool does not exist, create a new one
            self._data_recv([darr], {}, watched_var=(varname, uid))
        else:
            # Update data in the existing tool
            self._imagetool_wrappers[idx].set_source_input_ndim(darr.ndim)
            self._imagetool_wrappers[idx].set_source_input_dtype(darr.dtype)
            self.get_imagetool(idx).slicer_area.replace_source_data(darr)

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
            if index_or_uid in self._all_nodes and self._is_imagetool_target(
                index_or_uid
            ):
                index: int | str | None = index_or_uid
            else:
                index = self._find_watched_idx(index_or_uid)
        else:
            index = index_or_uid

        if index is None:
            return None
        with contextlib.suppress(KeyError):
            return self.get_imagetool(index).slicer_area._data
        return None

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
        if self._recent_name_filter is not None:  # pragma: no branch
            for k in erlab.io.loaders:  # pragma: no branch
                if self._recent_name_filter in erlab.io.loaders[k].file_dialog_methods:
                    return k
        return None

    def ensure_explorer_initialized(self) -> None:
        """Ensure that the data explorer window is initialized."""
        self._ensure_standalone_app("explorer")

    @QtCore.Slot()
    def show_explorer(self) -> None:
        """Show data explorer window."""
        self._show_standalone_app("explorer")

    @QtCore.Slot()
    def show_ptable(self) -> None:
        """Show the periodic-table explorer window."""
        self._show_standalone_app("ptable")

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
                self._handle_dropped_files(
                    [pathlib.Path(url.toLocalFile()) for url in urls]
                )

    def _handle_dropped_files(self, file_paths: list[pathlib.Path]) -> None:
        """Handle files dropped into the window."""
        if file_paths:  # pragma: no branch
            extensions: set[str] = {file_path.suffix for file_path in file_paths}
            if len(extensions) != 1:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    "Multiple file types cannot be opened at the same time.",
                )
                return
            self.open_multiple_files(
                file_paths,
                try_workspace=(extensions == {".itws"} or extensions == {".h5"}),
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
                    dt = xr.open_datatree(
                        p, engine="h5netcdf", chunks="auto", phony_dims="sort"
                    )
                except Exception:
                    logger.debug("Failed to open %s as datatree workspace", p)
                else:
                    if self._is_datatree_workspace(dt):
                        try:
                            self._from_datatree(dt)
                        except Exception:
                            logger.exception(
                                "Error while loading workspace",
                                extra={"suppress_ui_alert": True},
                            )
                            erlab.interactive.utils.MessageDialog.critical(
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
            if all(file_path.is_dir() for file_path in queued):
                # If all dropped paths are directories, open them in the explorer
                explorer = typing.cast(
                    "_TabbedExplorer", self._show_standalone_app("explorer")
                )
                for file_path in queued:
                    explorer.add_tab(root_path=file_path)
                return

            singular: bool = n_files == 1
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"The selected {'file' if singular else 'files'} "
                f"with extension '{queued[0].suffix}' {'is' if singular else 'are'} "
                "not supported by any available plugin.",
            )
            return

        if len(valid_loaders) == 1:
            name_filter, (func, kargs) = next(iter(valid_loaders.items()))
            if _is_loader_func(func):
                selected = self._select_loader_options(
                    valid_loaders, name_filter, sample_paths=queued
                )
                if selected is None:
                    return
                self._recent_name_filter, func, kargs = selected
            else:
                self._recent_name_filter = name_filter
        else:
            selected = self._select_loader_options(valid_loaders, sample_paths=queued)
            if selected is None:
                return
            self._recent_name_filter, func, kargs = selected

        self._add_from_multiple_files(
            loaded, queued, failed, func, kargs, self.open_multiple_files
        )

    def _error_creating_imagetool(self) -> None:
        """Show an error message when an ImageTool window could not be created."""
        erlab.interactive.utils.MessageDialog.critical(
            self,
            "Error",
            "An error occurred while creating the ImageTool window.",
            "The data may be incompatible with ImageTool.",
        )

    def _show_operation_error(self, log_message: str, text: str) -> None:
        logger.exception(log_message, extra={"suppress_ui_alert": True})
        erlab.interactive.utils.MessageDialog.critical(
            self,
            "Error",
            text,
            detailed_text=erlab.interactive.utils._format_traceback(
                traceback.format_exc()
            ),
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

        def _finished_callback(loaded_new, aborted, failed_new) -> None:
            self._show_loaded_info(
                loaded + loaded_new,
                aborted,
                failed + failed_new,
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
        self,
        tool: erlab.interactive.utils.ToolWindow,
        index: int | str,
        *,
        show: bool = True,
        uid: str | None = None,
    ) -> str:
        """Register a child tool window.

        This is mainly used for handling tool windows such as goldtool and dtool opened
        from child ImageTool windows.

        Parameters
        ----------
        tool
            The tool window to add.
        index
            Target of the parent managed window.
        show
            Whether to show the tool window after adding it, by default `True`.
        """
        parent = self._node_for_target(index)
        node = _ManagedWindowNode(
            self,
            self._next_node_uid(uid),
            parent.uid,
            tool,
        )
        if not tool._tool_display_name:
            tool._tool_display_name = parent.name

        def _parent_source_fetcher(parent_uid: str = parent.uid) -> xr.DataArray:
            return self._node_for_target(parent_uid).current_source_data()

        def _parent_provenance_fetcher(
            parent_uid: str = parent.uid,
        ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
            return self._node_for_target(parent_uid).provenance_spec

        tool.set_source_parent_fetcher(_parent_source_fetcher)
        tool.set_input_provenance_parent_fetcher(_parent_provenance_fetcher)
        self._register_child_node(node)
        self.tree_view.childtool_added(node.uid, index)
        if show:
            node.show()
        return node.uid

    def add_imagetool_child(
        self,
        tool: ImageTool,
        parent: int | str,
        *,
        show: bool = True,
        activate: bool = False,
        uid: str | None = None,
        provenance_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        output_id: str | None = None,
    ) -> str:
        parent_node = self._node_for_target(parent)
        if provenance_spec is None and source_spec is not None:
            provenance_spec = (
                erlab.interactive.imagetool.provenance.compose_display_provenance(
                    parent_node.provenance_spec,
                    source_spec,
                    parent_data=parent_node.current_source_data(),
                )
            )
        if provenance_spec is not None:
            tool.set_provenance_spec(provenance_spec)
        node = _ManagedWindowNode(
            self,
            self._next_node_uid(uid),
            parent_node.uid,
            tool,
            provenance_spec=provenance_spec,
            source_spec=source_spec,
            source_auto_update=source_auto_update,
            source_state=source_state,
            output_id=output_id,
        )
        self._register_child_node(node)
        if output_id is not None and parent_node.tool_window is not None:
            parent_node.tool_window._register_output_imagetool_target(
                output_id, node.uid
            )
        self.tree_view.childtool_added(node.uid, parent)
        if show:
            node.show()
        if activate and node.window is not None:
            node.window.activateWindow()
            node.window.raise_()
        return node.uid

    def index_from_slicer_area(
        self, slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea
    ) -> int | None:
        """Get the index corresponding to the given slicer area."""
        for index, wrapper in self._imagetool_wrappers.items():  # pragma: no branch
            if (wrapper.imagetool is not None) and (wrapper.slicer_area is slicer_area):
                return index
        return None

    def wrapper_from_slicer_area(
        self, slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea
    ) -> _ImageToolWrapper | None:
        """Get the ImageTool wrapper corresponding to the given slicer area."""
        index = self.index_from_slicer_area(slicer_area)
        if index is not None:
            return self._imagetool_wrappers[index]
        return None

    def node_from_slicer_area(
        self, slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        for node in self._all_nodes.values():
            if node.imagetool is not None and node.slicer_area is slicer_area:
                return node
        return None

    def target_from_slicer_area(
        self, slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea
    ) -> int | str | None:
        node = self.node_from_slicer_area(slicer_area)
        if node is None:
            return None
        if isinstance(node, _ImageToolWrapper):
            return node.index
        return node.uid

    def _add_childtool_from_slicerarea(
        self,
        tool: QtWidgets.QWidget,
        parent_slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea,
    ) -> None:
        target = self.target_from_slicer_area(parent_slicer_area)
        if target is not None:
            if isinstance(tool, erlab.interactive.utils.ToolWindow):
                self.add_childtool(tool, target)
                return
            if isinstance(tool, ImageTool):
                self.add_imagetool_child(tool, target)
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
        node = self._child_node(uid)
        tool = node.tool_window
        if tool is None or not erlab.interactive.utils.qt_is_valid(tool):
            self._remove_childtool(uid)
            raise KeyError(f"No child tool with UID {uid} found")
        return tool, self._root_wrapper_for_uid(uid).index

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
        self._child_node(uid).show()

    def _remove_childtool(self, uid: str) -> None:
        """Unregister a child tool window.

        Parameters
        ----------
        uid
            The unique ID of the child tool to remove.
        """
        if uid not in self._all_nodes:
            return
        self.tree_view.childtool_removed(uid)
        self._remove_uid_target(uid)

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
            self.server.stop()
        if self.watcher_server.isRunning():  # pragma: no branch
            self.watcher_server.stop()

    # def __del__(self):
    # """Ensure proper cleanup of server thread when the manager is deleted."""
    # self._stop_server()

    @QtCore.Slot()
    def open_settings(self) -> None:
        """Open the settings dialog for the ImageTool manager."""
        dialog = erlab.interactive._options.OptionDialog(self)
        dialog.exec()

    @QtCore.Slot()
    def check_for_updates(self) -> None:
        from erlab.interactive.imagetool.manager._updater_gui import AutoUpdater

        updater = AutoUpdater()
        updater.check_for_updates(self)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        """Handle proper termination of resources before closing the application."""
        logger.debug("Closing ImageTool Manager...")
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

        logger.debug("Waiting for file handlers to finish...")
        if len(self._file_handlers) > 0:  # pragma: no cover
            with erlab.interactive.utils.wait_dialog(
                self, "Waiting for file operations to finish..."
            ):
                for handler in list(self._file_handlers):
                    handler.wait()

        logger.debug("Removing all ImageTool windows...")
        self.remove_all_tools()

        logger.debug("Closing additional windows...")
        for widget in dict(self._additional_windows).values():
            widget.close()
            widget.deleteLater()

        logger.debug("Removing event filters...")
        for widget in (
            self.text_box,
            self.metadata_derivation_list,
        ):
            widget.removeEventFilter(self._kb_filter)
        self.tree_view._delegate._cleanup_filter()

        if hasattr(self, "console"):
            logger.debug("Shutting down console kernel...")
            self.console._console_widget.shutdown_kernel()
            self.console.close()
            self.console.deleteLater()

        if self._standalone_app_windows:
            logger.debug("Closing standalone apps...")
            self._close_standalone_apps()

        logger.debug("Cleaning up temporary directory...")
        self._tmp_dir.cleanup()

        logger.debug("Stopping servers...")
        self._stop_servers()

        logger.debug("Closing dask client (if any)...")
        self._dask_menu.close_client()

        root_logger = logging.getLogger()
        if self._warning_handler in root_logger.handlers:  # pragma: no branch
            root_logger.removeHandler(self._warning_handler)

        self._clear_all_alerts()

        if sys.excepthook == self._handle_uncaught_exception:
            sys.excepthook = self._previous_excepthook

        super().closeEvent(event)
