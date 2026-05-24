from __future__ import annotations

import contextlib
import copy
import datetime
import gc
import json
import logging
import os
import pathlib
import platform
import subprocess
import sys
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
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME, ImageTool
from erlab.interactive.imagetool.manager import _desktop
from erlab.interactive.imagetool.manager import _server as _manager_server
from erlab.interactive.imagetool.manager import _workspace as _manager_workspace
from erlab.interactive.imagetool.manager import _xarray as _manager_xarray
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
from erlab.interactive.imagetool.manager._registry import (
    activate_manager_record,
    refresh_manager_record,
    reserve_manager_record,
    unregister_manager_record,
)
from erlab.interactive.imagetool.manager._server import _ManagerServer, _WatcherServer
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
    _MetadataField,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping

    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool._load_source import _LoadSourceDetails
    from erlab.interactive.ptable import PeriodicTableWindow

logger = logging.getLogger(__name__)

_METADATA_DERIVATION_CODE_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
_METADATA_DERIVATION_COPYABLE_ROLE = _METADATA_DERIVATION_CODE_ROLE + 1
_RECENT_WORKSPACES_SETTINGS_KEY = "recent_workspaces"
_MAX_RECENT_WORKSPACES = 10


def _manager_settings() -> QtCore.QSettings:
    return QtCore.QSettings(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        "erlabpy",
        "imagetool-manager",
    )


def _launch_new_manager_instance() -> None:
    command = [sys.executable]
    if sys.platform == "darwin" and erlab.utils.misc._IS_PACKAGED:
        for parent in pathlib.Path(sys.executable).resolve().parents:
            if parent.suffix == ".app":
                command = ["/usr/bin/open", "-n", str(parent)]
                break
    elif not erlab.utils.misc._IS_PACKAGED:
        command.extend(["-m", "erlab.interactive.imagetool.manager"])

    kwargs: dict[str, typing.Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform.startswith("win"):
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        if flags:
            kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen(command, **kwargs)


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
        self.copy_path_button = typing.cast(
            "QtWidgets.QAbstractButton",
            self.button_box.addButton(
                "Copy Path", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        self.copy_path_button.clicked.connect(
            lambda: erlab.interactive.utils.copy_to_clipboard(self.path_edit.text())
        )
        self.copy_code_button = typing.cast(
            "QtWidgets.QAbstractButton",
            self.button_box.addButton(
                "Copy Load Code", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
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
        close_button = typing.cast(
            "QtWidgets.QAbstractButton",
            self.button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close),
        )
        close_button.clicked.connect(self.accept)
        layout.addWidget(self.button_box)
        self.adjustSize()


def _workspace_file_manager_action_text() -> str:
    if sys.platform == "darwin":
        return "Reveal in Finder"
    if sys.platform.startswith("win"):
        return "Reveal in File Explorer"
    return "Open Containing Folder"


@dataclass(frozen=True)
class _WorkspacePropertiesState:
    is_modified: bool
    top_level_window_count: int


def _workspace_file_type_text(path: pathlib.Path | None) -> str:
    if path is None:
        return "Unsaved ImageTool workspace"
    match path.suffix.lower():
        case ".itws":
            return "ImageTool Workspace (.itws)"
        case ".h5":
            return "xarray HDF5 Workspace (.h5)"
        case suffix if suffix:
            return f"{suffix.removeprefix('.').upper()} file"
        case _:
            return "File"


def _format_workspace_file_size(size: int) -> str:
    if size == 1:
        return "1 byte"
    bytes_text = f"{size:,} bytes"
    if size < 1000:
        return bytes_text
    value = float(size)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1000.0
        if value < 1000.0:
            return f"{value:.2f} {unit} ({bytes_text})"
    return f"{value:.2f} PB ({bytes_text})"


def _format_workspace_file_time(timestamp: float) -> str:
    return (
        datetime.datetime.fromtimestamp(timestamp)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M:%S %Z")
    )


class _WorkspacePropertiesDialog(QtWidgets.QDialog):
    def __init__(
        self,
        workspace_path: str | os.PathLike[str] | None,
        *,
        state: _WorkspacePropertiesState,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        workspace_name = (
            "Untitled Workspace"
            if workspace_path is None
            else pathlib.Path(workspace_path).resolve().name
        )
        self.setWindowTitle(f"{workspace_name} Properties")
        self.setModal(True)
        self.setMinimumWidth(540)

        self._workspace_path = (
            None if workspace_path is None else pathlib.Path(workspace_path).resolve()
        )
        self.value_labels: dict[str, QtWidgets.QLabel] = {}
        self.copy_path_button: QtWidgets.QAbstractButton | None = None
        self.reveal_button: QtWidgets.QAbstractButton | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(14)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)
        layout.addLayout(header_layout)

        icon_label = QtWidgets.QLabel(self)
        icon_label.setObjectName("manager_workspace_properties_icon_label")
        style = self.style()
        icon = (
            QtGui.QIcon()
            if style is None
            else style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon)
        )
        icon_label.setPixmap(icon.pixmap(32, 32))
        icon_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter
        )
        header_layout.addWidget(icon_label, 0)

        title_layout = QtWidgets.QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        header_layout.addLayout(title_layout, 1)

        title_label = QtWidgets.QLabel(workspace_name, self)
        title_label.setObjectName("manager_workspace_name_label")
        title_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        title_label.setWordWrap(True)
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + 1)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)

        status_label = QtWidgets.QLabel(
            self._status_text(self._workspace_path, state), self
        )
        status_label.setObjectName("manager_workspace_status_label")
        status_label.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
        title_layout.addWidget(status_label)

        details_layout = QtWidgets.QGridLayout()
        details_layout.setContentsMargins(44, 0, 0, 0)
        details_layout.setHorizontalSpacing(14)
        details_layout.setVerticalSpacing(7)
        details_layout.setColumnStretch(1, 1)
        layout.addLayout(details_layout)

        row = 0
        if self._workspace_path is None:
            row = self._add_detail(
                details_layout,
                row,
                "type",
                "Type",
                _workspace_file_type_text(None),
            )
        else:
            row = self._add_detail(
                details_layout,
                row,
                "path",
                "Path",
                str(self._workspace_path),
            )
            row = self._add_detail(
                details_layout,
                row,
                "type",
                "Type",
                _workspace_file_type_text(self._workspace_path),
            )
            stat: os.stat_result | None = None
            with contextlib.suppress(OSError):
                stat = self._workspace_path.stat()
            if stat is None:
                row = self._add_detail(
                    details_layout,
                    row,
                    "size",
                    "Size",
                    "File not found",
                )
                row = self._add_detail(
                    details_layout,
                    row,
                    "modified",
                    "Modified",
                    "File not found",
                )
            else:
                row = self._add_detail(
                    details_layout,
                    row,
                    "size",
                    "Size",
                    _format_workspace_file_size(stat.st_size),
                )
                row = self._add_detail(
                    details_layout,
                    row,
                    "modified",
                    "Modified",
                    _format_workspace_file_time(stat.st_mtime),
                )

        row = self._add_detail(
            details_layout,
            row,
            "unsaved_changes",
            "Unsaved changes",
            "Yes" if state.is_modified else "No",
        )
        self._add_detail(
            details_layout,
            row,
            "open_windows",
            "Open windows",
            str(state.top_level_window_count),
        )

        self.button_box = QtWidgets.QDialogButtonBox(self)
        if self._workspace_path is not None:
            self.copy_path_button = typing.cast(
                "QtWidgets.QAbstractButton",
                self.button_box.addButton(
                    "Copy Path", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
                ),
            )
            self.copy_path_button.setObjectName("manager_copy_workspace_path_button")
            self.copy_path_button.clicked.connect(lambda: self._copy_path())

            self.reveal_button = typing.cast(
                "QtWidgets.QAbstractButton",
                self.button_box.addButton(
                    _workspace_file_manager_action_text(),
                    QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
                ),
            )
            self.reveal_button.setObjectName("manager_reveal_workspace_path_button")
            self.reveal_button.clicked.connect(lambda: self._reveal_path())

        close_button = typing.cast(
            "QtWidgets.QAbstractButton",
            self.button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close),
        )
        close_button.clicked.connect(self.accept)
        layout.addWidget(self.button_box)
        self.adjustSize()

    @staticmethod
    def _status_text(
        workspace_path: pathlib.Path | None, state: _WorkspacePropertiesState
    ) -> str:
        if workspace_path is None:
            return "Not saved to a file"
        if state.is_modified:
            return "Associated file with unsaved changes"
        return "Associated file"

    def _add_detail(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        key: str,
        label: str,
        value: str,
    ) -> int:
        key_label = QtWidgets.QLabel(label, self)
        key_label.setObjectName(f"manager_workspace_{key}_label")
        key_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop
        )
        key_label.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)

        value_label = QtWidgets.QLabel(value, self)
        value_label.setObjectName(f"manager_workspace_{key}_value_label")
        value_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        value_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        value_label.setWordWrap(True)
        value_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        value_label.setToolTip(value)

        layout.addWidget(key_label, row, 0)
        layout.addWidget(value_label, row, 1)
        self.value_labels[key] = value_label
        return row + 1

    def _copy_path(self) -> None:
        if self._workspace_path is None:
            return
        erlab.interactive.utils.copy_to_clipboard(str(self._workspace_path))

    def _reveal_path(self) -> None:
        if self._workspace_path is None:
            return
        erlab.utils.misc.open_in_file_manager(self._workspace_path)


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


class _ApplicationQuitFilter(QtCore.QObject):
    """Route application quit requests through the manager window."""

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self._manager = manager

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if event is None:
            return False
        if event.type() == QtCore.QEvent.Type.Quit:
            return self._manager._handle_application_quit_request()
        if (
            event.type()
            in (
                QtCore.QEvent.Type.KeyPress,
                QtCore.QEvent.Type.ShortcutOverride,
            )
            and isinstance(event, QtGui.QKeyEvent)
            and event.matches(QtGui.QKeySequence.StandardKey.Quit)
        ):
            event.accept()
            return self._manager._handle_application_quit_request()
        return False


@dataclass
class _WorkspaceDocumentAccess:
    path: pathlib.Path
    _lock: QtCore.QLockFile | None = None

    def take_lock(self) -> QtCore.QLockFile | None:
        lock = self._lock
        self._lock = None
        return lock

    def release(self) -> None:
        if self._lock is not None:
            self._lock.unlock()
            self._lock = None


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

_WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME = "managerWorkspaceSaveShortcut"
_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS = 0.5
_WORKSPACE_REBIND_KEEP_CHUNKS = object()


def _workspace_lock_owner_text(
    lock_info: _manager_workspace._WorkspaceDocumentLockInfo,
) -> str:
    parts: list[str] = []
    if lock_info.owner:
        parts.append(lock_info.owner)
    if lock_info.hostname:
        parts.append(f"on {lock_info.hostname}")
    if lock_info.appname and lock_info.pid is not None:
        parts.append(f"using {lock_info.appname} (process {lock_info.pid})")
    elif lock_info.appname:
        parts.append(f"using {lock_info.appname}")
    elif lock_info.pid is not None:
        parts.append(f"using process {lock_info.pid}")
    return " ".join(parts)


def _workspace_lock_details_text(
    fname: str | os.PathLike[str],
    lock_info: _manager_workspace._WorkspaceDocumentLockInfo,
) -> str:
    details = [
        f"Workspace: {pathlib.Path(fname)}",
        f"Temporary workspace ownership marker: {lock_info.path}",
    ]
    if lock_info.owner:
        details.append(f"User: {lock_info.owner}")
    if lock_info.hostname:
        details.append(f"Computer: {lock_info.hostname}")
    if lock_info.appname:
        details.append(f"Application: {lock_info.appname}")
    if lock_info.pid is not None:
        details.append(f"Process ID: {lock_info.pid}")
    exc = sys.exception()
    if exc is not None:
        details.extend(("", str(exc)))
    return "\n".join(details)


def _show_workspace_file_lock_error(
    parent: QtWidgets.QWidget,
    fname: str | os.PathLike[str],
) -> None:
    lock_info = _manager_workspace._workspace_document_lock_info(fname)
    owner_text = _workspace_lock_owner_text(lock_info)
    if owner_text:
        informative_text = (
            f"{pathlib.Path(fname).name} is being used by {owner_text}. "
            "Close it there, then try again."
        )
    else:
        informative_text = (
            f"Close the other ImageTool Manager that has "
            f"{pathlib.Path(fname).name} open, then try again."
        )
    erlab.interactive.utils.MessageDialog.critical(
        parent,
        "Workspace Already Open",
        "This workspace is already open somewhere else.",
        informative_text,
        detailed_text=_workspace_lock_details_text(fname, lock_info),
    )


def _strip_workspace_modified_placeholder(title: object) -> str:
    """Return a persisted title without Qt's modified-window placeholder."""
    return str(title).replace("[*]", "")


def _window_title_with_modified_placeholder(title: object) -> str:
    title = _strip_workspace_modified_placeholder(title)
    if sys.platform == "darwin":
        return title
    return f"{title}[*]"


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

        self._manager_record = reserve_manager_record(host=_manager_server.HOST_IP)
        self.manager_index = self._manager_record.index

        try:
            (
                self.server,
                self.watcher_server,
                port,
                watch_port,
            ) = self._start_manager_servers()
            self._manager_record = activate_manager_record(
                self._manager_record.internal_id,
                port=port,
                watch_port=watch_port,
            )
        except Exception:
            unregister_manager_record(self._manager_record.internal_id)
            raise

        self._registry_heartbeat_timer = QtCore.QTimer(self)
        self._registry_heartbeat_timer.setInterval(3000)
        self._registry_heartbeat_timer.timeout.connect(self._refresh_manager_record)
        self._registry_heartbeat_timer.start()

        # Shared memory for detecting multiple instances
        # No longer used starting from v3.8.2, but kept for backward compatibility
        self._shm = QtCore.QSharedMemory(_SHM_NAME)
        self._shm.create(1)  # Create segment so that it can be attached to

        self.menu_bar: QtWidgets.QMenuBar = typing.cast(
            "QtWidgets.QMenuBar", self.menuBar()
        )

        self._workspace_path: pathlib.Path | None = None
        self._workspace_link_id: str = uuid.uuid4().hex
        self._workspace_loading_depth: int = 0
        self._workspace_saving_depth: int = 0
        self._workspace_structure_modified: bool = False
        self._workspace_dirty_added: set[str] = set()
        self._workspace_dirty_data: set[str] = set()
        self._workspace_dirty_state: set[str] = set()
        self._workspace_dirty_removed: list[str] = []
        self._workspace_structure_reasons: list[str] = []
        self._workspace_needs_full_save: bool = False
        self._workspace_dirty_generation: int = 0
        self._workspace_dirty_events: list[_manager_workspace._WorkspaceDirtyEvent] = []
        self._workspace_save_in_progress: bool = False
        self._workspace_delta_save_count: int = 0
        self._workspace_schema_version: int = (
            _manager_workspace._current_workspace_schema_version()
        )
        self._workspace_lock: QtCore.QLockFile | None = None
        self._application_quit_requested: bool = False
        self._update_workspace_window_title()

        qapp = QtWidgets.QApplication.instance()
        self._application_quit_filter: _ApplicationQuitFilter | None = None
        if isinstance(qapp, QtWidgets.QApplication):
            self._application_quit_filter = _ApplicationQuitFilter(self)
            qapp.installEventFilter(self._application_quit_filter)

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
                icon_name="applications-science",
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
        self.settings_action.setIcon(QtGui.QIcon.fromTheme("preferences-system"))

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

        self.open_action = QtWidgets.QAction("Add &Data Files…", self)
        self.open_action.setObjectName("manager_add_data_files_action")
        self.open_action.triggered.connect(self.open)
        self.open_action.setToolTip("Load data files as new ImageTool rows")

        self.new_manager_action = QtWidgets.QAction("New Manager Window", self)
        self.new_manager_action.setObjectName("manager_new_instance_action")
        self.new_manager_action.triggered.connect(self.open_new_manager_instance)
        self.new_manager_action.setToolTip("Open another ImageTool Manager window")
        self.new_manager_action.setIcon(QtGui.QIcon.fromTheme("window-new"))

        self.save_action = QtWidgets.QAction("&Save", self)
        self.save_action.setObjectName("manager_save_workspace_action")
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.setToolTip("Save this workspace")
        self.save_action.setIcon(QtGui.QIcon.fromTheme("document-save"))
        self.save_action.triggered.connect(self.save)

        self.save_as_action = QtWidgets.QAction("Save Workspace &As…", self)
        self.save_as_action.setObjectName("manager_save_workspace_as_action")
        self.save_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.save_as_action.setToolTip(
            "Save this workspace to a new file and use that file for future saves"
        )
        self.save_as_action.setIcon(QtGui.QIcon.fromTheme("document-save-as"))
        self.save_as_action.triggered.connect(self.save_as)

        self.compact_workspace_action = QtWidgets.QAction("Compact Workspace", self)
        self.compact_workspace_action.setObjectName("manager_compact_workspace_action")
        self.compact_workspace_action.setToolTip(
            "Rewrite this workspace file to remove unused space"
        )
        self.compact_workspace_action.triggered.connect(self.compact_workspace)

        self.workspace_properties_action = QtWidgets.QAction(
            "Workspace Properties", self
        )
        self.workspace_properties_action.setObjectName(
            "manager_workspace_properties_action"
        )
        self.workspace_properties_action.setMenuRole(QtWidgets.QAction.MenuRole.NoRole)
        self.workspace_properties_action.setShortcut(QtGui.QKeySequence("Alt+Return"))
        self.workspace_properties_action.setToolTip(
            "Show properties for the current workspace"
        )
        self.workspace_properties_action.setIcon(
            QtGui.QIcon.fromTheme("document-properties")
        )
        self.workspace_properties_action.triggered.connect(
            self.show_workspace_properties
        )

        self.load_action = QtWidgets.QAction("&Open Workspace…", self)
        self.load_action.setObjectName("manager_open_workspace_action")
        self.load_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.load_action.setToolTip("Replace this workspace with a workspace file")
        self.load_action.setIcon(QtGui.QIcon.fromTheme("document-open"))
        self.load_action.triggered.connect(self.load)

        self.open_recent_menu = QtWidgets.QMenu("Open &Recent", self)
        self.open_recent_menu.setObjectName("manager_open_recent_menu")
        open_recent_action = self.open_recent_menu.menuAction()
        # Qt creates a menu action for every QMenu; this keeps type narrowing explicit.
        if open_recent_action is None:  # pragma: no cover
            raise RuntimeError("Open Recent menu action was not created")
        open_recent_action.setObjectName("manager_open_recent_menu_action")
        open_recent_action.setIcon(QtGui.QIcon.fromTheme("document-open-recent"))
        self.open_recent_menu.setToolTipsVisible(True)
        self.open_recent_menu.aboutToShow.connect(self._populate_open_recent_menu)
        self._refresh_open_recent_menu_action()

        self.import_workspace_action = QtWidgets.QAction(
            "Add Windows From &Workspace…", self
        )
        self.import_workspace_action.setObjectName(
            "manager_add_windows_from_workspace_action"
        )
        self.import_workspace_action.setToolTip(
            "Add selected windows from another workspace file"
        )
        self.import_workspace_action.setIcon(QtGui.QIcon.fromTheme("list-add"))
        self.import_workspace_action.triggered.connect(self.import_workspace)

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
        self.duplicate_action.setIcon(QtGui.QIcon.fromTheme("edit-copy"))

        self.promote_action = QtWidgets.QAction("Promote Window", self)
        self.promote_action.triggered.connect(self.promote_selected)
        self.promote_action.setToolTip(
            "Promote the selected nested ImageTool to a top-level window"
        )
        self.promote_action.setIcon(QtGui.QIcon.fromTheme("go-up"))

        self.reindex_action = QtWidgets.QAction("Reset Index", self)
        self.reindex_action.triggered.connect(self.reindex)
        self.reindex_action.setToolTip("Reset indices of all windows")

        self.link_action = QtWidgets.QAction("Link", self)
        self.link_action.triggered.connect(lambda _checked=False: self.link_selected())
        self.link_action.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        self.link_action.setToolTip("Link selected windows")

        self.unlink_action = QtWidgets.QAction("Unlink", self)
        self.unlink_action.triggered.connect(
            lambda _checked=False: self.unlink_selected()
        )
        self.unlink_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+L"))
        self.unlink_action.setToolTip("Unlink selected windows")

        self.offload_action = QtWidgets.QAction("Offload to Workspace", self)
        self.offload_action.triggered.connect(self.offload_selected_to_workspace)
        self.offload_action.setToolTip(
            "Free this data from memory and use dask-backed data from the "
            "workspace file"
        )

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
        self.reload_action.setToolTip("Reload selected data and refresh child paths")
        self.reload_action.setIcon(QtGui.QIcon.fromTheme("view-refresh"))
        self.reload_action.setVisible(False)

        self.unwatch_action = QtWidgets.QAction("Stop Watching", self)
        self.unwatch_action.triggered.connect(self.unwatch_selected)
        self.unwatch_action.setToolTip("Stop watching selected windows")
        self.unwatch_action.setIcon(QtGui.QIcon.fromTheme("process-stop"))
        self.unwatch_action.setVisible(False)

        self.source_update_action = QtWidgets.QAction("Automatic Updates…", self)
        self.source_update_action.triggered.connect(self.show_selected_source_updates)
        self.source_update_action.setToolTip(
            "Turn automatic updates on or off for the selected child window"
        )
        self.source_update_action.setIcon(QtGui.QIcon.fromTheme("sync-synchronizing"))
        self.source_update_action.setVisible(False)

        self.about_action = QtWidgets.QAction("About", self)
        self.about_action.setIcon(QtGui.QIcon.fromTheme("help-about"))
        self.about_action.triggered.connect(self.about)

        self.check_update_action = QtWidgets.QAction("Check for Updates", self)
        self.check_update_action.setMenuRole(
            QtWidgets.QAction.MenuRole.ApplicationSpecificRole
        )
        self.check_update_action.triggered.connect(self.check_for_updates)
        self.check_update_action.setIcon(
            QtGui.QIcon.fromTheme("software-update-available")
        )
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
        file_menu.setObjectName("manager_file_menu")
        file_menu.aboutToShow.connect(self._refresh_open_recent_menu_action)
        file_menu.addAction(self.load_action)
        file_menu.addMenu(self.open_recent_menu)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.compact_workspace_action)
        file_menu.addAction(self.workspace_properties_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.import_workspace_action)
        file_menu.addAction(self.explorer_action)
        file_menu.addSeparator()
        file_menu.addAction(self.new_manager_action)
        file_menu.addSeparator()
        file_menu.addAction(self.store_action)
        file_menu.addSeparator()
        file_menu.addAction(self.gc_action)
        file_menu.addSeparator()
        file_menu.addAction(self.show_action)
        file_menu.addAction(self.hide_action)
        file_menu.addSeparator()
        file_menu.addAction(self.remove_action)
        file_menu.addAction(self.offload_action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)

        edit_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Edit")
        )
        edit_menu.setObjectName("manager_edit_menu")
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
        view_menu.setObjectName("manager_view_menu")
        view_menu.addAction(self.console_action)
        view_menu.addSeparator()
        view_menu.addAction(self.preview_action)
        view_menu.addSeparator()

        apps_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Apps")
        )
        apps_menu.setObjectName("manager_apps_menu")
        apps_menu.addAction(self.ptable_action)

        self._dask_menu = DaskMenu(self, "Dask")
        self.menu_bar.addMenu(self._dask_menu)

        help_menu: QtWidgets.QMenu = typing.cast(
            "QtWidgets.QMenu", self.menu_bar.addMenu("&Help")
        )
        help_menu.setObjectName("manager_help_menu")
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

        # Store most recent name filter and directory for new windows
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None
        self._recent_loader_extensions_by_filter: dict[str, dict[str, typing.Any]] = {}
        self._metadata_full_code: str | None = None
        self._metadata_node_uid: str | None = None
        self._metadata_copy_selected_action = QtGui.QAction("Copy Selected Code", self)
        self._metadata_copy_selected_action.setObjectName(
            "manager_copy_selected_code_action"
        )
        self._metadata_copy_selected_action.triggered.connect(
            self._copy_selected_derivation_code
        )
        self._metadata_copy_full_action = QtGui.QAction("Copy Full Code", self)
        self._metadata_copy_full_action.setObjectName("manager_copy_full_code_action")
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

    @staticmethod
    def _normalize_recent_workspace_paths(
        paths: Iterable[str | os.PathLike[str]],
    ) -> list[pathlib.Path]:
        recent_paths: list[pathlib.Path] = []
        seen: set[str] = set()
        for value in paths:
            path = pathlib.Path(value).expanduser().resolve()
            if path.suffix.lower() != ".itws":
                continue
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            recent_paths.append(path)
            seen.add(key)
            if len(recent_paths) >= _MAX_RECENT_WORKSPACES:
                break
        return recent_paths

    def _recent_workspace_paths(self) -> list[pathlib.Path]:
        settings = _manager_settings()
        settings.sync()
        values = settings.value(_RECENT_WORKSPACES_SETTINGS_KEY, [])
        if isinstance(values, str):
            stored_paths = [values] if values else []
        elif isinstance(values, (list, tuple)):
            stored_paths = [str(value) for value in values if value]
        else:
            stored_paths = []
        return self._normalize_recent_workspace_paths(stored_paths)

    def _set_recent_workspace_paths(
        self, paths: Iterable[str | os.PathLike[str]]
    ) -> None:
        recent_paths = self._normalize_recent_workspace_paths(paths)
        settings = _manager_settings()
        if recent_paths:
            settings.setValue(
                _RECENT_WORKSPACES_SETTINGS_KEY,
                [str(path) for path in recent_paths],
            )
        else:
            settings.remove(_RECENT_WORKSPACES_SETTINGS_KEY)
        settings.sync()

    def _record_recent_workspace(self, fname: str | os.PathLike[str]) -> None:
        path = pathlib.Path(fname).expanduser().resolve()
        if path.suffix.lower() != ".itws":
            return
        path_key = os.path.normcase(str(path))
        paths = [
            existing
            for existing in self._recent_workspace_paths()
            if os.path.normcase(str(existing)) != path_key
        ]
        self._set_recent_workspace_paths([path, *paths])
        self._refresh_open_recent_menu_action()
        if erlab.utils.misc._IS_PACKAGED:
            _desktop.record_recent_workspace(path)

    @QtCore.Slot()
    def _clear_recent_workspaces(self) -> None:
        self._set_recent_workspace_paths([])
        self._populate_open_recent_menu()

    def _refresh_open_recent_menu_action(self) -> None:
        self.open_recent_menu.setEnabled(bool(self._recent_workspace_paths()))

    def _populate_open_recent_menu(self) -> None:
        self.open_recent_menu.clear()
        paths = self._recent_workspace_paths()
        self.open_recent_menu.setEnabled(bool(paths))
        if not paths:
            return

        name_counts: dict[str, int] = {}
        for path in paths:
            name_counts[path.name] = name_counts.get(path.name, 0) + 1

        for index, path in enumerate(paths):
            label = path.name
            if name_counts[path.name] > 1:
                label = f"{path.name} ({path.parent.name or path.parent})"
            action = QtWidgets.QAction(label, self.open_recent_menu)
            action.setObjectName(f"manager_recent_workspace_action_{index}")
            action.setData(str(path))
            action.setToolTip(str(path))
            action.setStatusTip(str(path))
            action.triggered.connect(
                lambda _checked=False, recent_path=path: self.open_recent_workspace(
                    recent_path
                )
            )
            self.open_recent_menu.addAction(action)

        self.open_recent_menu.addSeparator()
        clear_action = QtWidgets.QAction("Clear Menu", self.open_recent_menu)
        clear_action.setObjectName("manager_clear_recent_workspaces_action")
        clear_action.triggered.connect(self._clear_recent_workspaces)
        self.open_recent_menu.addAction(clear_action)

    @QtCore.Slot(str)
    def open_recent_workspace(self, fname: str | os.PathLike[str]) -> bool:
        """Open a recently used workspace file."""
        path = pathlib.Path(fname).expanduser().resolve()
        if not path.exists():
            path_key = os.path.normcase(str(path))
            self._set_recent_workspace_paths(
                existing
                for existing in self._recent_workspace_paths()
                if os.path.normcase(str(existing)) != path_key
            )
            self._refresh_open_recent_menu_action()
            QtWidgets.QMessageBox.warning(
                self,
                "Workspace Not Found",
                f"The recent workspace file no longer exists:\n{path}",
            )
            return False
        if not self._confirm_save_dirty_workspace(
            "Opening a workspace replaces the windows currently in this manager."
        ):
            return False
        self._recent_directory = str(path.parent)
        try:
            loaded = self._load_workspace_file(
                path,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )
        except Exception as exc:
            if _manager_workspace._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    path,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self, path)
            else:
                logger.exception(
                    "Error while loading workspace",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self,
                    "Error",
                    "An error occurred while loading the workspace file.",
                )
            return False
        if loaded:
            self._record_recent_workspace(path)
        return loaded

    @property
    def workspace_path(self) -> str | None:
        """Path of the workspace document associated with this manager."""
        return None if self._workspace_path is None else str(self._workspace_path)

    @QtCore.Slot()
    def show_workspace_properties(self) -> None:
        """Show properties for the workspace associated with this manager."""
        _WorkspacePropertiesDialog(
            self.workspace_path,
            state=self._workspace_properties_state(),
            parent=self,
        ).exec()

    def _workspace_properties_state(self) -> _WorkspacePropertiesState:
        return _WorkspacePropertiesState(
            is_modified=self.is_workspace_modified,
            top_level_window_count=self.ntools,
        )

    @property
    def is_workspace_modified(self) -> bool:
        """Return whether this workspace has unsaved restorable changes."""
        if self._workspace_path is None and not getattr(self, "_all_nodes", {}):
            return False
        return (
            self._workspace_structure_modified
            or bool(self._workspace_dirty_added)
            or bool(self._workspace_dirty_data)
            or bool(self._workspace_dirty_state)
            or bool(self._workspace_dirty_removed)
        )

    def _refresh_manager_record(self) -> None:
        refresh_manager_record(
            self._manager_record.internal_id, workspace_path=self.workspace_path
        )

    def _update_workspace_window_title(self) -> None:
        if self._workspace_path is None:
            window_file_path = ""
        else:
            window_file_path = typing.cast("str", self.workspace_path)
        self.setWindowFilePath(window_file_path)
        workspace_display_name = (
            "Untitled" if self._workspace_path is None else self._workspace_path.name
        )
        self.setWindowTitle(
            f"{_window_title_with_modified_placeholder(workspace_display_name)}"
            f" - ImageTool Manager #{self.manager_index}"
        )
        self.setWindowModified(self.is_workspace_modified)

    def _release_workspace_lock(self) -> None:
        if self._workspace_lock is None:
            return
        self._workspace_lock.unlock()
        self._workspace_lock = None

    def _workspace_document_access(
        self, fname: str | os.PathLike[str]
    ) -> _WorkspaceDocumentAccess:
        workspace_path = pathlib.Path(fname).resolve()
        workspace_lock = None
        if workspace_path != self._workspace_path:
            workspace_lock = _manager_workspace._acquire_workspace_document_lock(
                workspace_path
            )
        return _WorkspaceDocumentAccess(workspace_path, workspace_lock)

    @contextlib.contextmanager
    def _workspace_document_access_context(
        self, fname: str | os.PathLike[str]
    ) -> Iterator[_WorkspaceDocumentAccess]:
        access = self._workspace_document_access(fname)
        try:
            yield access
        finally:
            access.release()

    def _set_workspace_path(
        self,
        fname: str | os.PathLike[str] | None,
        *,
        workspace_lock: QtCore.QLockFile | None = None,
    ) -> None:
        workspace_path = None if fname is None else pathlib.Path(fname).resolve()
        if workspace_path == self._workspace_path:
            if workspace_lock is not None:
                workspace_lock.unlock()
            self._update_workspace_window_title()
            self._refresh_manager_record()
            return

        if workspace_path is not None and workspace_lock is None:
            raise RuntimeError(
                "Changing the workspace path requires a pre-acquired document lock"
            )
        self._release_workspace_lock()
        self._workspace_lock = workspace_lock
        self._workspace_path = workspace_path
        if self._workspace_path is not None:
            self._recent_directory = str(self._workspace_path.parent)
        self._update_workspace_window_title()
        self._refresh_manager_record()

    def _adopt_workspace_path(self, fname: str | os.PathLike[str]) -> None:
        with self._workspace_document_access_context(fname) as access:
            self._set_workspace_path(access.path, workspace_lock=access.take_lock())

    @property
    def _suppress_workspace_visibility_dirty(self) -> bool:
        return self._application_quit_requested

    def _handle_application_quit_request(self) -> bool:
        self._application_quit_requested = True
        if self.close():
            return True
        self._application_quit_requested = False
        return True

    def _active_managed_window(self) -> QtWidgets.QWidget | None:
        active_window = QtWidgets.QApplication.activeWindow()
        if not isinstance(active_window, QtWidgets.QWidget):
            return None
        if self._node_uid_from_window(active_window) is None:
            return None
        if not erlab.interactive.utils.qt_is_valid(active_window):
            return None
        return active_window

    def _restore_focus_after_workspace_save(
        self, origin: QtWidgets.QWidget | None
    ) -> None:
        if (
            origin is None
            or not erlab.interactive.utils.qt_is_valid(origin)
            or not origin.isVisible()
        ):
            return
        active_window = QtWidgets.QApplication.activeWindow()
        if isinstance(active_window, QtWidgets.QWidget) and active_window not in (
            self,
            origin,
        ):
            return
        origin.activateWindow()
        origin.raise_()
        focus_widget = origin.focusWidget()
        if isinstance(
            focus_widget, QtWidgets.QWidget
        ) and erlab.interactive.utils.qt_is_valid(focus_widget):
            focus_widget.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _dirty_details_text(self) -> str:
        def _node_names(uids: set[str]) -> tuple[str, ...]:
            return tuple(
                self._all_nodes[uid].display_text
                for uid in sorted(uids)
                if uid in self._all_nodes
            )

        sections = (
            ("Added", _node_names(self._workspace_dirty_added)),
            ("Removed", tuple(dict.fromkeys(self._workspace_dirty_removed))),
            (
                "Data modified",
                _node_names(self._workspace_dirty_data - self._workspace_dirty_added),
            ),
            (
                "State modified",
                _node_names(
                    self._workspace_dirty_state
                    - self._workspace_dirty_data
                    - self._workspace_dirty_added
                ),
            ),
            (
                "Structure modified",
                tuple(dict.fromkeys(self._workspace_structure_reasons)),
            ),
        )
        blocks: list[str] = []
        for label, items in sections:
            if items:
                blocks.append(f"{label}:\n" + "\n".join(f"- {item}" for item in items))
        return "\n\n".join(blocks)

    def _set_node_window_modified(self, uid: str, modified: bool) -> None:
        node = self._all_nodes.get(uid)
        if node is None:
            return
        window = node.window
        if window is None or not erlab.interactive.utils.qt_is_valid(window):
            return
        if node.tool_window is not None:
            display_name = node.tool_window._tool_display_name
            base_title = (
                f"{node.tool_window.tool_name}: {display_name}"
                if display_name
                else node.tool_window.tool_name
            )
        else:
            base_title = node.label_text
        title = _window_title_with_modified_placeholder(base_title)
        if title != window.windowTitle():
            window.setWindowTitle(title)
        window.setWindowModified(modified)

    def _apply_workspace_dirty_event(
        self, event: _manager_workspace._WorkspaceDirtyEvent
    ) -> bool:
        dirty_changed = False
        if event.uid is not None:
            if event.added:
                self._workspace_dirty_added.add(event.uid)
            elif event.data:
                self._workspace_dirty_data.add(event.uid)
            elif event.state:
                self._workspace_dirty_state.add(event.uid)
            if event.added or event.data or event.state:
                dirty_changed = True
                self._set_node_window_modified(event.uid, True)
        if event.removed is not None:
            self._workspace_dirty_removed.append(event.removed)
            self._workspace_structure_modified = True
            dirty_changed = True
        if event.structure is not None:
            self._workspace_structure_reasons.append(event.structure)
            self._workspace_structure_modified = True
            dirty_changed = True
        return dirty_changed

    def _mark_workspace_dirty(
        self,
        *,
        uid: str | None = None,
        data: bool = False,
        state: bool = False,
        added: bool = False,
        removed: str | None = None,
        structure: str | None = None,
    ) -> None:
        if self._workspace_loading_depth > 0 or self._workspace_saving_depth > 0:
            return
        event = _manager_workspace._WorkspaceDirtyEvent(
            generation=self._workspace_dirty_generation + 1,
            uid=uid,
            data=data,
            state=state,
            added=added,
            removed=removed,
            structure=structure,
        )
        if self._apply_workspace_dirty_event(event):
            self._workspace_dirty_generation = event.generation
            self._workspace_dirty_events.append(event)
        self._update_workspace_window_title()

    def _mark_node_added(self, uid: str) -> None:
        self._mark_workspace_dirty(uid=uid, added=True, structure="Added window")

    def _mark_node_data_dirty(self, uid: str) -> None:
        self._mark_workspace_dirty(uid=uid, data=True)

    def _mark_node_state_dirty(self, uid: str) -> None:
        self._mark_workspace_dirty(uid=uid, state=True)

    def _mark_workspace_structure_dirty(self, reason: str) -> None:
        self._mark_workspace_dirty(structure=reason)

    def _mark_workspace_clean(self) -> None:
        self._workspace_structure_modified = False
        self._workspace_dirty_added.clear()
        self._workspace_dirty_data.clear()
        self._workspace_dirty_state.clear()
        self._workspace_dirty_removed.clear()
        self._workspace_structure_reasons.clear()
        self._workspace_dirty_events.clear()
        for uid in tuple(self._all_nodes):
            self._set_node_window_modified(uid, False)
        self._update_workspace_window_title()

    def _restore_workspace_dirty_events(
        self, events: Iterable[_manager_workspace._WorkspaceDirtyEvent]
    ) -> None:
        retained_events = list(events)
        self._mark_workspace_clean()
        for event in retained_events:
            self._apply_workspace_dirty_event(event)
        self._workspace_dirty_events = retained_events
        self._update_workspace_window_title()

    @contextlib.contextmanager
    def _workspace_load_context(self) -> Iterator[None]:
        self._workspace_loading_depth += 1
        try:
            yield
        finally:
            self._workspace_loading_depth -= 1

    def _drain_workspace_deferred_events(self) -> None:
        for _ in range(3):
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents()

    def _workspace_state_snapshot(self) -> dict[str, typing.Any]:
        return {
            "path": self._workspace_path,
            "link_id": self._workspace_link_id,
            "needs_full_save": self._workspace_needs_full_save,
            "node_uid_counter": self._node_uid_counter,
            "structure_modified": self._workspace_structure_modified,
            "dirty_added": frozenset(self._workspace_dirty_added),
            "dirty_data": frozenset(self._workspace_dirty_data),
            "dirty_state": frozenset(self._workspace_dirty_state),
            "dirty_removed": tuple(self._workspace_dirty_removed),
            "structure_reasons": tuple(self._workspace_structure_reasons),
            "dirty_generation": self._workspace_dirty_generation,
            "dirty_events": tuple(self._workspace_dirty_events),
            "delta_save_count": self._workspace_delta_save_count,
            "schema_version": self._workspace_schema_version,
        }

    def _restore_workspace_state_snapshot(
        self, snapshot: dict[str, typing.Any]
    ) -> None:
        self._workspace_path = snapshot["path"]
        self._workspace_link_id = snapshot["link_id"]
        self._workspace_needs_full_save = snapshot["needs_full_save"]
        self._node_uid_counter = snapshot["node_uid_counter"]
        self._workspace_structure_modified = snapshot["structure_modified"]
        self._workspace_dirty_added = set(snapshot["dirty_added"])
        self._workspace_dirty_data = set(snapshot["dirty_data"])
        self._workspace_dirty_state = set(snapshot["dirty_state"])
        self._workspace_dirty_removed = list(snapshot["dirty_removed"])
        self._workspace_structure_reasons = list(snapshot["structure_reasons"])
        self._workspace_dirty_generation = snapshot["dirty_generation"]
        self._workspace_dirty_events = list(snapshot["dirty_events"])
        self._workspace_delta_save_count = snapshot["delta_save_count"]
        self._workspace_schema_version = snapshot["schema_version"]
        if self._workspace_path is not None:
            self._recent_directory = str(self._workspace_path.parent)
        dirty_uids = (
            snapshot["dirty_added"] | snapshot["dirty_data"] | snapshot["dirty_state"]
        )
        for uid in tuple(self._all_nodes):
            self._set_node_window_modified(uid, uid in dirty_uids)
        self._update_workspace_window_title()
        self._refresh_manager_record()

    def _install_workspace_save_shortcut(self, widget: QtWidgets.QWidget) -> None:
        for shortcut in widget.findChildren(QtWidgets.QShortcut):
            if shortcut.objectName() == _WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME:
                return
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.StandardKey.Save, widget)
        shortcut.setObjectName(_WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self.save)

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
        return child_imagetool_uids[0]

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

    def _selected_reload_targets(
        self,
    ) -> tuple[list[int | str], dict[int | str, list[str]]] | None:
        selected_roots = self.tree_view.selected_imagetool_indices
        selected_children = self.tree_view.selected_childtool_uids
        if not selected_roots and not selected_children:
            return None

        reload_targets: list[int | str] = []
        seen_targets: set[int | str] = set()
        child_targets: dict[int | str, list[str]] = {}

        def _add_reload_target(target: int | str) -> None:
            if target in seen_targets:
                return
            seen_targets.add(target)
            reload_targets.append(target)

        for index in selected_roots:
            if not self._node_for_target(index).reloadable:
                return None
            _add_reload_target(index)

        for uid in selected_children:
            try:
                current: _ImageToolWrapper | _ManagedWindowNode = self._child_node(uid)
            except KeyError:
                return None
            if not current.has_source_binding:
                return None

            reload_target: int | str | None = None
            while True:
                try:
                    parent = self._parent_node(current)
                except KeyError:
                    break
                if parent.is_imagetool and parent.reloadable:
                    reload_target = (
                        parent.index
                        if isinstance(parent, _ImageToolWrapper)
                        else parent.uid
                    )
                if isinstance(parent, _ImageToolWrapper):
                    break
                current = parent

            if reload_target is None:
                return None
            _add_reload_target(reload_target)
            child_targets.setdefault(reload_target, []).append(uid)

        return reload_targets, child_targets

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

    def _mark_removed_subtree_dirty(self, uid: str) -> None:
        for node_uid in [uid, *self._iter_descendant_uids(uid)]:
            node = self._all_nodes.get(node_uid)
            if node is not None:
                self._set_node_window_modified(node_uid, False)
                self._mark_workspace_dirty(
                    removed=node.display_text, structure="Removed window"
                )

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
        self._mark_workspace_structure_dirty("Reindexed root windows")

    def get_imagetool(self, index: int | str) -> ImageTool:
        """Get the ImageTool object corresponding to the given index.

        Parameters
        ----------
        index
            Index of the ImageTool window to retrieve.

        Returns
        -------
        ImageTool
            The ImageTool object corresponding to the index.
        """
        node = self._node_for_target(index)
        if not node.is_imagetool:
            raise KeyError(f"Target {index!r} is not an ImageTool")

        tool = node.imagetool
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
        watched_workspace_link_id: str | None = None,
        watched_source_label: str | None = None,
        watched_source_uid: str | None = None,
        watched_connected: bool = True,
        source_input_ndim: int | None = None,
        source_input_dtype: np.dtype[typing.Any] | str | None = None,
        uid: str | None = None,
        provenance_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        index: int | None = None,
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
        if index is None or index in self._imagetool_wrappers:
            index = int(self.next_idx)
        else:
            index = int(index)
        wrapper = _ImageToolWrapper(
            self,
            index,
            self._next_node_uid(uid),
            tool,
            watched_var=watched_var,
            watched_workspace_link_id=watched_workspace_link_id,
            watched_source_label=watched_source_label,
            watched_source_uid=watched_source_uid,
            watched_connected=watched_connected,
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
        self._mark_node_added(wrapper.uid)

        return index

    @QtCore.Slot()
    def _node_info_html(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        return node.info_text

    def _clear_metadata(self) -> None:
        self._metadata_full_code = None
        self._metadata_node_uid = None
        with QtCore.QSignalBlocker(self.metadata_derivation_list):
            self.metadata_derivation_list.clear()
        self._set_metadata_fields([])
        self._update_metadata_pane()

    def _set_metadata_node(self, node: _ImageToolWrapper | _ManagedWindowNode) -> None:
        self._metadata_full_code = node.derivation_code
        self._metadata_node_uid = node.uid
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
                        self.metadata_derivation_list.palette().color(
                            QtGui.QPalette.ColorGroup.Disabled,
                            QtGui.QPalette.ColorRole.Text,
                        )
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
            key_label.setForegroundRole(QtGui.QPalette.ColorRole.Text)
            key_label.setEnabled(False)
            value_label: QtWidgets.QLabel
            if field.details is not None:
                value_label = _ElidedInteractiveLabel(
                    field.value,
                    self.metadata_details_widget,
                )
                value_label.setForegroundRole(QtGui.QPalette.ColorRole.Link)
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

    def _load_source_for_replay(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[str, str] | None:
        current = node
        while True:
            source_name = current.default_load_source_name()
            if source_name is not None:
                load_code = current.load_source_code(assign=source_name)
                if load_code is not None:
                    return source_name, load_code
            if current.parent_uid is None:
                return None
            if current.provenance_spec is None:
                return None
            current = self._parent_node(current)

    def _prompt_replay_input_name(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        data = node._metadata_data()
        candidate = None if data is None else data.name
        if not erlab.interactive.utils._is_kwarg_name(candidate) or candidate in {
            "data",
            "derived",
            "result",
        }:
            candidate = "source_data"

        dialog = QtWidgets.QInputDialog(self)
        dialog.setWindowTitle("Copy Full Code")
        dialog.setLabelText("Source variable name:")
        dialog.setTextValue(typing.cast("str", candidate))
        dialog.setInputMode(QtWidgets.QInputDialog.InputMode.TextInput)
        line_edit = dialog.findChild(QtWidgets.QLineEdit)
        if line_edit is not None:
            line_edit.setValidator(erlab.interactive.utils.IdentifierValidator())
            line_edit.selectAll()

        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return None
        source_name = dialog.textValue().strip()
        if not erlab.interactive.utils._is_kwarg_name(source_name):
            return None
        return source_name

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
        if not self._metadata_full_code:
            return
        code = self._metadata_full_code
        node = (
            None
            if self._metadata_node_uid is None
            else self._all_nodes.get(self._metadata_node_uid)
        )
        provenance = erlab.interactive.imagetool.provenance
        if node is not None and provenance.uses_default_replay_input(code):
            load_source = self._load_source_for_replay(node)
            if load_source is None:
                source_name = self._prompt_replay_input_name(node)
                if source_name is None:
                    return
                code = provenance.rebase_default_replay_input(code, source_name)
            else:
                source_name, load_code = load_source
                rebased_code = provenance.rebase_default_replay_input(code, source_name)
                code = "\n\n".join(part for part in (load_code, rebased_code) if part)
        if code:
            erlab.interactive.utils.copy_to_clipboard(code)

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
        reload_targets = self._selected_reload_targets()

        selection_watched: list[int] = []
        selection_offloadable: list[int | str] = []

        for target in imagetool_targets:
            node = self._node_for_target(target)
            if isinstance(node, _ImageToolWrapper) and node.watched:
                selection_watched.append(node.index)
            if (
                node.imagetool is not None
                and node.is_imagetool
                and not node.slicer_area.data_chunked
            ):
                selection_offloadable.append(target)

        something_selected = bool(imagetool_targets or selection_children)
        root_imagetool_count = len(self.tree_view.selected_imagetool_indices)
        total_selected = len(imagetool_targets) + len(selection_children)
        single_selected = total_selected == 1
        multiple_root_imagetools_selected = (
            root_imagetool_count > 1 and root_imagetool_count == total_selected
        )
        multiple_selected = len(imagetool_targets) > 1

        self.show_action.setEnabled(something_selected)
        self.hide_action.setEnabled(something_selected)
        self.remove_action.setEnabled(something_selected)
        self.rename_action.setEnabled(
            single_selected or multiple_root_imagetools_selected
        )
        self.duplicate_action.setEnabled(something_selected)
        self.promote_action.setEnabled(promotable_child_uid is not None)
        self.offload_action.setEnabled(
            bool(imagetool_targets)
            and len(selection_children) == 0
            and len(selection_offloadable) == len(imagetool_targets)
        )
        self.concat_action.setEnabled(
            multiple_selected and len(selection_children) == 0
        )
        self.store_action.setEnabled(bool(self.tree_view.selected_imagetool_indices))

        self.reload_action.setVisible(reload_targets is not None)
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

        if not imagetool_targets or selection_children:
            self.link_action.setDisabled(True)
            self.unlink_action.setDisabled(True)
            return

        self.link_action.setDisabled(len(imagetool_targets) <= 1)
        is_linked = [
            self.get_imagetool(index).slicer_area.is_linked
            for index in imagetool_targets
        ]
        self.unlink_action.setEnabled(any(is_linked))

        if len(imagetool_targets) > 1 and all(is_linked):
            proxies = [
                self.get_imagetool(index).slicer_area._linking_proxy
                for index in imagetool_targets
            ]
            if all(p == proxies[0] for p in proxies):  # pragma: no branch
                self.link_action.setEnabled(False)

    @QtCore.Slot(int)
    def remove_imagetool(self, index: int, *, update_view: bool = True) -> None:
        """Remove the ImageTool window corresponding to the given index."""
        if index not in self._imagetool_wrappers:
            return
        wrapper = self._imagetool_wrappers[index]
        self._mark_removed_subtree_dirty(wrapper.uid)
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
        action.setObjectName(f"manager_{key}_action")
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
        selected_reload_targets = self._selected_reload_targets()
        if selected_reload_targets is None:
            return

        reload_targets, child_targets = selected_reload_targets
        reloaded_targets: set[int | str] = set()
        for target in reload_targets:
            node = self._node_for_target(target)
            if node.imagetool is not None and node.slicer_area._reload():
                reloaded_targets.add(target)

        for target, child_uids in child_targets.items():
            if target not in reloaded_targets:
                continue
            for uid in child_uids:
                self._refresh_source_chain_to_uid(uid)

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

        with self._workspace_load_context():
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
        self._mark_workspace_structure_dirty("Promoted child ImageTool")
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
        dirty_uids: list[str] = []
        for index in self._selected_imagetool_targets():
            node = self._node_for_target(index)
            slicer_area = self.get_imagetool(index).slicer_area
            if slicer_area.is_linked:
                dirty_uids.append(node.uid)
            slicer_area.unlink()
        for uid in dirty_uids:
            self._mark_node_state_dirty(uid)
        self._sigReloadLinkers.emit()
        if deselect:
            self.tree_view.deselect_all()

    @QtCore.Slot()
    def offload_selected_to_workspace(self) -> None:
        """Replace selected in-memory data with dask-backed workspace data.

        .. versionadded:: 3.23.0
        """
        self.offload_to_workspace(self._selected_imagetool_targets())

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
        if len(indices) <= 1:
            return
        linker = erlab.interactive.imagetool.viewer.SlicerLinkProxy(
            *[self.get_imagetool(t).slicer_area for t in indices],
            link_colors=link_colors,
        )
        self._linkers.append(linker)
        for index in indices:
            self._mark_node_state_dirty(self._node_for_target(index).uid)
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
        if isinstance(node, _ImageToolWrapper) and node.watched:
            watched_metadata = node.watched_metadata()
            ds.attrs["manager_node_watched_varname"] = typing.cast(
                "str", watched_metadata["varname"]
            )
            ds.attrs["manager_node_watched_uid"] = typing.cast(
                "str", watched_metadata["uid"]
            )
            workspace_link_id = watched_metadata.get("workspace_link_id")
            if workspace_link_id is not None:
                ds.attrs["manager_node_watched_workspace_link_id"] = str(
                    workspace_link_id
                )
            source_label = watched_metadata.get("source_label")
            if source_label is not None:
                ds.attrs["manager_node_watched_source_label"] = str(source_label)
            source_uid = watched_metadata.get("source_uid")
            if source_uid is not None:
                ds.attrs["manager_node_watched_source_uid"] = str(source_uid)
            ds.attrs["manager_node_watched_connected"] = bool(
                watched_metadata.get("connected", False)
            )
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
            ds.attrs["itool_title"] = _strip_workspace_modified_placeholder(
                ds.attrs.get("itool_title", "")
            )
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
            ds = tool.to_dataset()
            ds.attrs["tool_title"] = _strip_workspace_modified_placeholder(
                ds.attrs.get("tool_title", "")
            )
            constructor[f"{path}/tool"] = self._annotate_workspace_dataset(
                ds, node, kind="tool"
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
        for index in self._workspace_root_indices():
            self._serialize_workspace_node(
                constructor,
                self._imagetool_wrappers[index],
                str(index),
                include_children=include_children,
            )
            if close:
                self.remove_imagetool(index)
        tree = xr.DataTree.from_dict(constructor)
        _manager_workspace._set_legacy_workspace_schema(tree.attrs)
        return tree

    def _load_workspace_imagetool_dataset(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        node_path: str | None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int | str:
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
        tool_kwargs: dict[str, typing.Any] = {"_in_manager": True}
        if _ITOOL_DATA_NAME in ds and ds[_ITOOL_DATA_NAME].chunks is not None:
            tool_kwargs["auto_compute"] = False
        tool = ImageTool.from_dataset(ds, **tool_kwargs)
        if parent_target is not None:
            target = self.add_imagetool_child(
                tool,
                parent_target,
                show=ds.attrs.get("itool_visible", True),
                **kwargs,
            )
            self._record_workspace_loaded_imagetool_target(
                ds, target, loaded_targets_by_uid
            )
            return target

        kwargs.pop("output_id", None)
        kwargs["source_input_ndim"] = typing.cast(
            "int | None",
            ds.attrs.get("manager_node_source_input_ndim"),
        )
        watched_varname = ds.attrs.get("manager_node_watched_varname")
        watched_uid = ds.attrs.get("manager_node_watched_uid")
        if watched_varname is not None and watched_uid is not None:
            kwargs["watched_var"] = (str(watched_varname), str(watched_uid))
            kwargs["watched_workspace_link_id"] = (
                None
                if ds.attrs.get("manager_node_watched_workspace_link_id") is None
                else str(ds.attrs["manager_node_watched_workspace_link_id"])
            )
            kwargs["watched_source_label"] = (
                None
                if ds.attrs.get("manager_node_watched_source_label") is None
                else str(ds.attrs["manager_node_watched_source_label"])
            )
            kwargs["watched_source_uid"] = (
                None
                if ds.attrs.get("manager_node_watched_source_uid") is None
                else str(ds.attrs["manager_node_watched_source_uid"])
            )
            # Loaded watched rows stay watched, but are disconnected until
            # a notebook explicitly reconnects them.
            kwargs["watched_connected"] = False
        preferred_index: int | None = None
        if node_path is not None and "/" not in node_path:
            with contextlib.suppress(ValueError):
                preferred_index = int(node_path)
            if preferred_index is not None and preferred_index < 0:
                preferred_index = None
        target = self.add_imagetool(
            tool,
            show=ds.attrs.get("itool_visible", True),
            index=preferred_index,
            **kwargs,
        )
        self._record_workspace_loaded_imagetool_target(
            ds, target, loaded_targets_by_uid
        )
        return target

    def _load_workspace_tool_dataset(
        self, ds: xr.Dataset, *, parent_target: int | str | None
    ) -> int | str:
        if parent_target is None:
            raise ValueError("Workspace tool node has no parent")
        return self.add_childtool(
            erlab.interactive.utils.ToolWindow.from_dataset(ds),
            parent_target,
            show=ds.attrs.get("tool_visible", True),
            uid=ds.attrs.get("manager_node_uid"),
        )

    @staticmethod
    def _workspace_saved_uid_from_dataset(ds: xr.Dataset) -> str | None:
        uid = ds.attrs.get("manager_node_uid")
        if isinstance(uid, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                uid = uid.decode()
        if isinstance(uid, str) and uid:
            return uid
        return None

    def _record_workspace_loaded_imagetool_target(
        self,
        ds: xr.Dataset,
        target: int | str,
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> None:
        if loaded_targets_by_uid is None:
            return
        saved_uid = self._workspace_saved_uid_from_dataset(ds)
        if saved_uid is not None:
            loaded_targets_by_uid[saved_uid] = target

    def _restore_workspace_link_groups(
        self,
        manifest: Mapping[str, typing.Any] | None,
        loaded_targets_by_uid: Mapping[str, int | str],
    ) -> None:
        if manifest is None:
            return
        nodes = manifest.get("nodes", ())
        if not isinstance(nodes, list):
            return

        group_targets: dict[int, list[int | str]] = {}
        group_colors: dict[int, bool] = {}
        invalid_groups: set[int] = set()
        for entry in nodes:
            if not isinstance(entry, dict) or "link_group" not in entry:
                continue
            uid = entry.get("uid")
            link_group = entry.get("link_group")
            link_colors = entry.get("link_colors")
            if (
                not isinstance(uid, str)
                or type(link_group) is not int
                or not isinstance(link_colors, bool)
            ):
                continue
            target = loaded_targets_by_uid.get(uid)
            if target is None:
                continue
            try:
                node = self._node_for_target(target)
            except KeyError:
                continue
            if not node.is_imagetool or node.imagetool is None:
                continue
            current_group_colors = group_colors.get(link_group)
            if current_group_colors is None:
                group_colors[link_group] = link_colors
            elif current_group_colors != link_colors:
                invalid_groups.add(link_group)
                continue
            targets = group_targets.setdefault(link_group, [])
            if target not in targets:
                targets.append(target)

        for link_group in sorted(group_targets):
            if link_group in invalid_groups:
                continue
            targets = [
                target
                for target in group_targets[link_group]
                if not self.get_imagetool(target).slicer_area.is_linked
            ]
            if len(targets) <= 1:
                continue
            self.link_imagetools(
                *targets,
                link_colors=group_colors.get(link_group, True),
            )

    def _load_workspace_node(
        self,
        node_tree: xr.DataTree,
        *,
        parent_target: int | str | None = None,
        selection_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        node_path: str | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int | str:
        if "imagetool" in node_tree:
            ds = None
            if (
                manifest is not None
                and node_path is not None
                and workspace_file_path is not None
            ):
                nodes = manifest.get("nodes", ())
                if isinstance(nodes, list):
                    for entry in nodes:
                        if (
                            isinstance(entry, dict)
                            and entry.get("path") == node_path
                            and entry.get("kind") == "imagetool"
                            and entry.get("data_backing") == "dask"
                        ):
                            opened = _manager_xarray.open_workspace_dataset(
                                workspace_file_path,
                                f"{node_path}/imagetool",
                                chunks={},
                            )
                            try:
                                ds = opened.copy(deep=False)
                            finally:
                                opened.close()
                            break
            if ds is None:
                ds = (
                    typing.cast("xr.DataTree", node_tree["imagetool"])
                    .to_dataset(inherit=False)
                    .load()
                )
            target = self._load_workspace_imagetool_dataset(
                ds,
                parent_target=parent_target,
                node_path=node_path,
                loaded_targets_by_uid=loaded_targets_by_uid,
            )
        elif "tool" in node_tree:
            ds = (
                typing.cast("xr.DataTree", node_tree["tool"])
                .to_dataset(inherit=False)
                .load()
            )
            target = self._load_workspace_tool_dataset(ds, parent_target=parent_target)
        else:
            raise ValueError("Workspace node has no supported window payload")

        if "childtools" in node_tree:
            childtools = typing.cast("xr.DataTree", node_tree["childtools"])
            child_keys: list[str] = []
            if manifest is not None and node_path is not None:
                nodes = manifest.get("nodes", ())
                prefix = f"{node_path}/childtools/"
                if isinstance(nodes, list):
                    for entry in nodes:
                        if not isinstance(entry, dict):
                            continue
                        path = entry.get("path")
                        if not isinstance(path, str) or not path.startswith(prefix):
                            continue
                        child_key = path.removeprefix(prefix)
                        if "/" not in child_key and child_key not in child_keys:
                            child_keys.append(child_key)
            child_keys.extend(
                str(key) for key in childtools if str(key) not in child_keys
            )

            for child_key in child_keys:
                if child_key not in childtools:
                    continue
                child_node = typing.cast("xr.DataTree", childtools[child_key])
                child_item = self._tree_item_child_by_key(selection_item, child_key)
                if (
                    child_item is not None
                    and child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked
                ):
                    continue
                self._load_workspace_node(
                    child_node,
                    parent_target=target,
                    selection_item=child_item,
                    manifest=manifest,
                    workspace_file_path=workspace_file_path,
                    node_path=(
                        None
                        if node_path is None
                        else f"{node_path}/childtools/{child_key}"
                    ),
                    loaded_targets_by_uid=loaded_targets_by_uid,
                )
        return target

    def _load_workspace_roots(
        self,
        tree: xr.DataTree,
        root_keys: Iterable[str],
        *,
        root_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> None:
        for key in root_keys:
            if key not in tree:
                continue
            node = typing.cast("xr.DataTree", tree[key])
            item = self._tree_item_child_by_key(root_item, key)
            if item is None or item.checkState(0) != QtCore.Qt.CheckState.Unchecked:
                self._load_workspace_node(
                    node,
                    selection_item=item,
                    manifest=manifest,
                    workspace_file_path=workspace_file_path,
                    node_path=key,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                )

    def _from_h5py_workspace_file(
        self,
        fname: str | os.PathLike[str],
        manifest: Mapping[str, typing.Any],
        *,
        replace: bool,
        mark_dirty: bool,
    ) -> bool:
        nodes = manifest.get("nodes", ())
        root_order = manifest.get("root_order", ())
        if not isinstance(nodes, list) or not isinstance(root_order, list):
            raise TypeError("Workspace manifest is missing node ordering")

        entries_by_path: dict[str, dict[str, typing.Any]] = {}
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            kind = entry.get("kind")
            if not isinstance(path, str) or kind not in {"imagetool", "tool"}:
                continue
            entries_by_path[path] = entry
        if not entries_by_path:
            raise ValueError("Workspace manifest has no loadable nodes")

        root_paths: list[str] = []
        for root in root_order:
            path = str(root)
            if path in entries_by_path and "/" not in path and path not in root_paths:
                root_paths.append(path)
        root_paths.extend(
            path
            for path in entries_by_path
            if "/" not in path and path not in root_paths
        )
        if not root_paths:
            raise ValueError("Workspace manifest has no root ImageTool nodes")

        loaded_targets_by_uid: dict[str, int | str] = {}
        child_paths: dict[str, list[str]] = {path: [] for path in entries_by_path}
        for path in entries_by_path:
            if "/childtools/" not in path:
                continue
            parent_path, child_key = path.rsplit("/childtools/", maxsplit=1)
            if "/" not in child_key and parent_path in entries_by_path:
                child_paths[parent_path].append(path)

        def _load_xarray_dataset(
            payload_path: str, *, chunks: typing.Any, load: bool
        ) -> xr.Dataset:
            opened = _manager_xarray.open_workspace_dataset(
                fname, payload_path, chunks=chunks
            )
            try:
                if load:
                    return opened.load()
                return opened.copy(deep=False)
            finally:
                opened.close()

        def _load_dataset(
            payload_path: str, *, entry: Mapping[str, typing.Any], imagetool: bool
        ) -> xr.Dataset:
            if imagetool and entry.get("data_backing") == "dask":
                return _load_xarray_dataset(payload_path, chunks={}, load=False)
            if imagetool:
                try:
                    ds = _manager_workspace._read_workspace_dataset_group_h5py(
                        fname,
                        payload_path,
                        preferred_data_name=_ITOOL_DATA_NAME,
                    )
                except Exception:
                    logger.debug(
                        "Failed h5py workspace payload read for %s",
                        payload_path,
                        exc_info=True,
                    )
                else:
                    if ds is not None:
                        return ds

            return _load_xarray_dataset(payload_path, chunks=None, load=True)

        def _load_path(path: str, parent_target: int | str | None = None) -> int | str:
            entry = entries_by_path[path]
            kind = typing.cast("str", entry["kind"])
            is_imagetool = kind == "imagetool"
            payload_path = f"{path}/{'imagetool' if is_imagetool else 'tool'}"
            ds = _load_dataset(payload_path, entry=entry, imagetool=is_imagetool)
            if is_imagetool:
                target = self._load_workspace_imagetool_dataset(
                    ds,
                    parent_target=parent_target,
                    node_path=path,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                )
            else:
                target = self._load_workspace_tool_dataset(
                    ds, parent_target=parent_target
                )
            for child_path in child_paths[path]:
                _load_path(child_path, target)
            return target

        if replace:
            manifest_workspace_link_id = manifest.get("workspace_link_id")
            self._workspace_link_id = (
                str(manifest_workspace_link_id)
                if manifest_workspace_link_id
                else uuid.uuid4().hex
            )

        maybe_guard = (
            self._workspace_load_context()
            if not mark_dirty
            else contextlib.nullcontext()
        )
        backup_tree: xr.DataTree | None = None
        backup_snapshot: dict[str, typing.Any] | None = None
        with (
            maybe_guard,
            erlab.interactive.utils.wait_dialog(self, "Loading workspace..."),
        ):
            try:
                if replace:
                    backup_snapshot = self._workspace_state_snapshot()
                    backup_tree = self._to_datatree()
                    self.remove_all_tools()
                for root_path in root_paths:
                    _load_path(root_path)
                self._restore_workspace_link_groups(manifest, loaded_targets_by_uid)
                if not mark_dirty:
                    self._drain_workspace_deferred_events()
            except Exception:
                if backup_tree is not None and backup_snapshot is not None:
                    try:
                        self._restore_replaced_workspace(backup_tree, backup_snapshot)
                    except Exception:
                        logger.exception("Failed to restore previous workspace")
                raise
            finally:
                if backup_tree is not None:
                    backup_tree.close()
        return True

    def _restore_replaced_workspace(
        self,
        backup_tree: xr.DataTree,
        snapshot: dict[str, typing.Any],
    ) -> None:
        with self._workspace_load_context():
            self.remove_all_tools()
            self._load_workspace_roots(backup_tree, [str(key) for key in backup_tree])
            self._drain_workspace_deferred_events()
        self._restore_workspace_state_snapshot(snapshot)

    def _from_datatree(
        self,
        tree: xr.DataTree,
        *,
        replace: bool = False,
        mark_dirty: bool = True,
        select: bool = True,
        workspace_file_path: str | os.PathLike[str] | None = None,
    ) -> bool:
        """Restore the state of the manager from a DataTree object."""
        opened_tree = tree
        try:
            if not self._is_datatree_workspace(tree):
                raise ValueError("Not a valid workspace file")

            schema_version, _delta_save_count, manifest = (
                _manager_workspace._workspace_file_metadata_from_attrs(tree.attrs)
            )
            match schema_version:
                case 1:
                    tree = self._parse_datatree_compat_v1(tree)
                case 2:
                    tree = self._parse_datatree_compat_v2(tree)
                case 3:
                    pass
                case 4:
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported workspace schema version {schema_version}, "
                        "file may be from a newer version of erlab"
                    )
            if replace:
                manifest_workspace_link_id = (
                    None if manifest is None else manifest.get("workspace_link_id")
                )
                self._workspace_link_id = (
                    str(manifest_workspace_link_id)
                    if manifest_workspace_link_id
                    else uuid.uuid4().hex
                )

            root_keys = _manager_workspace._workspace_root_keys(tree, manifest)

            dialog: _ChooseFromDataTreeDialog | None = None
            if select:
                dialog = _ChooseFromDataTreeDialog(
                    self, tree, mode="load", root_keys=root_keys
                )
                if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    return False

            maybe_guard = (
                self._workspace_load_context()
                if not mark_dirty
                else contextlib.nullcontext()
            )
            backup_tree: xr.DataTree | None = None
            backup_snapshot: dict[str, typing.Any] | None = None
            loaded_targets_by_uid: dict[str, int | str] = {}
            with (
                maybe_guard,
                erlab.interactive.utils.wait_dialog(self, "Loading workspace..."),
            ):
                try:
                    if replace:
                        backup_snapshot = self._workspace_state_snapshot()
                        backup_tree = self._to_datatree()
                        self.remove_all_tools()
                    root_item = (
                        None
                        if dialog is None
                        else dialog._tree_widget.invisibleRootItem()
                    )
                    self._load_workspace_roots(
                        tree,
                        root_keys,
                        root_item=root_item,
                        manifest=manifest,
                        workspace_file_path=workspace_file_path,
                        loaded_targets_by_uid=loaded_targets_by_uid,
                    )
                    self._restore_workspace_link_groups(manifest, loaded_targets_by_uid)
                    if not mark_dirty:
                        self._drain_workspace_deferred_events()
                except Exception:
                    if backup_tree is not None and backup_snapshot is not None:
                        try:
                            self._restore_replaced_workspace(
                                backup_tree, backup_snapshot
                            )
                        except Exception:
                            logger.exception("Failed to restore previous workspace")
                    raise
                finally:
                    if backup_tree is not None:
                        backup_tree.close()
            return True
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

    def _workspace_node_path(self, uid: str) -> str:
        node = self._all_nodes[uid]
        if isinstance(node, _ImageToolWrapper):
            return str(node.index)
        if node.parent_uid is None:
            raise KeyError(f"Node {uid!r} has no parent")
        return f"{self._workspace_node_path(node.parent_uid)}/childtools/{uid}"

    def _workspace_payload_path(self, uid: str) -> str:
        node = self._all_nodes[uid]
        payload_name = "imagetool" if node.is_imagetool else "tool"
        return f"{self._workspace_node_path(uid)}/{payload_name}"

    def _workspace_root_indices(self) -> tuple[int, ...]:
        displayed = [
            idx for idx in self._displayed_indices if idx in self._imagetool_wrappers
        ]
        remaining = [
            idx
            for idx in self._imagetool_wrappers
            if idx not in self._displayed_indices
        ]
        return (*displayed, *remaining)

    def _workspace_link_metadata_by_uid(self) -> dict[str, tuple[int, bool]]:
        metadata: dict[str, tuple[int, bool]] = {}
        group_index = 0
        for linker in self._linkers:
            linked_nodes: list[_ImageToolWrapper | _ManagedWindowNode] = []
            for slicer_area in linker.children:
                node = self.node_from_slicer_area(slicer_area)
                if (
                    node is None
                    or not node.is_imagetool
                    or node.imagetool is None
                    or node.slicer_area._linking_proxy is not linker
                ):
                    continue
                linked_nodes.append(node)
            if len(linked_nodes) <= 1:
                continue
            for node in linked_nodes:
                metadata[node.uid] = (group_index, bool(linker.link_colors))
            group_index += 1
        return metadata

    def _workspace_node_manifest_entries(self) -> list[dict[str, typing.Any]]:
        entries: list[dict[str, typing.Any]] = []
        link_metadata = self._workspace_link_metadata_by_uid()

        def _append(uid: str) -> None:
            node = self._all_nodes[uid]
            entry: dict[str, typing.Any] = {
                "uid": uid,
                "path": self._workspace_node_path(uid),
                "kind": "imagetool" if node.is_imagetool else "tool",
                "parent_uid": node.parent_uid,
                "display_name": node.display_text,
            }
            if node.is_imagetool and node.imagetool is not None:
                data = node.slicer_area._data
                if data.chunks is not None:
                    entry["data_backing"] = "dask"
                elif _manager_xarray.dataarray_is_file_backed(data):
                    entry["data_backing"] = "file_lazy"
                else:
                    entry["data_backing"] = "memory"
                link_info = link_metadata.get(uid)
                if link_info is not None:
                    entry["link_group"], entry["link_colors"] = link_info
            entries.append(entry)
            for child_uid in node._childtool_indices:
                if child_uid in self._all_nodes:
                    _append(child_uid)

        for index in self._workspace_root_indices():
            _append(self._imagetool_wrappers[index].uid)
        return entries

    @staticmethod
    def _tree_item_child_by_key(
        item: QtWidgets.QTreeWidgetItem | None, key: str
    ) -> QtWidgets.QTreeWidgetItem | None:
        if item is None:
            return None
        for i in range(item.childCount()):
            child = item.child(i)
            if (
                child is not None
                and child.data(0, QtCore.Qt.ItemDataRole.UserRole) == key
            ):
                return child
        return None

    def _workspace_root_attrs_payload(
        self, *, delta_save_count: int | None = None
    ) -> dict[str, typing.Any]:
        if delta_save_count is None:
            delta_save_count = self._workspace_delta_save_count
        return _manager_workspace._workspace_root_attrs_payload(
            root_order=self._workspace_root_indices(),
            nodes=self._workspace_node_manifest_entries(),
            delta_save_count=delta_save_count,
            erlab_version=str(erlab.__version__),
            workspace_link_id=self._workspace_link_id,
        )

    def _write_full_workspace_file(self, fname: str | os.PathLike[str]) -> None:
        tree: xr.DataTree = self._to_datatree()
        copy_source, copy_groups = self._workspace_full_save_copy_groups(tree)
        try:
            _manager_workspace._write_full_workspace_tree_file(
                fname,
                tree,
                self._workspace_root_attrs_payload(delta_save_count=0),
                copy_source=copy_source,
                copy_groups=copy_groups,
            )
        finally:
            tree.close()

    def _workspace_highest_dirty_data_roots(self) -> list[str]:
        dirty_existing = [
            uid for uid in self._workspace_dirty_data if uid in self._all_nodes
        ]
        dirty_set = set(dirty_existing)
        roots: list[str] = []
        for uid in sorted(
            dirty_existing, key=lambda value: self._workspace_node_path(value)
        ):
            node = self._all_nodes[uid]
            parent_uid = node.parent_uid
            has_dirty_ancestor = False
            while parent_uid is not None:
                if parent_uid in dirty_set:
                    has_dirty_ancestor = True
                    break
                parent_uid = self._all_nodes[parent_uid].parent_uid
            if not has_dirty_ancestor:
                roots.append(uid)
        return roots

    def _save_workspace_delta(self, fname: str | os.PathLike[str]) -> None:
        delta_save_count = self._workspace_delta_save_count + 1
        snapshot = self._workspace_delta_save_snapshot(
            self._workspace_dirty_generation,
            self._workspace_root_attrs_payload(delta_save_count=delta_save_count),
            delta_save_count,
        )
        try:
            _manager_workspace._write_workspace_transaction_file(
                fname,
                snapshot.rewrite_groups,
                snapshot.attr_updates,
                snapshot.root_attrs,
            )
        finally:
            snapshot.close()

    def _save_workspace_document(
        self,
        fname: str | os.PathLike[str],
        *,
        force_full: bool = False,
        document_access: _WorkspaceDocumentAccess | None = None,
    ) -> None:
        if document_access is None:
            with self._workspace_document_access_context(fname) as access:
                self._save_workspace_document(
                    access.path,
                    force_full=force_full,
                    document_access=access,
                )
            return

        fname = document_access.path
        self._workspace_saving_depth += 1
        try:
            _manager_workspace._recover_workspace_transactions(fname)
            requires_full_save = force_full or self._workspace_requires_full_save(fname)
            if requires_full_save:
                self._write_full_workspace_file(fname)
                self._workspace_delta_save_count = 0
                self._workspace_schema_version = (
                    _manager_workspace._current_workspace_schema_version()
                )
            else:
                self._save_workspace_delta(fname)
                self._workspace_delta_save_count += 1
        finally:
            self._workspace_saving_depth -= 1
        self._workspace_needs_full_save = False
        self._mark_workspace_clean()

    def _workspace_save_dialog(
        self,
        *,
        native: bool = True,
        caption: str = "Save Workspace",
        selected_file: str | os.PathLike[str] | None = None,
    ) -> str | None:
        dialog = QtWidgets.QFileDialog(self, caption)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("ImageTool Workspace Files (*.itws)")
        dialog.setDefaultSuffix("itws")
        if selected_file is not None:
            dialog.selectFile(str(selected_file))
        elif self._workspace_path is not None:
            dialog.selectFile(str(self._workspace_path))
        elif self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return None
        return dialog.selectedFiles()[0]

    def _confirm_save_dirty_workspace(self, action_text: str) -> bool:
        if not self.is_workspace_modified:
            return True

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Save changes to this workspace?")
        msg_box.setInformativeText(action_text)
        details = self._dirty_details_text()
        if details:
            msg_box.setDetailedText(details)
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Save)
        result = msg_box.exec()
        if result == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save()
        return result == QtWidgets.QMessageBox.StandardButton.Discard

    def _show_legacy_workspace_upgrade_message(
        self, fname: str | os.PathLike[str]
    ) -> None:
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Save Legacy Workspace")
        msg_box.setText("This workspace uses an older file format.")
        msg_box.setInformativeText(
            "Save it again so ImageTool Manager can read it properly."
        )
        msg_box.setDetailedText(str(pathlib.Path(fname)))
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def _save_legacy_workspace_as_v4(
        self,
        fname: str | os.PathLike[str],
        *,
        native: bool = True,
        existing_access: _WorkspaceDocumentAccess | None = None,
    ) -> tuple[str, QtCore.QLockFile | None] | None:
        self._show_legacy_workspace_upgrade_message(fname)
        converted_fname = self._workspace_save_dialog(
            native=native,
            caption="Save Converted Workspace",
            selected_file=fname,
        )
        if converted_fname is None:
            return None
        converted_path = pathlib.Path(converted_fname).resolve()
        if existing_access is not None and converted_path == existing_access.path:
            with erlab.interactive.utils.wait_dialog(self, "Saving workspace..."):
                self._save_workspace_document(
                    existing_access.path,
                    force_full=True,
                    document_access=existing_access,
                )
            return str(existing_access.path), existing_access.take_lock()

        with self._workspace_document_access_context(converted_fname) as access:
            with erlab.interactive.utils.wait_dialog(self, "Saving workspace..."):
                self._save_workspace_document(
                    access.path,
                    force_full=True,
                    document_access=access,
                )
            return str(access.path), access.take_lock()

    def _associate_loaded_workspace_file(
        self,
        fname: str | os.PathLike[str],
        schema_version: int,
        *,
        native: bool = True,
        delta_save_count: int = 0,
        workspace_access: _WorkspaceDocumentAccess | None = None,
        rebind_data: bool = True,
    ) -> None:
        associated_fname = fname
        associated_lock: QtCore.QLockFile | None = None
        if _manager_workspace._workspace_schema_requires_conversion(schema_version):
            converted = self._save_legacy_workspace_as_v4(
                fname, native=native, existing_access=workspace_access
            )
            if converted is None:
                self._set_workspace_path(None)
                self._workspace_needs_full_save = True
                self._mark_workspace_structure_dirty(
                    "Legacy workspace needs conversion"
                )
                return
            associated_fname, associated_lock = converted
            delta_save_count = 0
            schema_version = _manager_workspace._current_workspace_schema_version()
        elif workspace_access is not None:
            associated_lock = workspace_access.take_lock()

        self._set_workspace_path(associated_fname, workspace_lock=associated_lock)
        self._workspace_delta_save_count = delta_save_count
        self._workspace_schema_version = schema_version
        self._workspace_needs_full_save = (
            _manager_workspace._workspace_schema_requires_full_save(schema_version)
        )
        if rebind_data:
            self._rebind_workspace_backed_imagetools(associated_fname)
        self._drain_workspace_deferred_events()
        self._mark_workspace_clean()
        self._record_recent_workspace(associated_fname)

    def _workspace_rebind_data_for_uid(
        self,
        fname: str | os.PathLike[str],
        uid: str,
        *,
        chunks: typing.Any,
    ) -> xr.DataArray:
        ds = _manager_xarray.open_workspace_dataset(
            fname, self._workspace_payload_path(uid), chunks=chunks
        )
        try:
            data_name: Hashable
            if _ITOOL_DATA_NAME in ds.data_vars:
                data_name = _ITOOL_DATA_NAME
            else:
                data_name = next(iter(ds.data_vars))
            name = ds.attrs.get("itool_name", "")
            name = None if name == "" else name
            return ds[data_name].rename(name).copy(deep=False)
        finally:
            ds.close()

    def _workspace_data_backing_snapshot(
        self,
    ) -> dict[str, tuple[str, tuple[str, ...]]]:
        snapshot: dict[str, tuple[str, tuple[str, ...]]] = {}
        for node in self._all_nodes.values():
            if not node.is_imagetool or node.imagetool is None:
                continue
            data = node.slicer_area._data
            if data.chunks is not None:
                kind = "dask"
            elif _manager_xarray.dataarray_is_file_backed(data):
                kind = "file_lazy"
            else:
                kind = "memory"
            snapshot[node.uid] = (kind, _manager_xarray.dataarray_source_paths(data))
        return snapshot

    def _rebind_workspace_backed_imagetools(
        self,
        fname: str | os.PathLike[str],
        *,
        targets: Iterable[int | str] | None = None,
        chunks: typing.Any = _WORKSPACE_REBIND_KEEP_CHUNKS,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
        old_workspace_path: str | os.PathLike[str] | None = None,
    ) -> None:
        pending: list[
            tuple[
                _ManagedWindowNode,
                typing.Any,
                str,
                typing.Any,
            ]
        ] = []
        if targets is None:
            nodes = [
                node
                for node in self._all_nodes.values()
                if node.is_imagetool and node.imagetool is not None
            ]
        else:
            nodes = []
            for target in targets:
                node = self._node_for_target(target)
                if node.is_imagetool and node.imagetool is not None:
                    nodes.append(node)
        for node in sorted(nodes, key=lambda node: self._workspace_node_path(node.uid)):
            tool = node.imagetool
            if tool is None:
                continue
            slicer_area = tool.slicer_area
            rebind_chunks: typing.Any
            if backing_snapshot is not None:
                backing = backing_snapshot.get(node.uid)
                if backing is None:
                    continue
                kind, source_paths = backing
                if kind == "memory":
                    continue
                if kind == "file_lazy":
                    old_path = _manager_xarray._normalized_file_path(old_workspace_path)
                    if old_path is None or old_path not in source_paths:
                        continue
                rebind_chunks = {} if kind == "dask" else None
            else:
                rebind_chunks = chunks
                if rebind_chunks is _WORKSPACE_REBIND_KEEP_CHUNKS:
                    rebind_chunks = {} if slicer_area._data.chunks is not None else None
            pending.append(
                (
                    node,
                    copy.deepcopy(slicer_area.state),
                    node.name,
                    rebind_chunks,
                )
            )
        if not pending:
            return
        with self._workspace_load_context():
            for node, state, name, chunks in pending:
                tool = node.imagetool
                if tool is None:
                    continue
                slicer_area = tool.slicer_area
                data = self._workspace_rebind_data_for_uid(
                    fname, node.uid, chunks=chunks
                )
                slicer_area.set_data(data, auto_compute=False)
                slicer_area.state = state
                node._set_name(name, manual=False)

    def offload_to_workspace(
        self, targets: Iterable[int | str], *, native: bool = True
    ) -> bool:
        """Replace selected in-memory ImageTools with dask-backed workspace data.

        .. versionadded:: 3.23.0
        """
        offload_targets: list[int | str] = []
        for target in targets:
            node = self._node_for_target(target)
            if (
                node.is_imagetool
                and node.imagetool is not None
                and not node.slicer_area.data_chunked
            ):
                offload_targets.append(target)
        if not offload_targets:
            return False

        if self._workspace_path is None:
            if not self.save_as(native=native):
                return False
        elif (
            self.is_workspace_modified or self._workspace_needs_full_save
        ) and not self.save(native=native):
            return False

        if self._workspace_path is None:
            return False

        origin = self._active_managed_window()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self, "Offloading to workspace..."
            ):
                self._rebind_workspace_backed_imagetools(
                    self._workspace_path,
                    targets=offload_targets,
                    chunks={},
                )
                _manager_workspace._write_workspace_root_attrs_to_file(
                    self._workspace_path,
                    self._workspace_root_attrs_payload(
                        delta_save_count=self._workspace_delta_save_count
                    ),
                )
            self._status_bar.showMessage("Data offloaded to workspace", 5000)
        except Exception:
            self._show_operation_error(
                "Error while offloading to workspace",
                "An error occurred while reconnecting data from the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            return False

        self._restore_focus_after_workspace_save(origin)
        self._update_actions()
        self._update_info()
        return True

    def _workspace_requires_full_save(self, fname: str | os.PathLike[str]) -> bool:
        return _manager_workspace._workspace_requires_full_save(
            fname,
            needs_full_save=self._workspace_needs_full_save,
            schema_version=self._workspace_schema_version,
            structure_modified=self._workspace_structure_modified,
            has_dirty_added=bool(self._workspace_dirty_added),
            has_dirty_removed=bool(self._workspace_dirty_removed),
        )

    def _workspace_rewrite_group_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, xr.Dataset]]:
        constructor: dict[str, xr.Dataset] = {}
        node = self._all_nodes[uid]
        node_path = self._workspace_node_path(uid)
        self._serialize_workspace_node(
            constructor, node, node_path, include_children=True
        )
        return node_path, constructor

    def _workspace_attr_update_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]] | None:
        constructor: dict[str, xr.Dataset] = {}
        node = self._all_nodes[uid]
        node_path = self._workspace_node_path(uid)
        self._serialize_workspace_node(
            constructor, node, node_path, include_children=False
        )
        payload_path = self._workspace_payload_path(uid)
        ds = constructor.get(payload_path)
        if ds is None:
            return None
        return payload_path, dict(ds.attrs), (node_path, constructor)

    def _workspace_delta_save_snapshot(
        self,
        generation: int,
        root_attrs: dict[str, typing.Any],
        delta_save_count: int,
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        rewrite_groups: list[tuple[str, dict[str, xr.Dataset]]] = []
        rewritten_uids: set[str] = set()
        for uid in self._workspace_highest_dirty_data_roots():
            rewrite_groups.append(self._workspace_rewrite_group_snapshot(uid))
            rewritten_uids.add(uid)
            rewritten_uids.update(self._iter_descendant_uids(uid))

        attr_updates: list[
            tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
        ] = []
        for uid in sorted(self._workspace_dirty_state - rewritten_uids):
            if uid not in self._all_nodes:
                continue
            update = self._workspace_attr_update_snapshot(uid)
            if update is not None:
                attr_updates.append(update)

        return _manager_workspace._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=delta_save_count,
            rewrite_groups=tuple(rewrite_groups),
            attr_updates=tuple(attr_updates),
        )

    def _workspace_save_snapshot(
        self, fname: str | os.PathLike[str]
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        self._drain_workspace_deferred_events()
        generation = self._workspace_dirty_generation
        self._workspace_saving_depth += 1
        try:
            if self._workspace_requires_full_save(fname):
                return self._workspace_full_save_snapshot(generation)
            delta_save_count = self._workspace_delta_save_count + 1
            root_attrs = self._workspace_root_attrs_payload(
                delta_save_count=delta_save_count
            )
            return self._workspace_delta_save_snapshot(
                generation, root_attrs, delta_save_count
            )
        finally:
            self._workspace_saving_depth -= 1

    def _workspace_full_save_snapshot(
        self, generation: int
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        tree = self._to_datatree()
        copy_source, copy_groups = self._workspace_full_save_copy_groups(tree)
        return _manager_workspace._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=self._workspace_root_attrs_payload(delta_save_count=0),
            delta_save_count=0,
            full_tree=tree,
            copy_source=copy_source,
            copy_groups=copy_groups,
        )

    def _workspace_full_save_copy_groups(
        self, tree: xr.DataTree
    ) -> tuple[str | None, tuple[tuple[str, str, dict[str, typing.Any] | None], ...]]:
        if self._workspace_path is None:
            return None, ()
        workspace_path = pathlib.Path(self._workspace_path)
        if (
            self._workspace_schema_version
            != _manager_workspace._current_workspace_schema_version()
            or not workspace_path.exists()
        ):
            return None, ()

        try:
            root_attrs = _manager_workspace._read_workspace_root_attrs_h5py(
                workspace_path
            )
        except Exception:
            return None, ()
        schema_version, _delta_save_count, manifest = (
            _manager_workspace._workspace_file_metadata_from_attrs(root_attrs)
        )
        if (
            schema_version != _manager_workspace._current_workspace_schema_version()
            or manifest is None
        ):
            return None, ()

        identities: dict[tuple[str, str], str] = {}
        nodes = manifest.get("nodes", ())
        if isinstance(nodes, list):
            for entry in nodes:
                if not isinstance(entry, dict):
                    continue
                uid = entry.get("uid")
                kind = entry.get("kind")
                path = entry.get("path")
                if (
                    isinstance(uid, str)
                    and isinstance(kind, str)
                    and kind in {"imagetool", "tool"}
                    and isinstance(path, str)
                ):
                    identities[(uid, kind)] = f"{path}/{kind}"
        copy_groups: list[tuple[str, str, dict[str, typing.Any] | None]] = []
        for uid, node in self._all_nodes.items():
            if uid in self._workspace_dirty_data or uid in self._workspace_dirty_added:
                continue
            if not node.is_imagetool:
                tool = node.tool_window
                if tool is None or not tool.can_save_and_load():
                    continue
            kind = "imagetool" if node.is_imagetool else "tool"
            source_path = identities.get((uid, kind))
            if source_path is None:
                continue
            payload_path = self._workspace_payload_path(uid)
            try:
                payload_tree = typing.cast("xr.DataTree", tree[payload_path])
            except KeyError:
                continue
            attrs = None
            if uid in self._workspace_dirty_state or source_path != payload_path:
                attrs = dict(payload_tree.to_dataset(inherit=False).attrs)
            copy_groups.append((source_path, payload_path, attrs))
        return str(workspace_path), tuple(copy_groups)

    def _open_workspace_save_wait_dialog(
        self,
        parent: QtWidgets.QWidget,
        *,
        title: str = "Saving Workspace",
        label_text: str = "Saving workspace...",
    ) -> QtWidgets.QDialog:
        dialog = QtWidgets.QProgressDialog(label_text, "", 0, 0, parent)
        dialog.setCancelButton(None)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        return dialog

    def _set_workspace_save_actions_enabled(
        self, enabled: bool
    ) -> tuple[bool, bool, bool]:
        previous = (
            self.save_action.isEnabled(),
            self.save_as_action.isEnabled(),
            self.compact_workspace_action.isEnabled(),
        )
        self.save_action.setEnabled(enabled and previous[0])
        self.save_as_action.setEnabled(enabled and previous[1])
        self.compact_workspace_action.setEnabled(enabled and previous[2])
        return previous

    def _restore_workspace_save_actions_enabled(
        self, previous: tuple[bool, bool, bool]
    ) -> None:
        self.save_action.setEnabled(previous[0])
        self.save_as_action.setEnabled(previous[1])
        self.compact_workspace_action.setEnabled(previous[2])

    def _run_workspace_save_worker(
        self,
        fname: str | os.PathLike[str],
        snapshot: _manager_workspace._WorkspaceSaveSnapshot,
        origin: QtWidgets.QWidget | None,
        *,
        wait_dialog_title: str = "Saving Workspace",
        wait_dialog_text: str = "Saving workspace...",
    ) -> tuple[bool, float, str]:
        loop = QtCore.QEventLoop(self)
        result: dict[str, typing.Any] = {"ok": False, "elapsed": 0.0, "error": ""}
        receiver = _manager_workspace._WorkspaceSaveResultReceiver(loop, result, self)
        worker = _manager_workspace._WorkspaceSaveWorker(fname, snapshot)
        wait_dialog: QtWidgets.QDialog | None = None
        wait_timer = QtCore.QTimer(self)
        wait_timer.setSingleShot(True)

        def _show_wait_dialog() -> None:
            nonlocal wait_dialog
            if wait_dialog is None and self._workspace_save_in_progress:
                wait_dialog = self._open_workspace_save_wait_dialog(
                    origin or self,
                    title=wait_dialog_title,
                    label_text=wait_dialog_text,
                )

        wait_timer.timeout.connect(_show_wait_dialog)
        worker.signals.finished.connect(receiver.finish)
        self._workspace_save_in_progress = True
        previous_action_states = self._set_workspace_save_actions_enabled(False)
        try:
            wait_timer.start(
                max(0, int(_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS * 1000))
            )
            thread_pool = QtCore.QThreadPool.globalInstance()
            if thread_pool is None:
                raise RuntimeError("Qt thread pool is unavailable")
            thread_pool.start(worker)
            loop.exec()
        finally:
            self._workspace_save_in_progress = False
            wait_timer.stop()
            wait_timer.deleteLater()
            if wait_dialog is not None:
                wait_dialog.close()
                wait_dialog.deleteLater()
            receiver.deleteLater()
            self._restore_workspace_save_actions_enabled(previous_action_states)
        return (
            typing.cast("bool", result["ok"]),
            typing.cast("float", result["elapsed"]),
            typing.cast("str", result["error"]),
        )

    @QtCore.Slot()
    def save(self, *, native: bool = True) -> bool:
        """Save the current workspace document.

        Parameters
        ----------
        native
            Whether to use the native file dialog, by default `True`. This option is
            used when testing the application to ensure reproducibility.
        """
        if self._workspace_path is None:
            return self.save_as(native=native)
        if self._workspace_save_in_progress:
            self._status_bar.showMessage("Workspace save already in progress", 3000)
            return False
        origin = self._active_managed_window()
        old_workspace_path = self._workspace_path
        backing_snapshot = self._workspace_data_backing_snapshot()
        self._status_bar.showMessage("Saving workspace...")
        try:
            snapshot = self._workspace_save_snapshot(self._workspace_path)
            try:
                ok, elapsed, error_text = self._run_workspace_save_worker(
                    self._workspace_path, snapshot, origin
                )
            except Exception:
                snapshot.close()
                raise
        except Exception:
            self._status_bar.clearMessage()
            self._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            return False
        if not ok:
            self._status_bar.clearMessage()
            self._show_workspace_save_worker_error(error_text)
            self._restore_focus_after_workspace_save(origin)
            return False

        self._workspace_needs_full_save = False
        self._workspace_delta_save_count = snapshot.delta_save_count
        if snapshot.full_tree is not None:
            self._workspace_schema_version = (
                _manager_workspace._current_workspace_schema_version()
            )
            self._rebind_workspace_backed_imagetools(
                self._workspace_path,
                backing_snapshot=backing_snapshot,
                old_workspace_path=old_workspace_path,
            )
        self._drain_workspace_deferred_events()
        post_save_events = tuple(
            event
            for event in self._workspace_dirty_events
            if event.generation > snapshot.generation
        )
        if post_save_events:
            self._restore_workspace_dirty_events(post_save_events)
            message = "Workspace saved; new changes remain unsaved"
        else:
            self._mark_workspace_clean()
            message = (
                f"Workspace saved in {elapsed:.1f} s"
                if elapsed >= _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS
                else "Workspace saved"
            )
        self._status_bar.showMessage(message, 5000)
        self._restore_focus_after_workspace_save(origin)
        self._record_recent_workspace(self._workspace_path)
        return True

    @QtCore.Slot()
    def save_as(self, *, native: bool = True) -> bool:
        """Save the current workspace under a new path and bind to that path."""
        origin = self._active_managed_window()
        fname = self._workspace_save_dialog(native=native, caption="Save Workspace As")
        if fname is None:
            return False
        old_workspace_path = self._workspace_path
        backing_snapshot = self._workspace_data_backing_snapshot()
        try:
            dialog_parent = origin or self
            with self._workspace_document_access_context(fname) as access:
                with erlab.interactive.utils.wait_dialog(
                    dialog_parent, "Saving workspace..."
                ):
                    self._save_workspace_document(
                        access.path,
                        force_full=True,
                        document_access=access,
                    )
                    self._rebind_workspace_backed_imagetools(
                        access.path,
                        backing_snapshot=backing_snapshot,
                        old_workspace_path=old_workspace_path,
                    )
                self._set_workspace_path(access.path, workspace_lock=access.take_lock())
            self._workspace_needs_full_save = False
            self._drain_workspace_deferred_events()
            self._mark_workspace_clean()
            self._record_recent_workspace(access.path)
        except Exception:
            self._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            return False
        self._restore_focus_after_workspace_save(origin)
        return True

    def _compact_workspace_before_shutdown(self) -> None:
        if (
            self._workspace_path is None
            or self._workspace_delta_save_count <= 0
            or self.is_workspace_modified
            or self._workspace_save_in_progress
            or self._workspace_loading_depth > 0
        ):
            return
        try:
            logger.debug("Compacting workspace before shutdown...")
            self._drain_workspace_deferred_events()
            generation = self._workspace_dirty_generation
            self._workspace_saving_depth += 1
            try:
                snapshot = self._workspace_full_save_snapshot(generation)
            finally:
                self._workspace_saving_depth -= 1
            try:
                ok, _, error_text = self._run_workspace_save_worker(
                    self._workspace_path,
                    snapshot,
                    self,
                    wait_dialog_title="Optimizing Workspace",
                    wait_dialog_text="Optimizing workspace file…",
                )
            except Exception:
                snapshot.close()
                raise
            if not ok:
                logger.error(
                    "Failed to compact workspace before shutdown%s",
                    f":\n{error_text}" if error_text else "",
                    extra={"suppress_ui_alert": True},
                )
                return
            self._workspace_needs_full_save = False
            self._workspace_delta_save_count = 0
            self._workspace_schema_version = (
                _manager_workspace._current_workspace_schema_version()
            )
            self._drain_workspace_deferred_events()
            post_save_events = tuple(
                event
                for event in self._workspace_dirty_events
                if event.generation > snapshot.generation
            )
            if post_save_events:
                self._restore_workspace_dirty_events(post_save_events)
            else:
                self._mark_workspace_clean()
        except Exception:
            logger.exception(
                "Failed to compact workspace before shutdown",
                extra={"suppress_ui_alert": True},
            )

    @QtCore.Slot()
    def compact_workspace(self) -> bool:
        """Rewrite the current workspace file to remove unused space."""
        if self._workspace_path is None:
            return self.save_as()
        if self._workspace_save_in_progress:
            self._status_bar.showMessage("Workspace save already in progress", 3000)
            return False

        origin = self._active_managed_window()
        old_workspace_path = self._workspace_path
        backing_snapshot = self._workspace_data_backing_snapshot()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self, "Compacting workspace..."
            ):
                self._save_workspace_document(self._workspace_path, force_full=True)
                self._rebind_workspace_backed_imagetools(
                    self._workspace_path,
                    backing_snapshot=backing_snapshot,
                    old_workspace_path=old_workspace_path,
                )
            self._workspace_delta_save_count = 0
            self._status_bar.showMessage("Workspace compacted", 5000)
        except Exception:
            self._show_operation_error(
                "Error while compacting workspace",
                "An error occurred while compacting the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            return False

        self._restore_focus_after_workspace_save(origin)
        return True

    def _save_to_file(self, fname: str):
        """Export a selected subset of the workspace to ``fname``.

        This helper preserves the older selection-dialog behavior used by tests and
        private callers. Document-style Save and Save As use
        :meth:`_save_workspace_document` instead.
        """
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
                tree.to_netcdf(
                    fname,
                    engine="h5netcdf",
                    invalid_netcdf=True,
                    encoding=_manager_xarray.workspace_datatree_encoding(tree),
                )
        finally:
            tree.close()

    def _load_workspace_file(
        self,
        fname: str | os.PathLike[str],
        *,
        replace: bool,
        associate: bool,
        mark_dirty: bool,
        select: bool,
        native: bool = True,
    ) -> bool:
        with self._workspace_document_access_context(fname) as access:
            _manager_workspace._recover_workspace_transactions(access.path)
            if not select and replace:
                try:
                    root_attrs = _manager_workspace._read_workspace_root_attrs_h5py(
                        access.path
                    )
                    schema_version, delta_save_count, manifest = (
                        _manager_workspace._workspace_file_metadata_from_attrs(
                            root_attrs
                        )
                    )
                    if (
                        schema_version
                        == _manager_workspace._current_workspace_schema_version()
                        and manifest is not None
                    ):
                        loaded = self._from_h5py_workspace_file(
                            access.path,
                            manifest,
                            replace=replace,
                            mark_dirty=mark_dirty,
                        )
                        if loaded and associate:
                            self._associate_loaded_workspace_file(
                                access.path,
                                schema_version,
                                native=native,
                                delta_save_count=delta_save_count,
                                workspace_access=access,
                                rebind_data=False,
                            )
                        return loaded
                except Exception:
                    logger.debug(
                        "Failed h5py workspace load path; falling back to DataTree",
                        exc_info=True,
                    )
            tree = _manager_xarray.open_workspace_datatree(access.path, chunks=None)
            schema_version, delta_save_count, _manifest = (
                _manager_workspace._workspace_file_metadata_from_attrs(tree.attrs)
            )
            loaded = self._from_datatree(
                tree,
                replace=replace,
                mark_dirty=mark_dirty,
                select=select,
                workspace_file_path=access.path,
            )
            if loaded and associate:
                self._associate_loaded_workspace_file(
                    access.path,
                    schema_version,
                    native=native,
                    delta_save_count=delta_save_count,
                    workspace_access=access,
                    rebind_data=False,
                )
            return loaded

    @QtCore.Slot()
    def load(self, *, native: bool = True) -> bool:
        """Replace this manager with a workspace file."""
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

        if not dialog.exec():
            return False

        fname = dialog.selectedFiles()[0]
        if not self._confirm_save_dirty_workspace(
            "Opening a workspace replaces the windows currently in this manager."
        ):
            return False
        self._recent_directory = os.path.dirname(fname)
        try:
            return self._load_workspace_file(
                fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
                native=native,
            )
        except Exception as exc:
            if _manager_workspace._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    fname,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self, fname)
            else:
                logger.exception(
                    "Error while loading workspace",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self,
                    "Error",
                    "An error occurred while loading the workspace file.",
                )
            return False

    @QtCore.Slot()
    def import_workspace(self, *, native: bool = True) -> bool:
        """Import selected windows from another workspace file."""
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

        if not dialog.exec():
            return False
        fname = dialog.selectedFiles()[0]
        self._recent_directory = os.path.dirname(fname)
        try:
            loaded = self._load_workspace_file(
                fname,
                replace=False,
                associate=False,
                mark_dirty=True,
                select=True,
            )
        except Exception as exc:
            if _manager_workspace._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    fname,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self, fname)
            else:
                logger.exception(
                    "Error while importing workspace",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self,
                    "Error",
                    "An error occurred while importing the workspace file.",
                )
            return False
        else:
            if loaded:
                self._record_recent_workspace(fname)
            return loaded

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
        watched_metadata: Mapping[str, typing.Any] | None = None,
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
        watched_metadata = dict(watched_metadata or {})
        if watched_var is not None:
            watched_metadata.setdefault("workspace_link_id", self._workspace_link_id)

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
                        watched_workspace_link_id=typing.cast(
                            "str | None",
                            watched_metadata.get("workspace_link_id"),
                        ),
                        watched_source_label=typing.cast(
                            "str | None", watched_metadata.get("source_label")
                        ),
                        watched_source_uid=typing.cast(
                            "str | None", watched_metadata.get("source_uid")
                        ),
                        watched_connected=bool(
                            watched_metadata.get("connected", watched_var is not None)
                        ),
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

    def _watched_source_color_key(self, wrapper: _ImageToolWrapper) -> str:
        if wrapper._watched_source_uid:
            return f"source-uid:{wrapper._watched_source_uid}"
        if wrapper._watched_source_label:
            return f"source-label:{wrapper._watched_source_label}"
        watched_uid = typing.cast("str", wrapper._watched_uid)
        watched_varname = typing.cast("str", wrapper._watched_varname)
        return f"legacy:{watched_uid.removeprefix(f'{watched_varname} ')}"

    def color_for_watched_var_source(self, wrapper: _ImageToolWrapper) -> QtGui.QColor:
        """Return a different color for different watched-variable sources."""
        source_key = self._watched_source_color_key(wrapper)
        all_source_keys = tuple(
            dict.fromkeys(
                self._watched_source_color_key(v)
                for v in self._imagetool_wrappers.values()
                if v.watched
            )
        )
        idx = all_source_keys.index(source_key)
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

    @QtCore.Slot(str, str, object, object)
    def _data_watched_update(
        self,
        varname: str,
        uid: str,
        darr: xr.DataArray,
        watched_metadata: Mapping[str, typing.Any] | None = None,
    ) -> None:
        """Update ImageTool window corresponding to the given watched variable."""
        watched_metadata = dict(watched_metadata or {})
        idx = self._find_watched_idx(uid)
        if idx is None:
            # If the tool does not exist, create a new one
            self._data_recv(
                [darr],
                {},
                watched_var=(varname, uid),
                watched_metadata={
                    **dict(watched_metadata),
                    "connected": True,
                },
            )
        else:
            # Update data in the existing tool
            wrapper = self._imagetool_wrappers[idx]
            wrapper.set_watched_binding(
                varname,
                uid,
                workspace_link_id=typing.cast(
                    "str | None",
                    watched_metadata.get("workspace_link_id")
                    or wrapper._watched_workspace_link_id,
                ),
                source_label=typing.cast(
                    "str | None",
                    watched_metadata.get("source_label")
                    or wrapper._watched_source_label,
                ),
                source_uid=typing.cast(
                    "str | None",
                    watched_metadata.get("source_uid") or wrapper._watched_source_uid,
                ),
                connected=True,
            )
            wrapper.set_source_input_ndim(darr.ndim)
            wrapper.set_source_input_dtype(darr.dtype)
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

    def _watch_info(self) -> dict[str, typing.Any]:
        """Return watched-variable metadata used by notebook reconnect workflows."""
        return {
            "workspace_link_id": self._workspace_link_id,
            "watched": [
                wrapper.watched_metadata()
                for wrapper in self._imagetool_wrappers.values()
                if wrapper.watched
            ],
        }

    @QtCore.Slot()
    def _send_watch_info(self) -> None:
        """Send watched-variable metadata to the manager server."""
        self._sigReplyData.emit(self._watch_info())

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
        metadata_from_attrs = _manager_workspace._workspace_file_metadata_from_attrs

        if try_workspace:
            for p in list(queued):
                try:
                    dt = _manager_xarray.open_workspace_datatree(p, chunks=None)
                except Exception as exc:
                    if _manager_workspace._is_workspace_file_lock_error(exc):
                        logger.info(
                            "Workspace file is already open or locked: %s",
                            p,
                            extra={"suppress_ui_alert": True},
                        )
                        _show_workspace_file_lock_error(self, p)
                        queued.remove(p)
                    else:
                        logger.debug("Failed to open %s as datatree workspace", p)
                else:
                    if self._is_datatree_workspace(dt):
                        dt.close()
                        try:
                            with self._workspace_document_access_context(p) as access:
                                _manager_workspace._recover_workspace_transactions(
                                    access.path
                                )
                                workspace_dt = _manager_xarray.open_workspace_datatree(
                                    access.path, chunks=None
                                )
                                workspace_dt_owned = True
                                try:
                                    (
                                        schema_version,
                                        delta_save_count,
                                        _manifest,
                                    ) = metadata_from_attrs(workspace_dt.attrs)
                                    if not self._confirm_save_dirty_workspace(
                                        "Opening a workspace replaces the windows "
                                        "currently in this manager."
                                    ):
                                        return
                                    loaded_workspace = self._from_datatree(
                                        workspace_dt,
                                        replace=True,
                                        mark_dirty=False,
                                        select=False,
                                        workspace_file_path=access.path,
                                    )
                                    workspace_dt_owned = False
                                    if loaded_workspace:
                                        self._associate_loaded_workspace_file(
                                            access.path,
                                            schema_version,
                                            delta_save_count=delta_save_count,
                                            workspace_access=access,
                                            rebind_data=False,
                                        )
                                finally:
                                    if workspace_dt_owned:
                                        workspace_dt.close()
                        except Exception as exc:
                            if _manager_workspace._is_workspace_file_lock_error(exc):
                                logger.info(
                                    "Workspace file is already open or locked: %s",
                                    p,
                                    extra={"suppress_ui_alert": True},
                                )
                                _show_workspace_file_lock_error(self, p)
                            else:
                                logger.exception(
                                    "Error while loading workspace",
                                    extra={"suppress_ui_alert": True},
                                )
                                erlab.interactive.utils.MessageDialog.critical(
                                    self,
                                    "Error",
                                    "An error occurred while loading the workspace "
                                    "file.",
                                )
                        finally:
                            queued.remove(p)
                            loaded.append(p)
                    else:
                        dt.close()

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

    def _show_workspace_save_worker_error(self, error_text: str) -> None:
        logger.error(
            "Error while saving workspace\n%s",
            error_text,
            extra={"suppress_ui_alert": True},
        )
        erlab.interactive.utils.MessageDialog.critical(
            self,
            "Error",
            "An error occurred while saving the workspace file.",
            detailed_text=erlab.interactive.utils._format_traceback(error_text),
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
        ImageTool that opened them is removed.

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
        self._mark_node_added(node.uid)
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
        self._mark_node_added(node.uid)
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
        self._mark_removed_subtree_dirty(uid)
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

    def _start_server_pair(
        self, *, port: int, watch_port: int
    ) -> tuple[_ManagerServer, _WatcherServer, int, int]:
        server = _ManagerServer(port=port)
        server.sigReceived.connect(self._data_recv)
        server.sigLoadRequested.connect(self._data_load)
        server.sigReplaceRequested.connect(self._data_replace)
        server.sigDataRequested.connect(self._send_imagetool_data)
        server.sigWatchInfoRequested.connect(self._send_watch_info)
        self._sigReplyData.connect(server.set_return_value)
        server.sigRemoveIndex.connect(self.remove_imagetool)
        server.sigShowIndex.connect(self.show_imagetool)
        server.sigRemoveUID.connect(self._remove_watched)
        server.sigShowUID.connect(self._show_watched)
        server.sigUnwatchUID.connect(self._data_unwatch)
        server.sigWatchedVarChanged.connect(self._data_watched_update)
        watcher_server = _WatcherServer(port=watch_port)
        self._sigWatchedDataEdited.connect(watcher_server.send_parameters)
        try:
            server.start()
            bound_port = server.wait_until_bound()
            watcher_server.start()
            bound_watch_port = watcher_server.wait_until_bound()
        except Exception:
            with contextlib.suppress(TypeError, RuntimeError):
                self._sigReplyData.disconnect(server.set_return_value)
            with contextlib.suppress(TypeError, RuntimeError):
                self._sigWatchedDataEdited.disconnect(watcher_server.send_parameters)
            server.stop()
            watcher_server.stop()
            server.deleteLater()
            watcher_server.deleteLater()
            raise
        return server, watcher_server, bound_port, bound_watch_port

    def _start_manager_servers(
        self,
    ) -> tuple[_ManagerServer, _WatcherServer, int, int]:
        legacy_ports = (_manager_server.PORT, _manager_server.PORT_WATCH)
        if self.manager_index == 0 and legacy_ports != (0, 0):
            try:
                return self._start_server_pair(
                    port=legacy_ports[0], watch_port=legacy_ports[1]
                )
            except Exception:
                logger.info(
                    "ImageTool manager legacy ports are unavailable; "
                    "using dynamic ports instead."
                )

        return self._start_server_pair(port=0, watch_port=0)

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
    def open_new_manager_instance(self) -> None:
        """Open another ImageTool Manager window."""
        try:
            _launch_new_manager_instance()
        except Exception:
            logger.exception("Failed to open a new ImageTool Manager window")
            erlab.interactive.utils.MessageDialog.critical(
                self,
                title="New Manager Window",
                text="Could not open another ImageTool Manager window.",
            )

    @QtCore.Slot()
    def check_for_updates(self) -> None:
        from erlab.interactive.imagetool.manager._updater_gui import AutoUpdater

        updater = AutoUpdater()
        updater.check_for_updates(self)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        """Handle proper termination of resources before closing the application."""
        logger.debug("Closing ImageTool Manager...")
        if not self._confirm_save_dirty_workspace(
            "Closing this manager will discard unsaved workspace changes."
        ):
            self._application_quit_requested = False
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

        self._compact_workspace_before_shutdown()

        logger.debug("Removing all ImageTool windows...")
        with self._workspace_load_context():
            self.remove_all_tools()

        logger.debug("Closing additional windows...")
        for widget in dict(self._additional_windows).values():
            widget.close()
            widget.deleteLater()

        logger.debug("Removing event filters...")
        qapp = QtWidgets.QApplication.instance()
        if (
            isinstance(qapp, QtWidgets.QApplication)
            and self._application_quit_filter is not None
        ):
            qapp.removeEventFilter(self._application_quit_filter)
            self._application_quit_filter = None
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

        logger.debug("Releasing workspace lock...")
        self._release_workspace_lock()

        logger.debug("Stopping servers...")
        self._registry_heartbeat_timer.stop()
        self._stop_servers()
        unregister_manager_record(self._manager_record.internal_id)

        logger.debug("Closing dask client (if any)...")
        self._dask_menu.close_client()

        root_logger = logging.getLogger()
        if self._warning_handler in root_logger.handlers:  # pragma: no branch
            root_logger.removeHandler(self._warning_handler)

        self._clear_all_alerts()

        if sys.excepthook == self._handle_uncaught_exception:
            sys.excepthook = self._previous_excepthook

        super().closeEvent(event)
