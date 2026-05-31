from __future__ import annotations

import contextlib
import datetime
import logging
import os
import pathlib
import platform
import subprocess
import sys
import traceback
import typing
from dataclasses import dataclass

import numpy as np
import pyqtgraph
import qtpy
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive.imagetool.manager import _server as _manager_server
from erlab.interactive.imagetool.manager import _workspace as _manager_workspace
from erlab.interactive.imagetool.manager._logging import get_log_file_path
from erlab.interactive.imagetool.manager._server import _ManagerServer, _WatcherServer

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab.interactive.imagetool import provenance
    from erlab.interactive.imagetool._load_source import _LoadSourceDetails
    from erlab.interactive.imagetool.manager import ImageToolManager

logger = logging.getLogger(__name__)

_METADATA_DERIVATION_CODE_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
_METADATA_DERIVATION_COPYABLE_ROLE = _METADATA_DERIVATION_CODE_ROLE + 1
_QWIDGETSIZE_MAX = 16777215
_RECENT_WORKSPACES_SETTINGS_KEY = "recent_workspaces"
_MAX_RECENT_WORKSPACES = 10
_DEPENDENCY_STATUS_LABELS: dict[str, str] = {
    "current": "Current",
    "changed": "Changed",
    "missing": "Missing",
}
_DEPENDENCY_STATUS_BADGES: dict[str, str] = {
    "changed": "Changed",
    "missing": "Missing",
}
_DEPENDENCY_STATUS_TOOLTIPS: dict[str, str] = {
    "current": "All recorded live inputs are still open and unchanged.",
    "changed": "At least one recorded live input has changed since this data was made.",
    "missing": "At least one recorded live input is no longer open.",
}


@dataclass(frozen=True)
class _ScriptRebuildResult:
    data: xr.DataArray
    provenance_spec: provenance.ToolProvenanceSpec


class _ScriptRebuildError(RuntimeError):
    def __init__(self, message: str, *, details: str = "") -> None:
        super().__init__(message)
        self.details = details


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


class _HeightForWidthFrame(QtWidgets.QFrame):
    def sync_height_for_width(self) -> None:
        if self.isVisible() and self.hasHeightForWidth() and self.width() > 0:
            height = self.heightForWidth(self.width())
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)
        else:
            self.setMinimumHeight(0)
            self.setMaximumHeight(_QWIDGETSIZE_MAX)
        self.updateGeometry()

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        if self.hasHeightForWidth() and self.width() > 0:
            hint.setHeight(self.heightForWidth(self.width()))
        return hint

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self.sync_height_for_width()


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
        if not erlab.interactive.utils.qt_is_valid(self._emitter):
            return
        with contextlib.suppress(RuntimeError):
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
            event.accept()
            self._manager.close()
            return True
        if (
            event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.matches(QtGui.QKeySequence.StandardKey.Quit)
        ):
            event.accept()
            self._manager.close()
            return True
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


class _WidgetsController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager

    def about(self) -> None:
        """Show the about dialog."""
        import h5netcdf
        import xarray_lmfit

        msg_box = self._manager._make_icon_msgbox()

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
        msg_box = self._manager._make_icon_msgbox()
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

    def open_log_directory(self) -> None:
        """Open the log directory in the system file explorer."""
        erlab.utils.misc.open_in_file_manager(get_log_file_path().parent)

    def _parse_progressbar(self, message: str):
        """Parse and display a progress bar from a log message."""
        title_part, _, info_part = message.split("|", 2)
        title: str = title_part.split(":", 1)[0].strip()
        current: int = int(info_part.split("/", 1)[0].strip())
        total: int = int(info_part.split("/", 1)[1].split("[", 1)[0].strip())

        if total in self._manager._progress_bars:
            pbar = self._manager._progress_bars[total]
        else:
            pbar = QtWidgets.QProgressDialog(self._manager)
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
            self._manager._progress_bars[total] = pbar
            pbar.show()
            pbar.raise_()
            pbar.activateWindow()

        pbar.setValue(current)

    def _show_alert(
        self, levelname: str, levelno: int, message: str, formatted_traceback: str
    ) -> None:
        """Show a non-intrusive warning message in a floating window."""
        if _check_message_is_progressbar(message):
            try:
                self._manager._parse_progressbar(message)
            except Exception:  # pragma: no cover
                logger.exception("Failed to parse progress bar message %r", message)
            else:
                return

        if message in self._manager._ignored_warning_messages:
            return

        if levelno >= logging.ERROR:
            icon_pixmap = QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical
        elif levelno >= logging.WARNING:
            icon_pixmap = QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
        else:
            icon_pixmap = QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation

        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
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
            lambda *, msg=message: self._manager._ignore_warning_message(msg)
        )
        ignore_btn.setDefault(False)
        ignore_btn.setAutoDefault(False)
        dialog._button_box.addButton(
            ignore_btn, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )

        dismiss_btn = QtWidgets.QPushButton("Dismiss All", dialog)
        dismiss_btn.setObjectName("warningDismissAllButton")
        dismiss_btn.clicked.connect(self._manager._clear_all_alerts)
        dismiss_btn.setDefault(False)
        dismiss_btn.setAutoDefault(False)
        dialog._button_box.addButton(
            dismiss_btn, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self._manager._alert_dialogs.append(dialog)
        dialog.finished.connect(lambda *, d=dialog: self._manager._unregister_alert(d))

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ignore_warning_message(self, message: str) -> None:
        """Ignore future warnings with the same message for this session."""
        self._manager._ignored_warning_messages.add(message)
        for notification in list(self._manager._alert_dialogs):
            if notification.text() == message:
                notification.close()

    def _unregister_alert(self, alert: erlab.interactive.utils.MessageDialog) -> None:
        with contextlib.suppress(ValueError):  # pragma: no cover - defensive cleanup
            self._manager._alert_dialogs.remove(alert)

    def _clear_all_alerts(self) -> None:
        for notification in list(self._manager._alert_dialogs):
            notification.close()

    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Show a dialog for uncaught exceptions and log them."""
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.error(
                "An unexpected error occurred.",
                exc_info=(exc_type, exc_value, exc_traceback),
            )
        if self._manager._previous_excepthook is not None:  # pragma: no branch
            with contextlib.suppress(Exception):
                self._manager._previous_excepthook(exc_type, exc_value, exc_traceback)

    def _start_server_pair(
        self, *, port: int, watch_port: int
    ) -> tuple[_ManagerServer, _WatcherServer, int, int]:
        server = _ManagerServer(port=port)
        server.sigReceived.connect(self._manager._data_recv)
        server.sigLoadRequested.connect(self._manager._data_load)
        server.sigReplaceRequested.connect(self._manager._data_replace)
        server.sigDataRequested.connect(self._manager._send_imagetool_data)
        server.sigWatchInfoRequested.connect(self._manager._send_watch_info)
        self._manager._sigReplyData.connect(server.set_return_value)
        server.sigRemoveIndex.connect(self._manager.remove_imagetool)
        server.sigShowIndex.connect(self._manager.show_imagetool)
        server.sigRemoveUID.connect(self._manager._remove_watched)
        server.sigShowUID.connect(self._manager._show_watched)
        server.sigUnwatchUID.connect(self._manager._data_unwatch)
        server.sigWatchedVarChanged.connect(self._manager._data_watched_update)
        watcher_server = _WatcherServer(port=watch_port)
        self._manager._sigWatchedDataEdited.connect(watcher_server.send_parameters)
        try:
            server.start()
            bound_port = server.wait_until_bound()
            watcher_server.start()
            bound_watch_port = watcher_server.wait_until_bound()
        except Exception:
            with contextlib.suppress(TypeError, RuntimeError):
                self._manager._sigReplyData.disconnect(server.set_return_value)
            with contextlib.suppress(TypeError, RuntimeError):
                self._manager._sigWatchedDataEdited.disconnect(
                    watcher_server.send_parameters
                )
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
        if self._manager.manager_index == 0 and legacy_ports != (0, 0):
            try:
                return self._manager._start_server_pair(
                    port=legacy_ports[0], watch_port=legacy_ports[1]
                )
            except Exception:
                logger.info(
                    "ImageTool manager legacy ports are unavailable; "
                    "using dynamic ports instead."
                )

        return self._manager._start_server_pair(port=0, watch_port=0)

    def _stop_servers(self) -> None:
        """Stop the server thread properly."""
        if self._manager.server.isRunning():  # pragma: no branch
            self._manager.server.stop()
        if self._manager.watcher_server.isRunning():  # pragma: no branch
            self._manager.watcher_server.stop()

    def open_settings(self) -> None:
        """Open the settings dialog for the ImageTool manager."""
        dialog = erlab.interactive._options.OptionDialog(self._manager)
        dialog.exec()

    def open_new_manager_instance(self) -> None:
        """Open another ImageTool Manager window."""
        try:
            _launch_new_manager_instance()
        except Exception:
            logger.exception("Failed to open a new ImageTool Manager window")
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                title="New Manager Window",
                text="Could not open another ImageTool Manager window.",
            )

    def check_for_updates(self) -> None:
        from erlab.interactive.imagetool.manager._updater_gui import AutoUpdater

        updater = AutoUpdater()
        updater.check_for_updates(self._manager)


__all__ = [
    name for name in globals() if name.startswith("_") and not name.startswith("__")
]
