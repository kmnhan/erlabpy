from __future__ import annotations

import contextlib
import datetime
import itertools
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
import erlab.interactive.colors
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
_METADATA_DERIVATION_ROW_ROLE = _METADATA_DERIVATION_CODE_ROLE + 2
_METADATA_DERIVATION_ACTIVATABLE_ROLE = _METADATA_DERIVATION_CODE_ROLE + 3
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


class _TrustedScriptReplayCancelled(RuntimeError):
    """Raised when the user declines to execute trusted replay code."""


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


class _MetadataDerivationTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, text: str = "") -> None:
        super().__init__([text])

    def text(self, column: int = 0) -> str:
        return super().text(column)

    def setText(self, *args: typing.Any) -> None:
        if len(args) == 1:
            super().setText(0, args[0])
            return
        if len(args) == 2:
            super().setText(args[0], args[1])
            return
        raise TypeError("setText() takes 1 or 2 arguments")

    def data(self, *args: typing.Any) -> typing.Any:
        if len(args) == 1:
            return super().data(0, args[0])
        if len(args) == 2:
            return super().data(args[0], args[1])
        raise TypeError("data() takes 1 or 2 arguments")

    def setData(self, *args: typing.Any) -> None:
        if len(args) == 2:
            super().setData(0, args[0], args[1])
            return
        if len(args) == 3:
            super().setData(args[0], args[1], args[2])
            return
        raise TypeError("setData() takes 2 or 3 arguments")

    def toolTip(self, column: int = 0) -> str:
        return super().toolTip(column)

    def setToolTip(self, *args: typing.Any) -> None:
        if len(args) == 1:
            super().setToolTip(0, args[0])
            return
        if len(args) == 2:
            super().setToolTip(args[0], args[1])
            return
        raise TypeError("setToolTip() takes 1 or 2 arguments")

    def foreground(self, column: int = 0) -> QtGui.QBrush:
        return super().foreground(column)

    def setForeground(self, *args: typing.Any) -> None:
        if len(args) == 1:
            super().setForeground(0, QtGui.QBrush(args[0]))
            return
        if len(args) == 2:
            super().setForeground(args[0], QtGui.QBrush(args[1]))
            return
        raise TypeError("setForeground() takes 1 or 2 arguments")


class _MetadataDerivationListWidget(QtWidgets.QTreeWidget):
    copy_requested = QtCore.Signal()
    paste_requested = QtCore.Signal()
    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setItemsExpandable(True)
        self.setExpandsOnDoubleClick(False)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def _flattened_items(self) -> list[QtWidgets.QTreeWidgetItem]:
        items: list[QtWidgets.QTreeWidgetItem] = []

        def collect(item: QtWidgets.QTreeWidgetItem) -> None:
            items.append(item)
            for child_index in range(item.childCount()):
                child = item.child(child_index)
                # Valid child indices return items; guard for Qt binding safety.
                if child is not None:  # pragma: no branch
                    collect(child)

        for row in range(self.topLevelItemCount()):
            item = self.topLevelItem(row)
            # Valid top-level indices return items; guard for Qt binding safety.
            if item is not None:  # pragma: no branch
                collect(item)
        return items

    def addItem(self, item: QtWidgets.QTreeWidgetItem) -> None:
        self.addTopLevelItem(item)

    def setUniformItemSizes(self, enabled: bool) -> None:
        self.setUniformRowHeights(enabled)

    def count(self) -> int:
        return len(self._flattened_items())

    def item(self, row: int) -> QtWidgets.QTreeWidgetItem | None:
        items = self._flattened_items()
        if row < 0 or row >= len(items):
            return None
        return items[row]

    def row(self, item: QtWidgets.QTreeWidgetItem) -> int:
        return self.display_order(item)

    def display_order(self, item: QtWidgets.QTreeWidgetItem) -> int:
        try:
            return self._flattened_items().index(item)
        except ValueError:
            return -1

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_requested.emit()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            self.paste_requested.emit()
            event.accept()
            return
        if event.key() in {QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter}:
            item = self.currentItem()
            if item is not None:
                self.itemActivated.emit(item, 0)
                event.accept()
                return
        super().keyPressEvent(event)


class _HeightForWidthFrame(QtWidgets.QFrame):
    def sync_height_for_width(self) -> None:
        if self.isVisible() and self.hasHeightForWidth() and self.width() > 0:
            height = self.heightForWidth(self.width())
            self.setMinimumHeight(height)
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


class _ElidedValueLabel(QtWidgets.QLabel):
    _SIZE_HINT_EMS = 24

    def __init__(
        self,
        text: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        elide_mode: QtCore.Qt.TextElideMode = QtCore.Qt.TextElideMode.ElideMiddle,
    ) -> None:
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = elide_mode
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.set_full_text(text)

    @property
    def full_text(self) -> str:
        return self._full_text

    def setText(self, text: str | None) -> None:
        self.set_full_text("" if text is None else text)

    def set_full_text(self, text: str) -> None:
        if text == self._full_text:
            return
        self._full_text = text
        super().setText(text)
        self.setToolTip(text)
        self.setAccessibleName(text)
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        hint.setWidth(self._stable_hint_width())
        return hint

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        rect = self.contentsRect()
        margin = self.margin()
        if margin > 0:
            rect = rect.adjusted(margin, margin, -margin, -margin)
        text = self.fontMetrics().elidedText(
            self._full_text, self._elide_mode, max(rect.width(), 0)
        )
        style = self.style()
        if style is None:
            style = QtWidgets.QApplication.style()
        if style is None:  # pragma: no cover
            return
        style.drawItemText(
            painter,
            rect,
            int(self.alignment()) | int(QtCore.Qt.TextFlag.TextSingleLine.value),
            self.palette(),
            self.isEnabled(),
            text,
            self.foregroundRole(),
        )

    def _stable_hint_width(self) -> int:
        padding = 2 * (self.margin() + self.frameWidth())
        if self.indent() > 0:
            padding += self.indent()
        return self.fontMetrics().horizontalAdvance("m" * self._SIZE_HINT_EMS) + padding


class _LoadSourceDetailsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        details: _LoadSourceDetails,
        parent: QtWidgets.QWidget | None = None,
        *,
        show_in_data_explorer: Callable[[pathlib.Path], None] | None = None,
        can_edit_file_load: bool = False,
        edit_file_load_tooltip: str = (
            "This source was not recorded as an editable file-load step."
        ),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data Source")
        self.setModal(True)
        self.setMinimumWidth(540)
        self.setSizeGripEnabled(False)
        self.value_labels: dict[str, QtWidgets.QLabel] = {}
        self.reveal_button: QtWidgets.QAbstractButton | None = None
        self.data_explorer_button: QtWidgets.QAbstractButton | None = None
        self.edit_file_load_button: QtWidgets.QAbstractButton | None = None
        self.copy_code_button: QtWidgets.QAbstractButton | None = None
        self.edit_file_load_requested = False
        source_exists = details.path.exists()

        layout = QtWidgets.QVBoxLayout(self)

        header_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(header_layout)

        icon_label = QtWidgets.QLabel(self)
        icon_label.setObjectName("manager_load_source_icon_label")
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
        header_layout.addLayout(title_layout, 1)

        title_label = QtWidgets.QLabel(details.path.name, self)
        title_label.setObjectName("manager_load_source_name_label")
        title_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        title_label.setToolTip(details.path.name)
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + 1)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setWordWrap(True)
        title_layout.addWidget(title_label)

        status_label = QtWidgets.QLabel(
            "Loaded from file" if source_exists else "Source file not found", self
        )
        status_label.setObjectName("manager_load_source_status_label")
        if source_exists:
            status_label.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
        else:
            _mark_missing_source_label(status_label)
        title_layout.addWidget(status_label)

        details_layout = QtWidgets.QGridLayout()
        details_layout.setColumnStretch(1, 1)
        layout.addLayout(details_layout)

        row = self._add_detail(
            details_layout,
            0,
            "path",
            "Path",
            str(details.path),
            elide_mode=QtCore.Qt.TextElideMode.ElideMiddle,
        )
        if not source_exists:
            _mark_missing_source_label(self.value_labels["path"])
            self.edit_file_load_button = QtWidgets.QPushButton(
                "Locate…",
                self,
            )
            self.edit_file_load_button.setObjectName("manager_edit_load_source_button")
            self.edit_file_load_button.setAutoDefault(False)
            self.edit_file_load_button.setEnabled(can_edit_file_load)
            self.edit_file_load_button.setToolTip(edit_file_load_tooltip)
            self.edit_file_load_button.clicked.connect(self._request_file_load_edit)
            details_layout.addWidget(
                self.edit_file_load_button,
                row - 1,
                2,
                alignment=QtCore.Qt.AlignmentFlag.AlignLeft,
            )
        row = self._add_detail(
            details_layout,
            row,
            "loader",
            details.loader_label,
            details.loader_text,
            tool_tip=_load_source_loader_tooltip(details),
            elide_mode=QtCore.Qt.TextElideMode.ElideRight,
        )
        self._add_detail(
            details_layout,
            row,
            "arguments",
            "Load arguments",
            details.kwargs_text,
            wrap=True,
        )

        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setCenterButtons(False)
        if source_exists:
            self.reveal_button = typing.cast(
                "QtWidgets.QAbstractButton",
                self.button_box.addButton(
                    _file_manager_action_text(),
                    QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
                ),
            )
            self.reveal_button.setObjectName("manager_reveal_load_source_path_button")
            self.reveal_button.setToolTip(
                "Reveal this file in the system file manager."
            )

            def reveal_in_file_manager_and_close() -> None:
                erlab.utils.misc.open_in_file_manager(details.path)
                self.accept()

            self.reveal_button.clicked.connect(reveal_in_file_manager_and_close)
            self.data_explorer_button = typing.cast(
                "QtWidgets.QAbstractButton",
                self.button_box.addButton(
                    "Show in Data Explorer",
                    QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
                ),
            )
            self.data_explorer_button.setObjectName(
                "manager_show_load_source_in_explorer_button"
            )
            self.data_explorer_button.setEnabled(show_in_data_explorer is not None)
            self.data_explorer_button.setToolTip(
                "Show this file in Data Explorer."
                if show_in_data_explorer is not None
                else "Data Explorer is unavailable in this context."
            )
            if show_in_data_explorer is not None:

                def show_in_data_explorer_and_close() -> None:
                    show_in_data_explorer(details.path)
                    self.accept()

                self.data_explorer_button.clicked.connect(
                    show_in_data_explorer_and_close
                )
        self.copy_path_button = typing.cast(
            "QtWidgets.QAbstractButton",
            self.button_box.addButton(
                "Copy Path", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        self.copy_path_button.clicked.connect(
            lambda: erlab.interactive.utils.copy_to_clipboard(str(details.path))
        )
        load_code = details.load_code
        if load_code is not None:
            self.copy_code_button = typing.cast(
                "QtWidgets.QAbstractButton",
                self.button_box.addButton(
                    "Copy Load Code",
                    QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
                ),
            )
            self.copy_code_button.setToolTip(
                "Copy a replayable loading snippet for this source."
            )
            self.copy_code_button.clicked.connect(
                lambda _checked=False, code=load_code: (
                    erlab.interactive.utils.copy_to_clipboard(code)
                )
            )
        close_button = typing.cast(
            "QtWidgets.QAbstractButton",
            self.button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close),
        )
        close_button.clicked.connect(self.accept)
        layout.addWidget(self.button_box)
        target_width = max(self.minimumWidth(), self.sizeHint().width())
        self.resize(target_width, self.minimumSizeHint().height())
        target_height = self.height()
        self.setFixedSize(target_width, target_height)

    @QtCore.Slot()
    def _request_file_load_edit(self) -> None:
        self.edit_file_load_requested = True
        self.accept()

    def _add_detail(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        key: str,
        label: str,
        value: str,
        *,
        tool_tip: str | None = None,
        elide_mode: QtCore.Qt.TextElideMode | None = None,
        wrap: bool = False,
    ) -> int:
        key_label = QtWidgets.QLabel(label, self)
        key_label.setObjectName(f"manager_load_source_{key}_label")
        vertical_alignment = (
            QtCore.Qt.AlignmentFlag.AlignTop
            if wrap
            else QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        key_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | vertical_alignment)
        key_label.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)

        if wrap:
            value_label = QtWidgets.QLabel(value, self)
            value_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            value_label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            value_label.setWordWrap(True)
            value_label.setMinimumWidth(0)
        else:
            value_label = _ElidedValueLabel(
                value,
                self,
                elide_mode=(
                    QtCore.Qt.TextElideMode.ElideRight
                    if elide_mode is None
                    else elide_mode
                ),
            )
        value_label.setObjectName(f"manager_load_source_{key}_value_label")
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | vertical_alignment)
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding
            if wrap
            else QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        size_policy.setHeightForWidth(wrap)
        value_label.setSizePolicy(size_policy)
        value_label.setToolTip(value if tool_tip is None else tool_tip)

        layout.addWidget(key_label, row, 0)
        layout.addWidget(value_label, row, 1)
        self.value_labels[key] = value_label
        return row + 1


def _file_manager_action_text() -> str:
    if sys.platform == "darwin":
        return "Reveal in Finder"
    if sys.platform.startswith("win"):
        return "Reveal in File Explorer"
    return "Open Containing Folder"


def _load_source_loader_tooltip(details: _LoadSourceDetails) -> str:
    lines = [f"{details.loader_label}: {details.loader_text}"]
    if details.loader_label == "Loader" and details.loader_text in erlab.io.loaders:
        description = getattr(erlab.io.loaders[details.loader_text], "description", "")
        if description:
            lines.append(str(description))
    return "\n".join(lines)


def _mark_missing_source_label(label: QtWidgets.QLabel) -> None:
    palette = label.palette()
    palette.setColor(
        QtGui.QPalette.ColorRole.WindowText,
        QtGui.QColor(
            "#ffb4ab" if erlab.interactive.colors.is_dark_mode() else "#b3261e"
        ),
    )
    label.setForegroundRole(QtGui.QPalette.ColorRole.WindowText)
    label.setPalette(palette)
    label.setProperty("manager_source_missing", True)


def _workspace_file_manager_action_text() -> str:
    return _file_manager_action_text()


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
            return "HDF5 file (.h5)"
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
        self.setSizeGripEnabled(False)

        self._workspace_path = (
            None if workspace_path is None else pathlib.Path(workspace_path).resolve()
        )
        self.value_labels: dict[str, QtWidgets.QLabel] = {}
        self.copy_path_button: QtWidgets.QAbstractButton | None = None
        self.reveal_button: QtWidgets.QAbstractButton | None = None

        layout = QtWidgets.QVBoxLayout(self)

        header_layout = QtWidgets.QHBoxLayout()
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
        header_layout.addLayout(title_layout, 1)

        title_label = QtWidgets.QLabel(workspace_name, self)
        title_label.setObjectName("manager_workspace_name_label")
        title_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        title_label.setToolTip(workspace_name)
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + 1)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setWordWrap(True)
        title_layout.addWidget(title_label)

        status_label = QtWidgets.QLabel(
            self._status_text(self._workspace_path, state), self
        )
        status_label.setObjectName("manager_workspace_status_label")
        status_label.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
        title_layout.addWidget(status_label)

        details_layout = QtWidgets.QGridLayout()
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
            self.copy_path_button.clicked.connect(self._copy_path)

            self.reveal_button = typing.cast(
                "QtWidgets.QAbstractButton",
                self.button_box.addButton(
                    _workspace_file_manager_action_text(),
                    QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
                ),
            )
            self.reveal_button.setObjectName("manager_reveal_workspace_path_button")
            self.reveal_button.clicked.connect(self._reveal_path)

        close_button = typing.cast(
            "QtWidgets.QAbstractButton",
            self.button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close),
        )
        close_button.clicked.connect(self.accept)
        layout.addWidget(self.button_box)
        target_width = max(self.minimumWidth(), self.sizeHint().width())
        self.resize(target_width, self.minimumSizeHint().height())
        target_height = self.height()
        self.setFixedSize(target_width, target_height)

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

    @QtCore.Slot()
    def _copy_path(self) -> None:
        if self._workspace_path is None:
            return
        erlab.interactive.utils.copy_to_clipboard(str(self._workspace_path))

    @QtCore.Slot()
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

    def _close_manager_for_application_quit(self) -> None:
        erlab.interactive.utils._set_application_quit_requested(True)
        try:
            self._manager.close()
        finally:
            erlab.interactive.utils._set_application_quit_requested(False)

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if event is None:
            return False
        if not erlab.interactive.utils.qt_is_valid(self, self._manager, obj, event):
            return False
        try:
            event_type = event.type()
        except RuntimeError:
            return False
        if event_type == QtCore.QEvent.Type.Quit:
            event.accept()
            self._close_manager_for_application_quit()
            return True
        if (
            event_type == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.matches(QtGui.QKeySequence.StandardKey.Quit)
        ):
            event.accept()
            self._close_manager_for_application_quit()
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
_CURVE_PREVIEW_MAX_POINTS = 4096


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


def _curve_preview_data(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return None
    y = erlab.interactive.imagetool.slicer._display_safe_values(y)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return None
    if x.size <= _CURVE_PREVIEW_MAX_POINTS:
        return x, y

    bucket_count = max(1, _CURVE_PREVIEW_MAX_POINTS // 2)
    edges = np.linspace(0, x.size, bucket_count + 1, dtype=np.intp)
    point_indices: list[int] = []
    for start, stop in itertools.pairwise(edges):
        if stop <= start:
            continue
        local_valid = np.flatnonzero(finite[start:stop])
        if local_valid.size == 0:
            continue
        local_y = y[start + local_valid]
        min_index = int(start) + int(local_valid[int(np.nanargmin(local_y))])
        max_index = int(start) + int(local_valid[int(np.nanargmax(local_y))])
        if min_index == max_index:
            point_indices.append(min_index)
        elif min_index < max_index:
            point_indices.extend((min_index, max_index))
        else:
            point_indices.extend((max_index, min_index))
    if not point_indices:
        return None
    return x[point_indices], y[point_indices]


def _summary_accent_color() -> QtGui.QColor:
    color = QtGui.QColor(
        erlab.interactive.utils._apply_qt_accent_color(
            erlab.utils.formatting._DEFAULT_ACCENT_COLOR
        )
    )
    if color.isValid():
        return color
    return QtWidgets.QApplication.palette().color(QtGui.QPalette.ColorRole.Highlight)


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
    load_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        _scene = QtWidgets.QGraphicsScene(self)
        self.setScene(_scene)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

        self._pixmapitem = typing.cast(
            "QtWidgets.QGraphicsPixmapItem", _scene.addPixmap(QtGui.QPixmap())
        )
        self._curve_axes_item = QtWidgets.QGraphicsPathItem()
        self._curve_item = QtWidgets.QGraphicsPathItem()
        _scene.addItem(self._curve_axes_item)
        _scene.addItem(self._curve_item)
        self._curve_axes_item.hide()
        self._curve_item.hide()
        self._curve_data: tuple[np.ndarray, np.ndarray] | None = None
        self._aspect_ratio_mode = QtCore.Qt.AspectRatioMode.IgnoreAspectRatio
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.setToolTip("Main image preview")
        self._load_button = QtWidgets.QPushButton("Load Data for Preview", self)
        self._load_button.setObjectName("manager_pending_preview_load_button")
        self._load_button.setToolTip(
            "Load this ImageTool's saved data and build its preview."
        )
        self._load_button.clicked.connect(self.load_requested)
        self._load_button.hide()
        self.hide()

    def setPixmap(
        self,
        pixmap: QtGui.QPixmap,
        *,
        aspect_ratio_mode: QtCore.Qt.AspectRatioMode = (
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio
        ),
    ) -> None:
        self._aspect_ratio_mode = aspect_ratio_mode
        self._clear_curve()
        self._load_button.hide()
        if pixmap.isNull():
            self._pixmapitem.setPixmap(QtGui.QPixmap())
            scene = self.scene()
            if scene is not None:
                scene.setSceneRect(QtCore.QRectF())
            self.hide()
            self.updateGeometry()
            return
        self._pixmapitem.setPixmap(pixmap)
        scene = self.scene()
        if scene is not None:
            scene.setSceneRect(self._pixmapitem.boundingRect())
        self._fit_pixmap_in_view()

    def setCurve(self, x_values: np.ndarray, y_values: np.ndarray) -> None:
        curve = _curve_preview_data(x_values, y_values)
        self._pixmapitem.setPixmap(QtGui.QPixmap())
        self._load_button.hide()
        if curve is None:
            self._clear_curve()
            self.hide()
            self.updateGeometry()
            return
        self._curve_data = curve
        self._rebuild_curve_paths()
        super().setVisible(True)
        self.updateGeometry()

    def setLoadPrompt(self) -> None:
        self._pixmapitem.setPixmap(QtGui.QPixmap())
        self._clear_curve()
        self._load_button.adjustSize()
        self._load_button.show()
        self._position_load_button()
        super().setVisible(True)
        self.updateGeometry()

    def clearLoadPrompt(self) -> None:
        self._load_button.hide()

    def setVisible(self, visible: bool) -> None:
        if not visible:
            self.clearLoadPrompt()
        if (
            visible
            and self._pixmapitem.pixmap().isNull()
            and self._load_button.isHidden()
            and self._curve_data is None
        ):
            visible = False
        super().setVisible(visible)

    def show(self) -> None:
        if (
            self._pixmapitem.pixmap().isNull()
            and self._load_button.isHidden()
            and self._curve_data is None
        ):
            return
        super().show()

    def minimumSizeHint(self) -> QtCore.QSize:
        if not self._load_button.isHidden():
            return self._load_prompt_size_hint()
        if self._pixmapitem.pixmap().isNull() and self._curve_data is None:
            return QtCore.QSize(0, 0)
        return super().minimumSizeHint()

    def sizeHint(self) -> QtCore.QSize:
        if not self._load_button.isHidden():
            return self._load_prompt_size_hint()
        if self._pixmapitem.pixmap().isNull() and self._curve_data is None:
            return QtCore.QSize(0, 0)
        return super().sizeHint()

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._position_load_button()
        self._fit_pixmap_in_view()
        self._fit_curve_in_view()

    def wheelEvent(self, event: QtGui.QWheelEvent | None) -> None:
        # Disable scrolling by ignoring wheel events
        if event:
            event.ignore()

    def _fit_pixmap_in_view(self) -> None:
        if self._pixmapitem.pixmap().isNull():
            return
        self.fitInView(self._pixmapitem, self._aspect_ratio_mode)

    def _fit_curve_in_view(self) -> None:
        if self._curve_data is None:
            return
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)

    def _clear_curve(self) -> None:
        self._curve_data = None
        self._curve_axes_item.setPath(QtGui.QPainterPath())
        self._curve_item.setPath(QtGui.QPainterPath())
        self._curve_axes_item.hide()
        self._curve_item.hide()

    def _rebuild_curve_paths(self) -> None:
        if self._curve_data is None:
            self._clear_curve()
            return
        x, y = self._curve_data
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            self._clear_curve()
            return
        x_min = float(np.nanmin(x[finite]))
        x_max = float(np.nanmax(x[finite]))
        y_min = float(np.nanmin(y[finite]))
        y_max = float(np.nanmax(y[finite]))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        if y_min == y_max:
            y_pad = abs(y_min) * 0.05 if y_min != 0.0 else 0.5
            y_min -= y_pad
            y_max += y_pad
        x_span = x_max - x_min
        y_span = y_max - y_min

        curve_path = QtGui.QPainterPath()
        drawing = False
        for x_value, y_value in zip(x, y, strict=True):
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                drawing = False
                continue
            point_x = (float(x_value) - x_min) / x_span
            point_y = 1.0 - ((float(y_value) - y_min) / y_span)
            if drawing:
                curve_path.lineTo(point_x, point_y)
            else:
                curve_path.moveTo(point_x, point_y)
                drawing = True

        axes_path = QtGui.QPainterPath()
        if y_min <= 0.0 <= y_max:
            zero_y = 1.0 - ((0.0 - y_min) / y_span)
            axes_path.moveTo(0.0, zero_y)
            axes_path.lineTo(1.0, zero_y)

        palette = self.palette()
        curve_pen = QtGui.QPen(_summary_accent_color())
        curve_pen.setWidthF(1.6)
        curve_pen.setCosmetic(True)
        axes_color = palette.color(QtGui.QPalette.ColorRole.Mid)
        axes_color.setAlpha(170)
        axes_pen = QtGui.QPen(axes_color)
        axes_pen.setWidthF(1.0)
        axes_pen.setCosmetic(True)

        self._curve_item.setPath(curve_path)
        self._curve_item.setPen(curve_pen)
        self._curve_axes_item.setPath(axes_path)
        self._curve_axes_item.setPen(axes_pen)
        self._curve_axes_item.show()
        self._curve_item.show()
        scene = self.scene()
        if scene is not None:
            scene.setSceneRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        self._fit_curve_in_view()

    def _load_prompt_size_hint(self) -> QtCore.QSize:
        button_hint = self._load_button.sizeHint()
        return QtCore.QSize(button_hint.width() + 24, button_hint.height() + 24)

    def _position_load_button(self) -> None:
        if self._load_button.isHidden():
            return
        viewport = self.viewport()
        viewport_rect = self.rect() if viewport is None else viewport.geometry()
        button_size = self._load_button.sizeHint()
        x = viewport_rect.x() + max(
            0, (viewport_rect.width() - button_size.width()) // 2
        )
        y = viewport_rect.y() + max(
            0, (viewport_rect.height() - button_size.height()) // 2
        )
        self._load_button.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), button_size))


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
        server = self._manager.server
        if erlab.interactive.utils.qt_is_valid(server):
            if server.isRunning():  # pragma: no branch
                server.stop()
            if not server.isRunning():
                for signal, slot in (
                    (server.sigReceived, self._manager._data_recv),
                    (server.sigLoadRequested, self._manager._data_load),
                    (server.sigReplaceRequested, self._manager._data_replace),
                    (server.sigDataRequested, self._manager._send_imagetool_data),
                    (server.sigWatchInfoRequested, self._manager._send_watch_info),
                    (self._manager._sigReplyData, server.set_return_value),
                    (server.sigRemoveIndex, self._manager.remove_imagetool),
                    (server.sigShowIndex, self._manager.show_imagetool),
                    (server.sigRemoveUID, self._manager._remove_watched),
                    (server.sigShowUID, self._manager._show_watched),
                    (server.sigUnwatchUID, self._manager._data_unwatch),
                    (server.sigWatchedVarChanged, self._manager._data_watched_update),
                ):
                    with contextlib.suppress(TypeError, RuntimeError):
                        signal.disconnect(slot)
                server.deleteLater()

        watcher_server = self._manager.watcher_server
        if erlab.interactive.utils.qt_is_valid(watcher_server):
            if watcher_server.isRunning():  # pragma: no branch
                watcher_server.stop()
            if not watcher_server.isRunning():
                with contextlib.suppress(TypeError, RuntimeError):
                    self._manager._sigWatchedDataEdited.disconnect(
                        watcher_server.send_parameters
                    )
                watcher_server.deleteLater()

    def open_settings(self) -> None:
        """Open the settings dialog for the ImageTool manager."""
        dialog = self._manager._additional_windows.get("settings")
        if dialog is None or not erlab.interactive.utils.qt_is_valid(dialog):
            dialog = erlab.interactive._options.OptionDialog(self._manager)
            self._manager._additional_windows["settings"] = dialog

            def cleanup(_obj: QtCore.QObject | None = None) -> None:
                if self._manager._additional_windows.get("settings") is dialog:
                    self._manager._additional_windows.pop("settings", None)

            dialog.destroyed.connect(cleanup)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

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
