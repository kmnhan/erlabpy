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
import enum
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
from collections.abc import Iterable, ValuesView
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import qtawesome as qta
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.imagetool import ImageTool, _parse_input
from erlab.interactive.imagetool.core import SlicerLinkProxy
from erlab.interactive.utils import (
    IconActionButton,
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
"""Name of `QtCore.QSharedMemory` that enforces single instance of ImageToolManager.

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


class _WrapperItemDataRole(enum.IntEnum):
    ToolIndexRole = QtCore.Qt.ItemDataRole.UserRole + 1


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


def fill_rounded_rect(
    painter: QtGui.QPainter,
    rect: QtCore.QRect | QtCore.QRectF,
    facecolor: QtGui.QColor | QtGui.QBrush,
    edgecolor: QtGui.QColor | QtGui.QBrush,
    linewidth: float,
    radius: float,
):
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    rect = QtCore.QRectF(rect)
    rect.adjust(linewidth / 2, linewidth / 2, -linewidth / 2, -linewidth / 2)
    path = QtGui.QPainterPath()
    path.addRoundedRect(rect, radius, radius)

    painter.setClipPath(path)
    painter.fillPath(path, QtGui.QBrush(facecolor))
    painter.setPen(QtGui.QPen(edgecolor, linewidth))
    painter.drawPath(path)
    painter.restore()


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
                manager.list_view.selected_tool_indices, self.new_names(), strict=True
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


class _ImageToolWrapper(QtCore.QObject):
    """Wrapper for ImageTool objects.

    This class wraps an ImageTool object and provides additional functionality in the
    manager such as archiving and unarchiving and window geometry tracking.
    """

    def __init__(self, manager: ImageToolManager, index: int, tool: ImageTool) -> None:
        super().__init__(manager)
        self._manager = weakref.ref(manager)
        self._index: int = index
        self._tool: ImageTool | None = None
        self._recent_geometry: QtCore.QRect | None = None
        self._name: str = tool.windowTitle()
        self._archived_fname: str | None = None

        self.tool = tool

    @property
    def index(self) -> int:
        """Index of the ImageTool in the manager.

        This index is unique for each ImageTool and is used to identify the tool in the
        manager.
        """
        return self._index

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
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name
        cast(ImageTool, self.tool).setWindowTitle(self.label_text)

    @property
    def label_text(self) -> str:
        """Label text shown in the window title and the manager.

        The label text is a combination of the index and the name of the tool.
        """
        new_title = f"{self.index}"
        if self.name != "":
            new_title += f": {self.name}"
        return new_title

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

    @QtCore.Slot()
    @QtCore.Slot(str)
    def update_title(self, title: str | None = None) -> None:
        if not self.archived:
            if title is None:
                title = cast(ImageTool, self.tool).windowTitle()
            self.name = title

    @QtCore.Slot()
    def visibility_changed(self) -> None:
        tool = cast(ImageTool, self.tool)
        self._recent_geometry = tool.geometry()

    @QtCore.Slot()
    def show_tool(self) -> None:
        """Show the tool window.

        If the tool is not visible, it is shown and raised to the top. Archived tools
        are unarchived before being shown.
        """
        if self.tool is None:
            self.unarchive()

        if self.tool is not None:
            if not self.tool.isVisible() and self._recent_geometry is not None:
                self.tool.setGeometry(self._recent_geometry)
            self.tool.show()
            self.tool.raise_()
            self.tool.activateWindow()

    @QtCore.Slot()
    def close_tool(self) -> None:
        """Close the tool window.

        This method only closes the tool window. The tool object is not destroyed and
        can be reopened later.
        """
        if self.tool is not None:
            self.tool.close()

    @QtCore.Slot()
    def dispose_tool(self) -> None:
        """Dispose the tool object.

        This method closes the tool window and destroys the tool object. The tool object
        is not recoverable after this operation.
        """
        self.tool = None

    @QtCore.Slot()
    def archive(self) -> None:
        """Archive the ImageTool.

        Unlike :meth:`dispose_tool`, this method saves the tool object to a file and can
        be recovered later. The archived tools are grayed out in the manager.

        Instead of calling this directly, use :meth:`ImageToolManager.archive_selected`
        which displays a wait dialog.
        """
        if not self.archived:
            self._archived_fname = os.path.join(
                self.manager.cache_dir, str(uuid.uuid4())
            )
            cast(ImageTool, self.tool).to_file(self._archived_fname)
            self.dispose_tool()

    @QtCore.Slot()
    def unarchive(self) -> None:
        """
        Restore the ImageTool from the archive.

        Instead of calling this directly, use
        :meth:`ImageToolManager.unarchive_selected` which displays a wait dialog.
        """
        if self.archived:
            self.tool = ImageTool.from_file(cast(str, self._archived_fname))
            self.tool.show()
            self.manager._sigReloadLinkers.emit()


class _ResizingLineEdit(QtWidgets.QLineEdit):
    """:class:`QtWidgets.QLineEdit` that resizes itself to fit the text."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textChanged.connect(self._on_text_changed)

    @QtCore.Slot(str)
    def _on_text_changed(self, text):
        # https://stackoverflow.com/a/73663065
        font_metrics = QtGui.QFontMetrics(self.font())

        tm = self.textMargins()
        tm_size = QtCore.QSize(tm.left() + tm.right(), tm.top() + tm.bottom())

        cm = self.contentsMargins()
        cm_size = QtCore.QSize(cm.left() + cm.right(), cm.top() + cm.bottom())

        contents_size = (
            font_metrics.size(0, text) + tm_size + cm_size + QtCore.QSize(8, 4)
        )

        self.setFixedSize(
            self.style().sizeFromContents(
                QtWidgets.QStyle.ContentsType.CT_LineEdit, None, contents_size, self
            )
        )


class _Placeholder:
    pass


class _ImageToolWrapperItemDelegate(QtWidgets.QStyledItemDelegate):
    """
    A :class:`QtWidgets.QStyledItemDelegate` that handles displaying list view items.

    Methods
    -------
    manager
        Returns the manager instance, raises LookupError if the manager is destroyed.
    _combine_colors(c1, c2, weight=1.0)
        Combines two colors with a given weight.
    createEditor(parent, option, index)
        Creates an editor widget for editing item names.
    updateEditorGeometry(editor, option, index)
        Updates the geometry of the editor widget.
    paint(painter, option, index)
        Custom paint method for rendering items in the list view.
    """

    icon_width: int = 16
    icon_height: int = 16
    icon_right_pad: int = 5
    icon_inner_pad: float = 1.5
    icon_border_width: float = 1.5
    icon_corner_radius: float = 5.0

    def __init__(
        self, manager: ImageToolManager, parent: _ImageToolWrapperListView
    ) -> None:
        super().__init__(parent)
        self._manager = weakref.ref(manager)
        self._font_size = QtGui.QFont().pointSize()
        self._current_editor: weakref.ref[QtWidgets.QLineEdit | _Placeholder] = (
            weakref.ref(_Placeholder())
        )

    @property
    def manager(self) -> ImageToolManager:
        _manager = self._manager()
        if _manager:
            return _manager
        raise LookupError("Parent was destroyed")

    @staticmethod
    def _combine_colors(
        c1: QtGui.QColor, c2: QtGui.QColor, weight: float = 1.0
    ) -> QtGui.QColor:
        """Combine two colors with a given weight.

        Default weight is 1.0, which returns the average of the two colors for each RGB
        channel.
        """
        c3 = QtGui.QColor()
        c3.setRedF((c1.redF() * weight + c2.redF() * (2 - weight)) / 2.0)
        c3.setGreenF((c1.greenF() * weight + c2.greenF() * (2 - weight)) / 2.0)
        c3.setBlueF((c1.blueF() * weight + c2.blueF() * (2 - weight)) / 2.0)
        return c3

    def createEditor(
        self,
        parent: QtWidgets.QWidget | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtWidgets.QWidget | None:
        option.font.setPointSize(self._font_size)
        editor = _ResizingLineEdit(parent)
        editor.setFont(option.font)
        editor.setFrame(True)
        editor.setPlaceholderText("Enter new name")
        self._current_editor = weakref.ref(editor)
        return editor

    def updateEditorGeometry(
        self,
        editor: QtWidgets.QWidget | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        if editor is not None:
            rect = QtCore.QRectF(option.rect)
            rect.setLeft(rect.left() + 5)
            rect.setTop(rect.center().y() - editor.sizeHint().height() / 2)
            editor.setGeometry(rect.toRect())

    def paint(
        self,
        painter: QtGui.QPainter | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        if painter is None:
            return
        painter.save()

        # Set font size
        option.font.setPointSize(self._font_size)
        painter.setFont(option.font)

        # Draw background
        if QtWidgets.QStyle.StateFlag.State_Selected in option.state:
            # Dilute the highlight color with the base color
            painter.fillRect(
                option.rect,
                self._combine_colors(
                    option.palette.color(QtGui.QPalette.ColorRole.Highlight),
                    option.palette.color(QtGui.QPalette.ColorRole.Base),
                    weight=0.5,
                ),
            )
        else:
            painter.fillRect(option.rect, option.palette.base())

        # Draw text if not editing
        view = cast(_ImageToolWrapperListView, self.parent())
        if not (
            view.state() == QtWidgets.QAbstractItemView.State.EditingState
            and view.currentIndex() == index
        ):
            # Grey text for archived tools
            painter.setPen(index.data(role=QtCore.Qt.ItemDataRole.ForegroundRole))

            # A bit of left pad for cosmetic reasons
            text_rect = option.rect.adjusted(5, 0, 0, 0)

            # Space for icon
            right_pad = int(
                self.icon_width + self.icon_right_pad * 2 + self.icon_inner_pad * 2
            )

            # Elide text if necessary
            elided_text = QtGui.QFontMetrics(option.font).elidedText(
                index.data(role=QtCore.Qt.ItemDataRole.DisplayRole),  # Tool label
                view.textElideMode(),
                text_rect.width() - right_pad,
            )
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft,
                elided_text,
            )

        # Draw icon for linked tools
        tool_wrapper: _ImageToolWrapper = self.manager._tool_wrappers[
            index.data(role=_WrapperItemDataRole.ToolIndexRole)
        ]
        if not tool_wrapper.archived and tool_wrapper.slicer_area.is_linked:
            icon_x = option.rect.right() - self.icon_width - self.icon_right_pad
            icon_y = option.rect.center().y() - self.icon_height // 2

            icon = qta.icon(
                "mdi6.link-variant",
                color=self.manager.color_for_linker(
                    cast(SlicerLinkProxy, tool_wrapper.slicer_area._linking_proxy)
                ),
            )
            fill_rounded_rect(
                painter,
                QtCore.QRectF(
                    icon_x - self.icon_inner_pad,
                    icon_y - self.icon_inner_pad,
                    self.icon_width + 2 * self.icon_inner_pad,
                    self.icon_height + 2 * self.icon_inner_pad,
                ),
                facecolor=option.palette.base(),
                edgecolor=option.palette.mid(),
                linewidth=self.icon_border_width,
                radius=self.icon_corner_radius,
            )
            icon.paint(
                painter,
                QtCore.QRect(icon_x, icon_y, self.icon_width, self.icon_height),
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
            )

        painter.restore()


class _ImageToolWrapperListModel(QtCore.QAbstractListModel):
    def __init__(self, manager: ImageToolManager, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._manager = weakref.ref(manager)

    @property
    def manager(self) -> ImageToolManager:
        _manager = self._manager()
        if _manager:
            return _manager
        raise LookupError("Parent was destroyed")

    def _tool_index(self, row_index: int | QtCore.QModelIndex) -> int:
        if isinstance(row_index, QtCore.QModelIndex):
            row_index = row_index.row()
        return self.manager._displayed_indices[row_index]

    def _tool_wrapper(self, row_index: int | QtCore.QModelIndex) -> _ImageToolWrapper:
        return self.manager._tool_wrappers[self._tool_index(row_index)]

    def _row_index(self, tool_index: int) -> QtCore.QModelIndex:
        return self.index(self.manager._displayed_indices.index(tool_index))

    def _is_archived(self, row_index: int | QtCore.QModelIndex) -> bool:
        if isinstance(row_index, QtCore.QModelIndex):
            row_index = row_index.row()
        return self._tool_wrapper(row_index).archived

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return len(self.manager._displayed_indices)

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if not index.isValid():
            return None

        tool_idx: int = self._tool_index(index)

        match role:
            case QtCore.Qt.ItemDataRole.DisplayRole:
                if tool_idx < 0:
                    return ""
                return self.manager.label_of_tool(tool_idx)

            case QtCore.Qt.ItemDataRole.EditRole:
                if tool_idx < 0:
                    return ""
                return self.manager.name_of_tool(tool_idx)

            case _WrapperItemDataRole.ToolIndexRole:
                return tool_idx

            case QtCore.Qt.ItemDataRole.SizeHintRole:
                return QtCore.QSize(100, 30)

            case QtCore.Qt.ItemDataRole.ForegroundRole:
                palette = QtWidgets.QApplication.palette()
                if self._is_archived(index):
                    # Make text seem disabled for archived tools
                    return palette.color(
                        QtGui.QPalette.ColorGroup.Disabled,
                        QtGui.QPalette.ColorRole.Text,
                    )
                return palette.color(
                    QtGui.QPalette.ColorGroup.Active,
                    QtGui.QPalette.ColorRole.Text,
                )

        return None

    def removeRows(
        self, row: int, count: int, parent: QtCore.QModelIndex | None = None
    ) -> bool:
        if parent is None:
            parent = QtCore.QModelIndex()
        self.beginRemoveRows(parent, row, row + count - 1)
        del self.manager._displayed_indices[row : row + count]
        self.endRemoveRows()
        return True

    def insertRows(
        self, row: int, count: int, parent: QtCore.QModelIndex | None = None
    ) -> bool:
        if parent is None:
            parent = QtCore.QModelIndex()
        self.beginInsertRows(parent, row, row + count - 1)
        for i in range(count):
            self.manager._displayed_indices.insert(row + i, -1)
        self.endInsertRows()
        return True

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False

        if role == QtCore.Qt.ItemDataRole.EditRole:
            self.manager.rename_tool(self._tool_index(index), value)

            self.dataChanged.emit(index, index, [role])
            return True

        if role == _WrapperItemDataRole.ToolIndexRole:
            if index.row() >= len(self.manager._displayed_indices):
                self.manager._displayed_indices.append(value)
            else:
                self.manager._displayed_indices[index.row()] = value
            self.dataChanged.emit(index, index, [role])
            return True

        return False

    def canDropMimeData(
        self,
        data: QtCore.QMimeData | None,
        action: QtCore.Qt.DropAction,
        row: int,
        column: int,
        parent: QtCore.QModelIndex,
    ):
        if data is None:
            return False
        if not data.hasFormat("application/json"):
            return False
        return not column > 0

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        default_flags = (
            QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        )

        if not self._is_archived(index):
            default_flags |= QtCore.Qt.ItemFlag.ItemIsEditable

        if index.isValid():
            return (
                QtCore.Qt.ItemFlag.ItemIsDragEnabled
                | QtCore.Qt.ItemFlag.ItemIsDropEnabled
                | default_flags
            )
        return QtCore.Qt.ItemFlag.ItemIsDropEnabled | default_flags

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        return QtCore.Qt.DropAction.MoveAction

    def dropMimeData(
        self,
        data: QtCore.QMimeData | None,
        action: QtCore.Qt.DropAction,
        row: int,
        column: int,
        parent: QtCore.QModelIndex,
    ) -> bool:
        if data is None:
            return False
        if not self.canDropMimeData(data, action, row, column, parent):
            return False

        if action == QtCore.Qt.DropAction.IgnoreAction:
            return True

        start: int = -1
        if row != -1:
            # Inserting above/below an existing node
            start = row
        elif parent.isValid():
            # Inserting onto an existing node
            start = parent.row()
        else:
            # Inserting at the root
            start = self.rowCount(QtCore.QModelIndex())

        encoded_data = data.data("application/json")
        stream = QtCore.QDataStream(
            encoded_data, QtCore.QIODevice.OpenModeFlag.ReadOnly
        )
        new_items: list[int] = []
        rows: int = 0

        while not stream.atEnd():
            new_items.append(int(stream.readInt()))
            rows += 1

        self.insertRows(start, rows, QtCore.QModelIndex())
        for tool_idx in new_items:
            self.setData(
                self.index(start), tool_idx, _WrapperItemDataRole.ToolIndexRole
            )
            start += 1

        return True

    def mimeData(self, indexes: Iterable[QtCore.QModelIndex]) -> QtCore.QMimeData:
        mime_data = QtCore.QMimeData()
        encoded_data = QtCore.QByteArray()
        stream = QtCore.QDataStream(
            encoded_data, QtCore.QIODevice.OpenModeFlag.WriteOnly
        )

        for index in indexes:
            if index.isValid():
                tool_idx = self.data(index, _WrapperItemDataRole.ToolIndexRole)
                stream.writeInt(tool_idx)

        mime_data.setData("application/json", encoded_data)
        return mime_data

    def mimeTypes(self) -> list[str]:
        return ["application/json"]


class _ImageToolWrapperListView(QtWidgets.QListView):
    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__()
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.DragDropMode.InternalMove)
        self.setEditTriggers(
            self.EditTrigger.SelectedClicked | self.EditTrigger.EditKeyPressed
        )
        self.setWordWrap(True)

        self._model = _ImageToolWrapperListModel(manager, self)
        self.setModel(self._model)

        self.setItemDelegate(_ImageToolWrapperItemDelegate(manager, self))

        self._selection_model = cast(QtCore.QItemSelectionModel, self.selectionModel())

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)

        self.doubleClicked.connect(self.double_clicked)

        self._menu = QtWidgets.QMenu("Menu", self)
        self._menu.addAction(manager.open_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.show_action)
        self._menu.addAction(manager.hide_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.remove_action)
        self._menu.addAction(manager.archive_action)
        self._menu.addAction(manager.unarchive_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.rename_action)
        self._menu.addAction(manager.link_action)
        self._menu.addAction(manager.unlink_action)

    @QtCore.Slot(QtCore.QPoint)
    def _show_menu(self, position: QtCore.QPoint) -> None:
        self._menu.popup(self.mapToGlobal(position))

    @property
    def selected_tool_indices(self) -> list[int]:
        """Currently selected tools."""
        return [
            self._model.manager._displayed_indices[index.row()]
            for index in self.selectedIndexes()
        ]

    @QtCore.Slot(QtCore.QModelIndex)
    def double_clicked(self, index: QtCore.QModelIndex) -> None:
        self._model._tool_wrapper(index).show_tool()

    @QtCore.Slot()
    def select_all(self) -> None:
        self.selectAll()

    @QtCore.Slot()
    def deselect_all(self) -> None:
        self.clearSelection()

    @QtCore.Slot()
    def refresh(self) -> None:
        self._model.dataChanged.emit(
            self._model.index(0), self._model.index(self._model.rowCount() - 1)
        )

    def add_tool(self, index: int) -> None:
        n_rows = self._model.rowCount()
        self._model.insertRows(n_rows, 1)
        self._model.setData(
            self._model.index(n_rows),
            index,
            _WrapperItemDataRole.ToolIndexRole,
        )

    def remove_tool(self, index: int) -> None:
        for i, tool_idx in enumerate(self._model.manager._displayed_indices):
            if tool_idx == index:
                self._model.removeRows(i, 1)
                break


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

        self._tool_wrappers: dict[int, _ImageToolWrapper] = {}
        self._displayed_indices: list[int] = []
        self._linkers: list[SlicerLinkProxy] = []

        # Initialize actions

        self.show_action = QtWidgets.QAction("Show", self)
        self.show_action.triggered.connect(self.show_selected)
        self.show_action.setToolTip("Show selected windows")

        self.hide_action = QtWidgets.QAction("Hide", self)
        self.hide_action.triggered.connect(self.hide_selected)
        self.hide_action.setToolTip("Hide selected windows")

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

        self.remove_action = QtWidgets.QAction("Remove", self)
        self.remove_action.triggered.connect(self.remove_selected)
        self.remove_action.setToolTip("Remove selected windows")

        self.rename_action = QtWidgets.QAction("Rename", self)
        self.rename_action.triggered.connect(self.rename_selected)
        self.rename_action.setToolTip("Rename selected windows")

        self.link_action = QtWidgets.QAction("Link", self)
        self.link_action.triggered.connect(self.link_selected)
        self.link_action.setToolTip("Link selected windows")

        self.unlink_action = QtWidgets.QAction("Unlink", self)
        self.unlink_action.triggered.connect(self.unlink_selected)
        self.unlink_action.setToolTip("Unlink selected windows")

        self.archive_action = QtWidgets.QAction("Archive", self)
        self.archive_action.triggered.connect(self.archive_selected)
        self.archive_action.setToolTip("Archive selected windows")

        self.unarchive_action = QtWidgets.QAction("Unarchive", self)
        self.unarchive_action.triggered.connect(self.unarchive_selected)
        self.unarchive_action.setToolTip("Unarchive selected windows")

        # Construct GUI

        titlebar = QtWidgets.QWidget()
        titlebar_layout = QtWidgets.QHBoxLayout()
        titlebar_layout.setContentsMargins(0, 0, 0, 0)
        titlebar.setLayout(titlebar_layout)

        file_menu: QtWidgets.QMenu = cast(QtWidgets.QMenu, menu_bar.addMenu("&File"))
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.load_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.gc_action)

        edit_menu: QtWidgets.QMenu = cast(QtWidgets.QMenu, menu_bar.addMenu("&Edit"))
        edit_menu.addAction(self.remove_action)
        edit_menu.addAction(self.archive_action)
        edit_menu.addAction(self.unarchive_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.rename_action)
        edit_menu.addAction(self.link_action)
        edit_menu.addAction(self.unlink_action)

        view_menu: QtWidgets.QMenu = cast(QtWidgets.QMenu, menu_bar.addMenu("&View"))
        view_menu.addAction(self.show_action)
        view_menu.addAction(self.hide_action)

        self.open_button = IconActionButton(
            self.open_action,
            "mdi6.folder-open-outline",
        )
        self.remove_button = IconActionButton(
            self.remove_action,
            "mdi6.window-close",
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

        titlebar_layout.addWidget(self.open_button)
        titlebar_layout.addWidget(self.remove_button)
        titlebar_layout.addWidget(self.rename_button)
        titlebar_layout.addWidget(self.link_button)
        titlebar_layout.addWidget(self.unlink_button)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        container.setLayout(layout)

        layout.addWidget(titlebar)

        self.list_view = _ImageToolWrapperListView(self)
        layout.addWidget(self.list_view)
        self.list_view._selection_model.selectionChanged.connect(
            self._update_action_state
        )

        # Temporary directory for storing archived data
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="erlab_archive_")

        # Store most recent name filter and directory for new windows
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None

        self.setCentralWidget(container)
        self.sigLinkersChanged.connect(self._update_action_state)
        self.sigLinkersChanged.connect(self.list_view.refresh)
        self._sigReloadLinkers.connect(self._cleanup_linkers)
        self._update_action_state()

        self.server: _ManagerServer = _ManagerServer()
        self.server.sigReceived.connect(self.data_recv)
        self.server.start()

        # Shared memory for detecting multiple instances
        self._shm = QtCore.QSharedMemory(_SHM_NAME)
        self._shm.create(1)  # Create segment so that it can be attached to

        self.setMinimumWidth(300)
        self.setMinimumHeight(200)

    @property
    def cache_dir(self) -> str:
        """Name of the cache directory where archived data are stored."""
        return self._tmp_dir.name

    @property
    def ntools(self) -> int:
        """Number of ImageTool windows being handled by the manager."""
        return len(self._tool_wrappers)

    @property
    def next_idx(self) -> int:
        """Index for the next ImageTool window."""
        return max(self._tool_wrappers.keys(), default=-1) + 1

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
        if index not in self._tool_wrappers:
            raise KeyError(f"Tool of index '{index}' not found")

        wrapper = self._tool_wrappers[index]
        if wrapper.archived:
            if unarchive:
                wrapper.unarchive()
            else:
                raise KeyError(f"Tool of index '{index}' is archived")
        return cast(ImageTool, wrapper.tool)

    def color_for_linker(self, linker: SlicerLinkProxy) -> QtGui.QColor:
        """Get the color that should represent the given linker."""
        idx = self._linkers.index(linker)
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
        wrapper = _ImageToolWrapper(self, index, tool)
        self._tool_wrappers[index] = wrapper
        wrapper.update_title()

        self._sigReloadLinkers.emit()

        tool.show()

        if activate:
            tool.activateWindow()
            tool.raise_()

        # Add to view after initialization
        self.list_view.add_tool(index)

        return index

    @QtCore.Slot()
    def _update_action_state(self) -> None:
        """Update the state of the actions based on the current selection."""
        selection_archived: list[int] = []
        selection_unarchived: list[int] = []
        for s in self.list_view.selected_tool_indices:
            if self._tool_wrappers[s].archived:
                selection_archived.append(s)
            else:
                selection_unarchived.append(s)

        selection_all = selection_archived + selection_unarchived

        something_selected: bool = len(selection_all) != 0
        only_unarchived: bool = len(selection_archived) == 0
        only_archived: bool = len(selection_unarchived) == 0

        self.show_action.setEnabled(something_selected)
        self.hide_action.setEnabled(something_selected)
        self.remove_action.setEnabled(something_selected)
        self.rename_action.setEnabled(something_selected and only_unarchived)
        self.archive_action.setEnabled(something_selected and only_unarchived)
        self.unarchive_action.setEnabled(something_selected and only_archived)

        self.link_action.setDisabled(only_archived)
        self.unlink_action.setDisabled(only_archived)

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
                self.get_tool(index).slicer_area.is_linked
                for index in selection_unarchived
            ]
            self.unlink_action.setEnabled(any(is_linked))

            if all(is_linked):
                proxies = [
                    self.get_tool(index).slicer_area._linking_proxy
                    for index in selection_unarchived
                ]
                if all(p == proxies[0] for p in proxies):
                    self.link_action.setEnabled(False)

    def remove_tool(self, index: int) -> None:
        """Remove the ImageTool window corresponding to the given index."""
        self.list_view.remove_tool(index)

        wrapper = self._tool_wrappers.pop(index)
        if not wrapper.archived:
            cast(ImageTool, wrapper.tool).removeEventFilter(wrapper)
        wrapper.dispose_tool()
        del wrapper

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
        """Show selected ImageTool windows."""
        index_list = self.list_view.selected_tool_indices

        require_unarchive = any(self._tool_wrappers[i].archived for i in index_list)
        if require_unarchive:
            # This is just to display the wait dialog for unarchiving.
            # If this part is removed, the showing will just hang until the unarchiving
            # is finished without any feedback.
            self.unarchive_selected()

        for index in index_list:
            self._tool_wrappers[index].show_tool()

    @QtCore.Slot()
    def hide_selected(self) -> None:
        """Hide selected ImageTool windows."""
        for index in self.list_view.selected_tool_indices:
            self._tool_wrappers[index].close_tool()

    @QtCore.Slot()
    def remove_selected(self) -> None:
        """Close selected ImageTool windows."""
        checked_names = self.list_view.selected_tool_indices

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Close selected windows?")
        msg_box.setInformativeText(
            "1 selected window will be closed."
            if len(checked_names) == 1
            else f"{len(checked_names)} selected windows will be closed."
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

        if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
            for name in checked_names:
                self.remove_tool(name)

    @QtCore.Slot()
    def rename_selected(self) -> None:
        """Rename selected ImageTool windows."""
        selected = self.list_view.selected_tool_indices
        if len(selected) == 1:
            self.list_view.edit(self.list_view._model._row_index(selected[0]))
            return
        dialog = _RenameDialog(self, [self._tool_wrappers[i].name for i in selected])
        dialog.exec()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    @QtCore.Slot(bool, bool)
    def link_selected(self, link_colors: bool = True, deselect: bool = True) -> None:
        """Link selected ImageTool windows."""
        self.unlink_selected(deselect=False)
        self.link_tools(*self.list_view.selected_tool_indices, link_colors=link_colors)
        if deselect:
            self.list_view.deselect_all()

    @QtCore.Slot()
    @QtCore.Slot(bool)
    def unlink_selected(self, deselect: bool = True) -> None:
        """Unlink selected ImageTool windows."""
        for index in self.list_view.selected_tool_indices:
            self.get_tool(index).slicer_area.unlink()
        self._sigReloadLinkers.emit()
        if deselect:
            self.list_view.deselect_all()

    @QtCore.Slot()
    def archive_selected(self) -> None:
        """Archive selected ImageTool windows."""
        with wait_dialog(self, "Archiving..."):
            for index in self.list_view.selected_tool_indices:
                self._tool_wrappers[index].archive()

    @QtCore.Slot()
    def unarchive_selected(self) -> None:
        """Unarchive selected ImageTool windows."""
        with wait_dialog(self, "Unarchiving..."):
            for index in self.list_view.selected_tool_indices:
                self._tool_wrappers[index].unarchive()

    def rename_tool(self, index: int, new_name: str) -> None:
        """Rename the ImageTool window corresponding to the given index."""
        self._tool_wrappers[index].name = new_name

    def link_tools(self, *indices, link_colors: bool = True) -> None:
        """Link the ImageTool windows corresponding to the given indices."""
        linker = SlicerLinkProxy(
            *[self.get_tool(t).slicer_area for t in indices], link_colors=link_colors
        )
        self._linkers.append(linker)
        self._sigReloadLinkers.emit()

    def name_of_tool(self, index: int) -> str:
        """Get the name of the ImageTool window corresponding to the given index."""
        return self._tool_wrappers[index].name

    def label_of_tool(self, index: int) -> str:
        """Get the label of the ImageTool window corresponding to the given index."""
        return self._tool_wrappers[index].label_text

    @QtCore.Slot()
    def garbage_collect(self) -> None:
        """Run garbage collection to free up memory."""
        gc.collect()

    def _to_datatree(self, close: bool = False) -> xr.DataTree:
        """Convert the current state of the manager to a DataTree object."""
        constructor: dict[str, xr.Dataset] = {}
        for index in tuple(self._tool_wrappers.keys()):
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

            for tool in list(self._tool_wrappers.keys()):
                self.remove_tool(tool)

        # Clean up temporary directory
        self._tmp_dir.cleanup()

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
    if sys.platform != "win32":
        # Shared memory is removed on crash only on Windows
        unix_fix_shared_mem = QtCore.QSharedMemory(_SHM_NAME)
        if unix_fix_shared_mem.attach():
            # Detaching will release the shared memory if no other process is attached
            unix_fix_shared_mem.detach()

    # If attaching succeeds, another instance is running
    return QtCore.QSharedMemory(_SHM_NAME).attach()


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
