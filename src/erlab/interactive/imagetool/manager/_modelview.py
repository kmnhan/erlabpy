"""Model-view architecture used for displaying the list of ImageTool windows."""

from __future__ import annotations

__all__ = ["_ImageToolWrapperListView"]

import enum
import typing
import weakref

import qtawesome as qta
from qtpy import QtCore, QtGui, QtWidgets

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    import erlab
    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import _ImageToolWrapper


class _WrapperItemDataRole(enum.IntEnum):
    ToolIndexRole = QtCore.Qt.ItemDataRole.UserRole + 1


def _fill_rounded_rect(
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

        # Initialize popup preview
        self.preview_popup = QtWidgets.QLabel(parent)
        self.preview_popup.setWindowFlags(QtCore.Qt.WindowType.ToolTip)
        self.preview_popup.setScaledContents(True)
        self.preview_popup.hide()

        # Handle preview closing
        viewport = parent.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)

        self._force_hover: bool = False  # Flag for debugging

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

        # Draw text only if not editing
        view = typing.cast("_ImageToolWrapperListView", self.parent())
        is_editing: bool = (
            view.state() == QtWidgets.QAbstractItemView.State.EditingState
            and view.currentIndex() == index
        )
        if not is_editing:
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

        tool_wrapper: _ImageToolWrapper = self.manager._tool_wrappers[
            index.data(role=_WrapperItemDataRole.ToolIndexRole)
        ]

        # Draw icon for linked tools
        if not tool_wrapper.archived and tool_wrapper.slicer_area.is_linked:
            icon_x = option.rect.right() - self.icon_width - self.icon_right_pad
            icon_y = option.rect.center().y() - self.icon_height // 2

            icon = qta.icon(
                "mdi6.link-variant",
                color=self.manager.color_for_linker(
                    typing.cast(
                        "erlab.interactive.imagetool.core.SlicerLinkProxy",
                        tool_wrapper.slicer_area._linking_proxy,
                    )
                ),
            )
            _fill_rounded_rect(
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

        # Show preview on hover
        if (
            not tool_wrapper.archived
            and not is_editing
            and self.manager.preview_action.isChecked()
            and (
                QtWidgets.QStyle.StateFlag.State_MouseOver in option.state
                or self._force_hover
            )
        ):
            box_ratio, pixmap = tool_wrapper._preview_image
            popup_height = 150

            self.preview_popup.setFixedSize(
                round(popup_height / box_ratio), popup_height
            )
            self.preview_popup.setPixmap(pixmap)

            rect = QtCore.QRect(option.rect)
            rect.setTop(rect.center().y() + rect.height())
            self.preview_popup.move(
                option.widget.mapToGlobal(rect.center())
                - QtCore.QPoint(int(self.preview_popup.width() / 2), 0)
            )

            self.preview_popup.show()

        painter.restore()

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if event is not None:
            match event.type():
                case (
                    QtCore.QEvent.Type.Resize
                    | QtCore.QEvent.Type.Leave
                    | QtCore.QEvent.Type.WindowStateChange
                ):
                    self.preview_popup.hide()
                case QtCore.QEvent.Type.MouseMove:
                    index = typing.cast(
                        "_ImageToolWrapperListView", self.parent()
                    ).indexAt(typing.cast("QtGui.QMouseEvent", event).pos())
                    if not index.isValid():
                        self.preview_popup.hide()

        return super().eventFilter(obj, event)


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
    ) -> typing.Any:
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
                    QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Text
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
        value: typing.Any,
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

        if index.isValid():
            if not self._is_archived(index):
                default_flags |= QtCore.Qt.ItemFlag.ItemIsEditable
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
            new_items.append(int(stream.readInt64()))
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
                stream.writeInt64(tool_idx)

        mime_data.setData("application/json", encoded_data)
        return mime_data

    def mimeTypes(self) -> list[str]:
        return ["application/json"]


class _ImageToolWrapperListView(QtWidgets.QListView):
    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__()
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)

        # Enable drag & drop for reordering items
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.DragDropMode.InternalMove)

        self.setEditTriggers(
            self.EditTrigger.SelectedClicked | self.EditTrigger.EditKeyPressed
        )  # Enable editing of item names

        self.setWordWrap(True)  # Ellide text when width is too small
        self.setMouseTracking(True)  # Enable hover detection

        self._model = _ImageToolWrapperListModel(manager, self)
        self.setModel(self._model)
        self.setItemDelegate(_ImageToolWrapperItemDelegate(manager, self))
        self._selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self.selectionModel()
        )

        # Show tool on double-click
        self.doubleClicked.connect(self._model.manager.show_selected)

        # Right-click context menu
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self._menu = QtWidgets.QMenu("Menu", self)
        self._menu.addAction(manager.concat_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.show_action)
        self._menu.addAction(manager.hide_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.remove_action)
        self._menu.addAction(manager.archive_action)
        self._menu.addAction(manager.unarchive_action)
        self._menu.addAction(manager.reload_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.rename_action)
        self._menu.addAction(manager.link_action)
        self._menu.addAction(manager.unlink_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.store_action)

    @QtCore.Slot(QtCore.QPoint)
    def _show_menu(self, position: QtCore.QPoint) -> None:
        self._menu.popup(self.mapToGlobal(position))

    @property
    def selected_tool_indices(self) -> list[int]:
        """Currently selected tools.

        The tools are ordered by their position in the list view.
        """
        row_indices = sorted(index.row() for index in self.selectedIndexes())
        return [self._model.manager._displayed_indices[i] for i in row_indices]

    @QtCore.Slot()
    def select_all(self) -> None:
        self.selectAll()

    @QtCore.Slot()
    def deselect_all(self) -> None:
        self.clearSelection()

    @QtCore.Slot()
    @QtCore.Slot(int)
    def refresh(self, idx: int | None = None) -> None:
        """Trigger a refresh of the contents."""
        if idx is None:
            self._model.dataChanged.emit(
                self._model.index(0), self._model.index(self._model.rowCount() - 1)
            )
        else:
            if idx in self._model.manager._displayed_indices:
                self._model.dataChanged.emit(
                    self._model._row_index(idx), self._model._row_index(idx)
                )

    def tool_added(self, index: int) -> None:
        """Update the list view when a new tool is added to the manager.

        This must be called after a tool is added to the manager.
        """
        n_rows = self._model.rowCount()
        self._model.insertRows(n_rows, 1)
        self._model.setData(
            self._model.index(n_rows), index, _WrapperItemDataRole.ToolIndexRole
        )

    def tool_removed(self, index: int) -> None:
        """Update the list view when removing a tool from the manager.

        This must be called before the tool is removed from the manager.
        """
        for i, tool_idx in enumerate(self._model.manager._displayed_indices):
            if tool_idx == index:
                self._model.removeRows(i, 1)
                break
