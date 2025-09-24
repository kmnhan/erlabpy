"""Model-view architecture used for displaying the list of ImageTool windows."""

from __future__ import annotations

__all__ = ["_ImageToolWrapperTreeView"]

import json
import logging
import typing
import weakref

import qtawesome as qta
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.imagetool.manager._wrapper import _ImageToolWrapper

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    import erlab
    from erlab.interactive.imagetool.manager import ImageToolManager

logger = logging.getLogger(__name__)


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

    info_rect_hpad: int = 5

    def __init__(
        self, manager: ImageToolManager, parent: _ImageToolWrapperTreeView
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
        if viewport is not None:  # pragma: no branch
            viewport.installEventFilter(self)

        self._force_hover: bool = False  # Flag for debugging

    @property
    def manager(self) -> ImageToolManager:
        manager = self._manager()
        if manager:
            return manager
        raise LookupError("Parent was destroyed")

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
        if editor is not None:  # pragma: no branch
            rect = QtCore.QRectF(option.rect)
            rect.setTop(rect.center().y() - editor.sizeHint().height() / 2)
            editor.setGeometry(rect.toRect())

    def _show_popup(
        self,
        box_ratio: float,
        pixmap: QtGui.QPixmap,
        option: QtWidgets.QStyleOptionViewItem,
    ) -> None:
        popup_height = 150

        self.preview_popup.setFixedSize(round(popup_height / box_ratio), popup_height)
        self.preview_popup.setPixmap(pixmap)

        rect = QtCore.QRect(option.rect)
        rect.setTop(rect.center().y() + rect.height())
        self.preview_popup.move(
            option.widget.mapToGlobal(rect.center())
            - QtCore.QPoint(int(self.preview_popup.width() / 2), 0)
        )

        self.preview_popup.show()

    def paint(
        self,
        painter: QtGui.QPainter | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        if painter is None:  # pragma: no branch
            return

        painter.save()

        # Set font size
        option.font.setPointSize(self._font_size)
        painter.setFont(option.font)

        # Draw background
        if QtWidgets.QStyle.StateFlag.State_Selected in option.state:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.palette.base())

        ptr = index.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            self._paint_imagetool(painter, option, index)
        elif isinstance(ptr, str):
            self._paint_childtool(painter, option, index)

        painter.restore()

    def _paint_imagetool(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        # Get tool wrapper
        tool_wrapper: _ImageToolWrapper = typing.cast(
            "_ImageToolWrapper", index.internalPointer()
        )

        view = typing.cast("_ImageToolWrapperTreeView", self.parent())
        is_editing: bool = (
            view.state() == QtWidgets.QAbstractItemView.State.EditingState
            and view.currentIndex() == index
        )

        selected: bool = QtWidgets.QStyle.StateFlag.State_Selected in option.state
        if not is_editing:  # pragma: no branch
            if tool_wrapper.archived:
                # Make text seem disabled for archived tools
                group = QtGui.QPalette.ColorGroup.Disabled
            else:
                group = (
                    QtGui.QPalette.ColorGroup.Active
                    if QtWidgets.QStyle.StateFlag.State_Active in option.state
                    else QtGui.QPalette.ColorGroup.Inactive
                )
            role = (
                QtGui.QPalette.ColorRole.HighlightedText
                if selected
                else QtGui.QPalette.ColorRole.Text
            )
            painter.setPen(option.palette.color(group, role))

            # Space for icon
            right_pad = int(
                self.icon_width + self.icon_right_pad * 2 + self.icon_inner_pad * 2
            )

            # Elide text if necessary
            elided_text = QtGui.QFontMetrics(option.font).elidedText(
                index.data(role=QtCore.Qt.ItemDataRole.DisplayRole),  # Tool label
                view.textElideMode(),
                option.rect.width() - right_pad,
            )
            painter.drawText(
                option.rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft,
                elided_text,
            )

        is_linked: bool = (
            not tool_wrapper.archived and tool_wrapper.slicer_area.is_linked
        )

        is_watched: bool = tool_wrapper._watched_varname is not None

        # Draw icon for linked tools
        if is_linked:
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

        # Draw indicator for watched variables
        if is_watched:
            font: QtGui.QFont = option.font
            font.setPointSizeF(self._font_size * 0.9)
            painter.setFont(font)

            info_text: str = typing.cast("str", tool_wrapper._watched_varname)
            metrics = QtGui.QFontMetrics(font)
            bounding_rect = metrics.boundingRect(info_text)

            info_width = bounding_rect.width() + self.info_rect_hpad * 2
            info_height = self.icon_height + 2 * self.icon_inner_pad

            left = option.rect.right() - self.icon_inner_pad - info_width - 3.0
            if is_linked:
                left -= self.icon_width + self.icon_right_pad

            info_rect = QtCore.QRectF(
                left,
                option.rect.center().y() - info_height / 2,
                info_width,
                info_height,
            )

            color = option.palette.color(QtGui.QPalette.ColorRole.Highlight)
            if hasattr(QtGui.QPalette.ColorRole, "Accent"):  # pragma: no branch
                # Accent color is available from Qt 6.6
                color = option.palette.color(QtGui.QPalette.ColorRole.Accent)

            # Draw rounded rectangle
            _fill_rounded_rect(
                painter,
                info_rect,
                facecolor=option.palette.base(),
                edgecolor=color,
                linewidth=self.icon_border_width,
                radius=self.icon_corner_radius,
            )

            # Draw text
            painter.setPen(color)
            painter.drawText(
                info_rect.toRect(),
                QtCore.Qt.AlignmentFlag.AlignCenter
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
                info_text,
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
            self._show_popup(*tool_wrapper._preview_image, option)

    def _paint_childtool(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        view = typing.cast("_ImageToolWrapperTreeView", self.parent())
        is_editing: bool = (
            view.state() == QtWidgets.QAbstractItemView.State.EditingState
            and view.currentIndex() == index
        )

        selected: bool = QtWidgets.QStyle.StateFlag.State_Selected in option.state

        if not is_editing:  # pragma: no branch
            role = (
                QtGui.QPalette.ColorRole.HighlightedText
                if selected
                else QtGui.QPalette.ColorRole.Text
            )
            painter.setPen(option.palette.color(role))

            # Elide text if necessary
            elided_text = QtGui.QFontMetrics(option.font).elidedText(
                index.data(role=QtCore.Qt.ItemDataRole.DisplayRole),
                view.textElideMode(),
                option.rect.width(),
            )
            painter.drawText(
                option.rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft,
                elided_text,
            )

        # Show preview on hover
        if (
            not is_editing
            and self.manager.preview_action.isChecked()
            and (
                QtWidgets.QStyle.StateFlag.State_MouseOver in option.state
                or self._force_hover
            )
        ):
            child_tool = self.manager.get_childtool(index.internalPointer())
            image_item = child_tool.preview_imageitem
            if image_item is not None:
                vb_rect = image_item.getViewBox().rect()
                self._show_popup(
                    vb_rect.height() / vb_rect.width(),
                    image_item.getPixmap().transformed(
                        QtGui.QTransform().scale(1.0, -1.0)
                    ),
                    option,
                )

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if event is not None:  # pragma: no branch
            match event.type():
                case (
                    QtCore.QEvent.Type.Resize
                    | QtCore.QEvent.Type.Leave
                    | QtCore.QEvent.Type.WindowStateChange
                ):
                    self.preview_popup.hide()
                case QtCore.QEvent.Type.MouseMove:
                    index = typing.cast(
                        "_ImageToolWrapperTreeView", self.parent()
                    ).indexAt(typing.cast("QtGui.QMouseEvent", event).pos())
                    if not index.isValid():
                        self.preview_popup.hide()

        return super().eventFilter(obj, event)

    def _cleanup_filter(self) -> None:
        """Remove the event filter from the viewport."""
        viewport = typing.cast("_ImageToolWrapperTreeView", self.parent()).viewport()
        if viewport is not None:  # pragma: no branch
            viewport.removeEventFilter(self)


_MIME = "application/x-imagetool-manager-internal-move"


class _ImageToolWrapperItemModel(QtCore.QAbstractItemModel):
    def __init__(self, manager: ImageToolManager, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._manager = weakref.ref(manager)

    @property
    def manager(self) -> ImageToolManager:
        manager = self._manager()
        if manager:
            return manager
        raise LookupError("Parent was destroyed")

    def _imagetool_index(self, row_index: QtCore.QModelIndex | int) -> int:
        if isinstance(row_index, QtCore.QModelIndex):
            row_index = row_index.row()
        return self.manager._displayed_indices[row_index]

    def _imagetool_wrapper(
        self, row_index: QtCore.QModelIndex | int
    ) -> _ImageToolWrapper:
        if isinstance(row_index, QtCore.QModelIndex):
            ptr = row_index.internalPointer()
            if not isinstance(ptr, _ImageToolWrapper):
                raise KeyError("Index does not point to a tool wrapper")
            return ptr
        return self.manager._imagetool_wrappers[self._imagetool_index(row_index)]

    def _childtool_uid(
        self, row_index: QtCore.QModelIndex | int, parent_wrapper: _ImageToolWrapper
    ) -> str:
        if isinstance(row_index, QtCore.QModelIndex):
            row_index = row_index.row()
        return parent_wrapper._childtool_indices[row_index]

    def _childtool(
        self, row_index: QtCore.QModelIndex, parent_wrapper: _ImageToolWrapper
    ) -> erlab.interactive.utils.ToolWindow:
        return parent_wrapper._childtools[
            self._childtool_uid(row_index, parent_wrapper)
        ]

    def _row_index(self, imagetool_index: int) -> QtCore.QModelIndex:
        return self.index(self.manager._displayed_indices.index(imagetool_index), 0)

    def _is_archived(self, row_index: QtCore.QModelIndex) -> bool:
        return self._imagetool_wrapper(row_index).archived

    def index(
        self, row: int, column: int, parent: QtCore.QModelIndex | None = None
    ) -> QtCore.QModelIndex:
        if column != 0 or row < 0:
            return QtCore.QModelIndex()
        if parent is None:
            parent = QtCore.QModelIndex()

        if not parent.isValid():
            # Top-level; ImageTool
            if row >= len(self.manager._displayed_indices):
                return QtCore.QModelIndex()
            wrapper = self._imagetool_wrapper(row)
            return self.createIndex(row, column, wrapper)

        ptr = parent.internalPointer()
        if not isinstance(ptr, _ImageToolWrapper):
            return QtCore.QModelIndex()

        if row >= len(ptr._childtool_indices):
            return QtCore.QModelIndex()

        child = self._childtool_uid(row, ptr)
        return self.createIndex(row, column, child)

    @typing.overload
    def parent(self, child: QtCore.QModelIndex) -> QtCore.QModelIndex: ...

    @typing.overload
    def parent(self) -> QtCore.QObject | None: ...

    def parent(
        self, child: QtCore.QModelIndex | None = None
    ) -> QtCore.QModelIndex | QtCore.QObject | None:
        if child is None:
            return super().parent()

        if not child.isValid():
            return QtCore.QModelIndex()

        uid = child.internalPointer()
        if isinstance(uid, str):
            # Child tool
            for tool_idx, wrapper in self.manager._imagetool_wrappers.items():
                if uid in wrapper._childtools:
                    return self._row_index(tool_idx)
        return QtCore.QModelIndex()

    def hasChildren(self, parent: QtCore.QModelIndex | None = None) -> bool:
        if parent is None:  # pragma: no branch
            parent = QtCore.QModelIndex()

        if not parent.isValid():
            return len(self.manager._displayed_indices) > 0

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            return len(ptr._childtool_indices) > 0
        return False  # Child tool has no children

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        if parent is None:  # pragma: no branch
            parent = QtCore.QModelIndex()

        if not parent.isValid():
            # Top-level; ImageTool
            return len(self.manager._displayed_indices)

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            # Number of child tools
            return len(ptr._childtool_indices)
        return 0  # Child tool has no children

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return 1

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole
    ) -> typing.Any:
        if not index.isValid():  # pragma: no branch
            return None

        ptr = index.internalPointer()
        if isinstance(ptr, str):
            return self._data_childtool(index, role)
        return self._data_imagetool(index, role)

    def _data_imagetool(self, index: QtCore.QModelIndex, role: int) -> typing.Any:
        tool_idx: int = self._imagetool_index(index)

        match role:
            case QtCore.Qt.ItemDataRole.DisplayRole:
                if tool_idx < 0:
                    return ""
                return self.manager.label_of_imagetool(tool_idx)

            case QtCore.Qt.ItemDataRole.EditRole:
                if tool_idx < 0:
                    return ""
                return self.manager.name_of_imagetool(tool_idx)

            case QtCore.Qt.ItemDataRole.SizeHintRole:
                return QtCore.QSize(100, 30)

        return None

    def _data_childtool(self, index: QtCore.QModelIndex, role: int) -> typing.Any:
        # Child tool, get wrapper
        parent_wrapper = typing.cast(
            "_ImageToolWrapper", self.parent(index).internalPointer()
        )
        match role:
            case QtCore.Qt.ItemDataRole.DisplayRole:
                return self._childtool(index, parent_wrapper).windowTitle()

            case QtCore.Qt.ItemDataRole.EditRole:
                return self._childtool(index, parent_wrapper)._tool_display_name

            case QtCore.Qt.ItemDataRole.SizeHintRole:
                return QtCore.QSize(100, 30)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            # Allow drops at root for top-level reordering
            return (
                QtCore.Qt.ItemFlag.ItemIsDropEnabled | QtCore.Qt.ItemFlag.ItemIsEnabled
            )

        node = index.internalPointer()

        default_flags = (
            QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        if isinstance(node, _ImageToolWrapper):
            # Only parents accept drops (children are reordered via the parent)
            flags = default_flags | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            if not self._is_archived(index):
                # ImageTool, not archived
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            return flags

        # Child items are draggable, but not drop targets
        return default_flags | QtCore.Qt.ItemFlag.ItemIsEditable

    def supportedDragActions(self) -> QtCore.Qt.DropAction:
        return QtCore.Qt.DropAction.MoveAction

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        return QtCore.Qt.DropAction.MoveAction

    def _insert_imagetool(self, index: int) -> None:
        """Append a new tool to the end of the model.

        This must be called after the ImageTool is added to the manager.
        """
        n_rows = self.rowCount()

        self.beginInsertRows(QtCore.QModelIndex(), n_rows, n_rows)  # Insert at end
        self.manager._displayed_indices.insert(n_rows, index)
        self.endInsertRows()

    def _insert_childtool(self, uid: str, parent_idx: int) -> None:
        """Append a new tool to the end of the model.

        This must be called after the ImageTool is added to the manager.
        """
        n_rows = self.rowCount()

        parent = self._row_index(parent_idx)
        self.beginInsertRows(parent, n_rows, n_rows)  # Insert at end

        ptr = typing.cast("_ImageToolWrapper", parent.internalPointer())
        ptr._childtool_indices.insert(n_rows, uid)

        self.endInsertRows()

    def remove_rows(
        self, row: int, count: int, parent: QtCore.QModelIndex | None = None
    ) -> bool:
        """Remove rows from the model.

        Has the same signature as :meth:`QtCore.QAbstractItemModel.removeRows`.

        We do not implement this as `removeRows()` in order to avoid Qt automatically
        calling it after drag-and-drop.

        """
        if parent is None:  # pragma: no branch
            parent = QtCore.QModelIndex()

        if not parent.isValid():
            # Top-level; ImageTool
            self.beginRemoveRows(parent, row, row + count - 1)
            del self.manager._displayed_indices[row : row + count]
            self.endRemoveRows()
            return True

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            self.beginRemoveRows(parent, row, row + count - 1)
            del ptr._childtool_indices[row : row + count]
            self.endRemoveRows()
            return True

        return False

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: typing.Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():  # pragma: no branch
            return False

        ptr = index.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            if role == QtCore.Qt.ItemDataRole.EditRole:
                self.manager.rename_imagetool(self._imagetool_index(index), value)

                self.dataChanged.emit(index, index, [role])
                return True

        elif isinstance(ptr, str):
            # Child tool, get wrapper
            parent_wrapper = typing.cast(
                "_ImageToolWrapper", self.parent(index).internalPointer()
            )
            if role == QtCore.Qt.ItemDataRole.EditRole:
                child_tool = self._childtool(index, parent_wrapper)
                child_tool._tool_display_name = value

                self.dataChanged.emit(index, index, [role])
                return True

        return False

    def mimeTypes(self):
        return [_MIME]

    def mimeData(self, indexes: Iterable[QtCore.QModelIndex]) -> QtCore.QMimeData:
        # Collect unique (row, parent_ptr, level) from column 0 only.
        rows: list[int] = []
        parent_pointer = None
        level: int = -1

        for idx in indexes:
            if not idx.isValid() or idx.column() != 0:
                continue
            node = idx.internalPointer()
            if isinstance(node, _ImageToolWrapper):
                current_level = 0
                current_parent_pointer = None  # root
            else:
                current_level = 1
                current_parent: _ImageToolWrapper = typing.cast(
                    "_ImageToolWrapper", self.parent(idx).internalPointer()
                )
                current_parent_pointer = id(current_parent)

            if level == -1:
                level = current_level
                parent_pointer = current_parent_pointer
            else:
                # Enforce single level and single parent per drag
                if current_level != level or current_parent_pointer != parent_pointer:
                    return QtCore.QMimeData()  # refuse mixed drags

            rows.append(idx.row())

        if not rows or level == -1:
            return QtCore.QMimeData()

        mime_data = QtCore.QMimeData()
        mime_data.setData(
            _MIME,
            QtCore.QByteArray(
                json.dumps(
                    {
                        "level": level,
                        "parent_id": parent_pointer,
                        "rows": sorted(set(rows)),
                    }
                ).encode("utf-8")
            ),
        )
        return mime_data

    @staticmethod
    def _decode_mime(mime: QtCore.QMimeData) -> dict[str, typing.Any] | None:
        raw: bytes = mime.data(_MIME).data()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        if not {"level", "parent_id", "rows"} <= payload.keys():
            return None
        return payload

    @staticmethod
    def _contiguous_runs(rows: list[int]) -> list[tuple[int, int]]:
        """Convert sorted rows to list of (start, length)."""
        runs: list[tuple[int, int]] = []
        if not rows:
            return runs
        start = prev = rows[0]
        for r in rows[1:]:
            if r == prev + 1:
                prev = r
                continue
            runs.append((start, prev - start + 1))
            start = prev = r
        runs.append((start, prev - start + 1))
        return runs

    @staticmethod
    def _drop_is_noop(rows: list[int], dest: int) -> bool:
        """Check if dropping rows at dest is a no-op."""
        if not rows:
            return True
        return dest == rows[0] or dest == rows[-1] + 1

    def canDropMimeData(
        self,
        data: QtCore.QMimeData | None,
        action: QtCore.Qt.DropAction,
        row: int,
        column: int,
        parent: QtCore.QModelIndex,
    ) -> bool:
        if data is None:
            logger.debug("canDropMimeData: no data")
            return False
        if action != QtCore.Qt.DropAction.MoveAction or column > 0:
            logger.debug("canDropMimeData: wrong action/column")
            return False
        if _MIME not in data.formats():
            logger.debug("canDropMimeData: wrong mime type")
            return False

        payload = self._decode_mime(data)
        if payload is None:
            logger.debug("canDropMimeData: cannot decode mime")
            return False

        # Destination parent must match source constraints
        if payload["level"] == 0:
            # top-level → must drop at root
            if isinstance(parent.internalPointer(), str):
                logger.debug("canDropMimeData: level 0 but parent is child")
                return False
            return True

        # children → must drop on their same TopItem parent
        if not parent.isValid():
            logger.debug("canDropMimeData: level 1 but no parent")
            return False

        destination_node = parent.internalPointer()
        if not isinstance(destination_node, _ImageToolWrapper):
            logger.debug("canDropMimeData: level 1 but parent is not valid type")
            return False
        if id(destination_node) != payload["parent_id"]:
            logger.debug("canDropMimeData: parent mismatch")
            return False
        return True

    def _apply_target(
        self,
        level: typing.Literal[0, 1],
        values: list[int] | list[str],
        parent: QtCore.QModelIndex,
    ) -> None:
        """Apply the newly ordered list.

        Parameters
        ----------
        level
            0 for top-level ImageTool, 1 for child tools.
        values
            The new ordered list (int for ImageTool, str for child tools).
        parent
            The parent index, unused for level 0. If level is 1, this is the QModelIndex
            of the wrapper containing the child tools.
        """
        if level == 0:
            self.manager._displayed_indices = values  # type: ignore[assignment]
        else:
            parent.internalPointer()._childtool_indices = values

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

        payload = self._decode_mime(data)
        if payload is None:
            return False
        level = payload["level"]
        source_rows: list[int] = payload["rows"]

        # Destination list and parent index
        if level == 0:
            parent_index = QtCore.QModelIndex()
            target: list[str] | list[int] = list(self.manager._displayed_indices)
            if parent.isValid():
                # Dropping on a parent, adjust row to be relative to that item
                row = parent.row() + 1
        else:
            destination_parent = parent.internalPointer()
            parent_index = parent
            target = list(destination_parent._childtool_indices)

        if not target or not source_rows:
            logger.debug("dropMimeData: empty target or source")
            return False

        # Compute insertion position
        dest = len(target) if row < 0 or row > len(target) else row
        logger.debug("dropMimeData: level=%s, rows=%s -> %s", level, source_rows, dest)

        # # Early exit if dropping doesn't change anything
        if self._drop_is_noop(source_rows, dest):
            logger.debug("dropMimeData: no-op")
            return True

        # Build contiguous runs from source rows
        runs: list[tuple[int, int]] = self._contiguous_runs(sorted(source_rows))

        first_start: int = runs[0][0]
        moving_up: bool = dest <= first_start

        if moving_up:
            # Process from the top, change dest for each iteration
            # Keep runs updated as we move them
            mutable_runs: list[list[int]] = [list(r) for r in runs]
            for i in range(len(mutable_runs)):
                start, length = mutable_runs[i]
                logger.debug(
                    "dropMimeData: start=%s, length=%s, dest=%s", start, length, dest
                )
                self.beginMoveRows(
                    parent_index, start, start + length - 1, parent_index, dest
                )

                logger.debug("dropMimeData: original %s", target)
                block = target[start : start + length]
                del target[start : start + length]
                for j in range(i + 1, len(mutable_runs)):
                    # Subsequent starts shift left if they were after the removed block
                    if mutable_runs[j][0] > start:
                        mutable_runs[j][0] -= length
                target[dest:dest] = block  # type: ignore[assignment]
                logger.debug("dropMimeData: modified %s", target)
                self._apply_target(level, target, parent_index)

                self.endMoveRows()
                dest += length  # Next block goes after the previous

        else:
            # Process from bottom, keep dest constant
            for start, length in reversed(runs):
                logger.debug(
                    "dropMimeData: start=%s, length=%s, dest=%s", start, length, dest
                )
                self.beginMoveRows(
                    parent_index, start, start + length - 1, parent_index, dest
                )

                logger.debug("dropMimeData: original %s", target)
                block = target[start : start + length]
                del target[start : start + length]
                target[dest:dest] = block  # type: ignore[assignment]
                logger.debug("dropMimeData: modified %s", target)
                self._apply_target(level, target, parent_index)

                self.endMoveRows()

        logger.debug("dropMimeData: success")
        return True


class _ImageToolWrapperTreeView(QtWidgets.QTreeView):
    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__()

        self._model = _ImageToolWrapperItemModel(manager, self)
        self.setModel(self._model)

        self._delegate = _ImageToolWrapperItemDelegate(manager, self)
        self.setItemDelegate(self._delegate)

        self._selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self.selectionModel()
        )

        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        # Enable drag & drop for reordering items
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setUniformRowHeights(True)
        self.setAnimated(True)

        self.setEditTriggers(
            self.EditTrigger.EditKeyPressed
        )  # Only allow editing on edit key press (depends on platform)

        self.setWordWrap(True)  # Ellide text when width is too small
        self.setMouseTracking(True)  # Enable hover detection

        typing.cast("QtWidgets.QHeaderView", self.header()).hide()  # Hide header

        # Show tool on double-click
        self.doubleClicked.connect(self._model.manager.show_selected)

        # Right-click context menu
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self._menu = QtWidgets.QMenu("Menu", self)
        self._menu.setToolTipsVisible(True)
        self._menu.addAction(manager.reindex_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.concat_action)
        self._menu.addAction(manager.duplicate_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.show_action)
        self._menu.addAction(manager.hide_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.remove_action)
        self._menu.addAction(manager.unwatch_action)
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
    def selected_imagetool_indices(self) -> list[int]:
        """Currently selected ImageTool indices.

        Ignores any child tools that may be selected.

        The tools are ordered by their position in the list view.
        """
        row_indices = sorted(
            index.row()
            for index in self.selectedIndexes()
            if isinstance(index.internalPointer(), _ImageToolWrapper)
        )
        return [self._model.manager._displayed_indices[i] for i in row_indices]

    @property
    def selected_childtool_uids(self) -> list[str]:
        """UIDs of currently selected child tools."""
        return [
            index.internalPointer()
            for index in self.selectedIndexes()
            if isinstance(index.internalPointer(), str)
        ]

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
            top = self._model.index(0, 0)
            bottom = self._model.index(self._model.rowCount() - 1, 0)
            if top.isValid() and bottom.isValid():  # pragma: no branch
                self._model.dataChanged.emit(top, bottom)
        else:
            if idx in self._model.manager._displayed_indices:
                row_idx = self._model._row_index(idx)
                if row_idx.isValid():  # pragma: no branch
                    self._model.dataChanged.emit(row_idx, row_idx)

    def imagetool_added(self, index: int) -> None:
        """Update the list view when a new ImageTool is added to the manager.

        This must be called after the ImageTool is added to the manager.
        """
        self._model._insert_imagetool(index)

    def imagetool_removed(self, index: int) -> None:
        """Update the list view when removing an ImageTool from the manager.

        This must be called before the ImageTool is removed from the manager.
        """
        for i, tool_idx in enumerate(
            self._model.manager._displayed_indices
        ):  # pragma: no branch
            if tool_idx == index:  # pragma: no branch
                self._model.remove_rows(i, 1)
                break

    def childtool_added(self, uid: str, parent_idx: int) -> None:
        """Update the list view when a new child tool is added to the manager.

        This must be called after the child tool is added to the manager.
        """
        logger.debug("Adding child tool %s to parent index %s", uid, parent_idx)
        self._model._insert_childtool(uid, parent_idx)

    def childtool_removed(self, uid: str) -> None:
        """Update the list view when removing a child tool from the manager.

        This must be called before the child tool is removed from the manager.
        """
        for tool_idx, wrapper in self._model.manager._imagetool_wrappers.items():
            if uid in wrapper._childtool_indices:
                parent_index = self._model._row_index(tool_idx)
                for i, child_uid in enumerate(wrapper._childtool_indices):
                    if child_uid == uid:  # pragma: no branch
                        self._model.remove_rows(i, 1, parent_index)
                        return
