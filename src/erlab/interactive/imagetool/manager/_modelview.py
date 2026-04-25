"""Model-view architecture used for displaying the list of ImageTool windows."""

from __future__ import annotations

__all__ = ["_ImageToolWrapperTreeView"]

import functools
import json
import logging
import math
import os
import typing
import weakref

import qtawesome as qta
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from erlab.interactive.imagetool.manager import ImageToolManager

logger = logging.getLogger(__name__)

_NODE_UID_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 128
_TOOL_TYPE_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 129


def _fill_rounded_rect(
    painter: QtGui.QPainter,
    rect: QtCore.QRect | QtCore.QRectF,
    facecolor: QtGui.QColor | QtGui.QBrush,
    edgecolor: QtGui.QColor | QtGui.QBrush,
    linewidth: float,
    radius: float,
):
    painter.save()
    painter.setRenderHints(
        QtGui.QPainter.RenderHint.Antialiasing
        | QtGui.QPainter.RenderHint.SmoothPixmapTransform
    )
    rect = QtCore.QRectF(rect)
    rect.adjust(linewidth / 2, linewidth / 2, -linewidth / 2, -linewidth / 2)
    path = QtGui.QPainterPath()
    path.addRoundedRect(rect, radius, radius)

    painter.setClipPath(path)
    painter.fillPath(path, QtGui.QBrush(facecolor))
    painter.setPen(QtGui.QPen(edgecolor, linewidth))
    painter.drawPath(path)
    painter.restore()


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

    icon_size: int = 12
    icon_right_pad: int = 2
    icon_inner_pad: int = 2
    icon_border_width: float = 2.0
    icon_corner_radius: float = 3.0

    watched_rect_hpad: int = 5
    watched_font_scale: float = 0.9
    tool_type_rect_gap: int = 5
    tool_type_rect_hpad: int = 5
    tool_type_font_scale: float = 0.85
    child_status_rect_hpad: int = 5
    child_status_font_scale: float = 0.85

    def __init__(
        self, manager: ImageToolManager, parent: _ImageToolWrapperTreeView
    ) -> None:
        super().__init__(parent)
        self._manager = weakref.ref(manager)
        self._font_size = QtGui.QFont().pointSize()
        self._current_editor: QtWidgets.QLineEdit | None = None

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
        view = typing.cast("_ImageToolWrapperTreeView", self.parent())
        if parent is None:
            parent = view.viewport()
        editor = erlab.interactive.utils.ResizingLineEdit(parent)
        editor.setFont(option.font)
        editor.setFrame(True)
        editor.setPlaceholderText("Enter new name")
        self._current_editor = editor
        editor.destroyed.connect(self._clear_current_editor)
        return editor

    def updateEditorGeometry(
        self,
        editor: QtWidgets.QWidget | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        if editor is not None:  # pragma: no branch
            option.font.setPointSize(self._font_size)
            rect = QtCore.QRectF(option.rect)
            ptr = index.internalPointer()
            if isinstance(ptr, str):
                try:
                    child_node = self.manager._child_node(ptr)
                except KeyError:
                    child_node = None
                if child_node is not None:
                    type_rect, _, _ = self._compute_tool_type_info(option, child_node)
                    if type_rect is not None:
                        rect.setLeft(type_rect.right() + self.tool_type_rect_gap + 1)
                    status_rect, _, _ = self._compute_child_status_info(
                        option, child_node
                    )
                    if status_rect is not None:
                        rect.setRight(status_rect.left() - self.icon_right_pad)
            rect.setTop(rect.center().y() - editor.sizeHint().height() / 2)
            editor.setGeometry(rect.toRect())

    def destroyEditor(
        self, editor: QtWidgets.QWidget | None, index: QtCore.QModelIndex
    ) -> None:
        if self._current_editor is editor:
            self._current_editor = None
        super().destroyEditor(editor, index)

    @QtCore.Slot()
    def _clear_current_editor(self) -> None:
        self._current_editor = None

    def _show_popup(
        self,
        box_ratio: float,
        pixmap: QtGui.QPixmap,
        option: QtWidgets.QStyleOptionViewItem,
    ) -> None:
        if pixmap.isNull() or not math.isfinite(box_ratio) or box_ratio <= 0:
            self.preview_popup.hide()
            return

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

    def _compute_icons_info(
        self, option: QtWidgets.QStyleOptionViewItem, tool_wrapper: _ImageToolWrapper
    ) -> tuple[
        int,
        QtCore.QRect | None,
        QtCore.QRect | None,
        QtCore.QRect | None,
    ]:
        # Determine which icons should be visible
        is_linked: bool = (
            not tool_wrapper.archived and tool_wrapper.slicer_area.is_linked
        )
        is_watched: bool = tool_wrapper._watched_varname is not None
        is_dask: bool = (
            not tool_wrapper.archived and tool_wrapper.slicer_area.data_chunked
        )

        # Precompute geometry constants
        icon_size = self.icon_size
        rect_size = icon_size + 2 * self.icon_inner_pad
        rect_dx = rect_size + self.icon_right_pad

        rect_y = option.rect.center().y() - (rect_size // 2)

        dask_rect: QtCore.QRect | None = None
        link_rect: QtCore.QRect | None = None
        watched_rect: QtCore.QRect | None = None

        rect_x = option.rect.right()

        # Dask icon
        if is_dask:
            rect_x -= rect_dx
            dask_rect = QtCore.QRect(rect_x, rect_y, rect_size, rect_size)

        # Link indicator
        if is_linked:
            rect_x -= rect_dx
            link_rect = QtCore.QRect(rect_x, rect_y, rect_size, rect_size)

        # Watched variable name label
        if is_watched:
            watched_text: str = typing.cast("str", tool_wrapper._watched_varname)

            # Use a smaller font for the watched label
            watched_font = QtGui.QFont(option.font)
            watched_font.setPointSizeF(self._font_size * self.watched_font_scale)
            watched_width = (
                QtGui.QFontMetrics(watched_font).boundingRect(watched_text).width()
                + self.watched_rect_hpad * 2
            )

            rect_x -= watched_width + self.icon_right_pad
            watched_rect = QtCore.QRect(rect_x, rect_y, watched_width, rect_size)

        icons_width = option.rect.right() - rect_x

        return icons_width, dask_rect, link_rect, watched_rect

    def _paint_icon(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        icon_rect: QtCore.QRect,
        icon: QtGui.QIcon,
    ) -> None:
        """Paint an icon inside a rounded square."""
        _fill_rounded_rect(
            painter,
            icon_rect,
            facecolor=option.palette.base(),
            edgecolor=option.palette.mid(),
            linewidth=self.icon_border_width,
            radius=self.icon_corner_radius,
        )
        icon.paint(
            painter,
            icon_rect.adjusted(
                self.icon_inner_pad,
                self.icon_inner_pad,
                -self.icon_inner_pad,
                -self.icon_inner_pad,
            ),
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
        )

    @functools.cached_property
    def _dask_icon(self) -> QtGui.QIcon:
        return QtGui.QIcon(os.path.join(os.path.dirname(__file__), "dask.png"))

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
        tool_wrapper: _ImageToolWrapper = typing.cast(
            "_ImageToolWrapper", index.internalPointer()
        )
        view = typing.cast("_ImageToolWrapperTreeView", self.parent())

        is_editing: bool = (
            view.state() == QtWidgets.QAbstractItemView.State.EditingState
            and view.currentIndex() == index
        )

        # Precompute icon geometry
        icons_width, dask_rect, link_rect, watched_rect = self._compute_icons_info(
            option, tool_wrapper
        )

        # Draw label (skip while editing for inline editor)
        if not is_editing:  # pragma: no branch
            palette = option.palette
            if tool_wrapper.archived:
                color_group = QtGui.QPalette.ColorGroup.Disabled
            else:
                color_group = (
                    QtGui.QPalette.ColorGroup.Active
                    if QtWidgets.QStyle.StateFlag.State_Active in option.state
                    else QtGui.QPalette.ColorGroup.Inactive
                )
            role = (
                QtGui.QPalette.ColorRole.HighlightedText
                if QtWidgets.QStyle.StateFlag.State_Selected in option.state
                else QtGui.QPalette.ColorRole.Text
            )

            # Elide text to leave room for icons on the right
            fm = QtGui.QFontMetrics(option.font)
            text = index.data(role=QtCore.Qt.ItemDataRole.DisplayRole)
            elided = fm.elidedText(
                text,
                view.textElideMode(),
                option.rect.width() - self.icon_right_pad - icons_width,
            )
            painter.setPen(palette.color(color_group, role))
            painter.drawText(
                option.rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft,
                elided,
            )

        # Icons
        if dask_rect:
            self._paint_icon(painter, option, dask_rect, self._dask_icon)

        if link_rect:
            proxy = typing.cast(
                "erlab.interactive.imagetool.viewer.SlicerLinkProxy",
                tool_wrapper.slicer_area._linking_proxy,
            )
            link_color = self.manager.color_for_linker(proxy)
            link_icon = qta.icon("mdi6.link-variant", color=link_color)
            self._paint_icon(painter, option, link_rect, link_icon)

        if watched_rect:
            palette = option.palette
            watched_varname = str(tool_wrapper._watched_varname)
            watched_uid = str(tool_wrapper._watched_uid)
            kernel_uid = watched_uid.removeprefix(f"{watched_varname} ")
            color = self.manager.color_for_watched_var_kernel(kernel_uid)

            _fill_rounded_rect(
                painter,
                watched_rect,
                facecolor=palette.base(),
                edgecolor=color,
                linewidth=self.icon_border_width,
                radius=self.icon_corner_radius,
            )
            watched_font = QtGui.QFont(option.font)
            watched_font.setPointSizeF(self._font_size * self.watched_font_scale)
            painter.save()
            painter.setFont(watched_font)
            painter.setPen(color)
            painter.drawText(
                watched_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignCenter,
                watched_varname,
            )
            painter.restore()

        # Preview popup (hover)
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
        child_node: _ManagedWindowNode | None = None
        type_rect: QtCore.QRect | None = None
        type_text: str | None = None
        type_color: QtGui.QColor | None = None
        status_rect: QtCore.QRect | None = None
        status_text: str | None = None
        status_color: QtGui.QColor | None = None
        try:
            child_node = self.manager._child_node(index.internalPointer())
        except KeyError:
            child_node = None
        else:
            type_rect, type_text, type_color = self._compute_tool_type_info(
                option, child_node
            )
            status_rect, status_text, status_color = self._compute_child_status_info(
                option, child_node
            )

        if type_rect and type_text and type_color:
            _fill_rounded_rect(
                painter,
                type_rect,
                facecolor=option.palette.base(),
                edgecolor=type_color,
                linewidth=self.icon_border_width,
                radius=self.icon_corner_radius,
            )
            type_font = QtGui.QFont(option.font)
            type_font.setPointSizeF(self._font_size * self.tool_type_font_scale)
            painter.save()
            painter.setFont(type_font)
            painter.setPen(type_color)
            painter.drawText(
                type_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignCenter,
                type_text,
            )
            painter.restore()

        if not is_editing:  # pragma: no branch
            role = (
                QtGui.QPalette.ColorRole.HighlightedText
                if selected
                else QtGui.QPalette.ColorRole.Text
            )
            if child_node is not None and child_node.archived and not selected:
                role = QtGui.QPalette.ColorRole.Mid
            painter.setPen(option.palette.color(role))

            text_rect = QtCore.QRect(option.rect)
            if type_rect is not None:
                text_rect.setLeft(type_rect.right() + self.tool_type_rect_gap + 1)
            if status_rect is not None:
                text_rect.setRight(status_rect.left() - self.icon_right_pad)

            # Elide text if necessary
            elided_text = QtGui.QFontMetrics(option.font).elidedText(
                index.data(role=QtCore.Qt.ItemDataRole.DisplayRole),
                view.textElideMode(),
                max(text_rect.width(), 0),
            )
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft,
                elided_text,
            )
            if status_rect and status_text and status_color:
                _fill_rounded_rect(
                    painter,
                    status_rect,
                    facecolor=option.palette.base(),
                    edgecolor=status_color,
                    linewidth=self.icon_border_width,
                    radius=self.icon_corner_radius,
                )
                status_font = QtGui.QFont(option.font)
                status_font.setPointSizeF(
                    self._font_size * self.child_status_font_scale
                )
                painter.save()
                painter.setFont(status_font)
                painter.setPen(status_color)
                painter.drawText(
                    status_rect,
                    QtCore.Qt.AlignmentFlag.AlignVCenter
                    | QtCore.Qt.AlignmentFlag.AlignCenter,
                    status_text,
                )
                painter.restore()

        # Show preview on hover
        if (
            not is_editing
            and self.manager.preview_action.isChecked()
            and (
                QtWidgets.QStyle.StateFlag.State_MouseOver in option.state
                or self._force_hover
            )
        ):
            if child_node is None:
                self.preview_popup.hide()
                return

            if child_node.imagetool is not None:
                self._show_popup(*child_node._preview_image, option)
                return

            image_item = (
                child_node.tool_window.preview_imageitem
                if child_node.tool_window is not None
                else None
            )
            if image_item is None or not erlab.interactive.utils.qt_is_valid(
                image_item
            ):
                self.preview_popup.hide()
                return

            view_box = image_item.getViewBox()
            if not erlab.interactive.utils.qt_is_valid(view_box):
                self.preview_popup.hide()
                return

            vb_rect = view_box.rect()
            width = vb_rect.width()
            height = vb_rect.height()
            if width <= 0 or height <= 0:
                self.preview_popup.hide()
                return

            try:
                pixmap = image_item.getPixmap()
            except RuntimeError:
                self.preview_popup.hide()
                return

            self._show_popup(
                height / width,
                pixmap.transformed(QtGui.QTransform().scale(1.0, -1.0)),
                option,
            )

    def _compute_tool_type_info(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        node: _ManagedWindowNode,
    ) -> tuple[QtCore.QRect | None, str | None, QtGui.QColor | None]:
        text = node.type_badge_text
        if not text or node.display_text == text:
            return None, None, None

        rect_size = self.icon_size + 2 * self.icon_inner_pad
        rect_y = option.rect.center().y() - (rect_size // 2)
        badge_font = QtGui.QFont(option.font)
        badge_font.setPointSizeF(self._font_size * self.tool_type_font_scale)
        badge_width = (
            QtGui.QFontMetrics(badge_font).boundingRect(text).width()
            + self.tool_type_rect_hpad * 2
        )
        rect = QtCore.QRect(option.rect.left(), rect_y, badge_width, rect_size)
        return rect, text, option.palette.color(QtGui.QPalette.ColorRole.Mid)

    def _compute_child_status_info(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        child_node: _ManagedWindowNode,
    ) -> tuple[QtCore.QRect | None, str | None, QtGui.QColor | None]:
        match child_node.source_state:
            case "stale":
                text = "Stale"
                color = QtGui.QColor("#b26a00")
            case "unavailable":
                text = "Unavailable"
                color = QtGui.QColor("#b24444")
            case "fresh" if child_node.source_auto_update:
                text = "Auto"
                color = QtGui.QColor("#59636e")
            case _:
                return None, None, None

        rect_size = self.icon_size + 2 * self.icon_inner_pad
        rect_y = option.rect.center().y() - (rect_size // 2)
        badge_font = QtGui.QFont(option.font)
        badge_font.setPointSizeF(self._font_size * self.child_status_font_scale)
        badge_width = (
            QtGui.QFontMetrics(badge_font).boundingRect(text).width()
            + self.child_status_rect_hpad * 2
        )
        rect_x = option.rect.right() - badge_width - self.icon_right_pad
        return QtCore.QRect(rect_x, rect_y, badge_width, rect_size), text, color

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

    def helpEvent(
        self,
        event: QtGui.QHelpEvent | None,
        view: QtWidgets.QAbstractItemView | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> bool:
        if isinstance(event, QtGui.QHelpEvent) and index.isValid():
            tool_wrapper = index.internalPointer()
            if isinstance(tool_wrapper, _ImageToolWrapper):  # pragma: no branch
                (
                    _,
                    dask_rect,
                    link_rect,
                    watched_rect,
                ) = self._compute_icons_info(option, tool_wrapper)
                pos = event.pos()
                if dask_rect and dask_rect.contains(pos):
                    QtWidgets.QToolTip.showText(
                        event.globalPos(),
                        "Dask-backed data (chunked array)",
                        view,
                        dask_rect,
                    )
                    return True
                if link_rect and link_rect.contains(pos):
                    proxy = tool_wrapper.slicer_area._linking_proxy
                    if proxy:  # pragma: no branch
                        linker_index = self.manager._linkers.index(proxy)
                        QtWidgets.QToolTip.showText(
                            event.globalPos(),
                            f"Linked (#{linker_index})",
                            view,
                            link_rect,
                        )
                        return True
                if watched_rect and watched_rect.contains(pos):
                    QtWidgets.QToolTip.showText(
                        event.globalPos(),
                        "Variable synced with IPython",
                        view,
                        watched_rect,
                    )
                    return True
            elif isinstance(tool_wrapper, str):
                try:
                    child_node = self.manager._child_node(tool_wrapper)
                except KeyError:
                    child_node = None
                if child_node is not None:
                    type_rect, type_text, _ = self._compute_tool_type_info(
                        option, child_node
                    )
                    if type_rect and type_text and type_rect.contains(event.pos()):
                        QtWidgets.QToolTip.showText(
                            event.globalPos(),
                            f"Tool type: {type_text}",
                            view,
                            type_rect,
                        )
                        return True
                    status_rect, _, _ = self._compute_child_status_info(
                        option, child_node
                    )
                    if status_rect and status_rect.contains(event.pos()):
                        match child_node.source_state:
                            case "stale":
                                tooltip = (
                                    "Click to update this tool from the latest "
                                    "compatible data."
                                )
                            case "unavailable":
                                tooltip = (
                                    "Click to review why this tool cannot update from "
                                    "the current data."
                                )
                            case _:
                                tooltip = "Click to configure automatic updates."
                        QtWidgets.QToolTip.showText(
                            event.globalPos(), tooltip, view, status_rect
                        )
                        return True

        return super().helpEvent(event, view, option, index)


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

    def _node_from_uid(self, uid: str) -> _ManagedWindowNode | None:
        node = self.manager._all_nodes.get(uid)
        if isinstance(node, _ManagedWindowNode):
            return node
        return None

    def _childtool_uid(
        self,
        row_index: QtCore.QModelIndex | int,
        parent_wrapper: _ImageToolWrapper | str,
    ) -> str:
        if isinstance(row_index, QtCore.QModelIndex):
            row_index = row_index.row()
        if isinstance(parent_wrapper, str):
            parent_node = self._node_from_uid(parent_wrapper)
            if parent_node is None:
                raise KeyError(parent_wrapper)
            return parent_node._childtool_indices[row_index]
        return parent_wrapper._childtool_indices[row_index]

    def _childtool(
        self, row_index: QtCore.QModelIndex, parent_wrapper: _ImageToolWrapper | str
    ) -> _ManagedWindowNode:
        return self.manager._child_node(self._childtool_uid(row_index, parent_wrapper))

    def _row_index(self, index_or_uid: int | str) -> QtCore.QModelIndex:
        """Get the corresponding QModelIndex for a parent index or child UID."""
        if isinstance(index_or_uid, str):
            node = self.manager._all_nodes.get(index_or_uid)
            if node is None:
                return QtCore.QModelIndex()
            if isinstance(node, _ImageToolWrapper):
                return self._row_index(node.index)
            parent_uid = node.parent_uid
            if parent_uid is None:
                return QtCore.QModelIndex()
            parent_node = self.manager._all_nodes[parent_uid]
            if isinstance(parent_node, _ImageToolWrapper):
                parent_index = self._row_index(parent_node.index)
                row = parent_node._childtool_indices.index(index_or_uid)
            else:
                parent_index = self._row_index(parent_uid)
                row = parent_node._childtool_indices.index(index_or_uid)
            return self.index(row, 0, parent_index)
        if index_or_uid not in self.manager._displayed_indices:
            return QtCore.QModelIndex()
        return self.index(self.manager._displayed_indices.index(index_or_uid), 0)

    def _is_archived(self, row_index: QtCore.QModelIndex) -> bool:
        return self._imagetool_wrapper(row_index).archived

    def index(
        self, row: int, column: int, parent: QtCore.QModelIndex | None = None
    ) -> QtCore.QModelIndex:
        if column != 0 or row < 0:
            return QtCore.QModelIndex()
        if parent is None:
            parent = QtCore.QModelIndex()

        if not parent.isValid():  # pragma: no branch
            # Top-level; ImageTool
            if row >= len(self.manager._displayed_indices):
                return QtCore.QModelIndex()
            wrapper = self._imagetool_wrapper(row)
            return self.createIndex(row, column, wrapper)

        ptr = parent.internalPointer()
        if not isinstance(ptr, (_ImageToolWrapper, str)):
            return QtCore.QModelIndex()
        if isinstance(ptr, str):
            parent_node = self._node_from_uid(ptr)
            if parent_node is None:
                return QtCore.QModelIndex()
            child_list = parent_node._childtool_indices
        else:
            child_list = ptr._childtool_indices
        if row >= len(child_list):
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
        if child is None:  # pragma: no branch
            return super().parent()

        if not child.isValid():  # pragma: no branch
            return QtCore.QModelIndex()

        uid = child.internalPointer()
        if isinstance(uid, str):
            node = self._node_from_uid(uid)
            if node is None:
                return QtCore.QModelIndex()
            if node.parent_uid is None:
                return QtCore.QModelIndex()
            parent = self.manager._all_nodes[node.parent_uid]
            if isinstance(parent, _ImageToolWrapper):
                return self._row_index(parent.index)
            return self._row_index(parent.uid)
        return QtCore.QModelIndex()

    def hasChildren(self, parent: QtCore.QModelIndex | None = None) -> bool:
        if parent is None:  # pragma: no branch
            parent = QtCore.QModelIndex()

        if parent.column() > 0:
            return False
        if not parent.isValid():
            return len(self.manager._displayed_indices) > 0

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            return len(ptr._childtool_indices) > 0
        if isinstance(ptr, str):
            node = self._node_from_uid(ptr)
            return node is not None and len(node._childtool_indices) > 0
        return False  # Child tool has no children

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        if parent is None:  # pragma: no branch
            parent = QtCore.QModelIndex()

        if parent.column() > 0:
            return 0
        if not parent.isValid():
            # Top-level; ImageTool
            return len(self.manager._displayed_indices)

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            # Number of child tools
            return len(ptr._childtool_indices)
        if isinstance(ptr, str):
            node = self._node_from_uid(ptr)
            return len(node._childtool_indices) if node is not None else 0
        return 0  # Child tool has no children

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return 1

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole
    ) -> typing.Any:
        if not index.isValid() or index.column() != 0:  # pragma: no branch
            return None

        ptr = index.internalPointer()
        if isinstance(ptr, str):
            return self._data_childtool(index, role)
        if not isinstance(ptr, _ImageToolWrapper):
            return None
        return self._data_imagetool(index, role)

    def _data_imagetool(self, index: QtCore.QModelIndex, role: int) -> typing.Any:
        tool_idx: int = self._imagetool_index(index)
        wrapper = self.manager._imagetool_wrappers[tool_idx]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.manager.label_of_imagetool(tool_idx)
        if role == QtCore.Qt.ItemDataRole.EditRole:
            return self.manager.name_of_imagetool(tool_idx)
        if role == _TOOL_TYPE_ROLE:
            return None
        if role == _NODE_UID_ROLE:
            return wrapper.uid
        if role == QtCore.Qt.ItemDataRole.SizeHintRole:
            return QtCore.QSize(100, 25)

        return None

    def _data_childtool(self, index: QtCore.QModelIndex, role: int) -> typing.Any:
        child_node = self._node_from_uid(index.internalPointer())
        if child_node is None:
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return child_node.display_text
        if role == QtCore.Qt.ItemDataRole.EditRole:
            if child_node.tool_window is not None:
                return child_node.tool_window._tool_display_name
            return child_node.name
        if role == _TOOL_TYPE_ROLE:
            return child_node.type_badge_text
        if role == _NODE_UID_ROLE:
            return child_node.uid
        if role == QtCore.Qt.ItemDataRole.SizeHintRole:
            return QtCore.QSize(100, 25)

        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            # Allow drops at root for top-level reordering
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled
        if index.column() != 0:
            return QtCore.Qt.ItemFlag.NoItemFlags

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

        child_node = self._node_from_uid(typing.cast("str", node))
        if child_node is None:
            return QtCore.Qt.ItemFlag.NoItemFlags
        flags = default_flags | QtCore.Qt.ItemFlag.ItemIsDropEnabled
        if not child_node.archived:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

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

    def _insert_childtool(self, uid: str, parent_idx: int | str) -> None:
        """Append a new tool to the end of the model.

        This must be called after the ImageTool is added to the manager.
        """
        parent = self._row_index(parent_idx)
        if not parent.isValid():
            return
        parent_ptr = parent.internalPointer()
        if isinstance(parent_ptr, str):
            parent_node = self._node_from_uid(parent_ptr)
            if parent_node is None:
                return
            child_list = parent_node._childtool_indices
        else:
            child_list = parent_ptr._childtool_indices
        row = max(len(child_list) - 1, 0)
        self.beginInsertRows(parent, row, row)
        self.endInsertRows()

    def remove_rows(
        self, row: int, count: int, parent: QtCore.QModelIndex | None = None
    ) -> None:
        """Remove rows from the model.

        Has the same signature as :meth:`QtCore.QAbstractItemModel.removeRows`, but
        without a return value.

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
            return

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            self.beginRemoveRows(parent, row, row + count - 1)
            del ptr._childtool_indices[row : row + count]
            self.endRemoveRows()
            return
        if isinstance(ptr, str):
            node = self._node_from_uid(ptr)
            if node is None:
                return
            self.beginRemoveRows(parent, row, row + count - 1)
            del node._childtool_indices[row : row + count]
            self.endRemoveRows()
            return

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: typing.Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid() or index.column() != 0:  # pragma: no branch
            return False

        ptr = index.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            if role == QtCore.Qt.ItemDataRole.EditRole:
                self.manager.rename_imagetool(self._imagetool_index(index), value)

                self.dataChanged.emit(index, index, [role])
                return True

        elif isinstance(ptr, str) and role == QtCore.Qt.ItemDataRole.EditRole:
            child_node = self._node_from_uid(ptr)
            if child_node is None:
                return False
            child_node.name = value

            self.dataChanged.emit(index, index, [role])
            return True

        return False

    def mimeTypes(self) -> list[str]:
        return [_MIME]

    def mimeData(self, indexes: Iterable[QtCore.QModelIndex]) -> QtCore.QMimeData:
        # Collect unique sibling rows from a single parent.
        rows: list[int] = []
        parent_pointer: str | None = None

        for idx in indexes:
            if not idx.isValid() or idx.column() != 0:
                continue
            node = idx.internalPointer()
            if isinstance(node, _ImageToolWrapper):
                current_parent_pointer = None
            else:
                parent_index = self.parent(idx)
                if not parent_index.isValid():
                    return QtCore.QMimeData()
                current_parent = parent_index.internalPointer()
                if isinstance(current_parent, _ImageToolWrapper):
                    current_parent_pointer = current_parent.uid
                elif isinstance(current_parent, str):
                    current_parent_pointer = current_parent
                else:
                    return QtCore.QMimeData()

            if parent_pointer is None and current_parent_pointer is None:
                parent_pointer = None
            elif rows and current_parent_pointer != parent_pointer:
                return QtCore.QMimeData()
            else:
                parent_pointer = current_parent_pointer

            rows.append(idx.row())

        if not rows:
            return QtCore.QMimeData()

        mime_data = QtCore.QMimeData()
        mime_data.setData(
            _MIME,
            QtCore.QByteArray(
                json.dumps(
                    {
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
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict) or not {"parent_id", "rows"} <= payload.keys():
            return None

        parent_id = payload["parent_id"]
        rows = payload["rows"]
        if parent_id is not None and not isinstance(parent_id, str):
            return None
        if not isinstance(rows, list):
            return None
        if any(
            not isinstance(row, int) or isinstance(row, bool) or row < 0 for row in rows
        ):
            return None
        if len(set(rows)) != len(rows):
            return None
        return {"parent_id": parent_id, "rows": sorted(rows)}

    @staticmethod
    def _contiguous_runs(rows: list[int]) -> list[tuple[int, int]]:
        """Convert sorted rows to list of (start, length).

        Example: [2,3,4,7,8] -> [(2,3), (7,2)]
        """
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
        if (
            action != QtCore.Qt.DropAction.MoveAction
            or column < -1
            or column > 0
            or row < -1
        ):
            logger.debug("canDropMimeData: wrong action/column")
            return False
        if _MIME not in data.formats():
            logger.debug("canDropMimeData: wrong mime type")
            return False

        payload = self._decode_mime(data)
        if payload is None:
            logger.debug("canDropMimeData: cannot decode mime")
            return False

        expected_parent_id = payload["parent_id"]
        actual_parent_id: str | None
        if not parent.isValid():
            actual_parent_id = None
        else:
            parent_ptr = parent.internalPointer()
            if expected_parent_id is None and isinstance(parent_ptr, _ImageToolWrapper):
                actual_parent_id = None
            else:
                if isinstance(parent_ptr, _ImageToolWrapper):
                    actual_parent_id = parent_ptr.uid
                elif isinstance(parent_ptr, str):
                    actual_parent_id = parent_ptr
                else:
                    logger.debug("canDropMimeData: invalid parent index")
                    return False
        if actual_parent_id != expected_parent_id:
            logger.debug("canDropMimeData: parent mismatch")
            return False
        return True

    def _apply_moves(
        self,
        parent_id: str | None,
        moves: list[tuple[int, int]],
        source_parent: QtCore.QModelIndex,
        destination_parent: QtCore.QModelIndex,
    ) -> bool:
        if parent_id is None:
            root_result = self.manager._displayed_indices.copy()
            for src, dest in moves:
                root_result.insert(dest, root_result.pop(src))
                if not self.beginMoveRows(
                    source_parent, src, src, destination_parent, dest
                ):
                    return False
                self.manager._displayed_indices = root_result
                self.endMoveRows()
            return True

        child_result = self.manager._all_nodes[parent_id]._childtool_indices.copy()
        for src, dest in moves:
            child_result.insert(dest, child_result.pop(src))
            if not self.beginMoveRows(
                source_parent, src, src, destination_parent, dest
            ):
                return False
            self.manager._all_nodes[parent_id]._childtool_indices = child_result
            self.endMoveRows()
        return True

    @staticmethod
    def _get_moves(list_original: list, list_shuffled: list) -> list[tuple[int, int]]:
        """Get list of (from, to) moves to convert list_original to list_shuffled.

        This ensures that ``from`` is larger than ``to``.
        """
        actions: list[tuple[int, int]] = []
        current = list_original[:]
        for target_idx, value in enumerate(list_shuffled):
            curr_idx = current.index(value)
            if curr_idx != target_idx:
                # Move value from curr_idx to target_idx
                actions.append((curr_idx, target_idx))
                item = current.pop(curr_idx)
                current.insert(target_idx, item)
        return actions

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
        source_rows: list[int] = payload["rows"]
        parent_id = typing.cast("str | None", payload["parent_id"])

        # Destination list and parent index
        if parent_id is None:
            parent_index = QtCore.QModelIndex()
            original: list[str] | list[int] = self.manager._displayed_indices.copy()
            if parent.isValid():
                # Dropping on a parent, adjust row to be relative to that item
                row = parent.row() + 1
        else:
            parent_index = parent
            original = self.manager._all_nodes[parent_id]._childtool_indices.copy()

        if not original or not source_rows:
            logger.debug("dropMimeData: empty target or source")
            return False
        if row > len(original) or any(
            source >= len(original) for source in source_rows
        ):
            logger.debug("dropMimeData: invalid target or source row")
            return False

        # Compute insertion position
        dest: int = len(original) if row < 0 or row > len(original) else row
        logger.debug(
            "dropMimeData: parent_id=%s, rows=%s -> %s", parent_id, source_rows, dest
        )

        # Create modified list by removing source rows and inserting them at dest
        dest_adjusted = dest - sum(1 for r in source_rows if r < dest)
        modified = [v for i, v in enumerate(original) if i not in source_rows]
        for source in source_rows:
            modified.insert(dest_adjusted, original[source])
            dest_adjusted += 1

        logger.debug("dropMimeData: original=%s, modified=%s", original, modified)

        # Get list of moves to convert original → modified
        moves = self._get_moves(original, modified)
        logger.debug("dropMimeData: moves=%s", moves)

        if not moves:
            logger.debug("dropMimeData: invalid moves")
            return False

        return self._apply_moves(parent_id, moves, parent_index, parent_index)


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
        self.setExpandsOnDoubleClick(False)

        self.setEditTriggers(self.EditTrigger.SelectedClicked)

        self.setWordWrap(True)  # Ellide text when width is too small
        self.setMouseTracking(True)  # Enable hover detection
        self.setHeaderHidden(True)  # Hide header

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
        self._menu.addAction(manager.promote_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.show_action)
        self._menu.addAction(manager.hide_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.remove_action)
        self._menu.addAction(manager.unwatch_action)
        self._menu.addAction(manager.archive_action)
        self._menu.addAction(manager.unarchive_action)
        self._menu.addAction(manager.reload_action)
        self._menu.addAction(manager.source_update_action)
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
    def deselect_all(self) -> None:
        self.clearSelection()

    @QtCore.Slot()
    @QtCore.Slot(int)
    @QtCore.Slot(str)
    def refresh(self, idx: int | str | None = None) -> None:
        """Trigger a refresh of the contents."""
        if idx is None:
            top = self._model.index(0, 0)
            bottom = self._model.index(self._model.rowCount() - 1, 0)
            if top.isValid() and bottom.isValid():  # pragma: no branch
                self._model.dataChanged.emit(top, bottom)
        else:
            row_idx = self._model._row_index(idx)
            if row_idx.isValid():  # pragma: no branch
                self._model.dataChanged.emit(row_idx, row_idx)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent | None) -> None:
        if event is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid() and isinstance(index.internalPointer(), str):
                option = QtWidgets.QStyleOptionViewItem()
                option.rect = self.visualRect(index)
                option.font = self.font()
                try:
                    child_node = self._model.manager._child_node(
                        index.internalPointer()
                    )
                except KeyError:
                    child_node = None
                if child_node is not None:
                    status_rect, _, _ = self._delegate._compute_child_status_info(
                        option, child_node
                    )
                    if status_rect and status_rect.contains(event.pos()):
                        child_node.show_source_update_dialog(parent=self._model.manager)
                        event.accept()
                        return
        super().mouseReleaseEvent(event)

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

    def clear_imagetools(self) -> None:
        """Clear all top-level ImageTool rows in a single model reset."""
        self.clearSelection()
        self._model.beginResetModel()
        self._model.manager._displayed_indices.clear()
        self._model.endResetModel()

    def childtool_added(self, uid: str, parent_idx: int | str) -> None:
        """Update the list view when a new child tool is added to the manager.

        This must be called after the child tool is added to the manager.
        """
        logger.debug("Adding child tool %s to parent index %s", uid, parent_idx)
        self._model._insert_childtool(uid, parent_idx)

    def childtool_removed(self, uid: str) -> None:
        """Update the list view when removing a child tool from the manager.

        This must be called before the child tool is removed from the manager.
        """
        node = self._model.manager._all_nodes.get(uid)
        if node is None or isinstance(node, _ImageToolWrapper):
            return
        if node.parent_uid is None:
            return
        parent_node = self._model.manager._all_nodes.get(node.parent_uid)
        if parent_node is None:
            return
        parent_index = (
            self._model._row_index(parent_node.index)
            if isinstance(parent_node, _ImageToolWrapper)
            else self._model._row_index(parent_node.uid)
        )
        if not parent_index.isValid():
            return
        for i, child_uid in enumerate(parent_node._childtool_indices):
            if child_uid == uid:
                self._model.remove_rows(i, 1, parent_index)
                return
