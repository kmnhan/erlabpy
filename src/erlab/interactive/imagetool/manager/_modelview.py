"""Model-view architecture used for displaying the list of ImageTool windows."""

from __future__ import annotations

__all__ = ["_ImageToolWrapperTreeView"]

import contextlib
import functools
import json
import logging
import math
import os
import typing
import weakref
from dataclasses import dataclass

import qtawesome as qta
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
    _preview_image_for_node,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from erlab.interactive.imagetool.manager import ImageToolManager

logger = logging.getLogger(__name__)

_NODE_UID_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 128
_TOOL_TYPE_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 129
_RowBadgeKind = typing.Literal[
    "dask",
    "link",
    "watched",
    "tool_type",
    "source_status",
    "dependency_status",
]


@dataclass(frozen=True)
class _RowBadge:
    """Hit-test result for a painted manager-row badge.

    The delegate paints badges manually, so Qt does not expose child widgets for them.
    Keep this as the single payload passed from delegate hit-testing to tooltip and
    click handling so their geometry cannot drift apart.
    """

    kind: _RowBadgeKind
    rect: QtCore.QRect
    tooltip: str


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
                    _, dask_rect, _, _ = self._compute_icons_info(option, child_node)
                    status_rect, _, _ = self._compute_child_status_info(
                        option,
                        child_node,
                        right_edge=self._right_badge_edge(dask_rect),
                    )
                    if status_rect is None:
                        status_rect, _, _ = self._compute_dependency_status_info(
                            option,
                            child_node,
                            right_edge=self._right_badge_edge(dask_rect),
                        )
                    right_badge_rect = self._leftmost_rect(status_rect, dask_rect)
                    if right_badge_rect is not None:
                        rect.setRight(right_badge_rect.left() - self.icon_right_pad)
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
        self,
        option: QtWidgets.QStyleOptionViewItem,
        node: _ImageToolWrapper | _ManagedWindowNode,
    ) -> tuple[
        int,
        QtCore.QRect | None,
        QtCore.QRect | None,
        QtCore.QRect | None,
    ]:
        # Determine which icons should be visible
        is_linked = isinstance(node, _ImageToolWrapper) and node.workspace_linked
        is_watched: bool = (
            isinstance(node, _ImageToolWrapper) and node._watched_varname is not None
        )
        is_lazy: bool = (
            node.imagetool is not None
            and node.pending_workspace_memory_payload is None
            and node.slicer_area.data_loadable
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
        if is_lazy:
            rect_x -= rect_dx
            dask_rect = QtCore.QRect(rect_x, rect_y, rect_size, rect_size)

        # Link indicator
        if is_linked:
            rect_x -= rect_dx
            link_rect = QtCore.QRect(rect_x, rect_y, rect_size, rect_size)

        # Watched variable name label
        if is_watched:
            watched_text: str = typing.cast(
                "str", typing.cast("_ImageToolWrapper", node)._watched_varname
            )

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

    @staticmethod
    def _right_badge_edge(*rects: QtCore.QRect | None) -> int | None:
        visible = [rect.left() for rect in rects if rect is not None]
        return min(visible) if visible else None

    @staticmethod
    def _leftmost_rect(*rects: QtCore.QRect | None) -> QtCore.QRect | None:
        visible = [rect for rect in rects if rect is not None]
        return min(visible, key=lambda rect: rect.left()) if visible else None

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
        _icons_width, dask_rect, link_rect, watched_rect = self._compute_icons_info(
            option, tool_wrapper
        )
        icon_rects = [
            rect for rect in (dask_rect, link_rect, watched_rect) if rect is not None
        ]
        dependency_rect, dependency_text, dependency_color = (
            self._compute_dependency_status_info(
                option,
                tool_wrapper,
                right_edge=(
                    min(rect.left() for rect in icon_rects) if icon_rects else None
                ),
            )
        )

        # Draw label (skip while editing for inline editor)
        if not is_editing:  # pragma: no branch
            palette = option.palette
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
            text_rect = QtCore.QRect(option.rect)
            right_badge_rect = self._leftmost_rect(dependency_rect, *icon_rects)
            if right_badge_rect is not None:
                text_rect.setRight(right_badge_rect.left() - self.icon_right_pad)
            elided = fm.elidedText(
                text,
                view.textElideMode(),
                max(text_rect.width(), 0),
            )
            painter.setPen(palette.color(color_group, role))
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignLeft,
                elided,
            )

        # Icons
        if dask_rect:
            self._paint_icon(painter, option, dask_rect, self._dask_icon)

        if link_rect:
            link_color: QtGui.QColor | None = None
            if tool_wrapper.workspace_link_key is not None:
                link_color = self.manager.color_for_workspace_link_key(
                    tool_wrapper.workspace_link_key
                )
            else:
                proxy = tool_wrapper.slicer_area._linking_proxy
                if proxy is not None:
                    link_color = self.manager.color_for_linker(proxy)
            if link_color is not None:
                link_icon = qta.icon("mdi6.link-variant", color=link_color)
                self._paint_icon(painter, option, link_rect, link_icon)

        if watched_rect:
            palette = option.palette
            watched_varname = str(tool_wrapper._watched_varname)
            if tool_wrapper._watched_connected:
                color = self.manager.color_for_watched_var_source(tool_wrapper)
            else:
                color = option.palette.color(QtGui.QPalette.ColorRole.Mid)

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

        if dependency_rect and dependency_text and dependency_color:
            _fill_rounded_rect(
                painter,
                dependency_rect,
                facecolor=option.palette.base(),
                edgecolor=dependency_color,
                linewidth=self.icon_border_width,
                radius=self.icon_corner_radius,
            )
            dependency_font = QtGui.QFont(option.font)
            dependency_font.setPointSizeF(
                self._font_size * self.child_status_font_scale
            )
            painter.save()
            painter.setFont(dependency_font)
            painter.setPen(dependency_color)
            painter.drawText(
                dependency_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | QtCore.Qt.AlignmentFlag.AlignCenter,
                dependency_text,
            )
            painter.restore()

        # Preview popup (hover)
        if (
            not is_editing
            and self.manager.preview_action.isChecked()
            and (
                QtWidgets.QStyle.StateFlag.State_MouseOver in option.state
                or self._force_hover
            )
        ):
            self._show_popup(*_preview_image_for_node(tool_wrapper), option)

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
        dask_rect: QtCore.QRect | None = None
        try:
            child_node = self.manager._child_node(index.internalPointer())
        except KeyError:
            child_node = None
        else:
            type_rect, type_text, type_color = self._compute_tool_type_info(
                option, child_node
            )
            _, dask_rect, _, _ = self._compute_icons_info(option, child_node)
            status_rect, status_text, status_color = self._compute_child_status_info(
                option,
                child_node,
                right_edge=self._right_badge_edge(dask_rect),
            )
            if status_rect is None:
                status_rect, status_text, status_color = (
                    self._compute_dependency_status_info(
                        option,
                        child_node,
                        right_edge=self._right_badge_edge(dask_rect),
                    )
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
            painter.setPen(option.palette.color(role))

            text_rect = QtCore.QRect(option.rect)
            if type_rect is not None:
                text_rect.setLeft(type_rect.right() + self.tool_type_rect_gap + 1)
            right_badge_rect = self._leftmost_rect(status_rect, dask_rect)
            if right_badge_rect is not None:
                text_rect.setRight(right_badge_rect.left() - self.icon_right_pad)

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
            if dask_rect:
                self._paint_icon(painter, option, dask_rect, self._dask_icon)
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
                self._show_popup(*_preview_image_for_node(child_node), option)
                return

            preview_pixmap = (
                child_node.tool_window.preview_pixmap
                if child_node.tool_window is not None
                else None
            )
            if (
                preview_pixmap is None
                and child_node.pending_workspace_tool_payload is not None
            ):
                pending_preview = (
                    child_node.cached_pending_workspace_tool_preview_image()
                )
                preview_pixmap = None if pending_preview is None else pending_preview[1]
            if preview_pixmap is not None and not preview_pixmap.isNull():
                width = preview_pixmap.width()
                height = preview_pixmap.height()
                self._show_popup(
                    height / width if width > 0 else 1.0, preview_pixmap, option
                )
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
            if pixmap is None or pixmap.isNull():
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
        *,
        right_edge: int | None = None,
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
        if right_edge is None:
            right_edge = option.rect.right()
        rect_x = right_edge - badge_width - self.icon_right_pad
        return QtCore.QRect(rect_x, rect_y, badge_width, rect_size), text, color

    def _compute_dependency_status_info(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        right_edge: int | None = None,
    ) -> tuple[QtCore.QRect | None, str | None, QtGui.QColor | None]:
        match self.manager.dependency_status_for_uid(node.uid):
            case "changed":
                text = "Changed"
                color = QtGui.QColor("#8a6d00")
            case "missing":
                text = "Missing"
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
        if right_edge is None:
            right_edge = option.rect.right()
        rect_x = right_edge - badge_width - self.icon_right_pad
        return QtCore.QRect(rect_x, rect_y, badge_width, rect_size), text, color

    def _option_for_index(
        self,
        view: QtWidgets.QAbstractItemView,
        index: QtCore.QModelIndex,
    ) -> QtWidgets.QStyleOptionViewItem:
        """Build the same style option used for painting a visible row.

        Badge hit-testing must use the current visual row rectangle. Relying on a
        default style option gives correct font/palette data but a stale rect,
        especially after scrolling or expanding nested rows.
        """
        option = QtWidgets.QStyleOptionViewItem()
        self.initStyleOption(option, index)
        option.rect = view.visualRect(index)
        option.widget = view
        return option

    def _badge_at(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
        pos: QtCore.QPoint,
    ) -> _RowBadge | None:
        """Return the badge under ``pos`` using the same geometry as painting.

        The ordering here mirrors the visual layout. For top-level rows the badges are
        right-aligned; for child rows the tool-type badge is checked before right-side
        dask and source status badges. Tooltip, cursor, and click paths all call this
        method so any badge geometry change has one source of truth.
        """
        if not index.isValid():
            return None
        if option.rect.isEmpty():
            return None

        node = index.internalPointer()
        if isinstance(node, _ImageToolWrapper):
            _, dask_rect, link_rect, watched_rect = self._compute_icons_info(
                option, node
            )
            icon_rects = [
                rect
                for rect in (dask_rect, link_rect, watched_rect)
                if rect is not None
            ]
            dependency_rect, _, _ = self._compute_dependency_status_info(
                option,
                node,
                right_edge=(
                    min(rect.left() for rect in icon_rects) if icon_rects else None
                ),
            )
            if dask_rect is not None and dask_rect.contains(pos):
                tooltip = (
                    "Dask-backed data. Click to open Dask and chunk controls."
                    if node.slicer_area.data_chunked
                    else "File-backed data. Click to load into memory."
                )
                return _RowBadge(
                    "dask",
                    dask_rect,
                    tooltip,
                )
            if (
                link_rect is not None
                and node.workspace_linked
                and link_rect.contains(pos)
            ):
                proxy = (
                    None if node.imagetool is None else node.slicer_area._linking_proxy
                )
                if proxy is not None:
                    linker_index = self.manager.linker_index(proxy)
                    return _RowBadge(
                        "link",
                        link_rect,
                        f"Linked (#{linker_index}). Click to unlink this window.",
                    )
                return _RowBadge(
                    "link",
                    link_rect,
                    "Linked. Click to unlink this window.",
                )
            if watched_rect is not None and watched_rect.contains(pos):
                varname = str(node._watched_varname)
                if node._watched_connected:
                    tooltip = f"Watching variable {varname!r}. Click for watch actions."
                else:
                    tooltip = (
                        f"Watched variable {varname!r} is disconnected. "
                        f"Run %watch {varname} or %watch --restore after defining it."
                    )
                return _RowBadge(
                    "watched",
                    watched_rect,
                    tooltip,
                )
            if dependency_rect is not None and dependency_rect.contains(pos):
                tooltip = self.manager.dependency_status_tooltip_for_uid(node.uid)
                if tooltip is not None:
                    return _RowBadge("dependency_status", dependency_rect, tooltip)
            return None

        if not isinstance(node, str):
            return None
        try:
            child_node = self.manager._child_node(node)
        except KeyError:
            return None

        type_rect, type_text, _ = self._compute_tool_type_info(option, child_node)
        if type_rect is not None and type_text is not None and type_rect.contains(pos):
            return _RowBadge(
                "tool_type",
                type_rect,
                f"Tool type: {type_text}. Click to show this tool.",
            )

        _, dask_rect, _, _ = self._compute_icons_info(option, child_node)
        status_rect, _, _ = self._compute_child_status_info(
            option,
            child_node,
            right_edge=self._right_badge_edge(dask_rect),
        )
        source_status = status_rect is not None
        if status_rect is None:
            status_rect, _, _ = self._compute_dependency_status_info(
                option,
                child_node,
                right_edge=self._right_badge_edge(dask_rect),
            )
        if dask_rect is not None and dask_rect.contains(pos):
            tooltip = (
                "Dask-backed data. Click to open Dask and chunk controls."
                if child_node.slicer_area.data_chunked
                else "File-backed data. Click to load into memory."
            )
            return _RowBadge(
                "dask",
                dask_rect,
                tooltip,
            )

        if status_rect is None or not status_rect.contains(pos):
            return None
        if not source_status:
            tooltip = self.manager.dependency_status_tooltip_for_uid(child_node.uid)
            if tooltip is None:
                return None
            return _RowBadge("dependency_status", status_rect, tooltip)

        match child_node.source_state:
            case "stale":
                tooltip = (
                    "Stale. Click to update this tool from the latest compatible data."
                )
            case "unavailable":
                tooltip = (
                    "Unavailable. Click to review why this tool cannot update from the "
                    "current data."
                )
            case _:
                tooltip = (
                    "Automatic updates enabled. Click to configure automatic updates."
                )
        return _RowBadge("source_status", status_rect, tooltip)

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        view = self.parent()
        viewport = obj if isinstance(obj, QtWidgets.QWidget) else None
        if not (
            isinstance(view, _ImageToolWrapperTreeView)
            and erlab.interactive.utils.qt_is_valid(view, viewport)
        ):
            return super().eventFilter(obj, event)
        if event is not None:  # pragma: no branch
            match event.type():
                case (
                    QtCore.QEvent.Type.Resize
                    | QtCore.QEvent.Type.Leave
                    | QtCore.QEvent.Type.WindowStateChange
                ):
                    self.preview_popup.hide()
                    erlab.interactive.utils.set_widget_cursor(viewport, None)
                case QtCore.QEvent.Type.MouseMove:
                    if not isinstance(event, QtGui.QMouseEvent):
                        return super().eventFilter(obj, event)
                    pos = event.pos()
                    index = view.indexAt(pos)
                    if not index.isValid():
                        self.preview_popup.hide()
                    if not index.isValid():
                        badge = None
                    else:
                        option = self._option_for_index(view, index)
                        badge = self._badge_at(option, index, pos)
                    erlab.interactive.utils.set_widget_cursor(
                        viewport,
                        (
                            None
                            if badge is None
                            else QtCore.Qt.CursorShape.PointingHandCursor
                        ),
                    )

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
            badge = self._badge_at(option, index, event.pos())
            if badge is not None:
                QtWidgets.QToolTip.showText(
                    event.globalPos(), badge.tooltip, view, badge.rect
                )
                return True

        return super().helpEvent(event, view, option, index)


_MIME = "application/x-imagetool-manager-internal-move"
_FIGURE_SOURCE_MIME = "application/x-erlab-imagetool-manager-figure-sources"


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
        return self.manager._tool_graph.displayed_indices[row_index]

    def _imagetool_wrapper(
        self, row_index: QtCore.QModelIndex | int
    ) -> _ImageToolWrapper:
        if isinstance(row_index, QtCore.QModelIndex):
            ptr = row_index.internalPointer()
            if not isinstance(ptr, _ImageToolWrapper):
                raise KeyError("Index does not point to a tool wrapper")
            return ptr
        return self.manager._tool_graph.root_wrappers[self._imagetool_index(row_index)]

    def _node_from_uid(self, uid: str) -> _ManagedWindowNode | None:
        node = self.manager._tool_graph.nodes.get(uid)
        if isinstance(node, _ManagedWindowNode):
            return node
        return None

    def _is_visible_child_uid(self, uid: str) -> bool:
        node = self._node_from_uid(uid)
        return node is not None and not self.manager._is_figure_node(node)

    def _childtool_uids(self, parent_wrapper: _ImageToolWrapper | str) -> list[str]:
        if isinstance(parent_wrapper, str):
            parent_node = self._node_from_uid(parent_wrapper)
            if parent_node is None:
                raise KeyError(parent_wrapper)
            child_list = parent_node._childtool_indices
        else:
            child_list = parent_wrapper._childtool_indices
        return [uid for uid in child_list if self._is_visible_child_uid(uid)]

    def _childtool_uid(
        self,
        row_index: QtCore.QModelIndex | int,
        parent_wrapper: _ImageToolWrapper | str,
    ) -> str:
        if isinstance(row_index, QtCore.QModelIndex):
            row_index = row_index.row()
        return self._childtool_uids(parent_wrapper)[row_index]

    def _childtool(
        self, row_index: QtCore.QModelIndex, parent_wrapper: _ImageToolWrapper | str
    ) -> _ManagedWindowNode:
        return self.manager._child_node(self._childtool_uid(row_index, parent_wrapper))

    def _row_index(self, index_or_uid: int | str) -> QtCore.QModelIndex:
        """Get the corresponding QModelIndex for a parent index or child UID."""
        if isinstance(index_or_uid, str):
            node = self.manager._tool_graph.nodes.get(index_or_uid)
            if node is None:
                return QtCore.QModelIndex()
            if isinstance(node, _ImageToolWrapper):
                return self._row_index(node.index)
            parent_uid = node.parent_uid
            if parent_uid is None:
                return QtCore.QModelIndex()
            parent_node = self.manager._tool_graph.nodes[parent_uid]
            if isinstance(parent_node, _ImageToolWrapper):
                parent_index = self._row_index(parent_node.index)
                visible_children = self._childtool_uids(parent_node)
            else:
                parent_index = self._row_index(parent_uid)
                visible_children = self._childtool_uids(parent_uid)
            if index_or_uid not in visible_children:
                return QtCore.QModelIndex()
            row = visible_children.index(index_or_uid)
            return self.index(row, 0, parent_index)
        if index_or_uid not in self.manager._tool_graph.displayed_indices:
            return QtCore.QModelIndex()
        return self.index(
            self.manager._tool_graph.displayed_indices.index(index_or_uid), 0
        )

    def index(
        self, row: int, column: int, parent: QtCore.QModelIndex | None = None
    ) -> QtCore.QModelIndex:
        if column != 0 or row < 0:
            return QtCore.QModelIndex()
        if parent is None:
            parent = QtCore.QModelIndex()

        if not parent.isValid():  # pragma: no branch
            # Top-level; ImageTool
            if row >= len(self.manager._tool_graph.displayed_indices):
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
            child_list = self._childtool_uids(ptr)
        else:
            child_list = self._childtool_uids(ptr)
        if row >= len(child_list):
            return QtCore.QModelIndex()

        child = child_list[row]
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
            parent = self.manager._tool_graph.nodes[node.parent_uid]
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
            return len(self.manager._tool_graph.displayed_indices) > 0

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            return len(self._childtool_uids(ptr)) > 0
        if isinstance(ptr, str):
            node = self._node_from_uid(ptr)
            return node is not None and len(self._childtool_uids(ptr)) > 0
        return False  # Child tool has no children

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        if parent is None:  # pragma: no branch
            parent = QtCore.QModelIndex()

        if parent.column() > 0:
            return 0
        if not parent.isValid():
            # Top-level; ImageTool
            return len(self.manager._tool_graph.displayed_indices)

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            # Number of child tools
            return len(self._childtool_uids(ptr))
        if isinstance(ptr, str):
            node = self._node_from_uid(ptr)
            return len(self._childtool_uids(ptr)) if node is not None else 0
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
        wrapper = self.manager._tool_graph.root_wrappers[tool_idx]

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
            return (
                default_flags
                | QtCore.Qt.ItemFlag.ItemIsDropEnabled
                | QtCore.Qt.ItemFlag.ItemIsEditable
            )

        child_node = self._node_from_uid(typing.cast("str", node))
        if child_node is None:
            return QtCore.Qt.ItemFlag.NoItemFlags
        return (
            default_flags
            | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            | QtCore.Qt.ItemFlag.ItemIsEditable
        )

    def supportedDragActions(self) -> QtCore.Qt.DropAction:
        return QtCore.Qt.DropAction.MoveAction | QtCore.Qt.DropAction.CopyAction

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        return QtCore.Qt.DropAction.MoveAction

    def _insert_imagetool(self, index: int) -> None:
        """Append a new tool to the end of the model.

        This must be called after the ImageTool is added to the manager.
        """
        n_rows = self.rowCount()

        self.beginInsertRows(QtCore.QModelIndex(), n_rows, n_rows)  # Insert at end
        self.manager._tool_graph.insert_root_order(index, n_rows)
        self.endInsertRows()

    def _insert_childtool(self, uid: str, parent_idx: int | str) -> None:
        """Append a new tool to the end of the model.

        This must be called after the ImageTool is added to the manager.
        """
        parent = self._row_index(parent_idx)
        if not parent.isValid():
            return
        parent_ptr = parent.internalPointer()
        if not isinstance(parent_ptr, (_ImageToolWrapper, str)):
            return
        try:
            visible_children = self._childtool_uids(parent_ptr)
        except KeyError:
            return
        if uid not in visible_children:
            return
        row = visible_children.index(uid)
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
            self.manager._tool_graph.remove_root_rows(row, count)
            self.endRemoveRows()
            return

        ptr = parent.internalPointer()
        if isinstance(ptr, _ImageToolWrapper):
            visible_children = self._childtool_uids(ptr)
            if row >= len(visible_children):
                return
            uids = visible_children[row : row + count]
            self.beginRemoveRows(parent, row, row + count - 1)
            for uid in uids:
                with contextlib.suppress(ValueError):
                    ptr._childtool_indices.remove(uid)
            self.endRemoveRows()
            return
        if isinstance(ptr, str):
            node = self._node_from_uid(ptr)
            if node is None:
                return
            visible_children = self._childtool_uids(ptr)
            if row >= len(visible_children):
                return
            uids = visible_children[row : row + count]
            self.beginRemoveRows(parent, row, row + count - 1)
            for uid in uids:
                with contextlib.suppress(ValueError):
                    node._childtool_indices.remove(uid)
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
                self.manager.rename_imagetool(self._imagetool_index(index), str(value))

                self.dataChanged.emit(index, index, [role])
                return True

        elif isinstance(ptr, str) and role == QtCore.Qt.ItemDataRole.EditRole:
            child_node = self._node_from_uid(ptr)
            if child_node is None:
                return False
            child_node.name = str(value)

            self.dataChanged.emit(index, index, [role])
            return True

        return False

    def mimeTypes(self) -> list[str]:
        return [_MIME, _FIGURE_SOURCE_MIME, "text/plain"]

    def mimeData(self, indexes: Iterable[QtCore.QModelIndex]) -> QtCore.QMimeData:
        # Collect unique sibling rows from a single parent.
        rows: list[int] = []
        dragged_uids: list[str] = []
        source_uids: list[str] = []
        internal_move_valid = True
        parent_pointer: str | None = None

        for idx in indexes:
            if not idx.isValid() or idx.column() != 0:
                continue
            node = idx.internalPointer()
            if isinstance(node, _ImageToolWrapper):
                dragged_uid = node.uid
                if node.uid not in source_uids:
                    source_uids.append(node.uid)
                current_parent_pointer = None
            else:
                child_node = self._node_from_uid(typing.cast("str", node))
                if child_node is None:
                    internal_move_valid = False
                    continue
                if child_node.is_imagetool and child_node.uid not in source_uids:
                    source_uids.append(child_node.uid)
                parent_index = self.parent(idx)
                if not parent_index.isValid():
                    internal_move_valid = False
                    continue
                current_parent = parent_index.internalPointer()
                if isinstance(current_parent, _ImageToolWrapper):
                    current_parent_pointer = current_parent.uid
                elif isinstance(current_parent, str):
                    current_parent_pointer = current_parent
                else:
                    internal_move_valid = False
                    continue
                dragged_uid = child_node.uid

            if dragged_uid not in dragged_uids:
                dragged_uids.append(dragged_uid)

            if parent_pointer is None and current_parent_pointer is None:
                parent_pointer = None
            elif rows and current_parent_pointer != parent_pointer:
                internal_move_valid = False
                continue
            else:
                parent_pointer = current_parent_pointer

            rows.append(idx.row())

        if not rows and not source_uids:
            return QtCore.QMimeData()

        mime_data = QtCore.QMimeData()
        if rows and internal_move_valid:
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
        if source_uids:
            mime_data.setData(
                _FIGURE_SOURCE_MIME,
                QtCore.QByteArray(
                    json.dumps({"uids": tuple(source_uids)}).encode("utf-8")
                ),
            )
        if len(dragged_uids) == 1:
            dragged_node = self.manager._tool_graph.nodes.get(dragged_uids[0])
            if dragged_node is not None:
                path = self.manager._tool_graph.node_path(dragged_node)
                if path:
                    expression = f"tools[{path[0]}]"
                    expression += "".join(
                        f".children[{child_row}]" for child_row in path[1:]
                    )
                    mime_data.setText(expression)
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
    def decode_figure_source_mime(mime: QtCore.QMimeData | None) -> tuple[str, ...]:
        if mime is None or _FIGURE_SOURCE_MIME not in mime.formats():
            return ()
        raw: bytes = mime.data(_FIGURE_SOURCE_MIME).data()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return ()
        if not isinstance(payload, dict):
            return ()
        uids = payload.get("uids")
        if not isinstance(uids, list):
            return ()
        output: list[str] = []
        for uid in uids:
            if isinstance(uid, str) and uid not in output:
                output.append(uid)
        return tuple(output)

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
            for src, dest in moves:
                if not self.beginMoveRows(
                    source_parent, src, src, destination_parent, dest
                ):
                    return False
                self.manager._tool_graph.move_root_rows(((src, dest),))
                self.endMoveRows()
            return True

        for src, dest in moves:
            if not self.beginMoveRows(
                source_parent, src, src, destination_parent, dest
            ):
                return False
            self.manager._tool_graph.move_child_rows(parent_id, ((src, dest),))
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
            original: list[str] | list[int] = (
                self.manager._tool_graph.displayed_indices.copy()
            )
            if parent.isValid():
                # Dropping on a parent, adjust row to be relative to that item
                row = parent.row() + 1
        else:
            parent_index = parent
            original = self._childtool_uids(parent_id)

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

        if parent_id is not None:
            raw_children = self.manager._tool_graph.nodes[parent_id]._childtool_indices
            if len(raw_children) != len(original):
                modified_iter = iter(typing.cast("list[str]", modified))
                self.manager._tool_graph.nodes[parent_id]._childtool_indices = [
                    next(modified_iter) if self._is_visible_child_uid(uid) else uid
                    for uid in raw_children
                ]
                self.layoutChanged.emit()
                moved = True
            else:
                moved = self._apply_moves(parent_id, moves, parent_index, parent_index)
        else:
            moved = self._apply_moves(parent_id, moves, parent_index, parent_index)
        if moved:
            self.manager._mark_workspace_structure_dirty("Reordered windows")
        return moved


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
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
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
        self._badge_menu: QtWidgets.QMenu | None = None
        self._menu.setToolTipsVisible(True)
        self._menu.addAction(manager.reindex_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.concat_action)
        self._menu.addAction(manager.batch_action)
        self._menu.addAction(manager.create_figure_action)
        self._menu.addAction(manager.duplicate_action)
        self._menu.addAction(manager.promote_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.show_action)
        self._menu.addAction(manager.hide_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.remove_action)
        self._menu.addAction(manager.unwatch_action)
        self._menu.addAction(manager.offload_action)
        self._menu.addAction(manager.reload_action)
        self._menu.addAction(manager.source_update_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.rename_action)
        self._menu.addAction(manager.link_action)
        self._menu.addAction(manager.unlink_action)
        self._menu.addSeparator()
        self._menu.addAction(manager.edit_note_action)
        self._menu.addAction(manager.copy_note_action)
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
        return [
            self._model.manager._tool_graph.displayed_indices[i] for i in row_indices
        ]

    @property
    def selected_childtool_uids(self) -> list[str]:
        """UIDs of currently selected child tools."""
        return [
            index.internalPointer()
            for index in self.selectedIndexes()
            if isinstance(index.internalPointer(), str)
        ]

    def figure_source_uids_from_mime(
        self, mime: QtCore.QMimeData | None
    ) -> tuple[str, ...]:
        return self._model.decode_figure_source_mime(mime)

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
            if index.isValid():
                option = self._delegate._option_for_index(self, index)
                badge = self._delegate._badge_at(option, index, event.pos())
                if badge is not None:
                    self._handle_badge_click(index, badge)
                    event.accept()
                    return
        super().mouseReleaseEvent(event)

    def _handle_badge_click(self, index: QtCore.QModelIndex, badge: _RowBadge) -> None:
        """Dispatch a click on a painted badge to the row-specific action.

        Badge clicks intentionally operate on the clicked row only. They do not reuse
        selection-based actions because a selected group can contain unrelated rows,
        while the badge affordance belongs to one painted row.
        """
        node = index.internalPointer()
        match badge.kind:
            case "dask":
                if isinstance(node, _ImageToolWrapper):
                    self._show_dask_badge_menu(node, badge.rect)
                elif isinstance(node, str):
                    try:
                        child_node = self._model.manager._child_node(node)
                    except KeyError:
                        return
                    self._show_dask_badge_menu(child_node, badge.rect)
            case "link":
                if isinstance(node, _ImageToolWrapper):
                    self._unlink_badge_target(node)
            case "watched":
                if isinstance(node, _ImageToolWrapper):
                    self._show_watched_badge_menu(node, badge.rect)
            case "tool_type":
                if isinstance(node, str):
                    self._model.manager.show_childtool(node)
            case "source_status":
                if isinstance(node, str):
                    try:
                        child_node = self._model.manager._child_node(node)
                    except KeyError:
                        return
                    child_node.show_source_update_dialog(parent=self._model.manager)
            case "dependency_status":
                if isinstance(node, _ImageToolWrapper):
                    target: int | str | None = node.index
                elif isinstance(node, str):
                    target = node
                else:
                    target = None
                if target is None:
                    return
                self._model.manager._show_dependency_reload_dialog(target)

    def _show_dask_badge_menu(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        badge_rect: QtCore.QRect,
    ) -> None:
        """Open the clicked tool's Dask menu at the badge location."""
        tool = node.imagetool
        if tool is None:
            return
        tool.slicer_area.compute_act.setEnabled(tool.slicer_area.data_loadable)
        tool._dask_menu.update_actions_visibility()
        viewport = self.viewport()
        if viewport is None:
            return
        tool._dask_menu.popup(viewport.mapToGlobal(badge_rect.bottomLeft()))

    def _unlink_badge_target(self, wrapper: _ImageToolWrapper) -> None:
        """Confirm and unlink only the ImageTool represented by ``wrapper``."""
        if (
            QtWidgets.QMessageBox.question(
                self._model.manager,
                "Unlink ImageTool?",
                "Unlink this ImageTool from linked cursor and color updates?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return

        manager = self._model.manager
        manager._actions_controller.unlink_imagetool_nodes((wrapper,))
        self.refresh(wrapper.index)

    def _show_watched_badge_menu(
        self, wrapper: _ImageToolWrapper, badge_rect: QtCore.QRect
    ) -> None:
        """Show per-row watch actions for the clicked watched-variable badge."""
        menu = QtWidgets.QMenu("Watch", self)
        menu.setToolTipsVisible(True)
        refresh_action = typing.cast(
            "QtGui.QAction", menu.addAction("Refresh From Variable")
        )
        if wrapper._watched_connected:
            refresh_action.setToolTip(
                "Refresh this ImageTool from the watched variable"
            )
        else:
            refresh_action.setEnabled(False)
            refresh_action.setToolTip(
                "Reconnect this watched variable from the notebook"
            )
        refresh_action.triggered.connect(wrapper._trigger_watched_update)
        stop_action = typing.cast("QtGui.QAction", menu.addAction("Stop Watching"))
        stop_action.setToolTip("Detach this ImageTool from the watched variable")
        stop_action.triggered.connect(
            lambda _checked=False, target=wrapper: self._stop_watching_badge_target(
                target
            )
        )
        self._badge_menu = menu
        viewport = self.viewport()
        if viewport is None:
            return
        menu.popup(viewport.mapToGlobal(badge_rect.bottomLeft()))

    def _stop_watching_badge_target(self, wrapper: _ImageToolWrapper) -> None:
        """Confirm and detach the clicked ImageTool from its watched variable."""
        varname = wrapper._watched_varname
        if varname is None:
            return
        if (
            QtWidgets.QMessageBox.question(
                self._model.manager,
                "Stop Watching Variable?",
                f"Stop watching variable {varname!r} for this ImageTool?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return
        wrapper.unwatch()

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
            self._model.manager._tool_graph.displayed_indices
        ):  # pragma: no branch
            if tool_idx == index:  # pragma: no branch
                self._model.remove_rows(i, 1)
                break

    def clear_imagetools(self) -> None:
        """Clear all top-level ImageTool rows in a single model reset."""
        self.clearSelection()
        self._model.beginResetModel()
        self._model.manager._tool_graph.clear_root_order()
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
        node = self._model.manager._tool_graph.nodes.get(uid)
        if node is None or isinstance(node, _ImageToolWrapper):
            return
        if node.parent_uid is None:
            return
        parent_node = self._model.manager._tool_graph.nodes.get(node.parent_uid)
        if parent_node is None:
            return
        parent_index = (
            self._model._row_index(parent_node.index)
            if isinstance(parent_node, _ImageToolWrapper)
            else self._model._row_index(parent_node.uid)
        )
        if not parent_index.isValid():
            return
        visible_children = self._model._childtool_uids(
            parent_node
            if isinstance(parent_node, _ImageToolWrapper)
            else parent_node.uid
        )
        for i, child_uid in enumerate(visible_children):
            if child_uid == uid:
                self._model.remove_rows(i, 1, parent_index)
                return
