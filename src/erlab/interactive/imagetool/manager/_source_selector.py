"""Compact source selection from the Manager tree."""

from __future__ import annotations

import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.manager._modelview import (
    _TOOL_TYPE_ROLE,
    _ImageToolWrapperItemDelegate,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode


class _ManagerSourceDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent: QtWidgets.QTreeWidget) -> None:
        super().__init__(parent)
        self._manager: weakref.ReferenceType[ImageToolManager] | None = None

    def set_manager(self, manager: ImageToolManager | None) -> None:
        self._manager = None if manager is None else weakref.ref(manager)

    def _badge_info(
        self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex
    ) -> tuple[_ImageToolWrapperItemDelegate, QtCore.QRect, str, QtGui.QColor] | None:
        if not index.data(_TOOL_TYPE_ROLE):
            return None
        manager = None if self._manager is None else self._manager()
        if manager is None or not erlab.interactive.utils.qt_is_valid(manager):
            return None
        if not erlab.interactive.utils.qt_is_valid(manager.tree_view):
            return None
        node = manager._tool_graph.nodes.get(
            index.data(QtCore.Qt.ItemDataRole.UserRole)
        )
        if node is None or node.is_imagetool:
            return None
        delegate = manager.tree_view.itemDelegate()
        if not isinstance(delegate, _ImageToolWrapperItemDelegate) or not (
            erlab.interactive.utils.qt_is_valid(delegate)
        ):
            return None
        rect, text, color = delegate._compute_tool_type_info(option, node)
        if rect is None or text is None or color is None:
            return None
        return delegate, rect, text, color

    def paint(
        self,
        painter: QtGui.QPainter | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        if painter is None:
            return
        styled = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        badge = self._badge_info(styled, index)
        if badge is None:
            super().paint(painter, styled, index)
            return
        delegate, rect, text, color = badge
        widget = typing.cast("QtWidgets.QWidget | None", styled.widget)
        style = typing.cast(
            "QtWidgets.QStyle",
            widget.style() if widget is not None else QtWidgets.QApplication.style(),
        )
        painter.save()
        style.drawPrimitive(
            QtWidgets.QStyle.PrimitiveElement.PE_PanelItemViewItem,
            styled,
            painter,
            widget,
        )
        delegate._paint_tool_type_badge(painter, styled, rect, text, color)
        styled.rect.setLeft(rect.right() + delegate.tool_type_rect_gap + 1)
        super().paint(painter, styled, index)
        painter.restore()

    def sizeHint(
        self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex
    ) -> QtCore.QSize:
        size = super().sizeHint(option, index)
        styled = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        badge = self._badge_info(styled, index)
        if badge is not None:
            delegate, rect, _, _ = badge
            size.setWidth(size.width() + rect.width() + delegate.tool_type_rect_gap)
            size.setHeight(max(size.height(), rect.height()))
        return size


class _ManagerSourceTree(QtWidgets.QWidget):
    sourceActivated = QtCore.Signal(str)
    dismissed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self.paths: dict[str, str] = {}
        self.candidate_uids: tuple[str, ...] = ()
        self._expanded_before_search: set[str] | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setPlaceholderText("Search sources")
        self.search_edit.setAccessibleName("Search sources")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._filter_tree)
        self.search_edit.installEventFilter(self)
        layout.addWidget(self.search_edit)

        self.tree = QtWidgets.QTreeWidget(self)
        self._delegate = _ManagerSourceDelegate(self.tree)
        self.tree.setItemDelegate(self._delegate)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.tree.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.tree.itemClicked.connect(self._activate_item)
        self.tree.installEventFilter(self)
        layout.addWidget(self.tree)
        self.empty_label = QtWidgets.QLabel(self)
        self.empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty_label)

    def refresh(
        self,
        manager: ImageToolManager | None,
        eligibility: Callable[[_ManagedWindowNode], str | None],
    ) -> None:
        expanded = self._expanded_uids()
        had_items = bool(self.items)
        self._delegate.set_manager(manager)
        self.tree.clear()
        self.items.clear()
        self.paths.clear()
        candidates: list[str] = []

        def add_node(
            node: _ManagedWindowNode,
            path: tuple[str, ...],
            *,
            root_label: str | None = None,
        ) -> QtWidgets.QTreeWidgetItem | None:
            reason = eligibility(node)
            label = node.display_text
            if root_label is not None:
                label = root_label
            elif not label:
                label = "(unnamed)"
            item = QtWidgets.QTreeWidgetItem([label])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, node.uid)
            if not node.is_imagetool:
                item.setData(0, _TOOL_TYPE_ROLE, node.type_badge_text)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            if reason is None:
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsSelectable)
                candidates.append(node.uid)
            else:
                item.setForeground(
                    0,
                    self.palette().brush(
                        QtGui.QPalette.ColorGroup.Disabled,
                        QtGui.QPalette.ColorRole.Text,
                    ),
                )
                item.setToolTip(0, reason)
            node_path = (*path, label)
            if manager is not None:
                for uid in node._childtool_indices:
                    child = manager._tool_graph.nodes.get(uid)
                    if child is None or manager._is_figure_node(child):
                        continue
                    child_item = add_node(child, node_path)
                    if child_item is not None:
                        item.addChild(child_item)
            if reason is not None and item.childCount() == 0:
                return None
            self.items[node.uid] = item
            self.paths[node.uid] = " > ".join(node_path)
            return item

        if manager is not None:
            graph = manager._tool_graph
            for index in graph.displayed_indices:
                node = graph.root_wrappers[index]
                item = add_node(node, (), root_label=manager.label_of_imagetool(index))
                if item is not None:
                    self.tree.addTopLevelItem(item)
        self.candidate_uids = tuple(candidates)
        for uid, item in self.items.items():
            item.setExpanded(not had_items or uid in expanded)
        self._filter_tree(self.search_edit.text())

    def select_uid(self, uid: str | None) -> None:
        self.tree.setCurrentItem(self.items.get(uid) if uid is not None else None)
        if uid is not None and (item := self.items.get(uid)) is not None:
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()
            self.tree.scrollToItem(item)

    def _expanded_uids(self) -> set[str]:
        return {uid for uid, item in self.items.items() if item.isExpanded()}

    @QtCore.Slot(str)
    def _filter_tree(self, text: str) -> None:
        query = text.strip().casefold()
        if not query:
            for uid, item in self.items.items():
                item.setHidden(False)
                if self._expanded_before_search is not None:
                    item.setExpanded(uid in self._expanded_before_search)
            self._expanded_before_search = None
        else:
            if self._expanded_before_search is None:
                self._expanded_before_search = self._expanded_uids()

            def filter_item(
                item: QtWidgets.QTreeWidgetItem, *, ancestor_match: bool = False
            ) -> bool:
                badge_text = item.data(0, _TOOL_TYPE_ROLE) or ""
                matches = (
                    ancestor_match
                    or query in item.text(0).casefold()
                    or query in badge_text.casefold()
                )
                child_match = False
                for row in range(item.childCount()):
                    child = item.child(row)
                    if child is not None:
                        child_match |= filter_item(child, ancestor_match=matches)
                item.setHidden(not (matches or child_match))
                item.setExpanded(child_match)
                return matches or child_match

            for row in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(row)
                if item is not None:
                    filter_item(item)
        visible = any(not item.isHidden() for item in self.items.values())
        self.tree.setVisible(visible)
        self.empty_label.setVisible(not visible)
        self.empty_label.setText(
            "No matching sources" if self.candidate_uids else "No compatible sources"
        )

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _activate_item(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        uid = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if uid in self.candidate_uids:
            self.sourceActivated.emit(uid)

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if (
            isinstance(event, QtGui.QKeyEvent)
            and event.type() == QtCore.QEvent.Type.KeyPress
        ):
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self.dismissed.emit()
                return True
            if event.key() in {QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter}:
                item = self.tree.currentItem()
                if item is not None and not item.isHidden():
                    self._activate_item(item, 0)
                return True
            if event.key() in {QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up}:
                self._move_current(
                    1 if event.key() == QtCore.Qt.Key.Key_Down else -1,
                    enter_tree=watched is self.search_edit,
                )
                return True
        return super().eventFilter(watched, event)

    def _move_current(self, step: int, *, enter_tree: bool) -> None:
        visible: list[QtWidgets.QTreeWidgetItem] = []
        for uid in self.candidate_uids:
            item = self.items[uid]
            if item.isHidden():
                continue
            parent = item.parent()
            while parent is not None:
                if parent.isHidden() or not parent.isExpanded():
                    break
                parent = parent.parent()
            if parent is None:
                visible.append(item)
        self.tree.setFocus()
        if not visible:
            return
        current = self.tree.currentItem()
        if current in visible:
            if enter_tree:
                return
            index = visible.index(current) + step
            index = max(0, min(index, len(visible) - 1))
        else:
            index = 0 if step > 0 else len(visible) - 1
        self.tree.setCurrentItem(visible[index])
        self.tree.scrollToItem(visible[index])


class _SourceButton(QtWidgets.QToolButton):
    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is not None and event.key() in {
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
            QtCore.Qt.Key.Key_F4,
        }:
            self.click()
            event.accept()
            return
        super().keyPressEvent(event)

    def minimumSizeHint(self) -> QtCore.QSize:
        size = super().minimumSizeHint()
        size.setWidth(120)
        return size

    def sizeHint(self) -> QtCore.QSize:
        size = super().sizeHint()
        size.setWidth(max(120, min(size.width(), 320)))
        return size

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        option = QtWidgets.QStyleOptionComboBox()
        option.initFrom(self)
        option.frame = True
        if self.isDown():
            option.state |= QtWidgets.QStyle.StateFlag.State_Sunken
        style = typing.cast("QtWidgets.QStyle", self.style())
        edit_rect = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox,
            option,
            QtWidgets.QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        option.currentText = self.fontMetrics().elidedText(
            self.text(), QtCore.Qt.TextElideMode.ElideMiddle, edit_rect.width()
        )
        painter = QtWidgets.QStylePainter(self)
        painter.drawComplexControl(QtWidgets.QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QtWidgets.QStyle.ControlElement.CE_ComboBoxLabel, option)


class ManagerSourceSelector(QtWidgets.QWidget):
    """Select one compatible Manager source while retaining its tree context.

    ``eligibility`` returns ``None`` for a selectable node, or a reason for rejecting
    it. Rejected ancestors remain visible when they contain a selectable descendant.
    Selection uses node UIDs and starts empty. Invalid selections are cleared.
    """

    selectionChanged = QtCore.Signal()
    candidatesChanged = QtCore.Signal()

    def __init__(
        self,
        manager: ImageToolManager | None,
        *,
        eligibility: Callable[[_ManagedWindowNode], str | None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = None if manager is None else weakref.ref(manager)
        self._eligibility = eligibility
        self._current_uid: str | None = None
        self._button = _SourceButton(self)
        self._button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._button.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self._button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._button.clicked.connect(self.show_popup)
        self.setFocusProxy(self._button)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._button)

        self._popup = QtWidgets.QFrame(self, QtCore.Qt.WindowType.Popup)
        self._popup.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        popup_layout = QtWidgets.QVBoxLayout(self._popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        self._content = _ManagerSourceTree(self._popup)
        self._content.sourceActivated.connect(self._accept_source)
        self._content.dismissed.connect(self._popup.hide)
        popup_layout.addWidget(self._content)

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        if manager is not None:
            model = manager.tree_view._model
            for signal in (
                model.rowsInserted,
                model.rowsRemoved,
                model.rowsMoved,
                model.layoutChanged,
                model.modelReset,
                model.dataChanged,
                manager._sigDataReplaced,
            ):
                signal.connect(self._schedule_refresh)
            manager.destroyed.connect(self._schedule_refresh)
        self.refresh()

    @property
    def current_uid(self) -> str | None:
        return self._current_uid

    @property
    def candidate_uids(self) -> tuple[str, ...]:
        return self._content.candidate_uids

    def set_current_uid(self, uid: str | None) -> None:
        uid = uid if uid in self.candidate_uids else None
        changed = uid != self._current_uid
        self._current_uid = uid
        self._button.setText(
            self._content.paths[uid] if uid is not None else "Select source…"
        )
        self._button.setToolTip(self._content.paths[uid] if uid is not None else "")
        self._button.setAccessibleName(self._button.text())
        self._content.select_uid(uid)
        if changed:
            self.selectionChanged.emit()

    @QtCore.Slot()
    def _schedule_refresh(self) -> None:
        self._refresh_timer.start(0)

    @QtCore.Slot()
    def refresh(self) -> None:
        manager = None if self._manager is None else self._manager()
        if manager is not None and not erlab.interactive.utils.qt_is_valid(manager):
            manager = None
        previous = self.candidate_uids
        self._content.refresh(manager, self._eligibility)
        self.set_current_uid(self._current_uid)
        if previous != self.candidate_uids:
            self.candidatesChanged.emit()

    @QtCore.Slot()
    def show_popup(self) -> None:
        self.refresh()
        self._content.search_edit.clear()
        self._content.select_uid(self._current_uid)
        if self._current_uid is None and self.candidate_uids:
            self._content.select_uid(self.candidate_uids[0])
        anchor = self._button.mapToGlobal(QtCore.QPoint(0, self._button.height()))
        screen = QtGui.QGuiApplication.screenAt(anchor) or self.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(max(self.width(), 360), available.width())
        height = min(320, available.height())
        x = max(available.left(), min(anchor.x(), available.right() - width + 1))
        y = anchor.y()
        if y + height > available.bottom() + 1:
            y = self._button.mapToGlobal(QtCore.QPoint()).y() - height
        y = max(available.top(), min(y, available.bottom() - height + 1))
        self._popup.setGeometry(x, y, width, height)
        self._popup.show()
        self._content.search_edit.setFocus()

    @QtCore.Slot(str)
    def _accept_source(self, uid: str) -> None:
        self.refresh()
        if uid in self.candidate_uids:
            self.set_current_uid(uid)
            self._popup.hide()

    def hideEvent(self, event: QtGui.QHideEvent | None) -> None:
        self._popup.hide()
        super().hideEvent(event)
