"""Dialog for arranging provenance steps before one transactional replay."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool._provenance._model import (
    _ProvenanceReorderBlock,
    _ProvenanceReorderBlockRef,
    _ProvenanceReorderSection,
    _ProvenanceReorderSectionRef,
)

_REORDER_SECTION_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
_REORDER_BLOCK_ROLE = _REORDER_SECTION_ROLE + 1


class _ProvenanceReorderTree(QtWidgets.QTreeWidget):
    """Tree that only permits moving blocks inside their original section."""

    order_changed = QtCore.Signal()

    def __init__(
        self,
        sections: typing.Sequence[_ProvenanceReorderSection],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("manager_provenance_reorder_tree")
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setItemsExpandable(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)

        for section in sections:
            section_item = QtWidgets.QTreeWidgetItem([section.label])
            section_item.setData(0, _REORDER_SECTION_ROLE, section.ref)
            section_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            )
            section_item.setToolTip(
                0,
                "Steps can be reordered only inside this section. Other recorded or "
                "internal boundaries remain fixed.",
            )
            self.addTopLevelItem(section_item)
            for block in section.blocks:
                section_item.addChild(self._block_item(block))
            section_item.setExpanded(True)

        first_section = self.topLevelItem(0)
        if first_section is not None and first_section.childCount():
            self.setCurrentItem(first_section.child(0))

    @staticmethod
    def _block_item(
        block: _ProvenanceReorderBlock,
    ) -> QtWidgets.QTreeWidgetItem:
        entries = block.entries
        linked_count = (
            len(entries)
            if block.ref.kind == "stage"
            else block.ref.stop - block.ref.start
        )
        label = block.label or entries[0].label
        if linked_count > 1:
            if block.ref.kind == "stage":
                label = f"{label} ({linked_count} operations)"
            else:
                label = f"{label} (+{linked_count - 1} linked steps)"
        item = QtWidgets.QTreeWidgetItem([label])
        item.setData(0, _REORDER_BLOCK_ROLE, block.ref)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        if block.tooltip is not None:
            item.setToolTip(0, block.tooltip)
        elif linked_count > 1:
            item.setToolTip(
                0,
                "This is one atomic operation group:\n"
                + "\n".join(f"- {entry.label}" for entry in entries),
            )
        else:
            item.setToolTip(0, entries[0].label)
        return item

    @staticmethod
    def _is_block_item(item: QtWidgets.QTreeWidgetItem | None) -> bool:
        return item is not None and isinstance(
            item.data(0, _REORDER_BLOCK_ROLE),
            _ProvenanceReorderBlockRef,
        )

    def current_block_item(self) -> QtWidgets.QTreeWidgetItem | None:
        item = self.currentItem()
        return item if self._is_block_item(item) else None

    def move_current(self, offset: int) -> bool:
        item = self.current_block_item()
        if item is None or offset not in {-1, 1}:
            return False
        parent = item.parent()
        if parent is None:
            return False
        source_index = parent.indexOfChild(item)
        target_index = source_index + offset
        if not 0 <= target_index < parent.childCount():
            return False
        moved = parent.takeChild(source_index)
        if moved is None:  # pragma: no cover - valid child index returns the item.
            return False
        parent.insertChild(target_index, moved)
        self.setCurrentItem(moved)
        self.scrollToItem(moved)
        self.order_changed.emit()
        return True

    def section_orders(
        self,
    ) -> dict[
        _ProvenanceReorderSectionRef,
        tuple[_ProvenanceReorderBlockRef, ...],
    ]:
        orders: dict[
            _ProvenanceReorderSectionRef,
            tuple[_ProvenanceReorderBlockRef, ...],
        ] = {}
        for section_index in range(self.topLevelItemCount()):
            section_item = self.topLevelItem(section_index)
            if section_item is None:  # pragma: no cover - valid index is non-null.
                continue
            section_ref = section_item.data(0, _REORDER_SECTION_ROLE)
            if not isinstance(section_ref, _ProvenanceReorderSectionRef):
                continue
            block_refs: list[_ProvenanceReorderBlockRef] = []
            for block_index in range(section_item.childCount()):
                block_item = section_item.child(block_index)
                if block_item is None:  # pragma: no cover - valid index is non-null.
                    continue
                block_ref = block_item.data(0, _REORDER_BLOCK_ROLE)
                if not isinstance(block_ref, _ProvenanceReorderBlockRef):
                    raise TypeError("Reorder dialog contains an invalid step row")
                block_refs.append(block_ref)
            orders[section_ref] = tuple(block_refs)
        return orders

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        if event is None:
            return
        if event.source() is not self:
            event.ignore()
            return
        source_item = self.current_block_item()
        source_parent = None if source_item is None else source_item.parent()
        if source_item is None or source_parent is None:
            event.ignore()
            return

        target_item = self.itemAt(event.position().toPoint())
        if target_item is None:
            event.ignore()
            return
        drop_position = self.dropIndicatorPosition()
        if target_item is source_parent:
            if drop_position != (
                QtWidgets.QAbstractItemView.DropIndicatorPosition.OnItem
            ):
                event.ignore()
                return
        elif (
            target_item.parent() is not source_parent
            or not self._is_block_item(target_item)
            or drop_position
            not in {
                QtWidgets.QAbstractItemView.DropIndicatorPosition.AboveItem,
                QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem,
            }
        ):
            event.ignore()
            return

        super().dropEvent(event)
        if event.isAccepted():
            # QTreeWidget still owns the internal move until dropEvent unwinds. Read
            # the order on the next event-loop turn, after Qt finishes the transfer.
            erlab.interactive.utils.single_shot(self, 0, self.order_changed.emit)


class _ProvenanceReorderDialog(QtWidgets.QDialog):
    """Collect a provenance permutation without replaying intermediate orders."""

    apply_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        start_label: str,
        sections: typing.Sequence[_ProvenanceReorderSection],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("manager_provenance_reorder_dialog")
        self.setWindowTitle("Reorder Provenance Steps")
        self.setModal(True)
        self._busy = False
        self._original_orders = {
            section.ref: tuple(block.ref for block in section.blocks)
            for section in sections
        }

        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Arrange the steps, then choose Apply Order to replay the complete "
            "provenance once. No ImageTool data changes while you arrange rows.",
            self,
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        source_label = QtWidgets.QLabel(f"Fixed source: {start_label}", self)
        source_label.setObjectName("manager_provenance_reorder_source_label")
        source_label.setEnabled(False)
        source_label.setWordWrap(True)
        layout.addWidget(source_label)

        self.tree = _ProvenanceReorderTree(sections, self)
        self.tree.currentItemChanged.connect(self._update_controls)
        self.tree.order_changed.connect(self._update_controls)
        layout.addWidget(self.tree, 1)

        controls = QtWidgets.QHBoxLayout()
        self.move_up_button = QtWidgets.QPushButton("Move Up", self)
        self.move_up_button.setObjectName("manager_provenance_reorder_move_up_button")
        self.move_up_button.clicked.connect(lambda: self.tree.move_current(-1))
        controls.addWidget(self.move_up_button)
        self.move_down_button = QtWidgets.QPushButton("Move Down", self)
        self.move_down_button.setObjectName(
            "manager_provenance_reorder_move_down_button"
        )
        self.move_down_button.clicked.connect(lambda: self.tree.move_current(1))
        controls.addWidget(self.move_down_button)
        controls.addStretch(1)
        self.reset_button = QtWidgets.QPushButton("Reset Order", self)
        self.reset_button.setObjectName("manager_provenance_reorder_reset_button")
        self.reset_button.clicked.connect(self.reset_order)
        controls.addWidget(self.reset_button)
        layout.addLayout(controls)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.apply_button = typing.cast(
            "QtWidgets.QPushButton",
            self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok),
        )
        self.cancel_button = typing.cast(
            "QtWidgets.QPushButton",
            self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel),
        )
        self.apply_button.setText("Apply Order")
        self.apply_button.setObjectName("manager_provenance_reorder_apply_button")
        self.cancel_button.setObjectName("manager_provenance_reorder_cancel_button")
        self.apply_button.clicked.connect(self.apply_requested)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.button_box)

        self.resize(520, 420)
        self._update_controls()

    def reorder_plan(
        self,
    ) -> dict[
        _ProvenanceReorderSectionRef,
        tuple[_ProvenanceReorderBlockRef, ...],
    ]:
        return self.tree.section_orders()

    def order_changed(self) -> bool:
        return self.reorder_plan() != self._original_orders

    @QtCore.Slot()
    def reset_order(self) -> None:
        current_ref = None
        current_item = self.tree.current_block_item()
        if current_item is not None:
            current_ref = current_item.data(0, _REORDER_BLOCK_ROLE)
        for section_index in range(self.tree.topLevelItemCount()):
            section_item = self.tree.topLevelItem(section_index)
            if section_item is None:  # pragma: no cover - valid index is non-null.
                continue
            section_ref = section_item.data(0, _REORDER_SECTION_ROLE)
            original = self._original_orders.get(section_ref)
            if original is None:
                continue
            items = {
                item.data(0, _REORDER_BLOCK_ROLE): item
                for index in range(section_item.childCount())
                if (item := section_item.child(index)) is not None
            }
            section_item.takeChildren()
            for block_ref in original:
                section_item.addChild(items[block_ref])
        if current_ref is not None:
            for item in self.tree.findItems(
                "*",
                QtCore.Qt.MatchFlag.MatchWildcard | QtCore.Qt.MatchFlag.MatchRecursive,
            ):
                if item.data(0, _REORDER_BLOCK_ROLE) == current_ref:
                    self.tree.setCurrentItem(item)
                    break
        self._update_controls()

    @QtCore.Slot()
    def _update_controls(self) -> None:
        item = self.tree.current_block_item()
        parent = None if item is None else item.parent()
        index = -1 if item is None or parent is None else parent.indexOfChild(item)
        count = 0 if parent is None else parent.childCount()
        self.move_up_button.setEnabled(not self._busy and index > 0)
        self.move_down_button.setEnabled(not self._busy and 0 <= index < count - 1)
        changed = self.order_changed()
        self.reset_button.setEnabled(not self._busy and changed)
        self.apply_button.setEnabled(not self._busy and changed)
        self.cancel_button.setEnabled(not self._busy)
        self.tree.setEnabled(not self._busy)

    def set_busy(self, busy: bool) -> None:
        busy = bool(busy)
        if busy == self._busy:
            return
        self._busy = busy
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._update_controls()

    def finish_success(self) -> None:
        self.set_busy(False)
        QtWidgets.QDialog.accept(self)

    def reject(self) -> None:
        if not self._busy:
            super().reject()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        if event is not None and self._busy:
            event.ignore()
            return
        super().closeEvent(event)
