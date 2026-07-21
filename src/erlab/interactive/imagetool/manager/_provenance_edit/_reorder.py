"""Dialog for arranging provenance steps before one transactional replay."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from erlab.interactive.imagetool._provenance._model import (
        _ProvenanceReorderBlock,
        _ProvenanceReorderBlockRef,
        _ProvenanceReorderSection,
        _ProvenanceReorderSectionRef,
    )

_REORDER_BLOCK_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
_REORDER_MIME_TYPE = "application/x-erlab-provenance-reorder"


def _block_label(block: _ProvenanceReorderBlock) -> str:
    """Return a compact user-facing summary of an atomic operation block."""
    if not block.entries:
        return block.label or "Recorded action"
    labels = tuple(entry.label for entry in block.entries)
    visible_labels = labels[:3]
    label = " → ".join(visible_labels)
    if len(labels) > len(visible_labels):
        label = f"{label} → … (+{len(labels) - len(visible_labels)} more)"
    linked_count = block.ref.stop - block.ref.start
    hidden_count = linked_count - len(labels)
    return (
        label if hidden_count <= 0 else f"{label} (+{hidden_count} linked operations)"
    )


def _block_tooltip(block: _ProvenanceReorderBlock) -> str:
    labels = tuple(entry.label for entry in block.entries)
    if len(labels) <= 1:
        details = labels[0] if labels else _block_label(block)
    else:
        details = "Operations that move together:\n" + "\n".join(
            f"- {label}" for label in labels
        )
    if block.tooltip is None:
        return details
    return f"{details}\n\n{block.tooltip}"


class _ProvenanceReorderListModel(QtCore.QAbstractListModel):
    """Python-backed order for one independently movable provenance section."""

    order_changed = QtCore.Signal()

    def __init__(
        self,
        section: _ProvenanceReorderSection,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.section_ref = section.ref
        self._blocks = list(section.blocks)

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._blocks)

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = int(QtCore.Qt.ItemDataRole.DisplayRole),
    ) -> typing.Any:
        if not index.isValid() or not 0 <= index.row() < len(self._blocks):
            return None
        block = self._blocks[index.row()]
        if role == int(QtCore.Qt.ItemDataRole.DisplayRole):
            return _block_label(block)
        if role == int(QtCore.Qt.ItemDataRole.ToolTipRole):
            return _block_tooltip(block)
        if role == _REORDER_BLOCK_ROLE:
            return block.ref
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsDropEnabled
        return (
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        return QtCore.Qt.DropAction.MoveAction

    def mimeTypes(self) -> list[str]:
        return [_REORDER_MIME_TYPE]

    def mimeData(
        self,
        indexes: Iterable[QtCore.QModelIndex],
    ) -> QtCore.QMimeData:
        mime_data = QtCore.QMimeData()
        if any(index.isValid() for index in indexes):
            mime_data.setData(_REORDER_MIME_TYPE, b"internal-move")
        return mime_data

    def canDropMimeData(
        self,
        data: QtCore.QMimeData | None,
        action: QtCore.Qt.DropAction,
        row: int,
        column: int,
        parent: QtCore.QModelIndex,
    ) -> bool:
        return (
            data is not None
            and data.hasFormat(_REORDER_MIME_TYPE)
            and action == QtCore.Qt.DropAction.MoveAction
            and column in {-1, 0}
            and not parent.isValid()
            and -1 <= row <= len(self._blocks)
        )

    def order(self) -> tuple[_ProvenanceReorderBlockRef, ...]:
        return tuple(block.ref for block in self._blocks)

    def move_row(self, source_row: int, target_row: int) -> bool:
        """Move one row to its final index without transferring Qt item ownership."""
        row_count = len(self._blocks)
        if (
            not 0 <= source_row < row_count
            or not 0 <= target_row < row_count
            or source_row == target_row
        ):
            return False
        destination_child = target_row + 1 if target_row > source_row else target_row
        if not self.beginMoveRows(  # pragma: no cover - valid move is accepted by Qt.
            QtCore.QModelIndex(),
            source_row,
            source_row,
            QtCore.QModelIndex(),
            destination_child,
        ):
            return False
        block = self._blocks.pop(source_row)
        self._blocks.insert(target_row, block)
        self.endMoveRows()
        self.order_changed.emit()
        return True

    def reset_order(
        self,
        order: Sequence[_ProvenanceReorderBlockRef],
    ) -> bool:
        requested = tuple(order)
        current = self.order()
        if requested == current:
            return False
        if len(requested) != len(current) or set(requested) != set(current):
            raise ValueError("Reset order must contain every provenance block once")
        by_ref = {block.ref: block for block in self._blocks}
        self.beginResetModel()
        self._blocks = [by_ref[block_ref] for block_ref in requested]
        self.endResetModel()
        self.order_changed.emit()
        return True


class _ProvenanceReorderListView(QtWidgets.QListView):
    """Flat reorder view whose drops are applied through its list model."""

    order_changed = QtCore.Signal()

    def __init__(
        self,
        section: _ProvenanceReorderSection,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("manager_provenance_reorder_list")
        self.reorder_model = _ProvenanceReorderListModel(section, self)
        self.setModel(self.reorder_model)
        self.reorder_model.order_changed.connect(self.order_changed)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setUniformItemSizes(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)
        if self.reorder_model.rowCount():
            self.setCurrentIndex(self.reorder_model.index(0, 0))

    def move_current(self, offset: int) -> bool:
        index = self.currentIndex()
        if not index.isValid() or offset not in {-1, 1}:
            return False
        target_row = index.row() + offset
        if not self.reorder_model.move_row(index.row(), target_row):
            return False
        moved_index = self.reorder_model.index(target_row, 0)
        self.setCurrentIndex(moved_index)
        self.scrollTo(moved_index)
        return True

    def select_block(self, block_ref: _ProvenanceReorderBlockRef | None) -> None:
        if block_ref is None:
            return
        try:
            row = self.reorder_model.order().index(block_ref)
        except ValueError:
            return
        self.setCurrentIndex(self.reorder_model.index(row, 0))

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        if event is None:
            return
        if event.source() is not self:
            event.ignore()
            return
        source_index = self.currentIndex()
        if not source_index.isValid():
            event.ignore()
            return

        target_index = self.indexAt(event.position().toPoint())
        drop_position = self.dropIndicatorPosition()
        if not target_index.isValid():
            if drop_position != (
                QtWidgets.QAbstractItemView.DropIndicatorPosition.OnViewport
            ):
                event.ignore()
                return
            insertion_row = self.reorder_model.rowCount()
        elif drop_position == (
            QtWidgets.QAbstractItemView.DropIndicatorPosition.AboveItem
        ):
            insertion_row = target_index.row()
        elif drop_position == (
            QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem
        ):
            insertion_row = target_index.row() + 1
        else:
            event.ignore()
            return

        source_row = source_index.row()
        target_row = insertion_row - 1 if insertion_row > source_row else insertion_row
        target_row = min(target_row, self.reorder_model.rowCount() - 1)
        moved = self.reorder_model.move_row(source_row, target_row)
        if moved:
            self.setCurrentIndex(self.reorder_model.index(target_row, 0))
        event.setDropAction(QtCore.Qt.DropAction.MoveAction)
        event.accept()


class _ProvenanceReorderDialog(QtWidgets.QDialog):
    """Collect a provenance permutation without replaying intermediate orders."""

    apply_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        sections: Sequence[_ProvenanceReorderSection],
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
        self._views: list[_ProvenanceReorderListView] = []
        self._views_by_section: dict[
            _ProvenanceReorderSectionRef,
            _ProvenanceReorderListView,
        ] = {}

        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Drag steps to reorder them.",
            self,
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        scope_layout = QtWidgets.QHBoxLayout()
        self.scope_label = QtWidgets.QLabel("Reorder:", self)
        self.scope_label.setObjectName("manager_provenance_reorder_scope_label")
        scope_layout.addWidget(self.scope_label)
        self.scope_combo = QtWidgets.QComboBox(self)
        self.scope_combo.setObjectName("manager_provenance_reorder_scope_combo")
        self.scope_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        scope_layout.addWidget(self.scope_combo, 1)
        layout.addLayout(scope_layout)

        self.list_stack = QtWidgets.QStackedWidget(self)
        self.list_stack.setObjectName("manager_provenance_reorder_list_stack")
        scope_labels = self._disambiguated_scope_labels(sections)
        for section, scope_text in zip(sections, scope_labels, strict=True):
            view = _ProvenanceReorderListView(section, self.list_stack)
            view.order_changed.connect(self._update_controls)
            selection_model = typing.cast(
                "QtCore.QItemSelectionModel",
                view.selectionModel(),
            )
            selection_model.currentChanged.connect(self._update_controls)
            self._views.append(view)
            self._views_by_section[section.ref] = view
            self.scope_combo.addItem(scope_text)
            self.list_stack.addWidget(view)
        self.scope_combo.currentIndexChanged.connect(self._set_current_scope)
        layout.addWidget(self.list_stack, 1)

        show_scope = len(sections) > 1
        self.scope_label.setVisible(show_scope)
        self.scope_combo.setVisible(show_scope)

        controls = QtWidgets.QHBoxLayout()
        self.move_up_button = QtWidgets.QPushButton("Move Up", self)
        self.move_up_button.setObjectName("manager_provenance_reorder_move_up_button")
        self.move_up_button.clicked.connect(lambda: self.current_view.move_current(-1))
        controls.addWidget(self.move_up_button)
        self.move_down_button = QtWidgets.QPushButton("Move Down", self)
        self.move_down_button.setObjectName(
            "manager_provenance_reorder_move_down_button"
        )
        self.move_down_button.clicked.connect(lambda: self.current_view.move_current(1))
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

    @staticmethod
    def _disambiguated_scope_labels(
        sections: Sequence[_ProvenanceReorderSection],
    ) -> tuple[str, ...]:
        base_labels = tuple(section.label for section in sections)
        totals = {label: base_labels.count(label) for label in base_labels}
        seen: dict[str, int] = {}
        output: list[str] = []
        for label in base_labels:
            seen[label] = seen.get(label, 0) + 1
            output.append(
                label if totals[label] == 1 else f"{label} — group {seen[label]}"
            )
        return tuple(output)

    @property
    def current_view(self) -> _ProvenanceReorderListView:
        widget = self.list_stack.currentWidget()
        if not isinstance(widget, _ProvenanceReorderListView):
            raise TypeError("Reorder dialog does not have a current step list")
        return widget

    def view_for_section(
        self,
        section_ref: _ProvenanceReorderSectionRef,
    ) -> _ProvenanceReorderListView:
        return self._views_by_section[section_ref]

    @QtCore.Slot(int)
    def _set_current_scope(self, index: int) -> None:
        if 0 <= index < self.list_stack.count():
            self.list_stack.setCurrentIndex(index)
        self._update_controls()

    def reorder_plan(
        self,
    ) -> dict[
        _ProvenanceReorderSectionRef,
        tuple[_ProvenanceReorderBlockRef, ...],
    ]:
        return {
            section_ref: view.reorder_model.order()
            for section_ref, view in self._views_by_section.items()
        }

    def order_changed(self) -> bool:
        return self.reorder_plan() != self._original_orders

    @QtCore.Slot()
    def reset_order(self) -> None:
        for section_ref, view in self._views_by_section.items():
            index = view.currentIndex()
            current_ref = (
                None
                if not index.isValid()
                else view.reorder_model.data(index, _REORDER_BLOCK_ROLE)
            )
            view.reorder_model.reset_order(self._original_orders[section_ref])
            view.select_block(current_ref)
        self._update_controls()

    @QtCore.Slot()
    def _update_controls(self, *_args: typing.Any) -> None:
        if not self._views:
            index = -1
            count = 0
        else:
            current = self.current_view.currentIndex()
            index = current.row() if current.isValid() else -1
            count = self.current_view.reorder_model.rowCount()
        self.move_up_button.setEnabled(not self._busy and index > 0)
        self.move_down_button.setEnabled(not self._busy and 0 <= index < count - 1)
        changed = self.order_changed()
        self.reset_button.setEnabled(not self._busy and changed)
        self.apply_button.setEnabled(not self._busy and changed)
        self.cancel_button.setEnabled(not self._busy)
        self.scope_combo.setEnabled(not self._busy)
        self.list_stack.setEnabled(not self._busy)

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
