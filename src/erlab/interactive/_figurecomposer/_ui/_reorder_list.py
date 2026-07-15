"""Shared reorderable tree widgets for Figure Composer panes."""

from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

import erlab


def event_requests_context_menu(event: QtGui.QKeyEvent) -> bool:
    """Return whether *event* is the standard keyboard context-menu request."""
    return event.key() == QtCore.Qt.Key.Key_Menu or (
        event.key() == QtCore.Qt.Key.Key_F10
        and bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
    )


class ReorderList(QtWidgets.QTreeWidget):
    """Tree widget that reports stable row identities after internal moves."""

    rows_reordered = QtCore.Signal(object, object, object)

    def __init__(self, id_column: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._reorder_id_column = id_column
        self._rows_reordered_pending = False
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)

    def _row_ids(self) -> tuple[str, ...]:
        row_ids: list[str] = []
        for row in range(self.topLevelItemCount()):
            item = self.topLevelItem(row)
            row_id = (
                None
                if item is None
                else item.data(
                    self._reorder_id_column,
                    QtCore.Qt.ItemDataRole.UserRole,
                )
            )
            if not isinstance(row_id, str):
                return ()
            row_ids.append(row_id)
        return tuple(row_ids)

    def _queue_rows_reordered(self, *_args: object) -> None:
        if self._rows_reordered_pending:
            return
        self._rows_reordered_pending = True
        # Let QTreeWidget finish transferring ownership of the dragged items before
        # the owning pane refreshes this view.
        erlab.interactive.utils.single_shot(self, 0, self._emit_rows_reordered)

    def _emit_rows_reordered(self) -> None:
        self._rows_reordered_pending = False
        row_ids = self._row_ids()
        if not row_ids or len(set(row_ids)) != len(row_ids):
            return
        current_id = None
        current_item = self.currentItem()
        if current_item is not None:
            candidate = current_item.data(
                self._reorder_id_column,
                QtCore.Qt.ItemDataRole.UserRole,
            )
            if isinstance(candidate, str):
                current_id = candidate
        selected_ids = frozenset(
            row_id
            for item in self.selectedItems()
            if isinstance(
                row_id := item.data(
                    self._reorder_id_column,
                    QtCore.Qt.ItemDataRole.UserRole,
                ),
                str,
            )
        )
        self.rows_reordered.emit(row_ids, selected_ids, current_id)

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        if event is None:
            return
        if event.source() is not self:
            event.ignore()
            return
        super().dropEvent(event)
        if event.isAccepted():
            self._queue_rows_reordered()
