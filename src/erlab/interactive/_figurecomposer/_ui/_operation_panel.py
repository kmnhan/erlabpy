"""Recipe-step list and actions for Figure Composer."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._figurecomposer._ui._operation_editor import (
    FigureOperationEditor,
)
from erlab.interactive._figurecomposer._ui._reorder_list import (
    ReorderList,
    event_requests_context_menu,
)
from erlab.interactive._figurecomposer._ui._widgets import (
    _AxesTargetItemDelegate,
    _step_toolbar_button,
)

if typing.TYPE_CHECKING:
    from collections.abc import Collection, Sequence


_OPERATION_LIST_STEP_COLUMN = 0
_OPERATION_LIST_TARGET_COLUMN = 1
_OPERATION_LIST_STATUS_COLUMN = 2
_OPERATION_LIST_TARGET_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_OPERATION_LIST_STATUS_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2


class FigureOperationAction(typing.NamedTuple):
    """Presentation state for one Add Step menu action."""

    action_id: str
    text: str
    tooltip: str


class FigureOperationRow(typing.NamedTuple):
    """Immutable presentation state for one recipe-step row."""

    operation_id: str
    display: str
    enabled: bool
    tooltip: str
    target_descriptor: tuple[object, ...]
    target_description: str
    status: str
    status_codes: tuple[str, ...]
    status_tooltip: str


class _FigureComposerOperationList(ReorderList):
    copy_requested = QtCore.Signal()
    cut_requested = QtCore.Signal()
    paste_requested = QtCore.Signal()
    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(_OPERATION_LIST_STEP_COLUMN, parent)
        self.setColumnCount(3)
        self.setHeaderLabels(("Step", "Target", "Status"))
        self.setRootIsDecorated(False)
        self.setItemsExpandable(False)
        self.setIndentation(0)
        self.setUniformRowHeights(True)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def _operation_ids(self) -> tuple[str, ...]:
        return self._row_ids()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if event_requests_context_menu(event):
            item = self.currentItem()
            rect = self.visualItemRect(item) if item is not None else QtCore.QRect()
            self.context_menu_requested.emit(rect.center())
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_requested.emit()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Cut):
            self.cut_requested.emit()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            self.paste_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class FigureOperationPanel(QtWidgets.QWidget):
    """Recipe-step view that emits stable-ID operation intentions."""

    add_requested = QtCore.Signal(str)
    copy_requested = QtCore.Signal()
    cut_requested = QtCore.Signal()
    paste_requested = QtCore.Signal()
    delete_requested = QtCore.Signal()
    duplicate_requested = QtCore.Signal()
    move_requested = QtCore.Signal(int)
    reorder_requested = QtCore.Signal(object, object, object)
    enabled_requested = QtCore.Signal(str, bool)
    selection_changed = QtCore.Signal(object, object)

    def __init__(
        self,
        editor_tabs: QtWidgets.QTabWidget,
        add_actions: Sequence[FigureOperationAction],
    ) -> None:
        super().__init__(editor_tabs)
        self._operation_viewport: QtWidgets.QWidget | None = None
        self.setObjectName("figureComposerRecipePage")
        self._selection_input_event = False
        self._multi_select_event = False
        self._emitted_selection_state: tuple[str | None, frozenset[str]] | None = None
        self._selection_notification_pending = False
        self._context_menu: QtWidgets.QMenu | None = None
        self._can_duplicate = False
        self._can_move_up = False
        self._can_move_down = False
        self._target_delegate: _AxesTargetItemDelegate | None = None
        self._closing = False
        self._build_ui(editor_tabs, add_actions)

    def _build_ui(
        self,
        editor_tabs: QtWidgets.QTabWidget,
        add_actions: Sequence[FigureOperationAction],
    ) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setSpacing(4)
        self.add_step_button = _step_toolbar_button(
            self,
            "figureComposerAddStepButton",
            "Add Step ▾",
            "Add a plotting, ERLab method, Axes method, or Python step.",
        )
        self.add_step_button.setProperty("uses_inline_menu_arrow", True)
        self.add_step_menu = QtWidgets.QMenu(self.add_step_button)
        self.add_step_menu.setObjectName("figureComposerAddStepMenu")
        for action_state in add_actions:
            action = QtGui.QAction(action_state.text, self.add_step_menu)
            action.setData(action_state.action_id)
            action.setToolTip(action_state.tooltip)
            action.triggered.connect(
                lambda _checked=False, action_id=action_state.action_id: (
                    self.add_requested.emit(action_id)
                )
            )
            self.add_step_menu.addAction(action)
        self.add_step_button.clicked.connect(self._show_add_menu)
        action_layout.addWidget(self.add_step_button)

        self.copy_button = _step_toolbar_button(
            self,
            "figureComposerCopyStepButton",
            "Copy",
            "Copy the selected recipe step or steps.",
        )
        self.copy_button.clicked.connect(self.copy_requested)
        action_layout.addWidget(self.copy_button)
        self.cut_button = _step_toolbar_button(
            self,
            "figureComposerCutStepButton",
            "Cut",
            "Cut the selected recipe step or steps.",
        )
        self.cut_button.clicked.connect(self.cut_requested)
        action_layout.addWidget(self.cut_button)
        self.paste_button = _step_toolbar_button(
            self,
            "figureComposerPasteStepButton",
            "Paste",
            "Paste copied recipe steps after the current selection.",
        )
        self.paste_button.clicked.connect(self.paste_requested)
        action_layout.addWidget(self.paste_button)
        self.delete_button = _step_toolbar_button(
            self,
            "figureComposerDeleteStepButton",
            "Delete",
            "Remove the selected recipe step or steps.",
        )
        self.delete_button.clicked.connect(self.delete_requested)
        action_layout.addWidget(self.delete_button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter.setObjectName("figureComposerRecipeSplitter")
        self.splitter.setChildrenCollapsible(False)
        layout.addWidget(self.splitter, 1)

        self.operation_list = _FigureComposerOperationList(self)
        self.operation_list.setObjectName("figureComposerOperationList")
        self.operation_list.copy_requested.connect(self.copy_requested)
        self.operation_list.cut_requested.connect(self.cut_requested)
        self.operation_list.paste_requested.connect(self.paste_requested)
        self.operation_list.context_menu_requested.connect(self._show_context_menu)
        self.operation_list.rows_reordered.connect(self.reorder_requested)
        self.operation_list.currentItemChanged.connect(self._current_item_changed)
        self.operation_list.itemSelectionChanged.connect(self._selection_did_change)
        self.operation_list.itemChanged.connect(self._item_changed)
        self.operation_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.operation_list.setMinimumHeight(140)
        self.operation_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.operation_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        header = typing.cast("QtWidgets.QHeaderView", self.operation_list.header())
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        header.setSectionResizeMode(
            _OPERATION_LIST_STEP_COLUMN,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        header.setSectionResizeMode(
            _OPERATION_LIST_TARGET_COLUMN,
            QtWidgets.QHeaderView.ResizeMode.Interactive,
        )
        header.setSectionResizeMode(
            _OPERATION_LIST_STATUS_COLUMN,
            QtWidgets.QHeaderView.ResizeMode.Interactive,
        )
        header.resizeSection(_OPERATION_LIST_TARGET_COLUMN, 72)
        header.resizeSection(_OPERATION_LIST_STATUS_COLUMN, 88)
        self.splitter.addWidget(self.operation_list)

        self.editor = FigureOperationEditor(
            editor_tabs,
            (
                self.add_step_button,
                self.copy_button,
                self.cut_button,
                self.paste_button,
                self.delete_button,
                self.operation_list,
            ),
            self,
        )
        self.splitter.addWidget(self.editor)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes((140, 410))

        operation_viewport = typing.cast(
            "QtWidgets.QWidget", self.operation_list.viewport()
        )
        self._operation_viewport = operation_viewport
        operation_viewport.installEventFilter(self)

    @staticmethod
    def _operation_id_for_item(
        item: QtWidgets.QTreeWidgetItem | None,
    ) -> str | None:
        if item is None:
            return None
        operation_id = item.data(
            _OPERATION_LIST_STEP_COLUMN, QtCore.Qt.ItemDataRole.UserRole
        )
        return operation_id if isinstance(operation_id, str) else None

    def selected_ids(self) -> frozenset[str]:
        return frozenset(
            operation_id
            for item in self.operation_list.selectedItems()
            if (operation_id := self._operation_id_for_item(item)) is not None
        )

    def current_id(self) -> str | None:
        return self._operation_id_for_item(self.operation_list.currentItem())

    def current_index(self) -> int:
        item = self.operation_list.currentItem()
        return -1 if item is None else self.operation_list.indexOfTopLevelItem(item)

    def select_row(self, index: int) -> None:
        """Select a row and notify consumers through the normal selection signals."""
        item = self.operation_list.topLevelItem(index)
        if item is None:
            self.operation_list.setCurrentIndex(QtCore.QModelIndex())
        else:
            self.operation_list.setCurrentItem(item)

    def set_current_row(self, index: int, *, preserve_selection: bool = True) -> None:
        selected_ids = self.selected_ids() if preserve_selection else frozenset()
        was_blocked = self.operation_list.blockSignals(True)
        try:
            if not preserve_selection:
                self.operation_list.clearSelection()
            item = self.operation_list.topLevelItem(index)
            if item is None:
                self.operation_list.setCurrentIndex(QtCore.QModelIndex())
            else:
                self.operation_list.setCurrentItem(item)
            if preserve_selection and selected_ids:
                self._apply_selected_ids(selected_ids)
        finally:
            self.operation_list.blockSignals(was_blocked)
        self._synchronize_selection_cache()

    def install_target_delegate(self, color_source: QtWidgets.QWidget) -> None:
        """Install the operation-target preview delegate owned by this panel."""
        delegate = _AxesTargetItemDelegate(
            int(_OPERATION_LIST_TARGET_ROLE), color_source, self.operation_list
        )
        self.operation_list.setItemDelegateForColumn(
            _OPERATION_LIST_TARGET_COLUMN, delegate
        )
        self._target_delegate = delegate

    def set_selected_ids(self, operation_ids: Collection[str]) -> None:
        selected = set(operation_ids)
        was_blocked = self.operation_list.blockSignals(True)
        try:
            self._apply_selected_ids(selected)
        finally:
            self.operation_list.blockSignals(was_blocked)
        self._synchronize_selection_cache()

    def _apply_selected_ids(self, operation_ids: Collection[str]) -> None:
        """Apply selection while the caller owns signal blocking and cache sync."""
        selected = set(operation_ids)
        for row in range(self.operation_list.topLevelItemCount()):
            item = self.operation_list.topLevelItem(row)
            if item is not None:
                item.setSelected(self._operation_id_for_item(item) in selected)

    def set_rows(
        self,
        rows: Sequence[FigureOperationRow],
        *,
        selected_ids: Collection[str],
        current_id: str | None,
    ) -> None:
        """Render operation rows while preserving stable-ID selection."""
        operation_ids = tuple(row.operation_id for row in rows)
        reuse_items = self.operation_list._operation_ids() == operation_ids
        was_blocked = self.operation_list.blockSignals(True)
        try:
            if not reuse_items:
                self.operation_list.clear()
                self.operation_list.addTopLevelItems(
                    [QtWidgets.QTreeWidgetItem() for _row in rows]
                )
            for index, row in enumerate(rows):
                item = typing.cast(
                    "QtWidgets.QTreeWidgetItem",
                    self.operation_list.topLevelItem(index),
                )
                self._set_item(item, row)
            if not reuse_items:
                self._apply_selected_ids(selected_ids)
                if current_id is None:
                    self.operation_list.setCurrentIndex(QtCore.QModelIndex())
                else:
                    for index, row in enumerate(rows):
                        if row.operation_id == current_id:
                            self.operation_list.setCurrentItem(
                                self.operation_list.topLevelItem(index)
                            )
                            break
        finally:
            self.operation_list.blockSignals(was_blocked)
        self._synchronize_selection_cache()

    @staticmethod
    def _set_item(item: QtWidgets.QTreeWidgetItem, row: FigureOperationRow) -> None:
        item.setText(_OPERATION_LIST_STEP_COLUMN, row.display)
        item.setText(_OPERATION_LIST_STATUS_COLUMN, row.status)
        item.setFlags(
            (
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled
        )
        item.setCheckState(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.CheckState.Checked
            if row.enabled
            else QtCore.Qt.CheckState.Unchecked,
        )
        item.setData(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.ItemDataRole.UserRole,
            row.operation_id,
        )
        item.setData(
            _OPERATION_LIST_TARGET_COLUMN,
            _OPERATION_LIST_TARGET_ROLE,
            row.target_descriptor,
        )
        item.setData(
            _OPERATION_LIST_STATUS_COLUMN,
            _OPERATION_LIST_STATUS_ROLE,
            row.status_codes,
        )
        item.setSizeHint(_OPERATION_LIST_STEP_COLUMN, QtCore.QSize(0, 22))
        item.setToolTip(_OPERATION_LIST_STEP_COLUMN, row.tooltip)
        item.setData(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            row.tooltip,
        )
        item.setToolTip(_OPERATION_LIST_TARGET_COLUMN, row.target_description)
        item.setData(
            _OPERATION_LIST_TARGET_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            row.target_description,
        )
        item.setToolTip(_OPERATION_LIST_STATUS_COLUMN, row.status_tooltip)
        item.setData(
            _OPERATION_LIST_STATUS_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            row.status_tooltip,
        )
        item.setForeground(
            _OPERATION_LIST_STATUS_COLUMN,
            QtGui.QBrush(QtGui.QColor("darkRed"))
            if row.status_codes
            else QtGui.QBrush(),
        )

    def set_action_availability(
        self,
        *,
        selection: bool,
        paste: bool,
        duplicate: bool,
        move_up: bool,
        move_down: bool,
    ) -> None:
        self.delete_button.setEnabled(selection)
        self.copy_button.setEnabled(selection)
        self.cut_button.setEnabled(selection)
        self.paste_button.setEnabled(paste)
        self._can_duplicate = duplicate
        self._can_move_up = move_up
        self._can_move_down = move_down

    @QtCore.Slot()
    def _show_add_menu(self) -> None:
        self.add_step_menu.popup(
            self.add_step_button.mapToGlobal(
                QtCore.QPoint(0, self.add_step_button.height())
            )
        )

    @QtCore.Slot(QtCore.QPoint)
    def _show_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu("Recipe Steps", self.operation_list)
        self._context_menu = menu

        def clear_menu(_destroyed: object | None = None) -> None:
            if self._context_menu is menu:
                self._context_menu = None

        menu.destroyed.connect(clear_menu)
        menu.aboutToHide.connect(menu.deleteLater)
        for text, object_name, enabled, signal in (
            (
                "Copy",
                "figureComposerContextCopyStepsAction",
                self.copy_button.isEnabled(),
                self.copy_requested,
            ),
            (
                "Cut",
                "figureComposerContextCutStepsAction",
                self.cut_button.isEnabled(),
                self.cut_requested,
            ),
            (
                "Paste",
                "figureComposerContextPasteStepsAction",
                self.paste_button.isEnabled(),
                self.paste_requested,
            ),
        ):
            action = QtGui.QAction(text, menu)
            action.setObjectName(object_name)
            action.setEnabled(enabled)
            action.triggered.connect(signal)
            menu.addAction(action)
        menu.addSeparator()
        duplicate_action = QtGui.QAction("Duplicate", menu)
        duplicate_action.setObjectName("figureComposerContextDuplicateStepAction")
        duplicate_action.setEnabled(self._can_duplicate)
        duplicate_action.triggered.connect(self.duplicate_requested)
        menu.addAction(duplicate_action)
        for text, object_name, offset, enabled in (
            (
                "Move Up",
                "figureComposerContextMoveStepUpAction",
                -1,
                self._can_move_up,
            ),
            (
                "Move Down",
                "figureComposerContextMoveStepDownAction",
                1,
                self._can_move_down,
            ),
        ):
            action = QtGui.QAction(text, menu)
            action.setObjectName(object_name)
            action.setEnabled(enabled)
            action.triggered.connect(
                lambda _checked=False, direction=offset: self.move_requested.emit(
                    direction
                )
            )
            menu.addAction(action)
        delete_action = QtGui.QAction("Delete", menu)
        delete_action.setObjectName("figureComposerContextDeleteStepAction")
        delete_action.setEnabled(self.delete_button.isEnabled())
        delete_action.triggered.connect(self.delete_requested)
        menu.addAction(delete_action)
        viewport = typing.cast("QtWidgets.QWidget", self.operation_list.viewport())
        menu.popup(viewport.mapToGlobal(position))

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, QtWidgets.QTreeWidgetItem)
    def _current_item_changed(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        if current is not None and not self._multi_select_event:
            operation_id = self._operation_id_for_item(current)
            if operation_id is not None:
                was_blocked = self.operation_list.blockSignals(True)
                try:
                    self._apply_selected_ids((operation_id,))
                finally:
                    self.operation_list.blockSignals(was_blocked)
        self._selection_did_change()

    @QtCore.Slot()
    def _selection_did_change(self) -> None:
        if self._closing:
            return
        if self._selection_input_event:
            if not self._selection_notification_pending:
                self._selection_notification_pending = True
                erlab.interactive.utils.single_shot(
                    self, 0, self._emit_selection_change
                )
            return
        self._emit_selection_change()

    def _emit_selection_change(self) -> None:
        self._selection_notification_pending = False
        if self._closing or not erlab.interactive.utils.qt_is_valid(self):
            return
        state = (self.current_id(), self.selected_ids())
        if state == self._emitted_selection_state:
            return
        self._emitted_selection_state = state
        self.selection_changed.emit(*state)

    def _synchronize_selection_cache(self) -> None:
        """Record a signal-blocked selection without notifying consumers."""
        self._emitted_selection_state = (self.current_id(), self.selected_ids())

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if column != _OPERATION_LIST_STEP_COLUMN:
            return
        operation_id = self._operation_id_for_item(item)
        if operation_id is not None:
            self.enabled_requested.emit(
                operation_id,
                item.checkState(_OPERATION_LIST_STEP_COLUMN)
                == QtCore.Qt.CheckState.Checked,
            )

    @staticmethod
    def _modifiers_enable_multi_selection(
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> bool:
        multi_modifiers = (
            QtCore.Qt.KeyboardModifier.ShiftModifier
            | QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        )
        return bool(modifiers & multi_modifiers)

    def _clear_selection_input_state(self) -> None:
        if erlab.interactive.utils.qt_is_valid(self):
            self._multi_select_event = False
            self._selection_input_event = False

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if (
            self._operation_viewport is not None
            and watched is self._operation_viewport
            and event is not None
            and event.type()
            in {
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.MouseButtonDblClick,
                QtCore.QEvent.Type.KeyPress,
            }
        ):
            input_event = typing.cast("QtGui.QInputEvent", event)
            self._selection_input_event = True
            self._multi_select_event = self._modifiers_enable_multi_selection(
                input_event.modifiers()
            )
            erlab.interactive.utils.single_shot(
                self, 0, self._clear_selection_input_state
            )
        return super().eventFilter(watched, event)

    def release(self) -> None:
        """Detach event filters before tool teardown."""
        self._closing = True
        viewport = self._operation_viewport
        self._operation_viewport = None
        self._multi_select_event = False
        self._selection_input_event = False
        self._selection_notification_pending = False
        if viewport is not None and erlab.interactive.utils.qt_is_valid(self, viewport):
            viewport.removeEventFilter(self)
        self.editor.release()

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self.release()
        if event is not None:
            super().closeEvent(event)
