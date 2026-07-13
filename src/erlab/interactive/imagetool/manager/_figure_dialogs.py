"""Manager-owned dialogs for Figure Composer workflows."""

from __future__ import annotations

__all__ = [
    "_FIGURE_DIALOG_ADD_SOURCE",
    "_FIGURE_DIALOG_ADD_STEP",
    "_FIGURE_DIALOG_NEW",
    "_FIGURE_DIALOG_REPLACE_SOURCE",
    "_AppendFigureTargetDialog",
    "_FigureSourcePickerDialog",
]

import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive._figurecomposer
from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_axis_display_names,
    _gridspec_valid_axes_ids,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager

_NEW_FIGURE_TARGET = "__new_figure__"
_FIGURE_DIALOG_NEW = "new_figure"
_FIGURE_DIALOG_ADD_STEP = "add_step"
_FIGURE_DIALOG_ADD_SOURCE = "add_source"
_FIGURE_DIALOG_REPLACE_SOURCE = "replace_source"


class _AppendFigureTargetDialog(QtWidgets.QDialog):
    """Prompt for a Figure Composer target figure and source workflow."""

    def __init__(
        self,
        manager: ImageToolManager,
        figure_uids: tuple[str, ...],
        operation: erlab.interactive._figurecomposer.FigureOperationState | None,
        *,
        allow_new_figure: bool = False,
        source_count: int = 1,
        selected_figure_uid: str | None = None,
    ) -> None:
        from erlab.interactive._figurecomposer._widgets import (
            _AxesSelectorWidget,
            _GridSpecViewWidget,
        )

        super().__init__(manager)
        self._manager = manager
        self._figure_uids = figure_uids
        self._operation = operation
        self._allow_new_figure = allow_new_figure
        self._source_count = source_count
        self.setObjectName("managerAppendFigureTargetDialog")
        self.setWindowTitle("Add to Figure" if allow_new_figure else "Append to Figure")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        layout.addLayout(form)

        self.action_combo = QtWidgets.QComboBox(self)
        self.action_combo.setObjectName("managerFigureActionCombo")
        if allow_new_figure:
            self.action_combo.addItem("New Figure", _FIGURE_DIALOG_NEW)
            self.action_combo.addItem("Add New Step", _FIGURE_DIALOG_ADD_STEP)
            self.action_combo.addItem("Add Source Only", _FIGURE_DIALOG_ADD_SOURCE)
            self.action_combo.addItem("Replace Source", _FIGURE_DIALOG_REPLACE_SOURCE)
            form.addRow("Action", self.action_combo)
        else:
            self.action_combo.addItem("Add New Step", _FIGURE_DIALOG_ADD_STEP)
            self.action_combo.setVisible(False)

        self.figure_combo = QtWidgets.QComboBox(self)
        self.figure_combo.setObjectName("managerAppendFigureCombo")
        for uid in figure_uids:
            self.figure_combo.addItem(manager._child_node(uid).display_text, uid)
        self.figure_combo.setVisible(self.figure_combo.count() > 1)
        if self.figure_combo.count() > 1:
            form.addRow("Figure", self.figure_combo)
            self.figure_field_widget: QtWidgets.QWidget = self.figure_combo
        else:
            figure_label = QtWidgets.QLabel(
                manager._child_node(figure_uids[0]).display_text, self
            )
            form.addRow(
                "Figure",
                figure_label,
            )
            self.figure_field_widget = figure_label
        self.figure_label = form.labelForField(self.figure_field_widget)
        if selected_figure_uid in figure_uids:
            figure_index = self.figure_combo.findData(selected_figure_uid)
            if figure_index >= 0:
                self.figure_combo.setCurrentIndex(figure_index)

        self.source_combo = QtWidgets.QComboBox(self)
        self.source_combo.setObjectName("managerReplaceFigureSourceCombo")
        form.addRow("Source", self.source_combo)
        self.source_label = form.labelForField(self.source_combo)

        self.selector_stack = QtWidgets.QStackedWidget(self)
        self.selector_stack.setObjectName("managerAppendAxesSelectorStack")
        self.axes_selector = _AxesSelectorWidget(self)
        self.axes_selector.setObjectName("managerAppendAxesSelector")
        self.gridspec_axes_selector = _GridSpecViewWidget(self, mode="select")
        self.gridspec_axes_selector.setObjectName("managerAppendGridSpecAxesSelector")
        self.selector_stack.addWidget(self.axes_selector)
        self.selector_stack.addWidget(self.gridspec_axes_selector)
        form.addRow("Axes", self.selector_stack)
        self.axes_label = form.labelForField(self.selector_stack)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setObjectName("managerAppendAxesStatusLabel")
        layout.addWidget(self.status_label)

        action_layout = QtWidgets.QHBoxLayout()
        self.all_axes_button = QtWidgets.QToolButton(self)
        self.all_axes_button.setObjectName("managerAppendAllAxesButton")
        self.all_axes_button.setText("All axes")
        self.all_axes_button.setToolTip("Select every available axes in this figure.")
        action_layout.addWidget(self.all_axes_button)
        self.clear_axes_button = QtWidgets.QToolButton(self)
        self.clear_axes_button.setObjectName("managerAppendClearAxesButton")
        self.clear_axes_button.setText("Clear")
        self.clear_axes_button.setToolTip("Clear the current axes selection.")
        action_layout.addWidget(self.clear_axes_button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.setObjectName("managerAppendFigureButtonBox")
        layout.addWidget(self.button_box)

        self.action_combo.currentIndexChanged.connect(self._action_changed)
        self.figure_combo.currentIndexChanged.connect(self._figure_changed)
        self.source_combo.currentIndexChanged.connect(self._selection_changed)
        self.axes_selector.sigSelectionChanged.connect(self._selection_changed)
        self.axes_selector.sigAddRowRequested.connect(self._add_subplot_row)
        self.axes_selector.sigAddColumnRequested.connect(self._add_subplot_column)
        self.gridspec_axes_selector.sigSelectionChanged.connect(self._selection_changed)
        self.all_axes_button.clicked.connect(self._select_all_axes)
        self.clear_axes_button.clicked.connect(self._clear_axes)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self._set_default_action(selected_figure_uid)
        self._figure_changed()

    def figure_uid(self) -> str:
        uid = self.figure_combo.currentData()
        if isinstance(uid, str) and uid != _NEW_FIGURE_TARGET:
            return uid
        return self._figure_uids[0]

    def selected_action(self) -> str:
        action = self.action_combo.currentData()
        return action if isinstance(action, str) else _FIGURE_DIALOG_ADD_STEP

    def is_new_figure(self) -> bool:
        return self.selected_action() == _FIGURE_DIALOG_NEW

    def is_add_source_only(self) -> bool:
        return self.selected_action() == _FIGURE_DIALOG_ADD_SOURCE

    def is_replace_source(self) -> bool:
        return self.selected_action() == _FIGURE_DIALOG_REPLACE_SOURCE

    def selected_source_alias(self) -> str | None:
        alias = self.source_combo.currentData()
        return alias if isinstance(alias, str) else None

    def axes_selection(
        self,
    ) -> erlab.interactive._figurecomposer.FigureAxesSelectionState | None:
        if self.selected_action() != _FIGURE_DIALOG_ADD_STEP:
            return None
        tool = self._figure_tool()
        if tool is None:
            return None
        setup = tool.tool_status.setup
        if setup.layout_mode == "gridspec":
            axes_ids = self.gridspec_axes_selector.selected_axes_ids()
            if not axes_ids:
                return None
            return erlab.interactive._figurecomposer.FigureAxesSelectionState(
                axes_ids=axes_ids
            )
        axes = self.axes_selector.selected_axes()
        if not axes:
            return None
        return erlab.interactive._figurecomposer.FigureAxesSelectionState(axes=axes)

    def selected_target(
        self,
    ) -> tuple[str, erlab.interactive._figurecomposer.FigureAxesSelectionState] | None:
        if self.selected_action() != _FIGURE_DIALOG_ADD_STEP:
            return None
        selection = self.axes_selection()
        if selection is None:
            return None
        return self.figure_uid(), selection

    def _set_default_action(self, selected_figure_uid: str | None) -> None:
        if not self._allow_new_figure:
            return
        default_action = _FIGURE_DIALOG_ADD_STEP
        if (
            selected_figure_uid in self._figure_uids
            and self._source_count == 1
            and self._figure_source_count(selected_figure_uid) == 1
        ):
            default_action = _FIGURE_DIALOG_REPLACE_SOURCE
        action_index = self.action_combo.findData(default_action)
        if action_index >= 0:
            self.action_combo.setCurrentIndex(action_index)

    def _figure_source_count(self, figure_uid: str | None) -> int:
        if figure_uid is None:
            return 0
        current_index = self.figure_combo.currentIndex()
        figure_index = self.figure_combo.findData(figure_uid)
        if figure_index < 0:
            return 0
        try:
            self.figure_combo.setCurrentIndex(figure_index)
            tool = self._figure_tool()
            if tool is None:
                return 0
            return len(tool.source_states())
        finally:
            self.figure_combo.setCurrentIndex(current_index)

    def _action_requires_figure(self) -> bool:
        return self.selected_action() in {
            _FIGURE_DIALOG_ADD_STEP,
            _FIGURE_DIALOG_ADD_SOURCE,
            _FIGURE_DIALOG_REPLACE_SOURCE,
        }

    def _action_uses_axes(self) -> bool:
        return self.selected_action() == _FIGURE_DIALOG_ADD_STEP

    def _refresh_source_combo(self) -> None:
        self.source_combo.blockSignals(True)
        try:
            self.source_combo.clear()
            tool = self._figure_tool()
            if tool is None:
                return
            for source in tool.source_states():
                name = source.name
                display = tool._source_display_name(name)
                item_text = display if name in display else f"{display} ({name})"
                self.source_combo.addItem(item_text, name)
                index = self.source_combo.count() - 1
                self.source_combo.setItemData(
                    index,
                    tool._source_tooltip(name),
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )
        finally:
            self.source_combo.blockSignals(False)

    @QtCore.Slot()
    def _figure_changed(self) -> None:
        self._refresh_source_combo()
        if not self._action_requires_figure():
            self._selection_changed()
            return
        self.selector_stack.setVisible(True)
        self.all_axes_button.setVisible(True)
        self.clear_axes_button.setVisible(True)
        tool = self._figure_tool()
        if tool is None:
            self._selection_changed()
            return
        setup = tool.tool_status.setup
        if setup.layout_mode == "gridspec":
            axes_ids = _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup))
            labels = dict(
                zip(
                    axes_ids,
                    _gridspec_axis_display_names(setup, axes_ids),
                    strict=True,
                )
            )
            self.gridspec_axes_selector.set_layout(setup.gridspec.root, labels)
            self.selector_stack.setCurrentWidget(self.gridspec_axes_selector)
            self.gridspec_axes_selector.set_selected_axes_ids(
                self._default_gridspec_axes(axes_ids)
            )
        else:
            labels = {
                (row, col): f"{row}, {col}"
                for row in range(setup.nrows)
                for col in range(setup.ncols)
            }
            self.axes_selector.set_grid(setup.nrows, setup.ncols, labels)
            self.selector_stack.setCurrentWidget(self.axes_selector)
            self.axes_selector.set_selected_axes(
                self._default_subplot_axes(
                    self._manager._figure_all_axes(setup.nrows, setup.ncols)
                )
            )
        self._selection_changed()

    @QtCore.Slot()
    def _action_changed(self) -> None:
        self._figure_changed()

    @QtCore.Slot()
    def _selection_changed(self, *_args: typing.Any) -> None:
        action = self.selected_action()
        new_figure = action == _FIGURE_DIALOG_NEW
        uses_figure = self._action_requires_figure()
        uses_axes = self._action_uses_axes()
        replace_source = action == _FIGURE_DIALOG_REPLACE_SOURCE
        figure_tool = self._figure_tool() if uses_figure else None
        has_figure = figure_tool is not None

        figure_visible = uses_figure
        if self.figure_label is not None:
            self.figure_label.setVisible(figure_visible)
        self.figure_field_widget.setVisible(
            figure_visible and self.figure_combo.count() <= 1
        )
        self.figure_combo.setVisible(figure_visible and self.figure_combo.count() > 1)

        if self.axes_label is not None:
            self.axes_label.setVisible(uses_axes and has_figure)
        self.selector_stack.setVisible(uses_axes and has_figure)
        self.all_axes_button.setVisible(uses_axes and has_figure)
        self.clear_axes_button.setVisible(uses_axes and has_figure)

        if self.source_label is not None:
            self.source_label.setVisible(replace_source and has_figure)
        self.source_combo.setVisible(replace_source and has_figure)

        if new_figure:
            self._ok_button().setEnabled(True)
            self.status_label.setText("A new figure will be created.")
            return
        if not has_figure:
            self._ok_button().setEnabled(False)
            self.status_label.setText("The selected figure is unavailable.")
            return
        if action == _FIGURE_DIALOG_ADD_SOURCE:
            self._ok_button().setEnabled(True)
            self.status_label.setText("Source data will be added without a new step.")
            return
        if replace_source:
            selected_alias = self.selected_source_alias()
            enabled = self._source_count == 1 and selected_alias is not None
            self._ok_button().setEnabled(enabled)
            if self._source_count != 1:
                self.status_label.setText(
                    "Select one ImageTool source to replace one figure source."
                )
            elif selected_alias is None:
                self.status_label.setText("Select the figure source to replace.")
            else:
                self.status_label.setText("The selected source will be replaced.")
            return

        selection = self.axes_selection()
        self._ok_button().setEnabled(selection is not None)
        if selection is None:
            self.status_label.setText("Select at least one target axes.")
            return
        count = len(selection.axes_ids) if selection.axes_ids else len(selection.axes)
        suffix = "axis" if count == 1 else "axes"
        self.status_label.setText(f"{count} target {suffix} selected.")

    @QtCore.Slot()
    def _select_all_axes(self) -> None:
        tool = self._figure_tool()
        if tool is None:
            return
        setup = tool.tool_status.setup
        if setup.layout_mode == "gridspec":
            self.gridspec_axes_selector.set_selected_axes_ids(
                _gridspec_valid_axes_ids(setup, _gridspec_all_axes_ids(setup)),
                emit=True,
            )
        else:
            self.axes_selector.set_selected_axes(
                self._manager._figure_all_axes(setup.nrows, setup.ncols),
                emit=True,
            )

    @QtCore.Slot()
    def _clear_axes(self) -> None:
        if self.selector_stack.currentWidget() is self.gridspec_axes_selector:
            self.gridspec_axes_selector.set_selected_axes_ids((), emit=True)
        else:
            self.axes_selector.set_selected_axes((), emit=True)

    @QtCore.Slot()
    def _add_subplot_row(self) -> None:
        self._grow_subplot_grid("row")

    @QtCore.Slot()
    def _add_subplot_column(self) -> None:
        self._grow_subplot_grid("column")

    def _grow_subplot_grid(self, direction: typing.Literal["row", "column"]) -> None:
        tool = self._figure_tool()
        if tool is None or tool.tool_status.setup.layout_mode != "subplots":
            return
        selected = self.axes_selector.selected_axes()
        if not tool._grow_subplot_grid(direction):
            return
        setup = tool.tool_status.setup
        labels = {
            (row, col): f"{row}, {col}"
            for row in range(setup.nrows)
            for col in range(setup.ncols)
        }
        self.axes_selector.set_grid(setup.nrows, setup.ncols, labels)
        self.axes_selector.set_selected_axes(selected or ((0, 0),), emit=True)

    def _figure_tool(
        self,
    ) -> erlab.interactive._figurecomposer.FigureComposerTool | None:
        if not self._manager._is_figure_uid(self.figure_uid()):
            return None
        tool = self._manager._child_node(self.figure_uid()).tool_window
        return (
            tool
            if isinstance(tool, erlab.interactive._figurecomposer.FigureComposerTool)
            else None
        )

    def _ok_button(self) -> QtWidgets.QPushButton:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if button is None:
            raise RuntimeError("Append dialog OK button is unavailable")
        return button

    def _default_subplot_axes(
        self, axes: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        if self._operation_defaults_to_all_axes() or len(axes) <= 1:
            return axes
        return axes[:1]

    def _default_gridspec_axes(self, axes_ids: tuple[str, ...]) -> tuple[str, ...]:
        if self._operation_defaults_to_all_axes() or len(axes_ids) <= 1:
            return axes_ids
        return axes_ids[:1]

    def _operation_defaults_to_all_axes(self) -> bool:
        if self._operation is None:
            return True
        return (
            self._operation.kind
            == erlab.interactive._figurecomposer.FigureOperationKind.PLOT_SLICES
        )


class _FigureSourcePickerDialog(QtWidgets.QDialog):
    """Select live ImageTool manager rows to add as Figure Composer sources."""

    def __init__(
        self, manager: ImageToolManager, *, prechecked_uids: Iterable[str] = ()
    ) -> None:
        super().__init__(manager)
        self._manager = manager
        self._prechecked_uids = set(prechecked_uids)
        self._expanded_before_search: set[str] | None = None
        self.setObjectName("managerFigureSourcePickerDialog")
        self.setWindowTitle("Add Figure Sources")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setObjectName("managerFigureSourcePickerSearch")
        self.search_edit.setAccessibleName("Search ImageTools")
        self.search_edit.setPlaceholderText("Search ImageTools")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._filter_tree)
        layout.addWidget(self.search_edit)
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setObjectName("managerFigureSourcePickerTree")
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.tree.itemChanged.connect(self._item_changed)
        layout.addWidget(self.tree)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setObjectName("managerFigureSourcePickerStatus")
        layout.addWidget(self.status_label)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._populate_tree()
        self.tree.collapseAll()
        self._expand_prechecked_ancestors()
        self._refresh_ok_state()

    def _populate_tree(self) -> None:
        for index in self._manager._tool_graph.root_indices_for_workspace():
            wrapper = self._manager._tool_graph.root_wrappers.get(index)
            if wrapper is None:
                continue
            item = self._add_node_item(self.tree.invisibleRootItem(), wrapper)
            self._populate_child_items(item, wrapper)

    def _populate_child_items(
        self,
        parent_item: QtWidgets.QTreeWidgetItem,
        parent_node: _ImageToolWrapper | _ManagedWindowNode,
    ) -> None:
        for uid in parent_node._childtool_indices:
            node = self._manager._tool_graph.nodes.get(uid)
            if not isinstance(node, _ManagedWindowNode):
                continue
            if self._manager._is_figure_node(node):
                continue
            item = self._add_node_item(parent_item, node)
            self._populate_child_items(item, node)

    def _add_node_item(
        self,
        parent: QtWidgets.QTreeWidget | QtWidgets.QTreeWidgetItem | None,
        node: _ImageToolWrapper | _ManagedWindowNode,
    ) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem(parent, [node.display_text])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, node.uid)
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled
        if node.is_imagetool:
            flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
            item.setCheckState(
                0,
                QtCore.Qt.CheckState.Checked
                if node.uid in self._prechecked_uids
                else QtCore.Qt.CheckState.Unchecked,
            )
        else:
            item.setToolTip(0, "Only ImageTool rows can be added as figure sources.")
            item.setForeground(0, QtGui.QBrush(QtGui.QColor("gray")))
        item.setFlags(flags)
        return item

    def selected_targets(self) -> tuple[str, ...]:
        output: list[str] = []
        root = self.tree.invisibleRootItem()
        if root is None:
            return ()
        self._collect_checked_targets(root, output)
        return tuple(output)

    def _collect_checked_targets(
        self, item: QtWidgets.QTreeWidgetItem, output: list[str]
    ) -> None:
        uid = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(uid, str) and item.checkState(0) == QtCore.Qt.CheckState.Checked:
            node = self._manager._tool_graph.nodes.get(uid)
            if node is not None and node.is_imagetool and uid not in output:
                output.append(uid)
        for row in range(item.childCount()):
            child = item.child(row)
            if child is not None:
                self._collect_checked_targets(child, output)

    def _tree_items(self) -> Iterator[QtWidgets.QTreeWidgetItem]:
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while (item := iterator.value()) is not None:
            yield item
            iterator += 1

    def _expanded_uids(self) -> set[str]:
        return {
            uid
            for item in self._tree_items()
            if item.isExpanded()
            and isinstance(uid := item.data(0, QtCore.Qt.ItemDataRole.UserRole), str)
        }

    def _restore_expanded_uids(self, expanded_uids: set[str]) -> None:
        for item in self._tree_items():
            uid = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            item.setExpanded(isinstance(uid, str) and uid in expanded_uids)

    def _expand_prechecked_ancestors(self) -> None:
        for item in self._tree_items():
            if item.checkState(0) != QtCore.Qt.CheckState.Checked:
                continue
            parent = item.parent()
            while parent is not None:
                parent.setExpanded(True)
                parent = parent.parent()

    @QtCore.Slot(str)
    def _filter_tree(self, text: str) -> None:
        query = text.strip().casefold()
        root = self.tree.invisibleRootItem()
        if root is None:
            return
        if not query:
            for item in self._tree_items():
                item.setHidden(False)
            if self._expanded_before_search is not None:
                expanded = self._expanded_before_search
                self._expanded_before_search = None
                self.tree.collapseAll()
                self._restore_expanded_uids(expanded)
            return
        if self._expanded_before_search is None:
            self._expanded_before_search = self._expanded_uids()
        for row in range(root.childCount()):
            child = root.child(row)
            if child is not None:
                self._filter_tree_item(child, query)

    def _filter_tree_item(self, item: QtWidgets.QTreeWidgetItem, query: str) -> bool:
        child_match = False
        for row in range(item.childCount()):
            child = item.child(row)
            if child is not None:
                child_match |= self._filter_tree_item(child, query)
        matches = query in item.text(0).casefold()
        visible = matches or child_match
        item.setHidden(not visible)
        item.setExpanded(child_match)
        return visible

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _item_changed(self, _item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        self._refresh_ok_state()

    def _refresh_ok_state(self) -> None:
        selected = self.selected_targets()
        ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(bool(selected))
        count = len(selected)
        if count:
            suffix = "source" if count == 1 else "sources"
            self.status_label.setText(f"{count} ImageTool {suffix} selected.")
        else:
            self.status_label.setText("Select at least one ImageTool source.")
