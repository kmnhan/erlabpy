"""Dialogs for the ImageToolManager."""

from __future__ import annotations

import typing
import weakref

from qtpy import QtCore, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    import xarray

    from erlab.interactive.imagetool.manager import ImageToolManager


class _RenameDialog(QtWidgets.QDialog):
    def __init__(self, manager: ImageToolManager, original_names: list[str]) -> None:
        super().__init__(manager)
        self.setWindowTitle("Rename selected tools")
        self._manager = weakref.ref(manager)

        self._layout = QtWidgets.QGridLayout()
        self.setLayout(self._layout)

        self._new_name_lines: list[QtWidgets.QLineEdit] = []

        for i, name in enumerate(original_names):
            line_new = QtWidgets.QLineEdit(name)
            line_new.setPlaceholderText("New name")
            self._layout.addWidget(QtWidgets.QLabel(name), i, 0)
            self._layout.addWidget(QtWidgets.QLabel("→"), i, 1)
            self._layout.addWidget(line_new, i, 2)
            self._new_name_lines.append(line_new)

        fm = self._new_name_lines[0].fontMetrics()
        max_width = max(
            fm.boundingRect(line.text()).width() for line in self._new_name_lines
        )
        for line in self._new_name_lines:
            line.setMinimumWidth(max_width + 10)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self._layout.addWidget(button_box, len(original_names), 0, 1, 3)

    def new_names(self) -> list[str]:
        return [w.text() for w in self._new_name_lines]

    def accept(self) -> None:
        manager = self._manager()
        if manager is not None:
            for index, new_name in zip(
                manager.tree_view.selected_imagetool_indices,
                self.new_names(),
                strict=True,
            ):
                manager.rename_imagetool(index, new_name)
        super().accept()


class _StoreDialog(QtWidgets.QDialog):
    def __init__(self, manager: ImageToolManager, target_indices: list[int]) -> None:
        super().__init__(manager)
        self.setWindowTitle("Store with IPython")
        self._manager = weakref.ref(manager)
        self._target_indices: list[int] = target_indices

        self._layout = QtWidgets.QFormLayout()
        self.setLayout(self._layout)

        self._var_name_lines: list[QtWidgets.QLineEdit] = []

        self._layout.addRow("Data to store", QtWidgets.QLabel("Stored name"))

        for tool_idx in target_indices:
            data = manager.get_imagetool(tool_idx).slicer_area._data
            wrapper = manager._imagetool_wrappers[tool_idx]
            default_name = data.name
            if not (isinstance(default_name, str) and default_name.isidentifier()):
                if wrapper.name.isidentifier():
                    default_name = wrapper.name
                else:
                    default_name = f"data_{tool_idx}"

            line_new = QtWidgets.QLineEdit(default_name)
            line_new.setPlaceholderText("Enter variable name")
            line_new.setValidator(erlab.interactive.utils.IdentifierValidator())
            self._layout.addRow(wrapper.label_text, line_new)
            self._var_name_lines.append(line_new)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self._layout.addRow(button_box)

    def var_name_map(self) -> dict[int, str]:
        return {
            idx: w.text()
            for idx, w in zip(self._target_indices, self._var_name_lines, strict=True)
        }

    def accept(self) -> None:
        manager = self._manager()
        if manager is not None:
            for idx, var_name in self.var_name_map().items():
                manager.console._console_widget.store_data_as(idx, var_name)
        super().accept()


class _NameFilterDialog(QtWidgets.QDialog):
    def __init__(self, parent: ImageToolManager, valid_name_filters: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Loader")

        self._valid_name_filters = valid_name_filters

        layout = QtWidgets.QVBoxLayout(self)
        self._button_group = QtWidgets.QButtonGroup(self)

        for i, name in enumerate(valid_name_filters):
            radio_button = QtWidgets.QRadioButton(name)
            self._button_group.addButton(radio_button, i)
            layout.addWidget(radio_button)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def check_filter(self, name_filter: str | None) -> None:
        self._button_group.buttons()[
            self._valid_name_filters.index(name_filter)
            if name_filter in self._valid_name_filters
            else 0
        ].setChecked(True)

    def checked_filter(self) -> str:
        return self._valid_name_filters[self._button_group.checkedId()]


class _ChooseFromDataTreeDialog(QtWidgets.QDialog):
    def __init__(
        self,
        manager: ImageToolManager,
        tree: xarray.DataTree,
        mode: typing.Literal["save", "load"],
    ) -> None:
        super().__init__(manager)
        self._manager = weakref.ref(manager)

        self._saving: bool = mode == "save"

        if self._saving:
            self.setWindowTitle("Select tools to save")
        else:
            self.setWindowTitle("Select tools to add")

        layout = QtWidgets.QHBoxLayout(self)

        self._tree_widget = QtWidgets.QTreeWidget(self)
        self._tree_widget.setColumnCount(1)
        self._tree_widget.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._tree_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self._tree_widget.setUniformRowHeights(True)
        self._tree_widget.setAlternatingRowColors(True)
        self._tree_widget.setWordWrap(False)
        self._tree_widget.setHeaderHidden(True)
        self._tree_widget.setAnimated(True)
        self._tree_widget.itemChanged.connect(self._on_item_changed)
        self._tree_widget.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._tree_widget.customContextMenuRequested.connect(self._show_tree_menu)
        self._tree_menu = QtWidgets.QMenu(self._tree_widget)
        self._tree_menu.addAction("Expand All", self._tree_widget.expandAll)
        self._tree_menu.addAction("Collapse All", self._tree_widget.collapseAll)
        self._tree_menu.addAction("Select All", self._check_all)
        self._tree_menu.addAction("Deselect All", self._uncheck_all)
        self._tree_menu.addAction("ImageTools Only", self._uncheck_children)

        layout.addWidget(self._tree_widget)

        self._populate_tree(tree)

        button_box = QtWidgets.QDialogButtonBox(QtCore.Qt.Orientation.Vertical)
        btn_selectall = typing.cast(
            "QtWidgets.QPushButton",
            button_box.addButton(
                "Select All", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        btn_deselectall = typing.cast(
            "QtWidgets.QPushButton",
            button_box.addButton(
                "Deselect All", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        btn_itools_only = typing.cast(
            "QtWidgets.QPushButton",
            button_box.addButton(
                "ImageTools Only", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            ),
        )
        btn_ok = typing.cast(
            "QtWidgets.QPushButton",
            button_box.addButton(
                QtWidgets.QDialogButtonBox.StandardButton.Ok,
            ),
        )
        btn_cancel = typing.cast(
            "QtWidgets.QPushButton",
            button_box.addButton(
                QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            ),
        )

        btn_selectall.clicked.connect(self._check_all)
        btn_deselectall.clicked.connect(self._uncheck_all)
        btn_itools_only.clicked.connect(self._uncheck_children)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(button_box)

    @QtCore.Slot(QtCore.QPoint)
    def _show_tree_menu(self, pos: QtCore.QPoint) -> None:
        self._tree_menu.popup(self.mapToGlobal(pos))

    @QtCore.Slot()
    def _check_all(self) -> None:
        self._set_checked_all(QtCore.Qt.CheckState.Checked)

    @QtCore.Slot()
    def _uncheck_all(self) -> None:
        self._set_checked_all(QtCore.Qt.CheckState.Unchecked)

    @QtCore.Slot()
    def _uncheck_children(self) -> None:
        self._set_checked_all(QtCore.Qt.CheckState.Unchecked, only_children=True)

    def _set_checked_all(
        self, state: QtCore.Qt.CheckState, only_children: bool = False
    ) -> None:
        root: QtWidgets.QTreeWidgetItem | None = self._tree_widget.invisibleRootItem()
        if root is not None:  # pragma: no branch
            for i in range(root.childCount()):
                item = root.child(i)
                if item is not None:  # pragma: no branch
                    if not only_children:
                        item.setCheckState(0, state)
                    for j in range(item.childCount()):
                        child_item = item.child(j)
                        if child_item is not None:  # pragma: no branch
                            child_item.setCheckState(0, state)

    def _populate_tree(self, tree: xarray.DataTree) -> None:
        root: QtWidgets.QTreeWidgetItem | None = self._tree_widget.invisibleRootItem()
        manager = self._manager()

        if root is not None and manager is not None:  # pragma: no branch
            start = int(manager.next_idx)
            for i, (key, node) in enumerate(tree.items()):
                title = node["imagetool"].attrs["itool_title"]

                # Use candidate if loading, current index if saving
                name = str(key) if self._saving else str(start + i)
                if title:
                    name = f"{name}: {title}"

                item = QtWidgets.QTreeWidgetItem(root, [name])
                item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                )

                if "childtools" in node:
                    for cnode in typing.cast(
                        "xarray.DataTree", node["childtools"]
                    ).values():
                        citem = QtWidgets.QTreeWidgetItem(
                            item, [cnode.attrs["tool_title"]]
                        )
                        citem.setFlags(
                            QtCore.Qt.ItemFlag.ItemIsEnabled
                            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                        )
                        citem.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        item.addChild(citem)
                item.setCheckState(0, QtCore.Qt.CheckState.Checked)
                self._tree_widget.addTopLevelItem(item)
            self._tree_widget.expandAll()

    def imagetool_selected(self, index: int) -> bool:
        """Return whether the ImageTool at the given index is selected."""
        out = False
        item = self._tree_widget.topLevelItem(index)
        if item is not None:  # pragma: no branch
            out = item.checkState(0) != QtCore.Qt.CheckState.Unchecked
        return out

    def childtool_selected(self, parent_index: int, child_index: int) -> bool:
        """Return whether the child tool at the given indices is selected."""
        out = False
        parent_item = self._tree_widget.topLevelItem(parent_index)
        if parent_item is not None:  # pragma: no branch
            child_item = parent_item.child(child_index)
            if child_item is not None:  # pragma: no branch
                out = child_item.checkState(0) == QtCore.Qt.CheckState.Checked
        return out

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if column == 0:  # pragma: no branch
            if item.parent() is None:
                # Parent item changed
                check_state = item.checkState(0)
                # Checked or Unchecked since PartiallyChecked cannot be set by user

                # Make children check state match parent
                for i in range(item.childCount()):
                    child = item.child(i)
                    if child is not None:  # pragma: no branch
                        self._tree_widget.blockSignals(True)
                        child.setCheckState(0, check_state)
                        self._tree_widget.blockSignals(False)

                if check_state == QtCore.Qt.CheckState.Checked:
                    n_children = item.childCount()
                    if n_children == 0:
                        return

                    n_checked_children: int = sum(
                        typing.cast(
                            "QtWidgets.QTreeWidgetItem", item.child(i)
                        ).checkState(0)
                        == QtCore.Qt.CheckState.Checked
                        for i in range(n_children)
                    )
                    if n_checked_children < n_children:
                        # Some children are unchecked, set to partially checked
                        self._tree_widget.blockSignals(True)
                        item.setCheckState(0, QtCore.Qt.CheckState.PartiallyChecked)
                        self._tree_widget.blockSignals(False)
            else:
                # Child item changed
                parent = item.parent()
                if parent is not None:  # pragma: no branch
                    n_children = parent.childCount()
                    n_checked_children = sum(
                        typing.cast(
                            "QtWidgets.QTreeWidgetItem", parent.child(i)
                        ).checkState(0)
                        == QtCore.Qt.CheckState.Checked
                        for i in range(n_children)
                    )
                    if n_checked_children < n_children:
                        # Partial
                        self._tree_widget.blockSignals(True)
                        parent.setCheckState(0, QtCore.Qt.CheckState.PartiallyChecked)
                        self._tree_widget.blockSignals(False)

                    elif n_checked_children == n_children:
                        # All children checked, parent must be checked
                        self._tree_widget.blockSignals(True)
                        parent.setCheckState(0, QtCore.Qt.CheckState.Checked)
                        self._tree_widget.blockSignals(False)
