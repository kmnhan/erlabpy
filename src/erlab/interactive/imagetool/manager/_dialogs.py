"""Dialogs for the ImageToolManager."""

from __future__ import annotations

import ast
import html
import inspect
import pathlib
import typing
import weakref

import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.dialogs as imagetool_dialogs

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    import xarray

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager


_BatchDialogType = type[imagetool_dialogs._DataManipulationDialog]

_BATCH_OPERATION_CATEGORIES: tuple[tuple[str, tuple[_BatchDialogType, ...]], ...] = (
    (
        "Coordinates",
        (
            imagetool_dialogs.AssignCoordsDialog,
            imagetool_dialogs.AssignAttrsDialog,
            imagetool_dialogs.RenameDimsCoordsDialog,
            imagetool_dialogs.SwapDimsDialog,
        ),
    ),
    (
        "Selection",
        (
            imagetool_dialogs.SelectionDialog,
            imagetool_dialogs.CropDialog,
            imagetool_dialogs.CropToViewDialog,
        ),
    ),
    (
        "Transforms",
        (
            imagetool_dialogs.KspaceConversionDialog,
            imagetool_dialogs.RotationDialog,
            imagetool_dialogs.AggregateDialog,
            imagetool_dialogs.InterpolationDialog,
            imagetool_dialogs.SortByDialog,
            imagetool_dialogs.LeadingEdgeDialog,
            imagetool_dialogs.DivideByCoordDialog,
            imagetool_dialogs.CoarsenDialog,
            imagetool_dialogs.ThinDialog,
            imagetool_dialogs.SymmetrizeDialog,
            imagetool_dialogs.SymmetrizeNfoldDialog,
        ),
    ),
    (
        "Filters",
        (
            imagetool_dialogs.NormalizeDialog,
            imagetool_dialogs.GaussianFilterDialog,
        ),
    ),
)


def _dialog_batch_available(dialog_cls: _BatchDialogType) -> bool:
    return bool(dialog_cls.operation_types) and all(
        operation_type.batch_available for operation_type in dialog_cls.operation_types
    )


def _batch_operation_dialog_classes() -> tuple[_BatchDialogType, ...]:
    return tuple(
        dialog_cls
        for _category, dialog_classes in _BATCH_OPERATION_CATEGORIES
        for dialog_cls in dialog_classes
        if _dialog_batch_available(dialog_cls)
    )


class _BatchOperationDialog(QtWidgets.QDialog):
    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self.setWindowTitle("Batch Operation")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._manager = weakref.ref(manager)
        self._syncing_targets = False
        self._target_items: dict[int | str, QtWidgets.QTreeWidgetItem] = {}

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        panes = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        layout.addWidget(panes)

        target_group = QtWidgets.QGroupBox("Targets", self)
        target_layout = QtWidgets.QVBoxLayout(target_group)
        self._target_tree = QtWidgets.QTreeWidget(target_group)
        self._target_tree.setObjectName("manager_batch_target_tree")
        self._target_tree.setHeaderHidden(True)
        self._target_tree.itemChanged.connect(self._target_item_changed)
        target_layout.addWidget(self._target_tree)
        panes.addWidget(target_group)

        operation_group = QtWidgets.QGroupBox("Operations", self)
        operation_layout = QtWidgets.QVBoxLayout(operation_group)
        self._operation_tree = QtWidgets.QTreeWidget(operation_group)
        self._operation_tree.setObjectName("manager_batch_operation_tree")
        self._operation_tree.setHeaderHidden(True)
        self._operation_tree.itemSelectionChanged.connect(self._update_launch_enabled)
        operation_layout.addWidget(self._operation_tree)
        panes.addWidget(operation_group)

        panes.setSizes([220, 280])

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._launch_button = self._button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        if self._launch_button is not None:  # pragma: no branch
            self._launch_button.setObjectName("manager_batch_launch_button")
            self._launch_button.setText("Open")
        self._button_box.accepted.connect(self._launch_selected_operation)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

        selection_model = manager.tree_view.selectionModel()
        if selection_model is not None:  # pragma: no branch
            selection_model.selectionChanged.connect(self._sync_target_checks)
            selection_model.selectionChanged.connect(self._update_launch_enabled)

        self._populate_operations()
        self._populate_targets()
        self._update_launch_enabled()

    def _manager_instance(self) -> ImageToolManager | None:
        return self._manager()

    def _populate_operations(self) -> None:
        self._operation_tree.clear()
        for category, dialog_classes in _BATCH_OPERATION_CATEGORIES:
            visible_dialogs = [
                dialog_cls
                for dialog_cls in dialog_classes
                if _dialog_batch_available(dialog_cls)
            ]
            if not visible_dialogs:
                continue
            category_item = QtWidgets.QTreeWidgetItem([category])
            category_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self._operation_tree.addTopLevelItem(category_item)
            for dialog_cls in visible_dialogs:
                item = QtWidgets.QTreeWidgetItem(
                    [dialog_cls.title or dialog_cls.__name__]
                )
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, dialog_cls)
                category_item.addChild(item)
        self._operation_tree.expandAll()

    def _populate_targets(self) -> None:
        manager = self._manager_instance()
        if manager is None:
            return
        self._syncing_targets = True
        try:
            self._target_tree.clear()
            self._target_items.clear()
            for target in manager._tool_graph.displayed_indices:
                self._add_target_item(target)
            self._target_tree.expandAll()
        finally:
            self._syncing_targets = False
        self._sync_target_checks()

    def _add_target_item(
        self,
        target: int | str,
        parent: QtWidgets.QTreeWidgetItem | None = None,
    ) -> None:
        manager = self._manager_instance()
        if manager is None:
            return
        node = manager._node_for_target(target)
        item: QtWidgets.QTreeWidgetItem | None = None
        if node.is_imagetool:
            item = QtWidgets.QTreeWidgetItem([node.display_text])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, target)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            self._target_items[target] = item
            if parent is None:
                self._target_tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
        child_parent = item if item is not None else parent
        for child_uid in node._childtool_indices:
            self._add_target_item(child_uid, child_parent)

    @QtCore.Slot()
    def _sync_target_checks(self) -> None:
        manager = self._manager_instance()
        if manager is None:
            return
        selected = set(manager._selected_imagetool_targets())
        self._syncing_targets = True
        try:
            for target, item in self._target_items.items():
                checked = target in selected
                item.setCheckState(
                    0,
                    QtCore.Qt.CheckState.Checked
                    if checked
                    else QtCore.Qt.CheckState.Unchecked,
                )
        finally:
            self._syncing_targets = False
        self._update_launch_enabled()

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _target_item_changed(
        self, item: QtWidgets.QTreeWidgetItem, column: int
    ) -> None:
        if self._syncing_targets or column != 0:
            return
        manager = self._manager_instance()
        if manager is None:
            return
        target = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(target, (int, str)):
            return
        qmodelindex = manager.tree_view._model._row_index(target)
        if not qmodelindex.isValid():
            return
        selection_model = manager.tree_view.selectionModel()
        if selection_model is None:
            return
        flag = (
            QtCore.QItemSelectionModel.SelectionFlag.Select
            if item.checkState(0) == QtCore.Qt.CheckState.Checked
            else QtCore.QItemSelectionModel.SelectionFlag.Deselect
        )
        selection_model.select(QtCore.QItemSelection(qmodelindex, qmodelindex), flag)
        self._update_launch_enabled()

    def _selected_dialog_cls(self) -> _BatchDialogType | None:
        items = self._operation_tree.selectedItems()
        if not items:
            return None
        dialog_cls = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(dialog_cls, type) and issubclass(
            dialog_cls, imagetool_dialogs._DataManipulationDialog
        ):
            return dialog_cls
        return None

    @QtCore.Slot()
    def _update_launch_enabled(self) -> None:
        if self._launch_button is None:
            return
        manager = self._manager_instance()
        selected_targets = (
            [] if manager is None else manager._selected_imagetool_targets()
        )
        self._launch_button.setEnabled(
            self._selected_dialog_cls() is not None and len(selected_targets) >= 2
        )

    @QtCore.Slot()
    def _launch_selected_operation(self) -> None:
        manager = self._manager_instance()
        dialog_cls = self._selected_dialog_cls()
        if manager is None or dialog_cls is None:
            return
        targets = manager._selected_imagetool_targets()
        if len(targets) < 2:
            erlab.interactive.utils.MessageDialog.critical(
                self,
                "Error",
                "Select at least two ImageTool targets.",
            )
            return
        anchor_tool = manager.get_imagetool(targets[0])
        dialog = dialog_cls(anchor_tool.slicer_area, batch_manager=manager)
        dialog.setModal(True)
        anchor_tool.slicer_area.add_tool_window(
            dialog,
            update_title=False,
            transfer_to_manager=False,
        )
        super().accept()

    def show(self) -> None:
        self._populate_targets()
        super().show()

    def open(self) -> None:
        self._populate_targets()
        super().open()


class _ConcatDialog(QtWidgets.QDialog):
    _RESULT_NEW = "new"
    _RESULT_REPLACE = "replace"
    _SOURCES_KEEP = "keep"
    _SOURCES_REMOVE = "remove"

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self.setWindowTitle("Concatenate Selected Tools")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._manager = weakref.ref(manager)
        self._selected_targets: list[int | str] = []

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        # Initialize options group and layout
        option_group = QtWidgets.QGroupBox("Options")
        option_layout = QtWidgets.QFormLayout()
        option_group.setLayout(option_layout)

        self._dim_line = QtWidgets.QLineEdit()
        self._dim_line.setText("concat_dim")
        option_layout.addRow("dim", self._dim_line)

        self._coords_combo = QtWidgets.QComboBox()
        self._coords_combo.addItems(["minimal", "different", "all"])
        self._coords_combo.setCurrentText("minimal")
        option_layout.addRow("coords", self._coords_combo)

        self._compat_combo = QtWidgets.QComboBox()
        self._compat_combo.addItems(
            ["identical", "equals", "broadcast_equals", "no_conflicts", "override"]
        )
        self._compat_combo.setCurrentText("override")
        option_layout.addRow("compat", self._compat_combo)

        self._join_combo = QtWidgets.QComboBox()
        self._join_combo.addItems(
            ["outer", "inner", "left", "right", "exact", "override"]
        )
        self._join_combo.setCurrentText("outer")
        option_layout.addRow("join", self._join_combo)

        self._combine_attrs_combo = QtWidgets.QComboBox()
        self._combine_attrs_combo.addItems(
            ["drop", "identical", "no_conflicts", "drop_conflicts", "override"]
        )
        self._combine_attrs_combo.setCurrentText("override")
        option_layout.addRow("combine_attrs", self._combine_attrs_combo)

        link = "https://docs.xarray.dev/en/stable/generated/xarray.concat.html"
        docs_label = QtWidgets.QLabel(
            f"<a href='{link}'>xarray.concat documentation</a>"
        )
        docs_label.setOpenExternalLinks(True)
        option_layout.addRow(docs_label)

        self._result_combo = QtWidgets.QComboBox()
        self._result_combo.addItem("Open as New ImageTool", self._RESULT_NEW)
        self._result_combo.addItem("Replace Source Tool", self._RESULT_REPLACE)
        self._result_combo.currentIndexChanged.connect(self._update_result_controls)
        option_layout.addRow("Result", self._result_combo)

        self._replace_target_combo = QtWidgets.QComboBox()
        option_layout.addRow("Replace", self._replace_target_combo)

        self._sources_combo = QtWidgets.QComboBox()
        self._sources_combo.addItem("Keep Source Tools", self._SOURCES_KEEP)
        self._sources_combo.addItem("Remove Source Tools", self._SOURCES_REMOVE)
        option_layout.addRow("Sources", self._sources_combo)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Populate layout
        layout.addWidget(option_group)
        layout.addWidget(button_box)

    def open(self) -> None:
        self._result_combo.setCurrentIndex(
            self._result_combo.findData(self._RESULT_NEW)
        )
        self._sources_combo.setCurrentIndex(
            self._sources_combo.findData(self._SOURCES_KEEP)
        )
        self._refresh_selected_targets()
        super().open()

    def _refresh_selected_targets(self) -> None:
        manager = self._manager()
        self._selected_targets = (
            [] if manager is None else list(manager._selected_imagetool_targets())
        )
        current_target = self._replace_target_combo.currentData()
        self._replace_target_combo.clear()
        if manager is not None:
            for target in self._selected_targets:
                node = manager._node_for_target(target)
                self._replace_target_combo.addItem(node.display_text, target)

        current_index = self._replace_target_combo.findData(current_target)
        if current_index >= 0:
            self._replace_target_combo.setCurrentIndex(current_index)
        self._update_result_controls()

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _update_result_controls(self, *_args: object) -> None:
        replace_selected = self.result_mode() == self._RESULT_REPLACE
        self._replace_target_combo.setEnabled(replace_selected)

    def result_mode(self) -> str:
        value = self._result_combo.currentData()
        return self._RESULT_NEW if value is None else str(value)

    def sources_mode(self) -> str:
        value = self._sources_combo.currentData()
        return self._SOURCES_KEEP if value is None else str(value)

    def replacement_target(self) -> int | str | None:
        if self.result_mode() != self._RESULT_REPLACE:
            return None
        value = self._replace_target_combo.currentData()
        return typing.cast("int | str | None", value)

    def concat_kwargs(self) -> dict[str, typing.Any]:
        return {
            "dim": self._dim_line.text(),
            "coords": self._coords_combo.currentText(),
            "compat": self._compat_combo.currentText(),
            "join": self._join_combo.currentText(),
            "combine_attrs": self._combine_attrs_combo.currentText(),
        }

    def _operation_code(
        self, manager: ImageToolManager, selected: list[int | str]
    ) -> str:
        concat_kwargs = self.concat_kwargs()
        input_names = [
            manager._script_input_name_for_node(manager._node_for_target(target))
            for target in selected
        ]
        return (
            f"derived = xr.concat([{', '.join(input_names)}], "
            + ", ".join(f"{key}={value!r}" for key, value in concat_kwargs.items())
            + ")"
        )

    def _remove_source_targets(
        self,
        manager: ImageToolManager,
        selected: list[int | str],
        *,
        preserved_target: int | str | None = None,
    ) -> None:
        if self.sources_mode() != self._SOURCES_REMOVE:
            return

        preserved_uid = (
            None
            if preserved_target is None
            else manager._node_for_target(preserved_target).uid
        )
        preserved_subtree_uids = (
            set()
            if preserved_uid is None
            else set(manager._tool_graph.subtree_uids(preserved_uid))
        )
        to_remove: list[int | str] = []
        for target in selected:
            node = manager._node_for_target(target)
            if node.uid in preserved_subtree_uids:
                continue
            if (
                preserved_uid is not None
                and preserved_uid in manager._tool_graph.subtree_uids(node.uid)
            ):
                continue
            to_remove.append(target)
        manager._remove_imagetools(to_remove)

    def accept(self) -> None:
        manager = self._manager()
        if manager is not None:  # pragma: no branch
            try:
                selected = list(self._selected_targets)
                to_concat = [
                    manager.get_imagetool(idx)
                    .slicer_area.persistence_data_and_state()[0]
                    .copy(deep=False)
                    for idx in selected
                ]
                concat_kwargs = self.concat_kwargs()
                operation_code = self._operation_code(manager, selected)
                result_data = xr.concat(to_concat, **concat_kwargs)
                replacement_target = self.replacement_target()
                if replacement_target is None:
                    created_index = manager._show_multi_input_script_result(
                        result_data,
                        selected,
                        operation_label="Concatenate selected ImageTools",
                        operation_code=operation_code,
                        use_displayed_provenance=False,
                    )
                else:
                    replacement_node = manager._node_for_target(replacement_target)
                    replacement_node.replace_with_detached_data(
                        result_data,
                        manager._multi_input_script_provenance(
                            selected,
                            operation_label="Concatenate selected ImageTools",
                            operation_code=operation_code,
                            detached_input_uid=replacement_node.uid,
                            use_displayed_provenance=False,
                        ),
                        propagate_descendants=True,
                        preserve_filter=False,
                    )
                    manager.tree_view.refresh(replacement_node.uid)
                    if manager._metadata_node_uid == replacement_node.uid:
                        manager._set_metadata_node(replacement_node)
                    manager._sigDataReplaced.emit()
                    created_index = replacement_target
            except Exception:
                erlab.interactive.utils.MessageDialog.critical(
                    self,
                    "Error",
                    "An error occurred while concatenating data.",
                )
            else:
                if created_index is None:
                    erlab.interactive.utils.MessageDialog.critical(
                        self,
                        "Error",
                        "An error occurred while concatenating data.",
                    )
                else:
                    self._remove_source_targets(
                        manager,
                        selected,
                        preserved_target=replacement_target,
                    )
        super().accept()


class _RenameDialog(QtWidgets.QDialog):
    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self.setWindowTitle("Rename Selected Tools")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._manager = weakref.ref(manager)

        self._layout = QtWidgets.QGridLayout(self)

        self._new_name_lines: dict[int, QtWidgets.QLineEdit] = {}

        # Persistent button box; we re-add it when (re)building rows
        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

    def set_names(self, indices: list[int], original_names: list[str]) -> None:
        # Clear current rows but keep the button box object
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item:  # pragma: no branch
                widget = item.widget()
                if widget is not None and widget is not self._button_box:
                    widget.deleteLater()

        for k, v in self._new_name_lines.copy().items():
            self._new_name_lines.pop(k)
            v.deleteLater()

        # Rebuild rows
        for i, (tool_idx, name) in enumerate(zip(indices, original_names, strict=True)):
            lbl_from = QtWidgets.QLabel(f"{tool_idx}: {name}")
            lbl_arrow = QtWidgets.QLabel("→")
            line_new = QtWidgets.QLineEdit(name)
            line_new.setPlaceholderText("New name")

            self._layout.addWidget(lbl_from, i, 0)
            self._layout.addWidget(lbl_arrow, i, 1)
            self._layout.addWidget(line_new, i, 2)
            self._new_name_lines[tool_idx] = line_new

        # Resize inputs to longest current text
        if self._new_name_lines:  # pragma: no branch
            fm = next(iter(self._new_name_lines.values())).fontMetrics()
            max_width = max(
                fm.boundingRect(line.text()).width()
                for line in self._new_name_lines.values()
            )
            for line in self._new_name_lines.values():
                line.setMinimumWidth(max_width + 10)

        # Re-add button box at the last row
        self._layout.addWidget(self._button_box, len(original_names), 0, 1, 3)

    def new_names(self) -> list[tuple[int, str]]:
        return [(idx, w.text()) for idx, w in self._new_name_lines.items()]

    def accept(self) -> None:
        manager = self._manager()
        if manager is not None:  # pragma: no branch
            for index, new_name in self.new_names():
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
            data = manager.get_imagetool(tool_idx).slicer_area.displayed_data
            wrapper = manager._tool_graph.root_wrappers[tool_idx]
            default_name = data.name
            if not erlab.utils.misc._is_valid_identifier(default_name):
                if erlab.utils.misc._is_valid_identifier(wrapper.name):
                    default_name = wrapper.name
                else:
                    default_name = f"data_{tool_idx}"

            line_new = QtWidgets.QLineEdit(str(default_name))
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


def _kwargs_to_text(kwargs: dict) -> str:
    """Convert a kwargs dict to a Python-like argument list."""
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


def _parse_value(node: ast.expr) -> typing.Any:
    """Parse a node into a Python literal if possible.

    Handles special case for dict(...) calls.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
    ):
        result = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("Cannot evaluate dict with **kwargs")
            result[kw.arg] = ast.literal_eval(kw.value)
        return result
    return ast.literal_eval(node)


def _text_to_kwargs(text: str) -> dict[str, typing.Any]:
    """Parse ``a=1, b='x'`` style text into a kwargs dict."""
    text = text.strip()
    if not text:
        return {}

    # Wrap in fake function call so Python can parse it as a Call node
    expr = ast.parse(f"f({text})", mode="eval")

    if not isinstance(expr.body, ast.Call):
        raise TypeError("Input must be a comma-separated list of keyword arguments")

    call = expr.body
    if call.args:
        raise ValueError("Only keyword arguments are supported")

    result: dict = {}
    for kw in call.keywords:
        try:
            value = _parse_value(kw.value)
        except ValueError as e:
            raise ValueError(f"Value for {kw.arg!r} is not a valid literal") from e
        result[kw.arg] = value

    return result


def _is_loader_func(func: Callable) -> bool:
    return isinstance(getattr(func, "__self__", None), erlab.io.dataloader.LoaderBase)


def _text_to_loader_extension_value(
    key: str, text: str
) -> dict[str, typing.Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Value for {key!r} is not a valid literal") from e
    return {key: value}


def _string_tuple_from_literal(key: str, text: str) -> tuple[str, ...]:
    parsed = _text_to_loader_extension_value(key, text)
    if parsed is None:
        return ()
    value = parsed[key]
    if isinstance(value, str):
        raise TypeError(f"{key} must be an iterable, not a string")
    try:
        return tuple(str(v) for v in value)
    except TypeError as e:
        raise TypeError(f"{key} must be an iterable") from e


def _name_map_from_literal(text: str) -> dict[str, str | Iterable[str]]:
    parsed = _text_to_loader_extension_value("name_map", text)
    if parsed is None:
        return {}
    value = parsed["name_map"]
    if not isinstance(value, dict):
        raise TypeError("name_map must be a dict")
    return value


def _iter_name_map_pairs(
    name_map: dict[str, str | Iterable[str]],
) -> Iterator[tuple[str, str]]:
    for new_name, originals in name_map.items():
        if isinstance(originals, str):
            yield str(new_name), originals
        else:
            for original in originals:
                yield str(new_name), str(original)


def _name_map_from_pairs(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, str | Iterable[str]]:
    grouped: dict[str, list[str]] = {}
    for new_name, original in pairs:
        values = grouped.setdefault(new_name, [])
        if original not in values:
            values.append(original)
    out: dict[str, str | Iterable[str]] = {}
    for new_name, originals in grouped.items():
        out[new_name] = originals[0] if len(originals) == 1 else originals
    return out


def _name_map_literal(name_map: dict[str, str | Iterable[str]]) -> str:
    return repr(name_map) if name_map else ""


def _coordinate_attrs_literal(values: tuple[str, ...]) -> str:
    return repr(list(values)) if values else ""


def _coordinate_attrs_sample_attrs(
    loader: erlab.io.dataloader.LoaderBase, sample_path: pathlib.Path
) -> dict[str, typing.Any]:
    load_kwargs: dict[str, typing.Any] = {}
    load_single = getattr(loader, "_original_load_single", loader.load_single)
    if erlab.utils.misc.accepts_kwarg(load_single, "without_values", strict=False):
        load_kwargs["without_values"] = True

    try:
        data = loader.load_single(sample_path, **load_kwargs)
    except TypeError:
        if not load_kwargs:
            raise
        data = loader.load_single(sample_path)

    attrs = getattr(data, "attrs", None)
    if attrs is None:
        return {}
    return dict(attrs)


def _tooltip_with_example(text: str, example: str) -> str:
    return f"<qt>{html.escape(text)}<br>Example: <tt>{html.escape(example)}</tt></qt>"


_LOADER_EXTENSION_TOGGLE_TOOLTIP = _tooltip_with_example(
    "Show optional literal erlab.io.extend_loader settings for this load. "
    "Values are saved with reload metadata.",
    "loader_extensions={'coordinate_attrs': ['polar']}",
)

_LOADER_EXTENSION_TOOLTIPS = {
    "name_map": _tooltip_with_example(
        "Literal dict mapping new coordinate/attribute names to original names.",
        "{'theta': 'Theta'}",
    ),
    "coordinate_attrs": _tooltip_with_example(
        "Literal list or tuple of attributes to promote to coordinates.",
        "['polar', 'tilt']",
    ),
    "average_attrs": _tooltip_with_example(
        "Literal list or tuple of attributes or coordinates to average when combining "
        "files.",
        "['temperature']",
    ),
    "additional_attrs": _tooltip_with_example(
        "Literal dict of extra attributes to add after loading.",
        "{'sample': 'A'}",
    ),
    "overridden_attrs": _tooltip_with_example(
        "Literal list or tuple of additional_attrs keys that may replace existing "
        "attributes.",
        "['sample']",
    ),
    "additional_coords": _tooltip_with_example(
        "Literal dict of extra coordinates to add after loading.",
        "{'scan': 1}",
    ),
    "overridden_coords": _tooltip_with_example(
        "Literal list or tuple of additional_coords keys that may replace existing "
        "coordinates.",
        "['scan']",
    ),
}


class _NameMapEditorDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        loader: erlab.io.dataloader.LoaderBase,
        sample_path: pathlib.Path | None,
        current_name_map: dict[str, str | Iterable[str]],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Name Map")
        self._selected_name_map: dict[str, str | Iterable[str]] = current_name_map
        self._rows: list[tuple[str, QtWidgets.QTableWidgetItem, bool]] = []
        self._unmatched_pairs: tuple[tuple[str, str], ...] = ()

        layout = QtWidgets.QVBoxLayout(self)

        if sample_path is None:
            layout.addWidget(QtWidgets.QLabel("No sample file is available."))
            self._add_button_box(layout, editor_enabled=False)
            return

        try:
            attrs = _coordinate_attrs_sample_attrs(loader, sample_path)
        except Exception as e:
            label = QtWidgets.QLabel(
                "Could not inspect the selected file. Edit name_map directly instead."
                f"\n\n{type(e).__name__}: {e}"
            )
            label.setWordWrap(True)
            layout.addWidget(label)
            self._add_button_box(layout, editor_enabled=False)
            return

        if not attrs:
            layout.addWidget(
                QtWidgets.QLabel("No attributes were found in the sample.")
            )
            self._add_button_box(layout, editor_enabled=False)
            return

        table = QtWidgets.QTableWidget(len(attrs), 2, self)
        table.setHorizontalHeaderLabels(["File attribute", "Renamed to"])
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        layout.addWidget(table)
        self._table = table

        built_in_reversed = loader._reverse_mapping(loader.name_map)
        current_reversed = loader._reverse_mapping(current_name_map)
        represented_originals = set(attrs)
        self._unmatched_pairs = tuple(
            (new_name, original)
            for new_name, original in _iter_name_map_pairs(current_name_map)
            if original not in represented_originals
        )

        for row, original in enumerate(attrs):
            built_in_new = built_in_reversed.get(original)
            current_new = current_reversed.get(original)
            target = built_in_new or current_new or ""
            is_built_in = built_in_new is not None

            original_item = QtWidgets.QTableWidgetItem(original)
            target_item = QtWidgets.QTableWidgetItem(target)
            if is_built_in:
                flags = QtCore.Qt.ItemFlag.ItemIsSelectable
                tooltip = f"{original} is already renamed to {built_in_new}"
                original_item.setFlags(flags)
                target_item.setFlags(flags)
                original_item.setToolTip(tooltip)
                target_item.setToolTip(tooltip)
            else:
                original_item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                target_item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                    | QtCore.Qt.ItemFlag.ItemIsEditable
                )
            table.setItem(row, 0, original_item)
            table.setItem(row, 1, target_item)
            self._rows.append((original, target_item, is_built_in))

        table.resizeColumnsToContents()
        header = table.horizontalHeader()
        if header is not None:  # pragma: no branch
            header.setStretchLastSection(True)
        self._add_button_box(layout, editor_enabled=True)

    def _add_button_box(
        self, layout: QtWidgets.QVBoxLayout, *, editor_enabled: bool
    ) -> None:
        buttons = (
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            if editor_enabled
            else QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        button_box = QtWidgets.QDialogButtonBox(buttons)
        if editor_enabled:
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)
        else:
            button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def selected_name_map(self) -> dict[str, str | Iterable[str]]:
        return self._selected_name_map

    def accept(self) -> None:
        pairs = list(self._unmatched_pairs)
        for original, target_item, is_built_in in self._rows:
            if is_built_in:
                continue
            new_name = target_item.text().strip()
            if new_name and new_name != original:
                pairs.append((new_name, original))
        self._selected_name_map = _name_map_from_pairs(pairs)
        super().accept()


class _CoordinateAttrsPickerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        loader: erlab.io.dataloader.LoaderBase,
        sample_path: pathlib.Path | None,
        current_values: tuple[str, ...],
        name_map: dict[str, str | Iterable[str]],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Promote Attributes to Coordinates")
        self._selected_values: tuple[str, ...] = current_values
        self._items: list[QtWidgets.QTreeWidgetItem] = []
        self._unmatched_values: tuple[str, ...] = ()

        layout = QtWidgets.QVBoxLayout(self)

        if sample_path is None:
            layout.addWidget(QtWidgets.QLabel("No sample file is available."))
            self._add_button_box(layout, picker_enabled=False)
            return

        try:
            attrs = _coordinate_attrs_sample_attrs(loader, sample_path)
        except Exception as e:
            label = QtWidgets.QLabel(
                "Could not inspect the selected file. "
                "Edit coordinate_attrs directly instead.\n\n"
                f"{type(e).__name__}: {e}"
            )
            label.setWordWrap(True)
            layout.addWidget(label)
            self._add_button_box(layout, picker_enabled=False)
            return

        if not attrs:
            layout.addWidget(
                QtWidgets.QLabel("No attributes were found in the sample.")
            )
            self._add_button_box(layout, picker_enabled=False)
            return

        tree = QtWidgets.QTreeWidget(self)
        tree.setColumnCount(2)
        tree.setHeaderLabels(["File attribute", "Renamed to"])
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setUniformRowHeights(True)
        layout.addWidget(tree)
        self._tree = tree

        built_in = set(loader.coordinate_attrs)
        selected = set(current_values)
        name_map_reversed = loader._reverse_mapping(loader.name_map | name_map)
        represented_values: set[str] = set()

        for original in attrs:
            renamed = name_map_reversed.get(original, original)
            renamed_text = renamed if renamed != original else ""
            represented_values.add(renamed)
            is_built_in = renamed in built_in
            is_checked = is_built_in or renamed in selected or original in selected
            item = QtWidgets.QTreeWidgetItem(tree, [original, renamed_text])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, renamed)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, is_built_in)
            item.setCheckState(
                0,
                QtCore.Qt.CheckState.Checked
                if is_checked
                else QtCore.Qt.CheckState.Unchecked,
            )
            flags = (
                QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            if not is_built_in:
                flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
            item.setFlags(flags)
            item.setDisabled(is_built_in)
            if original != renamed:
                item.setToolTip(0, f"{original} is renamed to {renamed}")
                item.setToolTip(1, f"Original attribute: {original}")
            if is_built_in:
                item.setToolTip(
                    0,
                    "Already promoted by the selected loader"
                    if original == renamed
                    else f"{original} is renamed to {renamed} and already promoted "
                    "by the selected loader",
                )
                item.setToolTip(1, "Already promoted by the selected loader")
            self._items.append(item)

        for column in range(tree.columnCount()):
            tree.resizeColumnToContents(column)

        self._unmatched_values = tuple(
            v
            for v in current_values
            if v not in represented_values and v not in built_in
        )
        self._add_button_box(layout, picker_enabled=True)

    def _add_button_box(
        self, layout: QtWidgets.QVBoxLayout, *, picker_enabled: bool
    ) -> None:
        buttons = (
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            if picker_enabled
            else QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        button_box = QtWidgets.QDialogButtonBox(buttons)
        if picker_enabled:
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)
        else:
            button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def selected_coordinate_attrs(self) -> tuple[str, ...]:
        return self._selected_values

    def accept(self) -> None:
        values: list[str] = []
        for item in self._items:
            is_built_in = bool(item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1))
            if is_built_in:
                continue
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                values.append(str(item.data(0, QtCore.Qt.ItemDataRole.UserRole)))
        values.extend(v for v in self._unmatched_values if v not in values)
        self._selected_values = tuple(values)
        super().accept()


class _LoaderOptionsWidget(QtWidgets.QWidget):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        valid_loaders: dict[str, tuple[Callable, dict]],
        *,
        loader_extensions: dict[str, dict[str, typing.Any]] | None = None,
        sample_paths: Iterable[str | pathlib.Path] | None = None,
    ) -> None:
        super().__init__(parent)

        self._valid_loaders = valid_loaders
        self._loader_extensions = loader_extensions or {}
        self._sample_paths = tuple(pathlib.Path(p) for p in sample_paths or ())
        self._checked_kwargs: dict[str, typing.Any] | None = None
        self._checked_loader_extensions: dict[str, typing.Any] | None = None
        self._extensions_available = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._button_group = QtWidgets.QButtonGroup(self)

        for i, name in enumerate(self._valid_loaders.keys()):
            radio_button = QtWidgets.QRadioButton(name, self)
            self._button_group.addButton(radio_button, i)
            layout.addWidget(radio_button)

        self._button_group.idToggled.connect(self._update_func_kwargs)

        layout.addSpacing(10)
        layout.addStretch()

        self.func_label = QtWidgets.QLabel()
        layout.addWidget(self.func_label)

        self.kwargs_line = erlab.interactive.utils.SingleLinePlainTextEdit()
        self.kwargs_line.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        self.highlighter = erlab.interactive.utils.PythonHighlighter(
            self.kwargs_line.document()
        )
        self.kwargs_line.textChanged.connect(self._clear_checked_values)

        layout.addWidget(self.kwargs_line)

        self.extensions_toggle = QtWidgets.QToolButton()
        self.extensions_toggle.setText("Loader Extensions")
        self.extensions_toggle.setCheckable(True)
        self.extensions_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.extensions_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.extensions_toggle.setToolTip(_LOADER_EXTENSION_TOGGLE_TOOLTIP)
        self.extensions_toggle.toggled.connect(self._set_extensions_expanded)
        layout.addWidget(self.extensions_toggle)

        self.extensions_group = QtWidgets.QWidget()
        extensions_layout = QtWidgets.QFormLayout(self.extensions_group)
        self.loader_extension_lines: dict[
            str, erlab.interactive.utils.SingleLinePlainTextEdit
        ] = {}
        self.loader_extension_highlighters: list[
            erlab.interactive.utils.PythonHighlighter
        ] = []
        self.loader_extension_fields: dict[str, QtWidgets.QWidget] = {}
        self.name_map_editor_button: QtWidgets.QToolButton | None = None
        self.coordinate_attrs_picker_button: QtWidgets.QToolButton | None = None
        for key in inspect.signature(
            erlab.io.dataloader.LoaderBase.extend_loader
        ).parameters:
            if key == "self":
                continue
            line = erlab.interactive.utils.SingleLinePlainTextEdit()
            line.setFont(
                QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
            )
            self.loader_extension_highlighters.append(
                erlab.interactive.utils.PythonHighlighter(line.document())
            )
            self.loader_extension_lines[key] = line
            tooltip = _LOADER_EXTENSION_TOOLTIPS.get(
                key,
                _tooltip_with_example(
                    "Literal value passed to erlab.io.extend_loader for this load.",
                    "None",
                ),
            )
            line.setToolTip(tooltip)
            line.textChanged.connect(self._clear_checked_values)
            label = QtWidgets.QLabel(key)
            label.setToolTip(tooltip)
            if key in {"name_map", "coordinate_attrs"}:
                field = QtWidgets.QWidget()
                field_layout = QtWidgets.QHBoxLayout(field)
                field_layout.setContentsMargins(0, 0, 0, 0)
                field_layout.setSpacing(4)
                line.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
                field_layout.addWidget(line, 1)
                button = QtWidgets.QToolButton()
                if key == "name_map":
                    button.setText("Edit...")
                    button.setToolTip(
                        "Inspect the selected file and edit attribute renames."
                    )
                    button.clicked.connect(self._open_name_map_editor)
                    self.name_map_editor_button = button
                else:
                    button.setText("Edit...")
                    button.setToolTip(
                        "Inspect the selected file and choose coordinate attributes."
                    )
                    button.clicked.connect(self._open_coordinate_attrs_picker)
                    self.coordinate_attrs_picker_button = button
                field_layout.addWidget(button)
            else:
                field = line
            field.setToolTip(tooltip)
            self.loader_extension_fields[key] = field
            extensions_layout.addRow(label, field)
        layout.addWidget(self.extensions_group)
        self.extensions_toggle.hide()
        self.extensions_group.hide()

    def check_filter(self, name_filter: str | None) -> None:
        if not self._valid_loaders:
            raise ValueError("No loaders are available")
        index = (
            tuple(self._valid_loaders.keys()).index(name_filter)
            if name_filter in self._valid_loaders
            else 0
        )
        self._button_group.buttons()[index].setChecked(True)
        self._update_func_kwargs()

    @QtCore.Slot()
    def _update_func_kwargs(self) -> None:
        checked_id = self._button_group.checkedId()
        if checked_id < 0:
            return
        self._clear_checked_values()
        name_filter = list(self._valid_loaders.keys())[checked_id]
        func, kargs = self._valid_loaders[name_filter]
        self.func_label.setText(f"Arguments for <code>{func.__name__}</code>:")
        self.kwargs_line.setText(_kwargs_to_text(kargs))
        is_loader_func = _is_loader_func(func)
        self._extensions_available = is_loader_func
        self.extensions_toggle.setVisible(is_loader_func)
        self.extensions_toggle.setEnabled(is_loader_func)
        loader_extensions = self._loader_extensions.get(name_filter, {})
        n_extensions = sum(value is not None for value in loader_extensions.values())
        self.extensions_toggle.setText(
            f"Loader Extensions ({n_extensions} set)"
            if n_extensions
            else "Loader Extensions"
        )
        was_blocked = self.extensions_toggle.blockSignals(True)
        self.extensions_toggle.setChecked(False)
        self.extensions_toggle.blockSignals(was_blocked)
        self._set_extensions_expanded(False)
        for key, line in self.loader_extension_lines.items():
            value = loader_extensions.get(key)
            line.setText("" if value is None else repr(value))

    @QtCore.Slot()
    def _clear_checked_values(self) -> None:
        self._checked_kwargs = None
        self._checked_loader_extensions = None

    @QtCore.Slot()
    def _open_name_map_editor(self) -> None:
        line = self.loader_extension_lines["name_map"]
        try:
            current_name_map = _name_map_from_literal(line.text())
        except Exception as e:
            erlab.interactive.utils.MessageDialog.critical(
                self,
                "Error",
                "Invalid loader arguments.",
                str(e),
            )
            return

        name_filter = list(self._valid_loaders.keys())[self._button_group.checkedId()]
        func = self._valid_loaders[name_filter][0]
        loader = getattr(func, "__self__", None)
        if not isinstance(loader, erlab.io.dataloader.LoaderBase):
            return

        dialog = _NameMapEditorDialog(
            self,
            loader,
            self._sample_paths[0] if self._sample_paths else None,
            current_name_map,
        )
        if dialog.exec():
            line.setText(_name_map_literal(dialog.selected_name_map()))

    @QtCore.Slot()
    def _open_coordinate_attrs_picker(self) -> None:
        line = self.loader_extension_lines["coordinate_attrs"]
        try:
            current_values = _string_tuple_from_literal("coordinate_attrs", line.text())
            name_map = _name_map_from_literal(
                self.loader_extension_lines["name_map"].text()
            )
        except Exception as e:
            erlab.interactive.utils.MessageDialog.critical(
                self,
                "Error",
                "Invalid loader arguments.",
                str(e),
            )
            return

        name_filter = list(self._valid_loaders.keys())[self._button_group.checkedId()]
        func = self._valid_loaders[name_filter][0]
        loader = getattr(func, "__self__", None)
        if not isinstance(loader, erlab.io.dataloader.LoaderBase):
            return

        dialog = _CoordinateAttrsPickerDialog(
            self,
            loader,
            self._sample_paths[0] if self._sample_paths else None,
            current_values,
            name_map,
        )
        if dialog.exec():
            line.setText(_coordinate_attrs_literal(dialog.selected_coordinate_attrs()))

    @QtCore.Slot(bool)
    def _set_extensions_expanded(self, expanded: bool) -> None:
        self.extensions_group.setVisible(expanded and self._extensions_available)
        self.extensions_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if expanded
            else QtCore.Qt.ArrowType.RightArrow
        )
        self.updateGeometry()
        self.adjustSize()
        window = self.window()
        if window is not None:
            window.adjustSize()

    def _parse_checked_values(
        self,
    ) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
        kwargs = _text_to_kwargs(self.kwargs_line.text())
        loader_extensions: dict[str, typing.Any] = {}
        name_filter = list(self._valid_loaders.keys())[self._button_group.checkedId()]
        func = self._valid_loaders[name_filter][0]
        if _is_loader_func(func):
            for key, line in self.loader_extension_lines.items():
                parsed = _text_to_loader_extension_value(key, line.text())
                if parsed is not None:
                    loader_extensions.update(parsed)
            loader = getattr(func, "__self__", None)
            if loader_extensions and isinstance(loader, erlab.io.dataloader.LoaderBase):
                with loader.extend_loader(**loader_extensions):
                    pass
        return kwargs, loader_extensions

    def validate_checked_values(self) -> bool:
        try:
            self._checked_kwargs, self._checked_loader_extensions = (
                self._parse_checked_values()
            )
        except Exception as e:
            erlab.interactive.utils.MessageDialog.critical(
                self,
                "Error",
                "Invalid loader arguments.",
                str(e),
            )
            return False
        return True

    def checked_filter(self) -> tuple[str, Callable, dict[str, typing.Any]]:
        idx = self._button_group.checkedId()
        filter_name = list(self._valid_loaders.keys())[idx]
        func = self._valid_loaders[filter_name][0]
        if self._checked_kwargs is None or self._checked_loader_extensions is None:
            self._checked_kwargs, self._checked_loader_extensions = (
                self._parse_checked_values()
            )
        kwargs = self._checked_kwargs.copy()
        if self._checked_loader_extensions:
            kwargs["loader_extensions"] = self._checked_loader_extensions.copy()
        return filter_name, func, kwargs


class _NameFilterDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        valid_loaders: dict[str, tuple[Callable, dict]],
        *,
        loader_extensions: dict[str, dict[str, typing.Any]] | None = None,
        sample_paths: Iterable[str | pathlib.Path] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Loader")

        layout = QtWidgets.QVBoxLayout(self)
        self.options_widget = _LoaderOptionsWidget(
            self,
            valid_loaders,
            loader_extensions=loader_extensions,
            sample_paths=sample_paths,
        )
        layout.addWidget(self.options_widget)

        self._valid_loaders = self.options_widget._valid_loaders
        self._button_group = self.options_widget._button_group
        self.func_label = self.options_widget.func_label
        self.kwargs_line = self.options_widget.kwargs_line
        self.highlighter = self.options_widget.highlighter
        self.extensions_toggle = self.options_widget.extensions_toggle
        self.extensions_group = self.options_widget.extensions_group
        self.loader_extension_lines = self.options_widget.loader_extension_lines
        self.loader_extension_highlighters = (
            self.options_widget.loader_extension_highlighters
        )
        self.loader_extension_fields = self.options_widget.loader_extension_fields
        self.name_map_editor_button = self.options_widget.name_map_editor_button
        self.coordinate_attrs_picker_button = (
            self.options_widget.coordinate_attrs_picker_button
        )

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def check_filter(self, name_filter: str | None) -> None:
        self.options_widget.check_filter(name_filter)

    def accept(self) -> None:
        if not self.options_widget.validate_checked_values():
            return
        super().accept()

    def checked_filter(self) -> tuple[str, Callable, dict[str, typing.Any]]:
        return self.options_widget.checked_filter()


class _ChooseFromDataTreeDialog(QtWidgets.QDialog):
    def __init__(
        self,
        manager: ImageToolManager,
        tree: xarray.DataTree,
        mode: typing.Literal["save", "load"],
        root_keys: Iterable[str] | None = None,
    ) -> None:
        super().__init__(manager)
        self._manager = weakref.ref(manager)
        self._root_keys = None if root_keys is None else tuple(root_keys)

        self._saving: bool = mode == "save"

        if self._saving:
            self.setWindowTitle("Select Tools to Save")
        else:
            self.setWindowTitle("Select Tools to Add")

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
                    if not only_children or self._is_figure_item(item):
                        item.setCheckState(0, state)
                    self._set_child_check_state(item, state)

    def _set_child_check_state(
        self, item: QtWidgets.QTreeWidgetItem, state: QtCore.Qt.CheckState
    ) -> None:
        for i in range(item.childCount()):
            child = item.child(i)
            if child is None:
                continue
            child.setCheckState(0, state)
            self._set_child_check_state(child, state)

    @staticmethod
    def _is_figure_item(item: QtWidgets.QTreeWidgetItem) -> bool:
        path = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return isinstance(path, str) and path.startswith("figures/")

    def _node_payload(
        self, node: xarray.DataTree
    ) -> tuple[typing.Literal["imagetool", "tool"], xr.Dataset]:
        if "imagetool" in node:
            return (
                "imagetool",
                typing.cast("xarray.DataTree", node["imagetool"]).to_dataset(
                    inherit=False
                ),
            )
        if "tool" in node:
            return "tool", typing.cast("xarray.DataTree", node["tool"]).to_dataset(
                inherit=False
            )
        raise ValueError("Workspace node does not contain a supported payload")

    def _populate_tree_item(
        self,
        parent_item: QtWidgets.QTreeWidgetItem,
        node: xarray.DataTree,
        *,
        key: str,
        root_name: str | None = None,
    ) -> QtWidgets.QTreeWidgetItem:
        kind, ds = self._node_payload(node)
        title_attr = "itool_title" if kind == "imagetool" else "tool_title"
        title = str(ds.attrs.get(title_attr, ""))
        text = title
        if root_name is not None:
            text = root_name if not title else f"{root_name}: {title}"
        if not text:
            text = key

        item = QtWidgets.QTreeWidgetItem(parent_item, [text])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, key)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(0, QtCore.Qt.CheckState.Checked)

        if "childtools" in node:
            for child_key, child_node in typing.cast(
                "xarray.DataTree", node["childtools"]
            ).items():
                if isinstance(child_node, xr.DataTree):
                    self._populate_tree_item(item, child_node, key=child_key)
        return item

    def _populate_tree(self, tree: xarray.DataTree) -> None:
        root: QtWidgets.QTreeWidgetItem | None = self._tree_widget.invisibleRootItem()
        manager = self._manager()

        if root is not None and manager is not None:  # pragma: no branch
            start = int(manager.next_idx)
            n_items = 0
            keys = tree if self._root_keys is None else self._root_keys
            for key in keys:
                if key not in tree:
                    continue
                node = tree[key]
                if not isinstance(node, xr.DataTree):
                    continue
                name = str(key) if self._saving else str(start + n_items)
                item = self._populate_tree_item(root, node, key=key, root_name=name)
                self._tree_widget.addTopLevelItem(item)
                n_items += 1
            if not self._saving and "figures" in tree:
                figures = typing.cast("xarray.DataTree", tree["figures"])
                for figure_key, figure_node in figures.items():
                    if isinstance(figure_node, xr.DataTree):
                        item = self._populate_tree_item(
                            root,
                            figure_node,
                            key=f"figures/{figure_key}",
                        )
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
        if column != 0:
            return

        check_state = item.checkState(0)
        if check_state != QtCore.Qt.CheckState.PartiallyChecked:
            self._tree_widget.blockSignals(True)
            self._set_child_check_state(item, check_state)
            self._tree_widget.blockSignals(False)

        parent = item.parent()
        while parent is not None:
            child_states = [
                typing.cast("QtWidgets.QTreeWidgetItem", parent.child(i)).checkState(0)
                for i in range(parent.childCount())
            ]
            if not child_states:
                break
            if all(state == QtCore.Qt.CheckState.Checked for state in child_states):
                state = QtCore.Qt.CheckState.Checked
            elif all(state == QtCore.Qt.CheckState.Unchecked for state in child_states):
                state = QtCore.Qt.CheckState.Unchecked
            else:
                state = QtCore.Qt.CheckState.PartiallyChecked
            self._tree_widget.blockSignals(True)
            parent.setCheckState(0, state)
            self._tree_widget.blockSignals(False)
            parent = parent.parent()


class _ChooseFromWorkspaceManifestDialog(QtWidgets.QDialog):
    def __init__(
        self,
        manager: ImageToolManager,
        manifest: Mapping[str, typing.Any],
        mode: typing.Literal["load"] = "load",
    ) -> None:
        super().__init__(manager)
        del mode
        self._manager = weakref.ref(manager)
        self.setWindowTitle("Select Tools to Add")

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
        self._populate_tree(manifest)

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
            button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Ok),
        )
        btn_cancel = typing.cast(
            "QtWidgets.QPushButton",
            button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel),
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
        root = self._tree_widget.invisibleRootItem()
        if root is not None:  # pragma: no branch
            self._tree_widget.blockSignals(True)
            try:
                for i in range(root.childCount()):
                    item = root.child(i)
                    if item is None:
                        continue
                    if not only_children or self._is_figure_item(item):
                        item.setCheckState(0, state)
                    self._set_child_check_state(item, state)
            finally:
                self._tree_widget.blockSignals(False)

    def _set_child_check_state(
        self, item: QtWidgets.QTreeWidgetItem, state: QtCore.Qt.CheckState
    ) -> None:
        for i in range(item.childCount()):
            child = item.child(i)
            if child is None:
                continue
            child.setCheckState(0, state)
            self._set_child_check_state(child, state)

    @staticmethod
    def _is_figure_item(item: QtWidgets.QTreeWidgetItem) -> bool:
        path = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return isinstance(path, str) and path.startswith("figures/")

    @staticmethod
    def _manifest_entries(
        manifest: Mapping[str, typing.Any],
    ) -> dict[str, Mapping[str, typing.Any]]:
        entries: dict[str, Mapping[str, typing.Any]] = {}
        nodes = manifest.get("nodes", ())
        if not isinstance(nodes, list):
            return entries
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            kind = entry.get("kind")
            if isinstance(path, str) and kind in {"imagetool", "tool"}:
                entries[path] = entry
        return entries

    @staticmethod
    def _entry_text(entry: Mapping[str, typing.Any], fallback: str) -> str:
        display_name = entry.get("display_name")
        if isinstance(display_name, str) and display_name:
            return display_name
        return fallback

    def _add_manifest_item(
        self,
        parent_item: QtWidgets.QTreeWidgetItem,
        path: str,
        entry: Mapping[str, typing.Any],
        child_paths: Mapping[str, list[str]],
        entries_by_path: Mapping[str, Mapping[str, typing.Any]],
        *,
        text: str | None = None,
    ) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem(
            parent_item, [text or self._entry_text(entry, path.rsplit("/", 1)[-1])]
        )
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, path)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(0, QtCore.Qt.CheckState.Checked)
        for child_path in child_paths.get(path, ()):
            child_entry = entries_by_path.get(child_path)
            if child_entry is not None:
                self._add_manifest_item(
                    item, child_path, child_entry, child_paths, entries_by_path
                )
        return item

    def _populate_tree(self, manifest: Mapping[str, typing.Any]) -> None:
        root = self._tree_widget.invisibleRootItem()
        manager = self._manager()
        if root is None or manager is None:
            return

        entries_by_path = self._manifest_entries(manifest)
        child_paths: dict[str, list[str]] = {path: [] for path in entries_by_path}
        for path in entries_by_path:
            if "/childtools/" not in path:
                continue
            parent_path, child_key = path.rsplit("/childtools/", maxsplit=1)
            if "/" not in child_key and parent_path in child_paths:
                child_paths[parent_path].append(path)

        root_order = manifest.get("root_order", ())
        root_paths: list[str] = []
        if isinstance(root_order, list):
            for root_path in root_order:
                path = str(root_path)
                if (
                    path in entries_by_path
                    and "/" not in path
                    and path not in root_paths
                ):
                    root_paths.append(path)
        root_paths.extend(
            path
            for path in entries_by_path
            if "/" not in path and path not in root_paths
        )
        figure_paths = [
            path
            for path in entries_by_path
            if path.startswith("figures/") and path.count("/") == 1
        ]

        start = int(manager.next_idx)
        for offset, path in enumerate(root_paths):
            entry = entries_by_path[path]
            title = self._entry_text(entry, path)
            text = f"{start + offset}: {title}" if title else str(start + offset)
            self._tree_widget.addTopLevelItem(
                self._add_manifest_item(
                    root, path, entry, child_paths, entries_by_path, text=text
                )
            )
        for path in figure_paths:
            entry = entries_by_path[path]
            self._tree_widget.addTopLevelItem(
                self._add_manifest_item(root, path, entry, child_paths, entries_by_path)
            )
        self._tree_widget.expandAll()

    def selected_paths(self) -> set[str]:
        selected: set[str] = set()
        root = self._tree_widget.invisibleRootItem()
        if root is None:
            return selected

        def _collect(item: QtWidgets.QTreeWidgetItem) -> None:
            path = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if (
                isinstance(path, str)
                and item.checkState(0) != QtCore.Qt.CheckState.Unchecked
            ):
                selected.add(path)
            for i in range(item.childCount()):
                child = item.child(i)
                if child is not None:
                    _collect(child)

        for i in range(root.childCount()):
            item = root.child(i)
            if item is not None:
                _collect(item)
        return selected

    def item_for_path(self, path: str) -> QtWidgets.QTreeWidgetItem | None:
        root = self._tree_widget.invisibleRootItem()
        if root is None:
            return None

        def _find(item: QtWidgets.QTreeWidgetItem) -> QtWidgets.QTreeWidgetItem | None:
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == path:
                return item
            for i in range(item.childCount()):
                child = item.child(i)
                if child is None:
                    continue
                found = _find(child)
                if found is not None:
                    return found
            return None

        for i in range(root.childCount()):
            item = root.child(i)
            if item is None:
                continue
            found = _find(item)
            if found is not None:
                return found
        return None

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _on_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return

        check_state = item.checkState(0)
        if check_state != QtCore.Qt.CheckState.PartiallyChecked:
            self._tree_widget.blockSignals(True)
            self._set_child_check_state(item, check_state)
            self._tree_widget.blockSignals(False)

        parent = item.parent()
        while parent is not None:
            child_states = [
                typing.cast("QtWidgets.QTreeWidgetItem", parent.child(i)).checkState(0)
                for i in range(parent.childCount())
            ]
            if not child_states:
                break
            if all(state == QtCore.Qt.CheckState.Checked for state in child_states):
                state = QtCore.Qt.CheckState.Checked
            elif all(state == QtCore.Qt.CheckState.Unchecked for state in child_states):
                state = QtCore.Qt.CheckState.Unchecked
            else:
                state = QtCore.Qt.CheckState.PartiallyChecked
            self._tree_widget.blockSignals(True)
            parent.setCheckState(0, state)
            self._tree_widget.blockSignals(False)
            parent = parent.parent()
