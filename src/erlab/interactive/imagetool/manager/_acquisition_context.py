"""Workspace-scoped coordinate and attribute enrichment for ImageTool Manager."""

from __future__ import annotations

import collections.abc
import contextlib
import dataclasses
import json
import typing

import pydantic
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._figurecomposer._ui._panel_controls import _step_toolbar_button
from erlab.interactive._figurecomposer._ui._reorder_list import ReorderList
from erlab.interactive.imagetool._provenance._model import (
    ToolProvenanceOperation,
    decode_provenance_value,
    encode_provenance_value,
    full_data,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    AssignScalarCoordOperation,
)
from erlab.interactive.imagetool.manager._metadata_editor import (
    _parse_typed_value,
    _value_text,
    _value_type_name,
    _values_equal,
)

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    import xarray as xr

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager


_ContextFieldKind = typing.Literal["coordinate", "attribute"]
_ContextStatus = typing.Literal["add", "replace", "identical", "keep"]


class AcquisitionContextField(pydantic.BaseModel):
    """One reusable coordinate or attribute assignment."""

    kind: _ContextFieldKind
    name: str
    value: typing.Any
    replace_existing: bool = False

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field names cannot be empty.")
        return value

    @pydantic.model_validator(mode="after")
    def _validate_value(self) -> typing.Self:
        decoded = self.decoded_value
        if (
            self.kind == "coordinate"
            and getattr(decoded, "ndim", None) != 0
            and not isinstance(decoded, str | bytes)
        ):
            try:
                iter(decoded)
            except TypeError:
                pass
            else:
                raise ValueError("Coordinates require a scalar value.")
        try:
            json.dumps(self.value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Field values must have a stable serializable representation."
            ) from exc
        return self

    @classmethod
    def from_value(
        cls,
        *,
        kind: _ContextFieldKind,
        name: str,
        value: typing.Any,
        replace_existing: bool = False,
    ) -> typing.Self:
        return cls(
            kind=kind,
            name=name,
            value=encode_provenance_value(value),
            replace_existing=replace_existing,
        )

    @property
    def decoded_value(self) -> typing.Any:
        return decode_provenance_value(self.value)

    @property
    def key(self) -> tuple[_ContextFieldKind, str]:
        return self.kind, self.name

    def with_value(self, value: typing.Any) -> typing.Self:
        return type(self).from_value(
            kind=self.kind,
            name=self.name,
            value=value,
            replace_existing=self.replace_existing,
        )

    def operation(self) -> ToolProvenanceOperation:
        value = self.decoded_value
        if self.kind == "attribute":
            return AssignAttrsOperation(attrs={self.name: value})
        return AssignScalarCoordOperation(coord_name=self.name, value=value)

    def display_type(self) -> str:
        if self.kind == "attribute":
            return _value_type_name(self.decoded_value)
        return f"Scalar {_value_type_name(self.decoded_value)}"

    def display_value(self) -> str:
        return _value_text(self.decoded_value)


class AcquisitionContextState(pydantic.BaseModel):
    """Serializable state for one Manager workspace."""

    schema_version: int = 1
    enabled: bool = False
    fields: tuple[AcquisitionContextField, ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

    @pydantic.model_validator(mode="before")
    @classmethod
    def _disable_empty_context(cls, value: typing.Any) -> typing.Any:
        if (
            isinstance(value, collections.abc.Mapping)
            and value.get("enabled")
            and not value.get("fields")
        ):
            value = {**value, "enabled": False}
        return value

    @pydantic.model_validator(mode="after")
    def _validate_fields(self) -> typing.Self:
        keys = [field.key for field in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError("Acquisition context field names must be unique by kind.")
        return self


@dataclasses.dataclass(frozen=True)
class ContextResolution:
    operations: tuple[ToolProvenanceOperation, ...]
    statuses: tuple[_ContextStatus, ...]
    errors: tuple[str, ...] = ()

    @property
    def added(self) -> int:
        return self.statuses.count("add")

    @property
    def replaced(self) -> int:
        return self.statuses.count("replace")

    @property
    def identical(self) -> int:
        return self.statuses.count("identical")

    @property
    def kept(self) -> int:
        return self.statuses.count("keep")


@dataclasses.dataclass
class ContextIngressSummary:
    added: int = 0
    replaced: int = 0
    identical: int = 0
    kept: int = 0
    failed: int = 0

    def add_resolution(self, resolution: ContextResolution) -> None:
        self.identical += resolution.identical
        self.kept += resolution.kept
        if resolution.errors:
            self.failed += 1
            return
        self.added += resolution.added
        self.replaced += resolution.replaced


class _ContextFieldDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        field: AcquisitionContextField | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Acquisition Context Field")
        self._field: AcquisitionContextField | None = None

        layout = QtWidgets.QFormLayout(self)
        self.kind_combo = QtWidgets.QComboBox(self)
        self.kind_combo.addItem("Coordinate", userData="coordinate")
        self.kind_combo.addItem("Attribute", userData="attribute")
        layout.addRow("Kind", self.kind_combo)

        self.name_edit = QtWidgets.QLineEdit(self)
        layout.addRow("Name", self.name_edit)

        self.type_combo = QtWidgets.QComboBox(self)
        self.type_combo.addItems(["String", "Int", "Float", "Bool", "Python literal"])
        layout.addRow("Value Type", self.type_combo)

        self.value_edit = erlab.interactive.utils.SingleLinePlainTextEdit(self)
        self.value_edit.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        layout.addRow("Value", self.value_edit)

        self.existing_combo = QtWidgets.QComboBox(self)
        self.existing_combo.addItem("Keep Existing", userData=False)
        self.existing_combo.addItem("Replace Existing", userData=True)
        layout.addRow("If Present", self.existing_combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        if field is not None:
            self._restore_field(field)
        else:
            self.type_combo.setCurrentText("Float")
            self.value_edit.setText("0.0")

    @property
    def field(self) -> AcquisitionContextField:
        if self._field is None:
            raise RuntimeError("The context field dialog was not accepted.")
        return self._field

    def _restore_field(self, field: AcquisitionContextField) -> None:
        self.kind_combo.setCurrentIndex(
            self.kind_combo.findData(field.kind, QtCore.Qt.ItemDataRole.UserRole)
        )
        self.name_edit.setText(field.name)
        value = field.decoded_value
        self.type_combo.setCurrentText(_value_type_name(value))
        self.value_edit.setText(_value_text(value))
        self.existing_combo.setCurrentIndex(
            self.existing_combo.findData(
                field.replace_existing, QtCore.Qt.ItemDataRole.UserRole
            )
        )

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            kind = typing.cast("_ContextFieldKind", self.kind_combo.currentData())
            value = _parse_typed_value(
                self.type_combo.currentText(), self.value_edit.text()
            )
            self._field = AcquisitionContextField.from_value(
                kind=kind,
                name=self.name_edit.text(),
                value=value,
                replace_existing=bool(self.existing_combo.currentData()),
            )
        except Exception as exc:
            erlab.interactive.utils.MessageDialog.critical(
                self,
                "Invalid Field",
                "The acquisition context field is not valid.",
                str(exc),
            )
            return
        super().accept()


class _ContextSourcePickerDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, data: xr.DataArray) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Fields from ImageTool")
        self._selected_fields: tuple[AcquisitionContextField, ...] = ()

        layout = QtWidgets.QVBoxLayout(self)
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setHeaderLabels(["Field", "Value"])
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree)

        coords_root = QtWidgets.QTreeWidgetItem(["Coordinates"])
        attrs_root = QtWidgets.QTreeWidgetItem(["Attributes"])
        self.tree.addTopLevelItems([coords_root, attrs_root])
        for name, coord in data.coords.items():
            if not isinstance(name, str) or coord.ndim != 0:
                continue
            field: AcquisitionContextField | None = None
            with contextlib.suppress(Exception):
                field = AcquisitionContextField.from_value(
                    kind="coordinate", name=name, value=coord.item()
                )
            if field is None:
                continue
            self._add_picker_item(coords_root, field)
        for name, value in data.attrs.items():
            if not isinstance(name, str):
                continue
            field = None
            with contextlib.suppress(Exception):
                field = AcquisitionContextField.from_value(
                    kind="attribute", name=name, value=value
                )
            if field is None:
                continue
            self._add_picker_item(attrs_root, field)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_picker_item(
        self, parent: QtWidgets.QTreeWidgetItem, field: AcquisitionContextField
    ) -> None:
        item = QtWidgets.QTreeWidgetItem([field.name, field.display_value()])
        item.setFlags(
            item.flags()
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, field)
        parent.addChild(item)

    @property
    def selected_fields(self) -> tuple[AcquisitionContextField, ...]:
        return self._selected_fields

    @QtCore.Slot()
    def accept(self) -> None:
        fields: list[AcquisitionContextField] = []
        root = self.tree.invisibleRootItem()
        if root is None:
            return
        for group_index in range(root.childCount()):
            group = root.child(group_index)
            if group is None:
                continue
            for row in range(group.childCount()):
                item = group.child(row)
                if item is None:
                    continue
                if item.checkState(0) != QtCore.Qt.CheckState.Checked:
                    continue
                field = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(field, AcquisitionContextField):
                    fields.append(field)
        if not fields:
            QtWidgets.QMessageBox.warning(
                self, "No Fields Selected", "Select at least one field to add."
            )
            return
        self._selected_fields = tuple(fields)
        super().accept()


class AcquisitionContextDialog(QtWidgets.QDialog):
    def __init__(
        self, parent: QtWidgets.QWidget, controller: _AcquisitionContextController
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Acquisition Context")
        self._controller = controller
        self._manager = controller.manager
        state = controller.state
        self._fields = list(state.fields)

        layout = QtWidgets.QVBoxLayout(self)
        self.enabled_check = QtWidgets.QCheckBox(
            "Apply automatically when loading data from files", self
        )
        self.enabled_check.setChecked(state.enabled)
        layout.addWidget(self.enabled_check)

        self.table = ReorderList(0, self)
        self.table.setObjectName("manager_acquisition_context_table")
        self.table.setColumnCount(5)
        self.table.setHeaderLabels(["Kind", "Name", "Type", "Value", "If Present"])
        self.table.setRootIsDecorated(False)
        self.table.setItemsExpandable(False)
        self.table.setIndentation(0)
        self.table.setUniformRowHeights(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.itemDoubleClicked.connect(self._edit_field)
        self.table.rows_reordered.connect(self._rows_reordered)

        header = typing.cast("QtWidgets.QHeaderView", self.table.header())
        for column in (0, 1, 2, 4):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)

        field_buttons = QtWidgets.QHBoxLayout()
        field_buttons.setSpacing(4)
        self.add_button = _step_toolbar_button(
            self,
            "manager_acquisition_context_add_button",
            "Add ▾",
            "Add an acquisition context field.",
        )
        self.add_button.setAccessibleName("Add acquisition context field")
        self.add_button.setProperty("uses_inline_menu_arrow", True)
        self.add_menu = QtWidgets.QMenu(self.add_button)
        self.add_menu.setObjectName("manager_acquisition_context_add_menu")
        self.add_field_action = QtGui.QAction("Add Field…", self.add_menu)
        self.add_field_action.setObjectName(
            "manager_acquisition_context_add_field_action"
        )
        self.add_field_action.triggered.connect(self._add_field)
        self.add_menu.addAction(self.add_field_action)
        self.from_selected_action = QtGui.QAction(
            "Add from Selected ImageTool…", self.add_menu
        )
        self.from_selected_action.setObjectName(
            "manager_acquisition_context_from_selected_action"
        )
        self.from_selected_action.triggered.connect(self._from_selected)
        self.add_menu.addAction(self.from_selected_action)
        self.add_button.clicked.connect(self._show_add_menu)
        field_buttons.addWidget(self.add_button)

        self.edit_button = _step_toolbar_button(
            self,
            "manager_acquisition_context_edit_button",
            "Edit…",
            "Edit the selected acquisition context field.",
        )
        self.edit_button.setAccessibleName("Edit acquisition context field")
        self.edit_button.clicked.connect(self._edit_field)
        field_buttons.addWidget(self.edit_button)

        self.remove_button = _step_toolbar_button(
            self,
            "manager_acquisition_context_remove_button",
            "Remove",
            "Remove the selected acquisition context field.",
        )
        self.remove_button.setAccessibleName("Remove acquisition context field")
        self.remove_button.clicked.connect(self._remove_field)
        field_buttons.addWidget(self.remove_button)
        field_buttons.addStretch()
        layout.addLayout(field_buttons)
        layout.addWidget(self.table)

        footer = QtWidgets.QHBoxLayout()
        self.clear_button = QtWidgets.QPushButton("Clear Context…", self)
        self.clear_button.clicked.connect(self._clear)
        footer.addWidget(self.clear_button)
        footer.addStretch()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.accepted.connect(self._save)
        self.button_box.rejected.connect(self.reject)
        footer.addWidget(self.button_box)
        layout.addLayout(footer)

        self.table.itemSelectionChanged.connect(self._sync_buttons)
        self._populate_table()
        self._sync_buttons()
        self.resize(720, 420)

    def _selected_data(self) -> xr.DataArray | None:
        targets = self._manager._selected_imagetool_targets()
        if len(targets) != 1:
            return None
        return self._manager._metadata_editor.metadata_data(targets[0])

    def _state(self) -> AcquisitionContextState:
        return AcquisitionContextState(
            enabled=self.enabled_check.isChecked(), fields=tuple(self._fields)
        )

    @staticmethod
    def _field_row_id(field: AcquisitionContextField) -> str:
        return json.dumps(field.key, ensure_ascii=False, separators=(",", ":"))

    def _current_row(self) -> int:
        item = self.table.currentItem()
        return -1 if item is None else self.table.indexOfTopLevelItem(item)

    def _select_row(self, row: int) -> None:
        item = self.table.topLevelItem(row)
        if item is None:
            self.table.setCurrentIndex(QtCore.QModelIndex())
        else:
            self.table.setCurrentItem(item)

    def _populate_table(self) -> None:
        self.table.clear()
        for field in self._fields:
            values = (
                "Coordinate" if field.kind == "coordinate" else "Attribute",
                field.name,
                field.display_type(),
                field.display_value(),
                "Replace" if field.replace_existing else "Keep",
            )
            item = QtWidgets.QTreeWidgetItem(values)
            item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setData(
                0,
                QtCore.Qt.ItemDataRole.UserRole,
                self._field_row_id(field),
            )
            self.table.addTopLevelItem(item)

    @QtCore.Slot()
    def _sync_buttons(self) -> None:
        has_row = self._current_row() >= 0
        self.edit_button.setEnabled(has_row)
        self.remove_button.setEnabled(has_row)
        targets = self._manager._selected_imagetool_targets()
        self.from_selected_action.setEnabled(len(targets) == 1)
        self.clear_button.setEnabled(bool(self._fields))

    def _merge_fields(self, fields: Iterable[AcquisitionContextField]) -> None:
        positions = {field.key: index for index, field in enumerate(self._fields)}
        for field in fields:
            if field.key in positions:
                self._fields[positions[field.key]] = field
            else:
                positions[field.key] = len(self._fields)
                self._fields.append(field)
        self._populate_table()
        self._sync_buttons()

    @QtCore.Slot(object, object, object)
    def _rows_reordered(
        self,
        row_ids: object,
        _selected_ids: object,
        _current_id: object,
    ) -> None:
        if not isinstance(row_ids, tuple) or not all(
            isinstance(row_id, str) for row_id in row_ids
        ):
            return
        fields_by_id = {self._field_row_id(field): field for field in self._fields}
        if len(row_ids) != len(fields_by_id) or set(row_ids) != set(fields_by_id):
            self._populate_table()
            self._sync_buttons()
            return
        self._fields = [fields_by_id[row_id] for row_id in row_ids]

    @QtCore.Slot()
    def _show_add_menu(self) -> None:
        self.add_menu.popup(
            self.add_button.mapToGlobal(QtCore.QPoint(0, self.add_button.height()))
        )

    @QtCore.Slot()
    def _from_selected(self) -> None:
        data = self._selected_data()
        if data is None:
            return
        dialog = _ContextSourcePickerDialog(self, data)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._merge_fields(dialog.selected_fields)

    @QtCore.Slot()
    def _add_field(self) -> None:
        dialog = _ContextFieldDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._merge_fields((dialog.field,))

    @QtCore.Slot()
    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _edit_field(
        self,
        _item: QtWidgets.QTreeWidgetItem | None = None,
        _column: int = 0,
    ) -> None:
        row = self._current_row()
        if row < 0:
            return
        dialog = _ContextFieldDialog(self, field=self._fields[row])
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        duplicate = next(
            (
                index
                for index, field in enumerate(self._fields)
                if index != row and field.key == dialog.field.key
            ),
            None,
        )
        if duplicate is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Duplicate Field",
                "Coordinate and attribute names must be unique within each kind.",
            )
            return
        self._fields[row] = dialog.field
        self._populate_table()
        self._select_row(row)

    @QtCore.Slot()
    def _remove_field(self) -> None:
        row = self._current_row()
        if row < 0:
            return
        del self._fields[row]
        self._populate_table()
        self._sync_buttons()

    @QtCore.Slot()
    def _save(self) -> None:
        try:
            self._controller.set_state(self._state())
        except Exception as exc:
            erlab.interactive.utils.MessageDialog.critical(
                self,
                "Invalid Context",
                "The acquisition context is not valid.",
                str(exc),
            )
            return
        self.accept()

    @QtCore.Slot()
    def _clear(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Clear Acquisition Context",
            "Remove every field and turn off automatic application?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._controller.set_state(AcquisitionContextState())
        self.accept()


class _AcquisitionContextController(QtCore.QObject):
    """Coordinate acquisition-context state, UI, and file ingress."""

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self.manager = manager
        self._status_button: QtWidgets.QToolButton | None = None
        if not isinstance(manager._workspace_state.acquisition_context, dict):
            manager._workspace_state.acquisition_context = {}

    @property
    def state(self) -> AcquisitionContextState:
        try:
            return AcquisitionContextState.model_validate(
                self.manager._workspace_state.acquisition_context
            )
        except Exception:
            return AcquisitionContextState()

    def state_payload(self) -> dict[str, typing.Any]:
        return self.state.model_dump(mode="json")

    def set_state(
        self, state: AcquisitionContextState, *, mark_dirty: bool = True
    ) -> None:
        state = AcquisitionContextState.model_validate(state)
        payload = state.model_dump(mode="json")
        if state == self.state:
            self.manager._workspace_state.acquisition_context = payload
            self.refresh_ui()
            return
        self.manager._workspace_state.acquisition_context = payload
        self.refresh_ui()
        if mark_dirty:
            self.manager._workspace_controller._mark_workspace_context_dirty()

    def restore_state_payload(self, payload: typing.Any) -> None:
        try:
            state = AcquisitionContextState.model_validate(payload or {})
        except Exception:
            erlab.utils.misc.emit_user_level_warning(
                "Ignoring invalid saved acquisition context state."
            )
            state = AcquisitionContextState()
        self.set_state(state, mark_dirty=False)

    def bind_status_button(self, button: QtWidgets.QToolButton) -> None:
        self._status_button = button
        button.clicked.connect(self.show_editor)
        self.refresh_ui()

    def refresh_ui(self) -> None:
        button = self._status_button
        if button is None:
            return
        state = self.state
        active = state.enabled and bool(state.fields)
        button.setVisible(active)
        if not active:
            return
        button.setText(f"Context · {len(state.fields)} fields")
        button.setToolTip(
            "Applied automatically when loading data from files\n"
            + "\n".join(
                f"{field.name}: {field.display_value()}" for field in state.fields
            )
        )

    @QtCore.Slot()
    def show_editor(self) -> None:
        dialog = AcquisitionContextDialog(self.manager, self)
        dialog.exec()

    def resolve(
        self,
        data: xr.DataArray,
        fields: Sequence[AcquisitionContextField] | None = None,
    ) -> ContextResolution:
        if fields is None:
            fields = self.state.fields
        operations: list[ToolProvenanceOperation] = []
        attrs: dict[Hashable, typing.Any] = {}
        statuses: list[_ContextStatus] = []
        errors: list[str] = []
        for field in fields:
            try:
                desired = field.decoded_value
                if field.kind == "coordinate":
                    present = field.name in data.coords
                    if present:
                        coord = data.coords[field.name]
                        current = coord.values[()] if coord.ndim == 0 else coord.values
                    else:
                        current = None
                else:
                    present = field.name in data.attrs
                    current = data.attrs.get(field.name)
                if present and _values_equal(current, desired):
                    statuses.append("identical")
                    continue
                if present and not field.replace_existing:
                    statuses.append("keep")
                    continue
                operation = field.operation()
                operation.apply(data, parent_data=data)
                if field.kind == "attribute":
                    attrs[field.name] = desired
                else:
                    if attrs:
                        operations.append(AssignAttrsOperation(attrs=attrs))
                        attrs = {}
                    operations.append(operation)
                statuses.append("replace" if present else "add")
            except Exception as exc:
                errors.append(f"{field.name}: {exc}")
        if errors:
            return ContextResolution((), tuple(statuses), tuple(errors))
        if attrs:
            operations.append(AssignAttrsOperation(attrs=attrs))
        if operations:
            try:
                processed = full_data(*operations).apply(data)
                erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(
                    processed
                )
            except Exception as exc:
                return ContextResolution((), tuple(statuses), (str(exc),))
        return ContextResolution(tuple(operations), tuple(statuses))

    def apply_to_file_data(
        self, data: xr.DataArray
    ) -> tuple[xr.DataArray, tuple[ToolProvenanceOperation, ...], ContextResolution]:
        state = self.state
        if not state.enabled or not state.fields:
            return data, (), ContextResolution((), ())
        resolution = self.resolve(data, state.fields)
        if resolution.errors or not resolution.operations:
            return data, (), resolution
        processed = full_data(*resolution.operations).apply(data).rename(data.name)
        return processed, resolution.operations, resolution
