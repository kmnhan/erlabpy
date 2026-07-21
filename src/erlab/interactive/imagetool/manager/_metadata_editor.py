"""Logbook-style scalar metadata editing for ImageTool Manager."""

from __future__ import annotations

import ast
import collections.abc
import contextlib
import csv
import dataclasses
import io
import typing

import numpy as np
import pydantic
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive._figurecomposer._ui._panel_controls import _step_toolbar_button
from erlab.interactive.imagetool._provenance._model import (
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    _ProvenanceStepRef,
    compose_full_provenance,
    full_data,
    iter_operation_refs,
    require_live_source_spec,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignAttrsOperation,
    AssignScalarCoordOperation,
)

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence

    import xarray as xr

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._provenance_edit._controller import (
        _ValidatedProvenanceEdit,
    )
    from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode


MetadataFieldKind = typing.Literal["coordinate", "attribute"]
MetadataOrigin = typing.Literal["source", "assigned", "missing"]
MetadataValueType = typing.Literal["String", "Int", "Float", "Bool", "Python literal"]
_MISSING = object()
_UNKNOWN_REFERENCE = object()


def _values_equal(left: typing.Any, right: typing.Any) -> bool:
    left = erlab.utils.misc._convert_to_native(left)
    right = erlab.utils.misc._convert_to_native(right)
    if type(left) is not type(right):
        return False
    if isinstance(left, list | tuple):
        return len(left) == len(right) and all(
            _values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, collections.abc.Mapping):
        if left.keys() != right.keys():
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
    except Exception:
        return False
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind != right_array.dtype.kind:
        return False
    try:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left_array, right_array))


def _value_type_name(value: typing.Any) -> MetadataValueType:
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, int | np.integer):
        return "Int"
    if isinstance(value, float | np.floating):
        return "Float"
    if isinstance(value, str):
        return "String"
    return "Python literal"


def _value_text(value: typing.Any) -> str:
    if isinstance(value, str):
        return value
    return erlab.interactive.utils._parse_single_arg(value)


def _parse_typed_value(type_name: str, text: str) -> typing.Any:
    stripped = text.strip()
    match type_name:
        case "String":
            return text
        case "Int":
            return int(stripped)
        case "Float":
            return float(stripped)
        case "Bool":
            lowered = stripped.casefold()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
            raise ValueError("Boolean values must be True or False.")
        case "Python literal":
            return ast.literal_eval(stripped)
        case _:
            raise ValueError(f"Unknown value type {type_name!r}.")


def _parse_editor_value(reference: typing.Any, text: str) -> typing.Any:
    """Parse table input while allowing deliberate metadata type changes."""
    if reference is _UNKNOWN_REFERENCE:
        try:
            return ast.literal_eval(text.strip())
        except (SyntaxError, ValueError):
            return text
    try:
        return _parse_typed_value(_value_type_name(reference), text)
    except (SyntaxError, TypeError, ValueError) as typed_error:
        try:
            return ast.literal_eval(text.strip())
        except (SyntaxError, ValueError):
            raise typed_error from None


class MetadataField(pydantic.BaseModel):
    """Stable identity for one scalar coordinate or attribute column."""

    kind: MetadataFieldKind
    name: str

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Metadata field names cannot be empty.")
        return value

    def operation(self, value: typing.Any) -> ToolProvenanceOperation:
        if self.kind == "attribute":
            return AssignAttrsOperation(attrs={self.name: value})
        return AssignScalarCoordOperation(coord_name=self.name, value=value)


class MetadataFieldWidth(pydantic.BaseModel):
    """Saved width for a metadata column with a stable field identity."""

    field: MetadataField
    width: int = pydantic.Field(ge=1, le=10_000)

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class MetadataFieldType(pydantic.BaseModel):
    """Saved parse type for a metadata field that may be absent later."""

    field: MetadataField
    value_type: MetadataValueType

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class MetadataEditorLayout(pydantic.BaseModel):
    """Workspace-scoped logbook column layout."""

    schema_version: int = 3
    initialized: bool = False
    fields: tuple[MetadataField, ...] = ()
    data_column_width: int | None = pydantic.Field(default=None, ge=1, le=10_000)
    field_widths: tuple[MetadataFieldWidth, ...] = ()
    field_types: tuple[MetadataFieldType, ...] = ()

    model_config = pydantic.ConfigDict(frozen=True, extra="ignore")


@dataclasses.dataclass(frozen=True)
class MetadataCell:
    field: MetadataField
    present: bool
    value: typing.Any = None
    assignment_ref: _ProvenanceStepRef | None = None
    assignment: ToolProvenanceOperation | None = None
    editable: bool = True

    @property
    def origin(self) -> MetadataOrigin:
        if not self.present:
            return "missing"
        if self.assignment_ref is not None:
            return "assigned"
        return "source"


@dataclasses.dataclass(frozen=True)
class MetadataCellEdit:
    field: MetadataField
    value: typing.Any = None
    revert: bool = False


@dataclasses.dataclass(frozen=True)
class _MetadataTargetEdit:
    target: int | str
    node: _ManagedWindowNode
    scope: typing.Literal["display", "source"]
    kind: typing.Literal["none", "direct", "validated"]
    data: xr.DataArray | None = None
    spec: ToolProvenanceSpec | None = None
    replay_source_data: xr.DataArray | None = None
    validated: _ValidatedProvenanceEdit | None = None


def _field_value(data: xr.DataArray, field: MetadataField) -> typing.Any:
    if field.kind == "coordinate":
        if field.name not in data.coords or data.coords[field.name].ndim != 0:
            return _MISSING
        return data.coords[field.name].values[()]
    return data.attrs.get(field.name, _MISSING)


def _operation_value(
    operation: ToolProvenanceOperation, field: MetadataField
) -> typing.Any:
    if (
        field.kind == "coordinate"
        and isinstance(operation, AssignScalarCoordOperation)
        and operation.coord_name == field.name
    ):
        return operation.decoded_value
    if (
        field.kind == "attribute"
        and isinstance(operation, AssignAttrsOperation)
        and field.name in operation.attrs
    ):
        return operation.attrs[field.name]
    return _MISSING


def _stable_edit_value(field: MetadataField, value: typing.Any) -> bool:
    """Return whether a value survives provenance serialization unchanged."""
    try:
        parsed = _parse_editor_value(value, _value_text(value))
        operation = field.operation(parsed)
        restored_operation = type(operation).model_validate_json(
            operation.model_dump_json()
        )
        restored = _operation_value(restored_operation, field)
    except Exception:
        return False
    return (
        restored is not _MISSING
        and type(parsed) is type(restored)
        and _values_equal(parsed, restored)
    )


def _editable_attribute(value: typing.Any) -> bool:
    try:
        parsed = _parse_typed_value(_value_type_name(value), _value_text(value))
        operation = AssignAttrsOperation(attrs={"value": parsed})
        restored = AssignAttrsOperation.model_validate_json(
            operation.model_dump_json()
        ).attrs["value"]
    except Exception:
        return False
    return type(parsed) is type(restored) and _values_equal(parsed, restored)


def _primitive_value(value: typing.Any) -> bool:
    return value is None or isinstance(
        value, bool | int | float | str | np.integer | np.floating
    )


class _MetadataEditCommand(QtGui.QUndoCommand):
    def __init__(
        self,
        model: _MetadataTableModel,
        before_values: Mapping[tuple[int, MetadataField], typing.Any],
        before_reverted: Iterable[tuple[int, MetadataField]],
        after_values: Mapping[tuple[int, MetadataField], typing.Any],
        after_reverted: Iterable[tuple[int, MetadataField]],
        text: str,
    ) -> None:
        super().__init__(text)
        self._model = model
        self._before_values = dict(before_values)
        self._before_reverted = frozenset(before_reverted)
        self._after_values = dict(after_values)
        self._after_reverted = frozenset(after_reverted)

    def undo(self) -> None:
        self._model._restore_pending_state(self._before_values, self._before_reverted)

    def redo(self) -> None:
        self._model._restore_pending_state(self._after_values, self._after_reverted)


class _MetadataTableModel(QtCore.QAbstractTableModel):
    pendingChangesChanged = QtCore.Signal()
    editRejected = QtCore.Signal(str)

    OriginRole = QtCore.Qt.ItemDataRole.UserRole + 1
    FieldRole = QtCore.Qt.ItemDataRole.UserRole + 2
    TargetRole = QtCore.Qt.ItemDataRole.UserRole + 3
    DirtyRole = QtCore.Qt.ItemDataRole.UserRole + 4
    RevertRole = QtCore.Qt.ItemDataRole.UserRole + 5

    def __init__(
        self,
        controller: _MetadataEditorController,
        targets: Sequence[int | str],
        fields: Sequence[MetadataField],
        field_defaults: Mapping[MetadataField, typing.Any],
        parent: QtCore.QObject | None = None,
        *,
        metadata_by_target: Mapping[int | str, xr.DataArray] | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.targets = tuple(targets)
        self.fields = list(fields)
        self.field_defaults = dict(field_defaults)
        self.metadata_by_target = (
            None if metadata_by_target is None else dict(metadata_by_target)
        )
        self.cells: dict[tuple[int, MetadataField], MetadataCell] = {}
        self.edited_values: dict[tuple[int, MetadataField], typing.Any] = {}
        self.reverted: set[tuple[int, MetadataField]] = set()
        self.undo_stack = QtGui.QUndoStack(self)
        self._populate_cells()

    def _populate_cells(self) -> None:
        for row, target in enumerate(self.targets):
            data = (
                self.controller.metadata_data(target)
                if self.metadata_by_target is None
                else self.metadata_by_target[target]
            )
            matches = self.controller.matching_assignments(
                self.controller.spec_and_scope(
                    self.controller.manager._node_for_target(target)
                )[0],
                data,
                self.fields,
            )
            for field in self.fields:
                value = _field_value(data, field)
                match = matches.get(field)
                present = value is not _MISSING
                reference = (
                    value
                    if present
                    else self.field_defaults.get(field, _UNKNOWN_REFERENCE)
                )
                self.cells[row, field] = MetadataCell(
                    field=field,
                    present=present,
                    value=None if not present else value,
                    assignment_ref=None if match is None else match[0],
                    assignment=None if match is None else match[1],
                    editable=(
                        reference is _UNKNOWN_REFERENCE
                        or _stable_edit_value(field, reference)
                    ),
                )

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.targets)

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else 1 + len(self.fields)

    def field_at(self, column: int) -> MetadataField | None:
        if column <= 0 or column > len(self.fields):
            return None
        return self.fields[column - 1]

    def _owns_index(self, index: QtCore.QModelIndex) -> bool:
        return (
            index.isValid()
            and index.model() is self
            and 0 <= index.row() < len(self.targets)
            and 0 <= index.column() <= len(self.fields)
        )

    def cell_at(self, index: QtCore.QModelIndex) -> MetadataCell | None:
        if not self._owns_index(index):
            return None
        field = self.field_at(index.column())
        return None if field is None else self.cells.get((index.row(), field))

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> typing.Any:
        if not self._owns_index(index):
            return None
        target = self.targets[index.row()]
        if index.column() == 0:
            node = self.controller.manager._node_for_target(target)
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return node.display_text
            if role == QtCore.Qt.ItemDataRole.ToolTipRole:
                paths = node._file_label_paths()
                return "\n".join(str(path) for path in paths)
            if role == self.TargetRole:
                return target
            return None

        field = self.field_at(index.column())
        if field is None:
            return None
        key = (index.row(), field)
        cell = self.cells[key]
        value = self.edited_values.get(key, cell.value)
        if role in {
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        }:
            if key in self.reverted:
                return _value_text(cell.value)
            if key not in self.edited_values and not cell.present:
                return "" if role == QtCore.Qt.ItemDataRole.EditRole else "—"
            return _value_text(value)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if not cell.editable:
                return "This value type cannot be edited reliably in the table."
            if key in self.reverted:
                return "The explicit assignment will be reverted when changes apply."
            if cell.origin == "assigned":
                operation = cell.assignment
                method = (
                    "assign_coords"
                    if isinstance(operation, AssignScalarCoordOperation)
                    else "assign_attrs"
                )
                return f"Assigned by {method}."
            if cell.origin == "missing":
                return "This field is missing from the data."
            return "Loaded or produced without a matching assignment operation."
        if role == self.OriginRole:
            return cell.origin
        if role == self.FieldRole:
            return field
        if role == self.TargetRole:
            return target
        if role == self.DirtyRole:
            return key in self.edited_values or key in self.reverted
        if role == self.RevertRole:
            return key in self.reverted
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(
                QtCore.Qt.AlignmentFlag.AlignVCenter
                | (
                    QtCore.Qt.AlignmentFlag.AlignLeft
                    if isinstance(value, str)
                    else QtCore.Qt.AlignmentFlag.AlignRight
                )
            )
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> typing.Any:
        if section < 0:
            return None
        if orientation == QtCore.Qt.Orientation.Vertical:
            if section >= len(self.targets):
                return None
            return section + 1 if role == QtCore.Qt.ItemDataRole.DisplayRole else None
        if section == 0:
            return "Data" if role == QtCore.Qt.ItemDataRole.DisplayRole else None
        if section > len(self.fields):
            return None
        field = self.fields[section - 1]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            kind = "coord" if field.kind == "coordinate" else "attr"
            return f"{field.name} ({kind})"
        if role == self.FieldRole:
            return field
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not self._owns_index(index):
            return QtCore.Qt.ItemFlag.NoItemFlags
        flags = super().flags(index)
        cell = self.cell_at(index)
        if cell is not None and cell.editable:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def _parsed_edit(
        self, index: QtCore.QModelIndex, value: typing.Any
    ) -> tuple[tuple[int, MetadataField], typing.Any]:
        if not self._owns_index(index) or index.column() == 0:
            raise ValueError("Only metadata cells can be edited.")
        field = self.field_at(index.column())
        if field is None:
            raise ValueError("The metadata field is unavailable.")
        key = (index.row(), field)
        cell = self.cells[key]
        if not cell.editable:
            raise ValueError("This value type cannot be edited reliably in the table.")
        reference = self.edited_values.get(
            key,
            (
                cell.value
                if cell.present
                else self.field_defaults.get(field, _UNKNOWN_REFERENCE)
            ),
        )
        try:
            parsed = _parse_editor_value(reference, str(value))
        except Exception as exc:
            raise ValueError(
                f"Row {index.row() + 1}, {field.kind} {field.name!r}: {exc}"
            ) from exc
        if not _stable_edit_value(field, parsed):
            raise ValueError(
                f"Row {index.row() + 1}, {field.kind} {field.name!r}: "
                "the value cannot be stored without changing its type."
            )
        return key, parsed

    def _restore_pending_state(
        self,
        values: Mapping[tuple[int, MetadataField], typing.Any],
        reverted: Iterable[tuple[int, MetadataField]],
    ) -> None:
        self.edited_values = dict(values)
        self.reverted = set(reverted)
        if self.rowCount() and self.columnCount() > 1:
            self.dataChanged.emit(
                self.index(0, 1),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
                [
                    QtCore.Qt.ItemDataRole.DisplayRole,
                    self.DirtyRole,
                    self.RevertRole,
                ],
            )
        self.pendingChangesChanged.emit()

    def _push_pending_state(
        self,
        values: Mapping[tuple[int, MetadataField], typing.Any],
        reverted: Iterable[tuple[int, MetadataField]],
        command_text: str,
    ) -> None:
        reverted_set = set(reverted)
        if self.edited_values == values and self.reverted == reverted_set:
            return
        self.undo_stack.push(
            _MetadataEditCommand(
                self,
                self.edited_values,
                self.reverted,
                values,
                reverted_set,
                command_text,
            )
        )

    def set_values(
        self,
        values: Sequence[tuple[QtCore.QModelIndex, typing.Any]],
        *,
        command_text: str = "Edit Metadata",
    ) -> None:
        """Validate and stage several cell values as one model transaction."""
        prepared: dict[tuple[int, MetadataField], typing.Any] = {}
        for index, value in values:
            key, parsed = self._parsed_edit(index, value)
            prepared[key] = parsed
        edited_values = dict(self.edited_values)
        reverted = set(self.reverted)
        for key, parsed in prepared.items():
            cell = self.cells[key]
            reverted.discard(key)
            if cell.present and _values_equal(parsed, cell.value):
                edited_values.pop(key, None)
            else:
                edited_values[key] = parsed
        self._push_pending_state(edited_values, reverted, command_text)

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: typing.Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if (
            role != QtCore.Qt.ItemDataRole.EditRole
            or not self._owns_index(index)
            or index.column() == 0
        ):
            return False
        try:
            self.set_values(((index, value),))
        except ValueError as exc:
            self.editRejected.emit(str(exc))
            return False
        return True

    def set_fields(
        self,
        fields: Sequence[MetadataField],
        field_defaults: Mapping[MetadataField, typing.Any],
    ) -> None:
        self.beginResetModel()
        self.fields = list(fields)
        self.field_defaults.update(field_defaults)
        self.cells.clear()
        self._populate_cells()
        self.endResetModel()

    def add_field(self, field: MetadataField, value: typing.Any) -> None:
        fields = [*self.fields, field] if field not in self.fields else self.fields
        self.field_defaults[field] = value
        self.set_fields(fields, self.field_defaults)
        column = self.fields.index(field) + 1
        self.set_values(
            tuple(
                (self.index(row, column), _value_text(value))
                for row in range(self.rowCount())
            ),
            command_text="Add Metadata Field",
        )

    def revert_indexes(
        self,
        indexes: Iterable[QtCore.QModelIndex],
        *,
        command_text: str = "Revert Metadata Assignment",
    ) -> None:
        edited_values = dict(self.edited_values)
        reverted = set(self.reverted)
        for index in indexes:
            cell = self.cell_at(index)
            if cell is None or cell.assignment_ref is None:
                continue
            key = (index.row(), cell.field)
            edited_values.pop(key, None)
            reverted.add(key)
        self._push_pending_state(edited_values, reverted, command_text)

    def edits_by_target(
        self,
    ) -> dict[int | str, tuple[MetadataCellEdit, ...]]:
        edits: dict[int | str, list[MetadataCellEdit]] = {}
        for (row, field), value in self.edited_values.items():
            edits.setdefault(self.targets[row], []).append(
                MetadataCellEdit(field, value=value)
            )
        for row, field in self.reverted:
            edits.setdefault(self.targets[row], []).append(
                MetadataCellEdit(field, revert=True)
            )
        return {target: tuple(values) for target, values in edits.items()}

    @property
    def has_changes(self) -> bool:
        return bool(self.edited_values or self.reverted)


class _MetadataCellDelegate(QtWidgets.QStyledItemDelegate):
    """Paint unobtrusive per-cell assignment and pending-revert markers."""

    def paint(
        self,
        painter: QtGui.QPainter | None,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        super().paint(painter, option, index)
        if painter is None or index.column() == 0:
            return
        assigned = index.data(_MetadataTableModel.OriginRole) == "assigned"
        reverting = bool(index.data(_MetadataTableModel.RevertRole))
        if not assigned and not reverting:
            return
        size = min(8, max(4, option.rect.height() // 3))
        corner = option.rect.topRight()
        polygon = QtGui.QPolygon(
            [
                corner,
                corner + QtCore.QPoint(-size, 0),
                corner + QtCore.QPoint(0, size),
            ]
        )
        painter.save()
        color = option.palette.color(
            QtGui.QPalette.ColorRole.Link
            if not reverting
            else QtGui.QPalette.ColorRole.Highlight
        )
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(polygon)
        painter.restore()


def _set_bulk_values(
    parent: QtWidgets.QWidget,
    model: _MetadataTableModel,
    values: Sequence[tuple[QtCore.QModelIndex, typing.Any]],
    *,
    command_text: str,
    title: str,
    informative_text: str,
) -> bool:
    try:
        model.set_values(values, command_text=command_text)
    except ValueError as exc:
        erlab.interactive.utils.MessageDialog.critical(
            parent,
            title,
            informative_text,
            str(exc),
        )
        return False
    return True


class _MetadataTableView(QtWidgets.QTableView):
    _MIN_DATA_COLUMN_WIDTH = 160
    _MAX_DATA_COLUMN_WIDTH = 320
    _MIN_METADATA_COLUMN_WIDTH = 72
    _MAX_METADATA_COLUMN_WIDTH = 220

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.frozen_view = QtWidgets.QTableView(self)
        self.frozen_view.setObjectName("manager_metadata_editor_frozen_data_column")
        self.frozen_view.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        frozen_vertical_header = typing.cast(
            "QtWidgets.QHeaderView", self.frozen_view.verticalHeader()
        )
        frozen_horizontal_header = typing.cast(
            "QtWidgets.QHeaderView", self.frozen_view.horizontalHeader()
        )
        frozen_vertical_header.hide()
        frozen_horizontal_header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        self.frozen_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.frozen_view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.frozen_view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frozen_view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.frozen_view.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.setAlternatingRowColors(True)
        self.frozen_view.setAlternatingRowColors(True)
        horizontal_header = typing.cast(
            "QtWidgets.QHeaderView", self.horizontalHeader()
        )
        vertical_header = typing.cast("QtWidgets.QHeaderView", self.verticalHeader())
        vertical_scrollbar = typing.cast(
            "QtWidgets.QScrollBar", self.verticalScrollBar()
        )
        frozen_scrollbar = typing.cast(
            "QtWidgets.QScrollBar", self.frozen_view.verticalScrollBar()
        )
        horizontal_header.sectionResized.connect(self._sync_frozen_width)
        frozen_horizontal_header.sectionResized.connect(self._sync_main_width)
        horizontal_header.geometriesChanged.connect(self._update_frozen_geometry)
        vertical_header.sectionResized.connect(self._sync_frozen_row_height)
        vertical_header.geometriesChanged.connect(self._update_frozen_geometry)
        vertical_scrollbar.valueChanged.connect(frozen_scrollbar.setValue)
        frozen_scrollbar.valueChanged.connect(vertical_scrollbar.setValue)

    def setModel(self, model: QtCore.QAbstractItemModel | None) -> None:
        super().setModel(model)
        self.frozen_view.setModel(model)
        selection_model = self.selectionModel()
        if selection_model is not None:
            self.frozen_view.setSelectionModel(selection_model)
        if model is not None:
            model.modelReset.connect(self._sync_frozen_columns)
            model.columnsInserted.connect(self._sync_frozen_columns)
            model.columnsRemoved.connect(self._sync_frozen_columns)
        self._sync_frozen_columns()

    @QtCore.Slot()
    @QtCore.Slot(QtCore.QModelIndex, int, int)
    def _sync_frozen_columns(self, *_args: object) -> None:
        model = self.model()
        if model is None:
            return
        for column in range(model.columnCount()):
            self.frozen_view.setColumnHidden(column, column != 0)
        self.frozen_view.setColumnWidth(0, self.columnWidth(0))
        for row in range(model.rowCount()):
            self.frozen_view.setRowHeight(row, self.rowHeight(row))
        self._update_frozen_geometry()

    def resize_columns_to_contents(self) -> None:
        model = self.model()
        if model is None:
            return
        for column in range(model.columnCount()):
            self.resizeColumnToContents(column)
            minimum, maximum = (
                (self._MIN_DATA_COLUMN_WIDTH, self._MAX_DATA_COLUMN_WIDTH)
                if column == 0
                else (
                    self._MIN_METADATA_COLUMN_WIDTH,
                    self._MAX_METADATA_COLUMN_WIDTH,
                )
            )
            self.setColumnWidth(
                column, max(minimum, min(self.columnWidth(column), maximum))
            )
        self._update_frozen_geometry()

    @QtCore.Slot(int, int, int)
    def _sync_frozen_width(self, logical: int, _old: int, new: int) -> None:
        if logical == 0:
            self.frozen_view.setColumnWidth(0, new)
            self._update_frozen_geometry()

    @QtCore.Slot(int, int, int)
    def _sync_main_width(self, logical: int, _old: int, new: int) -> None:
        if logical == 0 and self.columnWidth(0) != new:
            self.setColumnWidth(0, new)

    @QtCore.Slot(int, int, int)
    def _sync_frozen_row_height(self, logical: int, _old: int, new: int) -> None:
        self.frozen_view.setRowHeight(logical, new)

    def _update_frozen_geometry(self) -> None:
        width = self.columnWidth(0)
        vertical_header = typing.cast("QtWidgets.QHeaderView", self.verticalHeader())
        horizontal_header = typing.cast(
            "QtWidgets.QHeaderView", self.horizontalHeader()
        )
        frozen_header = typing.cast(
            "QtWidgets.QHeaderView", self.frozen_view.horizontalHeader()
        )
        viewport = typing.cast("QtWidgets.QWidget", self.viewport())
        frozen_header.setFixedHeight(horizontal_header.height())
        self.frozen_view.setGeometry(
            vertical_header.width() + self.frameWidth(),
            self.frameWidth(),
            width,
            viewport.height() + horizontal_header.height(),
        )

    def resizeEvent(self, event: QtGui.QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._update_frozen_geometry()

    def moveCursor(
        self,
        cursor_action: QtWidgets.QAbstractItemView.CursorAction,
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> QtCore.QModelIndex:
        current = super().moveCursor(cursor_action, modifiers)
        if current.column() > 0:
            visual = self.visualRect(current)
            frozen_width = self.frozen_view.columnWidth(0)
            if visual.left() < frozen_width:
                scrollbar = typing.cast(
                    "QtWidgets.QScrollBar", self.horizontalScrollBar()
                )
                scrollbar.setValue(scrollbar.value() + visual.left() - frozen_width)
        return current

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is not None:
            if event.matches(QtGui.QKeySequence.StandardKey.Copy):
                self._copy_selection()
                event.accept()
                return
            if event.matches(QtGui.QKeySequence.StandardKey.Paste):
                self._paste_clipboard()
                event.accept()
                return
        super().keyPressEvent(event)

    def _copy_selection(self) -> None:
        app = QtWidgets.QApplication.instance()
        if not isinstance(app, QtWidgets.QApplication):
            return
        clipboard = app.clipboard()
        model = self.model()
        indexes = self.selectedIndexes()
        if clipboard is None or model is None or not indexes:
            return

        selected = {(index.row(), index.column()) for index in indexes}
        first_row = min(row for row, _column in selected)
        last_row = max(row for row, _column in selected)
        header = typing.cast("QtWidgets.QHeaderView", self.horizontalHeader())
        selected_visual_columns = {
            header.visualIndex(column) for _row, column in selected
        }
        first_visual = min(selected_visual_columns)
        last_visual = max(selected_visual_columns)
        logical_columns = tuple(
            logical
            for visual in range(first_visual, last_visual + 1)
            if (logical := header.logicalIndex(visual)) >= 0
            and not self.isColumnHidden(logical)
        )

        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        for row in range(first_row, last_row + 1):
            values: list[str] = []
            for column in logical_columns:
                if (row, column) not in selected:
                    values.append("")
                    continue
                index = model.index(row, column)
                role = (
                    QtCore.Qt.ItemDataRole.DisplayRole
                    if column == 0
                    else QtCore.Qt.ItemDataRole.EditRole
                )
                value = index.data(role)
                values.append("" if value is None else str(value))
            writer.writerow(values)
        clipboard.setText(output.getvalue().removesuffix("\n"))

    def _paste_clipboard(self) -> None:
        app = QtWidgets.QApplication.instance()
        if not isinstance(app, QtWidgets.QApplication):
            return
        start = self.currentIndex()
        clipboard = app.clipboard()
        if not start.isValid() or clipboard is None:
            return
        model = self.model()
        if model is None:
            return
        edits: list[tuple[QtCore.QModelIndex, str]] = []
        rows = csv.reader(io.StringIO(clipboard.text()), delimiter="\t")
        header = typing.cast("QtWidgets.QHeaderView", self.horizontalHeader())
        start_visual = header.visualIndex(start.column())
        for row_offset, values in enumerate(rows):
            for column_offset, value in enumerate(values):
                row = start.row() + row_offset
                visual_column = start_visual + column_offset
                column = header.logicalIndex(visual_column)
                if row >= model.rowCount() or column < 0 or self.isColumnHidden(column):
                    continue
                index = model.index(row, column)
                if column > 0:
                    edits.append((index, value))
        if isinstance(model, _MetadataTableModel):
            _set_bulk_values(
                self,
                model,
                edits,
                command_text="Paste Metadata",
                title="Paste Failed",
                informative_text="No metadata values were pasted.",
            )


class _MetadataFieldChooser(QtWidgets.QWidget):
    visibility_changed = QtCore.Signal(object, bool)
    visible_fields_changed = QtCore.Signal(object)
    field_added = QtCore.Signal(object, object)

    def __init__(
        self,
        fields: Sequence[MetadataField],
        visible: Sequence[MetadataField],
        coverage: Mapping[MetadataField, int],
        target_count: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._fields = tuple(fields)
        self._visible = set(visible)
        self._coverage = dict(coverage)
        self._target_count = target_count
        self._updating = False
        self._items: dict[MetadataField, QtWidgets.QTreeWidgetItem] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setPlaceholderText("Search fields…")
        self.search_edit.textChanged.connect(self._filter)
        layout.addWidget(self.search_edit)

        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setObjectName("manager_metadata_field_chooser_tree")
        self.tree.setHeaderLabels(["Field", "Present"])
        self.tree.setRootIsDecorated(True)
        self.tree.itemChanged.connect(self._item_changed)
        layout.addWidget(self.tree)

        presets = QtWidgets.QHBoxLayout()
        for text, mode in (
            ("All", "all"),
            ("Common", "common"),
            ("Incomplete", "incomplete"),
            ("Clear", "clear"),
        ):
            button = QtWidgets.QToolButton(self)
            button.setObjectName(f"manager_metadata_editor_fields_{mode}_button")
            button.setText(text)
            button.setAutoRaise(True)
            button.clicked.connect(lambda _checked=False, mode=mode: self._preset(mode))
            presets.addWidget(button)
        presets.addStretch()
        layout.addLayout(presets)

        self.add_toggle = QtWidgets.QToolButton(self)
        self.add_toggle.setText("Add Field")
        self.add_toggle.setAutoRaise(True)
        self.add_toggle.setCheckable(True)
        layout.addWidget(self.add_toggle)

        self.add_widget = QtWidgets.QWidget(self)
        add_layout = QtWidgets.QFormLayout(self.add_widget)
        add_layout.setContentsMargins(0, 4, 0, 0)
        self.kind_combo = QtWidgets.QComboBox(self.add_widget)
        self.kind_combo.addItem("Coordinate", userData="coordinate")
        self.kind_combo.addItem("Attribute", userData="attribute")
        add_layout.addRow("Kind", self.kind_combo)
        self.name_edit = QtWidgets.QLineEdit(self.add_widget)
        add_layout.addRow("Name", self.name_edit)
        self.type_combo = QtWidgets.QComboBox(self.add_widget)
        self.type_combo.addItems(["String", "Int", "Float", "Bool", "Python literal"])
        self.type_combo.setCurrentText("Float")
        add_layout.addRow("Type", self.type_combo)
        self.value_edit = QtWidgets.QLineEdit("0.0", self.add_widget)
        add_layout.addRow("Initial value", self.value_edit)
        add_buttons = QtWidgets.QHBoxLayout()
        add_buttons.addStretch()
        self.cancel_add_button = QtWidgets.QToolButton(self.add_widget)
        self.cancel_add_button.setText("Cancel")
        add_buttons.addWidget(self.cancel_add_button)
        self.confirm_add_button = QtWidgets.QToolButton(self.add_widget)
        self.confirm_add_button.setText("Add")
        add_buttons.addWidget(self.confirm_add_button)
        add_layout.addRow(add_buttons)
        self.add_widget.setVisible(False)
        layout.addWidget(self.add_widget)
        self.add_toggle.toggled.connect(self.add_widget.setVisible)
        self.cancel_add_button.clicked.connect(
            lambda: self.add_toggle.setChecked(False)
        )
        self.confirm_add_button.clicked.connect(self._add_field)

        self._populate()
        self.setMinimumSize(380, 360)

    def _populate(self) -> None:
        self._updating = True
        try:
            self.tree.clear()
            self._items.clear()
            groups = {
                "coordinate": QtWidgets.QTreeWidgetItem(["Coordinates"]),
                "attribute": QtWidgets.QTreeWidgetItem(["Attributes"]),
            }
            self.tree.addTopLevelItems(list(groups.values()))
            for field in self._fields:
                item = QtWidgets.QTreeWidgetItem(
                    [
                        field.name,
                        f"{self._coverage.get(field, 0)} / {self._target_count}",
                    ]
                )
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    QtCore.Qt.CheckState.Checked
                    if field in self._visible
                    else QtCore.Qt.CheckState.Unchecked,
                )
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, field)
                groups[field.kind].addChild(item)
                self._items[field] = item
            self.tree.expandAll()
            self.tree.resizeColumnToContents(0)
        finally:
            self._updating = False

    def add_field(self, field: MetadataField) -> None:
        if field not in self._fields:
            self._fields = (*self._fields, field)
        self._visible.add(field)
        self._coverage.setdefault(field, 0)
        self._populate()

    def set_field_visible(self, field: MetadataField, visible: bool) -> None:
        if visible:
            self._visible.add(field)
        else:
            self._visible.discard(field)
        item = self._items.get(field)
        if item is None:
            return
        self._updating = True
        try:
            item.setCheckState(
                0,
                QtCore.Qt.CheckState.Checked
                if visible
                else QtCore.Qt.CheckState.Unchecked,
            )
        finally:
            self._updating = False

    @QtCore.Slot(str)
    def _filter(self, text: str) -> None:
        needle = text.casefold().strip()
        root = self.tree.invisibleRootItem()
        if root is None:
            return
        for group_index in range(root.childCount()):
            group = root.child(group_index)
            if group is None:
                continue
            visible_children = 0
            for row in range(group.childCount()):
                item = group.child(row)
                if item is None:
                    continue
                hidden = bool(needle and needle not in item.text(0).casefold())
                item.setHidden(hidden)
                visible_children += not hidden
            group.setHidden(visible_children == 0)

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 0:
            return
        field = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(field, MetadataField):
            visible = item.checkState(0) == QtCore.Qt.CheckState.Checked
            if visible:
                self._visible.add(field)
            else:
                self._visible.discard(field)
            self.visibility_changed.emit(
                field,
                visible,
            )

    def _preset(self, mode: str) -> None:
        visible: set[MetadataField] = set()
        self._updating = True
        try:
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
                    field = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    if not isinstance(field, MetadataField):
                        continue
                    count = self._coverage.get(field, 0)
                    checked = (
                        mode == "all"
                        or (mode == "common" and count == self._target_count)
                        or (mode == "incomplete" and count < self._target_count)
                    )
                    item.setCheckState(
                        0,
                        QtCore.Qt.CheckState.Checked
                        if checked
                        else QtCore.Qt.CheckState.Unchecked,
                    )
                    if checked:
                        visible.add(field)
        finally:
            self._updating = False
        self._visible = visible
        self.visible_fields_changed.emit(
            tuple(field for field in self._fields if field in visible)
        )

    @QtCore.Slot()
    def _add_field(self) -> None:
        try:
            field = MetadataField(
                kind=typing.cast("MetadataFieldKind", self.kind_combo.currentData()),
                name=self.name_edit.text(),
            )
            value = _parse_typed_value(
                self.type_combo.currentText(), self.value_edit.text()
            )
            field.operation(value)
        except Exception as exc:
            erlab.interactive.utils.MessageDialog.critical(
                self,
                "Invalid Field",
                "The metadata field is not valid.",
                str(exc),
            )
            return
        self.field_added.emit(field, value)
        self.add_toggle.setChecked(False)


class MetadataEditorDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
        controller: _MetadataEditorController,
        targets: Sequence[int | str],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Metadata Editor")
        self.setObjectName("manager_metadata_editor_dialog")
        self.controller = controller
        self.targets = tuple(targets)
        self._column_width_tracking_enabled = False
        metadata_by_target = controller.metadata_for_targets(targets)
        self.available_fields, coverage, defaults = controller.discover_fields(
            targets, metadata_by_target=metadata_by_target
        )
        visible = controller.visible_fields(self.available_fields, defaults)

        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(4)
        self.fields_button = _step_toolbar_button(
            self,
            "manager_metadata_editor_fields_button",
            "Fields ▾",
            "Choose the metadata fields shown in the logbook.",
        )
        self.fields_button.setProperty("uses_inline_menu_arrow", True)
        toolbar.addWidget(self.fields_button)
        self.fill_down_button = _step_toolbar_button(
            self,
            "manager_metadata_editor_fill_down_button",
            "Fill Down",
            "Copy the current cell through the selected rows.",
        )
        toolbar.addWidget(self.fill_down_button)
        self.fill_series_button = _step_toolbar_button(
            self,
            "manager_metadata_editor_fill_series_button",
            "Fill Series…",
            "Fill selected rows with a numeric series.",
        )
        toolbar.addWidget(self.fill_series_button)
        self.revert_button = _step_toolbar_button(
            self,
            "manager_metadata_editor_revert_button",
            "Revert Assignment",
            "Remove the explicit assignment from the selected cells.",
        )
        toolbar.addWidget(self.revert_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.model = _MetadataTableModel(
            controller,
            targets,
            visible,
            defaults,
            parent=self,
            metadata_by_target=metadata_by_target,
        )
        self.model.editRejected.connect(self._show_cell_edit_error)
        self.undo_action = typing.cast(
            "QtGui.QAction", self.model.undo_stack.createUndoAction(self, "Undo")
        )
        self.undo_action.setObjectName("manager_metadata_editor_undo_action")
        self.undo_action.setShortcuts(
            QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.Undo)
        )
        self.undo_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        self.redo_action = typing.cast(
            "QtGui.QAction", self.model.undo_stack.createRedoAction(self, "Redo")
        )
        self.redo_action.setObjectName("manager_metadata_editor_redo_action")
        self.redo_action.setShortcuts(
            QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.Redo)
        )
        self.redo_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        self.table = _MetadataTableView(self)
        self.table.setObjectName("manager_metadata_editor_table")
        self.table.addActions((self.undo_action, self.redo_action))
        self.table.setModel(self.model)
        self.table.setItemDelegate(_MetadataCellDelegate(self.table))
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        header = typing.cast("QtWidgets.QHeaderView", self.table.horizontalHeader())
        header.setSectionsMovable(True)
        header.setFirstSectionMovable(False)
        header.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_header_menu)
        header.sectionMoved.connect(self._column_moved)
        self._resize_and_restore_columns()
        header.sectionResized.connect(self._column_resized)
        self._column_width_tracking_enabled = True
        layout.addWidget(self.table)

        self.summary_label = QtWidgets.QLabel(self)
        self.summary_label.setObjectName("manager_metadata_editor_summary")
        layout.addWidget(self.summary_label)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        apply_button = self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        if apply_button is not None:
            apply_button.setObjectName("manager_metadata_editor_apply_button")
            apply_button.clicked.connect(self._apply)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.fields_menu = QtWidgets.QMenu(self.fields_button)
        self.fields_menu.setObjectName("manager_metadata_editor_fields_menu")
        self.fields_widget = _MetadataFieldChooser(
            self.available_fields,
            visible,
            coverage,
            len(targets),
            self.fields_menu,
        )
        self.fields_widget.visibility_changed.connect(self._set_field_visible)
        self.fields_widget.visible_fields_changed.connect(self._set_visible_fields)
        self.fields_widget.field_added.connect(self._add_field)
        self.fields_action = QtWidgets.QWidgetAction(self.fields_menu)
        self.fields_action.setDefaultWidget(self.fields_widget)
        self.fields_menu.addAction(self.fields_action)
        self.fields_button.clicked.connect(self._show_fields_menu)

        self.fill_down_button.clicked.connect(self._fill_down)
        self.fill_series_button.clicked.connect(self._fill_series)
        self.revert_button.clicked.connect(self._revert_assignment)
        self.model.dataChanged.connect(self._refresh_summary)
        self.model.modelReset.connect(self._refresh_summary)
        self.model.pendingChangesChanged.connect(self._refresh_summary)
        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self.table.selectionModel()
        )
        selection_model.selectionChanged.connect(self._refresh_summary)
        selection_model.currentChanged.connect(self._refresh_summary)
        self._refresh_summary()
        self.resize(980, min(720, 280 + 34 * len(targets)))

    @QtCore.Slot(str)
    def _show_cell_edit_error(self, details: str) -> None:
        erlab.interactive.utils.MessageDialog.critical(
            self,
            "Metadata Edit Failed",
            "The metadata value could not be changed.",
            details,
        )

    @QtCore.Slot()
    def _show_fields_menu(self) -> None:
        self.fields_menu.popup(
            self.fields_button.mapToGlobal(
                QtCore.QPoint(0, self.fields_button.height())
            )
        )

    @contextlib.contextmanager
    def _suspend_column_width_tracking(self) -> Iterator[None]:
        enabled = self._column_width_tracking_enabled
        self._column_width_tracking_enabled = False
        try:
            yield
        finally:
            self._column_width_tracking_enabled = enabled

    def _resize_and_restore_columns(self) -> None:
        with self._suspend_column_width_tracking():
            self.table.resize_columns_to_contents()
            data_width = self.controller.saved_column_width(None)
            if data_width is not None:
                self.table.setColumnWidth(0, data_width)
            for column in range(1, self.model.columnCount()):
                field = self.model.field_at(column)
                if field is None:
                    continue
                width = self.controller.saved_column_width(field)
                if width is not None:
                    self.table.setColumnWidth(column, width)

    @QtCore.Slot(int, int, int)
    def _column_resized(self, logical: int, _old: int, new: int) -> None:
        if not self._column_width_tracking_enabled:
            return
        field = self.model.field_at(logical)
        if logical != 0 and field is None:
            return
        self.controller.set_column_width(field, new)

    @QtCore.Slot(object, bool)
    def _set_field_visible(self, field: object, visible: bool) -> None:
        if not isinstance(field, MetadataField):
            return
        fields = self._visual_fields()
        if visible and field not in fields:
            fields.append(field)
        elif not visible and field in fields:
            fields.remove(field)
        with self._suspend_column_width_tracking():
            self.model.set_fields(fields, self.model.field_defaults)
            self._resize_and_restore_columns()
        self.fields_widget.set_field_visible(field, visible)
        self.controller.set_layout_fields(fields, self.model.field_defaults)

    @QtCore.Slot(object)
    def _set_visible_fields(self, fields: object) -> None:
        if not isinstance(fields, tuple) or not all(
            isinstance(field, MetadataField) for field in fields
        ):
            return
        visible = typing.cast("tuple[MetadataField, ...]", fields)
        with self._suspend_column_width_tracking():
            self.model.set_fields(visible, self.model.field_defaults)
            self._resize_and_restore_columns()
        for field in self.available_fields:
            self.fields_widget.set_field_visible(field, field in visible)
        self.controller.set_layout_fields(visible, self.model.field_defaults)

    @QtCore.Slot(object, object)
    def _add_field(self, field: object, value: object) -> None:
        if not isinstance(field, MetadataField):
            return
        if field not in self.available_fields:
            self.available_fields = (*self.available_fields, field)
        with self._suspend_column_width_tracking():
            self.model.set_fields(self._visual_fields(), self.model.field_defaults)
            self.model.add_field(field, value)
            self._resize_and_restore_columns()
        self.fields_widget.add_field(field)
        self.controller.set_layout_fields(self.model.fields, self.model.field_defaults)
        self.fields_menu.close()

    @QtCore.Slot(QtCore.QPoint)
    def _show_header_menu(self, position: QtCore.QPoint) -> None:
        header = typing.cast("QtWidgets.QHeaderView", self.table.horizontalHeader())
        logical = header.logicalIndexAt(position)
        field = self.model.field_at(logical)
        if field is None:
            return
        menu = QtWidgets.QMenu(self.table)
        hide_action = menu.addAction("Hide Field")
        chosen = menu.exec(header.mapToGlobal(position))
        if chosen is hide_action:
            self._set_field_visible(field, False)

    @QtCore.Slot(int, int, int)
    def _column_moved(self, _logical: int, _old: int, _new: int) -> None:
        ordered = self._visual_fields()
        if len(ordered) == len(self.model.fields):
            self.controller.set_layout_fields(ordered, self.model.field_defaults)

    def _visual_fields(self) -> list[MetadataField]:
        header = typing.cast("QtWidgets.QHeaderView", self.table.horizontalHeader())
        ordered: list[MetadataField] = []
        for visual in range(header.count()):
            logical = header.logicalIndex(visual)
            field = self.model.field_at(logical)
            if field is not None:
                ordered.append(field)
        return ordered

    @QtCore.Slot()
    @QtCore.Slot(QtCore.QModelIndex, QtCore.QModelIndex)
    @QtCore.Slot(QtCore.QModelIndex, QtCore.QModelIndex, list)
    def _refresh_summary(self, *_args: object) -> None:
        changes = sum(len(values) for values in self.model.edits_by_target().values())
        assigned = sum(
            index.data(_MetadataTableModel.OriginRole) == "assigned"
            for index in self.table.selectedIndexes()
        )
        self.summary_label.setText(
            f"{len(self.targets)} data rows · {changes} pending changes"
        )
        self.revert_button.setEnabled(assigned > 0)
        current = self.table.currentIndex()
        cell = self.model.cell_at(current)
        value = (
            None
            if cell is None
            else self.model.edited_values.get((current.row(), cell.field), cell.value)
        )
        self.fill_down_button.setEnabled(cell is not None)
        self.fill_series_button.setEnabled(
            isinstance(value, int | float | np.integer | np.floating)
            and not isinstance(value, bool | np.bool_)
        )
        apply_button = self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        if apply_button is not None:
            apply_button.setEnabled(self.model.has_changes)

    @QtCore.Slot()
    def _fill_down(self) -> None:
        current = self.table.currentIndex()
        if self.model.cell_at(current) is None:
            return
        value = current.data(QtCore.Qt.ItemDataRole.EditRole)
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if len(rows) < 2:
            rows = list(range(current.row(), self.model.rowCount()))
        _set_bulk_values(
            self,
            self.model,
            [(self.model.index(row, current.column()), value) for row in rows],
            command_text="Fill Metadata Down",
            title="Fill Down Failed",
            informative_text="No metadata values were filled.",
        )

    @QtCore.Slot()
    def _fill_series(self) -> None:
        current = self.table.currentIndex()
        cell = self.model.cell_at(current)
        if cell is None:
            return
        value = self.model.edited_values.get((current.row(), cell.field), cell.value)
        if not isinstance(value, int | float | np.integer | np.floating) or isinstance(
            value, bool | np.bool_
        ):
            return
        start, ok = QtWidgets.QInputDialog.getDouble(
            self, "Fill Series", "Start", float(value), decimals=12
        )
        if not ok:
            return
        step, ok = QtWidgets.QInputDialog.getDouble(
            self, "Fill Series", "Step", 1.0, decimals=12
        )
        if not ok:
            return
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            rows = list(range(self.model.rowCount()))
        edits: list[tuple[QtCore.QModelIndex, str]] = []
        for offset, row in enumerate(rows):
            result = start + step * offset
            text = (
                str(int(result))
                if isinstance(value, int | np.integer)
                else repr(result)
            )
            edits.append((self.model.index(row, current.column()), text))
        _set_bulk_values(
            self,
            self.model,
            edits,
            command_text="Fill Metadata Series",
            title="Fill Series Failed",
            informative_text="No metadata values were filled.",
        )

    @QtCore.Slot()
    def _revert_assignment(self) -> None:
        self.model.revert_indexes(self.table.selectedIndexes())
        self._refresh_summary()

    @QtCore.Slot()
    def _apply(self) -> None:
        edits = self.model.edits_by_target()
        if edits and self.controller.apply_edits(edits):
            self.accept()


class _MetadataEditorController(QtCore.QObject):
    """Coordinate the logbook UI and canonical metadata operations."""

    def __init__(self, manager: ImageToolManager) -> None:
        super().__init__(manager)
        self.manager = manager
        if not isinstance(manager._workspace_state.metadata_editor_layout, dict):
            manager._workspace_state.metadata_editor_layout = {}

    @property
    def layout_state(self) -> MetadataEditorLayout:
        try:
            return MetadataEditorLayout.model_validate(
                self.manager._workspace_state.metadata_editor_layout
            )
        except Exception:
            return MetadataEditorLayout()

    def layout_payload(self) -> dict[str, typing.Any]:
        return self.layout_state.model_dump(mode="json")

    def restore_layout_payload(self, payload: typing.Any) -> None:
        try:
            state = MetadataEditorLayout.model_validate(payload or {})
        except Exception:
            erlab.utils.misc.emit_user_level_warning(
                "Ignoring invalid saved metadata editor layout."
            )
            state = MetadataEditorLayout()
        self.manager._workspace_state.metadata_editor_layout = state.model_dump(
            mode="json"
        )

    def set_layout_fields(
        self,
        fields: Sequence[MetadataField],
        field_defaults: Mapping[MetadataField, typing.Any] | None = None,
    ) -> None:
        state = self.layout_state
        field_types = {saved.field: saved.value_type for saved in state.field_types}
        if field_defaults is not None:
            for field in fields:
                value = field_defaults.get(field, _UNKNOWN_REFERENCE)
                if value is not _UNKNOWN_REFERENCE:
                    field_types[field] = _value_type_name(value)
        state = state.model_copy(
            update={
                "schema_version": 3,
                "initialized": True,
                "fields": tuple(fields),
                "field_types": tuple(
                    MetadataFieldType(field=field, value_type=value_type)
                    for field, value_type in field_types.items()
                ),
            }
        )
        self._set_layout_state(state)

    def saved_field_type(self, field: MetadataField) -> MetadataValueType | None:
        return next(
            (
                saved.value_type
                for saved in reversed(self.layout_state.field_types)
                if saved.field == field
            ),
            None,
        )

    def saved_column_width(self, field: MetadataField | None) -> int | None:
        state = self.layout_state
        if field is None:
            return state.data_column_width
        return next(
            (
                saved.width
                for saved in reversed(state.field_widths)
                if saved.field == field
            ),
            None,
        )

    def set_column_width(self, field: MetadataField | None, width: int) -> None:
        state = self.layout_state
        if field is None:
            if state.data_column_width == width:
                return
            state = state.model_copy(
                update={"schema_version": 3, "data_column_width": width}
            )
        else:
            widths = {saved.field: saved.width for saved in state.field_widths}
            if widths.get(field) == width:
                return
            widths[field] = width
            state = state.model_copy(
                update={
                    "schema_version": 3,
                    "field_widths": tuple(
                        MetadataFieldWidth(field=saved_field, width=saved_width)
                        for saved_field, saved_width in widths.items()
                    ),
                }
            )
        self._set_layout_state(state)

    def _set_layout_state(self, state: MetadataEditorLayout) -> None:
        payload = state.model_dump(mode="json")
        if payload == self.manager._workspace_state.metadata_editor_layout:
            return
        self.manager._workspace_state.metadata_editor_layout = payload
        self.manager._workspace_controller._mark_workspace_layout_dirty()

    def metadata_data(self, target: int | str) -> xr.DataArray:
        node = self.manager._node_for_target(target)
        if node.imagetool is not None:
            return node.current_public_data()
        if node.pending_workspace_memory_payload is not None:
            return self.manager._pending_workspace_imagetool_metadata_data(node)
        raise ValueError("ImageTool metadata is unavailable.")

    def metadata_for_targets(
        self, targets: Sequence[int | str]
    ) -> dict[int | str, xr.DataArray]:
        return {target: self.metadata_data(target) for target in targets}

    def discover_fields(
        self,
        targets: Sequence[int | str],
        *,
        metadata_by_target: Mapping[int | str, xr.DataArray] | None = None,
    ) -> tuple[
        tuple[MetadataField, ...],
        dict[MetadataField, int],
        dict[MetadataField, typing.Any],
    ]:
        ordered: list[MetadataField] = []
        coverage: dict[MetadataField, int] = {}
        defaults: dict[MetadataField, typing.Any] = {}
        seen: set[MetadataField] = set()
        for target in targets:
            data = (
                self.metadata_data(target)
                if metadata_by_target is None
                else metadata_by_target[target]
            )
            present: set[MetadataField] = set()
            for name, coord in data.coords.items():
                if not isinstance(name, str) or coord.ndim != 0:
                    continue
                field = MetadataField(kind="coordinate", name=name)
                if field not in seen:
                    seen.add(field)
                    ordered.append(field)
                    defaults[field] = _field_value(data, field)
                present.add(field)
            for name, value in data.attrs.items():
                if not isinstance(name, str) or not _editable_attribute(value):
                    continue
                field = MetadataField(kind="attribute", name=name)
                if field not in seen:
                    seen.add(field)
                    ordered.append(field)
                    defaults[field] = value
                present.add(field)
            for field in present:
                coverage[field] = coverage.get(field, 0) + 1

        context_fields = tuple(
            MetadataField(kind=field.kind, name=field.name)
            for field in self.manager._acquisition_context.state.fields
        )
        for context_field, source_field in zip(
            context_fields,
            self.manager._acquisition_context.state.fields,
            strict=True,
        ):
            if context_field not in seen:
                ordered.insert(0, context_field)
                seen.add(context_field)
            defaults.setdefault(context_field, source_field.decoded_value)
            coverage.setdefault(context_field, 0)
        for saved_field in self.layout_state.fields:
            if saved_field not in seen:
                ordered.append(saved_field)
                seen.add(saved_field)
            saved_type = self.saved_field_type(saved_field)
            default_by_type: dict[MetadataValueType, typing.Any] = {
                "String": "",
                "Int": 0,
                "Float": 0.0,
                "Bool": False,
                "Python literal": None,
            }
            defaults.setdefault(
                saved_field,
                (
                    _UNKNOWN_REFERENCE
                    if saved_type is None
                    else default_by_type[saved_type]
                ),
            )
            coverage.setdefault(saved_field, 0)
        ordered.sort(
            key=lambda field: (
                0 if field in context_fields else 1,
                0 if field.kind == "coordinate" else 1,
                field.name.casefold(),
            )
        )
        return tuple(ordered), coverage, defaults

    def visible_fields(
        self,
        available: Sequence[MetadataField],
        defaults: Mapping[MetadataField, typing.Any],
    ) -> tuple[MetadataField, ...]:
        state = self.layout_state
        if state.initialized:
            return state.fields
        context_keys = {
            MetadataField(kind=field.kind, name=field.name)
            for field in self.manager._acquisition_context.state.fields
        }
        return tuple(
            field
            for field in available
            if field.kind == "coordinate"
            or field in context_keys
            or _primitive_value(defaults.get(field))
        )

    @QtCore.Slot()
    def show_editor(self) -> None:
        targets = self.manager._selected_imagetool_targets()
        if not targets:
            return
        dialog = MetadataEditorDialog(self.manager, self, targets)
        dialog.exec()

    @staticmethod
    def spec_and_scope(
        node: _ManagedWindowNode,
    ) -> tuple[ToolProvenanceSpec | None, typing.Literal["display", "source"]]:
        source_spec = node.displayed_source_spec
        if source_spec is not None:
            return source_spec, "source"
        return node.displayed_provenance_spec, "display"

    @staticmethod
    def matching_assignments(
        spec: ToolProvenanceSpec | None,
        data: xr.DataArray,
        fields: Sequence[MetadataField],
    ) -> dict[MetadataField, tuple[_ProvenanceStepRef, ToolProvenanceOperation]]:
        if spec is None:
            return {}
        unmatched = set(fields)
        matches: dict[
            MetadataField, tuple[_ProvenanceStepRef, ToolProvenanceOperation]
        ] = {}
        for ref, operation in reversed(tuple(iter_operation_refs(spec))):
            for field in tuple(unmatched):
                operation_value = _operation_value(operation, field)
                if operation_value is _MISSING:
                    continue
                current_value = _field_value(data, field)
                if current_value is _MISSING or not _values_equal(
                    operation_value, current_value
                ):
                    continue
                matches[field] = (ref, operation)
                unmatched.remove(field)
        return matches

    @staticmethod
    def _append_operations(
        spec: ToolProvenanceSpec | None,
        operations: Sequence[ToolProvenanceOperation],
    ) -> ToolProvenanceSpec:
        appended = full_data(*operations)
        if spec is None:
            return appended
        with contextlib.suppress(TypeError):
            live_spec = require_live_source_spec(spec)
            if live_spec is not None:
                return live_spec.append_replacement_operations(*operations)
        composed = compose_full_provenance(spec, appended)
        if composed is None:
            raise RuntimeError("Could not compose metadata provenance.")
        return composed

    def _plan_target_edit(
        self,
        target: int | str,
        edits: Sequence[MetadataCellEdit],
    ) -> _MetadataTargetEdit:
        node = self.manager._node_for_target(target)
        current = node.current_public_data()
        spec, scope = self.spec_and_scope(node)
        matches = self.matching_assignments(
            spec, current, tuple(edit.field for edit in edits)
        )
        replacements: dict[_ProvenanceStepRef, ToolProvenanceOperation | None] = {}
        changed_refs: set[_ProvenanceStepRef] = set()
        patch_operations: list[ToolProvenanceOperation] = []
        added_coords: list[ToolProvenanceOperation] = []
        added_attrs: dict[Hashable, typing.Any] = {}
        removed_assignment = False

        for edit in edits:
            match = matches.get(edit.field)
            if edit.revert:
                if match is None:
                    continue
                ref, original = match
                replacement = replacements.get(ref, original)
                if replacement is None:
                    raise RuntimeError("Overlapping metadata assignments are invalid.")
                if isinstance(replacement, AssignAttrsOperation):
                    attrs = dict(replacement.attrs)
                    attrs.pop(edit.field.name, None)
                    replacement = (
                        AssignAttrsOperation(attrs=attrs, group=replacement.group)
                        if attrs
                        else None
                    )
                else:
                    replacement = None
                replacements[ref] = replacement
                changed_refs.add(ref)
                removed_assignment = True
                continue

            operation = edit.field.operation(edit.value)
            operation.apply(current)
            if match is not None:
                ref, original = match
                replacement = replacements.get(ref, original)
                if replacement is None:
                    raise RuntimeError("Overlapping metadata assignments are invalid.")
                if isinstance(replacement, AssignAttrsOperation):
                    attrs = dict(replacement.attrs)
                    attrs[edit.field.name] = edit.value
                    replacement = AssignAttrsOperation(
                        attrs=attrs, group=replacement.group
                    )
                else:
                    replacement = operation.model_copy(
                        update={"group": replacement.group}
                    )
                replacements[ref] = replacement
                changed_refs.add(ref)
                patch_operations.append(operation)
            elif edit.field.kind == "attribute":
                added_attrs[edit.field.name] = edit.value
            else:
                added_coords.append(operation)

        candidate = spec
        if replacements:
            if candidate is None:
                raise RuntimeError("Matching metadata assignments are unavailable.")
            for ref in sorted(
                replacements,
                key=lambda item: typing.cast("int", item.operation_index),
                reverse=True,
            ):
                replacement = replacements[ref]
                candidate = candidate._replace_operation_ref(
                    ref, () if replacement is None else (replacement,)
                )

        added_operations = [*added_coords]
        if added_attrs:
            added_operations.append(AssignAttrsOperation(attrs=added_attrs))
        if added_operations:
            candidate = self._append_operations(candidate, added_operations)

        if candidate == spec:
            return _MetadataTargetEdit(target, node, scope, "none")
        if candidate is None:
            raise RuntimeError("The edited metadata has no provenance.")

        operation_refs = tuple(iter_operation_refs(spec)) if spec is not None else ()
        positions = {ref: i for i, (ref, _operation) in enumerate(operation_refs)}
        terminal_edit = not removed_assignment
        if changed_refs:
            first_changed = min(positions[ref] for ref in changed_refs)
            terminal_edit = terminal_edit and all(
                isinstance(operation, AssignScalarCoordOperation | AssignAttrsOperation)
                for _ref, operation in operation_refs[first_changed:]
            )
        slicer_area = self.manager.get_imagetool(target).slicer_area
        terminal_edit = terminal_edit and not slicer_area.has_active_filter
        replay_source_data = node.resolved_replay_source_data()
        if candidate.kind in {"full_data", "public_data", "selection"}:
            if replay_source_data is None and (
                spec is None or not tuple(iter_operation_refs(spec))
            ):
                replay_source_data = current
            elif replay_source_data is None:
                terminal_edit = False

        if terminal_edit:
            processed = full_data(*patch_operations, *added_operations).apply(current)
            processed = processed.rename(current.name)
            erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(processed)
            return _MetadataTargetEdit(
                target,
                node,
                scope,
                "direct",
                data=processed,
                spec=candidate,
                replay_source_data=replay_source_data,
            )

        validated = self.manager._provenance_edit_controller._validated_edit(
            node,
            scope,
            candidate,
            where="validating the edited metadata",
        )
        return _MetadataTargetEdit(
            target, node, scope, "validated", validated=validated
        )

    def apply_edits(
        self,
        edits_by_target: Mapping[int | str, Sequence[MetadataCellEdit]],
    ) -> bool:
        plans: list[_MetadataTargetEdit] = []
        for target, edits in edits_by_target.items():
            try:
                plans.append(self._plan_target_edit(target, edits))
            except Exception as exc:
                erlab.interactive.utils.MessageDialog.critical(
                    self.manager,
                    "Metadata Edit Failed",
                    "The metadata changes could not be applied.",
                    str(exc),
                )
                return False

        data_changed = False
        try:
            for plan in plans:
                if plan.kind == "direct":
                    data = typing.cast("xr.DataArray", plan.data)
                    spec = typing.cast("ToolProvenanceSpec", plan.spec)
                    if plan.scope == "source":
                        self.manager._provenance_edit_controller._replace_node_data(
                            plan.node, plan.scope, data, spec, None
                        )
                    else:
                        plan.node.replace_with_detached_data(
                            data,
                            spec,
                            replay_source_data=plan.replay_source_data,
                        )
                    self.manager._update_info(uid=plan.node.uid)
                    data_changed = True
                elif plan.kind == "validated":
                    validated = typing.cast("_ValidatedProvenanceEdit", plan.validated)
                    self.manager._provenance_edit_controller._apply_validated_edit(
                        validated
                    )
                    data_changed = True
        except Exception as exc:
            erlab.interactive.utils.MessageDialog.critical(
                self.manager,
                "Metadata Edit Failed",
                "An error occurred while applying the metadata changes.",
                str(exc),
            )
            return False
        if data_changed:
            self.manager._sigDataReplaced.emit()
        return True

    def _preserved_assignments(
        self, target: int | str
    ) -> tuple[ToolProvenanceOperation, ...]:
        node = self.manager._node_for_target(target)
        current = node.current_public_data()
        spec, _scope = self.spec_and_scope(node)
        if spec is None:
            return ()
        fields: list[MetadataField] = []
        for _ref, operation in iter_operation_refs(spec):
            if isinstance(operation, AssignScalarCoordOperation) and isinstance(
                operation.coord_name, str
            ):
                field = MetadataField(kind="coordinate", name=operation.coord_name)
                if field not in fields:
                    fields.append(field)
            elif isinstance(operation, AssignAttrsOperation):
                for name in operation.attrs:
                    if isinstance(name, str):
                        field = MetadataField(kind="attribute", name=name)
                        if field not in fields:
                            fields.append(field)
        matches = self.matching_assignments(spec, current, fields)
        by_ref: dict[_ProvenanceStepRef, list[MetadataField]] = {}
        for field, (ref, _operation) in matches.items():
            by_ref.setdefault(ref, []).append(field)
        operations: list[ToolProvenanceOperation] = []
        for ref, operation in iter_operation_refs(spec):
            matched = by_ref.get(ref)
            if not matched:
                continue
            if isinstance(operation, AssignAttrsOperation):
                attrs: dict[Hashable, typing.Any] = {
                    field.name: operation.attrs[field.name] for field in matched
                }
                operations.append(
                    AssignAttrsOperation(attrs=attrs, group=operation.group)
                )
            else:
                operations.append(operation)
        return tuple(operations)

    def prepare_replacement(
        self,
        target: int | str,
        source_data: xr.DataArray,
        *,
        source_input_dtype: np.dtype[typing.Any] | str | None = None,
    ) -> tuple[xr.DataArray, ToolProvenanceSpec | None] | None:
        operations = self._preserved_assignments(target)
        if not operations:
            return None
        local_spec = full_data(*operations)
        processed = local_spec.apply(source_data).rename(source_data.name)
        erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(processed)
        node = self.manager._node_for_target(target)
        watched_input = getattr(node, "_watched_input_provenance_spec", None)
        base_spec = (
            watched_input(source_input_dtype=source_input_dtype)
            if callable(watched_input)
            else None
        )
        provenance = (
            compose_full_provenance(base_spec, local_spec)
            if base_spec is not None
            else local_spec
        )
        return processed, provenance

    def commit_replacement(
        self,
        target: int | str,
        source_data: xr.DataArray,
        processed: xr.DataArray,
        provenance: ToolProvenanceSpec | None,
    ) -> None:
        node = self.manager._node_for_target(target)
        if node.has_source_binding:
            self.manager.get_imagetool(target).slicer_area.replace_source_data(
                processed
            )
            return
        node.replace_with_detached_data(
            processed,
            provenance,
            replay_source_data=source_data,
        )
