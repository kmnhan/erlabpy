"""Discover and restore editors for provenance operations."""

from __future__ import annotations

import ast
import dataclasses
import typing

from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.utils
from erlab.interactive.imagetool import _kspace_conversion, dialogs
from erlab.interactive.imagetool._provenance._operations import (
    DivideByCoordOperation,
    GaussianFilterOperation,
    NormalizeOperation,
    ScriptCodeOperation,
    SortByOperation,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from erlab.interactive.imagetool._provenance._model import (
        ToolProvenanceOperation,
        ToolProvenanceSpec,
        _ProvenanceStepRef,
    )


@dataclasses.dataclass(frozen=True)
class _OperationDialogMatch:
    dialog_cls: type[dialogs._DataManipulationDialog]
    start: int
    stop: int
    focus: str | None = None


_NATIVE_TERMINAL_CURRENT_DATA_EDITORS: tuple[
    tuple[
        type[dialogs._DataManipulationDialog],
        type[ToolProvenanceOperation],
    ],
    ...,
] = (
    (
        dialogs.NormalizeDialog,
        NormalizeOperation,
    ),
    (
        dialogs.GaussianFilterDialog,
        GaussianFilterOperation,
    ),
    (
        dialogs.DivideByCoordDialog,
        DivideByCoordOperation,
    ),
    (
        dialogs.SortByDialog,
        SortByOperation,
    ),
)


class _ScriptCodeEditDialog(QtWidgets.QDialog):
    def __init__(
        self,
        operation: ScriptCodeOperation,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("managerProvenanceScriptCodeEditDialog")
        self.setWindowTitle("Edit Python Code")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)

        self.code_edit = erlab.interactive.utils.PythonCodeEditor(self)
        self.code_edit.setObjectName("managerProvenanceScriptCodeEditor")
        self.code_edit.setPlainText(operation.code or "")
        self.code_edit.setPlaceholderText("# Write Python code here")
        self.code_edit.setMinimumHeight(260)
        self.code_edit.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.code_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.code_edit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.code_edit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        layout.addWidget(self.code_edit)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.setObjectName("managerProvenanceScriptCodeEditButtonBox")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def code(self) -> str:
        return self.code_edit.toPlainText()

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            ast.parse(self.code(), mode="exec")
        except SyntaxError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Python Code",
                f"Python code is not valid: {exc}",
            )
            return
        super().accept()


def _iter_dialog_classes(
    cls: type[dialogs._DataManipulationDialog],
) -> Iterator[type[dialogs._DataManipulationDialog]]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _iter_dialog_classes(subclass)


def _iter_imagetool_dialog_classes(
    cls: type[dialogs._DataManipulationDialog],
) -> Iterator[type[dialogs._DataManipulationDialog]]:
    for dialog_cls in _iter_dialog_classes(cls):
        if dialog_cls.__module__ == dialogs.__name__:
            yield dialog_cls


def _dialog_declares_operation_type(
    dialog_cls: type[dialogs._DataManipulationDialog],
    operation_type: type[ToolProvenanceOperation],
) -> bool:
    return any(
        issubclass(operation_type, declared_type)
        for declared_type in dialog_cls.operation_types
    )


def _operation_matches_dialog(
    operation: ToolProvenanceOperation,
    dialog_cls: type[dialogs._DataManipulationDialog],
) -> bool:
    return any(
        isinstance(operation, operation_type)
        for operation_type in dialog_cls.operation_types
    )


def _dialog_supports_transform_restore(
    dialog_cls: type[dialogs.DataTransformDialog],
) -> bool:
    return (
        dialog_cls.restore_transform_operation
        is not dialogs.DataTransformDialog.restore_transform_operation
        or dialog_cls.restore_transform_operations
        is not dialogs.DataTransformDialog.restore_transform_operations
    )


def _dialog_supports_filter_restore(
    dialog_cls: type[dialogs.DataFilterDialog],
) -> bool:
    return (
        dialog_cls.restore_filter_operation
        is not dialogs.DataFilterDialog.restore_filter_operation
    )


def _dialog_is_group_editor(dialog_cls: type[dialogs.DataTransformDialog]) -> bool:
    return bool(dialog_cls.grouped_operation_only or dialog_cls.operation_group_kind)


def _dialog_overrides_operation_group_for_edit(
    dialog_cls: type[dialogs.DataTransformDialog],
) -> bool:
    return (
        typing.cast("typing.Any", dialog_cls.operation_group_for_edit).__func__
        is not typing.cast(
            "typing.Any", dialogs.DataTransformDialog.operation_group_for_edit
        ).__func__
    )


def _standalone_editor_dialog_classes_for_operation_type(
    operation_type: type[ToolProvenanceOperation],
) -> tuple[type[dialogs._DataManipulationDialog], ...]:
    matches: list[type[dialogs._DataManipulationDialog]] = []
    for dialog_base_cls in _iter_imagetool_dialog_classes(dialogs.DataTransformDialog):
        dialog_cls = typing.cast("type[dialogs.DataTransformDialog]", dialog_base_cls)
        if not _dialog_declares_operation_type(dialog_cls, operation_type):
            continue
        if not _dialog_supports_transform_restore(dialog_cls):
            continue
        if _dialog_is_group_editor(dialog_cls):
            continue
        matches.append(dialog_cls)
    for dialog_base_cls in _iter_imagetool_dialog_classes(dialogs.DataFilterDialog):
        dialog_cls = typing.cast("type[dialogs.DataFilterDialog]", dialog_base_cls)
        if not _dialog_declares_operation_type(dialog_cls, operation_type):
            continue
        if not _dialog_supports_filter_restore(dialog_cls):
            continue
        matches.append(dialog_cls)
    return tuple(matches)


def _grouped_editor_dialog_classes_for_operation_type(
    operation_type: type[ToolProvenanceOperation],
) -> tuple[type[dialogs.DataTransformDialog], ...]:
    matches: list[type[dialogs.DataTransformDialog]] = []
    for dialog_base_cls in _iter_imagetool_dialog_classes(dialogs.DataTransformDialog):
        dialog_cls = typing.cast("type[dialogs.DataTransformDialog]", dialog_base_cls)
        if not _dialog_declares_operation_type(dialog_cls, operation_type):
            continue
        if not _dialog_supports_transform_restore(dialog_cls):
            continue
        if _dialog_is_group_editor(
            dialog_cls
        ) and _dialog_overrides_operation_group_for_edit(dialog_cls):
            matches.append(dialog_cls)
    return tuple(matches)


def _standalone_editor_dialog_class_for_operation_type(
    operation_type: type[ToolProvenanceOperation],
) -> type[dialogs._DataManipulationDialog] | None:
    matches = _standalone_editor_dialog_classes_for_operation_type(operation_type)
    if len(matches) > 1:
        names = ", ".join(dialog_cls.__name__ for dialog_cls in matches)
        raise RuntimeError(
            f"Multiple standalone provenance editors handle "
            f"{operation_type.__name__}: {names}"
        )
    return matches[0] if matches else None


def _operation_editor_contract_errors() -> list[str]:
    operation_types: set[type[ToolProvenanceOperation]] = set()
    for base_cls in (dialogs.DataTransformDialog, dialogs.DataFilterDialog):
        for dialog_cls in _iter_imagetool_dialog_classes(base_cls):
            operation_types.update(dialog_cls.operation_types)

    errors: list[str] = []
    for operation_type in sorted(operation_types, key=lambda cls: cls.__name__):
        standalone = _standalone_editor_dialog_classes_for_operation_type(
            operation_type
        )
        grouped = _grouped_editor_dialog_classes_for_operation_type(operation_type)
        broken_grouped: list[type[dialogs.DataTransformDialog]] = []
        for dialog_base_cls in _iter_imagetool_dialog_classes(
            dialogs.DataTransformDialog
        ):
            dialog_cls = typing.cast(
                "type[dialogs.DataTransformDialog]", dialog_base_cls
            )
            if not _dialog_declares_operation_type(dialog_cls, operation_type):
                continue
            if not _dialog_supports_transform_restore(dialog_cls):
                continue
            if _dialog_is_group_editor(
                dialog_cls
            ) and not _dialog_overrides_operation_group_for_edit(dialog_cls):
                broken_grouped.append(dialog_cls)
        if len(standalone) > 1:
            names = ", ".join(dialog_cls.__name__ for dialog_cls in standalone)
            errors.append(
                f"{operation_type.__name__} has multiple standalone editors: {names}"
            )
        if len(grouped) > 1:
            names = ", ".join(dialog_cls.__name__ for dialog_cls in grouped)
            errors.append(
                f"{operation_type.__name__} has multiple grouped editors: {names}"
            )
        if standalone and grouped:
            standalone_names = ", ".join(
                dialog_cls.__name__ for dialog_cls in standalone
            )
            grouped_names = ", ".join(dialog_cls.__name__ for dialog_cls in grouped)
            errors.append(
                f"{operation_type.__name__} has both standalone and grouped editors: "
                f"{standalone_names}; {grouped_names}"
            )
        errors.extend(
            f"{dialog_cls.__name__} declares {operation_type.__name__} as a "
            "grouped editor but does not override operation_group_for_edit"
            for dialog_cls in broken_grouped
        )
        if not standalone and not grouped and not broken_grouped:
            errors.append(f"{operation_type.__name__} has no provenance editor")
    return errors


def _validate_operation_editor_contract() -> None:
    errors = _operation_editor_contract_errors()
    if errors:
        raise RuntimeError(
            "Invalid ImageTool provenance editor contract:\n- " + "\n- ".join(errors)
        )


def _dialog_class_for_operation(
    operation: ToolProvenanceOperation,
) -> type[dialogs._DataManipulationDialog] | None:
    _validate_operation_editor_contract()
    return _standalone_editor_dialog_class_for_operation_type(type(operation))


def _dialog_match_for_operation_ref(
    spec: ToolProvenanceSpec,
    ref: _ProvenanceStepRef,
) -> _OperationDialogMatch | None:
    _validate_operation_editor_contract()
    operation = spec._operation_for_ref(ref)
    if operation is None or ref.operation_index is None:
        return None
    operations = spec.operations
    if not operations:
        return None

    for dialog_base_cls in _iter_imagetool_dialog_classes(dialogs.DataTransformDialog):
        dialog_cls = typing.cast("type[dialogs.DataTransformDialog]", dialog_base_cls)
        if not _operation_matches_dialog(operation, dialog_cls):
            continue
        if not _dialog_supports_transform_restore(dialog_cls):
            continue
        group_kind = dialog_cls.operation_group_kind
        if group_kind is not None:
            marker = operation.group
            if marker is None or marker.kind != group_kind:
                continue
            group = dialog_cls.operation_group_for_edit(operations, ref.operation_index)
            if group is not None:
                return _OperationDialogMatch(
                    dialog_cls,
                    group[0],
                    group[1],
                    marker.focus,
                )
            continue
        group = dialog_cls.operation_group_for_edit(operations, ref.operation_index)
        if group is not None:
            return _OperationDialogMatch(dialog_cls, group[0], group[1])
        if dialog_cls.grouped_operation_only:
            continue
        return _OperationDialogMatch(
            dialog_cls,
            ref.operation_index,
            ref.operation_index + 1,
        )

    for dialog_base_cls in _iter_imagetool_dialog_classes(dialogs.DataFilterDialog):
        dialog_cls = typing.cast("type[dialogs.DataFilterDialog]", dialog_base_cls)
        if not _operation_matches_dialog(operation, dialog_cls):
            continue
        if not _dialog_supports_filter_restore(dialog_cls):
            continue
        return _OperationDialogMatch(
            dialog_cls,
            ref.operation_index,
            ref.operation_index + 1,
        )
    return None


def _editable_group_range_for_ref(
    spec: ToolProvenanceSpec,
    ref: _ProvenanceStepRef,
) -> tuple[int, int] | None:
    match = _dialog_match_for_operation_ref(spec, ref)
    if match is None or match.stop - match.start <= 1:
        return None
    return match.start, match.stop


def _uneditable_operation_reason(
    operation: ToolProvenanceOperation,
) -> str | None:
    return _kspace_conversion.incomplete_kspace_conversion_edit_reason(operation)
