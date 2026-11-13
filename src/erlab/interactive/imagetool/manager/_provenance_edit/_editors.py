"""Discover and restore editors for provenance operations."""

from __future__ import annotations

import ast
import dataclasses
import typing
import weakref

from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive.utils
from erlab.interactive.imagetool import _kspace_conversion, dialogs
from erlab.interactive.imagetool._provenance._operations import (
    ConcatOperation,
    DivideByCoordOperation,
    GaussianFilterOperation,
    NormalizeOperation,
    ScriptCodeOperation,
    SortByOperation,
)
from erlab.interactive.imagetool.manager._dialogs import _ConcatDialog

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from erlab.interactive.imagetool._provenance._model import (
        ToolProvenanceOperation,
        ToolProvenanceSpec,
        _ProvenanceStepRef,
    )
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager


@dataclasses.dataclass(frozen=True)
class _OperationDialogMatch:
    dialog_cls: type[dialogs._DataManipulationDialog]
    start: int
    stop: int
    focus: str | None = None


@dataclasses.dataclass(frozen=True)
class _OperationEditorContractState:
    dialog_cls: type[dialogs._DataManipulationDialog]
    operation_types: tuple[type[ToolProvenanceOperation], ...]
    kind: typing.Literal["transform", "filter"]
    supports_restore: bool
    grouped: bool
    overrides_group_matcher: bool


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


class _ConcatOperationEditDialog(_ConcatDialog):
    def __init__(
        self,
        operation: ConcatOperation,
        parent: ImageToolManager,
    ) -> None:
        super().__init__(parent)
        self._operation = operation
        self.setObjectName("managerProvenanceConcatEditDialog")
        self.setWindowTitle("Edit Concatenate")
        self._dim_line.setText(operation.dim)
        self._coords_combo.setCurrentText(operation.coords)
        self._compat_combo.setCurrentText(operation.compat)
        self._join_combo.setCurrentText(operation.join)
        self._combine_attrs_combo.setCurrentText(operation.combine_attrs)
        for widget in (
            self._result_combo,
            self._replace_target_combo,
            self._sources_combo,
        ):
            self._option_layout.setRowVisible(widget, False)

    def operation(self) -> ConcatOperation:
        return ConcatOperation(
            input_count=self._operation.input_count,
            group=self._operation.group,
            **self.concat_kwargs(),
        )

    def open(self) -> None:
        QtWidgets.QDialog.open(self)

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            self.operation()
        except (TypeError, ValueError):
            reason = (
                "Enter a dimension name."
                if not self._dim_line.text()
                else "coords='different' cannot be used with compat='override'."
            )
            QtWidgets.QMessageBox.warning(self, "Invalid Concatenate Options", reason)
            return
        QtWidgets.QDialog.accept(self)


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


def _operation_editor_contract_states() -> tuple[_OperationEditorContractState, ...]:
    states: list[_OperationEditorContractState] = []
    for dialog_base_cls in _iter_imagetool_dialog_classes(dialogs.DataTransformDialog):
        dialog_cls = typing.cast("type[dialogs.DataTransformDialog]", dialog_base_cls)
        states.append(
            _OperationEditorContractState(
                dialog_cls=dialog_cls,
                operation_types=tuple(dialog_cls.operation_types),
                kind="transform",
                supports_restore=_dialog_supports_transform_restore(dialog_cls),
                grouped=_dialog_is_group_editor(dialog_cls),
                overrides_group_matcher=(
                    _dialog_overrides_operation_group_for_edit(dialog_cls)
                ),
            )
        )
    for dialog_base_cls in _iter_imagetool_dialog_classes(dialogs.DataFilterDialog):
        dialog_cls = typing.cast("type[dialogs.DataFilterDialog]", dialog_base_cls)
        states.append(
            _OperationEditorContractState(
                dialog_cls=dialog_cls,
                operation_types=tuple(dialog_cls.operation_types),
                kind="filter",
                supports_restore=_dialog_supports_filter_restore(dialog_cls),
                grouped=False,
                overrides_group_matcher=False,
            )
        )
    return tuple(states)


def _operation_editor_contract_fingerprint(
    states: tuple[_OperationEditorContractState, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            weakref.ref(state.dialog_cls),
            state.dialog_cls.__module__,
            state.dialog_cls.__name__,
            state.dialog_cls.__qualname__,
            tuple(
                (
                    weakref.ref(operation_type),
                    operation_type.__module__,
                    operation_type.__name__,
                    operation_type.__qualname__,
                )
                for operation_type in state.operation_types
            ),
            state.kind,
            state.supports_restore,
            state.grouped,
            state.overrides_group_matcher,
        )
        for state in states
    )


# Keep only weak class identities so runtime dialog classes remain collectible.
_operation_editor_contract_cache: (
    tuple[tuple[tuple[object, ...], ...], tuple[str, ...]] | None
) = None


def _operation_editor_contract_errors() -> list[str]:
    global _operation_editor_contract_cache

    states = _operation_editor_contract_states()
    fingerprint = _operation_editor_contract_fingerprint(states)
    cached = _operation_editor_contract_cache
    if cached is None or cached[0] != fingerprint:
        errors = tuple(_compute_operation_editor_contract_errors(states))
        _operation_editor_contract_cache = (fingerprint, errors)
    else:
        errors = cached[1]
    return list(errors)


def _compute_operation_editor_contract_errors(
    states: tuple[_OperationEditorContractState, ...],
) -> list[str]:
    operation_types = {
        operation_type for state in states for operation_type in state.operation_types
    }

    errors: list[str] = []
    for operation_type in sorted(operation_types, key=lambda cls: cls.__name__):
        matching = [
            state
            for state in states
            if any(
                issubclass(operation_type, declared_type)
                for declared_type in state.operation_types
            )
            and state.supports_restore
        ]
        standalone = tuple(state.dialog_cls for state in matching if not state.grouped)
        grouped = tuple(
            state.dialog_cls
            for state in matching
            if state.grouped and state.overrides_group_matcher
        )
        broken_grouped = tuple(
            state.dialog_cls
            for state in matching
            if state.kind == "transform"
            and state.grouped
            and not state.overrides_group_matcher
        )
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
    if ref.operation_index is None or spec._operation_for_ref(ref) is None:
        return None
    return _dialog_match_for_operations(spec.operations, ref.operation_index)


def _dialog_match_for_operations(
    operations: Sequence[ToolProvenanceOperation],
    index: int,
) -> _OperationDialogMatch | None:
    operation = operations[index]
    standalone = _dialog_class_for_operation(operation)
    marker = operation.group
    for candidate in _iter_imagetool_dialog_classes(dialogs.DataTransformDialog):
        dialog_cls = typing.cast("type[dialogs.DataTransformDialog]", candidate)
        if not _dialog_overrides_operation_group_for_edit(
            dialog_cls
        ) or not _dialog_supports_transform_restore(dialog_cls):
            continue
        if marker is not None:
            # Shared setup operations such as coordinate assignment also have
            # standalone editors. The group marker takes precedence.
            if dialog_cls.operation_group_kind != marker.kind:
                continue
        elif (
            dialog_cls.operation_group_kind is not None
            or not _operation_matches_dialog(operation, dialog_cls)
        ):
            continue
        group = dialog_cls.operation_group_for_edit(operations, index)
        if group is not None:
            return _OperationDialogMatch(
                dialog_cls, *group, marker.focus if marker else None
            )
    if marker is not None or standalone is None:
        return None
    return _OperationDialogMatch(standalone, index, index + 1)


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
