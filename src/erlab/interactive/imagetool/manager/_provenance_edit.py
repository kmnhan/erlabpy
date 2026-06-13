from __future__ import annotations

import ast
import pathlib
import typing

from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive.imagetool import dialogs, provenance
from erlab.interactive.imagetool._load_source import _file_load_provenance_from_source

if typing.TYPE_CHECKING:
    import xarray as xr

    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )


def _parse_loader_kwargs(text: str) -> dict[str, typing.Any]:
    text = text.strip()
    if not text or text == "(none)":
        return {}
    if text.startswith("{"):
        value = ast.literal_eval(text)
        if not isinstance(value, dict):
            raise TypeError("Loader kwargs must be a dictionary")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Loader kwargs must use string keys")
        return dict(value)

    call = ast.parse(f"loader({text})", mode="eval").body
    if not isinstance(call, ast.Call) or call.args:
        raise TypeError("Use keyword arguments such as `key=value`")
    kwargs: dict[str, typing.Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise TypeError("Loader kwargs do not support ** unpacking")
        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    return kwargs


class _FileLoadEditDialog(QtWidgets.QDialog):
    def __init__(
        self,
        load_source: provenance.FileLoadSource,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("managerProvenanceFileLoadEditDialog")
        self.setWindowTitle("Edit File Load")
        self.setModal(True)

        layout = QtWidgets.QFormLayout(self)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.path_edit = QtWidgets.QLineEdit(str(load_source.path), self)
        self.path_edit.setObjectName("managerProvenanceFilePathEdit")
        browse_button = QtWidgets.QToolButton(self)
        browse_button.setObjectName("managerProvenanceFileBrowseButton")
        browse_button.setText("…")
        browse_button.clicked.connect(self._browse)

        path_widget = QtWidgets.QWidget(self)
        path_layout = QtWidgets.QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse_button)
        layout.addRow("File", path_widget)

        self.loader_label = QtWidgets.QLineEdit(load_source.loader_text, self)
        self.loader_label.setObjectName("managerProvenanceLoaderEdit")
        self.loader_label.setReadOnly(True)
        layout.addRow(load_source.loader_label, self.loader_label)

        replay_call = load_source.replay_call
        kwargs_text = load_source.kwargs_text
        if replay_call is not None and replay_call.kwargs:
            kwargs_text = erlab.interactive.utils.format_call_kwargs(
                typing.cast(
                    "dict[typing.Hashable, typing.Any]",
                    dict(replay_call.kwargs),
                )
            )
        elif kwargs_text == "(none)":
            kwargs_text = ""
        self.kwargs_edit = QtWidgets.QPlainTextEdit(kwargs_text, self)
        self.kwargs_edit.setObjectName("managerProvenanceLoaderKwargsEdit")
        line_height = self.kwargs_edit.fontMetrics().lineSpacing()
        frame_width = 2 * self.kwargs_edit.frameWidth()
        self.kwargs_edit.setMaximumHeight(line_height * 8 + frame_width + 8)
        layout.addRow("Loader Arguments", self.kwargs_edit)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.setObjectName("managerProvenanceFileEditButtonBox")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def file_path(self) -> pathlib.Path:
        return pathlib.Path(self.path_edit.text()).expanduser()

    def loader_kwargs(self) -> dict[str, typing.Any]:
        return _parse_loader_kwargs(self.kwargs_edit.toPlainText())

    @QtCore.Slot()
    def _browse(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose File",
            str(self.file_path().parent),
        )
        if path:
            self.path_edit.setText(path)

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            self.loader_kwargs()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Loader Arguments",
                str(exc),
            )
            return
        super().accept()


def _iter_dialog_classes(
    cls: type[dialogs._DataManipulationDialog],
) -> typing.Iterator[type[dialogs._DataManipulationDialog]]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _iter_dialog_classes(subclass)


def _dialog_class_for_operation(
    operation: provenance.ToolProvenanceOperation,
) -> type[dialogs._DataManipulationDialog] | None:
    for dialog_cls in (
        *_iter_dialog_classes(dialogs.DataTransformDialog),
        *_iter_dialog_classes(dialogs.DataFilterDialog),
    ):
        if not any(
            isinstance(operation, operation_type)
            for operation_type in dialog_cls.operation_types
        ):
            continue
        if (
            issubclass(dialog_cls, dialogs.DataTransformDialog)
            and dialog_cls.restore_transform_operation
            is dialogs.DataTransformDialog.restore_transform_operation
        ):
            continue
        if (
            issubclass(dialog_cls, dialogs.DataFilterDialog)
            and dialog_cls.restore_filter_operation
            is dialogs.DataFilterDialog.restore_filter_operation
        ):
            continue
        return dialog_cls
    return None


class _ProvenanceEditController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager

    def can_edit_row(
        self,
        row: provenance._ProvenanceDisplayRow | None,
    ) -> tuple[bool, str]:
        node = self._metadata_node()
        if row is None or row.edit_ref is None:
            return False, "This row does not support editing."
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to edit provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        if self._source_child_parent_row(node, row):
            return False, "Edit the parent ImageTool row directly."

        spec = self._display_spec_for_row(node, row)
        if spec is None:
            return False, "This row does not have replayable provenance."
        if spec.kind == "script":
            return False, "Script provenance rows are not editable in this version."
        if row.edit_ref.kind == "file_load":
            if spec.kind != "file" or spec.file_load_source is None:
                return False, "This row is not a file load step."
            if spec.file_load_source.replay_call is None:
                return False, "This file load step cannot be replayed."
            return True, ""
        operation = spec._operation_for_ref(row.edit_ref)
        if operation is None:
            return False, "This operation is not available."
        if isinstance(operation, provenance.ScriptCodeOperation):
            return False, "Free-form script steps are not editable."
        if (
            spec.kind in {"full_data", "public_data", "selection"}
            and row.scope != "source"
            and self._active_filter_ref(node, spec) != row.edit_ref
        ):
            return False, "This live row needs a parent source to replay."
        if _dialog_class_for_operation(operation) is None:
            return False, "No editing dialog is available for this step."
        return True, ""

    def can_revert_row(
        self,
        row: provenance._ProvenanceDisplayRow | None,
    ) -> tuple[bool, str]:
        node = self._metadata_node()
        if row is None or row.replay_ref is None:
            return False, "This row is not replayable."
        if not self._node_editable(node):
            return False, "Select an available ImageTool row to revert provenance."
        node = typing.cast("_ImageToolWrapper | _ManagedWindowNode", node)
        if self._source_child_parent_row(node, row):
            return False, "Revert the parent ImageTool row directly."
        if row.replay_ref.kind == "script_input":
            return False, "Dependency input rows are not revert targets."
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            return False, "This row does not have replayable provenance."
        if spec.kind == "script":
            try:
                candidate = spec._prefix_through_ref(row.replay_ref)
            except ValueError:
                return False, "This script row is not a replayable step."
            if not provenance.script_provenance_replayable(candidate):
                return False, "This script step cannot be replayed automatically."
            if not all(
                self._manager._script_input_can_reload(
                    script_input,
                    target_node_uid=node.uid,
                )
                for script_input in candidate.script_inputs
            ):
                return False, "This script step has unavailable inputs."
            return True, ""
        if spec.kind in {"full_data", "public_data", "selection"}:
            if row.scope == "source" and node.parent_uid is not None:
                return True, ""
            return False, "This live row needs a parent source to replay."
        return True, ""

    def edit_row(self, row: provenance._ProvenanceDisplayRow | None) -> None:
        editable, reason = self.can_edit_row(row)
        if not editable:
            self._show_unavailable(reason)
            return
        node = typing.cast(
            "_ImageToolWrapper | _ManagedWindowNode",
            self._metadata_node(),
        )
        row = typing.cast("provenance._ProvenanceDisplayRow", row)
        ref = typing.cast("provenance._ProvenanceStepRef", row.edit_ref)
        try:
            if ref.kind == "file_load":
                self._edit_file_load_row(node, row)
            else:
                self._edit_operation_row(node, row)
        except Exception as exc:
            self._show_failed("Edit Step Failed", exc)

    def revert_row(self, row: provenance._ProvenanceDisplayRow | None) -> None:
        revertible, reason = self.can_revert_row(row)
        if not revertible:
            self._show_unavailable(reason)
            return
        if not self._confirm_revert():
            return

        node = typing.cast(
            "_ImageToolWrapper | _ManagedWindowNode",
            self._metadata_node(),
        )
        row = typing.cast("provenance._ProvenanceDisplayRow", row)
        ref = typing.cast("provenance._ProvenanceStepRef", row.replay_ref)
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            self._show_failed(
                "Revert Step Failed",
                RuntimeError("No provenance spec is available"),
            )
            return
        try:
            candidate = spec._prefix_through_ref(ref)
            self._validate_and_replace(node, row.scope, candidate)
        except Exception as exc:
            self._show_failed("Revert Step Failed", exc)

    def _metadata_node(self) -> _ImageToolWrapper | _ManagedWindowNode | None:
        uid = self._manager._metadata_node_uid
        if uid is None:
            return None
        return self._manager._tool_graph.nodes.get(uid)

    @staticmethod
    def _node_editable(
        node: _ImageToolWrapper | _ManagedWindowNode | None,
    ) -> bool:
        return node is not None and node.is_imagetool and node.imagetool is not None

    @staticmethod
    def _source_child_parent_row(
        node: _ImageToolWrapper | _ManagedWindowNode | None,
        row: provenance._ProvenanceDisplayRow,
    ) -> bool:
        return (
            node is not None
            and node.parent_uid is not None
            and node.source_spec is not None
            and row.scope == "display"
        )

    def _display_spec_for_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
    ) -> provenance.ToolProvenanceSpec | None:
        if row.scope == "source":
            return node.displayed_source_spec
        return node.displayed_provenance_spec

    def _edit_file_load_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
    ) -> None:
        spec = self._display_spec_for_row(node, row)
        if spec is None or spec.kind != "file" or spec.file_load_source is None:
            raise RuntimeError("Selected row is not a file load step")
        dialog = _FileLoadEditDialog(spec.file_load_source, self._manager)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        candidate = _file_load_provenance_from_source(
            dialog.file_path(),
            spec.file_load_source,
            kwargs=dialog.loader_kwargs(),
            active_name=spec.active_name or "derived",
            replay_stages=spec.replay_stages,
        )
        self._validate_and_replace(node, row.scope, candidate)

    def _edit_operation_row(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        row: provenance._ProvenanceDisplayRow,
    ) -> None:
        ref = typing.cast("provenance._ProvenanceStepRef", row.edit_ref)
        spec = self._display_spec_for_row(node, row)
        if spec is None:
            raise RuntimeError("No provenance spec is available")
        operation = spec._operation_for_ref(ref)
        if operation is None:
            raise RuntimeError("Selected operation is not available")
        dialog_cls = _dialog_class_for_operation(operation)
        if dialog_cls is None:
            raise RuntimeError("No editing dialog is available for this step")
        if self._active_filter_ref(node, spec) == ref:
            self._edit_active_filter(node, operation, dialog_cls)
            return

        input_spec = spec._prefix_before_ref(ref)
        input_data = self._replay_candidate(node, row.scope, input_spec)
        replacements = self._edited_operations_from_dialog(
            dialog_cls,
            operation,
            input_data,
        )
        if replacements is None:
            return
        candidate = spec._replace_operation_ref(ref, replacements)
        self._validate_and_replace(node, row.scope, candidate)

    def _edited_operations_from_dialog(
        self,
        dialog_cls: type[dialogs._DataManipulationDialog],
        operation: provenance.ToolProvenanceOperation,
        input_data: xr.DataArray,
    ) -> list[provenance.ToolProvenanceOperation] | None:
        tool = typing.cast(
            "ImageTool | None",
            erlab.interactive.itool(input_data, manager=False, execute=False),
        )
        if tool is None:
            raise RuntimeError("Could not create a temporary ImageTool")
        tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        tool.hide()
        try:
            dialog = dialog_cls(tool.slicer_area)
            if isinstance(dialog, dialogs.DataFilterDialog):
                dialog.restore_filter_operation(operation)
                if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
                    return None
                new_operation = dialog.filter_operation()
                return [] if new_operation is None else [new_operation]

            transform_dialog = typing.cast("dialogs.DataTransformDialog", dialog)
            transform_dialog.restore_transform_operation(operation)
            replace_index = transform_dialog.launch_mode_combo.findData(
                "replace",
                QtCore.Qt.ItemDataRole.UserRole,
            )
            if replace_index >= 0:
                transform_dialog.launch_mode_combo.setCurrentIndex(replace_index)
            if transform_dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
                return None
            return transform_dialog.source_operations()
        finally:
            tool.close()
            tool.deleteLater()

    def _edit_active_filter(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        operation: provenance.ToolProvenanceOperation,
        dialog_cls: type[dialogs._DataManipulationDialog],
    ) -> None:
        if node.imagetool is None or not issubclass(
            dialog_cls,
            dialogs.DataFilterDialog,
        ):
            raise RuntimeError("Active display filter is not editable")
        dialog = dialog_cls(node.slicer_area)
        dialog.restore_filter_operation(operation)
        if dialog.exec() == int(QtWidgets.QDialog.DialogCode.Accepted):
            self._manager._update_info(uid=node.uid)

    def _validate_and_replace(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        candidate: provenance.ToolProvenanceSpec,
    ) -> None:
        candidate_data, candidate = self._replay_candidate_result(
            node,
            scope,
            candidate,
        )
        base_candidate, filter_operation = self._split_active_filter(node, candidate)
        if base_candidate == candidate:
            source_data = candidate_data
        else:
            source_data, base_candidate = self._replay_candidate_result(
                node,
                scope,
                base_candidate,
            )
        self._replace_node_data(
            node,
            scope,
            source_data,
            base_candidate,
            filter_operation,
        )
        self._manager._update_info(uid=node.uid)

    def _replay_candidate(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
    ) -> xr.DataArray:
        data, _spec = self._replay_candidate_result(node, scope, spec)
        return data

    def _replay_candidate_result(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        spec: provenance.ToolProvenanceSpec,
    ) -> tuple[xr.DataArray, provenance.ToolProvenanceSpec]:
        if spec.kind == "file":
            return provenance.replay_file_provenance(spec), spec
        if spec.kind in {"full_data", "public_data", "selection"}:
            if scope != "source" or node.parent_uid is None:
                raise RuntimeError("Live provenance needs a parent source to replay")
            parent = self._manager._parent_node(node)
            return spec.apply(parent.current_source_data()), spec
        if spec.kind == "script":
            result = self._manager._rebuild_script_provenance(
                spec,
                target_node_uid=node.uid,
            )
            return result.data, result.provenance_spec
        raise RuntimeError("Unsupported provenance kind")

    def _replace_node_data(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        scope: typing.Literal["display", "source"],
        data: xr.DataArray,
        spec: provenance.ToolProvenanceSpec,
        filter_operation: provenance.ToolProvenanceOperation | None,
    ) -> None:
        preserve_filter = filter_operation is not None
        if scope == "source":
            live_spec = provenance.require_live_source_spec(spec)
            if live_spec is None or node.parent_uid is None:
                raise RuntimeError("Source-bound edits need live source provenance")
            parent = self._manager._parent_node(node)
            parent_data = parent.current_source_data()
            displayed = provenance.compose_display_provenance(
                parent.displayed_provenance_spec,
                live_spec,
                parent_data=parent_data,
            )
            node.set_source_binding(
                live_spec,
                auto_update=node.source_auto_update,
                state="fresh",
                provenance_spec=displayed,
            )
            node._replace_imagetool_data(
                data,
                displayed,
                propagate_descendants=True,
                preserve_filter=preserve_filter,
            )
            return
        node.replace_with_detached_data(
            data,
            spec,
            preserve_filter=preserve_filter,
        )

    def _active_filter_ref(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: provenance.ToolProvenanceSpec,
    ) -> provenance._ProvenanceStepRef | None:
        if node.imagetool is None:
            return None
        active = node.slicer_area._accepted_filter_provenance_operation
        if active is None:
            return None
        if spec.kind == "file":
            for stage_index in range(len(spec.replay_stages) - 1, -1, -1):
                operations = spec.replay_stages[stage_index].operations
                for operation_index in range(len(operations) - 1, -1, -1):
                    if operations[operation_index] == active:
                        return provenance._ProvenanceStepRef(
                            "operation",
                            operation_index=operation_index,
                            stage_index=stage_index,
                        )
        elif spec.kind in {"full_data", "public_data", "selection", "script"}:
            for operation_index in range(len(spec.operations) - 1, -1, -1):
                if spec.operations[operation_index] == active:
                    return provenance._ProvenanceStepRef(
                        "operation",
                        operation_index=operation_index,
                    )
        return None

    def _split_active_filter(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        spec: provenance.ToolProvenanceSpec,
    ) -> tuple[
        provenance.ToolProvenanceSpec,
        provenance.ToolProvenanceOperation | None,
    ]:
        active_ref = self._active_filter_ref(node, spec)
        if active_ref is None:
            return spec, None
        active_operation = spec._operation_for_ref(active_ref)
        if active_operation is None:
            return spec, None
        base_spec = spec._replace_operation_ref(active_ref, ())
        if base_spec.kind == "file":
            stages = tuple(
                stage for stage in base_spec.replay_stages if stage.operations
            )
            base_spec = base_spec.model_copy(update={"replay_stages": stages})
        return base_spec, active_operation

    def _confirm_revert(self) -> bool:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Revert Provenance")
        msg_box.setText("Revert to the selected provenance step?")
        msg_box.setInformativeText(
            "Later provenance steps for this ImageTool will be dropped."
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        return msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes

    def _show_unavailable(self, reason: str) -> None:
        QtWidgets.QMessageBox.information(
            self._manager,
            "Provenance Step Unavailable",
            reason,
        )

    def _show_failed(self, title: str, exc: Exception) -> None:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText("The selected ImageTool was not changed.")
        msg_box.setInformativeText(str(exc))
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()
