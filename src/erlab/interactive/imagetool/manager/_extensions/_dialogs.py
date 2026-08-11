"""Dialogs used by the ImageTool Manager extension controller."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtWidgets

from erlab.extensions import (
    LoaderDescriptor,
    ParameterDescriptor,
    ParameterKind,
    RoutineDescriptor,
)
from erlab.extensions._models import _require_finite_parameter_values
from erlab.interactive.imagetool.manager._extensions._models import (
    _ExtensionCatalogModel,
    _ExtensionMetadata,
    _ExtensionRecord,
    _ResolvedWorkspaceRequirement,
)

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping


class _SourceReviewDialog(QtWidgets.QDialog):
    """Review source and optional author metadata before code is approved."""

    def __init__(
        self,
        path: pathlib.Path | None,
        parent: QtWidgets.QWidget,
        *,
        source_text: str | None = None,
        choose_approval_scope: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("manager_extension_source_review_dialog")
        self.setWindowTitle("Review Extension Source")
        self.resize(760, 600)
        layout = QtWidgets.QVBoxLayout(self)
        source = QtWidgets.QPlainTextEdit(self)
        source.setObjectName("manager_extension_source_review")
        source.setReadOnly(True)
        if source_text is None:
            if path is None:
                raise ValueError("path or source_text is required")
            source_text = path.read_text(encoding="utf-8")
        source.setPlainText(source_text)
        layout.addWidget(source, 1)
        form = QtWidgets.QFormLayout()
        self.author_edit = QtWidgets.QLineEdit(self)
        self.contact_edit = QtWidgets.QLineEdit(self)
        self.project_url_edit = QtWidgets.QLineEdit(self)
        self.change_summary_edit = QtWidgets.QLineEdit(self)
        self.changelog_edit = QtWidgets.QPlainTextEdit(self)
        for editor, object_name in (
            (self.author_edit, "manager_extension_author"),
            (self.contact_edit, "manager_extension_contact"),
            (self.project_url_edit, "manager_extension_project_url"),
            (self.change_summary_edit, "manager_extension_change_summary"),
            (self.changelog_edit, "manager_extension_changelog"),
        ):
            editor.setObjectName(object_name)
        form.addRow("Author", self.author_edit)
        form.addRow("Contact", self.contact_edit)
        form.addRow("Project URL", self.project_url_edit)
        form.addRow("Change summary", self.change_summary_edit)
        form.addRow("Changelog", self.changelog_edit)
        layout.addLayout(form)
        self._remember_approval = QtWidgets.QCheckBox(
            "Remember this extension in the application catalog", self
        )
        self._remember_approval.setObjectName("manager_extension_remember_approval")
        self._remember_approval.setVisible(choose_approval_scope)
        self._remember_approval.setChecked(not choose_approval_scope)
        layout.addWidget(self._remember_approval)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def metadata(self) -> _ExtensionMetadata:
        return _ExtensionMetadata(
            author=self.author_edit.text().strip(),
            contact=self.contact_edit.text().strip(),
            project_url=self.project_url_edit.text().strip(),
            change_summary=self.change_summary_edit.text().strip(),
            changelog=self.changelog_edit.toPlainText().strip(),
        )

    @property
    def remember_approval(self) -> bool:
        """Return whether approval must persist beyond this manager session."""
        return self._remember_approval.isChecked()


class _ExtensionParameterDialog(QtWidgets.QDialog):
    """Build binding-neutral editors from a validated capability descriptor."""

    def __init__(
        self,
        descriptor: RoutineDescriptor | LoaderDescriptor,
        parent: QtWidgets.QWidget,
        values: Mapping[str, typing.Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.descriptor = descriptor
        self.setObjectName("manager_extension_parameter_dialog")
        self.setWindowTitle(descriptor.name)
        layout = QtWidgets.QVBoxLayout(self)
        if descriptor.summary:
            summary = QtWidgets.QLabel(descriptor.summary, self)
            summary.setWordWrap(True)
            layout.addWidget(summary)
        form = QtWidgets.QFormLayout()
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._none_controls: dict[str, QtWidgets.QCheckBox] = {}
        for parameter in descriptor.parameters:
            value = (
                values[parameter.id]
                if values is not None and parameter.id in values
                else parameter.default
            )
            editor = self._make_editor(parameter, value)
            editor.setObjectName(f"manager_extension_parameter_{parameter.id}")
            self._editors[parameter.id] = editor
            field: QtWidgets.QWidget = editor
            if parameter.kind is ParameterKind.STRING and parameter.optional:
                field = QtWidgets.QWidget(self)
                field_layout = QtWidgets.QHBoxLayout(field)
                field_layout.setContentsMargins(0, 0, 0, 0)
                field_layout.addWidget(editor)
                none_control = QtWidgets.QCheckBox("None", field)
                none_control.setObjectName(
                    f"manager_extension_parameter_{parameter.id}_none"
                )
                none_control.setChecked(value is None)
                editor.setDisabled(none_control.isChecked())
                none_control.toggled.connect(editor.setDisabled)
                field_layout.addWidget(none_control)
                self._none_controls[parameter.id] = none_control
            form.addRow(parameter.id.replace("_", " ").title(), field)
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_editor(
        self, parameter: ParameterDescriptor, value: typing.Any
    ) -> QtWidgets.QWidget:
        if parameter.kind is ParameterKind.BOOLEAN:
            if parameter.optional:
                editor = QtWidgets.QComboBox(self)
                editor.addItem("None", None)
                editor.addItem("True", True)
                editor.addItem("False", False)
                if value is not None:
                    editor.setCurrentIndex(editor.findData(value))
                return editor
            editor = QtWidgets.QCheckBox(self)
            editor.setChecked(bool(value))
            return editor
        if parameter.kind is ParameterKind.INTEGER:
            editor = QtWidgets.QLineEdit(self)
            if value is not None:
                editor.setText(str(value))
            return editor
        if parameter.kind is ParameterKind.NUMBER:
            editor = QtWidgets.QLineEdit(self)
            if value is not None:
                editor.setText(str(value))
            return editor
        if parameter.kind in {ParameterKind.LITERAL, ParameterKind.ENUM}:
            editor = QtWidgets.QComboBox(self)
            if parameter.optional:
                editor.addItem("None", None)
            for choice in parameter.choices:
                editor.addItem(str(choice), choice)
            if value is not None:
                start = 1 if parameter.optional else 0
                for index, choice in enumerate(parameter.choices, start=start):
                    if type(value) is type(choice) and value == choice:
                        editor.setCurrentIndex(index)
                        break
            return editor
        editor = QtWidgets.QLineEdit(self)
        if value is not None:
            editor.setText(str(value))
        if parameter.kind is ParameterKind.PATH:
            editor.setPlaceholderText("Path")
        return editor

    @property
    def parameters(self) -> dict[str, typing.Any]:
        values: dict[str, typing.Any] = {}
        by_id = {item.id: item for item in self.descriptor.parameters}
        for name, editor in self._editors.items():
            parameter = by_id[name]
            none_control = self._none_controls.get(name)
            value: typing.Any
            if none_control is not None and none_control.isChecked():
                value = None
            elif isinstance(editor, QtWidgets.QCheckBox):
                value = editor.isChecked()
            elif isinstance(editor, QtWidgets.QComboBox):
                value = editor.currentData()
            else:
                value = typing.cast("QtWidgets.QLineEdit", editor).text()
                if parameter.optional and not value and none_control is None:
                    value = None
                elif parameter.kind is ParameterKind.INTEGER:
                    value = int(value)
                elif parameter.kind is ParameterKind.NUMBER:
                    value = float(value)
                elif (
                    parameter.kind is ParameterKind.PATH
                    and parameter.required
                    and not value
                ):
                    raise ValueError(f"{name!r} requires a value")
            values[name] = value
        _require_finite_parameter_values(values)
        return values

    def accept(self) -> None:
        try:
            _parameters = self.parameters
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameter", str(error))
            return
        super().accept()


class _RoutineSelectionDialog(QtWidgets.QDialog):
    def __init__(
        self,
        routines: tuple[tuple[str, str, RoutineDescriptor], ...],
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("manager_extension_routine_selection_dialog")
        self.setWindowTitle("Run Routine")
        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setObjectName("manager_extension_routine_list")
        for extension_id, extension_name, descriptor in routines:
            item = QtWidgets.QListWidgetItem(
                f"{descriptor.category} — {descriptor.name} ({extension_name})"
            )
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                (extension_id, descriptor.id),
            )
            self.list_widget.addItem(item)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.accept())
        layout.addWidget(self.list_widget)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selection(self) -> tuple[str, str] | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return typing.cast(
            "tuple[str, str]", item.data(QtCore.Qt.ItemDataRole.UserRole)
        )


class _ManageExtensionsDialog(QtWidgets.QDialog):
    action_requested = QtCore.Signal(str, str)

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("manager_manage_extensions_dialog")
        self.setWindowTitle("Manage Extensions")
        self.resize(800, 460)
        layout = QtWidgets.QVBoxLayout(self)
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setObjectName("manager_extension_catalog_tree")
        self.tree.setHeaderLabels(("Extension", "State", "Revision", "Source"))
        layout.addWidget(self.tree, 1)
        buttons = QtWidgets.QHBoxLayout()
        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        self._records: dict[str, _ExtensionRecord] = {}
        for action_id, text in (
            ("reload", "Reload Script"),
            ("toggle", "Enable or Disable"),
            ("favorite", "Toggle Favorite"),
            ("embedding", "Change Embedding…"),
            ("metadata", "Edit Metadata…"),
            ("remove", "Remove"),
        ):
            button = QtWidgets.QPushButton(text, self)
            button.setObjectName(f"manager_extension_{action_id}_button")
            button.clicked.connect(
                lambda _checked=False, value=action_id: self._emit_action(value)
            )
            self._buttons[action_id] = button
            buttons.addWidget(button)
        self.tree.currentItemChanged.connect(self._update_buttons)
        layout.addLayout(buttons)
        close_button = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, parent=self
        )
        close_button.rejected.connect(self.reject)
        layout.addWidget(close_button)

    def set_catalog(
        self,
        catalog: _ExtensionCatalogModel,
        source_states: dict[tuple[str, str], str] | None = None,
    ) -> None:
        selected_id = self.selected_extension_id
        self._records = dict(catalog.extensions)
        source_states = source_states or {}
        self.tree.clear()
        selected_item: QtWidgets.QTreeWidgetItem | None = None
        for record in catalog.extensions.values():
            current = record.revisions[record.current_revision]
            state = (
                "Removed"
                if record.removed
                else "Enabled"
                if record.enabled
                else "Disabled"
            )
            if current.import_error:
                state = "Import failed"
            elif not current.approved:
                state = "Approval required"
            item = QtWidgets.QTreeWidgetItem(
                (
                    record.name,
                    state,
                    current.source_modified_at or "",
                    source_states.get(
                        (record.id, record.current_revision),
                        current.source_path or current.entry_point_value or "",
                    ),
                )
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, record.id)
            self.tree.addTopLevelItem(item)
            for revision in record.revisions.values():
                child = QtWidgets.QTreeWidgetItem(
                    (
                        "Revision",
                        "Approved" if revision.approved else "Not approved",
                        revision.source_modified_at or revision.created_at,
                        revision.source_hash,
                    )
                )
                item.addChild(child)
                for kind, descriptors in (
                    ("Routine", revision.routines),
                    ("Loader", revision.loaders),
                ):
                    for descriptor in descriptors:
                        child.addChild(
                            QtWidgets.QTreeWidgetItem(
                                (
                                    kind,
                                    descriptor.name,
                                    descriptor.id,
                                    descriptor.summary,
                                )
                            )
                        )
                if revision.import_error:
                    error_item = QtWidgets.QTreeWidgetItem(
                        (
                            "Import failure",
                            "",
                            "",
                            revision.import_error.splitlines()[-1],
                        )
                    )
                    error_item.setToolTip(3, revision.import_error)
                    child.addChild(error_item)
            metadata_values = [
                f"{name.replace('_', ' ').title()}: {value}"
                for name, value in record.metadata.model_dump().items()
                if value
            ]
            if metadata_values:
                item.addChild(
                    QtWidgets.QTreeWidgetItem(
                        ("Metadata", "", "", "\n".join(metadata_values))
                    )
                )
            if record.id == selected_id:
                selected_item = item
        if selected_item is not None:
            self.tree.setCurrentItem(selected_item)
        self._update_buttons()

    @QtCore.Slot()
    def _update_buttons(self) -> None:
        record = self._records.get(self.selected_extension_id or "")
        available = record is not None and not record.removed
        is_script = (
            record is not None and not record.removed and record.source_type == "script"
        )
        current_source_path = (
            None
            if record is None
            else record.revisions[record.current_revision].source_path
        )
        self._buttons["reload"].setEnabled(bool(is_script and current_source_path))
        self._buttons["toggle"].setEnabled(available)
        self._buttons["favorite"].setEnabled(available)
        self._buttons["embedding"].setEnabled(bool(is_script))
        self._buttons["metadata"].setEnabled(record is not None)
        self._buttons["remove"].setEnabled(record is not None)
        self._buttons["remove"].setText(
            "Restore" if record is not None and record.removed else "Remove"
        )

    @property
    def selected_extension_id(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        parent = item.parent()
        while parent is not None:
            item = parent
            parent = item.parent()
        value = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return value if isinstance(value, str) else None

    def _emit_action(self, action_id: str) -> None:
        extension_id = self.selected_extension_id
        if extension_id is not None:
            self.action_requested.emit(action_id, extension_id)


class _MetadataDialog(QtWidgets.QDialog):
    def __init__(self, metadata: _ExtensionMetadata, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("manager_extension_metadata_dialog")
        self.setWindowTitle("Extension Metadata")
        layout = QtWidgets.QFormLayout(self)
        self._edits: dict[str, QtWidgets.QLineEdit | QtWidgets.QPlainTextEdit] = {}
        for name, value in metadata.model_dump().items():
            editor: QtWidgets.QLineEdit | QtWidgets.QPlainTextEdit
            if name == "changelog":
                editor = QtWidgets.QPlainTextEdit(str(value), self)
            else:
                editor = QtWidgets.QLineEdit(str(value), self)
            editor.setObjectName(f"manager_extension_metadata_{name}")
            self._edits[name] = editor
            layout.addRow(name.replace("_", " ").title(), editor)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    @property
    def metadata(self) -> _ExtensionMetadata:
        values = {
            name: (
                editor.toPlainText()
                if isinstance(editor, QtWidgets.QPlainTextEdit)
                else editor.text()
            ).strip()
            for name, editor in self._edits.items()
        }
        return _ExtensionMetadata(**values)


class _WorkspaceRequirementsDialog(QtWidgets.QDialog):
    approve_requested = QtCore.Signal(str, str)

    def __init__(
        self,
        requirements: tuple[_ResolvedWorkspaceRequirement, ...],
        parent: QtWidgets.QWidget,
        *,
        approvable: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        super().__init__(parent)
        self._approvable = frozenset(approvable)
        self.setObjectName("manager_workspace_extension_requirements_dialog")
        self.setWindowTitle("Workspace Requirements")
        layout = QtWidgets.QVBoxLayout(self)
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setObjectName("manager_workspace_extension_requirements")
        self.tree.setHeaderLabels(("Extension", "Capability", "State", "Details"))
        layout.addWidget(self.tree)
        self._approve_button = QtWidgets.QPushButton(
            "Review and Add Embedded Script…", self
        )
        self._approve_button.setObjectName(
            "manager_workspace_extension_approve_embedded_button"
        )
        self._approve_button.clicked.connect(self._approve_selected)
        self.tree.currentItemChanged.connect(self._update_approve_button)
        self.set_requirements(requirements)
        layout.addWidget(self._approve_button)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, parent=self
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_requirements(
        self, requirements: tuple[_ResolvedWorkspaceRequirement, ...]
    ) -> None:
        """Replace displayed states while preserving the selected requirement."""
        selected = self.tree.currentItem()
        selected_key = (
            None
            if selected is None
            else selected.data(0, QtCore.Qt.ItemDataRole.UserRole)
        )
        self.tree.clear()
        for resolved in requirements:
            requirement = resolved.requirement
            item = QtWidgets.QTreeWidgetItem(
                (
                    requirement.extension_id,
                    requirement.capability_id,
                    resolved.state,
                    resolved.detail,
                )
            )
            item.setData(
                0,
                QtCore.Qt.ItemDataRole.UserRole,
                (requirement.extension_id, requirement.revision_hash),
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, resolved.state)
            self.tree.addTopLevelItem(item)
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == selected_key:
                self.tree.setCurrentItem(item)
        self._update_approve_button()

    @QtCore.Slot()
    def _update_approve_button(self) -> None:
        item = self.tree.currentItem()
        key = None if item is None else item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        self._approve_button.setEnabled(
            item is not None
            and key in self._approvable
            and item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) == "approval-required"
        )

    @QtCore.Slot()
    def _approve_selected(self) -> None:
        item = self.tree.currentItem()
        if (
            item is None
            or item.data(0, QtCore.Qt.ItemDataRole.UserRole) not in self._approvable
            or item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) != "approval-required"
        ):
            return
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(key, tuple) and len(key) == 2:
            self.approve_requested.emit(str(key[0]), str(key[1]))
