"""Dialogs used by the ImageTool Manager extension controller."""

from __future__ import annotations

import datetime
import pathlib
import sys
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive.utils
from erlab.extensions import (
    LoaderDescriptor,
    ParameterDescriptor,
    ParameterKind,
    RoutineDescriptor,
)
from erlab.extensions._models import _require_finite_parameter_values, _script_name_key
from erlab.interactive.imagetool.manager._widgets import _ElidedValueLabel

if typing.TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from erlab.interactive.imagetool.manager._extensions._models import (
        _ExtensionCatalogModel,
        _ResolvedWorkspaceRequirement,
        _ScriptRecord,
    )


def _display_datetime(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.datetime.fromisoformat(value).astimezone()
    except ValueError:
        return value
    return parsed.strftime("%Y-%m-%d %H:%M")


def _source_is_unavailable(state: str) -> bool:
    return state in {
        "Source file missing",
        "Source file unreadable",
        "Source file changed",
        "No registered source file",
    }


class _SourceReviewDialog(QtWidgets.QDialog):
    """Show extension source before the user approves it."""

    def __init__(
        self,
        path: pathlib.Path | None,
        parent: QtWidgets.QWidget,
        *,
        source_text: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("manager_extension_source_review_dialog")
        self.setWindowTitle("Review Extension Source")
        self.resize(760, 600)
        layout = QtWidgets.QVBoxLayout(self)
        source = erlab.interactive.utils.PythonCodeEditor(self)
        source.setObjectName("manager_extension_source_review")
        source.setReadOnly(True)
        source.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        if source_text is None:
            if path is None:
                raise ValueError("path or source_text is required")
            source_text = path.read_text(encoding="utf-8")
        source.setPlainText(source_text)
        layout.addWidget(source, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


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
    favorite_requested = QtCore.Signal(str, str, bool)

    def __init__(
        self,
        routines: tuple[tuple[str, RoutineDescriptor], ...],
        parent: QtWidgets.QWidget,
        *,
        favorites: Collection[tuple[str, str]] = (),
    ) -> None:
        super().__init__(parent)
        self._favorites = set(favorites)
        self.setObjectName("manager_extension_routine_selection_dialog")
        self.setWindowTitle("Run Routine")
        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setObjectName("manager_extension_routine_list")
        for script_name, descriptor in routines:
            item = QtWidgets.QListWidgetItem(
                f"{descriptor.category} — {descriptor.name} ({script_name})"
            )
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                (script_name, descriptor.id),
            )
            self.list_widget.addItem(item)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.accept())
        self.list_widget.currentItemChanged.connect(self._update_favorite_button)
        layout.addWidget(self.list_widget)
        self.favorite_button = QtWidgets.QPushButton(self)
        self.favorite_button.setObjectName("manager_extension_routine_favorite_button")
        self.favorite_button.clicked.connect(self._toggle_favorite)
        layout.addWidget(self.favorite_button)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_favorite_button()

    @property
    def selection(self) -> tuple[str, str] | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return typing.cast(
            "tuple[str, str]", item.data(QtCore.Qt.ItemDataRole.UserRole)
        )

    @QtCore.Slot()
    def _update_favorite_button(self) -> None:
        selection = self.selection
        favorite = selection is not None and selection in self._favorites
        self.favorite_button.setEnabled(selection is not None)
        self.favorite_button.setText(
            "Remove from Favorites" if favorite else "Add to Favorites"
        )
        self.favorite_button.setProperty("favoriteState", favorite)

    @QtCore.Slot()
    def _toggle_favorite(self) -> None:
        selection = self.selection
        if selection is None:
            return
        favorite = selection not in self._favorites
        if favorite:
            self._favorites.add(selection)
        else:
            self._favorites.remove(selection)
        self.favorite_requested.emit(*selection, favorite)
        self._update_favorite_button()


class _SourceViewerDialog(QtWidgets.QDialog):
    """Show extension source with Python highlighting."""

    def __init__(
        self,
        source_text: str,
        parent: QtWidgets.QWidget,
        *,
        title: str,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("manager_extension_source_viewer_dialog")
        self.setWindowTitle(title)
        self.resize(800, 600)
        layout = QtWidgets.QVBoxLayout(self)
        self.source = erlab.interactive.utils.PythonCodeEditor(self)
        self.source.setObjectName("manager_extension_running_source")
        self.source.setReadOnly(True)
        self.source.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        self.source.setPlainText(source_text)
        layout.addWidget(self.source)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, parent=self
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class _ManageExtensionsDialog(QtWidgets.QDialog):
    action_requested = QtCore.Signal(str, str)
    add_script_requested = QtCore.Signal()
    selection_changed = QtCore.Signal(str)
    activated = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("manager_manage_extensions_dialog")
        self.setWindowTitle("Manage Extensions")
        self.resize(1120, 620)
        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.add_script_button = QtWidgets.QPushButton("Add Script…", self)
        self.add_script_button.setObjectName("manager_extension_add_script_button")
        self.add_script_button.clicked.connect(self.add_script_requested)
        controls.addWidget(self.add_script_button)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setObjectName("manager_extension_search")
        self.search_edit.setPlaceholderText("Search extensions")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._apply_search)
        controls.addWidget(self.search_edit, 1)
        layout.addLayout(controls)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        splitter.setObjectName("manager_extension_splitter")
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setObjectName("manager_extension_catalog_tree")
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(True)
        self.tree.setHeaderLabels(("Extension", "Status", "Updated", "Location"))
        self.tree.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        splitter.addWidget(self.tree)
        details_widget = QtWidgets.QWidget(splitter)
        details_layout = QtWidgets.QVBoxLayout(details_widget)
        self.name_label = QtWidgets.QLabel(details_widget)
        self.name_label.setObjectName("manager_extension_detail_name")
        name_font = self.name_label.font()
        name_font.setBold(True)
        name_font.setPointSize(name_font.pointSize() + 2)
        self.name_label.setFont(name_font)
        details_layout.addWidget(self.name_label)
        self.status_label = QtWidgets.QLabel(details_widget)
        self.status_label.setObjectName("manager_extension_detail_status")
        details_layout.addWidget(self.status_label)
        self.failure_label = QtWidgets.QLabel(details_widget)
        self.failure_label.setObjectName("manager_extension_detail_failure")
        self.failure_label.setWordWrap(True)
        details_layout.addWidget(self.failure_label)
        self.capabilities_label = QtWidgets.QLabel(details_widget)
        self.capabilities_label.setObjectName("manager_extension_detail_capabilities")
        self.capabilities_label.setWordWrap(True)
        details_layout.addWidget(self.capabilities_label)
        form = QtWidgets.QFormLayout()
        self._detail_labels: dict[str, QtWidgets.QLabel] = {}
        self._detail_form_labels: dict[str, QtWidgets.QWidget] = {}
        for key, title in (
            ("embedding", "Workspace embedding"),
            ("registered_date", "Registered"),
            ("source", "Registered source file"),
        ):
            if key == "embedding":
                self.embedding_combo = QtWidgets.QComboBox(details_widget)
                self.embedding_combo.setObjectName("manager_extension_embedding_policy")
                for option_label, value in (
                    ("Embed when referenced", "referenced"),
                    ("Always embed", "always"),
                    ("Never embed", "never"),
                ):
                    self.embedding_combo.addItem(option_label, value)
                self.embedding_combo.activated.connect(self._embedding_changed)
                form.addRow(title, self.embedding_combo)
                self._embedding_form_label = form.labelForField(self.embedding_combo)
                continue
            if key == "source":
                label = _ElidedValueLabel(parent=details_widget)
            else:
                label = QtWidgets.QLabel(details_widget)
            label.setObjectName(f"manager_extension_detail_{key}")
            label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            label.setWordWrap(key != "source")
            self._detail_labels[key] = label
            form.addRow(title, label)
            form_label = form.labelForField(label)
            if form_label is not None:
                self._detail_form_labels[key] = form_label
        details_layout.addLayout(form)
        extension_actions_group = QtWidgets.QGroupBox(
            "Extension actions", details_widget
        )
        extension_actions_layout = QtWidgets.QGridLayout(extension_actions_group)
        source_actions_group = QtWidgets.QGroupBox(
            "Source and location actions", details_widget
        )
        source_actions_layout = QtWidgets.QGridLayout(source_actions_group)
        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        for action_id, text in (
            ("toggle", "Enable"),
            ("reload", "Reload from Disk…"),
            ("remove", "Remove Extension…"),
            ("error", "Show Error Details"),
            ("view_source", "View Source"),
            ("open_source", "Open Source File"),
            (
                "reveal_source",
                "Reveal in Finder"
                if sys.platform == "darwin"
                else "Reveal in File Explorer"
                if sys.platform.startswith("win")
                else "Open Containing Folder",
            ),
            ("copy_source", "Copy Path"),
        ):
            button = QtWidgets.QPushButton(text, details_widget)
            button.setObjectName(f"manager_extension_{action_id}_button")
            button.setProperty("extensionAction", action_id)
            button.clicked.connect(
                lambda _checked=False, value=action_id: self._emit_action(value)
            )
            self._buttons[action_id] = button
            if action_id in {
                "toggle",
                "reload",
                "remove",
                "error",
            }:
                position = extension_actions_layout.count()
                extension_actions_layout.addWidget(button, position // 2, position % 2)
            else:
                position = source_actions_layout.count()
                source_actions_layout.addWidget(button, position // 2, position % 2)
        details_layout.addWidget(extension_actions_group)
        details_layout.addWidget(source_actions_group)
        self.removal_reason_label = QtWidgets.QLabel(details_widget)
        self.removal_reason_label.setObjectName(
            "manager_extension_removal_unavailable_reason"
        )
        self.removal_reason_label.setWordWrap(True)
        details_layout.addWidget(self.removal_reason_label)
        details_layout.addStretch(1)
        splitter.addWidget(details_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        self._records: dict[str, _ScriptRecord] = {}
        self._source_states: dict[tuple[str, str], str] = {}
        self._validation_errors: dict[tuple[str, str], str] = {}
        self._removal_reason: str | None = None
        self.tree.currentItemChanged.connect(self._selection_changed)
        close_button = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, parent=self
        )
        close_button.rejected.connect(self.reject)
        layout.addWidget(close_button)

    def set_catalog(
        self,
        catalog: _ExtensionCatalogModel,
        source_states: dict[tuple[str, str], str] | None = None,
        *,
        validation_errors: dict[tuple[str, str], str] | None = None,
    ) -> None:
        selected_name = self.selected_script_name
        selected_key = (
            None if selected_name is None else _script_name_key(selected_name)
        )
        scroll_bar = self.tree.verticalScrollBar()
        scroll_position = 0 if scroll_bar is None else scroll_bar.value()
        self._records = {
            record.script_name: record for record in catalog.extensions.values()
        }
        self._source_states = dict(source_states or {})
        self._validation_errors = dict(validation_errors or {})
        self.tree.setSortingEnabled(False)
        self.tree.clear()
        selected_item: QtWidgets.QTreeWidgetItem | None = None
        for record in catalog.extensions.values():
            source_state = self._source_states.get(
                (record.script_name, record.source_hash), ""
            )
            validation_error = self._validation_errors.get(
                (record.script_name, record.source_hash)
            )
            health = "Ready"
            if validation_error:
                health = "Import failed"
            elif _source_is_unavailable(source_state):
                health = "Source unavailable"
            elif not record.approved:
                health = "Approval required"
            activation = "Enabled" if record.enabled else "Disabled"
            location = record.source_path
            state = f"{activation} · {health}"
            item = QtWidgets.QTreeWidgetItem(
                (
                    record.script_name,
                    state,
                    _display_datetime(
                        record.source_modified_at or record.registered_at
                    ),
                    location,
                )
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, record.script_name)
            item.setData(
                0,
                QtCore.Qt.ItemDataRole.UserRole + 1,
                "enabled" if record.enabled else "disabled",
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 2, health)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 3, location)
            self.tree.addTopLevelItem(item)
            if _script_name_key(record.script_name) == selected_key:
                selected_item = item
        self.tree.setSortingEnabled(True)
        self._apply_search()
        if selected_item is not None and not selected_item.isHidden():
            self.tree.setCurrentItem(selected_item)
        else:
            current_item = self.tree.currentItem()
            if current_item is None or current_item.isHidden():
                self._select_first_visible()
        if scroll_bar is not None:
            scroll_bar.setValue(scroll_position)
        self._update_details()

    @QtCore.Slot()
    def _update_details(self) -> None:
        record = self._records.get(self.selected_script_name or "")
        if record is None:
            self.name_label.clear()
            self.status_label.clear()
            self.failure_label.clear()
            self.capabilities_label.clear()
            for label in self._detail_labels.values():
                label.setText("")
            for button in self._buttons.values():
                button.setEnabled(False)
            self.embedding_combo.setEnabled(False)
            self.removal_reason_label.clear()
            return
        source_state = self._source_states.get(
            (record.script_name, record.source_hash), ""
        )
        validation_error = self._validation_errors.get(
            (record.script_name, record.source_hash)
        )
        health = "Ready"
        failure = ""
        if validation_error:
            health = "Import failed"
            failure = validation_error.splitlines()[-1]
        elif _source_is_unavailable(source_state):
            health = "Source unavailable"
            failure = source_state
        elif not record.approved:
            health = "Approval required"
        self.name_label.setText(record.script_name)
        activation = "Enabled" if record.enabled else "Disabled"
        self.status_label.setText(f"{activation} · {health}")
        self.status_label.setProperty(
            "activationState",
            "enabled" if record.enabled else "disabled",
        )
        self.status_label.setProperty("healthState", health)
        self.failure_label.setText(failure)
        self.failure_label.setProperty("fullError", validation_error or "")
        routine_names = ", ".join(item.name for item in record.routines) or "None"
        loader_names = ", ".join(item.name for item in record.loaders) or "None"
        self.capabilities_label.setText(
            f"Routines: {routine_names}\nLoaders: {loader_names}"
        )
        self._detail_labels["registered_date"].setText(
            _display_datetime(record.registered_at)
        )
        source_path = record.source_path
        self._detail_labels["source"].setText(source_path)
        self._detail_labels["source"].setProperty("sourcePath", source_path)
        self._buttons["toggle"].setEnabled(True)
        self._buttons["toggle"].setText(
            "Retry Validation"
            if validation_error
            else "Disable"
            if record.enabled
            else "Enable"
        )
        self._buttons["toggle"].setProperty(
            "extensionActionState",
            "retry" if validation_error else "disable" if record.enabled else "enable",
        )
        source_available = pathlib.Path(source_path).is_file()
        source_changed = source_state == "Source file changed"
        self._buttons["reload"].setEnabled(True)
        reload_text = (
            "Locate Script…"
            if not source_available
            else "Review Update…"
            if source_changed
            else "Reload from Disk…"
        )
        self._buttons["reload"].setText(reload_text)
        self._buttons["reload"].setProperty(
            "extensionActionState",
            "locate"
            if not source_available
            else "review"
            if source_changed
            else "reload",
        )
        self._buttons["error"].setVisible(bool(validation_error))
        self._buttons["error"].setEnabled(bool(validation_error))
        source_viewable = source_state == "Ready"
        self._buttons["view_source"].setEnabled(source_viewable)
        self._buttons["view_source"].setToolTip(
            "Review the script update before you view its source."
            if source_changed
            else ""
            if source_viewable
            else "The registered source file is unavailable."
        )
        for action_id in ("open_source", "reveal_source", "copy_source"):
            self._buttons[action_id].setEnabled(source_available)
            self._buttons[action_id].setToolTip(
                "" if source_available else "The registered source file is unavailable."
            )
        self.embedding_combo.setEnabled(True)
        index = self.embedding_combo.findData(record.embed_policy)
        if index >= 0:
            blocker = QtCore.QSignalBlocker(self.embedding_combo)
            self.embedding_combo.setCurrentIndex(index)
            del blocker
        self._buttons["remove"].setEnabled(not self._removal_reason)
        self._buttons["remove"].setToolTip(self._removal_reason or "")
        self._buttons["remove"].setProperty(
            "removalBlocked", bool(self._removal_reason)
        )
        self.removal_reason_label.setText(self._removal_reason or "")
        self.removal_reason_label.setVisible(bool(self._removal_reason))

    def set_removal_reason(self, reason: str | None) -> None:
        self._removal_reason = reason
        self._update_details()

    @property
    def selected_script_name(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return value if isinstance(value, str) else None

    @QtCore.Slot()
    def _selection_changed(self) -> None:
        self._removal_reason = None
        self._update_details()
        script_name = self.selected_script_name
        if script_name is not None:
            self.selection_changed.emit(script_name)

    @QtCore.Slot(str)
    def _apply_search(self, _text: str = "") -> None:
        query = self.search_edit.text().strip().casefold()
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item is None:
                continue
            item.setHidden(
                bool(query)
                and query
                not in " ".join(item.text(column) for column in range(4)).casefold()
            )
        current = self.tree.currentItem()
        if current is None or current.isHidden():
            self._select_first_visible()

    def _select_first_visible(self) -> None:
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item is not None and not item.isHidden():
                self.tree.setCurrentItem(item)
                return
        self.tree.setCurrentItem(None)

    @QtCore.Slot(int)
    def _embedding_changed(self, _index: int) -> None:
        policy = self.embedding_combo.currentData()
        if isinstance(policy, str):
            self._emit_action(f"embedding:{policy}")

    def _emit_action(self, action_id: str) -> None:
        script_name = self.selected_script_name
        if script_name is not None:
            self.action_requested.emit(action_id, script_name)

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        super().showEvent(event)
        self.activated.emit()

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        super().changeEvent(event)
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.ActivationChange
            and self.isActiveWindow()
        ):
            self.activated.emit()


class _WorkspaceRequirementsDialog(QtWidgets.QDialog):
    register_requested = QtCore.Signal(str, str)

    def __init__(
        self,
        requirements: tuple[_ResolvedWorkspaceRequirement, ...],
        parent: QtWidgets.QWidget,
        *,
        recoverable: Collection[tuple[str, str]] = (),
    ) -> None:
        super().__init__(parent)
        self._recoverable = frozenset(recoverable)
        self.setObjectName("manager_workspace_extension_requirements_dialog")
        self.setWindowTitle("Workspace Requirements")
        layout = QtWidgets.QVBoxLayout(self)
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setObjectName("manager_workspace_extension_requirements")
        self.tree.setHeaderLabels(("Extension", "Capability", "State", "Details"))
        layout.addWidget(self.tree)
        self._register_button = QtWidgets.QPushButton("Save and Register Script…", self)
        self._register_button.setObjectName(
            "manager_workspace_extension_register_script_button"
        )
        self._register_button.clicked.connect(self._register_selected)
        self.tree.currentItemChanged.connect(self._update_register_button)
        layout.addWidget(self._register_button)
        self.set_requirements(requirements)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, parent=self
        )
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_requirements(
        self,
        requirements: tuple[_ResolvedWorkspaceRequirement, ...],
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
                    requirement.script_name,
                    requirement.capability_name,
                    resolved.state,
                    resolved.detail,
                )
            )
            item.setData(
                0,
                QtCore.Qt.ItemDataRole.UserRole,
                (
                    requirement.script_name,
                    requirement.source_hash,
                ),
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, resolved.state)
            self.tree.addTopLevelItem(item)
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == selected_key:
                self.tree.setCurrentItem(item)
        self._update_register_button()

    @QtCore.Slot()
    def _update_register_button(self) -> None:
        item = self.tree.currentItem()
        key = None if item is None else item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        self._register_button.setEnabled(
            item is not None
            and key in self._recoverable
            and item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)
            in {
                "approval-required",
                "missing",
                "hash-mismatch",
                "validation-failed",
            }
        )

    @QtCore.Slot()
    def _register_selected(self) -> None:
        item = self.tree.currentItem()
        if (
            item is None
            or item.data(0, QtCore.Qt.ItemDataRole.UserRole) not in self._recoverable
            or item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)
            not in {
                "approval-required",
                "missing",
                "hash-mismatch",
                "validation-failed",
            }
        ):
            return
        key = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(key, tuple) and len(key) == 2:
            self.register_requested.emit(str(key[0]), str(key[1]))


class _MissingScriptsDialog(QtWidgets.QDialog):
    """Recover several unavailable registered scripts in one dialog."""

    locate_requested = QtCore.Signal(str)

    def __init__(
        self,
        records: tuple[_ScriptRecord, ...],
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("manager_missing_extension_scripts_dialog")
        self.setWindowTitle("Extension Scripts Not Found")
        layout = QtWidgets.QVBoxLayout(self)
        message = QtWidgets.QLabel(
            "Some enabled extension scripts are not at their registered locations.",
            self,
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setObjectName("manager_missing_extension_scripts")
        self.tree.setRootIsDecorated(False)
        self.tree.setHeaderLabels(("Script", "Registered location"))
        layout.addWidget(self.tree)
        actions = QtWidgets.QHBoxLayout()
        self.locate_button = QtWidgets.QPushButton("Locate Script…", self)
        self.locate_button.setObjectName("manager_locate_extension_script_button")
        self.locate_button.clicked.connect(self._locate_selected)
        actions.addWidget(self.locate_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close, parent=self
        )
        close_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setText("Not Now")
            close_button.setObjectName("manager_missing_extension_not_now_button")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.tree.currentItemChanged.connect(self._update_actions)
        self.set_records(records)

    @property
    def selected_script_name(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        return value if isinstance(value, str) else None

    def set_records(self, records: tuple[_ScriptRecord, ...]) -> None:
        """Refresh missing scripts and retain the selected extension when possible."""
        selected_name = self.selected_script_name
        self.tree.clear()
        selected_item: QtWidgets.QTreeWidgetItem | None = None
        for record in records:
            item = QtWidgets.QTreeWidgetItem((record.script_name, record.source_path))
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, record.script_name)
            self.tree.addTopLevelItem(item)
            if record.script_name == selected_name:
                selected_item = item
        if selected_item is not None:
            self.tree.setCurrentItem(selected_item)
        elif self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self._update_actions()

    @QtCore.Slot()
    def _update_actions(self) -> None:
        enabled = self.selected_script_name is not None
        self.locate_button.setEnabled(enabled)

    @QtCore.Slot()
    def _locate_selected(self) -> None:
        if (script_name := self.selected_script_name) is not None:
            self.locate_requested.emit(script_name)
