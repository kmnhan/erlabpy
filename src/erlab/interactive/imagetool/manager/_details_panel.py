from __future__ import annotations

import contextlib
import html
import json
import logging
import traceback
import typing
from dataclasses import replace

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
import erlab.interactive.utils
from erlab.interactive._widgets import _CenteredIconToolButton
from erlab.interactive.imagetool import _replay_graph, provenance
from erlab.interactive.imagetool.manager._widgets import (
    _METADATA_DERIVATION_ACTIVATABLE_ROLE,
    _METADATA_DERIVATION_CODE_ROLE,
    _METADATA_DERIVATION_COPYABLE_ROLE,
    _METADATA_DERIVATION_ROW_ROLE,
    _QWIDGETSIZE_MAX,
    _ElidedValueLabel,
    _LoadSourceDetailsDialog,
    _MetadataDerivationTreeItem,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
    _MetadataField,
    _preview_curve_for_node,
    _preview_image_for_node,
)

if typing.TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterable

    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool._load_source import _LoadSourceDetails
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager

logger = logging.getLogger(__name__)

_TOOL_PREVIEW_UPDATE_DELAY_MS = 250
_PROVENANCE_STEPS_CLIPBOARD_MIME = "application/x-erlab-imagetool-provenance-steps+json"
_PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE = "erlab.imagetool.provenance.steps"
_PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_VERSION = 1


def _provenance_step_clipboard_payload(
    mime_data: QtCore.QMimeData | None,
) -> tuple[tuple[provenance.ToolProvenanceOperation, ...], str, bool] | None:
    if mime_data is None:
        return None
    payload_text: str | None = None
    if mime_data.hasFormat(_PROVENANCE_STEPS_CLIPBOARD_MIME):
        try:
            payload_text = (
                mime_data.data(_PROVENANCE_STEPS_CLIPBOARD_MIME).data().decode("utf-8")
            )
        except UnicodeDecodeError:
            return None
    elif mime_data.hasText():
        text = mime_data.text().strip()
        if text.startswith("{"):
            payload_text = text
    if payload_text is None:
        return None
    try:
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            return None
        if payload.get("type") != _PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE:
            return None
        if payload.get("version") != _PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_VERSION:
            return None
        operations_payload = payload.get("operations")
        if not isinstance(operations_payload, list):
            return None
        operations = tuple(
            provenance.parse_tool_provenance_operation(operation)
            for operation in operations_payload
        )
        operations = provenance.strip_partial_operation_groups(operations)
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    active_name = payload.get("active_name")
    if not isinstance(active_name, str) or not active_name:
        active_name = "derived"
    return (
        operations,
        active_name,
        any(not operation.live_applicable for operation in operations),
    )


class _DetailsPanelController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager
        self._tool_preview_update_generation = 0

    def _node_info_html(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        return node.info_text

    def _notes_ui_available(self) -> bool:
        return hasattr(self._manager, "notes_editor")

    def _selected_note_node(self) -> _ImageToolWrapper | _ManagedWindowNode | None:
        targets: list[int | str] = []
        targets.extend(self._manager._selected_imagetool_targets())
        targets.extend(self._manager._selected_tool_uids())
        if len(targets) != 1:
            return None
        return self._manager._node_for_target(targets[0])

    def _current_note_node(self) -> _ImageToolWrapper | _ManagedWindowNode | None:
        uid = self._manager._notes_node_uid
        if uid is None:
            return None
        return self._manager._tool_graph.nodes.get(uid)

    def _selected_or_current_note_node(
        self,
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        return self._selected_note_node() or self._current_note_node()

    def _note_kind_text(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        if self._manager._is_figure_uid(node.uid):
            return "Figure Composer"
        if node.is_imagetool:
            return "ImageTool"
        return node.type_badge_text or "Analysis Tool"

    def _set_notes_node(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode | None,
    ) -> None:
        if not self._notes_ui_available():
            return
        if node is None:
            self._manager._commit_note_editor()
            self._manager._notes_node_uid = None
            self._manager._updating_note_editor = True
            try:
                self._manager.notes_title_label.setText("")
                self._manager.notes_kind_label.clear()
                self._manager.notes_editor.clear()
            finally:
                self._manager._updating_note_editor = False
            self._manager.notes_editor.setEnabled(False)
            self._update_note_actions()
            return

        if self._manager._notes_node_uid != node.uid:
            self._manager._commit_note_editor()
        self._manager._notes_node_uid = node.uid
        self._manager.notes_title_label.setText(node.display_text)
        self._manager.notes_kind_label.setText(self._note_kind_text(node))
        self._manager._updating_note_editor = True
        try:
            if self._manager.notes_editor.toPlainText() != node.note:
                self._manager.notes_editor.setPlainText(node.note)
        finally:
            self._manager._updating_note_editor = False
        self._manager.notes_editor.setEnabled(True)
        self._update_note_actions()

    def _update_note_actions(self) -> None:
        if not self._notes_ui_available():
            return
        node = self._selected_or_current_note_node()
        can_edit = node is not None
        has_note = bool(node is not None and node.has_note)
        self._manager.edit_note_action.setEnabled(can_edit)
        self._manager.copy_note_action.setEnabled(has_note)
        self._manager.clear_note_action.setEnabled(has_note)

    def _schedule_note_commit(self) -> None:
        if not self._notes_ui_available():
            return
        if self._manager._updating_note_editor:
            return
        self._manager._note_commit_timer.start()

    def _commit_note_editor(self) -> None:
        if not self._notes_ui_available():
            return
        self._manager._note_commit_timer.stop()
        if self._manager._updating_note_editor:
            return
        node = self._current_note_node()
        if node is None:
            return
        note = self._manager.notes_editor.toPlainText()
        if node.note == note:
            self._update_note_actions()
            return
        node.note = note
        self._manager._mark_node_state_dirty(node.uid)
        self._manager.tree_view.refresh(node.uid)
        self._update_note_actions()

    def _edit_selected_note(self) -> None:
        if not self._notes_ui_available():
            return
        node = self._selected_note_node()
        if node is None:
            return
        self._set_notes_node(node)
        self._manager.inspector_tabs.setCurrentWidget(self._manager.notes_page)
        self._manager.notes_editor.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)

    def _copy_selected_note(self) -> None:
        if not self._notes_ui_available():
            return
        self._commit_note_editor()
        node = self._selected_or_current_note_node()
        if node is None or not node.note:
            return
        erlab.interactive.utils.copy_to_clipboard(node.note)

    def _clear_selected_note(self) -> None:
        if not self._notes_ui_available():
            return
        self._commit_note_editor()
        node = self._selected_or_current_note_node()
        if node is None:
            return
        self._set_notes_node(node)
        self._manager.notes_editor.clear()
        self._commit_note_editor()

    def _clear_metadata(self) -> None:
        self._manager._metadata_full_code_available = False
        self._manager._metadata_node_uid = None
        self._set_notes_node(None)
        with QtCore.QSignalBlocker(self._manager.metadata_derivation_list):
            self._manager.metadata_derivation_list.clear()
        self._manager._set_metadata_fields([])
        self._manager._update_metadata_pane()

    def _set_metadata_node(self, node: _ImageToolWrapper | _ManagedWindowNode) -> None:
        self._set_notes_node(node)
        self._manager._metadata_full_code_available = (
            node.displayed_provenance_spec is not None
        )
        self._manager._metadata_node_uid = node.uid
        self._manager._set_metadata_fields(node.metadata_fields)

        with QtCore.QSignalBlocker(self._manager.metadata_derivation_list):
            self._manager.metadata_derivation_list.clear()
            for row in self._current_derivation_display_rows(node):
                self._manager.metadata_derivation_list.addItem(
                    self._metadata_derivation_item(row)
                )
        self._manager._update_metadata_pane()

    def _script_input_current_node_label(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
    ) -> str:
        graph = getattr(self._manager, "_tool_graph", None)
        path: list[int] = []
        current = node
        while current.parent_uid is not None:
            parent = graph.nodes.get(current.parent_uid) if graph is not None else None
            if parent is None or current.uid not in parent._childtool_indices:
                path = []
                break
            path.append(parent._childtool_indices.index(current.uid))
            current = parent
        else:
            if graph is not None:
                for root_index, wrapper in graph.root_wrappers.items():
                    if wrapper.uid == current.uid:
                        path = [root_index, *reversed(path)]
                        break

        display_index = ".".join(str(index) for index in path) if path else node.uid
        label = (
            f"ImageTool {display_index}"
            if node.is_imagetool
            else f"{node.type_badge_text or 'Tool'} {display_index}"
        )
        if node.name:
            label += f": {node.name}"
        return label

    def _script_input_row_label(
        self,
        script_input: provenance.ScriptInput,
        *,
        include_history: bool,
    ) -> str:
        graph = getattr(self._manager, "_tool_graph", None)
        node_uid = script_input.node_uid
        if graph is not None and node_uid is not None:
            node = graph.nodes.get(node_uid)
            if node is not None:
                return (
                    f"Use {script_input.name} from "
                    f"{self._script_input_current_node_label(node)}"
                )
            label = f"Missing source for {script_input.name}"
            if include_history:
                label += f" (recorded as {script_input.label})"
            return label
        return f"Use {script_input.name} from {script_input.label}"

    @staticmethod
    def _script_input_for_row(
        spec: provenance.ToolProvenanceSpec | None,
        row: provenance._ProvenanceDisplayRow,
    ) -> provenance.ScriptInput | None:
        ref = row.replay_ref
        if (
            spec is None
            or ref is None
            or ref.kind != "script_input"
            or ref.script_input_index is None
        ):
            return None
        current = spec
        for index in row.script_input_path:
            if current.kind != "script" or index >= len(current.script_inputs):
                return None
            nested = current.script_inputs[index].parsed_provenance_spec()
            if nested is None:
                return None
            current = nested
        if current.kind != "script" or ref.script_input_index >= len(
            current.script_inputs
        ):
            return None
        return current.script_inputs[ref.script_input_index]

    def _current_derivation_display_row(
        self,
        row: provenance._ProvenanceDisplayRow,
        spec: provenance.ToolProvenanceSpec | None,
        *,
        include_history: bool,
    ) -> provenance._ProvenanceDisplayRow:
        children = tuple(
            self._current_derivation_display_row(
                child,
                spec,
                include_history=include_history,
            )
            for child in row.children
        )
        script_input = self._script_input_for_row(spec, row)
        label = (
            None
            if script_input is None
            else self._script_input_row_label(
                script_input,
                include_history=include_history,
            )
        )
        if label is None and children == row.children:
            return row
        entry = row.entry if label is None else replace(row.entry, label=label)
        return replace(row, entry=entry, children=children)

    def _current_derivation_display_rows(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        include_history: bool = False,
    ) -> list[provenance._ProvenanceDisplayRow]:
        spec = getattr(node, "displayed_provenance_spec", None)
        rows = getattr(node, "derivation_display_rows", None)
        if rows is None:
            rows = [
                provenance._ProvenanceDisplayRow(entry)
                for entry in getattr(node, "derivation_entries", ())
            ]
        return [
            self._current_derivation_display_row(
                row,
                spec,
                include_history=include_history,
            )
            for row in rows
        ]

    @staticmethod
    def _flatten_derivation_rows(
        rows: Iterable[provenance._ProvenanceDisplayRow],
    ) -> list[provenance._ProvenanceDisplayRow]:
        flattened: list[provenance._ProvenanceDisplayRow] = []
        for row in rows:
            flattened.append(row)
            flattened.extend(
                _DetailsPanelController._flatten_derivation_rows(row.children)
            )
        return flattened

    def _metadata_derivation_item(
        self,
        row: provenance._ProvenanceDisplayRow,
    ) -> _MetadataDerivationTreeItem:
        entry = row.entry
        item = _MetadataDerivationTreeItem(entry.label)
        can_activate, activation_reason = (
            self._manager._provenance_edit_controller.can_edit_row(row)
        )
        tooltip_lines = [entry.label]
        if not can_activate and activation_reason:
            tooltip_lines.extend(("", activation_reason))
        if (
            not entry.copyable
            and entry.code is None
            and not entry.label.startswith("Start from ")
        ):
            tooltip_lines.extend(("", "Replay code is unavailable for this step."))
        item.setToolTip("\n".join(tooltip_lines))
        item.setData(_METADATA_DERIVATION_CODE_ROLE, entry.code)
        item.setData(_METADATA_DERIVATION_COPYABLE_ROLE, entry.copyable)
        item.setData(_METADATA_DERIVATION_ROW_ROLE, row)
        item.setData(_METADATA_DERIVATION_ACTIVATABLE_ROLE, can_activate)
        if not can_activate:
            item.setForeground(
                self._manager.metadata_derivation_list.palette().color(
                    QtGui.QPalette.ColorGroup.Disabled,
                    QtGui.QPalette.ColorRole.Text,
                )
            )
        for child_row in row.children:
            item.addChild(self._metadata_derivation_item(child_row))
        return item

    def _set_metadata_fields(self, fields: list[_MetadataField]) -> None:
        layout = self._manager.metadata_details_layout
        for row in range(layout.rowCount()):
            layout.setRowMinimumHeight(row, 0)
            layout.setRowStretch(row, 0)
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._manager._metadata_detail_labels.clear()

        for row, field in enumerate(fields):
            row_vertical_alignment = (
                QtCore.Qt.AlignmentFlag.AlignTop
                if field.wrap
                else QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            key_label = QtWidgets.QLabel(
                field.label, self._manager.metadata_details_widget
            )
            key_label.setForegroundRole(QtGui.QPalette.ColorRole.Text)
            key_label.setEnabled(False)
            key_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | row_vertical_alignment
            )
            value_label: QtWidgets.QLabel
            value_widget: QtWidgets.QWidget
            details_button: QtWidgets.QToolButton | None = None
            if field.details is not None:
                value_label = _ElidedValueLabel(
                    field.value,
                    self._manager.metadata_details_widget,
                    elide_mode=QtCore.Qt.TextElideMode.ElideMiddle,
                )
                node_uid = self._manager._metadata_node_uid
                details_button = _CenteredIconToolButton(
                    self._manager.metadata_details_widget
                )
                details_button.setObjectName("manager_metadata_file_details_button")
                details_button.setAutoRaise(True)
                details_button.setToolTip("Show data source details")
                details_button.setAccessibleName("Show data source details")
                icon = QtGui.QIcon.fromTheme("dialog-information")
                if icon.isNull():
                    style = details_button.style() or QtWidgets.QApplication.style()
                    if style is not None:
                        icon = style.standardIcon(
                            QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation
                        )
                details_button.setIcon(icon)
                details_button.clicked.connect(
                    lambda _checked=False, d=field.details, uid=node_uid: (
                        self._manager._show_load_source_details(d, node_uid=uid)
                    )
                )
            elif field.wrap:
                value_label = QtWidgets.QLabel(
                    field.value, self._manager.metadata_details_widget
                )
                value_label.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                )
                value_label.setWordWrap(field.wrap)
                value_label.setToolTip(field.value)
                value_label.setMinimumWidth(0)
                size_policy = QtWidgets.QSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Preferred,
                )
                size_policy.setHeightForWidth(True)
                value_label.setSizePolicy(size_policy)
            else:
                value_label = _ElidedValueLabel(
                    field.value,
                    self._manager.metadata_details_widget,
                    elide_mode=QtCore.Qt.TextElideMode.ElideRight,
                )
            if not field.wrap:
                value_label.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored,
                    QtWidgets.QSizePolicy.Policy.Preferred,
                )
            if field.monospace:
                value_label.setFont(self._manager._metadata_monospace_font)
            value_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | row_vertical_alignment
            )
            if details_button is not None:
                style = details_button.style() or QtWidgets.QApplication.style()
                small_icon_size = (
                    style.pixelMetric(
                        QtWidgets.QStyle.PixelMetric.PM_SmallIconSize,
                        None,
                        details_button,
                    )
                    if style is not None
                    else value_label.fontMetrics().height()
                )
                row_height = max(
                    key_label.sizeHint().height(),
                    value_label.sizeHint().height(),
                    small_icon_size,
                )
                details_button.setIconSize(
                    QtCore.QSize(small_icon_size, small_icon_size)
                )
                details_button.setFixedSize(QtCore.QSize(row_height, row_height))
                spacing = layout.horizontalSpacing()
                if spacing < 0:
                    spacing = (
                        style.pixelMetric(
                            QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing,
                            None,
                            details_button,
                        )
                        if style is not None
                        else 0
                    )
                value_widget = QtWidgets.QWidget(self._manager.metadata_details_widget)
                value_widget.setMinimumWidth(0)
                value_layout = QtWidgets.QHBoxLayout(value_widget)
                value_layout.setContentsMargins(0, 0, 0, 0)
                value_layout.setSpacing(max(0, spacing))
                value_layout.addWidget(value_label, 1)
                value_layout.addWidget(
                    details_button,
                    0,
                    alignment=QtCore.Qt.AlignmentFlag.AlignVCenter,
                )
                if spacing > 0:
                    value_layout.addSpacing(spacing)
            else:
                value_widget = value_label
            layout.addWidget(key_label, row, 0)
            layout.addWidget(value_widget, row, 1)
            self._manager._metadata_detail_labels[field.label] = value_label

    def _show_load_source_details(
        self,
        details: _LoadSourceDetails,
        *,
        node_uid: str | None = None,
    ) -> None:
        if node_uid is None:
            node_uid = self._manager._metadata_node_uid
        node = (
            None if node_uid is None else self._manager._tool_graph.nodes.get(node_uid)
        )
        can_edit_file_load = False
        edit_file_load_tooltip = (
            "This source was not recorded as an editable file-load step."
        )
        if node is not None:
            can_edit_file_load, edit_file_load_tooltip = (
                self._manager._provenance_edit_controller.can_edit_file_load_source(
                    node,
                    details.path,
                )
            )
        dialog = _LoadSourceDetailsDialog(
            details,
            self._manager,
            show_in_data_explorer=self._show_load_source_in_data_explorer,
            can_edit_file_load=can_edit_file_load,
            edit_file_load_tooltip=edit_file_load_tooltip,
        )
        dialog.exec()
        if not dialog.edit_file_load_requested or node_uid is None:
            return
        node = self._manager._tool_graph.nodes.get(node_uid)
        if node is None:
            return
        self._manager._provenance_edit_controller.edit_file_load_source(
            node,
            details.path,
        )

    def _show_load_source_in_data_explorer(self, path: pathlib.Path) -> None:
        explorer = typing.cast(
            "_TabbedExplorer", self._manager._show_standalone_app("explorer")
        )
        explorer.show_path(path)

    def _load_source_for_replay(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[str, str] | None:
        current = node
        while True:
            source_name = current.default_load_source_name()
            if source_name is not None:
                load_code = current.load_source_code(assign=source_name)
                if load_code is not None:
                    return source_name, load_code
            if current.parent_uid is None:
                return None
            if current.provenance_spec is None:
                return None
            current = self._manager._parent_node(current)

    def _prompt_replay_input_name(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        data = node._metadata_data()
        candidate = None if data is None else data.name
        if not erlab.interactive.utils._is_kwarg_name(candidate) or candidate in {
            "data",
            "derived",
            "result",
        }:
            candidate = "source_data"

        dialog = QtWidgets.QInputDialog(self._manager)
        dialog.setWindowTitle("Copy Full Code")
        dialog.setLabelText("Source variable name:")
        dialog.setTextValue(typing.cast("str", candidate))
        dialog.setInputMode(QtWidgets.QInputDialog.InputMode.TextInput)
        line_edit = dialog.findChild(QtWidgets.QLineEdit)
        if line_edit is not None:
            line_edit.setValidator(erlab.interactive.utils.IdentifierValidator())
            line_edit.selectAll()

        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return None
        source_name = dialog.textValue().strip()
        if not erlab.interactive.utils._is_kwarg_name(source_name):
            return None
        return source_name

    def _update_metadata_pane(self) -> None:
        has_details = bool(self._manager._metadata_detail_labels)
        derivation_count = self._manager.metadata_derivation_list.count()
        has_note_target = self._manager._notes_node_uid is not None
        has_metadata = has_details or derivation_count > 0 or has_note_target

        self._manager.metadata_group.setVisible(has_metadata)
        self._manager.metadata_details_widget.setVisible(has_details)
        self._manager.metadata_derivation_list.setVisible(derivation_count > 0)
        details_index = self._manager.inspector_tabs.indexOf(
            self._manager.metadata_details_page
        )
        provenance_index = self._manager.inspector_tabs.indexOf(
            self._manager.metadata_provenance_page
        )
        notes_index = self._manager.inspector_tabs.indexOf(self._manager.notes_page)
        current_widget = self._manager.inspector_tabs.currentWidget()
        self._manager.inspector_tabs.setTabEnabled(
            details_index, has_details or not has_metadata
        )
        self._manager.inspector_tabs.setTabEnabled(
            provenance_index, derivation_count > 0
        )
        self._manager.inspector_tabs.setTabEnabled(notes_index, has_note_target)
        if current_widget is self._manager.metadata_details_page and not has_details:
            if derivation_count > 0:
                self._manager.inspector_tabs.setCurrentWidget(
                    self._manager.metadata_provenance_page
                )
            elif has_note_target:
                self._manager.inspector_tabs.setCurrentWidget(self._manager.notes_page)
        elif (
            current_widget is self._manager.metadata_provenance_page
            and derivation_count == 0
        ):
            if has_details:
                self._manager.inspector_tabs.setCurrentWidget(
                    self._manager.metadata_details_page
                )
            elif has_note_target:
                self._manager.inspector_tabs.setCurrentWidget(self._manager.notes_page)
        elif current_widget is self._manager.notes_page and not has_note_target:
            if has_details:
                self._manager.inspector_tabs.setCurrentWidget(
                    self._manager.metadata_details_page
                )
            elif derivation_count > 0:
                self._manager.inspector_tabs.setCurrentWidget(
                    self._manager.metadata_provenance_page
                )

        if derivation_count == 0:
            self._manager.metadata_derivation_list.setMinimumHeight(0)
            self._manager.metadata_derivation_list.setMaximumHeight(0)
        else:
            row_height = self._manager.metadata_derivation_list.sizeHintForRow(0)
            if row_height <= 0:
                row_height = self._manager.fontMetrics().height() + 8
            visible_rows = min(derivation_count, 4)
            frame = self._manager.metadata_derivation_list.frameWidth() * 2
            height = visible_rows * row_height + frame + 4
            self._manager.metadata_derivation_list.setMinimumHeight(height)
            self._manager.metadata_derivation_list.setMaximumHeight(_QWIDGETSIZE_MAX)

        self._manager.metadata_details_widget.updateGeometry()
        self._manager.metadata_derivation_list.updateGeometry()
        self._manager.metadata_details_widget.sync_height_for_width()
        self._manager.metadata_group.updateGeometry()

    def _selected_derivation_items(self) -> list[_MetadataDerivationTreeItem]:
        items = typing.cast(
            "list[_MetadataDerivationTreeItem]",
            list(self._manager.metadata_derivation_list.selectedItems()),
        )
        display_order = getattr(
            self._manager.metadata_derivation_list,
            "display_order",
            self._manager.metadata_derivation_list.row,
        )
        return sorted(items, key=display_order)

    def _selected_derivation_code(self) -> str | None:
        codes: list[str] = []
        for item in self._manager._selected_derivation_items():
            if not bool(item.data(_METADATA_DERIVATION_COPYABLE_ROLE)):
                continue
            code = typing.cast("str | None", item.data(_METADATA_DERIVATION_CODE_ROLE))
            if code:
                codes.append(code)
        if not codes:
            return None
        return "\n".join(codes)

    def _selected_derivation_step_payload(
        self,
    ) -> tuple[tuple[provenance.ToolProvenanceOperation, ...], str, bool] | None:
        if self._manager._metadata_node_uid is None:
            return None
        node = self._manager._tool_graph.nodes.get(self._manager._metadata_node_uid)
        if node is None:
            return None

        operations: list[provenance.ToolProvenanceOperation] = []
        script_active_names: list[str] = []
        for item in self._manager._selected_derivation_items():
            row = item.data(_METADATA_DERIVATION_ROW_ROLE)
            if not isinstance(row, provenance._ProvenanceDisplayRow):
                continue
            ref = row.replay_ref
            if ref is None or ref.kind != "operation":
                continue
            if not bool(item.data(_METADATA_DERIVATION_COPYABLE_ROLE)):
                continue
            if not item.data(_METADATA_DERIVATION_CODE_ROLE):
                continue
            spec = self._manager._provenance_edit_controller._display_spec_for_row(
                node,
                row,
            )
            if spec is None:
                continue
            operation = spec._operation_for_ref(ref)
            if operation is None:
                continue
            if isinstance(operation, provenance.ScriptCodeOperation):
                if operation.copyable and operation.code:
                    operations.append(operation)
                    script_active_names.append(spec.active_name or "derived")
                continue
            if operation.live_applicable:
                operations.append(operation)

        if not operations:
            return None
        script_active_name = (
            script_active_names[0] if script_active_names else "derived"
        )
        if any(
            active_name != script_active_name for active_name in script_active_names
        ):
            return None
        operations = list(provenance.strip_partial_operation_groups(operations))
        return (
            tuple(operations),
            script_active_name,
            any(not operation.live_applicable for operation in operations),
        )

    def _selected_derivation_row(
        self,
    ) -> provenance._ProvenanceDisplayRow | None:
        items = self._manager._selected_derivation_items()
        if len(items) != 1:
            return None
        row = items[0].data(_METADATA_DERIVATION_ROW_ROLE)
        if isinstance(row, provenance._ProvenanceDisplayRow):
            return row
        return None

    def _build_metadata_derivation_menu(
        self, *, include_row_actions: bool = True
    ) -> QtWidgets.QMenu | None:
        if self._manager.metadata_derivation_list.count() == 0:
            return None

        menu = QtWidgets.QMenu(self._manager.metadata_derivation_list)
        row = self._manager._selected_derivation_row() if include_row_actions else None
        edit_enabled, edit_reason = (
            self._manager._provenance_edit_controller.can_edit_row(row)
        )
        revert_enabled, revert_reason = (
            self._manager._provenance_edit_controller.can_revert_row(row)
        )
        delete_enabled, delete_reason = (
            self._manager._provenance_edit_controller.can_delete_row(row)
        )
        self._manager._metadata_edit_step_action.setEnabled(edit_enabled)
        self._manager._metadata_edit_step_action.setToolTip(edit_reason)
        self._manager._metadata_revert_step_action.setEnabled(revert_enabled)
        self._manager._metadata_revert_step_action.setToolTip(revert_reason)
        self._manager._metadata_delete_step_action.setEnabled(delete_enabled)
        self._manager._metadata_delete_step_action.setToolTip(delete_reason)
        menu.addAction(self._manager._metadata_edit_step_action)
        if edit_enabled:
            menu.setDefaultAction(self._manager._metadata_edit_step_action)
        menu.addAction(self._manager._metadata_revert_step_action)
        menu.addSeparator()

        selected_code = (
            self._manager._selected_derivation_code() if include_row_actions else None
        )
        self._manager._metadata_copy_selected_action.setEnabled(bool(selected_code))
        menu.addAction(self._manager._metadata_copy_selected_action)
        clipboard = QtWidgets.QApplication.clipboard()
        paste_payload = _provenance_step_clipboard_payload(
            None if clipboard is None else clipboard.mimeData()
        )
        paste_enabled, paste_reason = (
            self._manager._provenance_edit_controller.can_paste_steps(
                None if paste_payload is None else paste_payload[0]
            )
        )
        self._manager._metadata_paste_steps_action.setEnabled(paste_enabled)
        self._manager._metadata_paste_steps_action.setToolTip(paste_reason)
        menu.addAction(self._manager._metadata_paste_steps_action)
        if self._manager._metadata_full_code_available:
            self._manager._metadata_copy_full_action.setEnabled(True)
            menu.addAction(self._manager._metadata_copy_full_action)
        menu.addSeparator()
        menu.addAction(self._manager._metadata_delete_step_action)
        return menu

    def _show_metadata_derivation_menu(self, pos: QtCore.QPoint) -> None:
        item = self._manager.metadata_derivation_list.itemAt(pos)
        menu = self._manager._build_metadata_derivation_menu(
            include_row_actions=item is not None
        )
        if menu is None:
            return
        viewport = self._manager.metadata_derivation_list.viewport()
        if viewport is None:
            return
        menu.exec(viewport.mapToGlobal(pos))

    def _copy_selected_derivation_code(self) -> None:
        code = self._manager._selected_derivation_code()
        if not code:
            return
        payload = self._selected_derivation_step_payload()
        if payload is None:
            erlab.interactive.utils.copy_to_clipboard(code)
            return
        operations, active_name, _contains_script = payload
        payload_text = json.dumps(
            {
                "type": _PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_TYPE,
                "version": _PROVENANCE_STEPS_CLIPBOARD_PAYLOAD_VERSION,
                "active_name": active_name,
                "operations": [
                    operation.model_dump(mode="json") for operation in operations
                ],
            },
            separators=(",", ":"),
        )
        mime_data = QtCore.QMimeData()
        mime_data.setData(
            _PROVENANCE_STEPS_CLIPBOARD_MIME, payload_text.encode("utf-8")
        )
        mime_data.setText(code)
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is None:
            return
        clipboard.setMimeData(mime_data)

    def _paste_provenance_steps_from_clipboard(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        payload = _provenance_step_clipboard_payload(
            None if clipboard is None else clipboard.mimeData()
        )
        if payload is None:
            self._manager._provenance_edit_controller._show_unavailable(
                "The clipboard does not contain copied ImageTool provenance steps."
            )
            return
        operations, active_name, contains_script = payload
        self._manager._provenance_edit_controller.paste_steps(
            operations,
            active_name=active_name,
            contains_script=contains_script,
        )

    def _unavailable_replay_code_details(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str:
        unavailable_labels: list[str] = []
        replayable_input_labels = self._replayable_script_input_labels(
            getattr(node, "displayed_provenance_spec", None)
        )
        rows = self._flatten_derivation_rows(
            self._current_derivation_display_rows(node, include_history=True)
        )
        for row in rows[1:]:
            entry = row.entry
            if entry.code is not None:
                continue
            label = " ".join(entry.label.split())
            if label in replayable_input_labels:
                continue
            if label and label not in unavailable_labels:
                unavailable_labels.append(label)

        if unavailable_labels:
            return "\n".join(
                (
                    "The following recorded inputs or steps are not available as "
                    "replayable code:",
                    *(f"- {label}" for label in unavailable_labels),
                )
            )
        return "The replay graph could not be emitted as Python code."

    def _replayable_script_input_labels(
        self,
        spec: provenance.ToolProvenanceSpec | None,
    ) -> set[str]:
        if spec is None or spec.kind != "script":
            return set()
        replayable_labels: set[str] = set()
        for script_input in spec.script_inputs:
            try:
                _replay_graph.script_inputs_code((script_input,), display=True)
            except _replay_graph.ReplayGraphError:
                continue
            replayable_labels.add(
                " ".join(
                    self._script_input_row_label(
                        script_input,
                        include_history=True,
                    ).split()
                )
            )
        return replayable_labels

    def _unavailable_replay_code_traceback(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        spec = node.displayed_provenance_spec
        if spec is None or spec.kind != "script" or not spec.operations:
            return None
        try:
            graph = _replay_graph.compile_replay_graph(spec, display=True)
            _replay_graph.emit_replay_code(
                graph,
                output_name=typing.cast("str", spec.active_name),
            )
        except _replay_graph.ReplayGraphError as exc:
            return "".join(traceback.TracebackException.from_exception(exc).format())
        return None

    def _unavailable_replay_code_dialog_details(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str:
        details = "<br>".join(
            html.escape(line)
            for line in self._unavailable_replay_code_details(node).splitlines()
        )
        exc_text = self._unavailable_replay_code_traceback(node)
        if exc_text is None:
            return details
        traceback_html = erlab.interactive.utils._format_traceback(exc_text)
        return f"{details}<hr>{traceback_html}"

    def _show_unavailable_replay_code_dialog(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> None:
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title="Replay Code Unavailable",
            text="Replay code cannot be copied for the selected result.",
            informative_text=(
                "The result has provenance, but at least one recorded input or step "
                "cannot be converted to replayable Python, so nothing was copied."
            ),
            detailed_text=self._unavailable_replay_code_dialog_details(node),
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            default_button=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        dialog.exec()

    def _copy_full_derivation_code(self) -> None:
        node = (
            None
            if self._manager._metadata_node_uid is None
            else self._manager._tool_graph.nodes.get(self._manager._metadata_node_uid)
        )
        if node is None or not self._manager._metadata_full_code_available:
            return
        code = node.derivation_code
        if not code:
            self._show_unavailable_replay_code_dialog(node)
            return
        if provenance.uses_default_replay_input(code):
            load_source = self._manager._load_source_for_replay(node)
            if load_source is None:
                source_name = self._manager._prompt_replay_input_name(node)
                if source_name is None:
                    return
                code = provenance.rebase_default_replay_input(code, source_name)
            else:
                source_name, load_code = load_source
                rebased_code = provenance.rebase_default_replay_input(code, source_name)
                code = "\n\n".join(part for part in (load_code, rebased_code) if part)
        if code:
            erlab.interactive.utils.copy_to_clipboard(code)

    def _edit_selected_derivation_step(self) -> None:
        self._manager._provenance_edit_controller.edit_row(
            self._manager._selected_derivation_row()
        )

    def _activate_selected_derivation_step(self) -> None:
        row = self._manager._selected_derivation_row()
        editable, _reason = self._manager._provenance_edit_controller.can_edit_row(row)
        if editable:
            self._manager._edit_selected_derivation_step()

    def _revert_selected_derivation_step(self) -> None:
        self._manager._provenance_edit_controller.revert_row(
            self._manager._selected_derivation_row()
        )

    def _delete_selected_derivation_step(self) -> None:
        self._manager._provenance_edit_controller.delete_row(
            self._manager._selected_derivation_row()
        )

    def _update_info(self, *, uid: str | None = None) -> None:
        """Update the information text box.

        If a string ``uid`` is provided, the function will update the info box only if
        the given ``uid`` is the only selected node.
        """
        selected_imagetools = self._manager._selected_imagetool_targets()
        selected_childtools = self._manager._selected_tool_uids()

        n_itool: int = len(selected_imagetools)
        n_total: int = n_itool + len(selected_childtools)

        selected_node_uids = list(selected_childtools)
        if uid is not None and n_itool == 1:
            with contextlib.suppress(KeyError):
                selected_node_uids.append(
                    self._manager._node_for_target(selected_imagetools[0]).uid
                )

        if (uid is not None) and ((n_total != 1) or (uid not in selected_node_uids)):
            return

        match n_total:
            case 0:
                self._manager.text_box.setPlainText(
                    "Select a window to view its information."
                )
                self._manager._clear_metadata()
                self._manager.preview_widget.setVisible(False)

            case 1:
                selected_target: int | str
                if n_itool > 0:
                    selected_target = selected_imagetools[0]
                else:
                    selected_target = selected_childtools[0]

                node = self._manager._node_for_target(selected_target)
                self._manager.text_box.setHtml(self._manager._node_info_html(node))
                self._manager._set_metadata_node(node)

                if node.is_imagetool:
                    if node.pending_workspace_memory_payload is not None:
                        pending_curve = node.pending_workspace_preview_curve()
                        if pending_curve is not None:
                            self._manager.preview_widget.setCurve(*pending_curve)
                        elif (
                            pending_preview := node.pending_workspace_preview_image()
                        ) is None:
                            self._manager.preview_widget.setLoadPrompt()
                        else:
                            self._manager.preview_widget.setPixmap(pending_preview[1])
                        self._manager.preview_widget.setVisible(True)
                        return
                    if (curve := _preview_curve_for_node(node)) is not None:
                        self._manager.preview_widget.setCurve(*curve)
                        self._manager.preview_widget.setVisible(True)
                        return
                    self._manager.preview_widget.setPixmap(
                        _preview_image_for_node(node)[1]
                    )
                    self._manager.preview_widget.setVisible(True)
                    return

                tool_window = node.tool_window
                if (
                    tool_window is None
                    and node.pending_workspace_tool_payload is not None
                ):
                    pending_preview = node.pending_workspace_tool_preview_image()
                    if pending_preview is None:
                        self._manager.preview_widget.setLoadPrompt()
                    else:
                        self._manager.preview_widget.setPixmap(
                            pending_preview[1],
                            aspect_ratio_mode=QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        )
                    self._manager.preview_widget.setVisible(True)
                    return
                preview_pixmap = (
                    None if tool_window is None else tool_window.preview_pixmap
                )
                if tool_window is not None and getattr(
                    tool_window, "preview_pixmap_stale", False
                ):
                    self._schedule_tool_preview_update(node.uid)
                if preview_pixmap is not None and not preview_pixmap.isNull():
                    if self._is_figure_composer_tool(tool_window):
                        self._manager.preview_widget.setPixmap(
                            preview_pixmap,
                            aspect_ratio_mode=QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        )
                    else:
                        self._manager.preview_widget.setPixmap(preview_pixmap)
                    self._manager.preview_widget.setVisible(True)
                    return

                image_item = (
                    None if tool_window is None else tool_window.preview_imageitem
                )
                if image_item is None or not erlab.interactive.utils.qt_is_valid(
                    image_item
                ):
                    self._manager.preview_widget.setVisible(False)
                    return

                try:
                    pixmap = image_item.getPixmap()
                except RuntimeError:
                    pixmap = None
                if pixmap is None or pixmap.isNull():
                    self._manager.preview_widget.setVisible(False)
                    return

                self._manager.preview_widget.setPixmap(
                    pixmap.transformed(QtGui.QTransform().scale(1.0, -1.0))
                )
                self._manager.preview_widget.setVisible(True)

            case _:
                self._manager.text_box.setHtml(
                    "<p><b>Selected ImageTool windows</b></p>"
                    + "<br>".join(
                        self._manager._node_for_target(i).display_text
                        for i in selected_imagetools
                    )
                )
                self._manager._clear_metadata()
                self._manager.preview_widget.setVisible(False)

    def _schedule_tool_metadata_update(self, uid: str) -> None:
        """Refresh expensive selected-tool metadata after bursty info updates settle."""
        if not self._tool_metadata_update_relevant(uid):
            return
        self._manager._tool_metadata_queue.schedule(uid)

    def _tool_metadata_update_relevant(self, uid: str) -> bool:
        selected_imagetools = self._manager._selected_imagetool_targets()
        selected_childtools = self._manager._selected_tool_uids()
        if len(selected_imagetools) + len(selected_childtools) != 1:
            return False
        return uid in selected_childtools or uid in selected_imagetools

    def _schedule_tool_preview_update(self, uid: str) -> None:
        self._tool_preview_update_generation += 1
        generation = self._tool_preview_update_generation
        erlab.interactive.utils.single_shot(
            self._manager,
            _TOOL_PREVIEW_UPDATE_DELAY_MS,
            lambda: self._run_scheduled_tool_preview_update(uid, generation),
        )

    def _run_scheduled_tool_preview_update(self, uid: str, generation: int) -> None:
        if generation != self._tool_preview_update_generation:
            return
        if self._manager._selected_tool_uids() != [uid]:
            return
        try:
            node = self._manager._child_node(uid)
        except KeyError:
            return
        tool_window = node.tool_window
        request = (
            None
            if tool_window is None
            else getattr(tool_window, "request_preview_pixmap_update", None)
        )
        if callable(request):
            request(delay_ms=0)

    @staticmethod
    def _is_figure_composer_tool(tool_window: object | None) -> bool:
        from erlab.interactive._figurecomposer import FigureComposerTool

        return isinstance(tool_window, FigureComposerTool)

    def _flush_pending_tool_metadata_updates(self, pending: set[str]) -> None:
        for uid in sorted(pending):
            self._manager._update_info(uid=uid)

    @QtCore.Slot()
    def _load_selected_preview_data(self) -> None:
        selected_imagetools = self._manager._selected_imagetool_targets()
        selected_tools = self._manager._selected_tool_uids()
        if len(selected_imagetools) == 1 and not selected_tools:
            node = self._manager._node_for_target(selected_imagetools[0])
            if node.pending_workspace_memory_payload is None:
                return
            if node.materialize_pending_workspace_payload():
                self._manager._update_info(uid=node.uid)
            return
        if selected_imagetools or len(selected_tools) != 1:
            return
        node = self._manager._child_node(selected_tools[0])
        if node.pending_workspace_tool_payload is None:
            return
        if node.materialize_pending_workspace_payload():
            self._manager._update_info(uid=node.uid)

    def _update_actions(self) -> None:
        """Update the state of the actions based on the current selection."""
        selection_children = self._manager._selected_tool_uids()
        imagetool_targets = self._manager._selected_imagetool_targets()
        promotable_child_uid = self._manager._selected_promotable_child_imagetool_uid()
        source_update_child_uid = self._manager._selected_source_update_child_uid()
        reload_candidates = self._manager._selected_reload_candidates()

        selection_watched: list[int] = []
        selection_offloadable: list[int | str] = []

        for target in imagetool_targets:
            node = self._manager._node_for_target(target)
            if isinstance(node, _ImageToolWrapper) and node.watched:
                selection_watched.append(node.index)
            if (
                node.imagetool is not None
                and node.is_imagetool
                and not node.slicer_area.data_chunked
            ):
                selection_offloadable.append(target)

        something_selected = bool(imagetool_targets or selection_children)
        root_imagetool_count = len(self._manager.tree_view.selected_imagetool_indices)
        total_selected = len(imagetool_targets) + len(selection_children)
        single_selected = total_selected == 1
        multiple_root_imagetools_selected = (
            root_imagetool_count > 1 and root_imagetool_count == total_selected
        )
        multiple_selected = len(imagetool_targets) > 1

        self._manager.show_action.setEnabled(something_selected)
        self._manager.hide_action.setEnabled(something_selected)
        self._manager.remove_action.setEnabled(something_selected)
        self._manager.rename_action.setEnabled(
            single_selected or multiple_root_imagetools_selected
        )
        self._manager.duplicate_action.setEnabled(something_selected)
        self._manager.promote_action.setEnabled(promotable_child_uid is not None)
        self._manager.offload_action.setEnabled(
            bool(imagetool_targets)
            and len(selection_children) == 0
            and len(selection_offloadable) == len(imagetool_targets)
            and not self._manager._workspace_state.save_in_progress
        )
        self._manager.concat_action.setEnabled(
            multiple_selected and len(selection_children) == 0
        )
        self._manager.batch_action.setEnabled(self._manager.batch_target_count() >= 2)
        self._manager.create_figure_action.setEnabled(
            bool(self._manager._selected_figure_source_targets())
        )
        self._manager.store_action.setEnabled(
            bool(self._manager.tree_view.selected_imagetool_indices)
        )

        reload_relevant = reload_candidates is not None
        self._manager.reload_action.setVisible(reload_relevant)
        self._manager.reload_action.setEnabled(reload_relevant)
        reload_tooltip = "Reload selected data from its saved files, parent, or inputs"
        if reload_candidates is not None and reload_candidates[2] is not None:
            reload_tooltip = reload_candidates[2]
        self._manager.reload_action.setToolTip(reload_tooltip)
        self._manager.unwatch_action.setVisible(
            bool(imagetool_targets)
            and len(selection_watched) == len(imagetool_targets)
            and len(selection_children) == 0
            and all(
                isinstance(self._manager._node_for_target(s), _ImageToolWrapper)
                for s in imagetool_targets
            )
        )
        self._manager.source_update_action.setVisible(
            source_update_child_uid is not None
        )
        self._manager.source_update_action.setEnabled(
            source_update_child_uid is not None
        )
        self._update_note_actions()

        if not imagetool_targets or selection_children:
            self._manager.link_action.setDisabled(True)
            self._manager.unlink_action.setDisabled(True)
            return

        nodes = []
        for index in imagetool_targets:
            node = self._manager._node_for_target(index)
            if node.is_imagetool:
                nodes.append(node)
        self._manager.link_action.setDisabled(len(nodes) <= 1)
        is_linked = [node.workspace_linked for node in nodes]
        self._manager.unlink_action.setEnabled(any(is_linked))

        if len(nodes) > 1 and all(is_linked):
            link_keys = [node.workspace_link_key for node in nodes]
            proxies = [
                node.slicer_area._linking_proxy
                for node in nodes
                if node.imagetool is not None
            ]
            if (
                link_keys[0] is not None
                and all(key == link_keys[0] for key in link_keys)
            ) or (
                proxies
                and len(proxies) == len(nodes)
                and all(proxy == proxies[0] for proxy in proxies)
            ):
                self._manager.link_action.setEnabled(False)
