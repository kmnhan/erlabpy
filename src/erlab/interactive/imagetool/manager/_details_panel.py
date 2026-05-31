from __future__ import annotations

import logging
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive.imagetool import provenance_framework
from erlab.interactive.imagetool.manager._base import _ImageToolManagerBase
from erlab.interactive.imagetool.manager._widgets import (
    _METADATA_DERIVATION_CODE_ROLE,
    _METADATA_DERIVATION_COPYABLE_ROLE,
    _ElidedInteractiveLabel,
    _LoadSourceDetailsDialog,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
    _MetadataField,
)

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool._load_source import _LoadSourceDetails

logger = logging.getLogger(__name__)


class _DetailsPanelMixin(_ImageToolManagerBase):
    def _node_info_html(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        return node.info_text

    def _clear_metadata(self) -> None:
        self._metadata_full_code_available = False
        self._metadata_node_uid = None
        with QtCore.QSignalBlocker(self.metadata_derivation_list):
            self.metadata_derivation_list.clear()
        self._set_metadata_fields([])
        self._update_metadata_pane()

    def _set_metadata_node(self, node: _ImageToolWrapper | _ManagedWindowNode) -> None:
        self._metadata_full_code_available = node.displayed_provenance_spec is not None
        self._metadata_node_uid = node.uid
        self._set_metadata_fields(node.metadata_fields)

        with QtCore.QSignalBlocker(self.metadata_derivation_list):
            self.metadata_derivation_list.clear()
            for entry in node.derivation_entries:
                item = QtWidgets.QListWidgetItem(entry.label)
                item.setToolTip(entry.label)
                item.setData(_METADATA_DERIVATION_CODE_ROLE, entry.code)
                item.setData(_METADATA_DERIVATION_COPYABLE_ROLE, entry.copyable)
                if not entry.copyable:
                    item.setForeground(
                        self.metadata_derivation_list.palette().color(
                            QtGui.QPalette.ColorGroup.Disabled,
                            QtGui.QPalette.ColorRole.Text,
                        )
                    )
                    if entry.code is None and not entry.label.startswith("Start from "):
                        item.setToolTip("Replay code is unavailable for this step.")
                self.metadata_derivation_list.addItem(item)
        self._update_metadata_pane()

    def _set_metadata_fields(self, fields: list[_MetadataField]) -> None:
        while self.metadata_details_layout.count():
            item = self.metadata_details_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._metadata_detail_labels.clear()

        for row, field in enumerate(fields):
            key_label = QtWidgets.QLabel(field.label, self.metadata_details_widget)
            key_label.setForegroundRole(QtGui.QPalette.ColorRole.Text)
            key_label.setEnabled(False)
            key_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
            )
            value_label: QtWidgets.QLabel
            if field.details is not None:
                value_label = _ElidedInteractiveLabel(
                    field.value,
                    self.metadata_details_widget,
                )
                value_label.setForegroundRole(QtGui.QPalette.ColorRole.Link)
                value_label.set_full_text(field.value)
                value_label.clicked.connect(
                    lambda d=field.details: self._show_load_source_details(d)
                )
            else:
                value_label = QtWidgets.QLabel(
                    field.value, self.metadata_details_widget
                )
                value_label.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                )
                value_label.setWordWrap(field.wrap)
                value_label.setToolTip(field.value)
                value_label.setMinimumWidth(0)
                value_label.setAlignment(
                    QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
                )
                if field.wrap:
                    size_policy = QtWidgets.QSizePolicy(
                        QtWidgets.QSizePolicy.Policy.Expanding,
                        QtWidgets.QSizePolicy.Policy.Preferred,
                    )
                    size_policy.setHeightForWidth(True)
                    value_label.setSizePolicy(size_policy)
            if field.monospace:
                value_label.setFont(self._metadata_monospace_font)
            self.metadata_details_layout.addWidget(key_label, row, 0)
            self.metadata_details_layout.addWidget(value_label, row, 1)
            self._metadata_detail_labels[field.label] = value_label

    def _show_load_source_details(self, details: _LoadSourceDetails) -> None:
        _LoadSourceDetailsDialog(details, self).exec()

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
            current = self._parent_node(current)

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

        dialog = QtWidgets.QInputDialog(self)
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
        has_details = bool(self._metadata_detail_labels)
        derivation_count = self.metadata_derivation_list.count()

        self.metadata_group.setVisible(has_details or derivation_count > 0)
        self.metadata_details_widget.setVisible(has_details)
        self.metadata_derivation_list.setVisible(derivation_count > 0)

        if derivation_count == 0:
            self.metadata_derivation_list.setMinimumHeight(0)
            self.metadata_derivation_list.setMaximumHeight(0)
        else:
            row_height = self.metadata_derivation_list.sizeHintForRow(0)
            if row_height <= 0:
                row_height = self.fontMetrics().height() + 8
            visible_rows = min(derivation_count, 4)
            frame = self.metadata_derivation_list.frameWidth() * 2
            height = visible_rows * row_height + frame + 4
            self.metadata_derivation_list.setMinimumHeight(height)
            self.metadata_derivation_list.setMaximumHeight(height)

        self.metadata_details_widget.updateGeometry()
        self.metadata_derivation_list.updateGeometry()
        self.metadata_details_widget.sync_height_for_width()
        self.metadata_group.sync_height_for_width()
        self.metadata_group.updateGeometry()

    def _selected_derivation_items(self) -> list[QtWidgets.QListWidgetItem]:
        items = list(self.metadata_derivation_list.selectedItems())
        if not items:
            current_item = self.metadata_derivation_list.currentItem()
            if current_item is not None:
                items = [current_item]
        return sorted(items, key=self.metadata_derivation_list.row)

    def _selected_derivation_code(self) -> str | None:
        codes: list[str] = []
        for item in self._selected_derivation_items():
            if not bool(item.data(_METADATA_DERIVATION_COPYABLE_ROLE)):
                continue
            code = typing.cast("str | None", item.data(_METADATA_DERIVATION_CODE_ROLE))
            if code:
                codes.append(code)
        if not codes:
            return None
        return "\n".join(codes)

    def _build_metadata_derivation_menu(self) -> QtWidgets.QMenu | None:
        if self.metadata_derivation_list.count() == 0:
            return None

        menu = QtWidgets.QMenu(self.metadata_derivation_list)
        selected_code = self._selected_derivation_code()
        self._metadata_copy_selected_action.setEnabled(bool(selected_code))
        menu.addAction(self._metadata_copy_selected_action)
        if self._metadata_full_code_available:
            self._metadata_copy_full_action.setEnabled(True)
            menu.addAction(self._metadata_copy_full_action)
        return menu

    def _show_metadata_derivation_menu(self, pos: QtCore.QPoint) -> None:
        if self.metadata_derivation_list.itemAt(pos) is None:
            return
        menu = self._build_metadata_derivation_menu()
        if menu is None:
            return
        viewport = self.metadata_derivation_list.viewport()
        if viewport is None:
            return
        menu.exec(viewport.mapToGlobal(pos))

    def _copy_selected_derivation_code(self) -> None:
        code = self._selected_derivation_code()
        if code:
            erlab.interactive.utils.copy_to_clipboard(code)

    def _copy_full_derivation_code(self) -> None:
        node = (
            None
            if self._metadata_node_uid is None
            else self._tool_graph.nodes.get(self._metadata_node_uid)
        )
        if node is None or not self._metadata_full_code_available:
            return
        code = node.derivation_code
        if not code:
            self._status_bar.showMessage(
                "Replay code is unavailable for this result", 5000
            )
            return
        provenance = provenance_framework
        if provenance.uses_default_replay_input(code):
            load_source = self._load_source_for_replay(node)
            if load_source is None:
                source_name = self._prompt_replay_input_name(node)
                if source_name is None:
                    return
                code = provenance.rebase_default_replay_input(code, source_name)
            else:
                source_name, load_code = load_source
                rebased_code = provenance.rebase_default_replay_input(code, source_name)
                code = "\n\n".join(part for part in (load_code, rebased_code) if part)
        if code:
            erlab.interactive.utils.copy_to_clipboard(code)

    def _update_info(self, *, uid: str | None = None) -> None:
        """Update the information text box.

        If a string ``uid`` is provided, the function will update the info box only if
        the given ``uid`` is the only selected child tool.
        """
        selected_imagetools = self._selected_imagetool_targets()
        selected_childtools = self._selected_tool_uids()

        n_itool: int = len(selected_imagetools)
        n_total: int = n_itool + len(selected_childtools)

        selected_child_ids = list(selected_childtools)
        if uid is not None and n_itool == 1:
            target = selected_imagetools[0]
            if isinstance(target, str):
                selected_child_ids.append(target)

        if (uid is not None) and ((n_total != 1) or (uid not in selected_child_ids)):
            return

        match n_total:
            case 0:
                self.text_box.setPlainText("Select a window to view its information.")
                self._clear_metadata()
                self.preview_widget.setVisible(False)

            case 1:
                selected_target: int | str
                if n_itool > 0:
                    selected_target = selected_imagetools[0]
                else:
                    selected_target = selected_childtools[0]

                node = self._node_for_target(selected_target)
                self.text_box.setHtml(self._node_info_html(node))
                self._set_metadata_node(node)

                if node.is_imagetool:
                    self.preview_widget.setPixmap(node._preview_image[1])
                    self.preview_widget.setVisible(True)
                    return

                image_item = (
                    None
                    if node.tool_window is None
                    else node.tool_window.preview_imageitem
                )
                if image_item is None:
                    self.preview_widget.setVisible(False)
                else:
                    self.preview_widget.setPixmap(
                        image_item.getPixmap().transformed(
                            QtGui.QTransform().scale(1.0, -1.0)
                        )
                    )
                    self.preview_widget.setVisible(True)

            case _:
                self.text_box.setHtml(
                    "<p><b>Selected ImageTool windows</b></p>"
                    + "<br>".join(
                        self._node_for_target(i).display_text
                        for i in selected_imagetools
                    )
                )
                self._clear_metadata()
                self.preview_widget.setVisible(False)

    def _schedule_tool_metadata_update(self, uid: str) -> None:
        """Refresh expensive selected-tool metadata after bursty info updates settle."""
        self._pending_tool_metadata_update_uids.add(uid)
        self._tool_metadata_update_timer.start()

    def _flush_pending_tool_metadata_updates(self) -> None:
        pending = self._pending_tool_metadata_update_uids
        self._pending_tool_metadata_update_uids = set()
        for uid in sorted(pending):
            self._update_info(uid=uid)

    def _update_actions(self) -> None:
        """Update the state of the actions based on the current selection."""
        selection_children = self._selected_tool_uids()
        imagetool_targets = self._selected_imagetool_targets()
        promotable_child_uid = self._selected_promotable_child_imagetool_uid()
        source_update_child_uid = self._selected_source_update_child_uid()
        reload_targets = self._selected_reload_targets()

        selection_watched: list[int] = []
        selection_offloadable: list[int | str] = []

        for target in imagetool_targets:
            node = self._node_for_target(target)
            if isinstance(node, _ImageToolWrapper) and node.watched:
                selection_watched.append(node.index)
            if (
                node.imagetool is not None
                and node.is_imagetool
                and not node.slicer_area.data_chunked
            ):
                selection_offloadable.append(target)

        something_selected = bool(imagetool_targets or selection_children)
        root_imagetool_count = len(self.tree_view.selected_imagetool_indices)
        total_selected = len(imagetool_targets) + len(selection_children)
        single_selected = total_selected == 1
        multiple_root_imagetools_selected = (
            root_imagetool_count > 1 and root_imagetool_count == total_selected
        )
        multiple_selected = len(imagetool_targets) > 1

        self.show_action.setEnabled(something_selected)
        self.hide_action.setEnabled(something_selected)
        self.remove_action.setEnabled(something_selected)
        self.rename_action.setEnabled(
            single_selected or multiple_root_imagetools_selected
        )
        self.duplicate_action.setEnabled(something_selected)
        self.promote_action.setEnabled(promotable_child_uid is not None)
        self.offload_action.setEnabled(
            bool(imagetool_targets)
            and len(selection_children) == 0
            and len(selection_offloadable) == len(imagetool_targets)
        )
        self.concat_action.setEnabled(
            multiple_selected and len(selection_children) == 0
        )
        self.store_action.setEnabled(bool(self.tree_view.selected_imagetool_indices))

        reload_available = reload_targets is not None
        self.reload_action.setVisible(reload_available)
        self.reload_action.setEnabled(reload_available)
        self.unwatch_action.setVisible(
            bool(imagetool_targets)
            and len(selection_watched) == len(imagetool_targets)
            and len(selection_children) == 0
            and all(
                isinstance(self._node_for_target(s), _ImageToolWrapper)
                for s in imagetool_targets
            )
        )
        self.source_update_action.setVisible(source_update_child_uid is not None)
        self.source_update_action.setEnabled(source_update_child_uid is not None)

        if not imagetool_targets or selection_children:
            self.link_action.setDisabled(True)
            self.unlink_action.setDisabled(True)
            return

        self.link_action.setDisabled(len(imagetool_targets) <= 1)
        is_linked = [
            self.get_imagetool(index).slicer_area.is_linked
            for index in imagetool_targets
        ]
        self.unlink_action.setEnabled(any(is_linked))

        if len(imagetool_targets) > 1 and all(is_linked):
            proxies = [
                self.get_imagetool(index).slicer_area._linking_proxy
                for index in imagetool_targets
            ]
            if all(p == proxies[0] for p in proxies):  # pragma: no branch
                self.link_action.setEnabled(False)
