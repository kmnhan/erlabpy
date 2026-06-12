from __future__ import annotations

import logging
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive.imagetool import provenance
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
    _preview_image_for_node,
)

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool._load_source import _LoadSourceDetails
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager

logger = logging.getLogger(__name__)

_TOOL_PREVIEW_UPDATE_DELAY_MS = 250


class _DetailsPanelController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager
        self._tool_preview_update_generation = 0

    def _node_info_html(self, node: _ImageToolWrapper | _ManagedWindowNode) -> str:
        return node.info_text

    def _clear_metadata(self) -> None:
        self._manager._metadata_full_code_available = False
        self._manager._metadata_node_uid = None
        with QtCore.QSignalBlocker(self._manager.metadata_derivation_list):
            self._manager.metadata_derivation_list.clear()
        self._manager._set_metadata_fields([])
        self._manager._update_metadata_pane()

    def _set_metadata_node(self, node: _ImageToolWrapper | _ManagedWindowNode) -> None:
        self._manager._metadata_full_code_available = (
            node.displayed_provenance_spec is not None
        )
        self._manager._metadata_node_uid = node.uid
        self._manager._set_metadata_fields(node.metadata_fields)

        with QtCore.QSignalBlocker(self._manager.metadata_derivation_list):
            self._manager.metadata_derivation_list.clear()
            for entry in node.derivation_entries:
                item = QtWidgets.QListWidgetItem(entry.label)
                item.setToolTip(entry.label)
                item.setData(_METADATA_DERIVATION_CODE_ROLE, entry.code)
                item.setData(_METADATA_DERIVATION_COPYABLE_ROLE, entry.copyable)
                if not entry.copyable:
                    item.setForeground(
                        self._manager.metadata_derivation_list.palette().color(
                            QtGui.QPalette.ColorGroup.Disabled,
                            QtGui.QPalette.ColorRole.Text,
                        )
                    )
                    if entry.code is None and not entry.label.startswith("Start from "):
                        item.setToolTip("Replay code is unavailable for this step.")
                self._manager.metadata_derivation_list.addItem(item)
        self._manager._update_metadata_pane()

    def _set_metadata_fields(self, fields: list[_MetadataField]) -> None:
        while self._manager.metadata_details_layout.count():
            item = self._manager.metadata_details_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._manager._metadata_detail_labels.clear()

        for row, field in enumerate(fields):
            key_label = QtWidgets.QLabel(
                field.label, self._manager.metadata_details_widget
            )
            key_label.setForegroundRole(QtGui.QPalette.ColorRole.Text)
            key_label.setEnabled(False)
            key_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
            )
            value_label: QtWidgets.QLabel
            if field.details is not None:
                value_label = _ElidedInteractiveLabel(
                    field.value,
                    self._manager.metadata_details_widget,
                )
                value_label.setForegroundRole(QtGui.QPalette.ColorRole.Link)
                value_label.set_full_text(field.value)
                value_label.clicked.connect(
                    lambda d=field.details: self._manager._show_load_source_details(d)
                )
            else:
                value_label = QtWidgets.QLabel(
                    field.value, self._manager.metadata_details_widget
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
                size_policy = QtWidgets.QSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding
                    if field.wrap
                    else QtWidgets.QSizePolicy.Policy.Ignored,
                    QtWidgets.QSizePolicy.Policy.Preferred,
                )
                size_policy.setHeightForWidth(field.wrap)
                value_label.setSizePolicy(size_policy)
            if field.monospace:
                value_label.setFont(self._manager._metadata_monospace_font)
            self._manager.metadata_details_layout.addWidget(key_label, row, 0)
            self._manager.metadata_details_layout.addWidget(value_label, row, 1)
            self._manager._metadata_detail_labels[field.label] = value_label

    def _show_load_source_details(self, details: _LoadSourceDetails) -> None:
        _LoadSourceDetailsDialog(details, self._manager).exec()

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

        self._manager.metadata_group.setVisible(has_details or derivation_count > 0)
        self._manager.metadata_details_widget.setVisible(has_details)
        self._manager.metadata_derivation_list.setVisible(derivation_count > 0)

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
            self._manager.metadata_derivation_list.setMaximumHeight(height)

        self._manager.metadata_details_widget.updateGeometry()
        self._manager.metadata_derivation_list.updateGeometry()
        self._manager.metadata_details_widget.sync_height_for_width()
        self._manager.metadata_group.sync_height_for_width()
        self._manager.metadata_group.updateGeometry()

    def _selected_derivation_items(self) -> list[QtWidgets.QListWidgetItem]:
        items = list(self._manager.metadata_derivation_list.selectedItems())
        if not items:
            current_item = self._manager.metadata_derivation_list.currentItem()
            if current_item is not None:
                items = [current_item]
        return sorted(items, key=self._manager.metadata_derivation_list.row)

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

    def _build_metadata_derivation_menu(self) -> QtWidgets.QMenu | None:
        if self._manager.metadata_derivation_list.count() == 0:
            return None

        menu = QtWidgets.QMenu(self._manager.metadata_derivation_list)
        selected_code = self._manager._selected_derivation_code()
        self._manager._metadata_copy_selected_action.setEnabled(bool(selected_code))
        menu.addAction(self._manager._metadata_copy_selected_action)
        if self._manager._metadata_full_code_available:
            self._manager._metadata_copy_full_action.setEnabled(True)
            menu.addAction(self._manager._metadata_copy_full_action)
        return menu

    def _show_metadata_derivation_menu(self, pos: QtCore.QPoint) -> None:
        if self._manager.metadata_derivation_list.itemAt(pos) is None:
            return
        menu = self._manager._build_metadata_derivation_menu()
        if menu is None:
            return
        viewport = self._manager.metadata_derivation_list.viewport()
        if viewport is None:
            return
        menu.exec(viewport.mapToGlobal(pos))

    def _copy_selected_derivation_code(self) -> None:
        code = self._manager._selected_derivation_code()
        if code:
            erlab.interactive.utils.copy_to_clipboard(code)

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
            self._manager._status_bar.showMessage(
                "Replay code is unavailable for this result", 5000
            )
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

    def _update_info(self, *, uid: str | None = None) -> None:
        """Update the information text box.

        If a string ``uid`` is provided, the function will update the info box only if
        the given ``uid`` is the only selected child tool.
        """
        selected_imagetools = self._manager._selected_imagetool_targets()
        selected_childtools = self._manager._selected_tool_uids()

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
                    self._manager.preview_widget.setPixmap(
                        _preview_image_for_node(node)[1]
                    )
                    self._manager.preview_widget.setVisible(True)
                    return

                tool_window = node.tool_window
                preview_pixmap = (
                    None if tool_window is None else tool_window.preview_pixmap
                )
                if tool_window is not None and getattr(
                    tool_window, "preview_pixmap_stale", False
                ):
                    self._schedule_tool_preview_update(node.uid)
                if preview_pixmap is not None and not preview_pixmap.isNull():
                    self._manager.preview_widget.setPixmap(preview_pixmap)
                    self._manager.preview_widget.setVisible(True)
                    return

                image_item = (
                    None if tool_window is None else tool_window.preview_imageitem
                )
                if image_item is None:
                    self._manager.preview_widget.setVisible(False)
                else:
                    self._manager.preview_widget.setPixmap(
                        image_item.getPixmap().transformed(
                            QtGui.QTransform().scale(1.0, -1.0)
                        )
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
        self._manager._tool_metadata_queue.schedule(uid)

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

    def _flush_pending_tool_metadata_updates(self, pending: set[str]) -> None:
        for uid in sorted(pending):
            self._manager._update_info(uid=uid)

    def _update_actions(self) -> None:
        """Update the state of the actions based on the current selection."""
        selection_children = self._manager._selected_tool_uids()
        imagetool_targets = self._manager._selected_imagetool_targets()
        promotable_child_uid = self._manager._selected_promotable_child_imagetool_uid()
        source_update_child_uid = self._manager._selected_source_update_child_uid()
        reload_targets = self._manager._selected_reload_targets()

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

        reload_available = reload_targets is not None
        self._manager.reload_action.setVisible(reload_available)
        self._manager.reload_action.setEnabled(reload_available)
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

        if not imagetool_targets or selection_children:
            self._manager.link_action.setDisabled(True)
            self._manager.unlink_action.setDisabled(True)
            return

        self._manager.link_action.setDisabled(len(imagetool_targets) <= 1)
        is_linked = [
            self._manager.get_imagetool(index).slicer_area.is_linked
            for index in imagetool_targets
        ]
        self._manager.unlink_action.setEnabled(any(is_linked))

        if len(imagetool_targets) > 1 and all(is_linked):
            proxies = [
                self._manager.get_imagetool(index).slicer_area._linking_proxy
                for index in imagetool_targets
            ]
            if all(p == proxies[0] for p in proxies):  # pragma: no branch
                self._manager.link_action.setEnabled(False)
