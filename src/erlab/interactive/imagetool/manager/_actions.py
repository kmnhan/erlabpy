from __future__ import annotations

import contextlib
import html
import logging
import pathlib
import traceback
import typing
import uuid

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive.imagetool import _kspace_conversion
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool._provenance._model import (
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    compose_display_provenance,
    compose_full_provenance,
)
from erlab.interactive.imagetool._provenance._operations import (
    AssignCoord1DOperation,
    AssignScalarCoordOperation,
    ImageToolSelectionSourceBinding,
    RestoreNonuniformDimsOperation,
)
from erlab.interactive.imagetool.manager._dialogs import (
    _BatchOperationDialog,
    _ConcatDialog,
    _RenameDialog,
    _StoreDialog,
)
from erlab.interactive.imagetool.manager._widgets import _WATCHED_VAR_COLORS
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    import datetime
    from collections.abc import Iterable, Mapping

    import xarray as xr

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.viewer import ImageSlicerArea

logger = logging.getLogger(__name__)


class _DuplicateMaterializationError(RuntimeError):
    """A workspace payload already reported its materialization failure."""


class _ActionsController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager
        self.__rename_dialog: _RenameDialog | None = None
        self.__concat_dialog: _ConcatDialog | None = None
        self.__batch_dialog: _BatchOperationDialog | None = None

    @property
    def _rename_dialog(self) -> _RenameDialog:
        if self.__rename_dialog is None:
            self.__rename_dialog = _RenameDialog(self._manager)
        return self.__rename_dialog

    def rename_selected(self) -> None:
        """Rename selected ImageTool windows."""
        selected_images = self._manager._selected_imagetool_targets()
        selected_tools = self._manager._selected_tool_uids()
        if len(selected_images) + len(selected_tools) == 1:
            target = selected_images[0] if selected_images else selected_tools[0]
            if isinstance(target, str) and self._manager._is_figure_uid(target):
                pane = self._manager._figure_collection.pane
                if pane is None:
                    return
                item = self._manager._figure_collection.item_for_uid(target)
                if item is not None:
                    pane.list_widget.editItem(item)
                return
            self._manager.tree_view.edit(
                self._manager.tree_view._model._row_index(target)
            )
            return

        if selected_tools or any(
            not isinstance(target, int) for target in selected_images
        ):
            return

        dlg = self._rename_dialog
        root_selected = typing.cast("list[int]", selected_images)
        dlg.set_names(
            root_selected,
            [self._manager._tool_graph.root_wrappers[i].name for i in root_selected],
        )
        dlg.open()

    def duplicate_selected(self) -> None:
        """Duplicate selected windows."""
        indices = list(self._manager._selected_imagetool_targets())
        child_uids = list(self._manager._selected_tool_uids())
        self._manager.tree_view.deselect_all()

        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self._manager.tree_view.selectionModel()
        )
        try:
            for index in indices:
                new_index = self._manager.duplicate_imagetool(index)

                qmodelindex = self._manager.tree_view._model._row_index(new_index)

                selection_model.select(
                    QtCore.QItemSelection(qmodelindex, qmodelindex),
                    QtCore.QItemSelectionModel.SelectionFlag.Select,
                )

            for uid in child_uids:
                new_uid = self._manager.duplicate_childtool(uid)

                if self._manager._is_figure_uid(new_uid):
                    self._manager._figure_collection.select_uid(new_uid)
                    continue

                qmodelindex = self._manager.tree_view._model._row_index(new_uid)

                selection_model.select(
                    QtCore.QItemSelection(qmodelindex, qmodelindex),
                    QtCore.QItemSelectionModel.SelectionFlag.Select,
                )
        except _DuplicateMaterializationError:
            return
        except Exception:
            self._manager._show_operation_error(
                "Error while duplicating selected windows",
                "An error occurred while duplicating the selected window.",
            )

    def promote_selected(self) -> None:
        """Promote the selected nested ImageTool to a top-level window."""
        uid = self._manager._selected_promotable_child_imagetool_uid()
        if uid is None:
            return

        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Promote selected ImageTool to a top-level window?")
        msg_box.setInformativeText(
            "This will detach the ImageTool from its parent. Live update linkage and "
            "automatic updates from the parent will be removed, but existing "
            "provenance and derivation metadata will be retained as detached history."
        )
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)

        if msg_box.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self._manager.promote_child_imagetool(uid)

    def promote_child_imagetool(self, uid: str) -> int:
        """Promote the nested ImageTool identified by ``uid`` to a top-level row."""
        self._manager._commit_note_editor()
        node = self._manager._child_node(uid)
        if not node.is_imagetool:
            raise KeyError(f"Target {uid!r} is not an ImageTool")
        if not node.materialize_pending_workspace_payload():
            raise RuntimeError(
                "Could not read this ImageTool's saved data from the workspace file."
            )

        row_index = self._manager.tree_view._model._row_index(uid)
        was_expanded = row_index.isValid() and self._manager.tree_view.isExpanded(
            row_index
        )

        promoted_window = node.take_window()
        if not isinstance(promoted_window, ImageTool):
            raise TypeError(f"Unable to detach ImageTool window for {uid!r}")

        childtool_indices = list(node._childtool_indices)
        childtools = dict(node._childtools)
        created_time = node.created_time
        recent_geometry = node._recent_geometry
        persistence = node.persistence_view()
        provenance_spec = persistence.provenance_spec
        snapshot_token = node.snapshot_token
        source_snapshot_token = node.source_snapshot_token
        note = node.note

        self._manager.tree_view.childtool_removed(uid)
        self._manager._unregister_node(uid)

        with self._manager._workspace_load_context():
            new_index = self._manager.add_imagetool(
                promoted_window,
                show=False,
                uid=uid,
                provenance_spec=provenance_spec,
                snapshot_token=snapshot_token,
                source_snapshot_token=source_snapshot_token,
                created_time=created_time,
                note=note,
            )
        wrapper = self._manager._tool_graph.root_wrappers[new_index]
        wrapper._recent_geometry = recent_geometry
        self._manager._tool_graph.replace_child_references(
            wrapper.uid, childtool_indices, childtools
        )
        if wrapper.imagetool is not None:
            wrapper.imagetool.setWindowTitle(wrapper.label_text)
        node.deleteLater()

        promoted_index = self._manager.tree_view._model._row_index(new_index)
        if was_expanded:
            self._manager.tree_view.expand(promoted_index)
        self._manager.tree_view.deselect_all()
        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self._manager.tree_view.selectionModel()
        )
        selection_model.select(
            QtCore.QItemSelection(promoted_index, promoted_index),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )
        self._manager.tree_view.setCurrentIndex(promoted_index)
        self._manager.tree_view.scrollTo(promoted_index)
        self._manager.tree_view.refresh(new_index)
        self._manager._refresh_dependency_dependents(uid)
        self._manager._update_actions()
        self._manager._mark_workspace_structure_dirty("Promoted child ImageTool")
        return new_index

    def link_selected(self, link_colors: bool = True, deselect: bool = True) -> None:
        """Link selected ImageTool windows."""
        self._manager.unlink_selected(deselect=False)
        self._manager.link_imagetools(
            *self._manager._selected_imagetool_targets(), link_colors=link_colors
        )
        if deselect:
            self._manager.tree_view.deselect_all()

    def _prune_workspace_link_groups(
        self, link_keys: Iterable[str], dirty_uids: set[str]
    ) -> None:
        dirty_uids.update(
            self._manager._clear_singleton_workspace_link_groups(link_keys)
        )

    def unlink_imagetool_nodes(
        self, nodes: Iterable[_ImageToolWrapper | _ManagedWindowNode]
    ) -> None:
        """Unlink ImageTool nodes without materializing pending payloads."""
        dirty_uids: set[str] = set()
        touched_link_keys: list[str] = []
        for node in nodes:
            if not node.is_imagetool:
                raise KeyError(f"Node {node.uid!r} is not an ImageTool")
            link_key = node.workspace_link_key
            if link_key is not None:
                touched_link_keys.append(link_key)
            if node.workspace_linked:
                dirty_uids.add(node.uid)
            if node.imagetool is not None:
                node.slicer_area.unlink()
            node.clear_workspace_link_state()

        self._prune_workspace_link_groups(touched_link_keys, dirty_uids)
        for uid in sorted(dirty_uids):
            self._manager._mark_node_state_dirty(uid)
        self._manager._sigReloadLinkers.emit()

    def unlink_selected(self, deselect: bool = True) -> None:
        """Unlink selected ImageTool windows."""
        nodes: list[_ImageToolWrapper | _ManagedWindowNode] = []
        for index in self._manager._selected_imagetool_targets():
            node = self._manager._node_for_target(index)
            if not node.is_imagetool:
                raise KeyError(f"Target {index!r} is not an ImageTool")
            nodes.append(node)
        self.unlink_imagetool_nodes(nodes)
        if deselect:
            self._manager.tree_view.deselect_all()

    def offload_selected_to_workspace(self) -> None:
        """Replace selected in-memory data with dask-backed workspace data.

        .. versionadded:: 3.23.0
        """
        self._manager.offload_to_workspace(self._manager._selected_imagetool_targets())

    @property
    def _concat_dialog(self) -> _ConcatDialog:
        if self.__concat_dialog is None:
            self.__concat_dialog = _ConcatDialog(self._manager)
        return self.__concat_dialog

    def concat_selected(self) -> None:
        """Concatenate the selected data using :func:`xarray.concat`."""
        dlg = self._concat_dialog
        dlg.open()

    @property
    def _batch_dialog(self) -> _BatchOperationDialog:
        if self.__batch_dialog is None:
            self.__batch_dialog = _BatchOperationDialog(self._manager)
        return self.__batch_dialog

    def batch_target_count(self) -> int:
        return sum(
            1 for node in self._manager._tool_graph.nodes.values() if node.is_imagetool
        )

    def show_batch_operations(self) -> None:
        if self._manager.batch_target_count() < 2:
            return
        dialog = self._batch_dialog
        dialog.open()
        dialog.raise_()
        dialog.activateWindow()

    def _selected_batch_targets(self) -> list[int | str]:
        targets = list(self._manager._selected_imagetool_targets())
        if len(targets) < 2:
            raise ValueError("Select at least two ImageTool targets.")
        for target in targets:
            if not self._manager._is_imagetool_target(target):
                raise TypeError(f"Target {target!r} is not an ImageTool.")
        return targets

    def _target_display_text(self, target: int | str) -> str:
        return self._manager._node_for_target(target).display_text

    def _validate_batch_operations(
        self,
        dialog: typing.Any,
        operations: Iterable[ToolProvenanceOperation],
        *,
        extra_types: tuple[
            type[ToolProvenanceOperation],
            ...,
        ] = (),
    ) -> None:
        declared_types = tuple(dialog.operation_types)
        accepted_types = declared_types + extra_types
        for operation in operations:
            if accepted_types and not isinstance(operation, accepted_types):
                raise TypeError(
                    f"{type(dialog).__name__} produced unsupported batch operation "
                    f"{type(operation).__name__}."
                )
            if not type(operation).batch_available:
                raise TypeError(
                    f"{type(operation).__name__} is not available for batch use."
                )

    def _validate_batch_target_operations(
        self,
        operations: Iterable[ToolProvenanceOperation],
        data: xr.DataArray,
    ) -> None:
        for operation in operations:
            if not isinstance(
                operation,
                (
                    AssignScalarCoordOperation,
                    AssignCoord1DOperation,
                ),
            ):
                continue
            if operation.coord_name in data.dims or operation.coord_name in data.coords:
                raise ValueError(
                    f"Coordinate or dimension {operation.coord_name!r} already exists."
                )

    def _show_batch_error(
        self,
        text: str,
        *,
        target: int | str | None = None,
        error: BaseException | None = None,
    ) -> None:
        if isinstance(error, _kspace_conversion.KspaceConversionMemoryError):
            details = _kspace_conversion.kspace_conversion_memory_dialog_details(
                error.estimate
            )
            if target is not None:
                details = (
                    f"Target: {html.escape(self._target_display_text(target))}<br>"
                    f"{details}"
                )
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                _kspace_conversion.kspace_conversion_memory_dialog_title(),
                _kspace_conversion.kspace_conversion_memory_dialog_text(),
                informative_text=(
                    _kspace_conversion.kspace_conversion_memory_dialog_info(
                        error.estimate
                    )
                ),
                detailed_text=details,
                buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            )
            return

        details = ""
        if target is not None:
            details = f"Target: {self._target_display_text(target)}"
        if error is not None:
            details = "\n".join(part for part in (details, str(error)) if part)
        traceback_text = traceback.format_exc()
        detailed_text = (
            ""
            if traceback_text == "NoneType: None\n"
            else erlab.interactive.utils._format_traceback(traceback_text)
        )
        erlab.interactive.utils.MessageDialog.critical(
            self._manager,
            "Error",
            text,
            informative_text=details,
            detailed_text=detailed_text,
        )

    def apply_batch_transform_dialog(
        self,
        dialog: typing.Any,
        launch_mode: typing.Literal["replace", "detach", "nest"],
    ) -> bool:
        try:
            targets = self._selected_batch_targets()
            operations = dialog.source_operations()
        except Exception as exc:
            self._show_batch_error("Batch operation could not be started.", error=exc)
            return False
        if not operations:
            self._show_batch_error(
                "Batch operation could not be started.",
                error=ValueError("The dialog did not produce an operation."),
            )
            return False
        try:
            self._validate_batch_operations(dialog, operations)
        except Exception as exc:
            self._show_batch_error("Batch operation could not be started.", error=exc)
            return False

        preflight_plan: list[tuple[typing.Any, ...]] = []
        for target in targets:
            try:
                node = self._manager._node_for_target(target)
                slicer_area = self._manager.get_imagetool(target).slicer_area
                input_name = slicer_area.data.name
                new_name = "" if input_name is None else str(input_name)
                source_spec = dialog.source_spec_for_data(slicer_area.data, new_name)
                self._validate_batch_operations(
                    dialog,
                    source_spec.operations,
                    extra_types=(RestoreNonuniformDimsOperation,),
                )
                source_data = slicer_area.data
                if source_spec.kind == "public_data":
                    source_data = erlab.utils.array._restore_nonuniform_dims(
                        source_data
                    )
                self._validate_batch_target_operations(
                    source_spec.operations,
                    source_data,
                )
                if hasattr(dialog, "preflight_data"):
                    dialog.preflight_data(source_data)
                preflight_plan.append(
                    (
                        target,
                        node,
                        slicer_area,
                        input_name,
                        new_name,
                        source_spec,
                    )
                )
            except Exception as exc:
                self._show_batch_error(
                    "Batch operation failed before changing data.",
                    target=target,
                    error=exc,
                )
                return False

        plan: list[tuple[typing.Any, ...]] = []
        for (
            target,
            node,
            slicer_area,
            input_name,
            new_name,
            source_spec,
        ) in preflight_plan:
            try:
                processed = source_spec.apply(slicer_area.data).rename(input_name)
                erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(
                    processed
                )

                parent_provenance = node.displayed_provenance_spec
                if parent_provenance is None:
                    parent_provenance = slicer_area.displayed_provenance_spec()
                nested_provenance = compose_full_provenance(
                    parent_provenance,
                    source_spec,
                )
                detached_provenance = dialog._detached_provenance_spec(
                    parent_provenance,
                    source_spec,
                    new_name,
                )

                replace_kind = ""
                replace_provenance = None
                replace_replay_source_data = None
                if launch_mode == "replace":
                    displayed_source = node.displayed_source_spec
                    if displayed_source is not None:
                        replace_kind = "source"
                        replace_provenance = dialog._compose_transform_provenance(
                            displayed_source,
                            source_spec,
                            new_name,
                        )
                    elif node.displayed_provenance_spec is not None:
                        replace_kind = "detached"
                        replace_provenance = dialog._compose_transform_provenance(
                            node.displayed_provenance_spec,
                            source_spec,
                            new_name,
                        )
                    else:
                        replace_kind = "detached"
                        replace_provenance = detached_provenance
                    if replace_kind == "detached":
                        replace_replay_source_data = (
                            node.detached_replay_source_data
                            if node.detached_replay_source_data is not None
                            else slicer_area.data
                        )

                plan.append(
                    (
                        target,
                        node,
                        slicer_area,
                        processed,
                        source_spec,
                        nested_provenance,
                        detached_provenance,
                        replace_kind,
                        replace_provenance,
                        replace_replay_source_data,
                    )
                )
            except Exception as exc:
                self._show_batch_error(
                    "Batch operation failed before changing data.",
                    target=target,
                    error=exc,
                )
                return False

        try:
            for (
                target,
                node,
                slicer_area,
                processed,
                source_spec,
                nested_provenance,
                detached_provenance,
                replace_kind,
                replace_provenance,
                replace_replay_source_data,
            ) in plan:
                if launch_mode == "replace":
                    if replace_provenance is not None:
                        if replace_kind == "source":
                            node.set_source_binding(
                                replace_provenance,
                                auto_update=node.source_auto_update,
                                state=node.source_state,
                            )
                        else:
                            node.set_detached_provenance(
                                replace_provenance,
                                replay_source_data=replace_replay_source_data,
                            )
                    slicer_area.replace_source_data(processed, emit_edited=True)
                    continue

                itool_kw = dialog._itool_kwargs(processed, slicer_area)
                tool = typing.cast(
                    "ImageTool | None",
                    erlab.interactive.itool(manager=False, **itool_kw),
                )
                if tool is None:
                    self._show_batch_error(
                        "An error occurred while applying batch data.",
                        error=RuntimeError("Could not create the ImageTool window."),
                    )
                    return False

                if launch_mode == "nest":
                    tool.set_provenance_spec(nested_provenance)
                    self._manager.add_imagetool_child(
                        tool,
                        target,
                        source_spec=source_spec,
                    )
                else:
                    tool.set_provenance_spec(detached_provenance)
                    self._manager.add_imagetool(
                        tool,
                        activate=True,
                        provenance_spec=detached_provenance,
                    )
            if launch_mode == "replace":
                self._manager._sigDataReplaced.emit()
        except Exception as exc:
            self._show_batch_error(
                "An error occurred while applying batch data.",
                error=exc,
            )
            return False
        return True

    def apply_batch_filter_dialog(self, dialog: typing.Any) -> bool:
        try:
            targets = self._selected_batch_targets()
            operation = dialog.filter_operation()
            if operation is not None:
                self._validate_batch_operations(dialog, (operation,))
        except Exception as exc:
            self._show_batch_error("Batch filter could not be started.", error=exc)
            return False

        for target in targets:
            try:
                slicer_area = self._manager.get_imagetool(target).slicer_area
                if operation is not None:
                    slicer_area._filter_operation_result_for_data(
                        slicer_area._data,
                        operation,
                        tuple(slicer_area.data.dims),
                    )
            except Exception as exc:
                self._show_batch_error(
                    "Batch filter failed before changing data.",
                    target=target,
                    error=exc,
                )
                return False

        try:
            for target in targets:
                slicer_area = self._manager.get_imagetool(target).slicer_area
                emit_edited = (
                    operation is not None
                    or slicer_area._accepted_filter_provenance_operation is not None
                    or slicer_area._applied_func is not None
                    or slicer_area.has_active_filter
                )

                def _apply_filter(
                    *,
                    area: ImageSlicerArea = slicer_area,
                    op: ToolProvenanceOperation | None = operation,
                    edited: bool = emit_edited,
                ) -> None:
                    area.apply_filter_operation(op, emit_edited=edited)

                slicer_area.record_history_mutation(
                    None,
                    _apply_filter,
                )
        except Exception as exc:
            self._show_batch_error(
                "An error occurred while applying batch filter.",
                error=exc,
            )
            return False
        return True

    def store_selected(self) -> None:
        self._manager.ensure_console_initialized()
        dialog = _StoreDialog(
            self._manager,
            [
                target
                for target in self._manager._selected_imagetool_targets()
                if isinstance(target, int)
            ],
        )
        dialog.exec()

    def unwatch_selected(self) -> None:
        """Unwatch selected ImageTool windows."""
        for index in self._manager.tree_view.selected_imagetool_indices:
            self._manager._tool_graph.root_wrappers[index].unwatch()

    def rename_imagetool(self, index: int, new_name: str) -> None:
        """Rename the ImageTool window corresponding to the given index."""
        self._manager._tool_graph.root_wrappers[index].name = str(new_name)

    def _materialize_duplicate_subtree(self, target: int | str) -> None:
        node = self._manager._node_for_target(target)
        if not node.materialize_pending_workspace_payload():
            raise _DuplicateMaterializationError(
                f"Could not materialize {node.display_text!r} for duplication"
            )
        for child_uid in tuple(node._childtool_indices):
            self._materialize_duplicate_subtree(child_uid)

    def _discard_duplicated_subtree(self, target: int | str) -> None:
        if isinstance(target, int):
            self._manager.remove_imagetool(target)
        else:
            self._manager._remove_childtool(target)

    def _finish_duplicated_subtree(self, target: int | str) -> None:
        node = self._manager._node_for_target(target)
        duplicated_uids = self._manager._tool_graph.subtree_uids(node.uid)
        if self._manager._is_figure_node(node):
            self._manager._figure_collection.select_uid(node.uid)
        for uid in duplicated_uids:
            self._manager._tool_graph.nodes[uid].show()
        for uid in duplicated_uids:
            self._manager._mark_node_added(uid)

    def _duplicate_target(self, target: int | str) -> int | str:
        self._materialize_duplicate_subtree(target)
        workspace_state_snapshot = self._manager._workspace_state.snapshot(
            node_uid_counter=self._manager._tool_graph.uid_counter
        )
        duplicated: int | str | None = None
        try:
            with self._manager._workspace_load_context():
                duplicated = self._manager._duplicate_subtree(target)
            self._finish_duplicated_subtree(duplicated)
        except Exception:
            try:
                if duplicated is not None:
                    with self._manager._workspace_load_context():
                        self._discard_duplicated_subtree(duplicated)
            finally:
                self._manager._workspace_state.restore(workspace_state_snapshot)
                self._manager._update_workspace_window_title()
            raise
        return duplicated

    def _duplicate_subtree(
        self, target: int | str, *, parent_override: int | str | None = None
    ) -> int | str:
        node = self._manager._node_for_target(target)
        new_target: int | str | None = None
        try:
            if node.is_imagetool:
                persistence = node.persistence_view()
                duplicated_window = self._manager.get_imagetool(target).duplicate(
                    _in_manager=True
                )
                if isinstance(node, _ImageToolWrapper):
                    new_target = self._manager.add_imagetool(
                        duplicated_window,
                        show=False,
                        source_input_ndim=node.source_input_ndim,
                        provenance_spec=persistence.provenance_spec,
                        source_spec=persistence.source_spec,
                        source_binding=persistence.source_binding,
                        source_auto_update=persistence.source_auto_update,
                        source_state=persistence.source_state,
                        note=node.note,
                    )
                else:
                    parent_target = (
                        parent_override
                        if parent_override is not None
                        else (self._manager._parent_node(node).uid)
                    )
                    new_target = self._manager.add_imagetool_child(
                        duplicated_window,
                        parent_target,
                        show=False,
                        provenance_spec=persistence.provenance_spec,
                        source_spec=persistence.source_spec,
                        source_binding=persistence.source_binding,
                        source_auto_update=persistence.source_auto_update,
                        source_state=persistence.source_state,
                        output_id=persistence.output_id,
                        note=node.note,
                    )
            else:
                tool = self._manager.get_childtool(node.uid)
                if node.parent_uid is None and self._manager._is_figure_node(node):
                    duplicated_tool = tool.duplicate()
                    duplicated_tool._tool_display_name = (
                        self._manager._figure_collection.duplicated_display_name(
                            node.display_text
                        )
                    )
                    new_target = self._manager.add_figuretool(
                        duplicated_tool, show=False, note=node.note
                    )
                else:
                    parent_target = (
                        parent_override
                        if parent_override is not None
                        else self._manager._parent_node(node).uid
                    )
                    new_target = self._manager.add_childtool(
                        tool.duplicate(), parent_target, show=False, note=node.note
                    )

            for child_uid in tuple(node._childtool_indices):
                self._manager._duplicate_subtree(child_uid, parent_override=new_target)
        except Exception:
            if new_target is not None:
                self._discard_duplicated_subtree(new_target)
            raise
        else:
            return new_target

    def duplicate_imagetool(self, index: int | str) -> int | str:
        """Duplicate the ImageTool window corresponding to the given index.

        Parameters
        ----------
        index
            Index of the ImageTool window to duplicate.

        Returns
        -------
        int
            Index of the newly created ImageTool window.
        """
        self._manager._commit_note_editor()
        return self._duplicate_target(index)

    def duplicate_childtool(self, uid: str) -> str:
        """Duplicate the child tool corresponding to the given UID.

        Parameters
        ----------
        uid
            UID of the child tool to duplicate.

        Returns
        -------
        str
            UID of the newly created child tool.
        """
        self._manager._commit_note_editor()
        duplicated = self._duplicate_target(uid)
        if isinstance(duplicated, str):
            return duplicated
        raise TypeError("Expected duplicated child target to remain nested")

    def link_imagetools(self, *indices: int | str, link_colors: bool = True) -> None:
        """Link the ImageTool windows corresponding to the given indices."""
        if len(indices) <= 1:
            return
        nodes: list[_ImageToolWrapper | _ManagedWindowNode] = []
        slicers: list[ImageSlicerArea] = []
        for index in indices:
            node = self._manager._node_for_target(index)
            if not node.is_imagetool:
                raise KeyError(f"Target {index!r} is not an ImageTool")
            nodes.append(node)
            if node.imagetool is not None:
                slicers.append(node.slicer_area)
        if len(slicers) > 1:
            linker = erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy(
                *slicers,
                link_colors=link_colors,
            )
            self._manager._link_registry.append(linker)
        link_key = uuid.uuid4().hex
        for node in nodes:
            node.set_workspace_link_state(link_key, link_colors=link_colors)
            self._manager._mark_node_state_dirty(node.uid)
        self._manager._sigReloadLinkers.emit()

    def name_of_imagetool(self, index: int) -> str:
        """Get the name of the ImageTool window corresponding to the given index."""
        return self._manager._tool_graph.root_wrappers[index].name

    def label_of_imagetool(self, index: int) -> str:
        """Get the label of the ImageTool window corresponding to the given index."""
        return self._manager._tool_graph.root_wrappers[index].label_text

    def _data_load(
        self, paths: list[str], loader_name: str, kwargs: dict[str, typing.Any]
    ) -> None:
        """Load data from the given files using the specified loader."""
        if loader_name == "ask":
            self._manager._handle_dropped_files([pathlib.Path(p) for p in paths])
            return

        self._manager._data_ingress.add_from_multiple_files(
            [],
            [pathlib.Path(p) for p in paths],
            [],
            func=erlab.io.loaders[loader_name].load,
            kwargs=kwargs,
            retry_callback=lambda _: self._manager._data_load(
                paths, loader_name, kwargs
            ),
        )

    def _data_replace(
        self, data_list: list[xr.DataArray], indices: list[int | str]
    ) -> None:
        """Replace data in the ImageTool windows with the given data."""
        prepared_data = erlab.interactive.imagetool.viewer_state._prepare_input_data(
            data_list,
            self._manager,
        )
        if prepared_data is None:
            return

        plans: list[
            tuple[
                bool,
                int | str,
                xr.DataArray,
                tuple[xr.DataArray, ToolProvenanceSpec | None] | None,
            ]
        ] = []
        planned_next_idx = self._manager.next_idx
        for prepared, idx in zip(prepared_data, indices, strict=True):
            darr = prepared.data
            if isinstance(idx, int) and idx < 0:
                # Negative index counts from the end
                idx = sorted(self._manager._tool_graph.root_wrappers.keys())[idx]
            add_new = isinstance(idx, int) and idx == planned_next_idx
            if add_new:
                erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(darr)
                plans.append((True, idx, darr, None))
                planned_next_idx += 1
                continue
            replacement = self._manager._metadata_editor.prepare_replacement(
                idx, darr, source_input_dtype=prepared.source_dtype
            )
            if replacement is None:
                erlab.interactive.imagetool.slicer.ArraySlicer.preflight_array(darr)
            plans.append((False, idx, darr, replacement))

        data_changed = False
        for add_new, idx, darr, replacement in plans:
            if add_new:
                # If not yet created, add new tool.
                if idx != self._manager.next_idx:
                    break
                if self._manager._data_ingress.receive_data([darr], {}) != [True]:
                    break
            elif replacement is None:
                self._manager.get_imagetool(idx).slicer_area.replace_source_data(darr)
            else:
                processed, provenance = replacement
                self._manager._metadata_editor.commit_replacement(
                    idx, darr, processed, provenance
                )
            data_changed = True
        if data_changed:
            self._manager._sigDataReplaced.emit()

    def _find_watched_idx(self, uid: str) -> int | None:
        """Find the index of the watched ImageTool corresponding to the given UID."""
        for k, v in self._manager._tool_graph.root_wrappers.items():
            if v._watched_uid == uid:
                return k
        return None

    def _watched_source_color_key(self, wrapper: _ImageToolWrapper) -> str:
        if wrapper._watched_source_uid:
            return f"source-uid:{wrapper._watched_source_uid}"
        if wrapper._watched_source_label:
            return f"source-label:{wrapper._watched_source_label}"
        watched_uid = typing.cast("str", wrapper._watched_uid)
        watched_varname = typing.cast("str", wrapper._watched_varname)
        return f"legacy:{watched_uid.removeprefix(f'{watched_varname} ')}"

    def color_for_watched_var_source(self, wrapper: _ImageToolWrapper) -> QtGui.QColor:
        """Return a different color for different watched-variable sources."""
        source_key = self._manager._watched_source_color_key(wrapper)
        all_source_keys = tuple(
            dict.fromkeys(
                self._manager._watched_source_color_key(v)
                for v in self._manager._tool_graph.root_wrappers.values()
                if v.watched
            )
        )
        idx = all_source_keys.index(source_key)
        return QtGui.QColor(*_WATCHED_VAR_COLORS[idx % len(_WATCHED_VAR_COLORS)])

    def _remove_watched(self, uid: str) -> None:
        """Remove the ImageTool corresponding to the given watched variable UID."""
        idx = self._manager._find_watched_idx(uid)
        if idx is not None:  # pragma: no branch
            self._manager.remove_imagetool(idx)
            return
        if uid in self._manager._tool_graph.nodes:
            if isinstance(self._manager._tool_graph.nodes[uid], _ImageToolWrapper):
                self._manager.remove_imagetool(
                    typing.cast(
                        "_ImageToolWrapper", self._manager._tool_graph.nodes[uid]
                    ).index
                )
            else:
                self._manager._remove_childtool(uid)

    def _show_watched(self, uid: str) -> None:
        """Show the ImageTool corresponding to the given watched variable UID."""
        idx = self._manager._find_watched_idx(uid)
        if idx is not None:
            self._manager.show_imagetool(idx)
            return
        if uid in self._manager._tool_graph.nodes:
            self._manager._node_for_target(uid).show()

    def _data_watched_update(
        self,
        varname: str,
        uid: str,
        darr: xr.DataArray,
        watched_metadata: Mapping[str, typing.Any] | None = None,
    ) -> None:
        """Update ImageTool window corresponding to the given watched variable."""
        watched_metadata = dict(watched_metadata or {})
        idx = self._manager._find_watched_idx(uid)
        if idx is None:
            # If the tool does not exist, create a new one
            self._manager._data_ingress.receive_data(
                [darr],
                {},
                watched_var=(varname, uid),
                watched_metadata={
                    **dict(watched_metadata),
                    "connected": True,
                },
            )
        else:
            # Update data in the existing tool
            wrapper = self._manager._tool_graph.root_wrappers[idx]
            wrapper.set_watched_binding(
                varname,
                uid,
                workspace_link_id=typing.cast(
                    "str | None",
                    watched_metadata.get("workspace_link_id")
                    or wrapper._watched_workspace_link_id,
                ),
                source_label=typing.cast(
                    "str | None",
                    watched_metadata.get("source_label")
                    or wrapper._watched_source_label,
                ),
                source_uid=typing.cast(
                    "str | None",
                    watched_metadata.get("source_uid") or wrapper._watched_source_uid,
                ),
                connected=True,
            )
            try:
                prepared_data = (
                    erlab.interactive.imagetool.viewer_state._prepare_input_data(
                        darr,
                        self._manager,
                        allow_dialog=False,
                    )
                )
            except ValueError:
                logger.exception(
                    "Could not open watched variable update in ImageTool",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self._manager,
                    "Watched Data Update Failed",
                    f"Could not open watched variable {varname!r} in ImageTool.",
                    "The previous data remains visible.",
                )
                return
            if prepared_data is None:  # pragma: no cover - no dialog is shown here
                return
            prepared = prepared_data[0]
            try:
                replacement = self._manager._metadata_editor.prepare_replacement(
                    idx,
                    prepared.data,
                    source_input_dtype=prepared.source_dtype,
                )
            except Exception as exc:
                logger.exception(
                    "Could not preserve metadata assignments on watched variable "
                    "update",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self._manager,
                    "Watched Data Update Failed",
                    f"Could not update watched variable {varname!r} in ImageTool.",
                    str(exc),
                )
                return
            wrapper.set_source_input_ndim(prepared.source_ndim)
            wrapper.set_source_input_dtype(prepared.source_dtype)
            if replacement is None:
                # A notebook-side update replaces the watched variable itself, so
                # prior ImageTool operations no longer describe the displayed array.
                wrapper.set_displayed_provenance(None)
                self._manager.get_imagetool(idx).slicer_area.replace_source_data(
                    prepared.data
                )
            else:
                # Rebuild provenance from the updated watched source and only the
                # metadata assignments that still apply to it.
                processed, provenance = replacement
                self._manager._metadata_editor.commit_replacement(
                    idx, prepared.data, processed, provenance
                )

    def _data_unwatch(self, uid: str) -> None:
        idx = self._manager._find_watched_idx(uid)
        if idx is not None:
            # Convert the tool to a normal one
            self._manager._tool_graph.root_wrappers[idx].unwatch()

    def _get_imagetool_data(self, index_or_uid: int | str) -> xr.DataArray | None:
        """Request data from the ImageTool window corresponding to the given index."""
        if isinstance(index_or_uid, str):
            if (
                index_or_uid in self._manager._tool_graph.nodes
                and self._manager._is_imagetool_target(index_or_uid)
            ):
                index: int | str | None = index_or_uid
            else:
                index = self._manager._find_watched_idx(index_or_uid)
        else:
            index = index_or_uid

        if index is None:
            return None
        with contextlib.suppress(KeyError):
            return self._manager.get_imagetool(index).slicer_area.displayed_data
        return None

    def _send_imagetool_data(self, index_or_uid: int | str) -> None:
        """Send data of the ImageTool window corresponding to the given index."""
        self._manager._sigReplyData.emit(
            self._manager._get_imagetool_data(index_or_uid)
        )

    def _watch_info(self) -> dict[str, typing.Any]:
        """Return watched-variable metadata used by notebook reconnect workflows."""
        return {
            "workspace_link_id": self._manager._workspace_state.link_id,
            "watched": [
                wrapper.watched_metadata()
                for wrapper in self._manager._tool_graph.root_wrappers.values()
                if wrapper.watched
            ],
        }

    def _send_watch_info(self) -> None:
        """Send watched-variable metadata to the manager server."""
        self._manager._sigReplyData.emit(self._manager._watch_info())

    def ensure_console_initialized(self) -> None:
        """Ensure that the console window is initialized."""
        if not hasattr(self._manager, "console"):
            from erlab.interactive.imagetool.manager._console import (
                _ImageToolManagerJupyterConsole,
            )

            self._manager.console = _ImageToolManagerJupyterConsole(self._manager)

    def toggle_console(self) -> None:
        """Toggle the console window."""
        self._manager.ensure_console_initialized()
        if self._manager.console.isVisible():
            self._manager.console.hide()
        else:
            self._manager.console.show()
            self._manager.console.activateWindow()
            self._manager.console.raise_()
            self._manager.console._console_widget._control.setFocus()
            self._manager.console._initialize_visible_console()

    @property
    def _recent_loader_name(self) -> str | None:
        """Name of the most recently used loader."""
        if self._manager._recent_name_filter is not None:  # pragma: no branch
            for k in erlab.io.loaders:  # pragma: no branch
                if (
                    self._manager._recent_name_filter
                    in erlab.io.loaders[k].file_dialog_methods
                ):
                    return k
        return None

    def ensure_explorer_initialized(self) -> None:
        """Ensure that the data explorer window is initialized."""
        self._manager._ensure_standalone_app("explorer")

    def show_explorer(self) -> None:
        """Show data explorer window."""
        self._manager._show_standalone_app("explorer")

    def show_ptable(self) -> None:
        """Show the periodic-table explorer window."""
        self._manager._show_standalone_app("ptable")

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None) -> None:
        """Handle drag-and-drop operations entering the window."""
        if event:
            mime_data: QtCore.QMimeData | None = event.mimeData()
            if mime_data and mime_data.hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        """Handle drag-and-drop operations dropping files into the window."""
        if event:
            mime_data: QtCore.QMimeData | None = event.mimeData()
            if mime_data and mime_data.hasUrls():
                urls = mime_data.urls()
                self._manager._handle_dropped_files(
                    [pathlib.Path(url.toLocalFile()) for url in urls]
                )

    def _handle_dropped_files(self, file_paths: list[pathlib.Path]) -> None:
        """Handle files dropped into the window."""
        if file_paths:  # pragma: no branch
            extensions: set[str] = {
                file_path.suffix.lower() for file_path in file_paths
            }
            if len(extensions) != 1:
                QtWidgets.QMessageBox.critical(
                    self._manager,
                    "Error",
                    "Multiple file types cannot be opened at the same time.",
                )
                return
            self._manager._data_ingress.open_multiple_files(
                file_paths,
                try_workspace=extensions == {".itws"},
            )

    def _show_operation_error(self, log_message: str, text: str) -> None:
        logger.exception(log_message, extra={"suppress_ui_alert": True})
        erlab.interactive.utils.MessageDialog.critical(
            self._manager,
            "Error",
            text,
            detailed_text=erlab.interactive.utils._format_traceback(
                traceback.format_exc()
            ),
        )

    def _show_workspace_save_worker_error(self, error_text: str) -> None:
        logger.error(
            "Error while saving workspace\n%s",
            error_text,
            extra={"suppress_ui_alert": True},
        )
        erlab.interactive.utils.MessageDialog.critical(
            self._manager,
            "Error",
            "An error occurred while saving the workspace file.",
            detailed_text=erlab.interactive.utils._format_traceback(error_text),
        )

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        """Save a reference to an additional window widget.

        This is mainly used for handling tool windows such as goldtool and dtool opened
        from child ImageTool windows. This way, they can stay open even when the
        ImageTool that opened them is removed.

        All additional windows are closed when the manager is closed.

        Only pass widgets that are not associated with a parent widget.

        Parameters
        ----------
        widget
            The widget to add.
        """
        uid = str(uuid.uuid4())
        widget.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self._manager._additional_windows[uid] = widget  # Store reference to prevent gc
        widget.destroyed.connect(
            lambda: self._manager._additional_windows.pop(uid, None)
        )
        widget.show()

    def add_childtool(
        self,
        tool: erlab.interactive.utils.ToolWindow,
        index: int | str,
        *,
        show: bool = True,
        uid: str | None = None,
        snapshot_token: str | None = None,
        source_snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
        note: str | bytes | None = None,
    ) -> str:
        """Register a child tool window.

        This is mainly used for handling tool windows such as goldtool and dtool opened
        from child ImageTool windows.

        Parameters
        ----------
        tool
            The tool window to add.
        index
            Target of the parent managed window.
        show
            Whether to show the tool window after adding it, by default `True`.
        """
        if tool.manager_collection == "figures":
            return self._manager.add_figuretool(
                tool,
                show=show,
                uid=uid,
                snapshot_token=snapshot_token,
                source_snapshot_token=source_snapshot_token,
                created_time=created_time,
                note=note,
            )

        parent = self._manager._node_for_target(index)
        node = _ManagedWindowNode(
            self._manager,
            self._manager._next_node_uid(uid),
            parent.uid,
            tool,
            snapshot_token=snapshot_token,
            source_snapshot_token=source_snapshot_token,
            created_time=created_time,
            note=note,
        )
        if not tool._tool_display_name:
            tool._tool_display_name = parent.name

        def _parent_source_fetcher(parent_uid: str = parent.uid) -> xr.DataArray:
            return self._manager._node_for_target(parent_uid).current_source_data()

        def _parent_provenance_fetcher(
            parent_uid: str = parent.uid,
        ) -> ToolProvenanceSpec | None:
            return self._manager._node_for_target(parent_uid).displayed_provenance_spec

        tool.set_source_parent_fetcher(_parent_source_fetcher)
        tool.set_input_provenance_parent_fetcher(_parent_provenance_fetcher)
        self._manager._register_child_node(node)
        self._manager.tree_view.childtool_added(node.uid, index)
        self._manager._mark_node_added(node.uid)
        if self._manager._is_figure_node(node):
            self._manager._figure_collection.sync(select_uid=node.uid if show else None)
        if show:
            node.show()
        return node.uid

    def add_imagetool_child(
        self,
        tool: ImageTool,
        parent: int | str,
        *,
        show: bool = True,
        activate: bool = False,
        uid: str | None = None,
        provenance_spec: ToolProvenanceSpec | None = None,
        source_spec: ToolProvenanceSpec | None = None,
        source_binding: ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        output_id: str | None = None,
        snapshot_token: str | None = None,
        source_snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
        note: str | bytes | None = None,
    ) -> str:
        parent_node = self._manager._node_for_target(parent)
        if source_spec is None and source_binding is not None:
            source_spec = source_binding.materialize(parent_node.current_source_data())
            source_binding = None
        elif source_spec is not None:
            source_binding = None
        if provenance_spec is None and source_spec is not None:
            provenance_spec = compose_display_provenance(
                parent_node.displayed_provenance_spec,
                source_spec,
                parent_data=parent_node.current_source_data(),
            )
        if provenance_spec is not None:
            tool.set_provenance_spec(provenance_spec)
        node = _ManagedWindowNode(
            self._manager,
            self._manager._next_node_uid(uid),
            parent_node.uid,
            tool,
            provenance_spec=provenance_spec,
            source_spec=source_spec,
            source_binding=source_binding,
            source_auto_update=source_auto_update,
            source_state=source_state,
            output_id=output_id,
            snapshot_token=snapshot_token,
            source_snapshot_token=source_snapshot_token,
            created_time=created_time,
            note=note,
        )
        self._manager._register_child_node(node)
        if output_id is not None and parent_node.tool_window is not None:
            parent_node.tool_window._register_output_imagetool_target(
                output_id, node.uid
            )
        self._manager.tree_view.childtool_added(node.uid, parent)
        self._manager._mark_node_added(node.uid)
        self._manager._update_actions()
        if show:
            node.show()
        if activate and node.window is not None:
            node.window.activateWindow()
            node.window.raise_()
        return node.uid

    def index_from_slicer_area(
        self, slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea
    ) -> int | None:
        """Get the index corresponding to the given slicer area."""
        root_wrappers = self._manager._tool_graph.root_wrappers
        for index, wrapper in root_wrappers.items():  # pragma: no branch
            if (wrapper.imagetool is not None) and (wrapper.slicer_area is slicer_area):
                return index
        return None

    def wrapper_from_slicer_area(
        self, slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea
    ) -> _ImageToolWrapper | None:
        """Get the ImageTool wrapper corresponding to the given slicer area."""
        index = self._manager.index_from_slicer_area(slicer_area)
        if index is not None:
            return self._manager._tool_graph.root_wrappers[index]
        return None

    def node_from_slicer_area(
        self, slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        for node in self._manager._tool_graph.nodes.values():
            if node.imagetool is not None and node.slicer_area is slicer_area:
                return node
        return None

    def target_from_slicer_area(
        self, slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea
    ) -> int | str | None:
        node = self._manager.node_from_slicer_area(slicer_area)
        if node is None:
            return None
        if isinstance(node, _ImageToolWrapper):
            return node.index
        return node.uid

    def _add_childtool_from_slicerarea(
        self,
        tool: QtWidgets.QWidget,
        parent_slicer_area: erlab.interactive.imagetool.viewer.ImageSlicerArea,
    ) -> None:
        target = self._manager.target_from_slicer_area(parent_slicer_area)
        if target is not None:
            if isinstance(tool, erlab.interactive.utils.ToolWindow):
                self._manager.add_childtool(tool, target)
                return
            if isinstance(tool, ImageTool):
                self._manager.add_imagetool_child(tool, target)
                return

        # The parent slicer area is not owned by this manager; just keep track of it
        self._manager.add_widget(tool)

    def _get_childtool_and_parent(
        self, uid: str
    ) -> tuple[erlab.interactive.utils.ToolWindow, int]:
        """Get the child tool window and parent index corresponding to the given UID.

        Parameters
        ----------
        uid
            The unique ID of the child tool to get.

        Returns
        -------
        ToolWindow
            The child tool window corresponding to the given UID.
        int
            The index of the parent ImageTool window.
        """
        node = self._manager._child_node(uid)
        if (
            node.pending_workspace_tool_payload is not None
            and not node.materialize_pending_workspace_payload()
        ):
            raise KeyError(f"No child tool with UID {uid} found")
        tool = node.tool_window
        if tool is None or not erlab.interactive.utils.qt_is_valid(tool):
            self._manager._remove_childtool(uid)
            raise KeyError(f"No child tool with UID {uid} found")
        if node.parent_uid is None:
            return tool, -1
        return tool, self._manager._root_wrapper_for_uid(uid).index

    def get_childtool(self, uid: str) -> erlab.interactive.utils.ToolWindow:
        """Get the child tool window corresponding to the given UID.

        Parameters
        ----------
        uid
            The unique ID of the child tool to get.

        Returns
        -------
        ToolWindow
            The child tool window corresponding to the given UID.
        """
        return self._manager._get_childtool_and_parent(uid)[0]

    def show_childtool(self, uid: str) -> None:
        """Show the child tool window corresponding to the given UID."""
        self._manager._child_node(uid).show()

    def _remove_childtool(self, uid: str) -> None:
        """Unregister a child tool window.

        Parameters
        ----------
        uid
            The unique ID of the child tool to remove.
        """
        if uid not in self._manager._tool_graph.nodes:
            return
        was_figure = self._manager._is_figure_uid(uid)
        removed_link_keys = self._manager._workspace_link_keys_for_subtree(uid)
        self._manager._mark_removed_subtree_dirty(uid)
        closing_document = self._manager._workspace_state.closing_document
        if not closing_document:
            self._manager.tree_view.childtool_removed(uid)
        self._manager._remove_uid_target(uid)
        if closing_document:
            return
        self._manager._mark_singleton_workspace_link_groups_dirty(removed_link_keys)
        if was_figure:
            self._manager._figure_collection.sync()
        self._manager._update_actions()

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        """Event filter that intercepts select all and copy shortcuts.

        For some operating systems, shortcuts are often intercepted by actions in the
        menu bar. This filter ensures that the shortcuts work as expected when the
        target widget has focus.
        """
        if (
            event is not None
            and event.type() == QtCore.QEvent.Type.ShortcutOverride
            and isinstance(obj, QtWidgets.QWidget)
            and obj.hasFocus()
        ):
            event = typing.cast("QtGui.QKeyEvent", event)
            if event.matches(QtGui.QKeySequence.StandardKey.SelectAll) or event.matches(
                QtGui.QKeySequence.StandardKey.Copy
            ):
                event.accept()
                return True
        return QtWidgets.QMainWindow.eventFilter(self._manager, obj, event)
