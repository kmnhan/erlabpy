from __future__ import annotations

import contextlib
import logging
import pathlib
import traceback
import typing
import uuid

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive.imagetool import provenance
from erlab.interactive.imagetool._mainwindow import ImageTool
from erlab.interactive.imagetool.manager import _workspace as _manager_workspace
from erlab.interactive.imagetool.manager import _xarray as _manager_xarray
from erlab.interactive.imagetool.manager._dialogs import (
    _ConcatDialog,
    _is_loader_func,
    _RenameDialog,
    _StoreDialog,
)
from erlab.interactive.imagetool.manager._io import _MultiFileHandler
from erlab.interactive.imagetool.manager._widgets import (
    _WATCHED_VAR_COLORS,
    _show_workspace_file_lock_error,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    import datetime
    from collections.abc import Callable, Mapping

    import xarray as xr

    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager

logger = logging.getLogger(__name__)


class _ActionsController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager
        self.__rename_dialog: _RenameDialog | None = None
        self.__concat_dialog: _ConcatDialog | None = None

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

                qmodelindex = self._manager.tree_view._model._row_index(new_uid)

                selection_model.select(
                    QtCore.QItemSelection(qmodelindex, qmodelindex),
                    QtCore.QItemSelectionModel.SelectionFlag.Select,
                )
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
        node = self._manager._child_node(uid)
        if not node.is_imagetool:
            raise KeyError(f"Target {uid!r} is not an ImageTool")

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

        self._manager.tree_view.childtool_removed(uid)
        self._manager._unregister_node(uid)

        with self._manager._workspace_load_context():
            new_index = self._manager.add_imagetool(
                promoted_window,
                show=False,
                uid=uid,
                provenance_spec=provenance_spec,
                snapshot_token=snapshot_token,
                created_time=created_time,
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

    def unlink_selected(self, deselect: bool = True) -> None:
        """Unlink selected ImageTool windows."""
        dirty_uids: list[str] = []
        for index in self._manager._selected_imagetool_targets():
            node = self._manager._node_for_target(index)
            slicer_area = self._manager.get_imagetool(index).slicer_area
            if slicer_area.is_linked:
                dirty_uids.append(node.uid)
            slicer_area.unlink()
        for uid in dirty_uids:
            self._manager._mark_node_state_dirty(uid)
        self._manager._sigReloadLinkers.emit()
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

    def _duplicate_subtree(
        self, target: int | str, *, parent_override: int | str | None = None
    ) -> int | str:
        node = self._manager._node_for_target(target)
        if node.is_imagetool:
            persistence = node.persistence_view()
            duplicated_window = self._manager.get_imagetool(target).duplicate(
                _in_manager=True
            )
            if isinstance(node, _ImageToolWrapper):
                new_target: int | str = self._manager.add_imagetool(
                    duplicated_window,
                    activate=True,
                    source_input_ndim=node.source_input_ndim,
                    provenance_spec=persistence.provenance_spec,
                    source_spec=persistence.source_spec,
                    source_binding=persistence.source_binding,
                    source_auto_update=persistence.source_auto_update,
                    source_state=persistence.source_state,
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
                    activate=True,
                    provenance_spec=persistence.provenance_spec,
                    source_spec=persistence.source_spec,
                    source_binding=persistence.source_binding,
                    source_auto_update=persistence.source_auto_update,
                    source_state=persistence.source_state,
                    output_id=persistence.output_id,
                )
        else:
            tool = typing.cast("erlab.interactive.utils.ToolWindow", node.tool_window)
            parent_target = (
                parent_override
                if parent_override is not None
                else self._manager._parent_node(node).uid
            )
            new_target = self._manager.add_childtool(tool.duplicate(), parent_target)

        for child_uid in node._childtool_indices:
            self._manager._duplicate_subtree(child_uid, parent_override=new_target)
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
        return self._manager._duplicate_subtree(index)

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
        duplicated = self._manager._duplicate_subtree(uid)
        if isinstance(duplicated, str):
            return duplicated
        raise TypeError("Expected duplicated child target to remain nested")

    def link_imagetools(self, *indices: int | str, link_colors: bool = True) -> None:
        """Link the ImageTool windows corresponding to the given indices."""
        if len(indices) <= 1:
            return
        linker = erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy(
            *[self._manager.get_imagetool(t).slicer_area for t in indices],
            link_colors=link_colors,
        )
        self._manager._link_registry.append(linker)
        for index in indices:
            self._manager._mark_node_state_dirty(
                self._manager._node_for_target(index).uid
            )
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

        self._manager._add_from_multiple_files(
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
        for darr, idx in zip(data_list, indices, strict=True):
            if isinstance(idx, int) and idx < 0:
                # Negative index counts from the end
                idx = sorted(self._manager._tool_graph.root_wrappers.keys())[idx]
            elif isinstance(idx, int) and idx == self._manager.next_idx:
                # If not yet created, add new tool
                self._manager._data_recv([darr], {})
                continue
            self._manager.get_imagetool(idx).slicer_area.replace_source_data(darr)
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
        return _WATCHED_VAR_COLORS[idx % len(_WATCHED_VAR_COLORS)]

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
            self._manager._data_recv(
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
            wrapper.set_source_input_ndim(darr.ndim)
            wrapper.set_source_input_dtype(darr.dtype)
            self._manager.get_imagetool(idx).slicer_area.replace_source_data(darr)

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
            self._manager.open_multiple_files(
                file_paths,
                try_workspace=(extensions == {".itws"} or extensions == {".h5"}),
            )

    def _show_loaded_info(
        self,
        loaded: list[pathlib.Path],
        canceled: list[pathlib.Path],
        failed: list[pathlib.Path],
        retry_callback: Callable[[list[pathlib.Path]], typing.Any],
    ) -> None:
        """Show a message box with information about the loaded files.

        Nothing is shown if all files were successfully loaded.

        Parameters
        ----------
        loaded
            List of successfully loaded files.
        canceled
            List of files that were aborted before trying to load.
        failed
            List of files that failed to load.
        retry_callback
            Callback function to retry loading the failed files. It should accept a list
            of path objects as its only argument.

        """
        loaded, canceled, failed = (
            list(dict.fromkeys(loaded)),
            list(dict.fromkeys(canceled)),
            list(dict.fromkeys(failed)),
        )  # Remove duplicate entries

        n_done, n_fail = len(loaded), len(failed)

        status_msg = f"Loaded {n_done} {'file' if n_done == 1 else 'files'}"
        self._manager._status_bar.showMessage(status_msg, 5000)

        if n_fail == 0:
            return

        message = f"Loaded {n_done} file"
        if n_done != 1:
            message += "s"
        message = message + f" with {n_fail} failure"
        if n_fail != 1:
            message += "s"
        message += "."

        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setText(message)

        loaded_str = "\n".join(p.name for p in loaded)
        if loaded_str:
            loaded_str = f"Loaded:\n{loaded_str}\n\n"

        failed_str = "\n".join(p.name for p in failed)
        if failed_str:
            failed_str = f"Failed:\n{failed_str}\n\n"

        canceled_str = "\n".join(p.name for p in canceled)
        if canceled_str:
            canceled_str = f"Canceled:\n{canceled_str}\n\n"
        msg_box.setDetailedText(f"{loaded_str}{failed_str}{canceled_str}")

        msg_box.setInformativeText("Do you want to retry loading the failed files?")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Retry
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Retry)
        if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Retry:
            retry_callback(failed)

    def open_multiple_files(
        self, queued: list[pathlib.Path], try_workspace: bool = False
    ) -> None:
        """Open multiple files in the manager."""
        n_files: int = len(queued)
        loaded: list[pathlib.Path] = []
        failed: list[pathlib.Path] = []
        metadata_from_attrs = _manager_workspace._workspace_file_metadata_from_attrs

        if try_workspace:
            for p in list(queued):
                explicit_workspace = p.suffix.lower() == ".itws"
                try:
                    dt = _manager_xarray.open_workspace_datatree(p, chunks=None)
                except Exception as exc:
                    if _manager_workspace._is_workspace_file_lock_error(exc):
                        logger.info(
                            "Workspace file is already open or locked: %s",
                            p,
                            extra={"suppress_ui_alert": True},
                        )
                        _show_workspace_file_lock_error(self._manager, p)
                        queued.remove(p)
                    elif explicit_workspace:
                        self._manager._show_operation_error(
                            "Error while loading workspace",
                            "An error occurred while loading the workspace file.",
                        )
                        queued.remove(p)
                    else:
                        logger.debug(
                            "Failed to open %s as datatree workspace", p, exc_info=True
                        )
                else:
                    if self._manager._is_datatree_workspace(dt):
                        dt.close()
                        try:
                            with self._manager._workspace_document_access_context(
                                p
                            ) as access:
                                _manager_workspace._recover_workspace_transactions(
                                    access.path
                                )
                                workspace_dt = _manager_xarray.open_workspace_datatree(
                                    access.path, chunks=None
                                )
                                workspace_dt_owned = True
                                try:
                                    (
                                        schema_version,
                                        delta_save_count,
                                        _manifest,
                                    ) = metadata_from_attrs(workspace_dt.attrs)
                                    if not self._manager._confirm_save_dirty_workspace(
                                        "Opening a workspace replaces the windows "
                                        "currently in this manager."
                                    ):
                                        return
                                    loaded_workspace = self._manager._from_datatree(
                                        workspace_dt,
                                        replace=True,
                                        mark_dirty=False,
                                        select=False,
                                        workspace_file_path=access.path,
                                    )
                                    workspace_dt_owned = False
                                    if loaded_workspace:
                                        self._manager._associate_loaded_workspace_file(
                                            access.path,
                                            schema_version,
                                            delta_save_count=delta_save_count,
                                            workspace_access=access,
                                            rebind_data=False,
                                        )
                                finally:
                                    if workspace_dt_owned:
                                        workspace_dt.close()
                        except Exception as exc:
                            if _manager_workspace._is_workspace_file_lock_error(exc):
                                logger.info(
                                    "Workspace file is already open or locked: %s",
                                    p,
                                    extra={"suppress_ui_alert": True},
                                )
                                _show_workspace_file_lock_error(self._manager, p)
                            else:
                                self._manager._show_operation_error(
                                    "Error while loading workspace",
                                    "An error occurred while loading the workspace "
                                    "file.",
                                )
                        finally:
                            queued.remove(p)
                            loaded.append(p)
                    else:
                        dt.close()
                        if explicit_workspace:
                            logger.error(
                                "File with .itws extension is not an ImageTool "
                                "workspace: %s",
                                p,
                                extra={"suppress_ui_alert": True},
                            )
                            erlab.interactive.utils.MessageDialog.critical(
                                self._manager,
                                "Error",
                                "An error occurred while loading the workspace file.",
                                f"{p.name} is not a valid ImageTool workspace file.",
                            )
                            queued.remove(p)

        if len(queued) == 0:
            return

        # Get loaders applicable to input files
        valid_loaders: dict[str, tuple[Callable, dict]] = (
            erlab.interactive.utils.file_loaders(queued)
        )

        if len(valid_loaders) == 0:
            if all(file_path.is_dir() for file_path in queued):
                # If all dropped paths are directories, open them in the explorer
                explorer = typing.cast(
                    "_TabbedExplorer", self._manager._show_standalone_app("explorer")
                )
                for file_path in queued:
                    explorer.add_tab(root_path=file_path)
                return

            singular: bool = n_files == 1
            QtWidgets.QMessageBox.critical(
                self._manager,
                "Error",
                f"The selected {'file' if singular else 'files'} "
                f"with extension '{queued[0].suffix}' {'is' if singular else 'are'} "
                "not supported by any available plugin.",
            )
            return

        if len(valid_loaders) == 1:
            name_filter, (func, kargs) = next(iter(valid_loaders.items()))
            if _is_loader_func(func):
                selected = self._manager._select_loader_options(
                    valid_loaders, name_filter, sample_paths=queued
                )
                if selected is None:
                    return
                self._manager._recent_name_filter, func, kargs = selected
            else:
                self._manager._recent_name_filter = name_filter
        else:
            selected = self._manager._select_loader_options(
                valid_loaders, sample_paths=queued
            )
            if selected is None:
                return
            self._manager._recent_name_filter, func, kargs = selected

        self._manager._add_from_multiple_files(
            loaded, queued, failed, func, kargs, self._manager.open_multiple_files
        )

    def _error_creating_imagetool(self) -> None:
        """Show an error message when an ImageTool window could not be created."""
        erlab.interactive.utils.MessageDialog.critical(
            self._manager,
            "Error",
            "An error occurred while creating the ImageTool window.",
            "The data may be incompatible with ImageTool.",
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

    def _add_from_multiple_files(
        self,
        loaded: list[pathlib.Path],
        queued: list[pathlib.Path],
        failed: list[pathlib.Path],
        func: Callable,
        kwargs: dict[str, typing.Any],
        retry_callback: Callable,
    ) -> None:
        handler = _MultiFileHandler(self._manager, queued, func, kwargs)
        self._manager._file_handlers.add(handler)

        def _finished_callback(loaded_new, aborted, failed_new) -> None:
            self._manager._show_loaded_info(
                loaded + loaded_new,
                aborted,
                failed + failed_new,
                retry_callback=retry_callback,
            )
            self._manager._file_handlers.remove(handler)

        handler.sigFinished.connect(_finished_callback)
        handler.start()

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
        created_time: datetime.datetime | str | bytes | None = None,
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
        parent = self._manager._node_for_target(index)
        node = _ManagedWindowNode(
            self._manager,
            self._manager._next_node_uid(uid),
            parent.uid,
            tool,
            snapshot_token=snapshot_token,
            created_time=created_time,
        )
        if not tool._tool_display_name:
            tool._tool_display_name = parent.name

        def _parent_source_fetcher(parent_uid: str = parent.uid) -> xr.DataArray:
            return self._manager._node_for_target(parent_uid).current_source_data()

        def _parent_provenance_fetcher(
            parent_uid: str = parent.uid,
        ) -> provenance.ToolProvenanceSpec | None:
            return self._manager._node_for_target(parent_uid).displayed_provenance_spec

        tool.set_source_parent_fetcher(_parent_source_fetcher)
        tool.set_input_provenance_parent_fetcher(_parent_provenance_fetcher)
        self._manager._register_child_node(node)
        self._manager.tree_view.childtool_added(node.uid, index)
        self._manager._mark_node_added(node.uid)
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
        provenance_spec: provenance.ToolProvenanceSpec | None = None,
        source_spec: provenance.ToolProvenanceSpec | None = None,
        source_binding: provenance.ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        output_id: str | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
    ) -> str:
        parent_node = self._manager._node_for_target(parent)
        if source_spec is None and source_binding is not None:
            source_spec = source_binding.materialize(parent_node.current_source_data())
        if provenance_spec is None and source_spec is not None:
            provenance_spec = provenance.compose_display_provenance(
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
            created_time=created_time,
        )
        self._manager._register_child_node(node)
        if output_id is not None and parent_node.tool_window is not None:
            parent_node.tool_window._register_output_imagetool_target(
                output_id, node.uid
            )
        self._manager.tree_view.childtool_added(node.uid, parent)
        self._manager._mark_node_added(node.uid)
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
        tool = node.tool_window
        if tool is None or not erlab.interactive.utils.qt_is_valid(tool):
            self._manager._remove_childtool(uid)
            raise KeyError(f"No child tool with UID {uid} found")
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
        self._manager._mark_removed_subtree_dirty(uid)
        self._manager.tree_view.childtool_removed(uid)
        self._manager._remove_uid_target(uid)

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
