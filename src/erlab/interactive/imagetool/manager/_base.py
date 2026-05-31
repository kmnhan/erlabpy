"""Runtime base implementation shared by ImageTool manager mixins."""

from __future__ import annotations

__all__ = ["_ImageToolManagerBase"]

import contextlib
import threading
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.manager._dialogs import _NameFilterDialog
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    import datetime
    import pathlib
    import types
    from collections.abc import Callable, Iterable, Mapping

    import numpy as np
    import xarray as xr

    from erlab.interactive._dask import DaskMenu
    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool.manager._console import (
        _ImageToolManagerJupyterConsole,
    )
    from erlab.interactive.imagetool.manager._io import _MultiFileHandler
    from erlab.interactive.imagetool.manager._modelview import _ImageToolWrapperTreeView
    from erlab.interactive.imagetool.manager._registry import _ManagerRecord
    from erlab.interactive.imagetool.manager._server import (
        _ManagerServer,
        _WatcherServer,
    )
    from erlab.interactive.imagetool.manager._tool_graph import _ManagerToolGraph
    from erlab.interactive.imagetool.manager._widgets import (
        _HeightForWidthFrame,
        _SingleImagePreview,
        _StandaloneAppSpec,
        _WarningNotificationHandler,
    )
    from erlab.interactive.imagetool.manager._workspace_io import (
        _WorkspaceDocumentAccess,
    )
    from erlab.interactive.imagetool.manager._workspace_state import (
        _ManagerWorkspaceState,
    )
    from erlab.interactive.imagetool.provenance_framework import (
        ScriptInputDependencyRef,
        ToolProvenanceSpec,
    )
    from erlab.interactive.imagetool.provenance_operations import (
        ImageToolSelectionSourceBinding,
    )
    from erlab.interactive.imagetool.viewer import ImageSlicerArea
    from erlab.interactive.ptable import PeriodicTableWindow


class _ImageToolManagerBase(QtWidgets.QMainWindow):
    """Concrete manager core shared by feature mixins."""

    # Actions and widgets installed by ImageToolManager.__init__.
    console: _ImageToolManagerJupyterConsole
    concat_action: QtGui.QAction
    compact_workspace_action: QtGui.QAction
    duplicate_action: QtGui.QAction
    hide_action: QtGui.QAction
    link_action: QtGui.QAction
    metadata_derivation_list: QtWidgets.QListWidget
    metadata_details_layout: QtWidgets.QGridLayout
    metadata_details_widget: _HeightForWidthFrame
    metadata_group: _HeightForWidthFrame
    offload_action: QtGui.QAction
    open_recent_menu: QtWidgets.QMenu
    preview_widget: _SingleImagePreview
    promote_action: QtGui.QAction
    reload_action: QtGui.QAction
    remove_action: QtGui.QAction
    rename_action: QtGui.QAction
    save_action: QtGui.QAction
    save_as_action: QtGui.QAction
    show_action: QtGui.QAction
    source_update_action: QtGui.QAction
    store_action: QtGui.QAction
    text_box: QtWidgets.QTextEdit
    tree_view: _ImageToolWrapperTreeView
    unlink_action: QtGui.QAction
    unwatch_action: QtGui.QAction

    # Runtime services and manager state.
    manager_index: int
    menu_bar: QtWidgets.QMenuBar
    server: _ManagerServer
    sigLinkersChanged: QtCore.SignalInstance
    watcher_server: _WatcherServer
    _additional_windows: dict[str, QtWidgets.QWidget]
    _alert_dialogs: list[erlab.interactive.utils.MessageDialog]
    _application_quit_filter: QtCore.QObject | None
    _bulk_remove_depth: int
    _dask_menu: DaskMenu
    _dependency_ref_cache: dict[str, tuple[int, tuple[ScriptInputDependencyRef, ...]]]
    _file_handlers: set[_MultiFileHandler]
    _ignored_warning_messages: set[str]
    _kb_filter: erlab.interactive.utils.KeyboardEventFilter
    _linkers: list[erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy]
    _manager_record: _ManagerRecord
    _metadata_copy_full_action: QtGui.QAction
    _metadata_copy_selected_action: QtGui.QAction
    _metadata_detail_labels: dict[str, QtWidgets.QLabel]
    _metadata_full_code_available: bool
    _metadata_monospace_font: QtGui.QFont
    _metadata_node_uid: str | None
    _pending_linker_reload: bool
    _pending_source_refresh_targets: dict[str, set[str]]
    _pending_tool_metadata_update_uids: set[str]
    _previous_excepthook: Callable[
        [type[BaseException], BaseException, types.TracebackType | None], typing.Any
    ]
    _progress_bars: dict[int, QtWidgets.QProgressDialog]
    _recent_directory: str | None
    _recent_loader_extensions_by_filter: dict[str, dict[str, typing.Any]]
    _recent_name_filter: str | None
    _registry_heartbeat_timer: QtCore.QTimer
    _sigDataReplaced: QtCore.SignalInstance
    _sigReloadLinkers: QtCore.SignalInstance
    _sigReplyData: QtCore.SignalInstance
    _sigWatchedDataEdited: QtCore.SignalInstance
    _standalone_app_actions: dict[str, QtGui.QAction]
    _standalone_app_specs: dict[str, _StandaloneAppSpec]
    _standalone_app_windows: dict[str, QtWidgets.QWidget]
    _tool_graph: _ManagerToolGraph
    _tool_metadata_update_timer: QtCore.QTimer
    _warning_handler: _WarningNotificationHandler
    _workspace_state: _ManagerWorkspaceState

    @property
    def _status_bar(self) -> QtWidgets.QStatusBar:
        return typing.cast("QtWidgets.QStatusBar", self.statusBar())

    def _make_icon_msgbox(self) -> QtWidgets.QMessageBox:
        """Create a QMessageBox with the application icon."""
        msg_box = QtWidgets.QMessageBox(self)
        style = self.style()
        if style is not None:  # pragma: no branch
            icon_size = (
                style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_MessageBoxIconSize)
                or 48
            )
            msg_box.setIconPixmap(self.windowIcon().pixmap(icon_size, icon_size))
        return msg_box

    def _node_for_target(
        self, target: int | str
    ) -> _ImageToolWrapper | _ManagedWindowNode:
        return self._tool_graph.node(target)

    def _child_node(self, uid: str) -> _ManagedWindowNode:
        return self._tool_graph.child(uid)

    def _parent_node(
        self, node: _ManagedWindowNode
    ) -> _ImageToolWrapper | _ManagedWindowNode:
        return self._tool_graph.parent(node)

    def _root_wrapper_for_uid(self, uid: str) -> _ImageToolWrapper:
        return self._tool_graph.root_for_uid(uid)

    def _node_uid_from_window(self, widget: QtWidgets.QWidget) -> str | None:
        return self._tool_graph.uid_from_window(widget)

    def _is_imagetool_target(self, target: int | str) -> bool:
        node = self._node_for_target(target)
        return node.is_imagetool

    def _selected_imagetool_targets(self) -> list[int | str]:
        targets: list[int | str] = list(self.tree_view.selected_imagetool_indices)
        targets.extend(
            uid
            for uid in self.tree_view.selected_childtool_uids
            if self._is_imagetool_target(uid)
        )
        return targets

    def _selected_tool_uids(self) -> list[str]:
        return [
            uid
            for uid in self.tree_view.selected_childtool_uids
            if not self._is_imagetool_target(uid)
        ]

    def _selected_promotable_child_imagetool_uid(self) -> str | None:
        child_imagetool_uids = [
            uid
            for uid in self.tree_view.selected_childtool_uids
            if self._is_imagetool_target(uid)
        ]
        if len(child_imagetool_uids) != 1:
            return None
        if self.tree_view.selected_imagetool_indices or self._selected_tool_uids():
            return None
        return child_imagetool_uids[0]

    def _selected_source_update_child_uid(self) -> str | None:
        selected_child_uids = self.tree_view.selected_childtool_uids
        if len(selected_child_uids) != 1 or self.tree_view.selected_imagetool_indices:
            return None
        uid = selected_child_uids[0]
        try:
            node = self._child_node(uid)
        except KeyError:
            return None
        return uid if node.has_source_binding else None

    @property
    def _reindex_lock(self) -> threading.Lock:
        if not hasattr(self, "__reindex_lock"):
            self.__reindex_lock = threading.Lock()
        return self.__reindex_lock

    @QtCore.Slot()
    def reindex(self) -> None:
        """Reset indices of ImageTool windows to be consecutive in display order."""
        with self._reindex_lock:
            self._tool_graph.reindex_roots()

        self.tree_view.refresh()
        self._mark_workspace_structure_dirty("Reindexed root windows")

    def get_imagetool(self, index: int | str) -> ImageTool:
        """Get the ImageTool object corresponding to the given target."""
        node = self._node_for_target(index)
        if not node.is_imagetool:
            raise KeyError(f"Target {index!r} is not an ImageTool")

        tool = node.imagetool
        if tool is None or not erlab.interactive.utils.qt_is_valid(tool):
            raise KeyError(f"Tool of target '{index}' is not available")
        return tool

    @QtCore.Slot(int)
    def remove_imagetool(self, index: int, *, update_view: bool = True) -> None:
        """Remove the ImageTool window corresponding to the given index."""
        if index not in self._tool_graph.root_wrappers:
            return
        wrapper = self._tool_graph.root_wrappers[index]
        self._mark_removed_subtree_dirty(wrapper.uid)
        descendant_uids = list(wrapper._childtool_indices)
        if update_view:
            self.tree_view.imagetool_removed(index)

        for uid in list(descendant_uids):
            self._remove_uid_target(uid)

        self._tool_graph.unregister_root(index)
        self._refresh_dependency_dependents(wrapper.uid)
        wrapper.dispose()
        wrapper.deleteLater()

    @contextlib.contextmanager
    def _bulk_remove_context(self) -> typing.Iterator[None]:
        outermost = self._bulk_remove_depth == 0
        self._bulk_remove_depth += 1
        if outermost:
            self._pending_linker_reload = False
            self.setUpdatesEnabled(False)
            self.tree_view.setUpdatesEnabled(False)
        try:
            yield
        finally:
            self._bulk_remove_depth -= 1
            if outermost:
                self.tree_view.setUpdatesEnabled(True)
                self.setUpdatesEnabled(True)

                if self._pending_linker_reload:
                    self._pending_linker_reload = False
                    self._cleanup_linkers()

                self._update_actions()
                self._update_info()

    def _remove_imagetools(
        self,
        indices: list[int | str],
        *,
        child_uids: list[str] | None = None,
        clear_view: bool = False,
    ) -> None:
        root_indices: list[int] = []
        child_targets: list[str] = []
        covered_child_uids: set[str] = set()
        for target in indices:
            if isinstance(target, int):
                root_indices.append(target)
                wrapper = self._tool_graph.root_wrappers.get(target)
                if wrapper is not None:
                    direct_children = list(wrapper._childtool_indices)
                    covered_child_uids.update(direct_children)
                    for child_uid in direct_children:
                        covered_child_uids.update(self._iter_descendant_uids(child_uid))
            else:
                child_targets.append(target)

        for uid in child_uids or []:
            if uid not in covered_child_uids and uid not in child_targets:
                child_targets.append(uid)

        if len(root_indices) == 0 and len(child_targets) == 0:
            return

        with self._bulk_remove_context():
            if clear_view:
                self.tree_view.clear_imagetools()

            for index in root_indices:
                self.remove_imagetool(index, update_view=not clear_view)

            for uid in child_targets:
                self._remove_childtool(uid)

    def remove_all_tools(self) -> None:
        """Remove all ImageTool windows."""
        self._remove_imagetools(
            list(self._tool_graph.root_wrappers.keys()), clear_view=True
        )

    @QtCore.Slot(int)
    def show_imagetool(self, index: int) -> None:
        """Show the ImageTool window corresponding to the given index."""
        if index in self._tool_graph.root_wrappers:
            self._tool_graph.root_wrappers[index].show()

    @QtCore.Slot()
    def _request_reload_linkers(self) -> None:
        if self._bulk_remove_depth > 0:
            self._pending_linker_reload = True
            return
        self._cleanup_linkers()

    @QtCore.Slot()
    def _cleanup_linkers(self) -> None:
        """Remove linkers with one or no children."""
        for linker in list(self._linkers):
            if linker.num_children <= 1:
                linker.unlink_all()
                self._linkers.remove(linker)
        self.sigLinkersChanged.emit()

    @property
    def explorer(self) -> _TabbedExplorer:
        widget = self._standalone_app_windows.get("explorer")
        if widget is None or not erlab.interactive.utils.qt_is_valid(widget):
            raise AttributeError("Data explorer is not initialized.")
        return typing.cast("_TabbedExplorer", widget)

    @property
    def ptable_window(self) -> PeriodicTableWindow:
        widget = self._standalone_app_windows.get("ptable")
        if widget is None or not erlab.interactive.utils.qt_is_valid(widget):
            raise AttributeError("Periodic table window is not initialized.")
        return typing.cast("PeriodicTableWindow", widget)

    def _create_explorer_window(self) -> QtWidgets.QWidget:
        from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer

        return _TabbedExplorer(
            root_path=self._recent_directory, loader_name=self._recent_loader_name
        )

    def _create_ptable_window(self) -> QtWidgets.QWidget:
        from erlab.interactive.ptable import PeriodicTableWindow

        return PeriodicTableWindow()

    @property
    def _recent_loader_name(self) -> str | None:
        if self._recent_name_filter is not None:
            for key in erlab.io.loaders:
                if (
                    self._recent_name_filter
                    in erlab.io.loaders[key].file_dialog_methods
                ):
                    return key
        return None

    def _preferred_name_filter(
        self, valid_loaders: dict[str, tuple[Callable, dict]]
    ) -> str | None:
        if self._recent_name_filter in valid_loaders:
            return self._recent_name_filter

        default_loader = erlab.interactive.options.model.io.default_loader
        if default_loader == "None" or default_loader not in erlab.io.loaders:
            return None

        return next(
            (
                name_filter
                for name_filter in erlab.io.loaders[default_loader].file_dialog_methods
                if name_filter in valid_loaders
            ),
            None,
        )

    def _select_loader_options(
        self,
        valid_loaders: dict[str, tuple[Callable, dict]],
        name_filter: str | None = None,
        *,
        sample_paths: Iterable[str | pathlib.Path] | None = None,
    ) -> tuple[str, Callable, dict[str, typing.Any]] | None:
        dialog = _NameFilterDialog(
            self,
            valid_loaders,
            loader_extensions=self._recent_loader_extensions_by_filter,
            sample_paths=sample_paths,
        )
        dialog.check_filter(name_filter or self._preferred_name_filter(valid_loaders))

        if not dialog.exec():
            return None

        selected_filter, func, kwargs = dialog.checked_filter()
        self._recent_name_filter = selected_filter
        loader_extensions = kwargs.get("loader_extensions", {})
        self._recent_loader_extensions_by_filter[selected_filter] = (
            loader_extensions.copy() if isinstance(loader_extensions, dict) else {}
        )
        return selected_filter, func, kwargs

    def _create_standalone_app_action(self, key: str) -> QtGui.QAction:
        spec = self._standalone_app_specs[key]
        action = QtGui.QAction(spec.text, self)
        action.setObjectName(f"manager_{key}_action")
        action.triggered.connect(
            lambda _checked=False, app_key=key: self._show_standalone_app(app_key)
        )
        if spec.shortcut is not None:
            action.setShortcut(spec.shortcut)
        action.setToolTip(spec.tooltip)
        if spec.icon_name is not None:
            action.setIcon(QtGui.QIcon.fromTheme(spec.icon_name))
        self._standalone_app_actions[key] = action
        return action

    def _create_standalone_app_window(self, key: str) -> QtWidgets.QWidget:
        widget = self._standalone_app_specs[key].factory()
        widget.destroyed.connect(
            lambda _=None, app_key=key: self._standalone_app_windows.pop(app_key, None)
        )
        self._standalone_app_windows[key] = widget
        return widget

    def _ensure_standalone_app(self, key: str) -> QtWidgets.QWidget:
        widget = self._standalone_app_windows.get(key)
        if widget is None or not erlab.interactive.utils.qt_is_valid(widget):
            widget = self._create_standalone_app_window(key)
        return widget

    def _show_standalone_app(self, key: str) -> QtWidgets.QWidget:
        widget = self._ensure_standalone_app(key)
        widget.show()
        widget.activateWindow()
        widget.raise_()
        return widget

    def _close_standalone_apps(self) -> None:
        for widget in tuple(self._standalone_app_windows.values()):
            if erlab.interactive.utils.qt_is_valid(widget):
                widget.close()
                widget.deleteLater()
        self._standalone_app_windows.clear()

    @staticmethod
    def _missing_manager_feature(name: str) -> typing.NoReturn:
        raise NotImplementedError(
            f"{name} is provided by ImageToolManager or a manager feature mixin"
        )

    @property
    def ntools(self) -> int:
        return self._missing_manager_feature("ntools")

    @property
    def next_idx(self) -> int:
        return self._missing_manager_feature("next_idx")

    @property
    def workspace_path(self) -> str | None:
        return self._missing_manager_feature("workspace_path")

    @property
    def is_workspace_modified(self) -> bool:
        return self._missing_manager_feature("is_workspace_modified")

    def _next_node_uid(self, preferred: str | None = None) -> str:
        return self._missing_manager_feature("_next_node_uid")

    def _register_child_node(self, node: _ManagedWindowNode) -> None:
        self._missing_manager_feature("_register_child_node")

    def _unregister_node(self, uid: str) -> None:
        self._missing_manager_feature("_unregister_node")

    def add_imagetool(
        self,
        tool: ImageTool,
        *,
        show: bool = True,
        activate: bool = False,
        watched_var: tuple[str, str] | None = None,
        watched_workspace_link_id: str | None = None,
        watched_source_label: str | None = None,
        watched_source_uid: str | None = None,
        watched_connected: bool = True,
        source_input_ndim: int | None = None,
        source_input_dtype: np.dtype[typing.Any] | str | None = None,
        uid: str | None = None,
        provenance_spec: ToolProvenanceSpec | None = None,
        source_spec: ToolProvenanceSpec | None = None,
        source_binding: ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        index: int | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
    ) -> int:
        return self._missing_manager_feature("add_imagetool")

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
        return self._missing_manager_feature("add_childtool")

    def add_imagetool_child(
        self,
        tool: ImageTool,
        parent: int | str,
        *,
        show: bool = True,
        activate: bool = True,
        uid: str | None = None,
        provenance_spec: ToolProvenanceSpec | None = None,
        source_spec: ToolProvenanceSpec | None = None,
        source_binding: ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        output_id: str | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
    ) -> str:
        return self._missing_manager_feature("add_imagetool_child")

    def get_childtool(self, uid: str) -> erlab.interactive.utils.ToolWindow:
        return self._missing_manager_feature("get_childtool")

    def show_childtool(self, uid: str) -> None:
        self._missing_manager_feature("show_childtool")

    def rename_imagetool(self, index: int, new_name: str) -> None:
        self._missing_manager_feature("rename_imagetool")

    def link_imagetools(self, *indices: int | str, link_colors: bool = True) -> None:
        self._missing_manager_feature("link_imagetools")

    def node_from_slicer_area(
        self, slicer_area: ImageSlicerArea
    ) -> _ImageToolWrapper | _ManagedWindowNode | None:
        return self._missing_manager_feature("node_from_slicer_area")

    def target_from_slicer_area(self, slicer_area: ImageSlicerArea) -> int | str | None:
        return self._missing_manager_feature("target_from_slicer_area")

    def _data_recv(
        self,
        data: list[xr.DataArray] | list[xr.Dataset],
        kwargs: dict[str, typing.Any],
        *,
        watched_var: tuple[str, str] | None = None,
        watched_metadata: Mapping[str, typing.Any] | None = None,
        show: bool | None = None,
    ) -> list[bool]:
        return self._missing_manager_feature("_data_recv")

    def _data_load(
        self, paths: list[str], loader_name: str, kwargs: dict[str, typing.Any]
    ) -> None:
        self._missing_manager_feature("_data_load")

    def _data_replace(
        self, data_list: list[xr.DataArray], indices: list[int | str]
    ) -> None:
        self._missing_manager_feature("_data_replace")

    def _data_unwatch(self, uid: str) -> None:
        self._missing_manager_feature("_data_unwatch")

    def _data_watched_update(
        self,
        varname: str,
        uid: str,
        darr: xr.DataArray,
        watched_metadata: Mapping[str, typing.Any] | None = None,
    ) -> None:
        self._missing_manager_feature("_data_watched_update")

    def _remove_watched(self, uid: str) -> None:
        self._missing_manager_feature("_remove_watched")

    def _show_watched(self, uid: str) -> None:
        self._missing_manager_feature("_show_watched")

    def _send_imagetool_data(self, index_or_uid: int | str) -> None:
        self._missing_manager_feature("_send_imagetool_data")

    def _send_watch_info(self) -> None:
        self._missing_manager_feature("_send_watch_info")

    def _add_from_multiple_files(
        self,
        loaded: list[pathlib.Path],
        queued: list[pathlib.Path],
        failed: list[pathlib.Path],
        func: Callable[..., typing.Any],
        kwargs: dict[str, typing.Any],
        retry_callback: Callable[..., typing.Any],
    ) -> None:
        self._missing_manager_feature("_add_from_multiple_files")

    def _error_creating_imagetool(self) -> None:
        self._missing_manager_feature("_error_creating_imagetool")

    def _show_operation_error(self, log_message: str, text: str) -> None:
        self._missing_manager_feature("_show_operation_error")

    def _show_workspace_save_worker_error(self, error_text: str) -> None:
        self._missing_manager_feature("_show_workspace_save_worker_error")

    def _update_actions(self) -> None:
        self._missing_manager_feature("_update_actions")

    def _update_info(self, *, uid: str | None = None) -> None:
        self._missing_manager_feature("_update_info")

    def _set_metadata_node(self, node: _ImageToolWrapper | _ManagedWindowNode) -> None:
        self._missing_manager_feature("_set_metadata_node")

    def _schedule_tool_metadata_update(self, uid: str) -> None:
        self._missing_manager_feature("_schedule_tool_metadata_update")

    def dependency_input_summary_for_uid(self, uid: str) -> str | None:
        return self._missing_manager_feature("dependency_input_summary_for_uid")

    def _refresh_dependency_dependents(self, uid: str) -> None:
        self._missing_manager_feature("_refresh_dependency_dependents")

    def _script_input_name_for_node(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str:
        return self._missing_manager_feature("_script_input_name_for_node")

    def _show_multi_input_script_result(
        self,
        data: xr.DataArray,
        input_targets: Iterable[int | str],
        *,
        operation_label: str,
        operation_code: str,
    ) -> int | None:
        return self._missing_manager_feature("_show_multi_input_script_result")

    def _rebase_loaded_workspace_dependency_refs(
        self, loaded_targets_by_uid: Mapping[str, int | str]
    ) -> None:
        self._missing_manager_feature("_rebase_loaded_workspace_dependency_refs")

    def _selected_reload_targets(
        self,
    ) -> tuple[list[int | str], dict[int | str, list[str]]] | None:
        return self._missing_manager_feature("_selected_reload_targets")

    def _iter_descendant_uids(self, uid: str) -> list[str]:
        return self._tool_graph.descendant_uids(uid)

    def _mark_removed_subtree_dirty(self, uid: str) -> None:
        for node_uid in self._tool_graph.subtree_uids(uid):
            node = self._tool_graph.nodes.get(node_uid)
            if node is not None:
                self._set_node_window_modified(node_uid, False)
                self._mark_workspace_dirty(
                    removed=node.display_text, structure="Removed window"
                )

    def _remove_uid_target(self, uid: str) -> None:
        if uid not in self._tool_graph.nodes:
            return
        subtree = self._tool_graph.subtree_uids(uid)
        subtree.reverse()
        for child_uid in subtree:
            child = self._tool_graph.nodes.get(child_uid)
            if child is None or isinstance(child, _ImageToolWrapper):
                continue
            self._unregister_node(child_uid)
            if child.tool_window is not None:
                child.tool_window.set_source_parent_fetcher(None)
                child.tool_window.set_input_provenance_parent_fetcher(None)
            child.dispose()

    def _remove_childtool(self, uid: str) -> None:
        self._missing_manager_feature("_remove_childtool")

    def _reload_target_for_child(self, uid: str) -> int | str | None:
        return self._missing_manager_feature("_reload_target_for_child")

    def _reload_source_chain_for_child(self, uid: str) -> bool:
        return self._missing_manager_feature("_reload_source_chain_for_child")

    def _refresh_source_chain_to_uid(self, uid: str) -> bool:
        return self._missing_manager_feature("_refresh_source_chain_to_uid")

    def _resume_pending_source_refreshes(self, uid: str) -> None:
        self._missing_manager_feature("_resume_pending_source_refreshes")

    def _parent_source_data_for_uid(self, uid: str) -> xr.DataArray:
        return self._missing_manager_feature("_parent_source_data_for_uid")

    def _mark_descendants_source_state(
        self,
        uid: str,
        state: _ManagedWindowNode._source_state_type,
    ) -> None:
        self._missing_manager_feature("_mark_descendants_source_state")

    def _mark_descendants_source_unavailable(self, uid: str) -> None:
        self._missing_manager_feature("_mark_descendants_source_unavailable")

    def _propagate_source_change_from_uid(
        self, uid: str, parent_data: xr.DataArray | None = None
    ) -> None:
        self._missing_manager_feature("_propagate_source_change_from_uid")

    def _refresh_manager_record(self) -> None:
        self._missing_manager_feature("_refresh_manager_record")

    def _release_workspace_lock(self) -> None:
        self._missing_manager_feature("_release_workspace_lock")

    def _update_workspace_window_title(self) -> None:
        self._missing_manager_feature("_update_workspace_window_title")

    def _set_node_window_modified(self, uid: str, modified: bool) -> None:
        self._missing_manager_feature("_set_node_window_modified")

    def _mark_workspace_dirty(
        self,
        *,
        uid: str | None = None,
        data: bool = False,
        state: bool = False,
        added: bool = False,
        removed: str | None = None,
        structure: str | None = None,
    ) -> None:
        self._missing_manager_feature("_mark_workspace_dirty")

    def _mark_node_added(self, uid: str) -> None:
        self._missing_manager_feature("_mark_node_added")

    def _mark_node_data_dirty(self, uid: str) -> None:
        self._missing_manager_feature("_mark_node_data_dirty")

    def _mark_node_state_dirty(self, uid: str) -> None:
        self._missing_manager_feature("_mark_node_state_dirty")

    def _mark_tool_info_dirty(self, uid: str) -> None:
        self._missing_manager_feature("_mark_tool_info_dirty")

    def _mark_workspace_structure_dirty(self, reason: str) -> None:
        self._missing_manager_feature("_mark_workspace_structure_dirty")

    def _install_workspace_save_shortcut(self, widget: QtWidgets.QWidget) -> None:
        self._missing_manager_feature("_install_workspace_save_shortcut")

    def _workspace_load_context(self) -> contextlib.AbstractContextManager[None]:
        return self._missing_manager_feature("_workspace_load_context")

    def _workspace_document_access_context(
        self, fname: str | pathlib.Path
    ) -> contextlib.AbstractContextManager[_WorkspaceDocumentAccess]:
        return self._missing_manager_feature("_workspace_document_access_context")

    def _confirm_save_dirty_workspace(self, action_text: str) -> bool:
        return self._missing_manager_feature("_confirm_save_dirty_workspace")

    def _compact_workspace_before_shutdown(self) -> None:
        self._missing_manager_feature("_compact_workspace_before_shutdown")

    def _associate_loaded_workspace_file(
        self,
        fname: str | pathlib.Path,
        schema_version: int,
        *,
        native: bool = True,
        delta_save_count: int = 0,
        workspace_access: _WorkspaceDocumentAccess | None = None,
        rebind_data: bool = True,
    ) -> None:
        self._missing_manager_feature("_associate_loaded_workspace_file")

    def _from_datatree(
        self,
        tree: xr.DataTree,
        *,
        replace: bool = False,
        mark_dirty: bool = True,
        select: bool = True,
        workspace_file_path: str | pathlib.Path | None = None,
    ) -> bool:
        return self._missing_manager_feature("_from_datatree")

    def _is_datatree_workspace(self, tree: xr.DataTree) -> bool:
        return self._missing_manager_feature("_is_datatree_workspace")

    def _load_workspace_file(
        self,
        fname: str | pathlib.Path,
        *,
        replace: bool,
        associate: bool,
        mark_dirty: bool,
        select: bool,
        native: bool = True,
    ) -> bool:
        return self._missing_manager_feature("_load_workspace_file")

    def offload_to_workspace(
        self, targets: Iterable[int | str], *, native: bool = True
    ) -> bool:
        return self._missing_manager_feature("offload_to_workspace")

    def open_recent_workspace(self, fname: str | pathlib.Path) -> bool:
        return self._missing_manager_feature("open_recent_workspace")

    def _refresh_open_recent_menu_action(self) -> None:
        self._missing_manager_feature("_refresh_open_recent_menu_action")
