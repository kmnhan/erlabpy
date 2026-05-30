"""Typing contract shared by ImageTool manager mixins."""

from __future__ import annotations

__all__ = ["_ManagerMixinBase"]


import typing

if typing.TYPE_CHECKING:
    import contextlib
    import datetime
    import pathlib
    from collections.abc import Callable, Iterable, Mapping

    import xarray as xr
    from qtpy import QtCore, QtGui, QtWidgets

    import erlab
    import erlab.interactive.imagetool.slicer
    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool.manager import _workspace as _manager_workspace
    from erlab.interactive.imagetool.manager._io import _MultiFileHandler
    from erlab.interactive.imagetool.manager._modelview import _ImageToolWrapperTreeView
    from erlab.interactive.imagetool.manager._server import (
        _ManagerServer,
        _WatcherServer,
    )
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )
    from erlab.interactive.imagetool.provenance_framework import (
        ScriptInputDependencyRef,
        ToolProvenanceSpec,
    )
    from erlab.interactive.imagetool.provenance_operations import (
        ImageToolSelectionSourceBinding,
    )
    from erlab.interactive.imagetool.viewer import ImageSlicerArea

    class _HeightForWidthFrameLike(QtWidgets.QFrame):
        def sync_height_for_width(self) -> None: ...

    class _SingleImagePreviewLike(QtWidgets.QGraphicsView):
        def setPixmap(self, pixmap: QtGui.QPixmap) -> None: ...

    class _ManagerMixinBase(QtWidgets.QMainWindow):
        """Concrete ImageToolManager host contract used by manager mixins."""

        console: typing.Any
        hide_action: QtGui.QAction
        link_action: QtGui.QAction
        unlink_action: QtGui.QAction
        duplicate_action: QtGui.QAction
        promote_action: QtGui.QAction
        offload_action: QtGui.QAction
        concat_action: QtGui.QAction
        store_action: QtGui.QAction
        reload_action: QtGui.QAction
        unwatch_action: QtGui.QAction
        source_update_action: QtGui.QAction
        save_action: QtGui.QAction
        save_as_action: QtGui.QAction
        compact_workspace_action: QtGui.QAction
        manager_index: int
        menu_bar: QtWidgets.QMenuBar
        metadata_derivation_list: QtWidgets.QListWidget
        metadata_details_layout: QtWidgets.QGridLayout
        metadata_details_widget: typing.Any
        metadata_group: typing.Any
        open_recent_menu: QtWidgets.QMenu
        preview_widget: typing.Any
        remove_action: QtGui.QAction
        rename_action: QtGui.QAction
        server: _ManagerServer
        show_action: QtGui.QAction
        sigLinkersChanged: typing.Any
        text_box: QtWidgets.QTextEdit
        tree_view: _ImageToolWrapperTreeView
        watcher_server: _WatcherServer

        _additional_windows: dict[str, QtWidgets.QWidget]
        _alert_dialogs: list[erlab.interactive.utils.MessageDialog]
        _all_nodes: dict[str, _ImageToolWrapper | _ManagedWindowNode]
        _application_quit_filter: QtCore.QObject | None
        _bulk_remove_depth: int
        _closing_workspace_document: bool
        _dask_menu: typing.Any
        _dependency_ref_cache: dict[
            str, tuple[int, tuple[ScriptInputDependencyRef, ...]]
        ]
        _displayed_indices: list[int]
        _file_handlers: set[_MultiFileHandler]
        _ignored_warning_messages: set[str]
        _imagetool_wrappers: dict[int, _ImageToolWrapper]
        _kb_filter: erlab.interactive.utils.KeyboardEventFilter
        _linkers: list[erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy]
        _manager_record: typing.Any
        _metadata_copy_full_action: QtGui.QAction
        _metadata_copy_selected_action: QtGui.QAction
        _metadata_detail_labels: dict[str, QtWidgets.QLabel]
        _metadata_full_code_available: bool
        _metadata_monospace_font: QtGui.QFont
        _metadata_node_uid: str | None
        _node_uid_counter: int
        _pending_linker_reload: bool
        _pending_source_refresh_targets: dict[str, set[str]]
        _pending_tool_metadata_update_uids: set[str]
        _previous_excepthook: typing.Any
        _progress_bars: dict[int, QtWidgets.QProgressDialog]
        _recent_directory: str | None
        _recent_loader_extensions_by_filter: dict[str, dict[str, typing.Any]]
        _recent_name_filter: str | None
        _registry_heartbeat_timer: QtCore.QTimer
        _sigDataReplaced: typing.Any
        _sigReloadLinkers: typing.Any
        _sigReplyData: typing.Any
        _sigWatchedDataEdited: typing.Any
        _standalone_app_actions: dict[str, QtGui.QAction]
        _standalone_app_specs: dict[str, typing.Any]
        _standalone_app_windows: dict[str, QtWidgets.QWidget]
        _tool_metadata_update_timer: QtCore.QTimer
        _warning_handler: typing.Any
        _workspace_delta_save_count: int
        _workspace_dirty_added: set[str]
        _workspace_dirty_data: set[str]
        _workspace_dirty_events: list[_manager_workspace._WorkspaceDirtyEvent]
        _workspace_dirty_generation: int
        _workspace_dirty_removed: list[str]
        _workspace_dirty_state: set[str]
        _workspace_link_id: str
        _workspace_loading_depth: int
        _workspace_lock: QtCore.QLockFile | None
        _workspace_needs_full_save: bool
        _workspace_path: pathlib.Path | None
        _workspace_save_in_progress: bool
        _workspace_saving_depth: int
        _workspace_schema_version: int
        _workspace_structure_modified: bool
        _workspace_structure_reasons: list[str]

        @property
        def _recent_loader_name(self) -> str | None: ...

        @property
        def _status_bar(self) -> QtWidgets.QStatusBar: ...

        @property
        def current_indices(self) -> list[int]: ...

        @property
        def is_workspace_modified(self) -> bool: ...

        @property
        def next_idx(self) -> int: ...

        @property
        def ntools(self) -> int: ...

        @property
        def workspace_path(self) -> str | None: ...

        def _add_wrapper(
            self,
            wrapper: _ImageToolWrapper,
            *,
            select: bool = True,
            mark_added: bool = True,
        ) -> int: ...

        def _associate_loaded_workspace_file(
            self,
            fname: str | pathlib.Path,
            schema_version: int,
            *,
            native: bool = True,
            delta_save_count: int = 0,
            workspace_access: typing.Any = None,
            rebind_data: bool = True,
        ) -> None: ...

        def _bulk_remove_context(
            self,
        ) -> contextlib.AbstractContextManager[None]: ...

        def _child_node(self, uid: str) -> _ManagedWindowNode: ...

        def _cleanup_linkers(self) -> None: ...

        def _compact_workspace_before_shutdown(self) -> None: ...

        def _confirm_save_dirty_workspace(self, message: str) -> bool: ...

        def _data_load(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...

        def _data_recv(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...

        def _data_replace(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...

        def _data_unwatch(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...

        def _data_watched_update(
            self, *args: typing.Any, **kwargs: typing.Any
        ) -> None: ...

        def _from_datatree(
            self,
            tree: xr.DataTree,
            *,
            replace: bool = True,
            mark_dirty: bool = True,
            select: bool = False,
            workspace_file_path: str | pathlib.Path | None = None,
        ) -> bool: ...

        def _handle_uncaught_exception(
            self,
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_traceback: typing.Any,
        ) -> None: ...

        def _is_datatree_workspace(self, dt: xr.DataTree) -> bool: ...

        def _is_imagetool_target(self, target: int | str) -> bool: ...

        def _iter_descendant_uids(self, uid: str) -> list[str]: ...

        def _load_workspace_file(
            self,
            fname: str | pathlib.Path,
            *,
            replace: bool,
            associate: bool,
            mark_dirty: bool,
            select: bool,
            native: bool = True,
        ) -> bool: ...

        def _mark_node_added(self, uid: str) -> None: ...

        def _mark_node_data_dirty(self, uid: str) -> None: ...

        def _mark_node_state_dirty(self, uid: str) -> None: ...

        def _mark_removed_subtree_dirty(self, uid: str) -> None: ...

        def _mark_workspace_dirty(
            self,
            *,
            uid: str | None = None,
            data: bool = False,
            state: bool = False,
            added: bool = False,
            removed: str | None = None,
            structure: str | None = None,
        ) -> None: ...

        def _mark_workspace_structure_dirty(self, reason: str) -> None: ...

        def _next_node_uid(self, requested_uid: str | None = None) -> str: ...

        def _node_for_target(
            self, target: int | str
        ) -> _ImageToolWrapper | _ManagedWindowNode: ...

        def _node_uid_from_window(self, window: QtWidgets.QWidget) -> str | None: ...

        def _parent_node(
            self, node: _ImageToolWrapper | _ManagedWindowNode
        ) -> _ImageToolWrapper | _ManagedWindowNode: ...

        def _refresh_dependency_dependents(self, uid: str) -> None: ...

        def _refresh_manager_record(self) -> None: ...

        def _rebase_loaded_workspace_dependency_refs(
            self, loaded_targets_by_uid: Mapping[str, int | str]
        ) -> None: ...

        def _refresh_open_recent_menu_action(self) -> None: ...

        def _register_child_node(self, node: _ManagedWindowNode) -> None: ...

        def _release_workspace_lock(self) -> None: ...

        def _remove_childtool(self, uid: str) -> None: ...

        def _remove_uid_target(self, uid: str) -> None: ...

        def _remove_watched(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...

        def _remove_imagetools(
            self,
            indices: list[int | str],
            *,
            child_uids: list[str] | None = None,
            clear_view: bool = False,
        ) -> None: ...

        def _request_reload_linkers(self) -> None: ...

        def _root_wrapper_for_uid(self, uid: str) -> _ImageToolWrapper: ...

        def _selected_imagetool_targets(self) -> list[int | str]: ...

        def _selected_promotable_child_imagetool_uid(self) -> str | None: ...

        def _selected_reload_targets(
            self,
        ) -> tuple[list[int | str], dict[int | str, list[str]]] | None: ...

        def _selected_source_update_child_uid(self) -> str | None: ...

        def _selected_tool_uids(self) -> list[str]: ...

        def _send_imagetool_data(
            self, *args: typing.Any, **kwargs: typing.Any
        ) -> None: ...

        def _send_watch_info(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...

        def _add_from_multiple_files(
            self,
            loaded: list[pathlib.Path],
            queued: list[pathlib.Path],
            failed: list[pathlib.Path],
            func: Callable[..., typing.Any],
            kwargs: dict[str, typing.Any],
            retry_callback: Callable[..., typing.Any],
        ) -> None: ...

        def _error_creating_imagetool(self) -> None: ...

        def _preferred_name_filter(
            self, valid_loaders: dict[str, tuple[typing.Any, dict]]
        ) -> str | None: ...

        def _select_loader_options(
            self,
            valid_loaders: dict[str, tuple[typing.Any, dict]],
            name_filter: str | None = None,
            *,
            sample_paths: Iterable[str | pathlib.Path] | None = None,
        ) -> tuple[str, typing.Any, dict[str, typing.Any]] | None: ...

        def _show_standalone_app(self, key: str) -> QtWidgets.QWidget: ...

        def _set_metadata_node(
            self, node: _ImageToolWrapper | _ManagedWindowNode
        ) -> None: ...

        def _set_node_window_modified(self, uid: str, modified: bool) -> None: ...

        def _show_watched(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...

        def _show_operation_error(self, log_message: str, text: str) -> None: ...

        def _show_workspace_save_worker_error(self, error_text: str) -> None: ...

        def _unregister_node(self, uid: str) -> None: ...

        def _update_actions(self) -> None: ...

        def _update_info(self) -> None: ...

        def _update_workspace_window_title(self) -> None: ...

        def _workspace_document_access_context(
            self, fname: str | pathlib.Path
        ) -> contextlib.AbstractContextManager[typing.Any]: ...

        def _workspace_load_context(
            self,
        ) -> contextlib.AbstractContextManager[None]: ...

        def add_imagetool(
            self,
            tool: ImageTool,
            *,
            show: bool = True,
            activate: bool = True,
            watched_var: tuple[str, str] | None = None,
            watched_workspace_link_id: str | None = None,
            watched_source_label: str | None = None,
            watched_source_uid: str | None = None,
            watched_connected: bool = False,
            source_input_ndim: int | None = None,
            source_input_dtype: typing.Any = None,
            uid: str | None = None,
            provenance_spec: ToolProvenanceSpec | None = None,
            source_spec: ToolProvenanceSpec | None = None,
            source_binding: ImageToolSelectionSourceBinding | None = None,
            source_auto_update: bool = False,
            source_state: typing.Literal["fresh", "stale", "unavailable"] = "fresh",
            index: int | None = None,
            snapshot_token: str | None = None,
            created_time: datetime.datetime | str | bytes | None = None,
        ) -> int: ...

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
            source_state: typing.Literal["fresh", "stale", "unavailable"] = "fresh",
            output_id: str | None = None,
            snapshot_token: str | None = None,
            created_time: datetime.datetime | str | bytes | None = None,
        ) -> str: ...

        def dependency_status_for_uid(
            self, uid: str
        ) -> typing.Literal["current", "changed", "missing"] | None: ...

        def add_childtool(
            self,
            tool: erlab.interactive.utils.ToolWindow,
            index: int | str,
            *,
            show: bool = True,
            uid: str | None = None,
            snapshot_token: str | None = None,
            created_time: datetime.datetime | str | bytes | None = None,
        ) -> str: ...

        def get_imagetool(self, index: int | str) -> ImageTool: ...

        def get_childtool(self, uid: str) -> erlab.interactive.utils.ToolWindow: ...

        def link_imagetools(
            self, *indices: int | str, link_colors: bool = True
        ) -> None: ...

        def node_from_slicer_area(
            self, slicer_area: ImageSlicerArea
        ) -> _ImageToolWrapper | _ManagedWindowNode | None: ...

        def offload_to_workspace(
            self, targets: Iterable[int | str], *, native: bool = True
        ) -> bool: ...

        def target_from_slicer_area(
            self, slicer_area: ImageSlicerArea
        ) -> int | str | None: ...

        def open_recent_workspace(self, fname: str | pathlib.Path) -> bool: ...

        def remove_all_tools(self) -> None: ...

        def remove_imagetool(self, index: int, *, update_view: bool = True) -> None: ...

        def show_childtool(self, uid: str) -> None: ...

        def show_imagetool(self, index: int) -> None: ...

else:

    class _ManagerMixinBase:
        pass
