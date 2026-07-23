"""Runtime base implementation shared by ImageToolManager."""

from __future__ import annotations

__all__ = ["_ImageToolManagerBase"]

import os
import threading
import typing
import weakref

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._options.core
from erlab.interactive import _qt_state
from erlab.interactive.imagetool.manager._dialogs import _NameFilterDialog

if typing.TYPE_CHECKING:
    import pathlib
    import types
    from collections.abc import Callable, Iterable, Mapping

    from erlab.interactive._dask import DaskMenu
    from erlab.interactive._options.schema import AppOptions
    from erlab.interactive.explorer._tabbed_explorer import _TabbedExplorer
    from erlab.interactive.imagetool._mainwindow import ImageTool
    from erlab.interactive.imagetool.manager._acquisition_context import (
        _AcquisitionContextController,
    )
    from erlab.interactive.imagetool.manager._actions import _ActionsController
    from erlab.interactive.imagetool.manager._console import (
        _ImageToolManagerJupyterConsole,
    )
    from erlab.interactive.imagetool.manager._dependency import (
        _ManagerDependencyTracker,
    )
    from erlab.interactive.imagetool.manager._details_panel import (
        _DetailsPanelController,
    )
    from erlab.interactive.imagetool.manager._figurecomposer._collection import (
        _FigureCollectionController,
    )
    from erlab.interactive.imagetool.manager._figurecomposer._controller import (
        _FigureComposerWorkflowController,
    )
    from erlab.interactive.imagetool.manager._heartbeat import (
        _RegistryHeartbeatController,
    )
    from erlab.interactive.imagetool.manager._interaction import _ManagerInteractionGate
    from erlab.interactive.imagetool.manager._io import _MultiFileHandler
    from erlab.interactive.imagetool.manager._lineage import _LineageController
    from erlab.interactive.imagetool.manager._linking import _ManagerLinkRegistry
    from erlab.interactive.imagetool.manager._metadata import _ManagerToolMetadataQueue
    from erlab.interactive.imagetool.manager._metadata_editor import (
        _MetadataEditorController,
    )
    from erlab.interactive.imagetool.manager._modelview import _ImageToolWrapperTreeView
    from erlab.interactive.imagetool.manager._provenance_edit import (
        _ProvenanceEditController,
    )
    from erlab.interactive.imagetool.manager._registry import _ManagerRecord
    from erlab.interactive.imagetool.manager._server import (
        _ManagerServer,
        _WatcherServer,
    )
    from erlab.interactive.imagetool.manager._tool_graph import _ManagerToolGraph
    from erlab.interactive.imagetool.manager._widgets import (
        _HeightForWidthFrame,
        _MetadataDerivationListWidget,
        _SingleImagePreview,
        _StandaloneAppSpec,
        _WarningNotificationHandler,
        _WidgetsController,
    )
    from erlab.interactive.imagetool.manager._workspace._controller import (
        _WorkspaceController,
    )
    from erlab.interactive.imagetool.manager._workspace._state import (
        _ManagerWorkspaceState,
    )
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )
    from erlab.interactive.ptable import PeriodicTableWindow


class _StandaloneAppEventFilter(QtCore.QObject):
    _STATE_EVENT_TYPES = frozenset(
        {
            QtCore.QEvent.Type.Hide,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Show,
            QtCore.QEvent.Type.WindowStateChange,
        }
    )

    def __init__(self, manager: _ImageToolManagerBase) -> None:
        super().__init__(manager)
        self._manager = manager

    def eventFilter(
        self, obj: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if event is not None and event.type() in self._STATE_EVENT_TYPES:
            self._manager._mark_standalone_app_state_dirty()
        return super().eventFilter(obj, event)


def _loader_name_for_callable(func: Callable) -> str | None:
    loader = getattr(func, "__self__", None)
    return loader.name if isinstance(loader, erlab.io.dataloader.LoaderBase) else None


def _loader_kwargs_without_filter_defaults(
    kwargs: Mapping[str, typing.Any], defaults: Mapping[str, typing.Any]
) -> dict[str, typing.Any]:
    """Return loader-wide overrides, excluding file-dialog method defaults."""
    return {
        key: value
        for key, value in kwargs.items()
        if key not in defaults or value != defaults[key]
    }


class _ImageToolManagerBase(QtWidgets.QMainWindow):
    """Concrete Qt core for ImageToolManager."""

    # Actions and widgets installed by ImageToolManager.__init__.
    console: _ImageToolManagerJupyterConsole
    acquisition_context_action: QtGui.QAction
    acquisition_context_status_button: QtWidgets.QToolButton
    metadata_editor_action: QtGui.QAction
    concat_action: QtGui.QAction
    compact_workspace_action: QtGui.QAction
    create_figure_action: QtGui.QAction
    duplicate_action: QtGui.QAction
    edit_note_action: QtGui.QAction
    copy_note_action: QtGui.QAction
    clear_note_action: QtGui.QAction
    hide_action: QtGui.QAction
    left_tabs: QtWidgets.QTabWidget
    link_action: QtGui.QAction
    main_splitter: QtWidgets.QSplitter
    metadata_derivation_list: _MetadataDerivationListWidget
    metadata_details_layout: QtWidgets.QGridLayout
    metadata_details_widget: _HeightForWidthFrame
    metadata_details_page: QtWidgets.QWidget
    metadata_group: QtWidgets.QFrame
    metadata_provenance_page: QtWidgets.QWidget
    inspector_tabs: QtWidgets.QTabWidget
    notes_page: QtWidgets.QWidget
    notes_title_label: QtWidgets.QLabel
    notes_kind_label: QtWidgets.QLabel
    notes_editor: QtWidgets.QPlainTextEdit
    notes_copy_button: QtWidgets.QToolButton
    notes_clear_button: QtWidgets.QToolButton
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
    right_splitter: QtWidgets.QSplitter
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
    _actions_controller: _ActionsController
    _acquisition_context: _AcquisitionContextController
    _bulk_remove_depth: int
    _dask_menu: DaskMenu
    _dependency_tracker: _ManagerDependencyTracker
    _details_panel: _DetailsPanelController
    _file_handlers: set[_MultiFileHandler]
    _figure_collection: _FigureCollectionController
    _figure_workflows: _FigureComposerWorkflowController
    _ignored_warning_messages: set[str]
    _kb_filter: erlab.interactive.utils.KeyboardEventFilter
    _link_registry: _ManagerLinkRegistry
    _lineage_controller: _LineageController
    _manager_record: _ManagerRecord
    _manager_layout_tracking_enabled: bool
    _metadata_copy_full_action: QtGui.QAction
    _metadata_copy_selected_action: QtGui.QAction
    _metadata_delete_step_action: QtGui.QAction
    _metadata_edit_step_action: QtGui.QAction
    _metadata_paste_steps_action: QtGui.QAction
    _metadata_reorder_steps_action: QtGui.QAction
    _metadata_revert_step_action: QtGui.QAction
    _metadata_detail_labels: dict[str, QtWidgets.QLabel]
    _metadata_full_code_available: bool
    _metadata_monospace_font: QtGui.QFont
    _metadata_editor: _MetadataEditorController
    _metadata_node_uid: str | None
    _notes_node_uid: str | None
    _note_commit_timer: QtCore.QTimer
    _provenance_paste_filter: QtCore.QObject | None
    _updating_note_editor: bool
    _previous_excepthook: Callable[
        [type[BaseException], BaseException, types.TracebackType | None], typing.Any
    ]
    _progress_bars: dict[int, QtWidgets.QProgressDialog]
    _provenance_edit_controller: _ProvenanceEditController
    _recent_directory: str | None
    _recent_loader_kwargs_by_filter: dict[str, dict[str, typing.Any]]
    _recent_loader_extensions_by_filter: dict[str, dict[str, typing.Any]]
    _recent_name_filter: str | None
    _registry_heartbeat: _RegistryHeartbeatController
    _registry_heartbeat_timer: QtCore.QTimer
    _interaction_gate: _ManagerInteractionGate
    _sigDataReplaced: QtCore.SignalInstance
    _sigReloadLinkers: QtCore.SignalInstance
    _sigReplyData: QtCore.SignalInstance
    _sigWatchedDataEdited: QtCore.SignalInstance
    _trusted_script_replay_keys: set[str]
    _standalone_app_actions: dict[str, QtGui.QAction]
    _standalone_app_event_filters: dict[str, QtCore.QObject]
    _standalone_app_pending_states: dict[str, dict[str, typing.Any]]
    _standalone_app_specs: dict[str, _StandaloneAppSpec]
    _standalone_app_windows: dict[str, QtWidgets.QWidget]
    _tool_graph: _ManagerToolGraph
    _tool_metadata_queue: _ManagerToolMetadataQueue
    _warning_handler: _WarningNotificationHandler
    _workspace_controller: _WorkspaceController
    _workspace_state: _ManagerWorkspaceState
    _widgets_controller: _WidgetsController

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

    @QtCore.Slot()
    def _mark_workspace_layout_dirty(self) -> None:
        self._workspace_controller._mark_workspace_layout_dirty()

    def _mark_workspace_options_dirty(self) -> None:
        self._workspace_controller._mark_workspace_options_dirty()

    def workspace_option_overrides(self) -> dict[str, typing.Any]:
        """Return manager-local workspace option overrides."""
        return dict(self._workspace_state.option_overrides)

    def _set_workspace_option_overrides(
        self,
        overrides: typing.Mapping[str, typing.Any],
        *,
        mark_dirty: bool = True,
    ) -> None:
        normalized = (
            erlab.interactive._options.core.normalize_workspace_option_overrides(
                overrides
            )
        )
        if self._workspace_state.option_overrides == normalized:
            return
        self._workspace_state.option_overrides = normalized
        if mark_dirty:
            self._mark_workspace_options_dirty()

    def set_workspace_option_override(self, path: str, value: typing.Any) -> None:
        overrides = self.workspace_option_overrides()
        overrides[str(path)] = value
        self._set_workspace_option_overrides(overrides)

    def clear_workspace_option_override(self, path: str) -> None:
        overrides = self.workspace_option_overrides()
        try:
            del overrides[str(path)]
        except KeyError:
            return
        self._set_workspace_option_overrides(overrides)

    @property
    def effective_interactive_options(self) -> AppOptions:
        """Return user settings with this manager's workspace overrides applied."""
        return erlab.interactive._options.core.model_with_workspace_overrides(
            erlab.interactive.options.model,
            self._workspace_state.option_overrides,
        )

    def _recent_or_default_directory(self) -> str | None:
        if self._recent_directory is not None:
            return self._recent_directory
        default_directory = self.effective_interactive_options.io.default_directory
        if default_directory:
            return os.path.expanduser(default_directory)
        return None

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

    def _is_figure_node(self, node: _ImageToolWrapper | _ManagedWindowNode) -> bool:
        return node.uid in self._tool_graph.figure_uids or (
            node.tool_window is not None
            and node.tool_window.manager_collection == "figures"
        )

    def _is_figure_uid(self, uid: str) -> bool:
        try:
            node = self._child_node(uid)
        except KeyError:
            return False
        return self._is_figure_node(node)

    def _selected_figure_uids(self) -> list[str]:
        return self._figure_collection.selected_uids()

    def _selected_imagetool_targets(self) -> list[int | str]:
        targets: list[int | str] = list(self.tree_view.selected_imagetool_indices)
        targets.extend(
            uid
            for uid in self.tree_view.selected_childtool_uids
            if self._is_imagetool_target(uid)
        )
        return targets

    def _selected_tool_uids(self) -> list[str]:
        selected = [
            uid
            for uid in self.tree_view.selected_childtool_uids
            if not self._is_imagetool_target(uid)
        ]
        selected.extend(self._selected_figure_uids())
        return selected

    def _selected_figure_source_targets(self) -> list[int | str]:
        return list(self._selected_imagetool_targets())

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

    def get_imagetool(self, index: int | str) -> ImageTool:
        """Get a usable ImageTool object, materializing pending workspace data.

        Cheap UI or structural state paths that do not need array values should use
        node-level access instead of this method.
        """
        node = self._node_for_target(index)
        if not node.is_imagetool:
            raise KeyError(f"Target {index!r} is not an ImageTool")
        if not node.materialize_pending_workspace_payload():
            raise ValueError(
                "Could not read this ImageTool's saved data from the workspace file."
            )

        tool = node.imagetool
        if tool is None or not erlab.interactive.utils.qt_is_valid(tool):
            raise KeyError(f"Tool of target '{index}' is not available")
        return tool

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

        loader_name = self._recent_loader_name
        if loader_name is None:
            loader_name = self.effective_interactive_options.io.default_loader
        explorer = _TabbedExplorer(
            root_path=self._recent_or_default_directory(),
            loader_name=loader_name,
        )
        loader_kwargs, loader_extensions = (
            self._workspace_controller.loading._explorer_loader_state()
        )
        explorer.apply_loader_state(
            kwargs_by_name=loader_kwargs,
            extensions_by_name=loader_extensions,
        )
        return explorer

    def _shared_loader_state(
        self,
    ) -> tuple[dict[str, dict[str, typing.Any]], dict[str, dict[str, typing.Any]]]:
        state = self._workspace_controller._loader_state
        shared_kwargs = {
            str(name): dict(kwargs)
            for name, kwargs in state.explorer_loader_kwargs_by_name.items()
        }
        shared_extensions = {
            str(name): dict(extensions)
            for name, extensions in state.explorer_loader_extensions_by_name.items()
        }
        valid_loaders = erlab.interactive.utils.file_loaders()
        for name_filter, kwargs in self._recent_loader_kwargs_by_filter.items():
            loader_entry = valid_loaders.get(name_filter)
            if loader_entry is None:
                continue
            loader_name = _loader_name_for_callable(loader_entry[0])
            if loader_name is not None:
                shared_kwargs.setdefault(
                    loader_name,
                    _loader_kwargs_without_filter_defaults(kwargs, loader_entry[1]),
                )
        for name_filter, extensions in self._recent_loader_extensions_by_filter.items():
            loader_entry = valid_loaders.get(name_filter)
            if loader_entry is None:
                continue
            loader_name = _loader_name_for_callable(loader_entry[0])
            if loader_name is not None:
                shared_extensions.setdefault(loader_name, dict(extensions))
        return shared_kwargs, shared_extensions

    def _sync_shared_loader_state(
        self,
        kwargs_by_name: Mapping[str, Mapping[str, typing.Any]],
        extensions_by_name: Mapping[str, Mapping[str, typing.Any]],
        *,
        apply_explorer: bool,
    ) -> None:
        shared_kwargs = {
            str(name): dict(kwargs) for name, kwargs in kwargs_by_name.items()
        }
        shared_extensions = {
            str(name): dict(extensions)
            for name, extensions in extensions_by_name.items()
        }
        state = self._workspace_controller._loader_state
        state.explorer_loader_kwargs_by_name = shared_kwargs
        state.explorer_loader_extensions_by_name = shared_extensions

        recent_filter = self._recent_name_filter
        if recent_filter is not None:
            loader_entry = erlab.interactive.utils.file_loaders().get(recent_filter)
            if loader_entry is not None:
                loader_name = _loader_name_for_callable(loader_entry[0])
                if loader_name is not None and loader_name in shared_kwargs:
                    self._recent_loader_kwargs_by_filter[recent_filter] = (
                        dict(loader_entry[1]) | shared_kwargs[loader_name]
                    )
                if loader_name is not None and loader_name in shared_extensions:
                    self._recent_loader_extensions_by_filter[recent_filter] = dict(
                        shared_extensions[loader_name]
                    )

        if not apply_explorer:
            return
        explorer = self._standalone_app_windows.get("explorer")
        if explorer is not None and erlab.interactive.utils.qt_is_valid(explorer):
            apply_loader_state = getattr(explorer, "apply_loader_state", None)
            if callable(apply_loader_state):
                apply_loader_state(
                    kwargs_by_name=shared_kwargs,
                    extensions_by_name=shared_extensions,
                )

    def _set_shared_loader_options(
        self,
        loader_name: str,
        kwargs: Mapping[str, typing.Any],
        extensions: Mapping[str, typing.Any],
    ) -> None:
        shared_kwargs, shared_extensions = self._shared_loader_state()
        shared_kwargs[loader_name] = dict(kwargs)
        shared_extensions[loader_name] = dict(extensions)
        self._sync_shared_loader_state(
            shared_kwargs,
            shared_extensions,
            apply_explorer=True,
        )

    @QtCore.Slot(object, object)
    def _update_shared_loader_state_from_explorer(
        self,
        kwargs_by_name: dict[str, dict[str, typing.Any]],
        extensions_by_name: dict[str, dict[str, typing.Any]],
    ) -> None:
        self._sync_shared_loader_state(
            kwargs_by_name,
            extensions_by_name,
            apply_explorer=False,
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

        default_loader = self.effective_interactive_options.io.default_loader
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
        shared_kwargs, shared_extensions = self._shared_loader_state()
        dialog_loaders: dict[str, tuple[Callable, dict[str, typing.Any]]] = {}
        dialog_extensions: dict[str, dict[str, typing.Any]] = {}
        for current_filter, (func, kwargs) in valid_loaders.items():
            dialog_kwargs = kwargs.copy()
            loader_name = _loader_name_for_callable(func)
            if loader_name is not None and loader_name in shared_kwargs:
                dialog_kwargs.update(shared_kwargs[loader_name])
            else:
                dialog_kwargs.update(
                    self._recent_loader_kwargs_by_filter.get(current_filter, {})
                )
            dialog_loaders[current_filter] = (func, dialog_kwargs)
            if loader_name is not None and loader_name in shared_extensions:
                dialog_extensions[current_filter] = shared_extensions[loader_name]
            else:
                dialog_extensions[current_filter] = (
                    self._recent_loader_extensions_by_filter.get(current_filter, {})
                )

        dialog = _NameFilterDialog(
            self,
            dialog_loaders,
            loader_extensions=dialog_extensions,
            sample_paths=sample_paths,
        )
        dialog.check_filter(name_filter or self._preferred_name_filter(dialog_loaders))

        if not dialog.exec():
            return None

        selected_filter, func, kwargs = dialog.checked_filter()
        self._recent_name_filter = selected_filter
        loader_extensions = kwargs.get("loader_extensions", {})
        selected_kwargs = kwargs.copy()
        selected_kwargs.pop("loader_extensions", None)
        self._recent_loader_kwargs_by_filter[selected_filter] = selected_kwargs
        self._recent_loader_extensions_by_filter[selected_filter] = (
            loader_extensions.copy() if isinstance(loader_extensions, dict) else {}
        )
        loader_name = _loader_name_for_callable(func)
        if loader_name is not None:
            self._set_shared_loader_options(
                loader_name,
                _loader_kwargs_without_filter_defaults(
                    selected_kwargs, valid_loaders[selected_filter][1]
                ),
                loader_extensions if isinstance(loader_extensions, dict) else {},
            )
        self._mark_workspace_layout_dirty()
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
        event_filter = _StandaloneAppEventFilter(self)
        widget.installEventFilter(event_filter)
        self._standalone_app_event_filters[key] = event_filter
        signal = getattr(widget, "sigStateChanged", None)
        if signal is not None:
            signal.connect(self._mark_standalone_app_state_dirty)
        if key == "explorer":
            loader_signal = getattr(widget, "sigLoaderStateChanged", None)
            if loader_signal is not None:
                loader_signal.connect(self._update_shared_loader_state_from_explorer)

        widget_ref = weakref.ref(widget)

        def cleanup(_obj: QtCore.QObject | None = None) -> None:
            current = self._standalone_app_windows.get(key)
            if current is not None and current is not widget_ref():
                return
            self._standalone_app_windows.pop(key, None)
            self._standalone_app_event_filters.pop(key, None)

        widget.destroyed.connect(cleanup)
        self._standalone_app_windows[key] = widget
        pending_state = self._standalone_app_pending_states.get(key)
        if pending_state is not None:
            self._apply_standalone_app_state(key, widget, pending_state)
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

    def _mark_standalone_app_state_dirty(self) -> None:
        self._mark_workspace_layout_dirty()

    def _apply_standalone_app_state(
        self,
        key: str,
        widget: QtWidgets.QWidget,
        state: dict[str, typing.Any],
    ) -> None:
        restore_state = getattr(widget, "restore_workspace_state", None)
        if callable(restore_state):
            restore_state(state)

        window_state = _qt_state.parse_qt_window_state(state.get("window_state"))
        if window_state is None:
            return
        if window_state.visible:
            widget.show()
            _qt_state.restore_qt_window_state(widget, window_state)
        else:
            _qt_state.restore_qt_window_state(widget, window_state)
            widget.hide()
        self._standalone_app_pending_states[key] = state

    def _close_standalone_app(self, key: str) -> None:
        widget = self._standalone_app_windows.get(key)
        event_filter = self._standalone_app_event_filters.get(key)
        if widget is None or not erlab.interactive.utils.qt_is_valid(widget):
            self._standalone_app_windows.pop(key, None)
            self._standalone_app_event_filters.pop(key, None)
            self._standalone_app_pending_states.pop(key, None)
            return
        if event_filter is not None:
            widget.removeEventFilter(event_filter)
        if not widget.close():
            if event_filter is not None and erlab.interactive.utils.qt_is_valid(widget):
                widget.installEventFilter(event_filter)
            return
        self._standalone_app_windows.pop(key, None)
        self._standalone_app_event_filters.pop(key, None)
        widget.deleteLater()
        self._standalone_app_pending_states.pop(key, None)

    def _close_standalone_apps(self) -> None:
        for key in tuple(self._standalone_app_windows):
            self._close_standalone_app(key)
