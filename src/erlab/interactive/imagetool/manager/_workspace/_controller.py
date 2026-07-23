"""Workspace document lifecycle, persistence, and user-facing document actions."""

from __future__ import annotations

import contextlib
import copy
import functools
import logging
import os
import pathlib
import time
import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._options.core
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._loading as workspace_loading
import erlab.interactive.imagetool.manager._workspace._saving as workspace_saving
import erlab.interactive.imagetool.manager._workspace._state as workspace_state
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.slicer
import erlab.interactive.imagetool.viewer_linking
from erlab.interactive.imagetool import _serialization
from erlab.interactive.imagetool.manager import _desktop
from erlab.interactive.imagetool.manager._widgets import (
    _RECENT_WORKSPACES_SETTINGS_KEY,
    _WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME,
    _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS,
    _manager_settings,
    _show_workspace_file_lock_error,
    _window_title_with_modified_placeholder,
    _WorkspaceDocumentAccess,
    _WorkspacePropertiesDialog,
    _WorkspacePropertiesState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator, Mapping

    import h5py
    import xarray as xr

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._workspace._state import (
        _WorkspaceStateSnapshot,
    )
    from erlab.interactive.imagetool.manager._wrapper import _ManagedWindowNode
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

logger = logging.getLogger(__name__)
_WORKSPACE_SAVE_SUFFIX_WARNING = "ImageTool Manager saves workspaces as .itws files."


class _WorkspacePostSaveBindingError(RuntimeError):
    """Raised when a saved workspace cannot be rebound into the live session."""


def _show_itws_workspace_warning(parent: QtWidgets.QWidget) -> None:
    QtWidgets.QMessageBox.warning(
        parent,
        "Workspace Not Saved",
        _WORKSPACE_SAVE_SUFFIX_WARNING,
    )


class _WorkspaceController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager
        self._loader_state = workspace_format.WorkspaceLoaderState()
        self.loading = workspace_loading._WorkspaceLoader(manager, self)
        self.saving = workspace_saving._WorkspaceSaver(manager, self)
        self._workspace_window_state_applied: tuple[str, str, bool] | None = None
        self._node_window_state_applied: dict[
            str, tuple[tuple[tuple[int, str], ...], bool]
        ] = {}
        self._pending_node_window_modified: dict[str, bool] = {}
        self._background_save_worker: workspace_saving._WorkspaceSaveWorker | None = (
            None
        )
        self._background_save_receiver: (
            workspace_saving._WorkspaceSaveResultReceiver | None
        ) = None
        self._background_save_requested = False
        self._shutdown_compaction_attempted = False

    @staticmethod
    def _normalize_recent_workspace_paths(
        paths: Iterable[str | os.PathLike[str]],
        *,
        limit: int,
    ) -> list[pathlib.Path]:
        recent_paths: list[pathlib.Path] = []
        seen: set[str] = set()
        for value in paths:
            path = pathlib.Path(value).expanduser().resolve()
            if path.suffix.lower() != ".itws":
                continue
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            recent_paths.append(path)
            seen.add(key)
            if len(recent_paths) >= limit:
                break
        return recent_paths

    def _recent_workspace_paths(self) -> list[pathlib.Path]:
        limit = erlab.interactive.options.model.io.recent_workspace_limit
        settings = _manager_settings()
        settings.sync()
        values = settings.value(_RECENT_WORKSPACES_SETTINGS_KEY, [])
        if isinstance(values, str):
            stored_paths = [values] if values else []
        elif isinstance(values, (list, tuple)):
            stored_paths = [str(value) for value in values if value]
        else:
            stored_paths = []
        recent_paths = self._normalize_recent_workspace_paths(
            stored_paths,
            limit=limit,
        )
        if len(stored_paths) > limit:
            self._set_recent_workspace_paths(recent_paths)
        return recent_paths

    def _set_recent_workspace_paths(
        self, paths: Iterable[str | os.PathLike[str]]
    ) -> None:
        recent_paths = self._normalize_recent_workspace_paths(
            paths,
            limit=erlab.interactive.options.model.io.recent_workspace_limit,
        )
        settings = _manager_settings()
        if recent_paths:
            settings.setValue(
                _RECENT_WORKSPACES_SETTINGS_KEY,
                [str(path) for path in recent_paths],
            )
        else:
            settings.remove(_RECENT_WORKSPACES_SETTINGS_KEY)
        settings.sync()

    def _record_recent_workspace(self, fname: str | os.PathLike[str]) -> None:
        path = pathlib.Path(fname).expanduser().resolve()
        if path.suffix.lower() != ".itws":
            return
        path_key = os.path.normcase(str(path))
        paths = [
            existing
            for existing in self._recent_workspace_paths()
            if os.path.normcase(str(existing)) != path_key
        ]
        self._set_recent_workspace_paths([path, *paths])
        self._refresh_open_recent_menu_action()
        if erlab.utils.misc._IS_PACKAGED:
            _desktop.record_recent_workspace(path)

    def _clear_recent_workspaces(self) -> None:
        self._set_recent_workspace_paths([])
        self._populate_open_recent_menu()

    def _refresh_open_recent_menu_action(self) -> None:
        self._manager.open_recent_menu.setEnabled(
            bool(self._recent_workspace_paths())
            and not self._manager._workspace_state.save_in_progress
        )

    def _populate_open_recent_menu(self) -> None:
        self._manager.open_recent_menu.clear()
        paths = self._recent_workspace_paths()
        self._manager.open_recent_menu.setEnabled(
            bool(paths) and not self._manager._workspace_state.save_in_progress
        )
        if not paths:
            return

        name_counts: dict[str, int] = {}
        for path in paths:
            name_counts[path.name] = name_counts.get(path.name, 0) + 1

        for index, path in enumerate(paths):
            label = path.name
            if name_counts[path.name] > 1:
                label = f"{path.name} ({path.parent.name or path.parent})"
            action = QtWidgets.QAction(label, self._manager.open_recent_menu)
            action.setObjectName(f"manager_recent_workspace_action_{index}")
            action.setData(str(path))
            action.setToolTip(str(path))
            action.setStatusTip(str(path))
            action.triggered.connect(
                lambda _checked=False, recent_path=path: (
                    self._manager.open_recent_workspace(recent_path)
                )
            )
            self._manager.open_recent_menu.addAction(action)

        self._manager.open_recent_menu.addSeparator()
        clear_action = QtWidgets.QAction("Clear Menu", self._manager.open_recent_menu)
        clear_action.setObjectName("manager_clear_recent_workspaces_action")
        clear_action.triggered.connect(self._clear_recent_workspaces)
        self._manager.open_recent_menu.addAction(clear_action)

    def _load_workspace_path(self, path: pathlib.Path, *, native: bool = True) -> bool:
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        self._manager._recent_directory = str(path.parent)
        try:
            loaded = self.loading._load_workspace_file(
                path,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
                native=native,
            )
        except Exception as exc:
            if workspace_storage._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    path,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self._manager, path)
            else:
                logger.exception(
                    "Error while loading workspace",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self._manager,
                    "Error",
                    "An error occurred while loading the workspace file.",
                )
            return False
        if loaded:
            self._record_recent_workspace(path)
        return loaded

    def _request_workspace_open(
        self, path: pathlib.Path, *, native: bool = True
    ) -> typing.Literal["opened", "scheduled", "cancelled", "failed"]:
        """Prompt for unsaved changes and open one recognized workspace path."""
        choice = self._dirty_workspace_save_choice(
            "Opening a workspace replaces the windows currently in this manager."
        )
        if choice == "cancel":
            return "cancelled"
        if choice in {"clean", "discard"}:
            loaded = self._load_workspace_path(path, native=native)
            return "opened" if loaded else "failed"

        def _continue_after_save(save_succeeded: bool) -> None:
            if save_succeeded and not self._manager.is_workspace_modified:
                self._load_workspace_path(path, native=native)

        return (
            "scheduled"
            if self.save(native=native, on_finished=_continue_after_save)
            else "failed"
        )

    def open_workspace_candidate(
        self, fname: str | os.PathLike[str], *, native: bool = True
    ) -> typing.Literal["not-workspace", "handled", "stop"]:
        """Recognize and open a path supplied by the general file-ingress flow.

        The return value tells data ingress whether the path is ordinary data,
        whether the workspace path was fully handled, or whether a cancel/deferred
        save means the rest of the current file batch must stop.
        """
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return "stop"

        path = pathlib.Path(fname).expanduser().resolve()
        explicit_workspace = workspace_format._workspace_path_is_itws(path)
        try:
            tree = workspace_arrays.open_workspace_datatree(path, chunks=None)
        except Exception as exc:
            if workspace_storage._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    path,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self._manager, path)
                return "handled"
            if explicit_workspace:
                self._manager._show_operation_error(
                    "Error while loading workspace",
                    "An error occurred while loading the workspace file.",
                )
                return "handled"
            logger.debug("Failed to open %s as datatree workspace", path, exc_info=True)
            return "not-workspace"

        try:
            is_workspace = self.loading._is_datatree_workspace(tree)
        finally:
            tree.close()
        if not is_workspace:
            if not explicit_workspace:
                return "not-workspace"
            logger.error(
                "File with .itws extension is not an ImageTool workspace: %s",
                path,
                extra={"suppress_ui_alert": True},
            )
            erlab.interactive.utils.MessageDialog.critical(
                self._manager,
                "Error",
                "An error occurred while loading the workspace file.",
                f"{path.name} is not a valid ImageTool workspace file.",
            )
            return "handled"

        status = self._request_workspace_open(path, native=native)
        if status in {"cancelled", "scheduled"}:
            return "stop"
        return "handled"

    def _open_workspace_after_dirty_prompt(
        self, fname: str | os.PathLike[str], *, native: bool = True
    ) -> bool:
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        path = pathlib.Path(fname).expanduser().resolve()
        return self._request_workspace_open(path, native=native) in {
            "opened",
            "scheduled",
        }

    def open_recent_workspace(self, fname: str | os.PathLike[str]) -> bool:
        """Open a recently used workspace file."""
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        path = pathlib.Path(fname).expanduser().resolve()
        path_key = os.path.normcase(str(path))
        if not path.exists():
            self._set_recent_workspace_paths(
                existing
                for existing in self._recent_workspace_paths()
                if os.path.normcase(str(existing)) != path_key
            )
            self._refresh_open_recent_menu_action()
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Workspace Not Found",
                f"The recent workspace file no longer exists:\n{path}",
            )
            return False
        if not workspace_format._workspace_path_is_itws(path):
            self._set_recent_workspace_paths(
                existing
                for existing in self._recent_workspace_paths()
                if os.path.normcase(str(existing)) != path_key
            )
            self._refresh_open_recent_menu_action()
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Unsupported Workspace File",
                "ImageTool Manager opens workspace files with the .itws extension.",
            )
            return False
        return self._open_workspace_after_dirty_prompt(path)

    @property
    def workspace_path(self) -> str | None:
        """Path of the workspace document associated with this manager."""
        return (
            None
            if self._manager._workspace_state.path is None
            else str(self._manager._workspace_state.path)
        )

    def show_workspace_properties(self) -> None:
        """Show properties for the workspace associated with this manager."""
        _WorkspacePropertiesDialog(
            self._manager.workspace_path,
            state=self._workspace_properties_state(),
            parent=self._manager,
        ).exec()

    def _workspace_properties_state(self) -> _WorkspacePropertiesState:
        return _WorkspacePropertiesState(
            is_modified=self._manager.is_workspace_modified,
            top_level_window_count=self._manager.ntools,
        )

    @property
    def is_workspace_modified(self) -> bool:
        """Return whether this workspace has unsaved restorable changes."""
        return self._manager._workspace_state.is_modified(
            has_nodes=bool(self._manager._tool_graph.nodes)
        )

    def _refresh_manager_record(self, *, coalesce_if_busy: bool = True) -> None:
        self._manager._registry_heartbeat.request_refresh(
            self._manager.workspace_path,
            coalesce_if_busy=coalesce_if_busy,
        )

    def _workspace_window_state(self) -> tuple[str, str, bool]:
        if self._manager._workspace_state.path is None:
            window_file_path = ""
        else:
            window_file_path = typing.cast("str", self._manager.workspace_path)
        workspace_display_name = (
            "Untitled"
            if self._manager._workspace_state.path is None
            else self._manager._workspace_state.path.name
        )
        title = (
            f"{_window_title_with_modified_placeholder(workspace_display_name)}"
            f" - ImageTool Manager #{self._manager.manager_index}"
        )
        return window_file_path, title, self._manager.is_workspace_modified

    def _update_workspace_window_title(self, *, force: bool = True) -> None:
        if force:
            self._apply_workspace_window_title()
            return
        self._manager._queue_idle_work(
            ("workspace-window", "title"), self._apply_workspace_window_title
        )

    def _apply_workspace_window_title(self) -> None:
        window_file_path, title, modified = self._workspace_window_state()
        applied = self._workspace_window_state_applied
        if applied is None or applied[0] != window_file_path:
            self._manager.setWindowFilePath(window_file_path)
        if applied is None or applied[1] != title:
            self._manager.setWindowTitle(title)
        if applied is None or applied[2] != modified:
            self._manager.setWindowModified(modified)
        self._workspace_window_state_applied = (window_file_path, title, modified)

    def _release_workspace_lock(self) -> None:
        if self._manager._workspace_state.lock is None:
            return
        self._manager._workspace_state.lock.unlock()
        self._manager._workspace_state.lock = None

    def _current_workspace_document_path(self) -> pathlib.Path | None:
        path = self._manager._workspace_state.path
        if path is None or not workspace_format._workspace_path_is_itws(path):
            return None
        return path

    def _workspace_document_access(
        self, fname: str | os.PathLike[str]
    ) -> _WorkspaceDocumentAccess:
        workspace_path = pathlib.Path(fname).resolve()
        workspace_lock = None
        if workspace_path != self._manager._workspace_state.path:
            workspace_lock = workspace_storage._acquire_workspace_document_lock(
                workspace_path
            )
        return _WorkspaceDocumentAccess(workspace_path, workspace_lock)

    @contextlib.contextmanager
    def _workspace_document_access_context(
        self, fname: str | os.PathLike[str]
    ) -> Iterator[_WorkspaceDocumentAccess]:
        access = self._workspace_document_access(fname)
        try:
            yield access
        finally:
            access.release()

    def _set_workspace_path(
        self,
        fname: str | os.PathLike[str] | None,
        *,
        workspace_lock: QtCore.QLockFile | None = None,
    ) -> None:
        workspace_path = None if fname is None else pathlib.Path(fname).resolve()
        if workspace_path == self._manager._workspace_state.path:
            if workspace_lock is not None:
                workspace_lock.unlock()
            self._update_workspace_window_title()
            self._refresh_manager_record()
            return

        if workspace_path is not None and workspace_lock is None:
            raise RuntimeError(
                "Changing the workspace path requires a pre-acquired document lock"
            )
        old_workspace_path = self._manager._workspace_state.path
        self._release_workspace_lock()
        self._manager._workspace_state.lock = workspace_lock
        self._manager._workspace_state.path = workspace_path
        self._manager._workspace_state.advance_document_identity()
        self._manager._workspace_state.delta_save_count = 0
        self._manager._workspace_state.reset_repack_estimate()
        if old_workspace_path is not None and workspace_path is not None:
            self._repoint_pending_workspace_payloads(old_workspace_path, workspace_path)
        if self._manager._workspace_state.path is not None:
            self._manager._recent_directory = str(
                self._manager._workspace_state.path.parent
            )
        self._update_workspace_window_title()
        self._refresh_manager_record()

    def _repoint_pending_workspace_payloads(
        self,
        old_workspace_path: str | os.PathLike[str],
        new_workspace_path: str | os.PathLike[str],
    ) -> None:
        old_normalized = workspace_arrays._normalized_file_path(old_workspace_path)
        for node in self._manager._tool_graph.nodes.values():
            pending = node.pending_workspace_payload
            kind = node.pending_workspace_payload_kind
            if pending is None or kind is None:
                continue
            pending_workspace_path, payload_path = pending
            if (
                workspace_arrays._normalized_file_path(pending_workspace_path)
                == old_normalized
            ):
                node.set_pending_workspace_payload(
                    kind,
                    new_workspace_path,
                    payload_path,
                    payload_attrs=node.pending_workspace_payload_attrs,
                )

    def _repoint_saved_pending_workspace_payloads(
        self, workspace_path: str | os.PathLike[str]
    ) -> None:
        for node in self._manager._tool_graph.nodes.values():
            pending = node.pending_workspace_payload
            kind = node.pending_workspace_payload_kind
            if pending is None or kind is None:
                continue
            node.set_pending_workspace_payload(
                kind,
                workspace_path,
                self.saving._workspace_payload_path(node.uid),
                payload_attrs=node.pending_workspace_payload_attrs,
            )

    def _saved_tool_payload_dataset_for_rebind(
        self,
        workspace_path: str | os.PathLike[str],
        node: _ManagedWindowNode,
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset],
    ) -> xr.Dataset:
        payload_path = self.saving._workspace_payload_path(node.uid)
        key = self.loading._workspace_reference_key(workspace_path, payload_path)
        try:
            opened = reference_datasets[key]
        except KeyError:
            opened = workspace_arrays.open_workspace_dataset(
                workspace_path, payload_path, chunks={}
            )
            reference_datasets[key] = opened
        ds = workspace_format._restore_workspace_dataset_attrs(opened.copy(deep=False))
        return _serialization.restore_private_coords(
            ds, erlab.interactive.utils._SAVED_TOOL_DATA_NAME
        )

    def _rebind_workspace_referenced_tool_data(
        self,
        workspace_path: str | os.PathLike[str],
        *,
        exclude_data_uids: Collection[str] = frozenset(),
    ) -> None:
        for node in self._manager._tool_graph.nodes.values():
            tool = node.tool_window
            if (
                node.is_imagetool
                or tool is None
                or not tool.can_save_and_load()
                or not node._workspace_reference_datasets
                or node.uid in exclude_data_uids
                or self._workspace_tool_references_include_uids(
                    node._workspace_tool_data_references,
                    exclude_data_uids,
                    parent_uid=node.parent_uid,
                )
            ):
                continue
            reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] = {}
            try:
                with tool._save_tool_data_reference_context(
                    self._manager._tool_graph.nodes
                ):
                    ds = tool.to_dataset()
                references = type(tool)._saved_tool_data_references(ds)
                if not references:
                    ds = self._saved_tool_payload_dataset_for_rebind(
                        workspace_path, node, reference_datasets
                    )
                    references = type(tool)._saved_tool_data_references(ds)
                source_parent_data, tool_data_reference_resolver = (
                    self.loading._workspace_tool_restore_references(
                        ds,
                        parent_target=node.parent_uid,
                        owner_node=node,
                        reference_datasets=reference_datasets,
                        resolver_error_types=(Exception,),
                        log_resolver_errors=True,
                    )
                )
                data_items = type(tool)._tool_data_items_from_dataset(
                    ds,
                    source_parent_data=source_parent_data,
                    reference_resolver=tool_data_reference_resolver,
                )
                with (
                    self._workspace_load_context(),
                    tool._history_suppressed(),
                ):
                    tool._replace_persistence_data_items(data_items, ds)
                node._set_workspace_tool_data_references(references)
                node._replace_workspace_reference_datasets(reference_datasets)
                reference_datasets = {}
            except Exception as exc:
                self.loading._close_workspace_reference_datasets(reference_datasets)
                raise _WorkspacePostSaveBindingError(
                    "Workspace file was saved, but live ToolWindow data could not "
                    f"be rebound for node {node.uid!r}."
                ) from exc

    @staticmethod
    def _workspace_tool_references_include_uids(
        references: Mapping[str, Mapping[str, typing.Any]],
        uids: Collection[str],
        *,
        parent_uid: str | None,
    ) -> bool:
        if not uids:
            return False
        for reference in references.values():
            kind = reference.get("kind")
            if kind == "parent_source":
                if parent_uid in uids:
                    return True
            elif kind == "manager_node" and reference.get("node_uid") in uids:
                return True
        return False

    def _pending_workspace_payload_snapshot(
        self,
    ) -> dict[
        str,
        tuple[
            typing.Literal["imagetool", "tool"],
            tuple[pathlib.Path, str],
            dict[str, typing.Any] | None,
        ],
    ]:
        snapshot: dict[
            str,
            tuple[
                typing.Literal["imagetool", "tool"],
                tuple[pathlib.Path, str],
                dict[str, typing.Any] | None,
            ],
        ] = {}
        for uid, node in self._manager._tool_graph.nodes.items():
            pending = node.pending_workspace_payload
            kind = node.pending_workspace_payload_kind
            if pending is not None and kind is not None:
                snapshot[uid] = (kind, pending, node.pending_workspace_payload_attrs)
        return snapshot

    def _restore_pending_workspace_payload_snapshot(
        self,
        snapshot: Mapping[
            str,
            tuple[
                typing.Literal["imagetool", "tool"],
                tuple[pathlib.Path, str],
                dict[str, typing.Any] | None,
            ],
        ],
    ) -> None:
        for uid, (kind, pending, attrs) in snapshot.items():
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None:
                continue
            node.set_pending_workspace_payload(
                kind,
                pending[0],
                pending[1],
                payload_attrs=attrs,
            )

    def _live_imagetool_rebind_snapshot(
        self,
        *,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None,
        old_workspace_path: str | os.PathLike[str] | None,
    ) -> dict[str, tuple[_ManagedWindowNode, xr.DataArray, typing.Any, str]]:
        snapshot: dict[
            str, tuple[_ManagedWindowNode, xr.DataArray, typing.Any, str]
        ] = {}
        old_path = workspace_arrays._normalized_file_path(old_workspace_path)
        for uid, node in self._manager._tool_graph.nodes.items():
            if (
                not node.is_imagetool
                or node.imagetool is None
                or node.pending_workspace_memory_payload is not None
            ):
                continue
            if backing_snapshot is not None:
                backing = backing_snapshot.get(uid)
                if backing is None:
                    continue
                kind, source_paths = backing
                if kind == "memory":
                    continue
                if kind == "file_lazy" and (
                    old_path is None or old_path not in source_paths
                ):
                    continue
            snapshot[uid] = (
                node,
                node.slicer_area._data,
                copy.deepcopy(node.slicer_area.state),
                node.name,
            )
        return snapshot

    def _restore_live_imagetool_rebind_snapshot(
        self,
        snapshot: Mapping[
            str, tuple[_ManagedWindowNode, xr.DataArray, typing.Any, str]
        ],
    ) -> None:
        if not snapshot:
            return
        with self._workspace_load_context():
            for uid, (node, data, state, name) in snapshot.items():
                if uid not in self._manager._tool_graph.nodes or node.imagetool is None:
                    continue
                node.slicer_area.set_data(data, auto_compute=False)
                node.slicer_area.state = state
                node._set_name(name, manual=False)

    def _refresh_workspace_payload_bindings_after_full_save(
        self,
        workspace_path: str | os.PathLike[str],
        *,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
        old_workspace_path: str | os.PathLike[str] | None = None,
        skip_live_data_rebind_uids: Collection[str] = frozenset(),
    ) -> None:
        pending_snapshot = self._pending_workspace_payload_snapshot()
        live_imagetool_snapshot = self._live_imagetool_rebind_snapshot(
            backing_snapshot=backing_snapshot,
            old_workspace_path=old_workspace_path,
        )
        try:
            self._repoint_saved_pending_workspace_payloads(workspace_path)
            self.loading._rebind_workspace_backed_imagetools(
                workspace_path,
                backing_snapshot=backing_snapshot,
                old_workspace_path=old_workspace_path,
                exclude_uids=skip_live_data_rebind_uids,
            )
            self._rebind_workspace_referenced_tool_data(
                workspace_path, exclude_data_uids=skip_live_data_rebind_uids
            )
        except _WorkspacePostSaveBindingError:
            with contextlib.suppress(Exception):
                self._restore_pending_workspace_payload_snapshot(pending_snapshot)
            with contextlib.suppress(Exception):
                self._restore_live_imagetool_rebind_snapshot(live_imagetool_snapshot)
            raise
        except Exception as exc:
            with contextlib.suppress(Exception):
                self._restore_pending_workspace_payload_snapshot(pending_snapshot)
            with contextlib.suppress(Exception):
                self._restore_live_imagetool_rebind_snapshot(live_imagetool_snapshot)
            raise _WorkspacePostSaveBindingError(
                "Workspace file was saved, but live workspace data could not be "
                "rebound to the saved file."
            ) from exc

    def _active_managed_window(self) -> QtWidgets.QWidget | None:
        active_window = QtWidgets.QApplication.activeWindow()
        if not isinstance(active_window, QtWidgets.QWidget):
            return None
        if self._manager._node_uid_from_window(active_window) is None:
            return None
        if not erlab.interactive.utils.qt_is_valid(active_window):
            return None
        return active_window

    def _restore_focus_after_workspace_save(
        self, origin: QtWidgets.QWidget | None
    ) -> None:
        if (
            origin is None
            or not erlab.interactive.utils.qt_is_valid(origin)
            or not origin.isVisible()
        ):
            return
        active_window = QtWidgets.QApplication.activeWindow()
        if isinstance(active_window, QtWidgets.QWidget) and active_window not in (
            self._manager,
            origin,
        ):
            return
        origin.activateWindow()
        origin.raise_()
        focus_widget = origin.focusWidget()
        if isinstance(
            focus_widget, QtWidgets.QWidget
        ) and erlab.interactive.utils.qt_is_valid(focus_widget):
            focus_widget.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _dirty_details_text(self) -> str:
        def _node_names(uids: set[str]) -> tuple[str, ...]:
            return tuple(
                self._manager._tool_graph.nodes[uid].display_text
                for uid in sorted(uids)
                if uid in self._manager._tool_graph.nodes
            )

        sections = (
            ("Added", _node_names(self._manager._workspace_state.dirty_added)),
            (
                "Removed",
                tuple(dict.fromkeys(self._manager._workspace_state.dirty_removed)),
            ),
            (
                "Data modified",
                _node_names(
                    self._manager._workspace_state.dirty_data
                    - self._manager._workspace_state.dirty_added
                ),
            ),
            (
                "State modified",
                _node_names(
                    self._manager._workspace_state.dirty_state
                    - self._manager._workspace_state.dirty_data
                    - self._manager._workspace_state.dirty_added
                ),
            ),
            (
                "Structure modified",
                tuple(dict.fromkeys(self._manager._workspace_state.structure_reasons)),
            ),
            (
                "Layout modified",
                ("Manager window layout",)
                if self._manager._workspace_state.layout_modified
                else (),
            ),
            (
                "Settings modified",
                ("Workspace settings",)
                if self._manager._workspace_state.options_modified
                else (),
            ),
            (
                "Acquisition context modified",
                ("Acquisition context",)
                if self._manager._workspace_state.context_modified
                else (),
            ),
        )
        blocks: list[str] = []
        for label, items in sections:
            if items:
                blocks.append(f"{label}:\n" + "\n".join(f"- {item}" for item in items))
        return "\n\n".join(blocks)

    def _set_node_window_modified(self, uid: str, modified: bool) -> None:
        self._pending_node_window_modified.pop(uid, None)
        self._apply_node_window_modified(uid, modified)

    def _queue_node_window_modified(self, uid: str, modified: bool) -> None:
        self._pending_node_window_modified[uid] = modified
        self._manager._queue_idle_work(
            ("node-window", uid),
            functools.partial(self._flush_pending_node_window_modified, uid),
        )

    def _flush_pending_node_window_modified(self, uid: str) -> None:
        try:
            modified = self._pending_node_window_modified.pop(uid)
        except KeyError:
            return
        self._apply_node_window_modified(uid, modified)

    def _apply_node_window_modified(self, uid: str, modified: bool) -> None:
        node = self._manager._tool_graph.nodes.get(uid)
        if node is None:
            self._node_window_state_applied.pop(uid, None)
            return
        window = node.window
        if node.tool_window is not None:
            display_name = node.tool_window._tool_display_name
            base_title = (
                f"{node.tool_window.tool_name}: {display_name}"
                if display_name
                else node.tool_window.tool_name
            )
        else:
            base_title = node.label_text
        windows: list[tuple[QtWidgets.QWidget | None, str]] = [(window, base_title)]
        if node.tool_window is not None:
            windows.extend(node.tool_window._managed_secondary_windows())
        valid_windows: list[tuple[QtWidgets.QWidget, str]] = []
        for target_window, target_title in windows:
            if target_window is None or not erlab.interactive.utils.qt_is_valid(
                target_window
            ):
                continue
            valid_windows.append((target_window, target_title))
        target_state = (
            tuple(
                (id(target_window), target_title)
                for target_window, target_title in valid_windows
            ),
            modified,
        )
        if self._node_window_state_applied.get(uid) == target_state and all(
            target_window.windowTitle()
            == _window_title_with_modified_placeholder(target_title)
            for target_window, target_title in valid_windows
        ):
            return
        for target_window, target_title in valid_windows:
            title = _window_title_with_modified_placeholder(target_title)
            if title != target_window.windowTitle():
                target_window.setWindowTitle(title)
            target_window.setWindowModified(modified)
        self._node_window_state_applied[uid] = target_state

    def _apply_workspace_dirty_event(
        self, event: workspace_state._WorkspaceDirtyEvent
    ) -> bool:
        if event.uid is not None and (event.added or event.data or event.state):
            self._set_node_window_modified(event.uid, True)
        return self._manager._workspace_state.apply_dirty_event(event)

    def _mark_workspace_dirty(
        self,
        *,
        uid: str | None = None,
        data: bool = False,
        state: bool = False,
        added: bool = False,
        removed: str | None = None,
        structure: str | None = None,
    ) -> bool:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
        ):
            return False
        event = workspace_state._WorkspaceDirtyEvent(
            generation=self._manager._workspace_state.dirty_generation + 1,
            uid=uid,
            data=data,
            state=state,
            added=added,
            removed=removed,
            structure=structure,
        )
        was_modified = self._manager.is_workspace_modified
        node_was_modified = event.uid is not None and event.uid in (
            self._manager._workspace_state.dirty_added
            | self._manager._workspace_state.dirty_data
            | self._manager._workspace_state.dirty_state
        )
        if (
            event.uid is not None
            and (event.added or event.data or event.state)
            and not node_was_modified
        ):
            self._queue_node_window_modified(event.uid, True)
        dirty_changed = self._manager._workspace_state.mark_dirty(event)
        if not was_modified and self._manager.is_workspace_modified:
            self._update_workspace_window_title(force=False)
        return dirty_changed

    def _mark_node_added(self, uid: str) -> bool:
        return self._mark_workspace_dirty(uid=uid, added=True, structure="Added window")

    def _mark_node_data_dirty(self, uid: str) -> bool:
        return self._mark_workspace_dirty(uid=uid, data=True)

    def _mark_node_state_dirty(self, uid: str) -> bool:
        return self._mark_workspace_dirty(uid=uid, state=True)

    def _mark_tool_info_dirty(self, uid: str) -> bool:
        if uid not in self._manager._workspace_state.dirty_state:
            return self._mark_node_state_dirty(uid)
        return False

    def _mark_workspace_structure_dirty(self, reason: str) -> bool:
        return self._mark_workspace_dirty(structure=reason)

    def _mark_workspace_layout_dirty(self) -> None:
        if (
            not getattr(self._manager, "_manager_layout_tracking_enabled", False)
            or self._manager._workspace_state.path is None
            or self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return
        if self._manager._workspace_state.mark_layout_dirty():
            self._update_workspace_window_title(force=False)

    def _mark_workspace_options_dirty(self) -> None:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return
        if self._manager._workspace_state.mark_options_dirty():
            self._update_workspace_window_title(force=False)

    def _mark_workspace_context_dirty(self) -> None:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return
        if self._manager._workspace_state.mark_context_dirty():
            self._update_workspace_window_title(force=False)

    def _mark_workspace_clean(self) -> None:
        self._manager._workspace_state.mark_clean()
        for uid in tuple(self._manager._tool_graph.nodes):
            self._set_node_window_modified(uid, False)
        self._update_workspace_window_title()

    def _restore_workspace_dirty_events(
        self, events: Iterable[workspace_state._WorkspaceDirtyEvent]
    ) -> None:
        retained_events = list(events)
        self._manager._workspace_state.mark_clean()
        for uid in tuple(self._manager._tool_graph.nodes):
            self._set_node_window_modified(uid, False)
        for event in retained_events:
            self._apply_workspace_dirty_event(event)
        self._manager._workspace_state.dirty_events = retained_events
        self._update_workspace_window_title()

    @contextlib.contextmanager
    def _workspace_load_context(self) -> Iterator[None]:
        with (
            self._manager._workspace_state.load_context(),
            self._manager._workspace_ui_refresh_context(),
        ):
            yield

    def _send_workspace_posted_events(self, event_type: QtCore.QEvent.Type) -> None:
        for _ in range(3):
            QtWidgets.QApplication.sendPostedEvents(None, int(event_type.value))

    def _drain_workspace_restore_events(self) -> None:
        self._send_workspace_posted_events(QtCore.QEvent.Type.MetaCall)
        self._send_workspace_posted_events(QtCore.QEvent.Type.DeferredDelete)

    def _drain_workspace_deferred_events(self) -> None:
        self._send_workspace_posted_events(QtCore.QEvent.Type.MetaCall)
        for _ in range(3):
            QtWidgets.QApplication.processEvents()
        self._manager._flush_idle_work(force=True)
        self._send_workspace_posted_events(QtCore.QEvent.Type.MetaCall)
        for _ in range(3):
            QtWidgets.QApplication.processEvents()

    def _workspace_state_snapshot(self) -> _WorkspaceStateSnapshot:
        return self._manager._workspace_state.snapshot(
            node_uid_counter=self._manager._tool_graph.uid_counter
        )

    def _install_workspace_save_shortcut(self, widget: QtWidgets.QWidget) -> None:
        for shortcut in widget.findChildren(QtWidgets.QShortcut):
            if shortcut.objectName() == _WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME:
                return
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.StandardKey.Save, widget)
        shortcut.setObjectName(_WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._manager.save)

    @staticmethod
    def _validated_standalone_app_state(
        key: str, state: Mapping[str, typing.Any]
    ) -> dict[str, typing.Any] | None:
        model_type: typing.Any
        if key == "explorer":
            from erlab.interactive.explorer._tabbed_explorer import DataExplorerState

            model_type = DataExplorerState
        elif key == "ptable":
            from erlab.interactive.ptable._window import PeriodicTableState

            model_type = PeriodicTableState
        else:
            return None
        try:
            return typing.cast(
                "dict[str, typing.Any]",
                model_type.model_validate(state).model_dump(
                    mode="json", exclude_none=True
                ),
            )
        except Exception:
            logger.warning(
                "Ignoring invalid %s standalone app state", key, exc_info=True
            )
            return None

    def _workspace_save_dialog(
        self,
        *,
        native: bool = True,
        caption: str = "Save Workspace",
        selected_file: str | os.PathLike[str] | None = None,
    ) -> str | None:
        dialog = QtWidgets.QFileDialog(self._manager, caption)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("ImageTool Workspace Files (*.itws)")
        dialog.setDefaultSuffix("itws")
        if selected_file is not None:
            dialog.selectFile(str(selected_file))
        elif self._manager._workspace_state.path is not None:
            dialog.selectFile(str(self._manager._workspace_state.path))
        elif (directory := self._manager._recent_or_default_directory()) is not None:
            dialog.setDirectory(directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return None
        return dialog.selectedFiles()[0]

    def _dirty_workspace_save_choice(self, action_text: str) -> str:
        if not self._manager.is_workspace_modified:
            return "clean"

        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Save changes to this workspace?")
        msg_box.setInformativeText(action_text)
        details = self._dirty_details_text()
        if details:
            msg_box.setDetailedText(details)
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Save)
        result = msg_box.exec()
        if result == QtWidgets.QMessageBox.StandardButton.Save:
            return "save"
        if result == QtWidgets.QMessageBox.StandardButton.Discard:
            return "discard"
        return "cancel"

    def _run_after_dirty_workspace_saved_or_discarded(
        self,
        action_text: str,
        continuation: Callable[[], bool | None],
        *,
        native: bool = True,
    ) -> bool:
        choice = self._dirty_workspace_save_choice(action_text)
        if choice == "cancel":
            return False
        if choice in {"clean", "discard"}:
            return bool(continuation())

        def _continue_after_save(save_succeeded: bool) -> None:
            if save_succeeded and not self._manager.is_workspace_modified:
                continuation()

        return self.save(native=native, on_finished=_continue_after_save)

    def _show_legacy_workspace_upgrade_message(
        self, fname: str | os.PathLike[str]
    ) -> None:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Save Legacy Workspace")
        msg_box.setText("This workspace uses a legacy file format.")
        msg_box.setInformativeText(
            "Save it as an .itws file so ImageTool Manager can update it safely."
        )
        msg_box.setDetailedText(str(pathlib.Path(fname)))
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def _save_legacy_workspace_as_v4(
        self,
        fname: str | os.PathLike[str],
        *,
        native: bool = True,
        existing_access: _WorkspaceDocumentAccess | None = None,
    ) -> tuple[str, QtCore.QLockFile | None] | None:
        self._show_legacy_workspace_upgrade_message(fname)
        converted_fname = self._workspace_save_dialog(
            native=native,
            caption="Save Converted Workspace",
            selected_file=fname,
        )
        if converted_fname is None:
            return None
        converted_path = pathlib.Path(converted_fname).resolve()
        if not workspace_format._workspace_path_is_itws(converted_path):
            _show_itws_workspace_warning(self._manager)
            return None
        if existing_access is not None and converted_path == existing_access.path:
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Saving workspace..."
            ):
                self.saving._save_workspace_document(
                    existing_access.path,
                    force_full=True,
                    document_access=existing_access,
                )
            return str(existing_access.path), existing_access.take_lock()

        with self._workspace_document_access_context(converted_fname) as access:
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Saving workspace..."
            ):
                self.saving._save_workspace_document(
                    access.path,
                    force_full=True,
                    document_access=access,
                )
            return str(access.path), access.take_lock()

    def _associate_loaded_workspace_file(
        self,
        fname: str | os.PathLike[str],
        schema_version: int,
        *,
        native: bool = True,
        delta_save_count: int = 0,
        estimated_obsolete_bytes: int = 0,
        replacement_delta_count: int = 0,
        repack_estimate_known: bool = True,
        workspace_access: _WorkspaceDocumentAccess | None = None,
        rebind_data: bool = True,
    ) -> None:
        associated_fname = fname
        associated_lock: QtCore.QLockFile | None = None
        if workspace_format._workspace_schema_requires_conversion(schema_version):
            converted = self._save_legacy_workspace_as_v4(
                fname, native=native, existing_access=workspace_access
            )
            if converted is None:
                self._set_workspace_path(None)
                self._manager._workspace_state.needs_full_save = True
                self._mark_workspace_structure_dirty(
                    "Legacy workspace needs conversion"
                )
                return
            associated_fname, associated_lock = converted
            delta_save_count = 0
            estimated_obsolete_bytes = 0
            replacement_delta_count = 0
            repack_estimate_known = True
            schema_version = workspace_format._current_workspace_schema_version()
        elif workspace_access is not None:
            associated_lock = workspace_access.take_lock()

        self._set_workspace_path(associated_fname, workspace_lock=associated_lock)
        self._manager._workspace_state.delta_save_count = delta_save_count
        self._manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            known=repack_estimate_known,
        )
        self._manager._workspace_state.schema_version = schema_version
        self._manager._workspace_state.needs_full_save = (
            workspace_format._workspace_schema_requires_full_save(schema_version)
        )
        if rebind_data:
            self.loading._rebind_workspace_backed_imagetools(associated_fname)
        self._drain_workspace_restore_events()
        self._mark_workspace_clean()
        self._record_recent_workspace(associated_fname)

    def offload_to_workspace(
        self, targets: Iterable[int | str], *, native: bool = True
    ) -> bool:
        """Replace selected in-memory ImageTools with dask-backed workspace data.

        .. versionadded:: 3.23.0
        """
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; offload after it finishes", 3000
            )
            return False
        offload_targets: list[int | str] = []
        for target in targets:
            node = self._manager._node_for_target(target)
            if (
                node.is_imagetool
                and node.imagetool is not None
                and not node.slicer_area.data_chunked
                and node.pending_workspace_memory_payload is None
            ):
                offload_targets.append(target)
        if not offload_targets:
            return False

        def _offload_after_save(save_succeeded: bool) -> None:
            if not save_succeeded or self._manager.is_workspace_modified:
                return
            if self._manager._workspace_state.path is None:
                return
            self._offload_targets_to_current_workspace(offload_targets)

        state = self._manager._workspace_state
        if state.path is None:
            return self.save_as(native=native, on_finished=_offload_after_save)
        if self._manager.is_workspace_modified or state.needs_full_save:
            return self.save(native=native, on_finished=_offload_after_save)
        return self._offload_targets_to_current_workspace(offload_targets)

    def _offload_targets_to_current_workspace(
        self, offload_targets: Iterable[int | str]
    ) -> bool:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return False

        origin = self._active_managed_window()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self._manager, "Offloading to workspace..."
            ):
                self.loading._rebind_workspace_backed_imagetools(
                    workspace_path,
                    targets=offload_targets,
                    chunks={},
                )
                workspace_storage._write_workspace_root_attrs_to_file(
                    workspace_path,
                    self.saving._workspace_root_attrs_payload(
                        delta_save_count=self._manager._workspace_state.delta_save_count
                    ),
                )
            self._manager._status_bar.showMessage("Data offloaded to workspace", 5000)
        except Exception:
            self._manager._show_operation_error(
                "Error while offloading to workspace",
                "An error occurred while reconnecting data from the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            return False

        self._restore_focus_after_workspace_save(origin)
        self._manager._update_actions()
        self._manager._update_info()
        return True

    def _set_workspace_save_actions_enabled(self, enabled: bool) -> tuple[bool, ...]:
        open_recent_action = self._manager.open_recent_menu.menuAction()
        previous = (
            self._manager.save_action.isEnabled(),
            self._manager.save_as_action.isEnabled(),
            self._manager.compact_workspace_action.isEnabled(),
            self._manager.load_action.isEnabled(),
            False if open_recent_action is None else open_recent_action.isEnabled(),
            self._manager.offload_action.isEnabled(),
            self._manager.import_workspace_action.isEnabled(),
        )
        self._manager.save_action.setEnabled(enabled and previous[0])
        self._manager.save_as_action.setEnabled(enabled and previous[1])
        self._manager.compact_workspace_action.setEnabled(enabled and previous[2])
        self._manager.load_action.setEnabled(enabled and previous[3])
        if open_recent_action is not None:
            open_recent_action.setEnabled(enabled and previous[4])
        self._manager.offload_action.setEnabled(enabled and previous[5])
        self._manager.import_workspace_action.setEnabled(enabled and previous[6])
        return previous

    def _restore_workspace_save_actions_enabled(
        self, previous: tuple[bool, ...]
    ) -> None:
        open_recent_action = self._manager.open_recent_menu.menuAction()
        self._manager.save_action.setEnabled(previous[0])
        self._manager.save_as_action.setEnabled(previous[1])
        self._manager.compact_workspace_action.setEnabled(previous[2])
        self._manager.load_action.setEnabled(previous[3])
        if open_recent_action is not None:
            open_recent_action.setEnabled(previous[4])
        self._manager.offload_action.setEnabled(previous[5])
        self._manager.import_workspace_action.setEnabled(previous[6])

    def _start_workspace_save_worker(
        self,
        fname: str | os.PathLike[str],
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
        *,
        on_finished: Callable[[bool, float, str], None],
        on_start_error: Callable[[], None] | None = None,
    ) -> bool:
        thread_pool = QtCore.QThreadPool.globalInstance()
        if thread_pool is None:
            snapshot.close()
            if on_start_error is not None:
                on_start_error()
            return False

        worker = workspace_saving._WorkspaceSaveWorker(fname, snapshot)
        previous_action_states = self._set_workspace_save_actions_enabled(False)

        def _finish(ok: bool, elapsed: float, error_text: str) -> None:
            self._manager._workspace_state.save_in_progress = False
            self._restore_workspace_save_actions_enabled(previous_action_states)
            self._manager._update_actions()
            receiver = self._background_save_receiver
            self._background_save_receiver = None
            self._background_save_worker = None
            if receiver is not None:
                receiver.deleteLater()
            try:
                on_finished(ok, elapsed, error_text)
            except Exception:
                logger.exception(
                    "Error while finishing workspace save",
                    extra={"suppress_ui_alert": True},
                )
                self._manager._status_bar.clearMessage()
                self._manager._show_operation_error(
                    "Error while saving workspace",
                    "An error occurred while saving the workspace file.",
                )

        receiver = workspace_saving._WorkspaceSaveResultReceiver(
            callback=_finish,
            parent=self._manager,
        )
        worker.signals.finished.connect(receiver.finish)
        self._manager._workspace_state.save_in_progress = True
        self._background_save_worker = worker
        self._background_save_receiver = receiver
        try:
            thread_pool.start(worker)
        except Exception:
            self._manager._workspace_state.save_in_progress = False
            self._restore_workspace_save_actions_enabled(previous_action_states)
            self._background_save_worker = None
            self._background_save_receiver = None
            receiver.deleteLater()
            snapshot.close()
            if on_start_error is not None:
                on_start_error()
            return False
        return True

    def _show_workspace_post_save_binding_error(
        self, workspace_path: str | os.PathLike[str]
    ) -> None:
        self._manager._status_bar.clearMessage()
        self._manager._show_operation_error(
            "Workspace file saved but live references were not updated",
            "The workspace file was saved, but live tool data could not be "
            "updated to use the saved file. Reopen the workspace to continue "
            "from the saved version.",
        )

    def _mark_workspace_post_save_binding_refresh_failed(self) -> None:
        self._manager._workspace_state.needs_full_save = True
        self._mark_workspace_structure_dirty(
            "Live workspace data references need refresh"
        )

    def _finish_workspace_save_result(
        self,
        *,
        document_id: str,
        workspace_path: pathlib.Path,
        old_workspace_path: pathlib.Path | None,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]],
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
        ok: bool,
        worker_elapsed: float,
        error_text: str,
        origin: QtWidgets.QWidget | None,
        snapshot_elapsed: float,
        started_at: float,
        restore_focus: bool,
    ) -> bool:
        total_elapsed = time.perf_counter() - started_at
        logger.debug(
            "Workspace save timing: snapshot %.3f s, write %.3f s, total %.3f s",
            snapshot_elapsed,
            worker_elapsed,
            total_elapsed,
        )
        if self._manager._workspace_state.document_id != document_id:
            logger.info(
                "Ignoring completed workspace save for inactive document: %s",
                workspace_path,
                extra={"suppress_ui_alert": True},
            )
            return False
        if not ok:
            self._manager._status_bar.clearMessage()
            self._manager._show_workspace_save_worker_error(error_text)
            if restore_focus:
                self._restore_focus_after_workspace_save(origin)
            return False

        self._drain_workspace_deferred_events()
        post_save_events = tuple(
            event
            for event in self._manager._workspace_state.dirty_events
            if event.generation > snapshot.generation
        )
        has_new_dirty_generation = (
            self._manager._workspace_state.dirty_generation > snapshot.generation
            and self._manager.is_workspace_modified
        )
        post_save_data_uids = frozenset(
            event.uid
            for event in post_save_events
            if event.uid is not None and (event.data or event.added)
        )
        if snapshot.full_tree is not None:
            try:
                self._refresh_workspace_payload_bindings_after_full_save(
                    workspace_path,
                    backing_snapshot=backing_snapshot,
                    old_workspace_path=old_workspace_path,
                    skip_live_data_rebind_uids=post_save_data_uids,
                )
            except _WorkspacePostSaveBindingError:
                self._mark_workspace_post_save_binding_refresh_failed()
                self._show_workspace_post_save_binding_error(workspace_path)
                if restore_focus:
                    self._restore_focus_after_workspace_save(origin)
                return False
            self._manager._workspace_state.schema_version = (
                workspace_format._current_workspace_schema_version()
            )
        self._manager._workspace_state.needs_full_save = False
        self._manager._workspace_state.delta_save_count = snapshot.delta_save_count
        self._manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=snapshot.estimated_obsolete_bytes,
            replacement_delta_count=snapshot.replacement_delta_count,
            known=snapshot.repack_estimate_known,
        )
        if post_save_events:
            self._restore_workspace_dirty_events(post_save_events)
            message = "Workspace saved; new changes remain unsaved"
        elif has_new_dirty_generation:
            message = "Workspace saved; new changes remain unsaved"
        else:
            self._mark_workspace_clean()
            message = (
                f"Workspace saved in {total_elapsed:.1f} s"
                if total_elapsed >= _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS
                else "Workspace saved"
            )
        self._manager._status_bar.showMessage(message, 5000)
        if restore_focus:
            self._restore_focus_after_workspace_save(origin)
        self._record_recent_workspace(workspace_path)
        return True

    def _finish_background_workspace_save(
        self,
        *,
        document_id: str,
        workspace_path: pathlib.Path,
        old_workspace_path: pathlib.Path | None,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]],
        snapshot: workspace_saving._WorkspaceSaveSnapshot,
        ok: bool,
        worker_elapsed: float,
        error_text: str,
        origin: QtWidgets.QWidget | None,
        snapshot_elapsed: float,
        started_at: float,
        restore_focus: bool,
        on_finished: Callable[[bool], None] | None = None,
    ) -> None:
        try:
            save_succeeded = self._finish_workspace_save_result(
                document_id=document_id,
                workspace_path=workspace_path,
                old_workspace_path=old_workspace_path,
                backing_snapshot=backing_snapshot,
                snapshot=snapshot,
                ok=ok,
                worker_elapsed=worker_elapsed,
                error_text=error_text,
                origin=origin,
                snapshot_elapsed=snapshot_elapsed,
                started_at=started_at,
                restore_focus=restore_focus,
            )
            queued = self._background_save_requested
            if (
                save_succeeded
                and queued
                and self._manager.is_workspace_modified
                and self._current_workspace_document_path() == workspace_path
            ):
                QtCore.QTimer.singleShot(0, self.save)
            if on_finished is not None:
                on_finished(save_succeeded)
        except Exception:
            logger.exception(
                "Error while finishing background workspace save",
                extra={"suppress_ui_alert": True},
            )
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            if on_finished is not None:
                on_finished(False)
        finally:
            self._background_save_requested = False

    def save(
        self,
        *,
        native: bool = True,
        on_finished: Callable[[bool], None] | None = None,
        restore_focus: bool = True,
    ) -> bool:
        """Start a non-blocking save for the current workspace document."""
        workspace_path = self._current_workspace_document_path()
        if workspace_path is None:
            return self.save_as(native=native, on_finished=on_finished)
        if self._manager._workspace_state.save_in_progress:
            self._background_save_requested = True
            self._manager._status_bar.showMessage("Workspace save queued", 3000)
            return False

        origin = self._active_managed_window()
        document_id = self._manager._workspace_state.document_id
        old_workspace_path = workspace_path
        backing_snapshot = self.loading._workspace_data_backing_snapshot()
        self._manager._status_bar.showMessage("Saving workspace...")
        started_at = time.perf_counter()
        snapshot: workspace_saving._WorkspaceSaveSnapshot | None = None
        try:
            snapshot_started_at = time.perf_counter()
            snapshot = self.saving._workspace_save_snapshot(workspace_path)
            snapshot_elapsed = time.perf_counter() - snapshot_started_at
        except Exception:
            if snapshot is not None:
                snapshot.close()
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            if restore_focus:
                self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(False)
            return False
        if snapshot is None:  # pragma: no cover
            raise RuntimeError("Workspace save snapshot was not created")

        def _start_error() -> None:
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            if restore_focus:
                self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(False)

        self._background_save_requested = False
        return self._start_workspace_save_worker(
            workspace_path,
            snapshot,
            on_finished=lambda ok, elapsed, error_text: (
                self._finish_background_workspace_save(
                    document_id=document_id,
                    workspace_path=workspace_path,
                    old_workspace_path=old_workspace_path,
                    backing_snapshot=backing_snapshot,
                    snapshot=snapshot,
                    ok=ok,
                    worker_elapsed=elapsed,
                    error_text=error_text,
                    origin=origin,
                    snapshot_elapsed=snapshot_elapsed,
                    started_at=started_at,
                    restore_focus=restore_focus,
                    on_finished=on_finished,
                )
            ),
            on_start_error=_start_error,
        )

    def save_as(
        self,
        *,
        native: bool = True,
        on_finished: Callable[[bool], None] | None = None,
    ) -> bool:
        """Save the current workspace under a new path and bind to that path."""
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save already in progress", 3000
            )
            if on_finished is not None:
                on_finished(False)
            return False
        origin = self._active_managed_window()
        fname = self._workspace_save_dialog(native=native, caption="Save Workspace As")
        if fname is None:
            if on_finished is not None:
                on_finished(False)
            return False
        if not workspace_format._workspace_path_is_itws(fname):
            _show_itws_workspace_warning(self._manager)
            if on_finished is not None:
                on_finished(False)
            return False
        old_workspace_path = self._manager._workspace_state.path
        document_id = self._manager._workspace_state.document_id
        backing_snapshot = self.loading._workspace_data_backing_snapshot()
        access: _WorkspaceDocumentAccess | None = None
        snapshot: workspace_saving._WorkspaceSaveSnapshot | None = None
        self._manager._status_bar.showMessage("Saving workspace...")
        started_at = time.perf_counter()
        try:
            access = self._workspace_document_access(fname)
            self._drain_workspace_deferred_events()
            generation = self._manager._workspace_state.dirty_generation
            self._manager._workspace_state.saving_depth += 1
            try:
                snapshot_started_at = time.perf_counter()
                snapshot = self.saving._workspace_full_save_snapshot(
                    generation, fname=access.path
                )
                snapshot_elapsed = time.perf_counter() - snapshot_started_at
            finally:
                self._manager._workspace_state.saving_depth -= 1
        except Exception:
            if snapshot is not None:
                snapshot.close()
            if access is not None:
                access.release()
            logger.exception(
                "Error while preparing workspace Save As snapshot",
                extra={"suppress_ui_alert": True},
            )
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(False)
            return False
        if snapshot is None:  # pragma: no cover
            if access is not None:
                access.release()
            raise RuntimeError("Workspace save snapshot was not created")

        def _finish_save_as(ok: bool, worker_elapsed: float, error_text: str) -> None:
            nonlocal access
            total_elapsed = time.perf_counter() - started_at
            logger.debug(
                "Workspace save timing: snapshot %.3f s, write %.3f s, total %.3f s",
                snapshot_elapsed,
                worker_elapsed,
                total_elapsed,
            )
            if access is None:  # pragma: no cover
                if on_finished is not None:
                    on_finished(False)
                return
            if self._manager._workspace_state.document_id != document_id:
                logger.info(
                    "Ignoring completed Save As for inactive document: %s",
                    access.path,
                    extra={"suppress_ui_alert": True},
                )
                access.release()
                access = None
                if on_finished is not None:
                    on_finished(False)
                return
            if not ok:
                self._manager._status_bar.clearMessage()
                self._manager._show_workspace_save_worker_error(error_text)
                access.release()
                self._restore_focus_after_workspace_save(origin)
                if on_finished is not None:
                    on_finished(False)
                return

            self._drain_workspace_deferred_events()
            post_save_events = tuple(
                event
                for event in self._manager._workspace_state.dirty_events
                if event.generation > snapshot.generation
            )
            has_new_dirty_generation = (
                self._manager._workspace_state.dirty_generation > snapshot.generation
                and self._manager.is_workspace_modified
            )
            if post_save_events or has_new_dirty_generation:
                access.release()
                self._manager._status_bar.showMessage(
                    "Workspace saved; new changes remain unsaved", 5000
                )
                self._restore_focus_after_workspace_save(origin)
                if on_finished is not None:
                    on_finished(False)
                return

            if snapshot.full_tree is not None:
                try:
                    self._refresh_workspace_payload_bindings_after_full_save(
                        access.path,
                        backing_snapshot=backing_snapshot,
                        old_workspace_path=old_workspace_path,
                    )
                except _WorkspacePostSaveBindingError:
                    self._show_workspace_post_save_binding_error(access.path)
                    access.release()
                    self._restore_focus_after_workspace_save(origin)
                    if on_finished is not None:
                        on_finished(False)
                    return

            self._manager._workspace_state.needs_full_save = False
            self._manager._workspace_state.delta_save_count = snapshot.delta_save_count
            self._manager._workspace_state.set_repack_estimate(
                estimated_obsolete_bytes=snapshot.estimated_obsolete_bytes,
                replacement_delta_count=snapshot.replacement_delta_count,
                known=snapshot.repack_estimate_known,
            )
            self._manager._workspace_state.schema_version = (
                workspace_format._current_workspace_schema_version()
            )
            saved_path = access.path
            self._set_workspace_path(saved_path, workspace_lock=access.take_lock())
            access = None
            self._drain_workspace_deferred_events()
            self._mark_workspace_clean()
            self._record_recent_workspace(saved_path)
            message = (
                f"Workspace saved in {total_elapsed:.1f} s"
                if total_elapsed >= _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS
                else "Workspace saved"
            )
            self._manager._status_bar.showMessage(message, 5000)
            self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(True)

        def _start_error() -> None:
            if access is not None:
                access.release()
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            if on_finished is not None:
                on_finished(False)

        return self._start_workspace_save_worker(
            access.path,
            snapshot,
            on_finished=_finish_save_as,
            on_start_error=_start_error,
        )

    def _compact_workspace_before_shutdown(
        self, on_finished: Callable[[], None] | None = None
    ) -> bool:
        if (
            self._manager._workspace_state.path is None
            or self._manager._workspace_state.delta_save_count <= 0
            or self._manager.is_workspace_modified
            or self._manager._workspace_state.save_in_progress
            or self._manager._workspace_state.loading_depth > 0
            or self._shutdown_compaction_attempted
        ):
            return False
        if not self.saving._workspace_should_repack_before_shutdown():
            return False
        try:
            logger.debug("Compacting workspace before shutdown...")
            self._shutdown_compaction_attempted = True
            document_id = self._manager._workspace_state.document_id
            self._drain_workspace_deferred_events()
            generation = self._manager._workspace_state.dirty_generation
            self._manager._workspace_state.saving_depth += 1
            try:
                snapshot = self.saving._workspace_file_repack_snapshot(generation)
                if snapshot is None:
                    return False
            finally:
                self._manager._workspace_state.saving_depth -= 1
        except Exception:
            logger.exception(
                "Failed to compact workspace before shutdown",
                extra={"suppress_ui_alert": True},
            )
            return False

        def _finish_compaction(ok: bool, _elapsed: float, error_text: str) -> None:
            if self._manager._workspace_state.document_id != document_id:
                logger.info(
                    "Ignoring completed shutdown compaction for inactive document",
                    extra={"suppress_ui_alert": True},
                )
                if on_finished is not None:
                    on_finished()
                return
            if not ok:
                logger.error(
                    "Failed to compact workspace before shutdown%s",
                    f":\n{error_text}" if error_text else "",
                    extra={"suppress_ui_alert": True},
                )
            else:
                self._manager._workspace_state.needs_full_save = False
                self._manager._workspace_state.delta_save_count = 0
                self._manager._workspace_state.reset_repack_estimate()
                self._manager._workspace_state.schema_version = (
                    workspace_format._current_workspace_schema_version()
                )
                workspace_path = self._manager._workspace_state.path
                if workspace_path is not None:
                    try:
                        self._refresh_workspace_payload_bindings_after_full_save(
                            workspace_path
                        )
                    except _WorkspacePostSaveBindingError:
                        logger.exception(
                            "Workspace was compacted before shutdown, but live "
                            "workspace data could not be rebound",
                            extra={"suppress_ui_alert": True},
                        )
                        if on_finished is not None:
                            on_finished()
                        return
                self._drain_workspace_deferred_events()
                post_save_events = tuple(
                    event
                    for event in self._manager._workspace_state.dirty_events
                    if event.generation > snapshot.generation
                )
                if post_save_events:
                    self._restore_workspace_dirty_events(post_save_events)
                else:
                    self._mark_workspace_clean()
            if on_finished is not None:
                on_finished()

        def _start_error() -> None:
            logger.error(
                "Failed to start workspace compaction before shutdown",
                extra={"suppress_ui_alert": True},
            )
            if on_finished is not None:
                on_finished()

        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            snapshot.close()
            return False
        return self._start_workspace_save_worker(
            workspace_path,
            snapshot,
            on_finished=_finish_compaction,
            on_start_error=_start_error,
        )

    def compact_workspace(self) -> bool:
        """Rewrite the current workspace file to remove unused space."""
        workspace_path = self._current_workspace_document_path()
        if workspace_path is None:
            return self.save_as()
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save already in progress", 3000
            )
            return False

        origin = self._active_managed_window()
        old_workspace_path = workspace_path
        backing_snapshot = self.loading._workspace_data_backing_snapshot()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self._manager, "Compacting workspace..."
            ):
                self.saving._save_workspace_document(
                    workspace_path,
                    force_full=True,
                    require_matching_compression=True,
                    mark_clean=False,
                )
                self._refresh_workspace_payload_bindings_after_full_save(
                    workspace_path,
                    backing_snapshot=backing_snapshot,
                    old_workspace_path=old_workspace_path,
                )
            self._manager._workspace_state.delta_save_count = 0
            self._manager._status_bar.showMessage("Workspace compacted", 5000)
            self._manager._workspace_state.needs_full_save = False
            self._mark_workspace_clean()
        except _WorkspacePostSaveBindingError:
            self._mark_workspace_post_save_binding_refresh_failed()
            self._show_workspace_post_save_binding_error(workspace_path)
            self._restore_focus_after_workspace_save(origin)
            return False
        except Exception:
            self._manager._show_operation_error(
                "Error while compacting workspace",
                "An error occurred while compacting the workspace file.",
            )
            self._restore_focus_after_workspace_save(origin)
            return False

        self._restore_focus_after_workspace_save(origin)
        self._manager._workspace_state.reset_repack_estimate()
        return True

    def load(self, *, native: bool = True) -> bool:
        """Replace this manager with a workspace file."""
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        dialog = QtWidgets.QFileDialog(self._manager)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("ImageTool Workspace Files (*.itws)")
        if (directory := self._manager._recent_or_default_directory()) is not None:
            dialog.setDirectory(directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return False

        fname = dialog.selectedFiles()[0]
        return self._open_workspace_after_dirty_prompt(fname, native=native)

    def import_workspace(self, *, native: bool = True) -> bool:
        """Import selected windows from another workspace file."""
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; import after it finishes", 3000
            )
            return False
        dialog = QtWidgets.QFileDialog(self._manager)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("ImageTool Workspace Files (*.itws)")
        if (directory := self._manager._recent_or_default_directory()) is not None:
            dialog.setDirectory(directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return False
        fname = dialog.selectedFiles()[0]
        self._manager._recent_directory = os.path.dirname(fname)
        try:
            loaded = self.loading._load_workspace_file(
                fname,
                replace=False,
                associate=False,
                mark_dirty=True,
                select=True,
            )
        except Exception as exc:
            if workspace_storage._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    fname,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self._manager, fname)
            else:
                logger.exception(
                    "Error while importing workspace",
                    extra={"suppress_ui_alert": True},
                )
                erlab.interactive.utils.MessageDialog.critical(
                    self._manager,
                    "Error",
                    "An error occurred while importing the workspace file.",
                )
            return False
        else:
            if loaded:
                self._record_recent_workspace(fname)
            return loaded
