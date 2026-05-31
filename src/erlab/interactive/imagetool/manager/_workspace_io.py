from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import pathlib
import sys
import typing
import uuid

import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.slicer
from erlab.interactive import _qt_state
from erlab.interactive.imagetool import provenance
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME, ImageTool
from erlab.interactive.imagetool.manager import _desktop
from erlab.interactive.imagetool.manager import _workspace as _manager_workspace
from erlab.interactive.imagetool.manager import _xarray as _manager_xarray
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _is_loader_func,
)
from erlab.interactive.imagetool.manager._registry import refresh_manager_record
from erlab.interactive.imagetool.manager._widgets import (
    _MAX_RECENT_WORKSPACES,
    _RECENT_WORKSPACES_SETTINGS_KEY,
    _WORKSPACE_REBIND_KEEP_CHUNKS,
    _WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME,
    _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS,
    _manager_settings,
    _show_workspace_file_lock_error,
    _strip_workspace_modified_placeholder,
    _window_title_with_modified_placeholder,
    _WorkspaceDocumentAccess,
    _WorkspacePropertiesDialog,
    _WorkspacePropertiesState,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._workspace_state import (
        _WorkspaceStateSnapshot,
    )

logger = logging.getLogger(__name__)


def _workspace_dataset_window_visible(
    ds: xr.Dataset, prefix: str, *, default: bool = True
) -> bool:
    state = _qt_state.parse_qt_window_state(ds.attrs.get(f"{prefix}_window_state"))
    if state is not None:
        return state.visible
    return bool(ds.attrs.get(f"{prefix}_visible", default))


def _workspace_provenance_file_stems(
    spec: provenance.ToolProvenanceSpec | None,
) -> tuple[str, ...]:
    stems: list[str] = []

    def collect(
        current: provenance.ToolProvenanceSpec | None,
    ) -> None:
        if current is None:
            return
        if current.file_load_source is not None:
            stem = pathlib.Path(current.file_load_source.path).stem
            if stem not in stems:
                stems.append(stem)
        for script_input in current.script_inputs:
            collect(script_input.parsed_provenance_spec())

    collect(spec)
    return tuple(stems)


def _workspace_compact_file_suffix(stems: tuple[str, ...]) -> str:
    if not stems:
        return ""
    if len(stems) <= 2:
        return f" ({', '.join(stems)})"
    return f" ({', '.join(stems[:2])}, +{len(stems) - 2})"


def _legacy_saved_title_data_name(
    ds: xr.Dataset,
    provenance_spec: provenance.ToolProvenanceSpec | None,
) -> str | None:
    title = _strip_workspace_modified_placeholder(str(ds.attrs.get("itool_title", "")))
    if ": " in title:
        prefix, rest = title.split(": ", maxsplit=1)
        if prefix.isdigit():
            title = rest
    saved_name = str(ds.attrs.get("itool_name", ""))
    stems = _workspace_provenance_file_stems(provenance_spec)
    if not saved_name and not stems:
        return None
    compact_suffix = _workspace_compact_file_suffix(stems)
    if compact_suffix and title.endswith(compact_suffix):
        title = title[: -len(compact_suffix)]
    if not title or title == saved_name:
        return None

    for stem in stems:
        if (saved_name and title == f"{saved_name} ({stem})") or (
            not saved_name and title == stem
        ):
            return None
        if saved_name and saved_name == stem and title == f"{stem} ({stem})":
            return None
    return title


class _WorkspaceIOController:
    def __init__(self, manager: ImageToolManager) -> None:
        self._manager = manager
        self._missing_workspace_colormaps: list[tuple[str, str]] = []
        self._loader_state = _manager_workspace.WorkspaceLoaderState()

    def _record_missing_workspace_colormap(
        self, cmap: str, node_path: str | None
    ) -> None:
        node_label = node_path if node_path is not None else "unknown workspace node"
        record = (node_label, cmap)
        if record not in self._missing_workspace_colormaps:
            self._missing_workspace_colormaps.append(record)

    def _dataset_without_missing_workspace_colormap(
        self, ds: xr.Dataset, node_path: str | None
    ) -> xr.Dataset:
        state_payload = ds.attrs.get("itool_state")
        if not isinstance(state_payload, str):
            return ds
        try:
            state = json.loads(state_payload)
        except Exception:
            return ds
        if not isinstance(state, dict):
            return ds
        color_state = state.get("color")
        if not isinstance(color_state, dict):
            return ds
        cmap = color_state.get("cmap")
        if not isinstance(cmap, str):
            return ds
        try:
            erlab.interactive.colors.pg_colormap_from_name(cmap)
        except RuntimeError:
            color_state = dict(color_state)
            color_state.pop("cmap", None)
            state = dict(state)
            state["color"] = color_state
            ds = ds.copy(deep=False)
            ds.attrs["itool_state"] = json.dumps(state)
            self._record_missing_workspace_colormap(cmap, node_path)
        return ds

    def _show_missing_workspace_colormap_warning(self) -> None:
        if not self._missing_workspace_colormaps:
            return
        affected = "\n".join(
            f"- {node_label}: {cmap}"
            for node_label, cmap in self._missing_workspace_colormaps
        )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title="Unavailable Colormap",
            text="Some saved colormaps are unavailable.",
            informative_text=(
                "ImageTool Manager used the default colormap for these windows:\n"
                f"{affected}"
            ),
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        dialog.exec()

    def _finish_workspace_file_load(self, loaded: bool) -> bool:
        if loaded:
            self._show_missing_workspace_colormap_warning()
        return loaded

    @staticmethod
    def _normalize_recent_workspace_paths(
        paths: Iterable[str | os.PathLike[str]],
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
            if len(recent_paths) >= _MAX_RECENT_WORKSPACES:
                break
        return recent_paths

    def _recent_workspace_paths(self) -> list[pathlib.Path]:
        settings = _manager_settings()
        settings.sync()
        values = settings.value(_RECENT_WORKSPACES_SETTINGS_KEY, [])
        if isinstance(values, str):
            stored_paths = [values] if values else []
        elif isinstance(values, (list, tuple)):
            stored_paths = [str(value) for value in values if value]
        else:
            stored_paths = []
        return self._manager._normalize_recent_workspace_paths(stored_paths)

    def _set_recent_workspace_paths(
        self, paths: Iterable[str | os.PathLike[str]]
    ) -> None:
        recent_paths = self._manager._normalize_recent_workspace_paths(paths)
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
            for existing in self._manager._recent_workspace_paths()
            if os.path.normcase(str(existing)) != path_key
        ]
        self._manager._set_recent_workspace_paths([path, *paths])
        self._manager._refresh_open_recent_menu_action()
        if erlab.utils.misc._IS_PACKAGED:
            _desktop.record_recent_workspace(path)

    def _clear_recent_workspaces(self) -> None:
        self._manager._set_recent_workspace_paths([])
        self._manager._populate_open_recent_menu()

    def _refresh_open_recent_menu_action(self) -> None:
        self._manager.open_recent_menu.setEnabled(
            bool(self._manager._recent_workspace_paths())
        )

    def _populate_open_recent_menu(self) -> None:
        self._manager.open_recent_menu.clear()
        paths = self._manager._recent_workspace_paths()
        self._manager.open_recent_menu.setEnabled(bool(paths))
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
        clear_action.triggered.connect(self._manager._clear_recent_workspaces)
        self._manager.open_recent_menu.addAction(clear_action)

    def open_recent_workspace(self, fname: str | os.PathLike[str]) -> bool:
        """Open a recently used workspace file."""
        path = pathlib.Path(fname).expanduser().resolve()
        if not path.exists():
            path_key = os.path.normcase(str(path))
            self._manager._set_recent_workspace_paths(
                existing
                for existing in self._manager._recent_workspace_paths()
                if os.path.normcase(str(existing)) != path_key
            )
            self._manager._refresh_open_recent_menu_action()
            QtWidgets.QMessageBox.warning(
                self._manager,
                "Workspace Not Found",
                f"The recent workspace file no longer exists:\n{path}",
            )
            return False
        if not self._manager._confirm_save_dirty_workspace(
            "Opening a workspace replaces the windows currently in this manager."
        ):
            return False
        self._manager._recent_directory = str(path.parent)
        try:
            loaded = self._manager._load_workspace_file(
                path,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
            )
        except Exception as exc:
            if _manager_workspace._is_workspace_file_lock_error(exc):
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
            self._manager._record_recent_workspace(path)
        return loaded

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
            state=self._manager._workspace_properties_state(),
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

    def _refresh_manager_record(self) -> None:
        refresh_manager_record(
            self._manager._manager_record.internal_id,
            workspace_path=self._manager.workspace_path,
        )

    def _update_workspace_window_title(self) -> None:
        if self._manager._workspace_state.path is None:
            window_file_path = ""
        else:
            window_file_path = typing.cast("str", self._manager.workspace_path)
        # Work around a macOS Qt/Cocoa crash observed during close-time workspace
        # saves in QWidget.setWindowFilePath() -> QImage.toCGImage(). The window is
        # closing, so the document proxy icon update is unnecessary here. Add the
        # QTBUG ID once a reproducible report exists and remove this guard once fixed.
        if not (
            sys.platform == "darwin" and self._manager._workspace_state.closing_document
        ):
            self._manager.setWindowFilePath(window_file_path)
        workspace_display_name = (
            "Untitled"
            if self._manager._workspace_state.path is None
            else self._manager._workspace_state.path.name
        )
        self._manager.setWindowTitle(
            f"{_window_title_with_modified_placeholder(workspace_display_name)}"
            f" - ImageTool Manager #{self._manager.manager_index}"
        )
        self._manager.setWindowModified(self._manager.is_workspace_modified)

    def _release_workspace_lock(self) -> None:
        if self._manager._workspace_state.lock is None:
            return
        self._manager._workspace_state.lock.unlock()
        self._manager._workspace_state.lock = None

    def _workspace_document_access(
        self, fname: str | os.PathLike[str]
    ) -> _WorkspaceDocumentAccess:
        workspace_path = pathlib.Path(fname).resolve()
        workspace_lock = None
        if workspace_path != self._manager._workspace_state.path:
            workspace_lock = _manager_workspace._acquire_workspace_document_lock(
                workspace_path
            )
        return _WorkspaceDocumentAccess(workspace_path, workspace_lock)

    @contextlib.contextmanager
    def _workspace_document_access_context(
        self, fname: str | os.PathLike[str]
    ) -> Iterator[_WorkspaceDocumentAccess]:
        access = self._manager._workspace_document_access(fname)
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
            self._manager._update_workspace_window_title()
            self._manager._refresh_manager_record()
            return

        if workspace_path is not None and workspace_lock is None:
            raise RuntimeError(
                "Changing the workspace path requires a pre-acquired document lock"
            )
        self._manager._release_workspace_lock()
        self._manager._workspace_state.lock = workspace_lock
        self._manager._workspace_state.path = workspace_path
        if self._manager._workspace_state.path is not None:
            self._manager._recent_directory = str(
                self._manager._workspace_state.path.parent
            )
        self._manager._update_workspace_window_title()
        self._manager._refresh_manager_record()

    def _adopt_workspace_path(self, fname: str | os.PathLike[str]) -> None:
        with self._manager._workspace_document_access_context(fname) as access:
            self._manager._set_workspace_path(
                access.path, workspace_lock=access.take_lock()
            )

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
        )
        blocks: list[str] = []
        for label, items in sections:
            if items:
                blocks.append(f"{label}:\n" + "\n".join(f"- {item}" for item in items))
        return "\n\n".join(blocks)

    def _set_node_window_modified(self, uid: str, modified: bool) -> None:
        node = self._manager._tool_graph.nodes.get(uid)
        if node is None:
            return
        window = node.window
        if window is None or not erlab.interactive.utils.qt_is_valid(window):
            return
        if node.tool_window is not None:
            display_name = node.tool_window._tool_display_name
            base_title = (
                f"{node.tool_window.tool_name}: {display_name}"
                if display_name
                else node.tool_window.tool_name
            )
        else:
            base_title = node.label_text
        title = _window_title_with_modified_placeholder(base_title)
        if title != window.windowTitle():
            window.setWindowTitle(title)
        window.setWindowModified(modified)

    def _apply_workspace_dirty_event(
        self, event: _manager_workspace._WorkspaceDirtyEvent
    ) -> bool:
        if event.uid is not None and (event.added or event.data or event.state):
            self._manager._set_node_window_modified(event.uid, True)
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
    ) -> None:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
        ):
            return
        event = _manager_workspace._WorkspaceDirtyEvent(
            generation=self._manager._workspace_state.dirty_generation + 1,
            uid=uid,
            data=data,
            state=state,
            added=added,
            removed=removed,
            structure=structure,
        )
        if event.uid is not None and (event.added or event.data or event.state):
            self._manager._set_node_window_modified(event.uid, True)
        self._manager._workspace_state.mark_dirty(event)
        self._manager._update_workspace_window_title()

    def _mark_node_added(self, uid: str) -> None:
        self._manager._mark_workspace_dirty(
            uid=uid, added=True, structure="Added window"
        )

    def _mark_node_data_dirty(self, uid: str) -> None:
        self._manager._mark_workspace_dirty(uid=uid, data=True)

    def _mark_node_state_dirty(self, uid: str) -> None:
        self._manager._mark_workspace_dirty(uid=uid, state=True)

    def _mark_tool_info_dirty(self, uid: str) -> None:
        if uid not in self._manager._workspace_state.dirty_state:
            self._manager._mark_node_state_dirty(uid)

    def _mark_workspace_structure_dirty(self, reason: str) -> None:
        self._manager._mark_workspace_dirty(structure=reason)

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
            self._manager._update_workspace_window_title()

    def _mark_workspace_clean(self) -> None:
        self._manager._workspace_state.mark_clean()
        for uid in tuple(self._manager._tool_graph.nodes):
            self._manager._set_node_window_modified(uid, False)
        self._manager._update_workspace_window_title()

    def _restore_workspace_dirty_events(
        self, events: Iterable[_manager_workspace._WorkspaceDirtyEvent]
    ) -> None:
        retained_events = list(events)
        self._manager._workspace_state.mark_clean()
        for uid in tuple(self._manager._tool_graph.nodes):
            self._manager._set_node_window_modified(uid, False)
        for event in retained_events:
            self._manager._apply_workspace_dirty_event(event)
        self._manager._workspace_state.dirty_events = retained_events
        self._manager._update_workspace_window_title()

    @contextlib.contextmanager
    def _workspace_load_context(self) -> Iterator[None]:
        with self._manager._workspace_state.load_context():
            yield

    def _drain_workspace_deferred_events(self) -> None:
        for _ in range(3):
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents()

    def _workspace_state_snapshot(self) -> _WorkspaceStateSnapshot:
        return self._manager._workspace_state.snapshot(
            node_uid_counter=self._manager._tool_graph.uid_counter
        )

    def _restore_workspace_state_snapshot(
        self, snapshot: _WorkspaceStateSnapshot
    ) -> None:
        self._manager._tool_graph.restore_uid_counter(snapshot["node_uid_counter"])
        dirty_uids = self._manager._workspace_state.restore(snapshot)
        if self._manager._workspace_state.path is not None:
            self._manager._recent_directory = str(
                self._manager._workspace_state.path.parent
            )
        for uid in tuple(self._manager._tool_graph.nodes):
            self._manager._set_node_window_modified(uid, uid in dirty_uids)
        self._manager._update_workspace_window_title()
        self._manager._refresh_manager_record()

    def _install_workspace_save_shortcut(self, widget: QtWidgets.QWidget) -> None:
        for shortcut in widget.findChildren(QtWidgets.QShortcut):
            if shortcut.objectName() == _WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME:
                return
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.StandardKey.Save, widget)
        shortcut.setObjectName(_WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._manager.save)

    def _annotate_workspace_dataset(
        self,
        ds: xr.Dataset,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        kind: typing.Literal["imagetool", "tool"],
    ) -> xr.Dataset:
        ds.attrs["manager_node_uid"] = node.uid
        ds.attrs["manager_node_kind"] = kind
        ds.attrs["manager_node_snapshot_token"] = node.snapshot_token
        ds.attrs["manager_node_added_at"] = node.added_time_iso
        persistence = node.persistence_view()
        provenance_spec = persistence.provenance_spec
        if provenance_spec is not None:
            ds.attrs["manager_node_provenance_spec"] = json.dumps(
                provenance_spec.model_dump(mode="json")
            )
        if isinstance(node, _ImageToolWrapper) and node.source_input_ndim is not None:
            ds.attrs["manager_node_source_input_ndim"] = int(node.source_input_ndim)
        if isinstance(node, _ImageToolWrapper) and node.watched:
            watched_metadata = node.watched_metadata()
            ds.attrs["manager_node_watched_varname"] = typing.cast(
                "str", watched_metadata["varname"]
            )
            ds.attrs["manager_node_watched_uid"] = typing.cast(
                "str", watched_metadata["uid"]
            )
            workspace_link_id = watched_metadata.get("workspace_link_id")
            if workspace_link_id is not None:
                ds.attrs["manager_node_watched_workspace_link_id"] = str(
                    workspace_link_id
                )
            source_label = watched_metadata.get("source_label")
            if source_label is not None:
                ds.attrs["manager_node_watched_source_label"] = str(source_label)
            source_uid = watched_metadata.get("source_uid")
            if source_uid is not None:
                ds.attrs["manager_node_watched_source_uid"] = str(source_uid)
            ds.attrs["manager_node_watched_connected"] = bool(
                watched_metadata.get("connected", False)
            )
        output_id = persistence.output_id
        if kind == "imagetool" and output_id is not None:
            ds.attrs["manager_node_output_id"] = output_id
        source_spec = persistence.source_spec
        if kind == "imagetool" and source_spec is not None:
            ds.attrs["manager_node_live_source_spec"] = json.dumps(
                source_spec.model_dump(mode="json")
            )
        source_binding = persistence.source_binding
        if kind == "imagetool" and source_binding is not None:
            ds.attrs["manager_node_live_source_binding"] = json.dumps(
                source_binding.model_dump(mode="json")
            )
        if kind == "imagetool" and (
            source_spec is not None
            or source_binding is not None
            or output_id is not None
        ):
            ds.attrs["manager_node_source_state"] = persistence.source_state
            ds.attrs["manager_node_source_auto_update"] = bool(
                persistence.source_auto_update
            )
        return ds

    def _serialize_workspace_node(
        self,
        constructor: dict[str, xr.Dataset],
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: str,
        *,
        include_children: bool,
    ) -> None:
        if node.is_imagetool:
            target: int | str = (
                node.index if isinstance(node, _ImageToolWrapper) else node.uid
            )
            ds = self._manager.get_imagetool(target).to_dataset()
            ds.attrs["itool_title"] = node.name
            constructor[f"{path}/imagetool"] = (
                self._manager._annotate_workspace_dataset(ds, node, kind="imagetool")
            )
        else:
            tool = typing.cast("erlab.interactive.utils.ToolWindow", node.tool_window)
            if not tool.can_save_and_load():
                return
            ds = tool.to_dataset()
            ds.attrs["tool_title"] = _strip_workspace_modified_placeholder(
                ds.attrs.get("tool_title", "")
            )
            constructor[f"{path}/tool"] = self._manager._annotate_workspace_dataset(
                ds, node, kind="tool"
            )

        if not include_children:
            return
        for child_uid in node._childtool_indices:
            child = self._manager._child_node(child_uid)
            self._manager._serialize_workspace_node(
                constructor,
                child,
                f"{path}/childtools/{child_uid}",
                include_children=include_children,
            )

    def _to_datatree(
        self, close: bool = False, include_children: bool = True
    ) -> xr.DataTree:
        """Convert the current state of the manager to a DataTree object."""
        constructor: dict[str, xr.Dataset] = {}
        for index in self._manager._workspace_root_indices():
            self._manager._serialize_workspace_node(
                constructor,
                self._manager._tool_graph.root_wrappers[index],
                str(index),
                include_children=include_children,
            )
            if close:
                self._manager.remove_imagetool(index)
        tree = xr.DataTree.from_dict(constructor)
        _manager_workspace._set_legacy_workspace_schema(tree.attrs)
        return tree

    def _load_workspace_imagetool_dataset(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        node_path: str | None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int | str:
        uid = ds.attrs.get("manager_node_uid")
        provenance_spec = ds.attrs.get("manager_node_provenance_spec")
        live_source_spec = ds.attrs.get("manager_node_live_source_spec")
        live_source_binding = ds.attrs.get("manager_node_live_source_binding")
        parse_provenance_spec = provenance.parse_tool_provenance_spec
        parsed_provenance_spec = None
        if provenance_spec is not None:
            try:
                provenance_payload = typing.cast(
                    "Mapping[str, typing.Any]",
                    json.loads(provenance_spec),
                )
                parsed_provenance_spec = parse_provenance_spec(provenance_payload)
            except Exception:
                logger.warning(
                    "Ignoring invalid saved manager provenance for node %s",
                    uid,
                    exc_info=True,
                )
        parsed_source_spec = None
        if live_source_spec is not None:
            try:
                source_payload = typing.cast(
                    "Mapping[str, typing.Any]",
                    json.loads(live_source_spec),
                )
                parsed_source_spec = provenance.require_live_source_spec(
                    parse_provenance_spec(source_payload)
                )
            except Exception:
                logger.warning(
                    "Ignoring invalid saved manager source provenance for node %s",
                    uid,
                    exc_info=True,
                )
        parsed_source_binding = None
        if live_source_binding is not None:
            try:
                binding_payload = typing.cast(
                    "Mapping[str, typing.Any]",
                    json.loads(live_source_binding),
                )
                binding_type = provenance.ImageToolSelectionSourceBinding
                parsed_source_binding = binding_type.model_validate(binding_payload)
            except Exception:
                logger.warning(
                    "Ignoring invalid saved manager source binding for node %s",
                    uid,
                    exc_info=True,
                )
        kwargs: dict[str, typing.Any] = {
            "uid": uid,
            "snapshot_token": ds.attrs.get("manager_node_snapshot_token"),
            "created_time": ds.attrs.get("manager_node_added_at"),
            "provenance_spec": parsed_provenance_spec,
            "source_spec": parsed_source_spec,
            "source_binding": parsed_source_binding,
            "output_id": ds.attrs.get("manager_node_output_id"),
            "source_auto_update": bool(
                ds.attrs.get("manager_node_source_auto_update", False)
            ),
            "source_state": typing.cast(
                "_ManagedWindowNode._source_state_type",
                ds.attrs.get("manager_node_source_state", "fresh"),
            ),
        }
        tool_kwargs: dict[str, typing.Any] = {"_in_manager": True}
        if _ITOOL_DATA_NAME in ds and ds[_ITOOL_DATA_NAME].chunks is not None:
            tool_kwargs["auto_compute"] = False
        legacy_name = _legacy_saved_title_data_name(ds, parsed_provenance_spec)
        if legacy_name is not None:
            ds = ds.copy(deep=False)
            ds.attrs["itool_name"] = legacy_name
        elif "itool_name" not in ds.attrs:
            ds = ds.copy(deep=False)
            ds.attrs["itool_name"] = ""
        ds = self._dataset_without_missing_workspace_colormap(ds, node_path)
        tool = ImageTool.from_dataset(ds, **tool_kwargs)
        if parent_target is not None:
            target = self._manager.add_imagetool_child(
                tool,
                parent_target,
                show=_workspace_dataset_window_visible(ds, "itool"),
                **kwargs,
            )
            self._manager._record_workspace_loaded_imagetool_target(
                ds, target, loaded_targets_by_uid
            )
            return target

        kwargs.pop("output_id", None)
        kwargs["source_input_ndim"] = typing.cast(
            "int | None",
            ds.attrs.get("manager_node_source_input_ndim"),
        )
        watched_varname = ds.attrs.get("manager_node_watched_varname")
        watched_uid = ds.attrs.get("manager_node_watched_uid")
        if watched_varname is not None and watched_uid is not None:
            kwargs["watched_var"] = (str(watched_varname), str(watched_uid))
            kwargs["watched_workspace_link_id"] = (
                None
                if ds.attrs.get("manager_node_watched_workspace_link_id") is None
                else str(ds.attrs["manager_node_watched_workspace_link_id"])
            )
            kwargs["watched_source_label"] = (
                None
                if ds.attrs.get("manager_node_watched_source_label") is None
                else str(ds.attrs["manager_node_watched_source_label"])
            )
            kwargs["watched_source_uid"] = (
                None
                if ds.attrs.get("manager_node_watched_source_uid") is None
                else str(ds.attrs["manager_node_watched_source_uid"])
            )
            # Loaded watched rows stay watched, but are disconnected until
            # a notebook explicitly reconnects them.
            kwargs["watched_connected"] = False
        preferred_index: int | None = None
        if node_path is not None and "/" not in node_path:
            with contextlib.suppress(ValueError):
                preferred_index = int(node_path)
            if preferred_index is not None and preferred_index < 0:
                preferred_index = None
        target = self._manager.add_imagetool(
            tool,
            show=_workspace_dataset_window_visible(ds, "itool"),
            index=preferred_index,
            **kwargs,
        )
        self._manager._record_workspace_loaded_imagetool_target(
            ds, target, loaded_targets_by_uid
        )
        return target

    def _load_workspace_tool_dataset(
        self, ds: xr.Dataset, *, parent_target: int | str | None
    ) -> int | str:
        if parent_target is None:
            raise ValueError("Workspace tool node has no parent")
        return self._manager.add_childtool(
            erlab.interactive.utils.ToolWindow.from_dataset(ds),
            parent_target,
            show=_workspace_dataset_window_visible(ds, "tool"),
            uid=ds.attrs.get("manager_node_uid"),
            snapshot_token=ds.attrs.get("manager_node_snapshot_token"),
            created_time=ds.attrs.get("manager_node_added_at"),
        )

    @staticmethod
    def _workspace_saved_uid_from_dataset(ds: xr.Dataset) -> str | None:
        uid = ds.attrs.get("manager_node_uid")
        if isinstance(uid, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                uid = uid.decode()
        if isinstance(uid, str) and uid:
            return uid
        return None

    def _record_workspace_loaded_imagetool_target(
        self,
        ds: xr.Dataset,
        target: int | str,
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> None:
        if loaded_targets_by_uid is None:
            return
        saved_uid = self._manager._workspace_saved_uid_from_dataset(ds)
        if saved_uid is not None:
            loaded_targets_by_uid[saved_uid] = target

    def _restore_workspace_link_groups(
        self,
        manifest: Mapping[str, typing.Any] | None,
        loaded_targets_by_uid: Mapping[str, int | str],
    ) -> None:
        if manifest is None:
            return
        nodes = manifest.get("nodes", ())
        if not isinstance(nodes, list):
            return

        group_targets: dict[int, list[int | str]] = {}
        group_colors: dict[int, bool] = {}
        invalid_groups: set[int] = set()
        for entry in nodes:
            if not isinstance(entry, dict) or "link_group" not in entry:
                continue
            uid = entry.get("uid")
            link_group = entry.get("link_group")
            link_colors = entry.get("link_colors")
            if (
                not isinstance(uid, str)
                or type(link_group) is not int
                or not isinstance(link_colors, bool)
            ):
                continue
            target = loaded_targets_by_uid.get(uid)
            if target is None:
                continue
            try:
                node = self._manager._node_for_target(target)
            except KeyError:
                continue
            if not node.is_imagetool or node.imagetool is None:
                continue
            current_group_colors = group_colors.get(link_group)
            if current_group_colors is None:
                group_colors[link_group] = link_colors
            elif current_group_colors != link_colors:
                invalid_groups.add(link_group)
                continue
            targets = group_targets.setdefault(link_group, [])
            if target not in targets:
                targets.append(target)

        for link_group in sorted(group_targets):
            if link_group in invalid_groups:
                continue
            targets = [
                target
                for target in group_targets[link_group]
                if not self._manager.get_imagetool(target).slicer_area.is_linked
            ]
            if len(targets) <= 1:
                continue
            self._manager.link_imagetools(
                *targets,
                link_colors=group_colors.get(link_group, True),
            )

    def _load_workspace_node(
        self,
        node_tree: xr.DataTree,
        *,
        parent_target: int | str | None = None,
        selection_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        node_path: str | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int | str:
        if "imagetool" in node_tree:
            ds = None
            if (
                manifest is not None
                and node_path is not None
                and workspace_file_path is not None
            ):
                nodes = manifest.get("nodes", ())
                if isinstance(nodes, list):
                    for entry in nodes:
                        if (
                            isinstance(entry, dict)
                            and entry.get("path") == node_path
                            and entry.get("kind") == "imagetool"
                            and entry.get("data_backing") == "dask"
                        ):
                            opened = _manager_xarray.open_workspace_dataset(
                                workspace_file_path,
                                f"{node_path}/imagetool",
                                chunks={},
                            )
                            try:
                                ds = opened.copy(deep=False)
                            finally:
                                opened.close()
                            break
            if ds is None:
                ds = (
                    typing.cast("xr.DataTree", node_tree["imagetool"])
                    .to_dataset(inherit=False)
                    .load()
                )
            target = self._manager._load_workspace_imagetool_dataset(
                ds,
                parent_target=parent_target,
                node_path=node_path,
                loaded_targets_by_uid=loaded_targets_by_uid,
            )
        elif "tool" in node_tree:
            ds = (
                typing.cast("xr.DataTree", node_tree["tool"])
                .to_dataset(inherit=False)
                .load()
            )
            target = self._manager._load_workspace_tool_dataset(
                ds, parent_target=parent_target
            )
        else:
            raise ValueError("Workspace node has no supported window payload")

        if "childtools" in node_tree:
            childtools = typing.cast("xr.DataTree", node_tree["childtools"])
            child_keys: list[str] = []
            if manifest is not None and node_path is not None:
                nodes = manifest.get("nodes", ())
                prefix = f"{node_path}/childtools/"
                if isinstance(nodes, list):
                    for entry in nodes:
                        if not isinstance(entry, dict):
                            continue
                        path = entry.get("path")
                        if not isinstance(path, str) or not path.startswith(prefix):
                            continue
                        child_key = path.removeprefix(prefix)
                        if "/" not in child_key and child_key not in child_keys:
                            child_keys.append(child_key)
            child_keys.extend(
                str(key) for key in childtools if str(key) not in child_keys
            )

            for child_key in child_keys:
                if child_key not in childtools:
                    continue
                child_node = typing.cast("xr.DataTree", childtools[child_key])
                child_item = self._manager._tree_item_child_by_key(
                    selection_item, child_key
                )
                if (
                    child_item is not None
                    and child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked
                ):
                    continue
                self._manager._load_workspace_node(
                    child_node,
                    parent_target=target,
                    selection_item=child_item,
                    manifest=manifest,
                    workspace_file_path=workspace_file_path,
                    node_path=(
                        None
                        if node_path is None
                        else f"{node_path}/childtools/{child_key}"
                    ),
                    loaded_targets_by_uid=loaded_targets_by_uid,
                )
        return target

    def _load_workspace_roots(
        self,
        tree: xr.DataTree,
        root_keys: Iterable[str],
        *,
        root_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> None:
        for key in root_keys:
            if key not in tree:
                continue
            node = typing.cast("xr.DataTree", tree[key])
            item = self._manager._tree_item_child_by_key(root_item, key)
            if item is None or item.checkState(0) != QtCore.Qt.CheckState.Unchecked:
                self._manager._load_workspace_node(
                    node,
                    selection_item=item,
                    manifest=manifest,
                    workspace_file_path=workspace_file_path,
                    node_path=key,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                )

    def _from_h5py_workspace_file(
        self,
        fname: str | os.PathLike[str],
        manifest: Mapping[str, typing.Any],
        *,
        replace: bool,
        mark_dirty: bool,
    ) -> bool:
        nodes = manifest.get("nodes", ())
        root_order = manifest.get("root_order", ())
        if not isinstance(nodes, list) or not isinstance(root_order, list):
            raise TypeError("Workspace manifest is missing node ordering")

        entries_by_path: dict[str, dict[str, typing.Any]] = {}
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            kind = entry.get("kind")
            if not isinstance(path, str) or kind not in {"imagetool", "tool"}:
                continue
            entries_by_path[path] = entry
        if not entries_by_path:
            raise ValueError("Workspace manifest has no loadable nodes")

        root_paths: list[str] = []
        for root in root_order:
            path = str(root)
            if path in entries_by_path and "/" not in path and path not in root_paths:
                root_paths.append(path)
        root_paths.extend(
            path
            for path in entries_by_path
            if "/" not in path and path not in root_paths
        )
        if not root_paths:
            raise ValueError("Workspace manifest has no root ImageTool nodes")

        loaded_targets_by_uid: dict[str, int | str] = {}
        child_paths: dict[str, list[str]] = {path: [] for path in entries_by_path}
        for path in entries_by_path:
            if "/childtools/" not in path:
                continue
            parent_path, child_key = path.rsplit("/childtools/", maxsplit=1)
            if "/" not in child_key and parent_path in entries_by_path:
                child_paths[parent_path].append(path)

        def _load_xarray_dataset(
            payload_path: str, *, chunks: typing.Any, load: bool
        ) -> xr.Dataset:
            opened = _manager_xarray.open_workspace_dataset(
                fname, payload_path, chunks=chunks
            )
            try:
                if load:
                    return opened.load()
                return opened.copy(deep=False)
            finally:
                opened.close()

        def _load_dataset(
            payload_path: str, *, entry: Mapping[str, typing.Any], imagetool: bool
        ) -> xr.Dataset:
            if imagetool and entry.get("data_backing") == "dask":
                return _load_xarray_dataset(payload_path, chunks={}, load=False)
            if imagetool:
                try:
                    ds = _manager_workspace._read_workspace_dataset_group_h5py(
                        fname,
                        payload_path,
                        preferred_data_name=_ITOOL_DATA_NAME,
                    )
                except Exception:
                    logger.debug(
                        "Failed h5py workspace payload read for %s",
                        payload_path,
                        exc_info=True,
                    )
                else:
                    if ds is not None:
                        return ds

            return _load_xarray_dataset(payload_path, chunks=None, load=True)

        def _load_path(path: str, parent_target: int | str | None = None) -> int | str:
            entry = entries_by_path[path]
            kind = typing.cast("str", entry["kind"])
            is_imagetool = kind == "imagetool"
            payload_path = f"{path}/{'imagetool' if is_imagetool else 'tool'}"
            ds = _load_dataset(payload_path, entry=entry, imagetool=is_imagetool)
            if is_imagetool:
                target = self._manager._load_workspace_imagetool_dataset(
                    ds,
                    parent_target=parent_target,
                    node_path=path,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                )
            else:
                target = self._manager._load_workspace_tool_dataset(
                    ds, parent_target=parent_target
                )
            for child_path in child_paths[path]:
                _load_path(child_path, target)
            return target

        if replace:
            manifest_workspace_link_id = manifest.get("workspace_link_id")
            self._manager._workspace_state.link_id = (
                str(manifest_workspace_link_id)
                if manifest_workspace_link_id
                else uuid.uuid4().hex
            )

        maybe_guard = (
            self._manager._workspace_load_context()
            if not mark_dirty
            else contextlib.nullcontext()
        )
        backup_tree: xr.DataTree | None = None
        backup_snapshot: _WorkspaceStateSnapshot | None = None
        with (
            maybe_guard,
            erlab.interactive.utils.wait_dialog(self._manager, "Loading workspace..."),
        ):
            try:
                if replace:
                    backup_snapshot = self._manager._workspace_state_snapshot()
                    backup_tree = self._manager._to_datatree()
                    self._manager.remove_all_tools()
                for root_path in root_paths:
                    _load_path(root_path)
                self._manager._rebase_loaded_workspace_dependency_refs(
                    loaded_targets_by_uid
                )
                self._manager._restore_workspace_link_groups(
                    manifest, loaded_targets_by_uid
                )
                if replace:
                    self._manager._restore_workspace_layout(manifest)
                    self._restore_workspace_loader_state(manifest)
                    self._restore_standalone_apps_state(manifest)
                if not mark_dirty:
                    self._manager._drain_workspace_deferred_events()
            except Exception:
                if backup_tree is not None and backup_snapshot is not None:
                    try:
                        self._manager._restore_replaced_workspace(
                            backup_tree, backup_snapshot
                        )
                    except Exception:
                        logger.exception("Failed to restore previous workspace")
                raise
            finally:
                if backup_tree is not None:
                    backup_tree.close()
        return True

    def _restore_replaced_workspace(
        self,
        backup_tree: xr.DataTree,
        snapshot: _WorkspaceStateSnapshot,
    ) -> None:
        with self._manager._workspace_load_context():
            self._manager.remove_all_tools()
            self._manager._load_workspace_roots(
                backup_tree, [str(key) for key in backup_tree]
            )
            self._manager._drain_workspace_deferred_events()
        self._manager._restore_workspace_state_snapshot(snapshot)

    def _from_datatree(
        self,
        tree: xr.DataTree,
        *,
        replace: bool = False,
        mark_dirty: bool = True,
        select: bool = True,
        workspace_file_path: str | os.PathLike[str] | None = None,
    ) -> bool:
        """Restore the state of the manager from a DataTree object."""
        opened_tree = tree
        try:
            if not self._manager._is_datatree_workspace(tree):
                raise ValueError("Not a valid workspace file")

            schema_version, _delta_save_count, manifest = (
                _manager_workspace._workspace_file_metadata_from_attrs(tree.attrs)
            )
            match schema_version:
                case 1:
                    tree = self._manager._parse_datatree_compat_v1(tree)
                case 2:
                    tree = self._manager._parse_datatree_compat_v2(tree)
                case 3:
                    pass
                case 4:
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported workspace schema version {schema_version}, "
                        "file may be from a newer version of erlab"
                    )
            if replace:
                manifest_workspace_link_id = (
                    None if manifest is None else manifest.get("workspace_link_id")
                )
                self._manager._workspace_state.link_id = (
                    str(manifest_workspace_link_id)
                    if manifest_workspace_link_id
                    else uuid.uuid4().hex
                )

            root_keys = _manager_workspace._workspace_root_keys(tree, manifest)

            dialog: _ChooseFromDataTreeDialog | None = None
            if select:
                dialog = _ChooseFromDataTreeDialog(
                    self._manager,
                    tree,
                    mode="load",
                    root_keys=root_keys,
                )
                if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    return False

            maybe_guard = (
                self._manager._workspace_load_context()
                if not mark_dirty
                else contextlib.nullcontext()
            )
            backup_tree: xr.DataTree | None = None
            backup_snapshot: _WorkspaceStateSnapshot | None = None
            loaded_targets_by_uid: dict[str, int | str] = {}
            with (
                maybe_guard,
                erlab.interactive.utils.wait_dialog(
                    self._manager, "Loading workspace..."
                ),
            ):
                try:
                    if replace:
                        backup_snapshot = self._manager._workspace_state_snapshot()
                        backup_tree = self._manager._to_datatree()
                        self._manager.remove_all_tools()
                    root_item = (
                        None
                        if dialog is None
                        else dialog._tree_widget.invisibleRootItem()
                    )
                    self._manager._load_workspace_roots(
                        tree,
                        root_keys,
                        root_item=root_item,
                        manifest=manifest,
                        workspace_file_path=workspace_file_path,
                        loaded_targets_by_uid=loaded_targets_by_uid,
                    )
                    self._manager._rebase_loaded_workspace_dependency_refs(
                        loaded_targets_by_uid
                    )
                    self._manager._restore_workspace_link_groups(
                        manifest, loaded_targets_by_uid
                    )
                    if replace:
                        self._manager._restore_workspace_layout(manifest)
                        self._restore_workspace_loader_state(manifest)
                        self._restore_standalone_apps_state(manifest)
                    if not mark_dirty:
                        self._manager._drain_workspace_deferred_events()
                except Exception:
                    if backup_tree is not None and backup_snapshot is not None:
                        try:
                            self._manager._restore_replaced_workspace(
                                backup_tree, backup_snapshot
                            )
                        except Exception:
                            logger.exception("Failed to restore previous workspace")
                    raise
                finally:
                    if backup_tree is not None:
                        backup_tree.close()
            return True
        finally:
            tree.close()
            if tree is not opened_tree:
                opened_tree.close()

    def _parse_datatree_compat_v1(self, tree: xr.DataTree) -> xr.DataTree:
        """Restore the state of the manager from a DataTree object.

        This is for the legacy format where only imagetools are stored at the root level
        (saved from erlab v3.14.1 and earlier).
        """
        return xr.DataTree.from_dict(
            {f"{i}/imagetool": node.dataset for i, node in tree.items()}
        )

    def _parse_datatree_compat_v2(self, tree: xr.DataTree) -> xr.DataTree:
        constructor: dict[str, xr.Dataset] = {}
        for key, node in tree.items():
            constructor[f"{key}/imagetool"] = typing.cast(
                "xr.DataTree", node["imagetool"]
            ).to_dataset(inherit=False)
            if "childtools" in node:
                for child_key, child_node in typing.cast(
                    "xr.DataTree", node["childtools"]
                ).items():
                    constructor[f"{key}/childtools/{child_key}/tool"] = typing.cast(
                        "xr.DataTree", child_node
                    ).to_dataset(inherit=False)
        converted = xr.DataTree.from_dict(constructor)
        converted.attrs["imagetool_workspace_schema_version"] = 3
        return converted

    def _is_datatree_workspace(self, tree: xr.DataTree) -> bool:
        """Check if the given DataTree object is a valid workspace file."""
        if "imagetool_workspace_schema_version" in tree.attrs:
            return True
        # Legacy format
        return tree.attrs.get("is_itool_workspace", 0) == 1

    def _workspace_node_path(self, uid: str) -> str:
        node = self._manager._tool_graph.nodes[uid]
        if isinstance(node, _ImageToolWrapper):
            return str(node.index)
        if node.parent_uid is None:
            raise KeyError(f"Node {uid!r} has no parent")
        return f"{self._manager._workspace_node_path(node.parent_uid)}/childtools/{uid}"

    def _workspace_payload_path(self, uid: str) -> str:
        node = self._manager._tool_graph.nodes[uid]
        payload_name = "imagetool" if node.is_imagetool else "tool"
        return f"{self._manager._workspace_node_path(uid)}/{payload_name}"

    def _workspace_root_indices(self) -> tuple[int, ...]:
        return self._manager._tool_graph.root_indices_for_workspace()

    def _workspace_link_metadata_by_uid(self) -> dict[str, tuple[int, bool]]:
        metadata: dict[str, tuple[int, bool]] = {}
        group_index = 0
        for linker in self._manager._link_registry.linkers:
            linked_nodes: list[_ImageToolWrapper | _ManagedWindowNode] = []
            for slicer_area in linker.children:
                node = self._manager.node_from_slicer_area(slicer_area)
                if (
                    node is None
                    or not node.is_imagetool
                    or node.imagetool is None
                    or node.slicer_area._linking_proxy is not linker
                ):
                    continue
                linked_nodes.append(node)
            if len(linked_nodes) <= 1:
                continue
            for node in linked_nodes:
                metadata[node.uid] = (group_index, bool(linker.link_colors))
            group_index += 1
        return metadata

    def _workspace_node_manifest_entries(self) -> list[dict[str, typing.Any]]:
        entries: list[dict[str, typing.Any]] = []
        link_metadata = self._manager._workspace_link_metadata_by_uid()

        def _append(uid: str) -> None:
            node = self._manager._tool_graph.nodes[uid]
            entry: dict[str, typing.Any] = {
                "uid": uid,
                # Payload group path relative to the workspace root HDF5 group.
                "path": self._manager._workspace_node_path(uid),
                # Restores graph node type without probing payload attrs first.
                "kind": "imagetool" if node.is_imagetool else "tool",
                "parent_uid": node.parent_uid,
                "display_name": node.display_text,
            }
            if node.is_imagetool and node.imagetool is not None:
                persistence = node.persistence_view()
                # Distinguishes embedded data from lazy file-backed/dask payloads.
                entry["data_backing"] = persistence.data_backing
                link_info = link_metadata.get(uid)
                if link_info is not None:
                    # link_group is an ordinal within this manifest, not a stable id.
                    entry["link_group"], entry["link_colors"] = link_info
            entries.append(entry)
            for child_uid in node._childtool_indices:
                if child_uid in self._manager._tool_graph.nodes:
                    _append(child_uid)

        for index in self._manager._workspace_root_indices():
            _append(self._manager._tool_graph.root_wrappers[index].uid)
        return entries

    @staticmethod
    def _tree_item_child_by_key(
        item: QtWidgets.QTreeWidgetItem | None, key: str
    ) -> QtWidgets.QTreeWidgetItem | None:
        if item is None:
            return None
        for i in range(item.childCount()):
            child = item.child(i)
            if (
                child is not None
                and child.data(0, QtCore.Qt.ItemDataRole.UserRole) == key
            ):
                return child
        return None

    def _workspace_root_attrs_payload(
        self, *, delta_save_count: int | None = None
    ) -> dict[str, typing.Any]:
        if delta_save_count is None:
            delta_save_count = self._manager._workspace_state.delta_save_count
        return _manager_workspace._workspace_root_attrs_payload(
            root_order=self._manager._workspace_root_indices(),
            nodes=self._manager._workspace_node_manifest_entries(),
            delta_save_count=delta_save_count,
            erlab_version=str(erlab.__version__),
            workspace_link_id=self._manager._workspace_state.link_id,
            manager_layout=self._manager._workspace_layout_snapshot(),
            loader_state=self._workspace_loader_state_snapshot(),
            standalone_apps=self._workspace_standalone_apps_snapshot(),
        )

    def _workspace_layout_snapshot(self) -> dict[str, typing.Any]:
        return {
            "window_state": _qt_state.qt_window_state_payload(self._manager),
            # QSplitter state preserves pane sizes and collapsed/expanded handles.
            "main_splitter": erlab.interactive.utils._qt_bytearray_to_base64(
                self._manager.main_splitter.saveState()
            ),
            "right_splitter": erlab.interactive.utils._qt_bytearray_to_base64(
                self._manager.right_splitter.saveState()
            ),
        }

    def _restore_workspace_layout(
        self, manifest: Mapping[str, typing.Any] | None
    ) -> None:
        if manifest is None:
            return
        layout = manifest.get("manager_layout")
        if not isinstance(layout, dict):
            return

        _qt_state.restore_qt_window_state(self._manager, layout.get("window_state"))

        main_splitter = erlab.interactive.utils._qt_bytearray_from_base64(
            layout.get("main_splitter")
        )
        if main_splitter is not None:
            self._manager.main_splitter.restoreState(main_splitter)

        right_splitter = erlab.interactive.utils._qt_bytearray_from_base64(
            layout.get("right_splitter")
        )
        if right_splitter is not None:
            self._manager.right_splitter.restoreState(right_splitter)

    def _workspace_loader_state_snapshot(self) -> dict[str, typing.Any]:
        manager_loader_kwargs = self._manager._recent_loader_kwargs_by_filter
        manager_loader_extensions = self._manager._recent_loader_extensions_by_filter
        explorer_kwargs = self._loader_state.explorer_loader_kwargs_by_name
        explorer_extensions = self._loader_state.explorer_loader_extensions_by_name
        explorer = self._manager._standalone_app_windows.get("explorer")
        if explorer is not None and erlab.interactive.utils.qt_is_valid(explorer):
            kwargs_getter = getattr(explorer, "loader_kwargs_by_name", None)
            if callable(kwargs_getter):
                explorer_kwargs = kwargs_getter()
            extensions_getter = getattr(explorer, "loader_extensions_by_name", None)
            if callable(extensions_getter):
                explorer_extensions = extensions_getter()
        state = _manager_workspace.WorkspaceLoaderState(
            recent_directory=self._manager._recent_directory,
            recent_name_filter=self._manager._recent_name_filter,
            manager_loader_kwargs_by_filter={
                str(name): dict(kwargs)
                for name, kwargs in manager_loader_kwargs.items()
            },
            manager_loader_extensions_by_filter={
                str(name): dict(extensions)
                for name, extensions in manager_loader_extensions.items()
            },
            explorer_loader_kwargs_by_name={
                str(name): dict(kwargs) for name, kwargs in explorer_kwargs.items()
            },
            explorer_loader_extensions_by_name={
                str(name): dict(extensions)
                for name, extensions in explorer_extensions.items()
            },
        )
        self._loader_state = state
        return state.model_dump(mode="json", exclude_none=True)

    def _explorer_loader_state(
        self,
    ) -> tuple[dict[str, dict[str, typing.Any]], dict[str, dict[str, typing.Any]]]:
        loader_kwargs = self._loader_state.explorer_loader_kwargs_by_name
        loader_extensions = self._loader_state.explorer_loader_extensions_by_name
        return (
            {str(name): dict(kwargs) for name, kwargs in loader_kwargs.items()},
            {
                str(name): dict(extensions)
                for name, extensions in loader_extensions.items()
            },
        )

    def _restore_workspace_loader_state(
        self,
        manifest: Mapping[str, typing.Any] | None,
        *,
        apply_explorer: bool = True,
    ) -> None:
        if manifest is None:
            return
        payload = manifest.get("loader_state")
        if not isinstance(payload, dict):
            return
        try:
            state = _manager_workspace.WorkspaceLoaderState.model_validate(payload)
        except Exception:
            logger.warning("Ignoring invalid workspace loader state", exc_info=True)
            return

        self._loader_state = state
        self._manager._recent_directory = state.recent_directory
        self._manager._recent_name_filter = state.recent_name_filter
        self._manager._recent_loader_kwargs_by_filter = {
            str(name): dict(kwargs)
            for name, kwargs in state.manager_loader_kwargs_by_filter.items()
        }
        self._manager._recent_loader_extensions_by_filter = {
            str(name): dict(extensions)
            for name, extensions in state.manager_loader_extensions_by_filter.items()
        }
        explorer_kwargs = {
            str(name): dict(kwargs)
            for name, kwargs in state.explorer_loader_kwargs_by_name.items()
        }
        explorer_extensions = {
            str(name): dict(extensions)
            for name, extensions in state.explorer_loader_extensions_by_name.items()
        }
        if not apply_explorer:
            return
        explorer = self._manager._standalone_app_windows.get("explorer")
        if explorer is not None and erlab.interactive.utils.qt_is_valid(explorer):
            apply_loader_state = getattr(explorer, "apply_loader_state", None)
            if callable(apply_loader_state):
                apply_loader_state(
                    kwargs_by_name=explorer_kwargs,
                    extensions_by_name=explorer_extensions,
                )

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

    def _workspace_standalone_apps_snapshot(self) -> dict[str, typing.Any]:
        app_states: dict[str, dict[str, typing.Any]] = {}
        for key in self._manager._standalone_app_specs:
            widget = self._manager._standalone_app_windows.get(key)
            state: dict[str, typing.Any] | None = None
            if widget is not None and erlab.interactive.utils.qt_is_valid(widget):
                state_getter = getattr(widget, "workspace_state_payload", None)
                if callable(state_getter):
                    state = typing.cast("dict[str, typing.Any]", state_getter())
            elif key in self._manager._standalone_app_pending_states:
                state = self._manager._standalone_app_pending_states[key]
            if state is None:
                continue
            validated = self._validated_standalone_app_state(key, state)
            if validated is not None:
                app_states[key] = validated
        return _manager_workspace.StandaloneAppsState(apps=app_states).model_dump(
            mode="json", exclude_none=True
        )

    def _restore_standalone_apps_state(
        self, manifest: Mapping[str, typing.Any] | None
    ) -> None:
        if manifest is None:
            return
        payload = manifest.get("standalone_apps")
        if not isinstance(payload, dict):
            return
        try:
            state = _manager_workspace.StandaloneAppsState.model_validate(payload)
        except Exception:
            logger.warning("Ignoring invalid standalone app state", exc_info=True)
            return

        restored_states: dict[str, dict[str, typing.Any]] = {}
        for key, app_state in state.apps.items():
            if key not in self._manager._standalone_app_specs:
                continue
            validated = self._validated_standalone_app_state(key, app_state)
            if validated is not None:
                restored_states[key] = validated

        for key in set(self._manager._standalone_app_specs) - set(restored_states):
            self._manager._close_standalone_app(key)

        for key, app_state in restored_states.items():
            window_state = _qt_state.parse_qt_window_state(
                app_state.get("window_state")
            )
            if window_state is not None and window_state.visible:
                widget = self._manager._ensure_standalone_app(key)
                self._manager._apply_standalone_app_state(key, widget, app_state)
            else:
                widget = self._manager._standalone_app_windows.get(key)
                if widget is not None and erlab.interactive.utils.qt_is_valid(widget):
                    self._manager._apply_standalone_app_state(key, widget, app_state)
                else:
                    self._manager._standalone_app_pending_states[key] = app_state

    def _write_full_workspace_file(self, fname: str | os.PathLike[str]) -> None:
        tree: xr.DataTree = self._manager._to_datatree()
        copy_source, copy_groups = self._manager._workspace_full_save_copy_groups(tree)
        try:
            _manager_workspace._write_full_workspace_tree_file(
                fname,
                tree,
                self._manager._workspace_root_attrs_payload(delta_save_count=0),
                copy_source=copy_source,
                copy_groups=copy_groups,
            )
        finally:
            tree.close()

    def _workspace_highest_dirty_data_roots(self) -> list[str]:
        dirty_existing = [
            uid
            for uid in self._manager._workspace_state.dirty_data
            if uid in self._manager._tool_graph.nodes
        ]
        dirty_set = set(dirty_existing)
        roots: list[str] = []
        for uid in sorted(
            dirty_existing, key=lambda value: self._manager._workspace_node_path(value)
        ):
            node = self._manager._tool_graph.nodes[uid]
            parent_uid = node.parent_uid
            has_dirty_ancestor = False
            while parent_uid is not None:
                if parent_uid in dirty_set:
                    has_dirty_ancestor = True
                    break
                parent_uid = self._manager._tool_graph.nodes[parent_uid].parent_uid
            if not has_dirty_ancestor:
                roots.append(uid)
        return roots

    def _save_workspace_delta(self, fname: str | os.PathLike[str]) -> None:
        delta_save_count = self._manager._workspace_state.delta_save_count + 1
        snapshot = self._manager._workspace_delta_save_snapshot(
            self._manager._workspace_state.dirty_generation,
            self._manager._workspace_root_attrs_payload(
                delta_save_count=delta_save_count
            ),
            delta_save_count,
        )
        try:
            _manager_workspace._write_workspace_transaction_file(
                fname,
                snapshot.rewrite_groups,
                snapshot.attr_updates,
                snapshot.root_attrs,
            )
        finally:
            snapshot.close()

    def _save_workspace_document(
        self,
        fname: str | os.PathLike[str],
        *,
        force_full: bool = False,
        document_access: _WorkspaceDocumentAccess | None = None,
    ) -> None:
        if document_access is None:
            with self._manager._workspace_document_access_context(fname) as access:
                self._manager._save_workspace_document(
                    access.path,
                    force_full=force_full,
                    document_access=access,
                )
            return

        fname = document_access.path
        self._manager._workspace_state.saving_depth += 1
        try:
            _manager_workspace._recover_workspace_transactions(fname)
            requires_full_save = (
                force_full or self._manager._workspace_requires_full_save(fname)
            )
            if requires_full_save:
                self._manager._write_full_workspace_file(fname)
                self._manager._workspace_state.delta_save_count = 0
                self._manager._workspace_state.schema_version = (
                    _manager_workspace._current_workspace_schema_version()
                )
            else:
                self._manager._save_workspace_delta(fname)
                self._manager._workspace_state.delta_save_count += 1
        finally:
            self._manager._workspace_state.saving_depth -= 1
        self._manager._workspace_state.needs_full_save = False
        self._manager._mark_workspace_clean()

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
        elif self._manager._recent_directory is not None:
            dialog.setDirectory(self._manager._recent_directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return None
        return dialog.selectedFiles()[0]

    def _confirm_save_dirty_workspace(self, action_text: str) -> bool:
        if not self._manager.is_workspace_modified:
            return True

        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setText("Save changes to this workspace?")
        msg_box.setInformativeText(action_text)
        details = self._manager._dirty_details_text()
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
            return self._manager.save()
        return result == QtWidgets.QMessageBox.StandardButton.Discard

    def _show_legacy_workspace_upgrade_message(
        self, fname: str | os.PathLike[str]
    ) -> None:
        msg_box = QtWidgets.QMessageBox(self._manager)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Save Legacy Workspace")
        msg_box.setText("This workspace uses an older file format.")
        msg_box.setInformativeText(
            "Save it again so ImageTool Manager can read it properly."
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
        self._manager._show_legacy_workspace_upgrade_message(fname)
        converted_fname = self._manager._workspace_save_dialog(
            native=native,
            caption="Save Converted Workspace",
            selected_file=fname,
        )
        if converted_fname is None:
            return None
        converted_path = pathlib.Path(converted_fname).resolve()
        if existing_access is not None and converted_path == existing_access.path:
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Saving workspace..."
            ):
                self._manager._save_workspace_document(
                    existing_access.path,
                    force_full=True,
                    document_access=existing_access,
                )
            return str(existing_access.path), existing_access.take_lock()

        with self._manager._workspace_document_access_context(
            converted_fname
        ) as access:
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Saving workspace..."
            ):
                self._manager._save_workspace_document(
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
        workspace_access: _WorkspaceDocumentAccess | None = None,
        rebind_data: bool = True,
    ) -> None:
        associated_fname = fname
        associated_lock: QtCore.QLockFile | None = None
        if _manager_workspace._workspace_schema_requires_conversion(schema_version):
            converted = self._manager._save_legacy_workspace_as_v4(
                fname, native=native, existing_access=workspace_access
            )
            if converted is None:
                self._manager._set_workspace_path(None)
                self._manager._workspace_state.needs_full_save = True
                self._manager._mark_workspace_structure_dirty(
                    "Legacy workspace needs conversion"
                )
                return
            associated_fname, associated_lock = converted
            delta_save_count = 0
            schema_version = _manager_workspace._current_workspace_schema_version()
        elif workspace_access is not None:
            associated_lock = workspace_access.take_lock()

        self._manager._set_workspace_path(
            associated_fname, workspace_lock=associated_lock
        )
        self._manager._workspace_state.delta_save_count = delta_save_count
        self._manager._workspace_state.schema_version = schema_version
        self._manager._workspace_state.needs_full_save = (
            _manager_workspace._workspace_schema_requires_full_save(schema_version)
        )
        if rebind_data:
            self._manager._rebind_workspace_backed_imagetools(associated_fname)
        self._manager._drain_workspace_deferred_events()
        self._manager._mark_workspace_clean()
        self._manager._record_recent_workspace(associated_fname)

    def _workspace_rebind_data_for_uid(
        self,
        fname: str | os.PathLike[str],
        uid: str,
        *,
        chunks: typing.Any,
    ) -> xr.DataArray:
        ds = _manager_xarray.open_workspace_dataset(
            fname, self._manager._workspace_payload_path(uid), chunks=chunks
        )
        try:
            data_name: Hashable
            if _ITOOL_DATA_NAME in ds.data_vars:
                data_name = _ITOOL_DATA_NAME
            else:
                data_name = next(iter(ds.data_vars))
            name = ds.attrs.get("itool_name", "")
            name = None if name == "" else name
            return ds[data_name].rename(name).copy(deep=False)
        finally:
            ds.close()

    def _workspace_data_backing_snapshot(
        self,
    ) -> dict[str, tuple[str, tuple[str, ...]]]:
        snapshot: dict[str, tuple[str, tuple[str, ...]]] = {}
        for node in self._manager._tool_graph.nodes.values():
            if not node.is_imagetool or node.imagetool is None:
                continue
            persistence = node.persistence_view()
            kind = typing.cast("str", persistence.data_backing)
            snapshot[node.uid] = (kind, persistence.source_paths)
        return snapshot

    def _rebind_workspace_backed_imagetools(
        self,
        fname: str | os.PathLike[str],
        *,
        targets: Iterable[int | str] | None = None,
        chunks: typing.Any = _WORKSPACE_REBIND_KEEP_CHUNKS,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
        old_workspace_path: str | os.PathLike[str] | None = None,
    ) -> None:
        pending: list[
            tuple[
                _ManagedWindowNode,
                typing.Any,
                str,
                typing.Any,
            ]
        ] = []
        if targets is None:
            nodes = [
                node
                for node in self._manager._tool_graph.nodes.values()
                if node.is_imagetool and node.imagetool is not None
            ]
        else:
            nodes = []
            for target in targets:
                node = self._manager._node_for_target(target)
                if node.is_imagetool and node.imagetool is not None:
                    nodes.append(node)
        for node in sorted(
            nodes, key=lambda node: self._manager._workspace_node_path(node.uid)
        ):
            tool = node.imagetool
            if tool is None:
                continue
            persistence = node.persistence_view()
            rebind_chunks: typing.Any
            if backing_snapshot is not None:
                backing = backing_snapshot.get(node.uid)
                if backing is None:
                    continue
                kind, source_paths = backing
                if kind == "memory":
                    continue
                if kind == "file_lazy":
                    old_path = _manager_xarray._normalized_file_path(old_workspace_path)
                    if old_path is None or old_path not in source_paths:
                        continue
                rebind_chunks = {} if kind == "dask" else None
            else:
                rebind_chunks = chunks
                if rebind_chunks is _WORKSPACE_REBIND_KEEP_CHUNKS:
                    rebind_chunks = {} if persistence.data_backing == "dask" else None
            pending.append(
                (
                    node,
                    copy.deepcopy(persistence.state),
                    node.name,
                    rebind_chunks,
                )
            )
        if not pending:
            return
        with self._manager._workspace_load_context():
            for node, state, name, chunks in pending:
                tool = node.imagetool
                if tool is None:
                    continue
                slicer_area = tool.slicer_area
                data = self._manager._workspace_rebind_data_for_uid(
                    fname, node.uid, chunks=chunks
                )
                slicer_area.set_data(data, auto_compute=False)
                slicer_area.state = state
                node._set_name(name, manual=False)

    def offload_to_workspace(
        self, targets: Iterable[int | str], *, native: bool = True
    ) -> bool:
        """Replace selected in-memory ImageTools with dask-backed workspace data.

        .. versionadded:: 3.23.0
        """
        offload_targets: list[int | str] = []
        for target in targets:
            node = self._manager._node_for_target(target)
            if (
                node.is_imagetool
                and node.imagetool is not None
                and not node.slicer_area.data_chunked
            ):
                offload_targets.append(target)
        if not offload_targets:
            return False

        if self._manager._workspace_state.path is None:
            if not self._manager.save_as(native=native):
                return False
        elif (
            self._manager.is_workspace_modified
            or self._manager._workspace_state.needs_full_save
        ) and not self._manager.save(native=native):
            return False

        if self._manager._workspace_state.path is None:
            return False

        origin = self._manager._active_managed_window()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self._manager, "Offloading to workspace..."
            ):
                self._manager._rebind_workspace_backed_imagetools(
                    self._manager._workspace_state.path,
                    targets=offload_targets,
                    chunks={},
                )
                _manager_workspace._write_workspace_root_attrs_to_file(
                    self._manager._workspace_state.path,
                    self._manager._workspace_root_attrs_payload(
                        delta_save_count=self._manager._workspace_state.delta_save_count
                    ),
                )
            self._manager._status_bar.showMessage("Data offloaded to workspace", 5000)
        except Exception:
            self._manager._show_operation_error(
                "Error while offloading to workspace",
                "An error occurred while reconnecting data from the workspace file.",
            )
            self._manager._restore_focus_after_workspace_save(origin)
            return False

        self._manager._restore_focus_after_workspace_save(origin)
        self._manager._update_actions()
        self._manager._update_info()
        return True

    def _workspace_requires_full_save(self, fname: str | os.PathLike[str]) -> bool:
        return _manager_workspace._workspace_requires_full_save(
            fname,
            needs_full_save=self._manager._workspace_state.needs_full_save,
            schema_version=self._manager._workspace_state.schema_version,
            structure_modified=self._manager._workspace_state.structure_modified,
            has_dirty_added=bool(self._manager._workspace_state.dirty_added),
            has_dirty_removed=bool(self._manager._workspace_state.dirty_removed),
        )

    def _workspace_has_non_layout_modifications(self) -> bool:
        state = self._manager._workspace_state
        return (
            state.structure_modified
            or bool(state.dirty_added)
            or bool(state.dirty_data)
            or bool(state.dirty_state)
            or bool(state.dirty_removed)
        )

    def _workspace_layout_only_modified(self) -> bool:
        return (
            self._manager._workspace_state.layout_modified
            and not self._workspace_has_non_layout_modifications()
        )

    def _workspace_rewrite_group_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, xr.Dataset]]:
        constructor: dict[str, xr.Dataset] = {}
        node = self._manager._tool_graph.nodes[uid]
        node_path = self._manager._workspace_node_path(uid)
        self._manager._serialize_workspace_node(
            constructor, node, node_path, include_children=True
        )
        return node_path, constructor

    def _workspace_attr_update_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]] | None:
        constructor: dict[str, xr.Dataset] = {}
        node = self._manager._tool_graph.nodes[uid]
        node_path = self._manager._workspace_node_path(uid)
        self._manager._serialize_workspace_node(
            constructor, node, node_path, include_children=False
        )
        payload_path = self._manager._workspace_payload_path(uid)
        ds = constructor.get(payload_path)
        if ds is None:
            return None
        return payload_path, dict(ds.attrs), (node_path, constructor)

    def _workspace_delta_save_snapshot(
        self,
        generation: int,
        root_attrs: dict[str, typing.Any],
        delta_save_count: int,
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        rewrite_groups: list[tuple[str, dict[str, xr.Dataset]]] = []
        rewritten_uids: set[str] = set()
        for uid in self._manager._workspace_highest_dirty_data_roots():
            rewrite_groups.append(self._manager._workspace_rewrite_group_snapshot(uid))
            rewritten_uids.add(uid)
            rewritten_uids.update(self._manager._iter_descendant_uids(uid))

        attr_updates: list[
            tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
        ] = []
        for uid in sorted(self._manager._workspace_state.dirty_state - rewritten_uids):
            if uid not in self._manager._tool_graph.nodes:
                continue
            update = self._manager._workspace_attr_update_snapshot(uid)
            if update is not None:
                attr_updates.append(update)

        return _manager_workspace._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=delta_save_count,
            rewrite_groups=tuple(rewrite_groups),
            attr_updates=tuple(attr_updates),
        )

    def _workspace_save_snapshot(
        self, fname: str | os.PathLike[str]
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        self._manager._drain_workspace_deferred_events()
        generation = self._manager._workspace_state.dirty_generation
        self._manager._workspace_state.saving_depth += 1
        try:
            if self._manager._workspace_requires_full_save(fname):
                return self._manager._workspace_full_save_snapshot(generation)
            if self._workspace_layout_only_modified():
                delta_save_count = self._manager._workspace_state.delta_save_count
                root_attrs = self._manager._workspace_root_attrs_payload(
                    delta_save_count=delta_save_count
                )
                return self._manager._workspace_delta_save_snapshot(
                    generation, root_attrs, delta_save_count
                )
            delta_save_count = self._manager._workspace_state.delta_save_count + 1
            root_attrs = self._manager._workspace_root_attrs_payload(
                delta_save_count=delta_save_count
            )
            return self._manager._workspace_delta_save_snapshot(
                generation, root_attrs, delta_save_count
            )
        finally:
            self._manager._workspace_state.saving_depth -= 1

    def _workspace_full_save_snapshot(
        self, generation: int
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        tree = self._manager._to_datatree()
        copy_source, copy_groups = self._manager._workspace_full_save_copy_groups(tree)
        return _manager_workspace._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=self._manager._workspace_root_attrs_payload(delta_save_count=0),
            delta_save_count=0,
            full_tree=tree,
            copy_source=copy_source,
            copy_groups=copy_groups,
        )

    def _workspace_full_save_copy_groups(
        self, tree: xr.DataTree
    ) -> tuple[str | None, tuple[tuple[str, str, dict[str, typing.Any] | None], ...]]:
        if self._manager._workspace_state.path is None:
            return None, ()
        workspace_path = pathlib.Path(self._manager._workspace_state.path)
        if (
            self._manager._workspace_state.schema_version
            != _manager_workspace._current_workspace_schema_version()
            or not workspace_path.exists()
        ):
            return None, ()

        try:
            root_attrs = _manager_workspace._read_workspace_root_attrs_h5py(
                workspace_path
            )
        except Exception:
            return None, ()
        schema_version, _delta_save_count, manifest = (
            _manager_workspace._workspace_file_metadata_from_attrs(root_attrs)
        )
        if (
            schema_version != _manager_workspace._current_workspace_schema_version()
            or manifest is None
        ):
            return None, ()

        identities: dict[tuple[str, str], str] = {}
        nodes = manifest.get("nodes", ())
        if isinstance(nodes, list):
            for entry in nodes:
                if not isinstance(entry, dict):
                    continue
                uid = entry.get("uid")
                kind = entry.get("kind")
                path = entry.get("path")
                if (
                    isinstance(uid, str)
                    and isinstance(kind, str)
                    and kind in {"imagetool", "tool"}
                    and isinstance(path, str)
                ):
                    identities[(uid, kind)] = f"{path}/{kind}"
        copy_groups: list[tuple[str, str, dict[str, typing.Any] | None]] = []
        for uid, node in self._manager._tool_graph.nodes.items():
            if (
                uid in self._manager._workspace_state.dirty_data
                or uid in self._manager._workspace_state.dirty_added
            ):
                continue
            if not node.is_imagetool:
                tool = node.tool_window
                if tool is None or not tool.can_save_and_load():
                    continue
            kind = "imagetool" if node.is_imagetool else "tool"
            source_path = identities.get((uid, kind))
            if source_path is None:
                continue
            payload_path = self._manager._workspace_payload_path(uid)
            try:
                payload_tree = typing.cast("xr.DataTree", tree[payload_path])
            except KeyError:
                continue
            attrs = None
            if (
                uid in self._manager._workspace_state.dirty_state
                or source_path != payload_path
            ):
                attrs = dict(payload_tree.to_dataset(inherit=False).attrs)
            copy_groups.append((source_path, payload_path, attrs))
        return str(workspace_path), tuple(copy_groups)

    def _open_workspace_save_wait_dialog(
        self,
        parent: QtWidgets.QWidget,
        *,
        title: str = "Saving Workspace",
        label_text: str = "Saving workspace...",
    ) -> QtWidgets.QDialog:
        dialog = QtWidgets.QProgressDialog(label_text, "", 0, 0, parent)
        dialog.setCancelButton(None)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        return dialog

    def _set_workspace_save_actions_enabled(
        self, enabled: bool
    ) -> tuple[bool, bool, bool]:
        previous = (
            self._manager.save_action.isEnabled(),
            self._manager.save_as_action.isEnabled(),
            self._manager.compact_workspace_action.isEnabled(),
        )
        self._manager.save_action.setEnabled(enabled and previous[0])
        self._manager.save_as_action.setEnabled(enabled and previous[1])
        self._manager.compact_workspace_action.setEnabled(enabled and previous[2])
        return previous

    def _restore_workspace_save_actions_enabled(
        self, previous: tuple[bool, bool, bool]
    ) -> None:
        self._manager.save_action.setEnabled(previous[0])
        self._manager.save_as_action.setEnabled(previous[1])
        self._manager.compact_workspace_action.setEnabled(previous[2])

    def _run_workspace_save_worker(
        self,
        fname: str | os.PathLike[str],
        snapshot: _manager_workspace._WorkspaceSaveSnapshot,
        origin: QtWidgets.QWidget | None,
        *,
        wait_dialog_title: str = "Saving Workspace",
        wait_dialog_text: str = "Saving workspace...",
    ) -> tuple[bool, float, str]:
        loop = QtCore.QEventLoop(self._manager)
        result: dict[str, typing.Any] = {"ok": False, "elapsed": 0.0, "error": ""}
        receiver = _manager_workspace._WorkspaceSaveResultReceiver(
            loop, result, self._manager
        )
        worker = _manager_workspace._WorkspaceSaveWorker(fname, snapshot)
        wait_dialog: QtWidgets.QDialog | None = None
        wait_timer = QtCore.QTimer(self._manager)
        wait_timer.setSingleShot(True)

        def _show_wait_dialog() -> None:
            nonlocal wait_dialog
            if wait_dialog is None and self._manager._workspace_state.save_in_progress:
                wait_dialog = self._manager._open_workspace_save_wait_dialog(
                    origin or self._manager,
                    title=wait_dialog_title,
                    label_text=wait_dialog_text,
                )

        wait_timer.timeout.connect(_show_wait_dialog)
        worker.signals.finished.connect(receiver.finish)
        self._manager._workspace_state.save_in_progress = True
        previous_action_states = self._manager._set_workspace_save_actions_enabled(
            False
        )
        try:
            wait_timer.start(
                max(0, int(_WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS * 1000))
            )
            thread_pool = QtCore.QThreadPool.globalInstance()
            if thread_pool is None:
                raise RuntimeError("Qt thread pool is unavailable")
            thread_pool.start(worker)
            loop.exec()
        finally:
            self._manager._workspace_state.save_in_progress = False
            wait_timer.stop()
            wait_timer.deleteLater()
            if wait_dialog is not None:
                wait_dialog.close()
                wait_dialog.deleteLater()
            receiver.deleteLater()
            self._manager._restore_workspace_save_actions_enabled(
                previous_action_states
            )
        return (
            typing.cast("bool", result["ok"]),
            typing.cast("float", result["elapsed"]),
            typing.cast("str", result["error"]),
        )

    def save(self, *, native: bool = True) -> bool:
        """Save the current workspace document.

        Parameters
        ----------
        native
            Whether to use the native file dialog, by default `True`. This option is
            used when testing the application to ensure reproducibility.
        """
        if self._manager._workspace_state.path is None:
            return self._manager.save_as(native=native)
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save already in progress", 3000
            )
            return False
        origin = self._manager._active_managed_window()
        old_workspace_path = self._manager._workspace_state.path
        backing_snapshot = self._manager._workspace_data_backing_snapshot()
        self._manager._status_bar.showMessage("Saving workspace...")
        try:
            snapshot = self._manager._workspace_save_snapshot(
                self._manager._workspace_state.path
            )
            try:
                ok, elapsed, error_text = self._manager._run_workspace_save_worker(
                    self._manager._workspace_state.path, snapshot, origin
                )
            except Exception:
                snapshot.close()
                raise
        except Exception:
            self._manager._status_bar.clearMessage()
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            self._manager._restore_focus_after_workspace_save(origin)
            return False
        if not ok:
            self._manager._status_bar.clearMessage()
            self._manager._show_workspace_save_worker_error(error_text)
            self._manager._restore_focus_after_workspace_save(origin)
            return False

        self._manager._workspace_state.needs_full_save = False
        self._manager._workspace_state.delta_save_count = snapshot.delta_save_count
        if snapshot.full_tree is not None:
            self._manager._workspace_state.schema_version = (
                _manager_workspace._current_workspace_schema_version()
            )
            self._manager._rebind_workspace_backed_imagetools(
                self._manager._workspace_state.path,
                backing_snapshot=backing_snapshot,
                old_workspace_path=old_workspace_path,
            )
        self._manager._drain_workspace_deferred_events()
        post_save_events = tuple(
            event
            for event in self._manager._workspace_state.dirty_events
            if event.generation > snapshot.generation
        )
        if post_save_events:
            self._manager._restore_workspace_dirty_events(post_save_events)
            message = "Workspace saved; new changes remain unsaved"
        else:
            self._manager._mark_workspace_clean()
            message = (
                f"Workspace saved in {elapsed:.1f} s"
                if elapsed >= _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS
                else "Workspace saved"
            )
        self._manager._status_bar.showMessage(message, 5000)
        self._manager._restore_focus_after_workspace_save(origin)
        self._manager._record_recent_workspace(self._manager._workspace_state.path)
        return True

    def save_as(self, *, native: bool = True) -> bool:
        """Save the current workspace under a new path and bind to that path."""
        origin = self._manager._active_managed_window()
        fname = self._manager._workspace_save_dialog(
            native=native, caption="Save Workspace As"
        )
        if fname is None:
            return False
        old_workspace_path = self._manager._workspace_state.path
        backing_snapshot = self._manager._workspace_data_backing_snapshot()
        try:
            dialog_parent = origin or self._manager
            with self._manager._workspace_document_access_context(fname) as access:
                with erlab.interactive.utils.wait_dialog(
                    dialog_parent, "Saving workspace..."
                ):
                    self._manager._save_workspace_document(
                        access.path,
                        force_full=True,
                        document_access=access,
                    )
                    self._manager._rebind_workspace_backed_imagetools(
                        access.path,
                        backing_snapshot=backing_snapshot,
                        old_workspace_path=old_workspace_path,
                    )
                self._manager._set_workspace_path(
                    access.path, workspace_lock=access.take_lock()
                )
            self._manager._workspace_state.needs_full_save = False
            self._manager._drain_workspace_deferred_events()
            self._manager._mark_workspace_clean()
            self._manager._record_recent_workspace(access.path)
        except Exception:
            self._manager._show_operation_error(
                "Error while saving workspace",
                "An error occurred while saving the workspace file.",
            )
            return False
        self._manager._restore_focus_after_workspace_save(origin)
        return True

    def _compact_workspace_before_shutdown(self) -> None:
        if (
            self._manager._workspace_state.path is None
            or self._manager._workspace_state.delta_save_count <= 0
            or self._manager.is_workspace_modified
            or self._manager._workspace_state.save_in_progress
            or self._manager._workspace_state.loading_depth > 0
        ):
            return
        try:
            logger.debug("Compacting workspace before shutdown...")
            self._manager._drain_workspace_deferred_events()
            generation = self._manager._workspace_state.dirty_generation
            self._manager._workspace_state.saving_depth += 1
            try:
                snapshot = self._manager._workspace_full_save_snapshot(generation)
            finally:
                self._manager._workspace_state.saving_depth -= 1
            try:
                ok, _, error_text = self._manager._run_workspace_save_worker(
                    self._manager._workspace_state.path,
                    snapshot,
                    self._manager,
                    wait_dialog_title="Optimizing Workspace",
                    wait_dialog_text="Optimizing workspace file…",
                )
            except Exception:
                snapshot.close()
                raise
            if not ok:
                logger.error(
                    "Failed to compact workspace before shutdown%s",
                    f":\n{error_text}" if error_text else "",
                    extra={"suppress_ui_alert": True},
                )
                return
            self._manager._workspace_state.needs_full_save = False
            self._manager._workspace_state.delta_save_count = 0
            self._manager._workspace_state.schema_version = (
                _manager_workspace._current_workspace_schema_version()
            )
            self._manager._drain_workspace_deferred_events()
            post_save_events = tuple(
                event
                for event in self._manager._workspace_state.dirty_events
                if event.generation > snapshot.generation
            )
            if post_save_events:
                self._manager._restore_workspace_dirty_events(post_save_events)
            else:
                self._manager._mark_workspace_clean()
        except Exception:
            logger.exception(
                "Failed to compact workspace before shutdown",
                extra={"suppress_ui_alert": True},
            )

    def compact_workspace(self) -> bool:
        """Rewrite the current workspace file to remove unused space."""
        if self._manager._workspace_state.path is None:
            return self._manager.save_as()
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save already in progress", 3000
            )
            return False

        origin = self._manager._active_managed_window()
        old_workspace_path = self._manager._workspace_state.path
        backing_snapshot = self._manager._workspace_data_backing_snapshot()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self._manager, "Compacting workspace..."
            ):
                self._manager._save_workspace_document(
                    self._manager._workspace_state.path, force_full=True
                )
                self._manager._rebind_workspace_backed_imagetools(
                    self._manager._workspace_state.path,
                    backing_snapshot=backing_snapshot,
                    old_workspace_path=old_workspace_path,
                )
            self._manager._workspace_state.delta_save_count = 0
            self._manager._status_bar.showMessage("Workspace compacted", 5000)
        except Exception:
            self._manager._show_operation_error(
                "Error while compacting workspace",
                "An error occurred while compacting the workspace file.",
            )
            self._manager._restore_focus_after_workspace_save(origin)
            return False

        self._manager._restore_focus_after_workspace_save(origin)
        return True

    def _save_to_file(self, fname: str):
        """Export a selected subset of the workspace to ``fname``.

        This helper preserves the older selection-dialog behavior used by tests and
        private callers. Document-style Save and Save As use
        :meth:`_save_workspace_document` instead.
        """
        tree: xr.DataTree = self._manager._to_datatree()
        try:
            dialog = _ChooseFromDataTreeDialog(self._manager, tree, mode="save")
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return

            def _prune(node: xr.DataTree, item: QtWidgets.QTreeWidgetItem) -> None:
                if "childtools" not in node:
                    return
                child_tree = typing.cast("xr.DataTree", node["childtools"])
                for i in reversed(range(item.childCount())):
                    child_item = typing.cast("QtWidgets.QTreeWidgetItem", item.child(i))
                    child_key = str(child_item.data(0, QtCore.Qt.ItemDataRole.UserRole))
                    if child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked:
                        del child_tree[child_key]
                        continue
                    _prune(
                        typing.cast("xr.DataTree", child_tree[child_key]), child_item
                    )
                if len(child_tree) == 0:
                    del node["childtools"]

            root_item = dialog._tree_widget.invisibleRootItem()
            if root_item is None:
                return
            for i in reversed(range(root_item.childCount())):
                item = typing.cast("QtWidgets.QTreeWidgetItem", root_item.child(i))
                key = str(item.data(0, QtCore.Qt.ItemDataRole.UserRole))
                if item.checkState(0) == QtCore.Qt.CheckState.Unchecked:
                    del tree[key]
                    continue
                _prune(typing.cast("xr.DataTree", tree[key]), item)
            with erlab.interactive.utils.wait_dialog(
                self._manager, "Saving workspace..."
            ):
                tree.to_netcdf(
                    fname,
                    engine="h5netcdf",
                    invalid_netcdf=True,
                    encoding=_manager_xarray.workspace_datatree_encoding(tree),
                )
        finally:
            tree.close()

    def _load_workspace_file(
        self,
        fname: str | os.PathLike[str],
        *,
        replace: bool,
        associate: bool,
        mark_dirty: bool,
        select: bool,
        native: bool = True,
    ) -> bool:
        previous_missing_colormaps = self._missing_workspace_colormaps
        self._missing_workspace_colormaps = []
        try:
            with self._manager._workspace_document_access_context(fname) as access:
                _manager_workspace._recover_workspace_transactions(access.path)
                if not select and replace:
                    try:
                        root_attrs = _manager_workspace._read_workspace_root_attrs_h5py(
                            access.path
                        )
                        schema_version, delta_save_count, manifest = (
                            _manager_workspace._workspace_file_metadata_from_attrs(
                                root_attrs
                            )
                        )
                        if (
                            schema_version
                            == _manager_workspace._current_workspace_schema_version()
                            and manifest is not None
                        ):
                            loaded = self._manager._from_h5py_workspace_file(
                                access.path,
                                manifest,
                                replace=replace,
                                mark_dirty=mark_dirty,
                            )
                            if loaded and associate:
                                self._manager._associate_loaded_workspace_file(
                                    access.path,
                                    schema_version,
                                    native=native,
                                    delta_save_count=delta_save_count,
                                    workspace_access=access,
                                    rebind_data=False,
                                )
                                if replace:
                                    self._restore_workspace_loader_state(
                                        manifest, apply_explorer=False
                                    )
                            return self._finish_workspace_file_load(loaded)
                    except Exception:
                        logger.debug(
                            "Failed h5py workspace load path; falling back to DataTree",
                            exc_info=True,
                        )
                tree = _manager_xarray.open_workspace_datatree(access.path, chunks=None)
                schema_version, delta_save_count, manifest = (
                    _manager_workspace._workspace_file_metadata_from_attrs(tree.attrs)
                )
                loaded = self._manager._from_datatree(
                    tree,
                    replace=replace,
                    mark_dirty=mark_dirty,
                    select=select,
                    workspace_file_path=access.path,
                )
                if loaded and associate:
                    self._manager._associate_loaded_workspace_file(
                        access.path,
                        schema_version,
                        native=native,
                        delta_save_count=delta_save_count,
                        workspace_access=access,
                        rebind_data=False,
                    )
                    if replace:
                        self._restore_workspace_loader_state(
                            manifest, apply_explorer=False
                        )
                return self._finish_workspace_file_load(loaded)
        finally:
            self._missing_workspace_colormaps = previous_missing_colormaps

    def load(self, *, native: bool = True) -> bool:
        """Replace this manager with a workspace file."""
        dialog = QtWidgets.QFileDialog(self._manager)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(
            ["ImageTool Workspace Files (*.itws)", "xarray HDF5 Files (*.h5)"]
        )
        if self._manager._recent_directory is not None:
            dialog.setDirectory(self._manager._recent_directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return False

        fname = dialog.selectedFiles()[0]
        if not self._manager._confirm_save_dirty_workspace(
            "Opening a workspace replaces the windows currently in this manager."
        ):
            return False
        self._manager._recent_directory = os.path.dirname(fname)
        try:
            return self._manager._load_workspace_file(
                fname,
                replace=True,
                associate=True,
                mark_dirty=False,
                select=False,
                native=native,
            )
        except Exception as exc:
            if _manager_workspace._is_workspace_file_lock_error(exc):
                logger.info(
                    "Workspace file is already open or locked: %s",
                    fname,
                    extra={"suppress_ui_alert": True},
                )
                _show_workspace_file_lock_error(self._manager, fname)
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

    def import_workspace(self, *, native: bool = True) -> bool:
        """Import selected windows from another workspace file."""
        dialog = QtWidgets.QFileDialog(self._manager)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(
            ["ImageTool Workspace Files (*.itws)", "xarray HDF5 Files (*.h5)"]
        )
        if self._manager._recent_directory is not None:
            dialog.setDirectory(self._manager._recent_directory)
        if not native:  # pragma: no branch
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if not dialog.exec():
            return False
        fname = dialog.selectedFiles()[0]
        self._manager._recent_directory = os.path.dirname(fname)
        try:
            loaded = self._manager._load_workspace_file(
                fname,
                replace=False,
                associate=False,
                mark_dirty=True,
                select=True,
            )
        except Exception as exc:
            if _manager_workspace._is_workspace_file_lock_error(exc):
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
                self._manager._record_recent_workspace(fname)
            return loaded

    def open(self, *, native: bool = True) -> None:
        """Open files in a new ImageTool window.

        Parameters
        ----------
        native
            Whether to use the native file dialog, by default `True`. This option is
            used when testing the application to ensure reproducibility.
        """
        dialog = QtWidgets.QFileDialog(self._manager)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        valid_loaders: dict[str, tuple[Callable, dict]] = (
            erlab.interactive.utils.file_loaders()
        )
        dialog.setNameFilters(valid_loaders.keys())
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        preferred_name_filter = self._manager._preferred_name_filter(valid_loaders)
        if preferred_name_filter is not None:
            dialog.selectNameFilter(preferred_name_filter)
        if self._manager._recent_directory is not None:
            dialog.setDirectory(self._manager._recent_directory)

        if dialog.exec():
            file_names = dialog.selectedFiles()
            self._manager._recent_name_filter = dialog.selectedNameFilter()
            self._manager._recent_directory = os.path.dirname(file_names[0])
            func, kwargs = valid_loaders[self._manager._recent_name_filter]
            if _is_loader_func(func):
                selected = self._manager._select_loader_options(
                    {self._manager._recent_name_filter: (func, kwargs)},
                    self._manager._recent_name_filter,
                    sample_paths=file_names,
                )
                if selected is None:
                    return
                self._manager._recent_name_filter, func, kwargs = selected
            self._manager._add_from_multiple_files(
                loaded=[],
                queued=[pathlib.Path(f) for f in file_names],
                failed=[],
                func=func,
                kwargs=kwargs,
                retry_callback=lambda _: self._manager.open(native=native),
            )

    def _data_recv(
        self,
        data: list[xr.DataArray] | list[xr.Dataset],
        kwargs: dict[str, typing.Any],
        *,
        watched_var: tuple[str, str] | None = None,
        watched_metadata: Mapping[str, typing.Any] | None = None,
        show: bool | None = None,
    ) -> list[bool]:
        """Slot function to receive data from the server.

        DataArrays passed to this function are displayed in new ImageTool windows which
        are added to the manager.

        Parameters
        ----------
        data
            A list of xarray.DataArray objects representing the data.

            Also accepts a list of xarray.Dataset objects created with
            ``ImageTool.to_dataset()``, in which case all other parameters are ignored.
        kwargs
            Additional keyword arguments to be passed to the ImageTool.
        watched_var
            If the tool is created from a watched variable, this should be a tuple of
            the variable name and its unique ID.
        show
            Whether to show the created windows. By default, only show if `data`
            contains only one DataArray.

        Returns
        -------
        flags : list of bool
            List of flags indicating whether the data was successfully received.
        """
        flags: list[bool] = []
        if erlab.utils.misc.is_sequence_of(data, xr.Dataset):
            for ds in data:
                try:
                    self._manager.add_imagetool(
                        ImageTool.from_dataset(ds, _in_manager=True), activate=True
                    )
                except Exception:
                    flags.append(False)
                    logger.exception(
                        "Error creating ImageTool window",
                        extra={"suppress_ui_alert": True},
                    )
                    self._manager._error_creating_imagetool()
                else:
                    flags.append(True)
            return flags

        link = kwargs.pop("link", False)
        link_colors = kwargs.pop("link_colors", True)
        indices: list[int] = []
        kwargs["_in_manager"] = True

        load_func = kwargs.pop("load_func", None)
        load_indices = kwargs.pop("load_indices", None)
        if show is None:
            show = len(data) == 1
        watched_metadata = dict(watched_metadata or {})
        if watched_var is not None:
            watched_metadata.setdefault(
                "workspace_link_id", self._manager._workspace_state.link_id
            )

        for i, d in enumerate(data):
            # Set selection-specific load function if provided
            load_selection = (
                typing.cast("Sequence[typing.Any]", load_indices)[i]
                if load_indices is not None
                else i
            )
            this_load_func = (*load_func[:2], load_selection) if load_func else None
            try:
                indices.append(
                    self._manager.add_imagetool(
                        ImageTool(d, **kwargs, load_func=this_load_func),
                        show=show,
                        activate=show,
                        watched_var=watched_var,
                        watched_workspace_link_id=typing.cast(
                            "str | None",
                            watched_metadata.get("workspace_link_id"),
                        ),
                        watched_source_label=typing.cast(
                            "str | None", watched_metadata.get("source_label")
                        ),
                        watched_source_uid=typing.cast(
                            "str | None", watched_metadata.get("source_uid")
                        ),
                        watched_connected=bool(
                            watched_metadata.get("connected", watched_var is not None)
                        ),
                        source_input_ndim=d.ndim,
                        source_input_dtype=d.dtype,
                    )
                )
                if watched_var is not None:
                    # Refresh title to include variable name
                    self._manager.get_imagetool(indices[-1])._update_title()
            except Exception:
                flags.append(False)
                logger.exception(
                    "Error creating ImageTool window",
                    extra={"suppress_ui_alert": True},
                )
                self._manager._error_creating_imagetool()
            else:
                flags.append(True)

        if link:
            self._manager.link_imagetools(*indices, link_colors=link_colors)

        return flags
