"""Transactional persistence and save workers for manager workspaces."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import time
import traceback
import typing
import uuid
from dataclasses import dataclass

import xarray as xr
from qtpy import QtCore

import erlab
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.manager._workspace._store as workspace_store
from erlab.interactive import _qt_state
from erlab.interactive.imagetool._load_source import _serialize_loader_kwargs
from erlab.interactive.imagetool.manager._widgets import (
    _strip_workspace_modified_placeholder,
)
from erlab.interactive.imagetool.manager._wrapper import (
    _ImageToolWrapper,
    _ManagedWindowNode,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from erlab.interactive._options.schema import WorkspaceCompressionMode
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._widgets import _WorkspaceDocumentAccess
    from erlab.interactive.imagetool.manager._workspace._controller import (
        _WorkspaceController,
    )
from erlab.interactive.imagetool.manager._workspace._format import (
    _require_itws_workspace_path,
)

logger = logging.getLogger(__name__)
_WORKSPACE_SAVE_SUFFIX_ERROR = "ImageTool workspace documents must be saved as .itws"


@dataclass
class _WorkspaceSaveSnapshot:
    generation: int
    generation_plan: workspace_storage._WorkspaceGenerationPlan
    compression_mode: WorkspaceCompressionMode
    serialized_tool_data_references: tuple[
        tuple[str, str, dict[str, dict[str, typing.Any]]], ...
    ] = ()

    def close(self) -> None:
        closed: set[int] = set()
        for item in self.generation_plan.objects:
            if item.dataset is None or id(item.dataset) in closed:
                continue
            closed.add(id(item.dataset))
            item.dataset.close()


@dataclass(frozen=True)
class _WorkspaceSaveError:
    traceback_text: str
    missing_source_path: str | None = None
    publication_conflict_path: str | None = None
    access_denied_path: str | None = None


class _WorkspaceSaveWorkerSignals(QtCore.QObject):
    waiting = QtCore.Signal()
    finished = QtCore.Signal(float, object)


class _WorkspaceSaveResultReceiver(QtCore.QObject):
    def __init__(
        self,
        *,
        callback: Callable[[float, _WorkspaceSaveError | None], None] | None = None,
        waiting_callback: Callable[[], None] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._callback = callback
        self._waiting_callback = waiting_callback

    @QtCore.Slot()
    def wait(self) -> None:
        if self._waiting_callback is not None:
            self._waiting_callback()

    @QtCore.Slot(float, object)
    def finish(
        self,
        elapsed: float,
        error: _WorkspaceSaveError | None,
    ) -> None:
        if self._callback is not None:
            self._callback(elapsed, error)


class _WorkspaceSaveWorker(QtCore.QRunnable):
    def __init__(
        self,
        fname: str | os.PathLike[str],
        snapshot: _WorkspaceSaveSnapshot,
        *,
        store: workspace_store.WorkspaceStore | None = None,
        reader_closers: tuple[Callable[[], None], ...] = (),
    ) -> None:
        super().__init__()
        self.signals = _WorkspaceSaveWorkerSignals()
        self._fname = fname
        self._snapshot = snapshot
        self._store = store
        self._reader_closers = reader_closers
        self._waiting_reported = False

    def _handle_contention(self) -> None:
        for closer in self._reader_closers:
            closer()
        if not self._waiting_reported:
            self._waiting_reported = True
            self.signals.waiting.emit()

    def run(self) -> None:
        start_time = time.perf_counter()
        error: _WorkspaceSaveError | None = None
        try:
            owned_store = False
            target_store = self._store
            if target_store is None:
                target_store = workspace_store.WorkspaceStore.active(self._fname)
            if target_store is None:
                target_store = workspace_store.WorkspaceStore(
                    self._fname,
                    create=not pathlib.Path(self._fname).exists(),
                )
                owned_store = True
            try:
                workspace_storage._write_workspace_generation(
                    target_store,
                    self._snapshot.generation_plan,
                    compression_mode=self._snapshot.compression_mode,
                    on_contention=self._handle_contention,
                )
            finally:
                if owned_store:
                    target_store.close()
        except workspace_storage._WorkspaceBackingFileNotFoundError as exc:
            error = _WorkspaceSaveError(
                traceback_text=traceback.format_exc(),
                missing_source_path=exc.source_path,
            )
        except workspace_storage._WorkspacePublicationConflictError as exc:
            error = _WorkspaceSaveError(
                traceback_text=traceback.format_exc(),
                publication_conflict_path=exc.path,
            )
        except workspace_store.WorkspaceStoreConflictError:
            error = _WorkspaceSaveError(
                traceback_text=traceback.format_exc(),
                publication_conflict_path=os.fsdecode(self._fname),
            )
        except PermissionError as exc:
            error = _WorkspaceSaveError(
                traceback_text=traceback.format_exc(),
                access_denied_path=os.fsdecode(exc.filename or self._fname),
            )
        except Exception:
            error = _WorkspaceSaveError(traceback_text=traceback.format_exc())
        finally:
            with contextlib.suppress(Exception):
                self._snapshot.close()
        self.signals.finished.emit(time.perf_counter() - start_time, error)


class _WorkspaceGcWorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(bool, object)


class _WorkspaceGcResultReceiver(QtCore.QObject):
    def __init__(
        self,
        callback: Callable[[bool, str | None], None],
        *,
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self._callback = callback

    @QtCore.Slot(bool, object)
    def finish(self, more: bool, error: str | None) -> None:
        self._callback(more, error)


class _WorkspaceGcWorker(QtCore.QRunnable):
    """Unlink at most one obsolete payload object outside the save path."""

    def __init__(
        self,
        store: workspace_store.WorkspaceStore,
        *,
        reader_closers: tuple[Callable[[], None], ...] = (),
    ) -> None:
        super().__init__()
        self.signals = _WorkspaceGcWorkerSignals()
        self._store = store
        self._reader_closers = reader_closers

    def _handle_contention(self) -> None:
        for closer in self._reader_closers:
            closer()

    def run(self) -> None:
        more = False
        error: str | None = None
        try:
            more = self._store.collect_garbage(
                max_objects=1,
                on_contention=self._handle_contention,
            )
        except Exception:
            error = traceback.format_exc()
        self.signals.finished.emit(more, error)


class _WorkspaceSaver:
    """Serialize and snapshot workspace state for one manager."""

    def __init__(
        self, manager: ImageToolManager, controller: _WorkspaceController
    ) -> None:
        self._manager = manager
        self._controller = controller

    def _workspace_reader_closers(
        self, fname: str | os.PathLike[str]
    ) -> tuple[Callable[[], None], ...]:
        """Return reusable closers for displayed data from one workspace."""
        target = workspace_arrays._normalized_file_path(fname)
        if target is None:
            return ()
        closers: list[Callable[[], None]] = []
        seen: set[int] = set()
        for node in self._manager._tool_graph.nodes.values():
            if node.imagetool is None:
                continue
            if target not in workspace_arrays.dataarray_source_paths(
                node.slicer_area._data
            ):
                continue
            closer = node.slicer_area._data_resource_close_callback()
            if closer is None or id(closer) in seen:
                continue
            seen.add(id(closer))
            closers.append(closer)
        return tuple(closers)

    def _close_workspace_idle_readers(self, fname: str | os.PathLike[str]) -> None:
        for closer in self._workspace_reader_closers(fname):
            closer()

    @staticmethod
    def _serialized_tool_data_references(
        datasets: Iterable[xr.Dataset],
    ) -> tuple[tuple[str, str, dict[str, dict[str, typing.Any]]], ...]:
        references_by_uid: dict[str, tuple[str, dict[str, dict[str, typing.Any]]]] = {}
        for ds in datasets:
            if ds.attrs.get("manager_node_kind") != "tool":
                continue
            uid = ds.attrs.get("manager_node_uid")
            snapshot_token = ds.attrs.get("manager_node_snapshot_token")
            if (
                not isinstance(uid, str)
                or not uid
                or not isinstance(snapshot_token, str)
                or not snapshot_token
            ):
                continue
            references_by_uid[uid] = (
                snapshot_token,
                erlab.interactive.utils.ToolWindow._saved_tool_data_references(ds),
            )
        return tuple(
            (uid, snapshot_token, references)
            for uid, (snapshot_token, references) in sorted(references_by_uid.items())
        )

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
        ds.attrs["manager_node_source_snapshot_token"] = node.source_snapshot_token
        ds.attrs["manager_node_added_at"] = node.added_time_iso
        if node.note:
            ds.attrs["manager_node_note"] = node.note
        else:
            ds.attrs.pop("manager_node_note", None)
        persistence = node.persistence_view()
        provenance_spec = persistence.provenance_spec
        if kind == "imagetool" and persistence.replay_source_data is not None:
            ds = ds.copy(deep=False)
            blob_name = workspace_format._WORKSPACE_REPLAY_SOURCE_BLOB_NAME
            ds[blob_name] = erlab.interactive.utils._tool_data_to_blob(
                persistence.replay_source_data,
                blob_name,
            )
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
        if kind == "imagetool" and (source_spec is not None or output_id is not None):
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
            tool = self._manager.get_imagetool(target)
            ds = tool.to_dataset()
            ds.attrs["itool_title"] = node.name
            constructor[f"{path}/imagetool"] = self._annotate_workspace_dataset(
                ds,
                node,
                kind="imagetool",
            )
        else:
            if (
                node.pending_workspace_payload is not None
                and not node.materialize_pending_workspace_payload()
            ):
                message = "Could not read this saved tool from the workspace"
                if node.display_text:
                    message += f": {node.display_text!r}"
                message += f" ({node.uid})"
                unavailable_uids = (
                    self._pending_workspace_tool_unavailable_reference_uids(node)
                )
                if unavailable_uids:
                    message += ". Unavailable manager-node references: " + ", ".join(
                        unavailable_uids
                    )
                raise ValueError(f"{message}.")
            tool = typing.cast("erlab.interactive.utils.ToolWindow", node.tool_window)
            if not tool.can_save_and_load():
                return
            with tool._save_tool_data_reference_context(
                self._manager._tool_graph.nodes,
                reference_validator=(
                    self._controller._tool_data_reference_matches_current_data
                ),
            ):
                ds = tool.to_dataset()
            ds.attrs["tool_title"] = _strip_workspace_modified_placeholder(
                ds.attrs.get("tool_title", "")
            )
            constructor[f"{path}/tool"] = self._annotate_workspace_dataset(
                ds, node, kind="tool"
            )

        if not include_children:
            return
        for child_uid in node._childtool_indices:
            child = self._manager._child_node(child_uid)
            self._serialize_workspace_node(
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
        for index in self._workspace_root_indices():
            self._serialize_workspace_node(
                constructor,
                self._manager._tool_graph.root_wrappers[index],
                str(index),
                include_children=include_children,
            )
            if close:
                self._manager.remove_imagetool(index)
        for uid in list(self._manager._tool_graph.figure_uids):
            node = self._manager._tool_graph.nodes.get(uid)
            if not isinstance(node, _ManagedWindowNode):
                continue
            self._serialize_workspace_node(
                constructor,
                node,
                f"figures/{uid}",
                include_children=False,
            )
            if close:
                self._manager._remove_childtool(uid)
        tree = xr.DataTree.from_dict(constructor)
        workspace_format._set_legacy_workspace_schema(tree.attrs)
        return tree

    def _workspace_node_path(self, uid: str) -> str:
        node = self._manager._tool_graph.nodes[uid]
        if isinstance(node, _ImageToolWrapper):
            return str(node.index)
        if self._manager._is_figure_node(node):
            return f"figures/{uid}"
        if node.parent_uid is None:
            raise KeyError(f"Node {uid!r} has no parent")
        return f"{self._workspace_node_path(node.parent_uid)}/childtools/{uid}"

    def _workspace_payload_path(self, uid: str) -> str:
        node = self._manager._tool_graph.nodes[uid]
        payload_name = "imagetool" if node.is_imagetool else "tool"
        return f"{self._workspace_node_path(uid)}/{payload_name}"

    def _workspace_root_indices(self) -> tuple[int, ...]:
        return self._manager._tool_graph.root_indices_for_workspace()

    def _workspace_link_metadata_by_uid(self) -> dict[str, tuple[int, bool]]:
        metadata: dict[str, tuple[int, bool]] = {}
        group_index = 0
        structural_groups: dict[str, tuple[list[str], bool]] = {}
        for uid, node in self._manager._tool_graph.nodes.items():
            link_key = node.workspace_link_key
            if link_key is None:
                continue
            group_nodes, _link_colors = structural_groups.setdefault(
                link_key, ([], node.workspace_link_colors)
            )
            group_nodes.append(uid)
        for group_nodes, link_colors in structural_groups.values():
            if len(group_nodes) <= 1:
                continue
            for uid in group_nodes:
                metadata[uid] = (group_index, link_colors)
            group_index += 1
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
                if node.uid in metadata:
                    continue
                metadata[node.uid] = (group_index, bool(linker.link_colors))
            group_index += 1
        return metadata

    def _workspace_node_manifest_entries(self) -> list[dict[str, typing.Any]]:
        entries: list[dict[str, typing.Any]] = []
        link_metadata = self._workspace_link_metadata_by_uid()

        def _append(uid: str) -> None:
            node = self._manager._tool_graph.nodes[uid]
            entry: dict[str, typing.Any] = {
                "uid": uid,
                # Payload group path relative to the workspace root HDF5 group.
                "path": self._workspace_node_path(uid),
                # Restores graph node type without probing payload attrs first.
                "kind": "imagetool" if node.is_imagetool else "tool",
                "parent_uid": node.parent_uid,
                "display_name": node.display_text,
            }
            if node.is_imagetool:
                # Distinguishes embedded data from lazy file-backed/dask payloads.
                entry["data_backing"] = node.persistence_data_backing()[0]
                link_info = link_metadata.get(uid)
                if link_info is not None:
                    # link_group is an ordinal within this manifest, not a stable id.
                    entry["link_group"], entry["link_colors"] = link_info
            entries.append(entry)
            for child_uid in node._childtool_indices:
                if child_uid in self._manager._tool_graph.nodes:
                    _append(child_uid)

        for index in self._workspace_root_indices():
            _append(self._manager._tool_graph.root_wrappers[index].uid)
        for uid in self._manager._tool_graph.figure_uids:
            if uid in self._manager._tool_graph.nodes:
                _append(uid)
        return entries

    def _workspace_manifest(self) -> dict[str, typing.Any]:
        return workspace_format._workspace_manifest_payload(
            root_order=self._workspace_root_indices(),
            nodes=self._workspace_node_manifest_entries(),
            erlab_version=str(erlab.__version__),
            workspace_link_id=self._manager._workspace_state.link_id,
            manager_layout=self._workspace_layout_snapshot(),
            loader_state=self._workspace_loader_state_snapshot(),
            standalone_apps=self._workspace_standalone_apps_snapshot(),
            option_overrides=self._workspace_option_overrides_snapshot(),
            acquisition_context=(self._manager._acquisition_context.state_payload()),
            extension_requirements=(
                self._manager._extensions.workspace_requirement_payloads()
            ),
        )

    def _workspace_compression_mode(self) -> WorkspaceCompressionMode:
        return self._manager.effective_interactive_options.io.workspace.compression

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
            "metadata_editor": self._manager._metadata_editor.layout_payload(),
        }

    def _workspace_option_overrides_snapshot(self) -> dict[str, typing.Any]:
        return workspace_format.WorkspaceOptionOverridesState(
            overrides=erlab.interactive._options.core.normalize_workspace_option_overrides(
                self._manager._workspace_state.option_overrides
            )
        ).model_dump(mode="json")

    def _workspace_loader_state_snapshot(self) -> dict[str, typing.Any]:
        manager_loader_kwargs = self._manager._recent_loader_kwargs_by_filter
        manager_loader_extensions = self._manager._recent_loader_extensions_by_filter
        explorer_kwargs = self._controller._loader_state.explorer_loader_kwargs_by_name
        explorer_extensions = (
            self._controller._loader_state.explorer_loader_extensions_by_name
        )
        explorer = self._manager._standalone_app_windows.get("explorer")
        if explorer is not None and erlab.interactive.utils.qt_is_valid(explorer):
            kwargs_getter = getattr(explorer, "loader_kwargs_by_name", None)
            if callable(kwargs_getter):
                explorer_kwargs = kwargs_getter()
            extensions_getter = getattr(explorer, "loader_extensions_by_name", None)
            if callable(extensions_getter):
                explorer_extensions = extensions_getter()
        self._manager._sync_shared_loader_state(
            explorer_kwargs,
            explorer_extensions,
            apply_explorer=False,
        )
        runtime_state = workspace_format.WorkspaceLoaderState(
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
        self._controller._loader_state = runtime_state
        serialized_state = runtime_state.model_copy(
            update={
                "manager_loader_kwargs_by_filter": {
                    name: _serialize_loader_kwargs(kwargs)
                    for name, kwargs in (
                        runtime_state.manager_loader_kwargs_by_filter.items()
                    )
                },
                "explorer_loader_kwargs_by_name": {
                    name: _serialize_loader_kwargs(kwargs)
                    for name, kwargs in (
                        runtime_state.explorer_loader_kwargs_by_name.items()
                    )
                },
            }
        )
        return serialized_state.model_dump(mode="json", exclude_none=True)

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
            validated = self._controller._validated_standalone_app_state(key, state)
            if validated is not None:
                app_states[key] = validated
        return workspace_format.StandaloneAppsState(apps=app_states).model_dump(
            mode="json", exclude_none=True
        )

    def _workspace_stale_reference_rewrite_uids(
        self, available_uids: frozenset[str]
    ) -> list[str]:
        rewrite_uids: list[str] = []
        for uid, node in self._manager._tool_graph.nodes.items():
            if node.is_imagetool:
                continue
            if node.pending_workspace_tool_payload is not None:
                if not self._pending_workspace_tool_references_available(node):
                    rewrite_uids.append(uid)
                continue
            tool = typing.cast("erlab.interactive.utils.ToolWindow", node.tool_window)
            if tool is None:
                continue
            if not tool.can_save_and_load():
                continue
            if tool._persistence_reference_node_uids() - available_uids:
                rewrite_uids.append(uid)
                continue
            references = node._workspace_tool_data_references
            if not references:
                continue
            if not self._workspace_tool_references_match_current_snapshots(
                references.values()
            ):
                rewrite_uids.append(uid)
        return sorted(rewrite_uids, key=self._workspace_node_path)

    def _workspace_tool_references_match_current_snapshots(
        self, references: Iterable[Mapping[str, typing.Any]]
    ) -> bool:
        return all(
            self._controller._tool_data_reference_matches_current_snapshot(reference)
            for reference in references
        )

    def _save_workspace_document(
        self,
        fname: str | os.PathLike[str],
        *,
        document_access: _WorkspaceDocumentAccess | None = None,
        mark_clean: bool = True,
    ) -> None:
        if document_access is None:
            _require_itws_workspace_path(fname, _WORKSPACE_SAVE_SUFFIX_ERROR)
            self._controller._drain_workspace_deferred_events()
            with self._controller._workspace_document_access_context(fname) as access:
                self._save_workspace_document(
                    access.path,
                    document_access=access,
                    mark_clean=mark_clean,
                )
            return

        fname = document_access.path
        _require_itws_workspace_path(fname, _WORKSPACE_SAVE_SUFFIX_ERROR)
        self._manager._workspace_state.saving_depth += 1
        snapshot: _WorkspaceSaveSnapshot | None = None
        target_store: workspace_store.WorkspaceStore | None = None
        owned_store = False
        try:
            snapshot = self._workspace_generation_save_snapshot(
                self._manager._workspace_state.dirty_generation,
                fname=fname,
            )
            self._close_workspace_idle_readers(fname)
            target_store = workspace_store.WorkspaceStore.active(fname)
            if target_store is None:
                target_store = workspace_store.WorkspaceStore(
                    fname,
                    create=not pathlib.Path(fname).exists(),
                )
                owned_store = True
            committed_generation = workspace_storage._write_workspace_generation(
                target_store,
                snapshot.generation_plan,
                compression_mode=snapshot.compression_mode,
            )
            self._controller._adopt_committed_workspace_generation(
                fname,
                snapshot,
                manifest=committed_generation.manifest,
            )
        finally:
            if snapshot is not None:
                snapshot.close()
            if owned_store and target_store is not None:
                target_store.close()
            self._manager._workspace_state.saving_depth -= 1
        if mark_clean:
            self._controller._mark_workspace_clean()

    @staticmethod
    def _pending_workspace_node_attrs(
        node: _ImageToolWrapper | _ManagedWindowNode,
        attrs: Mapping[str, typing.Any] | None,
        *,
        kind: typing.Literal["imagetool", "tool"],
    ) -> dict[str, typing.Any]:
        if attrs is None:
            attrs = {}
        attrs = dict(attrs)
        attrs["manager_node_uid"] = node.uid
        attrs["manager_node_kind"] = kind
        attrs["manager_node_snapshot_token"] = node.snapshot_token
        attrs["manager_node_source_snapshot_token"] = node.source_snapshot_token
        attrs["manager_node_added_at"] = node.added_time_iso
        if node.note:
            attrs["manager_node_note"] = node.note
        else:
            attrs.pop("manager_node_note", None)
        return attrs

    def _pending_workspace_imagetool_attrs(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> dict[str, typing.Any]:
        attrs = self._pending_workspace_node_attrs(
            node, node.pending_workspace_payload_attrs, kind="imagetool"
        )
        attrs["itool_name"] = node.name
        attrs["itool_title"] = node.name

        provenance_spec = node.provenance_spec
        if provenance_spec is None:
            attrs.pop("manager_node_provenance_spec", None)
        else:
            attrs["manager_node_provenance_spec"] = json.dumps(
                provenance_spec.model_dump(mode="json")
            )

        if isinstance(node, _ImageToolWrapper):
            if node.source_input_ndim is None:
                attrs.pop("manager_node_source_input_ndim", None)
            else:
                attrs["manager_node_source_input_ndim"] = int(node.source_input_ndim)
            if node.watched:
                watched_metadata = node.watched_metadata()
                attrs["manager_node_watched_varname"] = watched_metadata["varname"]
                attrs["manager_node_watched_uid"] = watched_metadata["uid"]
                workspace_link_id = watched_metadata.get("workspace_link_id")
                if workspace_link_id is None:
                    attrs.pop("manager_node_watched_workspace_link_id", None)
                else:
                    attrs["manager_node_watched_workspace_link_id"] = str(
                        workspace_link_id
                    )
                source_label = watched_metadata.get("source_label")
                if source_label is None:
                    attrs.pop("manager_node_watched_source_label", None)
                else:
                    attrs["manager_node_watched_source_label"] = str(source_label)
                source_uid = watched_metadata.get("source_uid")
                if source_uid is None:
                    attrs.pop("manager_node_watched_source_uid", None)
                else:
                    attrs["manager_node_watched_source_uid"] = str(source_uid)
                attrs["manager_node_watched_connected"] = bool(
                    watched_metadata.get("connected", False)
                )
            else:
                for key in (
                    "manager_node_watched_varname",
                    "manager_node_watched_uid",
                    "manager_node_watched_workspace_link_id",
                    "manager_node_watched_source_label",
                    "manager_node_watched_source_uid",
                    "manager_node_watched_connected",
                ):
                    attrs.pop(key, None)

        output_id = node.output_id
        if output_id is None:
            attrs.pop("manager_node_output_id", None)
        else:
            attrs["manager_node_output_id"] = output_id

        source_spec = node.source_spec
        if source_spec is None:
            attrs.pop("manager_node_live_source_spec", None)
        else:
            attrs["manager_node_live_source_spec"] = json.dumps(
                source_spec.model_dump(mode="json")
            )
        if source_spec is None and output_id is None:
            attrs.pop("manager_node_source_state", None)
            attrs.pop("manager_node_source_auto_update", None)
        else:
            attrs["manager_node_source_state"] = node.source_state
            attrs["manager_node_source_auto_update"] = bool(node.source_auto_update)
        return attrs

    def _pending_workspace_tool_attrs(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> dict[str, typing.Any]:
        attrs = self._pending_workspace_node_attrs(
            node, node.pending_workspace_payload_attrs, kind="tool"
        )
        old_display_name = workspace_format._decode_workspace_attr_text(
            attrs.get("tool_display_name")
        )
        old_title = workspace_format._decode_workspace_attr_text(
            attrs.get("tool_title")
        )
        attrs["tool_display_name"] = node.name
        if old_title and old_display_name and old_title.endswith(old_display_name):
            attrs["tool_title"] = old_title[: -len(old_display_name)] + node.name
        else:
            attrs["tool_title"] = node.name

        source_spec = node.source_spec
        if source_spec is None:
            attrs.pop(erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR, None)
        else:
            attrs[erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR] = json.dumps(
                source_spec.model_dump(mode="json")
            )
        source_binding = node.source_binding
        if source_spec is not None or source_binding is None:
            attrs.pop(erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR, None)
        else:
            attrs[erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR] = json.dumps(
                source_binding.model_dump(mode="json")
            )
        if node.has_source_binding:
            attrs[erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR] = node.source_state
            attrs[erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR] = bool(
                node.source_auto_update
            )
        else:
            attrs.pop(erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR, None)
            attrs.pop(erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR, None)
        return attrs

    def _pending_workspace_payload_attrs_for_save(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> dict[str, typing.Any] | None:
        match node.pending_workspace_payload_kind:
            case "imagetool":
                return self._pending_workspace_imagetool_attrs(node)
            case "tool":
                return self._pending_workspace_tool_attrs(node)
            case _:
                return None

    def _pending_workspace_tool_references_available(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> bool:
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            return True
        payload = attrs.get(erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR)
        if payload is None:
            return True
        if isinstance(payload, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                payload = payload.decode()
        if not isinstance(payload, str):
            return False
        try:
            references = json.loads(payload)
        except Exception:
            return False
        if not isinstance(references, dict):
            return False
        for reference in references.values():
            if not isinstance(reference, dict):
                return False
            kind = reference.get("kind")
            if kind == "parent_source":
                if (
                    node.parent_uid is None
                    or node.parent_uid not in self._manager._tool_graph.nodes
                ):
                    return False
                continue
            if kind != "manager_node":
                return False
            if not self._controller._tool_data_reference_matches_current_snapshot(
                reference
            ):
                return False
        return True

    def _pending_workspace_tool_unavailable_reference_uids(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[str, ...]:
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            return ()
        payload = attrs.get(erlab.interactive.utils._TOOL_DATA_REFERENCES_ATTR)
        if isinstance(payload, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                payload = payload.decode()
        if not isinstance(payload, str):
            return ()
        try:
            references = json.loads(payload)
        except Exception:
            return ()
        if not isinstance(references, dict):
            return ()
        unavailable_uids: set[str] = set()
        for reference in references.values():
            if not isinstance(reference, dict):
                continue
            kind = reference.get("kind")
            if kind == "parent_source":
                parent_uid = node.parent_uid
                if (
                    isinstance(parent_uid, str)
                    and parent_uid not in self._manager._tool_graph.nodes
                ):
                    unavailable_uids.add(parent_uid)
                continue
            if kind != "manager_node":
                continue
            node_uid = reference.get("node_uid")
            if (
                isinstance(node_uid, str)
                and node_uid
                and not self._controller._tool_data_reference_matches_current_snapshot(
                    reference
                )
            ):
                unavailable_uids.add(node_uid)
        return tuple(sorted(unavailable_uids))

    def _workspace_save_snapshot(
        self, fname: str | os.PathLike[str]
    ) -> _WorkspaceSaveSnapshot:
        self._controller._drain_workspace_deferred_events()
        generation = self._manager._workspace_state.dirty_generation
        self._manager._workspace_state.saving_depth += 1
        try:
            return self._workspace_generation_save_snapshot(generation, fname=fname)
        finally:
            self._manager._workspace_state.saving_depth -= 1

    def _workspace_generation_save_snapshot(
        self,
        generation: int,
        *,
        fname: str | os.PathLike[str],
    ) -> _WorkspaceSaveSnapshot:
        """Capture one immutable generation without copying unchanged payloads."""
        source_store = getattr(self._controller, "_workspace_store", None)
        current_manifest: dict[str, typing.Any] | None = None
        if source_store is not None and not source_store.closed:
            with contextlib.suppress(Exception):
                current_manifest = source_store.current_generation().manifest
        if current_manifest is None and self._manager._workspace_state.path is not None:
            with contextlib.suppress(Exception):
                root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(
                    self._manager._workspace_state.path
                )
                schema_version, manifest = (
                    workspace_format._workspace_file_metadata_from_attrs(root_attrs)
                )
                if workspace_format._workspace_schema_uses_immutable_generations(
                    schema_version
                ):
                    current_manifest = manifest

        previous_entries: dict[str, Mapping[str, typing.Any]] = {}
        if current_manifest is not None:
            for entry in workspace_format._iter_workspace_manifest_node_entries(
                current_manifest
            ):
                uid = entry.get("uid")
                if isinstance(uid, str):
                    previous_entries[uid] = entry

        manifest = self._workspace_manifest()
        base_entries = [
            dict(entry)
            for entry in workspace_format._iter_workspace_manifest_node_entries(
                manifest
            )
        ]
        available_uids = frozenset(
            str(entry["uid"]) for entry in base_entries if "uid" in entry
        )
        stale_reference_uids = set(
            self._workspace_stale_reference_rewrite_uids(available_uids)
        )
        dirty_data = (
            self._manager._workspace_state.dirty_data
            | self._manager._workspace_state.dirty_added
            | stale_reference_uids
        )
        dirty_state = self._manager._workspace_state.dirty_state
        source_workspace_path = self._manager._workspace_state.path

        object_writes: dict[str, workspace_storage._WorkspaceObjectWrite] = {}
        final_entries: list[dict[str, typing.Any]] = []
        serialized_datasets: list[xr.Dataset] = []
        extension_object_ids = {
            object_id
            for requirement in manifest.get("extension_requirements", ())
            if isinstance(requirement, dict)
            and isinstance(object_id := requirement.get("embedded_object_id"), str)
            and object_id
        }

        def _serialize(uid: str) -> xr.Dataset | None:
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None:
                return None
            constructor: dict[str, xr.Dataset] = {}
            self._serialize_workspace_node(
                constructor,
                node,
                self._workspace_node_path(uid),
                include_children=False,
            )
            return constructor.get(self._workspace_payload_path(uid))

        for entry in base_entries:
            uid = str(entry["uid"])
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None:
                continue
            kind = typing.cast("typing.Literal['imagetool', 'tool']", entry["kind"])
            previous = previous_entries.get(uid)
            previous_object_id = (
                previous.get("payload_object_id") if previous is not None else None
            )
            can_reuse_object = (
                isinstance(previous_object_id, str)
                and bool(previous_object_id)
                and previous_object_id not in extension_object_ids
                and uid not in dirty_data
            )

            dataset: xr.Dataset | None = None
            attrs_payload = (
                previous.get("payload_attrs") if previous is not None else None
            )
            needs_current_attrs = (
                attrs_payload is None or uid in dirty_state or uid in dirty_data
            )
            if needs_current_attrs:
                pending_attrs = (
                    self._pending_workspace_payload_attrs_for_save(node)
                    if uid not in self._manager._workspace_state.dirty_data
                    else None
                )
                if pending_attrs is not None:
                    attrs_payload = workspace_format._workspace_manifest_attrs(
                        pending_attrs
                    )
                else:
                    dataset = _serialize(uid)
                    if dataset is None:
                        if not can_reuse_object:
                            continue
                    else:
                        attrs_payload = workspace_format._workspace_manifest_attrs(
                            dataset.attrs
                        )
                        serialized_datasets.append(dataset)
            if attrs_payload is not None:
                entry["payload_attrs"] = attrs_payload

            if can_reuse_object:
                object_id = typing.cast("str", previous_object_id)
                previous_entry = typing.cast("Mapping[str, typing.Any]", previous)
                source_path = previous_entry.get("payload_path")
                if not isinstance(source_path, str):
                    source_path = workspace_store.WorkspaceStore.object_path(object_id)
                if source_workspace_path is not None and object_id not in object_writes:
                    object_writes[object_id] = workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        source_file=str(source_workspace_path),
                        source_path=source_path,
                    )
                if dataset is not None:
                    dataset.close()
            else:
                object_id = f"node-{uuid.uuid4().hex}"
                pending_payload = node.pending_workspace_payload
                pending_kind = node.pending_workspace_payload_kind
                use_pending = (
                    dataset is None
                    and pending_payload is not None
                    and pending_kind == kind
                    and uid not in self._manager._workspace_state.dirty_data
                    and (
                        kind != "tool"
                        or self._pending_workspace_tool_references_available(node)
                    )
                )
                if use_pending:
                    pending_file, pending_path = typing.cast(
                        "tuple[str | os.PathLike[str], str]", pending_payload
                    )
                    object_writes[object_id] = workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        source_file=os.fsdecode(pending_file),
                        source_path=pending_path,
                    )
                else:
                    if dataset is None:
                        dataset = _serialize(uid)
                        if dataset is None:
                            continue
                    if all(existing is not dataset for existing in serialized_datasets):
                        serialized_datasets.append(dataset)
                    entry["payload_attrs"] = workspace_format._workspace_manifest_attrs(
                        dataset.attrs
                    )
                    object_writes[object_id] = workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        dataset=dataset,
                    )

            entry["payload_object_id"] = object_id
            entry["payload_path"] = workspace_store.WorkspaceStore.object_path(
                object_id
            )
            final_entries.append(entry)

        manifest["nodes"] = final_entries
        node_object_ids = {
            str(entry["payload_object_id"])
            for entry in final_entries
            if entry.get("payload_object_id")
        }
        previous_extension_object_ids: frozenset[str] = frozenset()
        if current_manifest is not None and source_workspace_path is not None:
            with contextlib.suppress(OSError, KeyError):
                previous_extension_object_ids = (
                    workspace_storage._existing_workspace_object_ids(
                        source_workspace_path,
                        workspace_store.WorkspaceStore.manifest_object_ids(
                            current_manifest
                        ),
                    )
                )
        for raw_requirement in manifest.get("extension_requirements", ()):
            if not isinstance(raw_requirement, dict):
                continue
            object_id = raw_requirement.get("embedded_object_id")
            if not isinstance(object_id, str) or not object_id:
                continue
            try:
                workspace_store.WorkspaceStore.object_path(object_id)
            except ValueError:
                logger.warning(
                    "Preserving an extension requirement with an invalid object ID",
                    extra={"suppress_ui_alert": True},
                )
                continue
            if object_id in node_object_ids:
                logger.warning(
                    "Ignoring an extension object ID that conflicts with node data",
                    extra={"suppress_ui_alert": True},
                )
                continue
            extension_id = raw_requirement.get("extension_id")
            revision_hash = raw_requirement.get("revision_hash")
            if (
                isinstance(extension_id, str)
                and isinstance(revision_hash, str)
                and object_id == f"extension-{revision_hash}"
            ):
                try:
                    source = self._manager._extensions.revision_source_bytes(
                        extension_id, revision_hash
                    )
                except (FileNotFoundError, KeyError):
                    pass
                else:
                    object_writes[object_id] = workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        blob=source,
                        blob_kind="extension-python-source-v1",
                    )
                    continue
            unresolved_object = (
                self._manager._extensions._workspace_unresolved_embedded_objects.get(
                    object_id
                )
            )
            if unresolved_object is not None:
                source, kind = unresolved_object
                object_writes.setdefault(
                    object_id,
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        blob=source,
                        blob_kind=kind,
                    ),
                )
                continue
            if (
                source_workspace_path is not None
                and object_id in previous_extension_object_ids
            ):
                object_writes.setdefault(
                    object_id,
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        source_file=str(source_workspace_path),
                        source_path=workspace_store.WorkspaceStore.object_path(
                            object_id
                        ),
                    ),
                )
        manifest["schema_version"] = (
            workspace_format._current_workspace_schema_version()
        )
        manifest["storage_model"] = "immutable-generations-v1"
        manifest.pop("transaction_protocol", None)
        preserved_groups: tuple[workspace_storage._WorkspaceGroupCopy, ...] = ()
        legacy_reader_rebindings: tuple[tuple[str, str], ...] = ()
        leased_legacy_group_paths: frozenset[str] = frozenset()
        if source_store is not None and not source_store.closed:
            leased_legacy_group_paths = source_store.leased_legacy_group_paths
            safe_rebindings: list[tuple[str, str]] = []
            for entry in final_entries:
                uid = entry.get("uid")
                object_id = entry.get("payload_object_id")
                if (
                    not isinstance(uid, str)
                    or uid in dirty_data
                    or not isinstance(object_id, str)
                ):
                    continue
                legacy_path = f"/{self._workspace_payload_path(uid).strip('/')}"
                if legacy_path in leased_legacy_group_paths:
                    safe_rebindings.append((legacy_path, object_id))
            legacy_reader_rebindings = tuple(sorted(safe_rebindings))
        if (
            source_store is not None
            and pathlib.Path(fname).resolve() != source_store.path
        ):
            source_path = str(source_store.path)
            for object_id in source_store.leased_object_ids:
                object_writes.setdefault(
                    object_id,
                    workspace_storage._WorkspaceObjectWrite(
                        object_id,
                        source_file=source_path,
                        source_path=source_store.object_path(object_id),
                    ),
                )
            preserved_groups = tuple(
                workspace_storage._WorkspaceGroupCopy(
                    source_file=source_path,
                    source_path=group_path,
                    target_path=group_path,
                )
                for group_path in sorted(leased_legacy_group_paths)
            )
        return _WorkspaceSaveSnapshot(
            generation=generation,
            generation_plan=workspace_storage._WorkspaceGenerationPlan(
                manifest=manifest,
                objects=tuple(object_writes.values()),
                preserved_groups=preserved_groups,
                legacy_reader_rebindings=legacy_reader_rebindings,
            ),
            compression_mode=self._workspace_compression_mode(),
            serialized_tool_data_references=(
                self._serialized_tool_data_references(serialized_datasets)
            ),
        )
