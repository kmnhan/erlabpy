"""Transactional persistence and save workers for manager workspaces."""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import os
import pathlib
import time
import traceback
import typing
from dataclasses import dataclass

import xarray as xr
from qtpy import QtCore

import erlab
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
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
    from collections.abc import Callable, Hashable, Iterable, Mapping

    import h5py

    from erlab.interactive._options.schema import WorkspaceCompressionMode
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._widgets import _WorkspaceDocumentAccess
    from erlab.interactive.imagetool.manager._workspace._controller import (
        _WorkspaceController,
    )
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

from erlab.interactive.imagetool.manager._workspace._format import (
    _require_itws_workspace_path,
)

logger = logging.getLogger(__name__)
_WORKSPACE_SAVE_SUFFIX_ERROR = "ImageTool workspace documents must be saved as .itws"
_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES = 128 * 1024 * 1024
_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO = 0.10


@dataclass
class _WorkspaceSaveSnapshot:
    generation: int
    root_attrs: dict[str, typing.Any]
    delta_save_count: int
    estimated_obsolete_bytes: int = 0
    replacement_delta_count: int = 0
    repack_estimate_known: bool = True
    compression_mode: WorkspaceCompressionMode = "zstd1"
    file_repack: bool = False
    full_tree: xr.DataTree | None = None
    copy_source: str | None = None
    copy_groups: tuple[workspace_storage._WorkspaceCopyGroup, ...] = ()
    copy_group_sources: tuple[workspace_storage._WorkspaceCopyGroupWithSource, ...] = ()
    rewrite_groups: tuple[tuple[str, dict[str, xr.Dataset]], ...] = ()
    attr_updates: tuple[
        tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]], ...
    ] = ()
    serialized_tool_data_references: tuple[
        tuple[str, dict[str, dict[str, typing.Any]]], ...
    ] = ()

    def close(self) -> None:
        if self.full_tree is not None:
            self.full_tree.close()


@dataclass(frozen=True)
class _WorkspaceSaveError:
    traceback_text: str
    missing_source_path: str | None = None


class _WorkspaceSaveWorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(float, object)


class _WorkspaceSaveResultReceiver(QtCore.QObject):
    def __init__(
        self,
        *,
        callback: Callable[[float, _WorkspaceSaveError | None], None] | None = None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._callback = callback

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
    ) -> None:
        super().__init__()
        self.signals = _WorkspaceSaveWorkerSignals()
        self._fname = fname
        self._snapshot = snapshot

    def run(self) -> None:
        start_time = time.perf_counter()
        error: _WorkspaceSaveError | None = None
        try:
            if self._snapshot.file_repack:
                workspace_storage._write_full_workspace_tree_file(
                    self._fname,
                    None,
                    self._snapshot.root_attrs,
                    copy_source=self._snapshot.copy_source,
                    copy_groups=self._snapshot.copy_groups,
                    copy_group_sources=self._snapshot.copy_group_sources,
                    compression_mode=self._snapshot.compression_mode,
                )
            elif self._snapshot.full_tree is None:
                workspace_storage._write_workspace_transaction_file(
                    self._fname,
                    self._snapshot.rewrite_groups,
                    self._snapshot.attr_updates,
                    self._snapshot.root_attrs,
                    compression_mode=self._snapshot.compression_mode,
                )
            else:
                workspace_storage._write_full_workspace_tree_file(
                    self._fname,
                    self._snapshot.full_tree,
                    self._snapshot.root_attrs,
                    copy_source=self._snapshot.copy_source,
                    copy_groups=self._snapshot.copy_groups,
                    copy_group_sources=self._snapshot.copy_group_sources,
                    compression_mode=self._snapshot.compression_mode,
                )
        except workspace_storage._WorkspaceBackingFileNotFoundError as exc:
            error = _WorkspaceSaveError(
                traceback_text=traceback.format_exc(),
                missing_source_path=exc.source_path,
            )
        except Exception:
            error = _WorkspaceSaveError(traceback_text=traceback.format_exc())
        finally:
            with contextlib.suppress(Exception):
                self._snapshot.close()
        self.signals.finished.emit(time.perf_counter() - start_time, error)


class _WorkspaceSaver:
    """Serialize and snapshot workspace state for one manager."""

    def __init__(
        self, manager: ImageToolManager, controller: _WorkspaceController
    ) -> None:
        self._manager = manager
        self._controller = controller

    @staticmethod
    def _serialized_tool_data_references(
        datasets: Iterable[xr.Dataset],
    ) -> tuple[tuple[str, dict[str, dict[str, typing.Any]]], ...]:
        references_by_uid: dict[str, dict[str, dict[str, typing.Any]]] = {}
        for ds in datasets:
            if ds.attrs.get("manager_node_kind") != "tool":
                continue
            uid = ds.attrs.get("manager_node_uid")
            if not isinstance(uid, str) or not uid:
                continue
            references_by_uid[uid] = (
                erlab.interactive.utils.ToolWindow._saved_tool_data_references(ds)
            )
        return tuple(sorted(references_by_uid.items()))

    @classmethod
    def _serialized_tool_data_references_from_tree(
        cls,
        tree: xr.DataTree,
        *,
        exclude_payload_paths: Iterable[str] = (),
    ) -> tuple[tuple[str, dict[str, dict[str, typing.Any]]], ...]:
        excluded = {path.strip("/") for path in exclude_payload_paths}
        return cls._serialized_tool_data_references(
            node.to_dataset(inherit=False)
            for node in tree.subtree
            if node.path.strip("/") not in excluded
        )

    @classmethod
    def _serialized_tool_data_references_from_delta(
        cls,
        rewrite_groups: Iterable[tuple[str, dict[str, xr.Dataset]]],
        attr_updates: Iterable[
            tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
        ],
    ) -> tuple[tuple[str, dict[str, dict[str, typing.Any]]], ...]:
        datasets = [
            ds
            for _group_path, constructor in rewrite_groups
            for ds in constructor.values()
        ]
        datasets.extend(
            ds
            for _payload_path, _attrs, (_node_path, constructor) in attr_updates
            for ds in constructor.values()
        )
        return cls._serialized_tool_data_references(datasets)

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
                raise ValueError("Could not read this saved tool from the workspace.")
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

    def _workspace_root_attrs_payload(
        self,
        *,
        delta_save_count: int | None = None,
        estimated_obsolete_bytes: int | None = None,
        replacement_delta_count: int | None = None,
        repack_estimate_known: bool | None = None,
    ) -> dict[str, typing.Any]:
        state = self._manager._workspace_state
        if delta_save_count is None:
            delta_save_count = state.delta_save_count
        if estimated_obsolete_bytes is None:
            estimated_obsolete_bytes = state.estimated_obsolete_bytes
        if replacement_delta_count is None:
            replacement_delta_count = state.replacement_delta_count
        if repack_estimate_known is None:
            repack_estimate_known = state.repack_estimate_known
        return workspace_format._workspace_root_attrs_payload(
            root_order=self._workspace_root_indices(),
            nodes=self._workspace_node_manifest_entries(),
            delta_save_count=delta_save_count,
            erlab_version=str(erlab.__version__),
            workspace_link_id=self._manager._workspace_state.link_id,
            manager_layout=self._workspace_layout_snapshot(),
            loader_state=self._workspace_loader_state_snapshot(),
            standalone_apps=self._workspace_standalone_apps_snapshot(),
            option_overrides=self._workspace_option_overrides_snapshot(),
            acquisition_context=(self._manager._acquisition_context.state_payload()),
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            repack_estimate_known=repack_estimate_known,
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

    def _workspace_datatree_for_payload_uids(self, uids: Iterable[str]) -> xr.DataTree:
        constructor: dict[str, xr.Dataset] = {}
        for uid in sorted(set(uids), key=self._workspace_node_path):
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None:
                continue
            self._serialize_workspace_node(
                constructor,
                node,
                self._workspace_node_path(uid),
                include_children=False,
            )
        tree = xr.DataTree.from_dict(constructor)
        workspace_format._set_legacy_workspace_schema(tree.attrs)
        return tree

    def _serialize_workspace_node_for_full_save_fallback(
        self,
        constructor: dict[str, xr.Dataset],
        node: _ImageToolWrapper | _ManagedWindowNode,
        path: str,
        copy_group_sources: list[tuple[str, str, str, dict[str, typing.Any] | None]],
        *,
        include_children: bool,
        pending_copy_allowed: Callable[[tuple[str | os.PathLike[str], str]], bool],
    ) -> None:
        pending_payload = node.pending_workspace_payload
        pending_kind = node.pending_workspace_payload_kind
        if (
            pending_payload is not None
            and pending_kind is not None
            and node.uid not in self._manager._workspace_state.dirty_data
            and (
                pending_kind != "tool"
                or self._pending_workspace_tool_references_available(node)
            )
            and pending_copy_allowed(pending_payload)
        ):
            source_file, source_path = pending_payload
            copy_group_sources.append(
                (
                    os.fsdecode(source_file),
                    source_path,
                    f"{path}/{pending_kind}",
                    self._pending_workspace_payload_attrs_for_save(node),
                )
            )
        else:
            self._serialize_workspace_node(
                constructor,
                node,
                path,
                include_children=False,
            )

        if not include_children:
            return
        for child_uid in node._childtool_indices:
            child = self._manager._child_node(child_uid)
            self._serialize_workspace_node_for_full_save_fallback(
                constructor,
                child,
                f"{path}/childtools/{child_uid}",
                copy_group_sources,
                include_children=True,
                pending_copy_allowed=pending_copy_allowed,
            )

    def _workspace_full_save_fallback_tree(
        self,
        *,
        require_matching_compression: bool,
        compression_mode: WorkspaceCompressionMode,
    ) -> tuple[
        xr.DataTree,
        tuple[tuple[str, str, str, dict[str, typing.Any] | None], ...],
    ]:
        constructor: dict[str, xr.Dataset] = {}
        copy_group_sources: list[
            tuple[str, str, str, dict[str, typing.Any] | None]
        ] = []
        with contextlib.ExitStack() as stack:
            pending_compression_cache: dict[tuple[str, str], bool] = {}
            pending_compression_files: dict[str, typing.Any] = {}

            def _pending_copy_allowed(
                pending_payload: tuple[str | os.PathLike[str], str],
            ) -> bool:
                if not require_matching_compression:
                    return True
                source_file, source_path = pending_payload
                source_key = os.fsdecode(source_file)
                source_path = source_path.strip("/")
                cache_key = (source_key, source_path)
                if cache_key in pending_compression_cache:
                    return pending_compression_cache[cache_key]
                try:
                    h5_file = pending_compression_files.get(source_key)
                    if h5_file is None:
                        workspace_arrays.ensure_workspace_hdf5_filters_registered()
                        stack.enter_context(
                            workspace_arrays._workspace_file_lock(source_key)
                        )
                        h5_file = stack.enter_context(h5py.File(source_key, "r"))
                        pending_compression_files[source_key] = h5_file
                    matches = workspace_arrays._h5_group_matches_compression(
                        h5_file,
                        source_path,
                        compression_mode,
                    )
                except Exception:
                    logger.debug(
                        "Cannot verify pending workspace payload compression",
                        exc_info=True,
                    )
                    matches = False
                pending_compression_cache[cache_key] = matches
                return matches

            for index in self._workspace_root_indices():
                self._serialize_workspace_node_for_full_save_fallback(
                    constructor,
                    self._manager._tool_graph.root_wrappers[index],
                    str(index),
                    copy_group_sources,
                    include_children=True,
                    pending_copy_allowed=_pending_copy_allowed,
                )
            for uid in list(self._manager._tool_graph.figure_uids):
                node = self._manager._tool_graph.nodes.get(uid)
                if not isinstance(node, _ManagedWindowNode):
                    continue
                self._serialize_workspace_node_for_full_save_fallback(
                    constructor,
                    node,
                    f"figures/{uid}",
                    copy_group_sources,
                    include_children=False,
                    pending_copy_allowed=_pending_copy_allowed,
                )
        tree = xr.DataTree.from_dict(constructor)
        workspace_format._set_legacy_workspace_schema(tree.attrs)
        return tree, tuple(copy_group_sources)

    def _workspace_full_save_source_identities(
        self,
    ) -> tuple[pathlib.Path, dict[tuple[str, str], str]] | None:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return None
        workspace_path = pathlib.Path(workspace_path)
        if (
            self._manager._workspace_state.schema_version
            != workspace_format._current_workspace_schema_version()
            or not workspace_path.exists()
        ):
            return None

        try:
            root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(
                workspace_path
            )
        except Exception:
            return None
        schema_version, _delta_save_count, manifest = (
            workspace_format._workspace_file_metadata_from_attrs(root_attrs)
        )
        if (
            schema_version != workspace_format._current_workspace_schema_version()
            or manifest is None
        ):
            return None

        manifest_entries = workspace_format._workspace_manifest_payload_entries(
            manifest
        )
        identities = {
            (uid, kind): payload_path for uid, kind, payload_path in manifest_entries
        }
        if not identities:
            return None
        return workspace_path, identities

    def _workspace_full_save_manifest_entries(
        self, root_attrs: Mapping[str, typing.Any]
    ) -> list[tuple[str, str, str]]:
        manifest = workspace_format._workspace_manifest_from_attrs(
            typing.cast("Mapping[Hashable, typing.Any]", root_attrs)
        )
        return workspace_format._workspace_manifest_payload_entries(manifest)

    def _workspace_full_save_dirty_payload_uids(self) -> set[str]:
        state = self._manager._workspace_state
        serialize_uids: set[str] = set()
        for uid in state.dirty_added | state.dirty_data:
            if uid not in self._manager._tool_graph.nodes:
                continue
            serialize_uids.add(uid)
            serialize_uids.update(self._manager._iter_descendant_uids(uid))
        serialize_uids.update(
            uid for uid in state.dirty_state if uid in self._manager._tool_graph.nodes
        )
        serialize_uids.update(
            self._workspace_stale_reference_rewrite_uids(
                frozenset(self._manager._tool_graph.nodes)
            )
        )
        return serialize_uids

    def _workspace_full_save_manifest_first_snapshot(
        self,
        generation: int,
        fname: str | os.PathLike[str],
        root_attrs: dict[str, typing.Any],
        *,
        compression_mode: WorkspaceCompressionMode,
        require_matching_compression: bool,
    ) -> _WorkspaceSaveSnapshot | None:
        if workspace_storage._workspace_path_is_likely_network_path(fname):
            return None
        source = self._workspace_full_save_source_identities()
        if source is None:
            return None
        workspace_path, identities = source

        copy_groups: list[tuple[str, str, dict[str, typing.Any] | None]] = []
        copy_group_sources: list[
            tuple[str, str, str, dict[str, typing.Any] | None]
        ] = []
        serialize_uids = self._workspace_full_save_dirty_payload_uids()
        reason_counts: collections.Counter[str] = collections.Counter()
        copied_bytes = 0
        serialized_existing_bytes = 0
        plan_started_at = time.perf_counter()
        try:
            workspace_arrays.ensure_workspace_hdf5_filters_registered()
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    workspace_arrays._workspace_file_lock(workspace_path)
                )
                h5_file = stack.enter_context(h5py.File(workspace_path, "r"))
                pending_compression_cache: dict[tuple[str, str], bool] = {}
                pending_compression_files: dict[str, typing.Any] = {}

                def _pending_copy_allowed(
                    pending_payload: tuple[str | os.PathLike[str], str],
                ) -> bool:
                    if not require_matching_compression:
                        return True
                    source_file, source_path = pending_payload
                    source_key = os.fsdecode(source_file)
                    source_path = source_path.strip("/")
                    cache_key = (source_key, source_path)
                    if cache_key in pending_compression_cache:
                        return pending_compression_cache[cache_key]
                    try:
                        pending_h5_file = pending_compression_files.get(source_key)
                        if pending_h5_file is None:
                            stack.enter_context(
                                workspace_arrays._workspace_file_lock(source_key)
                            )
                            pending_h5_file = stack.enter_context(
                                h5py.File(source_key, "r")
                            )
                            pending_compression_files[source_key] = pending_h5_file
                        matches = workspace_arrays._h5_group_matches_compression(
                            pending_h5_file,
                            source_path,
                            compression_mode,
                        )
                    except Exception:
                        logger.debug(
                            "Cannot verify pending workspace payload compression",
                            exc_info=True,
                        )
                        matches = False
                    pending_compression_cache[cache_key] = matches
                    return matches

                for (
                    uid,
                    kind,
                    payload_path,
                ) in self._workspace_full_save_manifest_entries(root_attrs):
                    node = self._manager._tool_graph.nodes.get(uid)
                    if node is None:
                        reason_counts["missing_node"] += 1
                        continue
                    if not node.is_imagetool:
                        if node.pending_workspace_tool_payload is not None:
                            if not self._pending_workspace_tool_references_available(
                                node
                            ):
                                serialize_uids.add(uid)
                                reason_counts["stale_tool_reference"] += 1
                        else:
                            tool = node.tool_window
                            if tool is None or not tool.can_save_and_load():
                                reason_counts["unsupported_tool"] += 1
                                continue
                    pending_payload = node.pending_workspace_payload
                    if (
                        pending_payload is not None
                        and node.pending_workspace_payload_kind == kind
                        and uid not in self._manager._workspace_state.dirty_data
                        and (
                            kind != "tool"
                            or self._pending_workspace_tool_references_available(node)
                        )
                    ):
                        source_path_for_pending = identities.get((uid, kind))
                        pending_workspace_path, pending_payload_path = pending_payload
                        pending_source_matches = (
                            source_path_for_pending is not None
                            and workspace_arrays._normalized_file_path(
                                pending_workspace_path
                            )
                            == workspace_arrays._normalized_file_path(workspace_path)
                            and pending_payload_path
                            == source_path_for_pending.strip("/")
                        )
                        pending_external_copy = (
                            uid in self._manager._workspace_state.dirty_added
                            or not pending_source_matches
                        )
                        if pending_external_copy:
                            if _pending_copy_allowed(pending_payload):
                                pending_source_file, pending_source_path = (
                                    pending_payload
                                )
                                copy_group_sources.append(
                                    (
                                        os.fsdecode(pending_source_file),
                                        pending_source_path,
                                        payload_path,
                                        self._pending_workspace_payload_attrs_for_save(
                                            node
                                        ),
                                    )
                                )
                                serialize_uids.discard(uid)
                                reason_counts["pending_external"] += 1
                                continue
                            serialize_uids.add(uid)
                            reason_counts["pending_compression_mismatch"] += 1
                            continue
                    source_path = identities.get((uid, kind))
                    if source_path is None or source_path != payload_path:
                        serialize_uids.add(uid)
                        reason_counts[
                            "missing_source" if source_path is None else "moved"
                        ] += 1
                        continue
                    source_group = h5_file.get(source_path)
                    if not isinstance(source_group, h5py.Group):
                        serialize_uids.add(uid)
                        reason_counts["missing_source_group"] += 1
                        continue
                    source_storage_size = (
                        workspace_arrays._workspace_h5_object_storage_size(source_group)
                    )
                    compression_mismatch = (
                        require_matching_compression
                        and not workspace_arrays._h5_group_matches_compression(
                            h5_file, source_path, compression_mode
                        )
                    )
                    pending_payload = node.pending_workspace_payload
                    pending_source_matches = False
                    if pending_payload is not None:
                        pending_workspace_path, pending_payload_path = pending_payload
                        pending_source_matches = workspace_arrays._normalized_file_path(
                            pending_workspace_path
                        ) == workspace_arrays._normalized_file_path(
                            workspace_path
                        ) and pending_payload_path == source_path.strip("/")
                    if compression_mismatch:
                        serialize_uids.add(uid)
                        reason_counts["compression_mismatch"] += 1
                        serialized_existing_bytes += source_storage_size
                        continue
                    if (
                        pending_source_matches
                        and uid in self._manager._workspace_state.dirty_state
                        and uid not in self._manager._workspace_state.dirty_data
                        and uid not in self._manager._workspace_state.dirty_added
                    ):
                        serialize_uids.discard(uid)
                        update = self._workspace_attr_update_snapshot(uid)
                        copy_groups.append(
                            (
                                source_path,
                                payload_path,
                                None if update is None else update[1],
                            )
                        )
                        reason_counts["pending_dirty_state"] += 1
                        copied_bytes += source_storage_size
                        continue
                    if uid in serialize_uids:
                        state = self._manager._workspace_state
                        if uid in state.dirty_added:
                            reason_counts["dirty_added"] += 1
                        elif uid in state.dirty_data:
                            reason_counts["dirty_data"] += 1
                        elif uid in state.dirty_state:
                            reason_counts["dirty_state"] += 1
                        else:
                            reason_counts["dirty_descendant"] += 1
                        serialized_existing_bytes += source_storage_size
                        continue
                    copy_groups.append((source_path, payload_path, None))
                    copied_bytes += source_storage_size
        except Exception:
            logger.debug(
                "Falling back to DataTree full-save snapshot",
                exc_info=True,
            )
            return None

        tree = self._workspace_datatree_for_payload_uids(serialize_uids)
        logger.debug(
            "Workspace manifest-first full-save plan: copied %d groups "
            "(%.1f MiB), serialized %d existing groups (%.1f MiB), "
            "reasons=%s, planning %.3f s",
            len(copy_groups),
            copied_bytes / 1024**2,
            len(serialize_uids),
            serialized_existing_bytes / 1024**2,
            dict(reason_counts),
            time.perf_counter() - plan_started_at,
        )
        return _WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=0,
            compression_mode=compression_mode,
            full_tree=tree,
            copy_source=str(workspace_path),
            copy_groups=tuple(copy_groups),
            copy_group_sources=tuple(copy_group_sources),
            serialized_tool_data_references=(
                self._serialized_tool_data_references_from_tree(tree)
            ),
        )

    def _write_full_workspace_file(
        self,
        fname: str | os.PathLike[str],
        *,
        reuse_unchanged_groups: bool = True,
        require_matching_compression: bool = False,
    ) -> None:
        snapshot = self._workspace_full_save_snapshot(
            self._manager._workspace_state.dirty_generation,
            fname=fname,
            reuse_unchanged_groups=reuse_unchanged_groups,
            require_matching_compression=require_matching_compression,
        )
        try:
            workspace_storage._write_full_workspace_tree_file(
                fname,
                snapshot.full_tree,
                snapshot.root_attrs,
                copy_source=snapshot.copy_source,
                copy_groups=snapshot.copy_groups,
                copy_group_sources=snapshot.copy_group_sources,
                compression_mode=snapshot.compression_mode,
            )
            self._controller._commit_saved_tool_data_references(snapshot)
        finally:
            snapshot.close()

    def _workspace_highest_dirty_data_roots(self) -> list[str]:
        dirty_existing = [
            uid
            for uid in self._manager._workspace_state.dirty_data
            if uid in self._manager._tool_graph.nodes
        ]
        dirty_set = set(dirty_existing)
        roots: list[str] = []
        for uid in sorted(
            dirty_existing, key=lambda value: self._workspace_node_path(value)
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

    @classmethod
    def _workspace_manifest_node_uids(
        cls, root_attrs: Mapping[str, typing.Any]
    ) -> frozenset[str]:
        manifest = workspace_format._workspace_manifest_from_attrs(
            typing.cast("Mapping[Hashable, typing.Any]", root_attrs)
        )
        uids: set[str] = set()
        for node in workspace_format._iter_workspace_manifest_node_entries(manifest):
            uid = node.get("uid")
            if uid is not None:
                uids.add(str(uid))
        return frozenset(uids)

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

    def _save_workspace_delta(self, fname: str | os.PathLike[str]) -> None:
        delta_save_count = self._manager._workspace_state.delta_save_count + 1
        snapshot = self._workspace_delta_save_snapshot(
            self._manager._workspace_state.dirty_generation,
            self._workspace_root_attrs_payload(delta_save_count=delta_save_count),
            delta_save_count,
        )
        try:
            workspace_storage._write_workspace_transaction_file(
                fname,
                snapshot.rewrite_groups,
                snapshot.attr_updates,
                snapshot.root_attrs,
                compression_mode=snapshot.compression_mode,
            )
            self._controller._commit_saved_tool_data_references(snapshot)
            self._manager._workspace_state.delta_save_count = snapshot.delta_save_count
            self._manager._workspace_state.set_repack_estimate(
                estimated_obsolete_bytes=snapshot.estimated_obsolete_bytes,
                replacement_delta_count=snapshot.replacement_delta_count,
                known=snapshot.repack_estimate_known,
            )
        finally:
            snapshot.close()

    def _save_workspace_document(
        self,
        fname: str | os.PathLike[str],
        *,
        force_full: bool = False,
        document_access: _WorkspaceDocumentAccess | None = None,
        reuse_unchanged_groups: bool = True,
        require_matching_compression: bool = False,
        mark_clean: bool = True,
    ) -> None:
        if document_access is None:
            _require_itws_workspace_path(fname, _WORKSPACE_SAVE_SUFFIX_ERROR)
            with self._controller._workspace_document_access_context(fname) as access:
                self._save_workspace_document(
                    access.path,
                    force_full=force_full,
                    document_access=access,
                    reuse_unchanged_groups=reuse_unchanged_groups,
                    require_matching_compression=require_matching_compression,
                    mark_clean=mark_clean,
                )
            return

        fname = document_access.path
        _require_itws_workspace_path(fname, _WORKSPACE_SAVE_SUFFIX_ERROR)
        self._manager._workspace_state.saving_depth += 1
        try:
            workspace_storage._recover_workspace_transactions(fname)
            requires_full_save = force_full or self._workspace_requires_full_save(fname)
            if requires_full_save:
                self._write_full_workspace_file(
                    fname,
                    reuse_unchanged_groups=reuse_unchanged_groups,
                    require_matching_compression=require_matching_compression,
                )
                self._manager._workspace_state.delta_save_count = 0
                self._manager._workspace_state.reset_repack_estimate()
                self._manager._workspace_state.schema_version = (
                    workspace_format._current_workspace_schema_version()
                )
            else:
                self._save_workspace_delta(fname)
        finally:
            self._manager._workspace_state.saving_depth -= 1
        if mark_clean:
            self._manager._workspace_state.needs_full_save = False
            self._controller._mark_workspace_clean()

    def _workspace_requires_full_save(self, fname: str | os.PathLike[str]) -> bool:
        return workspace_storage._workspace_requires_full_save(
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
            or self._manager._workspace_state.options_modified
            or self._manager._workspace_state.context_modified
        ) and not self._workspace_has_non_layout_modifications()

    def _workspace_rewrite_group_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, xr.Dataset]]:
        constructor: dict[str, xr.Dataset] = {}
        node = self._manager._tool_graph.nodes[uid]
        node_path = self._workspace_node_path(uid)
        self._serialize_workspace_node(
            constructor, node, node_path, include_children=True
        )
        return node_path, constructor

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

    def _workspace_attr_update_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]] | None:
        constructor: dict[str, xr.Dataset] = {}
        node = self._manager._tool_graph.nodes[uid]
        node_path = self._workspace_node_path(uid)
        payload_path = self._workspace_payload_path(uid)
        pending_attrs = self._pending_workspace_payload_attrs_for_save(node)
        if pending_attrs is not None:
            return (
                payload_path,
                pending_attrs,
                (node_path, constructor),
            )
        self._serialize_workspace_node(
            constructor,
            node,
            node_path,
            include_children=False,
        )
        ds = constructor.get(payload_path)
        if ds is None:
            return None
        return payload_path, dict(ds.attrs), (node_path, constructor)

    def _workspace_delta_save_snapshot(
        self,
        generation: int,
        root_attrs: dict[str, typing.Any],
        delta_save_count: int,
    ) -> _WorkspaceSaveSnapshot:
        state = self._manager._workspace_state
        rewrite_groups: list[tuple[str, dict[str, xr.Dataset]]] = []
        rewritten_uids: set[str] = set()
        for uid in self._workspace_highest_dirty_data_roots():
            rewrite_groups.append(self._workspace_rewrite_group_snapshot(uid))
            rewritten_uids.add(uid)
            rewritten_uids.update(self._manager._iter_descendant_uids(uid))

        manifest_uids = self._workspace_manifest_node_uids(root_attrs)
        for uid in self._workspace_stale_reference_rewrite_uids(manifest_uids):
            if uid in rewritten_uids:
                continue
            rewrite_groups.append(self._workspace_rewrite_group_snapshot(uid))
            rewritten_uids.add(uid)
            rewritten_uids.update(self._manager._iter_descendant_uids(uid))

        attr_updates: list[
            tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]]
        ] = []
        for uid in sorted(self._manager._workspace_state.dirty_state - rewritten_uids):
            if uid not in self._manager._tool_graph.nodes:
                continue
            update = self._workspace_attr_update_snapshot(uid)
            if update is not None:
                attr_updates.append(update)

        estimated_obsolete_bytes = state.estimated_obsolete_bytes
        replacement_delta_count = state.replacement_delta_count
        repack_estimate_known = state.repack_estimate_known
        if repack_estimate_known and rewrite_groups and state.path is not None:
            old_bytes, replaced_group_count = (
                workspace_arrays._workspace_h5_paths_storage_size(
                    state.path,
                    (group_path for group_path, _constructor in rewrite_groups),
                )
            )
            estimated_obsolete_bytes += old_bytes
            if replaced_group_count > 0:
                replacement_delta_count += 1
        root_attrs = workspace_format._workspace_root_attrs_with_repack_estimate(
            root_attrs,
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            repack_estimate_known=repack_estimate_known,
        )

        return _WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=delta_save_count,
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            repack_estimate_known=repack_estimate_known,
            compression_mode=self._workspace_compression_mode(),
            rewrite_groups=tuple(rewrite_groups),
            attr_updates=tuple(attr_updates),
            serialized_tool_data_references=(
                self._serialized_tool_data_references_from_delta(
                    rewrite_groups, attr_updates
                )
            ),
        )

    def _workspace_save_snapshot(
        self, fname: str | os.PathLike[str]
    ) -> _WorkspaceSaveSnapshot:
        self._controller._drain_workspace_deferred_events()
        generation = self._manager._workspace_state.dirty_generation
        self._manager._workspace_state.saving_depth += 1
        try:
            if self._workspace_requires_full_save(fname):
                return self._workspace_full_save_snapshot(generation)
            if self._workspace_layout_only_modified():
                delta_save_count = self._manager._workspace_state.delta_save_count
                root_attrs = self._workspace_root_attrs_payload(
                    delta_save_count=delta_save_count
                )
                return self._workspace_delta_save_snapshot(
                    generation, root_attrs, delta_save_count
                )
            delta_save_count = self._manager._workspace_state.delta_save_count + 1
            root_attrs = self._workspace_root_attrs_payload(
                delta_save_count=delta_save_count
            )
            return self._workspace_delta_save_snapshot(
                generation, root_attrs, delta_save_count
            )
        finally:
            self._manager._workspace_state.saving_depth -= 1

    def _workspace_full_save_snapshot(
        self,
        generation: int,
        *,
        fname: str | os.PathLike[str] | None = None,
        reuse_unchanged_groups: bool = True,
        require_matching_compression: bool = False,
    ) -> _WorkspaceSaveSnapshot:
        compression_mode = self._workspace_compression_mode()
        root_attrs = self._workspace_root_attrs_payload(delta_save_count=0)
        if fname is None:
            fname = self._manager._workspace_state.path
        target_drops_copy_groups = (
            fname is not None
            and workspace_storage._workspace_path_is_likely_network_path(fname)
        )
        if (
            reuse_unchanged_groups
            and fname is not None
            and not target_drops_copy_groups
        ):
            snapshot = self._workspace_full_save_manifest_first_snapshot(
                generation,
                fname,
                root_attrs,
                compression_mode=compression_mode,
                require_matching_compression=require_matching_compression,
            )
            if snapshot is not None:
                return snapshot

        tree, pending_copy_groups = self._workspace_full_save_fallback_tree(
            require_matching_compression=require_matching_compression,
            compression_mode=compression_mode,
        )
        if reuse_unchanged_groups and not target_drops_copy_groups:
            copy_source, copy_groups = self._workspace_full_save_copy_groups(
                tree,
                compression_mode=compression_mode,
                require_matching_compression=require_matching_compression,
            )
        else:
            copy_source, copy_groups = None, ()
        return _WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=0,
            compression_mode=compression_mode,
            full_tree=tree,
            copy_source=copy_source,
            copy_groups=copy_groups,
            copy_group_sources=pending_copy_groups,
            serialized_tool_data_references=(
                self._serialized_tool_data_references_from_tree(
                    tree,
                    exclude_payload_paths=(
                        destination_path
                        for _source_path, destination_path, attrs in copy_groups
                        if attrs is None
                    ),
                )
            ),
        )

    def _workspace_file_repack_snapshot(
        self, generation: int
    ) -> _WorkspaceSaveSnapshot | None:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return None
        if workspace_storage._workspace_path_is_likely_network_path(workspace_path):
            return None
        try:
            root_attrs, copy_groups = workspace_storage._workspace_file_repack_payload(
                workspace_path
            )
        except Exception:
            logger.debug(
                "Skipping shutdown compaction; file-level repack snapshot failed",
                exc_info=True,
            )
            return None
        return _WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=0,
            compression_mode=self._workspace_compression_mode(),
            file_repack=True,
            copy_source=str(workspace_path),
            copy_groups=copy_groups,
        )

    def _workspace_should_repack_before_shutdown(self) -> bool:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return False
        state = self._manager._workspace_state
        if state.repack_estimate_known:
            if state.replacement_delta_count <= 0:
                return False
            estimated_obsolete_bytes = state.estimated_obsolete_bytes
        else:
            try:
                estimated_obsolete_bytes = (
                    workspace_storage._workspace_obsolete_estimate(workspace_path)
                )
            except Exception:
                logger.debug(
                    "Failed to estimate workspace repack benefit before shutdown",
                    exc_info=True,
                )
                return False
        try:
            file_size = pathlib.Path(workspace_path).stat().st_size
        except OSError:
            return False
        if file_size <= 0:
            return False
        return (
            estimated_obsolete_bytes >= _WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES
            and estimated_obsolete_bytes / file_size
            >= _WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO
        )

    def _workspace_full_save_copy_groups(
        self,
        tree: xr.DataTree,
        *,
        compression_mode: WorkspaceCompressionMode | None = None,
        require_matching_compression: bool = True,
    ) -> tuple[str | None, tuple[tuple[str, str, dict[str, typing.Any] | None], ...]]:
        if self._manager._workspace_state.path is None:
            return None, ()
        workspace_path = pathlib.Path(self._manager._workspace_state.path)
        if (
            self._manager._workspace_state.schema_version
            != workspace_format._current_workspace_schema_version()
            or not workspace_path.exists()
        ):
            return None, ()

        try:
            root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(
                workspace_path
            )
        except Exception:
            return None, ()
        schema_version, _delta_save_count, manifest = (
            workspace_format._workspace_file_metadata_from_attrs(root_attrs)
        )
        if (
            schema_version != workspace_format._current_workspace_schema_version()
            or manifest is None
        ):
            return None, ()

        manifest_entries = workspace_format._workspace_manifest_payload_entries(
            manifest
        )
        identities = {
            (uid, kind): payload_path for uid, kind, payload_path in manifest_entries
        }
        copy_groups: list[tuple[str, str, dict[str, typing.Any] | None]] = []
        context = contextlib.nullcontext(None)
        if require_matching_compression and compression_mode is not None:
            context = h5py.File(workspace_path, "r")
        with context as h5_file:
            for uid, node in self._manager._tool_graph.nodes.items():
                if (
                    uid in self._manager._workspace_state.dirty_data
                    or uid in self._manager._workspace_state.dirty_added
                ):
                    continue
                if not node.is_imagetool:
                    if node.pending_workspace_tool_payload is not None:
                        if not self._pending_workspace_tool_references_available(node):
                            continue
                    else:
                        tool = node.tool_window
                        if tool is None or not tool.can_save_and_load():
                            continue
                        if not self._workspace_tool_references_match_current_snapshots(
                            node._workspace_tool_data_references.values()
                        ):
                            continue
                kind = "imagetool" if node.is_imagetool else "tool"
                source_path = identities.get((uid, kind))
                if source_path is None:
                    continue
                payload_path = self._workspace_payload_path(uid)
                try:
                    payload_tree = typing.cast("xr.DataTree", tree[payload_path])
                except KeyError:
                    continue
                payload_ds = payload_tree.to_dataset(inherit=False)
                compression_matches = (
                    h5_file is None
                    or compression_mode is None
                    or workspace_arrays._workspace_h5_group_matches_compression_mode(
                        h5_file,
                        source_path,
                        payload_ds,
                        compression_mode,
                    )
                )
                if (
                    require_matching_compression
                    and compression_mode is not None
                    and not compression_matches
                ):
                    continue
                attrs = None
                if (
                    uid in self._manager._workspace_state.dirty_state
                    or source_path != payload_path
                ):
                    attrs = dict(payload_ds.attrs)
                copy_groups.append((source_path, payload_path, attrs))
        return str(workspace_path), tuple(copy_groups)
