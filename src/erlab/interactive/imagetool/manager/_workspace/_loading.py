"""Load and restore ImageTool Manager workspace documents."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import pathlib
import time
import traceback
import typing
import uuid

import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab
import erlab.interactive._options.core
import erlab.interactive.imagetool.manager._workspace._arrays as workspace_arrays
import erlab.interactive.imagetool.manager._workspace._format as workspace_format
import erlab.interactive.imagetool.manager._workspace._pending as workspace_pending
import erlab.interactive.imagetool.manager._workspace._storage as workspace_storage
import erlab.interactive.imagetool.slicer
import erlab.interactive.imagetool.viewer_linking
from erlab.interactive import _qt_state
from erlab.interactive.imagetool._load_source import _deserialize_loader_kwargs
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME, ImageTool
from erlab.interactive.imagetool._provenance._model import (
    ScriptInputDataRole,
    ToolProvenanceSpec,
    parse_tool_provenance_spec,
    require_live_source_spec,
)
from erlab.interactive.imagetool._provenance._operations import (
    ImageToolSelectionSourceBinding,
)
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ChooseFromWorkspaceManifestDialog,
)
from erlab.interactive.imagetool.manager._widgets import (
    _WORKSPACE_REBIND_KEEP_CHUNKS,
    _strip_workspace_modified_placeholder,
)
from erlab.interactive.imagetool.manager._workspace._format import (
    _require_itws_workspace_path,
)

if typing.TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Hashable,
        Iterable,
        Iterator,
        Mapping,
    )

    import h5py

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._workspace._controller import (
        _WorkspaceController,
    )
    from erlab.interactive.imagetool.manager._workspace._state import (
        _WorkspaceStateSnapshot,
    )
    from erlab.interactive.imagetool.manager._wrapper import (
        _ImageToolWrapper,
        _ManagedWindowNode,
    )
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

logger = logging.getLogger(__name__)
_WORKSPACE_LOAD_TIMING_ENV = "ERLAB_WORKSPACE_LOAD_TIMING"
_WORKSPACE_LOAD_SUFFIX_ERROR = "ImageTool workspace files must use the .itws extension"


class _WorkspaceLoadProfiler:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self._path = None if path is None else os.fspath(path)
        self._start = time.perf_counter()
        self._durations: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._durations[name] = self._durations.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def log_summary(self) -> None:
        detailed = os.environ.get(_WORKSPACE_LOAD_TIMING_ENV) == "1"
        if not detailed and not logger.isEnabledFor(logging.DEBUG):
            return
        total = time.perf_counter() - self._start
        summary = ", ".join(
            f"{name}={duration:.3f}s"
            for name, duration in sorted(
                self._durations.items(), key=lambda item: item[1], reverse=True
            )
        )
        logger.debug(
            "Loaded workspace %s in %.3fs%s%s",
            self._path or "<memory>",
            total,
            ": " if summary else "",
            summary,
        )
        if detailed:
            for name, duration in self._durations.items():
                logger.debug("Workspace load stage %s: %.3fs", name, duration)


@contextlib.contextmanager
def _workspace_load_stage(
    profiler: _WorkspaceLoadProfiler | None, name: str
) -> Iterator[None]:
    if profiler is None:
        yield
    else:
        with profiler.stage(name):
            yield


class _WorkspaceReplaceBackup:
    __slots__ = ("file_path", "snapshot", "tree")

    def __init__(
        self,
        snapshot: _WorkspaceStateSnapshot,
        *,
        tree: xr.DataTree | None = None,
        file_path: pathlib.Path | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.tree = tree
        self.file_path = file_path

    def close(self) -> None:
        if self.tree is not None:
            self.tree.close()


def _workspace_dataset_window_visible(
    ds: xr.Dataset, prefix: str, *, default: bool = True
) -> bool:
    state = _qt_state.parse_qt_window_state(ds.attrs.get(f"{prefix}_window_state"))
    if state is not None:
        return state.visible
    return bool(ds.attrs.get(f"{prefix}_visible", default))


def _workspace_payload_window_visible_h5py(
    fname: str | os.PathLike[str],
    payload_path: str,
    prefix: str,
    *,
    default: bool = True,
) -> bool | None:
    with workspace_arrays._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        obj = h5_file.get(payload_path.strip("/"))
        if not isinstance(obj, h5py.Group):
            return None
        state = _qt_state.parse_qt_window_state(obj.attrs.get(f"{prefix}_window_state"))
        if state is not None:
            return state.visible
        return bool(obj.attrs.get(f"{prefix}_visible", default))


def _workspace_payload_attrs_h5py(
    fname: str | os.PathLike[str],
    payload_path: str,
) -> dict[typing.Hashable, typing.Any] | None:
    with workspace_arrays._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        obj = h5_file.get(payload_path.strip("/"))
        if not isinstance(obj, h5py.Group):
            return None
        return workspace_arrays._h5py_attrs_to_dict(obj.attrs)


def _workspace_provenance_file_stems(
    spec: ToolProvenanceSpec | None,
) -> tuple[str, ...]:
    stems: list[str] = []

    def collect(
        current: ToolProvenanceSpec | None,
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
    provenance_spec: ToolProvenanceSpec | None,
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


class _WorkspaceLoader:
    """Load and materialize workspace payloads for one manager."""

    def __init__(
        self, manager: ImageToolManager, controller: _WorkspaceController
    ) -> None:
        self._manager = manager
        self._controller = controller
        self.pending = workspace_pending._PendingWorkspacePayloads(self, controller)
        self._missing_workspace_colormaps: list[tuple[str, str]] = []
        self._skipped_workspace_nodes: list[tuple[str, str, str, Exception]] = []

    @staticmethod
    def _require_workspace_root_tool_is_figure(
        tool: erlab.interactive.utils.ToolWindow,
    ) -> None:
        """Reject root ToolWindow payloads that do not belong to the figure pane."""
        if tool.manager_collection != "figures":
            raise ValueError("Workspace tool node has no parent")

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

    @classmethod
    def _read_workspace_imagetool_payload_dataset(
        cls,
        workspace_path: str | os.PathLike[str],
        payload_path: str,
        *,
        load_data: bool,
    ) -> xr.Dataset:
        if not load_data:
            opened = workspace_arrays.open_workspace_dataset(
                workspace_path, payload_path, chunks={}
            )
            try:
                return workspace_format._restore_workspace_dataset_attrs(
                    opened.copy(deep=False)
                )
            finally:
                opened.close()

        ds = workspace_arrays._read_workspace_dataset_group_h5py(
            workspace_path,
            payload_path,
            preferred_data_name=_ITOOL_DATA_NAME,
        )
        if ds is not None:
            return ds

        opened = workspace_arrays.open_workspace_dataset(
            workspace_path, payload_path, chunks=None
        )
        try:
            return workspace_format._restore_workspace_dataset_attrs(opened.load())
        finally:
            opened.close()

    @staticmethod
    def _read_workspace_tool_payload_dataset(
        workspace_path: str | os.PathLike[str],
        payload_path: str,
    ) -> xr.Dataset:
        ds = workspace_arrays._read_workspace_dataset_group_h5py(
            workspace_path,
            payload_path,
            preferred_data_name=erlab.interactive.utils._SAVED_TOOL_DATA_NAME,
        )
        if ds is not None:
            return ds

        opened = workspace_arrays.open_workspace_dataset(
            workspace_path, payload_path, chunks=None
        )
        try:
            return workspace_format._restore_workspace_dataset_attrs(opened.load())
        finally:
            opened.close()

    @staticmethod
    def _close_workspace_reference_datasets(
        datasets: Mapping[tuple[pathlib.Path, str], xr.Dataset],
    ) -> None:
        for dataset in tuple(datasets.values()):
            with contextlib.suppress(Exception):
                dataset.close()

    @staticmethod
    def _workspace_reference_key(
        workspace_path: str | os.PathLike[str], payload_path: str
    ) -> tuple[pathlib.Path, str]:
        normalized = workspace_arrays._normalized_file_path(workspace_path)
        return pathlib.Path(
            os.fsdecode(workspace_path) if normalized is None else normalized
        ), payload_path.strip("/")

    @staticmethod
    def _open_workspace_imagetool_reference_dataset(
        workspace_path: str | os.PathLike[str], payload_path: str
    ) -> xr.Dataset:
        return workspace_arrays.open_workspace_dataset(
            workspace_path, payload_path, chunks={}
        )

    def _workspace_imagetool_reference_dataset(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        owner_node: _ImageToolWrapper | _ManagedWindowNode | None = None,
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] | None = None,
    ) -> xr.Dataset:
        pending = node.pending_workspace_memory_payload
        if pending is None:
            raise ValueError("Node has no pending ImageTool workspace payload")
        workspace_path, payload_path = pending
        key = self._workspace_reference_key(workspace_path, payload_path)

        def _open() -> xr.Dataset:
            return self._open_workspace_imagetool_reference_dataset(
                workspace_path, payload_path
            )

        if reference_datasets is not None:
            try:
                return reference_datasets[key]
            except KeyError:
                dataset = _open()
                reference_datasets[key] = dataset
                return dataset
        if owner_node is not None:
            return owner_node._workspace_reference_dataset(key, _open)
        raise RuntimeError("Pending workspace reference data requires an owner")

    @staticmethod
    def _workspace_imagetool_payload_data(ds: xr.Dataset) -> xr.DataArray:
        if _ITOOL_DATA_NAME not in ds:
            raise ValueError("Pending workspace payload has no ImageTool data")
        return ds[_ITOOL_DATA_NAME]

    def _workspace_tool_reference_source_data(
        self,
        target: int | str,
        *,
        data_role: ScriptInputDataRole = "displayed",
        owner_node: _ImageToolWrapper | _ManagedWindowNode | None = None,
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] | None = None,
    ) -> xr.DataArray:
        node = self._manager._node_for_target(target)
        if node.pending_workspace_memory_payload is not None:
            return self.pending._pending_workspace_lazy_source_data(
                node,
                data_role=data_role,
                owner_node=owner_node,
                reference_datasets=reference_datasets,
            )
        return node.data_for_role(data_role)

    def _workspace_tool_restore_references(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        loaded_targets_by_uid: Mapping[str, int | str] | None = None,
        owner_node: _ImageToolWrapper | _ManagedWindowNode | None = None,
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] | None = None,
        resolver_error_types: tuple[type[Exception], ...] = (KeyError, ValueError),
        log_resolver_errors: bool = False,
    ) -> tuple[
        xr.DataArray | None,
        Callable[[Mapping[str, typing.Any]], xr.DataArray | None],
    ]:
        reference_cache: dict[tuple[int | str, ScriptInputDataRole], xr.DataArray] = {}

        def _source_data_for_target(
            target: int | str,
            data_role: ScriptInputDataRole = "displayed",
        ) -> xr.DataArray:
            cache_key = (target, data_role)
            if cache_key in reference_cache:
                return reference_cache[cache_key]
            data = self._workspace_tool_reference_source_data(
                target,
                data_role=data_role,
                owner_node=owner_node,
                reference_datasets=reference_datasets,
            )
            reference_cache[cache_key] = data
            return data

        tool_cls = erlab.interactive.utils.ToolWindow
        tool_data_references = tool_cls._saved_tool_data_references(ds)
        source_parent_data: xr.DataArray | None = None
        if parent_target is not None and any(
            reference.get("kind") == "parent_source"
            for reference in tool_data_references.values()
        ):
            source_parent_data = _source_data_for_target(parent_target)

        def _tool_data_reference_resolver(
            payload: Mapping[str, typing.Any],
        ) -> xr.DataArray | None:
            node_uid = payload.get("node_uid")
            if not isinstance(node_uid, str) or not node_uid:
                return None
            target: int | str = node_uid
            if loaded_targets_by_uid is None:
                if node_uid not in self._manager._tool_graph.nodes:
                    return None
            else:
                target = loaded_targets_by_uid.get(node_uid, node_uid)
            data_role = payload.get("data_role", "displayed")
            if data_role not in {"source", "displayed"}:
                return None
            try:
                return _source_data_for_target(
                    target,
                    typing.cast("ScriptInputDataRole", data_role),
                )
            except resolver_error_types:
                if log_resolver_errors:
                    logger.debug(
                        "Could not resolve saved ToolWindow reference %s",
                        node_uid,
                        exc_info=True,
                    )
                return None

        return source_parent_data, _tool_data_reference_resolver

    @classmethod
    def _workspace_tool_display_name_from_attrs(
        cls, attrs: Mapping[str, typing.Any]
    ) -> str:
        for key in ("tool_display_name", "tool_title"):
            value = workspace_format._decode_workspace_attr_text(attrs.get(key))
            if value:
                value = _strip_workspace_modified_placeholder(value).strip()
                if value:
                    return value
        qualname = workspace_format._decode_workspace_attr_text(
            attrs.get("tool_cls_qualname")
        )
        if qualname:
            qualname = qualname.rsplit(":", maxsplit=1)[-1]
            return qualname.rsplit(".", maxsplit=1)[-1]
        return "ToolWindow"

    @classmethod
    def _workspace_tool_source_state_from_attrs(
        cls, attrs: Mapping[str, typing.Any]
    ) -> _ManagedWindowNode._source_state_type:
        source_state = workspace_format._decode_workspace_attr_text(
            attrs.get(erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR)
        )
        if source_state in {"fresh", "stale", "unavailable"}:
            return typing.cast("_ManagedWindowNode._source_state_type", source_state)
        return "fresh"

    def _root_workspace_imagetool_kwargs(
        self, ds: xr.Dataset, kwargs: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        root_kwargs = dict(kwargs)
        root_kwargs.pop("output_id", None)
        root_kwargs["source_input_ndim"] = typing.cast(
            "int | None",
            ds.attrs.get("manager_node_source_input_ndim"),
        )
        watched_varname = ds.attrs.get("manager_node_watched_varname")
        watched_uid = ds.attrs.get("manager_node_watched_uid")
        if watched_varname is not None and watched_uid is not None:
            root_kwargs["watched_var"] = (str(watched_varname), str(watched_uid))
            root_kwargs["watched_workspace_link_id"] = (
                None
                if ds.attrs.get("manager_node_watched_workspace_link_id") is None
                else str(ds.attrs["manager_node_watched_workspace_link_id"])
            )
            root_kwargs["watched_source_label"] = (
                None
                if ds.attrs.get("manager_node_watched_source_label") is None
                else str(ds.attrs["manager_node_watched_source_label"])
            )
            root_kwargs["watched_source_uid"] = (
                None
                if ds.attrs.get("manager_node_watched_source_uid") is None
                else str(ds.attrs["manager_node_watched_source_uid"])
            )
            # Loaded watched rows stay watched, but are disconnected until
            # a notebook explicitly reconnects them.
            root_kwargs["watched_connected"] = False
        return root_kwargs

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

    def _record_skipped_workspace_node(
        self, node_path: str | None, exc: Exception
    ) -> None:
        node_label = node_path if node_path is not None else "unknown workspace node"
        exc_summary = f"{type(exc).__name__}: {exc}"
        exc_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        logger.warning(
            "Skipping workspace node %s during workspace load",
            node_label,
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={"suppress_ui_alert": True},
        )
        self._skipped_workspace_nodes.append((node_label, exc_summary, exc_text, exc))

    def _raise_no_workspace_windows_loaded(self) -> typing.NoReturn:
        skipped_nodes = self._skipped_workspace_nodes
        details = "\n".join(
            f"- {node_label}: {exc_summary}"
            for node_label, exc_summary, _exc_text, _exc in skipped_nodes
        )
        exc = ValueError(f"No workspace windows could be loaded:\n{details}")
        if skipped_nodes:
            raise exc from skipped_nodes[0][3]
        raise exc

    def _show_skipped_workspace_node_warning(self) -> None:
        if not self._skipped_workspace_nodes:
            return
        skipped_nodes = self._skipped_workspace_nodes
        affected = "\n".join(
            f"- {node_label}: {exc_summary}"
            for node_label, exc_summary, _exc_text, _exc in skipped_nodes
        )
        tracebacks = "\n\n".join(
            f"Workspace node {node_label}\n{exc_text}"
            for node_label, _exc_summary, exc_text, _exc in skipped_nodes
        )
        dialog = erlab.interactive.utils.MessageDialog(
            self._manager,
            title="Workspace Partially Loaded",
            text="Some workspace windows could not be loaded.",
            informative_text=(
                f"The rest of the workspace was loaded. Skipped nodes:\n{affected}"
            ),
            detailed_text=erlab.interactive.utils._format_traceback(tracebacks),
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
            icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
        )
        dialog.exec()

    def _finish_workspace_file_load(self, loaded: bool) -> bool:
        if loaded:
            self._show_missing_workspace_colormap_warning()
            self._show_skipped_workspace_node_warning()
        return loaded

    def _workspace_file_can_restore_replace_backup(self, path: pathlib.Path) -> bool:
        try:
            root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(path)
            schema_version, _delta_save_count, manifest = (
                workspace_format._workspace_file_metadata_from_attrs(root_attrs)
            )
        except Exception:
            logger.debug(
                "Cannot use workspace file rollback backup for %s",
                path,
                exc_info=True,
            )
            return False
        return (
            schema_version == workspace_format._current_workspace_schema_version()
            and manifest is not None
        )

    def _workspace_replace_backup_for_load(
        self,
        incoming_path: str | os.PathLike[str] | None,
    ) -> _WorkspaceReplaceBackup:
        snapshot = self._controller._workspace_state_snapshot()
        current_path = self._manager._workspace_state.path
        incoming_resolved = (
            None if incoming_path is None else pathlib.Path(incoming_path).resolve()
        )
        if (
            current_path is not None
            and incoming_resolved is not None
            and current_path != incoming_resolved
            and not self._manager.is_workspace_modified
            and current_path.exists()
            and self._workspace_file_can_restore_replace_backup(current_path)
        ):
            return _WorkspaceReplaceBackup(snapshot, file_path=current_path)
        return _WorkspaceReplaceBackup(
            snapshot, tree=self._controller.saving._to_datatree()
        )

    def _restore_workspace_state_snapshot(
        self, snapshot: _WorkspaceStateSnapshot
    ) -> None:
        self._manager._tool_graph.restore_uid_counter(snapshot["node_uid_counter"])
        dirty_uids = self._manager._workspace_state.restore(snapshot)
        self._manager._acquisition_context.refresh_ui()
        if self._manager._workspace_state.path is not None:
            self._manager._recent_directory = str(
                self._manager._workspace_state.path.parent
            )
        for uid in tuple(self._manager._tool_graph.nodes):
            self._controller._set_node_window_modified(uid, uid in dirty_uids)
        self._controller._update_workspace_window_title()
        self._controller._refresh_manager_record()

    def _load_workspace_imagetool_dataset(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        node_path: str | None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
        profiler: _WorkspaceLoadProfiler | None = None,
        pending_workspace_memory_payload: tuple[str | os.PathLike[str], str]
        | None = None,
    ) -> int | str:
        with _workspace_load_stage(profiler, "imagetool metadata restore"):
            uid = ds.attrs.get("manager_node_uid")
            provenance_spec = ds.attrs.get("manager_node_provenance_spec")
            live_source_spec = ds.attrs.get("manager_node_live_source_spec")
            live_source_binding = ds.attrs.get("manager_node_live_source_binding")
            parse_provenance_spec = parse_tool_provenance_spec
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
                    parsed_source_spec = require_live_source_spec(
                        parse_provenance_spec(source_payload)
                    )
                except Exception:
                    logger.warning(
                        "Ignoring invalid saved manager source provenance for node %s",
                        uid,
                        exc_info=True,
                    )
            parsed_source_binding = None
            if parsed_source_spec is None and live_source_binding is not None:
                try:
                    binding_payload = typing.cast(
                        "Mapping[str, typing.Any]",
                        json.loads(live_source_binding),
                    )
                    binding_type = ImageToolSelectionSourceBinding
                    parsed_source_binding = binding_type.model_validate(binding_payload)
                except Exception:
                    logger.warning(
                        "Ignoring invalid saved manager source binding for node %s",
                        uid,
                        exc_info=True,
                    )
            replay_source_pending = (
                workspace_format._WORKSPACE_REPLAY_SOURCE_BLOB_NAME in ds
                and pending_workspace_memory_payload is not None
            )
            replay_source_data = None
            if (
                workspace_format._WORKSPACE_REPLAY_SOURCE_BLOB_NAME in ds
                and not replay_source_pending
            ):
                replay_source_data = self._workspace_replay_source_data(ds, uid=uid)
            kwargs: dict[str, typing.Any] = {
                "uid": uid,
                "snapshot_token": ds.attrs.get("manager_node_snapshot_token"),
                "source_snapshot_token": ds.attrs.get(
                    "manager_node_source_snapshot_token"
                ),
                "created_time": ds.attrs.get("manager_node_added_at"),
                "note": ds.attrs.get("manager_node_note"),
                "provenance_spec": parsed_provenance_spec,
                "replay_source_data": replay_source_data,
                "replay_source_pending": replay_source_pending,
                "source_spec": parsed_source_spec,
                "source_binding": (
                    parsed_source_binding if parsed_source_spec is None else None
                ),
                "output_id": ds.attrs.get("manager_node_output_id"),
                "source_auto_update": bool(
                    ds.attrs.get("manager_node_source_auto_update", False)
                ),
                "source_state": typing.cast(
                    "_ManagedWindowNode._source_state_type",
                    ds.attrs.get("manager_node_source_state", "fresh"),
                ),
            }
            window_visible = _workspace_dataset_window_visible(ds, "itool")
            tool_kwargs: dict[str, typing.Any] = {
                "_in_manager": True,
                "_defer_state_refresh": True,
                "_defer_secondary_plots": not window_visible,
            }
            if profiler is not None:
                tool_kwargs["_workspace_load_profiler"] = profiler
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

            if pending_workspace_memory_payload is not None:
                return self.pending._register_pending_workspace_imagetool(
                    ds,
                    parent_target=parent_target,
                    node_path=node_path,
                    kwargs=kwargs,
                    pending_workspace_memory_payload=pending_workspace_memory_payload,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                )

        with _workspace_load_stage(profiler, "imagetool widget restore"):
            tool = ImageTool.from_dataset(ds, **tool_kwargs)

        with _workspace_load_stage(profiler, "imagetool manager registration"):
            if parent_target is not None:
                target = self._manager.add_imagetool_child(
                    tool,
                    parent_target,
                    show=window_visible,
                    **kwargs,
                )
                if pending_workspace_memory_payload is not None:
                    self._manager._child_node(
                        target
                    ).set_pending_workspace_memory_payload(
                        *pending_workspace_memory_payload
                    )
                self._record_workspace_loaded_node_target(
                    ds, target, loaded_targets_by_uid
                )
                return target

            kwargs = self._root_workspace_imagetool_kwargs(ds, kwargs)
            preferred_index: int | None = None
            if node_path is not None and "/" not in node_path:
                with contextlib.suppress(ValueError):
                    preferred_index = int(node_path)
                if preferred_index is not None and preferred_index < 0:
                    preferred_index = None
            target = self._manager.add_imagetool(
                tool,
                show=window_visible,
                index=preferred_index,
                **kwargs,
            )
            if pending_workspace_memory_payload is not None:
                self._manager._tool_graph.root_wrappers[
                    target
                ].set_pending_workspace_memory_payload(
                    *pending_workspace_memory_payload
                )
        self._record_workspace_loaded_node_target(ds, target, loaded_targets_by_uid)
        return target

    @staticmethod
    def _workspace_replay_source_data(
        ds: xr.Dataset,
        *,
        uid: object,
    ) -> xr.DataArray | None:
        blob_name = workspace_format._WORKSPACE_REPLAY_SOURCE_BLOB_NAME
        if blob_name not in ds:
            return None
        try:
            return erlab.interactive.utils._tool_data_from_blob(ds[blob_name])
        except Exception:
            logger.warning(
                "Ignoring invalid saved replay source for node %s",
                uid,
                exc_info=True,
            )
            return None

    def _load_workspace_tool_dataset(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
        profiler: _WorkspaceLoadProfiler | None = None,
        pending_workspace_tool_payload: tuple[str | os.PathLike[str], str]
        | None = None,
    ) -> int | str:
        if pending_workspace_tool_payload is not None:
            self.pending._validate_pending_workspace_tool_dataset(
                ds, parent_target=parent_target
            )
            return self.pending._register_pending_workspace_tool(
                ds,
                parent_target=parent_target,
                pending_workspace_tool_payload=pending_workspace_tool_payload,
                loaded_targets_by_uid=loaded_targets_by_uid,
            )

        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] = {}
        try:
            with _workspace_load_stage(profiler, "tool reference restore"):
                source_parent_data, tool_data_reference_resolver = (
                    self._workspace_tool_restore_references(
                        ds,
                        parent_target=parent_target,
                        loaded_targets_by_uid=loaded_targets_by_uid,
                        reference_datasets=reference_datasets,
                    )
                )

            with _workspace_load_stage(profiler, "tool widget restore"):
                tool: erlab.interactive.utils.ToolWindow = (
                    erlab.interactive.utils.ToolWindow.from_dataset(
                        ds,
                        _source_parent_data=source_parent_data,
                        _tool_data_reference_resolver=tool_data_reference_resolver,
                        _defer_restore_work=True,
                    )
                )
            with _workspace_load_stage(profiler, "tool manager registration"):
                if parent_target is None:
                    self._require_workspace_root_tool_is_figure(tool)
                    target = self._manager.add_figuretool(
                        tool,
                        show=_workspace_dataset_window_visible(ds, "tool"),
                        uid=ds.attrs.get("manager_node_uid"),
                        snapshot_token=ds.attrs.get("manager_node_snapshot_token"),
                        source_snapshot_token=ds.attrs.get(
                            "manager_node_source_snapshot_token"
                        ),
                        created_time=ds.attrs.get("manager_node_added_at"),
                        note=ds.attrs.get("manager_node_note"),
                    )
                else:
                    target = self._manager.add_childtool(
                        tool,
                        parent_target,
                        show=_workspace_dataset_window_visible(ds, "tool"),
                        uid=ds.attrs.get("manager_node_uid"),
                        snapshot_token=ds.attrs.get("manager_node_snapshot_token"),
                        source_snapshot_token=ds.attrs.get(
                            "manager_node_source_snapshot_token"
                        ),
                        created_time=ds.attrs.get("manager_node_added_at"),
                        note=ds.attrs.get("manager_node_note"),
                    )
                    parent_uid = self._manager._node_for_target(parent_target).uid
                    registered_node = self._manager._node_for_target(target)

                    def _source_parent_fetcher() -> xr.DataArray:
                        return self._workspace_tool_reference_source_data(
                            parent_uid, owner_node=registered_node
                        )

                    tool.set_source_parent_fetcher(_source_parent_fetcher)
                registered_node = self._manager._node_for_target(target)
                registered_node._set_workspace_tool_data_references(
                    type(tool)._saved_tool_data_references(ds)
                )
                registered_node._adopt_workspace_reference_datasets(reference_datasets)
                reference_datasets = {}
        except Exception:
            self._close_workspace_reference_datasets(reference_datasets)
            raise
        self._record_workspace_loaded_node_target(ds, target, loaded_targets_by_uid)
        return target

    @staticmethod
    def _workspace_saved_uid_from_dataset(ds: xr.Dataset) -> str | None:
        uid = ds.attrs.get("manager_node_uid")
        if isinstance(uid, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                uid = uid.decode()
        if isinstance(uid, str) and uid:
            return uid
        return None

    def _record_workspace_loaded_node_target(
        self,
        ds: xr.Dataset,
        target: int | str,
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> None:
        if loaded_targets_by_uid is None:
            return
        saved_uid = self._workspace_saved_uid_from_dataset(ds)
        if saved_uid is not None:
            loaded_targets_by_uid[saved_uid] = target

    @classmethod
    def _workspace_manifest_node_entry(
        cls,
        manifest: Mapping[str, typing.Any] | None,
        node_path: str | None,
        kind: typing.Literal["imagetool", "tool"],
    ) -> Mapping[str, typing.Any] | None:
        if node_path is None:
            return None
        for entry in workspace_format._iter_workspace_manifest_node_entries(manifest):
            if entry.get("path") == node_path and entry.get("kind") == kind:
                return entry
        return None

    @classmethod
    def _workspace_manifest_direct_child_keys(
        cls, manifest: Mapping[str, typing.Any] | None, prefix: str
    ) -> list[str]:
        child_keys: list[str] = []
        for entry in workspace_format._iter_workspace_manifest_node_entries(manifest):
            path = entry.get("path")
            if not isinstance(path, str) or not path.startswith(prefix):
                continue
            child_key = path.removeprefix(prefix)
            if "/" not in child_key and child_key not in child_keys:
                child_keys.append(child_key)
        return child_keys

    def _restore_workspace_link_groups(
        self,
        manifest: Mapping[str, typing.Any] | None,
        loaded_targets_by_uid: Mapping[str, int | str],
    ) -> None:
        group_nodes: dict[int, list[_ImageToolWrapper | _ManagedWindowNode]] = {}
        group_colors: dict[int, bool] = {}
        invalid_groups: set[int] = set()
        for entry in workspace_format._iter_workspace_manifest_node_entries(manifest):
            if "link_group" not in entry:
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
            if not node.is_imagetool:
                continue
            current_group_colors = group_colors.get(link_group)
            if current_group_colors is None:
                group_colors[link_group] = link_colors
            elif current_group_colors != link_colors:
                invalid_groups.add(link_group)
                continue
            nodes_for_group = group_nodes.setdefault(link_group, [])
            if all(existing.uid != node.uid for existing in nodes_for_group):
                nodes_for_group.append(node)

        for link_group in sorted(group_nodes):
            if link_group in invalid_groups:
                continue
            link_key = f"workspace:{link_group}"
            for node in group_nodes[link_group]:
                node.set_workspace_link_state(
                    link_key, link_colors=group_colors.get(link_group, True)
                )
            slicers = [
                node.slicer_area
                for node in group_nodes[link_group]
                if node.imagetool is not None and not node.slicer_area.is_linked
            ]
            if len(slicers) <= 1:
                continue
            linker = erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy(
                *slicers,
                link_colors=group_colors.get(link_group, True),
            )
            self._manager._link_registry.append(linker)
            self._manager._sigReloadLinkers.emit()

    def _sync_materialized_workspace_link_group(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> None:
        link_key = node.workspace_link_key
        if link_key is None or node.imagetool is None:
            return
        linked_nodes = [
            candidate
            for candidate in self._manager._tool_graph.nodes.values()
            if (
                candidate.workspace_link_key == link_key
                and candidate.imagetool is not None
            )
        ]
        if len(linked_nodes) <= 1:
            return
        for linked_node in linked_nodes:
            if linked_node.slicer_area.is_linked:
                linked_node.slicer_area.unlink()
        linker = erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy(
            *(linked_node.slicer_area for linked_node in linked_nodes),
            link_colors=node.workspace_link_colors,
        )
        self._manager._link_registry.append(linker)
        self._manager._sigReloadLinkers.emit()

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
            pending_imagetool_payload: tuple[str | os.PathLike[str], str] | None = None
            entry = self._workspace_manifest_node_entry(
                manifest, node_path, "imagetool"
            )
            if entry is not None and workspace_file_path is not None:
                payload_path = f"{node_path}/imagetool"
                if entry.get("data_backing") == "dask":
                    opened = workspace_arrays.open_workspace_dataset(
                        workspace_file_path,
                        payload_path,
                        chunks={},
                    )
                    try:
                        opened_ds = opened.copy(deep=False)
                        ds = workspace_format._restore_workspace_dataset_attrs(
                            opened_ds
                        )
                    finally:
                        opened.close()
                elif entry.get(
                    "data_backing"
                ) == "memory" and not _workspace_payload_window_visible_h5py(
                    workspace_file_path, payload_path, "itool"
                ):
                    try:
                        attrs = _workspace_payload_attrs_h5py(
                            workspace_file_path, payload_path
                        )
                    except Exception:
                        logger.debug(
                            "Failed workspace payload attr read for %s",
                            payload_path,
                            exc_info=True,
                        )
                    else:
                        if attrs is not None and not _workspace_dataset_window_visible(
                            xr.Dataset(attrs=attrs), "itool"
                        ):
                            ds = xr.Dataset(attrs=attrs)
                            pending_imagetool_payload = (
                                workspace_file_path,
                                payload_path,
                            )
            if ds is None:
                ds = workspace_format._restore_workspace_dataset_attrs(
                    typing.cast("xr.DataTree", node_tree["imagetool"])
                    .to_dataset(inherit=False)
                    .load()
                )
            target = self._load_workspace_imagetool_dataset(
                ds,
                parent_target=parent_target,
                node_path=node_path,
                loaded_targets_by_uid=loaded_targets_by_uid,
                pending_workspace_memory_payload=pending_imagetool_payload,
            )
        elif "tool" in node_tree:
            ds = None
            pending_tool_payload: tuple[str | os.PathLike[str], str] | None = None
            entry = self._workspace_manifest_node_entry(manifest, node_path, "tool")
            if entry is not None and workspace_file_path is not None:
                payload_path = f"{node_path}/tool"
                if not _workspace_payload_window_visible_h5py(
                    workspace_file_path, payload_path, "tool"
                ):
                    try:
                        attrs = _workspace_payload_attrs_h5py(
                            workspace_file_path, payload_path
                        )
                    except Exception:
                        logger.debug(
                            "Failed workspace tool attr read for %s",
                            payload_path,
                            exc_info=True,
                        )
                    else:
                        if attrs is not None:
                            ds = xr.Dataset(attrs=attrs)
                            pending_tool_payload = (
                                workspace_file_path,
                                payload_path,
                            )
            if ds is None:
                ds = workspace_format._restore_workspace_dataset_attrs(
                    typing.cast("xr.DataTree", node_tree["tool"])
                    .to_dataset(inherit=False)
                    .load()
                )
            target = self._load_workspace_tool_dataset(
                ds,
                parent_target=parent_target,
                loaded_targets_by_uid=loaded_targets_by_uid,
                pending_workspace_tool_payload=pending_tool_payload,
            )
        else:
            raise ValueError("Workspace node has no supported window payload")

        if "childtools" in node_tree:
            childtools = typing.cast("xr.DataTree", node_tree["childtools"])
            child_keys: list[str] = []
            if manifest is not None and node_path is not None:
                child_keys = self._workspace_manifest_direct_child_keys(
                    manifest, f"{node_path}/childtools/"
                )
            child_keys.extend(
                str(key) for key in childtools if str(key) not in child_keys
            )

            for child_key in child_keys:
                if child_key not in childtools:
                    continue
                child_node = typing.cast("xr.DataTree", childtools[child_key])
                child_item = self._tree_item_child_by_key(selection_item, child_key)
                if (
                    child_item is not None
                    and child_item.checkState(0) == QtCore.Qt.CheckState.Unchecked
                ):
                    continue
                self._load_workspace_node_or_warn(
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

    def _load_workspace_node_or_warn(
        self,
        node_tree: xr.DataTree,
        *,
        parent_target: int | str | None = None,
        selection_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        node_path: str | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int | str | None:
        try:
            return self._load_workspace_node(
                node_tree,
                parent_target=parent_target,
                selection_item=selection_item,
                manifest=manifest,
                node_path=node_path,
                workspace_file_path=workspace_file_path,
                loaded_targets_by_uid=loaded_targets_by_uid,
            )
        except Exception as exc:
            self._record_skipped_workspace_node(node_path, exc)
            return None

    def _load_workspace_roots(
        self,
        tree: xr.DataTree,
        root_keys: Iterable[str],
        *,
        root_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int:
        loaded_count = 0
        for key in root_keys:
            if key not in tree:
                continue
            node = typing.cast("xr.DataTree", tree[key])
            item = self._tree_item_child_by_key(root_item, key)
            if item is None or item.checkState(0) != QtCore.Qt.CheckState.Unchecked:
                target = self._load_workspace_node_or_warn(
                    node,
                    selection_item=item,
                    manifest=manifest,
                    workspace_file_path=workspace_file_path,
                    node_path=key,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                )
                if target is not None:
                    loaded_count += 1
        return loaded_count

    def _load_workspace_figures(
        self,
        tree: xr.DataTree,
        *,
        root_item: QtWidgets.QTreeWidgetItem | None = None,
        manifest: dict[str, typing.Any] | None = None,
        workspace_file_path: str | os.PathLike[str] | None = None,
        loaded_targets_by_uid: dict[str, int | str] | None = None,
    ) -> int:
        if "figures" not in tree:
            return 0
        figures = typing.cast("xr.DataTree", tree["figures"])
        figure_keys = self._workspace_manifest_direct_child_keys(manifest, "figures/")
        figure_keys.extend(str(key) for key in figures if str(key) not in figure_keys)

        loaded_count = 0
        for figure_key in figure_keys:
            if figure_key not in figures:
                continue
            figure_path = f"figures/{figure_key}"
            item = self._tree_item_child_by_key(root_item, figure_path)
            if (
                item is not None
                and item.checkState(0) == QtCore.Qt.CheckState.Unchecked
            ):
                continue
            target = self._load_workspace_node_or_warn(
                typing.cast("xr.DataTree", figures[figure_key]),
                parent_target=None,
                selection_item=item,
                manifest=manifest,
                workspace_file_path=workspace_file_path,
                node_path=figure_path,
                loaded_targets_by_uid=loaded_targets_by_uid,
            )
            if target is not None:
                loaded_count += 1
        return loaded_count

    def _from_h5py_workspace_file(
        self,
        fname: str | os.PathLike[str],
        manifest: Mapping[str, typing.Any],
        *,
        replace: bool,
        mark_dirty: bool,
        selected_paths: set[str] | None = None,
        profiler: _WorkspaceLoadProfiler | None = None,
    ) -> bool:
        if profiler is None:
            profiler = _WorkspaceLoadProfiler(fname)
        self._skipped_workspace_nodes = []
        nodes = manifest.get("nodes", ())
        root_order = manifest.get("root_order", ())
        if not isinstance(nodes, list) or not isinstance(root_order, list):
            raise TypeError("Workspace manifest is missing node ordering")

        entries_by_path: dict[str, Mapping[str, typing.Any]] = {}
        for entry in workspace_format._iter_workspace_manifest_node_entries(manifest):
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
        figure_paths = [
            path
            for path in entries_by_path
            if path.startswith("figures/") and path.count("/") == 1
        ]
        if not root_paths and not figure_paths:
            raise ValueError("Workspace manifest has no loadable root nodes")

        loaded_targets_by_uid: dict[str, int | str] = {}
        child_paths: dict[str, list[str]] = {path: [] for path in entries_by_path}
        for path in entries_by_path:
            if "/childtools/" not in path:
                continue
            parent_path, child_key = path.rsplit("/childtools/", maxsplit=1)
            if "/" not in child_key and parent_path in entries_by_path:
                child_paths[parent_path].append(path)

        if selected_paths is not None:
            root_paths = [path for path in root_paths if path in selected_paths]
            figure_paths = [path for path in figure_paths if path in selected_paths]
            child_paths = {
                parent_path: [
                    child_path for child_path in paths if child_path in selected_paths
                ]
                for parent_path, paths in child_paths.items()
            }

        def _load_xarray_dataset(
            payload_path: str, *, chunks: typing.Any, load: bool
        ) -> xr.Dataset:
            opened = workspace_arrays.open_workspace_dataset(
                fname, payload_path, chunks=chunks
            )
            try:
                if load:
                    return workspace_format._restore_workspace_dataset_attrs(
                        opened.load()
                    )
                return workspace_format._restore_workspace_dataset_attrs(
                    opened.copy(deep=False)
                )
            finally:
                opened.close()

        def _load_dataset(
            payload_path: str, *, entry: Mapping[str, typing.Any], imagetool: bool
        ) -> tuple[xr.Dataset, tuple[str | os.PathLike[str], str] | None]:
            if imagetool and entry.get("data_backing") == "dask":
                return _load_xarray_dataset(payload_path, chunks={}, load=False), None
            if (
                imagetool
                and entry.get("data_backing") == "memory"
                and not _workspace_payload_window_visible_h5py(
                    fname, payload_path, "itool"
                )
            ):
                try:
                    ds = self._read_workspace_imagetool_payload_dataset(
                        fname, payload_path, load_data=False
                    )
                except Exception:
                    logger.debug(
                        "Failed metadata-only workspace payload read for %s",
                        payload_path,
                        exc_info=True,
                    )
                else:
                    if not _workspace_dataset_window_visible(ds, "itool"):
                        return ds, (fname, payload_path)
            if not imagetool and not _workspace_payload_window_visible_h5py(
                fname, payload_path, "tool"
            ):
                try:
                    attrs = _workspace_payload_attrs_h5py(fname, payload_path)
                except Exception:
                    logger.debug(
                        "Failed metadata-only workspace tool read for %s",
                        payload_path,
                        exc_info=True,
                    )
                else:
                    if attrs is not None:
                        return xr.Dataset(attrs=attrs), (fname, payload_path)
            preferred_data_name = (
                _ITOOL_DATA_NAME
                if imagetool
                else erlab.interactive.utils._SAVED_TOOL_DATA_NAME
            )
            try:
                ds = workspace_arrays._read_workspace_dataset_group_h5py(
                    fname,
                    payload_path,
                    preferred_data_name=preferred_data_name,
                )
            except Exception:
                logger.debug(
                    "Failed h5py workspace payload read for %s",
                    payload_path,
                    exc_info=True,
                )
            else:
                if ds is not None:
                    return ds, None

            return _load_xarray_dataset(payload_path, chunks=None, load=True), None

        def _load_path(path: str, parent_target: int | str | None = None) -> int | str:
            entry = entries_by_path[path]
            kind = typing.cast("str", entry["kind"])
            is_imagetool = kind == "imagetool"
            payload_path = f"{path}/{'imagetool' if is_imagetool else 'tool'}"
            with profiler.stage("payload read"):
                ds, pending_payload = _load_dataset(
                    payload_path, entry=entry, imagetool=is_imagetool
                )
            if is_imagetool:
                target = self._load_workspace_imagetool_dataset(
                    ds,
                    parent_target=parent_target,
                    node_path=path,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                    profiler=profiler,
                    pending_workspace_memory_payload=pending_payload,
                )
            else:
                target = self._load_workspace_tool_dataset(
                    ds,
                    parent_target=parent_target,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                    profiler=profiler,
                    pending_workspace_tool_payload=pending_payload,
                )
            for child_path in child_paths[path]:
                _load_path_or_warn(child_path, target)
            return target

        def _load_path_count(path: str) -> int:
            return 1 + sum(
                _load_path_count(child_path) for child_path in child_paths[path]
            )

        load_total = sum(_load_path_count(path) for path in root_paths) + len(
            figure_paths
        )
        load_progress = 0
        load_dialog_ref: typing.Any | None = None

        def _update_load_progress() -> None:
            nonlocal load_progress
            load_progress += 1
            if load_dialog_ref is None or load_total <= 1:
                return
            load_dialog_ref.set_message(
                f"Loading workspace... ({load_progress}/{load_total})"
            )

        def _load_path_or_warn(
            path: str, parent_target: int | str | None = None
        ) -> int | str | None:
            try:
                return _load_path(path, parent_target)
            except Exception as exc:
                self._record_skipped_workspace_node(path, exc)
                return None
            finally:
                _update_load_progress()

        if replace:
            manifest_workspace_link_id = manifest.get("workspace_link_id")
            self._manager._workspace_state.link_id = (
                str(manifest_workspace_link_id)
                if manifest_workspace_link_id
                else uuid.uuid4().hex
            )

        maybe_guard = (
            self._controller._workspace_load_context()
            if not mark_dirty
            else contextlib.nullcontext()
        )
        backup: _WorkspaceReplaceBackup | None = None
        with (
            maybe_guard,
            erlab.interactive.utils.wait_dialog(
                self._manager, "Loading workspace..."
            ) as workspace_load_dialog,
        ):
            load_dialog_ref = workspace_load_dialog
            try:
                if replace:
                    with profiler.stage("rollback backup"):
                        backup = self._workspace_replace_backup_for_load(fname)
                    with profiler.stage("ui catch-up"):
                        self._manager._workspace_state.advance_document_identity()
                        self._manager.remove_all_tools()
                        self._controller._drain_workspace_restore_events()
                loaded_count = 0
                for root_path in root_paths:
                    if _load_path_or_warn(root_path) is not None:
                        loaded_count += 1
                for figure_path in figure_paths:
                    if _load_path_or_warn(figure_path) is not None:
                        loaded_count += 1
                if loaded_count == 0 and self._skipped_workspace_nodes:
                    self._raise_no_workspace_windows_loaded()
                with profiler.stage("link/layout restore"):
                    self._manager._rebase_loaded_workspace_dependency_refs(
                        loaded_targets_by_uid
                    )
                    self._restore_workspace_link_groups(manifest, loaded_targets_by_uid)
                if replace:
                    with profiler.stage("link/layout restore"):
                        self._restore_workspace_layout(manifest)
                        self._restore_workspace_option_overrides(manifest)
                        self._restore_workspace_loader_state(manifest)
                        self._restore_acquisition_context(manifest)
                        self._restore_standalone_apps_state(manifest)
                if not mark_dirty:
                    with profiler.stage("ui catch-up"):
                        self._controller._drain_workspace_restore_events()
            except Exception:
                if backup is not None:
                    try:
                        self._restore_replaced_workspace_backup(backup)
                    except Exception:
                        logger.exception("Failed to restore previous workspace")
                raise
            finally:
                if backup is not None:
                    backup.close()
                profiler.log_summary()
        return True

    def _restore_replaced_workspace(
        self,
        backup_tree: xr.DataTree,
        snapshot: _WorkspaceStateSnapshot,
    ) -> None:
        with self._controller._workspace_load_context():
            self._manager.remove_all_tools()
            self._controller._drain_workspace_restore_events()
            self._load_workspace_roots(backup_tree, [str(key) for key in backup_tree])
            self._load_workspace_figures(backup_tree)
            self._controller._drain_workspace_restore_events()
        self._restore_workspace_state_snapshot(snapshot)

    def _restore_replaced_workspace_file(
        self, path: pathlib.Path, snapshot: _WorkspaceStateSnapshot
    ) -> None:
        with self._controller._workspace_load_context():
            self._manager.remove_all_tools()
            self._controller._drain_workspace_restore_events()
            root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(path)
            schema_version, _delta_save_count, manifest = (
                workspace_format._workspace_file_metadata_from_attrs(root_attrs)
            )
            if (
                schema_version == workspace_format._current_workspace_schema_version()
                and manifest is not None
            ):
                self._from_h5py_workspace_file(
                    path,
                    manifest,
                    replace=False,
                    mark_dirty=False,
                )
            else:
                tree = workspace_arrays.open_workspace_datatree(path, chunks=None)
                self._from_datatree(
                    tree,
                    replace=False,
                    mark_dirty=False,
                    select=False,
                    workspace_file_path=path,
                )
            self._controller._drain_workspace_restore_events()
        self._restore_workspace_state_snapshot(snapshot)

    def _restore_replaced_workspace_backup(
        self, backup: _WorkspaceReplaceBackup
    ) -> None:
        if backup.tree is not None:
            self._restore_replaced_workspace(backup.tree, backup.snapshot)
            return
        if backup.file_path is not None:
            self._restore_replaced_workspace_file(backup.file_path, backup.snapshot)

    def _from_datatree(
        self,
        tree: xr.DataTree,
        *,
        replace: bool = False,
        mark_dirty: bool = True,
        select: bool = True,
        workspace_file_path: str | os.PathLike[str] | None = None,
        profiler: _WorkspaceLoadProfiler | None = None,
    ) -> bool:
        """Restore the state of the manager from a DataTree object."""
        if profiler is None:
            profiler = _WorkspaceLoadProfiler(workspace_file_path)
        self._skipped_workspace_nodes = []
        opened_tree = tree
        try:
            if not self._is_datatree_workspace(tree):
                raise ValueError("Not a valid workspace file")

            schema_version, _delta_save_count, manifest = (
                workspace_format._workspace_file_metadata_from_attrs(tree.attrs)
            )
            match schema_version:
                case 1:
                    tree = self._parse_datatree_compat_v1(tree)
                case 2:
                    tree = self._parse_datatree_compat_v2(tree)
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

            root_keys = workspace_format._workspace_root_keys(tree, manifest)

            dialog: _ChooseFromDataTreeDialog | None = None
            if select:
                with profiler.stage("selection dialog setup"):
                    dialog = _ChooseFromDataTreeDialog(
                        self._manager,
                        tree,
                        root_keys=root_keys,
                    )
                if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    return False

            maybe_guard = (
                self._controller._workspace_load_context()
                if not mark_dirty
                else contextlib.nullcontext()
            )
            backup: _WorkspaceReplaceBackup | None = None
            loaded_targets_by_uid: dict[str, int | str] = {}
            with (
                maybe_guard,
                erlab.interactive.utils.wait_dialog(
                    self._manager, "Loading workspace..."
                ),
            ):
                try:
                    if replace:
                        with profiler.stage("rollback backup"):
                            backup = self._workspace_replace_backup_for_load(
                                workspace_file_path
                            )
                        with profiler.stage("ui catch-up"):
                            self._manager._workspace_state.advance_document_identity()
                            self._manager.remove_all_tools()
                            self._controller._drain_workspace_restore_events()
                    root_item = (
                        None
                        if dialog is None
                        else dialog._tree_widget.invisibleRootItem()
                    )
                    with profiler.stage("payload read"):
                        loaded_count = self._load_workspace_roots(
                            tree,
                            root_keys,
                            root_item=root_item,
                            manifest=manifest,
                            workspace_file_path=workspace_file_path,
                            loaded_targets_by_uid=loaded_targets_by_uid,
                        )
                        loaded_count += self._load_workspace_figures(
                            tree,
                            root_item=root_item,
                            manifest=manifest,
                            workspace_file_path=workspace_file_path,
                            loaded_targets_by_uid=loaded_targets_by_uid,
                        )
                    if loaded_count == 0 and self._skipped_workspace_nodes:
                        self._raise_no_workspace_windows_loaded()
                    with profiler.stage("link/layout restore"):
                        self._manager._rebase_loaded_workspace_dependency_refs(
                            loaded_targets_by_uid
                        )
                        self._restore_workspace_link_groups(
                            manifest, loaded_targets_by_uid
                        )
                    if replace:
                        with profiler.stage("link/layout restore"):
                            self._restore_workspace_layout(manifest)
                            self._restore_workspace_option_overrides(manifest)
                            self._restore_workspace_loader_state(manifest)
                            self._restore_acquisition_context(manifest)
                            self._restore_standalone_apps_state(manifest)
                    if not mark_dirty:
                        with profiler.stage("ui catch-up"):
                            self._controller._drain_workspace_restore_events()
                except Exception:
                    if backup is not None:
                        try:
                            self._restore_replaced_workspace_backup(backup)
                        except Exception:
                            logger.exception("Failed to restore previous workspace")
                    raise
                finally:
                    if backup is not None:
                        backup.close()
            return True
        finally:
            tree.close()
            if tree is not opened_tree:
                opened_tree.close()
            profiler.log_summary()

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

    def _restore_workspace_layout(
        self, manifest: Mapping[str, typing.Any] | None
    ) -> None:
        layout = None if manifest is None else manifest.get("manager_layout")
        self._manager._metadata_editor.restore_layout_payload(
            layout.get("metadata_editor") if isinstance(layout, dict) else None
        )
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

    def _restore_workspace_option_overrides(
        self, manifest: Mapping[str, typing.Any] | None
    ) -> None:
        if manifest is None:
            return
        payload = manifest.get("interactive_option_overrides")
        if not isinstance(payload, dict):
            self._manager._set_workspace_option_overrides({}, mark_dirty=False)
            return
        try:
            state = workspace_format.WorkspaceOptionOverridesState.model_validate(
                payload
            )
        except Exception:
            logger.warning(
                "Ignoring invalid workspace interactive option overrides",
                exc_info=True,
            )
            self._manager._set_workspace_option_overrides({}, mark_dirty=False)
            return
        self._manager._set_workspace_option_overrides(state.overrides, mark_dirty=False)

    def _explorer_loader_state(
        self,
    ) -> tuple[dict[str, dict[str, typing.Any]], dict[str, dict[str, typing.Any]]]:
        loader_kwargs = self._controller._loader_state.explorer_loader_kwargs_by_name
        loader_extensions = (
            self._controller._loader_state.explorer_loader_extensions_by_name
        )
        return (
            {str(name): dict(kwargs) for name, kwargs in loader_kwargs.items()},
            {
                str(name): dict(extensions)
                for name, extensions in loader_extensions.items()
            },
        )

    def _restore_acquisition_context(
        self, manifest: Mapping[str, typing.Any] | None
    ) -> None:
        payload = None if manifest is None else manifest.get("acquisition_context")
        self._manager._acquisition_context.restore_state_payload(payload)

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
            state = workspace_format.WorkspaceLoaderState.model_validate(payload)
        except Exception:
            logger.warning("Ignoring invalid workspace loader state", exc_info=True)
            return

        self._manager._recent_directory = state.recent_directory
        self._manager._recent_name_filter = state.recent_name_filter
        self._manager._recent_loader_kwargs_by_filter = {
            str(name): _deserialize_loader_kwargs(kwargs)
            for name, kwargs in state.manager_loader_kwargs_by_filter.items()
        }
        self._manager._recent_loader_extensions_by_filter = {
            str(name): dict(extensions)
            for name, extensions in state.manager_loader_extensions_by_filter.items()
        }
        explorer_kwargs = {
            str(name): _deserialize_loader_kwargs(kwargs)
            for name, kwargs in state.explorer_loader_kwargs_by_name.items()
        }
        explorer_extensions = {
            str(name): dict(extensions)
            for name, extensions in state.explorer_loader_extensions_by_name.items()
        }
        self._controller._loader_state = state
        self._manager._sync_shared_loader_state(
            explorer_kwargs,
            explorer_extensions,
            apply_explorer=apply_explorer,
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
            state = workspace_format.StandaloneAppsState.model_validate(payload)
        except Exception:
            logger.warning("Ignoring invalid standalone app state", exc_info=True)
            return

        restored_states: dict[str, dict[str, typing.Any]] = {}
        for key, app_state in state.apps.items():
            if key not in self._manager._standalone_app_specs:
                continue
            validated = self._controller._validated_standalone_app_state(key, app_state)
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

    def _workspace_rebind_data_for_uid(
        self,
        fname: str | os.PathLike[str],
        uid: str,
        *,
        chunks: typing.Any,
    ) -> xr.DataArray:
        ds = workspace_arrays.open_workspace_dataset(
            fname, self._controller.saving._workspace_payload_path(uid), chunks=chunks
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
            if not node.is_imagetool:
                continue
            kind, source_paths = node.persistence_data_backing()
            if kind is not None:
                snapshot[node.uid] = (kind, source_paths)
        return snapshot

    def _rebind_workspace_backed_imagetools(
        self,
        fname: str | os.PathLike[str],
        *,
        targets: Iterable[int | str] | None = None,
        chunks: typing.Any = _WORKSPACE_REBIND_KEEP_CHUNKS,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
        old_workspace_path: str | os.PathLike[str] | None = None,
        exclude_uids: Collection[str] = frozenset(),
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
            nodes,
            key=lambda node: self._controller.saving._workspace_node_path(node.uid),
        ):
            tool = node.imagetool
            if tool is None:
                continue
            if node.uid in exclude_uids:
                continue
            if node.pending_workspace_memory_payload is not None:
                continue
            rebind_chunks: typing.Any
            if backing_snapshot is not None:
                backing = backing_snapshot.get(node.uid)
                if backing is None:
                    continue
                kind, source_paths = backing
                if kind == "memory":
                    continue
                if kind == "file_lazy":
                    old_path = workspace_arrays._normalized_file_path(
                        old_workspace_path
                    )
                    if old_path is None or old_path not in source_paths:
                        continue
                rebind_chunks = {} if kind == "dask" else None
                persistence = node.persistence_view(materialize_pending=False)
            else:
                persistence = node.persistence_view(materialize_pending=False)
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
        with self._controller._workspace_load_context():
            for node, state, name, chunks in pending:
                tool = node.imagetool
                if tool is None:
                    continue
                slicer_area = tool.slicer_area
                data = self._workspace_rebind_data_for_uid(
                    fname, node.uid, chunks=chunks
                )
                slicer_area.set_data(data, auto_compute=False)
                slicer_area.state = state
                node._set_name(name, manual=False)

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
        _require_itws_workspace_path(fname, _WORKSPACE_LOAD_SUFFIX_ERROR)
        if replace and self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        previous_missing_colormaps = self._missing_workspace_colormaps
        previous_skipped_nodes = self._skipped_workspace_nodes
        self._missing_workspace_colormaps = []
        self._skipped_workspace_nodes = []
        profiler = _WorkspaceLoadProfiler(fname)
        try:
            with self._controller._workspace_document_access_context(fname) as access:
                with profiler.stage("metadata read"):
                    workspace_storage._recover_workspace_transactions(access.path)
                try:
                    with profiler.stage("metadata read"):
                        root_attrs = workspace_arrays._read_workspace_root_attrs_h5py(
                            access.path
                        )
                        schema_version, delta_save_count, manifest = (
                            workspace_format._workspace_file_metadata_from_attrs(
                                root_attrs
                            )
                        )
                    if (
                        schema_version
                        == workspace_format._current_workspace_schema_version()
                        and manifest is not None
                    ):
                        selected_paths: set[str] | None = None
                        if select:
                            with profiler.stage("selection dialog setup"):
                                dialog = _ChooseFromWorkspaceManifestDialog(
                                    self._manager, manifest
                                )
                            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                                return self._finish_workspace_file_load(False)
                            selected_paths = dialog.selected_paths()
                        loaded = self._from_h5py_workspace_file(
                            access.path,
                            manifest,
                            replace=replace,
                            mark_dirty=mark_dirty,
                            selected_paths=selected_paths,
                            profiler=profiler,
                        )
                        if loaded and associate:
                            (
                                estimated_obsolete_bytes,
                                replacement_delta_count,
                                repack_estimate_known,
                            ) = workspace_format._workspace_manifest_repack_estimate(
                                manifest,
                                delta_save_count=delta_save_count,
                            )
                            self._controller._associate_loaded_workspace_file(
                                access.path,
                                schema_version,
                                native=native,
                                delta_save_count=delta_save_count,
                                estimated_obsolete_bytes=estimated_obsolete_bytes,
                                replacement_delta_count=replacement_delta_count,
                                repack_estimate_known=repack_estimate_known,
                                workspace_access=access,
                                rebind_data=False,
                            )
                            if replace:
                                self._restore_workspace_option_overrides(manifest)
                                self._restore_workspace_loader_state(
                                    manifest, apply_explorer=False
                                )
                        return self._finish_workspace_file_load(loaded)
                except Exception:
                    logger.debug(
                        "Failed h5py workspace load path; falling back to DataTree",
                        exc_info=True,
                    )
                with profiler.stage("metadata read"):
                    tree = workspace_arrays.open_workspace_datatree(
                        access.path, chunks=None
                    )
                    schema_version, delta_save_count, manifest = (
                        workspace_format._workspace_file_metadata_from_attrs(tree.attrs)
                    )
                loaded = self._from_datatree(
                    tree,
                    replace=replace,
                    mark_dirty=mark_dirty,
                    select=select,
                    workspace_file_path=access.path,
                    profiler=profiler,
                )
                if loaded and associate:
                    (
                        estimated_obsolete_bytes,
                        replacement_delta_count,
                        repack_estimate_known,
                    ) = workspace_format._workspace_manifest_repack_estimate(
                        manifest,
                        delta_save_count=delta_save_count,
                    )
                    self._controller._associate_loaded_workspace_file(
                        access.path,
                        schema_version,
                        native=native,
                        delta_save_count=delta_save_count,
                        estimated_obsolete_bytes=estimated_obsolete_bytes,
                        replacement_delta_count=replacement_delta_count,
                        repack_estimate_known=repack_estimate_known,
                        workspace_access=access,
                        rebind_data=False,
                    )
                    if replace:
                        self._restore_workspace_option_overrides(manifest)
                        self._restore_workspace_loader_state(
                            manifest, apply_explorer=False
                        )
                return self._finish_workspace_file_load(loaded)
        finally:
            self._missing_workspace_colormaps = previous_missing_colormaps
            self._skipped_workspace_nodes = previous_skipped_nodes
