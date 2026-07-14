from __future__ import annotations

import base64
import binascii
import collections
import collections.abc
import contextlib
import copy
import functools
import html
import json
import logging
import os
import pathlib
import time
import traceback
import typing
import uuid
import warnings

import numpy as np
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive._options.core
import erlab.interactive.imagetool.slicer
import erlab.interactive.imagetool.viewer_linking
from erlab.interactive import _qt_state
from erlab.interactive.imagetool import _serialization, provenance
from erlab.interactive.imagetool._mainwindow import _ITOOL_DATA_NAME, ImageTool
from erlab.interactive.imagetool.manager import _desktop
from erlab.interactive.imagetool.manager import _workspace as _manager_workspace
from erlab.interactive.imagetool.manager import _xarray as _manager_xarray
from erlab.interactive.imagetool.manager._dialogs import (
    _ChooseFromDataTreeDialog,
    _ChooseFromWorkspaceManifestDialog,
    _is_loader_func,
)
from erlab.interactive.imagetool.manager._widgets import (
    _MAX_RECENT_WORKSPACES,
    _RECENT_WORKSPACES_SETTINGS_KEY,
    _WORKSPACE_REBIND_KEEP_CHUNKS,
    _WORKSPACE_SAVE_SHORTCUT_OBJECT_NAME,
    _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS,
    _curve_preview_data,
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
        Collection,
        Hashable,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    import h5py

    from erlab.interactive._options.schema import WorkspaceCompressionMode
    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.manager._workspace_state import (
        _WorkspaceStateSnapshot,
    )
    from erlab.interactive.imagetool.viewer import ImageSlicerArea
else:
    import lazy_loader as _lazy

    h5py = _lazy.load("h5py")

logger = logging.getLogger(__name__)
_WORKSPACE_LOAD_TIMING_ENV = "ERLAB_WORKSPACE_LOAD_TIMING"
_WORKSPACE_SAVE_SUFFIX_ERROR = "ImageTool workspace documents must be saved as .itws"
_WORKSPACE_LOAD_SUFFIX_ERROR = "ImageTool workspace files must use the .itws extension"
_WORKSPACE_SAVE_SUFFIX_WARNING = "ImageTool Manager saves workspaces as .itws files."
_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_BYTES = 128 * 1024 * 1024
_WORKSPACE_SHUTDOWN_REPACK_MIN_OBSOLETE_RATIO = 0.10
_PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES = 128 * 1024 * 1024
_PENDING_WORKSPACE_PREVIEW_DISPLAY_AXES = (0, 1)
_workspace_obsolete_estimate = _manager_workspace._workspace_obsolete_estimate
_workspace_repack_estimate = _manager_workspace._workspace_manifest_repack_estimate


class _WorkspacePostSaveBindingError(RuntimeError):
    """Raised when a saved workspace cannot be rebound into the live session."""


def _require_itws_workspace_path(fname: str | os.PathLike[str], message: str) -> None:
    if not _manager_workspace._workspace_path_is_itws(fname):
        raise ValueError(message)


def _show_itws_workspace_warning(parent: QtWidgets.QWidget) -> None:
    QtWidgets.QMessageBox.warning(
        parent,
        "Workspace Not Saved",
        _WORKSPACE_SAVE_SUFFIX_WARNING,
    )


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
    with _manager_xarray._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
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
    with _manager_xarray._workspace_file_lock(fname), h5py.File(fname, "r") as h5_file:
        obj = h5_file.get(payload_path.strip("/"))
        if not isinstance(obj, h5py.Group):
            return None
        return _manager_workspace._h5py_attrs_to_dict(obj.attrs)


class _PendingWorkspaceLinkTarget:
    def __init__(
        self, array_slicer: erlab.interactive.imagetool.slicer.ArraySlicer
    ) -> None:
        self.array_slicer = array_slicer

    @property
    def data(self) -> xr.DataArray:
        return self.array_slicer._obj


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
        self._skipped_workspace_nodes: list[tuple[str, str, str, Exception]] = []
        self._loader_state = _manager_workspace.WorkspaceLoaderState()
        self._workspace_window_state_applied: tuple[str, str, bool] | None = None
        self._node_window_state_applied: dict[
            str, tuple[tuple[tuple[int, str], ...], bool]
        ] = {}
        self._pending_node_window_modified: dict[str, bool] = {}
        self._background_save_worker: _manager_workspace._WorkspaceSaveWorker | None = (
            None
        )
        self._background_save_receiver: (
            _manager_workspace._WorkspaceSaveResultReceiver | None
        ) = None
        self._background_save_requested = False
        self._shutdown_compaction_attempted = False

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
            opened = _manager_xarray.open_workspace_dataset(
                workspace_path, payload_path, chunks={}
            )
            try:
                return _manager_workspace._restore_workspace_dataset_attrs(
                    opened.copy(deep=False)
                )
            finally:
                opened.close()

        ds = _manager_workspace._read_workspace_dataset_group_h5py(
            workspace_path,
            payload_path,
            preferred_data_name=_ITOOL_DATA_NAME,
        )
        if ds is not None:
            return ds

        opened = _manager_xarray.open_workspace_dataset(
            workspace_path, payload_path, chunks=None
        )
        try:
            return _manager_workspace._restore_workspace_dataset_attrs(opened.load())
        finally:
            opened.close()

    @staticmethod
    def _read_workspace_tool_payload_dataset(
        workspace_path: str | os.PathLike[str],
        payload_path: str,
    ) -> xr.Dataset:
        ds = _manager_workspace._read_workspace_dataset_group_h5py(
            workspace_path,
            payload_path,
            preferred_data_name=erlab.interactive.utils._SAVED_TOOL_DATA_NAME,
        )
        if ds is not None:
            return ds

        opened = _manager_xarray.open_workspace_dataset(
            workspace_path, payload_path, chunks=None
        )
        try:
            return _manager_workspace._restore_workspace_dataset_attrs(opened.load())
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
        normalized = _manager_xarray._normalized_file_path(workspace_path)
        return pathlib.Path(
            os.fsdecode(workspace_path) if normalized is None else normalized
        ), payload_path.strip("/")

    @staticmethod
    def _open_workspace_imagetool_reference_dataset(
        workspace_path: str | os.PathLike[str], payload_path: str
    ) -> xr.Dataset:
        return _manager_xarray.open_workspace_dataset(
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

    @staticmethod
    def _pending_workspace_data_with_saved_dim_order(
        data: xr.DataArray, attrs: Mapping[typing.Any, typing.Any]
    ) -> xr.DataArray:
        raw_state = attrs.get("itool_state")
        if isinstance(raw_state, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                raw_state = raw_state.decode()
        if not isinstance(raw_state, str):
            return data
        try:
            state = json.loads(raw_state)
        except Exception:
            logger.debug("Ignoring invalid pending ImageTool state", exc_info=True)
            return data
        if not isinstance(state, collections.abc.Mapping):
            return data
        slice_state = state.get("slice")
        if not isinstance(slice_state, collections.abc.Mapping):
            return data
        raw_dims = slice_state.get("dims")
        if not isinstance(raw_dims, (list, tuple)):
            return data

        saved_dims = tuple(str(dim) for dim in raw_dims)
        # Match ArraySlicer dimension promotion before applying its saved order.
        data = erlab.utils.array._make_dims_uniform(data)
        dims_by_text = {str(dim): dim for dim in data.dims}
        if (
            len(saved_dims) != data.ndim
            or len(dims_by_text) != data.ndim
            or set(saved_dims) != set(dims_by_text)
        ):
            logger.debug(
                "Ignoring incompatible saved pending ImageTool dimension order %s "
                "for data dims %s",
                saved_dims,
                data.dims,
            )
            return data
        ordered_dims = tuple(dims_by_text[dim] for dim in saved_dims)
        if ordered_dims == data.dims:
            return data
        try:
            return data.transpose(*ordered_dims, transpose_coords=True)
        except ValueError:
            logger.debug(
                "Could not apply saved pending ImageTool dimension order %s",
                saved_dims,
                exc_info=True,
            )
            return data

    @staticmethod
    def _apply_pending_workspace_filter(
        data: xr.DataArray, filter_payload: object
    ) -> xr.DataArray:
        if filter_payload is None:
            return data
        if not isinstance(filter_payload, dict):
            raise TypeError("Invalid pending filter operation")
        operation = provenance.parse_tool_provenance_operation(filter_payload)
        source_shape = tuple(data.sizes[dim] for dim in data.dims)
        filtered = operation.apply(data, parent_data=data)
        if filtered.ndim != data.ndim or set(filtered.dims) != set(data.dims):
            raise ValueError("Pending filter changed data dimensions")
        expected_shape = tuple(data.sizes[dim] for dim in filtered.dims)
        if filtered.shape != expected_shape:
            raise ValueError("Pending filter changed data shape")
        data = filtered.transpose(*data.dims, transpose_coords=True)
        if data.shape != source_shape:
            raise ValueError("Pending filter result is misaligned")
        return data

    def _pending_workspace_lazy_source_data(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        *,
        owner_node: _ImageToolWrapper | _ManagedWindowNode | None = None,
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] | None = None,
    ) -> xr.DataArray:
        opened = self._workspace_imagetool_reference_dataset(
            node,
            owner_node=owner_node,
            reference_datasets=reference_datasets,
        )
        ds = _manager_workspace._restore_workspace_dataset_attrs(
            opened.copy(deep=False)
        )
        ds = _serialization.restore_private_coords(ds, _ITOOL_DATA_NAME)
        name = None if node.name == "" else node.name
        data = self._workspace_imagetool_payload_data(ds).rename(name)
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            attrs = ds.attrs
        data = self._pending_workspace_data_with_saved_dim_order(data, attrs)
        raw_state = attrs.get("itool_state")
        if isinstance(raw_state, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                raw_state = raw_state.decode()
        if isinstance(raw_state, str):
            try:
                state = json.loads(raw_state)
            except Exception:
                logger.debug("Ignoring invalid pending ImageTool state", exc_info=True)
            else:
                if isinstance(state, collections.abc.Mapping):
                    data = self._apply_pending_workspace_filter(
                        data, state.get("filter_operation")
                    )
        if isinstance(node, _ImageToolWrapper) and node.source_input_ndim == 1:
            data = provenance.mark_promoted_1d_source(data)
        return data.copy(deep=False)

    def _workspace_tool_reference_source_data(
        self,
        target: int | str,
        *,
        owner_node: _ImageToolWrapper | _ManagedWindowNode | None = None,
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] | None = None,
    ) -> xr.DataArray:
        node = self._manager._node_for_target(target)
        if node.pending_workspace_memory_payload is not None:
            return self._pending_workspace_lazy_source_data(
                node,
                owner_node=owner_node,
                reference_datasets=reference_datasets,
            )
        return node.current_source_data()

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
        reference_cache: dict[int | str, xr.DataArray] = {}

        def _source_data_for_target(target: int | str) -> xr.DataArray:
            if target in reference_cache:
                return reference_cache[target]
            data = self._workspace_tool_reference_source_data(
                target,
                owner_node=owner_node,
                reference_datasets=reference_datasets,
            )
            reference_cache[target] = data
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
            try:
                return _source_data_for_target(target)
            except resolver_error_types:
                if log_resolver_errors:
                    logger.debug(
                        "Could not resolve saved ToolWindow reference %s",
                        node_uid,
                        exc_info=True,
                    )
                return None

        return source_parent_data, _tool_data_reference_resolver

    @staticmethod
    def _pending_workspace_data_with_loaded_coords(data: xr.DataArray) -> xr.DataArray:
        loaded_coords = {
            key: coord.copy(deep=False).load() for key, coord in data.coords.items()
        }
        return data.copy(deep=False).assign_coords(loaded_coords)

    @classmethod
    def _pending_workspace_imagetool_info_text(
        cls, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        pending = node.pending_workspace_memory_payload
        if pending is None:
            return None
        workspace_path, payload_path = pending
        try:
            ds = cls._read_workspace_imagetool_payload_dataset(
                workspace_path, payload_path, load_data=False
            )
            try:
                ds = _serialization.restore_private_coords(ds, _ITOOL_DATA_NAME)
                if _ITOOL_DATA_NAME not in ds:
                    return None
                name = None if node.name == "" else node.name
                data = ds[_ITOOL_DATA_NAME].rename(name)
                attrs = node.pending_workspace_payload_attrs
                if attrs is None:
                    attrs = ds.attrs
                data = cls._pending_workspace_data_with_saved_dim_order(data, attrs)
                data = erlab.utils.array._restore_nonuniform_dims(data)
                additional_info = [f"Added {node.added_time_display}"]
                try:
                    metadata_data = cls._pending_workspace_data_with_loaded_coords(data)
                    text = erlab.utils.formatting.format_darr_html(
                        metadata_data,
                        show_size=True,
                        additional_info=additional_info,
                    )
                except Exception:
                    logger.debug(
                        "Failed to load coordinates for pending workspace metadata",
                        exc_info=True,
                    )
                    text = erlab.utils.formatting.format_darr_html(
                        data,
                        show_size=True,
                        load_values=False,
                        additional_info=additional_info,
                    )
                return erlab.interactive.utils._apply_qt_accent_color(text)
            finally:
                ds.close()
        except Exception:
            logger.debug(
                "Failed to read metadata for pending workspace payload %s from %s",
                payload_path,
                workspace_path,
                exc_info=True,
            )
            return None

    @staticmethod
    def _decode_workspace_attr_text(value: object) -> str | None:
        if isinstance(value, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                value = value.decode()
        if isinstance(value, str) and value:
            return value
        return None

    @classmethod
    def _workspace_tool_display_name_from_attrs(
        cls, attrs: Mapping[str, typing.Any]
    ) -> str:
        for key in ("tool_display_name", "tool_title"):
            value = cls._decode_workspace_attr_text(attrs.get(key))
            if value:
                value = _strip_workspace_modified_placeholder(value).strip()
                if value:
                    return value
        qualname = cls._decode_workspace_attr_text(attrs.get("tool_cls_qualname"))
        if qualname:
            qualname = qualname.rsplit(":", maxsplit=1)[-1]
            return qualname.rsplit(".", maxsplit=1)[-1]
        return "ToolWindow"

    @classmethod
    def _workspace_tool_source_state_from_attrs(
        cls, attrs: Mapping[str, typing.Any]
    ) -> _ManagedWindowNode._source_state_type:
        source_state = cls._decode_workspace_attr_text(
            attrs.get(erlab.interactive.utils._TOOL_SOURCE_STATE_ATTR)
        )
        if source_state in {"fresh", "stale", "unavailable"}:
            return typing.cast("_ManagedWindowNode._source_state_type", source_state)
        return "fresh"

    @classmethod
    def _pending_workspace_tool_info_text(
        cls, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        if node.pending_workspace_tool_payload is None:
            return None
        attrs = node.pending_workspace_payload_attrs or {}
        tool_name = cls._workspace_tool_display_name_from_attrs(attrs)
        lines = [f"<p><b>{html.escape(tool_name)}</b></p>"]
        data_name = cls._decode_workspace_attr_text(attrs.get("tool_data_name"))
        if data_name is not None and data_name != "<none-value>":
            lines.append(f"<p>Data: <code>{html.escape(data_name)}</code></p>")
        source_state = cls._decode_workspace_attr_text(attrs.get("tool_source_state"))
        if source_state is not None:
            lines.append(f"<p>Source: {html.escape(source_state)}</p>")
        lines.append(f"<p>Added {html.escape(node.added_time_display)}</p>")
        return erlab.interactive.utils._apply_qt_accent_color("".join(lines))

    @classmethod
    def _pending_workspace_info_text(
        cls, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> str | None:
        match node.pending_workspace_payload_kind:
            case "imagetool":
                return cls._pending_workspace_imagetool_info_text(node)
            case "tool":
                return cls._pending_workspace_tool_info_text(node)
            case _:
                return None

    @staticmethod
    def _pending_workspace_tool_preview_image(
        node: _ImageToolWrapper | _ManagedWindowNode,
    ) -> tuple[float, QtGui.QPixmap] | None:
        if node.pending_workspace_tool_payload is None:
            return None
        attrs = node.pending_workspace_payload_attrs or {}
        encoded = attrs.get("figure_composer_preview_cache_png")
        if isinstance(encoded, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                encoded = encoded.decode()
        if not isinstance(encoded, str) or not encoded:
            return None
        try:
            png_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (ValueError, binascii.Error):
            return None
        pixmap = QtGui.QPixmap()
        if not pixmap.loadFromData(png_bytes, "PNG") or pixmap.isNull():
            return None
        width = pixmap.width()
        if width <= 0:
            return None
        return pixmap.height() / width, pixmap

    @staticmethod
    def _pending_preview_axis_state(
        shape: tuple[int, ...],
        state: Mapping[str, typing.Any],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[bool, ...]] | None:
        slice_state = state.get("slice")
        if not isinstance(slice_state, collections.abc.Mapping):
            return None
        state_bins = slice_state.get("bins")
        state_indices = slice_state.get("indices")
        if not isinstance(state_bins, list) or not state_bins:
            return None
        if not isinstance(state_indices, list):
            state_indices = []

        cursor = int(state.get("current_cursor", 0))
        cursor = min(max(cursor, 0), len(state_bins) - 1)
        bins_for_cursor = state_bins[cursor]
        indices_for_cursor = (
            state_indices[cursor] if cursor < len(state_indices) else []
        )
        if not isinstance(bins_for_cursor, list):
            return None
        if not isinstance(indices_for_cursor, list):
            indices_for_cursor = []

        center_indices = erlab.interactive.imagetool.slicer._center_indices_for_shape(
            shape
        )
        bins: list[int] = []
        indices: list[int] = []
        for axis, size in enumerate(shape):
            try:
                axis_bin = int(bins_for_cursor[axis])
            except (IndexError, TypeError, ValueError):
                axis_bin = 1
            if axis_bin < 1:
                axis_bin = 1
            try:
                axis_index = int(indices_for_cursor[axis])
            except (IndexError, TypeError, ValueError):
                axis_index = center_indices[axis]
            indices.append(
                erlab.interactive.imagetool.slicer._normalized_axis_index(
                    axis_index, size
                )
            )
            bins.append(axis_bin)
        return tuple(indices), tuple(bins), tuple(axis_bin != 1 for axis_bin in bins)

    @staticmethod
    def _pending_preview_dataset_dims(
        dataset: h5py.Dataset, state_dims: tuple[str, ...]
    ) -> tuple[tuple[str, ...], int | None] | None:
        stack_axis: int | None = None
        expected_dims = state_dims
        if len(state_dims) != dataset.ndim:
            if (
                dataset.ndim == 1
                and len(state_dims) == 2
                and state_dims.count("stack_dim") == 1
            ):
                stack_axis = state_dims.index("stack_dim")
                expected_dims = tuple(dim for dim in state_dims if dim != "stack_dim")
            else:
                return None

        dataset_dims: list[str] = []
        used_dims: set[str] = set()
        for axis, dim in enumerate(dataset.dims):
            dim_keys = tuple(str(key) for key in dim)
            candidates: list[str] = []
            for expected_dim in expected_dims:
                if expected_dim in used_dims:
                    continue
                if expected_dim in dim_keys or (
                    expected_dim.endswith("_idx")
                    and expected_dim.removesuffix("_idx") in dim_keys
                ):
                    candidates.append(expected_dim)
            candidates = list(dict.fromkeys(candidates))
            positional_dim = expected_dims[axis]
            if positional_dim in candidates:
                dataset_dim = positional_dim
            elif len(candidates) == 1:
                dataset_dim = candidates[0]
            else:
                return None
            dataset_dims.append(dataset_dim)
            used_dims.add(dataset_dim)
        stored_dims = tuple(dataset_dims)
        if set(expected_dims) != set(stored_dims):
            return None
        return stored_dims, stack_axis

    def _pending_workspace_imagetool_preview_curve(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pending = node.pending_workspace_memory_payload
        if pending is None:
            return None
        attrs = node.pending_workspace_payload_attrs
        workspace_path, payload_path = pending
        try:
            with (
                _manager_xarray._workspace_file_lock(workspace_path),
                h5py.File(workspace_path, "r") as h5_file,
            ):
                group = h5_file.get(payload_path.strip("/"))
                if not isinstance(group, h5py.Group):
                    return None
                dataset = group.get(_ITOOL_DATA_NAME)
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1:
                    return None
                if dataset.dtype.kind not in "biuf":
                    return None

                group_attrs = (
                    _manager_workspace._h5py_attrs_to_dict(group.attrs)
                    if attrs is None
                    else attrs
                )
                state = group_attrs.get("itool_state")
                if isinstance(state, bytes):
                    state = state.decode()
                if isinstance(state, str):
                    state = json.loads(state)
                if not isinstance(state, collections.abc.Mapping):
                    return None
                slice_state = state.get("slice")
                if not isinstance(slice_state, collections.abc.Mapping):
                    return None
                raw_state_dims = slice_state.get("dims")
                if not isinstance(raw_state_dims, (list, tuple)):
                    return None
                state_dims = tuple(str(dim) for dim in raw_state_dims)
                resolved_dims = self._pending_preview_dataset_dims(dataset, state_dims)
                if resolved_dims is None:
                    return None
                stored_dims, promoted_stack_axis = resolved_dims
                if promoted_stack_axis is None:
                    return None

                data_nbytes = int(dataset.size) * int(np.dtype(dataset.dtype).itemsize)
                if data_nbytes > _PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES:
                    return None
                y_values = np.asarray(dataset[()]).reshape(-1)

                x_values: np.ndarray | None = None
                coord_dataset = None
                stored_dim = stored_dims[0]
                coord_names = (stored_dim,)
                if stored_dim.endswith("_idx"):
                    coord_names = (stored_dim, stored_dim.removesuffix("_idx"))
                for coord_name in coord_names:
                    with contextlib.suppress(KeyError, TypeError, RuntimeError):
                        maybe_coord = dataset.dims[0][coord_name]
                        if isinstance(maybe_coord, h5py.Dataset):
                            coord_dataset = maybe_coord
                            break
                if coord_dataset is None:
                    for coord_name in coord_names:
                        maybe_coord = group.get(coord_name)
                        if isinstance(maybe_coord, h5py.Dataset):
                            coord_dataset = maybe_coord
                            break
                if (
                    coord_dataset is not None
                    and coord_dataset.shape == dataset.shape
                    and coord_dataset.dtype.kind in "biuf"
                ):
                    coord_nbytes = int(coord_dataset.size) * int(
                        np.dtype(coord_dataset.dtype).itemsize
                    )
                    if (
                        data_nbytes + coord_nbytes
                        <= _PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES
                    ):
                        x_values = np.asarray(coord_dataset[()]).reshape(-1)
                if x_values is None:
                    x_values = np.arange(y_values.size, dtype=np.float64)
                return _curve_preview_data(x_values, y_values)
        except Exception:
            logger.debug(
                "Failed to render curve preview for pending workspace payload "
                "%s from %s",
                payload_path,
                workspace_path,
                exc_info=True,
            )
            return None

    def _pending_workspace_imagetool_preview_image(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> tuple[float, QtGui.QPixmap] | None:
        pending = node.pending_workspace_memory_payload
        if pending is None:
            return None
        attrs = node.pending_workspace_payload_attrs
        workspace_path, payload_path = pending
        try:
            with (
                _manager_xarray._workspace_file_lock(workspace_path),
                h5py.File(workspace_path, "r") as h5_file,
            ):
                group = h5_file.get(payload_path.strip("/"))
                if not isinstance(group, h5py.Group):
                    return None
                dataset = group.get(_ITOOL_DATA_NAME)
                if not isinstance(dataset, h5py.Dataset):
                    return None
                if dataset.dtype.kind not in "biuf":
                    return None

                group_attrs = (
                    _manager_workspace._h5py_attrs_to_dict(group.attrs)
                    if attrs is None
                    else attrs
                )
                state = group_attrs.get("itool_state")
                if isinstance(state, bytes):
                    state = state.decode()
                if isinstance(state, str):
                    state = json.loads(state)
                if not isinstance(state, collections.abc.Mapping):
                    return None

                slice_state = state.get("slice")
                if not isinstance(slice_state, collections.abc.Mapping):
                    return None
                raw_state_dims = slice_state.get("dims")
                if not isinstance(raw_state_dims, (list, tuple)):
                    return None
                state_dims = tuple(str(dim) for dim in raw_state_dims)
                resolved_dims = self._pending_preview_dataset_dims(dataset, state_dims)
                if resolved_dims is None:
                    return None
                stored_dims, promoted_stack_axis = resolved_dims
                if promoted_stack_axis is not None:
                    return None

                stored_shape = tuple(int(size) for size in dataset.shape)
                if len(stored_shape) < 2:
                    return None
                slicer_shape = tuple(
                    stored_shape[stored_dims.index(dim)] for dim in state_dims
                )
                axis_state = self._pending_preview_axis_state(slicer_shape, state)
                if axis_state is None:
                    return None
                indices, bins, binned = axis_state
                hidden_axes = (
                    erlab.interactive.imagetool.slicer._hidden_axes_for_display(
                        len(slicer_shape),
                        _PENDING_WORKSPACE_PREVIEW_DISPLAY_AXES,
                    )
                )
                slicer_selection, reduction_axes, _any_binned, _all_binned = (
                    erlab.interactive.imagetool.slicer._reduced_axes_selection(
                        slicer_shape,
                        hidden_axes,
                        indices,
                        bins,
                        binned,
                    )
                )
                final_ndim = sum(
                    isinstance(item, slice) for item in slicer_selection
                ) - len(reduction_axes)
                if final_ndim != 2:
                    return None

                stored_selection: list[slice | int] = [slice(None)] * dataset.ndim
                for slicer_axis, item in enumerate(slicer_selection):
                    stored_axis = stored_dims.index(state_dims[slicer_axis])
                    stored_selection[stored_axis] = item

                read_nbytes = int(np.dtype(dataset.dtype).itemsize)
                for size, item in zip(stored_shape, stored_selection, strict=True):
                    if isinstance(item, slice):
                        start = 0 if item.start is None else item.start
                        stop = size if item.stop is None else item.stop
                        read_nbytes *= max(0, stop - start)
                if read_nbytes > _PENDING_WORKSPACE_PREVIEW_READ_LIMIT_BYTES:
                    return None

                data = np.asarray(dataset[tuple(stored_selection)])

            stored_remaining_dims = tuple(
                dim
                for dim, item in zip(stored_dims, stored_selection, strict=True)
                if isinstance(item, slice)
            )
            slicer_remaining_dims = tuple(
                dim
                for dim, item in zip(state_dims, slicer_selection, strict=True)
                if isinstance(item, slice)
            )
            if stored_remaining_dims != slicer_remaining_dims:
                axis_order = tuple(
                    stored_remaining_dims.index(dim) for dim in slicer_remaining_dims
                )
                data = data.transpose(axis_order)
            if reduction_axes:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.filterwarnings(
                        "ignore", r"Mean of empty slice", RuntimeWarning
                    )
                    data = np.nanmean(data, axis=reduction_axes)
            if data.ndim != 2 or data.size == 0:
                return None
            if 0 in _PENDING_WORKSPACE_PREVIEW_DISPLAY_AXES:
                data = data.T
            data = erlab.interactive.imagetool.slicer._display_safe_values(data)

            pixmap = self._workspace_pending_preview_pixmap(data, state)
            if pixmap is None or pixmap.isNull():
                return None
            pixmap = pixmap.transformed(QtGui.QTransform().scale(1.0, -1.0))
            if pixmap.width() <= 0:
                return None
            return pixmap.height() / pixmap.width(), pixmap
        except Exception:
            logger.debug(
                "Failed to render preview for pending workspace payload %s from %s",
                payload_path,
                workspace_path,
                exc_info=True,
            )
            return None

    @staticmethod
    def _workspace_pending_preview_pixmap(
        data: np.ndarray, state: Mapping[str, typing.Any]
    ) -> QtGui.QPixmap | None:
        color = state.get("color", {})
        if not isinstance(color, collections.abc.Mapping):
            color = {}
        image_item = erlab.interactive.colors.BetterImageItem()
        image_item.set_colormap(
            color.get("cmap", "viridis"),
            float(color.get("gamma", 1.0)),
            bool(color.get("reverse", False)),
            high_contrast=bool(color.get("high_contrast", False)),
            zero_centered=bool(color.get("zero_centered", False)),
            update=False,
        )
        levels_locked = bool(color.get("levels_locked", False))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"All-NaN (slice|axis) encountered", RuntimeWarning
            )
            image_item.setImage(data, autoLevels=not levels_locked)
        levels = color.get("levels")
        if levels_locked and isinstance(levels, (list, tuple)) and len(levels) == 2:
            image_item.setLevels((float(levels[0]), float(levels[1])), update=True)
        return image_item.getPixmap()

    @staticmethod
    def _workspace_imagetool_name_from_attrs(
        attrs: Mapping[typing.Any, typing.Any],
    ) -> str:
        name = attrs.get("itool_name", "")
        if isinstance(name, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                name = name.decode()
        return "" if name is None else str(name)

    def _workspace_tool_source_metadata(
        self, attrs: Mapping[str, typing.Any]
    ) -> tuple[
        provenance.ToolProvenanceSpec | None,
        provenance.ImageToolSelectionSourceBinding | None,
        bool,
        _ManagedWindowNode._source_state_type,
    ]:
        source_spec = None
        raw_source_spec = attrs.get(erlab.interactive.utils._TOOL_SOURCE_SPEC_ATTR)
        if raw_source_spec is not None:
            try:
                source_spec = provenance.require_live_source_spec(
                    provenance.parse_tool_provenance_spec(
                        typing.cast(
                            "Mapping[str, typing.Any]",
                            json.loads(raw_source_spec),
                        )
                    )
                )
            except Exception:
                logger.warning(
                    "Ignoring invalid saved tool source provenance",
                    exc_info=True,
                )
        source_binding = None
        if (
            source_spec is None
            and erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR in attrs
        ):
            try:
                source_binding = (
                    provenance.ImageToolSelectionSourceBinding.model_validate(
                        typing.cast(
                            "Mapping[str, typing.Any]",
                            json.loads(
                                attrs[erlab.interactive.utils._TOOL_SOURCE_BINDING_ATTR]
                            ),
                        )
                    )
                )
            except Exception:
                logger.warning(
                    "Ignoring invalid saved tool source binding",
                    exc_info=True,
                )
        return (
            source_spec,
            source_binding,
            bool(attrs.get(erlab.interactive.utils._TOOL_SOURCE_AUTO_UPDATE_ATTR)),
            self._workspace_tool_source_state_from_attrs(attrs),
        )

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

    def _register_pending_workspace_imagetool(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        node_path: str | None,
        kwargs: dict[str, typing.Any],
        pending_workspace_memory_payload: tuple[str | os.PathLike[str], str],
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> int | str:
        name = self._workspace_imagetool_name_from_attrs(ds.attrs)
        if parent_target is not None:
            parent_node = self._manager._node_for_target(parent_target)
            node = _ManagedWindowNode(
                self._manager,
                self._manager._next_node_uid(kwargs.get("uid")),
                parent_node.uid,
                None,
                window_kind="imagetool",
                name=name,
                provenance_spec=kwargs.get("provenance_spec"),
                source_spec=kwargs.get("source_spec"),
                source_binding=kwargs.get("source_binding"),
                source_auto_update=bool(kwargs.get("source_auto_update", False)),
                source_state=typing.cast(
                    "_ManagedWindowNode._source_state_type",
                    kwargs.get("source_state", "fresh"),
                ),
                output_id=kwargs.get("output_id"),
                snapshot_token=kwargs.get("snapshot_token"),
                created_time=kwargs.get("created_time"),
                note=kwargs.get("note"),
            )
            node.set_pending_workspace_memory_payload(
                *pending_workspace_memory_payload,
                payload_attrs=ds.attrs,
            )
            self._manager._register_child_node(node)
            if node.output_id is not None and parent_node.tool_window is not None:
                parent_node.tool_window._register_output_imagetool_target(
                    node.output_id, node.uid
                )
            self._manager.tree_view.childtool_added(node.uid, parent_target)
            self._manager._mark_node_added(node.uid)
            self._record_workspace_loaded_node_target(
                ds, node.uid, loaded_targets_by_uid
            )
            return node.uid

        root_kwargs = self._root_workspace_imagetool_kwargs(ds, kwargs)
        preferred_index: int | None = None
        if node_path is not None and "/" not in node_path:
            with contextlib.suppress(ValueError):
                preferred_index = int(node_path)
            if preferred_index is not None and preferred_index < 0:
                preferred_index = None
        if (
            preferred_index is None
            or preferred_index in self._manager._tool_graph.root_wrappers
        ):
            preferred_index = int(self._manager.next_idx)

        wrapper = _ImageToolWrapper(
            self._manager,
            preferred_index,
            self._manager._next_node_uid(root_kwargs.get("uid")),
            None,
            watched_var=root_kwargs.get("watched_var"),
            watched_workspace_link_id=root_kwargs.get("watched_workspace_link_id"),
            watched_source_label=root_kwargs.get("watched_source_label"),
            watched_source_uid=root_kwargs.get("watched_source_uid"),
            watched_connected=bool(root_kwargs.get("watched_connected", True)),
            source_input_ndim=typing.cast(
                "int | None", root_kwargs.get("source_input_ndim")
            ),
            provenance_spec=root_kwargs.get("provenance_spec"),
            source_spec=root_kwargs.get("source_spec"),
            source_binding=root_kwargs.get("source_binding"),
            source_auto_update=bool(root_kwargs.get("source_auto_update", False)),
            source_state=typing.cast(
                "_ManagedWindowNode._source_state_type",
                root_kwargs.get("source_state", "fresh"),
            ),
            snapshot_token=root_kwargs.get("snapshot_token"),
            created_time=root_kwargs.get("created_time"),
            note=root_kwargs.get("note"),
            name=name,
        )
        wrapper.set_pending_workspace_memory_payload(
            *pending_workspace_memory_payload,
            payload_attrs=ds.attrs,
        )
        self._manager._register_root_wrapper(wrapper)
        self._manager.tree_view.imagetool_added(preferred_index)
        self._manager._mark_node_added(wrapper.uid)
        self._record_workspace_loaded_node_target(
            ds, preferred_index, loaded_targets_by_uid
        )
        return preferred_index

    def _materialize_pending_workspace_payload(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> bool:
        if node.pending_workspace_payload_kind == "tool":
            return self._materialize_pending_workspace_tool_payload(node)

        pending = node.pending_workspace_memory_payload
        if pending is None:
            return True

        workspace_path, payload_path = pending
        origin = self._manager._active_managed_window() or self._manager
        try:
            name = node.name
            internal_workspace_operation = (
                self._manager._workspace_state.loading_depth > 0
                or self._manager._workspace_state.saving_depth > 0
                or self._manager._workspace_state.closing_document
            )
            wait_context = (
                contextlib.nullcontext()
                if internal_workspace_operation
                else erlab.interactive.utils.wait_dialog(
                    origin, "Loading ImageTool data..."
                )
            )
            with wait_context:
                ds = self._read_workspace_imagetool_payload_dataset(
                    workspace_path, payload_path, load_data=True
                )
                name = None if name == "" else name
                ds = ds.copy(deep=False)
                pending_attrs = node.pending_workspace_payload_attrs
                if pending_attrs is not None:
                    ds.attrs.update(pending_attrs)
                ds.attrs["itool_name"] = "" if name is None else name
                ds = self._dataset_without_missing_workspace_colormap(ds, payload_path)
                with self._manager._workspace_load_context():
                    if node.imagetool is None:
                        tool = ImageTool.from_dataset(
                            ds,
                            _in_manager=True,
                            _defer_state_refresh=True,
                            _defer_secondary_plots=True,
                            options_model=(self._manager.effective_interactive_options),
                        )
                        node.window = tool
                        if node.parent_uid is not None:
                            self._manager._parent_node(node).add_child_reference(
                                node.uid, tool
                            )
                    else:
                        state = copy.deepcopy(node.slicer_area.state)
                        data = ds[_ITOOL_DATA_NAME].rename(name)
                        node.slicer_area.set_data(data, auto_compute=False)
                        node.slicer_area.state = state
                    node.clear_pending_workspace_payload()
                    node.update_title()
                    self._sync_materialized_workspace_link_group(node)
                self._manager.tree_view.refresh(node.uid)
                self._manager._update_info(uid=node.uid)
                return True
        except Exception:
            logger.exception(
                "Error while loading pending workspace payload %s from %s",
                payload_path,
                workspace_path,
                extra={"suppress_ui_alert": True},
            )
            self._manager._show_operation_error(
                "Could Not Load ImageTool Data",
                "The saved data for this ImageTool could not be read from the "
                "workspace file. The file may have moved, been deleted, or changed "
                "on disk.",
            )
            return False

    @staticmethod
    def _require_workspace_root_tool_is_figure(
        tool: erlab.interactive.utils.ToolWindow,
    ) -> None:
        if tool.manager_collection != "figures":
            raise ValueError("Workspace tool node has no parent")

    def _materialize_pending_workspace_tool_payload(
        self, node: _ImageToolWrapper | _ManagedWindowNode
    ) -> bool:
        pending = node.pending_workspace_tool_payload
        if pending is None:
            return True

        workspace_path, payload_path = pending
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset] = {}
        try:
            ds = self._read_workspace_tool_payload_dataset(workspace_path, payload_path)
            pending_attrs = node.pending_workspace_payload_attrs
            if pending_attrs is not None:
                ds = ds.copy(deep=False)
                ds.attrs.update(pending_attrs)

            source_parent_data, tool_data_reference_resolver = (
                self._workspace_tool_restore_references(
                    ds,
                    parent_target=node.parent_uid,
                    owner_node=node,
                    reference_datasets=reference_datasets,
                    resolver_error_types=(Exception,),
                    log_resolver_errors=True,
                )
            )

            with self._manager._workspace_load_context():
                tool: erlab.interactive.utils.ToolWindow = (
                    erlab.interactive.utils.ToolWindow.from_dataset(
                        ds,
                        _source_parent_data=source_parent_data,
                        _tool_data_reference_resolver=tool_data_reference_resolver,
                        _defer_restore_work=True,
                    )
                )
                if node.parent_uid is None:
                    self._require_workspace_root_tool_is_figure(tool)
                    node.window = tool
                    if not tool._tool_display_name:
                        tool._tool_display_name = node.name
                    self._manager._configure_materialized_figure_tool(node, tool)
                    self._manager._figure_controller.sync(select_uid=None)
                else:
                    parent = self._manager._node_for_target(node.parent_uid)
                    node.window = tool
                    if not tool._tool_display_name:
                        tool._tool_display_name = parent.name
                    parent_uid = parent.uid

                    def _source_parent_fetcher() -> xr.DataArray:
                        return self._workspace_tool_reference_source_data(
                            parent_uid, owner_node=node
                        )

                    def _input_provenance_parent_fetcher() -> (
                        provenance.ToolProvenanceSpec | None
                    ):
                        return self._manager._node_for_target(
                            parent_uid
                        ).displayed_provenance_spec

                    tool.set_source_parent_fetcher(_source_parent_fetcher)
                    tool.set_input_provenance_parent_fetcher(
                        _input_provenance_parent_fetcher
                    )
                    parent.add_child_reference(node.uid, tool)
                node._set_workspace_tool_data_references(
                    type(tool)._saved_tool_data_references(ds)
                )
                node._adopt_workspace_reference_datasets(reference_datasets)
                reference_datasets = {}
                node.clear_pending_workspace_payload()
        except Exception:
            self._close_workspace_reference_datasets(reference_datasets)
            logger.exception(
                "Error while loading pending workspace ToolWindow payload %s from %s",
                payload_path,
                workspace_path,
                extra={"suppress_ui_alert": True},
            )
            self._manager._show_operation_error(
                "Could Not Open Saved Tool",
                "This saved tool could not be opened from the workspace file. "
                "The tool data may be unavailable or incompatible with this version.",
            )
            return False
        else:
            self._manager.tree_view.refresh(node.uid)
            self._manager._update_info(uid=node.uid)
            return True

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
            loaded = self._manager._load_workspace_file(
                path,
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

    def _open_workspace_after_dirty_prompt(
        self, fname: str | os.PathLike[str], *, native: bool = True
    ) -> bool:
        if self._manager._workspace_state.save_in_progress:
            self._manager._status_bar.showMessage(
                "Workspace save in progress; open after it finishes", 3000
            )
            return False
        path = pathlib.Path(fname).expanduser().resolve()
        return self._run_after_dirty_workspace_saved_or_discarded(
            "Opening a workspace replaces the windows currently in this manager.",
            lambda: self._load_workspace_path(path, native=native),
            native=native,
        )

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
        if not _manager_workspace._workspace_path_is_itws(path):
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
        if path is None or not _manager_workspace._workspace_path_is_itws(path):
            return None
        return path

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
            self._manager._update_workspace_window_title()
            self._manager._refresh_manager_record()
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
        self._manager._update_workspace_window_title()
        self._manager._refresh_manager_record()

    def _repoint_pending_workspace_payloads(
        self,
        old_workspace_path: str | os.PathLike[str],
        new_workspace_path: str | os.PathLike[str],
    ) -> None:
        old_normalized = _manager_xarray._normalized_file_path(old_workspace_path)
        for node in self._manager._tool_graph.nodes.values():
            pending = node.pending_workspace_payload
            kind = node.pending_workspace_payload_kind
            if pending is None or kind is None:
                continue
            pending_workspace_path, payload_path = pending
            if (
                _manager_xarray._normalized_file_path(pending_workspace_path)
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
                self._workspace_payload_path(node.uid),
                payload_attrs=node.pending_workspace_payload_attrs,
            )

    def _saved_tool_payload_dataset_for_rebind(
        self,
        workspace_path: str | os.PathLike[str],
        node: _ManagedWindowNode,
        reference_datasets: dict[tuple[pathlib.Path, str], xr.Dataset],
    ) -> xr.Dataset:
        payload_path = self._workspace_payload_path(node.uid)
        key = self._workspace_reference_key(workspace_path, payload_path)
        try:
            opened = reference_datasets[key]
        except KeyError:
            opened = _manager_xarray.open_workspace_dataset(
                workspace_path, payload_path, chunks={}
            )
            reference_datasets[key] = opened
        ds = _manager_workspace._restore_workspace_dataset_attrs(
            opened.copy(deep=False)
        )
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
                    self._workspace_tool_restore_references(
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
                    self._manager._workspace_load_context(),
                    tool._history_suppressed(),
                ):
                    tool._replace_persistence_data_items(data_items, ds)
                node._set_workspace_tool_data_references(references)
                node._replace_workspace_reference_datasets(reference_datasets)
                reference_datasets = {}
            except Exception as exc:
                self._close_workspace_reference_datasets(reference_datasets)
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
        old_path = _manager_xarray._normalized_file_path(old_workspace_path)
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
        with self._manager._workspace_load_context():
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
            self._manager._rebind_workspace_backed_imagetools(
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
            (
                "Settings modified",
                ("Workspace settings",)
                if self._manager._workspace_state.options_modified
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
    ) -> bool:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
        ):
            return False
        event = _manager_workspace._WorkspaceDirtyEvent(
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
            self._manager._update_workspace_window_title(force=False)
        return dirty_changed

    def _mark_node_added(self, uid: str) -> bool:
        return self._manager._mark_workspace_dirty(
            uid=uid, added=True, structure="Added window"
        )

    def _mark_node_data_dirty(self, uid: str) -> bool:
        return self._manager._mark_workspace_dirty(uid=uid, data=True)

    def _mark_node_state_dirty(self, uid: str) -> bool:
        return self._manager._mark_workspace_dirty(uid=uid, state=True)

    def _has_pending_workspace_linked_slicers(
        self, source: ImageSlicerArea, *, color: bool
    ) -> bool:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return False
        node = self._manager.node_from_slicer_area(source)
        if (
            node is None
            or not node.is_imagetool
            or node.workspace_link_key is None
            or (color and not node.workspace_link_colors)
        ):
            return False
        return any(
            linked_node.uid != node.uid
            and linked_node.is_imagetool
            and linked_node.workspace_link_key == node.workspace_link_key
            and linked_node.pending_workspace_memory_payload is not None
            for linked_node in self._manager._tool_graph.nodes.values()
        )

    def _sync_pending_workspace_linked_slicers(
        self,
        source: ImageSlicerArea,
        funcname: str,
        arguments: dict[str, typing.Any],
        source_dims: tuple[Hashable, ...],
        indices: bool,
        steps: bool,
        color: bool,
        transaction_id: str | None,
        keep_pending: bool,
    ) -> None:
        if not self._has_pending_workspace_linked_slicers(source, color=color):
            return
        node = self._manager.node_from_slicer_area(source)
        if node is None or node.workspace_link_key is None:
            return

        for linked_node in self._manager._tool_graph.nodes.values():
            if (
                linked_node.uid == node.uid
                or not linked_node.is_imagetool
                or linked_node.workspace_link_key != node.workspace_link_key
                or linked_node.pending_workspace_memory_payload is None
            ):
                continue
            if self._apply_pending_workspace_link_operation(
                source,
                linked_node,
                funcname,
                arguments,
                source_dims,
                indices,
                steps,
                transaction_id,
                keep_pending,
            ):
                self._manager._mark_workspace_dirty(uid=linked_node.uid, state=True)

    def _sync_pending_workspace_linked_manual_limits(
        self,
        source: ImageSlicerArea,
        manual_limits: Mapping[str, Sequence[typing.Any]],
    ) -> None:
        if not self._has_pending_workspace_linked_slicers(source, color=False):
            return
        node = self._manager.node_from_slicer_area(source)
        if node is None or node.workspace_link_key is None:
            return

        normalized_limits = self._normalized_pending_workspace_manual_limits(
            manual_limits
        )
        for linked_node in self._manager._tool_graph.nodes.values():
            if (
                linked_node.uid == node.uid
                or not linked_node.is_imagetool
                or linked_node.workspace_link_key != node.workspace_link_key
                or linked_node.pending_workspace_memory_payload is None
            ):
                continue
            if self._update_pending_workspace_manual_limits(
                linked_node, normalized_limits
            ):
                self._manager._mark_workspace_dirty(uid=linked_node.uid, state=True)

    @staticmethod
    def _normalized_pending_workspace_manual_limits(
        manual_limits: Mapping[str, Sequence[typing.Any]],
    ) -> dict[str, list[float]]:
        normalized: dict[str, list[float]] = {}
        for dim, limits in manual_limits.items():
            try:
                normalized[str(dim)] = [float(limits[0]), float(limits[1])]
            except (IndexError, TypeError, ValueError):
                continue
        return normalized

    def _update_pending_workspace_manual_limits(
        self,
        node: _ImageToolWrapper | _ManagedWindowNode,
        manual_limits: Mapping[str, list[float]],
    ) -> bool:
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            return False
        raw_state = attrs.get("itool_state")
        if isinstance(raw_state, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                raw_state = raw_state.decode()
        if not isinstance(raw_state, str):
            return False
        try:
            state = typing.cast("dict[str, typing.Any]", json.loads(raw_state))
        except Exception:
            logger.debug("Ignoring invalid pending ImageTool state", exc_info=True)
            return False
        slice_state = state.get("slice")
        valid_dims: tuple[str, ...] = ()
        if isinstance(slice_state, collections.abc.Mapping):
            raw_dims = slice_state.get("dims")
            if isinstance(raw_dims, (list, tuple)):
                valid_dims = tuple(str(dim) for dim in raw_dims)
        if not valid_dims:
            pending = node.pending_workspace_memory_payload
            if pending is None:
                return False
            workspace_path, payload_path = pending
            try:
                ds = self._read_workspace_imagetool_payload_dataset(
                    workspace_path, payload_path, load_data=False
                )
                try:
                    ds = _serialization.restore_private_coords(ds, _ITOOL_DATA_NAME)
                    if _ITOOL_DATA_NAME not in ds:
                        return False
                    valid_dims = tuple(str(dim) for dim in ds[_ITOOL_DATA_NAME].dims)
                finally:
                    ds.close()
            except Exception:
                logger.debug(
                    "Could not update manual limits for pending workspace payload "
                    "%s from %s",
                    payload_path,
                    workspace_path,
                    exc_info=True,
                )
                return False

        valid_dim_set = set(valid_dims)
        updated_limits = {
            dim: limits for dim, limits in manual_limits.items() if dim in valid_dim_set
        }
        if state.get("manual_limits", {}) == updated_limits:
            return False
        state["manual_limits"] = updated_limits
        updated_attrs = dict(attrs)
        updated_attrs["itool_state"] = json.dumps(state)
        node.update_pending_workspace_payload_attrs(updated_attrs)
        return True

    def _apply_pending_workspace_link_operation(
        self,
        source: ImageSlicerArea,
        node: _ImageToolWrapper | _ManagedWindowNode,
        funcname: str,
        arguments: dict[str, typing.Any],
        source_dims: tuple[Hashable, ...],
        indices: bool,
        steps: bool,
        transaction_id: str | None,
        keep_pending: bool,
    ) -> bool:
        attrs = node.pending_workspace_payload_attrs
        if attrs is None:
            return False
        raw_state = attrs.get("itool_state")
        if isinstance(raw_state, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                raw_state = raw_state.decode()
        if not isinstance(raw_state, str):
            return False
        try:
            state = typing.cast("dict[str, typing.Any]", json.loads(raw_state))
        except Exception:
            logger.debug("Ignoring invalid pending ImageTool state", exc_info=True)
            return False

        pending = node.pending_workspace_memory_payload
        if pending is None:
            return False
        workspace_path, payload_path = pending
        array_slicer: erlab.interactive.imagetool.slicer.ArraySlicer | None = None
        try:
            ds = self._read_workspace_imagetool_payload_dataset(
                workspace_path, payload_path, load_data=False
            )
            try:
                ds = _serialization.restore_private_coords(ds, _ITOOL_DATA_NAME)
                if _ITOOL_DATA_NAME not in ds:
                    return False
                target_data = ds[_ITOOL_DATA_NAME]
                target_data = self._pending_workspace_data_with_saved_dim_order(
                    target_data, attrs
                )
                array_slicer = erlab.interactive.imagetool.slicer.ArraySlicer(
                    target_data,
                    self._manager,
                    display_value_abs_limit=source.array_slicer.display_value_abs_limit,
                )
                array_slicer.state = copy.deepcopy(state["slice"])
                target = _PendingWorkspaceLinkTarget(array_slicer)
                converter = erlab.interactive.imagetool.viewer_linking.SlicerLinkProxy()
                converted_args = converter.convert_args(
                    source,
                    typing.cast("typing.Any", target),
                    dict(arguments),
                    source_dims,
                    indices,
                    steps,
                    transaction_id,
                    keep_pending,
                )
                if converted_args is None:
                    return False
                for key in (
                    "__slicer_skip_sync",
                    "__slicer_transaction_id",
                    "__slicer_keep_pending",
                ):
                    converted_args.pop(key, None)
                original_state_json = json.dumps(state)
                if not self._update_pending_link_state_for_operation(
                    state, array_slicer, source, funcname, converted_args
                ):
                    return False
                state_json = json.dumps(state)
                if state_json == original_state_json:
                    return False
                updated_attrs = dict(attrs)
                updated_attrs["itool_state"] = state_json
                node.update_pending_workspace_payload_attrs(updated_attrs)
                return True
            finally:
                ds.close()
        except Exception:
            logger.debug(
                "Could not apply linked operation %s to pending workspace payload %s "
                "from %s",
                funcname,
                payload_path,
                workspace_path,
                exc_info=True,
            )
            return False
        finally:
            if array_slicer is not None:
                array_slicer.deleteLater()

    def _update_pending_link_state_for_operation(
        self,
        state: dict[str, typing.Any],
        array_slicer: erlab.interactive.imagetool.slicer.ArraySlicer,
        source: ImageSlicerArea,
        funcname: str,
        arguments: dict[str, typing.Any],
    ) -> bool:
        if funcname in {"refresh", "refresh_current"}:
            return False
        if funcname == "view_all":
            state["manual_limits"] = {}
            return True
        if funcname == "center_all_cursors":
            for cursor in range(array_slicer.n_cursors):
                array_slicer.center_cursor(cursor, update=False)
            state["slice"] = array_slicer.state
            return True
        if funcname == "center_cursor":
            cursor = int(state.get("current_cursor", 0)) % array_slicer.n_cursors
            array_slicer.center_cursor(cursor, update=False)
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_current_cursor":
            state["current_cursor"] = int(arguments["cursor"]) % array_slicer.n_cursors
            return True
        if funcname == "set_axis_inverted":
            dim = str(arguments["dim"])
            if dim not in {str(dim_name) for dim_name in array_slicer._obj.dims}:
                return False
            axis_inversions = dict(state.get("axis_inversions", {}))
            if bool(arguments["inverted"]):
                axis_inversions[dim] = True
            else:
                axis_inversions.pop(dim, None)
            state["axis_inversions"] = axis_inversions
            return True
        if funcname == "swap_axes":
            array_slicer.swap_axes(int(arguments["ax1"]), int(arguments["ax2"]))
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_index":
            cursor = self._pending_link_cursor_argument(state, arguments)
            array_slicer.set_index(
                cursor,
                int(arguments["axis"]),
                int(arguments["value"]),
                update=False,
            )
            state["slice"] = array_slicer.state
            return True
        if funcname == "step_index":
            cursor = self._pending_link_cursor_argument(state, arguments)
            array_slicer.step_index(
                cursor,
                int(arguments["axis"]),
                int(arguments["value"]),
                update=False,
            )
            state["slice"] = array_slicer.state
            return True
        if funcname == "step_index_all":
            for cursor in range(array_slicer.n_cursors):
                array_slicer.step_index(
                    cursor,
                    int(arguments["axis"]),
                    int(arguments["value"]),
                    update=False,
                )
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_value":
            cursor = self._pending_link_cursor_argument(state, arguments)
            array_slicer.set_value(
                cursor,
                int(arguments["axis"]),
                float(arguments["value"]),
                update=False,
                uniform=bool(arguments.get("uniform", False)),
            )
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_bin":
            cursor = self._pending_link_cursor_argument(state, arguments)
            bins: list[int | None] = [None] * array_slicer._obj.ndim
            bins[int(arguments["axis"])] = int(arguments["value"])
            array_slicer.set_bins(cursor, bins, update=False)
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_bin_all":
            bins = [None] * array_slicer._obj.ndim
            bins[int(arguments["axis"])] = int(arguments["value"])
            for cursor in range(array_slicer.n_cursors):
                array_slicer.set_bins(cursor, bins, update=False)
            state["slice"] = array_slicer.state
            return True
        if funcname == "add_cursor":
            current_cursor = (
                int(state.get("current_cursor", 0)) % array_slicer.n_cursors
            )
            array_slicer.add_cursor(current_cursor, update=False)
            cursor_colors = list(state.get("cursor_colors", ()))
            color = arguments.get("color")
            if color is None:
                cursor_colors.append(
                    self._pending_link_default_cursor_color(
                        source, array_slicer.n_cursors - 1, cursor_colors
                    )
                )
            else:
                cursor_colors.append(QtGui.QColor(color).name())
            state["cursor_colors"] = cursor_colors
            state["current_cursor"] = array_slicer.n_cursors - 1
            state["slice"] = array_slicer.state
            return True
        if funcname == "remove_cursor":
            if array_slicer.n_cursors == 1:
                return False
            index = int(arguments["index"]) % array_slicer.n_cursors
            array_slicer.remove_cursor(index, update=False)
            cursor_colors = list(state.get("cursor_colors", ()))
            if index < len(cursor_colors):
                cursor_colors.pop(index)
            current_cursor = int(state.get("current_cursor", 0))
            if current_cursor == index:
                if index == 0:
                    current_cursor = 1
                current_cursor -= 1
            elif current_cursor > index:
                current_cursor -= 1
            state["cursor_colors"] = cursor_colors
            state["current_cursor"] = max(0, current_cursor)
            state["slice"] = array_slicer.state
            return True
        if funcname == "set_colormap":
            self._update_pending_link_colormap_state(state, arguments)
            return True
        if funcname == "toggle_snap":
            slice_state = dict(state["slice"])
            value = arguments.get("value")
            slice_state["snap_to_data"] = (
                not bool(slice_state.get("snap_to_data", False))
                if value is None
                else bool(value)
            )
            state["slice"] = slice_state
            return True
        return False

    @staticmethod
    def _pending_link_cursor_argument(
        state: Mapping[str, typing.Any], arguments: Mapping[str, typing.Any]
    ) -> int:
        cursor = arguments.get("cursor")
        if cursor is None:
            cursor = state.get("current_cursor", 0)
        return int(cursor)

    @staticmethod
    def _pending_link_default_cursor_color(
        source: ImageSlicerArea, index: int, cursor_colors: list[str]
    ) -> str:
        colors = [QtGui.QColor(color).name() for color in source.COLORS]
        color = colors[index % len(colors)]
        while color in cursor_colors:
            color = colors[index % len(colors)]
            if len(cursor_colors) + 1 > len(colors):
                break
            index += 1
        return color

    @staticmethod
    def _update_pending_link_colormap_state(
        state: dict[str, typing.Any], arguments: Mapping[str, typing.Any]
    ) -> None:
        color_state = dict(state.get("color", {}))
        for key in ("cmap", "gamma", "reverse", "high_contrast", "zero_centered"):
            value = arguments.get(key)
            if value is not None:
                color_state[key] = value
        levels_locked = arguments.get("levels_locked")
        if levels_locked is not None:
            color_state["levels_locked"] = bool(levels_locked)
        levels = arguments.get("levels")
        if levels is not None:
            color_state["levels"] = [float(levels[0]), float(levels[1])]
        state["color"] = color_state

    def _mark_tool_info_dirty(self, uid: str) -> bool:
        if uid not in self._manager._workspace_state.dirty_state:
            return self._manager._mark_node_state_dirty(uid)
        return False

    def _mark_workspace_structure_dirty(self, reason: str) -> bool:
        return self._manager._mark_workspace_dirty(structure=reason)

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
            self._manager._update_workspace_window_title(force=False)

    def _mark_workspace_options_dirty(self) -> None:
        if (
            self._manager._workspace_state.loading_depth > 0
            or self._manager._workspace_state.saving_depth > 0
            or self._manager._workspace_state.closing_document
        ):
            return
        if self._manager._workspace_state.mark_options_dirty():
            self._manager._update_workspace_window_title(force=False)

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
            self._apply_workspace_dirty_event(event)
        self._manager._workspace_state.dirty_events = retained_events
        self._manager._update_workspace_window_title()

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

    def _workspace_file_can_restore_replace_backup(self, path: pathlib.Path) -> bool:
        try:
            root_attrs = _manager_workspace._read_workspace_root_attrs_h5py(path)
            schema_version, _delta_save_count, manifest = (
                _manager_workspace._workspace_file_metadata_from_attrs(root_attrs)
            )
        except Exception:
            logger.debug(
                "Cannot use workspace file rollback backup for %s",
                path,
                exc_info=True,
            )
            return False
        return (
            schema_version == _manager_workspace._current_workspace_schema_version()
            and manifest is not None
        )

    def _workspace_replace_backup_for_load(
        self,
        incoming_path: str | os.PathLike[str] | None,
    ) -> _WorkspaceReplaceBackup:
        snapshot = self._manager._workspace_state_snapshot()
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
        return _WorkspaceReplaceBackup(snapshot, tree=self._manager._to_datatree())

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
        if node.note:
            ds.attrs["manager_node_note"] = node.note
        else:
            ds.attrs.pop("manager_node_note", None)
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
                self._manager._tool_graph.nodes
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
        for uid in list(self._manager._tool_graph.figure_uids):
            node = self._manager._tool_graph.nodes.get(uid)
            if not isinstance(node, _ManagedWindowNode):
                continue
            self._manager._serialize_workspace_node(
                constructor,
                node,
                f"figures/{uid}",
                include_children=False,
            )
            if close:
                self._manager._remove_childtool(uid)
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
        profiler: _WorkspaceLoadProfiler | None = None,
        pending_workspace_memory_payload: tuple[str | os.PathLike[str], str]
        | None = None,
    ) -> int | str:
        with _workspace_load_stage(profiler, "imagetool metadata restore"):
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
            if parsed_source_spec is None and live_source_binding is not None:
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
                "note": ds.attrs.get("manager_node_note"),
                "provenance_spec": parsed_provenance_spec,
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
                return self._register_pending_workspace_imagetool(
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

    def _register_pending_workspace_tool(
        self,
        ds: xr.Dataset,
        *,
        parent_target: int | str | None,
        pending_workspace_tool_payload: tuple[str | os.PathLike[str], str],
        loaded_targets_by_uid: dict[str, int | str] | None,
    ) -> int | str:
        uid = self._manager._next_node_uid(self._workspace_saved_uid_from_dataset(ds))
        attrs = dict(ds.attrs)
        name = self._workspace_tool_display_name_from_attrs(attrs)
        if parent_target is None:
            node = _ManagedWindowNode(
                self._manager,
                uid,
                None,
                None,
                window_kind="tool",
                name=name,
                snapshot_token=attrs.get("manager_node_snapshot_token"),
                created_time=attrs.get("manager_node_added_at"),
                note=attrs.get("manager_node_note"),
            )
            self._manager._register_figure_node(node)
            self._manager._figure_controller.sync(select_uid=None)
        else:
            parent_node = self._manager._node_for_target(parent_target)
            node = _ManagedWindowNode(
                self._manager,
                uid,
                parent_node.uid,
                None,
                window_kind="tool",
                name=name,
                snapshot_token=attrs.get("manager_node_snapshot_token"),
                created_time=attrs.get("manager_node_added_at"),
                note=attrs.get("manager_node_note"),
            )
            self._manager._register_child_node(node)
            self._manager.tree_view.childtool_added(node.uid, parent_target)
        source_spec, source_binding, auto_update, source_state = (
            self._workspace_tool_source_metadata(attrs)
        )
        node.set_restored_source_binding_metadata(
            source_spec,
            source_binding,
            auto_update=auto_update,
            state=source_state,
        )
        node.set_pending_workspace_payload(
            "tool",
            *pending_workspace_tool_payload,
            payload_attrs=attrs,
        )
        self._manager._mark_node_added(node.uid)
        self._record_workspace_loaded_node_target(ds, node.uid, loaded_targets_by_uid)
        return node.uid

    @staticmethod
    def _validate_pending_workspace_tool_dataset(
        ds: xr.Dataset, *, parent_target: int | str | None
    ) -> None:
        tool_cls = erlab.interactive.utils.ToolWindow._saved_tool_class_from_dataset(ds)
        if not tool_cls.can_save_and_load():
            raise TypeError("Saved tool class cannot be restored")
        tool_state = ds.attrs.get("tool_state")
        if not isinstance(tool_state, str):
            raise TypeError("Saved tool dataset is missing a valid tool state")
        tool_cls.StateModel.model_validate_json(tool_state)
        tool_cls._saved_tool_data_references(ds)
        if parent_target is None and tool_cls.manager_collection != "figures":
            raise ValueError("Workspace tool node has no parent")

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
            self._validate_pending_workspace_tool_dataset(
                ds, parent_target=parent_target
            )
            return self._register_pending_workspace_tool(
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
                    if tool.manager_collection != "figures":
                        raise ValueError("Workspace tool node has no parent")  # noqa: TRY301
                    target = self._manager.add_figuretool(
                        tool,
                        show=_workspace_dataset_window_visible(ds, "tool"),
                        uid=ds.attrs.get("manager_node_uid"),
                        snapshot_token=ds.attrs.get("manager_node_snapshot_token"),
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
        saved_uid = self._manager._workspace_saved_uid_from_dataset(ds)
        if saved_uid is not None:
            loaded_targets_by_uid[saved_uid] = target

    @staticmethod
    def _iter_workspace_manifest_node_entries(
        manifest: Mapping[str, typing.Any] | None,
    ) -> Iterator[Mapping[str, typing.Any]]:
        if manifest is None:
            return
        nodes = manifest.get("nodes", ())
        if not isinstance(nodes, list):
            return
        for entry in nodes:
            if isinstance(entry, collections.abc.Mapping):
                yield entry

    @classmethod
    def _workspace_manifest_node_entry(
        cls,
        manifest: Mapping[str, typing.Any] | None,
        node_path: str | None,
        kind: typing.Literal["imagetool", "tool"],
    ) -> Mapping[str, typing.Any] | None:
        if node_path is None:
            return None
        for entry in cls._iter_workspace_manifest_node_entries(manifest):
            if entry.get("path") == node_path and entry.get("kind") == kind:
                return entry
        return None

    @classmethod
    def _workspace_manifest_direct_child_keys(
        cls, manifest: Mapping[str, typing.Any] | None, prefix: str
    ) -> list[str]:
        child_keys: list[str] = []
        for entry in cls._iter_workspace_manifest_node_entries(manifest):
            path = entry.get("path")
            if not isinstance(path, str) or not path.startswith(prefix):
                continue
            child_key = path.removeprefix(prefix)
            if "/" not in child_key and child_key not in child_keys:
                child_keys.append(child_key)
        return child_keys

    @classmethod
    def _workspace_manifest_payload_entries(
        cls, manifest: Mapping[str, typing.Any] | None
    ) -> list[tuple[str, str, str]]:
        entries: list[tuple[str, str, str]] = []
        for entry in cls._iter_workspace_manifest_node_entries(manifest):
            uid = entry.get("uid")
            kind = entry.get("kind")
            path = entry.get("path")
            if (
                isinstance(uid, str)
                and isinstance(kind, str)
                and kind in {"imagetool", "tool"}
                and isinstance(path, str)
            ):
                entries.append((uid, kind, f"{path}/{kind}"))
        return entries

    def _restore_workspace_link_groups(
        self,
        manifest: Mapping[str, typing.Any] | None,
        loaded_targets_by_uid: Mapping[str, int | str],
    ) -> None:
        group_nodes: dict[int, list[_ImageToolWrapper | _ManagedWindowNode]] = {}
        group_colors: dict[int, bool] = {}
        invalid_groups: set[int] = set()
        for entry in self._iter_workspace_manifest_node_entries(manifest):
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
                    opened = _manager_xarray.open_workspace_dataset(
                        workspace_file_path,
                        payload_path,
                        chunks={},
                    )
                    try:
                        opened_ds = opened.copy(deep=False)
                        ds = _manager_workspace._restore_workspace_dataset_attrs(
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
                ds = _manager_workspace._restore_workspace_dataset_attrs(
                    typing.cast("xr.DataTree", node_tree["imagetool"])
                    .to_dataset(inherit=False)
                    .load()
                )
            target = self._manager._load_workspace_imagetool_dataset(
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
                ds = _manager_workspace._restore_workspace_dataset_attrs(
                    typing.cast("xr.DataTree", node_tree["tool"])
                    .to_dataset(inherit=False)
                    .load()
                )
            target = self._manager._load_workspace_tool_dataset(
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
                child_item = self._manager._tree_item_child_by_key(
                    selection_item, child_key
                )
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
            item = self._manager._tree_item_child_by_key(root_item, key)
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
            item = self._manager._tree_item_child_by_key(root_item, figure_path)
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
        for entry in self._iter_workspace_manifest_node_entries(manifest):
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
            opened = _manager_xarray.open_workspace_dataset(
                fname, payload_path, chunks=chunks
            )
            try:
                if load:
                    return _manager_workspace._restore_workspace_dataset_attrs(
                        opened.load()
                    )
                return _manager_workspace._restore_workspace_dataset_attrs(
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
                ds = _manager_workspace._read_workspace_dataset_group_h5py(
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
                target = self._manager._load_workspace_imagetool_dataset(
                    ds,
                    parent_target=parent_target,
                    node_path=path,
                    loaded_targets_by_uid=loaded_targets_by_uid,
                    profiler=profiler,
                    pending_workspace_memory_payload=pending_payload,
                )
            else:
                target = self._manager._load_workspace_tool_dataset(
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
            self._manager._workspace_load_context()
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
                        self._manager._drain_workspace_restore_events()
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
                        self._manager._restore_workspace_layout(manifest)
                        self._restore_workspace_option_overrides(manifest)
                        self._restore_workspace_loader_state(manifest)
                        self._restore_standalone_apps_state(manifest)
                if not mark_dirty:
                    with profiler.stage("ui catch-up"):
                        self._manager._drain_workspace_restore_events()
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
        with self._manager._workspace_load_context():
            self._manager.remove_all_tools()
            self._manager._drain_workspace_restore_events()
            self._load_workspace_roots(backup_tree, [str(key) for key in backup_tree])
            self._load_workspace_figures(backup_tree)
            self._manager._drain_workspace_restore_events()
        self._restore_workspace_state_snapshot(snapshot)

    def _restore_replaced_workspace_file(
        self, path: pathlib.Path, snapshot: _WorkspaceStateSnapshot
    ) -> None:
        with self._manager._workspace_load_context():
            self._manager.remove_all_tools()
            self._manager._drain_workspace_restore_events()
            root_attrs = _manager_workspace._read_workspace_root_attrs_h5py(path)
            schema_version, _delta_save_count, manifest = (
                _manager_workspace._workspace_file_metadata_from_attrs(root_attrs)
            )
            if (
                schema_version == _manager_workspace._current_workspace_schema_version()
                and manifest is not None
            ):
                self._from_h5py_workspace_file(
                    path,
                    manifest,
                    replace=False,
                    mark_dirty=False,
                )
            else:
                tree = _manager_xarray.open_workspace_datatree(path, chunks=None)
                self._from_datatree(
                    tree,
                    replace=False,
                    mark_dirty=False,
                    select=False,
                    workspace_file_path=path,
                )
            self._manager._drain_workspace_restore_events()
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
            if not self._manager._is_datatree_workspace(tree):
                raise ValueError("Not a valid workspace file")

            schema_version, _delta_save_count, manifest = (
                _manager_workspace._workspace_file_metadata_from_attrs(tree.attrs)
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

            root_keys = _manager_workspace._workspace_root_keys(tree, manifest)

            dialog: _ChooseFromDataTreeDialog | None = None
            if select:
                with profiler.stage("selection dialog setup"):
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
                            self._manager._drain_workspace_restore_events()
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
                            self._manager._restore_workspace_layout(manifest)
                            self._restore_workspace_option_overrides(manifest)
                            self._restore_workspace_loader_state(manifest)
                            self._restore_standalone_apps_state(manifest)
                    if not mark_dirty:
                        with profiler.stage("ui catch-up"):
                            self._manager._drain_workspace_restore_events()
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

        for index in self._manager._workspace_root_indices():
            _append(self._manager._tool_graph.root_wrappers[index].uid)
        for uid in self._manager._tool_graph.figure_uids:
            if uid in self._manager._tool_graph.nodes:
                _append(uid)
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
        return _manager_workspace._workspace_root_attrs_payload(
            root_order=self._workspace_root_indices(),
            nodes=self._workspace_node_manifest_entries(),
            delta_save_count=delta_save_count,
            erlab_version=str(erlab.__version__),
            workspace_link_id=self._manager._workspace_state.link_id,
            manager_layout=self._workspace_layout_snapshot(),
            loader_state=self._workspace_loader_state_snapshot(),
            standalone_apps=self._workspace_standalone_apps_snapshot(),
            option_overrides=self._workspace_option_overrides_snapshot(),
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

    def _workspace_option_overrides_snapshot(self) -> dict[str, typing.Any]:
        return _manager_workspace.WorkspaceOptionOverridesState(
            overrides=erlab.interactive._options.core.normalize_workspace_option_overrides(
                self._manager._workspace_state.option_overrides
            )
        ).model_dump(mode="json")

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
            state = _manager_workspace.WorkspaceOptionOverridesState.model_validate(
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

    def _workspace_datatree_for_payload_uids(self, uids: Iterable[str]) -> xr.DataTree:
        constructor: dict[str, xr.Dataset] = {}
        for uid in sorted(set(uids), key=self._workspace_node_path):
            node = self._manager._tool_graph.nodes.get(uid)
            if node is None:
                continue
            self._manager._serialize_workspace_node(
                constructor,
                node,
                self._workspace_node_path(uid),
                include_children=False,
            )
        tree = xr.DataTree.from_dict(constructor)
        _manager_workspace._set_legacy_workspace_schema(tree.attrs)
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
            self._manager._serialize_workspace_node(
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
                        _manager_xarray.ensure_workspace_hdf5_filters_registered()
                        stack.enter_context(
                            _manager_xarray._workspace_file_lock(source_key)
                        )
                        h5_file = stack.enter_context(h5py.File(source_key, "r"))
                        pending_compression_files[source_key] = h5_file
                    matches = _manager_workspace._h5_group_matches_compression(
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

            for index in self._manager._workspace_root_indices():
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
        _manager_workspace._set_legacy_workspace_schema(tree.attrs)
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
            != _manager_workspace._current_workspace_schema_version()
            or not workspace_path.exists()
        ):
            return None

        try:
            root_attrs = _manager_workspace._read_workspace_root_attrs_h5py(
                workspace_path
            )
        except Exception:
            return None
        schema_version, _delta_save_count, manifest = (
            _manager_workspace._workspace_file_metadata_from_attrs(root_attrs)
        )
        if (
            schema_version != _manager_workspace._current_workspace_schema_version()
            or manifest is None
        ):
            return None

        identities = {
            (uid, kind): payload_path
            for uid, kind, payload_path in self._workspace_manifest_payload_entries(
                manifest
            )
        }
        if not identities:
            return None
        return workspace_path, identities

    def _workspace_full_save_manifest_entries(
        self, root_attrs: Mapping[str, typing.Any]
    ) -> list[tuple[str, str, str]]:
        manifest = _manager_workspace._workspace_manifest_from_attrs(
            typing.cast("Mapping[Hashable, typing.Any]", root_attrs)
        )
        return self._workspace_manifest_payload_entries(manifest)

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
        return serialize_uids

    def _workspace_full_save_manifest_first_snapshot(
        self,
        generation: int,
        fname: str | os.PathLike[str],
        root_attrs: dict[str, typing.Any],
        *,
        compression_mode: WorkspaceCompressionMode,
        require_matching_compression: bool,
    ) -> _manager_workspace._WorkspaceSaveSnapshot | None:
        if _manager_workspace._workspace_path_is_likely_network_path(fname):
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
            _manager_xarray.ensure_workspace_hdf5_filters_registered()
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    _manager_xarray._workspace_file_lock(workspace_path)
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
                                _manager_xarray._workspace_file_lock(source_key)
                            )
                            pending_h5_file = stack.enter_context(
                                h5py.File(source_key, "r")
                            )
                            pending_compression_files[source_key] = pending_h5_file
                        matches = _manager_workspace._h5_group_matches_compression(
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
                            and _manager_xarray._normalized_file_path(
                                pending_workspace_path
                            )
                            == _manager_xarray._normalized_file_path(workspace_path)
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
                        _manager_workspace._workspace_h5_object_storage_size(
                            source_group
                        )
                    )
                    compression_mismatch = (
                        require_matching_compression
                        and not _manager_workspace._h5_group_matches_compression(
                            h5_file, source_path, compression_mode
                        )
                    )
                    pending_payload = node.pending_workspace_payload
                    pending_source_matches = False
                    if pending_payload is not None:
                        pending_workspace_path, pending_payload_path = pending_payload
                        pending_source_matches = _manager_xarray._normalized_file_path(
                            pending_workspace_path
                        ) == _manager_xarray._normalized_file_path(
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
                        update = self._manager._workspace_attr_update_snapshot(uid)
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
        return _manager_workspace._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=0,
            compression_mode=compression_mode,
            full_tree=tree,
            copy_source=str(workspace_path),
            copy_groups=tuple(copy_groups),
            copy_group_sources=tuple(copy_group_sources),
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
            _manager_workspace._write_full_workspace_tree_file(
                fname,
                snapshot.full_tree,
                snapshot.root_attrs,
                copy_source=snapshot.copy_source,
                copy_groups=snapshot.copy_groups,
                copy_group_sources=snapshot.copy_group_sources,
                compression_mode=snapshot.compression_mode,
            )
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
        manifest = _manager_workspace._workspace_manifest_from_attrs(
            typing.cast("Mapping[Hashable, typing.Any]", root_attrs)
        )
        uids: set[str] = set()
        for node in cls._iter_workspace_manifest_node_entries(manifest):
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
        return sorted(rewrite_uids, key=self._workspace_node_path)

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
                compression_mode=snapshot.compression_mode,
            )
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
            with self._workspace_document_access_context(fname) as access:
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
            _manager_workspace._recover_workspace_transactions(fname)
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
                    _manager_workspace._current_workspace_schema_version()
                )
            else:
                self._save_workspace_delta(fname)
        finally:
            self._manager._workspace_state.saving_depth -= 1
        if mark_clean:
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
        self._manager._show_legacy_workspace_upgrade_message(fname)
        converted_fname = self._manager._workspace_save_dialog(
            native=native,
            caption="Save Converted Workspace",
            selected_file=fname,
        )
        if converted_fname is None:
            return None
        converted_path = pathlib.Path(converted_fname).resolve()
        if not _manager_workspace._workspace_path_is_itws(converted_path):
            _show_itws_workspace_warning(self._manager)
            return None
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
        estimated_obsolete_bytes: int = 0,
        replacement_delta_count: int = 0,
        repack_estimate_known: bool = True,
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
            estimated_obsolete_bytes = 0
            replacement_delta_count = 0
            repack_estimate_known = True
            schema_version = _manager_workspace._current_workspace_schema_version()
        elif workspace_access is not None:
            associated_lock = workspace_access.take_lock()

        self._manager._set_workspace_path(
            associated_fname, workspace_lock=associated_lock
        )
        self._manager._workspace_state.delta_save_count = delta_save_count
        self._manager._workspace_state.set_repack_estimate(
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            known=repack_estimate_known,
        )
        self._manager._workspace_state.schema_version = schema_version
        self._manager._workspace_state.needs_full_save = (
            _manager_workspace._workspace_schema_requires_full_save(schema_version)
        )
        if rebind_data:
            self._manager._rebind_workspace_backed_imagetools(associated_fname)
        self._manager._drain_workspace_restore_events()
        self._manager._mark_workspace_clean()
        self._record_recent_workspace(associated_fname)

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
        for node in sorted(nodes, key=lambda node: self._workspace_node_path(node.uid)):
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
                    old_path = _manager_xarray._normalized_file_path(old_workspace_path)
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

        origin = self._manager._active_managed_window()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self._manager, "Offloading to workspace..."
            ):
                self._manager._rebind_workspace_backed_imagetools(
                    workspace_path,
                    targets=offload_targets,
                    chunks={},
                )
                _manager_workspace._write_workspace_root_attrs_to_file(
                    workspace_path,
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
            or self._manager._workspace_state.options_modified
        ) and not self._workspace_has_non_layout_modifications()

    def _workspace_rewrite_group_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, xr.Dataset]]:
        constructor: dict[str, xr.Dataset] = {}
        node = self._manager._tool_graph.nodes[uid]
        node_path = self._workspace_node_path(uid)
        self._manager._serialize_workspace_node(
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
        old_display_name = self._decode_workspace_attr_text(
            attrs.get("tool_display_name")
        )
        old_title = self._decode_workspace_attr_text(attrs.get("tool_title"))
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
            referenced_uid = reference.get("node_uid")
            if (
                not isinstance(referenced_uid, str)
                or not referenced_uid
                or referenced_uid not in self._manager._tool_graph.nodes
            ):
                return False
        return True

    def _workspace_attr_update_snapshot(
        self, uid: str
    ) -> tuple[str, dict[str, typing.Any], tuple[str, dict[str, xr.Dataset]]] | None:
        constructor: dict[str, xr.Dataset] = {}
        node = self._manager._tool_graph.nodes[uid]
        node_path = self._workspace_node_path(uid)
        payload_path = self._manager._workspace_payload_path(uid)
        pending_attrs = self._pending_workspace_payload_attrs_for_save(node)
        if pending_attrs is not None:
            return (
                payload_path,
                pending_attrs,
                (node_path, constructor),
            )
        self._manager._serialize_workspace_node(
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
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        state = self._manager._workspace_state
        rewrite_groups: list[tuple[str, dict[str, xr.Dataset]]] = []
        rewritten_uids: set[str] = set()
        for uid in self._manager._workspace_highest_dirty_data_roots():
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
                _manager_workspace._workspace_h5_paths_storage_size(
                    state.path,
                    (group_path for group_path, _constructor in rewrite_groups),
                )
            )
            estimated_obsolete_bytes += old_bytes
            if replaced_group_count > 0:
                replacement_delta_count += 1
        root_attrs = _manager_workspace._workspace_root_attrs_with_repack_estimate(
            root_attrs,
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            repack_estimate_known=repack_estimate_known,
        )

        return _manager_workspace._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=delta_save_count,
            estimated_obsolete_bytes=estimated_obsolete_bytes,
            replacement_delta_count=replacement_delta_count,
            repack_estimate_known=repack_estimate_known,
            compression_mode=self._workspace_compression_mode(),
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
    ) -> _manager_workspace._WorkspaceSaveSnapshot:
        compression_mode = self._workspace_compression_mode()
        root_attrs = self._manager._workspace_root_attrs_payload(delta_save_count=0)
        if fname is None:
            fname = self._manager._workspace_state.path
        target_drops_copy_groups = (
            fname is not None
            and _manager_workspace._workspace_path_is_likely_network_path(fname)
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
        return _manager_workspace._WorkspaceSaveSnapshot(
            generation=generation,
            root_attrs=root_attrs,
            delta_save_count=0,
            compression_mode=compression_mode,
            full_tree=tree,
            copy_source=copy_source,
            copy_groups=copy_groups,
            copy_group_sources=pending_copy_groups,
        )

    def _workspace_file_repack_snapshot(
        self, generation: int
    ) -> _manager_workspace._WorkspaceSaveSnapshot | None:
        workspace_path = self._manager._workspace_state.path
        if workspace_path is None:
            return None
        if _manager_workspace._workspace_path_is_likely_network_path(workspace_path):
            return None
        try:
            root_attrs, copy_groups = _manager_workspace._workspace_file_repack_payload(
                workspace_path
            )
        except Exception:
            logger.debug(
                "Skipping shutdown compaction; file-level repack snapshot failed",
                exc_info=True,
            )
            return None
        return _manager_workspace._WorkspaceSaveSnapshot(
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
                estimated_obsolete_bytes = _workspace_obsolete_estimate(workspace_path)
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

        identities = {
            (uid, kind): payload_path
            for uid, kind, payload_path in self._workspace_manifest_payload_entries(
                manifest
            )
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
                kind = "imagetool" if node.is_imagetool else "tool"
                source_path = identities.get((uid, kind))
                if source_path is None:
                    continue
                payload_path = self._manager._workspace_payload_path(uid)
                try:
                    payload_tree = typing.cast("xr.DataTree", tree[payload_path])
                except KeyError:
                    continue
                payload_ds = payload_tree.to_dataset(inherit=False)
                compression_matches = (
                    h5_file is None
                    or compression_mode is None
                    or _manager_workspace._workspace_h5_group_matches_compression_mode(
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
        snapshot: _manager_workspace._WorkspaceSaveSnapshot,
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

        worker = _manager_workspace._WorkspaceSaveWorker(fname, snapshot)
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

        receiver = _manager_workspace._WorkspaceSaveResultReceiver(
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
        self._manager._mark_workspace_structure_dirty(
            "Live workspace data references need refresh"
        )

    def _finish_workspace_save_result(
        self,
        *,
        document_id: str,
        workspace_path: pathlib.Path,
        old_workspace_path: pathlib.Path | None,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]],
        snapshot: _manager_workspace._WorkspaceSaveSnapshot,
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
                self._manager._restore_focus_after_workspace_save(origin)
            return False

        self._manager._drain_workspace_deferred_events()
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
                    self._manager._restore_focus_after_workspace_save(origin)
                return False
            self._manager._workspace_state.schema_version = (
                _manager_workspace._current_workspace_schema_version()
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
            self._manager._mark_workspace_clean()
            message = (
                f"Workspace saved in {total_elapsed:.1f} s"
                if total_elapsed >= _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS
                else "Workspace saved"
            )
        self._manager._status_bar.showMessage(message, 5000)
        if restore_focus:
            self._manager._restore_focus_after_workspace_save(origin)
        self._record_recent_workspace(workspace_path)
        return True

    def _finish_background_workspace_save(
        self,
        *,
        document_id: str,
        workspace_path: pathlib.Path,
        old_workspace_path: pathlib.Path | None,
        backing_snapshot: Mapping[str, tuple[str, tuple[str, ...]]],
        snapshot: _manager_workspace._WorkspaceSaveSnapshot,
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

        origin = self._manager._active_managed_window()
        document_id = self._manager._workspace_state.document_id
        old_workspace_path = workspace_path
        backing_snapshot = self._manager._workspace_data_backing_snapshot()
        self._manager._status_bar.showMessage("Saving workspace...")
        started_at = time.perf_counter()
        snapshot: _manager_workspace._WorkspaceSaveSnapshot | None = None
        try:
            snapshot_started_at = time.perf_counter()
            snapshot = self._workspace_save_snapshot(workspace_path)
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
                self._manager._restore_focus_after_workspace_save(origin)
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
                self._manager._restore_focus_after_workspace_save(origin)
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
        origin = self._manager._active_managed_window()
        fname = self._manager._workspace_save_dialog(
            native=native, caption="Save Workspace As"
        )
        if fname is None:
            if on_finished is not None:
                on_finished(False)
            return False
        if not _manager_workspace._workspace_path_is_itws(fname):
            _show_itws_workspace_warning(self._manager)
            if on_finished is not None:
                on_finished(False)
            return False
        old_workspace_path = self._manager._workspace_state.path
        document_id = self._manager._workspace_state.document_id
        backing_snapshot = self._manager._workspace_data_backing_snapshot()
        access: _WorkspaceDocumentAccess | None = None
        snapshot: _manager_workspace._WorkspaceSaveSnapshot | None = None
        self._manager._status_bar.showMessage("Saving workspace...")
        started_at = time.perf_counter()
        try:
            access = self._workspace_document_access(fname)
            self._manager._drain_workspace_deferred_events()
            generation = self._manager._workspace_state.dirty_generation
            self._manager._workspace_state.saving_depth += 1
            try:
                snapshot_started_at = time.perf_counter()
                snapshot = self._workspace_full_save_snapshot(
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
            self._manager._restore_focus_after_workspace_save(origin)
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
                self._manager._restore_focus_after_workspace_save(origin)
                if on_finished is not None:
                    on_finished(False)
                return

            self._manager._drain_workspace_deferred_events()
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
                self._manager._restore_focus_after_workspace_save(origin)
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
                    self._manager._restore_focus_after_workspace_save(origin)
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
                _manager_workspace._current_workspace_schema_version()
            )
            saved_path = access.path
            self._manager._set_workspace_path(
                saved_path, workspace_lock=access.take_lock()
            )
            access = None
            self._manager._drain_workspace_deferred_events()
            self._manager._mark_workspace_clean()
            self._record_recent_workspace(saved_path)
            message = (
                f"Workspace saved in {total_elapsed:.1f} s"
                if total_elapsed >= _WORKSPACE_SAVE_WAIT_DIALOG_THRESHOLD_SECONDS
                else "Workspace saved"
            )
            self._manager._status_bar.showMessage(message, 5000)
            self._manager._restore_focus_after_workspace_save(origin)
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
            self._manager._restore_focus_after_workspace_save(origin)
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
        if not self._workspace_should_repack_before_shutdown():
            return False
        try:
            logger.debug("Compacting workspace before shutdown...")
            self._shutdown_compaction_attempted = True
            document_id = self._manager._workspace_state.document_id
            self._manager._drain_workspace_deferred_events()
            generation = self._manager._workspace_state.dirty_generation
            self._manager._workspace_state.saving_depth += 1
            try:
                snapshot = self._workspace_file_repack_snapshot(generation)
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
                    _manager_workspace._current_workspace_schema_version()
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
                self._manager._drain_workspace_deferred_events()
                post_save_events = tuple(
                    event
                    for event in self._manager._workspace_state.dirty_events
                    if event.generation > snapshot.generation
                )
                if post_save_events:
                    self._restore_workspace_dirty_events(post_save_events)
                else:
                    self._manager._mark_workspace_clean()
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

        origin = self._manager._active_managed_window()
        old_workspace_path = workspace_path
        backing_snapshot = self._manager._workspace_data_backing_snapshot()
        try:
            with erlab.interactive.utils.wait_dialog(
                origin or self._manager, "Compacting workspace..."
            ):
                self._manager._save_workspace_document(
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
            self._manager._mark_workspace_clean()
        except _WorkspacePostSaveBindingError:
            self._mark_workspace_post_save_binding_refresh_failed()
            self._show_workspace_post_save_binding_error(workspace_path)
            self._manager._restore_focus_after_workspace_save(origin)
            return False
        except Exception:
            self._manager._show_operation_error(
                "Error while compacting workspace",
                "An error occurred while compacting the workspace file.",
            )
            self._manager._restore_focus_after_workspace_save(origin)
            return False

        self._manager._restore_focus_after_workspace_save(origin)
        self._manager._workspace_state.reset_repack_estimate()
        return True

    def _save_to_file(self, fname: str):
        """Export a selected subset of the workspace to ``fname``.

        This helper preserves the older selection-dialog behavior used by tests and
        private callers. Document-style Save and Save As use
        :meth:`_save_workspace_document` instead.
        """
        _require_itws_workspace_path(fname, _WORKSPACE_SAVE_SUFFIX_ERROR)
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
                for node in tree.subtree:
                    ds = node.to_dataset(inherit=False)
                    if ds.variables or ds.attrs:
                        node.dataset = (
                            _manager_workspace._sanitize_workspace_attr_names(ds)
                        )
                tree.to_netcdf(
                    fname,
                    engine="h5netcdf",
                    invalid_netcdf=True,
                    encoding=_manager_xarray.workspace_datatree_encoding(
                        tree,
                        compression_mode=self._workspace_compression_mode(),
                    ),
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
            with self._manager._workspace_document_access_context(fname) as access:
                with profiler.stage("metadata read"):
                    _manager_workspace._recover_workspace_transactions(access.path)
                try:
                    with profiler.stage("metadata read"):
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
                            ) = _workspace_repack_estimate(
                                manifest,
                                delta_save_count=delta_save_count,
                            )
                            self._manager._associate_loaded_workspace_file(
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
                    tree = _manager_xarray.open_workspace_datatree(
                        access.path, chunks=None
                    )
                    schema_version, delta_save_count, manifest = (
                        _manager_workspace._workspace_file_metadata_from_attrs(
                            tree.attrs
                        )
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
                    ) = _workspace_repack_estimate(
                        manifest,
                        delta_save_count=delta_save_count,
                    )
                    self._manager._associate_loaded_workspace_file(
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
                self._record_recent_workspace(fname)
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
        if (directory := self._manager._recent_or_default_directory()) is not None:
            dialog.setDirectory(directory)

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
                        ImageTool.from_dataset(
                            ds,
                            _in_manager=True,
                            options_model=self._manager.effective_interactive_options,
                        ),
                        activate=True,
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

        try:
            prepared_data = (
                erlab.interactive.imagetool.viewer_state._prepare_input_data(
                    typing.cast("list[xr.DataArray]", data),
                    self._manager,
                    allow_dialog=watched_var is None,
                )
            )
        except ValueError:
            logger.exception(
                "Error creating ImageTool window",
                extra={"suppress_ui_alert": True},
            )
            self._manager._error_creating_imagetool()
            return [False for _item in data]
        if prepared_data is None:
            return [False for _item in data]

        link = kwargs.pop("link", False)
        link_colors = kwargs.pop("link_colors", True)
        indices: list[int] = []
        kwargs["_in_manager"] = True
        kwargs.setdefault("options_model", self._manager.effective_interactive_options)

        load_func = kwargs.pop("load_func", None)
        load_indices = kwargs.pop("load_indices", None)
        load_preparation_operations = kwargs.pop("preparation_operations", None)
        source_input_ndims = kwargs.pop("source_input_ndims", None)
        source_input_dtypes = kwargs.pop("source_input_dtypes", None)
        if show is None:
            show = len(prepared_data) == 1
        watched_metadata = dict(watched_metadata or {})
        if watched_var is not None:
            watched_metadata.setdefault(
                "workspace_link_id", self._manager._workspace_state.link_id
            )

        for i, prepared in enumerate(prepared_data):
            d = prepared.data
            # Set selection-specific load function if provided
            load_selection = (
                typing.cast("Sequence[typing.Any]", load_indices)[i]
                if load_indices is not None
                else i
            )
            this_load_func = (*load_func[:2], load_selection) if load_func else None
            preparation_operations = (
                tuple(
                    typing.cast(
                        "Sequence[provenance.ToolProvenanceOperation]",
                        load_preparation_operations,
                    )[i]
                )
                if load_preparation_operations is not None
                else prepared.operations
            )
            source_input_ndim = (
                typing.cast("Sequence[int]", source_input_ndims)[i]
                if source_input_ndims is not None
                else prepared.source_ndim
            )
            source_input_dtype = (
                typing.cast("Sequence[typing.Any]", source_input_dtypes)[i]
                if source_input_dtypes is not None
                else prepared.source_dtype
            )
            try:
                indices.append(
                    self._manager.add_imagetool(
                        ImageTool(
                            d,
                            **kwargs,
                            load_func=this_load_func,
                            preparation_operations=preparation_operations,
                        ),
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
                        source_input_ndim=source_input_ndim,
                        source_input_dtype=source_input_dtype,
                    )
                )
                if watched_var is not None:
                    # Refresh title to include variable name
                    node = self._manager._node_for_target(indices[-1])
                    if node.imagetool is not None:
                        node.imagetool._update_title()
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
