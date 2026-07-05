"""Managed window nodes shown in ImageToolManager."""

from __future__ import annotations

__all__ = ["_ImageToolWrapper", "_ManagedWindowNode"]

import contextlib
import datetime
import functools
import importlib
import json
import keyword
import logging
import pathlib
import sys
import typing
import uuid
import weakref
from dataclasses import dataclass

import numpy as np
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._xarray as _manager_xarray
from erlab.interactive.imagetool import provenance
from erlab.interactive.imagetool._load_source import (
    _default_load_source_name,
    _load_code_from_file_details,
    _load_source_details_from_file,
    _load_source_details_from_provenance,
    _LoadSourceDetails,
)
from erlab.interactive.imagetool._mainwindow import ImageTool

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from erlab.interactive.imagetool.manager._mainwindow import ImageToolManager
    from erlab.interactive.imagetool.viewer import ImageSlicerArea
    from erlab.interactive.imagetool.viewer_state import ImageSlicerState


logger = logging.getLogger(__name__)


def _current_added_time() -> datetime.datetime:
    return datetime.datetime.now().astimezone().replace(microsecond=0)


def _coerce_added_time(
    value: datetime.datetime | str | bytes | None, *, node_uid: str | None = None
) -> datetime.datetime:
    if value is None:
        return _current_added_time()
    if isinstance(value, bytes):
        try:
            value = value.decode()
        except UnicodeDecodeError:
            logger.warning(
                "Ignoring invalid saved manager added timestamp for node %s",
                node_uid,
                exc_info=True,
            )
            return _current_added_time()
    if isinstance(value, str):
        try:
            value = datetime.datetime.fromisoformat(value)
        except ValueError:
            logger.warning(
                "Ignoring invalid saved manager added timestamp for node %s",
                node_uid,
                exc_info=True,
            )
            return _current_added_time()
    if not isinstance(value, datetime.datetime):
        logger.warning(
            "Ignoring invalid saved manager added timestamp for node %s",
            node_uid,
        )
        return _current_added_time()
    if value.tzinfo is None or value.utcoffset() is None:
        logger.warning(
            "Ignoring invalid saved manager added timestamp for node %s",
            node_uid,
        )
        return _current_added_time()
    return value.replace(microsecond=0)


def _format_added_time(value: datetime.datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%z)")


def _coerce_note(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode()
        except UnicodeDecodeError:
            logger.warning("Ignoring invalid saved manager note", exc_info=True)
            return ""
    if isinstance(value, str):
        return value
    logger.warning(
        "Ignoring invalid saved manager note of type %s", type(value).__name__
    )
    return ""


def _preview_from_imagetool(
    imagetool: ImageTool | None,
    fallback_ratio: float,
    fallback_pixmap: QtGui.QPixmap,
) -> tuple[float, QtGui.QPixmap]:
    if imagetool is None:
        return fallback_ratio, fallback_pixmap

    slicer_area = imagetool.slicer_area
    slicer_area._update_if_delayed()
    try:
        main_image = slicer_area.main_image
    except RuntimeError:
        return fallback_ratio, fallback_pixmap

    if not erlab.interactive.utils.qt_is_valid(main_image):
        return fallback_ratio, fallback_pixmap

    view_box = main_image.getViewBox()
    if not erlab.interactive.utils.qt_is_valid(view_box):
        return fallback_ratio, fallback_pixmap

    vb_rect = view_box.rect()
    width = vb_rect.width()
    height = vb_rect.height()
    if width <= 0 or height <= 0:
        return fallback_ratio, fallback_pixmap

    if not main_image.slicer_data_items:
        return fallback_ratio, fallback_pixmap

    image_item = main_image.slicer_data_items[0]
    if not erlab.interactive.utils.qt_is_valid(image_item):
        return fallback_ratio, fallback_pixmap

    try:
        pixmap = image_item.getPixmap()
    except RuntimeError:
        return fallback_ratio, fallback_pixmap

    if pixmap is None or pixmap.isNull():
        return fallback_ratio, fallback_pixmap

    return height / width, pixmap.transformed(QtGui.QTransform().scale(1.0, -1.0))


def _preview_image_for_node(node: object) -> tuple[float, QtGui.QPixmap]:
    fallback = (float("NaN"), QtGui.QPixmap())
    dynamic_node = typing.cast("typing.Any", node)
    try:
        preview = dynamic_node._preview_image
    except AttributeError:
        preview = None
    if (
        isinstance(preview, tuple)
        and len(preview) == 2
        and isinstance(preview[1], QtGui.QPixmap)
    ):
        with contextlib.suppress(TypeError, ValueError):
            return float(preview[0]), preview[1]

    try:
        imagetool = dynamic_node.imagetool
    except (AttributeError, RuntimeError, ValueError):
        return fallback
    try:
        return _preview_from_imagetool(
            typing.cast("ImageTool | None", imagetool),
            fallback[0],
            fallback[1],
        )
    except (AttributeError, RuntimeError, ValueError):
        return fallback


@dataclass(frozen=True)
class _MetadataField:
    label: str
    value: str
    monospace: bool = False
    wrap: bool = False
    details: _LoadSourceDetails | None = None


@dataclass(frozen=True)
class _NodePersistenceView:
    data: xr.DataArray | None
    state: ImageSlicerState | None
    provenance_spec: provenance.ToolProvenanceSpec | None
    source_spec: provenance.ToolProvenanceSpec | None
    source_binding: provenance.ImageToolSelectionSourceBinding | None
    output_id: str | None
    source_state: typing.Literal["fresh", "stale", "unavailable"]
    source_auto_update: bool
    data_backing: typing.Literal["dask", "file_lazy", "memory"] | None
    source_paths: tuple[str, ...] = ()


def _format_chunk_summary(data: xr.DataArray) -> str:
    chunks = data.chunks
    if chunks is None:
        return "In memory"

    parts: list[str] = []
    for dim, dim_chunks in zip(data.dims, chunks, strict=True):
        if all(chunk == dim_chunks[0] for chunk in dim_chunks):
            chunk_text = str(dim_chunks[0])
        else:
            chunk_text = ", ".join(str(chunk) for chunk in dim_chunks)
        parts.append(f"{dim}={chunk_text}")
    return "; ".join(parts)


def _dataarray_name(data: xr.DataArray | None) -> str:
    if data is None or data.name is None:
        return ""
    name = str(data.name)
    return "" if name.strip() == "" else name


def _append_unique_path(paths: list[pathlib.Path], path: str | pathlib.Path) -> None:
    normalized = pathlib.Path(path).expanduser()
    with contextlib.suppress(OSError, RuntimeError):
        normalized = normalized.resolve(strict=False)
    if normalized not in paths:
        paths.append(normalized)


def _collect_provenance_file_paths(
    spec: provenance.ToolProvenanceSpec | None,
    paths: list[pathlib.Path],
) -> None:
    if spec is None:
        return
    if spec.file_load_source is not None:
        _append_unique_path(paths, spec.file_load_source.path)
    for script_input in spec.script_inputs:
        _collect_provenance_file_paths(script_input.parsed_provenance_spec(), paths)


def _compact_file_suffix(paths: Sequence[pathlib.Path]) -> str:
    if not paths:
        return ""
    stems = tuple(dict.fromkeys(path.stem for path in paths))
    if len(stems) <= 2:
        return f" ({', '.join(stems)})"
    return f" ({', '.join(stems[:2])}, +{len(stems) - 2})"


def _spec_with_final_data_name(
    spec: provenance.ToolProvenanceSpec,
    name: str,
) -> provenance.ToolProvenanceSpec:
    rename = provenance.RenameOperation(name=name)
    if spec.operations:
        return spec.append_final_rename(name)

    stages = list(spec.replay_stages)
    if stages:
        last_stage = stages[-1]
        operations = tuple(last_stage.operations)
        if operations and isinstance(operations[-1], provenance.RenameOperation):
            stages[-1] = last_stage.model_copy(
                update={"operations": (*operations[:-1], rename)}
            )
            return spec.model_copy(update={"replay_stages": tuple(stages)})

    if spec.kind == "file" or (stages and spec.kind == "script"):
        return spec.append_replay_stage(provenance.full_data(rename))
    return spec.append_final_rename(name)


class _ManagedWindowNode(QtCore.QObject):
    """A recursively managed window node in ImageToolManager."""

    _source_state_type = typing.Literal["fresh", "stale", "unavailable"]

    def __init__(
        self,
        manager: ImageToolManager,
        uid: str,
        parent_uid: str | None,
        window: QtWidgets.QWidget | None,
        *,
        window_kind: typing.Literal["imagetool", "tool"] | None = None,
        name: str | None = None,
        provenance_spec: provenance.ToolProvenanceSpec | None = None,
        source_spec: provenance.ToolProvenanceSpec | None = None,
        source_binding: provenance.ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _source_state_type = "fresh",
        output_id: str | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
        note: str | bytes | None = None,
    ) -> None:
        super().__init__(manager)
        self._manager = weakref.ref(manager)
        self.uid = uid
        self.parent_uid = parent_uid
        self._recent_geometry: QtCore.QRect | None = None
        self._created_time = _coerce_added_time(created_time, node_uid=uid)

        self._childtools: dict[str, QtWidgets.QWidget] = {}
        self._childtool_indices: list[str] = []

        self._imagetool: ImageTool | None = None
        self._tool_window: erlab.interactive.utils.ToolWindow | None = None
        if window_kind is None:
            if window is None:
                raise TypeError("window_kind is required when window is None")
            window_kind = "imagetool" if isinstance(window, ImageTool) else "tool"
        self._window_kind = window_kind
        self._name = (
            name
            if name is not None
            else ("" if window is None else window.windowTitle())
        )

        self._source_spec: provenance.ToolProvenanceSpec | None = None
        self._source_binding: provenance.ImageToolSelectionSourceBinding | None = None
        self._provenance_spec: provenance.ToolProvenanceSpec | None = None
        self._detached_live_parent_data: xr.DataArray | None = None
        self._source_state: _ManagedWindowNode._source_state_type = "fresh"
        self._source_auto_update: bool = False
        self._output_id: str | None = None
        self._suspend_descendant_signal_propagation: bool = False
        self._pending_workspace_payload: tuple[pathlib.Path, str] | None = None
        self._pending_workspace_payload_kind: (
            typing.Literal["imagetool", "tool"] | None
        ) = None
        self._pending_workspace_payload_attrs: dict[str, typing.Any] | None = None
        self._pending_workspace_metadata_cache: (
            tuple[
                tuple[typing.Literal["imagetool", "tool"], tuple[pathlib.Path, str]],
                str,
            ]
            | None
        ) = None
        self._pending_workspace_preview_cache: (
            tuple[
                tuple[typing.Literal["imagetool", "tool"], tuple[pathlib.Path, str]],
                tuple[float, QtGui.QPixmap] | None,
            ]
            | None
        ) = None
        self._workspace_link_key: str | None = None
        self._workspace_link_colors: bool = True
        self._snapshot_token = (
            str(snapshot_token) if snapshot_token else uuid.uuid4().hex
        )
        self._note = _coerce_note(note)
        self._suspend_snapshot_token_updates = True

        self.window = window
        try:
            if source_spec is not None or source_binding is not None:
                self.set_source_binding(
                    source_spec,
                    source_binding=source_binding,
                    provenance_spec=provenance_spec,
                    auto_update=source_auto_update,
                    state=source_state,
                )
            elif output_id is not None:
                self.set_output_binding(
                    output_id,
                    provenance_spec=provenance_spec,
                    auto_update=source_auto_update,
                    state=source_state,
                )
            elif provenance_spec is not None:
                self.set_detached_provenance(provenance_spec)
            elif isinstance(window, ImageTool) and window.provenance_spec is not None:
                self.set_detached_provenance(window.provenance_spec)
        finally:
            self._suspend_snapshot_token_updates = False

    @property
    def manager(self) -> ImageToolManager:
        manager = self._manager()
        if manager:
            return manager
        raise LookupError("Parent was destroyed")

    @property
    def is_imagetool(self) -> bool:
        return self._window_kind == "imagetool"

    @property
    def window(self) -> QtWidgets.QWidget | None:
        if self.is_imagetool:
            return self.imagetool
        return self.tool_window

    @window.setter
    def window(self, value: QtWidgets.QWidget | None) -> None:
        if self.imagetool is not None:
            manager = self._manager()
            if manager is not None:
                manager._unregister_interaction_window(self.imagetool)
            self._detach_imagetool()
        elif self.tool_window is not None:
            manager = self._manager()
            if manager is not None:
                manager._unregister_interaction_window(self.tool_window)
            old = self.tool_window
            with contextlib.suppress(TypeError, RuntimeError):
                old.sigInfoChanged.disconnect(self._handle_tool_info_changed)
            with contextlib.suppress(TypeError, RuntimeError):
                old.sigStateChanged.disconnect(self._handle_tool_state_changed)
            with contextlib.suppress(TypeError, RuntimeError):
                old.sigDataChanged.disconnect(self._handle_tool_data_changed)
            old.removeEventFilter(self)
            old._set_managed_source_update_dialog(None)
            old._set_managed_source_reload(None)
            old._set_managed_secondary_window_callback(None)
            old.set_source_parent_fetcher(None)
            old.set_input_provenance_parent_fetcher(None)
            old.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            old.close()
            self._tool_window = None

        if value is None:
            return

        if isinstance(value, ImageTool):
            self._window_kind = "imagetool"
            self._imagetool = value
            self._tool_window = None
            self.manager._install_workspace_save_shortcut(value)
            self.manager._register_interaction_window(value)
            if self._provenance_spec is not None or self._source_spec is not None:
                value.set_provenance_spec(self._provenance_spec or self._source_spec)
            value.installEventFilter(self)
            value.sigTitleChanged.connect(self.update_title)
            value.slicer_area.sigHistoryChanged.connect(
                self._handle_imagetool_state_changed
            )
            value.slicer_area.sigDataEdited.connect(self._handle_imagetool_data_edited)
            value.slicer_area.sigDataBackingChanged.connect(
                self._handle_imagetool_backing_changed
            )
            value.slicer_area.sigSourceDataReplaced.connect(
                self._handle_source_data_replaced
            )
            value.slicer_area._in_manager = True
            value.remove_act.setVisible(True)
            for plot in value.slicer_area._materialized_axes():
                plot.ensure_manager_figure_actions()
            return

        tool = typing.cast("erlab.interactive.utils.ToolWindow", value)
        self._window_kind = "tool"
        self._tool_window = tool
        self._imagetool = None
        self.manager._install_workspace_save_shortcut(tool)
        self.manager._register_interaction_window(tool)
        tool.installEventFilter(self)
        tool.sigInfoChanged.connect(self._handle_tool_info_changed)
        tool.sigStateChanged.connect(self._handle_tool_state_changed)
        tool.sigDataChanged.connect(self._handle_tool_data_changed)
        tool.destroyed.connect(self._handle_tool_window_destroyed)
        tool._set_managed_source_update_dialog(self.show_source_update_dialog)
        tool._set_managed_source_reload(
            self.reload_source_data,
            self.can_reload_source_data,
            self.reload_unavailable_reason,
        )
        tool._set_managed_secondary_window_callback(
            self._configure_tool_secondary_window
        )
        for secondary_window, _title in tool._managed_secondary_windows():
            self._configure_tool_secondary_window(secondary_window)

    def _configure_tool_secondary_window(self, window: QtWidgets.QWidget) -> None:
        manager = self._manager()
        if manager is None or not erlab.interactive.utils.qt_is_valid(manager):
            return
        manager._install_workspace_save_shortcut(window)
        manager._register_interaction_window(window)
        if manager._tool_graph.nodes.get(self.uid) is self:
            manager._set_node_window_modified(
                self.uid,
                self.uid
                in (
                    manager._workspace_state.dirty_added
                    | manager._workspace_state.dirty_data
                    | manager._workspace_state.dirty_state
                ),
            )

    def _handle_tool_window_destroyed(self, _obj: QtCore.QObject | None = None) -> None:
        manager = self._manager()
        if manager is None or not erlab.interactive.utils.qt_is_valid(manager):
            return
        if manager._tool_graph.nodes.get(self.uid) is not self:
            return
        manager._remove_childtool(self.uid)

    def _detach_imagetool(
        self, *, close: bool = True, unlink: bool = True
    ) -> ImageTool | None:
        old = self.imagetool
        if old is None:
            return None
        if unlink:
            old.slicer_area.unlink()
        if close:
            old.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        old.removeEventFilter(self)
        with contextlib.suppress(TypeError, RuntimeError):
            old.sigTitleChanged.disconnect(self.update_title)
        with contextlib.suppress(TypeError, RuntimeError):
            old.slicer_area.sigHistoryChanged.disconnect(
                self._handle_imagetool_state_changed
            )
        with contextlib.suppress(TypeError, RuntimeError):
            old.slicer_area.sigDataEdited.disconnect(self._handle_imagetool_data_edited)
        with contextlib.suppress(TypeError, RuntimeError):
            old.slicer_area.sigDataBackingChanged.disconnect(
                self._handle_imagetool_backing_changed
            )
        with contextlib.suppress(TypeError, RuntimeError):
            old.slicer_area.sigSourceDataReplaced.disconnect(
                self._handle_source_data_replaced
            )
        if close:
            old.close()
        self._imagetool = None
        return old

    def take_window(self) -> QtWidgets.QWidget | None:
        """Detach the current ImageTool window without closing it."""
        if self.tool_window is not None:
            raise TypeError("Window transfer is only supported for ImageTool nodes")
        return self._detach_imagetool(close=False, unlink=False)

    @property
    def imagetool(self) -> ImageTool | None:
        if self._imagetool is not None and not erlab.interactive.utils.qt_is_valid(
            self._imagetool
        ):
            self._imagetool = None
        return self._imagetool

    @property
    def tool_window(self) -> erlab.interactive.utils.ToolWindow | None:
        if self._tool_window is not None and not erlab.interactive.utils.qt_is_valid(
            self._tool_window
        ):
            self._tool_window = None
        return self._tool_window

    @property
    def slicer_area(self) -> ImageSlicerArea:
        if self.imagetool is None:
            raise ValueError("ImageTool is not available")
        return self.imagetool.slicer_area

    @property
    def pending_workspace_memory_payload(self) -> tuple[pathlib.Path, str] | None:
        if self._pending_workspace_payload_kind != "imagetool":
            return None
        return self._pending_workspace_payload

    @property
    def pending_workspace_tool_payload(self) -> tuple[pathlib.Path, str] | None:
        if self._pending_workspace_payload_kind != "tool":
            return None
        return self._pending_workspace_payload

    @property
    def pending_workspace_payload_kind(
        self,
    ) -> typing.Literal["imagetool", "tool"] | None:
        return self._pending_workspace_payload_kind

    @property
    def pending_workspace_payload(self) -> tuple[pathlib.Path, str] | None:
        return self._pending_workspace_payload

    @property
    def pending_workspace_payload_attrs(self) -> dict[str, typing.Any] | None:
        if self._pending_workspace_payload_attrs is None:
            return None
        return dict(self._pending_workspace_payload_attrs)

    def update_pending_workspace_payload_attrs(
        self, attrs: Mapping[str, typing.Any]
    ) -> None:
        if self._pending_workspace_payload is None:
            return
        self._pending_workspace_payload_attrs = dict(attrs)
        self._pending_workspace_metadata_cache = None
        self._pending_workspace_preview_cache = None

    def set_pending_workspace_payload(
        self,
        kind: typing.Literal["imagetool", "tool"],
        workspace_path: str | os.PathLike[str],
        payload_path: str,
        payload_attrs: Mapping[str, typing.Any] | None = None,
    ) -> None:
        self._pending_workspace_payload_kind = kind
        self._pending_workspace_payload = (
            pathlib.Path(workspace_path),
            payload_path.strip("/"),
        )
        self._pending_workspace_payload_attrs = (
            None if payload_attrs is None else dict(payload_attrs)
        )
        self._pending_workspace_metadata_cache = None
        self._pending_workspace_preview_cache = None

    def set_pending_workspace_memory_payload(
        self,
        workspace_path: str | os.PathLike[str],
        payload_path: str,
        payload_attrs: Mapping[str, typing.Any] | None = None,
    ) -> None:
        self.set_pending_workspace_payload(
            "imagetool",
            workspace_path,
            payload_path,
            payload_attrs=payload_attrs,
        )

    def clear_pending_workspace_payload(self) -> None:
        self._pending_workspace_payload = None
        self._pending_workspace_payload_kind = None
        self._pending_workspace_payload_attrs = None
        self._pending_workspace_metadata_cache = None
        self._pending_workspace_preview_cache = None

    def materialize_pending_workspace_payload(self) -> bool:
        if self._pending_workspace_payload is None:
            return True
        return self.manager._materialize_pending_workspace_payload(self)

    def _pending_workspace_preview_for_kind(
        self,
        kind: typing.Literal["imagetool", "tool"],
        renderer: Callable[[_ManagedWindowNode], tuple[float, QtGui.QPixmap] | None]
        | None = None,
    ) -> tuple[float, QtGui.QPixmap] | None:
        pending = (
            self.pending_workspace_memory_payload
            if kind == "imagetool"
            else self.pending_workspace_tool_payload
        )
        if pending is None:
            return None
        cache_key = (kind, pending)
        if (
            self._pending_workspace_preview_cache is not None
            and self._pending_workspace_preview_cache[0] == cache_key
        ):
            return self._pending_workspace_preview_cache[1]
        if renderer is None:
            return None
        preview = renderer(self)
        self._pending_workspace_preview_cache = (cache_key, preview)
        return preview

    def pending_workspace_preview_image(self) -> tuple[float, QtGui.QPixmap] | None:
        return self._pending_workspace_preview_for_kind(
            "imagetool", self.manager._pending_workspace_imagetool_preview_image
        )

    def cached_pending_workspace_preview_image(
        self,
    ) -> tuple[float, QtGui.QPixmap] | None:
        return self._pending_workspace_preview_for_kind("imagetool")

    def pending_workspace_tool_preview_image(
        self,
    ) -> tuple[float, QtGui.QPixmap] | None:
        return self._pending_workspace_preview_for_kind(
            "tool", self.manager._pending_workspace_tool_preview_image
        )

    def cached_pending_workspace_tool_preview_image(
        self,
    ) -> tuple[float, QtGui.QPixmap] | None:
        return self._pending_workspace_preview_for_kind("tool")

    @property
    def workspace_link_key(self) -> str | None:
        return self._workspace_link_key

    @property
    def workspace_link_colors(self) -> bool:
        return self._workspace_link_colors

    @property
    def workspace_linked(self) -> bool:
        if self.imagetool is not None and self.slicer_area.is_linked:
            return True
        return self._workspace_link_key is not None

    def set_workspace_link_state(self, key: str, *, link_colors: bool) -> None:
        self._workspace_link_key = key
        self._workspace_link_colors = bool(link_colors)

    def clear_workspace_link_state(self) -> None:
        self._workspace_link_key = None
        self._workspace_link_colors = True

    @property
    def name(self) -> str:
        if self.imagetool is not None:
            return _dataarray_name(self.slicer_area._data)
        if self.tool_window is not None:
            return self.tool_window._tool_display_name or self.tool_window.windowTitle()
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._set_name(name, manual=True)

    def _set_name(self, name: str, *, manual: bool) -> None:
        if self.tool_window is not None:
            self.tool_window._tool_display_name = name
            if self.manager._is_figure_node(self):
                self.manager._sync_figures_ui(select_uid=self.uid)
            else:
                self.manager.tree_view.refresh(self.uid)
            self.manager._mark_node_state_dirty(self.uid)
            return
        if self.imagetool is not None:
            self._rename_imagetool_data(name, record_provenance=manual)
            return
        self._name = name
        self._pending_workspace_metadata_cache = None
        self._pending_workspace_preview_cache = None
        self.manager.tree_view.refresh(self.uid)
        self.manager._mark_node_state_dirty(self.uid)

    def _rename_imagetool_data(self, name: str, *, record_provenance: bool) -> None:
        if self.imagetool is None:
            return
        if name == self.name:
            self.imagetool.setWindowTitle(self.label_text)
            return
        slicer_area = self.slicer_area
        slicer_area._data = slicer_area._data.rename(name)
        slicer_area.array_slicer._obj = slicer_area.array_slicer._obj.rename(name)
        if slicer_area._accepted_filter_data is not None:
            slicer_area._accepted_filter_data = (
                slicer_area._accepted_filter_data.rename(name)
            )
        if record_provenance:
            self._record_data_rename_provenance(name)
        self.imagetool.setWindowTitle(self.label_text)
        self.manager.tree_view.refresh(self.uid)
        self.manager._update_info(uid=self.uid)
        self.manager._refresh_dependency_dependents(self.uid)
        self.manager._mark_node_state_dirty(self.uid)

    def _record_data_rename_provenance(self, name: str) -> None:
        spec = self.provenance_spec
        if spec is not None:
            self._provenance_spec = _spec_with_final_data_name(spec, name)
            if self.imagetool is not None:
                self.imagetool.set_provenance_spec(self._provenance_spec)
        if self._source_spec is not None:
            self._source_spec = provenance.require_live_source_spec(
                _spec_with_final_data_name(self._source_spec, name)
            )

    def _file_label_paths(self) -> tuple[pathlib.Path, ...]:
        paths: list[pathlib.Path] = []
        if self.imagetool is not None and self.slicer_area._file_path is not None:
            _append_unique_path(paths, self.slicer_area._file_path)
        _collect_provenance_file_paths(self.displayed_provenance_spec, paths)
        return tuple(paths)

    @property
    def file_suffix_text(self) -> str:
        return _compact_file_suffix(self._file_label_paths())

    @property
    def base_label_text(self) -> str:
        return self.name

    @property
    def label_text(self) -> str:
        return self.base_label_text

    @property
    def display_text(self) -> str:
        return self.label_text if self.is_imagetool else self.name

    @property
    def type_badge_text(self) -> str | None:
        if self.tool_window is not None:
            return self.tool_window.tool_name
        if self.pending_workspace_tool_payload is not None:
            attrs = self.pending_workspace_payload_attrs or {}
            qualname = attrs.get("tool_cls_qualname")
            if isinstance(qualname, bytes):
                with contextlib.suppress(UnicodeDecodeError):
                    qualname = qualname.decode()
            if isinstance(qualname, str) and qualname:
                return qualname.rsplit(":", maxsplit=1)[-1].rsplit(".", maxsplit=1)[-1]
            display_name = attrs.get("tool_display_name")
            if isinstance(display_name, bytes):
                with contextlib.suppress(UnicodeDecodeError):
                    display_name = display_name.decode()
            if isinstance(display_name, str) and display_name:
                return display_name
        return None

    @property
    def info_text(self) -> str:
        if self.tool_window is not None:
            return erlab.interactive.utils._apply_qt_accent_color(
                self.tool_window.info_text
            )
        pending_info = self._pending_workspace_info_text()
        if pending_info is not None:
            return pending_info
        data = self._metadata_data()
        if data is None:
            return erlab.interactive.utils._apply_qt_accent_color(
                f"Added {self.added_time_display}"
            )
        text = erlab.utils.formatting.format_darr_html(
            data,
            show_size=True,
            additional_info=[f"Added {self.added_time_display}"],
        )
        return erlab.interactive.utils._apply_qt_accent_color(text)

    def _pending_workspace_info_text(self) -> str | None:
        pending = self._pending_workspace_payload
        kind = self._pending_workspace_payload_kind
        if pending is None or kind is None:
            return None
        cache_key = (kind, pending)
        if (
            self._pending_workspace_metadata_cache is not None
            and self._pending_workspace_metadata_cache[0] == cache_key
        ):
            return self._pending_workspace_metadata_cache[1]
        text = self.manager._pending_workspace_info_text(self)
        if text is not None:
            self._pending_workspace_metadata_cache = (cache_key, text)
        return text

    @property
    def tree_uid_text(self) -> str:
        return self.uid

    @property
    def created_time(self) -> datetime.datetime:
        return self._created_time

    @property
    def note(self) -> str:
        return self._note

    @note.setter
    def note(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("note must be a string")
        self._note = value

    @property
    def has_note(self) -> bool:
        return bool(self._note.strip())

    @property
    def added_time_iso(self) -> str:
        return self._created_time.isoformat(timespec="seconds")

    @property
    def added_time_display(self) -> str:
        return _format_added_time(self._created_time)

    def _metadata_data(self) -> xr.DataArray | None:
        if self.imagetool is not None:
            if self.pending_workspace_memory_payload is not None:
                return None
            return self.slicer_area.displayed_data
        if self.tool_window is not None:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                return self.tool_window.tool_data
        return None

    def _load_source_details(self) -> _LoadSourceDetails | None:
        if self.imagetool is not None:
            file_path = self.slicer_area._file_path
            if file_path is not None:
                return _load_source_details_from_file(
                    file_path,
                    self.slicer_area._load_func,
                    source_input_dtype=self._load_source_input_dtype(),
                )
        provenance_spec = self.provenance_spec
        if provenance_spec is not None and provenance_spec.file_load_source is not None:
            return _load_source_details_from_provenance(
                provenance_spec.file_load_source
            )
        return self._pending_workspace_load_source_details()

    def _pending_workspace_load_source_details(self) -> _LoadSourceDetails | None:
        attrs = self._pending_workspace_payload_attrs
        if attrs is None:
            return None
        raw_state = attrs.get("itool_state")
        if isinstance(raw_state, bytes):
            with contextlib.suppress(UnicodeDecodeError):
                raw_state = raw_state.decode()
        if not isinstance(raw_state, str):
            return None
        try:
            state = typing.cast("dict[str, typing.Any]", json.loads(raw_state))
        except Exception:
            return None
        file_path = state.get("file_path")
        if not isinstance(file_path, str):
            return None
        load_func = state.get("load_func")
        return _load_source_details_from_file(
            pathlib.Path(file_path),
            self._load_func_from_serialized_state(load_func),
            source_input_dtype=self._load_source_input_dtype(),
        )

    @staticmethod
    def _load_func_from_serialized_state(
        load_func: typing.Any,
    ) -> (
        tuple[typing.Callable[..., typing.Any] | str, dict[str, typing.Any], typing.Any]
        | None
    ):
        if not isinstance(load_func, list | tuple) or len(load_func) != 3:
            return None
        fn, kwargs, selection = load_func
        if not isinstance(fn, str) or not isinstance(kwargs, dict):
            return None
        if ":" in fn:
            try:
                mod_name, qual = fn.split(":", maxsplit=1)
                func_obj: typing.Any = importlib.import_module(mod_name)
                for attr in qual.split("."):
                    func_obj = getattr(func_obj, attr)
            except Exception:
                return None
            if not callable(func_obj):
                return None
            return func_obj, dict(kwargs), selection
        if fn in erlab.io.loaders:
            return fn, dict(kwargs), selection
        return None

    def load_source_code(self, *, assign: str = "data") -> str | None:
        if self.imagetool is not None:
            file_path = self.slicer_area._file_path
            if file_path is None:
                return None
            return _load_code_from_file_details(
                file_path,
                self.slicer_area._load_func,
                assign=assign,
                source_input_dtype=self._load_source_input_dtype(),
            )

        details = self._load_source_details()
        if details is None or details.load_code is None:
            return None
        if not erlab.interactive.utils._is_kwarg_name(assign):
            raise ValueError("assign must be a valid Python identifier")
        if assign == "data":
            return details.load_code
        try:
            return provenance._replace_code_identifiers(
                details.load_code,
                {"data": assign},
            )
        except SyntaxError:
            return None

    def default_load_source_name(self) -> str | None:
        if self.imagetool is not None:
            file_path = self.slicer_area._file_path
            if file_path is None:
                return None
            return _default_load_source_name(file_path)
        details = self._load_source_details()
        if details is None:
            return None
        return _default_load_source_name(details.path)

    def _load_source_input_dtype(self) -> np.dtype[typing.Any] | None:
        return None

    @property
    def reloadable(self) -> bool:
        """Return whether this live ImageTool node can reload its displayed data."""
        return self.imagetool is not None and self.slicer_area.reloadable

    @property
    def metadata_fields(self) -> list[_MetadataField]:
        tool_window = self.tool_window
        kind_value = "ImageTool"
        if not self.is_imagetool:
            if tool_window is None:
                if self.pending_workspace_tool_payload is None:
                    raise RuntimeError(
                        "Managed non-ImageTool node is missing its tool."
                    )
                kind_value = self.type_badge_text or "ToolWindow"
            else:
                kind_value = tool_window.tool_name

        fields = [
            _MetadataField(
                "Kind",
                kind_value,
            ),
            _MetadataField(
                "Added",
                self.added_time_display,
                monospace=True,
            ),
        ]

        data = self._metadata_data()
        load_source_details = self._load_source_details()
        if load_source_details is not None:
            fields.append(
                _MetadataField(
                    "File",
                    str(load_source_details.path),
                    monospace=True,
                    details=load_source_details,
                )
            )
        dependency_inputs = self.manager.dependency_input_summary_for_uid(self.uid)
        if dependency_inputs is not None:
            fields.append(_MetadataField("Inputs", dependency_inputs, wrap=True))
        if data is not None and data.chunks is not None:
            fields.append(
                _MetadataField(
                    "Chunks",
                    _format_chunk_summary(data),
                    monospace=True,
                )
            )
        return fields

    @property
    def _preview_image(self) -> tuple[float, QtGui.QPixmap]:
        if self.pending_workspace_memory_payload is not None:
            preview = self.cached_pending_workspace_preview_image()
            if preview is not None:
                return preview
            return float("NaN"), QtGui.QPixmap()
        return _preview_from_imagetool(self.imagetool, float("NaN"), QtGui.QPixmap())

    @property
    def source_spec(
        self,
    ) -> provenance.ToolProvenanceSpec | None:
        if self.tool_window is not None:
            return self.tool_window.source_spec
        return self._source_spec

    @property
    def displayed_source_spec(
        self,
    ) -> provenance.ToolProvenanceSpec | None:
        source_spec = self.source_spec
        if self.imagetool is not None:
            return self.slicer_area.displayed_live_source_spec(source_spec)
        return source_spec

    @property
    def source_binding(
        self,
    ) -> provenance.ImageToolSelectionSourceBinding | None:
        if self.tool_window is not None:
            return self.tool_window.source_binding
        return self._source_binding

    @property
    def provenance_spec(
        self,
    ) -> provenance.ToolProvenanceSpec | None:
        if self.tool_window is not None:
            return self.tool_window.current_provenance_spec()
        if self._provenance_spec is not None:
            return self._provenance_spec
        return self._source_spec

    @property
    def displayed_provenance_spec(
        self,
    ) -> provenance.ToolProvenanceSpec | None:
        if self.imagetool is not None:
            return self.slicer_area.displayed_provenance_spec(self.provenance_spec)
        return self.provenance_spec

    def persistence_data_backing(
        self,
    ) -> tuple[typing.Literal["dask", "file_lazy", "memory"] | None, tuple[str, ...]]:
        """Return lightweight data backing metadata without capturing UI state."""
        if self.pending_workspace_memory_payload is not None:
            return "memory", ()
        if self.imagetool is None:
            return None, ()

        slicer_area = self.slicer_area
        data = slicer_area._data
        if slicer_area.data_chunked:
            data_backing: typing.Literal["dask", "file_lazy", "memory"] = "dask"
        elif slicer_area.data_file_backed:
            data_backing = "file_lazy"
        else:
            data_backing = "memory"
        return data_backing, _manager_xarray.dataarray_source_paths(data)

    def persistence_view(
        self, *, materialize_pending: bool = True
    ) -> _NodePersistenceView:
        """Return the only manager persistence/clone view for this node."""
        if materialize_pending and not self.materialize_pending_workspace_payload():
            raise ValueError(
                "Could not read this node's saved data from the workspace file."
            )
        if self.imagetool is None:
            return _NodePersistenceView(
                data=None,
                state=None,
                provenance_spec=self.provenance_spec,
                source_spec=self.source_spec,
                source_binding=self.source_binding,
                output_id=self.output_id,
                source_state=self.source_state,
                source_auto_update=self.source_auto_update,
                data_backing=None,
            )

        data, state = self.slicer_area.persistence_data_and_state()
        data_backing, source_paths = self.persistence_data_backing()
        return _NodePersistenceView(
            data=data,
            state=state,
            provenance_spec=self.provenance_spec,
            source_spec=self.source_spec,
            source_binding=self.source_binding,
            output_id=self.output_id,
            source_state=self.source_state,
            source_auto_update=self.source_auto_update,
            data_backing=data_backing,
            source_paths=source_paths,
        )

    @property
    def snapshot_token(self) -> str:
        return self._snapshot_token

    @staticmethod
    def _is_live_source_spec(
        spec: provenance.ToolProvenanceSpec | None,
    ) -> bool:
        with contextlib.suppress(TypeError):
            return provenance.require_live_source_spec(spec) is not None
        return False

    @property
    def detached_live_parent_data(self) -> xr.DataArray | None:
        return self._detached_live_parent_data

    def _set_detached_live_parent_data(
        self,
        provenance_spec: provenance.ToolProvenanceSpec | None,
        parent_data: xr.DataArray | None,
    ) -> None:
        if parent_data is None or not self._is_live_source_spec(provenance_spec):
            self._detached_live_parent_data = None
            return
        self._detached_live_parent_data = parent_data.copy(deep=False)

    def _advance_snapshot_token(self, *, defer_refresh: bool = False) -> None:
        if self._suspend_snapshot_token_updates:
            return
        self._snapshot_token = uuid.uuid4().hex
        if defer_refresh:
            self.manager._queue_idle_work(
                ("snapshot-token-refresh", self.uid),
                functools.partial(self._flush_snapshot_token_refresh, self.uid),
            )
            return
        self._flush_snapshot_token_refresh(self.uid)

    def _flush_snapshot_token_refresh(self, uid: str) -> None:
        if uid != self.uid:
            return
        manager = self._manager()
        if manager is None or not erlab.interactive.utils.qt_is_valid(manager):
            return
        if manager._tool_graph.nodes.get(self.uid) is not self:
            return
        manager.tree_view.refresh(self.uid)
        manager._update_info(uid=self.uid)
        manager._refresh_dependency_dependents(self.uid)
        manager._mark_node_state_dirty(self.uid)

    @property
    def source_state(self) -> _source_state_type:
        if self.tool_window is not None:
            return self.tool_window.source_state
        return self._source_state

    @property
    def source_auto_update(self) -> bool:
        if self.tool_window is not None:
            return self.tool_window.source_auto_update
        return self._source_auto_update

    @property
    def has_source_binding(self) -> bool:
        if self.tool_window is not None:
            return self.tool_window.has_source_binding
        return (
            self._source_spec is not None
            or self._source_binding is not None
            or self._output_id is not None
        )

    @property
    def output_id(self) -> str | None:
        return self._output_id

    def set_displayed_provenance(
        self,
        provenance_spec: provenance.ToolProvenanceSpec | None,
        *,
        advance_snapshot: bool = True,
    ) -> None:
        self._provenance_spec = provenance.parse_tool_provenance_spec(provenance_spec)
        self._detached_live_parent_data = None
        if self.imagetool is not None:
            self.imagetool.set_provenance_spec(self.provenance_spec)
        if advance_snapshot:
            self._advance_snapshot_token()

    @property
    def derivation_entries(
        self,
    ) -> list[provenance.DerivationEntry]:
        return [row.entry for row in self.derivation_display_rows]

    @property
    def derivation_display_rows(
        self,
    ) -> list[provenance._ProvenanceDisplayRow]:
        if self.parent_uid is not None and self.source_spec is not None:
            rows: list[provenance._ProvenanceDisplayRow] = []
            parent = self.manager._parent_node(self)
            parent_provenance = parent.displayed_provenance_spec
            if parent_provenance is not None:
                rows.extend(parent_provenance.display_rows())
            source_spec = self.displayed_source_spec
            if source_spec is not None:
                rows.extend(
                    source_spec.display_rows(
                        scope="source",
                    )
                )
            return rows
        provenance_spec = self.displayed_provenance_spec
        if provenance_spec is None:
            return []
        return provenance_spec.display_rows()

    @property
    def derivation_lines(self) -> list[str]:
        return [entry.label for entry in self.derivation_entries]

    @property
    def derivation_code(self) -> str | None:
        provenance_spec = self.displayed_provenance_spec
        if provenance_spec is None:
            return None
        return provenance_spec.display_code()

    def add_child_reference(self, uid: str, window: QtWidgets.QWidget | None) -> None:
        if uid not in self._childtool_indices:
            self._childtool_indices.append(uid)
        if window is not None:
            self._childtools[uid] = window

    def remove_child_reference(self, uid: str) -> None:
        self._childtools.pop(uid, None)
        with contextlib.suppress(ValueError):
            self._childtool_indices.remove(uid)

    def set_source_binding(
        self,
        source_spec: provenance.ToolProvenanceSpec | None,
        *,
        source_binding: provenance.ImageToolSelectionSourceBinding | None = None,
        provenance_spec: provenance.ToolProvenanceSpec | None = None,
        auto_update: bool = False,
        state: _source_state_type = "fresh",
    ) -> None:
        """Bind this node to data selected from its parent ImageTool.

        Parameters
        ----------
        source_spec
            Current source spec used for derivation display and refresh. When provided,
            refresh applies this spec as stored.
        source_binding
            Legacy selection state from an ImageTool plot. Used only to materialize a
            missing ``source_spec`` once; explicit ``source_spec`` values take priority.
        provenance_spec
            Displayed provenance to show immediately. If omitted and the source is
            fresh, provenance is rebuilt from the parent and current source spec.
        auto_update
            Whether compatible parent changes should refresh this node automatically.
        state
            Current refresh state for manager status UI.
        """
        if source_spec is not None and not isinstance(
            source_spec,
            provenance.ToolProvenanceSpec,
        ):
            raise TypeError(
                "source_spec must be a ToolProvenanceSpec or None. Use "
                "parse_tool_provenance_spec() when deserializing saved payloads."
            )
        if source_binding is not None and not isinstance(
            source_binding,
            provenance.ImageToolSelectionSourceBinding,
        ):
            raise TypeError("source_binding must be an ImageToolSelectionSourceBinding")
        if source_spec is None and source_binding is not None and self.parent_uid:
            source_spec = source_binding.materialize(
                self.manager._parent_node(self).current_source_data()
            )
        self._detached_live_parent_data = None
        self._source_spec = provenance.require_live_source_spec(source_spec)
        self._source_binding = None if self._source_spec is not None else source_binding
        if provenance_spec is not None and not isinstance(
            provenance_spec,
            provenance.ToolProvenanceSpec,
        ):
            raise TypeError(
                "provenance_spec must be a ToolProvenanceSpec or None. Use "
                "parse_tool_provenance_spec() when deserializing saved payloads."
            )
        self._source_auto_update = bool(auto_update)
        self._output_id = None
        if provenance_spec is not None:
            self.set_displayed_provenance(provenance_spec)
        elif self.has_source_binding and state == "fresh" and self.parent_uid:
            parent = self.manager._parent_node(self)
            parent_data = parent.current_source_data()
            source_spec = self._materialized_source_spec(parent_data)
            self.set_displayed_provenance(
                provenance.compose_display_provenance(
                    parent.displayed_provenance_spec,
                    source_spec,
                    parent_data=parent_data,
                )
            )
        self._set_source_state(state if self.has_source_binding else "fresh")
        self.manager._mark_node_state_dirty(self.uid)

    def set_restored_source_binding_metadata(
        self,
        source_spec: provenance.ToolProvenanceSpec | None,
        source_binding: provenance.ImageToolSelectionSourceBinding | None,
        *,
        auto_update: bool,
        state: _source_state_type,
    ) -> None:
        """Restore saved source metadata without reading parent data."""
        if source_spec is not None and not isinstance(
            source_spec,
            provenance.ToolProvenanceSpec,
        ):
            raise TypeError("source_spec must be a ToolProvenanceSpec or None")
        if source_binding is not None and not isinstance(
            source_binding,
            provenance.ImageToolSelectionSourceBinding,
        ):
            raise TypeError("source_binding must be an ImageToolSelectionSourceBinding")
        self._source_spec = provenance.require_live_source_spec(source_spec)
        self._source_binding = None if self._source_spec is not None else source_binding
        self._source_auto_update = bool(auto_update)
        self._source_state = state if self.has_source_binding else "fresh"

    def set_output_binding(
        self,
        output_id: str,
        *,
        provenance_spec: provenance.ToolProvenanceSpec | None = None,
        auto_update: bool = False,
        state: _source_state_type = "fresh",
    ) -> None:
        if output_id is None:
            raise ValueError("output_id must not be None")
        if not isinstance(output_id, str):
            raise TypeError("output_id must be a string")
        if not output_id:
            raise ValueError("output_id must not be empty")
        if provenance_spec is not None and not isinstance(
            provenance_spec,
            provenance.ToolProvenanceSpec,
        ):
            raise TypeError(
                "provenance_spec must be a ToolProvenanceSpec or None. Use "
                "parse_tool_provenance_spec() when deserializing saved payloads."
            )
        self._source_spec = None
        self._source_binding = None
        self._detached_live_parent_data = None
        self._source_auto_update = bool(auto_update)
        self._output_id = output_id
        if provenance_spec is not None:
            self.set_displayed_provenance(provenance_spec)
        self._set_source_state(state)
        self.manager._mark_node_state_dirty(self.uid)

    def set_detached_provenance(
        self,
        provenance_spec: provenance.ToolProvenanceSpec | None,
        *,
        live_parent_data: xr.DataArray | None = None,
    ) -> None:
        if provenance_spec is not None and not isinstance(
            provenance_spec,
            provenance.ToolProvenanceSpec,
        ):
            raise TypeError(
                "provenance_spec must be a ToolProvenanceSpec or None. Use "
                "parse_tool_provenance_spec() when deserializing saved payloads."
            )
        self._source_spec = None
        self._source_binding = None
        self._source_auto_update = False
        self._output_id = None
        self.set_displayed_provenance(provenance_spec)
        self._set_detached_live_parent_data(provenance_spec, live_parent_data)
        self._set_source_state("fresh")
        self.manager._mark_node_state_dirty(self.uid)

    def _materialized_source_spec(
        self, parent_data: xr.DataArray
    ) -> provenance.ToolProvenanceSpec:
        """Return the source spec to apply to ``parent_data``."""
        if self._source_spec is not None:
            return self._source_spec
        if self._source_binding is not None:
            self._source_spec = self._source_binding.materialize(parent_data)
            self._source_binding = None
            return self._source_spec
        raise RuntimeError("Node is not bound to an ImageTool source.")

    def _resolved_output_payload(
        self,
    ) -> (
        tuple[
            xr.DataArray,
            provenance.ToolProvenanceSpec | None,
        ]
        | None
    ):
        if self._output_id is None:
            return None
        parent = self.manager._parent_node(self)
        tool_window = parent.tool_window
        if tool_window is None:
            return None
        data = tool_window.output_imagetool_data(self._output_id)
        if data is None:
            return None
        return data, tool_window.output_imagetool_provenance(self._output_id, data)

    @contextlib.contextmanager
    def _suspend_descendant_propagation(self) -> Iterator[None]:
        previous = self._suspend_descendant_signal_propagation
        self._suspend_descendant_signal_propagation = True
        try:
            yield
        finally:
            self._suspend_descendant_signal_propagation = previous

    @contextlib.contextmanager
    def _suspend_snapshot_updates(self) -> Iterator[None]:
        previous = self._suspend_snapshot_token_updates
        self._suspend_snapshot_token_updates = True
        try:
            yield
        finally:
            self._suspend_snapshot_token_updates = previous

    def _replace_imagetool_data(
        self,
        data: xr.DataArray,
        provenance_spec: provenance.ToolProvenanceSpec | None,
        *,
        state: _source_state_type = "fresh",
        propagate_descendants: bool,
        preserve_filter: bool = False,
        live_parent_data: xr.DataArray | None = None,
    ) -> None:
        accepted_filter_operation = None
        filtered = None
        if preserve_filter and self.imagetool is not None:
            accepted_filter_operation = (
                self.slicer_area._accepted_filter_provenance_operation
            )
            if accepted_filter_operation is not None:
                filtered = self.slicer_area._filter_operation_result_for_replacement(
                    data,
                    accepted_filter_operation,
                )
        with self._suspend_descendant_propagation(), self._suspend_snapshot_updates():
            self.slicer_area.replace_source_data(data)
            if accepted_filter_operation is not None:
                self.slicer_area._apply_filter_result(
                    typing.cast("xr.DataArray", filtered),
                    self.slicer_area._filter_func_from_operation(
                        accepted_filter_operation
                    ),
                    operation=accepted_filter_operation,
                    accept=True,
                )
        self.set_displayed_provenance(provenance_spec, advance_snapshot=False)
        self._set_detached_live_parent_data(provenance_spec, live_parent_data)
        self._advance_snapshot_token()
        self._set_source_state(state)
        self.manager._mark_node_data_dirty(self.uid)
        if propagate_descendants:
            if state == "fresh":
                self.manager._propagate_source_change_from_uid(self.uid)
            else:
                self.manager._mark_descendants_source_state(self.uid, state)

    def replace_with_detached_data(
        self,
        data: xr.DataArray,
        provenance_spec: provenance.ToolProvenanceSpec | None,
        *,
        propagate_descendants: bool = True,
        preserve_filter: bool = False,
        live_parent_data: xr.DataArray | None = None,
    ) -> None:
        """Replace displayed ImageTool data with detached provenance."""
        self._source_spec = None
        self._source_auto_update = False
        self._output_id = None
        self._replace_imagetool_data(
            data,
            provenance_spec,
            state="fresh",
            propagate_descendants=propagate_descendants,
            preserve_filter=preserve_filter,
            live_parent_data=live_parent_data,
        )

    def _handle_tool_data_changed(self) -> None:
        self.manager._note_interaction_activity()
        self.manager._mark_node_data_dirty(self.uid)
        self._advance_snapshot_token(defer_refresh=True)
        if self._suspend_descendant_signal_propagation:
            return
        self.manager._queue_idle_work(
            ("tool-data-refresh", self.uid),
            functools.partial(self._flush_tool_data_changed, self.uid),
        )

    def _flush_tool_data_changed(self, uid: str) -> None:
        if uid != self.uid:
            return
        manager = self._manager()
        if manager is None or not erlab.interactive.utils.qt_is_valid(manager):
            return
        if manager._tool_graph.nodes.get(self.uid) is not self:
            return
        tool_window = self.tool_window
        if tool_window is None:
            return
        if tool_window.source_state == "fresh":
            manager._propagate_source_change_from_uid(self.uid)
        else:
            manager._mark_descendants_source_state(self.uid, tool_window.source_state)
        manager.tree_view.refresh(self.uid)
        if tool_window.source_state == "fresh":
            manager._resume_pending_source_refreshes(self.uid)

    def _set_source_auto_update(self, value: bool) -> None:
        self._source_auto_update = bool(value)
        self.manager.tree_view.refresh(self.uid)
        self.manager._update_info(uid=self.uid)
        self.manager._mark_node_state_dirty(self.uid)

    def _set_source_state(self, state: _source_state_type) -> None:
        self._source_state = state
        self.manager.tree_view.refresh(self.uid)
        self.manager._update_info(uid=self.uid)
        self.manager._mark_node_state_dirty(self.uid)

    def current_source_data(self) -> xr.DataArray:
        if (
            self.pending_workspace_payload is not None
            and not self.materialize_pending_workspace_payload()
        ):
            raise ValueError(
                "Could not read this node's saved data from the workspace file."
            )
        if self.imagetool is not None:
            return self.slicer_area._tool_source_parent_data()
        if self.tool_window is not None:
            return self.tool_window.tool_data.copy(deep=False)
        raise ValueError("Managed node is not available")

    def parent_source_data(self) -> xr.DataArray:
        return self.manager._parent_source_data_for_uid(self.uid)

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        event_type = None if event is None else event.type()
        tracked_event_types = (
            QtCore.QEvent.Type.Show,
            QtCore.QEvent.Type.Hide,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.WindowStateChange,
        )
        if obj == self.window and event_type in tracked_event_types:
            if self.imagetool is not None and event_type == QtCore.QEvent.Type.Show:
                erlab.interactive.utils.single_shot(
                    self.slicer_area, 0, self.slicer_area._ensure_secondary_plots
                )
                erlab.interactive.utils.single_shot(
                    self.slicer_area, 0, self.slicer_area._update_if_delayed
                )

            mark_dirty = (
                self.manager._workspace_state.loading_depth == 0
                and self.manager._workspace_state.saving_depth == 0
                and not self.manager._workspace_state.closing_document
            )

            def _mark_visibility_changed() -> None:
                self.visibility_changed(mark_dirty=mark_dirty)

            erlab.interactive.utils.single_shot(
                self,
                0,
                _mark_visibility_changed,
            )
        return super().eventFilter(obj, event)

    @QtCore.Slot()
    @QtCore.Slot(str)
    def update_title(self, title: str | None = None) -> None:
        del title
        if self.imagetool is not None:
            self.imagetool.setWindowTitle(self.label_text)
            self.manager.tree_view.refresh(self.uid)

    @QtCore.Slot()
    def visibility_changed(self, *, mark_dirty: bool = True) -> None:
        window = self.window
        if isinstance(window, QtWidgets.QWidget):
            self._recent_geometry = window.geometry()
            if mark_dirty:
                self.manager._mark_node_state_dirty(self.uid)

    @QtCore.Slot()
    def show(self) -> None:
        if not self.materialize_pending_workspace_payload():
            return
        window = self.window
        if window is None:
            return
        if not window.isVisible() and self._recent_geometry is not None:
            window.setGeometry(self._recent_geometry)

        if sys.platform == "win32":  # pragma: no cover
            window.setWindowFlags(
                window.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint
            )
            window.show()
            window.setWindowFlags(
                window.windowFlags() & ~QtCore.Qt.WindowType.WindowStaysOnTopHint
            )

        window.show()
        window.activateWindow()
        window.raise_()

    @QtCore.Slot()
    def hide(self) -> None:
        window = self.window
        if window is not None:
            window.hide()

    @QtCore.Slot()
    def dispose(self) -> None:
        self.window = None

    @QtCore.Slot()
    def reload(self) -> None:
        if self.imagetool is not None:
            self.slicer_area.reload()

    def reload_source_data(self) -> bool:
        if self.tool_window is None:
            return False
        return self.manager._reload_source_chain_for_child(self.uid)

    def can_reload_source_data(self) -> bool:
        return (
            self.tool_window is not None
            and self.manager._reload_target_for_child(self.uid) is not None
        )

    def reload_unavailable_reason(self) -> str | None:
        if self.tool_window is None:
            return "The selected tool is no longer available. Select an open item."
        return self.manager._reload_unavailable_reason_for_child(self.uid)

    @QtCore.Slot()
    def _refresh_node_info(self) -> None:
        self.manager._update_info(uid=self.uid)
        self.manager.tree_view.refresh(self.uid)

    @QtCore.Slot()
    def _handle_tool_info_changed(self) -> None:
        manager = self._manager()
        if manager is None or not erlab.interactive.utils.qt_is_valid(manager):
            return
        if manager._tool_graph.nodes.get(self.uid) is not self:
            return
        manager._note_interaction_activity()
        manager._mark_tool_info_dirty(self.uid)
        manager._queue_idle_work(
            ("tool-info-refresh", self.uid),
            functools.partial(self._flush_tool_info_changed, self.uid),
        )

    def _flush_tool_info_changed(self, uid: str) -> None:
        if uid != self.uid:
            return
        manager = self._manager()
        if manager is None or not erlab.interactive.utils.qt_is_valid(manager):
            return
        if manager._tool_graph.nodes.get(self.uid) is not self:
            return
        manager._update_figure_gallery_icon(self.uid)
        manager._update_info(uid=self.uid)

    @QtCore.Slot()
    def _handle_tool_state_changed(self) -> None:
        manager = self._manager()
        if manager is None or not erlab.interactive.utils.qt_is_valid(manager):
            return
        if manager._tool_graph.nodes.get(self.uid) is not self:
            return
        manager._note_interaction_activity()
        manager._mark_node_state_dirty(self.uid)

    @QtCore.Slot()
    def _handle_imagetool_state_changed(self) -> None:
        if self.manager._workspace_state.closing_document:
            return
        self.manager._note_interaction_activity()
        self.manager._mark_node_state_dirty(self.uid)

    @QtCore.Slot()
    def _handle_imagetool_data_edited(self) -> None:
        self.manager._note_interaction_activity()
        self.manager._mark_node_data_dirty(self.uid)
        self._advance_snapshot_token(defer_refresh=True)

    @QtCore.Slot()
    def _handle_imagetool_backing_changed(self) -> None:
        self.manager._note_interaction_activity()
        self.manager._mark_node_data_dirty(self.uid)
        self._advance_snapshot_token(defer_refresh=True)

    def _update_from_parent_source(self) -> bool:
        if self.tool_window is not None:
            with self._suspend_descendant_propagation():
                updated = self.tool_window._update_from_parent_source()
            if updated and self.tool_window.source_state == "fresh":
                self.manager._propagate_source_change_from_uid(self.uid)
            elif self.tool_window.source_state != "fresh":
                self.manager._mark_descendants_source_state(
                    self.uid, self.tool_window.source_state
                )
            self.manager.tree_view.refresh(self.uid)
            return updated

        try:
            if self._output_id is not None:
                parent_tool = self.manager._parent_node(self).tool_window
                if parent_tool is not None and parent_tool.source_state != "fresh":
                    self._set_source_state(parent_tool.source_state)
                    self.manager._mark_descendants_source_state(
                        self.uid, parent_tool.source_state
                    )
                    return False
                payload = self._resolved_output_payload()
                if payload is None:
                    self._set_source_state("unavailable")
                    return False
                resolved, provenance_spec = payload
            else:
                if not self.has_source_binding:
                    return False
                parent_data = self.parent_source_data()
                source_spec = self._materialized_source_spec(parent_data)
                resolved = source_spec.apply(parent_data)
                provenance_spec = provenance.compose_display_provenance(
                    self.manager._parent_node(self).displayed_provenance_spec,
                    source_spec,
                    parent_data=parent_data,
                )
            self._replace_imagetool_data(
                resolved,
                provenance_spec,
                propagate_descendants=True,
                preserve_filter=True,
            )
        except Exception:
            self._set_source_state("unavailable")
            self.manager._mark_descendants_source_unavailable(self.uid)
            return False

        return True

    def handle_parent_source_replaced(self, parent_data: xr.DataArray) -> bool:
        if self.tool_window is not None:
            if not self.tool_window.has_source_binding:
                return False
            with self._suspend_descendant_propagation():
                self.tool_window.handle_parent_source_replaced(parent_data)
            return self.tool_window.source_state == "fresh"

        if self._output_id is not None and (
            not self._source_auto_update or self.imagetool is None
        ):
            # Output-bound child ImageTools may be expensive or currently deferred.
            # Defer recomputation until the user explicitly refreshes or opens them
            # instead of resolving payloads through a missing/hidden slicer.
            self._set_source_state("stale")
            return False

        if self.imagetool is None and self.has_source_binding:
            self._set_source_state("stale")
            return False

        try:
            if self._output_id is not None:
                payload = self._resolved_output_payload()
                if payload is None:
                    self._set_source_state("unavailable")
                    return False
                resolved, provenance_spec = payload
            else:
                if not self.has_source_binding:
                    return False
                source_spec = self._materialized_source_spec(parent_data)
                resolved = source_spec.apply(parent_data)
                provenance_spec = provenance.compose_display_provenance(
                    self.manager._parent_node(self).displayed_provenance_spec,
                    source_spec,
                    parent_data=parent_data,
                )
        except Exception:
            self._set_source_state("unavailable")
            return False

        if self._source_auto_update:
            try:
                self._replace_imagetool_data(
                    resolved,
                    provenance_spec,
                    propagate_descendants=False,
                    preserve_filter=True,
                )
            except Exception:
                self._set_source_state("unavailable")
                return False
            return True
        self._set_source_state("stale")
        return False

    def show_source_update_dialog(
        self, *, parent: QtWidgets.QWidget | None = None
    ) -> int:
        if not self.has_source_binding:
            return int(QtWidgets.QDialog.DialogCode.Rejected)

        dialog = erlab.interactive.utils._ToolSourceUpdateDialog(
            parent if parent is not None else self.manager,
            state=self.source_state,
            auto_update=self.source_auto_update,
        )
        result = dialog.exec()
        if result == int(QtWidgets.QDialog.DialogCode.Accepted):
            if self.tool_window is not None:
                self.tool_window._set_source_auto_update(
                    dialog.auto_update_check.isChecked()
                )
            else:
                self._set_source_auto_update(dialog.auto_update_check.isChecked())
            if dialog.update_requested and self.source_state == "stale":
                self.manager._refresh_source_chain_to_uid(self.uid)
        return result

    @QtCore.Slot(object)
    def _handle_source_data_replaced(self, parent_data: object) -> None:
        if self.pending_workspace_memory_payload is not None:
            if self.manager._workspace_state.loading_depth > 0:
                return
            self.clear_pending_workspace_payload()
        self.manager._mark_node_data_dirty(self.uid)
        self._advance_snapshot_token()
        if self._suspend_descendant_signal_propagation:
            return
        if not isinstance(parent_data, xr.DataArray):
            try:
                parent_data = self.current_source_data()
            except Exception:
                self.manager._mark_descendants_source_unavailable(self.uid)
                return
        self.manager._propagate_source_change_from_uid(self.uid, parent_data)


class _ImageToolWrapper(_ManagedWindowNode):
    """Root ImageTool node wrapper used by ImageToolManager."""

    def __init__(
        self,
        manager: ImageToolManager,
        index: int,
        uid: str,
        tool: ImageTool | None,
        watched_var: tuple[str, str] | None = None,
        watched_workspace_link_id: str | None = None,
        watched_source_label: str | None = None,
        watched_source_uid: str | None = None,
        watched_connected: bool = True,
        source_input_ndim: int | None = None,
        source_input_dtype: np.dtype[typing.Any] | str | None = None,
        *,
        provenance_spec: provenance.ToolProvenanceSpec | None = None,
        source_spec: provenance.ToolProvenanceSpec | None = None,
        source_binding: provenance.ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
        note: str | bytes | None = None,
        name: str | None = None,
    ) -> None:
        self._index = index
        self._watched_varname: str | None = None
        self._watched_uid: str | None = None
        self._watched_workspace_link_id: str | None = None
        self._watched_source_label: str | None = None
        self._watched_source_uid: str | None = None
        self._watched_connected: bool = False
        self._source_input_ndim = source_input_ndim
        self._source_input_dtype: np.dtype[typing.Any] | None = (
            np.dtype(source_input_dtype) if source_input_dtype is not None else None
        )
        if watched_var is not None:
            self.set_watched_binding(
                *watched_var,
                workspace_link_id=watched_workspace_link_id,
                source_label=watched_source_label,
                source_uid=watched_source_uid,
                connected=watched_connected,
            )

        super().__init__(
            manager,
            uid,
            None,
            tool,
            window_kind="imagetool",
            name=name,
            provenance_spec=provenance_spec,
            source_spec=source_spec,
            source_binding=source_binding,
            source_auto_update=source_auto_update,
            source_state=source_state,
            snapshot_token=snapshot_token,
            created_time=created_time,
            note=note,
        )

    @property
    def index(self) -> int:
        return self._index

    @property
    def watched(self) -> bool:
        return self._watched_varname is not None and self._watched_uid is not None

    def set_watched_binding(
        self,
        varname: str,
        uid: str,
        *,
        workspace_link_id: str | None = None,
        source_label: str | None = None,
        source_uid: str | None = None,
        connected: bool = True,
    ) -> None:
        """Bind this root ImageTool to a watched variable."""
        self._watched_varname = varname
        self._watched_uid = uid
        self._watched_workspace_link_id = workspace_link_id
        self._watched_source_label = source_label
        self._watched_source_uid = source_uid
        self._watched_connected = connected

    def watched_metadata(self) -> dict[str, typing.Any]:
        """Return JSON-serializable watched binding metadata."""
        if not self.watched:
            return {}
        return {
            "varname": self._watched_varname,
            "uid": self._watched_uid,
            "workspace_link_id": self._watched_workspace_link_id,
            "source_label": self._watched_source_label,
            "source_uid": self._watched_source_uid,
            "connected": self._watched_connected,
        }

    def set_source_input_ndim(self, ndim: int | None) -> None:
        """Track the latest dimensionality of the root source before UI promotion."""
        self._source_input_ndim = ndim
        self.manager._mark_node_state_dirty(self.uid)

    def set_source_input_dtype(self, dtype: np.dtype[typing.Any] | str | None) -> None:
        """Track the latest dtype of the root source before UI promotion."""
        self._source_input_dtype = np.dtype(dtype) if dtype is not None else None
        self.manager._mark_node_state_dirty(self.uid)

    @property
    def source_input_ndim(self) -> int | None:
        return self._source_input_ndim

    def _load_source_input_dtype(self) -> np.dtype[typing.Any] | None:
        return self._source_input_dtype

    def _watched_root_provenance_spec(
        self,
    ) -> provenance.ToolProvenanceSpec | None:
        varname = self._watched_varname
        if (
            not self.watched
            or self._provenance_spec is not None
            or self._source_spec is not None
            or varname is None
            or not varname.isidentifier()
            or keyword.iskeyword(varname)
        ):
            return None
        seed_source = varname
        if self._source_input_dtype not in (
            None,
            np.dtype(np.float32),
            np.dtype(np.float64),
        ):
            seed_source = f"{seed_source}.astype(np.float64)"
        return provenance.script(
            start_label=f"Start from watched variable {varname!r}",
            seed_code=f"derived = {seed_source}",
            active_name="derived",
        )

    @property
    def provenance_spec(
        self,
    ) -> provenance.ToolProvenanceSpec | None:
        base_provenance = super().provenance_spec
        if base_provenance is not None:
            return base_provenance
        return self._watched_root_provenance_spec()

    @property
    def label_text(self) -> str:
        title = f"{self.index}"
        base_label = self.base_label_text
        if base_label:
            title += f": {base_label}"
        title += self.file_suffix_text
        return title

    def current_source_data(self) -> xr.DataArray:
        data = super().current_source_data()
        if self._source_input_ndim == 1:
            return provenance.mark_promoted_1d_source(data)
        return data

    @QtCore.Slot()
    def unwatch(self) -> None:
        if self.watched:
            self.manager._sigWatchedDataEdited.emit(
                self._watched_varname, self._watched_uid, "removed"
            )
            self._watched_varname = None
            self._watched_uid = None
            self._watched_workspace_link_id = None
            self._watched_source_label = None
            self._watched_source_uid = None
            self._watched_connected = False
            self.manager.tree_view.refresh(self.index)
            self.manager._mark_node_state_dirty(self.uid)

    @QtCore.Slot()
    def _trigger_watched_update(self) -> None:
        if self.watched:
            self.manager._sigWatchedDataEdited.emit(
                self._watched_varname, self._watched_uid, "updated"
            )

    @property
    def info_text(self) -> str:
        return super().info_text

    @property
    def display_text(self) -> str:
        return self.label_text

    @property
    def _preview_image(self) -> tuple[float, QtGui.QPixmap]:
        return super()._preview_image

    @property
    def window(self) -> QtWidgets.QWidget | None:
        return super().window

    @window.setter
    def window(self, value: QtWidgets.QWidget | None) -> None:
        old_imagetool = self.imagetool
        base_window: typing.Any = _ManagedWindowNode.window
        base_window.fset(self, value)
        if old_imagetool is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                old_imagetool.slicer_area.sigDataEdited.disconnect(
                    self._trigger_watched_update
                )
        if isinstance(value, ImageTool):
            value.slicer_area.sigDataEdited.connect(self._trigger_watched_update)

    def take_window(self) -> QtWidgets.QWidget | None:
        old_imagetool = self.imagetool
        window = super().take_window()
        if old_imagetool is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                old_imagetool.slicer_area.sigDataEdited.disconnect(
                    self._trigger_watched_update
                )
        return window
