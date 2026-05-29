# ruff: noqa: E501
"""Managed window nodes shown in ImageToolManager."""

from __future__ import annotations

__all__ = ["_ImageToolWrapper", "_ManagedWindowNode"]

import contextlib
import datetime
import keyword
import logging
import sys
import typing
import uuid
import weakref
from dataclasses import dataclass

import numpy as np
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool._load_source import (
    _default_load_source_name,
    _load_code_from_file_details,
    _load_source_details_from_file,
    _load_source_details_from_provenance,
    _LoadSourceDetails,
)
from erlab.interactive.imagetool._mainwindow import ImageTool

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.provenance_framework import (
        ImageToolSelectionSourceBinding,
    )
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

    if pixmap.isNull():
        return fallback_ratio, fallback_pixmap

    return height / width, pixmap.transformed(QtGui.QTransform().scale(1.0, -1.0))


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
    provenance_spec: (
        erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None
    )
    source_spec: (
        erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None
    )
    source_binding: ImageToolSelectionSourceBinding | None
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


class _ManagedWindowNode(QtCore.QObject):
    """A recursively managed window node in ImageToolManager."""

    _source_state_type = typing.Literal["fresh", "stale", "unavailable"]

    def __init__(
        self,
        manager: ImageToolManager,
        uid: str,
        parent_uid: str | None,
        window: QtWidgets.QWidget,
        *,
        provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None = None,
        source_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None = None,
        source_binding: ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _source_state_type = "fresh",
        output_id: str | None = None,
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
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
        self._window_kind: typing.Literal["imagetool", "tool"] = (
            "imagetool" if isinstance(window, ImageTool) else "tool"
        )
        self._name = window.windowTitle()
        self._name_manually_overridden = (
            isinstance(window, ImageTool)
            and self._name.replace("[*]", "") != window.slicer_area.display_name
        )

        self._source_spec: (
            erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None
        ) = None
        self._source_binding: ImageToolSelectionSourceBinding | None = None
        self._provenance_spec: (
            erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None
        ) = None
        self._source_state: _ManagedWindowNode._source_state_type = "fresh"
        self._source_auto_update: bool = False
        self._output_id: str | None = None
        self._suspend_descendant_signal_propagation: bool = False
        self._snapshot_token = (
            str(snapshot_token) if snapshot_token else uuid.uuid4().hex
        )
        self._suspend_snapshot_token_updates = True

        self.window = window
        try:
            if source_spec is not None:
                self.set_source_binding(
                    source_spec,
                    source_binding=source_binding,
                    provenance_spec=provenance_spec,
                    auto_update=source_auto_update,
                    state=source_state,
                )
            elif source_binding is not None:
                self.set_source_binding(
                    None,
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
            self._detach_imagetool()
        elif self.tool_window is not None:
            old = self.tool_window
            with contextlib.suppress(TypeError, RuntimeError):
                old.sigInfoChanged.disconnect(self._handle_tool_info_changed)
            with contextlib.suppress(TypeError, RuntimeError):
                old.sigDataChanged.disconnect(self._handle_tool_data_changed)
            old.removeEventFilter(self)
            old._set_managed_source_update_dialog(None)
            old._set_managed_source_reload(None)
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
            return

        tool = typing.cast("erlab.interactive.utils.ToolWindow", value)
        self._window_kind = "tool"
        self._tool_window = tool
        self._imagetool = None
        self.manager._install_workspace_save_shortcut(tool)
        tool.installEventFilter(self)
        tool.sigInfoChanged.connect(self._handle_tool_info_changed)
        tool.sigDataChanged.connect(self._handle_tool_data_changed)
        tool.destroyed.connect(self._handle_tool_window_destroyed)
        tool._set_managed_source_update_dialog(self.show_source_update_dialog)
        tool._set_managed_source_reload(
            self.reload_source_data, self.can_reload_source_data
        )

    def _handle_tool_window_destroyed(self, _obj: QtCore.QObject | None = None) -> None:
        manager = self._manager()
        if manager is None:
            return
        if manager._all_nodes.get(self.uid) is not self:
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
    def name(self) -> str:
        if self.tool_window is not None:
            return self.tool_window._tool_display_name or self.tool_window.windowTitle()
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._set_name(name, manual=True)

    def _set_name(self, name: str, *, manual: bool) -> None:
        if manual:
            self._name_manually_overridden = True
        if self.tool_window is not None:
            self.tool_window._tool_display_name = name
            self.manager.tree_view.refresh(self.uid)
            self.manager._mark_node_state_dirty(self.uid)
            return
        if name == self._name and self.imagetool is not None and not manual:
            return
        self._name = name
        if self.imagetool is not None:
            self.imagetool.setWindowTitle(self.label_text)
        self.manager.tree_view.refresh(self.uid)
        self.manager._mark_node_state_dirty(self.uid)

    @property
    def label_text(self) -> str:
        return self._name

    @property
    def display_text(self) -> str:
        return self.label_text if self.is_imagetool else self.name

    @property
    def type_badge_text(self) -> str | None:
        if self.tool_window is not None:
            return self.tool_window.tool_name
        return None

    @property
    def info_text(self) -> str:
        if self.tool_window is not None:
            return erlab.interactive.utils._apply_qt_accent_color(
                self.tool_window.info_text
            )
        text = erlab.utils.formatting.format_darr_html(
            self.slicer_area.displayed_data,
            show_size=True,
            additional_info=[f"Added {self.added_time_display}"],
        )
        return erlab.interactive.utils._apply_qt_accent_color(text)

    @property
    def tree_uid_text(self) -> str:
        return self.uid

    @property
    def created_time(self) -> datetime.datetime:
        return self._created_time

    @property
    def added_time_iso(self) -> str:
        return self._created_time.isoformat(timespec="seconds")

    @property
    def added_time_display(self) -> str:
        return _format_added_time(self._created_time)

    def _metadata_data(self) -> xr.DataArray | None:
        if self.imagetool is not None:
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
        return None

    def load_source_code(self, *, assign: str = "data") -> str | None:
        if self.imagetool is None:
            return None
        file_path = self.slicer_area._file_path
        if file_path is None:
            return None
        return _load_code_from_file_details(
            file_path,
            self.slicer_area._load_func,
            assign=assign,
            source_input_dtype=self._load_source_input_dtype(),
        )

    def default_load_source_name(self) -> str | None:
        if self.imagetool is None:
            return None
        file_path = self.slicer_area._file_path
        if file_path is None:
            return None
        return _default_load_source_name(file_path)

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
                raise RuntimeError("Managed non-ImageTool node is missing its tool.")
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
        return _preview_from_imagetool(self.imagetool, float("NaN"), QtGui.QPixmap())

    @property
    def source_spec(
        self,
    ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
        if self.tool_window is not None:
            return self.tool_window.source_spec
        return self._source_spec

    @property
    def displayed_source_spec(
        self,
    ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
        source_spec = self.source_spec
        if self.imagetool is not None:
            return self.slicer_area.displayed_live_source_spec(source_spec)
        return source_spec

    @property
    def source_binding(
        self,
    ) -> ImageToolSelectionSourceBinding | None:
        if self.tool_window is not None:
            return self.tool_window.source_binding
        return self._source_binding

    @property
    def provenance_spec(
        self,
    ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
        if self.tool_window is not None:
            return self.tool_window.current_provenance_spec()
        if self._provenance_spec is not None:
            return self._provenance_spec
        return self._source_spec

    @property
    def displayed_provenance_spec(
        self,
    ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
        if self.imagetool is not None:
            return self.slicer_area.displayed_provenance_spec(self.provenance_spec)
        return self.provenance_spec

    def persistence_view(self) -> _NodePersistenceView:
        """Return the only manager persistence/clone view for this node."""
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
        from erlab.interactive.imagetool.manager import _xarray as _manager_xarray

        data_backing: typing.Literal["dask", "file_lazy", "memory"]
        if data.chunks is not None:
            data_backing = "dask"
        elif _manager_xarray.dataarray_is_file_backed(data):
            data_backing = "file_lazy"
        else:
            data_backing = "memory"
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
            source_paths=_manager_xarray.dataarray_source_paths(data),
        )

    @property
    def snapshot_token(self) -> str:
        return self._snapshot_token

    def _advance_snapshot_token(self) -> None:
        if self._suspend_snapshot_token_updates:
            return
        self._snapshot_token = uuid.uuid4().hex
        self.manager.tree_view.refresh(self.uid)
        self.manager._update_info(uid=self.uid)
        self.manager._refresh_dependency_dependents(self.uid)
        self.manager._mark_node_state_dirty(self.uid)

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
        provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None,
        *,
        advance_snapshot: bool = True,
    ) -> None:
        self._provenance_spec = (
            erlab.interactive.imagetool.provenance_framework.parse_tool_provenance_spec(
                provenance_spec
            )
        )
        if self.imagetool is not None:
            self.imagetool.set_provenance_spec(self.provenance_spec)
        if advance_snapshot:
            self._advance_snapshot_token()

    @property
    def derivation_entries(
        self,
    ) -> list[erlab.interactive.imagetool.provenance_framework.DerivationEntry]:
        provenance_spec = self.displayed_provenance_spec
        if provenance_spec is None:
            return []
        return provenance_spec.display_entries()

    @property
    def derivation_lines(self) -> list[str]:
        return [entry.label for entry in self.derivation_entries]

    @property
    def derivation_code(self) -> str | None:
        provenance_spec = self.displayed_provenance_spec
        if provenance_spec is None:
            return None
        return provenance_spec.display_code()

    def add_child_reference(self, uid: str, window: QtWidgets.QWidget) -> None:
        if uid not in self._childtool_indices:
            self._childtool_indices.append(uid)
        self._childtools[uid] = window

    def remove_child_reference(self, uid: str) -> None:
        self._childtools.pop(uid, None)
        with contextlib.suppress(ValueError):
            self._childtool_indices.remove(uid)

    def set_source_binding(
        self,
        source_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None,
        *,
        source_binding: ImageToolSelectionSourceBinding | None = None,
        provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None = None,
        auto_update: bool = False,
        state: _source_state_type = "fresh",
    ) -> None:
        """Bind this node to data selected from its parent ImageTool.

        Parameters
        ----------
        source_spec
            Current source spec used for derivation display and older saved
            workspaces. If ``source_binding`` is not provided, refresh applies this
            spec as stored.
        source_binding
            Selection state from an ImageTool plot. When provided, refresh first builds
            a new ``source_spec`` from the current parent data.
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
            erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec,
        ):
            raise TypeError(
                "source_spec must be a ToolProvenanceSpec or None. Use "
                "parse_tool_provenance_spec() when deserializing saved payloads."
            )
        if source_binding is not None and not isinstance(
            source_binding,
            erlab.interactive.imagetool.provenance_framework.ImageToolSelectionSourceBinding,
        ):
            raise TypeError("source_binding must be an ImageToolSelectionSourceBinding")
        self._source_binding = source_binding
        if source_spec is None and self._source_binding is not None and self.parent_uid:
            source_spec = self._source_binding.materialize(
                self.manager._parent_node(self).current_source_data()
            )
        self._source_spec = (
            erlab.interactive.imagetool.provenance_framework.require_live_source_spec(
                source_spec
            )
        )
        if provenance_spec is not None and not isinstance(
            provenance_spec,
            erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec,
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
                erlab.interactive.imagetool.provenance_framework.compose_display_provenance(
                    parent.displayed_provenance_spec,
                    source_spec,
                    parent_data=parent_data,
                )
            )
        self._set_source_state(state if self.has_source_binding else "fresh")
        self.manager._mark_node_state_dirty(self.uid)

    def set_output_binding(
        self,
        output_id: str,
        *,
        provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None = None,
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
            erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec,
        ):
            raise TypeError(
                "provenance_spec must be a ToolProvenanceSpec or None. Use "
                "parse_tool_provenance_spec() when deserializing saved payloads."
            )
        self._source_spec = None
        self._source_binding = None
        self._source_auto_update = bool(auto_update)
        self._output_id = output_id
        if provenance_spec is not None:
            self.set_displayed_provenance(provenance_spec)
        self._set_source_state(state)
        self.manager._mark_node_state_dirty(self.uid)

    def set_detached_provenance(
        self,
        provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None,
    ) -> None:
        if provenance_spec is not None and not isinstance(
            provenance_spec,
            erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec,
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
        self._set_source_state("fresh")
        self.manager._mark_node_state_dirty(self.uid)

    def _materialized_source_spec(
        self, parent_data: xr.DataArray
    ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec:
        """Return the source spec to apply to ``parent_data``."""
        if self._source_binding is not None:
            self._source_spec = self._source_binding.materialize(parent_data)
            return self._source_spec
        if self._source_spec is None:
            raise RuntimeError("Node is not bound to an ImageTool source.")
        return self._source_spec

    def _resolved_output_payload(
        self,
    ) -> (
        tuple[
            xr.DataArray,
            erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None,
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
        provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None,
        *,
        state: _source_state_type = "fresh",
        propagate_descendants: bool,
        preserve_filter: bool = False,
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
        provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None,
        *,
        propagate_descendants: bool = True,
        preserve_filter: bool = False,
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
        )

    def _handle_tool_data_changed(self) -> None:
        self.manager._mark_node_data_dirty(self.uid)
        self._advance_snapshot_token()
        if self._suspend_descendant_signal_propagation:
            return
        tool_window = self.tool_window
        if tool_window is None:
            return
        if tool_window.source_state == "fresh":
            self.manager._propagate_source_change_from_uid(self.uid)
        else:
            self.manager._mark_descendants_source_state(
                self.uid, tool_window.source_state
            )
        self.manager.tree_view.refresh(self.uid)
        if tool_window.source_state == "fresh":
            self.manager._resume_pending_source_refreshes(self.uid)

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
                    self.slicer_area, 0, self.slicer_area._update_if_delayed
                )

            mark_dirty = (
                self.manager._workspace_loading_depth == 0
                and self.manager._workspace_saving_depth == 0
                and not self.manager._closing_workspace_document
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
        if self.imagetool is not None:
            if title is None:
                title = self.imagetool.windowTitle()
            title = title.replace("[*]", "")
            if self._name_manually_overridden:
                self.imagetool.setWindowTitle(self.label_text)
                return
            self._set_name(title, manual=False)

    @QtCore.Slot()
    def visibility_changed(self, *, mark_dirty: bool = True) -> None:
        window = self.window
        if isinstance(window, QtWidgets.QWidget):
            self._recent_geometry = window.geometry()
            if mark_dirty:
                self.manager._mark_node_state_dirty(self.uid)

    @QtCore.Slot()
    def show(self) -> None:
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

    @QtCore.Slot()
    def _refresh_node_info(self) -> None:
        self.manager._update_info(uid=self.uid)
        self.manager.tree_view.refresh(self.uid)

    @QtCore.Slot()
    def _handle_tool_info_changed(self) -> None:
        self.manager._mark_tool_info_dirty(self.uid)
        self.manager._schedule_tool_metadata_update(self.uid)

    @QtCore.Slot()
    def _handle_imagetool_state_changed(self) -> None:
        if self.manager._closing_workspace_document:
            return
        self.manager._mark_node_state_dirty(self.uid)

    @QtCore.Slot()
    def _handle_imagetool_data_edited(self) -> None:
        self.manager._mark_node_data_dirty(self.uid)
        self._advance_snapshot_token()

    @QtCore.Slot()
    def _handle_imagetool_backing_changed(self) -> None:
        self.manager._mark_node_data_dirty(self.uid)
        self._advance_snapshot_token()
        self._refresh_node_info()

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
                provenance_spec = erlab.interactive.imagetool.provenance_framework.compose_display_provenance(
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

        if self._output_id is not None and not self._source_auto_update:
            # Output-bound child ImageTools may be expensive to regenerate. When live
            # updates are disabled, defer the recomputation until the user explicitly
            # refreshes instead of resolving the payload just to mark the child stale.
            self._set_source_state("stale")
            return False

        if self.imagetool is None and self.has_source_binding:
            self._set_source_state("unavailable")
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
                provenance_spec = erlab.interactive.imagetool.provenance_framework.compose_display_provenance(
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
        tool: ImageTool,
        watched_var: tuple[str, str] | None = None,
        watched_workspace_link_id: str | None = None,
        watched_source_label: str | None = None,
        watched_source_uid: str | None = None,
        watched_connected: bool = True,
        source_input_ndim: int | None = None,
        source_input_dtype: np.dtype[typing.Any] | str | None = None,
        *,
        provenance_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None = None,
        source_spec: erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec
        | None = None,
        source_binding: ImageToolSelectionSourceBinding | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
        snapshot_token: str | None = None,
        created_time: datetime.datetime | str | bytes | None = None,
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
            provenance_spec=provenance_spec,
            source_spec=source_spec,
            source_binding=source_binding,
            source_auto_update=source_auto_update,
            source_state=source_state,
            snapshot_token=snapshot_token,
            created_time=created_time,
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
    ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
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
        return erlab.interactive.imagetool.provenance_framework.script(
            start_label=f"Start from watched variable {varname!r}",
            seed_code=f"derived = {seed_source}",
            active_name="derived",
        )

    @property
    def provenance_spec(
        self,
    ) -> erlab.interactive.imagetool.provenance_framework.ToolProvenanceSpec | None:
        base_provenance = super().provenance_spec
        if base_provenance is not None:
            return base_provenance
        return self._watched_root_provenance_spec()

    @property
    def label_text(self) -> str:
        title = f"{self.index}"
        if self._name:
            title += f": {self._name}"
        return title

    def current_source_data(self) -> xr.DataArray:
        data = super().current_source_data()
        if self._source_input_ndim == 1:
            return erlab.interactive.imagetool.provenance_framework.mark_promoted_1d_source(
                data
            )
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
