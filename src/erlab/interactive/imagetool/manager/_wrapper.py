"""Managed window nodes shown in ImageToolManager."""

from __future__ import annotations

__all__ = ["_ImageToolWrapper", "_ManagedWindowNode"]

import contextlib
import datetime
import importlib
import os
import sys
import typing
import weakref
from dataclasses import dataclass

import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool._mainwindow import ImageTool

if typing.TYPE_CHECKING:
    from pathlib import Path

    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.viewer import ImageSlicerArea


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
class _LoadSourceDetails:
    path: Path
    loader_label: str
    loader_text: str
    kwargs_text: str
    load_code: str | None


@dataclass(frozen=True)
class _MetadataField:
    label: str
    value: str
    monospace: bool = False
    wrap: bool = False
    details: _LoadSourceDetails | None = None


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


def _load_code_from_file_details(
    file_path: Path,
    load_func: tuple[typing.Callable | str, dict[str, typing.Any], int] | None,
) -> str | None:
    if load_func is None or load_func[2] != 0:
        return None

    imports: list[str] = []
    loader = load_func[0]
    if isinstance(loader, str):
        imports.append("import erlab")
        loader_expr = f"erlab.io.loaders[{loader!r}].load"
    else:
        callable_loader_expr = _loader_callable_text(loader)
        if callable_loader_expr is None:
            return None
        loader_expr = callable_loader_expr
        imports.insert(0, f"import {loader_expr.rpartition('.')[0]}")

    kwargs = load_func[1]
    kwargs_str = (
        erlab.interactive.utils.format_kwargs(
            typing.cast("dict[typing.Hashable, typing.Any]", kwargs)
        )
        if kwargs
        else ""
    )
    call_args = [repr(str(file_path))]
    if kwargs_str:
        call_args.append(kwargs_str)
    imports = list(dict.fromkeys(imports))
    return "\n".join(
        [
            *imports,
            "",
            f"data = {loader_expr}({', '.join(call_args)})",
        ]
    )


def _load_source_label_and_text(
    load_func: tuple[typing.Callable | str, dict[str, typing.Any], int] | None,
) -> tuple[str, str]:
    if load_func is None:
        return "Loader", "(unavailable)"

    loader = load_func[0]
    if isinstance(loader, str):
        return "Loader", loader

    loader_text = _loader_callable_text(loader)
    if loader_text is None:
        return "Load Function", repr(loader)
    return "Load Function", loader_text


def _loader_callable_text(loader: typing.Callable) -> str | None:
    module = getattr(loader, "__module__", None)
    qualname = getattr(loader, "__qualname__", getattr(loader, "__name__", None))
    if module is None or qualname is None or "<locals>" in qualname:
        return None

    # Prefer a public top-level alias when the callable is re-exported there so the
    # manager shows stable, user-facing load code instead of private module paths.
    top_level = module.split(".", maxsplit=1)[0]
    try:
        package = importlib.import_module(top_level)
    except Exception:
        return f"{module}.{qualname}"

    target: typing.Any = package
    for attr in qualname.split("."):
        target = getattr(target, attr, None)
        if target is None:
            break
    if target is loader:
        return f"{top_level}.{qualname}"
    return f"{module}.{qualname}"


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
        provenance_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_auto_update: bool = False,
        source_state: _source_state_type = "fresh",
    ) -> None:
        super().__init__(manager)
        self._manager = weakref.ref(manager)
        self.uid = uid
        self.parent_uid = parent_uid
        self._recent_geometry: QtCore.QRect | None = None
        self._created_time = datetime.datetime.now()

        self._childtools: dict[str, QtWidgets.QWidget] = {}
        self._childtool_indices: list[str] = []

        self._imagetool: ImageTool | None = None
        self._tool_window: erlab.interactive.utils.ToolWindow | None = None
        self._window_kind: typing.Literal["imagetool", "tool"] = (
            "imagetool" if isinstance(window, ImageTool) else "tool"
        )
        self._name = window.windowTitle()
        self._archived_fname: str | None = None
        self._info_text_archived: str = ""
        self._metadata_fields_archived: list[_MetadataField] = []
        self._box_ratio_archived: float = float("NaN")
        self._pixmap_archived: QtGui.QPixmap = QtGui.QPixmap()

        self._source_spec: (
            erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None
        ) = None
        self._provenance_spec: (
            erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None
        ) = None
        self._source_state: _ManagedWindowNode._source_state_type = "fresh"
        self._source_auto_update: bool = False

        self.touch_timer = QtCore.QTimer(self)
        self.touch_timer.setInterval(12 * 60 * 60 * 1000)
        self.touch_timer.timeout.connect(self.touch_archive)

        self.window = window
        if source_spec is not None:
            self.set_source_binding(
                source_spec,
                auto_update=source_auto_update,
                state=source_state,
            )
        elif provenance_spec is not None:
            self.set_detached_provenance(provenance_spec)

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
        old = self.window
        if isinstance(old, ImageTool):
            old.slicer_area.unlink()
            old.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            old.removeEventFilter(self)
            with contextlib.suppress(TypeError, RuntimeError):
                old.sigTitleChanged.disconnect(self.update_title)
            with contextlib.suppress(TypeError, RuntimeError):
                old.slicer_area.sigSourceDataReplaced.disconnect(
                    self._handle_source_data_replaced
                )
            old.close()
            self._imagetool = None
        elif isinstance(old, erlab.interactive.utils.ToolWindow):
            with contextlib.suppress(TypeError, RuntimeError):
                old.sigInfoChanged.disconnect(self._refresh_node_info)
            old.set_source_parent_fetcher(None)
            old.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            old.close()
            self._tool_window = None

        if value is None:
            return

        if isinstance(value, ImageTool):
            self._window_kind = "imagetool"
            self._imagetool = value
            self._tool_window = None
            value.installEventFilter(self)
            value.sigTitleChanged.connect(self.update_title)
            value.slicer_area.sigSourceDataReplaced.connect(
                self._handle_source_data_replaced
            )
            value.slicer_area._in_manager = True
            return

        tool = typing.cast("erlab.interactive.utils.ToolWindow", value)
        self._window_kind = "tool"
        self._tool_window = tool
        self._imagetool = None
        tool.sigInfoChanged.connect(self._refresh_node_info)
        tool.destroyed.connect(
            lambda _=None, uid=self.uid: self.manager._remove_childtool(uid)
        )

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
    def archived(self) -> bool:
        return self.is_imagetool and self.imagetool is None

    @property
    def name(self) -> str:
        if self.tool_window is not None:
            return self.tool_window._tool_display_name or self.tool_window.windowTitle()
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if self.tool_window is not None:
            self.tool_window._tool_display_name = name
            self.manager.tree_view.refresh(self.uid)
            return
        if name == self._name and self.imagetool is not None:
            return
        self._name = name
        if self.imagetool is not None:
            self.imagetool.setWindowTitle(self.label_text)
        self.manager.tree_view.refresh(self.uid)

    @property
    def label_text(self) -> str:
        return self._name

    @property
    def display_text(self) -> str:
        return self.label_text if self.is_imagetool else self.name

    @property
    def info_text(self) -> str:
        if self.tool_window is not None:
            return erlab.interactive.utils._apply_qt_accent_color(
                self.tool_window.info_text
            )
        if self.archived:
            text = self._info_text_archived
        else:
            text = erlab.utils.formatting.format_darr_html(
                self.slicer_area._data,
                show_size=True,
                additional_info=[
                    f"Added {self._created_time.isoformat(sep=' ', timespec='seconds')}"
                ],
            )
        return erlab.interactive.utils._apply_qt_accent_color(text)

    @property
    def tree_uid_text(self) -> str:
        return self.uid

    def _metadata_data(self) -> xr.DataArray | None:
        if self.imagetool is not None:
            return self.slicer_area._data
        if self.tool_window is not None:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                return self.tool_window.tool_data
        return None

    def _load_source_details(self) -> _LoadSourceDetails | None:
        if self.imagetool is not None:
            file_path = self.slicer_area._file_path
            if file_path is None:
                return None
            load_func = self.slicer_area._load_func
            loader_label, loader_text = _load_source_label_and_text(load_func)
            kwargs_text = "(unavailable)"
            if load_func is not None:
                kwargs_text = (
                    erlab.interactive.utils.format_kwargs(
                        typing.cast(
                            "dict[typing.Hashable, typing.Any]",
                            load_func[1],
                        )
                    )
                    if load_func[1]
                    else "(none)"
                )
            return _LoadSourceDetails(
                path=file_path,
                loader_label=loader_label,
                loader_text=loader_text,
                kwargs_text=kwargs_text,
                load_code=_load_code_from_file_details(file_path, load_func),
            )
        return None

    @property
    def metadata_fields(self) -> list[_MetadataField]:
        if self.archived and self._metadata_fields_archived:
            return list(self._metadata_fields_archived)

        tool_window = self.tool_window
        kind_value = "ImageTool"
        if not self.is_imagetool:
            assert tool_window is not None
            kind_value = tool_window.tool_name

        fields = [
            _MetadataField(
                "Kind",
                kind_value,
            ),
            _MetadataField(
                "Added",
                self._created_time.isoformat(sep=" ", timespec="seconds"),
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
        return _preview_from_imagetool(
            self.imagetool, self._box_ratio_archived, self._pixmap_archived
        )

    @property
    def source_spec(
        self,
    ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
        if self.tool_window is not None:
            return self.tool_window.source_spec
        return self._source_spec

    @property
    def provenance_spec(
        self,
    ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
        if self.tool_window is not None:
            return self.tool_window.source_spec
        if self._provenance_spec is not None:
            return self._provenance_spec
        return self._source_spec

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
        return self._source_spec is not None

    @property
    def derivation_entries(
        self,
    ) -> list[erlab.interactive.imagetool.provenance.DerivationEntry]:
        if self.provenance_spec is None:
            return []
        return self.provenance_spec.derivation_entries()

    @property
    def derivation_lines(self) -> list[str]:
        return [entry.label for entry in self.derivation_entries]

    @property
    def derivation_code(self) -> str | None:
        if self.provenance_spec is None:
            return None
        return self.provenance_spec.derivation_code()

    def add_child_reference(self, uid: str, window: QtWidgets.QWidget) -> None:
        if uid not in self._childtool_indices:
            self._childtool_indices.append(uid)
        self._childtools[uid] = window

    def remove_child_reference(self, uid: str) -> None:
        self._childtools.pop(uid, None)
        with contextlib.suppress(ValueError):
            self._childtool_indices.remove(uid)

    @QtCore.Slot()
    def touch_archive(self) -> None:
        if self._archived_fname is not None and os.path.exists(self._archived_fname):
            with open(self._archived_fname, "a"):
                os.utime(self._archived_fname)

    def set_source_binding(
        self,
        source_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None,
        *,
        auto_update: bool = False,
        state: _source_state_type = "fresh",
    ) -> None:
        if source_spec is not None and not isinstance(
            source_spec, erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        ):
            raise TypeError(
                "source_spec must be a ToolProvenanceSpec or None. Use "
                "parse_tool_provenance_spec() when deserializing saved payloads."
            )
        self._source_spec = source_spec
        self._provenance_spec = source_spec
        self._source_auto_update = bool(auto_update)
        self._set_source_state(state if self._source_spec is not None else "fresh")

    def set_detached_provenance(
        self,
        provenance_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None,
    ) -> None:
        if provenance_spec is not None and not isinstance(
            provenance_spec, erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        ):
            raise TypeError(
                "provenance_spec must be a ToolProvenanceSpec or None. Use "
                "parse_tool_provenance_spec() when deserializing saved payloads."
            )
        self._source_spec = None
        self._provenance_spec = provenance_spec
        self._source_auto_update = False
        self._set_source_state("fresh")

    def _set_source_auto_update(self, value: bool) -> None:
        self._source_auto_update = bool(value)
        self.manager.tree_view.refresh(self.uid)
        self.manager._update_info(uid=self.uid)

    def _set_source_state(self, state: _source_state_type) -> None:
        self._source_state = state
        self.manager.tree_view.refresh(self.uid)
        self.manager._update_info(uid=self.uid)

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
        if (
            obj == self.imagetool
            and event is not None
            and event.type()
            in (
                QtCore.QEvent.Type.Show,
                QtCore.QEvent.Type.Hide,
                QtCore.QEvent.Type.WindowStateChange,
            )
        ):
            if event.type() == QtCore.QEvent.Type.Show:
                erlab.interactive.utils.single_shot(
                    self.slicer_area, 0, self.slicer_area._update_if_delayed
                )
            erlab.interactive.utils.single_shot(self, 0, self.visibility_changed)
        return super().eventFilter(obj, event)

    @QtCore.Slot()
    @QtCore.Slot(str)
    def update_title(self, title: str | None = None) -> None:
        if self.imagetool is not None:
            if title is None:
                title = self.imagetool.windowTitle()
            self.name = title

    @QtCore.Slot()
    def visibility_changed(self) -> None:
        window = self.window
        if isinstance(window, QtWidgets.QWidget):
            self._recent_geometry = window.geometry()

    @QtCore.Slot()
    def show(self) -> None:
        if self.archived:
            self.unarchive()

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
    def archive(self) -> None:
        if not self.is_imagetool or self.archived:
            return
        self._archived_fname = os.path.join(self.manager.cache_dir, f"{self.uid}.nc")
        imagetool = typing.cast("ImageTool", self.imagetool)
        imagetool.to_file(self._archived_fname)
        self.touch_timer.start()

        self._info_text_archived = self.info_text
        self._metadata_fields_archived = self.metadata_fields
        self._box_ratio_archived, self._pixmap_archived = self._preview_image
        self.dispose()
        self.manager._mark_descendants_source_unavailable(self.uid)

    @QtCore.Slot()
    def unarchive(self) -> None:
        if not self.is_imagetool or not self.archived:
            return
        self.touch_timer.stop()
        self.window = ImageTool.from_file(
            typing.cast("str", self._archived_fname), _in_manager=True
        )
        typing.cast("ImageTool", self.imagetool).show()
        self._info_text_archived = ""
        self._metadata_fields_archived = []
        self._box_ratio_archived = float("NaN")
        self._pixmap_archived = QtGui.QPixmap()
        self.manager._sigReloadLinkers.emit()
        self.manager._propagate_source_change_from_uid(self.uid)

    @QtCore.Slot()
    def reload(self) -> None:
        if self.imagetool is not None:
            self.slicer_area.reload()

    @QtCore.Slot()
    def _refresh_node_info(self) -> None:
        self.manager._update_info(uid=self.uid)
        self.manager.tree_view.refresh(self.uid)

    def _update_from_parent_source(self) -> bool:
        if self.tool_window is not None:
            updated = self.tool_window._update_from_parent_source()
            if updated and self.tool_window.source_state == "fresh":
                self.manager._propagate_source_change_from_uid(self.uid)
            self.manager.tree_view.refresh(self.uid)
            return updated

        if self._source_spec is None:
            return False
        try:
            parent_data = self.parent_source_data()
            resolved = self._source_spec.apply(parent_data)
            self.slicer_area.replace_source_data(resolved)
        except Exception:
            self._set_source_state("unavailable")
            return False

        self._set_source_state("fresh")
        return True

    def handle_parent_source_replaced(self, parent_data: xr.DataArray) -> None:
        if self.tool_window is not None:
            self.tool_window.handle_parent_source_replaced(parent_data)
            if self.tool_window.source_state == "fresh":
                self.manager._propagate_source_change_from_uid(self.uid)
            self.manager.tree_view.refresh(self.uid)
            return

        if self._source_spec is None:
            return
        try:
            resolved = self._source_spec.apply(parent_data)
        except Exception:
            self._set_source_state("unavailable")
            return

        if self._source_auto_update:
            try:
                self.slicer_area.replace_source_data(resolved)
            except Exception:
                self._set_source_state("unavailable")
                return
            self._set_source_state("fresh")
        else:
            self._set_source_state("stale")

    def show_source_update_dialog(
        self, *, parent: QtWidgets.QWidget | None = None
    ) -> int:
        if self.tool_window is not None:
            return self.tool_window.show_source_update_dialog(parent=parent)

        if self._source_spec is None or self._source_state == "fresh":
            return int(QtWidgets.QDialog.DialogCode.Rejected)

        dialog = erlab.interactive.utils._ToolSourceUpdateDialog(
            parent if parent is not None else self.manager,
            state=self._source_state,
            auto_update=self._source_auto_update,
        )
        result = dialog.exec()
        if result == int(QtWidgets.QDialog.DialogCode.Accepted):
            self._set_source_auto_update(dialog.auto_update_check.isChecked())
            if self._source_state == "stale":
                self._update_from_parent_source()
        return result

    @QtCore.Slot(object)
    def _handle_source_data_replaced(self, parent_data: object) -> None:
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
        *,
        provenance_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_spec: erlab.interactive.imagetool.provenance.ToolProvenanceSpec
        | None = None,
        source_auto_update: bool = False,
        source_state: _ManagedWindowNode._source_state_type = "fresh",
    ) -> None:
        self._index = index
        self._watched_varname: str | None = None
        self._watched_uid: str | None = None
        if watched_var is not None:
            self._watched_varname, self._watched_uid = watched_var

        super().__init__(
            manager,
            uid,
            None,
            tool,
            provenance_spec=provenance_spec,
            source_spec=source_spec,
            source_auto_update=source_auto_update,
            source_state=source_state,
        )

    @property
    def index(self) -> int:
        return self._index

    @property
    def watched(self) -> bool:
        return self._watched_varname is not None and self._watched_uid is not None

    @property
    def label_text(self) -> str:
        title = f"{self.index}"
        if self._name:
            title += f": {self._name}"
        return title

    @QtCore.Slot()
    def unwatch(self) -> None:
        if self.watched:
            self.manager._sigWatchedDataEdited.emit(
                self._watched_varname, self._watched_uid, "removed"
            )
            self._watched_varname = None
            self._watched_uid = None
            self.manager.tree_view.refresh(self.index)

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
