"""Wrapper for ImageTool windows shown in ImageToolManager."""

from __future__ import annotations

__all__ = ["_ImageToolWrapper"]

import datetime
import os
import uuid
import weakref
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
from qtpy import QtCore, QtGui, QtWidgets
from xarray.core.formatting import render_human_readable_nbytes

import erlab
from erlab.interactive.imagetool._mainwindow import ImageTool

if TYPE_CHECKING:
    from collections.abc import Hashable

    import xarray as xr

    from erlab.interactive.imagetool.core import ImageSlicerArea
    from erlab.interactive.imagetool.manager import ImageToolManager


_ACCENT_PLACEHOLDER: str = "<info-accent-color>"
"""Placeholder for accent color in HTML strings."""


def _format_dim_name(s: Hashable) -> str:
    return f"<b>{s}</b>"


def _format_dim_sizes(darr: xr.DataArray, prefix: str) -> str:
    out = f"<p>{prefix}("

    dims_list = []
    for d in darr.dims:
        dim_label = _format_dim_name(d) if d in darr.coords else str(d)
        dims_list.append(f"{dim_label}: {darr.sizes[d]}")

    out += ", ".join(dims_list)
    out += r")</p>"
    return out


def _format_coord_dims(coord: xr.DataArray) -> str:
    dims = tuple(str(d) for d in coord.variable.dims)

    if len(dims) > 1:
        return f"({', '.join(dims)})&emsp;"

    if len(dims) == 1 and dims[0] != coord.name:
        return f"({dims[0]})&emsp;"

    return ""


def _format_array_values(val: npt.NDArray) -> str:
    if val.size == 1:
        return erlab.utils.formatting.format_value(val.item())

    val = val.squeeze()

    if val.ndim == 1:
        if len(val) == 2:
            return (
                f"[{erlab.utils.formatting.format_value(val[0])}, "
                f"{erlab.utils.formatting.format_value(val[1])}]"
            )

        if erlab.utils.array.is_uniform_spaced(val):
            if val[0] == val[-1]:
                return erlab.utils.formatting.format_value(val[0])

            start, end, step = tuple(
                erlab.utils.formatting.format_value(v)
                for v in (val[0], val[-1], val[1] - val[0])
            )
            return f"{start} : {step} : {end}"

        if erlab.utils.array.is_monotonic(val):
            if val[0] == val[-1]:
                return erlab.utils.formatting.format_value(val[0])

            return (
                f"{erlab.utils.formatting.format_value(val[0])} to "
                f"{erlab.utils.formatting.format_value(val[-1])}"
            )

    mn, mx = tuple(
        erlab.utils.formatting.format_value(v) for v in (np.nanmin(val), np.nanmax(val))
    )
    return f"min {mn} max {mx}"


def _format_coord_key(key: Hashable, is_dim: bool) -> str:
    style = f"color: {_ACCENT_PLACEHOLDER}; "
    if is_dim:
        style += "font-weight: bold; "
    return f"<span style='{style}'>{key}</span>&emsp;"


def _format_attr_key(key: Hashable) -> str:
    style = f"color: {_ACCENT_PLACEHOLDER};"
    return f"<span style='{style}'>{key}</span>&emsp;"


def _format_info_html(darr: xr.DataArray, created_time: datetime.datetime) -> str:
    out = ""

    name = ""
    if darr.name is not None and darr.name != "":
        name = f"'{darr.name}'&emsp;"

    out += _format_dim_sizes(darr, name)
    out += rf"<p>Size {render_human_readable_nbytes(darr.nbytes)}</p>"
    out += rf"<p>Added {created_time.isoformat(sep=' ', timespec='seconds')}</p>"

    out += r"Coordinates:"
    coord_rows: list[list[str]] = []
    for key, coord in darr.coords.items():
        is_dim: bool = key in darr.dims
        coord_rows.append(
            [
                _format_coord_key(key, is_dim),
                _format_coord_dims(coord),
                _format_array_values(coord.values),
            ]
        )
    out += erlab.utils.formatting.format_html_table(coord_rows)

    out += r"<br>Attributes:"
    attr_rows: list[list[str]] = []
    for key, attr in darr.attrs.items():
        attr_rows.append(
            [_format_attr_key(key), erlab.utils.formatting.format_value(attr)]
        )
    out += erlab.utils.formatting.format_html_table(attr_rows)

    return out


class _ImageToolWrapper(QtCore.QObject):
    """Wrapper for ImageTool objects.

    This class wraps an ImageTool object and provides additional functionality in the
    manager such as archiving and unarchiving and window geometry tracking.
    """

    def __init__(self, manager: ImageToolManager, index: int, tool: ImageTool) -> None:
        super().__init__(manager)
        self._manager = weakref.ref(manager)
        self._index: int = index
        self._tool: ImageTool | None = None
        self._recent_geometry: QtCore.QRect | None = None
        self._name: str = tool.windowTitle()
        self._archived_fname: str | None = None
        self._created_time: datetime.datetime = datetime.datetime.now()

        self._info_text_archived: str = ""

        self._box_ratio_archived: float = float("NaN")
        self._pixmap_archived: QtGui.QPixmap = QtGui.QPixmap()

        self.tool = tool

    @property
    def index(self) -> int:
        """Index of the ImageTool in the manager.

        This index is unique for each ImageTool and is used to identify the tool in the
        manager.
        """
        return self._index

    @property
    def manager(self) -> ImageToolManager:
        _manager = self._manager()
        if _manager:
            return _manager
        raise LookupError("Parent was destroyed")

    @property
    def info_text(self) -> str:
        if self.archived:
            text: str = self._info_text_archived
        else:
            text = _format_info_html(self.slicer_area._data, self._created_time)

        accent_color = "#0078d7"
        if hasattr(QtGui.QPalette.ColorRole, "Accent"):
            # Accent color is available from Qt 6.6
            accent_color = QtWidgets.QApplication.palette().accent().color().name()

        return text.replace(_ACCENT_PLACEHOLDER, accent_color)

    @property
    def _preview_image(self) -> tuple[float, QtGui.QPixmap]:
        """Get the preview image and box aspect ratio.

        Retrieves the main image pixmap and flips it to match the image displayed in the
        tool. The box ratio is calculated from the view box size of the main image.
        """
        if self.tool is not None:
            main_image = self.slicer_area.main_image
            vb_rect = main_image.getViewBox().rect()

            pixmap: QtGui.QPixmap = (
                self.slicer_area.main_image.slicer_data_items[0]
                .getPixmap()
                .transformed(QtGui.QTransform().scale(1.0, -1.0))
            )
            box_ratio: float = vb_rect.height() / vb_rect.width()

            return box_ratio, pixmap
        return self._box_ratio_archived, self._pixmap_archived

    @property
    def tool(self) -> ImageTool | None:
        return self._tool

    @tool.setter
    def tool(self, value: ImageTool | None) -> None:
        if self._tool is None:
            if self._archived_fname is not None:
                # Remove the archived file
                os.remove(self._archived_fname)
                self._archived_fname = None
        else:
            # Close and cleanup existing tool
            self._tool.slicer_area.unlink()
            self._tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self._tool.removeEventFilter(self)
            self._tool.sigTitleChanged.disconnect(self.update_title)
            self._tool.destroyed.connect(self._destroyed_callback)
            self._tool.close()

        if value is not None:
            # Install event filter to detect visibility changes
            value.installEventFilter(self)
            value.sigTitleChanged.connect(self.update_title)

        self._tool = value

    @property
    def slicer_area(self) -> ImageSlicerArea:
        if self.tool is None:
            raise ValueError("ImageTool is not available")
        return self.tool.slicer_area

    @property
    def archived(self) -> bool:
        return self._tool is None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name
        cast(ImageTool, self.tool).setWindowTitle(self.label_text)
        self.manager.list_view.refresh(self.index)

    @property
    def label_text(self) -> str:
        """Label text shown in the window title and the manager.

        The label text is a combination of the index and the name of the tool.
        """
        new_title = f"{self.index}"
        if self.name != "":
            new_title += f": {self.name}"
        return new_title

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if (
            obj == self.tool
            and event is not None
            and (
                event.type() == QtCore.QEvent.Type.Show
                or event.type() == QtCore.QEvent.Type.Hide
                or event.type() == QtCore.QEvent.Type.WindowStateChange
            )
        ):
            self.visibility_changed()
        return super().eventFilter(obj, event)

    def _destroyed_callback(self) -> None:
        self.manager._sigReloadLinkers.emit()

    @QtCore.Slot()
    @QtCore.Slot(str)
    def update_title(self, title: str | None = None) -> None:
        if not self.archived:
            if title is None:
                title = cast(ImageTool, self.tool).windowTitle()
            self.name = title

    @QtCore.Slot()
    def visibility_changed(self) -> None:
        tool = cast(ImageTool, self.tool)
        self._recent_geometry = tool.geometry()

    @QtCore.Slot()
    def show(self) -> None:
        """Show the tool window.

        If the tool is not visible, it is shown and raised to the top. Archived tools
        are unarchived before being shown.
        """
        if self.tool is None:
            self.unarchive()

        if self.tool is not None:
            if not self.tool.isVisible() and self._recent_geometry is not None:
                self.tool.setGeometry(self._recent_geometry)
            self.tool.show()
            self.tool.activateWindow()
            self.tool.raise_()

    @QtCore.Slot()
    def close(self) -> None:
        """Close the tool window.

        This method only closes the tool window. The tool object is not destroyed and
        can be reopened later.
        """
        if self.tool is not None:
            self.tool.close()

    @QtCore.Slot()
    def dispose(self) -> None:
        """Dispose the tool object.

        This method closes the tool window and destroys the tool object. The tool object
        is not recoverable after this operation.
        """
        self.tool = None

    @QtCore.Slot()
    def archive(self) -> None:
        """Archive the ImageTool.

        Unlike :meth:`dispose_tool`, this method saves the tool object to a file and can
        be recovered later. The archived tools are grayed out in the manager.

        Instead of calling this directly, use :meth:`ImageToolManager.archive_selected`
        which displays a wait dialog.
        """
        if not self.archived:
            self._archived_fname = os.path.join(
                self.manager.cache_dir, str(uuid.uuid4())
            )
            tool = cast(ImageTool, self.tool)
            tool.to_file(self._archived_fname)

            self._info_text_archived = _format_info_html(
                self.slicer_area._data, self._created_time
            )
            self._box_ratio_archived, self._pixmap_archived = self._preview_image
            self.dispose()

    @QtCore.Slot()
    def unarchive(self) -> None:
        """
        Restore the ImageTool from the archive.

        Instead of calling this directly, use
        :meth:`ImageToolManager.unarchive_selected` which displays a wait dialog.
        """
        if self.archived:
            self.tool = ImageTool.from_file(cast(str, self._archived_fname))
            self.tool.show()
            self._info_text_archived = ""
            self._box_ratio_archived = float("NaN")
            self._pixmap_archived = QtGui.QPixmap()
            self.manager._sigReloadLinkers.emit()
