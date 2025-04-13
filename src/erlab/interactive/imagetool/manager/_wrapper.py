"""Wrapper for ImageTool windows shown in ImageToolManager."""

from __future__ import annotations

__all__ = ["_ImageToolWrapper"]

import datetime
import os
import typing
import uuid
import weakref

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool._mainwindow import ImageTool

if typing.TYPE_CHECKING:
    from erlab.interactive.imagetool.core import ImageSlicerArea
    from erlab.interactive.imagetool.manager import ImageToolManager


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

        self.touch_timer = QtCore.QTimer(self)
        self.touch_timer.setInterval(12 * 60 * 60 * 1000)  # 12 hours
        self.touch_timer.timeout.connect(self.touch_archive)

    @QtCore.Slot()
    def touch_archive(self) -> None:
        """Touch the archived file to update the modified time.

        This is required to keep the archived file from being deleted by the system. For
        instance on macOS, the system deletes files in the cache directory if they are
        not accessed for a long time.
        """
        if self._archived_fname is not None and os.path.exists(self._archived_fname):
            with open(self._archived_fname, "a"):
                os.utime(self._archived_fname)

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
            text = erlab.utils.formatting.format_darr_html(
                self.slicer_area._data,
                show_size=True,
                additional_info=[
                    f"Added {self._created_time.isoformat(sep=' ', timespec='seconds')}"
                ],
            )

        if hasattr(QtGui.QPalette.ColorRole, "Accent"):
            # Accent color is available from Qt 6.6
            accent_color = QtWidgets.QApplication.palette().accent().color().name()
            text = text.replace(
                erlab.utils.formatting._DEFAULT_ACCENT_COLOR, accent_color
            )

        return text

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
        typing.cast("ImageTool", self.tool).setWindowTitle(self.label_text)
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
                title = typing.cast("ImageTool", self.tool).windowTitle()
            self.name = title

    @QtCore.Slot()
    def visibility_changed(self) -> None:
        tool = typing.cast("ImageTool", self.tool)
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
            tool = typing.cast("ImageTool", self.tool)
            tool.to_file(self._archived_fname)
            self.touch_timer.start()

            self._info_text_archived = self.info_text
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
            self.touch_timer.stop()
            self.tool = ImageTool.from_file(typing.cast("str", self._archived_fname))
            self.tool.show()
            self._info_text_archived = ""
            self._box_ratio_archived = float("NaN")
            self._pixmap_archived = QtGui.QPixmap()
            self.manager._sigReloadLinkers.emit()
