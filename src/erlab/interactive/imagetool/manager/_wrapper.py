"""Wrapper for ImageTool windows shown in ImageToolManager."""

from __future__ import annotations

__all__ = ["_ImageToolWrapper"]

import datetime
import os
import sys
import typing
import uuid
import weakref

from qtpy import QtCore, QtGui

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

    def __init__(
        self,
        manager: ImageToolManager,
        index: int,
        tool: ImageTool,
        watched_var: tuple[str, str] | None = None,
    ) -> None:
        super().__init__(manager)
        self._manager = weakref.ref(manager)
        self._index: int = index
        self._imagetool: ImageTool | None = None
        self._recent_geometry: QtCore.QRect | None = None
        self._name: str = tool.windowTitle()
        self._archived_fname: str | None = None
        self._created_time: datetime.datetime = datetime.datetime.now()

        self._childtools: dict[str, erlab.interactive.utils.ToolWindow] = {}
        self._childtool_indices: list[str] = []

        # Information about the watched variable
        self._watched_varname: str | None = None
        self._watched_uid: str | None = None
        if watched_var is not None:
            self._watched_varname, self._watched_uid = watched_var

        self._info_text_archived: str = ""

        self._box_ratio_archived: float = float("NaN")
        self._pixmap_archived: QtGui.QPixmap = QtGui.QPixmap()

        self.imagetool = tool

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
        manager = self._manager()
        if manager:
            return manager
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
        return erlab.interactive.utils._apply_qt_accent_color(text)

    @property
    def _preview_image(self) -> tuple[float, QtGui.QPixmap]:
        """Get the preview image and box aspect ratio.

        Retrieves the main image pixmap and flips it to match the image displayed in the
        tool. The box ratio is calculated from the view box size of the main image.
        """
        if self.imagetool is not None:
            main_image = self.slicer_area.main_image
            vb_rect = main_image.getViewBox().rect()

            pixmap: QtGui.QPixmap = (
                main_image.slicer_data_items[0]
                .getPixmap()
                .transformed(QtGui.QTransform().scale(1.0, -1.0))
            )
            box_ratio: float = vb_rect.height() / vb_rect.width()

            return box_ratio, pixmap
        return self._box_ratio_archived, self._pixmap_archived

    @property
    def imagetool(self) -> ImageTool | None:
        return self._imagetool

    @imagetool.setter
    def imagetool(self, value: ImageTool | None) -> None:
        if self._imagetool is None:
            if self._archived_fname is not None:
                # Remove the archived file
                os.remove(self._archived_fname)
                self._archived_fname = None
        else:
            # Close and cleanup existing tool
            self._imagetool.slicer_area.unlink()
            self._imagetool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self._imagetool.removeEventFilter(self)
            self._imagetool.sigTitleChanged.disconnect(self.update_title)
            self._imagetool.slicer_area.sigDataEdited.disconnect(
                self._trigger_watched_update
            )
            self._imagetool.destroyed.connect(self._destroyed_callback)
            self._imagetool.close()

        if value is not None:
            # Install event filter to detect visibility changes
            value.installEventFilter(self)
            value.sigTitleChanged.connect(self.update_title)
            value.slicer_area.sigDataEdited.connect(self._trigger_watched_update)
            value.slicer_area._in_manager = True

        self._imagetool = value

    @property
    def slicer_area(self) -> ImageSlicerArea:
        if self.imagetool is None:
            raise ValueError("ImageTool is not available")
        return self.imagetool.slicer_area

    @property
    def archived(self) -> bool:
        return self._imagetool is None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name
        typing.cast("ImageTool", self.imagetool).setWindowTitle(self.label_text)
        self.manager.tree_view.refresh(self.index)

    @property
    def watched(self) -> bool:
        """Whether the tool is synchronized to a variable in an IPython kernel."""
        return self._watched_varname is not None and self._watched_uid is not None

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
        """Event filter to detect visibility changes of the tool window.

        Stores the geometry of the tool window when it is shown or hidden so that it can
        be restored when the tool is shown again.
        """
        if (
            obj == self.imagetool
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
                title = typing.cast("ImageTool", self.imagetool).windowTitle()
            self.name = title

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
        """Trigger an update for a watched variable in the manager.

        This function notifies the listening IPython kernel to fetch the latest data for
        the specified watched variable.

        Parameters
        ----------
        varname
            Name of the watched variable.
        uid
            Unique identifier for the watched variable.
        """
        if self.watched:
            self.manager._sigWatchedDataEdited.emit(
                self._watched_varname, self._watched_uid, "updated"
            )

    @QtCore.Slot()
    def visibility_changed(self) -> None:
        tool = typing.cast("ImageTool", self.imagetool)
        self._recent_geometry = tool.geometry()

    @QtCore.Slot()
    def show(self) -> None:
        """Show the tool window.

        If the tool is not visible, it is shown and raised to the top. Archived tools
        are unarchived before being shown.
        """
        if self.imagetool is None:
            self.unarchive()

        if self.imagetool is not None:
            if not self.imagetool.isVisible() and self._recent_geometry is not None:
                self.imagetool.setGeometry(self._recent_geometry)

            if sys.platform == "win32":  # pragma: no cover
                # On Windows, window flags must be set to bring the window to the top
                self.imagetool.setWindowFlags(
                    self.imagetool.windowFlags()
                    | QtCore.Qt.WindowType.WindowStaysOnTopHint
                )
                self.imagetool.show()
                self.imagetool.setWindowFlags(
                    self.imagetool.windowFlags()
                    & ~QtCore.Qt.WindowType.WindowStaysOnTopHint
                )
            self.imagetool.show()
            self.imagetool.show()
            self.imagetool.activateWindow()
            self.imagetool.raise_()

    @QtCore.Slot()
    def hide(self) -> None:
        """Hide the tool window.

        This method only hides the tool window. The tool object is not destroyed and can
        be reopened later.
        """
        if self.imagetool is not None:
            self.imagetool.hide()

    @QtCore.Slot()
    def dispose(self, unwatch: bool = True) -> None:
        """Dispose the tool object.

        This method closes the tool window and destroys the tool object. The tool object
        is not recoverable after this operation.

        Parameters
        ----------
        unwatch
            If `True`, the watched variable is unwatched before disposing the tool.
            Default is `True`.
        """
        if unwatch and self.watched:
            self.unwatch()
        self.imagetool = None

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
            tool = typing.cast("ImageTool", self.imagetool)
            tool.to_file(self._archived_fname)
            self.touch_timer.start()

            self._info_text_archived = self.info_text
            self._box_ratio_archived, self._pixmap_archived = self._preview_image
            self.dispose(unwatch=False)

    @QtCore.Slot()
    def unarchive(self) -> None:
        """
        Restore the ImageTool from the archive.

        Instead of calling this directly, use
        :meth:`ImageToolManager.unarchive_selected` which displays a wait dialog.
        """
        if self.archived:
            self.touch_timer.stop()
            self.imagetool = ImageTool.from_file(
                typing.cast("str", self._archived_fname), _in_manager=True
            )
            self.imagetool.show()
            self._info_text_archived = ""
            self._box_ratio_archived = float("NaN")
            self._pixmap_archived = QtGui.QPixmap()
            self.manager._sigReloadLinkers.emit()

    def _add_childtool(
        self, tool: erlab.interactive.utils.ToolWindow, *, show: bool = True
    ) -> str:
        """Add a child tool window to the current tool."""
        uid = str(uuid.uuid4())
        self._childtools[uid] = tool
        if not tool._tool_display_name:
            tool._tool_display_name = str(self.name)

        tool.sigInfoChanged.connect(lambda u=uid: self.manager._update_info(uid=u))

        if show:
            tool.show()
        return uid

    def _remove_childtool(self, uid: str) -> None:
        """Remove a child tool window from the current tool."""
        if uid in self._childtools:
            tool = self._childtools.pop(uid)
            tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            tool.close()
