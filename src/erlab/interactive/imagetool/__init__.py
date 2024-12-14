"""Interactive data browser.

.. currentmodule:: erlab.interactive.imagetool

Modules
=======

.. autosummary::
   :toctree:

   core
   manager
   slicer
   fastbinning
   controls
   dialogs

"""

from __future__ import annotations

__all__ = ["BaseImageTool", "ImageTool", "itool"]

import json
import os
import sys
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.imagetool.controls import (
    ItoolBinningControls,
    ItoolColormapControls,
    ItoolCrosshairControls,
)
from erlab.interactive.imagetool.core import ImageSlicerArea, SlicerLinkProxy
from erlab.interactive.imagetool.dialogs import (
    CropDialog,
    NormalizeDialog,
    RotationDialog,
)
from erlab.interactive.utils import (
    DictMenuBar,
    copy_to_clipboard,
    file_loaders,
    wait_dialog,
)
from erlab.utils.misc import _convert_to_native

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Collection

    from erlab.interactive.imagetool.slicer import ArraySlicer

_ITOOL_DATA_NAME: str = "<erlab-itool-data>"
#: Name to use for the data variable in cached datasets


def _parse_input(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset,
) -> list[xr.DataArray]:
    if isinstance(data, np.ndarray | xr.DataArray):
        data = (data,)
    elif isinstance(data, xr.Dataset):
        data = tuple(d for d in data.data_vars.values() if d.ndim >= 2 and d.ndim <= 4)
        if len(data) == 0:
            raise ValueError("No valid data for ImageTool found in the Dataset")

    return [xr.DataArray(d) if not isinstance(d, xr.DataArray) else d for d in data]


def itool(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset,
    *,
    link: bool = False,
    link_colors: bool = True,
    use_manager: bool = True,
    execute: bool | None = None,
    **kwargs,
) -> ImageTool | list[ImageTool] | None:
    """Create and display ImageTool windows.

    Parameters
    ----------
    data : DataArray, Dataset, ndarray, list of DataArray or list of ndarray
        The data to be displayed. Data can be provided as:

        - A `xarray.DataArray` with 2 to 4 dimensions

          The DataArray will be displayed in an ImageTool window.

        - A numpy array with 2 to 4 dimensions

          The array will be converted to a DataArray and displayed in an ImageTool.

        - A list of the above objects

          Multiple ImageTool windows will be created and displayed.

        - A `xarray.Dataset`

          Every DataArray in the Dataset will be displayed across multiple ImageTool
          windows. Data variables that have less than 2 dimensions or more than 4
          dimensions are ignored. Dimensions with length 1 are automatically squeezed.
    link
        Whether to enable linking between multiple ImageTool windows when `data` is a
        sequence or a `xarray.Dataset`, by default `False`.
    link_colors
        Whether to link the color maps between multiple linked ImageTool windows, by
        default `True`. This argument has no effect if `link` is set to `False`.
    use_manager
        Whether to open the ImageTool window(s) using the :class:`ImageToolManager
        <erlab.interactive.imagetool.manager.ImageToolManager>` if it is running, by
        default `True`.
    execute
        Whether to execute the Qt event loop and display the window, by default `None`.
        If `None`, the execution is determined based on the current IPython shell. This
        argument has no effect if the :class:`ImageToolManager
        <erlab.interactive.imagetool.manager.ImageToolManager>` is running and
        `use_manager` is set to `True`. In most cases, the default value should be used.
    **kwargs
        Additional keyword arguments to be passed onto the underlying slicer area. For a
        full list of supported arguments, see the
        `erlab.interactive.imagetool.core.ImageSlicerArea` documentation.

    Returns
    -------
    ImageTool or list of ImageTool or None
        The created ImageTool window(s).

        If the window(s) are executed, the function will return `None`, since the event
        loop will prevent the function from returning until the window(s) are closed.

        If the window(s) are not executed, for example while running in an IPython shell
        with ``%gui qt``, the function will not block and return the ImageTool window(s)
        or a list of ImageTool windows depending on the input data.

        The function will also return `None` if the windows are opened in the
        :class:`ImageToolManager
        <erlab.interactive.imagetool.manager.ImageToolManager>`.

    Examples
    --------
    >>> itool(data, cmap="gray", gamma=0.5)
    >>> itool([data1, data2], link=True)
    """
    if use_manager:
        from erlab.interactive.imagetool.manager import is_running

        if not is_running():
            use_manager = False

    if use_manager:
        from erlab.interactive.imagetool.manager import show_in_manager

        show_in_manager(data, link=link, link_colors=link_colors, **kwargs)
        return None

    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    if isinstance(qapp, QtWidgets.QApplication):
        qapp.setStyle("Fusion")

    itool_list = [ImageTool(d, **kwargs) for d in _parse_input(data)]

    for w in itool_list:
        w.show()

    if link:
        linker = SlicerLinkProxy(  # noqa: F841
            *[w.slicer_area for w in itool_list], link_colors=link_colors
        )

    itool_list[-1].activateWindow()
    itool_list[-1].raise_()

    if execute is None:
        execute = True
        try:
            shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
            if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
                execute = False
                from IPython.lib.guisupport import start_event_loop_qt4

                start_event_loop_qt4(qapp)
        except NameError:
            pass

    if execute:
        if isinstance(qapp, QtWidgets.QApplication):
            qapp.exec()

        return None

    if len(itool_list) == 1:
        return itool_list[0]

    return itool_list


class BaseImageTool(QtWidgets.QMainWindow):
    """Base class for an ImageTool window.

    This class combines the :class:`ImageSlicerArea
    <erlab.interactive.imagetool.core.ImageSlicerArea>` and the controls into a single
    window.

    Use this class only if you want to extend ImageTool without the menubar or keyboard
    shortcuts. Otherwise, use :class:`ImageTool
    <erlab.interactive.imagetool.ImageTool>`.

    Parameters
    ----------
    data
        The data to be displayed.
    parent
        The parent widget.
    **kwargs
        Additional keyword arguments to the underlying :class:`ImageSlicerArea
        <erlab.interactive.imagetool.core.ImageSlicerArea>`.

    """

    def __init__(
        self, data=None, parent: QtWidgets.QWidget | None = None, **kwargs
    ) -> None:
        super().__init__(parent=parent)
        self._slicer_area = ImageSlicerArea(self, data, **kwargs)
        self.setCentralWidget(self.slicer_area)

        self.docks: tuple[QtWidgets.QDockWidget, ...] = tuple(
            QtWidgets.QDockWidget(name, self) for name in ("Cursor", "Color", "Binning")
        )
        for i, d in enumerate(self.docks):
            d.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            d.topLevelChanged.connect(
                lambda val, *, idx=i: self._sync_dock_float(val, idx)
            )

        self.docks[0].setWidget(
            self.widget_box(
                ItoolCrosshairControls(
                    self.slicer_area, orientation=QtCore.Qt.Orientation.Vertical
                )
            )
        )
        self.docks[1].setWidget(
            self.widget_box(ItoolColormapControls(self.slicer_area))
        )
        self.docks[2].setWidget(self.widget_box(ItoolBinningControls(self.slicer_area)))

        for d in self.docks:
            self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, d)
        self.resize(720, 720)

    @property
    def slicer_area(self) -> ImageSlicerArea:
        """
        The underlying :class:`ImageSlicerArea
        <erlab.interactive.imagetool.core.ImageSlicerArea>`.
        """  # noqa: D205
        return self._slicer_area

    @property
    def array_slicer(self) -> ArraySlicer:
        """
        The underlying :class:`ArraySlicer
        <erlab.interactive.imagetool.slicer.ArraySlicer>`.
        """  # noqa: D205
        return self.slicer_area.array_slicer

    def to_dataset(self) -> xr.Dataset:
        name = self.slicer_area._data.name
        if name is None:
            name = ""
        return self.slicer_area._data.to_dataset(
            name=_ITOOL_DATA_NAME, promote_attrs=False
        ).assign_attrs(
            {
                "itool_state": json.dumps(self.slicer_area.state),
                "itool_title": self.windowTitle(),
                "itool_name": name,
                "itool_rect": self.geometry().getRect(),
            }
        )

    def to_file(self, filename: str | os.PathLike) -> None:
        """Save the data, state, title, and geometry of the tool to a file.

        The saved pickle file can be used to recreate the ImageTool with the class
        method :meth:`from_pickle`.

        Parameters
        ----------
        filename
            The name of the pickle file.

        """
        self.to_dataset().to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, **kwargs) -> Self:
        """Restore a window from a dataset saved using :meth:`to_dataset`.

        Parameters
        ----------
        ds
            The dataset.
        **kwargs
            Additional keyword arguments passed to the constructor.

        """
        name = ds.attrs["itool_name"]
        name = None if name == "" else name
        tool = cls(
            ds[_ITOOL_DATA_NAME].rename(name),
            state=json.loads(ds.attrs["itool_state"]),
            **kwargs,
        )
        tool.setWindowTitle(ds.attrs["itool_title"])
        tool.setGeometry(*ds.attrs["itool_rect"])
        return tool

    @classmethod
    def from_file(cls, filename: str | os.PathLike, **kwargs) -> Self:
        """Restore a window from a file saved using :meth:`to_file`.

        Parameters
        ----------
        filename
            The name of the file.
        **kwargs
            Additional keyword arguments passed to the constructor.
        """
        return cls.from_dataset(xr.load_dataset(filename, engine="h5netcdf"), **kwargs)

    def _sync_dock_float(self, floating: bool, index: int) -> None:
        """Synchronize the floating state of the dock widgets.

        Parameters
        ----------
        floating
            The floating state.
        index
            The index of the dock widget.

        """
        for i in range(len(self.docks)):
            if i != index:
                self.docks[i].blockSignals(True)
                self.docks[i].setFloating(floating)
                self.docks[i].blockSignals(False)
        self.docks[index].blockSignals(True)
        self.docks[index].setFloating(floating)
        self.docks[index].blockSignals(False)

    # !TODO: this is ugly and temporary, fix it
    def widget_box(self, widget: QtWidgets.QWidget, **kwargs) -> QtWidgets.QGroupBox:
        """Create a box that surrounds the given widget.

        Parameters
        ----------
        widget
            The widget to be added to the box.
        **kwargs
            Additional keyword arguments passed to :class:`QtWidgets.QGroupBox`

        Returns
        -------
        group
            The created widget box.

        """
        group = QtWidgets.QGroupBox(**kwargs)
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred
        )
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 3)
        group_layout.setSpacing(3)
        group_layout.addWidget(widget)
        return group

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        # For some reason, the select all and copy shortcuts don't work in vbox menu
        # widgets. The events start at the ItoolGraphicsLayoutWidget and is never passed
        # to menu widgets so we need to intercept them at a higher level.
        if event is not None and event.type() == QtCore.QEvent.Type.ShortcutOverride:
            event = cast(QtGui.QKeyEvent, event)
            focused = QtWidgets.QApplication.focusWidget()
            if isinstance(
                focused,
                QtWidgets.QAbstractSpinBox | QtWidgets.QLineEdit,
            ) and (
                event.matches(QtGui.QKeySequence.StandardKey.SelectAll)
                or event.matches(QtGui.QKeySequence.StandardKey.Copy)
            ):
                QtWidgets.QApplication.sendEvent(focused, event)
                return True

        return super().eventFilter(obj, event)

    def closeEvent(self, evt: QtGui.QCloseEvent | None) -> None:
        self.slicer_area.close_associated_windows()
        super().closeEvent(evt)


class ImageTool(BaseImageTool):
    """The ImageTool window class.

    This class adds the menubar with keyboard shortcuts to :class:`BaseImageTool
    <erlab.interactive.imagetool.BaseImageTool>`. Use this class to create an ImageTool
    window.

    Parameters
    ----------
    data
        The data to be displayed.
    **kwargs
        Additional keyword arguments to :class:`BaseImageTool
        <erlab.interactive.imagetool.BaseImageTool>`.

    Signals
    -------
    sigTitleChanged(str)
        Emitted when the title of the window is changed by setting a new data or file.
        It is *not* emitted when the title is changed by other means such as
        :meth:`setWindowTitle`.

    """

    sigTitleChanged = QtCore.Signal(str)  #: :meta private:

    def __init__(self, data=None, **kwargs) -> None:
        super().__init__(data, **kwargs)
        self._recent_name_filter: str | None = None
        self._recent_directory: str | None = None
        self.setMenuBar(ItoolMenuBar(self))

        self.slicer_area.sigDataChanged.connect(self._update_title)
        self._update_title()
        self.slicer_area.installEventFilter(self)

    @property
    def mnb(self) -> ItoolMenuBar:
        return cast(ItoolMenuBar, self.menuBar())

    def _update_title(self) -> None:
        if self.slicer_area._data is not None:
            name: str | None = cast(str | None, self.slicer_area._data.name)
            path: pathlib.Path | None = self.slicer_area._file_path

            if name is not None and name.strip() == "":
                # Name contains only whitespace
                name = None

            if name is None:
                title = "" if path is None else path.stem
            elif path is None or name == path.stem:
                title = f"{name}"
            else:
                title = f"{name} ({path.stem})"
            self.setWindowTitle(title)
            self.sigTitleChanged.emit(title)

    @QtCore.Slot()
    def _open_file(
        self,
        *,
        native: bool = True,
    ) -> None:
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        valid_loaders: dict[str, tuple[Callable, dict]] = file_loaders()
        dialog.setNameFilters(valid_loaders.keys())
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if self._recent_name_filter is not None:
            dialog.selectNameFilter(self._recent_name_filter)

        if self._recent_directory is not None:
            dialog.setDirectory(self._recent_directory)

        if dialog.exec():
            fname = dialog.selectedFiles()[0]
            self._recent_name_filter = dialog.selectedNameFilter()
            self._recent_directory = os.path.dirname(fname)
            fn, kargs = valid_loaders[self._recent_name_filter]

            try:
                with wait_dialog(self, "Loading..."):
                    self.slicer_area.set_data(fn(fname, **kargs), file_path=fname)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"An error occurred while loading the file: {e}"
                    "\n\nTry again with a different loader.",
                    QtWidgets.QMessageBox.StandardButton.Ok,
                )
                self._open_file()
            else:
                self.slicer_area.view_all()

    @QtCore.Slot()
    def _export_file(self, *, native: bool = True) -> None:
        if self.slicer_area._data is None:
            raise ValueError("Data is Empty!")
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        if not native:
            dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        # To avoid importing erlab.io, we define the following functions here
        def _add_igor_scaling(darr: xr.DataArray) -> xr.DataArray:
            scaling = [[1, 0]]
            for i in range(darr.ndim):
                coord: npt.NDArray = np.asarray(darr[darr.dims[i]].values)
                delta = coord[1] - coord[0]
                scaling.append([delta, coord[0]])
            if darr.ndim == 4:
                scaling[0] = scaling.pop(-1)
            darr.attrs["IGORWaveScaling"] = scaling
            return darr

        def _to_netcdf(darr: xr.DataArray, file: str, **kwargs) -> None:
            darr.to_netcdf(file, **kwargs)

        def _to_hdf5(darr: xr.DataArray, file: str, **kwargs) -> None:
            _to_netcdf(_add_igor_scaling(darr), file, **kwargs)

        valid_savers: dict[str, tuple[Callable, dict[str, Any]]] = {
            "xarray HDF5 Files (*.h5)": (
                _to_hdf5,
                {"engine": "h5netcdf", "invalid_netcdf": True},
            ),
            "NetCDF Files (*.nc *.nc4 *.cdf)": (_to_netcdf, {}),
        }

        dialog.setNameFilters(valid_savers.keys())
        dialog.setDirectory(f"{self.slicer_area._data.name}.h5")
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if dialog.exec():
            files = dialog.selectedFiles()
            fn, kargs = valid_savers[dialog.selectedNameFilter()]
            with wait_dialog(self, "Saving..."):
                fn(self.slicer_area._data, files[0], **kargs)


class ItoolMenuBar(DictMenuBar):
    def __init__(self, tool: ImageTool) -> None:
        super().__init__(tool)
        self.createMenus()
        self.refreshMenus()
        self.refreshEditMenus()
        self.slicer_area.sigViewOptionChanged.connect(self.refreshMenus)
        self.slicer_area.sigHistoryChanged.connect(self.refreshEditMenus)

    @property
    def image_tool(self) -> ImageTool:
        return cast(ImageTool, self.parent())

    @property
    def slicer_area(self) -> ImageSlicerArea:
        return self.image_tool.slicer_area

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    @property
    def colorAct(self):
        return tuple(
            self.action_dict[k]
            for k in ("colorInvertAct", "highContrastAct", "zeroCenterAct")
        )

    def _generate_menu_kwargs(self) -> dict:
        _guideline_actions = self.slicer_area.main_image._guideline_actions
        menu_kwargs: dict[str, Any] = {
            "fileMenu": {
                "title": "&File",
                "actions": {
                    "&Open...": {
                        "shortcut": QtGui.QKeySequence.StandardKey.Open,
                        "triggered": self.image_tool._open_file,
                    },
                    "&Save As...": {
                        "shortcut": QtGui.QKeySequence.StandardKey.SaveAs,
                        "triggered": self.image_tool._export_file,
                    },
                    "&Close": {
                        "shortcut": QtGui.QKeySequence.StandardKey.Close,
                        "triggered": self.image_tool.close,
                        "sep_before": True,
                    },
                },
            },
            "viewMenu": {
                "title": "&View",
                "actions": {
                    "viewAllAct": {
                        "text": "View &All",
                        "shortcut": "Ctrl+A",
                        "triggered": self.slicer_area.view_all,
                    },
                    "transposeAct": {
                        "text": "&Transpose Main Image",
                        "shortcut": "T",
                        "triggered": lambda: self.slicer_area.swap_axes(0, 1),
                        "sep_after": True,
                    },
                    "addCursorAct": {
                        "text": "&Add New Cursor",
                        "shortcut": "Shift+A",
                        "triggered": self.slicer_area.add_cursor,
                    },
                    "remCursorAct": {
                        "text": "&Remove Current Cursor",
                        "shortcut": "Shift+R",
                        "triggered": self.slicer_area.remove_current_cursor,
                    },
                    "snapCursorAct": {
                        "text": "&Snap to Pixels",
                        "shortcut": "S",
                        "triggered": self.slicer_area.toggle_snap,
                        "checkable": True,
                    },
                    "cursorMoveMenu": {
                        "title": "Cursor Control",
                        "actions": {},
                    },
                    "colorInvertAct": {
                        "text": "Invert",
                        "shortcut": "R",
                        "checkable": True,
                        "toggled": self._set_colormap_options,
                        "sep_before": True,
                    },
                    "highContrastAct": {
                        "text": "High Contrast",
                        "checkable": True,
                        "toggled": self._set_colormap_options,
                    },
                    "zeroCenterAct": {
                        "text": "Center At Zero",
                        "checkable": True,
                        "toggled": self._set_colormap_options,
                        "sep_after": True,
                    },
                    "ktoolAct": {
                        "text": "Open ktool",
                        "triggered": self.slicer_area.open_in_ktool,
                        "sep_after": True,
                    },
                    "Normalize": {"triggered": self._normalize},
                    "resetAct": {
                        "text": "Reset",
                        "triggered": self._reset_filters,
                        "sep_after": True,
                    },
                },
            },
            "editMenu": {
                "title": "&Edit",
                "actions": {
                    "undoAct": {
                        "text": "Undo",
                        "shortcut": QtGui.QKeySequence.StandardKey.Undo,
                        "triggered": self.slicer_area.undo,
                    },
                    "redoAct": {
                        "text": "Redo",
                        "shortcut": QtGui.QKeySequence.StandardKey.Redo,
                        "triggered": self.slicer_area.redo,
                        "sep_after": True,
                    },
                    "&Copy Cursor Values": {
                        "shortcut": "Ctrl+C",
                        "triggered": self._copy_cursor_val,
                    },
                    "&Copy Cursor Indices": {
                        "shortcut": "Ctrl+Alt+C",
                        "triggered": self._copy_cursor_idx,
                        "sep_after": True,
                    },
                    "Rotate": {"triggered": self._rotate},
                    "Rotation Guidelines": {
                        "actions": {
                            f"guide{i}": act for i, act in enumerate(_guideline_actions)
                        },
                        "sep_after": True,
                    },
                    "Crop": {"triggered": self._crop},
                },
            },
            "helpMenu": {
                "title": "&Help",
                "actions": {
                    "helpAction": {"text": "Help (WIP)"},
                    "shortcutsAction": {
                        "text": "Keyboard Shortcuts Reference (WIP)",
                        "sep_before": True,
                    },
                },
            },
        }

        menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
            "centerCursorAct"
        ] = {
            "text": "&Center Current Cursor",
            "shortcut": "Shift+C",
            "triggered": self.slicer_area.center_cursor,
        }
        for i, ((t, s), axis, amount) in enumerate(
            zip(
                (
                    ("Shift Current Cursor Up", "Shift+Up"),
                    ("Shift Current Cursor Down", "Shift+Down"),
                    ("Shift Current Cursor Right", "Shift+Right"),
                    ("Shift Current Cursor Left", "Shift+Left"),
                    ("Shift Current Cursor Up × 10", "Ctrl+Shift+Up"),
                    ("Shift Current Cursor Down × 10", "Ctrl+Shift+Down"),
                    ("Shift Current Cursor Right × 10", "Ctrl+Shift+Right"),
                    ("Shift Current Cursor Left × 10", "Ctrl+Shift+Left"),
                ),
                (1, 1, 0, 0) * 2,
                (1, -1, 1, -1, 10, -10, 10, -10),
                strict=True,
            )
        ):
            menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
                f"ShiftCursorAct{i}"
            ] = {
                "text": t,
                "shortcut": s,
                "triggered": lambda *,
                ax=self.slicer_area.main_image.display_axis[axis],
                d=amount: self.slicer_area.step_index(ax, d),
            }
        menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
            "centerAllCursorsAct"
        ] = {
            "text": "&Center All Cursors",
            "shortcut": "Alt+Shift+C",
            "triggered": self.slicer_area.center_all_cursors,
            "sep_before": True,
        }
        for i, ((t, s), axis, amount) in enumerate(
            zip(
                (
                    ("Shift Cursors Up", "Alt+Shift+Up"),
                    ("Shift Cursors Down", "Alt+Shift+Down"),
                    ("Shift Cursors Right", "Alt+Shift+Right"),
                    ("Shift Cursors Left", "Alt+Shift+Left"),
                    ("Shift Cursors Up × 10", "Ctrl+Alt+Shift+Up"),
                    ("Shift Cursors Down × 10", "Ctrl+Alt+Shift+Down"),
                    ("Shift Cursors Right × 10", "Ctrl+Alt+Shift+Right"),
                    ("Shift Cursors Left × 10", "Ctrl+Alt+Shift+Left"),
                ),
                (1, 1, 0, 0) * 2,
                (1, -1, 1, -1, 10, -10, 10, -10),
                strict=True,
            )
        ):
            menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
                f"ShiftAllCursorAct{i}"
            ] = {
                "text": t,
                "shortcut": s,
                "triggered": lambda *,
                ax=self.slicer_area.main_image.display_axis[axis],
                d=amount: self.slicer_area.step_index_all(ax, d),
            }
        return menu_kwargs

    def createMenus(self) -> None:
        menu_kwargs = self._generate_menu_kwargs()
        self.add_items(**menu_kwargs)

        # Disable/Enable menus based on context
        self.menu_dict["viewMenu"].aboutToShow.connect(self._view_menu_visibility)

    @QtCore.Slot()
    def _view_menu_visibility(self) -> None:
        self.action_dict["remCursorAct"].setDisabled(self.slicer_area.n_cursors == 1)
        self.action_dict["ktoolAct"].setEnabled(
            self.slicer_area.data.kspace._interactive_compatible
        )
        self.action_dict["resetAct"].setEnabled(
            self.slicer_area._applied_func is not None
        )

    @QtCore.Slot()
    def refreshMenus(self) -> None:
        self.action_dict["snapCursorAct"].blockSignals(True)
        self.action_dict["snapCursorAct"].setChecked(self.array_slicer.snap_to_data)
        self.action_dict["snapCursorAct"].blockSignals(False)

        cmap_props = self.slicer_area.colormap_properties
        for ca, k in zip(
            self.colorAct, ["reverse", "high_contrast", "zero_centered"], strict=True
        ):
            k = cast(
                Literal["reverse", "high_contrast", "zero_centered"], k
            )  # for mypy
            ca.blockSignals(True)
            ca.setChecked(cmap_props[k])
            ca.blockSignals(False)

    @QtCore.Slot()
    def refreshEditMenus(self) -> None:
        self.action_dict["undoAct"].setEnabled(self.slicer_area.undoable)
        self.action_dict["redoAct"].setEnabled(self.slicer_area.redoable)

    def execute_dialog(self, dialog_cls: type[QtWidgets.QDialog]) -> None:
        dialog = dialog_cls(self.slicer_area)
        dialog.exec()

    @QtCore.Slot()
    def _rotate(self) -> None:
        self.execute_dialog(RotationDialog)

    @QtCore.Slot()
    def _crop(self) -> None:
        self.execute_dialog(CropDialog)

    @QtCore.Slot()
    def _normalize(self) -> None:
        self.execute_dialog(NormalizeDialog)

    @QtCore.Slot()
    def _reset_filters(self) -> None:
        self.slicer_area.apply_func(None)

    @QtCore.Slot()
    def _set_colormap_options(self) -> None:
        self.slicer_area.set_colormap(
            reverse=self.colorAct[0].isChecked(),
            high_contrast=self.colorAct[1].isChecked(),
            zero_centered=self.colorAct[2].isChecked(),
        )

    @QtCore.Slot()
    def _copy_cursor_val(self) -> None:
        copy_to_clipboard(
            str(_convert_to_native(self.slicer_area.array_slicer._values))
        )

    @QtCore.Slot()
    def _copy_cursor_idx(self) -> None:
        copy_to_clipboard(
            str(_convert_to_native(self.slicer_area.array_slicer._indices))
        )
