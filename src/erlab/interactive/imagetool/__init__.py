"""Interactive data browser.

.. currentmodule:: erlab.interactive.imagetool

Modules
=======

.. autosummary::
   :toctree:

   core
   slicer
   fastbinning
   controls

"""

from __future__ import annotations

__all__ = ["ImageTool", "itool"]

import gc
import pickle
import sys
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab.io
from erlab.interactive.imagetool.controls import (
    ItoolBinningControls,
    ItoolColormapControls,
    ItoolCrosshairControls,
)
from erlab.interactive.imagetool.core import ImageSlicerArea, SlicerLinkProxy
from erlab.interactive.utils import DictMenuBar, copy_to_clipboard

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from erlab.interactive.imagetool.core import ImageSlicerState
    from erlab.interactive.imagetool.slicer import ArraySlicer


def _parse_input(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset,
) -> list[xr.DataArray]:
    if isinstance(data, xr.Dataset):
        data = [d for d in data.data_vars.values() if d.ndim >= 2 and d.ndim <= 4]
        if len(data) == 0:
            raise ValueError("No valid data for ImageTool found in the Dataset")

    if isinstance(data, np.ndarray | xr.DataArray):
        data = (data,)

    return [xr.DataArray(d) if not isinstance(d, xr.DataArray) else d for d in data]


def itool(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset,
    link: bool = False,
    link_colors: bool = True,
    use_manager: bool = True,
    execute: bool | None = None,
    **kwargs,
) -> ImageTool | list[ImageTool] | None:
    """Create and display ImageTool windows.

    Parameters
    ----------
    data
        Array-like object or a sequence of such object with 2 to 4 dimensions. See
        notes.
    link
        Whether to enable linking between multiple ImageTool windows, by default
        `False`.
    link_colors
        Whether to link the color maps between multiple linked ImageTool windows, by
        default `True`.
    use_manager
        Whether to open the ImageTool windows using the ImageToolManager if it is
        running, by default `True`.
    execute
        Whether to execute the Qt event loop and display the window, by default `None`.
        If `None`, the execution is determined based on the current IPython shell. This
        argument has no effect if the ImageToolManager is running and `use_manager` is
        set to `True`.
    **kwargs
        Additional keyword arguments to be passed onto the underlying slicer area. For a
        full list of supported arguments, see the
        `erlab.interactive.imagetool.core.ImageSlicerArea` documentation.

    Returns
    -------
    ImageTool or list of ImageTool
        The created ImageTool window(s).

    Notes
    -----
    - If `data` is a sequence of valid data, multiple ImageTool windows will be created
      and displayed.
    - If `data` is a Dataset, each DataArray in the Dataset will be displayed in a
      separate ImageTool window. Data variables with 2 to 4 dimensions are considered as
      valid. Other variables are ignored.
    - If `link` is True, the ImageTool windows will be synchronized.

    Examples
    --------
    >>> itool(data, cmap="gray", gamma=0.5)
    >>> itool(data_list, link=True)
    """
    if use_manager:
        from erlab.interactive.imagetool.manager import is_running

        if not is_running():
            use_manager = False

    if use_manager:
        from erlab.interactive.imagetool.manager import show_in_manager

        return show_in_manager(data, link=link, link_colors=link_colors, **kwargs)

    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    if isinstance(qapp, QtWidgets.QApplication):
        qapp.setStyle("Fusion")

    itool_list = [ImageTool(d, **kwargs) for d in _parse_input(data)]

    for w in itool_list:
        w.show()

    if len(itool_list) == 0:
        raise ValueError("No data provided")

    if link:
        linker = SlicerLinkProxy(  # noqa: F841
            *[w.slicer_area for w in itool_list], link_colors=link_colors
        )

    itool_list[-1].activateWindow()
    itool_list[-1].raise_()

    if execute is None:
        execute = True
        try:
            shell = get_ipython().__class__.__name__  # type: ignore
            if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
                execute = False
                from IPython.lib.guisupport import start_event_loop_qt4

                start_event_loop_qt4(qapp)
        except NameError:
            pass

    if execute:
        if isinstance(qapp, QtWidgets.QApplication):
            qapp.exec()

        del itool_list
        gc.collect()

        return None

    if len(itool_list) == 1:
        return itool_list[0]

    return itool_list


class BaseImageTool(QtWidgets.QMainWindow):
    def __init__(self, data=None, parent=None, **kwargs):
        super().__init__(parent=parent)
        self.slicer_area = ImageSlicerArea(self, data, **kwargs)
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
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    def to_pickle(self, filename: str):
        info: tuple[xr.DataArray, ImageSlicerState] = (
            self.slicer_area.data,
            self.slicer_area.state,
        )
        with open(filename, "wb") as file:
            pickle.dump(info, file)

    @classmethod
    def from_pickle(cls, filename: str):
        with open(filename, "rb") as file:
            data, state = pickle.load(file)
        return cls(data, state=state)

    def _sync_dock_float(self, floating: bool, index: int):
        for i in range(len(self.docks)):
            if i != index:
                self.docks[i].blockSignals(True)
                self.docks[i].setFloating(floating)
                self.docks[i].blockSignals(False)
        self.docks[index].blockSignals(True)
        self.docks[index].setFloating(floating)
        self.docks[index].blockSignals(False)

    # !TODO: this is ugly and temporary, fix it
    def widget_box(self, widget: QtWidgets.QWidget, **kwargs):
        group = QtWidgets.QGroupBox(**kwargs)
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred
        )
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 3)
        group_layout.setSpacing(3)
        group_layout.addWidget(widget)
        return group


class ImageTool(BaseImageTool):
    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)
        self.mnb = ItoolMenuBar(self.slicer_area, self)

        self.slicer_area.sigDataChanged.connect(self.update_title)
        self.update_title()

    def update_title(self):
        if self.slicer_area._data is not None:
            if self.slicer_area._data.name:
                self.setWindowTitle(str(self.slicer_area._data.name))


class ItoolMenuBar(DictMenuBar):
    def __init__(
        self, slicer_area: ImageSlicerArea, parent: QtWidgets.QWidget | None
    ) -> None:
        super().__init__(parent)
        self.slicer_area = slicer_area

        self.createMenus()
        self.refreshMenus()
        self.refreshEditMenus()
        self.slicer_area.sigViewOptionChanged.connect(self.refreshMenus)
        self.slicer_area.sigHistoryChanged.connect(self.refreshEditMenus)

        self._recent_name_filter: str | None = None

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
        menu_kwargs: dict[str, Any] = {
            "fileMenu": {
                "title": "&File",
                "actions": {
                    "&Open...": {
                        "shortcut": QtGui.QKeySequence.StandardKey.Open,
                        "triggered": self._open_file,
                    },
                    "&Save As...": {
                        "shortcut": QtGui.QKeySequence.StandardKey.SaveAs,
                        "triggered": self._export_file,
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
                },
            },
            "helpMenu": {
                "title": "&Help",
                "actions": {
                    "helpAction": {"text": "DataSlicer Help (WIP)"},
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
            "triggered": lambda: self.slicer_area.center_cursor(),
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
            "triggered": lambda: self.slicer_area.center_all_cursors(),
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

    def createMenus(self):
        menu_kwargs = self._generate_menu_kwargs()
        self.add_items(**menu_kwargs)

        # Disable/Enable menus based on context
        self.menu_dict["viewMenu"].aboutToShow.connect(
            lambda: self.action_dict["remCursorAct"].setDisabled(
                self.slicer_area.n_cursors == 1
            )
        )

    @QtCore.Slot()
    def refreshMenus(self):
        self.action_dict["snapCursorAct"].blockSignals(True)
        self.action_dict["snapCursorAct"].setChecked(self.array_slicer.snap_to_data)
        self.action_dict["snapCursorAct"].blockSignals(False)

        cmap_props = self.slicer_area.colormap_properties
        for ca, k in zip(
            self.colorAct, ["reversed", "high_contrast", "zero_centered"], strict=True
        ):
            k = cast(
                Literal["reversed", "high_contrast", "zero_centered"], k
            )  # for mypy
            ca.blockSignals(True)
            ca.setChecked(cmap_props[k])
            ca.blockSignals(False)

    @QtCore.Slot()
    def refreshEditMenus(self):
        self.action_dict["undoAct"].setEnabled(self.slicer_area.undoable)
        self.action_dict["redoAct"].setEnabled(self.slicer_area.redoable)

    def _set_colormap_options(self):
        self.slicer_area.set_colormap(
            reversed=self.colorAct[0].isChecked(),
            high_contrast=self.colorAct[1].isChecked(),
            zero_centered=self.colorAct[2].isChecked(),
        )

    def _copy_cursor_val(self):
        copy_to_clipboard(str(self.slicer_area.array_slicer._values))

    def _copy_cursor_idx(self):
        copy_to_clipboard(str(self.slicer_area.array_slicer._indices))

    @QtCore.Slot()
    def _open_file(self, *, name_filter: str | None = None):
        valid_loaders: dict[str, tuple[Callable, dict]] = {
            "xarray HDF5 Files (*.h5)": (erlab.io.load_hdf5, {}),
            "NetCDF Files (*.nc *.nc4 *.cdf)": (xr.load_dataarray, {}),
        }
        for k in erlab.io.loaders.keys():
            valid_loaders = valid_loaders | erlab.io.loaders[k].file_dialog_methods

        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(valid_loaders.keys())

        if name_filter is None:
            name_filter = self._recent_name_filter

        if name_filter is not None:
            dialog.selectNameFilter(name_filter)
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            files = dialog.selectedFiles()
            self._recent_name_filter = dialog.selectedNameFilter()
            fn, kargs = valid_loaders[self._recent_name_filter]
            # !TODO: handle ambiguous datasets
            self.slicer_area.set_data(fn(files[0], **kargs))
            self.slicer_area.view_all()

    def _export_file(self):
        if self.slicer_area._data is None:
            raise ValueError("Data is Empty!")
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)

        valid_savers: dict[str, tuple[Callable, dict[str, Any]]] = {
            "xarray HDF5 Files (*.h5)": (erlab.io.save_as_hdf5, {}),
            "NetCDF Files (*.nc *.nc4 *.cdf)": (erlab.io.save_as_netcdf, {}),
        }
        dialog.setNameFilters(valid_savers.keys())
        dialog.setDirectory(f"{self.slicer_area.data.name}.h5")
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if dialog.exec():
            files = dialog.selectedFiles()
            fn, kargs = valid_savers[dialog.selectedNameFilter()]
            fn(self.slicer_area._data, files[0], **kargs)
