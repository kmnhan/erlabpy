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
import sys
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from qtpy import QtCore, QtWidgets

import erlab.io
from erlab.interactive.imagetool.controls import (
    ItoolBinningControls,
    ItoolColormapControls,
    ItoolCrosshairControls,
)
from erlab.interactive.imagetool.core import ImageSlicerArea, SlicerLinkProxy
from erlab.interactive.utilities import DictMenuBar, copy_to_clipboard
from erlab.io.plugins.merlin import BL403Loader

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from erlab.interactive.imagetool.slicer import ArraySlicer


def itool(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset,
    link: bool = False,
    link_colors: bool = True,
    execute: bool | None = None,
    **kwargs,
) -> ImageTool | list[ImageTool] | None:
    """Create and display an ImageTool window.

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
    execute
        Whether to execute the Qt event loop and display the window, by default `None`.
        If `None`, the execution is determined based on the current IPython shell.
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
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    if isinstance(qapp, QtWidgets.QApplication):
        qapp.setStyle("Fusion")

    if isinstance(data, xr.Dataset):
        data = [d for d in data.data_vars.values() if d.ndim >= 2 and d.ndim <= 4]
        if len(data) == 0:
            raise ValueError("No valid data for ImageTool found in the Dataset")

    if isinstance(data, np.ndarray | xr.DataArray):
        data = (data,)

    itool_list = [ImageTool(d, **kwargs) for d in data]

    for w in itool_list:
        w.show()

    if len(itool_list) == 0:
        raise ValueError("No data provided")

    itool_list[-1].activateWindow()
    itool_list[-1].raise_()

    if link:
        linker = SlicerLinkProxy(  # noqa: F841
            *[w.slicer_area for w in itool_list], link_colors=link_colors
        )

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
        qapp.exec()
        del itool_list
        gc.collect()

        return None
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
        self.slicer_area.sigViewOptionChanged.connect(self.refreshMenus)

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
                        "shortcut": "Ctrl+O",
                        "triggered": self._open_file,
                    },
                    "&Save As...": {
                        "shortcut": "Ctrl+Shift+S",
                        "triggered": self._export_file,
                    },
                    "&Copy Cursor Values": {
                        "shortcut": "Ctrl+C",
                        "triggered": self._copy_cursor_val,
                    },
                    "&Copy Cursor Indices": {
                        "shortcut": "Ctrl+Alt+C",
                        "triggered": self._copy_cursor_idx,
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

        self.menu_dict["viewMenu"].aboutToShow.connect(
            lambda: self.action_dict["remCursorAct"].setDisabled(
                self.slicer_area.n_cursors == 1
            )
        )

    def refreshMenus(self):
        self.action_dict["snapCursorAct"].blockSignals(True)
        self.action_dict["snapCursorAct"].setChecked(self.array_slicer.snap_to_data)
        self.action_dict["snapCursorAct"].blockSignals(False)

        cmap_props = self.slicer_area.colormap_properties
        for ca, k in zip(
            self.colorAct, ["reversed", "highContrast", "zeroCentered"], strict=True
        ):
            ca.blockSignals(True)
            ca.setChecked(
                cmap_props[cast(Literal["reversed", "highContrast", "zeroCentered"], k)]
            )
            ca.blockSignals(False)

    def _set_colormap_options(self):
        self.slicer_area.set_colormap(
            reversed=self.colorAct[0].isChecked(),
            highContrast=self.colorAct[1].isChecked(),
            zeroCentered=self.colorAct[2].isChecked(),
        )

    def _copy_cursor_val(self):
        copy_to_clipboard(str(self.slicer_area.array_slicer._values))

    def _copy_cursor_idx(self):
        copy_to_clipboard(str(self.slicer_area.array_slicer._indices))

    def _open_file(self):
        merlin_loader = cast(BL403Loader, erlab.io.loaders["merlin"])
        valid_loaders: dict[str, tuple[Callable, dict]] = {
            "xarray HDF5 Files (*.h5)": (erlab.io.load_hdf5, {}),
            "ALS BL4.0.3 Raw Data (*.pxt)": (merlin_loader.load, {}),
            "ALS BL4.0.3 Live (*.ibw)": (merlin_loader.load_live, {}),
            "DA30 Raw Data (*.ibw *.pxt *.zip)": (erlab.io.loaders["da30"].load, {}),
            "SSRL BL5-2 Raw Data (*.h5)": (erlab.io.loaders["ssrl52"].load, {}),
            "NetCDF Files (*.nc *.nc4 *.cdf)": (xr.load_dataarray, {}),
        }

        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(valid_loaders.keys())
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            files = dialog.selectedFiles()
            fn, kargs = valid_loaders[dialog.selectedNameFilter()]
            # !TODO: handle ambiguous datasets
            self.slicer_area.set_data(fn(files[0], **kargs))
            self.slicer_area.view_all()

    def _export_file(self):
        if self.slicer_area._data is None:
            raise ValueError("Data is Empty!")
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)

        valid_savers: dict[str, tuple[Callable, dict]] = {
            "xarray HDF5 Files (*.h5)": (erlab.io.save_as_hdf5, {}),
            "NetCDF Files (*.nc *.nc4 *.cdf)": (erlab.io.save_as_netcdf, {}),
        }
        dialog.setNameFilters(valid_savers.keys())
        dialog.setDirectory(f"{self.slicer_area._data.name}.h5")
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if dialog.exec():
            files = dialog.selectedFiles()
            fn, kargs = valid_savers[dialog.selectedNameFilter()]
            fn(self.slicer_area._data, files[0], **kargs)


if __name__ == "__main__":
    # import gc
    # import linecache
    # import tracemalloc

    # def display_top(snapshot: tracemalloc.Snapshot, limit=10):
    #     snapshot = snapshot.filter_traces(
    #         (
    #             tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
    #             tracemalloc.Filter(False, "<unknown>"),
    #         )
    #     )
    #     top_stats = snapshot.statistics("traceback")

    #     print(f"\nTop {limit} lines")
    #     for index, stat in enumerate(top_stats[:limit], 1):
    #         frame = stat.traceback[0]
    #         print(
    #             f"#{index} {stat.traceback.total_nframe}: "
    #             f"{frame.filename}:{frame.lineno}: {stat.size/1024:.1f} KiB"
    #         )
    #         line = linecache.getline(frame.filename, frame.lineno).strip()
    #         if line:
    #             print("    %s" % line)

    #     other = top_stats[limit:]
    #     if other:
    #         size = sum(stat.size for stat in other)
    #         print(f"{len(other)} other: {size/1024:.1f} KiB")
    #     total = sum(stat.size for stat in top_stats)
    #     print(f"Total allocated size: {total/1024:.1f} KiB")

    # while True:
    #     try:
    #         idx = int(input("Index: "))
    #         stat = top_stats[idx - 1]
    #     except (IndexError, ValueError):
    #         break
    #     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
    #     for line in stat.traceback.format():
    #         print(line)

    # tracemalloc.start()
    data = xr.load_dataarray(
        #     # "~/Documents/ERLab/TiSe2/kxy10.nc",
        #     # "~/Documents/ERLab/TiSe2/221213_SSRL_BL5-2/fullmap_kconv_.h5",
        "~/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc",
        # "~/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy.nc",
        # "~/Documents/ERLab/TiSe2/220410_ALS_BL4/map_mm_4d_.nc",
        # engine="h5netcdf",
    )

    # win = itool(data, bench=True)
    # win = itool(data)
    # del data

    # gc.collect()
    # data = xr.load_dataarray(
    #     "~/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy.nc",
    #     engine="h5netcdf",
    # )

    # from erlab.io.exampledata import generate_data

    # data = generate_data()

    # win = itool([data, data], link=True, link_colors=False)
    # win = itool(data.sel(eV=0, method='nearest'))
    win = itool(data)

    # snapshot = tracemalloc.take_snapshot()
    # print(
    #     *[
    #         f"{n} {m * 2**-20:.2f} MB\t"
    #         for n, m in zip(("Current", "Max"), tracemalloc.get_traced_memory())
    #     ],
    #     sep="",
    # )
    # tracemalloc.stop()
    # display_top(snapshot)
    # print(win.array_slicer._nanmeancalls)

    # qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
    # if not qapp:
    #     qapp = QtWidgets.QApplication(sys.argv)
    # qapp.setStyle("Fusion")
    # import numpy as np
    # win = ImageTool(np.ones((2,2)))
    # win.show()
    # win.raise_()
    # win.activateWindow()
    # qapp.exec()
