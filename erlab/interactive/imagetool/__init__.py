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

__all__ = ["itool", "ImageTool"]

import sys
import warnings
from typing import Any

import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab.io
from erlab.interactive.imagetool.controls import (
    ItoolBinningControls,
    ItoolColormapControls,
    ItoolControlsBase,
    ItoolCrosshairControls,
)
from erlab.interactive.imagetool.core import ImageSlicerArea, SlicerLinkProxy
from erlab.interactive.imagetool.slicer import ArraySlicer
from erlab.interactive.utilities import copy_to_clipboard


def itool(data, *args, link: bool = False, execute: bool | None = None, **kwargs):
    qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")

    if isinstance(data, (list, tuple)):
        win = tuple()
        for d in data:
            win += (ImageTool(d, *args, **kwargs),)
        for w in win:
            w.show()
        win[-1].activateWindow()
        win[-1].raise_()

        if link:
            linker = SlicerLinkProxy(*[w.slicer_area for w in win])
    else:
        win = ImageTool(data, *args, **kwargs)
        win.show()
        win.activateWindow()
        # win.raise_()
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
    return win


class ImageTool(QtWidgets.QMainWindow):
    def __init__(self, data=None, **kwargs):
        super().__init__()
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

        self.mnb = ItoolMenuBar(self.slicer_area, self)

        self.slicer_area.sigDataChanged.connect(self.update_title)
        self.update_title()

    def _sync_dock_float(self, floating: bool, index: int):
        for i in range(len(self.docks)):
            if i != index:
                self.docks[i].blockSignals(True)
                self.docks[i].setFloating(floating)
                self.docks[i].blockSignals(False)
        self.docks[index].blockSignals(True)
        self.docks[index].setFloating(floating)
        self.docks[index].blockSignals(False)

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    def update_title(self):
        if self.slicer_area._data is not None:
            if self.slicer_area._data.name:
                self.setWindowTitle(str(self.slicer_area._data.name))

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


class DictMenuBar(QtWidgets.QMenuBar):
    def __init__(self, parent: QtWidgets.QWidget | None = ..., **kwargs) -> None:
        super().__init__(parent)

        self.menu_dict: dict[str, QtWidgets.QMenu] = dict()
        self.action_dict: dict[str, QtWidgets.QAction] = dict()

        self.add_items(**kwargs)

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            try:
                out = self.menu_dict[__name]
            except KeyError:
                out = self.action_dict[__name]
            warnings.warn(
                f"Menu or Action '{__name}' called as an attribute",
                PendingDeprecationWarning,
            )
            return out

    def add_items(self, **kwargs):
        self.parse_menu(self, **kwargs)

    def parse_menu(self, parent: QtWidgets.QMenuBar | QtWidgets.QMenu, **kwargs: dict):
        for name, opts in kwargs.items():
            menu = opts.pop("menu", None)
            actions = opts.pop("actions")

            if menu is None:
                title = opts.pop("title", None)
                icon = opts.pop("icon", None)
                if title is None:
                    title = name
                if icon is None:
                    menu = parent.addMenu(title)
                else:
                    menu = parent.addMenu(icon, title)
            else:
                menu = parent.addMenu(menu)

            self.menu_dict[name] = menu

            for actname, actopts in actions.items():
                if isinstance(actopts, QtWidgets.QAction):
                    act = actopts
                    sep_before, sep_after = False, False
                else:
                    if "actions" in actopts:
                        self.parse_menu(menu, **{actname: actopts})
                        continue
                    sep_before = actopts.pop("sep_before", False)
                    sep_after = actopts.pop("sep_after", False)
                    if "text" not in actopts:
                        actopts["text"] = actname
                    act = self.parse_action(actopts)
                if sep_before:
                    menu.addSeparator()
                menu.addAction(act)
                if (
                    act.text() is not None
                ):  # check whether it's a separator without text
                    self.action_dict[actname] = act
                if sep_after:
                    menu.addSeparator()

    @staticmethod
    def parse_action(actopts: dict):
        shortcut = actopts.pop("shortcut", None)
        triggered = actopts.pop("triggered", None)
        toggled = actopts.pop("toggled", None)
        changed = actopts.pop("changed", None)

        if shortcut is not None:
            actopts["shortcut"] = QtGui.QKeySequence(shortcut)

        action = QtGui.QAction(**actopts)

        if triggered is not None:
            action.triggered.connect(triggered)
        if toggled is not None:
            action.toggled.connect(toggled)
        if changed is not None:
            action.changed.connect(changed)
        return action


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

    def createMenus(self):
        menu_kwargs = dict(
            fileMenu=dict(
                title="&File",
                actions={
                    "&Open...": dict(
                        shortcut="Ctrl+O",
                        triggered=self._open_file,
                    ),
                    "&Save As...": dict(
                        shortcut="Ctrl+Shift+S",
                        triggered=self._export_file,
                    ),
                    "&Copy Cursor Values": dict(
                        shortcut="Ctrl+C",
                        triggered=self._copy_cursor_val,
                    ),
                    "&Copy Cursor Indices": dict(
                        shortcut="Ctrl+Alt+C",
                        triggered=self._copy_cursor_idx,
                    ),
                },
            ),
            viewMenu=dict(
                title="&View",
                actions=dict(
                    viewAllAct=dict(
                        text="View &All",
                        shortcut="Ctrl+A",
                        triggered=self.slicer_area.view_all,
                    ),
                    transposeAct=dict(
                        text="&Transpose Main Image",
                        shortcut="T",
                        triggered=lambda: self.slicer_area.swap_axes(0, 1),
                        sep_after=True,
                    ),
                    addCursorAct=dict(
                        text="&Add New Cursor",
                        shortcut="Shift+A",
                        triggered=self.slicer_area.add_cursor,
                    ),
                    remCursorAct=dict(
                        text="&Remove Current Cursor",
                        shortcut="Shift+R",
                        triggered=self.slicer_area.remove_current_cursor,
                    ),
                    snapCursorAct=dict(
                        text="&Snap to Pixels",
                        shortcut="S",
                        triggered=self.slicer_area.toggle_snap,
                        checkable=True,
                    ),
                    cursorMoveMenu=dict(
                        title="Cursor Control",
                        actions=dict(),
                    ),
                    colorInvertAct=dict(
                        text="Invert",
                        shortcut="R",
                        checkable=True,
                        toggled=self._set_colormap_options,
                        sep_before=True,
                    ),
                    highContrastAct=dict(
                        text="High Contrast",
                        checkable=True,
                        toggled=self._set_colormap_options,
                    ),
                    zeroCenterAct=dict(
                        text="Center At Zero",
                        checkable=True,
                        toggled=self._set_colormap_options,
                        sep_after=True,
                    ),
                ),
            ),
            helpMenu=dict(
                title="&Help",
                actions=dict(
                    helpAction=dict(text="DataSlicer Help (WIP)"),
                    shortcutsAction=dict(
                        text="Keyboard Shortcuts Reference (WIP)", sep_before=True
                    ),
                ),
            ),
        )

        menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
            "centerCursorAct"
        ] = dict(
            text="&Center Current Cursor",
            shortcut="Shift+C",
            triggered=lambda: self.slicer_area.center_cursor(),
        )
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
            )
        ):
            menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
                f"ShiftCursorAct{i}"
            ] = dict(
                text=t,
                shortcut=s,
                triggered=lambda *, ax=self.slicer_area.main_image.display_axis[
                    axis
                ], d=amount: self.slicer_area.step_index(ax, d),
            )
        menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
            "centerAllCursorsAct"
        ] = dict(
            text="&Center All Cursors",
            shortcut="Alt+Shift+C",
            triggered=lambda: self.slicer_area.center_all_cursors(),
            sep_before=True,
        )
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
            )
        ):
            menu_kwargs["viewMenu"]["actions"]["cursorMoveMenu"]["actions"][
                f"ShiftAllCursorAct{i}"
            ] = dict(
                text=t,
                shortcut=s,
                triggered=lambda *, ax=self.slicer_area.main_image.display_axis[
                    axis
                ], d=amount: self.slicer_area.step_index_all(ax, d),
            )

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
        for ca, k in zip(self.colorAct, ["reversed", "highContrast", "zeroCentered"]):
            ca.blockSignals(True)
            ca.setChecked(cmap_props[k])
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
        valid_files = {
            "xarray HDF5 Files (*.h5)": (xr.load_dataarray, dict(engine="h5netcdf")),
            "NetCDF Files (*.nc *.nc4 *.cdf)": (xr.load_dataarray, dict()),
            "SSRL BL5-2 Raw Data (*.h5)": (erlab.io.load_ssrl, dict()),
            "ALS BL4.0.3 Raw Data (*.pxt)": (erlab.io.load_als_bl4, dict()),
            "ALS BL4.0.3 Live (*.ibw)": (erlab.io.load_live, dict()),
        }

        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(valid_files.keys())
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            files = dialog.selectedFiles()
            fn, kargs = valid_files[dialog.selectedNameFilter()]

            dat = fn(files[0], **kargs)
            # !TODO: handle ambiguous datasets

            self.slicer_area.set_data(dat)
            self.slicer_area.view_all()

    def _export_file(self):
        if self.slicer_area._data is None:
            raise ValueError("Data is Empty!")
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter(
            "xarray HDF5 Files (*.h5)",
        )
        dialog.setDirectory(f"{self.slicer_area._data.name}.h5")
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        if dialog.exec():
            files = dialog.selectedFiles()
            erlab.io.save_as_hdf5(self.slicer_area._data, files[0])


if __name__ == "__main__":
    import linecache
    import tracemalloc

    def display_top(snapshot: tracemalloc.Snapshot, limit=10):
        snapshot = snapshot.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            )
        )
        top_stats = snapshot.statistics("traceback")

        print(f"\nTop {limit} lines")
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            print(
                f"#{index} {stat.traceback.total_nframe}: {frame.filename}:{frame.lineno}: {stat.size/1024:.1f} KiB"
            )
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print("    %s" % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print(f"{len(other)} other: {size/1024:.1f} KiB")
        total = sum(stat.size for stat in top_stats)
        print(f"Total allocated size: {total/1024:.1f} KiB")

        # while True:
        #     try:
        #         idx = int(input("Index: "))
        #         stat = top_stats[idx - 1]
        #     except (IndexError, ValueError):
        #         break
        #     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        #     for line in stat.traceback.format():
        #         print(line)

    # data = xr.load_dataarray(
    #     # "~/Documents/ERLab/TiSe2/kxy10.nc",
    #     # "~/Documents/ERLab/TiSe2/221213_SSRL_BL5-2/fullmap_kconv_.h5",
    #     "~/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc",
    #     # "~/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy.nc",
    #     # "~/Documents/ERLab/TiSe2/220410_ALS_BL4/map_mm_4d_.nc",
    #     engine="h5netcdf",
    # )

    # tracemalloc.start()
    # win = itool(data, bench=True)
    # snapshot = tracemalloc.take_snapshot()

    from erlab.interactive.exampledata import generate_data

    data = generate_data()

    # win = itool([data, data], link=True)
    win = itool(data, link=True)

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
