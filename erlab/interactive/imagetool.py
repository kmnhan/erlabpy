from __future__ import annotations

import sys
import weakref
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
import qtawesome as qta
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.graphicsItems.ViewBox import ViewBoxMenu
from pyqtgraph.GraphicsScene import mouseEvents

import erlab.io
from erlab.interactive.colors import (
    pg_colormap_names,
    pg_colormap_powernorm,
    pg_colormap_to_QPixmap,
)
from erlab.interactive.slicer import ArraySlicer
from erlab.interactive.utilities import (
    BetterAxisItem,
    BetterColorBarItem,
    BetterImageItem,
    BetterSpinBox,
    copy_to_clipboard,
)

__all__ = ["itool", "ImageTool", "ImageSlicerArea"]

suppressnanwarning = np.testing.suppress_warnings()
suppressnanwarning.filter(RuntimeWarning, r"All-NaN (slice|axis) encountered")


def _link_splitters(
    s0: QtWidgets.QSplitter, s1: QtWidgets.QSplitter, reverse: bool = False
):
    s0.blockSignals(True)
    s1.blockSignals(True)
    sizes = s0.sizes()
    total = sum(sizes)
    if reverse:
        sizes = list(reversed(sizes))
        sizes[0] = s1.sizes()[-1]
    else:
        sizes[0] = s1.sizes()[0]
    if all([x == 0 for x in sizes[1:]]) and sizes[0] != total:
        sizes[1:] = [1] * len(sizes[1:])
    try:
        factor = (total - sizes[0]) / sum(sizes[1:])
    except ZeroDivisionError:
        factor = 0.0
    for k in range(1, len(sizes)):
        sizes[k] = round(sizes[k] * factor)
    if reverse:
        sizes = list(reversed(sizes))
    s0.setSizes(sizes)
    s0.blockSignals(False)
    s1.blockSignals(False)


def _sync_splitters(s0: QtWidgets.QSplitter, s1: QtWidgets.QSplitter):
    s0.splitterMoved.connect(lambda: _link_splitters(s1, s0))
    s1.splitterMoved.connect(lambda: _link_splitters(s0, s1))


def itool(data, execute=None, *args, **kwargs):
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
        # self.setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setCentralWidget(self.slicer_area)

        dock = QtWidgets.QDockWidget("Controls", self)

        self.controls = QtWidgets.QWidget(self)
        self.controls.setLayout(QtWidgets.QHBoxLayout(self.controls))
        self.controls.layout().setContentsMargins(3, 3, 3, 3)

        self.controls.layout().setSpacing(3)
        self.groups: list[QtWidgets.QGroupBox] = []
        self.group_layouts: list[QtWidgets.QVBoxLayout] = []
        self.group_widgets: list[list[QtWidgets.QWidget]] = []

        self.add_group()  # title="Info")
        self.add_group()  # title="Color")
        self.add_group()  # title="Bin")

        self.add_widget(
            0,
            ItoolCrosshairControls(
                self.slicer_area, orientation=QtCore.Qt.Orientation.Vertical
            ),
        )
        self.add_widget(1, ItoolColormapControls(self.slicer_area))
        self.add_widget(2, ItoolBinningControls(self.slicer_area))

        dock.setWidget(self.controls)
        # dock.setTitleBarWidget(QtWidgets.QWidget())
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock)
        self.resize(720, 720)

        self._createMenuBar()
        self._refreshMenu()
        self.slicer_area.sigViewOptionChanged.connect(self._refreshMenu)
        self.slicer_area.sigDataChanged.connect(self.update_title)
        self.update_title()

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    def update_title(self):
        if self.slicer_area._data is not None:
            if self.slicer_area._data.name:
                self.setWindowTitle(str(self.slicer_area._data.name))

    def add_widget(self, idx: int, widget: QtWidgets.QWidget):
        self.group_layouts[idx].addWidget(widget)
        self.group_widgets[idx].append(widget)

    def add_group(self, **kwargs):
        group = QtWidgets.QGroupBox(**kwargs)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 3)
        self.controls.layout().addWidget(group)
        group_layout.setSpacing(3)
        self.groups.append(group)
        self.group_widgets.append([])
        self.group_layouts.append(group_layout)

    def _createMenuBar(self):
        self._menu_bar = QtWidgets.QMenuBar(self)
        # self._menu_bar.setNativeMenuBar(False)

        ### FILE MENU
        self._file_menu = QtWidgets.QMenu("&File", self)
        self._menu_bar.addMenu(self._file_menu)
        self._file_menu.addSeparator()

        ### i/o
        self._open_action = self._file_menu.addAction(
            "&Open...", QtGui.QKeySequence("Ctrl+O")
        )
        self._open_action.triggered.connect(self._open_file)
        self._export_action = self._file_menu.addAction(
            "&Save As...", QtGui.QKeySequence("Ctrl+Shift+S")
        )
        self._export_action.triggered.connect(self._export_file)
        self._copy_cursor_val_action = self._file_menu.addAction(
            "&Copy Cursor Values", QtGui.QKeySequence("Ctrl+C")
        )
        self._copy_cursor_val_action.triggered.connect(self._copy_cursor_val)
        self._copy_cursor_idx_action = self._file_menu.addAction(
            "&Copy Cursor Indices", QtGui.QKeySequence("Ctrl+Alt+C")
        )
        self._copy_cursor_idx_action.triggered.connect(self._copy_cursor_idx)

        ### VIEW MENU
        self._view_menu = QtWidgets.QMenu("&View", self)
        self._menu_bar.addMenu(self._view_menu)
        self._view_menu.addSeparator()

        ### misc. view options
        self._viewall_action = self._view_menu.addAction(
            "View &All", QtGui.QKeySequence("Ctrl+A")
        )
        self._viewall_action.triggered.connect(self.slicer_area.view_all)
        self._transpose_action = self._view_menu.addAction(
            "&Transpose Main Image", QtGui.QKeySequence("T")
        )
        self._transpose_action.triggered.connect(
            lambda: self.slicer_area.swap_axes(0, 1)
        )

        ### cursor options
        self._view_menu.addSeparator()
        self._add_action = self._view_menu.addAction(
            "&Add New Cursor", QtGui.QKeySequence("Shift+A")
        )
        self._add_action.triggered.connect(self.slicer_area.add_cursor)
        self._rem_action = self._view_menu.addAction(
            "&Remove Current Cursor", QtGui.QKeySequence("Shift+R")
        )
        self._view_menu.aboutToShow.connect(
            lambda: self._rem_action.setDisabled(self.slicer_area.n_cursors == 1)
        )
        self._rem_action.triggered.connect(self.slicer_area.remove_current_cursor)
        self._snap_action = self._view_menu.addAction(
            "&Snap to Pixels", QtGui.QKeySequence("S")
        )
        self._snap_action.setCheckable(True)
        self._snap_action.toggled.connect(self.slicer_area.toggle_snap)

        ## cursor movement
        # single cursor
        self._cursor_move_menu = self._view_menu.addMenu("Cursor Control")
        self._center_action = self._cursor_move_menu.addAction(
            "&Center Current Cursor", QtGui.QKeySequence("Shift+C")
        )
        self._center_action.triggered.connect(self.slicer_area.center_cursor)
        self._cursor_step_actions = (
            self._cursor_move_menu.addAction(
                "Shift Current Cursor Up", QtGui.QKeySequence("Shift+Up")
            ),
            self._cursor_move_menu.addAction(
                "Shift Current Cursor Down", QtGui.QKeySequence("Shift+Down")
            ),
            self._cursor_move_menu.addAction(
                "Shift Current Cursor Right", QtGui.QKeySequence("Shift+Right")
            ),
            self._cursor_move_menu.addAction(
                "Shift Current Cursor Left", QtGui.QKeySequence("Shift+Left")
            ),
            self._cursor_move_menu.addAction(
                "Shift Current Cursor Up × 10", QtGui.QKeySequence("Ctrl+Shift+Up")
            ),
            self._cursor_move_menu.addAction(
                "Shift Current Cursor Down × 10", QtGui.QKeySequence("Ctrl+Shift+Down")
            ),
            self._cursor_move_menu.addAction(
                "Shift Current Cursor Right × 10",
                QtGui.QKeySequence("Ctrl+Shift+Right"),
            ),
            self._cursor_move_menu.addAction(
                "Shift Current Cursor Left × 10", QtGui.QKeySequence("Ctrl+Shift+Left")
            ),
        )
        for action, i, d in zip(
            self._cursor_step_actions,
            (1, 1, 0, 0) * 2,
            (1, -1, 1, -1, 10, -10, 10, -10),
        ):
            ax = self.slicer_area.main_image.display_axis[i]
            action.triggered.connect(
                lambda *, ax=ax, d=d: self.slicer_area.step_index(ax, d)
            )

        # multiple cursors
        self._cursor_move_menu.addSeparator()
        self._center_all_action = self._cursor_move_menu.addAction(
            "&Center All Cursors", QtGui.QKeySequence("Alt+Shift+C")
        )
        self._center_all_action.triggered.connect(self.slicer_area.center_all_cursors)
        self._cursor_step_all_actions = (
            self._cursor_move_menu.addAction(
                "Shift Cursors Up", QtGui.QKeySequence("Alt+Shift+Up")
            ),
            self._cursor_move_menu.addAction(
                "Shift Cursors Down", QtGui.QKeySequence("Alt+Shift+Down")
            ),
            self._cursor_move_menu.addAction(
                "Shift Cursors Right", QtGui.QKeySequence("Alt+Shift+Right")
            ),
            self._cursor_move_menu.addAction(
                "Shift Cursors Left", QtGui.QKeySequence("Alt+Shift+Left")
            ),
        )
        for action, i, d in zip(
            self._cursor_step_all_actions, (1, 1, 0, 0), (1, -1, 1, -1)
        ):
            ax = self.slicer_area.main_image.display_axis[i]
            action.triggered.connect(
                lambda *, ax=ax, d=d: self.slicer_area.step_index_all(ax, d)
            )

        ### colormap options
        self._view_menu.addSeparator()
        self._color_actions = (
            self._view_menu.addAction("Invert", QtGui.QKeySequence("R")),
            self._view_menu.addAction("High Contrast"),
            self._view_menu.addAction("Center At Zero"),
        )
        for ca in self._color_actions:
            ca.setCheckable(True)
            ca.toggled.connect(self._set_colormap_options)
            # ca.setShortcutContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self._view_menu.addSeparator()

        ### HELP MENU
        self._help_menu = QtWidgets.QMenu("&Help", self)
        self._menu_bar.addMenu(self._help_menu)
        self._help_action = self._help_menu.addAction("DataSlicer Help (WIP)")
        self._help_menu.addSeparator()
        self._shortcut_action = self._help_menu.addAction(
            "Keyboard Shortcuts Reference (WIP)"
        )

    def _refreshMenu(self):
        self._snap_action.blockSignals(True)
        self._snap_action.setChecked(self.array_slicer.snap_to_data)
        self._snap_action.blockSignals(False)

        cmap_props = self.slicer_area.colormap_properties
        for ca, k in zip(
            self._color_actions, ["reversed", "highContrast", "zeroCentered"]
        ):
            ca.blockSignals(True)
            ca.setChecked(cmap_props[k])
            ca.blockSignals(False)

    def _set_colormap_options(self):
        self.slicer_area.set_colormap(
            reversed=self._color_actions[0].isChecked(),
            highContrast=self._color_actions[1].isChecked(),
            zeroCentered=self._color_actions[2].isChecked(),
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
            "ALS BL4.0.3 LiveXY (*.ibw)": (erlab.io.load_livexy, dict()),
            "ALS BL4.0.3 LivePolar (*.ibw)": (erlab.io.load_livepolar, dict()),
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

            for group in self.group_widgets:
                for w in group:
                    if isinstance(w, ItoolControlsBase):
                        w.slicer_area = self.slicer_area

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


# class ItoolGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
class ItoolGraphicsLayoutWidget(pg.PlotWidget):
    # Unsure of whether to subclass GraphicsLayoutWidget or PlotWidget at the moment
    # Will need to run some benchmarks in the future
    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        display_axis: tuple[int, ...],
        image: bool = False,
        **item_kw,
    ):
        # super().__init__()
        # self.ci.layout.setContentsMargins(0, 0, 0, 0)
        # self.ci.layout.setSpacing(3)
        # self.plotItem = ItoolPlotItem(slicer_area, display_axis, image, **item_kw)
        # self.addItem(self.plotItem)

        super().__init__(
            plotItem=ItoolPlotItem(slicer_area, display_axis, image, **item_kw),
        )
        self.scene().sigMouseClicked.connect(self.getPlotItem().mouseDragEvent)

    def getPlotItem(self) -> ItoolPlotItem:
        return self.plotItem

    def getPlotItemViewBox(self) -> pg.ViewBox:
        return self.getPlotItem().vb


class ImageSlicerArea(QtWidgets.QWidget):
    """A interactive tool based on :mod:`pyqtgraph` for exploring 3D data.

    Parameters
    ----------
    parent
        Parent widget.
    data
        Data to display. The data must have 2 to 4 dimensions.
    cmap
        Default colormap of the data.
    gamma
        Default power law normalization of the colormap.
    zeroCentered
        If `True`, the normalization is applied symmetrically from the midpoint of
        the colormap.
    rad2deg
        If `True` and `data` is not `None`, converts some known angle coordinates to
        degrees. If an iterable of strings is given, only the coordinates that
        correspond to the given strings are converted.

    Signals
    -------
    sigDataChanged()

    sigCurrentCursorChanged(index)

    sigViewOptionChanged()

    sigCursorCountChanged(n_cursors)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigIndexChanged(cursor, axes)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigBinChanged(cursor, axes)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.

    """

    COLORS: list[QtGui.QColor] = [
        pg.mkColor(0.8),
        pg.mkColor("y"),
        pg.mkColor("m"),
        pg.mkColor("c"),
        pg.mkColor("g"),
        pg.mkColor("r"),
        pg.mkColor("b"),
    ]  #: List of :class:`PySide6.QtGui.QColor` that contains colors for multiple cursors.

    sigDataChanged = QtCore.Signal()  #: :meta private:
    sigCurrentCursorChanged = QtCore.Signal(int)  #: :meta private:
    sigViewOptionChanged = QtCore.Signal()  #: :meta private:

    @property
    def sigCursorCountChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""
        return self.array_slicer.sigCursorCountChanged

    @property
    def sigIndexChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""
        return self.array_slicer.sigIndexChanged

    @property
    def sigBinChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""
        return self.array_slicer.sigBinChanged

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        data: xr.DataArray | npt.ArrayLike | None = None,
        cmap: str | pg.ColorMap = "magma",
        gamma: float = 0.5,
        zeroCentered: bool = False,
        rad2deg: bool | Iterable[str] = False,
    ):
        super().__init__(parent)

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self._splitters = (
            QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical),
            QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal),
            QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical),
            QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical),
            QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal),
            QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical),
            QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal),
        )
        for s in self._splitters:
            s.setHandleWidth(4)
            s.setStyleSheet("QSplitter::handle{background: #222222;}")
            # palette = s.palette()
            # palette.setColor(QtGui.QPalette.ColorRole.Light, QtGui.QColor("yellow"))
            # s.setPalette(palette)
            # print(s.handleWidth())
            # pass
        self.layout().addWidget(self._splitters[0])
        for i, j in ((0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (4, 6)):
            self._splitters[i].addWidget(self._splitters[j])
        _sync_splitters(self._splitters[1], self._splitters[4])

        self.cursor_colors: list[QtGui.QColor] = [self.COLORS[0]]

        self._colorbar = ItoolColorBar(self)
        self.layout().addWidget(self._colorbar)
        self._colorbar.setVisible(False)

        self._plots: tuple[ItoolGraphicsLayoutWidget, ...] = (
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(0, 1)),
            ItoolGraphicsLayoutWidget(self, display_axis=(0,)),
            ItoolGraphicsLayoutWidget(self, display_axis=(1,), is_vertical=True),
            ItoolGraphicsLayoutWidget(self, display_axis=(2,)),
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(0, 2)),
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(2, 1)),
            ItoolGraphicsLayoutWidget(self, display_axis=(3,)),
        )
        for i in (1, 4):
            self._splitters[2].addWidget(self._plots[i])
        for i in (6, 3):
            self._splitters[3].addWidget(self._plots[i])
        self._splitters[5].addWidget(self._plots[0])
        for i in (5, 2):
            self._splitters[6].addWidget(self._plots[i])

        self.qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
        self.qapp.aboutToQuit.connect(self.on_close)

        self.colormap_properties: dict[str, str | pg.ColorMap | float | bool] = dict(
            cmap=cmap,
            gamma=gamma,
            reversed=False,
            highContrast=False,
            zeroCentered=zeroCentered,
        )

        self._data: xr.DataArray | None = None
        self.current_cursor = 0

        if data is not None:
            self.set_data(data, rad2deg=rad2deg)

    def on_close(self):
        pass

    def connect_signals(self):
        for ax in self.axes:
            ax.connect_signals()
        self.sigDataChanged.connect(self.refresh_all)
        self.sigCursorCountChanged.connect(lambda: self.set_colormap(update=True))

    @property
    def colormap(self) -> str | pg.ColorMap:
        return self.colormap_properties["cmap"]

    @property
    def main_image(self) -> ItoolPlotItem:
        """returns the main PlotItem"""
        return self.get_axes(0)

    @property
    def slices(self) -> tuple[ItoolPlotItem, ...]:
        if self.data.ndim == 2:
            return tuple()
        else:
            return tuple(self.get_axes(ax) for ax in (4, 5))

    @property
    def images(self) -> tuple[ItoolPlotItem, ...]:
        return (self.main_image,) + self.slices

    @property
    def profiles(self) -> tuple[ItoolPlotItem, ...]:
        if self.data.ndim == 2:
            profile_axes = (1, 2)
        elif self.data.ndim == 3:
            profile_axes = (1, 2, 3)
        else:
            profile_axes = (1, 2, 3, 6)
        return tuple(self.get_axes(ax) for ax in profile_axes)

    @property
    def axes(self) -> tuple[ItoolPlotItem, ...]:
        """Currently valid subset of self._plots"""
        return self.images + self.profiles

    @property
    def _imageitems(self) -> tuple[ItoolImageItem, ...]:
        return tuple(im for ax in self.images for im in ax.slicer_data_items)

    @property
    def array_slicer(self) -> ArraySlicer:
        return self._array_slicer

    @property
    def n_cursors(self) -> int:
        return self.array_slicer.n_cursors

    @property
    def current_indices(self) -> list[int]:
        return self.array_slicer.get_indices(self.current_cursor)

    @property
    def current_values(self) -> list[float]:
        return self.array_slicer.get_values(self.current_cursor, uniform=False)

    @property
    def current_values_uniform(self) -> list[float]:
        return self.array_slicer.get_values(self.current_cursor, uniform=True)

    @property
    def data(self) -> xr.DataArray:
        return self.array_slicer._obj

    @property
    def color_locked(self) -> bool:
        return self._colorbar.isVisible()

    def get_current_index(self, axis: int) -> int:
        return self.array_slicer.get_index(self.current_cursor, axis)

    def get_current_value(self, axis: int, uniform: bool = False) -> float:
        return self.array_slicer.get_value(self.current_cursor, axis, uniform=uniform)

    def get_axes_widget(self, index: int) -> ItoolGraphicsLayoutWidget:
        return self._plots[index]

    def get_axes(self, index: int) -> ItoolPlotItem:
        return self._plots[index].plotItem

    @QtCore.Slot(tuple, bool, bool)
    def refresh_all(
        self, axes: tuple[int, ...] | None = None, only_plots: bool = False
    ):
        for c in range(self.n_cursors):
            self.sigIndexChanged.emit(c, axes)
        if not only_plots:
            for ax in self.axes:
                ax.refresh_labels()

    @QtCore.Slot(tuple)
    def refresh_current(self, axes: tuple[int, ...] | None = None):
        self.sigIndexChanged.emit(self.current_cursor, axes)

    @QtCore.Slot(int, list)
    def refresh(self, cursor: int, axes: tuple[int, ...] | None = None):
        self.sigIndexChanged.emit(cursor, axes)

    def view_all(self):
        for ax in self.axes:
            ax.vb.enableAutoRange()
            ax.vb.updateAutoRange()

    def center_all_cursors(self):
        for i in range(self.n_cursors):
            self.array_slicer.center_cursor(i)

    def center_cursor(self):
        self.array_slicer.center_cursor(self.current_cursor)

    def set_current_cursor(self, cursor: int, update=True):
        if cursor > self.n_cursors - 1:
            raise IndexError("Cursor index out of range")
        self.current_cursor = cursor
        if update:
            self.refresh_current()
        self.sigCurrentCursorChanged.emit(cursor)

    def set_data(
        self, data: xr.DataArray | npt.ArrayLike, rad2deg: bool | Iterable[str] = False
    ):
        if hasattr(self, "_array_slicer"):
            n_cursors_old = self.n_cursors
            if isinstance(self._data, xr.DataArray):
                self._data.close()
            del self._data
            del self._array_slicer
        else:
            n_cursors_old = 1

        if not isinstance(data, xr.DataArray):
            if isinstance(data, xr.Dataset):
                data = data.spectrum
            else:
                data = xr.DataArray(np.asarray(data))

        if not rad2deg:
            self._data = data
        else:
            if np.iterable(rad2deg):
                conv_dims = rad2deg
            else:
                conv_dims = [
                    d
                    for d in ("phi", "theta", "beta", "alpha", "chi")
                    if d in data.dims
                ]
            self._data = data.assign_coords({d: np.rad2deg(data[d]) for d in conv_dims})
        self._array_slicer = ArraySlicer(self._data)
        while self.n_cursors != n_cursors_old:
            self.array_slicer.add_cursor(update=False)

        self.connect_signals()

        self.adjust_layout()

        if self.current_cursor > self.n_cursors - 1:
            self.set_current_cursor(self.n_cursors - 1, update=False)
        self.sigDataChanged.emit()

        # self.refresh_current()
        self.set_colormap(update=True)
        self._colorbar.cb.setImageItem()
        self.lock_levels(False)

    @QtCore.Slot(int, int)
    def swap_axes(self, ax1: int, ax2: int):
        self.array_slicer.swap_axes(ax1, ax2)
        self.sigDataChanged.emit()

    @QtCore.Slot(int, int, bool)
    def set_index(self, axis: int, value: int, update: bool = True):
        self.array_slicer.set_index(self.current_cursor, axis, value, update)

    @QtCore.Slot(int, int, bool)
    def step_index(self, axis: int, amount: int, update: bool = True):
        self.array_slicer.step_index(self.current_cursor, axis, amount, update)

    @QtCore.Slot(int, int, bool)
    def step_index_all(self, axis: int, amount: int, update: bool = True):
        for i in range(self.n_cursors):
            self.array_slicer.step_index(i, axis, amount, update)

    @QtCore.Slot(int, float, bool, bool)
    def set_value(
        self, axis: int, value: float, update: bool = True, uniform: bool = False
    ):
        self.array_slicer.set_value(self.current_cursor, axis, value, update, uniform)

    @QtCore.Slot(int, int, bool)
    def set_bin(self, axis: int, value: int, update: bool = True):
        new_bins: list[int | None] = [None] * self.data.ndim
        new_bins[axis] = value
        self.array_slicer.set_bins(self.current_cursor, new_bins, update)

    @QtCore.Slot(int, int, bool)
    def set_bin_all(self, axis: int, value: int, update: bool = True):
        new_bins: list[int | None] = [None] * self.data.ndim
        new_bins[axis] = value
        for c in range(self.n_cursors):
            self.array_slicer.set_bins(c, new_bins, update)

    @QtCore.Slot()
    def add_cursor(self):
        self.array_slicer.add_cursor(self.current_cursor, update=False)
        self.cursor_colors.append(self.gen_cursor_color(self.n_cursors - 1))
        self.current_cursor = self.n_cursors - 1
        for ax in self.axes:
            ax.add_cursor(update=False)
        self.refresh_current()
        self.sigCursorCountChanged.emit(self.n_cursors)
        self.sigCurrentCursorChanged.emit(self.current_cursor)

    @QtCore.Slot(int)
    def remove_cursor(self, index: int):
        if self.n_cursors == 1:
            return
        self.array_slicer.remove_cursor(index, update=False)
        self.cursor_colors.pop(index)
        if self.current_cursor == index:
            if index == 0:
                self.current_cursor = 1
            self.current_cursor -= 1
        for ax in self.axes:
            ax.remove_cursor(index)
        self.refresh_current()
        self.sigCursorCountChanged.emit(self.n_cursors)
        self.sigCurrentCursorChanged.emit(self.current_cursor)

    @QtCore.Slot()
    def remove_current_cursor(self):
        self.remove_cursor(self.current_cursor)

    def gen_cursor_color(self, index: int) -> QtGui.QColor:
        clr = self.COLORS[index % len(self.COLORS)]
        while clr in self.cursor_colors:
            clr = self.COLORS[index % len(self.COLORS)]
            if self.n_cursors > len(self.COLORS):
                break
            index += 1
        return clr

    def gen_cursor_colors(
        self, index: int
    ) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor, QtGui.QColor, QtGui.QColor]:
        clr = self.cursor_colors[index]

        clr_cursor = pg.mkColor(clr)
        clr_cursor_hover = pg.mkColor(clr)
        clr_span = pg.mkColor(clr)
        clr_span_edge = pg.mkColor(clr)

        clr_cursor.setAlphaF(0.75)
        clr_cursor_hover.setAlphaF(0.95)
        clr_span.setAlphaF(0.15)
        clr_span_edge.setAlphaF(0.35)

        return clr, clr_cursor, clr_cursor_hover, clr_span, clr_span_edge

    def set_colormap(
        self,
        cmap: str | pg.ColorMap | None = None,
        gamma: float | None = None,
        reversed: bool | None = None,
        highContrast: bool | None = None,
        zeroCentered: bool | None = None,
        update: bool = True,
    ):
        if cmap is not None:
            self.colormap_properties["cmap"] = cmap
        if gamma is not None:
            self.colormap_properties["gamma"] = gamma
        if reversed is not None:
            self.colormap_properties["reversed"] = reversed
        if highContrast is not None:
            self.colormap_properties["highContrast"] = highContrast
        if zeroCentered is not None:
            self.colormap_properties["zeroCentered"] = zeroCentered

        cmap = pg_colormap_powernorm(
            self.colormap_properties["cmap"],
            self.colormap_properties["gamma"],
            self.colormap_properties["reversed"],
            highContrast=self.colormap_properties["highContrast"],
            zeroCentered=self.colormap_properties["zeroCentered"],
        )
        for im in self._imageitems:
            im.set_pg_colormap(cmap, update=update)
        self.sigViewOptionChanged.emit()

    @QtCore.Slot(bool)
    def lock_levels(self, lock: bool):
        if lock:
            levels = self.array_slicer.limits
            self._colorbar.cb.setLimits(levels)
        for im in self._imageitems:
            if lock:
                im.setAutoLevels(False)
                im.setLevels(levels, update=False)
            else:
                im.setAutoLevels(True)
            im.refresh_data()

        self._colorbar.setVisible(lock)
        self.sigViewOptionChanged.emit()

    def adjust_layout(
        self,
        horiz_pad: int = 45,
        vert_pad: int = 30,
        font_size: float = 11.0,
        r: tuple[float, float, float, float] = (1.2, 1.5, 3.0, 1.0),
    ):
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))

        # parameters for layout: stretch and axis on/off
        """
             ┌───────────┬───────────┐
        r[0] │     1     │     6     │
             │───────────┤           │
             │           ├───────────┤
        r[1] │     4     │     3     │
             │           │           │
             │───────────┼───────┬───┤
             │           │       │   │
        r[2] │     0     │   5   │ 2 │
             │           │       │   │
             └───────────┴───────┴───┘
              r[3] * r[2]
        """

        valid_axis = (
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (0, 0, 1, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 1, 1, 0),
        )

        invalid: list[int] = []
        r0, r1, r2, r3 = r
        if self.data.ndim == 2:
            invalid = [4, 5, 6]
            r1 = r0 / 6
        elif self.data.ndim == 3:
            invalid = [6]

        r01 = r0 / r1
        scale = 100
        d = self._splitters[0].handleWidth() / scale  # padding due to splitters
        sizes: tuple[tuple[float, ...], ...] = (
            (r0 + r1, r2),
            (r3 * r2, r3 * (r0 + r1)),
            ((r0 + r1 - d) * r01, (r0 + r1 - d) / r01),
            ((r0 + r1 - d) / 2, (r0 + r1 - d) / 2),
            (r3 * r2, r3 * (r0 + r1)),
            (r2,),
            ((r3 * (r0 + r1) - d) / r01, (r3 * (r0 + r1) - d) * r01),
        )
        for split, sz in zip(self._splitters, sizes):
            split.setSizes(tuple(map(lambda s: round(s * scale), sz)))

        for i, sel in enumerate(valid_axis):
            self.get_axes_widget(i).setVisible(i not in invalid)
            if i in invalid:
                continue
            axes = self.get_axes(i)
            axes.setDefaultPadding(0)
            for axis in ("left", "bottom", "right", "top"):
                axes.getAxis(axis).setTickFont(font)
                axes.getAxis(axis).setStyle(
                    autoExpandTextSpace=True, autoReduceTextSpace=True
                )
            axes.showAxes(sel, showValues=sel, size=(horiz_pad, vert_pad))
            if i in (1, 4):
                axes.setXLink(self.get_axes(0))
            elif i in (2, 5):
                axes.setYLink(self.get_axes(0))

        # reserve space, only hide plotItem
        self.get_axes(3).setVisible(not self.data.ndim == 2)

        self._colorbar.set_dimensions(
            width=horiz_pad + 30, horiz_pad=None, vert_pad=vert_pad, font_size=font_size
        )

    def toggle_snap(self, value: bool | None = None):
        if value is None:
            value = not self.array_slicer.snap_to_data
        elif value == self.array_slicer.snap_to_data:
            return
        self.array_slicer.snap_to_data = value
        self.sigViewOptionChanged.emit()

    def changeEvent(self, evt: QtCore.QEvent):
        if evt.type() == QtCore.QEvent.Type.PaletteChange:
            self.qapp.setStyle(self.qapp.style().name())
        super().changeEvent(evt)


class ItoolCursorLine(pg.InfiniteLine):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()

    @property
    def plotItem(self) -> ItoolPlotItem:
        return self.parentItem().parentItem().parentItem()

    def setBounds(self, bounds: Sequence[float], value: float | None = None):
        if bounds[0] > bounds[1]:
            bounds = list(bounds)
            bounds.reverse()
        self.maxRange = bounds
        if value is None:
            value = self.value()
        self.setValue(value)

    def value(self) -> float:
        return float(super().value())

    def mouseDragEvent(self, ev: mouseEvents.MouseDragEvent):
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            not in self.qapp.queryKeyboardModifiers()
        ):
            if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
                if ev.isStart():
                    self.moving = True
                    self.cursorOffset = self.pos() - self.mapToParent(
                        ev.buttonDownPos()
                    )
                    self.startPosition = self.pos()
                ev.accept()

                if not self.moving:
                    return

                new_position = self.cursorOffset + self.mapToParent(ev.pos())
                if self.angle % 180 == 0:
                    self.temp_value = new_position.y()
                elif self.angle % 180 == 90:
                    self.temp_value = new_position.x()

                self.sigDragged.emit(self)
                if ev.isFinish():
                    self.moving = False
                    self.sigPositionChangeFinished.emit(self)
        else:
            self.setMouseHover(False)
            self.plotItem.mouseDragEvent(ev)

    def mouseClickEvent(self, ev: mouseEvents.MouseClickEvent):
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            not in self.qapp.queryKeyboardModifiers()
        ):
            super().mouseClickEvent(ev)
        else:
            self.setMouseHover(False)
            ev.ignore()

    def hoverEvent(self, ev):
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            not in self.qapp.queryKeyboardModifiers()
        ):
            super().hoverEvent(ev)
        else:
            self.setMouseHover(False)


class ItoolCursorSpan(pg.LinearRegionItem):
    def __init__(self, *args, **kargs):
        kargs.setdefault("movable", False)
        super().__init__(*args, **kargs)

    def setRegion(self, rgn):
        # hides when region width is 0
        if rgn[1] == rgn[0]:
            self.setVisible(False)
        else:
            self.setVisible(True)
            super().setRegion(rgn)


class ItoolDisplayObject(object):
    def __init__(self, axes, cursor: int | None = None):
        super().__init__()
        self.axes = axes
        if cursor is None:
            cursor = 0
        self._cursor_index = int(cursor)
        self.qapp: QtGui.QGuiApplication = QtGui.QGuiApplication.instance()

    @property
    def display_axis(self):
        return self.axes.display_axis

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.axes.array_slicer

    @property
    def cursor_index(self):
        return self._cursor_index

    @cursor_index.setter
    def cursor_index(self, value: int):
        self._cursor_index = int(value)

    def refresh_data(self):
        pass


class ItoolPlotDataItem(pg.PlotDataItem, ItoolDisplayObject):
    def __init__(
        self,
        axes,
        cursor: int | None = None,
        is_vertical: bool = False,
        *args,
        **kargs,
    ):
        pg.PlotDataItem.__init__(self, *args, **kargs)
        ItoolDisplayObject.__init__(self, axes, cursor)
        self.is_vertical = is_vertical

    def refresh_data(self):
        ItoolDisplayObject.refresh_data(self)
        coord, vals = self.array_slicer.slice_with_coord(
            self.cursor_index, self.display_axis
        )
        if self.is_vertical:
            self.setData(vals, coord)
        else:
            self.setData(coord, vals)


class ItoolImageItem(BetterImageItem, ItoolDisplayObject):
    def __init__(
        self,
        axes,
        cursor: int | None = None,
        *args,
        **kargs,
    ):
        BetterImageItem.__init__(self, *args, **kargs)
        ItoolDisplayObject.__init__(self, axes, cursor)

    @suppressnanwarning
    def refresh_data(self):
        ItoolDisplayObject.refresh_data(self)
        rect, img = self.array_slicer.slice_with_coord(
            self.cursor_index, self.display_axis
        )
        self.setImage(image=img, rect=rect)

    def mouseDragEvent(self, ev: mouseEvents.MouseDragEvent):
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            in self.qapp.queryKeyboardModifiers()
        ):
            ev.ignore()
        else:
            super().mouseDragEvent(ev)

    def mouseClickEvent(self, ev: mouseEvents.MouseClickEvent):
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            in self.qapp.queryKeyboardModifiers()
        ):
            ev.ignore()
        else:
            super().mouseClickEvent(ev)


class ItoolPlotItem(pg.PlotItem):
    sigDragged = QtCore.Signal(object, object)

    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        display_axis: tuple[int, ...],
        image: bool = False,
        **item_kw,
    ):
        super().__init__(
            axisItems={a: BetterAxisItem(a) for a in ("left", "right", "top", "bottom")}
        )

        for action in self.getMenu().actions():
            if action.text() in [
                "Transforms",
                "Downsample",
                "Average",
                "Alpha",
                "Points",
            ]:
                action.setVisible(False)

        for i in (0, 1):
            self.getViewBoxMenu().ctrl[i].linkCombo.setVisible(False)
            self.getViewBoxMenu().ctrl[i].label.setVisible(False)
        self.getViewBox().setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        self.slicer_area = slicer_area
        self._display_axis = display_axis

        self.is_image = image
        self._item_kw = item_kw
        self.slicer_data_items: list[ItoolImageItem | ItoolPlotDataItem] = []
        self.cursor_lines: list[dict[int, ItoolCursorLine]] = []
        self.cursor_spans: list[dict[int, ItoolCursorSpan]] = []
        self.add_cursor(update=False)

        self.proxy = pg.SignalProxy(
            self.sigDragged,
            delay=1 / 75,
            rateLimit=75,
            slot=self.process_drag,
        )

    def getMenu(self) -> QtWidgets.QMenu:
        return self.ctrlMenu

    def getViewBox(self) -> pg.ViewBox:
        return self.vb

    def getViewBoxMenu(self) -> ViewBoxMenu:
        return self.getViewBox().menu

    def mouseDragEvent(self, ev: mouseEvents.MouseDragEvent):
        modifiers = self.slicer_area.qapp.queryKeyboardModifiers()
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier in modifiers
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            ev.accept()
            self.sigDragged.emit(ev, modifiers)
        else:
            ev.ignore()

    def process_drag(
        self, sig: tuple[mouseEvents.MouseDragEvent, QtCore.Qt.KeyboardModifier]
    ):
        ev, modifiers = sig
        data_pos = self.getViewBox().mapSceneToView(ev.scenePos())

        if (not self.is_image) and self.slicer_data_items[-1].is_vertical:
            data_pos_coords = (data_pos.y(), data_pos.x())
        else:
            data_pos_coords = (data_pos.x(), data_pos.y())

        if QtCore.Qt.KeyboardModifier.AltModifier in modifiers:
            for c in range(self.slicer_area.n_cursors):
                for i, ax in enumerate(self.display_axis):
                    self.array_slicer.set_value(
                        c, ax, data_pos_coords[i], update=False, uniform=True
                    )
                self.slicer_area.refresh(c, self.display_axis)
        else:
            for i, ax in enumerate(self.display_axis):
                self.slicer_area.set_value(
                    ax, data_pos_coords[i], update=False, uniform=True
                )
            self.slicer_area.refresh_current(self.display_axis)

    def add_cursor(self, update=True):
        new_cursor = len(self.slicer_data_items)
        line_angles = (90, 0)

        (
            clr,
            clr_cursor,
            clr_cursor_hover,
            clr_span,
            clr_span_edge,
        ) = self.slicer_area.gen_cursor_colors(new_cursor)

        if self.is_image:
            item = ItoolImageItem(
                self,
                cursor=new_cursor,
                autoDownsample=True,
                axisOrder="row-major",
                **self._item_kw,
            )
            if self.slicer_area.color_locked:
                item.setAutoLevels(False)
                item.setLevels(self.array_slicer.limits, update=True)
        else:
            item = ItoolPlotDataItem(
                self,
                cursor=new_cursor,
                pen=pg.mkPen(pg.mkColor(clr)),
                defaultPadding=0.0,
                clipToView=False,
                connect="auto",
                **self._item_kw,
            )
            if item.is_vertical:
                line_angles = (0, 90)
        self.slicer_data_items.append(item)
        self.addItem(item)

        cursors = []
        spans = []
        for ang in line_angles:
            cursors.append(
                ItoolCursorLine(
                    pen=pg.mkPen(clr_cursor, width=1),
                    hoverPen=pg.mkPen(clr_cursor_hover, width=1),
                    angle=ang,
                    movable=True,
                )
            )
            spans.append(
                ItoolCursorSpan(
                    orientation="vertical" if ang == 90 else "horizontal",
                    movable=False,
                    pen=pg.mkPen(clr_span_edge),
                    brush=pg.mkBrush(clr_span),
                )
            )

        self.cursor_lines.append({})
        self.cursor_spans.append({})
        for c, s, ax in zip(cursors, spans, self.display_axis):
            self.cursor_lines[-1][ax] = c
            self.cursor_spans[-1][ax] = s
            self.addItem(c)
            c.setZValue(10)
            self.addItem(s)
            s.setZValue(9)
            c.sigDragged.connect(
                lambda v, *, line=c, axis=ax: self.line_drag(line, v.temp_value, axis)
            )
            c.sigClicked.connect(lambda *, line=c: self.line_click(line))

        if update:
            self.refresh_cursor(new_cursor)

    def index_of_line(self, line: ItoolCursorLine) -> int:
        for i, line_dict in enumerate(self.cursor_lines):
            for _, v in line_dict.items():
                if v == line:
                    return i
        raise ValueError("`line` is not a valid cursor.")

    def line_click(self, line: ItoolCursorLine):
        cursor = self.index_of_line(line)
        if cursor != self.slicer_area.current_cursor:
            self.slicer_area.set_current_cursor(cursor, update=True)

    def line_drag(self, line: ItoolCursorLine, value: float, axis: int):
        cursor = self.index_of_line(line)
        if cursor != self.slicer_area.current_cursor:
            self.slicer_area.set_current_cursor(cursor, update=True)
        if (
            self.slicer_area.qapp.queryKeyboardModifiers()
            != QtCore.Qt.KeyboardModifier.AltModifier
        ):
            self.array_slicer.set_value(cursor, axis, value, update=True, uniform=True)
        else:
            for i in range(self.array_slicer.n_cursors):
                self.array_slicer.set_value(i, axis, value, update=True, uniform=True)

    def remove_cursor(self, index: int):
        item = self.slicer_data_items.pop(index)
        self.removeItem(item)
        for line, span in zip(
            self.cursor_lines.pop(index).values(), self.cursor_spans.pop(index).values()
        ):
            self.removeItem(line)
            self.removeItem(span)
        for i, item in enumerate(self.slicer_data_items):
            item.cursor_index = i

    def refresh_cursor(self, cursor: int):
        for ax, line in self.cursor_lines[cursor].items():
            line.setBounds(
                self.array_slicer.lims_uniform[ax],
                self.array_slicer.get_value(cursor, ax, uniform=True),
            )
            self.cursor_spans[cursor][ax].setRegion(
                self.array_slicer.span_bounds(cursor, ax)
            )

    def connect_signals(self):
        self._slicer_area.sigIndexChanged.connect(self.refresh_items_data)
        self._slicer_area.sigBinChanged.connect(self.refresh_items_data)

    @QtCore.Slot(int, tuple)
    def refresh_items_data(self, cursor: int, axes: tuple[int] | None = None):
        self.refresh_cursor(cursor)
        if axes is not None:
            # display_axis는 축 dim 표시하는거임. 즉 해당 축만 바뀌면 데이터 변화 없음
            if all(elem in self.display_axis for elem in axes):
                return
        for item in self.slicer_data_items:
            if item.cursor_index != cursor:
                continue
            self.set_active_cursor(cursor)
            item.refresh_data()
        # TODO: autorange smarter
        self.vb.updateAutoRange()

    @QtCore.Slot()
    def refresh_labels(self):
        if self.is_image:
            label_kw = {
                a: self._get_label_unit(i)
                for a, i in zip(("top", "bottom", "left", "right"), (0, 0, 1, 1))
                if self.getAxis(a).isVisible()
            }
        else:
            label_kw = dict()

            if self.slicer_data_items[-1].is_vertical:
                valid_ax = ("left", "right")
            else:
                valid_ax = ("top", "bottom")

            for a in ("top", "bottom", "left", "right"):
                if self.getAxis(a).isVisible():
                    label_kw[a] = self._get_label_unit(0 if a in valid_ax else None)
        self.setLabels(**label_kw)

    def _get_label_unit(self, index: int | None = None):
        if index is None:
            return "", ""
        dim = self.slicer_area.data.dims[self.display_axis[index]]
        if dim == "eV":
            return dim, "eV"
        # if dim in ("kx", "ky", "phi", "theta", "beta", "alpha", "chi"):
        # return dim
        return dim, ""

    def set_active_cursor(self, index: int):
        if self.is_image:
            for i, (item, cursors) in enumerate(
                zip(self.slicer_data_items, self.cursor_lines)
            ):
                item.setVisible(i == index)
                # for line in cursors.values():
                # line.setMovable(i == index)
        else:
            pass
            # for i, cursors in enumerate(self.cursor_lines):
            # for line in cursors.values():
            # line.setMovable(i == index)

    @property
    def display_axis(self) -> tuple[int, ...]:
        return self._display_axis

    @display_axis.setter
    def display_axis(self, value: tuple[int, ...]):
        self._display_axis = value

    @property
    def slicer_area(self) -> ImageSlicerArea:
        return self._slicer_area

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea):
        self._slicer_area = value

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer


class IconButton(QtWidgets.QPushButton):
    ICON_ALIASES = dict(
        invert="mdi6.invert-colors",
        invert_off="mdi6.invert-colors-off",
        contrast="mdi6.contrast-box",
        lock="mdi6.lock",
        unlock="mdi6.lock-open-variant",
        bright_auto="mdi6.brightness-auto",
        bright_percent="mdi6.brightness-percent",
        colorbar="mdi6.gradient-vertical",
        transpose_0="mdi6.arrow-top-left-bottom-right",
        transpose_1="mdi6.arrow-up-down",
        transpose_2="mdi6.arrow-left-right",
        transpose_3="mdi6.axis-z-arrow",
        snap="mdi6.grid",
        snap_off="mdi6.grid-off",
        palette="mdi6.palette-advanced",
        styles="mdi6.palette-swatch",
        layout="mdi6.page-layout-body",
        zero_center="mdi6.format-vertical-align-center",
        table_eye="mdi6.table-eye",
        plus="mdi6.plus",
        minus="mdi6.minus",
        reset="mdi6.backup-restore",
        # all_cursors="mdi6.checkbox-multiple-outline",
        all_cursors="mdi6.select-multiple",
    )

    def __init__(self, on: str | None = None, off: str | None = None, **kwargs):
        self.icon_key_on = None
        self.icon_key_off = None

        if on is not None:
            self.icon_key_on = on
            kwargs["icon"] = self.get_icon(self.icon_key_on)

        if off is not None:
            if on is None and kwargs["icon"] is None:
                raise ValueError("Icon for `on` state was not supplied.")
            self.icon_key_off = off
            kwargs.setdefault("checkable", True)

        super().__init__(**kwargs)
        if self.isCheckable() and off is not None:
            self.toggled.connect(self.refresh_icons)

    def setChecked(self, value: bool):
        super().setChecked(value)
        self.refresh_icons()

    def get_icon(self, icon: str):
        try:
            return qta.icon(self.ICON_ALIASES[icon])
        except KeyError:
            return qta.icon(icon)

    def refresh_icons(self):
        if self.icon_key_off is not None:
            if self.isChecked():
                self.setIcon(self.get_icon(self.icon_key_off))
                return
        if self.icon_key_on is not None:
            self.setIcon(self.get_icon(self.icon_key_on))

    def changeEvent(self, evt: QtCore.QEvent):  # handles dark mode
        if evt.type() == QtCore.QEvent.Type.PaletteChange:
            qta.reset_cache()
            self.refresh_icons()
        super().changeEvent(evt)


class ColorMapComboBox(QtWidgets.QComboBox):
    LOAD_ALL_TEXT = "Load all..."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPlaceholderText("Select colormap...")
        self.setToolTip("Colormap")
        w, h = 64, 16
        self.setIconSize(QtCore.QSize(w, h))
        # for name in pg_colormap_names("local"):
        for name in pg_colormap_names("mpl"):
            self.addItem(name)
        self.insertItem(0, self.LOAD_ALL_TEXT)
        self.thumbnails_loaded = False
        self.currentIndexChanged.connect(self.load_thumbnail)
        self.default_cmap = None

        sc_p = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Alt+Up"), self)
        sc_p.activated.connect(self.previousIndex)
        sc_m = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Alt+Down"), self)
        sc_m.activated.connect(self.nextIndex)

    def load_thumbnail(self, index: int):
        if not self.thumbnails_loaded:
            text = self.itemText(index)
            try:
                self.setItemIcon(index, QtGui.QIcon(pg_colormap_to_QPixmap(text)))
            except KeyError:
                pass

    def load_all(self):
        self.clear()
        for name in pg_colormap_names("all"):
            self.addItem(QtGui.QIcon(pg_colormap_to_QPixmap(name)), name)
        self.resetCmap()
        self.showPopup()

    # https://forum.qt.io/topic/105012/qcombobox-specify-width-less-than-content/11
    def showPopup(self):
        maxWidth = self.maximumWidth()
        if maxWidth and maxWidth < 16777215:
            self.setPopupMinimumWidthForItems()
        if not self.thumbnails_loaded:
            for i in range(self.count()):
                self.load_thumbnail(i)
            self.thumbnails_loaded = True
        super().showPopup()

    def setPopupMinimumWidthForItems(self):
        view = self.view()
        fm = self.fontMetrics()
        maxWidth = max(
            [fm.boundingRect(self.itemText(i)).width() for i in range(self.count())]
        )
        if maxWidth:
            view.setMinimumWidth(maxWidth)

    @QtCore.Slot()
    def nextIndex(self):
        self.wheelEvent(
            QtGui.QWheelEvent(
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, -15),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.ScrollPhase.ScrollUpdate,
                True,
            )
        )

    @QtCore.Slot()
    def previousIndex(self):
        self.wheelEvent(
            QtGui.QWheelEvent(
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, 15),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.ScrollPhase.ScrollUpdate,
                True,
            )
        )

    def hidePopup(self):
        self.activated.emit(self.currentIndex())
        self.textActivated.emit(self.currentText())
        self.currentIndexChanged.emit(self.currentIndex())
        self.currentTextChanged.emit(self.currentText())
        super().hidePopup()

    def setDefaultCmap(self, cmap: str):
        self.default_cmap = cmap
        self.setCurrentText(cmap)

    def resetCmap(self):
        if self.default_cmap is None:
            self.setCurrentIndex(0)
        else:
            self.setCurrentText(self.default_cmap)


class ColorMapGammaWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.setLayout(QtWidgets.QHBoxLayout(self))
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().setSpacing(3)
        self.spin = BetterSpinBox(
            self,
            toolTip="Colormap gamma",
            decimals=2,
            wrapping=False,
            keyboardTracking=False,
            singleStep=0.1,
            minimum=0.01,
            maximum=99.99,
            value=value,
        )
        self.label = QtWidgets.QLabel(self, text="γ", buddy=self.spin)
        # self.label.setIndent(0)
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.slider = QtWidgets.QSlider(
            self,
            toolTip="Colormap gamma",
            value=self.gamma_scale(value),
            singleStep=1,
            orientation=QtCore.Qt.Orientation.Horizontal,
        )
        self.slider.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.slider.setMinimumWidth(30)
        self.spin.valueChanged.connect(self.spin_changed)

        self.slider.setRange(
            self.gamma_scale(self.spin.minimum()),
            self.gamma_scale(self.spin.maximum()),
        )
        self.slider.valueChanged.connect(self.slider_changed)

        self.layout().addWidget(self.label)
        self.layout().addWidget(self.spin)
        self.layout().addWidget(self.slider)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float):
        self.spin.setValue(value)
        self.slider.setValue(self.gamma_scale(value))

    def spin_changed(self, value: float):
        self.slider.blockSignals(True)
        self.slider.setValue(self.gamma_scale(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def slider_changed(self, value: float):
        self.spin.setValue(self.gamma_scale_inv(value))

    def gamma_scale(self, y: float) -> float:
        return 1e4 * np.log10(y)

    def gamma_scale_inv(self, x: float) -> float:
        return np.power(10, x * 1e-4)


def clear_layout(layout: QtWidgets.QLayout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


class ItoolColorBarItem(BetterColorBarItem):
    def __init__(self, slicer_area: ImageSlicerArea | None = None, **kwargs):
        self._slicer_area = slicer_area
        super().__init__(**kwargs)

    @property
    def slicer_area(self) -> ImageSlicerArea:
        return self._slicer_area

    @property
    def images(self):
        return [weakref.ref(x) for x in self._slicer_area._imageitems]

    @property
    def primary_image(self):
        return weakref.ref(self._slicer_area.main_image.slicer_data_items[0])

    def setImageItem(self, *args, **kwargs):
        self.slicer_area.sigViewOptionChanged.connect(self.limit_changed)

        self._span.blockSignals(True)
        self._span.setRegion(self.limits)
        self._span.blockSignals(False)
        self._span.sigRegionChanged.connect(self._level_change)
        self._span.sigRegionChangeFinished.connect(self._level_change_fin)
        self.color_changed()


class ItoolColorBar(pg.PlotWidget):
    def __init__(self, slicer_area: ImageSlicerArea | None = None, **cbar_kw):
        super().__init__(
            parent=slicer_area, plotItem=ItoolColorBarItem(slicer_area, **cbar_kw)
        )
        self.scene().sigMouseClicked.connect(self.mouseDragEvent)

    @property
    def cb(self) -> BetterColorBarItem:
        return self.plotItem

    def set_dimensions(
        self,
        width: int = 30,
        horiz_pad: int | None = None,
        vert_pad: int | None = None,
        font_size: float = 11.0,
    ):
        self.cb.set_dimensions(horiz_pad, vert_pad, font_size)
        self.setFixedWidth(width)

    def setVisible(self, visible: bool):
        super().setVisible(visible)
        self.cb.setVisible(visible)
        self.cb._span.blockSignals(not visible)

        if visible:
            self.cb._span.setRegion(self.cb.limits)


class ItoolControlsBase(QtWidgets.QWidget):
    def __init__(
        self, slicer_area: ImageSlicerArea | ItoolControlsBase, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._slicer_area = slicer_area
        self.sub_controls = []
        self.initialize_layout()
        self.initialize_widgets()
        self.connect_signals()
        self.update()

    @property
    def data(self) -> xr.DataArray:
        return self.slicer_area.data

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    @property
    def n_cursors(self) -> int:
        return self.slicer_area.n_cursors

    @property
    def current_cursor(self) -> int:
        return self.slicer_area.current_cursor

    def initialize_layout(self):
        self.setLayout(QtWidgets.QHBoxLayout(self))
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().setSpacing(3)

    def initialize_widgets(self):
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.initialize_widgets()

    def connect_signals(self):
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.connect_signals()

    def disconnect_signals(self):
        # Multiple inheritance disconnection is broken
        # https://bugreports.qt.io/browse/PYSIDE-229
        # Will not work correctly until this is fixed
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.disconnect_signals()

    def update(self):
        for ctrl in self.sub_controls:
            ctrl.update()

    def add_control(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.sub_controls.append(widget)
        return widget

    @property
    def is_nested(self) -> bool:
        return isinstance(self._slicer_area, ItoolControlsBase)

    @property
    def slicer_area(self) -> ImageSlicerArea:
        if isinstance(self._slicer_area, ItoolControlsBase):
            return self._slicer_area.slicer_area
        else:
            return self._slicer_area

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea):
        # ignore until https://bugreports.qt.io/browse/PYSIDE-229 is fixed
        try:
            self.disconnect_signals()
        except RuntimeError:
            pass
        self._slicer_area = value
        clear_layout(self.layout())
        self.sub_controls = []
        self.initialize_widgets()
        self.update()
        self.connect_signals()


class ColorControls(ItoolControlsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_widgets(self):
        self.btn_reverse = IconButton(
            on="invert",
            off="invert_off",
            checkable=True,
            toolTip="Invert colormap",
        )
        self.btn_contrast = IconButton(
            on="contrast",
            checkable=True,
            toolTip="High contrast mode",
        )
        self.btn_zero = IconButton(
            on="zero_center",
            checkable=True,
            toolTip="Center colormap at zero",
        )
        self.btn_lock = IconButton(
            # on="unlock", off="lock",
            on="bright_auto",
            off="bright_percent",
            checkable=True,
            toolTip="Lock color limits",
        )
        self.btn_reverse.toggled.connect(self.update_colormap)
        self.btn_contrast.toggled.connect(self.update_colormap)
        self.btn_zero.toggled.connect(self.update_colormap)

        self.layout().addWidget(self.btn_reverse)
        self.layout().addWidget(self.btn_contrast)
        self.layout().addWidget(self.btn_zero)
        self.layout().addWidget(self.btn_lock)

    def update(self):
        self.btn_reverse.blockSignals(True)
        self.btn_contrast.blockSignals(True)
        self.btn_zero.blockSignals(True)
        self.btn_lock.blockSignals(True)

        self.btn_reverse.setChecked(self.slicer_area.colormap_properties["reversed"])
        self.btn_contrast.setChecked(
            self.slicer_area.colormap_properties["highContrast"]
        )
        self.btn_zero.setChecked(self.slicer_area.colormap_properties["zeroCentered"])
        self.btn_lock.setChecked(self.slicer_area.color_locked)

        self.btn_reverse.blockSignals(False)
        self.btn_contrast.blockSignals(False)
        self.btn_zero.blockSignals(False)
        self.btn_lock.blockSignals(False)

    def update_colormap(self):
        self.slicer_area.set_colormap(
            reversed=self.btn_reverse.isChecked(),
            highContrast=self.btn_contrast.isChecked(),
            zeroCentered=self.btn_zero.isChecked(),
        )

    def connect_signals(self):
        super().connect_signals()
        self.btn_lock.toggled.connect(self.slicer_area.lock_levels)
        self.slicer_area.sigViewOptionChanged.connect(self.update)

    def disconnect_signals(self):
        super().disconnect_signals()
        self.btn_lock.toggled.disconnect(self.slicer_area.lock_levels)
        self.slicer_area.sigViewOptionChanged.disconnect(self.update)


# class ItoolAAAAAControls(ItoolControlsBase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def initialize_layout(self):
#         pass

#     def initialize_widgets(self):
#         pass

#     def connect_signals(self):

#         pass

#     def disconnect_signals(self):
#         pass

#     def update(self):
#         pass


class ItoolCrosshairControls(ItoolControlsBase):
    def __init__(self, *args, orientation=QtCore.Qt.Orientation.Vertical, **kwargs):
        if isinstance(orientation, QtCore.Qt.Orientation):
            self.orientation = orientation
        elif orientation == "vertical":
            self.orientation = QtCore.Qt.Orientation.Vertical
        elif orientation == "horizontal":
            self.orientation = QtCore.Qt.Orientation.Horizontal
        super().__init__(*args, **kwargs)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred
        )

    def initialize_widgets(self):
        super().initialize_widgets()
        self.values_groups = tuple(
            QtWidgets.QWidget() for _ in range(self.data.ndim + 1)
        )
        self.values_layouts = tuple(
            QtWidgets.QGridLayout(g) for g in self.values_groups
        )
        for s in self.values_layouts:
            s.setContentsMargins(0, 0, 0, 0)

            s.setSpacing(3)
        # buttons for multicursor control
        self.btn_add = IconButton("plus", toolTip="Add cursor")
        self.btn_add.clicked.connect(self.slicer_area.add_cursor)

        self.btn_rem = IconButton("minus", toolTip="Remove cursor")
        self.btn_rem.clicked.connect(
            lambda: self.slicer_area.remove_cursor(self.cb_cursors.currentIndex())
        )

        self.btn_snap = IconButton(
            on="snap", off="snap_off", toolTip="Snap cursor to data points"
        )
        self.btn_snap.toggled.connect(self.slicer_area.toggle_snap)

        # multicursor combobox
        self.cb_cursors = QtWidgets.QComboBox()
        self.cb_cursors.textActivated.connect(self.setActiveCursor)
        self.cb_cursors.setMaximumHeight(
            QtGui.QFontMetrics(self.cb_cursors.font()).height() + 3
        )
        self.cb_cursors.setIconSize(QtCore.QSize(10, 10))
        for i in range(self.n_cursors):
            self.cb_cursors.addItem(
                QtGui.QIcon(self._cursor_icon(i)), self._cursor_name(i)
            )
        if self.n_cursors == 1:
            # can't remove more cursors
            self.cb_cursors.setDisabled(True)
            self.btn_rem.setDisabled(True)

        # current value widget
        self.spin_dat = BetterSpinBox(
            self.values_groups[-1], discrete=False, scientific=True, readOnly=True
        )
        self.spin_dat.setDecimals(round(abs(np.log10(self.array_slicer.absnanmax)) + 1))

        # add multicursor widgets
        if self.orientation == QtCore.Qt.Orientation.Vertical:
            self.values_layouts[0].addWidget(self.btn_add, 0, 1, 1, 1)
            self.values_layouts[0].addWidget(self.btn_rem, 0, 2, 1, 1)
            self.values_layouts[0].addWidget(self.btn_snap, 0, 0, 1, 1)
            self.values_layouts[0].addWidget(self.cb_cursors, 1, 0, 1, 3)
            self.values_layouts[0].addWidget(self.spin_dat, 2, 0, 1, 3)
        else:
            self.values_layouts[0].addWidget(self.btn_add, 0, 1, 1, 1)
            self.values_layouts[0].addWidget(self.btn_rem, 0, 2, 1, 1)
            self.values_layouts[0].addWidget(self.btn_snap, 0, 3, 1, 1)
            self.values_layouts[0].addWidget(self.cb_cursors, 0, 0, 1, 1)
            self.values_layouts[0].addWidget(self.spin_dat, 0, 4, 1, 1)
        self.layout().addWidget(self.values_groups[0])

        # info widgets
        self.label_dim = tuple(
            QtWidgets.QPushButton(grp, checkable=True) for grp in self.values_groups[1:]
        )
        self.spin_idx = tuple(
            BetterSpinBox(
                grp,
                integer=True,
                singleStep=1,
                wrapping=False,
                minimumWidth=60,
                keyboardTracking=False,
            )
            for grp in self.values_groups[1:]
        )
        self.spin_val = tuple(
            BetterSpinBox(
                grp,
                discrete=True,
                decimals=3,
                wrapping=False,
                minimumWidth=70,
                keyboardTracking=False,
            )
            for grp in self.values_groups[1:]
        )

        self.btn_transpose = tuple(
            IconButton(on=f"transpose_{i}") for i in range(self.data.ndim)
        )

        # add and connect info widgets
        for i in range(self.data.ndim):
            # TODO: implelemnt cursor locking
            # self.label_dim[i].toggled.connect()
            self.spin_idx[i].valueChanged.connect(
                lambda ind, axis=i: self.slicer_area.set_index(axis, ind)
            )
            self.spin_val[i].valueChanged.connect(
                lambda val, axis=i: self.slicer_area.set_value(axis, val, uniform=False)
            )
            self.btn_transpose[i].clicked.connect(
                lambda *, idx=i: self._transpose_axes(idx)
            )
            if self.orientation == QtCore.Qt.Orientation.Vertical:
                self.values_layouts[i + 1].addWidget(self.label_dim[i], 0, 0, 1, 1)
                self.values_layouts[i + 1].addWidget(self.btn_transpose[i], 0, 1, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_idx[i], 1, 0, 1, 2)
                self.values_layouts[i + 1].addWidget(self.spin_val[i], 2, 0, 1, 2)
            else:
                self.values_layouts[i + 1].addWidget(self.label_dim[i], 0, 0, 1, 1)
                self.values_layouts[i + 1].addWidget(self.btn_transpose[i], 0, 1, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_idx[i], 0, 2, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_val[i], 0, 3, 1, 1)

            self.layout().addWidget(self.values_groups[i + 1])

    def _transpose_axes(self, idx):
        if self.data.ndim == 4:
            if idx == 3:
                self.slicer_area.swap_axes(2, 3)
            else:
                self.slicer_area.swap_axes(idx, (idx + 1) % 3)
        else:
            self.slicer_area.swap_axes(idx, (idx + 1) % self.data.ndim)

    def connect_signals(self):
        super().connect_signals()
        self.slicer_area.sigCurrentCursorChanged.connect(self.cursorChangeEvent)
        self.slicer_area.sigCursorCountChanged.connect(self.update_cursor_count)
        self.slicer_area.sigViewOptionChanged.connect(self.update_options)
        self.slicer_area.sigIndexChanged.connect(self.update_spins)
        self.slicer_area.sigBinChanged.connect(self.update_spins)
        self.slicer_area.sigDataChanged.connect(self.update)

    def disconnect_signals(self):
        super().disconnect_signals()
        self.slicer_area.sigCurrentCursorChanged.disconnect(self.cursorChangeEvent)
        self.slicer_area.sigCursorCountChanged.disconnect(self.update_cursor_count)
        self.slicer_area.sigViewOptionChanged.disconnect(self.update_options)
        self.slicer_area.sigIndexChanged.disconnect(self.update_spins)
        self.slicer_area.sigBinChanged.disconnect(self.update_spins)
        self.slicer_area.sigDataChanged.disconnect(self.update)

    def update(self):
        super().update()
        if len(self.label_dim) != self.data.ndim:
            # number of required cursors changed, resetting
            clear_layout(self.layout())
            self.initialize_widgets()

        label_width = 0
        for i in range(self.data.ndim):
            self.values_groups[i].blockSignals(True)
            self.spin_idx[i].blockSignals(True)
            self.spin_val[i].blockSignals(True)

            if i in self.array_slicer._nonuniform_axes:
                self.label_dim[i].setText(str(self.data.dims[i])[:-4])
            else:
                self.label_dim[i].setText(str(self.data.dims[i]))

            label_width = max(
                label_width,
                self.label_dim[i]
                .fontMetrics()
                .boundingRect(self.label_dim[i].text())
                .width()
                + 15,
            )

            # update spinbox properties to match new data
            self.spin_idx[i].setRange(0, self.data.shape[i] - 1)
            self.spin_idx[i].setValue(self.slicer_area.get_current_index(i))

            self.spin_val[i].setRange(*self.array_slicer.lims[i])
            self.spin_val[i].setSingleStep(self.array_slicer.incs[i])
            self.spin_val[i].setValue(self.slicer_area.get_current_value(i))

            self.label_dim[i].blockSignals(False)
            self.spin_idx[i].blockSignals(False)
            self.spin_val[i].blockSignals(False)

        self.spin_dat.setDecimals(round(abs(np.log10(self.array_slicer.absnanmax)) + 1))
        self.spin_dat.setValue(
            self.array_slicer.point_value(self.current_cursor, binned=True)
        )

        for lab in self.label_dim:
            # resize buttons to match dimension label length
            lab.setMaximumWidth(label_width)

    def update_spins(self, _=None, axes=None):
        if axes is None:
            axes = range(self.data.ndim)
        for i in axes:
            self.spin_idx[i].blockSignals(True)
            self.spin_val[i].blockSignals(True)
            self.spin_idx[i].setValue(self.slicer_area.current_indices[i])
            self.spin_val[i].setValue(self.slicer_area.current_values[i])
            self.spin_idx[i].blockSignals(False)
            self.spin_val[i].blockSignals(False)

        self.spin_dat.setValue(
            self.array_slicer.point_value(self.current_cursor, binned=True)
        )

    def update_options(self):
        self.btn_snap.blockSignals(True)
        self.btn_snap.setChecked(self.array_slicer.snap_to_data)
        # self.btn_snap.refresh_icons()
        self.btn_snap.blockSignals(False)

    def _cursor_name(self, i: int) -> str:
        # for cursor combobox content
        return f" Cursor {int(i)}"

    def _cursor_icon(self, i: int) -> QtGui.QPixmap:
        img = QtGui.QImage(32, 32, QtGui.QImage.Format.Format_RGBA64)
        img.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(img)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing, True)

        clr = self.slicer_area.cursor_colors[i]
        painter.setBrush(pg.mkColor(clr))
        painter.drawEllipse(img.rect())
        painter.end()
        return QtGui.QPixmap.fromImage(img)

    def update_cursor_count(self, count: int):
        if count == self.cb_cursors.count():
            return
        elif count > self.cb_cursors.count():
            self.addCursor()
        else:
            self.remCursor()

    def addCursor(self):
        self.cb_cursors.setDisabled(False)
        # self.slicer_area.add_cursor()
        self.cb_cursors.addItem(
            QtGui.QIcon(self._cursor_icon(self.current_cursor)),
            self._cursor_name(self.current_cursor),
        )
        self.cb_cursors.setCurrentIndex(self.current_cursor)
        self.btn_rem.setDisabled(False)

    def remCursor(self):
        # self.slicer_area.remove_cursor(self.cb_cursors.currentIndex())
        self.cb_cursors.removeItem(self.cb_cursors.currentIndex())
        for i in range(self.cb_cursors.count()):
            self.cb_cursors.setItemText(i, self._cursor_name(i))
            self.cb_cursors.setItemIcon(i, self._cursor_icon(i))
        self.cb_cursors.setCurrentText(self._cursor_name(self.current_cursor))
        if i == 0:
            self.cb_cursors.setDisabled(True)
            self.btn_rem.setDisabled(True)

    def cursorChangeEvent(self, idx: int):
        self.cb_cursors.setCurrentIndex(idx)
        self.update_spins()

    def setActiveCursor(self, value: str):
        self.slicer_area.set_current_cursor(self.cb_cursors.findText(value))


class ItoolColormapControls(ItoolControlsBase):
    def __init__(self, *args, orientation=QtCore.Qt.Orientation.Vertical, **kwargs):
        if isinstance(orientation, QtCore.Qt.Orientation):
            self.orientation = orientation
        elif orientation == "vertical":
            self.orientation = QtCore.Qt.Orientation.Vertical
        elif orientation == "horizontal":
            self.orientation = QtCore.Qt.Orientation.Horizontal
        super().__init__(*args, **kwargs)

    def initialize_layout(self):
        if self.orientation == QtCore.Qt.Orientation.Vertical:
            self.setLayout(QtWidgets.QVBoxLayout(self))
        else:
            self.setLayout(QtWidgets.QHBoxLayout(self))
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().setSpacing(3)

    def initialize_widgets(self):
        super().initialize_widgets()
        self.cb_colormap = ColorMapComboBox(self, maximumWidth=175)
        self.cb_colormap.textActivated.connect(self.change_colormap)

        self.gamma_widget = ColorMapGammaWidget()
        self.gamma_widget.valueChanged.connect(
            lambda g: self.slicer_area.set_colormap(gamma=g)
        )

        self.misc_controls = self.add_control(ColorControls(self))

        self.layout().addWidget(self.cb_colormap)
        self.layout().addWidget(self.gamma_widget)
        self.layout().addWidget(self.misc_controls)

    def update(self):
        super().update()
        if isinstance(self.slicer_area.colormap, str):
            self.cb_colormap.setDefaultCmap(self.slicer_area.colormap)
        self.gamma_widget.blockSignals(True)
        self.gamma_widget.setValue(self.slicer_area.colormap_properties["gamma"])
        self.gamma_widget.blockSignals(False)

    def change_colormap(self, name):
        if name == self.cb_colormap.LOAD_ALL_TEXT:
            self.cb_colormap.load_all()
        else:
            self.slicer_area.set_colormap(name)


class ItoolBinningControls(ItoolControlsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_layout(self):
        self.setLayout(QtWidgets.QGridLayout(self))
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().setSpacing(3)

    def initialize_widgets(self):
        super().initialize_widgets()
        self.labels = tuple(QtWidgets.QLabel() for _ in range(self.data.ndim))
        self.spins = tuple(
            BetterSpinBox(
                self,
                integer=True,
                singleStep=2,
                # minimumWidth=60,
                value=1,
                minimum=1,
                maximum=self.data.shape[i],
                keyboardTracking=False,
            )
            for i in range(self.data.ndim)
        )
        for i, spin in enumerate(self.spins):
            spin.valueChanged.connect(lambda n, axis=i: self._update_bin(axis, n))

        self.reset_btn = IconButton("reset")
        self.reset_btn.clicked.connect(self.reset)

        self.all_btn = IconButton(
            on="all_cursors",
            checkable=True,
            toolTip="Apply changes for all cursors",
        )

        for i in range(self.data.ndim):
            self.layout().addWidget(self.labels[i], 0, i, 1, 1)
            self.layout().addWidget(self.spins[i], 1, i, 1, 1)
        self.layout().addWidget(self.reset_btn, 2, 0, 1, 1)
        self.layout().addWidget(self.all_btn, 2, 1, 1, 1)
        # for spin in self.spins:
        # spin.setMinimumWidth(60)

    def _update_bin(self, axis, n):
        if self.all_btn.isChecked():
            self.slicer_area.set_bin_all(axis, n)
        else:
            self.slicer_area.set_bin(axis, n)

    def connect_signals(self):
        super().connect_signals()
        self.slicer_area.sigCurrentCursorChanged.connect(self.update)
        self.slicer_area.sigDataChanged.connect(self.update)

    def disconnect_signals(self):
        super().disconnect_signals()
        self.slicer_area.sigCurrentCursorChanged.disconnect(self.update)
        self.slicer_area.sigDataChanged.disconnect(self.update)

    def update(self):
        super().update()

        if len(self.labels) != self.data.ndim:
            clear_layout(self.layout())
            self.initialize_widgets()

        for i in range(self.data.ndim):
            self.spins[i].blockSignals(True)
            self.labels[i].setText(str(self.data.dims[i]))
            self.spins[i].setRange(1, self.data.shape[i] - 1)
            self.spins[i].setValue(self.array_slicer.get_bins(self.current_cursor)[i])
            self.spins[i].blockSignals(False)

    def reset(self):
        for spin in self.spins:
            spin.setValue(1)


if __name__ == "__main__":
    # data = xr.load_dataarray(
    # "~/Documents/ERLab/TiSe2/kxy10.nc"
    # "~/Documents/ERLab/TiSe2/221213_SSRL_BL5-2/fullmap_kconv_.h5"
    # "~/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
    # "~/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy.nc"
    # "~/Documents/ERLab/TiSe2/220410_ALS_BL4/map_mm_4d_.nc",
    # engine="h5netcdf",
    # )
    data = erlab.io.load_als_bl4(
        19, "/Users/khan/Documents/ERLab/TiSe2/220327_ALS_BL4/data"
    )
    win = itool(data)

    # print(win.array_slicer._nanmeancalls)
