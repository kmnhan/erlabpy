import colorsys
import sys

import darkdetect
import numpy as np
import pyqtgraph as pg
import qtawesome as qta
import xarray as xr
from matplotlib import colors as mcolors
from pyqtgraph.dockarea.Dock import Dock, DockLabel
from pyqtgraph.dockarea.DockArea import DockArea
from PySide6 import QtCore, QtGui, QtWidgets

from .colors import pg_colormap_names, pg_colormap_powernorm, pg_colormap_to_QPixmap
from .slicer import SlicerArray

suppressnanwarning = np.testing.suppress_warnings()
suppressnanwarning.filter(RuntimeWarning, r"All-NaN (slice|axis) encountered")


def itool_(data, execute=None, *args, **kwargs):
    qapp = QtWidgets.QApplication.instance()
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
        win.raise_()
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
        self.setAttribute(QtCore.Qt.WA_AcceptTouchEvents, True)
        self.setCentralWidget(self.slicer_area)

        self.setDockOptions(self.AnimatedDocks | self.AllowTabbedDocks)

        dock = QtWidgets.QDockWidget("Controls", self)

        self.controls = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QHBoxLayout(self.controls)
        self.controls_layout.setContentsMargins(3, 3, 3, 3)
        self.controls_layout.setSpacing(3)

        self.groups = []
        self.group_layouts = []
        self.group_widgets = []

        self.add_group()  # title="Info")
        self.add_group()  # title="Color")
        self.add_group()  # title="Bin")

        self.add_widget(0, ItoolCrosshairControls(self.slicer_area))

        self.add_widget(1, ItoolColormapControls(self.slicer_area))

        self.add_widget(2, ItoolBinningControls(self.slicer_area))

        dock.setWidget(self.controls)
        dock.setTitleBarWidget(QtWidgets.QWidget())
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, dock)
        self.resize(720, 720)

        self.keyboard_shortcuts = {
            "R": (
                "Reverse colormap",
                self.group_widgets[1].misc_controls.btn_reverse.click,
            ),
            # "L": ("Lock color levels", self.group_widgets[idx]._cmap_lock_button.click),
            "S": ("Toggle cursor snap", self.group_widgets[0].btn_snap.click),
            "T": ("Transpose main image", self.group_widgets[0].btn_transpose[0].click),
        }
        for k, v in self.keyboard_shortcuts.items():
            sc = QtGui.QShortcut(QtGui.QKeySequence(k), self)
            sc.activated.connect(v[-1])

    def add_widget(self, idx: int, widget: QtWidgets.QWidget):
        self.group_layouts[idx].addWidget(widget)
        self.group_widgets[idx] = widget

    def add_group(self, **kwargs):
        group = QtWidgets.QGroupBox(**kwargs)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 3)
        group_layout.setSpacing(3)
        self.controls_layout.addWidget(group)
        self.groups.append(group)
        self.group_widgets.append(None)
        self.group_layouts.append(group_layout)


class ImageSlicerArea(DockArea):

    sigDataChanged = QtCore.Signal()
    sigCurrentCursorChanged = QtCore.Signal(int)

    def __init__(self, parent=None, data=None, cmap="magma", gamma=0.5, rad2deg=False):
        super().__init__(parent)

        self.qapp = QtCore.QCoreApplication.instance()

        self.colormap_properties = dict(
            cmap=cmap,
            gamma=gamma,
            reversed=False,
            highContrast=False,
            zeroCentered=False,
        )

        self._data = None
        self.current_cursor = 0

        self.addPlotDock(
            name="0",
            size=(30, 30),
            image=True,
            display_axis=(0, 1),
        )
        self.addPlotDock(
            name="1", position="top", relativeTo="0", size=(30, 10), display_axis=(0,)
        )
        self.addPlotDock(
            name="2",
            position="right",
            relativeTo="0",
            size=(10, 30),
            display_axis=(1,),
            is_vertical=True,
        )
        self.addPlotDock(
            name="3",
            position="right",
            relativeTo="1",
            size=(25, 25),
            display_axis=(2,),
        )
        self.addPlotDock(
            name="4",
            position="bottom",
            relativeTo="1",
            size=(30, 15),
            image=True,
            display_axis=(0, 2),
        )
        self.addPlotDock(
            name="5",
            position="left",
            relativeTo="2",
            size=(15, 30),
            image=True,
            display_axis=(2, 1),
        )
        self.addPlotDock(
            name="6", position="top", relativeTo="3", size=(0, 0), display_axis=(3,)
        )

        self._container_bottom = self.getLargeContainer(self.get_dock(0))
        self._container_top = self.getLargeContainer(self.get_dock(1))

        self._container_bottom.splitterMoved.connect(
            lambda: self.sync_splitters(self._container_top, self._container_bottom)
        )
        self._container_top.splitterMoved.connect(
            lambda: self.sync_splitters(self._container_bottom, self._container_top)
        )

        if data is not None:
            self.set_data(data, rad2deg=rad2deg)

    def connect_signals(self):
        self.sigIndexChanged.connect(self.refresh_plots)
        self.sigBinChanged.connect(lambda c, _: self.refresh_plots(c))
        self.sigDataChanged.connect(self.refresh_all)
        self.sigCursorCountChanged.connect(lambda: self.set_colormap(update=True))

    @property
    def sigCursorCountChanged(self) -> type[QtCore.Signal]:
        return self.data_slicer.sigCursorCountChanged

    @property
    def sigIndexChanged(self) -> type[QtCore.Signal]:
        return self.data_slicer.sigIndexChanged

    @property
    def sigBinChanged(self) -> type[QtCore.Signal]:
        return self.data_slicer.sigBinChanged

    @property
    def colormap(self):
        return self.colormap_properties["cmap"]

    @property
    def main_image(self):
        """returns the PlotItem of main dock"""
        return self.get_dock(0).axes

    @property
    def slices(self):
        if self.data.ndim == 2:
            return tuple()
        else:
            slice_axes = [4, 5]
        return tuple(self.get_dock(ax).axes for ax in slice_axes)

    @property
    def images(self):
        return (self.main_image,) + self.slices

    @property
    def profiles(self):
        if self.data.ndim == 2:
            profile_axes = [1, 2]
        elif self.data.ndim == 3:
            profile_axes = [1, 2, 3]
        elif self.data.ndim == 4:
            profile_axes = [1, 2, 3, 6]
        return tuple(self.get_dock(ax).axes for ax in profile_axes)

    @property
    def axes(self):
        return self.images + self.profiles

    @property
    def data_slicer(self) -> SlicerArray:
        return self._data_slicer

    @property
    def n_cursors(self) -> int:
        return self.data_slicer.n_cursors

    @property
    def current_indices(self) -> list[int]:
        return self.data_slicer.get_indices(self.current_cursor)

    @property
    def current_values(self) -> list[float]:
        return self.data_slicer.get_values(self.current_cursor)

    @property
    def data(self) -> xr.DataArray:
        return self.data_slicer._obj

    def get_dock(self, index):
        return self.docks[str(index)]

    @QtCore.Slot()
    def refresh_all(self):
        for c in range(self.n_cursors):
            self.refresh_plots(c)

    @QtCore.Slot(tuple)
    def refresh(self, axes: tuple = None):
        self.sigIndexChanged.emit(self.current_cursor, axes)

    def refresh_plots(self, *args, **kwargs):
        for ax in self.axes:
            ax.plotItem.refresh_items_data(*args, **kwargs)
            # TODO: autorange smarter
            ax.plotItem.vb.updateAutoRange()

    def set_current_cursor(self, cursor: int, update=True):
        if cursor > self.n_cursors - 1:
            raise IndexError("Cursor index out of range")
        self.current_cursor = cursor
        if update:
            self.refresh()
        self.sigCurrentCursorChanged.emit(cursor)

    def set_data(self, data: xr.DataArray, rad2deg=None):
        if not rad2deg:
            self._data = data
        else:
            if np.iterable(rad2deg):
                conv_dims = rad2deg
            else:
                conv_dims = [d for d in ("phi", "theta", "beta", "alpha", "chi") if d in data.dims]
            self._data = data.assign_coords({d: np.rad2deg(data[d]) for d in conv_dims})
        self._data_slicer = SlicerArray(self._data)

        self.connect_signals()

        self.adjust_layout()

        if self.current_cursor > self.n_cursors - 1:
            self.set_current_cursor(self.n_cursors - 1, update=False)
        self.sigDataChanged.emit()

        self.refresh()
        self.set_colormap(update=True)

    @QtCore.Slot(int, int)
    def swap_axes(self, ax1: int, ax2: int):
        self.data_slicer.swap_axes(ax1, ax2)
        self.sigDataChanged.emit()

    def _ax_display(self, axis):
        axes = list(range(self.data.ndim))
        axes.remove(axis)
        return axes

    @QtCore.Slot(int, int, bool)
    def set_index(self, axis: int, value: int, update: bool = True):
        self.data_slicer.set_index(self.current_cursor, axis, value, update)

    @QtCore.Slot(int, float, bool)
    def set_value(self, axis: int, value: float, update: bool = True):
        self.data_slicer.set_value(self.current_cursor, axis, value, update)

    @QtCore.Slot(int, int, bool)
    def set_bin(self, axis: int, value: int, update: bool = True):
        new_bins = [None] * self.data.ndim
        new_bins[axis] = value
        self.data_slicer.set_bins(self.current_cursor, new_bins, update)

    @QtCore.Slot()
    def add_cursor(self):
        self.data_slicer.add_cursor(self.current_cursor, update=False)
        self.current_cursor = self.n_cursors - 1
        for ax in self.axes:
            ax.add_cursor(update=False)
        self.refresh()
        self.sigCursorCountChanged.emit(self.n_cursors)
        self.sigCurrentCursorChanged.emit(self.current_cursor)

    @QtCore.Slot(int)
    def remove_cursor(self, index: int):
        self.data_slicer.remove_cursor(index, update=False)
        if self.current_cursor == index:
            if index == 0:
                self.current_cursor = 1
            self.current_cursor -= 1
        for ax in self.axes:
            ax.remove_cursor(index)
        self.refresh()
        self.sigCursorCountChanged.emit(self.n_cursors)
        self.sigCurrentCursorChanged.emit(self.current_cursor)

    def cursor_color(self, index: int):
        colors = [
            pg.mkColor(0.8),
            pg.mkColor("y"),
            pg.mkColor("m"),
            pg.mkColor("c"),
            pg.mkColor("g"),
            pg.mkColor("r"),
        ]
        return colors[index % len(colors)]

    def set_colormap(
        self,
        cmap=None,
        gamma: float = None,
        reversed: bool = None,
        highContrast: bool = None,
        zeroCentered: bool = None,
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
        for ax in self.images:
            for im in ax.plotItem.slicer_data_items:
                im.set_pg_colormap(cmap, update=update)

    def adjust_layout(self, horiz_pad=45, vert_pad=30, font_size=11.0):
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))

        invalid = []
        if self.data.ndim == 2:
            invalid = [3, 4, 5, 6]
        elif self.data.ndim == 3:
            invalid = [6]
        for i in range(7):
            self.get_dock(i).setVisible(i not in invalid)

        # parameters for layout: stretch and axis on/off
        stretch = [
            (30, 30),
            (30, 10),
            (10, 30),
            (25, 15) if self.data.ndim == 4 else (25, 25),
            (30, 15),
            (15, 30),
            (25, 10) if self.data.ndim == 4 else (0, 0),
        ]
        if self.data.ndim == 2:
            stretch[3] = (10, 10)

        valid_axis = (
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 1, 0) if self.data.ndim == 4 else (0, 1, 1, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 1, 1, 0),
        )
        for i, sel in enumerate(valid_axis):
            dock = self.get_dock(i)
            dock.axes.plotItem.setDefaultPadding(0)
            dock.setStretch(*stretch[i])
            for axis in ["left", "bottom", "right", "top"]:
                dock.axes.plotItem.getAxis(axis).setTickFont(font)
                dock.axes.plotItem.getAxis(axis).setStyle(
                    autoExpandTextSpace=True, autoReduceTextSpace=True
                )
            dock.axes.plotItem.showAxes(sel, showValues=sel, size=(horiz_pad, vert_pad))
            if i in [1, 4]:
                dock.axes.plotItem.setXLink(self.get_dock(0).axes.plotItem)
            elif i in [2, 5]:
                dock.axes.plotItem.setYLink(self.get_dock(0).axes.plotItem)

    def toggle_snap(self, value: bool = None):
        if value is None:
            value = ~self.data_slicer.snap_to_data
        self.data_slicer.snap_to_data = value

    def sync_splitters(self, c0, c1):
        self.get_dock(0).blockSignals(True)
        self.get_dock(1).blockSignals(True)

        sizes = c0.sizes()
        total = sum(sizes)
        sizes[0] = c1.sizes()[0]
        if all([x == 0 for x in sizes[1:]]) and sizes[0] != total:
            sizes[1:] = [1] * len(sizes[1:])
        try:
            factor = (total - sizes[0]) / sum(sizes[1:])
        except ZeroDivisionError:
            factor = 0
        for k in range(1, len(sizes)):
            sizes[k] *= factor
        c0.setSizes(sizes)

        self.get_dock(0).blockSignals(False)
        self.get_dock(1).blockSignals(False)

    def getLargeContainer(self, obj):
        container = obj
        while container is not self.topContainer:
            container_prev = container
            container = self.getContainer(container)
        return container_prev

    def addControlDock(self, position="bottom", relativeTo=None, **kwds):
        dock = ItoolControlDock(**kwds)
        return super().addDock(dock, position, relativeTo, **kwds)

    def addPlotDock(self, position="bottom", relativeTo=None, **kwds):
        dock = ItoolPlotDock(slicer_area=self, **kwds)
        return super().addDock(dock, position, relativeTo, **kwds)

    def changeEvent(self, evt):
        if evt.type() == QtCore.QEvent.PaletteChange:
            self.qapp.setStyle(self.qapp.style().name())
        super().changeEvent(evt)


class ItoolCursorLine(pg.InfiniteLine):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.qapp = QtCore.QCoreApplication.instance()

    def setBounds(self, bounds, value: float = None):
        if bounds[0] > bounds[1]:
            bounds = list(bounds)
            bounds.reverse()
        self.maxRange = bounds
        if value is None:
            value = self.value()
        self.setValue(value)

    def mouseDragEvent(self, ev):
        if self.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            super().mouseDragEvent(ev)
        else:
            self.setMouseHover(False)
            ev.ignore()

    def mouseClickEvent(self, ev):
        if self.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            super().mouseClickEvent(ev)
        else:
            self.setMouseHover(False)
            ev.ignore()

    def hoverEvent(self, ev):
        if self.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            super().hoverEvent(ev)
        else:
            self.setMouseHover(False)


class ItoolCursorSpan(pg.LinearRegionItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def setSpan(self, rgn):
        # setRegion but hides when region width is 0
        if rgn[1] == rgn[0]:
            self.setVisible(False)
        else:
            self.setVisible(True)
            self.setRegion(rgn)


class ItoolDisplayObject(object):
    def __init__(self, axes, cursor: int = None):
        super().__init__()
        self.axes = axes
        if cursor is None:
            cursor = 0
        self._cursor_index = int(cursor)

    @property
    def display_axis(self):
        return self.axes.display_axis

    @property
    def data_slicer(self):
        return self.axes.data_slicer

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
        cursor: int = None,
        is_vertical: bool = False,
        *args,
        **kargs,
    ):
        pg.PlotDataItem.__init__(self, *args, **kargs)
        ItoolDisplayObject.__init__(self, axes, cursor)
        self.is_vertical = is_vertical

    def refresh_data(self, **kwargs):
        ItoolDisplayObject.refresh_data(self)
        coord, vals = self.data_slicer.slice_with_coord(
            self.cursor_index, self.display_axis
        )
        if self.is_vertical:
            self.setData(vals, coord, **kwargs)
        else:
            self.setData(coord, vals, **kwargs)


class ItoolImageItem(pg.ImageItem, ItoolDisplayObject):
    def __init__(
        self,
        axes,
        cursor: int = None,
        *args,
        **kargs,
    ):
        pg.ImageItem.__init__(self, *args, **kargs)
        ItoolDisplayObject.__init__(self, axes, cursor)

    @suppressnanwarning
    def refresh_data(self, **kwargs):
        ItoolDisplayObject.refresh_data(self)
        rect, img = self.data_slicer.slice_with_coord(
            self.cursor_index, self.display_axis
        )
        self.setImage(image=img, rect=rect, **kwargs)

    def set_colormap(
        self,
        cmap,
        gamma,
        reverse=False,
        highContrast=False,
        zeroCentered=False,
        update=True,
    ):
        cmap = pg_colormap_powernorm(
            cmap,
            gamma,
            reverse,
            highContrast=highContrast,
            zeroCentered=zeroCentered,
        )
        self.set_pg_colormap(cmap, update=update)

    def set_pg_colormap(self, cmap: pg.ColorMap, update=True):
        self._colorMap = cmap
        self.setLookupTable(cmap.getStops()[1], update=update)


class ItoolDockLabel(DockLabel):
    def __init__(self, *args, color="#591e71", **kwargs):
        self.bg_color = mcolors.to_hex(color)
        super().__init__(*args, **kwargs)

    def dim_color(self, color, l_factor=1.0, s_factor=1.0):
        h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
        return QtGui.QColor.fromRgbF(
            *colorsys.hls_to_rgb(h, min(1, l * l_factor), min(1, s * s_factor))
        ).name()

    def set_fg_color(self):
        rgb = list(mcolors.to_rgb(self.bg_color))
        L = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
        if L > 0.729:
            self.fg_color = "#000000"
        else:
            self.fg_color = "#ffffff"

    def updateStyle(self):
        r = "3px"
        self.set_fg_color()
        if self.dim:
            if darkdetect.isDark():
                self.dim_l_factor = 0.8
            else:
                self.dim_l_factor = 1.25
            fg = self.dim_color(self.fg_color, self.dim_l_factor, 0.8)
            bg = self.dim_color(self.bg_color, self.dim_l_factor, 0.8)
        else:
            fg = self.fg_color
            bg = self.bg_color
        border = self.dim_color(bg, 0.9)

        if self.orientation == "vertical":
            self.vStyle = f"""DockLabel {{
                background-color : {bg};
                color : {fg};
                border-top-right-radius: 0px;
                border-top-left-radius: {r};
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: {r};
                border-width: 0px;
                border-right: 2px solid {border};
                padding-top: 3px;
                padding-bottom: 3px;
                font-size: {self.fontSize};
            }}"""
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = f"""DockLabel {{
                background-color : {bg};
                color : {fg};
                border-top-right-radius: {r};
                border-top-left-radius: {r};
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: 0px;
                border-width: 0px;
                border-bottom: 2px solid {border};
                padding-left: 3px;
                padding-right: 3px;
                font-size: {self.fontSize};
            }}"""
            self.setStyleSheet(self.hStyle)


class ItoolControlDock(Dock):
    def __init__(
        self,
        name,
        area=None,
        size=(10, 0),
        widget=None,
        hideTitle=False,
        autoOrientation=True,
        closable=False,
        fontSize="13px",
        color="#591e71",
    ):
        super().__init__(
            name,
            area=area,
            size=size,
            widget=widget,
            hideTitle=hideTitle,
            autoOrientation=autoOrientation,
            closable=closable,
            fontSize=fontSize,
        )
        self.label.setVisible(False)
        self.label = ItoolDockLabel(name, closable, fontSize, color=color)
        if closable:
            self.label.sigCloseClicked.connect(self.close)
        self.topLayout.addWidget(self.label, 0, 1)
        self.topLayout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum
        )

    def changeEvent(self, evt):
        if evt.type() == QtCore.QEvent.PaletteChange:
            self.label.updateStyle()
        super().changeEvent(evt)


class ItoolPlotDock(Dock):
    def __init__(
        self,
        slicer_area,
        name,
        display_axis,
        area=None,
        size=(10, 10),
        image=False,
        hideTitle=True,
        autoOrientation=False,
        closable=False,
        fontSize="13px",
        color="#591e71",
        **plot_kw,
    ):
        self.slicer_area = slicer_area
        self.axes = ItoolGraphicsLayoutWidget(
            slicer_area, display_axis, image=image, **plot_kw
        )

        super().__init__(
            str(name),
            area=area,
            size=size,
            widget=self.axes,
            hideTitle=hideTitle,
            autoOrientation=autoOrientation,
            closable=closable,
            fontSize=fontSize,
        )
        self.label = ItoolDockLabel(str(name), closable, fontSize, color=color)
        if closable:
            self.label.sigCloseClicked.connect(self.close)
        self.topLayout.addWidget(self.label, 0, 1)
        self.topLayout.setContentsMargins(0, 0, 0, 0)
        if hideTitle:
            self.hideTitleBar()

    def startDrag(self):
        pass

    def float(self):
        pass

    def changeEvent(self, evt):
        if evt.type() == QtCore.QEvent.PaletteChange:
            self.label.updateStyle()
        super().changeEvent(evt)


class ItoolPlotItem(pg.PlotItem):
    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        display_axis: tuple,
        image: bool = False,
        **item_kw,
    ):
        super().__init__()

        for action in self.ctrlMenu.actions():
            if action.text() in [
                "Transforms",
                "Downsample",
                "Average",
                "Alpha",
                "Points",
            ]:
                action.setVisible(False)

        for i in (0, 1):
            self.vb.menu.ctrl[i].linkCombo.setVisible(False)
            self.vb.menu.ctrl[i].label.setVisible(False)
        self.vb.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self._slicer_area = None
        self._display_axis = None
        self.slicer_area = slicer_area
        self.display_axis = display_axis

        self.is_image = image
        self._item_kw = item_kw
        self.slicer_data_items = []
        self.cursor_lines = []
        self.cursor_spans = []
        self.add_cursor(update=False)

    def mouseDragEvent(self, evt):
        if (
            self.slicer_area.qapp.queryKeyboardModifiers() == QtCore.Qt.ControlModifier
            and evt.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            evt.accept()
            self.handle_mouse(evt)
        else:
            evt.ignore()

    def handle_mouse(self, evt):
        if self.slicer_area.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            evt.ignore()
            return
        data_pos = self.vb.mapSceneToView(evt.scenePos())
        data_pos_coords = (data_pos.x(), data_pos.y())
        if not self.is_image:
            if self.slicer_data_items[-1].is_vertical:
                data_pos_coords = (data_pos.y(), data_pos.x())

        for i, ax in enumerate(self.display_axis):
            self.slicer_area.set_value(ax, data_pos_coords[i], update=False)
        self.slicer_area.refresh(self.display_axis)

    def add_cursor(self, update=True):
        new_cursor = len(self.slicer_data_items)
        line_angles = (90, 0)
        clr = self.slicer_area.cursor_color(new_cursor)
        clr_cursor = pg.mkColor(clr)
        clr_cursor_hover = pg.mkColor(clr)
        clr_span = pg.mkColor(clr)
        clr_span_edge = pg.mkColor(clr)

        clr_cursor.setAlphaF(0.75)
        clr_cursor_hover.setAlphaF(0.95)
        clr_span.setAlphaF(0.15)
        clr_span_edge.setAlphaF(0.35)

        if self.is_image:
            item = ItoolImageItem(
                self,
                cursor=new_cursor,
                autoDownsample=True,
                axisOrder="row-major",
                **self._item_kw,
            )
        else:
            item = ItoolPlotDataItem(
                self,
                cursor=new_cursor,
                pen=pg.mkPen(pg.mkColor(clr)),
                defaultPadding=0.0,
                clipToView=False,
                connect="finite",
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
                lambda v, cursor=new_cursor, axis=ax: self.line_drag(
                    v.value(), cursor, axis
                )
            )
            c.sigClicked.connect(lambda *_, cursor=new_cursor: self.line_click(cursor))

        if update:
            self.refresh_cursor(new_cursor)

    def line_click(self, cursor):
        if cursor != self.slicer_area.current_cursor:
            self.slicer_area.set_current_cursor(cursor, update=True)

    def line_drag(self, value, cursor, axis):
        if cursor != self.slicer_area.current_cursor:
            self.slicer_area.set_current_cursor(cursor, update=True)
        self.data_slicer.set_value(cursor, axis, value, update=True)

    def remove_cursor(self, index: int):
        self.removeItem(self.slicer_data_items.pop(index))
        for line, span in zip(
            self.cursor_lines.pop(index).values(), self.cursor_spans.pop(index).values()
        ):
            self.removeItem(line)
            self.removeItem(span)
        for i, item in enumerate(self.slicer_data_items):
            item.cursor_index = i

    def refresh_cursor(self, cursor):
        for ax, line in self.cursor_lines[cursor].items():
            line.setBounds(
                self.data_slicer.lims[ax], self.data_slicer.get_values(cursor)[ax]
            )
            self.cursor_spans[cursor][ax].setSpan(
                self.data_slicer.span_bounds(cursor, ax)
            )

    def refresh_items_data(self, cursor, axes=None):
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

    def set_active_cursor(self, index):
        if self.is_image:
            for i, (item, cursors) in enumerate(
                zip(self.slicer_data_items, self.cursor_lines)
            ):
                item.setVisible(i == index)
                for line in cursors.values():
                    # line.setMovable(i == index)
                    pass
        else:
            for i, cursors in enumerate(self.cursor_lines):
                for line in cursors.values():
                    # line.setMovable(i == index)
                    pass

    @property
    def display_axis(self):
        return self._display_axis

    @display_axis.setter
    def display_axis(self, value: tuple):
        self._display_axis = value

    @property
    def slicer_area(self):
        return self._slicer_area

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea):
        self._slicer_area = value

    @property
    def data_slicer(self):
        return self.slicer_area.data_slicer


# Unsure of whether to subclass GraphicsLayoutWidget or PlotWidget at the moment
# Will need to run some benchmarks in the future

# class ItoolGraphicsLayoutWidget(pg.GraphicsLayoutWidget):
class ItoolGraphicsLayoutWidget(pg.PlotWidget):
    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        display_axis: tuple,
        image: bool = False,
        parent=None,
        **item_kw,
    ):
        # super().__init__(parent=parent)
        # self.ci.layout.setSpacing(0)
        # self.ci.layout.setContentsMargins(0, 0, 0, 0)
        # self.plotItem = ItoolPlotItem(slicer_area, display_axis, image, **item_kw)
        # self.addItem(self.plotItem)

        super().__init__(
            parent=parent,
            plotItem=ItoolPlotItem(slicer_area, display_axis, image, **item_kw),
        )

        self.scene().sigMouseClicked.connect(self.plotItem.handle_mouse)


class IconButton(QtWidgets.QPushButton):

    ICON_NAME = dict(
        invert="mdi6.invert-colors",
        invert_off="mdi6.invert-colors-off",
        contrast="mdi6.contrast-box",
        lock="mdi6.lock",
        unlock="mdi6.lock-open-variant",
        colorbar="mdi6.gradient-vertical",
        transpose_0="mdi6.arrow-top-left-bottom-right",
        transpose_1="mdi6.arrow-up-down",
        transpose_2="mdi6.arrow-left-right",
        transpose_3="mdi6.arrow-up-down",
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
    )

    def __init__(self, on: str = None, off: str = None, **kwargs):
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

    def get_icon(self, icon: str):
        try:
            return qta.icon(self.ICON_NAME[icon])
        except KeyError:
            return qta.icon(icon)

    def refresh_icons(self):
        if self.icon_key_off is not None:
            if self.isChecked():
                self.setIcon(self.get_icon(self.icon_key_off))
                return
        self.setIcon(self.get_icon(self.icon_key_on))

    def changeEvent(self, evt):  # handles dark mode
        if evt.type() == QtCore.QEvent.PaletteChange:
            qta.reset_cache()
            self.refresh_icons()
        super().changeEvent(evt)


class BetterSpinBox(QtWidgets.QAbstractSpinBox):
    valueChanged = QtCore.Signal(object)
    textChanged = QtCore.Signal(object)

    def __init__(
        self,
        *args,
        integer=False,
        compact=True,
        discrete=False,
        decimals=3,
        significant=False,
        scientific=False,
        value=0.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        integer : boolean, optional
            If `True`, the spinbox will only display integer values.
        compact : boolean, optional
            Whether to reduce the height of the spinbox.
        discrete : boolean, optional
            If `True` the spinbox will only step to pre-determined discrete values.
            If `False`, the spinbox will just add or subtract the predetermined
            increment when increasing or decreasing the step.
        scientific : boolean, optional
            Whether to print in scientific notation.
        decimals : int, optional
            The precision of the spinbox. See the `significant` argument for the
            meaning. When `int` is `True`, this argument is ignored.
        significant : boolean, optional
            If `True`, `decimals` will specify the total number of significant digits,
            before or after the decimal point, ignoring leading zeros.
            If `False`, `decimals` will specify the total number of digits after the
            decimal point, including leading zeros.
            When `int` or `scientific` is `True`, this argument is ignored.
        value : float, optional
            Initial value of the spinbox.
        """

        self._only_int = integer
        self._is_compact = compact
        self._is_discrete = discrete
        self._is_scientific = scientific
        self._decimal_significant = significant
        self.setDecimals(decimals)

        self._value = value
        self._min = -np.inf
        self._max = np.inf
        self._step = 1 if self._only_int else 0.01
        super().__init__(*args, **kwargs)
        # self.editingFinished.disconnect()
        # self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.editingFinished.connect(self.editingFinishedEvent)
        self._updateHeight()
        if self.isReadOnly():
            self.lineEdit().setReadOnly(True)
            self.setButtonSymbols(self.NoButtons)

    def setDecimals(self, decimals):
        self._decimals = decimals

    def decimals(self):
        return self._decimals

    def setRange(self, mn, mx):
        self.setMinimum(min(mn, mx))
        self.setMaximum(max(mn, mx))

    def widthFromText(self, text):
        return QtGui.QFontMetrics(self.lineEdit().font()).boundingRect(text).width()

    def widthFromValue(self, value):
        return self.widthFromText(self.textFromValue(value))

    def sizeHint(self):
        # !TODO: incorporate width of margin and buttons
        return QtCore.QSize(
            max(
                self.widthFromValue(self.maximum()), self.widthFromValue(self.minimum())
            ),
            0,
        )

    def setMaximum(self, mx):
        if self._only_int and np.isfinite(mx):
            mx = round(mx)
        elif np.isnan(mx):
            mx = np.inf
        self._max = mx

    def setMinimum(self, mn):
        if self._only_int and np.isfinite(mn):
            mn = round(mn)
        elif np.isnan(mn):
            mn = -np.inf
        self._min = mn

    def setSingleStep(self, step):
        if self._only_int:
            step = round(step)
        self._step = abs(step)

    def singleStep(self):
        return self._step

    def maximum(self):
        return self._max

    def minimum(self):
        return self._min

    def value(self):
        if self._only_int:
            return int(self._value)
        else:
            return self._value

    def text(self):
        return self.textFromValue(self.value())

    def textFromValue(self, value):
        if (not self._only_int) or (not np.isfinite(value)):
            if self._is_scientific:
                return np.format_float_scientific(
                    value,
                    precision=self.decimals(),
                    unique=False,
                    trim="k",
                    exp_digits=1,
                )
            else:
                return np.format_float_positional(
                    value,
                    precision=self.decimals(),
                    unique=False,
                    fractional=not self._decimal_significant,
                    trim="k",
                )
        else:
            return str(int(value))

    def valueFromText(self, text):
        if text == "":
            return np.nan
        if self._only_int:
            return int(text)
        else:
            return float(text)

    def stepBy(self, steps):
        inc = self.singleStep()
        if (
            all(np.isfinite([self.maximum(), self.minimum(), self.value()]))
            and self._is_discrete
        ):
            self.setValue(
                self.minimum()
                + inc
                * max(
                    min(
                        round((self.value() + steps * inc - self.minimum()) / inc),
                        int((self.maximum() - self.minimum()) / inc),
                    ),
                    0,
                )
            )
        else:
            if steps > 0:
                self.setValue(min(inc * steps + self.value(), self.maximum()))
            else:
                self.setValue(max(inc * steps + self.value(), self.minimum()))

    def stepEnabled(self):
        if self.isReadOnly():
            return self.StepNone
        if self.wrapping():
            return self.StepDownEnabled | self.StepUpEnabled
        if self.value() < self.maximum():
            if self.value() > self.minimum():
                return self.StepDownEnabled | self.StepUpEnabled
            else:
                return self.StepUpEnabled
        elif self.value() > self.minimum():
            return self.StepDownEnabled
        else:
            return self.StepNone

    def setValue(self, val):
        if np.isnan(val):
            self._value = np.nan
        else:
            self._value = max(self.minimum(), min(val, self.maximum()))

        if self._only_int and np.isfinite(self._value):
            self._value = round(self._value)

        self.valueChanged.emit(self.value())
        self.lineEdit().setText(self.text())
        self.textChanged.emit(self.text())

    # def fixup(self, input):
    #     # fixup is called when the spinbox loses focus with an invalid or intermediate string
    #     self.lineEdit().setText(self.text())

    #     # support both PyQt APIs (for Python 2 and 3 respectively)
    #     # http://pyqt.sourceforge.net/Docs/PyQt4/python_v3.html#qvalidator

    #     print(input)
    #     try:
    #         input.clear()
    #         input.append(self.lineEdit().text())
    #     except AttributeError:
    #         return self.lineEdit().text()

    # # def hasAcceptableInput(self) -> bool:
    # #     return True

    def validate(self, strn, pos):
        # if self.skipValidate:
        if False:
            ret = QtGui.QValidator.State.Acceptable
        else:
            ret = QtGui.QValidator.State.Intermediate
            try:
                val = float(self.value())
                if val < self.maximum() and val > self.minimum():
                    ret = QtGui.QValidator.State.Acceptable
            except ValueError:
                pass

        ## note: if text is invalid, we don't change the textValid flag
        ## since the text will be forced to its previous state anyway
        self.update()

        ## support 2 different pyqt APIs. Bleh.

        # print(strn, pos, ret)
        if hasattr(QtCore, "QString"):
            return (ret, pos)
        else:
            return (ret, strn, pos)

    def editingFinishedEvent(self):
        self.setValue(self.valueFromText(self.lineEdit().text()))

    def _updateHeight(self):
        if self._is_compact:
            self.setMaximumHeight(QtGui.QFontMetrics(self.font()).height() + 3)


class ColorMapComboBox(QtWidgets.QComboBox):

    LOAD_ALL_TEXT = "Load all..."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPlaceholderText("Select colormap...")
        self.setToolTip("Colormap")
        w, h = 64, 16
        self.setIconSize(QtCore.QSize(w, h))
        for name in pg_colormap_names("mpl"):
            # for name in pg_colormap_names("local"):
            self.addItem(name)
        self.insertItem(0, self.LOAD_ALL_TEXT)
        self.thumbnails_loaded = False
        self.currentIndexChanged.connect(self.load_thumbnail)
        self.default_cmap = None

    def load_thumbnail(self, index):
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
        maxWidth = max([fm.width(self.itemText(i)) for i in range(self.count())])
        if maxWidth:
            view.setMinimumWidth(maxWidth)

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

    def __init__(self, value=1.0):
        super().__init__()
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(3)

        self.spin = BetterSpinBox(
            self,
            toolTip="Colormap gamma",
            decimals=2,
            wrapping=False,
            keyboardTracking=False,
            singleStep=0.01,
        )
        self.spin.setRange(0.01, 99.99)
        self.spin.setValue(value)
        self.label = QtWidgets.QLabel("γ", buddy=self.spin)
        # self.label.setIndent(0)
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        self.slider = QtWidgets.QSlider(
            self,
            toolTip="Colormap gamma",
            value=self.gamma_scale(value),
            orientation=QtCore.Qt.Horizontal,
        )
        self.slider.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed
        )
        self.slider.setMinimumWidth(30)
        self.spin.valueChanged.connect(self.spin_changed)

        self.slider.setRange(
            self.gamma_scale(self.spin.minimum()),
            self.gamma_scale(self.spin.maximum()),
        )
        self.slider.valueChanged.connect(self.slider_changed)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spin)
        self.layout.addWidget(self.slider)

    def setValue(self, value):
        self.spin.setValue(value)
        self.slider.setValue(value)

    def spin_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(self.gamma_scale(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def slider_changed(self, value):
        self.spin.setValue(self.gamma_scale_inv(value))

    def gamma_scale(self, y):
        return 1e3 * np.log10(y)

    def gamma_scale_inv(self, x):
        return np.power(10, x * 1e-3)


def clear_layout(layout: QtWidgets.QLayout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


class ItoolControlsBase(QtWidgets.QWidget):
    def __init__(self, slicer_area: ImageSlicerArea, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slicer_area = slicer_area
        self.sub_controls = []
        self.initialize_layout()
        self.initialize_widgets()
        self.connect_signals()
        self.update()

    @property
    def data(self):
        return self.slicer_area.data

    @property
    def data_slicer(self):
        return self.slicer_area.data_slicer

    @property
    def n_cursors(self):
        return self.slicer_area.n_cursors

    @property
    def current_cursor(self):
        return self.slicer_area.current_cursor

    def initialize_layout(self):
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(3)

    def initialize_widgets(self):
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.initialize_widgets()

    def connect_signals(self):
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.connect_signals()

    def disconnect_signals(self):
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.disconnect_signals()

    def update(self):
        for ctrl in self.sub_controls:
            ctrl.update()

    def add_control(self, widget):
        self.sub_controls.append(widget)
        return widget

    @property
    def is_nested(self):
        return isinstance(self._slicer_area, ItoolControlsBase)

    @property
    def slicer_area(self):
        if self.is_nested:
            return self._slicer_area.slicer_area
        else:
            return self._slicer_area

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea):
        self.disconnect_signals()
        self._slicer_area = value
        clear_layout(self.layout)
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
        self.btn_reverse.toggled.connect(self.update_colormap)
        self.btn_contrast.toggled.connect(self.update_colormap)
        self.btn_zero.toggled.connect(self.update_colormap)

        self.btn_lock = IconButton(
            on="unlock",
            off="lock",
            checkable=True,
            toolTip="Lock colors",
        )

        self.layout.addWidget(self.btn_reverse)
        self.layout.addWidget(self.btn_contrast)
        self.layout.addWidget(self.btn_zero)
        self.layout.addWidget(self.btn_lock)

    def update(self):
        self.btn_reverse.blockSignals(True)
        self.btn_contrast.blockSignals(True)
        self.btn_zero.blockSignals(True)

        self.btn_reverse.setChecked(self.slicer_area.colormap_properties["reversed"])
        self.btn_contrast.setChecked(
            self.slicer_area.colormap_properties["highContrast"]
        )
        self.btn_zero.setChecked(self.slicer_area.colormap_properties["zeroCentered"])

        self.btn_reverse.blockSignals(False)
        self.btn_contrast.blockSignals(False)
        self.btn_zero.blockSignals(False)

    def update_colormap(self):
        self.slicer_area.set_colormap(
            reversed=self.btn_reverse.isChecked(),
            highContrast=self.btn_contrast.isChecked(),
            zeroCentered=self.btn_zero.isChecked(),
        )


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
    def __init__(self, *args, orientation=QtCore.Qt.Vertical, **kwargs):
        if isinstance(orientation, QtCore.Qt.Orientation):
            self.orientation = orientation
        elif orientation == "vertical":
            self.orientation = QtCore.Qt.Vertical
        elif orientation == "horizontal":
            self.orientation = QtCore.Qt.Horizontal
        super().__init__(*args, **kwargs)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred
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
        self.btn_add.clicked.connect(self.addCursor)

        self.btn_rem = IconButton("minus", toolTip="Remove cursor")
        self.btn_rem.clicked.connect(self.remCursor)

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
        self.cb_cursors.addItems([self._cursor_name(i) for i in range(self.n_cursors)])
        if self.n_cursors == 1:
            # can't remove more cursors
            self.cb_cursors.setDisabled(True)
            self.btn_rem.setDisabled(True)

        # current value widget
        self.spin_dat = BetterSpinBox(
            self.values_groups[-1], discrete=False, scientific=True, readOnly=True
        )
        self.spin_dat.setDecimals(
            round(abs(np.log10(self.data_slicer.absnanmax())) + 1)
        )

        # add multicursor widgets
        if self.orientation == QtCore.Qt.Vertical:
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
        self.layout.addWidget(self.values_groups[0])

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
        if self.data.ndim >= 4:
            self.btn_transpose = tuple(
                IconButton(on=f"transpose_2") for _ in range(self.data.ndim)
            )
        else:
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
                lambda val, axis=i: self.slicer_area.set_value(axis, val)
            )
            self.btn_transpose[i].clicked.connect(
                lambda ax1=i, ax2=(i + 1) % self.data.ndim: self.slicer_area.swap_axes(
                    ax1, ax2
                )
            )
            if self.orientation == QtCore.Qt.Vertical:
                self.values_layouts[i + 1].addWidget(self.label_dim[i], 0, 0, 1, 1)
                self.values_layouts[i + 1].addWidget(self.btn_transpose[i], 0, 1, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_idx[i], 1, 0, 1, 2)
                self.values_layouts[i + 1].addWidget(self.spin_val[i], 2, 0, 1, 2)
            else:
                self.values_layouts[i + 1].addWidget(self.label_dim[i], 0, 0, 1, 1)
                self.values_layouts[i + 1].addWidget(self.btn_transpose[i], 0, 1, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_idx[i], 0, 2, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_val[i], 0, 3, 1, 1)

            self.layout.addWidget(self.values_groups[i + 1])

    def connect_signals(self):
        super().connect_signals()
        self.slicer_area.sigCurrentCursorChanged.connect(self.cursorChangeEvent)
        self.slicer_area.sigIndexChanged.connect(self.update_spins)
        self.slicer_area.sigDataChanged.connect(self.update)

    def disconnect_signals(self):
        super().disconnect_signals()
        self.slicer_area.sigCurrentCursorChanged.disconnect(self.cursorChangeEvent)
        self.slicer_area.sigIndexChanged.disconnect(self.update_spins)
        self.slicer_area.sigDataChanged.disconnect(self.update)

    def update(self):
        super().update()
        if len(self.label_dim) != self.data.ndim:
            # number of required cursors changed, resetting
            clear_layout(self.layout)
            self.initialize_widgets()

        label_width = 0
        for i in range(self.data.ndim):
            self.values_groups[i].blockSignals(True)
            self.spin_idx[i].blockSignals(True)
            self.spin_val[i].blockSignals(True)
            self.label_dim[i].setText(self.data.dims[i])

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
            self.spin_idx[i].setValue(self.slicer_area.current_indices[i])

            self.spin_val[i].setRange(*self.data_slicer.lims[i])
            self.spin_val[i].setSingleStep(self.data_slicer.incs[i])
            self.spin_val[i].setValue(self.slicer_area.current_values[i])

            self.label_dim[i].blockSignals(False)
            self.spin_idx[i].blockSignals(False)
            self.spin_val[i].blockSignals(False)

        for l in self.label_dim:
            # resize buttons to match dimension label length
            l.setMaximumWidth(label_width)

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
        self.spin_dat.setValue(self.data_slicer.current_value(self.current_cursor))

    def _cursor_name(self, i):
        # for cursor combobox content
        return f"Cursor {int(i)}"

    def addCursor(self):
        self.cb_cursors.setDisabled(False)
        self.cb_cursors.addItem(self._cursor_name(self.n_cursors))
        self.slicer_area.add_cursor()
        self.btn_rem.setDisabled(False)

    def remCursor(self):
        self.slicer_area.remove_cursor(self.cb_cursors.currentIndex())
        self.cb_cursors.removeItem(self.cb_cursors.currentIndex())
        for i in range(self.cb_cursors.count()):
            self.cb_cursors.setItemText(i, self._cursor_name(i))
        self.cb_cursors.setCurrentText(self._cursor_name(self.current_cursor))
        if i == 0:
            self.cb_cursors.setDisabled(True)
            self.btn_rem.setDisabled(True)

    def cursorChangeEvent(self, idx):
        self.cb_cursors.setCurrentIndex(idx)
        self.update_spins()

    def setActiveCursor(self, value):
        self.slicer_area.set_current_cursor(self.cb_cursors.findText(value))

    # def index_changed(self, axis, index):
    #     self.spin_val[axis].blockSignals(True)
    #     self.slicer_area.set_index(axis, index)
    #     self.spin_val[axis].setValue(self.slicer_area.current_values[axis])
    #     self.spin_val[axis].blockSignals(False)

    # def value_changed(self, axis, index):
    #     self.spin_idx[axis].blockSignals(True)
    #     self.slicer_area.set_value(axis, index)
    #     self.spin_idx[axis].setValue(self.slicer_area.current_indices[axis])
    #     self.spin_idx[axis].blockSignals(False)


class ItoolColormapControls(ItoolControlsBase):
    def __init__(self, *args, orientation=QtCore.Qt.Vertical, **kwargs):
        if isinstance(orientation, QtCore.Qt.Orientation):
            self.orientation = orientation
        elif orientation == "vertical":
            self.orientation = QtCore.Qt.Vertical
        elif orientation == "horizontal":
            self.orientation = QtCore.Qt.Horizontal
        super().__init__(*args, **kwargs)

    def initialize_layout(self):
        if self.orientation == QtCore.Qt.Vertical:
            self.layout = QtWidgets.QVBoxLayout(self)
        else:
            self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(3)

    def initialize_widgets(self):
        super().initialize_widgets()
        self.cb_colormap = ColorMapComboBox(self, maximumWidth=175)
        self.cb_colormap.textActivated.connect(self.change_colormap)

        self.gamma_widget = ColorMapGammaWidget()
        self.gamma_widget.valueChanged.connect(
            lambda g: self.slicer_area.set_colormap(gamma=g)
        )

        self.misc_controls = self.add_control(ColorControls(self))

        self.layout.addWidget(self.cb_colormap)
        self.layout.addWidget(self.gamma_widget)
        self.layout.addWidget(self.misc_controls)

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

    def initialize_widgets(self):
        super().initialize_widgets()
        self.label = tuple(QtWidgets.QLabel() for _ in range(self.data.ndim))
        self.spins = tuple(
            QtWidgets.QSpinBox(value=1, singleStep=2) for _ in range(self.data.ndim)
        )
        for i, spin in enumerate(self.spins):
            spin.valueChanged.connect(
                lambda n, axis=i: self.slicer_area.set_bin(axis, n)
            )

        self.reset_btn = IconButton("reset")
        self.reset_btn.clicked.connect(self.reset)

        for i in range(self.data.ndim):
            self.layout.addWidget(self.label[i])
            self.layout.addWidget(self.spins[i])
        self.layout.addWidget(self.reset_btn)

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

        if len(self.label) != self.data.ndim:
            clear_layout(self.layout)
            self.initialize_widgets()

        for i in range(self.data.ndim):
            self.spins[i].blockSignals(True)
            self.label[i].setText(self.data.dims[i])
            self.spins[i].setRange(1, self.data.shape[i] - 1)
            self.spins[i].setValue(self.data_slicer.get_bins(self.current_cursor)[i])
            self.spins[i].blockSignals(False)

    def reset(self):
        for spin in self.spins:
            spin.setValue(1)


if __name__ == "__main__":
    data = xr.open_dataarray(
        # "/Users/khan/Documents/ERLab/TiSe2/kxy10.nc"
        "/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
        # "/Users/khan/Documents/ERLab/TiSe2/220410_ALS_BL4/map_mm_4d.nc"
    )
    itool_(data)