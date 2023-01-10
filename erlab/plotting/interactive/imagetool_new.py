import sys

import numpy as np
import pyqtgraph as pg
import qtawesome as qta
import xarray as xr
from PySide6 import QtCore, QtGui, QtWidgets

from .colors import pg_colormap_names, pg_colormap_powernorm, pg_colormap_to_QPixmap
from .slicer import SlicerArray
from .utilities import BetterSpinBox

suppressnanwarning = np.testing.suppress_warnings()
suppressnanwarning.filter(RuntimeWarning, r"All-NaN (slice|axis) encountered")


def _sync_splitters(s0, s1, reverse=False):
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
        factor = 0
    for k in range(1, len(sizes)):
        sizes[k] *= factor
    if reverse:
        sizes = list(reversed(sizes))
    s0.setSizes(sizes)

    s0.blockSignals(False)
    s1.blockSignals(False)


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


class ImageSlicerArea(QtWidgets.QWidget):

    sigDataChanged = QtCore.Signal()
    sigCurrentCursorChanged = QtCore.Signal(int)

    def __init__(self, parent=None, data=None, cmap="magma", gamma=0.5, rad2deg=False):
        super().__init__(parent)

        self.setLayout(QtWidgets.QStackedLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self._splitters = (
            QtWidgets.QSplitter(QtCore.Qt.Vertical),
            QtWidgets.QSplitter(QtCore.Qt.Horizontal),
            QtWidgets.QSplitter(QtCore.Qt.Vertical),
            QtWidgets.QSplitter(QtCore.Qt.Vertical),
            QtWidgets.QSplitter(QtCore.Qt.Horizontal),
            QtWidgets.QSplitter(QtCore.Qt.Vertical),
            QtWidgets.QSplitter(QtCore.Qt.Horizontal),
        )
        self.layout().addWidget(self._splitters[0])
        self._splitters[0].addWidget(self._splitters[1])
        self._splitters[1].addWidget(self._splitters[2])
        self._splitters[1].addWidget(self._splitters[3])
        self._splitters[0].addWidget(self._splitters[4])
        self._splitters[4].addWidget(self._splitters[5])
        self._splitters[4].addWidget(self._splitters[6])
        self._splitters[1].splitterMoved.connect(
            lambda: _sync_splitters(self._splitters[4], self._splitters[1])
        )
        self._splitters[4].splitterMoved.connect(
            lambda: _sync_splitters(self._splitters[1], self._splitters[4])
        )

        self._plots = (
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(0, 1)),
            ItoolGraphicsLayoutWidget(self, display_axis=(0,)),
            ItoolGraphicsLayoutWidget(self, display_axis=(1,), is_vertical=True),
            ItoolGraphicsLayoutWidget(self, display_axis=(2,)),
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(0, 2)),
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(2, 1)),
            ItoolGraphicsLayoutWidget(self, display_axis=(3,)),
        )
        for i in [1, 4]:
            self._splitters[2].addWidget(self._plots[i])
        for i in [6, 3]:
            self._splitters[3].addWidget(self._plots[i])
        self._splitters[5].addWidget(self._plots[0])
        for i in [5, 2]:
            self._splitters[6].addWidget(self._plots[i])

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

        if data is not None:
            self.set_data(data, rad2deg=rad2deg)
        self.set_keyboard_shortcuts()

    def set_keyboard_shortcuts(self):
        self.keyboard_shortcuts = {
            "Ctrl+A": (
                "View all",
                self.view_all,
            ),
        }
        for k, v in self.keyboard_shortcuts.items():
            sc = QtGui.QShortcut(QtGui.QKeySequence(k), self)
            sc.activated.connect(v[-1])

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
        """returns the main PlotItem"""
        return self.get_axes(0)

    @property
    def slices(self):
        if self.data.ndim == 2:
            return tuple()
        else:
            slice_axes = [4, 5]
        return tuple(self.get_axes(ax) for ax in slice_axes)

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
        return tuple(self.get_axes(ax) for ax in profile_axes)

    @property
    def axes(self):
        """Currently valid subset of self._plots"""
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

    def get_axes(self, index):
        return self._plots[index]

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

    def view_all(self):
        for ax in self.axes:
            ax.plotItem.vb.enableAutoRange()
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
                conv_dims = [
                    d
                    for d in ("phi", "theta", "beta", "alpha", "chi")
                    if d in data.dims
                ]
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

    @QtCore.Slot(int, int, bool)
    def set_bin_all(self, axis: int, value: int, update: bool = True):
        new_bins = [None] * self.data.ndim
        new_bins[axis] = value
        for c in range(self.n_cursors):
            self.data_slicer.set_bins(c, new_bins, update)

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

    def adjust_layout(
        self, horiz_pad=45, vert_pad=30, font_size=11.0, r=(1.2, 1.5, 3.0, 1.0)
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

        r01 = r[0] / r[1]
        scale = 100

        # padding due to splitters
        # there must be a smarter way to get this from QStyle
        d = 4 / scale
        sizes = [
            [r[0] + r[1], r[2]],
            [r[3] * r[2], r[3] * (r[0] + r[1])],
            [(r[0] + r[1] - d) * r01, (r[0] + r[1] - d) / r01],
            [(r[0] + r[1] - d) / 2, (r[0] + r[1] - d) / 2],
            [r[3] * r[2], r[3] * (r[0] + r[1])],
            [r[2]],
            [(r[3] * (r[0] + r[1]) - d) / r01, (r[3] * (r[0] + r[1]) - d) * r01],
        ]

        for split, sz in zip(self._splitters, sizes):
            split.setSizes([int(s * scale) for s in sz])

        valid_axis = (
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (0, 0, 1, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 1, 1, 0),
        )

        invalid = []
        if self.data.ndim == 2:
            invalid = [4, 5, 6]
        elif self.data.ndim == 3:
            invalid = [6]

        for i, sel in enumerate(valid_axis):
            axes = self.get_axes(i)
            axes.setVisible(i not in invalid)
            axes.plotItem.setDefaultPadding(0)
            for axis in ["left", "bottom", "right", "top"]:
                axes.plotItem.getAxis(axis).setTickFont(font)
                axes.plotItem.getAxis(axis).setStyle(
                    autoExpandTextSpace=True, autoReduceTextSpace=True
                )
            axes.plotItem.showAxes(sel, showValues=sel, size=(horiz_pad, vert_pad))
            if i in [1, 4]:
                axes.plotItem.setXLink(self.get_axes(0).plotItem)
            elif i in [2, 5]:
                axes.plotItem.setYLink(self.get_axes(0).plotItem)

        if self.data.ndim == 2:
            # reserve space, only hide plotItem
            self.get_axes(3).plotItem.setVisible(False)

    def toggle_snap(self, value: bool = None):
        if value is None:
            value = ~self.data_slicer.snap_to_data
        self.data_slicer.snap_to_data = value

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

        self._slicer_area = slicer_area
        self._display_axis = display_axis

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
        if self.slicer_area.qapp.queryKeyboardModifiers() != QtCore.Qt.AltModifier:
            self.data_slicer.set_value(cursor, axis, value, update=True)
        else:
            for i in range(self.data_slicer.n_cursors):
                self.data_slicer.set_value(i, axis, value, update=True)

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

        if self.is_image:
            label_kw = {
                a: self.slicer_area.data.dims[self.display_axis[i]]
                for a, i in zip(("top", "bottom", "left", "right"), (0, 0, 1, 1))
                if self.getAxis(a).isVisible()
            }
        else:
            if self.slicer_data_items[-1].is_vertical:
                label_kw = {
                    a: self.slicer_area.data.dims[self.display_axis[0]]
                    for a in ("left", "right")
                    if self.getAxis(a).isVisible()
                }
            else:
                label_kw = {
                    a: self.slicer_area.data.dims[self.display_axis[0]]
                    for a in ("top", "bottom")
                    if self.getAxis(a).isVisible()
                }
        self.setLabels(**label_kw)

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

    ICON_ALIASES = dict(
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
        # all_cursors="mdi6.checkbox-multiple-outline",
        all_cursors="mdi6.select-multiple",
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
            return qta.icon(self.ICON_ALIASES[icon])
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

    def initialize_layout(self):
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(3)

    def initialize_widgets(self):
        super().initialize_widgets()
        self.label = tuple(QtWidgets.QLabel() for _ in range(self.data.ndim))
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
            self.layout.addWidget(self.label[i], 0, i, 1, 1)
            self.layout.addWidget(self.spins[i], 1, i, 1, 1)
        self.layout.addWidget(self.reset_btn, 2, 0, 1, 1)
        self.layout.addWidget(self.all_btn, 2, 1, 1, 1)
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
        # "~/Documents/ERLab/TiSe2/kxy10.nc"
        "~/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
        # "~/Documents/ERLab/TiSe2/220410_ALS_BL4/map_mm_4d.nc"
    )
    itool_(data)
