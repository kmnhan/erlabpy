"""Provides core functionality of imagetool."""

from __future__ import annotations

__all__ = ["ImageSlicerArea"]

import collections
import functools
import gc
import inspect
import os
import time
import weakref
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.colors import (
    BetterColorBarItem,
    BetterImageItem,
    pg_colormap_powernorm,
)
from erlab.interactive.imagetool.slicer import ArraySlicer
from erlab.interactive.utilities import BetterAxisItem

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from pyqtgraph.graphicsItems.ViewBox import ViewBoxMenu
    from pyqtgraph.GraphicsScene import mouseEvents

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
    if all(x == 0 for x in sizes[1:]) and sizes[0] != total:
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
        self.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False
        )
        self.scene().sigMouseClicked.connect(self.getPlotItem().mouseDragEvent)

    def getPlotItem(self) -> ItoolPlotItem:
        return self.plotItem

    def getPlotItemViewBox(self) -> pg.ViewBox:
        return self.getPlotItem().vb


def link_slicer(
    func: Callable | None = None,
    *,
    indices: bool = False,
    steps: bool = False,
    color: bool = False,
):
    """
    An internal decorator for choosing which functions to sync accross multiple
    instances of `ImageSlicerArea`.

    Parameters
    ----------
    func
        The method to sync across multiple instances of `ImageSlicerArea`.
    indices
        If `True`, the input argument named `value` given to `func` are interpreted as
        indices, and will be converted to appropriate values for other instances of
        `ImageSlicerArea`. The behavior of this conversion is determined by `steps`.
    steps
        If `False`, considers `value` as an absolute index. If `True`, considers
        `value` as a relative value such as the number of steps or bins. See the
        implementation of `SlicerLinkProxy` for more information.
    color
        Boolean whether the decorated method is related to visualization, such as
        colormap control.

    """

    def my_decorator(func: Callable):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # skip sync if already synced
            skip_sync: bool = kwargs.pop("__slicer_skip_sync", False)

            out = func(*args, **kwargs)
            if args[0].is_linked:
                if not skip_sync:
                    all_args = inspect.Signature.from_callable(func).bind(
                        *args, **kwargs
                    )
                    all_args.apply_defaults()
                    obj: ImageSlicerArea = all_args.arguments.pop("self")
                    obj._linking_proxy.sync(
                        obj, func.__name__, all_args.arguments, indices, steps, color
                    )
            return out

        return wrapped

    if func is not None:
        return my_decorator(func)
    return my_decorator


class SlicerLinkProxy:
    """Internal class for handling linked `ImageSlicerArea` s.

    Parameters
    ----------
    link_colors
        Whether to sync color related changes.
    *slicers
        The slicers to link.

    """

    def __init__(self, *slicers: list[ImageSlicerArea], link_colors: bool = True):
        self.link_colors = link_colors
        self._slicers: set[ImageSlicerArea] = set()
        for s in slicers:
            self.add(s)

    def add(self, slicer: ImageSlicerArea):
        if slicer.is_linked:
            if slicer._linking_proxy == self:
                return
            else:
                raise ValueError("Already linked to another proxy.")
        self._slicers.add(slicer)
        slicer._linking_proxy = self

    def remove(self, slicer: ImageSlicerArea):
        self._slicers.remove(slicer)
        slicer._linking_proxy = None

    def sync(
        self,
        source: ImageSlicerArea,
        funcname: str,
        arguments: dict[str, Any],
        indices: bool,
        steps: bool,
        color: bool,
    ):
        """The core method that propagates changes across multiple `ImageSlicerArea`s.

        This method is invoked every time a method decorated with :func:`link_slicer` in
        a linked `ImageSlicerArea` is called.

        Parameters
        ----------
        source
            Instance of `ImageSlicerArea` corresponding to the called method.
        funcname
            Name of the called method.
        arguments
            Arguments included in the function call.
        indices, steps, color
            Arguments given to the decorator. See :func:`link_slicer`

        """
        if color and not self.link_colors:
            return
        for target in self._slicers.difference({source}):
            getattr(target, funcname)(
                **self.convert_args(source, target, arguments, indices, steps)
            )

    def convert_args(
        self,
        source: ImageSlicerArea,
        target: ImageSlicerArea,
        args: dict[str, Any],
        indices: bool,
        steps: bool,
    ):
        if indices:
            axis: int | None = args.get("axis")
            index: int | None = args.get("value")

            if index is not None:
                if axis is None:
                    args["value"] = [
                        self.convert_index(source, target, a, i, steps)
                        for (a, i) in zip(axis, index)
                    ]
                else:
                    args["value"] = self.convert_index(
                        source, target, axis, index, steps
                    )

        args["__slicer_skip_sync"] = True  # passed onto the decorator
        return args

    @staticmethod
    def convert_index(
        source: ImageSlicerArea,
        target: ImageSlicerArea,
        axis: int,
        index: int,
        steps: bool,
    ):
        if steps:
            return round(
                index * source.array_slicer.incs[axis] / target.array_slicer.incs[axis]
            )
        else:
            value: np.float32 = source.array_slicer.value_of_index(
                axis, index, uniform=False
            )
            new_index: int = target.array_slicer.index_of_value(
                axis, value, uniform=False
            )
            return new_index


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
    bench
        Prints the fps on Ctrl + drag, for debug purposes

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
    sigShapeChanged()
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
    ]  #: List of :class:`PySide6.QtGui.QColor` containing colors for multiple cursors.

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

    @property
    def sigShapeChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""
        return self.array_slicer.sigShapeChanged

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        data: xr.DataArray | npt.ArrayLike | None = None,
        cmap: str | pg.ColorMap = "magma",
        gamma: float = 0.5,
        zeroCentered: bool = False,
        rad2deg: bool | Iterable[str] = False,
        *,
        bench: bool = False,
        image_cls=None,
        plotdata_cls=None,
    ):
        super().__init__(parent)

        self._linking_proxy: SlicerLinkProxy | None = None

        self.bench = bench

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

        pkw = {"image_cls": image_cls, "plotdata_cls": plotdata_cls}
        self.manual_limits: dict[str | list[float]] = {}
        self._plots: tuple[ItoolGraphicsLayoutWidget, ...] = (
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(0, 1), **pkw),
            ItoolGraphicsLayoutWidget(self, display_axis=(0,), **pkw),
            ItoolGraphicsLayoutWidget(self, display_axis=(1,), is_vertical=True, **pkw),
            ItoolGraphicsLayoutWidget(self, display_axis=(2,), **pkw),
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(0, 2), **pkw),
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(2, 1), **pkw),
            ItoolGraphicsLayoutWidget(self, display_axis=(3,), **pkw),
            ItoolGraphicsLayoutWidget(self, image=True, display_axis=(3, 2), **pkw),
        )
        for i in (1, 4):
            self._splitters[2].addWidget(self._plots[i])
        for i in (6, 3, 7):
            self._splitters[3].addWidget(self._plots[i])
        self._splitters[5].addWidget(self._plots[0])
        for i in (5, 2):
            self._splitters[6].addWidget(self._plots[i])

        self.qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
        self.qapp.aboutToQuit.connect(self.on_close)

        cmap_reversed = False

        if isinstance(cmap, str):
            if cmap.endswith("_r"):
                cmap = cmap[:-2]
                cmap_reversed = True
            if cmap.startswith("cet_CET"):
                cmap = cmap[4:]

        self.colormap_properties: dict[str, str | pg.ColorMap | float | bool] = {
            "cmap": cmap,
            "gamma": gamma,
            "reversed": cmap_reversed,
            "highContrast": False,
            "zeroCentered": zeroCentered,
        }

        self._data: xr.DataArray | None = None
        self.current_cursor: int = 0

        self.levels_locked: bool = False

        if data is not None:
            self.set_data(data, rad2deg=rad2deg)

        if self.bench:
            print("\n")

    def on_close(self):
        self.array_slicer.clear_cache()
        self.data.close()
        if hasattr(self, "_data"):
            self._data.close()
            del self._data
        gc.collect()

    def connect_axes_signals(self):
        for ax in self.axes:
            ax.connect_signals()

    def disconnect_axes_signals(self):
        for ax in self.axes:
            ax.disconnect_signals()

    def connect_signals(self):
        self.connect_axes_signals()
        self.sigDataChanged.connect(self.refresh_all)
        self.sigShapeChanged.connect(self.refresh_all)
        self.sigCursorCountChanged.connect(lambda: self.set_colormap(update=True))

    def add_link(self, proxy: SlicerLinkProxy):
        proxy.add(self)

    def remove_link(self):
        self._linking_proxy.remove(self)

    @property
    def is_linked(self) -> bool:
        return self._linking_proxy is not None

    @property
    def colormap(self) -> str | pg.ColorMap:
        return self.colormap_properties["cmap"]

    @property
    def slices(self) -> tuple[ItoolPlotItem, ...]:
        if self.data.ndim == 2:
            return ()
        elif self.data.ndim == 3:
            return tuple(self.get_axes(ax) for ax in (4, 5))
        elif self.data.ndim == 4:
            return tuple(self.get_axes(ax) for ax in (4, 5, 7))

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
    def main_image(self) -> ItoolPlotItem:
        """returns the main PlotItem"""
        return self.get_axes(0)

    @property
    def images(self) -> tuple[ItoolPlotItem, ...]:
        return (self.main_image, *self.slices)

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

    @QtCore.Slot()
    @QtCore.Slot(tuple)
    @QtCore.Slot(tuple, bool)
    def refresh_all(
        self, axes: tuple[int, ...] | None = None, only_plots: bool = False
    ):
        for c in range(self.n_cursors):
            self.sigIndexChanged.emit(c, axes)
        if not only_plots:
            for ax in self.axes:
                ax.refresh_labels()

    @QtCore.Slot(tuple)
    @link_slicer
    def refresh_current(self, axes: tuple[int, ...] | None = None):
        self.sigIndexChanged.emit(self.current_cursor, axes)

    @QtCore.Slot(int, list)
    @link_slicer
    def refresh(self, cursor: int, axes: tuple[int, ...] | None = None):
        self.sigIndexChanged.emit(cursor, axes)

    def view_all(self):
        for ax in self.axes:
            ax.vb.enableAutoRange()
            ax.vb.updateAutoRange()

    @link_slicer
    def center_all_cursors(self):
        for i in range(self.n_cursors):
            self.array_slicer.center_cursor(i)

    @link_slicer
    def center_cursor(self):
        self.array_slicer.center_cursor(self.current_cursor)

    @link_slicer
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
            self.disconnect_axes_signals()
        else:
            n_cursors_old = 1

        if not isinstance(data, xr.DataArray):
            if isinstance(data, xr.Dataset):
                try:
                    data = data.spectrum
                except AttributeError:
                    data = data[next(iter(data.data_vars.keys()))]
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
                    for d in ("phi", "theta", "beta", "alpha", "chi", "xi")
                    if d in data.dims
                ]
            self._data = data.assign_coords({d: np.rad2deg(data[d]) for d in conv_dims})

        if hasattr(self, "_array_slicer"):
            self._array_slicer.set_array(self._data, reset=True)
        else:
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
    @link_slicer
    def swap_axes(self, ax1: int, ax2: int):
        self.array_slicer.swap_axes(ax1, ax2)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True)
    def set_index(
        self, axis: int, value: int, update: bool = True, cursor: int | None = None
    ):
        if cursor is None:
            cursor = self.current_cursor
        self.array_slicer.set_index(cursor, axis, value, update)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    def step_index(
        self, axis: int, value: int, update: bool = True, cursor: int | None = None
    ):
        if cursor is None:
            cursor = self.current_cursor
        self.array_slicer.step_index(cursor, axis, value, update)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    def step_index_all(self, axis: int, value: int, update: bool = True):
        for i in range(self.n_cursors):
            self.array_slicer.step_index(i, axis, value, update)

    @QtCore.Slot(int, float, bool, bool)
    @link_slicer
    def set_value(
        self,
        axis: int,
        value: float,
        update: bool = True,
        uniform: bool = False,
        cursor: int | None = None,
    ):
        if cursor is None:
            cursor = self.current_cursor
        self.array_slicer.set_value(cursor, axis, value, update, uniform)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    def set_bin(
        self, axis: int, value: int, update: bool = True, cursor: int | None = None
    ):
        if cursor is None:
            cursor = self.current_cursor
        new_bins: list[int | None] = [None] * self.data.ndim
        new_bins[axis] = value
        self.array_slicer.set_bins(cursor, new_bins, update)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    def set_bin_all(self, axis: int, value: int, update: bool = True):
        new_bins: list[int | None] = [None] * self.data.ndim
        new_bins[axis] = value
        for c in range(self.n_cursors):
            self.array_slicer.set_bins(c, new_bins, update)

    @QtCore.Slot()
    @link_slicer
    def add_cursor(self):
        self.array_slicer.add_cursor(self.current_cursor, update=False)
        self.cursor_colors.append(self.gen_cursor_color(self.n_cursors - 1))
        self.current_cursor = self.n_cursors - 1
        for ax in self.axes:
            ax.add_cursor(update=False)
        self._colorbar.cb.level_change()
        self.refresh_current()
        self.sigCursorCountChanged.emit(self.n_cursors)
        self.sigCurrentCursorChanged.emit(self.current_cursor)

    @QtCore.Slot(int)
    @link_slicer
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

    @link_slicer(color=True)
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
        self.levels_locked: bool = lock

        if self.levels_locked:
            levels = self.array_slicer.limits
            self._colorbar.cb.setLimits(levels)
        for im in self._imageitems:
            if self.levels_locked:
                im.setLevels(levels, update=False)
            else:
                im.levels = None
            im.refresh_data()

        self._colorbar.setVisible(self.levels_locked)
        self.sigViewOptionChanged.emit()

    def adjust_layout(
        self,
        horiz_pad: int = 45,
        vert_pad: int = 30,
        font_size: float = 11.0,
        r: tuple[float, float, float, float] = (1.2, 1.5, 3.0, 1.0),
    ):
        """Determines the padding and aspect ratios.

        Parameters
        ----------
        horiz_pad, vert_pad
            Reserved space for the x and y axes.
        font_size
            Font size in points.
        r
            4 numbers that determine the layout aspect ratios. See notes.

        Notes
        -----
        Axes indices and layout parameters.

        .. code-block:: text

                 ┌───────────┬───────────┐
            r[0] │     1     │     6     │
                 │           ├───────────┤
                 ├───────────┤     3     │
            r[1] │     4     ├───────────┤
                 │           │     7     │
                 │───────────┼───────┬───┤
                 │           │       │   │
            r[2] │     0     │   5   │ 2 │
                 │           │       │   │
                 └───────────┴───────┴───┘
                  r[3] * r[2]

        """

        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))

        valid_axis: tuple[tuple[bool, bool, bool, bool]] = (
            (1, 0, 0, 1),
            (1, 1, 0, 0),
            (0, 0, 1, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 1, 0),
        )  # booleans corresponding to the (left, top, right, bottom) axes of each plot.

        invalid: list[int] = []  # axes to hide.
        r0, r1, r2, r3 = r

        # !TODO: automate this based on ItoolPlotItem.display_axis
        if self.data.ndim == 2:
            invalid = [4, 5, 6, 7]
            r1 = r0 / 6
        elif self.data.ndim == 3:
            invalid = [6, 7]

        r01 = r0 / r1
        scale = 100
        d = self._splitters[0].handleWidth() / scale  # padding due to splitters
        sizes: list[tuple[float, ...]] = [
            (r0 + r1, r2),
            (r3 * r2, r3 * (r0 + r1)),
            ((r0 + r1 - d) * r01, (r0 + r1 - d) / r01),
            ((r0 + r1 - d) / 2, (r0 + r1 - d) / 2, 0),
            (r3 * r2, r3 * (r0 + r1)),
            (r2,),
            ((r3 * (r0 + r1) - d) / r01, (r3 * (r0 + r1) - d) * r01),
        ]
        if self.data.ndim == 4:
            sizes[3] = (0, 0, (r0 + r1 - d))
        for split, sz in zip(self._splitters, sizes):
            split.setSizes(tuple(round(s * scale) for s in sz))

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
        self.get_axes(3).setVisible(self.data.ndim != 2)

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
            not in self.qapp.keyboardModifiers()
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
            not in self.qapp.keyboardModifiers()
        ):
            super().mouseClickEvent(ev)
        else:
            self.setMouseHover(False)
            ev.ignore()

    def hoverEvent(self, ev):
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            not in self.qapp.keyboardModifiers()
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


class ItoolDisplayObject:
    def __init__(self, axes, cursor: int | None = None):
        self.axes = axes
        if cursor is None:
            cursor = 0
        self._cursor_index = int(cursor)
        self.qapp: QtGui.QGuiApplication = QtGui.QGuiApplication.instance()

    @property
    def display_axis(self):
        return self.axes.display_axis

    @property
    def slicer_area(self) -> ImageSlicerArea:
        return self.axes.slicer_area

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.axes.array_slicer

    @property
    def cursor_index(self):
        return self._cursor_index

    @cursor_index.setter
    def cursor_index(self, value: int):
        self._cursor_index = int(value)

    @property
    def sliced_data(self) -> xr.DataArray:
        return self.array_slicer.xslice(self.cursor_index, self.display_axis)

    def refresh_data(self):
        pass


class ItoolPlotDataItem(ItoolDisplayObject, pg.PlotDataItem):
    def __init__(
        self,
        axes,
        cursor: int | None = None,
        is_vertical: bool = False,
        **kargs,
    ):
        pg.PlotDataItem.__init__(self, axes=axes, cursor=cursor, **kargs)
        ItoolDisplayObject.__init__(self, axes=axes, cursor=cursor)
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


class ItoolImageItem(ItoolDisplayObject, BetterImageItem):
    def __init__(
        self,
        axes,
        cursor: int | None = None,
        **kargs,
    ):
        BetterImageItem.__init__(self, axes=axes, cursor=cursor, **kargs)
        ItoolDisplayObject.__init__(self, axes=axes, cursor=cursor)

    def updateImage(self, *args, **kargs):
        defaults = {"autoLevels": not self.slicer_area.levels_locked}
        defaults.update(kargs)
        return self.setImage(*args, **defaults)

    @suppressnanwarning
    def refresh_data(self):
        ItoolDisplayObject.refresh_data(self)
        rect, img = self.array_slicer.slice_with_coord(
            self.cursor_index, self.display_axis
        )
        self.setImage(
            image=img, rect=rect, autoLevels=not self.slicer_area.levels_locked
        )

    def mouseDragEvent(self, ev: mouseEvents.MouseDragEvent):
        if QtCore.Qt.KeyboardModifier.ControlModifier in self.qapp.keyboardModifiers():
            ev.ignore()
        else:
            super().mouseDragEvent(ev)

    def mouseClickEvent(self, ev: mouseEvents.MouseClickEvent):
        if QtCore.Qt.KeyboardModifier.ControlModifier in self.qapp.keyboardModifiers():
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
        image_cls=None,
        plotdata_cls=None,
        **item_kw,
    ):
        super().__init__(
            axisItems={a: BetterAxisItem(a) for a in ("left", "right", "top", "bottom")}
        )
        for act in ["Transforms", "Downsample", "Average", "Alpha", "Points"]:
            self.setContextMenuActionVisible(act, False)

        save_action = self.vb.menu.addAction("Save data as HDF5")
        save_action.triggered.connect(lambda: self.save_current_data())

        for i in (0, 1):
            self.getViewBoxMenu().ctrl[i].linkCombo.setVisible(False)
            self.getViewBoxMenu().ctrl[i].label.setVisible(False)
        self.getViewBox().setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        self.slicer_area = slicer_area
        self._display_axis = display_axis

        self.is_image = image
        self._item_kw = item_kw

        if image_cls is None:
            self.image_cls = ItoolImageItem
        if plotdata_cls is None:
            self.plotdata_cls = ItoolPlotDataItem

        self.slicer_data_items: list[ItoolImageItem | ItoolPlotDataItem] = []
        self.cursor_lines: list[dict[int, ItoolCursorLine]] = []
        self.cursor_spans: list[dict[int, ItoolCursorSpan]] = []
        self.add_cursor(update=False)

        self.proxy = pg.SignalProxy(
            self.sigDragged,
            delay=1 / 60,
            rateLimit=60,
            slot=self.process_drag,
        )
        if self.slicer_area.bench:
            self._time_start = None
            self._time_end = None
            self._single_queue = collections.deque([0], maxlen=9)
            self._next_queue = collections.deque([0], maxlen=9)

    @property
    def axis_dims(self) -> list[str]:
        dim_list = [self.slicer_area.data.dims[ax] for ax in self.display_axis]
        if not self.is_image:
            if self.slicer_data_items[-1].is_vertical:
                dim_list = [None, *dim_list]
            else:
                dim_list = [*dim_list, None]
        return dim_list

    @property
    def is_independent(self) -> bool:
        return self.vb.state["linkedViews"] == [None, None]

    def refresh_manual_range(self):
        if self.is_independent:
            return
        for dim, auto, rng in zip(
            self.axis_dims, self.vb.state["autoRange"], self.vb.state["viewRange"]
        ):
            if dim is not None:
                if auto:
                    self.slicer_area.manual_limits.pop("dim", None)
                else:
                    self.slicer_area.manual_limits[dim] = rng

    def update_manual_range(self):
        if self.is_independent:
            return
        self.set_range_from(self.slicer_area.manual_limits)

    def set_range_from(self, limits: dict[str, list[float]], **kwargs: dict):
        for dim, key in zip(self.axis_dims, ("xRange", "yRange")):
            if dim is not None:
                try:
                    kwargs[key] = limits[dim]
                except KeyError:
                    pass
        if len(kwargs) != 0:
            self.setRange(**kwargs)

    def getMenu(self) -> QtWidgets.QMenu:
        return self.ctrlMenu

    def getViewBox(self) -> pg.ViewBox:
        return self.vb

    def getViewBoxMenu(self) -> ViewBoxMenu:
        return self.getViewBox().menu

    def mouseDragEvent(self, ev: mouseEvents.MouseDragEvent):
        modifiers = self.slicer_area.qapp.keyboardModifiers()
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier in modifiers
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            ev.accept()
            self.sigDragged.emit(ev, modifiers)
            # self.process_drag((ev, modifiers))
        else:
            ev.ignore()

    def process_drag(
        self, sig: tuple[mouseEvents.MouseDragEvent, QtCore.Qt.KeyboardModifier]
    ):
        if self.slicer_area.bench:
            if self._time_end is not None:
                self._single_queue.append(1 / (self._time_end - self._time_start))
            self._time_end = self._time_start
            self._time_start = time.perf_counter()
            if self._time_end is not None:
                self._next_queue.append(1 / (self._time_start - self._time_end))
            print(
                "\x1b[2K\x1b[1A\x1b[2K"
                f"SingleUpdate\t{np.mean(self._single_queue):.1f}"
                f" ± {np.std(self._single_queue):.1f}",
                f"NextUpdate\t{np.mean(self._next_queue):.1f}"
                f" ± {np.std(self._next_queue):.1f}",
                sep="\t",
            )
            self._time_start = time.perf_counter()

        ev, modifiers = sig
        data_pos = self.getViewBox().mapSceneToView(ev.scenePos())

        if (not self.is_image) and self.slicer_data_items[-1].is_vertical:
            data_pos_coords = (data_pos.y(), data_pos.x())
        else:
            data_pos_coords = (data_pos.x(), data_pos.y())

        if QtCore.Qt.KeyboardModifier.AltModifier in modifiers:
            for c in range(self.slicer_area.n_cursors):
                for i, ax in enumerate(self.display_axis):
                    self.slicer_area.set_value(
                        ax, data_pos_coords[i], update=False, uniform=True, cursor=c
                    )
                self.slicer_area.refresh(c, self.display_axis)
        else:
            for i, ax in enumerate(self.display_axis):
                self.slicer_area.set_value(
                    ax, data_pos_coords[i], update=False, uniform=True
                )
            self.slicer_area.refresh_current(self.display_axis)

        if self.slicer_area.bench:
            self._time_end = time.perf_counter()

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
            item = self.image_cls(
                self,
                cursor=new_cursor,
                autoDownsample=True,
                axisOrder="row-major",
                **self._item_kw,
            )
            if self.slicer_area.color_locked:
                item.setLevels(self.array_slicer.limits, update=True)
        else:
            item = self.plotdata_cls(
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

        cursors: list[ItoolCursorLine] = []
        spans: list[ItoolCursorSpan] = []
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
            for v in line_dict.values():
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
            self.slicer_area.qapp.keyboardModifiers()
            != QtCore.Qt.KeyboardModifier.AltModifier
        ):
            self.slicer_area.set_value(
                axis, value, update=True, uniform=True, cursor=cursor
            )
        else:
            for i in range(self.slicer_area.n_cursors):
                self.slicer_area.set_value(
                    axis, value, update=True, uniform=True, cursor=i
                )

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
        self._slicer_area.sigShapeChanged.connect(self.update_manual_range)
        self.vb.sigRangeChanged.connect(self.refresh_manual_range)

    def disconnect_signals(self):
        self._slicer_area.sigIndexChanged.disconnect(self.refresh_items_data)
        self._slicer_area.sigBinChanged.disconnect(self.refresh_items_data)
        self._slicer_area.sigShapeChanged.disconnect(self.update_manual_range)
        self.vb.sigRangeChanged.disconnect(self.refresh_manual_range)

    @QtCore.Slot(int, object)
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
            label_kw = {}

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
            for i, item in enumerate(self.slicer_data_items):
                item.setVisible(i == index)

    def save_current_data(self, fileName=None):
        if fileName is None:
            self.fileDialog = QtWidgets.QFileDialog()
            self.fileDialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
            self.fileDialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
            self.fileDialog.setNameFilter("xarray HDF5 Files (*.h5)")
            if pg.PlotItem.lastFileDir is not None:
                self.fileDialog.setDirectory(
                    os.path.join(
                        pg.PlotItem.lastFileDir, f"{self.slicer_area._data.name}.h5"
                    )
                )
            else:
                self.fileDialog.setDirectory(
                    os.path.join(os.getcwd(), f"{self.slicer_area._data.name}.h5")
                )
            self.fileDialog.show()
            self.fileDialog.fileSelected.connect(self.save_current_data)
            return

        fileName = str(fileName)
        pg.PlotItem.lastFileDir = os.path.dirname(fileName)

        import erlab.io

        erlab.io.save_as_hdf5(
            self.slicer_data_items[self.slicer_area.current_cursor].sliced_data,
            fileName,
        )

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


class ItoolColorBarItem(BetterColorBarItem):
    def __init__(self, slicer_area: ImageSlicerArea | None = None, **kwargs):
        self._slicer_area = slicer_area
        kwargs.setdefault(
            "axisItems",
            {a: BetterAxisItem(a) for a in ("left", "right", "top", "bottom")},
        )
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
        self._span.sigRegionChanged.connect(self.level_change)
        self._span.sigRegionChangeFinished.connect(self.level_change_fin)
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
