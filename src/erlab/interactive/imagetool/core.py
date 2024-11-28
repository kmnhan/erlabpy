"""Provides core functionality of ImageTool."""

from __future__ import annotations

import collections
import contextlib
import copy
import functools
import inspect
import os
import pathlib
import time
import uuid
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import xarray as xr
from pyqtgraph.GraphicsScene import mouseEvents
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive.colors import (
    BetterColorBarItem,
    BetterImageItem,
    pg_colormap_powernorm,
)
from erlab.interactive.imagetool.slicer import ArraySlicer
from erlab.interactive.utils import BetterAxisItem, copy_to_clipboard, make_crosshairs

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Sequence

    from erlab.interactive.imagetool.slicer import ArraySlicerState


class ColorMapState(TypedDict):
    """A dictionary containing the colormap state of an `ImageSlicerArea` instance."""

    cmap: str | pg.ColorMap
    gamma: float
    reverse: bool
    high_contrast: bool
    zero_centered: bool
    levels_locked: bool
    levels: NotRequired[tuple[float, float]]


class ImageSlicerState(TypedDict):
    """A dictionary containing the state of an `ImageSlicerArea` instance."""

    color: ColorMapState
    slice: ArraySlicerState
    current_cursor: int
    manual_limits: dict[str, list[float]]
    cursor_colors: list[str]
    splitter_sizes: NotRequired[list[list[int]]]


suppressnanwarning = np.testing.suppress_warnings()
suppressnanwarning.filter(RuntimeWarning, r"All-NaN (slice|axis) encountered")


def _link_splitters(
    s0: QtWidgets.QSplitter, s1: QtWidgets.QSplitter, reverse: bool = False
) -> None:
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


def _sync_splitters(s0: QtWidgets.QSplitter, s1: QtWidgets.QSplitter) -> None:
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
    ) -> None:
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


def suppress_history(method: Callable | None = None):
    """Ignore history changes made within the decorated method."""

    def my_decorator(method: Callable):
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            area = self.slicer_area if hasattr(self, "slicer_area") else self
            with area.history_suppressed():
                return method(self, *args, **kwargs)

        return wrapped

    if method is not None:
        return my_decorator(method)
    return my_decorator


def record_history(method: Callable | None = None):
    """Log history before calling the decorated method."""

    def my_decorator(method: Callable):
        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):
            area = self.slicer_area if hasattr(self, "slicer_area") else self
            area.sigWriteHistory.emit()
            with area.history_suppressed():
                # Prevent making additional records within the method
                return method(self, *args, **kwargs)

        return wrapped

    if method is not None:
        return my_decorator(method)
    return my_decorator


def link_slicer(
    func: Callable | None = None,
    *,
    indices: bool = False,
    steps: bool = False,
    color: bool = False,
):
    """Sync decorated methods across multiple `ImageSlicerArea` instances.

    Parameters
    ----------
    func
        The method to sync across multiple instances of `ImageSlicerArea`.
    indices
        If `True`, the input argument named `value` given to `func` are interpreted as
        indices, and will be converted to appropriate values for other instances of
        `ImageSlicerArea`. The behavior of this conversion is determined by `steps`. If
        `True`, An input argument named `axis` of type integer must be present in the
        decorated method to determine the axis along which the index is to be changed.
    steps
        If `False`, considers `value` as an absolute index. If `True`, considers `value`
        as a relative value such as the number of steps or bins. See the implementation
        of `SlicerLinkProxy` for more information.
    color
        Boolean whether the decorated method is related to visualization, such as
        colormap control.

    """

    def my_decorator(func: Callable):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # skip sync if already synced
            skip_sync = kwargs.pop("__slicer_skip_sync", False)

            out = func(*args, **kwargs)
            if args[0].is_linked and not skip_sync:
                all_args = inspect.Signature.from_callable(func).bind(*args, **kwargs)
                all_args.apply_defaults()
                obj = all_args.arguments.pop("self")
                if obj._linking_proxy is not None:
                    obj._linking_proxy.sync(
                        obj,
                        func.__name__,
                        all_args.arguments,
                        indices,
                        steps,
                        color,
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
    *slicers
        The slicers to link.
    link_colors
        Whether to sync color related changes, by default `True`.

    """

    def __init__(self, *slicers: ImageSlicerArea, link_colors: bool = True) -> None:
        self.link_colors = link_colors

        self._children: weakref.WeakSet[ImageSlicerArea] = weakref.WeakSet()
        for s in slicers:
            self.add(s)

    @property
    def children(self) -> weakref.WeakSet[ImageSlicerArea]:
        return self._children

    @property
    def num_children(self) -> int:
        return len(self.children)

    def unlink_all(self) -> None:
        for s in self.children:
            s._linking_proxy = None
        self.children.clear()

    def add(self, slicer_area: ImageSlicerArea) -> None:
        if slicer_area.is_linked:
            if slicer_area._linking_proxy == self:
                return
            raise ValueError("Already linked to another proxy.")
        self.children.add(slicer_area)
        slicer_area._linking_proxy = self

    def remove(self, slicer_area: ImageSlicerArea) -> None:
        self.children.remove(slicer_area)
        slicer_area._linking_proxy = None

    def sync(
        self,
        source: ImageSlicerArea,
        funcname: str,
        arguments: dict[str, Any],
        indices: bool,
        steps: bool,
        color: bool,
    ) -> None:
        r"""Propagate changes across multiple :class:`ImageSlicerArea`\ s.

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
        for target in self.children.difference({source}):
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
            index: int | None = args.get("value")

            if index is not None:
                axis: int | None = args.get("axis")

                if axis is None:
                    raise ValueError(
                        "Axis argument not found in method decorated "
                        "with the `indices=True` argument"
                    )

                args["value"] = self.convert_index(source, target, axis, index, steps)

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
                index
                * source.array_slicer.incs_uniform[axis]
                / target.array_slicer.incs_uniform[axis]
            )
        value = source.array_slicer.value_of_index(axis, index, uniform=False)
        new_index: int = target.array_slicer.index_of_value(
            axis, float(value), uniform=False
        )
        return new_index


class ImageSlicerArea(QtWidgets.QWidget):
    """A interactive tool based on :mod:`pyqtgraph` for exploring 3D data.

    Parameters
    ----------
    parent
        Parent widget.
    data : DataArray or array-like
        Data to display. The data must have 2 to 4 dimensions.
    cmap
        Default colormap of the data.
    gamma
        Default power law normalization of the colormap.
    zero_centered
        If `True`, the normalization is applied symmetrically from the midpoint of
        the colormap.
    rad2deg
        If `True` and `data` is not `None`, converts some known angle coordinates to
        degrees. If an iterable of strings is given, only the coordinates that
        correspond to the given strings are converted.
    bench
        Prints the fps on Ctrl + drag for debugging purposes.
    state
        Initial state containing the settings and cursor position.

    Signals
    -------
    sigDataChanged()

    sigCurrentCursorChanged(index)

    sigViewOptionChanged()

    sigHistoryChanged()

    sigWriteHistory()

    sigCursorCountChanged(n_cursors)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigIndexChanged(cursor, axes)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigBinChanged(cursor, axes)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigShapeChanged()
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.

    """

    COLORS: tuple[QtGui.QColor, ...] = (
        pg.mkColor(0.8),
        pg.mkColor("y"),
        pg.mkColor("m"),
        pg.mkColor("c"),
        pg.mkColor("g"),
        pg.mkColor("r"),
        pg.mkColor("b"),
    )  #: :class:`PySide6.QtGui.QColor`\ s for multiple cursors.

    sigDataChanged = QtCore.Signal()  #: :meta private:
    sigCurrentCursorChanged = QtCore.Signal(int)  #: :meta private:
    sigViewOptionChanged = QtCore.Signal()  #: :meta private:
    sigHistoryChanged = QtCore.Signal()  #: :meta private:
    sigWriteHistory = QtCore.Signal()  #: :meta private:

    @property
    def sigCursorCountChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""  # noqa: D400
        return self.array_slicer.sigCursorCountChanged

    @property
    def sigIndexChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""  # noqa: D400
        return self.array_slicer.sigIndexChanged

    @property
    def sigBinChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""  # noqa: D400
        return self.array_slicer.sigBinChanged

    @property
    def sigShapeChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""  # noqa: D400
        return self.array_slicer.sigShapeChanged

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        data: xr.DataArray | npt.ArrayLike | None = None,
        cmap: str | pg.ColorMap = "magma",
        gamma: float = 0.5,
        zero_centered: bool = False,
        rad2deg: bool | Iterable[str] = False,
        *,
        bench: bool = False,
        state: ImageSlicerState | None = None,
        image_cls=None,
        plotdata_cls=None,
        _in_manager: bool = False,
    ) -> None:
        super().__init__(parent)

        self._in_manager: bool = _in_manager

        self._linking_proxy: SlicerLinkProxy | None = None

        self.bench: bool = bench

        # Stores ktool, dtool, goldtool, etc.
        self._associated_tools: dict[str, QtWidgets.QWidget] = {}

        # Queues to handle undo and redo
        self._prev_states: collections.deque[ImageSlicerState] = collections.deque(
            maxlen=1000
        )
        self._next_states: collections.deque[ImageSlicerState] = collections.deque(
            maxlen=1000
        )

        # Flag to prevent writing history when restoring state
        self._write_history: bool = True

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._splitters: tuple[QtWidgets.QSplitter, ...] = (
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

        layout.addWidget(self._splitters[0])
        for i, j in ((0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (4, 6)):
            self._splitters[i].addWidget(self._splitters[j])
        _sync_splitters(self._splitters[1], self._splitters[4])

        self.cursor_colors: list[QtGui.QColor] = [self.COLORS[0]]

        self._colorbar = ItoolColorBar(self)
        self._colorbar.setVisible(False)
        layout.addWidget(self._colorbar)

        cmap_reversed: bool = False
        if isinstance(cmap, str):
            if cmap.endswith("_r"):
                cmap = cmap[:-2]
                cmap_reversed = True
            if cmap.startswith("cet_CET"):
                cmap = cmap[4:]
        self._colormap_properties: ColorMapState = {
            "cmap": cmap,
            "gamma": gamma,
            "reverse": cmap_reversed,
            "high_contrast": False,
            "zero_centered": zero_centered,
            "levels_locked": False,
        }

        pkw = {"image_cls": image_cls, "plotdata_cls": plotdata_cls}
        self.manual_limits: dict[str, list[float]] = {}
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

        self.qapp = cast(QtWidgets.QApplication, QtWidgets.QApplication.instance())
        self.qapp.aboutToQuit.connect(self.on_close)

        self._data: xr.DataArray | None = None
        self._file_path: pathlib.Path | None = None
        self.current_cursor: int = 0

        if data is not None:
            self.set_data(data, rad2deg=rad2deg)

        if self.bench:
            print("\n")

        if state is not None:
            self.state = state

    @property
    def colormap_properties(self) -> ColorMapState:
        prop = copy.deepcopy(self._colormap_properties)
        if prop["levels_locked"]:
            prop["levels"] = copy.deepcopy(self.levels)
        return prop

    @property
    def state(self) -> ImageSlicerState:
        return {
            "color": self.colormap_properties,
            "slice": self.array_slicer.state,
            "current_cursor": int(self.current_cursor),
            "manual_limits": copy.deepcopy(self.manual_limits),
            "splitter_sizes": self.splitter_sizes,
            "cursor_colors": [c.name() for c in self.cursor_colors],
        }

    @state.setter
    def state(self, state: ImageSlicerState) -> None:
        if "splitter_sizes" in state:
            self.splitter_sizes = state["splitter_sizes"]

        # Restore cursor number and colors
        self.make_cursors(len(state["cursor_colors"]), colors=state["cursor_colors"])

        # Set current cursor before restoring coordinates
        self.set_current_cursor(state["current_cursor"], update=False)

        # Restore coordinates, bins, etc.
        self.array_slicer.state = state["slice"]

        self.manual_limits = state.get("manual_limits", {})
        self.sigShapeChanged.emit()  # to trigger manual limits update
        self.refresh_all()

        # Restore colormap settings
        try:
            self.set_colormap(**state.get("color", {}), update=True)
        except Exception:
            warnings.warn("Failed to restore colormap settings, skipping", stacklevel=1)

    @property
    def splitter_sizes(self) -> list[list[int]]:
        return [s.sizes() for s in self._splitters]

    @splitter_sizes.setter
    def splitter_sizes(self, sizes: list[list[int]]) -> None:
        for s, size in zip(self._splitters, sizes, strict=True):
            s.setSizes(size)

    @property
    def is_linked(self) -> bool:
        return self._linking_proxy is not None

    @property
    def linked_slicers(self) -> weakref.WeakSet[ImageSlicerArea]:
        if self._linking_proxy is not None:
            return self._linking_proxy.children - {self}

        return weakref.WeakSet()

    @property
    def colormap(self) -> str | pg.ColorMap:
        return self.colormap_properties["cmap"]

    @property
    def levels_locked(self) -> bool:
        return self.colormap_properties["levels_locked"]

    @levels_locked.setter
    def levels_locked(self, value: bool) -> None:
        self.lock_levels(value)

    @property
    def levels(self) -> tuple[float, float]:
        return self._colorbar.cb.spanRegion()

    @levels.setter
    def levels(self, levels: tuple[float, float]) -> None:
        self._colorbar.cb.setSpanRegion(levels)

    @property
    def slices(self) -> tuple[ItoolPlotItem, ...]:
        match self.data.ndim:
            case 2:
                return ()
            case 3:
                return tuple(self.get_axes(ax) for ax in (4, 5))
            case 4:
                return tuple(self.get_axes(ax) for ax in (4, 5, 7))
            case _:
                raise ValueError("Data must have 2 to 4 dimensions")

    @property
    def profiles(self) -> tuple[ItoolPlotItem, ...]:
        match self.data.ndim:
            case 2:
                profile_axes: tuple[int, ...] = (1, 2)
            case 3:
                profile_axes = (1, 2, 3)
            case _:
                profile_axes = (1, 2, 3, 6)

        return tuple(self.get_axes(ax) for ax in profile_axes)

    @property
    def main_image(self) -> ItoolPlotItem:
        """Return the main PlotItem."""
        return self.get_axes(0)

    @property
    def images(self) -> tuple[ItoolPlotItem, ...]:
        return (self.main_image, *self.slices)

    @property
    def axes(self) -> tuple[ItoolPlotItem, ...]:
        """Currently valid subset of self._plots."""
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
    def undoable(self) -> bool:
        return len(self._prev_states) > 0

    @property
    def redoable(self) -> bool:
        return len(self._next_states) > 0

    @contextlib.contextmanager
    def history_suppressed(self):
        original = bool(self._write_history)
        self._write_history = False
        try:
            yield
        finally:
            self._write_history = original

    def on_close(self) -> None:
        if hasattr(self, "array_slicer"):
            self.array_slicer.clear_cache()
        if hasattr(self, "data"):
            self.data.close()
        if hasattr(self, "_data") and self._data is not None:
            self._data.close()
            del self._data

    @QtCore.Slot()
    def write_state(self) -> None:
        if not self._write_history:
            return

        last_state = self._prev_states[-1] if self.undoable else None
        curr_state = self.state

        # Don't store splitter sizes in history
        if last_state is not None:
            last_state.pop("splitter_sizes", None)
        curr_state.pop("splitter_sizes", None)

        if last_state is None or last_state != curr_state:
            self._prev_states.append(curr_state)
            self._next_states.clear()
            self.sigHistoryChanged.emit()

    @QtCore.Slot()
    @suppress_history
    def flush_history(self) -> None:
        self._prev_states.clear()
        self._next_states.clear()
        self.sigHistoryChanged.emit()

    @QtCore.Slot()
    @link_slicer
    @suppress_history
    def undo(self) -> None:
        if not self.undoable:
            raise RuntimeError("Nothing to undo")
        self._next_states.append(self.state)
        self.state = self._prev_states.pop()
        self.sigHistoryChanged.emit()

    @QtCore.Slot()
    @link_slicer
    @suppress_history
    def redo(self) -> None:
        if not self.redoable:
            raise RuntimeError("Nothing to redo")
        self._prev_states.append(self.state)
        self.state = self._next_states.pop()
        self.sigHistoryChanged.emit()

    def connect_axes_signals(self) -> None:
        for ax in self.axes:
            ax.connect_signals()

    def disconnect_axes_signals(self) -> None:
        for ax in self.axes:
            ax.disconnect_signals()

    def connect_signals(self) -> None:
        self.connect_axes_signals()
        self.sigDataChanged.connect(self.refresh_all)
        self.sigShapeChanged.connect(self.refresh_all)
        self.sigCursorCountChanged.connect(lambda: self.set_colormap(update=True))
        self.sigWriteHistory.connect(self.write_state)

    def link(self, proxy: SlicerLinkProxy) -> None:
        proxy.add(self)

    def unlink(self) -> None:
        if self.is_linked:
            cast(SlicerLinkProxy, self._linking_proxy).remove(self)

    def get_current_index(self, axis: int) -> int:
        return self.array_slicer.get_index(self.current_cursor, axis)

    def get_current_value(self, axis: int, uniform: bool = False) -> float:
        return self.array_slicer.get_value(self.current_cursor, axis, uniform=uniform)

    def get_axes_widget(self, index: int) -> ItoolGraphicsLayoutWidget:
        return self._plots[index]

    def get_axes(self, index: int) -> ItoolPlotItem:
        return self._plots[index].plotItem

    @QtCore.Slot()
    def close_associated_windows(self) -> None:
        for tool in dict(self._associated_tools).values():
            tool.close()

    @QtCore.Slot()
    @QtCore.Slot(tuple)
    @QtCore.Slot(tuple, bool)
    def refresh_all(
        self, axes: tuple[int, ...] | None = None, only_plots: bool = False
    ) -> None:
        for c in range(self.n_cursors):
            self.sigIndexChanged.emit(c, axes)
        if not only_plots:
            for ax in self.axes:
                ax.refresh_labels()

    @QtCore.Slot(tuple)
    @link_slicer
    def refresh_current(self, axes: tuple[int, ...] | None = None) -> None:
        self.sigIndexChanged.emit(self.current_cursor, axes)

    @QtCore.Slot(int, list)
    @link_slicer
    def refresh(self, cursor: int, axes: tuple[int, ...] | None = None) -> None:
        self.sigIndexChanged.emit(cursor, axes)

    @QtCore.Slot()
    def view_all(self) -> None:
        for ax in reversed(self.axes):
            # Updating linked axes before the main image to prevent fixed limits
            ax.vb.enableAutoRange()
            ax.vb.updateAutoRange()

    @QtCore.Slot()
    @link_slicer
    @record_history
    def center_all_cursors(self) -> None:
        for i in range(self.n_cursors):
            self.array_slicer.center_cursor(i)

    @QtCore.Slot()
    @link_slicer
    @record_history
    def center_cursor(self) -> None:
        self.array_slicer.center_cursor(self.current_cursor)

    @link_slicer
    @record_history
    def set_current_cursor(self, cursor: int, update: bool = True) -> None:
        cursor = cursor % self.n_cursors
        if cursor > self.n_cursors - 1:
            raise IndexError("Cursor index out of range")
        self.current_cursor = cursor
        if update:
            self.refresh_current()
        self.sigCurrentCursorChanged.emit(cursor)

    def set_data(
        self,
        data: xr.DataArray | npt.ArrayLike,
        rad2deg: bool | Iterable[str] = False,
        file_path: str | os.PathLike | None = None,
    ) -> None:
        """Set the data to be displayed.

        Parameters
        ----------
        data
            The data to be displayed. If a `xarray.DataArray` is given, the
            dimensions and coordinates are used to determine the axes of the plots. If a
            :class:`xarray.Dataset` is given, the first data variable is used. If a
            :class:`numpy.ndarray` is given, it is converted to a `xarray.DataArray`
            with default dimensions.
        rad2deg
            If `True`, converts coords along dimensions that have angle-like names to
            degrees. If an iterable of strings is given, coordinates for dimensions that
            correspond to the given strings are converted.
        file_path
            Path to the file from which the data was loaded. If given, the file path is
            used to set the window title.

        """
        self._file_path = pathlib.Path(file_path) if file_path is not None else None
        if hasattr(self, "_array_slicer") and hasattr(self, "_data"):
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
                    data = cast(xr.DataArray, data[next(iter(data.data_vars.keys()))])
                except StopIteration as e:
                    raise ValueError("No data variables found in Dataset") from e
            else:
                data = xr.DataArray(np.asarray(data))
        if hasattr(data.data, "flags") and not data.data.flags["WRITEABLE"]:
            data = data.copy()

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

        try:
            if hasattr(self, "_array_slicer"):
                self._array_slicer.set_array(self._data, reset=True)
            else:
                self._array_slicer: ArraySlicer = ArraySlicer(self._data)
        except Exception as e:
            if self._in_manager:
                # Let the manager handle the exception
                raise
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.set_data(xr.DataArray(np.zeros((2, 2))))
            return

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
        self.flush_history()

    def update_values(
        self, values: npt.NDArray | xr.DataArray, update: bool = True
    ) -> None:
        """Update only the values of the data.

        The coords and shape of the data array are not changed.

        Parameters
        ----------
        values
            The new values to be set. If a `xarray.DataArray` is given, the dimensions
            must match the current data array. If a `numpy.ndarray` is given, the shape
            must match the current data array. Note that if the user has transposed the
            current data array, passing a `numpy.ndarray` with the original shape will
            fail.
        update
            If `True`, the plots are updated after setting the new values.

        Note
        ----
        This method only checks for matching dimension name and shape, and does not
        check for equal coordinate values.

        """
        if isinstance(values, xr.DataArray):
            if self.data.ndim != values.ndim:
                raise ValueError("DataArray dimensions do not match")
            if set(self.data.dims) != set(values.dims):
                raise ValueError("DataArray dimensions do not match")

            if self.data.dims != values.dims:
                values = values.transpose(*self.data.dims)
            if self.data.shape != values.shape:
                raise ValueError("DataArray shape does not match")

            values = values.values
        elif self.data.shape != values.shape:
            raise ValueError(
                "Data shape does not match. Array is "
                f"{self.data.shape} but {values.shape} given"
            )
        self.array_slicer._obj[:] = values

        if update:
            self.array_slicer.clear_val_cache(include_vals=True)
            self.refresh_all(only_plots=True)

            # This will update colorbar limits if visible
            self.lock_levels(self.levels_locked)

    def apply_func(self, func: Callable[[xr.DataArray], xr.DataArray] | None) -> None:
        """Apply a function to the data.

        The function must accept the data as the first argument and return a new
        DataArray. The returned DataArray must have the same dimensions and coordinates
        as the original data.

        This action is not recorded in the history, and the data is not affected. Only
        one function can be applied at a time.

        Parameters
        ----------
        func
            The function to apply to the data. if None, the data is restored.

        """
        # self._data is original data passed to `set_data`
        # self.data is the current data transformed by ArraySlicer
        if self._data is None:
            return

        if func is None:
            self.update_values(self._data)
        else:
            self.update_values(func(self._data))

    @QtCore.Slot(int, int)
    @link_slicer
    @record_history
    def swap_axes(self, ax1: int, ax2: int) -> None:
        self.array_slicer.swap_axes(ax1, ax2)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True)
    @record_history
    def set_index(
        self, axis: int, value: int, update: bool = True, cursor: int | None = None
    ) -> None:
        if cursor is None:
            cursor = self.current_cursor
        self.array_slicer.set_index(cursor, axis, value, update)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    @record_history
    def step_index(
        self, axis: int, value: int, update: bool = True, cursor: int | None = None
    ) -> None:
        if cursor is None:
            cursor = self.current_cursor
        self.array_slicer.step_index(cursor, axis, value, update)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    @record_history
    def step_index_all(self, axis: int, value: int, update: bool = True) -> None:
        for i in range(self.n_cursors):
            self.array_slicer.step_index(i, axis, value, update)

    @QtCore.Slot(int, float, bool, bool)
    @link_slicer
    @record_history
    def set_value(
        self,
        axis: int,
        value: float,
        update: bool = True,
        uniform: bool = False,
        cursor: int | None = None,
    ) -> None:
        if cursor is None:
            cursor = self.current_cursor
        self.array_slicer.set_value(cursor, axis, value, update, uniform)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    @record_history
    def set_bin(
        self, axis: int, value: int, update: bool = True, cursor: int | None = None
    ) -> None:
        if cursor is None:
            cursor = self.current_cursor
        new_bins: list[int | None] = [None] * self.data.ndim
        new_bins[axis] = value
        self.array_slicer.set_bins(cursor, new_bins, update)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    @record_history
    def set_bin_all(self, axis: int, value: int, update: bool = True) -> None:
        new_bins: list[int | None] = [None] * self.data.ndim
        new_bins[axis] = value
        for c in range(self.n_cursors):
            self.array_slicer.set_bins(c, new_bins, update)

    def make_cursors(self, n: int, colors: Iterable[QtGui.QColor | str] | None) -> None:
        # Used when restoring state to match the number of cursors
        while self.n_cursors > 1:
            self.remove_cursor(0)

        if colors is None:
            colors = [self.gen_cursor_color(i) for i in range(n)]
        else:
            colors = list(colors)

        if len(colors) != n:
            raise ValueError("Number of colors must match number of cursors")

        for clr in colors:
            self.add_cursor(color=clr)

        self.remove_cursor(0)
        self.refresh_all()

    @QtCore.Slot()
    @QtCore.Slot(object)
    @link_slicer
    @record_history
    def add_cursor(self, color: QtGui.QColor | str | None = None) -> None:
        self.array_slicer.add_cursor(self.current_cursor, update=False)
        if color is None:
            self.cursor_colors.append(self.gen_cursor_color(self.n_cursors - 1))
        else:
            self.cursor_colors.append(QtGui.QColor(color))

        self.current_cursor = self.n_cursors - 1
        for ax in self.axes:
            ax.add_cursor(update=False)
        self._colorbar.cb.level_change()
        self.refresh_current()
        self.sigCursorCountChanged.emit(self.n_cursors)
        self.sigCurrentCursorChanged.emit(self.current_cursor)

    @QtCore.Slot(int)
    @link_slicer
    @record_history
    def remove_cursor(self, index: int) -> None:
        index = index % self.n_cursors
        if self.n_cursors == 1:
            return
        self.array_slicer.remove_cursor(index, update=False)
        self.cursor_colors.pop(index)
        if self.current_cursor == index:
            if index == 0:
                self.current_cursor = 1
            self.current_cursor -= 1
        elif self.current_cursor > index:
            self.current_cursor -= 1

        for ax in self.axes:
            ax.remove_cursor(index)
        self.refresh_current()
        self.sigCursorCountChanged.emit(self.n_cursors)
        self.sigCurrentCursorChanged.emit(self.current_cursor)

    @QtCore.Slot()
    @record_history
    def remove_current_cursor(self) -> None:
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
        reverse: bool | None = None,
        high_contrast: bool | None = None,
        zero_centered: bool | None = None,
        levels_locked: bool | None = None,
        levels: tuple[float, float] | None = None,
        update: bool = True,
    ) -> None:
        if gamma is None and levels_locked is None and levels is None:
            # These will be handled in their respective methods or calling widgets
            self.sigWriteHistory.emit()

        if cmap is not None:
            self._colormap_properties["cmap"] = cmap
        if gamma is not None:
            self._colormap_properties["gamma"] = gamma
        if reverse is not None:
            self._colormap_properties["reverse"] = reverse
        if high_contrast is not None:
            self._colormap_properties["high_contrast"] = high_contrast
        if zero_centered is not None:
            self._colormap_properties["zero_centered"] = zero_centered
        if levels_locked is not None:
            self.levels_locked = levels_locked
        if levels is not None:
            self.levels = levels

        cmap = pg_colormap_powernorm(
            self._colormap_properties["cmap"],
            self._colormap_properties["gamma"],
            self._colormap_properties["reverse"],
            high_contrast=self._colormap_properties["high_contrast"],
            zero_centered=self._colormap_properties["zero_centered"],
        )
        for im in self._imageitems:
            im.set_pg_colormap(cmap, update=update)
        self.sigViewOptionChanged.emit()

    @QtCore.Slot(bool)
    def lock_levels(self, lock: bool) -> None:
        if lock != self.levels_locked:
            self.sigWriteHistory.emit()

        self._colormap_properties["levels_locked"] = lock

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

    def add_tool_window(self, tool: QtWidgets.QWidget) -> None:
        """Add and show a tool window to this slicer.

        The tool window is automatically cleared from memory when it is closed. Be sure
        to not pass a widget that is already associated with another parent.

        Parameters
        ----------
        tool
            The tool window widget to add.
        """
        uid: str = str(uuid.uuid4())
        tool.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self._associated_tools[uid] = tool

        tool.destroyed.connect(lambda: self._associated_tools.pop(uid))
        tool.show()

    @QtCore.Slot()
    def open_in_ktool(self) -> None:
        """Open the interactive momentum conversion tool."""
        from erlab.interactive.kspace import KspaceTool

        cmap_info = self.colormap_properties
        if isinstance(cmap_info["cmap"], str):
            cmap = cmap_info["cmap"]
            if cmap_info["reverse"]:
                cmap = cmap + "_r"
            gamma = cmap_info["gamma"]
        else:
            cmap = None
            gamma = 0.5

        self.add_tool_window(KspaceTool(self.data, cmap=cmap, gamma=gamma))

    def adjust_layout(
        self,
        horiz_pad: int = 45,
        vert_pad: int = 30,
        font_size: float = 11.0,
        r: tuple[float, float, float, float] = (1.2, 1.5, 3.0, 1.0),
    ) -> None:
        """Determine the padding and aspect ratios.

        Parameters
        ----------
        horiz_pad, vert_pad
            Reserved space for the x and y axes.
        font_size
            Font size of axis ticks in points.
        r
            4 numbers that determine the aspect ratio of the layout. See notes.

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

        valid_axis: tuple[tuple[Literal[0, 1], ...], ...] = (
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
        for split, sz in zip(self._splitters, sizes, strict=True):
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

    def _cursor_name(self, i: int) -> str:
        return f" Cursor {int(i)}"

    def _cursor_icon(self, i: int) -> QtGui.QIcon:
        img = QtGui.QImage(32, 32, QtGui.QImage.Format.Format_RGBA64)
        img.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(img)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(pg.mkColor(self.cursor_colors[i]))
        painter.drawEllipse(img.rect())
        painter.end()
        return QtGui.QIcon(QtGui.QPixmap.fromImage(img))

    @record_history
    def toggle_snap(self, value: bool | None = None) -> None:
        if value is None:
            value = not self.array_slicer.snap_to_data
        elif value == self.array_slicer.snap_to_data:
            return
        self.array_slicer.snap_to_data = value
        self.sigViewOptionChanged.emit()

    def changeEvent(self, evt: QtCore.QEvent | None) -> None:
        if evt is not None and evt.type() == QtCore.QEvent.Type.PaletteChange:
            style = self.qapp.style()
            if style is not None:
                self.qapp.setStyle(style.name())
        super().changeEvent(evt)


class ItoolCursorLine(pg.InfiniteLine):
    """A subclass of :class:`pyqtgraph.InfiniteLine` used in ImageTool."""

    _sigDragStarted = QtCore.Signal(object)  # :meta private:

    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)

    @property
    def plotItem(self) -> ItoolPlotItem:
        return self.parentItem().parentItem().parentItem()

    def setBounds(
        self, bounds: Sequence[np.floating], value: float | None = None
    ) -> None:
        if bounds[0] > bounds[1]:
            bounds = list(bounds)
            bounds.reverse()
        self.maxRange = bounds
        if value is None:
            value = self.value()
        self.setValue(value)

    def value(self) -> float:
        return float(super().value())

    def mouseDragEvent(self, ev: mouseEvents.MouseDragEvent) -> None:
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            not in QtWidgets.QApplication.keyboardModifiers()
        ):
            if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
                if ev.isStart():
                    self.moving = True
                    self.cursorOffset = self.pos() - self.mapToParent(
                        ev.buttonDownPos()
                    )
                    self.startPosition = self.pos()
                    self._sigDragStarted.emit(self)
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

    def mouseClickEvent(self, ev: mouseEvents.MouseClickEvent) -> None:
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            not in QtWidgets.QApplication.keyboardModifiers()
        ):
            super().mouseClickEvent(ev)
        else:
            self.setMouseHover(False)
            ev.ignore()

    def hoverEvent(self, ev) -> None:
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            not in QtWidgets.QApplication.keyboardModifiers()
        ):
            super().hoverEvent(ev)
        else:
            self.setMouseHover(False)

    def _computeBoundingRect(self):
        """CursorLine debugging."""
        _ = self.getViewBox().size()
        return super()._computeBoundingRect()


class ItoolCursorSpan(pg.LinearRegionItem):
    def __init__(self, *args, **kargs) -> None:
        kargs.setdefault("movable", False)
        super().__init__(*args, **kargs)

    def setRegion(self, rgn) -> None:
        # hides when region width is 0
        if rgn[1] == rgn[0]:
            self.setVisible(False)
        else:
            self.setVisible(True)
            super().setRegion(rgn)


class ItoolDisplayObject:
    def __init__(self, axes, cursor: int | None = None) -> None:
        self.axes = axes
        if cursor is None:
            cursor = 0
        self._cursor_index = int(cursor)
        self.qapp = QtGui.QGuiApplication.instance()

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
    def cursor_index(self, value: int) -> None:
        self._cursor_index = int(value)

    @property
    def sliced_data(self) -> xr.DataArray:
        with xr.set_options(keep_attrs=True):
            return self.array_slicer.xslice(self.cursor_index, self.display_axis)

    def refresh_data(self) -> None:
        pass


class ItoolPlotDataItem(ItoolDisplayObject, pg.PlotDataItem):
    def __init__(
        self,
        axes,
        cursor: int | None = None,
        is_vertical: bool = False,
        **kargs,
    ) -> None:
        pg.PlotDataItem.__init__(self, axes=axes, cursor=cursor, **kargs)
        ItoolDisplayObject.__init__(self, axes=axes, cursor=cursor)
        self.is_vertical = is_vertical
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

    def refresh_data(self) -> None:
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
    ) -> None:
        BetterImageItem.__init__(self, axes=axes, cursor=cursor, **kargs)
        ItoolDisplayObject.__init__(self, axes=axes, cursor=cursor)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

    def updateImage(self, *args, **kargs):
        defaults = {"autoLevels": not self.slicer_area.levels_locked}
        defaults.update(kargs)
        return self.setImage(*args, **defaults)

    @suppressnanwarning
    def refresh_data(self) -> None:
        ItoolDisplayObject.refresh_data(self)
        rect, img = self.array_slicer.slice_with_coord(
            self.cursor_index, self.display_axis
        )
        self.setImage(
            image=img, rect=rect, autoLevels=not self.slicer_area.levels_locked
        )

    def mouseDragEvent(self, ev: mouseEvents.MouseDragEvent) -> None:
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            in QtWidgets.QApplication.keyboardModifiers()
        ):
            ev.ignore()
        else:
            super().mouseDragEvent(ev)

    def mouseClickEvent(self, ev: mouseEvents.MouseClickEvent) -> None:
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            in QtWidgets.QApplication.keyboardModifiers()
        ):
            ev.ignore()
        else:
            super().mouseClickEvent(ev)


class ItoolPlotItem(pg.PlotItem):
    """A subclass of :class:`pyqtgraph.PlotItem` used in ImageTool.

    This class tracks axes and cursors for the image it displays.

    """

    _sigDragged = QtCore.Signal(object, object)  # :meta private:

    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        display_axis: tuple[int, ...],
        image: bool = False,
        image_cls=None,
        plotdata_cls=None,
        **item_kw,
    ) -> None:
        super().__init__(
            axisItems={a: BetterAxisItem(a) for a in ("left", "right", "top", "bottom")}
        )
        for act in ["Transforms", "Downsample", "Average", "Alpha", "Points"]:
            self.setContextMenuActionVisible(act, False)

        self.vb.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        for i in (0, 1):
            # Hide unnecessary menu items
            self.vb.menu.ctrl[i].linkCombo.setVisible(False)
            self.vb.menu.ctrl[i].label.setVisible(False)

        self.vb.menu.addSeparator()

        save_action = self.vb.menu.addAction("Save data as HDF5")
        save_action.triggered.connect(self.save_current_data)

        copy_code_action = self.vb.menu.addAction("Copy selection code")
        copy_code_action.triggered.connect(self.copy_selection_code)

        self.vb.menu.addSeparator()

        if image:
            itool_action = self.vb.menu.addAction("Open in new window")
            itool_action.triggered.connect(self.open_in_new_window)

            goldtool_action = self.vb.menu.addAction("Open in goldtool")
            goldtool_action.triggered.connect(self.open_in_goldtool)

            dtool_action = self.vb.menu.addAction("Open in dtool")
            dtool_action.triggered.connect(self.open_in_dtool)

            self.vb.menu.addSeparator()

            equal_aspect_action = self.vb.menu.addAction("Equal aspect ratio")
            equal_aspect_action.setCheckable(True)
            equal_aspect_action.setChecked(False)
            equal_aspect_action.toggled.connect(self.toggle_aspect_equal)

            def _update_aspect_lock_state() -> None:
                locked: bool = self.getViewBox().state["aspectLocked"] is not False
                if equal_aspect_action.isChecked() != locked:
                    equal_aspect_action.blockSignals(True)
                    equal_aspect_action.setChecked(locked)
                    equal_aspect_action.blockSignals(False)

            self.getViewBox().sigStateChanged.connect(_update_aspect_lock_state)

            self.vb.menu.addSeparator()

        self.slicer_area = slicer_area
        self.display_axis = display_axis

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
            self._sigDragged,
            delay=1 / 60,
            rateLimit=60,
            slot=self.process_drag,
        )
        if self.slicer_area.bench:
            self._time_start: float | None = None
            self._time_end: float | None = None
            self._single_queue = collections.deque([0.0], maxlen=9)
            self._next_queue = collections.deque([0.0], maxlen=9)

        if image:
            # Rotatable alignment guidelines
            self._guidelines_items: list[pg.GraphicsObject] = []
            self._guideline_actions: list[QtWidgets.QAction] = []
            self._action_group = QtWidgets.QActionGroup(self)

            self._guideline_angle: float = 0.0
            self._guideline_offset: list[float] = [0.0, 0.0]

            for i, text in enumerate(["None", "C2", "C4", "C6"]):
                qact = QtWidgets.QAction(text)
                qact.setCheckable(True)
                qact.toggled.connect(
                    lambda b, idx=i: self._set_guidelines(idx) if b else None
                )
                qact.setActionGroup(self._action_group)
                self._guideline_actions.append(qact)
            self._guideline_actions[0].setChecked(True)

            self._rotate_action = QtWidgets.QAction("Apply Rotation")

    @property
    def axis_dims(self) -> list[str | None]:
        dim_list: list[str | None] = [
            str(self.slicer_area.data.dims[ax]) for ax in self.display_axis
        ]
        if not self.is_image:
            if self.slicer_data_items[-1].is_vertical:
                dim_list = [None, *dim_list]
            else:
                dim_list = [*dim_list, None]
        return dim_list

    @property
    def is_independent(self) -> bool:
        return self.vb.state["linkedViews"] == [None, None]

    @property
    def current_data(self) -> xr.DataArray:
        return self.slicer_data_items[self.slicer_area.current_cursor].sliced_data

    @property
    def selection_code(self) -> str:
        return self.array_slicer.qsel_code(
            self.slicer_area.current_cursor, self.display_axis
        )

    @property
    def is_guidelines_visible(self) -> bool:
        return len(self._guidelines_items) != 0

    @QtCore.Slot()
    def open_in_new_window(self) -> None:
        if self.is_image:
            from erlab.interactive.imagetool import itool

            tool = cast(
                QtWidgets.QWidget | None, itool(self.current_data, execute=False)
            )
            if tool is not None:
                self.slicer_area.add_tool_window(tool)

    @QtCore.Slot()
    def open_in_goldtool(self) -> None:
        if self.is_image:
            data = self.current_data.T

            if "alpha" not in data.dims:
                QtWidgets.QMessageBox.critical(
                    None,
                    "Error",
                    "Data must have an 'alpha' dimension to be opened in GoldTool",
                )
                return

            from erlab.interactive.fermiedge import GoldTool

            self.slicer_area.add_tool_window(
                GoldTool(data, data_name="data" + self.selection_code, execute=False)
            )

    @QtCore.Slot()
    def open_in_dtool(self) -> None:
        if self.is_image:
            from erlab.interactive.derivative import DerivativeTool

            self.slicer_area.add_tool_window(
                DerivativeTool(
                    self.current_data.T, data_name="data" + self.selection_code
                )
            )

    def refresh_manual_range(self) -> None:
        if self.is_independent:
            return
        for dim, auto, rng in zip(
            self.axis_dims,
            self.vb.state["autoRange"],
            self.vb.state["viewRange"],
            strict=True,
        ):
            if dim is not None:
                if auto:
                    self.slicer_area.manual_limits.pop(dim, None)
                else:
                    self.slicer_area.manual_limits[dim] = rng

    def update_manual_range(self) -> None:
        if self.is_independent:
            return
        self.set_range_from(self.slicer_area.manual_limits)

    def set_range_from(self, limits: dict[str, list[float]], **kwargs) -> None:
        for dim, key in zip(self.axis_dims, ("xRange", "yRange"), strict=True):
            if dim is not None:
                with contextlib.suppress(KeyError):
                    kwargs[key] = limits[dim]
        if len(kwargs) != 0:
            self.setRange(**kwargs)

    @QtCore.Slot()
    def toggle_aspect_equal(self) -> None:
        vb = self.getViewBox()

        if vb.state["aspectLocked"] is False:
            vb.setAspectLocked(True, ratio=1.0)
        else:
            vb.setAspectLocked(False)

    def getMenu(self) -> QtWidgets.QMenu:
        return self.ctrlMenu

    def getViewBox(self) -> pg.ViewBox:
        # override for type hinting
        return self.vb

    def mouseDragEvent(
        self, ev: mouseEvents.MouseDragEvent | mouseEvents.MouseClickEvent
    ) -> None:
        modifiers = self.slicer_area.qapp.keyboardModifiers()
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier in modifiers
            and ev.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            ev.accept()
            if isinstance(ev, mouseEvents.MouseDragEvent):
                if ev.isStart():
                    self.slicer_area.sigWriteHistory.emit()
            else:
                self.slicer_area.sigWriteHistory.emit()

            self._sigDragged.emit(ev, modifiers)
            # self.process_drag((ev, modifiers))
        else:
            ev.ignore()

    @QtCore.Slot(tuple)
    @suppress_history
    def process_drag(
        self, sig: tuple[mouseEvents.MouseDragEvent, QtCore.Qt.KeyboardModifier]
    ) -> None:
        if self.slicer_area.bench:
            if self._time_end is not None and self._time_start is not None:
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

    def add_cursor(self, update: bool = True) -> None:
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
                autoDownsample=False,
                axisOrder="row-major",
                **self._item_kw,
            )
            if self.slicer_area.levels_locked:
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
        for c, s, ax in zip(cursors, spans, self.display_axis, strict=False):
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
            c._sigDragStarted.connect(lambda: self.slicer_area.sigWriteHistory.emit())

        if update:
            self.refresh_cursor(new_cursor)

    def index_of_line(self, line: ItoolCursorLine) -> int:
        for i, line_dict in enumerate(self.cursor_lines):
            for v in line_dict.values():
                if v == line:
                    return i
        raise ValueError("`line` is not a valid cursor.")

    def line_click(self, line: ItoolCursorLine) -> None:
        cursor = self.index_of_line(line)
        if cursor != self.slicer_area.current_cursor:
            self.slicer_area.set_current_cursor(cursor, update=True)

    def line_drag(self, line: ItoolCursorLine, value: float, axis: int) -> None:
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

    def remove_cursor(self, index: int) -> None:
        item = self.slicer_data_items.pop(index)
        self.removeItem(item)
        for line, span in zip(
            self.cursor_lines.pop(index).values(),
            self.cursor_spans.pop(index).values(),
            strict=True,
        ):
            self.removeItem(line)
            self.removeItem(span)
        for i, item in enumerate(self.slicer_data_items):
            item.cursor_index = i

    def refresh_cursor(self, cursor: int) -> None:
        for ax, line in self.cursor_lines[cursor].items():
            line.setBounds(
                self.array_slicer.lims_uniform[ax],
                self.array_slicer.get_value(cursor, ax, uniform=True),
            )
            self.cursor_spans[cursor][ax].setRegion(
                self.array_slicer.span_bounds(cursor, ax)
            )

    @QtCore.Slot(int)
    def set_guidelines(self, n: Literal[0, 1, 2, 3]) -> None:
        """Show rotating crosshairs for alignment."""
        if not self.is_image:
            return
        self._guideline_actions[n].setChecked(True)

    @QtCore.Slot()
    def remove_guidelines(self) -> None:
        """Hide rotating crosshairs."""
        self.set_guidelines(0)

    @QtCore.Slot(int)
    def _set_guidelines(self, n: Literal[0, 1, 2, 3]) -> None:
        if not self.is_image:
            return
        if n == 0:
            self._remove_guidelines()
            return

        old_pos: pg.Point | None = None
        if len(self._guidelines_items) != n + 1 and self.is_guidelines_visible:
            old_pos = self._guidelines_items[0].pos()
            old_angle = float(
                self._guidelines_items[0].angle - self._guidelines_items[0].offset
            )
            self._remove_guidelines()

        for w in make_crosshairs(n):
            self.addItem(w)
            self._guidelines_items.append(w)

        line = self._guidelines_items[0]

        if old_pos is not None:
            line.setPos(old_pos)
            line.setAngle(old_angle)

        def _print_angle():
            line_pos = line.pos()
            self._guideline_angle = line.angle_effective
            self._guideline_offset = [line_pos.x(), line_pos.y()]
            for i in range(2):
                self._guideline_offset[i] = float(
                    np.round(
                        self._guideline_offset[i],
                        self.array_slicer.get_significant(self.display_axis[i]),
                    )
                )

            self.setTitle(
                f"{self._guideline_angle}° "
                + str(tuple(self._guideline_offset)).replace("-", "−")
            )

        line.sigAngleChanged.connect(lambda: _print_angle())
        line.sigPositionChanged.connect(lambda: _print_angle())
        _print_angle()

    @QtCore.Slot()
    def _remove_guidelines(self) -> None:
        if not self.is_image:
            return
        for item in list(self._guidelines_items):
            self.removeItem(item)
            self._guidelines_items.remove(item)
        self._guideline_angle = 0.0
        self._guideline_offset = [0.0, 0.0]
        self.setTitle(None)

    def connect_signals(self) -> None:
        self.slicer_area.sigIndexChanged.connect(self.refresh_items_data)
        self.slicer_area.sigBinChanged.connect(self.refresh_items_data)
        self.slicer_area.sigShapeChanged.connect(self.update_manual_range)
        self.slicer_area.sigShapeChanged.connect(self.remove_guidelines)
        self.vb.sigRangeChanged.connect(self.refresh_manual_range)

    def disconnect_signals(self) -> None:
        self.slicer_area.sigIndexChanged.disconnect(self.refresh_items_data)
        self.slicer_area.sigBinChanged.disconnect(self.refresh_items_data)
        self.slicer_area.sigShapeChanged.disconnect(self.update_manual_range)
        self.slicer_area.sigShapeChanged.disconnect(self.remove_guidelines)
        self.vb.sigRangeChanged.disconnect(self.refresh_manual_range)

    @QtCore.Slot(int, object)
    def refresh_items_data(self, cursor: int, axes: tuple[int] | None = None) -> None:
        self.refresh_cursor(cursor)
        if axes is not None and all(elem in self.display_axis for elem in axes):
            # display_axis는 축 dim 표시하는거임. 즉 해당 축만 바뀌면 데이터 변화 없음
            return
        for item in self.slicer_data_items:
            if item.cursor_index != cursor:
                continue
            self.set_active_cursor(cursor)
            item.refresh_data()
        # TODO: autorange smarter
        self.vb.updateAutoRange()

    @QtCore.Slot()
    def refresh_labels(self) -> None:
        if self.is_image:
            label_kw = {
                a: self._get_label_unit(i)
                for a, i in zip(
                    ("top", "bottom", "left", "right"), (0, 0, 1, 1), strict=True
                )
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

    def set_active_cursor(self, index: int) -> None:
        if self.is_image:
            for i, item in enumerate(self.slicer_data_items):
                item.setVisible(i == index)

    @QtCore.Slot()
    def save_current_data(self) -> None:
        default_name: Hashable | None = None
        if self.slicer_area._data is not None:
            default_name = self.slicer_area._data.name
        if not default_name:
            default_name = "data"

        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter("xarray HDF5 Files (*.h5)")

        last_dir = pg.PlotItem.lastFileDir
        if not last_dir:
            last_dir = os.getcwd()

        dialog.setDirectory(os.path.join(last_dir, f"{default_name}.h5"))

        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            self.current_data.to_netcdf(
                filename, engine="h5netcdf", invalid_netcdf=True
            )
            pg.PlotItem.lastFileDir = os.path.dirname(filename)

    @QtCore.Slot()
    def copy_selection_code(self) -> None:
        if self.selection_code == "":
            QtWidgets.QMessageBox.critical(
                self, "Error", "Selection code is undefined for main image of 2D data."
            )
            return
        copy_to_clipboard(self.selection_code)

    @property
    def display_axis(self) -> tuple[int, ...]:
        return self._display_axis

    @display_axis.setter
    def display_axis(self, value: tuple[int, ...]) -> None:
        self._display_axis = value

    @property
    def slicer_area(self) -> ImageSlicerArea:
        _slicer_area = self._slicer_area()
        if _slicer_area:
            return _slicer_area
        raise LookupError("Parent was destroyed")

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea) -> None:
        self._slicer_area = weakref.ref(value)

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer


class ItoolColorBarItem(BetterColorBarItem):
    def __init__(self, slicer_area: ImageSlicerArea, **kwargs) -> None:
        self.slicer_area = slicer_area
        kwargs.setdefault(
            "axisItems",
            {a: BetterAxisItem(a) for a in ("left", "right", "top", "bottom")},
        )
        super().__init__(**kwargs)

        copy_action = self.vb.menu.addAction("Copy color limits to clipboard")
        copy_action.triggered.connect(self._copy_limits)

    @property
    def slicer_area(self) -> ImageSlicerArea:
        _slicer_area = self._slicer_area()
        if _slicer_area:
            return _slicer_area
        raise LookupError("Parent was destroyed")

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea) -> None:
        self._slicer_area = weakref.ref(value)

    @property
    def images(self):
        return [weakref.ref(x) for x in self.slicer_area._imageitems]

    @property
    def primary_image(self):
        return weakref.ref(self.slicer_area.main_image.slicer_data_items[0])

    @QtCore.Slot()
    def _copy_limits(self) -> str:
        return copy_to_clipboard(str(self.slicer_area.levels))

    def setImageItem(self, *args, **kwargs) -> None:
        self.slicer_area.sigViewOptionChanged.connect(self.limit_changed)

        self._span.blockSignals(True)
        self._span.setRegion(self.limits)
        self._span.blockSignals(False)
        self._span.sigRegionChangeStarted.connect(
            lambda: self.slicer_area.sigWriteHistory.emit()
        )
        self._span.sigRegionChanged.connect(self.level_change)
        self._span.sigRegionChangeFinished.connect(self.level_change_fin)
        self.color_changed()


class ItoolColorBar(pg.PlotWidget):
    def __init__(self, slicer_area: ImageSlicerArea, **cbar_kw) -> None:
        super().__init__(
            parent=slicer_area, plotItem=ItoolColorBarItem(slicer_area, **cbar_kw)
        )
        self.scene().sigMouseClicked.connect(self.mouseDragEvent)

    @property
    def cb(self) -> ItoolColorBarItem:
        return self.plotItem

    def set_dimensions(
        self,
        width: int = 30,
        horiz_pad: int | None = None,
        vert_pad: int | None = None,
        font_size: float = 11.0,
    ) -> None:
        self.cb.set_dimensions(horiz_pad, vert_pad, font_size)
        self.setFixedWidth(width)

    def setVisible(self, visible: bool) -> None:
        super().setVisible(visible)
        self.cb.setVisible(visible)
        self.cb._span.blockSignals(not visible)

        if visible:
            self.cb.setSpanRegion(self.cb.limits)
