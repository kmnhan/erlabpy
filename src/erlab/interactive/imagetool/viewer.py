"""Provides core functionality of ImageTool.

This module contains :class:`ImageSlicerArea` which handles the core functionality of
ImageTool, including the slicing and plotting of data.

"""

from __future__ import annotations

import collections
import contextlib
import copy
import functools
import importlib
import inspect
import itertools
import logging
import pathlib
import threading
import typing
import uuid
import weakref

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool._viewer_dialogs import (
    _AssociatedCoordsDialog,
    _CursorColorCoordDialog,
)

if typing.TYPE_CHECKING:
    import os
    from collections.abc import Callable, Collection, Hashable, Iterable

    import qtawesome

    from erlab.interactive.imagetool.plot_items import (
        ItoolGraphicsLayoutWidget,
        ItoolImageItem,
        ItoolPlotItem,
    )
    from erlab.interactive.imagetool.slicer import ArraySlicerState
else:
    import lazy_loader as _lazy

    qtawesome = _lazy.load("qtawesome")

logger = logging.getLogger(__name__)


class ColorMapState(typing.TypedDict):
    """A dictionary containing the colormap state of an `ImageSlicerArea` instance."""

    cmap: str | pg.ColorMap
    gamma: float
    reverse: bool
    high_contrast: bool
    zero_centered: bool
    levels_locked: bool
    levels: typing.NotRequired[tuple[float, float]]


class PlotItemState(typing.TypedDict):
    """A dictionary containing the state of a `PlotItem` instance."""

    vb_aspect_locked: bool | float
    vb_x_inverted: bool
    vb_y_inverted: bool
    vb_autorange: typing.NotRequired[tuple[bool, bool]]
    roi_states: typing.NotRequired[list[dict[str, typing.Any]]]


class ImageSlicerState(typing.TypedDict):
    """A dictionary containing the state of an `ImageSlicerArea` instance."""

    color: ColorMapState
    slice: ArraySlicerState
    current_cursor: int
    manual_limits: dict[str, list[float]]
    cursor_colors: list[str]
    file_path: typing.NotRequired[str | None]
    load_func: typing.NotRequired[tuple[str, dict[str, typing.Any], int] | None]
    splitter_sizes: typing.NotRequired[list[list[int]]]
    plotitem_states: typing.NotRequired[list[PlotItemState]]


suppressnanwarning = np.testing.suppress_warnings()
suppressnanwarning.filter(RuntimeWarning, r"All-NaN (slice|axis) encountered")


def _processed_ndim(darr: xr.DataArray) -> int:
    if darr.ndim == 1:
        nd = 2
    elif darr.ndim > 4:
        nd = len(tuple(s for s in darr.shape if s != 1))
    else:
        nd = darr.ndim
    return nd


def _supported_shape(darr: xr.DataArray) -> bool:
    return _processed_ndim(darr) in (2, 3, 4)


def _parse_dataset(ds: xr.Dataset) -> tuple[xr.DataArray, ...]:
    return tuple(d for d in ds.data_vars.values() if _supported_shape(d))


def _make_cursor_colors(
    clr: QtGui.QColor,
) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor, QtGui.QColor, QtGui.QColor]:
    """Given a color, return a tuple of colors used for cursors and spans.

    This function generates a set of colors based on the input color `clr` with
    pre-defined transparency levels.
    """
    clr_cursor = pg.mkColor(clr)
    clr_cursor_hover = pg.mkColor(clr)
    clr_span = pg.mkColor(clr)
    clr_span_edge = pg.mkColor(clr)

    clr_cursor.setAlphaF(0.75)
    clr_cursor_hover.setAlphaF(0.95)
    clr_span.setAlphaF(0.15)
    clr_span_edge.setAlphaF(0.35)

    return clr, clr_cursor, clr_cursor_hover, clr_span, clr_span_edge


def _parse_input(
    data: Collection[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
) -> list[xr.DataArray]:
    input_cls: str = data.__class__.__name__
    if isinstance(data, np.ndarray | xr.DataArray):
        data = (data,)
    elif isinstance(data, xr.Dataset):
        data = _parse_dataset(data)
    elif isinstance(data, xr.DataTree):
        data = tuple(
            itertools.chain.from_iterable(
                _parse_dataset(leaf.dataset) for leaf in data.leaves
            )
        )

    if len(data) == 0:
        raise ValueError(f"No valid data for ImageTool found in {input_cls}")

    if not isinstance(next(iter(data)), xr.DataArray | np.ndarray):
        raise TypeError(
            f"Unsupported input type {input_cls}. Expected DataArray, Dataset, "
            "DataTree, numpy array, or a list of DataArray or numpy arrays."
        )

    return [xr.DataArray(d) if not isinstance(d, xr.DataArray) else d for d in data]


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
                if obj._linking_proxy is not None:  # pragma: no branch
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
        arguments: dict[str, typing.Any],
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
        args: dict[str, typing.Any],
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
    """An interactive tool based on :mod:`pyqtgraph` for exploring 3D data.

    Parameters
    ----------
    parent
        Parent widget.
    data : DataArray or array-like
        Data to display. The data must have 2 to 4 dimensions.
    cmap
        Default colormap of the data.
    gamma : float, optional
        Default power law normalization of the colormap.
    high_contrast
        If `True`, the colormap is displayed in high contrast mode. This changes the
        behavior of the exponent scaling of the colormap. See
        :mod:`erlab.plotting.colors` for a detailed explanation of the difference.
    zero_centered
        If `True`, the normalization is applied symmetrically from the midpoint of the
        colormap.
    vmin
        Minimum value of the colormap.
    vmax
        Maximum value of the colormap.
    rad2deg
        If `True` and `data` is not `None`, converts some known angle coordinates to
        degrees. If an iterable of strings is given, only the coordinates that
        correspond to the given strings are converted.
    transpose
        If `True`, the main image is transposed before being displayed.
    bench
        Prints the fps on Ctrl + drag for debugging purposes.
    state
        Initial state containing the settings and cursor position.
    file_path
        Path to the file from which the data was loaded. If given, the file path is used
        to set the window title and when reloading the data. To successfully reload the
        data, ``load_func`` must also be provided.
    load_func
        3-tuple containing the function, a dictionary of keyword arguments, and the
        index of the data variable used when loading the data. The function is called
        when reloading the data. If using a data loader plugin, the function may be as a
        string representing the loader name. If the function always returns a single
        DataArray, the last element should be 0.

    Signals
    -------
    sigDataChanged()

    sigCurrentCursorChanged(index)

    sigViewOptionChanged()

    sigHistoryChanged()

    sigWriteHistory()

    sigCursorColorsChanged()

    sigDataEdited()
        Signal to track when the data has been modified by user actions.
    sigPointValueChanged(value)
        Signal emitted when the point value at the current cursor has been computed.
        Only emitted for dask-backed data.
    sigCursorCountChanged(n_cursors)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigIndexChanged(cursor, axes)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigBinChanged(cursor, axes)
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigShapeChanged()
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.
    sigTwinChanged()
        Inherited from :class:`erlab.interactive.slicer.ArraySlicer`.

    """

    @property
    def COLORS(self) -> tuple[QtGui.QColor, ...]:
        r""":class:`PySide6.QtGui.QColor`\ s for multiple cursors."""
        return tuple(
            QtGui.QColor(c) for c in erlab.interactive.options.model.colors.cursors
        )

    TWIN_COLORS: tuple[QtGui.QColor, ...] = (
        pg.mkColor("#ffa500"),
        pg.mkColor("#008080"),
        pg.mkColor("#8a2be2"),
        pg.mkColor("#ff69b4"),
        pg.mkColor("#bfff00"),
    )  #: :class:`PySide6.QtGui.QColor`\ s for twin plots.

    HORIZ_PAD: int = 45  #: Reserved space for the x axes in each plot.
    VERT_PAD: int = 30  #: Reserved space for the y axes in each plot.
    TICK_FONT_SIZE: float = 11.0  #: Font size of axis ticks in points.

    sigDataChanged = QtCore.Signal()  #: :meta private:
    sigCurrentCursorChanged = QtCore.Signal(int)  #: :meta private:
    sigViewOptionChanged = QtCore.Signal()  #: :meta private:
    sigHistoryChanged = QtCore.Signal()  #: :meta private:
    sigWriteHistory = QtCore.Signal()  #: :meta private:
    sigCursorColorsChanged = QtCore.Signal()  #: :meta private:
    sigDataEdited = QtCore.Signal()  #: :meta private:
    sigPointValueChanged = QtCore.Signal(float)  #: :meta private:

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

    @property
    def sigTwinChanged(self) -> QtCore.SignalInstance:
        """:meta private:"""  # noqa: D400
        return self.array_slicer.sigTwinChanged

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        data: xr.DataArray | npt.NDArray,
        cmap: str | pg.ColorMap | None = None,
        gamma: float | None = None,
        high_contrast: bool = False,
        zero_centered: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        rad2deg: bool | Iterable[str] = False,
        *,
        transpose: bool = False,
        bench: bool = False,
        state: ImageSlicerState | None = None,
        file_path: str | os.PathLike | None = None,
        load_func: tuple[Callable | str, dict[str, typing.Any], int] | None = None,
        image_cls=None,
        plotdata_cls=None,
        _in_manager: bool = False,
    ) -> None:
        super().__init__(parent)
        self.qapp = typing.cast(
            "QtWidgets.QApplication", QtWidgets.QApplication.instance()
        )
        from erlab.interactive.imagetool.plot_items import (
            ItoolColorBar,
            ItoolGraphicsLayoutWidget,
        )

        # Handle default values
        opts = erlab.interactive.options.model
        if cmap is None:
            cmap = opts.colors.cmap.name
            if opts.colors.cmap.reverse:
                cmap = f"{cmap}_r"
        if gamma is None:
            gamma = opts.colors.cmap.gamma

        self.initialize_actions()

        self._in_manager: bool = _in_manager  #: Internal flag for tools inside manager
        self._update_delayed: bool = _in_manager  #: Internal flag for delayed updates

        self._linking_proxy: SlicerLinkProxy | None = None

        self.bench: bool = bench

        # Stores ktool, dtool, goldtool, etc.
        self._associated_tools: dict[str, QtWidgets.QWidget] = {}
        self._assoc_tools_lock = threading.RLock()

        # Applied filter function
        self._applied_func: Callable[[xr.DataArray], xr.DataArray] | None = None

        # Queues to handle undo and redo
        self._prev_states: collections.deque[ImageSlicerState] = collections.deque(
            maxlen=1000
        )
        self._next_states: collections.deque[ImageSlicerState] = collections.deque(
            maxlen=1000
        )

        # Flag to prevent writing history when restoring state, must be set to True
        # after initialization is complete
        self._write_history = False

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
        if isinstance(cmap, str):  # pragma: no branch
            cmap = cmap.strip().strip("'").strip('"')
            if cmap.endswith("_r"):
                cmap = cmap.removesuffix("_r")
                cmap_reversed = True
            if cmap.startswith("cet_CET"):
                cmap = cmap.removeprefix("cet_")
        self._colormap_properties = {
            "cmap": cmap,
            "gamma": gamma,
            "reverse": cmap_reversed,
            "high_contrast": high_contrast,
            "zero_centered": zero_centered,
            "levels_locked": False,
        }

        self.manual_limits: dict[str, list[float]] = {}
        # Dictionary of current axes limits for each data dimension to ensure all plots
        # show the same range for a given dimension. The keys are the dimension names.

        pkw = {"image_cls": image_cls, "plotdata_cls": plotdata_cls}
        self._plots: tuple[ItoolGraphicsLayoutWidget, ...] = (
            ItoolGraphicsLayoutWidget(
                self,
                display_axis=(0, 1),
                axis_enabled=(1, 0, 0, 1),
                image=True,
                **pkw,
            ),
            ItoolGraphicsLayoutWidget(
                self,
                display_axis=(0,),
                axis_enabled=(1, 1, 0, 0),
                **pkw,
            ),
            ItoolGraphicsLayoutWidget(
                self,
                display_axis=(1,),
                axis_enabled=(0, 0, 1, 1),
                is_vertical=True,
                **pkw,
            ),
            ItoolGraphicsLayoutWidget(
                self,
                display_axis=(2,),
                axis_enabled=(0, 1, 1, 0),
                **pkw,
            ),
            ItoolGraphicsLayoutWidget(
                self,
                image=True,
                display_axis=(0, 2),
                axis_enabled=(1, 0, 0, 0),
                **pkw,
            ),
            ItoolGraphicsLayoutWidget(
                self,
                display_axis=(2, 1),
                axis_enabled=(0, 0, 0, 1),
                image=True,
                **pkw,
            ),
            ItoolGraphicsLayoutWidget(
                self,
                display_axis=(3,),
                axis_enabled=(0, 1, 1, 0),
                **pkw,
            ),
            ItoolGraphicsLayoutWidget(
                self,
                display_axis=(3, 2),
                axis_enabled=(0, 1, 1, 0),
                image=True,
                **pkw,
            ),
        )
        for i in (1, 4):
            self._splitters[2].addWidget(self._plots[i])
        for i in (6, 3, 7):
            self._splitters[3].addWidget(self._plots[i])
        self._splitters[5].addWidget(self._plots[0])
        for i in (5, 2):
            self._splitters[6].addWidget(self._plots[i])

        self._file_path: pathlib.Path | None = None
        self._load_func: tuple[Callable | str, dict[str, typing.Any], int] | None = None
        self.current_cursor: int = 0

        self.set_data(data, rad2deg=rad2deg, file_path=file_path, load_func=load_func)

        if self.bench:
            print("\n")

        self.high_contrast_act.setChecked(high_contrast)
        self.reverse_act.setChecked(cmap_reversed)
        self.zero_centered_act.setChecked(zero_centered)

        if vmin is not None or vmax is not None:
            if vmin is None:
                vmin = self.array_slicer.nanmin
            if vmax is None:
                vmax = self.array_slicer.nanmax
            self.set_colormap(levels_locked=True, levels=(vmin, vmax))

        if state is not None:
            self.state = state

        if transpose:
            self.transpose_main_image()

    @property
    def _associated_tools_list(self) -> list[QtWidgets.QWidget]:
        with self._assoc_tools_lock:
            return list(self._associated_tools.values())

    @property
    def parent_title(self) -> str:
        parent = self.parent()
        if isinstance(parent, QtWidgets.QWidget):  # pragma: no branch
            return parent.windowTitle()
        return ""

    @property
    def display_name(self) -> str:
        """Generate a display name for the slicer.

        Depending on the source of the data and the 'name' attribute of the underlying
        DataArray, the display name is generated differently.

        If nothing can be inferred, an empty string is returned.
        """
        name: str | None = typing.cast("str | None", self._data.name)
        info: str | None = None
        if self._file_path is not None:
            info = self._file_path.stem
        if self.watched_data_name is not None:
            info = self.watched_data_name

        if name is not None and name.strip() == "":
            # Name contains only whitespace
            name = None

        if name is None:
            disp_name = "" if info is None else info
        elif info is None or name == info:
            disp_name = f"{name}"
        else:
            disp_name = f"{name} ({info})"

        return disp_name

    @property
    def colormap_properties(self) -> ColorMapState:
        prop = copy.deepcopy(self._colormap_properties)

        prop["levels_locked"] = self.levels_locked  # history handled by lock_levels()
        if prop["levels_locked"]:
            prop["levels"] = copy.deepcopy(self.levels)
        return typing.cast("ColorMapState", prop)

    @property
    def state(self) -> ImageSlicerState:
        load_func = self._load_func
        if load_func is not None:
            fn = load_func[0]
            func_name = f"{fn.__module__}:{fn.__qualname__}" if callable(fn) else fn
            load_func = (func_name, *load_func[1:])
        return {
            "color": self.colormap_properties,
            "slice": self.array_slicer.state,
            "current_cursor": int(self.current_cursor),
            "manual_limits": copy.deepcopy(self.manual_limits),
            "splitter_sizes": self.splitter_sizes,
            "file_path": str(self._file_path) if self._file_path is not None else None,
            "load_func": load_func,
            "cursor_colors": [c.name() for c in self.cursor_colors],
            "plotitem_states": [p._serializable_state for p in self.axes],
        }

    @state.setter
    def state(self, state: ImageSlicerState) -> None:
        logger.debug("Restoring state...")
        if "splitter_sizes" in state:
            self.splitter_sizes = state["splitter_sizes"]

        plotitem_states = state.get("plotitem_states", None)
        if plotitem_states is not None:  # pragma: no branch
            for ax, plotitem_state in zip(self.axes, plotitem_states, strict=True):
                ax._serializable_state = plotitem_state
        logger.debug("Restored plotitem states")

        self.set_manual_limits(state.get("manual_limits", {}))
        logger.debug("Restored manual limits")

        self.make_cursors(state["cursor_colors"], update=False)
        logger.debug("Restored cursor number and colors")

        self.set_current_cursor(state["current_cursor"], update=False)
        logger.debug("Restored current cursor")

        self.array_slicer.state = state["slice"]
        logger.debug("Restored array slicer state")

        file_path = state.get("file_path", None)
        if file_path is not None:
            self._file_path = pathlib.Path(file_path)

        load_func = state.get("load_func", None)
        if load_func is not None:
            fn: str = load_func[0]
            if ":" in fn:
                self._load_func = load_func
                try:
                    mod_name, qual = fn.split(":")
                    mod = importlib.import_module(mod_name)
                    func_obj = mod
                    for attr in qual.split("."):
                        func_obj = getattr(func_obj, attr)
                    self._load_func = (
                        typing.cast("Callable", func_obj),
                        *load_func[1:],
                    )
                except Exception:
                    self._load_func = None
            elif fn in erlab.io.loaders:
                self._load_func = load_func
            else:
                self._load_func = None

        if file_path is not None:
            self.sigDataChanged.emit()
            logger.debug("Restored file path")

        # Restore colormap settings
        try:
            self.set_colormap(**state.get("color", {}), update=False)
        except Exception:
            erlab.utils.misc.emit_user_level_warning(
                "Failed to restore colormap settings, skipping"
            )
        logger.debug("Restored colormap settings")

        self.refresh_all()
        logger.debug("Refreshed after state restoration")

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
        if self._linking_proxy is not None:  # pragma: no branch
            return self._linking_proxy.children - {self}

        return weakref.WeakSet()

    @property
    def colormap(self) -> str | pg.ColorMap:
        return self.colormap_properties["cmap"]

    @property
    def levels_locked(self) -> bool:
        return self.lock_levels_act.isChecked()

    @levels_locked.setter
    def levels_locked(self, value: bool) -> None:
        self.lock_levels_act.setChecked(value)

    @property
    def levels(self) -> tuple[float, float]:
        return self._colorbar.cb.spanRegion()

    @levels.setter
    def levels(self, levels: tuple[float, float]) -> None:
        self._colorbar.cb.setSpanRegion(
            (
                max(levels[0], self.array_slicer.nanmin),
                min(levels[1], self.array_slicer.nanmax),
            )
        )

    @property
    def slices(self) -> tuple[ItoolPlotItem, ...]:
        match self.data.ndim:
            case 2:
                return ()
            case 3:
                return tuple(self.get_axes(ax) for ax in (4, 5))
            case 4:  # pragma: no branch
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
    def array_slicer(self) -> erlab.interactive.imagetool.slicer.ArraySlicer:
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

    @QtCore.Slot()
    def write_state(self) -> None:
        if not self._write_history:
            return

        last_state = self._prev_states[-1] if self.undoable else None
        curr_state = self.state.copy()

        # Don't store splitter sizes in history
        if last_state is not None:
            last_state.pop("splitter_sizes", None)
        curr_state.pop("splitter_sizes", None)

        if last_state is None or last_state != curr_state:
            # Only store state if it has changed
            self._prev_states.append(curr_state)
            self._next_states.clear()
            self.sigHistoryChanged.emit()

    @QtCore.Slot()
    @suppress_history
    def flush_history(self) -> None:
        """Clear the undo and redo history."""
        self._prev_states.clear()
        self._next_states.clear()
        self.sigHistoryChanged.emit()

    @QtCore.Slot()
    @link_slicer
    @suppress_history
    def undo(self) -> None:
        """Undo the most recent action."""
        if self.undoable:  # pragma: no branch
            self._next_states.append(self.state)
            self.state = self._prev_states.pop()
            self.sigHistoryChanged.emit()

    @QtCore.Slot()
    @link_slicer
    @suppress_history
    def redo(self) -> None:
        """Redo the most recently undone action."""
        if self.redoable:  # pragma: no branch
            self._prev_states.append(self.state)
            self.state = self._next_states.pop()
            self.sigHistoryChanged.emit()

    def history_states(self) -> tuple[list[ImageSlicerState], int]:
        states = list(self._prev_states)
        current_state = self.state.copy()
        if not states or states[-1] != current_state:
            states.append(current_state)
        current_index = len(states) - 1

        if self._next_states:
            states.extend(reversed(self._next_states))

        return states, current_index

    def go_to_history_index(self, index: int) -> None:
        """Go to a specific index in the history.

        Parameters
        ----------
        index
            Index in the history to go to. 0 corresponds to the current state. Positive
            indices point to the future, while negative indices point to the past.
        """
        if index == 0:
            return
        if index < 0:
            for _ in range(-index):
                self.undo_act.trigger()
        else:
            for _ in range(index):
                self.redo_act.trigger()

    def initialize_actions(self) -> None:
        """Initialize :class:`QtWidgets.QAction` instances."""
        self.reload_act = QtWidgets.QAction("&Reload Data", self)
        self.reload_act.setShortcut(QtGui.QKeySequence.StandardKey.Refresh)
        self.reload_act.triggered.connect(self.reload)
        self.reload_act.setToolTip("Reload data from the original source")
        self.reload_act.setIcon(QtGui.QIcon.fromTheme("view-refresh"))

        self.view_all_act = QtWidgets.QAction("View &All", self)
        self.view_all_act.setShortcut("Ctrl+A")
        self.view_all_act.triggered.connect(self.view_all)
        self.view_all_act.setToolTip("Reset view limits for all axes")
        self.view_all_act.setIcon(QtGui.QIcon.fromTheme("zoom-fit-best"))

        self.transpose_act = QtWidgets.QAction("&Transpose Main Image", self)
        self.transpose_act.setShortcut("T")
        self.transpose_act.triggered.connect(self.transpose_main_image)
        self.transpose_act.setToolTip("Transpose the main image")

        self.add_cursor_act = QtWidgets.QAction("&Add Cursor", self)
        self.add_cursor_act.setShortcut("Shift+A")
        self.add_cursor_act.triggered.connect(self.add_cursor)
        self.add_cursor_act.setToolTip("Add a new cursor")
        self.add_cursor_act.setIcon(QtGui.QIcon.fromTheme("list-add"))

        self.rem_cursor_act = QtWidgets.QAction("&Remove Cursor", self)
        self.rem_cursor_act.setShortcut("Shift+R")
        self.rem_cursor_act.setDisabled(True)
        self.rem_cursor_act.triggered.connect(self.remove_current_cursor)
        self.rem_cursor_act.setToolTip("Remove the current cursor")
        self.rem_cursor_act.setIcon(QtGui.QIcon.fromTheme("list-remove"))

        self.toggle_cursor_act = QtWidgets.QAction("Cursor Visibility", self)
        self.toggle_cursor_act.setShortcut("Shift+V")
        self.toggle_cursor_act.setCheckable(True)
        self.toggle_cursor_act.setChecked(True)
        self.toggle_cursor_act.setToolTip("Toggle visibility of all cursors")
        self.toggle_cursor_act.toggled.connect(self.toggle_cursor_visibility)

        self.cursor_color_act = QtWidgets.QAction("Edit Cursor Colors...", self)
        self.cursor_color_act.triggered.connect(self.edit_cursor_colors)

        self.cursor_colors_by_coord_act = QtWidgets.QAction(
            "Set Cursor Colors by Coordinate...", self
        )
        self.cursor_colors_by_coord_act.triggered.connect(
            self._set_cursor_colors_by_coord
        )
        self.cursor_colors_by_coord_act.setToolTip(
            "Set cursor colors based on a coordinate value"
        )

        self.undo_act = QtWidgets.QAction("&Undo", self)
        self.undo_act.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.undo_act.setDisabled(True)
        self.undo_act.triggered.connect(self.undo)
        self.undo_act.setToolTip("Undo the last action")
        self.undo_act.setIcon(QtGui.QIcon.fromTheme("edit-undo"))

        self.redo_act = QtWidgets.QAction("&Redo", self)
        self.redo_act.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        self.redo_act.setDisabled(True)
        self.redo_act.triggered.connect(self.redo)
        self.redo_act.setToolTip("Redo the last undone action")
        self.redo_act.setIcon(QtGui.QIcon.fromTheme("edit-redo"))

        self.center_act = QtWidgets.QAction("&Center Current Cursor", self)
        self.center_act.setShortcut("Shift+C")
        self.center_act.triggered.connect(self.center_cursor)
        self.center_act.setToolTip("Center the current cursor")

        self.center_all_act = QtWidgets.QAction("&Center All Cursors", self)
        self.center_all_act.setShortcut("Alt+Shift+C")
        self.center_all_act.triggered.connect(self.center_all_cursors)
        self.center_all_act.setToolTip("Center all cursors")

        self.reverse_act = QtWidgets.QAction("&Reverse Colormap", self)
        self.reverse_act.setShortcut("R")
        self.reverse_act.setCheckable(True)
        self.reverse_act.setToolTip("Reverse the colormap")
        self.reverse_act.toggled.connect(self.refresh_colormap)

        self.high_contrast_act = QtWidgets.QAction("High Contrast", self)
        self.high_contrast_act.setCheckable(True)
        self.high_contrast_act.setToolTip("Change gamma scaling mode")
        self.high_contrast_act.toggled.connect(self.refresh_colormap)

        self.zero_centered_act = QtWidgets.QAction("Centered Scaling", self)
        self.zero_centered_act.setCheckable(True)
        self.zero_centered_act.setToolTip("Apply symmetric scaling from the center")
        self.zero_centered_act.toggled.connect(self.refresh_colormap)

        self.lock_levels_act = QtWidgets.QAction("Lock Levels", self)
        self.lock_levels_act.setCheckable(True)
        self.lock_levels_act.setToolTip("Lock the colormap levels and show a colorbar")
        self.lock_levels_act.toggled.connect(self.lock_levels)

        self.ktool_act = QtWidgets.QAction("Open ktool", self)
        self.ktool_act.triggered.connect(self.open_in_ktool)
        self.ktool_act.setToolTip(
            "Open data in the interactive momentum conversion tool"
        )

        self.meshtool_act = QtWidgets.QAction("Open meshtool", self)
        self.meshtool_act.triggered.connect(self.open_in_meshtool)
        self.meshtool_act.setToolTip("Open data in the interactive mesh removal tool")

        self.associated_coords_act = QtWidgets.QAction(
            "Plot Associated Coordinates", self
        )
        self.associated_coords_act.triggered.connect(self._choose_associated_coords)
        self.associated_coords_act.setToolTip("Plot associated coordinates")

        self.compute_act = QtWidgets.QAction("Load Into Memory", self)
        self.compute_act.triggered.connect(self._compute_chunked)
        self.compute_act.setToolTip("Load the entire data into memory")

        self.chunk_auto_act = QtWidgets.QAction("Auto Chunk", self)
        self.chunk_auto_act.triggered.connect(self._auto_chunk)
        self.chunk_auto_act.setToolTip(
            "Automatically set the chunk size for dask-backed data"
        )

        self.chunk_act = QtWidgets.QAction("Chunk…", self)
        self.chunk_act.triggered.connect(self._edit_chunks)
        self.chunk_act.setToolTip("Set the chunk size for dask-backed data")

    @QtCore.Slot()
    def edit_cursor_colors(self) -> None:
        """Open a dialog to edit cursor colors."""
        dialog = erlab.interactive.colors.ColorCycleDialog(
            self.cursor_colors,
            parent=self,
            preview_cursors=True,
            default_colors=tuple(
                self.COLORS[i % len(self.COLORS)] for i in range(self.n_cursors)
            ),
        )
        dialog.setWindowTitle("Edit Cursor Colors")
        dialog.setModal(True)
        dialog.sigAccepted.connect(self.set_cursor_colors)
        self.add_tool_window(dialog, update_title=False, transfer_to_manager=False)

    @QtCore.Slot()
    def _choose_associated_coords(self) -> None:
        dialog = _AssociatedCoordsDialog(self)
        dialog.exec()

    @QtCore.Slot()
    def _set_cursor_colors_by_coord(self) -> None:
        dialog = _CursorColorCoordDialog(self)
        dialog.exec()

    @QtCore.Slot()
    def _history_changed(self) -> None:
        """Enable undo and redo actions based on the current history.

        This slot is triggered when the history changes.
        """
        self.undo_act.setEnabled(self.undoable)
        self.redo_act.setEnabled(self.redoable)

    @QtCore.Slot()
    def _cursor_count_changed(self) -> None:
        """Enable or disable the remove cursor action based on the number of cursors.

        This slot is triggered when the number of cursors changes.
        """
        self.rem_cursor_act.setDisabled(self.n_cursors == 1)
        self.refresh_colormap()

    @QtCore.Slot()
    def refresh_actions_enabled(self) -> None:
        """Refresh the enabled state of miscellaneous actions.

        This slot is triggered from the parent widget when the menubar containing the
        actions is about to be shown.
        """
        self.ktool_act.setEnabled(self.data.kspace._interactive_compatible)
        self.meshtool_act.setEnabled(
            all(dim in self.data.dims for dim in {"alpha", "eV"})
        )

    def connect_axes_signals(self) -> None:
        for ax in self.axes:
            ax.connect_signals()

    def disconnect_axes_signals(self) -> None:
        for ax in self.axes:
            ax.disconnect_signals()

    def connect_signals(self) -> None:
        self.connect_axes_signals()
        self.sigHistoryChanged.connect(self._history_changed)
        self.sigCursorCountChanged.connect(self._cursor_count_changed)
        self.sigDataChanged.connect(self.refresh_all)
        self.sigShapeChanged.connect(self.refresh_all)
        self.sigWriteHistory.connect(self.write_state)

        self.sigIndexChanged.connect(self._handle_refresh_dask)
        self.sigIndexChanged.connect(self._refresh_cursor_colors)
        self.sigBinChanged.connect(self._handle_refresh_dask)

        logger.debug("Connected signals")

    @QtCore.Slot(int, object)
    @QtCore.Slot(object, object)
    def _refresh_cursor_colors(
        self, cursor: int | tuple[int, ...], axes: tuple[int, ...] | None
    ) -> None:
        cursor_color_params = self.array_slicer._cursor_color_params
        if cursor_color_params is not None:
            dim_name, coord_name, cmap, reverse, vmin, vmax = cursor_color_params
            if (axes is not None) and (
                dim_name not in tuple(self.data.dims[ax] for ax in axes)
            ):
                return

            if isinstance(cursor, int):
                cursor = (cursor,)

            axis_idx = self.data.dims.index(dim_name)
            if coord_name == dim_name:
                target_coord = self.array_slicer.coords[axis_idx]
            else:
                target_coord = self.array_slicer.associated_coords[dim_name][
                    coord_name
                ][1]
            mn, mx = np.min(target_coord), np.max(target_coord)
            scale_factor = (vmax - vmin) / (mx - mn)
            cmap = erlab.interactive.colors.pg_colormap_from_name(cmap)

            for cursor_idx in cursor:
                value_idx: int = self.array_slicer.get_indices(cursor_idx)[axis_idx]
                value = (target_coord[value_idx] - mn) * scale_factor + vmin
                if reverse:
                    value = 1 - value
                color = cmap.map(value, mode=cmap.QCOLOR)
                for ax in self.axes:
                    self.cursor_colors[cursor_idx] = color
                    ax.set_cursor_color(cursor_idx, color)
            self.sigCursorColorsChanged.emit()

    @QtCore.Slot(int, object)
    @QtCore.Slot(object, object)
    def _handle_refresh_dask(
        self, cursor: int | tuple[int, ...], axes: tuple[int, ...] | None
    ) -> None:
        """Handle refresh for dask-backed data.

        This method is called when the data is dask-backed and a refresh is requested.
        It computes the necessary chunks and updates the plots accordingly.

        For non-dask data, this method does nothing. The updates are calculated by each
        PlotItem individually.

        This method exists to compute multiple slicing operations across different
        PlotItems in a single dask.compute() call.

        The implementation essentially loops over multiple PlotItems and executes what
        :meth:`ItoolPlotItem.refresh_items_data` should do. The lazy dask arrays are
        returned by :meth:`ItoolPlotItem.collect_dask_objects`.

        Parameters
        ----------
        cursor
            Index of the cursor to refresh.
        axes
            Tuple of axis indices to refresh. If `None`, all axes are refreshed.

        """
        if not self.data_chunked:
            return

        import dask

        if isinstance(cursor, int):
            cursor = (cursor,)

        obj_list = []
        axes_list = []
        coord_or_rect_list = []
        arrays_list_flat = []

        for ax in self.axes:
            objs, coord_or_rects, arrays = ax.collect_dask_objects(cursor, axes)
            if objs:
                obj_list.append(objs)
                axes_list.append(ax)
                coord_or_rect_list.append(coord_or_rects)
                arrays_list_flat.extend(arrays)

        # Also get the point value at the current cursor
        arrays_list_flat.append(
            self.array_slicer.point_value(self.current_cursor, binned=True)
        )

        if arrays_list_flat:
            arrays_list_flat = dask.compute(*arrays_list_flat)

        arrays_it = iter(arrays_list_flat)

        for ax in self.axes:
            for c in cursor:
                ax.refresh_cursor(c)

        for objs, ax, coord_or_rects in zip(
            obj_list, axes_list, coord_or_rect_list, strict=True
        ):
            if len(cursor) == 1:
                ax.set_active_cursor(cursor[0])
            ax.vb.blockSignals(True)
            for obj, coord_or_rect in zip(objs, coord_or_rects, strict=True):
                obj.update_data(coord_or_rect, next(arrays_it))
            ax.vb.blockSignals(False)

        self.sigPointValueChanged.emit(float(next(arrays_it)))

    def link(self, proxy: SlicerLinkProxy) -> None:
        proxy.add(self)

    def unlink(self) -> None:
        if self.is_linked:
            typing.cast("SlicerLinkProxy", self._linking_proxy).remove(self)

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
        with self._assoc_tools_lock:
            for tool in dict(self._associated_tools).values():
                tool.close()
                tool.deleteLater()

    @QtCore.Slot()
    @QtCore.Slot(tuple)
    @QtCore.Slot(tuple, bool)
    def refresh_all(
        self, axes: tuple[int, ...] | None = None, only_plots: bool = False
    ) -> None:
        self.sigIndexChanged.emit(tuple(c for c in range(self.n_cursors)), axes)
        if not only_plots:
            for ax in self.axes:
                ax.refresh_labels()
                ax.update_manual_range()  # Handle axis limits (in case of transpose)

    @QtCore.Slot(tuple)
    @link_slicer
    def refresh_current(self, axes: tuple[int, ...] | None = None) -> None:
        self.sigIndexChanged.emit(self.current_cursor, axes)

    @QtCore.Slot(int, list)
    @link_slicer
    def refresh(self, cursor: int, axes: tuple[int, ...] | None = None) -> None:
        self.sigIndexChanged.emit(cursor, axes)

    @QtCore.Slot()
    @link_slicer
    def view_all(self) -> None:
        self.manual_limits = {}
        for ax in self.axes:
            ax.enableAutoRange()

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
        data: xr.DataArray | npt.NDArray,
        rad2deg: bool | Iterable[str] = False,
        file_path: str | os.PathLike | None = None,
        load_func: tuple[Callable | str, dict[str, typing.Any], int] | None = None,
        auto_compute: bool = True,
    ) -> None:
        """Set the data to be displayed.

        Parameters
        ----------
        data
            The data to be displayed. If a `xarray.DataArray` is given, the dimensions
            and coordinates are used to determine the axes of the plots. If a
            :class:`xarray.Dataset` is given, the first data variable is used. If a
            :class:`numpy.ndarray` is given, it is converted to a `xarray.DataArray`
            with default dimensions.
        rad2deg
            If `True`, converts coords along dimensions that have angle-like names to
            degrees. If an iterable of strings is given, coordinates for dimensions that
            correspond to the given strings are converted.
        file_path
            Path to the file from which the data was loaded. If given, the file path is
            used to set the window title and when reloading the data. To successfully
            reload the data, ``load_func`` must also be provided.
        load_func
            3-tuple containing the function, a dictionary of keyword arguments, and the
            index of the data variable used when loading the data. The function is
            called when reloading the data. If using a data loader plugin, the function
            be given as a string representing the loader name. If the function always
            returns a single DataArray, the last element should be 0.
        auto_compute
            If `True` and the data is dask-backed, automatically compute the data if its
            size is below the threshold defined in options.

        """
        self._file_path = pathlib.Path(file_path) if file_path is not None else None

        if load_func is not None:
            func: Callable | str = load_func[0]
            func_instance = getattr(func, "__self__", None)
            if isinstance(func_instance, erlab.io.dataloader.LoaderBase):
                func = func_instance.name
            self._load_func = (func, *load_func[1:])
        else:
            self._load_func = None

        if hasattr(self, "_array_slicer") and hasattr(self, "_data"):
            n_cursors_old = self.n_cursors
            if isinstance(self._data, xr.DataArray):  # pragma: no branch
                self._data.close()
            del self._data
            self.disconnect_axes_signals()
        else:
            n_cursors_old = 1

        darr_list: list[xr.DataArray] = _parse_input(data)
        if len(darr_list) > 1:
            raise ValueError(
                "This object cannot be opened in a single window. "
                "Use the manager instead."
            )

        data = darr_list[0]
        if hasattr(data.data, "flags") and not data.data.flags["WRITEABLE"]:
            data = data.copy()

        if not rad2deg:
            self._data: xr.DataArray = data
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

        if (
            auto_compute
            and self.data_chunked
            and (self._data.nbytes * 1e-6)
            < erlab.interactive.options.model.io.dask.compute_threshold
        ):
            self._data = self._data.compute()

        # Save color limits so we may restore them later
        cached_levels: tuple[float, float] | None = None
        if self.levels_locked:
            cached_levels = copy.deepcopy(self.levels)

        ndim_changed: bool = True
        cursors_reset: bool = True
        try:
            if hasattr(self, "_array_slicer"):
                if self._array_slicer._obj.ndim == _processed_ndim(self._data):
                    ndim_changed = False
                cursors_reset = self._array_slicer.set_array(self._data, reset=True)

            else:
                self._array_slicer: erlab.interactive.imagetool.slicer.ArraySlicer = (
                    erlab.interactive.imagetool.slicer.ArraySlicer(self._data, self)
                )
                logger.debug("Initialized ArraySlicer")
        except Exception:
            if self._in_manager:
                # Let the manager handle the exception
                raise
            erlab.interactive.utils.MessageDialog.critical(
                self, "Error", "An error occurred while setting data"
            )
            self.set_data(xr.DataArray(np.zeros((2, 2))))
            return

        while self.n_cursors != n_cursors_old:
            self.array_slicer.add_cursor(update=False)

        self.connect_signals()

        if ndim_changed:
            self.adjust_layout()

        if self.current_cursor > self.n_cursors - 1:
            self.set_current_cursor(self.n_cursors - 1, update=False)
        if not self._update_delayed:
            self.sigDataChanged.emit()
            logger.debug("Data refresh triggered")

        # self.refresh_current()
        if not self._update_delayed:
            self.refresh_colormap()

        # Refresh colorbar and color limits
        self._colorbar.cb.setImageItem()
        self.lock_levels(self.levels_locked)
        if self.levels_locked and (cached_levels is not None) and (not cursors_reset):
            # If the levels were cached, restore them
            # This is needed if the data was reloaded and the levels were locked
            self.levels = cached_levels

        self.flush_history()

    def _update_if_delayed(self) -> None:
        if self._update_delayed:
            self._update_delayed = False
            self.sigDataChanged.emit()
            logger.debug("Data refresh triggered (delayed)")

            self.refresh_colormap()

    @property
    def reloadable(self) -> bool:
        """Check if the data can be reloaded from the file.

        The data can be reloaded if the data was loaded from a file that still exists
        and the data loader name is stored in the data attributes.

        Returns
        -------
        bool
            `True` if the data can be reloaded, `False` otherwise.
        """
        return (
            (self._file_path is not None)
            and self._file_path.exists()
            and self._load_func is not None
            and (callable(self._load_func[0]) or self._load_func[0] in erlab.io.loaders)
        )

    def _fetch_for_reload(self) -> xr.DataArray:
        file_path = self._file_path
        load_func = self._load_func
        if file_path is None or load_func is None:
            raise RuntimeError("Data cannot be reloaded")

        func = (
            load_func[0]
            if callable(load_func[0])
            else erlab.io.loaders[load_func[0]].load
        )
        reloaded = typing.cast(
            "xr.DataArray | xr.Dataset | xr.DataTree",
            func(file_path, **load_func[1]),
        )
        return _parse_input(reloaded)[load_func[2]]

    @QtCore.Slot()
    def reload(self) -> None:
        """Reload the data from the file it was loaded from, using the same loader.

        Silently fails if the data cannot be reloaded. If an error occurs while
        reloading the data, a message is shown to the user.

        See Also
        --------
        :attr:`reloadable`
        """
        if self.reloadable:  # pragma: no branch
            try:
                self.set_data(
                    self._fetch_for_reload(),
                    file_path=self._file_path,
                    load_func=self._load_func,
                )
            except Exception:
                erlab.interactive.utils.MessageDialog.critical(
                    self, "Error", "An error occurred while reloading data."
                )

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

        if update:  # pragma: no branch
            self.array_slicer.clear_val_cache()
            self.refresh_all(only_plots=True)

            # This will update colorbar limits if visible
            self.lock_levels(self.levels_locked)

    def apply_func(
        self, func: Callable[[xr.DataArray], xr.DataArray] | None, update: bool = True
    ) -> None:
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
        update
            If `True`, the plots are updated after setting the new values.

        """
        # self._data is original data passed to `set_data`
        # self.data is the current data transformed by ArraySlicer
        self._applied_func = func

        if self._applied_func is None:
            self.update_values(self._data, update=update)
        else:
            self.update_values(self._applied_func(self._data), update=update)

    def set_manual_limits(self, manual_limits: dict[str, list[float]]) -> None:
        """Set manual limits for the axes.

        Replaces the current manual limits with the given limits dictionary and updates
        all child axes accordingly.
        """
        self.manual_limits = copy.deepcopy(manual_limits)
        for ax in self.axes:
            ax.update_manual_range()

    def propagate_limit_change(self, axes: ItoolPlotItem) -> None:
        """Propagate manual limits changes to all linked slicers.

        Called when the limits of a child axes are changed by the user.

        This method first propagates the limits to all linked slicers (if any), and then
        triggers the update of the manual limits for all other axes. This ensures that
        all plots share the same limits for a given dimension.
        """
        # Apply current manual limits to all linked slicers
        if self._linking_proxy is not None:
            for target in self._linking_proxy.children.difference({self}):
                target.set_manual_limits(self.manual_limits)

        # Set manual limits for all other axes
        for ax in self.axes:
            if ax is not axes:
                ax.update_manual_range()

    def make_slice_dict(self) -> dict[Hashable, slice]:
        """Create a dictionary of slices for current manual limits.

        Returns
        -------
        dict
            A dictionary mapping dimension names to slices.
        """
        slice_dict: dict[Hashable, slice] = {}
        for k, v in self.manual_limits.items():
            ax_idx = self.data.dims.index(k)
            sig_digits = self.array_slicer.get_significant(ax_idx, uniform=True)
            bounds = (
                float(np.round(v[0], sig_digits)),
                float(np.round(v[1], sig_digits)),
            )
            start, end = min(bounds), max(bounds)
            if self.array_slicer.incs_uniform[ax_idx] < 0:
                start, end = end, start
            if sig_digits == 0:
                start, end = int(start), int(end)
            slice_dict[k] = slice(start, end)
        return slice_dict

    @property
    def watched_data_name(self) -> str | None:
        """Get the name of the watched data variable.

        Only applicable if in an ImageTool Manager and the data is linked to a watched
        variable in a notebook. Returns None if otherwise.
        """
        if self._in_manager:
            manager = self._manager_instance
            if manager:  # pragma: no branch
                wrapper = manager.wrapper_from_slicer_area(self)
                if wrapper:  # pragma: no branch
                    return wrapper._watched_varname
        return None

    @property
    def data_chunked(self) -> bool:
        """Check if the data is chunked (backed by dask).

        Returns
        -------
        bool
            `True` if the data is chunked, `False` otherwise.
        """
        return self._data.chunks is not None

    @QtCore.Slot()
    def _compute_chunked(self) -> None:
        """Load chunked data into memory.

        This method computes the entire data array and loads it into memory if the data
        is chunked.
        """
        if self.data_chunked:
            try:
                with erlab.interactive.utils.wait_dialog(self, "Computing…"):
                    self.set_data(self._data.compute())
            except Exception:
                erlab.interactive.utils.MessageDialog.critical(
                    self, "Error", "An error occurred while loading data into memory."
                )

    @QtCore.Slot()
    def _edit_chunks(self) -> None:
        """Open a dialog to set chunk sizes."""
        dlg = erlab.interactive.utils.ChunkEditDialog(self._data, parent=self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._set_chunks(dlg.result_chunks)

    @QtCore.Slot()
    def _auto_chunk(self) -> None:
        """Call ``.chunk("auto")`` on the underlying DataArray."""
        self._set_chunks("auto")

    def _set_chunks(self, chunks) -> None:
        """Set the chunk size of the underlying data.

        Parameters
        ----------
        chunks
            Chunk size to set. Passed to :meth:`xarray.DataArray.chunk`.
        """
        with erlab.interactive.utils.wait_dialog(self, "Setting Chunks…"):
            self.set_data(self._data.chunk(chunks), auto_compute=False)

    @QtCore.Slot(int, int)
    @link_slicer
    @record_history
    def swap_axes(self, ax1: int, ax2: int) -> None:
        self.array_slicer.swap_axes(ax1, ax2)

    @QtCore.Slot()
    def transpose_main_image(self) -> None:
        self.swap_axes(0, 1)

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
        if cursor is None:  # pragma: no branch
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
        if cursor is None:  # pragma: no branch
            cursor = self.current_cursor
        self.array_slicer.set_value(cursor, axis, value, update, uniform)

    @QtCore.Slot(int, int, bool)
    @link_slicer(indices=True, steps=True)
    @record_history
    def set_bin(
        self, axis: int, value: int, update: bool = True, cursor: int | None = None
    ) -> None:
        if cursor is None:  # pragma: no branch
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

    def make_cursors(
        self, colors: Iterable[QtGui.QColor | str], *, update: bool = True
    ) -> None:
        """Create cursors with the specified colors.

        Used when restoring the state of the slicer. All existing cursors are removed.
        """
        colors = list(colors)
        if len(colors) == 1 and self.n_cursors == 1:
            # Fast path for state restore. Avoid recreating the only cursor because it
            # is expensive and can be flaky on some Qt backends.
            color = pg.mkColor(colors[0])
            self.cursor_colors[0] = color
            for ax in self.axes:
                ax.set_cursor_colors(self.cursor_colors)
            if update:
                self.refresh_all()
            return

        while self.n_cursors > 1:
            self.remove_cursor(0, update=False)

        for clr in colors:
            self.add_cursor(color=clr, update=False)

        self.remove_cursor(0, update=False)
        if update:
            self.refresh_all()

    @QtCore.Slot()
    @QtCore.Slot(object)
    @link_slicer
    @record_history
    def add_cursor(
        self, color: QtGui.QColor | str | None = None, *, update: bool = True
    ) -> None:
        self.array_slicer.add_cursor(self.current_cursor, update=False)
        if color is None:
            self.cursor_colors.append(self.color_for_cursor(self.n_cursors - 1))
        else:
            self.cursor_colors.append(QtGui.QColor(color))

        self.current_cursor = self.n_cursors - 1
        for ax in self.axes:
            ax.add_cursor(update=False)
        if update:
            self._colorbar.cb.level_change()  # <- why is this required?
            self.refresh_current()
            self.sigCursorCountChanged.emit(self.n_cursors)
            self.sigCurrentCursorChanged.emit(self.current_cursor)

    @QtCore.Slot(int)
    @link_slicer
    @record_history
    def remove_cursor(self, index: int, *, update: bool = True) -> None:
        index = index % self.n_cursors
        if self.n_cursors == 1:
            return
        self.array_slicer.remove_cursor(index, update=False)
        self.cursor_colors.pop(index)
        if self.current_cursor == index:
            if index == 0:
                self.current_cursor = 1
            self.current_cursor -= 1
        elif self.current_cursor > index:  # pragma: no branch
            self.current_cursor -= 1

        for ax in self.axes:
            ax.remove_cursor(index)
        if update:
            self.refresh_current()
            self.sigCursorCountChanged.emit(self.n_cursors)
            self.sigCurrentCursorChanged.emit(self.current_cursor)

    @QtCore.Slot()
    @record_history
    def remove_current_cursor(self) -> None:
        self.remove_cursor(self.current_cursor)

    @QtCore.Slot()
    def toggle_cursor_visibility(self) -> None:
        """Toggle the visibility of the cursor lines."""
        for ax in self.axes:
            ax.set_cursor_visible(self.toggle_cursor_act.isChecked())

    def color_for_cursor(self, index: int) -> QtGui.QColor:
        """Pick a cursor color based on the index."""
        n_color: int = len(self.COLORS)
        color = self.COLORS[index % n_color]
        while color in self.cursor_colors:
            # Try to find a unique color
            color = self.COLORS[index % n_color]
            if self.n_cursors > n_color:
                break
            index += 1
        return color

    @QtCore.Slot(tuple)
    def set_cursor_colors(self, colors: Iterable[QtGui.QColor]) -> None:
        """Set the colors of the cursors.

        Parameters
        ----------
        colors
            An iterable of colors to set for the cursors.
        """
        self.array_slicer._cursor_color_params = None
        self.cursor_colors = [pg.mkColor(c) for c in colors]
        for ax in self.axes:
            ax.set_cursor_colors(self.cursor_colors)
        self.sigCursorColorsChanged.emit()

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
            prop = copy.deepcopy(self._colormap_properties)
            new_reverse = self.reverse_act.isChecked()
            new_high_contrast = self.high_contrast_act.isChecked()
            new_zero_centered = self.zero_centered_act.isChecked()
            if any(
                (
                    prop["reverse"] != new_reverse,
                    prop["high_contrast"] != new_high_contrast,
                    prop["zero_centered"] != new_zero_centered,
                )
            ):
                self._colormap_properties["reverse"] = new_reverse
                self._colormap_properties["high_contrast"] = new_high_contrast
                self._colormap_properties["zero_centered"] = new_zero_centered

        if cmap is not None:
            self._colormap_properties["cmap"] = cmap
        if gamma is not None:
            self._colormap_properties["gamma"] = float(gamma)
        if reverse is not None:
            # Don't block signals here to trigger updates to linked buttons.
            # Will be called twice, but unnoticable
            self.reverse_act.setChecked(reverse)
        if high_contrast is not None:
            self.high_contrast_act.setChecked(high_contrast)
        if zero_centered is not None:
            self.zero_centered_act.setChecked(zero_centered)
        if levels_locked is not None:
            self.levels_locked = levels_locked
        if levels is not None:
            self.levels = levels

        properties = self.colormap_properties
        cmap = erlab.interactive.colors.pg_colormap_powernorm(
            properties["cmap"],
            properties["gamma"],
            properties["reverse"],
            high_contrast=properties["high_contrast"],
            zero_centered=properties["zero_centered"],
        )
        for im in self._imageitems:
            im.set_pg_colormap(cmap, update=update)
        self.sigViewOptionChanged.emit()

    @QtCore.Slot()
    def refresh_colormap(self) -> None:
        self.set_colormap(update=True)
        logger.debug("Colormap refreshed")

    @QtCore.Slot(bool)
    def lock_levels(self, lock: bool) -> None:
        if lock != self.levels_locked:
            self.sigWriteHistory.emit()
            self.levels_locked = lock

        if self.levels_locked:
            levels = self.array_slicer.limits
            self._colorbar.cb.setLimits(levels)
        for im in self._imageitems:
            if self.levels_locked:
                im.setLevels(levels, update=False)
            else:
                im.levels = None
            im.updateImage()

        self._colorbar.setVisible(self.levels_locked)
        self.sigViewOptionChanged.emit()

    @property
    def _manager_instance(
        self,
    ) -> erlab.interactive.imagetool.manager.ImageToolManager | None:
        return erlab.interactive.imagetool.manager._manager_instance

    def remove_from_manager(self) -> None:
        """Remove this ImageTool from the manager, if it is in one."""
        if self._in_manager:
            manager = self._manager_instance
            if manager:  # pragma: no branch
                index = manager.index_from_slicer_area(self)
                if index is not None:  # pragma: no branch
                    msg_box = QtWidgets.QMessageBox(self)
                    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                    msg_box.setText("Remove window?")
                    msg_box.setInformativeText(
                        f"The ImageTool window at index {index} will be removed. "
                        "This cannot be undone."
                    )
                    msg_box.setStandardButtons(
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.Cancel
                    )
                    msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

                    if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                        erlab.interactive.utils.single_shot(
                            manager, 0, lambda: manager.remove_imagetool(index)
                        )

    def add_tool_window(
        self,
        widget: QtWidgets.QWidget,
        update_title: bool = True,
        transfer_to_manager: bool = True,
    ) -> None:
        """Save a reference to an additional window widget.

        This is mainly used for handling tool windows such as goldtool and dtool.

        The tool window is cleared from memory immediately when it is closed. Closing
        the main window will close all associated tool windows.

        Only pass widgets that are not associated with a parent widget.

        If the parent ImageTool is in the manager, the widget is transferred to the
        manager instead

        Parameters
        ----------
        widget
            The widget to add.
        update_title
            If `True`, the window title is updated to include the parent title. If
            `False`, the window title is not changed.
        transfer_to_manager
            If `True`, the ownership of the widget is transferred to the manager if the
            parent ImageTool is in the manager. Use `False` for dialog windows that
            should not be managed by the manager.
        """
        if update_title:
            old_title = widget.windowTitle().strip()
            new_title = self.parent_title.strip()
            if new_title != "" and old_title != "":
                new_title += f" - {old_title}"
            widget.setWindowTitle(new_title)

        if transfer_to_manager and self._in_manager:
            manager = self._manager_instance
            if manager:  # pragma: no branch
                if isinstance(widget, erlab.interactive.utils.ToolWindow):
                    manager._add_childtool_from_slicerarea(widget, self)
                else:
                    manager.add_widget(widget)
                return

        uid: str = str(uuid.uuid4())
        with self._assoc_tools_lock:
            self._associated_tools[uid] = widget  # Store reference to prevent gc

        old_close_event = widget.closeEvent

        def new_close_event(event: QtGui.QCloseEvent) -> None:
            with self._assoc_tools_lock:
                if uid in self._associated_tools:
                    tool = self._associated_tools.pop(uid)
                    tool.deleteLater()
            old_close_event(event)

        widget.closeEvent = new_close_event  # type: ignore[assignment]
        widget.show()
        widget.raise_()
        widget.activateWindow()

    @QtCore.Slot()
    def open_in_ktool(self) -> None:
        """Open the interactive momentum conversion tool."""
        cmap_info = self.colormap_properties
        if isinstance(cmap_info["cmap"], str):
            cmap = cmap_info["cmap"]
            if cmap_info["reverse"]:
                cmap = cmap + "_r"
            gamma = cmap_info["gamma"]
        else:
            cmap = None
            gamma = 0.5

        self.add_tool_window(
            erlab.interactive.ktool(
                self.data,
                cmap=cmap,
                gamma=gamma,
                data_name=self.watched_data_name,
                execute=False,
            )
        )

    @QtCore.Slot()
    def open_in_meshtool(self) -> None:
        """Open the interactive mesh removal tool."""
        self.add_tool_window(
            erlab.interactive.meshtool(
                self.data, data_name=self.watched_data_name, execute=False
            )
        )

    def adjust_layout(
        self,
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
        invalid: list[int] = []  # axes to hide.
        r0, r1, r2, r3 = r

        # TODO: automate this based on ItoolPlotItem.display_axis
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
            full = r0 + r1 - d
            sizes[3] = (full / 4, full / 4, full / 2)
        for split, sz in zip(self._splitters, sizes, strict=True):
            split.setSizes(tuple(round(s * scale) for s in sz))

        for i in range(8):
            visible: bool = i not in invalid
            self.get_axes_widget(i).setVisible(visible)

        # reserve space, only hide plotItem
        self.get_axes(3).setVisible(self.data.ndim != 2)

        self._colorbar.set_dimensions(
            width=self.HORIZ_PAD + 30,
            horiz_pad=None,
            vert_pad=self.VERT_PAD,
            font_size=self.TICK_FONT_SIZE,
        )

        # Remove all ROI since they may not be valid anymore
        for ax in self.axes:
            ax.clear_rois()

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

    @link_slicer
    @record_history
    def toggle_snap(self, value: bool | None = None) -> None:
        if value is None:
            value = not self.array_slicer.snap_to_data
        elif value == self.array_slicer.snap_to_data:
            return
        self.array_slicer.snap_to_data = value

    def changeEvent(self, evt: QtCore.QEvent | None) -> None:
        if evt is not None and evt.type() == QtCore.QEvent.Type.PaletteChange:
            style = self.qapp.style()
            if style is not None:
                self.qapp.setStyle(style.name())
        super().changeEvent(evt)
