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
import os
import pathlib
import threading
import time
import typing
import uuid
import weakref

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import xarray as xr
from pyqtgraph.GraphicsScene import mouseEvents
from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable, Iterable, Sequence

    import dask.array
    import matplotlib.colors
    import qtawesome

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
class ItoolGraphicsLayoutWidget(pg.PlotWidget):
    # Unsure of whether to subclass GraphicsLayoutWidget or PlotWidget at the moment
    # Will need to run some benchmarks in the future
    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        display_axis: tuple[int] | tuple[int, int],
        axis_enabled: tuple[int, int, int, int],
        image: bool = False,
        **item_kw,
    ) -> None:
        # super().__init__()
        # self.ci.layout.setContentsMargins(0, 0, 0, 0)
        # self.ci.layout.setSpacing(3)
        # self.plotItem = ItoolPlotItem(slicer_area, display_axis, image, **item_kw)
        # self.addItem(self.plotItem)

        super().__init__(
            plotItem=ItoolPlotItem(
                slicer_area,
                display_axis,
                axis_enabled=axis_enabled,
                image=image,
                **item_kw,
            )
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


class _AssociatedCoordsDialog(QtWidgets.QDialog):
    def __init__(self, slicer_area: ImageSlicerArea) -> None:
        super().__init__(slicer_area)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        self._slicer_area = weakref.ref(slicer_area)

        self._layout = QtWidgets.QFormLayout()
        self.setLayout(self._layout)

        self._checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        for dim, coords in slicer_area.array_slicer.associated_coords.items():
            for k in coords:
                self._checks[k] = QtWidgets.QCheckBox(str(k))
                self._checks[k].setChecked(
                    k in slicer_area.array_slicer.twin_coord_names
                )
                self._layout.addRow(self._checks[k], QtWidgets.QLabel(f"({dim})"))

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        self._layout.addRow(self._button_box)

    def exec(self) -> int:
        if len(self._checks) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Associated Coordinates",
                "No 1D non-dimension coordinates were found in the data.",
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super().exec()

    @QtCore.Slot()
    def accept(self) -> None:
        slicer_area = self._slicer_area()
        if slicer_area:  # pragma: no branch
            slicer_area.array_slicer.twin_coord_names = {
                coord for coord, check in self._checks.items() if check.isChecked()
            }
        super().accept()


class _CursorColorCoordDialog(QtWidgets.QDialog):
    def __init__(self, slicer_area: ImageSlicerArea) -> None:
        super().__init__(slicer_area)
        self._slicer_area = weakref.ref(slicer_area)
        self.setup_ui()

    def update_params(self) -> None:
        slicer_area = self._slicer_area()
        if slicer_area:  # pragma: no branch
            cursor_color_params = slicer_area.array_slicer._cursor_color_params
            if cursor_color_params is not None:
                _, coord_name, cmap, reverse, vmin, vmax = cursor_color_params
                self.choose_coord(coord_name)
                self.cmap_combo.setCurrentText(cmap)
                self.reverse_check.setChecked(reverse)
                self.start_spin.setValue(vmin)
                self.stop_spin.setValue(vmax)
            else:
                self.main_group.setChecked(False)

    def choose_coord(self, coord_name: Hashable) -> None:
        self.coord_combo.setCurrentText(str(coord_name))

    def get_checked_coord_name(self) -> tuple[Hashable, Hashable] | None:
        slicer_area = self._slicer_area()
        if not slicer_area:  # pragma: no cover
            return None
        coord_name = self.coord_combo.currentText()
        for dim_name in slicer_area.data.dims:
            if coord_name == str(dim_name):
                return dim_name, dim_name
        for dim, coords in slicer_area.array_slicer.associated_coords.items():
            for k in coords:
                if coord_name == str(k):
                    return dim, k
        return None

    def setup_ui(self):
        slicer_area = self._slicer_area()
        if not slicer_area:  # pragma: no cover
            return

        self.layout_ = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout_)

        self.main_group = QtWidgets.QGroupBox("Set cursor color by coordinate", self)
        self.layout_.addWidget(self.main_group)
        main_layout = QtWidgets.QVBoxLayout()
        self.main_group.setLayout(main_layout)
        self.main_group.setCheckable(True)

        # Coord selection
        coord_layout = QtWidgets.QFormLayout()
        main_layout.addLayout(coord_layout)

        self.coord_combo = QtWidgets.QComboBox()
        coord_layout.addRow("Coordinate", self.coord_combo)

        for name in slicer_area.data.dims:
            self.coord_combo.addItem(str(name))

        for coords in slicer_area.array_slicer.associated_coords.values():
            for k in coords:
                self.coord_combo.addItem(str(k))

        # Colormap selection
        cmap_group = QtWidgets.QGroupBox("Colormap parameters", self)
        main_layout.addWidget(cmap_group)
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_group.setLayout(cmap_layout)

        self.cmap_combo = erlab.interactive.colors.ColorMapComboBox(self)
        self.cmap_combo.setToolTip("Select a colormap to sample colors from")
        self.cmap_combo.setDefaultCmap("coolwarm")
        cmap_layout.addWidget(self.cmap_combo)

        self.reverse_check = QtWidgets.QCheckBox("Reverse", self)
        self.reverse_check.setToolTip("Reverse the colormap")
        cmap_layout.addWidget(self.reverse_check)

        cmap_layout.addStretch()
        cmap_layout.addWidget(QtWidgets.QLabel("Range:"))
        self.start_spin = QtWidgets.QDoubleSpinBox(self)
        self.start_spin.setRange(0.0, 1.0)
        self.start_spin.setDecimals(2)
        self.start_spin.setSingleStep(0.1)
        self.start_spin.setValue(0.1)
        self.start_spin.setToolTip("Start of the colormap")
        cmap_layout.addWidget(self.start_spin)

        self.stop_spin = QtWidgets.QDoubleSpinBox(self)
        self.stop_spin.setRange(0.0, 1.0)
        self.stop_spin.setDecimals(2)
        self.stop_spin.setSingleStep(0.1)
        self.stop_spin.setValue(0.9)
        self.stop_spin.setToolTip("End of the colormap")
        cmap_layout.addWidget(self.stop_spin)
        cmap_layout.addStretch()

        # Bottom layout
        bottom_layout = QtWidgets.QHBoxLayout()
        self.layout_.addLayout(bottom_layout)

        # Dialog buttons
        self.button_box = QtWidgets.QDialogButtonBox()
        btn_ok = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Ok,
        )
        btn_cancel = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        )
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        self.layout_.addWidget(self.button_box)
        self.update_params()

    def accept(self) -> None:
        slicer_area = self._slicer_area()
        if slicer_area:  # pragma: no branch
            dim_and_coord_names = self.get_checked_coord_name()
            if dim_and_coord_names is None:
                slicer_area.array_slicer._cursor_color_params = None
            else:
                dim_name, coord_name = dim_and_coord_names
                slicer_area.array_slicer._cursor_color_params = (
                    dim_name,
                    coord_name,
                    self.cmap_combo.currentText(),
                    self.reverse_check.isChecked(),
                    self.start_spin.value(),
                    self.stop_spin.value(),
                )
            slicer_area._refresh_cursor_colors(
                tuple(i for i in range(slicer_area.n_cursors)), None
            )
        super().accept()


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
        when reloading the data. If from a data loader plugin, the function may be given
        as a string representing the loader name. If the function always returns a
        single DataArray, the last element should be 0.

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
            called when reloading the data. If from a data loader plugin, the function
            may be given as a string representing the loader name. If the function
            always returns a single DataArray, the last element should be 0.
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
        self.sigDataChanged.emit()
        logger.debug("Data refresh triggered")

        # self.refresh_current()
        self.refresh_colormap()

        # Refresh colorbar and color limits
        self._colorbar.cb.setImageItem()
        self.lock_levels(self.levels_locked)
        if self.levels_locked and (cached_levels is not None) and (not cursors_reset):
            # If the levels were cached, restore them
            # This is needed if the data was reloaded and the levels were locked
            self.levels = cached_levels

        self.flush_history()

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
            start, end = (
                float(np.round(v[0], sig_digits)),
                float(np.round(v[1], sig_digits)),
            )
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
                        QtCore.QTimer.singleShot(
                            0, lambda: manager.remove_imagetool(index)
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

                if self.moving:
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
    """Parent class for sliced data.

    Stores the axes and cursor index for the object, and retrieves the sliced data from
    :class:`erlab.interactive.imagetool.slicer.ArraySlicer` when needed.
    """

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
    def array_slicer(self) -> erlab.interactive.imagetool.slicer.ArraySlicer:
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
            sliced = self.array_slicer.xslice(self.cursor_index, self.display_axis)
            if sliced.name is not None and sliced.name != "":
                sliced = sliced.rename(f"{sliced.name} Sliced")
            return sliced

    def fetch_new_data(
        self,
    ) -> tuple[
        npt.NDArray[np.floating] | tuple[float, float, float, float],
        npt.NDArray[np.floating] | dask.array.Array,
    ]:
        raise NotImplementedError

    def update_data(self, *args) -> None:
        raise NotImplementedError

    @suppressnanwarning
    def refresh_data(self) -> None:
        self.update_data(*self.fetch_new_data())


def _pad_1d_plot(
    coord: npt.NDArray, values: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    delta = 1.0 if len(coord) < 2 else coord[1] - coord[0]
    pad = 0.5 * delta
    return np.r_[coord[0] - pad, coord, coord[-1] + pad], np.r_[np.nan, values, np.nan]


class ItoolPlotDataItem(ItoolDisplayObject, pg.PlotDataItem):
    """Display a 1D slice of data in a plot."""

    def __init__(
        self, axes, cursor: int | None = None, is_vertical: bool = False, **kargs
    ) -> None:
        pg.PlotDataItem.__init__(self, axes=axes, cursor=cursor, **kargs)
        ItoolDisplayObject.__init__(self, axes=axes, cursor=cursor)
        self.is_vertical: bool = is_vertical
        self.normalize: bool = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

    def fetch_new_data(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | dask.array.Array]:
        return self.array_slicer.slice_with_coord(self.cursor_index, self.display_axis)

    def update_data(self, coord, vals) -> None:
        if self.normalize:
            avg = np.nanmean(vals)
            if not np.isnan(avg):  # pragma: no branch
                vals = vals / avg

        coord, vals = _pad_1d_plot(coord, vals)

        if self.is_vertical:
            self.setData(vals, coord)
        else:
            self.setData(coord, vals)


class ItoolImageItem(ItoolDisplayObject, erlab.interactive.colors.BetterImageItem):
    """Display a 2D slice of data as an image."""

    def __init__(self, axes, cursor: int | None = None, **kargs) -> None:
        erlab.interactive.colors.BetterImageItem.__init__(
            self, axes=axes, cursor=cursor, **kargs
        )
        ItoolDisplayObject.__init__(self, axes=axes, cursor=cursor)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

    def updateImage(self, *args, **kargs):
        defaults = {"autoLevels": not self.slicer_area.levels_locked}
        defaults.update(kargs)
        return self.setImage(*args, **defaults)

    def fetch_new_data(
        self,
    ) -> tuple[
        tuple[float, float, float, float],
        npt.NDArray[np.floating] | dask.array.Array,
    ]:
        return self.array_slicer.slice_with_coord(self.cursor_index, self.display_axis)

    def update_data(self, rect, img) -> None:
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


class _OptionKeyMenuFilter(QtCore.QObject):
    """Filter to catch and modify the text of menu items.

    Adds a '(Crop)' suffix to actions if the option key is down.
    """

    def __init__(self, menu: QtWidgets.QMenu, actions: list[QtWidgets.QAction]) -> None:
        super().__init__(menu)
        self.menu = menu

        self._actions: dict[str, QtWidgets.QAction] = {}
        for act in actions:
            self._actions[act.text()] = act

    def eventFilter(
        self, obj: QtCore.QObject | None = None, event: QtCore.QEvent | None = None
    ) -> bool:
        if (
            (event is not None)
            and isinstance(obj, QtWidgets.QMenu)
            and obj.isVisible()
            and event.type()
            in (
                QtCore.QEvent.Type.Show,
                QtCore.QEvent.Type.KeyPress,
                QtCore.QEvent.Type.KeyRelease,
            )
        ):
            alt_pressed: bool = (
                QtCore.Qt.KeyboardModifier.AltModifier
                in QtWidgets.QApplication.queryKeyboardModifiers()
            )

            for k, v in self._actions.items():
                v.setText(f"{k} (Crop)" if alt_pressed else k)

            return True

        return super().eventFilter(obj, event)


class ItoolPlotItem(pg.PlotItem):
    """A subclass of :class:`pyqtgraph.PlotItem` used in ImageTool.

    This class tracks axes and cursors for the data displayed in the plot, and provides
    context menu actions for interacting with the data.

    """

    _sigDragged = QtCore.Signal(object, object)  # :meta private:
    _sigPaletteChanged = QtCore.Signal()  # :meta private:

    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        display_axis: tuple[int] | tuple[int, int],
        axis_enabled: tuple[int, int, int, int],
        image: bool = False,
        image_cls=None,
        plotdata_cls=None,
        **item_kw,
    ) -> None:
        super().__init__(
            axisItems={
                a: erlab.interactive.utils.BetterAxisItem(a)
                for a in ("left", "right", "top", "bottom")
            }
        )
        self._axis_enabled = axis_enabled

        self.vb.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))

        self.slicer_area = slicer_area
        self.display_axis = display_axis

        self.is_image = image
        self._item_kw = item_kw

        self.setup_actions()

        if image_cls is None:  # pragma: no branch
            self.image_cls = ItoolImageItem
        if plotdata_cls is None:  # pragma: no branch
            self.plotdata_cls = ItoolPlotDataItem

        self.slicer_data_items: list[ItoolImageItem | ItoolPlotDataItem] = []
        #: Data items added for each cursor. Contains image or line slice of data

        self.other_data_items: list[pg.PlotDataItem] = []
        #: Data items plotted in twin axes.

        self.cursor_lines: list[dict[int, ItoolCursorLine]] = []
        self.cursor_spans: list[dict[int, ItoolCursorSpan]] = []
        self.add_cursor(update=False)

        self.proxy = pg.SignalProxy(
            self._sigDragged, delay=1 / 120, rateLimit=120, slot=self.process_drag
        )
        self.slicer_area.qapp.primaryScreenChanged.connect(
            self._update_signal_refresh_rate
        )
        self._update_signal_refresh_rate()

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

        # Set axis visibility
        self.setDefaultPadding(0)
        font = QtGui.QFont()
        font.setPointSizeF(float(self.slicer_area.TICK_FONT_SIZE))

        for ax in ("left", "top", "right", "bottom"):
            self.getAxis(ax).setTickFont(font)
            self.getAxis(ax).setStyle(
                autoExpandTextSpace=True, autoReduceTextSpace=True
            )

        self.showAxes(
            self._axis_enabled,
            showValues=self._axis_enabled,
            size=(self.slicer_area.HORIZ_PAD, self.slicer_area.VERT_PAD),
        )

        self.vb1: pg.ViewBox | None = None
        self._twin_visible: bool = False

        self._roi_list: list[ItoolPolyLineROI] = []

    def setup_actions(self) -> None:
        for act in ["Transforms", "Downsample", "Average", "Alpha", "Points"]:
            self.setContextMenuActionVisible(act, False)

        for i in (0, 1):
            # Hide unnecessary menu items
            self.vb.menu.ctrl[i].linkCombo.setVisible(False)
            self.vb.menu.ctrl[i].label.setVisible(False)

        if self.is_image:
            # ROI actions
            self.vb.menu.addSeparator()
            poly_roi_action = self.vb.menu.addAction("Add Polygon ROI")
            poly_roi_action.triggered.connect(self.add_roi)

        self.vb.menu.addSeparator()

        save_action = self.vb.menu.addAction("Save data as HDF5")
        save_action.triggered.connect(self.save_current_data)

        copy_code_action = self.vb.menu.addAction("Copy selection code")
        copy_code_action.triggered.connect(self.copy_selection_code)

        if self.slicer_area._in_manager:
            plot_with_matplotlib_action = self.vb.menu.addAction("Plot with matplotlib")
            plot_with_matplotlib_action.triggered.connect(self.plot_with_matplotlib)

        copy_mpl_code_action = self.vb.menu.addAction("Copy matplotlib code")
        copy_mpl_code_action.triggered.connect(self.copy_matplotlib_code)

        self.vb.menu.addSeparator()

        itool_action = self.vb.menu.addAction("New Window")
        itool_action.triggered.connect(self.open_in_new_window)
        itool_action.setIcon(qtawesome.icon("mdi6.export"))
        itool_action.setIconVisibleInMenu(True)

        # List of actions that should have '(Crop)' appended when Alt is pressed
        croppable_actions: list[QtWidgets.QAction] = [
            save_action,
            copy_code_action,
            itool_action,
        ]

        if self.is_image:
            # Actions that open new windows
            goldtool_action = self.vb.menu.addAction("goldtool")
            goldtool_action.triggered.connect(self.open_in_goldtool)

            restool_action = self.vb.menu.addAction("restool")
            restool_action.triggered.connect(self.open_in_restool)

            dtool_action = self.vb.menu.addAction("dtool")
            dtool_action.triggered.connect(self.open_in_dtool)

            croppable_actions.extend((goldtool_action, restool_action, dtool_action))

            self.vb.menu.addSeparator()

            # Aspect ratio lock checkbox
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

            # AdjustCT-like action
            adjust_color_action = self.vb.menu.addAction("Normalize to View")
            adjust_color_action.setToolTip(
                "Set color limits from the currently visible area of this image.\n"
                "Similar to 'AdjustCT' in Igor Pro."
            )
            adjust_color_action.triggered.connect(self.normalize_to_current_view)

            def _set_icons():
                for act in (goldtool_action, restool_action, dtool_action):
                    act.setIcon(qtawesome.icon("mdi6.export"))
                    act.setIconVisibleInMenu(True)

            self._sigPaletteChanged.connect(_set_icons)
            _set_icons()

        else:
            norm_action = self.vb.menu.addAction("Normalize by mean")
            norm_action.setCheckable(True)
            norm_action.setChecked(False)
            norm_action.toggled.connect(self.set_normalize)
        self.vb.menu.addSeparator()

        self._menu_filter = _OptionKeyMenuFilter(self.vb.menu, croppable_actions)
        self.vb.menu.installEventFilter(self._menu_filter)

    @QtCore.Slot()
    def _update_signal_refresh_rate(self) -> None:
        screen = self.slicer_area.qapp.primaryScreen()
        if screen:
            rate = screen.refreshRate()
            self.proxy.rateLimit = rate if rate > 0 else 60
            self.proxy.delay = 1 / (rate if rate > 0 else 60)

    def connect_signals(self) -> None:
        self.slicer_area.sigIndexChanged.connect(self.refresh_items_data)
        self.slicer_area.sigBinChanged.connect(self.refresh_items_data)
        self.getViewBox().sigRangeChangedManually.connect(self.range_changed_manually)
        self.getViewBox().sigStateChanged.connect(self.refresh_manual_range)
        if self.is_image:
            self.slicer_area.sigShapeChanged.connect(self.remove_guidelines)
            self.slicer_area.sigShapeChanged.connect(self.clear_rois)
        else:
            self.slicer_area.sigDataChanged.connect(self.update_twin_plots)
            self.slicer_area.sigShapeChanged.connect(self.update_twin_plots)
            self.slicer_area.sigTwinChanged.connect(self.update_twin_plots)

    def disconnect_signals(self) -> None:
        self.slicer_area.sigIndexChanged.disconnect(self.refresh_items_data)
        self.slicer_area.sigBinChanged.disconnect(self.refresh_items_data)
        self.getViewBox().sigRangeChangedManually.connect(self.range_changed_manually)
        self.getViewBox().sigStateChanged.disconnect(self.refresh_manual_range)
        if self.is_image:
            self.slicer_area.sigShapeChanged.disconnect(self.remove_guidelines)
            self.slicer_area.sigShapeChanged.disconnect(self.clear_rois)
        else:
            self.slicer_area.sigDataChanged.disconnect(self.update_twin_plots)
            self.slicer_area.sigShapeChanged.disconnect(self.update_twin_plots)
            self.slicer_area.sigTwinChanged.disconnect(self.update_twin_plots)

    def setup_twin(self) -> None:
        """Initialize twin axis for plotting associated coordinates."""
        if not self.is_image and self.vb1 is None:
            self.vb1 = pg.ViewBox(enableMenu=False)
            self.vb1.setDefaultPadding(0)
            loc = self.twin_axes_location
            self.scene().addItem(self.vb1)
            self.getAxis(loc).linkToView(self.vb1)

            # Pass right clicks to original vb
            self.getAxis(loc).mouseClickEvent = self.vb.mouseClickEvent

            self._update_twin_geometry()
            self.vb.sigResized.connect(self._update_twin_geometry)

    @property
    def _axis_to_link_twin(self) -> int:
        """Get the axis to link the twin axis to based on the orientation."""
        return (
            pg.ViewBox.YAxis
            if self.slicer_data_items[-1].is_vertical
            else pg.ViewBox.XAxis
        )

    @property
    def twin_axes_location(self) -> typing.Literal["top", "bottom", "left", "right"]:
        if self.slicer_data_items[-1].is_vertical:
            return "top" if self._axis_enabled[-1] else "bottom"
        return "right" if self._axis_enabled[0] else "left"

    @property
    def twin_visible(self) -> bool:
        return self._twin_visible

    @QtCore.Slot()
    def _update_twin_geometry(self) -> None:
        if self.vb1 is not None and self._twin_visible:
            self.vb1.setGeometry(self.vb.sceneBoundingRect())

    def enableAutoRange(self, axis=None, enable=True, x=None, y=None):
        self.vb.enableAutoRange(axis=axis, enable=enable, x=x, y=y)
        if self.vb1 is not None and self._twin_visible:
            self.vb1.enableAutoRange(axis=None, enable=enable, x=x, y=y)

    @QtCore.Slot()
    def update_twin_range(self, autorange: bool = True) -> None:
        if self.vb1 is not None and self._twin_visible:
            kwargs = {}
            full_bounds = self.vb1.childrenBoundingRect()
            if self.slicer_data_items[-1].is_vertical:
                kwargs["yRange"] = self.getViewBox().state["viewRange"][1]
                if autorange:  # pragma: no branch
                    kwargs["xRange"] = [full_bounds.left(), full_bounds.right()]
            else:
                kwargs["xRange"] = self.getViewBox().state["viewRange"][0]
                if autorange:  # pragma: no branch
                    kwargs["yRange"] = [full_bounds.bottom(), full_bounds.top()]
            self.vb1.setRange(**kwargs)

    @QtCore.Slot()
    def update_twin_plots(self) -> None:
        if not self.isVisible():
            return
        if self.vb1 is None:
            # Never initialized
            if self.array_slicer.twin_coord_names:
                # May or may not have twin coords to plot
                self.setup_twin()
            else:
                # Defer setup until first use
                return

        display_dim: str = str(self.slicer_area.data.dims[self.display_axis[0]])
        associated: dict[Hashable, tuple[npt.NDArray, npt.NDArray]] = (
            self.array_slicer.associated_coords[display_dim]
        )

        n_plots: int = 0
        labels: list[str] = []
        if self.vb1 is None:  # pragma: no cover
            raise RuntimeError(
                "Twin axis ViewBox is not initialized; this should not happen. "
                "Please report a bug."
            )

        for k in tuple(self.array_slicer.twin_coord_names):
            if k in associated:
                x, y = associated[k]
                if n_plots >= len(self.other_data_items):
                    item = pg.PlotDataItem()
                    self.other_data_items.append(item)
                    self.vb1.addItem(item)
                else:
                    item = self.other_data_items[n_plots]

                clr: QtGui.QColor = self.slicer_area.TWIN_COLORS[
                    tuple(associated.keys()).index(k)
                    % len(self.slicer_area.TWIN_COLORS)
                ]  # Color by index among coords associated with this dim
                labels.append(
                    "<tr>"
                    f"<td style='color:{clr.name()}; text-align: center;'>{k}</td>"
                    "</tr>"
                )

                x, y = _pad_1d_plot(x, y)
                if self.slicer_data_items[-1].is_vertical:
                    item.setData(y, x)
                else:
                    item.setData(x, y)
                item.setPen(width=2, color=clr)
                n_plots += 1

        self._twin_visible = n_plots > 0
        ax = self.getAxis(self.twin_axes_location)
        ax.show() if self._twin_visible else ax.hide()
        ax.setStyle(showValues=self._twin_visible)

        if self._twin_visible:
            label_html = "<table cellspacing='0'>" + "".join(labels) + "</table>"
            ax.setLabel(text=label_html)
            ax.resizeEvent()

        while len(self.other_data_items) != n_plots:
            item = self.other_data_items.pop()
            self.vb1.removeItem(item)
            item.forgetViewBox()

    @QtCore.Slot()
    @record_history
    def add_roi(self) -> None:
        # Start from current cursor position
        x0, y0 = tuple(
            self.array_slicer.get_value(
                self.slicer_area.current_cursor, ax, uniform=True
            )
            for ax in self.display_axis
        )
        xrange, yrange = self.vb.state["viewRange"]
        dx, dy = 0.2 * (xrange[1] - xrange[0]), 0.2 * (yrange[1] - yrange[0])
        roi = ItoolPolyLineROI(self, positions=[(x0, y0), (x0 + dx, y0 + dy)])
        self._roi_list.append(roi)
        self.addItem(roi)
        roi.sigRemoveRequested.connect(self.remove_roi)

    @QtCore.Slot(object)
    @record_history
    def remove_roi(self, roi: ItoolPolyLineROI) -> None:
        if roi in self._roi_list:
            self._roi_list.remove(roi)
            self.removeItem(roi)
            roi.deleteLater()

    @QtCore.Slot()
    @suppress_history
    def clear_rois(self) -> None:
        """Remove all ROIs from the plot."""
        for roi in self._roi_list.copy():
            self.remove_roi(roi)

    @property
    def _serializable_state(self) -> PlotItemState:
        """Subset of the state of the underlying viewbox that should be restorable."""
        vb_state = self.getViewBox().getState()
        return {
            "vb_aspect_locked": vb_state["aspectLocked"],
            "vb_x_inverted": vb_state["xInverted"],
            "vb_y_inverted": vb_state["yInverted"],
            "vb_autorange": tuple(vb_state["autoRange"]),
            "roi_states": [roi.saveState() for roi in self._roi_list],
        }

    @_serializable_state.setter
    def _serializable_state(self, state: PlotItemState) -> None:
        vb = self.getViewBox()
        state_dict: dict[str, typing.Any] = {
            "aspectLocked": state["vb_aspect_locked"],
            "xInverted": state["vb_x_inverted"],
            "yInverted": state["vb_y_inverted"],
        }
        autorange = state.get("vb_autorange", None)
        if autorange:
            state_dict["autoRange"] = list(autorange)
        vb.state.update()

        if "roi_states" in state:
            self.clear_rois()

            with self.slicer_area.history_suppressed():
                for s in state["roi_states"]:
                    roi = ItoolPolyLineROI(self, positions=s["points"])
                    self._roi_list.append(roi)
                    self.addItem(roi)
                    roi.sigRemoveRequested.connect(self.remove_roi)
                    roi.setState(s)

    def _get_axis_dims(self, uniform: bool) -> tuple[str | None, str | None]:
        dim_list: list[str] = [
            str(self.slicer_area.data.dims[ax]) for ax in self.display_axis
        ]

        if not uniform and self.array_slicer._nonuniform_axes:
            for i, ax in enumerate(self.display_axis):
                if ax in self.array_slicer._nonuniform_axes:
                    dim_list[i] = dim_list[i].removesuffix("_idx")

        dims = typing.cast("tuple[str] | tuple[str, str]", tuple(dim_list))
        if not self.is_image:
            if self.slicer_data_items[-1].is_vertical:
                dims = (None, *dims)
            else:
                dims = (*dims, None)
        return typing.cast("tuple[str | None, str | None]", dims)

    @property
    def axis_dims(self) -> tuple[str | None, str | None]:
        """Get the names of the data dimensions plotted on each axis.

        Removes '_idx' suffix for non-uniform axes.
        """
        return self._get_axis_dims(uniform=False)

    @property
    def axis_dims_uniform(self) -> tuple[str | None, str | None]:
        """Get the names of the data dimensions plotted on each axis.

        Retains '_idx' suffix for non-uniform axes.
        """
        return self._get_axis_dims(uniform=True)

    @property
    def is_independent(self) -> bool:
        return self.vb.state["linkedViews"] == [None, None]

    @property
    def _current_data(self) -> xr.DataArray:
        """Data in the current plot item."""
        return self.slicer_data_items[self.slicer_area.current_cursor].sliced_data

    @property
    def is_view_cropped(self) -> bool:
        """Whether the current view limits are smaller than the full data range."""
        manual_limits = dict(self.slicer_area.manual_limits)
        for ax_idx in self.display_axis:
            dim_name = str(self.slicer_area.data.dims[ax_idx])
            if dim_name in manual_limits:
                mn, mx = sorted(self.slicer_area.array_slicer.lims_uniform[ax_idx])
                manual_mn, manual_mx = sorted(manual_limits[dim_name])

                if manual_mn > mn or manual_mx < mx:
                    return True

        return False

    @property
    def _crop_indexers(self) -> dict[Hashable, slice]:
        """Returns argument to `DataArray.sel` for cropping to current view limits."""
        return {
            k: v
            for k, v in self.slicer_area.make_slice_dict().items()
            if k in self._current_data.dims
        }

    @property
    def _current_data_cropped(self) -> xr.DataArray:
        """Data in the current plot item, cropped to the current axes view limits."""
        return self._current_data.sel(self._crop_indexers)

    @property
    def current_data(self) -> xr.DataArray:
        """Data in the current plot item, optionally cropped to view limits.

        If accessed while the Alt (Option) key is pressed, the data is cropped to the
        current axes view limits.
        """
        alt_pressed: bool = (
            QtCore.Qt.KeyboardModifier.AltModifier
            in QtWidgets.QApplication.queryKeyboardModifiers()
        )
        data = self._current_data_cropped if alt_pressed else self._current_data

        return erlab.utils.array.sort_coord_order(
            data, self.slicer_area._data.coords.keys()
        )

    def selection_code_for_cursor(self, cursor: int) -> str:
        """Get the selection code for the data.

        Returns a string that looks like ``.sel(...)`` or ``.qsel(...)`` that selects
        the current slice of data based on the current cursor location and bin size.
        """
        sel_code = self.array_slicer.qsel_code(cursor, self.display_axis)
        # sel_code will be ".qsel(...)" or ".isel(...)" or empty string

        if (
            QtCore.Qt.KeyboardModifier.AltModifier
            in QtWidgets.QApplication.queryKeyboardModifiers()
        ):
            sel_indexers = self._crop_indexers
            isel_indexers: dict[Hashable, slice] = {}
            for k in list(sel_indexers.keys()):
                if str(k).endswith("_idx"):
                    isel_indexers[str(k).removesuffix("_idx")] = sel_indexers.pop(k)

            if sel_code.startswith(".qsel"):
                qsel_kw = self.array_slicer.qsel_args(cursor, self.display_axis)
                qsel_kw = qsel_kw | sel_indexers

                sel_code = erlab.interactive.utils.format_kwargs(qsel_kw)
                sel_code = f".qsel({sel_code})"

                if isel_indexers:
                    isel_code = erlab.interactive.utils.format_kwargs(isel_indexers)
                    sel_code = sel_code + f".isel({isel_code})"

                return sel_code

            if sel_code.startswith(".isel"):
                isel_kw = self.array_slicer.isel_args(
                    cursor, self.display_axis, int_if_one=True
                )
                isel_kw = isel_kw | isel_indexers

                sel_code = erlab.interactive.utils.format_kwargs(isel_kw)
                sel_code = f".isel({sel_code})"

            if sel_indexers:
                crop_code = erlab.interactive.utils.format_kwargs(sel_indexers)
                sel_code = sel_code + f".sel({crop_code})"

        return sel_code

    def get_selection_code(self, placeholder: str = "data") -> str:
        """Get selection code for the current cursor and display axis.

        Adds a placeholder data name as a prefix to the selection code returned by
        :attr:`selection_code`. If the data is linked to a watched variable in a
        notebook through the ImageTool manager, the variable name in the notebook is
        used instead.

        Parameters
        ----------
        placeholder : str, optional
            Name to fall back to if the data is not linked to a watched variable in a
            notebook. By default, uses "data".
        """
        data_name = self.slicer_area.watched_data_name
        if not data_name:
            data_name = placeholder
        sel_code: str = self.selection_code_for_cursor(self.slicer_area.current_cursor)
        return f"{data_name}{sel_code}"

    @property
    def is_guidelines_visible(self) -> bool:
        return len(self._guidelines_items) != 0

    def _qsel_kwargs_multicursor(
        self,
    ) -> tuple[dict[Hashable, float | list[float]], Hashable]:
        """Generate keyword arguments for :meth:`xarray.DataArray.qsel` for all cursors.

        Only supports varying a single dimension across cursors and uniform axes.
        """
        if any(
            a in self.array_slicer._nonuniform_axes
            for a in set(range(self.slicer_area.data.ndim)) - set(self.display_axis)
        ):
            raise ValueError("Cannot plot when indexing along non-uniform axes.")

        all_qsel_kws: list[dict[Hashable, float]] = [
            self.array_slicer.qsel_args(cursor, self.display_axis)
            for cursor in range(self.slicer_area.n_cursors)
        ]

        all_keys: set[Hashable] = set().union(*(d.keys() for d in all_qsel_kws))
        result: dict[Hashable, float | list[float]] = {}
        varying: list[Hashable] = []

        for key in all_keys:
            if str(key).endswith("_width"):
                values = [d.get(key, 0.0) for d in all_qsel_kws]
            else:
                values = [d[key] for d in all_qsel_kws]

            if len(set(values)) == 1:
                result[key] = values[0]
            else:
                varying.append(key)
                result[key] = values

        if len(varying) > 1 and not (
            len(varying) == 2 and any(f"{k}_width" in varying for k in varying)
        ):
            raise ValueError(
                "Cannot plot when more than one dimension has differing values "
                "across cursors: "
                f"{sorted(map(str, varying))}"
            )
        if len(varying) == 1 and str(varying[0]).endswith("_width"):
            # Only widths vary; we can't index on widths alone
            raise ValueError(
                "Cannot plot when all cursor positions are the same but widths differ."
            )

        match len(varying):
            case 0:
                variable_dim_name: Hashable | None = None
            case 1:
                variable_dim_name = varying[0]
            case _:
                for k in varying:
                    if not str(k).endswith("_width"):
                        variable_dim_name = k
                        break

        # Put varying keys first
        variable_keys = (variable_dim_name, f"{variable_dim_name}_width")
        other_keys = sorted((k for k in result if k not in variable_keys), key=str)
        ordered_keys = [k for k in variable_keys if k in result] + other_keys
        result = {k: result[k] for k in ordered_keys}

        return result, variable_dim_name

    def _plot_code_multicursor(self, *, placeholder_name: str | None = None) -> str:
        """Generate matplotlib plot code for all cursors."""
        result, variable_dim = self._qsel_kwargs_multicursor()

        if placeholder_name is not None:
            data_name: str = placeholder_name
        else:
            data_name = self.slicer_area.watched_data_name or "data"

        # Determine order of plotted dimensions
        dim_order_plot = list(self.slicer_area._data.dims)
        for k in result:
            if k in dim_order_plot:
                dim_order_plot.remove(k)
        if len(dim_order_plot) != len(self.display_axis):  # pragma: no cover
            raise ValueError(
                "Could not determine order of plotted dimensions for multi-cursor "
                "plot code. This should not happen; please report a bug."
            )

        # Order so that index 0 is always the x axis (for matplotlib plots)
        dim_order_plot.reverse()

        if self.is_image:
            return self._plot_code_image(
                data_name, result, variable_dim, dim_order_plot
            )

        return self._plot_code_line(data_name, result, variable_dim, dim_order_plot[0])

    def _plot_code_line(
        self,
        data_name: str,
        qsel_kwargs: dict[Hashable, float | list[float]],
        variable_dim: Hashable | None,
        x_dim: Hashable,
    ) -> str:
        TAB: str = "    "
        plot_lines: list[str] = ["fig, ax = plt.subplots()"]

        colors: list[str] = [
            self.slicer_area.cursor_colors[i].name()
            for i in range(self.slicer_area.n_cursors)
        ]
        default_colors = erlab.interactive._options.schema.ColorOptions.model_fields[
            "cursors"
        ].default
        colors_changed: bool = any(c not in default_colors for c in colors)

        selected: str = (
            f"{data_name}.qsel({erlab.interactive.utils.format_kwargs(qsel_kwargs)})"
        )
        is_normalized: bool = self.slicer_data_items[
            self.slicer_area.current_cursor
        ].normalize

        if variable_dim is None:
            selected = selected + ".plot(ax=ax)"
            plot_lines.append(selected)
            return "\n".join(plot_lines)

        selected = selected + f'.transpose("{variable_dim}", ...)'

        plot_code: str = erlab.interactive.utils.generate_code(
            xr.DataArray.plot,
            args=[],
            kwargs={"color": "|line_colors[i]|"} if colors_changed else {},
            module="line" if not is_normalized else "(line / line.mean())",
            name="plot",
        )

        set_kwargs: dict[str, typing.Any] = {}
        if x_dim in self._crop_indexers:
            slice_obj = self._crop_indexers[x_dim]
            set_kwargs["xlim"] = (float(slice_obj.start), float(slice_obj.stop))

        if colors_changed:
            plot_lines.append(f"line_colors = {colors!s}")
            plot_lines.append(f"for i, line in enumerate({selected}):")
        else:
            plot_lines.append(f"for line in {selected}:")
        plot_lines.append(TAB + plot_code)

        if set_kwargs:
            import matplotlib.axes

            plot_lines.append(
                erlab.interactive.utils.generate_code(
                    matplotlib.axes.Axes.set, args=[], kwargs=set_kwargs, module="ax"
                )
            )
        return "\n".join(plot_lines)

    def _plot_code_image(
        self,
        data_name: str,
        qsel_kwargs: dict[Hashable, float | list[float]],
        variable_dim: Hashable | None,
        dim_order_plot: list[Hashable],
    ) -> str:
        # Setup plot keyword arguments
        plot_args: list[typing.Any] = [f"|[{data_name}]|"]
        if all(isinstance(k, str) and k.isidentifier() for k in qsel_kwargs):
            plot_kwargs: dict[str, typing.Any] = {
                str(k): v for k, v in qsel_kwargs.items()
            }
        else:
            plot_kwargs = {}
            plot_args.append(
                f"|**{erlab.interactive.utils.format_kwargs(qsel_kwargs)}|"
            )

        if dim_order_plot[0] != self.axis_dims[0]:
            plot_kwargs["transpose"] = True
            dim_order_plot.reverse()

        for k, v in self._crop_indexers.items():
            if k == dim_order_plot[0]:
                plot_kwargs["xlim"] = (float(v.start), float(v.stop))
            elif k == dim_order_plot[1]:
                plot_kwargs["ylim"] = (float(v.start), float(v.stop))

        colormap_props = self.slicer_area.colormap_properties.copy()
        levels: tuple[float, float] | None = None
        if colormap_props["levels_locked"]:
            plot_kwargs["same_limits"] = True
            levels = colormap_props.get("levels")
        if self.getViewBox().state["aspectLocked"]:
            plot_kwargs["axis"] = "image"
        plot_kwargs["cmap"] = colormap_props["cmap"]

        if colormap_props["reverse"] and isinstance(plot_kwargs["cmap"], str):
            plot_kwargs["cmap"] = f"{plot_kwargs['cmap']}_r"

        norm_kws: dict[str, typing.Any] = {}
        if levels is not None:
            if colormap_props["zero_centered"]:
                vmin, vmax = levels
                norm_kws["vcenter"] = 0.5 * (vmin + vmax)
                norm_kws["halfrange"] = (vmax - vmin) / 2
            else:
                norm_kws["vmin"], norm_kws["vmax"] = levels

        if colormap_props["high_contrast"]:
            if colormap_props["zero_centered"]:
                norm_cls: type[matplotlib.colors.Normalize] | None = (
                    erlab.plotting.CenteredInversePowerNorm
                )
            else:
                norm_cls = erlab.plotting.InversePowerNorm
        else:
            if colormap_props["zero_centered"]:
                norm_cls = erlab.plotting.CenteredPowerNorm
            else:
                norm_cls = None

        if norm_cls is None:
            plot_kwargs["gamma"] = colormap_props["gamma"]
            plot_kwargs.update(norm_kws)
        else:
            norm_code = erlab.interactive.utils.generate_code(
                norm_cls,
                args=[colormap_props["gamma"]],
                kwargs=norm_kws,
                module="eplt",
            )
            plot_kwargs["norm"] = f"|{norm_code}|"

        return erlab.interactive.utils.generate_code(
            erlab.plotting.plot_slices,
            args=plot_args,
            kwargs=plot_kwargs,
            module="eplt",
            assign=("fig", "axs" if variable_dim else "ax"),
        )

    @QtCore.Slot(bool)
    def set_normalize(self, normalize: bool) -> None:
        """Toggle normalization for 1D plots."""
        if not self.is_image:  # pragma: no branch
            for item in self.slicer_data_items:
                item.normalize = normalize
                item.refresh_data()

    @QtCore.Slot()
    def open_in_new_window(self) -> None:
        """Open the current data in a new window."""
        data = self.current_data

        color_props = self.slicer_area.colormap_properties
        itool_kw: dict[str, typing.Any] = {
            "data": data,
            "cmap": color_props["cmap"],
            "gamma": color_props["gamma"],
            "high_contrast": color_props["high_contrast"],
            "zero_centered": color_props["zero_centered"],
            "transpose": (self.is_image and data.dims[0] != self.axis_dims[0]),
            "file_path": self.slicer_area._file_path,
            "execute": False,
        }
        if color_props["levels_locked"]:
            itool_kw["vmin"], itool_kw["vmax"] = color_props["levels"]
        if color_props["reverse"] and isinstance(itool_kw["cmap"], str):
            itool_kw["cmap"] = f"{itool_kw['cmap']}_r"

        tool = typing.cast(
            "QtWidgets.QWidget | None", erlab.interactive.itool(**itool_kw)
        )
        if tool is not None:  # pragma: no branch
            self.slicer_area.add_tool_window(tool)

    @QtCore.Slot()
    def open_in_goldtool(self) -> None:
        if self.is_image:  # pragma: no branch
            data = self.current_data
            try:
                self.slicer_area.add_tool_window(
                    erlab.interactive.goldtool(
                        data, data_name=self.get_selection_code(), execute=False
                    )
                )
            except Exception:
                erlab.interactive.utils.MessageDialog.critical(
                    None, "Error", "An error occurred while opening goldtool."
                )

    @QtCore.Slot()
    def open_in_restool(self) -> None:
        if self.is_image:  # pragma: no branch
            data = self.current_data

            if "eV" not in data.dims:
                QtWidgets.QMessageBox.critical(
                    None,
                    "Error",
                    "Data must have an 'eV' dimension to be opened in restool.",
                )
                return
            tool = erlab.interactive.restool(
                data, data_name=self.get_selection_code(), execute=False
            )
            self.slicer_area.add_tool_window(tool)

    @QtCore.Slot()
    def open_in_dtool(self) -> None:
        if self.is_image:  # pragma: no branch
            self.slicer_area.add_tool_window(
                erlab.interactive.dtool(
                    self.current_data.T,
                    data_name=self.get_selection_code(),
                    execute=False,
                )
            )

    @QtCore.Slot()
    def normalize_to_current_view(self) -> None:
        """Adjust color limits to the currently visible area.

        Only available for image plots.

        Sets the color limits of the slicer area to the min and max of the currently
        visible area of this image (similar to AdjustCT in ImageTool).
        """
        if self.is_image:  # pragma: no branch
            self.slicer_area.lock_levels(True)
            self.slicer_area.levels = erlab.utils.array.minmax_darr(
                self._current_data_cropped
            )

    @QtCore.Slot()
    def range_changed_manually(self) -> None:
        """Propagate manual range changes to other plots.

        This slot propagates the limit change to related other plots in the same slicer
        area, and to other linked slicer areas.
        """
        self.slicer_area.propagate_limit_change(self)
        self.update_twin_range(autorange=False)

    @QtCore.Slot()
    def refresh_manual_range(self) -> None:
        """Store manual limit changes in the parent slicer area.

        This slot ensures that the manual limits stored in the parent slicer area are
        always up to date with the current view range.

        When a user manually changes the view range, this slot is called before
        `range_changed_manually` is called.
        """
        if self.vb.state["aspectLocked"] is not False:
            # If aspect ratio is locked, treat all axes as if manual limits are set
            for dim, rng in zip(
                self.axis_dims_uniform, self.vb.state["viewRange"], strict=True
            ):
                if dim is not None:  # pragma: no branch
                    self.slicer_area.manual_limits[dim] = rng
            return

        for dim, auto, rng in zip(
            self.axis_dims_uniform,
            self.vb.state["autoRange"],
            self.vb.state["viewRange"],
            strict=True,
        ):
            if dim is not None:
                # dim is None for intensity axis of line plots (not related to any dim)
                if auto:
                    if dim in self.slicer_area.manual_limits:
                        # Clear manual limits if auto range is enabled
                        # Also trigger update
                        del self.slicer_area.manual_limits[dim]
                        logger.debug("%s manual limits cleared", dim)
                        self.range_changed_manually()
                else:
                    # Store manual limits
                    if self.slicer_area.manual_limits.get(dim, None) != rng:
                        self.slicer_area.manual_limits[dim] = rng
                        logger.debug("%s manual range set to %s", dim, rng)

    def update_manual_range(self) -> None:
        """Update view range from values stored in the parent slicer area."""
        self.set_range_from(self.slicer_area.manual_limits)

    def set_range_from(self, limits: dict[str, list[float]], **kwargs) -> None:
        for i, (dim, key) in enumerate(
            zip(self.axis_dims_uniform, ("xRange", "yRange"), strict=True)
        ):
            if dim is not None and dim in limits:
                kwargs[key] = limits[dim]
            else:
                if self.getViewBox().state["aspectLocked"] is not False:
                    # If aspect locked, do not attempt to set autorange
                    continue

                if not self.getViewBox().state["autoRange"][i] and dim is not None:
                    # If manual limits are not set and auto range is disabled, set to
                    # bounding rect. Internally, vb.autoRange() just calls setRange(...)
                    # so calling setRange only once with the full bounds prevents
                    # recursive calls. Pyqtgraph will handle the rest.
                    full_bounds = self.getViewBox().childrenBoundingRect(
                        items=self.slicer_data_items
                    )
                    kwargs[key] = (
                        [full_bounds.left(), full_bounds.right()]
                        if i == 0
                        else [full_bounds.bottom(), full_bounds.top()]
                    )

        if len(kwargs) != 0:
            self.getViewBox().setRange(**kwargs)
            self.update_twin_range()

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
            and self.isVisible()
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
            self.slicer_area.refresh_all(self.display_axis)
        else:
            for i, ax in enumerate(self.display_axis):
                self.slicer_area.set_value(
                    ax, data_pos_coords[i], update=False, uniform=True
                )
            self.slicer_area.refresh_current(self.display_axis)

        if self.slicer_area.bench:
            self._time_end = time.perf_counter()

    def set_cursor_colors(self, colors: Iterable[QtGui.QColor]) -> None:
        """Set the colors of the cursors and spans."""
        for i, clr in enumerate(colors):
            self.set_cursor_color(i, clr)

    def set_cursor_color(self, index: int, color: QtGui.QColor) -> None:
        """Set the color of a specific cursor and span."""
        _, clr_cursor, clr_cursor_hover, clr_span, clr_span_edge = _make_cursor_colors(
            color
        )
        for line in self.cursor_lines[index].values():
            line.setPen(pg.mkPen(clr_cursor, width=1))
            line.setHoverPen(pg.mkPen(clr_cursor_hover, width=1))
        for span in self.cursor_spans[index].values():
            for span_line in span.lines:
                span_line.setPen(pg.mkPen(clr_span_edge))
            span.setBrush(pg.mkBrush(clr_span))

        if not self.is_image:
            # For line plots, set the pen color of the data item
            self.slicer_data_items[index].setPen(pg.mkPen(color))

    def add_cursor(self, update: bool = True) -> None:
        new_cursor: int = len(self.slicer_data_items)
        line_angles: tuple[int, int] = (90, 0)

        clr, clr_cursor, clr_cursor_hover, clr_span, clr_span_edge = (
            _make_cursor_colors(self.slicer_area.cursor_colors[new_cursor])
        )

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

    def set_cursor_visible(self, visible: bool) -> None:
        """Set visibility of all cursors."""
        for cursor, (line_dict, span_dict) in enumerate(
            zip(self.cursor_lines, self.cursor_spans, strict=True)
        ):
            for line in line_dict.values():
                line.setVisible(visible)
            for ax, span in span_dict.items():
                # Span visibility is controlled by region bounds
                if visible:
                    span.setRegion(self.array_slicer.span_bounds(cursor, ax))
                else:
                    span.setVisible(visible)

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
            cursors = tuple(c for c in range(self.slicer_area.n_cursors))
            for c in cursors:
                self.slicer_area.set_value(
                    axis, value, update=False, uniform=True, cursor=c
                )
            self.sigIndexChanged.emit(cursors, (axis,))

    def remove_cursor(self, index: int) -> None:
        item = self.slicer_data_items.pop(index)

        self.removeItem(item)
        for line, span in zip(
            self.cursor_lines.pop(index).values(),
            self.cursor_spans.pop(index).values(),
            strict=True,
        ):
            self.removeItem(line)
            line.forgetViewBox()
            line.deleteLater()
            self.removeItem(span)
            span.forgetViewBox()
            span.deleteLater()
        for i, item in enumerate(self.slicer_data_items):
            item.cursor_index = i

    def refresh_cursor(self, cursor: int) -> None:
        for ax, line in self.cursor_lines[cursor].items():
            line.setBounds(
                self.array_slicer.lims_uniform[ax],
                self.array_slicer.get_value(cursor, ax, uniform=True),
            )
            span = self.cursor_spans[cursor][ax]
            span.setRegion(self.array_slicer.span_bounds(cursor, ax))
            if not line.isVisible():
                span.setVisible(False)

    @QtCore.Slot(int)
    def set_guidelines(self, n: typing.Literal[0, 1, 2, 3]) -> None:
        """Show rotating crosshairs for alignment."""
        if self.is_image:  # pragma: no branch
            self._guideline_actions[n].setChecked(True)

    @QtCore.Slot()
    def remove_guidelines(self) -> None:
        """Hide rotating crosshairs."""
        self.set_guidelines(0)

    @QtCore.Slot(int)
    def _set_guidelines(self, n: typing.Literal[0, 1, 2, 3]) -> None:
        if self.is_image:  # pragma: no branch
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

            for w in erlab.interactive.utils.make_crosshairs(n):
                self.addItem(w)
                self._guidelines_items.append(w)

            # Select first line since moving any line will move all lines
            line = self._guidelines_items[0]
            target = self._guidelines_items[-1]

            if old_pos is None:
                target.setPos(
                    (
                        self.slicer_area.get_current_value(0, uniform=True),
                        self.slicer_area.get_current_value(1, uniform=True),
                    )
                )
            else:
                target.setPos(old_pos)
                line.setAngle(old_angle)

            def _print_angle():
                line_pos = line.pos()
                self._guideline_angle = line.angle_effective
                self._guideline_offset = [line_pos.x(), line_pos.y()]
                for i in range(2):
                    self._guideline_offset[i] = float(
                        np.round(
                            self._guideline_offset[i],
                            self.array_slicer.get_significant(
                                self.display_axis[i], uniform=True
                            ),
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
        if self.is_image:  # pragma: no branch
            for item in list(self._guidelines_items):
                self._guidelines_items.remove(item)
                self.removeItem(item)
                item.forgetViewBox()
                item.deleteLater()
            self._guideline_angle = 0.0
            self._guideline_offset = [0.0, 0.0]
            self.setTitle(None)

    @QtCore.Slot(int, object)
    @QtCore.Slot(object, object)
    def refresh_items_data(
        self, cursor: int | tuple[int, ...], axes: tuple[int, ...] | None = None
    ) -> None:
        if self.slicer_area.data_chunked:
            # When data is chunked, refreshing is handled by _handle_refresh_dask
            return

        if isinstance(cursor, int):
            cursor = (cursor,)

        for c in cursor:
            # Set cursor lines and spans positions
            self.refresh_cursor(c)

        if axes is not None and all(elem in self.display_axis for elem in axes):
            # When only the indices along display_axis change, it has no effect on the
            # sliced data, so we do not need to refresh the data.
            return

        if len(cursor) == 1:
            # May have been called from refresh_current upon cursor change, handle
            # active cursor switching here

            # This hides images corresponding to other plots and hidden items are not
            # updated due to the isVisible() check below.

            # If multiple cursors are being updated, assume visibility is already set as
            # expected, so skip
            self.set_active_cursor(cursor[0])

        self.vb.blockSignals(True)
        for item in self.slicer_data_items:
            if item.cursor_index not in cursor or not item.isVisible():
                continue
            item.refresh_data()
        self.vb.blockSignals(False)
        # Block vb state signals, handle axes limits in refresh_all after update by
        # calling update_manual_range

    def collect_dask_objects(
        self, cursor: int | tuple[int, ...], axes: tuple[int, ...] | None = None
    ) -> tuple[
        list[ItoolDisplayObject],
        list[npt.NDArray[np.floating] | tuple[float, float, float, float]],
        list[dask.array.Array],
    ]:
        # When data is dask-backed, collect all dask arrays in the items that requires
        # computation
        if axes is not None and all(elem in self.display_axis for elem in axes):
            return [], [], []

        if isinstance(cursor, int):
            cursor = (cursor,)

        objs: list[ItoolDisplayObject] = []
        coord_or_rects: list[
            npt.NDArray[np.floating] | tuple[float, float, float, float]
        ] = []
        arrays: list[dask.array.Array] = []

        if len(cursor) == 1:
            self.set_active_cursor(cursor[0])

        for item in self.slicer_data_items:
            if item.cursor_index not in cursor or not item.isVisible():
                continue
            objs.append(item)
            c, arr = item.fetch_new_data()
            coord_or_rects.append(c)
            arrays.append(typing.cast("dask.array.Array", arr))

        return objs, coord_or_rects, arrays

    @QtCore.Slot()
    def refresh_labels(self) -> None:
        if self.is_image:
            label_kw = {
                loc: self._get_label_unit(axis)
                for visible, loc, axis in zip(
                    self._axis_enabled,
                    ("left", "top", "right", "bottom"),
                    (1, 0, 1, 0),
                    strict=True,
                )
                if visible
            }
        else:
            label_kw = {}

            if self.slicer_data_items[-1].is_vertical:
                valid_ax = ("left", "right")
            else:
                valid_ax = ("top", "bottom")

            for visible, a in zip(
                self._axis_enabled, ("left", "top", "right", "bottom"), strict=True
            ):
                if visible:
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
        data_to_save = self.current_data

        default_name = data_to_save.name
        if default_name is not None and default_name != "":
            default_name = "data"

        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter("xarray HDF5 Files (*.h5)")

        last_dir = pg.PlotItem.lastFileDir
        if not last_dir:
            last_dir = erlab.interactive.imagetool.manager._get_recent_directory()
        if not last_dir:
            last_dir = os.getcwd()

        dialog.setDirectory(os.path.join(last_dir, f"{default_name}.h5"))

        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            data_to_save.to_netcdf(filename, engine="h5netcdf", invalid_netcdf=True)
            pg.PlotItem.lastFileDir = os.path.dirname(filename)

    @QtCore.Slot()
    def copy_selection_code(self) -> None:
        code = self.get_selection_code(placeholder="")
        if code == "":
            QtWidgets.QMessageBox.critical(
                None, "Error", "Selection code is unavailable for this data."
            )
            return
        erlab.interactive.utils.copy_to_clipboard(code)

    @QtCore.Slot()
    def copy_matplotlib_code(self) -> None:
        """Copy matplotlib plot code for all cursors to the clipboard."""
        erlab.interactive.utils.copy_to_clipboard(self._plot_code_multicursor())

    @QtCore.Slot()
    def plot_with_matplotlib(self) -> None:
        """Show the current data using matplotlib (only works in ImageTool Manager)."""
        if self.slicer_area._in_manager:  # pragma: no branch
            manager = self.slicer_area._manager_instance
            if manager:  # pragma: no branch
                manager.ensure_console_initialized()
                console = manager.console._console_widget
                console.initialize_kernel()
                idx = manager.index_from_slicer_area(self.slicer_area)
                code = self._plot_code_multicursor(
                    placeholder_name=f"tools[{idx}].data"
                )
                code += "\nfig.show()"
                console.execute(code)

    @property
    def display_axis(self) -> tuple[int, ...]:
        return self._display_axis

    @display_axis.setter
    def display_axis(self, value: tuple[int, ...]) -> None:
        self._display_axis = value

    @property
    def slicer_area(self) -> ImageSlicerArea:
        slicer_area = self._slicer_area()
        if slicer_area:
            return slicer_area
        raise LookupError("Parent was destroyed")

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea) -> None:
        self._slicer_area = weakref.ref(value)

    @property
    def array_slicer(self) -> erlab.interactive.imagetool.slicer.ArraySlicer:
        return self.slicer_area.array_slicer

    def changeEvent(self, evt: QtCore.QEvent | None) -> None:
        if (
            evt is not None and evt.type() == QtCore.QEvent.Type.PaletteChange
        ):  # pragma: no branch
            self._sigPaletteChanged.emit()
        super().changeEvent(evt)


class ItoolColorBarItem(erlab.interactive.colors.BetterColorBarItem):
    def __init__(self, slicer_area: ImageSlicerArea, **kwargs) -> None:
        self.slicer_area = slicer_area
        kwargs.setdefault(
            "axisItems",
            {
                a: erlab.interactive.utils.BetterAxisItem(a)
                for a in ("left", "right", "top", "bottom")
            },
        )
        kwargs["show_colormap_edit_menu"] = False
        super().__init__(**kwargs)

        copy_action = self.vb.menu.addAction("Copy color limits to clipboard")
        copy_action.triggered.connect(self._copy_limits)

    @property
    def slicer_area(self) -> ImageSlicerArea:
        slicer_area = self._slicer_area()
        if slicer_area:
            return slicer_area
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
        return erlab.interactive.utils.copy_to_clipboard(str(self.slicer_area.levels))

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


class ItoolPolyLineROI(pg.PolyLineROI):
    """Custom ROI for ImageTool.

    Additional functionality includes context menu actions for editing the ROI, slicing
    along the ROI path, and masking data with the ROI.

    Parameters
    ----------
    plot_item : ItoolPlotItem
        Parent plot item.
    positions : list[tuple[float, float]]
        List of (x, y) positions for the ROI vertices.
    closed : bool, optional
        Whether the ROI is closed (polygon) or open (polyline). Default is False.

    """

    def __init__(
        self,
        plot_item: ItoolPlotItem,
        positions: list[tuple[float, float]],
        closed: bool = False,
    ) -> None:
        super().__init__(
            positions, closed=closed, rotatable=False, resizable=False, removable=True
        )
        self._plot_item = weakref.ref(plot_item)

    @property
    def plot_item(self) -> ItoolPlotItem:
        plot_item = self._plot_item()
        if plot_item:
            return plot_item
        raise LookupError("Parent was destroyed")

    @property
    def slicer_area(self) -> ImageSlicerArea:
        return self.plot_item.slicer_area

    def getMenu(self):
        if self.menu is None:
            self.menu = super().getMenu()

            edit_act = QtWidgets.QAction("Edit ROI...", self.menu)
            edit_act.triggered.connect(self.edit_roi)
            self.menu.addAction(edit_act)
            self.menu.edit_act = edit_act

            slice_path_act = QtWidgets.QAction("Slice Along ROI Path", self.menu)
            slice_path_act.triggered.connect(self.slice_along_path)
            self.menu.addAction(slice_path_act)
            self.menu.slice_path_act = slice_path_act

            mask_roi_act = QtWidgets.QAction("Mask Data with ROI", self.menu)
            mask_roi_act.triggered.connect(self.mask_with_roi)
            self.menu.addAction(mask_roi_act)
            self.menu.mask_roi_act = mask_roi_act

        return self.menu

    @record_history
    def handleMoveStarted(self):
        """Inherited to add history recording on move start."""
        super().handleMoveStarted()

    @record_history
    def segmentClicked(self, segment, ev=None, pos=None):
        """Inherited to add history recording on segment click."""
        super().segmentClicked(segment, ev, pos)

    @QtCore.Slot()
    def edit_roi(self) -> None:
        """Open dialog to edit ROI vertices."""
        dialog = _PolyROIEditDialog(self)
        dialog.exec()

    @QtCore.Slot()
    def slice_along_path(self) -> None:
        """Extract line profile as an xarray DataArray."""
        dialog = erlab.interactive.imagetool.dialogs.ROIPathDialog(self)
        dialog.exec()

    @QtCore.Slot()
    def mask_with_roi(self) -> None:
        """Mask data with the polygon defined by the ROI."""
        dialog = erlab.interactive.imagetool.dialogs.ROIMaskDialog(self)
        dialog.exec()

    def _get_vertices(self, uniform: bool = False) -> dict[Hashable, list[float]]:
        array_slicer: erlab.interactive.imagetool.slicer.ArraySlicer = (
            self.plot_item.array_slicer
        )

        coords = self.getState()["points"]
        raw_xy = (
            np.array([p[0] for p in coords], dtype=float),
            np.array([p[1] for p in coords], dtype=float),
        )

        vertices: dict[Hashable, list[float]] = {}
        for ax, dim, values in zip(
            self.plot_item.display_axis, self.plot_item.axis_dims, raw_xy, strict=True
        ):
            decimals: int = array_slicer.get_significant(ax, uniform=uniform)
            if not uniform and ax in array_slicer._nonuniform_axes:
                # Convert from index to true coordinates
                vertices[dim] = (
                    np.interp(
                        values, array_slicer.coords_uniform[ax], array_slicer.coords[ax]
                    )
                    .round(decimals)
                    .tolist()
                )
            else:
                vertices[dim] = values.round(decimals).tolist()

        return vertices


class _PolyROIEditDialog(QtWidgets.QDialog):
    def __init__(self, roi: ItoolPolyLineROI) -> None:
        super().__init__()
        self.roi = roi
        self.setWindowTitle("Edit ROI")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(self.roi.plot_item.axis_dims_uniform)
        hdr = self.table.horizontalHeader()
        if hdr:  # pragma: no branch
            hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        self.closed_check = QtWidgets.QCheckBox("Closed", self)
        self.closed_check.setChecked(self.roi.closed)
        layout.addWidget(self.closed_check)

        # Add/Remove row buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.add_row_btn = QtWidgets.QPushButton("Add Row")
        self.del_row_btn = QtWidgets.QPushButton("Delete Row")
        btn_layout.addWidget(self.add_row_btn)
        btn_layout.addWidget(self.del_row_btn)
        layout.addLayout(btn_layout)

        self.add_row_btn.clicked.connect(self._add_row)
        self.del_row_btn.clicked.connect(self._delete_row)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        model = self.table.model()
        if model:  # pragma: no branch
            model.rowsRemoved.connect(self._rowcount_changed)

        self._populate_table()

    def _populate_table(self) -> None:
        values = np.column_stack(tuple(self.roi._get_vertices(uniform=True).values()))
        self.table.setRowCount(len(values))
        for i in range(values.shape[0]):
            for j in range(2):
                item = QtWidgets.QTableWidgetItem(
                    np.format_float_positional(values[i, j], trim="-")
                )
                item.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(i, j, item)

    @QtCore.Slot()
    def _rowcount_changed(self) -> None:
        """Enable/disable delete row button based on row count."""
        self.del_row_btn.setEnabled(self.table.rowCount() >= 3)

    @QtCore.Slot()
    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        # Optionally, initialize with zeros or NaN
        for j in range(2):
            item = QtWidgets.QTableWidgetItem("0")
            item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, j, item)

    @QtCore.Slot()
    def _delete_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def accept(self) -> None:
        points = []
        for i in range(self.table.rowCount()):
            x_item = self.table.item(i, 0)
            y_item = self.table.item(i, 1)
            try:
                if x_item is not None and y_item is not None:
                    points.append((float(x_item.text()), float(y_item.text())))
            except ValueError:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Invalid value at row {i + 1}. Please enter numeric values.",
                )
                return
        self.roi.slicer_area.sigWriteHistory.emit()
        self.roi.setPoints(points, closed=self.closed_check.isChecked())
        super().accept()
