"""Provides core functionality of ImageTool.

This module contains :class:`ImageSlicerArea` which handles the core functionality of
ImageTool, including the slicing and plotting of data.

"""

from __future__ import annotations

import collections
import contextlib
import copy
import importlib
import logging
import os
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
from erlab.interactive.imagetool import _history, provenance
from erlab.interactive.imagetool._viewer_dialogs import (
    _AssociatedCoordsDialog,
    _CursorColorCoordDialog,
)

if typing.TYPE_CHECKING:
    import datetime
    from collections.abc import Callable, Hashable, Iterable, Mapping

    import qtawesome

    from erlab.interactive.imagetool.plot_items import (
        ItoolGraphicsLayoutWidget,
        ItoolImageItem,
        ItoolPlotItem,
    )
else:
    import lazy_loader as _lazy

    qtawesome = _lazy.load("qtawesome")

from erlab.interactive.imagetool.viewer_linking import (
    SlicerLinkProxy,
    _sync_splitters,
    link_slicer,
    record_history,
    suppress_history,
)
from erlab.interactive.imagetool.viewer_state import (
    ColorMapState,
    ImageSlicerState,
    PlotItemState,
    _parse_input,
    _processed_ndim,
)

logger = logging.getLogger(__name__)

_FileDataSelection = provenance.FileDataSelection
_FileSelection: typing.TypeAlias = int | dict[str, typing.Any] | _FileDataSelection
_LoadFunc: typing.TypeAlias = tuple[
    collections.abc.Callable[..., typing.Any] | str,
    dict[str, typing.Any],
    _FileSelection,
]


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
        selection of the data variable used when loading the data. The function is
        called when reloading the data. If using a data loader plugin, the function may
        be given as a string representing the loader name.

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
    sigSourceDataReplaced()
        Signal emitted when the underlying source data is replaced or otherwise
        changed at the source level, such as file open, reload, manager-driven
        replacement, accepted filters, or in-place console edits.
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

    _HISTORY_GROUP_IDLE_TIMEOUT_MS = 300

    @property
    def provenance_spec(
        self,
    ) -> provenance.ToolProvenanceSpec | None:
        """Canonical replay provenance for the current ImageTool data."""
        return typing.cast(
            "provenance.ToolProvenanceSpec | None",
            getattr(self.parent(), "provenance_spec", None),
        )

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
    sigDataBackingChanged = QtCore.Signal()  #: :meta private:
    sigSourceDataReplaced = QtCore.Signal(object)  #: :meta private:
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
        load_func: _LoadFunc | None = None,
        auto_compute: bool = True,
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

        # Current plot filter. Accepted filters are represented by the operation below.
        self._applied_func: Callable[[xr.DataArray], xr.DataArray] | None = None
        self._applied_provenance_operation: (
            provenance.ToolProvenanceOperation | None
        ) = None
        self._accepted_filter_provenance_operation: (
            provenance.ToolProvenanceOperation | None
        ) = None
        self._accepted_filter_data: xr.DataArray | None = None
        # `_data` is the public/source array, while `ArraySlicer._obj` is the internal
        # validated view used for slicing. The two share values by default and detach
        # only when a write needs isolation.

        # If the following flag is True, `self._data` shares memory with a user-provided
        # array. It turns False when pre-processing or validations cause the data to
        # detach from the original array.
        self._data_shares_external_values: bool = False

        # If the following flag is True, `self._data` shares memory with
        # `self.array_slicer._obj`. It turns False when user actions such as a call to
        # `self.apply_func` cause the data to detach.
        self._obj_shares_data_values: bool = False

        # Queues to handle undo and redo
        self._prev_states: collections.deque[_history.HistoryEntry] = collections.deque(
            maxlen=1000
        )
        self._next_states: collections.deque[_history.HistoryEntry] = collections.deque(
            maxlen=1000
        )
        self._pending_history_entry: _history.HistoryEntry | None = None
        self._pending_history_committed = False
        self._link_sync_suppressed = 0
        self._history_group_active = False
        self._history_group_timer = QtCore.QTimer(self)
        self._history_group_timer.setSingleShot(True)
        self._history_group_timer.timeout.connect(self.end_history_group)

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
        self.axis_inversions: dict[str, bool] = {}
        # Dictionary of inverted view state for each plotted data dimension.

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
        self._load_func: _LoadFunc | None = None
        self._source_input_dtype: np.dtype[typing.Any] | None = None
        self.current_cursor: int = 0
        self._data: xr.DataArray
        self._array_slicer: erlab.interactive.imagetool.slicer.ArraySlicer

        self.set_data(
            data,
            rad2deg=rad2deg,
            file_path=file_path,
            load_func=load_func,
            auto_compute=auto_compute,
        )

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
            levels = self.levels
            prop["levels"] = (float(levels[0]), float(levels[1]))
        return typing.cast("ColorMapState", prop)

    @property
    def controls_visible(self) -> bool:
        parent = self.parent()
        if hasattr(parent, "controls_visible"):
            return bool(typing.cast("typing.Any", parent).controls_visible)
        return True

    @property
    def state(self) -> ImageSlicerState:
        load_func = self._load_func
        if load_func is not None:
            fn = load_func[0]
            func_name = f"{fn.__module__}:{fn.__qualname__}" if callable(fn) else fn
            selection = load_func[2]
            if isinstance(selection, _FileDataSelection):
                selection = selection.model_dump(mode="json")
            load_func = (func_name, load_func[1], selection)
        state: ImageSlicerState = {
            "color": self.colormap_properties,
            "slice": self.array_slicer.state,
            "current_cursor": int(self.current_cursor),
            "manual_limits": copy.deepcopy(self.manual_limits),
            "axis_inversions": copy.deepcopy(self.axis_inversions),
            "splitter_sizes": self.splitter_sizes,
            "file_path": str(self._file_path) if self._file_path is not None else None,
            "load_func": load_func,
            "cursor_colors": [c.name() for c in self.cursor_colors],
            "controls_visible": self.controls_visible,
            "plotitem_states": [p._serializable_state for p in self.axes],
        }
        filter_operation = self._accepted_filter_provenance_operation
        if filter_operation is not None:
            state["filter_operation"] = filter_operation.model_dump(mode="json")
        return state

    @state.setter
    def state(self, state: ImageSlicerState) -> None:
        with self.history_suppressed(), self.link_sync_suppressed():
            self._restore_state(state)

    def _restore_state(
        self,
        state: ImageSlicerState,
        *,
        emit_filter_edited: bool = False,
    ) -> None:
        logger.debug("Restoring state...")
        parent = self.parent()
        if hasattr(parent, "_set_controls_visible"):
            typing.cast("typing.Any", parent)._set_controls_visible(
                bool(state.get("controls_visible", True))
            )

        if "splitter_sizes" in state:
            self.splitter_sizes = state["splitter_sizes"]

        plotitem_states = state.get("plotitem_states", None)
        if plotitem_states is not None:  # pragma: no branch
            for ax, plotitem_state in zip(self.axes, plotitem_states, strict=True):
                ax._serializable_state = plotitem_state
        logger.debug("Restored plotitem states")

        self.make_cursors(state["cursor_colors"], update=False)
        self.sigCursorCountChanged.emit(self.n_cursors)
        self.sigCursorColorsChanged.emit()
        logger.debug("Restored cursor number and colors")

        self.set_current_cursor(state["current_cursor"], update=False)
        logger.debug("Restored current cursor")

        self.array_slicer.state = state["slice"]
        self.sigBinChanged.emit(self.current_cursor, tuple(range(self.data.ndim)))
        for ax in self.axes:
            if ax.is_image:
                ax.sync_guidelines_to_active_cursor()
        logger.debug("Restored array slicer state")

        self.set_manual_limits(state.get("manual_limits", {}))
        logger.debug("Restored manual limits")

        axis_inversions = state.get("axis_inversions", None)
        if axis_inversions is None:
            axis_inversions = self._axis_inversions_from_plotitem_states(
                plotitem_states
            )
        self.axis_inversions = self._normalized_axis_inversions(axis_inversions)
        self.apply_axis_inversions()
        logger.debug("Restored axis inversions")

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
                self._load_func = (fn, *load_func[1:])
            else:
                self._load_func = None

        if file_path is not None:
            self.sigDataChanged.emit()
            logger.debug("Restored file path")

        self._restore_filter_operation_from_state(
            state.get("filter_operation", None),
            emit_edited=emit_filter_edited,
        )
        logger.debug("Restored filter operation")

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
        if not erlab.interactive.utils.qt_is_valid(self.lock_levels_act):
            return bool(self._colormap_properties["levels_locked"])
        return self.lock_levels_act.isChecked()

    @levels_locked.setter
    def levels_locked(self, value: bool) -> None:
        self._colormap_properties["levels_locked"] = bool(value)
        if not erlab.interactive.utils.qt_is_valid(self.lock_levels_act):
            return
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
    def displayed_data(self) -> xr.DataArray:
        """Return the public data values committed for derived outputs."""
        return self._accepted_data_for_dims(tuple(self._data.dims))

    def _tool_source_parent_data(self) -> xr.DataArray:
        """Return the current slicer view used for source-bound child tools."""
        return self._accepted_data_for_dims(tuple(self.data.dims)).copy(deep=False)

    @property
    def has_active_filter(self) -> bool:
        return (
            self._applied_func is not None
            or self._accepted_filter_provenance_operation is not None
        )

    def persistence_data_and_state(self) -> tuple[xr.DataArray, ImageSlicerState]:
        """Return data/state for save and clone paths."""
        return self._data, self.state

    def displayed_live_source_spec(
        self,
        base_spec: provenance.ToolProvenanceSpec | None,
    ) -> provenance.ToolProvenanceSpec | None:
        """Return live source provenance including the accepted display filter."""
        source_spec = provenance.require_live_source_spec(base_spec)
        operation = self._accepted_filter_provenance_operation
        if source_spec is None or operation is None:
            return source_spec
        return source_spec.append_display_operation(operation)

    def _accepted_data_for_dims(self, dims: tuple[Hashable, ...]) -> xr.DataArray:
        operation = self._accepted_filter_provenance_operation
        if operation is None:
            return self._data_aligned_to_dims(self._data, dims)
        accepted = self._accepted_filter_data
        if accepted is None:
            raise RuntimeError("Accepted filter data is missing")
        return self._data_aligned_to_dims(accepted, dims)

    def _normalize_filter_result_for_source_dims(
        self,
        source_data: xr.DataArray,
        filtered: xr.DataArray,
        dims: tuple[Hashable, ...],
    ) -> xr.DataArray:
        """Validate a filter result and align it with the requested data layout."""
        if source_data.ndim != filtered.ndim:
            raise ValueError("DataArray dimensions do not match")
        if set(source_data.dims) != set(filtered.dims):
            raise ValueError("DataArray dimensions do not match")

        expected_source_shape = tuple(source_data.sizes[dim] for dim in filtered.dims)
        if filtered.shape != expected_source_shape:
            raise ValueError("DataArray shape does not match")

        filtered = self._data_aligned_to_dims(filtered, dims)
        expected_shape = self._expected_layout_shape(source_data, dims)
        if filtered.shape != expected_shape:
            raise ValueError("DataArray shape does not match")
        return filtered

    def _normalize_filter_result_for_dims(
        self, filtered: xr.DataArray, dims: tuple[Hashable, ...]
    ) -> xr.DataArray:
        return self._normalize_filter_result_for_source_dims(self._data, filtered, dims)

    def _expected_layout_shape(
        self, source_data: xr.DataArray, dims: tuple[Hashable, ...]
    ) -> tuple[int, ...]:
        aligned = self._data_aligned_to_dims(source_data, dims)
        return tuple(aligned.sizes[dim] for dim in dims)

    @staticmethod
    def _data_aligned_to_dims(
        data: xr.DataArray, dims: tuple[Hashable, ...]
    ) -> xr.DataArray:
        missing_dims = tuple(dim for dim in dims if dim not in data.dims)
        if missing_dims and any(str(dim).endswith("_idx") for dim in missing_dims):
            data = erlab.interactive.imagetool.slicer.make_dims_uniform(data)
            missing_dims = tuple(dim for dim in dims if dim not in data.dims)
        if missing_dims == ("stack_dim",):
            data = data.expand_dims("stack_dim", axis=dims.index("stack_dim"))
        if data.dims != dims:
            return data.transpose(*dims)
        return data

    def _filtered_data_for_dims(
        self, func: Callable[[xr.DataArray], xr.DataArray], dims: tuple[Hashable, ...]
    ) -> xr.DataArray:
        return self._normalize_filter_result_for_dims(func(self._data), dims)

    def _filtered_data_for_display(
        self, func: Callable[[xr.DataArray], xr.DataArray]
    ) -> xr.DataArray:
        return self._filtered_data_for_dims(func, tuple(self.data.dims))

    def _filter_operation_result_for_data(
        self,
        data: xr.DataArray,
        operation: provenance.ToolProvenanceOperation,
        dims: tuple[Hashable, ...],
    ) -> xr.DataArray:
        return self._normalize_filter_result_for_source_dims(
            data,
            operation.apply(data, parent_data=data),
            dims,
        )

    def _replacement_display_dims(self, data: xr.DataArray) -> tuple[Hashable, ...]:
        validated = erlab.interactive.imagetool.slicer.ArraySlicer.validate_array(
            data,
            copy_values=False,
        )
        if erlab.interactive.imagetool.slicer.check_cursors_compatible(
            self.data,
            validated,
        ):
            return tuple(self.data.dims)
        return tuple(validated.dims)

    def _filter_operation_result_for_replacement(
        self,
        data: xr.DataArray,
        operation: provenance.ToolProvenanceOperation,
    ) -> xr.DataArray:
        return self._filter_operation_result_for_data(
            data,
            operation,
            self._replacement_display_dims(data),
        )

    @staticmethod
    def _filter_func_from_operation(
        operation: provenance.ToolProvenanceOperation,
    ) -> Callable[[xr.DataArray], xr.DataArray]:
        def _apply_filter(darr: xr.DataArray) -> xr.DataArray:
            return operation.apply(darr, parent_data=darr)

        return _apply_filter

    def _parse_filter_operation_state(
        self,
        payload: dict[str, typing.Any] | None,
    ) -> provenance.ToolProvenanceOperation | None:
        if payload is None:
            return None
        return provenance.parse_tool_provenance_operation(payload)

    def _restore_filter_operation_from_state(
        self,
        payload: dict[str, typing.Any] | None,
        *,
        emit_edited: bool = False,
    ) -> None:
        operation = self._parse_filter_operation_state(payload)
        if operation == self._accepted_filter_provenance_operation and (
            (operation is None and self._accepted_filter_data is None)
            or (operation is not None and self._accepted_filter_data is not None)
        ):
            return
        self.apply_filter_operation(operation, update=False, emit_edited=emit_edited)

    @staticmethod
    def _owned_values_copy(darr: xr.DataArray) -> typing.Any:
        """Return a writable owned copy of the underlying values buffer."""
        values = darr.data
        if getattr(values, "chunks", None) is not None and hasattr(values, "compute"):
            return values.compute()
        if hasattr(values, "copy"):
            return values.copy()
        return np.array(values, copy=True)

    @staticmethod
    def _replace_data_values(darr: xr.DataArray, values: typing.Any) -> None:
        """Swap a DataArray onto a new values buffer without changing coords/dims."""
        darr._variable = darr.variable.copy(deep=False, data=values)

    def _restore_obj_from_source(self, update: bool = True) -> None:
        """Rebuild the internal slicer array so it shares the current source values."""
        preserve_dims = (
            tuple(self.array_slicer._obj.dims)
            if hasattr(self, "_array_slicer")
            else None
        )
        self.array_slicer.set_array(
            self._data,
            reset=False,
            copy_values=False,
            preserve_dims=preserve_dims,
        )
        self._obj_shares_data_values = True
        if update:
            self.refresh_all(only_plots=True)
            self.lock_levels(self.levels_locked)

    def _set_source_item(self, key, value, *, emit_signals: bool = True) -> None:
        """Safely update a subset of the public source array in place."""
        need_restore = (
            self._applied_func is not None or not self._obj_shares_data_values
        )
        self._applied_func = None
        self._applied_provenance_operation = None
        self._accepted_filter_provenance_operation = None
        self._accepted_filter_data = None
        restored_obj = False

        if self._data_shares_external_values:
            self._replace_data_values(self._data, self._owned_values_copy(self._data))
            self._data_shares_external_values = False
            if self._obj_shares_data_values:
                self._restore_obj_from_source(update=False)
                restored_obj = True

        if need_restore and not restored_obj:
            self._restore_obj_from_source(update=False)

        updated = False
        try:
            self._data[key] = value
            updated = True
        finally:
            self.array_slicer.clear_val_cache()
            self.refresh_all(only_plots=True)
            self.lock_levels(self.levels_locked)
            if updated and emit_signals:
                self.sigSourceDataReplaced.emit(self._tool_source_parent_data())
                self.sigDataEdited.emit()

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

    @contextlib.contextmanager
    def link_sync_suppressed(self):
        self._link_sync_suppressed += 1
        try:
            yield
        finally:
            self._link_sync_suppressed -= 1

    @contextlib.contextmanager
    def history_group(self, timeout_ms: int = _HISTORY_GROUP_IDLE_TIMEOUT_MS):
        self.begin_history_group(timeout_ms)
        try:
            yield
        finally:
            self.end_history_group()

    @QtCore.Slot()
    def begin_history_group(
        self, timeout_ms: int = _HISTORY_GROUP_IDLE_TIMEOUT_MS
    ) -> None:
        self._history_group_active = True
        self._history_group_timer.start(timeout_ms)

    @QtCore.Slot()
    def end_history_group(self) -> None:
        self.finalize_history_entry()
        if erlab.interactive.utils.qt_is_valid(self._history_group_timer):
            self._history_group_timer.stop()
        self._history_group_active = False

    def _discard_pending_history_entry(self) -> None:
        self._pending_history_entry = None
        self._pending_history_committed = False
        self._history_group_active = False
        if erlab.interactive.utils.qt_is_valid(self._history_group_timer):
            self._history_group_timer.stop()

    def _history_state_snapshot(self) -> ImageSlicerState:
        state = copy.deepcopy(self.state)
        state.pop("splitter_sizes", None)
        return state

    def next_linked_history_transaction_id(self) -> str:
        if (
            self._history_group_active
            and self._pending_history_entry is not None
            and self._pending_history_entry.transaction_id is not None
        ):
            return self._pending_history_entry.transaction_id
        return uuid.uuid4().hex

    def begin_history_entry(self, transaction_id: str | None) -> None:
        if not self._write_history:
            return
        if self._pending_history_entry is not None:
            if self._pending_history_entry.transaction_id == transaction_id:
                return
            self.finalize_history_entry()

        before_state = self._history_state_snapshot()
        entry = _history.HistoryEntry(
            before_state=typing.cast("dict[str, typing.Any]", before_state),
            after_state=typing.cast(
                "dict[str, typing.Any]", copy.deepcopy(before_state)
            ),
            changed_paths=frozenset(),
            transaction_id=transaction_id,
        )
        self._pending_history_entry = entry
        self._pending_history_committed = False

    def finalize_history_entry(self, *, keep_pending: bool = False) -> None:
        entry = self._pending_history_entry
        if entry is None:
            return
        if not erlab.interactive.utils.qt_is_valid(self, self.lock_levels_act):
            self._discard_pending_history_entry()
            return

        after_state = self._history_state_snapshot()
        entry.after_state = typing.cast("dict[str, typing.Any]", after_state)
        entry.changed_paths = _history.changed_paths(entry.before_state, after_state)
        if not entry.changed_paths and not keep_pending:
            if self._pending_history_committed:
                self._prev_states.remove(entry)
            self._pending_history_entry = None
            self._pending_history_committed = False
            self.sigHistoryChanged.emit()
            return

        if entry.changed_paths and not self._pending_history_committed:
            self._prev_states.append(entry)
            self._next_states.clear()
            self._pending_history_committed = True

        if not keep_pending:
            self._pending_history_entry = None
            self._pending_history_committed = False
        self.sigHistoryChanged.emit()

    def record_history_mutation(
        self,
        transaction_id: str | None,
        mutation: Callable[[], typing.Any],
        *,
        keep_pending: bool = False,
    ) -> typing.Any:
        recording = self._write_history
        self.begin_history_entry(transaction_id)
        try:
            with self.history_suppressed():
                return mutation()
        finally:
            if recording:
                self.finalize_history_entry(keep_pending=keep_pending)

    def _apply_history_entry(self, entry: _history.HistoryEntry, *, undo: bool) -> None:
        source = entry.before_state if undo else entry.after_state
        patched = _history.patch_state(self.state, source, entry.changed_paths)
        emit_filter_edited = any(
            bool(path) and path[0] == "filter_operation" for path in entry.changed_paths
        )
        with self.history_suppressed(), self.link_sync_suppressed():
            self._restore_state(
                typing.cast("ImageSlicerState", patched),
                emit_filter_edited=emit_filter_edited,
            )

    def _apply_matching_linked_history_entry(
        self, transaction_id: str, *, undo: bool
    ) -> None:
        self.end_history_group()
        stack = self._prev_states if undo else self._next_states
        entry = next(
            (
                candidate
                for candidate in reversed(stack)
                if candidate.transaction_id == transaction_id
            ),
            None,
        )
        if entry is None or not _history.entry_matches_current(
            entry, self._history_state_snapshot(), undo=undo
        ):
            return

        stack.remove(entry)
        self._apply_history_entry(entry, undo=undo)
        if undo:
            self._next_states.append(entry)
        else:
            self._prev_states.append(entry)
        self.sigHistoryChanged.emit()

    def _propagate_history_entry(
        self, entry: _history.HistoryEntry, *, undo: bool
    ) -> None:
        if entry.transaction_id is None or self._linking_proxy is None:
            return
        for target in tuple(self.linked_slicers):
            target._apply_matching_linked_history_entry(entry.transaction_id, undo=undo)

    @QtCore.Slot()
    def write_state(self) -> None:
        if not self._write_history:
            return

        if self._history_group_active:
            self.begin_history_entry(None)
            self._history_group_timer.start()
            return

        self.begin_history_entry(None)
        erlab.interactive.utils.single_shot(
            self, 0, self.finalize_history_entry, self.lock_levels_act
        )

    @QtCore.Slot()
    @suppress_history
    def flush_history(self) -> None:
        """Clear the undo and redo history."""
        self.end_history_group()
        self._prev_states.clear()
        self._next_states.clear()
        self.sigHistoryChanged.emit()

    @QtCore.Slot()
    @suppress_history
    def undo(self) -> None:
        """Undo the most recent action."""
        self.end_history_group()
        if self.undoable:  # pragma: no branch
            entry = self._prev_states.pop()
            self._apply_history_entry(entry, undo=True)
            self._next_states.append(entry)
            self._propagate_history_entry(entry, undo=True)
            self.sigHistoryChanged.emit()

    @QtCore.Slot()
    @suppress_history
    def redo(self) -> None:
        """Redo the most recently undone action."""
        self.end_history_group()
        if self.redoable:  # pragma: no branch
            entry = self._next_states.pop()
            self._apply_history_entry(entry, undo=False)
            self._prev_states.append(entry)
            self._propagate_history_entry(entry, undo=False)
            self.sigHistoryChanged.emit()

    def history_states(
        self, *, finalize_pending: bool = True
    ) -> tuple[list[ImageSlicerState], int]:
        if finalize_pending:
            self.finalize_history_entry()
        states = [
            typing.cast("ImageSlicerState", entry.before_state)
            for entry in self._prev_states
        ]
        current_state = self._history_state_snapshot()
        if not states or states[-1] != current_state:
            states.append(current_state)
        current_index = len(states) - 1

        if self._next_states:
            states.extend(
                typing.cast("ImageSlicerState", entry.after_state)
                for entry in reversed(self._next_states)
            )

        return states, current_index

    def history_entry_changes(
        self, *, finalize_pending: bool = True
    ) -> list[
        tuple[
            int,
            int,
            ImageSlicerState,
            ImageSlicerState,
            bool,
            datetime.datetime | None,
        ]
    ]:
        if finalize_pending:
            self.finalize_history_entry()
        entries: list[
            tuple[
                int,
                int,
                ImageSlicerState,
                ImageSlicerState,
                bool,
                datetime.datetime | None,
            ]
        ] = []
        current_index = len(self._prev_states)
        if current_index == 0 and self._next_states:
            current_state = self._history_state_snapshot()
            entries.append((0, 0, current_state, current_state, True, None))
        for i, entry in enumerate(self._prev_states, start=1):
            entries.append(
                (
                    i,
                    i - current_index,
                    typing.cast("ImageSlicerState", entry.before_state),
                    typing.cast("ImageSlicerState", entry.after_state),
                    i == current_index,
                    entry.created_at,
                )
            )
        for i, entry in enumerate(reversed(self._next_states), start=1):
            entries.append(
                (
                    current_index + i,
                    i,
                    typing.cast("ImageSlicerState", entry.before_state),
                    typing.cast("ImageSlicerState", entry.after_state),
                    False,
                    entry.created_at,
                )
            )
        return entries

    def go_to_history_index(self, index: int) -> None:
        """Go to a specific index in the history.

        Parameters
        ----------
        index
            Index in the history to go to. 0 corresponds to the current state. Positive
            indices point to the future, while negative indices point to the past.
        """
        self.end_history_group()
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
        self.reload_act.setToolTip(
            "Reload data from its saved files, parent, or inputs"
        )
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

        self.cursor_color_act = QtWidgets.QAction("Edit Cursor Colors…", self)
        self.cursor_color_act.triggered.connect(self.edit_cursor_colors)

        self.cursor_colors_by_coord_act = QtWidgets.QAction(
            "Set Cursor Colors by Coordinate…", self
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
            coord_dims, coord_name, cmap, reverse, vmin, vmax = cursor_color_params
            if isinstance(cursor, int):
                cursor = (cursor,)

            coord_info = self.array_slicer.cursor_color_coord(
                cursor[0], coord_dims, coord_name
            )
            if coord_info is None:
                self.array_slicer._cursor_color_params = None
                return
            coord_dims, target_coord, _ = coord_info
            if axes is not None and not any(
                self.data.dims[ax] in coord_dims for ax in axes
            ):
                return

            mn, mx = np.min(target_coord), np.max(target_coord)
            scale_factor = (vmax - vmin) / (mx - mn)
            cmap = erlab.interactive.colors.pg_colormap_from_name(cmap)

            for cursor_idx in cursor:
                coord_info = typing.cast(
                    "tuple[tuple[Hashable, ...], npt.NDArray[np.float64], float]",
                    self.array_slicer.cursor_color_coord(
                        cursor_idx, coord_dims, coord_name
                    ),
                )
                coord_value = coord_info[2]
                value = (coord_value - mn) * scale_factor + vmin
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
        if self.array_slicer._obj.chunks is None:
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

        self.sigPointValueChanged.emit(
            erlab.interactive.imagetool.slicer._display_safe_float(next(arrays_it))
        )

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
                ax.update_axis_inversions()

    @QtCore.Slot(tuple)
    @link_slicer
    def refresh_current(self, axes: tuple[int, ...] | None = None) -> None:
        self.sigIndexChanged.emit(self.current_cursor, axes)

    @QtCore.Slot(object, object)
    @link_slicer
    def refresh(
        self, cursor: int | tuple[int, ...], axes: tuple[int, ...] | None = None
    ) -> None:
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
        load_func: _LoadFunc | None = None,
        auto_compute: bool = True,
        *,
        source_replaced: bool = False,
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
            selection of the data variable used when loading the data. The function is
            called when reloading the data. If using a data loader plugin, the function
            may be given as a string representing the loader name.
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
        self._source_input_dtype = np.dtype(data.dtype)
        shares_external_values = True
        if hasattr(self, "_data") and data is self._data:
            shares_external_values = False
        data = data.copy(deep=False)

        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)
            shares_external_values = False

        if not rad2deg:
            source = data
        else:
            if np.iterable(rad2deg):
                conv_dims = rad2deg
            else:
                conv_dims = [
                    d
                    for d in ("phi", "theta", "beta", "alpha", "chi", "xi")
                    if d in data.dims
                ]
            source = data.assign_coords({d: np.rad2deg(data[d]) for d in conv_dims})

        if (
            auto_compute
            and source.chunks is not None
            and (source.nbytes * 1e-6)
            < erlab.interactive.options.model.io.dask.compute_threshold
        ):
            source = source.compute()
            shares_external_values = False

        self._data = source.copy(deep=False)
        self._data_shares_external_values = shares_external_values
        self._obj_shares_data_values = True
        self._applied_func = None
        self._applied_provenance_operation = None
        self._accepted_filter_provenance_operation = None
        self._accepted_filter_data = None

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
                cursors_reset = self._array_slicer.set_array(
                    self._data, reset=True, copy_values=False
                )

            else:
                self._array_slicer = erlab.interactive.imagetool.slicer.ArraySlicer(
                    self._data, self
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
        self.axis_inversions = self._normalized_axis_inversions(self.axis_inversions)

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
        parent = self.parent()
        if (
            self._file_path is not None
            and hasattr(parent, "_sync_file_load_provenance")
            and hasattr(parent, "_slicer_area")
        ):
            typing.cast("typing.Any", parent)._sync_file_load_provenance()
        if source_replaced:
            self.sigSourceDataReplaced.emit(self._tool_source_parent_data())

    def replace_source_data(
        self,
        data: xr.DataArray | npt.NDArray,
        rad2deg: bool | Iterable[str] = False,
        file_path: str | os.PathLike | None = None,
        load_func: _LoadFunc | None = None,
        auto_compute: bool = True,
        *,
        emit_edited: bool = False,
    ) -> None:
        """Replace the underlying source data and notify bound child tools.

        Parameters are the same as :meth:`set_data`, with the addition of
        ``emit_edited`` to indicate that the replacement should also be treated as a
        user edit for watcher propagation.
        """
        self.set_data(
            data,
            rad2deg=rad2deg,
            file_path=file_path,
            load_func=load_func,
            auto_compute=auto_compute,
            source_replaced=True,
        )
        if emit_edited:
            self.sigDataEdited.emit()

    def _update_if_delayed(self) -> None:
        if self._update_delayed:
            self._update_delayed = False
            self.sigDataChanged.emit()
            logger.debug("Data refresh triggered (delayed)")

            self.refresh_colormap()

    @property
    def reloadable(self) -> bool:
        """Check if the displayed data can be reloaded.

        Direct file-backed windows reload from their stored loader. Detached
        file-rooted provenance reloads by replaying its self-contained code.
        Managed script-derived ImageTools reload through the manager from their
        recorded inputs.

        Returns
        -------
        bool
            `True` if the data can be reloaded, `False` otherwise.
        """
        if self._direct_reloadable() or self._provenance_reloadable():
            return True
        manager = self._manager_instance if self._in_manager else None
        return manager is not None and manager._script_reload_from_slicer_area(
            self, execute=False
        )

    def _direct_reloadable(self) -> bool:
        """Return whether direct file metadata can reload the current source."""
        return (
            (self._file_path is not None)
            and self._file_path.exists()
            and self._load_func is not None
            and (callable(self._load_func[0]) or self._load_func[0] in erlab.io.loaders)
        )

    def _provenance_reloadable(self) -> bool:
        """Return whether replay provenance can rebuild the displayed data from file."""
        provenance_spec = self.provenance_spec
        return not (
            provenance_spec is None
            or provenance_spec.kind != "file"
            or provenance_spec.file_load_source is None
            or not pathlib.Path(provenance_spec.file_load_source.path).exists()
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
        return provenance._select_replay_input(
            reloaded,
            load_func[2],
        )

    def _fetch_for_provenance_reload(self) -> xr.DataArray:
        """Replay file-rooted provenance and return the active displayed data."""
        provenance_spec = self.provenance_spec
        if (
            provenance_spec is None
            or provenance_spec.kind != "file"
            or provenance_spec.file_load_source is None
        ):
            raise RuntimeError("Data cannot be reloaded from provenance")
        file_path = pathlib.Path(provenance_spec.file_load_source.path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        return provenance.replay_file_provenance(provenance_spec)

    def _fetch_reload_data(self) -> tuple[xr.DataArray, dict[str, typing.Any]]:
        """Return reload data and replacement kwargs for the active reload source."""
        provenance_spec = self.provenance_spec
        if (
            provenance_spec is not None
            and provenance_spec.kind == "file"
            and bool(provenance_spec.replay_stages)
            and self._provenance_reloadable()
        ):
            return self._fetch_for_provenance_reload(), {}
        if self._direct_reloadable():
            return (
                self._fetch_for_reload(),
                {"file_path": self._file_path, "load_func": self._load_func},
            )
        if self._provenance_reloadable():
            return self._fetch_for_provenance_reload(), {}
        raise RuntimeError("Data cannot be reloaded")

    def _reload(self) -> bool:
        """Reload the displayed data and return whether the reload succeeded."""
        if self._direct_reloadable() or self._provenance_reloadable():
            try:
                data, kwargs = self._fetch_reload_data()
                self._replace_reload_data(data, kwargs)
            except Exception:
                erlab.interactive.utils.MessageDialog.critical(
                    self, "Error", "An error occurred while reloading data."
                )
                return False
            return True
        manager = self._manager_instance if self._in_manager else None
        if manager is not None:
            return manager._script_reload_from_slicer_area(self, execute=True)
        if not self.reloadable:
            return False
        try:
            data, kwargs = self._fetch_reload_data()
            self._replace_reload_data(data, kwargs)
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self, "Error", "An error occurred while reloading data."
            )
            return False
        return True

    def _replace_reload_data(
        self,
        data: xr.DataArray,
        kwargs: dict[str, typing.Any],
    ) -> None:
        """Replace source data during reload and reapply the accepted filter."""
        accepted_filter_operation = self._accepted_filter_provenance_operation
        if accepted_filter_operation is None:
            self.replace_source_data(data, **kwargs)
            return

        filtered = self._filter_operation_result_for_replacement(
            data,
            accepted_filter_operation,
        )
        self.set_data(data, **kwargs)
        self._apply_filter_result(
            filtered,
            self._filter_func_from_operation(accepted_filter_operation),
            operation=accepted_filter_operation,
            accept=True,
            emit_edited=True,
        )

    @QtCore.Slot()
    def reload(self) -> None:
        """Reload the displayed data from recorded source information.

        Silently fails if the data cannot be reloaded. If an error occurs while
        reloading the data, a message is shown to the user.

        See Also
        --------
        :attr:`reloadable`
        """
        self._reload()

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
        if self._obj_shares_data_values:
            # Temporary view updates must detach the slicer buffer so previews and
            # derived views never mutate the public source array.
            self._replace_data_values(
                self.array_slicer._obj,
                self._owned_values_copy(self.array_slicer._obj),
            )
            self._obj_shares_data_values = False
            self.array_slicer.clear_val_cache()
        self.array_slicer._obj[:] = values

        if update:  # pragma: no branch
            self.array_slicer.clear_val_cache()
            self.refresh_all(only_plots=True)

            # This will update colorbar limits if visible
            self.lock_levels(self.levels_locked)

    def displayed_provenance_spec(
        self,
        base_spec: provenance.ToolProvenanceSpec | None = None,
    ) -> provenance.ToolProvenanceSpec | None:
        """Return provenance for the currently displayed data values."""
        if base_spec is None:
            base_spec = self.provenance_spec
        operation = self._accepted_filter_provenance_operation
        if operation is None:
            return base_spec
        return provenance.compose_full_provenance(
            base_spec,
            provenance.full_data(operation),
        )

    def apply_func(
        self,
        func: Callable[[xr.DataArray], xr.DataArray] | None,
        update: bool = True,
        *,
        operation: (provenance.ToolProvenanceOperation | None) = None,
        emit_edited: bool = False,
        preview: bool = True,
    ) -> None:
        """Apply a function to the data.

        The function must accept the data as the first argument and return a new
        DataArray. The returned DataArray must have the same dimensions and coordinates
        as the original data.

        This preview-only action is not recorded in the history, and the source data
        remains unchanged. Accepted filters must use :meth:`apply_filter_operation`.
        Only one preview function can be applied at a time.

        Parameters
        ----------
        func
            The function to apply to the data. if None, the data is restored.
        update
            If `True`, the plots are updated after setting the new values.
        operation
            Deprecated. Accepted filters must use :meth:`apply_filter_operation`.
        emit_edited
            If `True`, emit edit and source-change signals after updating the preview.
        preview
            Must be `True`. The argument is kept for existing internal callers.

        """
        if operation is not None:
            raise ValueError(
                "Use apply_filter_operation() for operation-backed filters"
            )
        if not preview:
            raise ValueError("apply_func() is preview-only")
        self._apply_filter_func(
            func,
            update=update,
            operation=None,
            accept=False,
            emit_edited=emit_edited,
        )

    def _apply_filter_func(
        self,
        func: Callable[[xr.DataArray], xr.DataArray] | None,
        update: bool = True,
        *,
        operation: (provenance.ToolProvenanceOperation | None) = None,
        accept: bool = False,
        emit_edited: bool = False,
    ) -> None:
        if func is None:
            self._applied_func = None
            self._applied_provenance_operation = None
            self._restore_obj_from_source(update=update)
            if accept:
                self._accepted_filter_provenance_operation = None
                self._accepted_filter_data = None
            if emit_edited:
                self.sigSourceDataReplaced.emit(self._tool_source_parent_data())
                self.sigDataEdited.emit()
            return

        filtered = self._filtered_data_for_display(func)
        self._apply_filter_result(
            filtered,
            func,
            update=update,
            operation=operation,
            accept=accept,
            emit_edited=emit_edited,
        )

    def _apply_filter_result(
        self,
        filtered: xr.DataArray,
        func: Callable[[xr.DataArray], xr.DataArray],
        update: bool = True,
        *,
        operation: (provenance.ToolProvenanceOperation | None) = None,
        accept: bool = False,
        emit_edited: bool = False,
    ) -> None:
        if filtered.chunks is None:
            self.update_values(filtered, update=update)
            self._applied_func = func
            self._applied_provenance_operation = operation
            if accept:
                self._accepted_filter_provenance_operation = operation
                self._accepted_filter_data = filtered.copy(deep=True)
            if emit_edited:
                self.sigSourceDataReplaced.emit(self._tool_source_parent_data())
                self.sigDataEdited.emit()
            return

        self.array_slicer.set_array(
            filtered,
            reset=False,
            copy_values=False,
            preserve_dims=tuple(filtered.dims),
        )
        self._obj_shares_data_values = False
        if update:
            self.array_slicer.clear_val_cache()
            self.refresh_all(only_plots=True)
            self.lock_levels(self.levels_locked)
        self._applied_func = func
        self._applied_provenance_operation = operation
        if accept:
            self._accepted_filter_provenance_operation = operation
            self._accepted_filter_data = filtered.copy(deep=True)
        if emit_edited:
            self.sigSourceDataReplaced.emit(self._tool_source_parent_data())
            self.sigDataEdited.emit()

    def apply_filter_operation(
        self,
        operation: (provenance.ToolProvenanceOperation | None),
        update: bool = True,
        *,
        emit_edited: bool = False,
        preview: bool = False,
    ) -> None:
        """Apply an operation-backed reversible display filter."""
        if operation is None:
            self._apply_filter_func(
                None,
                update=update,
                accept=not preview,
                emit_edited=emit_edited,
            )
            return
        filtered = self._filter_operation_result_for_data(
            self._data,
            operation,
            tuple(self.data.dims),
        )
        self._apply_filter_result(
            filtered,
            self._filter_func_from_operation(operation),
            update=update,
            operation=operation,
            accept=not preview,
            emit_edited=emit_edited,
        )

    def set_manual_limits(self, manual_limits: dict[str, list[float]]) -> None:
        """Set manual limits for the axes.

        Replaces the current manual limits with the given limits dictionary and updates
        all child axes accordingly.
        """
        valid_dims = set(self.data.dims)
        self.manual_limits = {
            dim: copy.deepcopy(limits)
            for dim, limits in manual_limits.items()
            if dim in valid_dims
        }
        for ax in self.axes:
            ax.update_manual_range()

    def _normalized_axis_inversions(
        self, axis_inversions: Mapping[str, bool] | None
    ) -> dict[str, bool]:
        if axis_inversions is None:
            return {}
        valid_dims = {str(dim) for dim in self.data.dims}
        return {
            str(dim): True
            for dim, inverted in axis_inversions.items()
            if bool(inverted) and str(dim) in valid_dims
        }

    def _axis_inversions_from_plotitem_states(
        self, plotitem_states: list[PlotItemState] | None
    ) -> dict[str, bool]:
        if plotitem_states is None:
            return {}
        axis_inversions: dict[str, bool] = {}
        for ax, plotitem_state in zip(self.axes, plotitem_states, strict=False):
            for dim, key in zip(
                ax.axis_dims_uniform, ("vb_x_inverted", "vb_y_inverted"), strict=True
            ):
                if dim is not None and bool(plotitem_state.get(key, False)):
                    axis_inversions[dim] = True
        return axis_inversions

    def apply_axis_inversions(self) -> None:
        for ax in self.axes:
            ax.update_axis_inversions()

    @link_slicer
    @record_history
    def set_axis_inverted(self, dim: str, inverted: bool) -> None:
        valid_dims = {str(dim_name) for dim_name in self.data.dims}
        if dim not in valid_dims:
            return
        if bool(self.axis_inversions.get(dim, False)) == inverted:
            return

        if inverted:
            self.axis_inversions[dim] = True
        else:
            self.axis_inversions.pop(dim, None)
        self.apply_axis_inversions()

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
            if k not in self.data.dims:
                continue
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

    @property
    def data_file_backed(self) -> bool:
        """Check if non-dask data is still lazily backed by a file."""
        sources = [self._data.encoding.get("source")]
        sources.extend(
            coord.encoding.get("source") for coord in self._data.coords.values()
        )
        return (
            self._data.chunks is None
            and not isinstance(self._data.variable._data, (np.ndarray, np.generic))
            and any(isinstance(value, (str, bytes, os.PathLike)) for value in sources)
        )

    @property
    def data_loadable(self) -> bool:
        """Check if the data can be explicitly loaded into memory."""
        return self.data_chunked or self.data_file_backed

    @QtCore.Slot()
    def _compute_chunked(self) -> None:
        """Load lazy data into memory.

        This method computes or loads the entire data array when it is dask-backed or
        lazily backed by a file.
        """
        if self.data_loadable:
            try:
                state = copy.deepcopy(self.state)
                with erlab.interactive.utils.wait_dialog(self, "Computing…"):
                    self.set_data(self._data.load())
                    self.state = state
                self.sigDataBackingChanged.emit()
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
        state = copy.deepcopy(self.state)
        with erlab.interactive.utils.wait_dialog(self, "Setting Chunks…"):
            self.set_data(self._data.chunk(chunks), auto_compute=False)
            self.state = state
        self.sigDataBackingChanged.emit()

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
    @record_history
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
        action_values = gamma is None and levels_locked is None and levels is None
        next_cmap = (
            cmap
            if cmap is not None
            else typing.cast("str | pg.ColorMap", self._colormap_properties["cmap"])
        )
        next_gamma = (
            float(gamma)
            if gamma is not None
            else float(self._colormap_properties["gamma"])
        )

        reverse_value = (
            self.reverse_act.isChecked()
            if reverse is None and action_values
            else reverse
        )
        high_contrast_value = (
            self.high_contrast_act.isChecked()
            if high_contrast is None and action_values
            else high_contrast
        )
        zero_centered_value = (
            self.zero_centered_act.isChecked()
            if zero_centered is None and action_values
            else zero_centered
        )

        next_reverse = bool(self._colormap_properties["reverse"])
        next_high_contrast = bool(self._colormap_properties["high_contrast"])
        next_zero_centered = bool(self._colormap_properties["zero_centered"])
        if reverse_value is not None:
            next_reverse = bool(reverse_value)
        if high_contrast_value is not None:
            next_high_contrast = bool(high_contrast_value)
        if zero_centered_value is not None:
            next_zero_centered = bool(zero_centered_value)

        pg_colormap = erlab.interactive.colors._pg_colormap_powernorm_lut(
            next_cmap,
            next_gamma,
            next_reverse,
            high_contrast=next_high_contrast,
            zero_centered=next_zero_centered,
        )

        self._colormap_properties.update(
            {
                "cmap": next_cmap,
                "gamma": next_gamma,
                "reverse": next_reverse,
                "high_contrast": next_high_contrast,
                "zero_centered": next_zero_centered,
            }
        )
        reverse_blocker = QtCore.QSignalBlocker(self.reverse_act)
        high_contrast_blocker = QtCore.QSignalBlocker(self.high_contrast_act)
        zero_centered_blocker = QtCore.QSignalBlocker(self.zero_centered_act)
        try:
            self.reverse_act.setChecked(next_reverse)
            self.high_contrast_act.setChecked(next_high_contrast)
            self.zero_centered_act.setChecked(next_zero_centered)
        finally:
            del reverse_blocker
            del high_contrast_blocker
            del zero_centered_blocker

        if levels_locked is not None:
            self.levels_locked = levels_locked
        if levels is not None:
            self.levels = levels

        for im in self._imageitems:
            im.set_pg_colormap(pg_colormap, update=update)
        self.sigViewOptionChanged.emit()

    @QtCore.Slot()
    def refresh_colormap(self) -> None:
        self.set_colormap(
            reverse=self.reverse_act.isChecked(),
            high_contrast=self.high_contrast_act.isChecked(),
            zero_centered=self.zero_centered_act.isChecked(),
            update=True,
        )
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
                target = manager.target_from_slicer_area(self)
                if target is not None:  # pragma: no branch
                    msg_box = QtWidgets.QMessageBox(self)
                    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                    msg_box.setText("Remove window?")
                    if isinstance(target, int):
                        informative_text = (
                            f"The ImageTool window at index {target} will be removed. "
                            "This cannot be undone."
                        )
                    else:
                        informative_text = (
                            "The current ImageTool window will be removed. "
                            "This cannot be undone."
                        )
                    msg_box.setInformativeText(informative_text)
                    msg_box.setStandardButtons(
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.Cancel
                    )
                    msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)

                    if msg_box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
                        if isinstance(target, int):
                            erlab.interactive.utils.single_shot(
                                manager, 0, lambda: manager.remove_imagetool(target)
                            )
                        else:
                            erlab.interactive.utils.single_shot(
                                manager, 0, lambda: manager._remove_childtool(target)
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
                if isinstance(
                    widget,
                    (
                        erlab.interactive.utils.ToolWindow,
                        erlab.interactive.imagetool.ImageTool,
                    ),
                ):
                    manager._add_childtool_from_slicerarea(widget, self)
                else:
                    manager.add_widget(widget)
                return

        if isinstance(widget, erlab.interactive.utils.ToolWindow):
            widget.set_source_parent_fetcher(lambda: self._tool_source_parent_data())
            self.sigSourceDataReplaced.connect(widget.handle_parent_source_replaced)
            widget.set_input_provenance_parent_fetcher(self.displayed_provenance_spec)

        uid: str = str(uuid.uuid4())
        with self._assoc_tools_lock:
            self._associated_tools[uid] = widget  # Store reference to prevent gc

        old_close_event = widget.closeEvent

        def new_close_event(event: QtGui.QCloseEvent) -> None:
            old_close_event(event)
            if event.isAccepted():
                with self._assoc_tools_lock:
                    if uid in self._associated_tools:
                        tool = self._associated_tools.pop(uid)
                        tool.deleteLater()

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

        dim_values = {
            str(dim): float(value)
            for dim, value in zip(self.data.dims, self.current_values, strict=True)
        }
        beta_value = dim_values.get("beta")
        if beta_value is None and "beta" in self.data.coords:
            beta_coord = self.data["beta"]
            if beta_coord.size == 1:
                beta_value = float(beta_coord.values)
        if "alpha" in dim_values and beta_value is not None:
            initial_normal_emission: tuple[float, float] | None = (
                dim_values["alpha"],
                beta_value,
            )
            initial_delta: float | None = None
            guideline_dims = tuple(
                str(self.data.dims[axis]) for axis in self.main_image.display_axis
            )
            if self.main_image.is_guidelines_visible and set(guideline_dims) == {
                "alpha",
                "beta",
            }:
                guideline_values: dict[str, float] = {}
                for axis, value in zip(
                    self.main_image.display_axis,
                    self.main_image._guideline_offset,
                    strict=True,
                ):
                    dim = str(self.data.dims[axis])
                    if axis in self.array_slicer._nonuniform_axes_set:
                        value = float(
                            np.interp(
                                value,
                                self.array_slicer.coords_uniform[axis],
                                self.array_slicer.coords[axis],
                            )
                        )
                    guideline_values[dim] = float(value)
                initial_normal_emission = (
                    guideline_values["alpha"],
                    guideline_values["beta"],
                )
                initial_delta = -self.main_image._guideline_angle
        else:
            initial_normal_emission = None
            initial_delta = None

        tool = erlab.interactive.ktool(
            self.data,
            cmap=cmap,
            gamma=gamma,
            data_name=self.watched_data_name,
            initial_normal_emission=initial_normal_emission,
            initial_delta=initial_delta,
            execute=False,
        )
        if isinstance(tool, erlab.interactive.utils.ToolWindow):
            tool.set_source_binding(provenance.full_data())
        self.add_tool_window(tool)

    @QtCore.Slot()
    def open_in_meshtool(self) -> None:
        """Open the interactive mesh removal tool."""
        tool = erlab.interactive.meshtool(
            self.data, data_name=self.watched_data_name, execute=False
        )
        if isinstance(tool, erlab.interactive.utils.ToolWindow):
            tool.set_source_binding(provenance.full_data())
        self.add_tool_window(tool)

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
