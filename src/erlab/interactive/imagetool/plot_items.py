"""Plotting primitives and interactive graphics items for ImageTool."""

from __future__ import annotations

import collections
import logging
import os
import time
import typing
import weakref

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import xarray as xr
from pyqtgraph.GraphicsScene import mouseEvents
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool.viewer import (
    PlotItemState,
    _make_cursor_colors,
    record_history,
    suppress_history,
    suppressnanwarning,
)

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    import dask.array
    import matplotlib.colors
    import qtawesome

    from erlab.interactive.imagetool.viewer import ImageSlicerArea
else:
    import lazy_loader as _lazy

    qtawesome = _lazy.load("qtawesome")

logger = logging.getLogger(__name__)


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

        self.vb.menu.addSeparator()

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

            ftool_action = self.vb.menu.addAction("ftool")
            ftool_action.triggered.connect(self.open_in_ftool)

            croppable_actions.extend(
                (goldtool_action, restool_action, dtool_action, ftool_action)
            )

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
                for act in (
                    goldtool_action,
                    restool_action,
                    dtool_action,
                    ftool_action,
                ):
                    act.setIcon(qtawesome.icon("mdi6.export"))
                    act.setIconVisibleInMenu(True)

            self._sigPaletteChanged.connect(_set_icons)
            _set_icons()

        else:
            ftool_action = self.vb.menu.addAction("ftool")
            ftool_action.triggered.connect(self.open_in_ftool)
            ftool_action.setIcon(qtawesome.icon("mdi6.export"))
            ftool_action.setIconVisibleInMenu(True)
            croppable_actions.append(ftool_action)

            self.vb.menu.addSeparator()

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

    def _uniform_qsel_kwargs_multicursor(
        self,
    ) -> tuple[dict[Hashable, float | list[float]], Hashable | None]:
        """Generate multi-cursor ``qsel`` keyword arguments for uniform axes.

        Returns
        -------
        tuple[dict[Hashable, float | list[float]], Hashable | None]
            A 2-tuple of:

            - ``qsel_kwargs``: keyword arguments to pass to
              :meth:`xarray.DataArray.qsel`. Keys that vary across cursors are stored
              as lists and ordered first.
            - ``variable_dim``: the varying dimension key (without ``_width``), or
              `None` if no key varies across cursors.

        Raises
        ------
        ValueError
            If any non-display axis is non-uniform, if more than one dimension varies
            across cursors, or if only width keys vary across cursors.
        """
        if any(
            a in self.array_slicer._nonuniform_axes
            for a in set(range(self.slicer_area.data.ndim)) - set(self.display_axis)
        ):
            raise ValueError(
                "Cannot generate uniform qsel kwargs when indexing along "
                "non-uniform axes."
            )

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

        variable_dim_name = self._multicursor_variable_key(varying)

        # Put varying keys first
        variable_keys = (variable_dim_name, f"{variable_dim_name}_width")
        other_keys = sorted((k for k in result if k not in variable_keys), key=str)
        ordered_keys = [k for k in variable_keys if k in result] + other_keys
        result = {k: result[k] for k in ordered_keys}

        return result, variable_dim_name

    def _multicursor_selection_plan(
        self,
        *,
        data_name: str,
        non_display_axes: tuple[int, ...],
        has_nonuniform_non_display_axes: bool,
    ) -> tuple[
        dict[Hashable, float | list[float]] | None,
        list[str] | None,
        Hashable | None,
        set[Hashable],
    ]:
        """Build a unified selection plan used by multi-cursor code generation.

        Parameters
        ----------
        data_name
            Variable name used in generated code (e.g. ``"data"``).
        non_display_axes
            Axis indices that are selected over, i.e. axes not shown in this plot.
        has_nonuniform_non_display_axes
            Whether any axis in ``non_display_axes`` is non-uniform.

        Returns
        -------
        tuple
            A 4-tuple ``(qsel_kwargs, selection_exprs, variable_dim, selected_dims)``:

            - ``qsel_kwargs``: populated for uniform selection paths, else `None`.
            - ``selection_exprs``: populated for non-uniform selection paths, else
              `None`.
            - ``variable_dim``: varying dimension key (without ``_width``), or `None`.
            - ``selected_dims``: set of selected dimension keys used to infer plotted
              dimension order.

        Raises
        ------
        ValueError
            Propagated from variable-key validation when cursor variation cannot be
            represented by a single varying dimension.
        """
        if has_nonuniform_non_display_axes:
            varying: list[Hashable] = []
            for axis in non_display_axes:
                dim_name = self._selection_dim_name(axis)
                center_values = [
                    self.array_slicer.get_index(cursor, axis)
                    for cursor in range(self.slicer_area.n_cursors)
                ]
                width_values = [
                    (self.array_slicer.get_bins(cursor)[axis] if binned[axis] else 0)
                    for cursor in range(self.slicer_area.n_cursors)
                    for binned in (self.array_slicer.get_binned(cursor),)
                ]

                if len(set(center_values)) > 1:
                    varying.append(dim_name)
                if len(set(width_values)) > 1:
                    varying.append(f"{dim_name}_width")

            variable_dim = self._multicursor_variable_key(varying)
            selection_exprs = [
                self._selection_expr_for_cursor(data_name, cursor, non_display_axes)
                for cursor in range(self.slicer_area.n_cursors)
            ]
            if variable_dim is None:
                selection_exprs = selection_exprs[:1]
            selected_dims = {
                self._selection_dim_name(axis) for axis in non_display_axes
            }
            return None, selection_exprs, variable_dim, selected_dims

        qsel_kwargs, variable_dim = self._uniform_qsel_kwargs_multicursor()
        selected_dims = set(qsel_kwargs)
        return qsel_kwargs, None, variable_dim, selected_dims

    def _multicursor_variable_key(self, varying: list[Hashable]) -> Hashable | None:
        """Validate varying keys and return the effective varying dimension key.

        Parameters
        ----------
        varying
            List of keys that differ across cursors. Keys may include ``_width``
            suffixes.

        Returns
        -------
        Hashable or None
            Dimension key that varies across cursors (without ``_width``), or `None`
            when no key varies.

        Raises
        ------
        ValueError
            If more than one independent dimension varies, or if only width keys vary.
        """
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
                return None
            case 1:
                return varying[0]
            case _:
                for key in varying:
                    if not str(key).endswith("_width"):
                        return key
                return None  # pragma: no cover

    def _selection_dim_name(self, axis: int) -> Hashable:
        """Return user-facing dimension name for a non-display axis.

        Internal non-uniform ``*_idx`` dimensions are mapped back to their original
        coordinate dimension names.
        """
        dim_name: Hashable = self.slicer_area.data.dims[axis]
        if (
            axis in self.array_slicer._nonuniform_axes
            and isinstance(dim_name, str)
            and dim_name.endswith("_idx")
        ):
            return dim_name.removesuffix("_idx")
        return dim_name

    def _selection_expr_for_cursor(
        self, data_name: str, cursor: int, non_display_axes: tuple[int, ...]
    ) -> str:
        """Build a per-cursor mixed ``isel``/``qsel`` selection expression.

        Parameters
        ----------
        data_name
            Variable name used in generated code.
        cursor
            Cursor index used to obtain current selection and binning state.
        non_display_axes
            Axis indices that are selected over for this plot.

        Non-uniform non-display axes are selected with ``isel`` while uniform
        non-display axes are selected with ``qsel`` (including width terms for binned
        selections). Binned non-uniform non-display axes are averaged via
        ``qsel.average`` after indexing.

        Returns
        -------
        str
            A valid Python expression that evaluates to a selected
            :class:`xarray.DataArray`.
        """
        isel_kwargs: dict[Hashable, slice | int] = {}
        qsel_kwargs: dict[Hashable, float] = {}
        avg_nonuniform_dims: list[str] = []
        binned = self.array_slicer.get_binned(cursor)

        for axis in non_display_axes:
            if axis in self.array_slicer._nonuniform_axes:
                dim_name = self._selection_dim_name(axis)
                isel_kwargs[dim_name] = self.array_slicer._bin_slice(
                    cursor, axis, int_if_one=True
                )
                if binned[axis]:
                    avg_nonuniform_dims.append(str(dim_name))
                continue

            # Build qsel args one axis at a time so non-uniform axes are never passed
            # into qsel_args.
            disp_for_axis = tuple(
                i for i in range(self.slicer_area.data.ndim) if i != axis
            )
            qsel_kwargs.update(self.array_slicer.qsel_args(cursor, disp_for_axis))

        selected = data_name
        if isel_kwargs:
            selected += f".isel({erlab.interactive.utils.format_kwargs(isel_kwargs)})"
        if qsel_kwargs:
            selected += f".qsel({erlab.interactive.utils.format_kwargs(qsel_kwargs)})"
        if avg_nonuniform_dims:

            def _double_quoted_literal(value: str) -> str:
                if '"' in value:
                    return f"'{value}'"
                return f'"{value}"'

            if len(avg_nonuniform_dims) == 1:
                avg_arg = _double_quoted_literal(avg_nonuniform_dims[0])
            else:
                avg_arg = (
                    "("
                    + ", ".join(
                        _double_quoted_literal(dim) for dim in avg_nonuniform_dims
                    )
                    + ")"
                )
            selected += f".qsel.average({avg_arg})"
        return selected

    def _plot_code_multicursor(self, *, placeholder_name: str | None = None) -> str:
        """Generate matplotlib plot code for all cursors.

        Parameters
        ----------
        placeholder_name
            Optional data variable name to use in generated code.

        Returns
        -------
        str
            Executable Python code that creates a matplotlib figure.
        """
        if placeholder_name is not None:
            data_name: str = placeholder_name
        else:
            data_name = self.slicer_area.watched_data_name or "data"

        non_display_axes = tuple(
            sorted(set(range(self.slicer_area.data.ndim)) - set(self.display_axis))
        )
        has_nonuniform_non_display_axes = any(
            axis in self.array_slicer._nonuniform_axes for axis in non_display_axes
        )
        result, selection_exprs, variable_dim, selected_dims = (
            self._multicursor_selection_plan(
                data_name=data_name,
                non_display_axes=non_display_axes,
                has_nonuniform_non_display_axes=has_nonuniform_non_display_axes,
            )
        )

        # Determine order of plotted dimensions
        dim_order_plot = list(self.slicer_area._data.dims)
        for k in selected_dims:
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
                data_name,
                variable_dim,
                dim_order_plot,
                qsel_kwargs=result,
                selected_maps=selection_exprs,
            )

        return self._plot_code_line(
            data_name,
            variable_dim,
            dim_order_plot[0],
            qsel_kwargs=result,
            selected_lines=selection_exprs,
        )

    def _plot_code_line(
        self,
        data_name: str,
        variable_dim: Hashable | None,
        x_dim: Hashable,
        *,
        qsel_kwargs: dict[Hashable, float | list[float]] | None = None,
        selected_lines: list[str] | None = None,
    ) -> str:
        """Generate matplotlib code for 1D multi-cursor plots.

        Parameters
        ----------
        data_name
            Variable name used in generated code.
        variable_dim
            Varying dimension key across cursors, or `None` when all cursor selections
            are equivalent.
        x_dim
            Dimension plotted along the x-axis.
        qsel_kwargs
            Uniform-path ``qsel`` kwargs.
        selected_lines
            Non-uniform-path selected line expressions.

        Returns
        -------
        str
            Executable Python code that plots one or more lines on a matplotlib axis.

        Raises
        ------
        ValueError
            If neither or both selection inputs are provided.
        """
        TAB = "    "
        plot_lines = ["fig, ax = plt.subplots()"]
        iterable_expr: str | None = None

        if (qsel_kwargs is None) == (selected_lines is None):  # pragma: no cover
            raise ValueError(
                "Exactly one of qsel_kwargs or selected_lines must be provided."
            )

        if selected_lines is not None:
            if variable_dim is None:
                plot_lines.append(f"{selected_lines[0]}.plot(ax=ax)")
                return "\n".join(plot_lines)
        else:
            qsel_kwargs_nonnull = typing.cast(
                "dict[Hashable, float | list[float]]", qsel_kwargs
            )
            selected = (
                f"{data_name}.qsel("
                f"{erlab.interactive.utils.format_kwargs(qsel_kwargs_nonnull)})"
            )
            if variable_dim is None:
                plot_lines.append(selected + ".plot(ax=ax)")
                return "\n".join(plot_lines)
            iterable_expr = selected + f'.transpose("{variable_dim}", ...)'
            selected_lines = None

        colors: list[str] = [
            self.slicer_area.cursor_colors[i].name()
            for i in range(self.slicer_area.n_cursors)
        ]
        default_colors = erlab.interactive._options.schema.ColorOptions.model_fields[
            "cursors"
        ].default
        colors_changed = any(c not in default_colors for c in colors)
        is_normalized: bool = self.slicer_data_items[
            self.slicer_area.current_cursor
        ].normalize

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
            if iterable_expr is not None:
                plot_lines.append(f"for i, line in enumerate({iterable_expr}):")
            else:
                plot_lines.append("for i, line in enumerate([")
        else:
            if iterable_expr is not None:
                plot_lines.append(f"for line in {iterable_expr}:")
            else:
                plot_lines.append("for line in [")

        if selected_lines is not None:
            plot_lines.extend(f"{TAB}{selected}," for selected in selected_lines)
            plot_lines.append("]):" if colors_changed else "]:")

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
        variable_dim: Hashable | None,
        dim_order_plot: list[Hashable],
        *,
        qsel_kwargs: dict[Hashable, float | list[float]] | None = None,
        selected_maps: list[str] | None = None,
    ) -> str:
        """Generate matplotlib code for image/slice multi-cursor plots.

        Parameters
        ----------
        data_name
            Variable name used in generated code.
        variable_dim
            Varying dimension key across cursors, or `None` when selections are
            equivalent.
        dim_order_plot
            Ordered plotted dimensions used to derive axis-orientation kwargs.
        qsel_kwargs
            Uniform-path ``qsel`` kwargs.
        selected_maps
            Non-uniform-path selected map expressions.

        Returns
        -------
        str
            Executable Python code that calls :func:`erlab.plotting.plot_slices`.

        Raises
        ------
        ValueError
            If neither or both selection inputs are provided.
        """
        plot_kwargs = self._plot_code_image_common_kwargs(dim_order_plot)

        if (qsel_kwargs is None) == (selected_maps is None):  # pragma: no cover
            raise ValueError(
                "Exactly one of qsel_kwargs or selected_maps must be provided."
            )

        if selected_maps is not None:
            plot_lines: list[str] = []
            if variable_dim is None:
                plot_lines.append(f"selected = {selected_maps[0]}")
            else:
                plot_lines.append("selected = [")
                plot_lines.extend(f"    {selected}," for selected in selected_maps)
                plot_lines.append("]")
            plot_lines.append(
                erlab.interactive.utils.generate_code(
                    erlab.plotting.plot_slices,
                    args=["|selected|"],
                    kwargs=plot_kwargs,
                    module="eplt",
                    assign=("fig", "axs" if variable_dim else "ax"),
                )
            )
            return "\n".join(plot_lines)

        # Setup plot keyword arguments
        qsel_kwargs_nonnull = typing.cast(
            "dict[Hashable, float | list[float]]", qsel_kwargs
        )
        plot_args: list[typing.Any] = [f"|[{data_name}]|"]
        if all(isinstance(k, str) and k.isidentifier() for k in qsel_kwargs_nonnull):
            plot_kwargs.update({str(k): v for k, v in qsel_kwargs_nonnull.items()})
        else:
            plot_args.append(
                f"|**{erlab.interactive.utils.format_kwargs(qsel_kwargs_nonnull)}|"
            )

        return erlab.interactive.utils.generate_code(
            erlab.plotting.plot_slices,
            args=plot_args,
            kwargs=plot_kwargs,
            module="eplt",
            assign=("fig", "axs" if variable_dim else "ax"),
        )

    def _plot_code_image_common_kwargs(
        self, dim_order_plot: list[Hashable]
    ) -> dict[str, typing.Any]:
        """Return shared ``plot_slices`` keyword arguments for image code paths.

        Parameters
        ----------
        dim_order_plot
            Ordered plotted dimensions used to infer transpose and crop limits.

        Returns
        -------
        dict[str, Any]
            Keyword arguments shared by uniform and non-uniform image code-generation
            paths.
        """
        plot_kwargs: dict[str, typing.Any] = {}
        plot_dims = list(dim_order_plot)

        if plot_dims[0] != self.axis_dims[0]:
            plot_kwargs["transpose"] = True
            plot_dims.reverse()

        for k, v in self._crop_indexers.items():
            if k == plot_dims[0]:
                plot_kwargs["xlim"] = (float(v.start), float(v.stop))
            elif k == plot_dims[1]:  # pragma: no branch
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

        return plot_kwargs

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
    def open_in_ftool(self) -> None:
        self.slicer_area.add_tool_window(
            erlab.interactive.ftool(
                self.current_data.squeeze(),
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
            self.slicer_area.sigIndexChanged.emit(cursors, (axis,))

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
