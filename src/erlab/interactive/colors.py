"""Functions for manipulating colors in Qt."""

from __future__ import annotations

__all__ = [
    "BetterColorBarItem",
    "BetterImageItem",
    "ColorCycleDialog",
    "ColorMapComboBox",
    "ColorMapGammaWidget",
    "color_to_QColor",
    "is_dark_mode",
    "pg_colormap_from_name",
    "pg_colormap_names",
    "pg_colormap_powernorm",
    "pg_colormap_to_QPixmap",
]

import contextlib
import importlib.util
import typing
import weakref
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    from matplotlib.typing import ColorType


def load_all_colormaps() -> None:
    """Load all colormaps from additional sources."""
    import erlab.plotting

    for package in erlab.interactive.options.model.colors.cmap.packages:
        if importlib.util.find_spec(package):
            importlib.import_module(package)


class ColorMapComboBox(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setPlaceholderText("Select colormap...")
        self.setToolTip("Colormap")
        self.setIconSize(QtCore.QSize(64, 16))

        for name in pg_colormap_names("matplotlib", exclude_local=True):
            self.addItem(name)

        self.thumbnails_loaded: bool = False
        self.loaded_all: bool = False
        self.currentIndexChanged.connect(self.load_thumbnail)
        self.default_cmap: str | None = None

        sc_p = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Alt+Up"), self)
        sc_p.activated.connect(self._prev_idx)
        sc_m = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Alt+Down"), self)
        sc_m.activated.connect(self._next_idx)

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self._menu = QtWidgets.QMenu("Menu", self)
        self._menu.addAction("Load All Colormaps", self.load_all)

    @QtCore.Slot(QtCore.QPoint)
    def _show_menu(self, position: QtCore.QPoint) -> None:
        if not self.loaded_all:
            self._menu.popup(self.mapToGlobal(position))

    def load_thumbnail(self, index: int) -> None:
        if not self.thumbnails_loaded:
            text = self.itemText(index)
            with contextlib.suppress(RuntimeError):
                self.setItemIcon(index, QtGui.QIcon(pg_colormap_to_QPixmap(text)))

    def load_all(self) -> None:
        with QtCore.QSignalBlocker(self):
            load_all_colormaps()
            self.loaded_all = True
            self.clear()
            for name in pg_colormap_names("all", exclude_local=True):
                self.addItem(QtGui.QIcon(pg_colormap_to_QPixmap(name)), name)
        self.thumbnails_loaded = False
        self.resetCmap()

    # https://forum.qt.io/topic/105012/qcombobox-specify-width-less-than-content/11
    def showPopup(self) -> None:
        maxWidth = self.maximumWidth()
        if maxWidth and maxWidth < 16777215:  # default maxwidth of QWidgets
            self.setPopupMinimumWidthForItems()
        if not self.thumbnails_loaded:
            for i in range(self.count()):
                self.load_thumbnail(i)
            self.thumbnails_loaded = True
        super().showPopup()

    def setPopupMinimumWidthForItems(self) -> None:
        view = self.view()
        fm = self.fontMetrics()
        maxWidth = max(
            fm.boundingRect(self.itemText(i)).width() for i in range(self.count())
        )
        if maxWidth and view is not None:
            view.setMinimumWidth(maxWidth)

    @QtCore.Slot()
    def _next_idx(self) -> None:
        self.wheelEvent(
            QtGui.QWheelEvent(
                QtCore.QPointF(0, 0),
                QtCore.QPointF(0, 0),
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, -15),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.ScrollPhase.ScrollUpdate,
                True,
            )
        )

    @QtCore.Slot()
    def _prev_idx(self) -> None:
        self.wheelEvent(
            QtGui.QWheelEvent(
                QtCore.QPointF(0, 0),
                QtCore.QPointF(0, 0),
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, 15),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.ScrollPhase.ScrollUpdate,
                True,
            )
        )

    def hidePopup(self) -> None:
        self.activated.emit(self.currentIndex())
        self.textActivated.emit(self.currentText())
        self.currentIndexChanged.emit(self.currentIndex())
        self.currentTextChanged.emit(self.currentText())
        super().hidePopup()

    def setDefaultCmap(self, cmap: str) -> None:
        self.default_cmap = cmap
        self.setCurrentText(cmap)

    def resetCmap(self) -> None:
        if self.default_cmap is None:
            self.setCurrentIndex(0)
        else:
            self.setCurrentText(self.default_cmap)

    def setCurrentText(self, text: str | None) -> None:
        """Set the current text of the combobox."""
        if self.findText(text) < 0:
            # If the value is not in the combobox, try loading all and retry
            self.load_all()
            self.setCurrentText(text)
        else:
            super().setCurrentText(text)


class ColorMapGammaWidget(QtWidgets.QWidget):
    """Slider and spinbox for adjusting colormap gamma.

    Signals
    -------
    valueChanged(float)

    """

    valueChanged = QtCore.Signal(float)  #: :meta private:

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        value: float = 1.0,
        slider_cls: type | None = None,
        spin_cls: type | None = None,
    ) -> None:
        super().__init__(parent=parent)
        layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        if slider_cls is None:
            slider_cls = QtWidgets.QSlider
        if spin_cls is None:
            spin_cls = QtWidgets.QDoubleSpinBox

        self.spin = spin_cls(
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
        self.label = QtWidgets.QLabel("𝛾", self)
        self.label.setBuddy(self.spin)
        # self.label.setIndent(0)
        self.slider = slider_cls(
            self,
            toolTip="Colormap gamma",
            value=self.gamma_scale(value),
            singleStep=1,
            orientation=QtCore.Qt.Orientation.Horizontal,
        )
        self.slider.setMinimumWidth(75)
        self.spin.valueChanged.connect(self.spin_changed)

        self.slider.setRange(
            self.gamma_scale(self.spin.minimum()), self.gamma_scale(self.spin.maximum())
        )
        self.slider.valueChanged.connect(self.slider_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.spin)
        layout.addWidget(self.slider)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)
        self.slider.setValue(self.gamma_scale(value))

    def spin_changed(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self.gamma_scale(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def slider_changed(self, value: float) -> None:
        self.spin.setValue(self.gamma_scale_inv(value))

    def gamma_scale(self, y: float) -> int:
        return round(1e4 * np.log10(y))

    def gamma_scale_inv(self, x: float) -> float:
        return np.power(10, x * 1e-4)


class BetterImageItem(pg.ImageItem):
    """:class:`pyqtgraph.ImageItem` with improved colormap support.

    Parameters
    ----------
    image
        Image data
    **kwargs
        Additional arguments to :class:`pyqtgraph.ImageItem`.

    Signals
    -------
    sigColorChanged()
    sigLimitChanged(float, float)

    """

    sigColorChanged = QtCore.Signal()  #: :meta private:

    def __init__(self, image: npt.NDArray | None = None, **kwargs) -> None:
        super().__init__(image, **kwargs)

    def set_colormap(
        self,
        cmap: pg.ColorMap | str,
        gamma: float,
        reverse: bool = False,
        high_contrast: bool = False,
        zero_centered: bool = False,
        update: bool = True,
    ) -> None:
        cmap = pg_colormap_powernorm(
            cmap,
            gamma,
            reverse,
            high_contrast=high_contrast,
            zero_centered=zero_centered,
        )
        self.set_pg_colormap(cmap, update=update)

    def set_pg_colormap(self, cmap: pg.ColorMap, update: bool = True) -> None:
        self._colorMap = cmap
        self.setLookupTable(cmap.getStops()[1], update=update)
        self.sigColorChanged.emit()


class TrackableLinearRegionItem(pg.LinearRegionItem):
    sigRegionChangeStarted = QtCore.Signal(object)  #: :meta private:

    def mouseDragEvent(self, ev) -> None:
        if not self.movable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()

        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [ln.pos() - bdp for ln in self.lines]
            self.startPositions = [ln.pos() for ln in self.lines]
            self.moving = True
            self.sigRegionChangeStarted.emit(self)

        if not self.moving:
            return

        self.lines[0].blockSignals(True)  # only want to update once
        for i, ln in enumerate(self.lines):
            ln.setPos(self.cursorOffsets[i] + ev.pos())
        self.lines[0].blockSignals(False)
        self.prepareGeometryChange()

        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)


class _ColorBarLimitWidget(QtWidgets.QWidget):
    def __init__(self, colorbar: BetterColorBarItem) -> None:
        super().__init__()
        self.setGeometry(QtCore.QRect(0, 640, 242, 182))
        self.cb = colorbar

        layout = QtWidgets.QFormLayout(self)
        self.setLayout(layout)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.max_spin = pg.SpinBox(dec=True, compactHeight=False, finite=False)
        self.min_spin = pg.SpinBox(dec=True, compactHeight=False, finite=False)
        self.rst_btn = QtWidgets.QPushButton("Reset")

        self.max_spin.setObjectName("_vmax_spin")
        self.min_spin.setObjectName("_vmin_spin")
        self.rst_btn.setObjectName("_vlim_reset_btn")

        layout.addRow("Max", self.max_spin)
        layout.addRow("Min", self.min_spin)
        layout.addRow(self.rst_btn)

        self.cb._span.sigRegionChanged.connect(self.region_changed)
        self.min_spin.sigValueChanged.connect(self.update_region)
        self.max_spin.sigValueChanged.connect(self.update_region)
        self.rst_btn.clicked.connect(self.reset)

    def _set_spin_values(self, mn: float, mx: float) -> None:
        self.max_spin.blockSignals(True)
        self.min_spin.blockSignals(True)
        self.max_spin.setValue(mx)
        self.min_spin.setValue(mn)
        self.max_spin.blockSignals(False)
        self.min_spin.blockSignals(False)

    @QtCore.Slot()
    def region_changed(self):
        mn, mx = self.cb._span.getRegion()
        self._set_spin_values(mn, mx)

    @QtCore.Slot()
    def center_zero(self):
        old_min, old_max = self.cb._span.getRegion()
        self.reset()

        mn, mx = self.cb._span.getRegion()
        if mn < 0 < mx:
            half_len = min(abs(mn), abs(mx))
            self._set_spin_values(-half_len, half_len)
        else:
            self._set_spin_values(old_min, old_max)

        self.update_region()

    @QtCore.Slot()
    def reset(self):
        self._set_spin_values(-np.inf, np.inf)
        self.update_region()

    @QtCore.Slot()
    def update_region(self):
        self.cb.setSpanRegion((self.min_spin.value(), self.max_spin.value()))

    def setVisible(self, visible: bool) -> None:
        super().setVisible(visible)
        if visible:
            self.region_changed()


class _ColorBarEditWidget(QtWidgets.QWidget):
    def __init__(self, colorbar: BetterColorBarItem) -> None:
        super().__init__()
        self.cb = colorbar

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._cmap_combo = ColorMapComboBox()
        self._gamma_widget = ColorMapGammaWidget()
        self._reversed_check = QtWidgets.QCheckBox("Reverse")
        self._high_contrast_check = QtWidgets.QCheckBox("High Contrast")

        self._cmap_combo.currentTextChanged.connect(self.apply_changes)
        self._gamma_widget.valueChanged.connect(self.apply_changes)
        self._reversed_check.toggled.connect(self.apply_changes)
        self._high_contrast_check.toggled.connect(self.apply_changes)

        layout.addWidget(self._cmap_combo)
        layout.addWidget(self._gamma_widget)
        layout.addWidget(self._reversed_check)
        layout.addWidget(self._high_contrast_check)

    @QtCore.Slot()
    def apply_changes(self) -> None:
        cmap_name: str = self._cmap_combo.currentText()
        gamma: float = self._gamma_widget.value()
        reverse: bool = self._reversed_check.isChecked()
        high_contrast: bool = self._high_contrast_check.isChecked()

        for img_ref in self.cb.images:
            img = img_ref()
            if img is not None:
                img.set_colormap(
                    cmap_name,
                    gamma,
                    reverse=reverse,
                    high_contrast=high_contrast,
                    update=True,
                )

    def populate_cmap_info(self) -> None:
        with (
            QtCore.QSignalBlocker(self._cmap_combo),
            QtCore.QSignalBlocker(self._gamma_widget),
            QtCore.QSignalBlocker(self._reversed_check),
            QtCore.QSignalBlocker(self._high_contrast_check),
        ):
            cmap = self.cb.primary_image()._colorMap

            if hasattr(cmap, "_erlab_attrs"):
                self._cmap_combo.setCurrentText(cmap.name)
                self._gamma_widget.setValue(cmap._erlab_attrs["gamma"])
                self._reversed_check.setChecked(cmap._erlab_attrs["reverse"])
                self._high_contrast_check.setChecked(
                    cmap._erlab_attrs.get("high_contrast", False)
                )

    def setVisible(self, visible: bool) -> None:
        super().setVisible(visible)
        if visible:
            self.populate_cmap_info()


class BetterColorBarItem(pg.PlotItem):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        image: Iterable[BetterImageItem] | BetterImageItem | None = None,
        autoLevels: bool = False,
        limits: tuple[float, float] | None = None,
        pen: QtGui.QPen | str = "c",
        hoverPen: QtGui.QPen | str = "m",
        hoverBrush: QtGui.QBrush | str = "#FFFFFF33",
        *,
        show_colormap_edit_menu: bool = True,
        **kargs,
    ) -> None:
        super().__init__(parent, **kargs)

        self.setDefaultPadding(0)
        # self.hideButtons()
        self.setMenuEnabled(False, enableViewBoxMenu=True)
        self.vb.setMouseEnabled(x=False, y=True)

        self._colorbar = pg.ImageItem(
            np.linspace(0, 1, 4096).reshape((-1, 1)),
            axisOrder="row-major",
            autoDownsample=False,
        )
        self.addItem(self._colorbar)

        self._span = TrackableLinearRegionItem(
            (0, 1),
            "horizontal",
            swapMode="block",
            pen=pen,
            brush=pg.mkBrush(None),
            hoverPen=hoverPen,
            hoverBrush=hoverBrush,
        )
        self._span.setZValue(1000)
        self._span.lines[0].addMarker("<|>", size=6)
        self._span.lines[1].addMarker("<|>", size=6)

        self.addItem(self._span)

        self._fixedlimits: tuple[float, float] | None = None
        self.setAutoLevels(autoLevels)
        self._images: set[weakref.ref[BetterImageItem]] = set()
        self._primary_image: weakref.ref[BetterImageItem] | None = None

        # Add colorbar limit editor to context menu
        self.vb.menu.addSeparator()
        self._clim_menu: QtWidgets.QMenu = self.vb.menu.addMenu("Edit color limits")
        clw = _ColorBarLimitWidget(self)
        act = QtWidgets.QWidgetAction(self._clim_menu)
        act.setDefaultWidget(clw)
        self._clim_menu.addAction(act)

        center_zero_action = self.vb.menu.addAction("Center zero")
        center_zero_action.triggered.connect(clw.center_zero)

        if show_colormap_edit_menu:
            self._cmap_menu: QtWidgets.QMenu = self.vb.menu.addMenu("Edit colormap")
            cew = _ColorBarEditWidget(self)
            act = QtWidgets.QWidgetAction(self._cmap_menu)
            act.setDefaultWidget(cew)
            self._cmap_menu.addAction(act)

        if image is not None:
            self.setImageItem(image)
        if limits is not None:
            self.setLimits(limits)
        self.setLabels(right=("", ""))
        self.set_dimensions()

    @property
    def images(self):
        return self._images

    @property
    def primary_image(self):
        return self._primary_image

    @property
    def levels(self) -> Sequence[float]:
        return self.primary_image().getLevels()

    @property
    def limits(self) -> tuple[float, float]:
        if self._fixedlimits is not None:
            return self._fixedlimits
        return self.primary_image().quickMinMax(targetSize=2**16)

    def set_width(self, width: int) -> None:
        self.layout.setColumnFixedWidth(1, width)

    def set_dimensions(
        self,
        horiz_pad: int | None = None,
        vert_pad: int | None = None,
        font_size: float = 11.0,
    ) -> None:
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.showAxes((True, True, True, True), showValues=(False, False, True, False))
        self.getAxis("top").setHeight(vert_pad)
        self.getAxis("bottom").setHeight(vert_pad)
        self.getAxis("right").setWidth(horiz_pad)
        self.getAxis("left").setWidth(None)

        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        for axis in ("left", "bottom", "right", "top"):
            self.getAxis(axis).setTickFont(font)

    @QtCore.Slot()
    def level_change(self) -> None:
        if not self.isVisible():
            return
        for img_ref in self.images:
            img_ref().setLevels(self.spanRegion())
        self.limit_changed()

    @QtCore.Slot()
    def level_change_fin(self) -> None:
        pass

    def spanRegion(self) -> tuple[float, float]:
        return self._span.getRegion()

    def setSpanRegion(self, levels: tuple[float, float]) -> None:
        # Trigger region change start and finish signals to simulate drag
        self._span.sigRegionChangeStarted.emit(self._span)
        self._span.setRegion(levels)
        self._span.sigRegionChangeFinished.emit(self._span)

    def setLimits(self, limits: tuple[float, float] | None) -> None:
        self._fixedlimits = limits
        if self._primary_image is not None:
            self.limit_changed()

    def addImage(self, image: Iterable[BetterImageItem] | BetterImageItem) -> None:
        if not isinstance(image, Iterable):
            self._images.add(weakref.ref(image))
        else:
            for img in image:
                self._images.add(weakref.ref(img))

    def removeImage(self, image: Iterable[BetterImageItem] | BetterImageItem) -> None:
        if isinstance(image, Iterable):
            for img in image:
                self._images.remove(weakref.ref(img))
        else:
            self._images.remove(weakref.ref(image))

    def setImageItem(
        self,
        image: Iterable[BetterImageItem] | BetterImageItem,
        insert_in: pg.PlotItem | None = None,
    ) -> None:
        self.addImage(image)
        for img_ref in self._images:
            img = img_ref()
            if img is not None:
                if hasattr(img, "sigLevelsChanged"):
                    img.sigLevelsChanged.connect(self.image_level_changed)
                if hasattr(img, "sigImageChanged"):
                    img.sigImageChanged.connect(self.image_changed)

        for img_ref in self._images:
            img = img_ref()
            if img is not None and img.getColorMap() is not None:
                self._primary_image = img_ref
                break

        if self._primary_image is None:
            raise ValueError("ImageItem with a colormap was not found")
        # self.primary_image().sigLimitChanged.connect(self.limit_changed)

        # self.primary_image().sigImageChanged.connect(self.limit_changed)
        # self.primary_image().sigColorChanged.connect(self.color_changed)
        self.primary_image().sigColorChanged.connect(self.limit_changed)
        # else:
        # print("hello")

        if insert_in is not None:
            insert_in.layout.addItem(self, 2, 5)
            insert_in.layout.setColumnFixedWidth(4, 5)

        self._span.blockSignals(True)
        if hasattr(self, "limits"):
            self._span.setRegion(self.limits)
        self._span.blockSignals(False)

        self._span.sigRegionChanged.connect(self.level_change)
        self._span.sigRegionChangeFinished.connect(self.level_change_fin)
        self.image_changed()
        # self.color_changed()
        self.limit_changed()

    def image_level_changed(self) -> None:
        levels = self.primary_image().getLevels()
        if levels is not None:
            self._span.setRegion(levels)

    def image_changed(self) -> None:
        self.level_change()
        if self._auto_levels:
            self.reset_levels()

    def reset_levels(self) -> None:
        self._span.setRegion(self.limits)

    def setAutoLevels(self, value) -> None:
        self._auto_levels = bool(value)
        self._span.setVisible(not self._auto_levels)

        # self.isocurve.setParentItem(image)

    # def hideEvent(self, event: QtGui.QHideEvent):
    #     super().hideEvent(event)
    #     print("hide")

    # def showEvent(self, event: QtGui.QShowEvent):
    #     super().showEvent(event)
    #     # self._level_change()
    #     print("show")
    #     self.color_changed()
    #     self.limit_changed()

    # def setVisible(self, visible:bool, *args, **kwargs):
    # super().setVisible(visible, *args, **kwargs)
    # if visible:
    # self._level_change()
    # print('e')
    # self.isocurve.setVisible(visible, *args, **kwargs)

    def color_changed(self) -> None:
        if not self.isVisible():
            return
        cmap = self.primary_image()._colorMap
        if not cmap:
            return
        lut = cmap.getStops()[1]
        if self._colorbar.image.shape[0] != lut.shape[0]:
            self._colorbar.setImage(cmap.pos.reshape((-1, 1)))
        self._colorbar._colorMap = cmap
        self._colorbar.setLookupTable(lut, update=True)

    # def limit_changed(self, mn: float | None = None, mx: float | None = None):
    def limit_changed(self) -> None:
        if not self.isVisible():
            return
        if not hasattr(self, "limits"):
            return
        self.color_changed()
        # if (self._fixedlimits is not None) or (mn is None):
        mn, mx = self.limits
        self._colorbar.setRect(0.0, mn, 1.0, mx - mn)
        if self.levels is not None:
            self._colorbar.setLevels(
                (np.asarray(self.levels) - mn) / max(mx - mn, 1e-15)
            )
        self._span.setBounds((mn, mx))

    # def cmap_changed(self):
    #     cmap = self.imageItem()._colorMap
    #     # lut = self.imageItem().lut
    #     # lut = cmap.getLookupTable(nPts=4096)
    #     # lut = self._eff_lut_for_image(self.imageItem())
    #     # if lut is None:
    #         # lut = self.imageItem()._effectiveLut
    #     # if lut is not None:
    #         # print(lut)
    #     # if lut is None:
    #     lut = cmap.getStops()[1]
    #     # if not self.npts == lut.shape[0]:
    #     # self.npts = lut.shape[0]
    #     if not self._colorbar.image.shape[0] == lut.shape[0]:
    #         self._colorbar.setImage(cmap.pos.reshape((-1, 1)))
    #     self._colorbar._colorMap = cmap
    #     self._colorbar.setLookupTable(lut)
    #     # self._colorbar.setColorMap(cmap)

    def mouseDragEvent(self, ev) -> None:
        ev.ignore()


def is_dark_mode() -> bool:
    """Check if the system is in dark mode.

    If a QApplication is not running, this will always return False.
    """
    hints = QtGui.QGuiApplication.styleHints()
    if (
        hints is not None
        and hasattr(hints, "colorScheme")
        and hasattr(QtCore.Qt, "ColorScheme")
    ):
        return hints.colorScheme() == QtCore.Qt.ColorScheme.Dark

    # Fallback for Qt < 6.5
    default_palette = QtGui.QPalette()
    text = default_palette.color(QtGui.QPalette.ColorRole.WindowText)
    window = default_palette.color(QtGui.QPalette.ColorRole.Window)
    return text.lightness() > window.lightness()


def color_to_QColor(c: ColorType, alpha: float | None = None) -> QtGui.QColor:
    """Convert a matplotlib color to a :class:`PySide6.QtGui.QColor`.

    Parameters
    ----------
    c
        A valid matplotlib color. See the `matplotlib documentation
        <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ for more
        information.
    alpha
        If supplied, applies transparency to the color.

    Returns
    -------
    PySide6.QtGui.QColor

    """
    import matplotlib.colors

    return QtGui.QColor.fromRgbF(*matplotlib.colors.to_rgba(c, alpha=alpha))


def pg_colormap_names(
    source: typing.Literal["local", "all", "matplotlib"] = "all",
    exclude_local: bool = False,
) -> list[str]:
    """
    Get all valid pyqtgraph colormap names.

    Parameters
    ----------
    source
        - "local": only built-in/pyqtgraph colormaps
        - "matplotlib": only matplotlib colormaps
        - "all": local + colorcet + matplotlib
    exclude_local
        If True, drop local colormaps from the final list.

    Returns
    -------
    list of str
        Ordered, de-duplicated list of colormap names.
    """
    global_exclude: set[str] = set(erlab.interactive.options.model.colors.cmap.exclude)

    all_cmaps: list[str] = []

    if not exclude_local:
        all_cmaps += sorted(pg.colormap.listMaps())

    if source != "local":
        all_cmaps += sorted(
            cm
            for cm in pg.colormap.listMaps(source="matplotlib")
            if not cm.startswith("cet_") and not cm.endswith(("_r", "_r_i"))
        )

    if source == "all":
        all_cmaps += sorted(
            cm
            for cm in pg.colormap.listMaps(source="colorcet")
            if not cm.startswith("glasbey")
        )

    combined: list[str] = []
    seen: set[str] = set()
    for cm in all_cmaps:
        if cm in global_exclude:
            continue
        if cm not in seen:
            seen.add(cm)
            combined.append(cm)

    return combined


def pg_colormap_from_name(
    name: str, skipCache: bool = True, _loaded_all: bool = False
) -> pg.ColorMap:
    """Get a :class:`pyqtgraph.ColorMap` from its name.

    Parameters
    ----------
    name
        A valid colormap name.
    skipCache
        Whether to skip cache, by default `True`. Passed onto
        :func:`pyqtgraph.colormap.get`.

    Returns
    -------
    pyqtgraph.ColorMap

    """
    cmap: pg.ColorMap | None = None
    with contextlib.suppress(ValueError):
        cmap = pg.colormap.get(name, source="matplotlib", skipCache=skipCache)

    if not cmap:
        with contextlib.suppress(FileNotFoundError, IndexError, KeyError):
            cmap = pg.colormap.get(name, skipCache=skipCache)
    if not cmap:
        with contextlib.suppress(KeyError):
            cmap = pg.colormap.get(name, source="colorcet", skipCache=skipCache)
    if not cmap and not _loaded_all:
        with contextlib.suppress(ValueError, FileNotFoundError, IndexError, KeyError):
            load_all_colormaps()
            cmap = pg_colormap_from_name(name, skipCache=skipCache, _loaded_all=True)
    if not cmap:
        raise RuntimeError(f"Colormap '{name}' not found in any source.")
    return cmap


def pg_colormap_powernorm(
    cmap: str | pg.ColorMap,
    gamma: float,
    reverse: bool = False,
    high_contrast: bool = False,
    zero_centered: bool = False,
    N: int = 65536,
) -> pg.ColorMap:
    if isinstance(cmap, str):
        cmap = pg_colormap_from_name(cmap, skipCache=True)

    gamma = float(gamma)

    if gamma == 1.0:

        def mapping_fn(x):
            return x

    elif high_contrast:

        def mapping_fn(x):
            return 1 - np.power(np.flip(x), 1.0 / gamma)

    else:

        def mapping_fn(x):
            return np.power(x, gamma)

    x = np.linspace(0, 1, N)
    if zero_centered:
        mapping = np.piecewise(
            x,
            [x < 0.5, x >= 0.5],
            [
                lambda x: 0.5 - 0.5 * mapping_fn(-2 * x + 1),
                lambda x: 0.5 + 0.5 * mapping_fn(2 * x - 1),
            ],
        )
    else:
        mapping = mapping_fn(x)
    mapping[mapping > 1] = 1
    mapping[mapping < 0] = 0

    if reverse:
        cmap.reverse()
    cmap.color = cmap.mapToFloat(mapping)
    cmap.pos = np.linspace(0, 1, N)
    pg.colormap._mapCache = {}  # disable cache to reduce memory usage

    cmap._erlab_attrs = {
        "gamma": gamma,
        "reverse": reverse,
        "high_contrast": high_contrast,
        "zero_centered": zero_centered,
    }
    return cmap


def pg_colormap_to_QPixmap(
    cmap: str | pg.ColorMap, w: int = 64, h: int = 16, skipCache: bool = True
) -> QtGui.QPixmap:
    """Convert a :class:`pyqtgraph.ColorMap` to a ``w``-by-``h`` QPixmap thumbnail.

    Parameters
    ----------
    cmap
        The colormap.
    w, h
        Specifies the dimension of the pixmap.
    skipCache : bool, optional
        Whether to skip cache, by default `True`. Passed onto
        :func:`pg_colormap_from_name`.

    Returns
    -------
    PySide6.QtGui.QPixmap

    """
    if isinstance(cmap, str):
        cmap = pg_colormap_from_name(cmap, skipCache=skipCache)
    # cmap_arr = np.reshape(cmap.getColors()[:, None], (1, -1, 4), order='C')
    # cmap_arr = np.reshape(
    # cmap.getLookupTable(0, 1, w, alpha=True)[:, None], (1, -1, 4),
    # order='C')
    cmap_arr = cmap.getLookupTable(0, 1, w, alpha=True)[:, None]

    # print(cmap_arr.shape)
    img = QtGui.QImage(cmap_arr, w, 1, QtGui.QImage.Format.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img).scaled(w, h)


class ColorCycleDialog(QtWidgets.QDialog):
    """Dialog for selecting a color cycle.

    This dialog takes a list of colors and allows the user to edit each color in the
    cycle, or overwrite the entire cycle by sampling from a colormap.

    Parameters
    ----------
    colors
        An iterable of colors to use as the initial color cycle.
    parent
        The parent widget for the dialog.
    preview_cursors
        If `True`, the preview will inclue cursor lines and spans with the selected
        colors.
    opacity_values
        A tuple of four float values representing the opacity for the cursor line,
        cursor line hover, span background, and span edge, respectively. Each value
        should be between 0 and 1, where 1 is fully opaque and 0 is fully transparent.
    default_colors
        An iterable of default colors to return to when the user clicks "Restore
        Defaults". If `None`, the restore defaults button will not be shown.

    """

    sigAccepted = QtCore.Signal(tuple)  #: :meta private:

    def __init__(
        self,
        colors: Iterable[QtGui.QColor],
        *,
        parent: QtWidgets.QWidget | None = None,
        preview_cursors: bool = False,
        opacity_values: tuple[float, float, float, float] = (0.95, 0.75, 0.15, 0.35),
        default_colors: Iterable[QtGui.QColor] | None = None,
    ) -> None:
        super().__init__(parent)
        self.original_colors: Iterable[QtGui.QColor] = colors
        self._colors: tuple[QtGui.QColor] = tuple(pg.mkColor(c) for c in colors)

        self.preview_cursors = preview_cursors
        self.opacity_values = opacity_values
        self.default_colors = default_colors
        self.setup_ui()

    @property
    def colors(self) -> tuple[QtGui.QColor]:
        """Return the current color cycle."""
        return self._colors

    @colors.setter
    def colors(self, colors: Iterable[QtGui.QColor]) -> None:
        """Set the color cycle to a new list of colors."""
        self._colors = tuple(pg.mkColor(c) for c in colors)
        for i, clr in enumerate(self._colors):
            self.color_btns[i].blockSignals(True)
            self.color_btns[i].setColor(clr)
            self.color_btns[i].blockSignals(False)
            self.curves[i].setPen(pg.mkPen(clr))

            if self.preview_cursors:
                clr_cursor = pg.mkColor(clr)
                clr_cursor_hover = pg.mkColor(clr)
                clr_span = pg.mkColor(clr)
                clr_span_edge = pg.mkColor(clr)

                clr_cursor.setAlphaF(self.opacity_values[0])
                clr_cursor_hover.setAlphaF(self.opacity_values[1])
                clr_span.setAlphaF(self.opacity_values[2])
                clr_span_edge.setAlphaF(self.opacity_values[3])

                self.lines[i].setPen(pg.mkPen(clr_cursor))
                self.lines[i].setHoverPen(pg.mkPen(clr_cursor_hover))

                for span_line in self.spans[i].lines:
                    span_line.setPen(pg.mkPen(clr_span_edge))
                self.spans[i].setBrush(pg.mkBrush(clr_span))

    def setup_ui(self):
        self.layout_ = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout_)

        # Colormap selection
        cmap_group = QtWidgets.QGroupBox("Choose from colormap", self)
        self.layout_.addWidget(cmap_group)
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

        self.apply_btn = QtWidgets.QPushButton("Apply", self)
        self.apply_btn.setToolTip("Sample colors from the colormap")
        self.apply_btn.clicked.connect(self.set_from_cmap)
        cmap_layout.addWidget(self.apply_btn)

        # Bottom layout
        bottom_layout = QtWidgets.QHBoxLayout()
        self.layout_.addLayout(bottom_layout)

        # Color editor
        editor_group = QtWidgets.QGroupBox("Edit Colors", self)
        bottom_layout.addWidget(editor_group)

        editor_layout = QtWidgets.QFormLayout(editor_group)
        editor_group.setLayout(editor_layout)

        self.color_btns = []
        for i, color in enumerate(self.colors):
            btn = pg.ColorButton(self)
            btn.colorDialog.setOption(
                QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog, False
            )
            btn.setColor(color)
            btn.setToolTip(f"Cursor {i}")
            btn.sigColorChanged.connect(self._update_color)
            self.color_btns.append(btn)
            editor_layout.addRow(f"Cursor {i}", btn)

        # Dialog buttons
        self.button_box = QtWidgets.QDialogButtonBox()
        btn_ok = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Ok,
        )
        btn_cancel = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        )
        btn_reset = self.button_box.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Reset,
        )
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_reset.clicked.connect(self.reset)
        btn_reset.setToolTip("Reset to current colors")

        if self.default_colors is not None:
            btn_default = self.button_box.addButton(
                QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
            )
            btn_default.clicked.connect(self.restore_defaults)
            btn_default.setToolTip("Restore from current user defaults")

        # Preview
        preview_group = QtWidgets.QGroupBox("Preview", self)
        bottom_layout.addWidget(preview_group)

        preview_group_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_group.setLayout(preview_group_layout)

        self.plot_widget = pg.PlotWidget(preview_group)
        preview_group_layout.addWidget(self.plot_widget)

        self.curves = []
        self.lines = []
        self.spans = []
        for i in range(len(self.colors)):
            curve = pg.PlotDataItem(
                x=np.linspace(0, 1, 100),
                y=np.sin(np.linspace(0, 2 * np.pi, 100) + i * np.pi / 12),
                name=f"Cursor {i}",
            )
            self.curves.append(curve)
            self.plot_widget.addItem(curve)

            if self.preview_cursors:
                cursor_center = i * np.pi / 64 + np.pi / 64
                span_halfwidth = np.pi / 128 - 0.01
                line = pg.InfiniteLine(cursor_center, angle=90, movable=True)
                self.lines.append(line)
                self.plot_widget.addItem(line)

                span = pg.LinearRegionItem(
                    (cursor_center - span_halfwidth, cursor_center + span_halfwidth),
                    movable=False,
                )

                # Make the span move with the line
                def update_span(line=line, span=span, hw=span_halfwidth):
                    center = line.value()
                    span.setRegion((center - hw, center + hw))

                line.sigPositionChanged.connect(update_span)
                self.spans.append(span)
                self.plot_widget.addItem(span)

        self.layout_.addWidget(self.button_box)

        self.reset()  # Initialize with original colors

    @QtCore.Slot()
    def _update_color(self) -> None:
        """Update internal color state when a color button is changed."""
        sender = typing.cast("pg.ColorButton", self.sender())
        if sender in self.color_btns:
            sender.blockSignals(True)
            index = self.color_btns.index(sender)
            color_list = list(self.colors)
            color_list[index] = sender.color()
            self.colors = color_list
            sender.blockSignals(False)

    @QtCore.Slot()
    def set_from_cmap(self) -> None:
        """Set the color cycle from a colormap."""
        pg_cmap = pg_colormap_powernorm(
            self.cmap_combo.currentText(),
            gamma=1.0,
            reverse=self.reverse_check.isChecked(),
        )
        self.colors = pg_cmap.getLookupTable(
            start=self.start_spin.value(),
            stop=self.stop_spin.value(),
            nPts=len(self.colors),
            mode=pg_cmap.QCOLOR,
        )

    @QtCore.Slot()
    def reset(self) -> None:
        self.colors = self.original_colors

    @QtCore.Slot()
    def restore_defaults(self) -> None:
        """Restore the default colors if provided."""
        if self.default_colors is not None:  # pragma: no branch
            self.colors = self.default_colors

    def accept(self) -> None:
        """Accept the dialog and emit the new colors."""
        self.sigAccepted.emit(self.colors)
        super().accept()
