"""Functions for manipulating colors in Qt."""

from __future__ import annotations

__all__ = [
    "BetterColorBarItem",
    "BetterImageItem",
    "ColorMapComboBox",
    "ColorMapGammaWidget",
    "color_to_QColor",
    "pg_colormap_from_name",
    "pg_colormap_names",
    "pg_colormap_powernorm",
    "pg_colormap_to_QPixmap",
]

import weakref
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib.colors
import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    from matplotlib.typing import ColorType

EXCLUDED_CMAPS: tuple[str, ...] = (
    "prism",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "flag",
    "Set1",
    "Set2",
    "Set3",
    "Pastel1",
    "Pastel2",
    "Pastel3",
    "Paired",
    "Dark2",
)
"""
Colormaps to exclude from the list of available colormaps. They are not suitable for
continuous data, and looks horrible.
"""


class ColorMapComboBox(QtWidgets.QComboBox):
    LOAD_ALL_TEXT = "Load all..."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPlaceholderText("Select colormap...")
        self.setToolTip("Colormap")
        w, h = 64, 16
        self.setIconSize(QtCore.QSize(w, h))
        # for name in pg_colormap_names("local"):
        for name in pg_colormap_names("all", exclude_local=True):
            self.addItem(name)
        # self.insertItem(0, self.LOAD_ALL_TEXT)
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
        for name in pg_colormap_names("all", exclude_local=True):
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
            fm.boundingRect(self.itemText(i)).width() for i in range(self.count())
        )
        if maxWidth and view is not None:
            view.setMinimumWidth(maxWidth)

    @QtCore.Slot()
    def nextIndex(self):
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
    def previousIndex(self):
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

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        value: float = 1.0,
        slider_cls: type | None = None,
        spin_cls: type | None = None,
    ):
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
            self.gamma_scale(self.spin.minimum()),
            self.gamma_scale(self.spin.maximum()),
        )
        self.slider.valueChanged.connect(self.slider_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.spin)
        layout.addWidget(self.slider)

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

    def slider_changed(self, value: float | int):
        self.spin.setValue(self.gamma_scale_inv(value))

    def gamma_scale(self, y: float) -> int:
        return round(1e4 * np.log10(y))

    def gamma_scale_inv(self, x: float | int) -> float:
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

    def __init__(self, image: npt.NDArray | None = None, **kwargs):
        super().__init__(image, **kwargs)

    def set_colormap(
        self,
        cmap: pg.ColorMap | str,
        gamma: float,
        reverse: bool = False,
        high_contrast: bool = False,
        zero_centered: bool = False,
        update: bool = True,
    ):
        cmap = pg_colormap_powernorm(
            cmap,
            gamma,
            reverse,
            high_contrast=high_contrast,
            zero_centered=zero_centered,
        )
        self.set_pg_colormap(cmap, update=update)

    def set_pg_colormap(self, cmap: pg.ColorMap, update: bool = True):
        self._colorMap = cmap
        self.setLookupTable(cmap.getStops()[1], update=update)
        self.sigColorChanged.emit()


class TrackableLinearRegionItem(pg.LinearRegionItem):
    sigRegionChangeStarted = QtCore.Signal(object)  #: :meta private:

    def mouseDragEvent(self, ev):
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
        **kargs,
    ):
        super().__init__(parent, **kargs)

        self.setDefaultPadding(0)
        # self.hideButtons()
        self.setMenuEnabled(False)
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
        else:
            return self.primary_image().quickMinMax(targetSize=2**16)

    def set_width(self, width: int):
        self.layout.setColumnFixedWidth(1, width)

    def set_dimensions(
        self,
        horiz_pad: int | None = None,
        vert_pad: int | None = None,
        font_size: float = 11.0,
    ):
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
    def level_change(self):
        if not self.isVisible():
            return
        for img_ref in self.images:
            img_ref().setLevels(self.spanRegion())
        self.limit_changed()

    @QtCore.Slot()
    def level_change_fin(self):
        pass

    def spanRegion(self) -> tuple[float, float]:
        return self._span.getRegion()

    def setSpanRegion(self, levels: tuple[float, float]):
        self._span.setRegion(levels)

    def setLimits(self, limits: tuple[float, float] | None):
        self._fixedlimits = limits
        if self._primary_image is not None:
            self.limit_changed()

    def addImage(self, image: Iterable[BetterImageItem] | BetterImageItem):
        if not isinstance(image, Iterable):
            self._images.add(weakref.ref(image))
        else:
            for img in image:
                self._images.add(weakref.ref(img))

    def removeImage(self, image: Iterable[BetterImageItem] | BetterImageItem):
        if isinstance(image, Iterable):
            for img in image:
                self._images.remove(weakref.ref(img))
        else:
            self._images.remove(weakref.ref(image))

    def setImageItem(
        self,
        image: Iterable[BetterImageItem] | BetterImageItem,
        insert_in: pg.PlotItem | None = None,
    ):
        self.addImage(image)
        for img_ref in self._images:
            img = img_ref()
            if img is not None:
                if hasattr(img, "sigLevelsChanged"):
                    img.sigLevelsChanged.connect(self.image_level_changed)
                if hasattr(img, "sigImageChanged"):
                    img.sigImageChanged.connect(self.image_changed)
                if img.getColorMap() is not None:
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

    def image_level_changed(self):
        levels = self.primary_image().getLevels()
        if levels is not None:
            self._span.setRegion(levels)

    def image_changed(self):
        self.level_change()
        if self._auto_levels:
            self.reset_levels()

    def reset_levels(self):
        self._span.setRegion(self.limits)

    def setAutoLevels(self, value):
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

    def color_changed(self):
        if not self.isVisible():
            return
        cmap = self.primary_image()._colorMap
        lut = cmap.getStops()[1]
        if self._colorbar.image.shape[0] != lut.shape[0]:
            self._colorbar.setImage(cmap.pos.reshape((-1, 1)))
        self._colorbar._colorMap = cmap
        self._colorbar.setLookupTable(lut, update=True)

    # def limit_changed(self, mn: float | None = None, mx: float | None = None):
    def limit_changed(self):
        if not self.isVisible():
            return
        if not hasattr(self, "limits"):
            return
        self.color_changed()
        # if (self._fixedlimits is not None) or (mn is None):
        mn, mx = self.limits
        self._colorbar.setRect(0.0, mn, 1.0, mx - mn)
        if self.levels is not None:
            self._colorbar.setLevels((np.asarray(self.levels) - mn) / (mx - mn))
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

    def mouseDragEvent(self, ev):
        ev.ignore()


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
    return QtGui.QColor.fromRgbF(*matplotlib.colors.to_rgba(c, alpha=alpha))


def pg_colormap_names(
    source: Literal["local", "all", "matplotlib"] = "all", exclude_local: bool = False
) -> list[str]:
    """Get all valid :obj:`pyqtgraph` colormap names.

    Parameters
    ----------
    source
        If ``'all'``, includes all colorcet colormaps.

    Returns
    -------
    list of str

    """
    local = sorted(pg.colormap.listMaps())
    if source == "local":
        return local
    else:
        _mpl = sorted(pg.colormap.listMaps(source="matplotlib"))
        for cmap in _mpl:
            if (
                cmap.startswith("cet_")
                or cmap.endswith(("_r", "_r_i"))
                or cmap in EXCLUDED_CMAPS
            ):
                _mpl = list(filter((cmap).__ne__, _mpl))
        if source == "all":
            cet = sorted(pg.colormap.listMaps(source="colorcet"))
            for cmap in cet:
                if cmap.startswith("glasbey"):
                    cet = list(filter((cmap).__ne__, cet))

            # if (_mpl != []) and (cet != []):
            # local = []

            if exclude_local:
                all_cmaps = cet + _mpl
            else:
                all_cmaps = local + cet + _mpl
        elif exclude_local:
            all_cmaps = _mpl
        else:
            all_cmaps = local + _mpl
    return list(dict.fromkeys(all_cmaps))


def pg_colormap_from_name(name: str, skipCache: bool = True) -> pg.ColorMap:
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
    try:
        return pg.colormap.get(name, skipCache=skipCache)
    except (FileNotFoundError, IndexError):
        try:
            return pg.colormap.get(name, source="matplotlib", skipCache=skipCache)
        except ValueError:
            return pg.colormap.get(name, source="colorcet", skipCache=skipCache)


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

    if gamma == 1:

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
