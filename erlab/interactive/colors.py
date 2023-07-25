"""Functions for manipulating colors in Qt.

"""
from __future__ import annotations

__all__ = [
    "ColorMapComboBox",
    "ColorMapGammaWidget",
    "color_to_QColor",
    "pg_colormap_names",
    "pg_colormap_from_name",
    "pg_colormap_powernorm",
    "pg_colormap_to_QPixmap",
]

from typing import Literal

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets


class ColorMapComboBox(QtWidgets.QComboBox):
    LOAD_ALL_TEXT = "Load all..."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPlaceholderText("Select colormap...")
        self.setToolTip("Colormap")
        w, h = 64, 16
        self.setIconSize(QtCore.QSize(w, h))
        # for name in pg_colormap_names("local"):
        for name in pg_colormap_names("mpl"):
            self.addItem(name)
        self.insertItem(0, self.LOAD_ALL_TEXT)
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
        maxWidth = max(
            [fm.boundingRect(self.itemText(i)).width() for i in range(self.count())]
        )
        if maxWidth:
            view.setMinimumWidth(maxWidth)

    @QtCore.Slot()
    def nextIndex(self):
        self.wheelEvent(
            QtGui.QWheelEvent(
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, 0),
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
                QtCore.QPoint(0, 0),
                QtCore.QPoint(0, 0),
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
    valueChanged = QtCore.Signal(float)  #: :meta private:

    def __init__(
        self,
        parent: QtWidgets.QWidget = None,
        value: float = 1.0,
        slider_cls: type = None,
        spin_cls: type = None,
    ):
        super().__init__(parent=parent)
        self.setLayout(QtWidgets.QHBoxLayout(self))
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().setSpacing(3)

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
        self.label = QtWidgets.QLabel("γ", self)
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

        self.layout().addWidget(self.label)
        self.layout().addWidget(self.spin)
        self.layout().addWidget(self.slider)

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

    def slider_changed(self, value: float):
        self.spin.setValue(self.gamma_scale_inv(value))

    def gamma_scale(self, y: float) -> int:
        return round(1e4 * np.log10(y))

    def gamma_scale_inv(self, x: int) -> float:
        return np.power(10, x * 1e-4)


def color_to_QColor(
    c: str | tuple[float, ...], alpha: float | None = None
) -> QtGui.QColor:
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
    return QtGui.QColor.fromRgbF(*mcolors.to_rgba(c, alpha=alpha))


def pg_colormap_names(
    source: Literal["local", "all", "matplotlib"] = "all"
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
            if cmap.startswith("cet_"):
                _mpl = list(filter((cmap).__ne__, _mpl))
            elif cmap.endswith("_r"):
                # _mpl_r.append(cmap)
                _mpl = list(filter((cmap).__ne__, _mpl))
        if source == "all":
            cet = sorted(pg.colormap.listMaps(source="colorcet"))
            # if (_mpl != []) and (cet != []):
            # local = []
            # _mpl_r = []
            all_cmaps = local + cet + _mpl  # + _mpl_r
        else:
            all_cmaps = local + _mpl
    return list({value: None for value in all_cmaps})


def pg_colormap_from_name(name: str, skipCache: bool = True) -> pg.ColorMap:
    """Gets a :class:`pyqtgraph.ColorMap` from its name.

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
    highContrast: bool = False,
    zeroCentered: bool = False,
    N: int = 65536,
) -> pg.ColorMap:
    if isinstance(cmap, str):
        cmap = pg_colormap_from_name(cmap, skipCache=True)

    if gamma == 1:

        def mapping_fn(x):
            return x

    elif highContrast:

        def mapping_fn(x):
            return 1 - np.power(np.flip(x), 1.0 / gamma)

    else:

        def mapping_fn(x):
            return np.power(x, gamma)

    x = np.linspace(0, 1, N)
    if zeroCentered:
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
    """Converts a :class:`pyqtgraph.ColorMap` to a ``w``-by-``h`` QPixmap thumbnail.

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
    img = QtGui.QImage(cmap_arr, w, 1, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img).scaled(w, h)
