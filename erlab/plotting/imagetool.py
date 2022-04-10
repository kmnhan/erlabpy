from audioop import avg
import sys
import unittest.mock
import weakref
from itertools import chain, compress
from time import perf_counter

import bottleneck as bn
import darkdetect
import numba
import numbagg
import numpy as np
import pyqtgraph as pg
import xarray as xr
from matplotlib import colors

# pg.setConfigOption("imageAxisOrder", "row-major")
# pg.setConfigOption('useNumba', True)
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PySide6 import QtSvg, QtSvgWidgets, QtWebEngineWidgets

if __name__ != "__main__":
    from .colors import pg_colormap_names, pg_colormap_powernorm, pg_colormap_to_QPixmap
else:
    from erlab.plotting.colors import (
        pg_colormap_names,
        pg_colormap_powernorm,
        pg_colormap_to_QPixmap,
    )
__all__ = ["itool"]

suppressnanwarning = np.testing.suppress_warnings()
suppressnanwarning.filter(RuntimeWarning, r"All-NaN (slice|axis) encountered")

import qtawesome as qta

fonticons = dict(
    invert="mdi6.invert-colors",
    invert_off="mdi6.invert-colors-off",
    contrast="mdi6.contrast-box",
    lock="mdi6.lock",
    unlock="mdi6.lock-open-variant",
    colorbar="mdi6.gradient-vertical",
    transpose=[
        "mdi6.arrow-left-right",
        "mdi6.arrow-top-left-bottom-right",
        "mdi6.arrow-up-down",
    ],
    snap="mdi6.grid",
    snap_off="mdi6.grid-off",
    palette="mdi6.palette-advanced",
    styles="mdi6.palette-swatch",
    layout="mdi6.page-layout-body",
)


# import urllib.request
# req = urllib.request.Request('')
# with urllib.request.urlopen(req) as resp:
#     mathjax = resp.read()


import matplotlib.mathtext
from matplotlib import figure, rc_context, rcParams
from matplotlib.backends import backend_agg, backend_svg

# rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
#     "pgf.texsystem": "lualatex",
# })


class FlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)

        # if parent is not None:
        # self.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))

        self._item_list = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]

        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)

        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._do_layout(QtCore.QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()

        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())

        size += QtCore.QSize(
            2 * self.contentsMargins().top(), 2 * self.contentsMargins().top()
        )
        return size

    def _do_layout(self, rect, test_only):

        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._item_list:
            style = item.widget().style()
            layout_spacing_x = style.layoutSpacing(
                QtWidgets.QSizePolicy.PushButton,
                QtWidgets.QSizePolicy.PushButton,
                QtCore.Qt.Horizontal,
            )
            layout_spacing_y = style.layoutSpacing(
                QtWidgets.QSizePolicy.PushButton,
                QtWidgets.QSizePolicy.PushButton,
                QtCore.Qt.Vertical,
            )
            space_x = spacing + layout_spacing_x
            space_y = spacing + layout_spacing_y
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()


# 0: html, 1: svg, 2: pixmap
label_mode = 2


def get_pixmap_label(s, prop=None, dpi=300, **text_kw):
    """
    The get_pixmap_label function creates a pixmap of the mathtext label.

    The function accepts a string containing LaTeX markup, and returns an instance of QtGui.QPixmap that contains the rendered image. The image is rendered using matplotlib's MathText parser, so all mathematical symbols can be used in the input string.

    :param s: Used to Pass the string to be rendered.
    :param prop=None: Used to Specify the font properties of the text.
    :param dpi=300: Used to Set the resolution of the image.
    :param **text_kw: Used to Pass keyword arguments to the text function of the figure class.
    :return: The pixmap of the label.

    :doc-author: Trelent
    """

    with rc_context({"text.usetex": False}):
        parser = matplotlib.mathtext.MathTextParser("path")
        width, height, depth, _, _ = parser.parse(s, dpi=72, prop=prop)

        fig = figure.Figure(figsize=(width / 72.0, height / 72.0), dpi=dpi)
        fig.patch.set_facecolor("none")
        text_kw["fontproperties"] = prop
        text_kw["fontsize"] = 9
        fig.text(0, depth / height, s, **text_kw)

    backend_agg.FigureCanvasAgg(fig)
    buf, size = fig.canvas.print_to_buffer()
    img = QtGui.QImage(buf, size[0], size[1], QtGui.QImage.Format_ARGB32)
    img.setDevicePixelRatio(fig._dpi / 100.0)
    pixmap = QtGui.QPixmap(img.rgbSwapped())
    return pixmap


def get_svg_label(s, prop=None, dpi=300, **text_kw):
    with rc_context({"text.usetex": True}):
        parser = matplotlib.mathtext.MathTextParser("path")
        width, height, depth, _, _ = parser.parse(s, dpi=1000, prop=prop)

        fig = figure.Figure(figsize=(width / 1000.0, height / 1000.0), dpi=dpi)
        fig.patch.set_facecolor("none")
        text_kw["fontproperties"] = prop
        text_kw["fontsize"] = 12
        fig.text(0, depth / height, s, **text_kw)

    backend_svg.FigureCanvasSVG(fig)
    file = QtCore.QTemporaryFile()
    if file.open():
        fig.canvas.print_svg(file.fileName())
    return file.fileName()


def mathtextLabelPixmap(self):
    if self.labelUnits == "":
        if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
            units = ""
        else:
            units = "(x%g)" % (1.0 / self.autoSIPrefixScale)
    else:
        units = "(%s%s)" % (self.labelUnitPrefix, self.labelUnits)

    s = "%s %s" % (self.labelText, units)

    if label_mode == 1:
        return get_svg_label(s, **self.labelStyle)
    elif label_mode == 0:
        style = ";".join(["%s: %s" % (k, self.labelStyle[k]) for k in self.labelStyle])
        src = """
             <html><head>
             <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_HTMLorMML">                     
             </script></head>
             <body>
             <p><mathjax style="font-size:0.1em">%s</mathjax></p>
             </body></html>
             """
        #  <p><mathjax style="%s">%s</mathjax></p>
        # return src % (style, s)
        return src % s
    else:
        return get_pixmap_label(s, **self.labelStyle)


def _updateMathtextLabel(self):
    if label_mode == 1:
        try:
            self.svg_renderers[0]
        except AttributeError:
            self.svg_renderers = []
        self.svg_renderers.append(QtSvg.QSvgRenderer(self.mathtextLabelPixmap()))
        self.label.setSharedRenderer(self.svg_renderers[-1])
    elif label_mode == 0:
        # self.scene().removeItem(self.label)
        # del self._labelwidget
        # self._labelwidget = QtWebEngineWidgets.QWebEngineView()
        self._labelwidget.setHtml(self.mathtextLabelPixmap())
        self._labelwidget.show()
        # self.label = self.scene().addWidget(self._labelwidget)
        self._labelwidget.update()
        self._labelwidget.reload()
        self.label.setWidget(self._labelwidget)
        # print(self.label.isVisible())
    else:
        self.label.setPixmap(self.mathtextLabelPixmap())

    self._adjustSize()
    self.picture = None
    self.update()


def resizeEvent(self, ev=None):
    # s = self.size()

    ## Set the position of the label
    if label_mode == 1:
        nudge = -5
    elif label_mode == 0:
        nudge = 0
    else:
        nudge = -5
    if (
        self.label is None
    ):  # self.label is set to None on close, but resize events can still occur.
        self.picture = None
        return

    br = self.label.boundingRect()
    p = QtCore.QPointF(0, 0)
    if self.orientation == "left":
        p.setY(int(self.size().height() / 2 + br.width() / 2))
        p.setX(-nudge)
    elif self.orientation == "right":
        p.setY(int(self.size().height() / 2 + br.width() / 2))
        p.setX(int(self.size().width() - br.height() + nudge))
    elif self.orientation == "top":
        p.setY(-nudge)
        p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
    elif self.orientation == "bottom":
        p.setX(int(self.size().width() / 2.0 - br.width() / 2.0))
        p.setY(int(self.size().height() - br.height() + nudge))
    self.label.setPos(p)
    self.picture = None


def disableMathtextLabels(AxisItem):
    AxisItem.label = AxisItem.label_unpatched
    AxisItem._updateLabel = AxisItem._updateLabel_unpatched
    del AxisItem.label_unpatched
    del AxisItem._updateLabel_unpatched
    del AxisItem.mathtextLabelPixmap


def enableMathtextLabels(item: pg.AxisItem):
    item.label_unpatched = item.label
    item._updateLabel_unpatched = item._updateLabel
    if label_mode == 1:
        item.label = QtSvgWidgets.QGraphicsSvgItem(item)
    elif label_mode == 0:
        item._labelwidget = QtWebEngineWidgets.QWebEngineView()
        # item.label = item.scene().addWidget(item._labelwidget)
        item.label = QtWidgets.QGraphicsProxyWidget(item)
        item.label.setWidget(item._labelwidget)
    else:
        item.label = QtWidgets.QGraphicsPixmapItem(item)
        item.label.setTransformationMode(QtCore.Qt.SmoothTransformation)
    item.label.setRotation(item.label_unpatched.rotation())
    item.mathtextLabelPixmap = mathtextLabelPixmap.__get__(item)
    item._updateLabel = _updateMathtextLabel.__get__(item)
    item.resizeEvent = resizeEvent.__get__(item)


def setMathLabels(self, **kwds):
    if not self.useMathLabels:
        for k in kwds.keys():
            if k != "title":
                enableMathtextLabels(self.getAxis(k))
    self.useMathLabels = True
    self.setLabels(**kwds)


# pg.PlotItem.setMathLabels = setMathLabels#.__get__(pg.PlotItem)
# pg.PlotItem.useMathLabels = False
pg.PlotItem.setMathLabels = pg.PlotItem.setLabels


def qt_style_names():
    """Return a list of styles, default platform style first"""
    default_style_name = QtWidgets.QApplication.style().objectName().lower()
    result = []
    for style in QtWidgets.QStyleFactory.keys():
        if style.lower() == default_style_name:
            result.insert(0, style)
        else:
            result.append(style)
    return result


def change_style(style_name):
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create(style_name))


def move_mean_centered(a, window, min_count=None, axis=-1):
    w = (window - 1) // 2
    shift = w + 1
    if min_count is None:
        min_count = w + 1
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (0, shift)
    a = np.pad(a, pad_width, constant_values=np.nan)
    val = bn.move_mean(a, window, min_count=min_count, axis=axis)
    return val[(slice(None),) * (axis % a.ndim) + (slice(w, -1),)]


def move_mean_centered_multiaxis(a, window_list, min_count_list=None):
    w_list = [(window - 1) // 2 for window in window_list]
    pad_width = [(0, 0)] * a.ndim
    slicer = [
        slice(None),
    ] * a.ndim
    if min_count_list is None:
        min_count_list = [w + 1 for w in w_list]
    for axis in range(a.ndim):
        pad_width[axis] = (0, w_list[axis] + 1)
        slicer[axis] = slice(w_list[axis], -1)
    a = np.pad(a, pad_width, constant_values=np.nan)
    val = move_mean(a, numba.typed.List(window_list), numba.typed.List(min_count_list))
    return val[tuple(slicer)]


def move_mean(a, window, min_count):
    if a.ndim == 3:
        return move_mean3d(a, window, min_count)
    elif a.ndim == 2:
        return move_mean2d(a, window, min_count)
    else:
        raise NotImplementedError


# @numba.njit(nogil=True, cache=True, fastmath={'nnan','ninf', 'nsz', 'contract', 'reassoc', 'afn', 'arcp'})
@numba.njit(
    nogil=True,
    cache=True,
    fastmath={"ninf", "nsz", "contract", "reassoc", "afn", "arcp"},
)
# @numba.njit(nogil=True, cache=True)
def move_mean1d(a, window, min_count):
    out = np.empty_like(a)
    asum = 0.0
    count = 0

    for i in range(min_count - 1):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = np.nan

    for i in range(min_count - 1, window):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = asum / count if count >= min_count else np.nan

    count_inv = 1 / count if count >= min_count else np.nan
    for i in range(window, len(a)):
        ai = a[i]
        aold = a[i - window]

        ai_valid = not np.isnan(ai)
        aold_valid = not np.isnan(aold)

        if ai_valid and aold_valid:
            asum += ai - aold
        elif ai_valid:
            asum += ai
            count += 1
            count_inv = 1 / count if count >= min_count else np.nan
        elif aold_valid:
            asum -= aold
            count -= 1
            count_inv = 1 / count if count >= min_count else np.nan

        out[i] = asum * count_inv
    return out


@numba.njit(nogil=True, parallel=True, cache=True)
def move_mean2d(a, window_list, min_count_list):
    ii, jj = a.shape
    for i in numba.prange(ii):
        a[i, :] = move_mean1d(a[i, :], window_list[0], min_count_list[0])
    for j in numba.prange(jj):
        a[:, j] = move_mean1d(a[:, j], window_list[1], min_count_list[1])
    return a


@numba.njit(nogil=True, parallel=True, cache=True)
def move_mean3d(a, window_list, min_count_list):
    ii, jj, kk = a.shape
    for i in numba.prange(ii):
        for k in numba.prange(kk):
            a[i, :, k] = move_mean1d(a[i, :, k], window_list[1], min_count_list[1])
    for i in numba.prange(ii):
        for j in numba.prange(jj):
            a[i, j, :] = move_mean1d(a[i, j, :], window_list[2], min_count_list[2])
    for j in numba.prange(jj):
        for k in numba.prange(kk):
            a[:, j, k] = move_mean1d(a[:, j, k], window_list[0], min_count_list[0])
    return a


def parse_data(data):
    if isinstance(data, xr.Dataset):
        try:
            data = data.spectrum
        except:
            raise TypeError(
                "input argument data must be a xarray.DataArray or a "
                "numpy.ndarray. Create an xarray.DataArray "
                "first, either with indexing on the Dataset or by "
                "invoking the `to_array()` method."
            ) from None
    elif isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    return data


def get_xy_x(a, b):
    return np.array([[a, 0.0], [a, 1.0], [b, 1.0], [b, 0.0], [a, 0.0]])


def get_xy_y(a, b):
    return np.array([[0.0, a], [0.0, b], [1.0, b], [1.0, a], [0.0, a]])


def get_true_indices(a):
    return list(compress(range(len(a)), a))


# https://www.pythonguis.com/widgets/qcolorbutton-a-color-selector-tool-for-pyqt/
class ColorButton(QtWidgets.QPushButton):
    """
    Custom Qt Widget to show a chosen color.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    """

    colorChanged = QtCore.Signal(object)

    def __init__(self, *args, color=None, **kwargs):
        super(ColorButton, self).__init__(*args, **kwargs)

        self._color = None
        self._default = color
        self.pressed.connect(self.onColorPicker)

        # Set the initial/default state.
        self.setColor(self._default)

    def setColor(self, color):
        self._color = color
        self.colorChanged.emit(color.getRgbF())
        if self._color:
            self.setStyleSheet(
                "QWidget { background-color: %s; border: 0; }"
                % self._color.name(QtGui.QColor.HexArgb)
            )
        else:
            self.setStyleSheet("")

    def color(self):
        return self._color

    def onColorPicker(self):
        """
        Show color-picker dialog to select color.

        Qt will use the native dialog by default.

        """
        dlg = QtWidgets.QColorDialog(self)
        if self._color:
            dlg.setCurrentColor(QtGui.QColor(self._color))
        if dlg.exec_():
            self.setColor(dlg.currentColor())

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.RightButton:
            self.setColor(self._default)
        return super(ColorButton, self).mousePressEvent(e)


class ItoolImageItem(pg.ImageItem):
    def __init__(self, itool, *args, **kargs):
        self.itool = itool
        super().__init__(*args, **kargs)

    def mouseDragEvent(self, ev):
        if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            super().mouseDragEvent(ev)
            return
        else:
            ev.accept()
            self.itool.onMouseDrag(ev)


class ItoolPlotItem(pg.PlotItem):
    def __init__(self, itool, *args, **kargs):
        self.itool = itool
        super().__init__(*args, **kargs)

    def mouseDragEvent(self, ev):
        if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            ev.ignore()
            return
        else:
            ev.accept()
            self.itool.onMouseDrag(ev)


class ItoolCursorLine(pg.InfiniteLine):
    def __init__(self, itool, *args, **kargs):
        self.itool = itool
        super().__init__(*args, **kargs)

    def mouseDragEvent(self, ev):
        if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            self.setMovable(True)
            super().mouseDragEvent(ev)
        else:
            self.setMovable(False)
            self.setMouseHover(False)
            ev.ignore()

    def mouseClickEvent(self, ev):
        if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            self.setMovable(True)
            super().mouseClickEvent(ev)
        else:
            self.setMovable(False)
            self.setMouseHover(False)
            ev.ignore()

    def hoverEvent(self, ev):
        if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            self.setMovable(True)
            super().hoverEvent(ev)
        else:
            self.setMovable(False)
            self.setMouseHover(False)


class pg_itool(pg.GraphicsLayoutWidget):
    """A interactive tool based on `pyqtgraph` for exploring 3D data.

    For the tool to remain responsive you must
    keep a reference to it.

    Parameters
    ----------
    data : `xarray.DataArray`
        The data to explore. Must have three coordinate axes.

    snap :  bool, default: True
        Wheter to snap the cursor to data pixels.

    gamma : float, default: 0.5
        Colormap default gamma.
    cmap : str or `pyqtgraph.colorMap`, default: 'magma'
        Default colormap.

    bench : bool, default: False
        Whether to print frames per second.

    plot_kw : dict, optional
        Extra arguments to `matplotlib.pyplot.plot`: refer to the
        `matplotlib` documentation for a list of all possible arguments.
    cursor_kw : dict, optional
        Extra arguments to `pyqtgraph.InfiniteLine`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.
    image_kw : dict, optional
        Extra arguments to `pyqtgraph.ImageItem`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.
    profile_kw : dict, optional
        Extra arguments to `PlotDataItem.__init__`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.
    span_kw : dict, optional
        Extra arguments to `pyqtgraph.LinearRegionItem`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.
    fermi_kw : dict, optional
        Extra arguments to `pyqtgraph.InfiniteLine`: refer to the
        `pyqtgraph` documentation for a list of all possible arguments.

    Notes
    -----
    Axes indices for 3D data:
        ┌───┬─────┐
        │ 1 │     │
        │───┤  3  │
        │ 4 │     │
        │───┼───┬─│
        │ 0 │ 5 │2│
        └───┴───┴─┘
    Axes indices for 2D data:
        ┌───┬───┐
        │ 1 │   │
        │───┼───│
        │ 0 │ 2 │
        └───┴───┘

    Signals
    -------
    sigDataChanged(self)
    sigIndexChanged(indices, values)

    """

    sigDataChanged = QtCore.Signal(object)
    sigIndexChanged = QtCore.Signal(list, list)

    _only_axis = ("x", "y", "z")
    _only_maps = "maps"

    _get_middle_index = lambda _, x: len(x) // 2 - (1 if len(x) % 2 == 0 else 0)

    def __init__(
        self,
        data,
        snap=True,
        gamma=0.5,
        cmap="BlWh",
        bench=False,
        plot_kw={},
        cursor_kw={},
        image_kw={},
        profile_kw={},
        span_kw={},
        fermi_kw={},
        zero_centered=False,
        *args,
        **kwargs,
    ):

        super().__init__(show=True, *args, **kwargs)
        self.qapp = QtCore.QCoreApplication.instance()
        self.screen = self.qapp.primaryScreen()
        self.snap = snap
        self.gamma = gamma
        self.cmap = cmap
        self.bench = bench
        self.colorbar = None
        self.plot_kw = plot_kw
        self.cursor_kw = cursor_kw
        self.image_kw = image_kw
        self.profile_kw = profile_kw
        self.span_kw = span_kw
        self.fermi_kw = fermi_kw
        self.zero_centered = zero_centered
        if self.zero_centered:
            self.gamma = 1.0
            self.cmap = "bwr"

        # cursor_c = pg.mkColor(0.5)
        cursor_c, cursor_c_hover, span_c, span_c_edge = [pg.mkColor('cyan') for _ in range(4)]
        cursor_c.setAlphaF(0.75)
        cursor_c_hover.setAlphaF(0.9)
        span_c.setAlphaF(0.15)
        span_c_edge.setAlphaF(0.35)
        # span_c_hover = pg.mkColor(0.75)
        # span_c_hover.setAlphaF(0.5)

        self.cursor_kw.update(
            dict(
                pen=pg.mkPen(cursor_c),
                hoverPen=pg.mkPen(cursor_c_hover),
            )
        )
        self.plot_kw.update(dict(defaultPadding=0.0, clipToView=True))
        # self.profile_kw.update(dict(
        #     linestyle='-', linewidth=.8,
        #     color=colors.to_rgba(plt.rcParams.get('axes.edgecolor'),
        #                          alpha=1),
        #     animated=self.useblit, visible=True,
        # ))
        # self.fermi_kw.update(dict(
        #     linestyle='--', linewidth=.8,
        #     color=colors.to_rgba(plt.rcParams.get('axes.edgecolor'),
        #                          alpha=1),
        #     animated=False,
        # ))
        self.image_kw.update(
            dict(
                autoDownsample=True,
                axisOrder="row-major",
            )
        )
        self.span_kw.update(
            dict(
                movable=False,
                pen=pg.mkPen(span_c_edge, width=1),
                brush=pg.mkBrush(span_c),
            )
        )

        self.data_ndim = None

        # self.data_vals = None
        # self.data_vals_T = None
        # self.data_dims = None
        # self.data_coords = None
        # self.data_shape = None
        # self.data_incs = None
        # self.data_lims = None
        # self.cursor_pos = None

        self.set_data(data, update_all=True, reset_cursor=True)
        self.set_cmap()
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()
        self.connect_signals()

    def _update_stretch(self, factor=None):
        if factor is None:
            if self.data_ndim == 2:
                factor = [25000, 75000]
            elif self.data_ndim == 3:
                factor = [7500, 35000, 57500]
        # elif factor == 0:
        # for i in range(self.ci.layout.columnCount()):
        #         self.ci.layout.setColumnStretchFactor(i, 0)
        # self.ci.layout.setColumnAlignment(i, QtCore.Qt.AlignCenter)
        # for i in range(self.ci.layout.rowCount()):
        # self.ci.layout.setRowAlignment(i, QtCore.Qt.AlignCenter)
        #         self.ci.layout.setRowStretchFactor(i, 0)
        #     return

        for i in range(len(factor)):

            self.ci.layout.setColumnMinimumWidth(i, 0.0)
            self.ci.layout.setColumnStretchFactor(i, factor[-i - 1])
            self.ci.layout.setRowStretchFactor(i, factor[i])

    def _initialize_layout(
        self, horiz_pad=45, vert_pad=30, inner_pad=15, font_size=10.0
    ):
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        self.ci.layout.setSpacing(inner_pad)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        if self.data_ndim == 2:
            self.axes = [ItoolPlotItem(self, **self.plot_kw) for _ in range(3)]
            self.addItem(self.axes[0], 1, 0, 1, 1)
            self.addItem(self.axes[1], 0, 0, 1, 1)
            self.addItem(self.axes[2], 1, 1, 1, 1)
            valid_selection = ((1, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1))
        elif self.data_ndim == 3:
            self.axes = [ItoolPlotItem(self, **self.plot_kw) for _ in range(6)]
            self.addItem(self.axes[0], 2, 0, 1, 1)
            self.addItem(self.axes[1], 0, 0, 1, 1)
            self.addItem(self.axes[2], 2, 2, 1, 1)
            self.addItem(self.axes[3], 0, 1, 2, 2)
            self.addItem(self.axes[4], 1, 0, 1, 1)
            self.addItem(self.axes[5], 2, 1, 1, 1)
            valid_selection = (
                (1, 0, 0, 1),
                (1, 1, 0, 0),
                (0, 0, 1, 1),
                (0, 1, 1, 0),
                (1, 0, 0, 0),
                (0, 0, 0, 1),
            )
        else:
            raise NotImplementedError("Only supports 2D and 3D arrays.")

        for i, (p, sel) in enumerate(zip(self.axes, valid_selection)):
            # p.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
            # QtWidgets.QSizePolicy.Expanding)
            p.setDefaultPadding(0)
            for axis in ["left", "bottom", "right", "top"]:
                p.getAxis(axis).setTickFont(font)
                p.getAxis(axis).setStyle(
                    autoExpandTextSpace=True, autoReduceTextSpace=True
                )
            p.showAxes(sel, showValues=sel, size=(horiz_pad, vert_pad))
            if i in [1, 4]:
                p.setXLink(self.axes[0])
            elif i in [2, 5]:
                p.setYLink(self.axes[0])
        self._update_stretch()

    def _lims_to_rect(self, i, j):
        x = self.data_lims[i][0] - self.data_incs[i]
        y = self.data_lims[j][0] - self.data_incs[j]
        w = self.data_lims[i][-1] - x
        h = self.data_lims[j][-1] - y
        x += 0.5 * self.data_incs[i]
        y += 0.5 * self.data_incs[j]
        return QtCore.QRectF(x, y, w, h)

    def _initialize_plots(self):
        if self.data_ndim == 2:
            self.maps = (ItoolImageItem(self, name="Main Image", **self.image_kw),)
            self.hists = (
                self.axes[1].plot(name="X Profile", **self.profile_kw),
                self.axes[2].plot(name="Y Profile", **self.profile_kw),
            )
            self.cursors = (
                (
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                ),
                (
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                ),
            )
            self.spans = (
                (
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                ),
            )
            self.axes[0].addItem(self.maps[0])
            self.axes[0].addItem(self.cursors[0][0])
            self.axes[1].addItem(self.cursors[0][1])
            self.axes[0].addItem(self.cursors[1][0])
            self.axes[2].addItem(self.cursors[1][1])
            self.axes[0].addItem(self.spans[0][0])
            self.axes[1].addItem(self.spans[0][1])
            self.axes[0].addItem(self.spans[1][0])
            self.axes[2].addItem(self.spans[1][1])
        elif self.data_ndim == 3:
            self.maps = (
                ItoolImageItem(self, name="Main Image", **self.image_kw),
                ItoolImageItem(self, name="Horiz Slice", **self.image_kw),
                ItoolImageItem(self, name="Vert Slice", **self.image_kw),
            )
            self.hists = (
                self.axes[1].plot(name="X Profile", **self.profile_kw),
                self.axes[2].plot(name="Y Profile", **self.profile_kw),
                self.axes[3].plot(name="Z Profile", **self.profile_kw),
            )
            self.cursors = (
                (
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="X Cursor", **self.cursor_kw
                    ),
                ),
                (
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Y Cursor", **self.cursor_kw
                    ),
                ),
                (
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="Z Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=90, movable=True, name="Z Cursor", **self.cursor_kw
                    ),
                    ItoolCursorLine(
                        self, angle=0, movable=True, name="Z Cursor", **self.cursor_kw
                    ),
                ),
            )
            self.spans = (
                (
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="vertical", **self.span_kw),
                    pg.LinearRegionItem(orientation="horizontal", **self.span_kw),
                ),
            )
            self.axes[0].addItem(self.maps[0])
            self.axes[4].addItem(self.maps[1])
            self.axes[5].addItem(self.maps[2])
            self.axes[0].addItem(self.cursors[0][0])
            self.axes[1].addItem(self.cursors[0][1])
            self.axes[4].addItem(self.cursors[0][2])
            self.axes[0].addItem(self.cursors[1][0])
            self.axes[2].addItem(self.cursors[1][1])
            self.axes[5].addItem(self.cursors[1][2])
            self.axes[3].addItem(self.cursors[2][0])
            self.axes[5].addItem(self.cursors[2][1])
            self.axes[4].addItem(self.cursors[2][2])
            self.axes[0].addItem(self.spans[0][0])
            self.axes[1].addItem(self.spans[0][1])
            self.axes[4].addItem(self.spans[0][2])
            self.axes[0].addItem(self.spans[1][0])
            self.axes[2].addItem(self.spans[1][1])
            self.axes[5].addItem(self.spans[1][2])
            self.axes[3].addItem(self.spans[2][0])
            self.axes[5].addItem(self.spans[2][1])
            self.axes[4].addItem(self.spans[2][2])
            # if self.data_lims[-1][-1] * self.data_lims[-1][0] < 0:
            # self.axes[3].axvline(0., label='Fermi Level', **self.fermi_kw)
            # for m in self.maps:
            # m.sigImageChanged.connect(self._refresh_histograms)

        self.all = self.maps + self.hists + self.cursors
        for s in chain.from_iterable(self.spans):
            s.setVisible(False)

    def set_labels(self, labels=None):
        """labels: list or tuple of str"""
        if labels is None:
            labels = self.data_dims
        # labels_ = [self.labelify(l) for l in labels]
        labels_ = labels
        self.axes[0].setMathLabels(left=labels_[1], bottom=labels_[0])
        self.axes[1].setMathLabels(top=labels_[0])
        self.axes[2].setMathLabels(right=labels_[1])
        if self.data_ndim == 3:
            self.axes[3].setMathLabels(top=labels_[2])
            self.axes[4].setMathLabels(left=labels_[2])
            self.axes[5].setMathLabels(bottom=labels_[2])

    def set_data(self, data, update_all=False, reset_cursor=True):

        # Data properties
        self.data = parse_data(data)
        ndim_old = self.data_ndim
        self.data_ndim = self.data.ndim
        if self.data_ndim != ndim_old:
            update_all = True
        self.data_vals = self.data.values
        self._assign_vals_T()
        self.data_dims = self.data.dims
        self.data_shape = self.data.shape
        self.data_coords = tuple(self.data[dim].values for dim in self.data_dims)
        self.data_incs = tuple(coord[1] - coord[0] for coord in self.data_coords)
        self.data_lims = tuple((coord[0], coord[-1]) for coord in self.data_coords)
        if update_all:
            self.clear()
            self._initialize_layout()
            self._initialize_plots()
            self.clim_locked = False
            self.clim_list = [()] * self.data_ndim
            self.avg_win = [
                1,
            ] * self.data_ndim
            self.averaged = [
                False,
            ] * self.data_ndim
            self.axis_locked = [
                False,
            ] * self.data_ndim

        # Imagetool properties
        if reset_cursor:
            self.cursor_pos = [
                None,
            ] * self.data_ndim
            self._last_ind = [
                None,
            ] * self.data_ndim
            self.reset_cursor()
        self.set_labels()
        self._apply_change()

        self.sigDataChanged.emit(self)

    def toggle_colorbar(self, val):
        if self.colorbar is None:
            self.colorbar = ItoolColorBar(self, width=20)
            self.addItem(self.colorbar, None, None, self.ci.layout.rowCount(), 1)
        self.colorbar.setVisible(val)

    def reset_cursor(self):
        """Return the cursor to the center of the image."""
        for axis, coord in enumerate(self.data_coords):
            self.set_index(axis, self._get_middle_index(coord), update=False)

    def _cursor_drag(self, axis, line):
        self.set_value(axis, line.value())

    def connect_signals(self):
        """Connect events."""
        for axis, cursors in enumerate(self.cursors):
            for c in cursors:
                c.sigDragged.connect(lambda v, i=axis: self.set_value(i, v.value()))
        # self.proxy = pg.SignalProxy(
        #     self.scene().sigMouseMoved,
        #     rateLimit=self.screen.refreshRate(),
        #     slot=self.onMouseDrag,
        # )
        self.scene().sigMouseClicked.connect(self.onMouseDrag)

        if self.bench:
            from collections import deque

            self._elapsed = deque(maxlen=100)
            timer = QtCore.QTimer()
            # timer.timeout.connect(self._apply_change)
            timer.start(0)
            self._fpsLastUpdate = perf_counter()

    def _get_curr_axes_index(self, pos):
        for i, ax in enumerate(self.axes):
            if ax.vb.sceneBoundingRect().contains(pos):
                return i, self._get_mouse_datapos(ax, pos)
        if self.colorbar is not None:
            if self.colorbar.sceneBoundingRect().contains(pos):
                return 6, self._get_mouse_datapos(self.colorbar, pos)
        return None, None

    def reset_timer(self, *args):
        self._elapsed.clear()

    def _measure_fps(self):
        self.qapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        self._t_end = perf_counter()
        self._elapsed.append(self._t_end - self._t_start)
        if self._t_end - self._fpsLastUpdate > 0.2:
            self._fpsLastUpdate = self._t_end
            average = np.mean(self._elapsed)
            fps = 1 / average
            self.axes[1].setTitle("%0.2f fps - %0.1f ms avg" % (fps, average * 1_000))

    def labelify(self, text):
        """Prettify some frequently used axis labels."""
        labelformats = dict(
            kx="$k_x$",
            ky="$k_y$",
            kz="$k_z$",
            alpha="$\\alpha$",
            beta="$\\beta$",
            theta="$\\theta$",
            phi="$\\phi$",
            chi="$\\chi$",
            eV="$E$",
        )
        try:
            return labelformats[text]
        except KeyError:
            return text

    def _assign_vals_T(self):
        if self.data_ndim == 2:
            self.data_vals_T = np.ascontiguousarray(self.data_vals.T)
        elif self.data_ndim == 3:
            self.data_vals_T = np.ascontiguousarray(
                np.transpose(self.data_vals, axes=(1, 2, 0))
            )
        else:
            raise NotImplementedError("Wrong data dimensions")

    def set_cmap(self, cmap=None, gamma=None, reverse=False, highContrast=False):
        if cmap is None:
            cmap = self.cmap
        if gamma is None:
            gamma = self.gamma
        if cmap is not self.cmap:
            self.cmap = cmap
        if gamma is not self.gamma:
            self.gamma = gamma
        self.norm_cmap = pg_colormap_powernorm(
            self.cmap,
            self.gamma,
            reverse=reverse,
            highContrast=highContrast,
            zeroCentered=self.zero_centered,
        )
        for im in self.maps:
            im._colorMap = self.norm_cmap
            im.setLookupTable(self.norm_cmap.getStops()[1], update=False)
        self._apply_change(self._only_maps)

    def set_clim_lock(self, lock):
        if self.colorbar is not None:
            self.colorbar.autolevels = ~lock
        if lock:
            self.clim_locked = True
            for i, m in enumerate(self.maps):
                self.clim_list[i] = m.getLevels()
        else:
            self.clim_locked = False

    def set_index(self, axis, index, update=True):
        self._last_ind[axis] = index
        self.cursor_pos[axis] = self.data_coords[axis][index]
        if update is True:
            self._apply_change(self._only_axis[axis])

    def set_value(self, axis, val, update=True):
        self._last_ind[axis] = self.get_index_of_value(axis, val)
        self.cursor_pos[axis] = val
        if update is True:
            self._apply_change(self._only_axis[axis])

    def set_cursor_color(self, c):
        for cursor in self.cursors:
            cursor.setPen(pg.mkPen(c))
        self._apply_change()

    def set_line_color(self, c):
        for line in self.hists:
            line.setPen(pg.mkPen(c))
        self._apply_change()

    def set_navg(self, axis, n, update=True):
        self.avg_win[axis] = n
        if n == 1:
            self.averaged[axis] = False
        else:
            self.averaged[axis] = True
        if update:
            self._refresh_navg(reset=False)

    def _refresh_navg(self, reset=False):
        self._slice_block = None
        if reset:
            for axis in range(self.data_ndim):
                self.averaged[axis] = False
                self.avg_win[axis] = 1
        for axis in range(self.data_ndim):
            for s in self.spans[axis]:
                s.setVisible(self.averaged[axis])
        # if not any(self.averaged):
        #     self.data_vals = self.data.values
        # else:
        #     vals = self.data.values
        #     self.data_vals = move_mean_centered_multiaxis(vals, self.avg_win)
        # self._assign_vals_T()
        self._apply_change()

    def _get_bin_slice(self, axis):
        if self.averaged[axis]:
            center = self._last_ind[axis]
            window = self.avg_win[axis]
            return slice(center - window // 2, center + (window - 1) // 2 + 1)
        else:
            return slice(self._last_ind[axis], self._last_ind[axis] + 1)

    def _get_binned_data(self, axis):
        self._slice_block = None
        if not self.averaged[axis]:
            return np.take(self.data_vals_T, self._last_ind[axis], axis=axis - 1)
        else:
            axis -= 1
            return numbagg.nanmean(
                self.data_vals_T[
                    (slice(None),) * (axis % self.data_ndim)
                    + (self._get_bin_slice(axis + 1),)
                ],
                axis=axis,
            )

    def _binned_profile(self, avg_axis):
        if not any(self.averaged):
            return self._block_slicer(avg_axis, [self._last_ind[i] for i in avg_axis])
        slices = tuple(self._get_bin_slice(ax) for ax in avg_axis)
        self._slice_block = self._block_slicer(avg_axis, slices)
        return numbagg.nanmean(self._slice_block, axis=avg_axis)
    
    def _block_slicer(self, axis=None, slices=None):
        axis = [ax % self.data_ndim for ax in axis]
        return self.data_vals[
            tuple(
                slices[axis.index(d)] if d in axis else slice(None) for d in range(self.data_ndim)
            )
        ]

    def slicer2(self, axis=None, slices=None):
        axis = [ax % self.data_ndim for ax in axis]
        slice_selector = dict(zip(axis, slices))
        element = lambda dim_: slice_selector[dim_] if dim_ in slice_selector.keys() else slice(None)
        return self.data_vals[tuple(element(dim) for dim in range(self.data_ndim))]

    def update_spans(self, axis):
        slc = self._get_bin_slice(axis)
        lb = max(0, slc.start)
        ub = min(self.data_shape[axis] - 1, slc.stop - 1)
        region = self.data_coords[axis][[lb, ub]]
        for span in self.spans[axis]:
            span.setRegion(region)

    def get_index_of_value(self, axis, val):
        ind = min(
            round((val - self.data_lims[axis][0]) / self.data_incs[axis]),
            self.data_shape[axis] - 1,
        )
        if ind < 0:
            return 0
        return ind

    def set_axis_lock(self, axis, lock):
        self.axis_locked[axis] = lock

    def transpose_axes(self, axis1, axis2):
        dims_new = list(self.data_dims)
        dims_new[axis1], dims_new[axis2] = self.data_dims[axis2], self.data_dims[axis1]
        data_new = self.data.transpose(*dims_new)
        self._last_ind[axis1], self._last_ind[axis2] = (
            self._last_ind[axis2],
            self._last_ind[axis1],
        )
        self.cursor_pos[axis1], self.cursor_pos[axis2] = (
            self.cursor_pos[axis2],
            self.cursor_pos[axis1],
        )
        self.clim_list[axis1], self.clim_list[axis2] = (
            self.clim_list[axis2],
            self.clim_list[axis1],
        )
        self.avg_win[axis1], self.avg_win[axis2] = (
            self.avg_win[axis2],
            self.avg_win[axis1],
        )
        self.averaged[axis1], self.averaged[axis2] = (
            self.averaged[axis2],
            self.averaged[axis1],
        )
        self.axis_locked[axis1], self.axis_locked[axis2] = (
            self.axis_locked[axis2],
            self.axis_locked[axis1],
        )
        self.set_data(data_new, update_all=False, reset_cursor=False)

    def get_key_modifiers(self):
        Qmods = self.qapp.queryKeyboardModifiers()
        mods = []
        if (Qmods & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            mods.append("shift")
        if (Qmods & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            mods.append("control")
        if (Qmods & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier:
            mods.append("alt")
        return mods

    def _get_mouse_datapos(self, plot, pos):
        """Returns mouse position in data coords"""
        mouse_point = plot.vb.mapSceneToView(pos)
        return mouse_point.x(), mouse_point.y()

    def onMouseDrag(self, evt):
        if self.bench:
            self._t_start = perf_counter()

        axis_ind, datapos = self._get_curr_axes_index(evt.scenePos())
        if hasattr(evt, "_buttonDownScenePos"):
            axis_start, _ = self._get_curr_axes_index(evt.buttonDownScenePos())
            if axis_ind != axis_start:
                return
        elif self.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            evt.ignore()
            return

        if axis_ind is None:
            return

        V = [None] * self.data_ndim
        if axis_ind == 0:
            V[0:2] = datapos
        elif axis_ind == 1:
            V[0] = datapos[0]
        elif axis_ind == 2:
            V[1] = datapos[1]
        elif axis_ind == 3:
            V[2] = datapos[0]
        elif axis_ind == 4:
            V[0], V[2] = datapos
        elif axis_ind == 5:
            V[2], V[1] = datapos
        elif axis_ind == 6:
            self.colorbar.isoline.setPos(datapos[1])
            return
        D = [v is not None for v in V]
        if not any(D):
            return
        for i in range(self.data_ndim):
            if D[i]:
                ind = self.get_index_of_value(i, V[i])
                if self.snap and (ind == self._last_ind[i]) or self.axis_locked[i]:
                    D[i] = False
                else:
                    self._last_ind[i] = ind
            if not D[i]:
                V[i] = self.cursor_pos[i]
        if not any(D):
            return
        if not self.snap:
            self.cursor_pos = V
        self._apply_change(D)
        if self.bench:
            self._measure_fps()

    def _apply_change(self, cond=None):
        if cond is None:
            update = (True,) * len(self.all)
        elif isinstance(cond, str):
            if cond == self._only_maps:
                if self.data_ndim == 2:
                    update = (True,) + (False,) * 4
                elif self.data_ndim == 3:
                    update = (True,) * 3 + (False,) * 6
            elif cond in self._only_axis:
                update = [False] * self.data_ndim
                update[self._only_axis.index(cond)] = True
            else:
                raise ValueError
        else:
            update = cond
        if len(update) == self.data_ndim:
            if self.data_ndim == 2:
                update = (False, update[1], update[0], update[0], update[1])
            elif self.data_ndim == 3:
                update = (
                    update[2],
                    update[1],
                    update[0],
                    update[1] or update[2],
                    update[0] or update[2],
                    update[0] or update[1],
                    update[0],
                    update[1],
                    update[2],
                )
        elif len(update) != len(self.all):
            raise ValueError
        for i in get_true_indices(update):
            self._refresh_data(i)

    @suppressnanwarning
    def _refresh_data(self, i):
        if self.snap:
            self.cursor_pos = [
                self.data_coords[i][self._last_ind[i]] for i in range(self.data_ndim)
            ]
        if self.data_ndim == 2:
            self._refresh_data_2d(i)
        elif self.data_ndim == 3:
            self._refresh_data_3d(i)
        self.sigIndexChanged.emit(self._last_ind, self.cursor_pos)

    # def get_levels(self, i):
    #     if self.clim_locked:
    #         if self.zero_centered:

    #         else:
    #             return self.clim_list[i]
    #     else:
    #         self.all[i].getLevels()
    # def _refresh_histograms(self):
    #     self.hists[0].setData(self.data_coords[0], self.maps[0].image[self._last_ind[1], :])
    #     self.hists[1].setData(self.maps[0].image[:, self._last_ind[0]], self.data_coords[1])
    #     if self.data_ndim == 3:
    #         self.hists[2].setData(self.data_coords[2], self.maps[1].image[:, self._last_ind[0]])

    def _refresh_data_2d(self, i):
        if i == 0:
            if self.clim_locked:
                self.all[i].setImage(
                    self.data_vals_T,
                    levels=self.clim_list[0],
                    rect=self._lims_to_rect(0, 1),
                )
            else:
                self.all[i].setImage(self.data_vals_T, rect=self._lims_to_rect(0, 1))
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 1:
            self.all[i].setData(
                self.data_coords[0],
                # self.data_vals_T[self._last_ind[1], :]
                # self.maps[0].image[self._last_ind[1], :],
                self._binned_profile([1]),
            )
        elif i == 2:
            self.all[i].setData(
                # self.data_vals_T[:, self._last_ind[0]],
                # self.maps[0].image[:, self._last_ind[0]],
                self._binned_profile([0]),
                self.data_coords[1],
            )
        elif i in [3, 4]:
            for cursor in self.all[i]:
                cursor.maxRange = self.data_lims[i - 3]
                cursor.setPos(self.cursor_pos[i - 3])
                if self.averaged[i - 3]:
                    self.update_spans(i - 3)

    def _refresh_data_3d(self, i):
        if i == 0:
            if self.clim_locked:
                self.all[i].setImage(
                    # self.data_vals_T[:, self._last_ind[2], :],
                    self._get_binned_data(2),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(0, 1),
                )
            else:
                self.all[i].setImage(
                    # self.data_vals_T[:, self._last_ind[2], :],
                    self._get_binned_data(2),
                    rect=self._lims_to_rect(0, 1),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 1:
            if self.clim_locked:
                self.all[i].setImage(
                    # self.data_vals_T[self._last_ind[1], :, :],
                    self._get_binned_data(1),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(0, 2),
                )
            else:
                self.all[i].setImage(
                    # self.data_vals_T[self._last_ind[1], :, :],
                    self._get_binned_data(1),
                    rect=self._lims_to_rect(0, 2),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 2:
            if self.clim_locked:
                self.all[i].setImage(
                    self._get_binned_data(0),
                    levels=self.clim_list[i],
                    rect=self._lims_to_rect(2, 1),
                )
            else:
                self.all[i].setImage(
                    self._get_binned_data(0),
                    rect=self._lims_to_rect(2, 1),
                )
                if self.zero_centered:
                    lim = np.amax(np.abs(np.asarray(self.all[i].getLevels())))
                    self.all[i].setLevels([-lim, lim])
        elif i == 3:
            self.hists[0].setData(
                self.data_coords[0],
                # self.data_vals[:, self._last_ind[1], self._last_ind[2]],
                self._binned_profile((1, 2)),
            )
        elif i == 4:
            self.hists[1].setData(
                # self.data_vals[self._last_ind[0], :, self._last_ind[2]],
                # self.maps[0].image[:,self._last_ind[0]],
                self._binned_profile((0, 2)),
                self.data_coords[1],
            )
        elif i == 5:
            self.hists[2].setData(
                self.data_coords[2],
                # self.data_vals[self._last_ind[0], self._last_ind[1], :],
                # self.maps[1].image[:, self._last_ind[0]]
                self._binned_profile((0, 1)),
            )
        elif i in [6, 7, 8]:
            for cursor in self.all[i]:
                cursor.maxRange = self.data_lims[i - 6]
                cursor.setPos(self.cursor_pos[i - 6])
                if self.averaged[i - 6]:
                    self.update_spans(i - 6)

    def _drawpath(self):
        # ld = LineDrawer(self.canvas, self.axes[0])
        # points = ld.draw_line()
        # print(points)
        # TODO
        pass

    def _onselectpath(self, verts):
        print(verts)


class ImageToolColors(QtWidgets.QDialog):
    def __init__(self, parent):
        self.parent = parent
        super().__init__(self.parent)
        self.setWindowTitle("Colors")
        raise NotImplementedError

    #     self.cursor_default = color_to_QColor(self.parent.itool.cursor_kw["color"])
    #     self.line_default = color_to_QColor(self.parent.itool.profile_kw["color"])
    #     self.cursor_current = color_to_QColor(self.parent.itool.cursors[0].get_color())
    #     self.line_current = color_to_QColor(self.parent.itool.hists[0].get_color())

    #     if (self.cursor_default.getRgbF() == self.cursor_current.getRgbF()) & (
    #         self.line_default.getRgbF() == self.line_current.getRgbF()
    #     ):
    #         buttons = QtWidgets.QDialogButtonBox(
    #             QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    #         )
    #     else:
    #         buttons = QtWidgets.QDialogButtonBox(
    #             QtWidgets.QDialogButtonBox.RestoreDefaults
    #             | QtWidgets.QDialogButtonBox.Ok
    #             | QtWidgets.QDialogButtonBox.Cancel
    #         )
    #         buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
    #             self.reset_colors
    #         )
    #     buttons.rejected.connect(self.reject)
    #     buttons.accepted.connect(self.accept)

    #     cursorlabel = QtWidgets.QLabel("Cursors:")
    #     linelabel = QtWidgets.QLabel("Lines:")
    #     self.cursorpicker = ColorButton(color=self.cursor_current)
    #     self.cursorpicker.colorChanged.connect(self.parent.itool.set_cursor_color)
    #     self.linepicker = ColorButton(color=self.line_current)
    #     self.linepicker.colorChanged.connect(self.parent.itool.set_line_color)

    #     layout = QtWidgets.QGridLayout()
    #     layout.addWidget(cursorlabel, 0, 0)
    #     layout.addWidget(self.cursorpicker, 0, 1)
    #     layout.addWidget(linelabel, 1, 0)
    #     layout.addWidget(self.linepicker, 1, 1)
    #     layout.addWidget(buttons)
    #     self.setLayout(layout)

    # def reject(self):
    #     self.cursorpicker.setColor(self.cursor_current)
    #     self.linepicker.setColor(self.line_current)
    #     super().reject()

    # def reset_colors(self):
    #     self.cursorpicker.setColor(self.cursor_default)
    #     self.linepicker.setColor(self.line_default)


@numba.njit(nogil=True)
def fast_isocurve_extend(data):
    d2 = np.empty((data.shape[0] + 2, data.shape[1] + 2), dtype=data.dtype)
    d2[1:-1, 1:-1] = data
    d2[0, 1:-1] = data[0]
    d2[-1, 1:-1] = data[-1]
    d2[1:-1, 0] = data[:, 0]
    d2[1:-1, -1] = data[:, -1]
    d2[0, 0] = d2[0, 1]
    d2[0, -1] = d2[1, -1]
    d2[-1, 0] = d2[-1, 1]
    d2[-1, -1] = d2[-1, -2]
    return d2


@numba.njit(nogil=True)
def fast_isocurve_lines(data, level, index, extendToEdge=False):
    sideTable = (
        [np.int64(x) for x in range(0)],
        [0, 1],
        [1, 2],
        [0, 2],
        [0, 3],
        [1, 3],
        [0, 1, 2, 3],
        [2, 3],
        [2, 3],
        [0, 1, 2, 3],
        [1, 3],
        [0, 3],
        [0, 2],
        [1, 2],
        [0, 1],
        [np.int64(x) for x in range(0)],
    )
    edgeKey = [[(0, 1), (0, 0)], [(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 1)]]
    lines = []
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            sides = sideTable[index[i, j]]
            for l in range(0, len(sides), 2):
                edges = sides[l : l + 2]
                pts = []
                for m in range(2):
                    p1, p2 = edgeKey[edges[m]][0], edgeKey[edges[m]][1]
                    v1, v2 = data[i + p1[0], j + p1[1]], data[i + p2[0], j + p2[1]]
                    f = (level - v1) / (v2 - v1)
                    fi = 1.0 - f
                    p = (
                        p1[0] * fi + p2[0] * f + i + 0.5,
                        p1[1] * fi + p2[1] * f + j + 0.5,
                    )
                    if extendToEdge:
                        p = (
                            min(data.shape[0] - 2, max(0, p[0] - 1)),
                            min(data.shape[1] - 2, max(0, p[1] - 1)),
                        )
                    pts.append(p)
                lines.append(pts)
    return lines


@numba.njit(nogil=True)
def fast_isocurve_lines_connected(data, level, index, extendToEdge=False):
    sideTable = (
        [np.int64(x) for x in range(0)],
        [0, 1],
        [1, 2],
        [0, 2],
        [0, 3],
        [1, 3],
        [0, 1, 2, 3],
        [2, 3],
        [2, 3],
        [0, 1, 2, 3],
        [1, 3],
        [0, 3],
        [0, 2],
        [1, 2],
        [0, 1],
        [np.int64(x) for x in range(0)],
    )
    edgeKey = [[(0, 1), (0, 0)], [(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 1)]]
    lines = []
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            sides = sideTable[index[i, j]]
            for l in range(0, len(sides), 2):
                edges = sides[l : l + 2]
                pts = []
                for m in range(2):
                    p1, p2 = edgeKey[edges[m]][0], edgeKey[edges[m]][1]
                    v1, v2 = data[i + p1[0], j + p1[1]], data[i + p2[0], j + p2[1]]
                    f = (level - v1) / (v2 - v1)
                    fi = 1.0 - f
                    p = (
                        p1[0] * fi + p2[0] * f + i + 0.5,
                        p1[1] * fi + p2[1] * f + j + 0.5,
                    )
                    if extendToEdge:
                        p = (
                            min(data.shape[0] - 2, max(0, p[0] - 1)),
                            min(data.shape[1] - 2, max(0, p[1] - 1)),
                        )
                    gridKey = (
                        i + (1 if edges[m] == 2 else 0),
                        j + (1 if edges[m] == 3 else 0),
                        edges[m] % 2,
                    )
                    pts.append((p, gridKey))
                lines.append(pts)
    return lines


def fast_isocurve(data, level, connected=False, extendToEdge=False, path=False):
    """
    Generate isocurve from 2D data using marching squares algorithm.

    ============== =========================================================
    **Arguments:**
    data           2D numpy array of scalar values
    level          The level at which to generate an isosurface
    connected      If False, return a single long list of point pairs
                   If True, return multiple long lists of connected point
                   locations. (This is slower but better for drawing
                   continuous lines)
    extendToEdge   If True, extend the curves to reach the exact edges of
                   the data.
    path           if True, return a QPainterPath rather than a list of
                   vertex coordinates.
    ============== =========================================================
    """

    if path is True:
        connected = True
    np.nan_to_num(data, copy=False)
    if extendToEdge:
        data = fast_isocurve_extend(data)

    ## mark everything below the isosurface level
    mask = data < level
    index = np.zeros([x - 1 for x in mask.shape], dtype=np.int64)
    fields = np.empty((2, 2), dtype=object)
    slices = [slice(0, -1), slice(1, None)]
    for i in range(2):
        for j in range(2):
            fields[i, j] = mask[slices[i], slices[j]]
            vertIndex = i + 2 * j
            index += fields[i, j] * 2**vertIndex
    ### make four sub-fields and compute indexes for grid cells
    if connected:
        lines = fast_isocurve_lines_connected(data, level, index, extendToEdge)
        points = dict()
        for a, b in lines:
            if a[1] not in points:
                points[a[1]] = [[a, b]]
            else:
                points[a[1]].append([a, b])
            if b[1] not in points:
                points[b[1]] = [[b, a]]
            else:
                points[b[1]].append([b, a])
        lines = fast_isocurve_chain(points)
    else:
        lines = fast_isocurve_lines(data, level, index, extendToEdge)

    if not path:
        return lines  ## a list of pairs of points

    path = QtGui.QPainterPath()
    for line in lines:
        path.moveTo(*line[0])
        for p in line[1:]:
            path.lineTo(*p)

    return path


def fast_isocurve_chain(points):
    for k in list(points.keys()):
        try:
            chains = points[k]
        except KeyError:
            continue
        for chain in chains:
            x = None
            while True:
                if x == chain[-1][1]:
                    break
                x = chain[-1][1]
                if x == k:
                    break
                y = chain[-2][1]
                connects = points[x]
                for conn in connects[:]:
                    if conn[1][1] != y:
                        chain.extend(conn[1:])
                del points[x]
            if chain[0][1] == chain[-1][1]:
                chains.pop()
                break
    lines_linked = [np.float64(x) for x in range(0)]
    for chain in points.values():
        if len(chain) == 2:
            chain = chain[1][1:][::-1] + chain[0]  # join together ends of chain
        else:
            chain = chain[0]
        lines_linked.append([p[0] for p in chain])
    return lines_linked


class betterIsocurve(pg.IsocurveItem):
    def __init__(
        self,
        data=None,
        level=0,
        pen="cyan",
        axisOrder=None,
        connected=False,
        extendToEdge=False,
    ):
        super().__init__(data, level, pen, axisOrder)
        self.connected = connected
        self.extendToEdge = extendToEdge

    def generatePath(self):
        if self.data is None:
            self.path = None
            return

        if self.axisOrder == "row-major":
            data = self.data.T
        else:
            data = self.data

        lines = fast_isocurve(data, self.level, self.connected, self.extendToEdge)
        # lines = pg.functions.isocurve(data, self.level, connected=True, extendToEdge=True)
        self.path = QtGui.QPainterPath()
        for line in lines:
            self.path.moveTo(*line[0])
            for p in line[1:]:
                self.path.lineTo(*p)


class ItoolColorBar(pg.PlotItem):
    def __init__(
        self,
        itool,
        width=25,
        horiz_pad=45,
        vert_pad=30,
        inner_pad=5,
        font_size=10,
        curve_kw={},
        line_kw={"pen": "cyan"},
        *args,
        **kwargs,
    ):
        super(ItoolColorBar, self).__init__(*args, **kwargs)
        self.itool = itool
        self.setDefaultPadding(0)
        self.cbar = ItoolImageItem(self.itool, axisOrder="row-major")
        self.npts = 4096
        self.autolevels = True

        self.cbar.setImage(np.linspace(0, 1, self.npts).reshape((-1, 1)))
        self.addItem(self.cbar)

        self.isocurve = betterIsocurve(**curve_kw)
        self.isocurve.setZValue(5)

        self.isoline = ItoolCursorLine(self.itool, angle=0, movable=True, **line_kw)
        self.addItem(self.isoline)
        self.isoline.setZValue(1000)
        self.isoline.sigPositionChanged.connect(self.update_level)

        self.setImageItem(self.itool.maps[0])
        self.setMouseEnabled(x=False, y=True)
        self.setMenuEnabled(True)

        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        for axis in ["left", "bottom", "right", "top"]:
            self.getAxis(axis).setTickFont(font)
        self.layout.setColumnFixedWidth(1, width)
        self.layout.setSpacing(inner_pad)
        self.layout.setContentsMargins(0.0, 0.0, 0.0, 0.0)
        self.showAxes(
            (True, True, True, True),
            showValues=(False, False, True, False),
            size=(horiz_pad, 0.0),
        )
        # self.getAxis('right').setStyle(
        # showValues=True, tickTextWidth=horiz_pad,
        # autoExpandTextSpace=False, autoReduceTextSpace=False)
        self.getAxis("top").setHeight(vert_pad)
        self.getAxis("bottom").setHeight(vert_pad)
        # self.getAxis('left').setWidth(inner_pad)

    def setImageItem(self, img):
        self.imageItem = weakref.ref(img)
        self.isocurve.setParentItem(img)
        img.sigImageChanged.connect(self.image_changed)
        self.image_changed()

    def image_changed(self):
        self.cmap_changed()
        levels = self.imageItem().getLevels()
        if self.autolevels:
            self.cbar.setRect(0.0, levels[0], 1.0, levels[1] - levels[0])
        else:
            mn, mx = self.imageItem().quickMinMax(targetSize=2**16)
            self.cbar.setRect(0.0, mn, 1.0, mx - mn)
            self.cbar.setLevels(levels / (mx - mn) - mn)
        self.isoline.setBounds(levels)
        self.update_isodata()

    def cmap_changed(self):
        self.cmap = self.imageItem()._colorMap
        self.lut = self.imageItem().lut
        # self.lut = self.cmap.getStops()[1]
        if not self.npts == self.lut.shape[0]:
            self.npts = self.lut.shape[0]
            self.cbar.setImage(self.cmap.pos.reshape((-1, 1)))
        self.cbar._colorMap = self.cmap
        self.cbar.setLookupTable(self.lut)
        # self.cbar.setColorMap(self.cmap)
        # pg.ImageItem

    def update_isodata(self):
        self.isocurve.setData(self.imageItem().image)

    def update_level(self, line):
        self.isocurve.setLevel(line.value())

    def setVisible(self, visible, *args, **kwargs):
        super().setVisible(visible, *args, **kwargs)
        self.isocurve.setVisible(visible, *args, **kwargs)
        # self.showAxes((False, False, True, False),
        #               showValues=(False, False, True, False),
        #               size=(45, 30))


class itoolCursors(QtWidgets.QWidget):
    def __init__(self, itool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itool = itool
        self.ndim = self.itool.data_ndim
        # self.layout = QtWidgets.QHBoxLayout(self)
        self.layout = FlowLayout(self)
        self._cursor_group = QtWidgets.QGroupBox(self)
        self._transpose_group = QtWidgets.QGroupBox(self)
        self.initialize_widgets()
        self.update_content()
        self.itool.sigIndexChanged.connect(self.update_spin)
        self.itool.sigDataChanged.connect(self.update_content)

    def initialize_widgets(self):
        cursor_layout = QtWidgets.QGridLayout(self._cursor_group)
        transpose_layout = QtWidgets.QHBoxLayout(self._transpose_group)
        self._spinlabels = tuple(QtWidgets.QPushButton(self) for _ in range(self.ndim))
        self._spin = tuple(QtWidgets.QSpinBox(self) for _ in range(self.ndim))
        self._dblspin = tuple(QtWidgets.QDoubleSpinBox(self) for _ in range(self.ndim))
        self._transpose_button = tuple(
            QtWidgets.QPushButton(self) for _ in range(self.ndim)
        )
        # if self.ndim == 2:
        #     self._hide_button = (QtWidgets.QPushButton(self),
        #                          QtWidgets.QPushButton(self))
        #     self._hide_button[0].toggled.connect(
        #         lambda val, i=1: self.toggle_axes(val, i))
        #     self._hide_button[1].toggled.connect(
        #         lambda val, i=2: self.toggle_axes(val, i))
        # elif self.ndim == 3:
        #     self._hide_button = (QtWidgets.QPushButton(self),
        #                          QtWidgets.QPushButton(self),
        #                          QtWidgets.QPushButton(self),
        #                          QtWidgets.QPushButton(self))
        #     self._hide_button[0].toggled.connect(
        #         lambda val, i=3: self.toggle_axes(val, i))
        #     self._hide_button[1].toggled.connect(
        #         lambda val, i=2: self.toggle_axes(val, i))
        #     self._hide_button[2].toggled.connect(
        #         lambda val, i=4: self.toggle_axes(val, i))
        #     self._hide_button[3].toggled.connect(
        #         lambda val, i=5: self.toggle_axes(val, i))

        self._snap_button = QtWidgets.QPushButton(self)
        self._snap_button.setCheckable(True)
        self._snap_button.toggled.connect(self._assign_snap)
        self._snap_button.setIcon(qta.icon(fonticons["snap"]))
        self._snap_button.setToolTip("Snap cursor to data")
        for i in range(self.ndim):
            self._spinlabels[i].setCheckable(True)
            self._spinlabels[i].toggled.connect(
                lambda v, axis=i: self.itool.set_axis_lock(axis, v)
            )
            self._spin[i].setSingleStep(1)
            self._spin[i].setWrapping(True)
            self._dblspin[i].setDecimals(3)
            self._dblspin[i].setWrapping(True)
            self._spin[i].valueChanged.connect(
                lambda v, axis=i: self._index_changed(axis, v)
            )
            self._dblspin[i].valueChanged.connect(
                lambda v, axis=i: self._value_changed(axis, v)
            )
            self._transpose_button[i].clicked.connect(
                lambda axis1=i, axis2=i - 1: self.itool.transpose_axes(axis1, axis2)
            )
            cursor_layout.addWidget(self._spinlabels[i], 0, 3 * i)
            cursor_layout.addWidget(self._spin[i], 0, 3 * i + 1)
            cursor_layout.addWidget(self._dblspin[i], 0, 3 * i + 2)
            # cursor_layout.addSpacing(5)
        self.layout.addWidget(self._cursor_group)
        self.layout.addWidget(self._transpose_group)
        for i, button in enumerate(self._transpose_button):
            transpose_layout.addWidget(button)
            button.setIcon(qta.icon(fonticons["transpose"][i]))
        transpose_layout.addWidget(self._snap_button)
        # for hb in self._hide_button:
        #     hb.setIcon(qta.icon(fonticons['layout']))
        #     hb.setCheckable(True)
        #     transpose_layout.addWidget(hb)
        # self.layout.addStretch()

    def toggle_axes(self, toggle, i):
        # self.itool._update_stretch(factor=0)
        self.itool.axes[i].setVisible(not toggle)

    def update_content(self):
        ndim = self.itool.data_ndim
        if ndim != self.ndim:
            self.layout.clear()
            self.ndim = ndim
            self.initialize_widgets()
        self._snap_button.blockSignals(True)
        self._snap_button.setChecked(self.itool.snap)
        self._snap_button.blockSignals(False)

        for i in range(self.ndim):
            self._spinlabels[i].blockSignals(True)
            self._spin[i].blockSignals(True)
            self._dblspin[i].blockSignals(True)

            self._spinlabels[i].setText(self.itool.data_dims[i])
            self._spinlabels[i].setChecked(self.itool.axis_locked[i])
            self._spinlabels[i].setMaximumWidth(
                self._spinlabels[i]
                .fontMetrics()
                .boundingRect(self._spinlabels[i].text())
                .width()
                + 10
            )

            self._spin[i].setRange(0, self.itool.data_shape[i] - 1)
            self._spin[i].setValue(self.itool._last_ind[i])

            self._dblspin[i].setRange(*self.itool.data_lims[i])
            self._dblspin[i].setSingleStep(self.itool.data_incs[i])
            self._dblspin[i].setValue(
                self.itool.data_coords[i][self.itool._last_ind[i]]
            )

            self._spinlabels[i].blockSignals(False)
            self._spin[i].blockSignals(False)
            self._dblspin[i].blockSignals(False)

    def _assign_snap(self, value):
        if value:
            self._snap_button.setIcon(qta.icon(fonticons["snap_off"]))
        else:
            self._snap_button.setIcon(qta.icon(fonticons["snap"]))
        self.itool.snap = value

    def _index_changed(self, axis, index):
        self._dblspin[axis].blockSignals(True)
        self.itool.set_index(axis, index)
        self._dblspin[axis].setValue(self.itool.data_coords[axis][index])
        self._dblspin[axis].blockSignals(False)

    def _value_changed(self, axis, value):
        self._spin[axis].blockSignals(True)
        self.itool.set_value(axis, value)
        self._spin[axis].setValue(self.itool._last_ind[axis])
        self._spin[axis].blockSignals(False)

    def update_spin(self, index, value):
        for i in range(self.ndim):
            self._spin[i].blockSignals(True)
            self._dblspin[i].blockSignals(True)
            self._spin[i].setValue(index[i])
            self._dblspin[i].setValue(value[i])
            self._spin[i].blockSignals(False)
            self._dblspin[i].blockSignals(False)


class itoolColors(QtWidgets.QWidget):
    def __init__(self, itool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itool = itool
        self.layout = FlowLayout(self)
        self._cmap_group = QtWidgets.QGroupBox(self)
        self._button_group = QtWidgets.QGroupBox(self)
        self.initialize_widgets()

    def initialize_widgets(self):
        cmap_layout = QtWidgets.QGridLayout(self._cmap_group)
        button_layout = QtWidgets.QGridLayout(self._button_group)

        self._gamma_spin = QtWidgets.QDoubleSpinBox(self)
        self._gamma_spin.setToolTip("Colormap gamma")
        self._gamma_spin.setSingleStep(0.01)
        self._gamma_spin.setRange(0.01, 100.0)
        self._gamma_spin.setValue(self.itool.gamma)
        self._gamma_spin.valueChanged.connect(self.set_cmap)
        gamma_label = QtWidgets.QLabel("g")
        gamma_label.setBuddy(self._gamma_spin)
        # gamma_label.setMaximumWidth(10)

        self._cmap_combo = itoolColorMaps(self)
        self._cmap_combo.setMaximumWidth(175)
        if isinstance(self.itool.cmap, str):
            self._cmap_combo.setCurrentText(self.itool.cmap)
        self._cmap_combo.insertItem(0, "Load all...")
        self._cmap_combo.textActivated.connect(self._cmap_combo_changed)

        self._cmap_r_button = QtWidgets.QPushButton(self)
        self._cmap_r_button.setCheckable(True)
        self._cmap_r_button.toggled.connect(self._set_cmap_reverse)
        self._cmap_r_button.setIcon(qta.icon(fonticons["invert"]))
        self._cmap_r_button.setToolTip("Invert colormap")

        self._cmap_mode_button = QtWidgets.QPushButton(self)
        self._cmap_mode_button.setCheckable(True)
        self._cmap_mode_button.toggled.connect(self.set_cmap)
        self._cmap_mode_button.setIcon(qta.icon(fonticons["contrast"]))
        self._cmap_mode_button.setToolTip("High contrast mode")

        self._cmap_lock_button = QtWidgets.QPushButton(self)
        self._cmap_lock_button.setCheckable(True)
        self._cmap_lock_button.toggled.connect(self._set_clim_lock)
        self._cmap_lock_button.setIcon(qta.icon(fonticons["unlock"]))
        self._cmap_lock_button.setToolTip("Lock colors")

        self._cbar_show_button = QtWidgets.QPushButton(self)
        self._cbar_show_button.setCheckable(True)
        self._cbar_show_button.toggled.connect(self.itool.toggle_colorbar)
        self._cbar_show_button.setIcon(qta.icon(fonticons["colorbar"]))
        self._cbar_show_button.setToolTip("Show colorbar")

        colors_button = QtWidgets.QPushButton(self)
        colors_button.clicked.connect(self._color_button_clicked)
        colors_button.setIcon(qta.icon(fonticons["palette"]))
        # style_label = QtWidgets.QLabel('Style:', parent=self)
        # style_combo = QtWidgets.QComboBox(self)
        # style_combo.setToolTip('Qt style')
        # style_combo.addItems(qt_style_names())
        # style_combo.textActivated.connect(change_style)
        # style_combo.setCurrentText('Fusion')
        # style_label.setBuddy(style_combo)

        cmap_layout.addWidget(gamma_label, 0, 0)
        cmap_layout.addWidget(self._gamma_spin, 0, 1)
        cmap_layout.addWidget(self._cmap_combo, 0, 2)
        button_layout.addWidget(self._cmap_r_button, 0, 0)
        button_layout.addWidget(self._cmap_lock_button, 0, 1)
        button_layout.addWidget(self._cmap_mode_button, 0, 2)
        button_layout.addWidget(self._cbar_show_button, 0, 3)
        button_layout.addWidget(colors_button, 0, 4)

        # self._cmap_group.setSizePolicy(QtWidgets.QSizePolicy.Minimum,
        #    QtWidgets.QSizePolicy.Minimum)

        self.layout.addWidget(self._cmap_group)
        self.layout.addWidget(self._button_group)
        # self.layout.addStretch()
        # self.layout.addWidget(style_combo)

    def _cmap_combo_changed(self, text=None):
        if text == "Load all...":
            self._cmap_combo.load_all()
        else:
            self.set_cmap(name=text)

    def set_cmap(self, name=None):
        reverse = self._cmap_r_button.isChecked()
        gamma = self._gamma_spin.value()
        if isinstance(name, str):
            cmap = name
        else:
            cmap = self._cmap_combo.currentText()
        mode = self._cmap_mode_button.isChecked()
        self.itool.set_cmap(cmap, gamma=gamma, reverse=reverse, highContrast=mode)

    def _set_cmap_reverse(self, v):
        if v:
            self._cmap_r_button.setIcon(qta.icon(fonticons["invert_off"]))
        else:
            self._cmap_r_button.setIcon(qta.icon(fonticons["invert"]))
        self.set_cmap()

    def _set_clim_lock(self, v):
        if v:
            self._cmap_lock_button.setIcon(qta.icon(fonticons["lock"]))
        else:
            self._cmap_lock_button.setIcon(qta.icon(fonticons["unlock"]))
        self.itool.set_clim_lock(v)

    def _color_button_clicked(self, s):
        # print("click", s)
        dialog = ImageToolColors(self)
        if dialog.exec():
            # print("Success!")
            pass
        else:
            pass
            # print("Cancel!")


class itoolColorMaps(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPlaceholderText("Select colormap...")
        self.colors = itoolColors
        self.setToolTip("Colormap")
        w, h = 64, 16
        self.setIconSize(QtCore.QSize(w, h))
        for name in pg_colormap_names("mpl"):
            # for name in pg_colormap_names('local'):
            self.addItem(QtGui.QIcon(pg_colormap_to_QPixmap(name, w, h)), name)

    def load_all(self):
        self.clear()
        for name in pg_colormap_names("all"):
            self.addItem(QtGui.QIcon(pg_colormap_to_QPixmap(name)), name)

    # https://forum.qt.io/topic/105012/qcombobox-specify-width-less-than-content/11
    def showPopup(self):
        maxWidth = self.maximumWidth()
        if maxWidth and maxWidth < 16777215:
            self.setPopupMinimumWidthForItems()
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


class itoolSmoothing(QtWidgets.QWidget):
    def __init__(self, itool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itool = itool
        self.ndim = self.itool.data_ndim
        self.layout = FlowLayout(self)
        self._bin_group = QtWidgets.QGroupBox(self)
        self.initialize_widgets()
        self.update_content()
        self.itool.sigDataChanged.connect(self.update_content)

    def initialize_widgets(self):
        bin_layout = QtWidgets.QHBoxLayout(self._bin_group)
        self._spinlabels = tuple(QtWidgets.QLabel(self) for _ in range(self.ndim))
        self._spin = tuple(QtWidgets.QSpinBox(self) for _ in range(self.ndim))
        self._reset = QtWidgets.QPushButton("Reset")
        self._reset.clicked.connect(self._navg_reset)
        for i in range(self.ndim):
            self._spin[i].setSingleStep(2)
            self._spin[i].setValue(1)
            self._spin[i].setWrapping(False)
            self._spin[i].valueChanged.connect(
                lambda n, axis=i: self.itool.set_navg(axis, n)
            )
        for i in range(self.ndim):
            bin_layout.addWidget(self._spinlabels[i])
            bin_layout.addWidget(self._spin[i])
            # bin_layout.addSpacing( , 20)
        bin_layout.addWidget(self._reset)
        # bin_layout.addStretch()
        self.layout.addWidget(self._bin_group)

    def initialize_functions(self):
        # numba overhead
        move_mean_centered_multiaxis(np.zeros((2, 2, 2), dtype=np.float64), [1, 1, 1])

    def _navg_reset(self):
        for i in range(self.ndim):
            self._spin[i].blockSignals(True)
            self._spin[i].setValue(1)
            self._spin[i].blockSignals(False)
        self.itool._refresh_navg(reset=True)

    def update_content(self):
        ndim = self.itool.data_ndim
        if ndim != self.ndim:
            self.layout.clear()
            self.ndim = ndim
            self.initialize_widgets()
        for i in range(self.ndim):
            self._spin[i].blockSignals(True)
            self._spinlabels[i].setText(self.itool.data_dims[i])
            self._spin[i].setRange(1, self.itool.data_shape[i] - 1)
            self._spin[i].setValue(self.itool.avg_win[i])
            self._spin[i].blockSignals(False)
        self.itool._refresh_navg()


class ImageTool(QtWidgets.QMainWindow):
    def __init__(self, data, title=None, *args, **kwargs):
        super().__init__()
        self._main = QtWidgets.QWidget(self)
        self.data = parse_data(data)
        if title is None:
            title = self.data.name
        self.setWindowTitle(title)
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)

        self.data_ndim = self.data.ndim

        self.itool = pg_itool(self.data, *args, **kwargs)

        self.cursortab = itoolCursors(self.itool)
        self.colorstab = itoolColors(self.itool)
        self.smoothtab = itoolSmoothing(self.itool)

        # self.pathtab = QtWidgets.QWidget()
        # pathtabcontent = QtWidgets.QHBoxLayout()
        # pathlabel = QtWidgets.QLabel('Add point: `space`\nRemove point: `delete`\nFinish selection: `enter`')
        # pathstart = QtWidgets.QPushButton()
        # pathstart.clicked.connect(self.itool._drawpath)
        # pathtabcontent.addWidget(pathlabel)
        # pathtabcontent.addWidget(pathstart)
        # self.pathtab.setLayout(pathtabcontent)

        self.tabwidget = QtWidgets.QTabWidget()
        self.tabwidget.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.tabwidget.addTab(self.cursortab, "Cursor")
        self.tabwidget.addTab(self.colorstab, "Appearance")
        self.tabwidget.addTab(self.smoothtab, "Binning")
        # self.tabwidget.currentChanged.connect(self.tab_changed)
        # self.tabwidget.addTab(self.pathtab, 'Path')

        self.layout.addWidget(self.itool)
        self.layout.addWidget(self.tabwidget)
        self.resize(700, 700)
        self.itool.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.itool.setFocus()

    def tab_changed(self, i):
        pass
        # if i == self.tabwidget.indexOf(self.smoothtab):
            # lazy loading
            # self.smoothtab.initialize_functions()


def itool(data, execute=None, *args, **kwargs):

    # TODO: implement multiple windows, add transpose, equal aspect settings
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    change_style("Fusion")

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
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                execute = False
            elif shell == "TerminalInteractiveShell":
                execute = False
        except NameError:
            pass
    if execute:
        qapp.exec()
    return win


if __name__ == "__main__":
    # from pyimagetool import RegularDataArray, imagetool
    # from erlab.plotting import ximagetool
    from arpes.io import load_data

    # dat = xr.open_dataarray('/Users/khan/Documents/ERLab/TiSe2/kxy10.nc')
    dat = xr.open_dataarray(
        "/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc"
    )#.sel(eV=-0.15, method='nearest')
    # dat = xr.open_dataarray('/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy.nc')
    # dat = dat.sel(ky=slice(None, 1.452), eV=slice(-1.281, 0.2), kx=slice(-1.23, None))
    # dat10 = load_data('/Users/khan/Documents/ERLab/TiSe2/data/20211212_00010.fits',
    # location='BL1001-ERLab').spectrum
    # dat16 = load_data('/Users/khan/Documents/ERLab/TiSe2/data/20211212_00016.fits',
    # location='BL1001-ERLab').spectrum
    itool(dat)
    # from erlab.plotting.imagetool_mpl import itoolmpl
    # itoolmpl(dat)
    # gkmk_cvs = load_data('/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/211217 ALS BL4/csvtisb1/f_003.pxt',location="BL4").spectrum
    # itool([gkmk_cvs, dat])
    # itool([dat10, dat16])
    # itool(dat, bench=False)
    # itool(dat.sel(eV=0,method='nearest'), bench=False)s
    # imagetool(dat)
