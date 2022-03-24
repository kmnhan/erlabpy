import sys
import weakref
from itertools import chain, compress

import numpy as np
import numba
import bottleneck as bn
import numbagg
import xarray as xr
import darkdetect
import unittest.mock

from time import perf_counter
from matplotlib import colors
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
# pg.setConfigOption('useNumba', True)
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')



from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PySide6 import QtSvgWidgets, QtSvg, QtWebEngineWidgets


__all__ = ['itool']

suppressnanwarning = np.testing.suppress_warnings()
suppressnanwarning.filter(RuntimeWarning, r'All-NaN (slice|axis) encountered')

import qtawesome as qta      
fonticons = dict(
    invert='mdi6.invert-colors',
    invert_off='mdi6.invert-colors-off',
    contrast='mdi6.contrast-box',
    lock='mdi6.lock',
    unlock='mdi6.lock-open-variant',
    colorbar='mdi6.gradient-vertical',
    transpose=['mdi6.arrow-left-right',
               'mdi6.arrow-top-left-bottom-right',
               'mdi6.arrow-up-down'],
    snap='mdi6.grid',
    snap_off='mdi6.grid-off',
)
        

# import urllib.request
# req = urllib.request.Request('')
# with urllib.request.urlopen(req) as resp:
#     mathjax = resp.read()



from matplotlib import figure, rc_context, rcParams
from matplotlib.backends import backend_agg, backend_svg
import matplotlib.mathtext

# rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
#     "pgf.texsystem": "lualatex",
# })

# 0: html, 1: svg, 2: pixmap
label_mode = 2

def get_pixmap_label(s, prop=None, dpi=300, **text_kw):
    with rc_context({'text.usetex':False}):
        parser = matplotlib.mathtext.MathTextParser('path')
        width, height, depth, _, _ = parser.parse(s, dpi=72, prop=prop)

        fig = figure.Figure(figsize=(width / 72.0, height / 72.0), dpi=dpi)
        fig.patch.set_facecolor('none')
        text_kw['fontproperties'] = prop
        text_kw['fontsize'] = 9
        fig.text(0, depth/height, s, **text_kw)
    
    backend_agg.FigureCanvasAgg(fig)
    buf, size = fig.canvas.print_to_buffer()
    img = QtGui.QImage(buf, size[0], size[1], QtGui.QImage.Format_ARGB32)
    img.setDevicePixelRatio(fig._dpi / 100.0)
    pixmap = QtGui.QPixmap(img.rgbSwapped())
    return pixmap

def get_svg_label(s, prop=None, dpi=300, **text_kw):
    with rc_context({'text.usetex':True}):
        parser = matplotlib.mathtext.MathTextParser('path')
        width, height, depth, _, _ = parser.parse(s, dpi=1000, prop=prop)

        fig = figure.Figure(figsize=(width / 1000.0, height / 1000.0), dpi=dpi)
        fig.patch.set_facecolor('none')
        text_kw['fontproperties'] = prop
        text_kw['fontsize'] = 12
        fig.text(0, depth/height, s, **text_kw)
    
    backend_svg.FigureCanvasSVG(fig)
    file = QtCore.QTemporaryFile()
    if file.open():
        fig.canvas.print_svg(file.fileName())
    return file.fileName()

def mathtextLabelPixmap(self):
    if self.labelUnits == '':
        if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
            units = ''
        else:
            units = '(x%g)' % (1.0/self.autoSIPrefixScale)
    else:
        units = '(%s%s)' % (self.labelUnitPrefix, self.labelUnits)

    s = '%s %s' % (self.labelText, units)

    if label_mode == 1:
        return get_svg_label(s, **self.labelStyle)
    elif label_mode == 0:
        style = ';'.join(['%s: %s' % (k, self.labelStyle[k]) for k in self.labelStyle])
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
    #s = self.size()

    ## Set the position of the label
    if label_mode == 1:
        nudge = -5
    elif label_mode == 0:
        nudge = 0
    else:
        nudge = -5
    if self.label is None: # self.label is set to None on close, but resize events can still occur.
        self.picture = None
        return
        
    br = self.label.boundingRect()
    p = QtCore.QPointF(0, 0)
    if self.orientation == 'left':
        p.setY(int(self.size().height()/2 + br.width()/2))
        p.setX(-nudge)
    elif self.orientation == 'right':
        p.setY(int(self.size().height()/2 + br.width()/2))
        p.setX(int(self.size().width()-br.height()+nudge))
    elif self.orientation == 'top':
        p.setY(-nudge)
        p.setX(int(self.size().width()/2. - br.width()/2.))
    elif self.orientation == 'bottom':
        p.setX(int(self.size().width()/2. - br.width()/2.))
        p.setY(int(self.size().height()-br.height()+nudge))
    self.label.setPos(p)
    self.picture = None
    
def disableMathtextLabels(AxisItem):
    AxisItem.label = AxisItem.label_unpatched
    AxisItem._updateLabel = AxisItem._updateLabel_unpatched
    del AxisItem.label_unpatched
    del AxisItem._updateLabel_unpatched
    del AxisItem.mathtextLabelPixmap

def enableMathtextLabels(item:pg.AxisItem):
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
            if k != 'title':
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
    QtWidgets.QApplication.setStyle(
        QtWidgets.QStyleFactory.create(style_name))

def get_colormap_names(source='all'):
    local = pg.colormap.listMaps()
    if source == 'local':
        return local
    elif source == 'all':
        cet = pg.colormap.listMaps(source='colorcet')
        mpl = pg.colormap.listMaps(source='matplotlib')
        # if (mpl != []) and (cet != []):
            # local = []
        local.sort()
        cet.sort()
        mpl.sort()
        mpl_r = []
        for cmap in mpl:
            if cmap.startswith('cet_'):
                mpl = list(filter((cmap).__ne__, mpl))
            elif cmap.endswith('_r'):
                mpl_r.append(cmap)
                mpl = list(filter((cmap).__ne__, mpl))
        all_cmaps = local + cet + mpl + mpl_r
        return list({value:None for value in all_cmaps})
def get_colormap_from_name(name:str, skipCache=True):
    try:
        return pg.colormap.get(name, skipCache=skipCache)
    except FileNotFoundError:
        try:
            return pg.colormap.get(name, source='matplotlib', skipCache=skipCache)
        except ValueError:
            return pg.colormap.get(name, source='colorcet', skipCache=skipCache)

def get_powernorm_colormap(cmap, gamma, reverse=False,
                           skipCache=True, highContrast=False):
    if isinstance(cmap, str):
        cmap = get_colormap_from_name(cmap, skipCache=skipCache)
    if reverse:
        cmap.reverse()
    N = 4096
    if gamma == 1:
        mapping = np.linspace(0, 1, N)
    elif highContrast and (gamma < 1):
        mapping = 1 - np.power(np.linspace(1, 0, N), 1./gamma)
    else:
        if gamma < 0.4:
            N = 65536
        mapping = np.power(np.linspace(0, 1, N), gamma)
    cmap.color = cmap.mapToFloat(mapping)
    cmap.pos = np.linspace(0, 1, N)
    return cmap

def colormap_to_QPixmap(cmap, w=64, h=16, skipCache=True):
    """Convert pyqtgraph colormap to a `w`-by-`h` QPixmap thumbnail."""
    if isinstance(cmap, str):
        cmap = get_colormap_from_name(cmap, skipCache=skipCache)
    cmap_arr = np.reshape(cmap.getColors()[:, None], (1, -1, 4), order='C')
    img = QtGui.QImage(cmap_arr, cmap_arr.shape[1], 1,
                       QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img).scaled(w, h)

def color_to_QColor(c, alpha=None):
    """Convert matplotlib color to QtGui.Qcolor."""
    return QtGui.QColor.fromRgbF(*colors.to_rgba(c, alpha=alpha))



def move_mean_centered(a, window, min_count=None, axis=-1):
    w = (window - 1) // 2
    shift = w + 1
    if min_count is None:
        min_count = w + 1
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = ((0, shift))
    a = np.pad(a, pad_width, constant_values=np.nan)
    val = bn.move_mean(a, window, min_count=min_count, axis=axis)
    return val[(slice(None),) * (axis % a.ndim) + (slice(w, -1),)]

def move_mean_centered_multiaxis(a, window_list, min_count_list=None):
    w_list = [(window - 1) // 2 for window in window_list]
    pad_width = [(0, 0)] * a.ndim
    slicer = [slice(None),] * a.ndim
    if min_count_list is None:
        min_count_list = [w + 1 for w in w_list]
    for axis in range(a.ndim):
        pad_width[axis] = ((0, w_list[axis] + 1))
        slicer[axis] = slice(w_list[axis], -1)
    a = np.pad(a, pad_width, constant_values=np.nan)
    val = move_mean(a, 
                    numba.typed.List(window_list), 
                    numba.typed.List(min_count_list))
    return val[tuple(slicer)]

def move_mean(a, window, min_count):
    if a.ndim == 3:
        return move_mean3d(a, window, min_count)
    elif a.ndim == 2:
        return move_mean2d(a, window, min_count)
    else:
        raise NotImplementedError

@numba.njit(nogil=True)
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

@numba.njit(nogil=True, parallel=True)
def move_mean2d(a, window_list, min_count_list):
    ii, jj = a.shape
    for i in numba.prange(ii):
        a[i,:] = move_mean1d(a[i,:], window_list[0], min_count_list[0])
    for j in numba.prange(jj):
        a[:,j] = move_mean1d(a[:,j], window_list[1], min_count_list[1])
    return a
@numba.njit(nogil=True, parallel=True)
def move_mean3d(a, window_list, min_count_list):
    ii, jj, kk = a.shape
    for i in numba.prange(ii):
        for k in range(kk):
            a[i,:,k] = move_mean1d(a[i,:,k], window_list[1], min_count_list[1])
    for j in numba.prange(jj):
        for k in range(kk):
            a[:,j,k] = move_mean1d(a[:,j,k], window_list[0], min_count_list[0])
    for i in numba.prange(ii):
        for j in range(jj):
            a[i,j,:] = move_mean1d(a[i,j,:], window_list[2], min_count_list[2])
    return a

def parse_data(data):
    if isinstance(data, xr.Dataset):
        try:
            data = data.spectrum
        except:
            raise TypeError(
                'input argument data must be a xarray.DataArray or a '
                'numpy.ndarray. Create an xarray.DataArray '
                'first, either with indexing on the Dataset or by '
                'invoking the `to_array()` method.'
            ) from None
    elif isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    return data

def get_xy_x(a, b):
    return np.array([[a, 0.], [a, 1.], [b, 1.], [b, 0.], [a, 0.]])

def get_xy_y(a, b):
    return np.array([[0., a], [0., b], [1., b], [1., a], [0., a]])

def get_true_indices(a):
    return list(compress(range(len(a)), a))

# https://www.pythonguis.com/widgets/qcolorbutton-a-color-selector-tool-for-pyqt/
class ColorButton(QtWidgets.QPushButton):
    '''
    Custom Qt Widget to show a chosen color.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    '''

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
                'QWidget { background-color: %s; border: 0; }'
                % self._color.name(QtGui.QColor.HexArgb))
        else:
            self.setStyleSheet('')

    def color(self):
        return self._color

    def onColorPicker(self):
        '''
        Show color-picker dialog to select color.

        Qt will use the native dialog by default.

        '''
        dlg = QtWidgets.QColorDialog(self)
        if self._color:
            dlg.setCurrentColor(QtGui.QColor(self._color))
        if dlg.exec_():
            self.setColor(dlg.currentColor())

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.RightButton:
            self.setColor(self._default)
        return super(ColorButton, self).mousePressEvent(e)

class pg_itool(pg.GraphicsLayoutWidget):
    """A interactive tool based on `pyqtgraph` for exploring 3D data.
    
    For the tool to remain responsive you must 
    keep a reference to it.
    
    Parameters
    ----------
    canvas : `matplotlib.backend_bases.FigureCanvasBase`
        The FigureCanvas that contains all the axes.
    axes : list of `matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to. See Notes for the
        order of axes.
    data : `xarray.DataArray`
        The data to explore. Must have three coordinate axes.
    snap :  bool, default: True
        Snaps cursor to data pixels.
    parallel : bool, default: False
        Use multithreading. Currently has no performance improvement due
        to the python global interpreter lock.
    bench : bool, default: False
        Whether to print frames per second.
    
    Other Parameters
    ----------------
    **self.cursor_kw
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

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

    def __init__(self, data, snap=False, gamma=0.5,
                 cmap='magma', bench=False, plot_kw={}, cursor_kw={},
                 image_kw={}, profile_kw={}, span_kw={}, fermi_kw={},
                 *args, **kwargs):
        super().__init__(show=True, *args, **kwargs)
        self.qapp = QtCore.QCoreApplication.instance()
        self.screen = self.qapp.primaryScreen()
        self.snap = snap
        self.gamma = gamma
        self.cmap = cmap
        self.norm_cmap = get_powernorm_colormap(
            self.cmap, self.gamma,
            reverse=False, skipCache=False, highContrast=False,
        )
        self.bench = bench
        self.colorbar = None
        self.plot_kw = plot_kw
        self.cursor_kw = cursor_kw
        self.image_kw = image_kw
        self.profile_kw = profile_kw
        self.span_kw = span_kw
        self.fermi_kw = fermi_kw

        cursor_c = pg.mkColor(0.5)
        cursor_c.setAlphaF(0.75)
        cursor_c_hover = pg.mkColor(0.75)
        cursor_c_hover.setAlphaF(0.75)
        span_c = pg.mkColor(0.5)
        span_c.setAlphaF(0.15)
        span_c_edge = pg.mkColor(0.5)
        span_c_edge.setAlphaF(0.35)
        # span_c_hover = pg.mkColor(0.75)
        # span_c_hover.setAlphaF(0.5)
        

        self.cursor_kw.update(dict(
            pen=pg.mkPen(cursor_c),
            hoverPen=pg.mkPen(cursor_c_hover),
        ))
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
        self.image_kw.update(dict(
            # colorMap=get_colormap_from_name(self.cmap),
            colorMap=self.norm_cmap,
            autoDownsample=True,
            axisOrder='row-major',
        ))
        self.span_kw.update(dict(
            movable=False,
            pen=pg.mkPen(span_c_edge, width=1),
            brush=pg.mkBrush(span_c),
        ))

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
        
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()
        self.connect_signals()

    def _update_stretch(self, factor=None):
        if factor is None:
            if self.data_ndim == 2:
                factor = [25000, 75000]
            elif self.data_ndim == 3:
                factor = [7500, 35000, 57500]
        for i in range(len(factor)):
            self.ci.layout.setColumnMinimumWidth(i, 0.)
            self.ci.layout.setColumnStretchFactor(i, factor[-i-1])
            self.ci.layout.setRowStretchFactor(i, factor[i])

    def _initialize_layout(self, horiz_pad=45, vert_pad=30, inner_pad=15,
                           font_size=10.):
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        self.ci.layout.setSpacing(inner_pad)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        if self.data_ndim == 2:
            self.axes = [
                self.addPlot(1, 0, 1, 1, **self.plot_kw),
                self.addPlot(0, 0, 1, 1, **self.plot_kw),
                self.addPlot(1, 1, 1, 1, **self.plot_kw),
            ]
            valid_selection = ((1, 0, 0, 1),
                               (1, 1, 0, 0),
                               (0, 0, 1, 1))
        elif self.data_ndim == 3:
            self.axes = [
                self.addPlot(2, 0, 1, 1, **self.plot_kw),
                self.addPlot(0, 0, 1, 1, **self.plot_kw),
                self.addPlot(2, 2, 1, 1, **self.plot_kw),
                self.addPlot(0, 1, 2, 2, **self.plot_kw),
                self.addPlot(1, 0, 1, 1, **self.plot_kw),
                self.addPlot(2, 1, 1, 1, **self.plot_kw),
            ]
            valid_selection = ((1, 0, 0, 1),
                               (1, 1, 0, 0),
                               (0, 0, 1, 1),
                               (0, 1, 1, 0),
                               (1, 0, 0, 0),
                               (0, 0, 0, 1))
        else:
            raise NotImplementedError('Only supports 2D and 3D arrays.')

        for i, (p, sel) in enumerate(zip(self.axes, valid_selection)):
            p.setDefaultPadding(0)
            for axis in ['left', 'bottom', 'right', 'top']:
                p.getAxis(axis).setTickFont(font)
                p.getAxis(axis).setStyle(autoExpandTextSpace=True,
                                         autoReduceTextSpace=True)
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
            self.maps = (
                pg.ImageItem(name='Main Image', **self.image_kw),
            )
            self.hists = (
                self.axes[1].plot(name='X Profile', **self.profile_kw),
                self.axes[2].plot(name='Y Profile', **self.profile_kw),
            )
            self.cursors = (
                (
                    pg.InfiniteLine(angle=90, movable=True, name='X Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(angle=90, movable=True, name='X Cursor',
                                    **self.cursor_kw),
                ),
                (
                    pg.InfiniteLine(angle=0, movable=True, name='Y Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(angle=0, movable=True, name='Y Cursor',
                                    **self.cursor_kw),
                ),
            )
            self.spans = (
                (
                    pg.LinearRegionItem(orientation='vertical',
                                        **self.span_kw),
                    pg.LinearRegionItem(orientation='vertical',
                                        **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation='horizontal',
                                        **self.span_kw),
                    pg.LinearRegionItem(orientation='horizontal',
                                        **self.span_kw),
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
            self.ax_index = (0, 1, 2, 0, 1, 0, 2, 1, 2)
            self.span_ax_index = ((0, 1), (0, 2))
            self._only_axis = ((False, False, True, True, True, False, False),
                               (False, True, False, False, False, True, True))
            self._only_maps = (True, False, False, False, False, False, False)
        elif self.data_ndim == 3:
            self.maps = (
                pg.ImageItem(name='Main Image', **self.image_kw),
                pg.ImageItem(name='Horiz Slice', **self.image_kw),
                pg.ImageItem(name='Vert Slice', **self.image_kw),
            )
            self.hists = (
                self.axes[1].plot(name='X Profile', **self.profile_kw),
                self.axes[2].plot(name='Y Profile', **self.profile_kw),
                self.axes[3].plot(name='Z Profile', **self.profile_kw),
            )
            self.cursors = (
                (
                    pg.InfiniteLine(angle=90, movable=True, name='X Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(angle=90, movable=True, name='X Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(angle=90, movable=True, name='X Cursor',
                                    **self.cursor_kw),
                ),
                (
                    pg.InfiniteLine(angle=0, movable=True, name='Y Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(angle=0, movable=True, name='Y Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(angle=0, movable=True, name='Y Cursor',
                                    **self.cursor_kw),
                ),
                (                    
                    pg.InfiniteLine(angle=90, movable=True, name='Z Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(angle=90, movable=True, name='Z Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(angle=0, movable=True, name='Z Cursor',
                                    **self.cursor_kw),  
                ),
            )
            self.spans = (
                (
                    pg.LinearRegionItem(orientation='vertical',
                                        **self.span_kw),
                    pg.LinearRegionItem(orientation='vertical',
                                        **self.span_kw),
                    pg.LinearRegionItem(orientation='vertical',
                                        **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation='horizontal',
                                        **self.span_kw),
                    pg.LinearRegionItem(orientation='horizontal',
                                        **self.span_kw),
                    pg.LinearRegionItem(orientation='horizontal',
                                        **self.span_kw),
                ),
                (
                    pg.LinearRegionItem(orientation='vertical',
                                        **self.span_kw),
                    pg.LinearRegionItem(orientation='vertical',
                                        **self.span_kw),
                    pg.LinearRegionItem(orientation='horizontal',
                                        **self.span_kw),
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
            self.ax_index = (0, 4, 5, # images
                             1, 2, 3, # profiles
                             0, 1, 4, 0, 2, 5, 3, 5, 4, # cursors
                             1, 2, 3) # axes with dynamic limits
            # self.span_ax_index = ((0, 1, 4), (0, 2, 5), (3, 5, 4))
            self._only_axis = (
                (False, False, True, False, True, True,
                 True, True, True, False, False, False, False, False, False),
                (False, True, False, True, False, True,
                 False, False, False, True, True, True, False, False, False),
                (True, False, False, True, True, False,
                 False, False, False, False, False, False, True, True, True),
            )
            self._only_maps = (
                True, True, True, False, False, False,
                 False, False, False, False, False, False, False, False, False,
            )
        self.all = self.maps + self.hists
        for s in chain.from_iterable(self.spans):
            s.setVisible(False)
        for i in range(len(self.cursors)):
            self.all += self.cursors[i]

    def _get_middle_index(self, x):
        return len(x)//2 - (1 if len(x) % 2 == 0 else 0)

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
            self.clim_list = [()]  * self.data_ndim
            self.avg_win = [1,] * self.data_ndim
            self.averaged = [False, ] * self.data_ndim
        
        # Imagetool properties
        if reset_cursor:
            self.cursor_pos = [None, ] * self.data_ndim
            self._last_ind = [None, ] * self.data_ndim
            self.reset_cursor()
        self.set_labels()
        self._apply_change()
        # if update_all:
        
        self.sigDataChanged.emit(self)

    def toggle_colorbar(self, val):
        if self.colorbar is None:
            self.colorbar = myColorBar(image=self.maps[0], width=20)
            self.addItem(self.colorbar, None, None,
                         self.ci.layout.rowCount(), 1)
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
                c.sigDragged.connect(
                    lambda v, i=axis: self.set_value(i, v.value()))
        self.proxy = pg.SignalProxy(
            self.scene().sigMouseMoved,
            rateLimit=self.screen.refreshRate(),
            slot=self.onmove
        )
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

    def _store_curr_axes(self, evt):
        axis_ind, _ = self._get_curr_axes_index(evt.scenePos())
        if axis_ind == 0:
            self.last_axes = 0
        elif axis_ind == 1:
            self.last_axes = 1
        elif axis_ind == 2:
            self.last_axes = 2
        elif axis_ind == 6:
            self.last_axes = 6
        elif self.data_ndim == 2:
            return
        elif axis_ind == 4:
            self.last_axes = 4
        elif axis_ind == 5:
            self.last_axes = 5
        elif axis_ind == 3:
            self.last_axes = 3
        else:
            return
        

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
            self.axes[1].setTitle('%0.2f fps - %0.1f ms avg' % (fps, average * 1_000))
        
    def labelify(self, text):
        """Prettify some frequently used axis labels."""
        labelformats = dict(
            kx = '$k_x$',
            ky = '$k_y$',
            kz = '$k_z$',
            alpha = '$\\alpha$',
            beta = '$\\beta$',
            theta = '$\\theta$',
            phi = '$\\phi$',
            chi = '$\\chi$',
            eV = '$E$'
        )
        try:
            return labelformats[text]
        except KeyError:
            return text

    def _assign_vals_T(self):
        if self.data_ndim == 2:
            self.data_vals_T = self.data_vals.T
        elif self.data_ndim == 3:
            self.data_vals_T = np.transpose(self.data_vals, axes=(1, 2, 0))
        else:
            raise NotImplementedError('Wrong data dimensions')
    
    def set_cmap(self, cmap=None, gamma=None, reverse=False, highContrast=False):
        if cmap is not self.cmap:
            self.cmap = cmap
        if gamma is not self.gamma:
            self.gamma = gamma
        self.norm_cmap = get_powernorm_colormap(
            self.cmap, self.gamma, reverse=reverse, highContrast=highContrast
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
        if reset:
            for axis in range(self.data_ndim):
                self.averaged[axis] = False
                self.avg_win[axis] = 1
        for axis in range(self.data_ndim):
            for s in self.spans[axis]:
                s.setVisible(self.averaged[axis])
        if not any(self.averaged):
            self.data_vals = self.data.values
        else:
            vals = self.data.values
            self.data_vals = move_mean_centered_multiaxis(vals, self.avg_win)
        self._assign_vals_T()
        self._apply_change()

    def update_spans(self, axis):
        center = self.data_coords[axis][self._last_ind[axis]]
        region = (
            center - self.avg_win[axis] // 2 * self.data_incs[axis],
            center + (self.avg_win[axis] - 1) // 2 * self.data_incs[axis],
        )
        for span in self.spans[axis]:
            span.setRegion(region)


    def get_index_of_value(self, axis, val):
        ind = min(round((val - self.data_lims[axis][0]) / self.data_incs[axis]),
                  self.data_shape[axis] - 1)
        if ind < 0: return 0
        return ind
        # return min(
        #     np.searchsorted(self.data_coords[axis] + 0.5 * self.data_incs[axis], val),
        #     self.data_shape[axis] - 1,
        # )

    def get_key_modifiers(self):
        Qmods = self.qapp.queryKeyboardModifiers()
        mods = []
        if (Qmods & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            mods.append('shift')
        if (Qmods & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            mods.append('control')
        if (Qmods & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier:
            mods.append('alt')
        return mods
    
    def _get_mouse_datapos(self, plot, pos):
        """Returns mouse position in data coords"""
        mouse_point = plot.vb.mapSceneToView(pos)
        return mouse_point.x(), mouse_point.y()

    def onmove(self, evt):
        if self.bench:
            self._t_start = perf_counter()
        if self.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            for c in chain.from_iterable(self.cursors):
                c.setMovable(True)
            return
        else:
            for c in chain.from_iterable(self.cursors):
                c.setMovable(False)
        axis_ind, datapos = self._get_curr_axes_index(evt[0])
        if axis_ind is None:
            return
        x, y, z = None, None, None
        if axis_ind == 0:
            x, y = datapos
        elif axis_ind == 1:
            x = datapos[0]
        elif axis_ind == 2:
            y = datapos[1]
        elif axis_ind == 3:
            z = datapos[0]
        elif axis_ind == 4:
            x, z = datapos
        elif axis_ind == 5:
            z, y = datapos
        elif axis_ind == 6:
            self.colorbar.isoline.setPos(datapos[1])
            return
        else:
            return
        
        dx, dy, dz = x is not None, y is not None, z is not None

        if self.data_ndim == 2:
            cond = (False, dy, dx, dx, dx, dy, dy)
        elif self.data_ndim == 3:
            cond = (dz, dy, dx,
                    dy or dz, dx or dz, dx or dy,
                    dx, dx, dx,
                    dy, dy, dy,
                    dz, dz, dz)
        if dx:
            ind_x = self.get_index_of_value(0, x)
            if self.snap & (ind_x == self._last_ind[0]):
                dx = False
            else:
                self._last_ind[0] = ind_x
        else:
            x = self.cursor_pos[0]
        if dy:
            ind_y = self.get_index_of_value(1, y)
            if self.snap & (ind_y == self._last_ind[1]):
                dy = False
            else:
                self._last_ind[1] = ind_y
        else:
            y = self.cursor_pos[1]
        if dz:
            ind_z = self.get_index_of_value(2, z)
            if self.snap & (ind_z == self._last_ind[2]):
                dz = False
            else:
                self._last_ind[2] = ind_z
        elif self.data_ndim != 2:
            z = self.cursor_pos[2]
        if not any((dx, dy, dz)):
            return
        if not self.snap:
            self.cursor_pos = [x, y, z]
        self._apply_change(cond)
        if self.bench:
            self._measure_fps()
    
    def _apply_change(self, cond=None):
        if cond is None:
            cond = (True,) * len(self.all)
        for i in get_true_indices(cond):
            self._refresh_data(i)
    def transpose_axes(self, axis1, axis2):
        dims_new = list(self.data_dims)
        dims_new[axis1], dims_new[axis2] = self.data_dims[axis2], self.data_dims[axis1]
        data_new = self.data.transpose(*dims_new)
        self._last_ind[axis1], self._last_ind[axis2] = (self._last_ind[axis2],
                                                        self._last_ind[axis1])
        self.cursor_pos[axis1], self.cursor_pos[axis2] = (self.cursor_pos[axis2],
                                                          self.cursor_pos[axis1])
        self.clim_list[axis1], self.clim_list[axis2] = (self.clim_list[axis2],
                                                        self.clim_list[axis1])
        self.avg_win[axis1], self.avg_win[axis2] = (self.avg_win[axis2],
                                                    self.avg_win[axis1])
        self.averaged[axis1], self.averaged[axis2] = (self.averaged[axis2],
                                                      self.averaged[axis1])
        self.set_data(data_new, update_all=False, reset_cursor=False)
    @suppressnanwarning
    def _refresh_data(self, i):
        if self.snap:
            self.cursor_pos = [self.data_coords[i][self._last_ind[i]]
                               for i in range(self.data_ndim)]
        if self.data_ndim == 2:
            self._refresh_data_2d(i)
        elif self.data_ndim == 3:
            self._refresh_data_3d(i)
        self.sigIndexChanged.emit(self._last_ind, self.cursor_pos)
    def _refresh_data_2d(self, i):
        if i == 0:
            if self.clim_locked:
                self.all[i].setImage(self.data_vals_T, levels=self.clim_list[0],
                                     rect=self._lims_to_rect(0, 1))
            else:
                self.all[i].setImage(self.data_vals_T,
                                     rect=self._lims_to_rect(0, 1))
        elif i == 1: 
            self.all[i].setData(
                self.data_coords[0],
                self.data_vals_T[self._last_ind[1],:]
            )
        elif i == 2: 
            self.all[i].setData(
                self.data_vals_T[:,self._last_ind[0]],
                self.data_coords[1]
            )
        elif i in [3, 4]:
            self.all[i].maxRange = self.data_lims[0]
            self.all[i].setPos(self.cursor_pos[0])
            if self.averaged[0]:
                self.update_spans(0)
        elif i in [5, 6]: 
            self.all[i].maxRange = self.data_lims[1]
            self.all[i].setPos(self.cursor_pos[1])
            if self.averaged[1]:
                self.update_spans(1)
    def _refresh_data_3d(self, i):
        if i == 0: 
            if self.clim_locked:
                self.all[i].setImage(self.data_vals_T[:,self._last_ind[2],:],
                                     levels=self.clim_list[i],
                                     rect=self._lims_to_rect(0, 1))
            else:
                self.all[i].setImage(self.data_vals_T[:,self._last_ind[2],:],
                                     rect=self._lims_to_rect(0, 1))
        elif i == 1: 
            if self.clim_locked:
                self.all[i].setImage(self.data_vals_T[self._last_ind[1],:,:],
                                     levels=self.clim_list[i],
                                     rect=self._lims_to_rect(0, 2))
            else:
                self.all[i].setImage(self.data_vals_T[self._last_ind[1],:,:],
                                     rect=self._lims_to_rect(0, 2))
        elif i == 2: 
            if self.clim_locked:
                self.all[i].setImage(self.data_vals_T[:,:,self._last_ind[0]],
                                     levels=self.clim_list[i],
                                     rect=self._lims_to_rect(2, 1))
            else:
                self.all[i].setImage(self.data_vals_T[:,:,self._last_ind[0]],
                                     rect=self._lims_to_rect(2, 1))
        elif i == 3:
            self.all[i].setData(
                self.data_coords[0],
                self.data_vals[:,self._last_ind[1],self._last_ind[2]]
            )
        elif i == 4:
            self.all[i].setData(
                self.data_vals[self._last_ind[0],:,self._last_ind[2]],
                self.data_coords[1]
            )
        elif i == 5:
            self.all[i].setData(
                self.data_coords[2],
                self.data_vals[self._last_ind[0],self._last_ind[1],:]
            )
        elif i in [6, 7, 8]:
            self.all[i].maxRange = self.data_lims[0]
            self.all[i].setPos(self.cursor_pos[0])
            if self.averaged[0]:
                self.update_spans(0)
        elif i in [9, 10, 11]: 
            self.all[i].maxRange = self.data_lims[1]
            self.all[i].setPos(self.cursor_pos[1])
            if self.averaged[1]:
                self.update_spans(1)
        elif i in [12, 13, 14]: 
            self.all[i].maxRange = self.data_lims[2]
            self.all[i].setPos(self.cursor_pos[2])
            if self.averaged[2]:
                self.update_spans(2)

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
        self.setWindowTitle('Colors')

        self.cursor_default = color_to_QColor(
            self.parent.itool.cursor_kw['color'])
        self.line_default = color_to_QColor(
            self.parent.itool.profile_kw['color'])
        self.cursor_current = color_to_QColor(
            self.parent.itool.cursors[0].get_color())
        self.line_current = color_to_QColor(
            self.parent.itool.hists[0].get_color())

        if ((self.cursor_default.getRgbF() == self.cursor_current.getRgbF()) &
            (self.line_default.getRgbF() == self.line_current.getRgbF())):
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok |
                QtWidgets.QDialogButtonBox.Cancel
            )
        else:
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.RestoreDefaults |
                QtWidgets.QDialogButtonBox.Ok |
                QtWidgets.QDialogButtonBox.Cancel
            )
            buttons.button(
                QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
                    self.reset_colors)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        cursorlabel = QtWidgets.QLabel('Cursors:')
        linelabel = QtWidgets.QLabel('Lines:')
        self.cursorpicker = ColorButton(color=self.cursor_current)
        self.cursorpicker.colorChanged.connect(self.parent.itool.set_cursor_color)
        self.linepicker = ColorButton(color=self.line_current)
        self.linepicker.colorChanged.connect(self.parent.itool.set_line_color)
        
        layout = QtWidgets.QGridLayout()
        layout.addWidget(cursorlabel, 0, 0)
        layout.addWidget(self.cursorpicker, 0, 1)
        layout.addWidget(linelabel, 1, 0)
        layout.addWidget(self.linepicker, 1, 1)
        layout.addWidget(buttons)
        self.setLayout(layout)
    
    def reject(self):
        self.cursorpicker.setColor(self.cursor_current)
        self.linepicker.setColor(self.line_current)
        super().reject()

    def reset_colors(self):
        self.cursorpicker.setColor(self.cursor_default)
        self.linepicker.setColor(self.line_default)



@numba.njit(nogil=True)
def fast_isocurve_extend(data):
    d2 = np.empty((data.shape[0]+2, data.shape[1]+2), dtype=data.dtype)
    d2[1:-1, 1:-1] = data
    d2[0, 1:-1] = data[0]
    d2[-1, 1:-1] = data[-1]
    d2[1:-1, 0] = data[:, 0]
    d2[1:-1, -1] = data[:, -1]
    d2[0,0] = d2[0,1]
    d2[0,-1] = d2[1,-1]
    d2[-1,0] = d2[-1,1]
    d2[-1,-1] = d2[-1,-2]
    return d2

@numba.njit(nogil=True)
def fast_isocurve_lines(data, level, index, extendToEdge=False):
    sideTable = ([np.int64(x) for x in range(0)], [0, 1], [1, 2], [0, 2],
                 [0, 3], [1, 3], [0, 1, 2, 3], [2, 3], [2, 3], [0, 1, 2, 3],
                 [1, 3], [0, 3], [0, 2], [1, 2], [0, 1],
                 [np.int64(x) for x in range(0)])
    edgeKey = [[(0, 1), (0, 0)],
               [(0, 0), (1, 0)],
               [(1, 0), (1, 1)],
               [(1, 1), (0, 1)]]
    lines = []
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            sides = sideTable[index[i, j]]
            for l in range(0, len(sides), 2):
                edges = sides[l:l+2]
                pts = []
                for m in range(2):
                    p1, p2 = edgeKey[edges[m]][0], edgeKey[edges[m]][1]
                    v1, v2 = data[i+p1[0], j+p1[1]], data[i+p2[0], j+p2[1]]
                    f = (level - v1) / (v2 - v1)
                    fi = 1.0 - f
                    p = (p1[0] * fi + p2[0] * f + i + 0.5, 
                         p1[1] * fi + p2[1] * f + j + 0.5)
                    if extendToEdge:
                        p = (min(data.shape[0] - 2, max(0, p[0] - 1)),
                             min(data.shape[1] - 2, max(0, p[1] - 1)))
                    pts.append(p)
                lines.append(pts)
    return lines

@numba.njit(nogil=True)
def fast_isocurve_lines_connected(data, level, index, extendToEdge=False):
    sideTable = ([np.int64(x) for x in range(0)], [0, 1], [1, 2], [0, 2],
                 [0, 3], [1, 3], [0, 1, 2, 3], [2, 3], [2, 3], [0, 1, 2, 3],
                 [1, 3], [0, 3], [0, 2], [1, 2], [0, 1],
                 [np.int64(x) for x in range(0)])
    edgeKey = [[(0, 1), (0, 0)],
               [(0, 0), (1, 0)],
               [(1, 0), (1, 1)],
               [(1, 1), (0, 1)]]
    lines = []
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            sides = sideTable[index[i, j]]
            for l in range(0, len(sides), 2):
                edges = sides[l:l+2]
                pts = []
                for m in range(2):
                    p1, p2 = edgeKey[edges[m]][0], edgeKey[edges[m]][1]
                    v1, v2 = data[i+p1[0], j+p1[1]], data[i+p2[0], j+p2[1]]
                    f = (level - v1) / (v2 - v1)
                    fi = 1.0 - f
                    p = (p1[0] * fi + p2[0] * f + i + 0.5, 
                         p1[1] * fi + p2[1] * f + j + 0.5)
                    if extendToEdge:
                        p = (min(data.shape[0] - 2, max(0, p[0] - 1)),
                             min(data.shape[1] - 2, max(0, p[1] - 1)))
                    gridKey = (i + (1 if edges[m] == 2 else 0),
                                j + (1 if edges[m] == 3 else 0),
                                edges[m] % 2)
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
    index = np.zeros([x-1 for x in mask.shape], dtype=np.int64)
    fields = np.empty((2, 2), dtype=object)
    slices = [slice(0,-1), slice(1,None)]
    for i in range(2):
        for j in range(2):
            fields[i,j] = mask[slices[i], slices[j]]
            vertIndex = i+2*j
            index += fields[i,j] * 2**vertIndex
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
        return lines ## a list of pairs of points
    
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
    def __init__(self, data=None, level=0, pen='w', axisOrder=None,
                 connected=False, extendToEdge=False):
        super().__init__(data, level, pen, axisOrder)
        self.connected = connected
        self.extendToEdge = extendToEdge
    
    def generatePath(self):
        if self.data is None:
            self.path = None
            return
        
        if self.axisOrder == 'row-major':
            data = self.data.T
        else:
            data = self.data
        
        lines = fast_isocurve(data, self.level,
                              self.connected, self.extendToEdge)
        # lines = pg.functions.isocurve(data, self.level, connected=True, extendToEdge=True)
        self.path = QtGui.QPainterPath()
        for line in lines:
            self.path.moveTo(*line[0])
            for p in line[1:]:
                self.path.lineTo(*p)

class myColorBar(pg.PlotItem):
    
    def __init__(self, image, width=25, horiz_pad=45, vert_pad=30, inner_pad=5,     
                 font_size=10, curve_kw={}, line_kw={'pen':'cyan'},
                 *args, **kwargs):
        super(myColorBar, self).__init__(*args, **kwargs)
        self.setDefaultPadding(0)
        self.cbar = pg.ImageItem(axisOrder='row-major')
        self.npts = 4096
        self.autolevels = True

        self.cbar.setImage(np.linspace(0, 1, self.npts).reshape((-1, 1)))
        self.addItem(self.cbar)
        
        self.isocurve = betterIsocurve(**curve_kw)
        self.isocurve.setZValue(5)

        self.isoline = pg.InfiniteLine(angle=0, movable=True, **line_kw)
        self.addItem(self.isoline)
        self.isoline.setZValue(1000)
        self.isoline.sigPositionChanged.connect(self.update_level)

        self.setImageItem(image)
        self.setMouseEnabled(x=False, y=True)
        self.setMenuEnabled(True)
        
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        for axis in ['left', 'bottom', 'right', 'top']:
            self.getAxis(axis).setTickFont(font)
        self.layout.setColumnFixedWidth(1, width)
        self.layout.setSpacing(inner_pad)
        self.layout.setContentsMargins(0., 0., 0., 0.)
        self.showAxes((True, True, True, True),
                      showValues=(False, False, True, False),
                      size=(horiz_pad, 0.))
        # self.getAxis('right').setStyle(
            # showValues=True, tickTextWidth=horiz_pad,
            # autoExpandTextSpace=False, autoReduceTextSpace=False)
        self.getAxis('top').setHeight(vert_pad)   
        self.getAxis('bottom').setHeight(vert_pad)
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
            self.cbar.setRect(0., levels[0], 1., levels[1] - levels[0])
        else:
            mn, mx = self.imageItem().quickMinMax(targetSize=2**16)
            self.cbar.setRect(0., mn, 1., mx - mn)
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
        self.layout = QtWidgets.QHBoxLayout(self)
        self.initialize_widgets()
        self.update_content()
        self.itool.sigIndexChanged.connect(self.update_spin)
        self.itool.sigDataChanged.connect(self.update_content)

    def initialize_widgets(self):
        self._spinlabels = tuple(QtWidgets.QLabel(self)
                                 for _ in range(self.ndim))
        self._spin = tuple(QtWidgets.QSpinBox(self)
                                  for _ in range(self.ndim))
        self._dblspin = tuple(QtWidgets.QDoubleSpinBox(self)
                                     for _ in range(self.ndim))
        self._transpose_button = tuple(QtWidgets.QPushButton(self)
                                       for _ in range(self.ndim))
        self._snap_button = QtWidgets.QPushButton(self)
        self._snap_button.setCheckable(True)
        self._snap_button.toggled.connect(self._assign_snap)
        self._snap_button.setIcon(qta.icon(fonticons['snap']))
        self._snap_button.setToolTip('Snap cursor to data')
        for i in range(self.ndim):
            self._spin[i].setSingleStep(1)
            self._spin[i].setWrapping(True)
            self._dblspin[i].setDecimals(3)
            self._dblspin[i].setWrapping(True)
            self._spin[i].valueChanged.connect(
                lambda v, axis=i: self._index_changed(axis, v))
            self._dblspin[i].valueChanged.connect(
                lambda v, axis=i: self._value_changed(axis, v))
            self._transpose_button[i].clicked.connect(
                lambda axis1=i, axis2=i-1: self.itool.transpose_axes(axis1, axis2))
            self.layout.addWidget(self._spinlabels[i])
            self.layout.addWidget(self._spin[i])
            self.layout.addWidget(self._dblspin[i])
            self.layout.addSpacing(5)

        self.layout.addStretch()
        for i, button in enumerate(self._transpose_button):
            self.layout.addWidget(button)
            button.setIcon(qta.icon(fonticons['transpose'][i]))
        self.layout.addStretch()
        self.layout.addWidget(self._snap_button)
    
    def update_content(self):
        ndim = self.itool.data_ndim
        if ndim != self.ndim:
            self.layout.clear()
            self.ndim = ndim
            self.initialize_widgets()
        self._snap_button.setChecked(self.itool.snap)
        
        for i in range(self.ndim):
            self._spinlabels[i].setText(self.itool.data_dims[i])
            self._spin[i].blockSignals(True)
            self._dblspin[i].blockSignals(True)
            self._spin[i].setRange(0, self.itool.data_shape[i] - 1)
            self._spin[i].setValue(self.itool._last_ind[i])
            self._dblspin[i].setRange(*self.itool.data_lims[i])
            self._dblspin[i].setSingleStep(self.itool.data_incs[i])
            self._dblspin[i].setValue(
                self.itool.data_coords[i][self.itool._last_ind[i]])
            self._spin[i].blockSignals(False)
            self._dblspin[i].blockSignals(False)

    def _assign_snap(self, value):
        if value:
            self._snap_button.setIcon(qta.icon(fonticons['snap_off']))
        else:
            self._snap_button.setIcon(qta.icon(fonticons['snap']))
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
        self.layout = QtWidgets.QHBoxLayout(self)
        self.initialize_widgets()
    
    def initialize_widgets(self):
        self._gamma_spin = QtWidgets.QDoubleSpinBox(self)
        self._gamma_spin.setToolTip('Colormap gamma')
        self._gamma_spin.setSingleStep(0.01)
        self._gamma_spin.setRange(0.01, 100.)
        self._gamma_spin.setValue(self.itool.gamma)
        self._gamma_spin.valueChanged.connect(self.set_cmap)
        gamma_label = QtWidgets.QLabel('g')
        gamma_label.setBuddy(self._gamma_spin)

        self._cmap_combo = itoolColorMaps(self)
        self._cmap_combo.setMaximumWidth(175)
        if isinstance(self.itool.cmap, str):
            self._cmap_combo.setCurrentText(self.itool.cmap)
        self._cmap_combo.insertItem(0, 'Load all...')
        self._cmap_combo.textActivated.connect(self._cmap_combo_changed)

        self._cmap_r_button = QtWidgets.QPushButton(self)
        self._cmap_r_button.setCheckable(True)
        self._cmap_r_button.toggled.connect(self._set_cmap_reverse)
        self._cmap_r_button.setIcon(qta.icon(fonticons['invert']))
        self._cmap_r_button.setToolTip('Invert colormap')

        self._cmap_mode_button = QtWidgets.QPushButton(self)
        self._cmap_mode_button.setCheckable(True)
        self._cmap_mode_button.toggled.connect(self.set_cmap)
        self._cmap_mode_button.setIcon(qta.icon(fonticons['contrast']))
        self._cmap_mode_button.setToolTip('High contrast mode')

        self._cmap_lock_button = QtWidgets.QPushButton(self)
        self._cmap_lock_button.setCheckable(True)
        self._cmap_lock_button.toggled.connect(self._set_clim_lock)
        self._cmap_lock_button.setIcon(qta.icon(fonticons['unlock']))
        self._cmap_lock_button.setToolTip('Lock colors')

        self._cbar_show_button = QtWidgets.QPushButton(self)
        self._cbar_show_button.setCheckable(True)
        self._cbar_show_button.toggled.connect(self.itool.toggle_colorbar)
        self._cbar_show_button.setIcon(qta.icon(fonticons['colorbar']))
        self._cbar_show_button.setToolTip('Show colorbar')

        colors_button = QtWidgets.QPushButton('Colors', parent=self)
        colors_button.clicked.connect(self._color_button_clicked)
        style_label = QtWidgets.QLabel('Style:', parent=self)
        style_combo = QtWidgets.QComboBox(self)
        style_combo.setToolTip('Qt style')
        style_combo.addItems(qt_style_names())
        style_combo.textActivated.connect(change_style)
        style_combo.setCurrentText('Fusion')
        style_label.setBuddy(style_combo)
        self.layout.addWidget(gamma_label)
        self.layout.addWidget(self._gamma_spin)
        self.layout.addWidget(self._cmap_combo)
        self.layout.addWidget(self._cmap_r_button)
        self.layout.addWidget(self._cmap_lock_button)
        self.layout.addWidget(self._cmap_mode_button)
        self.layout.addWidget(self._cbar_show_button)
        self.layout.addStretch()
        self.layout.addWidget(colors_button)
        self.layout.addStretch()
        self.layout.addWidget(style_label)
        self.layout.addWidget(style_combo)

    def _cmap_combo_changed(self, text=None):
        if text == 'Load all...':
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
        self.itool.set_cmap(cmap, gamma=gamma,
                            reverse=reverse, highContrast=mode)
    
    def _set_cmap_reverse(self, v):
        if v:
            self._cmap_r_button.setIcon(qta.icon(fonticons['invert_off']))
        else:
            self._cmap_r_button.setIcon(qta.icon(fonticons['invert']))
        self.set_cmap()

    def _set_clim_lock(self, v):
        if v:
            self._cmap_lock_button.setIcon(qta.icon(fonticons['lock']))
        else:
            self._cmap_lock_button.setIcon(qta.icon(fonticons['unlock']))
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
        self.setPlaceholderText('Select colormap...')
        self.colors = itoolColors
        self.setToolTip('Colormap')
        w, h = 64, 16
        self.setIconSize(QtCore.QSize(w, h))
        for name in get_colormap_names('local'):
            self.addItem(QtGui.QIcon(colormap_to_QPixmap(name, w, h)), name)

    def load_all(self):
        self.clear()
        for name in get_colormap_names('all'):
            self.addItem(QtGui.QIcon(colormap_to_QPixmap(name)), name)
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

class itoolBinning(QtWidgets.QWidget):
    def __init__(self, itool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itool = itool
        self.ndim = self.itool.data_ndim
        self.layout = QtWidgets.QHBoxLayout(self)
        self.initialize_widgets()
        self.update_content()
        self.itool.sigDataChanged.connect(self.update_content)
    
    def initialize_widgets(self):
        self._spinlabels = tuple(QtWidgets.QLabel(self)
                                 for _ in range(self.ndim))
        self._spin = tuple(QtWidgets.QSpinBox(self)
                            for _ in range(self.ndim))
        self._reset = QtWidgets.QPushButton('Reset')
        self._reset.clicked.connect(self._navg_reset)
        for i in range(self.ndim):
            self._spin[i].setRange(1, self.itool.data_shape[i] - 1)
            self._spin[i].setSingleStep(2)
            self._spin[i].setValue(1)
            self._spin[i].setWrapping(False)
            self._spin[i].valueChanged.connect(
                lambda n, axis=i: self.itool.set_navg(axis, n))
        for i in range(self.ndim):
            self.layout.addWidget(self._spinlabels[i])
            self.layout.addWidget(self._spin[i])
            self.layout.addSpacing(20)
        self.layout.addWidget(self._reset)
        self.layout.addStretch()

    def initialize_functions(self):
        # numba overhead
        move_mean_centered_multiaxis(
            np.zeros((2,2,2), dtype=np.float64), [1,1,1])

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
            self._spin[i].setValue(self.itool.avg_win[i])
            self._spin[i].blockSignals(False)
        self.itool._refresh_navg()
    
class ImageTool(QtWidgets.QMainWindow):
    def __init__(self, data, *args, **kwargs):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)
        self.data = parse_data(data)
        self.data_ndim = self.data.ndim

        self.itool = pg_itool(self.data, *args, **kwargs)

        self.cursortab = itoolCursors(self.itool)
        self.colorstab = itoolColors(self.itool)
        self.smoothtab = itoolBinning(self.itool)

        # self.pathtab = QtWidgets.QWidget()
        # pathtabcontent = QtWidgets.QHBoxLayout()
        # pathlabel = QtWidgets.QLabel('Add point: `space`\nRemove point: `delete`\nFinish selection: `enter`')
        # pathstart = QtWidgets.QPushButton()
        # pathstart.clicked.connect(self.itool._drawpath)
        # pathtabcontent.addWidget(pathlabel)
        # pathtabcontent.addWidget(pathstart)
        # self.pathtab.setLayout(pathtabcontent)

        self.tabwidget = QtWidgets.QTabWidget()
        self.tabwidget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                           QtWidgets.QSizePolicy.Maximum)
        self.tabwidget.addTab(self.cursortab, 'Cursor')
        self.tabwidget.addTab(self.colorstab, 'Appearance')
        self.tabwidget.addTab(self.smoothtab, 'Smoothing')
        self.tabwidget.currentChanged.connect(self.tab_changed)
        # self.tabwidget.addTab(self.pathtab, 'Path')
        
        self.layout.addWidget(self.itool)
        self.layout.addWidget(self.tabwidget)
        self.resize(700, 700)
        self.itool.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.itool.setFocus()

    def tab_changed(self, i):
        if i == self.tabwidget.indexOf(self.smoothtab):
            # lazy loading
            self.smoothtab.initialize_functions()


def itool(data, *args, **kwargs):
    # TODO: implement multiple windows, add transpose, equal aspect settings
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    # if darkdetect.isDark():
        # pass
    app = ImageTool(data, *args, **kwargs)
    change_style('Fusion')
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()

if __name__ == "__main__":
    # from pyimagetool import RegularDataArray, imagetool
    # from erlab.plotting import ximagetool
    # dat = xr.open_dataarray('/Users/khan/Documents/ERLab/TiSe2/kxy09.nc')
    dat = xr.open_dataarray('/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy_small.nc')
    # dat = xr.open_dataarray('/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/Data/cvs_kxy.nc')
    dat = dat.sel(ky=slice(None, 1.452), eV=slice(-1.281, 0.2), kx=slice(-1.23, None))
    itool(dat)
    # from erlab.plotting.imagetool_mpl import itoolmpl
    # itoolmpl(dat)
    # from arpes.io import load_data
    # gkmk_cvs = load_data('/Users/khan/Documents/ERLab/CsV3Sb5/2021_Dec_ALS_CV3Sb5/211217 ALS BL4/csvtisb1/f_003.pxt',location="BL4").spectrum
    # itool(gkmk_cvs)
    # itool(dat, bench=False)
    # itool(dat.sel(eV=0,method='nearest'), bench=False)
    # imagetool(dat)
    