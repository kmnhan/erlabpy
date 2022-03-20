import sys
from typing import Type
import weakref
from itertools import compress

import numpy as np
import bottleneck as bn
# import numbagg
import xarray as xr
import darkdetect
import qtawesome as qta
import matplotlib.pyplot as plt
from time import perf_counter
from matplotlib import colors
from scipy.interpolate import interp1d, interp2d
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
# pg.setConfigOption('useNumba', True)
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

__all__ = ['itool']

supnan = np.testing.suppress_warnings()
supnan.filter(RuntimeWarning, r'All-NaN (slice|axis) encountered')

def get_all_colormaps():
    local = pg.colormap.listMaps()
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

def get_colormap_from_name(name:str, skipCache=True):
    try:
        return pg.colormap.get(name, skipCache=skipCache)
    except FileNotFoundError:
        try:
            return pg.colormap.get(name, source='matplotlib', skipCache=skipCache)
        except ValueError:
            return pg.colormap.get(name, source='colorcet', skipCache=skipCache)

def get_powernorm_colormap(cmap, gamma, reverse=False, skipCache=True, mode=0):
    if isinstance(cmap, str):
        cmap = get_colormap_from_name(cmap, skipCache=skipCache)
    if reverse:
        cmap.reverse()
    if gamma == 1:
        return cmap
    if mode and (gamma < 1):
        mapping = 1 - np.power(np.linspace(1, 0, 4096), 1./gamma)
    else:
        mapping = np.power(np.linspace(0, 1, 4096), gamma)
    cmap.color = cmap.mapToFloat(mapping)
    cmap.pos = np.linspace(0.0, 1.0, num=4096)
    cmap.pos = np.linspace(0.0, 1.0, num=4096)
    return cmap

def colormap_to_QPixmap(name:str, w=64, h=16, skipCache=True):
    """Convert pyqtgraph colormap to a `w`-by-`h` QPixmap thumbnail."""
    # cmap = plt.colormaps[name]
    cmap = get_colormap_from_name(name, skipCache=skipCache)
    # cmap_arr = cmap(np.tile(np.linspace(0, 1, 256), (h, 1))) * 255
    cmap_arr = np.tile(cmap.getLookupTable(0, 1, w, alpha=True), (h, 1, 1))
    img = QtGui.QImage(cmap_arr.astype(np.uint8).data,
                       cmap_arr.shape[1], cmap_arr.shape[0],
                       QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img)

def color_to_QColor(c, alpha=None):
    """Convert matplotlib color to QtGui.Qcolor."""
    return QtGui.QColor.fromRgbF(*colors.to_rgba(c, alpha=alpha))

class cmapComboBox(QtWidgets.QComboBox):
    def __init__(self, *args, **kwargs):
        super(cmapComboBox, self).__init__(*args, **kwargs)
        self.setToolTip('Colormap')
        for name in get_all_colormaps():
            self.addItem(QtGui.QIcon(colormap_to_QPixmap(name)), name)
        self.setIconSize(QtCore.QSize(64, 16))

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

def move_mean_centered_multiaxis(a, window_list,
                                 min_count_list=None, axis_list=[-1, ]):
    if not isinstance(axis_list, list):
        axis_list = [axis_list, ]
    w_list = [(window - 1) // 2 for window in window_list]
    pad_width = [(0, 0)] * a.ndim
    slicer = [slice(None),] * a.ndim
    if min_count_list is None:
        min_count_list = [w + 1 for w in w_list]
    for axis in axis_list:
        pad_width[axis] = ((0, w_list[axis] + 1))
        slicer[axis] = slice(w_list[axis], -1)
    a = np.pad(a, pad_width, constant_values=np.nan)
    val = _move_mean_multiaxis_calc(a, window_list, min_count_list, axis_list)
    return val[tuple(slicer)]
def _move_mean_multiaxis_calc(a_padded, window_list,
                              min_count_list, axis_list):
    val = a_padded
    for axis in axis_list:
        val = bn.move_mean(val, window_list[axis],
                           min_count=min_count_list[axis], axis=axis)
    return val

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

def is_vertical(artist):
    return artist.axes.get_xaxis_transform() == artist.get_transform()

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
    """

    def __init__(self, data, snap=False, gamma=0.5,
                 cmap='magma', bench=False, plot_kw={}, cursor_kw={},
                 image_kw={}, profile_kw={}, span_kw={}, fermi_kw={},
                 *args, **kwargs):
        super(pg_itool, self).__init__(show=True, *args, **kwargs)
        self.qapp = QtCore.QCoreApplication.instance()
        self.screen = self.qapp.primaryScreen()

        self.snap = snap
        self.gamma = gamma
        self.cmap = cmap
        self.norm_cmap = get_powernorm_colormap(self.cmap, self.gamma)
        self.bench = bench
        self.plot_kw = plot_kw
        self.cursor_kw = cursor_kw
        self.image_kw = image_kw
        self.profile_kw = profile_kw
        self.span_kw = span_kw
        self.fermi_kw = fermi_kw

        self.cursor_kw.update(dict(
            pen=pg.mkPen(0.5, alpha=0.5),
            hoverPen=pg.mkPen(0.75, alpha=0.5),
        ))
        # self.plot_kw.update(dict(defaultPadding=0.01, clipToView=True))
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
            # autoDownsample=True,
            axisOrder='row-major',
        ))
        # self.span_kw.update(dict(
        #     # edgecolor=plt.rcParams.get('axes.edgecolor'),
        #     # lw=0.5, ls='--',
        #     facecolor=colors.to_rgba(self.cursor_kw['color'], alpha=1),
        #     alpha=0.15,
        #     animated=self.useblit, visible=True,
        # ))


        self.ndim = None
        
        # self.vals = None
        # self.vals_T = None
        # self.dims = None
        # self.coords = None
        # self.shape = None
        # self.incs = None
        # self.lims = None
        # self.cursor_pos = None
        self.set_data(data, update_all=True)
        
        self.need_redraw = False
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()
        self.connect_signals()

    def _update_stretch(self, factor=None):
        self.ci.layout.setSpacing(0.)
        self.ci.layout.setContentsMargins(0, 0, 0, 0)
        if factor is None:
            if self.ndim == 2:
                factor = [25000, 75000]
            elif self.ndim == 3:
                factor = [7500, 35000, 57500]
        for i in range(len(factor)):
            self.ci.layout.setColumnMinimumWidth(i, 0.)
            self.ci.layout.setColumnStretchFactor(i, factor[-i-1])
            self.ci.layout.setRowStretchFactor(i, factor[i])

    def _initialize_layout(self, horiz_pad=45, vert_pad=30, inner_pad=5,
                           font_size=10.):
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        if self.ndim == 2:
            self.axes = [
                self.addPlot(1, 0, 1, 1, **self.plot_kw),
                self.addPlot(0, 0, 1, 1, **self.plot_kw),
                self.addPlot(1, 1, 1, 1, **self.plot_kw),
            ]
        elif self.ndim == 3:
            self.axes = [
                self.addPlot(2, 0, 1, 1, **self.plot_kw),
                self.addPlot(0, 0, 1, 1, **self.plot_kw),
                self.addPlot(2, 2, 1, 1, **self.plot_kw),
                self.addPlot(0, 1, 2, 2, **self.plot_kw),
                self.addPlot(1, 0, 1, 1, **self.plot_kw),
                self.addPlot(2, 1, 1, 1, **self.plot_kw),
            ]
        else:
            raise NotImplementedError('Only supports 2D and 3D arrays.')
        for i, p in enumerate(self.axes):
            for axis in ['left', 'bottom', 'right', 'top']:
                p.getAxis(axis).setTickFont(font)
                p.getAxis(axis).setStyle(
                    autoExpandTextSpace=False,
                    autoReduceTextSpace=False,
                )
            p.showAxes((True, True, True, True),
                       showValues=True, size=(inner_pad, inner_pad))
            if i in [1, 4]:
                self.axes[i].setXLink(self.axes[0])
                self.axes[i].getAxis('bottom').setStyle(showValues=False)
                self.axes[i].getAxis('left').setWidth(horiz_pad)
            elif i in [2, 5]:
                self.axes[i].setYLink(self.axes[0])
                self.axes[i].getAxis('left').setStyle(showValues=False)
                self.axes[i].getAxis('bottom').setHeight(vert_pad)
            elif i == 3:        
                self.axes[i].getAxis('bottom').setStyle(showValues=False)
                self.axes[i].getAxis('left').setStyle(showValues=False)
                self.axes[i].getAxis('top').setStyle(showValues=True)
                self.axes[i].getAxis('right').setStyle(showValues=True)
                self.axes[i].getAxis('right').setWidth(horiz_pad)
                self.axes[i].getAxis('top').setHeight(vert_pad)
            else: # i == 0
                self.axes[i].getAxis('bottom').setHeight(vert_pad)
                self.axes[i].getAxis('left').setWidth(horiz_pad)
        self.axes[1].getAxis('top').setStyle(showValues=True)
        self.axes[2].getAxis('right').setStyle(showValues=True)
        self.axes[1].getAxis('top').setHeight(vert_pad)
        self.axes[2].getAxis('right').setWidth(horiz_pad)
        pg.ViewBox.suggestPadding = lambda *_: 0.
        self._update_stretch()
    
    def _lims_to_rect(self, i, j):
        x = self.lims[i][0]
        y = self.lims[j][0]
        w = self.lims[i][-1] - x
        h = self.lims[j][-1] - y
        return x, y, w, h
    
    def _initialize_plots(self):
        if self.ndim == 2:
            self.maps = (
                pg.ImageItem(image=self.vals_T,
                             rect=self._lims_to_rect(0, 1),
                             name='Main Image', **self.image_kw),
            )
            self.hists = (
                self.axes[1].plot(self.coords[0], self.vals[:,mids[1]],
                                  name='X Profile', **self.profile_kw),
                self.axes[2].plot(self.vals[mids[0],:], self.coords[1],
                                  name='Y Profile', **self.profile_kw),
            )
            self.cursors = (
                (
                    pg.InfiniteLine(self.coords[0][mids[0]], angle=90,
                                    bounds=self.lims[0], movable=True,
                                    name='X Cursor', **self.cursor_kw),
                    pg.InfiniteLine(self.coords[0][mids[0]], angle=90,
                                    bounds=self.lims[0], movable=True,
                                    name='X Cursor', **self.cursor_kw),
                ),
                (
                    pg.InfiniteLine(self.coords[1][mids[1]], angle=0,
                                    movable=True, name='Y Cursor',
                                    **self.cursor_kw),
                    pg.InfiniteLine(self.coords[1][mids[1]], angle=0,
                                    movable=True, name='Y Cursor',
                                    **self.cursor_kw),
                ),
            )
            self.axes[0].addItem(self.maps[0])
            self.axes[0].addItem(self.cursors[0][0])
            self.axes[1].addItem(self.cursors[0][1])
            self.axes[0].addItem(self.cursors[1][0])
            self.axes[2].addItem(self.cursors[1][1])
            # self.spans = (
            #     (
            #         self.axes[0].axvspan(
            #             self.coords[0][self._last_ind[0]],
            #             self.coords[0][self._last_ind[0]],
            #             label='X Span', **self.span_kw),
            #         self.axes[1].axvspan(
            #             self.coords[0][self._last_ind[0]],
            #             self.coords[0][self._last_ind[0]],
            #             label='X Span', **self.span_kw),
            #     ),
            #     (
            #         self.axes[0].axhspan(
            #             self.coords[1][self._last_ind[1]],
            #             self.coords[1][self._last_ind[1]],
            #             label='Y Span', **self.span_kw),
            #         self.axes[2].axhspan(
            #             self.coords[1][self._last_ind[1]],
            #             self.coords[1][self._last_ind[1]],
            #             label='Y Span', **self.span_kw),
            #     ),
            # )
            self.ax_index = (0, 1, 2, 0, 1, 0, 2, 1, 2)
            self.span_ax_index = ((0, 1), (0, 2))
            self._only_axis = (
                (False, False, True, True, True, False, False),
                (False, True, False, False, False, True, True),
            )
            self._only_maps = (
                True, False, False, False, False, False, False,
            )
        elif self.ndim == 3:
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
            # self.spans = (
            #     (
            #         self.axes[0].axvspan(
            #             self.coords[0][self._last_ind[0]],
            #             self.coords[0][self._last_ind[0]],
            #             label='X Span', **self.span_kw),
            #         self.axes[1].axvspan(
            #             self.coords[0][self._last_ind[0]],
            #             self.coords[0][self._last_ind[0]],
            #             label='X Span', **self.span_kw),
            #         self.axes[4].axvspan(
            #             self.coords[0][self._last_ind[0]],
            #             self.coords[0][self._last_ind[0]],
            #             label='X Span', **self.span_kw),
            #     ),
            #     (
            #         self.axes[0].axhspan(
            #             self.coords[1][self._last_ind[1]],
            #             self.coords[1][self._last_ind[1]],
            #             label='Y Span', **self.span_kw),
            #         self.axes[2].axhspan(
            #             self.coords[1][self._last_ind[1]],
            #             self.coords[1][self._last_ind[1]],
            #             label='Y Span', **self.span_kw),
            #         self.axes[5].axhspan(
            #             self.coords[1][self._last_ind[1]],
            #             self.coords[1][self._last_ind[1]],
            #             label='Y Span', **self.span_kw),
            #     ),
            #     (
            #         self.axes[3].axvspan(
            #             self.coords[2][self._last_ind[2]],
            #             self.coords[2][self._last_ind[2]],
            #             label='Z Span', **self.span_kw),
            #         self.axes[5].axvspan(
            #             self.coords[2][self._last_ind[2]],
            #             self.coords[2][self._last_ind[2]],
            #             label='Z Span', **self.span_kw),
            #         self.axes[4].axhspan(
            #             self.coords[2][self._last_ind[2]],
            #             self.coords[2][self._last_ind[2]],
            #             label='Z Span', **self.span_kw),
            #     ),
            # )
            # if self.lims[-1][-1] * self.lims[-1][0] < 0:
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
        for i in range(len(self.cursors)): self.all += self.cursors[i]

    def _get_middle_index(self, x):
        return len(x)//2 - (1 if len(x) % 2 == 0 else 0)

    def _refresh_bounds(self):
        self.maps[0].setRect(self._lims_to_rect(0, 1))
        if self.ndim == 3:
            self.maps[1].setRect(self._lims_to_rect(0, 2))
            self.maps[2].setRect(self._lims_to_rect(2, 1))

        for axis, cursors in enumerate(self.cursors):
            for c in cursors:
                c.setBounds(self.lims[axis])

    def set_labels(self, labels=None):
        """labels: list or tuple of str"""
        if labels is None:
            labels = self.dims
        self.axes[0].setLabels(left=labels[1], bottom=labels[0]),
        self.axes[1].setLabels(top=labels[0]),
        self.axes[2].setLabels(right=labels[1]),
        if self.ndim == 3:
            self.axes[3].setLabels(top=labels[2]),
            self.axes[4].setLabels(left=labels[2]),
            self.axes[5].setLabels(bottom=labels[2]),
        
    def set_data(self, data, update_all=False, reset_cursor=True):
        
        # Data properties
        self.data = parse_data(data)
        ndim_old = self.ndim
        self.ndim = self.data.ndim
        if self.ndim != ndim_old:
            update_all = True
        self.vals = self.data.values
        self._assign_vals_T()
        self.dims = self.data.dims
        self.shape = self.data.shape
        self.coords = tuple(self.data[dim].values for dim in self.dims)
        self.incs = tuple(coord[1] - coord[0] for coord in self.coords)
        self.lims = tuple((coord[0], coord[-1]) for coord in self.coords)

        if update_all:
            self.clear()
            self._initialize_layout()
            self._initialize_plots()
            self.avg_win = [1,] * self.ndim
            self.clim_locked = False
            self.clim_list = [()]  * self.ndim
            self.averaged = [False, ] * self.ndim
        
        # Imagetool properties
        if reset_cursor is True:
            self.cursor_pos = [None, ] * self.ndim
            self._last_ind = [None, ] * self.ndim
            self.reset_cursor()
        self.set_labels()
        self._apply_change()
        self._refresh_bounds()
        if update_all:
            self.colorbar = myColorBar(image=self.maps[0], width=20)
            self.addItem(self.colorbar, None, None, 4, 1)
            self.colorbar.setVisible(False)

    def reset_cursor(self):
        """Return the cursor to the center of the image."""
        for axis, coord in enumerate(self.coords):
            self.set_index(axis, self._get_middle_index(coord), update=False)

    def _cursor_drag(self, axis, line):
        self.set_value(axis, line.value())

    def connect_signals(self):
        """Connect events."""
        self.proxy = pg.SignalProxy(
            self.scene().sigMouseMoved,
            rateLimit=self.screen.refreshRate(),
            slot=self.onmove
        )
        for axis, cursors in enumerate(self.cursors):
            for c in cursors:
                c.sigDragged.connect(
                    lambda v, i=axis: self.set_value(i, v.value()))
        if self.bench:
            from collections import deque
            self._elapsed = deque(maxlen=1000)
            timer = QtCore.QTimer()
            # timer.timeout.connect(self._apply_change)
            timer.start(0)
            self._fpsLastUpdate = perf_counter()
    
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
        
    # def labelify(self, dim):
    #     """Prettify some frequently used axis labels."""
    #     labelformats = dict(
    #         kx = '$k_x$',
    #         ky = '$k_y$',
    #         kz = '$k_z$',
    #         alpha = '$\\alpha$',
    #         beta = '$\\beta$',
    #         theta = '$\\theta$',
    #         phi = '$\\phi$',
    #         chi = '$\\chi$',
    #         eV = '$E$'
    #     )
    #     try:
    #         return labelformats[dim]
    #     except KeyError:
    #         return dim

    def _assign_vals_T(self):
        if self.ndim == 2:
            self.vals_T = self.vals.T
        elif self.ndim == 3:
            self.vals_T = np.transpose(self.vals, axes=(1, 2, 0))
        else:
            raise NotImplementedError('Wrong data dimensions')
    
    def set_cmap(self, cmap=None, gamma=None, reverse=False):
        if cmap is not self.cmap:
            self.cmap = cmap
        if gamma is not self.gamma:
            self.gamma = gamma
        self.norm_cmap = get_powernorm_colormap(self.cmap, self.gamma,      
                                                reverse=reverse)
        for im in self.maps:
            im._colorMap = self.norm_cmap
            im.setLookupTable(self.norm_cmap.getStops()[1], update=False)
        self._apply_change(self._only_maps)

    def set_clim_lock(self, lock):
        self.colorbar.autolevels = ~lock
        if lock:
            self.clim_locked = True
            for i, m in enumerate(self.maps):
                self.clim_list[i] = m.getLevels()
        else:
            self.clim_locked = False
    
    def set_index(self, axis, index, update=True):
        self._last_ind[axis] = index
        self.cursor_pos[axis] = self.coords[axis][index]
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

    def set_navg(self, axis, n):
        self.avg_win[axis] = n
        if n == 1:
            self.averaged[axis] = False
            if not any(self.averaged):
                self.vals = self.data.values
                self._assign_vals_T()
                self._apply_change()
                return
        else:
            self.averaged[axis] = True
        
        if self.averaged[axis]: # if already averaged
            self.vals = move_mean_centered_multiaxis(
                self.data.values,
                self.avg_win,
                axis_list=get_true_indices(self.averaged)
            )
        else:
            self.averaged[axis] = True
            self.vals = move_mean_centered(self.vals, window=n, axis=axis)
        self._assign_vals_T()
        self._apply_change()
    
    def update_spans(self):
        for axis in range(self.ndim):
            domain = (
                self.coords[axis][self._last_ind[axis]]
                    - self.avg_win[axis] // 2 * self.incs[axis],
                self.coords[axis][self._last_ind[axis]] 
                    + (self.avg_win[axis] - 1) // 2 * self.incs[axis],
            )
            for span in self.spans[axis]:
                if is_vertical(span):
                    span.set_xy(get_xy_x(*domain))
                else:
                    span.set_xy(get_xy_y(*domain))
                span.set_visible(self.visible)
            if self.useblit:
                for i, span in list(
                    zip(self.span_ax_index[axis], self.spans[axis])):
                    self.axes[i].draw_artist(span)

    def get_index_of_value(self, axis, val):
        # return np.rint((val-self.lims[axis][0])/self.incs[axis]).astype(int)
        return min(
            np.searchsorted(self.coords[axis] + 0.5 * self.incs[axis], val),
            self.shape[axis] - 1,
        )

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
    
    def _get_mouse_datapos(self, i, pos):
        """Returns mouse position in data coords of `i`th axis"""
        mouse_point = self.axes[i].vb.mapSceneToView(pos)
        return mouse_point.x(), mouse_point.y()

    def onmove(self, evt):
        if self.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            return
        pos = evt[0]
        x, y, z = None, None, None
        if self.axes[0].sceneBoundingRect().contains(pos):
            datapos = self._get_mouse_datapos(0, pos)
            dx, dy, dz = True, True, False
            x, y = datapos
        elif self.axes[1].sceneBoundingRect().contains(pos):
            datapos = self._get_mouse_datapos(1, pos)
            dx, dy, dz = True, False, False
            x = datapos[0]
        elif self.axes[2].sceneBoundingRect().contains(pos):
            datapos = self._get_mouse_datapos(2, pos)
            dx, dy, dz = False, True, False
            y = datapos[1]
        elif self.axes[4].sceneBoundingRect().contains(pos):
            datapos = self._get_mouse_datapos(4, pos)
            dx, dy, dz = True, False, True
            x, z = datapos
        elif self.axes[5].sceneBoundingRect().contains(pos):
            datapos = self._get_mouse_datapos(5, pos)
            dx, dy, dz = False, True, True
            z, y = datapos
        elif self.axes[3].sceneBoundingRect().contains(pos):
            datapos = self._get_mouse_datapos(3, pos)
            dx, dy, dz = False, False, True
            z = datapos[0]
        else:
            return
        if self.bench: self._t_start = perf_counter()
        self.need_redraw = True

        if self.ndim == 2:
            cond = (False, dy, dx, dx, dx, dy, dy)
        elif self.ndim == 3:
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
        if dy:
            ind_y = self.get_index_of_value(1, y)
            if self.snap & (ind_y == self._last_ind[1]):
                dy = False
            else:
                self._last_ind[1] = ind_y
        if dz:
            ind_z = self.get_index_of_value(2, z)
            if self.snap & (ind_z == self._last_ind[2]):
                dz = False
            else:
                self._last_ind[2] = ind_z
        if self.snap:
            self.cursor_pos = [
                self.coords[i][self._last_ind[i]] for i in range(self.ndim)
            ]
        else:
            self.cursor_pos = [x, y, z]
        self._apply_change(cond)
        self.need_redraw = False
        if self.bench: self._measure_fps()
    
    def _apply_change(self, cond=None):
        if cond is None:
            cond = (True,) * len(self.all)
        for i in get_true_indices(cond):
            self._refresh_data(i)
    def transpose_axes(self, axis1, axis2):
        dims_new = list(self.dims)
        dims_new[axis1], dims_new[axis2] = self.dims[axis2], self.dims[axis1]
        new_data = self.data.transpose(*dims_new)
        self.cursor_pos[axis2], self.cursor_pos[axis1] = self.cursor_pos[axis1], self.cursor_pos[axis2]
        self._last_ind[axis2], self._last_ind[axis1] = self._last_ind[axis1], self._last_ind[axis2]
        self.set_data(new_data, update_all=False, reset_cursor=True)
    @supnan
    def _refresh_data(self, i):
        if self.ndim == 2:
            self._refresh_data_2d(i)
        elif self.ndim == 3:
            self._refresh_data_3d(i)
    def _refresh_data_2d(self, i):
        if i == 0:
            if self.clim_locked:
                self.all[i].setImage(self.vals_T, levels=self.clim_list[0])
            else:
                self.all[i].setImage(self.vals_T)
        elif i == 1: 
            self.all[i].setData(
                self.coords[0],
                self.vals_T[self._last_ind[1],:]
            )
        elif i == 2: 
            self.all[i].setData(
                self.vals_T[:,self._last_ind[0]],
                self.coords[1]
            )
        elif i in [3, 4]:
            self.all[i].setPos(self.cursor_pos[0])
        elif i in [5, 6]: 
            self.all[i].setPos(self.cursor_pos[1])
    def _refresh_data_3d(self, i):
        if i == 0: 
            if self.clim_locked:
                self.all[i].setImage(self.vals_T[:,self._last_ind[2],:],
                                     levels=self.clim_list[i])
            else:
                self.all[i].setImage(self.vals_T[:,self._last_ind[2],:])
        elif i == 1: 
            if self.clim_locked:
                self.all[i].setImage(self.vals_T[self._last_ind[1],:,:],
                                     levels=self.clim_list[i])
            else:
                self.all[i].setImage(self.vals_T[self._last_ind[1],:,:])
        elif i == 2: 
            if self.clim_locked:
                self.all[i].setImage(self.vals_T[:,:,self._last_ind[0]],
                                     levels=self.clim_list[i])
            else:
                self.all[i].setImage(self.vals_T[:,:,self._last_ind[0]])
        elif i == 3:
            self.all[i].setData(
                self.coords[0],
                self.vals[:,self._last_ind[1],self._last_ind[2]]
            )
        elif i == 4:
            self.all[i].setData(
                self.vals[self._last_ind[0],:,self._last_ind[2]],
                self.coords[1]
            )
        elif i == 5:
            self.all[i].setData(
                self.coords[2],
                self.vals[self._last_ind[0],self._last_ind[1],:]
            )
        elif i in [6, 7, 8]:
            self.all[i].setPos(self.cursor_pos[0])
        elif i in [9, 10, 11]: 
            self.all[i].setPos(self.cursor_pos[1])
        elif i in [12, 13, 14]: 
            self.all[i].setPos(self.cursor_pos[2])

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

class myColorBar(pg.PlotItem):
    
    def __init__(self, image, width=25, horiz_pad=45, vert_pad=30, inner_pad=5,     
                 font_size=10, curve_kw={}, line_kw={'pen':'cyan'},
                 *args, **kwargs):
        super(myColorBar, self).__init__(*args, **kwargs)

        self.cbar = pg.ImageItem(axisOrder='row-major')
        self.npts = 4096
        self.autolevels = True

        self.cbar.setImage(np.linspace(0, 1, self.npts).reshape((-1, 1)))
        self.addItem(self.cbar)
        
        self.isocurve = pg.IsocurveItem(**curve_kw)
        self.isocurve.setZValue(5)

        self.isoline = pg.InfiniteLine(angle=0, movable=True, **line_kw)
        self.addItem(self.isoline)
        self.isoline.setZValue(1000)
        self.isoline.sigDragged.connect(self.update_level)

        self.setImageItem(image)
        self.setMouseEnabled(x=False, y=True)
        self.setMenuEnabled(True)
        
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        self.layout.setColumnFixedWidth(1, width)
        self.layout.setSpacing(0.)
        self.layout.setContentsMargins(0., vert_pad, 0., vert_pad)
        self.showAxes((True, True, True, True), size=(inner_pad, inner_pad))
        self.getAxis('bottom').setStyle(showValues=False)
        self.getAxis('left').setStyle(showValues=False)
        self.getAxis('right').setStyle(
            showValues=True, tickTextWidth=horiz_pad,
            autoExpandTextSpace=False, autoReduceTextSpace=False)
        # self.getAxis('top').setHeight(vert_pad)   
        # self.getAxis('bottom').setHeight(vert_pad)
        # self.getAxis('left').setWidth(inner_pad)
        for axis in ['left', 'bottom', 'right', 'top']:
            self.getAxis(axis).setTickFont(font)
        
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
        self.update_isodata()
        
    def cmap_changed(self):
        self.cmap = self.imageItem()._colorMap
        self.lut = self.imageItem().lut
        self.cbar._colorMap = self.cmap
        self.cbar.setLookupTable(self.lut)
        if not self.npts == self.lut.shape[0]:
            self.npts = self.lut.shape[0]
            self.cbar.setImage(np.linspace(0, 1, self.npts).reshape((-1, 1)))

    def update_isodata(self):
        self.isocurve.setData(self.imageItem().image)
    def update_level(self, line):
        self.isocurve.setLevel(line.value())

    def setVisible(self, visible, *args, **kwargs):
        super().setVisible(visible, *args, **kwargs)
        self.isocurve.setVisible(visible, *args, **kwargs)
        self.getAxis('top').setStyle()
        self.getAxis('bottom').setStyle()
        self.getAxis('left').setStyle()
        self.getAxis('right').setStyle()

class dynamic_colorbar(pg.ColorBarItem):

    def __init__(self, image,
                 npts=4096, font_size=10, padding=(5, 30, 5, 30),
                 curve_kw={}, line_kw={'pen':'cyan'}, *args, **kwargs):
        kwargs['interactive'] = False
        super(dynamic_colorbar, self).__init__(*args, **kwargs)
        self.layout.setContentsMargins(*padding)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)

        self.npts = npts
        self.setImageItem(image)
        font = QtGui.QFont()
        font.setPointSizeF(float(font_size))
        self.axis.setTickFont(font)
        if self.horizontal:
            self.setRange( xRange=(0,self.npts), yRange=(0,1), padding=0 )
            self.bar.setImage( np.linspace(0, 1, self.npts).reshape( (-1,1) ) )
        else:
            self.setRange( xRange=(0,1), yRange=(0,self.npts), padding=0 )
            self.bar.setImage( np.linspace(0, 1, self.npts).reshape( (1,-1) ) )
        self.imageItem = self.img_list[0]
        # self.setLevels(self.imageItem().quickMinMax())
        self.imageItem().sigImageChanged.connect(self.imagechanged)
        self.isocurve = pg.IsocurveItem(**curve_kw)
        self.isocurve.setParentItem(self.imageItem())
        self.isocurve.setZValue(5)
        self.update_isodata()
        self.isoline = pg.InfiniteLine(angle=0, movable=True, **line_kw)
        self.vb.addItem(self.isoline)
        self.isoline.setZValue(1000)
        # self.isoline.setValue(0.5)
        self.imageItem().sigImageChanged.connect(self.update_isodata)
        self.isoline.sigDragged.connect(self.update_level)
    
    def imagechanged(self):
        self.axis.unlinkFromView()
        self.setLevels(self.imageItem().quickMinMax())
        self._colorMap = self.imageItem().getColorMap()
        self.bar.setLookupTable(self._colorMap.getLookupTable(nPts=self.npts))
        self.axis.linkToView(self.vb)
    
    def update_isodata(self):
        self.isocurve.setData(self.imageItem().image)

    def update_level(self, line):
        self.isocurve.setLevel(line.value()*self.imageItem().getLevels()[-1])

class isoHistogram(pg.HistogramLUTItem):

    def __init__(self, *args, curve_kw={}, line_kw={}, **kwargs):
        super(isoHistogram, self).__init__(*args, **kwargs)
        self.layout.setContentsMargins(5, 20, 5, 20)
        # self.gradient.setColorMap(self.imageItem().getColorMap())
        # self.setLevels(*self.imageItem().quickMinMax())
        # Isocurve drawing
        self.isocurve = pg.IsocurveItem(**curve_kw)
        self.isocurve.setParentItem(self.imageItem())
        self.isocurve.setZValue(5)
        self.update_isodata()
        # Draggable line for setting isocurve level
        self.isoline = pg.InfiniteLine(angle=0, movable=True, **line_kw)
        self.vb.addItem(self.isoline)
        # self.vb.setMouseEnabled(y=False)
        self.isoline.setZValue(1000)
        self.isoline.setValue(np.mean(self.imageItem().quickMinMax()))
        # # Contrast/color control
        # hist = pg.HistogramLUTItem()
        # hist.setImageItem(self.imageItem)
        # win.addItem(hist)
        self.imageItem().sigImageChanged.connect(self.update_isodata)
        self.isoline.sigDragged.connect(self.update_level)
        # self.sigLevelsChanged.connect(self.update_isodata)

    def update_isodata(self):
        self.isocurve.setData(self.imageItem().image)

    def update_level(self, line):
        self.isocurve.setLevel(line.value())


class ImageTool(QtWidgets.QMainWindow):
    def __init__(self, data, *args, **kwargs):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)
        self.data = parse_data(data)
        self.ndim = self.data.ndim
        self.itool = pg_itool(self.data, *args, **kwargs)
        self.icons = dict(
            invert = qta.icon('mdi6.invert-colors'),
            invert_off = qta.icon('mdi6.invert-colors-off'),
            lock = qta.icon('mdi6.lock'),
            unlock = qta.icon('mdi6.lock-open-variant'),
            colorbar = qta.icon('mdi6.gradient-vertical'),
            transpose = qta.icon('mdi6.arrow-left-right'),
        )
        
        self.cursortab = QtWidgets.QWidget()
        cursortab_content = QtWidgets.QHBoxLayout(self.cursortab)
        self._spinlabels = tuple(QtWidgets.QLabel(self.itool.dims[i])
                                 for i in range(self.ndim))
        self._cursor_spin = tuple(QtWidgets.QSpinBox(self.cursortab)
                                  for i in range(self.ndim))
        self._cursor_dblspin = tuple(QtWidgets.QDoubleSpinBox(self.cursortab)
                                     for i in range(self.ndim))
        self._transpose_button = tuple(QtWidgets.QPushButton(self.cursortab)
                                       for i in range(self.ndim))
        for i in range(self.ndim):
            self._cursor_spin[i].setRange(0, self.itool.shape[i] - 1)
            self._cursor_spin[i].setSingleStep(1)
            self._cursor_spin[i].setValue(self.itool._last_ind[i])
            self._cursor_spin[i].setWrapping(True)
            self._cursor_spin[i].valueChanged.connect(
                lambda v, axis=i: self._cursor_index_changed(axis, v))
            self._cursor_dblspin[i].setRange(*self.itool.lims[i])
            self._cursor_dblspin[i].setSingleStep(self.itool.incs[i])
            self._cursor_dblspin[i].setDecimals(3)
            self._cursor_dblspin[i].setValue(
                self.itool.coords[i][self.itool._last_ind[i]])
            self._cursor_dblspin[i].setWrapping(True)
            self._cursor_dblspin[i].valueChanged.connect(
                lambda v, axis=i: self._cursor_value_changed(axis, v))
            self._transpose_button[i].setIcon(self.icons['transpose'])
            self._transpose_button[i].clicked.connect(
                lambda axis1=i, axis2=i-1: self.transpose_axes(axis1,axis2))
        
        snap_check = QtWidgets.QCheckBox(self.cursortab)
        snap_check.setChecked(self.itool.snap)
        snap_check.stateChanged.connect(self._assign_snap)
        snap_label = QtWidgets.QLabel('Snap to data')
        snap_label.setBuddy(snap_check)

        for i in range(self.ndim):
            cursortab_content.addWidget(self._transpose_button[i])
            cursortab_content.addSpacing(10)
            cursortab_content.addWidget(self._spinlabels[i])
            cursortab_content.addWidget(self._cursor_spin[i])
            cursortab_content.addWidget(self._cursor_dblspin[i])
            cursortab_content.addSpacing(10)
        cursortab_content.addStretch()
        cursortab_content.addWidget(snap_check)
        cursortab_content.addWidget(snap_label)


        self.colorstab = QtWidgets.QWidget()
        colorstab_content = QtWidgets.QHBoxLayout(self.colorstab)

        self._gamma_spin = QtWidgets.QDoubleSpinBox()
        self._gamma_spin.setToolTip('Colormap gamma')
        self._gamma_spin.setSingleStep(0.01)
        self._gamma_spin.setRange(0.01, 100.)
        self._gamma_spin.setValue(self.itool.gamma)
        self._gamma_spin.valueChanged.connect(self.set_cmap)
        gamma_label = QtWidgets.QLabel('g')
        gamma_label.setBuddy(self._gamma_spin)

        self._cmap_combo = cmapComboBox(self.colorstab)
        self._cmap_combo.setMaximumWidth(175)
        self._cmap_combo.setCurrentText(self.itool.cmap)
        self._cmap_combo.currentTextChanged.connect(self.set_cmap)

        self._cmap_r_button = QtWidgets.QPushButton(self.colorstab)
        self._cmap_r_button.setCheckable(True)
        self._cmap_r_button.toggled.connect(self._set_cmap_reverse)
        self._cmap_r_button.setIcon(self.icons['invert'])
        self._cmap_r_button.setToolTip('Invert colormap')

        self._cmap_lock_button = QtWidgets.QPushButton(self.colorstab)
        self._cmap_lock_button.setCheckable(True)
        self._cmap_lock_button.toggled.connect(self._set_clim_lock)
        self._cmap_lock_button.setIcon(self.icons['unlock'])
        self._cmap_lock_button.setToolTip('Lock colors')

        self._cbar_show_button = QtWidgets.QPushButton(self.colorstab)
        self._cbar_show_button.setCheckable(True)
        self._cbar_show_button.toggled.connect(self.itool.colorbar.setVisible)
        self._cbar_show_button.setIcon(self.icons['colorbar'])
        self._cbar_show_button.setToolTip('Show colorbar')

        colors_button = QtWidgets.QPushButton('Colors')
        colors_button.clicked.connect(self._color_button_clicked)
        style_combo = QtWidgets.QComboBox(self.colorstab)
        style_combo.setToolTip('Qt style')
        style_combo.addItems(qt_style_names())
        style_combo.textActivated.connect(change_style)
        style_label = QtWidgets.QLabel('Style:')
        style_label.setBuddy(style_combo)
        style_combo.setCurrentIndex(style_combo.findText('Fusion'))
        colorstab_content.addWidget(gamma_label)
        colorstab_content.addWidget(self._gamma_spin)
        colorstab_content.addWidget(self._cmap_combo)
        colorstab_content.addWidget(self._cmap_r_button)
        colorstab_content.addWidget(self._cmap_lock_button)
        colorstab_content.addWidget(self._cbar_show_button)
        colorstab_content.addStretch()
        colorstab_content.addWidget(colors_button)
        colorstab_content.addStretch()
        colorstab_content.addWidget(style_label)
        colorstab_content.addWidget(style_combo)
        

        self.smoothtab = QtWidgets.QWidget()
        smoothtab_content = QtWidgets.QHBoxLayout(self.smoothtab)
        navg_label = tuple(QtWidgets.QLabel(self.itool.dims[i])
                                  for i in range(self.ndim))
        self._navg_spin = tuple(QtWidgets.QSpinBox(self.smoothtab)
                                  for i in range(self.ndim))
        navg_resetbutton = QtWidgets.QPushButton('Reset')
        navg_resetbutton.clicked.connect(self._navg_reset)
        for i in range(self.ndim):
            self._navg_spin[i].setRange(1, self.itool.shape[i] - 1)
            self._navg_spin[i].setSingleStep(2)
            self._navg_spin[i].setValue(1)
            self._navg_spin[i].setWrapping(False)
            self._navg_spin[i].valueChanged.connect(
                lambda n, axis=i: self._navg_changed(axis, n))
        for i in range(self.ndim):
            smoothtab_content.addWidget(navg_label[i])
            smoothtab_content.addWidget(self._navg_spin[i])
            smoothtab_content.addSpacing(20)
        smoothtab_content.addWidget(navg_resetbutton)
        smoothtab_content.addStretch()

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
        # self.tabwidget.addTab(self.pathtab, 'Path')
        
        self.layout.addWidget(self.itool)
        self.layout.addWidget(self.tabwidget)
        self.resize(700,700)
        self.itool.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.itool.setFocus()
        self.proxy = pg.SignalProxy(
            self.itool.scene().sigMouseMoved,
            rateLimit=self.itool.screen.refreshRate() * 0.5,
            slot=self.update_cursor_spins
        )
        for cursors in self.itool.cursors:
            for c in cursors:
                c.sigDragged.connect(self.update_cursor_spins)
        # self.show()
        # self._cbar_show_button.toggle()

    def update_cursor_spins(self):
        # if self.itool.qapp.queryKeyboardModifiers() != QtCore.Qt.ControlModifier:
            # return
        for i in range(self.ndim):
            self._cursor_spin[i].blockSignals(True)
            self._cursor_spin[i].setValue(self.itool._last_ind[i])
            self._cursor_spin[i].blockSignals(False)
            self._cursor_dblspin[i].blockSignals(True)
            self._cursor_dblspin[i].setValue(
                self.itool.coords[i][self.itool._last_ind[i]])
            self._cursor_dblspin[i].blockSignals(False)

    def transpose_axes(self, axis1, axis2):
        self.itool.transpose_axes(axis1, axis2)
        for i in range(self.ndim):
            self._spinlabels[i].setText(self.itool.dims[i])
            self._cursor_spin[i].setRange(0, self.itool.shape[i] - 1)
            # self._cursor_spin[i].setSingleStep(1)
            self._cursor_spin[i].setValue(self.itool._last_ind[i])
            # self._cursor_spin[i].setWrapping(True)
            # self._cursor_spin[i].valueChanged.connect(
                # lambda v, axis=i: self._cursor_index_changed(axis, v))
            self._cursor_dblspin[i].setRange(*self.itool.lims[i])
            self._cursor_dblspin[i].setSingleStep(self.itool.incs[i])
            # self._cursor_dblspin[i].setDecimals(3)
            self._cursor_dblspin[i].setValue(
                self.itool.coords[i][self.itool._last_ind[i]])
            # self._cursor_dblspin[i].setWrapping(True)
            # self._cursor_dblspin[i].valueChanged.connect(
                # lambda v, axis=i: self._cursor_value_changed(axis, v))
            # self._transpose_button[i].setIcon(self.icons['transpose'])
            # self._transpose_button[i].clicked.connect(
                # lambda axis1=i, axis2=i-1: self.transpose_axes(axis1,axis2))

    def set_cmap(self):
        reverse = self._cmap_r_button.isChecked()
        gamma = self._gamma_spin.value()
        cmap = self._cmap_combo.currentText()
        self.itool.set_cmap(cmap, gamma=gamma, reverse=reverse)


    def _set_cmap_reverse(self, v):
        if v:
            self._cmap_r_button.setIcon(self.icons['invert_off'])
        else:
            self._cmap_r_button.setIcon(self.icons['invert'])
        self.set_cmap()

    def _set_clim_lock(self, v):
        if v:
            self._cmap_lock_button.setIcon(self.icons['lock'])
        else:
            self._cmap_lock_button.setIcon(self.icons['unlock'])
        self.itool.set_clim_lock(v)
    
    def _navg_changed(self, axis, n):
        self.itool.set_navg(axis, n)
    
    def _navg_reset(self):
        for i in range(self.ndim):
            self._navg_spin[i].setValue(1)

    def _cursor_index_changed(self, axis, index):
        self._cursor_dblspin[axis].blockSignals(True)
        self.itool.set_index(axis, index)
        self._cursor_dblspin[axis].setValue(self.itool.coords[axis][index])
        self._cursor_dblspin[axis].blockSignals(False)

    def _cursor_value_changed(self, axis, value):
        self._cursor_spin[axis].blockSignals(True)
        self.itool.set_value(axis, value)
        self._cursor_spin[axis].setValue(self.itool._last_ind[axis])
        self._cursor_spin[axis].blockSignals(False)

    def _color_button_clicked(self, s):
        # print("click", s)
        dialog = ImageToolColors(self)
        if dialog.exec():
            # print("Success!")
            pass
        else:
            pass
            # print("Cancel!")
    
    def _assign_snap(self, value):
        self.itool.snap = value
    

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
    itool(dat.transpose('kx','ky','eV'), bench=False)
    # imagetool(dat)
    
    # itool(dat.sel(eV=0,method='nearest'), bench=False)