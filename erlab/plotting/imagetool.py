import sys
import time
from itertools import compress

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backend_bases import NavigationToolbar2, _Mode
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator
from matplotlib.widgets import Widget
from joblib import Parallel, delayed

__all__ = ['itool']

class mpl_itool(Widget):
    """A interactive tool based on `matplotlib` for exploring 3D data.
    
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
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.
    
    Examples
    --------
    See :doc:`/gallery/widgets/multicursor`.

    Notes
    -----
    Axes indices:
        ┌───┬─────┐
        │ 1 │     │
        │───┤  3  │
        │ 4 │     │
        │───┼───┬─│
        │ 0 │ 5 │2│
        └───┴───┴─┘
    """

    def __init__(self, canvas, axes, data, snap=False, gamma=0.5, cmap='terrain_r', parallel=False, bench=False, **improps):
        self.canvas = canvas
        self.axes = axes
        self.data = data
        self.vals = data.values
        self.snap = snap
        self.gamma = gamma
        self.cmap = cmap
        self.parallel = parallel
        self.bench = bench

        if not self.canvas.supports_blit:
            raise RuntimeError('Canvas does not support blit. '
                               'If running in ipython, add `%matplotlib qt`.')
        
        self.visible = True
        self.background = None
        self.needclear = False

        lineprops=dict(
            ls='-', lw=.8, c='grey', animated=True, visible=False,
        )

        improps.update(dict(
            animated=True, visible=False,
            interpolation='none', aspect='auto', origin='lower',
            norm=colors.PowerNorm(self.gamma), 
            cmap=self.cmap
        ))

        self.dim_y, self.dim_z, self.dim_x = self.data.dims

        self.coord_x, self.coord_y, self.coord_z = (
            self.data[self.dim_x].values,
            self.data[self.dim_y].values,
            self.data[self.dim_z].values,
        )
        self.len_x, self.len_y, self.len_z = (len(self.coord_x), 
                                              len(self.coord_y), 
                                              len(self.coord_z))

        self.inc_x, self.inc_y, self.inc_z = (
            self.coord_x[1] - self.coord_x[0],
            self.coord_y[1] - self.coord_y[0],
            self.coord_z[1] - self.coord_z[0],
        )

        self.lims_x, self.lims_y, self.lims_z = (
            (self.coord_x[0], self.coord_x[-1]),
            (self.coord_y[0], self.coord_y[-1]),
            (self.coord_z[0], self.coord_z[-1]),
        )
        _get_middle_index = lambda x: len(x)//2 - (1 if len(x) % 2 == 0 else 0)
        xmid, ymid, zmid = (_get_middle_index(self.coord_x),
                            _get_middle_index(self.coord_y),
                            _get_middle_index(self.coord_z))
        
        self._last_ind_x, self._last_ind_y, self._last_ind_z = xmid, ymid, zmid
        self._shift = False

        self.map_xy = self.axes[0].imshow(self.vals[:,0,:],
                                          extent=(*self.lims_x, *self.lims_y), **improps)
        self.map_xz = self.axes[4].imshow(self.vals[0,:,:],
                                          extent=(*self.lims_x, *self.lims_z), **improps)
        self.map_zy = self.axes[5].imshow(self.vals[:,:,0],
                                          extent=(*self.lims_z, *self.lims_y), **improps)
        self.histx = self.axes[1].plot(
            self.coord_x, self.vals[ymid,zmid,:],
            visible=False, animated=True, color='k', lw=.8)[0]
        self.histy = self.axes[2].plot(
            self.vals[:,zmid,xmid], self.coord_y,
            visible=False, animated=True, color='k', lw=.8)[0]
        self.histz = self.axes[3].plot(
            self.coord_z, self.vals[ymid,:,xmid],
            visible=False, animated=True, color='k', lw=.8)[0]
        self.xcursor = [
            self.axes[0].axvline(self.coord_x[xmid], **lineprops),
            self.axes[1].axvline(self.coord_x[xmid], **lineprops),
            self.axes[4].axvline(self.coord_x[xmid], **lineprops),
        ]
        self.ycursor = [
            self.axes[0].axhline(self.coord_y[ymid], **lineprops),
            self.axes[2].axhline(self.coord_y[ymid], **lineprops),
            self.axes[5].axhline(self.coord_y[ymid], **lineprops),
        ]
        self.zcursor = [
            self.axes[3].axvline(self.coord_z[zmid], **lineprops),
            self.axes[5].axvline(self.coord_z[zmid], **lineprops),
            self.axes[4].axhline(self.coord_z[zmid], **lineprops),
        ]
        axes[3].axvline(0., animated=False, color='k', lw=.8, ls='--')
        self.maps = [self.map_zy, self.map_xz, self.map_xy]
        self.hists = [self.histx, self.histy, self.histz]
        self.cursors = self.xcursor + self.ycursor + self.zcursor
        self.all = self.maps + self.hists + self.cursors
        

        for xc in self.xcursor: xc.set_label('X Cursor')
        for yc in self.ycursor: yc.set_label('Y Cursor')
        for zc in self.zcursor: zc.set_label('Z Cursor')
        self.histx.set_label('X Profile')
        self.histy.set_label('Y Profile')
        self.histz.set_label('Z Profile')

        self.axes[0].set_xlabel(self.labelify(self.dim_x))
        self.axes[0].set_ylabel(self.labelify(self.dim_y))
        self.axes[1].set_xlabel(self.labelify(self.dim_x))
        self.axes[2].set_ylabel(self.labelify(self.dim_y))
        self.axes[3].set_xlabel(self.labelify(self.dim_z))
        self.axes[4].set_ylabel(self.labelify(self.dim_z))
        self.axes[5].set_xlabel(self.labelify(self.dim_z))
        self.axes[1].xaxis.set_label_position('top')
        self.axes[2].yaxis.set_label_position('right')
        self.axes[3].xaxis.set_label_position('top')

        self.axes[0].set_xlim(self.lims_x)
        self.axes[0].set_ylim(self.lims_y)
        self.axes[3].set_xlim(self.lims_z)
        self.axes[4].set_ylim(self.lims_z)
        self.axes[5].set_xlim(self.lims_z)
        self.axes[1].set_yticks([])
        self.axes[2].set_xticks([])
        self.axes[3].set_yticks([])
        if self.bench:
            self.counter = 0.
            self.fps = 0.
            self.lastupdate = time.time()

        self._only_x = [True, False, False,
                        False, True, True,
                        True, True, True,
                        False, False, False,
                        False, False, False]
        self._only_y = [False, True, False,
                        True, False, True, 
                        False, False, False,
                        True, True, True,
                        False, False, False]
        self._only_z = [False, False, True,
                        True, True, False,
                        False, False, False,
                        False, False, False,
                        True, True, True]
        # if self.parallel:
        # self.all = self.xcursor + self.zcursor[:-1] + [self.histy] \
        #             + self.ycursor + [self.zcursor[-1], self.histx, self.histz] \
        #             + self.maps
        # 0 1 2  3 4 5  678 91011 121314
        # 6  7  8  12 13 4  9  10 11 14 3   5   0   1   2 
        # 0  1  2  3  4  5  6  7  8  9  10  11  12  13  14
        # 12,13,14,10,5,11,0,1,2,6,7,8,3,4,9
        # self.coord_list = ['x', 'x', 'x', 'x', 'x', 'x',
                    # 'y', 'y', 'y', 'y', 'y', 'y', '', '', '']
        self.coord_list = ['', '', '', 'y', 'x', 'y', 'x', 'x', 'x', 'y', 'y', 'y', 'x', 'x', 'y']
        self.ax_index = [0, 4, 5, 1, 2, 3, 0, 1, 4, 0, 2, 5, 3, 5, 4]
        self.axes[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=False)
        self.axes[2].ticklabel_format(axis='x', style='sci', scilimits=(-2, 3), useMathText=False)
        self.axes[3].ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=False)
        if self.parallel:
            self.pool = Parallel(n_jobs=-1,require='sharedmem',verbose=0)
        self.connect()
        # self.canvas.draw()
        # self._apply_change()

    def connect(self):
        """Connect events."""
        self._cidmotion = self.canvas.mpl_connect('motion_notify_event',
                                                  self.onmove)
        self._ciddraw = self.canvas.mpl_connect('draw_event', self.clear)
        self._cidpress = self.canvas.mpl_connect('key_press_event',
                                                 self.onpress)
        self._cidrelease = self.canvas.mpl_connect('key_release_event',
                                                   self.onrelease)

    def disconnect(self):
        """Disconnect events."""
        self.canvas.mpl_disconnect(self._cidmotion)
        self.canvas.mpl_disconnect(self._ciddraw)
        self.canvas.mpl_disconnect(self._cidpress)
        self.canvas.mpl_disconnect(self._cidrelease)

    def clear(self, event):
        """Clear the cursor."""
        if self.ignore(event):
            return
        for obj in self.all:
            obj.set_visible(False)
        self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self.axes[1].set_yticks([])
        self.axes[2].set_xticks([])
        self.axes[3].set_yticks([])

    def labelify(self, dim):
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
            return labelformats[dim]
        except KeyError:
            return dim

    def onpress(self,event):
        if event.key == 'shift':
            self._shift = True

    def onrelease(self,event):
        if event.key == 'shift':
            self._shift = False
    
    def set_cmap(self, cmap):
        self.cmap = cmap
        for im in self.maps: 
            im.set_cmap(self.cmap)
        for obj in self.all:
            obj.set_visible(False)
        self._apply_change()

    def set_gamma(self, gamma):
        self.gamma = gamma
        self._apply_change()

    def set_index_x(self, xi):
        self._last_ind_x = xi
        self._apply_change(self._only_x)
        self.cursor_pos[0] = self.coord_x[xi]
        
    def set_index_y(self, yi):
        self._last_ind_y = yi
        self._apply_change(self._only_y)
        self.cursor_pos[1] = self.coord_y[yi]

    def set_index_z(self, zi):
        self._last_ind_z = zi
        self._apply_change(self._only_z)
        self.cursor_pos[2] = self.coord_z[zi]

    def set_value_x(self, x):
        self.set_index_x(np.rint((x-self.lims_x[0])/self.inc_x).astype(int))
        self.cursor_pos[0] = x
        
    def set_value_y(self, y):
        self.set_index_y(np.rint((y-self.lims_y[0])/self.inc_y).astype(int))
        self.cursor_pos[1] = y

    def set_value_z(self, z):
        self.set_index_z(np.rint((z-self.lims_z[0])/self.inc_z).astype(int))
        self.cursor_pos[2] = z
            
    def onmove(self, event):
        if self.ignore(event):
            return
        if event.inaxes not in self.axes:
            return
        if not self.canvas.widgetlock.available(self):
            return
        if not event.button:
            if not self._shift:
                return
        self.needclear = True
        if not self.visible:
            return
        x, y, z = None, None, None
        if event.inaxes == self.axes[0]:
            dx, dy, dz = True, True, False
            x, y = event.xdata, event.ydata
        elif event.inaxes == self.axes[4]:
            dx, dy, dz = True, False, True
            x, z = event.xdata, event.ydata
        elif event.inaxes == self.axes[5]:
            dx, dy, dz = False, True, True
            z, y = event.xdata, event.ydata
        elif event.inaxes == self.axes[1]:
            dx, dy, dz = True, False, False
            x = event.xdata
        elif event.inaxes == self.axes[2]:
            dx, dy, dz = False, True, False
            y = event.ydata
        elif event.inaxes == self.axes[3]:
            dx, dy, dz = False, False, True
            z = event.xdata

        cond = [dx, dy, dz,
                dy or dz, dx or dz, dx or dy,
                dx, dx, dx,
                dy, dy, dy,
                dz, dz, dz]
        if dx:
            ind_x = min(
                np.searchsorted(self.coord_x + 0.5 * self.inc_x, x),
                self.len_x - 1,
            )
            if (ind_x == self._last_ind_x) & self.snap:
                dx = False
            else:
                self._last_ind_x = ind_x
        if dy:
            ind_y = min(
                np.searchsorted(self.coord_y + 0.5 * self.inc_y, y),
                self.len_y - 1,
            )
            if (ind_y == self._last_ind_y) & self.snap:
                dy = False
            else:
                self._last_ind_y = ind_y
        if dz:
            ind_z = min(
                np.searchsorted(self.coord_z + 0.5 * self.inc_z, z),
                self.len_z - 1,
            )
            if (ind_z == self._last_ind_z) & self.snap:
                dz = False
            else:
                self._last_ind_z = ind_z

        if self.snap:
            self.cursor_pos = [
                self.coord_x[self._last_ind_x],
                self.coord_y[self._last_ind_y],
                self.coord_z[self._last_ind_z]
            ]
        else:
            self.cursor_pos = [x, y, z]
        self._apply_change(cond)
        if self.bench:
            self.print_time()
    
    def _apply_change(self, cond=[True]*15):
        if self.parallel:
            self.pool(delayed(self.set_data)(i)
                    for i in list(compress(range(len(cond)), cond)))
            self.pool(delayed(a.set_visible)(self.visible)
                    for a in self.all)
            self._update()
        else:
            for i in list(compress(range(len(cond)), cond)): self.set_data(i)
            for a in self.all: a.set_visible(self.visible)
            self._update()
    def _update(self):
        self.axes[1].yaxis.set_major_locator(AutoLocator())
        self.axes[2].xaxis.set_major_locator(AutoLocator())
        self.axes[3].yaxis.set_major_locator(AutoLocator())
        for i in range(3):
            self.axes[i+1].relim()
            self.axes[i+1].autoscale_view()
        for im in self.maps: 
            im.set_norm(colors.PowerNorm(self.gamma))
        if self.background is not None:
            self.canvas.restore_region(self.background)
        if self.parallel:
            self.pool(delayed(self.axes[i].draw_artist)(art) for i, art in list(zip(
                [0, 4, 5, 1, 2, 3, 1, 2, 3],
                self.maps + self.hists + [self.axes[1].yaxis,
                                          self.axes[2].xaxis,
                                          self.axes[3].yaxis])))
            self.pool(delayed(self.axes[i].draw_artist)(art) for i, art in list(zip(
                [0, 1, 4, 0, 2, 5, 3, 5, 4], self.cursors)))
        else:
            for i, art in list(zip(self.ax_index + [1, 2, 3],
                                   self.all + [self.axes[1].yaxis,
                                               self.axes[2].xaxis,
                                               self.axes[3].yaxis])):
                self.axes[i].draw_artist(art)
        self.canvas.blit()
    def print_time(self):
        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        print(tx, end='\r')
    
    def set_data(self, i):
        if i == 0: self.all[i].set_data(self.vals[:, :, self._last_ind_x])
        elif i == 1: self.all[i].set_data(self.vals[self._last_ind_y, :, :])
        elif i == 2: self.all[i].set_data(self.vals[:, self._last_ind_z, :])
        elif i == 3: 
            self.all[i].set_ydata(self.vals[self._last_ind_y, self._last_ind_z, :])
        elif i == 4: 
            self.all[i].set_xdata(self.vals[:, self._last_ind_z, self._last_ind_x])
        elif i == 5:
            self.all[i].set_ydata(self.vals[self._last_ind_y, :, self._last_ind_x])
        elif i in [6, 7, 8]:
            self.all[i].set_xdata((self.cursor_pos[0], self.cursor_pos[0]))
        elif i in [9, 10, 11]: 
            self.all[i].set_ydata((self.cursor_pos[1], self.cursor_pos[1]))
        elif i in [12, 13]: 
            self.all[i].set_xdata((self.cursor_pos[2], self.cursor_pos[2]))
        elif i == 14: 
            self.all[i].set_ydata((self.cursor_pos[2], self.cursor_pos[2]))

    def _drawpath(self):
        # ld = LineDrawer(self.canvas, self.axes[0])
        # points = ld.draw_line()
        # print(points)
        # TODO
        pass
    def _onselectpath(self, verts):
        print(verts)



class ImageTool(QtWidgets.QMainWindow):
    def __init__(self, data, *args, **kwargs):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)
        self.main_canvas = FigureCanvas(Figure())
        gs = self.main_canvas.figure.add_gridspec(3, 3, width_ratios=(6,4,2), height_ratios=(2,4,6))
        self._axes_main = self.main_canvas.figure.add_subplot(gs[2, 0])
        self._axes = [
            self._axes_main,
            self.main_canvas.figure.add_subplot(gs[0, 0], sharex=self._axes_main),
            self.main_canvas.figure.add_subplot(gs[2, 2], sharey=self._axes_main),
            self.main_canvas.figure.add_subplot(gs[:2, 1:]),
            self.main_canvas.figure.add_subplot(gs[1, 0], sharex=self._axes_main),
            self.main_canvas.figure.add_subplot(gs[2, 1], sharey=self._axes_main),
        ]
        self._axes[0].set_label('Main Image')
        self._axes[1].set_label('X Profile')
        self._axes[2].set_label('Y Profile')
        self._axes[3].set_label('Z Profile')
        self._axes[4].set_label('Horiz Slice')
        self._axes[5].set_label('Vert Slice')
        self._axes[4].label_outer()
        self._axes[5].label_outer()
        self._axes[1].xaxis.tick_top()
        self._axes[2].yaxis.tick_right()
        self._axes[3].xaxis.tick_top()
        self._axes[3].yaxis.tick_right()
        self.main_canvas.figure.set_tight_layout(True)
                
        self.mc = mpl_itool(self.main_canvas, self._axes,
                            data, *args, **kwargs)
        
        self.NavBar = NavigationToolbar
        init_old = self.NavBar.__init__
        def init_new(self, canvas, parent, coordinates=True):
            self.parent = parent
            init_old(self, canvas, parent, coordinates=coordinates)
        home_old = self.NavBar.home
        def home_new(self, *args):
            home_old(self, *args)
            axes = self.canvas.figure.axes
            axes[1].set_ylim(auto=True)
            axes[2].set_xlim(auto=True)
            axes[3].set_ylim(auto=True)
        pan_old = self.NavBar.pan
        def pan_new(self, *args):
            if self.mode == _Mode.PAN:
                self.parent.mc.connect()
            else:
                self.parent.mc.disconnect()
            pan_old(self, *args)
        zoom_old = self.NavBar.zoom
        def zoom_new(self, *args):
            if self.mode == _Mode.ZOOM:
                self.parent.mc.connect()
            else:
                self.parent.mc.disconnect()
            zoom_old(self, *args)
        self.NavBar.__init__ = init_new
        self.NavBar.home = home_new
        self.NavBar.pan = pan_new
        self.NavBar.zoom = zoom_new
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        self.NavBar(self.main_canvas, self))
        self.infotab = QtWidgets.QWidget()
        infotabcontent = QtWidgets.QHBoxLayout(self.infotab)
        spinxlabel = QtWidgets.QLabel(self.mc.dim_x)
        spinylabel = QtWidgets.QLabel(self.mc.dim_y)
        spinzlabel = QtWidgets.QLabel(self.mc.dim_z)
        self.infospin_x = QtWidgets.QSpinBox(self.infotab)
        self.infospin_y = QtWidgets.QSpinBox(self.infotab)
        self.infospin_z = QtWidgets.QSpinBox(self.infotab)
        self.infodblspin_x = QtWidgets.QDoubleSpinBox(self.infotab)
        self.infodblspin_y = QtWidgets.QDoubleSpinBox(self.infotab)
        self.infodblspin_z = QtWidgets.QDoubleSpinBox(self.infotab)
        self.infospin_x.setRange(0, self.mc.len_x - 1)
        self.infospin_y.setRange(0, self.mc.len_y - 1)
        self.infospin_z.setRange(0, self.mc.len_z - 1)
        self.infospin_x.setSingleStep(1)
        self.infospin_y.setSingleStep(1)
        self.infospin_z.setSingleStep(1)
        self.infospin_x.setValue(self.mc._last_ind_x)
        self.infospin_y.setValue(self.mc._last_ind_y)
        self.infospin_z.setValue(self.mc._last_ind_z)
        self.infospin_x.valueChanged.connect(lambda v: self._spinchanged('x', v))
        self.infospin_y.valueChanged.connect(lambda v: self._spinchanged('y', v))
        self.infospin_z.valueChanged.connect(lambda v: self._spinchanged('z', v))
        self.infospin_x.setWrapping(True)
        self.infospin_y.setWrapping(True)
        self.infospin_z.setWrapping(True)
        self.infodblspin_x.setRange(*self.mc.lims_x)
        self.infodblspin_y.setRange(*self.mc.lims_y)
        self.infodblspin_z.setRange(*self.mc.lims_z)
        self.infodblspin_x.setSingleStep(self.mc.inc_x)
        self.infodblspin_y.setSingleStep(self.mc.inc_y)
        self.infodblspin_z.setSingleStep(self.mc.inc_z)
        self.infodblspin_x.setDecimals(3)
        self.infodblspin_y.setDecimals(3)
        self.infodblspin_z.setDecimals(3)
        self.infodblspin_x.setValue(self.mc.coord_x[self.mc._last_ind_x])
        self.infodblspin_y.setValue(self.mc.coord_y[self.mc._last_ind_y])
        self.infodblspin_z.setValue(self.mc.coord_z[self.mc._last_ind_z])
        self.infodblspin_x.valueChanged.connect(lambda v: self._spindblchanged('x', v))
        self.infodblspin_y.valueChanged.connect(lambda v: self._spindblchanged('y', v))
        self.infodblspin_z.valueChanged.connect(lambda v: self._spindblchanged('z', v))
        self.infodblspin_x.setWrapping(True)
        self.infodblspin_y.setWrapping(True)
        self.infodblspin_z.setWrapping(True)
        cursorsnapcheck = QtWidgets.QCheckBox(self.infotab)
        cursorsnapcheck.setChecked(self.mc.snap)
        cursorsnapcheck.stateChanged.connect(self._assign_snap)
        infotabcontent.addWidget(spinxlabel)
        infotabcontent.addWidget(self.infodblspin_x)
        infotabcontent.addWidget(self.infospin_x)
        infotabcontent.addSpacing(20)
        infotabcontent.addWidget(spinylabel)
        infotabcontent.addWidget(self.infodblspin_y)
        infotabcontent.addWidget(self.infospin_y)
        infotabcontent.addSpacing(20)
        infotabcontent.addWidget(spinzlabel)
        infotabcontent.addWidget(self.infodblspin_z)
        infotabcontent.addWidget(self.infospin_z)
        infotabcontent.addStretch()
        infotabcontent.addWidget(cursorsnapcheck)
        infotabcontent.addWidget(QtWidgets.QLabel('Snap to Data'))

        self.colorstab = QtWidgets.QWidget()
        colorstabcontent = QtWidgets.QHBoxLayout()
        gammalabel = QtWidgets.QLabel('g')
        gammaspin = QtWidgets.QDoubleSpinBox()
        gammaspin.setToolTip("Colormap Gamma")
        gammaspin.setSingleStep(0.05)
        gammaspin.setValue(self.mc.gamma)
        gammaspin.valueChanged.connect(self.mc.set_gamma)
        colormaps = QtWidgets.QComboBox()
        colormaps.setToolTip("Colormap")
        colormaps.addItems(plt.colormaps())
        colormaps.setCurrentIndex(colormaps.findText(self.mc.cmap))
        self.main_canvas.draw()
        colormaps.currentTextChanged.connect(self.mc.set_cmap)
        colorstabcontent.addWidget(gammalabel)
        colorstabcontent.addWidget(gammaspin)
        colorstabcontent.addWidget(colormaps)
        colorstabcontent.addStretch()
        self.colorstab.setLayout(colorstabcontent)

        self.pathtab = QtWidgets.QWidget()
        pathtabcontent = QtWidgets.QHBoxLayout()
        pathlabel = QtWidgets.QLabel('Add point: `space`\nRemove point: `delete`\nFinish selection: `enter`')
        pathstart = QtWidgets.QPushButton()
        pathstart.clicked.connect(self.mc._drawpath)
        pathtabcontent.addWidget(pathlabel)
        pathtabcontent.addWidget(pathstart)
        self.pathtab.setLayout(pathtabcontent)

        self.tabwidget = QtWidgets.QTabWidget()
        self.tabwidget.addTab(self.infotab, "Info")
        self.tabwidget.addTab(self.colorstab, "Colors")
        # self.tabwidget.addTab(self.pathtab, "Path")
        self.layout.addWidget(self.tabwidget)
        
        self.layout.addWidget(self.main_canvas)
        self.main_canvas.mpl_connect('motion_notify_event',self.onmove_super)
        self.main_canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.main_canvas.setFocus()

    def onmove_super(self, event):
        if event.inaxes not in self._axes:
            return
        if not event.button:
            if not self.mc._shift:
                return
        self.infospin_x.blockSignals(True)
        self.infospin_y.blockSignals(True)
        self.infospin_z.blockSignals(True)
        self.infodblspin_x.blockSignals(True)
        self.infodblspin_y.blockSignals(True)
        self.infodblspin_z.blockSignals(True)
        self.infospin_x.setValue(self.mc._last_ind_x)
        self.infospin_y.setValue(self.mc._last_ind_y)
        self.infospin_z.setValue(self.mc._last_ind_z)
        self.infodblspin_x.setValue(self.mc.coord_x[self.mc._last_ind_x])
        self.infodblspin_y.setValue(self.mc.coord_y[self.mc._last_ind_y])
        self.infodblspin_z.setValue(self.mc.coord_z[self.mc._last_ind_z])
        self.infospin_x.blockSignals(False)
        self.infospin_y.blockSignals(False)
        self.infospin_z.blockSignals(False)
        self.infodblspin_x.blockSignals(False)
        self.infodblspin_y.blockSignals(False)
        self.infodblspin_z.blockSignals(False)

    def _spinchanged(self, axis, index):
        if axis == 'x':
            self.infodblspin_x.blockSignals(True)
            self.mc.set_index_x(index)
            self.infodblspin_x.setValue(self.mc.coord_x[index])
            self.infodblspin_x.blockSignals(False)
        elif axis == 'y':
            self.infodblspin_y.blockSignals(True)
            self.mc.set_index_y(index)
            self.infodblspin_y.setValue(self.mc.coord_y[index])
            self.infodblspin_y.blockSignals(False)
        elif axis == 'z':
            self.infodblspin_z.blockSignals(True)
            self.mc.set_index_z(index)
            self.infodblspin_z.setValue(self.mc.coord_z[index])
            self.infodblspin_z.blockSignals(False)
    def _spindblchanged(self, axis, value):
        if axis == 'x':
            self.infospin_x.blockSignals(True)
            self.mc.set_value_x(value)
            self.infospin_x.setValue(self.mc._last_ind_x)
            self.infospin_x.blockSignals(False)
        elif axis == 'y':
            self.infospin_y.blockSignals(True)
            self.mc.set_value_y(value)
            self.infospin_y.setValue(self.mc._last_ind_y)
            self.infospin_y.blockSignals(False)
        elif axis == 'z':
            self.infospin_z.blockSignals(True)
            self.mc.set_value_z(value)
            self.infospin_z.setValue(self.mc._last_ind_z)
            self.infospin_z.blockSignals(False)
    def _assign_snap(self, value):
        self.mc.snap = value
def itool(data, *args, **kwargs):
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    with plt.rc_context({
        'text.usetex':False,
    #     #  'mathtext.fontset':'stixsans',
        'font.size':7,
        'font.family':'sans',
        # 'font.family':'Helvetica',
    }):
        app = ImageTool(data, *args, **kwargs)
    qapp.setStyle('Fusion')
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
