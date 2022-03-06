import sys
import time
from itertools import compress

import numpy as np
import darkdetect
import qtawesome as qta
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.backend_bases import _Mode
from matplotlib.backends.backend_qtagg import (FigureCanvas,
                                               NavigationToolbar2QT)
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator
from matplotlib.widgets import Widget
from joblib import Parallel, delayed

__all__ = ['itool']

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

def cmap2qpixmap(name:str):
    cmap = plt.colormaps[name]
    cmap_arr = cmap(np.tile(np.linspace(0, 1, 256), (256, 1))) * 255
    img = QtGui.QImage(cmap_arr.astype(np.uint8).data,
                       cmap_arr.shape[1], cmap_arr.shape[0],
                       QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img)

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
    **cursorprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.
    
    Examples
    --------
    See :doc:`/gallery/widgets/multicursor`.

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

    def __init__(self, canvas, axes, data, snap=False, gamma=0.5,
                 cmap='magma', parallel=False, bench=False, **improps):
        self.canvas = canvas
        self.axes = axes
        self.data = data
        self.snap = snap
        self.gamma = gamma
        self.cmap = cmap
        self.parallel = parallel
        self.bench = bench

        if not self.canvas.supports_blit:
            raise RuntimeError('Canvas does not support blit. '
                               'If running in ipython, add `%matplotlib qt`.')
        for ax in self.axes:
            for loc, spine in ax.spines.items():
                spine.set_position(('outward', 1))
        cursorprops = dict(
            ls='-', lw=.8, c=plt.rcParams.get('axes.edgecolor'), alpha=0.5,
            animated=False, visible=True,
        )
        lineprops = dict(
            ls='-', lw=.8, c=plt.rcParams.get('axes.edgecolor'), alpha=1,
            animated=False, visible=True,
        )
        fermilineprops = dict(
            ls='--', lw=.8, c=plt.rcParams.get('axes.edgecolor'), alpha=1,
            animated=False, 
        )
        improps.update(dict(
            animated=False, visible=True,
            interpolation='none', aspect='auto', origin='lower',
            norm=colors.PowerNorm(self.gamma), 
            cmap=self.cmap, rasterized=True
        ))
        self.vals = self.data.values
        self.ndim = self.data.ndim
        if self.ndim == 2:
            self.vals_T = self.vals.T
        elif self.ndim == 3:
            self.vals_T = np.transpose(self.vals, axes=(1,2,0))
        self.dims = self.data.dims
        self.coords = tuple(self.data[self.dims[i]] for i in range(self.ndim))
        self.shape = self.data.shape
        self.incs = tuple(self.coords[i][1] - self.coords[i][0]
                          for i in range(self.ndim))
        self.lims = tuple((self.coords[i][0], self.coords[i][-1])
                          for i in range(self.ndim))
        
        _get_middle_index = lambda x: len(x)//2 - (1 if len(x) % 2 == 0 else 0)
        mids = tuple(_get_middle_index(self.coords[i])
                     for i in range(self.ndim))
        self.cursor_pos = [self.coords[i][mids[i]] for i in range(self.ndim)]
        self._last_ind = list(mids)
        self._shift = False

        if self.ndim == 2:
            self.maps = (
                self.axes[0].imshow(self.vals_T,
                                    extent=(*self.lims[0], *self.lims[1]),
                                    label='Main Image',
                                    **improps),
            )
            self.hists = (
                self.axes[1].plot(self.coords[0], self.vals[:,mids[1]],
                                  label='X Profile', **lineprops)[0],
                self.axes[2].plot(self.vals[mids[0],:], self.coords[1],
                                  label='Y Profile', **lineprops)[0],
            )
            self.cursors = (
                self.axes[0].axvline(self.coords[0][mids[0]],
                                    label='X Cursor', **cursorprops),
                self.axes[1].axvline(self.coords[0][mids[0]],
                                    label='X Cursor', **cursorprops),
                self.axes[0].axhline(self.coords[1][mids[1]],
                                    label='Y Cursor', **cursorprops),
                self.axes[2].axhline(self.coords[1][mids[1]],
                                    label='Y Cursor', **cursorprops),
            )
            self.scaling_axes = (self.axes[1].yaxis,
                                 self.axes[2].xaxis)
            self.ax_index = (0, 1, 2, 0, 1, 0, 2, 1, 2)
            self._only_axis = (
                (False, False, True, True, True, False, False),
                (False, True, False, False, False, True, True),
            )
        elif self.ndim == 3:
            self.maps = (
                self.axes[0].imshow(self.vals_T[:,mids[2],:], 
                                    extent=(*self.lims[0], *self.lims[1]),
                                    label='Main Image',
                                    **improps),
                self.axes[4].imshow(self.vals_T[mids[1],:,:],
                                    extent=(*self.lims[0], *self.lims[2]),
                                    label='Horiz Slice',
                                    **improps),
                self.axes[5].imshow(self.vals_T[:,:,mids[0]],
                                    extent=(*self.lims[2], *self.lims[1]),
                                    label='Vert Slice',
                                    **improps),
            )
            self.hists = (
                self.axes[1].plot(self.coords[0], self.vals[:,mids[1],mids[2]],
                                  label='X Profile', **lineprops)[0],
                self.axes[2].plot(self.vals[mids[0],:,mids[2]], self.coords[1],
                                  label='Y Profile', **lineprops)[0],
                self.axes[3].plot(self.coords[2], self.vals[mids[0],mids[1],:],
                                  label='Z Profile', **lineprops)[0],
            )
            self.cursors = (
                self.axes[0].axvline(self.coords[0][mids[0]],
                                    label='X Cursor', **cursorprops),
                self.axes[1].axvline(self.coords[0][mids[0]],
                                    label='X Cursor', **cursorprops),
                self.axes[4].axvline(self.coords[0][mids[0]],
                                    label='X Cursor', **cursorprops),
                self.axes[0].axhline(self.coords[1][mids[1]],
                                    label='Y Cursor', **cursorprops),
                self.axes[2].axhline(self.coords[1][mids[1]],
                                    label='Y Cursor', **cursorprops),
                self.axes[5].axhline(self.coords[1][mids[1]],
                                    label='Y Cursor', **cursorprops),
                self.axes[3].axvline(self.coords[2][mids[2]],
                                    label='Z Cursor', **cursorprops),
                self.axes[5].axvline(self.coords[2][mids[2]],
                                    label='Z Cursor', **cursorprops),
                self.axes[4].axhline(self.coords[2][mids[2]],
                                    label='Z Cursor', **cursorprops),
            )
            self.scaling_axes = (self.axes[1].yaxis,
                                 self.axes[2].xaxis,
                                 self.axes[3].yaxis)
            if self.lims[-1][-1] * self.lims[-1][0] < 0:
                axes[3].axvline(0., label='Fermi Level', **fermilineprops)
            self.ax_index = (0, 4, 5, # images
                             1, 2, 3, # profiles
                             0, 1, 4, 0, 2, 5, 3, 5, 4, # cursors
                             1, 2, 3) # axes with dynamic limits
            self._only_axis = (
                (False, False, True, False, True, True,
                 True, True, True, False, False, False, False, False, False),
                (False, True, False, True, False, True,
                 False, False, False, True, True, True, False, False, False),
                (True, False, False, True, True, False,
                 False, False, False, False, False, False, True, True, True),
            )
            self.axes[3].set_xlabel(self.labelify(self.dims[2]))
            self.axes[4].set_ylabel(self.labelify(self.dims[2]))
            self.axes[5].set_xlabel(self.labelify(self.dims[2]))
            self.axes[3].xaxis.set_label_position('top')
            self.axes[3].set_xlim(self.lims[2])
            self.axes[4].set_ylim(self.lims[2])
            self.axes[5].set_xlim(self.lims[2])
            self.axes[3].set_yticks([])
            self.axes[3].ticklabel_format(axis='y', style='sci',
                                          scilimits=(-2,3), useMathText=False)

        self.all = self.maps + self.hists + self.cursors
        
        self.axes[0].set_xlabel(self.labelify(self.dims[0]))
        self.axes[0].set_ylabel(self.labelify(self.dims[1]))
        self.axes[1].set_xlabel(self.labelify(self.dims[0]))
        self.axes[2].set_ylabel(self.labelify(self.dims[1]))
        self.axes[1].xaxis.set_label_position('top')
        self.axes[2].yaxis.set_label_position('right')
        self.axes[0].set_xlim(self.lims[0])
        self.axes[0].set_ylim(self.lims[1])
        self.axes[1].set_yticks([])
        self.axes[2].set_xticks([])
        self.axes[1].ticklabel_format(axis='y', style='sci',
                                      scilimits=(-2,3), useMathText=False)
        self.axes[2].ticklabel_format(axis='x', style='sci',
                                      scilimits=(-2,3), useMathText=False)

        if self.bench:
            self.counter = 0.
            self.fps = 0.
            self.lastupdate = time.time()
        
        if self.parallel:
            self.pool = Parallel(n_jobs=-1, require='sharedmem', verbose=0)

        self.visible = True
        self.background = None
        self.needclear = False

        self.connect()
        # self.canvas.draw()
        # self.axes[1].xaxis.get_major_ticks()[0].label1.set_visible(False)
        # self.axes[3].xaxis.get_major_ticks()[-1].label1.set_visible(False)
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
        for ax in self.scaling_axes:
            ax.set_ticks([])

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

    def set_index(self, axis, index):
        self._last_ind[axis] = index
        self._apply_change(self._only_axis[axis])
        self.cursor_pos[axis] = self.coords[axis][index]

    def set_value(self, axis, val):
        self.set_index(axis, self.get_index_of_value(axis, val))
        self.cursor_pos[axis] = val

    def get_index_of_value(self, axis, val):
        # return np.rint((val-self.lims[axis][0])/self.incs[axis]).astype(int)
        return min(
            np.searchsorted(self.coords[axis] + 0.5 * self.incs[axis], val),
            self.shape[axis] - 1,
        )
    
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
        elif event.inaxes == self.axes[1]:
            dx, dy, dz = True, False, False
            x = event.xdata
        elif event.inaxes == self.axes[2]:
            dx, dy, dz = False, True, False
            y = event.ydata
        elif event.inaxes == self.axes[4]:
            dx, dy, dz = True, False, True
            x, z = event.xdata, event.ydata
        elif event.inaxes == self.axes[5]:
            dx, dy, dz = False, True, True
            z, y = event.xdata, event.ydata
        elif event.inaxes == self.axes[3]:
            dx, dy, dz = False, False, True
            z = event.xdata

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
        if self.bench:
            self.print_time()
    
    def _apply_change(self, cond=None):
        if cond is None:
            cond = (True,) * len(self.all)
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
        for ax in self.scaling_axes:
            ax.set_major_locator(AutoLocator())
            ax.axes.relim()
            ax.axes.autoscale_view()
        for im in self.maps: 
            im.set_norm(colors.PowerNorm(self.gamma))
        if self.background is not None:
            self.canvas.restore_region(self.background)
        if self.parallel:
            raise NotImplementedError
            # self.pool(delayed(self.axes[i].draw_artist)(art) for i, art in list(zip(
            #     (0, 4, 5, 1, 2, 3, 1, 2, 3),
            #     self.maps + self.hists + (self.axes[1].yaxis,
            #                               self.axes[2].xaxis,
            #                               self.axes[3].yaxis))))
            # self.pool(delayed(self.axes[i].draw_artist)(art) for i, art in list(zip(
            #     (0, 1, 4, 0, 2, 5, 3, 5, 4), self.cursors)))
        else:
            for i, art in list(zip(self.ax_index,
                                   self.all + self.scaling_axes)):
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
        if self.ndim == 2:
            self.set_data_2d(i)
        elif self.ndim == 3:
            self.set_data_3d(i)

    def set_data_2d(self, i):
        if i == 0: 
            self.all[i].set_data(self.vals_T)
        elif i == 1: 
            self.all[i].set_ydata(self.vals_T[self._last_ind[1],:])
        elif i == 2: 
            self.all[i].set_xdata(self.vals_T[:,self._last_ind[0]])
        elif i in [3, 4]:
            self.all[i].set_xdata((self.cursor_pos[0],self.cursor_pos[0]))
        elif i in [5, 6]: 
            self.all[i].set_ydata((self.cursor_pos[1],self.cursor_pos[1]))
    
    def set_data_3d(self, i):
        if i == 0: 
            self.all[i].set_data(self.vals_T[:,self._last_ind[2],:])
        elif i == 1: 
            self.all[i].set_data(self.vals_T[self._last_ind[1],:,:])
        elif i == 2: 
            self.all[i].set_data(self.vals_T[:,:,self._last_ind[0]])
        elif i == 3:
            self.all[i].set_ydata(self.vals[:,self._last_ind[1],self._last_ind[2]])
        elif i == 4: 
            self.all[i].set_xdata(self.vals[self._last_ind[0],:,self._last_ind[2]])
        elif i == 5:
            self.all[i].set_ydata(self.vals[self._last_ind[0],self._last_ind[1],:])
        elif i in [6, 7, 8]:
            self.all[i].set_xdata((self.cursor_pos[0],self.cursor_pos[0]))
        elif i in [9, 10, 11]: 
            self.all[i].set_ydata((self.cursor_pos[1],self.cursor_pos[1]))
        elif i in [12, 13]: 
            self.all[i].set_xdata((self.cursor_pos[2],self.cursor_pos[2]))
        elif i == 14: 
            self.all[i].set_ydata((self.cursor_pos[2],self.cursor_pos[2]))

    def _drawpath(self):
        # ld = LineDrawer(self.canvas, self.axes[0])
        # points = ld.draw_line()
        # print(points)
        # TODO
        pass
    def _onselectpath(self, verts):
        print(verts)

class ImageToolNavBar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        self.parent = parent
        NavigationToolbar2QT.__init__(self, canvas, parent, coordinates=coordinates)
    def _icon(self, name):
        """
        Construct a `.QIcon` from an image file *name*, including the extension
        and relative to Matplotlib's "images" data directory.
        """
        name = name.replace('.png', '')
        icons_dict = dict(
            # back = qta.icon('ph.arrow-arc-left-fill'),
            # forward = qta.icon('ph.arrow-arc-right-fill'),
            # filesave = qta.icon('ph.floppy-disk-fill'),
            # home = qta.icon('ph.corners-out-fill'),
            # move = qta.icon('ph.arrows-out-cardinal-fill'),
            # qt4_editor_options = qta.icon('ph.palette-fill'),
            # zoom_to_rect = qta.icon('ph.crop-fill'),
            # subplots = qta.icon('ph.squares-four-fill'),
            back = qta.icon('msc.chevron-left'),
            forward = qta.icon('msc.chevron-right'),
            filesave = qta.icon('msc.save'),
            home = qta.icon('msc.debug-step-back'),
            move = qta.icon('msc.move'),
            qt4_editor_options = qta.icon('msc.graph-line'),
            zoom_to_rect = qta.icon('msc.search'),
            subplots = qta.icon('msc.editor-layout'),
        )
        try:
            return icons_dict[name]
        except:
            print(name)
            raise Exception
        # name = name.replace('.png', '_large.png')
        # pm = QtGui.QPixmap(str(cbook._get_data_path('images', name)))
        # _setDevicePixelRatio(pm, _devicePixelRatioF(self))
        # if self.palette().color(self.backgroundRole()).value() < 128:
            # icon_color = self.palette().color(self.foregroundRole())
            # mask = pm.createMaskFromColor(
                # QtGui.QColor('black'),
                # _enum("QtCore.Qt.MaskMode").MaskOutColor)
            # pm.fill(icon_color)
            # pm.setMask(mask)
        # return QtGui.QIcon(pm)

class ImageTool(QtWidgets.QMainWindow):
    def __init__(self, data, *args, **kwargs):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)
        self.figure = Figure(
            figsize=(10,10), dpi=75, frameon=True, layout='constrained'
        )
        self.ndim = data.ndim
        if self.ndim == 3:
            gs = self.figure.add_gridspec(3, 3,
                    width_ratios=(6,4,2), height_ratios=(2,4,6))
            self._axes_main = self.figure.add_subplot(gs[2, 0])
            self._axes = [
                self._axes_main,
                self.figure.add_subplot(gs[0,0], sharex=self._axes_main),
                self.figure.add_subplot(gs[2,2], sharey=self._axes_main),
                self.figure.add_subplot(gs[:2,1:]),
                self.figure.add_subplot(gs[1,0], sharex=self._axes_main),
                self.figure.add_subplot(gs[2,1], sharey=self._axes_main),
            ]
            self._axes[3].set_label('Z Profile Axes')
            self._axes[4].set_label('Horiz Slice Axes')
            self._axes[5].set_label('Vert Slice Axes')
            self._axes[4].label_outer()
            self._axes[5].label_outer()
            self._axes[3].xaxis.tick_top()
            self._axes[3].yaxis.tick_right()
        elif self.ndim == 2:
            gs = self.figure.add_gridspec(
                    2, 2, width_ratios=(6,2), height_ratios=(2,6))
            self._axes_main = self.figure.add_subplot(gs[1, 0])
            self._axes = [
                self._axes_main,
                self.figure.add_subplot(gs[0, 0], sharex=self._axes_main),
                self.figure.add_subplot(gs[1, 1], sharey=self._axes_main),
            ]
        self._axes[0].set_label('Main Image Axes')
        self._axes[1].set_label('X Profile Axes')
        self._axes[2].set_label('Y Profile Axes')
        self._axes[1].xaxis.tick_top()
        self._axes[2].yaxis.tick_right()
        
        self.main_canvas = FigureCanvas(self.figure)
        self.mc = mpl_itool(self.main_canvas, self._axes,
                            data, *args, **kwargs)
        self.NavBar = ImageToolNavBar
        home_old = self.NavBar.home
        def home_new(self, *args):
            home_old(self, *args)
            axes = self.canvas.figure.axes
            axes[1].set_ylim(auto=True)
            axes[2].set_xlim(auto=True)
            if self.parent.mc.ndim == 3:
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
        self.NavBar.home = home_new
        self.NavBar.pan = pan_new
        self.NavBar.zoom = zoom_new
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        self.NavBar(self.main_canvas, self))
        self.infotab = QtWidgets.QWidget()
        infotab_content = QtWidgets.QHBoxLayout(self.infotab)
        spinlabels = tuple(QtWidgets.QLabel(self.mc.dims[i])
                           for i in range(self.ndim))
        self.infospinners = tuple(QtWidgets.QSpinBox(self.infotab)
                                  for i in range(self.ndim))
        self.infodblspinners = tuple(QtWidgets.QDoubleSpinBox(self.infotab)
                                     for i in range(self.ndim))
        for i in range(self.ndim):
            self.infospinners[i].setRange(0, self.mc.shape[i] - 1)
            self.infospinners[i].setSingleStep(1)
            self.infospinners[i].setValue(self.mc._last_ind[i])
            self.infospinners[i].setWrapping(True)
            self.infospinners[i].valueChanged.connect(
                lambda v, axis=i: self._spinchanged(axis, v))
            self.infodblspinners[i].setRange(*self.mc.lims[i])
            self.infodblspinners[i].setSingleStep(self.mc.incs[i])
            self.infodblspinners[i].setDecimals(3)
            self.infodblspinners[i].setValue(
                self.mc.coords[i][self.mc._last_ind[i]])
            self.infodblspinners[i].setWrapping(True)
            self.infodblspinners[i].valueChanged.connect(
                lambda v, axis=i: self._spindblchanged(axis, v))
        
        cursorsnapcheck = QtWidgets.QCheckBox(self.infotab)
        cursorsnapcheck.setChecked(self.mc.snap)
        cursorsnapcheck.stateChanged.connect(self._assign_snap)

        for i in range(self.ndim):
            infotab_content.addWidget(spinlabels[i])
            infotab_content.addWidget(self.infodblspinners[i])
            infotab_content.addWidget(self.infospinners[i])
            infotab_content.addSpacing(20)
        infotab_content.addStretch()
        infotab_content.addWidget(cursorsnapcheck)
        infotab_content.addWidget(QtWidgets.QLabel('Snap to Data'))

        self.colorstab = QtWidgets.QWidget()
        colorstab_content = QtWidgets.QHBoxLayout()

        _gamma_spin = QtWidgets.QDoubleSpinBox()
        _gamma_spin.setToolTip("Colormap Gamma")
        _gamma_spin.setSingleStep(0.01)
        _gamma_spin.setRange(0.01, 100.)
        _gamma_spin.setValue(self.mc.gamma)
        _gamma_spin.valueChanged.connect(self.mc.set_gamma)
        gamma_label = QtWidgets.QLabel('g')
        gamma_label.setBuddy(_gamma_spin)
        _cmap_combo = QtWidgets.QComboBox()
        _cmap_combo.setToolTip("Colormap")
        for name in plt.colormaps():
            _cmap_combo.addItem(QtGui.QIcon(cmap2qpixmap(name)), name)
        _cmap_combo.setCurrentIndex(_cmap_combo.findText(self.mc.cmap))

        _style_combo = QtWidgets.QComboBox()
        _style_combo.setToolTip("Qt Style")
        _style_combo.addItems(qt_style_names())
        _style_combo.textActivated.connect(change_style)
        _style_combo.setCurrentIndex(_style_combo.findText('Fusion'))
        style_label = QtWidgets.QLabel("Style:")
        style_label.setBuddy(_style_combo)

        qt_style_names()
        self.main_canvas.draw()
        _cmap_combo.currentTextChanged.connect(self.mc.set_cmap)
        colorstab_content.addWidget(gamma_label)
        colorstab_content.addWidget(_gamma_spin)
        colorstab_content.addWidget(_cmap_combo)
        colorstab_content.addStretch()
        colorstab_content.addWidget(style_label)
        colorstab_content.addWidget(_style_combo)
        self.colorstab.setLayout(colorstab_content)
        
        # self.pathtab = QtWidgets.QWidget()
        # pathtabcontent = QtWidgets.QHBoxLayout()
        # pathlabel = QtWidgets.QLabel('Add point: `space`\nRemove point: `delete`\nFinish selection: `enter`')
        # pathstart = QtWidgets.QPushButton()
        # pathstart.clicked.connect(self.mc._drawpath)
        # pathtabcontent.addWidget(pathlabel)
        # pathtabcontent.addWidget(pathstart)
        # self.pathtab.setLayout(pathtabcontent)

        self.tabwidget = QtWidgets.QTabWidget()
        self.tabwidget.addTab(self.infotab, "Info")
        self.tabwidget.addTab(self.colorstab, "Appearance")
        # self.tabwidget.addTab(self.pathtab, "Path")

        self.layout.addWidget(self.main_canvas)
        self.layout.addWidget(self.tabwidget)
        self.main_canvas.mpl_connect('motion_notify_event', self.onmove_super)
        self.main_canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.main_canvas.setFocus()

    def onmove_super(self, event):
        if event.inaxes not in self._axes:
            return
        if not event.button:
            if not self.mc._shift:
                return
        for i in range(self.ndim):
            self.infospinners[i].blockSignals(True)
            self.infospinners[i].setValue(self.mc._last_ind[i])
            self.infospinners[i].blockSignals(False)
            self.infodblspinners[i].blockSignals(True)
            self.infodblspinners[i].setValue(
                self.mc.coords[i][self.mc._last_ind[i]])
            self.infodblspinners[i].blockSignals(False)

    def _spinchanged(self, axis, index):
        self.infodblspinners[axis].blockSignals(True)
        self.mc.set_index(axis, index)
        self.infodblspinners[axis].setValue(self.mc.coords[axis][index])
        self.infodblspinners[axis].blockSignals(False)

    def _spindblchanged(self, axis, value):
        self.infospinners[axis].blockSignals(True)
        self.mc.set_value(axis, value)
        self.infospinners[axis].setValue(self.mc._last_ind[axis])
        self.infospinners[axis].blockSignals(False)

    def _assign_snap(self, value):
        self.mc.snap = value

def itool(data, *args, **kwargs):
    # TODO: implement multiple windows, add transpose, add binning
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    # print(qapp.devicePixelRatio())
    if darkdetect.isDark():
        mpl_style = 'dark_background'
    else:
        mpl_style = 'default'
    with plt.rc_context({
        'text.usetex':False,
        # 'font.family':'SF Pro',
        # 'font.size':8,
        # 'font.stretch':'condensed',
        # 'mathtext.fontset':'cm',
        # 'font.family':'fantasy',
    }):
        
        with plt.style.context(mpl_style):
            app = ImageTool(data, *args, **kwargs)
    change_style('Fusion')
    # qapp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps) 
    # qapp.setAttribute(QtCore.Qt.AA_Use96Dpi)
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
