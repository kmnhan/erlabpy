"""
.. deprecated:: 0.1.

    This module is deprecated, and is only kept for reference purposes.
    Use `erlab.interactive.imagetool` instead.

"""

import sys
import time
from itertools import compress

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import qtawesome as qta

# import numbagg
import xarray as xr
from joblib import Parallel, delayed
from matplotlib import colors
from matplotlib.backend_bases import _Mode
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator
from matplotlib.widgets import Widget

__all__ = ["mpl_itool"]


def qt_style_names():
    """Return a list of styles, default platform style first."""
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


def colormap_to_QPixmap(name: str, h=64):
    """Convert matplotlib colormap to a 256-by-`h` QPixmap."""
    cmap = plt.colormaps[name]
    cmap_arr = cmap(np.tile(np.linspace(0, 1, 256), (h, 1))) * 255
    img = QtGui.QImage(
        cmap_arr.astype(np.uint8).data,
        cmap_arr.shape[1],
        cmap_arr.shape[0],
        QtGui.QImage.Format_RGBA8888,
    )
    return QtGui.QPixmap.fromImage(img)


def color_to_QColor(c, alpha=None):
    """Convert matplotlib color to QtGui.Qcolor."""
    return QtGui.QColor.fromRgbF(*colors.to_rgba(c, alpha=alpha))


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


def move_mean_centered_multiaxis(
    a,
    window_list,
    min_count_list=None,
    axis_list=None,
):
    if axis_list is None:
        axis_list = [-1]
    if not isinstance(axis_list, list):
        axis_list = [
            axis_list,
        ]
    w_list = [(window - 1) // 2 for window in window_list]
    pad_width = [(0, 0)] * a.ndim
    slicer = [
        slice(None),
    ] * a.ndim
    if min_count_list is None:
        min_count_list = [w + 1 for w in w_list]
    for axis in axis_list:
        pad_width[axis] = (0, w_list[axis] + 1)
        slicer[axis] = slice(w_list[axis], -1)
    a = np.pad(a, pad_width, constant_values=np.nan)
    val = _move_mean_multiaxis_calc(a, window_list, min_count_list, axis_list)
    return val[tuple(slicer)]


def _move_mean_multiaxis_calc(a_padded, window_list, min_count_list, axis_list):
    val = a_padded
    for axis in axis_list:
        val = bn.move_mean(
            val, window_list[axis], min_count=min_count_list[axis], axis=axis
        )
    return val


def parse_data(data):
    if isinstance(data, xr.Dataset):
        try:
            data = data.spectrum
        except BaseException:
            raise TypeError(
                "input argument data must be a xarray.DataArray or a "
                "numpy.ndarray. Create an xarray.DataArray "
                "first, either with indexing on the Dataset or by "
                "invoking the `to_array()` method."
            ) from None
    elif isinstance(data, np.ndarray):
        data = xr.DataArray(data)
    return data


def is_vertical(artist):
    return artist.axes.get_xaxis_transform() == artist.get_transform()


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
        super().__init__(*args, **kwargs)

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
                f"QWidget {{ background-color: {self._color.name(QtGui.QColor.HexArgb)}; border: 0; }}"
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
        return super().mousePressEvent(e)


class mpl_itool(Widget):
    """A interactive tool based on `matplotlib` for exploring 3D data.

    For the tool to remain responsive you must
    keep a reference to it.

    Parameters
    ----------
    canvas : `matplotlib.backend_bases.FigureCanvasBase`
        The FigureCanvas that contains all the axes.
    axes : list of `matplotlib.axes.Axes`
        The `matplotlib.axes.Axes` to attach the cursor to. See Notes for the
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
    **self.cursorprops
        `matplotlib.lines.Line2D` properties that control the appearance of the lines.
        See also `matplotlib.axes.Axes.axhline`.

    """

    # Notes
    # -----
    # Axes indices for 3D data:
    #     ┌───┬─────┐
    #     │ 1 │     │
    #     │───┤  3  │
    #     │ 4 │     │
    #     │───┼───┬─│
    #     │ 0 │ 5 │2│
    #     └───┴───┴─┘
    # Axes indices for 2D data:
    #     ┌───┬───┐
    #     │ 1 │   │
    #     │───┼───│
    #     │ 0 │ 2 │
    #     └───┴───┘

    def __init__(
        self,
        canvas,
        axes,
        data,
        snap=False,
        gamma=0.5,
        cmap="magma",
        useblit=True,
        parallel=False,
        bench=False,
        cursorprops=None,
        lineprops=None,
        fermilineprops=None,
        **improps,
    ):
        if fermilineprops is None:
            fermilineprops = {}
        if lineprops is None:
            lineprops = {}
        if cursorprops is None:
            cursorprops = {}
        self.canvas = canvas
        self.axes = axes
        self.data = parse_data(data)
        self.snap = snap
        self.gamma = gamma
        self.cmap = cmap
        self.useblit = useblit
        self.parallel = parallel
        self.bench = bench

        if self.useblit and not self.canvas.supports_blit:
            raise RuntimeError(
                "Canvas does not support blit. "
                "If running in ipython, add `%matplotlib qt`."
            )
        for ax in self.axes:
            for spine in ax.spines.values():
                spine.set_position(("outward", 1))
        self.cursorprops = cursorprops
        self.lineprops = lineprops
        self.fermilineprops = fermilineprops
        self.improps = improps
        self.cursorprops.update(
            {
                "linestyle": "-",
                "linewidth": 0.8,
                "color": colors.to_rgba(plt.rcParams.get("axes.edgecolor"), alpha=0.5),
                "animated": self.useblit,
                "visible": True,
            }
        )
        self.lineprops.update(
            {
                "linestyle": "-",
                "linewidth": 0.8,
                "color": colors.to_rgba(plt.rcParams.get("axes.edgecolor"), alpha=1),
                "animated": self.useblit,
                "visible": True,
            }
        )
        self.fermilineprops.update(
            {
                "linestyle": "--",
                "linewidth": 0.8,
                "color": colors.to_rgba(plt.rcParams.get("axes.edgecolor"), alpha=1),
                "animated": False,
            }
        )
        self.improps.update(
            {
                "animated": self.useblit,
                "visible": True,
                "interpolation": "none",
                "aspect": "auto",
                "origin": "lower",
                "norm": colors.PowerNorm(self.gamma),
                "cmap": self.cmap,
                "rasterized": True,
            }
        )
        self.spanprops = {
            # edgecolor=plt.rcParams.get('axes.edgecolor'),
            # lw=0.5, ls='--',
            "facecolor": colors.to_rgba(self.cursorprops["color"], alpha=1),
            "alpha": 0.15,
            "animated": self.useblit,
            "visible": True,
        }
        self._get_middle_index = lambda x: len(x) // 2 - (1 if len(x) % 2 == 0 else 0)

        self.vals = self.data.values

        self.ndim = self.data.ndim
        self._assign_vals_T()
        self.avg_win = [
            1,
        ] * self.ndim
        self.clim_locked = False
        self.clim_list = [()] * self.ndim
        self.dims = self.data.dims
        self.coords = tuple(self.data[self.dims[i]] for i in range(self.ndim))
        self.shape = self.data.shape
        self.incs = tuple(
            self.coords[i][1] - self.coords[i][0] for i in range(self.ndim)
        )
        self.lims = tuple(
            (self.coords[i][0], self.coords[i][-1]) for i in range(self.ndim)
        )

        mids = tuple(self._get_middle_index(self.coords[i]) for i in range(self.ndim))
        self.cursor_pos = [self.coords[i][mids[i]] for i in range(self.ndim)]
        self._last_ind = list(mids)
        self._shift = False

        if self.ndim == 2:
            self.maps = (
                self.axes[0].imshow(
                    self.vals_T,
                    extent=(*self.lims[0], *self.lims[1]),
                    label="Main Image",
                    **self.improps,
                ),
            )
            self.hists = (
                self.axes[1].plot(
                    self.coords[0],
                    self.vals[:, mids[1]],
                    label="X Profile",
                    **self.lineprops,
                )[0],
                self.axes[2].plot(
                    self.vals[mids[0], :],
                    self.coords[1],
                    label="Y Profile",
                    **self.lineprops,
                )[0],
            )
            self.cursors = (
                self.axes[0].axvline(
                    self.coords[0][mids[0]], label="X Cursor", **self.cursorprops
                ),
                self.axes[1].axvline(
                    self.coords[0][mids[0]], label="X Cursor", **self.cursorprops
                ),
                self.axes[0].axhline(
                    self.coords[1][mids[1]], label="Y Cursor", **self.cursorprops
                ),
                self.axes[2].axhline(
                    self.coords[1][mids[1]], label="Y Cursor", **self.cursorprops
                ),
            )
            self.spans = (
                (
                    self.axes[0].axvspan(
                        self.coords[0][self._last_ind[0]],
                        self.coords[0][self._last_ind[0]],
                        label="X Span",
                        **self.spanprops,
                    ),
                    self.axes[1].axvspan(
                        self.coords[0][self._last_ind[0]],
                        self.coords[0][self._last_ind[0]],
                        label="X Span",
                        **self.spanprops,
                    ),
                ),
                (
                    self.axes[0].axhspan(
                        self.coords[1][self._last_ind[1]],
                        self.coords[1][self._last_ind[1]],
                        label="Y Span",
                        **self.spanprops,
                    ),
                    self.axes[2].axhspan(
                        self.coords[1][self._last_ind[1]],
                        self.coords[1][self._last_ind[1]],
                        label="Y Span",
                        **self.spanprops,
                    ),
                ),
            )
            self.scaling_axes = (self.axes[1].yaxis, self.axes[2].xaxis)
            self.ax_index = (0, 1, 2, 0, 1, 0, 2, 1, 2)
            self.span_ax_index = ((0, 1), (0, 2))
            self._only_axis = (
                (False, False, True, True, True, False, False),
                (False, True, False, False, False, True, True),
            )
        elif self.ndim == 3:
            self.maps = (
                self.axes[0].imshow(
                    self.vals_T[:, mids[2], :],
                    extent=(*self.lims[0], *self.lims[1]),
                    label="Main Image",
                    **self.improps,
                ),
                self.axes[4].imshow(
                    self.vals_T[mids[1], :, :],
                    extent=(*self.lims[0], *self.lims[2]),
                    label="Horiz Slice",
                    **self.improps,
                ),
                self.axes[5].imshow(
                    self.vals_T[:, :, mids[0]],
                    extent=(*self.lims[2], *self.lims[1]),
                    label="Vert Slice",
                    **self.improps,
                ),
            )
            self.hists = (
                self.axes[1].plot(
                    self.coords[0],
                    self.vals[:, mids[1], mids[2]],
                    label="X Profile",
                    **self.lineprops,
                )[0],
                self.axes[2].plot(
                    self.vals[mids[0], :, mids[2]],
                    self.coords[1],
                    label="Y Profile",
                    **self.lineprops,
                )[0],
                self.axes[3].plot(
                    self.coords[2],
                    self.vals[mids[0], mids[1], :],
                    label="Z Profile",
                    **self.lineprops,
                )[0],
            )
            self.cursors = (
                self.axes[0].axvline(
                    self.coords[0][mids[0]], label="X Cursor", **self.cursorprops
                ),
                self.axes[1].axvline(
                    self.coords[0][mids[0]], label="X Cursor", **self.cursorprops
                ),
                self.axes[4].axvline(
                    self.coords[0][mids[0]], label="X Cursor", **self.cursorprops
                ),
                self.axes[0].axhline(
                    self.coords[1][mids[1]], label="Y Cursor", **self.cursorprops
                ),
                self.axes[2].axhline(
                    self.coords[1][mids[1]], label="Y Cursor", **self.cursorprops
                ),
                self.axes[5].axhline(
                    self.coords[1][mids[1]], label="Y Cursor", **self.cursorprops
                ),
                self.axes[3].axvline(
                    self.coords[2][mids[2]], label="Z Cursor", **self.cursorprops
                ),
                self.axes[5].axvline(
                    self.coords[2][mids[2]], label="Z Cursor", **self.cursorprops
                ),
                self.axes[4].axhline(
                    self.coords[2][mids[2]], label="Z Cursor", **self.cursorprops
                ),
            )
            self.spans = (
                (
                    self.axes[0].axvspan(
                        self.coords[0][self._last_ind[0]],
                        self.coords[0][self._last_ind[0]],
                        label="X Span",
                        **self.spanprops,
                    ),
                    self.axes[1].axvspan(
                        self.coords[0][self._last_ind[0]],
                        self.coords[0][self._last_ind[0]],
                        label="X Span",
                        **self.spanprops,
                    ),
                    self.axes[4].axvspan(
                        self.coords[0][self._last_ind[0]],
                        self.coords[0][self._last_ind[0]],
                        label="X Span",
                        **self.spanprops,
                    ),
                ),
                (
                    self.axes[0].axhspan(
                        self.coords[1][self._last_ind[1]],
                        self.coords[1][self._last_ind[1]],
                        label="Y Span",
                        **self.spanprops,
                    ),
                    self.axes[2].axhspan(
                        self.coords[1][self._last_ind[1]],
                        self.coords[1][self._last_ind[1]],
                        label="Y Span",
                        **self.spanprops,
                    ),
                    self.axes[5].axhspan(
                        self.coords[1][self._last_ind[1]],
                        self.coords[1][self._last_ind[1]],
                        label="Y Span",
                        **self.spanprops,
                    ),
                ),
                (
                    self.axes[3].axvspan(
                        self.coords[2][self._last_ind[2]],
                        self.coords[2][self._last_ind[2]],
                        label="Z Span",
                        **self.spanprops,
                    ),
                    self.axes[5].axvspan(
                        self.coords[2][self._last_ind[2]],
                        self.coords[2][self._last_ind[2]],
                        label="Z Span",
                        **self.spanprops,
                    ),
                    self.axes[4].axhspan(
                        self.coords[2][self._last_ind[2]],
                        self.coords[2][self._last_ind[2]],
                        label="Z Span",
                        **self.spanprops,
                    ),
                ),
            )
            self.scaling_axes = (
                self.axes[1].yaxis,
                self.axes[2].xaxis,
                self.axes[3].yaxis,
            )
            if self.lims[-1][-1] * self.lims[-1][0] < 0:
                axes[3].axvline(0.0, label="Fermi Level", **self.fermilineprops)
            self.ax_index = (
                0,
                4,
                5,  # images
                1,
                2,
                3,  # profiles
                0,
                1,
                4,
                0,
                2,
                5,
                3,
                5,
                4,  # cursors
                1,
                2,
                3,
            )  # axes with dynamic limits
            self.span_ax_index = ((0, 1, 4), (0, 2, 5), (3, 5, 4))
            self._only_axis = (
                (
                    False,
                    False,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ),
                (
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                ),
                (
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ),
            )
            self.axes[3].set_xlabel(self.labelify(self.dims[2]))
            self.axes[4].set_ylabel(self.labelify(self.dims[2]))
            self.axes[5].set_xlabel(self.labelify(self.dims[2]))
            self.axes[3].xaxis.set_label_position("top")
            self.axes[3].set_xlim(self.lims[2])
            self.axes[4].set_ylim(self.lims[2])
            self.axes[5].set_xlim(self.lims[2])
            self.axes[3].set_yticks([])
            self.axes[3].ticklabel_format(
                axis="y", style="sci", scilimits=(-2, 3), useMathText=False
            )

        self.all = self.maps + self.hists + self.cursors

        self.axes[0].set_xlabel(self.labelify(self.dims[0]))
        self.axes[0].set_ylabel(self.labelify(self.dims[1]))
        self.axes[1].set_xlabel(self.labelify(self.dims[0]))
        self.axes[2].set_ylabel(self.labelify(self.dims[1]))
        self.axes[1].xaxis.set_label_position("top")
        self.axes[2].yaxis.set_label_position("right")
        self.axes[0].set_xlim(self.lims[0])
        self.axes[0].set_ylim(self.lims[1])
        self.axes[1].set_yticks([])
        self.axes[2].set_xticks([])
        self.axes[1].ticklabel_format(
            axis="y", style="sci", scilimits=(-2, 3), useMathText=False
        )
        self.axes[2].ticklabel_format(
            axis="x", style="sci", scilimits=(-2, 3), useMathText=False
        )

        if self.bench:
            self.counter = 0.0
            self.fps = 0.0
            self.lastupdate = time.time()

        if self.parallel:
            self.pool = Parallel(n_jobs=-1, require="sharedmem", verbose=0)

        self.averaged = [
            False,
        ] * self.ndim
        self.visible = True
        self.background = None
        self.needclear = False

        self.connect()
        # self.axes[1].xaxis.get_major_ticks()[0].label1.set_visible(False)
        # self.axes[3].xaxis.get_major_ticks()[-1].label1.set_visible(False)

    def connect(self):
        """Connect events."""
        self._cidmotion = self.canvas.mpl_connect("motion_notify_event", self.onmove)
        self._ciddraw = self.canvas.mpl_connect("draw_event", self.clear)
        self._cidpress = self.canvas.mpl_connect("key_press_event", self.onpress)
        self._cidrelease = self.canvas.mpl_connect("key_release_event", self.onrelease)

    def disconnect(self):
        """Disconnect events."""
        self.canvas.mpl_disconnect(self._cidmotion)
        self.canvas.mpl_disconnect(self._ciddraw)
        self.canvas.mpl_disconnect(self._cidpress)
        self.canvas.mpl_disconnect(self._cidrelease)

    def clear(self, event):
        """Clear objects and save background."""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
            for obj in self.all:
                obj.set_visible(False)
            for axis in range(self.ndim):
                for span in self.spans[axis]:
                    span.set_visible(False)
        for ax in self.scaling_axes:
            ax.set_ticks([])

    def labelify(self, dim):
        """Prettify some frequently used axis labels."""
        labelformats = {
            "kx": "$k_x$",
            "ky": "$k_y$",
            "kz": "$k_z$",
            "alpha": "$\\alpha$",
            "beta": "$\\beta$",
            "theta": "$\\theta$",
            "phi": "$\\phi$",
            "chi": "$\\chi$",
            "eV": "$E$",
        }
        try:
            return labelformats[dim]
        except KeyError:
            return dim

    def onpress(self, event):
        if event.key == "shift":
            self._shift = True

    def onrelease(self, event):
        if event.key == "shift":
            self._shift = False

    def _assign_vals_T(self):
        if self.ndim == 2:
            self.vals_T = self.vals.T
        elif self.ndim == 3:
            self.vals_T = np.transpose(self.vals, axes=(1, 2, 0))

    def set_cmap(self, cmap):
        self.cmap = cmap
        for im in self.maps:
            im.set_cmap(self.cmap)
        for obj in self.all:
            obj.set_visible(False)
        self._apply_change()

    def toggle_clim_lock(self, lock):
        if lock:
            self.clim_locked = True
            for i, m in enumerate(self.maps):
                self.clim_list[i] = m.get_clim()
        else:
            self.clim_locked = False

    def set_gamma(self, gamma):
        self.gamma = gamma
        self._apply_change()

    def set_index(self, axis, index):
        self._last_ind[axis] = index
        self.cursor_pos[axis] = self.coords[axis][index]
        self._apply_change(self._only_axis[axis])

    def set_value(self, axis, val):
        self._last_ind[axis] = self.get_index_of_value(axis, val)
        self.cursor_pos[axis] = val
        self._apply_change(self._only_axis[axis])

    def set_cursor_color(self, c):
        for cursor in self.cursors:
            cursor.set_color(c)
        self._apply_change()

    def set_line_color(self, c):
        for line in self.hists:
            line.set_color(c)
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

        if self.averaged[axis]:  # if already averaged
            self.vals = move_mean_centered_multiaxis(
                self.data.values,
                self.avg_win,
                axis_list=get_true_indices(self.averaged),
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
                    zip(self.span_ax_index[axis], self.spans[axis], strict=True)
                ):
                    self.axes[i].draw_artist(span)

    def get_index_of_value(self, axis, val):
        # return np.rint((val-self.lims[axis][0])/self.incs[axis]).astype(int)
        return min(
            np.searchsorted(self.coords[axis] + 0.5 * self.incs[axis], val),
            self.shape[axis] - 1,
        )

    def onmove(self, event):
        if self.ignore(event):
            return
        if not event.button:
            if not self._shift:
                return
        if event.inaxes not in self.axes:
            return
        if not self.canvas.widgetlock.available(self):
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
            cond = (
                dz,
                dy,
                dx,
                dy or dz,
                dx or dz,
                dx or dy,
                dx,
                dx,
                dx,
                dy,
                dy,
                dy,
                dz,
                dz,
                dz,
            )
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
            self.pool(delayed(self.set_data)(i) for i in get_true_indices(cond))
            self.pool(delayed(a.set_visible)(self.visible) for a in self.all)
            self._update()
        else:
            for i in get_true_indices(cond):
                self.set_data(i)
            for a in self.all:
                a.set_visible(self.visible)
            self._update()

    def _update(self):
        for ax in self.scaling_axes:
            ax.set_major_locator(AutoLocator())
            ax.axes.relim()
            ax.axes.autoscale_view()
        for i, m in enumerate(self.maps):
            m.set_norm(colors.PowerNorm(self.gamma))
            if self.clim_locked:
                m.set_clim(self.clim_list[i])

        if self.useblit:
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
                for i, art in list(
                    zip(self.ax_index, self.all + self.scaling_axes, strict=True)
                ):
                    self.axes[i].draw_artist(art)
            if any(self.averaged):
                self.update_spans()
            self.canvas.blit()
        else:
            if any(self.averaged):
                self.update_spans()
            self.canvas.draw_idle()

    def print_time(self):
        now = time.time()
        dt = now - self.lastupdate
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = f"Mean Frame Rate:  {self.fps:.3f} FPS"
        print(tx, end="\r")

    def set_data(self, i):
        if self.ndim == 2:
            self.set_data_2d(i)
        elif self.ndim == 3:
            self.set_data_3d(i)

    def set_data_2d(self, i):
        if i == 0:
            self.all[i].set_data(self.vals_T)
        elif i == 1:
            self.all[i].set_ydata(self.vals_T[self._last_ind[1], :])
        elif i == 2:
            self.all[i].set_xdata(self.vals_T[:, self._last_ind[0]])
        elif i in [3, 4]:
            self.all[i].set_xdata((self.cursor_pos[0], self.cursor_pos[0]))
        elif i in [5, 6]:
            self.all[i].set_ydata((self.cursor_pos[1], self.cursor_pos[1]))

    def set_data_3d(self, i):
        if i == 0:
            self.all[i].set_data(self.vals_T[:, self._last_ind[2], :])
        elif i == 1:
            self.all[i].set_data(self.vals_T[self._last_ind[1], :, :])
        elif i == 2:
            self.all[i].set_data(self.vals_T[:, :, self._last_ind[0]])
        elif i == 3:
            self.all[i].set_ydata(self.vals[:, self._last_ind[1], self._last_ind[2]])
        elif i == 4:
            self.all[i].set_xdata(self.vals[self._last_ind[0], :, self._last_ind[2]])
        elif i == 5:
            self.all[i].set_ydata(self.vals[self._last_ind[0], self._last_ind[1], :])
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
        pass

    def _onselectpath(self, verts):
        print(verts)


class ImageToolNavBar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        self.parent = parent
        NavigationToolbar2QT.__init__(self, canvas, parent, coordinates=coordinates)

    def _icon(self, name):
        """Construct a `.QIcon` from an image file name.

        The name must include the extension and be given relative to Matplotlib's
        "images" data directory.
        """
        name = name.replace(".png", "")
        icons_dict = {
            # back = qta.icon('ph.arrow-arc-left-fill'),
            # forward = qta.icon('ph.arrow-arc-right-fill'),
            # filesave = qta.icon('ph.floppy-disk-fill'),
            # home = qta.icon('ph.corners-out-fill'),
            # move = qta.icon('ph.arrows-out-cardinal-fill'),
            # qt4_editor_options = qta.icon('ph.palette-fill'),
            # zoom_to_rect = qta.icon('ph.crop-fill'),
            # subplots = qta.icon('ph.squares-four-fill'),
            "back": qta.icon("msc.chevron-left"),
            "forward": qta.icon("msc.chevron-right"),
            "filesave": qta.icon("msc.save"),
            "home": qta.icon("msc.debug-step-back"),
            "move": qta.icon("msc.move"),
            "qt4_editor_options": qta.icon("msc.graph-line"),
            "zoom_to_rect": qta.icon("msc.search"),
            "subplots": qta.icon("msc.editor-layout"),
        }
        return icons_dict[name]
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


class ImageToolColors(QtWidgets.QDialog):
    def __init__(self, parent):
        self.parent = parent
        super().__init__(self.parent)
        self.setWindowTitle("Colors")

        self.cursor_default = color_to_QColor(self.parent.itool.cursorprops["color"])
        self.line_default = color_to_QColor(self.parent.itool.lineprops["color"])
        self.cursor_current = color_to_QColor(self.parent.itool.cursors[0].get_color())
        self.line_current = color_to_QColor(self.parent.itool.hists[0].get_color())

        if (self.cursor_default.getRgbF() == self.cursor_current.getRgbF()) & (
            self.line_default.getRgbF() == self.line_current.getRgbF()
        ):
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
        else:
            buttons = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.RestoreDefaults
                | QtWidgets.QDialogButtonBox.Ok
                | QtWidgets.QDialogButtonBox.Cancel
            )
            buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(
                self.reset_colors
            )
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        cursorlabel = QtWidgets.QLabel("Cursors:")
        linelabel = QtWidgets.QLabel("Lines:")
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


class ImageTool(QtWidgets.QMainWindow):
    def __init__(self, data, *args, **kwargs):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)
        self.figure = Figure(
            figsize=(10, 10), dpi=70, frameon=True, layout="constrained"
        )
        self.data = parse_data(data)
        self.ndim = self.data.ndim
        if self.ndim == 3:
            gs = self.figure.add_gridspec(
                3, 3, width_ratios=(6, 4, 2), height_ratios=(2, 4, 6)
            )
            self._main_axes = self.figure.add_subplot(gs[2, 0])
            self.axes = [
                self._main_axes,
                self.figure.add_subplot(gs[0, 0], sharex=self._main_axes),
                self.figure.add_subplot(gs[2, 2], sharey=self._main_axes),
                self.figure.add_subplot(gs[:2, 1:]),
                self.figure.add_subplot(gs[1, 0], sharex=self._main_axes),
                self.figure.add_subplot(gs[2, 1], sharey=self._main_axes),
            ]
            self.axes[3].set_label("Z Profile Axes")
            self.axes[4].set_label("Horiz Slice Axes")
            self.axes[5].set_label("Vert Slice Axes")
            self.axes[4].label_outer()
            self.axes[5].label_outer()
            self.axes[3].xaxis.tick_top()
            self.axes[3].yaxis.tick_right()
        elif self.ndim == 2:
            gs = self.figure.add_gridspec(
                2, 2, width_ratios=(6, 2), height_ratios=(2, 6)
            )
            self._main_axes = self.figure.add_subplot(gs[1, 0])
            self.axes = [
                self._main_axes,
                self.figure.add_subplot(gs[0, 0], sharex=self._main_axes),
                self.figure.add_subplot(gs[1, 1], sharey=self._main_axes),
            ]
        self.axes[0].set_label("Main Image Axes")
        self.axes[1].set_label("X Profile Axes")
        self.axes[2].set_label("Y Profile Axes")
        self.axes[1].xaxis.tick_top()
        self.axes[2].yaxis.tick_right()

        self.main_canvas = FigureCanvas(self.figure)
        self.itool = mpl_itool(self.main_canvas, self.axes, self.data, *args, **kwargs)
        self.NavBar = ImageToolNavBar
        home_old = self.NavBar.home

        def home_new(self, *args):
            home_old(self, *args)
            axes = self.canvas.figure.axes
            axes[1].set_ylim(auto=True)
            axes[2].set_xlim(auto=True)
            if self.parent.itool.ndim == 3:
                axes[3].set_ylim(auto=True)

        pan_old = self.NavBar.pan

        def pan_new(self, *args):
            if self.mode == _Mode.PAN:
                self.parent.itool.connect()
            else:
                self.parent.itool.disconnect()
            pan_old(self, *args)

        zoom_old = self.NavBar.zoom

        def zoom_new(self, *args):
            if self.mode == _Mode.ZOOM:
                self.parent.itool.connect()
            else:
                self.parent.itool.disconnect()
            zoom_old(self, *args)

        self.NavBar.home = home_new
        self.NavBar.pan = pan_new
        self.NavBar.zoom = zoom_new
        self.addToolBar(
            QtCore.Qt.BottomToolBarArea, self.NavBar(self.main_canvas, self)
        )

        self.icons = {
            "swap": qta.icon("msc.arrow-swap"),
            "lock": qta.icon("msc.lock"),
            "unlock": qta.icon("msc.unlock"),
        }

        self.cursortab = QtWidgets.QWidget()
        cursortab_content = QtWidgets.QHBoxLayout(self.cursortab)
        spinlabels = tuple(
            QtWidgets.QLabel(self.itool.dims[i]) for i in range(self.ndim)
        )
        self._cursor_spin = tuple(
            QtWidgets.QSpinBox(self.cursortab) for i in range(self.ndim)
        )
        self._cursor_dblspin = tuple(
            QtWidgets.QDoubleSpinBox(self.cursortab) for i in range(self.ndim)
        )
        for i in range(self.ndim):
            self._cursor_spin[i].setRange(0, self.itool.shape[i] - 1)
            self._cursor_spin[i].setSingleStep(1)
            self._cursor_spin[i].setValue(self.itool._last_ind[i])
            self._cursor_spin[i].setWrapping(True)
            self._cursor_spin[i].valueChanged.connect(
                lambda v, axis=i: self._cursor_index_changed(axis, v)
            )
            self._cursor_dblspin[i].setRange(*self.itool.lims[i])
            self._cursor_dblspin[i].setSingleStep(self.itool.incs[i])
            self._cursor_dblspin[i].setDecimals(3)
            self._cursor_dblspin[i].setValue(
                self.itool.coords[i][self.itool._last_ind[i]]
            )
            self._cursor_dblspin[i].setWrapping(True)
            self._cursor_dblspin[i].valueChanged.connect(
                lambda v, axis=i: self._cursor_value_changed(axis, v)
            )

        snap_check = QtWidgets.QCheckBox(self.cursortab)
        snap_check.setChecked(self.itool.snap)
        snap_check.stateChanged.connect(self._assign_snap)
        snap_label = QtWidgets.QLabel("Snap to data")
        snap_label.setBuddy(snap_check)

        for i in range(self.ndim):
            cursortab_content.addWidget(spinlabels[i])
            cursortab_content.addWidget(self._cursor_dblspin[i])
            cursortab_content.addWidget(self._cursor_spin[i])
            cursortab_content.addSpacing(20)
        cursortab_content.addStretch()
        cursortab_content.addWidget(snap_check)
        cursortab_content.addWidget(snap_label)

        self.colorstab = QtWidgets.QWidget()
        colorstab_content = QtWidgets.QHBoxLayout(self.colorstab)
        gamma_spin = QtWidgets.QDoubleSpinBox()
        gamma_spin.setToolTip("Colormap Gamma")
        gamma_spin.setSingleStep(0.01)
        gamma_spin.setRange(0.01, 100.0)
        gamma_spin.setValue(self.itool.gamma)
        gamma_spin.valueChanged.connect(self.itool.set_gamma)
        gamma_label = QtWidgets.QLabel("g")
        gamma_label.setBuddy(gamma_spin)
        self._cmap_combo = QtWidgets.QComboBox(self.colorstab)
        self._cmap_combo.setToolTip("Colormap")
        for name in plt.colormaps():
            self._cmap_combo.addItem(QtGui.QIcon(colormap_to_QPixmap(name)), name)
        self._cmap_combo.setCurrentText(self.itool.cmap)
        self._cmap_combo.setIconSize(QtCore.QSize(60, 15))
        self._cmap_combo.currentTextChanged.connect(self.itool.set_cmap)
        cmap_r_button = QtWidgets.QPushButton()
        cmap_r_button.setIcon(self.icons["swap"])
        cmap_r_button.setCheckable(True)
        cmap_r_button.toggled.connect(self.reverse_cmap)
        self._lock_check = QtWidgets.QPushButton(self.colorstab)
        self._lock_check.setIcon(self.icons["unlock"])
        self._lock_check.setCheckable(True)
        self._lock_check.toggled.connect(self._toggle_clim_lock)
        colors_button = QtWidgets.QPushButton("Colors")
        colors_button.clicked.connect(self._color_button_clicked)
        style_combo = QtWidgets.QComboBox(self.colorstab)
        style_combo.setToolTip("Qt Style")
        style_combo.addItems(qt_style_names())
        style_combo.textActivated.connect(change_style)
        style_label = QtWidgets.QLabel("Style:")
        style_label.setBuddy(style_combo)
        style_combo.setCurrentIndex(style_combo.findText("Fusion"))
        self.main_canvas.draw()
        colorstab_content.addWidget(gamma_label)
        colorstab_content.addWidget(gamma_spin)
        colorstab_content.addWidget(self._cmap_combo)
        colorstab_content.addWidget(cmap_r_button)
        colorstab_content.addWidget(self._lock_check)
        colorstab_content.addStretch()
        colorstab_content.addWidget(colors_button)
        colorstab_content.addStretch()
        colorstab_content.addWidget(style_label)
        colorstab_content.addWidget(style_combo)

        self.smoothtab = QtWidgets.QWidget()
        smoothtab_content = QtWidgets.QHBoxLayout(self.smoothtab)
        navg_label = tuple(
            QtWidgets.QLabel(self.itool.dims[i]) for i in range(self.ndim)
        )
        self._navg_spin = tuple(
            QtWidgets.QSpinBox(self.smoothtab) for i in range(self.ndim)
        )
        navg_resetbutton = QtWidgets.QPushButton("Reset")
        navg_resetbutton.clicked.connect(self._navg_reset)
        for i in range(self.ndim):
            self._navg_spin[i].setRange(1, self.itool.shape[i] - 1)
            self._navg_spin[i].setSingleStep(2)
            self._navg_spin[i].setValue(1)
            self._navg_spin[i].setWrapping(False)
            self._navg_spin[i].valueChanged.connect(
                lambda n, axis=i: self._navg_changed(axis, n)
            )
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
        self.tabwidget.addTab(self.cursortab, "Cursor")
        self.tabwidget.addTab(self.colorstab, "Appearance")
        self.tabwidget.addTab(self.smoothtab, "Smoothing")
        # self.tabwidget.addTab(self.pathtab, 'Path')

        self.layout.addWidget(self.main_canvas)
        self.layout.addWidget(self.tabwidget)
        self.main_canvas.mpl_connect("motion_notify_event", self.onmove_super)
        self.main_canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.main_canvas.setFocus()
        self.main_canvas.draw()

    def onmove_super(self, event):
        if event.inaxes not in self.axes:
            return
        if not event.button:
            if not self.itool._shift:
                return
        for i in range(self.ndim):
            self._cursor_spin[i].blockSignals(True)
            self._cursor_spin[i].setValue(self.itool._last_ind[i])
            self._cursor_spin[i].blockSignals(False)
            self._cursor_dblspin[i].blockSignals(True)
            self._cursor_dblspin[i].setValue(
                self.itool.coords[i][self.itool._last_ind[i]]
            )
            self._cursor_dblspin[i].blockSignals(False)

    def reverse_cmap(self, v):
        if v:
            self.itool.set_cmap(plt.get_cmap(self._cmap_combo.currentText()).reversed())
        else:
            self.itool.set_cmap(plt.get_cmap(self._cmap_combo.currentText()))

    def _toggle_clim_lock(self, v):
        if v:
            self._lock_check.setIcon(self.icons["lock"])
        else:
            self._lock_check.setIcon(self.icons["unlock"])
        self.itool.toggle_clim_lock(v)

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


def itoolmpl(data, *args, **kwargs):
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    # print(qapp.devicePixelRatio())
    mpl_style = "default"
    with plt.rc_context(
        {
            "text.usetex": False,
            # 'font.family':'SF Pro',
            # 'font.size':8,
            # 'font.stretch':'condensed',
            # 'mathtext.fontset':'cm',
            # 'font.family':'fantasy',
        }
    ):
        with plt.style.context(mpl_style):
            app = ImageTool(data, *args, **kwargs)
    change_style("Fusion")
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
