"""General plotting utilities."""

from __future__ import annotations

__all__ = [
    "figwh",
    "autoscale_off",
    "autoscale_to",
    "LabeledCursor",
    "place_inset",
    "plot_array",
    "plot_array_2d",
    "gradient_fill",
    "plot_slices",
    "fermiline",
]

import contextlib
import copy
from typing import TYPE_CHECKING, Literal

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.image
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from matplotlib.widgets import AxesWidget

from erlab.plotting.annotations import fancy_labels, label_subplot_properties
from erlab.plotting.colors import (
    InversePowerNorm,
    _ez_inset,
    axes_textcolor,
    gen_2d_colormap,
    nice_colorbar,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

figure_width_ref = {
    "aps": [3.4, 7.0],
    "aip": [3.37, 6.69],
    "nature": [88 / 25.4, 180 / 25.4],
}


def figwh(ratio=0.6180339887498948, wide=0, wscale=1, style="aps", fixed_height=True):
    if isinstance(ratio, str):
        ratio = float(ratio) * 2 / (1 + np.sqrt(5))
    w = figure_width_ref[style][wide]
    if fixed_height:
        h = w * ratio
    else:
        h = w * wscale * ratio
    return w * wscale, h


@contextlib.contextmanager
def autoscale_off(ax: matplotlib.axes.Axes | None = None):
    if ax is None:
        ax = plt.gca()
    xauto, yauto = ax.get_autoscalex_on(), ax.get_autoscaley_on()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    try:
        yield
    finally:
        ax.autoscale(enable=xauto, axis="x")
        ax.autoscale(enable=yauto, axis="y")


def autoscale_to(arr, margin=0.2):
    mn, mx = min(arr), max(arr)
    diff = margin * (mx - mn)
    return mn - diff, mx + diff


class LabeledCursor(AxesWidget):
    r"""A crosshair cursor that spans the axes and moves with mouse cursor.

    For the cursor to remain responsive you must keep a reference to it.
    Unlike `matplotlib.widgets.Cursor`, this also shows the current
    cursor location.

    Parameters
    ----------
    ax
        The `matplotlib.axes.Axes` to attach the cursor to.
    horizOn
        Whether to draw the horizontal line.
    vertOn
        Whether to draw the vertical line.
    textOn
        Whether to show current cursor location.
    useblit
        Use blitting for faster drawing if supported by the backend.
    textprops
        Keyword arguments to pass onto the text object.

    Other Parameters
    ----------------
    **lineprops
        `matplotlib.lines.Line2D` properties that control the appearance of the
        lines. See also `matplotlib.axes.Axes.axhline`.

    """

    def __init__(
        self,
        ax: matplotlib.axes.Axes,
        horizOn: bool = True,
        vertOn: bool = True,
        textOn: bool = True,
        useblit: bool = True,
        textprops: dict | None = None,
        **lineprops: dict,
    ):
        super().__init__(ax)

        if textprops is None:
            textprops = {}

        self.connect_event("motion_notify_event", self.onmove)
        self.connect_event("draw_event", self.clear)

        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.textOn = textOn
        self.useblit = useblit and self.canvas.supports_blit

        if self.useblit:
            lineprops["animated"] = True
            textprops["animated"] = True

        lcolor = lineprops.pop("color", lineprops.pop("c", "0.35"))
        ls = lineprops.pop("ls", lineprops.pop("linestyle", "--"))
        lw = lineprops.pop("lw", lineprops.pop("linewidth", 0.5))
        lineprops.update({"color": lcolor, "ls": ls, "lw": lw, "visible": False})

        tcolor = textprops.pop("color", textprops.pop("c", lcolor))
        textprops.update(
            {
                "color": tcolor,
                "visible": False,
                "horizontalalignment": "right",
                "verticalalignment": "top",
                "transform": ax.transAxes,
            }
        )

        self.lineh = ax.axhline(ax.get_ybound()[0], **lineprops)
        self.linev = ax.axvline(ax.get_xbound()[0], **lineprops)
        with plt.rc_context({"text.usetex": False}):
            self.label = ax.text(0.95, 0.95, "", **textprops)
        self.background = None
        self.needclear = False

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(False)
        self.lineh.set_visible(False)
        self.label.set_visible(False)

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)
            self.label.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return
        self.linev.set_xdata((event.xdata, event.xdata))
        self.lineh.set_ydata((event.ydata, event.ydata))
        self.label.set_text(f"({event.xdata:1.3f}, {event.ydata:1.3f})")
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)
        self.label.set_visible(self.visible and self.textOn)

        self._update()

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.ax.draw_artist(self.label)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False


def place_inset(
    parent_axes: matplotlib.axes.Axes,
    width: float | str,
    height: float | str,
    pad: float | tuple[float, float] = 0.1,
    loc: Literal[
        "upper left",
        "upper center",
        "upper right",
        "center left",
        "center",
        "center right",
        "lower left",
        "lower center",
        "lower right",
    ] = "upper right",
    **kwargs: dict,
) -> matplotlib.axes.Axes:
    """Easy placement of inset axes.

    Parameters
    ----------
    parent_axes
        `matplotlib.axes.Axes` to place the inset axes.
    width, height
        Size of the inset axes to create. If `float`, specifies the size in inches, e.g.
        ``1.3``. If `str`, specifies the size in relative units, e.g. ``'40%'`` of
        `parent_axes`.
    pad
        Padding between `parent_axes` and inset in inches.
    loc
        Location to place the inset axes.
    **kwargs
        Keyword arguments are passed onto `matplotlib.axes.Axes.inset_axes`.

    Returns
    -------
    matplotlib.axes.Axes


    """

    return _ez_inset(parent_axes, width, height, pad, loc, **kwargs)


def array_extent(data: xr.DataArray) -> tuple[float, float, float, float]:
    """
    Gets the extent of a :class:`xarray.DataArray` to be used in
    :func:`matplotlib.pyplot.imshow`.

    Parameters
    ----------
    data
        A two-dimensional :class:`xarray.DataArray`.

    Returns
    -------
    x0, x1, y0, y1 : float

    """

    data_coords = tuple(data[dim].values for dim in data.dims)
    data_incs = tuple(coord[1] - coord[0] for coord in data_coords)
    data_lims = tuple((coord[0], coord[-1]) for coord in data_coords)
    y0, x0 = data_lims[0][0] - 0.5 * data_incs[0], data_lims[1][0] - 0.5 * data_incs[1]
    y1, x1 = (
        data_lims[0][-1] + 0.5 * data_incs[0],
        data_lims[1][-1] + 0.5 * data_incs[1],
    )
    return x0, x1, y0, y1


def plot_array(
    arr: xr.DataArray,
    ax: matplotlib.axes.Axes | None = None,
    *,
    colorbar: bool = False,
    colorbar_kw: dict | None = None,
    gamma: float = 1.0,
    norm: mcolors.Normalize | None = None,
    xlim: float | tuple[float, float] | None = None,
    ylim: float | tuple[float, float] | None = None,
    crop: bool = False,
    rad2deg: bool | Iterable[str] = False,
    func: Callable | None = None,
    func_args: dict | None = None,
    **improps: dict,
) -> matplotlib.image.AxesImage:
    """Plots a 2D :class:`xarray.DataArray` using :func:`matplotlib.pyplot.imshow`.

    Parameters
    ----------
    arr
        A two-dimensional :class:`xarray.DataArray` to be plotted.
    ax
        The target :class:`matplotlib.axes.Axes`.
    colorbar
        Whether to plot a colorbar.
    colorbar_kw
        Keyword arguments passed onto :func:`erlab.plotting.colors.nice_colorbar`.
    xlim, ylim
        If given a sequence of length 2, those values are set as the lower and upper
        limits of each axis. If given a single `float`, the limits are set as ``(-lim,
        lim)``.  If `None`, automatically determines the limits from the data.
    rad2deg
        If `True`, converts some known angle coordinates from radians to degrees. If an
        iterable of `str` is given, only the coordinates that correspond to the given
        strings are converted.
    func
        A callable that processes the values prior to display. Its output must have the
        same shape as the input.
    func_args
        Keyword arguments passed onto `func`.
    **improps
        Keyword arguments passed onto :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.image.AxesImage

    """
    if colorbar_kw is None:
        colorbar_kw = {}
    if func_args is None:
        func_args = {}

    if isinstance(arr, np.ndarray):
        arr = xr.DataArray(arr)
    if ax is None:
        ax = plt.gca()
    if xlim is not None and not np.iterable(xlim):
        xlim = (-xlim, xlim)
    if ylim is not None and not np.iterable(ylim):
        ylim = (-ylim, ylim)
    if rad2deg is not False:
        if np.iterable(rad2deg):
            conv_dims = rad2deg
        else:
            conv_dims = [
                d for d in ["phi", "theta", "beta", "alpha", "chi"] if d in arr.dims
            ]
        arr = arr.assign_coords({d: np.rad2deg(arr[d]) for d in conv_dims})

    norm_kw = {}
    if "vmin" in improps:
        norm_kw["vmin"] = improps.pop("vmin")
        if "vmax" in improps:
            norm_kw["vmax"] = improps.pop("vmax")
            colorbar_kw.setdefault("extend", "both")
        else:
            colorbar_kw.setdefault("extend", "min")
    elif "vmax" in improps:
        norm_kw["vmax"] = improps.pop("vmax")
        colorbar_kw.setdefault("extend", "max")

    if norm is None:
        norm = copy.deepcopy(mcolors.PowerNorm(gamma, **norm_kw))

    improps_default = {
        "interpolation": "none",
        "extent": array_extent(arr),
        "aspect": "auto",
        "origin": "lower",
        "rasterized": True,
    }
    for k, v in improps_default.items():
        improps.setdefault(k, v)

    if crop:
        if xlim is not None:
            arr = arr.copy(deep=True).sel({arr.dims[1]: slice(*xlim)})
        if ylim is not None:
            arr = arr.copy(deep=True).sel({arr.dims[0]: slice(*ylim)})
    if func is not None:
        img = ax.imshow(func(arr.values, **func_args), norm=norm, **improps)
    else:
        img = ax.imshow(arr.values, norm=norm, **improps)
    ax.set_xlabel(arr.dims[1])
    ax.set_ylabel(arr.dims[0])
    fancy_labels(ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if colorbar:
        nice_colorbar(ax=ax, **colorbar_kw)
    return img


def plot_array_2d(
    larr: xr.DataArray,
    carr: xr.DataArray,
    ax: matplotlib.axes.Axes | None = None,
    *,
    xlim: float | tuple[float, float] | None = None,
    ylim: float | tuple[float, float] | None = None,
    cmap: mcolors.Colormap | str = None,
    lnorm: mcolors.Normalize | None = None,
    cnorm: mcolors.Normalize | None = None,
    background: Sequence[float] | None = None,
    colorbar: bool = True,
    cax: matplotlib.axes.Axes | None = None,
    colorbar_kw: dict | None = None,
    imshow_kw: dict | None = None,
    **indexers_kwargs: dict,
):
    if lnorm is None:
        lnorm = plt.Normalize()
    else:
        lnorm = copy.deepcopy(lnorm)
    if cnorm is None:
        cnorm = plt.Normalize()
    else:
        cnorm = copy.deepcopy(cnorm)
    if colorbar_kw is None:
        colorbar_kw = {}
    if imshow_kw is None:
        imshow_kw = {}
    improps_default = {
        "interpolation": "none",
        "aspect": "equal",
        "origin": "lower",
        "rasterized": True,
    }
    for k, v in improps_default.items():
        imshow_kw.setdefault(k, v)
    if ax is None:
        ax = plt.gca()

    larr = larr.qsel(**indexers_kwargs).copy(deep=True)
    carr = carr.qsel(**indexers_kwargs).copy(deep=True)
    sel_kw = {}
    if xlim is not None:
        if not np.iterable(xlim):
            xlim = (-xlim, xlim)
        sel_kw[larr.dims[1]] = slice(*xlim)
    if ylim is not None:
        if not np.iterable(ylim):
            ylim = (-ylim, ylim)
        sel_kw[larr.dims[0]] = slice(*ylim)
    larr = larr.sel(**sel_kw)
    carr = carr.sel(**sel_kw)

    cmap_img, img = gen_2d_colormap(
        larr.values, carr.values, cmap, lnorm=lnorm, cnorm=cnorm, background=background
    )

    if colorbar:
        if cax is None:
            colorbar_kw.setdefault("aspect", 2)
            colorbar_kw.setdefault("anchor", (0, 1))
            colorbar_kw.setdefault("panchor", (0, 1))
            cb = ax.get_figure().colorbar(plt.cm.ScalarMappable(), ax=ax, **colorbar_kw)
            cax = cb.ax
            cax.clear()

        cax.imshow(
            cmap_img.transpose(1, 0, 2),
            extent=(lnorm.vmin, lnorm.vmax, cnorm.vmin, cnorm.vmax),
            origin="lower",
        )

    im = ax.imshow(img, extent=array_extent(larr), **imshow_kw)
    ax.set_xlabel(larr.dims[0])
    ax.set_ylabel(larr.dims[1])

    if colorbar:
        return im, cb
    else:
        return im, None


def gradient_fill(
    x: Sequence[int | float],
    y: Sequence[int | float],
    y0: float | None = None,
    color: str | tuple[float, float, float] | tuple[float, float, float, float] = "C0",
    cmap: str | mcolors.Colormap | None = None,
    transpose: bool = False,
    reverse: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs: dict,
) -> matplotlib.image.AxesImage:
    """Applies a gradient fill to a line plot.

    Parameters
    ----------
    x, y
        Data of the plot to fill under.
    y0
        The minimum y value of the gradient. If `None`, defaults to the minimum of `y`.
    color
        A valid matplotlib color to make the gradient from.
    cmap
        If given, ignores `color` and fills with the given colormap.
    transpose
        Transpose the gradient.
    reverse
        Reverse the gradient.
    ax
        The :class:`matplotlib.axes.Axes` to plot in.
    **kwargs
        Keyword arguments passed onto :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.image.AxesImage

    """
    color = kwargs.pop("c", color)
    kwargs.setdefault("norm", InversePowerNorm(0.5))
    kwargs.setdefault("alpha", 0.75)
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", colors=[(1, 1, 1, 0), mcolors.to_rgba(color)], N=1024
        )
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    if reverse:
        cmap = cmap.reversed()

    if ax is None:
        ax = plt.gca()
    x, y = np.asarray(x), np.asarray(y)
    valid_inds = (~np.isnan(x)) * (~np.isnan(y))
    x, y = x[valid_inds], y[valid_inds]

    if y0 is None:
        y0 = min(y)
    xn = np.r_[x[0], x, x[-1]]
    yn = np.r_[y0, y, y0]
    patch = matplotlib.patches.PathPatch(
        matplotlib.path.Path(np.array([xn, yn]).T), edgecolor="none", facecolor="none"
    )
    ax.add_patch(patch)

    im = matplotlib.image.AxesImage(
        ax, cmap=cmap, interpolation="bicubic", origin="lower", zorder=0, **kwargs
    )
    im.use_sticky_edges = False
    ax.add_artist(im)
    if transpose:
        im.set_data(np.linspace(0, 1, 1024).reshape(1024, 1).T)
    else:
        im.set_data(np.linspace(0, 1, 1024).reshape(1024, 1))
    # with autoscale_off(ax):
    im.set_extent((min(xn), max(xn), min(yn), max(yn)))
    im.set_clip_path(patch)

    return im


def plot_slices(
    maps: xr.DataArray | Sequence[xr.DataArray],
    figsize: tuple[float, float] | None = None,
    *,
    transpose: bool = False,
    xlim: float | tuple[float, float] | None = None,
    ylim: float | tuple[float, float] | None = None,
    crop: bool = True,
    same_limits: bool = False,
    axis: Literal[
        "on", "off", "equal", "scaled", "tight", "auto", "image", "scaled", "square"
    ] = "auto",
    show_all_labels: bool = False,
    colorbar: Literal["none", "right", "rightspan", "all"] = "none",
    hide_colorbar_ticks: bool = True,
    annotate: bool = True,
    cmap: str | mcolors.Colormap | Iterable[mcolors.Colormap | str] | None = None,
    norm: mcolors.Normalize | Iterable[mcolors.Normalize] | None = None,
    order: Literal["C", "F"] = "C",
    cmap_order: Literal["C", "F"] = "C",
    norm_order: Literal["C", "F"] | None = None,
    gradient: bool = False,
    gradient_kw: dict | None = None,
    subplot_kw: dict | None = None,
    annotate_kw: dict | None = None,
    colorbar_kw: dict | None = None,
    axes: npt.NDArray[matplotlib.axes.Axes] | None = None,
    **values: dict,
) -> tuple[matplotlib.figure.Figure, npt.NDArray[matplotlib.axes.Axes]]:
    """Automated comparison plot of slices.

    Parameters
    ----------
    maps
        Arrays to plot.
    figsize
        Figure size.
    transpose
        Transpose each map before plotting.
    xlim, ylim
        If given a sequence of length 2, those values are set as the lower and upper
        limits of each axis. If given a single `float`, the limits are set as ``(-lim,
        lim)``.  If `None`, automatically determines the limits from the data.
    crop
        If `True`, crops the data to the limits given by `xlim` and `ylim` prior to
        plotting.
    same_limits
        If `True`, all images will have the same vmin and vmax.
    axis
        Passed onto :func:`matplotlib.axes.Axes.axis`. Possible values are:

        ======== ============================================================
        Value    Description
        ======== ============================================================
        'on'     Turn on axis lines and labels.
        'off'    Turn off axis lines and labels.
        'equal'  Set equal scaling (i.e., make circles circular) by
                 changing axis limits. This is the same as ``ax.set_aspect('equal',
                 adjustable='datalim')``. Explicit data limits may not be respected in
                 this case.
        'scaled' Set equal scaling (i.e., make circles circular) by changing dimensions
                 of the plot box. This is the same as ``ax.set_aspect('equal',
                 adjustable='box', anchor='C')``. Additionally, further autoscaling will
                 be disabled.
        'tight'  Set limits just large enough to show all data, then
                 disable further autoscaling.
        'auto'   Automatic scaling (fill plot box with data).
        'image'  'scaled' with axis limits equal to data limits.
        'square' Square plot; similar to 'scaled', but initially forcing
                 ``xmax-xmin == ymax-ymin``.
        ======== ============================================================
    show_all_labels
        If `True`, shows every xlabel and ylabel. If `False`, labels on shared axes are
        minimized. When `False` and the `axes` argument is given, the `order` must be
        specified to correctly hide shared labels.
    colorbar
        Controls colorbar behavior. Possible values are:

        =========== =========================================================
        Value       Description
        =========== =========================================================
        'none'      Do not show colorbars.
        'right'     Creates a colorbar on the right for each row.
        'rightspan' Create a single colorbar that spans all axes.
        'all'       Plot a colorbar for every axes.
        =========== =========================================================
    hide_colorbar_ticks
        If `True`, hides colorbar ticks.
    annotate
        If `False`, turn off automatic annotation.
    cmap
        If supplied a single `str` or :class:`matplotlib.colors.Colormap`, the colormap
        is applied to all axes. Otherwise, a nested sequence with the same shape as the
        resulting axes can be provided to use different colormaps for different axes. If
        the slices are 1D, this argument can be used to supply valid colors as line
        colors for differnet slices.
    norm
        If supplied a single :class:`matplotlib.colors.Normalize`, the norm is applied
        to all axes. Otherwise, a nested sequence with the same shape as the resulting
        axes can be provided to use different norms for different axes.
    order
        Order to display the data. Effectively, this determines if each map is displayed
        along the same row or the same column. 'C' means to flatten in row-major
        (C-style) order, and 'F' means to flatten in column-major (Fortran-style) order.
    cmap_order
        The order to flatten when given a nested sequence for `cmap`,
        Defaults to `order`.
    norm_order
        The order to flatten when given a nested sequence for `norm`,
        Defaults to `cmap_order`.
    gradient
        If `True`, for 1D slices, fills the area under the curve with a gradient. Has no
        effect for 2D slices.
    gradient_kw
        Extra arguments to :func:`gradient_fill`.
    subplot_kw
        Extra arguments to :func:`matplotlib.pyplot.subplots`: refer to the
        `matplotlib` documentation for a list of all possible arguments.
    annotate_kw
        Extra arguments to :func:`erlab.plotting.annotations.label_subplot_properties`.
        Only applied when `annotate` is `True`.
    colorbar_kw
        Extra arguments to :func:`erlab.plotting.colors.proportional_colorbar`.
    axes
        A nested sequence of :class:`matplotlib.axes.Axes`. If supplied, the returned
        :class:`matplotlib.figure.Figure` is inferred from the first axes.
    **values
        Key-value pair of cut location and bin widths. See examples.
        Remaining arguments are passed onto :func:`plot_array`.

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : array-like of matplotlib.axes.Axes

    Examples
    --------
    ::

        # Two maps: map1, map2
        # Create a figure with a 3 by 2 grid.

        fig, axes = plot_slices([map1, map2], eV=[0, -0.1, -0.2], eV_width=0.05)

    """
    if isinstance(maps, xr.DataArray):
        maps = [maps]

    if transpose:
        maps = [m.T for m in maps]

    if cmap_order is None:
        cmap_order = order

    if norm_order is None:
        norm_order = cmap_order

    if gradient_kw is None:
        gradient_kw = {}

    if subplot_kw is None:
        subplot_kw = {}

    if annotate_kw is None:
        annotate_kw = {}

    if colorbar_kw is None:
        colorbar_kw = {}

    dims = maps[0].dims
    kwargs = {k: v for k, v in values.items() if k not in dims}
    slice_kw = {k: v for k, v in values.items() if k not in kwargs}

    if len(slice_kw) == 0:
        slice_dim = None
        slice_levels = [None]
        slice_width = None

    else:
        slice_dim = next(k for k in slice_kw if not k.endswith("_width"))
        slice_levels = slice_kw[slice_dim]
        slice_width = kwargs.pop(slice_dim + "_width", None)

    plot_dims = [d for d in dims if d != slice_dim]

    if len(plot_dims) not in (1, 2):
        raise ValueError("The data to plot must be 1D or 2D")

    if not np.iterable(slice_levels):
        slice_levels = [slice_levels]

    if xlim is not None and not np.iterable(xlim):
        xlim = (-xlim, xlim)

    if ylim is not None and not np.iterable(ylim):
        ylim = (-ylim, ylim)

    auto_gradient_color = all(k not in gradient_kw for k in ("c", "color"))

    if hide_colorbar_ticks:
        colorbar_kw["ticks"] = []

    for k, v in {"sharex": "col", "sharey": "row", "layout": "constrained"}.items():
        subplot_kw.setdefault(k, v)

    if order == "F":
        nrow, ncol = len(slice_levels), len(maps)

    elif order == "C":
        nrow, ncol = len(maps), len(slice_levels)

    cmap_name = cmap
    cmap_norm = norm
    if axes is None:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, **subplot_kw)

    else:
        fig = axes.flat[0].get_figure()

    if nrow == 1:
        axes = axes[:, np.newaxis].reshape(1, -1)

    if ncol == 1:
        axes = axes[:, np.newaxis].reshape(-1, 1)

    qsel_kw = {}

    if crop:
        if len(plot_dims) == 1:
            if transpose and (ylim is not None):
                qsel_kw[plot_dims[0]] = slice(*ylim)

            elif xlim is not None:
                qsel_kw[plot_dims[0]] = slice(*xlim)

        elif len(plot_dims) == 2:
            if xlim is not None:
                qsel_kw[plot_dims[1]] = slice(*xlim)

            if ylim is not None:
                qsel_kw[plot_dims[0]] = slice(*ylim)

    if slice_width is not None:
        qsel_kw[slice_dim + "_width"] = slice_width

    for i in range(len(slice_levels)):
        if slice_dim is not None:
            qsel_kw[slice_dim] = slice_levels[i]

        for j in range(len(maps)):
            dat_sel = maps[j].copy(deep=True).qsel(**qsel_kw)

            if order == "F":
                ax = axes[i, j]
            elif order == "C":
                ax = axes[j, i]

            if np.iterable(cmap_name) and not isinstance(cmap_name, str):
                if cmap_order == "F":
                    if isinstance(cmap_name[i], str):
                        cmap = cmap_name[i]
                    else:
                        cmap = cmap_name[i][j]
                elif cmap_order == "C":
                    if isinstance(cmap_name[j], str):
                        cmap = cmap_name[j]
                    else:
                        cmap = cmap_name[j][i]
            else:
                cmap = cmap_name

            if len(plot_dims) == 1:
                if cmap is not None:
                    kwargs["color"] = cmap
                    if auto_gradient_color:
                        gradient_kw["color"] = cmap

                if transpose:
                    ax.plot(dat_sel.values, dat_sel[plot_dims[0]], **kwargs)
                    ax.set_xlabel(dat_sel.name)
                    ax.set_ylabel(plot_dims[0])

                    if gradient:
                        gradient_fill(
                            dat_sel.values,
                            dat_sel[plot_dims[0]],
                            ax=ax,
                            transpose=True,
                            **gradient_kw,
                        )

                else:
                    dat_sel.plot(ax=ax, **kwargs)
                    ax.set_title("")

                    if gradient:
                        gradient_fill(
                            dat_sel[plot_dims[0]], dat_sel.values, ax=ax, **gradient_kw
                        )

            elif len(plot_dims) == 2:
                if np.iterable(cmap_norm):
                    if norm_order == "F":
                        try:
                            norm = cmap_norm[i][j]
                        except TypeError:
                            norm = cmap_norm[i]

                    elif norm_order == "C":
                        try:
                            norm = cmap_norm[j][i]
                        except TypeError:
                            norm = cmap_norm[j]
                else:
                    norm = copy.deepcopy(cmap_norm)
                plot_array(dat_sel, ax=ax, norm=norm, cmap=cmap, **kwargs)

    if same_limits and len(plot_dims) == 2:
        vmn, vmx = [], []
        for ax in axes.flat:
            vmn.append(ax.get_images()[0].norm.vmin)
            vmx.append(ax.get_images()[0].norm.vmax)
        vmn, vmx = min(vmn), max(vmx)
        for ax in axes.flat:
            ax.get_images()[0].norm.vmin = vmn
            ax.get_images()[0].norm.vmax = vmx

    for ax in axes.flatten():
        if not show_all_labels:
            if ax not in axes[:, 0]:
                ax.set_ylabel("")
            if ax not in axes[-1, :]:
                ax.set_xlabel("")
        if not gradient:
            ax.axis(axis)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if colorbar not in ["none", "rightspan"]:
            if colorbar == "all" or ax in axes[:, -1]:
                nice_colorbar(mappable=ax.images[0], ax=ax, **colorbar_kw)
    if colorbar == "rightspan":
        nice_colorbar(ax=axes, **colorbar_kw)
    if annotate and slice_dim is not None:
        if slice_dim == "eV":
            slice_dim = "Eb"
        label_subplot_properties(
            axes,
            values={slice_dim: slice_levels * len(maps)},
            order=order,
            **annotate_kw,
        )
    return fig, axes


def fermiline(
    ax: matplotlib.axes.Axes | None = None,
    value: float = 0.0,
    orientation: Literal["h", "v"] = "h",
    **kwargs: dict,
) -> matplotlib.lines.Line2D:
    """Plots a constant energy line to denote the Fermi level.

    Parameters
    ----------
    ax
        The `matplotlib.axes.Axes` to annotate.
    value
        The coordinate of the line. Defaults to 0, assuming binding energy.
    orientation
        If 'h', a horizontal line is plotted. If 'v', a vertical line is plotted.
    **kwargs
        Keyword arguments passed onto `matplotlib.lines.Line2D`.

    Returns
    -------
    matplotlib.lines.Line2D

    """
    if ax is None:
        ax = plt.gca()
    if np.iterable(ax):
        return [fermiline(a, value, orientation, **kwargs) for a in np.asarray(ax).flat]

    c = kwargs.pop("color", kwargs.pop("c", axes_textcolor(ax)))
    lw = kwargs.pop("lw", kwargs.pop("linewidth", 0.25))
    ls = kwargs.pop("ls", kwargs.pop("linestyle", "-"))
    match orientation:
        case "h":
            return ax.axhline(value, ls=ls, lw=lw, c=c, **kwargs)
        case "v":
            return ax.axvline(value, ls=ls, lw=lw, c=c, **kwargs)
        case _:
            raise ValueError("`orientation` must be either 'v' or 'h'")
