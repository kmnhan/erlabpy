"""General plotting utilities.

"""
from __future__ import annotations

from collections.abc import Callable, Sequence, Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib.widgets import AxesWidget

from erlab.plotting.annotations import label_subplot_properties, fancy_labels
from erlab.plotting.colors import axes_textcolor, nice_colorbar

__all__ = [
    "LabeledCursor",
    "place_inset",
    "plot_array",
    "plot_slices",
    "figwh",
    "fermiline",
]

figure_width_ref = dict(
    aps=[3.4, 7.0],
    aip=[3.37, 6.69],
    nature=[89 / 25.4, 183 / 25.4],
)


def figwh(ratio=0.6180339887498948, wide=0, wscale=1, style="aps", fixed_height=True):
    if isinstance(ratio, str):
        ratio = float(ratio) * 2 / (1 + np.sqrt(5))
    w = figure_width_ref[style][wide]
    if fixed_height:
        h = w * ratio
    else:
        h = w * wscale * ratio
    return w * wscale, h


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
        textprops: dict = {},
        **lineprops: dict,
    ):
        super().__init__(ax)

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
        lineprops.update(dict(color=lcolor, ls=ls, lw=lw, visible=False))

        tcolor = textprops.pop("color", textprops.pop("c", lcolor))
        textprops.update(
            dict(
                color=tcolor,
                visible=False,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
            )
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
        self.label.set_text("(%1.3f, %1.3f)" % (event.xdata, event.ydata))
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
    polar: bool = False,
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
    polar
        If `True`, the inset axes will have polar coordinates.
    **kwargs
        Keyword arguments are passed onto `matplotlib.axes.Axes.inset_axes`.

    Returns
    -------
    matplotlib.axes.Axes


    """

    fig = parent_axes.get_figure()
    sizes = [width, height]

    ax_sizes = (
        parent_axes.get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        .bounds[2:]
    )
    from numbers import Number

    for i, size in enumerate(sizes):
        if isinstance(size, Number):
            sizes[i] = size / ax_sizes[i]
        elif isinstance(size, str):
            if size[-1] == "%":
                sizes[i] = float(size[:-1]) / 100
        else:
            raise ValueError("Unknown format")

    bounds = [1 - sizes[0], 1 - sizes[1]]
    pad_num = False
    if isinstance(pad, Number):
        pad_num = True
        pad = [pad, pad]
    pads = [-pad[0], -pad[1]]

    if "center" in loc:
        bounds[0] /= 2
        bounds[1] /= 2
        pads[0] *= -1
        pads[1] *= -1
        if "upper" in loc or "lower" in loc:
            if pad_num:
                pads[0] = 0
            pads[1] *= -1
            bounds[1] *= 2
        elif "left" in loc or "right" in loc:
            if pad_num:
                pads[1] = 0
            pads[0] *= -1
            bounds[0] *= 2
    if "left" in loc:
        bounds[0] = 0
        pads[0] *= -1
    if "lower" in loc:
        bounds[1] = 0
        pads[1] *= -1

    tform = parent_axes.transAxes + ScaledTranslation(*pads, fig.dpi_scale_trans)

    if not polar:
        return parent_axes.inset_axes(bounds + sizes, transform=tform, **kwargs)
    else:
        prect = (fig.transFigure.inverted() + parent_axes.transAxes).transform(
            bounds + sizes
        )
        return fig.add_axes(prect, projection="polar", **kwargs)


def array_extent(data: xr.DataArray) -> tuple[float, float, float, float]:
    """
    Gets the extent of a `xarray.DataArray` to be used in `matplotlib.pyplot.imshow`.

    Parameters
    ----------
    data
        A two-dimensional `xarray.DataArray`.

    Returns
    -------
    tuple of float

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
    colorbar_kw: dict = dict(),
    cursor: bool = False,
    cursor_kw: dict = dict(),
    xlim: float | tuple[float, float] | None = None,
    ylim: float | tuple[float, float] | None = None,
    rad2deg: bool | Iterable[str] = False,
    func: Callable = None,
    func_args: dict = dict(),
    **improps: dict,
) -> tuple[matplotlib.image.AxesImage, LabeledCursor] | matplotlib.image.AxesImage:
    """Plots a 2D `xarray.DataArray` using `matplotlib.pyplot.imshow`.

    Parameters
    ----------
    arr
        A two-dimensional `xarray.DataArray` to be plotted.
    ax
        The target `matplotlib.axes.Axes`.
    colorbar_kw
        Keyword arguments passed onto `erlab.plotting.colors.nice_colorbar`.
    cursor
        Whether to display a dynamic cursor.
    cursor_kw
        Arguments passed onto `erlab.plotting.general.LabeledCursor`. Ignored if `cursor` is `False`.
    xlim, ylim
        If given a sequence of length 2, those values are set as the lower and upper
        limits of each axis. If given a single `float`, the limits are set as ``(-lim,
        lim)``.  If `None`, automatically determines the limits from the data.
    func
        A callable that processes the values prior to displaying. Its output must have
        the same shape as the input.
    func_args
        Keyword arguments passed onto `func`.
    rad2deg
        If `True`, converts some known angle coordinates to degrees. If an iterable of
        `str` is given, only the coordinates that correspond to the given strings are
        converted.
    **improps
        Keyword arguments passed onto `matplotlib.axes.Axes.imshow`.

    Returns
    -------
    img : matplotlib.image.AxesImage

    c : erlab.plotting.general.LabeledCursor

    """
    if isinstance(arr, xr.Dataset):
        arr = arr.spectrum
    elif isinstance(arr, np.ndarray):
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

    improps.setdefault("cmap", "BuWh_r")
    colorbar = improps.pop("colorbar", False)
    gamma = improps.pop("gamma", 1.0)
    try:
        if improps["norm"] is None:
            improps.pop("norm")
    except KeyError:
        pass
    norm_kw = dict()

    if "vmin" in improps.keys():
        norm_kw["vmin"] = improps.pop("vmin")
        if "vmax" in improps.keys():
            norm_kw["vmax"] = improps.pop("vmax")
            colorbar_kw.setdefault("extend", "both")
        else:
            colorbar_kw.setdefault("extend", "min")
    elif "vmax" in improps.keys():
        norm_kw["vmax"] = improps.pop("vmax")
        colorbar_kw.setdefault("extend", "max")

    improps["norm"] = improps.pop("norm", colors.PowerNorm(gamma, **norm_kw))

    improps_default = dict(
        interpolation="none",
        extent=array_extent(arr),
        aspect="auto",
        origin="lower",
        rasterized=True,
    )
    for k, v in improps_default.items():
        improps.setdefault(k, v)

    if func is not None:
        img = ax.imshow(func(arr.values, **func_args), **improps)
    else:
        img = ax.imshow(arr.values, **improps)
    ax.set_xlabel(arr.dims[1])
    ax.set_ylabel(arr.dims[0])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if colorbar:
        nice_colorbar(ax=ax, **colorbar_kw)
    if cursor:
        cursor_kw.setdefault("textOn", False)
        c = LabeledCursor(ax, **cursor_kw)
        return img, c
    else:
        return img


def plot_slices(
    maps: xr.DataArray | Sequence[xr.DataArray],
    figsize: tuple[float, float] | None = None,
    transpose: bool = False,
    xlim: float | tuple[float, float] | None = None,
    ylim: float | tuple[float, float] | None = None,
    axis: Literal[
        "on", "off", "equal", "scaled", "tight", "auto", "image", "scaled", "square"
    ] = "auto",
    show_all_labels: bool = False,
    colorbar: Literal["none", "right", "rightspan", "all"] = "none",
    hide_colorbar_ticks: bool = True,
    annotate: bool = True,
    cmap: str
    | matplotlib.colors.Colormap
    | Iterable[matplotlib.colors.Colormap | str]
    | None = None,
    norm: matplotlib.colors.Normalize
    | Iterable[matplotlib.colors.Normalize]
    | None = None,
    order: Literal["C", "F"] = "C",
    cmap_order: Literal["C", "F"] = "C",
    norm_order: Literal["C", "F"] | None = None,
    subplot_kw: dict = dict(),
    annotate_kw: dict = dict(),
    colorbar_kw: dict = dict(),
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
    axis
        Passed onto `matplotlib.axes.Axes.axis`. Possible values are:

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
        minimized.
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
        If supplied a single `str` or `matplotlib.colors.Colormap`, the colormap is
        applied to all axes. Otherwise, a nested sequence with the same shape as the
        resulting axes can be provided to use different colormaps for different axes.
    norm
        If supplied a single `matplotlib.colors.Normalize`, the norm is applied to all
        axes. Otherwise, a nested sequence with the same shape as the resulting axes can
        be provided to use different norms for different axes.
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
    subplot_kw
        Extra arguments to `matplotlib.pyplot.subplots`: refer to the
        `matplotlib` documentation for a list of all possible arguments.
    annotate_kw
        Extra arguments to `erlab.plotting.annotations.label_subplot_properties`. Only
        applied when `annotate` is `True`.
    colorbar_kw
        Extra arguments to `erlab.plotting.colors.proportional_colorbar`.
    axes
        A nested sequence of `matplotlib.axes.Axes`. If supplied, the returned
        `matplotlib.figure.Figure` is inferred from the first axes.
    **values
        Key-value pair of cut location and bin widths. See examples.
        Remaining arguments are passed onto `plot_array`.

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : array-like of matplotlib.axes.Axes

    Examples
    --------
    ::

        # Two maps: map1, map2
        # Create a figure with a 3 by 2 grid.

        fig, axes = plot_slices([map1, map2],
                                 eV=[0, -0.1, -0.2],
                                 eV_width=0.05)

    """
    if isinstance(maps, xr.DataArray):
        maps = [maps]
    if transpose:
        maps = [m.T for m in maps]
    if cmap_order is None:
        cmap_order = order
    if norm_order is None:
        norm_order = cmap_order
    dims = maps[0].dims

    kwargs = {k: v for k, v in values.items() if k not in dims}
    slice_kw = {k: v for k, v in values.items() if k not in kwargs}
    slice_dim = list(slice_kw.keys())[0]
    slice_levels = slice_kw[slice_dim]
    slice_width = kwargs.pop(slice_dim + "_width", None)

    if not np.iterable(slice_levels):
        slice_levels = [slice_levels]

    if xlim is not None and not np.iterable(xlim):
        xlim = (-xlim, xlim)
    if ylim is not None and not np.iterable(ylim):
        ylim = (-ylim, ylim)

    if hide_colorbar_ticks:
        colorbar_kw["ticks"] = []

    for k, v in dict(sharex="col", sharey="row", layout="constrained").items():
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

    for i in range(len(slice_levels)):
        fatsel_kw = {slice_dim: slice_levels[i]}
        if slice_width is not None:
            fatsel_kw[slice_dim + "_width"] = slice_width
        for j in range(len(maps)):
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
                norm = cmap_norm
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
            if order == "F":
                ax = axes[i, j]
            elif order == "C":
                ax = axes[j, i]
            # if slice_width == 0:
            #     plot_array(
            #         maps[j].sel({slice_dim: slice_levels[i]}, method='nearest'),
            #         ax=ax,
            #         norm=norm,
            #         cmap=cmap,
            #         **kwargs
            #     )
            # else:
            plot_array(
                maps[j].S.fat_sel(**fatsel_kw), ax=ax, norm=norm, cmap=cmap, **kwargs
            )

    for ax in axes.flatten():
        if not show_all_labels:
            if ax not in axes[:, 0]:
                ax.set_ylabel("")
            if ax not in axes[-1, :]:
                ax.set_xlabel("")
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
    fancy_labels(axes)
    if annotate:
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
