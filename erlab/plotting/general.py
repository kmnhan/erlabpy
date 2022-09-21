"""General plotting utilities."""

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.transforms import ScaledTranslation
from matplotlib.widgets import AxesWidget
import matplotlib.projections

from .annotations import label_subplot_properties, fancy_labels
from .colors import get_mappable, image_is_light, proportional_colorbar, nice_colorbar

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


def figwh(ratio=0.75, wide=0, wscale=1, style="aps", fixed_height=True):
    w = figure_width_ref[style][wide]
    if fixed_height:
        h = w * ratio
    else:
        h = w * wscale * ratio
    return w * wscale, h


class LabeledCursor(AxesWidget):
    """
    A crosshair cursor that spans the axes and moves with mouse cursor.
    For the cursor to remain responsive you must keep a reference to it.
    Unlike `matplotlib.widgets.Cursor`, this also shows the current
    cursor location.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.
    horizOn : bool, default: True
        Whether to draw the horizontal line.
    vertOn : bool, default: True
        Whether to draw the vertical line.
    textOn : bool, default: True
        Whether to show current cursor location.
    useblit : bool, default: False
        Use blitting for faster drawing if supported by the backend.
    textprops : dict, default: {}
        Keyword arguments to pass onto the text object.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.
    """

    def __init__(
        self,
        ax,
        horizOn=True,
        vertOn=True,
        textOn=True,
        useblit=True,
        textprops={},
        **lineprops
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
        lw = lineprops.pop("lw", lineprops.pop("linewidth", 0.8))
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
    parent_axes, width, height, pad=0.1, loc="upper right", polar=False, **kwargs
):
    """Place inset axes, top right of parent

    Parameters
    ----------
    parent_axes : matplotlib.axes.Axes
        Axes to place the inset axes.
    width, height : float or str
        Size of the inset axes to create. If a float is provided, it is
        the size in inches, e.g. `width=1.3`. If a string is provided, it is
        the size in relative units, e.g. `width='40%'` to the parent_axes.
    pad : float or 2-tuple of floats, optional
        Pad between parent axes and inset in inches, by default 0.1
    loc : str, default: 'upper right'
        Location to place the inset axes. Valid locations are
        'upper left', 'upper center', 'upper right',
        'center left', 'center', 'center right',
        'lower left', 'lower center, 'lower right'.
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
        print(prect)
        return fig.add_axes(prect, projection="polar", **kwargs)


def array_extent(data):
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
    ax=None,
    colorbar_kw=dict(),
    cursor=False,
    cursor_kw=dict(),
    xlim=None,
    ylim=None,
    func=None,
    rad2deg=False,
    func_args=dict(),
    **improps
):
    """Plots a 2D `xr.DataArray` using imshow, which is much faster."""
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

    improps.setdefault("cmap", "magma")
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
        c = LabeledCursor(ax, **cursor_kw)
        return img, c
    else:
        return img


def plot_slices(
    maps,
    figsize=None,
    transpose=False,
    xlim=None,
    ylim=None,
    axis="auto",
    show_all_labels=False,
    colorbar="none",
    hide_colorbar_ticks=True,
    annotate=True,
    order="C",
    cmap_order="C",
    norm_order=None,
    subplot_kw=dict(),
    annotate_kw=dict(),
    colorbar_kw=dict(),
    **values
):
    r"""Automated comparison plot of slices.

    Parameters
    ----------

    maps : `xarray.DataArray`, list of DataArray
        Arrays to compare.

    figsize : 2-tuple of floats, default: plt.rcParams["figure.figsize"]

    transpose : bool, default=False
        Transpose each map before plotting.

    xlim : float or (float, float)

    ylim : float or (float, float)

    axis : {'equal', 'tight', 'scaled', 'auto', 'image', 'square'}

    show_all_labels : bool, default=False

    colorbar : {'right', 'rightspan', 'all', 'none'}, optional

    hide_colorbar_ticks : bool, default=True

    annotate : bool, default=True

    order : {'C', 'F'}, optional
    cmap_order : {'C', 'F'}, optional
    norm_order : {'C', 'F'}, optional
        Same as `cmap_order` by default.

    subplot_kw : dict, optional
        Extra arguments to `matplotlib.pyplot.subplots`: refer to the
        `matplotlib` documentation for a list of all possible arguments.

    annotate_kw : dict, optional
        Extra arguments to
        `erlab.plotting.annotations.label_subplot_properties`.

    colorbar_kw : dict, optional
        Extra arguments to
        `erlab.plotting.general.proportional_colorbar`.

    **values : dict
        key-value pair of cut location and bin widths. See examples.
        Remaining arguments are passed onto
        `erlab.plotting.general.plot_array`.

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

    cmap_name = kwargs.pop("cmap", None)
    cmap_norm = kwargs.pop("norm", None)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, **subplot_kw)
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
            **annotate_kw
        )
    return fig, axes


def fermiline(ax=None, y=0, **kwargs):
    if ax is None:
        ax = plt.gca()
    default_color = "k"
    mappable = get_mappable(ax, error=False)
    if mappable is not None:
        if isinstance(mappable, (mpl.image._ImageBase, mpl.collections.QuadMesh)):
            if not image_is_light(mappable):
                default_color = "w"
    c = kwargs.pop("color", kwargs.pop("c", default_color))
    lw = kwargs.pop("lw", kwargs.pop("linewidth", 0.25))
    ls = kwargs.pop("ls", kwargs.pop("linestyle", "-"))
    return ax.axhline(y, ls=ls, lw=lw, c=c)
