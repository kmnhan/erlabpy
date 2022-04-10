"""General plotting utilities."""

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arpes.plotting.utils import fancy_labels
from matplotlib.widgets import AxesWidget

from .annotations import label_subplot_properties
from .colors import get_mappable, image_is_light, proportional_colorbar

__all__ = [
    "proportional_colorbar",
    "LabeledCursor",
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


def figwh(ratio=0.75, wide=0, wscale=1, style="aps"):
    w = figure_width_ref[style][wide]
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


def plot_array(
    arr: xr.DataArray,
    ax=None,
    colorbar_kw=dict(),
    cursor=False,
    cursor_kw=dict(),
    xlim=None,
    ylim=None,
    **improps
):
    """Plots a 2D `xr.DataArray` using imshow, which is much faster."""
    if isinstance(arr, xr.Dataset):
        arr = arr.spectrum
    if ax is None:
        ax = plt.gca()
    if xlim is not None and not np.iterable(xlim):
        xlim = (-xlim, xlim)
    if ylim is not None and not np.iterable(ylim):
        ylim = (-ylim, ylim)
    coords = [arr[d] for d in arr.dims]
    coords.reverse()
    extent = tuple(m for c in coords for m in (c[0], c[-1]))

    improps.setdefault("cmap", "twilight")
    colorbar = improps.pop("colorbar", False)
    gamma = improps.pop("gamma", 0.5)
    try:
        if improps["norm"] is None:
            improps.pop("norm")
    except KeyError:
        pass
    norm_kw = dict()
    try:
        norm_kw["vmin"] = improps.pop("vmin")
        colorbar_kw.setdefault("extend", "min")
    except KeyError:
        pass
    try:
        norm_kw["vmax"] = improps.pop("vmax")
        try:
            norm_kw["vmin"]
            colorbar_kw.setdefault("extend", "both")
        except KeyError:
            colorbar_kw.setdefault("extend", "max")
    except KeyError:
        pass
    improps["norm"] = improps.pop("norm", colors.PowerNorm(gamma, **norm_kw))

    improps_default = dict(
        interpolation="none",
        extent=extent,
        aspect="auto",
        origin="lower",
        rasterized=True,
    )
    for k, v in improps_default.items():
        improps.setdefault(k, v)

    img = ax.imshow(arr.values, **improps)
    ax.set_xlabel(arr.dims[1])
    ax.set_ylabel(arr.dims[0])
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if colorbar:
        proportional_colorbar(ax=ax, **colorbar_kw)
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
    cmap_order=None,
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
        Opposite of `order` by default.
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
        if order == "F":
            cmap_order = "C"
        else:
            cmap_order = "F"
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
                maps[j].S.fat_sel(**fatsel_kw),
                ax=ax,
                norm=norm,
                cmap=cmap,
                **kwargs
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
                proportional_colorbar(ax.images[0], ax=ax, **colorbar_kw)
    if colorbar == "rightspan":
        colorbar_kw.setdefault("shrink", 1.0 / float(nrow))
        colorbar_kw.setdefault("pad", 0.1 / float(ncol))
        proportional_colorbar(ax=axes, **colorbar_kw)
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


def fermiline(ax=None, **kwargs):
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
    return ax.axhline(0, ls=ls, lw=lw, c=c)
