import pkgutil
from io import StringIO
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


__all__ = [
    "TwoSlopePowerNorm",
    "get_mappable",
    "proportional_colorbar",
    "nice_colorbar",
    "color_distance",
    "close_to_white",
    "prominent_color",
    "image_is_light",
    "axes_textcolor",
]


class TwoSlopePowerNorm(colors.Normalize):
    def __init__(self, gamma, vcenter=None, vmin=None, vmax=None):
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        gamma : float
            Power law exponent
        vcenter : float, optional
            The data value that defines ``0.5`` in the normalization.
            Defaults to the average between `vmin` and `vmax`.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the max value of the dataset.

        """

        super().__init__(vmin=vmin, vmax=vmax)
        self._vcenter = vcenter
        if vcenter is not None and vmax is not None and vcenter >= vmax:
            raise ValueError("vmin, vcenter, and vmax must be in " "ascending order")
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError("vmin, vcenter, and vmax must be in " "ascending order")
        self.gamma = gamma

    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vcenter, vmax = self.vmin, self.vcenter, self.vmax
        if not vmin <= vcenter <= vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        if vmin == vmax:
            result.fill(0)
        else:
            resdat = result.data
            resdat_ = resdat.copy()
            resdat_l = resdat[resdat_ < vcenter]
            resdat_u = resdat[resdat_ >= vcenter]
            resdat_l -= vcenter
            resdat_u -= vcenter
            resdat_l[resdat_l >= 0] = 0
            resdat_u[resdat_u < 0] = 0
            np.power(resdat_u, gamma, resdat_u)
            np.power(-resdat_l, gamma, resdat_l)
            resdat_u /= (vmax - vcenter) ** gamma
            resdat_l /= (vcenter - vmin) ** gamma
            resdat_u *= 0.5
            resdat_u += 0.5
            resdat_l *= -0.5
            resdat_l += 0.5
            resdat[resdat_ < vcenter] = resdat_l
            resdat[resdat_ >= vcenter] = resdat_u
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result

    @property
    def vcenter(self):
        return self._vcenter

    @vcenter.setter
    def vcenter(self, value):
        if value != self._vcenter:
            self._vcenter = value
            self._changed()

    def autoscale_None(self, A):
        """
        Get vmin and vmax, and then clip at vcenter
        """
        super().autoscale_None(A)
        if self.vcenter is None:
            self.vcenter = (self.vmin + self.vmax) / 2
        if self.vmin > self.vcenter:
            self.vmin = self.vcenter
        if self.vmax < self.vcenter:
            self.vmax = self.vcenter

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        gamma = self.gamma
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        (vcenter,), _ = self.process_value(self.vcenter)
        if np.iterable(value):
            val = np.ma.asarray(value)
            val_ = val.copy()
            val_l = val[val_ < 0.5]
            val_u = val[val_ >= 0.5]
            val[val_ < 0.5] = (
                np.ma.power(1 - 2 * val_l, 1.0 / gamma) * (vmin - vcenter) + vcenter
            )
            val[val_ >= 0.5] = (
                np.ma.power(2 * val_u - 1, 1.0 / gamma) * (vmax - vcenter) + vcenter
            )
            return np.ma.asarray(val)
        else:
            if value < 0.5:
                return pow(1 - 2 * value, 1.0 / gamma) * (vmin - vcenter) + vcenter
            else:
                return pow(2 * value - 1, 1.0 / gamma) * (vmax - vcenter) + vcenter


def get_mappable(ax, image_only=False, error=True):
    try:
        if not image_only:
            mappable = ax.collections[-1]
        else:
            raise AttributeError
    except (IndexError, AttributeError):
        try:
            mappable = ax.get_images()[-1]
        except (IndexError, AttributeError):
            mappable = None
    if error is True and mappable is None:
        raise RuntimeError(
            "No mappable was found to use for colorbar "
            "creation. First define a mappable such as "
            "an image (with imshow) or a contour set ("
            "with contourf)."
        )
    return mappable


def proportional_colorbar(mappable=None, cax=None, ax=None, **kwargs):
    r"""Replaces the current colorbar or creates a new colorbar with
    proportional spacing.

    The default behavior of colorbars in `matplotlib` does not support
    colors proportional to data in different norms. This function
    circumvents this behavior.

    Parameters
    ----------
    mappable : `matplotlib.cm.ScalarMappable`, optional
        The `matplotlib.cm.ScalarMappable` described by this colorbar.

    cax : `matplotlib.axes.Axes`, optional
        Axes into which the colorbar will be drawn.

    ax : `matplotlib.axes.Axes`, list of Axes, optional
        One or more parent axes from which space for a new colorbar axes
        will be stolen, if `cax` is None.  This has no effect if `cax`
        is set. If `mappable` is None and `ax` is given with more than
        one Axes, the function will try to get the mappable from the
        first one.

    **kwargs : dict, optional
        Extra arguments to `matplotlib.pyplot.colorbar`: refer to the
        `matplotlib` documentation for a list of all possible arguments.

    Returns
    -------
    cbar : `matplotlib.colorbar.Colorbar`

    Examples
    --------
    ::

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        # Create example data and plot
        X, Y = np.mgrid[0:3:complex(0, 100), 0:2:complex(0, 100)]
        pcm = plt.pcolormesh(X, Y, (1 + np.sin(Y * 10.)) * X**2,
                             norm=colors.PowerNorm(gamma=0.5),
                             cmap='Blues_r', shading='auto')

        # Plot evenly spaced colorbar
        proportional_colorbar()

    """
    fontsize = kwargs.pop("fontsize", None)

    # if cax is None:
    if ax is None:
        ax = plt.gca()
        ax_ref = ax
    else:
        try:
            ax_ref = ax.flatten()[0]
        except AttributeError:
            ax_ref = ax
    # else:
    #     ax_ref = cax
    if mappable is None:
        mappable = get_mappable(ax_ref)
    if mappable.colorbar is None:
        plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)
    ticks = mappable.colorbar.get_ticks()
    mappable.colorbar.remove()
    kwargs.setdefault("ticks", ticks)
    kwargs.setdefault("cmap", mappable.cmap)
    kwargs.setdefault("norm", mappable.norm)
    kwargs.setdefault("pad", 0.05)
    kwargs.setdefault("fraction", 0.05)
    kwargs.setdefault("aspect", 25)
    
    cbar = plt.colorbar(
        mappable=mappable,
        cax=cax,
        ax=ax,
        spacing="proportional",
        boundaries=kwargs["norm"].inverse(np.linspace(0, 1, kwargs["cmap"].N)),
        **kwargs,
    )
    if fontsize is not None:
        cbar.ax.tick_params(labelsize=fontsize)
    return cbar

# TODO: fix colorbar size properly
def nice_colorbar(
    ax, mappable=None, width=5, aspect=5, minmax=False, ticklabels=None, *args, **kwargs
):
    r"""
    Creates a colorbar with fixed width and aspect to ensure uniformity of plots.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The `matplotlib.axes.Axes` instance in which the colorbar is drawn.

    mappable : `.ScalarMappable`, optional
        The mappable whose colormap and norm will be used.

    width : float, default: 5
        The width of the colorbar in points.

    aspect : float, default: 5
        aspect ratio of the colorbar.

    minmax : bool
        If *False* the ticks and the ticklabels will be determined from the keyword
        arguments (the default). If *True* the minimum and maximum of the colorbar will
        be labeled.

    **kwargs
        Keyword arguments are passed to `erlab.plotting.proportional_colorbar`.

    Returns
    -------
    colorbar : matplotlib.colorbar.Colorbar

    """
    if isinstance(ax, np.ndarray):
        parents = list(ax.flat)
    elif not isinstance(ax, list):
        parents = [ax]
    fig = parents[0].get_figure()

    bbox = mpl.transforms.Bbox.union(
        [p.get_position(original=True).frozen() for p in parents]
    ).transformed(fig.transFigure + fig.dpi_scale_trans.inverted())
    # bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fraction = width / (72 * bbox.width)
    shrink = width * aspect / (72 * bbox.height)

    cbar = proportional_colorbar(
        mappable=mappable,
        ax=ax,
        fraction=fraction,
        pad=0.5 * fraction,
        shrink=shrink,
        aspect=aspect,
        anchor=(0, 1),
        panchor=(0, 1),
        *args,
        **kwargs,
    )
    if minmax:
        cbar.set_ticks(cbar.ax.get_ylim())
        cbar.set_ticklabels(("Min", "Max"))
        cbar.ax.tick_params(labelsize="small")
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    return cbar


def color_distance(c1, c2):
    # https://www.compuphase.com/cmetric.htm
    R1, G1, B1 = (np.array(colors.to_rgb(c1)) * 255).astype(int)
    R2, G2, B2 = (np.array(colors.to_rgb(c2)) * 255).astype(int)
    dR2 = (R2 - R1) ** 2
    dG2 = (G2 - G1) ** 2
    dB2 = (B2 - B1) ** 2
    r = 0.5 * (R1 + R2) / 256
    return np.sqrt((2 + r) * dR2 + 4 * dG2 + (2 + 255 / 256 - r) * dB2)


def close_to_white(c):
    c2k = color_distance(c, (0, 0, 0))
    c2w = color_distance(c, (1, 1, 1))
    if c2k > c2w:
        return True
    else:
        return False


def prominent_color(im):
    im_array = im.get_array()
    if im_array is None:
        return colors.to_rgba("w")
    hist, edges = np.histogram(np.nan_to_num(im_array), "auto")
    mx = hist.argmax()
    return im.to_rgba(edges[mx : mx + 2].mean())


def image_is_light(im):
    return close_to_white(prominent_color(im))

def axes_textcolor(ax, light="k", dark="w"):
    c = light
    mappable = get_mappable(ax, error=False)
    if mappable is not None:
        if isinstance(mappable, (mpl.image._ImageBase, mpl.collections.QuadMesh)):
            if not image_is_light(mappable):
                c = dark
    return c
    