"""Utilities related to manipulating colors.

"""
from collections.abc import Iterable, Sequence

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot as plt

__all__ = [
    "InversePowerNorm",
    "TwoSlopePowerNorm",
    "CenteredPowerNorm",
    "TwoSlopeInversePowerNorm",
    "CenteredInversePowerNorm",
    "get_mappable",
    "proportional_colorbar",
    "nice_colorbar",
    "color_distance",
    "close_to_white",
    "prominent_color",
    "image_is_light",
    "axes_textcolor",
]


class InversePowerNorm(mcolors.PowerNorm):
    r"""
    Linearly map a given value to the 0-1 range and then apply an inverse power-law
    normalization over that range.

    For values :math:`x`, `matplotlib.colors.PowerNorm` calculates
    :math:`x^\gamma`, whereas `InversePowerNorm` calculates :math:`1-x^{1/\gamma}`.
    This provides higher contrast for values closer to ``vmin``.

    Parameters
    ----------
    gamma
        Power law normalization parameter. If equal to 1, the colormap is linear.
    vmin, vmax
        If ``vmin`` and/or ``vmax`` is not given, they are initialized from the
        minimum and maximum value, respectively, of the first input
        processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``
    clip
        If ``True`` values falling outside the range ``[vmin, vmax]``,
        are mapped to 0 or 1, whichever is closer, and masked values are
        set to 1.  If ``False`` masked values remain masked.

        Clipping silently defeats the purpose of setting the over, under,
        and masked colors in a colormap, so it is likely to lead to
        surprises; therefore the default is ``clip=False``.

    """

    def __init__(
        self,
        gamma: float,
        vmin: float | None = None,
        vmax: float | None = None,
        clip: bool = False,
    ):
        super().__init__(gamma, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(
                    np.clip(result.filled(vmax), vmin, vmax), mask=mask
                )

            resdat = result.data

            resdat *= -1
            resdat += vmax
            resdat /= vmax - vmin
            resdat[resdat < 0] = 0
            np.power(resdat, 1.0 / gamma, resdat)
            resdat *= -1
            resdat += 1

            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax
        if np.iterable(value):
            val = np.ma.asarray(value)
            return np.ma.power(1 - val, gamma) * (vmin - vmax) + vmax
        else:
            return pow(1 - value, gamma) * (vmin - vmax) + vmax


def _diverging_powernorm(result, gamma, vmin, vmax, vcenter):
    resdat = result.data
    resdat_ = resdat.copy()

    resdat_l = resdat[resdat_ < vcenter]
    resdat_l -= vcenter
    resdat_l[resdat_l >= 0] = 0
    np.power(-resdat_l, gamma, resdat_l)
    resdat_l /= (vcenter - vmin) ** gamma
    resdat_l *= -0.5
    resdat_l += 0.5
    resdat[resdat_ < vcenter] = resdat_l

    resdat_u = resdat[resdat_ >= vcenter]
    resdat_u -= vcenter
    resdat_u[resdat_u < 0] = 0
    np.power(resdat_u, gamma, resdat_u)
    resdat_u /= (vmax - vcenter) ** gamma
    resdat_u *= 0.5
    resdat_u += 0.5
    resdat[resdat_ >= vcenter] = resdat_u

    result = np.ma.array(resdat, mask=result.mask, copy=False)
    return result


def _diverging_powernorm_inv(value, gamma, vmin, vmax, vcenter):
    if np.iterable(value):
        val = np.ma.asarray(value)
        val_ = val.copy()
        val_l = val_[val < 0.5]
        val_u = val_[val >= 0.5]
        val_[val < 0.5] = (
            np.ma.power(1 - 2 * val_l, 1.0 / gamma) * (vmin - vcenter) + vcenter
        )
        val_[val >= 0.5] = (
            np.ma.power(2 * val_u - 1, 1.0 / gamma) * (vmax - vcenter) + vcenter
        )
        return np.ma.asarray(val_)
    else:
        if value < 0.5:
            return pow(1 - 2 * value, 1.0 / gamma) * (vmin - vcenter) + vcenter
        else:
            return pow(2 * value - 1, 1.0 / gamma) * (vmax - vcenter) + vcenter


def _diverging_inversepowernorm(result, gamma, vmin, vmax, vcenter):
    resdat = result.data
    resdat_ = resdat.copy()

    resdat_l = resdat[resdat_ < vcenter]
    resdat_l *= -1
    resdat_l += vmin
    resdat_l /= vmin - vcenter
    resdat_l[resdat_l < 0] = 0
    np.power(resdat_l, 1.0 / gamma, resdat_l)
    resdat_l *= 0.5
    resdat[resdat_ < vcenter] = resdat_l

    resdat_u = resdat[resdat_ >= vcenter]
    resdat_u *= -1
    resdat_u += vmax
    resdat_u /= vmax - vcenter
    resdat_u[resdat_u < 0] = 0
    np.power(resdat_u, 1.0 / gamma, resdat_u)
    resdat_u *= -0.5
    resdat_u += 1
    resdat[resdat_ >= vcenter] = resdat_u

    result = np.ma.array(resdat, mask=result.mask, copy=False)
    return result


def _diverging_inversepowernorm_inv(value, gamma, vmin, vmax, vcenter):
    if np.iterable(value):
        val = np.ma.asarray(value)
        val_ = val.copy()
        val_l = val_[val < 0.5]
        val_u = val_[val >= 0.5]
        val_[val < 0.5] = np.ma.power(2 * val_l, gamma) * (vcenter - vmin) + vmin
        val_[val >= 0.5] = np.ma.power(2 - 2 * val_u, gamma) * (vcenter - vmax) + vmax
        return np.ma.asarray(val_)
    else:
        if value < 0.5:
            return pow(2 * value, gamma) * (vcenter - vcenter) + vmin
        else:
            return pow(2 - 2 * value, gamma) * (vcenter - vmax) + vmax


class TwoSlopePowerNorm(mcolors.TwoSlopeNorm):
    r"""Power-law normalization of data with a set center.

    Useful when mapping data with an unequal rates of change around a
    conceptual center, e.g., data that range from -2 to 4, with 0 as
    the midpoint.

    Parameters
    ----------
    gamma
        Power law exponent.
    vcenter
        The data value that defines ``0.5`` in the normalization.
        Defaults to ``0``.
    vmin
        The data value that defines ``0.0`` in the normalization.
        Defaults to the min value of the dataset.
    vmax
        The data value that defines ``1.0`` in the normalization.
        Defaults to the max value of the dataset.

    """

    def __init__(
        self,
        gamma: float,
        vcenter: float = 0.0,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        super().__init__(vcenter=vcenter, vmin=vmin, vmax=vmax)
        self.gamma = gamma
        self._func = _diverging_powernorm
        self._func_i = _diverging_powernorm_inv

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
            result = self._func(result, gamma, vmin, vmax, vcenter)
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        gamma = self.gamma
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        (vcenter,), _ = self.process_value(self.vcenter)
        return self._func_i(value, gamma, vmin, vmax, vcenter)


class CenteredPowerNorm(mcolors.CenteredNorm):
    r"""Power-law normalization of symmetrical data around a center.

    Unlike `TwoSlopePowerNorm`, `CenteredPowerNorm` applies an equal rate of
    change around the center.

    Useful when mapping symmetrical data around a conceptual center e.g., data that
    range from -2 to 4, with 0 as the midpoint, and with equal rates of change
    around that midpoint.

    Parameters
    ----------
    gamma
        Power law exponent.
    vcenter
        The data value that defines ``0.5`` in the normalization. Defaults to ``0``.
    halfrange
        The range of data values that defines a range of ``0.5`` in the
        normalization, so that `vcenter` - `halfrange` is ``0.0`` and `vcenter` +
        `halfrange` is ``1.0`` in the normalization. Defaults to the largest
        absolute difference to `vcenter` for the values in the dataset.
    clip
        If ``True`` values falling outside the range ``[vmin, vmax]``,
        are mapped to 0 or 1, whichever is closer, and masked values are
        set to 1.  If ``False`` masked values remain masked.

        Clipping silently defeats the purpose of setting the over, under,
        and masked colors in a colormap, so it is likely to lead to
        surprises; therefore the default is ``clip=False``.

    """

    def __init__(
        self,
        gamma: float,
        vcenter: float = 0,
        halfrange: float | None = None,
        clip: bool = False,
    ):
        super().__init__(vcenter=vcenter, halfrange=halfrange, clip=clip)
        self.gamma = gamma
        self._func = _diverging_powernorm
        self._func_i = _diverging_powernorm_inv

    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1].
        """
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vcenter, vmax = self.vmin, self.vcenter, self.vmax
        if not vmin <= vcenter <= vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        if vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(
                    np.clip(result.filled(vmax), vmin, vmax), mask=mask
                )
            result = self._func(result, gamma, vmin, vmax, vcenter)
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        gamma = self.gamma
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        (vcenter,), _ = self.process_value(self.vcenter)
        return self._func_i(value, gamma, vmin, vmax, vcenter)


class TwoSlopeInversePowerNorm(TwoSlopePowerNorm):
    r"""Inverse power-law normalization of data with a set center.

    Useful when mapping data with an unequal rates of change around a
    conceptual center, e.g., data that range from -2 to 4, with 0 as
    the midpoint.

    Parameters
    ----------
    gamma
        Power law exponent.
    vcenter
        The data value that defines ``0.5`` in the normalization.
        Defaults to ``0``.
    vmin
        The data value that defines ``0.0`` in the normalization.
        Defaults to the min value of the dataset.
    vmax
        The data value that defines ``1.0`` in the normalization.
        Defaults to the max value of the dataset.

    """

    def __init__(
        self,
        gamma: float,
        vcenter: float = 0.0,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        super().__init__(gamma, vcenter, vmin, vmax)
        self._func = _diverging_inversepowernorm
        self._func_i = _diverging_inversepowernorm_inv


class CenteredInversePowerNorm(CenteredPowerNorm):
    r"""Inverse power-law normalization of symmetrical data around a center.

    Unlike `TwoSlopeInversePowerNorm`, `CenteredInversePowerNorm` applies an
    equal rate of change around the center.

    Useful when mapping symmetrical data around a conceptual center e.g., data that
    range from -2 to 4, with 0 as the midpoint, and with equal rates of change
    around that midpoint.

    Parameters
    ----------
    gamma
        Power law exponent.
    vcenter
        The data value that defines ``0.5`` in the normalization. Defaults to ``0``.
    halfrange
        The range of data values that defines a range of ``0.5`` in the
        normalization, so that `vcenter` - `halfrange` is ``0.0`` and `vcenter` +
        `halfrange` is ``1.0`` in the normalization. Defaults to the largest
        absolute difference to `vcenter` for the values in the dataset.
    clip
        If ``True`` values falling outside the range ``[vmin, vmax]``,
        are mapped to 0 or 1, whichever is closer, and masked values are
        set to 1.  If ``False`` masked values remain masked.

        Clipping silently defeats the purpose of setting the over, under,
        and masked colors in a colormap, so it is likely to lead to
        surprises; therefore the default is ``clip=False``.

    """

    def __init__(
        self,
        gamma: float,
        vcenter: float = 0,
        halfrange: float | None = None,
        clip: bool = False,
    ):
        super().__init__(gamma, vcenter, halfrange, clip)
        self._func = _diverging_inversepowernorm
        self._func_i = _diverging_inversepowernorm_inv


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


def proportional_colorbar(
    mappable: matplotlib.cm.ScalarMappable | None = None,
    cax: matplotlib.axes.Axes | None = None,
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    **kwargs: dict
) -> matplotlib.colorbar.Colorbar:
    r"""Replaces the current colorbar or creates a new colorbar with proportional spacing.

    The default behavior of colorbars in `matplotlib` does not support colors
    proportional to data in different norms. This function circumvents this behavior.

    Parameters
    ----------
    mappable
        The `matplotlib.cm.ScalarMappable` described by this colorbar.
    cax
        Axes into which the colorbar will be drawn.
    ax
        One or more parent axes from which space for a new colorbar axes
        will be stolen, if `cax` is `None`.  This has no effect if `cax`
        is set. If `mappable` is `None` and `ax` is given with more than
        one Axes, the function will try to infer the mappable from the
        first one.
    **kwargs
        Extra arguments to `matplotlib.pyplot.colorbar`: refer to the `matplotlib` documentation for a list of all possible arguments.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar.

    Examples
    --------
    ::

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        # Create example data and plot
        X, Y = np.mgrid[0:3:complex(0, 100), 0:2:complex(0, 100)]
        pcm = plt.pcolormesh(X, Y, (1 + np.sin(Y * 10.)) * X**2,
                             norm=mcolors.PowerNorm(gamma=0.5),
                             cmap='Blues_r', shading='auto')

        # Plot evenly spaced colorbar
        proportional_colorbar()

    """
    fontsize = kwargs.pop("fontsize", None)

    if ax is None:
        if cax is None:
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
    if cax is None:
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
    ax:matplotlib.axes.Axes, mappable:matplotlib.cm.ScalarMappable|None=None, width:float=5.0, aspect:float=5.0, minmax:bool=False, ticklabels:Sequence[str]|None=None, *args, **kwargs:dict
):
    r"""Creates a colorbar with fixed width and aspect to ensure uniformity of plots.

    Parameters
    ----------
    ax
        The `matplotlib.axes.Axes` instance in which the colorbar is drawn.
    mappable
        The mappable whose colormap and norm will be used.
    width
        The width of the colorbar in points.
    aspect
        aspect ratio of the colorbar.
    minmax
        If `False`, the ticks and the ticklabels will be determined from the keyword
        arguments (the default). If `True`, the minimum and maximum of the colorbar will
        be labeled.
    **kwargs
        Keyword arguments are passed to `proportional_colorbar`.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar.

    """
    if isinstance(ax, np.ndarray):
        parents = list(ax.flat)
    elif not isinstance(ax, list):
        parents = [ax]
    fig = parents[0].get_figure()

    bbox = matplotlib.transforms.Bbox.union(
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
    R1, G1, B1 = (np.array(mcolors.to_rgb(c1)) * 255).astype(int)
    R2, G2, B2 = (np.array(mcolors.to_rgb(c2)) * 255).astype(int)
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
        return mcolors.to_rgba("w")
    hist, edges = np.histogram(np.nan_to_num(im_array), "auto")
    mx = hist.argmax()
    return im.to_rgba(edges[mx : mx + 2].mean())


def image_is_light(im):
    return close_to_white(prominent_color(im))


def axes_textcolor(ax, light="k", dark="w"):
    c = light
    mappable = get_mappable(ax, error=False)
    if mappable is not None:
        if isinstance(
            mappable, (matplotlib.image._ImageBase, matplotlib.collections.QuadMesh)
        ):
            if not image_is_light(mappable):
                c = dark
    return c
