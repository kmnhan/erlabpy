"""Utilities related to manipulating colors.

Colormaps
---------

In addition to the default `matplotlib
<https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_ colormaps, `cmasher
<https://cmasher.readthedocs.io>`_, `cmocean <https://matplotlib.org/cmocean/>`_, and
`colorcet <https://colorcet.holoviz.org>`_ packages can be installed to extend the
available colormaps. If these packages are installed, they will be automatically
imported upon importing `erlab.plotting`.

Colormap Normalization
----------------------

.. plot:: norms.py example_1
   :width: 65 %

   Demonstration of `InversePowerNorm`.

.. plot:: norms.py example_2
   :width: 65 %

   Demonstration of `CenteredPowerNorm` and `CenteredInversePowerNorm`.

"""

__all__ = [
    "InversePowerNorm",
    "TwoSlopePowerNorm",
    "CenteredPowerNorm",
    "TwoSlopeInversePowerNorm",
    "CenteredInversePowerNorm",
    "get_mappable",
    "unify_clim",
    "proportional_colorbar",
    "nice_colorbar",
    "flatten_transparency",
    "gen_2d_colormap",
    "color_distance",
    "close_to_white",
    "prominent_color",
    "image_is_light",
    "axes_textcolor",
]

from collections.abc import Iterable, Sequence
from numbers import Number
from typing import Literal

import matplotlib
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


class InversePowerNorm(mcolors.Normalize):
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
        super().__init__(vmin, vmax, clip)
        self.gamma = gamma

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
    elif value < 0.5:
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
    elif value < 0.5:
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


def get_mappable(
    ax: matplotlib.axes.Axes, image_only: bool = False, silent: bool = False
) -> matplotlib.cm.ScalarMappable | None:
    """Gets the `matplotlib.cm.ScalarMappable` from a given `matplotlib.axes.Axes`.

    Parameters
    ----------
    ax
        Parent axes.
    image_only
        Only consider images as a valid mappable, by default `False`.
    silent
        If `False`, raises a `RuntimeError`. If `True`, silently returns `None`.

    Returns
    -------
    matplotlib.cm.ScalarMappable or None

    """
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
    if not silent and mappable is None:
        raise RuntimeError(
            "No mappable was found to use for colorbar "
            "creation. First define a mappable such as "
            "an image (with imshow) or a contour set ("
            "with contourf)."
        )
    return mappable


def unify_clim(axes, target=None, image_only: bool = False):
    if target is None:
        vmn, vmx = [], []
        for ax in axes.flat:
            mappable = get_mappable(ax, image_only=image_only)
            vmn.append(mappable.norm.vmin)
            vmx.append(mappable.norm.vmax)
        vmn, vmx = min(vmn), max(vmx)
    else:
        mappable = get_mappable(target, image_only=image_only)
        vmn, vmx = mappable.norm.vmin, mappable.norm.vmax
    for ax in axes.flat:
        mappable = get_mappable(ax, image_only=image_only)
        mappable.norm.vmin, mappable.norm.vmax = vmn, vmx


def proportional_colorbar(
    mappable: matplotlib.cm.ScalarMappable | None = None,
    cax: matplotlib.axes.Axes | None = None,
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    **kwargs: dict,
) -> matplotlib.colorbar.Colorbar:
    """
    Replaces the current colorbar or creates a new colorbar with proportional spacing.

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
        Extra arguments to `matplotlib.pyplot.colorbar`: refer to the `matplotlib`
        documentation for a list of all possible arguments.

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
        X, Y = np.mgrid[0 : 3 : complex(0, 100), 0 : 2 : complex(0, 100)]
        pcm = plt.pcolormesh(
            X,
            Y,
            (1 + np.sin(Y * 10.0)) * X**2,
            norm=mcolors.PowerNorm(gamma=0.5),
            cmap="Blues_r",
            shading="auto",
        )

        # Plot evenly spaced colorbar
        proportional_colorbar()

    """
    fontsize = kwargs.pop("fontsize", None)

    if ax is None:
        if cax is None:
            ax = plt.gca()
            if mappable is None:
                mappable = get_mappable(ax)
    elif isinstance(ax, np.ndarray):
        i = 0
        while mappable is None and i < len(ax.flat):
            mappable = get_mappable(ax.flatten()[i], silent=(i != (len(ax.flat) - 1)))
            i += 1
    elif mappable is None:
        mappable = get_mappable(ax)

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


def _size_to_bounds(ax, width, height, loc):
    fig = ax.get_figure()
    sizes = [width, height]

    ax_sizes = (
        ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).bounds[2:]
    )
    sizes = [
        float(size[:-1]) / 100 if isinstance(size, str) else size / ax_sizes[i]
        for i, size in enumerate(sizes)
    ]

    origin = [1 - sizes[0], 1 - sizes[1]]
    if "center" in loc:
        origin[0] /= 2
        origin[1] /= 2
        if "upper" in loc or "lower" in loc:
            origin[1] *= 2
        elif "left" in loc or "right" in loc:
            origin[0] *= 2
    if "left" in loc:
        origin[0] = 0
    if "lower" in loc:
        origin[1] = 0
    return origin + sizes


def _refresh_pads(ax, cax, pads, loc):
    ref = _size_to_bounds(ax, 0, 0, loc)[:2]
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    x0, y0 = (
        bbox.x0 + ref[0] * (bbox.x1 - bbox.x0),
        bbox.y1 + ref[1] * (bbox.y1 - bbox.y0),
    )
    bbox = cax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    x1, y1 = (
        bbox.x0 + ref[0] * (bbox.x1 - bbox.x0),
        bbox.y1 + ref[1] * (bbox.y1 - bbox.y0),
    )

    pads[0] += x1 - x0
    pads[1] += y1 - y0
    return pads


def _get_pad(pad, loc):
    pad_num = False
    if isinstance(pad, Number):
        pad_num = True
        pad = [pad, pad]
    pads = [-pad[0], -pad[1]]
    if "center" in loc:
        pads[0] *= -1
        pads[1] *= -1
        if "upper" in loc or "lower" in loc:
            if pad_num:
                pads[0] = 0
            pads[1] *= -1
        elif "left" in loc or "right" in loc:
            if pad_num:
                pads[1] = 0
            pads[0] *= -1
    if "left" in loc:
        pads[0] *= -1
    if "lower" in loc:
        pads[1] *= -1
    return pads


def _ez_inset(
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
    fig = parent_axes.get_figure()
    locator = InsetAxesLocator(parent_axes, width, height, pad, loc)
    ax_ = fig.add_axes(locator(parent_axes, None).bounds, **kwargs)
    ax_.set_axes_locator(locator)
    return ax_


class InsetAxesLocator:
    def __init__(self, ax, width, height, pad, loc):
        self._ax = ax
        self._transAxes = ax.transAxes
        self._width = width
        self._height = height
        self._loc = loc
        self.set_pad(pad)

    def __call__(self, ax, renderer):
        return mtransforms.TransformedBbox(
            mtransforms.Bbox.from_bounds(*self._size_to_bounds(ax)),
            self._transAxes
            + mtransforms.ScaledTranslation(*self.pads, ax.figure.dpi_scale_trans)
            - ax.figure.transSubfigure,
        )

    def set_pad(self, pad):
        pad_num = False
        if isinstance(pad, Number):
            pad_num = True
            pad = [pad, pad]
        self.pads = [-pad[0], -pad[1]]
        if "center" in self._loc:
            self.pads[0] *= -1
            self.pads[1] *= -1
            if "upper" in self._loc or "lower" in self._loc:
                if pad_num:
                    self.pads[0] = 0
                self.pads[1] *= -1
            elif "left" in self._loc or "right" in self._loc:
                if pad_num:
                    self.pads[1] = 0
                self.pads[0] *= -1
        if "left" in self._loc:
            self.pads[0] *= -1
        if "lower" in self._loc:
            self.pads[1] *= -1

    def add_pad(self, delta):
        self.pads[0] += delta[0]
        self.pads[1] += delta[1]

    def sizes(self, ax):
        ax_sizes = (
            ax.get_window_extent()
            .transformed(ax.figure.dpi_scale_trans.inverted())
            .bounds[2:]
        )
        return [
            float(sz[:-1]) / 100 if isinstance(sz, str) else sz / ax_sizes[i]
            for i, sz in enumerate([self._width, self._height])
        ]

    def _size_to_bounds(self, ax):
        sizes = self.sizes(ax)
        origin = [1 - sizes[0], 1 - sizes[1]]
        if "center" in self._loc:
            origin[0] /= 2
            origin[1] /= 2
            if "upper" in self._loc or "lower" in self._loc:
                origin[1] *= 2
            elif "left" in self._loc or "right" in self._loc:
                origin[0] *= 2
        if "left" in self._loc:
            origin[0] = 0
        if "lower" in self._loc:
            origin[1] = 0
        return origin + sizes


def _gen_cax(ax, width=4.0, aspect=7.0, pad=3.0, horiz=False, **kwargs):
    w, h = width / 72, aspect * width / 72
    if horiz:
        cax = _ez_inset(ax, h, w, pad=(0, -w - pad / 72), **kwargs)
    else:
        cax = _ez_inset(ax, w, h, pad=(-w - pad / 72, 0), **kwargs)
    return cax


# TODO: fix colorbar size properly
def nice_colorbar(
    ax: matplotlib.axes.Axes | None = None,
    mappable: matplotlib.cm.ScalarMappable | None = None,
    width: float = 5.0,
    aspect: float = 5.0,
    pad: float = 3.0,
    minmax: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    floating=False,
    ticklabels: Sequence[str] | None = None,
    **kwargs: dict,
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
    pad
        The pad between the colorbar and axes in points.
    minmax
        If `False`, the ticks and the ticklabels will be determined from the keyword
        arguments (the default). If `True`, the minimum and maximum of the colorbar will
        be labeled.
    orientation
        Colorbar orientation.
    **kwargs
        Keyword arguments are passed to `proportional_colorbar`.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar.

    """

    is_horizontal = orientation == "horizontal"

    if ax is None:
        ax = plt.gca()

    if floating:
        if isinstance(ax, np.ndarray):
            if ax.ndim == 1:
                parent = ax[-1]
            elif ax.ndim == 2:
                parent = ax[0, -1]
            else:
                raise ValueError
        else:
            parent = ax

        cbar = proportional_colorbar(
            mappable=mappable,
            ax=ax,
            cax=_gen_cax(parent, width, aspect, pad, is_horizontal),
            orientation=orientation,
            **kwargs,
        )

    else:
        if np.iterable(ax):
            bbox = mtransforms.Bbox.union(
                [
                    x.get_window_extent().transformed(
                        x.figure.dpi_scale_trans.inverted()
                    )
                    for x in ax.flat
                ]
            )
        else:
            bbox = ax.get_window_extent().transformed(
                ax.figure.dpi_scale_trans.inverted()
            )

        if orientation == "horizontal":
            kwargs["anchor"] = (1, 1)
            kwargs["location"] = "top"
            kwargs["fraction"] = width / (72 * bbox.height)
            kwargs["pad"] = pad / (72 * bbox.height)
            kwargs["shrink"] = width * aspect / (72 * bbox.width)
        else:
            kwargs["anchor"] = (0, 1)
            kwargs["fraction"] = width / (72 * bbox.width)
            kwargs["pad"] = pad / (72 * bbox.width)
            kwargs["shrink"] = width * aspect / (72 * bbox.height)

        cbar = proportional_colorbar(
            mappable=mappable,
            ax=ax,
            aspect=aspect,
            panchor=(0, 1),
            orientation=orientation,
            **kwargs,
        )

    if minmax:
        if is_horizontal:
            cbar.set_ticks(cbar.ax.get_xlim())
        else:
            cbar.set_ticks(cbar.ax.get_ylim())
        cbar.set_ticklabels(("Min", "Max"))
        cbar.ax.tick_params(labelsize="small")

    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)

    if is_horizontal:
        cbar.ax.set_box_aspect(1 / aspect)
    else:
        cbar.ax.set_box_aspect(aspect)

    return cbar


def flatten_transparency(rgba, background: Sequence[float] | None = None):
    if background is None:
        background = (1, 1, 1)
    original_shape = rgba.shape
    rgba = rgba.reshape(-1, 4)
    rgb = rgba[:, :-1]
    a = rgba[:, -1][:, np.newaxis]
    rgb *= a
    rgb += (1 - a) * background
    return rgb.reshape(original_shape[:-1] + (3,))


def _is_segment_iterable(cmap: mcolors.Colormap) -> bool:
    if not isinstance(cmap, mcolors.LinearSegmentedColormap):
        return False
    if any(callable(cmap._segmentdata[c]) for c in ["red", "green", "blue"]):
        return False
    return True


def combined_cmap(
    cmap1: mcolors.Colormap | str,
    cmap2: mcolors.Colormap | str,
    name: str,
    register: bool = False,
    N=256,
):
    """Stitch two existing colormaps to create a new colormap."""
    if isinstance(cmap1, str):
        cmap1 = matplotlib.colormaps[cmap1]
    if isinstance(cmap2, str):
        cmap2 = matplotlib.colormaps[cmap2]

    if all(_is_segment_iterable(c) for c in (cmap1, cmap2)):
        segnew = {}
        for c in ["red", "green", "blue"]:
            seg1_c, seg2_c = (
                np.asarray(cmap1._segmentdata[c]),
                np.asarray(cmap2._segmentdata[c]),
            )
            seg1_c[:, 0] = seg1_c[:, 0] * 0.5
            seg2_c[:, 0] = seg2_c[:, 0] * 0.5 + 0.5
            segnew[c] = np.r_[seg1_c, seg2_c]
        cmap = mcolors.LinearSegmentedColormap(name=name, segmentdata=segnew, N=N)
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            name=name,
            colors=np.r_[
                cmap1(np.linspace(0, 1, int(N / 2))),
                cmap2(np.linspace(0, 1, int(N / 2))),
            ],
        )
    if register:
        matplotlib.colormaps.register(cmap)
        matplotlib.colormaps.register(cmap.reversed())
    return cmap


def gen_2d_colormap(
    ldat,
    cdat,
    cmap: mcolors.Colormap | str = None,
    colorbar: bool = True,
    *,
    lnorm: plt.Normalize | None = None,
    cnorm: plt.Normalize | None = None,
    background: Sequence[float] | None = None,
    N: int = 256,
):
    if cmap is None:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", colors=[[0, 0, 1], [0, 0, 0], [1, 0, 0]], N=N
        )
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if lnorm is None:
        lnorm = plt.Normalize()
    if cnorm is None:
        cnorm = plt.Normalize()
    if background is None:
        background = (1, 1, 1)

    lnorm.autoscale_None(ldat)
    cnorm.autoscale_None(cdat)

    l_vals = lnorm(ldat)[:, :, np.newaxis]
    img = cmap(cnorm(cdat))[:, :, :-1]
    img *= l_vals
    img += (1 - l_vals) * background

    if colorbar:
        l_vals = lnorm(np.linspace(lnorm.vmin, lnorm.vmax, N))[
            :, np.newaxis, np.newaxis
        ]
        cmap_img = np.repeat(
            cmap(cnorm(np.linspace(cnorm.vmin, cnorm.vmax, N)))[np.newaxis, :], N, 0
        )[:, :, :-1]
        cmap_img *= l_vals
        cmap_img += (1 - l_vals) * background
        return cmap_img, img
    else:
        return None, img


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
    return bool(c2k > c2w)


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
    mappable = get_mappable(ax, silent=True)
    if mappable is not None:
        if isinstance(
            mappable, matplotlib.image._ImageBase | matplotlib.collections.QuadMesh
        ):
            if not image_is_light(mappable):
                c = dark
    return c


try:
    combined_cmap("bone_r", "hot", "bonehot", register=True)
except ValueError:
    pass
