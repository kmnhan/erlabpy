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
    "CenteredInversePowerNorm",
    "CenteredPowerNorm",
    "InversePowerNorm",
    "TwoSlopeInversePowerNorm",
    "TwoSlopePowerNorm",
    "axes_textcolor",
    "close_to_white",
    "color_distance",
    "combined_cmap",
    "flatten_transparency",
    "gen_2d_colormap",
    "get_mappable",
    "image_is_light",
    "nice_colorbar",
    "prominent_color",
    "proportional_colorbar",
    "unify_clim",
]

from collections.abc import Iterable, Sequence
from numbers import Number
from typing import Any, Literal, cast

import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.collections
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import numpy.typing as npt
from matplotlib.typing import ColorType, RGBColorType


class InversePowerNorm(matplotlib.colors.Normalize):
    r"""Inverse power-law normalization.

    Linearly map a given value to the 0-1 range and then apply an inverse power-law
    normalization over that range.

    For values :math:`x`, `matplotlib.colors.PowerNorm` calculates :math:`x^\gamma`,
    whereas `InversePowerNorm` calculates :math:`1-x^{1/\gamma}`. This provides higher
    contrast for values closer to ``vmin``.

    Parameters
    ----------
    gamma
        Power law normalization parameter. If equal to 1, the colormap is linear.
    vmin, vmax
        If ``vmin`` and/or ``vmax`` is not given, they are initialized from the minimum
        and maximum value, respectively, of the first input processed; i.e.,
        ``__call__(A)`` calls ``autoscale_None(A)``
    clip
        If ``True`` values falling outside the range ``[vmin, vmax]``, are mapped to 0
        or 1, whichever is closer, and masked values are set to 1.  If ``False`` masked
        values remain masked.

        Clipping silently defeats the purpose of setting the over, under, and masked
        colors in a colormap, so it is likely to lead to surprises; therefore the
        default is ``clip=False``.

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


class TwoSlopePowerNorm(matplotlib.colors.TwoSlopeNorm):
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
        """Map value to the interval [0, 1]. The clip argument is unused."""
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


class CenteredPowerNorm(matplotlib.colors.CenteredNorm):
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
        """Map value to the interval [0, 1]."""
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
    """Get the `matplotlib.cm.ScalarMappable` from a given `matplotlib.axes.Axes`.

    Parameters
    ----------
    ax
        Parent axes.
    image_only
        Only consider images as a valid mappable, by default `False`.
    silent
        If `False`, raises a `RuntimeError` when no mappable is found. If `True`,
        silently returns `None`.

    Returns
    -------
    matplotlib.cm.ScalarMappable or None

    """
    if not image_only:
        try:
            mappable: Any = ax.collections[-1]
        except (IndexError, AttributeError):
            mappable = None

    if image_only or mappable is None:
        try:
            mappable = ax.get_images()[-1]
        except (IndexError, AttributeError):
            mappable = None

    if mappable is None:
        if not silent:
            raise RuntimeError(
                "No mappable was found to use for colorbar "
                "creation. First define a mappable such as "
                "an image (with imshow) or a contour set ("
                "with contourf)."
            )
    return mappable


def unify_clim(
    axes: np.ndarray,
    target: matplotlib.axes.Axes | None = None,
    image_only: bool = False,
) -> None:
    """Unify the color limits for mappables in multiple axes.

    Parameters
    ----------
    axes
        Array of :class:`matplotlib.axes.Axes` to unify the color limits.
    target
        The target axis to unify the color limits. If provided, the target color limits
        will be taken from this axes. Otherwise, the color limits will be set to include
        all mappables in the ``axes``.
    image_only
        If `True`, only consider mappables that are images. Default is `False`.

    """
    vmn: float | None
    vmx: float | None

    if target is None:
        vmn_list, vmx_list = [], []
        for ax in axes.flat:
            mappable = get_mappable(ax, image_only=image_only, silent=True)
            if mappable is not None:
                if mappable.norm.vmin is not None:
                    vmn_list.append(mappable.norm.vmin)
                if mappable.norm.vmax is not None:
                    vmx_list.append(mappable.norm.vmax)
        vmn, vmx = min(vmn_list), max(vmx_list)
    else:
        mappable = get_mappable(target, image_only=image_only, silent=True)
        if mappable is not None:
            vmn, vmx = mappable.norm.vmin, mappable.norm.vmax

    # Apply color limits
    for ax in axes.flat:
        mappable = get_mappable(ax, image_only=image_only, silent=True)
        if mappable is not None:
            mappable.norm.vmin, mappable.norm.vmax = vmn, vmx


def proportional_colorbar(
    mappable: matplotlib.cm.ScalarMappable | None = None,
    cax: matplotlib.axes.Axes | None = None,
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    **kwargs,
) -> matplotlib.colorbar.Colorbar:
    """
    Replace the current colorbar or creates a new colorbar with proportional spacing.

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
        import matplotlib.colors

        # Create example data and plot
        X, Y = np.mgrid[0 : 3 : complex(0, 100), 0 : 2 : complex(0, 100)]
        pcm = plt.pcolormesh(
            X,
            Y,
            (1 + np.sin(Y * 10.0)) * X**2,
            norm=matplotlib.colors.PowerNorm(gamma=0.5),
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
    elif isinstance(ax, Iterable):
        if not isinstance(ax, np.ndarray):
            ax = np.array(ax, dtype=object)
        i = 0
        while mappable is None and i < len(ax.flat):
            mappable = get_mappable(ax.flatten()[i], silent=(i != (len(ax.flat) - 1)))
            i += 1
    elif mappable is None:
        mappable = get_mappable(ax)

    if mappable is None:
        raise RuntimeError("No mappable was found to use for colorbar creation")

    if mappable.colorbar is None:
        plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)
        mappable.colorbar = cast(matplotlib.colorbar.Colorbar, mappable.colorbar)

    ticks = mappable.colorbar.get_ticks()
    if cax is None:
        mappable.colorbar.remove()
    kwargs.setdefault("ticks", ticks)
    kwargs.setdefault("cmap", mappable.cmap)
    kwargs.setdefault("norm", mappable.norm)

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
    **kwargs,
) -> matplotlib.axes.Axes:
    fig = parent_axes.get_figure()
    if fig is None:
        raise RuntimeError("Parent axes is not attached to a figure")
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
        return matplotlib.transforms.TransformedBbox(
            matplotlib.transforms.Bbox.from_bounds(*self._size_to_bounds(ax)),
            self._transAxes
            + matplotlib.transforms.ScaledTranslation(
                self.pads[0], self.pads[1], ax.figure.dpi_scale_trans
            )
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


def nice_colorbar(
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    mappable: matplotlib.cm.ScalarMappable | None = None,
    width: float = 8.0,
    aspect: float = 5.0,
    pad: float = 3.0,
    minmax: bool = False,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    floating=False,
    ticklabels: Sequence[str] | None = None,
    **kwargs,
):
    r"""Create a colorbar with fixed width and aspect to ensure uniformity of plots.

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
        if isinstance(ax, Iterable):
            if not isinstance(ax, np.ndarray):
                ax = np.array(ax, dtype=object)
            bbox = matplotlib.transforms.Bbox.union(
                [
                    x.get_position(original=True)
                    .frozen()
                    .transformed(x.figure.transFigure)
                    .transformed(x.figure.dpi_scale_trans.inverted())
                    for x in ax.flat
                ]
            )
        else:
            fig = ax.get_figure()

            if fig is None:
                raise RuntimeError("Axes is not attached to a figure")

            bbox = (
                ax.get_position(original=True)
                .frozen()
                .transformed(fig.transFigure)
                .transformed(fig.dpi_scale_trans.inverted())
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


def flatten_transparency(rgba: npt.NDArray, background: RGBColorType | None = None):
    """
    Flatten the transparency of an RGBA image by blending it with a background color.

    Parameters
    ----------
    rgba
        The input RGBA image as a numpy array.
    background : RGBColorType, optional
        The background color to blend with. Defaults to white.

    """
    if background is None:
        background = (1, 1, 1)
    else:
        background = matplotlib.colors.to_rgb(background)

    original_shape = rgba.shape
    rgba = rgba.reshape(-1, 4)
    rgb = rgba[:, :-1]
    a = rgba[:, -1][:, np.newaxis]
    rgb *= a
    rgb += (1 - a) * background
    return rgb.reshape(original_shape[:-1] + (3,))


def _get_segment_for_color(
    cmap: matplotlib.colors.LinearSegmentedColormap,
    color: Literal["red", "green", "blue", "alpha"],
) -> Any:
    if hasattr(cmap, "_segmentdata"):
        if color in cmap._segmentdata:
            return cmap._segmentdata[color]
    return None


def _is_segment_iterable(cmap: matplotlib.colors.Colormap) -> bool:
    if not isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        return False

    if any(callable(_get_segment_for_color(cmap, c)) for c in ["red", "green", "blue"]):  # type: ignore[arg-type]
        return False
    return True


def combined_cmap(
    cmap1: matplotlib.colors.Colormap | str,
    cmap2: matplotlib.colors.Colormap | str,
    name: str,
    register: bool = False,
    N=256,
) -> matplotlib.colors.Colormap:
    """Stitch two existing colormaps to create a new colormap.

    This was implemented before :func:`cmasher.combine_cmaps` existed. Using that might
    be better.

    Parameters
    ----------
    cmap1
        The first colormap to be combined. Can be either a
        :class:`matplotlib.colors.Colormap` or a string representing the name of a
        registered colormap.
    cmap2
        The second colormap to be combined. Can be either a
        :class:`matplotlib.colors.Colormap` or a string representing the name of a
        registered colormap.
    name
        The name of the combined colormap.
    register
        Whether to register the combined colormap. Defaults to `False`.
    N
        The number of colors in the resulting colormap. Defaults to 256.

    Returns
    -------
    matplotlib.colors.Colormap
        The combined colormap.

    """
    if isinstance(cmap1, str):
        cmap1 = matplotlib.colormaps[cmap1]
    if isinstance(cmap2, str):
        cmap2 = matplotlib.colormaps[cmap2]

    if all(_is_segment_iterable(c) for c in (cmap1, cmap2)):
        cmap1 = cast(
            matplotlib.colors.LinearSegmentedColormap, cmap1
        )  # to appease mypy
        cmap2 = cast(
            matplotlib.colors.LinearSegmentedColormap, cmap2
        )  # to appease mypy

        segnew: dict[
            Literal["red", "green", "blue", "alpha"], Sequence[tuple[float, ...]]
        ] = {}

        for c in ["red", "green", "blue"]:
            seg1_c, seg2_c = (
                np.asarray(_get_segment_for_color(cmap1, c)),  # type: ignore[arg-type]
                np.asarray(_get_segment_for_color(cmap2, c)),  # type: ignore[arg-type]
            )
            seg1_c[:, 0] = seg1_c[:, 0] * 0.5
            seg2_c[:, 0] = seg2_c[:, 0] * 0.5 + 0.5
            segnew[c] = np.r_[seg1_c, seg2_c]  # type: ignore[index]
        cmap = matplotlib.colors.LinearSegmentedColormap(
            name=name, segmentdata=segnew, N=N
        )
    else:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
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
    cmap: matplotlib.colors.Colormap | str | None = None,
    *,
    lnorm: plt.Normalize | None = None,
    cnorm: plt.Normalize | None = None,
    background: ColorType | None = None,
    N: int = 256,
):
    """Generate a 2D colormap image from lightness and color data.

    Parameters
    ----------
    ldat : array-like
        The lightness data.
    cdat : array-like
        The color data. Must have the same shape as `ldat`.
    cmap
        The colormap to use for the color axis. If `None`, a predefined linear segmented
        colormap is used.
    lnorm
        The normalization for the lightness axes.
    cnorm
        The normalization for the color axes.
    background : ColorType, optional
        The background color. If `None`, it is set to white.
    N
        The number of levels in the colormap. Default is 256. The resulting colormap
        will have a shape of ``(N, N, 4)``, where the last dimension represents the RGBA
        values.

    Returns
    -------
    cmap_img : array-like
        RGBA image of the colormap.
    img : array-like
        RGBA image with the 2D colormap applied.

    """
    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", colors=[[0, 0, 1], [0, 0, 0], [1, 0, 0]], N=N
        )

    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if lnorm is None:
        lnorm = plt.Normalize()

    if cnorm is None:
        cnorm = plt.Normalize()

    if background is None:
        background_arr: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    else:
        background_arr = matplotlib.colors.to_rgba(background)

    ldat_masked = np.ma.masked_invalid(ldat)
    cdat_masked = np.ma.masked_invalid(cdat)

    lnorm.autoscale_None(ldat_masked)
    cnorm.autoscale_None(cdat_masked)

    l_vals = lnorm(ldat_masked)
    c_vals = cnorm(cdat_masked)
    l_vals = l_vals[:, :, np.newaxis]

    img = cmap(c_vals)
    img *= l_vals
    img += (1 - l_vals) * background_arr

    lmin, lmax = cast(float, lnorm.vmin), cast(float, lnorm.vmax)  # to appease mypy
    cmin, cmax = cast(float, cnorm.vmin), cast(float, cnorm.vmax)

    l_linear = lnorm(np.linspace(lmin, lmax, N))[:, np.newaxis, np.newaxis]
    cmap_img = np.repeat(cmap(cnorm(np.linspace(cmin, cmax, N)))[np.newaxis, :], N, 0)
    cmap_img *= l_linear
    cmap_img += (1 - l_linear) * background_arr

    return cmap_img, img


def color_distance(c1: ColorType, c2: ColorType) -> float:
    """Calculate the color distance between two matplotlib colors.

    Parameters
    ----------
    c1, c2 : ColorType
        Color to calculate the distance between in any format that
        :func:`matplotlib.colors.to_rgb` can handle.

    Returns
    -------
    distance : float
        The color distance between the two colors.

    Note
    ----
    The color distance is calculated using the Euclidean distance formula in the RGB
    color space. The formula takes into account the perceptual differences between
    colors.

    See Also
    --------
    - https://www.compuphase.com/cmetric.htm

    """
    R1, G1, B1 = (np.array(matplotlib.colors.to_rgb(c1)) * 255).astype(int)
    R2, G2, B2 = (np.array(matplotlib.colors.to_rgb(c2)) * 255).astype(int)
    dR2 = (R2 - R1) ** 2
    dG2 = (G2 - G1) ** 2
    dB2 = (B2 - B1) ** 2
    r = 0.5 * (R1 + R2) / 256
    return np.sqrt((2 + r) * dR2 + 4 * dG2 + (2 + 255 / 256 - r) * dB2)


def close_to_white(c: ColorType) -> bool:
    """Check if a given color is closer to white than black.

    Parameters
    ----------
    c : ColorType
        Color in any format that :func:`matplotlib.colors.to_rgb` can handle.

    Returns
    -------
    bool
        `True` if the color is closer to white than black, `False` otherwise.
    """
    c2k = color_distance(c, (0, 0, 0))
    c2w = color_distance(c, (1, 1, 1))
    return bool(c2k > c2w)


def prominent_color(im: matplotlib.image._ImageBase | matplotlib.collections.QuadMesh):
    """Calculate the prominent color of an image.

    This function calculates the prominent color of an image by finding the most
    frequent color in the image's histogram in color space. If the image array is
    `None`, returns white.

    """
    im_array = im.get_array()
    if im_array is None:
        return matplotlib.colors.to_rgba("w")

    # https://github.com/numpy/numpy/issues/11879
    hist, edges = np.histogram(np.nan_to_num(im_array), "sqrt")
    mx = hist.argmax()
    return im.to_rgba(edges[mx : mx + 2].mean())


def image_is_light(
    im: matplotlib.image._ImageBase | matplotlib.collections.QuadMesh,
) -> bool:
    """Determine if an image is *light* or *dark*.

    Checks whether the prominent color is closer to white than black.
    """
    return close_to_white(prominent_color(im))


def axes_textcolor(
    ax: matplotlib.axes.Axes, light: ColorType = "k", dark: ColorType = "w"
):
    """Determine the text color based on the color of the mappable in an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object for which the text color needs to be determined.
    light : ColorType
        The *light* color, returned when :func:`image_is_light
        <erlab.plotting.colors.image_is_light>` returns `False`. Default is ``'w'``
        (white).
    dark : ColorType
        The *dark* color, returned when :func:`image_is_light
        <erlab.plotting.colors.image_is_light>` returns `True`. Default is ``'k'``
        (black).

    """
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
