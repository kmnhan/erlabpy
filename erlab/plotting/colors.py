import pkgutil
from io import StringIO

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyqtgraph.Qt import QtGui

__all__ = ['TwoSlopePowerNorm', 'get_mappable', 'proportional_colorbar',
           'color_distance', 'close_to_white', 'prominent_color',
           'image_is_light', 'mpl_color_to_QColor', 'pg_colormap_names',
           'pg_colormap_from_name', 'pg_colormap_powernorm',
           'pg_colormap_to_QPixmap']


def load_igor_ct(file, name):
    file = pkgutil.get_data(__package__, 'IgorCT/' + file)
    cmap = LinearSegmentedColormap.from_list(
        name, np.genfromtxt(StringIO(file.decode())) / 65535)
    plt.colormaps.register(cmap)
    plt.colormaps.register(cmap.reversed())


load_igor_ct('Blue-White.txt', 'BlWh')


class TwoSlopePowerNorm(colors.Normalize):
    def __init__(self, gamma, vcenter=0, vmin=None, vmax=None):
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        gamma : float
            Power law exponent
        vcenter : float, default: 0
            The data value that defines ``0.5`` in the normalization.
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
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
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
            val[val_ < 0.5] = np.ma.power(1 - 2 * val_l, 1. / gamma) \
                * (vmin - vcenter) + vcenter
            val[val_ >= 0.5] = np.ma.power(2 * val_u - 1, 1. / gamma) \
                * (vmax - vcenter) + vcenter
            return np.ma.asarray(val)
        else:
            if value < 0.5:
                return pow(1 - 2 * value, 1. / gamma) \
                    * (vmin - vcenter) + vcenter
            else:
                return pow(2 * value - 1, 1. / gamma) \
                    * (vmax - vcenter) + vcenter


def get_mappable(ax, error=True):
    try:
        mappable = ax.collections[-1]
    except (IndexError, AttributeError):
        try:
            mappable = ax.get_images()[-1]
        except (IndexError, AttributeError):
            mappable = None
    if error is True and mappable is None:
        raise RuntimeError('No mappable was found to use for colorbar '
                           'creation. First define a mappable such as '
                           'an image (with imshow) or a contour set ('
                           'with contourf).')
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
    if cax is None:
        if ax is None:
            ax = plt.gca()
            ax_ref = ax
        else:
            try:
                ax_ref = ax.flatten()[0]
            except AttributeError:
                ax_ref = ax
    else:
        ax_ref = cax
    if mappable is None:
        mappable = get_mappable(ax_ref)
    if mappable.colorbar is None:
        plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)
    ticks = mappable.colorbar.get_ticks()
    mappable.colorbar.remove()
    kwargs.setdefault('ticks', ticks)
    cbar = plt.colorbar(
        mappable=mappable,
        cax=cax, ax=ax,
        spacing='proportional',
        boundaries=mappable.norm.inverse(np.linspace(0, 1, mappable.cmap.N)),
        **kwargs,
    )
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
    hist, edges = np.histogram(np.nan_to_num(im.get_array()), 'auto')
    mx = hist.argmax()
    return im.to_rgba(edges[mx:mx + 2].mean())


def image_is_light(im):
    return close_to_white(prominent_color(im))


def mpl_color_to_QColor(c, alpha=None):
    """Convert matplotlib color to QtGui.Qcolor."""
    return QtGui.QColor.fromRgbF(*colors.to_rgba(c, alpha=alpha))


def pg_colormap_names(source='all'):
    local = sorted(pg.colormap.listMaps())
    if source == 'local':
        return local
    else:
        mpl = sorted(pg.colormap.listMaps(source='matplotlib'))
        for cmap in mpl:
            if cmap.startswith('cet_'):
                mpl = list(filter((cmap).__ne__, mpl))
            elif cmap.endswith('_r'):
                # mpl_r.append(cmap)
                mpl = list(filter((cmap).__ne__, mpl))
        if source == 'all':
            cet = sorted(pg.colormap.listMaps(source='colorcet'))
            # if (mpl != []) and (cet != []):
            # local = []
            # mpl_r = []
            all_cmaps = local + cet + mpl  # + mpl_r
        else:
            all_cmaps = local + mpl
    return list({value: None for value in all_cmaps})


def pg_colormap_from_name(name: str, skipCache=True):
    try:
        return pg.colormap.get(name, skipCache=skipCache)
    except FileNotFoundError:
        try:
            return pg.colormap.get(name, source='matplotlib',
                                   skipCache=skipCache)
        except ValueError:
            return pg.colormap.get(name, source='colorcet',
                                   skipCache=skipCache)


def pg_colormap_powernorm(cmap, gamma, reverse=False, skipCache=True,
                          highContrast=False, zeroCentered=False):
    if isinstance(cmap, str):
        cmap = pg_colormap_from_name(cmap, skipCache=skipCache)
    if reverse:
        cmap.reverse()
    N = 4096
    if gamma == 1:
        mapping = np.linspace(0, 1, N)
    elif highContrast and (gamma < 1):
        if zeroCentered:
            map_half = (1 - np.power(
                np.linspace(1, 0, int(N / 2)), 1. / gamma)) * 0.5 + 0.5
            mapping = np.concatenate((-np.flip(map_half) + 1, map_half))
        else:
            mapping = 1 - np.power(np.linspace(1, 0, N), 1. / gamma)
    else:
        if gamma < 1:
            N = 65536
        if zeroCentered:
            map_half = np.power(
                np.linspace(
                    0, 1, int(
                        N / 2)), gamma) * 0.5 + 0.5
            mapping = np.concatenate((-np.flip(map_half) + 1, map_half))
        else:
            mapping = np.power(np.linspace(0, 1, N), gamma)
    cmap.color = cmap.mapToFloat(mapping)
    cmap.pos = np.linspace(0, 1, N)
    return cmap


def pg_colormap_to_QPixmap(cmap, w=64, h=16, skipCache=False):
    """Convert pyqtgraph colormap to a `w`-by-`h` QPixmap thumbnail."""
    if isinstance(cmap, str):
        cmap = pg_colormap_from_name(cmap, skipCache=skipCache)
    # cmap_arr = np.reshape(cmap.getColors()[:, None], (1, -1, 4), order='C')
    # cmap_arr = np.reshape(
        # cmap.getLookupTable(0, 1, w, alpha=True)[:, None], (1, -1, 4),
        # order='C')
    cmap_arr = cmap.getLookupTable(0, 1, w, alpha=True)[:, None]

    # print(cmap_arr.shape)
    img = QtGui.QImage(cmap_arr, w, 1,
                       QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img).scaled(w, h)
