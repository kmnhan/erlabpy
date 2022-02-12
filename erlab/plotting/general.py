"""General plotting utilities."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
    
from matplotlib.widgets import AxesWidget

from pyimagetool import imagetool, RegularDataArray

__all__ = ['ximagetool','proportional_colorbar']


def ximagetool(da:xr.DataArray, prebin=None, cmap='viridis'):
    """
    Open imagetool from xarray.DataArray. Note that NaN values are displayed as zero.
    """
    if isinstance(da, xr.Dataset):
        da = da.spectrum
    if prebin is not None:
        da = da.coarsen(prebin).mean()
    dims = da.dims
    if set(dims) == set(('kx', 'ky', 'eV')):
        da = da.transpose('kx', 'ky', 'eV')

    try:
        units = ['('+da[dims[i]].units+')' for i in range(len(dims))]
    except AttributeError:
        units = ['' for i in range(len(dims))]
    data = RegularDataArray(
        da.fillna(0),
        dims=[dims[i]+' '+units[i] for i in range(len(dims))]
    )

    # TODO: implement 4dim and raise error
    tool = imagetool(data)
    # tool.set_all_cmaps(cmap)
    # tool.pg_win.load_ct('viridis')
    # tool.pg_win.update()
    # tool.info_bar.cmap_combobox.currentTextChanged.connect(tool.set_all_cmaps)
    
    return tool

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
        mappable = plt.gci()
        if mappable is None:
            try:
                mappable = ax_ref.collections[-1]
            except IndexError:
                pass
            if mappable is None:
                raise RuntimeError('No mappable was found to use for colorbar '
                                   'creation. First define a mappable such as '
                                   'an image (with imshow) or a contour set ('
                                   'with contourf).')
    if mappable.colorbar is None:
        plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)
    ticks = mappable.colorbar.get_ticks()
    mappable.colorbar.remove()
    kwargs.setdefault('ticks',ticks)
    cbar = plt.colorbar(
        mappable=mappable,
        cax=cax, ax=ax,
        spacing='proportional',
        boundaries=mappable.norm.inverse(np.linspace(0,1,mappable.cmap.N)),
        **kwargs,
    )
    return cbar