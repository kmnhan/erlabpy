"""Plotting utilities."""

from typing import Type
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.spatial.transform import Rotation

from arpes.utilities import bz
from arpes.plotting.bz import bz_plot
from arpes.plotting.utils import name_for_dim, unit_for_dim
from arpes.utilities.conversion.forward import (
    convert_coordinates_to_kspace_forward
)

__all__ = ['proportional_colorbar','plot_hex_bz','plot_hv_text',
           'label_subplots','annotate_cuts_erlab, label_subplot_properties']

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

def plot_hex_bz(a=3.54, rotate=0, ax=None, **kwargs):
    """
    Plots a 2D hexagonal BZ overlay on the specified axes.
    """
    kwargs.setdefault('alpha',1)
    kwargs.setdefault('color','k')
    kwargs.setdefault('linestyle','-')
    kwargs.setdefault('linewidth',0.5)
    kwargs.setdefault('zorder',5)

    if ax is None:
        ax = plt.gca()
        
    bz_plot(
        ax=ax,
        cell=bz.hex_cell_2d(a / (2 * np.pi)),
        paths=[],
        repeat=None,
        set_equal_aspect=False,
        hide_ax=False,
        transformations=[Rotation.from_rotvec([0, 0, rotate*np.pi/180])],
        **kwargs
    )

def annotate_cuts_erlab(data: xr.DataArray, plotted_dims,
                        ax=None, include_text_labels=False, color='k',
                        textoffset=[0, 0], plot_kw={}, text_kw={}, **kwargs):
    r"""Annotates a cut location onto a plot. 

    Does what `arpes.plotting.annotations.annotate_cuts` aims to do, but
    is much more robust and customizable.

    Parameters
    ----------
    data : xarray.DataArray
        The data before momentum space conversion.

    plotted_dims: list of str
        The dimension names currently plotted on the target axes.

    ax : `matplotlib.axes.Axes`, optional
        The `matplotlib.axes.Axes` instance in which the annotation is
        placed, defaults to the current axes when optional.

    include_text_labels: bool, default=False
        Whether to include text labels.

    color : color. optional
        Color of both the line and label text. Each color can be
        overridden by `plot_kw` and `text_kw`.

    plot_kw : dict, optional
        Extra arguments to `matplotlib.pyplot.plot`: refer to the 
        `matplotlib` documentation for a list of all possible arguments.

    text_kw : dict, optional
        Extra arguments to `matplotlib.pyplot.text`: refer to the 
        `matplotlib` documentation for a list of all possible arguments.
        Has no effect if `include_text_labels` is False.

    textoffset : list of float or tuple of float
        Horizontal and vertical offset of text labels. Has no effect if
        `include_text_labels` is False.

    **kwargs : dict
        Defines the coordinates of the cut location.

    Examples
    --------
    Annotate :math:`k_z`-:math:`k_x` plot in the current axes with lines
    at :math:`h\nu=56` and :math:`60`.

    >>> annotate_cuts(hv_scan, ['kx', 'kz'], hv=[56, 60])

    Annotate with thin dashed red lines.

    >>> annotate_cuts(hv_scan, ['kx', 'kz'], hv=[56, 60],
                      plot_kw={'ls': '--', 'lw': 0.5, 'color': 'red'})

    """
    assert len(plotted_dims) == 2, 'Only 2D axes can be annotated.'
    converted_coordinates = convert_coordinates_to_kspace_forward(data)
    text_kw.setdefault('horizontalalignment', 'left')
    text_kw.setdefault('verticalalignment', 'top')
    for k, v in kwargs.items():
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]
        selected = converted_coordinates.sel(**dict([[k,v]]), method='nearest')
        for coords_dict, obj in selected.G.iterate_axis(k):
            plt_css = [np.mean(obj[d].values,axis=1) for d in plotted_dims]
            with plt.rc_context({'lines.linestyle': '--',
                                 'lines.linewidth': 0.85,
                                 'lines.color': color}):
                ax.plot(*plt_css, **plot_kw)
            if include_text_labels:
                idx = np.argmin(plt_css[0])
                with plt.rc_context({'text.color':color}):
                    plt.text(
                        plt_css[0][idx] + 0.02 + textoffset[0],
                        plt_css[1][idx] + 0.04 + textoffset[1],
                        "{} = {} {}".format(
                            name_for_dim(k),
                            int(np.rint(coords_dict[k].item())),
                            unit_for_dim(k)
                        ),
                        **text_kw
                    )

def _alph_label(val, prefix, suffix, capital):
    """Generate labels from string or integer."""
    if isinstance(val, (int, np.integer)) or val.isdigit():
        if capital:
            ref_char = 'A'
        else:
            ref_char = 'a'
        val = chr(int(val) + ord(ref_char) - 1)
    elif isinstance(val, str):
        pass
    else:
        raise TypeError('Input values must be integers or strings.')
    return prefix + val + suffix

def label_subplots(axs, values=None, order='C',
                   loc='upper left', bbox_to_anchor=None,
                   prefix='(', suffix=')', capital=False,
                   fontweight='bold', fontsize='medium', **kwargs):
    r"""Labels subplots with automatically generated labels.

    Parameters
    ----------

    axs : `matplotlib.axes.Axes`, list of Axes
        Axes to label. If an array is given, the order will be
        determined by the flattening method given by `order`.

    values : list of int or list of str, optional
        Integer or string labels corresponding to each Axes in `axs` for
        manual labels.

    order : {'C', 'F', 'A', 'K'}, optional
        Order in which to flatten `ax`. 'C' means to flatten in
        row-major (C-style) order. 'F' means to flatten in column-major 
        (Fortran- style) order. 'A' means to flatten in column-major 
        order if a is Fortran contiguous in memory, row-major order 
        otherwise. 'K' means to flatten a in the order the elements 
        occur in memory. The default is 'C'.

    loc : {'upper left', 'upper center', 'upper right', 'center left',
    'center', 'center right', 'lower left', 'lower center, 'lower
    right'}, optional
        The box location. The default is 'upper left'. 
    bbox_to_anchor : 2-tuple, or 4-tuple of floats, optional
        Box that is used to position the legend in conjunction with 
        `loc`, given in axes units.

    prefix : str, optional
        String to prepend to the alphabet label. The default is '('.
    suffix : str, optional
        String to append to the alphabet label. The default is ')'.
    capital: bool, default=False
        Capitalize automatically generated alphabetical labels.

    fontweight : {'ultralight', 'light', 'normal', 'regular', 'book', 
    'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 
    'extra bold', 'black'}, optional
        Set the font weight.
    fontsize :  float or {'xx-small', 'x-small', 'small', 'medium',
    'large', 'x-large', 'xx-large'}, optional
        Set the font size.
    **kwargs : dict, optional
        Extra arguments to `matplotlib.pyplot.colorbar`: refer to the 
        `matplotlib` documentation for a list of all possible arguments.

    """
    if plt.rcParams['text.usetex'] & (fontweight == 'bold'):
        prefix = '\\bf{' + prefix
        suffix = suffix + '}'
    axlist = np.array(axs, dtype=object).flatten(order=order)
    if values is None:
        values = np.array([i + 1 for i in range(len(axlist))], dtype=np.int64)
    else:
        values = np.array(values).flatten(order=order)
        if not (axlist.size == values.size):
            raise IndexError('The number of given values must match the number'
                             ' of given axes.')
    with plt.rc_context({'text.color': 'k'}):
        for i in range(len(axlist)):
            at = AnchoredText(_alph_label(values[i], prefix, suffix, capital),
                    loc=loc, frameon=False,
                    pad=0, borderpad=0.5,
                    prop=dict(fontsize=fontsize,**kwargs),
                    bbox_to_anchor=bbox_to_anchor,
                    bbox_transform=axlist[i].transAxes)
            axlist[i].add_artist(at)

def label_subplot_properties(axs, values, **kwargs):
    r"""Labels subplots with automatically generated labels.

    Parameters
    ----------

    axs : `matplotlib.axes.Axes`, list of Axes
        Axes to label. If an array is given, the order will be
        determined by the flattening method given by `order`.
    values : dict
        key-value pair of annotations.

    """
    kwargs.setdefault('fontweight','medium')
    kwargs.setdefault('prefix','')
    kwargs.setdefault('suffix','')
    kwargs.setdefault('loc','upper right')
    for k, v in values.items():
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]
        label_subplots(axs,["{} = {} {}".format(
                                name_for_dim(k),
                                val,
                                unit_for_dim(k)
                            ) for val in v], **kwargs)

# TODO: fix format using name_for_dim and unit_for_dim
def plot_hv_text(ax,val,x=0.025,y=0.975,**kwargs):
    s = '$h\\nu='+str(val)+'$~eV'
    ax.text(x, y, s, family="serif",horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,**kwargs)
def plot_hv_text_right(ax,val,x=1-0.025,y=0.975,**kwargs):
    s = '$h\\nu='+str(val)+'$~eV'
    ax.text(x, y, s, family="serif",horizontalalignment='right',verticalalignment='top', transform=ax.transAxes,**kwargs)