"""Plot annotations."""
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from arpes.plotting.utils import name_for_dim, unit_for_dim
from arpes.utilities.conversion.forward import (
    convert_coordinates_to_kspace_forward
)

__all__ = ['plot_hv_text','label_subplots','annotate_cuts_erlab', 'label_subplot_properties']


def annotate_cuts_erlab(data:xr.DataArray, plotted_dims,
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
    plot_kw.setdefault('color', color)
    for k, v in kwargs.items():
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]
        selected = converted_coordinates.sel(**dict([[k,v]]), method='nearest')
        for coords_dict, obj in selected.G.iterate_axis(k):
            plt_css = [np.mean(obj[d].values,axis=1) for d in plotted_dims]
            with plt.rc_context({'lines.linestyle': '--',
                                 'lines.linewidth': 0.85}):
                ax.plot(*plt_css, **plot_kw)
            if include_text_labels:
                idx = np.argmin(plt_css[0])
                with plt.rc_context({'text.color':color}):
                    ax.text(
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
        prefix = '\\textbf{' + prefix
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

def property_label(key, value):
    if plt.rcParams['text.usetex']:
        prefix = "$"
        base = "{} = {}~{}"
        suffix = "$"
    else:
        prefix = ""
        base = "{} = {} {}"
        suffix = ""

    if (key == 'Eb'):
        value = np.rint(value * 1000).astype(int)
    
    if (value == 0) & (key == 'Eb'):
        label = prefix + 'E = E_F' + suffix
    else:
        label = prefix + str(base.format(name_for_dim(key),
                                         value, 
                                         unit_for_dim(key))) + suffix
    return label    

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

    strlist = []
    for k, v in values.items():
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]
        strlist.append([property_label(k,val) for val in v])
    strlist = list(zip(*strlist))
    strlist = ['\n'.join(strlist[i]) for i in range(len(strlist))]
    label_subplots(axs, strlist, **kwargs)

# TODO: fix format using name_for_dim and unit_for_dim
def plot_hv_text(ax,val,x=0.025,y=0.975,**kwargs):
    s = '$h\\nu='+str(val)+'$~eV'
    ax.text(x, y, s, family="serif",horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,**kwargs)
def plot_hv_text_right(ax,val,x=1-0.025,y=0.975,**kwargs):
    s = '$h\\nu='+str(val)+'$~eV'
    ax.text(x, y, s, family="serif",horizontalalignment='right',verticalalignment='top', transform=ax.transAxes,**kwargs)