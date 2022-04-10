"""Plot annotations."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arpes.plotting.utils import name_for_dim, unit_for_dim
from arpes.utilities.conversion.forward import \
    convert_coordinates_to_kspace_forward
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText
from matplotlib.transforms import Affine2D

from .colors import get_mappable, image_is_light

__all__ = ['plot_hv_text','label_subplots','annotate_cuts_erlab', 
           'label_subplot_properties', 'mark_points']


SI_PREFIXES = {24: 'Y', 21: 'Z', 18: 'E', 15: 'P', 12: 'T', 9: 'G',
               6: 'M', 3: 'k', 2: 'h', 1: 'da', 0: '', -1: 'd', -2: 'c',
               -3: 'm', -6: 'μ', -9: 'n', -12: 'p', -15: 'f', -18: 'a',
               -21: 'z', -24: 'y'}

SI_PREFIX_NAMES = ['yotta','zetta','exa','peta','tera',
                   'giga','mega','kilo','hecto','deca', '',
                   'deci','centi','milli','micro','nano',
                   'pico','femto','atto','zepto','yocto']

# SI_FACTORS = [24, 21, 18, 15, 12, 9, 6, 3, 2, 1, 0,
            #   -1, -2, -3, -6, -9, -12, -15, -18, -21, -24]
# SI_PREFIXES = ['Y', 'Z', 'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da', '',
#                'd', 'c', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y']

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


def _alph_label(val, prefix, suffix, numeric, capital):
    """Generate labels from string or integer."""
    if isinstance(val, (int, np.integer)) or val.isdigit():
        if numeric:
            val = str(val)
        else:
            if capital:
                ref_char = 'A'
            else:
                ref_char = 'a'
            val = chr(int(val) + ord(ref_char) - 1)
    elif not isinstance(val, str):
        raise TypeError('Input values must be integers or strings.')
    return prefix + val + suffix

def label_subplots(axes, values=None, startfrom=1, order='C',
                   loc='upper left', offset=(0.0, 0.0),
                   prefix='(', suffix=')', numeric=False, capital=False,
                   fontweight='normal', fontsize=None, **kwargs):
    r"""Labels subplots with automatically generated labels.

    Parameters
    ----------

    axes : `matplotlib.axes.Axes`, list of Axes
        Axes to label. If an array is given, the order will be
        determined by the flattening method given by `order`.

    values : list of int or list of str, optional
        Integer or string labels corresponding to each Axes in `axes` for
        manual labels.
        
    startfrom : int, optional
        Start from this number when creating automatic labels. Has no
        effect when `values` is not `None`.

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
    offset : 2-tuple of floats, optional
        Values that are used to position the legend in conjunction with 
        `loc`, given in display units.

    prefix : str, optional
        String to prepend to the alphabet label. The default is '('.
    suffix : str, optional
        String to append to the alphabet label. The default is ')'.
    numeric: bool, default=False
        Use integer labels instead of alphabets.
    capital: bool, default=False
        Capitalize automatically generated alphabetical labels.

    fontweight : {'ultralight', 'light', 'normal', 'regular', 'book', 
    'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 
    'extra bold', 'black'}, optional
        Set the font weight. The default is `'normal'`.
    fontsize :  float or {'xx-small', 'x-small', 'small', 'medium',
    'large', 'x-large', 'xx-large'}, optional
        Set the font size. The default is `'medium'` for axes, and
        `'large'` for figures.
    **kwargs : dict, optional
        Extra arguments to `matplotlib.pyplot.colorbar`: refer to the 
        `matplotlib` documentation for a list of all possible arguments.

    """
    if plt.rcParams['text.usetex'] & (fontweight == 'bold'):
        prefix = '\\textbf{' + prefix
        suffix = suffix + '}'
    axlist = np.array(axes, dtype=object).flatten(order=order)
    if values is None:
        values = np.array([i + startfrom for i in range(len(axlist))],
                          dtype=np.int64)
    else:
        values = np.array(values).flatten(order=order)
        if not (axlist.size == values.size):
            raise IndexError('The number of given values must match the number'
                             ' of given axes.')
    
    for i in range(len(axlist)):
        bbox_to_anchor = axlist[i].bbox
        if fontsize is None:
            if isinstance(axlist[i], Figure):
                fs = 'large'
            else:
                fs = 'medium'
        else:
            fs = fontsize
        bbox_transform = Affine2D().translate(*offset)
        label_str = _alph_label(values[i], prefix, suffix,
                                numeric, capital)
        mappable = get_mappable(axlist[i], error=False)
        clr = 'k'
        if mappable is not None:
            if isinstance(mappable,(mpl.image._ImageBase,
                                    mpl.collections.QuadMesh)):
                if not image_is_light(mappable):
                    clr = 'w'
        with plt.rc_context({'text.color': clr}):
            at = AnchoredText(label_str,
                    loc=loc, frameon=False,
                    pad=0, borderpad=0.5,
                    prop=dict(fontsize=fs, **kwargs),
                    bbox_to_anchor=bbox_to_anchor,
                    bbox_transform=bbox_transform)
        axlist[i].add_artist(at)

def property_label(key, value, decimals=0, si=0, name=None, unit=None):
    if name == '':
        delim = ''
    else:
        delim = ' = '
    if name is None:
        name = name_for_dim(key)
        if name is None:
            name = ''
    if unit is None:
        unit = unit_for_dim(key)
        if unit is None:
            unit = ''
        
    unit = SI_PREFIXES[si] + unit
    value /= 10 ** si
    if decimals is not None:
        value = np.around(value, decimals=decimals)
    if int(value) == value:
        value = int(value)
        
    if key == 'Eb':
        if value == 0:
            if delim == '':
                return '$E_F$'
            else:
                return '$E - E_F$'
        elif delim == '':
            if value > 0:
                delim = '+'
            name = 'E_F'

    base = '${}' + delim + '{}$ {}'
    return str(base.format(name, value, unit))

def label_subplot_properties(axes, values, decimals=None, si=0,
                             name=None, unit=None, order='C', **kwargs):
    r"""Labels subplots with automatically generated labels.

    Parameters
    ----------

    axes : `matplotlib.axes.Axes`, list of Axes
        Axes to label. If an array is given, the order will be
        determined by the flattening method given by `order`.
    values : dict
        key-value pair of annotations.
    decimals : None or int, optional
        Number of decimal places to round to. If decimals is None, no
        rounding is performed. If decimals is negative, it specifies the
        number of positions to the left of the decimal point. 
    si : int, optional
        Powers of 10 for automatic SI prefix setting.
    name : str, optional
        When set, overrides automatic dimension name setting.
    unit : str, optional
        When set, overrides automatic unit setting.
    short: bool, default=False
        Whether to omit
    order : {'C', 'F', 'A', 'K'}, optional
        Order in which to flatten `ax`. 'C' means to flatten in
        row-major (C-style) order. 'F' means to flatten in column-major 
        (Fortran- style) order. 'A' means to flatten in column-major 
        order if a is Fortran contiguous in memory, row-major order 
        otherwise. 'K' means to flatten a in the order the elements 
        occur in memory. The default is 'C'.
        
    short : bool, default=False

    """
    kwargs.setdefault('fontweight','medium')
    kwargs.setdefault('prefix','')
    kwargs.setdefault('suffix','')
    kwargs.setdefault('loc','upper right')

    strlist = []
    for k, v in values.items():
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]
        else:
            v = np.array(v).flatten(order=order)
        strlist.append([property_label(k, val, decimals=decimals, si=si,
                                       name=name, unit=unit) for val in v])
    strlist = list(zip(*strlist))
    strlist = ['\n'.join(strlist[i]) for i in range(len(strlist))]
    label_subplots(axes, strlist, order=order, **kwargs)

# TODO: fix format using name_for_dim and unit_for_dim
def plot_hv_text(ax,val,x=0.025,y=0.975,**kwargs):
    s = '$h\\nu='+str(val)+'$~eV'
    ax.text(x, y, s, family="serif",horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,**kwargs)
def plot_hv_text_right(ax,val,x=1-0.025,y=0.975,**kwargs):
    s = '$h\\nu='+str(val)+'$~eV'
    ax.text(x, y, s, family="serif",horizontalalignment='right',verticalalignment='top', transform=ax.transAxes,**kwargs)



def parse_point_labels(name:str, bar=False):
    special_points = {'G': r'\Gamma',
                      'D': r'\Delta'}
    try:
        name = special_points[name]
    except KeyError:
        name = r'\mathrm{{{}}}'.format(name)
    if bar:
        return r'$\overline{{{}}}$'.format(name)
    else:
        return r'${}$'.format(name)
    


def mark_points(pts, labels, bar=False, ax=None):
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, (tuple, list, np.ndarray)):
        ax = [ax]
    for a in np.array(ax, dtype=object).flatten():
        label_ax = a.twiny()
        label_ax.set_xlim(a.get_xlim())
        label_ax.set_xticks(pts)
        # label_ax.set_xlabel('')
        label_ax.set_xticklabels([parse_point_labels(l, bar) for l in labels])
