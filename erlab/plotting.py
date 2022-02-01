"""Plotting utilities."""

import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from arpes.utilities import bz
from arpes.plotting.bz import bz_plot
from arpes.plotting.utils import name_for_dim, unit_for_dim
from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward

__all__ = ['proportional_colorbar','plot_hex_bz','plot_hv_text','label_subplots','annotate_cuts_erlab']

def proportional_colorbar(ax=None,**kwargs):
    """Creates or replaces the colorbar with proportional spacing.

    The default behavior of colorbars in `matplotlib` does not support
    colors proportional to data in different norms. This function circumvents
    this behavior. 

    Returns
    -------
    cbar : `~matplotlib.colorbar.Colorbar`


    """
    if ax is None:
        ax = plt.gca()
    mappable = ax.collections[-1]
    if mappable.colorbar is not None:
        ticks = mappable.colorbar.get_ticks()
        mappable.colorbar.remove()
        kwargs.setdefault('ticks',ticks)
    cbar = plt.colorbar(
        ax=ax,
        mappable=mappable,
        spacing='proportional',
        boundaries=mappable.norm.inverse(np.linspace(0,1,mappable.cmap.N)),
        **kwargs,
    )
    return cbar

def plot_hex_bz(a=3.54,rotate=0,ax=None,**kwargs):
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
                        ax=None, include_text_labels=False,  textoffset=[0, 0], plot_kw={}, text_kw={}, **kwargs):
    r"""Annotates a cut location onto a plot. 

    Does the same job as `arpes.plotting.annotations.annotate_cuts`, but
    handles line styles and annotations much better. 

    Parameters
    ----------
    data : xarray.DataArray
        The data before momentum space conversion.
    plotted_dims: list of str
        The dimension names currently plotted on the target axes.
    ax : `~matplotlib.axes.Axes`, optional
        The `~.axes.Axes` instance in which the annotation is placed,
        defaults to the current axes when optional.
    include_text_labels: bool, default=False
        Wheter to include text labels.
    plot_kw : dict, optional
    text_kw : dict, optional
    textoffset : list of float or tuple of float
    **kwargs : dict
        Defines the coordinates of the cut location

    Examples
    --------
    >>> annotate_cuts(ax, conv, ['kz', 'ky'], hv=80)

    """
    converted_coordinates = convert_coordinates_to_kspace_forward(data)
    assert len(plotted_dims) == 2, 'Only 2D axes can be can be annotated'
    color = 'k'
    for k, v in kwargs.items():
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]

        selected = converted_coordinates.sel(**dict([[k,v]]), method='nearest')
        
        text_kw.setdefault('horizontalalignment','left')
        text_kw.setdefault('verticalalignment','top')
        text_kw.setdefault('color',color)
        
        for coords_dict, obj in selected.G.iterate_axis(k):
            # css = [obj[d].values for d in plotted_dims]
            plt_css = [np.mean(obj[d].values,axis=1) for d in plotted_dims]
            ax.plot(*plt_css, color=color, ls='--', lw=0.85)
            if include_text_labels:
                idx = np.argmin(plt_css[0])
                # print([plt_css[0][idx],plt_css[1][idx]])
                plt.text(
                    x = plt_css[0][idx]+0.02+textoffset[0],
                    y = plt_css[1][idx]+0.04+textoffset[1],
                    s = "{} = {} {}".format(
                        name_for_dim(k),
                        int(np.rint(coords_dict[k].item())),
                        unit_for_dim(k)
                    ),
                    # transform=ax.transAxes
                    **text_kw
                )

# TODO: fix format using name_for_dim and unit_for_dim, and change backend to matplotlib.pyplot.annotate
def plot_hv_text(ax,val,x=0.025,y=0.975,**kwargs):
    s = '$h\\nu='+str(val)+'$~eV'
    ax.text(x, y, s, family="serif",horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,**kwargs)
def plot_hv_text_right(ax,val,x=1-0.025,y=0.975,**kwargs):
    s = '$h\\nu='+str(val)+'$~eV'
    ax.text(x, y, s, family="serif",horizontalalignment='right',verticalalignment='top', transform=ax.transAxes,**kwargs)
def label_subplots(ax,val,x=-0.19,y=0.99,prefix='(',suffix=')',**kwargs):
    if isinstance(val, int):
        val = chr(val+96)
    s = '\\textbf{'+prefix+val+suffix+'}'
    ax.text(x,y, s,family="serif",horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,fontsize='large',**kwargs)