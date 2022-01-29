import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from arpes.utilities import bz
from arpes.plotting.bz import bz_plot
from arpes.plotting.utils import name_for_dim, unit_for_dim
from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward

__all__ = ['proportional_colorbar','plot_hex_bz','plot_hv_text','label_subplots']

def proportional_colorbar(ax=None,**kwargs):
    """
    Creates or replaces the colorbar with proper proportional spacing.
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

def annotate_cuts(ax, data, plotted_axes, include_text_labels=False, rotation=0, offset=[0,0], plot_kw={}, text_kw={}, **kwargs):
    """Annotates a cut location onto a plot.
    Example:
        >>> annotate_cuts(ax, conv, ['kz', 'ky'], hv=80)
    Args:
        ax: The axes to plot onto
        data: The original data
        plotted_axes: The dimension names which were plotted
        include_text_labels: Whether to include text labels
        kwargs: Defines the coordinates of the cut location
    """
    converted_coordinates = convert_coordinates_to_kspace_forward(data)
    assert len(plotted_axes) == 2, 'Only 2D axes can be can be annotated'
    color = 'k'
    for k, v in kwargs.items():
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]

        selected = converted_coordinates.sel(**dict([[k, v]]), method="nearest")
        
        for coords_dict, obj in selected.G.iterate_axis(k):
            # css = [obj[d].values for d in plotted_axes]
            plt_css = [np.mean(obj[d].values,axis=1) for d in plotted_axes]
            ax.plot(*plt_css, color=color, ls='--', lw=0.85)
            if include_text_labels:
                idx = np.argmin(plt_css[0])
                # print([plt_css[0][idx],plt_css[1][idx]])
                ax.text(
                    plt_css[0][idx]+0.02+offset[0],
                    plt_css[1][idx]+0.04+offset[1],
                    "{} = {} {}".format(name_for_dim(k), int(np.rint(coords_dict[k].item())), unit_for_dim(k)),
                    color=color,
                    # size="medium",
                    horizontalalignment='left',
                    verticalalignment='top',
                    # transform=ax.transAxes
                    rotation=rotation,
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