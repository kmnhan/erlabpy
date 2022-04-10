"""Utilities for plotting Brillouin zones."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from arpes.plotting.bz import bz_plot
from arpes.utilities import bz
from scipy.spatial.transform import Rotation

__all__ = ['plot_hex_bz']

abbrv_kws = dict(
    edgecolor=['ec', 'k'],
    facecolor=['fc', '#0000'],
    linestyle=['ls', '--'],
    linewidth=['lw', 0.5],
)

def plot_hex_bz(a=3.54, rotate=0.0, offset=(0.0, 0.0), ax=None, **kwargs):
    """
    Plots a 2D hexagonal BZ overlay on the specified axes.
    """
    kwargs.setdefault('zorder',5)
    for k, v in abbrv_kws.items():
        kwargs[k] = kwargs.pop(k, kwargs.pop(*v))
    if ax is None:
        ax = plt.gca()
        
    
    poly = RegularPolygon(offset, 6, 
                          radius=4 * np.pi / a / 3, 
                          orientation=np.deg2rad(rotate),
                          **kwargs)
    ax.add_patch(poly)
    # bz_plot(
    #     ax=ax,
    #     cell=bz.hex_cell_2d(a / (2 * np.pi)),
    #     paths=[],
    #     repeat=None,
    #     set_equal_aspect=False,
    #     hide_ax=False,
    #     transformations=[Rotation.from_rotvec([0, 0, rotate*np.pi/180])],
    #     **kwargs
    # )
