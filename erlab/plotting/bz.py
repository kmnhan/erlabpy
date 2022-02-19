"""Utilities for plotting Brillouin zones."""

import numpy as np
import matplotlib.pyplot as plt
from arpes.plotting.bz import bz_plot
from arpes.utilities import bz
from scipy.spatial.transform import Rotation

__all__ = ['plot_hex_bz']

def plot_hex_bz(a=3.54, rotate=0, ax=None, **kwargs):
    """
    Plots a 2D hexagonal BZ overlay on the specified axes.
    """
    kwargs.setdefault('alpha',1)
    kwargs.setdefault('c','k')
    kwargs.setdefault('linestyle','-')
    kwargs.setdefault('lw',0.5)
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
