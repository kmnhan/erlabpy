"""Utilities for plotting Brillouin zones."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

__all__ = ["plot_hex_bz"]

abbrv_kws = dict(
    edgecolor=["ec", "k"],
    facecolor=["fc", "none"],
    linestyle=["ls", "--"],
    linewidth=["lw", 0.5],
)


def plot_hex_bz(a=3.54, rotate=0.0, offset=(0.0, 0.0), ax=None, **kwargs):
    """
    Plots a 2D hexagonal BZ overlay on the specified axes.
    """
    kwargs.setdefault("zorder", 5)
    for k, v in abbrv_kws.items():
        kwargs[k] = kwargs.pop(k, kwargs.pop(*v))
    if ax is None:
        ax = plt.gca()
    if np.iterable(ax):
        return [plot_hex_bz(a=a, rotate=rotate, offset=offset, ax=x, **kwargs) for x in ax]

    poly = RegularPolygon(
        offset, 6, radius=4 * np.pi / a / 3, orientation=np.deg2rad(rotate), **kwargs
    )
    ax.add_patch(poly)
    return poly
