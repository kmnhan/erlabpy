"""Utilities for plotting Brillouin zones."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from matplotlib.patches import RegularPolygon

from erlab.plotting.colors import axes_textcolor

__all__ = ["get_bz_edge", "plot_hex_bz"]

abbrv_kws = dict(
    facecolor=["fc", "none"],
    linestyle=["ls", "--"],
    linewidth=["lw", 0.5],
)


def get_bz_edge(avec):
    """
    Calculates the edge of the first Brillouin zone (BZ) from the real
    space lattice vectors.

    Parameters
    ----------
    avec : np.ndarray
        2 x 2 or 3 x 3 numpy array with each row containing the real space lattice
        vectors.

    Returns
    -------
    lines
        numpy array containing the endpoints of the lines that make up the BZ edge.
    vertices
        numpy array containing the vertices of the BZ.

    """
    if not (avec.shape == (2, 2) or avec.shape == (3, 3)):
        raise ValueError("Shape of `avec` must be (2, 2) or (3, 3).")

    ndim = avec.shape[-1]
    points = (
        np.tensordot(avec, np.mgrid[[slice(-1, 2) for _ in range(ndim)]], axes=[0, 0])
        .reshape((ndim, 3**ndim))
        .T
    )
    zero_ind = np.where((points == 0).all(axis=1))[0][0]

    vor = scipy.spatial.Voronoi(points)

    lines = []
    valid_indices = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if zero_ind in pointidx:
            lines.append(vor.vertices[simplex + [simplex[0]]])
            valid_indices += simplex
    vertices = vor.vertices[valid_indices]

    # remove duplicates
    lines_new = []
    for l in lines:
        for i in range(l.shape[0] - 1):
            if not any(
                np.allclose(l[i : i + 2], ln)
                or np.allclose(l[i : i + 2], np.flipud(ln))
                for ln in lines_new
            ):
                lines_new.append(l[i : i + 2])
    vertices_new = []
    for v in vertices:
        if not any(np.allclose(v, vn) for vn in vertices_new):
            vertices_new.append(v)

    return np.asarray(lines_new), np.asarray(vertices_new)


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
        return [
            plot_hex_bz(a=a, rotate=rotate, offset=offset, ax=x, **kwargs) for x in ax
        ]
    else:
        kwargs["edgecolor"] = kwargs.pop(
            "edgecolor", kwargs.pop("ec", axes_textcolor(ax))
        )

    poly = RegularPolygon(
        offset, 6, radius=4 * np.pi / a / 3, orientation=np.deg2rad(rotate), **kwargs
    )
    ax.add_patch(poly)
    return poly
