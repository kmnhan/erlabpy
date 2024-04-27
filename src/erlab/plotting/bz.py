"""Utilities for plotting Brillouin zones."""

__all__ = ["get_bz_edge", "plot_hex_bz"]

import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.spatial
from matplotlib.patches import RegularPolygon

from erlab.plotting.colors import axes_textcolor

abbrv_kws: dict[str, tuple[str, Any]] = {
    "facecolor": ("fc", "none"),
    "linestyle": ("ls", "--"),
    "linewidth": ("lw", 0.5),
}


def get_bz_edge(
    basis: npt.NDArray[np.float64],
    reciprocal: bool = True,
    extend: tuple[int, ...] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the edge of the first Brillouin zone (BZ) from lattice vectors.

    Parameters
    ----------
    basis
        ``(N, N)`` numpy array where ``N = 2``or ``3`` with each row containing the
        lattice vectors.
    reciprocal
        If `False`, the `basis` are given in real space lattice vectors.
    extend
        Tuple of positive integers specifying the number of times to extend the BZ in
        each direction. If `None`, only the first BZ is returned (equivalent to ``(1,) *
        N``).

    Returns
    -------
    lines : array-like
        ``(M, 2, N)`` array that specifies the endpoints of the ``M`` lines that make up
        the BZ edge, where ``N = len(basis)``.
    vertices : array-like
        Vertices of the BZ.

    """
    if basis.shape == (2, 2):
        ndim = 2
    elif basis.shape == (3, 3):
        ndim = 3
    else:
        raise ValueError("Shape of `basis` must be (N, N) where N = 2 or 3.")

    if not reciprocal:
        basis = 2 * np.pi * np.linalg.inv(basis).T

    if extend is None:
        extend = (1,) * ndim

    points = (
        np.tensordot(basis, np.mgrid[[slice(-1, 2) for _ in range(ndim)]], axes=(0, 0))
        .reshape((ndim, 3**ndim))
        .T
    )

    # Get index of origin
    zero_ind = np.where((points == 0).all(axis=1))[0][0]

    vor = scipy.spatial.Voronoi(points)

    lines = []
    vertices = []

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
        simplex = np.asarray(simplex)
        if zero_ind in pointidx:
            # If the origin is included in the ridge, add the vertices
            lines.append(vor.vertices[np.r_[simplex, simplex[0]]])
            vertices.append(vor.vertices[simplex])

    # Remove duplicates
    lines_new: list[npt.NDArray] = []
    vertices_new: list[npt.NDArray] = []

    for line in lines:
        for i in range(line.shape[0] - 1):
            if not any(
                np.allclose(line[i : i + 2], line_new)
                or np.allclose(line[i : i + 2], np.flipud(line_new))
                for line_new in lines_new
            ):
                lines_new.append(line[i : i + 2])
    for v in np.r_[*vertices]:
        if not any(np.allclose(v, vn) for vn in vertices_new):
            vertices_new.append(v)

    lines_arr = np.asarray(lines_new)
    vertices_arr = np.asarray(vertices_new)

    # Extend the BZ
    additional_lines = []
    additional_verts = []
    for vals in itertools.product(*[range(-n + 1, n) for n in extend]):
        if vals != (0,) * ndim:
            displacement = np.dot(vals, basis)
            additional_lines.append(lines_arr + displacement)
            additional_verts.append(vertices_arr + displacement)
    lines_arr = np.r_[lines_arr, *additional_lines]
    vertices_arr = np.r_[vertices_arr, *additional_verts]

    return lines_arr, vertices_arr


def plot_hex_bz(
    a=3.54, rotate=0.0, offset=(0.0, 0.0), reciprocal=True, ax=None, **kwargs
):
    """Plot a 2D hexagonal BZ overlay on the specified axes."""
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
    if reciprocal:
        r = 4 * np.pi / (a * 3)
    else:
        r = 2 * a
    clip = kwargs.pop("clip_path", None)
    poly = RegularPolygon(offset, 6, radius=r, orientation=np.deg2rad(rotate), **kwargs)
    ax.add_patch(poly)
    if clip is not None:
        poly.set_clip_path(clip)
    return poly
