"""Utilities for plotting Brillouin zones."""

__all__ = ["plot_bz", "plot_hex_bz"]

import typing

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import erlab
from erlab.plotting.colors import axes_textcolor

abbrv_kws: dict[str, tuple[str, typing.Any]] = {
    "facecolor": ("fc", "none"),
    "linestyle": ("ls", "--"),
    "linewidth": ("lw", 0.5),
}


def get_bz_edge(
    basis: npt.NDArray[np.floating],
    reciprocal: bool = True,
    extend: tuple[int, ...] | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:  # pragma: no cover
    """Calculate the edge of the first Brillouin zone (BZ) from lattice vectors.

    .. deprecated:: 3.14.1

        Use :func:`erlab.lattice.get_bz_edge` instead.

    """
    import warnings

    warnings.warn(
        "erlab.plotting.bz.get_bz_edge is deprecated, "
        "use erlab.lattice.get_bz_edge instead",
        FutureWarning,
        stacklevel=1,
    )
    return erlab.lattice.get_bz_edge(basis, reciprocal=reciprocal, extend=extend)


def plot_bz(
    basis: npt.NDArray[np.floating],
    *,
    reciprocal: bool = False,
    rotate: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> matplotlib.patches.Polygon:
    """Plot a Brillouin zone, given the basis vectors.

    Parameters
    ----------
    basis
        A 2D or 3D numpy array with shape ``(N, N)`` where ``N = 2`` or ``3``,
        containing the basis vectors of the lattice. If N is 3, only the upper left 2x2
        submatrix is used.
    reciprocal
        If `True`, the basis vectors are interpreted as reciprocal lattice vectors.
    rotate
        Rotation angle in degrees to apply to the BZ.
    offset
        Offset for the Brillouin zone center in the form of a tuple ``(x, y)``.
    ax
        The axes to plot the BZ on. If `None`, the current axes are used.
    **kwargs
        Additional keyword arguments passed to :class:`matplotlib.patches.Polygon`.

    """
    if ax is None:
        ax = plt.gca()

    # Populate default keyword arguments
    kwargs["edgecolor"] = kwargs.pop("edgecolor", kwargs.pop("ec", axes_textcolor(ax)))
    kwargs.setdefault("zorder", 5)
    kwargs.setdefault("closed", True)
    kwargs.setdefault("fill", False)
    for k, v in abbrv_kws.items():
        kwargs[k] = kwargs.pop(k, kwargs.pop(*v))

    patch = matplotlib.patches.Polygon(
        erlab.lattice.get_2d_vertices(
            basis, reciprocal=reciprocal, rotate=rotate, offset=offset
        ),
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def plot_hex_bz(
    a: float = 3.54,
    *,
    reciprocal: bool = False,
    rotate: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    ax: matplotlib.axes.Axes | None = None,
    **kwargs,
):
    """Plot a 2D hexagonal BZ overlay on the specified axes.

    Parameters
    ----------
    a
        Lattice constant of the hexagonal lattice.
    reciprocal
        If `True`, ``a`` is interpreted as the periodicity of the reciprocal lattice.
    rotate
        Rotation angle in degrees to apply to the BZ.
    offset
        Offset for the Brillouin zone center in the form of a tuple ``(x, y)``.
    ax
        The axes to plot the BZ on. If `None`, the current axes are used.
    **kwargs
        Additional keyword arguments passed to
        :class:`matplotlib.patches.RegularPolygon`.

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

    kwargs["edgecolor"] = kwargs.pop("edgecolor", kwargs.pop("ec", axes_textcolor(ax)))

    r = a / np.sqrt(3) if reciprocal else 4 * np.pi / (a * 3)

    clip = kwargs.pop("clip_path", None)
    poly = matplotlib.patches.RegularPolygon(
        offset, 6, radius=r, orientation=np.deg2rad(rotate), **kwargs
    )
    ax.add_patch(poly)
    if clip is not None:
        poly.set_clip_path(clip)
    return poly
