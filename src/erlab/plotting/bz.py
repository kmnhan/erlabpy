"""Utilities for plotting Brillouin zones."""

__all__ = [
    "plot_bz",
    "plot_hex_bz",
    "plot_in_plane_bz",
    "plot_out_of_plane_bz",
]

import typing

import matplotlib.collections
import matplotlib.lines
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


def _plot_bz_slice(
    segments: npt.NDArray[np.floating],
    vertex_points: npt.NDArray[np.floating],
    midpoint_points: npt.NDArray[np.floating],
    *,
    ax: matplotlib.axes.Axes,
    vertices: bool,
    midpoints: bool,
    vertex_kwargs: dict[str, typing.Any] | None,
    midpoint_kwargs: dict[str, typing.Any] | None,
    line_kwargs: dict[str, typing.Any],
) -> tuple[
    tuple[matplotlib.lines.Line2D, ...],
    matplotlib.collections.PathCollection | None,
    matplotlib.collections.PathCollection | None,
]:
    if "color" not in line_kwargs and "c" not in line_kwargs:
        line_kwargs["color"] = line_kwargs.pop("edgecolor", None)
        if line_kwargs["color"] is None:
            line_kwargs["color"] = line_kwargs.pop("ec", axes_textcolor(ax))
    else:
        line_kwargs.pop("edgecolor", None)
        line_kwargs.pop("ec", None)
    line_kwargs["linestyle"] = line_kwargs.pop("linestyle", line_kwargs.pop("ls", "--"))
    line_kwargs["linewidth"] = line_kwargs.pop("linewidth", line_kwargs.pop("lw", 0.5))
    line_kwargs.setdefault("zorder", 5)

    lines: list[matplotlib.lines.Line2D] = []
    for segment in segments:
        (line,) = ax.plot(segment[:, 0], segment[:, 1], **line_kwargs)
        lines.append(line)

    color = line_kwargs.get("color", line_kwargs.get("c", axes_textcolor(ax)))
    vertex_artist = None
    midpoint_artist = None
    if vertices:
        vertex_kwargs = {"marker": "o", "s": 20, "zorder": line_kwargs["zorder"]} | (
            vertex_kwargs or {}
        )
        if "color" not in vertex_kwargs and "c" not in vertex_kwargs:
            vertex_kwargs["color"] = color
        vertex_artist = ax.scatter(
            vertex_points[:, 0],
            vertex_points[:, 1],
            **vertex_kwargs,
        )
    if midpoints:
        midpoint_kwargs = {
            "marker": "x",
            "s": 20,
            "zorder": line_kwargs["zorder"],
        } | (midpoint_kwargs or {})
        if "color" not in midpoint_kwargs and "c" not in midpoint_kwargs:
            midpoint_kwargs["color"] = color
        midpoint_artist = ax.scatter(
            midpoint_points[:, 0],
            midpoint_points[:, 1],
            **midpoint_kwargs,
        )
    return tuple(lines), vertex_artist, midpoint_artist


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


def plot_in_plane_bz(
    bvec: npt.NDArray[np.floating],
    *,
    kz: float = 0.0,
    angle: float = 0.0,
    bounds: tuple[float, float, float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    vertices: bool = False,
    midpoints: bool = False,
    vertex_kwargs: dict[str, typing.Any] | None = None,
    midpoint_kwargs: dict[str, typing.Any] | None = None,
    **line_kwargs,
) -> tuple[
    tuple[matplotlib.lines.Line2D, ...],
    matplotlib.collections.PathCollection | None,
    matplotlib.collections.PathCollection | None,
]:
    """Plot Brillouin-zone boundaries on a constant-``kz`` plane.

    Parameters
    ----------
    bvec
        Reciprocal lattice basis vectors.
    kz
        Out-of-plane momentum of the slice.
    angle
        Rotation angle in degrees about the ``kz`` axis.
    bounds
        ``(kx_min, kx_max, ky_min, ky_max)`` bounds. If `None`, bounds are inferred
        from the current axes limits.
    ax
        The axes to plot the BZ boundaries on. If `None`, the current axes are used.
    vertices
        If `True`, also mark BZ vertices.
    midpoints
        If `True`, also mark segment midpoints.
    vertex_kwargs, midpoint_kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.scatter`
        for vertices and midpoints.
    **line_kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    tuple
        Line artists, vertex scatter artist, and midpoint scatter artist.

    """
    if ax is None:
        ax = plt.gca()
    if bounds is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xmin, xmax = sorted((float(x0), float(x1)))
        ymin, ymax = sorted((float(y0), float(y1)))
        bounds = (xmin, xmax, ymin, ymax)
    bvec = np.asarray(bvec, dtype=float)
    segments, vertex_points, midpoint_points = erlab.lattice.get_in_plane_bz(
        bvec, kz=kz, angle=angle, bounds=bounds, return_midpoints=True
    )
    return _plot_bz_slice(
        segments,
        vertex_points,
        midpoint_points,
        ax=ax,
        vertices=vertices,
        midpoints=midpoints,
        vertex_kwargs=vertex_kwargs,
        midpoint_kwargs=midpoint_kwargs,
        line_kwargs=line_kwargs,
    )


def plot_out_of_plane_bz(
    bvec: npt.NDArray[np.floating],
    *,
    k_parallel: float = 0.0,
    angle: float = 0.0,
    bounds: tuple[float, float, float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    vertices: bool = False,
    midpoints: bool = False,
    vertex_kwargs: dict[str, typing.Any] | None = None,
    midpoint_kwargs: dict[str, typing.Any] | None = None,
    **line_kwargs,
) -> tuple[
    tuple[matplotlib.lines.Line2D, ...],
    matplotlib.collections.PathCollection | None,
    matplotlib.collections.PathCollection | None,
]:
    """Plot Brillouin-zone boundaries on an out-of-plane momentum slice.

    Parameters
    ----------
    bvec
        Reciprocal lattice basis vectors.
    k_parallel
        Fixed in-plane momentum component along ``angle``.
    angle
        Angle in degrees of the fixed in-plane momentum direction.
    bounds
        ``(kp_min, kp_max, kz_min, kz_max)`` bounds. If `None`, bounds are inferred
        from the current axes limits.
    ax
        The axes to plot the BZ boundaries on. If `None`, the current axes are used.
    vertices
        If `True`, also mark BZ vertices.
    midpoints
        If `True`, also mark segment midpoints.
    vertex_kwargs, midpoint_kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.scatter`
        for vertices and midpoints.
    **line_kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    tuple
        Line artists, vertex scatter artist, and midpoint scatter artist.

    """
    if ax is None:
        ax = plt.gca()
    if bounds is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xmin, xmax = sorted((float(x0), float(x1)))
        ymin, ymax = sorted((float(y0), float(y1)))
        bounds = (xmin, xmax, ymin, ymax)
    bvec = np.asarray(bvec, dtype=float)
    segments, vertex_points, midpoint_points = erlab.lattice.get_out_of_plane_bz(
        bvec,
        k_parallel=k_parallel,
        angle=angle,
        bounds=bounds,
        return_midpoints=True,
    )
    return _plot_bz_slice(
        segments,
        vertex_points,
        midpoint_points,
        ax=ax,
        vertices=vertices,
        midpoints=midpoints,
        vertex_kwargs=vertex_kwargs,
        midpoint_kwargs=midpoint_kwargs,
        line_kwargs=line_kwargs,
    )
