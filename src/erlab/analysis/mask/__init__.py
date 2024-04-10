"""Functions related to masking.

Polygon masking is adapted from the `CGAL
<https://doc.cgal.org/5.3.2/Polygon/index.html>`_ library. More information on
point-in-polygon strategies can be found in Ref. :cite:p:`schirra2008polygon`.

.. currentmodule:: erlab.analysis.mask

Modules
=======

.. autosummary::
   :toctree:

   polygon

"""

__all__ = [
    "hex_bz_mask_points",
    "mask_with_hex_bz",
    "mask_with_polygon",
    "polygon_mask",
    "polygon_mask_points",
]

import numba
import numpy as np
import numpy.typing as npt
import xarray as xr

from erlab.analysis.mask import polygon


def mask_with_polygon(arr, vertices, dims=("kx", "ky"), invert=False):
    mask = xr.DataArray(
        polygon_mask(
            vertices.astype(np.float64), *(arr[d].values for d in dims), invert=invert
        ),
        dims=dims,
        coords={d: arr.coords[d] for d in dims},
    )
    return arr.where(mask)


@numba.njit(parallel=True, cache=True)
def polygon_mask(
    vertices: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    invert: bool = False,
) -> npt.NDArray[np.bool_]:
    """Create a mask based on a polygon defined by its vertices.

    Parameters
    ----------
    vertices
        The vertices of the polygon. The shape should be (N, 2), where N is the number
        of vertices.
    x
        The x-coordinates of the grid points.
    y
        The y-coordinates of the grid points.
    invert
        If `True`, invert the mask (i.e., set `True` where the polygon is outside and
        `False` where it is inside). Default is `False`.

    Returns
    -------
    mask : ndarray
        The mask array with shape ``(len(x), len(y))``. The mask contains `True` where
        the polygon is inside and `False` where it is outside (or vice versa if `invert`
        is `True`).

    Note
    ----
    This function uses the `erlab.analysis.mask.polygon.bounded_side_bool` to determine
    whether a point is inside or outside the polygon.

    Example
    -------
    >>> vertices = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2]])
    >>> x = np.linspace(0, 1, 5)
    >>> y = np.linspace(0, 1, 5)
    >>> polygon_mask(vertices, x, y)
    array([[False, False, False, False, False],
           [False,  True,  True,  True, False],
           [False,  True,  True,  True, False],
           [False,  True,  True,  True, False],
           [False, False, False, False, False]])
    """
    shape = (len(x), len(y))
    mask = np.empty(shape, dtype=np.bool_)
    vertices = np.concatenate((vertices, vertices[0].reshape(1, -1)))

    for i in numba.prange(shape[0]):
        for j in range(shape[1]):
            mask[i, j] = polygon.bounded_side_bool(vertices, (x[i], y[j]))
    if invert:
        return ~mask
    else:
        return mask


@numba.njit(parallel=True, cache=True)
def polygon_mask_points(
    vertices: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    invert: bool = False,
) -> npt.NDArray[np.bool_]:
    """Compute a mask indicating whether points are inside or outside a polygon.

    Parameters
    ----------
    vertices
        The vertices of the polygon. The shape should be (N, 2), where N is the number
        of vertices.
    x
        The x-coordinates of the points.
    y
        The y-coordinates of the points.
    invert
        If `True`, invert the mask (i.e., set `True` where the polygon is outside and
        `False` where it is inside). Default is `False`.

    Returns
    -------
    mask : ndarray
        A boolean array of shape ``(len(x),)`` indicating whether each point is inside
        or outside the polygon.

    Raises
    ------
    ValueError
        If the lengths of x and y are not equal.

    Notes
    -----
    This function uses the `erlab.analysis.mask.polygon.bounded_side_bool` to determine
    whether a point is inside or outside the polygon.
    """
    if len(x) != len(y):
        raise ValueError
    mask = np.empty(len(x), dtype=np.bool_)
    vertices = np.concatenate((vertices, vertices[0].reshape(1, -1)))
    for i in numba.prange(len(x)):
        mask[i] = polygon.bounded_side_bool(vertices, (x[i], y[i]))
    if invert:
        return ~mask
    else:
        return mask


def mask_with_hex_bz(
    kxymap: xr.DataArray, a: float = 3.54, rotate: float = 0.0, invert: bool = False
) -> xr.DataArray:
    """Returns map masked with a hexagonal BZ."""

    if "kx" in kxymap.dims or "qx" in kxymap.dims:
        dims = ("kx", "ky")

    d = 2 * np.pi / (a * 3)
    ang = rotate + np.array([0, 60, 120, 180, 240, 300])
    vertices = np.array(
        [[2 * d * np.cos(t), 2 * d * np.sin(t)] for t in np.deg2rad(ang)]
    )
    return mask_with_polygon(kxymap, vertices, dims, invert=invert)


def hex_bz_mask_points(
    x,
    y,
    a: float = 3.54,
    rotate: float = 0,
    offset: tuple[float, float] = (0.0, 0.0),
    reciprocal: bool = False,
    invert: bool = False,
) -> npt.NDArray[np.bool_]:
    """Returns a mask for given points."""

    if reciprocal:
        d = 2 * np.pi / (a * 3)
    else:
        d = a
    ang = rotate + np.array([0, 60, 120, 180, 240, 300])
    vertices = np.array(
        [
            [2 * d * np.cos(t) + offset[0], 2 * d * np.sin(t) + offset[1]]
            for t in np.deg2rad(ang)
        ]
    )
    return polygon_mask_points(vertices, x, y, invert)
