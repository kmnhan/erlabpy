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
    "spherical_mask",
]

from collections.abc import Hashable, Iterable

import numba
import numpy as np
import numpy.typing as npt
import xarray as xr

from erlab.analysis.mask import polygon


def mask_with_polygon(
    arr: xr.DataArray,
    vertices: npt.NDArray[np.floating],
    dims: Iterable[Hashable] = ("kx", "ky"),
    invert: bool = False,
):
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


def spherical_mask(
    darr: xr.DataArray,
    radius: float | dict[Hashable, float],
    boundary: bool = True,
    **sel_kw,
) -> xr.DataArray:
    """Generate a spherical boolean mask.

    Depending on the radius and dimensions provided, the mask will be hyperellipsoid in
    the dimensions specified in sel_kw. Points at the boundary are included in the mask.

    The resulting mask can be used with :meth:`xarray.DataArray.where` to mask the data.

    Parameters
    ----------
    darr
        The input DataArray.
    radius
        The radius of the spherical mask. If a single number, the same radius is used
        for all dimensions. If a dictionary, the keys should match the dimensions
        provided in sel_kw.
    boundary
        Whether to consider points on the boundary to be inside the mask. Default is
        `True`.
    **sel_kw
        Keyword arguments for selecting specific dimensions and values. Must be a
        mapping of valid dimension names to coordinate values.

    Returns
    -------
    mask : xr.DataArray
        A boolean mask indicating whether each point in the data array is within the
        spherical mask.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from erlab.analysis.mask import spherical_mask
    >>> darr = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
    >>> darr
    <xarray.DataArray (x: 5, y: 5)> Size: 200B
    array([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24]])
    Dimensions without coordinates: x, y
    >>> spherical_mask(darr, radius=2, x=2, y=2)
    <xarray.DataArray (x: 5, y: 5)> Size: 25B
    array([[False, False,  True, False, False],
        [False,  True,  True,  True, False],
        [ True,  True,  True,  True,  True],
        [False,  True,  True,  True, False],
        [False, False,  True, False, False]])
    Dimensions without coordinates: x, y
    >>> spherical_mask(darr, radius=1, x=2)
    <xarray.DataArray (x: 5)> Size: 5B
    array([False,  True,  True,  True, False])
    Dimensions without coordinates: x
    """
    if isinstance(radius, dict):
        if set(radius.keys()) != set(sel_kw.keys()):
            raise ValueError("Keys in radius and sel_kw must match")

    if len(sel_kw) == 0:
        raise ValueError("No dimensions provided for mask")

    delta_squared = xr.DataArray(0.0)

    for k, v in sel_kw.items():
        if k not in darr.dims:
            raise ValueError(f"Dimension {k} not found in data")

        if isinstance(radius, dict):
            r = radius[k]
        else:
            r = float(radius)

        delta_squared = delta_squared + ((darr[k] - v) / r) ** 2

    if boundary:
        return delta_squared <= 1.0
    else:
        return delta_squared < 1.0


def mask_with_hex_bz(
    kxymap: xr.DataArray, a: float = 3.54, rotate: float = 0.0, invert: bool = False
) -> xr.DataArray:
    """Mask an ARPES map with a hexagonal Brillouin zone.

    Parameters
    ----------
    kxymap
        The input map to be masked.
    a
        The lattice constant of the hexagonal BZ. Default is 3.54.
    rotate
        The rotation angle of the BZ in degrees. Default is 0.0.
    invert
        Whether to invert the mask. Default is False.

    Returns
    -------
    masked : xr.DataArray
        The masked map.

    """
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
    """Return a mask for given points."""
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
