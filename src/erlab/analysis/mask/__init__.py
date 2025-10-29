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
    "mask_with_bz",
    "mask_with_hex_bz",
    "mask_with_polygon",
    "mask_with_radius",
    "mask_with_regular_polygon",
]

import typing
from collections.abc import Hashable

import numba
import numpy as np
import numpy.typing as npt
import xarray as xr

import erlab
from erlab.analysis.mask import polygon


@numba.njit(parallel=True, cache=True)
def _polygon_mask(
    vertices: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    invert: bool = False,
) -> npt.NDArray[np.bool_]:
    """Create a mask based on a polygon defined by its vertices.

    Parameters
    ----------
    vertices : array-like of shape (N, 2)
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
    return mask


@numba.njit(parallel=True, cache=True)
def _polygon_mask_points(
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

    Note
    ----
    This function uses :func:`erlab.analysis.mask.polygon.bounded_side_bool` to
    determine whether a point is inside or outside the polygon.
    """
    if len(x) != len(y):
        raise ValueError
    mask = np.empty(len(x), dtype=np.bool_)
    vertices = np.concatenate((vertices, vertices[0].reshape(1, -1)))
    for i in numba.prange(len(x)):
        mask[i] = polygon.bounded_side_bool(vertices, (x[i], y[i]))
    if invert:
        return ~mask
    return mask


def _spherical_mask(
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
    if isinstance(radius, dict) and set(radius.keys()) != set(sel_kw.keys()):
        raise ValueError("Keys in radius and sel_kw must match")

    if len(sel_kw) == 0:
        raise ValueError("No dimensions provided for mask")

    delta_squared = xr.DataArray(0.0)

    for k, v in sel_kw.items():
        if k not in darr.dims:
            raise ValueError(f"Dimension {k} not found in data")

        r = radius[k] if isinstance(radius, dict) else float(radius)

        delta_squared = delta_squared + ((darr[k] - v) / r) ** 2

    if boundary:
        return delta_squared <= 1.0
    return delta_squared < 1.0


def mask_with_polygon(
    darr: xr.DataArray,
    vertices: npt.NDArray[np.floating],
    *,
    dims: tuple[Hashable, Hashable] | None = None,
    invert: bool = False,
    drop: bool = False,
):
    """Mask an :class:`xarray.DataArray` by a polygon in coordinate space.

    Builds a boolean mask from a 2D polygon in the plane spanned by two coordinates
    (specified by ``dims``) and returns a DataArray with values outside the polygon set
    to NaN (or dropped if drop=True). If ``invert=True``, the selection is inverted so
    that values inside the polygon are masked instead.

    Parameters
    ----------
    darr : DataArray
        Input data to be masked. Must contain the coordinate variables named in dims.
    vertices : array-like of shape (N, 2)
        Polygon vertices as [x, y] pairs in the order as ``dims``. The polygon is
        treated as closed; the last vertex will be implicitly connected to the first if
        needed.
    dims
        Names of the two dimensions over which the polygon is defined. The column order
        of vertices must correspond to (dims[0], dims[1]). If None (default), the first
        two dimensions of the data are used.
    invert
        If `True`, invert the mask so that points inside the polygon are masked and
        points outside are kept.
    drop
        If `True`, drop coordinate labels along dims for which all values are masked.
        This typically reduces the extent to a bounding rectangle of the polygon; masked
        points inside that extent remain NaN.

    Returns
    -------
    DataArray
        The masked DataArray.

    """
    if dims is None:
        dims = typing.cast("tuple[Hashable, Hashable]", darr.dims[:2])

    mask = xr.DataArray(
        _polygon_mask(
            vertices.astype(np.float64), *(darr[d].values for d in dims), invert=invert
        ),
        dims=dims,
        coords={d: darr.coords[d] for d in dims},
    )
    return darr.where(mask, drop=drop)


def mask_with_regular_polygon(
    darr: xr.DataArray,
    n_verts: int,
    radius: float,
    *,
    rotate: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    dims: tuple[Hashable, Hashable] | None = None,
    invert: bool = False,
    drop: bool = False,
):
    """Mask an xarray.DataArray by a regular polygon in coordinate space.

    Builds a boolean mask from a regular 2D polygon in the plane spanned by two
    coordinates (specified by ``dims``) and returns a DataArray with values outside the
    polygon set to NaN (or dropped if drop=True). If ``invert=True``, the selection is
    inverted so that values inside the polygon are masked instead. The polygon is
    oriented so that a vertice is at the top (along the positive y-axis) when
    ``rotate=0``.

    Parameters
    ----------
    darr : DataArray
        Input data to be masked. Must contain the coordinate variables named in dims.
    n_verts : int
        Number of vertices of the regular polygon.
    radius : float
        Distance from the polygon center to each vertex.
    rotate : float
        Rotation angle of the polygon in degrees. Default is 0.0.
    offset : tuple of float
        Offset for the polygon center in the form of a tuple. Default is (0.0, 0.0).
    dims
        Names of the two dimensions over which the polygon is defined. The column order
        of vertices must correspond to (dims[0], dims[1]). If None (default), the first
        two dimensions of the data are used.
    invert
        If `True`, invert the mask so that points inside the polygon are masked and
        points outside are kept.
    drop
        If `True`, drop coordinate labels along dims for which all values are masked.
        This typically reduces the extent to a bounding rectangle of the polygon; masked
        points inside that extent remain NaN.

    Returns
    -------
    DataArray
        The masked DataArray.
    """
    # Align polygon so that a vertex is at the top
    angles = (
        (2 * np.pi / n_verts) * np.arange(n_verts + 1) + np.pi / 2 + np.deg2rad(rotate)
    )
    vertices = np.column_stack(
        (radius * np.cos(angles) + offset[0], radius * np.sin(angles) + offset[1])
    )
    return mask_with_polygon(darr, vertices, dims=dims, invert=invert, drop=drop)


def mask_with_bz(
    darr: xr.DataArray,
    basis: npt.NDArray[np.floating],
    *,
    reciprocal: bool = False,
    rotate: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    dims: tuple[Hashable, Hashable] | None = None,
    invert: bool = False,
    drop: bool = False,
) -> xr.DataArray:
    """Mask an xarray.DataArray by a 2D Brillouin zone defined by lattice basis vectors.

    The Brillouin zone is constructed from the provided lattice basis vectors. If
    ``reciprocal=True``, the basis vectors are interpreted as those of the reciprocal
    lattice; otherwise, they are the real space lattice basis vectors. The polygon is
    oriented so that a vertex is at the top (along the positive y-axis) when
    ``rotate=0``.

    Parameters
    ----------
    darr : DataArray
        Input data to be masked. Must contain the coordinate variables named in dims.
    basis
        A 2D or 3D numpy array with shape ``(N, N)`` where ``N = 2`` or ``3``,
        containing the basis vectors of the lattice. If N is 3, only the upper left 2x2
        submatrix is used.
    reciprocal : bool
        If `True`, the basis vectors are interpreted as those of the reciprocal lattice.
        Default is `False`, i.e., the basis vectors are treated as real space lattice
        vectors.
    rotate : float
        Rotation angle of the BZ in degrees. Default is 0.0.
    offset : tuple of float
        Offset for the zone center in the form of a tuple. Default is (0.0, 0.0).
    dims
        Names of the two dimensions over which the BZ is defined. The column order of
        vertices must correspond to (dims[0], dims[1]). If None (default) and the
        dimensions ``('kx', 'ky')`` exist in the data, those are used. Otherwise, the
        first two dimensions of the data are used.
    invert
        If `True`, invert the mask so that points inside the BZ are masked and points
        outside are kept.
    drop
        If `True`, drop coordinate labels along dims for which all values are masked.
        This typically reduces the extent to a bounding rectangle of the BZ; masked
        points inside that extent remain NaN.

    Returns
    -------
    DataArray
        The masked DataArray.
    """
    if dims is None and "kx" in darr.dims and "ky" in darr.dims:
        dims = ("kx", "ky")
    vertices = erlab.lattice.get_2d_vertices(
        basis, reciprocal=reciprocal, rotate=rotate, offset=offset
    )
    return mask_with_polygon(darr, vertices, dims=dims, invert=invert, drop=drop)


def mask_with_hex_bz(
    darr: xr.DataArray,
    a: float,
    *,
    reciprocal: bool = False,
    rotate: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    dims: tuple[Hashable, Hashable] | None = None,
    invert: bool = False,
    drop: bool = False,
) -> xr.DataArray:
    """Mask an xarray.DataArray by a 2D hexagon.

    The hexagonal Brillouin zone is constructed from the lattice constant ``a``. If
    ``reciprocal=True``, ``a`` is interpreted as the periodicity of the reciprocal
    lattice; otherwise, it is the real space lattice constant.  The hexagon is oriented
    so that a vertex is at the top (along the positive y-axis) when ``rotate=0``.

    Parameters
    ----------
    darr : DataArray
        Input data to be masked. Must contain the coordinate variables named in dims.
    a : float
        Lattice constant of the hexagonal lattice.
    reciprocal : bool
        If `True`, ``a`` is interpreted as the periodicity of the reciprocal lattice.
        Default is `False`.
    rotate : float
        Rotation angle of the hexagon in degrees. Default is 0.0.
    offset : tuple of float
        Offset for the hexagon center in the form of a tuple. Default is (0.0, 0.0).
    dims
        Names of the two dimensions over which the BZ is defined. The column order of
        vertices must correspond to (dims[0], dims[1]). If None (default) and the
        dimensions ``('kx', 'ky')`` exist in the data, those are used. Otherwise, the
        first two dimensions of the data are used.
    invert
        If `True`, invert the mask so that points inside the hexagon are masked and
        points outside are kept.
    drop
        If `True`, drop coordinate labels along dims for which all values are masked.
        This typically reduces the extent to a bounding rectangle of the hexagon; masked
        points inside that extent remain NaN.

    Returns
    -------
    DataArray
        The masked DataArray.
    """
    if dims is None and "kx" in darr.dims and "ky" in darr.dims:
        dims = ("kx", "ky")
    return mask_with_regular_polygon(
        darr,
        n_verts=6,
        radius=a / np.sqrt(3) if reciprocal else 4 * np.pi / (a * 3),
        rotate=rotate,
        offset=offset,
        dims=dims,
        invert=invert,
        drop=drop,
    )


def mask_with_radius(
    darr: xr.DataArray,
    radius: float | dict[Hashable, float],
    *,
    boundary: bool = True,
    invert: bool = False,
    drop: bool = False,
    **sel_kw,
) -> xr.DataArray:
    """
    Average data within a specified radius of a specified point.

    For instance, consider an ARPES map with dimensions ``'kx'``, ``'ky'``, and
    ``'eV'``. Providing ``'kx'`` and ``'ky'`` points will average the data within a
    cylindrical region centered at that point. The radius of the cylinder is specified
    by ``radius``.

    If different radii are given for ``kx`` and ``ky``, the region will be elliptic.

    Parameters
    ----------
    radius
        The radius of the region. If a single number, the same radius is used for all
        dimensions. If a dictionary, keys must be valid dimension names and the values
        are the radii for the corresponding dimensions.
    boundary
        Whether to consider points on the boundary to be inside the mask. Default is
        `True`.
    invert
        If `True`, invert the mask so that points inside are masked and points outside
        are kept. This is applied after the mask is created.
    drop
        If `True`, drop coordinate labels along dims for which all values are masked.
    **sel_kw
        The center of the spherical region. Must be a mapping of valid dimension names
        to coordinate values.

    Returns
    -------
    DataArray
        The data masked to the specified region.

    Note
    ----
    The region is defined by a spherical mask. Depending on the radius and dimensions
    provided, the mask will be hyperellipsoid in the dimensions specified in ``sel_kw``.

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
    <xarray.DataArray (x: 5, y: 5)> Size: 200B
    array([[nan, nan,  2., nan, nan],
        [nan,  6.,  7.,  8., nan],
        [10., 11., 12., 13., 14.],
        [nan, 16., 17., 18., nan],
        [nan, nan, 22., nan, nan]])
    Dimensions without coordinates: x, y
    >>> spherical_mask(darr, radius=1, x=2)
    <xarray.DataArray (x: 5, y: 5)> Size: 200B
    array([[nan, nan, nan, nan, nan],
        [ 5.,  6.,  7.,  8.,  9.],
        [10., 11., 12., 13., 14.],
        [15., 16., 17., 18., 19.],
        [nan, nan, nan, nan, nan]])
    Dimensions without coordinates: x, y

    """
    mask = _spherical_mask(darr, radius, boundary=boundary, **sel_kw)
    if invert:
        mask = mask.copy(data=~mask.data)
    return darr.where(mask, drop=drop)
