"""
Tools related to the real and reciprocal lattice.

.. currentmodule:: erlab.lattice

"""

__all__ = [
    "abc2avec",
    "angle_between",
    "avec2abc",
    "get_2d_vertices",
    "get_bz_edge",
    "get_bz_slice",
    "get_in_plane_bz",
    "get_out_of_plane_bz",
    "to_primitive",
    "to_real",
    "to_reciprocal",
]

import itertools
import typing

import numpy as np
import numpy.typing as npt


def angle_between(v1: npt.NDArray[np.floating], v2: npt.NDArray[np.floating]) -> float:
    """Return the angle between two vectors.

    Parameters
    ----------
    v1, v2 : array-like
        1D array of length 3, specifying a vector.

    Returns
    -------
    float
        The angle in degrees.
    """
    return float(np.rad2deg(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))


def abc2avec(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> npt.NDArray[np.floating]:
    r"""Construct lattice vectors from lattice parameters.

    Parameters
    ----------
    a
        Lattice parameter :math:`a`.
    b
        Lattice parameter :math:`b`.
    c
        Lattice parameter :math:`c`.
    alpha
        Lattice parameter :math:`\alpha` in degrees.
    beta
        Lattice parameter :math:`\beta` in degrees.
    gamma
        Lattice parameter :math:`\gamma` in degrees.

    Returns
    -------
    avec
        Real lattice vectors, given as a 3 by 3 numpy array with each basis vector given
        in each row.
    """
    alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    sa, ca, sb, cb, cg = (
        np.sin(alpha),
        np.cos(alpha),
        np.sin(beta),
        np.cos(beta),
        np.cos(gamma),
    )

    gp = np.arccos(np.clip((ca * cb - cg) / (sa * sb), -1.0, 1.0))
    cgp, sgp = np.cos(gp), np.sin(gp)
    return np.array(
        [
            [a * sb, 0, a * cb],
            [-b * sa * cgp, b * sa * sgp, b * ca],
            [0, 0, c],
        ]
    )


def avec2abc(
    avec: npt.NDArray[np.floating],
) -> tuple[float, float, float, float, float, float]:
    """Determine lattice parameters from lattice vectors.

    Parameters
    ----------
    avec
        Real lattice vectors, given as a 3 by 3 numpy array with each basis vector given
        in each row.

    Returns
    -------
    a, b, c, alpha, beta, gamma
    """
    a, b, c = tuple(float(np.linalg.norm(x)) for x in avec)
    alpha = angle_between(avec[1] / b, avec[2] / c)
    beta = angle_between(avec[2] / c, avec[0] / a)
    gamma = angle_between(avec[0] / a, avec[1] / b)
    return a, b, c, alpha, beta, gamma


def to_reciprocal(avec: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Construct the reciprocal lattice vectors from real lattice vectors.

    Parameters
    ----------
    avec
        Real lattice vectors, given as a 3 by 3 numpy array with each basis vector given
        in each row.

    Returns
    -------
    bvec
        The reciprocal lattice vectors.
    """
    return 2 * np.pi * np.linalg.inv(avec).T


def to_real(bvec: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Construct the real lattice vectors from reciprocal lattice vectors.

    Parameters
    ----------
    bvec
        Reciprocal lattice vectors, given as a 3 by 3 numpy array with each basis vector
        given in each row.

    Returns
    -------
    avec
        The real lattice vectors.
    """
    return np.linalg.inv(bvec.T / 2 / np.pi)


def _get_centering_matrix(
    centering_type: typing.Literal["P", "A", "B", "C", "F", "I", "R"],
) -> npt.NDArray:
    match centering_type:
        case "A":
            return np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0, -0.5, 0.5]])
        case "B":
            return np.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.5]])
        case "C":
            return np.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
        case "F":
            return np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        case "I":
            return np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
        case "R":
            return np.array([[-1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [-1.0, -2.0, 1.0]]) / 3
        case _:
            return np.eye(3)


def to_primitive(
    avec: npt.NDArray, centering_type: typing.Literal["P", "A", "B", "C", "F", "I", "R"]
) -> npt.NDArray:
    """Convert lattice vectors to primitive cell vectors.

    Transforms the given conventional cell lattice vectors into primitive cell basis
    given the centering type.

    Parameters
    ----------
    avec
        Conventional cell basis vectors, shape (3, 3). Each row represents a lattice
        vector in Cartesian coordinates.
    centering_type : {"P", "A", "B", "C", "I", "F", "R"}
        Bravais lattice centering type:

        - "P": Primitive (no change)
        - "A": Extra point on bc face
        - "B": Extra point on ac face
        - "C": Extra point on ab face
        - "F": Face-centered (extra points on all faces)
        - "I": Body-centered (extra point in cell center)
        - "R": Rhombohedral (in hexagonal setting)

    Returns
    -------
    avec_p
        Primitive cell lattice vectors, shape (3, 3). Each row represents a primitive
        lattice vector in Cartesian coordinates.

    Example
    -------
    >>> import erlab
    >>> avec = np.eye(3) * 2.0  # 2 Å conventional cubic cell
    >>> avec_prim = erlab.lattice.to_primitive(avec, "I")  # Body-centered cubic
    """
    return _get_centering_matrix(centering_type) @ avec


def get_bz_edge(
    basis: npt.NDArray[np.floating],
    reciprocal: bool = True,
    extend: tuple[int, ...] | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Calculate the edge of the first Brillouin zone (BZ) from lattice vectors.

    Parameters
    ----------
    basis
        ``(N, N)`` numpy array where ``N = 2`` or ``3`` with each row containing the
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
        basis = to_reciprocal(basis)

    if extend is None:
        extend = (1,) * ndim

    points = (
        np.tensordot(basis, np.mgrid[[slice(-1, 2) for _ in range(ndim)]], axes=(0, 0))
        .reshape((ndim, 3**ndim))
        .T
    )

    # Get index of origin
    zero_ind = np.where((points == 0).all(axis=1))[0][0]

    import scipy.spatial

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

    for v in np.concatenate(vertices):
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
    lines_arr = np.concatenate((lines_arr, *additional_lines))
    vertices_arr = np.concatenate((vertices_arr, *additional_verts))

    return lines_arr, vertices_arr


def get_2d_vertices(
    basis: npt.NDArray[np.floating],
    *,
    reciprocal: bool = True,
    rotate: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> npt.NDArray[np.floating]:
    """Get the vertices of a 2D Brillouin zone.

    Unlike :func:`get_bz_edge`, this function only returns the vertices of the BZ. The
    vertices are ordered such that they can be used to create a polygon. Also, this
    function allows for rotation and offset of the BZ.

    Parameters
    ----------
    basis
        A 2D or 3D numpy array with shape ``(N, N)`` where ``N = 2`` or ``3``,
        containing the basis vectors of the lattice. If N is 3, only the upper left 2x2
        submatrix is used.
    reciprocal
        If `True`, the ``basis`` are treated as reciprocal lattice vectors. If `False`,
        the `basis` are treated as real space lattice vectors.
    rotate
        Rotation angle in degrees to apply to the BZ.
    offset
        Offset for the Brillouin zone center in the form of a tuple ``(x, y)``.

    Returns
    -------
    vertices : array-like
        Vertices of the BZ.

    """
    lines, _ = get_bz_edge(
        np.asarray(basis)[:2, :2], reciprocal=reciprocal, extend=None
    )

    # Reconstruct ordered vertices for the polygon
    # Start from the first point, follow connections
    verts = [lines[0][0]]
    current = lines[0][1]
    used = {0}
    while len(used) < len(lines):
        for i, line in enumerate(lines):  # pragma: no branch
            if i in used:
                continue
            if np.allclose(line[0], current):
                verts.append(line[0])
                current = line[1]
                used.add(i)
                break
            if np.allclose(line[1], current):
                verts.append(line[1])
                current = line[0]
                used.add(i)
                break

    cx, sx = np.cos(np.deg2rad(rotate)), np.sin(np.deg2rad(rotate))
    rotation_matrix = np.array([[cx, sx], [-sx, cx]])
    verts = verts @ rotation_matrix
    verts += offset
    return verts


def _plane_frame(
    n: npt.NDArray[np.floating],
) -> tuple[
    npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
]:
    """Create basis vectors for a plane given normal vector n.

    The basis are chosen so that if n has a large z component, the first basis vector u
    is mostly along x, and if n has a large x component, u is mostly along z.

    Parameters
    ----------
    n : array-like
        Normal vector of the plane.

    Returns
    -------
    n
        Normalized normal vector.
    u, v
        Orthonormal basis vectors spanning the plane.
    """
    n = np.asarray(n, float)
    n /= np.linalg.norm(n)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def _clip_segments_rect(
    segments: npt.NDArray[np.floating],
    bounds: tuple[float, float, float, float],
    eps: float = 1e-15,
) -> npt.NDArray[np.floating]:
    """Liang-Barsky algorithm to clip 2D line segments to a rectangle.

    Parameters
    ----------
    segments : (S, 2, 2) array_like
        Each segment is [[x0, y0], [x1, y1]].
    bounds : tuple of float
        (xmin, xmax, ymin, ymax)
    eps : float
        Tolerance for treating a segment as parallel to a clipping edge.

    Returns
    -------
    clipped_segments : (S', 2, 2) ndarray
        Clipped segments that intersect the rectangle.
    """
    segments = np.asarray(segments, dtype=float)
    if segments.size == 0:
        return segments.reshape(0, 2, 2)

    xmin, xmax, ymin, ymax = map(float, bounds)
    bmin = np.array([xmin, ymin], dtype=float)  # (2,)
    bmax = np.array([xmax, ymax], dtype=float)  # (2,)

    p0 = segments[:, 0]  # (S,2)
    p1 = segments[:, 1]  # (S,2)
    d = p1 - p0  # (S,2)

    parallel = np.abs(d) < eps  # (S,2)

    # Reject if parallel to an axis and outside that slab on that axis
    outside_parallel = parallel & ((p0 < bmin) | (p0 > bmax))  # (S,2)
    keep = ~outside_parallel.any(axis=1)  # (S,)

    # For non-parallel axes compute t-interval for intersection with slab
    # Set parallel axes to (-inf, +inf) so they don't constrain the interval.
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (bmin - p0) / d  # (S,2)
        t1 = (bmax - p0) / d  # (S,2)

    tmin_axis = np.minimum(t0, t1)
    tmax_axis = np.maximum(t0, t1)

    tmin_axis = np.where(parallel, -np.inf, tmin_axis)
    tmax_axis = np.where(parallel, +np.inf, tmax_axis)

    t_enter = np.maximum.reduce(tmin_axis, axis=1)
    t_exit = np.minimum.reduce(tmax_axis, axis=1)

    t_enter = np.maximum(t_enter, 0.0)
    t_exit = np.minimum(t_exit, 1.0)

    keep &= t_enter <= t_exit
    if not np.any(keep):
        return np.empty((0, 2, 2), dtype=float)

    p0k = p0[keep]
    dk = d[keep]
    te = t_enter[keep, None]
    tx = t_exit[keep, None]

    clipped = np.empty((p0k.shape[0], 2, 2), dtype=float)
    clipped[:, 0] = p0k + te * dk
    clipped[:, 1] = p0k + tx * dk
    return clipped


def _order_face(face3):
    X = face3 - face3.mean(0)
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    n /= np.linalg.norm(n)

    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    xy = np.empty((face3.shape[0], 2))
    for i in range(face3.shape[0]):
        xy[i, 0] = np.dot(X[i], u)
        xy[i, 1] = np.dot(X[i], v)

    angles = np.arctan2(xy[:, 1], xy[:, 0])
    sorted_indices = np.argsort(angles)
    return face3[sorted_indices]


def _bz_face_edges(bvec):
    """Compute face edges of the first Brillouin zone.

    Parameters
    ----------
    bvec: (3, 3) array-like
        Reciprocal lattice basis vectors.

    Returns
    -------
    edges: (N,2,3) array
        Edges of the BZ faces, ordered around each face.
    fid: (N,) array
        Face index for each edge.

    """
    pts = (
        np.tensordot(
            bvec, np.mgrid[slice(-1, 2), slice(-1, 2), slice(-1, 2)], axes=(0, 0)
        )
        .reshape(3, 27)
        .T
    )
    z = np.where((pts == 0).all(1))[0][0]

    import scipy.spatial

    vor = scipy.spatial.Voronoi(pts)

    faces = []
    for pidx, vids in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
        if z not in pidx:
            continue
        vids = np.asarray(vids, int)
        if np.any(vids < 0):
            continue
        faces.append(_order_face(vor.vertices[vids]))

    edges = np.concatenate(
        [np.stack([face, np.roll(face, -1, 0)], 1) for face in faces], 0
    )
    fid = np.concatenate(
        [np.full(len(face), i, int) for i, face in enumerate(faces)], 0
    )
    return edges, fid


def _dedup_segments_2d(segs2, tol):
    if len(segs2) == 0:
        return segs2
    q = np.rint(segs2 / tol).astype(np.int64)  # (S,2,2)
    a, b = q[:, 0], q[:, 1]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    q[swap] = q[swap][:, ::-1]
    key = q.reshape(len(q), 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    return segs2[np.sort(idx)]


def _deduplicate_and_exclude(points, bounds, tolerance):
    _, idx = np.unique(np.rint(points / tolerance), axis=0, return_index=True)
    deduplicated = points[idx]
    x = deduplicated[:, 0]
    y = deduplicated[:, 1]
    xmin, xmax, ymin, ymax = bounds
    mask = (xmin <= x) & (x <= xmax) & (ymin <= y) & (y <= ymax)
    return deduplicated[mask]


def get_bz_slice(
    bvec: npt.NDArray[np.floating],
    plane_point: npt.NDArray[np.floating],
    plane_normal: npt.NDArray[np.floating],
    plane_bounds: tuple[float, float, float, float],
    *,
    pad_cells: int = 1,
    eps: float = 1e-10,
    return_3d: bool = False,
    return_midpoints: bool = False,
) -> (
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    | tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]
):
    """Get Brillouin zone boundaries along an arbitrary 2D slicing plane.

    Parameters
    ----------
    bvec : (3, 3) array_like
        Reciprocal lattice basis vectors. Columns are the basis used for grid @ bvec.
    plane_point : (3,) array_like
        A point on the slicing plane, in the same coordinate system as `bvec`.
    plane_normal : (3,) array_like
        Normal vector of the slicing plane.
    plane_bounds : tuple of float
        Bounds (xmin, xmax, ymin, ymax) of the rectangular window in the plane's local
        (u, v) coordinates.
    pad_cells : int, optional
        Extra integer padding on the reciprocal lattice translation grid to ensure
        coverage. Default is 1.
    eps : float, optional
        Numerical tolerance used for coplanarity and parallel tests. Default is 1e-10.
    return_3d : bool, optional
        If True, returns all coordinates in 3D momentum space. If False, returns 2D
        coordinates in the plane's basis. Default is False.
    return_midpoints : bool, optional
        If True, also return the midpoints of each segment.

    Returns
    -------
    segments : (S, 2, 2) or (S, 2, 3) numpy array
        2D segments in plane coordinates (u, v) (if ``return_3d`` is False) or 3D
        segments in the original coordinate system (if ``return_3d`` is True).
    vertices : (V, 2) or (V, 3) numpy array
        Unique vertices of the segments, in 2D plane coordinates (u, v) (if
        ``return_3d`` is False) or 3D momentum space coordinates (if ``return_3d`` is
        True).
    midpoints : (S, 2) or (S, 3) numpy array
        Midpoints of each segment, in 2D plane coordinates (u, v) (if ``return_3d`` is
        False) or 3D momentum space coordinates (if ``return_3d`` is True).
    """
    edges, fid = _bz_face_edges(bvec)

    p0 = np.asarray(plane_point, float)
    n, u, v = _plane_frame(plane_normal)
    xmin, xmax, ymin, ymax = plane_bounds
    tol = 10 * eps

    # Compute fractional coords of rect corners to determine number of BZs to cover
    corners2 = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]], float)
    corners3 = p0 + corners2[:, :1] * u + corners2[:, 1:] * v
    frac = np.linalg.solve(bvec.T, corners3.T).T
    lo = np.floor(frac.min(0)).astype(int) - pad_cells
    hi = np.ceil(frac.max(0)).astype(int) + pad_cells
    grid = np.moveaxis(
        np.mgrid[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1], 0, -1
    ).reshape(-1, 3)
    shifts = grid @ bvec  # (K,3)
    segments2d_collected = []

    edge_uv = edges @ np.stack((u, v), axis=1)  # (E,2,2)
    edge_n = edges @ n  # (E,2)
    p0_uvn = np.array([np.dot(p0, u), np.dot(p0, v), np.dot(p0, n)])
    base_uv = edge_uv - p0_uvn[:2]
    base_n = edge_n - p0_uvn[2]
    shift_uvn = shifts @ np.stack((u, v, n), axis=1)

    # Iterate over reciprocal lattice shifts (translated BZ copies)
    for shift2d, shift_n in zip(shift_uvn[:, :2], shift_uvn[:, 2], strict=True):
        dist_start = base_n[:, 0] + shift_n
        dist_end = base_n[:, 1] + shift_n

        # Edge classification relative to plane
        edge_is_coplanar = (np.abs(dist_start) < eps) & (np.abs(dist_end) < eps)
        edge_crosses_plane = (dist_start * dist_end <= 0) & ~edge_is_coplanar

        # Coplanar edges: project full segments and clip
        if np.any(edge_is_coplanar):
            coplanar_edges2d = base_uv[edge_is_coplanar] + shift2d  # (Ec, 2, 2)
            if len(coplanar_edges2d):
                segments2d_collected.append(coplanar_edges2d)

        if not np.any(edge_crosses_plane):
            continue

        # Proper intersections: edge -> point, grouped by face -> segment
        crossing_edges2d = base_uv[edge_crosses_plane] + shift2d  # (P, 2, 2)
        crossing_face_ids = fid[edge_crosses_plane]  # (P,)

        dist_start_cross = dist_start[edge_crosses_plane]
        dist_end_cross = dist_end[edge_crosses_plane]
        dist_delta = dist_start_cross - dist_end_cross

        t_param = dist_start_cross / dist_delta  # (P,)
        intersection_points2d = crossing_edges2d[:, 0] + t_param[:, None] * (
            crossing_edges2d[:, 1] - crossing_edges2d[:, 0]
        )  # (P, 2)

        # Quantize for stable ordering/grouping
        quantized_points2d = np.rint(intersection_points2d / tol).astype(
            np.int64
        )  # (P, 2)

        # Sort by (face_id, qx, qy) so first/last per face become segment endpoints
        sort_order = np.lexsort(
            (quantized_points2d[:, 1], quantized_points2d[:, 0], crossing_face_ids)
        )
        face_ids_sorted = crossing_face_ids[sort_order]
        points2d_sorted = intersection_points2d[sort_order]

        # Find contiguous groups with same face id
        is_new_face_group = np.r_[True, face_ids_sorted[1:] != face_ids_sorted[:-1]]
        group_starts = np.flatnonzero(is_new_face_group)
        group_ends = np.r_[group_starts[1:], len(face_ids_sorted)]

        group_lengths = group_ends - group_starts
        has_at_least_two_points = group_lengths >= 2
        if not np.any(has_at_least_two_points):
            continue

        first_indices = group_starts[has_at_least_two_points]
        last_indices = group_ends[has_at_least_two_points] - 1

        face_segments2d = np.stack(
            [points2d_sorted[first_indices], points2d_sorted[last_indices]],
            axis=1,
        )
        if len(face_segments2d):
            segments2d_collected.append(face_segments2d)

    segments = (
        np.concatenate(segments2d_collected, axis=0)
        if segments2d_collected
        else np.empty((0, 2, 2))
    )

    # Collect unique vertices within bounds
    vertices = _deduplicate_and_exclude(segments.reshape(-1, 2), plane_bounds, tol)

    # Compute midpoints of segments
    if return_midpoints:
        midpoints = _deduplicate_and_exclude(
            np.mean(segments, axis=1), plane_bounds, tol
        )

    # Clip to bounds
    segments = _clip_segments_rect(segments, plane_bounds)

    # Deduplicate across shared faces / neighboring cells
    segments = _dedup_segments_2d(segments, tol=tol)

    if return_3d:
        segments = p0 + segments[..., :1] * u + segments[..., 1:] * v
        vertices = p0 + vertices[:, :1] * u + vertices[:, 1:] * v
        if return_midpoints:
            midpoints = p0 + midpoints[:, :1] * u + midpoints[:, 1:] * v

    if return_midpoints:
        return segments, vertices, midpoints
    return segments, vertices


def get_out_of_plane_bz(
    bvec: npt.NDArray[np.floating],
    k_parallel: float,
    angle: float,
    bounds: tuple[float, float, float, float],
    **kwargs,
):
    """Get the Brillouin zone boundaries along kz.

    Given a line along in-plane momentum defined by ``k_parallel`` and ``angle``,
    computes the BZ boundaries along the out-of-plane momentum.

    Parameters
    ----------
    bvec : (3, 3) array_like
        Reciprocal lattice basis vectors.
    k_parallel : float
        In-plane momentum magnitude.
    angle : float
        Angle (in degrees) of the in-plane momentum direction, measured from the
        positive kx axis toward the positive ky axis.
    bounds : tuple of float
        (kp_min, kp_max, kz_min, kz_max) bounds, where ``kp`` is the in-plane momentum
        in the direction perpendicular to the one defined by ``angle``.
    kwargs
        Additional keyword arguments passed to :func:`get_bz_slice`.
    """
    theta = np.deg2rad(angle)
    kx, ky = k_parallel * np.cos(theta), k_parallel * np.sin(theta)

    return get_bz_slice(
        bvec,
        plane_point=np.array([kx, ky, 0.0]),
        plane_normal=np.array([np.cos(theta), np.sin(theta), 0.0]),
        plane_bounds=bounds,
        **kwargs,
    )


def get_in_plane_bz(
    bvec,
    kz: float,
    angle: float,
    bounds: tuple[float, float, float, float],
    **kwargs,
):
    """Get the Brillouin zone boundary sliced by a constant kz plane.

    Given a constant out-of-plane momentum ``kz``, computes the BZ boundary of in-plane
    momenta, rotated by ``angle``.

    Parameters
    ----------
    bvec : (3, 3) array_like
        Reciprocal lattice basis vectors.
    kz : float
        Out-of-plane momentum.
    angle : float
        Rotation angle (in degrees) of the Brillouin zone about the kz axis.
    bounds : tuple of float
        (kx_min, kx_max, ky_min, ky_max) bounds of the in-plane momenta.
    kwargs
        Additional keyword arguments passed to :func:`get_bz_slice`.
    """
    theta = np.deg2rad(angle)
    bvec_rot = bvec @ np.array(
        [
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return get_bz_slice(
        bvec_rot,
        plane_point=np.array([0.0, 0.0, kz]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
        plane_bounds=bounds,
        **kwargs,
    )
