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
    "get_surface_bz",
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


def _surface_bz_translation_grid(
    bvec: npt.NDArray[np.floating],
    surface: npt.NDArray[np.floating],
    *,
    pad_cells: int,
) -> npt.NDArray[np.floating]:
    """Return the reciprocal translations needed by :func:`get_surface_bz`.

    This helper is useful when the displayed momentum slice is sampled on a curved
    surface, such as the exact ``hv`` preview used by
    :func:`erlab.interactive.ktool`. It expands only the repeated reciprocal cells
    needed to cover the visible sampled surface, which keeps later distance evaluation
    bounded to the cells that can actually contribute visible BZ boundaries.

    Parameters
    ----------
    bvec
        Reciprocal lattice basis vectors.
    surface
        Sampled momentum surface with shape ``(ny, nx, 3)``.
    pad_cells
        Number of extra integer cell translations to include beyond the surface-derived
        bounds.

    Returns
    -------
    ndarray
        Reciprocal-lattice translation vectors with shape ``(nshift, 3)``.
    """
    frac = np.linalg.solve(bvec.T, np.reshape(surface, (-1, 3)).T).T
    lo = np.floor(np.nanmin(frac, axis=0)).astype(int) - pad_cells
    hi = np.ceil(np.nanmax(frac, axis=0)).astype(int) + pad_cells
    grid = np.stack(
        np.meshgrid(
            np.arange(lo[0], hi[0] + 1, dtype=int),
            np.arange(lo[1], hi[1] + 1, dtype=int),
            np.arange(lo[2], hi[2] + 1, dtype=int),
            indexing="ij",
        ),
        axis=-1,
    )
    return grid.reshape(-1, 3) @ bvec


def _surface_bz_active_pair_bounds(
    owner: npt.NDArray[np.integer],
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Return active owner pairs and local contour bounds for :func:`get_surface_bz`.

    This helper is useful in the exact surface-BZ path because it replaces an expensive
    full-grid rescan for every active pair. It records where neighboring owner labels
    actually change on the display grid, then compresses those crossings into a compact
    row/column bounding box that `get_surface_bz` can contour locally.

    Parameters
    ----------
    owner
        Integer owner-label grid returned by ``argmin`` over reciprocal-cell distances.

    Returns
    -------
    dict
        Mapping from sorted owner-pair labels to inclusive
        ``(row_min, row_max, col_min, col_max)`` bounds on the display grid.
    """
    pair_bounds: dict[tuple[int, int], list[int]] = {}

    def _update_pair_bounds(
        first_grid: npt.NDArray[np.integer],
        second_grid: npt.NDArray[np.integer],
        *,
        row_delta: int,
        col_delta: int,
    ) -> None:
        rows, cols = np.nonzero(first_grid != second_grid)
        for row, col, first, second in zip(
            rows,
            cols,
            first_grid[rows, cols],
            second_grid[rows, cols],
            strict=False,
        ):
            row = int(row)
            col = int(col)
            first_i = int(first)
            second_i = int(second)
            pair: tuple[int, int] = (
                (first_i, second_i) if first_i <= second_i else (second_i, first_i)
            )
            bounds = pair_bounds.setdefault(
                pair, [row, row + row_delta, col, col + col_delta]
            )
            bounds[0] = min(bounds[0], row)
            bounds[1] = max(bounds[1], row + row_delta)
            bounds[2] = min(bounds[2], col)
            bounds[3] = max(bounds[3], col + col_delta)

    _update_pair_bounds(owner[:, :-1], owner[:, 1:], row_delta=0, col_delta=1)
    _update_pair_bounds(owner[:-1, :], owner[1:, :], row_delta=1, col_delta=0)

    return {
        pair: (bounds[0], bounds[1], bounds[2], bounds[3])
        for pair, bounds in pair_bounds.items()
    }


def _polyline_segment_signature(
    line: npt.NDArray[np.floating], tol: float
) -> bytes | None:
    """Return a stable segment signature for contour deduplication.

    This helper is useful after `contourpy` extraction in :func:`get_surface_bz`,
    where the same BZ boundary can be emitted more than once from neighboring active
    cell pairs. It converts a polyline into a direction-independent byte signature so
    duplicate segments can be removed cheaply without solving a more expensive geometric
    matching problem.

    Parameters
    ----------
    line
        Polyline vertices in display coordinates.
    tol
        Quantization tolerance used to normalize nearly identical segments.

    Returns
    -------
    bytes or None
        Direction-independent signature for the polyline, or `None` for degenerate
        single-point lines.
    """
    if line.shape[0] < 2:
        return None

    segs = _dedup_segments_2d(np.stack([line[:-1], line[1:]], axis=1), tol=tol)
    q = np.rint(segs / tol).astype(np.int64)
    a, b = q[:, 0], q[:, 1]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    q[swap] = q[swap][:, ::-1]
    key = q.reshape(len(q), 4)
    key = key[np.lexsort((key[:, 3], key[:, 2], key[:, 1], key[:, 0]))]
    return key.tobytes()


def _endpoint_tangent(
    line: npt.NDArray[np.floating], endpoint_idx: int
) -> npt.NDArray[np.floating] | None:
    """Return the inward tangent used by the cheap endpoint snap heuristic.

    This helper is useful in :func:`_snap_polyline_endpoints`, where nearby endpoints
    must be classified as true segment continuations or unrelated crossings without
    running an expensive global reconstruction pass. The normalized inward tangent gives
    a fast local direction test for that decision.

    Parameters
    ----------
    line
        Polyline vertices in display coordinates.
    endpoint_idx
        Endpoint index, typically ``0`` or ``-1``.

    Returns
    -------
    ndarray or None
        Unit tangent vector pointing inward from the endpoint, or `None` for degenerate
        segments.
    """
    if line.shape[0] < 2:
        return None
    tangent = line[1] - line[0] if endpoint_idx == 0 else line[-2] - line[-1]
    norm = float(np.linalg.norm(tangent))
    if norm <= 0:
        return None
    return tangent / norm


def _pair_labels_form_cycle(pair_labels: set[tuple[int, int]]) -> bool:
    """Return whether close owner-pair labels form a valid local BZ junction.

    This helper is useful in :func:`_snap_polyline_endpoints`, where a cluster of close
    contour endpoints should only be merged when its owner-pair graph matches a small
    Voronoi-style cycle. That keeps exact ``hv`` overlays visually connected while
    avoiding accidental merges of unrelated nearby vertices.

    Parameters
    ----------
    pair_labels
        Sorted owner-pair labels attached to candidate contour endpoints.

    Returns
    -------
    bool
        `True` when every participating owner has degree two and the labels form a
        simple cycle.
    """
    if len(pair_labels) < 3:
        return False

    degree: dict[int, int] = {}
    for first, second in pair_labels:
        degree[first] = degree.get(first, 0) + 1
        degree[second] = degree.get(second, 0) + 1
    return len(degree) == len(pair_labels) and all(val == 2 for val in degree.values())


def _snap_polyline_endpoints(
    lines: list[npt.NDArray[np.floating]],
    line_pairs: list[tuple[int, int]],
    snap_tol: float,
    sig_tol: float,
) -> list[npt.NDArray[np.floating]]:
    """Snap contour endpoints using a fast local topology check.

    This helper is useful for exact surface overlays returned by
    :func:`get_surface_bz`, especially in the interactive ``hv`` path used by
    :func:`erlab.interactive.ktool`. `contourpy` can leave nearly coincident segment
    endpoints that should meet at one displayed vertex, but a full geometric stitcher is
    too expensive for live updates. This routine keeps the cost low by only merging
    mutual same-pair continuations and small owner-cycle junctions.

    Parameters
    ----------
    lines
        Contour polylines in display coordinates.
    line_pairs
        Sorted owner-pair labels corresponding to ``lines``.
    snap_tol
        Maximum endpoint separation allowed for snapping.
    sig_tol
        Quantization tolerance used when deduplicating snapped contours.

    Returns
    -------
    list of ndarray
        Snapped and deduplicated contour polylines.
    """
    if len(lines) < 2 or snap_tol <= 0:
        return lines

    refs = [
        (line_idx, endpoint_idx)
        for line_idx in range(len(lines))
        for endpoint_idx in (0, -1)
    ]
    endpoints = np.asarray([lines[i][j] for i, j in refs], dtype=float)
    endpoint_pairs = [line_pairs[line_idx] for line_idx, _ in refs]
    tangents = [_endpoint_tangent(lines[i], j) for i, j in refs]
    line_ids = [line_idx for line_idx, _ in refs]

    dist_sq = np.sum((endpoints[:, None, :] - endpoints[None, :, :]) ** 2, axis=-1)
    pair_groups: dict[tuple[int, int], list[int]] = {}
    for endpoint_idx, pair in enumerate(endpoint_pairs):
        pair_groups.setdefault(pair, []).append(endpoint_idx)

    snapped_lines = [line.copy() for line in lines]
    used: set[int] = set()
    same_pair_matches: list[tuple[float, int, int]] = []

    for indices in pair_groups.values():
        if len(indices) < 2:
            continue

        nearest: dict[int, int] = {}
        for idx in indices:
            candidates = [
                other
                for other in indices
                if other != idx and line_ids[other] != line_ids[idx]
            ]
            if not candidates:
                continue
            nearest_idx = min(candidates, key=lambda other: float(dist_sq[idx, other]))
            nearest[idx] = nearest_idx

        for idx, nearest_idx in nearest.items():
            if nearest.get(nearest_idx) != idx:
                continue
            if dist_sq[idx, nearest_idx] > snap_tol**2:
                continue
            tangent_i = tangents[idx]
            tangent_j = tangents[nearest_idx]
            if tangent_i is None or tangent_j is None:
                continue
            if float(np.dot(tangent_i, tangent_j)) > -0.95:
                continue
            same_pair_matches.append(
                (float(dist_sq[idx, nearest_idx]), idx, nearest_idx)
            )

    for _, idx, nearest_idx in sorted(same_pair_matches):
        if idx in used or nearest_idx in used:
            continue
        used.update((idx, nearest_idx))
        snapped = np.mean(endpoints[[idx, nearest_idx]], axis=0)
        for member in (idx, nearest_idx):
            line_idx, endpoint_idx = refs[member]
            snapped_lines[line_idx][endpoint_idx] = snapped

    neighbors: dict[int, set[int]] = {
        idx: set() for idx in range(len(refs)) if idx not in used
    }
    for i, j in zip(*np.triu_indices(len(refs), k=1), strict=True):
        if i in used or j in used:
            continue
        if endpoint_pairs[i] == endpoint_pairs[j]:
            continue
        if dist_sq[i, j] > snap_tol**2:
            continue
        if len(set(endpoint_pairs[i]) & set(endpoint_pairs[j])) != 1:
            continue
        tangent_i = tangents[i]
        tangent_j = tangents[j]
        if (
            tangent_i is not None
            and tangent_j is not None
            and abs(float(np.dot(tangent_i, tangent_j))) > 0.95
        ):
            continue
        neighbors[i].add(int(j))
        neighbors[j].add(int(i))

    seen: set[int] = set()
    components: list[list[int]] = []
    for idx in sorted(neighbors):
        if idx in seen:
            continue
        stack = [idx]
        component: list[int] = []
        seen.add(idx)
        while stack:
            current = stack.pop()
            component.append(current)
            for other in neighbors[current]:
                if other in seen:
                    continue
                seen.add(other)
                stack.append(other)
        components.append(sorted(component))

    cycle_matches: list[tuple[float, tuple[int, ...]]] = []
    for component in components:
        for size in (3, 4):
            if len(component) < size:
                continue
            for members in itertools.combinations(component, size):
                if len({line_ids[member] for member in members}) != size:
                    continue
                if any(
                    dist_sq[i, j] > snap_tol**2
                    for i, j in itertools.combinations(members, 2)
                ):
                    continue
                pair_labels = {endpoint_pairs[member] for member in members}
                if len(pair_labels) != size or not _pair_labels_form_cycle(pair_labels):
                    continue
                center = np.mean(endpoints[list(members)], axis=0)
                max_line_distance = 0.25 * snap_tol
                valid_center = True
                for member in members:
                    tangent = tangents[member]
                    if tangent is None:
                        valid_center = False
                        break
                    offset = endpoints[member] - center
                    line_distance = abs(
                        float(offset[0] * tangent[1] - offset[1] * tangent[0])
                    )
                    if line_distance > max_line_distance:
                        valid_center = False
                        break
                if not valid_center:
                    continue
                cycle_matches.append(
                    (
                        float(
                            max(
                                dist_sq[i, j]
                                for i, j in itertools.combinations(members, 2)
                            )
                        ),
                        members,
                    )
                )

    for _, members in sorted(cycle_matches):
        if any(member in used for member in members):
            continue
        used.update(members)
        snapped = np.mean(endpoints[list(members)], axis=0)
        for member in members:
            line_idx, endpoint_idx = refs[member]
            snapped_lines[line_idx][endpoint_idx] = snapped

    deduped: list[npt.NDArray[np.floating]] = []
    seen_signatures: set[bytes] = set()
    for line in snapped_lines:
        if line.shape[0] < 2:
            continue
        signature = _polyline_segment_signature(line, sig_tol)
        if signature is None or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(line)

    return deduped


def get_surface_bz(
    bvec: npt.NDArray[np.floating],
    plot_x: npt.ArrayLike,
    plot_y: npt.ArrayLike,
    surface: npt.NDArray[np.floating],
    *,
    pad_cells: int = 1,
    eps: float = 1e-10,
    return_midpoints: bool = False,
) -> (
    tuple[list[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
    | tuple[
        list[npt.NDArray[np.floating]],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]
):
    """Get Brillouin zone boundaries on an arbitrary sampled 2D momentum surface.

    This function evaluates the repeated-zone Brillouin-zone boundaries on a regular
    2D display grid whose points are embedded in 3D momentum space. Unlike
    :func:`get_bz_slice`, the sampled surface does not need to be planar; each point in
    ``surface`` may have its own ``(kx, ky, kz)`` coordinates.

    It is primarily useful for exact overlays on displays whose carried momentum
    coordinates are no longer scalar, such as ``hv``-dependent previews in
    :func:`erlab.interactive.ktool`. Those previews cannot be described by a single
    constant-``k_parallel`` plane, so the BZ boundary must be evaluated on the sampled
    surface itself.

    The algorithm computes the nearest reciprocal-lattice point for every sampled
    surface point, detects only the neighboring reciprocal cells that are actually
    adjacent on the displayed grid, and contours the exact equality
    :math:`d_i^2 - d_j^2 = 0` for those active pairs. A short endpoint-snapping pass is
    applied afterward so neighboring contour segments meet cleanly on the display grid.

    Parameters
    ----------
    bvec
        Reciprocal lattice basis vectors.
    plot_x, plot_y
        One-dimensional display coordinates for the surface grid. The returned polylines
        are expressed in this coordinate system.
    surface
        Sampled momentum surface with shape ``(len(plot_y), len(plot_x), 3)``. The last
        axis contains the 3D momentum coordinates corresponding to each display-grid
        point.
    pad_cells
        Extra integer padding on the reciprocal-lattice translation grid used to ensure
        coverage near the displayed surface boundary.
    eps
        Numerical tolerance used when deduplicating contour segments and derived marker
        points.
    return_midpoints
        If `True`, an empty midpoint array is returned for API compatibility. Unlike
        planar BZ slices, curved surface intersections do not have a well-defined
        reciprocal-space midpoint between symmetry points.

    Returns
    -------
    lines
        A list of polyline arrays in ``(plot_x, plot_y)`` coordinates.
    vertices
        Deduplicated contour endpoints that lie within the displayed bounds. Closed
        loops contribute no endpoints.
    midpoints
        Empty ``(0, 2)`` array returned only if ``return_midpoints`` is `True`.

    Notes
    -----
    For planar slices, :func:`get_bz_slice`, :func:`get_in_plane_bz`, or
    :func:`get_out_of_plane_bz` are usually cheaper and return analytically cleaner
    segment geometry. This function is intended for cases where the displayed
    coordinates correspond to a curved or otherwise non-planar sampled surface.
    """
    plot_x = np.asarray(plot_x, dtype=float)
    plot_y = np.asarray(plot_y, dtype=float)
    surface = np.asarray(surface, dtype=float)

    if plot_x.ndim != 1 or plot_y.ndim != 1:
        raise ValueError("`plot_x` and `plot_y` must be one-dimensional arrays.")
    if surface.shape != (plot_y.size, plot_x.size, 3):
        raise ValueError("`surface` must have shape (len(plot_y), len(plot_x), 3).")

    import contourpy

    tol = 10 * eps
    step_x = float(np.nanmedian(np.abs(np.diff(plot_x)))) if plot_x.size > 1 else 0.0
    step_y = float(np.nanmedian(np.abs(np.diff(plot_y)))) if plot_y.size > 1 else 0.0
    snap_tol = float(np.hypot(step_x, step_y))
    shifts = _surface_bz_translation_grid(bvec, surface, pad_cells=pad_cells)
    surface_sq = np.sum(surface * surface, axis=-1)
    shift_sq = np.sum(shifts * shifts, axis=1)[:, None, None]
    dist_sq = shift_sq - 2.0 * np.tensordot(shifts, surface, axes=([1], [2]))
    dist_sq += surface_sq[None, ...]
    owner = np.argmin(dist_sq, axis=0)

    lines: list[npt.NDArray[np.floating]] = []
    line_pairs: list[tuple[int, int]] = []
    seen_signatures: set[bytes] = set()

    for (i, j), (row_min, row_max, col_min, col_max) in _surface_bz_active_pair_bounds(
        owner
    ).items():
        r0 = max(row_min - 1, 0)
        r1 = min(row_max + 2, owner.shape[0])
        c0 = max(col_min - 1, 0)
        c1 = min(col_max + 2, owner.shape[1])

        local_owner = owner[r0:r1, c0:c1]
        local_mask = (local_owner == i) | (local_owner == j)
        delta = dist_sq[i, r0:r1, c0:c1] - dist_sq[j, r0:r1, c0:c1]
        contours = contourpy.contour_generator(
            x=plot_x[c0:c1],
            y=plot_y[r0:r1],
            z=np.ma.array(delta, mask=~local_mask),
        ).lines(0.0)

        for contour_line in contours:
            arr = np.asarray(contour_line, dtype=float)
            if arr.shape[0] < 2:
                continue
            signature = _polyline_segment_signature(arr, tol)
            if signature is None or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            lines.append(arr)
            line_pairs.append((i, j))

    lines = _snap_polyline_endpoints(lines, line_pairs, snap_tol, tol)

    bounds = (
        float(plot_x.min()),
        float(plot_x.max()),
        float(plot_y.min()),
        float(plot_y.max()),
    )
    endpoint_groups: dict[
        tuple[int, ...],
        list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]],
    ] = {}
    for polyline in lines:
        if np.allclose(polyline[0], polyline[-1], atol=tol, rtol=0.0):
            continue
        for endpoint_idx in (0, -1):
            point = polyline[endpoint_idx]
            key = tuple(np.rint(point / tol).astype(np.int64))
            endpoint_groups.setdefault(key, []).append(
                (point, _endpoint_tangent(polyline, endpoint_idx))
            )

    vertex_keys: set[tuple[int, ...]] = set()
    vertex_values: list[npt.NDArray[np.floating]] = []
    for key, group in endpoint_groups.items():
        if len(group) < 2:
            continue
        if len(group) == 2:
            tangent0 = group[0][1]
            tangent1 = group[1][1]
            if (
                tangent0 is not None
                and tangent1 is not None
                and float(np.dot(tangent0, tangent1)) < -0.95
            ):
                continue
        vertex_keys.add(key)
        vertex_values.append(np.mean([point for point, _ in group], axis=0))

    vertices = (
        _deduplicate_and_exclude(np.asarray(vertex_values, dtype=float), bounds, tol)
        if vertex_values
        else np.empty((0, 2), dtype=float)
    )

    if return_midpoints:
        return lines, vertices, np.empty((0, 2), dtype=float)

    return lines, vertices


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
