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
    "to_real",
    "to_reciprocal",
]

import itertools

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
