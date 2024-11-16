"""
Tools related to the real and reciprocal lattice.

.. currentmodule:: erlab.lattice

"""

__all__ = ["abc2avec", "angle_between", "avec2abc", "to_real", "to_reciprocal"]


import numpy as np
import numpy.typing as npt


def angle_between(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]) -> float:
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
) -> npt.NDArray:
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
    avec: npt.NDArray[np.float64],
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


def to_reciprocal(avec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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


def to_real(bvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
