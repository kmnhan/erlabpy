"""Point-in-polygon algorithm.

The implementation has been adapted from the `CGAL C++ library
<https://doc.cgal.org/5.3.2/Polygon/index.html>`_.

"""

from __future__ import annotations

import enum
from typing import Annotated, Literal

import numba
import numpy as np
import numpy.typing as npt


class Comparison(enum.Enum):
    SMALLER = -1
    EQUAL = 0
    LARGER = 1


class Side(enum.Enum):
    ON_UNBOUNDED_SIDE = -1
    ON_BOUNDARY = 0
    ON_BOUNDED_SIDE = 1


@numba.njit(nogil=True, cache=True)
def _comp_x(p0, p1):
    if np.float32(p0[0]) < np.float32(p1[0]):
        return Comparison.SMALLER
    elif np.float32(p0[0]) == np.float32(p1[0]):
        return Comparison.EQUAL
    else:
        return Comparison.LARGER


@numba.njit(nogil=True, cache=True)
def _comp_y(p0, p1):
    if np.float32(p0[1]) < np.float32(p1[1]):
        return Comparison.SMALLER
    elif np.float32(p0[1]) == np.float32(p1[1]):
        return Comparison.EQUAL
    else:
        return Comparison.LARGER


@numba.njit(nogil=True, cache=True)
def _get_argmin_all(arr):
    return np.nonzero(arr == arr.min())[0]


@numba.njit(nogil=True, cache=True)
def _get_argmax_all(arr):
    return np.nonzero(arr == arr.max())[0]


@numba.njit(nogil=True, cache=True)
def left_vertex(points: Annotated[npt.NDArray[np.float64], Literal[1, 2]]) -> int:
    """Return the index of the leftmost point of a polygon.

    In case of a tie, the point with the smallest y-coordinate is taken.

    Parameters
    ----------
    points
        Input array of polygon vertices.

    Returns
    -------
    index : int

    """
    ind = _get_argmin_all(points[:, 0].astype(np.float32))
    return ind[np.argmin(points[ind][:, 1])]


@numba.njit(nogil=True, cache=True)
def right_vertex(points: Annotated[npt.NDArray[np.float64], Literal[1, 2]]):
    """Return the index of the rightmost point of a polygon.

    In case of a tie, the point with the largest y-coordinate is taken.

    Parameters
    ----------
    points
        Input array of polygon vertices.

    Returns
    -------
    int

    """
    ind = _get_argmax_all(points[:, 0].astype(np.float32))
    return ind[np.argmax(points[ind][:, 1])]


@numba.njit(nogil=True, cache=True)
def polygon_orientation(points):
    ind = left_vertex(points)
    return _orientation(points[ind - 1], points[ind], points[ind + 1])


@numba.njit(nogil=True, cache=True)
def _orientation(p0, p1, p2):
    # 1: left turn
    # -1: right turn
    # 0: colinear
    p10, p20 = p1 - p0, p2 - p0
    return np.sign(p10[0] * p20[1] - p20[0] * p10[1])


@numba.njit(nogil=True, cache=True)
def which_side_in_slab(point, low, high, points):
    point = np.asarray(point)
    low_x_comp_res = _comp_x(point, points[low])
    high_x_comp_res = _comp_x(point, points[high])

    if low_x_comp_res == Comparison.SMALLER:
        if high_x_comp_res == Comparison.SMALLER:
            return -1
    else:
        match high_x_comp_res:
            case Comparison.LARGER:
                return 1
            case Comparison.SMALLER:
                pass
            case Comparison.EQUAL:
                return int(low_x_comp_res != Comparison.EQUAL)

    return _orientation(points[low], point, points[high])


@numba.njit(nogil=True, cache=True)
def bounded_side_bool(
    points: npt.NDArray[np.float64], point: tuple[float, float], boundary: bool = True
) -> bool:
    """Compute whether a point lies inside a polygon using `bounded_side`.

    Parameters
    ----------
    points
        (N, 2) input array of polygon vertices.
    point
        2-tuple of float specifying point of interest.
    boundary
        Whether to consider points on the boundary to be inside the polygon. Default is
        `True`.

    Returns
    -------
    bool
        `True` if the point is on the bounded side of the polygon, `False` otherwise.
    """
    match bounded_side(points, point):
        case Side.ON_UNBOUNDED_SIDE:
            return False
        case Side.ON_BOUNDED_SIDE:
            return True
        case Side.ON_BOUNDARY:
            return boundary
        case _:
            return False


@numba.njit(nogil=True, cache=True)
def bounded_side(points: npt.NDArray[np.float64], point: tuple[float, float]) -> Side:
    """Compute if a point is inside, outside, or on the boundary of a polygon.

    The polygon is defined by the sequence of points [first,last). Being inside is
    defined by the odd-even rule. If the point is on a polygon edge, a special value is
    returned. A simple polygon divides the plane in an unbounded and a bounded region.
    According to the definition points in the bounded region are inside the polygon.

    Parameters
    ----------
    points
        (N, 2) input array of polygon vertices.
    point
        2-tuple of float specifying point of interest.

    Returns
    -------
    Side
        Enum indicating the location of the point.

    Note
    ----
    We shoot a horizontal ray from the point to the right and count the number of
    intersections with polygon segments. If the number of intersections is odd, the
    point is inside. We don't count intersections with horizontal segments. With
    non-horizontal segments, the top vertex is considered to be part of the segment, but
    the bottom vertex is not. (Segments are half-closed).

    """
    last = len(points) - 1
    if last < 2:
        return Side.ON_UNBOUNDED_SIDE

    is_inside = False

    cur_y_comp_res = _comp_y(points[0], point)
    for i in range(len(points) - 1):
        next_y_comp_res = _comp_y(points[i + 1], point)

        match cur_y_comp_res:
            case Comparison.SMALLER:
                match next_y_comp_res:
                    case Comparison.SMALLER:
                        pass

                    case Comparison.EQUAL:
                        match _comp_x(point, points[i + 1]):
                            case Comparison.SMALLER:
                                is_inside = not is_inside
                            case Comparison.EQUAL:
                                return Side.ON_BOUNDARY
                            case Comparison.LARGER:
                                pass

                    case Comparison.LARGER:
                        match which_side_in_slab(point, i, i + 1, points):
                            case -1:
                                is_inside = not is_inside
                            case 0:
                                return Side.ON_BOUNDARY

            case Comparison.EQUAL:
                match next_y_comp_res:
                    case Comparison.SMALLER:
                        match _comp_x(point, points[i]):
                            case Comparison.SMALLER:
                                is_inside = not is_inside
                            case Comparison.EQUAL:
                                return Side.ON_BOUNDARY
                            case Comparison.LARGER:
                                pass

                    case Comparison.EQUAL:
                        match _comp_x(point, points[i]):
                            case Comparison.SMALLER:
                                if _comp_x(point, points[i + 1]) != Comparison.SMALLER:
                                    return Side.ON_BOUNDARY
                            case Comparison.EQUAL:
                                return Side.ON_BOUNDARY
                            case Comparison.LARGER:
                                if _comp_x(point, points[i + 1]) != Comparison.LARGER:
                                    return Side.ON_BOUNDARY

                    case Comparison.LARGER:
                        if _comp_x(point, points[i]) == Comparison.EQUAL:
                            return Side.ON_BOUNDARY

            case Comparison.LARGER:
                match next_y_comp_res:
                    case Comparison.SMALLER:
                        match which_side_in_slab(point, i + 1, i, points):
                            case -1:
                                is_inside = not is_inside
                            case 0:
                                return Side.ON_BOUNDARY
                    case Comparison.EQUAL:
                        if _comp_x(point, points[i + 1]) == Comparison.EQUAL:
                            return Side.ON_BOUNDARY
                    case Comparison.LARGER:
                        pass

        cur_y_comp_res = next_y_comp_res

    if is_inside:
        return Side.ON_BOUNDED_SIDE
    else:
        return Side.ON_UNBOUNDED_SIDE
