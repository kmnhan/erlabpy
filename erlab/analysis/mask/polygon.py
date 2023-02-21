"""Polygon mask generation code adapted from the CGAL C++ library."""

import numba
from . import comparison
import numpy as np

ON_UNBOUNDED_SIDE = -1
ON_BOUNDARY = 0
ON_BOUNDED_SIDE = 1


@numba.njit(nogil=True, cache=True)
def get_argmin_all(arr):
    return np.nonzero(arr == arr.min())[0]


@numba.njit(nogil=True, cache=True)
def get_argmax_all(arr):
    return np.nonzero(arr == arr.max())[0]


@numba.njit(nogil=True, cache=True)
def left_vertex(points):
    """left_vertex returns the index of the leftmost point of a polygon.

    In case of a tie, the point with the smallest y-coordinate is taken.

    Parameters
    ----------
    points : (M, 2) array_like
        Input array of polygon vertices.

    Returns
    -------
    int

    """
    ind = get_argmin_all(points[:, 0].astype(np.float32))
    return ind[np.argmin(points[ind][:, 1])]


@numba.njit(nogil=True, cache=True)
def right_vertex(points):
    """right_vertex returns the index of the rightmost point of a polygon.

    In case of a tie, the point with the largest y-coordinate is taken.

    Parameters
    ----------
    points : (M, 2) array_like
        Input array of polygon vertices.

    Returns
    -------
    int

    """
    ind = get_argmax_all(points[:, 0].astype(np.float32))
    return ind[np.argmax(points[ind][:, 1])]


@numba.njit(nogil=True, cache=True)
def polygon_orientation(points):
    ind = left_vertex(points)
    return orientation(points[ind - 1], points[ind], points[ind + 1])


@numba.njit(nogil=True, cache=True)
def orientation(p0, p1, p2):
    p10, p20 = p1 - p0, p2 - p0
    return np.sign(p10[0] * p20[1] - p20[0] * p10[1])


@numba.njit(nogil=True, cache=True)
def which_side_in_slab(point, low, high, points):
    point = np.asarray(point)
    low_x_comp_res = comparison.comp_x(point, points[low])
    high_x_comp_res = comparison.comp_x(point, points[high])

    if low_x_comp_res == comparison.SMALLER:
        if high_x_comp_res == comparison.SMALLER:
            return -1
    else:
        match high_x_comp_res:
            case comparison.LARGER:
                return 1
            case comparison.SMALLER:
                pass
            case comparison.EQUAL:
                return int(low_x_comp_res != comparison.EQUAL)

    return orientation(points[low], point, points[high])


@numba.njit(nogil=True, cache=True)
def bounded_side(points, point):
    """Computes if a point lies inside a polygon.

    The polygon is defined by the sequence of points [first,last). Being inside is
    defined by the odd-even rule. If the point is on a polygon edge, a special value is
    returned. A simple polygon divides the plane in an unbounded and a bounded region.
    According to the definition points in the bounded region are inside the polygon.

    Parameters
    ----------
    points : (M, 2) array_like
        Input array of polygon vertices.
    point : (1, 2) array_like
        Point of interest

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
        return ON_UNBOUNDED_SIDE

    is_inside = False

    cur_y_comp_res = comparison.comp_y(points[0], point)
    for i in range(len(points) - 1):
        next_y_comp_res = comparison.comp_y(points[i + 1], point)

        match cur_y_comp_res:
            case comparison.SMALLER:
                match next_y_comp_res:
                    case comparison.SMALLER:
                        pass

                    case comparison.EQUAL:
                        match comparison.comp_x(point, points[i + 1]):
                            case comparison.SMALLER:
                                is_inside = not is_inside
                            case comparison.EQUAL:
                                return ON_BOUNDARY
                            case comparison.LARGER:
                                pass

                    case comparison.LARGER:
                        match which_side_in_slab(point, i, i + 1, points):
                            case -1:
                                is_inside = not is_inside
                            case 0:
                                return ON_BOUNDARY

            case comparison.EQUAL:
                match next_y_comp_res:
                    case comparison.SMALLER:
                        match comparison.comp_x(point, points[i]):
                            case comparison.SMALLER:
                                is_inside = not is_inside
                            case comparison.EQUAL:
                                return ON_BOUNDARY
                            case comparison.LARGER:
                                pass

                    case comparison.EQUAL:
                        match comparison.comp_x(point, points[i]):
                            case comparison.SMALLER:
                                if (
                                    comparison.comp_x(point, points[i + 1])
                                    != comparison.SMALLER
                                ):
                                    return ON_BOUNDARY
                            case comparison.EQUAL:
                                return ON_BOUNDARY
                            case comparison.LARGER:
                                if (
                                    comparison.comp_x(point, points[i + 1])
                                    != comparison.LARGER
                                ):
                                    return ON_BOUNDARY

                    case comparison.LARGER:
                        if comparison.comp_x(point, points[i]) == comparison.EQUAL:
                            return ON_BOUNDARY

            case comparison.LARGER:
                match next_y_comp_res:
                    case comparison.SMALLER:
                        match which_side_in_slab(point, i + 1, i, points):
                            case -1:
                                is_inside = not is_inside
                            case 0:
                                return ON_BOUNDARY
                    case comparison.EQUAL:
                        if comparison.comp_x(point, points[i + 1]) == comparison.EQUAL:
                            return ON_BOUNDARY
                    case comparison.LARGER:
                        pass

        cur_y_comp_res = next_y_comp_res

    if is_inside:
        return ON_BOUNDED_SIDE
    else:
        return ON_UNBOUNDED_SIDE
