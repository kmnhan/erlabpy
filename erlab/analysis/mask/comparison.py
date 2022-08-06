import numpy as np
import numba

SMALLER = -1
EQUAL = 0
LARGER = 1


@numba.njit
def comp_x(p0, p1):
    if np.float32(p0[0]) < np.float32(p1[0]):
        return SMALLER
    elif np.float32(p0[0]) == np.float32(p1[0]):
        return EQUAL
    else:
        return LARGER


@numba.njit
def comp_y(p0, p1):
    if np.float32(p0[1]) < np.float32(p1[1]):
        return SMALLER
    elif np.float32(p0[1]) == np.float32(p1[1]):
        return EQUAL
    else:
        return LARGER
