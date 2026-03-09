"""Helpers for fast slicing of 2D, 3D, and 4D arrays."""

import numba
import numpy as np
import numpy.typing as npt


def _array_rect(
    i: int,
    j: int,
    lims: tuple[tuple[np.floating, np.floating], ...],
    incs: tuple[np.floating, ...],
) -> tuple[np.floating, np.floating, np.floating, np.floating]:
    lim_i0, lim_i1 = lims[i]
    lim_j0, lim_j1 = lims[j]
    inc_i = incs[i]
    inc_j = incs[j]
    return (
        lim_i0 - 0.5 * inc_i,
        lim_j0 - 0.5 * inc_j,
        lim_i1 - lim_i0 + inc_i,
        lim_j1 - lim_j0 + inc_j,
    )


@numba.njit(
    [
        numba.int64(numba.int32[::1], numba.int32),
        numba.int64(numba.int64[::1], numba.int64),
        numba.int64(numba.float32[::1], numba.float32),
        numba.int64(numba.float64[::1], numba.float64),
    ],
    cache=True,
)
def _index_of_value_nonuniform(arr: npt.NDArray[np.floating], val: np.floating) -> int:
    if arr.size <= 1:
        return 0

    if arr[0] > arr[-1]:
        # Reverse descending coordinates to reuse midpoint search logic.
        arr_rev = arr[::-1]
        return arr.size - 1 - np.searchsorted((arr_rev[:-1] + arr_rev[1:]) / 2, val)

    return np.searchsorted((arr[:-1] + arr[1:]) / 2, val)


@numba.njit(
    [
        numba.int32(numba.int32[::1]),
        numba.int64(numba.int64[::1]),
        numba.float32(numba.float32[::1]),
        numba.float64(numba.float64[::1]),
    ],
    cache=True,
)
def _avg_nonzero_abs_diff(arr: npt.NDArray[np.number]) -> np.number:
    diff = np.diff(arr)

    if np.all(diff == 0.0):  # Prevent division by zero
        return diff[0]
    return np.mean(diff[diff != 0])
