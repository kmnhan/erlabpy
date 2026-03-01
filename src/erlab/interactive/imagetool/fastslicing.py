"""Numba-compiled functions for fast slicing of 2D, 3D, and 4D arrays."""

import numba
import numpy as np
import numpy.typing as npt

VALID_NDIM = (2, 3, 4)

_signature_array_rect = [
    numba.types.UniTuple(ftype, 4)(
        numba.int64,
        numba.int64,
        numba.types.UniTuple(numba.types.UniTuple(ftype, 2), i),
        numba.types.UniTuple(ftype, i),
    )
    for ftype in (numba.float32, numba.float64)
    for i in VALID_NDIM
]
_signature_index_of_value = [
    numba.int64(
        numba.int64,
        ftype,
        numba.types.UniTuple(numba.types.UniTuple(ftype, 2), i),
        numba.types.UniTuple(ftype, i),
        numba.types.UniTuple(numba.int64, i),
    )
    for ftype in (numba.float32, numba.float64)
    for i in VALID_NDIM
]


@numba.njit(_signature_array_rect, cache=True, fastmath=True)
def _array_rect(
    i: int,
    j: int,
    lims: tuple[tuple[np.floating, np.floating], ...],
    incs: tuple[np.floating, ...],
) -> tuple[np.floating, np.floating, np.floating, np.floating]:
    x = lims[i][0] - incs[i]
    y = lims[j][0] - incs[j]
    w = lims[i][-1] - x
    h = lims[j][-1] - y
    x += 0.5 * incs[i]
    y += 0.5 * incs[j]
    return x, y, w, h


@numba.njit(_signature_index_of_value, cache=True)
def _index_of_value(
    axis: int,
    val: np.floating,
    lims: tuple[tuple[np.floating, np.floating], ...],
    incs: tuple[np.floating, ...],
    shape: tuple[int],
) -> int:
    delta = incs[axis]
    if delta == 0:
        return 0
    ind = min(round((val - lims[axis][0]) / delta), shape[axis] - 1)
    if ind < 0:
        return 0
    return ind


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
