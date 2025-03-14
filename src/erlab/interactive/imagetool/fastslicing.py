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
    ind = min(round((val - lims[axis][0]) / incs[axis]), shape[axis] - 1)
    if ind < 0:
        return 0
    return ind


@numba.njit(cache=True)
def _transposed(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if arr.ndim == 2:
        return arr.T
    if arr.ndim == 3:
        return arr.transpose(1, 2, 0)
    return arr.transpose(1, 2, 3, 0)


@numba.njit(
    [
        numba.boolean(numba.int32[::1]),
        numba.boolean(numba.int64[::1]),
        numba.boolean(numba.float32[::1]),
        numba.boolean(numba.float64[::1]),
    ],
    cache=True,
)
def _is_uniform(arr: npt.NDArray[np.float64]) -> bool:
    dif = np.diff(arr)
    if dif[0] == 0.0:
        # Treat constant coordinate array as non-uniform
        return False
    return np.allclose(dif, dif[0], rtol=3e-05, atol=3e-05, equal_nan=True)


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
