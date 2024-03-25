"""
This module provides a fast, parallelized version of nanmean that supports multiple axes
based on numba. Enables efficient real-time multidimensional binning.

"""

__all__ = ["fast_nanmean"]

import numpy as np
import numpy.typing as npt
from collections.abc import Iterable
import numba
import numba.typed
import numba.core.registry
import numbagg


@numba.njit(cache=True)
def _nanmean_all(a: npt.NDArray[np.float32 | np.float64]) -> np.float64:
    return np.nanmean(a)


@numba.njit(cache=True, parallel=True)
def _nanmean_2_0(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[1]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, i])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_2_1(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[0]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[i, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_3_0(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, n0, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[:, i, j])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_3_1(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, _, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, :, j])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_3_2(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, n1, _ = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, j, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_3_01(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[2]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, :, i])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_3_02(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[1]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, i, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_3_12(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[0]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[i, :, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_0(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, n0, n1, n2 = a.shape
    output = np.empty((n0, n1, n2), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            for k in range(n2):
                output[i, j, k] = np.nanmean(a[:, i, j, k])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_1(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, _, n1, n2 = a.shape
    output = np.empty((n0, n1, n2), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            for k in range(n2):
                output[i, j, k] = np.nanmean(a[i, :, j, k])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_2(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, n1, _, n2 = a.shape
    output = np.empty((n0, n1, n2), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            for k in range(n2):
                output[i, j, k] = np.nanmean(a[i, j, :, k])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_3(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, n1, n2, _ = a.shape
    output = np.empty((n0, n1, n2), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            for k in range(n2):
                output[i, j, k] = np.nanmean(a[i, j, k, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_01(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, _, n0, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[:, :, i, j])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_02(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, n0, _, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[:, i, :, j])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_03(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, n0, n1, _ = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[:, i, j, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_12(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, _, _, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, :, :, j])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_13(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, _, n1, _ = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, :, j, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_23(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, n1, _, _ = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, j, :, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_012(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[3]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, :, :, i])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_013(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[2]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, :, i, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_023(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[1]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, i, :, :])
    return output


@numba.njit(cache=True, parallel=True)
def _nanmean_4_123(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[0]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[i, :, :, :])
    return output


nanmean_funcs = {
    2: {
        0: _nanmean_2_0,
        1: _nanmean_2_1,
        frozenset({0}): _nanmean_2_0,
        frozenset({1}): _nanmean_2_1,
    },
    3: {
        0: _nanmean_3_0,
        1: _nanmean_3_1,
        2: _nanmean_3_2,
        frozenset({0}): _nanmean_3_0,
        frozenset({1}): _nanmean_3_1,
        frozenset({2}): _nanmean_3_2,
        frozenset({0, 1}): _nanmean_3_01,
        frozenset({0, 2}): _nanmean_3_02,
        frozenset({1, 2}): _nanmean_3_12,
    },
    4: {
        0: _nanmean_4_0,
        1: _nanmean_4_1,
        2: _nanmean_4_2,
        3: _nanmean_4_3,
        frozenset({0}): _nanmean_4_0,
        frozenset({1}): _nanmean_4_1,
        frozenset({2}): _nanmean_4_2,
        frozenset({3}): _nanmean_4_3,
        frozenset({0, 1}): _nanmean_4_01,
        frozenset({0, 2}): _nanmean_4_02,
        frozenset({0, 3}): _nanmean_4_03,
        frozenset({1, 2}): _nanmean_4_12,
        frozenset({1, 3}): _nanmean_4_13,
        frozenset({2, 3}): _nanmean_4_23,
        frozenset({0, 1, 2}): _nanmean_4_012,
        frozenset({0, 1, 3}): _nanmean_4_013,
        frozenset({0, 2, 3}): _nanmean_4_023,
        frozenset({1, 2, 3}): _nanmean_4_123,
    },
}


def fast_nanmean(
    a: npt.NDArray[np.float32 | np.float64], axis: int | Iterable[int] | None = None
) -> npt.NDArray[np.float32 | np.float64] | float:
    """A fast, parallelized arithmetic mean for floating point arrays that ignores NaNs.

    Parameters
    ----------
    a
        A numpy array of floats.
    axis
        Axis or iterable of axis along which the means are computed. If `None`, the mean
        of the flattend array is computed.

    Returns
    -------
    numpy.ndarray or float
        The calculated mean. The output array is always C-contiguous.

    Note
    ----
    Parallelization is only applied for :code:`N`-dimensional arrays with :code:`N <= 4`
    and :code:`len(axis) < N`. For bigger :code:`N`, :obj:`numbagg.nanmean` is used. For
    calculating the average of a flattened array (:code:`axis = None` or
    :code:`len(axis) == N`), the :obj:`numba` implemenation of :obj:`numpy.nanmean` is
    used. This function does not keep the input dimensions, i.e., the output is
    squeezed.

    """
    if a.ndim == 1 or axis is None:
        return _nanmean_all(a)
    elif a.ndim > 4:
        return np.ascontiguousarray(numbagg.nanmean(a, axis))
    if hasattr(axis, "__iter__"):
        if len(axis) == a.ndim:
            return _nanmean_all(a)
        axis = frozenset(x % a.ndim for x in axis)
    else:
        axis = axis % a.ndim
    return nanmean_funcs[a.ndim][axis](a).astype(a.dtype)


def _fast_nanmean_skipcheck(
    a: npt.NDArray[np.float32 | np.float64], axis: int | Iterable[int]
) -> npt.NDArray[np.float32 | np.float64] | float:
    """A version of `fast_nanmean` with near-zero overhead. Meant for internal use.

    Strict assumptions on the input parameters allow skipping some checks.

    Parameters
    ----------
    a
        A numpy array of floats. :code:`a.ndim` must be one of 2, 3, and 4.
    axis
        Axis or iterable of axis along which the means are computed. All elements must
        be nonnegative integers that are less than or equal to :code:`a.ndim`, i.e.,
        negative indexing is not allowed.

    Returns
    -------
    numpy.ndarray or float
        The calculated mean. The output array is always C-contiguous.

    """
    if hasattr(axis, "__iter__"):
        if len(axis) == a.ndim:
            return _nanmean_all(a)
        axis = frozenset(axis)
    return nanmean_funcs[a.ndim][axis](a).astype(a.dtype)


if __name__ == "__main__":
    for nd, funcs in nanmean_funcs.items():
        x = np.random.RandomState(42).randn(*((30,) * nd))
        for axis, func in funcs.items():
            if isinstance(axis, frozenset):
                axis = tuple(axis)
            if not np.allclose(np.nanmean(x, axis), fast_nanmean(x, axis)):
                print(func)
