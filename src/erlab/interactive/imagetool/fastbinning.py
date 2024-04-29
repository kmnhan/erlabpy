"""Fast parallelized averaging for multidimensional arrays.

This module provides a numba-based fast, parallelized version of nanmean that supports
multiple axes. This enables efficient real-time multidimensional binning.

"""

__all__ = ["NANMEAN_FUNCS", "fast_nanmean", "fast_nanmean_skipcheck"]

import importlib
from collections.abc import Callable, Collection

import numba
import numba.core.registry
import numba.typed
import numpy as np
import numpy.typing as npt

if importlib.util.find_spec("numbagg"):
    import numbagg

    _general_nanmean_func: Callable = numbagg.nanmean
else:
    _general_nanmean_func = np.nanmean

# _SIG_N_M: List of signatures that reduces from N to M dimensions.

_SIG_2_1 = [
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float32, 2, "C")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float32, 2, "A")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float64, 2, "C")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float64, 2, "A")),
]
_SIG_3_1 = [
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float32, 3, "C")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float32, 3, "A")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float64, 3, "C")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float64, 3, "A")),
]
_SIG_3_2 = [
    numba.types.Array(numba.float64, 2, "C")(numba.types.Array(numba.float32, 3, "C")),
    numba.types.Array(numba.float64, 2, "C")(numba.types.Array(numba.float32, 3, "A")),
    numba.types.Array(numba.float64, 2, "C")(numba.types.Array(numba.float64, 3, "C")),
    numba.types.Array(numba.float64, 2, "C")(numba.types.Array(numba.float64, 3, "A")),
]
_SIG_4_1 = [
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float32, 4, "C")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float32, 4, "A")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float64, 4, "C")),
    numba.types.Array(numba.float64, 1, "C")(numba.types.Array(numba.float64, 4, "A")),
]
_SIG_4_2 = [
    numba.types.Array(numba.float64, 2, "C")(numba.types.Array(numba.float32, 4, "C")),
    numba.types.Array(numba.float64, 2, "C")(numba.types.Array(numba.float32, 4, "A")),
    numba.types.Array(numba.float64, 2, "C")(numba.types.Array(numba.float64, 4, "C")),
    numba.types.Array(numba.float64, 2, "C")(numba.types.Array(numba.float64, 4, "A")),
]
_SIG_4_3 = [
    numba.types.Array(numba.float64, 3, "C")(numba.types.Array(numba.float32, 4, "C")),
    numba.types.Array(numba.float64, 3, "C")(numba.types.Array(numba.float32, 4, "A")),
    numba.types.Array(numba.float64, 3, "C")(numba.types.Array(numba.float64, 4, "C")),
    numba.types.Array(numba.float64, 3, "C")(numba.types.Array(numba.float64, 4, "A")),
]


@numba.njit(cache=True)
def _nanmean_all(a: npt.NDArray[np.float32 | np.float64]) -> np.float64:
    return np.nanmean(a)


@numba.njit(_SIG_2_1, cache=True, parallel=True)
def _nanmean_2_0(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[1]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, i])
    return output


@numba.njit(_SIG_2_1, cache=True, parallel=True)
def _nanmean_2_1(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[0]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[i, :])
    return output


@numba.njit(_SIG_3_2, cache=True, parallel=True)
def _nanmean_3_0(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, n0, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[:, i, j])
    return output


@numba.njit(_SIG_3_2, cache=True, parallel=True)
def _nanmean_3_1(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, _, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, :, j])
    return output


@numba.njit(_SIG_3_2, cache=True, parallel=True)
def _nanmean_3_2(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, n1, _ = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, j, :])
    return output


@numba.njit(_SIG_3_1, cache=True, parallel=True)
def _nanmean_3_01(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[2]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, :, i])
    return output


@numba.njit(_SIG_3_1, cache=True, parallel=True)
def _nanmean_3_02(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[1]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, i, :])
    return output


@numba.njit(_SIG_3_1, cache=True, parallel=True)
def _nanmean_3_12(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[0]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[i, :, :])
    return output


@numba.njit(_SIG_4_3, cache=True, parallel=True)
def _nanmean_4_0(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, n0, n1, n2 = a.shape
    output = np.empty((n0, n1, n2), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            for k in range(n2):
                output[i, j, k] = np.nanmean(a[:, i, j, k])
    return output


@numba.njit(_SIG_4_3, cache=True, parallel=True)
def _nanmean_4_1(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, _, n1, n2 = a.shape
    output = np.empty((n0, n1, n2), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            for k in range(n2):
                output[i, j, k] = np.nanmean(a[i, :, j, k])
    return output


@numba.njit(_SIG_4_3, cache=True, parallel=True)
def _nanmean_4_2(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, n1, _, n2 = a.shape
    output = np.empty((n0, n1, n2), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            for k in range(n2):
                output[i, j, k] = np.nanmean(a[i, j, :, k])
    return output


@numba.njit(_SIG_4_3, cache=True, parallel=True)
def _nanmean_4_3(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, n1, n2, _ = a.shape
    output = np.empty((n0, n1, n2), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            for k in range(n2):
                output[i, j, k] = np.nanmean(a[i, j, k, :])
    return output


@numba.njit(_SIG_4_2, cache=True, parallel=True)
def _nanmean_4_01(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, _, n0, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[:, :, i, j])
    return output


@numba.njit(_SIG_4_2, cache=True, parallel=True)
def _nanmean_4_02(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, n0, _, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[:, i, :, j])
    return output


@numba.njit(_SIG_4_2, cache=True, parallel=True)
def _nanmean_4_03(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    _, n0, n1, _ = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[:, i, j, :])
    return output


@numba.njit(_SIG_4_2, cache=True, parallel=True)
def _nanmean_4_12(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, _, _, n1 = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, :, :, j])
    return output


@numba.njit(_SIG_4_2, cache=True, parallel=True)
def _nanmean_4_13(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, _, n1, _ = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, :, j, :])
    return output


@numba.njit(_SIG_4_2, cache=True, parallel=True)
def _nanmean_4_23(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n0, n1, _, _ = a.shape
    output = np.empty((n0, n1), dtype=np.float64)
    for i in numba.prange(n0):
        for j in range(n1):
            output[i, j] = np.nanmean(a[i, j, :, :])
    return output


@numba.njit(_SIG_4_1, cache=True, parallel=True)
def _nanmean_4_012(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[3]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, :, :, i])
    return output


@numba.njit(_SIG_4_1, cache=True, parallel=True)
def _nanmean_4_013(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[2]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, :, i, :])
    return output


@numba.njit(_SIG_4_1, cache=True, parallel=True)
def _nanmean_4_023(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[1]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[:, i, :, :])
    return output


@numba.njit(_SIG_4_1, cache=True, parallel=True)
def _nanmean_4_123(a: npt.NDArray[np.float32 | np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[0]
    output = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        output[i] = np.nanmean(a[i, :, :, :])
    return output


NANMEAN_FUNCS: dict[int, dict[int | frozenset[int], Callable]] = {
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
}  #: Mapping from array dimensions to axis combinations to corresponding functions.


def fast_nanmean(
    a: npt.NDArray[np.float32 | np.float64], axis: int | Collection[int] | None = None
) -> npt.NDArray[np.float32 | np.float64] | np.float64:
    """Compute the mean for floating point arrays while ignoring NaNs.

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

    Notes
    -----
    - Parallelization is only applied for ``N``-dimensional arrays with ``N <= 4`` and
      ``len(axis) < N``.

    - For calculating the average of a flattened array (``axis = None`` or ``len(axis)
      == N``), the `numba` implemenation of `numpy.nanmean` is used.

    - For bigger ``N``, ``numbagg.nanmean`` is used if `numbagg
      <https://github.com/numbagg/numbagg>`_ is installed. Otherwise, the calculation
      falls back to `numpy.nanmean`.

    - This function does not keep the input dimensions, i.e., the output is squeezed.

    - For single precision input, the calculation is performed in double precision and
      converted back to single precision. This may lead to different results compared to
      `numpy.nanmean`.

    """
    if a.ndim == 1 or axis is None:
        return _nanmean_all(a)
    elif a.ndim > 4:
        return np.ascontiguousarray(
            _general_nanmean_func(a.astype(np.float64), axis), dtype=a.dtype
        )
    if isinstance(axis, Collection):
        if len(axis) == a.ndim:
            return _nanmean_all(a)
        axis = frozenset(x % a.ndim for x in axis)
    else:
        axis = axis % a.ndim
    return NANMEAN_FUNCS[a.ndim][axis](a).astype(a.dtype)


def fast_nanmean_skipcheck(
    a: npt.NDArray[np.float32 | np.float64], axis: int | Collection[int]
) -> npt.NDArray[np.float32 | np.float64] | np.float64:
    """Compute the mean for specific floating point arrays while ignoring NaNs.

    This is a version of `fast_nanmean` with near-zero overhead meant for internal use.
    Strict assumptions on the input parameters allow skipping some checks. Failure to
    meet these assumptions may lead to undefined behavior.

    Parameters
    ----------
    a
        A numpy array of floats. ``a.ndim`` must be one of 2, 3, and 4.
    axis
        Axis or iterable of axis along which the means are computed. All elements must
        be nonnegative integers that are less than or equal to ``a.ndim``, i.e.,
        negative indexing is not allowed.

    Returns
    -------
    numpy.ndarray or float
        The calculated mean. The output array is always C-contiguous.

    """
    if isinstance(axis, Collection):
        if len(axis) == a.ndim:
            return _nanmean_all(a)
        axis = frozenset(axis)
    return NANMEAN_FUNCS[a.ndim][axis](a).astype(a.dtype)
