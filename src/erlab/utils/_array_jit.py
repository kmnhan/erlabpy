import numba
import numpy as np
import numpy.typing as npt


@numba.njit(numba.boolean(numba.float64[::1]), cache=True)
def _check_uniform(arr: npt.NDArray[np.float64]) -> bool:
    """Check if a 1D NumPy array is uniformly spaced."""
    dif = np.diff(arr)
    if dif.size == 0:
        return True
    return np.allclose(dif, dif[0])


@numba.njit(numba.types.ListType(numba.float64[:])(numba.float64[:]), cache=True)
def _split_uniform_segments(
    arr: npt.NDArray[np.float64],
) -> numba.typed.List[npt.NDArray[np.float64]]:
    """Split a 1D NumPy array into segments that are evenly spaced."""
    n = arr.shape[0]
    if n < 2:
        segments = numba.typed.List()
        segments.append(arr)
        return segments

    diff = np.empty(n - 1, arr.dtype)
    for i in range(n - 1):
        diff[i] = arr[i + 1] - arr[i]

    segments = numba.typed.List()
    start = 0

    for i in range(1, n - 1):
        if not np.isclose(diff[i], diff[i - 1]):
            if i - start > 1:
                segments.append(arr[start : (i + 1)])
            start = i
    segments.append(arr[start:n])

    return segments
