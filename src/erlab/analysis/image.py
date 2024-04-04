"""
Various image processing functions including tools for visualizing dispersive features.
"""

import numpy as np
import numpy.typing as npt
import scipy
import scipy.ndimage
import xarray as xr
from numba import carray, cfunc, types


def gradient_magnitude(
    input: npt.NDArray[np.float64],
    dx: np.float64,
    dy: np.float64,
    mode: str = "nearest",
    cval: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Calculate the gradient magnitude of an input array using a given dx and dy.

    Parameters
    ----------
    input
        Input array.
    dx
        Step size in the x-direction.
    dy
        Step size in the y-direction.
    mode, cval
        Arguments to be passed to `scipy.ndimage.generic_filter`.

    Returns
    -------
    output : ndarray
        Gradient magnitude of the input array. Has the same shape as `input`.

    Notes
    -----
    This function calculates the gradient magnitude of an input array by applying a
    filter to the input array using the given dx and dy values. The filter is defined by
    a kernel function that computes the squared difference between each element of the
    input array and the central element, divided by the corresponding distance value.
    The gradient magnitude is then calculated as the square root of the sum of the
    squared differences.
    """

    dxy = np.sqrt(dx**2 + dy**2)
    dist = np.array([[dxy, dy, dxy], [dx, 1.0, dx], [dxy, dy, dxy]]).flatten()

    @cfunc(
        types.intc(
            types.CPointer(types.float64),
            types.intp,
            types.CPointer(types.float64),
            types.voidptr,
        )
    )
    def _kernel(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=types.float64)
        val = 0.0
        for i in range(9):
            if i != 4:
                val += ((values[i] - values[4]) / dist[i]) ** 2
        result[0] = np.sqrt(val)
        return 1

    # https://github.com/jni/llc-tools/issues/3#issuecomment-757134814
    func = scipy.LowLevelCallable(
        _kernel.ctypes, signature="int (double *, npy_intp, double *, void *)"
    )

    return scipy.ndimage.generic_filter(input, func, size=(3, 3), mode=mode, cval=cval)


def minimum_gradient(
    darr: xr.DataArray, mode: str = "nearest", cval: float = 0.0
) -> xr.DataArray:
    """Minimum gradient method for detecting dispersive features in 2D data as described
    in Ref. :cite:p:`He2017`.

    Parameters
    ----------
    darr : xr.DataArray
        The 2D DataArray for which to calculate the minimum gradient.
    mode : str, optional
        The mode parameter controls how the gradient is calculated at the boundaries.
        Default is 'nearest'. See `scipy.ndimage.generic_filter` for more information.
    cval : float, optional
        The value to use for points outside the boundaries when mode is 'constant'.
        Default is 0.0. See `scipy.ndimage.generic_filter` for more information.

    Returns
    -------
    xarray.DataArray
        The minimum gradient of the input DataArray. Has the same shape as `input`.

    Raises
    ------
    ValueError
        If the input DataArray is not 2D.

    Notes
    -----
    The minimum gradient is calculated by dividing the input DataArray by the gradient
    magnitude. Any zero gradient values are replaced with NaN.
    """

    if darr.ndim != 2:
        raise ValueError("DataArray must be 2D")

    xvals = darr[darr.dims[1]].values
    yvals = darr[darr.dims[0]].values

    dx = xvals[1] - xvals[0]
    dy = yvals[1] - yvals[0]

    grad = gradient_magnitude(darr.values, dx, dy, mode=mode, cval=cval)
    grad[grad == 0] = np.nan
    return darr / grad
