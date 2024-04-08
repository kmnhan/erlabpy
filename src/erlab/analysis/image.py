"""
Various image processing functions including tools for visualizing dispersive features.

Note
----
For scipy-based filter functions, the default value of the `mode` argument is 'nearest',
unlike the scipy default of 'reflect'.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy
import scipy.ndimage
import xarray as xr
from numba import carray, cfunc, types


def gaussian_filter(
    darr: xr.DataArray,
    sigma: float | dict[str, float] | Sequence[float],
    order: int | Sequence[int] | dict[str, int] = 0,
    mode: str | Sequence[str] | dict[str, str] = "nearest",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: None | float | Sequence[float] | dict[str, float] = None,
) -> xr.DataArray:
    """Coordinate-aware wrapper around `scipy.ndimage.gaussian_filter`.

    Parameters
    ----------
    darr
        The input DataArray.
    sigma
        The standard deviation(s) of the Gaussian filter in data dimensions. If a float,
        the same value is used for all dimensions, each scaled by the data step. If a
        dict, the value can be specified for each dimension using dimension names as
        keys. The filter is only applied to the dimensions specified in the dict. If a
        sequence, the values are used in the same order as the dimensions of the
        DataArray.
    order
        The order of the filter along each dimension. If an int, the same order is used
        for all dimensions. See Notes below for other options. Defaults to 0.
    mode
        The boundary mode used for the filter. If a str, the same mode is used for all
        dimensions. See Notes below for other options. Defaults to 'nearest'.
    cval
        Value to fill past edges of input if mode is 'constant'. Defaults to 0.0.
    truncate
        The truncation value used for the Gaussian filter. Defaults to 4.0.
    radius
        The radius of the Gaussian filter in data units. See Notes below. Defaults to
        None.

    Returns
    -------
    gaussian_filter : xarray.DataArray
        The filtered array with the same shape as the input DataArray.

    Note
    ----
    - The `sigma` and `radius` values should be in data coordinates, not pixels.
    - The input array is assumed to be regularly spaced.
    - `order`, `mode`, and `radius` can be specified for each dimension using a dict or
      a sequence. If a dict, the value can be specified for each dimension using
      dimension names as keys. If a sequence and `sigma` is given as a dictionary, the
      order is assumed to be the same as the keys in `sigma`. If `sigma` is not a
      dictionary, the order is assumed to be the same as the dimensions of the
      DataArray.

    See Also
    --------
    :func:`scipy.ndimage.gaussian_filter` : The underlying function used to apply the
        filter.

    Example
    -------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import erlab.analysis as era
    >>> darr = xr.DataArray(np.arange(50, step=2).reshape((5, 5)), dims=["x", "y"])
    >>> darr
    <xarray.DataArray (x: 5, y: 5)> Size: 200B
    array([[ 0,  2,  4,  6,  8],
        [10, 12, 14, 16, 18],
        [20, 22, 24, 26, 28],
        [30, 32, 34, 36, 38],
        [40, 42, 44, 46, 48]])
    Dimensions without coordinates: x, y
    >>> era.image.gaussian_filter(darr, sigma=dict(x=1.0, y=1.0))
    <xarray.DataArray (x: 5, y: 5)> Size: 200B
    array([[ 3,  5,  7,  8, 10],
        [10, 12, 14, 15, 17],
        [20, 22, 24, 25, 27],
        [29, 31, 33, 34, 36],
        [36, 38, 40, 41, 43]])
    Dimensions without coordinates: x, y

    """
    if np.isscalar(sigma):
        sigma = {d: sigma for d in darr.dims}
    elif not isinstance(sigma, dict):
        sigma = dict(zip(darr.dims, sigma))

    # Get the axis indices to apply the filter
    axes = tuple(darr.get_axis_num(d) for d in sigma.keys())

    # Convert arguments to tuples acceptable by scipy
    if isinstance(order, dict):
        order = tuple(order.get(d, 0) for d in sigma.keys())
    if isinstance(mode, dict):
        mode = tuple(mode[d] for d in sigma.keys())
    if radius is not None:
        if len(radius) != len(sigma):
            raise ValueError("`radius` does not match dimensions of `sigma`")

        if np.isscalar(radius):
            radius = {d: radius for d in sigma.keys()}
        elif not isinstance(radius, dict):
            radius = dict(zip(sigma.keys(), radius))

        # Calculate radius in pixels
        radius: tuple[int, ...] = tuple(
            round(r / (darr[d].values[1] - darr[d].values[0]))
            for d, r in radius.items()
        )

    # Calculate sigma in pixels
    sigma: tuple[float, ...] = tuple(
        val / (darr[d].values[1] - darr[d].values[0]) for d, val in sigma.items()
    )

    return darr.copy(
        data=scipy.ndimage.gaussian_filter(
            darr.values,
            sigma=sigma,
            order=order,
            mode=mode,
            cval=cval,
            truncate=truncate,
            radius=radius,
            axes=axes,
        )
    )


def gaussian_laplace(
    darr: xr.DataArray,
    sigma: float | dict[str, float] | Sequence[float],
    mode: str | Sequence[str] | dict[str, str] = "nearest",
    cval: float = 0.0,
    **kwargs,
) -> xr.DataArray:
    """Coordinate-aware wrapper around `scipy.ndimage.gaussian_laplace`.

    This function calculates the Laplacian of the given array using Gaussian second
    derivatives.

    Parameters
    ----------
    darr
        The input DataArray.
    sigma
        The standard deviation(s) of the Gaussian filter in data dimensions. If a float,
        the same value is used for all dimensions, each scaled by the data step. If a
        dict, the value can be specified for each dimension using dimension names as
        keys. If a sequence, the values are used in the same order as the dimensions of
        the DataArray.
    mode
        The mode parameter determines how the input array is extended beyond its
        boundaries. If a string, the same mode is used for all dimensions. If a
        sequence, the values should be the modes for each dimension in the same order as
        the dimensions in the DataArray. If a dictionary, the keys should be dimension
        names and the values should be the corresponding modes, and every dimension in
        the DataArray must be present. Default is "nearest".
    cval
        Value to fill past edges of input if mode is 'constant'. Defaults to 0.0.
    **kwargs
        Additional keyword arguments to `scipy.ndimage.gaussian_filter`.

    Returns
    -------
    gaussian_laplace : xarray.DataArray
        The filtered array with the same shape as the input DataArray.

    Note
    ----
    - `sigma` should be in data coordinates, not pixels.
    - The input array is assumed to be regularly spaced.

    See Also
    --------
    :func:`scipy.ndimage.gaussian_laplace` : The underlying function used to apply the
        filter.
    """
    if np.isscalar(sigma):
        sigma = {d: sigma for d in darr.dims}
    elif not isinstance(sigma, dict):
        sigma = dict(zip(darr.dims, sigma))

    if len(sigma) != darr.ndim:
        raise ValueError(
            "`sigma` must be provided for every dimension of the DataArray"
        )

    # Convert mode to tuple acceptable by scipy
    if isinstance(mode, dict):
        mode = tuple(mode[d] for d in sigma.keys())

    # Calculate sigma in pixels
    sigma: tuple[float, ...] = tuple(
        val / (darr[d].values[1] - darr[d].values[0]) for d, val in sigma.items()
    )

    return darr.copy(
        data=scipy.ndimage.gaussian_laplace(
            darr.values, sigma=sigma, mode=mode, cval=cval, **kwargs
        )
    )


def gradient_magnitude(
    input: npt.NDArray[np.float64],
    dx: np.float64,
    dy: np.float64,
    mode: str = "nearest",
    cval: float = 0.0,
) -> npt.NDArray[np.float64]:
    r"""Calculate the gradient magnitude of an input array.

    The gradient magnitude is calculated as defined in Ref. :cite:p:`he2017mingrad`,
    using given :math:`\Delta x` and :math:`\Delta y` values.

    Parameters
    ----------
    input
        Input array.
    dx
        Step size in the x-direction.
    dy
        Step size in the y-direction.
    mode
        The mode parameter controls how the gradient is calculated at the boundaries.
        Default is 'nearest'. See `scipy.ndimage.generic_filter` for more information.
    cval
        The value to use for points outside the boundaries when mode is 'constant'.
        Default is 0.0. See `scipy.ndimage.generic_filter` for more information.

    Returns
    -------
    gradient_magnitude : numpy.ndarray
        Gradient magnitude of the input array. Has the same shape as :code:`input`.

    Note
    ----
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


def laplace(
    darr, mode: str | Sequence[str] | dict[str, str] = "nearest", cval: float = 0.0
) -> xr.DataArray:
    """Coordinate-aware wrapper around `scipy.ndimage.laplace`.

    This function calculates the Laplacian of the given array using approximate second
    derivatives.

    Parameters
    ----------
    darr
        The input DataArray.
    mode
        The mode parameter determines how the input array is extended beyond its
        boundaries. If a dictionary, the keys should be dimension names and the values
        should be the corresponding modes, and every dimension in the DataArray must be
        present. Otherwise, it retains the same behavior as in `scipy.ndimage.laplace`.
        Default is 'nearest'.
    cval
        Value to fill past edges of input if mode is 'constant'. Defaults to 0.0.

    Returns
    -------
    laplace : xarray.DataArray
        The filtered array with the same shape as the input DataArray.

    See Also
    --------
    :func:`scipy.ndimage.laplace` : The underlying function used to apply the filter.
    """
    if isinstance(mode, dict):
        mode = tuple(mode[d] for d in darr.dims)
    return darr.copy(data=scipy.ndimage.laplace(darr.values, mode=mode, cval=cval))


def minimum_gradient(
    darr: xr.DataArray, mode: str = "nearest", cval: float = 0.0
) -> xr.DataArray:
    """Minimum gradient method for detecting dispersive features in 2D data.

    The minimum gradient is calculated by dividing the input DataArray by the gradient
    magnitude. See Ref. :cite:p:`he2017mingrad`.

    Parameters
    ----------
    darr
        The 2D DataArray for which to calculate the minimum gradient.
    mode
        The mode parameter controls how the gradient is calculated at the boundaries.
        Default is 'nearest'. See `scipy.ndimage.generic_filter` for more information.
    cval
        The value to use for points outside the boundaries when mode is 'constant'.
        Default is 0.0. See `scipy.ndimage.generic_filter` for more information.

    Returns
    -------
    minimum_gradient : xarray.DataArray
        The minimum gradient of the input DataArray. Has the same shape as
        :code:`input`.

    Raises
    ------
    ValueError
        If the input DataArray is not 2D.

    Note
    ----
    - The input array is assumed to be regularly spaced.
    - Any zero gradient values are replaced with NaN.
    """

    if darr.ndim != 2:
        raise ValueError("DataArray must be 2D")

    xvals = darr[darr.dims[1]].values
    yvals = darr[darr.dims[0]].values

    dx = xvals[1] - xvals[0]
    dy = yvals[1] - yvals[0]

    grad = gradient_magnitude(
        darr.values.astype(np.float64), dx, dy, mode=mode, cval=cval
    )
    grad[grad == 0] = np.nan
    return darr / grad


def scaled_laplace(
    darr,
    factor: float = 1.0,
    mode: str | Sequence[str] | dict[str, str] = "nearest",
    cval: float = 0.0,
) -> xr.DataArray:
    r"""Calculate the Laplacian of a 2D DataArray with different scaling for each axis.

    This function calculates the Laplacian of the given array using approximate second
    derivatives, taking the different scaling for each axis into account.

    .. math::
        \Delta f \sim \frac{\partial^2 f}{\partial x^2} \left(\frac{\Delta x}{\Delta y}\right)^{\!2} + \frac{\partial^2 f}{\partial y^2}

    See Ref. :cite:p:`zhang2011curvature` for more information.

    Parameters
    ----------
    darr
        The 2D DataArray for which to calculate the scaled Laplacian.
    factor
        The factor by which to scale the x-axis derivative. Negative values will scale
        the y-axis derivative instead. Default is 1.0.
    mode
        The mode parameter determines how the input array is extended beyond its
        boundaries. If a dictionary, the keys should be dimension names and the values
        should be the corresponding modes, and every dimension in the DataArray must be
        present. Otherwise, it retains the same behavior as in
        `scipy.ndimage.generic_laplace`. Default is 'nearest'.
    cval
        Value to fill past edges of input if mode is 'constant'. Defaults to 0.0.

    Returns
    -------
    scaled_laplace : xarray.DataArray
        The filtered array with the same shape as the input DataArray.

    Raises
    ------
    ValueError
        If the input DataArray is not 2D.

    Note
    ----
    The input array is assumed to be regularly spaced.

    See Also
    --------
    :func:`scipy.ndimage.generic_laplace` : The underlying function used to apply the
        filter.
    """
    if darr.ndim != 2:
        raise ValueError("DataArray must be 2D")

    xvals = darr[darr.dims[1]].values
    yvals = darr[darr.dims[0]].values

    dx = xvals[1] - xvals[0]
    dy = yvals[1] - yvals[0]
    weight = (dx / dy) ** 2

    if factor > 0:
        weight *= factor
    elif factor < 0:
        weight /= abs(factor)

    if isinstance(mode, dict):
        mode = tuple(mode[d] for d in darr.dims)

    def d2_scaled(input, axis, output, mode, cval):
        out = scipy.ndimage.correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)
        if axis == 1:
            out *= weight
        return out

    return darr.copy(
        data=scipy.ndimage.generic_laplace(darr.values, d2_scaled, mode=mode, cval=cval)
    )


def curvature(darr: xr.DataArray, a0: float = 1.0, factor: float = 1.0) -> xr.DataArray:
    """2D curvature method for detecting dispersive features.

    The curvature is calculated as defined by :cite:t:`zhang2011curvature`.

    Parameters
    ----------
    darr
        The 2D DataArray for which to calculate the curvature.
    a0
        The regularization constant. Reasonable values range from 0.001 to 10. Default
        is 1.0.
    factor
        The factor by which to scale the x-axis curvature. Negative values will scale
        the y-axis curvature instead. Default is 1.0.

    Returns
    -------
    curvature : xarray.DataArray
        The 2D curvature of the input DataArray. Has the same shape as :code:`input`.

    Raises
    ------
    ValueError
        If the input DataArray is not 2D.

    Note
    ----
    The input array is assumed to be regularly spaced.
    """

    if darr.ndim != 2:
        raise ValueError("DataArray must be 2D")

    xvals = darr[darr.dims[1]].values
    yvals = darr[darr.dims[0]].values

    dx = xvals[1] - xvals[0]
    dy = yvals[1] - yvals[0]
    weight = (dx / dy) ** 2

    if factor > 0:
        weight *= factor
    elif factor < 0:
        weight /= abs(factor)

    dfdx, dfdy = np.gradient(darr.values, axis=(1, 0))
    d2fdx2, d2fdydx = np.gradient(dfdx, axis=(1, 0))
    d2fdy2 = np.gradient(dfdy, axis=0)

    max_abs_dfdx_sq: float = np.max(np.abs(dfdx)) ** 2
    max_abs_dfdy_sq: float = np.max(np.abs(dfdy)) ** 2

    c0 = a0 * max(max_abs_dfdy_sq, weight * max_abs_dfdx_sq)

    curv = (
        (c0 + weight * dfdx**2) * d2fdy2
        - 2 * weight * dfdx * dfdy * d2fdydx
        + weight * (c0 + dfdy**2) * d2fdx2
    ) / (c0 + weight * dfdx**2 + dfdy**2) ** 1.5

    return darr.copy(data=curv)
