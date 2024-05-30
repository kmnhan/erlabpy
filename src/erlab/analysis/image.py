"""
Various image processing functions including tools for visualizing dispersive features.

Some filter functions in `scipy.ndimage` and `scipy.signal` are extended to work with
regularly spaced xarray DataArrays.

Notes
-----
- For many scipy-based filter functions, the default value of the `mode` argument is
  different from scipy.
- Many functions in this module has conflicting names with the SciPy functions. It is
  good practice to avoid direct imports.

"""

import itertools
import math
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy
import scipy.ndimage
import xarray as xr
from numba import carray, cfunc, types

from erlab.utils.array import (
    check_arg_2d_darr,
    check_arg_uniform_dims,
    is_uniform_spaced,
)


def _parse_dict_arg(
    dims: Sequence[Hashable],
    sigma: float | Collection[float] | Mapping[Hashable, float],
    arg_name: str,
    reference_name: str,
    allow_subset: bool = False,
) -> dict[Hashable, float]:
    if isinstance(sigma, Mapping):
        sigma_dict = dict(sigma)

    elif np.isscalar(sigma):
        sigma_dict = dict.fromkeys(dims, sigma)

    elif isinstance(sigma, Collection):
        if len(sigma) != len(dims):
            raise ValueError(
                f"`{arg_name}` does not match dimensions of {reference_name}"
            )

        sigma_dict = dict(zip(dims, sigma, strict=True))

    else:
        raise TypeError(f"`{arg_name}` must be a scalar, sequence, or mapping")

    if not allow_subset and len(sigma_dict) != len(dims):
        required_dims = set(dims) - set(sigma_dict.keys())
        raise ValueError(
            f"`{arg_name}` missing for the following dimension"
            f"{'' if len(required_dims) == 1 else 's'}: {required_dims}"
        )

    else:
        for d in sigma_dict.keys():
            if d not in dims:
                raise ValueError(
                    f"Dimension `{d}` in {arg_name} not found in {reference_name}"
                )

    # Make sure that sigma_dict is ordered in temrs of data dims
    sigma_dict = {d: sigma_dict[d] for d in dims if d in sigma_dict.keys()}

    return sigma_dict


def gaussian_filter(
    darr: xr.DataArray,
    sigma: float | Collection[float] | Mapping[Hashable, float],
    order: int | Sequence[int] | Mapping[Hashable, int] = 0,
    mode: str | Sequence[str] | Mapping[Hashable, str] = "nearest",
    cval: float = 0.0,
    truncate: float = 4.0,
    *,
    radius: None | float | Collection[float] | Mapping[Hashable, float] = None,
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
    sigma_dict: dict[Hashable, float] = _parse_dict_arg(
        darr.dims,
        sigma,
        arg_name="sigma",
        reference_name="DataArray",
        allow_subset=True,
    )

    # Get the axis indices to apply the filter
    axes = tuple(darr.get_axis_num(d) for d in sigma_dict.keys())

    # Convert arguments to tuples acceptable by scipy
    if isinstance(order, Mapping):
        order = tuple(order.get(str(d), 0) for d in sigma_dict.keys())

    if isinstance(mode, Mapping):
        mode = tuple(mode[str(d)] for d in sigma_dict.keys())

    if radius is not None:
        radius_dict = _parse_dict_arg(
            tuple(sigma_dict.keys()), radius, "radius", "`sigma`"
        )

        # Calculate radius in pixels
        radius_pix: tuple[int, ...] | None = tuple(
            round(r / (darr[d].values[1] - darr[d].values[0]))
            for d, r in radius_dict.items()
        )
    else:
        radius_pix = None

    for d in sigma_dict.keys():
        if not is_uniform_spaced(darr[d].values):
            raise ValueError(f"Dimension `{d}` is not uniformly spaced")

    # Calculate sigma in pixels
    sigma_pix: tuple[float, ...] = tuple(
        val / (darr[d].values[1] - darr[d].values[0]) for d, val in sigma_dict.items()
    )

    return darr.copy(
        data=scipy.ndimage.gaussian_filter(
            darr.values,
            sigma=sigma_pix,
            order=order,
            mode=mode,
            cval=cval,
            truncate=truncate,
            radius=radius_pix,
            axes=axes,
        )
    )


def gaussian_laplace(
    darr: xr.DataArray,
    sigma: float | Collection[float] | Mapping[Hashable, float],
    mode: str | Sequence[str] | Mapping[str, str] = "nearest",
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
    sigma_dict: dict[Hashable, float] = _parse_dict_arg(
        darr.dims,
        sigma,
        arg_name="sigma",
        reference_name="DataArray",
        allow_subset=False,
    )

    # Convert mode to tuple acceptable by scipy
    if isinstance(mode, dict):
        mode = tuple(mode[d] for d in sigma_dict.keys())

    # Calculate sigma in pixels
    sigma_pix: tuple[float, ...] = tuple(
        val / (darr[d].values[1] - darr[d].values[0]) for d, val in sigma_dict.items()
    )

    return darr.copy(
        data=scipy.ndimage.gaussian_laplace(
            darr.values, sigma=sigma_pix, mode=mode, cval=cval, **kwargs
        )
    )


def _ndpoly_degree_combinations(polyorder: int, ndim: int) -> list[tuple[int, ...]]:
    degrees = [range(polyorder + 1)] * ndim
    return [d for d in itertools.product(*degrees) if sum(d) <= polyorder]


def _ndsavgol_vandermonde(window_shape: tuple[int, ...], polyorder: int):
    """Calculate the Vandermonde matrix for Savitzky-Golay filtering."""
    # Get the number of dimensions
    ndim: int = len(window_shape)

    # Half of the window size
    half_sizes = np.array([[s // 2 for s in window_shape]], dtype=np.float64).T

    # Create an array of indices for each dimension
    indices = (
        np.indices(window_shape, dtype=np.float64).reshape((ndim, -1)) - half_sizes
    ).T

    # Create all combinations of degrees
    degree_combinations = np.array(
        _ndpoly_degree_combinations(polyorder, ndim), dtype=np.float64
    )

    # Create the Vandermonde matrix
    vander = np.prod(
        np.power(indices[:, None, :], degree_combinations[None, :, :]), axis=-1
    )
    return vander


def _ndsavgol_scale(deriv_idx: int, delta: tuple[float, ...], polyorder: int):
    """Calculate the scale factor for the Savitzky-Golay filter."""
    # Get the derivative order for each axis
    deriv_for_ax = _ndpoly_degree_combinations(polyorder, len(delta))[deriv_idx]

    # Calculate the correction factor for the derivative order and sample point spacing
    scale = math.factorial(sum(deriv_for_ax)) / sum(np.power(delta, deriv_for_ax))

    return scale


def _ndsavgol_coeffs(
    window_shape: tuple[int, ...],
    polyorder: int,
    deriv_idx: int,
    delta: tuple[float, ...],
):
    """Calculate the Savitzky-Golay filter coefficients."""
    vander = _ndsavgol_vandermonde(window_shape, polyorder)
    scale = _ndsavgol_scale(deriv_idx, delta, polyorder)

    # Invert the Vandermonde matrix to get the filter coefficients
    coeffs = np.linalg.pinv(vander)[deriv_idx] * scale

    # SciPy uses lstsq for this, but calculating the pseudo-inverse directly seems to
    # return more accurate results

    return coeffs


def ndsavgol(
    arr: npt.NDArray[np.float64],
    window_shape: int | tuple[int, ...],
    polyorder: int,
    deriv: int | tuple[int, ...] = 0,
    delta: float | tuple[float, ...] = 1.0,
    mode: Literal["mirror", "constant", "nearest", "wrap"] = "mirror",
    cval: float = 0.0,
    method: Literal["pinv", "lstsq"] = "pinv",
):
    """Apply a Savitzky-Golay filter to an N-dimensional array.

    Unlike `scipy.signal.savgol_filter` which is limited to 1D arrays, this function
    calculates multi-dimensional Savitzky-Golay filters. There are some subtle
    differences in the implementation, so the results may not be identical. See Notes.

    Parameters
    ----------
    arr
        The input N-dimensional array to be filtered. The array will be cast to float64
        before filtering.
    window_shape
        The shape of the window used for filtering. If an integer, the same size will be
        used across all axes.
    polyorder
        The order of the polynomial used to fit the samples. `polyorder` must be less
        than the minimum of `window_shape`.
    deriv
        The order of the derivative to compute given as a single integer or a tuple of
        integers. If an integer, the derivative of that order is computed along all
        axes. If a tuple of integers, the derivative of each order is computed along the
        corresponding dimension. The default is 0, which means to filter the data
        without differentiating.
    delta
        The spacing of the samples to which the filter will be applied. If a float, the
        same value is used for all axes. If a tuple, the values are used in the same
        order as in `deriv`. The default is 1.0.
    mode
        Must be 'mirror', 'constant', 'nearest', or 'wrap'. This determines the type of
        extension to use for the padded signal to which the filter is applied.  When
        `mode` is 'constant', the padding value is given by `cval`.
    cval
        Value to fill past the edges of the input if `mode` is 'constant'. Default is
        0.0.
    method
        Must be 'pinv' or 'lstsq'. Determines the method used to calculate the filter
        coefficients. 'pinv' uses the pseudoinverse of the Vandermonde matrix, while
        'lstsq' uses least squares for each window position. 'lstsq' is much slower but
        may be more numerically stable in some cases. The difference is more pronounced
        for higher dimensions, larger window size, and higher polynomial orders. The
        default is 'pinv'.

    Returns
    -------
    numpy.ndarray
        The filtered array.

    See Also
    --------
    :func:`scipy.signal.savgol_filter` : The 1D Savitzky-Golay filter function in SciPy.

    Notes
    -----
    - For even window sizes, the results may differ slightly from
      `scipy.signal.savgol_filter` due to differences in the implementation.
    - This function is not suitable for cases where accumulated floating point errors
      are comparable to the filter coefficients, i.e., for high number of dimensions and
      large window sizes.
    - ``mode='interp'`` is not implemented as it is not clear how to handle the edge
      cases in higher dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> import erlab.analysis as era

    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> era.image.ndsavgol(arr, (3,), polyorder=2)
    array([1., 2., 3., 4., 5.])

    >>> era.image.ndsavgol(arr, (3,), polyorder=2, deriv=1)
    array([0., 1., 1., 1., 0.])

    >>> arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> era.image.ndsavgol(arr, (3, 3), polyorder=2)
    array([[0.5, 1. , 1.5],
           [2. , 2.5, 3. ],
           [3.5, 4. , 4.5]])

    """
    if mode not in ["mirror", "constant", "nearest", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest', or 'wrap'")

    if method not in ["pinv", "lstsq"]:
        raise ValueError("method must be 'pinv' or 'lstsq'")

    if method == "lstsq":
        accurate = True
    else:
        accurate = False

    if isinstance(window_shape, int):
        window_shape = (window_shape,) * arr.ndim

    # Convert deriv to a tuple for 2D or higher arrays
    if isinstance(deriv, int) and arr.ndim > 1:
        deriv = (deriv,) * arr.ndim

    # Convert to an index of the list of combinations
    if not isinstance(deriv, int):
        if len(deriv) != arr.ndim:
            raise ValueError(
                "`deriv` must have the same length as the number of dimensions"
            )
        deriv_idx = _ndpoly_degree_combinations(polyorder, arr.ndim).index(tuple(deriv))
    else:
        # 1D case, the two are equivalent
        deriv_idx = deriv

    # Ensure delta is a tuple
    if isinstance(delta, int | float | np.floating):
        delta = (float(delta),) * arr.ndim

    if len(delta) != arr.ndim:
        raise ValueError(
            "`delta` must have the same length as the number of dimensions"
        )

    if accurate:
        vander = _ndsavgol_vandermonde(window_shape, polyorder)
        scale = _ndsavgol_scale(deriv_idx, delta, polyorder)

    else:
        # Invert the Vandermonde matrix to get the filter coefficients
        coeffs = _ndsavgol_coeffs(window_shape, polyorder, deriv_idx, delta)

    if arr.ndim == 1:
        # Cfunc definition overhead is a bottleneck for small arrays
        # Python function is faster for 1D arrays of reasonable size

        def _func(values):
            if accurate:
                out, _, _, _ = np.linalg.lstsq(vander, values, rcond=-1.0)
                return out[deriv_idx] * scale
            else:
                return np.dot(coeffs, values)

    else:

        @cfunc(
            types.intc(
                types.CPointer(types.float64),
                types.intp,
                types.CPointer(types.float64),
                types.voidptr,
            )
        )
        def _calc_savgol(values_ptr, len_values, result, data):
            values = carray(values_ptr, (len_values,), dtype=types.float64)

            if accurate:
                out, _, _, _ = np.linalg.lstsq(vander, values, rcond=-1.0)
                result[0] = out[deriv_idx] * scale
            else:
                result[0] = np.dot(coeffs, values)

            return 1

        _func = scipy.LowLevelCallable(
            _calc_savgol.ctypes, signature="int (double *, npy_intp, double *, void *)"
        )

    return scipy.ndimage.generic_filter(
        arr.astype(np.float64), _func, size=window_shape, mode=mode, cval=cval
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
    dist = np.array([[dxy, dy, dxy], [dx, 0.0, dx], [dxy, dy, dxy]]).flatten()

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


@check_arg_2d_darr
@check_arg_uniform_dims
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
    Any zero gradient values are replaced with NaN.
    """
    xvals = darr[darr.dims[1]].values
    yvals = darr[darr.dims[0]].values

    dx = abs(xvals[1] - xvals[0])
    dy = abs(yvals[1] - yvals[0])

    grad = gradient_magnitude(
        darr.values.astype(np.float64), dx, dy, mode=mode, cval=cval
    )
    grad[np.isclose(grad, 0.0)] = np.nan
    return darr / darr.max(skipna=True) / grad


@check_arg_2d_darr
@check_arg_uniform_dims
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

    See Also
    --------
    :func:`scipy.ndimage.generic_laplace` : The underlying function used to apply the
        filter.
    """
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


@check_arg_2d_darr
@check_arg_uniform_dims
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
    """
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
