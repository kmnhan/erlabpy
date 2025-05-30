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

__all__ = [
    "boxcar_filter",
    "curvature",
    "curvature1d",
    "diffn",
    "gaussian_filter",
    "gaussian_laplace",
    "gradient_magnitude",
    "laplace",
    "minimum_gradient",
    "ndsavgol",
    "remove_stripe",
    "scaled_laplace",
]

import itertools
import math
import typing
import warnings
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import scipy
import scipy.ndimage
import xarray as xr
from numba import carray, cfunc, types

import erlab

if typing.TYPE_CHECKING:
    import findiff

else:
    import lazy_loader as _lazy

    findiff = _lazy.load("findiff")


def _parse_dict_arg(
    dims: Sequence[Hashable],
    arg_value: float | Collection[float] | Mapping[Hashable, float],
    arg_name: str,
    reference_name: str,
    allow_subset: bool = False,
) -> dict[Hashable, float]:
    """Parse the input argument to a dictionary with dimensions as keys."""
    if isinstance(arg_value, Mapping):
        arg_dict = dict(arg_value)

    elif np.isscalar(arg_value):
        arg_dict = dict.fromkeys(dims, arg_value)

    elif isinstance(arg_value, Collection):
        if len(arg_value) != len(dims):
            raise ValueError(
                f"`{arg_name}` does not match dimensions of {reference_name}"
            )

        arg_dict = dict(zip(dims, arg_value, strict=True))

    else:
        raise TypeError(f"`{arg_name}` must be a scalar, sequence, or mapping")

    if not allow_subset and len(arg_dict) != len(dims):
        required_dims = set(dims) - set(arg_dict.keys())
        raise ValueError(
            f"`{arg_name}` missing for the following dimension"
            f"{'' if len(required_dims) == 1 else 's'}: {required_dims}"
        )

    for d in arg_dict:
        if d not in dims:
            raise ValueError(
                f"Dimension `{d}` in {arg_name} not found in {reference_name}"
            )

    # Make sure that sigma_dict is ordered in temrs of data dims
    return {d: arg_dict[d] for d in dims if d in arg_dict}


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
    darr : DataArray
        The input DataArray.
    sigma : float or Sequence of floats or dict
        The standard deviation(s) of the Gaussian filter in data dimensions. If a float,
        the same value is used for all dimensions, each scaled by the data step. If a
        dict, the value can be specified for each dimension using dimension names as
        keys. The filter is only applied to the dimensions specified in the dict. If a
        sequence, the values are used in the same order as the dimensions of the
        DataArray.
    order : int or Sequence of ints or dict
        The order of the filter along each dimension. If an int, the same order is used
        for all dimensions. See Notes below for other options. Defaults to 0.
    mode : str or Sequence of str or dict
        The boundary mode used for the filter. If a str, the same mode is used for all
        dimensions. See Notes below for other options. Defaults to 'nearest'.
    cval
        Value to fill past edges of input if mode is 'constant'. Defaults to 0.0.
    truncate
        The truncation value used for the Gaussian filter. Defaults to 4.0.
    radius : float or Sequence of floats or dict, optional
        The radius of the Gaussian filter in data units. See Notes below. If specified,
        the size of the kernel along each axis will be ``2*radius + 1``, and `truncate`
        is ignored.

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
    axes = tuple(darr.get_axis_num(d) for d in sigma_dict)

    # Convert arguments to tuples acceptable by scipy
    if isinstance(order, Mapping):
        order = tuple(order.get(str(d), 0) for d in sigma_dict)

    if isinstance(mode, Mapping):
        mode = tuple(mode[str(d)] for d in sigma_dict)

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

    for d in sigma_dict:
        if not erlab.utils.array.is_uniform_spaced(darr[d].values):
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


@cfunc(
    types.intc(
        types.CPointer(types.float64),
        types.intp,
        types.CPointer(types.float64),
        types.voidptr,
    )
)
def _boxcar_kernel_nb(values_ptr, len_values, result, data) -> int:
    values = carray(values_ptr, (len_values,), dtype=types.float64)
    result[0] = np.mean(values)
    return 1


# https://github.com/jni/llc-tools/issues/3#issuecomment-757134814
_boxcar_func = scipy.LowLevelCallable(
    _boxcar_kernel_nb.ctypes, signature="int (double *, npy_intp, double *, void *)"
)


def boxcar_filter(
    darr: xr.DataArray,
    size: int | Collection[int] | Mapping[Hashable, int],
    mode: str = "nearest",
    cval: float = 0.0,
) -> xr.DataArray:
    """Coordinate-aware boxcar filter.

    Parameters
    ----------
    darr : DataArray
        The input DataArray.
    size : int or Sequence of ints or dict
        The size of the boxcar filter in pixels.
    mode : str
        The boundary mode used for the filter. Defaults to 'nearest'.
    cval
        Value to fill past edges of input if mode is 'constant'. Defaults to 0.0.

    Returns
    -------
    boxcar_filter : xarray.DataArray
        The filtered array with the same shape as the input DataArray.

    """
    size_dict: dict[Hashable, float] = _parse_dict_arg(
        darr.dims, size, arg_name="size", reference_name="DataArray", allow_subset=True
    )

    size_pix: list[int] = []
    for d in darr.dims:
        if d in size_dict:
            size_pix.append(int(size_dict[d]))
        else:
            size_pix.append(1)

    return darr.copy(
        data=scipy.ndimage.generic_filter(
            darr.values.astype(np.float64),
            _boxcar_func,
            size=tuple(size_pix),
            mode=mode,
            cval=cval,
        ).astype(darr.dtype)
    )


def gaussian_laplace(
    darr: xr.DataArray,
    sigma: float | Collection[float] | Mapping[Hashable, float],
    mode: str | Sequence[str] | Mapping[Hashable, str] = "nearest",
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
    if isinstance(mode, Mapping):
        mode = tuple(mode[d] for d in sigma_dict)

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
    return np.prod(
        np.power(indices[:, None, :], degree_combinations[None, :, :]), axis=-1
    )


def _ndsavgol_scale(deriv_idx: int, delta: tuple[float, ...], polyorder: int):
    """Calculate the scale factor for the Savitzky-Golay filter."""
    # Get the derivative order for each axis
    deriv_for_ax = _ndpoly_degree_combinations(polyorder, len(delta))[deriv_idx]

    # Calculate the correction factor for the derivative order and sample point spacing
    return math.factorial(sum(deriv_for_ax)) / sum(np.power(delta, deriv_for_ax))


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
    return np.linalg.pinv(vander)[deriv_idx] * scale

    # SciPy uses lstsq for this, but calculating the pseudo-inverse directly seems to
    # return more accurate results


def ndsavgol(
    arr: npt.NDArray[np.float64],
    window_shape: int | tuple[int, ...],
    polyorder: int,
    deriv: int | tuple[int, ...] = 0,
    delta: float | tuple[float, ...] = 1.0,
    mode: typing.Literal["mirror", "constant", "nearest", "wrap"] = "mirror",
    cval: float = 0.0,
    method: typing.Literal["pinv", "lstsq"] = "pinv",
):
    """Apply a Savitzky-Golay filter to an N-dimensional array.

    Unlike `scipy.signal.savgol_filter` which is limited to 1D arrays, this function
    calculates multi-dimensional Savitzky-Golay filters. There are some subtle
    differences in the implementation, so the results may not be identical. See Notes.

    Parameters
    ----------
    arr : array-like
        The input N-dimensional array to be filtered. The array will be cast to float64
        before filtering.
    window_shape : int or tuple of ints
        The shape of the window used for filtering. If an integer, the same size will be
        used across all axes.
    polyorder : int
        The order of the polynomial used to fit the samples. `polyorder` must be less
        than the minimum of `window_shape`.
    deriv : int or tuple of ints
        The order of the derivative to compute given as a single integer or a tuple of
        integers. If an integer, the derivative of that order is computed along all
        axes. If a tuple of integers, the derivative of each order is computed along the
        corresponding dimension. The default is 0, which means to filter the data
        without differentiating.
    delta : float or tuple of floats
        The spacing of the samples to which the filter will be applied. If a float, the
        same value is used for all axes. If a tuple, the values are used in the same
        order as in `deriv`. The default is 1.0.
    mode
        Must be 'mirror', 'constant', 'nearest', or 'wrap'. This determines the type of
        extension to use for the padded signal to which the filter is applied.  When
        `mode` is 'constant', the padding value is given by `cval`.
    cval : float
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

    accurate = method == "lstsq"

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
        def _calc_savgol(values_ptr, len_values, result, data) -> int:
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
    arr: npt.NDArray[np.float64],
    dx: float,
    dy: float,
    mode: str = "nearest",
    cval: float = 0.0,
) -> npt.NDArray[np.float64]:
    r"""Calculate the gradient magnitude of an image.

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
    This function calculates the gradient magnitude of an image by applying a filter
    that uses the given dx and dy values. The filter is defined by a kernel function
    that computes the squared difference between each element of the input array and the
    central element, divided by the corresponding distance value. The gradient magnitude
    is then calculated as the square root of the sum of the squared differences.
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
    def _kernel(values_ptr, len_values, result, data) -> int:
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

    return scipy.ndimage.generic_filter(arr, func, size=(3, 3), mode=mode, cval=cval)


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
    if isinstance(mode, Mapping):
        mode = tuple(mode[d] for d in darr.dims)
    return darr.copy(data=scipy.ndimage.laplace(darr.values, mode=mode, cval=cval))


@typing.overload
def diffn(
    darr: xr.DataArray, coord: Hashable, order: Iterable[int] = ..., **kwargs
) -> tuple[xr.DataArray, ...]: ...


@typing.overload
def diffn(
    darr: xr.DataArray, coord: Hashable, order: int = 1, **kwargs
) -> xr.DataArray: ...


def diffn(
    darr: xr.DataArray, coord: Hashable, order: int | Iterable[int] = 1, **kwargs
) -> xr.DataArray | tuple[xr.DataArray, ...]:
    """Calculate the nth derivative of a DataArray along a given dimension.

    Parameters
    ----------
    darr
        The input DataArray.
    coord
        The name of the coordinate along which to calculate the derivative.
    order
        The order of the derivative. If given as a tuple, a tuple of derivatives for
        each order is returned. Default is 1.
    **kwargs
        Additional keyword arguments to :class:`findiff.Diff`.

    Returns
    -------
    DataArray or tuple of DataArray
        The differentiated array or a tuple of differentiated arrays corresponding to
        the provided order.
    """
    xvals = darr[coord].values.astype(np.float64)
    grid = (
        (xvals[1] - xvals[0]) if erlab.utils.array.is_uniform_spaced(xvals) else xvals
    )
    d_dx = findiff.Diff(darr.get_axis_num(coord), grid=grid, **kwargs)

    if not isinstance(order, int) and isinstance(order, Iterable):
        return tuple(_apply_diffn(darr, d_dx, o) for o in order)
    return _apply_diffn(darr, d_dx, order)


def _apply_diffn(
    darr: xr.DataArray, operator: findiff.Diff, order: int
) -> xr.DataArray:
    return darr.copy(data=(operator**order)(darr.values.astype(np.float64)))


@erlab.utils.array.check_arg_2d_darr
@erlab.utils.array.check_arg_uniform_dims
@erlab.utils.array.check_arg_has_no_nans
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


@erlab.utils.array.check_arg_2d_darr
@erlab.utils.array.check_arg_uniform_dims
@erlab.utils.array.check_arg_has_no_nans
def scaled_laplace(darr, factor: float = 1.0, **kwargs) -> xr.DataArray:
    r"""Calculate the Laplacian of a 2D DataArray with different scaling for each axis.

    This function calculates the Laplacian of the given array using approximate second
    derivatives, taking the different scaling for each axis into account.

    .. math::

        \Delta f \sim \frac{\partial^2 f}{\partial x^2}
        \left(\frac{\Delta x}{\Delta y}\right)^{\!2} + \frac{\partial^2 f}{\partial y^2}

    See Ref. :cite:p:`zhang2011curvature` for more information.

    Parameters
    ----------
    darr
        The 2D DataArray for which to calculate the scaled Laplacian.
    factor
        The factor by which to scale the x-axis derivative. Negative values will scale
        the y-axis derivative instead. Default is 1.0.
    **kwargs
        Additional keyword arguments to :class:`findiff.Diff`.

    Returns
    -------
    scaled_laplace : xarray.DataArray
        The filtered array with the same shape as the input DataArray.

    """
    for _deprecated_kw in ("mode", "cval"):
        if _deprecated_kw in kwargs:
            warnings.warn(
                f"Keyword argument '{_deprecated_kw}' for scaled_laplace is not used, "
                "and will be removed in a future version.",
                FutureWarning,
                stacklevel=1,
            )
            kwargs.pop(_deprecated_kw)
    xvals = darr[darr.dims[1]].values.astype(np.float64)
    yvals = darr[darr.dims[0]].values.astype(np.float64)

    dx, dy = xvals[1] - xvals[0], yvals[1] - yvals[0]
    weight = (dx / dy) ** 2

    if factor > 0:
        weight *= factor
    elif factor < 0:
        weight /= abs(factor)

    d_dx, d_dy = findiff.Diff(1, dx, **kwargs), findiff.Diff(0, dy, **kwargs)
    scaled_lapl_operator = weight * d_dx**2 + d_dy**2
    return darr.copy(data=scaled_lapl_operator(darr.values.astype(np.float64)))


@erlab.utils.array.check_arg_uniform_dims
@erlab.utils.array.check_arg_has_no_nans
def curvature(
    darr: xr.DataArray, a0: float = 1.0, factor: float = 1.0, **kwargs
) -> xr.DataArray:
    """2D curvature method for detecting dispersive features.

    The curvature is calculated as defined by :cite:t:`zhang2011curvature`.

    Parameters
    ----------
    darr
        The DataArray for which to calculate the curvature. The curvature is calculated
        along the first two dimensions of the DataArray.
    a0
        The regularization constant. Reasonable values range from 0.001 to 10, but
        different values may be needed depending on the data. Default is 1.0.
    factor
        The factor by which to scale the x-axis curvature. Negative values will scale
        the y-axis curvature instead. Default is 1.0.
    **kwargs
        Additional keyword arguments to :class:`findiff.Diff`.

    Returns
    -------
    curvature : xarray.DataArray
        The 2D curvature of the input DataArray. Has the same shape as :code:`input`.

    Raises
    ------
    ValueError
        If the input DataArray is not 2D.
    """
    xvals = darr[darr.dims[1]].values.astype(np.float64)
    yvals = darr[darr.dims[0]].values.astype(np.float64)

    dx, dy = xvals[1] - xvals[0], yvals[1] - yvals[0]
    weight = (dx / dy) ** 2

    if factor > 0:
        weight *= factor
    elif factor < 0:
        weight /= abs(factor)

    d_dx, d_dy = findiff.Diff(1, dx, **kwargs), findiff.Diff(0, dy, **kwargs)

    values = darr.values.astype(np.float64)
    dfdx = d_dx(values)
    dfdy = d_dy(values)
    d2fdx2 = (d_dx**2)(values)
    d2fdydx = (d_dy * d_dx)(values)
    d2fdy2 = (d_dy**2)(values)
    del values

    max_abs_dfdx_sq = np.max(np.abs(dfdx)) ** 2
    max_abs_dfdy_sq = np.max(np.abs(dfdy)) ** 2

    c0 = a0 * max(max_abs_dfdy_sq, weight * max_abs_dfdx_sq)

    curv = (
        (c0 + weight * dfdx**2) * d2fdy2
        - 2 * weight * dfdx * dfdy * d2fdydx
        + weight * (c0 + dfdy**2) * d2fdx2
    ) / (c0 + weight * dfdx**2 + dfdy**2) ** 1.5

    return darr.copy(data=curv)


def curvature1d(
    darr: xr.DataArray, along: Hashable, a0: float = 1.0, **kwargs
) -> xr.DataArray:
    """1D curvature method for detecting dispersive features.

    The curvature is calculated as defined by :cite:t:`zhang2011curvature`.

    Parameters
    ----------
    darr
        The DataArray for which to calculate the curvature.
    along
        The dimension along which to calculate the curvature.
    a0
        The regularization constant. Reasonable values range from 0.001 to 10, but
        different values may be needed depending on the data. Default is 1.0.
    **kwargs
        Additional keyword arguments to :class:`findiff.Diff`.

    Returns
    -------
    curvature : xarray.DataArray
        The 1D curvature of the input DataArray. Has the same shape as :code:`input`.

    Raises
    ------
    ValueError
        If the input DataArray is not 2D.
    """
    dfdx, d2fdx2 = diffn(darr, along, order=(1, 2), **kwargs)
    curv = d2fdx2 / (a0 + dfdx**2 / (np.abs(dfdx).max() ** 2)) ** 1.5
    return darr.copy(data=curv)


@erlab.utils.array.check_arg_has(dims=("alpha", "eV"))
def remove_stripe(
    darr: xr.DataArray, deg: int, full: bool = False, **sel_kw
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    r"""Remove angle-dependent stripe artifact from cuts and maps.

    Energy-independent stripe artifacts may be introduced during the acquisition of
    ARPES data due to imperfect alignment of the slit or other experimental factors.

    Assume an original intensity :math:`I_0(\alpha, \omega)` that is corrupted by a
    energy-independent stripe pattern :math:`S(\alpha)`:

    .. math::

        I(\alpha, \omega) = I_0(\alpha, \omega) \cdot S(\alpha).

    If we assume that :math:`S(\alpha)` to be a high-frequency noise, we may approximate
    :math:`I_0(\alpha, \omega)` by smoothing :math:`I(\alpha, \omega)` along
    :math:`\alpha`. We can then obtain an approximation of :math:`1/S(\alpha)` by
    dividing the smoothed data with :math:`I(\alpha, \omega)` and averaging the result
    over :math:`\omega`. Finally, we can remove the stripe pattern by multiplying
    :math:`I(\alpha, \omega)` with the obtained :math:`1/S(\alpha)`.

    Works best for data with a high signal-to-noise ratio and a high background level.
    Since the stripe is assumed to be energy-independent, the method is only suitable
    for data acquired with sweep mode.

    This method may introduce artifacts that are not present in the original data,
    making it unsuitable for quantitative analysis. Use only for visualization purposes.

    Parameters
    ----------
    darr
        The data containing the stripe artifact. Data must have the dimensions "alpha"
        and "eV".
    deg
        The degree of the polynomial fit. The degree should be high enough to capture
        all intrinsic features of the data, but low enough to avoid overfitting. A good
        starting value is around 20.
    full
        Flag determining whether to return the full stripe pattern. If True,
        :math:`1/S(\alpha)` is also returned. Default is False.
    **sel_kw
        Keyword arguments to :meth:`xarray.DataArray.sel`. Specify the range of angles
        and energies to use for the polynomial fit.

    Returns
    -------
    corrected : xarray.DataArray
        The data with the stripe artifact removed.
    stripe : xarray.DataArray
        The stripe pattern :math:`1/S(\alpha)`. Only returned if `full` is True.

    """
    cropped = darr.sel(**sel_kw)

    poly_fit = xr.polyval(
        cropped["alpha"], cropped.polyfit("alpha", deg).polyfit_coefficients
    )

    with xr.set_options(keep_attrs=True):
        stripe = xr.ones_like(darr).isel(eV=0).squeeze()
        sel_kw_copy = sel_kw.copy()
        sel_kw_copy.pop("eV", None)
        stripe.loc[sel_kw_copy] = (poly_fit / cropped).mean("eV", skipna=True)
        corrected = darr * stripe

        if full:
            return corrected, stripe
        return corrected
