"""Transformations."""

__all__ = ["rotate", "shift", "rotateinplane", "rotatestackinplane"]

import itertools
import warnings
from collections.abc import Hashable, Iterable, Mapping
from typing import cast

import numpy as np
import scipy.ndimage
import scipy.special
import xarray as xr

from erlab.utils.array import trim_na, uniform_dims


def rotate(
    darr: xr.DataArray,
    angle: float,
    axes: tuple[int, int] | tuple[Hashable, Hashable] = (0, 1),
    center: tuple[float, float] | Mapping[Hashable, float] = (0.0, 0.0),
    reshape: bool = True,
    order: int = 1,
    mode="constant",
    cval=np.nan,
    prefilter: bool = True,
):
    """Rotate an array in the plane defined by the two axes.

    Parameters
    ----------
    darr
        The array to rotate.
    angle
        The rotation angle in degrees.
    axes : tuple of 2 ints or 2 strings, optional
        The two axes that define the plane of rotation. Default is the first two axes.
        If strings are provided, they must be valid dimension names in the input array.
    center : tuple of 2 floats or dict, optional
        The center of rotation in data coordinates. If a tuple, it is given as values
        along the dimensions specified in `axes`. If a dict, it must have keys that
        correspond to `axes`. Default is (0, 0).
    reshape
        If `True`, the output shape is adapted so that the input array is contained
        completely in the output. Default is `True`.
    order
        The order of the spline interpolation, default is 1. The order has to be in the
        range 0-5.
    mode, cval, prefilter
        Passed to :func:`scipy.ndimage.affine_transform`. See the scipy documentation
        for more information.

    Returns
    -------
    darr : xarray.DataArray
        The rotated array.

    See Also
    --------
    scipy.ndimage.affine_transform
        The function that performs the affine transformation on the input array.
    scipy.ndimage.rotate
        Similar function that rotates a numpy array.

    """
    input_arr = darr.values
    ndim = input_arr.ndim

    if ndim < 2:
        raise ValueError("input array should be at least 2D")

    if isinstance(axes[0], int):
        axes_dims: list[Hashable] = [darr.dims[a] for a in cast(tuple[int, int], axes)]
    else:
        axes_dims = list(axes)
    axes = list(darr.get_axis_num(axes_dims))

    if isinstance(center, Mapping):
        if set(center.keys()) != set(axes_dims):
            raise ValueError("center must have keys that match axes")
        centers = [center[dim] for dim in axes_dims]
    else:
        centers = list(center)

    if not uniform_dims(darr).issuperset(axes_dims):
        raise ValueError("all coordinates along axes must be evenly spaced")

    # Sort with respect to axis index
    axes, axes_dims, centers = map(
        list, zip(*sorted(zip(axes, axes_dims, centers, strict=True)), strict=True)
    )

    # Get pixel sizes
    ycoords, xcoords = darr[axes_dims[0]].values, darr[axes_dims[1]].values
    dy, dx = ycoords[1] - ycoords[0], xcoords[1] - xcoords[0]
    pixel_ratio = np.abs(dy / dx)  # pixel aspect ratio

    # Center in coordinate space
    center_y, center_x = centers

    # Rotation center in pixel space before transformation
    in_pixel_center = np.array(
        [
            (center_y - ycoords[0]) / dy,
            (center_x - xcoords[0]) / dx,
            1.0,
        ]
    )

    # Build rotation matrix
    c, s = scipy.special.cosdg(angle), scipy.special.sindg(angle)
    matrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    # Since rotation is applied in pixel space, scale for non-square pixels
    scale_matrix = np.diag([1.0 / pixel_ratio, 1.0, 1.0])
    scale_matrix_inverse = np.diag([pixel_ratio, 1.0, 1.0])
    matrix = scale_matrix @ matrix @ scale_matrix_inverse

    img_shape = np.asarray(input_arr.shape)
    in_plane_shape = img_shape[axes]

    if reshape:
        iy, ix = in_plane_shape
        out_bounds = matrix @ [[0, 0, iy, iy], [0, ix, 0, ix], [1, 1, 1, 1]]
        out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)

        out_center = (matrix @ ((out_plane_shape - 1) / 2))[:2]
        out_plane_shape = out_plane_shape[:2]
        in_center = (in_plane_shape - 1) / 2

    else:
        out_plane_shape = img_shape[axes]
        out_center = (matrix @ in_pixel_center)[:2]
        in_center = in_pixel_center[:2]

    offset = in_center - out_center

    # Build translation matrix
    translation_matrix = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]])
    matrix = translation_matrix @ matrix

    output_shape = img_shape
    output_shape[axes] = out_plane_shape
    output_shape = tuple(output_shape)

    if np.iscomplexobj(input_arr):
        output = np.zeros(
            output_shape, dtype=np.promote_types(input_arr.dtype, np.complex64)
        )
    else:
        output = np.zeros(output_shape, dtype=input_arr.dtype.name)

    if ndim <= 2:
        scipy.ndimage.affine_transform(
            input_arr, matrix, 0.0, output_shape, output, order, mode, cval, prefilter
        )
    else:
        # If ndim > 2, the rotation is applied over all the planes parallel to axes
        planes_coord = itertools.product(
            *[
                [slice(None)] if ax in axes else range(img_shape[ax])
                for ax in range(ndim)
            ]
        )

        for coordinates in cast(Iterable[tuple[slice | int, ...]], planes_coord):
            ia = input_arr[coordinates]
            oa = output[coordinates]
            scipy.ndimage.affine_transform(
                ia,
                matrix,
                0.0,
                tuple(out_plane_shape),
                oa,
                order,
                mode,
                cval,
                prefilter,
            )

    shape_diff = out_plane_shape - in_plane_shape

    # Coords associated with rotated axes are meaningless after rotation
    for dim, coord in dict(darr.coords).items():
        if dim in axes_dims:
            continue
        for ax in axes_dims:
            if ax in coord.dims:
                darr = darr.copy().drop_vars((dim,))
                break

    if reshape:
        # Adjust DataArray shape to match the output shape
        for i, diff in zip(axes, shape_diff, strict=True):
            if diff < 0:
                darr = darr.isel({axes_dims[i]: slice(None, diff)})
            elif diff > 0:
                darr = darr.copy(deep=True).pad({axes_dims[i]: (0, diff)})

        # Rotation center in pixel space after transformation
        out_pixel_center = np.linalg.lstsq(matrix, in_pixel_center, rcond=None)[0][:2]

        start0 = -out_pixel_center[0] * dy + center_y
        end0 = start0 + (out_plane_shape[0] - 1) * dy

        start1 = -out_pixel_center[1] * dx + center_x
        end1 = start1 + (out_plane_shape[1] - 1) * dx

        darr = darr.assign_coords(
            {
                axes_dims[0]: np.linspace(start0, end0, out_plane_shape[0]),
                axes_dims[1]: np.linspace(start1, end1, out_plane_shape[1]),
            }
        )

    darr = darr.copy(data=output)

    if reshape:
        darr = trim_na(darr, axes_dims)

    return darr


def shift(
    darr: xr.DataArray,
    shift: float | xr.DataArray,
    along: str,
    shift_coords: bool = False,
    **shift_kwargs,
) -> xr.DataArray:
    """Shifts the values of a DataArray along a single dimension.

    The shift is applied using `scipy.ndimage.shift` which uses spline interpolation. By
    default, the spline is of order 1 (linear interpolation).

    Parameters
    ----------
    darr
        The array to shift.
    shift
        The amount of shift to be applied along the specified dimension. If
        :code:`shift` is a DataArray, different shifts can be applied to different
        coordinates. The dimensions of :code:`shift` must be a subset of the dimensions
        of `darr`. For more information, see the note below. If :code:`shift` is a
        `float`, the same shift is applied to all values along dimension `along`. This
        is equivalent to providing a 0-dimensional DataArray.
    along
        Name of the dimension along which the shift is applied.
    shift_coords
        If `True`, the coordinates of the output data will be changed so that the output
        contains all the values of the original data. If `False`, the coordinates and
        shape of the original data will be retained, and only the data will be shifted.
        Defaults to `False`.
    **shift_kwargs
        Additional keyword arguments passed onto `scipy.ndimage.shift`. Default values
        of `cval` and `order` are set to `np.nan` and `1` respectively.

    Returns
    -------
    xarray.DataArray
        The shifted DataArray.

    Note
    ----
    - All dimensions in :code:`shift` must be a dimension in `darr`.
    - The :code:`shift` array values are divided by the step size along the `along`
      dimension.
    - NaN values in :code:`shift` are treated as zero.

    Example
    -------

    >>> import xarray as xr
    >>> import erlab.analysis as era
    >>> darr = xr.DataArray(
    ...     np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(float), dims=["x", "y"]
    ... )
    >>> shift_arr = xr.DataArray([1, 0, 2], dims=["x"])
    >>> shifted = era.transform.shift(darr, shift_arr, along="y")
    >>> print(shifted)
    <xarray.DataArray (x: 3, y: 3)> Size: 72B
    nan 1.0 2.0 4.0 5.0 6.0 nan nan 7.0
    Dimensions without coordinates: x, y
    """
    shift_kwargs.setdefault("order", 1)
    shift_kwargs.setdefault("mode", "constant")
    if shift_kwargs["mode"] == "constant":
        shift_kwargs.setdefault("cval", np.nan)

    if not isinstance(shift, xr.DataArray):
        shift = xr.DataArray(float(shift))

    for dim in shift.dims:
        if dim not in darr.dims:
            raise ValueError(f"Dimension {dim} in shift array not found in input array")
        if darr[dim].size != shift[dim].size:
            raise ValueError(
                f"Dimension {dim} in shift array has different size than input array"
            )

    domain_indices: tuple[int, ...] = darr.get_axis_num(shift.dims)

    # `along` must be evenly spaced and monotonic increasing
    out = darr.sortby(along).copy()

    # Normalize shift values
    along_step: float = out[along].values[1] - out[along].values[0]
    shift = (shift.copy() / along_step).fillna(0.0)

    if shift_coords:
        # We first apply the integer part of the average shift to the coords
        rigid_shift: float = float(np.round(shift.values.mean()))
        shift = shift - rigid_shift

        # Apply coordinate shift
        out = out.assign_coords({along: out[along].values + rigid_shift * along_step})

        # The bounds of the remaining shift values are used to pad the data
        nshift_min, nshift_max = shift.values.min(), shift.values.max()
        pads: tuple[int, int] = min(0, round(nshift_min)), max(0, round(nshift_max))

        # Construct new coordinate array
        new_along = np.linspace(
            out[along].values[0] + pads[0] * along_step,
            out[along].values[-1] + pads[1] * along_step,
            out[along].size + sum(np.abs(pads)),
        )

        # Pad the data and assign new coordinates
        out = out.pad(
            {along: tuple(np.abs(pads))}, mode="constant", constant_values=np.nan
        )
        out = out.assign_coords({along: new_along})

    for idxs in itertools.product(*[range(darr.shape[i]) for i in domain_indices]):
        # Construct slices for indexing
        _slices: list[slice | int] = [slice(None)] * darr.ndim
        for domain_index, i in zip(domain_indices, idxs, strict=True):
            _slices[domain_index] = i

        slices: tuple[slice | int, ...] = tuple(_slices)

        # Initialize arguments to `scipy.ndimage.shift`
        arr = out[slices]
        shifts: list[float] = [0.0] * arr.ndim
        shift_val: float = float(shift.isel(dict(zip(shift.dims, idxs, strict=True))))
        shifts[cast(int, arr.get_axis_num(along))] = shift_val

        # Apply shift
        out[slices] = scipy.ndimage.shift(arr.values, shifts, **shift_kwargs)

    return out


def rotateinplane(data: xr.DataArray, rotate, **interp_kwargs):
    """Rotate a 2D DataArray in the plane defined by the two dimensions.

    .. deprecated:: 2.9.0

        Use :func:`erlab.analysis.transform.rotate` instead.
    """
    warnings.warn(
        "erlab.analysis.transform.rotateinplane is deprecated, "
        "use erlab.analysis.transform.rotate instead",
        DeprecationWarning,
        stacklevel=1,
    )
    interp_kwargs.setdefault("method", "linearfast")

    theta = np.radians(rotate)
    d0, d1 = data.dims
    x = xr.DataArray(data[d0] * np.cos(theta) - data[d1] * np.sin(theta))
    y = xr.DataArray(data[d0] * np.sin(theta) + data[d1] * np.cos(theta))
    return data.interp({d0: x, d1: y}, **interp_kwargs)


def rotatestackinplane(data: xr.DataArray, rotate, **interp_kwargs):
    """Rotate a 3D DataArray in the plane defined by the two dimensions.

    .. deprecated:: 2.9.0

        Use :func:`erlab.analysis.transform.rotate` instead.

    """
    warnings.warn(
        "erlab.analysis.transform.rotateinplane is deprecated, "
        "use erlab.analysis.transform.rotate instead",
        DeprecationWarning,
        stacklevel=1,
    )
    interp_kwargs.setdefault("method", "linearfast")

    theta = np.radians(rotate)
    d0, d1, _ = data.dims
    x = xr.DataArray(data[d0] * np.cos(theta) - data[d1] * np.sin(theta))
    y = xr.DataArray(data[d0] * np.sin(theta) + data[d1] * np.cos(theta))
    return data.interp({d0: x, d1: y}, **interp_kwargs)
