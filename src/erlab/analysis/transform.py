"""Transformations."""

__all__ = ["rotate"]

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
    axes : tuple of 2 ints or strings, optional
        The two axes that define the plane of rotation. Default is the first two axes.
        If strings are provided, they must be valid dimension names in the input array.
    center : tuple of 2 floats or dict, optional
        The center of rotation in data coordinates. If a tuple, it is given as values
        along the dimensions specified in `axes`. If a dict, it must have keys that
        correspond to `axes`. Default is (0, 0).
    reshape
        If `reshape` is true, the output shape is adapted so that the input array is
        contained completely in the output. Default is True.
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
        [(center_y - ycoords[0]) / dy, (center_x - xcoords[0]) / dx, 1.0]
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


def rotateinplane(data: xr.DataArray, rotate, **interp_kwargs):
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
