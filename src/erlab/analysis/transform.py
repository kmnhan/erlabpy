"""Transformations."""

__all__ = ["rotate", "rotateinplane", "rotatestackinplane", "shift", "symmetrize"]

import typing
import warnings
from collections.abc import Hashable, Mapping

import numpy as np
import scipy
import xarray as xr

import erlab

if typing.TYPE_CHECKING:
    import scipy.ndimage
    import scipy.special  # noqa: TC004


def rotate(
    darr: xr.DataArray,
    angle: float,
    axes: tuple[int, int] | tuple[Hashable, Hashable] = (0, 1),
    center: tuple[float, float] | Mapping[Hashable, float] = (0.0, 0.0),
    *,
    reshape: bool = True,
    order: int = 1,
    mode: str = "constant",
    cval: float = np.nan,
    prefilter: bool = True,
) -> xr.DataArray:
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
    # Resolve axes to dimension names and indices
    if isinstance(axes[0], int):
        axes_dims: list[Hashable] = [
            darr.dims[a] for a in typing.cast("tuple[int, ...]", axes)
        ]
    else:
        axes_dims = list(axes)

    if len(axes_dims) != 2:
        raise ValueError("Exactly two axes must be specified for rotation")

    if not erlab.utils.array.uniform_dims(darr).issuperset(axes_dims):
        raise ValueError("all coordinates along axes must be evenly spaced")

    ax_idx = list(darr.get_axis_num(axes_dims))

    # Sort axes by index
    ax_idx, axes_dims = map(
        list, zip(*sorted(zip(ax_idx, axes_dims, strict=True)), strict=True)
    )

    ydim, xdim = axes_dims
    ycoords = darr[ydim].values
    xcoords = darr[xdim].values

    if ycoords.size < 2 or xcoords.size < 2:
        raise ValueError("axes must have at least 2 points each")

    # Get pixel sizes
    dy = ycoords[1] - ycoords[0]
    dx = xcoords[1] - xcoords[0]
    pixel_ratio = float(abs(dy / dx))

    # Interpret center in data coords
    if isinstance(center, Mapping):
        if set(center.keys()) != {ydim, xdim}:
            raise ValueError("center must have keys matching the two rotation axes")
        center_y = float(center[ydim])
        center_x = float(center[xdim])
    else:
        center_y, center_x = center

    # Build affine matrix in (y, x) pixel space
    c, s = scipy.special.cosdg(angle), scipy.special.sindg(angle)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    # Scale for non-square pixels
    scale = np.diag([1.0 / pixel_ratio, 1.0, 1.0])
    scale_inv = np.diag([pixel_ratio, 1.0, 1.0])
    matrix = scale @ rot @ scale_inv

    # Input shape and rotation plane shape
    in_shape = np.array(darr.shape)
    in_plane_shape = in_shape[ax_idx]  # (ny, nx)

    # Center in pixel space (input)
    in_pixel_center = np.array(
        [
            (center_y - ycoords[0]) / dy,
            (center_x - xcoords[0]) / dx,
            1.0,
        ]
    )

    if reshape:
        # Compute bounding box of rotated input to get output plane shape
        iy, ix = in_plane_shape
        corners = np.array([[0, 0, iy, iy], [0, ix, 0, ix], [1, 1, 1, 1]])
        out_bounds = matrix @ corners
        out_plane_shape = (np.ptp(out_bounds, axis=1) + 0.5).astype(int)

        # We want output centered so that original center maps to output center
        out_center = (matrix @ ((out_plane_shape - 1) / 2.0))[:2]
        out_plane_shape = out_plane_shape[:2]
        in_center = (in_plane_shape - 1) / 2.0
    else:
        out_plane_shape = in_plane_shape.copy()
        out_center = (matrix @ in_pixel_center)[:2]
        in_center = in_pixel_center[:2]

    # Translation to align centers
    offset = in_center - out_center
    translation = np.array(
        [
            [1.0, 0.0, offset[0]],
            [0.0, 1.0, offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    matrix = translation @ matrix  # Final 3x3

    # Per-plane transform function
    def _affine_2d(arr2d: np.ndarray) -> np.ndarray:
        out = np.empty(tuple(out_plane_shape), dtype=arr2d.dtype)
        scipy.ndimage.affine_transform(
            arr2d,
            matrix,
            output_shape=tuple(out_plane_shape),
            output=out,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )
        return out

    if reshape:
        # Rename dims to temporary names to avoid conflicts
        rot_ydim: Hashable = f"__rot_{ydim}"
        rot_xdim: Hashable = f"__rot_{xdim}"
        output_core_dims = [[rot_ydim, rot_xdim]]
        output_sizes = {
            rot_ydim: int(out_plane_shape[0]),
            rot_xdim: int(out_plane_shape[1]),
        }
    else:
        # We can keep original dim names if sizes don't change
        rot_ydim = ydim
        rot_xdim = xdim
        output_core_dims = [[ydim, xdim]]
        output_sizes = None

    rotated: xr.DataArray = xr.apply_ufunc(
        _affine_2d,
        darr,
        input_core_dims=[[ydim, xdim]],
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=[darr.dtype],
        dask_gufunc_kwargs={"output_sizes": output_sizes},
        vectorize=True,
        keep_attrs="no_conflicts",
    )

    if reshape:
        # Rename rotated dims back to original names
        rotated = rotated.rename({rot_ydim: ydim, rot_xdim: xdim})

    # Coords associated with rotated axes are meaningless after rotation
    for cname, coord in list(rotated.coords.items()):
        if cname in axes_dims:
            continue
        if any(ax in coord.dims for ax in axes_dims):
            rotated = rotated.drop_vars((cname,))

    if reshape:
        # Compute output coords in data space

        # Solve for the output pixel center in original space
        out_pixel_center = np.linalg.lstsq(matrix, in_pixel_center, rcond=None)[0][:2]

        start_y = -out_pixel_center[0] * dy + center_y
        end_y = start_y + (out_plane_shape[0] - 1) * dy

        start_x = -out_pixel_center[1] * dx + center_x
        end_x = start_x + (out_plane_shape[1] - 1) * dx

        rotated = rotated.assign_coords(
            {
                ydim: np.linspace(start_y, end_y, out_plane_shape[0]),
                xdim: np.linspace(start_x, end_x, out_plane_shape[1]),
            }
        )

        # Trim all-NaN edges
        rotated = erlab.utils.array.trim_na(rotated, axes_dims)

    return rotated.transpose(*darr.dims)


def shift(
    darr: xr.DataArray,
    shift: float | xr.DataArray,
    along: str,
    *,
    shift_coords: bool = False,
    **shift_kwargs,
) -> xr.DataArray:
    """Shifts the values of a DataArray along a single dimension.

    The shift is applied using :func:scipy.ndimage.shift` which uses spline
    interpolation. By default, the spline is of order 1 (linear interpolation).

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
        Additional keyword arguments passed onto `scipy.ndimage.shift`. The default
        values of some parameters are different from scipy. ``order`` is set to 1,
        ``cval`` is set to ``np.nan``, and ``prefilter`` is set to `False`.

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
    >>> import numpy as np
    >>> import erlab.analysis as era
    >>> darr = xr.DataArray(
    ...     np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(float), dims=["x", "y"]
    ... )
    >>> shift_arr = xr.DataArray([1, 0, 2], dims=["x"])
    >>> shifted = era.transform.shift(darr, shift_arr, along="y")
    >>> print(shifted)
    <xarray.DataArray (x: 3, y: 3)> Size: 72B
    array([[nan,  1.,  2.],
           [ 4.,  5.,  6.],
           [nan, nan,  7.]])
    Dimensions without coordinates: x, y
    """
    shift_kwargs.setdefault("order", 1)
    shift_kwargs.setdefault("mode", "constant")
    shift_kwargs.setdefault("prefilter", False)
    if shift_kwargs["mode"] == "constant":
        shift_kwargs.setdefault("cval", np.nan)

    if not isinstance(shift, xr.DataArray):
        shift = xr.DataArray(float(shift))

    # Check shift dims are valid
    for dim in shift.dims:
        if dim not in darr.dims:
            raise ValueError(f"Dimension {dim} in shift array not found in input array")
        if darr.sizes[dim] != shift.sizes[dim]:
            raise ValueError(
                f"Dimension {dim} in shift array has different size than input array"
            )

    if along in shift.dims:
        raise ValueError("Dimension to shift along cannot be in shift DataArray")

    # Sort along the target dimension
    out = darr.sortby(along)

    # Get step along the dimension (must be evenly spaced, same as before)
    coord = out[along].values
    if coord.size < 2:
        raise ValueError(f"Dimension {along} must have at least 2 points.")
    along_step: float = float(coord[1] - coord[0])

    # Normalize shift values to "index units" and fill NaNs
    shift = (shift.copy() / along_step).fillna(0.0)

    if shift_coords:
        # We first apply the integer part of the average shift to the coords
        rigid_shift: float = float(np.round(shift.values.mean()))
        shift = shift - rigid_shift

        # Apply rigid shift to coordinates
        out = out.assign_coords({along: out[along].values + rigid_shift * along_step})

        # Figure out padding needed from remaining shift range
        nshift_min, nshift_max = shift.values.min(), shift.values.max()
        pads: tuple[int, int] = (min(0, round(nshift_min)), max(0, round(nshift_max)))

        # Construct new coordinate array
        new_along = np.linspace(
            out[along].values[0] + pads[0] * along_step,
            out[along].values[-1] + pads[1] * along_step,
            out[along].size + abs(pads[0]) + abs(pads[1]),
        )

        # Pad data and assign new coords
        out = out.pad(
            {along: (abs(pads[0]), abs(pads[1]))},
            mode="constant",
            constant_values=np.nan,
        )
        if bool(out.chunks):
            out = out.chunk({along: -1})
        out = out.assign_coords({along: new_along})

    # Broadcast shift array to match non-along dims of output array
    shift_broadcast = shift.broadcast_like(out.isel({along: 0}, drop=True))

    # Core function to shift a 1D array
    def _shift_1d(arr_1d: np.ndarray, shift_scalar: np.ndarray) -> np.ndarray:
        # shift_scalar is 0-D here
        s = float(shift_scalar)
        return scipy.ndimage.shift(arr_1d, (s,), **shift_kwargs)

    # Apply over the `along` axis, vectorized over the rest
    # - arr has core dim [along]
    # - shift has no core dims (scalar for each outer position)
    return xr.apply_ufunc(
        _shift_1d,
        out,
        shift_broadcast,
        input_core_dims=[[along], []],
        output_core_dims=[[along]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[out.dtype],
    ).transpose(*out.dims)


def symmetrize(
    darr: xr.DataArray,
    dim: Hashable,
    *,
    center: float = 0.0,
    subtract: bool = False,
    mode: typing.Literal["full", "valid"] = "full",
    part: typing.Literal["both", "below", "above"] = "both",
    interp_kw: dict[str, typing.Any] | None = None,
) -> xr.DataArray:
    """
    Symmetrize a DataArray along a specified dimension around a given center.

    This function takes an input DataArray and symmetrizes its values along the
    specified dimension by reflecting and summing the data in regions below and above a
    given center.

    The operation assumes that the coordinate corresponding to the dimension is evenly
    spaced. Internally, the function interpolates the data to a shifted coordinate grid
    to align with the nearest grid point, performs the reflection, and concatenates the
    resulting halves.

    Parameters
    ----------
    darr : DataArray
        The input xarray DataArray to be symmetrized. Its coordinate along the specified
        dimension must be uniformly spaced.
    dim : Hashable
        The dimension along which to perform the symmetrization.
    center : float, optional
        The central value about which the data is symmetrized (default is 0.0).
    subtract : bool, optional
        If True, the reflected part is subtracted from the original data instead of
        being added, resulting in an antisymmetrized output instead of a symmetrized
        one. Default is False (i.e., the reflected part is added).
    mode: {'valid', 'full'}, optional
        How to handle the parts of the symmetrized data that does not overlap with the
        original data. If 'valid', only the part that exists in both the original and
        reflected data is returned. If 'full', the full symmetrized data is returned. In
        this case, all NaN values in the part that exists in the overlapping region are
        replaced with 0.0.
    part : {'both', 'below', 'above'}, optional
        The part of the symmetrized data to return. If 'both', the full symmetrized data
        is returned. If 'below', only the part below the center is returned. If 'above',
        only the part above the center is returned.
    interp_kw : dict, optional
        Additional keyword arguments passed to :meth:`xarray.DataArray.interp`.

    Returns
    -------
    DataArray
        A symmetrized DataArray where each value is the sum of its original and
        reflected counterpart.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> import erlab.analysis as era
    >>> # Create a sample DataArray with uniform coordinates.
    >>> da = xr.DataArray(
    ...     np.array([1, 2, 3, 4, 5, 6]), dims="x", coords={"x": np.linspace(-2, 2, 6)}
    ... )
    >>> sym_da = era.transform.symmetrize(da, dim="x", center=0.0)
    >>> print(sym_da)
    <xarray.DataArray (x: 6)> Size: 48B
    array([2., 4., 6., 6., 4., 2.])
    Coordinates:
      * x        (x) float64 48B -2.0 -1.2 -0.4 0.4 1.2 2.0
    """
    if not erlab.utils.array.is_dims_uniform(darr, (dim,)):
        raise ValueError(f"Coordinate along dimension {dim} must be uniformly spaced")

    if interp_kw is None:
        interp_kw = {}

    interp_kw.setdefault("assume_sorted", True)

    center = float(center)

    # Ensure coord is increasing

    is_increasing = darr[dim].values[1] > darr[dim].values[0]
    out = darr.copy()

    if not is_increasing:
        out = out.sortby(dim)

    with xr.set_options(keep_attrs=True):
        coord: xr.DataArray = out[dim]

        step = float(np.abs(coord[1] - coord[0]))
        closest_val = (
            float(typing.cast("xr.DataArray", np.abs(coord - center)).idxmin(dim))
            - center
        )  # displacement relative to nearest grid point

        shifted_coords = coord.values - closest_val - step / 2
        shifted_coords = np.append(shifted_coords, shifted_coords[-1] + step)

        # Prevent interpolation outside of original coordinate range
        if shifted_coords[0] < coord[0]:
            shifted_coords = shifted_coords[1:]
        if shifted_coords[-1] > coord[-1]:
            shifted_coords = shifted_coords[:-1]

        # Interpolate to shifted coordinate grid
        out_shifted = out.interp({dim: shifted_coords}, **interp_kw)

        # Split into parts below and above center
        below = out_shifted.where(out_shifted[dim] < center, drop=True)
        above = out_shifted.where(out_shifted[dim] > center, drop=True)

        n_below, n_above = len(below[dim]), len(above[dim])
        if n_below == 0 or n_above == 0:
            raise ValueError("Center does not lie within the coordinate range")

        if mode == "valid":
            len_valid = min(n_below, n_above)
            below = below.isel({dim: slice(-len_valid, None)})
            above = above.isel({dim: slice(0, len_valid)})

        # Reflect above
        above = above.assign_coords({dim: center - (above[dim] - center)}).sortby(dim)

        # Ensure flipped coord matches exactly with original
        match mode:
            case "valid":
                above = above.assign_coords({dim: below[dim]})
            case "full":
                if n_below > n_above:
                    above = above.interp({dim: below[dim]}).fillna(0.0)
                else:
                    below = below.interp({dim: above[dim]}).fillna(0.0)

        # Symmetrize
        sym_below = (below - above) if subtract else (below + above)

        # Retain coordinate attributes
        sym_below = sym_below.assign_coords(
            {dim: sym_below[dim].assign_attrs(coord.attrs)}
        )

        if part == "below":
            return (
                sym_below
                if is_increasing
                else sym_below.isel({dim: slice(None, None, -1)})
            )

        # Flip symmetrized data
        sym_above = (
            sym_below.copy()
            .assign_coords({dim: center - (sym_below[dim] - center)})
            .sortby(dim)
        )
        if subtract:
            sym_above = -sym_above

        if part == "above":
            return (
                sym_above
                if is_increasing
                else sym_above.isel({dim: slice(None, None, -1)})
            )

        out = xr.concat([sym_below, sym_above], dim=dim)

        if not is_increasing:
            out = out.isel({dim: slice(None, None, -1)})

        return out


def rotateinplane(data: xr.DataArray, rotate, **interp_kwargs):  # pragma: no cover
    """Rotate a 2D DataArray in the plane defined by the two dimensions.

    .. deprecated:: 2.9.0

        Use :func:`erlab.analysis.transform.rotate` instead.
    """
    warnings.warn(
        "erlab.analysis.transform.rotateinplane is deprecated, "
        "use erlab.analysis.transform.rotate instead",
        FutureWarning,
        stacklevel=1,
    )
    interp_kwargs.setdefault("method", "linearfast")

    theta = np.radians(rotate)
    d0, d1 = data.dims
    x = xr.DataArray(data[d0] * np.cos(theta) - data[d1] * np.sin(theta))
    y = xr.DataArray(data[d0] * np.sin(theta) + data[d1] * np.cos(theta))
    return data.interp({d0: x, d1: y}, **interp_kwargs)


def rotatestackinplane(data: xr.DataArray, rotate, **interp_kwargs):  # pragma: no cover
    """Rotate a 3D DataArray in the plane defined by the two dimensions.

    .. deprecated:: 2.9.0

        Use :func:`erlab.analysis.transform.rotate` instead.

    """
    warnings.warn(
        "erlab.analysis.transform.rotateinplane is deprecated, "
        "use erlab.analysis.transform.rotate instead",
        FutureWarning,
        stacklevel=1,
    )
    interp_kwargs.setdefault("method", "linearfast")

    theta = np.radians(rotate)
    d0, d1, _ = data.dims
    x = xr.DataArray(data[d0] * np.cos(theta) - data[d1] * np.sin(theta))
    y = xr.DataArray(data[d0] * np.sin(theta) + data[d1] * np.cos(theta))
    return data.interp({d0: x, d1: y}, **interp_kwargs)
