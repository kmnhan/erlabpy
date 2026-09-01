"""Transformations."""

__all__ = [
    "rotate",
    "rotateinplane",
    "rotatestackinplane",
    "shift",
    "symmetrize",
    "symmetrize_nfold",
]

import threading
import typing
import warnings
from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numba
import numpy as np
import scipy
import xarray as xr

import erlab

if typing.TYPE_CHECKING:
    import scipy.ndimage
    import scipy.special  # noqa: TC004

_NUMBA_AFFINE_DTYPES = frozenset(
    {
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    }
)


@dataclass(frozen=True, slots=True)
class _RotationPlane:
    axes_dims: tuple[Hashable, Hashable]
    ax_idx: tuple[int, int]
    ydim: Hashable
    xdim: Hashable
    ycoords: np.ndarray
    xcoords: np.ndarray
    dy: float
    dx: float
    center_y: float
    center_x: float
    in_plane_shape: tuple[int, int]
    in_pixel_center: np.ndarray
    scale: np.ndarray
    scale_inv: np.ndarray

    def base_matrix(self, angle: float) -> np.ndarray:
        c, s = scipy.special.cosdg(angle), scipy.special.sindg(angle)
        return (
            self.scale
            @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            @ self.scale_inv
        )


def _resolve_rotation_plane(
    darr: xr.DataArray,
    axes: tuple[int, int] | tuple[Hashable, Hashable],
    center: tuple[float, float] | Mapping[Hashable, float],
) -> _RotationPlane:
    # Resolve axes to dimension names.
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

    # Sort the rotation plane to match the array storage order.
    ax_idx = list(darr.get_axis_num(axes_dims))
    ax_idx, axes_dims = map(
        list, zip(*sorted(zip(ax_idx, axes_dims, strict=True)), strict=True)
    )

    # Record the sampled coordinates and spacing along the plane.
    ydim, xdim = axes_dims
    ycoords = darr[ydim].values
    xcoords = darr[xdim].values

    if ycoords.size < 2 or xcoords.size < 2:
        raise ValueError("axes must have at least 2 points each")

    dy = float(ycoords[1] - ycoords[0])
    dx = float(xcoords[1] - xcoords[0])
    pixel_ratio = float(abs(dy / dx))
    # Keep nominally square pixels exact across affine compiler implementations.
    if np.isclose(
        pixel_ratio,
        1.0,
        rtol=8 * np.finfo(float).eps,
        atol=0.0,
    ):
        pixel_ratio = 1.0

    # Interpret the center in data coordinates.
    if isinstance(center, Mapping):
        if set(center.keys()) != {ydim, xdim}:
            raise ValueError("center must have keys matching the two rotation axes")
        center_y = float(center[ydim])
        center_x = float(center[xdim])
    else:
        center_y, center_x = center

    # Express the center in pixel coordinates for ndimage transforms.
    in_pixel_center = np.array(
        [
            (center_y - ycoords[0]) / dy,
            (center_x - xcoords[0]) / dx,
            1.0,
        ]
    )

    return _RotationPlane(
        axes_dims=(ydim, xdim),
        ax_idx=(int(ax_idx[0]), int(ax_idx[1])),
        ydim=ydim,
        xdim=xdim,
        ycoords=ycoords,
        xcoords=xcoords,
        dy=dy,
        dx=dx,
        center_y=center_y,
        center_x=center_x,
        in_plane_shape=(int(darr.shape[ax_idx[0]]), int(darr.shape[ax_idx[1]])),
        in_pixel_center=in_pixel_center,
        scale=np.diag([1.0 / pixel_ratio, 1.0, 1.0]),
        scale_inv=np.diag([pixel_ratio, 1.0, 1.0]),
    )


def _rotation_output_signature(
    ydim: Hashable,
    xdim: Hashable,
    out_plane_shape: tuple[int, int],
    *,
    reshape: bool,
) -> tuple[Hashable, Hashable, list[list[Hashable]], dict[Hashable, int] | None]:
    # Use temporary dim names when the rotated plane changes size.
    if reshape:
        rot_ydim: Hashable = f"__rot_{ydim}"
        rot_xdim: Hashable = f"__rot_{xdim}"
        return (
            rot_ydim,
            rot_xdim,
            [[rot_ydim, rot_xdim]],
            {rot_ydim: out_plane_shape[0], rot_xdim: out_plane_shape[1]},
        )

    return ydim, xdim, [[ydim, xdim]], None


def _drop_rotated_axis_coords(
    darr: xr.DataArray, axes_dims: tuple[Hashable, Hashable]
) -> xr.DataArray:
    out = darr
    for cname, coord in list(out.coords.items()):
        # Coordinates that depend on rotated axes no longer describe the output grid.
        if cname in axes_dims:
            continue
        if any(ax in coord.dims for ax in axes_dims):
            out = out.drop_vars((cname,))
    return out


def _plane_midpoint(shape: tuple[int, int]) -> np.ndarray:
    return (np.asarray(shape, dtype=float) - 1.0) / 2.0


def _aligned_affine_matrix(
    base_matrix: np.ndarray, input_center: np.ndarray, output_center: np.ndarray
) -> np.ndarray:
    # Translate the rotated plane so the chosen centers coincide.
    output_center_h = np.array([output_center[0], output_center[1], 1.0])
    offset = np.asarray(input_center, dtype=float) - (base_matrix @ output_center_h)[:2]
    translation = np.array(
        [
            [1.0, 0.0, offset[0]],
            [0.0, 1.0, offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return translation @ base_matrix


@numba.njit(nogil=True, inline="always", cache=True, fastmath={"contract"})
def _map_affine_point(
    matrix: np.ndarray, y_index: int, x_index: int
) -> tuple[float, float]:
    """Map one output point to input pixel coordinates."""
    mapped_y = matrix[0, 2]
    mapped_y += matrix[0, 0] * y_index
    mapped_y += matrix[0, 1] * x_index
    mapped_x = matrix[1, 2]
    mapped_x += matrix[1, 0] * y_index
    mapped_x += matrix[1, 1] * x_index
    return mapped_y, mapped_x


@numba.njit(nogil=True, cache=True, fastmath={"contract"})
def _rotation_geometry_bounds(
    matrices: np.ndarray,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the tight output bounds that sample the input plane."""
    input_y, input_x = input_shape
    output_y, output_x = output_shape
    y_start = output_y
    y_stop = 0
    x_start = output_x
    x_stop = 0

    for y_index in range(output_y):
        for x_index in range(output_x):
            for matrix_index in range(matrices.shape[0]):
                mapped_y, mapped_x = _map_affine_point(
                    matrices[matrix_index], y_index, x_index
                )
                if 0.0 <= mapped_y <= input_y - 1 and 0.0 <= mapped_x <= input_x - 1:
                    y_start = min(y_start, y_index)
                    y_stop = max(y_stop, y_index + 1)
                    x_start = min(x_start, x_index)
                    x_stop = max(x_stop, x_index + 1)
                    break

    if y_stop == 0:
        return 0, 0, 0, 0
    return y_start, y_stop, x_start, x_stop


@numba.njit(nogil=True, inline="always", cache=True, fastmath={"contract"})
def _fill_affine_row(
    arr: np.ndarray,
    matrices: np.ndarray,
    output_origin: tuple[int, int],
    cval: complex,
    out: np.ndarray,
    batch_index: int,
    y_index: int,
) -> None:
    """Fill one output row with one affine transform or their mean."""
    matrix_count = matrices.shape[0]
    _, input_y, input_x = arr.shape
    output_y_index = y_index + output_origin[0]
    for x_index in range(out.shape[2]):
        output_x_index = x_index + output_origin[1]
        count = 0
        for matrix_index in range(matrix_count):
            mapped_y, mapped_x = _map_affine_point(
                matrices[matrix_index], output_y_index, output_x_index
            )
            if (
                mapped_y < 0.0
                or mapped_y > input_y - 1
                or mapped_x < 0.0
                or mapped_x > input_x - 1
            ):
                value = cval
            else:
                y0 = int(np.floor(mapped_y))
                x0 = int(np.floor(mapped_x))
                y_weight = mapped_y - y0
                x_weight = mapped_x - x0
                if y0 == input_y - 1:
                    y0 -= 1
                    y_weight = 1.0
                if x0 == input_x - 1:
                    x0 -= 1
                    x_weight = 1.0
                y1 = y0 + 1
                x1 = x0 + 1
                value00 = arr[batch_index, y0, x0]
                value01 = arr[batch_index, y0, x1]
                value10 = arr[batch_index, y1, x0]
                value11 = arr[batch_index, y1, x1]
                weight00 = (1.0 - y_weight) * (1.0 - x_weight)
                weight01 = (1.0 - y_weight) * x_weight
                weight10 = y_weight * (1.0 - x_weight)
                weight11 = y_weight * x_weight
                if isinstance(value00, (complex, np.complex64, np.complex128)):
                    value = complex(
                        weight00 * value00.real
                        + weight01 * value01.real
                        + weight10 * value10.real
                        + weight11 * value11.real,
                        weight00 * value00.imag
                        + weight01 * value01.imag
                        + weight10 * value10.imag
                        + weight11 * value11.imag,
                    )
                elif (
                    np.isnan(value00)
                    or np.isnan(value01)
                    or np.isnan(value10)
                    or np.isnan(value11)
                ):
                    value = np.nan
                else:
                    value = (
                        weight00 * value00
                        + weight01 * value01
                        + weight10 * value10
                        + weight11 * value11
                    )

            if matrix_count == 1:
                out[batch_index, y_index, x_index] = value
            elif not np.isnan(value):
                if count == 0:
                    out[batch_index, y_index, x_index] = value
                else:
                    out[batch_index, y_index, x_index] += value
                count += 1

        if matrix_count != 1:
            if count == 0:
                out[batch_index, y_index, x_index] = np.nan
            else:
                out[batch_index, y_index, x_index] /= count


@numba.njit(nogil=True, parallel=True, cache=True, fastmath={"contract"})
def _apply_affine_linear_numba(
    arr: np.ndarray,
    matrices: np.ndarray,
    output_origin: tuple[int, int],
    cval: complex,
    out: np.ndarray,
) -> None:
    """Apply one or average multiple affine transforms in parallel."""
    for row_index in numba.prange(arr.shape[0] * out.shape[1]):
        batch_index = row_index // out.shape[1]
        y_index = row_index - batch_index * out.shape[1]
        _fill_affine_row(arr, matrices, output_origin, cval, out, batch_index, y_index)


@numba.njit(nogil=True, cache=True, fastmath={"contract"})
def _apply_affine_linear_numba_serial(
    arr: np.ndarray,
    matrices: np.ndarray,
    output_origin: tuple[int, int],
    cval: complex,
    out: np.ndarray,
) -> None:
    """Apply one or average multiple affine transforms serially."""
    for row_index in range(arr.shape[0] * out.shape[1]):
        batch_index = row_index // out.shape[1]
        y_index = row_index - batch_index * out.shape[1]
        _fill_affine_row(arr, matrices, output_origin, cval, out, batch_index, y_index)


def _apply_affine_linear(
    arr: np.ndarray,
    matrices: np.ndarray,
    output_shape: tuple[int, int],
    cval: complex,
    *,
    serial: bool,
    output_origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Apply affine transforms over arbitrary batch dimensions."""
    input_shape = arr.shape
    arr = np.ascontiguousarray(arr).reshape((-1, *arr.shape[-2:]))
    out = np.empty((arr.shape[0], *output_shape), dtype=arr.dtype)
    use_serial = serial or threading.current_thread() is not threading.main_thread()
    kernel = (
        _apply_affine_linear_numba_serial if use_serial else _apply_affine_linear_numba
    )
    kernel(
        arr,
        matrices,
        output_origin,
        np.asarray(cval, dtype=arr.dtype)[()],
        out,
    )
    return out.reshape((*input_shape[:-2], *output_shape))


def _rotated_plane_shape(
    base_matrix: np.ndarray, in_plane_shape: tuple[int, int]
) -> tuple[int, int]:
    # Rotate the input corners to determine the output bounding box.
    iy, ix = in_plane_shape
    corners = np.array([[0, 0, iy, iy], [0, ix, 0, ix], [1, 1, 1, 1]])
    out_bounds = base_matrix @ corners
    return tuple((np.ptp(out_bounds, axis=1) + 0.5).astype(int)[:2])


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
        If `True`, the output shape is adapted to the full rotated bounding box. The
        extent depends on the input coordinates, not on finite data values. Default is
        `True`.
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
    # Resolve rotation metadata once and build the pixel-space rotation matrix.
    plane = _resolve_rotation_plane(darr, axes, center)
    base_matrix = plane.base_matrix(angle)

    # Either expand to the full bounding box or keep the original grid.
    if reshape:
        out_plane_shape = _rotated_plane_shape(base_matrix, plane.in_plane_shape)
        matrix = _aligned_affine_matrix(
            base_matrix,
            _plane_midpoint(plane.in_plane_shape),
            _plane_midpoint(out_plane_shape),
        )
    else:
        out_plane_shape = plane.in_plane_shape
        matrix = _aligned_affine_matrix(
            base_matrix,
            plane.in_pixel_center[:2],
            plane.in_pixel_center[:2],
        )

    rot_ydim, rot_xdim, output_core_dims, output_sizes = _rotation_output_signature(
        plane.ydim, plane.xdim, out_plane_shape, reshape=reshape
    )

    if order == 1 and mode == "constant" and darr.dtype in _NUMBA_AFFINE_DTYPES:
        apply_affine = _apply_affine_linear
        apply_kwargs = {
            "matrices": matrix[None, ...],
            "output_shape": out_plane_shape,
            "cval": cval,
            "serial": darr.chunks is not None,
        }
        vectorize = False
    else:

        def _apply_affine_scipy(arr2d: np.ndarray) -> np.ndarray:
            out = np.empty(out_plane_shape, dtype=arr2d.dtype)
            scipy.ndimage.affine_transform(
                arr2d,
                matrix,
                output_shape=out_plane_shape,
                output=out,
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
            )
            return out

        apply_affine = _apply_affine_scipy
        apply_kwargs = {}
        vectorize = True

    rotated: xr.DataArray = xr.apply_ufunc(
        apply_affine,
        darr,
        input_core_dims=[[plane.ydim, plane.xdim]],
        output_core_dims=output_core_dims,
        kwargs=apply_kwargs,
        dask="parallelized",
        output_dtypes=[darr.dtype],
        dask_gufunc_kwargs={"output_sizes": output_sizes},
        vectorize=vectorize,
        keep_attrs="no_conflicts",
    )

    if reshape:
        # Rename rotated dims back to original names
        rotated = rotated.rename({rot_ydim: plane.ydim, rot_xdim: plane.xdim})

    # Coords associated with rotated axes are meaningless after rotation
    rotated = _drop_rotated_axis_coords(rotated, plane.axes_dims)

    if reshape:
        # Compute output coords in data space

        # Solve for the output pixel center in original space
        out_pixel_center = np.linalg.lstsq(matrix, plane.in_pixel_center, rcond=None)[
            0
        ][:2]

        start_y = -out_pixel_center[0] * plane.dy + plane.center_y
        end_y = start_y + (out_plane_shape[0] - 1) * plane.dy

        start_x = -out_pixel_center[1] * plane.dx + plane.center_x
        end_x = start_x + (out_plane_shape[1] - 1) * plane.dx

        rotated = rotated.assign_coords(
            {
                plane.ydim: np.linspace(start_y, end_y, out_plane_shape[0]),
                plane.xdim: np.linspace(start_x, end_x, out_plane_shape[1]),
            }
        )

    return rotated.transpose(*darr.dims)


def symmetrize_nfold(
    darr: xr.DataArray,
    fold: int,
    axes: tuple[int, int] | tuple[Hashable, Hashable] = (0, 1),
    center: tuple[float, float] | Mapping[Hashable, float] = (0.0, 0.0),
    *,
    reshape: bool = True,
    order: int = 1,
    mode: str = "constant",
    cval: float = np.nan,
    prefilter: bool = True,
) -> xr.DataArray:
    r"""Symmetrize a plane by averaging equally spaced rotations.

    The input is rotated in the plane defined by `axes` at angles :math:`360° i / n`,
    where :math:`i = 0, \ldots, n - 1`, and the rotated copies are averaged on a
    common output grid.

    Parameters
    ----------
    darr
        The array to symmetrize.
    fold
        The order of the rotational symmetry. Must be at least 2. For example,
        ``fold=4`` applies 4-fold symmetrization by averaging over the original array
        and arrays rotated by 90°, 180°, and 270°.
    axes : tuple of 2 ints or 2 strings, optional
        The two axes that define the plane of rotation. Default is the first two axes.
        If strings are provided, they must be valid dimension names in the input array.
    center : tuple of 2 floats or dict, optional
        The center of rotation in data coordinates. If a tuple, it is given as values
        along the dimensions specified in `axes`. If a dict, it must have keys that
        correspond to `axes`. Default is (0, 0).
    reshape
        If `True`, the output shape is expanded to contain the full extent of all
        rotated copies. The extent depends on the input coordinates, not on finite data
        values. If `False`, the symmetrized result is returned on the original grid.
        Default is `True`.
    order
        The order of the spline interpolation, default is 1. The order has to be in the
        range 0-5.
    mode, cval, prefilter
        Passed to :func:`scipy.ndimage.affine_transform`. See the scipy documentation
        for more information.

    Returns
    -------
    darr : xarray.DataArray
        The rotationally symmetrized array on the original or expanded grid.

    """
    if fold < 2:
        raise ValueError("fold must be at least 2")

    # Interpolation and averaging need a floating or complex dtype.
    rotated_input = darr
    if not (
        np.issubdtype(rotated_input.dtype, np.floating)
        or np.issubdtype(rotated_input.dtype, np.complexfloating)
    ):
        rotated_input = rotated_input.astype(np.result_type(rotated_input.dtype, float))

    # Resolve the rotation plane once and reuse it for every angle.
    plane = _resolve_rotation_plane(rotated_input, axes, center)

    def _expanded_coords(
        coord0: float, step: float, edge_min: float, edge_max: float
    ) -> np.ndarray:
        # Snap the rotated extent back onto the original coordinate lattice.
        if step < 0:
            return -_expanded_coords(-coord0, -step, -edge_max, -edge_min)

        idx_min = int(np.floor((edge_min + step / 2 - coord0) / step))
        idx_max = int(np.ceil((edge_max - step / 2 - coord0) / step))
        return coord0 + np.arange(idx_min, idx_max + 1) * step

    # Either expand to the union of all rotated copies or reuse the input grid.
    if reshape:
        y_edges = np.array(
            [
                plane.ycoords[0] - plane.dy / 2,
                plane.ycoords[0] - plane.dy / 2,
                plane.ycoords[-1] + plane.dy / 2,
                plane.ycoords[-1] + plane.dy / 2,
            ]
        )
        x_edges = np.array(
            [
                plane.xcoords[0] - plane.dx / 2,
                plane.xcoords[-1] + plane.dx / 2,
                plane.xcoords[0] - plane.dx / 2,
                plane.xcoords[-1] + plane.dx / 2,
            ]
        )

        all_y_edges = []
        all_x_edges = []
        for idx in range(fold):
            angle = 360.0 * idx / fold
            c, s = scipy.special.cosdg(angle), scipy.special.sindg(angle)
            y_offset = y_edges - plane.center_y
            x_offset = x_edges - plane.center_x
            all_y_edges.append(plane.center_y + c * y_offset + s * x_offset)
            all_x_edges.append(plane.center_x - s * y_offset + c * x_offset)

        y_edge_min = float(np.min(all_y_edges))
        y_edge_max = float(np.max(all_y_edges))
        x_edge_min = float(np.min(all_x_edges))
        x_edge_max = float(np.max(all_x_edges))

        out_ycoords = _expanded_coords(
            float(plane.ycoords[0]), plane.dy, y_edge_min, y_edge_max
        )
        out_xcoords = _expanded_coords(
            float(plane.xcoords[0]), plane.dx, x_edge_min, x_edge_max
        )
        out_plane_shape = (len(out_ycoords), len(out_xcoords))
        # Express the common symmetrization center on the expanded grid.
        out_center = np.array(
            [
                (plane.center_y - out_ycoords[0]) / plane.dy,
                (plane.center_x - out_xcoords[0]) / plane.dx,
            ]
        )
    else:
        out_plane_shape = plane.in_plane_shape
        out_ycoords = None
        out_xcoords = None
        out_center = plane.in_pixel_center[:2]

    # Precompute aligned affine matrices for each rotated copy.
    matrices = np.stack(
        [
            _aligned_affine_matrix(
                plane.base_matrix(360.0 * idx / fold),
                plane.in_pixel_center[:2],
                out_center,
            )
            for idx in range(fold)
        ]
    )

    full_out_plane_shape = out_plane_shape
    output_origin = (0, 0)
    output_slices = (slice(None), slice(None))
    if reshape and mode == "constant" and bool(np.isnan(cval)):
        y_start, y_stop, x_start, x_stop = _rotation_geometry_bounds(
            matrices, plane.in_plane_shape, out_plane_shape
        )
        y_slice = slice(y_start, y_stop)
        x_slice = slice(x_start, x_stop)
        out_ycoords = typing.cast("np.ndarray", out_ycoords)[y_slice]
        out_xcoords = typing.cast("np.ndarray", out_xcoords)[x_slice]
        out_plane_shape = (len(out_ycoords), len(out_xcoords))
        output_origin = (y_start, x_start)
        output_slices = (y_slice, x_slice)

    dtype = rotated_input.dtype
    rot_ydim, rot_xdim, output_core_dims, output_sizes = _rotation_output_signature(
        plane.ydim, plane.xdim, out_plane_shape, reshape=reshape
    )

    if order == 1 and mode == "constant" and dtype in _NUMBA_AFFINE_DTYPES:
        average_rotations = _apply_affine_linear
        apply_kwargs = {
            "matrices": matrices,
            "output_shape": out_plane_shape,
            "output_origin": output_origin,
            "cval": cval,
            "serial": rotated_input.chunks is not None,
        }
        vectorize = False
    else:
        nan_value = np.array(np.nan, dtype=dtype)[()]

        # Accumulate the mean directly to avoid concat/mean overhead.
        def _average_rotations_scipy(arr2d: np.ndarray) -> np.ndarray:
            total = np.zeros(full_out_plane_shape, dtype=dtype)
            count = np.zeros(full_out_plane_shape, dtype=np.intp)
            rotated = np.empty(full_out_plane_shape, dtype=dtype)

            for matrix in matrices:
                scipy.ndimage.affine_transform(
                    arr2d,
                    matrix,
                    output_shape=full_out_plane_shape,
                    output=rotated,
                    order=order,
                    mode=mode,
                    cval=cval,
                    prefilter=prefilter,
                )

                valid = ~np.isnan(rotated)
                if bool(valid.all()):
                    total += rotated
                    count += 1
                else:
                    np.copyto(rotated, 0, where=~valid)
                    total += rotated
                    count += valid

            out = np.full(full_out_plane_shape, nan_value, dtype=dtype)
            np.divide(total, count, out=out, where=count > 0)
            return out[output_slices]

        average_rotations = _average_rotations_scipy
        apply_kwargs = {}
        vectorize = True

    out = xr.apply_ufunc(
        average_rotations,
        rotated_input,
        input_core_dims=[[plane.ydim, plane.xdim]],
        output_core_dims=output_core_dims,
        kwargs=apply_kwargs,
        dask="parallelized",
        output_dtypes=[dtype],
        dask_gufunc_kwargs={"output_sizes": output_sizes},
        vectorize=vectorize,
        keep_attrs="no_conflicts",
    )

    # Restore output axis coordinates after any reshape expansion.
    if reshape:
        out = out.rename({rot_ydim: plane.ydim, rot_xdim: plane.xdim}).assign_coords(
            {plane.ydim: out_ycoords, plane.xdim: out_xcoords}
        )

    # Drop dependent coordinates tied to the rotated plane.
    out = _drop_rotated_axis_coords(out, plane.axes_dims)

    return out.assign_attrs(darr.attrs).transpose(*darr.dims)


def _ndimage_shift(arr, shift, order=3, mode="constant", cval=0.0, prefilter=False):
    if order == 1 and mode == "constant":
        x = np.arange(arr.size)
        return erlab.analysis.interpolate._interp1_serial(x, arr, x - shift[0], cval)

    return scipy.ndimage.shift(
        arr, shift, order=order, mode=mode, cval=cval, prefilter=prefilter
    )


def _validate_shift_inputs(
    data: xr.DataArray,
    shift: float | xr.DataArray,
    along: str,
    *,
    assume_sorted: bool = False,
) -> None:
    """Validate the structure and coordinates of inputs for :func:`shift`."""
    if along not in data.dims:
        raise ValueError(f"Dimension {along} not found in input array")

    along_coord = data[along]
    if not np.issubdtype(along_coord.dtype, np.number):
        raise ValueError(f"Coordinate {along} must have a numeric dtype")
    if np.issubdtype(along_coord.dtype, np.complexfloating):
        raise ValueError(f"Coordinate {along} must contain real values")

    coord = np.asarray(along_coord.values)
    if not assume_sorted:
        coord = np.sort(coord)
    if coord.size < 2:
        raise ValueError(f"Dimension {along} must have at least 2 points")
    if not np.all(np.isfinite(coord)):
        raise ValueError(f"Coordinate {along} must contain only finite values")

    steps = np.diff(coord)
    along_step = float(steps[0])
    if along_step == 0.0:
        raise ValueError(f"Coordinate {along} must have a nonzero step")
    if not np.allclose(steps, along_step):
        raise ValueError(f"Coordinate {along} must be uniformly spaced")

    shift_dtype = (
        shift.dtype if isinstance(shift, xr.DataArray) else np.asarray(shift).dtype
    )
    if not np.issubdtype(shift_dtype, np.number):
        raise ValueError("Shift values must have a numeric dtype")
    if np.issubdtype(shift_dtype, np.complexfloating):
        raise ValueError("Shift values must be real")
    if not isinstance(shift, xr.DataArray):
        return
    if along in shift.dims:
        raise ValueError("Dimension to shift along cannot be in shift DataArray")

    for dim in shift.dims:
        if dim not in data.dims:
            raise ValueError(f"Dimension {dim} in shift array not found in input array")
        if data.sizes[dim] != shift.sizes[dim]:
            raise ValueError(
                f"Dimension {dim} in shift array has different size than input array"
            )

        if dim in data.indexes and dim in shift.indexes:
            input_index = data.indexes[dim]
            shift_index = shift.indexes[dim]
            if (
                not input_index.is_unique
                or not shift_index.is_unique
                or not input_index.isin(shift_index).all()
                or not shift_index.isin(input_index).all()
            ):
                raise ValueError(
                    f"Indexes for dimension {dim} in shift and input arrays "
                    "do not align exactly"
                )


def _align_shift_indexes(darr: xr.DataArray, shift: xr.DataArray) -> xr.DataArray:
    """Align shift indexes to input indexes after structural validation."""
    indexers: dict[Hashable, typing.Any] = {}
    for dim in shift.dims:
        if dim in darr.indexes and dim in shift.indexes:
            indexers[dim] = darr.indexes[dim]
        elif dim in darr.indexes:
            shift = shift.assign_coords({dim: darr[dim]})
        elif dim in shift.indexes:
            shift = shift.drop_vars([dim])
    return shift.sel(indexers) if indexers else shift


def shift(
    darr: xr.DataArray,
    shift: float | xr.DataArray,
    along: str,
    *,
    shift_coords: bool = False,
    keep_dim_order: bool = True,
    assume_sorted: bool = False,
    **shift_kwargs,
) -> xr.DataArray:
    """Shifts the values of a DataArray along a single dimension.

    The shift is applied using :func:`scipy.ndimage.shift`, which uses spline
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
    keep_dim_order
        If `True`, the output array will be transposed to match the input data.
        Otherwise, the axis order may change due to the application of
        :func:`xarray.apply_ufunc`. Default is `True`.
    assume_sorted
        If `False`, the data is sorted with respect to ``along`` using
        :meth:`xarray.DataArray.sortby`. Providing `True` skips the sort. Use `True`
        when you are already sure that the data is sorted ascending with respect to
        ``along``.
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

    _validate_shift_inputs(darr, shift, along, assume_sorted=assume_sorted)
    if not isinstance(shift, xr.DataArray):
        shift = xr.DataArray(shift)
    shift = _align_shift_indexes(darr, shift)

    # Sort along the target dimension
    out = darr if assume_sorted else darr.sortby(along)

    along_step = float(out[along].values[1] - out[along].values[0])

    # Normalize shift values to "index units"
    shift = shift.copy() / along_step

    if shift_coords:
        # We first apply the integer part of the average shift to the coords
        rigid_shift: float = round(float(shift.mean(skipna=True).fillna(0.0)))

        shift = (shift - rigid_shift).fillna(0.0)

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
    else:
        shift = shift.fillna(0.0)

    # Broadcast shift array to match non-along dims of output array
    shift_broadcast = shift.broadcast_like(out.isel({along: 0}, drop=True))

    # Core function to shift a 1D array
    def _shift_1d(arr_1d: np.ndarray, shift_scalar: np.ndarray) -> np.ndarray:
        # shift_scalar is 0-D here
        s = float(shift_scalar)
        return _ndimage_shift(arr_1d, (s,), **shift_kwargs)

    # Apply over the `along` axis, vectorized over the rest
    # - arr has core dim [along]
    # - shift has no core dims (scalar for each outer position)
    original_dims = tuple(out.dims)
    out = xr.apply_ufunc(
        _shift_1d,
        out,
        shift_broadcast,
        input_core_dims=[[along], []],
        output_core_dims=[[along]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[out.dtype],
    )
    if keep_dim_order:
        out = out.transpose(*original_dims)
    return out


def symmetrize(
    darr: xr.DataArray,
    dim: Hashable,
    *,
    center: float = 0.0,
    subtract: bool = False,
    average: bool = False,
    mode: typing.Literal["full", "valid"] = "full",
    part: typing.Literal["both", "below", "above"] = "both",
    interp_kw: dict[str, typing.Any] | None = None,
) -> xr.DataArray:
    """
    Symmetrize a DataArray along a specified dimension around a given center.

    This function takes an input DataArray and symmetrizes its values along the
    specified dimension by reflecting and combining the data in regions below and
    above a given center.

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
    average : bool, optional
        If True, divide the summed or subtracted values by 2 where the original and
        reflected coordinate ranges overlap. Values outside the overlapping region
        remain unchanged when ``mode="full"``. Default is False.
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
        A symmetrized DataArray containing the sum or difference of each value and its
        reflected counterpart, optionally divided by 2 in the overlapping region.

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
    if darr[dim].size < 2:
        raise ValueError(
            f"Coordinate along dimension {dim} must contain at least two values"
        )

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
                    above = (
                        above.assign_coords(
                            {dim: below[dim].isel({dim: slice(-n_above, None)})}
                        )
                        .reindex({dim: below[dim]}, fill_value=0.0)
                        .fillna(0.0)
                    )
                else:
                    below = (
                        below.assign_coords(
                            {dim: above[dim].isel({dim: slice(-n_below, None)})}
                        )
                        .reindex({dim: above[dim]}, fill_value=0.0)
                        .fillna(0.0)
                    )

        # Symmetrize
        sym_below = (below - above) if subtract else (below + above)
        if average:
            overlap_size = min(n_below, n_above)
            overlap_divisor = np.ones(sym_below.sizes[dim], dtype=float)
            overlap_divisor[-overlap_size:] = 2.0
            sym_below = sym_below / xr.DataArray(
                overlap_divisor,
                dims=(dim,),
                coords={dim: sym_below[dim]},
                name=sym_below.name,
            )

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
