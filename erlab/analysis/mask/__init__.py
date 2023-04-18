"""Functions related to masking.

"""
import numpy as np
import numpy.typing as npt
import numba
import xarray as xr

from erlab.analysis.mask import polygon

__all__ = [
    "mask_with_polygon",
    "polygon_mask",
    "polygon_mask_points",
    "mask_with_hex_bz",
]


def mask_with_polygon(arr, vertices, dims=("kx", "ky"), invert=False):
    mask = xr.DataArray(
        polygon_mask(
            vertices.astype(np.float64), *(arr[d].values for d in dims), invert=invert
        ),
        dims=dims,
        coords={d: arr.coords[d] for d in dims},
    )
    return arr.where(mask == 1)


@numba.njit(parallel=True, cache=True)
def polygon_mask(vertices, x, y, invert=False):
    shape = (len(x), len(y))
    mask = np.empty(shape, dtype=np.int16)
    vertices = np.concatenate((vertices, vertices[0].reshape(1, -1)))

    for i in numba.prange(shape[0]):
        for j in range(shape[1]):
            mask[i,j] = polygon.bounded_side_bool(vertices, (x[i], y[j]))
    if invert:
        return ~mask
    else:
        return mask


@numba.njit(parallel=True, cache=True)
def polygon_mask_points(vertices, x, y, invert=False):
    if len(x) != len(y):
        raise ValueError
    mask = np.empty(len(x), dtype=np.int16)
    vertices = np.concatenate((vertices, vertices[0].reshape(1, -1)))
    for i in numba.prange(len(x)):
        mask[i] = polygon.bounded_side_bool(vertices, (x[i], y[i]))
    if invert:
        return ~mask
    else:
        return mask


def mask_with_hex_bz(kxymap: xr.DataArray, a=3.54, rotate=0, invert=False):
    """Returns array masked with a hexagonal BZ."""
    if isinstance(kxymap, xr.Dataset):
        kxymap = kxymap.spectrum

    if "kx" in kxymap.dims:
        dims = ("kx", "ky")
    elif "qx" in kxymap.dims:
        dims = ("kx", "ky")

    l = 2 * np.pi / (a * 3)
    ang = rotate + np.array([0, 60, 120, 180, 240, 300])
    vertices = np.array(
        [[2 * l * np.cos(t), 2 * l * np.sin(t)] for t in np.deg2rad(ang)]
    )
    return mask_with_polygon(kxymap, vertices, dims, invert=invert)
