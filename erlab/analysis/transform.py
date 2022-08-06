"""Transformations."""

import numpy as np
import xarray as xr

__all__ = ["rotateinplane", "rotatestackinplane"]


def rotateinplane(data: xr.DataArray, rotate):
    if isinstance(data, xr.Dataset):
        data = data.spectrum
    theta = np.radians(rotate)
    d0, d1 = data.dims
    x = xr.DataArray(data[d0] * np.cos(theta) - data[d1] * np.sin(theta))
    y = xr.DataArray(data[d0] * np.sin(theta) + data[d1] * np.cos(theta))
    data_r = data.interp({d0: x, d1: y})
    return data_r


def rotatestackinplane(data: xr.DataArray, rotate):
    if isinstance(data, xr.Dataset):
        data = data.spectrum
    theta = np.radians(rotate)
    d0, d1, _ = data.dims
    x = xr.DataArray(data[d0] * np.cos(theta) - data[d1] * np.sin(theta))
    y = xr.DataArray(data[d0] * np.sin(theta) + data[d1] * np.cos(theta))
    data_r = data.interp({d0: x, d1: y})
    return data_r
