"""Macros for correlation analysis.

"""

__all__ = ["acf2stack", "acf2", "match_dims", "xcorr1d"]


from itertools import chain, product

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.signal import correlate, correlation_lags


def autocorrelate(arr, *args, **kwargs):
    acf = correlate(arr, arr, *args, **kwargs)
    m, n = [s // 2 for s in acf.shape]
    return acf / acf[m, n]


def autocorrelation_lags(in_len, *args, **kwargs):
    return correlation_lags(in_len, in_len, *args, **kwargs)


def nanacf(arr, *args, **kwargs):
    acf = autocorrelate(np.nan_to_num(arr), *args, **kwargs)
    if np.isnan(arr).any():
        nan_mask = ~np.isnan(arr)
        acf_nan = autocorrelate(nan_mask.astype(float), *args, **kwargs)
        acf_nan[acf_nan < 1e7 * np.finfo(float).eps] = np.nan
        return acf / acf_nan
    else:
        return acf


def acf2(arr, mode: str = "full", method: str = "fft"):
    out = arr.copy(deep=True)
    acf = nanacf(out.values, mode=mode, method=method)
    coords = [out[d].values for d in out.dims]
    steps = [c[1] - c[0] for c in coords]
    out = xr.DataArray(
        acf,
        {
            d: autocorrelation_lags(l, mode) * s
            for s, l, d in zip(steps, arr.shape, out.dims)
        },
        attrs=out.attrs,
    )
    if all(i in out.dims for i in ["kx", "ky"]):
        out = out.rename(dict(kx="qx", ky="qy"))
    return out


def acf2stack(arr, stack_dims=["eV"], mode: str = "full", method: str = "fft"):
    if arr.ndim == 2:
        return acf2(arr, mode, method)
    elif arr.ndim >= 3:
        if arr.ndim - len(stack_dims) != 2:
            raise ValueError(
                "The number of dimensions excluding the stacking dimensions must be 2"
            )

        stack_coords = tuple(arr[d] for d in stack_dims)
        stack_shape = tuple(len(x) for x in stack_coords)
        stack_iter = tuple(range(s) for s in stack_shape)
        stack_axis = arr.get_axis_num(stack_dims)
        out_list = Parallel(n_jobs=-1, pre_dispatch="3 * n_jobs")(
            delayed(nanacf)(
                np.squeeze(arr.isel({s: v for s, v in zip(stack_dims, vals)}).values),
                mode,
                method,
            )
            for vals in product(*stack_iter)
        )
        acf_dims = tuple(filter(lambda d: d not in stack_dims, arr.dims))
        acf_shape = out_list[0].shape
        acf_steps = tuple(arr[d].values[1] - arr[d].values[0] for d in acf_dims)

        if mode == "same":
            out = arr.copy(deep=True)
        else:
            out = xr.DataArray(
                np.empty(stack_shape + acf_shape),
                dims=[d for d in chain(stack_dims, acf_dims)],
                attrs=arr.attrs,
            )
            out = out.assign_coords({d: arr[d] for d in stack_dims})
        for i, cut in enumerate(out_list):
            index = np.unravel_index(i, stack_shape)
            out[
                tuple(
                    slice(index[stack_axis.index(d)], index[stack_axis.index(d)] + 1)
                    if d in stack_axis
                    else slice(None)
                    for d in range(out.ndim)
                )
            ] = cut
        out = out.assign_coords(
            {
                d: autocorrelation_lags(len(arr[d]), mode) * s
                for s, d in zip(acf_steps, acf_dims)
            }
        )
        if all(i in out.dims for i in ["kx", "ky"]):
            out = out.rename(dict(kx="qx", ky="qy"))
    return out


def match_dims(da1: xr.DataArray, da2: xr.DataArray):
    """
    Returns the second array interpolated with the coordinates of the first array,
    making them the same size.

    """
    return da2.interp({dim: da1[dim] for dim in da2.dims})


def xcorr1d(in1: xr.DataArray, in2: xr.DataArray, method="direct"):
    """Performs 1-dimensional correlation analysis on `xarray.DataArray` s."""
    in2 = match_dims(in1, in2)
    out = in1.copy(deep=True)
    xind = correlation_lags(in1.values.size, in2.values.size, mode="same")
    xzero = np.flatnonzero(xind == 0)[0]
    out.values = correlate(
        in1.fillna(0).values, in2.fillna(0).values, mode="same", method=method
    )
    out[in1.dims[0]] = out[in1.dims[0]] - out[in1.dims[0]][xzero]
    return out
