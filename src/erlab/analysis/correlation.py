"""Macros for correlation analysis."""

__all__ = ["acf2", "acf2stack", "xcorr1d"]


import itertools

import joblib
import numpy as np
import scipy.signal
import xarray as xr


def autocorrelate(arr, *args, **kwargs):
    """Calculate the autocorrelation of a N-dimensional array, normalized to 1.

    Parameters
    ----------
    arr
        Input array to calculate the autocorrelation.
    *args, **kwargs
        Additional arguments and keyword arguments to be passed to
        `scipy.signal.correlate`.

    Returns
    -------
    autocorr
        Autocorrelation of the input array.

    """
    acf = scipy.signal.correlate(arr, arr, *args, **kwargs)
    return acf / acf[tuple(s // 2 for s in acf.shape)]


def autocorrelation_lags(in_len, *args, **kwargs):
    return scipy.signal.correlation_lags(in_len, in_len, *args, **kwargs)


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
    """Calculate the autocorrelation function (ACF) of a 2D array including nans.

    Parameters
    ----------
    arr
        The input array for which the ACF needs to be calculated.
    mode
        The mode of the ACF calculation, by default ``"full"``. For more information,
        see `scipy.signal.correlate`.
    method
        The method used for ACF calculation, by default ``"fft"``. For more information,
        see `scipy.signal.correlate`.

    Returns
    -------
    xarray.DataArray
        The ACF of the input array.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> np.random.seed(0)  # Set the random seed for reproducibility
    >>> arr = xr.DataArray(np.random.rand(10, 10), dims=("kx", "ky"))
    >>> acf = acf2(arr)
    >>> acf
    <xarray.DataArray (qx: 19, qy: 19)> Size: 3kB
    8.403e-05 0.01495 0.01979 0.02734 0.03215 ... 0.02734 0.01979 0.01495 8.403e-05
    Coordinates:
    * qx       (qx) int64 152B -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9
    * qy       (qy) int64 152B -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9

    """
    out = arr.copy(deep=True)
    acf = nanacf(out.values, mode=mode, method=method)
    coords = [out[d].values for d in out.dims]
    steps = [c[1] - c[0] for c in coords]
    out = xr.DataArray(
        acf,
        {
            d: autocorrelation_lags(n, mode) * s
            for s, n, d in zip(steps, arr.shape, out.dims, strict=True)
        },
        attrs=out.attrs,
    )
    if all(i in out.dims for i in ["kx", "ky"]):
        out = out.rename({"kx": "qx", "ky": "qy"})
    return out


def acf2stack(arr, stack_dims=("eV",), mode: str = "full", method: str = "fft"):
    if arr.ndim == 2:
        return acf2(arr, mode, method)
    elif arr.ndim >= 3:
        if arr.ndim - len(stack_dims) != 2:
            raise ValueError(
                "The number of dimensions excluding the stacking dimensions must be 2"
            )

        stack_sizes = {d: len(arr[d]) for d in stack_dims}
        stack_iter = tuple(range(s) for s in stack_sizes.values())

        out_list = joblib.Parallel(n_jobs=-1, pre_dispatch="3 * n_jobs")(
            joblib.delayed(nanacf)(
                np.squeeze(arr.isel(dict(zip(stack_dims, vals, strict=True))).values),
                mode,
                method,
            )
            for vals in itertools.product(*stack_iter)
        )
        acf_dims = tuple(filter(lambda d: d not in stack_dims, arr.dims))
        acf_sizes = dict(zip(acf_dims, out_list[0].shape, strict=True))
        acf_steps = tuple(arr[d].values[1] - arr[d].values[0] for d in acf_dims)

        out_sizes = stack_sizes | acf_sizes

        if mode == "same":
            out = arr.copy(deep=True)
        else:
            out = xr.DataArray(
                np.empty(tuple(out_sizes[d] for d in arr.dims)),
                dims=arr.dims,
                attrs=arr.attrs,
            )
            out = out.assign_coords({d: arr[d] for d in stack_dims})

        for i, vals in enumerate(itertools.product(*stack_iter)):
            out.loc[{s: arr[s][v] for s, v in zip(stack_dims, vals, strict=True)}] = (
                out_list[i]
            )

        out = out.assign_coords(
            {
                d: autocorrelation_lags(len(arr[d]), mode) * s
                for s, d in zip(acf_steps, acf_dims, strict=True)
            }
        )
        if all(i in out.dims for i in ["kx", "ky"]):
            out = out.rename({"kx": "qx", "ky": "qy"})
    return out


def xcorr1d(in1: xr.DataArray, in2: xr.DataArray, method="direct"):
    """Perform 1-dimensional correlation analysis on `xarray.DataArray` s."""
    in2 = in2.interp_like(in1)
    out = in1.copy(deep=True)
    xind = scipy.signal.correlation_lags(in1.values.size, in2.values.size, mode="same")
    xzero = np.flatnonzero(xind == 0)[0]
    out.values = scipy.signal.correlate(
        in1.fillna(0).values, in2.fillna(0).values, mode="same", method=method
    )
    out[in1.dims[0]] = out[in1.dims[0]] - out[in1.dims[0]][xzero]
    return out
