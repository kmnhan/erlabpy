import xarray as xr

try:
    import csaps
except ImportError as e:
    raise ImportError(
        "`erlab.analysis.fit.spline` requires `csaps` to be installed."
    ) from e


def xcsaps(arr: xr.DataArray, **kwargs) -> tuple[xr.DataArray, csaps.ISmoothingSpline]:
    """`xarray` compatible `csaps.csaps`.

    Parameters
    ----------
    arr
        Input array for smoothing spline calculation.
    **kwargs
        Keyword arguments for :func:`csaps.csaps`. `normalizedsmooth` is set to `True`
        by default.

    Returns
    -------
    out : xarray.DataArray
        Smoothing spline evaluated at `arr` coordinates.
    spl : csaps.ISmoothingSpline
        The spline object.
    """
    kwargs.setdefault("normalizedsmooth", True)
    coords = [arr[d].values for d in arr.dims]
    spl = csaps.csaps(coords, arr.values, **kwargs)
    out = xr.DataArray(
        data=spl(coords),
        dims=arr.dims,
        coords={arr.dims[i]: coords[i] for i in range(arr.ndim)},
    )
    return out, spl


# def smoothing_spline(arr:xr.DataArray, dim:):

# return scipy.interpolate.make_smoothing_spline()
