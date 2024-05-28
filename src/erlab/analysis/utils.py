__all__ = ["shift"]

import itertools
import warnings
from typing import cast

import numpy as np
import scipy.ndimage
import xarray as xr


def shift(
    darr: xr.DataArray,
    shift: float | xr.DataArray,
    along: str,
    shift_coords: bool = False,
    **shift_kwargs,
) -> xr.DataArray:
    """Shifts the values of a DataArray along a single dimension.

    The shift is applied using `scipy.ndimage.shift` with the specified keyword
    arguments. Linear interpolation is used by default.

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
    >>> darr = xr.DataArray(
    ...     np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(float), dims=["x", "y"]
    ... )
    >>> shift_arr = xr.DataArray([1, 0, 2], dims=["x"])
    >>> shifted = erlab.analysis.utils.shift(darr, shift_arr, along="y")
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
        rigid_shift: float = np.round(shift.values.mean())
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
        input = out[slices]
        shifts: list[float] = [0.0] * input.ndim
        shift_val: float = float(shift.isel(dict(zip(shift.dims, idxs, strict=True))))
        shifts[cast(int, input.get_axis_num(along))] = shift_val

        # Apply shift
        out[slices] = scipy.ndimage.shift(input.values, shifts, **shift_kwargs)

    return out


def correct_with_edge(*args, **kwargs):
    from erlab.analysis.gold import correct_with_edge

    warnings.warn(
        "erlab.analysis.utils.correct_with_edge is deprecated, "
        "use erlab.analysis.gold.correct_with_edge instead",
        DeprecationWarning,
        stacklevel=1,
    )
    return correct_with_edge(*args, **kwargs)
