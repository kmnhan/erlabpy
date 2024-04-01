__all__ = ["shift", "correct_with_edge"]

import itertools
from collections.abc import Callable

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import xarray as xr

from erlab.analysis.fit.models import FermiEdge2dModel
from erlab.plotting.colors import proportional_colorbar
from erlab.plotting.general import plot_array


def shift(
    darr: xr.DataArray,
    shift: float | xr.DataArray,
    along: str,
    shift_coords: bool = False,
    **shift_kwargs,
) -> xr.DataArray:
    """Shifts the values of a DataArray along a specified dimension.

    Parameters
    ----------
    darr
        The array to shift.
    shift
        The amount of shift to be applied along the specified dimension. If `shift` is a
        DataArray, different shifts can be applied to different coordinates. The
        dimensions of `shift` must be a subset of the dimensions of `darr`. For more
        information, see the note below. If `shift` is a `float`, the same shift is
        applied to all values along dimension `along`. This is equivalent to providing a
        0-dimensional DataArray.
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
    >>> shifted = erlab.analysis.utilities.shift(darr, shift_arr, along="y")
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

    domain_indices: list[int] = [darr.get_axis_num(ax) for ax in shift.dims]

    # `along` must be evenly spaced and monotonic increasing
    out = darr.sortby(along).copy()

    # Normalize shift values
    along_step: float = out[along].values[1] - out[along].values[0]
    shift = (shift.copy() / along_step).fillna(0.0)

    if shift_coords:
        # We first apply the integer part of the average shift to the coords
        rigid_shift = np.round(shift.values.mean())
        shift = shift - rigid_shift

        # Apply coordinate shift
        out = out.assign_coords({along: out[along].values + rigid_shift * along_step})

        # The bounds of the remaining shift values are used to pad the data
        nshift_min, nshift_max = shift.values.min(), shift.values.max()
        pads: tuple[int] = min(0, round(nshift_min)), max(0, round(nshift_max))

        # Construct new coordinate array
        new_along = np.linspace(
            out[along].values[0] + pads[0] * along_step,
            out[along].values[-1] + pads[1] * along_step,
            out[along].size + sum(np.abs(pads)),
        )

        # Pad the data and assign new coordinates
        out = out.pad({along: np.abs(pads)}, mode="constant", constant_values=np.nan)
        out = out.assign_coords({along: new_along})

    for idxs in itertools.product(*[range(darr.shape[i]) for i in domain_indices]):
        # Construct slices for indexing
        slices = [slice(None)] * darr.ndim
        for domain_index, i in zip(domain_indices, idxs):
            slices[domain_index] = i
        slices = tuple(slices)

        # Initialize arguments to `scipy.ndimage.shift`
        input = out[slices]
        shifts = [0] * input.ndim
        shift_val: float = float(shift.isel(dict(zip(shift.dims, idxs))))
        shifts[input.get_axis_num(along)] = shift_val

        # Apply shift
        out[slices] = scipy.ndimage.shift(input.values, shifts, **shift_kwargs)

    return out


def correct_with_edge(
    darr: xr.DataArray,
    modelresult: lmfit.model.ModelResult | npt.NDArray[np.floating] | Callable,
    shift_coords: bool = True,
    plot: bool = False,
    plot_kw: dict | None = None,
    **shift_kwargs,
):
    """
    Corrects the given data array `darr` using the edge correction method.

    Parameters
    ----------
    darr
        The input data array to be corrected.
    modelresult
        The model result that contains the fermi edge information. It can be an instance
        of `lmfit.model.ModelResult`, a numpy array, or a callable function that takes
        an array of angles and returns the corresponding energy value.
    shift_coords
        Whether to shift the coordinates of the data array. Defaults to True.
    plot
        Whether to plot the original and corrected data arrays. Defaults to False.
    plot_kw
        Additional keyword arguments for the plot. Defaults to None.
    **shift_kwargs
        Additional keyword arguments to `shift`.

    Returns
    -------
    xarray.DataArray
        The edge corrected data.
    """
    if plot_kw is None:
        plot_kw = {}

    if isinstance(modelresult, lmfit.model.ModelResult):
        if isinstance(modelresult.model, FermiEdge2dModel):
            edge_quad = np.polynomial.polynomial.polyval(
                darr.alpha,
                np.array(
                    [
                        modelresult.best_values[f"c{i}"]
                        for i in range(modelresult.model.func.poly.degree + 1)
                    ]
                ),
            )
        else:
            edge_quad = modelresult.eval(x=darr.alpha)

    elif callable(modelresult):
        edge_quad = modelresult(darr.alpha.values)

    elif isinstance(modelresult, np.ndarray | xr.DataArray):
        if len(darr.alpha) != len(modelresult):
            raise ValueError(
                "Length of modelresult must be equal to the length of alpha in data"
            )
        else:
            edge_quad = modelresult

    else:
        raise ValueError(
            "modelresult must be one of "
            "lmfit.model.ModelResult, "
            "and np.ndarray or a callable"
        )

    if isinstance(edge_quad, np.ndarray):
        edge_quad = xr.DataArray(
            edge_quad, coords={"alpha": darr.alpha}, dims=["alpha"]
        )

    corrected = shift(darr, -edge_quad, "eV", shift_coords=shift_coords, **shift_kwargs)

    if plot is True:
        _, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))

        plot_kw.setdefault("cmap", "copper")
        plot_kw.setdefault("gamma", 0.5)

        if darr.ndim > 2:
            avg_dims = list(darr.dims)[:]
            avg_dims.remove("alpha")
            avg_dims.remove("eV")
            plot_array(darr.mean(avg_dims), ax=axes[0], **plot_kw)
            plot_array(corrected.mean(avg_dims), ax=axes[1], **plot_kw)
        else:
            plot_array(darr, ax=axes[0], **plot_kw)
            plot_array(corrected, ax=axes[1], **plot_kw)
        edge_quad.plot(ax=axes[0], ls="--", color="0.35")

        proportional_colorbar(ax=axes[0])
        proportional_colorbar(ax=axes[1])
        axes[0].set_title("Data")
        axes[1].set_title("Edge Corrected")

    return corrected
