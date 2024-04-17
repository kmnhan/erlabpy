__all__ = [
    "ImageToolAccessor",
    "PlotAccessor",
    "SelectionAccessor",
]

import warnings
from collections.abc import Hashable, Mapping
from typing import Any, TypeGuard, TypeVar, cast

import xarray as xr

import erlab.plotting.erplot as eplt
from erlab.interactive.imagetool import ImageTool, itool

T = TypeVar("T")

# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets, from xarray.core.dataarray
_THIS_ARRAY = "<this-array>"


class ERLabAccessor:
    """Base class for accessors."""

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj


def is_dict_like(value: Any) -> TypeGuard[Mapping[Any, Any]]:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def either_dict_or_kwargs(
    pos_kwargs: Mapping[Any, T] | None,
    kw_kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    if pos_kwargs is None or pos_kwargs == {}:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return cast(Mapping[Hashable, T], kw_kwargs)

    if not is_dict_like(pos_kwargs):
        raise ValueError(f"the first argument to .{func_name} must be a dictionary")
    if kw_kwargs:
        raise ValueError(
            f"cannot specify both keyword and positional arguments to .{func_name}"
        )
    return pos_kwargs


@xr.register_dataarray_accessor("qplot")
class PlotAccessor(ERLabAccessor):
    """`xarray.DataArray.qplot` accessor for plotting data."""

    def __call__(self, *args, **kwargs):
        """
        Plot the data. If a 2D data array is provided, it is plotted using
        :func:`plot_array <erlab.plotting.general.plot_array>`. Otherwise, it is
        equivalent to calling :meth:`xarray.DataArray.plot`.

        Parameters
        ----------
        *args
            Positional arguments to be passed to the plotting function.
        **kwargs
            Keyword arguments to be passed to the plotting function.

        """
        if len(self._obj.dims) == 2:
            return eplt.plot_array(self._obj, *args, **kwargs)
        else:
            return self._obj.plot(*args, **kwargs)


@xr.register_dataarray_accessor("qshow")
class ImageToolAccessor(ERLabAccessor):
    """`xarray.DataArray.qshow` accessor for interactive visualization."""

    def __call__(self, *args, **kwargs) -> ImageTool:
        if len(self._obj.dims) >= 2:
            return itool(self._obj, *args, **kwargs)
        else:
            raise ValueError("Data must have at leasst two dimensions.")


@xr.register_dataarray_accessor("qsel")
class SelectionAccessor(ERLabAccessor):
    """
    `xarray.DataArray.qsel` accessor for conveniently selecting and averaging
    data.
    """

    def __call__(
        self,
        indexers: dict[str, float | slice] | None = None,
        *,
        verbose: bool = False,
        **indexers_kwargs: dict[str, float | slice],
    ):
        """Select and average data along specified dimensions.

        Parameters
        ----------
        indexers
            Dictionary specifying the dimensions and their values or slices.
            Position along a dimension can be specified in three ways:

            - As a scalar value: `alpha=-1.2`

              If no width is specified, the data is selected along the nearest value. It
              is equivalent to `xarray.DataArray.sel` with `method='nearest'`.

            - As a value and width: `alpha=5, alpha_width=0.5`

              The data is *averaged* over a slice of width `alpha_width`, centered at
              `alpha`.

            - As a slice: `alpha=slice(-10, 10)`

              The data is selected over the specified slice. No averaging is performed.

            One of `indexers` or `indexers_kwargs` must be provided.
        verbose
            If `True`, print information about the selected data and averaging process.
            Default is `False`.
        **indexers_kwargs
            The keyword arguments form of `indexers`. One of `indexers` or
            `indexers_kwargs` must be provided.

        Returns
        -------
        xarray.DataArray
            The selected and averaged data.

        Raises
        ------
        ValueError
            If a specified dimension is not present in the data.
        """

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "qsel")

        # Bin widths for each dimension, zero if width not specified
        bin_widths: dict[str, float] = {}

        for dim in indexers:
            if not dim.endswith("_width"):
                bin_widths[dim] = indexers.get(f"{dim}_width", 0.0)
                if dim not in self._obj.dims:
                    raise ValueError(f"Dimension `{dim}` not found in data.")

        scalars: dict[str, float] = {}
        slices: dict[str, slice] = {}
        avg_dims: list[str] = []

        for dim, width in bin_widths.items():
            if width == 0.0:
                if isinstance(indexers[dim], slice):
                    slices[dim] = indexers[dim]
                else:
                    scalars[dim] = float(indexers[dim])
            else:
                slices[dim] = slice(
                    indexers[dim] - width / 2, indexers[dim] + width / 2
                )
                avg_dims.append(dim)

        if len(scalars) >= 1:
            for k, v in scalars.items():
                if v < self._obj[k].min() or v > self._obj[k].max():
                    warnings.warn(
                        f"Selected value {v} for `{k}` is outside coordinate bounds",
                        stacklevel=2,
                    )
            out = self._obj.sel(**scalars, method="nearest")
        else:
            out = self._obj

        if len(slices) >= 1:
            out = out.sel(**slices)

            lost_coords = {k: out[k].mean() for k in avg_dims}
            out = out.mean(dim=avg_dims, keep_attrs=True)
            out = out.assign_coords(lost_coords)

        if verbose:
            print(
                f"Selected data with {scalars} and {slices}, averaging over {avg_dims}"
            )

        return out
