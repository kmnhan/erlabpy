__all__ = [
    "ImageToolAccessor",
    "PlotAccessor",
    "SelectionAccessor",
]

import warnings
from collections.abc import Hashable, Mapping
from typing import Any, TypeGuard, TypeVar, cast

import matplotlib.pyplot as plt
import xarray as xr

import erlab.plotting.erplot as eplt

T = TypeVar("T")

# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets, from xarray.core.dataarray
_THIS_ARRAY: str = "<this-array>"


class ERLabDataArrayAccessor:
    """Base class for accessors."""

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj


class ERLabDatasetAccessor:
    """Base class for accessors."""

    def __init__(self, xarray_obj: xr.Dataset):
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
class PlotAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.qplot` accessor for plotting data."""

    def __call__(self, *args, **kwargs):
        """Plot the data.

        If a 2D data array is provided, it is plotted using :func:`plot_array
        <erlab.plotting.general.plot_array>`. Otherwise, it is equivalent to calling
        :meth:`xarray.DataArray.plot`.

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
            ax = kwargs.pop("ax", None)
            if ax is None:
                ax = plt.gca()
            kwargs["ax"] = ax

            out = self._obj.plot(*args, **kwargs)
            eplt.fancy_labels(ax)
            return out


@xr.register_dataarray_accessor("qshow")
class ImageToolAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.qshow` accessor for interactive visualization."""

    def __call__(self, *args, **kwargs):
        from erlab.interactive.imagetool import itool

        if len(self._obj.dims) >= 2:
            return itool(self._obj, *args, **kwargs)
        else:
            raise ValueError("Data must have at leasst two dimensions.")


@xr.register_dataarray_accessor("qsel")
class SelectionAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.qsel` accessor for convenient selection and averaging."""

    def __call__(
        self,
        indexers: Mapping[Hashable, float | slice] | None = None,
        *,
        verbose: bool = False,
        **indexers_kwargs,
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
        bin_widths: dict[Hashable, float] = {}

        for dim in indexers:
            if not str(dim).endswith("_width"):
                width = indexers.get(f"{dim}_width", 0.0)
                if isinstance(width, slice):
                    raise ValueError(
                        f"Slice not allowed for width of dimension `{dim}`"
                    )
                else:
                    bin_widths[dim] = float(width)
                if dim not in self._obj.dims:
                    raise ValueError(f"Dimension `{dim}` not found in data.")

        scalars: dict[Hashable, float] = {}
        slices: dict[Hashable, slice] = {}
        avg_dims: list[Hashable] = []

        for dim, width in bin_widths.items():
            value = indexers[dim]

            if width == 0.0:
                if isinstance(value, slice):
                    slices[dim] = value
                else:
                    scalars[dim] = float(value)
            else:
                if isinstance(value, slice):
                    raise ValueError(
                        f"Slice not allowed for value of dimension `{dim}` "
                        "with width specified"
                    )
                slices[dim] = slice(value - width / 2, value + width / 2)
                avg_dims.append(dim)

        if len(scalars) >= 1:
            for k, v in scalars.items():
                if v < self._obj[k].min() or v > self._obj[k].max():
                    warnings.warn(
                        f"Selected value {v} for `{k}` is outside coordinate bounds",
                        stacklevel=2,
                    )
            out = self._obj.sel(
                {str(k): v for k, v in scalars.items()}, method="nearest"
            )
        else:
            out = self._obj

        if len(slices) >= 1:
            out = out.sel(slices)

            lost_coords = {k: out[k].mean() for k in avg_dims}
            out = out.mean(dim=avg_dims, keep_attrs=True)
            out = out.assign_coords(lost_coords)

        if verbose:
            print(
                f"Selected data with {scalars} and {slices}, averaging over {avg_dims}"
            )

        return out
