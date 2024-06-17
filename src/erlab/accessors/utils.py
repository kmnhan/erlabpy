__all__ = [
    "InteractiveDataArrayAccessor",
    "InteractiveDatasetAccessor",
    "PlotAccessor",
    "SelectionAccessor",
]

import importlib
import warnings
from collections.abc import Hashable, Mapping
from typing import Any, TypeGuard, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
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
class InteractiveDataArrayAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.qshow` accessor for interactive visualization."""

    def __call__(self, *args, **kwargs):
        """Visualize the data interactively.

        Chooses the appropriate interactive visualization method based on the number of
        dimensions in the data.

        Parameters
        ----------
        *args
            Positional arguments passed onto the interactive visualization function.
        **kwargs
            Keyword arguments passed onto the interactive visualization function.
        """
        if self._obj.ndim >= 2 and self._obj.ndim <= 4:
            return self.itool(*args, **kwargs)
        else:
            if importlib.util.find_spec("hvplot"):
                self.hvplot(*args, **kwargs)
            else:
                raise ValueError("Data must have at least two dimensions.")

    def itool(self, *args, **kwargs):
        """Shortcut for :func:`itool <erlab.interactive.imagetool.itool>`.

        Parameters
        ----------
        *args
            Positional arguments passed onto :func:`itool
            <erlab.interactive.imagetool.itool>`.
        **kwargs
            Keyword arguments passed onto :func:`itool
            <erlab.interactive.imagetool.itool>`.

        """
        from erlab.interactive.imagetool import itool

        return itool(self._obj, *args, **kwargs)

    def hvplot(self, *args, **kwargs):
        """:mod:`hvplot`-based interactive visualization.

        Parameters
        ----------
        *args
            Positional arguments passed onto :meth:`DataArray.hvplot`.
        **kwargs
            Keyword arguments passed onto :meth:`DataArray.hvplot`.

        Raises
        ------
        ImportError
            If :mod:`hvplot` is not installed.
        """
        if not importlib.util.find_spec("hvplot"):
            raise ImportError("hvplot is required for qshow.hvplot()")
        import hvplot.xarray  # noqa: F401

        return self._obj.hvplot(*args, **kwargs)


@xr.register_dataset_accessor("qshow")
class InteractiveDatasetAccessor(ERLabDatasetAccessor):
    """`xarray.Dataset.qshow` accessor for interactive visualization."""

    def __call__(self, *args, **kwargs):
        """Visualize the data interactively.

        Chooses the appropriate interactive visualization method based on the data
        variables.

        Parameters
        ----------
        *args
            Positional arguments passed onto the interactive visualization function.
        **kwargs
            Keyword arguments passed onto the interactive visualization function.
        """
        if self._is_fitresult:
            return self.fit(*args, **kwargs)
        else:
            return self.itool(*args, **kwargs)

    @property
    def _is_fitresult(self) -> bool:
        from erlab.accessors.fit import ParallelFitDataArrayAccessor

        for var in set(ParallelFitDataArrayAccessor._VAR_KEYS) - {"modelfit_results"}:
            if var not in self._obj.data_vars:
                return False
        return True

    def itool(self, *args, **kwargs):
        from erlab.interactive.imagetool import itool

        return itool(self._obj, *args, **kwargs)

    def hvplot(self, *args, **kwargs):
        if not importlib.util.find_spec("hvplot"):
            raise ImportError("hvplot is required for qshow.hvplot()")
        import hvplot.xarray  # noqa: F401

        return self._obj.hvplot(*args, **kwargs)

    itool.__doc__ = InteractiveDataArrayAccessor.itool.__doc__
    hvplot.__doc__ = str(InteractiveDataArrayAccessor.hvplot.__doc__).replace(
        "DataArray", "Dataset"
    )

    def fit(self, plot_components: bool = False):
        """Interactive visualization of fit results.

        Parameters
        ----------
        plot_components
            If `True`, plot the components of the fit. Default is `False`. Requires the
            Dataset to have a `modelfit_results` variable.

        Returns
        -------
        :class:`panel.Column`
            A panel containing the interactive visualization.
        """
        if not importlib.util.find_spec("hvplot"):
            raise ImportError("hvplot is required for interactive fit visualization")

        import hvplot.xarray
        import panel
        import panel.widgets

        from erlab.accessors.fit import ParallelFitDataArrayAccessor

        for var in set(ParallelFitDataArrayAccessor._VAR_KEYS) - {"modelfit_results"}:
            if var not in self._obj.data_vars:
                raise ValueError("Dataset is not a fit result")

        coord_dims = [
            d
            for d in self._obj.modelfit_stats.dims
            if d in self._obj.modelfit_data.dims
        ]
        other_dims = [d for d in self._obj.modelfit_data.dims if d not in coord_dims]

        if len(other_dims) != 1:
            raise ValueError("Only 1D fits are supported")

        sliders = [
            panel.widgets.DiscreteSlider(name=d, options=list(np.array(self._obj[d])))
            for d in coord_dims
        ]

        def get_slice(*s):
            return self._obj.sel(dict(zip(coord_dims, s, strict=False)))

        def get_slice_params(*s):
            res_part = get_slice(*s).rename(param="Parameter")
            return xr.merge(
                [
                    res_part.modelfit_coefficients.rename("Value"),
                    res_part.modelfit_stderr.rename("Stderr"),
                ]
            )

        def get_comps(*s):
            partial_res = get_slice(*s)
            return xr.merge(
                [
                    xr.DataArray(
                        v, dims=other_dims, coords=[self._obj[other_dims[0]]]
                    ).rename(k)
                    for k, v in partial_res.modelfit_results.item()
                    .eval_components()
                    .items()
                ]
                + [
                    partial_res.modelfit_data,
                    partial_res.modelfit_best_fit,
                ]
            )

        part = hvplot.bind(get_slice, *sliders).interactive()
        part_params = hvplot.bind(get_slice_params, *sliders).interactive()

        if "modelfit_results" not in self._obj.data_vars:
            warnings.warn(
                "`model_results` not included in Dataset. "
                "Components will not be plotted",
                stacklevel=2,
            )
            plot_components = False

        plot_kwargs = {
            "responsive": True,
            "min_width": 400,
            "min_height": 500,
            "title": "",
        }
        if plot_components:
            part_comps = hvplot.bind(get_comps, *sliders).interactive()
            data = part_comps.modelfit_data.hvplot.scatter(**plot_kwargs)
            fit = part_comps.modelfit_best_fit.hvplot(c="k", ylabel="", **plot_kwargs)
            components = part_comps.hvplot(
                y=list(
                    self._obj.modelfit_results.values.flatten()[0]
                    .eval_components()
                    .keys()
                ),
                legend="top_right",
                group_label="Component",
                **plot_kwargs,
            )
            plots = components * data * fit
        else:
            data = part.modelfit_data.hvplot.scatter(**plot_kwargs)
            fit = part.modelfit_best_fit.hvplot(c="k", ylabel="", **plot_kwargs)
            plots = data * fit

        return panel.Column(
            plots,
            panel.Spacer(height=30),
            panel.Tabs(
                (
                    "Parameters",
                    part_params.hvplot.table(
                        columns=["Parameter", "Value", "Stderr"],
                        title="",
                        responsive=True,
                    ),
                ),
                (
                    "Fit statistics",
                    part.modelfit_stats.hvplot.table(
                        columns=["fit_stat", "modelfit_stats"],
                        title="",
                        responsive=True,
                    ),
                ),
            ),
        )


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

    def around(self, radius: float | dict[Hashable, float], **sel_kw) -> xr.DataArray:
        """
        Average data within a specified radius of a specified point.

        For instance, consider an ARPES map with dimensions ``kx``, ``ky``, and ``eV``.
        Providing ``kx`` and ``ky`` points will average the data within a cylindrical
        region centered at that point. The radius of the cylinder is specified by the
        ``radius`` parameter. If different radii is given for ``kx`` and ``ky``, the
        region will be an elliptic cylinder.

        Parameters
        ----------
        radius
            The radius of the region. If a single number, the same radius is used for
            all dimensions. If a dictionary, each value
        **sel_kw
            The center of the spherical region. Must be a mapping of valid dimension
            names to coordinate values.

        Returns
        -------
        xr.DataArray
            The mean value of the data within the region.

        Note
        ----
        The region is defined by a spherical mask, which is generated with
        `erlab.analysis.mask.spherical_mask`. Depending on the radius and dimensions
        provided, the mask will be hyperellipsoid in the dimensions specified in
        `sel_kw`.

        """
        import erlab.analysis

        return self._obj.where(
            erlab.analysis.mask.spherical_mask(self._obj, radius, **sel_kw), drop=True
        ).mean(sel_kw.keys())
