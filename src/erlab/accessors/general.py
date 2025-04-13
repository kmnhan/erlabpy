"""Defines miscellaneous accessors for general data manipulation and visualization."""

__all__ = [
    "InfoDataArrayAccessor",
    "InteractiveDataArrayAccessor",
    "InteractiveDatasetAccessor",
    "PlotAccessor",
    "SelectionAccessor",
]

import functools
import importlib
import typing
from collections.abc import Collection, Hashable, Mapping

import numpy as np
import xarray as xr

import erlab
from erlab.accessors.utils import (
    ERLabDataArrayAccessor,
    ERLabDatasetAccessor,
    either_dict_or_kwargs,
)

if typing.TYPE_CHECKING:
    import lmfit


def _check_hvplot():
    """Check if hvplot is installed and raise an ImportError if not."""
    if not importlib.util.find_spec("hvplot"):
        raise ImportError(
            "The hvplot package is required to visualize this data interactively. "
        )


@xr.register_dataarray_accessor("qplot")
class PlotAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.qplot` accessor for plotting data."""

    def __call__(self, *args, **kwargs):
        """Plot the data.

        Plots two-dimensional data using :func:`plot_array
        <erlab.plotting.general.plot_array>`. For non-two-dimensional data, the method
        falls back to :meth:`xarray.DataArray.plot`.

        Also sets fancy labels using :func:`fancy_labels
        <erlab.plotting.annotations.fancy_labels>`.

        Parameters
        ----------
        *args
            Positional arguments to be passed to the plotting function.
        **kwargs
            Keyword arguments to be passed to the plotting function.

        """
        import matplotlib.pyplot

        if len(self._obj.dims) == 2:
            return erlab.plotting.plot_array(self._obj, *args, **kwargs)

        ax = kwargs.pop("ax", None)
        if ax is None:
            ax = matplotlib.pyplot.gca()
        kwargs["ax"] = ax

        out = self._obj.plot(*args, **kwargs)
        erlab.plotting.fancy_labels(ax)
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

        _check_hvplot()
        return self.hvplot(*args, **kwargs)

    def itool(self, *args, **kwargs):
        """Shortcut for :func:`erlab.interactive.imagetool.itool`.

        Parameters
        ----------
        *args
            Positional arguments passed onto :func:`itool
            <erlab.interactive.imagetool.itool>`.
        **kwargs
            Keyword arguments passed onto :func:`itool
            <erlab.interactive.imagetool.itool>`.

        """
        return erlab.interactive.itool(self._obj, *args, **kwargs)

    def hvplot(self, *args, **kwargs):
        """`hvplot <https://hvplot.holoviz.org/>`_-based interactive visualization.

        This method is a convenience wrapper that handles importing ``hvplot``.

        Parameters
        ----------
        *args
            Positional arguments passed onto ``DataArray.hvplot``.
        **kwargs
            Keyword arguments passed onto ``DataArray.hvplot``.

        Raises
        ------
        ImportError
            If `hvplot <https://hvplot.holoviz.org/>`_ is not installed.
        """
        _check_hvplot()
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
        return self.itool(*args, **kwargs)

    @property
    def _is_fitresult(self) -> bool:
        """Check if the Dataset is a fit result."""
        from erlab.accessors.fit import ParallelFitDataArrayAccessor

        all_keys = set(ParallelFitDataArrayAccessor._VAR_KEYS) - {"modelfit_results"}
        for k in self._obj.data_vars:
            for var in list(all_keys):
                if str(k).endswith(var):
                    all_keys.remove(var)
                    break
            if len(all_keys) == 0:
                return True
        return False

    @property
    def _fitresult_data_vars(self) -> list[str]:
        """Name of original data variables in a fit result Dataset.

        If the Dataset is not a fit result or is a fit to a DataArray, an empty list is
        returned.
        """
        if "modelfit_data" in self._obj.data_vars:
            return []
        return [
            str(k).removesuffix("modelfit_data").rstrip("_")
            for k in self._obj.data_vars
            if str(k).endswith("modelfit_data")
        ]

    def itool(self, *args, **kwargs):
        return erlab.interactive.itool(self._obj, *args, **kwargs)

    def hvplot(self, *args, **kwargs):
        _check_hvplot()
        import hvplot.xarray  # noqa: F401

        return self._obj.hvplot(*args, **kwargs)

    itool.__doc__ = InteractiveDataArrayAccessor.itool.__doc__
    hvplot.__doc__ = str(InteractiveDataArrayAccessor.hvplot.__doc__).replace(
        "DataArray", "Dataset"
    )

    def _determine_prefix(self, data_var: str | None) -> str:
        """Determine the prefix for fit results.

        Parameters
        ----------
        data_var
            The name of the data variable to visualize. Required only if the fit result
            dataset is a result of fitting to a Dataset with multiple data variables.

        Returns
        -------
        prefix : str
            The prefix for the fit results. If the fit result is from a fit to a
            DataArray, an empty string is returned. Otherwise, the targeted data
            variable followed by an underscore is returned.
        """
        all_data_vars: list[str] = self._fitresult_data_vars

        if data_var is not None and data_var not in all_data_vars:
            raise ValueError(
                f"Fit results for data variable `{data_var}` "
                "were not found in the Dataset."
            )

        if len(all_data_vars) == 1:
            data_var = all_data_vars[0]

        if "modelfit_results" not in self._obj.data_vars and data_var is None:
            raise ValueError(
                "Dataset contains fits from multiple data variables. "
                "Provide the `data_var` argument to select the variable to visualize."
            )

        return "" if data_var is None else f"{data_var}_"

    def fit(self, plot_components: bool = False, data_var: str | None = None):
        """Interactive visualization of fit results.

        Parameters
        ----------
        plot_components
            If `True`, plot the components of the fit. Default is `False`. Requires the
            Dataset to have a `modelfit_results` variable.
        data_var
            The name of the data variable to visualize. Required only if the Dataset
            contains fits across multiple data variables.

        Returns
        -------
        :class:`panel.layout.Column`
            A panel containing the interactive visualization.
        """
        _check_hvplot()

        import hvplot.xarray
        import panel
        import panel.widgets

        if not self._is_fitresult:
            raise ValueError("Dataset is not a fit result")

        prefix = self._determine_prefix(data_var)

        coord_dims = [
            d
            for d in self._obj[f"{prefix}modelfit_stats"].dims
            if d in self._obj[f"{prefix}modelfit_data"].dims
        ]
        other_dims = [
            d for d in self._obj[f"{prefix}modelfit_data"].dims if d not in coord_dims
        ]

        if len(other_dims) != 1:
            raise ValueError("Only 1D fits are supported")

        sliders = [
            panel.widgets.DiscreteSlider(name=d, options=list(np.array(self._obj[d])))
            for d in coord_dims
        ]

        if plot_components:
            # Plot correctly across different models
            all_comps: list[str] = []
            for res in self._obj[f"{prefix}modelfit_results"].values.flat:
                for comp in res.eval_components():
                    if comp not in all_comps:
                        all_comps.append(comp)

        def get_slice(*s) -> xr.Dataset:
            return self._obj.sel(dict(zip(coord_dims, s, strict=False)))

        def get_slice_params(*s) -> xr.Dataset:
            res_part = get_slice(*s).rename(param="Parameter")
            return xr.merge(
                [
                    res_part[f"{prefix}modelfit_coefficients"].rename("Value"),
                    res_part[f"{prefix}modelfit_stderr"].rename("Stderr"),
                ]
            )

        def get_comps(*s) -> xr.Dataset:
            partial_res = get_slice(*s)
            mod_res: lmfit.model.ModelResult = partial_res[
                f"{prefix}modelfit_results"
            ].item()

            main_coord = mod_res.userkws[mod_res.model.independent_vars[0]]

            main_coord = xr.DataArray(
                main_coord, dims=[other_dims[0]], coords=[main_coord]
            )

            components = mod_res.eval_components()

            component_arrays = []
            for dim in all_comps:
                comp_arr = components.get(dim, None)
                if comp_arr is None:
                    comp_arr = np.empty_like(main_coord)
                    comp_arr.fill(np.nan)

                component_arrays.append(
                    xr.DataArray(comp_arr, dims=other_dims, coords=[main_coord]).rename(
                        dim
                    )
                )

            return xr.merge(
                [
                    *component_arrays,
                    partial_res[f"{prefix}modelfit_data"],
                    partial_res[f"{prefix}modelfit_best_fit"],
                ]
            )

        part = hvplot.bind(get_slice, *sliders).interactive()
        part_params = hvplot.bind(get_slice_params, *sliders).interactive()

        if f"{prefix}modelfit_results" not in self._obj.data_vars:
            erlab.utils.misc.emit_user_level_warning(
                "`modelfit_results` not included in Dataset. "
                "Components will not be plotted"
            )
            plot_components = False

        plot_kwargs = {
            "responsive": True,
            "min_width": 400,
            "min_height": 500,
            "title": "" if data_var is None else data_var,
        }
        if plot_components:
            part_comps = hvplot.bind(get_comps, *sliders).interactive()
            data = part_comps[f"{prefix}modelfit_data"].hvplot.scatter(**plot_kwargs)
            fit = part_comps[f"{prefix}modelfit_best_fit"].hvplot(
                c="k", ylabel="", **plot_kwargs
            )
            components = part_comps.hvplot(
                y=all_comps, legend="top_right", group_label="Component", **plot_kwargs
            )
            plots = components * data * fit
        else:
            data = part[f"{prefix}modelfit_data"].hvplot.scatter(**plot_kwargs)
            fit = part[f"{prefix}modelfit_best_fit"].hvplot(
                c="k", ylabel="", **plot_kwargs
            )
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
                    part[f"{prefix}modelfit_stats"].hvplot.table(
                        columns=["fit_stat", f"{prefix}modelfit_stats"],
                        title="",
                        responsive=True,
                    ),
                ),
            ),
        )

    def params(self, data_var: str | None = None):
        """Plot the coefficients and standard errors for each fitting parameter.

        Parameters
        ----------
        data_var
            The name of the data variable to visualize. Required only if the Dataset
            contains fits across multiple data variables.

        """
        if not self._is_fitresult:
            raise ValueError("Dataset is not a fit result")

        _check_hvplot()

        prefix = self._determine_prefix(data_var)

        import hvplot.xarray
        import panel
        import panel.widgets

        def _select_param(d):
            part = self._obj.sel(param=d)
            return xr.merge(
                [
                    part[f"{prefix}modelfit_coefficients"],
                    part[f"{prefix}modelfit_stderr"],
                ]
            )

        coord_dims = [
            d
            for d in self._obj[f"{prefix}modelfit_stats"].dims
            if d in self._obj[f"{prefix}modelfit_data"].dims
        ]
        if len(coord_dims) != 1:
            raise ValueError(
                "Parameters can only be plotted for fits along one dimension"
            )

        param_sel = panel.widgets.Select(
            name="Parameter", options=list(self._obj.param.values)
        )
        sliced = hvplot.bind(_select_param, param_sel).interactive()

        plot_kw = {"responsive": True, "min_height": 400, "title": ""}
        sc = sliced.hvplot.scatter(
            x=coord_dims[0], y=f"{prefix}modelfit_coefficients", **plot_kw
        )
        err = sliced.hvplot.errorbars(
            x=coord_dims[0],
            y=f"{prefix}modelfit_coefficients",
            yerr1=f"{prefix}modelfit_stderr",
            **plot_kw,
        )

        return sc * err


@xr.register_dataarray_accessor("qsel")
class SelectionAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.qsel` accessor for convenient selection and averaging."""

    @staticmethod
    def _qsel_scalar(
        obj: xr.DataArray,
        scalars: dict[Hashable, float | Collection[float]],
        slices: dict[Hashable, slice],
        avg_dims: list[Hashable],
        lost_dims: list[Hashable],
    ) -> xr.DataArray:
        unindexed_dims: list[Hashable] = [
            k for k in slices | scalars if k not in obj.indexes
        ]  # Unindexed dimensions, i.e. dimensions without coordinates

        if len(unindexed_dims) >= 1:
            out = obj.assign_coords(
                {k: np.arange(obj.sizes[k]) for k in unindexed_dims}
            )  # Assign temporary coordinates
        else:
            out = obj

        if len(scalars) >= 1:
            for k, v in scalars.items():
                if not isinstance(v, Collection) and (
                    v < out[k].min() or v > out[k].max()
                ):
                    erlab.utils.misc.emit_user_level_warning(
                        f"Selected value {v} for `{k}` is outside coordinate bounds"
                    )
            out = out.sel({str(k): v for k, v in scalars.items()}, method="nearest")

        if len(slices) >= 1:
            out = out.sel(slices)

            lost_coords = {
                k: out[k].mean(keep_attrs=True)
                for k in lost_dims
                if k not in unindexed_dims
            }
            out = out.mean(dim=avg_dims, keep_attrs=True)
            out = out.assign_coords(lost_coords)

        return out.drop_vars(unindexed_dims, errors="ignore")

    def __call__(
        self,
        indexers: Mapping[Hashable, float | slice] | None = None,
        **indexers_kwargs,
    ):
        """Select and average data along specified dimensions.

        Parameters
        ----------
        indexers
            Dictionary specifying the dimensions and their values or slices. Position
            along a dimension can be specified in three ways:

            - As a scalar value or a collection of scalar values: ``alpha=-1.2`` or
              ``alpha=[-1.2, 0.0, 1.2]``:

              If no width is specified, the data is selected along the nearest value for
              each element. It is equivalent to calling :meth:`xarray.DataArray.sel`
              with ``method='nearest'``.

            - As a value and width: ``alpha=5, alpha_width=0.5``

              The data is *averaged* over a slice of width ``alpha_width``, centered at
              ``alpha``. If ``alpha`` is a collection, the data is averaged over
              multiple slices and concatenated along the dimension.

            - As a slice: ``alpha=slice(-10, 10)``

              The data is selected over the specified slice. No averaging is performed.
              This is equivalent to calling :meth:`xarray.DataArray.sel` with a slice.

            One of ``indexers`` or ``indexers_kwargs`` must be provided.
        **indexers_kwargs
            The keyword arguments form of ``indexers``. One of ``indexers`` or
            ``indexers_kwargs`` must be provided.

        Returns
        -------
        DataArray
            The selected and averaged data.

        Note
        ----
        Unlike :meth:`xarray.DataArray.sel`, this method treats all dimensions without
        coordinates as equivalent to having coordinates assigned from 0 to ``n-1``,
        where ``n`` is the size of the dimension. For example:

        .. code-block:: python

            da = xr.DataArray(np.random.rand(10), dims=("x",))

            da.sel(x=slice(2, 3))  # This works

            da.sel(x=slice(2.0, 3.0))  # This raises a TypeError

            da.qsel(x=slice(2.0, 3.0))  # This works

        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "qsel")

        coord_order = list(self._obj.coords.keys())

        # Bin widths for each dimension, zero if width not specified
        bin_widths: dict[Hashable, float] = {}

        for dim in indexers:
            if not str(dim).endswith("_width"):
                width = indexers.get(f"{dim}_width", 0.0)
                if isinstance(width, slice):
                    raise ValueError(
                        f"Slice not allowed for width of dimension `{dim}`"
                    )

                bin_widths[dim] = float(width)
                if dim not in self._obj.dims:
                    raise ValueError(f"Dimension `{dim}` not found in data")
            else:
                target_dim = str(dim).removesuffix("_width")
                if target_dim not in indexers:
                    raise ValueError(
                        f"`{target_dim}_width` was specified without `{target_dim}`"
                    )

        scalars: dict[Hashable, float | Collection[float]] = {}
        slices: dict[Hashable, slice] = {}
        avg_dims: list[Hashable] = []
        lost_dims: list[Hashable] = []
        slice_collections: dict[Hashable, Collection[slice]] = {}

        for dim, width in bin_widths.items():
            value = indexers[dim]

            if width == 0.0:
                if isinstance(value, slice):
                    slices[dim] = value
                else:
                    scalars[dim] = value
            else:
                if isinstance(value, slice):
                    raise ValueError(
                        f"Slice not allowed for value of dimension `{dim}` "
                        "with width specified"
                    )
                if isinstance(value, Collection):
                    # Given a list of center values, create a list of slices
                    slice_collections[dim] = [
                        slice(v - width / 2.0, v + width / 2.0) for v in value
                    ]
                else:
                    slices[dim] = slice(value - width / 2.0, value + width / 2.0)

                    avg_dims.append(dim)
                    for k, v in self._obj.coords.items():
                        if dim in v.dims:
                            lost_dims.append(k)

        if len(slice_collections) >= 1:
            # Sort the slice collections by reverse dimension order to ensure that
            # non-dimension coords are also concatenated in the correct order
            slice_collections = {
                k: slice_collections[k]
                for k in sorted(
                    slice_collections, key=lambda k: -self._obj.dims.index(k)
                )
            }

            out = self._obj
            for dim, slice_list in slice_collections.items():
                lost_dims_extend: list[Hashable] = [
                    k for k, v in self._obj.coords.items() if dim in v.dims
                ]

                out = xr.concat(
                    [
                        self._qsel_scalar(
                            out,
                            scalars,
                            slices | {dim: s},
                            [*avg_dims, dim],
                            lost_dims + lost_dims_extend,
                        )
                        for s in slice_list
                    ],
                    dim=dim,
                )
        else:
            out = self._qsel_scalar(self._obj, scalars, slices, avg_dims, lost_dims)

        return erlab.utils.array.sort_coord_order(
            out, keys=coord_order, dims_first=True
        )

    def average(self, dim: str | Collection[Hashable]) -> xr.DataArray:
        """Average the data along the specified dimension(s).

        The difference between this method and :meth:`xarray.DataArray.mean` is that
        this method averages the data along the specified dimension(s) and retains the
        averaged coordinate. This method also implicitly averages all coordinates that
        depend on the averaged dimension(s) instead of dropping them.

        Parameters
        ----------
        dim
            The dimension(s) along which to average the data.

        Returns
        -------
        DataArray
            The data averaged along the specified dimension(s).
        """
        if isinstance(dim, str):
            dim = (dim,)

        qsel_kwargs: dict[Hashable, float] = {}

        for d in dim:
            qsel_kwargs[d] = 0.0
            qsel_kwargs[f"{d}_width"] = np.inf

        return self.__call__(qsel_kwargs)

    def around(
        self, radius: float | dict[Hashable, float], *, average: bool = True, **sel_kw
    ) -> xr.DataArray:
        """
        Average data within a specified radius of a specified point.

        For instance, consider an ARPES map with dimensions ``'kx'``, ``'ky'``, and
        ``'eV'``. Providing ``'kx'`` and ``'ky'`` points will average the data within a
        cylindrical region centered at that point. The radius of the cylinder is
        specified by ``radius``.

        If different radii are given for ``kx`` and ``ky``, the region will be elliptic.

        Parameters
        ----------
        radius
            The radius of the region. If a single number, the same radius is used for
            all dimensions. If a dictionary, keys must be valid dimension names and the
            values are the radii for the corresponding dimensions.
        average
            If `True`, return the mean value of the data within the region. If `False`,
            return the masked data.
        **sel_kw
            The center of the spherical region. Must be a mapping of valid dimension
            names to coordinate values.

        Returns
        -------
        DataArray
            The mean value of the data within the region.

        Note
        ----
        The region is defined by a spherical mask, which is generated with
        :func:`erlab.analysis.mask.spherical_mask`. Depending on the radius and
        dimensions provided, the mask will be hyperellipsoid in the dimensions specified
        in ``sel_kw``.

        See Also
        --------
        :func:`erlab.analysis.mask.spherical_mask`

        """
        masked = self._obj.where(
            erlab.analysis.mask.spherical_mask(self._obj, radius, **sel_kw),
            drop=average,
        )
        if average:
            return masked.qsel.average(sel_kw.keys())
        return masked


@xr.register_dataarray_accessor("qinfo")
class InfoDataArrayAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.qinfo` accessor for displaying information about the data."""

    def get_value(self, attr_or_coord_name: str) -> typing.Any:
        """Get the value of the specified attribute or coordinate.

        If the attribute or coordinate is not found, `None` is returned.

        Parameters
        ----------
        attr_or_coord_name
            The name of the attribute or coordinate.

        """
        if attr_or_coord_name in self._obj.attrs:
            return self._obj.attrs[attr_or_coord_name]
        if attr_or_coord_name in self._obj.coords:
            return self._obj.coords[attr_or_coord_name]
        return None

    @functools.cached_property
    def _summary_table(self) -> list[tuple[str, str, str]]:
        if "data_loader_name" in self._obj.attrs:
            loader = erlab.io.loaders[self._obj.attrs["data_loader_name"]]
        else:
            raise ValueError("Data loader information not found in data attributes")

        out: list[tuple[str, str, str]] = []

        for key, true_key in loader.summary_attrs.items():
            val = loader.get_formatted_attr_or_coord(self._obj, true_key)
            if callable(true_key):
                true_key = ""
            out.append((key, loader.value_to_string(val), true_key))

        return out

    def _repr_html_(self) -> str:
        return erlab.utils.formatting.format_html_table(
            [("Name", "Value", "Key"), *self._summary_table],
            header_cols=1,
            header_rows=1,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"{key}: {val}" if not true_key else f"{key} ({true_key}): {val}"
                for key, val, true_key in self._summary_table
            ]
        )
