"""Defines accessors for curve fitting."""

from __future__ import annotations

__all__ = ["ParallelFitDataArrayAccessor"]

import typing
import warnings
from collections.abc import Hashable, Mapping

import xarray as xr
import xarray_lmfit

from erlab.accessors.utils import ERLabDataArrayAccessor, ERLabDatasetAccessor

if typing.TYPE_CHECKING:
    # Avoid importing until runtime for initial import performance
    import joblib
    import lmfit
    import tqdm.auto as tqdm
else:
    import lazy_loader as _lazy

    from erlab.utils.misc import LazyImport

    lmfit = _lazy.load("lmfit")
    joblib = _lazy.load("joblib")
    tqdm = LazyImport("tqdm.auto")


@xr.register_dataset_accessor("modelfit")
class ModelFitDatasetAccessor(ERLabDatasetAccessor):
    """`xarray.Dataset.modelfit` accessor for fitting lmfit models."""

    def __call__(self, *args, **kwargs) -> xr.Dataset:  # pragma: no cover
        """Alias for :meth:`xarray.Dataset.xlm.modelfit`.

        .. deprecated:: 3.8.0

            Use :meth:`xarray.Dataset.xlm.modelfit` instead.

        """
        warnings.warn(
            "`Dataset.modelfit` is deprecated, use "
            "`Dataset.xlm.modelfit` after importing `xarray_lmfit`",
            FutureWarning,
            stacklevel=1,
        )
        return self._obj.xlm.modelfit(*args, **kwargs)


@xr.register_dataarray_accessor("modelfit")
class ModelFitDataArrayAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.modelfit` accessor for fitting lmfit models."""

    def __call__(self, *args, **kwargs) -> xr.Dataset:  # pragma: no cover
        """Alias for :meth:`xarray.DataArray.xlm.modelfit`.

        .. deprecated:: 3.8.0

            Use :meth:`xarray.DataArray.xlm.modelfit` instead.

        """
        warnings.warn(
            "`DataArray.modelfit` is deprecated, use "
            "`DataArray.xlm.modelfit` after importing `xarray_lmfit`",
            FutureWarning,
            stacklevel=1,
        )
        return self._obj.xlm.modelfit(*args, **kwargs)


@xr.register_dataarray_accessor("parallel_fit")
class ParallelFitDataArrayAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.parallel_fit` accessor for fitting lmfit models in parallel."""

    _VAR_KEYS: tuple[str, ...] = (
        "modelfit_results",
        "modelfit_coefficients",
        "modelfit_stderr",
        "modelfit_covariance",
        "modelfit_stats",
        "modelfit_data",
        "modelfit_best_fit",
    )

    def __call__(self, dim: str, model: lmfit.Model, **kwargs) -> xr.Dataset:
        """
        Fit the specified model to the data along the given dimension.

        Parameters
        ----------
        dim : str
            The name of the dimension along which to fit the model. Note that this is
            the dimension along which the model will be parallelized over, not the
            independent dimension(s) of the model.
        model : lmfit.Model
            The model to fit.
        **kwargs : dict
            Additional keyword arguments to be passed to
            :meth:`xarray.Dataset.xlm.modelfit`.

        Returns
        -------
        curvefit_results : xarray.Dataset
            The dataset containing the results of the fit. See
            :meth:`xarray.DataArray.xlm.modelfit` for details.

        """
        if self._obj.chunks is not None:
            raise ValueError(
                "The input DataArray is chunked. Parallel fitting will not offer any "
                "performance benefits. Use `DataArray.xlm.modelfit` instead"
            )

        ds = self._obj.to_dataset(dim, promote_attrs=True)

        kwargs.setdefault("parallel", True)
        kwargs.setdefault("progress", True)

        if isinstance(kwargs.get("params"), Mapping):
            kwargs["params"] = xarray_lmfit.modelfit._parse_params(
                kwargs["params"], dask=False
            )

        if isinstance(kwargs.get("params"), xr.DataArray):
            kwargs["params"] = kwargs["params"].to_dataset(dim, promote_attrs=True)

        fitres = ds.xlm.modelfit(set(self._obj.dims) - {dim}, model, **kwargs)

        drop_keys = []
        concat_vars: dict[Hashable, list[xr.DataArray]] = {}
        for k in ds.data_vars:
            for var in self._VAR_KEYS:
                key = f"{k}_{var}"
                if key in fitres:
                    if var not in concat_vars:
                        concat_vars[var] = []
                    concat_vars[var].append(fitres[key])
                    drop_keys.append(key)

        return (
            fitres.drop_vars(drop_keys)
            .assign(
                {
                    k: xr.concat(
                        v, dim, coords="minimal", compat="override", join="override"
                    )
                    for k, v in concat_vars.items()
                }
            )
            .assign_coords({dim: self._obj[dim]})
        )
