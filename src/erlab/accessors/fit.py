"""Defines accessors for curve fitting.

All accessors in this module have been deprecated in favor of using `xarray-lmfit
<https://xarray-lmfit.readthedocs.io/stable/>`_ directly.

"""

from __future__ import annotations

__all__ = []

import typing
import warnings

import xarray as xr

from erlab.accessors.utils import ERLabDataArrayAccessor, ERLabDatasetAccessor

if typing.TYPE_CHECKING:
    # Avoid importing until runtime for initial import performance
    import lmfit
else:
    import lazy_loader as _lazy

    lmfit = _lazy.load("lmfit")


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

        .. deprecated:: 3.14.1

            Use :meth:`xarray.DataArray.xlm.modelfit` with ``dask`` instead.


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
        warnings.warn(
            "`DataArray.parallel_fit` is deprecated, use "
            "`DataArray.xlm.modelfit` after importing `xarray_lmfit` with dask",
            FutureWarning,
            stacklevel=1,
        )
        chunked = self._obj.chunk({dim: 1})
        fitres = chunked.xlm.modelfit(set(self._obj.dims) - {dim}, model, **kwargs)
        return fitres.compute()
