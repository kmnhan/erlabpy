__all__ = ["ModelFitDataArrayAccessor", "ModelFitDatasetAccessor"]

import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Literal

import lmfit
import numpy as np
import xarray as xr
from xarray.core.types import Dims

from erlab.accessors.utils import _THIS_ARRAY, ERLabAccessor


@xr.register_dataset_accessor("modelfit")
class ModelFitDatasetAccessor(ERLabAccessor):
    """`xarray.Dataset.modelfit` accessor for fitting lmfit models."""

    def __call__(
        self,
        coords: str | xr.DataArray | Iterable[str | xr.DataArray],
        model: lmfit.Model,
        reduce_dims: Dims = None,
        skipna: bool = True,
        params: str
        | lmfit.Parameters
        | Mapping[str, float | dict[str, Any]]
        | xr.DataArray
        | None = None,
        guess: bool = False,
        errors: Literal["raise", "ignore"] = "raise",
        parallel: bool | None = None,
        parallel_kw: dict[str, Any] | None = None,
        progress: bool = False,
        output_result: bool = True,
        **kwargs,
    ) -> xr.Dataset:
        """
        Curve fitting optimization for arbitrary functions.

        Wraps :func:`lmfit.Model.fit` with `apply_ufunc`.

        Parameters
        ----------
        coords : hashable, xarray.DataArray, or sequence of hashable or xarray.DataArray
            Independent coordinate(s) over which to perform the curve fitting. Must
            share at least one dimension with the calling object. When fitting
            multi-dimensional functions, supply `coords` as a sequence in the same order
            as arguments in `func`. To fit along existing dimensions of the calling
            object, `coords` can also be specified as a str or sequence of strs.
        model : lmfit.Model
            A model object to fit to the data. The model must be an instance of
            `lmfit.Model`.
        reduce_dims : str, Iterable of Hashable or None, optional
            Additional dimension(s) over which to aggregate while fitting. For example,
            calling `ds.curvefit(coords='time', reduce_dims=['lat', 'lon'], ...)` will
            aggregate all lat and lon points and fit the specified function along the
            time dimension.
        skipna : bool, default: True
            Whether to skip missing values when fitting. Default is True.
        params : str, lmfit.Parameters, dict-like, or xarray.DataArray, optional
            Optional input parameters to the fit. If a string, it should be a JSON
            string representation of the parameters, generated by
            `lmfit.Parameters.dumps`. If a `lmfit.Parameters` object, it will be used as
            is. If a dict-like object, it will be converted to a `lmfit.Parameters`
            object. If the values are DataArrays, they will be appropriately broadcast
            to the coordinates of the array. If none or only some parameters are passed,
            the rest will be assigned initial values and bounds with
            :meth:`lmfit.Model.make_params`, or guessed with :meth:`lmfit.Model.guess`
            if `guess` is `True`.
        guess : bool, default: `False`
            Whether to guess the initial parameters with :meth:`lmfit.Model.guess`. For
            composite models, the parameters will be guessed for each component.
        errors : {"raise", "ignore"}, default: "raise"
            If `'raise'`, any errors from the :func:`lmfit.Model.fit` optimization will
            raise an exception. If `'ignore'`, the return values for the coordinates
            where the fitting failed will be NaN.
        **kwargs : optional
            Additional keyword arguments to passed to :func:`lmfit.Model.fit`.

        Returns
        -------
        curvefit_results : xarray.Dataset
            A single dataset which contains:

            [var]_modelfit_coefficients
                The coefficients of the best fit.
            [var]_modelfit_stderr
                The standard errors of the coefficients.
            [var]_modelfit_stats
                Statistics from the fit. See :func:`lmfit.minimize`.

        See Also
        --------
        xarray.Dataset.curve_fit xarray.Dataset.polyfit lmfit.model.Model.fit
        scipy.optimize.curve_fit

        """
        # Implementation analogous to xarray.Dataset.curve_fit

        if params is None:
            params = lmfit.create_params()

        if kwargs is None:
            kwargs = {}

        # Input to apply_ufunc cannot be a Mapping, so convert parameters to str
        if isinstance(params, lmfit.Parameters):
            params: str = params.dumps()
        elif isinstance(params, Mapping):
            # Given as a mapping from str to float or dict
            params: str = lmfit.create_params(**params).dumps()

        reduce_dims_: list[Hashable]
        if not reduce_dims:
            reduce_dims_ = []
        elif isinstance(reduce_dims, str) or not isinstance(reduce_dims, Iterable):
            reduce_dims_ = [reduce_dims]
        else:
            reduce_dims_ = list(reduce_dims)

        if (
            isinstance(coords, str)
            or isinstance(coords, xr.DataArray)
            or not isinstance(coords, Iterable)
        ):
            coords = [coords]
        coords_: Sequence[xr.DataArray] = [
            self._obj[coord] if isinstance(coord, str) else coord for coord in coords
        ]

        # Determine whether any coords are dims on self._obj
        for coord in coords_:
            reduce_dims_ += [c for c in self._obj.dims if coord.equals(self._obj[c])]
        reduce_dims_ = list(set(reduce_dims_))
        preserved_dims = list(set(self._obj.dims) - set(reduce_dims_))
        if not reduce_dims_:
            raise ValueError(
                "No arguments to `coords` were identified as a dimension on the "
                "calling object, and no dims were supplied to `reduce_dims`. This "
                "would result in fitting on scalar data."
            )

        # Check that initial guess and bounds only contain coords in preserved_dims
        if isinstance(params, xr.DataArray):
            unexpected = set(params.dims) - set(preserved_dims)
            if unexpected:
                raise ValueError(
                    f"Initial guess has unexpected dimensions {tuple(unexpected)}. It "
                    "should only have dimensions that are in data dimensions "
                    f"{preserved_dims}."
                )

        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        # Broadcast all coords with each other
        coords_ = xr.broadcast(*coords_)
        coords_ = [
            coord.broadcast_like(self._obj, exclude=preserved_dims) for coord in coords_
        ]
        n_coords = len(coords_)

        # Get the parameter names
        param_names: list[str] = model.param_names
        n_params = len(param_names)

        # Define the statistics to extract from the fit result
        stat_names = [
            "nfev",
            "nvarys",
            "ndata",
            "nfree",
            "chisqr",
            "redchi",
            "aic",
            "bic",
        ]
        n_stats = len(stat_names)

        def _wrapper(Y, *args, **kwargs):
            # Wrap Model.fit with raveled coordinates and pointwise NaN handling
            # *args contains:
            #   - the coordinates
            #   - parameters object
            coords__ = args[:n_coords]
            init_params_ = args[n_coords]

            if guess:
                initial_params = lmfit.create_params()
            else:
                initial_params = model.make_params()
            if isinstance(init_params_, str):
                initial_params.update(lmfit.Parameters().loads(init_params_))
            elif isinstance(init_params_, lmfit.Parameters):
                initial_params.update(init_params_)
            elif isinstance(init_params_, Mapping):
                for p, v in init_params_.items():
                    if isinstance(v, Mapping):
                        initial_params[p].set(**v)
                    else:
                        initial_params[p].set(value=v)

            x = np.vstack([c.ravel() for c in coords__])
            y = Y.ravel()
            if skipna:
                mask = np.all([np.any(~np.isnan(x), axis=0), ~np.isnan(y)], axis=0)
                x = x[:, mask]
                y = y[mask]
                if not len(y):
                    popt = np.full([n_params], np.nan)
                    perr = np.full([n_params, n_params], np.nan)
                    stats = np.full([n_stats], np.nan)
                    return popt, perr, stats
            x = np.squeeze(x)

            if n_coords == 1:
                indep_var_kwargs = {model.independent_vars[0]: x}
            else:
                indep_var_kwargs = dict(zip(model.independent_vars[:n_coords], x))

            if guess:
                if isinstance(model, lmfit.model.CompositeModel):
                    guessed_params = model.make_params()
                    for comp in model.components:
                        try:
                            guessed_params.update(comp.guess(y, **indep_var_kwargs))
                        except NotImplementedError:
                            pass
                    # Given parameters must override guessed parameters
                    initial_params = guessed_params.update(initial_params)

                else:
                    try:
                        initial_params = model.guess(y, **indep_var_kwargs).update(
                            initial_params
                        )
                    except NotImplementedError:
                        warnings.warn(
                            f"`guess` is not implemented for {model}, "
                            "using supplied initial parameters",
                            stacklevel=1,
                        )
                        initial_params = model.make_params().update(initial_params)
            try:
                fitresult: lmfit.model.ModelResult = model.fit(
                    y, **indep_var_kwargs, params=initial_params, **kwargs
                )
            except RuntimeError:
                if errors == "raise":
                    raise
                popt = np.full([n_params], np.nan)
                perr = np.full([n_params, n_params], np.nan)
                stats = np.full([n_stats], np.nan)
            else:
                if fitresult.success:
                    popt, perr = [], []
                    for name in param_names:
                        p = fitresult.params[name]
                        popt.append(p.value if p.value is not None else np.nan)
                        perr.append(p.stderr if p.stderr is not None else np.nan)
                    popt, perr = np.array(popt), np.array(perr)

                    stats = [getattr(fitresult, s) for s in stat_names]
                    stats = np.array([s if s is not None else np.nan for s in stats])
                else:
                    popt = np.full([n_params], np.nan)
                    perr = np.full([n_params, n_params], np.nan)
                    stats = np.full([n_stats], np.nan)

            return popt, perr, stats

        result = type(self._obj)()
        for name, da in self._obj.data_vars.items():
            if name is _THIS_ARRAY:
                name = ""
            else:
                name = f"{name!s}_"

            input_core_dims = [reduce_dims_ for _ in range(n_coords + 1)]
            input_core_dims.extend([[] for _ in range(1)])  # core_dims for parameters

            popt, perr, stats = xr.apply_ufunc(
                _wrapper,
                da,
                *coords_,
                params,
                vectorize=True,
                dask="parallelized",
                input_core_dims=input_core_dims,
                output_core_dims=[["param"], ["param"], ["fit_stat"]],
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "param": n_params,
                        "stat": n_stats,
                    },
                },
                output_dtypes=(np.float64, np.float64, np.float64),
                exclude_dims=set(reduce_dims_),
                kwargs=kwargs,
            )
            result[name + "modelfit_coefficients"] = popt
            result[name + "modelfit_stderr"] = perr
            result[name + "modelfit_stats"] = stats

        result = result.assign_coords({"param": param_names, "fit_stat": stat_names})
        result.attrs = self._obj.attrs.copy()

        return result


@xr.register_dataarray_accessor("modelfit")
class ModelFitDataArrayAccessor(ERLabAccessor):
    """`xarray.DataArray.modelfit` accessor for fitting lmfit models."""

    def __call__(self, *args, **kwargs):
        return self._obj.to_dataset(name=_THIS_ARRAY).modelfit(*args, **kwargs)

    __call__.__doc__ = (
        ModelFitDatasetAccessor.__call__.__doc__.replace(
            "Dataset.curve_fit", "DataArray.curve_fit"
        )
        .replace("Dataset.polyfit", "DataArray.polyfit")
        .replace("[var]_", "")
    )
