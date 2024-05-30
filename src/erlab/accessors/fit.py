from __future__ import annotations

__all__ = [
    "ModelFitDataArrayAccessor",
    "ModelFitDatasetAccessor",
    "ParallelFitDataArrayAccessor",
]

import copy
import itertools
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import joblib
import lmfit
import numpy as np
import tqdm.auto
import xarray as xr

from erlab.accessors.utils import (
    _THIS_ARRAY,
    ERLabDataArrayAccessor,
    ERLabDatasetAccessor,
)
from erlab.utils.parallel import joblib_progress

if TYPE_CHECKING:
    from xarray.core.types import Dims


def _nested_dict_vals(d):
    for v in d.values():
        if isinstance(v, Mapping):
            yield from _nested_dict_vals(v)
        else:
            yield v


def _broadcast_dict_values(d: dict[str, Any]) -> dict[str, xr.DataArray]:
    to_broadcast = {}
    for k, v in d.items():
        if isinstance(v, xr.DataArray | xr.Dataset):
            to_broadcast[k] = v
        else:
            to_broadcast[k] = xr.DataArray(v)

    d = dict(
        zip(to_broadcast.keys(), xr.broadcast(*to_broadcast.values()), strict=True)
    )
    return cast(dict[str, xr.DataArray], d)


def _concat_along_keys(d: dict[str, xr.DataArray], dim_name: str) -> xr.DataArray:
    return xr.concat(d.values(), d.keys(), coords="minimal").rename(concat_dim=dim_name)


def _parse_params(
    d: dict[str, Any] | lmfit.Parameters, dask: bool
) -> xr.DataArray | _ParametersWraper:
    if isinstance(d, lmfit.Parameters):
        # Input to apply_ufunc cannot be a Mapping, so wrap in a class
        return _ParametersWraper(d)

    # Iterate over all values
    for v in _nested_dict_vals(d):
        if isinstance(v, xr.DataArray):
            # For dask arrays, auto rechunking with object dtype is unsupported, so must
            # convert to str
            return _parse_multiple_params(copy.deepcopy(d), dask)

    return _ParametersWraper(lmfit.create_params(**d))


def _parse_multiple_params(d: dict[str, Any], as_str: bool) -> xr.DataArray:
    for k in d.keys():
        if isinstance(d[k], int | float | complex | xr.DataArray):
            d[k] = {"value": d[k]}

        d[k] = _concat_along_keys(_broadcast_dict_values(d[k]), "__dict_keys")

    da = _concat_along_keys(_broadcast_dict_values(d), "__param_names")

    pnames = tuple(da["__param_names"].values)
    argnames = tuple(da["__dict_keys"].values)

    def _reduce_to_param(arr, axis=0):
        out_arr = np.empty_like(arr.mean(axis=axis), dtype=object)
        for i in range(out_arr.size):
            out_arr.flat[i] = {}

        for i, par in enumerate(pnames):
            for j, name in enumerate(argnames):
                for k, val in enumerate(arr[i, j].flat):
                    if par not in out_arr.flat[k]:
                        out_arr.flat[k][par] = {}

                    if np.isfinite(val):
                        out_arr.flat[k][par][name] = val

        for i in range(out_arr.size):
            out_arr.flat[i] = lmfit.create_params(**out_arr.flat[i])
            if as_str:
                out_arr.flat[i] = out_arr.flat[i].dumps()

        if as_str:
            return out_arr.astype(str)
        else:
            return out_arr

    da = da.reduce(_reduce_to_param, ("__dict_keys", "__param_names"))
    return da


class _ParametersWraper:
    def __init__(self, params: lmfit.Parameters):
        self.params = params


@xr.register_dataset_accessor("modelfit")
class ModelFitDatasetAccessor(ERLabDatasetAccessor):
    """`xarray.Dataset.modelfit` accessor for fitting lmfit models."""

    def __call__(
        self,
        coords: str | xr.DataArray | Iterable[str | xr.DataArray],
        model: lmfit.Model,
        reduce_dims: Dims = None,
        skipna: bool = True,
        params: lmfit.Parameters
        | dict[str, float | dict[str, Any]]
        | xr.DataArray
        | xr.Dataset
        | _ParametersWraper
        | None = None,
        guess: bool = False,
        errors: Literal["raise", "ignore"] = "raise",
        parallel: bool | None = None,
        parallel_kw: dict[str, Any] | None = None,
        progress: bool = False,
        output_result: bool = True,
        **kwargs,
    ) -> xr.Dataset:
        """Curve fitting optimization for arbitrary models.

        Wraps :meth:`lmfit.Model.fit <lmfit.model.Model.fit>` with
        :func:`xarray.apply_ufunc`.

        Parameters
        ----------
        coords : Hashable, xarray.DataArray, or Sequence of Hashable or xarray.DataArray
            Independent coordinate(s) over which to perform the curve fitting. Must
            share at least one dimension with the calling object. When fitting
            multi-dimensional functions, supply `coords` as a sequence in the same order
            as arguments in `func`. To fit along existing dimensions of the calling
            object, `coords` can also be specified as a str or sequence of strs.
        model : `lmfit.Model <lmfit.model.Model>`
            A model object to fit to the data. The model must be an *instance* of
            :class:`lmfit.Model <lmfit.model.Model>`.
        reduce_dims : str, Iterable of Hashable or None, optional
            Additional dimension(s) over which to aggregate while fitting. For example,
            calling `ds.modelfit(coords='time', reduce_dims=['lat', 'lon'], ...)` will
            aggregate all lat and lon points and fit the specified function along the
            time dimension.
        skipna : bool, default: True
            Whether to skip missing values when fitting. Default is True.
        params : lmfit.Parameters, dict-like, or xarray.DataArray, optional
            Optional input parameters to the fit. If a `lmfit.Parameters
            <lmfit.parameter.Parameters>` object, it will be used for all fits. If a
            dict-like object, it must look like the keyword arguments to
            :func:`lmfit.create_params <lmfit.parameter.create_params>`. Additionally,
            each value of the dictionary may also be a DataArray, which will be
            broadcasted appropriately. If a DataArray, each entry must be a
            dictionary-like object, a `lmfit.Parameters <lmfit.parameter.Parameters>`
            object, or a JSON string created with :meth:`lmfit.Parameters.dumps
            <lmfit.parameter.Parameters.dumps>`. If given a Dataset, the name of the
            data variables in the Dataset must match the name of the data variables in
            the calling object, and each data variable will be used as the parameters
            for the corresponding data variable. If none or only some parameters are
            passed, the rest will be assigned initial values and bounds with
            :meth:`lmfit.Model.make_params <lmfit.model.Model.make_params>`, or guessed
            with :meth:`lmfit.Model.guess <lmfit.model.Model.guess>` if `guess` is
            `True`.
        guess : bool, default: `False`
            Whether to guess the initial parameters with :meth:`lmfit.Model.guess
            <lmfit.model.Model.guess>`. For composite models, the parameters will be
            guessed for each component.
        errors : {"raise", "ignore"}, default: "raise"
            If `'raise'`, any errors from the :meth:`lmfit.Model.fit
            <lmfit.model.Model.fit>` optimization will raise an exception. If
            `'ignore'`, the return values for the coordinates where the fitting failed
            will be NaN.
        parallel : bool, optional
            Whether to parallelize the fits over the data variables. If not provided,
            parallelization is only applied for non-dask Datasets with more than 200
            data variables.
        parallel_kw : dict, optional
            Additional keyword arguments to pass to the parallelization backend
            :class:`joblib.Parallel` if `parallel` is `True`.
        progress : bool, default: `False`
            Whether to show a progress bar for fitting over data variables. Only useful
            if there are multiple data variables to fit.
        output_result : bool, default: `True`
            Whether to include the full :class:`lmfit.model.ModelResult` object in the
            output dataset. If `True`, the result will be stored in a variable named
            `[var]_modelfit_results`.
        **kwargs : optional
            Additional keyword arguments to passed to :meth:`lmfit.Model.fit
            <lmfit.model.Model.fit>`.

        Returns
        -------
        curvefit_results : xarray.Dataset
            A single dataset which contains:

            [var]_modelfit_results
                The full :class:`lmfit.model.ModelResult` object from the fit. Only
                included if `output_result` is `True`.
            [var]_modelfit_coefficients
                The coefficients of the best fit.
            [var]_modelfit_stderr
                The standard errors of the coefficients.
            [var]_modelfit_covariance
                The covariance matrix of the coefficients. Note that elements
                corresponding to non varying parameters are set to NaN, and the actual
                size of the covariance matrix may be smaller than the array.
            [var]_modelfit_stats
                Statistics from the fit. See :func:`lmfit.minimize
                <lmfit.minimizer.minimize>`.
            [var]_modelfit_data
                Data used for the fit.
            [var]_modelfit_best_fit
                The best fit data of the fit.

        See Also
        --------
        xarray.Dataset.curvefit

        xarray.Dataset.polyfit

        lmfit.model.Model.fit

        scipy.optimize.curve_fit

        """
        # Implementation analogous to xarray.Dataset.curve_fit

        if params is None:
            params = lmfit.create_params()

        if parallel_kw is None:
            parallel_kw = {}

        if kwargs is None:
            kwargs = {}

        is_dask: bool = not (
            self._obj.chunksizes is None or len(self._obj.chunksizes) == 0
        )

        if not isinstance(params, xr.Dataset) and isinstance(params, Mapping):
            # Given as a mapping from str to ...
            # float or DataArray or dict of str to Any (including DataArray of Any)
            params = _parse_params(params, is_dask)

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
        if isinstance(params, xr.DataArray | xr.Dataset):
            unexpected = set(params.dims) - set(preserved_dims)
            if unexpected:
                raise ValueError(
                    f"Parameters object has unexpected dimensions {tuple(unexpected)}. It "
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

        # Call make_params before getting parameter names as it may add param hints
        model.make_params()

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

            if isinstance(init_params_, _ParametersWraper):
                initial_params.update(init_params_.params)

            elif isinstance(init_params_, str):
                initial_params.update(lmfit.Parameters().loads(init_params_))

            elif isinstance(init_params_, lmfit.Parameters):
                initial_params.update(init_params_)

            elif isinstance(init_params_, Mapping):
                for p, v in init_params_.items():
                    if isinstance(v, Mapping):
                        initial_params[p].set(**v)
                    else:
                        initial_params[p].set(value=v)

            popt = np.full([n_params], np.nan)
            perr = np.full([n_params], np.nan)
            pcov = np.full([n_params, n_params], np.nan)
            stats = np.full([n_stats], np.nan)
            data = Y.copy()
            best = np.full_like(data, np.nan)

            x = np.vstack([c.ravel() for c in coords__])
            y = Y.ravel()

            if skipna:
                mask = np.all([np.any(~np.isnan(x), axis=0), ~np.isnan(y)], axis=0)
                x = x[:, mask]
                y = y[mask]
                if not len(y):
                    modres: lmfit.model.ModelResult = lmfit.model.ModelResult(
                        model, model.make_params(), data=y
                    )
                    modres.success = False
                    return popt, perr, pcov, stats, data, best, modres
            else:
                mask = np.full_like(y, True)

            x = np.squeeze(x)

            if model.independent_vars is not None:
                if n_coords == 1:
                    indep_var_kwargs = {model.independent_vars[0]: x}
                    if len(model.independent_vars) == 2:
                        # Y-dependent data, like background models
                        indep_var_kwargs[model.independent_vars[1]] = y
                else:
                    indep_var_kwargs = dict(
                        zip(model.independent_vars[:n_coords], x, strict=True)
                    )
            else:
                raise ValueError("Independent variables not defined in model")

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
                modres = model.fit(
                    y, **indep_var_kwargs, params=initial_params, **kwargs
                )
            except ValueError:
                if errors == "raise":
                    raise
                modres = lmfit.model.ModelResult(model, initial_params, data=y)
                modres.success = False
                return popt, perr, pcov, stats, data, best, modres
            else:
                if modres.success:
                    popt_list, perr_list = [], []
                    for name in param_names:
                        p = modres.params[name]
                        popt_list.append(p.value if p.value is not None else np.nan)
                        perr_list.append(p.stderr if p.stderr is not None else np.nan)

                    popt, perr = np.array(popt_list), np.array(perr_list)

                    stats = np.array(
                        [
                            s if s is not None else np.nan
                            for s in [getattr(modres, s) for s in stat_names]
                        ]
                    )

                    # Fill in covariance matrix entries, entries for non-varying
                    # parameters are left as NaN
                    if modres.covar is not None:
                        var_names = modres.var_names
                        for vi in range(modres.nvarys):
                            i = param_names.index(var_names[vi])
                            for vj in range(modres.nvarys):
                                j = param_names.index(var_names[vj])
                                pcov[i, j] = modres.covar[vi, vj]

                    best.flat[mask] = modres.best_fit

            return popt, perr, pcov, stats, data, best, modres

        def _output_wrapper(name, da, out=None) -> dict:
            if name is _THIS_ARRAY:
                name = ""
            else:
                name = f"{name!s}_"

            if out is None:
                out = {}

            input_core_dims = [reduce_dims_ for _ in range(n_coords + 1)]
            input_core_dims.extend([[] for _ in range(1)])  # core_dims for parameters

            if isinstance(params, xr.Dataset):
                try:
                    params_to_apply = params[name.rstrip("_")]
                except KeyError:
                    params_to_apply = params[float(name.rstrip("_"))]
            else:
                params_to_apply = params

            popt, perr, pcov, stats, data, best, modres = xr.apply_ufunc(
                _wrapper,
                da,
                *coords_,
                params_to_apply,
                vectorize=True,
                dask="parallelized",
                input_core_dims=input_core_dims,
                output_core_dims=[
                    ["param"],
                    ["param"],
                    ["cov_i", "cov_j"],
                    ["fit_stat"],
                    reduce_dims_,
                    reduce_dims_,
                    [],
                ],
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "param": n_params,
                        "fit_stat": n_stats,
                        "cov_i": n_params,
                        "cov_j": n_params,
                    }
                    | {dim: self._obj.coords[dim].size for dim in reduce_dims_}
                },
                output_dtypes=(
                    np.float64,
                    np.float64,
                    np.float64,
                    np.float64,
                    np.float64,
                    np.float64,
                    lmfit.model.ModelResult,
                ),
                exclude_dims=set(reduce_dims_),
                kwargs=kwargs,
            )

            if output_result:
                out[name + "modelfit_results"] = modres

            out[name + "modelfit_coefficients"] = popt
            out[name + "modelfit_stderr"] = perr
            out[name + "modelfit_covariance"] = pcov
            out[name + "modelfit_stats"] = stats
            out[name + "modelfit_data"] = data
            out[name + "modelfit_best_fit"] = best

            return out

        if parallel is None:
            parallel = (not is_dask) and (len(self._obj.data_vars) > 200)

        tqdm_kw = {
            "desc": "Fitting",
            "total": len(self._obj.data_vars),
            "disable": not progress,
        }

        if parallel:
            if is_dask:
                warnings.warn(
                    "The input Dataset is chunked. Parallel fitting will not offer any "
                    "performance benefits.",
                    stacklevel=1,
                )

            parallel_kw.setdefault("n_jobs", -1)
            parallel_kw.setdefault("max_nbytes", None)
            parallel_kw.setdefault("return_as", "generator_unordered")
            parallel_kw.setdefault("pre_dispatch", "n_jobs")
            parallel_kw.setdefault("prefer", "processes")

            parallel_obj = joblib.Parallel(**parallel_kw)

            if parallel_obj.return_generator:
                out_dicts = tqdm.auto.tqdm(  # type: ignore[call-overload]
                    parallel_obj(
                        itertools.starmap(
                            joblib.delayed(_output_wrapper), self._obj.data_vars.items()
                        )
                    ),
                    **tqdm_kw,
                )
            else:
                with joblib_progress(**tqdm_kw) as _:
                    out_dicts = parallel_obj(
                        itertools.starmap(
                            joblib.delayed(_output_wrapper), self._obj.data_vars.items()
                        )
                    )
            result = type(self._obj)(
                dict(itertools.chain.from_iterable(d.items() for d in out_dicts))
            )
            del out_dicts

        else:
            result = type(self._obj)()
            for name, da in tqdm.auto.tqdm(self._obj.data_vars.items(), **tqdm_kw):  # type: ignore[call-overload]
                _output_wrapper(name, da, result)

        result = result.assign_coords(
            {
                "param": param_names,
                "fit_stat": stat_names,
                "cov_i": param_names,
                "cov_j": param_names,
            }
            | {dim: self._obj.coords[dim] for dim in reduce_dims_}
        )
        result.attrs = self._obj.attrs.copy()

        return result


@xr.register_dataarray_accessor("modelfit")
class ModelFitDataArrayAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.modelfit` accessor for fitting lmfit models."""

    def __call__(self, *args, **kwargs) -> xr.Dataset:
        return self._obj.to_dataset(name=_THIS_ARRAY).modelfit(*args, **kwargs)

    __call__.__doc__ = (
        str(ModelFitDatasetAccessor.__call__.__doc__)
        .replace("Dataset.curvefit", "DataArray.curvefit")
        .replace("Dataset.polyfit", "DataArray.polyfit")
        .replace("[var]_", "")
    )


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
            The name of the dimension along which to fit the model.
        model : lmfit.Model
            The model to fit.
        **kwargs : dict
            Additional keyword arguments to be passed to :func:`xarray.Dataset.modelfit
            <erlab.accessors.fit.ModelFitDatasetAccessor.__call__>`.

        Returns
        -------
        curvefit_results : xarray.Dataset
            The dataset containing the results of the fit. See
            :func:`xarray.DataArray.modelfit
            <erlab.accessors.fit.ModelFitDataArrayAccessor.__call__>` for details.

        """
        if self._obj.chunks is None is not None:
            raise ValueError(
                "The input DataArray is chunked. Parallel fitting will not offer any "
                "performance benefits. Use `modelfit` instead"
            )

        ds = self._obj.to_dataset(dim, promote_attrs=True)

        kwargs.setdefault("parallel", True)
        kwargs.setdefault("progress", True)

        if isinstance(kwargs.get("params", None), Mapping):
            kwargs["params"] = _parse_params(kwargs["params"], dask=False)

        if isinstance(kwargs.get("params", None), xr.DataArray):
            kwargs["params"] = kwargs["params"].to_dataset(dim, promote_attrs=True)

        fitres = ds.modelfit(set(self._obj.dims) - {dim}, model, **kwargs)

        drop_keys = []
        concat_vars: dict[Hashable, list[xr.DataArray]] = {}
        for k in ds.data_vars.keys():
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
