import erlab.accessors  # noqa: F401
import lmfit
import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize("use_dask", [True, False])
def test_modelfit(use_dask: bool):
    # Tests are adapted from xarray's curvefit tests

    def exp_decay(t, n0, tau=1):
        return n0 * np.exp(-t / tau)

    def power(t, a):
        return np.power(t, a)

    t = np.arange(0, 5, 0.5)
    da = xr.DataArray(
        np.stack([exp_decay(t, 3, 3), exp_decay(t, 5, 4), np.nan * t], axis=-1),
        dims=("t", "x"),
        coords={"t": t, "x": [0, 1, 2]},
    )
    da[0, 0] = np.nan

    expected = xr.DataArray(
        [[3, 3], [5, 4], [np.nan, np.nan]],
        dims=("x", "param"),
        coords={"x": [0, 1, 2], "param": ["n0", "tau"]},
    )

    if use_dask:
        da = da.chunk({"x": 1})

    # Create model
    model = lmfit.Model(exp_decay)

    # Params as dictionary
    fit = da.modelfit(
        coords=[da.t],
        model=model,
        params={"n0": 4, "tau": {"min": 2, "max": 6}},
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected, rtol=1e-3)

    # Params as lmfit.Parameters
    fit = da.modelfit(
        coords=[da.t],
        model=model,
        params=lmfit.create_params(n0=4, tau={"min": 2, "max": 6}),
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected, rtol=1e-3)

    # Test parallel fits
    if not use_dask:
        fit = da.parallel_fit(
            dim="x",
            model=model,
            params={"n0": 4, "tau": {"min": 2, "max": 6}},
            output_result=False,
        )
        np.testing.assert_allclose(fit.modelfit_coefficients, expected, rtol=1e-3)

        fit = da.parallel_fit(
            dim="x",
            model=model,
            params=lmfit.create_params(n0=4, tau={"min": 2, "max": 6}),
            output_result=False,
        )
        np.testing.assert_allclose(fit.modelfit_coefficients, expected, rtol=1e-3)

    if use_dask:
        da = da.compute()

    # Test 0dim output
    fit = da.modelfit(
        coords="t",
        model=lmfit.Model(power),
        reduce_dims="x",
        params={"a": {"value": 0.3, "vary": True}},
    )

    assert "a" in fit.param
    assert fit.modelfit_results.dims == ()


@pytest.mark.parametrize("use_dask", [True, False])
def test_modelfit_params(use_dask: bool):
    def sine(t, a, f, p):
        return a * np.sin(2 * np.pi * (f * t + p))

    t = np.arange(0, 2, 0.02)
    da = xr.DataArray(
        np.stack([sine(t, 1.0, 2, 0), sine(t, 1.0, 2, 0)]),
        coords={"x": [0, 1], "t": t},
    )

    expected = xr.DataArray(
        [[1, 2, 0], [-1, 2, 0.5]],
        coords={"x": [0, 1], "param": ["a", "f", "p"]},
    )

    # Different initial guesses for different values of x
    a_guess = [1.0, -1.0]
    p_guess = [0.0, 0.5]

    if use_dask:
        da = da.chunk({"x": 1})

    # params as DataArray of JSON strings
    params = []
    for a, p, f in zip(a_guess, p_guess, np.full_like(da.x, 2, dtype=float)):
        params.append(lmfit.create_params(a=a, p=p, f=f).dumps())
    params = xr.DataArray(params, coords=[da.x])
    fit = da.modelfit(
        coords=[da.t],
        model=lmfit.Model(sine),
        params=params,
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected)

    # params as mixed dictionary
    fit = da.modelfit(
        coords=[da.t],
        model=lmfit.Model(sine),
        params={
            "a": xr.DataArray(a_guess, coords=[da.x]),
            "p": xr.DataArray(p_guess, coords=[da.x]),
            "f": 2.0,
        },
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected)

    def sine(t, a, f, p):
        return a * np.sin(2 * np.pi * (f * t + p))

    t = np.arange(0, 2, 0.02)
    da = xr.DataArray(
        np.stack([sine(t, 1.0, 2, 0), sine(t, 1.0, 2, 0)]),
        coords={"x": [0, 1], "t": t},
    )

    # Fit a sine with different bounds: positive amplitude should result in a fit with
    # phase 0 and negative amplitude should result in phase 0.5 * 2pi.

    expected = xr.DataArray(
        [[1, 2, 0], [-1, 2, 0.5]],
        coords={"x": [0, 1], "param": ["a", "f", "p"]},
    )

    if use_dask:
        da = da.chunk({"x": 1})

    # params as DataArray of JSON strings
    fit = da.modelfit(
        coords=[da.t],
        model=lmfit.Model(sine),
        params=xr.DataArray(
            [
                lmfit.create_params(**param_dict).dumps()
                for param_dict in (
                    {"f": 2, "p": 0.25, "a": {"value": 1, "min": 0, "max": 2}},
                    {"f": 2, "p": 0.25, "a": {"value": -1, "min": -2, "max": 0}},
                )
            ],
            coords=[da.x],
        ),
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected, atol=1e-8)

    # params as mixed dictionary
    fit = da.modelfit(
        coords=[da.t],
        model=lmfit.Model(sine),
        params={
            "f": {"value": 2},
            "p": 0.25,
            "a": {
                "value": xr.DataArray([1, -1], coords=[da.x]),
                "min": xr.DataArray([0, -2], coords=[da.x]),
                "max": xr.DataArray([2, 0], coords=[da.x]),
            },
        },
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected, atol=1e-8)
