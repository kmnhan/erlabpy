import lmfit
import numpy as np
import pytest
import xarray as xr

import erlab.accessors  # noqa: F401


def power(t, a):
    return np.power(t, a)


@pytest.fixture
def fit_expected_darr():
    return xr.DataArray(
        [[3, 3], [5, 4], [np.nan, np.nan]],
        dims=("x", "param"),
        coords={"x": [0, 1, 2], "param": ["n0", "tau"]},
    )


def test_parallel_fit(fit_test_darr, fit_expected_darr) -> None:
    def exp_decay(t, n0, tau=1):
        return n0 * np.exp(-t / tau)

    model = lmfit.Model(exp_decay)

    # Test parallel fits
    fit = fit_test_darr.parallel_fit(
        dim="x",
        model=model,
        params={"n0": 4, "tau": {"min": 2, "max": 6}},
        output_result=False,
        parallel_kw={"n_jobs": 1},
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)

    fit = fit_test_darr.parallel_fit(
        dim="x",
        model=model,
        params=lmfit.create_params(n0=4, tau={"min": 2, "max": 6}),
        output_result=False,
        parallel_kw={"n_jobs": 1},
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)
