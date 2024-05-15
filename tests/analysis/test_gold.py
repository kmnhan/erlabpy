import numpy as np
import pytest
from erlab.analysis.gold import poly, spline
from erlab.io.exampledata import generate_gold_edge
from numpy.testing import assert_allclose


@pytest.fixture()
def gold():
    return generate_gold_edge(
        temp=100, seed=1, nx=15, ny=150, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )


@pytest.mark.parametrize("parallel_kw", [{"n_jobs": 1}, {"n_jobs": -1}])
@pytest.mark.parametrize("fast", [True, False])
def test_poly(gold, parallel_kw: dict, fast: bool):
    res = poly(
        gold,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        temp=100.0,
        fast=fast,
        vary_temp=False,
        degree=2,
        plot=False,
        parallel_kw=parallel_kw,
    )

    assert_allclose(
        np.array(list(res.best_values.values())),
        np.array([0.04, 1e-5, -3e-4]),
        atol=1e-2,
    )


def test_spline(gold):
    spl = spline(
        gold,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        temp=100.0,
        vary_temp=False,
        fast=True,
        lam=None,
        plot=False,
    )

    assert_allclose(spl(0.0, 0.0), 0.04, rtol=1e-4)
