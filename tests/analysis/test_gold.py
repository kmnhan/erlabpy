import matplotlib.pyplot as plt
import numpy as np
import pytest
from erlab.analysis.gold import correct_with_edge, poly, spline
from numpy.testing import assert_allclose


@pytest.mark.parametrize("parallel_kw", [{"n_jobs": 1, "return_as": "list"}])
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
        plot=True,
        parallel_kw=parallel_kw,
    )
    plt.close()

    assert_allclose(
        np.array(list(res.best_values.values())),
        np.array([0.04, 1e-5, -3e-4]),
        atol=1e-2,
    )

    correct_with_edge(gold, res, shift_coords=True, plot=False)
    correct_with_edge(gold, res, shift_coords=False, plot=False)


def test_spline(gold):
    spl = spline(
        gold,
        angle_range=(-15, 15),
        eV_range=(-0.2, 0.2),
        temp=100.0,
        vary_temp=False,
        fast=True,
        lam=None,
        plot=True,
    )
    plt.close()

    assert_allclose(spl(0.0, 0.0), 0.04, rtol=1e-4)

    correct_with_edge(gold, spl, shift_coords=True, plot=False)
    correct_with_edge(gold, spl, shift_coords=False, plot=True)
    plt.close()
