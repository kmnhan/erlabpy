import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from erlab.analysis.gold import correct_with_edge, poly, quick_fit, spline


@pytest.mark.parametrize(
    "parallel_kw",
    [None, {"n_jobs": 1, "return_as": "list"}],
    ids=["parallel_generator", "serial_list"],
)
@pytest.mark.parametrize("fast", [True, False], ids=["fast", "regular"])
def test_poly(gold, parallel_kw: dict, fast: bool) -> None:
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
        np.array(list(res.modelfit_results.item().best_values.values())),
        np.array([0.04, 1e-5, -3e-4]),
        atol=1e-2,
    )

    correct_with_edge(gold, res, shift_coords=True, plot=False)
    correct_with_edge(gold, res, shift_coords=False, plot=False)


def test_spline(gold) -> None:
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

    assert_allclose(spl(0.0), 0.04, atol=1e-4)

    correct_with_edge(gold, spl, shift_coords=True, plot=False)
    correct_with_edge(gold, spl, shift_coords=False, plot=True)
    plt.close()


@pytest.mark.parametrize("bkg_slope", [True, False], ids=["slope", "no_slope"])
@pytest.mark.parametrize("fix_resolution", [False, True], ids=["fix_res", "vary_res"])
@pytest.mark.parametrize("fix_center", [False, True], ids=["fix_center", "vary_center"])
@pytest.mark.parametrize("fix_temp", [True, False], ids=["fix_temp", "vary_temp"])
@pytest.mark.parametrize("resolution", [None, 1e-2], ids=["res_None", "res_1e-2"])
@pytest.mark.parametrize("temp", [None, 100.0], ids=["temp_None", "temp_100"])
@pytest.mark.parametrize("eV_range", [None, (-0.2, 0.2)], ids=["eV_full", "eV_range"])
def test_quick_fit(
    gold, eV_range, temp, resolution, fix_temp, fix_center, fix_resolution, bkg_slope
) -> None:
    ds = quick_fit(
        gold,
        eV_range=eV_range,
        temp=temp,
        resolution=resolution,
        fix_temp=fix_temp,
        fix_center=fix_center,
        fix_resolution=fix_resolution,
        bkg_slope=bkg_slope,
        plot=True,
    )
    plt.close()
    assert ds.modelfit_results.item().success
