import lmfit
import numpy as np
import pytest
import xarray as xr
from erlab.io.exampledata import generate_data_angles, generate_gold_edge


def exp_decay(t, n0, tau=1):
    return n0 * np.exp(-t / tau)


@pytest.fixture()
def exp_decay_model():
    return lmfit.Model(exp_decay)


@pytest.fixture()
def fit_test_darr():
    t = np.arange(0, 5, 0.5)
    da = xr.DataArray(
        np.stack([exp_decay(t, 3, 3), exp_decay(t, 5, 4), np.nan * t], axis=-1),
        dims=("t", "x"),
        coords={"t": t, "x": [0, 1, 2]},
    )
    da[0, 0] = np.nan
    return da


@pytest.fixture()
def anglemap():
    return generate_data_angles(shape=(10, 10, 10), assign_attributes=True)


@pytest.fixture()
def gold():
    return generate_gold_edge(
        temp=100, Eres=1e-2, nx=15, ny=150, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )
