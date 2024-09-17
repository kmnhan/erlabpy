import os
import pathlib

import lmfit
import numpy as np
import pooch
import pytest
import xarray as xr

from erlab.io.exampledata import generate_data_angles, generate_gold_edge


def exp_decay(t, n0, tau=1):
    return n0 * np.exp(-t / tau)


@pytest.fixture
def exp_decay_model():
    return lmfit.Model(exp_decay)


@pytest.fixture
def fit_test_darr():
    t = np.arange(0, 5, 0.5)
    da = xr.DataArray(
        np.stack([exp_decay(t, 3, 3), exp_decay(t, 5, 4), np.nan * t], axis=-1),
        dims=("t", "x"),
        coords={"t": t, "x": [0, 1, 2]},
    )
    da[0, 0] = np.nan
    return da


@pytest.fixture
def anglemap():
    return generate_data_angles(shape=(10, 10, 10), assign_attributes=True)


@pytest.fixture
def gold():
    return generate_gold_edge(
        temp=100, Eres=1e-2, nx=15, ny=150, edge_coeffs=(0.04, 1e-5, -3e-4), noise=False
    )


DATA_COMMIT_HASH = "83e1a63b0880f7ddf257cace89d5550ed562a611"
DATA_KNOWN_HASH = "000009aee8d36eae95687996dc1b70a89ce648b968df177c0ae12355afb05f86"


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    path = os.getenv("ERLAB_TEST_DATA_DIR", None)
    if path is None:
        cache_folder = pooch.os_cache("erlabpy")
        pooch.retrieve(
            "https://api.github.com/repos/kmnhan/erlabpy-data/tarball/"
            + DATA_COMMIT_HASH,
            known_hash=DATA_KNOWN_HASH,
            path=cache_folder,
            processor=pooch.Untar(extract_dir=""),
        )
        path = cache_folder / f"kmnhan-erlabpy-data-{DATA_COMMIT_HASH[:7]}"

    return pathlib.Path(path)
