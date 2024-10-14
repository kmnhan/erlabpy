import os
import pathlib

import lmfit
import numpy as np
import pooch
import pytest
import xarray as xr

from erlab.io.exampledata import generate_data_angles, generate_gold_edge

DATA_COMMIT_HASH = "4503bd2a789d5182c2aee434b53bf12ae3925974"
"""The commit hash of the commit to retrieve from `kmnhan/erlabpy-data`."""

DATA_KNOWN_HASH = "7426d2654fe9d00ea00d9f11dc346180ebc5e14353d7adbc756b6c2e73713c32"
"""The hash of the `.tar.gz` file."""


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


def _exp_decay(t, n0, tau=1):
    return n0 * np.exp(-t / tau)


@pytest.fixture
def exp_decay_model():
    return lmfit.Model(_exp_decay)


@pytest.fixture
def fit_test_darr():
    t = np.arange(0, 5, 0.5)
    da = xr.DataArray(
        np.stack([_exp_decay(t, 3, 3), _exp_decay(t, 5, 4), np.nan * t], axis=-1),
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
