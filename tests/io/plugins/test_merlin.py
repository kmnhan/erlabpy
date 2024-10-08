import pytest
import xarray as xr

import erlab.io


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("merlin")
    erlab.io.set_data_dir(test_data_dir / "merlin")
    return test_data_dir / "merlin"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


def test_load_xps(expected_dir):
    xr.testing.assert_identical(
        erlab.io.load("core.pxt"),
        xr.load_dataarray(expected_dir / "core.nc"),
    )


def test_load_multiple(expected_dir):
    xr.testing.assert_identical(
        erlab.io.load("f_005_S001.pxt"),
        xr.load_dataarray(expected_dir / "5.nc"),
    )
    xr.testing.assert_identical(
        erlab.io.load(5),
        xr.load_dataarray(expected_dir / "5.nc"),
    )


def test_load_live(expected_dir):
    for live in ("lp", "lxy"):
        xr.testing.assert_identical(
            erlab.io.load(f"{live}.ibw"),
            xr.load_dataarray(expected_dir / f"{live}.nc"),
        )


def test_summarize(data_dir):
    erlab.io.summarize()
