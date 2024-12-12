import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("i05")
    erlab.io.set_data_dir(test_data_dir / "i05")
    return test_data_dir / "i05"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize("fname", ["core", "cut", "fs", "kz"])
def test_load(expected_dir, fname) -> None:
    xr.testing.assert_identical(
        erlab.io.load(f"{fname}.nxs"),
        xr.load_dataarray(expected_dir / f"{fname}.h5", engine="h5netcdf"),
    )
