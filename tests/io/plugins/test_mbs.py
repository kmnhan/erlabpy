import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("mbs")
    erlab.io.set_data_dir(test_data_dir / "mbs")
    return test_data_dir / "mbs"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("FIX00001_00000.txt", "FIX00001_00000.h5"),
        (1, "FIX00001_00000.h5"),
    ],
)
def test_load(expected_dir, args, expected) -> None:
    loaded = erlab.io.load(**args) if isinstance(args, dict) else erlab.io.load(args)

    xr.testing.assert_identical(
        loaded, xr.load_dataarray(expected_dir / expected, engine="h5netcdf")
    )
