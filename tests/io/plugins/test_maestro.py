import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("maestro")
    erlab.io.set_data_dir(test_data_dir / "maestro")
    return test_data_dir / "maestro"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (1, "20241026_00001.h5"),
        ("20241026_00001.h5", "20241026_00001.h5"),
    ],
)
def test_load(expected_dir, args, expected) -> None:
    loaded = erlab.io.load(**args) if isinstance(args, dict) else erlab.io.load(args)

    xr.testing.assert_identical(
        loaded, xr.load_dataarray(expected_dir / expected, engine="h5netcdf")
    )


def test_summarize(data_dir) -> None:
    erlab.io.summarize()
