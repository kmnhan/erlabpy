import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("lorea")
    erlab.io.set_data_dir(test_data_dir / "lorea")
    return test_data_dir / "lorea"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"identifier": 3, "krax": True}, "sample-1-00003_0.h5"),
        ({"identifier": 1}, "001_sample.h5"),
        ({"identifier": 2}, "002_sample.h5"),
        ("001_sample.nxs", "001_sample.h5"),
        ("002_sample.nxs", "002_sample.h5"),
        ("sample-1-00003_0.krx", "sample-1-00003_0.h5"),
    ],
)
def test_load(data_dir, expected_dir, args, expected) -> None:
    loaded = erlab.io.load(**args) if isinstance(args, dict) else erlab.io.load(args)

    xr.testing.assert_identical(
        loaded, xr.load_dataarray(expected_dir / expected, engine="h5netcdf")
    )

    if isinstance(args, str) and args.endswith(".krx"):
        assert (
            loaded.shape
            == erlab.io.plugins.mbs.load_krax(
                data_dir / args, without_values=True
            ).shape
        )
