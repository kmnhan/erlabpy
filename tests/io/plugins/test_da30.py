import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("da30")
    erlab.io.set_data_dir(test_data_dir / "da30")
    return test_data_dir / "da30"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("f0001f_001.ibw", "f0001f_001.h5"),
        (1, "f0001f_001.h5"),
        ("f0002.zip", "f0002.h5"),
        (2, "f0002.h5"),
        ("f0003.pxt", "f0003.h5"),
        (3, "f0003.h5"),
    ],
)
def test_load(expected_dir, args, expected) -> None:
    loaded = erlab.io.load(**args) if isinstance(args, dict) else erlab.io.load(args)

    xr.testing.assert_identical(
        loaded, xr.load_dataarray(expected_dir / expected, engine="h5netcdf")
    )


def test_zip_libarchive(data_dir):
    darr_with_libarchive = erlab.io.plugins.da30.load_zip(
        data_dir / "f0002.zip", use_libarchive=True, without_values=False
    )
    darr_without_libarchive = erlab.io.plugins.da30.load_zip(
        data_dir / "f0002.zip", use_libarchive=False, without_values=False
    )

    xr.testing.assert_identical(darr_with_libarchive, darr_without_libarchive)
