import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("erpes")
    erlab.io.set_data_dir(test_data_dir / "erpes")
    return test_data_dir / "erpes"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("arg", "index"),
    [
        (1, 1),  # Single cut
        (2, 2),  # Multi-region cut (DataTree)
        (3, 3),  # DA map (DataTree)
        (4, 4),  # Single motor cut
        (5, 5),  # Single motor DA map
        ("test0002.pxt", 2),
        ("test0004_S00001.pxt", 4),
        ("test0005_S00002.zip", 5),
    ],
)
def test_load(expected_dir, arg, index) -> None:
    loaded = erlab.io.load(arg)

    expected_file = expected_dir / f"{index}.h5"

    if index in {2, 3}:
        assert isinstance(loaded, xr.DataTree)
        xr.testing.assert_identical(
            loaded, xr.open_datatree(expected_file, engine="h5netcdf")
        )
    else:
        assert isinstance(loaded, xr.DataArray)
        xr.testing.assert_identical(
            loaded, xr.load_dataarray(expected_file, engine="h5netcdf")
        )


def test_summarize(data_dir) -> None:
    erlab.io.summarize(cache=False)
