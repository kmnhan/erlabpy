from typing import TYPE_CHECKING

import pytest
import xarray as xr

import erlab

if TYPE_CHECKING:
    import IPython


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("hers")
    erlab.io.set_data_dir(test_data_dir / "hers")
    return test_data_dir / "hers"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (5, "20211211_00005.h5"),
        ("20211211_00005.fits", "20211211_00005.h5"),
        (11, "20211216_00011.h5"),
        ("20211216_00011.fits", "20211216_00011.h5"),
    ],
)
def test_load(expected_dir, args, expected) -> None:
    # Start IPython session to avoid astropy logging issues
    from IPython.testing.globalipapp import start_ipython

    ip_session: IPython.InteractiveShell = start_ipython()

    loaded = erlab.io.load(**args) if isinstance(args, dict) else erlab.io.load(args)

    xr.testing.assert_identical(
        loaded, xr.load_dataarray(expected_dir / expected, engine="h5netcdf")
    )
    # Properly clean up the IPython session
    ip_session.clear_instance()
    del start_ipython.already_called
