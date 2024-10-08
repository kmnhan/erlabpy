import pytest
import xarray as xr

import erlab.io


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("esm")
    erlab.io.set_data_dir(test_data_dir / "esm")
    return test_data_dir / "esm"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize("identifier", [25, "Sample0025025.ibw", "Sample0025.pxt"])
def test_load(expected_dir, identifier):
    xr.testing.assert_identical(
        erlab.io.load(identifier),
        xr.load_dataarray(expected_dir / "Sample0025.nc"),
    )
