import pytest
import xarray as xr

import erlab


@pytest.fixture(scope="module")
def data_dir(test_data_dir):
    erlab.io.set_loader("esm")
    erlab.io.set_data_dir(test_data_dir / "esm")
    return test_data_dir / "esm"


@pytest.fixture(scope="module")
def expected_dir(data_dir):
    return data_dir / "expected"


@pytest.mark.parametrize("identifier", [25, "Sample0025025.ibw", "Sample0025.pxt"])
def test_load(expected_dir, identifier) -> None:
    if isinstance(identifier, int):
        with pytest.warns(
            UserWarning,
            match=r"Multiple files found for scan 25, using .*/esm/Sample0025.pxt",
        ):
            loaded = erlab.io.load(identifier)
    else:
        loaded = erlab.io.load(identifier)
    xr.testing.assert_identical(
        loaded, xr.load_dataarray(expected_dir / "Sample0025.h5")
    )
